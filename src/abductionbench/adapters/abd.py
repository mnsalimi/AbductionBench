"""ABD: default-exception abduction in finite first-order worlds.

Source: https://github.com/SerafimBatzoglou/concept-synth
        (``benchmarks/abduction/``, release "abduction-v1.2")

Each instance gives a first-order theory containing a **default rule with an
exception predicate** (``Ab``), a set of *training worlds* -- finite domains with
the extensions of each predicate, some of which are exceptions -- and the gold
hypothesis ``alpha``: a formula about the free variable ``x`` that, added to the
theory, explains the observations at minimum cost.  Instances come in three
scenarios (``ABD_FULL``, ``ABD_PARTIAL``, ``ABD_SKEPTICAL``) and carry a
difficulty label.

The task here is exactly the release's task: state ``alpha``.  Because ``alpha``
is a formula, scoring combines a normalized structural match, token overlap, and
a check that only the allowed predicates were used; the release also ships a
human-readable ``description`` of the gold formula, which is offered as an
accepted alternative surface form.
"""

from __future__ import annotations

import gzip
import re
from typing import Any, Sequence

from ..core.adapter import SkippedDataset
from ..core.metrics import aggregate_mean_metrics, extract_answer_span, normalize_text, token_f1
from ..core.types import AdapterDocumentation, ModelResponse, SampleScore, SampleSpec
from . import _common as C
from ._base import PooledDatasetAdapter, unparsed_score

REPO_URL = "https://github.com/SerafimBatzoglou/concept-synth"


class ABDAdapter(PooledDatasetAdapter):
    """State the minimal-cost explanatory formula for a default-exception theory."""

    adapter_version = "1.0"
    primary_metric = "formula_match"

    def load_items(self) -> list[dict[str, Any]]:
        root = C.ensure_git_repo(
            REPO_URL, self.context.data_dir / "repo", offline=self.context.offline
        )
        archives = C.find_files(root / "benchmarks" / "abduction" / "data", ["*instances*.yaml.gz"])
        if not archives:
            raise SkippedDataset("ABD instance archive not found in the repository")
        import yaml

        with gzip.open(archives[0], "rt", encoding="utf-8", errors="replace") as handle:
            payload = yaml.safe_load(handle)
        rows = payload if isinstance(payload, list) else []
        scenarios = self.context.option("scenarios")
        items: list[dict[str, Any]] = []
        self._per_scenario: dict[str, int] = {}
        for row in rows:
            problem = (row or {}).get("problem") or {}
            gold = (problem.get("gold") or {}).get("alpha")
            scenario = str(problem.get("scenario") or "unknown")
            if not gold:
                continue
            if scenarios and scenario not in scenarios:
                continue
            items.append(row)
            self._per_scenario[scenario] = self._per_scenario.get(scenario, 0) + 1
        if not items:
            raise SkippedDataset("ABD instances carry no gold alpha formulas")
        self.split_used = (
            f"{archives[0].name} in full ({len(items)} instances: "
            + ", ".join(f"{name} {count}" for name, count in sorted(self._per_scenario.items()))
            + "); the release ships a single evaluation set"
        )
        return items

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        problem = item.get("problem") or {}
        description = item.get("problemDescription") or {}
        gold = problem.get("gold") or {}
        alpha = C.normalize_whitespace(gold.get("alpha"))
        axioms = [C.normalize_whitespace(axiom) for axiom in C.as_list(problem.get("axioms"))]
        allowed = [str(name) for name in C.as_list(problem.get("allowedAlphaPreds"))]
        worlds = C.as_list(problem.get("trainWorlds"))
        if not alpha or not axioms or not worlds:
            return None

        max_worlds = int(self.context.option("max_worlds", 4))
        world_blocks = []
        for position, world in enumerate(worlds[:max_worlds], start=1):
            domain = ", ".join(str(element) for element in C.as_list(world.get("domain")))
            predicates = world.get("predicates") or {}
            predicate_lines = []
            for name, extension in predicates.items():
                if isinstance(extension, dict):
                    true_set = C.as_list(extension.get("true"))
                    rendered = ", ".join(str(entry) for entry in true_set) or "(empty)"
                else:
                    rendered = str(extension)
                predicate_lines.append(f"    {name} holds of: {rendered}")
            mode = world.get("observationMode")
            world_blocks.append(
                f"World {position} (observation mode: {mode})\n"
                f"  Domain: {domain}\n" + "\n".join(predicate_lines)
            )

        return SampleSpec(
            sample_id=C.stable_id("abd", problem.get("instanceId") or index),
            fields={
                "context": (
                    "Theory (a default rule with exception predicate Ab):\n"
                    + "\n".join(f"- {axiom}" for axiom in axioms)
                    + f"\n\nThe hypothesis may only use these predicates: {', '.join(allowed)}."
                    + "\n\nObserved worlds:\n"
                    + "\n\n".join(world_blocks)
                ),
                "observation": (
                    "In each world the theory's conclusion holds except for a few objects marked "
                    "abnormal (Ab). One hypothesis about x explains all the worlds at minimum cost."
                ),
                "instructions": (
                    "State that hypothesis as a first-order formula about the free variable x, in "
                    "the same s-expression syntax as the theory above (for example "
                    "'(exists y (and (S x y) (P y)))'). Use only the allowed predicates."
                ),
            },
            reference={
                "gold": alpha,
                "description": C.normalize_whitespace(gold.get("description")),
                "allowed": allowed,
            },
            task_kind="knowledge_completion",
            # Formal hypothesis synthesis over several worlds: room to reason.
            max_tokens=self.clamp_max_tokens(
                768 + 128 * int(description.get("alphaTier") or 1), low=768, high=1536
            ),
            metadata={
                "instance_id": problem.get("instanceId"),
                "scenario": problem.get("scenario"),
                "difficulty": description.get("difficulty"),
                "alpha_tier": description.get("alphaTier"),
                "n_worlds": len(worlds),
            },
        )

    def score(
        self,
        sample: SampleSpec,
        response: ModelResponse,
        *,
        output_contract: dict[str, Any] | None = None,
    ) -> SampleScore:
        answer = extract_answer_span(response.text, output_contract)
        if not answer:
            return unparsed_score(
                ["formula_match", "formula_token_f1", "predicate_compliance"],
                raw=response.text[:200],
            )
        gold = sample.reference["gold"]
        candidate = _formula(answer)
        exact = float(_canonical(candidate) == _canonical(gold))
        described = sample.reference.get("description") or ""
        metrics = {
            "formula_match": exact,
            "formula_token_f1": max(
                token_f1(candidate, gold),
                token_f1(candidate, described) if described else 0.0,
            ),
            "predicate_compliance": _predicate_compliance(candidate, sample.reference["allowed"]),
        }
        scenario = sample.metadata.get("scenario")
        if scenario:
            metrics[f"formula_match_{scenario}"] = exact
        difficulty = sample.metadata.get("difficulty")
        if difficulty:
            metrics[f"formula_match_{difficulty}"] = exact
        return SampleScore(
            metrics=metrics, prediction=candidate[:400], details={"gold": gold[:300]}
        )

    def aggregate(self, scores: Sequence[SampleScore]) -> dict[str, float]:
        return aggregate_mean_metrics([score.metrics for score in scores])

    def judge_request(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore
    ) -> dict[str, Any] | None:
        if not response.text:
            return None
        return {
            "candidate": score.prediction or response.text[:500],
            "gold": sample.reference.get("description") or sample.reference["gold"],
            "criteria": (
                "Correct if the candidate formula is logically equivalent to the reference, even "
                "if written differently (variable renaming, reordered conjuncts)."
            ),
        }

    def apply_judge(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore, verdict: Any
    ) -> SampleScore:
        metrics = dict(score.metrics)
        metrics["formula_judged"] = 1.0 if getattr(verdict, "positive", False) else 0.0
        return SampleScore(
            metrics=metrics,
            prediction=score.prediction,
            parse_ok=score.parse_ok,
            details={**score.details, "judge_label": getattr(verdict, "label", None)},
        )

    def documentation(self) -> AdapterDocumentation:
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="ABD",
            domain="Formal Reasoning: Default-Exception Logic",
            source_url=REPO_URL,
            processing_mode="Generation",
            split_used=self.split_used,
            abductive_subset=(
                "The abduction benchmark only (benchmarks/abduction/). The repository's induction "
                "benchmark is a different inference type, and its eval/ and predictions/ "
                "directories hold model outputs, which are never read."
            ),
            sampling_procedure=self.sampling_note(),
            metrics_description={
                "formula_match": "1 if the answer's formula is structurally identical to the gold "
                "alpha after canonicalization (whitespace, parentheses, case) -- primary, and "
                "strict: logically equivalent rewrites count as misses",
                "formula_token_f1": "token F1 against the gold formula or its human-readable "
                "description, whichever matches better -- partial credit",
                "predicate_compliance": "1 if the answer uses only the allowed predicates, i.e. "
                "whether the model respected the hypothesis space",
                "formula_match_<scenario>": "per scenario (ABD_FULL / PARTIAL / SKEPTICAL)",
                "formula_match_<difficulty>": "per difficulty label",
                "formula_judged": "LLM-judge verdict on logical equivalence (only when "
                "engine.judge.enabled)",
            },
            primary_metric="formula_match",
            decisions=[
                "Read the gold alpha from the instance archive (abd_instances_v1.yaml.gz); the "
                "holdout file alone carries worlds and costs but not the gold formula.",
                "Showed at most 4 training worlds per prompt (options.max_worlds); the release's "
                "own prompts run to ~11k characters and the full world list would crowd the "
                "input budget.",
                "Kept predicate_compliance as a separate metric because respecting the allowed "
                "hypothesis space is a distinct competence from finding the right formula.",
                "max_tokens scales with the instance's alphaTier (formula complexity).",
            ],
            caveats=[
                "formula_match is exact structural equality, so a logically equivalent formula "
                "written differently scores 0; formula_token_f1 and the judge stage exist for "
                "exactly that reason, and the primary metric should be read as a lower bound.",
                "Cost-optimality (the release's own criterion) is not verified -- that needs their "
                "model-counting harness -- so a cheaper valid explanation would also count as a "
                "miss.",
            ],
            statistics={**self.base_statistics(), "instances_per_scenario": self._per_scenario},
        )


def _formula(text: str) -> str:
    """Extract the first s-expression from a model answer, else the text."""
    body = text.strip().strip("`")
    start = body.find("(")
    if start < 0:
        return body
    depth = 0
    for position in range(start, len(body)):
        if body[position] == "(":
            depth += 1
        elif body[position] == ")":
            depth -= 1
            if depth == 0:
                return body[start : position + 1]
    return body[start:]


def _canonical(formula: str) -> str:
    """Whitespace/parenthesis/case-insensitive canonical form."""
    compact = re.sub(r"\s+", " ", (formula or "").strip().lower())
    return re.sub(r"\s*([()])\s*", r"\1", compact)


def _predicate_compliance(formula: str, allowed: Sequence[str]) -> float:
    """1.0 when the formula's predicate symbols are within the allowed set."""
    if not formula:
        return 0.0
    tokens = set(re.findall(r"\(([A-Za-z_][A-Za-z0-9_]*)", formula))
    logical = {
        "and", "or", "not", "implies", "iff", "exists", "forall", "=", "if", "then",
    }
    used = {token for token in tokens if token.lower() not in logical}
    allowed_set = {name for name in allowed} | {"="}
    if not used:
        return 0.0
    return float(used <= allowed_set)
