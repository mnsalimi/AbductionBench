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

The task here is exactly the release's task: state ``alpha``.

**How a formula is scored.**  A correct hypothesis can be written many ways --
conjuncts reordered, bound variables renamed, a double negation, a
contrapositive -- so a string comparison measures the wrong thing.  Logical
equivalence of first-order formulas is undecidable in general, but this release
hands us the thing that makes it decidable here: every instance ships the
finite worlds themselves, domain and full predicate extensions.  So the check
is a **model checker** (:mod:`._folmodel`): evaluate the candidate and the gold
at every individual of every world shown and see whether they ever disagree.
That is a proof, not an estimate, and it is right where a judge would only be
plausible.  The LLM judge is kept for the residue the checker cannot decide --
a malformed answer, an unknown predicate -- and for nothing else.
"""

from __future__ import annotations

import gzip
import re
from collections.abc import Sequence
from typing import Any

from ..core.adapter import SkippedDataset
from ..core.metrics import aggregate_mean_metrics, extract_answer_span
from ..core.types import AdapterDocumentation, ModelResponse, SampleScore, SampleSpec
from . import _common as C
from ._base import PooledDatasetAdapter, apply_judged_metric, unparsed_score
from ._folmodel import extensionally_equal

REPO_URL = "https://github.com/SerafimBatzoglou/concept-synth"


class ABDAdapter(PooledDatasetAdapter):
    """State the minimal-cost explanatory formula for a default-exception theory."""

    adapter_version = "1.0"

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. Here the evidence is a "
        "default-exception theory and an observation it does not yet explain. A good answer "
        "is the minimal-cost set of literals that, added to the theory, derives the "
        "observation without contradicting a stated exception. Respect the theory's own "
        "predicate vocabulary: an explanation outside it does not count."
    )
    data_delivery_mode = "static"

    answer_format = "one formula"
    answer_constraints = (
        "output exactly one formula",
        "use only the symbols given",
        "output only the formula",
        "do not use introductory phrases or commentary",
    )
    #: Verifiable, and checked as such: the answer is a formula over a closed
    #: predicate vocabulary, and the worlds shipped with each instance decide
    #: equivalence outright. The judge sees only what the model checker cannot
    #: parse, so this dataset keeps self-consistency over its repeats.
    objective_metrics = True
    selection_cardinality = None
    primary_metric = "formula_equivalent"

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

        # All of them by default: the full world list is ~3.5k characters, well
        # inside the input budget, and it is also what the answer is scored
        # against -- the model is never asked to match a formula on evidence it
        # was not shown. options.max_worlds truncates for a deliberate
        # partial-evidence condition.
        max_worlds = self.context.option("max_worlds", None)
        shown = worlds[: int(max_worlds)] if max_worlds else worlds
        world_blocks = []
        for position, world in enumerate(shown, start=1):
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
                # Exactly the worlds the prompt showed, stripped to what the
                # model checker needs: the domain and the true extensions. The
                # complementary `false` lists the release also ships are
                # redundant under a closed domain and would bloat every record.
                "worlds": [_compact_world(world) for world in shown],
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
                "n_worlds": len(shown),
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
                ["formula_equivalent", "formula_match", "predicate_compliance"],
                raw=response.text[:200],
            )
        gold = sample.reference["gold"]
        candidate = _formula(answer)
        verdict = extensionally_equal(candidate, gold, sample.reference.get("worlds") or [])
        # None means the checker does not apply -- unparseable, or naming a
        # predicate the worlds do not define. Only those go to the judge.
        equivalent = 0.0 if verdict is None else float(verdict)
        metrics = {
            "formula_equivalent": equivalent,
            "formula_match": float(_canonical(candidate) == _canonical(gold)),
            "predicate_compliance": _predicate_compliance(candidate, sample.reference["allowed"]),
            "equivalence_undecidable": 1.0 if verdict is None else 0.0,
        }
        for key in ("scenario", "difficulty"):
            value = sample.metadata.get(key)
            if value:
                metrics[f"formula_equivalent_{value}"] = equivalent
        return SampleScore(
            metrics=metrics,
            prediction=candidate[:400],
            details={"gold": gold[:300], "decided": verdict is not None},
        )

    def aggregate(self, scores: Sequence[SampleScore]) -> dict[str, float]:
        return aggregate_mean_metrics([score.metrics for score in scores])

    def judge_request(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore
    ) -> dict[str, Any] | None:
        """Only what the model checker could not decide.

        A judge asked to confirm a proof can only weaken it, so a decided
        verdict is never second-guessed and never paid for.
        """
        if not response.text or not score.metrics.get("equivalence_undecidable"):
            return None
        return {
            "candidate": score.prediction or response.text[:500],
            "gold": sample.reference.get("description") or sample.reference["gold"],
            "criteria": (
                "Correct if the candidate formula is logically equivalent to the reference, even "
                "if written differently (variable renaming, reordered conjuncts). The candidate "
                "could not be parsed as a formula, so judge what it evidently asserts; an answer "
                "that states no single condition on x is not equivalent to anything."
            ),
        }

    def apply_judge(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore, verdict: Any
    ) -> SampleScore:
        return apply_judged_metric(score, verdict, "formula_equivalent")

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
                "self_consistency_<metric>":
                "Every metric also gets a self_consistency_ counterpart: the plurality answer over "
                "modes.repeats samples of the same record, read off those samples rather than bought "
                "again. Available because this dataset's answers are checkable and so can coincide. "
                "It votes on the formula as written, so two equivalent phrasings of the right "
                "answer do not pool their votes -- read it as a floor.",
                "formula_equivalent": "(PRIMARY, higher is better, 0-1) 1 when the candidate and "
                "the gold pick out the same individuals in every world shown, checked by "
                "evaluating both formulas over those finite structures. Reordered conjuncts, "
                "renamed bound variables and double negations all count as correct; a formula "
                "that separates different individuals does not.",
                "formula_match": "1 if the answer is structurally identical to the gold after "
                "canonicalization -- a diagnostic, showing how often the model reproduces the "
                "gold verbatim rather than an equivalent of it",
                "predicate_compliance": "1 if the answer uses only the allowed predicates, i.e. "
                "whether the model respected the hypothesis space",
                "equivalence_undecidable": "(lower is better, 0-1) fraction the model checker "
                "could not decide -- an unparseable answer or an unknown predicate. These, and "
                "only these, are sent to the LLM judge, whose verdict then fills "
                "formula_equivalent for them.",
                "formula_equivalent_<scenario>": "per scenario (ABD_FULL / PARTIAL / SKEPTICAL)",
                "formula_equivalent_<difficulty>": "per difficulty label",
            },
            primary_metric="formula_equivalent",
            decisions=[
                "Read the gold alpha from the instance archive (abd_instances_v1.yaml.gz); the "
                "holdout file alone carries worlds and costs but not the gold formula.",
                "Showed every training world by default. The full list is ~3.5k characters "
                "(6.1k at worst), comfortably inside the input budget, and it is also exactly "
                "what the answer is scored against -- the model is never asked to match a "
                "formula on evidence it was not shown. options.max_worlds truncates for a "
                "deliberate partial-evidence condition.",
                "Scored logical equivalence by evaluating both formulas over the worlds "
                "themselves rather than by comparing strings or asking a judge. The worlds are "
                "finite and fully specified, so this is a decision procedure; all 600 of the "
                "release's own golds are decided by it. An LLM judge grades only the residue it "
                "cannot parse.",
                "Repaired two golds that the release serialises as 'Var(y)' instead of 'y' -- a "
                "bug in its writer, not a different syntax -- so those instances are scored "
                "rather than quietly handed to the judge.",
                "Kept predicate_compliance as a separate metric because respecting the allowed "
                "hypothesis space is a distinct competence from finding the right formula.",
                "max_tokens scales with the instance's alphaTier (formula complexity).",
            ],
            caveats=[
                "Equivalence is decided over the worlds shown, not in general. Two formulas "
                "that differ only on individuals no world contains are counted as the same "
                "hypothesis -- which is the right notion for this task, since the gold is "
                "defined as what explains those worlds, but it is not full logical equivalence.",
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


def _compact_world(world: dict[str, Any]) -> dict[str, Any]:
    """A world reduced to what the model checker reads.

    The release states each predicate twice -- the individuals it holds of and
    the individuals it does not. Under a closed domain the second list is
    implied by the first, and it is much the larger of the two, so only the
    true extension is carried into the sample record.
    """
    predicates: dict[str, list[str]] = {}
    for name, extension in (world.get("predicates") or {}).items():
        entries = extension.get("true") if isinstance(extension, dict) else extension
        predicates[str(name)] = [str(entry) for entry in (entries or [])]
    return {
        "domain": [str(element) for element in C.as_list(world.get("domain"))],
        "predicates": predicates,
    }
