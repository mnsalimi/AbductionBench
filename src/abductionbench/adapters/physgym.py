"""PhysGym: physics-law discovery under varying prior knowledge.

Source: https://github.com/principia-ai/PhysGym

``physgym/samples/full_samples.json`` holds 97 physics problems: a described
setup (``content``), the named ``input_variables`` and ``output_variable``, and
the target relation as both LaTeX (``answer``) and a Python expression
(``equation``).  PhysGym's own harness lets an agent run simulated experiments;
single-turn, the abductive core is: from the described situation, state the law
that relates the output variable to the inputs.

The ``solution`` field (the full derivation) is **withheld** -- it contains the
answer.

**Scoring.** Physics relations have many equivalent forms, so string overlap is
a poor metric.  The adapter tries symbolic equivalence with SymPy first
(``simplify(candidate - reference) == 0`` over the problem's own variable
names), and reports overlap metrics alongside it.  Items whose reference
expression SymPy cannot parse are still scored on overlap, and the counts are
reported.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..core.adapter import SkippedDataset
from ..core.metrics import aggregate_mean_metrics, extract_answer_span, token_f1
from ..core.types import (
    AdapterDocumentation,
    ChatMessage,
    ModelResponse,
    SampleScore,
    SampleSpec,
)
from . import _common as C
from ._base import PooledDatasetAdapter, unparsed_score
from ._interactive import InteractiveMixin, parse_action
from ._mathnorm import equal_expressions

REPO_URL = "https://github.com/principia-ai/PhysGym"


class PhysGymAdapter(InteractiveMixin, PooledDatasetAdapter):
    """State the law relating a physical setup's output variable to its inputs."""

    adapter_version = "1.0"

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. You are given a physical setup "
        "and the quantities it involves. State the law relating the target quantity to the "
        "others, as an equation in the given symbols. A relation that fits the described "
        "behaviour matters more than one that looks like a familiar formula."
    )
    data_delivery_mode = "interactive"
    objective_metrics = True
    selection_cardinality = None
    primary_metric = "symbolic_match"

    def load_items(self) -> list[dict[str, Any]]:
        root = C.ensure_git_repo(
            REPO_URL, self.context.data_dir / "repo", offline=self.context.offline
        )
        candidates = C.find_files(root, ["full_samples.json"]) or C.find_files(
            root, ["*samples*.json"]
        )
        if not candidates:
            raise SkippedDataset("PhysGym samples file not found in the repository")
        payload = C.read_json(candidates[0])
        rows = payload if isinstance(payload, list) else []
        if not rows:
            raise SkippedDataset("PhysGym samples file is empty or not a list")
        self.split_used = (
            f"whole sample set ({len(rows)} problems in {candidates[0].name}); no official split, "
            "and fewer problems than the 300-sample target"
        )
        return rows

    # ------------------------------------------------------------------ #
    # the experiment loop -- PhysGym ships the environment as code
    # ------------------------------------------------------------------ #

    ACTIONS = ("experiment", "answer")
    #: The release's default experiment budget per problem.
    max_turns = 12
    category_limits = {"experiment": 10}

    #: PhysGym's own system message, verbatim from
    #: ``methods/baseline_researcher.py`` (BaselineResearcher.analyze_and_propose).
    AUTHORS_SYSTEM = (
        "You are a top-tier AI Physicist and Experimental Design Researcher. Your "
        "mission is to analyze experimental data, propose hypotheses, and design new "
        "experiments to discover and validate the mathematical relationships between "
        "physical quantities."
    )
    #: The release's per-iteration quotas (experiments/run_baseline.py,
    #: ExperimentConfig): 20 proposals an iteration against a 100-sample budget,
    #: and 2 chances to formally test a hypothesis.
    AUTHORS_EXPERIMENTS_PER_ITERATION = 20
    AUTHORS_SAMPLE_QUOTA = 100
    AUTHORS_TEST_QUOTA = 2

    def _authors_prompt_template(self) -> str:
        """``methods/prompts/baseline_researcher.txt`` from the cloned release.

        The task instruction, the input schema and the output schema are all in
        that file, and it is what BaselineResearcher loads. It is read rather
        than restated: this is an interactive benchmark, so the protocol it
        defines -- ``next_experiments`` / ``test_hypothesis_flag`` /
        ``current_hypothesis_formula`` -- is part of what is being measured.
        """
        cached = getattr(self, "_prompt_cache", None)
        if cached is not None:
            return cached
        root = self.context.data_dir / "repo"
        for candidate in (
            root / "methods" / "prompts" / "baseline_researcher.txt",
            root / "prompts" / "baseline_researcher.txt",
        ):
            if candidate.is_file():
                self._prompt_cache = C.read_text(candidate).strip()
                return self._prompt_cache
        raise SkippedDataset(
            f"PhysGym's own agent prompt (methods/prompts/baseline_researcher.txt) is not "
            f"in the clone at {root}. It defines the JSON protocol this benchmark scores, "
            f"so it cannot be substituted with a prompt written here."
        )

    def _environment(self, sample: SampleSpec):
        """Compile the sample's own ``env_function`` from the release's code.

        The code is PhysGym's, shipped with the dataset, and is compiled the way
        the release compiles it -- with ``math`` and ``numpy`` in scope and
        nothing else.  It is arithmetic over floats, and it is the only way an
        experiment can return the true value rather than a simulated one.
        """
        code = sample.metadata.get("_python_code")
        if not code:
            return None
        namespace: dict[str, Any] = {"np": np, "math": math}
        try:
            renamed = re.sub(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", "def env_function(", code)
            exec(renamed, namespace)  # noqa: S102 - the benchmark's own environment
        except Exception as exc:  # noqa: BLE001 - a broken sample loses its loop, not the run
            self.log.warning("physgym: cannot compile env for %s: %s", sample.sample_id, exc)
            return None
        return namespace.get("env_function")

    def _authors_turn(self, sample: SampleSpec, state: dict[str, Any]) -> str:
        """One user turn, built the way ``analyze_and_propose`` builds it.

        The release serialises a five-key JSON input, appends it to the prompt
        template, and asks for the Output. Nothing else is added.
        """
        remaining = self.AUTHORS_SAMPLE_QUOTA - len(state["experiments"])
        payload = {
            "problem_description": sample.fields.get("observation", ""),
            "controllable_variables": sample.metadata.get("_inputs") or {},
            "observable_variable": {
                sample.metadata.get("output_name", "y"):
                    sample.metadata.get("output_description", "the observed quantity")
            },
            "historical_experiments": state["experiments"],
            "quota": {
                "experiments_quota": max(
                    0, min(remaining, self.AUTHORS_EXPERIMENTS_PER_ITERATION)
                ),
                "test_quota": state["test_quota"],
            },
        }
        return (
            f"{self._authors_prompt_template()}\n\n**Input:**\n```json\n"
            f"{json.dumps(payload, indent=2)}\n```\n\nProvide the **Output:**\n"
        )

    def interactive_start(self, sample: SampleSpec) -> tuple[list[ChatMessage], dict[str, Any]]:
        state = {
            "env": self._environment(sample),
            "experiments": [],
            "test_quota": self.AUTHORS_TEST_QUOTA,
            "hypothesis": "",
            "counts": {},
        }
        return (
            [
                ChatMessage(role="system", content=self.AUTHORS_SYSTEM),
                ChatMessage(role="user", content=self._authors_turn(sample, state)),
            ],
            state,
        )

    def interactive_step(
        self, sample: SampleSpec, state: dict[str, Any], assistant_text: str
    ) -> str | None:
        """Run the proposed batch, then ask again -- the release's iteration.

        PhysGym does not ask for one experiment per turn. It asks for a *batch*
        of ``next_experiments`` together with the running
        ``current_hypothesis_formula`` and a ``test_hypothesis_flag``, runs the
        batch, and iterates until the sample quota or the test quota is spent.
        """
        reply = _parse_authors_output(assistant_text)
        if reply is None:
            state["parse_errors"] = state.get("parse_errors", 0) + 1
            if state["parse_errors"] > 2:
                return None
            return (
                "Your previous response could not be parsed. Provide the **Output:** as a "
                "single JSON object with the keys next_experiments, test_hypothesis_flag "
                "and current_hypothesis_formula."
            )

        hypothesis = reply.get("current_hypothesis_formula")
        if isinstance(hypothesis, str) and hypothesis.strip():
            state["hypothesis"] = hypothesis.strip()

        if reply.get("test_hypothesis_flag"):
            # The release spends a test on the stated formula; when the test
            # quota runs out the episode is over and that formula is the answer.
            state["test_quota"] -= 1
            if state["test_quota"] <= 0:
                return None

        env = state.get("env")
        if env is None:
            # Nothing executable for this item, so no observation can be
            # returned: the stated hypothesis is what gets scored.
            return None

        proposed = reply.get("next_experiments")
        if not isinstance(proposed, list) or not proposed:
            if state["hypothesis"]:
                return None
            return "No experiments were proposed. Provide next_experiments as a list of settings."

        remaining = self.AUTHORS_SAMPLE_QUOTA - len(state["experiments"])
        for setting in proposed[: max(0, min(remaining, self.AUTHORS_EXPERIMENTS_PER_ITERATION))]:
            if not isinstance(setting, dict):
                continue
            try:
                values = {str(k): float(v) for k, v in setting.items()}
                observed = env(**values)
            except Exception as exc:  # noqa: BLE001 - a rejected setting is a result too
                state["experiments"].append({**setting, "error": f"{type(exc).__name__}: {exc}"})
                continue
            output_name = sample.metadata.get("output_name", "y")
            state["experiments"].append({**values, output_name: observed})

        if len(state["experiments"]) >= self.AUTHORS_SAMPLE_QUOTA:
            return None
        return self._authors_turn(sample, state)

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        content = C.normalize_whitespace(item.get("content"))
        equation = C.normalize_whitespace(item.get("equation"))
        answer_latex = C.normalize_whitespace(item.get("answer"))
        inputs = item.get("input_variables") or {}
        outputs = item.get("output_variable") or {}
        if not content or not (equation or answer_latex) or not outputs:
            return None
        output_name = next(iter(outputs))
        variable_lines = [f"- {name}: {description}" for name, description in inputs.items()]
        output_line = f"- {output_name}: {outputs[output_name]}"
        return SampleSpec(
            sample_id=C.stable_id("physgym", item.get("id", index)),
            fields={
                "observation": content,
                "context": (
                    "Known quantities:\n" + "\n".join(variable_lines) +
                    f"\n\nQuantity to be explained:\n{output_line}"
                ),
                "question": (
                    f"What relation determines {output_name} in terms of the known quantities?"
                ),
                "instructions": (
                    "Give the relation as a single Python-style expression for "
                    f"{output_name}, using exactly the variable names listed above "
                    "(e.g. 'v = a * b / c**2'). No prose in the final line."
                ),
            },
            reference={
                "equation": equation,
                "latex": answer_latex,
                "output": output_name,
                "variables": sorted(set(inputs) | set(outputs)),
            },
            task_kind="generation",
            # Deriving a physical law needs room to work, even though the final
            # line is one expression. Measured: at 1,536 tokens half of
            # gemma-4-E4B's answers were cut off mid-derivation, which makes the
            # metric about verbosity rather than physics.
            max_tokens=2560,
            metadata={
                "tag": item.get("tag"),
                "n_inputs": len(inputs),
                "output_name": output_name,
                # The environment itself, for the experiment loop: the
                # release's own code and variable descriptions.
                "_python_code": item.get("python_code"),
                "_inputs": dict(inputs),
            },
        )

    def score(
        self,
        sample: SampleSpec,
        response: ModelResponse,
        *,
        output_contract: dict[str, Any] | None = None,
    ) -> SampleScore:
        # The answer is the release's own field: whatever the episode last put
        # in `current_hypothesis_formula`. Reading free text instead would score
        # a different thing than PhysGym scores.
        state = sample.metadata.get("_episode_state") or {}
        hypothesis = str(state.get("hypothesis") or "").strip()
        answer = hypothesis or extract_answer_span(response.text, output_contract)
        if not answer:
            return unparsed_score(
                ["symbolic_match", "expression_token_f1"], raw=(response.text or "")[:300]
            )
        reference = sample.reference["equation"] or sample.reference["latex"]
        candidate = (
            hypothesis if hypothesis
            else _extract_expression(answer, sample.reference["output"])
        )
        metrics: dict[str, float] = {
            "expression_token_f1": token_f1(candidate or answer, reference),
        }
        verdict = _symbolic_equal(candidate, reference, sample.reference["variables"])
        if verdict is None:
            metrics["symbolic_match"] = 0.0
            metrics["symbolic_undecidable"] = 1.0
        else:
            metrics["symbolic_match"] = float(verdict)
            metrics["symbolic_undecidable"] = 0.0
        return SampleScore(
            metrics=metrics,
            prediction=(candidate or answer)[:400],
            details={"reference_equation": reference[:300]},
        )

    def aggregate(self, scores: Sequence[SampleScore]) -> dict[str, float]:
        metrics = aggregate_mean_metrics([score.metrics for score in scores])
        # symbolic_match counts an undecidable comparison as a miss; report the
        # decidable-only rate too so the metric can be read honestly.
        undecidable = metrics.get("symbolic_undecidable", 0.0)
        if undecidable < 1.0:
            decidable = [
                score.metrics["symbolic_match"]
                for score in scores
                if not score.metrics.get("symbolic_undecidable")
            ]
            if decidable:
                metrics["symbolic_match_decidable"] = sum(decidable) / len(decidable)
        return metrics

    def documentation(self) -> AdapterDocumentation:
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="PhysGym",
            domain="Scientific Discovery: Physics",
            source_url=REPO_URL,
            processing_mode="Generation",
            split_used=self.split_used,
            abductive_subset=(
                "The law-discovery step: from the described physical situation and the named "
                "quantities, state the relation that explains the target quantity. PhysGym's "
                "interactive experiment loop is not reproducible single-turn; the `solution` "
                "derivation is withheld because it contains the answer."
            ),
            sampling_procedure=self.sampling_note()
            + "; only 97 problems exist, so the draw covers all of them",
            metrics_description={
                "self_consistency_<metric>":
                "Every metric also gets a self_consistency_ counterpart: the plurality answer over "
                "modes.repeats samples of the same record, read off those samples rather than bought "
                "again. Available because this dataset's answers are checkable and so can coincide.",
                "symbolic_match": "(PRIMARY, higher is better) 1 if SymPy proves the answer equivalent to the reference "
                "expression (primary; an undecidable comparison counts as 0)",
                "symbolic_match_decidable": "symbolic_match over the items SymPy could compare",
                "symbolic_undecidable": "fraction of items where parsing/simplification failed; "
                "rare now that answers are normalized and parsed with implicit multiplication, so "
                "a prose answer counts as a mismatch rather than as unscorable",
                "expression_token_f1": "token F1 against the reference expression, a lenient "
                "fallback view",
            },
            primary_metric="symbolic_match",
            decisions=[
                "Scored by symbolic equivalence rather than string match: a physical law has many "
                "algebraically equal forms. PhysGym itself uses an LLM equivalence check; SymPy "
                "was chosen here because it is deterministic and free.",
                "Answers are normalized before parsing (LaTeX \\frac/\\sqrt, unicode Greek and "
                "subscripts, implicit multiplication). Without this, models answering in "
                "mathematical notation were scored 'undecidable' even when correct -- measured on "
                "a live run, every item was undecidable while token overlap with the reference "
                "reached 0.95.",
                "Asked for a Python-style expression using the dataset's own variable names, so "
                "the answer is parseable; the reference is the dataset's `equation` field.",
                "Counted undecidable comparisons as misses in the primary metric but reported "
                "them separately, so the number cannot be silently inflated.",
                "max_tokens=2560: at 1,536 half of one model's answers were truncated "
                "mid-derivation. A cut-off equation is unscorable rather than merely shorter, so "
                "runs on verbose models should also set "
                "engine.retry.escalate_truncated_responses=true.",
            ],
            caveats=[
                "Only 97 problems, so this dataset reports far fewer than the 300-sample target.",
                "SymPy cannot always decide equivalence involving special functions or implicit "
                "constants; see symbolic_undecidable.",
                "Textbook physics problems are likely present in pretraining data.",
            ],
            statistics=self.base_statistics(),
        )


def _parse_authors_output(text: str | None) -> dict[str, Any] | None:
    """The JSON object PhysGym asks for, from whatever wrapping it arrives in.

    The release asks for the Output as a fenced JSON block; models supply it
    fenced, bare, or after prose, so the last balanced object carrying any of
    the three expected keys wins.
    """
    if not text:
        return None
    for match in reversed(list(re.finditer(r"\{.*\}", text, flags=re.S))):
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and (
            "next_experiments" in parsed
            or "current_hypothesis_formula" in parsed
            or "test_hypothesis_flag" in parsed
        ):
            return parsed
    return None


def _extract_expression(answer: str, output_name: str) -> str | None:
    """Pull the right-hand side of ``output = ...`` out of a model answer."""
    text = answer.strip().strip("`")
    # Prefer the last assignment to the target variable.
    matches = re.findall(rf"{re.escape(output_name)}\s*=\s*([^\n;]+)", text)
    if matches:
        return matches[-1].strip()
    if "=" in text:
        return text.split("=")[-1].strip()
    return text if text else None


def _symbolic_equal(candidate: str | None, reference: str, variables: Sequence[str]) -> bool | None:
    """Symbolic equality, tolerant of LaTeX/unicode notation.

    Delegates to :mod:`abductionbench.adapters._mathnorm`, which rewrites
    ``\\dfrac``/``\\sqrt``/unicode Greek/subscripts and parses with implicit
    multiplication -- models answer in mathematical notation regardless of the
    prompt's request for Python syntax, and without this a correct answer is
    scored as undecidable.
    """
    if not candidate:
        return None
    return equal_expressions(candidate, reference, variables)
