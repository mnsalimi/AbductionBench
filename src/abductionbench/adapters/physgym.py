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

import re
from collections.abc import Sequence
from typing import Any

from ..core.adapter import SkippedDataset
from ..core.metrics import aggregate_mean_metrics, extract_answer_span, token_f1
from ..core.types import AdapterDocumentation, ModelResponse, SampleScore, SampleSpec
from . import _common as C
from ._base import PooledDatasetAdapter, unparsed_score
from ._mathnorm import equal_expressions

REPO_URL = "https://github.com/principia-ai/PhysGym"


class PhysGymAdapter(PooledDatasetAdapter):
    """State the law relating a physical setup's output variable to its inputs."""

    adapter_version = "1.0"
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
            metadata={"tag": item.get("tag"), "n_inputs": len(inputs)},
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
                ["symbolic_match", "expression_token_f1"], raw=response.text[:300]
            )
        reference = sample.reference["equation"] or sample.reference["latex"]
        candidate = _extract_expression(answer, sample.reference["output"])
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
                "symbolic_match": "1 if SymPy proves the answer equivalent to the reference "
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
