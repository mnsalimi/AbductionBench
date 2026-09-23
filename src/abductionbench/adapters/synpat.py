"""SynPAT: synthetic physical axiom systems with noisy data.

Source: https://github.com/jlenchner/theorizer
Data:   https://huggingface.co/datasets/Karan0901/synpat-dataset (``dataset.zip``)

Each ``System_*`` directory contains

* ``system.txt`` -- the true axiom system: variables, constants, derivatives,
  the equations (each an expression equal to zero) and their units;
* ``replacement_k.txt`` -- the same system with equation *k* swapped for an
  incorrect one;
* ``system_<noise>.dat`` -- tabular data generated from the *true* system at
  noise levels 0.001 / 0.01 / 0.05 / 0.1;
* ``consequence.txt`` and ``consequence_<noise>.dat`` -- a derived consequence.

**The abductive item** (constructed, and documented as such): the model is shown
a corrupted system -- one equation replaced -- together with the system's
variables, constants and derivatives *and their units of measure*, and is told
which equation is the wrong one and what units the correct one must carry.  It
must state the corrected equation.  The gold answer is that equation from
``system.txt``.

**No data rows are shown.**  The ``.dat`` files are generated *from* the true
system, so a sample of them is a set of worked examples: a model can fit an
equation to the numbers and never form a hypothesis about the system at all.
That is induction from instances, which is the inference type this suite exists
to separate abduction from.  What is left is the abductive question -- which
axiom, added to this theory, would make it a coherent physical system -- posed
from the theory alone.

**Scoring.**  Equations here are expressions set to zero, so what matters is
where the expression *vanishes*, not what it looks like.  SymPy decides that:
two expressions match when their numerators have the same irreducible factors,
which credits a sign flip, a scalar multiple, and a rearrangement by a
non-constant factor (``Fg/Fc - dxdt/c`` for ``c*Fg - dxdt*Fc``) alike.  That is
a proof rather than an estimate, and all 540 of the release's own equations are
decided by it.  An LLM judge grades only the residue SymPy cannot parse.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from ..core.adapter import SkippedDataset
from ..core.metrics import aggregate_mean_metrics, extract_answer_span
from ..core.types import AdapterDocumentation, ModelResponse, SampleScore, SampleSpec
from . import _common as C
from ._base import PooledDatasetAdapter, apply_judged_metric, unparsed_score
from ._mathnorm import equal_as_zero_set as _shared_equal_as_zero_set
from ._mathnorm import equal_up_to_scale as _shared_equal_up_to_scale
from ._mathnorm import same_monomials as _shared_same_monomials

HF_REPO = "Karan0901/synpat-dataset"
GITHUB_URL = "https://github.com/jlenchner/theorizer"


def _parse_system(text: str) -> dict[str, Any]:
    """Parse a SynPAT ``system.txt`` / ``replacement_k.txt`` file.

    The units are read as well as the equations. They are not decoration: with
    the generated data withheld, the dimensions of the quantities and of the
    equation to be recovered are what keep the task determinate rather than a
    guess, and they are part of the axiom system rather than a sample from it.
    """
    out: dict[str, Any] = {
        "equations": [],
        "variables": [],
        "constants": [],
        "derivatives": [],
        "units_variables": [],
        "units_constants": [],
        "units_derivatives": [],
        "units_equations": [],
    }
    section = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if lowered.startswith("variables:"):
            out["variables"] = _parse_list(stripped)
            section = None
        elif lowered.startswith("constants:"):
            out["constants"] = _parse_list(stripped)
            section = None
        elif lowered.startswith("derivatives:"):
            out["derivatives"] = _parse_list(stripped)
            section = None
        elif lowered.startswith("equations:"):
            section = "equations"
        elif lowered.startswith("units of measure of variables"):
            out["units_variables"] = _parse_list(stripped)
            section = None
        elif lowered.startswith("units of measure of constants"):
            out["units_constants"] = _parse_list(stripped)
            section = None
        elif lowered.startswith("units of measure of derivatives"):
            out["units_derivatives"] = _parse_list(stripped)
            section = None
        elif lowered.startswith("units of measure of equations"):
            # This one is a block, one unit per following line.
            section = "units_equations"
        elif lowered.startswith("units of measure"):
            section = None
        elif section:
            out[section].append(stripped)
    return out


def _with_units(names: Sequence[str], units: Sequence[str]) -> str:
    """``Fc [s^(-2)*kg*m], Fg [s^(-2)*kg*m], ...`` -- names with their units."""
    if not names:
        return "(none)"
    if len(units) != len(names):
        return ", ".join(names)
    return ", ".join(f"{name} [{unit}]" for name, unit in zip(names, units, strict=True))


def _parse_list(line: str) -> list[str]:
    match = re.search(r"\[(.*)\]", line)
    if not match:
        return []
    return [item.strip().strip("'\"") for item in match.group(1).split(",") if item.strip()]


class SynPATAdapter(PooledDatasetAdapter):
    """Recover the equation of an axiom system that the data contradicts."""

    adapter_version = "2.0"

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. You are given an axiom system of "
        "physical equations, one of which is known to be wrong, together with the quantities "
        "it is written in and their units. State the corrected form of that equation: the law "
        "that would make the system a coherent physical theory. Dimensional consistency is a "
        "hard constraint -- an equation whose terms do not share the stated units cannot be it."
    )
    data_delivery_mode = "static"

    answer_format = "one equation in the given symbols"
    task_requirements = (
        "use only the symbols given",
    )
    #: Verifiable, and checked as such: an equation set to zero has a zero set,
    #: and SymPy decides whether two expressions share it. The judge sees only
    #: what SymPy cannot parse, so this dataset keeps self-consistency over its
    #: repeats -- read as a floor, since the vote is over the formula as written
    #: and two equivalent phrasings of the right answer do not pool their votes.
    objective_metrics = True
    selection_cardinality = None
    primary_metric = "equation_equivalent"

    def load_items(self) -> list[dict[str, Any]]:
        root = C.ensure_hf_snapshot(
            HF_REPO,
            self.context.data_dir / "hf",
            offline=self.context.offline,
            allow_patterns=["dataset.zip", "*.md"],
        )
        archives = C.find_files(root, ["dataset.zip"])
        if not archives:
            raise SkippedDataset("SynPAT dataset.zip not found in the release")
        extracted = C.extract_archive(archives[0], self.context.data_dir / "dataset")
        system_files = [
            path for path in C.find_files(extracted, ["system.txt"]) if path.parent.is_dir()
        ]
        if not system_files:
            raise SkippedDataset("no system.txt files found in the SynPAT archive")
        items: list[dict[str, Any]] = []
        for path in system_files:
            directory = path.parent
            replacements = sorted(directory.glob("replacement_*.txt"))
            if not replacements:
                continue
            for replacement in replacements:
                items.append(
                    {
                        "directory": str(directory),
                        "config": directory.parent.name,
                        "system_name": directory.name,
                        "replacement": replacement.name,
                    }
                )
        if not items:
            raise SkippedDataset("SynPAT systems carry no replacement files")
        self.split_used = (
            f"all systems in dataset.zip: {len(system_files)} axiom systems x their replacement "
            f"variants = {len(items)} items; the release ships no split"
        )
        return items

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        from pathlib import Path

        directory = Path(item["directory"])
        true_system = _parse_system(C.read_text(directory / "system.txt"))
        corrupted = _parse_system(C.read_text(directory / item["replacement"]))
        if not true_system["equations"] or not corrupted["equations"]:
            return None
        if len(true_system["equations"]) != len(corrupted["equations"]):
            return None
        differing = [
            position
            for position, (a, b) in enumerate(
                zip(true_system["equations"], corrupted["equations"], strict=True)
            )
            if a.strip() != b.strip()
        ]
        if len(differing) != 1:
            return None  # ambiguous corruption; skip rather than guess
        position = differing[0]
        gold = true_system["equations"][position]

        equations_text = "\n".join(
            f"{i + 1}. {equation}" + ("   <- this one is wrong" if i == position else "")
            for i, equation in enumerate(corrupted["equations"])
        )
        # The dimension the corrected equation must carry, from the true
        # system. It is a property of the law being recovered, not a sample
        # drawn from it, and without it -- and with the data withheld -- the
        # task would be underdetermined rather than hard.
        target_units = ""
        units = true_system.get("units_equations") or []
        if position < len(units):
            target_units = units[position]
        return SampleSpec(
            sample_id=C.stable_id("synpat", item["config"], item["system_name"], item["replacement"]),
            fields={
                "context": (
                    "Quantities, with their units of measure:\n"
                    f"  Variables: {_with_units(true_system['variables'], true_system['units_variables'])}\n"
                    f"  Constants: {_with_units(true_system['constants'], true_system['units_constants'])}\n"
                    f"  Derivatives: {_with_units(true_system['derivatives'], true_system['units_derivatives'])}\n\n"
                    "Proposed axiom system (each equation is an expression equal to 0):\n"
                    + equations_text
                ),
                "observation": (
                    f"Equation {position + 1} of the proposed system is not a law of this system: "
                    "it is inconsistent with the rest of the theory."
                    + (
                        f" The correct equation in its place has units of {target_units}."
                        if target_units
                        else ""
                    )
                ),
                "instructions": (
                    f"Give the corrected equation {position + 1}: a single expression that equals "
                    "zero, using only the listed variables, constants and derivatives, written in "
                    "Python-style notation (** for powers). Every term of it must carry the same "
                    "units."
                ),
            },
            reference={
                "gold": gold,
                "symbols": true_system["variables"]
                + true_system["constants"]
                + true_system["derivatives"],
                "position": position,
                "units": target_units,
            },
            task_kind="knowledge_completion",
            # Dimensional analysis over ten-odd quantities needs working room.
            max_tokens=1536,
            metadata={
                "config": item["config"],
                "system": item["system_name"],
                "replacement": item["replacement"],
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
                ["equation_equivalent", "structure_match"], raw=response.text[:200]
            )
        gold = sample.reference["gold"]
        candidate = _clean_expression(answer)
        # Seconds one comparison may take before it counts as undecidable.
        # SymPy does not always terminate on a model's rearrangement of a
        # six-variable law, and one that did not took a whole run down with it
        # (see adapters/_symbolic.py). `None` means the module default.
        budget = self.context.option("symbolic_timeout_s")
        verdict = _equal_as_zero_set(
            candidate, gold, sample.reference["symbols"], timeout_s=budget
        )
        equivalent = 0.0 if verdict is None else float(verdict)
        # Reported beside the verdict, not folded into it: a reference such as
        # `4*c*dx1dt + d2x1dt2*d1` carries a coefficient that nothing in the
        # theory fixes, so an answer with the right terms and the wrong number
        # is wrong -- and this is how often that is what went wrong.
        structure = _shared_same_monomials(
            candidate, gold, sample.reference["symbols"], timeout_s=budget
        )
        metrics = {
            "equation_equivalent": equivalent,
            "structure_match": 0.0 if structure is None else float(structure),
            "symbolic_undecidable": 1.0 if verdict is None else 0.0,
        }
        config = sample.metadata.get("config")
        if config:
            metrics[f"equation_equivalent_{config}"] = equivalent
        return SampleScore(
            metrics=metrics,
            prediction=candidate or answer,
            details={"gold": gold[:200], "decided": verdict is not None},
        )

    def judge_request(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore
    ) -> dict[str, Any] | None:
        """Only what SymPy could not parse.

        A judge asked to second-guess an algebraic proof can only make it
        worse, so a decided verdict is never sent and never paid for.
        """
        if not response.text or not score.metrics.get("symbolic_undecidable"):
            return None
        return {
            "candidate": score.prediction or response.text,
            "gold": sample.reference["gold"],
            "observation": (
                "Both are expressions that the system sets equal to zero, in the symbols: "
                + ", ".join(sample.reference["symbols"])
            ),
            "criteria": (
                "Judge mathematical equivalence, not wording. Because both sides are set "
                "equal to zero, the candidate is correct if it vanishes exactly where the "
                "reference does: multiplying through by a non-zero factor, flipping every "
                "sign, or moving terms across the equals sign all give the same equation. "
                "A different relationship between the quantities, or one that merely shares "
                "symbols with the reference, is not equivalent. The candidate reached you "
                "only because it could not be parsed as an expression, so say no unless it "
                "clearly states one."
            ),
        }

    def apply_judge(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore, verdict: Any
    ) -> SampleScore:
        return apply_judged_metric(score, verdict, "equation_equivalent")

    def aggregate(self, scores: Sequence[SampleScore]) -> dict[str, float]:
        metrics = aggregate_mean_metrics([score.metrics for score in scores])
        decidable = [
            score.metrics["equation_equivalent"]
            for score in scores
            if "equation_equivalent" in score.metrics
            and not score.metrics.get("symbolic_undecidable")
        ]
        if decidable:
            metrics["equation_equivalent_decidable"] = sum(decidable) / len(decidable)
        return metrics

    def documentation(self) -> AdapterDocumentation:
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="SynPAT",
            domain="Scientific Discovery: Physics",
            source_url=GITHUB_URL,
            processing_mode="Generation",
            split_used=self.split_used,
            abductive_subset=(
                "Constructed from the release's own files: a system whose equation k has been "
                "replaced (replacement_k.txt), with the true equation k as the reference. The "
                "generated data files are NOT used -- they are samples drawn from the true "
                "system, and fitting an equation to them is induction from instances rather "
                "than abduction from a theory. What the model sees is the corrupted theory, "
                "the quantities with their units, and the units the corrected equation must "
                "carry. Items where the corruption does not isolate to exactly one equation "
                "are skipped."
            ),
            sampling_procedure=self.sampling_note(),
            metrics_description={
                "self_consistency_<metric>":
                "Every metric also gets a self_consistency_ counterpart: the plurality answer over "
                "modes.repeats samples of the same record, read off those samples rather than bought "
                "again. Available because this dataset's answers are checkable and so can coincide. "
                "It votes on the expression as written, so two equivalent phrasings of the right "
                "equation do not pool their votes -- read it as a floor.",
                "equation_equivalent": "(PRIMARY, higher is better, 0-1) 1 when SymPy proves the "
                "answer vanishes exactly where the gold expression does: the numerators share "
                "their irreducible factors. A sign flip, a scalar multiple and a rearrangement "
                "by a non-constant factor (Fg/Fc - dxdt/c for c*Fg - dxdt*Fc) all count as "
                "correct. An answer SymPy cannot parse is graded by the LLM judge instead.",
                "equation_equivalent_decidable": "the same, over the items SymPy could compare",
                "symbolic_undecidable": "(lower is better, 0-1) fraction SymPy could not parse. "
                "These, and only these, are sent to the judge, whose verdict then fills "
                "equation_equivalent for them.",
                "structure_match": "(diagnostic, higher is better, 0-1) 1 when the answer uses "
                "the same monomials as the gold, ignoring numeric coefficients. The gap to "
                "equation_equivalent is the cost of withholding the data: a coefficient like the "
                "4 in 4*c*dx1dt + d2x1dt2*d1 is fixed by the measurements and by nothing else in "
                "the theory, so an answer with the right form and the wrong number is scored "
                "wrong -- correctly, but this is what makes that visible.",
                "equation_equivalent_<config>": "per system-size configuration (variables/"
                "derivatives/equations counts encoded in the directory name)",
            },
            primary_metric="equation_equivalent",
            decisions=[
                "REMOVED THE DATA ROWS FROM THE PROMPT. Earlier versions showed 8 rows sampled "
                "from the true system's .dat file. Those rows are generated by the equation "
                "being asked for, so a model can regress an equation onto them and never "
                "hypothesise about the system at all -- induction from examples, which is the "
                "inference type this suite exists to hold apart from abduction. The .dat files "
                "and options.noise_level are therefore no longer read.",
                "Showed the units of every variable, constant and derivative, and the units the "
                "corrected equation must carry. These are part of the axiom system rather than "
                "samples from it, and with the data gone they are what keeps the task "
                "determinate: dimensional consistency rules out most candidate equations.",
                "Marked which equation is wrong, so the task is recovering that equation rather "
                "than also locating it; locating it as well would conflate two abilities.",
                "Scored by zero set rather than by scalar factor. An equation set to zero is "
                "invariant under multiplication by any non-zero factor, constant or not, so the "
                "comparison is of the numerators' irreducible factors. All 540 equations in the "
                "release decide against themselves under this test.",
                "Sent only the unparseable residue to the LLM judge. Where SymPy decides, it has "
                "proved the answer; a judge could only make that verdict less reliable.",
            ],
            caveats=[
                "Items are constructed here, so scores are not comparable to published SynPAT "
                "results -- and less so now, since the published setting supplies the data this "
                "adapter withholds.",
                "Without the data the task is genuinely underdetermined: more than one "
                "dimensionally consistent equation can complete the system, and only the "
                "generating one is credited. The score is a strict lower bound on physical "
                "reasonableness, and should be read as 'recovered the intended law', not "
                "'proposed a wrong law'.",
                "NUMERIC COEFFICIENTS ARE NOT RECOVERABLE from the theory alone. Golds such as "
                "4*c*dx1dt + d2x1dt2*d1 carry a factor that only the measurements fix, so a "
                "model that reasons perfectly about the structure can still score 0 on the "
                "primary metric. structure_match is reported alongside for exactly this reason, "
                "and the two should be read together.",
                "The units of the corrected equation are shown. That is a real hint -- it fixes "
                "the dimension of the answer -- and it is given deliberately, because without "
                "it and without the data the item would be a lottery rather than a hard "
                "problem.",
            ],
            statistics=self.base_statistics(),
        )


def _clean_expression(text: str) -> str:
    """Strip prose and normalize an equation-like answer to an expression."""
    body = text.strip().strip("`").replace("^", "**")
    body = re.sub(r"^\s*(?:equation\s*\d+\s*[:=]?)\s*", "", body, flags=re.IGNORECASE)
    # Keep only the first line that looks like an expression.
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        if "=" in line:
            left, _, right = line.partition("=")
            right = right.strip()
            if right in ("0", "0.0"):
                return left.strip()
            return f"({left.strip()}) - ({right})"
        return line
    return body


def _equal_up_to_scale(
    candidate: str, gold: str, symbols: Sequence[str], timeout_s: float | None = None
) -> bool | None:
    """Equality up to a non-zero *scalar* factor, tolerant of LaTeX/unicode input.

    Kept as the narrow test; :func:`_equal_as_zero_set` is what scoring uses.
    """
    return _shared_equal_up_to_scale(candidate, gold, symbols, timeout_s=timeout_s)


def _equal_as_zero_set(
    candidate: str, gold: str, symbols: Sequence[str], timeout_s: float | None = None
) -> bool | None:
    """Do the two expressions vanish in the same place?

    The right question for this dataset. A scalar-factor test misses a correct
    answer written as a ratio -- ``Fg/Fc - dxdt/c`` is not a constant multiple
    of ``c*Fg - dxdt*Fc``, but the two equations say the same thing once both
    are set to zero.
    """
    return _shared_equal_as_zero_set(candidate, gold, symbols, timeout_s=timeout_s)
