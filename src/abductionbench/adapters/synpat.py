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
a corrupted system (one equation replaced), told that exactly one equation is
inconsistent with the measurements, given a sample of the true system's data,
and asked for the corrected equation.  The gold answer is that equation from
``system.txt``.  Which system, which replacement and which data rows are shown
is decided by a seeded RNG, so all models see identical items.

Scoring uses SymPy: equations are expressions equal to zero, so a candidate is
correct if it is equal to the gold expression **up to a non-zero scalar factor**.
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from typing import Any

from ..core.adapter import SkippedDataset
from ..core.metrics import aggregate_mean_metrics, extract_answer_span, token_f1
from ..core.types import AdapterDocumentation, ModelResponse, SampleScore, SampleSpec
from . import _common as C
from ._base import PooledDatasetAdapter, unparsed_score
from ._mathnorm import equal_up_to_scale as _shared_equal_up_to_scale

HF_REPO = "Karan0901/synpat-dataset"
GITHUB_URL = "https://github.com/jlenchner/theorizer"


def _parse_system(text: str) -> dict[str, Any]:
    """Parse a SynPAT ``system.txt`` / ``replacement_k.txt`` file."""
    out: dict[str, Any] = {"equations": [], "variables": [], "constants": [], "derivatives": []}
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
        elif lowered.startswith("units of measure"):
            section = None
        elif section == "equations":
            out["equations"].append(stripped)
    return out


def _parse_list(line: str) -> list[str]:
    match = re.search(r"\[(.*)\]", line)
    if not match:
        return []
    return [item.strip().strip("'\"") for item in match.group(1).split(",") if item.strip()]


class SynPATAdapter(PooledDatasetAdapter):
    """Recover the equation of an axiom system that the data contradicts."""

    adapter_version = "1.0"

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. You are given an axiom system of "
        "physical equations and data that contradicts it. Identify the equation that is wrong "
        "and state its corrected form, in the symbols the system uses."
    )
    data_delivery_mode = "static"

    answer_format = "one equation in the given symbols"
    answer_constraints = (
        "output exactly one equation",
        "use only the symbols given",
        "output only the equation",
        "do not use introductory phrases or commentary",
    )
    objective_metrics = True
    selection_cardinality = None
    primary_metric = "symbolic_match"

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

        noise = str(self.context.option("noise_level", "0.001"))
        data_path = directory / f"system_{noise}.dat"
        if not data_path.exists():
            candidates = sorted(directory.glob("system_*.dat"))
            data_path = candidates[0] if candidates else None
        rows = int(self.context.option("data_rows", 8))
        data_block = ""
        if data_path is not None:
            lines = C.read_lines(data_path)
            if lines:
                rng = random.Random(f"{self.context.seed}::synpat::{item['directory']}")
                header, body = lines[0], lines[1:]
                chosen = rng.sample(body, min(rows, len(body))) if body else []
                formatted = [
                    "\t".join(f"{float(value):.4g}" for value in line.split())
                    for line in chosen
                ]
                data_block = "\n".join([header, *formatted])

        equations_text = "\n".join(
            f"{i + 1}. {equation}" + ("   <- inconsistent with the data" if i == position else "")
            for i, equation in enumerate(corrupted["equations"])
        )
        return SampleSpec(
            sample_id=C.stable_id("synpat", item["config"], item["system_name"], item["replacement"]),
            fields={
                "context": (
                    f"Variables: {', '.join(true_system['variables'])}\n"
                    f"Constants: {', '.join(true_system['constants'])}\n"
                    f"Derivatives: {', '.join(true_system['derivatives'])}\n\n"
                    "Proposed axiom system (each equation is an expression equal to 0):\n"
                    + equations_text
                    + (
                        f"\n\nMeasurements from the real system (noise level {noise}):\n{data_block}"
                        if data_block
                        else ""
                    )
                ),
                "observation": (
                    f"Equation {position + 1} of the proposed system is inconsistent with the "
                    "measurements."
                ),
                "instructions": (
                    f"Give the corrected equation {position + 1}: a single expression that equals "
                    "zero for the measured data, using only the listed variables, constants and "
                    "derivatives, written in Python-style notation (** for powers)."
                ),
            },
            reference={
                "gold": gold,
                "symbols": true_system["variables"]
                + true_system["constants"]
                + true_system["derivatives"],
                "position": position,
            },
            task_kind="knowledge_completion",
            # Fitting an equation to data needs working room.
            max_tokens=1536,
            metadata={
                "config": item["config"],
                "system": item["system_name"],
                "replacement": item["replacement"],
                "noise_level": noise,
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
                ["symbolic_match", "equation_token_f1"], raw=response.text[:200]
            )
        gold = sample.reference["gold"]
        candidate = _clean_expression(answer)
        verdict = _equal_up_to_scale(candidate, gold, sample.reference["symbols"])
        metrics = {
            "equation_token_f1": token_f1(candidate or answer, gold),
            "symbolic_match": 0.0 if verdict is None else float(verdict),
            "symbolic_undecidable": 1.0 if verdict is None else 0.0,
        }
        config = sample.metadata.get("config")
        if config:
            metrics[f"symbolic_match_{config}"] = metrics["symbolic_match"]
        return SampleScore(
            metrics=metrics,
            prediction=(candidate or answer)[:300],
            details={"gold": gold[:200]},
        )

    def aggregate(self, scores: Sequence[SampleScore]) -> dict[str, float]:
        metrics = aggregate_mean_metrics([score.metrics for score in scores])
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
            name="SynPAT",
            domain="Scientific Discovery: Physics",
            source_url=GITHUB_URL,
            processing_mode="Generation",
            split_used=self.split_used,
            abductive_subset=(
                "Constructed from the release's own files: a system whose equation k has been "
                "replaced (replacement_k.txt) plus data generated from the true system, with the "
                "true equation k as the reference. Items where the corruption does not isolate to "
                "exactly one equation are skipped."
            ),
            sampling_procedure=self.sampling_note()
            + "; the data rows shown are chosen by a directory-seeded RNG, identical across models",
            metrics_description={
                "symbolic_match": "1 if SymPy proves the answer equal to the gold expression up to "
                "a non-zero scalar factor (equations are expressions equal to zero) -- primary; an "
                "undecidable comparison counts as 0",
                "symbolic_match_decidable": "the same over items SymPy could compare",
                "symbolic_undecidable": "fraction of items where parsing/simplification failed",
                "equation_token_f1": "token F1 against the gold expression, a lenient view",
                "symbolic_match_<config>": "per system-size configuration (variables/derivatives/"
                "equations counts encoded in the directory name)",
            },
            primary_metric="symbolic_match",
            decisions=[
                "Used the low-noise data file (0.001) by default; options.noise_level selects "
                "0.01/0.05/0.1, which makes the task progressively harder.",
                "Showed 8 randomly chosen data rows (options.data_rows) -- enough to constrain an "
                "equation while keeping the prompt inside the input budget.",
                "Marked which equation is inconsistent, so the task is recovering that equation "
                "rather than also locating it; locating it as well would conflate two abilities.",
                "Scored equality up to a scalar factor, since an equation set to zero is invariant "
                "under non-zero scaling.",
            ],
            caveats=[
                "Items are constructed here, so scores are not comparable to published SynPAT "
                "results.",
                "A different equation can fit 8 noisy rows; symbolic_match against the generating "
                "equation is therefore a strict lower bound.",
            ],
            statistics={
                **self.base_statistics(),
                "noise_level": str(self.context.option("noise_level", "0.001")),
                "data_rows": int(self.context.option("data_rows", 8)),
            },
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


def _equal_up_to_scale(candidate: str, gold: str, symbols: Sequence[str]) -> bool | None:
    """Equality up to a non-zero scalar factor, tolerant of LaTeX/unicode input."""
    return _shared_equal_up_to_scale(candidate, gold, symbols)
