"""GEAR: general evaluation framework for abductive reasoning (He et al., 2025).

Paper:  arXiv:2509.24096
Source: https://github.com/KaiyuHe998/GEAR-Abduction_evaluation
        (confirmed from the arXiv page's Code-and-Data link; the URL given in
        the suite's table was a placeholder pointing at a different project)

The repository's ``data/`` directory holds two task families: **ACRE** (abstract
causal reasoning -- "blicket" experiments) and **ARC-AGI** grid puzzles.

**ACRE is used.** Each item shows six experiments in which a set of objects is
placed on a detector that turns on or off, then asks what happens for new object
sets.  The answer is ``on``, ``off``, or -- crucially -- ``undetermined``:
inferring which objects are blickets is abduction, and the ``undetermined``
option makes the *defeasibility* of that inference measurable, which is exactly
what this suite wants to probe.  Test cases are typed (``direct``, ``indirect``,
``screen_off``, ``potential``), and accuracy is reported per type.

**ARC is not used by default.** ARC-AGI asks for a grid transformation program;
it is closer to program induction, its prompts are large grid dumps, and the
released split with solutions is the training/evaluation set of a different
competition. ``options.subtask = arc`` exists for completeness but is not the
suite's default, and this choice is recorded here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..core.adapter import SkippedDataset
from ..core.metrics import aggregate_mean_metrics
from ..core.types import AdapterDocumentation, ModelResponse, SampleScore, SampleSpec
from . import _common as C
from ._base import PooledDatasetAdapter, selection_score

REPO_URL = "https://github.com/KaiyuHe998/GEAR-Abduction_evaluation"
LABELS = ["A", "B", "C"]
OUTCOMES = ["on", "off", "undetermined"]


class GearAdapter(PooledDatasetAdapter):
    """ACRE blicket experiments: on / off / undetermined."""

    adapter_version = "1.0"

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. You are shown blicket-detector "
        "experiments: which objects were placed on the detector and whether it activated. "
        "Decide whether the queried object activates the detector. Answer 'undetermined' when "
        "the experiments genuinely do not settle it -- guessing is worse than admitting the "
        "evidence is incomplete."
    )
    data_delivery_mode = "static"
    objective_metrics = True
    selection_cardinality = "single"
    primary_metric = "accuracy"

    def load_items(self) -> list[dict[str, Any]]:
        subtask = str(self.context.option("subtask", "acre"))
        if subtask != "acre":
            raise SkippedDataset(
                f"GEAR subtask {subtask!r} is not implemented; only the ACRE family is used "
                "(see the adapter docstring for why ARC-AGI is excluded)"
            )
        root = C.ensure_git_repo(
            REPO_URL, self.context.data_dir / "repo", offline=self.context.offline
        )
        path = root / "data" / "acre.jsonl"
        if not path.exists():
            raise SkippedDataset(f"GEAR ACRE file not found: {path}")
        rows = C.read_jsonl(path)
        items: list[dict[str, Any]] = []
        for row in rows:
            train = row.get("train") or []
            for position, test in enumerate(row.get("test") or []):
                items.append(
                    {
                        "idx": row.get("idx"),
                        "position": position,
                        "train": train,
                        "test": test,
                    }
                )
        if not items:
            raise SkippedDataset("GEAR ACRE file produced no test cases")
        self.split_used = (
            f"acre.jsonl in full ({len(rows)} experiment sets flattened into {len(items)} test "
            "cases); the release ships a single evaluation file"
        )
        return items

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        test = item["test"]
        outcome = str(test.get("output", "")).lower()
        if outcome not in OUTCOMES:
            return None
        experiments = []
        for position, trial in enumerate(item["train"], start=1):
            objects = ", ".join(str(obj) for obj in C.as_list(trial.get("input")))
            result = str(trial.get("output", "")).lower()
            experiments.append(f"{position}. Objects on the detector: {objects} -> light {result}")
        query_objects = ", ".join(str(obj) for obj in C.as_list(test.get("input")))
        if not experiments or not query_objects:
            return None
        return SampleSpec(
            sample_id=C.stable_id("gear-acre", item.get("idx"), item["position"]),
            fields={
                "context": (
                    "A detector lights up when at least one 'blicket' object is placed on it. "
                    "These experiments were observed:\n" + "\n".join(experiments)
                ),
                "observation": f"Now these objects are placed on the detector: {query_objects}",
                "question": "What does the light do?",
                "options": [
                    "The light turns on.",
                    "The light stays off.",
                    "It cannot be determined from the experiments.",
                ],
                "option_labels": LABELS,
                "instructions": (
                    "Work out which objects must be blickets, which cannot be, and which are "
                    "still unknown. Choose 'cannot be determined' only if the experiments leave "
                    "the outcome genuinely open."
                ),
            },
            reference={"gold_label": LABELS[OUTCOMES.index(outcome)], "outcome": outcome},
            task_kind="selection",
            # Six experiments to reason over; the answer is one label.
            max_tokens=768,
            metadata={
                "acre_idx": item.get("idx"),
                "case_type": test.get("type"),
                "outcome": outcome,
            },
        )

    def score(
        self,
        sample: SampleSpec,
        response: ModelResponse,
        *,
        output_contract: dict[str, Any] | None = None,
    ) -> SampleScore:
        score = selection_score(
            response,
            labels=sample.fields["option_labels"],
            gold_label=sample.reference["gold_label"],
            output_contract=output_contract,
            metric_name="accuracy",
        )
        case_type = sample.metadata.get("case_type") or "unknown"
        score.metrics[f"accuracy_{case_type}"] = score.metrics.get("accuracy", 0.0)
        outcome = sample.metadata.get("outcome")
        if outcome == "undetermined":
            # Defeasibility: does the model admit that the evidence is insufficient?
            score.metrics["undetermined_recall"] = score.metrics.get("accuracy", 0.0)
        else:
            chosen = score.prediction
            score.metrics["overcaution_rate"] = float(chosen == LABELS[2])
        return score

    def aggregate(self, scores: Sequence[SampleScore]) -> dict[str, float]:
        return aggregate_mean_metrics([score.metrics for score in scores])

    def documentation(self) -> AdapterDocumentation:
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="GEAR (ACRE family)",
            domain="General: Abductive Hypothesis Evaluation",
            source_url=REPO_URL,
            processing_mode="Selection",
            split_used=self.split_used,
            abductive_subset=(
                "The ACRE blicket task: infer the hidden causal property of objects from six "
                "experiments, then predict a new configuration -- with 'undetermined' available "
                "when the evidence underdetermines the answer. GEAR's ARC-AGI family is program "
                "induction over grids and is excluded by default (options.subtask = arc)."
            ),
            sampling_procedure=self.sampling_note()
            + "; experiment sets are first flattened into one sample per test case",
            metrics_description={
                "accuracy": "1 if the chosen outcome (on / off / undetermined) is correct (primary)",
                "accuracy_<case_type>": "accuracy on direct, indirect, screen_off or potential cases",
                "undetermined_recall": "accuracy restricted to items whose gold answer is "
                "'undetermined' -- how well the model recognizes underdetermined evidence, i.e. "
                "the defeasible character of abduction",
                "overcaution_rate": "fraction of determinate items where the model answered "
                "'undetermined' -- the opposite failure",
            },
            primary_metric="accuracy",
            decisions=[
                "Confirmed the repository from the arXiv page (arXiv:2509.24096); the URL in the "
                "suite's table pointed at an unrelated project and was not used.",
                "Used the ACRE family only, and reported why ARC-AGI is excluded.",
                "Flattened each experiment set into one sample per test case, so a score is not "
                "an average over four coupled answers.",
                "Added undetermined_recall and overcaution_rate, because a single accuracy number "
                "hides whether a model is over- or under-committing -- the property this dataset "
                "is uniquely able to measure.",
            ],
            caveats=[
                "Three options means chance accuracy is ~33%.",
                "ACRE items are synthetic and short; scores are not comparable to the visual ACRE "
                "benchmark, which uses images.",
            ],
            statistics=self.base_statistics(),
        )
