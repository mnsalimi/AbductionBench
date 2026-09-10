"""COPA and Balanced COPA: choose the more plausible cause of a premise.

Source: https://huggingface.co/datasets/pkavumba/balanced-copa
        (original COPA: Roemmele et al. 2011; Balanced COPA: Kavumba et al. 2019)

**Only the cause half is used.**  Every COPA item asks either *"what caused
this?"* or *"what happened as a result?"*, tagged in the release's ``question``
column.  The cause half is abduction -- infer the antecedent that best explains
the premise -- while the effect half is prediction in the other direction.
Items with ``question == "effect"`` are dropped, which halves each split and is
the same restriction the suite applies to XCOPA and e-CARE.

**Two datasets, one release.**  The Hugging Face mirror carries both:

``copa``    ``test.csv`` -- 500 original COPA items, no mirrored counterparts,
            250 of them cause questions.
``b_copa``  ``train.csv`` -- 1,000 items, half original and half *mirrored*.
            A mirrored item keeps the premise and swaps which alternative is
            correct, so a model that has learned a surface cue between the two
            alternatives scores well on one and badly on its mirror. That is
            the point of Balanced COPA, and it is why the two are reported
            separately rather than pooled.

**Scoring.**  Two alternatives, one correct, named by the release's ``label``
column: accuracy, checked mechanically. No judge.
"""

from __future__ import annotations

import csv
from typing import Any

from ..core.adapter import SkippedDataset
from ..core.metrics import aggregate_mean_metrics
from ..core.types import AdapterDocumentation, ModelResponse, SampleScore, SampleSpec
from . import _common as C
from ._base import PooledDatasetAdapter, selection_score

REPO_ID = "pkavumba/balanced-copa"


class _COPABase(PooledDatasetAdapter):
    """Shared loader: the cause half of one COPA file."""

    adapter_version = "1.0"

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. You are given something that "
        "happened and two candidate causes of it. Choose the one that more plausibly brought "
        "it about -- judge by everyday causal knowledge, not by which sentence shares more "
        "words with the premise."
    )
    options_heading = "Answer options:"
    data_delivery_mode = "static"
    objective_metrics = True
    selection_cardinality = "single"
    hypothesis_modes = ("selection",)
    hypothesis_mode_options = {"selection": {"subtask": "selection"}}
    table_hypothesis_mode = "Selection"
    primary_metric = "accuracy"

    #: Which file of the release this dataset is, set by the subclass.
    source_file = "test.csv"
    #: Whether the mirrored counterparts are part of this dataset.
    includes_mirrored = False

    def load_items(self) -> list[dict[str, Any]]:
        root = C.ensure_hf_snapshot(
            REPO_ID,
            self.context.data_dir / "hf",
            offline=self.context.offline,
            allow_patterns=["*.csv", "*.md"],
        )
        path = root / self.source_file
        if not path.is_file():
            raise SkippedDataset(f"{path} is not in the snapshot of {REPO_ID}")
        with open(path, encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise SkippedDataset(f"{path} contained no rows")
        # The abductive half only: an effect question asks what followed, which
        # is inference in the other direction.
        cause = [r for r in rows if str(r.get("question", "")).strip().lower() == "cause"]
        if not cause:
            raise SkippedDataset(
                f"{path} has {len(rows)} rows but none with question == 'cause'"
            )
        self.dropped_effect = len(rows) - len(cause)
        mirrored = sum(1 for r in cause if str(r.get("mirrored", "")).lower() == "true")
        self.mirrored_count = mirrored
        self.split_used = (
            f"{self.source_file}: {len(cause)} cause questions "
            f"({self.dropped_effect} effect questions dropped"
            + (f"; {mirrored} of them mirrored" if mirrored else "")
            + ")"
        )
        return cause

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        premise = C.normalize_whitespace(item.get("premise"))
        first = C.normalize_whitespace(item.get("choice1"))
        second = C.normalize_whitespace(item.get("choice2"))
        raw_label = str(item.get("label", "")).strip()
        if not premise or not first or not second or raw_label not in {"0", "1"}:
            return None
        options = [first, second]
        labels = C.choice_labels(2)
        return SampleSpec(
            sample_id=C.stable_id(self.dataset_id, item.get("id", index)),
            fields={
                "observation": premise,
                "question": "What was the more plausible cause of this?",
                "options": options,
                "option_labels": labels,
            },
            reference={"gold_label": labels[int(raw_label)], "gold": options[int(raw_label)]},
            task_kind="selection",
            metadata={
                "mirrored": str(item.get("mirrored", "")).lower() == "true",
                "copa_id": item.get("id"),
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
        if self.includes_mirrored:
            # The comparison Balanced COPA exists to make: an item and its
            # mirror share a premise, so a gap between these two means a
            # surface cue is being used rather than causal knowledge.
            key = "accuracy_mirrored" if sample.metadata["mirrored"] else "accuracy_original"
            score.metrics[key] = score.metrics.get("accuracy", 0.0)
        return score

    def aggregate(self, scores) -> dict[str, float]:
        metrics = aggregate_mean_metrics([score.metrics for score in scores])
        original, mirrored = metrics.get("accuracy_original"), metrics.get("accuracy_mirrored")
        if original is not None and mirrored is not None:
            # Positive means the model does better on the items whose surface
            # cues point the usual way.
            metrics["mirror_gap"] = original - mirrored
        return metrics


class COPAAdapter(_COPABase):
    """Original COPA, cause questions only."""

    source_file = "test.csv"
    includes_mirrored = False

    def documentation(self) -> AdapterDocumentation:
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="COPA",
            domain="Commonsense: Causal Explanation",
            source_url=f"https://huggingface.co/datasets/{REPO_ID}",
            processing_mode="Selection",
            split_used=getattr(self, "split_used", "test.csv"),
            abductive_subset=(
                "THE CAUSE QUESTIONS ONLY. Half of COPA asks what followed from the premise, "
                "which is inference in the predictive direction; those rows are dropped on "
                "the release's own `question` column. What remains asks which of two "
                "alternatives more plausibly brought the premise about, which is selection "
                "among candidate explanations."
            ),
            sampling_procedure=self.sampling_note(),
            metrics_description={
                "accuracy": (
                    "(PRIMARY, higher is better, 0-1) 1.0 when the chosen alternative is the "
                    "one the release marks correct. Two options, so chance is 0.5."
                ),
                "parse_failure_rate": (
                    "(lower is better, 0-1) fraction of responses no option could be read "
                    "from; these score 0 and are counted separately from being wrong."
                ),
                "self_consistency_<metric>":
                "Every metric also gets a self_consistency_ counterpart: the plurality answer "
                "over modes.repeats samples of the same record, read off those samples rather "
                "than bought again. Available because this dataset's answers are checkable and "
                "so can coincide.",
            },
            primary_metric="accuracy",
            decisions=[
                "Used the cause half only, filtered on the release's own question column.",
                "Read test.csv, which carries the original items with no mirrored "
                "counterparts; the balanced set is the separate b_copa dataset.",
                "Presented the two alternatives in the release's own order, so no ordering "
                "choice made here can favour either.",
            ],
            caveats=[
                "Two options means a model that always guesses scores about 0.5; read "
                "accuracy against that floor, not against zero.",
                "COPA is old and widely reproduced, so it may sit in pretraining data.",
            ],
            statistics={
                **self.base_statistics(),
                "effect_questions_dropped": getattr(self, "dropped_effect", 0),
            },
        )


class BalancedCOPAAdapter(_COPABase):
    """Balanced COPA: the same premises, with mirrored counterparts."""

    source_file = "train.csv"
    includes_mirrored = True

    def documentation(self) -> AdapterDocumentation:
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="Balanced COPA",
            domain="Commonsense: Balanced Causal Explanation",
            source_url=f"https://huggingface.co/datasets/{REPO_ID}",
            processing_mode="Selection",
            split_used=getattr(self, "split_used", "train.csv"),
            abductive_subset=(
                "THE CAUSE QUESTIONS ONLY, as for COPA: the effect half asks what followed "
                "and is dropped on the release's own `question` column. What is added here "
                "over COPA is the mirrored counterparts -- same premise, correct alternative "
                "swapped -- which is what makes the set balanced."
            ),
            sampling_procedure=self.sampling_note(),
            metrics_description={
                "accuracy": (
                    "(PRIMARY, higher is better, 0-1) 1.0 when the chosen alternative is the "
                    "one the release marks correct, over original and mirrored items alike."
                ),
                "accuracy_original": "accuracy restricted to the original items",
                "accuracy_mirrored": "accuracy restricted to their mirrored counterparts",
                "mirror_gap": (
                    "(closer to 0 is better) accuracy_original minus accuracy_mirrored. This "
                    "is what Balanced COPA exists to expose: an item and its mirror share a "
                    "premise, so a gap means a surface cue between the alternatives is being "
                    "used instead of causal knowledge."
                ),
                "parse_failure_rate": (
                    "(lower is better, 0-1) fraction of responses no option could be read from"
                ),
                "self_consistency_<metric>":
                "Every metric also gets a self_consistency_ counterpart: the plurality answer "
                "over modes.repeats samples of the same record, read off those samples rather "
                "than bought again. Available because this dataset's answers are checkable and "
                "so can coincide.",
            },
            primary_metric="accuracy",
            decisions=[
                "Used the cause half only, filtered on the release's own question column.",
                "Read train.csv, which is where the mirrored counterparts live; test.csv "
                "carries none and is the separate copa dataset.",
                "Reported original and mirrored accuracy separately and their gap, because a "
                "pooled mean hides exactly the effect the dataset was built to measure.",
            ],
            caveats=[
                "This is the release's train split. It is used as an evaluation set here "
                "because it is the only file carrying mirrored items -- worth knowing if a "
                "model under test may have trained on it.",
                "Two options means chance is about 0.5.",
                "It shares premises with copa, so the two are not independent evidence; a "
                "suite-level average over both counts those premises twice.",
            ],
            statistics={
                **self.base_statistics(),
                "effect_questions_dropped": getattr(self, "dropped_effect", 0),
                "mirrored_items": getattr(self, "mirrored_count", 0),
            },
        )
