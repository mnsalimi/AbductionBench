"""CLIMATE-FEVER: which claim status best explains the retrieved evidence.

Source: https://huggingface.co/datasets/tdiggelm/climate_fever
Formulation: CEDAR (Salimi et al., 2026), Stage-II abductive hypothesis selection

**The task, in the CEDAR formulation.**  A real-world climate claim arrives
with the evidence sentences retrieved for it from Wikipedia.  The evidence is
the observation; the claim's *status* is the hypothesis that accounts for it.
The model chooses which status does: the evidence supports the claim, refutes
it, does not settle it, or supports contradictory readings.  That last option
is what makes the task abductive rather than a straight entailment
classification -- DISPUTED is not a property of any single evidence sentence
but an explanation of why the set as a whole pulls both ways.

**All four statuses are offered.**  The release's ``claim_label`` is
0=SUPPORTS, 1=REFUTES, 2=NOT_ENOUGH_INFO, 3=DISPUTED, and dropping DISPUTED
would remove the part of the task that is not plain entailment.  The label
distribution is uneven (654/253/474/154 in the released split), so per-status
accuracy is reported beside the mean.

**Scoring.**  A four-way choice against the release's own label: accuracy,
checked mechanically.  No judge.
"""

from __future__ import annotations

from typing import Any

from ..core.adapter import SkippedDataset
from ..core.metrics import aggregate_mean_metrics
from ..core.types import AdapterDocumentation, ModelResponse, SampleScore, SampleSpec
from . import _common as C
from ._base import PooledDatasetAdapter, selection_score

REPO_ID = "tdiggelm/climate_fever"

#: The release's own label order, kept as the option order so a status's
#: position never depends on which status is correct.
STATUSES = (
    "The evidence supports the claim.",
    "The evidence refutes the claim.",
    "The evidence does not contain enough information to settle the claim.",
    "The evidence is disputed: it supports contradictory readings of the claim.",
)
STATUS_NAMES = ("SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED")


class ClimateFeverAdapter(PooledDatasetAdapter):
    """Select the claim status that accounts for the retrieved evidence."""

    adapter_version = "1.0"

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. You are given a claim about "
        "climate science and the evidence retrieved for it. Decide what relationship "
        "between the evidence and the claim best explains the evidence as a whole -- "
        "including the possibility that the evidence pulls in both directions, or that it "
        "simply does not settle the matter."
    )
    options_heading = "Answer options:"
    data_delivery_mode = "static"
    objective_metrics = True
    selection_cardinality = "single"
    hypothesis_modes = ("selection",)
    hypothesis_mode_options = {"selection": {"subtask": "selection"}}
    table_hypothesis_mode = "Selection"
    primary_metric = "accuracy"

    def load_items(self) -> list[dict[str, Any]]:
        root = C.ensure_hf_snapshot(
            REPO_ID,
            self.context.data_dir / "hf",
            offline=self.context.offline,
            allow_patterns=["*.parquet", "*.md"],
        )
        paths = C.find_files(root, ["*.parquet"])
        if not paths:
            raise SkippedDataset(f"no parquet files in the snapshot of {REPO_ID}")
        rows = C.read_parquet_rows(paths[0])
        if not rows:
            raise SkippedDataset(f"{paths[0]} contained no rows")
        self.split_used = f"{paths[0].name} in full ({len(rows)} claims; no official splits)"
        return rows

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        claim = C.normalize_whitespace(item.get("claim"))
        try:
            label = int(item.get("claim_label"))
        except (TypeError, ValueError):
            return None
        if not claim or not 0 <= label < len(STATUSES):
            return None
        sentences = []
        for entry in item.get("evidences") or []:
            if isinstance(entry, dict):
                text = C.normalize_whitespace(entry.get("evidence"))
                article = C.normalize_whitespace(entry.get("article"))
                if text:
                    sentences.append(f"- ({article}) {text}" if article else f"- {text}")
        if not sentences:
            return None
        labels = C.choice_labels(len(STATUSES))
        return SampleSpec(
            sample_id=C.stable_id("climatefever", item.get("claim_id", index)),
            fields={
                "context": "Retrieved evidence:\n" + "\n".join(sentences),
                "observation": f"Claim: {claim}",
                "question": "Which relationship between the evidence and the claim holds?",
                "options": list(STATUSES),
                "option_labels": labels,
            },
            reference={"gold_label": labels[label], "gold": STATUSES[label]},
            task_kind="selection",
            metadata={"status": STATUS_NAMES[label], "n_evidence": len(sentences)},
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
        # The label distribution is uneven, so a mean alone would let a model
        # that only ever answers SUPPORTS look competent.
        score.metrics[f"accuracy_{sample.metadata['status'].lower()}"] = score.metrics.get(
            "accuracy", 0.0
        )
        return score

    def aggregate(self, scores) -> dict[str, float]:
        metrics = aggregate_mean_metrics([score.metrics for score in scores])
        per_status = [
            metrics[f"accuracy_{name.lower()}"]
            for name in STATUS_NAMES
            if f"accuracy_{name.lower()}" in metrics
        ]
        if len(per_status) > 1:
            # Unweighted mean over the statuses present: the score a model gets
            # if the four are treated as equally important, which the uneven
            # release distribution otherwise hides.
            metrics["balanced_accuracy"] = sum(per_status) / len(per_status)
        return metrics

    def documentation(self) -> AdapterDocumentation:
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="CLIMATE-FEVER (CEDAR formulation)",
            domain="Scientific Reasoning: Evidence-Based Claim Explanation",
            source_url=f"https://huggingface.co/datasets/{REPO_ID}",
            processing_mode="Selection",
            split_used=getattr(self, "split_used", "the released split"),
            abductive_subset=(
                "The whole released set, under CEDAR's formulation: the retrieved evidence "
                "is the observation and the claim's status is the hypothesis that accounts "
                "for it. All four statuses are offered, DISPUTED included -- dropping it "
                "would leave plain entailment classification, because DISPUTED is not a "
                "property of any single evidence sentence but an explanation of why the set "
                "as a whole pulls both ways."
            ),
            sampling_procedure=self.sampling_note(),
            metrics_description={
                "accuracy": (
                    "(PRIMARY, higher is better, 0-1) 1.0 when the chosen status is the "
                    "release's claim_label. Four options, so chance is 0.25."
                ),
                "accuracy_<status>": (
                    "the same metric per status (supports / refutes / not_enough_info / "
                    "disputed) -- which relationships the model can recognise"
                ),
                "balanced_accuracy": (
                    "(higher is better, 0-1) unweighted mean of the per-status accuracies. "
                    "The released label distribution is uneven (654/253/474/154), so the "
                    "plain mean rewards a model that favours the common statuses."
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
                "Kept all four statuses, including DISPUTED, which is the part of the task "
                "that is abductive rather than entailment classification.",
                "Held the statuses in the release's own label order, so a status's position "
                "never depends on which one is correct.",
                "Showed every retrieved evidence sentence with its source article, since "
                "which article a sentence comes from bears on how much weight it carries.",
                "Reported balanced_accuracy beside accuracy because the label distribution "
                "is uneven.",
            ],
            caveats=[
                "Four options means chance is about 0.25.",
                "The evidence was retrieved by the release's own pipeline, so an item can be "
                "unanswerable because retrieval missed the relevant sentence rather than "
                "because the claim is genuinely unsettled -- NOT_ENOUGH_INFO conflates the "
                "two.",
                "Climate claims are contested in public discourse, so a model's answer may "
                "reflect a stance learned in pretraining rather than a reading of the "
                "evidence shown.",
            ],
            statistics={**self.base_statistics()},
        )
