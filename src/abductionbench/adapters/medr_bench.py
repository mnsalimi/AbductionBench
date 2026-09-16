"""MedR-Bench (MedRBench): end-to-end clinical reasoning.

Source: https://github.com/MAGIC-AI4Med/MedRBench/tree/main/data

The release ships two case collections: 957 **diagnosis** cases and 496
**treatment** cases.  Only the diagnosis collection is abductive -- inferring
the disease that explains the findings.  Treatment planning selects an
intervention for an already-known diagnosis, which is decision-making rather
than inference to the best explanation, so it is excluded.

Each diagnosis case provides ``case_summary`` (the findings), a
``differential_diagnosis`` narrative, a long ``final_diagnosis`` discussion and
a concise ``diagnosis_results`` string.  The concise string is the reference;
the differential and final-diagnosis narratives are withheld from the prompt
because they state the answer.

``checked_rare_disease`` marks rare-disease cases, so accuracy is also reported
separately for them -- the split MedR-Bench itself highlights.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..core.adapter import SkippedDataset
from ..core.metrics import aggregate_mean_metrics, extract_answer_span
from ..core.types import AdapterDocumentation, ModelResponse, SampleScore, SampleSpec
from . import _common as C
from ._base import PooledDatasetAdapter, apply_judged_metric, judged_only_score

REPO_URL = "https://github.com/MAGIC-AI4Med/MedRBench"


class MedRBenchAdapter(PooledDatasetAdapter):
    """Diagnosis-collection abduction from MedR-Bench case summaries."""

    adapter_version = "1.0"

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. You are given a case summary. "
        "State the diagnosis that accounts for the findings; where the case is of a rare "
        "disease, the common look-alike is not the answer."
    )
    data_delivery_mode = "static"

    answer_format = "a single diagnosis"
    answer_constraints = (
        "give exactly one diagnosis",
        "output only the diagnosis name",
        "do not explain why",
        "do not use introductory phrases or commentary",
    )
    #: Measured, not assumed: the model writes a disease name into an open
    #: vocabulary with no candidate list, so a correct answer routinely differs
    #: from the gold in wording -- synonym, eponym, abbreviation, subtype -- and
    #: fails a string comparison. The gold exists; its surface form is not the answer.
    objective_metrics = False
    selection_cardinality = None
    primary_metric = "diagnosis_judged"

    def load_items(self) -> list[dict[str, Any]]:
        root = C.ensure_git_repo(
            REPO_URL, self.context.data_dir / "repo", offline=self.context.offline
        )
        candidates = C.find_files(root / "data", ["*diagnosis*.json"])
        if not candidates:
            raise SkippedDataset("MedR-Bench diagnosis JSON not found in the repository")
        payload = C.read_json(candidates[0])
        if not isinstance(payload, dict):
            raise SkippedDataset("unexpected MedR-Bench structure (expected a dict of cases)")
        rows = [{"pmcid": key, **value} for key, value in payload.items() if isinstance(value, dict)]
        self.split_used = (
            f"whole diagnosis collection ({len(rows)} cases from {candidates[0].name}); the "
            "release has no train/test split, and the treatment collection is excluded as "
            "non-abductive"
        )
        return rows

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        case = item.get("generate_case") or {}
        summary = C.normalize_whitespace(case.get("case_summary"))
        gold = C.normalize_whitespace(case.get("diagnosis_results"))
        if not summary or not gold:
            return None
        rare = bool(item.get("checked_rare_disease"))
        return SampleSpec(
            sample_id=C.stable_id("medrb", item.get("pmcid", index)),
            fields={
                "observation": summary,
                "question": "What is the most likely diagnosis?",
                "instructions": (
                    "Name the diagnosis that best explains the findings. If the case involves a "
                    "complication or an underlying cause, state both."
                ),
            },
            reference={"gold": gold},
            task_kind="generation",
            # A diagnosis (sometimes two-part) after a structured case summary.
            max_tokens=1024,
            metadata={
                "pmcid": item.get("pmcid"),
                "rare_disease": rare,
                "body_category": C.as_list(item.get("body_category"))[:2],
                "disorder_category": C.as_list(item.get("disorder_category"))[:2],
            },
        )

    def score(
        self,
        sample: SampleSpec,
        response: ModelResponse,
        *,
        output_contract: dict[str, Any] | None = None,
    ) -> SampleScore:
        key = (
            "diagnosis_judged_rare"
            if sample.metadata.get("rare_disease")
            else "diagnosis_judged_common"
        )
        # The stratum is seeded here and filled by the judge along with the base
        # metric, so the rare/common split is the same verdict seen through a filter.
        return judged_only_score(
            response,
            metric="diagnosis_judged",
            output_contract=output_contract,
            extra_metrics={key: 0.0},
            details={"gold": str(sample.reference["gold"])[:300]},
        )

    def aggregate(self, scores: Sequence[SampleScore]) -> dict[str, float]:
        metrics = aggregate_mean_metrics([score.metrics for score in scores])
        rare = metrics.get("diagnosis_judged_rare")
        common = metrics.get("diagnosis_judged_common")
        if rare is not None and common is not None:
            metrics["rare_disease_gap"] = common - rare
        return metrics

    def judge_request(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore
    ) -> dict[str, Any] | None:
        if not response.text:
            return None
        return {
            "candidate": extract_answer_span(response.text, None)[:600],
            "gold": sample.reference["gold"],
            "observation": C.clip_words(sample.fields["observation"], 200),
            "criteria": (
                "The candidate is correct if it names the same disease entity as the "
                "reference, however it is written: synonyms, abbreviations, eponyms and "
                "spelling variants all count. A broader category that does not identify "
                "the reference disease, or a different disease that shares symptoms with "
                "it, does not count."
            ),
        }

    def apply_judge(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore, verdict: Any
    ) -> SampleScore:
        return apply_judged_metric(score, verdict, "diagnosis_judged")

    def documentation(self) -> AdapterDocumentation:
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="MedR-Bench",
            domain="Healthcare: End-to-End Clinical Reasoning",
            source_url=REPO_URL,
            processing_mode="Generation (selection not supported by the release)",
            split_used=self.split_used,
            abductive_subset=(
                "The 957-case diagnosis collection only. The 496 treatment cases ask which "
                "intervention to choose for a known diagnosis -- decision-making, not abduction "
                "-- and are excluded. Within a case, only case_summary is shown; the "
                "differential_diagnosis and final_diagnosis narratives are withheld because they "
                "contain the answer."
            ),
            sampling_procedure=self.sampling_note(),
            metrics_description={
                "best_of_n_<metric>":
                "Every metric also gets a best_of_n_ counterpart: per record, the repeat the "
                "judge scored highest. A diagnosis is free text with many correct surface forms, "
                "so a plurality over repeats is not meaningful and Best-of-N replaces it.",
                "diagnosis_judged": "(PRIMARY, higher is better, 0-1) LLM-judge verdict on "
                "whether the stated diagnosis is the same disease entity as the concise gold "
                "diagnosis, however written. 1.0 when the judge affirms.",
                "diagnosis_judged_rare/_common": "the primary metric restricted to cases "
                "MedR-Bench flags as rare / not rare",
                "rare_disease_gap": "common minus rare score -- how much rarity costs",
                "parse_failure_rate": "(lower is better, 0-1) fraction of responses no diagnosis "
                "could be read from; these score 0 and are counted separately from being wrong.",
            },
            primary_metric="diagnosis_judged",
            decisions=[
                "Used diagnosis_results (the concise gold string) as the reference rather than the "
                "long final_diagnosis discussion, which mixes the answer with its justification.",
                "Reported a rare/common breakdown using the dataset's own checked_rare_disease "
                "field, since that distinction is the benchmark's headline claim.",
                "Scored by an LLM judge rather than by string comparison: no candidate list is "
                "shown, so a correct diagnosis is written into an open vocabulary where the same "
                "disease has many correct surface forms -- and rare diseases, the point of this "
                "benchmark, are exactly where synonyms and eponyms proliferate.",
                "No selection mode: the release provides no candidate diagnosis sets.",
                "Did not attempt MedR-Bench's reasoning-quality metrics (factuality, efficiency), "
                "which require their own LLM-judge pipeline over intermediate reasoning steps; "
                "diagnosis accuracy is the abductive outcome measured here.",
            ],
            caveats=[
                "Cases are PubMed case reports and may be memorized by large models.",
                "Two-part gold answers ('X secondary to Y') are credited when the answer contains "
                "the gold string, so a partially correct answer may score 0 on the primary metric "
                "while scoring well on token_f1.",
            ],
            statistics=self.base_statistics(),
        )
