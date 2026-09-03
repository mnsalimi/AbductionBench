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
from ._base import PooledDatasetAdapter, text_match_score

REPO_URL = "https://github.com/MAGIC-AI4Med/MedRBench"


class MedRBenchAdapter(PooledDatasetAdapter):
    """Diagnosis-collection abduction from MedR-Bench case summaries."""

    adapter_version = "1.0"
    primary_metric = "diagnosis_match"

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
        score = text_match_score(
            response,
            gold=sample.reference["gold"],
            output_contract=output_contract,
            primary="diagnosis_match",
        )
        key = "diagnosis_match_rare" if sample.metadata.get("rare_disease") else "diagnosis_match_common"
        score.metrics[key] = score.metrics.get("diagnosis_match", 0.0)
        return score

    def aggregate(self, scores: Sequence[SampleScore]) -> dict[str, float]:
        metrics = aggregate_mean_metrics([score.metrics for score in scores])
        rare = metrics.get("diagnosis_match_rare")
        common = metrics.get("diagnosis_match_common")
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
            "criteria": "Equivalent disease entities (synonyms, abbreviations) count as correct.",
        }

    def apply_judge(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore, verdict: Any
    ) -> SampleScore:
        metrics = dict(score.metrics)
        metrics["diagnosis_match_judged"] = 1.0 if getattr(verdict, "positive", False) else 0.0
        return SampleScore(
            metrics=metrics,
            prediction=score.prediction,
            parse_ok=score.parse_ok,
            details={**score.details, "judge_label": getattr(verdict, "label", None)},
        )

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
                "diagnosis_match": "1 if the answer equals or contains the concise gold diagnosis "
                "(primary)",
                "exact_match": "strict normalized equality with the gold diagnosis",
                "token_f1": "bag-of-tokens F1 against the gold diagnosis",
                "rouge_l": "LCS F-measure against the gold diagnosis",
                "diagnosis_match_rare/_common": "the primary metric restricted to cases MedR-Bench "
                "flags as rare / not rare",
                "rare_disease_gap": "common minus rare accuracy -- how much rarity costs",
                "diagnosis_match_judged": "LLM-judge equivalence verdict (only when "
                "engine.judge.enabled)",
            },
            primary_metric="diagnosis_match",
            decisions=[
                "Used diagnosis_results (the concise gold string) as the reference rather than the "
                "long final_diagnosis discussion, which mixes the answer with its justification.",
                "Reported a rare/common breakdown using the dataset's own checked_rare_disease "
                "field, since that distinction is the benchmark's headline claim.",
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
