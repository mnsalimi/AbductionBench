"""House M.D. rare-disease vignettes.

Source: https://bit.ly/4p9ltW8 -> Kaggle dataset
        ``arshgupta23/housemd-data-for-rare-disease-accuracy-using-llms``
        (the shortened URL in the suite's table resolves there)

The release is four Excel workbooks -- one per model that was evaluated -- each
with the same 176 rows: ``Prompt`` (a clinical vignette derived from an episode),
``Expected Response`` (the clinician-style differential and reasoning),
``Disease`` (the gold diagnosis), and one ``Actual Response (<model>)`` column
holding that model's answer.

Only ``Prompt``, ``Disease`` and ``Expected Response`` are used: the
``Actual Response`` columns are other models' outputs, and scoring against them
would measure imitation.  Rows are de-duplicated by vignette so the four
workbooks do not inflate the item count.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..core.adapter import SkippedDataset
from ..core.metrics import aggregate_mean_metrics, extract_answer_span, token_f1
from ..core.types import AdapterDocumentation, ModelResponse, SampleScore, SampleSpec
from . import _common as C
from ._base import PooledDatasetAdapter, text_match_score

KAGGLE_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "arshgupta23/housemd-data-for-rare-disease-accuracy-using-llms"
)
SOURCE_URL = "https://www.kaggle.com/datasets/arshgupta23/housemd-data-for-rare-disease-accuracy-using-llms"


class HouseMDAdapter(PooledDatasetAdapter):
    """Diagnose a rare-disease vignette; reference is the episode's disease."""

    adapter_version = "1.0"

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. You are given a clinical vignette "
        "whose presentation is unusual. Name the underlying disease that explains the "
        "combination of findings -- rare diseases are expected here, so do not default to the "
        "common condition that explains only part of the picture."
    )
    data_delivery_mode = "static"

    answer_format = "a single diagnosis"
    answer_constraints = (
        "give exactly one diagnosis",
        "output only the diagnosis name",
        "do not explain why",
        "do not use introductory phrases or commentary",
    )
    objective_metrics = True
    selection_cardinality = None
    primary_metric = "diagnosis_match"

    def load_items(self) -> list[dict[str, Any]]:
        archive = C.ensure_download(
            KAGGLE_URL, self.context.data_dir / "housemd.zip", offline=self.context.offline
        )
        extracted = C.extract_archive(archive, self.context.data_dir / "extracted")
        workbooks = C.find_files(extracted, ["*.xlsx"])
        if not workbooks:
            raise SkippedDataset("no Excel workbook found in the House M.D. archive")
        import pandas as pd

        seen: set[str] = set()
        items: list[dict[str, Any]] = []
        for path in workbooks:
            try:
                frame = pd.read_excel(path)
            except Exception as exc:  # noqa: BLE001 - a bad workbook must not stop the run
                self.log.warning("cannot read %s: %s", path.name, exc)
                continue
            for record in frame.to_dict(orient="records"):
                prompt = C.normalize_whitespace(record.get("Prompt"))
                disease = C.normalize_whitespace(record.get("Disease"))
                if not prompt or not disease:
                    continue
                key = prompt[:400]  # vignette text identifies the case
                if key in seen:
                    continue
                seen.add(key)
                items.append(
                    {
                        "prompt": prompt,
                        "disease": disease,
                        "expected": C.normalize_whitespace(record.get("Expected Response")),
                        "episode": C.normalize_whitespace(record.get("Episode")),
                        "workbook": path.name,
                    }
                )
        if not items:
            raise SkippedDataset("House M.D. workbooks contained no usable rows")
        self.split_used = (
            f"all unique vignettes across {len(workbooks)} workbook(s): {len(items)} cases "
            "(the release has no official split)"
        )
        return items

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        return SampleSpec(
            sample_id=C.stable_id("housemd", item.get("episode") or index, index),
            fields={
                "observation": item["prompt"],
                "question": "What is the most likely diagnosis?",
                "instructions": (
                    "Give a short differential, then commit to the single most likely diagnosis. "
                    "These are deliberately rare presentations."
                ),
            },
            reference={"gold": item["disease"], "expected": item.get("expected", "")},
            task_kind="generation",
            # Differential plus a committed answer, after a long vignette.
            max_tokens=1024,
            metadata={"episode": item.get("episode"), "workbook": item.get("workbook")},
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
        # Did the gold diagnosis appear anywhere (e.g. inside the differential)
        # even if it was not the committed answer?
        if response.text:
            from ..core.metrics import contains_match

            score.metrics["diagnosis_in_differential"] = float(
                contains_match(response.text, sample.reference["gold"])
            )
            expected = sample.reference.get("expected")
            if expected:
                score.metrics["reasoning_overlap"] = token_f1(response.text, expected)
        return score

    def aggregate(self, scores: Sequence[SampleScore]) -> dict[str, float]:
        return aggregate_mean_metrics([score.metrics for score in scores])

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
            name="using House M.D.",
            domain="Healthcare: Rare-Disease Diagnosis",
            source_url=SOURCE_URL,
            processing_mode="Generation",
            split_used=self.split_used,
            abductive_subset=(
                "All vignettes: infer the rare disease that explains the presentation. The "
                "per-model 'Actual Response' columns are excluded as model outputs."
            ),
            sampling_procedure=self.sampling_note()
            + "; vignettes are de-duplicated across the four workbooks first",
            metrics_description={
                "self_consistency_<metric>":
                "Every metric also gets a self_consistency_ counterpart: the plurality answer over "
                "modes.repeats samples of the same record, read off those samples rather than bought "
                "again. Available because this dataset's answers are checkable and so can coincide.",
                "diagnosis_match": "(PRIMARY, higher is better) 1 if the committed answer names the gold disease (primary)",
                "diagnosis_in_differential": "1 if the gold disease appears anywhere in the "
                "response -- the gap to the primary metric shows failures of commitment rather "
                "than of recall",
                "exact_match": "strict normalized equality with the gold disease",
                "token_f1": "bag-of-tokens F1 against the gold disease",
                "reasoning_overlap": "token F1 between the response and the reference differential",
                "diagnosis_match_judged": "LLM-judge equivalence verdict (only when "
                "engine.judge.enabled)",
            },
            primary_metric="diagnosis_match",
            decisions=[
                "Resolved the suite's shortened URL to the Kaggle dataset and download it through "
                "Kaggle's public API endpoint, which needs no credentials for this dataset.",
                "De-duplicated by vignette text, because the four workbooks differ only in which "
                "model's answers they carry.",
                "Ignored the 'Actual Response' columns; scoring against another model's output "
                "would not measure abduction.",
                "Reported diagnosis_in_differential separately, since these vignettes invite a "
                "differential and a model may list the right disease without committing to it.",
            ],
            caveats=[
                "The four workbooks hold overlapping but not identical vignette sets; after "
                "de-duplication the pool is larger than any single workbook, and the exact count "
                "is recorded in split_used.",
                "Vignettes derive from a television series whose episode diagnoses are widely "
                "discussed online; memorization is plausible.",
            ],
            statistics=self.base_statistics(),
        )
