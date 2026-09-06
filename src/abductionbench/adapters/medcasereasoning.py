"""MedCaseReasoning: diagnostic reasoning from clinician case reports.

Source: https://github.com/kevinwu23/Stanford-MedCaseReasoning
Data:   https://huggingface.co/datasets/zou-lab/MedCaseReasoning

The dataset pairs a case presentation (``case_prompt``) with the clinician's
``final_diagnosis`` and their explicit ``diagnostic_reasoning`` trace.  The
abductive task is to infer the diagnosis from the presentation; the reasoning
trace is deliberately **not** shown to the model (it contains the answer) but is
used to derive a secondary metric: how many of the clinician's reasoning
statements the model's own explanation recovers.
"""

from __future__ import annotations

import re
from typing import Any

from ..core.adapter import SkippedDataset
from ..core.metrics import extract_answer_span, token_f1
from ..core.types import AdapterDocumentation, ModelResponse, SampleScore, SampleSpec
from . import _common as C
from ._base import PooledDatasetAdapter, text_match_score

REPO_ID = "zou-lab/MedCaseReasoning"
GITHUB_URL = "https://github.com/kevinwu23/Stanford-MedCaseReasoning"


class MedCaseReasoningAdapter(PooledDatasetAdapter):
    """Diagnose a case report; also measure recovery of the clinician's reasoning."""

    adapter_version = "1.0"

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. You are given a clinical case "
        "report. State the final diagnosis, and let your reasoning follow the diagnostic "
        "evidence in the case rather than prior probability alone."
    )
    data_delivery_mode = "static"
    objective_metrics = True
    selection_cardinality = None
    primary_metric = "diagnosis_match"

    def load_items(self) -> list[dict[str, Any]]:
        root = C.ensure_hf_snapshot(
            REPO_ID, self.context.data_dir / "hf", offline=self.context.offline
        )
        files = C.find_files(root, ["*.parquet"])
        found = C.pick_split_file(files)
        if not found:
            raise SkippedDataset("no MedCaseReasoning parquet split found")
        path, split = found
        rows = C.read_parquet_rows(path)
        self.split_used = f"{split} ({len(rows)} case reports) from {path.name}"
        return rows

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        presentation = C.normalize_whitespace(item.get("case_prompt"))
        diagnosis = C.normalize_whitespace(item.get("final_diagnosis"))
        if not presentation or not diagnosis:
            return None
        reasoning = C.normalize_whitespace(item.get("diagnostic_reasoning"))
        return SampleSpec(
            sample_id=C.stable_id("mcr", item.get("pmcid") or index),
            fields={
                "observation": presentation,
                "question": "What is the most likely final diagnosis?",
                "instructions": (
                    "State the diagnosis, then briefly justify it from the findings that support "
                    "it."
                ),
            },
            reference={"gold": diagnosis, "reasoning": reasoning},
            task_kind="generation",
            # Diagnosis plus a short justification, after a long case.
            max_tokens=1024,
            metadata={"pmcid": item.get("pmcid"), "journal": item.get("journal")},
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
        # Secondary: did the model's own explanation touch the clinician's
        # reasoning statements?  Computed over the full response, not the answer
        # span, because the justification may precede the answer line.
        statements = _split_statements(sample.reference.get("reasoning") or "")
        if statements and response.text:
            hits = [
                1.0 if token_f1(response.text, statement) >= 0.3 else 0.0
                for statement in statements
            ]
            score.metrics["reasoning_recall"] = sum(hits) / len(hits)
        return score

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
            name="MedCaseReasoning",
            domain="Healthcare: Diagnostic Reasoning",
            source_url=GITHUB_URL,
            processing_mode="Generation",
            split_used=self.split_used,
            abductive_subset=(
                "The diagnostic inference: infer the final diagnosis that explains the case "
                "presentation. The dataset's clinician reasoning traces are withheld from the "
                "prompt (they state the answer) and used only for the reasoning_recall metric."
            ),
            sampling_procedure=self.sampling_note(),
            metrics_description={
                "diagnosis_match": "1 if the answer equals or contains the gold diagnosis (primary)",
                "exact_match": "strict normalized equality with the gold diagnosis",
                "token_f1": "bag-of-tokens F1 against the gold diagnosis",
                "rouge_l": "LCS F-measure against the gold diagnosis",
                "reasoning_recall": "fraction of the clinician's reasoning statements the model's "
                "response covers (token-F1 >= 0.3 per statement)",
                "diagnosis_match_judged": "LLM-judge equivalence verdict (only when "
                "engine.judge.enabled)",
            },
            primary_metric="diagnosis_match",
            decisions=[
                "Took the data from the authors' Hugging Face mirror; the GitHub repository "
                "contains code only.",
                "reasoning_recall uses a 0.3 token-F1 threshold per clinician statement -- the "
                "dataset defines no threshold, and this one credits a paraphrase while rejecting "
                "an unrelated sentence.",
                "Asked for a brief justification alongside the diagnosis so reasoning_recall is "
                "measurable; the answer line is still parsed for the diagnosis itself.",
                "max_tokens=1024 to fit a diagnosis plus justification after a long case report.",
            ],
            caveats=[
                "Case reports are published literature; large models may have memorized them.",
                "reasoning_recall rewards restating the clinician's specific statements, which "
                "correlates with, but is not identical to, sound reasoning.",
            ],
            statistics=self.base_statistics(),
        )


def _split_statements(reasoning: str) -> list[str]:
    """Split a clinician reasoning trace into its numbered statements."""
    if not reasoning:
        return []
    parts = re.split(r"(?:^|\s)\d+\.\s+", reasoning)
    statements = [part.strip() for part in parts if len(part.strip().split()) >= 4]
    return statements[:12]
