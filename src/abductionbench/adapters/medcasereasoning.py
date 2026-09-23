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
from ._base import PooledDatasetAdapter, apply_judged_metric, judged_only_score

REPO_ID = "zou-lab/MedCaseReasoning"
GITHUB_URL = "https://github.com/kevinwu23/Stanford-MedCaseReasoning"


class MedCaseReasoningAdapter(PooledDatasetAdapter):
    """Diagnose a case report; also measure recovery of the clinician's reasoning."""

    adapter_version = "2.0"

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. You are given a clinical case "
        "report. State the final diagnosis that the case's own diagnostic evidence "
        "supports, rather than the one prior probability favours."
    )
    data_delivery_mode = "static"

    answer_format = "a single diagnosis"
    #: Measured, not assumed: the model writes a disease name into an open
    #: vocabulary with no candidate list, so a correct answer routinely differs
    #: from the gold in wording -- synonym, eponym, abbreviation, subtype -- and
    #: fails a string comparison. The gold exists; its surface form is not the answer.
    objective_metrics = False
    selection_cardinality = None
    primary_metric = "diagnosis_judged"

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
                "instructions": "State the final diagnosis the case supports.",
            },
            reference={"gold": diagnosis, "reasoning": reasoning},
            task_kind="generation",
            # Room for a diagnosis, and for the reasoning cot asks for.
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
        score = judged_only_score(
            response,
            metric="diagnosis_judged",
            output_contract=output_contract,
            details={"gold": str(sample.reference["gold"])[:300]},
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
            # score.prediction is what this dataset's own scorer already read
            # out of the response *with the answer contract* -- i.e. the text
            # after the Answer: marker. Re-parsing here without that contract is
            # what sent the judge the chain of reasoning instead of the answer:
            # extract_answer_span falls back to the whole response when no marker
            # is supplied, so under cot the judge graded the model's thinking,
            # on 79% of house_md's records and 48-70% of the other four.
            "candidate": (score.prediction or extract_answer_span(response.text, None))[:600],
            "gold": sample.reference["gold"],
            "observation": sample.fields["observation"],
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
                "best_of_n_<metric>":
                "Every metric also gets a best_of_n_ counterpart: per record, the repeat the "
                "judge scored highest. A diagnosis is free text with many correct surface forms, "
                "so a plurality over repeats is not meaningful and Best-of-N replaces it.",
                "diagnosis_judged": "(PRIMARY, higher is better, 0-1) LLM-judge verdict on "
                "whether the stated diagnosis is the same disease entity as the gold, however "
                "written. 1.0 when the judge affirms.",
                "parse_failure_rate": "(lower is better, 0-1) fraction of responses no diagnosis "
                "could be read from; these score 0 and are counted separately from being wrong.",
                "reasoning_recall": "fraction of the clinician's reasoning statements the model's "
                "response covers (token-F1 >= 0.3 per statement). Read per prompt mode: only cot "
                "asks for reasoning, so io scores near zero here by construction.",
            },
            primary_metric="diagnosis_judged",
            decisions=[
                "Took the data from the authors' Hugging Face mirror; the GitHub repository "
                "contains code only.",
                "Scored by an LLM judge against the gold diagnosis rather than by string "
                "comparison: no candidate list is shown, so the answer is written into an open "
                "vocabulary where the same disease has many correct surface forms.",
                "reasoning_recall uses a 0.3 token-F1 threshold per clinician statement -- the "
                "dataset defines no threshold, and this one credits a paraphrase while rejecting "
                "an unrelated sentence.",
                "The task line asks for the diagnosis alone. It used to ask for a brief "
                "justification as well, so that reasoning_recall had something to measure -- but "
                "under io the prompt then forbade the justification it had just requested. "
                "reasoning_recall now measures the reasoning the prompt mode actually elicits, "
                "which makes it a reading of cot rather than a constant of the dataset.",
                "max_tokens=1024 to fit a diagnosis plus reasoning after a long case report.",
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
