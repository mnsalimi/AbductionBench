"""MedUPS: diagnostic reasoning in uncommon medical cases.

Source: https://huggingface.co/collections/oriel9p/medups
Data:   ``oriel9p/MedUPS_mid_stream`` -- the split the paper evaluates

The collection URL is not itself a dataset repository, so the adapter targets
the collection's constituent datasets by id.  Only the **final-diagnosis**
subset is used: it gives a case presentation and the published final diagnosis,
which is the abductive task.  The mid-stream subset contains generated
follow-up questions of many kinds (risk factors, next investigations, prognosis)
and is not abduction-only, so it is excluded rather than guessed at.

**Fields withheld.** ``cot``, ``final_answer``, ``raw_response`` and
``diagnosis_match`` are outputs of the authors' own model; using them as gold
(or showing them) would measure imitation of that model.  The reference is the
published ``final diagnosis`` field.  The five ``distractor*`` columns are real
retrieved ICD candidates and are used only for the optional selection subtask.
"""

from __future__ import annotations

from typing import Any

from ..core.adapter import SkippedDataset
from ..core.metrics import extract_answer_span
from ..core.types import AdapterDocumentation, ModelResponse, SampleScore, SampleSpec
from . import _common as C
from ._base import PooledDatasetAdapter, text_match_score

REPO_ID = "oriel9p/MedUPS_mid_stream"
COLLECTION_URL = "https://huggingface.co/collections/oriel9p/medups"


class MedUPSAdapter(PooledDatasetAdapter):
    """Diagnose an uncommon published case; free text (default) or 6-way choice."""

    adapter_version = "1.0"

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. You are given an uncommon "
        "published case, delivered as a sequence of clinical steps in the order the "
        "clinicians received them. State the diagnosis that explains the case given "
        "everything disclosed so far."
    )
    data_delivery_mode = "sequential"

    answer_format = "a single diagnosis"
    answer_constraints = (
        "give exactly one diagnosis",
        "output only the diagnosis name",
        "do not explain why",
        "do not use introductory phrases or commentary",
    )
    objective_metrics = True
    selection_cardinality = None
    hypothesis_modes = ("generation",)
    hypothesis_mode_options = {
        "generation": {'subtask': 'generation'},
    }
    table_hypothesis_mode = "Generation"
    hypothesis_mode_justification = (
        "MedUPS is run as GENERATION only, from MedUPS_mid_stream, which is the split the paper "
        "evaluates: the diagnosis is made part-way through the case, before the record is "
        "complete. The final-diagnosis multiple-choice adaptation is not run and no distractors "
        "are added -- both would replace the benchmark's under-specified-diagnosis task with an "
        "easier closed-set one."
    )
    primary_metric = "diagnosis_match"

    primary_metric_by_mode = {
        # The two tasks are scored by different things, so each names
        # the metric it actually produces rather than inheriting one.
        "generation": "diagnosis_match",
        "selection": "accuracy",
    }

    @property
    def _subtask(self) -> str:
        return str(self.context.option("subtask", "generation"))

    def load_items(self) -> list[dict[str, Any]]:
        root = C.ensure_hf_snapshot(
            REPO_ID,
            self.context.data_dir / "hf",
            offline=self.context.offline,
            allow_patterns=["*.jsonl", "*.md"],
        )
        files = C.find_files(root, ["*.jsonl"])
        found = C.pick_split_file(files)
        if not found:
            raise SkippedDataset("no MedUPS mid-stream split file found")
        path, split = found
        rows = C.read_jsonl(path)
        if not rows:
            raise SkippedDataset(f"{path} contained no rows")
        # The release ships its own validity judgements per row, and rows it
        # marks invalid have an unreliable gold answer -- the question was not
        # answerable from the revealed context, or the reference answer did not
        # address it. Scoring against those measures the release's noise.
        usable = [
            row
            for row in rows
            if str(row.get("question_judgment", "")).lower() == "valid"
            and str(row.get("model_response_judgment", "")).lower() == "valid"
        ]
        if not usable:
            raise SkippedDataset(
                f"{path} has {len(rows)} rows but none are marked valid by the release's "
                "own question_judgment/model_response_judgment fields"
            )
        self.dropped_invalid = len(rows) - len(usable)
        self.split_used = (
            f"{split} ({len(usable)} of {len(rows)} questions) of MedUPS_mid_stream"
        )
        return usable

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        """One mid-stream question: the case so far, and what to infer from it.

        This is the paper's task. The case is revealed only up to
        ``answer_chunk_num``, so the diagnosis has to be inferred from an
        incomplete record -- which is the point, and what the final-diagnosis
        subset (a complete case, one published answer) does not test.
        """
        revealed = C.normalize_whitespace(item.get("context_chunks"))
        question = C.normalize_whitespace(item.get("question"))
        # `final_answer` is the concise form of the same answer `answer` gives
        # at length; a diagnosis is what is being scored, not an essay.
        gold = C.normalize_whitespace(item.get("final_answer")) or C.normalize_whitespace(
            item.get("answer")
        )
        if not revealed or not question or not gold:
            return None
        return SampleSpec(
            sample_id=C.stable_id(
                "medups", item.get("case_id", index), item.get("answer_chunk_num", "")
            ),
            fields={
                "observation": revealed,
                "question": question,
            },
            reference={"gold": gold, "long_answer": C.normalize_whitespace(item.get("answer"))},
            task_kind="generation",
            metadata={
                "case_id": item.get("case_id"),
                "revealed_chunks": item.get("context_length"),
                "answer_chunk": item.get("answer_chunk_num"),
                "subtask": "generation",
            },
        )

    def score(
        self,
        sample: SampleSpec,
        response: ModelResponse,
        *,
        output_contract: dict[str, Any] | None = None,
    ) -> SampleScore:
        return text_match_score(
            response,
            gold=sample.reference["gold"],
            output_contract=output_contract,
            primary="diagnosis_match",
        )

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
            name="MedUPS",
            domain="Healthcare: Clinical Decision-Making",
            source_url=COLLECTION_URL,
            processing_mode="Generation (default) / Selection",
            split_used=self.split_used,
            abductive_subset=(
                "The mid-stream subset: the case is revealed only up to a chunk boundary and "
                "the model is asked what follows from what it has seen. Inferring from an "
                "incomplete record is the benchmark's task, and it is what the "
                "final-diagnosis subset (a whole case, one published answer) does not test."
            ),
            sampling_procedure=self.sampling_note(),
            metrics_description={
                "self_consistency_<metric>":
                "Every metric also gets a self_consistency_ counterpart: the plurality answer over "
                "modes.repeats samples of the same record, read off those samples rather than bought "
                "again. Available because this dataset's answers are checkable and so can coincide.",
                "diagnosis_match": "(PRIMARY, higher is better) 1 if the answer equals or contains the published diagnosis "
                "(primary)",
                "exact_match": "strict normalized equality with the published diagnosis",
                "token_f1": "bag-of-tokens F1 against the published diagnosis",
                "rouge_l": "LCS F-measure against the published diagnosis",
                "accuracy": "(PRIMARY, higher is better) selection subtask: 1 if the chosen option is the gold diagnosis",
                "diagnosis_match_judged": "LLM-judge equivalence verdict (only when "
                "engine.judge.enabled)",
            },
            primary_metric="accuracy" if self._subtask == "selection" else "diagnosis_match",
            decisions=[
                "The suite's URL points at a Hugging Face *collection*, which is not a loadable "
                "dataset; resolved it to its member datasets and used MedUPS_mid_stream, which is "
                "the split the paper evaluates.",
                "Scored against `final_answer`, the concise form of the same answer that "
                "`answer` gives at length: a diagnosis is what is being judged, not an essay.",
                "No multiple-choice adaptation and no synthesised distractors: turning this into "
                "a closed-set choice would replace the under-specified-diagnosis task with an "
                "easier one.",
                "Used the official test split of MedUPS_mid_stream.",
                "Withheld and never scored against cot / final_answer / raw_response / "
                "diagnosis_match: these are the authors' own model outputs, not gold data.",
                "The selection subtask uses the five retrieved ICD distractors shipped with each "
                "case; no distractors are invented.",
            ],
            caveats=[
                "NO AGENT PROMPT EXISTS TO ADOPT. MedUPS is published as a Hugging Face "
                "dataset with no agent harness or prompt of its own, so the wording used "
                "to pose the mid-stream question is written here rather than taken from "
                "the authors.",
                "Mid-stream questions are heterogeneous: alongside "
                "what-is-the-diagnosis they ask for risk factors, expected findings and "
                "next investigations. Not every item is abduction in the narrow sense, so "
                "this dataset measures mid-stream clinical inference rather than "
                "diagnosis-from-observation alone.",
                "Rows the release marks invalid by its own question_judgment or "
                "model_response_judgment fields are dropped: their reference answer does "
                "not reliably answer the question asked.",
                "Cases are published reports of uncommon presentations and may be memorized.",
                "Cases can appear more than once with different chunk_number values; the sample "
                "id includes the chunk so duplicates are visible in the records.",
            ],
            statistics={**self.base_statistics(), "subtask": self._subtask},
        )
