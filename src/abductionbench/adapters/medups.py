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
from ._base import PooledDatasetAdapter, apply_judged_metric, judged_only_score

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
    #: Measured, not assumed: the model writes a disease name into an open
    #: vocabulary with no candidate list, so a correct answer routinely differs
    #: from the gold in wording -- synonym, eponym, abbreviation, subtype -- and
    #: fails a string comparison. The gold exists; its surface form is not the answer.
    objective_metrics = False
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
    primary_metric = "diagnosis_judged"

    primary_metric_by_mode = {
        # The two tasks are scored by different things, so each names
        # the metric it actually produces rather than inheriting one.
        "generation": "diagnosis_judged",
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
        # `answer` is the CASE REPORT's own text answering this row's question;
        # `final_answer` is the authors' MODEL's answer to it. Scoring against
        # the latter -- which this adapter used to do -- graded one model
        # against another model, which is the one thing this file's own
        # decisions list says it does not do.
        #
        # Established from the row schema, not assumed. Every row carries
        # `cot` ("Okay, let's see. The patient has a complex medical
        # history..."), `final_answer`, and `training_format`, which is
        # literally "<think>" + cot + "</think>" + final_answer -- an SFT
        # target assembled from a generation. `raw_response` holds a judge's
        # JSON verdict on that generation, surfaced as this row's own
        # `model_response_judgment`. A field the release ships a judgement
        # *about* is an output, not an answer key.
        #
        # The two also disagree on content. On case 132 the question asks for
        # the most important risk factor; `answer` says the patient's primary
        # immunodeficiency is it, while `final_answer` returns a full
        # mechanistic diagnosis ending in PML -- the eventual diagnosis, which
        # is not what was asked.
        gold = C.normalize_whitespace(item.get("answer"))
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
            # `final_answer` kept for reference only -- never scored against
            # (see above); it is the authors' own model's output, not gold.
            reference={"gold": gold, "final_answer_unscored": C.normalize_whitespace(item.get("final_answer"))},
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
        return judged_only_score(
            response,
            metric="diagnosis_judged",
            output_contract=output_contract,
            details={"gold": str(sample.reference["gold"])[:300]},
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
                "best_of_n_<metric>":
                "Every metric also gets a best_of_n_ counterpart: per record, the repeat the "
                "judge scored highest. A diagnosis is free text with many correct surface forms, "
                "so a plurality over repeats is not meaningful and Best-of-N replaces it.",
                "diagnosis_judged": "(PRIMARY, higher is better, 0-1) LLM-judge verdict on "
                "whether the stated diagnosis is the same disease entity as the published one, "
                "however written. 1.0 when the judge affirms.",
                "parse_failure_rate": "(lower is better, 0-1) fraction of responses no diagnosis "
                "could be read from; these score 0 and are counted separately from being wrong.",
            },
            primary_metric="diagnosis_judged",
            decisions=[
                "The suite's URL points at a Hugging Face *collection*, which is not a loadable "
                "dataset; resolved it to its member datasets and used MedUPS_mid_stream, which is "
                "the split the paper evaluates.",
                "Scored against `answer`, the case report's own text for this row's question "
                "-- NOT `final_answer`, which is the authors' MODEL's answer to it. Every row "
                "pairs `final_answer` with a `cot` trace and assembles the two into "
                "`training_format` as <think>cot</think>final_answer, and ships a "
                "`model_response_judgment` verdict about it: a field the release judges is an "
                "output, not an answer key. A prior version of this adapter scored against it, "
                "grading one model against another and contradicting this file's own decision "
                "to withhold the authors' model outputs.",
                "Scored by an LLM judge against the published diagnosis rather than by string "
                "comparison: no candidate list is shown, so the answer is written into an open "
                "vocabulary where the same disease has many correct surface forms.",
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
                "The gold is the case report's own prose and sometimes carries the article's "
                "figure captions with it (\"Fig. 3 Small bowel series indicated (A) Multiple "
                "smooth-surface round filling defects...\"). The judge is asked whether the "
                "candidate says the same thing, which tolerates that, but the reference is "
                "source text rather than a curated answer and reads like it.",
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
