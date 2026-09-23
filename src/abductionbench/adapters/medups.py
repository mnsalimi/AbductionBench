"""MedUPS: diagnostic reasoning in uncommon medical cases.

Source: https://huggingface.co/collections/oriel9p/medups
Data:   ``oriel9p/MedUPS_mid_stream`` -- the split the paper evaluates

The collection URL is not itself a dataset repository, so the adapter targets
the collection's constituent datasets by id.  The **mid-stream** subset is what
is used, because it is the benchmark: MedUPSQA is 21,874 mid-stream decision
points, and the paper is explicit that "the accompanying free-text final
diagnosis is not used anywhere in this work" (S3.2).  An earlier version of
this docstring claimed the reverse -- that only a final-diagnosis subset was
used -- while the code already loaded ``MedUPS_mid_stream``.  The code was
right.

**The task is the NEXT CLINICAL STEP, not the diagnosis.**  At decision point
*i* the model sees the accumulated chunks and a question about what comes next,
and must produce what the case report actually did next.  The paper's own
examples (Table 2) span four kinds -- Diagnosis, Management, Workup and
Pathology -- and the release ships no question-type column, so the questions
arrive mixed.  "What will be the next step in management?" and "What will be
the expected histopathological findings?" are not diagnoses and must not be
answered as though they were.

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

    adapter_version = "2.0"

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. You are given an uncommon "
        "published case, delivered as a sequence of clinical steps in the order the "
        "clinicians received them, and a question about what comes next. Answer that "
        "question from what has been disclosed so far."
    )
    # STATIC, NOT SEQUENTIAL. The label was declarative and wrong: this adapter
    # has no turn machinery at all -- no environment, no next_turn, no
    # max_turns -- and `PooledDatasetAdapter.make_sample` builds one
    # independent prompt per item like every static dataset here.
    #
    # What the release actually does is hand the model a MASKED TRAJECTORY in a
    # single prompt. It does not disclose new observations turn by turn, and
    # two samples cut from the same trajectory share no history: the model is
    # not told what it answered about an earlier prefix of the same case. So
    # nothing about the delivery was sequential except the word.
    #
    # The label was not free. `data_delivery_mode in ("interactive",
    # "sequential")` routes a task down `_run_episodes`, so every medups sample
    # was driven as a one-turn episode instead of batched; and the delivery mode
    # gates the raw-payload cap, the oversize policy and the input-token budget,
    # each of which was applying the episode rule to a static prompt.
    data_delivery_mode = "static"

    #: NOT "a single diagnosis". The question decides the shape of its own
    #: answer: a next test, an imaging study, a management step, an expected
    #: histopathological finding. This adapter used to close every prompt with
    #: "Your entire response must be: Answer: <a single diagnosis>" underneath
    #: questions like "What will be the expected radiographic findings to
    #: confirm no recurrence of infection?", and then judge the reply against a
    #: gold that describes radiographs. Three of the paper's four question
    #: kinds are not diagnoses.
    answer_format = "the answer to the question asked"
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
    #: NOT `diagnosis_judged`. What is scored is the next clinical step, which
    #: is a diagnosis in only one of the paper's four question kinds. The name
    #: mattered: a column called `diagnosis_judged` invites a reader to compare
    #: it with the diagnosis accuracy of every other medical dataset here, and
    #: it is not that.
    #:
    #: The paper's own evaluation metric is a BINARY equivalence rate -- "at
    #: evaluation time we use a stricter binary equivalence judge (DeepSeek-Chat)
    #: that scores a prediction correct only when its clinical action matches
    #: the reference" (S4) -- which is what this suite's answer judges do, so
    #: the metric shape matches. (The four-criterion 0-14 rubric in the paper is
    #: the GRPO training reward, not the reported metric.)
    primary_metric = "next_step_judged"

    primary_metric_by_mode = {
        # The two tasks are scored by different things, so each names
        # the metric it actually produces rather than inheriting one.
        "generation": "next_step_judged",
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

    #: The paper's evaluation pool covers positions 1 through 8 along the
    #: trajectory: "a fixed evaluation pool of 500 decision points drawn from
    #: the 2,226-item test split, stratified by the number of context chunks
    #: available at prediction time (1 through 8) so that positions along the
    #: trajectory are covered" (S4).
    _MAX_CONTEXT_POSITION = 8

    @staticmethod
    def _context_position(item: dict[str, Any]) -> int | None:
        """How many chunks the model can see at this decision point.

        The answer is realized in chunk ``answer_chunk_num``, so the context is
        everything before it.
        """
        try:
            return int(item.get("answer_chunk_num")) - 1
        except (TypeError, ValueError):
            return None

    def ordered_pool(self, population: Any, salt: str = "") -> list[Any]:
        """Stratified by position along the trajectory, as the paper's pool is.

        A plain shuffle draws positions in proportion to how common they are,
        and in this split that is a long tail: answers sit in chunk 2 through
        chunk 20-odd. Fifty samples off the top of a shuffle therefore land
        mostly in the middle of the distribution and may contain no early
        decision point at all -- which is the hardest and most interesting
        case, because almost nothing has been revealed yet.

        Positions 1-8 are kept, each position is shuffled independently, and
        the positions are then interleaved, so a prefix of ANY length is
        balanced across the trajectory rather than only the full 500.
        """
        by_position: dict[int, list[Any]] = {}
        for item in population:
            position = self._context_position(item)
            if position is None or not 1 <= position <= self._MAX_CONTEXT_POSITION:
                continue
            by_position.setdefault(position, []).append(item)
        if not by_position:
            # No usable position field: fall back rather than return nothing.
            return super().ordered_pool(population, salt)
        strata = [
            super(MedUPSAdapter, self).ordered_pool(items, salt=f"{salt or self.dataset_id}-p{position}")
            for position, items in sorted(by_position.items())
        ]
        self.positions_used = sorted(by_position)
        interleaved: list[Any] = []
        for row in range(max(len(stratum) for stratum in strata)):
            for stratum in strata:
                if row < len(stratum):
                    interleaved.append(stratum[row])
        return interleaved

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
            metric="next_step_judged",
            output_contract=output_contract,
            details={"gold": str(sample.reference["gold"])[:300]},
        )

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
            # The question, so the judge grades the answer to the question
            # that was asked. Without it the criteria below are being applied
            # to an answer whose subject the judge has to guess.
            "context": f"The question the candidate was answering:\n{sample.fields['question']}",
            "criteria": (
                "The reference is what the case report actually did next. The candidate "
                "is correct if it names the same clinical step -- the same test, imaging "
                "study, management decision, finding or diagnosis -- however it is "
                "written: synonyms, abbreviations, eponyms, brand and generic drug "
                "names, and spelling variants all count, and the reference's extra "
                "narrative detail need not be reproduced. It is incorrect if it names a "
                "different step, or is so vague that it does not identify one. Judge it "
                "against the question that was asked: a question about the next "
                "investigation is not answered by a diagnosis, and a question about "
                "expected findings is not answered by naming the disease."
            ),
        }

    def apply_judge(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore, verdict: Any
    ) -> SampleScore:
        return apply_judged_metric(score, verdict, "next_step_judged")

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
                "THE TASK IS THE NEXT CLINICAL STEP, NOT THE DIAGNOSIS. MedUPSQA is 21,874 "
                "mid-stream decision points, and the paper states that 'the accompanying "
                "free-text final diagnosis is not used anywhere in this work' (S3.2). The "
                "questions span the paper's four kinds -- Diagnosis, Management, Workup, "
                "Pathology -- and the release ships no question-type column, so they arrive "
                "mixed and cannot be filtered to the diagnostic ones. This adapter used to "
                "close every prompt with 'Answer: <a single diagnosis>' regardless of what "
                "was asked; the metric is now `next_step_judged` and the answer format is "
                "the question's own.",
                "THE EVALUATION POOL IS STRATIFIED BY POSITION, as the paper's is: 'a fixed "
                "evaluation pool of 500 decision points drawn from the 2,226-item test "
                "split, stratified by the number of context chunks available at prediction "
                "time (1 through 8)' (S4). Positions 1-8 are kept and interleaved, so a "
                "draw of any size is balanced across the trajectory rather than following "
                "the split's long tail. Decision points past position 8 are excluded, as "
                "upstream excludes them.",
                "THE BINARY JUDGE MATCHES THE PAPER HERE. 'At evaluation time we use a "
                "stricter binary equivalence judge (DeepSeek-Chat) that scores a prediction "
                "correct only when its clinical action matches the reference' (S4). The "
                "four-criterion 0-14 rubric in the paper is the GRPO TRAINING reward, not "
                "the reported metric, so this suite's binary answer judge is the right "
                "shape. The judge model differs (gpt-oss-20b here, DeepSeek-Chat there), "
                "and the paper itself notes judge choice moves scores.",
                "NOT REPRODUCED: the paper reports the mean over 5 resamples of the 500-point "
                "pool (seeds 1001-1005) with a 95% t-based interval, which bounds sampling "
                "variation within the pool. This suite draws one pool at its configured "
                "sample_size, so no such interval is reported.",
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
