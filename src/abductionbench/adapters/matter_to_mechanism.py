"""Matter to Mechanism: materials/battery research hypotheses.

Source: https://huggingface.co/datasets/matter2mech/matter-to-mechanism

2,645 rows distilled from battery-materials papers.  Each row states a research
``problem_statement`` (a limitation or gap) and the ``hypothesis`` the paper
proposed to address it, together with structured fields describing the failure
mode, the intervention and the ``mechanism_or_rationale``.

The abductive task: given the problem -- and nothing that reveals the answer --
propose the mechanism-level hypothesis.  Fields that contain the answer
(``hypothesis``, ``intervention_or_solution``, ``mechanism_or_rationale``,
``claimed_outcome``, ``reasoning_process``) are withheld from the prompt; only
``problem_statement``, ``problem_core``, the battery system and the component
are shown, which is the information a researcher would have *before* forming the
hypothesis.
"""

from __future__ import annotations

from typing import Any

from ..core.adapter import SkippedDataset
from ..core.metrics import extract_answer_span
from ..core.types import AdapterDocumentation, ModelResponse, SampleScore, SampleSpec
from . import _common as C
from ._base import PooledDatasetAdapter, apply_judged_metric, judged_only_score, unparsed_score

REPO_ID = "matter2mech/matter-to-mechanism"


class MatterToMechanismAdapter(PooledDatasetAdapter):
    """Propose the mechanism-level hypothesis for a materials research problem."""

    adapter_version = "1.0"

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. You are given a materials-science "
        "research problem and its observations. Propose the mechanism that explains them at "
        "the level of structure and process -- what is happening in the material, not what "
        "should be measured next."
    )
    data_delivery_mode = "static"

    answer_format = "one mechanism"
    answer_constraints = (
        "state one mechanism, not several",
        "name the physical or chemical process responsible",
        "do not restate the observation",
        "do not use introductory phrases or commentary",
    )
    objective_metrics = False
    selection_cardinality = None
    primary_metric = "hypothesis_judged"

    def load_items(self) -> list[dict[str, Any]]:
        root = C.ensure_hf_snapshot(
            REPO_ID,
            self.context.data_dir / "hf",
            offline=self.context.offline,
            allow_patterns=["*.csv", "*.md"],
        )
        files = C.find_files(root, ["*.csv"])
        if not files:
            raise SkippedDataset("no CSV found in the matter-to-mechanism release")
        rows = C.read_csv_rows(files[0])
        self.split_used = (
            f"whole release ({len(rows)} rows in {files[0].name}); the dataset ships a single "
            "unsplit table"
        )
        return rows

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        problem = C.normalize_whitespace(item.get("problem_statement"))
        hypothesis = C.normalize_whitespace(item.get("hypothesis"))
        if not problem or not hypothesis:
            return None
        core = C.normalize_whitespace(item.get("problem_core"))
        system = C.normalize_whitespace(item.get("battery_system"))
        component = C.normalize_whitespace(item.get("component"))
        limitation = C.normalize_whitespace(item.get("failure_mode_or_limitation"))
        context_lines = [
            line
            for line in (
                f"Core bottleneck: {core}" if core else "",
                f"System: {system}" if system and system.lower() != "unknown" else "",
                f"Component: {component}" if component else "",
                f"Known limitation: {limitation}" if limitation else "",
            )
            if line
        ]
        try:
            steps = int(float(item.get("num_reasoning_steps") or 0))
        except (TypeError, ValueError):
            steps = 0
        return SampleSpec(
            sample_id=C.stable_id("m2m", item.get("sample_id", index)),
            fields={
                "observation": problem,
                "context": "\n".join(context_lines),
                "question": (
                    "What mechanism-level hypothesis would explain and address this problem?"
                ),
                "instructions": (
                    "State one hypothesis: the intervention and the mechanism by which it would "
                    "resolve the limitation. Two or three sentences."
                ),
            },
            reference={
                "gold": hypothesis,
                "mechanism": C.normalize_whitespace(item.get("mechanism_or_rationale")),
            },
            task_kind="generation",
            # Adapter-detected complexity: the dataset labels how many reasoning
            # steps the paper's hypothesis took, so longer chains get more budget.
            max_tokens=self.clamp_max_tokens(640 + 128 * max(0, steps - 2), low=640, high=1536),
            metadata={
                "doi": item.get("doi"),
                "problem_type": item.get("problem_type_broad"),
                "num_reasoning_steps": steps,
                "novelty_axis": item.get("novelty_axis"),
            },
        )

    def score(
        self,
        sample: SampleSpec,
        response: ModelResponse,
        *,
        output_contract: dict[str, Any] | None = None,
    ) -> SampleScore:
        """Parse only: the judge is what scores this dataset.

        No overlap metric is emitted. The reference here is one
        acceptable explanation among many, so similarity to it measures
        resemblance to one particular wording rather than correctness.
        """
        return judged_only_score(
            response,
            metric="hypothesis_judged",
            output_contract=output_contract,
            details={},
        )

    def judge_request(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore
    ) -> dict[str, Any] | None:
        if not response.text:
            return None
        return {
            "candidate": score.prediction or response.text[:800],
            "gold": sample.reference["gold"],
            "observation": C.clip_words(sample.fields["observation"], 150),
            "criteria": (
                "Correct if the candidate proposes the same intervention and the same mechanism "
                "as the reference, even if worded differently or less specifically."
            ),
        }

    def apply_judge(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore, verdict: Any
    ) -> SampleScore:
        """The verdict is the score -- and the score of every stratum of it."""
        return apply_judged_metric(score, verdict, "hypothesis_judged")

    def documentation(self) -> AdapterDocumentation:
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="Matter to Mechanism",
            domain="Scientific Discovery: Materials & Batteries",
            source_url=f"https://huggingface.co/datasets/{REPO_ID}",
            processing_mode="Generation",
            split_used=self.split_used,
            abductive_subset=(
                "Every row is a problem/hypothesis pair, i.e. scientific abduction. Only the "
                "problem-side fields are shown; hypothesis, intervention, mechanism, outcome and "
                "reasoning_process are withheld because they contain the answer."
            ),
            sampling_procedure=self.sampling_note(),
            metrics_description={
                "best_of_n_<metric>":
                "Every metric also gets a best_of_n_ counterpart: per record, the repeat the judge "
                "scored highest. This dataset has no checkable answer, so a plurality is meaningless "
                "-- free-text answers never repeat verbatim -- and Best-of-N replaces it.",
                "hypothesis_judged": "(PRIMARY, higher is better, 0-1) LLM-judge verdict on whether the mechanism "
                "names the process actually responsible. 1.0 when the judge affirms, 0.0 when it "
                "does not or when the response could not be parsed. The dataset score is the mean "
                "over repeats x records.",
                "best_of_n_hypothesis_judged": "(higher is better, 0-1) Best-of-N: per record, the repeat the judge scored "
                "highest, then averaged over records. Read off the same modes.repeats samples -- "
                "no extra calls. This is what replaces self-consistency here: a plurality needs "
                "answers that can coincide, and free-text hypotheses do not.",
                "hypothesis_judged_repeat_std": "(lower is better) mean within-record standard deviation of the primary metric "
                "across repeats -- how much the same question's answers varied.",
                "repeat_agreement": "(0-1) fraction of records whose repeats all produced the identical prediction. "
                "Near 0 is expected for free-text generation.",
                "parse_failure_rate": "(lower is better, 0-1) fraction of responses no answer could be extracted from. "
                "These score 0 on the primary metric and are counted here separately, so being "
                "unparseable is distinguishable from being wrong.",
            },
            primary_metric="hypothesis_judged",
            decisions=[
                "The release is a single unsplit table, so the whole table is the population.",
                "Withheld every answer-bearing column from the prompt and listed them explicitly "
                "above, so it is auditable that the task is not trivially solvable.",
                "max_tokens scales with the dataset's own num_reasoning_steps field "
                "(640 + 128 per step beyond two, capped at 1536) rather than one flat value.",
                "Kept a separate mechanism_rouge_l metric because a model can name the right "
                "intervention while missing the mechanism, which is the scientifically "
                "interesting part.",
            ],
            caveats=[
                "These overlap numbers are diagnostics, not the evaluation: "
                "character/n-gram similarity punishes a correct paraphrase and "
                "rewards a wrong sentence that reuses the reference's words, so "
                "this dataset is scored by an LLM judge instead.",
                "Rows are LLM-distilled summaries of papers, so the reference hypothesis wording "
                "is itself model-generated; overlap metrics are relative measures here.",
                "Papers may be in pretraining data; the judge stage does not fix that.",
            ],
            statistics=self.base_statistics(),
        )
