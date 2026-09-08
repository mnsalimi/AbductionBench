"""UNcommonsense: abductive explanation of *uncommon* outcomes (Kim et al., 2024).

Source: https://huggingface.co/datasets/allenai/UNcommonsense

Each item pairs a short context with an outcome that is *unexpected* given that
context, plus human-written explanations that make the outcome make sense.  This
is abduction at its most ampliative: the plausible-sounding continuation is
wrong by construction, so the model has to invent a story that reconciles the
two.

**Split.** The release has ``train`` and ``validation`` only, so **validation**
is used and reported.

**References.** ``human_explanations`` are the crowd-written gold explanations
and are used as the reference set.  ``gpt4_explanations`` and
``enhanced_explanations`` are model generations distributed with the dataset and
are *not* used as gold -- scoring against another model's output would measure
imitation rather than abduction.
"""

from __future__ import annotations

from typing import Any

from ..core.adapter import SkippedDataset
from ..core.metrics import extract_answer_span
from ..core.types import AdapterDocumentation, ModelResponse, SampleScore, SampleSpec
from . import _common as C
from ._base import PooledDatasetAdapter, apply_judged_metric, judged_only_score, unparsed_score

REPO_ID = "allenai/UNcommonsense"


class UncommonsenseAdapter(PooledDatasetAdapter):
    """Explain an uncommon outcome; scored against the human explanation set."""

    adapter_version = "1.0"

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. You are given a situation with an "
        "outcome that is surprising given the context. Explain how it could plausibly have "
        "come about -- the explanation has to make the uncommon outcome likely, not merely "
        "possible."
    )
    data_delivery_mode = "static"

    answer_format = "1 to 3 sentences"
    answer_constraints = (
        "write 1 to 3 sentences",
        "make the outcome more likely, leaving as little of an information gap as possible",
        "do not restate the context or the outcome",
        "write only the explanation",
        "do not use introductory phrases or commentary",
    )
    objective_metrics = False
    selection_cardinality = None
    primary_metric = "plausibility_judged"

    def load_items(self) -> list[dict[str, Any]]:
        root = C.ensure_hf_snapshot(
            REPO_ID,
            self.context.data_dir / "hf",
            offline=self.context.offline,
            allow_patterns=["data/*", "*.md"],
        )
        files = C.find_files(root, ["*.parquet"])
        found = C.pick_split_file(files)
        if not found:
            raise SkippedDataset("no UNcommonsense parquet split found")
        path, split = found
        rows = C.read_parquet_rows(path)
        self.split_used = (
            f"{split} ({len(rows)} items); the release has train and validation only"
        )
        return rows

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        context_text = C.normalize_whitespace(item.get("context"))
        outcome = C.normalize_whitespace(item.get("outcome"))
        references = [
            C.normalize_whitespace(text)
            for text in C.as_list(item.get("human_explanations"))
            if C.normalize_whitespace(text)
        ]
        if not context_text or not outcome or not references:
            return None
        return SampleSpec(
            sample_id=C.stable_id("uncs", item.get("source", ""), index),
            fields={
                "context": context_text,
                "observation": (
                    f"Despite the above, this is what happened: {outcome}"
                ),
                "question": "What would make this unexpected outcome make sense?",
                "instructions": (
                    "Give one plausible explanation that reconciles the outcome with the "
                    "background, in a single sentence."
                ),
            },
            reference={"gold": references[0], "references": references},
            task_kind="generation",
            # One-sentence explanations; budget covers reasoning about why the
            # outcome is surprising.
            max_tokens=448,
            metadata={
                "source": item.get("source"),
                "n_references": len(references),
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
            metric="plausibility_judged",
            output_contract=output_contract,
            details={"n_references": len(sample.reference["references"])},
        )

    def judge_request(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore
    ) -> dict[str, Any] | None:
        if not response.text:
            return None
        return {
            "candidate": score.prediction or response.text[:600],
            "gold": sample.reference["gold"],
            "observation": sample.fields["observation"],
            "criteria": (
                "The candidate counts as correct if it makes the unexpected outcome plausible, "
                "even if it differs from the reference explanation."
            ),
        }

    def apply_judge(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore, verdict: Any
    ) -> SampleScore:
        """The verdict is the score -- and the score of every stratum of it."""
        return apply_judged_metric(score, verdict, "plausibility_judged")

    def documentation(self) -> AdapterDocumentation:
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="UNCOMMONSENSE",
            domain="Commonsense: Uncommon-Event Abduction",
            source_url=f"https://huggingface.co/datasets/{REPO_ID}",
            processing_mode="Generation",
            split_used=self.split_used,
            abductive_subset=(
                "The entire dataset is abductive explanation generation for outcomes that are "
                "unlikely given their context."
            ),
            sampling_procedure=self.sampling_note(),
            metrics_description={
                "best_of_n_<metric>":
                "Every metric also gets a best_of_n_ counterpart: per record, the repeat the judge "
                "scored highest. This dataset has no checkable answer, so a plurality is meaningless "
                "-- free-text answers never repeat verbatim -- and Best-of-N replaces it.",
                "plausibility_judged": "(PRIMARY, higher is better, 0-1) LLM-judge verdict on whether the explanation "
                "makes the unexpected outcome plausible given the context. 1.0 when the judge "
                "affirms, 0.0 when it does not or when the response could not be parsed. The "
                "dataset score is the mean over repeats x records.",
                "best_of_n_plausibility_judged": "(higher is better, 0-1) Best-of-N: per record, the repeat the judge scored "
                "highest, then averaged over records. Read off the same modes.repeats samples -- "
                "no extra calls. This is what replaces self-consistency here: a plurality needs "
                "answers that can coincide, and free-text hypotheses do not.",
                "plausibility_judged_repeat_std": "(lower is better) mean within-record standard deviation of the primary metric "
                "across repeats -- how much the same question's answers varied.",
                "repeat_agreement": "(0-1) fraction of records whose repeats all produced the identical prediction. "
                "Near 0 is expected for free-text generation.",
                "parse_failure_rate": "(lower is better, 0-1) fraction of responses no answer could be extracted from. "
                "These score 0 on the primary metric and are counted here separately, so being "
                "unparseable is distinguishable from being wrong.",
            },
            primary_metric="plausibility_judged",
            decisions=[
                "Used the validation split: the release has no test split.",
                "Scored against human_explanations only; the gpt4_explanations and "
                "enhanced_explanations columns are model outputs, and using them as gold would "
                "measure imitation of another model.",
                "Multi-reference scoring takes the best-matching human explanation, since all of "
                "them are acceptable answers.",
                "Framed the prompt as 'despite the above, this happened' so the model sees the "
                "outcome as the surprising observation to be explained.",
            ],
            caveats=[
                "These overlap numbers are diagnostics, not the evaluation: "
                "character/n-gram similarity punishes a correct paraphrase and "
                "rewards a wrong sentence that reuses the reference's words, so "
                "this dataset is scored by an LLM judge instead.",
                "Valid explanations here are open-ended and lexically diverse, so overlap metrics "
                "systematically understate quality; the judge stage is the meaningful measure "
                "and this dataset's overlap scores are best read as relative, not absolute.",
            ],
            statistics=self.base_statistics(),
        )
