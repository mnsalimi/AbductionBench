"""HypoGen: hypothesis generation from a stated research limitation.

Source: https://huggingface.co/datasets/UniverseTBD/hypogen-dr1

Each row distils a paper into a ``bit`` (the conventional approach and its
limitation), a ``flip`` (the paper's hypothesis that overturns it), a one-line
``spark``, and a ``chain_of_reasoning``.

The abductive task: given the **bit**, produce the **flip**.  The ``abstract``,
``title``, ``spark`` and ``chain_of_reasoning`` are withheld -- the abstract and
title name the method, and the spark is the flip in miniature, so showing any of
them would leak the answer.
"""

from __future__ import annotations

from typing import Any

from ..core.adapter import SkippedDataset
from ..core.metrics import extract_answer_span
from ..core.types import AdapterDocumentation, ModelResponse, SampleScore, SampleSpec
from . import _common as C
from ._base import PooledDatasetAdapter, apply_judged_metric, judged_only_score, unparsed_score

REPO_ID = "UniverseTBD/hypogen-dr1"


class HypoGenAdapter(PooledDatasetAdapter):
    """Given the conventional-wisdom 'bit', abduce the paper's 'flip'."""

    adapter_version = "1.0"

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. You are given the conventional "
        "wisdom in a research area (the 'bit'). Propose the 'flip': the hypothesis that "
        "overturns it and would explain the results the paper reports."
    )
    data_delivery_mode = "static"

    answer_format = "one hypothesis"
    answer_constraints = (
        "state one hypothesis, not several",
        "say which condition changes the outcome and how",
        "do not restate the observation",
        "do not use introductory phrases or commentary",
    )
    objective_metrics = False
    selection_cardinality = None
    primary_metric = "flip_judged"

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
            raise SkippedDataset("no HypoGen parquet split found")
        path, split = found
        rows = C.read_parquet_rows(path)
        self.split_used = (
            f"{split} ({len(rows)} items) -- the official test split, which is smaller than the "
            "300-sample target"
        )
        return rows

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        bit = C.normalize_whitespace(item.get("bit"))
        flip = C.normalize_whitespace(item.get("flip"))
        if not bit or not flip:
            return None
        return SampleSpec(
            sample_id=C.stable_id("hypogen", item.get("url") or item.get("paper_id") or index),
            fields={
                "observation": bit,
                "question": (
                    "What alternative approach or hypothesis would overturn this limitation?"
                ),
                "instructions": (
                    "State the hypothesis as a research claim: what to do differently and why "
                    "that removes the limitation. Two or three sentences."
                ),
            },
            reference={"gold": flip, "spark": C.normalize_whitespace(item.get("spark"))},
            task_kind="generation",
            max_tokens=640,
            metadata={
                "venue": item.get("venue"),
                "year": item.get("year"),
                "url": item.get("url"),
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
            metric="flip_judged",
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
            "observation": sample.fields["observation"],
            "criteria": (
                "Correct if the candidate proposes essentially the same idea as the reference "
                "hypothesis, even if less detailed."
            ),
        }

    def apply_judge(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore, verdict: Any
    ) -> SampleScore:
        """The verdict is the score -- and the score of every stratum of it."""
        return apply_judged_metric(score, verdict, "flip_judged")

    def documentation(self) -> AdapterDocumentation:
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="HypoGen",
            domain="Scientific Discovery: Computer Science Hypotheses",
            source_url=f"https://huggingface.co/datasets/{REPO_ID}",
            processing_mode="Generation",
            split_used=self.split_used,
            abductive_subset=(
                "bit -> flip is scientific abduction: infer the hypothesis that resolves a stated "
                "limitation. All answer-revealing fields (title, abstract, spark, "
                "chain_of_reasoning) are withheld from the prompt."
            ),
            sampling_procedure=self.sampling_note(),
            metrics_description={
                "flip_judged": "(PRIMARY, higher is better, 0-1) LLM-judge verdict on whether the hypothesis "
                "states the paper's flip -- the condition that reverses the outcome. 1.0 when the "
                "judge affirms, 0.0 when it does not or when the response could not be parsed. "
                "The dataset score is the mean over repeats x records.",
                "best_of_n_flip_judged": "(higher is better, 0-1) Best-of-N: per record, the repeat the judge scored "
                "highest, then averaged over records. Read off the same modes.repeats samples -- "
                "no extra calls. This is what replaces self-consistency here: a plurality needs "
                "answers that can coincide, and free-text hypotheses do not.",
                "flip_judged_repeat_std": "(lower is better) mean within-record standard deviation of the primary metric "
                "across repeats -- how much the same question's answers varied.",
                "repeat_agreement": "(0-1) fraction of records whose repeats all produced the identical prediction. "
                "Near 0 is expected for free-text generation.",
                "parse_failure_rate": "(lower is better, 0-1) fraction of responses no answer could be extracted from. "
                "These score 0 on the primary metric and are counted here separately, so being "
                "unparseable is distinguishable from being wrong.",
            },
            primary_metric="flip_judged",
            decisions=[
                "Used the official test split even though it has only 50 items, rather than "
                "topping it up from train, so the evaluation stays on held-out data. The "
                "shortfall against 300 samples is reported.",
                "Withheld the title and abstract: both name the proposed method and would make "
                "the task retrieval rather than abduction.",
            ],
            caveats=[
                "These overlap numbers are diagnostics, not the evaluation: "
                "character/n-gram similarity punishes a correct paraphrase and "
                "rewards a wrong sentence that reuses the reference's words, so "
                "this dataset is scored by an LLM judge instead.",
                "Only 50 test items, so this dataset's numbers have wide error bars.",
                "One paper has one recorded flip, but many valid hypotheses may resolve the same "
                "limitation; overlap metrics under-credit alternatives.",
            ],
            statistics=self.base_statistics(),
        )
