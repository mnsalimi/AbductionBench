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
from ..core.metrics import extract_answer_span, rouge_l, token_f1
from ..core.types import AdapterDocumentation, ModelResponse, SampleScore, SampleSpec
from . import _common as C
from ._base import PooledDatasetAdapter, unparsed_score

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
        answer = extract_answer_span(response.text, output_contract)
        if not answer:
            return unparsed_score(["flip_rouge_l", "flip_token_f1"], raw=response.text[:300])
        gold = sample.reference["gold"]
        metrics = {
            "flip_rouge_l": rouge_l(answer, gold)["f"],
            "flip_token_f1": token_f1(answer, gold),
        }
        spark = sample.reference.get("spark")
        if spark:
            metrics["spark_token_f1"] = token_f1(answer, spark)
        return SampleScore(metrics=metrics, prediction=answer[:600], details={"gold": gold[:300]})

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
        metrics = dict(score.metrics)
        metrics["flip_judged"] = 1.0 if getattr(verdict, "positive", False) else 0.0
        return SampleScore(
            metrics=metrics,
            prediction=score.prediction,
            parse_ok=score.parse_ok,
            details={**score.details, "judge_label": getattr(verdict, "label", None)},
        )

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
                "flip_rouge_l": "ROUGE-L F against the paper's flip",
                "flip_token_f1": "token F1 against the flip",
                "spark_token_f1": "token F1 against the one-line spark, a terser view of the same "
                "hypothesis",
                "flip_judged": "(primary) LLM-judge verdict on idea equivalence (only when "
                "engine.judge.enabled)",
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
