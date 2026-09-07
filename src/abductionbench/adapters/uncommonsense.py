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
from ..core.metrics import extract_answer_span, rouge_l, token_f1
from ..core.types import AdapterDocumentation, ModelResponse, SampleScore, SampleSpec
from . import _common as C
from ._base import PooledDatasetAdapter, unparsed_score

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
        answer = extract_answer_span(response.text, output_contract)
        if not answer:
            return unparsed_score(["best_rouge_l", "best_token_f1"], raw=response.text[:300])
        references = sample.reference["references"]
        # Multi-reference scoring: credit the closest human explanation, which is
        # the standard treatment when several gold answers are all acceptable.
        rouges = [rouge_l(answer, reference)["f"] for reference in references]
        f1s = [token_f1(answer, reference) for reference in references]
        best_index = max(range(len(rouges)), key=lambda i: rouges[i])
        return SampleScore(
            metrics={
                "best_rouge_l": max(rouges),
                "best_token_f1": max(f1s),
                "mean_rouge_l": sum(rouges) / len(rouges),
            },
            prediction=answer[:500],
            details={"closest_reference": references[best_index][:300]},
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
        metrics = dict(score.metrics)
        metrics["plausibility_judged"] = 1.0 if getattr(verdict, "positive", False) else 0.0
        return SampleScore(
            metrics=metrics,
            prediction=score.prediction,
            parse_ok=score.parse_ok,
            details={**score.details, "judge_label": getattr(verdict, "label", None)},
        )

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
                "best_rouge_l": "ROUGE-L F against the closest human explanation (primary)",
                "best_token_f1": "token F1 against the closest human explanation",
                "mean_rouge_l": "mean ROUGE-L over all human explanations, a stricter view",
                "plausibility_judged": "LLM-judge verdict on whether the explanation makes the "
                "outcome plausible (only when engine.judge.enabled)",
            },
            primary_metric="best_rouge_l",
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
                "Valid explanations here are open-ended and lexically diverse, so overlap metrics "
                "systematically understate quality; the judge stage is the meaningful measure "
                "and this dataset's overlap scores are best read as relative, not absolute.",
            ],
            statistics=self.base_statistics(),
        )
