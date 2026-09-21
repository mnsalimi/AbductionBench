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
from ..core.types import AdapterDocumentation, ModelResponse, SampleScore, SampleSpec
from . import _common as C
from ._base import PooledDatasetAdapter, apply_proxy_score, judged_only_score

REPO_ID = "allenai/UNcommonsense"


def _numbered_references(references: Any) -> str:
    """The item's reference texts, numbered so a judge can name the one it used.

    Numbered rather than bulleted because the judge is asked which it scored
    against, and an index is the only handle that survives into the record.
    """
    items = [str(item).strip() for item in (references or []) if str(item).strip()]
    return "\n".join(f"{index}. {item}" for index, item in enumerate(items, start=1))


class UncommonsenseAdapter(PooledDatasetAdapter):
    """Explain an uncommon outcome; scored against the human explanation set."""

    adapter_version = "1.0"

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. You are given a situation with an "
        "outcome that is surprising given the context. State the explanation that accounts "
        "for it -- it has to make the uncommon outcome likely, not merely possible."
    )
    data_delivery_mode = "static"

    answer_format = "1 to 3 sentences"
    task_requirements = (
        "make the outcome more likely, leaving as little of an information gap as possible",
        "do not restate the context or the outcome",
    )
    objective_metrics = False
    selection_cardinality = None
    #: A PROJECT-SPECIFIC PROXY, not this benchmark's own protocol.
    #: The `proxy_` prefix is load-bearing: these numbers must never be
    #: read as the paper's metric, and the prefix is what a reader sees
    #: first in a sheet.
    judge_template = "proxy_closest_explanation_v1"
    primary_metric = "proxy_closest_explanation_score"

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
                    "background."
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
            metric="proxy_closest_explanation_score",
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
            # Every human explanation for this item, not just the first one
            # stored. They are alternatives -- people explained the same outcome
            # differently -- so scoring against whichever happened to be first
            # penalised a candidate for matching one of the others.
            "references": _numbered_references(sample.reference.get("references")),
            "observation": sample.fields["observation"],
        }

    def apply_judge(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore, verdict: Any
    ) -> SampleScore:
        """The verdict is the score -- and the score of every stratum of it."""
        return apply_proxy_score(score, verdict, "proxy_closest_explanation_score")

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
                "proxy_closest_explanation_score": "(PRIMARY, PROJECT-SPECIFIC PROXY -- not UNcommonsense's own evaluation. higher is better, 0-1) The fraction of records whose generated explanation an LLM judge accepted as SAYING THE SAME THING as the CLOSEST of the item's human-written explanations. The verdict is strictly BINARY, 1 or 0: similar to that gold explanation, or not. There is no partial credit -- close but a different reason scores 0. The item's explanations are alternatives, not a set to cover, so the judge picks the one the candidate comes nearest and records which it used. A sample whose judge call failed is left BLANK, never 0.0: a 0 here means the judge read it and rejected it.",
                "best_of_n_proxy_closest_explanation_score": "(higher is better, 0-1) Best-of-N: per record, the repeat the judge scored "
                "highest, then averaged over records. Read off the same modes.repeats samples -- "
                "no extra calls. This is what replaces self-consistency here: a plurality needs "
                "answers that can coincide, and free-text hypotheses do not.",
                "proxy_closest_explanation_score_repeat_std": "(lower is better) mean within-record standard deviation of the primary metric "
                "across repeats -- how much the same question's answers varied.",
                "repeat_agreement": "(0-1) fraction of records whose repeats all produced the identical prediction. "
                "Near 0 is expected for free-text generation.",
                "parse_failure_rate": "(lower is better, 0-1) fraction of responses no answer could be extracted from. "
                "These score 0 on the primary metric and are counted here separately, so being "
                "unparseable is distinguishable from being wrong.",
            },
            primary_metric="proxy_closest_explanation_score",
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
                "proxy_closest_explanation_score IS NOT THE PAPER'S METRIC. UNcommonsense evaluates with human preference judgements against its own baselines; this is an LLM-judge proxy defined by this suite for comparing runs here, and the two numbers are not interchangeable. It replaced a binary verdict taken against references[0] alone, which marked a candidate wrong for matching any human explanation but the first one stored.",
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
