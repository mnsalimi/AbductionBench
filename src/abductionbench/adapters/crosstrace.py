"""CrossTrace: cross-domain scientific hypothesis generation.

Source: https://github.com/andrewbouras/crosstrace

The release is chat-formatted: each row's ``messages`` carry a system prompt, a
user turn describing the field context and the conventional assumption being
challenged, and an assistant turn containing the reference hypothesis with its
reasoning.  ``metadata`` records the source, the domain, the paper id and
extraction-confidence scores.

**How it is adapted.** The user turn's content is used as the observation (so
the prompt wording still comes from this framework's configured template, not
from the dataset's own system prompt), and the assistant turn is the reference.
Only the ``test_ours.jsonl`` split is used; ``test_hypogen.jsonl`` is a subset
re-derived from HypoGen, which this suite already evaluates separately as its
own dataset, and including it would double-count those items.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from ..core.adapter import SkippedDataset
from ..core.metrics import aggregate_mean_metrics, extract_answer_span
from ..core.types import AdapterDocumentation, ModelResponse, SampleScore, SampleSpec
from . import _common as C
from ._base import PooledDatasetAdapter, apply_judged_metric, judged_only_score, unparsed_score

REPO_URL = "https://github.com/andrewbouras/crosstrace"


class CrossTraceAdapter(PooledDatasetAdapter):
    """Generate the hypothesis that challenges a field's conventional assumption."""

    adapter_version = "1.0"

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. You are given a field's "
        "conventional assumption and the observation that strains it. Propose the hypothesis "
        "that would explain the observation while contradicting the assumption -- state the "
        "mechanism, not a call for further research."
    )
    data_delivery_mode = "static"

    answer_format = "one short hypothesis"
    answer_constraints = (
        "write exactly one sentence",
        "name the underlying fault, not its symptoms",
        "do not restate the observation",
        "do not use introductory phrases or commentary",
    )
    objective_metrics = False
    selection_cardinality = None
    primary_metric = "insight_judged"

    def load_items(self) -> list[dict[str, Any]]:
        root = C.ensure_git_repo(
            REPO_URL, self.context.data_dir / "repo", offline=self.context.offline
        )
        preferred = str(self.context.option("split_file", "test_ours.jsonl"))
        path = root / "data" / preferred
        if not path.exists():
            candidates = C.find_files(root / "data", ["test*.jsonl", "val*.jsonl"])
            found = C.pick_split_file(candidates)
            if not found:
                raise SkippedDataset("no CrossTrace test/val split found")
            path = found[0]
        rows = C.read_jsonl(path)
        if not rows:
            raise SkippedDataset(f"{path} contained no rows")
        self.split_used = f"{path.name} ({len(rows)} items); held-out split of the release"
        return rows

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        messages = item.get("messages") or []
        user = next((m.get("content") for m in messages if m.get("role") == "user"), "")
        gold = next((m.get("content") for m in messages if m.get("role") == "assistant"), "")
        user, gold = C.normalize_whitespace(user), C.normalize_whitespace(gold)
        if not user or not gold:
            return None
        metadata = item.get("metadata") or {}
        # The reference opens with "Core insight: <hypothesis>"; keep that line as
        # the headline reference and the whole turn as the full reference.
        insight = ""
        match = re.search(r"core insight\s*:\s*(.+)", gold, re.IGNORECASE)
        if match:
            insight = match.group(1).split("\n")[0].strip()
        return SampleSpec(
            sample_id=C.stable_id("crosstrace", metadata.get("paper_id") or index, index),
            fields={
                "observation": user,
                "question": (
                    "What novel hypothesis would challenge the conventional assumption described "
                    "above?"
                ),
                "instructions": (
                    "State the core insight in one line, then give the reasoning that supports it "
                    "in a few numbered steps."
                ),
            },
            reference={"gold": gold, "insight": insight},
            task_kind="generation",
            # A hypothesis plus multi-step reasoning: the reference itself is
            # ~400 words, so the budget matches the expected output.
            max_tokens=1280,
            metadata={
                "domain": metadata.get("domain"),
                "source": metadata.get("source"),
                "paper_id": metadata.get("paper_id"),
                "explicitness": metadata.get("explicitness"),
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
        stratum = str(sample.metadata.get("domain", "unknown"))
        extra = {f"insight_judged_{stratum}": 0.0}
        return judged_only_score(
            response,
            metric="insight_judged",
            output_contract=output_contract,
            extra_metrics=extra,
            details={},
        )

    def aggregate(self, scores: Sequence[SampleScore]) -> dict[str, float]:
        return aggregate_mean_metrics([score.metrics for score in scores])

    def judge_request(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore
    ) -> dict[str, Any] | None:
        if not response.text:
            return None
        return {
            "candidate": response.text[:900],
            "gold": sample.reference.get("insight") or sample.reference["gold"][:600],
            "observation": C.clip_words(sample.fields["observation"], 200),
            "criteria": (
                "Correct if the candidate's core insight matches the reference hypothesis, even "
                "if the supporting reasoning differs."
            ),
        }

    def apply_judge(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore, verdict: Any
    ) -> SampleScore:
        """The verdict is the score -- and the score of every stratum of it."""
        return apply_judged_metric(score, verdict, "insight_judged")

    def documentation(self) -> AdapterDocumentation:
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="CrossTrace",
            domain="Scientific Discovery: Cross-Domain Hypotheses",
            source_url=REPO_URL,
            processing_mode="Generation",
            split_used=self.split_used,
            abductive_subset=(
                "Every item asks for a hypothesis that explains/overturns a stated assumption. "
                "test_ours.jsonl is used; test_hypogen.jsonl re-derives HypoGen items, which this "
                "suite evaluates as its own dataset, so it is excluded to avoid double counting."
            ),
            sampling_procedure=self.sampling_note(),
            metrics_description={
                "best_of_n_<metric>":
                "Every metric also gets a best_of_n_ counterpart: per record, the repeat the judge "
                "scored highest. This dataset has no checkable answer, so a plurality is meaningless "
                "-- free-text answers never repeat verbatim -- and Best-of-N replaces it.",
                "insight_judged": "(PRIMARY, higher is better, 0-1) LLM-judge verdict on whether the hypothesis "
                "names the underlying fault behind the trace. 1.0 when the judge affirms, 0.0 "
                "when it does not or when the response could not be parsed. The dataset score is "
                "the mean over repeats x records.",
                "insight_judged_<domain>": "the same verdict restricted to one domain; identical definition, filtered "
                "population",
                "best_of_n_insight_judged": "(higher is better, 0-1) Best-of-N: per record, the repeat the judge scored "
                "highest, then averaged over records. Read off the same modes.repeats samples -- "
                "no extra calls. This is what replaces self-consistency here: a plurality needs "
                "answers that can coincide, and free-text hypotheses do not.",
                "insight_judged_repeat_std": "(lower is better) mean within-record standard deviation of the primary metric "
                "across repeats -- how much the same question's answers varied.",
                "repeat_agreement": "(0-1) fraction of records whose repeats all produced the identical prediction. "
                "Near 0 is expected for free-text generation.",
                "parse_failure_rate": "(lower is better, 0-1) fraction of responses no answer could be extracted from. "
                "These score 0 on the primary metric and are counted here separately, so being "
                "unparseable is distinguishable from being wrong.",
            },
            primary_metric="insight_judged",
            decisions=[
                "Used only the user turn's content as the observation and ignored the dataset's "
                "own system prompt, so prompt wording stays under this framework's configuration.",
                "Scored the full response against the full reference turn (the task asks for "
                "insight plus reasoning) and additionally isolated the 'Core insight' line.",
                "max_tokens=1280, matched to the reference length rather than a flat default.",
            ],
            caveats=[
                "These overlap numbers are diagnostics, not the evaluation: "
                "character/n-gram similarity punishes a correct paraphrase and "
                "rewards a wrong sentence that reuses the reference's words, so "
                "this dataset is scored by an LLM judge instead.",
                "References are extracted from papers with a confidence score (metadata records "
                "explicitness/extraction_confidence); low-confidence items have noisier gold text.",
                "Overlap with the reference rewards mimicking its structure; the judge stage is "
                "the better measure of whether the hypothesis itself matches.",
            ],
            statistics=self.base_statistics(),
        )
