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
from ..core.metrics import aggregate_mean_metrics, extract_answer_span, rouge_l, token_f1
from ..core.types import AdapterDocumentation, ModelResponse, SampleScore, SampleSpec
from . import _common as C
from ._base import PooledDatasetAdapter, unparsed_score

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
    objective_metrics = False
    selection_cardinality = None
    primary_metric = "hypothesis_rouge_l"

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
        text = response.text
        if not text.strip():
            return unparsed_score(
                ["hypothesis_rouge_l", "hypothesis_token_f1"], raw=text[:300]
            )
        gold = sample.reference["gold"]
        insight = sample.reference.get("insight") or ""
        # The whole response is compared with the whole reference turn, since the
        # task asks for insight + reasoning; the insight line is scored too.
        metrics = {
            "hypothesis_rouge_l": rouge_l(text, gold)["f"],
            "hypothesis_token_f1": token_f1(text, gold),
        }
        if insight:
            answer_line = extract_answer_span(text, output_contract) or text
            metrics["insight_token_f1"] = token_f1(answer_line, insight)
        domain = sample.metadata.get("domain")
        if domain:
            metrics[f"hypothesis_rouge_l_{domain}"] = metrics["hypothesis_rouge_l"]
        return SampleScore(
            metrics=metrics,
            prediction=text[:600],
            details={"gold_insight": insight[:200]},
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
        metrics = dict(score.metrics)
        metrics["insight_judged"] = 1.0 if getattr(verdict, "positive", False) else 0.0
        return SampleScore(
            metrics=metrics,
            prediction=score.prediction,
            parse_ok=score.parse_ok,
            details={**score.details, "judge_label": getattr(verdict, "label", None)},
        )

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
                "hypothesis_rouge_l": "ROUGE-L F between the whole response and the reference "
                "assistant turn (primary)",
                "hypothesis_token_f1": "token F1 against the reference turn",
                "insight_token_f1": "token F1 between the answer line and the reference's 'Core "
                "insight' line -- the hypothesis itself, separate from its reasoning",
                "hypothesis_rouge_l_<domain>": "the primary metric per domain",
                "insight_judged": "LLM-judge verdict on core-insight equivalence (only when "
                "engine.judge.enabled)",
            },
            primary_metric="hypothesis_rouge_l",
            decisions=[
                "Used only the user turn's content as the observation and ignored the dataset's "
                "own system prompt, so prompt wording stays under this framework's configuration.",
                "Scored the full response against the full reference turn (the task asks for "
                "insight plus reasoning) and additionally isolated the 'Core insight' line.",
                "max_tokens=1280, matched to the reference length rather than a flat default.",
            ],
            caveats=[
                "References are extracted from papers with a confidence score (metadata records "
                "explicitness/extraction_confidence); low-confidence items have noisier gold text.",
                "Overlap with the reference rewards mimicking its structure; the judge stage is "
                "the better measure of whether the hypothesis itself matches.",
            ],
            statistics=self.base_statistics(),
        )
