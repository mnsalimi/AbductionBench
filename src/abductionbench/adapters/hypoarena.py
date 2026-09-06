"""HypoArena: prospective hypothesis discovery across six domains.

Source: https://huggingface.co/datasets/HypoArena/HypoData

830 items spanning biomedical science, machine learning, IT operations, social
science, safety investigation and financial analysis.  Each gives a ``context``
describing what is observed or puzzling, and one or more reference
``hypotheses`` -- each with the ``evidence`` that would test it and, for some
domains, a ``category`` (``causal``, ``latent``, ``intervention``, ...).

The abductive task: read the context and propose the hypothesis.  Where an item
has several reference hypotheses, all of them are acceptable and the closest is
credited.

**No selection mode.** The release contains reference hypotheses only -- there
are no labelled candidate sets or distractors -- so the "Generation & Selection"
mode in the suite's table is covered here as generation only.  Building
distractors would mean inventing data, which this adapter does not do.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..core.adapter import SkippedDataset
from ..core.metrics import aggregate_mean_metrics, extract_answer_span, rouge_l, token_f1
from ..core.types import AdapterDocumentation, ModelResponse, SampleScore, SampleSpec
from . import _common as C
from ._base import PooledDatasetAdapter, unparsed_score

REPO_ID = "HypoArena/HypoData"


class HypoArenaAdapter(PooledDatasetAdapter):
    """Propose a hypothesis for an observed situation; multi-reference scoring."""

    adapter_version = "1.0"

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. You are given an observed "
        "situation. Propose the hypothesis that explains it: a specific, testable claim about "
        "what is going on, not a restatement of the observation."
    )
    data_delivery_mode = "static"
    objective_metrics = False
    selection_cardinality = None
    primary_metric = "best_rouge_l"

    def load_items(self) -> list[dict[str, Any]]:
        root = C.ensure_hf_snapshot(
            REPO_ID,
            self.context.data_dir / "hf",
            offline=self.context.offline,
            allow_patterns=["hypodata.json", "*.md"],
        )
        files = C.find_files(root, ["hypodata.json"]) or C.find_files(root, ["*.json"])
        if not files:
            raise SkippedDataset("hypodata.json not found in the HypoArena release")
        payload = C.read_json(files[0])
        if not isinstance(payload, list):
            raise SkippedDataset("unexpected HypoArena structure (expected a list)")
        domains = self.context.option("domains")
        rows = [
            row
            for row in payload
            if not domains or row.get("domain") in domains
        ]
        self.split_used = (
            f"whole release ({len(rows)} items across "
            f"{len({row.get('domain') for row in rows})} domains); no official split"
        )
        return rows

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        context_text = C.normalize_whitespace(item.get("context"))
        hypotheses = [
            C.normalize_whitespace(entry.get("hypothesis"))
            for entry in item.get("hypotheses") or []
            if isinstance(entry, dict) and C.normalize_whitespace(entry.get("hypothesis"))
        ]
        if not context_text or not hypotheses:
            return None
        domain = str(item.get("domain") or "unknown")
        return SampleSpec(
            sample_id=C.stable_id("hypoarena", item.get("id", index)),
            fields={
                "observation": context_text,
                "question": (
                    "What is the most plausible hypothesis that would account for this situation?"
                ),
                "instructions": (
                    "State one specific, testable hypothesis, and name the evidence that would "
                    "test it. Keep it under five sentences."
                ),
            },
            reference={"gold": hypotheses[0], "references": hypotheses},
            task_kind="generation",
            # Research-grade hypotheses with a testing plan: a real writing budget.
            max_tokens=1024,
            metadata={
                "domain": domain,
                "n_references": len(hypotheses),
                "categories": [
                    entry.get("category")
                    for entry in item.get("hypotheses") or []
                    if isinstance(entry, dict)
                ],
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
        rouges = [rouge_l(answer, reference)["f"] for reference in references]
        f1s = [token_f1(answer, reference) for reference in references]
        domain = sample.metadata.get("domain", "unknown")
        return SampleScore(
            metrics={
                "best_rouge_l": max(rouges),
                "best_token_f1": max(f1s),
                f"best_rouge_l_{domain}": max(rouges),
            },
            prediction=answer[:600],
            details={"n_references": len(references)},
        )

    def aggregate(self, scores: Sequence[SampleScore]) -> dict[str, float]:
        return aggregate_mean_metrics([score.metrics for score in scores])

    def judge_request(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore
    ) -> dict[str, Any] | None:
        if not response.text:
            return None
        return {
            "candidate": score.prediction or response.text[:900],
            "gold": sample.reference["gold"],
            "observation": C.clip_words(sample.fields["observation"], 200),
            "criteria": (
                "Correct if the candidate identifies the same underlying mechanism or cause as "
                "the reference hypothesis."
            ),
        }

    def apply_judge(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore, verdict: Any
    ) -> SampleScore:
        metrics = dict(score.metrics)
        metrics["hypothesis_judged"] = 1.0 if getattr(verdict, "positive", False) else 0.0
        return SampleScore(
            metrics=metrics,
            prediction=score.prediction,
            parse_ok=score.parse_ok,
            details={**score.details, "judge_label": getattr(verdict, "label", None)},
        )

    def documentation(self) -> AdapterDocumentation:
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="HypoArena",
            domain="General: Prospective Hypothesis Discovery",
            source_url=f"https://huggingface.co/datasets/{REPO_ID}",
            processing_mode="Generation (selection not supported by the release)",
            split_used=self.split_used,
            abductive_subset=(
                "All items are hypothesis-discovery tasks. Every domain is kept; "
                "options.domains can restrict them."
            ),
            sampling_procedure=self.sampling_note(),
            metrics_description={
                "best_rouge_l": "ROUGE-L F against the closest reference hypothesis (primary)",
                "best_token_f1": "token F1 against the closest reference hypothesis",
                "best_rouge_l_<domain>": "the same metric restricted to one domain",
                "hypothesis_judged": "LLM-judge verdict on same-mechanism (only when "
                "engine.judge.enabled)",
            },
            primary_metric="best_rouge_l",
            decisions=[
                "No official split exists, so the whole release is the population.",
                "Multi-reference items credit the closest reference; all listed hypotheses are "
                "legitimate answers.",
                "Implemented generation only and reported why: the release ships reference "
                "hypotheses without candidate sets, so a selection mode would require inventing "
                "distractors.",
                "max_tokens=1024, because the task asks for a hypothesis plus the evidence that "
                "would test it.",
            ],
            caveats=[
                "Reference hypotheses are long, domain-specific prose; overlap metrics are a weak "
                "proxy here and the judge stage is strongly recommended.",
            ],
            statistics=self.base_statistics(),
        )
