"""CommonWhy: entity-grounded causal explanation.

Source: https://github.com/faezemoradik/CommonWhyDataset

Questions of the form *"Why couldn't <entity> have <done X>?"* paired with the
explanation that resolves them ("Because <entity> died before <year>, while X
occurred in <year>") and the general inference rule behind it.  Answering
requires abducing the entity-specific fact that makes the impossibility hold.

The release provides two popularity strata: ``Head.json`` (popular entities) and
``Longtail.json`` (rare ones).  Both are pooled and the stratum is recorded, so
head/long-tail accuracy can be compared -- the distinction the dataset was built
to expose.  The ``rule`` field is **withheld** from the prompt: it states the
general form of the answer.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..core.adapter import SkippedDataset
from ..core.metrics import aggregate_mean_metrics, extract_answer_span, rouge_l, token_f1
from ..core.types import AdapterDocumentation, ModelResponse, SampleScore, SampleSpec
from . import _common as C
from ._base import PooledDatasetAdapter, unparsed_score

REPO_URL = "https://github.com/faezemoradik/CommonWhyDataset"
STRATA = ("Head", "Longtail")


class CommonWhyAdapter(PooledDatasetAdapter):
    """Explain why an entity could not have done something."""

    adapter_version = "1.0"

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. You are told that an entity could "
        "not do something. Explain why not, using what is commonly known about that kind of "
        "entity. The explanation should be the property that actually rules the action out."
    )
    data_delivery_mode = "static"
    objective_metrics = False
    selection_cardinality = None
    primary_metric = "explanation_judged"

    def load_items(self) -> list[dict[str, Any]]:
        root = C.ensure_git_repo(
            REPO_URL, self.context.data_dir / "repo", offline=self.context.offline
        )
        wanted = list(self.context.option("strata", STRATA))
        items: list[dict[str, Any]] = []
        self._per_stratum: dict[str, int] = {}
        for stratum in wanted:
            path = root / "data_files" / f"{stratum}.json"
            if not path.exists():
                self.log.warning("CommonWhy stratum file missing: %s", path)
                continue
            payload = C.read_json(path)
            rows = payload if isinstance(payload, list) else []
            for row in rows:
                items.append({**row, "stratum": stratum})
            self._per_stratum[stratum] = len(rows)
        if not items:
            raise SkippedDataset("no CommonWhy stratum files could be read")
        self.split_used = "no official split; pooled " + ", ".join(
            f"{name} ({count} questions)" for name, count in self._per_stratum.items()
        )
        return items

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        question = C.normalize_whitespace(item.get("question"))
        answer = C.normalize_whitespace(item.get("answer"))
        if not question or not answer:
            return None
        try:
            popularity = float(item.get("popularity") or 0)
        except (TypeError, ValueError):
            popularity = 0.0
        return SampleSpec(
            sample_id=C.stable_id("cwhy", item["stratum"], index),
            fields={
                "observation": question,
                "instructions": (
                    "Explain the impossibility by naming the specific fact about the entity that "
                    "makes it impossible, and how it conflicts with the event. One sentence."
                ),
            },
            reference={"gold": answer, "rule": C.normalize_whitespace(item.get("rule"))},
            task_kind="generation",
            # One-sentence explanation; the budget is reasoning headroom.
            max_tokens=384,
            metadata={"stratum": item["stratum"], "popularity": popularity},
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
            return unparsed_score(
                ["explanation_rouge_l", "explanation_token_f1"], raw=response.text[:300]
            )
        gold = sample.reference["gold"]
        stratum = sample.metadata.get("stratum", "unknown")
        metrics = {
            "explanation_rouge_l": rouge_l(answer, gold)["f"],
            "explanation_token_f1": token_f1(answer, gold),
            f"explanation_rouge_l_{stratum.lower()}": rouge_l(answer, gold)["f"],
        }
        rule = sample.reference.get("rule")
        if rule:
            metrics["rule_token_f1"] = token_f1(answer, rule)
        return SampleScore(metrics=metrics, prediction=answer[:400], details={"gold": gold[:300]})

    def aggregate(self, scores: Sequence[SampleScore]) -> dict[str, float]:
        metrics = aggregate_mean_metrics([score.metrics for score in scores])
        head = metrics.get("explanation_rouge_l_head")
        tail = metrics.get("explanation_rouge_l_longtail")
        if head is not None and tail is not None:
            metrics["popularity_gap"] = head - tail
        return metrics

    def judge_request(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore
    ) -> dict[str, Any] | None:
        if not response.text:
            return None
        return {
            "candidate": score.prediction or response.text[:500],
            "gold": sample.reference["gold"],
            "observation": sample.fields["observation"],
            "criteria": (
                "Correct if the candidate gives the same reason (the same entity fact and the "
                "same conflict) as the reference."
            ),
        }

    def apply_judge(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore, verdict: Any
    ) -> SampleScore:
        metrics = dict(score.metrics)
        metrics["explanation_judged"] = 1.0 if getattr(verdict, "positive", False) else 0.0
        return SampleScore(
            metrics=metrics,
            prediction=score.prediction,
            parse_ok=score.parse_ok,
            details={**score.details, "judge_label": getattr(verdict, "label", None)},
        )

    def documentation(self) -> AdapterDocumentation:
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="CommonWhy",
            domain="Commonsense: Entity-Grounded Causality",
            source_url=REPO_URL,
            processing_mode="Generation",
            split_used=self.split_used,
            abductive_subset=(
                "The grounded question/answer pairs in Head.json and Longtail.json: each requires "
                "abducing the entity fact that makes an event impossible. The repository's "
                "SPARQL/rule-construction files are dataset-building artifacts, not evaluation "
                "items, and are ignored."
            ),
            sampling_procedure=self.sampling_note()
            + "; the two popularity strata are pooled before drawing",
            metrics_description={
                "explanation_rouge_l": "ROUGE-L F against the gold explanation (primary)",
                "explanation_token_f1": "token F1 against the gold explanation",
                "explanation_rouge_l_head/_longtail": "the primary metric per popularity stratum",
                "popularity_gap": "head minus long-tail score -- how much entity popularity helps",
                "rule_token_f1": "token F1 against the general inference rule, i.e. whether the "
                "answer articulates the underlying principle",
                "explanation_judged": "LLM-judge verdict on same-reason (only when "
                "engine.judge.enabled)",
            },
            primary_metric="explanation_rouge_l",
            decisions=[
                "Pooled Head and Longtail and report both separately plus the gap, since the "
                "head/tail contrast is what this dataset is for.",
                "Withheld the `rule` field from the prompt (it is the answer in general form) and "
                "reused it only as a secondary metric.",
                "No official split exists; the pooled files are the population.",
            ],
            caveats=[
                "Gold explanations are templated ('Because X died before Y, while Z occurred in "
                "Y'), so overlap metrics reward matching that template as much as the underlying "
                "reasoning; the judge stage is a better measure of the latter.",
            ],
            statistics={**self.base_statistics(), "questions_per_stratum": self._per_stratum},
        )
