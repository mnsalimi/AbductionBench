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
from ..core.metrics import aggregate_mean_metrics, extract_answer_span
from ..core.types import AdapterDocumentation, ModelResponse, SampleScore, SampleSpec
from . import _common as C
from ._base import PooledDatasetAdapter, apply_judged_metric, judged_only_score, unparsed_score

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

    answer_format = "one short sentence"
    answer_constraints = (
        "write exactly one sentence",
        "give the reason itself",
        "do not restate the observation",
        "do not use introductory phrases or commentary",
    )
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
        """Parse only: the judge is what scores this dataset.

        No overlap metric is emitted. The reference here is one
        acceptable explanation among many, so similarity to it measures
        resemblance to one particular wording rather than correctness.
        """
        stratum = str(sample.metadata.get("stratum", "unknown")).lower()
        extra = {f"explanation_judged_{stratum}": 0.0}
        return judged_only_score(
            response,
            metric="explanation_judged",
            output_contract=output_contract,
            extra_metrics=extra,
            details={},
        )

    def aggregate(self, scores: Sequence[SampleScore]) -> dict[str, float]:
        metrics = aggregate_mean_metrics([score.metrics for score in scores])
        head = metrics.get("explanation_judged_head")
        tail = metrics.get("explanation_judged_longtail")
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
        """The verdict is the score -- and the score of every stratum of it."""
        return apply_judged_metric(score, verdict, "explanation_judged")

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
                "explanation_judged": "(PRIMARY, higher is better, 0-1) LLM-judge verdict on whether the explanation "
                "gives a valid reason for the event. 1.0 when the judge affirms, 0.0 when it does "
                "not or when the response could not be parsed. The dataset score is the mean over "
                "repeats x records.",
                "explanation_judged_<stratum>": "the same verdict restricted to one stratum; identical definition, filtered "
                "population",
                "best_of_n_explanation_judged": "(higher is better, 0-1) Best-of-N: per record, the repeat the judge scored "
                "highest, then averaged over records. Read off the same modes.repeats samples -- "
                "no extra calls. This is what replaces self-consistency here: a plurality needs "
                "answers that can coincide, and free-text hypotheses do not.",
                "explanation_judged_repeat_std": "(lower is better) mean within-record standard deviation of the primary metric "
                "across repeats -- how much the same question's answers varied.",
                "repeat_agreement": "(0-1) fraction of records whose repeats all produced the identical prediction. "
                "Near 0 is expected for free-text generation.",
                "parse_failure_rate": "(lower is better, 0-1) fraction of responses no answer could be extracted from. "
                "These score 0 on the primary metric and are counted here separately, so being "
                "unparseable is distinguishable from being wrong.",
                "popularity_gap": "(closer to 0 is better) explanation_judged_head minus "
                "explanation_judged_longtail: how much better the model does on common events "
                "than on rare ones.",
            },
            primary_metric="explanation_judged",
            decisions=[
                "Pooled Head and Longtail and report both separately plus the gap, since the "
                "head/tail contrast is what this dataset is for.",
                "Withheld the `rule` field from the prompt (it is the answer in general form) and "
                "reused it only as a secondary metric.",
                "No official split exists; the pooled files are the population.",
            ],
            caveats=[
                "These overlap numbers are diagnostics, not the evaluation: "
                "character/n-gram similarity punishes a correct paraphrase and "
                "rewards a wrong sentence that reuses the reference's words, so "
                "this dataset is scored by an LLM judge instead.",
                "Gold explanations are templated ('Because X died before Y, while Z occurred in "
                "Y'), so overlap metrics reward matching that template as much as the underlying "
                "reasoning; the judge stage is a better measure of the latter.",
            ],
            statistics={**self.base_statistics(), "questions_per_stratum": self._per_stratum},
        )
