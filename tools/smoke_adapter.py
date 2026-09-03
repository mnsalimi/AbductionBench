"""A tiny hand-written abduction set used to smoke-test the engine live.

This is a *diagnostic*, not one of the benchmark datasets: it needs no download,
runs in seconds, and exercises both a generation and a selection task kind
against a real endpoint so the batch route, retry logic, scoring, checkpointing
and reporting can be verified end to end before spending time on real data.

Run it with::

    abench run configs/runs/smoke.yaml
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from abductionbench.core.adapter import DatasetAdapter
from abductionbench.core.metrics import (
    aggregate_mean_metrics,
    contains_match,
    extract_answer_span,
    extract_choice_label,
    token_f1,
)
from abductionbench.core.types import (
    AdapterDocumentation,
    ModelResponse,
    SampleScore,
    SampleSpec,
)

#: (observation, gold explanation, accepted keywords)
GENERATION_ITEMS: list[tuple[str, str, list[str]]] = [
    (
        "The grass in the front yard is wet this morning, but the pavement next to it is dry "
        "and no rain was forecast.",
        "The sprinkler system watered the lawn.",
        ["sprinkler", "irrigation", "watered"],
    ),
    (
        "A patient's bread has gone mouldy in three days although the kitchen is cool, and the "
        "loaf was bought sealed.",
        "The bread was already contaminated with mould spores before it was sealed.",
        ["spore", "contaminat", "mould", "mold"],
    ),
    (
        "Every laptop in the office lost its network connection at 14:02, but the phones on "
        "cellular data kept working.",
        "The office router or wired network failed at 14:02.",
        ["router", "switch", "network", "wifi", "wi-fi", "access point"],
    ),
    (
        "A car that started fine yesterday now makes no sound at all when the key is turned, "
        "and the interior lights do not come on.",
        "The battery is dead or disconnected.",
        ["battery", "terminal", "power"],
    ),
    (
        "The kitchen smells of gas only in the evenings, and only when the oven is off.",
        "A leak in the gas supply becomes noticeable when line pressure rises in the evening.",
        ["leak", "gas line", "supply", "pressure"],
    ),
    (
        "A houseplant on a sunny windowsill has yellow lower leaves and damp soil a week after "
        "the last watering.",
        "The plant has been overwatered and its roots cannot get oxygen.",
        ["overwater", "over-water", "root rot", "too much water", "drainage"],
    ),
]

#: (observation, options, index of the best explanation)
SELECTION_ITEMS: list[tuple[str, list[str], int]] = [
    (
        "The bathroom mirror is fogged and the floor tiles are warm.",
        [
            "Someone has just taken a hot shower.",
            "The window was left open overnight.",
            "The mirror is defective.",
            "The house lost power.",
        ],
        0,
    ),
    (
        "A website returns errors only for users in one country, while the servers report "
        "normal load.",
        [
            "The application code has a syntax error.",
            "A network route or DNS entry serving that region is broken.",
            "The database has run out of disk space.",
            "Every user's browser is out of date.",
        ],
        1,
    ),
    (
        "A cake came out of the oven dense and flat, though the recipe was followed and the "
        "oven was at the right temperature.",
        [
            "The oven door was opened once near the end.",
            "The baking powder was past its expiry date and no longer active.",
            "Too much sugar was used.",
            "The cake tin was too small.",
        ],
        1,
    ),
    (
        "A dog that eats normally has been losing weight for a month and drinks far more water "
        "than before.",
        [
            "The dog is getting more exercise than usual.",
            "The dog dislikes its new food.",
            "The dog has an endocrine disorder such as diabetes.",
            "The water bowl is larger than it used to be.",
        ],
        2,
    ),
]

LABELS = ["A", "B", "C", "D"]


class SmokeAdapter(DatasetAdapter):
    """Six generation items and four selection items, no I/O required."""

    adapter_version = "smoke-1.0"
    primary_metric = "abduction_score"

    def build_samples(self) -> list[SampleSpec]:
        samples: list[SampleSpec] = []
        for index, (observation, gold, keywords) in enumerate(GENERATION_ITEMS):
            samples.append(
                SampleSpec(
                    sample_id=f"gen-{index:02d}",
                    fields={"observation": observation},
                    reference={"gold": gold, "keywords": keywords},
                    task_kind="generation",
                    # Short, everyday abduction: a small budget is enough, but a
                    # reasoning model needs headroom for its hidden CoT.
                    max_tokens=int(self.context.option("gen_max_tokens", 512)),
                    metadata={"kind": "generation"},
                )
            )
        for index, (observation, options, gold_index) in enumerate(SELECTION_ITEMS):
            samples.append(
                SampleSpec(
                    sample_id=f"sel-{index:02d}",
                    fields={
                        "observation": observation,
                        "options": options,
                        "option_labels": LABELS[: len(options)],
                    },
                    reference={"gold_label": LABELS[gold_index]},
                    task_kind="selection",
                    max_tokens=int(self.context.option("sel_max_tokens", 512)),
                    metadata={"kind": "selection"},
                )
            )
        return samples[: self.context.sample_size]

    def score(
        self,
        sample: SampleSpec,
        response: ModelResponse,
        *,
        output_contract: dict[str, Any] | None = None,
    ) -> SampleScore:
        if sample.task_kind == "selection":
            labels = sample.fields["option_labels"]
            chosen = extract_choice_label(response.text, labels, output_contract)
            if chosen is None:
                return SampleScore(
                    metrics={"abduction_score": 0.0, "selection_accuracy": 0.0},
                    prediction=None,
                    parse_ok=False,
                )
            correct = float(chosen == sample.reference["gold_label"])
            return SampleScore(
                metrics={"abduction_score": correct, "selection_accuracy": correct},
                prediction=chosen,
            )

        answer = extract_answer_span(response.text, output_contract)
        if not answer:
            return SampleScore(
                metrics={"abduction_score": 0.0, "keyword_hit": 0.0, "token_f1": 0.0},
                prediction="",
                parse_ok=False,
            )
        keywords = sample.reference["keywords"]
        hit = float(any(contains_match(answer, keyword) for keyword in keywords))
        overlap = token_f1(answer, sample.reference["gold"])
        return SampleScore(
            metrics={"abduction_score": hit, "keyword_hit": hit, "token_f1": overlap},
            prediction=answer[:300],
            details={"matched_keywords": [k for k in keywords if contains_match(answer, k)]},
        )

    def aggregate(self, scores: Sequence[SampleScore]) -> dict[str, float]:
        return aggregate_mean_metrics([score.metrics for score in scores])

    def documentation(self) -> AdapterDocumentation:
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="Engine smoke test",
            domain="Diagnostic (not a benchmark dataset)",
            source_url="hand-written in tools/smoke_adapter.py",
            processing_mode="Generation & Selection",
            split_used="n/a (10 hand-written items)",
            abductive_subset="all items are single-step everyday abduction",
            sampling_procedure="fixed order, no sampling",
            metrics_description={
                "abduction_score": "keyword hit for generation items, label accuracy for selection",
                "keyword_hit": "1 if the answer mentions an accepted cause keyword",
                "token_f1": "token overlap with the reference explanation",
                "selection_accuracy": "1 if the chosen option label is the gold one",
            },
            primary_metric="abduction_score",
            decisions=[
                "Generation items are graded by accepted-keyword match plus token F1 rather "
                "than exact match, since free-form explanations are paraphrases by nature.",
            ],
        )
