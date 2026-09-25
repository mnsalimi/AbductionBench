"""The metrics an interactive or sequential task reports -- and nothing else.

A static task reports whatever its adapter measures, plus the engine's repeat,
vote and strict columns. An episode is a different measurement: the question
is how the model ran the conversation, and the sheet for one reports exactly
these columns.

Interactive (medqdx, med_inquire, ddxplus, vivabench, cloud_opsbench):

``final_answer_accuracy``        the adapter's own final-answer score -- the
                                 LLM judge's verdict where the answer is free
                                 text, an exact match where it is a label
``turns_to_final_output``        turns the episode took, the last being the one
                                 that committed to the answer
``interaction_step_relevance``   the share of the episode's actions that moved
                                 toward the gold answer (the step-relevance
                                 judge); ``None`` when there was no action
                                 before the answer to grade

Sequential (athena_bench) -- computed by the adapter, per turn:

``final_answer_accuracy``            the last turn's prediction, judged
``turns_to_correctness``             first turn whose prediction was judged
                                     correct; ``None`` if none was
``average_sample_accuracy``          mean of the per-turn correctness
``turns_to_final_output``            turns the episode took
``turns_to_first_final_prediction``  first turn whose prediction was already
                                     the final one

Everything else an adapter scores is kept, in ``details["adapter_metrics"]``,
so nothing measured is lost -- it is only no longer a column. Every function
here is idempotent: a record projected on one pass and read back on a resume
projects to the same thing.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .metrics import MISSING_METRIC, _is_nan
from .types import ModelResponse, SampleScore

INTERACTIVE = "interactive"
SEQUENTIAL = "sequential"

EPISODE_METRICS: dict[str, tuple[str, ...]] = {
    INTERACTIVE: (
        "final_answer_accuracy",
        "turns_to_final_output",
        "interaction_step_relevance",
    ),
    SEQUENTIAL: (
        "final_answer_accuracy",
        "turns_to_correctness",
        "average_sample_accuracy",
        "turns_to_final_output",
        "turns_to_first_final_prediction",
    ),
}

#: The one primary every episode task is reported on.
PRIMARY = "final_answer_accuracy"


def is_episode(delivery_mode: str | None) -> bool:
    return delivery_mode in EPISODE_METRICS


def project(
    adapter: Any,
    delivery_mode: str,
    score: SampleScore,
    response: ModelResponse | None = None,
) -> SampleScore:
    """``score`` with only this delivery's metrics left as columns."""
    allowed = EPISODE_METRICS[delivery_mode]
    details = dict(score.details or {})
    merged: dict[str, Any] = {**(details.get("adapter_metrics") or {})}
    merged.update({k: v for k, v in (score.metrics or {}).items() if k not in allowed})
    kept = {k: v for k, v in (score.metrics or {}).items() if k in allowed}

    if delivery_mode == INTERACTIVE:
        source = adapter.final_answer_metric
        if source in merged and not _is_nan(merged[source]):
            kept["final_answer_accuracy"] = float(merged[source])
        elif "final_answer_accuracy" not in kept:
            # No readable answer, or none the scorer could grade: a miss, as
            # the adapter's own column counts it.
            kept["final_answer_accuracy"] = 0.0
        rate = merged.get("interaction_step_relevance_rate")
        kept["interaction_step_relevance"] = (
            float(rate) if rate is not None and not _is_nan(rate) else MISSING_METRIC
        )
    turns = merged.get("interaction_steps")
    if turns is None and response is not None:
        turns = (response.usage or {}).get("turns")
    if turns and "turns_to_final_output" not in kept:
        kept["turns_to_final_output"] = float(turns)

    details["adapter_metrics"] = merged
    return SampleScore(
        metrics=kept,
        prediction=score.prediction,
        parse_ok=score.parse_ok,
        details=details,
    )


def aggregate(delivery_mode: str, scores: Iterable[SampleScore]) -> dict[str, float]:
    """Mean of each episode metric over the samples that have a value for it.

    ``None`` (a sample never correct, an episode with no action to grade) sits
    out of that metric's mean rather than counting as zero.
    """
    buckets: dict[str, list[float]] = {name: [] for name in EPISODE_METRICS[delivery_mode]}
    for score in scores:
        for name in buckets:
            value = (score.metrics or {}).get(name)
            if value is not None and not _is_nan(value):
                buckets[name].append(float(value))
    return {name: sum(v) / len(v) for name, v in buckets.items() if v}
