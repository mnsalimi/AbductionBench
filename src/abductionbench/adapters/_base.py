"""Reusable adapter bases and scoring helpers for child adapters.

Two things every adapter in the suite needs, factored once:

* :class:`PooledDatasetAdapter` -- the deterministic-draw idiom. A subclass says
  how to *load* raw items and how to turn one into a
  :class:`~abductionbench.core.types.SampleSpec`; the base class handles the
  seeded shuffle, taking the first N, and serving replacements from the same
  ordered pool when the engine has to drop an oversize sample.  Replacements are
  therefore never re-draws of an item already seen.
* scoring helpers -- ``selection_score`` and ``text_match_score`` implement the
  two shapes almost every benchmark here reduces to (pick a label; produce text
  that should name the gold hypothesis), including the "unparseable is not the
  same as wrong" rule.

Dataset-specific logic stays in the dataset's own module.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from collections.abc import Sequence
from typing import Any

from ..core.adapter import DatasetAdapter, SkippedDataset
from ..core.metrics import (
    aggregate_mean_metrics,
    any_exact_match,
    contains_match,
    exact_match,
    extract_answer_span,
    extract_choice_label,
    rouge_l,
    token_f1,
)
from ..core.types import ModelResponse, SampleScore, SampleSpec

logger = logging.getLogger(__name__)

__all__ = [
    "PooledDatasetAdapter",
    "selection_score",
    "text_match_score",
    "unparsed_score",
]


class PooledDatasetAdapter(DatasetAdapter):
    """Adapter over a materialized list of raw items, drawn deterministically."""

    #: Set by :meth:`load_items` implementations for the run documentation.
    split_used: str = ""
    #: Total items available in the chosen split, before sampling.
    split_size: int = 0

    def __init__(self, context: Any):
        super().__init__(context)
        self._pool: list[Any] = []
        self._built: list[str] = []

    # -- subclass hooks ------------------------------------------------- #

    @abstractmethod
    def load_items(self) -> list[Any]:
        """Materialize and return the raw items of the chosen split.

        Implementations should set :attr:`split_used` and may raise
        :class:`~abductionbench.core.adapter.SkippedDataset`.
        """

    @abstractmethod
    def make_sample(self, item: Any, index: int) -> SampleSpec | None:
        """Convert one raw item into a sample, or ``None`` to skip it."""

    # -- lifecycle ------------------------------------------------------ #

    def prepare(self) -> None:
        items = self.load_items()
        if not items:
            raise SkippedDataset("no items found in the chosen split")
        self.split_size = len(items)
        # One seeded shuffle defines both the evaluation set (its prefix) and the
        # replacement order (the rest), so the draw is fully reproducible.
        self._pool = self.ordered_pool(items)

    def build_samples(self) -> list[SampleSpec]:
        samples: list[SampleSpec] = []
        for index, item in enumerate(self._pool):
            if len(samples) >= self.context.sample_size:
                break
            sample = self._safe_make(item, index)
            if sample is not None:
                samples.append(sample)
        self._built = [s.sample_id for s in samples]
        return samples

    def replacement_samples(self, count: int, exclude: set[str]) -> list[SampleSpec]:
        out: list[SampleSpec] = []
        for index, item in enumerate(self._pool):
            if len(out) >= count:
                break
            sample = self._safe_make(item, index)
            if sample is None or sample.sample_id in exclude:
                continue
            out.append(sample)
        return out

    def _safe_make(self, item: Any, index: int) -> SampleSpec | None:
        try:
            return self.make_sample(item, index)
        except Exception as exc:  # noqa: BLE001 - one malformed row must not stop a run
            self.log.warning("skipping item %d (%s): %s", index, type(exc).__name__, exc)
            return None

    # -- default aggregation -------------------------------------------- #

    def aggregate(self, scores: Sequence[SampleScore]) -> dict[str, float]:
        """Mean of every per-sample metric (override for anything else)."""
        return aggregate_mean_metrics([score.metrics for score in scores])

    # -- documentation helpers ------------------------------------------ #

    def base_statistics(self) -> dict[str, Any]:
        return {
            "split_size": self.split_size,
            "requested_sample_size": self.context.sample_size,
            "seed": self.context.seed,
            "adapter_version": self.adapter_version,
        }

    def sampling_note(self) -> str:
        return (
            f"seeded shuffle of the whole split (seed={self.context.seed}, "
            f"salt='{self.dataset_id}'), then the first {self.context.sample_size} items; "
            "oversize items are replaced by the next unused item of the same shuffle"
        )


# --------------------------------------------------------------------------- #
# scoring helpers
# --------------------------------------------------------------------------- #


def unparsed_score(metric_names: Sequence[str], **details: Any) -> SampleScore:
    """A score for a response no prediction could be extracted from.

    Metrics are zeroed *and* ``parse_ok`` is false, so the engine can report
    ``parse_failure_rate`` separately from being wrong.
    """
    return SampleScore(
        metrics={name: 0.0 for name in metric_names},
        prediction=None,
        parse_ok=False,
        details=details,
    )


def selection_score(
    response: ModelResponse,
    *,
    labels: Sequence[str],
    gold_label: str,
    output_contract: dict[str, Any] | None = None,
    metric_name: str = "accuracy",
    extra_metrics: dict[str, float] | None = None,
) -> SampleScore:
    """Score a multiple-choice (hypothesis selection) response."""
    chosen = extract_choice_label(response.text, labels, output_contract)
    if chosen is None:
        score = unparsed_score([metric_name], raw=response.text[:300])
        score.metrics.update(extra_metrics or {})
        return score
    correct = float(str(chosen).strip().upper() == str(gold_label).strip().upper())
    metrics = {metric_name: correct}
    metrics.update(extra_metrics or {})
    return SampleScore(metrics=metrics, prediction=chosen, details={"gold": gold_label})


def text_match_score(
    response: ModelResponse,
    *,
    gold: str,
    accepted: Sequence[str] | None = None,
    output_contract: dict[str, Any] | None = None,
    primary: str = "match",
    extra_metrics: dict[str, float] | None = None,
) -> SampleScore:
    """Score a free-form (hypothesis generation) response against a gold string.

    Emits several complementary views, because a single number cannot describe
    free-form abduction:

    ``match``       1.0 if the answer equals, or contains, an accepted gold string
    ``exact_match`` strict normalized equality with the gold string
    ``token_f1``    bag-of-tokens overlap with the gold string
    ``rouge_l``     longest-common-subsequence F-measure with the gold string

    ``accepted`` lists alternative gold surface forms (abbreviations, synonyms)
    that the dataset itself provides; it never invents them.
    """
    answer = extract_answer_span(response.text, output_contract)
    if not answer:
        return unparsed_score(
            [primary, "exact_match", "token_f1", "rouge_l"], raw=response.text[:300]
        )
    candidates = [gold, *(accepted or [])]
    strict = any_exact_match(answer, candidates)
    loose = max(
        [strict] + [contains_match(answer, candidate) for candidate in candidates if candidate]
    )
    metrics = {
        primary: float(loose),
        "exact_match": float(exact_match(answer, gold)),
        "token_f1": token_f1(answer, gold),
        "rouge_l": rouge_l(answer, gold)["f"],
    }
    metrics.update(extra_metrics or {})
    return SampleScore(
        metrics=metrics,
        prediction=answer[:500],
        details={"gold": str(gold)[:300]},
    )
