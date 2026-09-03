"""The abstract dataset adapter -- the single seam between core and datasets.

A child adapter is responsible for exactly four things:

1. **Materialize** its dataset (``prepare``): find it on disk under
   ``context.data_dir`` or fetch it, and pick the split (test → validation →
   train) it will evaluate.
2. **Sample** deterministically (``build_samples``): a pseudo-random draw of
   ``context.sample_size`` items, seeded from ``context.seed``, yielding
   :class:`~abductionbench.core.types.SampleSpec` objects whose ``fields`` are
   the *content* of the prompt (never its wording -- that is the template's job)
   and whose ``max_tokens`` reflects that item's task complexity.
3. **Score** one response (``score``), using whatever metric that dataset's
   task defines, and **aggregate** those per-sample metrics (``aggregate``).
4. **Document** itself (``documentation``), so the run report states which
   split, which abductive subset, which seed and which decisions were made.

The engine does everything else: prompt rendering from swappable templates,
input-token budgeting and replacement draws, batch packing, retries, endpoint
recovery, checkpointing, reporting.

Adapters must not import engine internals beyond this module,
:mod:`abductionbench.core.types` and :mod:`abductionbench.core.metrics`.
"""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .types import (
    AdapterDocumentation,
    ModelResponse,
    SampleScore,
    SampleSpec,
)

__all__ = ["AdapterContext", "DatasetAdapter", "deterministic_sample"]

logger = logging.getLogger(__name__)


def deterministic_sample(
    population: Sequence[Any],
    size: int,
    seed: int,
    *,
    salt: str = "",
) -> list[Any]:
    """Reproducible pseudo-random subset, order-stable across runs.

    Uses an explicitly seeded :class:`random.Random` (never the global RNG, so
    adapters cannot disturb each other) and returns items in the shuffled order,
    so truncating the result to a smaller size stays a prefix of the same draw.
    ``salt`` lets one adapter draw several independent subsets from one seed.
    """
    rng = random.Random(f"{seed}::{salt}")
    indices = list(range(len(population)))
    rng.shuffle(indices)
    return [population[i] for i in indices[:size]]


@dataclass(slots=True)
class AdapterContext:
    """Everything an adapter is allowed to know about its environment.

    Handed to the adapter by the engine; adapters never read global config.

    Attributes
    ----------
    dataset_id:
        Id from the run config (also the directory name used in outputs).
    data_dir:
        Per-dataset directory under ``engine.data_root`` for materialized data
        and caches.  Created before ``prepare`` is called.
    sample_size:
        How many items to evaluate (``300`` by default, from config).
    seed:
        Determinism seed for this dataset (dataset seed, else run seed).
    options:
        Opaque per-dataset options from config, passed through untouched.
    input_token_budget:
        The input-token ceiling the engine will enforce.  Adapters may use it to
        pre-filter obviously huge items, but enforcement is the engine's job.
    offline:
        When ``True``, adapters must not hit the network; they either use
        already-materialized data or raise
        :class:`~abductionbench.core.errors.AdapterError`.
    """

    dataset_id: str
    data_dir: Path
    sample_size: int = 300
    seed: int = 0
    options: dict[str, Any] = field(default_factory=dict)
    input_token_budget: int = 16000
    offline: bool = False
    cache_dir: Path | None = None
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("adapter"))

    def option(self, key: str, default: Any = None) -> Any:
        """Read an adapter-specific option with a default."""
        return self.options.get(key, default)

    def subdir(self, *parts: str) -> Path:
        """Create and return a subdirectory of the adapter's data dir."""
        path = self.data_dir.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path


class DatasetAdapter(ABC):
    """Base class every child adapter derives from."""

    #: Stable adapter identity, used in logs and reports.
    dataset_id: str = ""
    #: Bump when a change to this adapter invalidates previous records.
    adapter_version: str = "1.0"
    #: Which metric heads the report for this dataset.
    primary_metric: str = "accuracy"
    #: Metrics where a *higher* value is better; used only for presentation.
    higher_is_better: bool = True

    def __init__(self, context: AdapterContext):
        self.context = context
        self.log = context.logger
        if not self.dataset_id:
            self.dataset_id = context.dataset_id

    # ------------------------------------------------------------------ #
    # data lifecycle
    # ------------------------------------------------------------------ #

    def prepare(self) -> None:
        """Materialize the dataset.  Default: nothing to do.

        Called once per task before sampling.  Implementations should be
        idempotent and cache downloads under ``context.data_dir``.
        """
        return None

    @abstractmethod
    def build_samples(self) -> list[SampleSpec]:
        """Return the deterministic evaluation sample for this dataset.

        Must return at most ``context.sample_size`` items.  If the chosen split
        has fewer items, return all of them and record the shortfall in
        :meth:`documentation` (``statistics``), as the run documentation is
        required to report it.
        """

    def replacement_samples(
        self, count: int, exclude: set[str]
    ) -> list[SampleSpec]:
        """Extra items used to replace samples the engine had to drop.

        The engine calls this when a rendered prompt exceeds the configured
        input-token budget and ``engine.limits.on_oversize == "resample"``:
        rather than shrinking the evaluation set, it draws replacements from the
        same split.  ``exclude`` holds every ``sample_id`` already used (whether
        accepted or rejected), so implementations must not return those.

        The default implementation returns nothing, which makes the engine fall
        back to skipping oversize samples.  Adapters whose datasets contain long
        items should override it.
        """
        return []

    # ------------------------------------------------------------------ #
    # scoring
    # ------------------------------------------------------------------ #

    @abstractmethod
    def score(
        self,
        sample: SampleSpec,
        response: ModelResponse,
        *,
        output_contract: dict[str, Any] | None = None,
    ) -> SampleScore:
        """Score one response against ``sample.reference``.

        ``output_contract`` is the active prompt template's declared answer
        format (e.g. ``{"answer_prefix": "Answer:"}``).  Honouring it is what
        lets the same adapter score responses produced by a different prompt
        template without code changes.

        Implementations must be pure and must never raise for a malformed
        response: return ``SampleScore(parse_ok=False, ...)`` with zeroed
        metrics instead, so the engine can distinguish "wrong" from
        "unparseable".
        """

    @abstractmethod
    def aggregate(self, scores: Sequence[SampleScore]) -> dict[str, float]:
        """Reduce per-sample metrics to the dataset's reported metrics.

        Receives only scored samples (errors/skips are accounted separately by
        the engine, which adds ``coverage``, ``parse_failure_rate`` and token /
        latency statistics to whatever this returns).
        """

    # ------------------------------------------------------------------ #
    # optional LLM-judge hooks (used only when engine.judge.enabled)
    # ------------------------------------------------------------------ #

    def judge_request(
        self,
        sample: SampleSpec,
        response: ModelResponse,
        score: SampleScore,
    ) -> dict[str, Any] | None:
        """Fields for the judge template, or ``None`` to skip judging this sample.

        Return the variables the configured judge template needs (e.g.
        ``{"observation": ..., "gold": ..., "candidate": ...}``).  The engine
        renders them with the judge template, batches the calls, parses the
        verdict per that template's ``output_contract``, and hands it back to
        :meth:`apply_judge`.  Default: no judging.
        """
        return None

    def apply_judge(
        self,
        sample: SampleSpec,
        response: ModelResponse,
        score: SampleScore,
        verdict: Any,
    ) -> SampleScore:
        """Fold a judge verdict into a sample's score.

        ``verdict`` is a :class:`~abductionbench.core.judge.JudgeVerdict`; it is
        duck-typed here so this module stays independent of the judge stage.
        Default: return the score unchanged.
        """
        return score

    # ------------------------------------------------------------------ #
    # documentation
    # ------------------------------------------------------------------ #

    @abstractmethod
    def documentation(self) -> AdapterDocumentation:
        """Self-description written into every run's documentation."""

    # ------------------------------------------------------------------ #
    # helpers available to child adapters
    # ------------------------------------------------------------------ #

    def draw(self, population: Sequence[Any], size: int | None = None, salt: str = "") -> list[Any]:
        """Deterministic draw using this adapter's configured seed."""
        return deterministic_sample(
            population, size if size is not None else self.context.sample_size,
            self.context.seed, salt=salt or self.dataset_id,
        )

    def ordered_pool(self, population: Sequence[Any], salt: str = "") -> list[Any]:
        """The full population in deterministic shuffled order.

        Convenient for ``build_samples`` + ``replacement_samples``: take the
        first N as the evaluation set and later items as replacements, so a
        replacement is never a re-draw of an already-seen item.
        """
        return deterministic_sample(population, len(population), self.context.seed,
                                    salt=salt or self.dataset_id)

    @staticmethod
    def clamp_max_tokens(value: int, *, low: int = 64, high: int = 8192) -> int:
        """Clamp an adapter's own complexity estimate into a sane band."""
        return max(low, min(int(value), high))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} dataset_id={self.dataset_id!r} v{self.adapter_version}>"


class SkippedDataset(Exception):
    """Raised by an adapter that deliberately declines to be evaluated.

    Phase 2 requires that a dataset which cannot be obtained, cannot be parsed,
    or whose abductive subset cannot be confidently identified is *skipped and
    reported*, not guessed at.  Raising this from ``prepare`` or
    ``build_samples`` records the dataset as skipped with the given reason and
    lets the rest of the run continue.
    """

    def __init__(self, reason: str, *, dataset_id: str | None = None):
        super().__init__(reason)
        self.reason = reason
        self.dataset_id = dataset_id
