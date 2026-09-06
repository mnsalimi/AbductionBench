"""The abstract dataset adapter -- the single seam between core and datasets.

A child adapter is responsible for five things:

1. **Materialize** its dataset (``prepare``): find it on disk under
   ``context.data_dir`` or fetch it, and pick the split (test → validation →
   train) it will evaluate.
2. **Sample** deterministically (``build_samples``): a pseudo-random draw of
   ``context.sample_size`` items, seeded from ``context.seed``, yielding
   :class:`~abductionbench.core.types.SampleSpec` objects carrying the *content*
   of each item.
3. **Prompt** (``system_prompt`` / ``build_messages``): its own wording, in its
   benchmark's terms.  The core has no system prompt to impose and no template
   to bind; ``abductionbench.adapters._prompting`` supplies only the scaffolding
   the execution modes share.  An interactive benchmark also implements its
   environment here (``interactive_start`` / ``interactive_step``).
4. **Score** one response (``score``), using whatever metric that dataset's
   task defines, and **aggregate** those per-sample metrics (``aggregate``).
   Where a mode asks one item as several requests, ``reduce_group`` folds them.
5. **Document** itself (``documentation``), so the run report states which
   split, which abductive subset, which seed and which decisions were made.

The engine does everything else: input-token budgeting and replacement draws,
output budgeting, batch packing, multi-turn episodes, retries, endpoint
recovery, checkpointing, reporting.

Adapters must not import engine internals beyond this module,
:mod:`abductionbench.core.types` and :mod:`abductionbench.core.metrics`.
"""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .metrics import extract_answer_span
from .modes import BOV, MCS, SCS, SELECTION_MODES, SELF_CONSISTENCY, TaskModes
from .types import (
    AdapterDocumentation,
    ChatMessage,
    ModelResponse,
    SampleScore,
    SampleSpec,
)

__all__ = ["AdapterContext", "DatasetAdapter", "deterministic_sample", "replace_sample"]

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
    modes:
        The prompt / selection / delivery modes this task is being run in.  The
        adapter owns its prompts, so it is the adapter -- not the engine -- that
        decides what ``cot`` or ``BOV`` means for its own task.
    """

    dataset_id: str
    data_dir: Path
    modes: TaskModes = field(default_factory=TaskModes)
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
    #: Set by an adapter that never evaluates anything (a dataset declared
    #: unavailable), so checks meant for real datasets can skip it.
    always_skips: bool = False

    #: Which metric heads the report for this dataset.
    primary_metric: str = "accuracy"
    #: Metrics where a *higher* value is better; used only for presentation.
    higher_is_better: bool = True

    #: The dataset's own system prompt.  There is deliberately no default here:
    #: the core must not impose a universal system instruction on every dataset,
    #: because what counts as a good instruction is a property of the task.
    #: Every concrete adapter sets this (or overrides :meth:`system_prompt_for`).
    system_prompt: str = ""

    #: How this dataset's data reaches the model: static | interactive |
    #: sequential.  A property of the benchmark, never a configuration choice.
    data_delivery_mode: str = "static"

    #: Whether this dataset's metrics are objectively verifiable -- scorable
    #: statistically without an LLM judge.  Only these datasets offer the
    #: ``cot`` and ``self-consistency`` prompt modes, because a reasoning mode
    #: can only be credited or penalised when correctness is decidable.
    objective_metrics: bool = False

    #: For selection datasets, how many hypotheses the task admits:
    #:
    #: ``"single"``    exactly one is correct  -> SCS, BOV
    #: ``"multi"``     several are correct, and selecting several is part of the
    #:                 benchmark's semantics -> MCS, BOV (never SCS)
    #: ``"flexible"``  one *or* several are admissible -> SCS, MCS, BOV
    #: ``None``        not a selection dataset -> no selection_mode column value
    selection_cardinality: str | None = None

    #: Which abductive tasks this benchmark poses as *independent* evaluations.
    #: ``("generation", "selection")`` means the two are run as separate tasks
    #: with separate scores, never mixed into one.  A benchmark whose pipeline
    #: is generation *and* selection (the table's ampersand) stays a single
    #: entry here, because splitting a pipeline would not reproduce it.
    hypothesis_modes: tuple[str, ...] = ("generation",)

    #: Dataset options to apply for each hypothesis mode, e.g.
    #: ``{"selection": {"subtask": "cause_selection"}}``.  This is how one
    #: adapter serves both tasks without the engine knowing what a "subtask" is.
    hypothesis_mode_options: dict[str, dict[str, Any]] = {}

    #: What the published dataset table lists in its hypothesis-mode column.
    #: When a mode is run that this does not name, the run log records that the
    #: mode was introduced here and why -- see
    #: :attr:`hypothesis_mode_justification`.
    table_hypothesis_mode: str = "Generation"

    #: The benchmark's own formulation, quoted, that justifies treating an
    #: additional mode as a separate task.  Required whenever
    #: :attr:`hypothesis_modes` goes beyond :attr:`table_hypothesis_mode`.
    hypothesis_mode_justification: str = ""

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
    # prompts -- owned by the child adapter, never by the core
    # ------------------------------------------------------------------ #

    def system_prompt_for(self, sample: SampleSpec) -> str:
        """The system instruction for one sample.

        Defaults to the adapter's :attr:`system_prompt`.  Override when one
        dataset needs different instructions for different item kinds (a
        dataset that both generates and selects, say).
        """
        return self.system_prompt

    @abstractmethod
    def build_messages(self, sample: SampleSpec) -> tuple[list[ChatMessage], dict[str, Any]]:
        """Render one sample into a conversation plus its answer contract.

        This is the seam that item 4 of the specification moves out of the core:
        the engine no longer owns prompt wording, template binding or a shared
        system prompt.  It hands the adapter a sample and the active
        :class:`~abductionbench.core.modes.TaskModes` (on ``context.modes``) and
        takes back the exact messages to send.

        The returned contract is passed to :meth:`score`, so a prompt change and
        the parsing it implies stay together in one file.
        """

    # -- one evaluation item -> the requests the active modes need ------- #

    def expand_for_modes(self, samples: list[SampleSpec]) -> list[SampleSpec]:
        """Turn evaluation items into the requests the active modes require.

        Two modes ask one item as several requests, and both need the pieces to
        find each other again afterwards, which is what ``group_id`` is for:

        * ``BOV``  -- one yes/no request per candidate hypothesis.
        * ``self-consistency`` -- the same request k times, voted on afterwards.

        The two compose: a BOV item under self-consistency is asked
        ``k x len(options)`` times.  Every derived sample keeps the original
        item's ``sample_id`` as its ``group_id``, so the reduction in
        :meth:`reduce_group` puts the item back together.
        """
        modes = self.context.modes
        expanded = samples
        if modes.selection_mode == BOV:
            expanded = [derived for sample in expanded for derived in self._bov_samples(sample)]
        if modes.prompt_mode == SELF_CONSISTENCY:
            expanded = [
                derived for sample in expanded
                for derived in self._vote_samples(sample, modes.votes)
            ]
        return expanded

    def _bov_samples(self, sample: SampleSpec) -> list[SampleSpec]:
        """One request per candidate hypothesis, each answerable yes or no."""
        options = list(sample.fields.get("options") or [])
        if not options:
            return [sample]
        labels = _option_labels_for(sample.fields)
        out: list[SampleSpec] = []
        for position, (label, option) in enumerate(zip(labels, options, strict=False)):
            fields = dict(sample.fields)
            fields["options"] = [option]
            fields["option_labels"] = [label]
            fields["hypothesis"] = option
            out.append(
                replace_sample(
                    sample,
                    sample_id=f"{sample.sample_id}#bov{position}",
                    fields=fields,
                    group_id=sample.group_id or sample.sample_id,
                    metadata={**sample.metadata, "bov_label": label, "bov_index": position,
                              "bov_hypothesis": option},
                )
            )
        return out

    @staticmethod
    def _vote_samples(sample: SampleSpec, votes: int) -> list[SampleSpec]:
        """The same request k times; identical prompts, independent samples."""
        if votes <= 1:
            return [sample]
        return [
            replace_sample(
                sample,
                sample_id=f"{sample.sample_id}#sc{index}",
                group_id=sample.group_id or sample.sample_id,
                metadata={**sample.metadata, "vote_index": index},
            )
            for index in range(votes)
        ]

    # ------------------------------------------------------------------ #
    # interactive delivery -- the environment lives in the adapter
    # ------------------------------------------------------------------ #

    #: How many model turns one episode may take before it is cut off.  Only
    #: read for ``data_delivery_mode`` of ``interactive`` or ``sequential``.
    max_turns: int = 8

    def interactive_start(self, sample: SampleSpec) -> tuple[list[ChatMessage], dict[str, Any]]:
        """Open an episode: the first conversation, and the environment's state.

        An interactive benchmark is not a prompt, it is a loop: the model asks
        for evidence and the benchmark answers from the case, the cluster, the
        simulator.  The engine drives the loop and batches each turn across
        episodes; what a request *means* -- which findings exist, what an
        experiment returns -- is the adapter's, because it is the benchmark's.

        The default opens with the same messages a static run would use, which
        makes a single-turn episode identical to a static one.
        """
        messages, contract = self.build_messages(sample)
        return list(messages), {"contract": contract, "turn": 0}

    def interactive_step(
        self, sample: SampleSpec, state: dict[str, Any], assistant_text: str
    ) -> str | None:
        """The environment's reply to one model turn, or ``None`` to end the episode.

        Returning ``None`` means the model has committed to an answer (or the
        environment has nothing left to give), and the last assistant message is
        what gets scored.
        """
        return None

    # ------------------------------------------------------------------ #
    # mode support -- which of the configured modes this dataset admits
    # ------------------------------------------------------------------ #

    @classmethod
    def supports_modes(cls, modes: TaskModes) -> str | None:
        """``None`` if this dataset can be run in ``modes``, else why not.

        The engine asks before planning a task, so an unsupported combination is
        reported once as a skipped *mode* rather than producing a task whose
        numbers would not mean what the column says.
        """
        if modes.prompt_mode != "io" and not cls.objective_metrics:
            return (
                f"prompt_mode={modes.prompt_mode} applies only to datasets with objectively "
                "verifiable metrics; this one is not scored statistically"
            )
        if modes.selection_mode is None:
            return None
        if cls.selection_cardinality is None:
            return f"selection_mode={modes.selection_mode} but this dataset is not a selection task"
        if modes.selection_mode == SCS and cls.selection_cardinality == "multi":
            return (
                "SCS is not offered: this benchmark's task definition requires selecting every "
                "applicable hypothesis, so collapsing it to one choice would change the task"
            )
        if modes.selection_mode == MCS and cls.selection_cardinality == "single":
            return (
                "MCS is not offered: this benchmark's items have exactly one correct hypothesis"
            )
        if modes.selection_mode not in SELECTION_MODES:
            return f"unknown selection_mode {modes.selection_mode!r}"
        return None

    @classmethod
    def introduced_hypothesis_mode(cls, mode: str) -> str | None:
        """Why ``mode`` is run even though the dataset table does not list it.

        Returns the justification to record in the run log, or ``None`` when the
        table already names this mode.  The specification requires both facts to
        be logged: that the mode was introduced, and the exact formulation from
        the benchmark that makes it a separate task rather than a re-reading of
        the same one.
        """
        if mode.lower() in cls.table_hypothesis_mode.lower():
            return None
        return cls.hypothesis_mode_justification or (
            "introduced without a recorded justification -- this is a bug in the adapter"
        )

    @classmethod
    def selection_modes_offered(cls) -> list[str]:
        """Every selection mode this dataset legitimately admits."""
        if cls.selection_cardinality is None:
            return []
        if cls.selection_cardinality == "multi":
            return [MCS, BOV]
        if cls.selection_cardinality == "single":
            return [SCS, BOV]
        return [SCS, MCS, BOV]

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
    # group reduction (self-consistency votes, BOV per-hypothesis answers)
    # ------------------------------------------------------------------ #

    def reduce_group(
        self,
        members: Sequence[tuple[SampleSpec, ModelResponse, SampleScore]],
    ) -> SampleScore | None:
        """One score for an item that was asked as several requests.

        BOV first: the selected set is every hypothesis whose own yes/no
        question was answered yes, and that set is then scored by the dataset's
        own scorer -- so a BOV run and an MCS run are graded by exactly the same
        code, which is what makes their numbers comparable.

        Then the self-consistency vote: a plurality over the members'
        predictions.  Ties keep the first-seen prediction, which is stable
        because samples are ordered deterministically.
        """
        if not members:
            return None
        modes = self.context.modes
        if modes.selection_mode == BOV:
            return self._reduce_bov(members)
        if modes.prompt_mode == SELF_CONSISTENCY:
            return self._reduce_votes(members)
        return None

    def _reduce_bov(
        self, members: Sequence[tuple[SampleSpec, ModelResponse, SampleScore]]
    ) -> SampleScore | None:
        parent = members[0][0].metadata.get("_parent_sample")
        if parent is None:
            return None
        selected: list[str] = []
        yes_count = 0
        for sample, response, _score in members:
            if _said_yes(response.text):
                selected.append(str(sample.metadata.get("bov_label", "")))
                yes_count += 1
        contract = {
            "answer_prefix": "Answer:",
            "style": "multi_label",
            "labels_from_field": "option_labels",
            "option_labels": _option_labels_for(parent.fields),
        }
        synthetic = ModelResponse(
            sample_id=parent.sample_id,
            model_id=members[0][1].model_id,
            status=members[0][1].status,
            content="Answer: " + (", ".join(label for label in selected if label) or "none"),
        )
        score = self.score(parent, synthetic, output_contract=contract)
        score.details = {
            **(score.details or {}),
            "bov_selected": selected,
            "bov_questions": len(members),
        }
        score.metrics = {
            **score.metrics,
            # How choosy the model was when asked one hypothesis at a time. A
            # model that says yes to everything scores well on recall alone, so
            # this is reported next to the score rather than folded into it.
            "bov_yes_rate": yes_count / len(members),
        }
        return score

    @staticmethod
    def _reduce_votes(
        members: Sequence[tuple[SampleSpec, ModelResponse, SampleScore]]
    ) -> SampleScore | None:
        counts: dict[str, int] = {}
        for _sample, _response, score in members:
            if not score.parse_ok:
                continue
            key = str(score.prediction)
            counts[key] = counts.get(key, 0) + 1
        if not counts:
            # Nothing parsed: keep the first member's (failed) score as the
            # item's score rather than inventing an answer.
            return members[0][2]
        winner = max(counts, key=lambda key: counts[key])
        for _sample, _response, score in members:
            if score.parse_ok and str(score.prediction) == winner:
                chosen = score
                break
        else:  # pragma: no cover - unreachable while counts is non-empty
            chosen = members[0][2]
        chosen.metrics = {
            **chosen.metrics,
            # How often the k samples agreed on the winning answer: 1.0 means the
            # vote was unanimous, 1/k means every sample said something different.
            "self_consistency_agreement": counts[winner] / len(members),
        }
        chosen.details = {**(chosen.details or {}), "votes": counts}
        return chosen

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
    def clamp_max_tokens(value: int, *, low: int = 64, high: int = 32_000) -> int:
        """Clamp an adapter's own complexity estimate into a sane band.

        Retained for adapters that record an estimate, but the engine no longer
        budgets from it: see :attr:`SampleSpec.max_tokens`.
        """
        return max(low, min(int(value), high))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} dataset_id={self.dataset_id!r} v{self.adapter_version}>"


_YES_WORDS = ("yes", "true", "correct", "affirmative")
_NO_WORDS = ("no", "false", "incorrect", "negative")


def _said_yes(text: str) -> bool:
    """Whether a BOV answer is a direct yes.

    The specification is explicit that only a direct "yes" selects a hypothesis,
    so anything hedged, absent or unparseable counts as not selected. The answer
    line is read first; a reply that never produces one is read from its tail,
    which is where a model that ignored the format still puts its verdict.
    """
    if not text:
        return False
    span = extract_answer_span(text, {"answer_prefix": "Answer:"}) or text.strip()[-120:]
    lowered = span.strip().strip("*_`\"' .").lower()
    first = lowered.split()[0].strip(".,;:!") if lowered.split() else ""
    if first in _NO_WORDS:
        return False
    return first in _YES_WORDS


def _option_labels_for(fields: dict[str, Any]) -> list[str]:
    """Labels a selection sample's options are answered with (A, B, C by default)."""
    labels = fields.get("option_labels")
    if labels:
        return [str(label) for label in labels]
    options = fields.get("options") or []
    return [chr(ord("A") + index) for index in range(len(options))]


def replace_sample(sample: SampleSpec, **changes: Any) -> SampleSpec:
    """A copy of ``sample`` with fields replaced (``dataclasses.replace`` for slots)."""
    data = {
        "sample_id": sample.sample_id,
        "fields": dict(sample.fields),
        "reference": sample.reference,
        "task_kind": sample.task_kind,
        "max_tokens": sample.max_tokens,
        "sampling_overrides": dict(sample.sampling_overrides),
        "metadata": dict(sample.metadata),
        "messages_override": sample.messages_override,
        "group_id": sample.group_id,
    }
    data.update(changes)
    derived = SampleSpec(**data)
    # Keep a handle on the item the derived request came from, so a reduction
    # can score the original item rather than a reconstruction of it. The key is
    # underscore-prefixed, which is what keeps it out of the serialized record.
    derived.metadata.setdefault("_parent_sample", sample)
    return derived


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
