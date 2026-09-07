"""The execution modes a task can be run in, and the identity they compose.

Four independent axes decide *how* an item is put to the model.  They are
orthogonal on purpose: the same dataset can be evaluated as direct answering or
as chain-of-thought, as a single pick or as one yes/no question per hypothesis,
without either choice changing the other.

``prompt_mode``
    How the answer is elicited.  ``io`` asks for the answer directly; ``cot``
    asks the model to reason first and then answer; ``self-consistency`` samples
    the same question several times at a non-zero temperature and takes the
    majority answer.  Because self-consistency and CoT are only meaningful when
    a wrong answer can be recognised mechanically, they are offered only for
    datasets whose metrics are objective (:attr:`DatasetAdapter.objective_metrics`).

``selection_mode``
    Only for selection tasks.  ``SCS`` asks for exactly one hypothesis, ``MCS``
    for every hypothesis that applies, ``BOV`` asks one yes/no question per
    hypothesis and rebuilds the selected set from the yeses.  A dataset whose
    task definition *requires* several selections never offers ``SCS``.

``data_delivery_mode``
    How the item reaches the model: ``static`` (the whole observation in one
    prompt), ``interactive`` (the model queries an environment over several
    turns), ``sequential`` (the observation arrives in a fixed order of steps).
    This is a property of the dataset, not a choice.

``task_kind``
    The adapter's own label for what the item asks for (``generation``,
    ``selection``, ...).  It varies per sample, so it enters the identity of a
    *record*; a task uses the kind its prompt set is built around.

Orthogonal to all four is ``repeats``: how many times each record is asked.
It is not a mode -- it does not change what is asked or how -- it changes how
many independent observations of the same question the run collects.  Every
repeat is scored on its own and they are averaged, so a score becomes a mean
over ``repeats x records`` observations and the spread between repeats of one
record becomes measurable.  That is what separates it from
``self-consistency``, which asks k times and then *votes*, yielding one answer.

Together they form ``template_mode``, which is the only thing that identifies a
run version -- it replaces the old "which YAML template was bound" identity now
that each adapter owns its own prompts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import ConfigError

__all__ = [
    "PROMPT_MODES",
    "SELECTION_MODES",
    "DATA_DELIVERY_MODES",
    "IO",
    "COT",
    "SELF_CONSISTENCY",
    "SCS",
    "MCS",
    "BOV",
    "HYPOTHESIS_MODES",
    "TaskModes",
    "template_mode_of",
]

IO = "io"
COT = "cot"
SELF_CONSISTENCY = "self-consistency"

SCS = "SCS"
MCS = "MCS"
BOV = "BOV"

STATIC = "static"
INTERACTIVE = "interactive"
SEQUENTIAL = "sequential"

#: Every legal value of the ``prompt_mode`` column.
PROMPT_MODES: tuple[str, ...] = (IO, COT, SELF_CONSISTENCY)
#: Every legal value of the ``selection_mode`` column.
SELECTION_MODES: tuple[str, ...] = (SCS, MCS, BOV)
#: Every legal value of the ``data_delivery_mode`` column.
DATA_DELIVERY_MODES: tuple[str, ...] = (STATIC, INTERACTIVE, SEQUENTIAL)

#: Written in the ``selection_mode`` column of a non-selection task.  A column
#: that is sometimes a mode and sometimes blank is hard to filter on, so
#: generation rows say so explicitly.
NOT_APPLICABLE = "n/a"


def template_mode_of(
    prompt_mode: str,
    selection_mode: str | None,
    task_kind: str,
    data_delivery_mode: str,
) -> str:
    """The run-version identity of one (prompt, selection, kind, delivery) combination.

    Deliberately a flat string rather than a structure: it names a directory, a
    column value and a report row, and it has to survive a round trip through
    JSON, Excel and a filesystem path unchanged.
    """
    return "|".join(
        (
            prompt_mode,
            selection_mode or NOT_APPLICABLE,
            task_kind or "unknown",
            data_delivery_mode,
        )
    )


#: The two abductive tasks a benchmark can pose.  Where the dataset table marks
#: "Generation / Selection (separate tasks)" these are run as two independent
#: evaluations, never as one mixed task.
HYPOTHESIS_MODES: tuple[str, ...] = ("generation", "selection")


@dataclass(frozen=True, slots=True)
class TaskModes:
    """One combination of execution modes, as run for a whole task."""

    prompt_mode: str = IO
    selection_mode: str | None = None
    data_delivery_mode: str = STATIC
    #: Which abductive task is being run: generating a hypothesis, or selecting
    #: among given ones.  ``None`` for a dataset that poses only one.
    hypothesis_mode: str | None = None
    #: How many samples a self-consistency vote draws.  Ignored in other modes.
    self_consistency_n: int = 5
    #: Temperature used for self-consistency sampling; identical samples at
    #: temperature 0 would make the vote meaningless.
    self_consistency_temperature: float = 0.7
    #: How many times each record is asked.  Every repeat is scored
    #: independently and the results are averaged; nothing is voted on.
    #: The run config's default is 3; 1 here so a bare TaskModes is a single
    #: call, which is what every test and preview wants.
    repeats: int = 1
    #: Temperature used once ``repeats > 1``.  Repeating a question at
    #: temperature 0 would return the same answer every time on a server that
    #: is deterministic, which measures nothing.
    repeat_temperature: float = 0.7

    def __post_init__(self) -> None:
        if self.prompt_mode not in PROMPT_MODES:
            raise ConfigError(
                f"unknown prompt_mode {self.prompt_mode!r}; expected one of {list(PROMPT_MODES)}"
            )
        if self.selection_mode is not None and self.selection_mode not in SELECTION_MODES:
            raise ConfigError(
                f"unknown selection_mode {self.selection_mode!r}; "
                f"expected one of {list(SELECTION_MODES)}"
            )
        if self.data_delivery_mode not in DATA_DELIVERY_MODES:
            raise ConfigError(
                f"unknown data_delivery_mode {self.data_delivery_mode!r}; "
                f"expected one of {list(DATA_DELIVERY_MODES)}"
            )
        if self.prompt_mode == SELF_CONSISTENCY and self.self_consistency_n < 2:
            raise ConfigError("self-consistency needs self_consistency_n >= 2 to have a majority")
        if self.repeats < 1:
            raise ConfigError(f"modes.repeats must be at least 1, got {self.repeats}")

    # -- derived identity ------------------------------------------------ #

    def template_mode(self, task_kind: str) -> str:
        return template_mode_of(
            self.prompt_mode, self.selection_mode, task_kind, self.data_delivery_mode
        )

    @property
    def slug(self) -> str:
        """Filesystem-safe form, used as the task's directory name."""
        parts = [self.prompt_mode, self.selection_mode or NOT_APPLICABLE]
        if self.hypothesis_mode:
            parts.append(self.hypothesis_mode)
        parts.append(self.data_delivery_mode)
        return "_".join(part.replace("/", "-") for part in parts)

    @property
    def sampling_temperature(self) -> float | None:
        """Decoding temperature the modes require, or ``None`` for the model's own.

        Both repeats and self-consistency need the answers to be able to differ;
        at temperature 0 a repeat is a copy and a vote is unanimous by
        construction.
        """
        if self.prompt_mode == SELF_CONSISTENCY:
            return self.self_consistency_temperature
        if self.repeats > 1:
            return self.repeat_temperature
        return None

    @property
    def votes(self) -> int:
        """How many samples one question is asked as."""
        return self.self_consistency_n if self.prompt_mode == SELF_CONSISTENCY else 1

    @property
    def needs_group_reduction(self) -> bool:
        """Whether several responses must be folded into one scored answer.

        Self-consistency votes over repeats of one question; BOV rebuilds one
        selected set from one yes/no answer per hypothesis.  Both produce many
        responses per evaluation item and exactly one score.
        """
        return self.prompt_mode == SELF_CONSISTENCY or self.selection_mode == BOV

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt_mode": self.prompt_mode,
            "selection_mode": self.selection_mode or NOT_APPLICABLE,
            "hypothesis_mode": self.hypothesis_mode or NOT_APPLICABLE,
            "data_delivery_mode": self.data_delivery_mode,
            "self_consistency_n": self.votes,
            "repeats": self.repeats,
        }

    def describe(self) -> str:
        parts = [f"prompt_mode={self.prompt_mode}"]
        if self.prompt_mode == SELF_CONSISTENCY:
            parts.append(f"n={self.self_consistency_n}@T={self.self_consistency_temperature}")
        if self.selection_mode:
            parts.append(f"selection_mode={self.selection_mode}")
        if self.hypothesis_mode:
            parts.append(f"hypothesis_mode={self.hypothesis_mode}")
        if self.repeats > 1:
            parts.append(f"repeats={self.repeats}@T={self.repeat_temperature}")
        parts.append(f"delivery={self.data_delivery_mode}")
        return ", ".join(parts)
