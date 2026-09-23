"""LLM-judged structural metrics for chains of reasoning.

What this stage measures is *how* a model got to its answer, not whether the
answer was right -- that is :mod:`abductionbench.core.judge`, a separate stage
with its own prompts and its own cache.  Eight metric families are computed:
observation coverage, step structure and backtracking, branchiness and
diversity, redundancy and completeness, directionality, differential
elimination, uncertainty marking, and use of prior knowledge.

Three rules shape the whole module.

**Only chains.**  The stage runs for ``cot`` and ``self-consistency`` outputs
and for nothing else.  An ``io`` output has no chain of reasoning in it, so
there is nothing here to measure and no row of these columns is written for one.

**One judge call per metric family, and the grouping is the measurement's.**
Metric 2's four counts -- total, useful, useless and backtracking steps -- must
come from one segmentation of one chain, so they are one prompt returning four
numbers rather than four prompts that would each re-segment it differently.

**The judge is never told about normalization.**  Every prompt asks for raw
counts only; every ratio, fraction and rate in the sheet is computed here, in
:func:`derive_reasoning_metrics`, once all the raw values for that sample are
in.  A judge asked for a ratio would have to do arithmetic on its own counts,
and the sheet could then disagree with its own columns.

Values that cannot be computed are omitted rather than defaulted.  A blank cell
always has an explanation next to it, in ``reasoning_metrics_status``,
``reasoning_metrics_inapplicable`` or ``reasoning_judge_errors``: a metric that
genuinely does not apply to a task must never look like a metric that failed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import orjson

from .adapter import DatasetAdapter
from .batching import iter_chunks
from .client import ModelClient
from .config import ReasoningJudgeConfig
from .errors import ConfigError, EndpointError
from .judge import clip_middle, exceeds_budget
from .modes import BOV, COT, SELF_CONSISTENCY
from .prompts import PromptRegistry, PromptRenderer, PromptTemplate
from .retry import RetryPolicy, with_retry
from .types import (
    ModelResponse,
    RenderedPrompt,
    SampleScore,
    SampleSpec,
    SamplingParams,
    TaskIdentity,
    stable_hash,
)

logger = logging.getLogger(__name__)

#: Characters per token, for sizing a budget rather than for describing prose.
#:
#: Four is the usual English approximation and it is wrong for this suite,
#: which is not prose: measured over one run's own judge prompts the ratio is
#: 2.89, and JudgeConfig.max_prompt_chars records 2.72 at its densest (abd's
#: s-expressions) against a 4.57 median. Formal logic, LaTeX and s-expressions
#: tokenize far finer than English -- and those are exactly the chains long
#: enough for the budget to matter.
#:
#: 2.72, the densest measured, rather than 2.89 or the median. The two errors
#: are not symmetric: too low a ratio reserves output tokens that are never
#: generated and so never billed, while too high a ratio truncates the reply
#: and loses the call outright.
_CHARS_PER_TOKEN = 2.72

__all__ = [
    "ReasoningJudgeStage", "derive_reasoning_metrics",
    "REASONING_METRIC_COLUMNS", "REASONING_LIST_COLUMNS",
]

#: Adapters label an item with their own task kind.  Everything that asks for a
#: free-form answer is generation-shaped for these metrics, and everything that
#: picks from supplied options is selection-shaped: ``knowledge_completion`` and
#: ``multi_selection`` are not special cases, they are other names for the same
#: two shapes, and eight datasets use them.
_GENERATION_KINDS = frozenset({"generation", "knowledge_completion"})
_SELECTION_KINDS = frozenset({"selection", "multi_selection"})

_REQUIRED_TEMPLATES = {
    # wave one
    "steps",
    "observation_inventory",
    "option_count",
    # wave two
    "observation_coverage",
    "branchiness_selection",
    "branchiness_generation",
    "directionality",
    "step_directionality",
    "differential_elimination",
    "uncertainty",
    "prior_knowledge",
    "anchoring_point",
    "unresolved_contradiction",
    # interaction, bought separately for interactive deliveries
    "step_relevance",
}

#: Per-step outputs, stored whole as JSON arrays on the sample row.
#:
#: One column per list, never one column per step: step counts vary from sample
#: to sample, so exploding them would give a sheet whose width depends on its
#: longest chain and whose columns mean different things in different rows.
#: Kept raw so they can be re-aggregated or plotted as histograms later, which
#: an average alone cannot support.
#:
#: ``reasoning_steps`` is the segmentation every other list is indexed by, so
#: its length is the authoritative step count. ``reasoning_total_steps`` reports
#: that length as a number, which is what makes an average step count possible
#: in a sheet of aggregates -- a column of JSON arrays cannot be averaged. It is
#: written from ``len(steps)`` and never from anything the judge said, so the
#: two cannot drift apart.
REASONING_LIST_COLUMNS: tuple[str, ...] = (
    "reasoning_steps",
    # The wave-one inventory, kept whole: coverage is a ratio against it, and a
    # ratio nobody can see the denominator of is not auditable.
    "reasoning_observations",
    "reasoning_proof_disproof_per_step",
    "reasoning_observations_per_step",
    "reasoning_branchiness_per_step",
    "reasoning_step_directionality_per_step",
    "reasoning_comparisons_per_step",
    "reasoning_uncertainty_per_step",
    "reasoning_prior_knowledge_per_step",
    "reasoning_unresolved_per_step",
    # One verdict per ACTION of an interactive episode, not per reasoning step.
    "interaction_step_relevance_per_step",
)

REASONING_METRIC_COLUMNS: tuple[str, ...] = (
    # 1. observation (evidence) coverage
    "reasoning_observations_total",
    "reasoning_observations_used",
    "reasoning_observation_coverage",
    # 2. reasoning steps & backtracking
    "reasoning_total_steps",
    "reasoning_useful_steps",
    "reasoning_useless_steps",
    "reasoning_backtracking_steps",
    "reasoning_useful_step_fraction",
    "reasoning_useless_step_fraction",
    "reasoning_backtracking_rate",
    # 3. branchiness & diversity
    "reasoning_branchiness_total",
    "reasoning_diversity",
    "reasoning_option_count",
    # 4. directionality (whole chain)
    "reasoning_directionality",
    # 5. step-level directionality
    "reasoning_step_directionality_mean",
    # 6. differential elimination
    "reasoning_differential_elimination",
    "reasoning_differential_elimination_normalized",
    # 7. comparison exhaustiveness
    "reasoning_comparison_exhaustiveness",
    # 8. uncertainty marking
    "reasoning_uncertainty_steps",
    "reasoning_uncertainty_rate",
    # 9. prior knowledge
    "reasoning_prior_knowledge",
    "reasoning_prior_knowledge_normalized",
    # 10. anchoring point
    "reasoning_anchoring_point",
    "reasoning_anchoring_point_normalized",
    # 11. unresolved contradiction
    "reasoning_unresolved_contradictions",
    "reasoning_unresolved_contradiction_normalized",
)


@dataclass(slots=True)
class _Target:
    """One model output that has a chain of reasoning to measure."""

    index: int
    sample: SampleSpec
    response: ModelResponse
    score: SampleScore
    question: str
    reasoning: str
    answer: str
    reference: str
    options: list[str]
    generation_like: bool
    selection_like: bool
    pipeline_like: bool
    bov: bool
    raw: dict[str, dict[str, Any]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    inapplicable: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# parsing helpers: a judge's number is only a number if it really is one
# --------------------------------------------------------------------------- #


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, float) and value >= 0 and value.is_integer():
        return int(value)
    return None


def _nonnegative_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number >= 0 and math.isfinite(number) else None


def _binary(value: Any) -> int | None:
    parsed = _nonnegative_int(value)
    return parsed if parsed in (0, 1) else None




def comparison_pairs(option_count: int) -> int:
    """C(n, 2) -- the unordered pairs that could be compared out of ``n``.

    The denominator for comparison exhaustiveness. Pairs, not all subsets: a
    chain that weighs every pair once has compared exhaustively, and counting
    the 2^n - n - 1 larger groupings too would make a complete comparison score
    far below 1.
    """
    if option_count < 2:
        return 0
    return option_count * (option_count - 1) // 2


def _int_list(value: Any, expected: int | None = None) -> list[int] | None:
    """A list of non-negative integers, optionally of an exact length."""
    if not isinstance(value, list):
        return None
    out: list[int] = []
    for item in value:
        parsed = _nonnegative_int(item)
        if parsed is None:
            return None
        out.append(parsed)
    if expected is not None and len(out) != expected:
        return None
    return out


def _binary_list(value: Any, expected: int | None = None) -> list[int] | None:
    parsed = _int_list(value, expected)
    if parsed is None or any(item not in (0, 1) for item in parsed):
        return None
    return parsed


def _directionality_list(value: Any, expected: int | None = None) -> list[float] | None:
    if not isinstance(value, list):
        return None
    out: list[float] = []
    for item in value:
        number = _nonnegative_number(item)
        if number is None or number not in (0.0, 0.5, 1.0):
            return None
        out.append(number)
    if expected is not None and len(out) != expected:
        return None
    return out


#: The declared ``json_fields`` types, mapped to the validator that decides
#: them. Two spellings each for the binary and directionality lists because
#: both are in use across the shipped templates; they mean the same thing, and
#: an unknown name is treated as unconstrained rather than as a failure, so a
#: new template cannot be silently rejected by a name this map has not learnt.
_CONTRACT_VALIDATORS: dict[str, Any] = {
    "list_of_strings": lambda value: _string_list(value) is not None,
    "list_of_nonnegative_integers": lambda value: _int_list(value) is not None,
    "list_of_binary": lambda value: _binary_list(value) is not None,
    "list_of_binary_values": lambda value: _binary_list(value) is not None,
    "list_of_directionality_values": lambda value: _directionality_list(value) is not None,
    "nonnegative_integer": lambda value: _nonnegative_int(value) is not None,
    "nonnegative_integer_or_null": lambda value: value is None
    or _nonnegative_int(value) is not None,
    "binary": lambda value: _binary(value) is not None,
    "directionality": lambda value: _nonnegative_number(value) in (0.0, 0.5, 1.0),
}


def _satisfies_contract(value: Any, declared: str) -> bool | None:
    """``True``/``False`` against a known declared type, ``None`` if unknown.

    ``None`` rather than ``False`` for a type this map has not learnt: an
    unrecognised declaration is a gap in this map, not evidence about the
    judge's reply, and failing closed would silently discard every verdict
    from a newly added template.
    """
    validator = _CONTRACT_VALIDATORS.get(declared)
    if validator is None:
        return None
    return bool(validator(value))


def _proportion_of_steps(values: Sequence[float], total_steps: int) -> float | None:
    """What fraction of the steps exhibit the property at all.

    Binarize, then mean: ``[1, 4, 0, 6]`` -> ``[1, 1, 0, 1]`` -> 0.75.

    The alternative -- ``sum(values) / total_steps``, which is what these
    columns used to be -- is not a proportion and does not stay inside [0, 1].
    The same list gives 11/4 = 2.75, a "normalized" metric reported as 2.75,
    and a step that cites four facts counts four times as much as a step that
    cites one. That is a density, and a density is already reported: the raw
    sum is kept beside every one of these as its own column
    (``reasoning_prior_knowledge`` beside ``reasoning_prior_knowledge_normalized``),
    so nothing is lost by making the normalized one mean what its name says.

    Whether a step drew on prior knowledge is a property of the step; how many
    times it did is a different measurement, and mixing them makes a column
    that is neither.
    """
    if not total_steps:
        return None
    return sum(1.0 for value in values if value) / total_steps


def _numbered(items: list[str]) -> str:
    """``1. first\n2. second`` -- how a list reaches a judge.

    Numbered rather than bulleted because every per-step list that comes back
    is positional: the judge has to be able to see which step is which, and the
    validation downstream rejects a list whose length does not match.
    """
    return "\n".join(f"{index}. {item}" for index, item in enumerate(items, start=1))


def _string_list(value: Any) -> list[str] | None:
    if not isinstance(value, list) or not value:
        return None
    if any(not isinstance(item, str) or not item.strip() for item in value):
        return None
    return [item.strip() for item in value]


def derive_reasoning_metrics(
    raw: dict[str, dict[str, Any]],
    *,
    generation_like: bool,
    selection_like: bool,
    option_count: int,
    pipeline_like: bool = False,
    bov: bool = False,
) -> tuple[dict[str, float], dict[str, list[Any]], list[str], list[str]]:
    """Validate the judges' raw outputs and compute every derived value.

    Returns ``(metrics, lists, errors, inapplicable)``.  Nothing is coerced: a
    value that is missing, malformed or impossible produces an entry in
    ``errors`` and no metric, and a metric that does not apply to this task
    shape produces an entry in ``inapplicable`` and no metric.  The two are kept
    apart because "the judge could not answer" and "there was nothing to ask"
    are different facts about a blank cell.

    **Every normalization happens here**, after the raw values are in hand. No
    judge is ever told about a ratio, a denominator or a derived value; each is
    asked for counts and lists alone, and the arithmetic is this function's.

    **The step list governs everything.** Metric 2 segments the chain once, and
    every per-step list is indexed by that segmentation, so a list whose length
    does not match it is rejected rather than padded or truncated -- a
    mismatched list would silently attribute one step's count to another.
    """
    metrics: dict[str, float] = {}
    lists: dict[str, list[Any]] = {}
    errors: list[str] = []
    inapplicable: list[str] = []

    # -- metric 2: the canonical segmentation, first because all else needs it #
    steps_blob = raw.get("steps") or {}
    steps = _string_list(steps_blob.get("steps"))
    if steps is None:
        errors.append("steps:invalid_or_missing_step_list")
        total_steps = 0
    else:
        lists["reasoning_steps"] = steps
        total_steps = len(steps)
        # Always the length of the list above, never a number the judge
        # reported: one of them has to be authoritative, and it is the list.
        metrics["reasoning_total_steps"] = float(total_steps)

    proofs = _int_list(steps_blob.get("proof_disproof_counts"), total_steps or None)
    if steps is None:
        pass  # already reported; nothing downstream can be indexed
    elif proofs is None:
        errors.append("steps:proof_disproof_list_missing_or_not_one_value_per_step")
    else:
        lists["reasoning_proof_disproof_per_step"] = proofs
        useless = sum(1 for count in proofs if count == 0)
        useful = total_steps - useless
        metrics["reasoning_useful_steps"] = float(useful)
        metrics["reasoning_useless_steps"] = float(useless)
        if total_steps:
            metrics["reasoning_useful_step_fraction"] = useful / total_steps
            metrics["reasoning_useless_step_fraction"] = useless / total_steps

    backtracking = _nonnegative_int(steps_blob.get("backtracking_steps"))
    if backtracking is None:
        errors.append("steps:invalid_or_missing_backtracking_count")
    elif steps is not None and backtracking > total_steps:
        # More corrections than steps cannot be a reading of this chain.
        errors.append("steps:backtracking_exceeds_total_steps")
    else:
        metrics["reasoning_backtracking_steps"] = float(backtracking)
        if total_steps:
            metrics["reasoning_backtracking_rate"] = backtracking / total_steps

    def per_step(family: str, key: str, column: str, kind=_int_list) -> list[Any] | None:
        """One per-step list, validated against the canonical step count."""
        if steps is None:
            inapplicable.append(f"{family}:no_step_list_to_index_against")
            return None
        parsed = kind((raw.get(family) or {}).get(key), total_steps)
        if parsed is None:
            errors.append(f"{family}:missing_or_not_one_value_per_step")
            return None
        lists[column] = parsed
        return parsed

    # -- metric 1: observation coverage --------------------------------------- #
    #
    # Two calls, deliberately. The wave-one inventory lists the question's
    # observations and is the denominator; observation_coverage then places
    # those same observations in the chain and supplies the numerator. Asking
    # one judge for both let it place more observations than it had just
    # found, and nothing could tell which half was wrong.
    inventory = _string_list((raw.get("observation_inventory") or {}).get("observations"))
    observations_total = len(inventory) if inventory else 0
    if not observations_total:
        errors.append("observation_inventory:invalid_or_missing_observations")
    else:
        metrics["reasoning_observations_total"] = float(observations_total)
        lists["reasoning_observations"] = list(inventory or [])

    covered = per_step("observation_coverage", "observations_per_step",
                       "reasoning_observations_per_step")
    if covered is not None:
        # Each observation is counted at its first appearance only, so the list
        # already sums to the distinct observations the chain reached -- asking
        # the judge for that sum as well would only give it a second chance to
        # disagree with itself.
        used = sum(covered)
        if observations_total and used > observations_total:
            # More observations placed than the inventory holds. The inventory
            # is the fixed list and the coverage judge was told not to add to
            # it, so this is that judge disagreeing with its own instructions.
            errors.append("observation_coverage:used_exceeds_the_inventory")
        else:
            metrics["reasoning_observations_used"] = float(used)
            if observations_total:
                metrics["reasoning_observation_coverage"] = used / observations_total

    # -- metric 3: branchiness, and the n each task shape normalizes by ------- #
    family = "branchiness_selection" if selection_like and not generation_like else (
        "branchiness_generation"
    )
    branchiness = per_step(family, "branchiness_per_step", "reasoning_branchiness_per_step")
    branchiness_total: int | None = None
    if branchiness is not None:
        branchiness_total = sum(branchiness)
        metrics["reasoning_branchiness_total"] = float(branchiness_total)

    if family == "branchiness_generation":
        diversity = _binary((raw.get(family) or {}).get("diversity"))
        if diversity is None:
            errors.append("branchiness_generation:invalid_or_missing_diversity")
        else:
            metrics["reasoning_diversity"] = float(diversity)
        inapplicable.append("option_count:generation_task_has_no_answer_options")
        hypothesis_count = branchiness_total
    else:
        # From its own wave-one prompt, not from branchiness. Counting the
        # options and counting what the model raised are different readings of
        # the question, and one prompt doing both made the second depend on the
        # first having gone well.
        judged_options = _nonnegative_int((raw.get("option_count") or {}).get("option_count"))
        if judged_options is None:
            errors.append("option_count:invalid_or_missing_option_count")
        else:
            metrics["reasoning_option_count"] = float(judged_options)
        inapplicable.append("diversity:selection_candidates_are_the_questions_not_the_models")
        # The question's own option count is the ground truth where we have it;
        # the judge's reading is a fallback, not an override.
        hypothesis_count = option_count if option_count >= 2 else judged_options

    # -- metric 4: directionality over the whole chain ------------------------ #
    direction = _nonnegative_number((raw.get("directionality") or {}).get("directionality"))
    if direction is None or direction not in (0.0, 0.5, 1.0):
        errors.append("directionality:expected_0_0.5_or_1")
    else:
        metrics["reasoning_directionality"] = direction

    # -- metric 5: directionality per step ------------------------------------ #
    step_direction = per_step("step_directionality", "directionality_per_step",
                              "reasoning_step_directionality_per_step",
                              kind=_directionality_list)
    if step_direction is not None and total_steps:
        metrics["reasoning_step_directionality_mean"] = sum(step_direction) / total_steps

    # -- metrics 6 and 7: elimination, and how exhaustive it was -------------- #
    comparisons = per_step("differential_elimination", "comparisons_per_step",
                           "reasoning_comparisons_per_step")
    if comparisons is not None:
        compared = sum(comparisons)
        metrics["reasoning_differential_elimination"] = float(compared)
        if total_steps:
            metrics["reasoning_differential_elimination_normalized"] = (
                _proportion_of_steps(comparisons, total_steps)
            )
        pairs = comparison_pairs(hypothesis_count or 0)
        if pairs:
            metrics["reasoning_comparison_exhaustiveness"] = compared / pairs
        elif bov:
            inapplicable.append("comparison_exhaustiveness:bov_shows_one_option_per_chain")
        else:
            inapplicable.append("comparison_exhaustiveness:fewer_than_two_hypotheses_to_pair")

    # -- metric 8: uncertainty ------------------------------------------------ #
    uncertainty = per_step("uncertainty", "uncertainty_per_step",
                           "reasoning_uncertainty_per_step")
    if uncertainty is not None:
        marked = sum(uncertainty)
        metrics["reasoning_uncertainty_steps"] = float(marked)
        if total_steps:
            metrics["reasoning_uncertainty_rate"] = _proportion_of_steps(
                uncertainty, total_steps
            )

    # -- metric 9: prior knowledge -------------------------------------------- #
    prior = per_step("prior_knowledge", "prior_knowledge_per_step",
                     "reasoning_prior_knowledge_per_step")
    if prior is not None:
        borrowed = sum(prior)
        metrics["reasoning_prior_knowledge"] = float(borrowed)
        if total_steps:
            metrics["reasoning_prior_knowledge_normalized"] = _proportion_of_steps(
                prior, total_steps
            )

    # -- metric 10: anchoring point ------------------------------------------- #
    anchor_blob = raw.get("anchoring_point") or {}
    if "anchoring_step_index" not in anchor_blob:
        errors.append("anchoring_point:missing_output")
    else:
        anchor = anchor_blob["anchoring_step_index"]
        if anchor is None:
            # The model never held the right answer. A real finding, and a
            # different one from holding it at step 0, so it stays blank rather
            # than becoming a number.
            inapplicable.append("anchoring_point:correct_answer_never_considered")
        else:
            index = _nonnegative_int(anchor)
            if index is None or (steps is not None and index >= total_steps):
                errors.append("anchoring_point:not_an_index_into_the_step_list")
            else:
                metrics["reasoning_anchoring_point"] = float(index)
                if total_steps:
                    metrics["reasoning_anchoring_point_normalized"] = index / total_steps

    # -- metric 11: unresolved contradictions --------------------------------- #
    unresolved = per_step("unresolved_contradiction", "unresolved_per_step",
                          "reasoning_unresolved_per_step", kind=_binary_list)
    if unresolved is not None:
        left = sum(unresolved)
        metrics["reasoning_unresolved_contradictions"] = float(left)
        if total_steps:
            metrics["reasoning_unresolved_contradiction_normalized"] = (
                _proportion_of_steps(unresolved, total_steps)
            )

    return metrics, lists, errors, inapplicable


class ReasoningJudgeStage:
    """Runs every reasoning-chain judge prompt, batched and in parallel.

    Concurrency is the whole point of the shape of this class.  A run judges
    tens of thousands of chains across nine metric families, and issuing those
    calls one at a time would take longer than the inference being measured.
    Three things run at once instead:

    * **Within a family, requests are batched** ``group_size`` at a time into
      one call on the server's batch route.
    * **Across families, the batches are issued together.**  Only two real
      dependencies exist -- coverage and redundancy need the observation
      inventory, uncertainty needs the step count -- so everything else goes out
      in one wave rather than in nine sequential steps.
    * **Across tasks, the stage is shared**, so several tasks judge at the same
      time and a question's inventory is bought once for all of them.

    All of it is throttled by one semaphore, which the engine owns and shares
    with the answer judge, because both talk to the same server: in-flight
    sequences are ``group_size x max_parallel_calls`` regardless of how many
    families and tasks are judging.  That is what fills the judge's scheduler
    slots without pushing vLLM into KV-cache preemption, which is slower than
    not parallelising at all.
    """

    def __init__(
        self,
        *,
        config: ReasoningJudgeConfig,
        registry: PromptRegistry,
        renderer: PromptRenderer,
        clients: dict[str, ModelClient],
        retry_policy: RetryPolicy,
        cache_dir: Path,
        batch_disabled: set[str] | None = None,
        calls: asyncio.Semaphore | None = None,
        log_path: Path | None = None,
        run_dir: Path | None = None,
    ):
        self.config = config
        self.registry = registry
        self.renderer = renderer
        self.retry_policy = retry_policy
        self.cache_dir = Path(cache_dir)
        if config.model not in clients:
            raise ConfigError(
                f"engine.reasoning_judge.model={config.model!r} is not one of the run's "
                f"models ({sorted(clients)}); add it to the run's model list"
            )
        missing = _REQUIRED_TEMPLATES - set(config.templates)
        if missing:
            raise ConfigError(
                "engine.reasoning_judge.templates is missing: " + ", ".join(sorted(missing))
            )
        self.client = clients[config.model]
        #: Live view of the models whose batch endpoint the engine found
        #: unusable.  Held by reference and read at call time, because the
        #: engine fills it in after this stage is built.
        self._batch_disabled = batch_disabled if batch_disabled is not None else set()
        self.templates: dict[str, PromptTemplate] = {
            name: registry.get(template_id) for name, template_id in config.templates.items()
        }
        self._cache_path = self.cache_dir / "verdicts.json"
        self._cache: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._calls = calls or asyncio.Semaphore(config.max_parallel_calls)
        #: A standalone JSONL log of these metrics alone, one line per judged
        #: output, written next to the records.  The sheet is the report and
        #: records.jsonl is the evidence; this is the audit trail for this
        #: stage specifically -- every raw count, every derived value, and the
        #: status that explains any blank -- and it lands inside the run
        #: directory, so `engine.sync` mirrors it off-box with everything else.
        self.log_path = Path(log_path) if log_path else None
        #: The run directory, so every judge call can be written next to the
        #: task whose chains it graded.  One audit file per task rather than one
        #: per run on purpose: the sync mirrors whole files, and a single
        #: growing multi-hundred-megabyte log would be re-uploaded in full on
        #: every tick, while a finished task's file stops changing and stops
        #: costing anything.
        self.run_dir = Path(run_dir) if run_dir else None
        #: Counters for the stage's own log line.
        self.stats: dict[str, int] = {"judged": 0, "cached": 0, "calls": 0, "failed": 0}
        if config.cache and self._cache_path.exists():
            try:
                payload = orjson.loads(self._cache_path.read_bytes())
                if isinstance(payload, dict):
                    self._cache = payload
            except orjson.JSONDecodeError:
                logger.warning(
                    "reasoning judge cache %s is corrupt; starting fresh", self._cache_path
                )

    # -- entry point --------------------------------------------------------- #

    async def apply(
        self,
        adapter: DatasetAdapter,
        identity: TaskIdentity,
        prompts: list[RenderedPrompt],
        scored: list[tuple[SampleSpec, ModelResponse, SampleScore]],
    ) -> list[tuple[SampleSpec, ModelResponse, SampleScore]]:
        """Add reasoning metrics to COT outputs, leaving every IO score untouched."""
        if identity.prompt_mode not in (COT, SELF_CONSISTENCY) or not scored:
            return scored

        prompt_by_id = {prompt.sample_id: prompt for prompt in prompts}
        try:
            processing_mode = adapter.documentation().processing_mode.lower()
        except Exception:  # pragma: no cover - adapters document during preparation
            processing_mode = ""
        # A pipeline benchmark poses generation and selection as one task rather
        # than as two, so its chain both proposes explanations and chooses among
        # them: both the generation-only and the selection-only families apply.
        pipeline = "generation & selection" in processing_mode and "separate" not in processing_mode

        targets: list[_Target] = []
        updated = list(scored)
        for index, (sample, response, score) in enumerate(scored):
            prompt = prompt_by_id.get(sample.sample_id)
            if prompt is None:
                self._mark(updated, index, "not_applicable:prompt_unavailable")
                continue
            question, reasoning = self._question_and_reasoning(prompt, response)
            too_big = exceeds_budget(
                {"question": question, "reasoning_chain": reasoning},
                {
                    "question": self.config.max_chain_chars,
                    "reasoning_chain": self.config.max_chain_chars,
                },
            )
            if too_big:
                # A judge shown a chain with its middle removed is answering a
                # different question -- "how many steps are there" least of all
                # survives it -- and the answer would be averaged in beside
                # verdicts read from whole chains. Skipped, counted, reported.
                self.stats["skipped_oversize"] = self.stats.get("skipped_oversize", 0) + 1
                self._mark(updated, index, f"not_applicable:oversize:{too_big}")
                continue
            if not reasoning.strip():
                self._mark(updated, index, "not_applicable:no_reasoning_chain")
                continue
            selection_like = sample.task_kind in _SELECTION_KINDS
            generation_like = sample.task_kind in _GENERATION_KINDS
            if not (selection_like or generation_like or pipeline):
                self._mark(
                    updated, index, f"not_applicable:unsupported_task_kind:{sample.task_kind}"
                )
                continue
            root = self._root_sample(sample)
            options = [str(value) for value in (root.fields.get("options") or [])]
            targets.append(
                _Target(
                    index=index,
                    sample=sample,
                    response=response,
                    score=score,
                    question=question,
                    reasoning=reasoning,
                    # Not truncated at a fixed few thousand characters any more:
                    # what a judge may be shown is a property of its context
                    # window, and _clip_chain applies that one budget to all of
                    # it. A reference cut mid-JSON told the judge less than
                    # nothing.
                    answer=clip_middle(
                        score.prediction or response.text or "",
                        self.config.max_reference_chars,
                    ),
                    reference=clip_middle(
                        json.dumps(sample.reference, ensure_ascii=False, default=str),
                        self.config.max_reference_chars,
                    ),
                    options=options,
                    generation_like=generation_like,
                    selection_like=selection_like,
                    pipeline_like=pipeline,
                    bov=identity.selection_mode == BOV,
                )
            )

        if not targets:
            return updated

        await self._evaluate(targets, identity)
        async with self._lock:
            self._save_cache()

        # Raw values for every family are in hand before a single normalized
        # value is computed -- including the per-sample observation total, which
        # metric 1 and metric 4 both divide by.
        log_lines: list[dict[str, Any]] = []
        for target in targets:
            metrics, raw_lists, errors, inapplicable = derive_reasoning_metrics(
                target.raw,
                generation_like=target.generation_like,
                selection_like=target.selection_like,
                pipeline_like=target.pipeline_like,
                option_count=len(target.options),
                bov=target.bov,
            )
            errors = [*target.errors, *errors]
            inapplicable = [*target.inapplicable, *inapplicable]
            details = dict(target.score.details)
            details["reasoning_metrics_status"] = "ok" if not errors else "partial"
            if errors:
                details["reasoning_judge_errors"] = errors
            if inapplicable:
                details["reasoning_metrics_inapplicable"] = inapplicable
            # The per-step lists ride on the record rather than on the metrics,
            # because metrics are numbers a task can average and these are not.
            # Kept whole: the point of asking per step was to be able to plot
            # the distribution later, which an average has already thrown away.
            if raw_lists:
                details["reasoning_lists"] = raw_lists
            updated[target.index] = (
                target.sample,
                target.response,
                SampleScore(
                    metrics={**target.score.metrics, **metrics},
                    prediction=target.score.prediction,
                    parse_ok=target.score.parse_ok,
                    details=details,
                ),
            )
            self.stats["judged"] += 1
            log_lines.append(
                {
                    "run_id": identity.run_id,
                    "dataset_id": identity.dataset_id,
                    "model_id": identity.model_id,
                    "prompt_mode": identity.prompt_mode,
                    "selection_mode": identity.selection_mode,
                    "task_kind": target.sample.task_kind,
                    "sample_id": target.sample.sample_id,
                    "status": details["reasoning_metrics_status"],
                    "option_count": len(target.options),
                    # Exactly what each judge returned, beside what was derived
                    # from it, so a number in the sheet can always be traced
                    # back to the reply it came from.
                    "raw": target.raw,
                    "metrics": {name: metrics.get(name) for name in REASONING_METRIC_COLUMNS},
                    "lists": {name: raw_lists.get(name) for name in REASONING_LIST_COLUMNS},
                    "errors": errors,
                    "inapplicable": inapplicable,
                }
            )
        self._append_log(log_lines)
        return updated

    def _append_log(self, lines: list[dict[str, Any]]) -> None:
        if not self.log_path or not lines:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        blob = b"".join(orjson.dumps(line, default=str) + b"\n" for line in lines)
        with self.log_path.open("ab") as handle:
            handle.write(blob)

    #: Version of the audit record's shape, so a reader can tell what to
    #: expect from a file written by an older run.
    AUDIT_SCHEMA = "reasoning_judge_call/v1"
    AUDIT_FILENAME = "reasoning_judge_calls.jsonl"

    def _audit_path(self, identity: TaskIdentity | None) -> Path | None:
        """Where this task's judge calls are written, or ``None`` if nowhere.

        Beside the task's ``records.jsonl`` and ``raw/``, because that is where
        a reader already goes to see what the evaluated model was asked and
        what it said; the judge's side of the same sample belongs next to it
        and is mirrored off-box by the same sync.
        """
        if self.run_dir is None or identity is None:
            return None
        return (
            self.run_dir
            / "datasets"
            / identity.dataset_id
            / identity.model_id
            / f"{identity.template_id}@{identity.template_version}"
            / self.AUDIT_FILENAME
        )

    def _append_audit(self, identity: TaskIdentity | None, lines: list[dict[str, Any]]) -> None:
        """Append judge-call records, untruncated.

        Never raises: an audit trail that can break a run is worse than one
        that reports it missed a line.
        """
        path = self._audit_path(identity)
        if path is None or not lines:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            blob = b"".join(orjson.dumps(line, default=str) + b"\n" for line in lines)
            with path.open("ab") as handle:
                handle.write(blob)
        except OSError as exc:  # pragma: no cover - disk full, permissions
            logger.warning("reasoning judge: could not write the audit log %s: %s", path, exc)

    @staticmethod
    def _audit_record(
        *,
        identity: TaskIdentity | None,
        family: str,
        template: PromptTemplate,
        cache_key: str,
        judge_model: str | None,
        context: dict[str, Any],
        messages: Any,
        outcome: str,
        content: str | None = None,
        reasoning: str | None = None,
        parsed: dict[str, Any] | None = None,
        parsed_from: str | None = None,
        error: str | None = None,
        usage: dict[str, Any] | None = None,
        finish_reason: str | None = None,
    ) -> dict[str, Any]:
        """One judge call, in full.

        The prompt and the answer are stored whole. Clipping is what the metric
        log already does, and it is exactly what makes a score unauditable: the
        question "why did the judge say 14 steps" cannot be answered from a
        truncated reply.

        ``reasoning_trace.available`` is recorded explicitly rather than left to
        be inferred from an empty string, because "the API returned no trace"
        and "the trace was empty" are different facts and only the first is
        something this harness can state.
        """
        return {
            "schema": ReasoningJudgeStage.AUDIT_SCHEMA,
            "ts": datetime.now(timezone.utc).isoformat(),
            # -- what was judged ------------------------------------------- #
            "run_id": getattr(identity, "run_id", None),
            "dataset_id": getattr(identity, "dataset_id", None),
            "model_id": getattr(identity, "model_id", None),
            "template_id": getattr(identity, "template_id", None),
            "template_version": getattr(identity, "template_version", None),
            "prompt_mode": getattr(identity, "prompt_mode", None),
            **context,
            # -- who judged it --------------------------------------------- #
            "metric_family": family,
            "judge_model": judge_model,
            "judge_template": template.ref,
            "cache_key": cache_key,
            # -- the call --------------------------------------------------- #
            "request": {"messages": messages},
            "response": {
                "outcome": outcome,
                "content": content,
                "reasoning_trace": {
                    "available": reasoning is not None,
                    "text": reasoning,
                },
                "finish_reason": finish_reason,
                "usage": usage,
                "error": error,
            },
            "parse": {
                "ok": parsed is not None,
                "values": parsed,
                # Which channel the values came out of, so the fallback that
                # reads a verdict from the reasoning channel is visible rather
                # than silent.
                "source": parsed_from,
            },
        }

    @staticmethod
    def _mark(
        updated: list[tuple[SampleSpec, ModelResponse, SampleScore]], index: int, status: str
    ) -> None:
        """Record why an output got no reasoning metrics, without inventing any."""
        sample, response, score = updated[index]
        updated[index] = (
            sample,
            response,
            SampleScore(
                metrics=dict(score.metrics),
                prediction=score.prediction,
                parse_ok=score.parse_ok,
                details={**score.details, "reasoning_metrics_status": status},
            ),
        )

    # -- the two waves ------------------------------------------------------- #

    async def _run_family(
        self,
        family: str,
        selected: list[_Target],
        field_names: tuple[str, ...],
        common: dict[int, dict[str, Any]],
        identity: TaskIdentity | None,
    ) -> None:
        """One metric family's calls, and their results filed on the targets."""
        if not selected:
            return
        template = self.templates[family]
        requests = {
            str(target.index): {
                key: common[target.index][key]
                for key in field_names
                if common[target.index].get(key) is not None
            }
            for target in selected
        }
        # A missing dependency is reported, never rendered into the prompt as a
        # made-up zero.
        callable_requests = {
            key: fields
            for key, fields in requests.items()
            if all(required in fields for required in template.required_fields)
        }
        results = await self._judge_many(
            family,
            callable_requests,
            identity=identity,
            context={
                str(target.index): {
                    "sample_id": self._sample_id(target),
                    "group_id": getattr(target.sample, "group_id", None),
                    "repeat_of": (getattr(target.sample, "metadata", {}) or {}).get("repeat_of"),
                    "target_index": target.index,
                }
                for target in selected
            },
        )
        for target in selected:
            key = str(target.index)
            if key not in callable_requests:
                target.errors.append(f"{family}:missing_required_dependency")
                target.raw[family] = {}
            elif results.get(key) is None:
                target.errors.append(f"{family}:judge_failed_or_unparseable")
                target.raw[family] = {}
            else:
                target.raw[family] = results[key]

    async def _evaluate(self, targets: list[_Target], identity: TaskIdentity | None = None) -> None:
        """Buy every metric for one task's chains, in dependency order.

        Two waves, and the split is a real dependency rather than a schedule.
        The segmentation goes out first because every per-step list below is
        indexed by the steps it returns -- none of them can even be asked until
        it has come back. Everything else then goes out together.

        Only two prompts read the chain itself: this segmentation, and
        whole-chain directionality, which judges the chain as one thing and
        needs no step boundaries.
        """
        common: dict[int, dict[str, Any]] = {
            target.index: {
                "question": target.question,
                "reasoning_chain": target.reasoning,
                "model_answer": target.answer,
                "reference_answer": target.reference,
                "options": target.options,
                "option_count": len(target.options),
            }
            for target in targets
        }

        # -- wave one: three readings, once each, before anything else ------ #
        #
        # Three calls, not one. Each reads a different thing and each produces
        # something the second wave depends on:
        #
        #   steps                 -- segments the model's chain. Every per-step
        #                            list below is indexed by what it returns.
        #   observation_inventory -- lists the question's facts. The denominator
        #                            of coverage.
        #   option_count          -- counts the question's answer options.
        #                            SELECTION TASKS ONLY; a generation task has
        #                            none and is not sent.
        #
        # They are kept apart because merging them makes one reading's mistake
        # become the other's: a judge that both inventories the observations and
        # places them can place four of the three it just found, and nothing
        # downstream can tell which half was wrong. Split, the inventory is
        # fixed before anything is measured against it.
        #
        # They run concurrently -- none reads another's output -- and each
        # result is then substituted into `common` so wave two receives it
        # rather than asking for it again.
        selection_ids = {id(t) for t in targets if t.selection_like and not t.generation_like}
        needs_options = [t for t in targets if id(t) in selection_ids]
        await asyncio.gather(
            self._run_family(
                "steps", targets, ("question", "reasoning_chain", "options"), common, identity
            ),
            self._run_family(
                "observation_inventory", targets, ("question",), common, identity
            ),
            self._run_family(
                "option_count", needs_options, ("question",), common, identity
            ),
        )

        for target in targets:
            # The inventory, numbered so the judge's i-th observation and this
            # code's i-th are the same one.
            inventory = _string_list(
                (target.raw.get("observation_inventory") or {}).get("observations")
            )
            if inventory is None:
                target.errors.append(
                    "observation_inventory:no_inventory_so_no_coverage"
                )
            else:
                common[target.index]["observations"] = _numbered(inventory)
            # The judged option count replaces the structural one where we have
            # it. Selection tasks only; for the rest the key stays as the
            # question's own len(options), which nothing in wave two reads.
            if id(target) in selection_ids:
                judged = _nonnegative_int(
                    (target.raw.get("option_count") or {}).get("option_count")
                )
                if judged is not None:
                    common[target.index]["option_count"] = judged

        for target in targets:
            segmented = _string_list((target.raw.get("steps") or {}).get("steps"))
            if segmented is None:
                # Nothing downstream can be indexed against a chain that did not
                # segment. Reported once here rather than nine times below.
                target.errors.append("steps:no_segmentation_so_no_per_step_metric")
                continue
            # Numbered on the way in so the judge's i-th value and this code's
            # i-th step are the same step. The prompts say "ordered", which is
            # the property that matters and is true either way.
            common[target.index]["steps"] = _numbered(segmented)

        ready = [t for t in targets if "steps" in common[t.index]]

        # -- wave two: everything the segmentation unlocked ----------------- #
        #
        # Branchiness is routed on the task's known shape, never inferred by the
        # judge: whether the candidates are the question's or the model's
        # changes what "newly proposed" counts. A pipeline task proposes its own
        # explanations as well as choosing, so it takes the generation prompt.
        run = lambda family, chosen, fields: self._run_family(  # noqa: E731 - a local alias
            family, chosen, fields, common, identity
        )
        inventoried = [t for t in ready if "observations" in common[t.index]]
        wave_two = [
            run("observation_coverage", inventoried, ("question", "steps", "observations")),
            run("directionality", targets, ("question", "reasoning_chain")),
            run("step_directionality", ready, ("steps",)),
            run("differential_elimination", ready, ("steps",)),
            run("uncertainty", ready, ("steps",)),
            run("prior_knowledge", ready, ("question", "steps")),
            run("unresolved_contradiction", ready, ("steps",)),
            run("anchoring_point", [t for t in ready if t.reference],
                ("steps", "reference_answer")),
            run("branchiness_selection", [t for t in ready if id(t) in selection_ids],
                ("steps", "question", "option_count")),
            run("branchiness_generation", [t for t in ready if id(t) not in selection_ids],
                ("steps",)),
        ]
        for target in ready:
            if not target.reference:
                target.inapplicable.append("anchoring_point:no_reference_answer_to_anchor_on")
        await asyncio.gather(*wave_two)

    # -- interaction relevance: a sibling pass, not part of the above -------- #

    async def apply_interaction(
        self,
        adapter: DatasetAdapter,
        identity: TaskIdentity,
        prompts: list[RenderedPrompt],
        scored: list[tuple[SampleSpec, ModelResponse, SampleScore]],
    ) -> list[tuple[SampleSpec, ModelResponse, SampleScore]]:
        """Did each action of an interactive episode move toward the GOLD answer?

        Relevance is graded with full hindsight against the true answer, not
        against the answer the model gave. The two are different measurements
        and only one of them is worth having: judged against the model's own
        answer, a model that investigated confidently down a wrong path scores
        a perfect relevance rate for the steps that built its mistake, because
        every one of them did work toward the answer it ended up giving. Judged
        against the gold, that episode scores what it earned.

        Deliberately separate from :meth:`apply`, and sharing nothing with it
        but the cache, the audit log and the call budget:

        * It runs on **interactive deliveries**, not on cot. Every interactive
          dataset here is ``io_only``, so the reasoning pass never sees one --
          and what this measures is the episode's *actions*, which exist
          whether or not the model was asked to reason aloud.
        * It reads the **transcript**, not a chain of thought. The steps are
          what the model did and what the environment answered.

        The pair with ``interaction_steps`` is the point. That counts what an
        episode spent; this says how much of it did any work. A model that
        reaches the answer in nineteen actions of which four mattered is not
        the same as one that took four.
        """
        if identity.data_delivery_mode != "interactive" or not scored:
            return scored

        prompt_by_id = {prompt.sample_id: prompt for prompt in prompts}
        updated = list(scored)
        requests: dict[str, dict[str, Any]] = {}
        steps_by_id: dict[str, int] = {}
        for index, (sample, response, score) in enumerate(scored):
            prompt = prompt_by_id.get(sample.sample_id)
            if prompt is None:
                continue
            actions, final = self._episode_actions(sample)
            if not actions or not final:
                # One action and nothing before it is not an investigation;
                # there is no step whose relevance could differ.
                continue
            gold = self._gold_answer(adapter, sample, response, score)
            if not gold:
                # The standard this metric is measured against is missing, so
                # there is no measurement to make. Said out loud on the sample
                # rather than judged against the model's own answer instead,
                # which is the question this metric deliberately does not ask.
                _, _, existing = updated[index]
                updated[index] = (
                    sample,
                    response,
                    SampleScore(
                        metrics=existing.metrics,
                        prediction=existing.prediction,
                        parse_ok=existing.parse_ok,
                        details={
                            **(existing.details or {}),
                            "interaction_step_relevance": (
                                "unjudged: no reference answer to measure relevance against"
                            ),
                        },
                    ),
                )
                continue
            request_id = f"{index}"
            requests[request_id] = {
                "question": self._opening(sample, prompt),
                "steps": _numbered(actions),
                "model_answer": final,
                "reference_answer": gold,
            }
            steps_by_id[request_id] = len(actions)

        if not requests:
            return updated

        results = await self._judge_many("step_relevance", requests, identity=identity)
        for request_id, blob in results.items():
            index = int(request_id)
            sample, response, score = updated[index]
            verdicts = _binary_list((blob or {}).get("relevance_per_step"),
                                    steps_by_id[request_id])
            metrics = dict(score.metrics)
            details = dict(score.details or {})
            if verdicts is None:
                details["interaction_step_relevance"] = (
                    "unjudged: the judge returned no usable per-step list"
                )
            else:
                metrics["interaction_relevant_steps"] = float(sum(verdicts))
                metrics["interaction_irrelevant_steps"] = float(len(verdicts) - sum(verdicts))
                # A RATE, so datasets with different budgets are comparable:
                # four relevant actions out of four is not the same result as
                # four out of nineteen.
                metrics["interaction_step_relevance_rate"] = sum(verdicts) / len(verdicts)
                details["interaction_step_relevance_per_step"] = verdicts
            updated[index] = (
                sample,
                response,
                SampleScore(
                    metrics=metrics,
                    prediction=score.prediction,
                    parse_ok=score.parse_ok,
                    details=details,
                ),
            )
        return updated

    @staticmethod
    def _gold_answer(
        adapter: DatasetAdapter,
        sample: SampleSpec,
        response: ModelResponse,
        score: SampleScore,
    ) -> str:
        """The true answer, as the dataset itself states it.

        Asked of the adapter first, because ``sample.reference`` is not a
        common shape and reading one key out of it does not generalise:
        cloud_opsbench keys its answer ``root_cause`` and has no ``gold`` at
        all, and vivabench's answer is a *list* of accepted diagnoses of which
        any one counts. ``judge_request`` is where each adapter already states
        its own answer for the answer judge -- cloud_opsbench's root cause,
        vivabench's joined accepted list -- so it is the same gold both metrics
        are measured against rather than a second, divergent one.

        It can decline: vivabench returns nothing for an episode that never
        committed to a diagnosis, and every adapter returns nothing for an
        empty response. The gold exists either way, so ``reference`` is the
        fallback -- and only if that is empty too is there no standard and no
        measurement.
        """
        gold: Any = None
        try:
            request = adapter.judge_request(sample, response, score)
        except Exception as exc:  # noqa: BLE001 - a bad hook must not lose the metric
            logger.debug(
                "interaction relevance: %s judge_request failed (%s); "
                "falling back to sample.reference",
                getattr(adapter, "dataset_id", "?"), exc,
            )
        else:
            if request:
                gold = request.get("gold")
        if not gold:
            reference = sample.reference or {}
            gold = reference.get("gold") or reference.get("root_cause")
            accepted = reference.get("accepted")
            if not gold and isinstance(accepted, (list, tuple)) and accepted:
                gold = "; ".join(str(item) for item in accepted)
        if isinstance(gold, (list, tuple)):
            gold = "; ".join(str(item) for item in gold)
        # 600, not 400: vivabench's gold is every accepted diagnosis joined,
        # and clipping that mid-list would hide alternatives that count as
        # correct -- so a step ruling one of them in would be graded against a
        # standard it does not appear in.
        return str(gold or "").strip()[:600]

    @staticmethod
    def _episode_actions(sample: SampleSpec) -> tuple[list[str], str]:
        """``(actions with their results, the final answer)`` from the transcript.

        The engine records the episode as alternating turns: the opening
        messages, then the model's action and the environment's reply, over and
        over. The LAST model turn is the answer; everything before it is an
        action whose relevance is in question.
        """
        transcript = sample.metadata.get("_transcript") or []
        turns = [row for row in transcript if isinstance(row, dict)]
        assistant = [i for i, row in enumerate(turns) if row.get("role") == "assistant"]
        if len(assistant) < 2:
            return [], ""
        final = str(turns[assistant[-1]].get("content") or "").strip()
        actions: list[str] = []
        for position in assistant[:-1]:
            action = str(turns[position].get("content") or "").strip()
            reply = ""
            if position + 1 < len(turns) and turns[position + 1].get("role") == "user":
                reply = str(turns[position + 1].get("content") or "").strip()
            actions.append(f"ACTION: {action}\nRESULT: {reply}" if reply else f"ACTION: {action}")
        return actions, final

    @staticmethod
    def _opening(sample: SampleSpec, prompt: RenderedPrompt) -> str:
        """The situation the model started from, before it did anything."""
        transcript = sample.metadata.get("_transcript") or []
        for row in transcript:
            if isinstance(row, dict) and row.get("role") == "user":
                return str(row.get("content") or "")
        return str(sample.fields.get("observation") or "")

    # -- one family's calls -------------------------------------------------- #

    @staticmethod
    def _sample_id(target: _Target) -> str | None:
        return getattr(getattr(target, "sample", None), "sample_id", None)

    async def _judge_many(
        self,
        family: str,
        requests: dict[str, dict[str, Any]],
        *,
        identity: TaskIdentity | None = None,
        context: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, dict[str, Any] | None]:
        template = self.templates[family]
        ctx = context or {}
        keyed: list[tuple[str, dict[str, Any], str]] = []
        out: dict[str, dict[str, Any] | None] = {}
        audit: list[dict[str, Any]] = []
        for request_id, fields in requests.items():
            # Keyed by the judge as well as the prompt: see the same change in
            # core/judge.py. A verdict is not a property of the question alone.
            key = stable_hash(
                {
                    "template": template.ref,
                    "judge": self.config.model,
                    "fields": fields,
                },
                length=32,
            )
            cached = self._cache.get(key)
            if cached is not None:
                values = cached.get("values")
                out[request_id] = values if isinstance(values, dict) else None
                self.stats["cached"] += 1
                # No call was made, so there is no response to record -- and
                # inventing one would be the fabrication this log exists to
                # prevent. The row says where the values came from instead, and
                # the cache key leads to the call that first produced them.
                audit.append(
                    self._audit_record(
                        identity=identity,
                        family=family,
                        template=template,
                        judge_model=self.config.model,
                        cache_key=key,
                        context=ctx.get(request_id, {}),
                        messages=None,
                        outcome="cache_hit",
                        parsed=values if isinstance(values, dict) else None,
                        parsed_from="cache",
                    )
                )
            else:
                keyed.append((request_id, fields, key))
        if audit:
            self._append_audit(identity, audit)

        # A group is always ``group_size`` requests, whether or not the server
        # has a batch route, so in-flight judge sequences stay at
        # ``group_size x max_parallel_calls`` either way. A server without one
        # gets the group as that many concurrent single calls inside one
        # semaphore slot; dropping the group size to 1 instead -- which is the
        # obvious thing to do -- would quietly cut the stage's concurrency by a
        # factor of group_size on exactly the servers that are slowest already.
        can_batch = self.client.supports_batch and self.config.model not in self._batch_disabled

        def record(
            request_id: str,
            key: str,
            raw: str,
            reasoning: str | None = None,
            *,
            messages: Any = None,
            finish_reason: str | None = None,
            usage: dict[str, Any] | None = None,
        ) -> None:
            values = self._parse_json(raw, template)
            parsed_from = "content" if values is not None else None
            if values is None and reasoning:
                # A reasoning model puts its chain in a separate channel and
                # `content` comes back null when that chain ran to the budget.
                # Where it got as far as the JSON before running out, the
                # verdict is in there and throwing it away costs a metric for
                # nothing. Deliberately a fallback, not the first choice: the
                # content channel is where a finished answer belongs.
                values = self._parse_json(reasoning, template)
                if values is not None:
                    parsed_from = "reasoning_trace"
                    self.stats["recovered_from_reasoning"] = (
                        self.stats.get("recovered_from_reasoning", 0) + 1
                    )
            if values is not None:
                self._cache[key] = {
                    "template": template.ref,
                    "values": values,
                    "raw": raw[:2000],
                    "parsed": True,
                }
            else:
                # An unparseable reply is deliberately not cached: caching it
                # would make one bad sample permanent for every later
                # continuation of the run, which is the opposite of what the
                # cache is for.
                logger.warning("reasoning judge %s: unparseable reply %r", family, raw[:200])
                self.stats["failed"] += 1
            out[request_id] = values
            # Written whether or not it parsed: an unparseable reply is exactly
            # the one worth being able to read back.
            self._append_audit(
                identity,
                [
                    self._audit_record(
                        identity=identity,
                        family=family,
                        template=template,
                        judge_model=self.config.model,
                        cache_key=key,
                        context=ctx.get(request_id, {}),
                        messages=messages,
                        outcome="ok" if values is not None else "unparseable",
                        content=raw,
                        reasoning=reasoning,
                        parsed=values,
                        parsed_from=parsed_from,
                        finish_reason=finish_reason,
                        usage=usage,
                    )
                ],
            )

        async def run_chunk(chunk: list[tuple[str, dict[str, Any], str]]) -> None:
            conversations = []
            # The exact bytes sent, kept per request so the audit can record the
            # prompt that produced each answer rather than a re-render of it.
            sent: dict[str, Any] = {}
            for request_id, fields, key in chunk:
                sample = SampleSpec(
                    sample_id=f"reasoning-judge::{key}", fields=fields, task_kind="judge"
                )
                messages, _contract = self.renderer.render(sample, template)
                conversations.append(messages)
                sent[request_id] = [m.to_dict() for m in messages]
            sampling = SamplingParams(
                max_tokens=self._budget_for(family, chunk),
                temperature=self.config.temperature,
                extra=self._sampling_extra(),
            )
            async with self._calls:
                if can_batch and len(conversations) > 1:
                    self.stats["calls"] += 1
                    try:
                        result, _outcome = await with_retry(
                            lambda convs=conversations, params=sampling: self.client.chat_batch(
                                convs, params
                            ),
                            policy=self.retry_policy,
                            description=f"reasoning judge {family} ({len(conversations)} items)",
                        )
                    except EndpointError as exc:
                        logger.warning(
                            "reasoning judge %s failed permanently: %s", family, exc
                        )
                        for request_id, _fields, _key in chunk:
                            out[request_id] = None
                            self.stats["failed"] += 1
                        self._append_audit(
                            identity,
                            [
                                self._audit_record(
                                    identity=identity,
                                    family=family,
                                    template=template,
                                    judge_model=self.config.model,
                                    cache_key=key,
                                    context=ctx.get(request_id, {}),
                                    messages=sent.get(request_id),
                                    outcome='call_failed',
                                    error=f'{type(exc).__name__}: {exc}',
                                )
                                for request_id, _fields, key in chunk
                            ],
                        )
                        return
                    if len(result.choices) != len(chunk):
                        # Losing the alignment between requests and answers
                        # would file one chain's metrics against another chain's
                        # row, so the whole group is reported unjudged instead.
                        logger.warning(
                            "reasoning judge %s: %d answer(s) for %d request(s); "
                            "dropping the group",
                            family,
                            len(result.choices),
                            len(chunk),
                        )
                        for request_id, _fields, _key in chunk:
                            out[request_id] = None
                            self.stats["failed"] += 1
                        self._append_audit(
                            identity,
                            [
                                self._audit_record(
                                    identity=identity,
                                    family=family,
                                    template=template,
                                    judge_model=self.config.model,
                                    cache_key=key,
                                    context=ctx.get(request_id, {}),
                                    messages=sent.get(request_id),
                                    outcome='call_failed',
                                    error=f'{len(result.choices)} answer(s) for {len(chunk)} request(s)',
                                )
                                for request_id, _fields, key in chunk
                            ],
                        )
                        return
                    for (request_id, _fields, key), choice in zip(
                        chunk, result.choices, strict=True
                    ):
                        record(
                            request_id,
                            key,
                            choice.content or "",
                            choice.reasoning,
                            messages=sent.get(request_id),
                            finish_reason=getattr(choice, "finish_reason", None),
                            usage=getattr(result, "usage", None),
                        )
                    return

                async def one(request_id: str, key: str, messages: Any) -> None:
                    self.stats["calls"] += 1
                    try:
                        result, _outcome = await with_retry(
                            lambda convs=messages, params=sampling: self.client.chat_single(
                                convs, params
                            ),
                            policy=self.retry_policy,
                            description=f"reasoning judge {family} (1 item)",
                        )
                    except EndpointError as exc:
                        logger.warning("reasoning judge %s failed permanently: %s", family, exc)
                        out[request_id] = None
                        self.stats["failed"] += 1
                        self._append_audit(
                            identity,
                            [
                                self._audit_record(
                                    identity=identity,
                                    family=family,
                                    template=template,
                                    judge_model=self.config.model,
                                    cache_key=key,
                                    context=ctx.get(request_id, {}),
                                    messages=sent.get(request_id),
                                    outcome="call_failed",
                                    error=f"{type(exc).__name__}: {exc}",
                                )
                            ],
                        )
                        return
                    choice = result.choices[0] if result.choices else None
                    if choice is None:
                        out[request_id] = None
                        self.stats["failed"] += 1
                        self._append_audit(
                            identity,
                            [
                                self._audit_record(
                                    identity=identity,
                                    family=family,
                                    template=template,
                                    judge_model=self.config.model,
                                    cache_key=key,
                                    context=ctx.get(request_id, {}),
                                    messages=sent.get(request_id),
                                    outcome="call_failed",
                                    error="the endpoint returned no choices",
                                )
                            ],
                        )
                        return
                    record(
                        request_id,
                        key,
                        choice.content or "",
                        choice.reasoning,
                        messages=sent.get(request_id),
                        finish_reason=getattr(choice, "finish_reason", None),
                        usage=getattr(result, "usage", None),
                    )

                await asyncio.gather(
                    *(
                        one(request_id, key, messages)
                        for (request_id, _fields, key), messages in zip(
                            chunk, conversations, strict=True
                        )
                    )
                )

        await asyncio.gather(
            *(run_chunk(chunk) for chunk in iter_chunks(keyed, self.config.group_size))
        )
        return out

    def _budget_for(self, family: str, chunk: list[tuple[str, dict[str, Any], str]]) -> int:
        """How many output tokens one group of calls may use.

        Every family but one answers with a short list of integers, and the
        configured budget is generous for that. ``steps`` is different in kind:
        it returns the chain's segmentation, so its reply is about as long as
        the chain it read. A fixed budget is the wrong shape for that -- too
        small and a long chain is cut off mid-JSON, which costs the sample every
        metric that indexes against the segmentation; too large and every short
        chain pays for headroom it never uses.

        So the steps budget follows its input: the chain's own length plus a
        constant for the JSON scaffolding and the judge's overhead. A batch
        shares one sampling object, so the group takes the largest chain in it
        -- sizing to the smallest would truncate the rest.
        """
        configured = self.config.max_tokens_by_family.get(family, self.config.max_tokens)
        if family != "steps":
            return configured
        longest = 0
        for _request_id, fields, _key in chunk:
            chain = str(fields.get("reasoning_chain") or "")
            longest = max(longest, self._estimate_tokens(chain))
        if not longest:
            return configured
        wanted = longest + self.config.steps_budget_headroom
        # Never below the configured floor, never above the ceiling that keeps
        # the request inside the judge's own context window.
        return max(configured, min(wanted, self.config.max_tokens_ceiling))

    def _estimate_tokens(self, text: str) -> int:
        """Roughly how many tokens ``text`` is, without paying for a tokenizer.

        Deliberately an over-estimate. This sizes ``steps``' output budget, and
        at four characters per token it was an under-estimate that truncated
        142 of one run's 1,334 steps calls -- every one of them returning
        ``finish_reason: "length"`` with the JSON cut off mid-token, and every
        one costing the sample the whole per-step metric set, since every
        per-step family indexes against this segmentation.
        """
        return int(len(text) / _CHARS_PER_TOKEN)

    def _sampling_extra(self) -> tuple[tuple[str, Any], ...]:
        """Vendor knobs sent with every judge call."""
        if not self.config.reasoning_effort:
            return ()
        return (("reasoning_effort", self.config.reasoning_effort),)

    def _clip_chain(self, chain: str) -> str:  # noqa: D401 - kept for the short fields
        """Keep a chain inside the judge's own context window.

        Clipped from the middle rather than the end: the opening says what the
        model set out to do and the close is where it commits, and both matter
        to every metric here.  A chain long enough to need this is one where
        losing the middle costs less than losing the whole call, which is what
        happens otherwise -- the request is rejected for length and the sample
        gets no metrics at all.
        """
        limit = self.config.max_chain_chars
        if len(chain) <= limit:
            return chain
        self.stats["clipped"] = self.stats.get("clipped", 0) + 1
        return clip_middle(chain, limit)

    # -- parsing -------------------------------------------------------------- #

    @staticmethod
    def _json_objects(raw: str) -> list[str]:
        """Every balanced ``{...}`` span in the text, outermost and in order.

        A judge that reasons in the open returns its analysis and then the
        object, so one greedy match from the first brace to the last spans both
        and parses as nothing.
        """
        spans: list[str] = []
        depth = 0
        start = -1
        in_string = False
        escaped = False
        for position, char in enumerate(raw):
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                if depth == 0:
                    start = position
                depth += 1
            elif char == "}" and depth:
                depth -= 1
                if depth == 0 and start >= 0:
                    spans.append(raw[start : position + 1])
        return spans

    @classmethod
    def _parse_json(cls, text: str, template: PromptTemplate) -> dict[str, Any] | None:
        raw = (text or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
        fields = (template.output_contract or {}).get("json_fields") or {}

        def usable(candidate: str) -> dict[str, Any] | None:
            try:
                parsed = orjson.loads(candidate)
            except orjson.JSONDecodeError:
                return None
            if not isinstance(parsed, dict):
                return None
            if any(name not in parsed for name in fields):
                return None
            # AND THE VALUES MUST BE WHAT THE CONTRACT SAYS THEY ARE.
            #
            # Checking only that the key names are present is what let
            # `{"steps": 42}` -- declared `list_of_strings` -- count as a
            # successful parse. The derivation downstream rejects it, correctly,
            # so the metric is lost; but by then `_judge_many` has already
            # written it to the cache as a verdict, and the cache is only
            # written for values that parsed. So the bad reply is permanent:
            # re-running the reasoning judge serves it straight back without
            # calling the judge, and every dependent metric stays missing for
            # the life of the run directory. That is a mechanism by which a
            # rejudge pass cannot repair what it was run to repair.
            #
            # The validators already existed -- these are the same ones the
            # derivation uses. They were simply applied one step too late.
            for name, declared in fields.items():
                if _satisfies_contract(parsed.get(name), str(declared)) is False:
                    return None
            return parsed

        answer = usable(raw)
        if answer is not None:
            return answer
        # Last one wins: the verdict is what the judge settled on, not the
        # example object it may have echoed from the instructions first.
        for span in reversed(cls._json_objects(raw)):
            answer = usable(span)
            if answer is not None:
                return answer
        # A stray brace in the judge's prose unbalances the scan above, so fall
        # back to flat brace pairs -- which is the shape every verdict has.
        for span in reversed(re.findall(r"\{[^{}]*\}", raw, flags=re.DOTALL)):
            answer = usable(span)
            if answer is not None:
                return answer
        return None

    def _save_cache(self) -> None:
        if not self.config.cache:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        temp = self._cache_path.with_suffix(".json.tmp")
        temp.write_bytes(orjson.dumps(self._cache, option=orjson.OPT_INDENT_2))
        temp.replace(self._cache_path)

    # -- what the judge is shown ---------------------------------------------- #

    @staticmethod
    def _root_sample(sample: SampleSpec) -> SampleSpec:
        """The sample a repeat or a BOV question was expanded from."""
        root = sample
        seen: set[int] = set()
        while id(root) not in seen:
            seen.add(id(root))
            parent = root.metadata.get("_parent_sample")
            if not isinstance(parent, SampleSpec):
                break
            root = parent
        return root

    @staticmethod
    def _question_and_reasoning(
        prompt: RenderedPrompt, response: ModelResponse
    ) -> tuple[str, str]:
        """What the model was asked, and the chain it produced.

        An interactive benchmark is a conversation rather than a prompt, so its
        question is every non-assistant turn and its chain is everything the
        model said across the episode.
        """
        transcript = prompt.sample.metadata.get("_transcript")
        if isinstance(transcript, list) and transcript:
            question_parts: list[str] = []
            reasoning_parts: list[str] = []
            for message in transcript:
                if not isinstance(message, dict):
                    continue
                role = str(message.get("role", "unknown"))
                content = str(message.get("content", ""))
                if role == "assistant":
                    if content:
                        reasoning_parts.append(content)
                else:
                    question_parts.append(f"[{role.upper()}]\n{content}")
            if response.reasoning:
                reasoning_parts.append(response.reasoning)
            return "\n\n".join(question_parts), "\n\n".join(reasoning_parts)

        question = "\n\n".join(
            f"[{message.role.upper()}]\n{message.content}" for message in prompt.messages
        )
        # BOTH CHANNELS, IN ORDER, WHEN BOTH ARE THERE.
        #
        # This used to be `reasoning or content`, which takes whichever channel
        # is populated and silently drops the other when both are. That is only
        # correct if a model puts its whole chain in exactly one of them, and
        # gpt-5.6-luna does not: measured over one sample in six prompt/
        # reasoning settings, it returned a trace in the reasoning channel for
        # one of them, nothing there but 58-293 tokens of explanation in the
        # content for two more, and eight tokens of bare answer for the rest.
        # So `reasoning or content` fed the metrics the trace alone where both
        # existed -- losing the conclusion the trace was building toward -- and
        # the content alone everywhere else, which reads as two different
        # measurements sharing a column.
        #
        # Concatenated the way the model produced them: the hidden chain first,
        # then what it actually said. Where only one is present it is used as
        # it is, with no separator and nothing implying the other was empty for
        # a reason.
        parts = [part for part in (response.reasoning, response.content) if part and part.strip()]
        return question, "\n\n".join(parts)
