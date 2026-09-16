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
from pathlib import Path
from typing import Any

import orjson

from .adapter import DatasetAdapter
from .batching import iter_chunks
from .client import ModelClient
from .config import ReasoningJudgeConfig
from .errors import ConfigError, EndpointError
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

__all__ = ["ReasoningJudgeStage", "derive_reasoning_metrics", "REASONING_METRIC_COLUMNS"]

#: Adapters label an item with their own task kind.  Everything that asks for a
#: free-form answer is generation-shaped for these metrics, and everything that
#: picks from supplied options is selection-shaped: ``knowledge_completion`` and
#: ``multi_selection`` are not special cases, they are other names for the same
#: two shapes, and eight datasets use them.
_GENERATION_KINDS = frozenset({"generation", "knowledge_completion"})
_SELECTION_KINDS = frozenset({"selection", "multi_selection"})

_REQUIRED_TEMPLATES = {
    "observation_inventory",
    "observation_coverage",
    "steps",
    "branchiness_diversity",
    "redundancy_completeness",
    "directionality",
    "differential_elimination",
    "uncertainty",
    "prior_knowledge",
}

#: Every column this stage can contribute, in reporting order, so the sheet can
#: show a consistent set of columns even where a value is missing.
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
    # 3. branchiness & diversity (generation)
    "reasoning_branchiness",
    "reasoning_diversity",
    # 4. redundancy & completeness
    "reasoning_redundancy",
    "reasoning_completeness",
    "reasoning_redundancy_normalized",
    "reasoning_completeness_normalized",
    # 5. directionality
    "reasoning_directionality",
    # 6. differential elimination (selection)
    "reasoning_differential_elimination",
    "reasoning_differential_elimination_normalized",
    # 7. uncertainty marking
    "reasoning_uncertainty_steps",
    "reasoning_uncertainty_rate",
    # 8. prior knowledge
    "reasoning_prior_knowledge",
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


def comparison_combinations(option_count: int) -> int:
    """Unordered combinations of two or more options, out of ``option_count``.

    The sum of C(n, k) for k = 2..n, which is 2^n - n - 1.  For three options
    that is C(3,2) + C(3,3) = 3 + 1 = 4.
    """
    if option_count < 2:
        return 0
    return (2**option_count) - option_count - 1


def derive_reasoning_metrics(
    raw: dict[str, dict[str, Any]],
    *,
    generation_like: bool,
    selection_like: bool,
    option_count: int,
    pipeline_like: bool = False,
    bov: bool = False,
) -> tuple[dict[str, float], list[str], list[str]]:
    """Validate the judges' raw outputs and compute every derived value.

    Returns ``(metrics, errors, inapplicable)``.  Nothing is coerced: a count
    that is missing, malformed or impossible produces an entry in ``errors`` and
    no metric, and a metric that does not apply to this task shape produces an
    entry in ``inapplicable`` and no metric.  The two are kept apart because
    they mean opposite things to a reader of the sheet.

    Every normalization happens here, after the raw values are in hand, and
    never in a judge prompt.
    """
    metrics: dict[str, float] = {}
    errors: list[str] = []
    inapplicable: list[str] = []

    # -- 1. observation (evidence) coverage --------------------------------- #
    # The total is the one value bought once per sample rather than once per
    # output, so it is the authoritative one here and the normalizer for both
    # this metric and metric 4.
    inventory = raw.get("observation_inventory") or {}
    total_observations = _nonnegative_int(inventory.get("total_observations"))
    if total_observations is None:
        errors.append("observation_inventory:invalid_or_missing_total")
    else:
        metrics["reasoning_observations_total"] = float(total_observations)

    coverage = raw.get("observation_coverage") or {}
    observations_used = _nonnegative_int(coverage.get("observations_used"))
    coverage_total = _nonnegative_int(coverage.get("total_observations"))
    if observations_used is None:
        errors.append("observation_coverage:invalid_or_missing_used_count")
    elif total_observations is None:
        errors.append("observation_coverage:no_inventory_to_count_against")
    elif observations_used > total_observations:
        errors.append("observation_coverage:used_exceeds_inventory_total")
    else:
        if coverage_total is not None and coverage_total != total_observations:
            # Worth recording but not worth voiding the count: the inventory is
            # the authority and the used count was taken against it.
            errors.append("observation_coverage:echoed_total_differs_from_inventory")
        metrics["reasoning_observations_used"] = float(observations_used)
        if total_observations > 0:
            metrics["reasoning_observation_coverage"] = observations_used / total_observations
        else:
            errors.append("observation_coverage:inventory_total_is_zero")

    # -- 2. reasoning steps & backtracking ---------------------------------- #
    steps = raw.get("steps") or {}
    total_steps = _nonnegative_int(steps.get("total_steps"))
    useless_steps = _nonnegative_int(steps.get("useless_steps"))
    useful_steps = _nonnegative_int(steps.get("useful_steps"))
    backtracking = _nonnegative_int(steps.get("backtracking_steps"))
    if (
        total_steps is None
        or useless_steps is None
        or useful_steps is None
        or useful_steps + useless_steps != total_steps
    ):
        # Every step is useful or useless, so a pair that does not add up is a
        # judge that did not segment the chain rather than a partial reading.
        errors.append("steps:invalid_counts_or_sum")
        total_steps = None
    else:
        metrics.update(
            {
                "reasoning_total_steps": float(total_steps),
                "reasoning_useful_steps": float(useful_steps),
                "reasoning_useless_steps": float(useless_steps),
            }
        )
        if total_steps > 0:
            metrics["reasoning_useful_step_fraction"] = useful_steps / total_steps
            metrics["reasoning_useless_step_fraction"] = useless_steps / total_steps
        else:
            errors.append("steps:total_is_zero")

    if backtracking is None:
        errors.append("steps:invalid_or_missing_backtracking_count")
    elif total_steps is not None and backtracking > total_steps:
        # A backtrack is a step, so more backtracks than steps is not a value
        # with a missing normalizer -- it is a judge that did not count. Keeping
        # such a raw number is how one reply of "147" against a 14-step chain
        # once moved this metric's mean by fifty-fold.
        errors.append("steps:backtracking_exceeds_total_steps")
    else:
        if (
            total_steps is not None
            and useless_steps is not None
            and backtracking > useless_steps
        ):
            # Definitionally a backtrack is one of the useless steps. Recorded,
            # not fatal: the count itself is still a count of backtracks.
            errors.append("steps:backtracking_exceeds_useless_steps")
        metrics["reasoning_backtracking_steps"] = float(backtracking)
        if total_steps is not None and total_steps > 0:
            metrics["reasoning_backtracking_rate"] = backtracking / total_steps
        else:
            errors.append("steps:no_step_total_to_rate_backtracking_against")

    # -- 3. branchiness & diversity (generation only) ----------------------- #
    if generation_like or pipeline_like:
        branch = raw.get("branchiness_diversity") or {}
        branchiness = _nonnegative_int(branch.get("branchiness"))
        diversity = _binary(branch.get("diversity"))
        if branchiness is None or diversity is None:
            errors.append("branchiness_diversity:invalid_or_missing_output")
        else:
            metrics["reasoning_branchiness"] = float(branchiness)
            metrics["reasoning_diversity"] = float(diversity)
    else:
        inapplicable.append("branchiness_diversity:generation_only")

    # -- 4. redundancy & completeness --------------------------------------- #
    redundancy_raw = raw.get("redundancy_completeness") or {}
    redundant = _nonnegative_int(redundancy_raw.get("redundancy"))
    complete = _nonnegative_int(redundancy_raw.get("completeness"))
    if redundant is None or complete is None:
        errors.append("redundancy_completeness:invalid_or_missing_output")
    else:
        metrics["reasoning_redundancy"] = float(redundant)
        metrics["reasoning_completeness"] = float(complete)
        if total_observations is None:
            errors.append("redundancy_completeness:no_inventory_to_normalize_against")
        elif total_observations == 0:
            errors.append("redundancy_completeness:inventory_total_is_zero")
        elif redundant > total_observations or complete > total_observations:
            errors.append("redundancy_completeness:count_exceeds_inventory_total")
        else:
            metrics["reasoning_redundancy_normalized"] = redundant / total_observations
            metrics["reasoning_completeness_normalized"] = complete / total_observations

    # -- 5. directionality --------------------------------------------------- #
    directionality = _nonnegative_number((raw.get("directionality") or {}).get("directionality"))
    if directionality not in (0.0, 0.5, 1.0):
        errors.append("directionality:expected_0_0.5_or_1")
    else:
        metrics["reasoning_directionality"] = directionality

    # -- 6. differential elimination (selection, and pipeline) --------------- #
    if selection_like or pipeline_like:
        if bov:
            # A BOV request shows one candidate per call, so there is no set of
            # options in the chain for combinations to be drawn from.
            inapplicable.append("differential_elimination:bov_shows_one_option_per_chain")
        elif option_count < 2:
            errors.append("differential_elimination:fewer_than_two_visible_options")
        else:
            differential = _nonnegative_int(
                (raw.get("differential_elimination") or {}).get("differential_elimination")
            )
            combinations = comparison_combinations(option_count)
            if differential is None:
                errors.append("differential_elimination:invalid_or_missing_output")
            elif differential > combinations:
                errors.append("differential_elimination:exceeds_possible_combinations")
            else:
                metrics["reasoning_differential_elimination"] = float(differential)
                metrics["reasoning_differential_elimination_normalized"] = (
                    differential / combinations
                )
    else:
        inapplicable.append("differential_elimination:selection_and_pipeline_only")

    # -- 7. uncertainty marking ---------------------------------------------- #
    uncertainty = _nonnegative_int((raw.get("uncertainty") or {}).get("uncertainty_steps"))
    if uncertainty is None:
        errors.append("uncertainty:invalid_or_missing_output")
    elif total_steps is not None and uncertainty > total_steps:
        # Same rule as backtracking: it counts steps, so it cannot exceed them.
        errors.append("uncertainty:exceeds_total_steps")
    else:
        metrics["reasoning_uncertainty_steps"] = float(uncertainty)
        if total_steps is not None and total_steps > 0:
            metrics["reasoning_uncertainty_rate"] = uncertainty / total_steps
        else:
            errors.append("uncertainty:no_step_total_to_rate_against")

    # -- 8. prior knowledge --------------------------------------------------- #
    prior = _binary((raw.get("prior_knowledge") or {}).get("prior_knowledge"))
    if prior is None:
        errors.append("prior_knowledge:invalid_or_missing_output")
    else:
        metrics["reasoning_prior_knowledge"] = float(prior)

    return metrics, errors, inapplicable


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
        #: Observation inventories currently being bought, by question key.
        #: The inventory is the one purchase shared between tasks, and the
        #: obvious way to stop two of them paying for the same question twice
        #: -- hold a lock across the whole inventory step -- serializes every
        #: task's first wave behind every other task's. This lets tasks asking
        #: *different* questions proceed at once while a task asking one
        #: already in flight waits on that same answer.
        self._inventories_inflight: dict[str, asyncio.Future] = {}
        self._calls = calls or asyncio.Semaphore(config.max_parallel_calls)
        #: A standalone JSONL log of these metrics alone, one line per judged
        #: output, written next to the records.  The sheet is the report and
        #: records.jsonl is the evidence; this is the audit trail for this
        #: stage specifically -- every raw count, every derived value, and the
        #: status that explains any blank -- and it lands inside the run
        #: directory, so `engine.sync` mirrors it off-box with everything else.
        self.log_path = Path(log_path) if log_path else None
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
                    answer=(score.prediction or response.text or "")[:2000],
                    reference=json.dumps(sample.reference, ensure_ascii=False, default=str)[:2000],
                    options=options,
                    generation_like=generation_like,
                    selection_like=selection_like,
                    pipeline_like=pipeline,
                    bov=identity.selection_mode == BOV,
                )
            )

        if not targets:
            return updated

        await self._evaluate(targets)
        async with self._lock:
            self._save_cache()

        # Raw values for every family are in hand before a single normalized
        # value is computed -- including the per-sample observation total, which
        # metric 1 and metric 4 both divide by.
        log_lines: list[dict[str, Any]] = []
        for target in targets:
            metrics, errors, inapplicable = derive_reasoning_metrics(
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

    async def _evaluate(self, targets: list[_Target]) -> None:
        inventory_requests: dict[str, dict[str, Any]] = {}
        question_keys: dict[int, str] = {}
        for target in targets:
            question_key = stable_hash({"question": target.question}, length=32)
            question_keys[target.index] = question_key
            inventory_requests.setdefault(question_key, {"question": target.question})

        # The inventory is the one shared purchase: it depends on the question
        # alone, so it is bought once per distinct question and reused for every
        # model, repeat and task that asks it.
        inventories = await self._inventories(inventory_requests)

        common: dict[int, dict[str, Any]] = {}
        for target in targets:
            inventory = inventories.get(question_keys[target.index])
            if inventory is None:
                target.errors.append("observation_inventory:judge_failed_or_unparseable")
                inventory = {}
            target.raw["observation_inventory"] = inventory
            common[target.index] = {
                "question": target.question,
                "reasoning_chain": target.reasoning,
                "model_answer": target.answer,
                "reference_answer": target.reference,
                "options": target.options,
                "option_count": len(target.options),
                "total_observations": _nonnegative_int(inventory.get("total_observations")),
            }

        async def run_family(
            family: str, selected: list[_Target], field_names: tuple[str, ...]
        ) -> None:
            template = self.templates[family]
            requests = {
                str(target.index): {
                    key: common[target.index][key]
                    for key in field_names
                    if common[target.index].get(key) is not None
                }
                for target in selected
            }
            # A missing dependency is reported, never rendered into the prompt
            # as a made-up zero.
            callable_requests = {
                key: fields
                for key, fields in requests.items()
                if all(required in fields for required in template.required_fields)
            }
            results = await self._judge_many(family, callable_requests)
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

        generation = [t for t in targets if t.generation_like or t.pipeline_like]
        differential = [
            t
            for t in targets
            if (t.selection_like or t.pipeline_like) and not t.bov and len(t.options) >= 2
        ]

        # Wave one: everything that needs at most the observation inventory.
        # Seven families' batches are in flight together here.
        wave_one = [
            run_family(
                "observation_coverage", targets, ("question", "reasoning_chain",
                                                  "total_observations")
            ),
            run_family(
                "redundancy_completeness",
                targets,
                ("question", "reasoning_chain", "model_answer", "reference_answer",
                 "total_observations"),
            ),
            run_family("steps", targets, ("question", "reasoning_chain", "options")),
            run_family("directionality", targets, ("question", "reasoning_chain")),
            run_family("prior_knowledge", targets, ("question", "reasoning_chain")),
        ]
        if generation:
            wave_one.append(
                run_family("branchiness_diversity", generation, ("question", "reasoning_chain"))
            )
        if differential:
            wave_one.append(
                run_family(
                    "differential_elimination",
                    differential,
                    ("question", "reasoning_chain", "options", "option_count"),
                )
            )
        await asyncio.gather(*wave_one)

        # Wave two: uncertainty counts steps, so it has to see the same step
        # count metric 2 produced -- the one thing here that genuinely waits.
        for target in targets:
            common[target.index]["total_steps"] = _nonnegative_int(
                (target.raw.get("steps") or {}).get("total_steps")
            )
        await run_family("uncertainty", targets, ("question", "reasoning_chain", "total_steps"))

    async def _inventories(
        self, requests: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any] | None]:
        """Buy each question's inventory once, without serializing the tasks.

        Deduplication is per question rather than per step: a task claims the
        questions nobody else is buying, fetches exactly those, and awaits the
        rest on the futures whoever claimed them will resolve.
        """
        loop = asyncio.get_running_loop()
        mine: dict[str, dict[str, Any]] = {}
        waiting: dict[str, asyncio.Future] = {}
        async with self._lock:
            for key, fields in requests.items():
                pending = self._inventories_inflight.get(key)
                if pending is not None:
                    waiting[key] = pending
                else:
                    self._inventories_inflight[key] = loop.create_future()
                    mine[key] = fields

        out: dict[str, dict[str, Any] | None] = {}
        if mine:
            try:
                out.update(await self._judge_many("observation_inventory", mine))
            finally:
                # Resolve every claimed key even on failure, or the tasks
                # waiting on this question would hang for the whole run.
                async with self._lock:
                    for key in mine:
                        future = self._inventories_inflight.pop(key, None)
                        if future is not None and not future.done():
                            future.set_result(out.get(key))
        for key, future in waiting.items():
            out[key] = await future
        return out

    # -- one family's calls -------------------------------------------------- #

    async def _judge_many(
        self, family: str, requests: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any] | None]:
        template = self.templates[family]
        keyed: list[tuple[str, dict[str, Any], str]] = []
        out: dict[str, dict[str, Any] | None] = {}
        for request_id, fields in requests.items():
            key = stable_hash({"template": template.ref, "fields": fields}, length=32)
            cached = self._cache.get(key)
            if cached is not None:
                values = cached.get("values")
                out[request_id] = values if isinstance(values, dict) else None
                self.stats["cached"] += 1
            else:
                keyed.append((request_id, fields, key))

        # A group is always ``group_size`` requests, whether or not the server
        # has a batch route, so in-flight judge sequences stay at
        # ``group_size x max_parallel_calls`` either way. A server without one
        # gets the group as that many concurrent single calls inside one
        # semaphore slot; dropping the group size to 1 instead -- which is the
        # obvious thing to do -- would quietly cut the stage's concurrency by a
        # factor of group_size on exactly the servers that are slowest already.
        can_batch = self.client.supports_batch and self.config.model not in self._batch_disabled

        def record(request_id: str, key: str, raw: str) -> None:
            values = self._parse_json(raw, template)
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

        async def run_chunk(chunk: list[tuple[str, dict[str, Any], str]]) -> None:
            conversations = []
            for _request_id, fields, key in chunk:
                sample = SampleSpec(
                    sample_id=f"reasoning-judge::{key}", fields=fields, task_kind="judge"
                )
                messages, _contract = self.renderer.render(sample, template)
                conversations.append(messages)
            sampling = SamplingParams(
                max_tokens=self.config.max_tokens, temperature=self.config.temperature
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
                        return
                    for (request_id, _fields, key), choice in zip(
                        chunk, result.choices, strict=True
                    ):
                        record(request_id, key, choice.content or "")
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
                        return
                    choice = result.choices[0] if result.choices else None
                    if choice is None:
                        out[request_id] = None
                        self.stats["failed"] += 1
                        return
                    record(request_id, key, choice.content or "")

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
            return None if any(name not in parsed for name in fields) else parsed

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
        # Servers that expose a separate reasoning channel put the chain there;
        # the rest put it in the content, ahead of the answer line.
        reasoning = response.reasoning or response.content or ""
        return question, reasoning
