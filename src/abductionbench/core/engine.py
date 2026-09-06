"""The evaluation engine: plan, execute, score, persist.

Execution model
---------------
Work is decomposed into three stages so that results stay comparable and cheap
to resume:

**Stage A -- dataset bundles** (once per dataset).  The adapter is resolved from
config, prepared, and asked for its deterministic sample.  An adapter that
raises ``SkippedDataset`` is recorded as skipped and the run continues.

**Stage B -- prompt sets** (once per dataset x prompt-template variant).
Samples are rendered through the bound template and checked against the
input-token budget, replacing oversize items with fresh draws from the same
split.  This stage is deliberately *model-independent*: every model then sees
exactly the same prompts, which is what makes cross-model numbers comparable.

**Stage C -- tasks** (dataset x variant x model).  Sampling params are resolved
against that model's limits, samples are packed into native batch calls of that
model's own group size, submitted with retry/recovery, scored by the adapter and
appended to ``records.jsonl`` as each batch lands.

Failure handling worth knowing about
------------------------------------
* A batch call that fails with an invalid-request/context-length error is
  **bisected**: vLLM rejects the whole call because of one bad conversation, so
  the engine splits the batch to isolate the offender and salvages the rest.
* A connection-level failure triggers **endpoint recovery** (re-run the
  configured discovery command -- quick-tunnel URLs rotate -- then probe
  ``/v1/models``) before the next attempt.
* Everything is checkpointed per sample, so a dropped connection costs at most
  the in-flight batches.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from . import batching as batching_mod
from .adapter import AdapterContext, DatasetAdapter, SkippedDataset
from .checkpoint import RecordStore, TaskCheckpoint
from .client import BatchResult, ModelClient, RawChoice
from .config import DatasetConfig, ModelConfig, RunConfig, dump_resolved
from .errors import AbenchError, AdapterError, AuthError, ErrorClass, TemplateError
from .judge import JudgeStage
from .metrics import mean
from .modes import SELF_CONSISTENCY, TaskModes
from .prompts import PromptRegistry, PromptRenderer, PromptTemplate
from .registry import resolve_adapter
from .retry import RetryPolicy, summarize, with_retry
from .sync import ArtifactSync
from .telemetry import EventLog, clip, setup_logging
from .tokenizer import build_token_counter
from .types import (
    AdapterDocumentation,
    ChatMessage,
    EvalRecord,
    ModelResponse,
    RenderedPrompt,
    ResponseStatus,
    SampleScore,
    SampleSpec,
    SamplingParams,
    TaskIdentity,
)

logger = logging.getLogger(__name__)

__all__ = ["EvaluationEngine", "RunResult", "TaskResult", "DatasetBundle", "PromptSet"]


# --------------------------------------------------------------------------- #
# Plan / result containers
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class DatasetBundle:
    """A prepared dataset: its adapter, its sample, its self-documentation."""

    config: DatasetConfig
    adapter: DatasetAdapter | None
    modes: TaskModes = field(default_factory=TaskModes)
    samples: list[SampleSpec] = field(default_factory=list)
    documentation: AdapterDocumentation | None = None
    skipped_reason: str | None = None
    prepare_seconds: float = 0.0

    @property
    def key(self) -> tuple[str, str]:
        return (self.config.id, self.modes.slug)

    @property
    def skipped(self) -> bool:
        return self.skipped_reason is not None


@dataclass(slots=True)
class PromptSet:
    """Rendered prompts for one (dataset, template variant), shared by models."""

    dataset_id: str
    variant: str
    modes: TaskModes
    template: PromptTemplate
    entries: list[tuple[SampleSpec, list[ChatMessage], int]] = field(default_factory=list)
    output_contract: dict[str, Any] = field(default_factory=dict)
    oversize_dropped: list[str] = field(default_factory=list)
    replacements_used: int = 0
    unusable_reason: str | None = None

    @property
    def size(self) -> int:
        return len(self.entries)


@dataclass(slots=True)
class TaskResult:
    """Outcome of one (dataset, model, template) evaluation unit."""

    identity: TaskIdentity
    output_dir: Path
    metrics: dict[str, float] = field(default_factory=dict)
    checkpoint: TaskCheckpoint | None = None
    n_planned: int = 0
    n_scored: int = 0
    n_error: int = 0
    n_skipped: int = 0
    n_reused: int = 0
    primary_metric: str = ""
    documentation: AdapterDocumentation | None = None
    failure: str | None = None
    duration_s: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RunResult:
    """Everything a run produced -- the input to reporting."""

    run_id: str
    run_dir: Path
    config: RunConfig
    tasks: list[TaskResult] = field(default_factory=list)
    skipped_datasets: list[dict[str, str]] = field(default_factory=list)
    #: (dataset, mode, reason) for every requested mode a dataset does not admit.
    skipped_modes: list[dict[str, str]] = field(default_factory=list)
    #: Additional generation/selection tasks introduced beyond the dataset
    #: table, each with the benchmark formulation that justifies it.
    introduced_modes: list[dict[str, str]] = field(default_factory=list)
    endpoint_reports: list[dict[str, Any]] = field(default_factory=list)
    sync_stats: dict[str, Any] = field(default_factory=dict)
    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def duration_s(self) -> float:
        return max(0.0, self.finished_at - self.started_at)


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #


class EvaluationEngine:
    """Runs one :class:`~abductionbench.core.config.RunConfig` end to end."""

    def __init__(
        self,
        config: RunConfig,
        *,
        run_id: str | None = None,
        run_dir: Path | None = None,
        offline: bool = False,
        dry_run: bool = False,
    ):
        self.config = config
        self.engine_cfg = config.engine
        self.offline = offline
        self.dry_run = dry_run
        self.run_id = run_id or self._default_run_id(config.name)
        self.run_dir = Path(run_dir) if run_dir else Path(self.engine_cfg.output_root) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        setup_logging(
            self.run_dir,
            level=self.engine_cfg.logging.level,
            json_log=self.engine_cfg.logging.json_log,
        )
        self.events = EventLog(self.run_dir / "events.jsonl")
        # Off-box backup runs in its own thread and shells out to rclone, so it
        # never touches the event loop that drives inference.
        self.sync = ArtifactSync(
            self.engine_cfg.sync,
            self.run_dir,
            self.run_id,
            on_event=self.events.emit,
        )
        self.registry = PromptRegistry(list(config.prompts.template_dirs))
        self.renderer = PromptRenderer(self.registry, config.prompts)
        tokenizer_cfg = self.engine_cfg.tokenizer
        if not tokenizer_cfg.hf_model and config.models:
            # Count with the tokenizer of the model being evaluated. Nothing
            # else can be exact, and an inexact count against a context window
            # is what makes a request overshoot it.
            tokenizer_cfg = tokenizer_cfg.model_copy(
                update={"hf_model": config.models[0].model_name}
            )
        self.token_counter = build_token_counter(tokenizer_cfg)
        self.retry_policy = RetryPolicy(self.engine_cfg.retry)
        self._clients: dict[str, ModelClient] = {}
        self._global_batch_sem = asyncio.Semaphore(
            self.engine_cfg.concurrency.max_parallel_batches_global
        )
        self._model_sems: dict[str, asyncio.Semaphore] = {}
        self._scoring_sem = asyncio.Semaphore(self.engine_cfg.concurrency.scoring_workers)
        self._batch_disabled: set[str] = set()
        #: Mode combinations a dataset declined, reported rather than dropped.
        self._skipped_modes: list[dict[str, str]] = []
        #: Hypothesis modes run beyond what the dataset table lists, with the
        #: benchmark formulation that justifies each (specification item 15).
        self._introduced_modes: list[dict[str, str]] = []

    def _run_id_tz(self):
        """The timezone a new run id is stamped in.

        Falls back to UTC with a warning rather than failing a run over a
        timezone name -- a run that cannot start is worse than one named in the
        wrong timezone, and the fallback says which happened.
        """
        name = (self.engine_cfg.run_id_timezone or "UTC").strip()
        if name.upper() == "UTC":
            return timezone.utc
        try:
            return ZoneInfo(name)
        except Exception as exc:  # noqa: BLE001 - unknown zone, or no tzdata
            logger.warning(
                "engine.run_id_timezone=%r is not a usable timezone (%s); "
                "stamping run ids in UTC",
                name, exc,
            )
            return timezone.utc

    def _default_run_id(self, name: str) -> str:
        stamp = datetime.now(self._run_id_tz()).strftime("%Y%m%d-%H%M%S")
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in name).strip("-")
        return f"{stamp}_{safe or 'run'}"

    # ------------------------------------------------------------------ #
    # entry point
    # ------------------------------------------------------------------ #

    async def run(self) -> RunResult:
        """Execute the whole run and return its result."""
        result = RunResult(
            run_id=self.run_id, run_dir=self.run_dir, config=self.config, started_at=time.time()
        )
        dump_resolved(self.config, self.run_dir / "run_config.resolved.yaml")
        self.sync.start()
        logger.info(
            "run %s: %d model(s) x %d dataset(s), output=%s",
            self.run_id,
            len(self.config.models),
            len(self.config.enabled_datasets()),
            self.run_dir,
        )
        self.events.emit(
            "run_started",
            run_id=self.run_id,
            models=[m.id for m in self.config.models],
            datasets=[d.id for d in self.config.enabled_datasets()],
            dry_run=self.dry_run,
        )

        try:
            # --- endpoints ------------------------------------------------ #
            for model in self.config.models:
                client = ModelClient(
                    model,
                    self.engine_cfg.timeouts,
                    rediscover=self.engine_cfg.retry.recovery.rediscover_on_connection_error,
                )
                self._clients[model.id] = client
                self._model_sems[model.id] = asyncio.Semaphore(model.limits.max_parallel_batches)

            if not self.dry_run:
                result.endpoint_reports = await self._verify_endpoints()

            # --- Stage A: dataset bundles --------------------------------- #
            bundles = self._build_bundles()
            for bundle in bundles:
                if bundle.skipped:
                    result.skipped_datasets.append(
                        {"dataset_id": bundle.config.id, "reason": bundle.skipped_reason or ""}
                    )

            # --- Stage B: prompt sets ------------------------------------- #
            prompt_sets: dict[tuple[str, str], PromptSet] = {}
            for bundle in bundles:
                if bundle.skipped or bundle.adapter is None:
                    continue
                try:
                    prompt_set = self._build_prompt_set(bundle)
                except (TemplateError, AdapterError) as exc:
                    logger.error(
                        "dataset %s [%s]: prompt rendering failed: %s",
                        bundle.config.id,
                        bundle.modes.slug,
                        exc,
                    )
                    result.skipped_datasets.append(
                        {
                            "dataset_id": bundle.config.id,
                            "reason": f"prompt rendering failed ({bundle.modes.slug}): {exc}",
                        }
                    )
                    continue
                prompt_sets[bundle.key] = prompt_set

            result.skipped_modes = list(self._skipped_modes)
            result.introduced_modes = list(self._introduced_modes)

            # --- Stage C: tasks ------------------------------------------- #
            tasks = self._plan_tasks(bundles, prompt_sets)
            logger.info("planned %d task(s)", len(tasks))
            if self.dry_run:
                for identity, prompt_set, _model, _bundle in tasks:
                    logger.info(
                        "[dry-run] %s: %d prompts, template=%s, mean input tokens=%.0f",
                        identity.slug,
                        prompt_set.size,
                        prompt_set.template.ref,
                        mean([tokens for _, _, tokens in prompt_set.entries]),
                    )
                    result.tasks.append(
                        TaskResult(
                            identity=identity,
                            output_dir=self._task_dir(identity),
                            n_planned=prompt_set.size,
                            primary_metric="",
                            documentation=None,
                        )
                    )
                result.finished_at = time.time()
                return result

            semaphore = asyncio.Semaphore(self.engine_cfg.concurrency.max_parallel_tasks)

            async def _guarded(identity, prompt_set, model, bundle) -> TaskResult:
                async with semaphore:
                    return await self._run_task(identity, prompt_set, model, bundle)

            gathered = await asyncio.gather(
                *(_guarded(*task) for task in tasks), return_exceptions=True
            )
            for task, outcome in zip(tasks, gathered, strict=True):
                identity = task[0]
                if isinstance(outcome, TaskResult):
                    result.tasks.append(outcome)
                elif isinstance(outcome, BaseException):
                    logger.exception("task %s crashed: %s", identity.slug, outcome)
                    result.tasks.append(
                        TaskResult(
                            identity=identity,
                            output_dir=self._task_dir(identity),
                            failure=f"{type(outcome).__name__}: {outcome}",
                        )
                    )
        finally:
            for client in self._clients.values():
                await client.aclose()
            result.finished_at = time.time()
            # One last upload of everything the run produced.  Reports are
            # written after this returns, so the CLI calls flush_sync() again.
            sync_stats = self.sync.stop()
            result.sync_stats = sync_stats.as_dict() if self.engine_cfg.sync.enabled else {}
            self.events.emit(
                "run_finished",
                run_id=self.run_id,
                tasks=len(result.tasks),
                skipped=len(result.skipped_datasets),
                duration_s=round(result.duration_s, 1),
            )
        return result

    def flush_sync(self) -> dict[str, Any]:
        """Upload once more, after reports have been written."""
        if not self.engine_cfg.sync.enabled:
            return {}
        return self.sync.flush().as_dict()

    # ------------------------------------------------------------------ #
    # endpoint verification
    # ------------------------------------------------------------------ #

    async def _verify_endpoints(self) -> list[dict[str, Any]]:
        """Probe every model before spending time on data preparation."""
        reports: list[dict[str, Any]] = []
        for model_id, client in self._clients.items():
            try:
                report = await client.verify()
            except AuthError as exc:
                raise AbenchError(
                    f"model {model_id!r}: authentication failed against "
                    f"{client.base_url} -- check the API key ({exc})"
                ) from exc
            reports.append(report)
            self.events.emit("endpoint_verified", **report)
            if not report.get("model_available"):
                logger.warning(
                    "model %s (%s) is not in the endpoint's model list (%s); "
                    "requests may fail",
                    model_id,
                    client.model.model_name,
                    report.get("served_models"),
                )
            if not report.get("batch_ok"):
                error_class = report.get("batch_error_class")
                structural = error_class in (
                    ErrorClass.INVALID_REQUEST.value,
                    ErrorClass.CONTEXT_LENGTH.value,
                    ErrorClass.PROTOCOL.value,
                    "config",
                )
                if not structural:
                    # Transient probe failure (busy server, tunnel hiccup): keep
                    # batch mode on and let per-batch retry/recovery handle it,
                    # rather than silently downgrading the whole run.
                    logger.warning(
                        "model %s: batch probe failed transiently (%s); keeping batch mode "
                        "enabled and relying on per-batch retry/recovery",
                        model_id,
                        clip(report.get("batch_error"), 200),
                    )
                elif self.engine_cfg.batching.fallback_to_single:
                    logger.warning(
                        "model %s: batch endpoint unusable (%s); falling back to single calls",
                        model_id,
                        clip(report.get("batch_error"), 200),
                    )
                    self._batch_disabled.add(model_id)
                else:
                    raise AbenchError(
                        f"model {model_id!r}: batch endpoint unusable and "
                        "engine.batching.fallback_to_single is false: "
                        f"{report.get('batch_error')}"
                    )
            else:
                logger.info(
                    "model %s: batch endpoint OK (%s), group size %d",
                    model_id,
                    client.batch_url,
                    client.model.endpoint.batch.group_size,
                )
        return reports

    # ------------------------------------------------------------------ #
    # Stage A
    # ------------------------------------------------------------------ #

    def _modes_for(self, dataset_cfg: DatasetConfig) -> list[TaskModes]:
        """Every mode combination this dataset will be evaluated in.

        The prompt modes come from the run config; the selection modes come from
        the run config *intersected with what the dataset's task definition
        admits*, so a benchmark whose items have several correct hypotheses is
        never asked to pick one.  A dataset that is not a selection task gets a
        single mode with no selection axis.  Delivery is read from the adapter:
        it is a property of the benchmark, not a choice.

        Combinations the adapter rejects are logged once, with the reason, and
        recorded on the run so the report can say which modes were not run.
        """
        cfg = self.config.modes
        try:
            adapter_cls = resolve_adapter(dataset_cfg.impl)
        except Exception:  # noqa: BLE001 - a bad impl is reported when it is instantiated
            return [TaskModes()]

        requested_selection: list[str | None]
        offered = adapter_cls.selection_modes_offered()
        if not offered:
            requested_selection = [None]
        elif cfg.selection_modes:
            requested_selection = [m for m in cfg.selection_modes] or [None]
        else:
            # No explicit request: run the mode the benchmark itself defines.
            requested_selection = [offered[0]]

        # "Generation / Selection (separate tasks)" in the dataset table means two
        # independent evaluations, so they are crossed here rather than mixed
        # inside one task.
        hypothesis_modes: list[str | None] = list(adapter_cls.hypothesis_modes) or [None]
        if cfg.hypothesis_modes:
            wanted = [m for m in hypothesis_modes if m in cfg.hypothesis_modes]
            hypothesis_modes = wanted or hypothesis_modes[:1]
        if len(hypothesis_modes) == 1 and not adapter_cls.hypothesis_mode_options:
            # Only one task and no option to switch: leave the axis unset so the
            # slug and the columns stay uncluttered for single-task datasets.
            hypothesis_modes = [None]

        out: list[TaskModes] = []
        for prompt_mode in cfg.prompt_modes:
            for hypothesis_mode in hypothesis_modes:
                for selection_mode in requested_selection:
                    if hypothesis_mode == "generation" and selection_mode is not None:
                        # Generating a hypothesis has no candidate list to pick from.
                        selection_mode = None
                    modes = TaskModes(
                        prompt_mode=prompt_mode,
                        selection_mode=selection_mode,
                        hypothesis_mode=hypothesis_mode,
                        # A dataset config may run an interactive benchmark in
                        # its static form as an ablation ("how much does the
                        # interaction actually buy?"); the benchmark's own mode
                        # is the default.
                        data_delivery_mode=str(
                            dataset_cfg.options.get("delivery")
                            or adapter_cls.data_delivery_mode
                        ),
                        self_consistency_n=cfg.self_consistency_n,
                        self_consistency_temperature=cfg.self_consistency_temperature,
                    )
                    if any(m.slug == modes.slug for m in out):
                        continue
                    introduced = (
                        adapter_cls.introduced_hypothesis_mode(hypothesis_mode)
                        if hypothesis_mode else None
                    )
                    if introduced:
                        logger.info(
                            "dataset %s: running an additional hypothesis mode (%s) that the "
                            "dataset table does not list. Justification from the benchmark: %s",
                            dataset_cfg.id, hypothesis_mode, introduced,
                        )
                        self._introduced_modes.append(
                            {
                                "dataset_id": dataset_cfg.id,
                                "mode": modes.slug,
                                "hypothesis_mode": hypothesis_mode,
                                "table_says": adapter_cls.table_hypothesis_mode,
                                "justification": introduced,
                            }
                        )
                    problem = adapter_cls.supports_modes(modes)
                    if problem:
                        logger.info("dataset %s: mode %s not run -- %s",
                                    dataset_cfg.id, modes.slug, problem)
                        self._skipped_modes.append(
                            {"dataset_id": dataset_cfg.id, "mode": modes.slug, "reason": problem}
                        )
                        continue
                    out.append(modes)
        if not out:
            # Everything requested was inadmissible; fall back to the plain mode
            # so the dataset is still evaluated rather than silently dropped.
            out.append(TaskModes(data_delivery_mode=adapter_cls.data_delivery_mode))
        return out

    def _build_bundles(self) -> list[DatasetBundle]:
        bundles: list[DatasetBundle] = []
        for dataset_cfg in self.config.enabled_datasets():
            for modes in self._modes_for(dataset_cfg):
                bundles.append(self._prepare_bundle(dataset_cfg, modes))
        return bundles

    def _prepare_bundle(self, dataset_cfg: DatasetConfig, modes: TaskModes) -> DatasetBundle:
        """Prepare, sample and document one dataset in one mode combination."""
        started = time.time()
        bundle = DatasetBundle(config=dataset_cfg, adapter=None, modes=modes)
        try:
            adapter = self._instantiate_adapter(dataset_cfg, modes)
            adapter.prepare()
            samples = list(adapter.build_samples())
            if not samples:
                raise SkippedDataset("adapter produced no samples")
            # One evaluation item can need several requests: BOV asks about each
            # hypothesis separately, self-consistency asks k times. Expanding
            # here rather than inside a base class means every adapter gets the
            # modes, including one that builds its samples its own way.
            samples = adapter.expand_for_modes(samples)
            # A mode may derive several requests per item (BOV asks one
            # question per hypothesis, self-consistency asks k times), so
            # the size guard counts items, not requests.
            items = len({s.group_id or s.sample_id for s in samples})
            if items > dataset_cfg.sample_size:
                logger.warning(
                    "dataset %s: adapter returned %d samples for a requested size of %d; "
                    "truncating to the requested size",
                    dataset_cfg.id,
                    items,
                    dataset_cfg.sample_size,
                )
                keep = list(dict.fromkeys(
                    s.group_id or s.sample_id for s in samples
                ))[: dataset_cfg.sample_size]
                allowed = set(keep)
                samples = [s for s in samples if (s.group_id or s.sample_id) in allowed]
            seen: set[str] = set()
            unique: list[SampleSpec] = []
            for sample in samples:
                if sample.sample_id in seen:
                    logger.warning(
                        "dataset %s: duplicate sample_id %r dropped",
                        dataset_cfg.id,
                        sample.sample_id,
                    )
                    continue
                seen.add(sample.sample_id)
                unique.append(sample)
            bundle.adapter = adapter
            bundle.samples = unique
            bundle.documentation = adapter.documentation()
            bundle.documentation.statistics.setdefault("n_samples_built", len(unique))
            bundle.documentation.statistics.setdefault(
                "requested_sample_size", dataset_cfg.sample_size
            )
            if len(unique) < dataset_cfg.sample_size:
                bundle.documentation.caveats.append(
                    f"Only {len(unique)} samples were available in the chosen split "
                    f"(requested {dataset_cfg.sample_size})."
                )
            logger.info(
                "dataset %s: prepared %d sample(s) in %.1fs",
                dataset_cfg.id,
                len(unique),
                time.time() - started,
            )
            self.events.emit(
                "dataset_prepared",
                dataset_id=dataset_cfg.id,
                n_samples=len(unique),
                seconds=round(time.time() - started, 2),
            )
        except SkippedDataset as exc:
            bundle.skipped_reason = exc.reason
            logger.warning("dataset %s skipped: %s", dataset_cfg.id, exc.reason)
            self.events.emit("dataset_skipped", dataset_id=dataset_cfg.id, reason=exc.reason)
        except Exception as exc:  # noqa: BLE001 - one bad dataset must not stop the run
            bundle.skipped_reason = f"{type(exc).__name__}: {exc}"
            logger.exception("dataset %s failed during preparation", dataset_cfg.id)
            self.events.emit(
                "dataset_skipped", dataset_id=dataset_cfg.id, reason=bundle.skipped_reason
            )
        bundle.prepare_seconds = time.time() - started
        return bundle

    def _instantiate_adapter(
        self, dataset_cfg: DatasetConfig, modes: TaskModes
    ) -> DatasetAdapter:
        adapter_cls = resolve_adapter(dataset_cfg.impl)
        data_dir = Path(self.engine_cfg.data_root) / dataset_cfg.id
        data_dir.mkdir(parents=True, exist_ok=True)
        context = AdapterContext(
            dataset_id=dataset_cfg.id,
            data_dir=data_dir,
            modes=modes,
            sample_size=dataset_cfg.sample_size,
            seed=dataset_cfg.seed if dataset_cfg.seed is not None else self.config.seed,
            options={
                **dict(dataset_cfg.options),
                # The adapter's own mapping from hypothesis mode to whatever
                # option its dataset uses to switch task.
                **(
                    resolve_adapter(dataset_cfg.impl).hypothesis_mode_options.get(
                        modes.hypothesis_mode or "", {}
                    )
                    if modes.hypothesis_mode
                    else {}
                ),
            },
            input_token_budget=dataset_cfg.input_token_budget
            or self.engine_cfg.limits.input_token_budget,
            offline=self.offline,
            cache_dir=Path(self.engine_cfg.data_root) / "_cache",
            logger=logging.getLogger(f"adapter.{dataset_cfg.id}"),
        )
        return adapter_cls(context)

    # ------------------------------------------------------------------ #
    # Stage B
    # ------------------------------------------------------------------ #

    def _build_prompt_set(self, bundle: DatasetBundle) -> PromptSet:
        """Ask the adapter to render its samples, and enforce the input-token budget.

        The engine no longer owns prompt wording: it calls the adapter's
        ``build_messages`` for every sample and takes back both the conversation
        and the contract its scorer will parse.  What is still the engine's job
        is what it can do generically -- counting input tokens and replacing
        oversize samples with fresh draws from the same split
        (``engine.limits.on_oversize == "resample"``) so the evaluation set keeps
        its configured size instead of silently shrinking.
        """
        assert bundle.adapter is not None
        dataset_cfg = bundle.config
        adapter = bundle.adapter
        # An interactive episode's context is a transcript, not a prompt, so its
        # budget is set by how the dataset is delivered unless the dataset names
        # its own. Oversize interactive items are dropped rather than replaced.
        delivery = bundle.modes.data_delivery_mode
        budget = dataset_cfg.input_token_budget or self.engine_cfg.limits.budget_for(delivery)
        policy = self.engine_cfg.limits.oversize_policy_for(delivery)
        target = len(bundle.samples)
        modes = bundle.modes

        prompt_set = PromptSet(
            dataset_id=dataset_cfg.id,
            variant=modes.slug,
            modes=modes,
            template=PromptTemplate(
                # The "template" is now the mode identity plus the adapter that
                # owns the wording, so a record still names what produced it.
                id=modes.slug,
                version=adapter.adapter_version,
                messages=[{"role": "system", "content": "(owned by the dataset adapter)"}],
                description=f"{dataset_cfg.id}: {modes.describe()}",
                task_kinds=sorted({sample.task_kind for sample in bundle.samples}),
            ),
            output_contract={},
        )

        queue: list[SampleSpec] = list(bundle.samples)
        attempted: set[str] = {s.sample_id for s in bundle.samples}
        replacement_attempts = 0
        oversize_seen = 0

        while queue:
            sample = queue.pop(0)
            messages, contract = adapter.build_messages(sample)
            tokens = self.token_counter.count_messages(messages)
            if tokens <= budget:
                prompt_set.entries.append((sample, messages, tokens))
                sample.metadata["_output_contract"] = contract
                continue

            oversize_seen += 1
            prompt_set.oversize_dropped.append(sample.sample_id)
            logger.info(
                "dataset %s: sample %s needs %d input tokens (> budget %d)",
                dataset_cfg.id,
                sample.sample_id,
                tokens,
                budget,
            )
            self.events.emit(
                "sample_oversize",
                dataset_id=dataset_cfg.id,
                sample_id=sample.sample_id,
                input_tokens=tokens,
                budget=budget,
                policy=policy,
            )
            if policy == "fail":
                raise AdapterError(
                    f"dataset {dataset_cfg.id}: sample {sample.sample_id} exceeds the input "
                    f"token budget ({tokens} > {budget}) and on_oversize='fail'"
                )
            if policy == "skip":
                continue
            # policy == "resample"
            if replacement_attempts >= self.engine_cfg.limits.max_resample_attempts:
                logger.warning(
                    "dataset %s: replacement budget exhausted (%d attempts)",
                    dataset_cfg.id,
                    replacement_attempts,
                )
                continue
            try:
                replacements = adapter.replacement_samples(1, set(attempted))
            except Exception as exc:  # noqa: BLE001 - adapters must not break the run
                logger.warning(
                    "dataset %s: replacement_samples() failed: %s", dataset_cfg.id, exc
                )
                replacements = []
            replacement_attempts += 1
            fresh = [r for r in replacements if r.sample_id not in attempted]
            if not fresh:
                logger.info(
                    "dataset %s: no replacement available for oversize sample %s",
                    dataset_cfg.id,
                    sample.sample_id,
                )
                continue
            for candidate in fresh[:1]:
                attempted.add(candidate.sample_id)
                queue.append(candidate)
                prompt_set.replacements_used += 1

        fraction = oversize_seen / max(1, target + prompt_set.replacements_used)
        if fraction > self.engine_cfg.limits.oversize_abort_fraction:
            prompt_set.unusable_reason = (
                f"{oversize_seen} of {target + prompt_set.replacements_used} rendered prompts "
                f"exceeded the {budget}-token input budget "
                f"({fraction:.0%} > {self.engine_cfg.limits.oversize_abort_fraction:.0%}); "
                "the dataset does not fit this context budget"
            )
            logger.error("dataset %s unusable: %s", dataset_cfg.id, prompt_set.unusable_reason)

        if bundle.documentation is not None:
            stats = bundle.documentation.statistics
            stats[f"oversize_dropped[{modes.slug}]"] = len(prompt_set.oversize_dropped)
            stats[f"replacements_used[{modes.slug}]"] = prompt_set.replacements_used
            stats[f"prompts_kept[{modes.slug}]"] = prompt_set.size
            stats[f"input_tokens_mean[{modes.slug}]"] = round(
                mean([t for _, _, t in prompt_set.entries]), 1
            )
            stats[f"input_tokens_max[{modes.slug}]"] = max(
                [t for _, _, t in prompt_set.entries] or [0]
            )
            stats["input_token_budget"] = budget
            stats["token_counter_backend"] = self.token_counter.backend

        logger.info(
            "dataset %s [%s]: %d prompt(s) ready (%s, %d oversize, "
            "%d replacement(s))",
            dataset_cfg.id,
            modes.slug,
            prompt_set.size,
            modes.describe(),
            len(prompt_set.oversize_dropped),
            prompt_set.replacements_used,
        )
        return prompt_set

    # ------------------------------------------------------------------ #
    # Stage C planning
    # ------------------------------------------------------------------ #

    def _plan_tasks(
        self,
        bundles: Sequence[DatasetBundle],
        prompt_sets: dict[tuple[str, str], PromptSet],
    ) -> list[tuple[TaskIdentity, PromptSet, ModelConfig, DatasetBundle]]:
        tasks: list[tuple[TaskIdentity, PromptSet, ModelConfig, DatasetBundle]] = []
        by_key = {bundle.key: bundle for bundle in bundles}
        for key, prompt_set in prompt_sets.items():
            bundle = by_key[key]
            modes = bundle.modes
            kinds = sorted({sample.task_kind for sample in bundle.samples})
            for model in self.config.models:
                identity = TaskIdentity(
                    run_id=self.run_id,
                    dataset_id=bundle.config.id,
                    model_id=model.id,
                    template_id=prompt_set.template.id,
                    template_version=prompt_set.template.version,
                    prompt_mode=modes.prompt_mode,
                    selection_mode=modes.selection_mode or "n/a",
                    # A task whose samples mix kinds names the kind it is built
                    # around; each record still carries its own.
                    task_kind=kinds[0] if len(kinds) == 1 else "mixed",
                    data_delivery_mode=modes.data_delivery_mode,
                )
                tasks.append((identity, prompt_set, model, bundle))
        return tasks

    def _task_dir(self, identity: TaskIdentity) -> Path:
        return (
            self.run_dir
            / "datasets"
            / identity.dataset_id
            / identity.model_id
            / f"{identity.template_id}@{identity.template_version}"
        )

    # ------------------------------------------------------------------ #
    # Stage C execution
    # ------------------------------------------------------------------ #

    async def _run_task(
        self,
        identity: TaskIdentity,
        prompt_set: PromptSet,
        model: ModelConfig,
        bundle: DatasetBundle,
    ) -> TaskResult:
        started = time.time()
        assert bundle.adapter is not None
        adapter = bundle.adapter
        output_dir = self._task_dir(identity)
        store = RecordStore(
            output_dir,
            fsync_every=self.engine_cfg.checkpoint.fsync_every,
            store_raw_payloads=self.engine_cfg.checkpoint.store_raw_payloads,
            max_raw_payloads=self.engine_cfg.checkpoint.max_raw_payloads,
        )
        result = TaskResult(
            identity=identity,
            output_dir=output_dir,
            primary_metric=bundle.config.primary_metric or adapter.headline_metric,
            documentation=bundle.documentation,
        )

        if prompt_set.unusable_reason:
            result.failure = prompt_set.unusable_reason
            store.write_json("metrics.json", {"failure": result.failure})
            return result

        client = self._clients[model.id]
        use_batch = (
            client.supports_batch
            and model.id not in self._batch_disabled
            and self.engine_cfg.batching.max_group_size > 1
        )
        group_size = bundle.config.batch_group_size or model.endpoint.batch.group_size

        # Compose model-specific sampling params for every prompt.
        rendered: list[RenderedPrompt] = []
        clamped: list[tuple[str, int, int]] = []
        no_room: list[tuple[str, int]] = []
        # The least room worth sending a request for.
        min_answer_tokens = max(64, self.engine_cfg.batching.max_tokens_quantum)
        for sample, messages, tokens in prompt_set.entries:
            contract = sample.metadata.get("_output_contract", prompt_set.output_contract)
            sampling = batching_mod.resolve_sampling(
                per_sample_overrides=sample.sampling_overrides,
                model_sampling=model.sampling,
                template_sampling=prompt_set.template.sampling,
                batching=self.engine_cfg.batching,
                context_window=model.limits.context_window,
                input_tokens=tokens,
                exact_tokens=getattr(self.token_counter, "exact", False),
            )
            if prompt_set.modes.prompt_mode == SELF_CONSISTENCY:
                # A vote needs the k samples to be able to differ, so the
                # temperature comes from the mode and the fixed seed is dropped.
                sampling = sampling.merged(
                    temperature=prompt_set.modes.self_consistency_temperature
                )
                sampling = SamplingParams(
                    max_tokens=sampling.max_tokens,
                    temperature=sampling.temperature,
                    top_p=sampling.top_p,
                    seed=None,
                    stop=sampling.stop,
                    extra=sampling.extra,
                )
            if sampling.max_tokens < min_answer_tokens:
                # The prompt fits the dataset's input budget but leaves this
                # model no room to answer in. Sending it earns a 400 that takes
                # the whole batch down with it, so it is skipped and reported:
                # the sample is fine, this model's window is too small for it.
                no_room.append((sample.sample_id, tokens))
                continue
            if sampling.max_tokens < model.sampling.max_tokens_cap:
                # The window, not the cap, is what limits this request. Counted
                # so a dataset whose prompts crowd out the answer is visible.
                clamped.append(
                    (sample.sample_id, model.sampling.max_tokens_cap, sampling.max_tokens)
                )
            rendered.append(
                RenderedPrompt(
                    sample=sample,
                    messages=messages,
                    template_id=prompt_set.template.id,
                    template_version=prompt_set.template.version,
                    sampling=sampling,
                    input_tokens_est=tokens,
                    output_contract=contract,
                )
            )

        if no_room:
            longest = max(no_room, key=lambda item: item[1])
            logger.warning(
                "task %s: %d prompt(s) left %s no room to answer in and were skipped "
                "(longest: sample %s at %d input tokens against a %s-token window). "
                "Lower engine.limits.input_token_budget for this model, or use a model "
                "with a larger window.",
                identity.slug, len(no_room), model.id, longest[0], longest[1],
                model.limits.context_window,
            )
            self.events.emit(
                "prompts_without_answer_room",
                **identity.as_dict(),
                n_skipped=len(no_room),
                longest_sample=longest[0],
                longest_input_tokens=longest[1],
                context_window=model.limits.context_window,
            )

        result.n_planned = len(rendered)
        self._clamped_output_budgets = len(clamped)
        if clamped:
            worst = min(clamped, key=lambda item: item[2])
            logger.warning(
                "task %s: %d of %d prompt(s) had their output budget clamped by the model's "
                "%s-token context window (worst: sample %s asked for %d, got %d). Scores for "
                "those samples may reflect a truncated answer.",
                identity.slug,
                len(clamped),
                len(rendered),
                model.limits.context_window,
                worst[0],
                worst[1],
                worst[2],
            )
            self.events.emit(
                "output_budget_clamped",
                **identity.as_dict(),
                n_clamped=len(clamped),
                n_planned=len(rendered),
                context_window=model.limits.context_window,
                worst_sample=worst[0],
                worst_requested=worst[1],
                worst_granted=worst[2],
            )
        checkpoint = TaskCheckpoint(task=identity.as_dict(), total_planned=len(rendered))
        previous = store.load_checkpoint()
        if previous:
            checkpoint.notes["resumed_from"] = previous.get("updated_at")

        # Resume: reuse records that are still valid under the configured policy.
        reusable = (
            store.completed_keys(policy=self.engine_cfg.checkpoint.resume_policy)
            if self.engine_cfg.checkpoint.enabled
            else {}
        )
        pending: list[RenderedPrompt] = []
        reused_records: list[dict[str, Any]] = []
        for prompt in rendered:
            fingerprint = prompt.fingerprint(model.id)
            key = (
                f"{prompt.sample_id}::{fingerprint}"
                if self.engine_cfg.checkpoint.resume_policy == "strict"
                else prompt.sample_id
            )
            record = reusable.get(key)
            if record is not None:
                reused_records.append(record)
            else:
                pending.append(prompt)
        result.n_reused = len(reused_records)
        if reused_records:
            logger.info(
                "task %s: reusing %d checkpointed record(s), %d to run",
                identity.slug,
                len(reused_records),
                len(pending),
            )

        batches = batching_mod.plan_batches(
            pending,
            group_size=group_size if use_batch else 1,
            batching=self.engine_cfg.batching,
            prefix=identity.dataset_id[:12],
        )
        logger.info(
            "task %s: %d prompt(s) -> %d %s call(s) (group size %d, model=%s)",
            identity.slug,
            len(pending),
            len(batches),
            "batch" if use_batch else "single",
            group_size if use_batch else 1,
            model.model_name,
        )
        self.events.emit(
            "task_started",
            **identity.as_dict(),
            n_planned=len(rendered),
            n_pending=len(pending),
            n_reused=len(reused_records),
            n_batches=len(batches),
            batch_mode=use_batch,
            group_size=group_size,
        )

        scores: list[tuple[SampleSpec, ModelResponse, SampleScore]] = []
        fatal: BaseException | None = None
        model_sem = self._model_sems[model.id]

        # An interactive or sequential benchmark is not a list of prompts but a
        # set of episodes, so it is driven turn by turn instead of batch by
        # batch. Everything after this -- scoring, records, metrics -- is the
        # same, because an episode still produces one response per sample.
        interactive = prompt_set.modes.data_delivery_mode in ("interactive", "sequential")
        if interactive and pending:
            logger.info(
                "task %s: %d episode(s), up to %d turn(s) each (%s delivery)",
                identity.slug, len(pending), getattr(adapter, "max_turns", 8),
                prompt_set.modes.data_delivery_mode,
            )
            async with self._global_batch_sem, model_sem:
                pairs = await self._run_episodes(
                    adapter, pending, client=client, store=store, use_batch=use_batch,
                    checkpoint=checkpoint, model=model, group_size=group_size,
                )
            records: list[EvalRecord] = []
            for prompt, response in pairs:
                score = await self._score(adapter, prompt, response)
                records.append(self._make_record(identity, prompt, response, score))
                if response.status is ResponseStatus.ERROR:
                    checkpoint.failed += 1
                elif response.status is ResponseStatus.SKIPPED:
                    checkpoint.skipped += 1
                else:
                    checkpoint.completed += 1
                    scores.append((prompt.sample, response, score))
            store.append_many(records)
            store.save_checkpoint(checkpoint)
            batches = []

        async def _process(batch: batching_mod.Batch) -> None:
            nonlocal fatal
            if fatal is not None:
                return
            async with self._global_batch_sem, model_sem:
                if fatal is not None:
                    return
                try:
                    pairs = await self._execute_batch(batch, client, store, use_batch, checkpoint)
                except AuthError as exc:
                    fatal = exc
                    logger.error("task %s aborted: %s", identity.slug, exc)
                    return
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    logger.exception("task %s: batch %s failed hard", identity.slug, batch.batch_id)
                    pairs = [
                        (
                            prompt,
                            ModelResponse(
                                sample_id=prompt.sample_id,
                                model_id=model.id,
                                status=ResponseStatus.ERROR,
                                error=f"{type(exc).__name__}: {exc}",
                                error_class=ErrorClass.UNKNOWN.value,
                                batch_id=batch.batch_id,
                            ),
                        )
                        for prompt in batch.prompts
                    ]

            records: list[EvalRecord] = []
            for prompt, response in pairs:
                score = await self._score(adapter, prompt, response)
                records.append(self._make_record(identity, prompt, response, score))
                if response.status is ResponseStatus.ERROR:
                    checkpoint.failed += 1
                elif response.status is ResponseStatus.SKIPPED:
                    checkpoint.skipped += 1
                else:
                    checkpoint.completed += 1
                    scores.append((prompt.sample, response, score))
            store.append_many(records)
            store.save_checkpoint(checkpoint)

        await asyncio.gather(*(_process(batch) for batch in batches))

        if fatal is not None:
            result.failure = f"{type(fatal).__name__}: {fatal}"

        # Optional LLM-judge pass over the scored samples.
        if self.engine_cfg.judge.enabled and scores and not fatal:
            try:
                judge = JudgeStage(
                    config=self.engine_cfg.judge,
                    registry=self.registry,
                    renderer=self.renderer,
                    clients=self._clients,
                    retry_policy=self.retry_policy,
                    cache_dir=output_dir / "judge_cache",
                )
                scores = await judge.apply(adapter, scores)
            except Exception as exc:  # noqa: BLE001 - judging is best-effort
                logger.exception("task %s: judge stage failed: %s", identity.slug, exc)
                checkpoint.notes["judge_error"] = str(exc)

        # Fold in reused records so metrics cover the whole planned set.
        reused_scores = [
            SampleScore(
                metrics={k: float(v) for k, v in (rec.get("metrics") or {}).items()},
                prediction=rec.get("prediction"),
                parse_ok=bool(rec.get("parse_ok", True)),
                details=rec.get("details") or {},
            )
            for rec in reused_records
            if rec.get("status") != ResponseStatus.SKIPPED.value
        ]
        reused_errors = sum(
            1 for rec in reused_records if rec.get("status") == ResponseStatus.ERROR.value
        )
        reused_skips = sum(
            1 for rec in reused_records if rec.get("status") == ResponseStatus.SKIPPED.value
        )

        # Modes that ask one item as several requests are folded back here, so
        # everything downstream -- metrics, coverage, the sample sheet -- counts
        # evaluation items rather than the requests they were split into.
        if prompt_set.modes.needs_group_reduction and scores:
            scores, reduced_records = self._reduce_groups(adapter, identity, scores)
            if reduced_records:
                store.append_many(reduced_records)

        all_scores = [score for _, _, score in scores] + reused_scores
        result.n_scored = len(all_scores)
        result.n_error = checkpoint.failed + reused_errors
        # Prompts this model had no room to answer are skipped before anything
        # is sent, so they are counted here rather than by the checkpoint.
        result.n_skipped = checkpoint.skipped + reused_skips + len(no_room)

        metrics = self._aggregate(adapter, all_scores, result, scores, reused_records)
        result.metrics = metrics
        if result.primary_metric not in metrics and metrics:
            # A blank headline cell is the one outcome a summary must not have:
            # it reads as "this task failed" when the task ran fine and merely
            # scores itself under another name. Fall back to the first metric
            # the adapter documented that this task actually produced.
            documented = list((bundle.documentation.metrics_description or {})
                              if bundle.documentation else {})
            candidates = [name for name in documented if name in metrics] or [
                name for name in metrics if not name.startswith("n_")
            ]
            if candidates:
                logger.warning(
                    "task %s: primary metric %r was not produced; reporting %r instead",
                    identity.slug, result.primary_metric, candidates[0],
                )
                result.primary_metric = candidates[0]
        checkpoint.finished = True
        store.save_checkpoint(checkpoint)
        result.checkpoint = checkpoint
        result.duration_s = time.time() - started
        result.diagnostics = {
            "batch_mode": use_batch,
            "group_size": group_size if use_batch else 1,
            "batches_submitted": checkpoint.batches_submitted,
            "batches_failed": checkpoint.batches_failed,
            "bisections": checkpoint.bisections,
            "output_budgets_clamped": getattr(self, "_clamped_output_budgets", 0),
            "endpoint": client.batch_url if use_batch else client.base_url,
            "template": f"{prompt_set.template.id}@{prompt_set.template.version}",
            "variant": prompt_set.variant,
        }
        store.write_json(
            "metrics.json",
            {
                "task": identity.as_dict(),
                "primary_metric": result.primary_metric,
                "metrics": metrics,
                "counts": {
                    "planned": result.n_planned,
                    "scored": result.n_scored,
                    "errors": result.n_error,
                    "skipped": result.n_skipped,
                    "reused": result.n_reused,
                },
                "diagnostics": result.diagnostics,
                "failure": result.failure,
            },
        )
        logger.info(
            "task %s finished in %.1fs: %s=%.4f (scored %d/%d, %d error(s))",
            identity.slug,
            result.duration_s,
            result.primary_metric,
            metrics.get(result.primary_metric, float("nan")),
            result.n_scored,
            result.n_planned,
            result.n_error,
        )
        self.events.emit(
            "task_finished",
            **identity.as_dict(),
            metrics=metrics,
            counts={
                "planned": result.n_planned,
                "scored": result.n_scored,
                "errors": result.n_error,
                "skipped": result.n_skipped,
            },
            duration_s=round(result.duration_s, 1),
        )
        return result

    # ------------------------------------------------------------------ #
    # batch execution with retry + bisect
    # ------------------------------------------------------------------ #

    async def _run_episodes(
        self,
        adapter: DatasetAdapter,
        prompts: list[RenderedPrompt],
        *,
        client: ModelClient,
        store: RecordStore,
        use_batch: bool,
        checkpoint: TaskCheckpoint,
        model: ModelConfig,
        group_size: int,
    ) -> list[tuple[RenderedPrompt, ModelResponse]]:
        """Drive an interactive benchmark to completion, one turn at a time.

        Every live episode's *current* conversation is submitted together, so a
        multi-turn benchmark still uses the batch endpoint: turn 1 of all 300
        cases is one set of batch calls, then turn 2 of whatever is still live,
        and so on.  Episodes finish independently -- a model that commits after
        two questions stops costing anything while its neighbours keep asking.

        What ends an episode: the adapter's environment returning ``None``, the
        adapter's ``max_turns``, or a failed request (kept as the episode's
        result rather than retried forever).  The whole transcript is attached
        to the response, so a scorer or a reader can see what was asked.
        """
        episodes: list[dict[str, Any]] = []
        for prompt in prompts:
            try:
                messages, state = adapter.interactive_start(prompt.sample)
            except Exception as exc:  # noqa: BLE001 - one bad episode must not stop the task
                # An episode whose environment will not start is an *error*, not
                # a single-turn episode: falling back to the bare prompt would
                # score the model on a question the benchmark never asked.
                logger.warning("episode %s could not start: %s", prompt.sample_id, exc)
                episodes.append(
                    {
                        "prompt": prompt,
                        "messages": [],
                        "state": {},
                        "turn": 0,
                        "transcript": [],
                        "response": ModelResponse(
                            sample_id=prompt.sample_id,
                            model_id=model.id,
                            status=ResponseStatus.ERROR,
                            error=f"environment failed to start: {type(exc).__name__}: {exc}",
                            error_class=ErrorClass.INVALID_REQUEST.value,
                        ),
                        "dead": True,
                    }
                )
                continue
            episodes.append(
                {
                    "prompt": prompt,
                    "messages": list(messages),
                    "state": state,
                    "turn": 0,
                    "response": None,
                    "transcript": [m.to_dict() for m in messages],
                }
            )

        limit = max(1, int(getattr(adapter, "max_turns", 8)))
        # The least room an answer needs for a turn to be worth sending at all.
        min_answer_tokens = max(64, self.engine_cfg.batching.max_tokens_quantum)
        live = [episode for episode in episodes if not episode.get("dead")]
        turn = 0
        while live and turn < limit:
            turn += 1
            turn_prompts: list[RenderedPrompt] = []
            exhausted: list[dict[str, Any]] = []
            for episode in live:
                base = episode["prompt"]
                tokens = self.token_counter.count_messages(episode["messages"])
                window = model.limits.context_window
                room = window - tokens - batching_mod.context_reserve(
                    tokens,
                    self.engine_cfg.batching,
                    exact_tokens=getattr(self.token_counter, "exact", False),
                ) if window else min_answer_tokens
                if window and room < min_answer_tokens:
                    # An episode's transcript grows with every turn, and a long
                    # one eventually leaves no room to answer in. Sending the
                    # request anyway just earns a context_length rejection and
                    # loses the whole batch it travelled in, so the episode ends
                    # here and says why -- which is also a real property of the
                    # model being measured: it ran out of room to think in.
                    episode["context_exhausted"] = True
                    exhausted.append(episode)
                    continue
                sampling = batching_mod.resolve_sampling(
                    per_sample_overrides=base.sample.sampling_overrides,
                    model_sampling=model.sampling,
                    template_sampling={},
                    batching=self.engine_cfg.batching,
                    context_window=model.limits.context_window,
                    input_tokens=tokens,
                    exact_tokens=getattr(self.token_counter, "exact", False),
                )
                turn_prompts.append(
                    RenderedPrompt(
                        sample=base.sample,
                        messages=list(episode["messages"]),
                        template_id=base.template_id,
                        template_version=base.template_version,
                        sampling=sampling,
                        input_tokens_est=tokens,
                        output_contract=base.output_contract,
                    )
                )

            if exhausted:
                logger.info(
                    "%d episode(s) ran out of context after %d turn(s); ending them there",
                    len(exhausted), turn - 1,
                )
                live = [episode for episode in live if not episode.get("context_exhausted")]
            if not turn_prompts:
                break

            by_id = {id(p): episode for p, episode in zip(turn_prompts, live, strict=True)}
            index = {p.sample_id: by_id[id(p)] for p in turn_prompts}
            results: list[tuple[RenderedPrompt, ModelResponse]] = []
            batches = batching_mod.plan_batches(
                turn_prompts,
                group_size=group_size if use_batch else 1,
                batching=self.engine_cfg.batching,
                prefix=f"turn{turn}",
            )
            for batch in batches:
                results.extend(
                    await self._execute_batch(batch, client, store, use_batch, checkpoint)
                )

            still_live: list[dict[str, Any]] = []
            for turn_prompt, response in results:
                episode = index[turn_prompt.sample_id]
                episode["response"] = response
                episode["turn"] = turn
                text = response.text
                episode["messages"].append(ChatMessage(role="assistant", content=text))
                episode["transcript"].append({"role": "assistant", "content": text})
                if response.status is ResponseStatus.ERROR:
                    continue
                try:
                    reply = adapter.interactive_step(episode["prompt"].sample, episode["state"], text)
                except Exception as exc:  # noqa: BLE001 - environments must not break the run
                    logger.warning(
                        "episode %s: environment step failed: %s",
                        turn_prompt.sample_id, exc,
                    )
                    reply = None
                if reply is None:
                    continue
                episode["messages"].append(ChatMessage(role="user", content=reply))
                episode["transcript"].append({"role": "user", "content": reply})
                still_live.append(episode)
            live = still_live

        if live:
            logger.info(
                "%d episode(s) hit the %d-turn limit without committing to an answer",
                len(live), limit,
            )

        pairs: list[tuple[RenderedPrompt, ModelResponse]] = []
        for episode in episodes:
            prompt = episode["prompt"]
            response = episode["response"]
            if response is None:
                response = ModelResponse(
                    sample_id=prompt.sample_id,
                    model_id=model.id,
                    status=ResponseStatus.ERROR,
                    error="episode produced no response",
                    error_class=ErrorClass.UNKNOWN.value,
                )
            # The turn count and the transcript are part of the result: an
            # interactive benchmark is as much about what was asked as about the
            # final answer.
            response.usage = {
                **(response.usage or {}),
                "turns": episode["turn"],
                "transcript_messages": len(episode["transcript"]),
                "context_exhausted": bool(episode.get("context_exhausted")),
            }
            prompt.sample.metadata["turns_used"] = episode["turn"]
            if episode.get("context_exhausted"):
                prompt.sample.metadata["context_exhausted"] = True
            prompt.sample.metadata["_transcript"] = episode["transcript"]
            # The environment's final state, for scorers that grade the episode
            # rather than the last message -- an experiment log, a set of
            # predictions, the evidence that was actually requested.
            prompt.sample.metadata["_episode_state"] = episode["state"]
            pairs.append((prompt, response))
        return pairs

    async def _execute_batch(
        self,
        batch: batching_mod.Batch,
        client: ModelClient,
        store: RecordStore,
        use_batch: bool,
        checkpoint: TaskCheckpoint,
    ) -> list[tuple[RenderedPrompt, ModelResponse]]:
        """Submit one batch, retrying transient failures and bisecting bad ones."""
        recovery = self.engine_cfg.retry.recovery

        async def _attempt() -> BatchResult:
            if use_batch and batch.size > 1:
                return await client.chat_batch(batch.conversations(), batch.sampling)
            if use_batch and batch.size == 1:
                # A single-item native batch call keeps the code path identical
                # to a real batch (same endpoint, same response shape).
                return await client.chat_batch(batch.conversations(), batch.sampling)
            return await client.chat_single(batch.prompts[0].messages, batch.sampling)

        async def _recover(_exc: BaseException) -> bool:
            if not recovery.enabled:
                return False
            return await client.recover(
                probe_interval_s=recovery.probe_interval_s,
                max_probe_attempts=recovery.max_probe_attempts,
                rediscover_on_connection_error=recovery.rediscover_on_connection_error,
            )

        checkpoint.batches_submitted += 1
        try:
            batch_result, outcome = await with_retry(
                _attempt,
                policy=self.retry_policy,
                description=f"batch {batch.batch_id} ({batch.size} sample(s), {client.model.id})",
                on_recover=_recover,
                on_attempt_failed=lambda exc, attempt, sleep_s: self.events.emit(
                    "batch_attempt_failed",
                    batch_id=batch.batch_id,
                    model_id=client.model.id,
                    attempt=attempt,
                    sleep_s=round(sleep_s, 2),
                    error=clip(str(exc), self.engine_cfg.logging.log_text_clip),
                    error_class=self.retry_policy.error_class_of(exc).value,
                ),
            )
        except AuthError:
            checkpoint.batches_failed += 1
            raise
        except BaseException as exc:  # noqa: BLE001
            checkpoint.batches_failed += 1
            error_class = self.retry_policy.error_class_of(exc)
            bisect_cfg = self.engine_cfg.batching
            bisectable = (
                error_class in (ErrorClass.INVALID_REQUEST, ErrorClass.CONTEXT_LENGTH)
                and bisect_cfg.bisect_on_invalid_request
            ) or (
                error_class in (ErrorClass.TRANSIENT, ErrorClass.RATE_LIMIT, ErrorClass.UNKNOWN)
                and bisect_cfg.bisect_on_timeout
            )
            if bisectable and batch.size > 1:
                checkpoint.bisections += 1
                logger.warning(
                    "batch %s (%d samples) failed [%s]; splitting it and retrying the halves",
                    batch.batch_id,
                    batch.size,
                    error_class.value,
                )
                self.events.emit(
                    "batch_bisected",
                    batch_id=batch.batch_id,
                    size=batch.size,
                    error_class=error_class.value,
                    error=clip(str(exc), self.engine_cfg.logging.log_text_clip),
                )
                pairs: list[tuple[RenderedPrompt, ModelResponse]] = []
                for half in batching_mod.bisect(batch):
                    pairs.extend(
                        await self._execute_batch(half, client, store, use_batch, checkpoint)
                    )
                return pairs
            status = (
                ResponseStatus.SKIPPED
                if error_class is ErrorClass.CONTEXT_LENGTH
                else ResponseStatus.ERROR
            )
            self.events.emit(
                "batch_failed",
                batch_id=batch.batch_id,
                size=batch.size,
                model_id=client.model.id,
                error_class=error_class.value,
                error=clip(str(exc), self.engine_cfg.logging.log_text_clip),
                status=status.value,
            )
            return [
                (
                    prompt,
                    ModelResponse(
                        sample_id=prompt.sample_id,
                        model_id=client.model.id,
                        status=status,
                        error=str(exc)[:2000],
                        error_class=error_class.value,
                        batch_id=batch.batch_id,
                        batch_size=batch.size,
                        attempts=self.engine_cfg.retry.max_attempts,
                    ),
                )
                for prompt in batch.prompts
            ]

        if self.engine_cfg.checkpoint.store_raw_payloads:
            store.save_raw(
                f"{batch.batch_id}",
                {
                    "request": {
                        "url": batch_result.endpoint_url,
                        "model": client.model.model_name,
                        "sampling": batch.sampling.to_payload(),
                        "sample_ids": batch.sample_ids,
                        "messages": [
                            [m.to_dict() for m in prompt.messages] for prompt in batch.prompts
                        ],
                    },
                    "response": {
                        "id": batch_result.response_id,
                        "usage": batch_result.usage,
                        "latency_s": batch_result.latency_s,
                        "choices": [choice.raw for choice in batch_result.choices],
                    },
                    "retry": summarize(outcome),
                },
            )

        usage = batch_result.usage or {}
        checkpoint.prompt_tokens_total += int(usage.get("prompt_tokens") or 0)
        checkpoint.completion_tokens_total += int(usage.get("completion_tokens") or 0)
        self.events.emit(
            "batch_completed",
            batch_id=batch.batch_id,
            size=batch.size,
            model_id=client.model.id,
            latency_s=round(batch_result.latency_s, 2),
            attempts=outcome.attempts,
            usage=usage,
            endpoint=batch_result.endpoint_url,
        )

        pairs = []
        for position, prompt in enumerate(batch.prompts):
            choice = self._choice_for(batch_result, position)
            pairs.append((prompt, self._normalize(prompt, choice, batch_result, client, batch)))
        return pairs

    @staticmethod
    def _choice_for(batch_result: BatchResult, position: int) -> RawChoice | None:
        for choice in batch_result.choices:
            if choice.index == position:
                return choice
        if len(batch_result.choices) > position:
            return batch_result.choices[position]
        return None

    def _normalize(
        self,
        prompt: RenderedPrompt,
        choice: RawChoice | None,
        batch_result: BatchResult,
        client: ModelClient,
        batch: batching_mod.Batch,
    ) -> ModelResponse:
        """Turn one choice into a :class:`ModelResponse` with a status."""
        if choice is None:
            return ModelResponse(
                sample_id=prompt.sample_id,
                model_id=client.model.id,
                status=ResponseStatus.ERROR,
                error="no choice returned for this sample",
                error_class=ErrorClass.PROTOCOL.value,
                batch_id=batch.batch_id,
                batch_size=batch.size,
                latency_s=batch_result.latency_s,
            )
        content = choice.content
        finish = choice.finish_reason
        if content is None or not content.strip():
            status = ResponseStatus.EMPTY
        elif finish == "length":
            status = ResponseStatus.TRUNCATED
        else:
            status = ResponseStatus.OK
        if status is ResponseStatus.EMPTY and client.model.reasoning_model:
            logger.debug(
                "sample %s: empty content with finish_reason=%s -- reasoning consumed the "
                "%d-token budget",
                prompt.sample_id,
                finish,
                prompt.sampling.max_tokens,
            )
        completion_est = None
        if content or choice.reasoning:
            completion_est = self.token_counter.count_text((content or "") + (choice.reasoning or ""))
        return ModelResponse(
            sample_id=prompt.sample_id,
            model_id=client.model.id,
            status=status,
            content=content,
            reasoning=choice.reasoning,
            finish_reason=finish,
            batch_id=batch.batch_id,
            batch_size=batch.size,
            batch_index=choice.index,
            latency_s=batch_result.latency_s,
            usage=dict(batch_result.usage or {}),
            usage_is_batch_aggregate=batch_result.is_batch and batch.size > 1,
            completion_tokens_est=completion_est,
        )

    # ------------------------------------------------------------------ #
    # scoring / aggregation / records
    # ------------------------------------------------------------------ #

    async def _score(
        self, adapter: DatasetAdapter, prompt: RenderedPrompt, response: ModelResponse
    ) -> SampleScore:
        """Run the adapter's scorer off the event loop, never letting it crash a run."""
        if response.status in (ResponseStatus.ERROR, ResponseStatus.SKIPPED):
            return SampleScore(metrics={}, parse_ok=False, details={"unscored": response.status.value})
        async with self._scoring_sem:
            try:
                return await asyncio.to_thread(
                    adapter.score, prompt.sample, response, output_contract=prompt.output_contract
                )
            except Exception as exc:  # noqa: BLE001 - a bad scorer must not kill the run
                logger.exception(
                    "adapter %s: score() raised for sample %s", adapter.dataset_id, prompt.sample_id
                )
                return SampleScore(
                    metrics={},
                    parse_ok=False,
                    details={"scorer_error": f"{type(exc).__name__}: {exc}"},
                )

    def _aggregate(
        self,
        adapter: DatasetAdapter,
        all_scores: list[SampleScore],
        result: TaskResult,
        fresh: list[tuple[SampleSpec, ModelResponse, SampleScore]],
        reused_records: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Adapter metrics + engine-level reliability/efficiency metrics.

        The engine always adds:

        ``coverage``            fraction of planned samples that were scored;
        ``parse_failure_rate``  fraction of scored samples the adapter could not parse;
        ``<primary>_strict``    the primary metric with unscored samples counted as 0,
                                so a run with many endpoint failures cannot look good;
        plus latency and token statistics.
        """
        metrics: dict[str, float] = {}
        if all_scores:
            try:
                metrics.update(
                    {k: float(v) for k, v in (adapter.aggregate(all_scores) or {}).items()}
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("adapter %s: aggregate() failed: %s", adapter.dataset_id, exc)
                result.failure = (result.failure or "") + f" aggregate() failed: {exc}"

        planned = max(1, result.n_planned)
        coverage = result.n_scored / planned
        metrics["coverage"] = coverage
        metrics["n_planned"] = float(result.n_planned)
        metrics["n_scored"] = float(result.n_scored)
        metrics["n_error"] = float(result.n_error)
        metrics["n_skipped"] = float(result.n_skipped)
        metrics["parse_failure_rate"] = (
            sum(1 for s in all_scores if not s.parse_ok) / len(all_scores) if all_scores else 0.0
        )
        primary = result.primary_metric
        if primary and primary in metrics:
            metrics[f"{primary}_strict"] = metrics[primary] * coverage

        latencies = [r.latency_s for _, r, _ in fresh if r.latency_s]
        if latencies:
            metrics["batch_latency_s_mean"] = mean(latencies)
        completions = [
            float(r.completion_tokens_est) for _, r, _ in fresh if r.completion_tokens_est
        ]
        if completions:
            metrics["completion_tokens_mean"] = mean(completions)
        truncated = sum(1 for _, r, _ in fresh if r.status is ResponseStatus.TRUNCATED)
        empty = sum(1 for _, r, _ in fresh if r.status is ResponseStatus.EMPTY)
        reused_truncated = sum(
            1 for rec in reused_records if rec.get("status") == ResponseStatus.TRUNCATED.value
        )
        reused_empty = sum(
            1 for rec in reused_records if rec.get("status") == ResponseStatus.EMPTY.value
        )
        metrics["truncation_rate"] = (truncated + reused_truncated) / planned
        metrics["output_budget_clamped_rate"] = (
            getattr(self, "_clamped_output_budgets", 0) / planned
        )
        metrics["empty_response_rate"] = (empty + reused_empty) / planned
        return metrics

    def _reduce_groups(
        self,
        adapter: DatasetAdapter,
        identity: TaskIdentity,
        scores: list[tuple[SampleSpec, ModelResponse, SampleScore]],
    ) -> tuple[list[tuple[SampleSpec, ModelResponse, SampleScore]], list[EvalRecord]]:
        """Fold each item's several responses into one scored answer.

        Self-consistency votes over k samples of one question; BOV rebuilds a
        selected set from one yes/no answer per hypothesis.  Either way the
        adapter decides what the fold means -- the engine only groups by
        ``group_id`` and writes the result as an extra record marked
        ``reduced``, keeping the members in the log so a vote can be inspected.
        """
        grouped: dict[str, list[tuple[SampleSpec, ModelResponse, SampleScore]]] = {}
        ungrouped: list[tuple[SampleSpec, ModelResponse, SampleScore]] = []
        for sample, response, score in scores:
            if sample.group_id:
                grouped.setdefault(sample.group_id, []).append((sample, response, score))
            else:
                ungrouped.append((sample, response, score))

        out = list(ungrouped)
        records: list[EvalRecord] = []
        for group_id, members in grouped.items():
            try:
                reduced = adapter.reduce_group(members)
            except Exception as exc:  # noqa: BLE001 - a bad fold must not lose the run
                logger.warning(
                    "task %s: reducing group %s failed: %s", identity.slug, group_id, exc
                )
                reduced = None
            if reduced is None:
                out.extend(members)
                continue
            sample, response, _ = members[0]
            parent = sample.metadata.get("_parent_sample", sample)
            out.append((parent, response, reduced))
            records.append(
                EvalRecord(
                    task=identity,
                    sample_id=group_id,
                    status=response.status,
                    prompt_fingerprint=f"reduced::{group_id}",
                    task_kind=parent.task_kind,
                    input_tokens_est=sum(m[0].max_tokens or 0 for m in members),
                    sampling={"members": len(members)},
                    response={
                        "content": None,
                        "status": response.status.value,
                        "reduced_from": [m[0].sample_id for m in members],
                    },
                    metrics=reduced.metrics,
                    prediction=reduced.prediction,
                    parse_ok=reduced.parse_ok,
                    reference=parent.reference,
                    metadata={
                        **{k: v for k, v in parent.metadata.items() if not k.startswith("_")},
                        "reduced": True,
                        "n_members": len(members),
                    },
                    details=reduced.details,
                    group_id=group_id,
                )
            )
        if records:
            logger.info(
                "task %s: %d group(s) reduced from %d response(s)",
                identity.slug,
                len(records),
                sum(len(m) for m in grouped.values()),
            )
        return out, records

    def _make_record(
        self,
        identity: TaskIdentity,
        prompt: RenderedPrompt,
        response: ModelResponse,
        score: SampleScore,
    ) -> EvalRecord:
        clip_chars = self.engine_cfg.reporting.response_clip_chars
        return EvalRecord(
            task=identity,
            sample_id=prompt.sample_id,
            status=response.status,
            prompt_fingerprint=prompt.fingerprint(identity.model_id),
            task_kind=prompt.sample.task_kind,
            input_tokens_est=prompt.input_tokens_est,
            sampling=prompt.sampling.to_payload(),
            response={
                "content": response.content,
                "reasoning": clip(response.reasoning, clip_chars) if response.reasoning else None,
                "finish_reason": response.finish_reason,
                "status": response.status.value,
                "error": response.error,
                "error_class": response.error_class,
                "attempts": response.attempts,
                "latency_s": round(response.latency_s, 3),
                "batch_id": response.batch_id,
                "batch_size": response.batch_size,
                "batch_index": response.batch_index,
                "usage": response.usage,
                "usage_is_batch_aggregate": response.usage_is_batch_aggregate,
                "completion_tokens_est": response.completion_tokens_est,
            },
            metrics=score.metrics,
            prediction=score.prediction,
            parse_ok=score.parse_ok,
            reference=prompt.sample.reference,
            metadata={k: v for k, v in prompt.sample.metadata.items() if not k.startswith("_")},
            details=score.details,
            group_id=prompt.sample.group_id,
        )
