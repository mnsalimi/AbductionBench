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

from . import batching as batching_mod
from .adapter import AdapterContext, DatasetAdapter, SkippedDataset
from .checkpoint import RecordStore, TaskCheckpoint
from .client import BatchResult, ModelClient, RawChoice
from .config import DatasetConfig, ModelConfig, RunConfig, dump_resolved
from .errors import AbenchError, AdapterError, AuthError, ErrorClass, TemplateError
from .judge import JudgeStage
from .metrics import mean
from .prompts import PromptRegistry, PromptRenderer, PromptTemplate, TemplateBinding
from .registry import resolve_adapter
from .retry import RetryPolicy, summarize, with_retry
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
    samples: list[SampleSpec] = field(default_factory=list)
    documentation: AdapterDocumentation | None = None
    skipped_reason: str | None = None
    prepare_seconds: float = 0.0

    @property
    def skipped(self) -> bool:
        return self.skipped_reason is not None


@dataclass(slots=True)
class PromptSet:
    """Rendered prompts for one (dataset, template variant), shared by models."""

    dataset_id: str
    variant: str
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
    endpoint_reports: list[dict[str, Any]] = field(default_factory=list)
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
        self.registry = PromptRegistry(list(config.prompts.template_dirs))
        self.renderer = PromptRenderer(self.registry, config.prompts)
        self.token_counter = build_token_counter(self.engine_cfg.tokenizer)
        self.retry_policy = RetryPolicy(self.engine_cfg.retry)
        self._clients: dict[str, ModelClient] = {}
        self._global_batch_sem = asyncio.Semaphore(
            self.engine_cfg.concurrency.max_parallel_batches_global
        )
        self._model_sems: dict[str, asyncio.Semaphore] = {}
        self._scoring_sem = asyncio.Semaphore(self.engine_cfg.concurrency.scoring_workers)
        self._batch_disabled: set[str] = set()

    @staticmethod
    def _default_run_id(name: str) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
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
                for binding in self.renderer.bindings_for_dataset(
                    bundle.config.id, bundle.config.prompt_bindings
                ):
                    try:
                        prompt_set = self._build_prompt_set(bundle, binding)
                    except (TemplateError, AdapterError) as exc:
                        logger.error(
                            "dataset %s variant %s: prompt rendering failed: %s",
                            bundle.config.id,
                            binding.variant,
                            exc,
                        )
                        result.skipped_datasets.append(
                            {
                                "dataset_id": bundle.config.id,
                                "reason": f"prompt rendering failed ({binding.variant}): {exc}",
                            }
                        )
                        continue
                    prompt_sets[(bundle.config.id, binding.variant)] = prompt_set

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
            self.events.emit(
                "run_finished",
                run_id=self.run_id,
                tasks=len(result.tasks),
                skipped=len(result.skipped_datasets),
                duration_s=round(result.duration_s, 1),
            )
        return result

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

    def _build_bundles(self) -> list[DatasetBundle]:
        bundles: list[DatasetBundle] = []
        for dataset_cfg in self.config.enabled_datasets():
            started = time.time()
            bundle = DatasetBundle(config=dataset_cfg, adapter=None)
            try:
                adapter = self._instantiate_adapter(dataset_cfg)
                adapter.prepare()
                samples = list(adapter.build_samples())
                if not samples:
                    raise SkippedDataset("adapter produced no samples")
                if len(samples) > dataset_cfg.sample_size:
                    logger.warning(
                        "dataset %s: adapter returned %d samples for a requested size of %d; "
                        "truncating to the requested size",
                        dataset_cfg.id,
                        len(samples),
                        dataset_cfg.sample_size,
                    )
                    samples = samples[: dataset_cfg.sample_size]
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
            bundles.append(bundle)
        return bundles

    def _instantiate_adapter(self, dataset_cfg: DatasetConfig) -> DatasetAdapter:
        adapter_cls = resolve_adapter(dataset_cfg.impl)
        data_dir = Path(self.engine_cfg.data_root) / dataset_cfg.id
        data_dir.mkdir(parents=True, exist_ok=True)
        context = AdapterContext(
            dataset_id=dataset_cfg.id,
            data_dir=data_dir,
            sample_size=dataset_cfg.sample_size,
            seed=dataset_cfg.seed if dataset_cfg.seed is not None else self.config.seed,
            options=dict(dataset_cfg.options),
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

    def _build_prompt_set(self, bundle: DatasetBundle, binding: TemplateBinding) -> PromptSet:
        """Render a dataset's samples and enforce the input-token budget.

        Oversize samples are replaced with fresh draws from the same split
        (``engine.limits.on_oversize == "resample"``) so the evaluation set keeps
        its configured size instead of silently shrinking.
        """
        assert bundle.adapter is not None
        dataset_cfg = bundle.config
        adapter = bundle.adapter
        budget = dataset_cfg.input_token_budget or self.engine_cfg.limits.input_token_budget
        policy = self.engine_cfg.limits.on_oversize
        target = len(bundle.samples)

        template_ids = {sample.task_kind: binding.template_for(sample.task_kind)
                        for sample in bundle.samples}
        distinct = set(template_ids.values())
        primary_template_id = (
            next(iter(distinct))
            if len(distinct) == 1
            else binding.template_for(bundle.samples[0].task_kind)
        )
        prompt_set = PromptSet(
            dataset_id=dataset_cfg.id,
            variant=binding.variant,
            template=self.registry.get(primary_template_id),
            output_contract=dict(self.registry.get(primary_template_id).output_contract),
        )

        queue: list[SampleSpec] = list(bundle.samples)
        attempted: set[str] = {s.sample_id for s in bundle.samples}
        replacement_attempts = 0
        oversize_seen = 0

        while queue:
            sample = queue.pop(0)
            template = self.registry.get(binding.template_for(sample.task_kind))
            messages, contract = self.renderer.render(sample, template)
            tokens = self.token_counter.count_messages(messages)
            if tokens <= budget:
                prompt_set.entries.append((sample, messages, tokens))
                if len(distinct) > 1:
                    # Mixed-kind datasets: remember the per-sample contract.
                    sample.metadata.setdefault("_output_contract", contract)
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
            stats[f"oversize_dropped[{binding.variant}]"] = len(prompt_set.oversize_dropped)
            stats[f"replacements_used[{binding.variant}]"] = prompt_set.replacements_used
            stats[f"prompts_kept[{binding.variant}]"] = prompt_set.size
            stats[f"input_tokens_mean[{binding.variant}]"] = round(
                mean([t for _, _, t in prompt_set.entries]), 1
            )
            stats[f"input_tokens_max[{binding.variant}]"] = max(
                [t for _, _, t in prompt_set.entries] or [0]
            )
            stats["input_token_budget"] = budget
            stats["token_counter_backend"] = self.token_counter.backend

        logger.info(
            "dataset %s variant %s: %d prompt(s) ready (template=%s, %d oversize, "
            "%d replacement(s))",
            dataset_cfg.id,
            binding.variant,
            prompt_set.size,
            prompt_set.template.ref,
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
        by_id = {bundle.config.id: bundle for bundle in bundles}
        for (dataset_id, _variant), prompt_set in prompt_sets.items():
            bundle = by_id[dataset_id]
            for model in self.config.models:
                identity = TaskIdentity(
                    run_id=self.run_id,
                    dataset_id=dataset_id,
                    model_id=model.id,
                    template_id=prompt_set.template.id,
                    template_version=prompt_set.template.version,
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
            primary_metric=bundle.config.primary_metric or adapter.primary_metric,
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
        for sample, messages, tokens in prompt_set.entries:
            contract = sample.metadata.get("_output_contract", prompt_set.output_contract)
            sampling = batching_mod.resolve_sampling(
                requested_max_tokens=sample.max_tokens,
                per_sample_overrides=sample.sampling_overrides,
                model_sampling=model.sampling,
                template_sampling=prompt_set.template.sampling,
                batching=self.engine_cfg.batching,
                context_window=model.limits.context_window,
                input_tokens=tokens,
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

        result.n_planned = len(rendered)
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

            pairs = await self._escalate_empty(
                pairs, client=client, store=store, use_batch=use_batch,
                checkpoint=checkpoint, model=model,
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

        all_scores = [score for _, _, score in scores] + reused_scores
        result.n_scored = len(all_scores)
        result.n_error = checkpoint.failed + reused_errors
        result.n_skipped = checkpoint.skipped + reused_skips

        metrics = self._aggregate(adapter, all_scores, result, scores, reused_records)
        result.metrics = metrics
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
            "empty_escalations": checkpoint.empty_escalations,
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

    async def _escalate_empty(
        self,
        pairs: list[tuple[RenderedPrompt, ModelResponse]],
        *,
        client: ModelClient,
        store: RecordStore,
        use_batch: bool,
        checkpoint: TaskCheckpoint,
        model: ModelConfig,
    ) -> list[tuple[RenderedPrompt, ModelResponse]]:
        """Re-issue samples that came back empty because the budget ran out.

        A reasoning model can spend its whole ``max_tokens`` on hidden
        chain-of-thought and return ``content: null`` with
        ``finish_reason="length"``.  That is not a wrong answer and not an
        endpoint failure -- it is an under-budgeted request.  Rather than raise
        the budget for every sample in the dataset, only the affected samples
        are retried with a multiplied budget (bounded by the model's cap and its
        context window).
        """
        config = self.engine_cfg.retry
        if not config.escalate_empty_responses or config.max_empty_escalations <= 0:
            return pairs

        keep: list[tuple[RenderedPrompt, ModelResponse]] = []
        retry_prompts: list[RenderedPrompt] = []
        for prompt, response in pairs:
            budget_exhausted = (
                response.status is ResponseStatus.EMPTY
                and (response.finish_reason == "length" or response.finish_reason is None)
            )
            escalated = prompt.sample.metadata.get("_empty_escalations", 0)
            if not budget_exhausted or escalated >= config.max_empty_escalations:
                keep.append((prompt, response))
                continue
            bigger = int(prompt.sampling.max_tokens * config.empty_budget_multiplier)
            bigger = min(bigger, model.sampling.max_tokens_cap)
            if model.limits.context_window:
                bigger = min(bigger, model.limits.context_window - prompt.input_tokens_est - 8)
            if bigger <= prompt.sampling.max_tokens:
                # No headroom left; keep the empty response and report it.
                keep.append((prompt, response))
                continue
            prompt.sample.metadata["_empty_escalations"] = escalated + 1
            retry_prompts.append(
                RenderedPrompt(
                    sample=prompt.sample,
                    messages=prompt.messages,
                    template_id=prompt.template_id,
                    template_version=prompt.template_version,
                    sampling=prompt.sampling.merged(max_tokens=bigger),
                    input_tokens_est=prompt.input_tokens_est,
                    output_contract=prompt.output_contract,
                )
            )

        if not retry_prompts:
            return pairs

        checkpoint.empty_escalations += len(retry_prompts)
        logger.info(
            "%d sample(s) returned empty content at the token budget; retrying them with a "
            "%.1fx budget",
            len(retry_prompts),
            config.empty_budget_multiplier,
        )
        self.events.emit(
            "empty_escalation",
            model_id=model.id,
            n_samples=len(retry_prompts),
            sample_ids=[p.sample_id for p in retry_prompts],
            multiplier=config.empty_budget_multiplier,
        )
        for batch in batching_mod.plan_batches(
            retry_prompts,
            group_size=(
                model.endpoint.batch.group_size if use_batch else 1
            ),
            batching=self.engine_cfg.batching,
            prefix="empty-retry",
        ):
            keep.extend(
                await self._execute_batch(batch, client, store, use_batch, checkpoint)
            )
        return keep

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
        metrics["empty_response_rate"] = (empty + reused_empty) / planned
        return metrics

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
