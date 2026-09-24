"""Apply the reasoning-metric judge to a run that already happened.

A run's answers are expensive and its records keep everything these metrics
need, so measuring a chain of reasoning does not mean asking the model under
test anything again.  This walks a finished (or interrupted) run directory,
re-renders the prompts that run used -- deterministically, from the same
resolved config and seed -- pairs them with the stored responses, and runs
:class:`~abductionbench.core.reasoning_judge.ReasoningJudgeStage` over every
``cot`` and ``self-consistency`` record in it.

It is the same stage, the same prompts and the same cache the engine uses
inline during a live run, so a run judged here and a run judged in flight get
identical columns.  ``abench judge-reasoning <run-dir>`` is the entry point.

One limitation, reported rather than worked around: an **interactive** task's
question is the episode transcript, which exists only while the episode is
running and is not stored in the record.  Re-rendering offline would show the
judge a prompt the model never actually saw, so interactive tasks are skipped
here and named in the summary.  The inline stage judges them correctly, because
there the transcript is still in hand.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any

import orjson

from .checkpoint import dedupe_records, load_records
from .client import ModelClient
from .config import RunConfig, load_run_config, load_yaml
from .engine import EvaluationEngine
from .errors import ConfigError
from .metrics import stored_metrics
from .modes import COT, INTERACTIVE, SELF_CONSISTENCY
from .reasoning_judge import ReasoningJudgeStage
from .types import (
    ChatMessage,
    ModelResponse,
    RenderedPrompt,
    ResponseStatus,
    SampleScore,
    SamplingParams,
    TaskIdentity,
)

logger = logging.getLogger(__name__)

__all__ = ["rejudge_reasoning"]


def _rendered_prompts(prompt_set: Any) -> list[RenderedPrompt]:
    """Minimal RenderedPrompts from a prompt set.

    The reasoning judge reads only the conversation and the sample, so the
    sampling parameters are a placeholder here rather than a re-derivation of
    what the run used: nothing downstream of this asks the model under test
    anything.
    """
    return [
        RenderedPrompt(
            sample=sample,
            messages=messages,
            template_id=prompt_set.template.id,
            template_version=prompt_set.template.version,
            sampling=SamplingParams(max_tokens=1),
            input_tokens_est=tokens,
            output_contract=sample.metadata.get(
                "_output_contract", prompt_set.output_contract
            ),
        )
        for sample, messages, tokens in prompt_set.entries
    ]


def _stored_conversations(task_dir: Path) -> dict[str, list[ChatMessage]]:
    """The exact conversations a task sent, read back from its raw payloads.

    Re-rendering would give the prompt this checkout *would* send today, which
    is not necessarily the one the stored answer was produced from -- prompt
    wording is configuration and it changes.  The judge has to see what the
    model actually saw, so the raw payload wins wherever one was kept.
    """
    conversations: dict[str, list[ChatMessage]] = {}
    raw_dir = task_dir / "raw"
    if not raw_dir.is_dir():
        return conversations
    for path in sorted(raw_dir.glob("*.json")):
        try:
            payload = orjson.loads(path.read_bytes())
        except orjson.JSONDecodeError:
            continue
        request = payload.get("request") or {}
        sample_ids = request.get("sample_ids") or []
        messages = request.get("messages") or []
        # A batch payload holds one conversation per sample, in the same order.
        if len(sample_ids) == len(messages) and sample_ids:
            pairs = zip(sample_ids, messages, strict=True)
        elif len(sample_ids) == 1:
            pairs = zip(sample_ids, [messages], strict=True)
        else:
            continue
        for sample_id, conversation in pairs:
            if not isinstance(conversation, list):
                continue
            conversations[str(sample_id)] = [
                ChatMessage(role=str(m.get("role", "user")), content=str(m.get("content", "")))
                for m in conversation
                if isinstance(m, dict)
            ]
    return conversations


def _triplet(record: dict[str, Any], prompt: RenderedPrompt, model_id: str):
    payload = record.get("response") or {}
    try:
        status = ResponseStatus(record.get("status", ResponseStatus.ERROR.value))
    except ValueError:
        status = ResponseStatus.ERROR
    response = ModelResponse(
        sample_id=prompt.sample_id,
        model_id=model_id,
        status=status,
        content=payload.get("content"),
        reasoning=payload.get("reasoning"),
        finish_reason=payload.get("finish_reason"),
        attempts=int(payload.get("attempts") or 1),
        latency_s=float(payload.get("latency_s") or 0.0),
        batch_id=payload.get("batch_id"),
        batch_size=payload.get("batch_size"),
        batch_index=payload.get("batch_index"),
        usage=payload.get("usage") or {},
        completion_tokens_est=payload.get("completion_tokens_est"),
    )
    score = SampleScore(
        metrics=stored_metrics(record.get("metrics")),
        prediction=record.get("prediction"),
        parse_ok=bool(record.get("parse_ok", True)),
        details=record.get("details") or {},
    )
    return prompt.sample, response, score


def _identity_of(record: dict[str, Any]) -> TaskIdentity:
    selection = record.get("selection_mode")
    return TaskIdentity(
        run_id=str(record.get("run_id") or ""),
        dataset_id=str(record.get("dataset_id")),
        model_id=str(record.get("model_id")),
        template_id=str(record.get("template_id")),
        template_version=str(record.get("template_version", "1.0")),
        prompt_mode=str(record.get("prompt_mode")),
        selection_mode=str(selection or "n/a"),
        data_delivery_mode=str(record.get("data_delivery_mode", "static")),
        task_kind=str(record.get("task_kind") or ""),
    )


def _append_records(path: Path, records: list[dict[str, Any]]) -> None:
    """Append judged copies; de-duplication later keeps the newest per sample."""
    blob = b"".join(orjson.dumps(record, default=str) + b"\n" for record in records)
    with path.open("ab") as handle:
        handle.write(blob)


async def _judge_run(
    run_dir: Path,
    config: RunConfig,
    *,
    dataset_filter: list[str] | None,
    limit: int | None,
) -> dict[str, Any]:
    reasoning_cfg = config.engine.reasoning_judge
    if not reasoning_cfg.model:
        raise ConfigError(
            "engine.reasoning_judge.model must name one of the run's models; set it in "
            "the run config or with -s engine.reasoning_judge.model=<id>"
        )
    # Only cot tasks are judged, so only cot bundles are worth building.
    # _build_bundles *prepares* every dataset it touches -- materializing the
    # split and drawing the sample -- so leaving io in the mode list makes this
    # pass pay that cost twice over, for bundles it then skips.
    config.modes.prompt_modes = [
        mode for mode in config.modes.prompt_modes if mode in (COT, SELF_CONSISTENCY)
    ] or [COT]
    # Point the engine at the run being judged rather than letting it mint a
    # new one. Constructing it creates run_dir and opens the engine log there,
    # so without this every offline pass left an otherwise-empty run directory
    # behind -- and its log, which belongs with the run whose records it judged.
    engine = EvaluationEngine(config, dry_run=True, run_dir=run_dir, run_id=run_dir.name)

    clients = {
        model.id: ModelClient(
            model,
            config.engine.timeouts,
            rediscover=config.engine.retry.recovery.rediscover_on_connection_error,
        )
        for model in config.models
        if model.id == reasoning_cfg.model
    }
    if not clients:
        raise ConfigError(
            f"engine.reasoning_judge.model={reasoning_cfg.model!r} is not in the run's "
            f"model list ({[m.id for m in config.models]})"
        )

    batch_disabled: set[str] = set()
    for model_id, client in clients.items():
        try:
            report = await client.verify()
        except Exception as exc:  # noqa: BLE001 - a probe must not end the pass
            logger.warning("judge model %s: endpoint probe failed (%s)", model_id, exc)
            continue
        if client.supports_batch and not report.get("batch_ok"):
            batch_disabled.add(model_id)
            logger.info(
                "judge model %s: no usable batch route (%s); sending one conversation "
                "per call instead",
                model_id,
                report.get("batch_error_class") or "probe failed",
            )

    stage = ReasoningJudgeStage(
        config=reasoning_cfg,
        registry=engine.registry,
        renderer=engine.renderer,
        clients=clients,
        retry_policy=engine.retry_policy,
        cache_dir=run_dir / "reasoning_judge_cache",
        batch_disabled=batch_disabled,
        calls=asyncio.Semaphore(reasoning_cfg.max_parallel_calls),
        log_path=run_dir / "reasoning_metrics.jsonl",
        # A post-pass writes its audit beside the same tasks a live run would.
        run_dir=run_dir,
    )

    summary: dict[str, Any] = {
        "tasks": [],
        "judged": 0,
        "statuses": Counter(),
        "skipped_interactive": [],
        "unmatched_records": 0,
        "prompts_from_raw": 0,
        "failed_tasks": [],
    }
    started = time.time()

    # Collect the work first, then run it. Judging one task at a time -- which
    # is what walking the tree and awaiting inline does -- leaves most of the
    # judge server idle: a task's own waves are already concurrent, but a small
    # task cannot fill 64 scheduler slots on its own. Tasks overlap here, and
    # the stage's shared semaphore is what actually bounds the load.
    jobs: list[tuple[Any, TaskIdentity, Path, list[dict[str, Any]], list[RenderedPrompt]]] = []
    for bundle in engine._build_bundles():  # noqa: SLF001 - stage A, reused deliberately
        if bundle.skipped or bundle.adapter is None:
            continue
        if bundle.modes.prompt_mode not in (COT, SELF_CONSISTENCY):
            continue
        if dataset_filter and bundle.config.id not in dataset_filter:
            continue
        if bundle.modes.data_delivery_mode == INTERACTIVE:
            summary["skipped_interactive"].append(bundle.config.id)
            continue

        by_id = {
            prompt.sample_id: prompt
            for prompt in _rendered_prompts(engine._build_prompt_set(bundle))  # noqa: SLF001
        }
        if not by_id:
            continue

        for model in config.evaluated_models():
            model_dir = run_dir / "datasets" / bundle.config.id / model.id
            if not model_dir.is_dir():
                continue
            # Only this bundle's own task directory. A dataset that poses
            # several tasks builds several bundles, and scanning every directory
            # from each of them judged the selection tasks once per bundle --
            # the same records, bought again.
            for task_dir in sorted(model_dir.iterdir()):
                if not task_dir.name.startswith(f"{bundle.modes.slug}@"):
                    continue
                records_path = task_dir / "records.jsonl"
                if not records_path.exists():
                    continue
                stored_records = dedupe_records(load_records(records_path))
                if not stored_records or stored_records[0].get("prompt_mode") not in (
                    COT,
                    SELF_CONSISTENCY,
                ):
                    continue
                answered = [
                    record
                    for record in stored_records
                    if record.get("status") == ResponseStatus.OK.value
                ]
                usable = [
                    record for record in answered if str(record.get("sample_id")) in by_id
                ]
                summary["unmatched_records"] += len(answered) - len(usable)
                if not usable:
                    continue
                if limit:
                    usable = usable[:limit]

                # Prefer the conversation actually sent; fall back to the
                # re-rendered one where no raw payload was kept.
                conversations = _stored_conversations(task_dir)
                task_prompts = []
                for record in usable:
                    base = by_id[str(record["sample_id"])]
                    conversation = conversations.get(str(record["sample_id"]))
                    task_prompts.append(
                        base
                        if conversation is None
                        else RenderedPrompt(
                            sample=base.sample,
                            messages=conversation,
                            template_id=base.template_id,
                            template_version=base.template_version,
                            sampling=base.sampling,
                            input_tokens_est=base.input_tokens_est,
                            output_contract=base.output_contract,
                        )
                    )
                summary["prompts_from_raw"] += sum(
                    1 for record in usable if str(record["sample_id"]) in conversations
                )
                jobs.append(
                    (bundle.adapter, _identity_of(usable[0]), records_path, usable, task_prompts)
                )

    # Enough tasks in flight to keep the judge's call budget full, and no more:
    # the real bound is the stage's own semaphore, which every task shares.
    # engine.concurrency.max_parallel_tasks is about not flooding the model
    # under test, and nothing here asks it anything.
    task_limit = max(1, reasoning_cfg.max_parallel_calls)
    logger.info(
        "judging %d record(s) across %d task(s), %d task(s) at a time, "
        "%d judge call(s) in flight",
        sum(len(records) for _a, _i, _p, records, _pr in jobs),
        len(jobs),
        task_limit,
        reasoning_cfg.max_parallel_calls,
    )
    task_slots = asyncio.Semaphore(task_limit)

    async def judge_task(adapter, identity, records_path, records, task_prompts) -> None:
        try:
            await _judge_one_task(adapter, identity, records_path, records, task_prompts)
        except Exception as exc:  # noqa: BLE001 - one task must not end the pass
            logger.exception(
                "judging %s failed: %s", records_path.parent.relative_to(run_dir), exc
            )
            summary["failed_tasks"].append(str(records_path.parent.relative_to(run_dir)))

    async def _judge_one_task(adapter, identity, records_path, records, task_prompts) -> None:
        triplets = [
            _triplet(record, prompt, identity.model_id)
            for record, prompt in zip(records, task_prompts, strict=True)
        ]
        async with task_slots:
            judged = await stage.apply(adapter, identity, task_prompts, triplets)
        refreshed: list[dict[str, Any]] = []
        for record, (_sample, _response, score) in zip(records, judged, strict=True):
            updated = dict(record)
            updated["metrics"] = dict(score.metrics)
            updated["details"] = dict(score.details)
            refreshed.append(updated)
            summary["statuses"][score.details.get("reasoning_metrics_status", "absent")] += 1
        _append_records(records_path, refreshed)
        summary["judged"] += len(refreshed)
        summary["tasks"].append(
            {"task": str(records_path.parent.relative_to(run_dir)), "records": len(refreshed)}
        )
        logger.info(
            "judged %d record(s) in %s",
            len(refreshed),
            records_path.parent.relative_to(run_dir),
        )

    try:
        await asyncio.gather(*(judge_task(*job) for job in jobs))
    finally:
        for client in clients.values():
            await client.aclose()

    summary["elapsed_s"] = round(time.time() - started, 1)
    summary["metrics_log"] = str(run_dir / "reasoning_metrics.jsonl")
    summary["judge_calls"] = stage.stats["calls"]
    summary["cache_hits"] = stage.stats["cached"]
    summary["failed_requests"] = stage.stats["failed"]
    summary["statuses"] = dict(summary["statuses"])
    return summary


def rejudge_reasoning(
    run_dir: Path | str,
    *,
    config_path: Path | str | None = None,
    dataset_filter: list[str] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run the reasoning judge over an existing run's stored COT outputs."""
    run_dir = Path(run_dir)
    if config_path:
        source = Path(config_path)
        if not source.exists():
            raise ConfigError(f"{source} not found -- cannot judge this run")
        # An author-written config may use `extends:` and ${env:...}, so it goes
        # through the full loader; the run's own resolved copy has neither.
        # The dataset filter goes in here rather than being applied later:
        # building a bundle *prepares* the dataset, so filtering afterwards
        # would materialize all 44 of them to judge one.
        config = load_run_config(source, dataset_filter=dataset_filter)
    else:
        source = run_dir / "run_config.resolved.yaml"
        if not source.exists():
            raise ConfigError(f"{source} not found -- cannot judge this run")
        config = RunConfig.model_validate(load_yaml(source))
        if dataset_filter:
            wanted = set(dataset_filter)
            config.datasets = [d for d in config.datasets if d.id in wanted]
    summary = asyncio.run(
        _judge_run(run_dir, config, dataset_filter=dataset_filter, limit=limit)
    )
    if summary["judged"]:
        summary["sync"] = sync_run(run_dir, config)
    return summary


def sync_run(run_dir: Path, config: RunConfig) -> dict[str, Any]:
    """Mirror the judged run off-box, if the run config configures a backup.

    A live run uploads continuously while it proceeds; an offline judging pass
    has nothing running to carry its output, so it uploads once at the end.
    Without this the metrics would exist only on a filesystem the README is
    explicit about not trusting.
    """
    if not config.engine.sync.enabled or not config.engine.sync.remote_path:
        return {"enabled": False}
    from .sync import ArtifactSync

    sync_config = config.engine.sync
    if not sync_config.exclude:
        # The raw per-batch payloads are the *inference* artifacts, and this
        # pass did not produce any: it read them, and the run that wrote them
        # has already backed them up. On the suite they are ~600 MB against
        # ~700 KB of metrics, so re-sending them to deliver the metrics costs
        # hours and changes nothing on the far end. This overrides even an
        # explicit `exclude: []`, because that setting is about what a *run*
        # should back up and this pass produces none of it.
        sync_config = sync_config.model_copy(update={"exclude": ["raw/"]})
    syncer = ArtifactSync(sync_config, run_dir, run_dir.name)
    problem = syncer.preflight()
    if problem:
        logger.warning("backup destination unusable, nothing uploaded: %s", problem)
        return {"enabled": True, "error": problem}
    stats = syncer.flush().as_dict()
    stats["destination"] = syncer.destination
    return {"enabled": True, **stats}
