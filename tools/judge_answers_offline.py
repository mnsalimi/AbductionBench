"""Run the ANSWER judge over answers already on disk -- nothing is generated.

`abench judge-reasoning` does this for the reasoning metrics; the answer judge
had no offline entry point, only a resume of the run -- which also GENERATES
every sample still missing. This walks the run folder instead: for each task
of the chosen models it rebuilds the samples from the run's own config, pairs
them with the stored replies (the conversation actually sent, from raw/, where
kept), and passes them through the same JudgeStage the engine uses, with the
same judge_cache/. A record whose score the verdict changes is appended with
the new metrics (the newest line per sample wins, as everywhere else).

Only replies that exist are judged: records stored as `error` (never
answered) are left alone, and a reply that never gave an answer is scored as
unreadable by the stage itself, not judged.

    .venv/bin/python tools/judge_answers_offline.py runs/<run> \\
        --config configs/runs/trio_judge_all.yaml \\
        --models gemma-4-e4b-local,gemma-4-e2b-local,qwen3-5-2b-local,qwen3-5-4b-local,qwen3-5-27b-openrouter
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import logging
import sys
import time
from pathlib import Path

from abductionbench.core.checkpoint import dedupe_records, load_records
from abductionbench.core.client import ModelClient
from abductionbench.core.config import load_run_config
from abductionbench.core.engine import EvaluationEngine
from abductionbench.core.judge import JudgeStage
from abductionbench.core.modes import INTERACTIVE
from abductionbench.core.rejudge import (
    _append_records,
    _rendered_prompts,
    _stored_conversations,
    _triplet,
)
from abductionbench.core.types import RenderedPrompt, ResponseStatus

logger = logging.getLogger("judge_answers_offline")
ANSWERED = {ResponseStatus.OK.value, ResponseStatus.EMPTY.value, ResponseStatus.TRUNCATED.value}


def _changed(record: dict, score) -> bool:
    return (
        dict(record.get("metrics") or {}) != dict(score.metrics)
        or dict(record.get("details") or {}) != dict(score.details)
        or record.get("prediction") != score.prediction
        or bool(record.get("parse_ok", True)) != bool(score.parse_ok)
    )


async def _run(run_dir: Path, config, models: list[str]) -> dict:
    judge_cfg = config.engine.judge
    engine = EvaluationEngine(config, dry_run=True, run_dir=run_dir, run_id=run_dir.name)
    judge_model = next(m for m in config.models if m.id == judge_cfg.model)
    client = ModelClient(
        judge_model,
        config.engine.timeouts,
        rediscover=config.engine.retry.recovery.rediscover_on_connection_error,
    )
    batch_disabled: set[str] = set()
    report = await client.verify()
    if client.supports_batch and not report.get("batch_ok"):
        batch_disabled.add(judge_model.id)
    # ONE STAGE PER TASK, as the engine does: apply() sets the judge template
    # on the stage itself, so a stage shared by concurrent tasks grades one
    # dataset with another's template. The stages share the call budget and
    # ONE in-memory cache: each writes the whole cache back when its task ends,
    # and separate copies would overwrite each other's new verdicts.
    calls = asyncio.Semaphore(judge_cfg.max_parallel_calls)
    shared_cache: dict | None = None

    def new_stage() -> JudgeStage:
        nonlocal shared_cache
        stage = JudgeStage(
            config=judge_cfg,
            registry=engine.registry,
            renderer=engine.renderer,
            clients={judge_model.id: client},
            retry_policy=engine.retry_policy,
            cache_dir=run_dir / "judge_cache",
            batch_disabled=batch_disabled,
            calls=calls,
        )
        if shared_cache is None:
            shared_cache = stage._cache  # noqa: SLF001
        else:
            stage._cache = shared_cache  # noqa: SLF001
        return stage

    summary = collections.Counter()
    per_model = collections.Counter()
    jobs = []
    for bundle in engine._build_bundles():  # noqa: SLF001 - stage A, reused deliberately
        if bundle.skipped or bundle.adapter is None:
            continue
        if bundle.modes.data_delivery_mode == INTERACTIVE:
            continue
        by_id = {p.sample_id: p for p in _rendered_prompts(engine._build_prompt_set(bundle))}  # noqa: SLF001
        for model_id in models:
            model_dir = run_dir / "datasets" / bundle.config.id / model_id
            if not model_dir.is_dir():
                continue
            for task_dir in sorted(model_dir.iterdir()):
                if not task_dir.name.startswith(f"{bundle.modes.slug}@"):
                    continue
                path = task_dir / "records.jsonl"
                if not path.exists():
                    continue
                records = [
                    r for r in dedupe_records(load_records(path))
                    if r.get("status") in ANSWERED and str(r.get("sample_id")) in by_id
                ]
                if not records:
                    continue
                conversations = _stored_conversations(task_dir)
                prompts = []
                for record in records:
                    base = by_id[str(record["sample_id"])]
                    sent = conversations.get(str(record["sample_id"]))
                    prompts.append(
                        base if sent is None else RenderedPrompt(
                            sample=base.sample, messages=sent,
                            template_id=base.template_id, template_version=base.template_version,
                            sampling=base.sampling, input_tokens_est=base.input_tokens_est,
                            output_contract=base.output_contract,
                        )
                    )
                jobs.append((bundle.adapter, model_id, path, records, prompts))

    slots = asyncio.Semaphore(max(1, judge_cfg.max_parallel_calls))

    async def one(adapter, model_id, path, records, prompts):
        triplets = [_triplet(r, p, model_id) for r, p in zip(records, prompts, strict=True)]
        stage = new_stage()
        async with slots:
            judged = await stage.apply(adapter, triplets, prompts)
        summary["skipped_no_answer"] += getattr(stage, "skipped_no_answer", 0)
        updated = []
        for record, (_s, _r, score) in zip(records, judged, strict=True):
            if _changed(record, score):
                row = dict(record)
                row.update(
                    metrics=dict(score.metrics), details=dict(score.details),
                    prediction=score.prediction, parse_ok=score.parse_ok,
                )
                updated.append(row)
        if updated:
            _append_records(path, updated)
        summary["records_seen"] += len(records)
        summary["records_updated"] += len(updated)
        per_model[model_id] += len(updated)
        logger.info("%s: %d of %d record(s) updated", path.parent.relative_to(run_dir), len(updated), len(records))

    started = time.time()
    try:
        results = await asyncio.gather(*(one(*job) for job in jobs), return_exceptions=True)
    finally:
        await client.aclose()
    failed = [str(job[2].parent.relative_to(run_dir)) for job, res in zip(jobs, results) if isinstance(res, BaseException)]
    for job, res in zip(jobs, results):
        if isinstance(res, BaseException):
            logger.error("judging %s failed: %r", job[2].parent.relative_to(run_dir), res)
    return {
        "tasks": len(jobs),
        "failed_tasks": failed,
        "per_model_updated": dict(per_model),
        "elapsed_s": round(time.time() - started, 1),
        **summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--config", required=True)
    parser.add_argument("--models", required=True, help="comma-separated evaluated model ids")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    config = load_run_config(args.config, model_filter=models)
    summary = asyncio.run(_run(args.run_dir.resolve(), config, models))
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 1 if summary["failed_tasks"] else 0


if __name__ == "__main__":
    sys.exit(main())
