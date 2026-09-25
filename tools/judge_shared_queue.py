"""Judge what is still owed with TWO gpt-oss-120b judges sharing one queue.

Nothing is asked of any model under test; only stored replies are judged.

  local   gpt-oss-120b-local      this box's vLLM (:18004), batches of 8
  remote  gpt-oss-120b-coreweave  OpenRouter pinned to CoreWeave (same weights)

Work, rebuilt from the records on disk every start -- so a stopped run resumes
where it stopped and nothing already judged is judged again:

  trio-reason   trio cot records with no reasoning verdict yet     LOCAL only
                (their cached verdicts are keyed to the local judge: free
                there, a paid miss on the remote one)
  answer        the answer judge over the five new models' io + cot LOCAL only
                (the local judge cache makes a re-run free)
  reason        the five new models' cot records with no reasoning  local OR remote
                verdict, in chunks of --chunk records

Each worker takes the next job it is eligible for; the local workers prefer
trio-reason, then answer, then reason. Both judges run to the end together,
so the total time is about the work divided by their combined rate.

A reasoning chunk goes through ONE complete ReasoningJudgeStage.apply, which
always runs its own waves in dependency order (steps, the observation
inventory and the option count first; every per-step family after the step
list it is indexed by), so splitting samples between judges never splits a
sample's dependencies. Every judged record says which judge judged it
(details.reasoning_judge_model / details.answer_judge_model).

Shared, single process: one reasoning cache (saved at most every 5 minutes,
atomically, and at the end), one audit log, one call budget for the local
server (the answer and reasoning judges both use it -- the server runs 256
sequences, and more would only queue against the read timeout).

    .venv/bin/python tools/judge_shared_queue.py runs/<run> --plan      # counts + estimate, no calls
    .venv/bin/python tools/judge_shared_queue.py runs/<run> --limit 40  # a small real probe
    .venv/bin/python tools/judge_shared_queue.py runs/<run>             # everything
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import logging
import sys
import time
from pathlib import Path

from abductionbench.core import reasoning_judge as rjm
from abductionbench.core.checkpoint import dedupe_records, load_records
from abductionbench.core.client import ModelClient
from abductionbench.core.config import load_run_config
from abductionbench.core.engine import EvaluationEngine
from abductionbench.core.judge import JudgeStage
from abductionbench.core.modes import COT, INTERACTIVE
from abductionbench.core.rejudge import (
    _append_records,
    _identity_of,
    _rendered_prompts,
    _stored_conversations,
    _triplet,
)
from abductionbench.core.types import RenderedPrompt, ResponseStatus

REASONING_KEYS = frozenset(rjm.REASONING_METRIC_COLUMNS)
REASONING_DETAILS = frozenset({"reasoning_metrics_status", "reasoning_judge_errors",
                               "reasoning_metrics_inapplicable", "reasoning_lists",
                               "reasoning_source", "reasoning_judge_model"})

log = logging.getLogger("judge_shared_queue")
CONFIG = "configs/runs/judge_shared_queue.yaml"
LOCAL, REMOTE = "gpt-oss-120b-local", "gpt-oss-120b-coreweave"
TRIO = ("gpt-5.6-luna-openrouter", "gemini-3.8-flash-openrouter", "gemma-4-31b-it-openrouter")
NEW = ("gemma-4-e4b-local", "gemma-4-e2b-local", "qwen3-5-2b-local", "qwen3-5-4b-local",
       "qwen3-5-27b-openrouter")
ANSWERED = {ResponseStatus.OK.value, ResponseStatus.EMPTY.value, ResponseStatus.TRUNCATED.value}
CALLS_PER_SAMPLE = 14            # measured: 280 calls for 20 new-model records (probe, 2026-09-25)
PRICE_IN, PRICE_OUT = 0.03e-6, 0.17e-6
TOKENS_IN, TOKENS_OUT = 3000, 500


class ThrottledStage(rjm.ReasoningJudgeStage):
    """Saves the (shared, ~130 MB) cache at most every 5 minutes."""

    last_save = 0.0

    def _save_cache(self, force: bool = False) -> None:
        if force or time.monotonic() - ThrottledStage.last_save >= 300:
            ThrottledStage.last_save = time.monotonic()
            super()._save_cache()


class Job:
    __slots__ = ("kind", "adapter", "model_id", "path", "records", "prompts", "eligible")

    def __init__(self, kind, adapter, model_id, path, records, prompts, eligible):
        self.kind, self.adapter, self.model_id, self.path = kind, adapter, model_id, path
        self.records, self.prompts, self.eligible = records, prompts, eligible


def _prompts_for(records, by_id, conversations):
    out = []
    for record in records:
        base = by_id[str(record["sample_id"])]
        sent = conversations.get(str(record["sample_id"]))
        out.append(base if sent is None else RenderedPrompt(
            sample=base.sample, messages=sent, template_id=base.template_id,
            template_version=base.template_version, sampling=base.sampling,
            input_tokens_est=base.input_tokens_est, output_contract=base.output_contract))
    return out


def build_jobs(engine, run_dir: Path, chunk: int, limit: int | None, answers: bool):
    jobs: list[Job] = []
    counts = collections.Counter()
    latest: dict[tuple, dict] = {}   # (records path, sample_id, fingerprint) -> the record as it is now
    for bundle in engine._build_bundles():  # noqa: SLF001 - stage A, reused deliberately
        if bundle.skipped or bundle.adapter is None or bundle.modes.data_delivery_mode == INTERACTIVE:
            continue
        by_id = {p.sample_id: p for p in _rendered_prompts(engine._build_prompt_set(bundle))}  # noqa: SLF001
        is_cot = bundle.modes.prompt_mode == COT
        for model_id in (*TRIO, *NEW):
            model_dir = run_dir / "datasets" / bundle.config.id / model_id
            if not model_dir.is_dir():
                continue
            for task_dir in sorted(model_dir.iterdir()):
                if not task_dir.name.startswith(f"{bundle.modes.slug}@"):
                    continue
                path = task_dir / "records.jsonl"
                if not path.exists():
                    continue
                records = [r for r in dedupe_records(load_records(path))
                           if str(r.get("sample_id")) in by_id]
                for r in records:
                    latest[(path, str(r.get("sample_id")), str(r.get("prompt_fingerprint", "")))] = r
                conversations = None
                if is_cot:
                    # REASONING: only OK replies (as `abench judge-reasoning`),
                    # and only those with no reasoning verdict yet.
                    todo = [r for r in records if r.get("status") == ResponseStatus.OK.value
                            and "reasoning_metrics_status" not in (r.get("details") or {})]
                    counts[f"reason_done:{model_id in TRIO and 'trio' or 'new'}"] += sum(
                        1 for r in records if "reasoning_metrics_status" in (r.get("details") or {}))
                    if todo:
                        conversations = _stored_conversations(task_dir)
                        kind = "trio-reason" if model_id in TRIO else "reason"
                        eligible = {"local"} if model_id in TRIO else {"local", "remote"}
                        for start in range(0, len(todo), chunk):
                            part = todo[start:start + chunk]
                            jobs.append(Job(kind, bundle.adapter, model_id, path, part,
                                            _prompts_for(part, by_id, conversations), eligible))
                            counts[kind] += len(part)
                if answers and model_id in NEW:
                    answered = [r for r in records if r.get("status") in ANSWERED]
                    if answered:
                        conversations = conversations or _stored_conversations(task_dir)
                        jobs.append(Job("answer", bundle.adapter, model_id, path, answered,
                                        _prompts_for(answered, by_id, conversations), {"local"}))
                        counts["answer"] += len(answered)
    if limit:
        kept, n = [], 0
        for job in jobs:
            if n >= limit:
                break
            job.records, job.prompts = job.records[: limit - n], job.prompts[: limit - n]
            n += len(job.records)
            kept.append(job)
        jobs = kept
    return jobs, counts, latest


def merge(current: dict, score, kind: str, judge_id: str) -> dict:
    """The record with ONLY this judge's fields replaced.

    An answer job and a reasoning job can hold the same record, each from its
    own snapshot. Writing either snapshot whole would drop the other judge's
    result, so each update is merged into the record as it is at write time:
    the answer judge owns every non-reasoning metric and detail plus the
    prediction; the reasoning judge owns the reasoning metrics and details.
    """
    metrics, details = dict(current.get("metrics") or {}), dict(current.get("details") or {})
    new_m, new_d = dict(score.metrics), dict(score.details)
    row = dict(current)
    if kind == "answer":
        m = {k: v for k, v in new_m.items() if k not in REASONING_KEYS}
        m.update({k: v for k, v in metrics.items() if k in REASONING_KEYS})
        d = {k: v for k, v in new_d.items() if k not in REASONING_DETAILS}
        d.update({k: v for k, v in details.items() if k in REASONING_DETAILS})
        d["answer_judge_model"] = judge_id
        row.update(metrics=m, details=d, prediction=score.prediction, parse_ok=score.parse_ok)
    else:
        m = {k: v for k, v in metrics.items() if k not in REASONING_KEYS}
        m.update({k: v for k, v in new_m.items() if k in REASONING_KEYS})
        d = {k: v for k, v in details.items() if k not in REASONING_DETAILS}
        d.update({k: v for k, v in new_d.items() if k in REASONING_DETAILS})
        d["reasoning_judge_model"] = judge_id
        row.update(metrics=m, details=d)
    return row


class _Score:
    """A merged row seen as a score, for _changed()."""

    def __init__(self, row):
        self.metrics, self.details = row.get("metrics") or {}, row.get("details") or {}
        self.prediction, self.parse_ok = row.get("prediction"), row.get("parse_ok", True)


def _changed(record, score) -> bool:
    return (dict(record.get("metrics") or {}) != dict(score.metrics)
            or dict(record.get("details") or {}) != dict(score.details)
            or record.get("prediction") != score.prediction
            or bool(record.get("parse_ok", True)) != bool(score.parse_ok))


async def run(run_dir: Path, args) -> int:
    config = load_run_config(CONFIG)
    engine = EvaluationEngine(config, dry_run=True, run_dir=run_dir, run_id=run_dir.name)
    jobs, counts, latest = build_jobs(engine, run_dir, args.chunk, args.limit, not args.no_answers)
    reason_new = sum(len(j.records) for j in jobs if j.kind == "reason")
    trio_left = sum(len(j.records) for j in jobs if j.kind == "trio-reason")
    answer_n = sum(len(j.records) for j in jobs if j.kind == "answer")
    calls = reason_new * CALLS_PER_SAMPLE
    log.info("to judge: %d new-model cot record(s) (~%d calls), %d trio remainder, %d record(s) "
             "for the answer judge; already judged: %s", reason_new, calls, trio_left, answer_n,
             {k: v for k, v in counts.items() if k.startswith("reason_done")})
    for local_rate, remote_rate in ((4.0, 10.0), (5.0, 15.0)):
        share = remote_rate / (local_rate + remote_rate)
        hours = calls / (local_rate + remote_rate) / 3600
        cost = calls * share * (TOKENS_IN * PRICE_IN + TOKENS_OUT * PRICE_OUT)
        log.info("estimate at local %.0f/s + remote %.0f/s: ~%.1f h, ~$%.0f on OpenRouter",
                 local_rate, remote_rate, hours, cost)
    if args.probe:
        # One small real chunk for EACH judge, so both paths are tested end to end.
        reason_jobs = [j for j in jobs if j.kind == "reason"]
        picked = []
        for job, side in zip(reason_jobs[:2], ("local", "remote")):
            job.records, job.prompts, job.eligible = job.records[:20], job.prompts[:20], {side}
            picked.append(job)
        jobs = picked
        reason_new = sum(len(j.records) for j in jobs)
        log.info("PROBE: %s", [(sorted(j.eligible)[0], str(j.path.parent.relative_to(run_dir)), len(j.records)) for j in jobs])
    if args.plan or not jobs:
        return 0

    rj_cfg = config.engine.reasoning_judge
    local_model = next(m for m in config.models if m.id == LOCAL)
    remote_model = next(m for m in config.models if m.id == REMOTE)
    timeouts, rediscover = config.engine.timeouts, config.engine.retry.recovery.rediscover_on_connection_error
    local_client = ModelClient(local_model, timeouts, rediscover=rediscover)
    remote_client = ModelClient(remote_model, timeouts, rediscover=rediscover)
    local_report = await local_client.verify()
    await remote_client.verify()
    local_batch_off = set() if (local_client.supports_batch and local_report.get("batch_ok")) else {LOCAL}

    # ONE call budget for everything that queues on the local server.
    local_calls = asyncio.Semaphore(args.local_calls)
    remote_calls = asyncio.Semaphore(args.remote_calls)
    common = dict(registry=engine.registry, renderer=engine.renderer, retry_policy=engine.retry_policy,
                  cache_dir=run_dir / "reasoning_judge_cache",
                  log_path=run_dir / "reasoning_metrics.jsonl", run_dir=run_dir)
    stages = {
        "local": ThrottledStage(config=rj_cfg.model_copy(update={"model": LOCAL}),
                                clients={LOCAL: local_client}, batch_disabled=local_batch_off,
                                calls=local_calls, **common),
        "remote": ThrottledStage(config=rj_cfg.model_copy(update={"model": REMOTE}),
                                 clients={REMOTE: remote_client}, batch_disabled={REMOTE},
                                 calls=remote_calls, **common),
    }
    stages["remote"]._cache = stages["local"]._cache  # noqa: SLF001 - one cache, one writer
    judge_ids = {"local": LOCAL, "remote": REMOTE}
    answer_cache: dict | None = None

    def answer_stage() -> JudgeStage:
        nonlocal answer_cache
        stage = JudgeStage(config=config.engine.judge, registry=engine.registry, renderer=engine.renderer,
                           clients={LOCAL: local_client}, retry_policy=engine.retry_policy,
                           cache_dir=run_dir / "judge_cache", batch_disabled=local_batch_off,
                           calls=local_calls)
        if answer_cache is None:
            answer_cache = stage._cache  # noqa: SLF001
        else:
            stage._cache = answer_cache  # noqa: SLF001
        return stage

    # "reason" is the shared queue; a reasoning job pinned to one judge (only
    # the probe pins them) waits in that judge's own queue instead.
    queues = {k: collections.deque() for k in ("trio-reason", "answer", "reason", "local-only", "remote-only")}
    for job in jobs:
        if job.kind == "reason" and job.eligible == {"local"}:
            queues["local-only"].append(job)
        elif job.kind == "reason" and job.eligible == {"remote"}:
            queues["remote-only"].append(job)
        else:
            queues[job.kind].append(job)
    preference = {"local": ("trio-reason", "local-only", "answer", "reason"),
                  "remote": ("remote-only", "reason")}
    done = collections.Counter()
    failed: list[str] = []
    started = time.time()

    def take(side: str) -> Job | None:
        for kind in preference[side]:
            if queues[kind]:
                return queues[kind].popleft()
        return None

    async def do(job: Job, side: str) -> None:
        judge_id = judge_ids[side]
        triplets = [_triplet(r, p, job.model_id) for r, p in zip(job.records, job.prompts, strict=True)]
        if job.kind == "answer":
            judged = await answer_stage().apply(job.adapter, triplets, job.prompts)
        else:
            judged = await stages[side].apply(job.adapter, _identity_of(job.records[0]), job.prompts, triplets)
        updated = []
        # Merged at WRITE time, into the record as it is now (see merge()).
        for record, (_s, _r, score) in zip(job.records, judged, strict=True):
            key = (job.path, str(record.get("sample_id")), str(record.get("prompt_fingerprint", "")))
            current = latest.get(key, record)
            row = merge(current, score, "answer" if job.kind == "answer" else "reason", judge_id)
            if job.kind != "answer" or _changed(current, _Score(row)):
                latest[key] = row
                updated.append(row)
        if updated:
            _append_records(job.path, updated)
        done[f"{job.kind}@{side}"] += len(job.records)

    async def worker(side: str) -> None:
        while (job := take(side)) is not None:
            try:
                await do(job, side)
            except Exception as exc:  # noqa: BLE001 - one chunk must not end the pass
                log.exception("%s job %s failed: %s", side, job.path.parent.relative_to(run_dir), exc)
                failed.append(f"{job.kind}:{job.path.parent.relative_to(run_dir)}")

    async def progress() -> None:
        while True:
            await asyncio.sleep(60)
            elapsed = time.time() - started
            reason_done = done["reason@local"] + done["reason@remote"]
            rate = reason_done / elapsed if elapsed else 0
            eta = (reason_new - reason_done) / rate / 3600 if rate else float("nan")
            log.info("progress %.0f min: %s | reasoning calls local %s remote %s | new-model ETA %.1f h",
                     elapsed / 60, dict(done), stages["local"].stats.get("calls"),
                     stages["remote"].stats.get("calls"), eta)

    ticker = asyncio.create_task(progress())
    try:
        sides = ["local"] * args.local_jobs + ([] if args.no_remote else ["remote"] * args.remote_jobs)
        await asyncio.gather(*(worker(s) for s in sides))
    finally:
        ticker.cancel()
        stages["local"]._save_cache(force=True)  # noqa: SLF001
        await local_client.aclose()
        await remote_client.aclose()
    log.info("DONE in %.1f h: %s; failed jobs: %d %s; local stats %s; remote stats %s",
             (time.time() - started) / 3600, dict(done), len(failed), failed[:10],
             stages["local"].stats, stages["remote"].stats)
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--plan", action="store_true", help="count the work and estimate; call nothing")
    ap.add_argument("--limit", type=int, default=None, help="judge at most this many records")
    ap.add_argument("--probe", action="store_true", help="20 real records on EACH judge, then stop")
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--local-jobs", type=int, default=12)
    ap.add_argument("--remote-jobs", type=int, default=24)
    ap.add_argument("--local-calls", type=int, default=32, help="x8 per batch = 256, the server's MAX_NUM_SEQS")
    ap.add_argument("--remote-calls", type=int, default=128)
    ap.add_argument("--no-answers", action="store_true")
    ap.add_argument("--no-remote", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return asyncio.run(run(args.run_dir.resolve(), args))


if __name__ == "__main__":
    sys.exit(main())
