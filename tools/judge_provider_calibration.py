"""Is a remote gpt-oss-120b host distinguishable from our local vLLM judge?

Three readings of EXACTLY THE SAME reasoning-judge prompts:

  A  the local verdict already in the run's cache (what the run used)
  B  the local judge asked again, no cache        -> A vs B = local noise floor
  C  the remote host (configs/runs/judge_calibration_coreweave.yaml: CoreWeave,
     MXFP4, pinned, every parameter required)     -> A vs C = remote difference

The requests are rebuilt by the real re-judge code path (rejudge ->
ReasoningJudgeStage) over a sample of already-judged cot records, with a judge
that answers nothing, so only requests the cache answers are kept (their A is
known). B and C are asked through a real ReasoningJudgeStage with an EMPTY
scratch cache, so the prompt bytes, sampling and parsing are the ones a run
uses. Nothing in the run folder is written.

For every judge output key: per-step lists are compared element by element
(when the lengths match; a length mismatch counts as full disagreement) and by
their sum (the numerator of the metric); scalars exactly; the step list by its
length. The verdict: if A-vs-C disagreement is within the A-vs-B noise, the
remote judge cannot be told apart from ours.

    .venv/bin/python tools/judge_provider_calibration.py runs/<run> [--per-task 5]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import abductionbench.core.rejudge as rejudge
from abductionbench.core import reasoning_judge as rjm
from abductionbench.core.client import ModelClient
from abductionbench.core.config import load_run_config
from abductionbench.core.engine import EvaluationEngine
from abductionbench.core.errors import InvalidRequestError

CONFIG = "configs/runs/judge_calibration_coreweave.yaml"
LOCAL_ID, REMOTE_ID = "gpt-oss-120b-local", "gpt-oss-120b-openrouter"


class OfflineClient:
    def __init__(self, model, *_a, **_k):
        self.model = model

    supports_batch = False

    async def verify(self, **_k):
        return {"batch_ok": False}

    async def chat_single(self, *_a, **_k):
        raise InvalidRequestError("calibration record pass: not asked")

    chat_batch = chat_single

    async def aclose(self):
        return None


def record(run_dir: Path, per_task: int, scratch: Path, config_path: str = CONFIG):
    """(family, request_id, fields, cached values) for every cache-answered request."""
    recorded: list[tuple[str, str, dict, dict]] = []
    cache_copy = scratch / "record_cache"
    for attempt in range(5):  # the live pass may be mid-write
        try:
            shutil.rmtree(cache_copy, ignore_errors=True)
            shutil.copytree(run_dir / "reasoning_judge_cache", cache_copy)
            json.loads((cache_copy / "verdicts.json").read_bytes())
            break
        except (json.JSONDecodeError, OSError):
            time.sleep(5)

    class RecordingStage(rjm.ReasoningJudgeStage):
        def __init__(self, **kw):
            kw.update(cache_dir=cache_copy, log_path=None, run_dir=None)
            super().__init__(**kw)

        async def _judge_many(self, family, requests, **kw):
            template = self.templates[family]
            for rid, fields in requests.items():
                key = rjm.stable_hash(
                    {"template": template.ref, "judge": self.config.model, "fields": fields}, length=32
                )
                hit = self._cache.get(key)
                if hit and isinstance(hit.get("values"), dict):
                    recorded.append((family, f"{family}::{key}", fields, hit["values"]))
            return await super()._judge_many(family, requests, **kw)

    rejudge.ModelClient = OfflineClient
    rejudge.ReasoningJudgeStage = RecordingStage
    rejudge.EvaluationEngine = lambda config, **_k: EvaluationEngine(
        config, dry_run=True, run_dir=scratch / "engine", run_id=run_dir.name
    )
    rejudge._append_records = lambda _p, _r: None
    trio = ["gpt-5.6-luna-openrouter", "gemini-3.8-flash-openrouter", "gemma-4-31b-it-openrouter"]
    config = load_run_config(config_path, model_filter=trio)
    asyncio.run(rejudge._judge_run(run_dir, config, dataset_filter=None, limit=per_task))
    # one entry per distinct request (the same prompt can recur across repeats)
    seen, unique = set(), []
    for item in recorded:
        if item[1] not in seen:
            seen.add(item[1])
            unique.append(item)
    return unique, config


async def replay(config, recorded, target_id: str, parallel: int, scratch: Path):
    """Ask `target_id` every recorded request through a real stage with no cache."""
    engine = EvaluationEngine(config, dry_run=True, run_dir=scratch / "engine2", run_id="calibration")
    model = next(m for m in config.models if m.id == target_id)
    client = ModelClient(model, config.engine.timeouts,
                         rediscover=config.engine.retry.recovery.rediscover_on_connection_error)
    report = await client.verify()
    batch_disabled = set() if (client.supports_batch and report.get("batch_ok")) else {config.engine.reasoning_judge.model}
    stage = rjm.ReasoningJudgeStage(
        config=config.engine.reasoning_judge, registry=engine.registry, renderer=engine.renderer,
        clients={config.engine.reasoning_judge.model: client}, retry_policy=engine.retry_policy,
        cache_dir=scratch / f"cache_{target_id}", batch_disabled=batch_disabled,
        calls=asyncio.Semaphore(parallel), log_path=None, run_dir=None,
    )
    by_family: dict[str, dict[str, dict]] = defaultdict(dict)
    for family, rid, fields, _v in recorded:
        by_family[family][rid] = fields
    out: dict[str, dict | None] = {}
    try:
        results = await asyncio.gather(*(stage._judge_many(f, reqs) for f, reqs in by_family.items()))
        for res in results:
            out.update(res)
    finally:
        await client.aclose()
    return out, dict(stage.stats)


def compare(a, b):
    """{output key: (agreement 0..1, |sum difference| or None)} for one request."""
    out = {}
    if a is None or b is None:
        return None
    for key in set(a) | set(b):
        x, y = a.get(key), b.get(key)
        if isinstance(x, list) and x and all(isinstance(s, str) for s in x):  # the step list itself
            out[key + " (length)"] = (float(len(x) == len(y or [])), abs(len(x) - len(y or [])))
        elif isinstance(x, list) or isinstance(y, list):
            if not isinstance(x, list) or not isinstance(y, list) or len(x) != len(y):
                out[key] = (0.0, None)
            elif not x:
                out[key] = (1.0, 0.0)
            else:
                agree = sum(1 for p, q in zip(x, y) if p == q) / len(x)
                try:
                    diff = abs(sum(float(v) for v in x) - sum(float(v) for v in y))
                except (TypeError, ValueError):
                    diff = None
                out[key] = (agree, diff)
        else:
            out[key] = (float(x == y), None)
    return out


def summarise(recorded, B, C):
    rows = defaultdict(lambda: {"n": 0, "AB": [], "AC": [], "BC": [], "dAB": [], "dAC": []})
    failures = {"B": 0, "C": 0}
    pairs = []
    for family, rid, _f, A in recorded:
        b, c = B.get(rid), C.get(rid)
        failures["B"] += b is None
        failures["C"] += c is None
        if b is None or c is None:
            continue
        ab, ac, bc = compare(A, b), compare(A, c), compare(b, c)
        for key in ab:
            r = rows[(family, key)]
            r["n"] += 1
            r["AB"].append(ab[key][0]); r["AC"].append(ac.get(key, (0.0, None))[0]); r["BC"].append(bc.get(key, (0.0, None))[0])
            if ab[key][1] is not None and ac.get(key, (0, None))[1] is not None:
                r["dAB"].append(ab[key][1]); r["dAC"].append(ac[key][1])
            pairs.append((ab[key][0], ac.get(key, (0.0, None))[0]))
    return rows, failures, pairs


def bootstrap_ci(pairs, reps=2000, seed=0):
    rng = random.Random(seed)
    diffs = []
    n = len(pairs)
    for _ in range(reps):
        s = [pairs[rng.randrange(n)] for _ in range(n)]
        diffs.append(sum(p[1] for p in s) / n - sum(p[0] for p in s) / n)
    diffs.sort()
    return diffs[int(0.025 * reps)], diffs[int(0.975 * reps)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--per-task", type=int, default=5)
    ap.add_argument("--local-parallel", type=int, default=8, help="local judge calls in flight (x8 per batch)")
    ap.add_argument("--remote-parallel", type=int, default=64)
    ap.add_argument("--max-requests", type=int, default=0, help="cap on requests (0 = all) -- for a probe")
    ap.add_argument("--config", default=CONFIG, help="run config holding both judge entries")
    ap.add_argument("--remote", default=REMOTE_ID, help="judge id of the remote host under test")
    ap.add_argument("--skip-local", action="store_true",
                    help="do not re-ask the local judge; compare with the local-vs-local baseline "
                         "already measured (analysis/judge_calibration/summary.json)")
    args = ap.parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = run_dir / "analysis" / "judge_calibration" / ("" if args.remote == REMOTE_ID else args.remote)
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="judge-calib-"))

    t0 = time.time()
    recorded, config = record(run_dir, args.per_task, scratch, args.config)
    if args.max_requests:
        recorded = recorded[: args.max_requests]
    print(f"recorded {len(recorded)} cache-answered requests in {time.time() - t0:.0f}s", flush=True)

    async def both():
        remote = replay(config, recorded, args.remote, args.remote_parallel, scratch)
        if args.skip_local:
            return (({}, {}), await remote)
        return await asyncio.gather(replay(config, recorded, LOCAL_ID, args.local_parallel, scratch), remote)

    (B, b_stats), (C, c_stats) = asyncio.run(both())
    if args.skip_local:
        # No local re-ask: B is the cached verdict itself (agreement 1), and the
        # local noise floor is the baseline measured on 2026-09-25.
        B = {rid: vals for _f, rid, _fields, vals in recorded}
    rows, failures, pairs = summarise(recorded, B, C)
    lo, hi = bootstrap_ci(pairs) if pairs else (float("nan"), float("nan"))
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")  # noqa: E731

    table = []
    for (family, key), r in sorted(rows.items()):
        table.append({
            "family": family, "output": key, "n": r["n"],
            "agree_local_vs_local": round(mean(r["AB"]), 3), "agree_local_vs_remote": round(mean(r["AC"]), 3),
            "agree_rerun_vs_remote": round(mean(r["BC"]), 3),
            "sum_absdiff_local_vs_local": round(mean(r["dAB"]), 3) if r["dAB"] else None,
            "sum_absdiff_local_vs_remote": round(mean(r["dAC"]), 3) if r["dAC"] else None,
        })
    overall_ab, overall_ac = mean([p[0] for p in pairs]), mean([p[1] for p in pairs])
    summary = {
        "requests": len(recorded), "failed_local_rerun": failures["B"], "failed_remote": failures["C"],
        "compared_outputs": len(pairs),
        "agreement_local_vs_local": round(overall_ab, 4), "agreement_local_vs_remote": round(overall_ac, 4),
        "difference_remote_minus_local": round(overall_ac - overall_ab, 4),
        "difference_95ci": [round(lo, 4), round(hi, 4)],
        "disagreement_ratio_remote_over_local": round((1 - overall_ac) / (1 - overall_ab), 3) if overall_ab < 1 else None,
        "local_stage_stats": b_stats, "remote_stage_stats": c_stats,
        "elapsed_s": round(time.time() - t0, 1),
    }
    if args.skip_local:
        base = json.loads((run_dir / "analysis" / "judge_calibration" / "summary.json").read_text())
        summary["baseline_local_vs_local"] = base.get("agreement_local_vs_local")
        summary["baseline_local_vs_coreweave"] = base.get("agreement_local_vs_remote")
        summary["difference_vs_local_baseline"] = round(overall_ac - base["agreement_local_vs_local"], 4)
        summary["note"] = "skip-local: agreement_local_vs_local here is 1 by construction; compare with the baseline"
    summary["remote"] = args.remote
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    import csv
    with (out_dir / "per_output.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0].keys()) if table else ["family"])
        w.writeheader(); w.writerows(table)
    print(json.dumps(summary, indent=2))
    for row in table:
        print(f"{row['family']:26} {row['output']:34} n={row['n']:5} "
              f"A~B {row['agree_local_vs_local']:.3f}  A~C {row['agree_local_vs_remote']:.3f}  "
              f"B~C {row['agree_rerun_vs_remote']:.3f}  |Δsum| {row['sum_absdiff_local_vs_local']} vs {row['sum_absdiff_local_vs_remote']}")
    shutil.rmtree(scratch, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
