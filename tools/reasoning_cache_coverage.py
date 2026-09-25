"""How much of a reasoning re-judge the cache already covers -- READ-ONLY.

Runs the real re-judge code path (core/rejudge.py -> ReasoningJudgeStage) over
the chosen models' cot records, with three things swapped out so that nothing
is asked and nothing in the run folder changes:

  * the judge client is a stand-in that answers every call with an immediate
    "not in cache" error -- no request leaves this machine;
  * the stage reads a SCRATCH COPY of reasoning_judge_cache/, and writes its
    audit log and per-task call logs nowhere;
  * records are not written back, and the engine is built against a scratch
    directory, so not even the run's engine.log is touched.

Every judge request is built exactly as a real re-judge would build it and
looked up in the cache under the same key. Printed per family: requests, cache
hits, misses. A family that depends on the step list (all but option_count and
observation_inventory) can only be built when the step list itself hit; the
samples whose step list missed are counted separately -- each of them needs
its step list and every downstream family asked.

    .venv/bin/python tools/reasoning_cache_coverage.py runs/<run> \\
        [--config configs/runs/trio_judge_all.yaml] \\
        [--models gpt-5.6-luna-openrouter,gemini-3.8-flash-openrouter,gemma-4-31b-it-openrouter] \\
        [--out coverage.txt]
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import shutil
import sys
import tempfile
from pathlib import Path

import abductionbench.core.rejudge as rejudge
from abductionbench.core import reasoning_judge as rjm
from abductionbench.core.config import load_run_config
from abductionbench.core.engine import EvaluationEngine
from abductionbench.core.errors import InvalidRequestError

TRIO = "gpt-5.6-luna-openrouter,gemini-3.8-flash-openrouter,gemma-4-31b-it-openrouter"
DOWNSTREAM_OF_STEPS = {
    "observation_coverage", "directionality", "step_directionality",
    "differential_elimination", "uncertainty", "prior_knowledge",
    "unresolved_contradiction", "anchoring_point", "proof_disproof",
    "helpfulness", "branchiness_selection", "branchiness_generation",
}

requests_by_family: collections.Counter = collections.Counter()
hits_by_family: collections.Counter = collections.Counter()


class OfflineClient:
    """Answers nothing: a miss fails at once, and is not retried."""

    def __init__(self, model, *_args, **_kwargs):
        self.model = model

    @property
    def supports_batch(self) -> bool:
        return False

    async def verify(self, **_kwargs):
        return {"batch_ok": False}

    async def chat_single(self, *_args, **_kwargs):
        raise InvalidRequestError("coverage check: not in the cache, not asked")

    async def chat_batch(self, *_args, **_kwargs):
        raise InvalidRequestError("coverage check: not in the cache, not asked")

    async def aclose(self) -> None:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--config", default="configs/runs/trio_judge_all.yaml")
    parser.add_argument("--models", default=TRIO)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="records per task (a smoke test)")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    scratch = Path(tempfile.mkdtemp(prefix="rj-coverage-"))
    cache_copy = scratch / "reasoning_judge_cache"
    source_cache = run_dir / "reasoning_judge_cache"
    if source_cache.is_dir():
        shutil.copytree(source_cache, cache_copy)
    else:
        cache_copy.mkdir()

    class CountingStage(rjm.ReasoningJudgeStage):
        def __init__(self, **kwargs):
            kwargs.update(cache_dir=cache_copy, log_path=None, run_dir=None)
            super().__init__(**kwargs)

        async def _judge_many(self, family, requests, **kwargs):
            template = self.templates[family]
            for fields in requests.values():
                key = rjm.stable_hash(
                    {"template": template.ref, "judge": self.config.model, "fields": fields},
                    length=32,
                )
                requests_by_family[family] += 1
                hits_by_family[family] += key in self._cache
            return await super()._judge_many(family, requests, **kwargs)

    def scratch_engine(config, **_kwargs):
        return EvaluationEngine(config, dry_run=True, run_dir=scratch / "engine", run_id=run_dir.name)

    rejudge.ModelClient = OfflineClient
    rejudge.ReasoningJudgeStage = CountingStage
    rejudge.EvaluationEngine = scratch_engine
    rejudge._append_records = lambda _path, _records: None

    config = load_run_config(args.config, model_filter=models)
    summary = asyncio.run(rejudge._judge_run(run_dir, config, dataset_filter=None, limit=args.limit))

    steps_requests = requests_by_family["steps"]
    steps_misses = steps_requests - hits_by_family["steps"]
    per_sample = [
        requests_by_family[f] / max(1, hits_by_family["steps"])
        for f in DOWNSTREAM_OF_STEPS
        if requests_by_family[f]
    ]
    downstream_per_sample = sum(per_sample)
    lines = [
        f"reasoning-judge cache coverage -- {run_dir.name}",
        f"models: {', '.join(models)}",
        f"cot records looked at: {sum(t['records'] for t in summary['tasks'])}"
        f" in {len(summary['tasks'])} task(s)",
        "",
        f"{'family':28} {'requests':>9} {'cache hits':>11} {'misses':>8}",
    ]
    total_req = total_hit = 0
    for family in sorted(requests_by_family):
        req, hit = requests_by_family[family], hits_by_family[family]
        total_req += req
        total_hit += hit
        lines.append(f"{family:28} {req:>9} {hit:>11} {req - hit:>8}")
    lines += [
        f"{'TOTAL (built)':28} {total_req:>9} {total_hit:>11} {total_req - total_hit:>8}",
        "",
        f"samples whose step list missed: {steps_misses} -- their downstream families could not",
        f"be built here; each needs its step list plus ~{downstream_per_sample:.1f} downstream calls,",
        f"about {int(steps_misses * downstream_per_sample)} more calls in all.",
        f"=> judge calls a real re-judge would make: about "
        f"{(total_req - total_hit) + int(steps_misses * downstream_per_sample)}",
    ]
    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")
    shutil.rmtree(scratch, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
