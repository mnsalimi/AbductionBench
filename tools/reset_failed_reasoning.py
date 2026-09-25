"""Queue again the new-model cot samples whose reasoning judging FAILED somewhere.

A sample is re-judged by tools/judge_shared_queue.py only if it carries no
reasoning verdict (details.reasoning_metrics_status). A sample judged while a
judge call failed -- a timeout, a rate limit that outlasted its retries, a step
list cut off by its token budget -- does carry one ("partial"), so it would
never be asked again. This appends a copy of each such record with its
reasoning metrics and reasoning details removed (the newest line per sample
wins; the old line stays in the file, so nothing is lost), which puts it back
in the queue. On the re-run every family that succeeded is read from the cache
for free; only what failed is asked.

"Failed" = details.reasoning_judge_errors has an entry `<family>:judge_failed...`.
Samples that were judged and simply found something inapplicable are left alone.

    .venv/bin/python tools/reset_failed_reasoning.py runs/<run>            # dry run
    .venv/bin/python tools/reset_failed_reasoning.py runs/<run> --apply
"""

from __future__ import annotations

import collections
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from abductionbench.core.reasoning_judge import REASONING_METRIC_COLUMNS  # noqa: E402

NEW = ("gemma-4-e4b-local", "gemma-4-e2b-local", "qwen3-5-2b-local", "qwen3-5-4b-local",
       "qwen3-5-27b-openrouter")
REASONING_DETAILS = ("reasoning_metrics_status", "reasoning_judge_errors",
                     "reasoning_metrics_inapplicable", "reasoning_lists", "reasoning_source",
                     "reasoning_judge_model")
RECENT_S = 120


def failed_families(record: dict) -> list[str]:
    errors = (record.get("details") or {}).get("reasoning_judge_errors") or []
    return sorted({e.split(":")[0] for e in errors if ":judge_failed" in e})


def enabled_datasets() -> set[str]:
    """Datasets the judging still covers: a disabled dataset is never re-judged,
    so resetting its samples would only delete results nothing will replace."""
    os.environ.setdefault("ABENCH_API_KEY", "unused")      # only the dataset list is read
    os.environ.setdefault("OPENROUTER_API_KEY", "unused")
    from abductionbench.core.config import load_run_config
    config = load_run_config(str(Path(__file__).resolve().parents[1] / "configs/runs/judge_shared_queue.yaml"))
    return {d.id for d in config.enabled_datasets()}


def main() -> int:
    run_dir = Path(sys.argv[1]).resolve()
    apply = "--apply" in sys.argv[2:]
    now = time.time()
    enabled = enabled_datasets()
    counts, fams, recent = collections.Counter(), collections.Counter(), []
    plan: list[tuple[Path, list[dict]]] = []
    for model in NEW:
        for path in sorted(run_dir.glob(f"datasets/*/{model}/cot*/records.jsonl")):
            if path.parent.parent.parent.name not in enabled:
                continue
            latest: dict[tuple, dict] = {}
            for line in path.open():
                if line.strip():
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    latest[(row.get("sample_id"), row.get("prompt_fingerprint"))] = row
            resets = []
            for row in latest.values():
                failed = failed_families(row)
                if not failed:
                    continue
                clean = dict(row)
                clean["metrics"] = {k: v for k, v in (row.get("metrics") or {}).items()
                                    if k not in REASONING_METRIC_COLUMNS}
                clean["details"] = {k: v for k, v in (row.get("details") or {}).items()
                                    if k not in REASONING_DETAILS}
                resets.append(clean)
                counts[model] += 1
                fams.update(failed)
            if resets:
                if now - path.stat().st_mtime < RECENT_S:
                    recent.append(path)
                else:
                    plan.append((path, resets))
    print(f"{'APPLYING' if apply else 'DRY RUN (add --apply)'}: {run_dir.name} "
          f"({len(enabled)} enabled datasets; disabled ones are left alone)")
    print(f"  samples to queue again: {sum(counts.values())} {dict(counts)}")
    print(f"  failed families among them: {dict(fams.most_common())}")
    for path in recent:
        print(f"  SKIPPED {path.relative_to(run_dir)}: written in the last {RECENT_S}s -- is a run live?")
    if not apply:
        return 0
    for path, resets in plan:
        with path.open("a") as handle:
            for row in resets:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  done: {sum(len(r) for _, r in plan)} record(s) queued again in {len(plan)} file(s)")
    return 1 if recent else 0


if __name__ == "__main__":
    sys.exit(main())
