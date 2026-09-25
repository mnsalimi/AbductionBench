"""Remove every reasoning-metric result and log from a run folder, locally.

For a clean re-judge: afterwards nothing of the reasoning judge is left for
a resume to reuse, so every sample is judged afresh by the current prompts
and code. The answers, their scores, the answer judge and its cache are not
touched.

    python tools/clear_reasoning_metrics.py runs/<run>            # dry run: counts only
    python tools/clear_reasoning_metrics.py runs/<run> --apply    # remove
    python tools/clear_reasoning_metrics.py runs/<run> --apply --keep-cache --exclude hypobench

--exclude <id,...> leaves those datasets' task folders exactly as they are --
for a dataset taken out of the run, whose metrics will not be re-judged.

--keep-cache leaves reasoning_judge_cache/ in place: the metrics are cleared,
but a re-judge under the same judge id then reads every verdict it already
bought from the cache instead of asking again. Only what changed -- a new
template version, a different input -- is asked.

Removed:
  * run level:  reasoning_metrics.jsonl, reasoning_judge_cache/
  * per task:   reasoning_judge_calls.jsonl (the judge-call audit);
                in records.jsonl every reasoning-judge metric and the details
                reasoning_metrics_status / reasoning_judge_errors /
                reasoning_metrics_inapplicable / reasoning_lists / reasoning_source;
                in metrics.json the aggregates of those metrics (and their
                self_consistency_ and best_of_n_ counterparts)

"Reasoning-judge metric" means a name the reasoning judge produces: its column
list (reasoning_judge.REASONING_METRIC_COLUMNS) plus every name it wrote into
this run's reasoning_metrics.jsonl. NOT every name starting "reasoning_":
medcasereasoning scores its own answers as reasoning_recall and
reasoning_overlap, and those are results, not reasoning metrics.

A records.jsonl with nothing to remove is not rewritten -- which is what makes
this safe beside a generation pass that is appending to its own files (a
judge-less pass writes no reasoning metrics, so its files are skipped). A file
changed in the last 2 minutes is skipped too, and reported.

Before anything is changed, everything it removes is archived once to
runs/_superseded/<run>__reasoning_metrics_cleared.tar.gz. Delete that file
yourself if you want nothing kept.
"""

from __future__ import annotations

import json
import sys
import tarfile
import time
from pathlib import Path

DETAIL_KEYS = (
    "reasoning_metrics_status",
    "reasoning_judge_errors",
    "reasoning_metrics_inapplicable",
    "reasoning_lists",
    "reasoning_source",
)
RECENT_S = 120


JUDGE_METRICS: set[str] = set()
#: How the engine names a task-level aggregate of a per-sample metric.
AGGREGATE_PREFIXES = ("self_consistency_", "best_of_n_")


def _judge_metric_names(run_dir: Path) -> set[str]:
    from abductionbench.core.reasoning_judge import REASONING_METRIC_COLUMNS

    names = set(REASONING_METRIC_COLUMNS)
    log = run_dir / "reasoning_metrics.jsonl"
    if log.exists():
        for line in log.read_text().splitlines():
            if line.strip():
                names.update((json.loads(line).get("metrics") or {}).keys())
    return names


def _strip_record(row: dict) -> bool:
    changed = False
    metrics = row.get("metrics")
    if isinstance(metrics, dict):
        for key in [k for k in metrics if k in JUDGE_METRICS]:
            del metrics[key]
            changed = True
    details = row.get("details")
    if isinstance(details, dict):
        for key in DETAIL_KEYS:
            if key in details:
                del details[key]
                changed = True
    return changed


def main(run_dir: Path, apply: bool, keep_cache: bool = False, exclude: frozenset = frozenset()) -> int:
    run_dir = run_dir.resolve()
    JUDGE_METRICS.update(_judge_metric_names(run_dir))
    assert not {"reasoning_recall", "reasoning_overlap"} & JUDGE_METRICS, "dataset metrics in the list"
    now = time.time()
    to_archive: list[Path] = []
    rewrites: list[tuple[Path, list[str]]] = []
    metrics_rewrites: list[tuple[Path, dict]] = []
    deletes: list[Path] = []
    skipped_recent: list[Path] = []
    stripped_lines = stripped_aggregates = 0

    for name in ("reasoning_metrics.jsonl",):
        path = run_dir / name
        if path.exists():
            deletes.append(path)
    cache = run_dir / "reasoning_judge_cache"
    if cache.exists() and not keep_cache:
        deletes.extend(sorted(p for p in cache.rglob("*") if p.is_file()))

    for task_dir in sorted(run_dir.glob("datasets/*/*/*")):
        if task_dir.parent.parent.name in exclude:
            continue
        audit = task_dir / "reasoning_judge_calls.jsonl"
        if audit.exists():
            deletes.append(audit)
        records = task_dir / "records.jsonl"
        if records.exists():
            lines = records.read_text().splitlines()
            out, changed = [], 0
            for line in lines:
                if not line.strip():
                    continue
                row = json.loads(line)
                if _strip_record(row):
                    changed += 1
                out.append(json.dumps(row, ensure_ascii=False))
            if changed:
                if now - records.stat().st_mtime < RECENT_S:
                    skipped_recent.append(records)
                else:
                    rewrites.append((records, out))
                    stripped_lines += changed
        metrics = task_dir / "metrics.json"
        if metrics.exists():
            blob = json.loads(metrics.read_text())
            inner = blob.get("metrics") if isinstance(blob.get("metrics"), dict) else {}
            gone = [
                k for k in inner
                if k in JUDGE_METRICS
                or any(k.removeprefix(p) in JUDGE_METRICS for p in AGGREGATE_PREFIXES)
            ]
            if gone:
                for key in gone:
                    del inner[key]
                metrics_rewrites.append((metrics, blob))
                stripped_aggregates += len(gone)

    to_archive = deletes + [p for p, _ in rewrites] + [p for p, _ in metrics_rewrites]
    size = sum(p.stat().st_size for p in deletes)
    print(f"{'APPLYING' if apply else 'DRY RUN -- nothing changes (add --apply)'}: {run_dir.name}")
    print(f"  delete   {len(deletes)} file(s), {size / 1e6:.0f} MB: reasoning_metrics.jsonl, "
          f"{'(cache KEPT)' if keep_cache else 'the reasoning-judge cache'}, {sum(p.name == 'reasoning_judge_calls.jsonl' for p in deletes)} "
          f"task audit log(s)")
    print(f"  rewrite  {len(rewrites)} records.jsonl ({stripped_lines} line(s) lose their reasoning "
          f"metrics and details)")
    print(f"  rewrite  {len(metrics_rewrites)} metrics.json ({stripped_aggregates} reasoning aggregate(s) removed)")
    for path in skipped_recent:
        print(f"  SKIPPED  {path.relative_to(run_dir)}: written in the last {RECENT_S}s -- run again later")
    if not apply:
        return 0

    suffix = "__keep-cache" if keep_cache else ""
    archive = run_dir.parent / "_superseded" / f"{run_dir.name}__reasoning_metrics_cleared{suffix}.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        with tarfile.open(archive, "w:gz") as tar:
            for path in to_archive:
                tar.add(path, arcname=str(path.relative_to(run_dir)))
        print(f"  archived what is removed -> {archive}")
    for path, out in rewrites:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text("\n".join(out) + "\n")
        tmp.replace(path)
    for path, blob in metrics_rewrites:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(blob, indent=2, ensure_ascii=False))
        tmp.replace(path)
    for path in deletes:
        path.unlink(missing_ok=True)
    if cache.exists() and not any(cache.rglob("*")):
        cache.rmdir()
    print("  done")
    return 1 if skipped_recent else 0


if __name__ == "__main__":
    argv = sys.argv[1:]
    exclude: frozenset = frozenset()
    if "--exclude" in argv:
        at = argv.index("--exclude")
        exclude = frozenset(x for x in argv[at + 1].split(",") if x)
        argv = argv[:at] + argv[at + 2:]
    args = [a for a in argv if a not in ("--apply", "--keep-cache")]
    if len(args) != 1:
        print(__doc__)
        sys.exit(2)
    if exclude:
        print(f"  excluded (left untouched): {', '.join(sorted(exclude))}")
    sys.exit(main(Path(args[0]), "--apply" in argv, "--keep-cache" in argv, exclude))
