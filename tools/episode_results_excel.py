"""One workbook for an interactive + sequential run: metrics, samples, turns.

    PYTHONPATH=src .venv/bin/python tools/episode_results_excel.py runs/<run> [out.xlsx]
    PYTHONPATH=src .venv/bin/python tools/episode_results_excel.py runs/<a> runs/<b> ... --out all.xlsx

Several runs go into ONE workbook (every row says which run and model it
came from).

Sheets:
  summary      one row per task: the episode metrics (core/episode_metrics.py)
               and the bookkeeping columns
  <dataset>    one row per record: its metrics, prediction, gold, turns
  turns        every turn of every episode: request, reply and, for the
               sequential dataset, that turn's prediction and verdicts
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from abductionbench.core.checkpoint import dedupe_records, load_records
from abductionbench.core.episode_metrics import EPISODE_METRICS
from abductionbench.core.reporting import build_turns_frame

BOOKKEEPING = ("n_planned", "n_scored", "n_error", "coverage", "parse_failure_rate",
               "truncation_rate", "empty_response_rate", "completion_tokens_mean")


def _clip(value, n=3000):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if text is None or len(text) <= n else text[:n] + " ..."


def main() -> int:
    args = sys.argv[1:]
    out_arg = None
    if "--out" in args:
        i = args.index("--out")
        out_arg = Path(args[i + 1])
        args = args[:i] + args[i + 2:]
    run_dirs = [Path(a).resolve() for a in args]
    if out_arg is None and len(run_dirs) == 2 and run_dirs[1].suffix == ".xlsx":
        out_arg, run_dirs = run_dirs[1], run_dirs[:1]        # the old two-argument form
    out = out_arg or run_dirs[0] / "reports" / "episode_results.xlsx"
    summary, per_dataset, task_dirs = [], {}, []
    metrics_paths = [p for run_dir in run_dirs
                     for p in sorted(run_dir.glob("datasets/*/*/*/metrics.json"))]
    for metrics_path in metrics_paths:
        task_dir = metrics_path.parent
        task_dirs.append(task_dir)
        blob = json.loads(metrics_path.read_text())
        identity = blob.get("task") or {}
        mode = identity.get("data_delivery_mode", "")
        metrics = blob.get("metrics") or {}
        row = {"dataset": task_dir.parents[1].name, "task": task_dir.name,
               "delivery": mode, "model": task_dir.parent.name, "run": task_dir.parents[3].name}
        for name in EPISODE_METRICS.get(mode, ()):
            row[name] = metrics.get(name)
        for name in BOOKKEEPING:
            row[name] = metrics.get(name)
        summary.append(row)

        for record in dedupe_records(load_records(task_dir / "records.jsonl")):
            details = record.get("details") or {}
            ref = record.get("reference")
            # Most adapters keep the answer under `gold`; a structured one
            # (cloud_opsbench: root_cause, fault_object, fault_taxonomy) is
            # the whole reference.
            gold = details.get("gold")
            if gold is None:
                gold = ref.get("gold", ref) if isinstance(ref, dict) else ref
            r = {"model": task_dir.parent.name, "task": task_dir.name,
                 "sample_id": record.get("sample_id"),
                 "status": record.get("status"),
                 **{f"metric.{k}": v for k, v in (record.get("metrics") or {}).items()},
                 "prediction": _clip(record.get("prediction")),
                 "gold": _clip(gold),
                 "turns": (record.get("response") or {}).get("usage", {}).get("turns")}
            for key in ("turn_predictions", "turn_correct", "turn_matches_final",
                        "interaction_step_relevance_per_step", "interaction_step_relevance"):
                if key in details:
                    r[key] = _clip(details[key])
            r["other_adapter_metrics"] = _clip(details.get("adapter_metrics"))
            r["final_reply"] = _clip((record.get("response") or {}).get("content"))
            per_dataset.setdefault(task_dir.parents[1].name, []).append(r)

    turns = build_turns_frame(task_dirs, clip=3000)
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        pd.DataFrame(summary).to_excel(writer, sheet_name="summary", index=False)
        for dataset, rows in sorted(per_dataset.items()):
            pd.DataFrame(rows).to_excel(writer, sheet_name=dataset[:31], index=False)
        turns.to_excel(writer, sheet_name="turns", index=False)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
