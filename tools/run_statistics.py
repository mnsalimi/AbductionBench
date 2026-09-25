"""Paper-ready counts for a run: what was planned and what came back.

READ-ONLY on the run. For every dataset x model x task (template) it reads the
deduplicated records and counts records (distinct questions), samples
(records x repeats), and how each sample ended: answered, answered but cut off
at the token limit, empty, or error (never answered). It also counts the
answer-judge verdicts and the reasoning-judge coverage of cot samples.

    .venv/bin/python tools/run_statistics.py runs/<run> [--exclude hypobench]

Writes runs/<run>/analysis/run_statistics/{summary.md, per_dataset.csv,
per_model_mode.csv, per_task.csv}.
"""

from __future__ import annotations

import argparse
import collections
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from abductionbench.core.checkpoint import dedupe_records, load_records  # noqa: E402

REPEAT = re.compile(r"#r\d+$")
MODEL_ORDER = ["gpt-5.6-luna-openrouter", "gemini-3.8-flash-openrouter", "gemma-4-31b-it-openrouter",
               "qwen3-5-27b-openrouter", "gemma-4-e4b-local", "gemma-4-e2b-local",
               "qwen3-5-4b-local", "qwen3-5-2b-local"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--exclude", default="", help="comma-separated dataset ids to leave out")
    args = ap.parse_args()
    run_dir = args.run_dir.resolve()
    exclude = {x for x in args.exclude.split(",") if x}
    out = run_dir / "analysis" / "run_statistics"
    out.mkdir(parents=True, exist_ok=True)

    tasks = []
    for path in sorted(run_dir.glob("datasets/*/*/*/records.jsonl")):
        template_dir, model_dir, dataset_dir = path.parent, path.parent.parent, path.parent.parent.parent
        dataset, model, template = dataset_dir.name, model_dir.name, template_dir.name.split("@")[0]
        if dataset in exclude:
            continue
        recs = dedupe_records(load_records(path))
        if not recs:
            continue
        mode = str(recs[0].get("prompt_mode") or template.split("_")[0])
        c = collections.Counter()
        questions = set()
        for r in recs:
            questions.add(REPEAT.sub("", str(r.get("sample_id"))))
            status = r.get("status")
            c["samples"] += 1
            c[f"status_{status}"] += 1
            details = r.get("details") or {}
            if status in ("ok", "empty", "truncated") and details.get("no_answer"):
                c["no_answer"] += 1
            if any(k.startswith(("judge_", "proxy_judgement")) or k in ("judge_label",) for k in details):
                c["answer_judged"] += 1
            if mode == "cot" and details.get("reasoning_metrics_status"):
                c["reasoning_judged"] += 1
                if (r.get("metrics") or {}).get("reasoning_total_steps") is not None:
                    c["reasoning_with_steps"] += 1
        tasks.append({"dataset": dataset, "model": model, "template": template, "mode": mode,
                      "records": len(questions), **c})

    keys = ["samples", "status_ok", "status_truncated", "status_empty", "status_error", "no_answer",
            "answer_judged", "reasoning_judged", "reasoning_with_steps"]
    fields = ["dataset", "model", "template", "mode", "records", *keys]
    with (out / "per_task.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for t in tasks:
            w.writerow({k: t.get(k, 0) for k in fields})

    datasets = sorted({t["dataset"] for t in tasks})
    models = [m for m in MODEL_ORDER if any(t["model"] == m for t in tasks)] + sorted(
        {t["model"] for t in tasks} - set(MODEL_ORDER))

    per_ds = []
    for d in datasets:
        dt = [t for t in tasks if t["dataset"] == d]
        io_t = sorted({t["template"] for t in dt if t["mode"] == "io"})
        cot_t = sorted({t["template"] for t in dt if t["mode"] == "cot"})
        records = max(t["records"] for t in dt)
        spt = max(t["samples"] for t in dt)
        per_ds.append({"dataset": d, "records": records, "repeats": round(spt / records) if records else 0,
                       "io_templates": len(io_t), "cot_templates": len(cot_t),
                       "samples_per_task": spt,
                       "samples_per_model": spt * (len(io_t) + len(cot_t)),
                       "models": len({t["model"] for t in dt}),
                       "samples_stored": sum(t["samples"] for t in dt)})
    with (out / "per_dataset.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_ds[0].keys()))
        w.writeheader(); w.writerows(per_ds)

    per_mm = []
    for m in models:
        for mode in ("io", "cot"):
            mt = [t for t in tasks if t["model"] == m and t["mode"] == mode]
            if not mt:
                continue
            row = {"model": m, "mode": mode, "tasks": len(mt), "records": sum(t["records"] for t in mt)}
            row.update({k: sum(t.get(k, 0) for t in mt) for k in keys})
            per_mm.append(row)
    with (out / "per_model_mode.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_mm[0].keys()))
        w.writeheader(); w.writerows(per_mm)

    n_models = len(models)
    planned_tasks_per_model = sum(d["io_templates"] + d["cot_templates"] for d in per_ds)
    planned_samples = sum(d["samples_per_model"] for d in per_ds) * n_models
    tot = collections.Counter()
    for t in tasks:
        for k in keys:
            tot[(t["mode"], k)] += t.get(k, 0)
    lines = [f"# Run statistics -- {run_dir.name}", "",
             f"Datasets: **{len(datasets)}**{' (excluded: ' + ', '.join(sorted(exclude)) + ')' if exclude else ''}; "
             f"models: **{n_models}**; records per dataset: **{sorted({d['records'] for d in per_ds})}**; "
             f"repeats per record: **{sorted({d['repeats'] for d in per_ds})}**.",
             f"Distinct records (questions) across datasets: **{sum(d['records'] for d in per_ds):,}**; "
             f"dataset x model pairs: **{len(datasets) * n_models}**; "
             f"distinct records x models: **{sum(d['records'] for d in per_ds) * n_models:,}**.",
             f"Tasks (dataset x template) per model: **{planned_tasks_per_model}** "
             f"({sum(d['io_templates'] for d in per_ds)} io + {sum(d['cot_templates'] for d in per_ds)} cot); "
             f"tasks in all: **{planned_tasks_per_model * n_models}**.",
             f"Samples planned (every task x 150): **{planned_samples:,}** "
             f"({sum(d['samples_per_task'] * d['io_templates'] for d in per_ds) * n_models:,} io + "
             f"{sum(d['samples_per_task'] * d['cot_templates'] for d in per_ds) * n_models:,} cot); "
             f"stored: **{sum(t['samples'] for t in tasks):,}**.", "",
             "## Outcomes", "",
             "| mode | samples | answered | cut off at the token limit | empty | error (never answered) | no answer | answer-judged | reasoning-judged | ...with a step list |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for mode in ("io", "cot"):
        lines.append("| " + mode + " | " + " | ".join(f"{tot[(mode, k)]:,}" for k in keys) + " |")
    lines += ["", "## Per model and mode", "",
              "| model | mode | tasks | records | samples | answered | cut off | empty | error | no answer | answer-judged | reasoning-judged | with steps |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in per_mm:
        lines.append(f"| {r['model']} | {r['mode']} | {r['tasks']} | {r['records']:,} | " +
                     " | ".join(f"{r[k]:,}" for k in keys) + " |")
    lines += ["", "## Per dataset", "",
              "| dataset | records | repeats | io tasks | cot tasks | samples / task | samples / model | models | samples stored |",
              "|---|---|---|---|---|---|---|---|---|"]
    for d in per_ds:
        lines.append(f"| {d['dataset']} | {d['records']} | {d['repeats']} | {d['io_templates']} | {d['cot_templates']} | "
                     f"{d['samples_per_task']} | {d['samples_per_model']:,} | {d['models']} | {d['samples_stored']:,} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
