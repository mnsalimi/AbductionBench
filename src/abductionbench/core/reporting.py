"""Reporting: the unified result grid, CSVs, and run documentation.

Outputs written under a run directory:

``reports/abductionbench_results.xlsx``
    * ``Summary``      -- dataset x model matrix of each dataset's primary metric.
    * ``Metrics``      -- long form: one row per (dataset, model, template, metric).
    * ``Tasks``        -- per-task counts, diagnostics, endpoint, duration, failures.
    * ``Datasets``     -- each adapter's self-documentation (split, subset, seed,
                          decisions, caveats, statistics), including skipped ones.
    * ``Skipped``      -- datasets that were not evaluated, with the reason.
    * ``Models``       -- endpoint/verification details per model.
    * ``Samples:<ds>`` -- per-sample grid per dataset (optional, capped).
``reports/metrics_long.csv``, ``reports/summary.csv``
    The same tables as CSV, for scripting.
``datasets/<dataset>/<model>/<template>/run_documentation.md``
    Per-task markdown: what was run, on what, with which prompt, and what the
    adapter decided.
``RUN_REPORT.md``
    Top-level human summary of the whole run.
"""

from __future__ import annotations

import logging
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .checkpoint import load_records
from .engine import RunResult, TaskResult
from .types import AdapterDocumentation

logger = logging.getLogger(__name__)

__all__ = ["write_reports", "build_summary_frame", "build_metrics_frame"]


def _fmt(value: Any, digits: int = 4) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    return value


def build_metrics_frame(result: RunResult) -> pd.DataFrame:
    """Long-form metric table: one row per (task, metric)."""
    rows: list[dict[str, Any]] = []
    for task in result.tasks:
        for metric, value in sorted(task.metrics.items()):
            rows.append(
                {
                    "run_id": result.run_id,
                    "dataset_id": task.identity.dataset_id,
                    "model_id": task.identity.model_id,
                    "template": f"{task.identity.template_id}@{task.identity.template_version}",
                    "metric": metric,
                    "value": _fmt(value),
                    "is_primary": metric == task.primary_metric,
                    "n_planned": task.n_planned,
                    "n_scored": task.n_scored,
                    "n_error": task.n_error,
                    "n_skipped": task.n_skipped,
                }
            )
    return pd.DataFrame(rows)


def build_summary_frame(result: RunResult) -> pd.DataFrame:
    """Dataset x model matrix of primary metrics (the headline table)."""
    rows: list[dict[str, Any]] = []
    for task in result.tasks:
        primary = task.primary_metric
        rows.append(
            {
                "dataset_id": task.identity.dataset_id,
                "model_id": task.identity.model_id,
                "template": f"{task.identity.template_id}@{task.identity.template_version}",
                "primary_metric": primary,
                "value": _fmt(task.metrics.get(primary)),
                "value_strict": _fmt(task.metrics.get(f"{primary}_strict")),
                "coverage": _fmt(task.metrics.get("coverage")),
                "n_scored": task.n_scored,
                "n_planned": task.n_planned,
                "failure": task.failure or "",
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["dataset_id", "model_id", "template"]).reset_index(drop=True)


def _build_tasks_frame(result: RunResult) -> pd.DataFrame:
    rows = []
    for task in result.tasks:
        checkpoint = task.checkpoint.to_dict() if task.checkpoint else {}
        rows.append(
            {
                "dataset_id": task.identity.dataset_id,
                "model_id": task.identity.model_id,
                "template": f"{task.identity.template_id}@{task.identity.template_version}",
                "primary_metric": task.primary_metric,
                "primary_value": _fmt(task.metrics.get(task.primary_metric)),
                "n_planned": task.n_planned,
                "n_scored": task.n_scored,
                "n_error": task.n_error,
                "n_skipped": task.n_skipped,
                "n_reused": task.n_reused,
                "batch_mode": task.diagnostics.get("batch_mode"),
                "group_size": task.diagnostics.get("group_size"),
                "batches_submitted": checkpoint.get("batches_submitted"),
                "batches_failed": checkpoint.get("batches_failed"),
                "bisections": checkpoint.get("bisections"),
                "prompt_tokens_total": checkpoint.get("prompt_tokens_total"),
                "completion_tokens_total": checkpoint.get("completion_tokens_total"),
                "duration_s": round(task.duration_s, 1),
                "endpoint": task.diagnostics.get("endpoint"),
                "failure": task.failure or "",
                "output_dir": str(task.output_dir),
            }
        )
    return pd.DataFrame(rows)


def _build_datasets_frame(result: RunResult) -> pd.DataFrame:
    """Adapter self-documentation, one row per dataset."""
    seen: dict[str, AdapterDocumentation] = {}
    for task in result.tasks:
        if task.documentation and task.identity.dataset_id not in seen:
            seen[task.identity.dataset_id] = task.documentation
    rows = []
    for dataset_id, doc in sorted(seen.items()):
        rows.append(
            {
                "dataset_id": dataset_id,
                "name": doc.name,
                "domain": doc.domain,
                "processing_mode": doc.processing_mode,
                "source_url": doc.source_url,
                "split_used": doc.split_used,
                "abductive_subset": doc.abductive_subset,
                "sampling_procedure": doc.sampling_procedure,
                "primary_metric": doc.primary_metric,
                "metrics": "; ".join(f"{k}: {v}" for k, v in doc.metrics_description.items()),
                "decisions": " | ".join(doc.decisions),
                "caveats": " | ".join(doc.caveats),
                "statistics": "; ".join(f"{k}={v}" for k, v in doc.statistics.items()),
            }
        )
    return pd.DataFrame(rows)


def _build_samples_frame(task_dirs: list[Path], *, clip: int, limit: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for directory in task_dirs:
        for record in load_records(directory / "records.jsonl"):
            response = record.get("response") or {}
            row: dict[str, Any] = {
                "dataset_id": record.get("dataset_id"),
                "model_id": record.get("model_id"),
                "template": f"{record.get('template_id')}@{record.get('template_version')}",
                "sample_id": record.get("sample_id"),
                "group_id": record.get("group_id"),
                "task_kind": record.get("task_kind"),
                "status": record.get("status"),
                "parse_ok": record.get("parse_ok"),
                "input_tokens_est": record.get("input_tokens_est"),
                "max_tokens": (record.get("sampling") or {}).get("max_tokens"),
                "finish_reason": response.get("finish_reason"),
                "prediction": _clip(record.get("prediction"), clip),
                "reference": _clip(record.get("reference"), clip),
                "response": _clip(response.get("content"), clip),
                "error": _clip(response.get("error"), 300),
                "batch_id": response.get("batch_id"),
                "batch_size": response.get("batch_size"),
                "latency_s": response.get("latency_s"),
            }
            for metric, value in (record.get("metrics") or {}).items():
                row[f"metric.{metric}"] = _fmt(value)
            rows.append(row)
            if len(rows) >= limit:
                break
        if len(rows) >= limit:
            break
    return pd.DataFrame(rows)


def _clip(value: Any, limit: int) -> Any:
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)
    if limit and len(text) > limit:
        return text[:limit] + "..."
    return text


def _safe_sheet_name(name: str, used: set[str]) -> str:
    """Excel sheet names: <=31 chars, no ``[]:*?/\\``, unique."""
    cleaned = "".join("_" if ch in "[]:*?/\\" else ch for ch in name)[:31]
    candidate = cleaned or "sheet"
    suffix = 2
    while candidate in used:
        tail = f"_{suffix}"
        candidate = cleaned[: 31 - len(tail)] + tail
        suffix += 1
    used.add(candidate)
    return candidate


def write_reports(result: RunResult) -> dict[str, Path]:
    """Write every report artifact for a finished run.  Returns paths by name."""
    reporting = result.config.engine.reporting
    reports_dir = result.run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    summary = build_summary_frame(result)
    metrics = build_metrics_frame(result)
    tasks = _build_tasks_frame(result)
    datasets = _build_datasets_frame(result)
    skipped = pd.DataFrame(result.skipped_datasets or [], columns=["dataset_id", "reason"])
    models = pd.DataFrame(
        [
            {
                "model_id": report.get("model_id"),
                "model_name": report.get("model_name"),
                "base_url": report.get("base_url"),
                "batch_url": report.get("batch_url"),
                "model_available": report.get("model_available"),
                "batch_ok": report.get("batch_ok"),
                "batch_probe_latency_s": report.get("batch_probe_latency_s"),
                "error": report.get("batch_error") or report.get("discovery_error") or "",
            }
            for report in result.endpoint_reports
        ]
    )

    if reporting.write_csv:
        if not summary.empty:
            path = reports_dir / "summary.csv"
            summary.to_csv(path, index=False)
            written["summary_csv"] = path
        if not metrics.empty:
            path = reports_dir / "metrics_long.csv"
            metrics.to_csv(path, index=False)
            written["metrics_csv"] = path

    excel_path = reports_dir / reporting.excel_filename
    try:
        with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
            used: set[str] = set()
            _write_sheet(writer, _pivot(summary), _safe_sheet_name("Summary", used), index=True)
            _write_sheet(writer, summary, _safe_sheet_name("Summary_Long", used))
            _write_sheet(writer, metrics, _safe_sheet_name("Metrics", used))
            _write_sheet(writer, tasks, _safe_sheet_name("Tasks", used))
            _write_sheet(writer, datasets, _safe_sheet_name("Datasets", used))
            _write_sheet(writer, skipped, _safe_sheet_name("Skipped", used))
            _write_sheet(writer, models, _safe_sheet_name("Models", used))
            if reporting.include_sample_sheets:
                by_dataset: dict[str, list[Path]] = {}
                for task in result.tasks:
                    by_dataset.setdefault(task.identity.dataset_id, []).append(task.output_dir)
                for dataset_id, directories in sorted(by_dataset.items()):
                    frame = _build_samples_frame(
                        directories,
                        clip=reporting.response_clip_chars,
                        limit=reporting.max_sample_rows_per_sheet,
                    )
                    if frame.empty:
                        continue
                    _write_sheet(
                        writer, frame, _safe_sheet_name(f"S_{dataset_id}", used)
                    )
        written["excel"] = excel_path
    except Exception as exc:  # noqa: BLE001 - never lose a run over a report
        logger.exception("failed to write Excel workbook: %s", exc)

    if reporting.write_run_documentation:
        for task in result.tasks:
            try:
                doc_path = _write_task_documentation(result, task)
                written[f"doc:{task.identity.slug}"] = doc_path
            except Exception as exc:  # noqa: BLE001
                logger.warning("could not write documentation for %s: %s", task.identity.slug, exc)

    report_path = result.run_dir / "RUN_REPORT.md"
    report_path.write_text(_render_run_report(result, summary), encoding="utf-8")
    written["run_report"] = report_path
    logger.info("reports written to %s", reports_dir)
    return written


def _pivot(summary: pd.DataFrame) -> pd.DataFrame:
    """Dataset x model matrix of primary-metric values."""
    if summary.empty:
        return summary
    frame = summary.copy()
    frame["row"] = frame["dataset_id"] + " [" + frame["primary_metric"].fillna("") + "]"
    frame["column"] = frame["model_id"] + " (" + frame["template"] + ")"
    return frame.pivot_table(
        index="row", columns="column", values="value", aggfunc="first"
    ).sort_index()


def _write_sheet(writer: Any, frame: pd.DataFrame, name: str, *, index: bool = False) -> None:
    if frame is None or frame.empty:
        pd.DataFrame({"note": ["no data"]}).to_excel(writer, sheet_name=name, index=False)
        return
    frame.to_excel(writer, sheet_name=name, index=index)
    worksheet = writer.sheets[name]
    for position, column in enumerate(frame.columns, start=1 if index else 0):
        # str() per cell rather than astype(str): pandas >= 2.1 leaves missing
        # values as float NaN under astype(str), which has no len().
        widths = [len(str(column))]
        series = frame.iloc[:, position - 1 if index else position].head(200)
        widths.extend(len(str(value)) for value in series)
        worksheet.set_column(position, position, min(60, max(10, max(widths) + 2)))


def _write_task_documentation(result: RunResult, task: TaskResult) -> Path:
    """Per-task markdown documentation."""
    doc = task.documentation
    identity = task.identity
    lines: list[str] = []
    lines.append(f"# Run documentation -- {identity.dataset_id} x {identity.model_id}")
    lines.append("")
    lines.append(f"* **Run id**: `{result.run_id}`")
    lines.append(f"* **Generated**: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    lines.append(f"* **Dataset**: `{identity.dataset_id}`")
    lines.append(f"* **Model**: `{identity.model_id}`")
    lines.append(
        f"* **Prompt template**: `{identity.template_id}@{identity.template_version}`"
        f" (variant `{task.diagnostics.get('variant', 'default')}`)"
    )
    lines.append(f"* **Endpoint**: `{task.diagnostics.get('endpoint')}`")
    lines.append(
        f"* **Batch mode**: {task.diagnostics.get('batch_mode')} "
        f"(group size {task.diagnostics.get('group_size')})"
    )
    lines.append(f"* **Duration**: {task.duration_s:.1f}s")
    lines.append(f"* **Output directory**: `{task.output_dir}`")
    lines.append("")

    lines.append("## Counts")
    lines.append("")
    lines.append("| planned | scored | errors | skipped | reused from checkpoint |")
    lines.append("|---|---|---|---|---|")
    lines.append(
        f"| {task.n_planned} | {task.n_scored} | {task.n_error} | {task.n_skipped} "
        f"| {task.n_reused} |"
    )
    lines.append("")

    lines.append("## Metrics")
    lines.append("")
    if task.metrics:
        lines.append("| metric | value |")
        lines.append("|---|---|")
        primary = task.primary_metric
        for key, value in sorted(task.metrics.items()):
            marker = " **(primary)**" if key == primary else ""
            lines.append(f"| `{key}`{marker} | {_fmt(value)} |")
    else:
        lines.append("_No metrics were produced._")
    lines.append("")

    if task.failure:
        lines.append("## Failure")
        lines.append("")
        lines.append(f"```\n{task.failure}\n```")
        lines.append("")

    if doc is not None:
        lines.append("## Dataset adapter documentation")
        lines.append("")
        lines.append(f"* **Name**: {doc.name}")
        lines.append(f"* **Domain**: {doc.domain}")
        lines.append(f"* **Source**: {doc.source_url}")
        lines.append(f"* **Processing mode**: {doc.processing_mode}")
        lines.append(f"* **Split used**: {doc.split_used}")
        lines.append(f"* **Abductive subset**: {doc.abductive_subset}")
        lines.append(f"* **Sampling procedure**: {doc.sampling_procedure}")
        lines.append(f"* **Primary metric**: `{doc.primary_metric}`")
        lines.append("")
        if doc.metrics_description:
            lines.append("### Metric definitions")
            lines.append("")
            for name, description in doc.metrics_description.items():
                lines.append(f"* `{name}` -- {description}")
            lines.append("")
        if doc.decisions:
            lines.append("### Decisions made by this adapter")
            lines.append("")
            for decision in doc.decisions:
                lines.append(f"* {decision}")
            lines.append("")
        if doc.caveats:
            lines.append("### Caveats")
            lines.append("")
            for caveat in doc.caveats:
                lines.append(f"* {caveat}")
            lines.append("")
        if doc.statistics:
            lines.append("### Statistics")
            lines.append("")
            for key, value in doc.statistics.items():
                lines.append(f"* `{key}`: {value}")
            lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    lines.append("* `records.jsonl` -- one JSON object per sample (prompt fingerprint, response,")
    lines.append("  per-sample metrics, reference, metadata).")
    lines.append("* `metrics.json` -- aggregated metrics and diagnostics for this task.")
    lines.append("* `checkpoint.json` -- resume state and batch statistics.")
    lines.append("* `raw/` -- raw request/response payloads per batch call (when enabled).")
    lines.append("")

    task.output_dir.mkdir(parents=True, exist_ok=True)
    path = task.output_dir / "run_documentation.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _render_run_report(result: RunResult, summary: pd.DataFrame) -> str:
    config = result.config
    lines: list[str] = []
    lines.append(f"# AbductionBench run report -- `{result.run_id}`")
    lines.append("")
    lines.append(f"* **Name**: {config.name}")
    if config.description:
        lines.append(f"* **Description**: {config.description}")
    lines.append(f"* **Seed**: {config.seed}")
    lines.append(f"* **Started**: {datetime.fromtimestamp(result.started_at, timezone.utc).isoformat(timespec='seconds')}")
    lines.append(f"* **Duration**: {result.duration_s / 60:.1f} min")
    lines.append(f"* **Models**: {', '.join(m.id for m in config.models)}")
    lines.append(f"* **Tasks**: {len(result.tasks)}")
    lines.append(f"* **Config sources**: {', '.join(config.source_files)}")
    lines.append("")

    lines.append("## Headline results")
    lines.append("")
    if summary.empty:
        lines.append("_No tasks produced results._")
    else:
        pivot = _pivot(summary)
        lines.append(pivot.to_markdown())
    lines.append("")

    failed = [task for task in result.tasks if task.failure]
    if failed:
        lines.append("## Failed tasks")
        lines.append("")
        for task in failed:
            lines.append(f"* `{task.identity.slug}`: {task.failure}")
        lines.append("")

    if result.skipped_datasets:
        lines.append("## Skipped datasets")
        lines.append("")
        for entry in result.skipped_datasets:
            reason = textwrap.shorten(entry.get("reason", ""), width=300, placeholder="...")
            lines.append(f"* `{entry.get('dataset_id')}`: {reason}")
        lines.append("")

    lines.append("## Reliability")
    lines.append("")
    lines.append("| task | coverage | errors | skipped | truncation | empty | bisections |")
    lines.append("|---|---|---|---|---|---|---|")
    for task in result.tasks:
        checkpoint = task.checkpoint.to_dict() if task.checkpoint else {}
        lines.append(
            f"| `{task.identity.slug}` | {_fmt(task.metrics.get('coverage'))} | {task.n_error} "
            f"| {task.n_skipped} | {_fmt(task.metrics.get('truncation_rate'))} "
            f"| {_fmt(task.metrics.get('empty_response_rate'))} "
            f"| {checkpoint.get('bisections', 0)} |"
        )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"* `reports/{config.engine.reporting.excel_filename}` -- unified result grid.")
    lines.append("* `reports/metrics_long.csv`, `reports/summary.csv` -- machine-readable tables.")
    lines.append("* `datasets/<dataset>/<model>/<template>/` -- per-task records, metrics,")
    lines.append("  checkpoint, raw payloads and run documentation.")
    lines.append("* `run_config.resolved.yaml` -- the fully resolved configuration (keys redacted).")
    lines.append("* `engine.log`, `engine.jsonl`, `events.jsonl` -- logs and structured telemetry.")
    lines.append("")
    return "\n".join(lines)
