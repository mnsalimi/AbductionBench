"""Reporting: the unified result grid, CSVs, and run documentation.

Outputs written under a run directory:

``reports/abductionbench_results.xlsx``
    * ``Summary``      -- dataset x model matrix of each dataset's primary metric.
    * ``Metrics``      -- long form: one row per (dataset, model, template, metric).
    * ``Tasks``        -- per-task counts, diagnostics, endpoint, duration, failures.
    * ``Datasets``     -- each adapter's self-documentation (split, subset, seed,
                          decisions, caveats, statistics), including skipped ones.
    * ``Skipped``      -- datasets that were not evaluated, with the reason.
    * ``Skipped_modes``    -- mode combinations a dataset declined, with the reason.
    * ``Introduced_modes`` -- hypothesis modes run beyond the dataset table, each
                          with the benchmark formulation that justifies it.
    * ``Models``       -- endpoint/verification details per model.
    * ``S_<ds>``       -- per-sample grid per dataset (optional, capped).
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

from .checkpoint import dedupe_records, load_records
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
                    "prompt_mode": task.identity.prompt_mode,
                    "selection_mode": task.identity.selection_mode,
                    "data_delivery_mode": task.identity.data_delivery_mode,
                    "task_kind": task.identity.task_kind,
                    "template_mode": task.identity.template_mode,
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
                "prompt_mode": task.identity.prompt_mode,
                "selection_mode": task.identity.selection_mode,
                "data_delivery_mode": task.identity.data_delivery_mode,
                "task_kind": task.identity.task_kind,
                "template_mode": task.identity.template_mode,
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
    return frame.sort_values(
        ["dataset_id", "model_id", "template_mode"]
    ).reset_index(drop=True)


def _build_tasks_frame(result: RunResult) -> pd.DataFrame:
    rows = []
    for task in result.tasks:
        checkpoint = task.checkpoint.to_dict() if task.checkpoint else {}
        rows.append(
            {
                "dataset_id": task.identity.dataset_id,
                "model_id": task.identity.model_id,
                "prompt_mode": task.identity.prompt_mode,
                "selection_mode": task.identity.selection_mode,
                "data_delivery_mode": task.identity.data_delivery_mode,
                "task_kind": task.identity.task_kind,
                "template_mode": task.identity.template_mode,
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


def _build_skipped_modes_frame(result: RunResult) -> pd.DataFrame:
    """Mode combinations a dataset declined, and why.

    A mode that was asked for and not run has to be visible: silence would look
    like the dataset simply had nothing to say in that mode.
    """
    return pd.DataFrame(
        [
            {
                "dataset_id": entry.get("dataset_id"),
                "mode": entry.get("mode"),
                "reason": entry.get("reason"),
            }
            for entry in result.skipped_modes
        ]
    )


def _build_introduced_modes_frame(result: RunResult) -> pd.DataFrame:
    """Hypothesis modes run beyond what the dataset table lists.

    Specification item 15 requires both facts on the record: that the mode was
    introduced, and the benchmark's own formulation that makes it a separate
    task rather than a re-reading of the same one.
    """
    return pd.DataFrame(
        [
            {
                "dataset_id": entry.get("dataset_id"),
                "hypothesis_mode": entry.get("hypothesis_mode"),
                "mode": entry.get("mode"),
                "dataset_table_says": entry.get("table_says"),
                "justification": entry.get("justification"),
            }
            for entry in result.introduced_modes
        ]
    )


def _build_samples_frame(task_dirs: list[Path], *, clip: int, limit: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for directory in task_dirs:
        for record in dedupe_records(load_records(directory / "records.jsonl")):
            response = record.get("response") or {}
            row: dict[str, Any] = {
                "dataset_id": record.get("dataset_id"),
                "model_id": record.get("model_id"),
                # The three mode columns, then the identity they compose.
                "prompt_mode": record.get("prompt_mode"),
                "selection_mode": record.get("selection_mode"),
                "data_delivery_mode": record.get("data_delivery_mode"),
                "task_kind": record.get("task_kind"),
                "template_mode": record.get("template_mode"),
                "sample_id": record.get("sample_id"),
                # The record this request came from, and which of its repeats
                # this is: with repeats > 1 the sample_id is unique per call,
                # so these are what group a record's observations together.
                "record_id": (record.get("metadata") or {}).get("repeat_of")
                or record.get("group_id")
                or record.get("sample_id"),
                "repeat_index": (record.get("metadata") or {}).get("repeat_index"),
                "group_id": record.get("group_id"),
                "reduced": bool((record.get("metadata") or {}).get("reduced")),
                "status": record.get("status"),
                "parse_ok": record.get("parse_ok"),
                "input_tokens_est": record.get("input_tokens_est"),
                "max_tokens": (record.get("sampling") or {}).get("max_tokens"),
                "finish_reason": response.get("finish_reason"),
                "prediction": _clip(record.get("prediction"), clip),
                "reference": _clip(record.get("reference"), clip),
                "response": _clip(response.get("content"), clip),
                "reasoning": _clip(response.get("reasoning"), clip),
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


#: Excel's hard limit on the number of characters in one cell.  Anything longer
#: cannot be written at all, so this is the ceiling on "do not truncate".
EXCEL_CELL_LIMIT = 32_767


def _clip(value: Any, limit: int) -> Any:
    """Keep the whole value, bounded only by what a cell can physically hold.

    ``limit`` of 0 means "no configured limit"; the Excel ceiling still applies,
    because a longer cell raises rather than silently truncating.  When a value
    does hit the ceiling the marker says so, so a reader can tell a complete
    response from one that ran out of cell.
    """
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)
    ceiling = min(limit or EXCEL_CELL_LIMIT, EXCEL_CELL_LIMIT)
    if len(text) > ceiling:
        marker = "...[truncated at cell limit]"
        return text[: ceiling - len(marker)] + marker
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
            _write_sheet(
                writer, _build_skipped_modes_frame(result),
                _safe_sheet_name("Skipped_modes", used),
            )
            _write_sheet(
                writer, _build_introduced_modes_frame(result),
                _safe_sheet_name("Introduced_modes", used),
            )
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
    """Dataset x model matrix of primary-metric values.

    Columns are keyed by model and *execution mode*: one column per model when a
    run uses a single mode, and one per (model, mode) when it compares several,
    so that io and cot -- or SCS and BOV -- sit side by side for the same
    dataset.  The full identity of every task stays in ``Summary_Long``,
    ``Metrics`` and ``Tasks``, where ``template_mode`` names it exactly.
    """
    if summary.empty:
        return summary
    frame = summary.copy()
    frame["row"] = frame["dataset_id"] + " [" + frame["primary_metric"].fillna("") + "]"
    modes = frame.get("prompt_mode")
    if modes is None:
        modes = pd.Series(["io"] * len(frame), index=frame.index)
    selection = frame.get("selection_mode")
    if selection is None:
        selection = pd.Series(["n/a"] * len(frame), index=frame.index)
    variants = modes.fillna("io").str.cat(
        selection.fillna("n/a").where(selection != "n/a", ""), sep="/"
    ).str.rstrip("/")
    if variants.nunique() > 1:
        frame["column"] = frame["model_id"] + " (" + variants + ")"
    else:
        frame["column"] = frame["model_id"]
    return frame.pivot_table(
        index="row", columns="column", values="value", aggfunc="first"
    ).sort_index()


def _header_format(writer: Any) -> Any:
    """The one header format, made once per workbook.

    xlsxwriter formats belong to a workbook, and re-adding an identical one for
    every sheet grows the file for no benefit, so it is cached on the writer.
    """
    cached = getattr(writer, "_abench_header_format", None)
    if cached is None:
        cached = writer.book.add_format(
            {
                "bold": True,
                "align": "center",
                "valign": "vcenter",
                "text_wrap": True,
                "bottom": 1,
            }
        )
        writer._abench_header_format = cached
    return cached


def _write_sheet(writer: Any, frame: pd.DataFrame, name: str, *, index: bool = False) -> None:
    """Write one sheet with a bold, centred, frozen header row.

    pandas writes the header with its own default format, so it is rewritten
    afterwards rather than fought with: same cells, same order, our format.
    Freezing the header keeps the column names on screen in sheets that run to
    thousands of rows -- which the per-sample sheets do -- and the first column
    is frozen too when it carries the row labels, so a wide matrix stays
    readable when it is scrolled sideways.
    """
    if frame is None or frame.empty:
        frame = pd.DataFrame({"note": ["no data"]})
        index = False
    frame.to_excel(writer, sheet_name=name, index=index)
    worksheet = writer.sheets[name]
    header = _header_format(writer)
    offset = 1 if index else 0

    if index:
        # The corner cell holds the index's name (often blank) and is part of
        # the header row, so it gets the same treatment.
        worksheet.write(0, 0, str(frame.index.name or ""), header)
    for position, column in enumerate(frame.columns):
        worksheet.write(0, position + offset, str(column), header)

    for position, column in enumerate(frame.columns, start=offset):
        # str() per cell rather than astype(str): pandas >= 2.1 leaves missing
        # values as float NaN under astype(str), which has no len().
        widths = [len(str(column))]
        series = frame.iloc[:, position - offset].head(200)
        widths.extend(len(str(value)) for value in series)
        worksheet.set_column(position, position, min(60, max(10, max(widths) + 2)))
    if index:
        labels = [len(str(frame.index.name or ""))]
        labels.extend(len(str(value)) for value in frame.index[:200])
        worksheet.set_column(0, 0, min(60, max(10, max(labels) + 2)))

    # Row 1 down scrolls; the header stays. With an index, its column stays too.
    worksheet.freeze_panes(1, offset)


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
    lines.append(f"* **Template mode**: `{identity.template_mode}`")
    lines.append(
        f"* **Prompt mode**: `{identity.prompt_mode}` \u00b7 "
        f"**selection mode**: `{identity.selection_mode}` \u00b7 "
        f"**delivery**: `{identity.data_delivery_mode}`"
    )
    lines.append(f"* **Prompts**: owned by the `{identity.dataset_id}` adapter "
                 f"(v{identity.template_version})")
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

    if result.introduced_modes:
        lines.append("## Additional hypothesis modes")
        lines.append("")
        lines.append(
            "These datasets were run in a mode the published dataset table does not list, "
            "because the benchmark itself defines it as a separate task. The formulation "
            "that justifies each is quoted."
        )
        lines.append("")
        for entry in result.introduced_modes:
            lines.append(
                f"* `{entry.get('dataset_id')}` -- ran **{entry.get('hypothesis_mode')}**; "
                f"the table lists \"{entry.get('table_says')}\".  \n"
                f"  {entry.get('justification')}"
            )
        lines.append("")

    if result.skipped_modes:
        lines.append("## Modes not run")
        lines.append("")
        for entry in result.skipped_modes:
            lines.append(
                f"* `{entry.get('dataset_id')}` / `{entry.get('mode')}`: {entry.get('reason')}"
            )
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
    if result.sync_stats:
        stats = result.sync_stats
        lines.append("## Off-box backup")
        lines.append("")
        lines.append(
            f"* {stats.get('successes', 0)} successful upload(s), "
            f"{stats.get('failures', 0)} failure(s) over {stats.get('ticks', 0)} tick(s)."
        )
        if stats.get("last_error"):
            lines.append(f"* Last error: `{_clip(stats['last_error'], 300)}`")
        lines.append(
            "* A non-zero failure count means the local copy in this directory is the "
            "authoritative one."
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
