"""Reconstruct a :class:`~abductionbench.core.engine.RunResult` from disk.

``abench report <run-dir>`` uses this to regenerate the workbook and
documentation for a run that already happened -- including one that was
interrupted -- without re-running any inference.  Metrics are recomputed from
``records.jsonl`` through the dataset's own adapter, so a fixed scorer can be
re-applied to stored responses for free.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import orjson

from .adapter import AdapterContext
from .checkpoint import TaskCheckpoint, load_records
from .config import RunConfig, load_yaml
from .engine import RunResult, TaskResult
from .errors import ConfigError
from .metrics import mean
from .registry import resolve_adapter
from .types import AdapterDocumentation, ResponseStatus, SampleScore, TaskIdentity

logger = logging.getLogger(__name__)

__all__ = ["rebuild_run_result"]


def _load_config(run_dir: Path) -> RunConfig:
    resolved = run_dir / "run_config.resolved.yaml"
    if not resolved.exists():
        raise ConfigError(f"{resolved} not found -- cannot rebuild reports for this run")
    data = load_yaml(resolved)
    return RunConfig.model_validate(data)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return orjson.loads(path.read_bytes())
    except orjson.JSONDecodeError:
        logger.warning("%s is not valid JSON; ignoring", path)
        return {}


def rebuild_run_result(run_dir: Path | str) -> RunResult:
    """Rebuild a run's result object by walking its task directories."""
    run_dir = Path(run_dir)
    config = _load_config(run_dir)
    result = RunResult(
        run_id=run_dir.name,
        run_dir=run_dir,
        config=config,
        started_at=0.0,
        finished_at=0.0,
    )

    datasets_root = run_dir / "datasets"
    if not datasets_root.exists():
        logger.warning("%s has no datasets/ directory", run_dir)
        return result

    adapters: dict[str, Any] = {}
    documentation: dict[str, AdapterDocumentation] = {}

    for dataset_dir in sorted(p for p in datasets_root.iterdir() if p.is_dir()):
        dataset_id = dataset_dir.name
        dataset_cfg = next((d for d in config.datasets if d.id == dataset_id), None)
        for model_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
            for template_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
                records = load_records(template_dir / "records.jsonl")
                if not records:
                    continue
                metrics_blob = _read_json(template_dir / "metrics.json")
                checkpoint_blob = _read_json(template_dir / "checkpoint.json")
                template_id, _, template_version = template_dir.name.partition("@")
                identity = TaskIdentity(
                    run_id=result.run_id,
                    dataset_id=dataset_id,
                    model_id=model_dir.name,
                    template_id=template_id,
                    template_version=template_version or "1.0",
                )

                # Recompute aggregates from stored per-sample metrics so a
                # rebuild reflects the records on disk, and fall back to the
                # stored metrics.json when no adapter is available.
                scores = [
                    SampleScore(
                        metrics={k: float(v) for k, v in (rec.get("metrics") or {}).items()},
                        prediction=rec.get("prediction"),
                        parse_ok=bool(rec.get("parse_ok", True)),
                        details=rec.get("details") or {},
                    )
                    for rec in records
                    if rec.get("status")
                    in (
                        ResponseStatus.OK.value,
                        ResponseStatus.EMPTY.value,
                        ResponseStatus.TRUNCATED.value,
                    )
                ]
                adapter = None
                if dataset_cfg is not None:
                    if dataset_id not in adapters:
                        try:
                            adapter_cls = resolve_adapter(dataset_cfg.impl)
                            context = AdapterContext(
                                dataset_id=dataset_id,
                                data_dir=Path(config.engine.data_root) / dataset_id,
                                sample_size=dataset_cfg.sample_size,
                                seed=dataset_cfg.seed
                                if dataset_cfg.seed is not None
                                else config.seed,
                                options=dict(dataset_cfg.options),
                                offline=True,
                            )
                            adapters[dataset_id] = adapter_cls(context)
                            documentation[dataset_id] = adapters[dataset_id].documentation()
                        except Exception as exc:  # noqa: BLE001
                            logger.warning(
                                "cannot instantiate adapter for %s (%s); using stored metrics",
                                dataset_id,
                                exc,
                            )
                            adapters[dataset_id] = None
                    adapter = adapters.get(dataset_id)

                metrics: dict[str, float] = {}
                if adapter is not None and scores:
                    try:
                        metrics.update(
                            {k: float(v) for k, v in (adapter.aggregate(scores) or {}).items()}
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("aggregate() failed for %s: %s", dataset_id, exc)
                if not metrics:
                    metrics.update(
                        {
                            k: float(v)
                            for k, v in (metrics_blob.get("metrics") or {}).items()
                            if isinstance(v, (int, float))
                        }
                    )

                n_error = sum(
                    1 for rec in records if rec.get("status") == ResponseStatus.ERROR.value
                )
                n_skipped = sum(
                    1 for rec in records if rec.get("status") == ResponseStatus.SKIPPED.value
                )
                planned = int(
                    (metrics_blob.get("counts") or {}).get("planned")
                    or checkpoint_blob.get("total_planned")
                    or len(records)
                )
                metrics.setdefault("n_planned", float(planned))
                metrics["n_scored"] = float(len(scores))
                metrics["n_error"] = float(n_error)
                metrics["n_skipped"] = float(n_skipped)
                metrics["coverage"] = len(scores) / max(1, planned)
                metrics["parse_failure_rate"] = (
                    sum(1 for s in scores if not s.parse_ok) / len(scores) if scores else 0.0
                )
                metrics["truncation_rate"] = (
                    sum(
                        1
                        for rec in records
                        if rec.get("status") == ResponseStatus.TRUNCATED.value
                    )
                    / max(1, planned)
                )
                metrics["empty_response_rate"] = (
                    sum(
                        1 for rec in records if rec.get("status") == ResponseStatus.EMPTY.value
                    )
                    / max(1, planned)
                )
                latencies = [
                    float((rec.get("response") or {}).get("latency_s") or 0.0) for rec in records
                ]
                if any(latencies):
                    metrics["batch_latency_s_mean"] = mean([v for v in latencies if v])

                primary = (
                    metrics_blob.get("primary_metric")
                    or (dataset_cfg.primary_metric if dataset_cfg else None)
                    or (adapter.primary_metric if adapter else "")
                )
                if primary and primary in metrics:
                    metrics[f"{primary}_strict"] = metrics[primary] * metrics["coverage"]

                checkpoint = None
                if checkpoint_blob:
                    checkpoint = TaskCheckpoint(task=identity.as_dict())
                    for key, value in checkpoint_blob.items():
                        if hasattr(checkpoint, key) and key != "task":
                            setattr(checkpoint, key, value)

                result.tasks.append(
                    TaskResult(
                        identity=identity,
                        output_dir=template_dir,
                        metrics=metrics,
                        checkpoint=checkpoint,
                        n_planned=planned,
                        n_scored=len(scores),
                        n_error=n_error,
                        n_skipped=n_skipped,
                        primary_metric=primary or "",
                        documentation=documentation.get(dataset_id),
                        failure=metrics_blob.get("failure"),
                        diagnostics=metrics_blob.get("diagnostics") or {},
                    )
                )

    events = run_dir / "events.jsonl"
    if events.exists():
        for line in events.read_bytes().splitlines():
            if not line.strip():
                continue
            try:
                event = orjson.loads(line)
            except orjson.JSONDecodeError:
                continue
            if event.get("event") == "dataset_skipped":
                result.skipped_datasets.append(
                    {"dataset_id": event.get("dataset_id", ""), "reason": event.get("reason", "")}
                )
            elif event.get("event") == "endpoint_verified":
                result.endpoint_reports.append(
                    {k: v for k, v in event.items() if k not in ("ts", "event")}
                )
            elif event.get("event") == "run_started":
                result.started_at = float(event.get("ts", 0.0))
            elif event.get("event") == "run_finished":
                result.finished_at = float(event.get("ts", 0.0))
    return result
