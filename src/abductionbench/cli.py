"""Command-line interface.

::

    abench validate   configs/runs/pilot.yaml         # config + templates + adapters
    abench doctor     configs/runs/pilot.yaml         # probe endpoints (incl. batch route)
    abench prepare    configs/runs/pilot.yaml         # materialize datasets only
    abench run        configs/runs/pilot.yaml         # full evaluation
    abench report     runs/<run-id>                   # rebuild reports from records
    abench templates  configs/runs/pilot.yaml         # list/preview prompt templates

Every command takes the same config file; ``--set dotted.key=value`` overrides
anything in it, and ``--datasets``/``--models`` restrict the run.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .core.config import RunConfig, load_run_config
from .core.engine import EvaluationEngine, RunResult
from .core.errors import AbenchError
from .core.prompts import PromptRegistry, PromptRenderer
from .core.telemetry import setup_logging

app = typer.Typer(
    add_completion=False,
    help="AbductionBench -- configurable evaluation for abductive reasoning benchmarks.",
    no_args_is_help=True,
)
console = Console()
logger = logging.getLogger(__name__)

ConfigArg = typer.Argument(..., help="Path to a run configuration YAML file.")
SetOpt = typer.Option(None, "--set", "-s", help="Override config: dotted.key=value (repeatable).")
DatasetsOpt = typer.Option(None, "--datasets", "-d", help="Restrict to these dataset ids.")
ModelsOpt = typer.Option(None, "--models", "-m", help="Restrict to these model ids.")


def _load(
    config: Path,
    overrides: list[str] | None,
    datasets: list[str] | None,
    models: list[str] | None,
) -> RunConfig:
    try:
        return load_run_config(
            config, overrides=overrides, dataset_filter=datasets, model_filter=models
        )
    except AbenchError as exc:
        console.print(f"[red]configuration error:[/red] {exc}")
        raise typer.Exit(code=2) from exc


def _split_csv(values: list[str] | None) -> list[str] | None:
    """Accept both ``-d a -d b`` and ``-d a,b``."""
    if not values:
        return None
    out: list[str] = []
    for value in values:
        out.extend(part.strip() for part in value.split(",") if part.strip())
    return out or None


@app.command()
def validate(
    config: Path = ConfigArg,
    set_: list[str] | None = SetOpt,
    datasets: list[str] | None = DatasetsOpt,
    models: list[str] | None = ModelsOpt,
    check_adapters: bool = typer.Option(
        True, help="Import each dataset's adapter class (does not download data)."
    ),
) -> None:
    """Validate a run config: schema, prompt templates, bindings, adapter imports."""
    setup_logging(level="WARNING")
    run_config = _load(config, set_, _split_csv(datasets), _split_csv(models))
    registry = PromptRegistry(list(run_config.prompts.template_dirs))
    renderer = PromptRenderer(registry, run_config.prompts)

    table = Table(title=f"run '{run_config.name}' validates")
    table.add_column("check")
    table.add_column("detail")
    table.add_row("models", ", ".join(m.id for m in run_config.models))
    table.add_row("datasets", str(len(run_config.enabled_datasets())))
    table.add_row("templates found", f"{len(registry)}: {', '.join(registry.ids())}")

    problems: list[str] = []
    for dataset in run_config.enabled_datasets():
        for binding in renderer.bindings_for_dataset(dataset.id, dataset.prompt_bindings):
            for task_kind, template_id in binding.mapping.items():
                try:
                    registry.get(template_id)
                except AbenchError as exc:
                    problems.append(f"{dataset.id}/{binding.variant}/{task_kind}: {exc}")
        if check_adapters:
            from .core.registry import resolve_adapter

            try:
                resolve_adapter(dataset.impl)
            except AbenchError as exc:
                problems.append(f"{dataset.id}: {exc}")
    table.add_row("binding/adapter problems", str(len(problems)))
    console.print(table)
    for problem in problems:
        console.print(f"[red]x[/red] {problem}")
    if problems:
        raise typer.Exit(code=1)
    console.print("[green]configuration is valid[/green]")


@app.command()
def templates(
    config: Path = ConfigArg,
    set_: list[str] | None = SetOpt,
    show: str | None = typer.Option(None, help="Print the full body of one template id."),
) -> None:
    """List the prompt templates a run can use (and preview one)."""
    setup_logging(level="WARNING")
    run_config = _load(config, set_, None, None)
    registry = PromptRegistry(list(run_config.prompts.template_dirs))
    if show:
        template = registry.get(show)
        console.print(f"[bold]{template.id}@{template.version}[/bold] -- {template.description}")
        console.print(f"task_kinds: {template.task_kinds}")
        console.print(f"required_fields: {template.required_fields}")
        console.print(f"optional_fields: {template.optional_fields}")
        console.print(f"output_contract: {json.dumps(template.output_contract, indent=2)}")
        for message in template.messages:
            console.rule(message["role"])
            console.print(message["content"])
        return
    table = Table(title="prompt templates")
    for column in ("id", "version", "task kinds", "required fields", "description"):
        table.add_column(column)
    for template_id in registry.ids():
        template = registry.get(template_id)
        table.add_row(
            template.id,
            template.version,
            ",".join(template.task_kinds),
            ",".join(template.required_fields),
            template.description[:60],
        )
    console.print(table)


@app.command()
def doctor(
    config: Path = ConfigArg,
    set_: list[str] | None = SetOpt,
    models: list[str] | None = ModelsOpt,
) -> None:
    """Probe every model endpoint: discovery, chat, and the native batch route."""
    setup_logging(level="INFO")
    run_config = _load(config, set_, None, _split_csv(models))

    async def _probe() -> list[dict]:
        from .core.client import ModelClient

        reports = []
        for model in run_config.models:
            client = ModelClient(model, run_config.engine.timeouts)
            try:
                reports.append(await client.verify())
            except AbenchError as exc:
                reports.append({"model_id": model.id, "error": str(exc)})
            finally:
                await client.aclose()
        return reports

    reports = asyncio.run(_probe())
    table = Table(title="endpoint diagnostics")
    for column in ("model", "served?", "batch?", "base url", "batch url", "note"):
        table.add_column(column, overflow="fold")
    unhealthy = 0
    for report in reports:
        ok_model = bool(report.get("model_available"))
        ok_batch = bool(report.get("batch_ok"))
        unhealthy += 0 if (ok_model and ok_batch) else 1
        table.add_row(
            str(report.get("model_id")),
            "[green]yes[/green]" if ok_model else "[red]no[/red]",
            "[green]yes[/green]" if ok_batch else "[yellow]no[/yellow]",
            str(report.get("base_url", "")),
            str(report.get("batch_url", "")),
            str(report.get("error") or report.get("batch_error") or report.get("discovery_error") or ""),
        )
    console.print(table)
    if unhealthy:
        raise typer.Exit(code=1)


@app.command()
def prepare(
    config: Path = ConfigArg,
    set_: list[str] | None = SetOpt,
    datasets: list[str] | None = DatasetsOpt,
    offline: bool = typer.Option(False, help="Fail instead of downloading anything."),
) -> None:
    """Materialize datasets and build their samples without calling any model."""
    run_config = _load(config, set_, _split_csv(datasets), None)
    engine = EvaluationEngine(run_config, offline=offline, dry_run=True)
    bundles = engine._build_bundles()  # noqa: SLF001 - deliberate reuse of stage A
    table = Table(title="dataset preparation")
    for column in ("dataset", "samples", "split", "primary metric", "status"):
        table.add_column(column, overflow="fold")
    failures = 0
    for bundle in bundles:
        if bundle.skipped:
            failures += 1
            table.add_row(bundle.config.id, "-", "-", "-", f"[red]skipped:[/red] {bundle.skipped_reason}")
            continue
        doc = bundle.documentation
        table.add_row(
            bundle.config.id,
            str(len(bundle.samples)),
            (doc.split_used if doc else "") or "",
            (doc.primary_metric if doc else "") or "",
            "[green]ok[/green]",
        )
    console.print(table)
    console.print(f"prepared {len(bundles) - failures}/{len(bundles)} dataset(s)")


@app.command()
def run(
    config: Path = ConfigArg,
    set_: list[str] | None = SetOpt,
    datasets: list[str] | None = DatasetsOpt,
    models: list[str] | None = ModelsOpt,
    run_id: str | None = typer.Option(None, help="Reuse a run id to resume into its directory."),
    resume: Path | None = typer.Option(
        None, help="Resume into an existing run directory (implies its run id)."
    ),
    dry_run: bool = typer.Option(False, help="Plan and render prompts, but call no model."),
    offline: bool = typer.Option(False, help="Fail instead of downloading datasets."),
    no_report: bool = typer.Option(False, help="Skip report generation."),
) -> None:
    """Run an evaluation end to end."""
    run_config = _load(config, set_, _split_csv(datasets), _split_csv(models))
    run_dir = Path(resume) if resume else None
    effective_run_id = run_id or (run_dir.name if run_dir else None)

    engine = EvaluationEngine(
        run_config,
        run_id=effective_run_id,
        run_dir=run_dir,
        offline=offline,
        dry_run=dry_run,
    )
    try:
        result: RunResult = asyncio.run(engine.run())
    except KeyboardInterrupt:
        console.print("[yellow]interrupted -- partial results are checkpointed[/yellow]")
        raise typer.Exit(code=130) from None
    except AbenchError as exc:
        console.print(f"[red]run failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if not no_report and not dry_run:
        from .core.reporting import write_reports

        written = write_reports(result)
        console.print(f"[green]reports:[/green] {written.get('excel', '-')}")
        # The reports were written after the engine's final upload, so send them too.
        stats = engine.flush_sync()
        if stats:
            console.print(
                f"[green]backup:[/green] {engine.sync.destination} "
                f"({stats['successes']} ok, {stats['failures']} failed)"
            )
    _print_summary(result)
    if any(task.failure for task in result.tasks):
        raise typer.Exit(code=3)


@app.command()
def report(
    run_dir: Path = typer.Argument(..., help="An existing run directory."),
) -> None:
    """Rebuild reports for a finished (or interrupted) run from its records."""
    setup_logging(level="INFO")
    from .core.rebuild import rebuild_run_result
    from .core.reporting import write_reports

    result = rebuild_run_result(run_dir)
    written = write_reports(result)
    console.print(f"[green]reports rebuilt:[/green] {written.get('excel', '-')}")
    _print_summary(result)


def _print_summary(result: RunResult) -> None:
    from .core.reporting import build_summary_frame

    frame = build_summary_frame(result)
    if frame.empty:
        console.print("[yellow]no results[/yellow]")
        return
    table = Table(title=f"run {result.run_id}")
    for column in frame.columns:
        table.add_column(str(column), overflow="fold")
    for _, row in frame.iterrows():
        table.add_row(*[str(value) for value in row.tolist()])
    console.print(table)
    if result.skipped_datasets:
        console.print(f"[yellow]{len(result.skipped_datasets)} dataset(s) skipped[/yellow]")
        for entry in result.skipped_datasets:
            console.print(f"  - {entry['dataset_id']}: {entry['reason'][:160]}")


def main() -> None:
    """Console-script entry point."""
    try:
        app()
    except AbenchError as exc:  # pragma: no cover - top-level guard
        console.print(f"[red]error:[/red] {exc}")
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
