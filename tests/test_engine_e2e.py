"""End-to-end engine tests over real HTTP against a fake vLLM server.

These cover the behaviours that make the engine trustworthy in production:
native batching with the configured group size, per-sample checkpointing and
resume, retry on transient failures, bisection of a batch poisoned by one bad
sample, oversize-sample replacement, single-call fallback, and reporting.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import orjson
import pytest

from abductionbench.core.config import load_run_config
from abductionbench.core.engine import EvaluationEngine
from abductionbench.core.reporting import write_reports


def _run(config_path: Path, **kwargs):
    config = load_run_config(config_path)
    engine = EvaluationEngine(config, **kwargs)
    return asyncio.run(engine.run()), engine


def _records(task_dir: Path) -> list[dict]:
    path = task_dir / "records.jsonl"
    return [orjson.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


def test_batched_run_produces_records_and_metrics(fake_server, write_run_config, fake_dataset):
    config_path = write_run_config(
        base_url=fake_server.base_url, datasets=[fake_dataset("fake", n=8, sample_size=8)]
    )
    result, _ = _run(config_path)

    assert len(result.tasks) == 1
    task = result.tasks[0]
    assert task.n_planned == 8
    assert task.n_scored == 8
    assert task.n_error == 0
    assert task.metrics["coverage"] == 1.0
    # The first batch call is the engine's startup probe (2 conversations);
    # then group_size=4 over 8 samples -> exactly two batch calls, run server-side.
    assert fake_server.state.batch_calls == [2, 4, 4]
    assert fake_server.state.single_calls == 0

    records = _records(task.output_dir)
    assert len(records) == 8
    assert all(record["status"] == "ok" for record in records)
    assert all(record["response"]["batch_size"] == 4 for record in records)
    # Batch usage is an aggregate; the record says so rather than pretending
    # it is per-sample.
    assert all(record["response"]["usage_is_batch_aggregate"] for record in records)
    assert (task.output_dir / "metrics.json").exists()
    assert (task.output_dir / "checkpoint.json").exists()
    assert list((task.output_dir / "raw").glob("*.json"))


def test_per_model_group_size_is_honoured(fake_server, write_run_config, fake_dataset):
    models = [
        {
            "id": "m8",
            "model_name": "test/model",
            "endpoint": {
                "base_url": fake_server.base_url,
                "api_key": "test-key",
                "batch": {"enabled": True, "group_size": 8},
            },
            "sampling": {"max_tokens_default": 64},
            "limits": {"max_parallel_batches": 1},
        },
        {
            "id": "m3",
            "model_name": "test/model",
            "endpoint": {
                "base_url": fake_server.base_url,
                "api_key": "test-key",
                "batch": {"enabled": True, "group_size": 3},
            },
            "sampling": {"max_tokens_default": 64},
            "limits": {"max_parallel_batches": 1},
        },
    ]
    config_path = write_run_config(
        base_url=fake_server.base_url,
        datasets=[fake_dataset("fake", n=6, sample_size=6)],
        models=models,
        engine={"concurrency": {"max_parallel_tasks": 1}},
    )
    result, _ = _run(config_path)
    assert len(result.tasks) == 2
    # 6 samples: model m8 -> one call of 6; model m3 -> two calls of 3.
    # The verification probe also issues one 2-conversation call per model.
    sizes = sorted(fake_server.state.batch_calls)
    assert sizes.count(2) == 2  # probes
    assert 6 in sizes and sizes.count(3) == 2


def test_resume_skips_completed_samples(fake_server, write_run_config, fake_dataset):
    config_path = write_run_config(
        base_url=fake_server.base_url, datasets=[fake_dataset("fake", n=8, sample_size=8)]
    )
    result, engine = _run(config_path)
    calls_first = len(fake_server.state.batch_calls)

    # Re-run into the same directory: every sample is checkpointed already.
    config = load_run_config(config_path)
    engine2 = EvaluationEngine(config, run_id=result.run_id, run_dir=result.run_dir)
    result2 = asyncio.run(engine2.run())
    task = result2.tasks[0]
    assert task.n_reused == 8
    assert task.n_scored == 8
    # Only the endpoint verification probe was issued the second time.
    assert len(fake_server.state.batch_calls) == calls_first + 1


def test_resume_is_invalidated_by_a_template_change(fake_server, write_run_config, fake_dataset):
    config_path = write_run_config(
        base_url=fake_server.base_url, datasets=[fake_dataset("fake", n=4, sample_size=4)]
    )
    result, _ = _run(config_path)

    changed = write_run_config(
        base_url=fake_server.base_url,
        datasets=[fake_dataset("fake", n=4, sample_size=4)],
        prompts={"bindings": {"generation": "gen_cot_v1"}},
        name="test-run-2",
    )
    config = load_run_config(changed)
    engine = EvaluationEngine(config, run_id=result.run_id, run_dir=result.run_dir)
    result2 = asyncio.run(engine.run())
    # Different template id -> a different task directory, nothing reused.
    assert result2.tasks[0].n_reused == 0
    assert result2.tasks[0].identity.template_id == "gen_cot_v1"


def test_transient_failures_are_retried(fake_server, write_run_config, fake_dataset):
    fake_server.state.fail_next = [503, 429]  # both consumed by the first calls
    config_path = write_run_config(
        base_url=fake_server.base_url, datasets=[fake_dataset("fake", n=4, sample_size=4)]
    )
    result, _ = _run(config_path)
    task = result.tasks[0]
    assert task.n_scored == 4
    assert task.n_error == 0
    events = [
        json.loads(line)
        for line in (result.run_dir / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert any(event["event"] == "batch_attempt_failed" for event in events)


def test_permanent_failure_is_recorded_per_sample(fake_server, write_run_config, fake_dataset):
    fake_server.state.fail_next = [503] * 40  # exceeds max_attempts for every call
    config_path = write_run_config(
        base_url=fake_server.base_url, datasets=[fake_dataset("fake", n=4, sample_size=4)]
    )
    result, _ = _run(config_path)
    task = result.tasks[0]
    assert task.n_error == 4
    assert task.n_scored == 0
    assert task.metrics["coverage"] == 0.0
    statuses = {record["status"] for record in _records(task.output_dir)}
    assert statuses == {"error"}


def test_poison_sample_is_isolated_by_bisection(fake_server, write_run_config, fake_dataset):
    """One bad conversation fails the whole batch; the rest must still land."""
    fake_server.state.poison_marker = "POISON"
    config_path = write_run_config(
        base_url=fake_server.base_url,
        # sample index 0 carries the marker (poison_every=8 -> only s0000)
        datasets=[fake_dataset("fake", n=8, sample_size=8, poison_every=8)],
    )
    result, _ = _run(config_path)
    task = result.tasks[0]
    records = {record["sample_id"]: record for record in _records(task.output_dir)}
    assert len(records) == 8
    # The offending sample is skipped (context-length class), everything else scored.
    assert records["s0000"]["status"] == "skipped"
    assert sum(1 for r in records.values() if r["status"] == "ok") == 7
    assert task.checkpoint.bisections >= 1


def test_oversize_samples_are_replaced_not_dropped(fake_server, write_run_config, fake_dataset):
    config_path = write_run_config(
        base_url=fake_server.base_url,
        datasets=[
            fake_dataset("fake", n=6, sample_size=6, long_every=3, long_chars=20000, replacements=10)
        ],
        engine={"limits": {"input_token_budget": 2000, "on_oversize": "resample"}},
    )
    result, _ = _run(config_path)
    task = result.tasks[0]
    # Oversize samples (every 3rd, including a drawn replacement) were replaced
    # rather than dropped, so the evaluated set keeps its configured size.
    assert task.n_planned == 6
    assert task.n_scored == 6
    doc = task.documentation
    assert doc.statistics["oversize_dropped[default]"] == 3
    assert doc.statistics["replacements_used[default]"] == 3
    assert doc.statistics["prompts_kept[default]"] == 6


def test_oversize_skip_policy_shrinks_the_set(fake_server, write_run_config, fake_dataset):
    config_path = write_run_config(
        base_url=fake_server.base_url,
        datasets=[fake_dataset("fake", n=6, sample_size=6, long_every=3, long_chars=20000)],
        engine={"limits": {"input_token_budget": 2000, "on_oversize": "skip"}},
    )
    result, _ = _run(config_path)
    assert result.tasks[0].n_planned == 4


def test_dataset_mostly_oversize_is_reported_unusable(fake_server, write_run_config, fake_dataset):
    config_path = write_run_config(
        base_url=fake_server.base_url,
        datasets=[
            fake_dataset("fake", n=4, sample_size=4, long_every=1, long_chars=20000, replacements=0)
        ],
        engine={"limits": {"input_token_budget": 1000, "on_oversize": "resample"}},
    )
    result, _ = _run(config_path)
    task = result.tasks[0]
    assert task.failure and "input budget" in task.failure


def test_empty_response_is_its_own_status(fake_server, write_run_config, fake_dataset):
    """A reasoning model can burn max_tokens and return content: null."""
    fake_server.state.empty_marker = "observation number 1"
    config_path = write_run_config(
        base_url=fake_server.base_url, datasets=[fake_dataset("fake", n=4, sample_size=4)]
    )
    result, _ = _run(config_path)
    records = {r["sample_id"]: r for r in _records(result.tasks[0].output_dir)}
    assert records["s0001"]["status"] == "empty"
    assert records["s0001"]["parse_ok"] is False
    assert result.tasks[0].metrics["empty_response_rate"] == pytest.approx(0.25)


def test_fallback_to_single_calls_when_batch_route_missing(
    fake_server, write_run_config, fake_dataset
):
    fake_server.state.batch_enabled = False
    config_path = write_run_config(
        base_url=fake_server.base_url, datasets=[fake_dataset("fake", n=4, sample_size=4)]
    )
    result, _ = _run(config_path)
    task = result.tasks[0]
    assert task.n_scored == 4
    assert fake_server.state.single_calls == 4
    assert task.diagnostics["batch_mode"] is False


def test_batch_route_missing_without_fallback_aborts(fake_server, write_run_config, fake_dataset):
    fake_server.state.batch_enabled = False
    config_path = write_run_config(
        base_url=fake_server.base_url,
        datasets=[fake_dataset("fake", n=2, sample_size=2)],
        engine={"batching": {"fallback_to_single": False}},
    )
    from abductionbench.core.errors import AbenchError

    with pytest.raises(AbenchError, match="batch endpoint unusable"):
        _run(config_path)


def test_bad_api_key_fails_fast(fake_server, write_run_config, fake_dataset):
    fake_server.state.api_key = "correct-key"
    config_path = write_run_config(
        base_url=fake_server.base_url, datasets=[fake_dataset("fake", n=2, sample_size=2)]
    )
    from abductionbench.core.errors import AbenchError

    with pytest.raises(AbenchError, match="authentication failed"):
        _run(config_path)


def test_skipped_dataset_does_not_stop_the_run(fake_server, write_run_config, fake_dataset):
    config_path = write_run_config(
        base_url=fake_server.base_url,
        datasets=[
            fake_dataset("broken", skip="source repository is unreachable"),
            fake_dataset("fine", n=4, sample_size=4),
        ],
    )
    result, _ = _run(config_path)
    assert [entry["dataset_id"] for entry in result.skipped_datasets] == ["broken"]
    assert [task.identity.dataset_id for task in result.tasks] == ["fine"]


def test_template_variants_create_parallel_tasks(fake_server, write_run_config, fake_dataset):
    config_path = write_run_config(
        base_url=fake_server.base_url,
        datasets=[fake_dataset("fake", n=4, sample_size=4)],
        prompts={"template_variants": {"cot": {"generation": "gen_cot_v1"}}},
    )
    result, _ = _run(config_path)
    templates = sorted(task.identity.template_id for task in result.tasks)
    assert templates == ["gen_cot_v1", "gen_freeform_v1"]
    # Both variants evaluated the same samples.
    assert {task.n_planned for task in result.tasks} == {4}


def test_dry_run_calls_nothing(fake_server, write_run_config, fake_dataset):
    config_path = write_run_config(
        base_url=fake_server.base_url, datasets=[fake_dataset("fake", n=4, sample_size=4)]
    )
    result, _ = _run(config_path, dry_run=True)
    assert fake_server.state.requests == 0
    assert result.tasks[0].n_planned == 4


def test_reports_are_written(fake_server, write_run_config, fake_dataset):
    config_path = write_run_config(
        base_url=fake_server.base_url,
        datasets=[fake_dataset("a", n=4, sample_size=4), fake_dataset("b", n=4, sample_size=4)],
    )
    result, _ = _run(config_path)
    written = write_reports(result)
    excel = written["excel"]
    assert excel.exists() and excel.stat().st_size > 0
    assert (result.run_dir / "RUN_REPORT.md").exists()
    assert (written["summary_csv"]).exists()
    for task in result.tasks:
        assert (task.output_dir / "run_documentation.md").exists()

    import pandas as pd

    sheets = pd.read_excel(excel, sheet_name=None)
    assert {"Summary", "Metrics", "Tasks", "Datasets", "Skipped", "Models"} <= set(sheets)
    assert any(name.startswith("S_") for name in sheets)


def test_report_command_rebuilds_from_records(fake_server, write_run_config, fake_dataset):
    config_path = write_run_config(
        base_url=fake_server.base_url, datasets=[fake_dataset("fake", n=4, sample_size=4)]
    )
    result, _ = _run(config_path)
    from abductionbench.core.rebuild import rebuild_run_result

    rebuilt = rebuild_run_result(result.run_dir)
    assert len(rebuilt.tasks) == 1
    assert rebuilt.tasks[0].n_scored == 4
    assert rebuilt.tasks[0].metrics["coverage"] == 1.0


def test_exhausted_transient_retries_bisect_the_batch(fake_server, write_run_config, fake_dataset):
    """A batch too slow/flaky as a whole is halved rather than failed wholesale.

    Four injected 503s exhaust the retry budget of the first (size-4) call and
    of its first half, leaving the second half to succeed -- so half the samples
    survive a failure that would otherwise have lost all four.
    """
    fake_server.state.fail_next = [503] * 4
    config_path = write_run_config(
        base_url=fake_server.base_url,
        datasets=[fake_dataset("fake", n=4, sample_size=4)],
        engine={"batching": {"bisect_on_timeout": True}},
    )
    result, _ = _run(config_path)
    task = result.tasks[0]
    assert task.checkpoint.bisections >= 1
    # Samples are salvaged by the split instead of the whole batch being lost.
    assert task.n_scored >= 2
    assert task.n_scored + task.n_error == 4
    assert task.diagnostics["batch_mode"] is True


def test_timeout_bisection_can_be_disabled(fake_server, write_run_config, fake_dataset):
    fake_server.state.fail_next = [503] * 4
    config_path = write_run_config(
        base_url=fake_server.base_url,
        datasets=[fake_dataset("fake", n=4, sample_size=4)],
        engine={"batching": {"bisect_on_timeout": False}},
    )
    result, _ = _run(config_path)
    task = result.tasks[0]
    assert task.checkpoint.bisections == 0
    assert task.n_error == 4


def test_transient_probe_failure_keeps_batch_mode(fake_server, write_run_config, fake_dataset):
    """A momentary failure of the startup probe must not downgrade the run."""
    fake_server.state.fail_next = [503]  # only the probe call fails
    config_path = write_run_config(
        base_url=fake_server.base_url, datasets=[fake_dataset("fake", n=4, sample_size=4)]
    )
    result, _ = _run(config_path)
    task = result.tasks[0]
    assert task.diagnostics["batch_mode"] is True
    assert fake_server.state.single_calls == 0
    assert task.n_scored == 4
