"""COT-only reasoning-structure metrics, derivation, caching, and persistence."""

from __future__ import annotations

import asyncio
import re
from collections import Counter
from pathlib import Path

import pytest

from abductionbench.core.checkpoint import dedupe_records, load_records
from abductionbench.core.config import load_run_config
from abductionbench.core.engine import EvaluationEngine
from abductionbench.core.reasoning_judge import derive_reasoning_metrics
from abductionbench.core.reporting import _build_samples_frame


def _raw_metrics() -> dict:
    return {
        "observation_inventory": {"total_observations": 4},
        "observation_coverage": {"total_observations": 4, "observations_used": 3},
        "branchiness_diversity": {"branchiness": 2, "diversity": 1},
        "density": {
            "total_steps": 4,
            "useless_steps": 1,
            "useful_steps": 3,
            "reasoning_density": 1.0,
        },
        "redundancy_completeness": {"redundancy": 1, "completeness": 2},
        "directionality": {"directionality": 1},
        "backtracking": {"backtracking": 1},
        "differential_elimination": {"differential_elimination": 2},
        "prior_knowledge": {"prior_knowledge": 1},
        "uncertainty": {"uncertainty_steps": 2},
    }


def test_generation_derivations_reuse_shared_normalizers():
    metrics, errors, inapplicable = derive_reasoning_metrics(
        _raw_metrics(), generation_like=True, selection_like=False, option_count=0
    )

    assert errors == []
    assert inapplicable == ["differential_elimination:not_selection_or_pipeline"]
    assert metrics["reasoning_observation_coverage"] == pytest.approx(3 / 4)
    assert metrics["reasoning_useless_step_fraction"] == pytest.approx(1 / 4)
    assert metrics["reasoning_useful_step_fraction"] == pytest.approx(3 / 4)
    assert metrics["reasoning_density_normalized"] == pytest.approx(1 / 2)
    assert metrics["reasoning_redundancy_normalized"] == pytest.approx(1 / 4)
    assert metrics["reasoning_completeness_normalized"] == pytest.approx(2 / 4)
    assert metrics["reasoning_backtracking_normalized"] == pytest.approx(1 / 4)
    assert metrics["reasoning_uncertainty_normalized"] == pytest.approx(2 / 4)
    assert "reasoning_differential_elimination" not in metrics


def test_selection_and_pipeline_use_the_required_different_density_normalizers():
    selection, errors, _ = derive_reasoning_metrics(
        _raw_metrics(), generation_like=False, selection_like=True, option_count=3
    )
    assert errors == []
    assert selection["reasoning_density_normalized"] == pytest.approx(1 / 3)
    # sum(C(3,k), k=2..3) = 3 + 1 = 4
    assert selection["reasoning_differential_elimination_normalized"] == pytest.approx(2 / 4)
    assert "reasoning_branchiness" not in selection

    pipeline, errors, _ = derive_reasoning_metrics(
        _raw_metrics(),
        generation_like=False,
        selection_like=False,
        pipeline_like=True,
        option_count=3,
    )
    assert errors == []
    # Pipeline density uses Branchiness, not the option count.
    assert pipeline["reasoning_density_normalized"] == pytest.approx(1 / 2)
    assert pipeline["reasoning_differential_elimination_normalized"] == pytest.approx(2 / 4)


def test_inconsistent_step_counts_are_reported_not_forced():
    raw = _raw_metrics()
    raw["density"] = {
        "total_steps": 4,
        "useless_steps": 2,
        "useful_steps": 3,
        "reasoning_density": 1.0,
    }
    metrics, errors, _ = derive_reasoning_metrics(
        raw, generation_like=True, selection_like=False, option_count=0
    )
    assert "density:invalid_counts_or_sum" in errors
    assert "reasoning_total_steps" not in metrics
    assert "reasoning_backtracking_normalized" not in metrics
    assert "reasoning_uncertainty_normalized" not in metrics


def _judge_responder(counts: Counter):
    def respond(conversation, _max_tokens):
        text = "\n".join(str(message.get("content", "")) for message in conversation)
        if "CACHED TOTAL OBSERVATIONS" in text:
            counts["coverage"] += 1
            return '{"total_observations": 4, "observations_used": 3}'
        if "inventory evidence" in text:
            counts["inventory"] += 1
            return '{"total_observations": 4}'
        if "Branchiness is the number" in text:
            counts["branchiness"] += 1
            return '{"branchiness": 2, "diversity": 1}'
        if "Reasoning density is" in text:
            counts["density"] += 1
            return (
                '{"total_steps": 4, "useless_steps": 1, "useful_steps": 3, '
                '"reasoning_density": 1.0}'
            )
        if "Redundancy is the number" in text:
            counts["redundancy"] += 1
            return '{"redundancy": 1, "completeness": 2}'
        if "Score 0 when" in text:
            counts["directionality"] += 1
            return '{"directionality": 1}'
        if "count backtracking" in text:
            counts["backtracking"] += 1
            return '{"backtracking": 1}'
        if "differential elimination" in text:
            counts["differential"] += 1
            return '{"differential_elimination": 1}'
        if "prior/background knowledge" in text:
            counts["prior"] += 1
            return '{"prior_knowledge": 1}'
        if "count uncertainty marking" in text:
            counts["uncertainty"] += 1
            return '{"uncertainty_steps": 2}'
        if "You measure differential elimination" in text:
            counts["differential"] += 1
            return '{"differential_elimination": 2}'

        match = re.search(r"observation number (\d+)", text)
        index = match.group(1) if match else "0"
        return (
            f"Observation {index} supports explanation A. Explanation B conflicts with it. "
            f"I may need to reconsider, but A is best. Answer: echo observation number {index}"
        )

    return respond


def test_cot_metrics_are_cached_and_persisted_per_sample(
    fake_server, write_run_config, fake_dataset
):
    counts: Counter = Counter()
    fake_server.state.responder = _judge_responder(counts)
    config_path = write_run_config(
        base_url=fake_server.base_url,
        datasets=[fake_dataset(sample_size=2, n=2)],
        modes={"prompt_modes": ["cot"], "repeats": 2},
        engine={
            "reasoning_judge": {
                "enabled": True,
                "model": "fake-model",
                "group_size": 8,
            }
        },
    )
    result = asyncio.run(EvaluationEngine(load_run_config(config_path)).run())

    task = result.tasks[0]
    records = dedupe_records(load_records(task.output_dir / "records.jsonl"))
    assert len(records) == 4
    for record in records:
        metrics = record["metrics"]
        assert metrics["reasoning_observations_total"] == 4
        assert metrics["reasoning_observations_used"] == 3
        assert metrics["reasoning_observation_coverage"] == pytest.approx(0.75)
        assert metrics["reasoning_branchiness"] == 2
        assert metrics["reasoning_density_normalized"] == pytest.approx(0.5)
        assert metrics["reasoning_backtracking_normalized"] == pytest.approx(0.25)
        assert metrics["reasoning_uncertainty_normalized"] == pytest.approx(0.5)
        assert record["details"]["reasoning_metrics_status"] == "ok"
        assert record["details"]["reasoning_metrics_inapplicable"] == [
            "differential_elimination:not_selection_or_pipeline"
        ]

    # Two distinct questions, each repeated twice: inventory is bought once per
    # question while output-dependent metrics are judged once per output.
    assert counts["inventory"] == 2
    assert counts["coverage"] == 4
    assert (result.run_dir / "reasoning_judge_cache" / "verdicts.json").exists()

    sheet = _build_samples_frame([task.output_dir], clip=32_000, limit=100)
    expected_columns = {
        "metric.reasoning_observations_total",
        "metric.reasoning_observations_used",
        "metric.reasoning_observation_coverage",
        "metric.reasoning_branchiness",
        "metric.reasoning_diversity",
        "metric.reasoning_total_steps",
        "metric.reasoning_useless_steps",
        "metric.reasoning_useful_steps",
        "metric.reasoning_density",
        "metric.reasoning_useless_step_fraction",
        "metric.reasoning_useful_step_fraction",
        "metric.reasoning_density_normalized",
        "metric.reasoning_redundancy",
        "metric.reasoning_completeness",
        "metric.reasoning_redundancy_normalized",
        "metric.reasoning_completeness_normalized",
        "metric.reasoning_directionality",
        "metric.reasoning_backtracking",
        "metric.reasoning_backtracking_normalized",
        "metric.reasoning_prior_knowledge",
        "metric.reasoning_uncertainty_steps",
        "metric.reasoning_uncertainty_normalized",
        "reasoning_metrics_status",
        "reasoning_metrics_inapplicable",
        "reasoning_judge_errors",
    }
    assert expected_columns <= set(sheet.columns)

    calls_after_first = counts.copy()
    continued = asyncio.run(
        EvaluationEngine(
            load_run_config(config_path),
            run_id=result.run_id,
            run_dir=result.run_dir,
        ).run()
    )
    assert continued.tasks[0].n_reused == 4
    assert counts == calls_after_first  # every reasoning verdict came from the run cache
    continued_records = dedupe_records(
        load_records(continued.tasks[0].output_dir / "records.jsonl")
    )
    assert all(
        record["metrics"]["reasoning_observation_coverage"] == pytest.approx(0.75)
        for record in continued_records
    )


def test_io_outputs_never_call_the_reasoning_judge(
    fake_server, write_run_config, fake_dataset
):
    counts: Counter = Counter()
    fake_server.state.responder = _judge_responder(counts)
    config_path = write_run_config(
        base_url=fake_server.base_url,
        datasets=[fake_dataset(sample_size=2, n=2)],
        modes={"prompt_modes": ["io"]},
        engine={
            "reasoning_judge": {
                "enabled": True,
                "model": "fake-model",
                "group_size": 8,
            }
        },
    )
    result = asyncio.run(EvaluationEngine(load_run_config(config_path)).run())

    records = dedupe_records(load_records(result.tasks[0].output_dir / "records.jsonl"))
    assert counts == Counter()
    assert all(
        not any(name.startswith("reasoning_") for name in record["metrics"])
        for record in records
    )


def test_coworker_smoke_runs_and_backs_up_cot_metrics(fake_server, monkeypatch, tmp_path):
    """Exercise the shipped startup config through inference, judging and backup."""
    counts: Counter = Counter()
    fake_server.state.responder = _judge_responder(counts)
    for key in ("ABENCH_API_KEY", "ABENCH_ADMIN_KEY"):
        monkeypatch.setenv(key, "test-key")
    monkeypatch.setenv("ABENCH_LOCAL_URL", fake_server.base_url)
    monkeypatch.setenv("ABENCH_JUDGE_URL", fake_server.base_url)
    monkeypatch.setenv("ABENCH_JUDGE_MODEL", "gpt-oss-20b-local")
    root = Path(__file__).resolve().parents[1]
    config = load_run_config(
        root / "configs/runs/coworker_smoke.yaml",
        overrides=[
            f"engine.output_root={tmp_path}/runs", f"engine.data_root={tmp_path}/data",
            f"engine.sync.remote_path={tmp_path}/backup", "engine.sync.interval_s=3600",
            "engine.tokenizer.backend=heuristic", "engine.retry.recovery.enabled=false",
        ],
    )
    result = asyncio.run(EvaluationEngine(config, offline=True).run())
    assert len(result.tasks) == 4
    records = [
        record
        for task in result.tasks
        for record in dedupe_records(load_records(task.output_dir / "records.jsonl"))
    ]
    assert len(records) == 8
    cot = [record for record in records if record["prompt_mode"] == "cot"]
    io = [record for record in records if record["prompt_mode"] == "io"]
    assert len(cot) == len(io) == 4
    assert all(record["details"]["reasoning_metrics_status"] == "ok" for record in cot)
    assert all("reasoning_density_normalized" in record["metrics"] for record in cot)
    assert all(
        not any(key.startswith("reasoning_") for key in record["metrics"])
        for record in io
    )
    assert counts["inventory"] == 4
    assert counts["differential"] == 2
    assert result.sync_stats["successes"] >= 1
    assert result.sync_stats["failures"] == 0
    backup = tmp_path / "backup" / result.run_id
    assert (backup / "reasoning_judge_cache/verdicts.json").exists()
    backed_up_records = list(backup.rglob("records.jsonl"))
    assert len(backed_up_records) == 4
