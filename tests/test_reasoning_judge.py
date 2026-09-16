"""COT-only reasoning-structure metrics, derivation, caching, and persistence."""

from __future__ import annotations

import asyncio
import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from abductionbench.core.checkpoint import dedupe_records, load_records
from abductionbench.core.config import load_run_config
from abductionbench.core.engine import EvaluationEngine
from abductionbench.core.reasoning_judge import (
    _GENERATION_KINDS,
    _SELECTION_KINDS,
    ReasoningJudgeStage,
    derive_reasoning_metrics,
)
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


def test_every_shipped_task_kind_has_a_reasoning_shape():
    """A task kind no shape claims would drop metrics with no explanation."""
    adapters = Path(__file__).resolve().parent.parent / "src" / "abductionbench" / "adapters"
    kinds = {
        match
        for path in adapters.glob("*.py")
        for match in re.findall(r'task_kind=["\']([a-z_]+)', path.read_text(encoding="utf-8"))
    }
    assert kinds - {"judge"} <= _GENERATION_KINDS | _SELECTION_KINDS


def test_multi_selection_is_judged_as_a_selection():
    metrics, errors, inapplicable = derive_reasoning_metrics(
        _raw_metrics(), generation_like=False, selection_like=True, option_count=4
    )
    assert errors == []
    assert inapplicable == ["branchiness_diversity:not_generation_or_pipeline"]
    assert metrics["reasoning_density_normalized"] == pytest.approx(1 / 4)
    assert "reasoning_differential_elimination" in metrics


def test_a_count_larger_than_the_chain_is_dropped_not_averaged():
    """A judge once answered 147 backtracks for a 14-step chain.

    The raw number was recorded and only the ratio was withheld, so one reply
    moved that metric's mean by fifty-fold across otherwise identical runs.
    """
    raw = _raw_metrics()
    raw["backtracking"] = {"backtracking": 147}
    raw["uncertainty"] = {"uncertainty_steps": 99}
    metrics, errors, _ = derive_reasoning_metrics(
        raw, generation_like=True, selection_like=False, option_count=0
    )
    assert "reasoning_backtracking" not in metrics
    assert "reasoning_backtracking_normalized" not in metrics
    assert "reasoning_uncertainty_steps" not in metrics
    assert "backtracking:exceeds_total_steps" in errors
    assert "uncertainty:exceeds_total_steps" in errors

    # A count that fits is still reported, ratio and all.
    ok, errors, _ = derive_reasoning_metrics(
        _raw_metrics(), generation_like=True, selection_like=False, option_count=0
    )
    assert ok["reasoning_backtracking"] == 1.0
    assert errors == []


def test_a_shapeless_task_reports_why_density_has_no_normalizer():
    _metrics, errors, _inapplicable = derive_reasoning_metrics(
        _raw_metrics(), generation_like=False, selection_like=False, option_count=0
    )
    assert "density:no_applicable_normalizer_for_task_shape" in errors


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ('{"backtracking": 2}', 2),
        ('```json\n{"backtracking": 2}\n```', 2),
        ('Return exactly:\n{"backtracking": <integer>}\nAnswer: {"backtracking": 3}', 3),
        ('The chain has a stray { brace, then {"backtracking": 4}', 4),
    ],
)
def test_a_verdict_is_read_out_of_a_talkative_judge(reply, expected):
    template = SimpleNamespace(output_contract={"json_fields": {"backtracking": "int"}})
    assert ReasoningJudgeStage._parse_json(reply, template) == {"backtracking": expected}


def test_a_judge_without_a_batch_route_still_scores_every_sample(
    fake_server, write_run_config, fake_dataset
):
    """chat_single answers one conversation, so a group of 8 must not be sent."""
    counts: Counter = Counter()
    fake_server.state.responder = _judge_responder(counts)
    config_path = write_run_config(
        base_url=fake_server.base_url,
        datasets=[fake_dataset(sample_size=4, n=4)],
        models=[
            {
                "id": "fake-model",
                "model_name": "test/model",
                "endpoint": {
                    "base_url": fake_server.base_url,
                    "api_key": "test-key",
                    "batch": {"enabled": False},
                },
                "sampling": {
                    "max_tokens_default": 64,
                    "max_tokens_cap": 512,
                    "max_tokens_floor": 16,
                },
                "limits": {"max_parallel_batches": 2, "context_window": 4096},
            }
        ],
        modes={"prompt_modes": ["cot"], "repeats": 1},
        engine={
            "reasoning_judge": {"enabled": True, "model": "fake-model", "group_size": 8}
        },
    )
    result = asyncio.run(EvaluationEngine(load_run_config(config_path)).run())

    records = dedupe_records(load_records(result.tasks[0].output_dir / "records.jsonl"))
    assert len(records) == 4
    for record in records:
        assert record["details"]["reasoning_metrics_status"] == "ok"
        assert record["metrics"]["reasoning_observation_coverage"] == pytest.approx(0.75)


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
