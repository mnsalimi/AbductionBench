"""The reasoning-metric judge: what it measures, and what it refuses to."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from abductionbench.core.reasoning_judge import (
    REASONING_METRIC_COLUMNS,
    ReasoningJudgeStage,
    comparison_combinations,
    derive_reasoning_metrics,
)

JUDGE_PROMPTS = Path("configs/prompts/judge")


def _raw(**overrides):
    """A complete, self-consistent set of judge outputs."""
    base = {
        "observation_inventory": {"total_observations": 8},
        "observation_coverage": {"total_observations": 8, "observations_used": 6},
        "steps": {
            "total_steps": 10,
            "useful_steps": 7,
            "useless_steps": 3,
            "backtracking_steps": 2,
        },
        "branchiness_diversity": {"branchiness": 4, "diversity": 1},
        "redundancy_completeness": {"redundancy": 2, "completeness": 5},
        "directionality": {"directionality": 0.5},
        "differential_elimination": {"differential_elimination": 3},
        "uncertainty": {"uncertainty_steps": 3},
        "prior_knowledge": {"prior_knowledge": 1},
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
# the numbers themselves
# --------------------------------------------------------------------------- #


def test_every_raw_and_derived_value_is_reported():
    metrics, errors, inapplicable = derive_reasoning_metrics(
        _raw(), generation_like=False, selection_like=True, option_count=3
    )
    assert not errors
    assert not inapplicable or inapplicable == ["branchiness_diversity:generation_only"]
    # metric 1
    assert metrics["reasoning_observations_total"] == 8
    assert metrics["reasoning_observations_used"] == 6
    assert metrics["reasoning_observation_coverage"] == 6 / 8
    # metric 2: four raw counts and three derived values
    assert metrics["reasoning_total_steps"] == 10
    assert metrics["reasoning_useful_steps"] == 7
    assert metrics["reasoning_useless_steps"] == 3
    assert metrics["reasoning_backtracking_steps"] == 2
    assert metrics["reasoning_useful_step_fraction"] == 0.7
    assert metrics["reasoning_useless_step_fraction"] == 0.3
    assert metrics["reasoning_backtracking_rate"] == 0.2
    # metric 4 normalizes on metric 1's per-sample total, not on its own counts
    assert metrics["reasoning_redundancy_normalized"] == 2 / 8
    assert metrics["reasoning_completeness_normalized"] == 5 / 8
    # metric 5, 6, 7, 8
    assert metrics["reasoning_directionality"] == 0.5
    assert metrics["reasoning_differential_elimination"] == 3
    assert metrics["reasoning_differential_elimination_normalized"] == 3 / 4
    assert metrics["reasoning_uncertainty_steps"] == 3
    assert metrics["reasoning_uncertainty_rate"] == 0.3
    assert metrics["reasoning_prior_knowledge"] == 1


def test_the_combination_count_is_every_subset_of_two_or_more():
    # The specification's own worked example: C(3,2) + C(3,3) = 3 + 1 = 4.
    assert comparison_combinations(3) == 4
    assert comparison_combinations(2) == 1
    assert comparison_combinations(4) == 11
    assert comparison_combinations(1) == 0


def test_generation_gets_branchiness_and_selection_gets_elimination():
    generation, _e, gen_inapplicable = derive_reasoning_metrics(
        _raw(), generation_like=True, selection_like=False, option_count=0
    )
    assert "reasoning_branchiness" in generation
    assert "reasoning_diversity" in generation
    assert "reasoning_differential_elimination" not in generation
    assert gen_inapplicable == ["differential_elimination:selection_and_pipeline_only"]

    selection, _e, sel_inapplicable = derive_reasoning_metrics(
        _raw(), generation_like=False, selection_like=True, option_count=3
    )
    assert "reasoning_differential_elimination" in selection
    assert "reasoning_branchiness" not in selection
    assert sel_inapplicable == ["branchiness_diversity:generation_only"]


def test_a_pipeline_task_gets_both_generation_and_selection_families():
    """A benchmark that generates and selects in one task does both things."""
    metrics, errors, inapplicable = derive_reasoning_metrics(
        _raw(),
        generation_like=False,
        selection_like=False,
        pipeline_like=True,
        option_count=3,
    )
    assert not errors
    assert not inapplicable
    assert "reasoning_branchiness" in metrics
    assert "reasoning_differential_elimination" in metrics


# --------------------------------------------------------------------------- #
# what it refuses: an impossible count is dropped, never coerced
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "gone", "error"),
    [
        # More backtracks than steps is a judge that did not count. One reply of
        # "147" against a 14-step chain moved this metric's mean fifty-fold.
        (
            _raw(steps={"total_steps": 4, "useful_steps": 2, "useless_steps": 2,
                        "backtracking_steps": 9}),
            "reasoning_backtracking_steps",
            "steps:backtracking_exceeds_total_steps",
        ),
        (
            _raw(observation_coverage={"total_observations": 8, "observations_used": 99}),
            "reasoning_observations_used",
            "observation_coverage:used_exceeds_inventory_total",
        ),
        (
            _raw(steps={"total_steps": 10, "useful_steps": 7, "useless_steps": 9,
                        "backtracking_steps": 1}),
            "reasoning_total_steps",
            "steps:invalid_counts_or_sum",
        ),
        (
            _raw(uncertainty={"uncertainty_steps": 40}),
            "reasoning_uncertainty_steps",
            "uncertainty:exceeds_total_steps",
        ),
        (
            _raw(directionality={"directionality": 0.7}),
            "reasoning_directionality",
            "directionality:expected_0_0.5_or_1",
        ),
        (
            _raw(redundancy_completeness={"redundancy": 20, "completeness": 1}),
            "reasoning_redundancy_normalized",
            "redundancy_completeness:count_exceeds_inventory_total",
        ),
        (
            _raw(prior_knowledge={"prior_knowledge": 7}),
            "reasoning_prior_knowledge",
            "prior_knowledge:invalid_or_missing_output",
        ),
    ],
)
def test_an_impossible_count_is_dropped_with_a_reason(raw, gone, error):
    metrics, errors, _inapplicable = derive_reasoning_metrics(
        raw, generation_like=True, selection_like=False, option_count=0
    )
    assert gone not in metrics
    assert error in errors


def test_more_comparisons_than_combinations_is_rejected():
    metrics, errors, _ = derive_reasoning_metrics(
        _raw(differential_elimination={"differential_elimination": 99}),
        generation_like=False,
        selection_like=True,
        option_count=3,
    )
    assert "reasoning_differential_elimination" not in metrics
    assert "differential_elimination:exceeds_possible_combinations" in errors


def test_a_zero_observation_inventory_blocks_the_ratios_but_keeps_the_counts():
    """Dividing by it would be a made-up number; the raw counts still stand."""
    metrics, errors, _ = derive_reasoning_metrics(
        _raw(
            observation_inventory={"total_observations": 0},
            observation_coverage={"total_observations": 0, "observations_used": 0},
            redundancy_completeness={"redundancy": 0, "completeness": 0},
        ),
        generation_like=True,
        selection_like=False,
        option_count=0,
    )
    assert metrics["reasoning_observations_total"] == 0
    assert metrics["reasoning_observations_used"] == 0
    assert "reasoning_observation_coverage" not in metrics
    assert "reasoning_redundancy_normalized" not in metrics
    assert "observation_coverage:inventory_total_is_zero" in errors
    assert "redundancy_completeness:inventory_total_is_zero" in errors


def test_a_missing_family_costs_only_its_own_metrics():
    metrics, errors, _ = derive_reasoning_metrics(
        {**_raw(), "steps": {}},
        generation_like=True,
        selection_like=False,
        option_count=0,
    )
    assert "reasoning_total_steps" not in metrics
    assert "reasoning_uncertainty_rate" not in metrics  # normalizer is gone
    assert metrics["reasoning_uncertainty_steps"] == 3  # the raw count survives
    assert metrics["reasoning_observation_coverage"] == 6 / 8  # untouched
    assert "steps:invalid_counts_or_sum" in errors


# --------------------------------------------------------------------------- #
# the judge prompts
# --------------------------------------------------------------------------- #


def _judge_prompt_files():
    return sorted(JUDGE_PROMPTS.glob("reasoning_*.yaml"))


def test_one_prompt_per_metric_family_ships():
    from abductionbench.core.config import ReasoningJudgeConfig

    ids = {path.stem for path in _judge_prompt_files()}
    assert set(ReasoningJudgeConfig().templates.values()) == ids
    assert len(ids) == 9


def test_no_judge_prompt_mentions_normalization():
    """Raw counts only: every ratio is computed in code, after the fact.

    A judge asked for a ratio has to do arithmetic on its own counts, and the
    sheet can then disagree with its own columns.
    """
    import re

    forbidden = re.compile(
        r"normali|\bdivide|\bdivided\b|\bratio\b|\bfraction\b|\bpercent|\bper step\b|÷",
        re.IGNORECASE,
    )
    offenders = []
    for path in _judge_prompt_files():
        for number, line in enumerate(path.read_text().splitlines(), 1):
            found = forbidden.search(line)
            if found:
                offenders.append((path.name, number, found.group(0)))
    assert not offenders, f"judge prompt mentions normalization: {offenders}"


def test_every_judge_prompt_asks_for_exactly_its_declared_fields():
    """The contract the parser enforces has to be the one the prompt asks for."""
    from abductionbench.core.config import load_yaml

    for path in _judge_prompt_files():
        blob = load_yaml(path)
        declared = set((blob.get("output_contract") or {}).get("json_fields") or {})
        assert declared, path.name
        body = "\n".join(message["content"] for message in blob["messages"])
        for name in declared:
            assert f'"{name}"' in body, (path.name, name)


def test_the_step_counts_are_one_prompt_and_not_four():
    """Four counts from four calls would be four different segmentations."""
    from abductionbench.core.config import load_yaml

    blob = load_yaml(JUDGE_PROMPTS / "reasoning_steps_v1.yaml")
    assert set((blob.get("output_contract") or {}).get("json_fields")) == {
        "total_steps",
        "useless_steps",
        "useful_steps",
        "backtracking_steps",
    }


# --------------------------------------------------------------------------- #
# parsing a judge that thinks out loud
# --------------------------------------------------------------------------- #


class _Template:
    ref = "t@1"
    output_contract = {"json_fields": {"total_steps": "nonnegative_integer"}}


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ('{"total_steps": 3}', 3),
        ('```json\n{"total_steps": 4}\n```', 4),
        ("Let me think. First... Therefore:\n{\"total_steps\": 5}", 5),
        # An echoed example followed by the real verdict: the last one wins.
        ('{"total_steps": <integer>} ... my answer: {"total_steps": 6}', 6),
        ('I counted {a} steps. {"total_steps": 7}', 7),
    ],
)
def test_the_verdict_is_read_out_of_whatever_the_judge_wrote_around_it(reply, expected):
    parsed = ReasoningJudgeStage._parse_json(reply, _Template())
    assert parsed is not None and parsed["total_steps"] == expected


def test_a_reply_without_the_declared_fields_is_unparsed():
    assert ReasoningJudgeStage._parse_json('{"something_else": 1}', _Template()) is None
    assert ReasoningJudgeStage._parse_json("no json at all", _Template()) is None


# --------------------------------------------------------------------------- #
# the stage as a whole
# --------------------------------------------------------------------------- #


def test_the_column_list_covers_every_metric_the_derivation_can_emit():
    metrics, _e, _i = derive_reasoning_metrics(
        _raw(), generation_like=True, selection_like=True, option_count=3
    )
    assert set(metrics) <= set(REASONING_METRIC_COLUMNS)
    assert len(REASONING_METRIC_COLUMNS) == len(set(REASONING_METRIC_COLUMNS))


def test_io_outputs_are_never_judged(tmp_path):
    """The guard that makes these columns mean what they say.

    An io output has no chain of reasoning in it, so a number measured over one
    would be a number about the answer line.
    """
    from abductionbench.core.config import ReasoningJudgeConfig
    from abductionbench.core.types import (
        ModelResponse,
        ResponseStatus,
        SampleScore,
        SampleSpec,
        TaskIdentity,
    )

    class _Boom:
        """Any call to the judge at all is the failure this test looks for."""

        supports_batch = False

        def chat_single(self, *args, **kwargs):  # pragma: no cover - must not run
            raise AssertionError("the reasoning judge called the model for an io output")

        chat_batch = chat_single

    stage = ReasoningJudgeStage(
        config=ReasoningJudgeConfig(enabled=True, model="m"),
        registry=_FakeRegistry(),
        renderer=None,
        clients={"m": _Boom()},
        retry_policy=None,
        cache_dir=tmp_path,
    )
    sample = SampleSpec(sample_id="s1", fields={"observation": "x"}, task_kind="generation")
    scored = [
        (
            sample,
            ModelResponse(sample_id="s1", model_id="m", status=ResponseStatus.OK,
                          content="Answer: something"),
            SampleScore(metrics={"match": 1.0}),
        )
    ]
    identity = TaskIdentity(
        run_id="r", dataset_id="d", model_id="m", template_id="t", template_version="1.0",
        prompt_mode="io", selection_mode="n/a", data_delivery_mode="static",
        task_kind="generation",
    )
    out = asyncio.run(stage.apply(_FakeAdapter(), identity, [], scored))
    assert out is scored
    assert not any(key.startswith("reasoning_") for key in out[0][2].metrics)


class _FakeRegistry:
    def get(self, template_id):
        class _T:
            ref = f"{template_id}@1.0"
            id = template_id
            version = "1.0"
            required_fields: list[str] = []
            output_contract: dict = {}

        return _T()


class _FakeAdapter:
    dataset_id = "d"

    def documentation(self):
        raise RuntimeError("not needed")
