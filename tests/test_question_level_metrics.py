"""Two requests that disagree, reported as two columns rather than one compromise.

How many observations a question supplies, and how many options it offers, are
properties of the QUESTION -- the observation inventory is even cached by the
exact question text and bought once for the whole run. So a difference between
two models' columns is not a finding about the models; it is a difference in
which samples each one got a verdict for.

One request was to exclude skipped samples, which keeps each model's figure
honest about what it was measured on. The other was to include every question
so the denominator does not move between models. Both are right about
something and they cannot both be one column, so there are two.
"""

from __future__ import annotations

import pandas as pd

from abductionbench.core.reporting import QUESTION_LEVEL_METRICS, _add_question_level_overall

KEYS = {
    "prompt_mode": "cot",
    "selection_mode": "n-a",
    "template_mode": "cot_n-a_static",
}


def _frame(rows):
    return pd.DataFrame([{**KEYS, **row} for row in rows])


def test_the_per_model_value_still_excludes_that_models_skipped_samples():
    """Unchanged: it is what every other column in the row is consistent with."""
    frame = _add_question_level_overall(
        _frame([
            {"dataset_id": "abd", "model_id": "m1",
             "metric.reasoning_observations_total": 10.0},
            {"dataset_id": "abd", "model_id": "m2",
             "metric.reasoning_observations_total": 20.0},
        ])
    )
    assert list(frame["metric.reasoning_observations_total"]) == [10.0, 20.0]


def test_the_overall_column_is_the_same_number_on_every_models_row():
    """Repeating it is the point: it makes the column visibly model-independent."""
    frame = _add_question_level_overall(
        _frame([
            {"dataset_id": "abd", "model_id": "m1",
             "metric.reasoning_observations_total": 10.0},
            {"dataset_id": "abd", "model_id": "m2",
             "metric.reasoning_observations_total": 20.0},
            {"dataset_id": "abd", "model_id": "m3",
             "metric.reasoning_observations_total": 30.0},
        ])
    )
    assert list(frame["metric.reasoning_observations_total_overall"]) == [20.0, 20.0, 20.0]


def test_three_models_give_four_values():
    """One per model, plus the overall -- which is what was asked for."""
    frame = _add_question_level_overall(
        _frame([
            {"dataset_id": "abd", "model_id": f"m{i}",
             "metric.reasoning_observations_total": value}
            for i, value in enumerate((6.0, 9.0, 12.0), start=1)
        ])
    )
    per_model = list(frame["metric.reasoning_observations_total"])
    overall = set(frame["metric.reasoning_observations_total_overall"])
    assert per_model == [6.0, 9.0, 12.0]
    assert overall == {9.0}


def test_a_model_with_no_value_still_sees_the_overall():
    """Its own cell stays blank; the shared figure does not become blank with it."""
    frame = _add_question_level_overall(
        _frame([
            {"dataset_id": "abd", "model_id": "m1",
             "metric.reasoning_observations_total": 10.0},
            {"dataset_id": "abd", "model_id": "m2",
             "metric.reasoning_observations_total": 20.0},
            {"dataset_id": "abd", "model_id": "m3",
             "metric.reasoning_observations_total": None},
        ])
    )
    assert pd.isna(frame["metric.reasoning_observations_total"].iloc[2])
    assert list(frame["metric.reasoning_observations_total_overall"]) == [15.0, 15.0, 15.0]


def test_datasets_do_not_pool_with_each_other():
    frame = _add_question_level_overall(
        _frame([
            {"dataset_id": "abd", "model_id": "m1",
             "metric.reasoning_observations_total": 10.0},
            {"dataset_id": "aer", "model_id": "m1",
             "metric.reasoning_observations_total": 50.0},
        ])
    )
    assert list(frame["metric.reasoning_observations_total_overall"]) == [10.0, 50.0]


def test_a_selection_variant_does_not_pool_with_a_generation_one():
    """Same dataset, different question set: it does not offer the same options."""
    rows = [
        {**KEYS, "dataset_id": "rb", "model_id": "m1",
         "template_mode": "io_SCS_selection_static", "selection_mode": "SCS",
         "metric.reasoning_option_count": 4.0},
        {**KEYS, "dataset_id": "rb", "model_id": "m1",
         "template_mode": "io_n-a_generation_static", "selection_mode": "n-a",
         "metric.reasoning_option_count": 0.0},
    ]
    frame = _add_question_level_overall(pd.DataFrame(rows))
    assert list(frame["metric.reasoning_option_count_overall"]) == [4.0, 0.0]


def test_both_question_level_metrics_are_covered():
    assert set(QUESTION_LEVEL_METRICS) == {
        "reasoning_observations_total",
        "reasoning_option_count",
    }


def test_a_frame_without_the_columns_is_left_alone():
    frame = _add_question_level_overall(_frame([{"dataset_id": "abd", "model_id": "m1"}]))
    assert not [c for c in frame.columns if c.endswith("_overall")]
