"""Metrics a reader divides must be averaged over the same samples.

A mean is comparable with another mean only when both were taken over the same
rows. These columns are not read one at a time -- someone divides one by
another, or checks a ratio column against the two it came from -- and each was
averaged over whatever subset happened to carry it.

Measured on one abd cot task's 300 records before the fix:

    reasoning_observations_total   148 samples
    reasoning_observations_used     58
    reasoning_total_steps           56
    reasoning_useful_steps          40

so the sheet reported used/total = 18.69/89.99 = 0.208 while
`reasoning_observation_coverage` -- the same ratio, taken per sample and then
averaged -- said 0.346. Both were arithmetically correct and they described
different populations.
"""

from __future__ import annotations

from abductionbench.core.metrics import CO_AVERAGED_METRIC_GROUPS, aggregate_mean_metrics

OBSERVATION = {
    "reasoning_observations_total",
    "reasoning_observations_used",
    "reasoning_observation_coverage",
}
STEPS = {
    "reasoning_total_steps",
    "reasoning_useful_steps",
    "reasoning_useless_steps",
    "reasoning_useful_step_fraction",
    "reasoning_useless_step_fraction",
}


def test_a_sample_missing_one_of_a_group_sits_out_the_whole_group():
    rows = [
        {
            "reasoning_observations_total": 10.0,
            "reasoning_observations_used": 5.0,
            "reasoning_observation_coverage": 0.5,
        },
        # The judge produced an inventory for this one and nothing else, which
        # is exactly the shape that skewed the denominators.
        {"reasoning_observations_total": 90.0},
    ]
    out = aggregate_mean_metrics(rows)
    assert out["reasoning_observations_total"] == 10.0
    assert out["reasoning_observations_used"] == 5.0
    assert out["reasoning_observation_coverage"] == 0.5


def test_the_ratio_of_the_two_means_now_matches_the_averaged_ratio():
    """The property the whole fix exists for."""
    rows = [
        {
            "reasoning_observations_total": 10.0,
            "reasoning_observations_used": 5.0,
            "reasoning_observation_coverage": 0.5,
        },
        {
            "reasoning_observations_total": 20.0,
            "reasoning_observations_used": 5.0,
            "reasoning_observation_coverage": 0.25,
        },
        {"reasoning_observations_total": 900.0},
    ]
    out = aggregate_mean_metrics(rows)
    ratio_of_means = out["reasoning_observations_used"] / out["reasoning_observations_total"]
    assert abs(ratio_of_means - 1 / 3) < 1e-12
    assert out["reasoning_observation_coverage"] == 0.375  # mean of 0.5 and 0.25


def test_a_sample_carrying_none_of_a_group_is_untouched():
    """An io task has no reasoning metrics at all; that is not what this is for."""
    rows = [
        {"accuracy": 1.0},
        {"accuracy": 0.0, "reasoning_total_steps": 4.0},  # partial: steps sits out
    ]
    out = aggregate_mean_metrics(rows)
    assert out["accuracy"] == 0.5
    assert "reasoning_total_steps" not in out


def test_an_unrelated_metric_on_the_same_sample_still_averages():
    """Exclusion is per group, not per sample.

    A sample whose observation metrics are incomplete but whose step metrics
    are whole keeps its step metrics -- otherwise one missing judge call would
    cost the sample every column it has.
    """
    rows = [
        {
            "reasoning_observations_total": 10.0,   # partial observation group
            "reasoning_total_steps": 4.0,
            "reasoning_useful_steps": 1.0,
            "reasoning_useless_steps": 3.0,
            "reasoning_useful_step_fraction": 0.25,
            "reasoning_useless_step_fraction": 0.75,
            "reasoning_directionality": 1.0,
        },
    ]
    out = aggregate_mean_metrics(rows)
    assert "reasoning_observations_total" not in out
    assert out["reasoning_total_steps"] == 4.0
    assert out["reasoning_directionality"] == 1.0


def test_a_complete_group_is_never_excluded():
    rows = [dict.fromkeys(STEPS, 1.0)]
    out = aggregate_mean_metrics(rows)
    assert set(STEPS) <= set(out)


def test_the_groups_are_the_documented_ones():
    """Narrow on purpose: every exclusion throws a measurement away."""
    assert set(CO_AVERAGED_METRIC_GROUPS) == {frozenset(OBSERVATION), frozenset(STEPS)}
    # Metrics whose normalized form is computed per sample never span two
    # averages, so binding them would cost rows for nothing.
    flat = set().union(*CO_AVERAGED_METRIC_GROUPS)
    for independent in (
        "reasoning_prior_knowledge_normalized",
        "reasoning_uncertainty_rate",
        "reasoning_directionality",
    ):
        assert independent not in flat
