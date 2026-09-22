"""A judge documented as strictly binary must record a strictly binary score.

The binary templates say "reply 1 or 0" and match the score line with a
`[01]` pattern. That pattern was right; what followed it was not. When it did
not match, `expect_numeric_score` fell back to the first number found
ANYWHERE in the reply, and whatever that was became the score -- on metrics
whose whole meaning is 0 or 1, feeding means and rates that are only
interpretable in [0, 1].

Measured against the shipped `judge_binary_v1` before the fix:

    "Score: 2"            -> 2.0
    "Score: 7"            -> 7.0
    "Score: -1"           -> -1.0, and label "1", so `positive` was True
    "Closest: 3"          -> 3.0   (the proxy templates' own detail line)
    "Score: 0.7"          -> 0.0   (the `[01]` pattern matched the "0")

The last two are the instructive ones. `Closest:` is a field those templates
ask for themselves, so a well-formed reply could poison its own score. And
0.7 is a judge declining to answer in binary being written down as a
confident "wrong".

The rule now: a score must satisfy the contract it was parsed under, and a
reply that fails it is UNPARSED, not zero. `apply_judged_metric` then leaves
the metric off the sample rather than recording 0, because
`aggregate_mean_metrics` skips absent keys instead of averaging them down --
"not judged" and "judged wrong" are different results.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from abductionbench.adapters._base import apply_judged_metric
from abductionbench.core.judge import JudgeStage
from abductionbench.core.prompts import PromptRegistry
from abductionbench.core.types import SampleScore

BINARY_TEMPLATES = (
    "judge_binary_v1",
    "judge_binary_plausibility_v1",
    "proxy_closest_explanation_v1",
    "proxy_closest_hypothesis_v1",
    "proxy_hypothesis_quality_v1",
)


def _stage(template_id: str) -> JudgeStage:
    stage = object.__new__(JudgeStage)
    stage.template = PromptRegistry([Path("configs/prompts")]).get(template_id)
    return stage


@pytest.mark.parametrize("template_id", BINARY_TEMPLATES)
@pytest.mark.parametrize("reply", ["Score: 2", "Score: 7", "Score: 42", "Score: -1"])
def test_an_out_of_range_number_is_not_a_score(template_id, reply):
    verdict = _stage(template_id)._parse(reply)
    assert verdict.score is None, f"{reply!r} became {verdict.score}"
    assert verdict.parsed is False


@pytest.mark.parametrize("template_id", BINARY_TEMPLATES)
def test_a_fractional_score_is_rejected_rather_than_rounded_down(template_id):
    """"Score: 0.7" is a judge answering the wrong question, not a 0."""
    verdict = _stage(template_id)._parse("Score: 0.7")
    assert verdict.score is None
    assert verdict.parsed is False


@pytest.mark.parametrize("template_id", BINARY_TEMPLATES)
def test_the_templates_own_detail_line_cannot_become_the_score(template_id):
    """`Closest: 3` is a field these prompts ask for themselves."""
    assert _stage(template_id)._parse("Closest: 3").score is None


def test_a_valid_reply_that_also_carries_a_detail_line_still_scores():
    """The fix must not reject well-formed replies -- that is the whole point."""
    verdict = _stage("proxy_closest_hypothesis_v1")._parse(
        "Score: 1\nClosest: 3\nDirection: matches"
    )
    assert verdict.score == 1.0
    assert verdict.parsed is True
    assert verdict.details.get("closest_reference") == "3"


@pytest.mark.parametrize("template_id", BINARY_TEMPLATES)
@pytest.mark.parametrize(("reply", "expected"), [("Score: 1", 1.0), ("Score: 0", 0.0)])
def test_the_two_admissible_values_are_untouched(template_id, reply, expected):
    verdict = _stage(template_id)._parse(reply)
    assert verdict.score == expected
    assert verdict.parsed is True


def test_a_rejected_score_does_not_leave_an_affirmative_label_behind():
    """One reply, one verdict.

    The label fallback scrapes the `labels` list out of the raw text. Against
    "Score: -1" it found the "1" and reported an affirmative -- from the very
    line whose score had just been rejected -- and `positive` reads the label
    when the score is None, so the sample scored as correct.
    """
    verdict = _stage("judge_binary_v1")._parse("Score: -1")
    assert verdict.label is None
    assert verdict.positive is False


def test_yes_and_no_still_parse_through_the_label_path():
    """The label path is legitimate on its own; only the poisoned one is cut."""
    for reply, positive in (("yes", True), ("no", False)):
        verdict = _stage("judge_binary_v1")._parse(reply)
        assert verdict.parsed is True
        assert verdict.positive is positive


def test_every_binary_template_declares_its_admissible_values():
    """Enforcement needs the contract to say what it admits."""
    registry = PromptRegistry([Path("configs/prompts")])
    for template_id in BINARY_TEMPLATES:
        contract = registry.get(template_id).output_contract
        assert contract.get("score_values") == [0, 1], template_id


def test_an_unparsed_verdict_leaves_the_metric_absent_rather_than_zero():
    """"Not judged" must not be recorded as "judged wrong"."""
    verdict = _stage("judge_binary_v1")._parse("Score: 2")
    seeded = SampleScore(metrics={"hypothesis_judged": 0.0}, prediction="x")
    out = apply_judged_metric(seeded, verdict, "hypothesis_judged")
    assert "hypothesis_judged" not in out.metrics
    assert out.details["judge_unparsed"] is True


def test_strata_of_an_unparsed_metric_go_too():
    """A stratum is the same verdict through a filter, so it is equally absent."""
    verdict = _stage("judge_binary_v1")._parse("nothing numeric here")
    seeded = SampleScore(
        metrics={"hypothesis_judged": 0.0, "hypothesis_judged_chemistry": 0.0, "n_items": 3.0},
        prediction="x",
    )
    out = apply_judged_metric(seeded, verdict, "hypothesis_judged")
    assert "hypothesis_judged" not in out.metrics
    assert "hypothesis_judged_chemistry" not in out.metrics
    # Unrelated metrics survive.
    assert out.metrics["n_items"] == 3.0


def test_a_parsed_zero_is_still_recorded_as_zero():
    """The fix must not turn real negative verdicts into missing data."""
    verdict = _stage("judge_binary_v1")._parse("Score: 0")
    out = apply_judged_metric(
        SampleScore(metrics={"hypothesis_judged": 0.0}, prediction="x"),
        verdict,
        "hypothesis_judged",
    )
    assert out.metrics["hypothesis_judged"] == 0.0
    assert "judge_unparsed" not in out.details
