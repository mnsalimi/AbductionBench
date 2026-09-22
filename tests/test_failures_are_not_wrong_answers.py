"""A model that produced nothing readable is not a model that was wrong.

`value` used to be the adapter's mean over every scored sample, and a sample
the adapter could not parse scores 0 -- so an empty reply, a truncated one and
a confidently incorrect answer were the same number. They are different
results: one says the model is wrong about the task, the other says it did not
produce something the harness could read, which is as often a token-budget or
format problem as a reasoning one.

`value` is now "how often right WHEN IT ANSWERED" and `value_strict` is "how
often right over everything planned", with failures and unscored samples alike
counted as 0. The gap between the two is what was being hidden.

"I don't know" is not a failure. It parses, so it is an answer, and a wrong
one -- `parse_ok` is the adapter's own judgement about whether it could read
the output, which is where that line belongs.
"""

from __future__ import annotations

from pathlib import Path

from abductionbench.core.engine import EvaluationEngine, TaskResult
from abductionbench.core.types import SampleScore


class _Adapter:
    dataset_id = "d"
    objective_metrics = True
    higher_is_better = True

    def aggregate(self, scores):
        values = [s.metrics.get("accuracy", 0.0) for s in scores]
        return {"accuracy": sum(values) / len(values)} if values else {}


def _aggregate(scores, *, planned):
    engine = object.__new__(EvaluationEngine)
    result = TaskResult(identity=None, output_dir=Path("."))
    result.n_planned = planned
    result.n_scored = len(scores)
    result.primary_metric = "accuracy"
    return EvaluationEngine._aggregate(
        engine, _Adapter(), scores, result, fresh=[], reused_records=[]
    )


def _score(*, correct: bool, parsed: bool = True) -> SampleScore:
    return SampleScore(
        metrics={"accuracy": 1.0 if correct else 0.0}, prediction="x", parse_ok=parsed
    )


def test_a_failure_does_not_count_as_a_wrong_answer_in_value():
    """Two right, one wrong, one unreadable: 2/3 answered, not 2/4."""
    scores = [
        _score(correct=True),
        _score(correct=True),
        _score(correct=False),
        _score(correct=False, parsed=False),
    ]
    metrics = _aggregate(scores, planned=4)
    assert metrics["accuracy"] == 2 / 3
    assert metrics["n_answered"] == 3.0
    assert metrics["n_parse_failures"] == 1.0
    assert metrics["parse_failure_rate"] == 0.25


def test_value_strict_counts_a_failure_as_zero():
    """Strict is over everything planned, so the failure is charged."""
    scores = [
        _score(correct=True),
        _score(correct=True),
        _score(correct=False),
        _score(correct=False, parsed=False),
    ]
    metrics = _aggregate(scores, planned=4)
    assert metrics["accuracy_strict"] == 2 / 4


def test_strict_also_charges_samples_that_were_never_scored():
    """Four planned, three scored, one of those unreadable."""
    scores = [_score(correct=True), _score(correct=True), _score(correct=False, parsed=False)]
    metrics = _aggregate(scores, planned=4)
    assert metrics["accuracy"] == 1.0          # both answers it gave were right
    assert metrics["accuracy_strict"] == 2 / 4  # but two of four planned produced nothing


def test_an_answer_that_says_i_dont_know_is_wrong_not_failed():
    """It parses, so it is an answer -- and it counts against `value`."""
    scores = [_score(correct=True), _score(correct=False)]  # the second parsed fine
    metrics = _aggregate(scores, planned=2)
    assert metrics["n_parse_failures"] == 0.0
    assert metrics["accuracy"] == 0.5
    assert metrics["accuracy_strict"] == 0.5


def test_a_task_with_no_failures_is_unchanged():
    """The fix must not move numbers that were already right."""
    scores = [_score(correct=True), _score(correct=False)]
    metrics = _aggregate(scores, planned=2)
    assert metrics["accuracy"] == 0.5
    assert metrics["accuracy_strict"] == 0.5
    assert metrics["parse_failure_rate"] == 0.0


def test_a_derived_primary_with_no_per_sample_value_keeps_the_old_strict_rule():
    """Some primaries are computed in aggregate() and have no per-sample form.

    Re-meaning a value that does not exist would report a different metric
    under the same name, so those keep `primary x coverage`.
    """

    class _Derived(_Adapter):
        def aggregate(self, scores):
            return {"joint": 0.5}

    engine = object.__new__(EvaluationEngine)
    result = TaskResult(identity=None, output_dir=Path("."))
    result.n_planned, result.n_scored, result.primary_metric = 4, 2, "joint"
    metrics = EvaluationEngine._aggregate(
        engine, _Derived(), [_score(correct=True), _score(correct=False, parsed=False)],
        result, fresh=[], reused_records=[],
    )
    assert metrics["joint_strict"] == 0.5 * (2 / 4)
