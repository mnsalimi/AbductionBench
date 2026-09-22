"""Four ways a score could be wrong that passing tests did not catch.

Each was reported by an external audit, reproduced against this checkout
before anything was changed, and is pinned here by the property that was
violated rather than by the code that violated it.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from abductionbench.core.engine import EvaluationEngine, TaskResult
from abductionbench.core.prompts import PromptRegistry
from abductionbench.core.reasoning_judge import ReasoningJudgeStage

# --------------------------------------------------------------------------- #
# an invalid reasoning verdict must not be cached as a good one
# --------------------------------------------------------------------------- #


def _steps_template():
    return PromptRegistry([Path("configs/prompts")]).get("reasoning_steps_v2")


@pytest.mark.parametrize(
    "reply",
    [
        '{"steps": 42, "proof_disproof_counts": [], "backtracking_steps": 0}',
        '{"steps": "not a list", "proof_disproof_counts": [], "backtracking_steps": 0}',
        '{"steps": null, "proof_disproof_counts": null, "backtracking_steps": null}',
        '{"steps": [], "proof_disproof_counts": [], "backtracking_steps": 0}',
        '{"steps": ["a"], "proof_disproof_counts": [-1], "backtracking_steps": 0}',
        '{"steps": ["a"], "proof_disproof_counts": ["x"], "backtracking_steps": 0}',
        '{"steps": ["a"], "proof_disproof_counts": [0], "backtracking_steps": -2}',
    ],
)
def test_a_reply_that_violates_the_declared_types_is_not_a_parse(reply):
    """`steps` is declared `list_of_strings`; 42 is not one.

    This matters because of what happens next rather than what happens here.
    `_judge_many` writes the cache only for replies that parsed, so a reply
    accepted here is cached as a verdict -- and the derivation downstream then
    rejects it, correctly, leaving the metric missing. The bad value is now
    permanent: re-running the reasoning judge serves it back without calling
    the judge at all, so a rejudge pass cannot repair what it was run for.
    """
    assert ReasoningJudgeStage._parse_json(reply, _steps_template()) is None


@pytest.mark.parametrize(
    "reply",
    [
        '{"steps": ["a", "b"], "proof_disproof_counts": [1, 2], "backtracking_steps": 0}',
        '{"steps": ["a"], "proof_disproof_counts": [0], "backtracking_steps": 2.0}',
    ],
)
def test_a_well_formed_reply_still_parses(reply):
    """Including an integral float, which judges do return for an integer."""
    assert ReasoningJudgeStage._parse_json(reply, _steps_template()) is not None


def test_an_unknown_declared_type_is_treated_as_unconstrained():
    """A gap in the validator map is not evidence about the judge's reply.

    Failing closed would silently discard every verdict from a newly added
    template whose type name this map has not learnt.
    """
    from abductionbench.core.reasoning_judge import _satisfies_contract

    assert _satisfies_contract("anything", "a_type_nobody_has_written_yet") is None


def test_every_type_the_shipped_templates_declare_is_actually_validated():
    """The map has to cover what the prompts say, or the check is decorative."""
    import re

    from abductionbench.core.reasoning_judge import _CONTRACT_VALIDATORS

    declared: set[str] = set()
    for path in Path("configs/prompts/judge").glob("*.yaml"):
        body = path.read_text()
        block = re.search(r"json_fields:\n((?:\s+\S+:\s*\S+\n)+)", body)
        if block:
            declared |= set(re.findall(r":\s*([a-z_]+)\s*$", block.group(1), re.MULTILINE))
    missing = declared - set(_CONTRACT_VALIDATORS)
    assert not missing, f"declared but never validated: {sorted(missing)}"


# --------------------------------------------------------------------------- #
# grouped resume must reduce the reused members too
# --------------------------------------------------------------------------- #


def test_group_reduction_covers_reused_records():
    """An item asked as k requests is one evaluation item, on resume as well.

    The reduced record is written under the PARENT's sample id with a
    `reduced::<group_id>` fingerprint, and the parent is not one of the
    rendered prompts -- so resume matches the members, not the reduction. Folding
    only the fresh scores therefore counted one self-consistency item as one on
    the first pass and as k after a resume: coverage 1 -> 3, and a metric
    bounded at 1 reporting 3.
    """
    source = inspect.getsource(EvaluationEngine._run_task)
    assert "[*scores, *unskipped]" in source, "reduction no longer spans reused records"
    assert "reused_scores = []" in source, "reused scores would be counted twice"


def test_reduction_is_attempted_when_only_reused_records_exist():
    """A fully-resumed task has no fresh scores at all -- the worst case."""
    source = inspect.getsource(EvaluationEngine._run_task)
    assert "needs_group_reduction and (scores or reused_scores)" in source


# --------------------------------------------------------------------------- #
# a verdict belongs to the judge that gave it
# --------------------------------------------------------------------------- #


def test_both_judge_caches_key_on_the_judge_model():
    """Resuming with a changed judge must not serve the old judge's verdicts.

    The caches live in the run directory, so this bites a resume rather than a
    fresh run -- and a resume with a changed judge is exactly what someone does
    when they decide the first judge was not good enough.
    """
    from abductionbench.core import judge as answer_judge

    assert '"judge": self.config.model' in inspect.getsource(answer_judge.JudgeStage.apply)
    assert '"judge": self.config.model' in inspect.getsource(
        ReasoningJudgeStage._judge_many
    )


# --------------------------------------------------------------------------- #
# an exchange too big to judge is not a wrong answer
# --------------------------------------------------------------------------- #


def test_oversize_skips_are_carried_on_the_task_result():
    """Neither scored nor wrong, so it needs its own number."""
    assert TaskResult(identity=None, output_dir=Path(".")).n_judge_skipped == 0


def test_oversize_skips_are_surfaced_and_fail_a_judge_only_dataset():
    """They used to be counted, and then read by nothing.

    The sample keeps the 0.0 its score was seeded with, so "we could not look
    at this one" was reported as "the model got this one wrong" -- the same
    confusion an unreachable judge produced, which the `unavailable` check
    already exists to prevent.
    """
    source = inspect.getsource(EvaluationEngine._run_task)
    assert 'checkpoint.notes["judge_skipped_oversize"]' in source
    assert "judge.skipped_oversize and not adapter.objective_metrics" in source
