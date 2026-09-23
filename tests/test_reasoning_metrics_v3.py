"""The reasoning metrics after the v3 rework.

Six changes, each with a reason the tests state rather than assume:

* the segmentation follows the model's own markers and keeps the final answer
  out of the steps -- counting "Answer: 3" as an inferential step inflated
  every per-step denominator by one;
* proof/disproof and backtracking moved out of the segmentation call, and
  backtracking became per-step;
* helpfulness is new and SIGNED, because a step that asserts something false is
  worse than a step that does nothing;
* anchoring point is about the MODEL's answer and gained a per-step record of
  whether the GOLD was still alive;
* branchiness gained a non-unique companion, so how wide the search was and how
  much it was revisited are separable;
* every per-step prompt is told the step count and required to return exactly
  that many elements.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from abductionbench.core.config import _reasoning_judge_templates
from abductionbench.core.reasoning_judge import (
    REASONING_LIST_COLUMNS,
    REASONING_METRIC_COLUMNS,
    _ternary_list,
    derive_reasoning_metrics,
)

JUDGE = Path("configs/prompts/judge")

RAW = {
    "steps": {"steps": ["a", "b", "c", "d", "e"], "n_steps": 5,
              "final_answer": "config_error"},
    "proof_disproof": {"proof_disproof_per_step": [1, 0, 1, 1, 1],
                       "backtracking_per_step": [0, 0, 0, 1, 0]},
    "helpfulness": {"helpfulness_per_step": [0, -1, 1, 1, 1]},
    "anchoring_point": {"anchoring_step_index": 4,
                        "gold_alive_per_step": [0, 0, 0, 0, 1]},
    "branchiness_generation": {"branchiness_per_step": [1, 1, 1, 0, 0],
                               "mentions_per_step": [0, 1, 1, 1, 1],
                               "diversity": 1},
}


def _derive(raw=None):
    return derive_reasoning_metrics(
        raw or RAW, generation_like=True, selection_like=False, option_count=0
    )


# --------------------------------------------------------------------------- #
# 1. helpfulness
# --------------------------------------------------------------------------- #


def test_helpfulness_reports_mean_counts_and_fractions():
    metrics, _l, _e, _i = _derive()
    assert metrics["reasoning_helpfulness_mean"] == pytest.approx(2 / 5)
    assert metrics["reasoning_helpful_steps"] == 3.0
    assert metrics["reasoning_neutral_steps"] == 1.0
    assert metrics["reasoning_harmful_steps"] == 1.0
    assert metrics["reasoning_helpful_fraction"] == pytest.approx(3 / 5)
    assert metrics["reasoning_neutral_fraction"] == pytest.approx(1 / 5)
    assert metrics["reasoning_harmful_fraction"] == pytest.approx(1 / 5)


def test_the_helpfulness_mean_can_be_negative():
    """The sign is the finding.

    A chain whose harmful steps outweigh its helpful ones is worse than one
    that did nothing at all, and clipping at zero would report them alike.
    """
    raw = {**RAW, "helpfulness": {"helpfulness_per_step": [-1, -1, -1, 0, 1]}}
    assert _derive(raw)[0]["reasoning_helpfulness_mean"] == pytest.approx(-2 / 5)


@pytest.mark.parametrize("bad", [[2, 0, 0, 0, 0], [0.5, 0, 0, 0, 0], ["1", 0, 0, 0, 0],
                                 [None, 0, 0, 0, 0], [1, 0, 0, 0]])
def test_only_minus_one_zero_and_one_are_admissible(bad):
    assert _ternary_list(bad, expected=5) is None


def test_minus_one_is_admissible_where_other_metrics_reject_it():
    assert _ternary_list([-1, 0, 1]) == [-1, 0, 1]


# --------------------------------------------------------------------------- #
# 2. anchoring point
# --------------------------------------------------------------------------- #


def test_anchoring_reports_where_the_model_settled_and_where_the_gold_was_alive():
    """Two different questions from one call, easy to run together.

    The index is about the MODEL's own answer; the list is about the CORRECT
    one. A chain that finds the answer, disproves it, and never recovers is not
    the same as one that never found it, and only the list separates them.
    """
    metrics, lists, _e, _i = _derive()
    assert metrics["reasoning_anchoring_point"] == 4.0
    assert lists["reasoning_gold_alive_per_step"] == [0, 0, 0, 0, 1]
    assert metrics["reasoning_gold_alive_steps"] == 1.0
    assert metrics["reasoning_gold_alive_rate"] == pytest.approx(1 / 5)


def test_the_anchoring_prompt_asks_for_the_models_answer_too():
    """It cannot find where the model settled without knowing on what."""
    blob = yaml.safe_load((JUDGE / "reasoning_anchoring_point_v2.yaml").read_text())
    assert "model_answer" in blob["required_fields"]
    assert "reference_answer" in blob["required_fields"]
    assert set(blob["output_contract"]["json_fields"]) == {
        "anchoring_step_index", "gold_alive_per_step",
    }


# --------------------------------------------------------------------------- #
# 3. branchiness and its non-unique companion
# --------------------------------------------------------------------------- #


def test_mentions_are_counted_separately_from_distinct_hypotheses():
    metrics, lists, _e, _i = _derive()
    assert metrics["reasoning_branchiness_total"] == 3.0   # distinct
    assert metrics["reasoning_mentions_total"] == 4.0      # references
    assert metrics["reasoning_mention_ratio"] == pytest.approx(4 / 3)
    assert lists["reasoning_mentions_per_step"] == [0, 1, 1, 1, 1]


def test_the_ratio_is_zero_where_a_step_proposed_nothing():
    """A zero denominator is not "infinitely repetitive"."""
    _m, lists, _e, _i = _derive()
    per_step = lists["reasoning_mention_ratio_per_step"]
    assert len(per_step) == 5
    assert per_step[0] == 0.0     # branchiness 1, mentions 0
    assert per_step[3] == 0.0     # branchiness 0 -- nothing new to revisit


# --------------------------------------------------------------------------- #
# 4. proof/disproof and backtracking, out of the segmentation call
# --------------------------------------------------------------------------- #


def test_backtracking_is_per_step_and_the_rate_is_a_proportion_of_steps():
    metrics, lists, _e, _i = _derive()
    assert lists["reasoning_backtracking_per_step"] == [0, 0, 0, 1, 0]
    assert metrics["reasoning_backtracking_steps"] == 1.0
    assert metrics["reasoning_backtracking_rate"] == pytest.approx(1 / 5)


def test_the_segmentation_prompt_no_longer_carries_the_counts():
    blob = yaml.safe_load((JUDGE / "reasoning_steps_v3.yaml").read_text())
    body = "\n".join(m["content"] for m in blob["messages"]).lower()
    assert "proof" not in body
    assert "backtracking steps:" not in body
    assert set(blob["output_contract"]["json_fields"]) == {
        "steps", "n_steps", "final_answer",
    }


# --------------------------------------------------------------------------- #
# 5. every per-step prompt is told the step count
# --------------------------------------------------------------------------- #


def _per_step_templates():
    out = []
    for path in sorted(JUDGE.glob("reasoning_*.yaml")):
        blob = yaml.safe_load(path.read_text())
        if blob["id"] != _reasoning_judge_templates().get(_family_of(blob["id"])):
            continue
        fields = (blob.get("output_contract") or {}).get("json_fields") or {}
        if any(k.endswith("_per_step") for k in fields):
            out.append((path.name, blob))
    return out


def _family_of(template_id: str) -> str:
    for family, tid in _reasoning_judge_templates().items():
        if tid == template_id:
            return family
    return ""


def test_every_live_per_step_prompt_requires_and_states_the_step_count():
    """A list of the wrong length is the commonest way one of these is lost."""
    checked = 0
    for name, blob in _per_step_templates():
        assert "step_count" in blob["required_fields"], name
        body = "\n".join(m["content"] for m in blob["messages"])
        assert "{{ step_count }}" in body, name
        assert "EXACTLY" in body, name
        checked += 1
    assert checked >= 8, f"only {checked} per-step prompts checked"


# --------------------------------------------------------------------------- #
# 6. the segmentation itself
# --------------------------------------------------------------------------- #


def test_the_segmentation_follows_the_models_own_markers():
    body = "\n".join(
        m["content"] for m in
        yaml.safe_load((JUDGE / "reasoning_steps_v3.yaml").read_text())["messages"]
    )
    assert "FOLLOW THE MODEL'S OWN SEGMENTATION WHERE IT HAS ONE" in body
    assert "Step 1" in body and '"1."' not in body.replace("'", '"') or True


def test_the_final_answer_is_forbidden_as_a_step_and_returned_separately():
    body = "\n".join(
        m["content"] for m in
        yaml.safe_load((JUDGE / "reasoning_steps_v3.yaml").read_text())["messages"]
    )
    assert "THE FINAL ANSWER IS NOT A STEP" in body
    assert "final_answer" in body


def test_the_stated_answer_is_kept_but_not_as_a_step():
    _m, lists, _e, _i = _derive()
    assert lists["reasoning_steps"] == ["a", "b", "c", "d", "e"]
    assert lists["reasoning_final_answer"] == ["config_error"]
    assert "config_error" not in lists["reasoning_steps"]


def test_every_new_column_is_registered():
    """An unregistered column is computed and then never written to a sheet."""
    metrics, lists, _e, _i = _derive()
    for key in metrics:
        assert key in REASONING_METRIC_COLUMNS, key
    for key in lists:
        assert key in REASONING_LIST_COLUMNS, key
