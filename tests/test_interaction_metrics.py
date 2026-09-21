"""Two metrics for interactive datasets: what an episode cost, and how much of
it did any work.

An interactive benchmark measures two things at once. `interaction_steps` is
what the model spent -- pure bookkeeping, counted by the engine that drove the
turns. `interaction_step_relevance_rate` is how much of that spending mattered,
which only a judge can say.

They belong together. Four relevant actions out of four is not the same result
as four out of nineteen, and reporting either number alone hides that.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from abductionbench.core.reasoning_judge import ReasoningJudgeStage
from abductionbench.core.types import (
    ModelResponse,
    RenderedPrompt,
    ResponseStatus,
    SampleScore,
    SampleSpec,
    SamplingParams,
    TaskIdentity,
)


def _sample(actions, final, sample_id="s1"):
    transcript = [
        {"role": "system", "content": "You investigate."},
        {"role": "user", "content": "A patient presents with syncope."},
    ]
    for action, result in actions:
        transcript.append({"role": "assistant", "content": action})
        transcript.append({"role": "user", "content": result})
    transcript.append({"role": "assistant", "content": final})
    return SampleSpec(
        sample_id=sample_id,
        fields={"observation": "A patient presents with syncope."},
        reference={"gold": "Lyme carditis"},
        metadata={"_transcript": transcript, "turns_used": len(actions) + 1},
    )


def _prompt(sample):
    return RenderedPrompt(
        sample=sample, messages=[], template_id="t", template_version="1",
        sampling=SamplingParams(max_tokens=16), input_tokens_est=1,
    )


# --------------------------------------------------------------------------- #
# metric one: the step count, no judge involved
# --------------------------------------------------------------------------- #


def test_interaction_steps_is_counted_for_interactive_datasets_only():
    """The engine drove the turns, so the engine counts them.

    Static and sequential deliveries have no episode. A column of 1.0s there
    would invite exactly the cross-dataset comparison that would be
    meaningless, so they get nothing rather than a placeholder.
    """
    from abductionbench.core.engine import EvaluationEngine

    class _Adapter:
        data_delivery_mode = "interactive"

    sample = _sample([("a", "r"), ("b", "r")], "final")
    score = SampleScore(metrics={"accuracy": 1.0}, prediction="x")
    out = EvaluationEngine._with_interaction_steps(_Adapter(), _prompt(sample), score)
    assert out.metrics["interaction_steps"] == 3.0
    assert out.metrics["accuracy"] == 1.0, "the adapter's own metrics survive"

    for mode in ("static", "sequential"):
        class _Other:
            data_delivery_mode = mode

        out = EvaluationEngine._with_interaction_steps(_Other(), _prompt(sample), score)
        assert "interaction_steps" not in out.metrics, mode


def test_interaction_steps_is_absent_rather_than_zero_when_unknown():
    """A missing turn count is not a zero-step episode."""
    from abductionbench.core.engine import EvaluationEngine

    class _Adapter:
        data_delivery_mode = "interactive"

    sample = _sample([("a", "r")], "final")
    sample.metadata.pop("turns_used")
    out = EvaluationEngine._with_interaction_steps(
        _Adapter(), _prompt(sample), SampleScore(metrics={})
    )
    assert "interaction_steps" not in out.metrics


# --------------------------------------------------------------------------- #
# metric two: relevance, one binary verdict per action
# --------------------------------------------------------------------------- #


def test_the_actions_are_read_off_the_transcript_with_their_results():
    """An action without what it returned cannot be judged for relevance."""
    sample = _sample([("history: chest pain?", "No chest pain."),
                      ("investigation: ECG", "Third-degree AV block.")],
                     "diagnosis_final: Lyme carditis")
    actions, final = ReasoningJudgeStage._episode_actions(sample)
    assert len(actions) == 2
    assert "ACTION: history: chest pain?" in actions[0]
    assert "RESULT: No chest pain." in actions[0]
    assert final == "diagnosis_final: Lyme carditis"


def test_an_episode_with_nothing_before_the_answer_has_no_steps_to_judge():
    """One action and nothing before it is not an investigation."""
    sample = _sample([], "diagnosis_final: Lyme carditis")
    actions, final = ReasoningJudgeStage._episode_actions(sample)
    assert actions == [] and final == ""


def _stage_with(reply):
    """A stage whose only judge call returns `reply`."""
    stage = object.__new__(ReasoningJudgeStage)

    async def _judge_many(family, requests, *, identity=None, context=None):
        assert family == "step_relevance"
        stage.seen = requests
        return {key: reply for key in requests}

    stage._judge_many = _judge_many
    return stage


def _run(stage, sample, metrics=None):
    identity = TaskIdentity(
        run_id="r", dataset_id="d", model_id="m", template_id="io_n-a_interactive",
        template_version="1", data_delivery_mode="interactive",
    )
    response = ModelResponse(
        sample_id=sample.sample_id, model_id="m", status=ResponseStatus.OK, content="x",
    )
    scored = [(sample, response, SampleScore(metrics=metrics or {}, prediction="x"))]
    return asyncio.run(
        stage.apply_interaction(None, identity, [_prompt(sample)], scored)
    )[0][2]


def test_relevance_is_reported_as_a_rate_and_a_count():
    sample = _sample([("a", "r1"), ("b", "r2"), ("c", "r3"), ("d", "r4")], "final")
    score = _run(_stage_with({"relevance_per_step": [1, 0, 1, 1]}), sample)
    assert score.metrics["interaction_relevant_steps"] == 3.0
    assert score.metrics["interaction_irrelevant_steps"] == 1.0
    assert score.metrics["interaction_step_relevance_rate"] == 0.75
    assert score.details["interaction_step_relevance_per_step"] == [1, 0, 1, 1]


def test_a_verdict_the_judge_could_not_give_is_blank_not_zero():
    """An unusable reply means "not judged", not "nothing was relevant"."""
    sample = _sample([("a", "r1"), ("b", "r2")], "final")
    score = _run(_stage_with({"relevance_per_step": "not a list"}), sample)
    assert "interaction_step_relevance_rate" not in score.metrics
    assert "unjudged" in score.details["interaction_step_relevance"]


def test_a_list_of_the_wrong_length_is_rejected():
    """One verdict per action, or the verdicts cannot be lined up with steps."""
    sample = _sample([("a", "r1"), ("b", "r2"), ("c", "r3")], "final")
    score = _run(_stage_with({"relevance_per_step": [1, 0]}), sample)
    assert "interaction_step_relevance_rate" not in score.metrics


def test_the_judge_is_shown_the_actions_the_answer_and_the_opening():
    sample = _sample([("history: chest pain?", "No chest pain.")], "final answer here")
    stage = _stage_with({"relevance_per_step": [1]})
    _run(stage, sample)
    fields = next(iter(stage.seen.values()))
    assert "syncope" in fields["question"]
    assert "history: chest pain?" in fields["steps"]
    assert fields["model_answer"] == "final answer here"
    # The reference is passed as context only; the prompt tells the judge so.
    assert fields["reference_answer"] == "Lyme carditis"


def test_only_interactive_deliveries_are_judged():
    sample = _sample([("a", "r")], "final")
    stage = _stage_with({"relevance_per_step": [1]})
    for mode in ("static", "sequential"):
        identity = TaskIdentity(
            run_id="r", dataset_id="d", model_id="m", template_id="t",
            template_version="1", data_delivery_mode=mode,
        )
        response = ModelResponse(
            sample_id="s1", model_id="m", status=ResponseStatus.OK, content="x",
        )
        scored = [(sample, response, SampleScore(metrics={}))]
        out = asyncio.run(stage.apply_interaction(None, identity, [_prompt(sample)], scored))
        assert "interaction_step_relevance_rate" not in out[0][2].metrics, mode


# --------------------------------------------------------------------------- #
# the prompt itself
# --------------------------------------------------------------------------- #


def test_the_relevance_prompt_grades_against_the_given_answer_not_the_right_one():
    """Relevance is not correctness.

    A model that investigated well and concluded wrongly still took relevant
    steps; grading relevance against the gold would collapse this metric into
    accuracy and measure nothing new.
    """
    from abductionbench.core.prompts import PromptRegistry

    template = PromptRegistry([Path("configs/prompts")]).get("interaction_step_relevance_v1")
    # Whitespace-normalized: the prompt is wrapped, so a phrase can straddle
    # a newline and a plain substring test would miss it.
    text = " ".join(
        "\n".join(m["content"] for m in template.messages).lower().split()
    )
    assert "not judging whether the model was right" in text
    assert "measured against the answer it gave" in text
    assert "do not grade relevance against it" in text
    # And it must not reward hindsight.
    assert "not on hindsight" in text
    assert "negative is not thereby irrelevant" in text
    assert template.output_contract["json_fields"] == {"relevance_per_step": "list_of_binary"}


def test_the_relevance_prompt_is_not_one_of_the_reasoning_chain_judges():
    """It reads a transcript of actions, not a chain of thought."""
    from abductionbench.core.config import _reasoning_judge_templates

    assert _reasoning_judge_templates()["step_relevance"] == "interaction_step_relevance_v1"
    assert not _reasoning_judge_templates()["step_relevance"].startswith("reasoning_")
