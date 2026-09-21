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


class _Adapter:
    """The gold, stated the way every real adapter states it.

    `judge_request` is the adapter-owned hook the answer judge already reads,
    so relevance is measured against the same answer accuracy is -- not a
    second one read out of `reference` by a key that not every dataset uses.
    """

    dataset_id = "d"

    def __init__(self, gold="Lyme carditis"):
        self._gold = gold

    def judge_request(self, sample, response, score):
        return {"candidate": "x", "gold": self._gold} if self._gold else None


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


def _run(stage, sample, metrics=None, adapter=None):
    identity = TaskIdentity(
        run_id="r", dataset_id="d", model_id="m", template_id="io_n-a_interactive",
        template_version="1", data_delivery_mode="interactive",
    )
    response = ModelResponse(
        sample_id=sample.sample_id, model_id="m", status=ResponseStatus.OK, content="x",
    )
    scored = [(sample, response, SampleScore(metrics=metrics or {}, prediction="x"))]
    return asyncio.run(
        stage.apply_interaction(
            adapter if adapter is not None else _Adapter(), identity, [_prompt(sample)], scored
        )
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
    # The gold, and it is the standard rather than context: relevance is graded
    # against it with full hindsight, and `model_answer` is the context.
    assert fields["reference_answer"] == "Lyme carditis"


def test_the_gold_comes_from_the_adapter_not_from_a_reference_key():
    """`sample.reference` is not a common shape, so it cannot be the source.

    cloud_opsbench keys its answer `root_cause` and has no `gold` at all, and
    vivabench's answer is a list of accepted diagnoses. Reading `reference
    ["gold"]` would have sent cloud_opsbench an empty standard -- and with
    `reference_answer` now required, an empty one is not even renderable.
    """
    sample = _sample([("kubectl get pods", "CrashLoopBackOff")], "final")
    sample.reference.clear()
    sample.reference["root_cause"] = "missing_service_account"
    stage = _stage_with({"relevance_per_step": [1]})
    _run(stage, sample, adapter=_Adapter("missing_service_account"))
    assert next(iter(stage.seen.values()))["reference_answer"] == "missing_service_account"


def test_a_reference_the_adapter_declines_falls_back_to_the_sample():
    """An adapter can decline for a reason that has nothing to do with the gold.

    vivabench returns no judge request for an episode that never committed to a
    diagnosis, and every adapter returns none for an empty response. The case
    still has an answer, so the metric is still measurable.
    """
    sample = _sample([("a", "r1")], "final")
    sample.reference.clear()
    sample.reference["accepted"] = ["Lyme carditis", "Lyme disease with AV block"]
    stage = _stage_with({"relevance_per_step": [1]})
    _run(stage, sample, adapter=_Adapter(None))
    assert next(iter(stage.seen.values()))["reference_answer"] == (
        "Lyme carditis; Lyme disease with AV block"
    )


def test_no_gold_anywhere_is_unjudged_rather_than_graded_on_the_models_answer():
    """The fallback is not "use the answer it gave" -- that is the old metric."""
    sample = _sample([("a", "r1"), ("b", "r2")], "final")
    sample.reference.clear()
    stage = _stage_with({"relevance_per_step": [1, 1]})
    score = _run(stage, sample, adapter=_Adapter(None))
    assert getattr(stage, "seen", None) is None
    assert "interaction_step_relevance_rate" not in score.metrics
    assert "no reference answer" in score.details["interaction_step_relevance"]


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
        out = asyncio.run(
            stage.apply_interaction(_Adapter(), identity, [_prompt(sample)], scored)
        )
        assert "interaction_step_relevance_rate" not in out[0][2].metrics, mode


# --------------------------------------------------------------------------- #
# the prompt itself
# --------------------------------------------------------------------------- #


def _relevance_template():
    from abductionbench.core.prompts import PromptRegistry

    return PromptRegistry([Path("configs/prompts")]).get("interaction_step_relevance_v1")


def _flat(template):
    """Whitespace-normalized prompt text.

    The prompt is wrapped, so a phrase can straddle a newline and a plain
    substring test would miss it.
    """
    return " ".join("\n".join(m["content"] for m in template.messages).lower().split())


def test_the_relevance_prompt_grades_against_the_gold_with_full_hindsight():
    """The standard is the correct answer, not the one the model gave.

    Graded against the model's own answer, an episode that marched confidently
    down a wrong path scores a perfect relevance rate: every step it took did
    work toward the answer it ended up giving. That reads as a well-run
    investigation and it is the opposite of one. The gold is what makes the
    number mean "did this action get closer to being right".
    """
    template = _relevance_template()
    text = _flat(template)
    assert "judge with full hindsight, against the correct answer" in text
    assert "regardless of what the model ultimately answered" in text
    assert "it led the investigation away from the correct answer" in text
    assert template.output_contract["json_fields"] == {"relevance_per_step": "list_of_binary"}


def test_the_relevance_prompt_keeps_none_of_the_answer_relative_language():
    """The v1 wording said the opposite, in four places.

    Each of these is a sentence that, left in beside the new instruction, would
    tell the judge to do exactly what the metric no longer asks for -- and a
    contradictory prompt does not fail, it just returns something else.
    """
    text = _flat(_relevance_template())
    for gone in (
        "not judging whether the model was right",
        "measured against the answer it gave",
        "do not grade relevance against it",
        "context for reading the transcript and nothing more",
        # hindsight is now required, not forbidden
        "not on hindsight",
        "what was known when it was taken",
        # and a move that did not pan out no longer counts
        "reasonable next move",
        "negative is not thereby irrelevant",
    ):
        assert gone not in text, gone


def test_the_relevance_prompt_requires_the_reference_answer():
    """Optional would make it silently a different metric.

    A prompt rendered without the gold still renders -- `optional_fields` are
    filled with "" -- and the judge would then fall back to the only answer it
    could see, the model's own. That is the v1 measurement wearing v2's name.
    """
    template = _relevance_template()
    assert "reference_answer" in template.required_fields
    assert "reference_answer" not in template.optional_fields
    assert template.optional_fields == []


def test_the_relevance_prompt_is_not_one_of_the_reasoning_chain_judges():
    """It reads a transcript of actions, not a chain of thought."""
    from abductionbench.core.config import _reasoning_judge_templates

    assert _reasoning_judge_templates()["step_relevance"] == "interaction_step_relevance_v1"
    assert not _reasoning_judge_templates()["step_relevance"].startswith("reasoning_")
