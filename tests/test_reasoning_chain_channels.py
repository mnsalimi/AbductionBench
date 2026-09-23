"""The chain a metric reads is everything the model produced, in order.

A model can put its reasoning in a dedicated `reasoning` channel, in the
`content` ahead of the answer line, or in both. This used to be
`reasoning or content`, which takes whichever is populated and silently drops
the other when both are -- correct only if a model always uses exactly one.

gpt-5.6-luna does not. Measured on one sample across six prompt/reasoning
settings: a trace in the reasoning channel for one setting, nothing there but
58-293 tokens of explanation in the content for two more, and eight tokens of
bare answer for the rest. So the old rule fed the metrics the trace alone
where both existed -- losing the conclusion it was building toward -- and the
content alone everywhere else, which is two different measurements sharing one
column.
"""

from __future__ import annotations

from abductionbench.core.reasoning_judge import ReasoningJudgeStage
from abductionbench.core.types import (
    ChatMessage, ModelResponse, RenderedPrompt, ResponseStatus, SampleSpec, SamplingParams,
)


def _chain(reasoning, content, transcript=None):
    metadata = {"_transcript": transcript} if transcript else {}
    sample = SampleSpec(sample_id="s", fields={"observation": "o"}, metadata=metadata)
    prompt = RenderedPrompt(
        sample=sample,
        messages=[ChatMessage(role="user", content="the question")],
        template_id="t", template_version="1",
        sampling=SamplingParams(max_tokens=8), input_tokens_est=1,
    )
    response = ModelResponse(
        sample_id="s", model_id="gpt-5.6-luna-openrouter", status=ResponseStatus.OK,
        content=content, reasoning=reasoning,
    )
    _question, chain = ReasoningJudgeStage._question_and_reasoning(prompt, response)
    return chain


def test_both_channels_are_concatenated_reasoning_first():
    chain = _chain("I weighed the two candidates.", "Answer: 3")
    assert chain == "I weighed the two candidates.\n\nAnswer: 3"


def test_only_a_reasoning_trace_is_used_as_it_is():
    assert _chain("the trace", None) == "the trace"


def test_only_content_is_used_as_it_is():
    """The common case: the chain sits in the content, ahead of the answer."""
    assert _chain(None, "Step 1 ... Step 2 ... Answer: 3") == "Step 1 ... Step 2 ... Answer: 3"


def test_neither_channel_gives_an_empty_chain():
    assert _chain(None, None) == ""


def test_a_whitespace_only_channel_counts_as_absent():
    """No separator dangling off a field that holds nothing."""
    assert _chain("   \n ", "Answer: 3") == "Answer: 3"
    assert _chain("the trace", "\n\t") == "the trace"


def test_the_conclusion_is_no_longer_dropped_when_a_trace_exists():
    """The regression this exists for.

    Under `reasoning or content`, a model that returned both had its answer --
    and any reasoning it wrote in the content -- discarded, so every per-step
    metric was computed over the hidden trace alone.
    """
    chain = _chain("thinking about it", "Because the tool output was misread. Answer: 9")
    assert "thinking about it" in chain
    assert "Answer: 9" in chain
    assert chain.index("thinking about it") < chain.index("Answer: 9")
