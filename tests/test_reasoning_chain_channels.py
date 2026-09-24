"""The step judge reads ONE reasoning trace per reply, chosen by precedence.

A reply can carry visible CoT in ``<think>...</think>``, provider-native
reasoning in ``message.reasoning`` / ``reasoning_content``, both, or neither.
The two used to be concatenated. Where a reply carried both that was usually
the same inferential process twice -- gemini-3.8-flash's native field is a
provider-written summary ("**Analyzing...** I'm currently...") of the thinking
its visible block then spells out -- so the judge segmented it twice and every
step-based metric was inflated.

The rule now: neither -> a blank trace; visible CoT present -> the visible CoT
alone, whatever the native field holds; otherwise -> the native field alone.
The reply's content and its native field are kept untouched on the record, and
which one the trace came from is written beside every reasoning metric.
"""

from __future__ import annotations

import pytest

from abductionbench.adapters._prompting import (
    TRACE_NATIVE,
    TRACE_NONE,
    TRACE_UNTAGGED,
    TRACE_VISIBLE,
    reasoning_trace,
)
from abductionbench.core.reasoning_judge import ReasoningJudgeStage, _ungrounded_spans
from abductionbench.core.types import (
    ChatMessage, ModelResponse, RenderedPrompt, ResponseStatus, SampleSpec, SamplingParams,
)

VISIBLE = "Step 1: O1 says the mirror broke. Step 2: only hypothesis 1 explains that."
NATIVE = "**Analyzing the mirror**\n\nI'm currently weighing the two hypotheses."


def _judge_view(reasoning, content, transcript=None):
    metadata = {"_transcript": transcript} if transcript else {}
    sample = SampleSpec(sample_id="s", fields={"observation": "o"}, metadata=metadata)
    prompt = RenderedPrompt(
        sample=sample,
        messages=[ChatMessage(role="user", content="the question")],
        template_id="t", template_version="1",
        sampling=SamplingParams(max_tokens=8), input_tokens_est=1,
    )
    response = ModelResponse(
        sample_id="s", model_id="gemini-3.8-flash-openrouter", status=ResponseStatus.OK,
        content=content, reasoning=reasoning,
    )
    _question, chain, source = ReasoningJudgeStage._question_and_reasoning(prompt, response)
    return chain, source, response


# -- the four combinations -------------------------------------------------- #


def test_neither_channel_gives_a_blank_trace():
    assert _judge_view(None, "<answer>3</answer>")[:2] == ("", TRACE_NONE)
    assert _judge_view(None, None)[:2] == ("", TRACE_NONE)


def test_visible_cot_alone_is_used():
    assert _judge_view(None, f"<think>{VISIBLE}</think><answer>3</answer>")[:2] == (
        VISIBLE, TRACE_VISIBLE,
    )


def test_native_reasoning_alone_is_used():
    assert _judge_view(NATIVE, "<answer>3</answer>")[:2] == (NATIVE, TRACE_NATIVE)


def test_visible_cot_takes_precedence_when_both_channels_are_populated():
    """The regression this exists for: the chain was NATIVE + VISIBLE."""
    chain, source, _ = _judge_view(NATIVE, f"<think>{VISIBLE}</think><answer>3</answer>")
    assert source == TRACE_VISIBLE
    assert chain == VISIBLE
    assert "Analyzing" not in chain and "currently weighing" not in chain


# -- what counts as visible CoT --------------------------------------------- #


@pytest.mark.parametrize("placeholder", ["", " ", ".", " . \n"])
def test_a_placeholder_think_block_is_no_visible_cot(placeholder):
    """gpt-5.6-luna writes <think>.</think> when its chain went to the native
    field; that is a filled-in tag, not a chain -- so the native field is used."""
    content = f"<think>{placeholder}</think><answer>3</answer>"
    assert _judge_view(NATIVE, content)[:2] == (NATIVE, TRACE_NATIVE)
    assert _judge_view(None, content)[:2] == ("", TRACE_NONE)


def test_a_think_block_cut_off_by_the_budget_is_still_visible_cot():
    """A truncated reply opens <think> and never closes it."""
    chain, source, _ = _judge_view(NATIVE, "<think>Step 1: O1 says the mirror")
    assert (chain, source) == ("Step 1: O1 says the mirror", TRACE_VISIBLE)


def test_an_untagged_reply_with_no_native_field_keeps_its_prose():
    """A reply written before the tagged format: its prose is where it reasoned."""
    chain, source, _ = _judge_view(None, "I thought about it. Answer: 3")
    assert (chain, source) == ("I thought about it. Answer: 3", TRACE_UNTAGGED)


# -- nothing but the model's reasoning reaches the judge -------------------- #


def test_the_answer_block_and_the_tags_are_never_in_the_trace():
    for reasoning, content in [
        (None, f"<think>{VISIBLE}</think><answer>3</answer>"),
        (NATIVE, "<answer>3</answer>"),
        (None, f"<think>{VISIBLE} <answer>2</answer></think><answer>3</answer>"),
        ("see </think> and <reasoning_chain> here", "<answer>3</answer>"),
    ]:
        chain = reasoning_trace(content, reasoning).text
        for tag in ("<think>", "</think>", "<answer>", "</answer>", "<reasoning_chain>"):
            assert tag not in chain, (reasoning, content, chain)


def test_a_step_that_is_harness_markup_is_rejected_even_where_its_word_occurs():
    """"</think>" would pass a word check against a chain that says "I think"."""
    chain = "I think hypothesis 1 explains it."
    assert _ungrounded_spans(["I think hypothesis 1 explains it."], chain) == []
    assert _ungrounded_spans(["</think>"], chain) == [0]
    assert _ungrounded_spans(["<reasoning_chain>"], chain) == [0]
    assert _ungrounded_spans(["<answer>3</answer>"], chain) == [0]


# -- the originals are kept for auditing ------------------------------------ #


def test_the_reply_keeps_both_original_channels():
    content = f"<think>{VISIBLE}</think><answer>3</answer>"
    _chain, _source, response = _judge_view(NATIVE, content)
    assert response.content == content
    assert response.reasoning == NATIVE


# -- interactive episodes follow the same precedence ------------------------ #


def test_an_episode_with_visible_cot_does_not_append_the_native_field():
    transcript = [
        {"role": "user", "content": "case"},
        {"role": "assistant", "content": f"<think>{VISIBLE}</think><answer>ask</answer>"},
    ]
    chain, source, _ = _judge_view(NATIVE, None, transcript)
    assert source == TRACE_VISIBLE
    assert "Analyzing" not in chain


def test_an_episode_without_visible_cot_uses_the_native_field():
    transcript = [
        {"role": "user", "content": "case"},
        {"role": "assistant", "content": "<answer>ask about fever</answer>"},
    ]
    chain, source, _ = _judge_view(NATIVE, None, transcript)
    assert source == TRACE_NATIVE
    assert NATIVE in chain
