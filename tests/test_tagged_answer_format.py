"""The tagged output contract: <think> for reasoning, <answer> for the answer.

Three properties, and the tests exist because each can break silently:

* what the prompt ASKS FOR and what the parser READS BACK come from one place,
  so they cannot drift into a run that scores nothing;
* io and cot no longer share a format, and the difference is confined to the
  system prompt -- the question itself is byte-identical between them, which is
  what makes an io/cot comparison a comparison;
* strict format compliance is measured and reported SEPARATELY from
  correctness, because a model that answers correctly in the wrong shape and
  one that answers wrongly are different findings.
"""

from __future__ import annotations

import pytest

from abductionbench.adapters._prompting import (
    ANSWER_TAG_REGEX,
    PromptParts,
    build_messages,
    format_compliance,
    format_segment,
)
from abductionbench.core.metrics import extract_answer_span, extract_choice_label
from abductionbench.core.modes import BOV, MCS, SCS, TaskModes

PARTS = PromptParts(
    system="You are an expert at abductive reasoning.",
    observation="The lawn is wet.",
    question="Why is the lawn wet?",
    options=["It rained", "The sprinkler ran", "Dew formed"],
    option_labels=["1", "2", "3"],
)
SELECTIONS = (None, SCS, MCS, BOV)


def _messages(prompt_mode, selection):
    return build_messages(PARTS, TaskModes(prompt_mode=prompt_mode, selection_mode=selection))


# --------------------------------------------------------------------------- #
# the format block, and where it sits
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("selection", SELECTIONS)
def test_the_format_block_ends_the_system_prompt(selection):
    """Immediately after the mode instruction, because the two are one
    statement: whether to reason, and where the reasoning and the answer go."""
    for mode in ("io", "cot"):
        system = _messages(mode, selection)[0][0].content
        assert system.endswith(format_segment(TaskModes(prompt_mode=mode)))


@pytest.mark.parametrize("selection", SELECTIONS)
def test_io_asks_for_an_answer_block_alone(selection):
    system = _messages("io", selection)[0][0].content
    assert "<answer>your answer</answer>" in system
    assert "<think>your reasoning</think>" not in system
    assert "Do not include a <think> block." in system


@pytest.mark.parametrize("selection", SELECTIONS)
def test_cot_asks_for_a_think_block_then_an_answer_block(selection):
    system = _messages("cot", selection)[0][0].content
    assert system.index("<think>your reasoning</think>") < system.index(
        "<answer>your answer</answer>"
    )


@pytest.mark.parametrize("selection", SELECTIONS)
def test_the_question_is_identical_in_both_modes(selection):
    """The mode changes how to reply, never what is asked."""
    assert _messages("io", selection)[0][-1].content == _messages("cot", selection)[0][-1].content


@pytest.mark.parametrize("selection", SELECTIONS)
@pytest.mark.parametrize("mode", ["io", "cot"])
def test_no_obsolete_format_wording_survives_anywhere(mode, selection):
    """`Answer:`, "on the last line", "your entire response" -- all describe a
    reply whose answer has no container. There is one now."""
    messages, _c = _messages(mode, selection)
    whole = "\n".join(m.content for m in messages)
    for banned in ("Answer:", "on the last line", "On the last line",
                   "Your entire response must be", "End your reply with the answer block"):
        assert banned not in whole, (mode, selection, banned)


# --------------------------------------------------------------------------- #
# emit and parse come from one place
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("selection", SELECTIONS)
@pytest.mark.parametrize("mode", ["io", "cot"])
def test_the_contract_parses_what_the_prompt_asked_for(mode, selection):
    """The invariant that makes drift impossible to ship.

    If the prompt asked for tags and the contract still read a marker, every
    dataset would silently score the whole response.
    """
    _messages_, contract = _messages(mode, selection)
    assert contract["answer_regex"] == ANSWER_TAG_REGEX
    assert "answer_prefix" not in contract
    assert extract_answer_span("<answer>2</answer>", contract) == "2"


def test_the_last_answer_block_wins():
    """Same property the marker parser had: a reply that mentions the tag while
    reasoning still parses to what it finally submitted."""
    contract = {"answer_regex": ANSWER_TAG_REGEX}
    text = "<think>maybe <answer>1</answer> no</think><answer>3</answer>"
    assert extract_answer_span(text, contract) == "3"


def test_a_label_is_still_recoverable_from_a_tagged_reply():
    contract = {"answer_regex": ANSWER_TAG_REGEX}
    assert extract_choice_label("<answer>B</answer>", ["A", "B", "C"], contract) == "B"


def test_a_multiline_answer_survives_the_tags():
    contract = {"answer_regex": ANSWER_TAG_REGEX}
    reply = "<answer>fact one\nfact two\nfact three</answer>"
    assert extract_answer_span(reply, contract) == "fact one\nfact two\nfact three"


# --------------------------------------------------------------------------- #
# compliance, measured strictly, reported separately
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("reply", "mode", "strict"),
    [
        ("<answer>3</answer>", "io", True),
        ("<answer>3</answer>", "cot", False),
        ("<think>because</think><answer>3</answer>", "cot", True),
        ("<think>because</think>\n<answer>3</answer>", "cot", True),
        ("<think>because</think><answer>3</answer>", "io", False),
        ("Sure! <answer>3</answer>", "io", False),
        ("<answer>3</answer> hope that helps", "io", False),
        ("<answer>3</answer><answer>4</answer>", "io", False),
        ("Answer: 3", "io", False),
        ("", "io", False),
    ],
)
def test_strict_compliance(reply, mode, strict):
    assert format_compliance(reply, TaskModes(prompt_mode=mode)) is strict


def test_a_sloppy_reply_is_still_scored_even_though_it_is_non_compliant():
    """The parser stays lenient on purpose. Compliance is a separate column,
    not a reason to throw an answer away."""
    contract = {"answer_regex": ANSWER_TAG_REGEX}
    reply = "Sure! <answer>3</answer> hope that helps"
    assert format_compliance(reply, TaskModes(prompt_mode="io")) is False
    assert extract_answer_span(reply, contract) == "3"


def test_compliance_is_reported_as_its_own_metric():
    import inspect

    from abductionbench.core.engine import EvaluationEngine

    source = inspect.getsource(EvaluationEngine._aggregate)
    assert "format_compliance_rate" in source
    assert "n_format_violations" in source
    # Static only: an interactive episode answers in its own protocol.
    assert 'data_delivery_mode == "static"' in source


# --------------------------------------------------------------------------- #
# the chain the reasoning metrics read
# --------------------------------------------------------------------------- #


def _chain(reasoning, content):
    from abductionbench.core.reasoning_judge import ReasoningJudgeStage
    from abductionbench.core.types import (
        ChatMessage, ModelResponse, RenderedPrompt, ResponseStatus, SampleSpec, SamplingParams,
    )

    prompt = RenderedPrompt(
        sample=SampleSpec(sample_id="s", fields={"observation": "o"}),
        messages=[ChatMessage(role="user", content="q")],
        template_id="t", template_version="1",
        sampling=SamplingParams(max_tokens=8), input_tokens_est=1,
    )
    response = ModelResponse(
        sample_id="s", model_id="m", status=ResponseStatus.OK,
        content=content, reasoning=reasoning,
    )
    return ReasoningJudgeStage._question_and_reasoning(prompt, response)[1]


def test_the_chain_is_the_think_block_when_there_is_one():
    assert _chain(None, "<think>step one</think><answer>3</answer>") == "step one"


def test_the_answer_block_is_never_part_of_the_chain():
    """It is the conclusion, not an inferential step -- the same reason
    `reasoning_steps_v3` returns it separately."""
    for content in ("<think>step one</think><answer>3</answer>", "reasoning here<answer>3</answer>"):
        assert "<answer>" not in _chain(None, content)
        assert "3" not in _chain(None, content).replace("step one", "")


def test_a_providers_separate_reasoning_field_is_still_used():
    """Some providers return the chain in their own field whether or not the
    prompt asked for tags. Dropping it would make cot extraction depend on the
    vendor rather than on the model."""
    chain = _chain("hidden trace", "<think>visible</think><answer>3</answer>")
    assert "hidden trace" in chain and "visible" in chain
    assert chain.index("hidden trace") < chain.index("visible")


def test_an_untagged_reply_still_yields_a_chain():
    """A model that reasoned without tagging is not a model that did not reason."""
    assert _chain(None, "I thought about it. Answer: 3") == "I thought about it. Answer: 3"
