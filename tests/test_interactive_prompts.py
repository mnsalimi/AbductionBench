"""The four localised interactive protocols: io integrity, and the protocol intact.

These adapters used to send their releases' own prompts, three of which asked
the model to reason -- a ``"reasoning"`` JSON field, a ``Thought:`` line, "explain
the decision value of this test". Their prompts are now this suite's, built from
the same layers as the static ones, and they run ``io`` only.

Two things therefore have to hold at once, and each test here holds one of them
against a *whole compiled conversation* rather than against the opening alone:
nothing in any turn asks the model to expose reasoning, and everything the
environment needs -- the action vocabulary, the evidence rules, the gates, the
submission shape -- is still there.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from conftest import step

from abductionbench.core.adapter import AdapterContext
from abductionbench.core.modes import TaskModes
from abductionbench.core.registry import resolve_adapter

#: The four adapters this applies to, with the dataset id their data lives under.
PROTOCOL_ADAPTERS = [
    ("cloud_opsbench", "cloud_opsbench:CloudOpsBenchAdapter"),
    ("med_inquire", "med_inquire:MedInquireAdapter"),
    ("medqdx", "medqdx:MedQDxAdapter"),
    ("vivabench", "vivabench:VivaBenchAdapter"),
]

#: Phrasings that ask the model to produce or expose reasoning. A prompt that
#: carries any of these *and* "Answer directly. Do not explain your reasoning."
#: is telling the model two different things, which is the failure this suite
#: removed from the static prompts and must not reintroduce here.
REASONING_REQUESTS = [
    r"\bstep[- ]by[- ]step\b",
    r"\bthink\s+(?:through|about|carefully)\b",
    r"\bwork\s+through\s+the\s+evidence\b",
    r"\bexplain\s+(?:your|the|why)\b",
    r"\bjustify\b",
    r"\bprovide\s+(?:a\s+)?(?:clear\s+)?reasoning\b",
    r"\breasoning\s+chain\b",
    r"\bchain[- ]of[- ]thought\b",
    r"\byour\s+rationale\b",
    r"\bshow\s+your\s+work(?:ing)?\b",
    # The JSON/prefix fields that make reasoning part of the output contract.
    r'"reasoning"\s*:',
    r"^\s*Thought\s*:",
    r"\bline\s+of\s+reasoning\b",
]

#: The mode instruction these prompts must carry, byte for byte.
IO_INSTRUCTION = "Answer directly. Do not explain your reasoning."


def _adapter(dataset_id: str, impl: str, prompt_mode: str = "io"):
    cls = resolve_adapter(f"abductionbench.adapters.{impl}")
    modes = TaskModes(prompt_mode=prompt_mode, data_delivery_mode="interactive")
    adapter = cls(
        AdapterContext(
            dataset_id=dataset_id,
            data_dir=Path("data") / dataset_id,
            modes=modes,
            sample_size=3,
            offline=True,
        )
    )
    adapter.prepare()
    return adapter, adapter.build_samples()


def _offending(text: str) -> list[str]:
    """Reasoning *requests* in one message.

    The io instruction is removed before scanning, and deliberately so: it
    contains the words "explain your reasoning" as a prohibition, and the thing
    being looked for is the opposite -- a turn that asks for reasoning while the
    io instruction forbids it. Scanning the sentence we are protecting would
    flag every correct prompt and nothing else.
    """
    body = (text or "").replace(IO_INSTRUCTION, "")
    return [pattern for pattern in REASONING_REQUESTS if re.search(pattern, body, re.I | re.M)]


def _compile_conversation(adapter, sample, replies: list[str]) -> list[str]:
    """Every message the model would see: the opening, then each environment turn.

    The point is that an audit of the opening alone proves nothing -- three of
    these four benchmarks restate their protocol mid-episode, and VivaBench's
    parse-error retry used to restate the JSON schema *with* its reasoning
    field. A conversation is only clean if every turn in it is.
    """
    messages, state = adapter.interactive_start(sample)
    seen = [m.content for m in messages]
    for reply in replies:
        environment = step(adapter, sample, state, reply)
        if environment is None:
            break
        seen.append(environment)
    return seen


# --------------------------------------------------------------------------- #
# io integrity
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dataset_id,impl", PROTOCOL_ADAPTERS)
def test_the_opening_carries_our_io_instruction_unchanged(dataset_id, impl):
    adapter, samples = _adapter(dataset_id, impl)
    messages, _state = adapter.interactive_start(samples[0])
    opening = "\n".join(m.content for m in messages)
    assert IO_INSTRUCTION in opening, f"{dataset_id} lost the shared io instruction"


@pytest.mark.parametrize("dataset_id,impl", PROTOCOL_ADAPTERS)
def test_no_turn_of_a_compiled_conversation_asks_for_reasoning(dataset_id, impl):
    """The whole episode, not just its first message."""
    adapter, samples = _adapter(dataset_id, impl)
    # Replies chosen to walk the branches that produce environment text:
    # a well-formed action, an unparseable one (the retry path), and enough
    # actions to trip a category limit and any phase change.
    replies = [
        '{"action": "history", "query": "How long has this been going on?"}',
        "this is not an action at all",
        '{"action": "examination", "query": "Listen to the chest"}',
        "Action: GetAlerts\nAction Input: {}",
        '{"action_type": "AskQuestion", "action_text": "Any fever?"}',
        "Does it hurt when you breathe in?",
    ] * 6
    for sample in samples[:2]:
        for turn, text in enumerate(_compile_conversation(adapter, sample, replies)):
            found = _offending(text)
            assert not found, (
                f"{dataset_id} turn {turn} asks for reasoning ({found}):\n{text[:400]}"
            )


@pytest.mark.parametrize("dataset_id,impl", PROTOCOL_ADAPTERS)
def test_these_adapters_run_io_only(dataset_id, impl):
    cls = resolve_adapter(f"abductionbench.adapters.{impl}")
    assert cls.io_only is True, dataset_id
    # The prompts are ours now, so the provenance flag must not claim otherwise.
    assert cls.authors_prompt is False, dataset_id
    assert cls.supports_modes(TaskModes(prompt_mode="io")) is None, dataset_id
    for refused in ("cot", "self-consistency"):
        why = cls.supports_modes(TaskModes(prompt_mode=refused))
        assert why, f"{dataset_id} would schedule a {refused} task"
        assert "protocol" in why, dataset_id


# --------------------------------------------------------------------------- #
# the protocol still works
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dataset_id,impl", PROTOCOL_ADAPTERS)
def test_the_opening_names_every_action_the_environment_accepts(dataset_id, impl):
    """A protocol prompt that omits an action makes that action unreachable."""
    adapter, samples = _adapter(dataset_id, impl)
    messages, _state = adapter.interactive_start(samples[0])
    opening = "\n".join(m.content for m in messages)
    if dataset_id == "medqdx":
        # MedQDx has no action vocabulary: its turn is a plain question, and
        # the shape of that question is what the opening has to state.
        assert "one question" in opening.lower()
        assert "?" in opening
        return
    if dataset_id == "cloud_opsbench":
        # Its vocabulary is this case's own recorded tools plus Finalize, not a
        # fixed ACTIONS tuple, so the opening has to list what the cache holds.
        _messages2, state = adapter.interactive_start(samples[0])
        tools = {key.split(":")[0] for key in state["cache"] if ":" in key}
        assert tools, "no recorded tools for this case"
        for tool in tools:
            assert tool in opening, f"cloud_opsbench: {tool} missing from the prompt"
        assert "Finalize" in opening
        return
    for action in adapter.ACTIONS:
        assert re.search(re.escape(action), opening, re.I), f"{dataset_id}: {action} missing"


def test_vivabench_keeps_its_gates_vocabulary_and_diagnosis_shape():
    adapter, samples = _adapter("vivabench", "vivabench:VivaBenchAdapter")
    messages, state = adapter.interactive_start(samples[0])
    opening = "\n".join(m.content for m in messages)

    # The workflow the examiner actually enforces has to be stated.
    assert "before ordering any test" in opening
    assert "provisional diagnosis" in opening
    assert "no longer available" in opening
    assert "one action per turn" in opening
    # The payload the scorer reads back.
    assert '"condition"' in opening and '"confidence"' in opening
    # And the field that was removed is gone from the contract.
    assert '"reasoning"' not in opening

    # The gate still fires: an investigation closes history.
    step(adapter, samples[0], state, '{"action": "investigation", "query": "FBC"}')
    closed = step(adapter, 
        samples[0], state, '{"action": "history", "query": "Any fevers?"}'
    )
    assert "no longer review the patient" in closed


def test_vivabench_retry_does_not_reintroduce_the_reasoning_field():
    """The release's ERROR_RETURN_MSG restated the schema including `reasoning`."""
    adapter, samples = _adapter("vivabench", "vivabench:VivaBenchAdapter")
    _messages, state = adapter.interactive_start(samples[0])
    reply = step(adapter, samples[0], state, "I would like to examine the patient")
    assert reply is not None
    assert not _offending(reply), reply
    assert '"action"' in reply and '"query"' in reply


def test_vivabench_still_extracts_the_top_ranked_final_diagnosis():
    """Removing the reasoning field must not disturb what the scorer reads."""
    from abductionbench.core.types import ModelResponse, ResponseStatus

    adapter, samples = _adapter("vivabench", "vivabench:VivaBenchAdapter")
    sample = samples[0]
    _messages, state = adapter.interactive_start(sample)
    submission = (
        '{"action": "diagnosis_final", "query": ['
        '{"condition": "Aortic stenosis", "icd_10_name": "Aortic valve stenosis", '
        '"icd_10": "I35.0", "confidence": 0.8}, '
        '{"condition": "Anaemia", "icd_10_name": "Anaemia", "icd_10": "D64", '
        '"confidence": 0.2}]}'
    )
    assert step(adapter, sample, state, submission) is None
    sample.metadata["_episode_state"] = state
    score = adapter.score(
        sample,
        ModelResponse(sample_id="s", model_id="m", status=ResponseStatus.OK, content=submission),
    )
    assert score.metrics["committed"] == 1.0
    assert score.prediction == "Aortic stenosis", "top-1 is what R@1 scores"


def test_medqdx_hands_off_to_the_diagnosis_turn_without_a_reasoning_request():
    adapter, samples = _adapter("medqdx", "medqdx:MedQDxAdapter")
    sample = samples[0]
    _messages, state = adapter.interactive_start(sample)
    handoff = None
    for index in range(20):
        reply = step(adapter, sample, state, f"Is symptom {index} present?")
        if reply is None:
            break
        if "Name the condition" in reply:
            handoff = reply
            break
    assert handoff, "the interview never reached the diagnosis turn"
    assert not _offending(handoff), handoff
    assert IO_INSTRUCTION in handoff
    assert "one condition" in handoff


def test_cloud_opsbench_still_replays_a_recorded_tool_call():
    adapter, samples = _adapter("cloud_opsbench", "cloud_opsbench:CloudOpsBenchAdapter")
    sample = samples[0]
    messages, state = adapter.interactive_start(sample)
    opening = "\n".join(m.content for m in messages)
    assert "Action Input" in opening and "Finalize" in opening
    tool = next(iter(state["cache"])).split(":")[0]
    reply = step(adapter, sample, state, f"Action: {tool}\nAction Input: {{}}")
    assert reply is not None and reply.strip()
    assert not _offending(reply)
    # Finalising ends the episode, which is what the scorer reads.
    assert step(adapter, 
        sample, state, 'Action: Finalize\nAction Input: {"root_cause": "x", "fault_object": "a/b"}'
    ) is None


def test_med_inquire_actions_and_unavailable_test_reply():
    adapter, samples = _adapter("med_inquire", "med_inquire:MedInquireAdapter")
    sample = samples[0]
    _messages, state = adapter.interactive_start(sample)
    reply = step(adapter, 
        sample, state, '{"action_type": "OrderTest", "action_text": "zzzz unrecorded assay"}'
    )
    assert reply is not None and not _offending(reply)
    assert step(adapter, 
        sample, state, '{"action_type": "SubmitDiagnosis", "action_text": "Sarcoidosis"}'
    ) is None
