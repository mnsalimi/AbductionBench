"""A decision endpoint is not a chat endpoint.

TypeSafe's jev, reached through OpenRouter's System One API, does not take a
rendered prompt. It takes the evidence as `state` and the answer options as
named `criteria`, and returns which key it chose with a probability over all
of them. Nothing about a system instruction, an answer-format contract or a
chain-of-thought instruction reaches it.

Two things follow and are tested here. It can only answer selection tasks --
there is no free text for it to produce -- and its reply has to be turned into
the answer line the dataset's own scorer already reads, or its column would be
scored by different code from every other model's and the comparison would
mean nothing.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from abductionbench.core.client import BatchProtocolError, ModelClient
from abductionbench.core.config import ModelConfig, TimeoutConfig, load_layered
from abductionbench.core.types import RenderedPrompt, SampleSpec, SamplingParams

REPLY = {
    "model": "typesafe/jev-1.13-20260917",
    "answers": {
        "answer": {
            "type": "choice",
            "choice": "3",
            "probabilities": {"1": 0.12, "2": 0.19, "3": 0.68, "4": 0.01},
            "confidence": 0.58,
        }
    },
    "usage": {"input_tokens": 414, "output_tokens": 45, "cost": 1.7e-05},
    "id": "gen-dec-1",
    "provider": "TypeSafe",
}


def _client(**endpoint):
    model = ModelConfig.model_validate(
        {
            "id": "jev",
            "model_name": "typesafe/jev-1.13",
            "endpoint": {
                "base_url": "https://openrouter.ai/api",
                "protocol": "systemone",
                "batch": {"enabled": False},
                **endpoint,
            },
        }
    )
    return ModelClient(model, TimeoutConfig())


def _prompt(labels=("1", "2", "3", "4"), options=None, **fields):
    sample = SampleSpec(
        sample_id="s-1",
        fields={
            "observation": "The run failed.",
            "context": "",
            "question": "Which category is the root cause?",
            "instructions": "Choose the cause, not a later symptom.",
            "options": list(options or ["Guardrails", "Adherence", "Intent", "Underspecified"]),
            "option_labels": list(labels),
            **fields,
        },
        reference={"gold_label": "4"},
    )
    return RenderedPrompt(
        sample=sample, messages=[], template_id="t", template_version="1",
        sampling=SamplingParams(max_tokens=512), input_tokens_est=10,
    )


def _decide(client, prompts, reply=REPLY):
    sent: list[dict] = []

    async def _post(url, payload, *, batch):
        sent.append({"url": url, "payload": payload})
        return reply

    client._post = _post  # noqa: SLF001
    result = asyncio.run(client.decide(prompts, SamplingParams(max_tokens=512)))
    return result, sent


def test_the_endpoint_is_recognised_as_a_decision_api():
    client = _client()
    assert client.speaks_systemone is True
    assert client.supports_batch is False, "one question per request; there is no batch route"


def test_a_chat_endpoint_is_untouched():
    model = ModelConfig.model_validate(
        {"id": "m", "model_name": "x", "endpoint": {"base_url": "http://x", "batch": {"enabled": True}}}
    )
    client = ModelClient(model, TimeoutConfig())
    assert client.speaks_systemone is False
    assert client.supports_batch is True


def test_the_criteria_are_keyed_by_the_datasets_own_labels():
    """So the chosen key IS the answer, with nothing matched back by text.

    Text matching would be ambiguous here anyway: agentrx offers both
    "Intent Not Supported" and "Intent not supported" as separate options.
    """
    _, sent = _decide(_client(), [_prompt()])
    question = sent[0]["payload"]["questions"]["answer"]
    assert question["type"] == "choice"
    assert question["criteria"] == {
        "1": "Guardrails", "2": "Adherence", "3": "Intent", "4": "Underspecified",
    }
    assert question["instructions"] == "Choose the cause, not a later symptom."


def test_the_state_carries_the_evidence_and_the_question():
    _, sent = _decide(_client(), [_prompt()])
    state = sent[0]["payload"]["state"]
    assert "The run failed." in state
    assert "Question: Which category is the root cause?" in state


def test_the_choice_becomes_the_answer_line_the_scorer_reads():
    """The whole point: jev is scored by the same code as every other model."""
    result, _ = _decide(_client(), [_prompt()])
    assert result.choices[0].content == "Answer: 3"
    assert result.choices[0].finish_reason == "stop"


def test_the_probabilities_are_kept_rather_than_discarded():
    """They are the one thing this endpoint gives that a chat model does not."""
    result, _ = _decide(_client(), [_prompt()])
    raw = result.choices[0].raw
    assert raw["choice"] == "3"
    assert raw["confidence"] == 0.58
    assert raw["probabilities"]["3"] == 0.68
    assert raw["provider"] == "TypeSafe"


def test_the_request_goes_to_the_decision_path():
    _, sent = _decide(_client(), [_prompt()])
    assert sent[0]["url"] == "https://openrouter.ai/api/v1/systemone"


def test_options_and_labels_that_do_not_line_up_are_refused():
    """Silently zipping them short would answer a different question."""
    with pytest.raises(BatchProtocolError, match="one criterion per option"):
        _decide(_client(), [_prompt(labels=("1", "2"), options=["a", "b", "c"])])


def test_a_sample_with_no_options_is_refused():
    """A generation task has nothing to choose between."""
    with pytest.raises(BatchProtocolError):
        _decide(_client(), [_prompt(labels=(), options=[])])


def test_a_reply_with_no_choice_is_an_error_not_an_empty_answer():
    reply = {**REPLY, "answers": {"answer": {"type": "choice"}}}
    result, _ = _decide(_client(), [_prompt()], reply=reply)
    assert result.choices[0].content is None
    assert result.choices[0].finish_reason == "error"


def test_every_prompt_gets_its_own_request():
    result, sent = _decide(_client(), [_prompt(), _prompt(), _prompt()])
    assert len(sent) == 3
    assert len(result.choices) == 3
    assert [c.index for c in result.choices] == [0, 1, 2]


# --------------------------------------------------------------------------- #
# the run config
# --------------------------------------------------------------------------- #


def test_jev_runs_selection_scs_only():
    """Not a preference. A cot instruction is part of a rendered prompt, and a
    decision endpoint never receives one, so a "jev cot" column would be the io
    measurement under another name.
    """
    modes = load_layered("configs/runs/jev_scs.yaml")["modes"]
    assert modes["prompt_modes"] == ["io"]
    assert modes["selection_modes"] == ["SCS"]
    assert modes["hypothesis_modes"] == ["selection"]
    assert modes["bov"] is False


def test_jev_is_asked_three_times_like_every_other_model():
    assert load_layered("configs/runs/jev_scs.yaml")["modes"]["repeats"] == 3


def test_no_judge_runs_for_jev():
    """An SCS answer is checked against the gold label by the dataset's scorer,
    and the reasoning judge has no chain to read: jev returns a choice, not an
    argument.
    """
    engine = load_layered("configs/runs/jev_scs.yaml")["engine"]
    assert engine["judge"]["enabled"] is False
    assert engine["judge"]["defer"] is True
    assert engine["reasoning_judge"]["enabled"] is False


def test_the_shipped_model_config_declares_the_decision_protocol():
    endpoint = load_layered("configs/models/jev-openrouter.yaml")["model"]["endpoint"]
    assert endpoint["protocol"] == "systemone"
    assert endpoint["systemone_path"] == "/v1/systemone"
    assert endpoint["batch"]["enabled"] is False
