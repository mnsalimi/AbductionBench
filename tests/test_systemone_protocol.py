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
    assert "Choose the cause, not a later symptom." in question["instructions"]


def test_the_choice_becomes_the_answer_line_the_scorer_reads():
    """The whole point: jev is scored by the same code as every other model."""
    result, _ = _decide(_client(), [_prompt()])
    # The answer tag, not "Answer: 3": with a LETTER label the scorers' lenient
    # parser read the "A" of "Answer" (see test below).
    assert result.choices[0].content == "<answer>3</answer>"
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


# --------------------------------------------------------------------------- #
# it is not a language model, and the request has to stop pretending it is
# --------------------------------------------------------------------------- #


def test_state_holds_the_evidence_and_not_the_question():
    """TypeSafe's reference: state is "the content to evaluate", instructions
    are "what the model should decide". The question in the state made it one
    more piece of evidence rather than the thing being asked.
    """
    _, sent = _decide(_client(), [_prompt()])
    state = sent[0]["payload"]["state"]
    assert "The run failed." in state
    assert "Which category" not in state, "the question does not belong in the state"


def test_instructions_hold_the_question_and_the_datasets_framing():
    _, sent = _decide(_client(), [_prompt()])
    instructions = sent[0]["payload"]["questions"]["answer"]["instructions"]
    assert instructions.startswith("Which category is the root cause?")
    assert "Choose the cause, not a later symptom." in instructions


def test_the_instructions_do_not_ask_it_to_reason():
    """It scores each option against its rubric in parallel and in isolation.

    A "think step by step" clause would be a sentence with nothing to act on
    it, and would misdescribe the model in the audit log besides.
    """
    _, sent = _decide(_client(), [_prompt()])
    instructions = sent[0]["payload"]["questions"]["answer"]["instructions"].lower()
    for phrase in ("step by step", "reason", "think", "explain", "chain of thought"):
        assert phrase not in instructions, phrase


def test_no_sampling_parameter_is_sent():
    """The API reference documents model, state and questions. Nothing else."""
    _, sent = _decide(_client(), [_prompt()])
    payload = sent[0]["payload"]
    assert set(payload) == {"model", "state", "questions"}
    for banned in ("temperature", "seed", "max_tokens", "top_p"):
        assert banned not in payload


def test_a_decision_endpoint_has_no_prompt_mode():
    """So its column is not labelled io, which would claim a condition that was
    never applied and invite comparison with a cot column that cannot exist.
    """
    from abductionbench.core.config import ModelConfig

    jev = ModelConfig.model_validate(
        {"id": "jev", "model_name": "typesafe/jev-1.13",
         "endpoint": {"base_url": "https://openrouter.ai/api", "protocol": "systemone"}}
    )
    chat = ModelConfig.model_validate(
        {"id": "m", "model_name": "x", "endpoint": {"base_url": "http://x"}}
    )
    assert jev.has_prompt_mode is False
    assert chat.has_prompt_mode is True


def test_the_planner_labels_a_decision_endpoints_task_n_a():
    import inspect

    from abductionbench.core.engine import EvaluationEngine

    source = inspect.getsource(EvaluationEngine._plan_tasks)
    assert 'if model.has_prompt_mode else "n/a"' in source


def test_the_raw_record_carries_no_sampling_for_a_decision_endpoint():
    """Recording temperature against a request that never had one describes a
    call that was not made."""
    import inspect

    from abductionbench.core.engine import EvaluationEngine

    source = inspect.getsource(EvaluationEngine._execute_batch)
    assert 'if client.speaks_systemone' in source
    assert '"sampling": batch.sampling.to_payload()' in source  # still there for chat models


def test_jev_is_never_asked_an_mcs_task():
    """An MCS task is graded as a SET; a choice question returns one option.

    Answering MCS with a single choice scores a one-element set against a
    multi-label gold -- which is what produced aer's set_f1 0.6467 off
    precision 0.86 and recall 0.55 before this filter existed.
    """
    jev = _jev_only_tasks()
    assert not jev.admits(prompt_mode="io", selection_mode="MCS", task_kinds=["selection"])
    assert jev.admits(prompt_mode="io", selection_mode="SCS", task_kinds=["selection"])


def _jev_only_tasks():
    from abductionbench.core.config import ModelTaskFilter, load_layered

    return ModelTaskFilter.model_validate(
        load_layered("configs/models/jev-openrouter.yaml")["model"]["only_tasks"]
    )


def test_the_shipped_jev_config_sends_no_temperature_or_seed():
    from abductionbench.core.config import load_layered

    sampling = load_layered("configs/models/jev-openrouter.yaml")["model"]["sampling"]
    assert "temperature" not in sampling
    assert "seed" not in sampling
    assert "top_p" not in sampling


def test_a_letter_choice_is_scored_as_that_letter_not_the_a_of_answer():
    """Regression, 2026-09-25: jev chose D on scir and was scored A.

    "Answer: D" went through the selection parser as the label "A" (the first
    letter of "Answer"), so jev read as answering A on every letter-labelled
    dataset. The tag it now emits is what every scorer reads first.
    """
    from abductionbench.core.metrics import extract_choice_label

    labels = list("ABCDEFGHIJ")
    assert extract_choice_label("<answer>D</answer>", labels) == "D"
