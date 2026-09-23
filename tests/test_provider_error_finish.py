"""A choice the provider finished with "error" is an outage, not an answer.

OpenRouter reports an upstream failure mid-generation as HTTP 200 whose choice
has ``finish_reason: "error"``, content null (or cut off) and all-zero usage.
Taken as the model's reply it became an EMPTY record -- a parse failure charged
to the model -- and EMPTY is reusable on resume, so it could never be retried.
Seen on ~3% of gemini-3.8-flash's samples in the openrouter-trio run.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from abductionbench.core.checkpoint import RecordStore
from abductionbench.core.client import ModelClient
from abductionbench.core.config import ModelConfig, TimeoutConfig
from abductionbench.core.errors import TransientEndpointError
from abductionbench.core.types import (
    ChatMessage,
    EvalRecord,
    ResponseStatus,
    SamplingParams,
    TaskIdentity,
)


def _client(reply):
    model = ModelConfig.model_validate(
        {"id": "m", "model_name": "x", "endpoint": {"base_url": "http://x", "batch": {"enabled": False}}}
    )
    client = ModelClient(model, TimeoutConfig())

    async def _post(url, payload, *, batch):
        return reply

    client._post = _post  # noqa: SLF001
    return client


def _chat(client):
    return asyncio.run(
        client.chat_single([ChatMessage(role="user", content="q")], SamplingParams(max_tokens=64))
    )


def test_an_error_finish_is_raised_as_transient_so_it_is_retried():
    reply = {
        "choices": [
            {
                "index": 0,
                "finish_reason": "error",
                "message": {"role": "assistant", "content": None, "reasoning": "**Analyzing**"},
                "error": {"code": 502, "message": "upstream error"},
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
    }
    with pytest.raises(TransientEndpointError, match="upstream error"):
        _chat(_client(reply))


def test_an_ordinary_finish_is_untouched():
    reply = {"choices": [{"index": 0, "finish_reason": "stop", "message": {"content": "hi"}}]}
    result = _chat(_client(reply))
    assert result.choices[0].content == "hi"
    assert result.choices[0].finish_reason == "stop"


def _record(sample_id: str, status: ResponseStatus, finish: str) -> EvalRecord:
    return EvalRecord(
        task=TaskIdentity(
            run_id="r", dataset_id="d", model_id="m", template_id="t", template_version="1.0"
        ),
        sample_id=sample_id,
        status=status,
        prompt_fingerprint="fp",
        task_kind="generation",
        input_tokens_est=10,
        sampling={"max_tokens": 64},
        response={"content": None, "finish_reason": finish},
        metrics={},
    )


def test_records_already_written_from_an_error_finish_are_retried_on_resume(tmp_path: Path):
    store = RecordStore(tmp_path)
    store.append(_record("fine", ResponseStatus.OK, "stop"))
    store.append(_record("outage-empty", ResponseStatus.EMPTY, "error"))
    store.append(_record("outage-fragment", ResponseStatus.OK, "error"))
    store.append(_record("real-empty", ResponseStatus.EMPTY, "length"))
    assert set(store.completed_keys(policy="sample_id")) == {"fine", "real-empty"}
