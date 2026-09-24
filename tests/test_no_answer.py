"""A reply that never answered has no reasoning to measure, and is counted.

A model that loops until its token budget runs out -- gemma-4-31b did on 16%
of house_md's cot replies in openrouter-trio -- leaves a chain that is a
fragment or a loop, not the reasoning behind an answer. Its step counts used
to be averaged in beside real ones. Now the reasoning judge skips it (blank,
not zero), and the workbook's "No_answer" sheet says how often it happened per
dataset, model and mode.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from abductionbench.adapters._prompting import (
    NO_ANSWER_CUT_OFF,
    NO_ANSWER_EMPTY,
    missing_answer,
)
from abductionbench.core.config import ReasoningJudgeConfig
from abductionbench.core.reasoning_judge import ReasoningJudgeStage
from abductionbench.core.types import (
    ChatMessage, ModelResponse, RenderedPrompt, ResponseStatus, SampleScore, SampleSpec,
    SamplingParams, TaskIdentity,
)


# -- what counts as no answer ----------------------------------------------- #


@pytest.mark.parametrize(
    ("content", "finish", "why"),
    [
        (None, "stop", NO_ANSWER_EMPTY),
        ("   \n", "length", NO_ANSWER_EMPTY),
        ("<think>still going and going", "length", NO_ANSWER_CUT_OFF),
        ("Step 1 ... Step 2 ... (Sigh). (Sigh).", "length", NO_ANSWER_CUT_OFF),
        # Cut off AFTER answering: the answer is there.
        ("<think>t</think><answer>3</answer> and then more", "length", None),
        ("<think>t</think><ANSWER>3</ANSWER>", "length", None),
        # Finished on its own without tags: a formatting lapse, not a missing answer.
        ("The answer is 3.", "stop", None),
        ("<think>t</think><answer>3</answer>", "stop", None),
    ],
)
def test_missing_answer(content, finish, why):
    assert missing_answer(content, finish) == why


# -- the reasoning judge skips it ------------------------------------------- #


class _Registry:
    def get(self, template_id):
        class _T:
            ref = f"{template_id}@1.0"
            id = template_id
            version = "1.0"
            required_fields: list[str] = []
            output_contract: dict = {}
            messages: list = []

        return _T()


class _Adapter:
    dataset_id = "d"

    def documentation(self):
        class _D:
            processing_mode = "selection"

        return _D()


def _judge(tmp_path, content, finish):
    stage = ReasoningJudgeStage(
        config=ReasoningJudgeConfig(enabled=True, model="m"),
        registry=_Registry(), renderer=None, clients={"m": object()},
        retry_policy=None, cache_dir=tmp_path,
    )
    sent: list = []

    async def capture(targets, identity):
        sent.extend(targets)
        for target in targets:
            target.raw = {}

    stage._evaluate = capture
    sample = SampleSpec(sample_id="s1", fields={"observation": "x"},
                        reference={"gold": "g"}, task_kind="generation")
    prompt = RenderedPrompt(
        sample=sample, messages=[ChatMessage(role="user", content="what happened?")],
        template_id="t", template_version="1.0", sampling=SamplingParams(max_tokens=64),
        input_tokens_est=1,
    )
    response = ModelResponse(sample_id="s1", model_id="m", status=ResponseStatus.OK,
                             content=content, finish_reason=finish)
    identity = TaskIdentity(
        run_id="r", dataset_id="d", model_id="m", template_id="t", template_version="1.0",
        prompt_mode="cot", selection_mode="n/a", data_delivery_mode="static",
        task_kind="generation",
    )
    scored = [(sample, response, SampleScore(metrics={}, prediction=""))]
    out = asyncio.run(stage.apply(_Adapter(), identity, [prompt], scored))
    return stage, sent, out


def test_a_reply_cut_off_before_answering_is_not_sent_to_the_reasoning_judge(tmp_path):
    stage, sent, out = _judge(tmp_path, "<think>step one. step two. step two again", "length")
    assert sent == []
    assert stage.stats["skipped_no_answer"] == 1
    status = out[0][2].details.get("reasoning_metrics_status", "")
    assert f"no_answer:{NO_ANSWER_CUT_OFF}" in status


def test_a_reply_that_answered_is_still_judged(tmp_path):
    _stage, sent, _out = _judge(
        tmp_path, "<think>step one. step two.</think><answer>3</answer>", "stop"
    )
    assert len(sent) == 1


# -- the workbook counts it ------------------------------------------------- #


def test_the_no_answer_sheet_counts_per_task(tmp_path):
    from abductionbench.core.engine import RunResult, TaskResult
    from abductionbench.core.reporting import build_no_answer_frame

    task_dir = tmp_path / "datasets" / "d" / "m" / "cot"
    task_dir.mkdir(parents=True)
    records = [
        {"sample_id": "a", "status": "ok", "prompt_fingerprint": "1",
         "response": {"content": "<think>t</think><answer>1</answer>", "finish_reason": "stop"}},
        {"sample_id": "b", "status": "truncated", "prompt_fingerprint": "1",
         "response": {"content": "<think>loop loop loop", "finish_reason": "length"}},
        {"sample_id": "c", "status": "empty", "prompt_fingerprint": "1",
         "response": {"content": None, "finish_reason": "stop"}},
        {"sample_id": "d", "status": "ok", "prompt_fingerprint": "1",
         "response": {"content": "<think>t</think><answer>2</answer>", "finish_reason": "stop"}},
        # A failed call: the model was never heard from -- not "no answer".
        {"sample_id": "e", "status": "error", "prompt_fingerprint": "1", "response": {}},
    ]
    (task_dir / "records.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    identity = TaskIdentity(
        run_id="r", dataset_id="d", model_id="m", template_id="cot_n-a_static",
        template_version="2.0", prompt_mode="cot", selection_mode="n/a",
        data_delivery_mode="static", task_kind="generation",
    )
    result = RunResult(run_id="r", run_dir=tmp_path, config=None)
    result.tasks.append(TaskResult(identity=identity, output_dir=task_dir))

    frame = build_no_answer_frame(result)
    row = frame.iloc[0].to_dict()
    assert row["replies"] == 4
    assert row["no_answer"] == 2
    assert row["no_answer_rate"] == 0.5
    assert row["cut_off_before_answering"] == 1
    assert row["empty_reply"] == 1
    assert row["failed_calls_not_counted"] == 1
