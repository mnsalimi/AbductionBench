"""The reasoning judge stops buying calls once too many of them are wasted.

On a paid API a judge that keeps returning replies the metrics cannot be
computed from spends money on nothing. After at least `kill_switch_min_samples`
samples have had a real call, more than `kill_switch_max_failure_rate` of
them failing stops the stage for the rest of the run.
"""

from __future__ import annotations

import asyncio

from abductionbench.core.config import ReasoningJudgeConfig
from abductionbench.core.reasoning_judge import ReasoningJudgeStage
from abductionbench.core.types import TaskIdentity

GOOD = '{"directionality": 1}'
BAD = "I am not sure what you want."


class _Choice:
    def __init__(self, content):
        self.content, self.reasoning, self.finish_reason = content, None, "stop"


class _Result:
    def __init__(self, content):
        self.choices, self.usage = [_Choice(content)], {"completion_tokens": 5}


class _Registry:
    def get(self, template_id):
        class _T:
            ref = f"{template_id}@1.0"
            id = template_id
            version = "1.0"
            required_fields: list = []
            output_contract = {"json_fields": {"directionality": "directionality"}}
            messages: list = []

        return _T()


class _Model:
    def __init__(self, base_url):
        class _E:
            pass

        self.endpoint = _E()
        self.endpoint.base_url = base_url
        self.model_name = "openai/gpt-oss-120b"
        self.limits = None


def _stage(tmp_path, *, bad_every=None, base_url="https://openrouter.ai/api", **config):
    config.setdefault("kill_switch", "remote")
    calls = {"n": 0}

    class _Client:
        supports_batch = False
        model = _Model(base_url)

        async def chat_single(self, messages, params):
            calls["n"] += 1
            bad = bad_every is not None and calls["n"] % bad_every == 0
            return _Result(BAD if bad else GOOD)

    class _Renderer:
        def render(self, sample, template):
            from abductionbench.core.types import ChatMessage

            return [ChatMessage(role="user", content="u")], {}

    stage = ReasoningJudgeStage(
        config=ReasoningJudgeConfig(enabled=True, model="judge", cache=False, **config),
        registry=_Registry(), renderer=_Renderer(), clients={"judge": _Client()},
        retry_policy=None, cache_dir=tmp_path / "c", run_dir=tmp_path / "run",
    )
    return stage, calls


def _judge(stage, n, *, start=0, task="t"):
    identity = TaskIdentity(run_id="r", dataset_id="d", model_id="m", template_id=task,
                            template_version="1", prompt_mode="cot", selection_mode="n/a",
                            data_delivery_mode="static", task_kind="generation")
    requests = {str(i): {"reasoning_chain": f"chain {i}"} for i in range(start, start + n)}
    context = {str(i): {"sample_id": f"s{i}"} for i in range(start, start + n)}

    async def run():
        async def fake_retry(fn, **kwargs):
            return await fn(), None

        import abductionbench.core.reasoning_judge as rj

        original, rj.with_retry = rj.with_retry, fake_retry
        try:
            return await stage._judge_many("directionality", requests, identity=identity,
                                           context=context)
        finally:
            rj.with_retry = original

    return asyncio.run(asyncio.wait_for(run(), timeout=30))


def test_it_does_not_judge_on_the_first_samples(tmp_path):
    """Even every reply failing does not trip it before the minimum."""
    stage, calls = _stage(tmp_path, bad_every=1, kill_switch_min_samples=400)
    _judge(stage, 399)
    assert calls["n"] == 399
    assert stage.kill_switch_tripped is False


def test_it_trips_over_the_limit_and_then_buys_nothing(tmp_path):
    # 1 in 20 fails = 5%, over 2%.
    stage, calls = _stage(tmp_path, bad_every=20, kill_switch_min_samples=400,
                          kill_switch_max_failure_rate=0.02, max_parallel_calls=1, group_size=1)
    _judge(stage, 400)
    made = calls["n"]
    assert stage._ks_check() is True
    out = _judge(stage, 200, start=400)
    assert calls["n"] == made, "no request may be sent once tripped"
    assert all(value is None for value in out.values())


def test_exactly_at_the_limit_does_not_trip(tmp_path):
    # 1 in 50 fails = 2.0%, not over 2%.
    stage, _calls = _stage(tmp_path, bad_every=50, kill_switch_min_samples=400,
                           kill_switch_max_failure_rate=0.02)
    _judge(stage, 400)
    assert stage._ks_check() is False


def test_a_later_task_is_marked_stopped_not_failed(tmp_path):
    from abductionbench.core.types import (
        ChatMessage, ModelResponse, RenderedPrompt, ResponseStatus, SampleScore, SampleSpec,
        SamplingParams,
    )

    stage, calls = _stage(tmp_path, bad_every=1, kill_switch_min_samples=10,
                          max_parallel_calls=1, group_size=1)
    _judge(stage, 10)
    assert stage._ks_check() is True
    made = calls["n"]

    class _Adapter:
        dataset_id = "d"

        def documentation(self):
            class _D:
                processing_mode = "selection"

            return _D()

    sample = SampleSpec(sample_id="x", fields={"observation": "o"}, reference={"gold": "g"},
                        task_kind="generation")
    prompt = RenderedPrompt(sample=sample, messages=[ChatMessage(role="user", content="q")],
                            template_id="t", template_version="1",
                            sampling=SamplingParams(max_tokens=8), input_tokens_est=1)
    response = ModelResponse(sample_id="x", model_id="m", status=ResponseStatus.OK,
                             content="<think>a. b.</think><answer>1</answer>", finish_reason="stop")
    identity = TaskIdentity(run_id="r", dataset_id="d", model_id="m", template_id="t2",
                            template_version="1", prompt_mode="cot", selection_mode="n/a",
                            data_delivery_mode="static", task_kind="generation")
    out = asyncio.run(stage.apply(_Adapter(), identity, [prompt],
                                  [(sample, response, SampleScore(metrics={}, prediction="1"))]))
    assert out[0][2].details["reasoning_metrics_status"] == "not_applicable:reasoning_judge_stopped"
    assert calls["n"] == made


def test_it_is_off_unless_a_config_turns_it_on(tmp_path):
    """The default is off: no request is ever withheld by it."""
    assert ReasoningJudgeConfig().kill_switch == "off"
    default, calls = _stage(tmp_path / "z", bad_every=1, kill_switch="off",
                            kill_switch_min_samples=10)
    _judge(default, 50)
    assert default._ks_armed is False and default.kill_switch_tripped is False
    assert calls["n"] == 50, "every request is sent"


def test_remote_mode_arms_it_for_a_rented_judge_only(tmp_path):
    remote, _ = _stage(tmp_path / "a", base_url="https://openrouter.ai/api")
    local, _ = _stage(tmp_path / "b", base_url="http://127.0.0.1:18004")
    assert remote._ks_armed is True and local._ks_armed is False
    always, _ = _stage(tmp_path / "c", base_url="http://127.0.0.1:18004", kill_switch="always")
    off, _ = _stage(tmp_path / "d", base_url="https://openrouter.ai/api", kill_switch="off")
    assert always._ks_armed is True and off._ks_armed is False
