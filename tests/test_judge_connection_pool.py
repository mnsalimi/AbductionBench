"""A judge may not be asked for more sockets than its pool holds.

This is not a tuning preference. Past the pool, httpx does not slow the
surplus down -- it waits for a connection and raises `PoolTimeout` at
`engine.timeouts.pool_s`, and `PoolTimeout` stringifies to nothing, so the
run logs "timeout talking to <url>:" with an empty reason and the endpoint
takes the blame. Each casualty then retries into the same exhausted pool.

Measured on the run that prompted the check: judge 48 + reasoning judge 32 =
80 against the default pool of 64 gave 744 such failures in twenty minutes,
while the tasks waiting on those verdicts held every task slot and two of the
three GPUs sat at zero requests.
"""

from __future__ import annotations

import pytest

from abductionbench.core.config import ConfigError, RunConfig


def _body(*, pool: int, judge_calls: int, reasoning_calls: int) -> dict:
    return {
        "name": "t",
        "engine": {
            "judge": {
                "enabled": True,
                "model": "j",
                "max_parallel_calls": judge_calls,
            },
            "reasoning_judge": {
                "enabled": True,
                "model": "j",
                "max_parallel_calls": reasoning_calls,
            },
        },
        "models": [
            {
                "id": "j",
                "model_name": "openai/gpt-oss-120b",
                "judge_only": True,
                "endpoint": {"base_url": "https://example.invalid/api"},
                "limits": {"max_connections": pool},
            }
        ],
        "datasets": [],
        "prompts": {},
    }


def test_a_pool_smaller_than_both_judges_together_is_refused():
    """The exact shape that produced the timeouts: 48 + 32 against 64."""
    with pytest.raises(ConfigError) as excinfo:
        RunConfig.model_validate(_body(pool=64, judge_calls=48, reasoning_calls=32))
    message = str(excinfo.value)
    # The number it must be raised to, not just "too small".
    assert "80 concurrent" in message
    assert "holds 64" in message
    assert "max_connections" in message


def test_the_sum_is_what_counts_not_either_stage_alone():
    """Each stage fits on its own; they share one client, so neither is the test."""
    body = _body(pool=64, judge_calls=48, reasoning_calls=32)
    assert body["engine"]["judge"]["max_parallel_calls"] < 64
    assert body["engine"]["reasoning_judge"]["max_parallel_calls"] < 64
    with pytest.raises(ConfigError):
        RunConfig.model_validate(body)


def test_a_pool_that_covers_them_is_accepted():
    config = RunConfig.model_validate(_body(pool=160, judge_calls=48, reasoning_calls=32))
    assert config.models[0].limits.max_connections == 160


def test_a_disabled_stage_does_not_count_toward_the_pool():
    """Only what can actually be in flight."""
    body = _body(pool=64, judge_calls=48, reasoning_calls=32)
    body["engine"]["reasoning_judge"]["enabled"] = False
    config = RunConfig.model_validate(body)
    assert config.engine.judge.max_parallel_calls == 48


def test_a_model_under_test_is_not_measured_against_the_judges():
    """It is not the judge, so the judges' concurrency is not aimed at it."""
    body = _body(pool=160, judge_calls=48, reasoning_calls=32)
    body["models"].append(
        {
            "id": "m",
            "model_name": "local/model",
            "endpoint": {"base_url": "http://127.0.0.1:18000"},
            "limits": {"max_connections": 8, "max_parallel_batches": 8},
        }
    )
    config = RunConfig.model_validate(body)
    assert config.models[1].limits.max_connections == 8


def test_the_shipped_openrouter_judge_covers_the_shipped_run():
    """The two files have to agree, and nothing else checks that they do."""
    from abductionbench.core.config import load_layered

    model = load_layered("configs/models/gpt-oss-120b-openrouter.yaml")["model"]
    run = load_layered("configs/runs/reasoning_openrouter.yaml")["engine"]
    asked = (
        int(str(run["judge"]["max_parallel_calls"]).split("|")[-1].rstrip("}"))
        + int(str(run["reasoning_judge"]["max_parallel_calls"]).split("|")[-1].rstrip("}"))
    )
    assert model["limits"]["max_connections"] >= asked


def test_an_evaluated_models_pool_must_hold_its_own_requests():
    """The same defect on the other side of the run.

    In-flight requests for a model are `max_parallel_batches` -- and where the
    endpoint has no batch route, a "batch" IS one request. A pool smaller than
    that produces the identical empty-message PoolTimeout, hours into a run,
    looking like a flaky provider.
    """
    body = _body(pool=160, judge_calls=48, reasoning_calls=32)
    body["models"].append(
        {
            "id": "under-test",
            "model_name": "openai/gpt-5.6-luna",
            "endpoint": {"base_url": "https://openrouter.ai/api"},
            "limits": {"max_parallel_batches": 64, "max_connections": 32},
        }
    )
    with pytest.raises(ConfigError) as excinfo:
        RunConfig.model_validate(body)
    message = str(excinfo.value)
    assert "under-test" in message
    assert "64 request(s) in flight" in message
    assert "holds 32" in message


def test_a_model_whose_pool_covers_its_requests_is_accepted():
    body = _body(pool=160, judge_calls=48, reasoning_calls=32)
    body["models"].append(
        {
            "id": "under-test",
            "model_name": "openai/gpt-5.6-luna",
            "endpoint": {"base_url": "https://openrouter.ai/api"},
            "limits": {"max_parallel_batches": 64, "max_connections": 128},
        }
    )
    config = RunConfig.model_validate(body)
    assert config.models[1].limits.max_parallel_batches == 64


def test_the_shipped_luna_config_can_actually_run_at_its_stated_parallelism():
    """The two numbers live in one file and still have to agree."""
    from abductionbench.core.config import load_layered

    limits = load_layered("configs/models/gpt-5.6-luna-openrouter.yaml")["model"]["limits"]
    asked = int(str(limits["max_parallel_batches"]).split("|")[-1].rstrip("}"))
    assert asked >= 64, "luna must not be the bottleneck of a two-model run"
    assert limits["max_connections"] >= asked


def test_the_generate_only_run_lets_both_models_reach_their_ceilings():
    """A global cap below the sum silently becomes the real limit."""
    from abductionbench.core.config import load_layered

    run = load_layered("configs/runs/generate_only.yaml")
    local = load_layered("configs/models/gemma-4-31b-local.yaml")["model"]["limits"]
    luna = load_layered("configs/models/gpt-5.6-luna-openrouter.yaml")["model"]["limits"]
    wanted = local["max_parallel_batches"] + int(
        str(luna["max_parallel_batches"]).split("|")[-1].rstrip("}")
    )
    assert run["engine"]["concurrency"]["max_parallel_batches_global"] >= wanted


def test_generate_only_defers_both_judges_rather_than_silently_dropping_them():
    """`enabled: false` alone is refused on judged datasets, and should be."""
    from abductionbench.core.config import load_layered

    engine = load_layered("configs/runs/generate_only.yaml")["engine"]
    assert engine["judge"]["enabled"] is False
    assert engine["judge"]["defer"] is True
    assert engine["reasoning_judge"]["enabled"] is False
