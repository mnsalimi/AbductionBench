"""Sampling resolution, batch packing and bisection."""

from __future__ import annotations

from abductionbench.core.batching import (
    Batch,
    bisect,
    plan_batches,
    quantize_max_tokens,
    resolve_sampling,
)
from abductionbench.core.config import BatchingConfig, ModelSamplingConfig
from abductionbench.core.types import ChatMessage, RenderedPrompt, SampleSpec, SamplingParams


def _prompt(sample_id: str, max_tokens: int, tokens: int = 100) -> RenderedPrompt:
    return RenderedPrompt(
        sample=SampleSpec(sample_id=sample_id, fields={}),
        messages=[ChatMessage("user", "hi")],
        template_id="t",
        template_version="1.0",
        sampling=SamplingParams(max_tokens=max_tokens),
        input_tokens_est=tokens,
    )


def test_quantization_rounds_up():
    assert quantize_max_tokens(1, 256) == 256
    assert quantize_max_tokens(256, 256) == 256
    assert quantize_max_tokens(257, 256) == 512
    assert quantize_max_tokens(300, 1) == 300


def test_resolve_sampling_precedence_and_clamping():
    batching = BatchingConfig(max_tokens_quantum=64)
    model = ModelSamplingConfig(
        temperature=0.0, top_p=1.0, max_tokens_default=100, max_tokens_cap=256, max_tokens_floor=32
    )
    # adapter request is quantized up, then capped
    params = resolve_sampling(
        requested_max_tokens=1000,
        per_sample_overrides={},
        model_sampling=model,
        template_sampling={},
        batching=batching,
    )
    assert params.max_tokens == 256

    # per-sample override beats the adapter request; template sampling applies
    params = resolve_sampling(
        requested_max_tokens=1000,
        per_sample_overrides={"max_tokens": 40, "temperature": 0.9},
        model_sampling=model,
        template_sampling={"stop": ["</s>"], "temperature": 0.5},
        batching=batching,
    )
    assert params.max_tokens == 64  # 40 -> quantized to 64
    assert params.temperature == 0.9  # sample override beats template
    assert params.stop == ("</s>",)

    # context window limits the output budget
    params = resolve_sampling(
        requested_max_tokens=256,
        per_sample_overrides={},
        model_sampling=model,
        template_sampling={},
        batching=batching,
        context_window=1000,
        input_tokens=900,
    )
    assert params.max_tokens <= 92


def test_plan_batches_groups_by_signature_and_size():
    prompts = [_prompt(f"a{i}", 128) for i in range(5)] + [_prompt(f"b{i}", 512) for i in range(3)]
    batches = plan_batches(prompts, group_size=2, batching=BatchingConfig())
    # 5 items of one signature -> 3 batches; 3 of the other -> 2 batches
    assert [b.size for b in batches] == [2, 2, 1, 2, 1]
    for batch in batches:
        signatures = {p.sampling.signature() for p in batch.prompts}
        assert len(signatures) == 1, "a batch call cannot mix sampling parameters"


def test_plan_batches_is_deterministic_and_sorts_by_tokens():
    prompts = [_prompt("a", 128, tokens=500), _prompt("b", 128, tokens=10), _prompt("c", 128, tokens=100)]
    first = plan_batches(prompts, group_size=3, batching=BatchingConfig(sort_by_input_tokens=True))
    second = plan_batches(prompts, group_size=3, batching=BatchingConfig(sort_by_input_tokens=True))
    assert first[0].sample_ids == ["b", "c", "a"]
    assert first[0].sample_ids == second[0].sample_ids
    assert first[0].batch_id == second[0].batch_id


def test_group_size_respects_global_cap():
    prompts = [_prompt(f"x{i}", 128) for i in range(10)]
    batches = plan_batches(prompts, group_size=100, batching=BatchingConfig(max_group_size=4))
    assert max(b.size for b in batches) == 4


def test_bisect_splits_and_terminates():
    batch = Batch(batch_id="b", prompts=[_prompt(f"x{i}", 128) for i in range(5)],
                  sampling=SamplingParams(max_tokens=128))
    halves = bisect(batch)
    assert [h.size for h in halves] == [2, 3]
    assert all(h.depth == 1 for h in halves)
    assert bisect(Batch(batch_id="s", prompts=[_prompt("only", 128)],
                        sampling=SamplingParams(max_tokens=128))) == []
