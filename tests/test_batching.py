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


def test_quantization_rounds_down_onto_the_grid():
    """Down, not up: the budget is what is left of the context window."""
    assert quantize_max_tokens(1, 256) == 256      # never below one quantum
    assert quantize_max_tokens(256, 256) == 256
    assert quantize_max_tokens(511, 256) == 256    # rounding up would overrun
    assert quantize_max_tokens(512, 256) == 512
    assert quantize_max_tokens(300, 1) == 300


def test_output_budget_is_the_rest_of_the_context_window():
    """The budget is computed, not requested: cap vs what the prompt left over."""
    batching = BatchingConfig(max_tokens_quantum=64)
    model = ModelSamplingConfig(
        temperature=0.0, top_p=1.0, max_tokens_default=32_000, max_tokens_cap=32_000
    )

    # No window known: the suite-wide 32,000 ceiling applies.
    params = resolve_sampling(
        per_sample_overrides={},
        model_sampling=model,
        template_sampling={},
        batching=batching,
    )
    assert params.max_tokens == 32_000

    # With a window, the budget is what is left of it after the prompt, and it
    # never exceeds that -- the quantum rounds down, so it cannot overrun.
    params = resolve_sampling(
        per_sample_overrides={},
        model_sampling=model,
        template_sampling={},
        batching=batching,
        context_window=1000,
        input_tokens=900,
    )
    # A prompt that nearly fills the window leaves nothing: the budget collapses
    # rather than overshooting, and the engine skips such prompts for this model.
    assert params.max_tokens <= 1000 - 900

    # With room to spare, the budget is the window less the prompt and a reserve,
    # rounded down onto the grid -- never more than the window can hold.
    params = resolve_sampling(
        per_sample_overrides={},
        model_sampling=model,
        template_sampling={},
        batching=batching,
        context_window=16384,
        input_tokens=2049,
    )
    assert params.max_tokens % 64 == 0
    assert params.max_tokens + 2049 < 16384

    # An adapter cannot ask for more (or less): only decoding params layer.
    params = resolve_sampling(
        per_sample_overrides={"max_tokens": 40, "temperature": 0.9},
        model_sampling=model,
        template_sampling={"stop": ["</s>"], "temperature": 0.5},
        batching=batching,
        context_window=4096,
        input_tokens=96,
    )
    assert params.max_tokens > 40           # the override is ignored by design
    assert params.max_tokens <= 4096 - 96
    assert params.temperature == 0.9        # sample override still beats template
    assert params.stop == ("</s>",)


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
