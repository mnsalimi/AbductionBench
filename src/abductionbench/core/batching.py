"""Packing samples into native batch calls.

Constraint that shapes this module: vLLM's ``/v1/chat/completions/batch``
applies **one** sampling-parameter set to every conversation in the call.  So a
batch may only contain samples whose :meth:`SamplingParams.signature` matches.
Because adapters set a per-sample ``max_tokens`` (task-complexity dependent),
naively distinct budgets would shatter batches into singletons -- hence
``max_tokens`` is quantized upward onto a configurable grid
(``engine.batching.max_tokens_quantum``) before grouping.  Quantizing *up* only
ever grants more output headroom, never less, so it cannot truncate a response
the adapter expected to fit.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from .config import BatchingConfig, ModelSamplingConfig
from .types import RenderedPrompt, SamplingParams

__all__ = ["quantize_max_tokens", "resolve_sampling", "Batch", "plan_batches", "bisect"]


def quantize_max_tokens(value: int, quantum: int) -> int:
    """Round ``value`` up to the next multiple of ``quantum`` (>= quantum)."""
    if quantum <= 1:
        return value
    return max(quantum, int(math.ceil(value / quantum) * quantum))


def resolve_sampling(
    *,
    requested_max_tokens: int | None,
    per_sample_overrides: dict,
    model_sampling: ModelSamplingConfig,
    template_sampling: dict,
    batching: BatchingConfig,
    context_window: int | None = None,
    input_tokens: int = 0,
) -> SamplingParams:
    """Compose the effective decoding parameters for one sample.

    Precedence, lowest to highest: model defaults → template ``sampling`` →
    adapter per-sample overrides.  The adapter's ``max_tokens`` request is
    quantized up, then clamped to the model's floor/cap and, when a context
    window is known, to what actually fits alongside the prompt.
    """
    max_tokens = requested_max_tokens or model_sampling.max_tokens_default
    if "max_tokens" in per_sample_overrides:
        max_tokens = int(per_sample_overrides["max_tokens"])
    max_tokens = quantize_max_tokens(int(max_tokens), batching.max_tokens_quantum)
    max_tokens = max(model_sampling.max_tokens_floor, min(max_tokens, model_sampling.max_tokens_cap))
    if context_window:
        room = context_window - input_tokens - 8  # small slack for template scaffolding
        if room > 0:
            max_tokens = min(max_tokens, room)
        max_tokens = max(1, max_tokens)

    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=model_sampling.temperature,
        top_p=model_sampling.top_p,
        seed=model_sampling.seed,
        stop=tuple(model_sampling.stop),
    )
    # Template-level hints, then per-sample overrides.
    template_overrides = {k: v for k, v in (template_sampling or {}).items() if k != "max_tokens"}
    if template_overrides:
        params = params.merged(**template_overrides)
    sample_overrides = {k: v for k, v in (per_sample_overrides or {}).items() if k != "max_tokens"}
    if sample_overrides:
        params = params.merged(**sample_overrides)
    return params


@dataclass(slots=True)
class Batch:
    """A group of samples submitted in one native batch call."""

    batch_id: str
    prompts: list[RenderedPrompt]
    sampling: SamplingParams
    #: Bisection depth: 0 for an original batch, incremented for each split.
    depth: int = 0
    attempt_history: list[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.prompts)

    @property
    def sample_ids(self) -> list[str]:
        return [p.sample_id for p in self.prompts]

    @property
    def input_tokens_est(self) -> int:
        return sum(p.input_tokens_est for p in self.prompts)

    def conversations(self) -> list[list]:
        return [p.messages for p in self.prompts]


def plan_batches(
    prompts: Iterable[RenderedPrompt],
    *,
    group_size: int,
    batching: BatchingConfig,
    prefix: str = "b",
) -> list[Batch]:
    """Group rendered prompts into deterministic batches.

    Samples are first partitioned by sampling signature (a hard requirement of
    the batch endpoint), then optionally sorted by estimated input length so
    that conversations inside one call have similar cost -- a batch call returns
    only when its slowest conversation finishes, so mixing a 200-token and a
    15k-token prompt wastes wall-clock.

    Ordering is stable given the same input, which keeps runs reproducible and
    makes resume deterministic.
    """
    effective_size = max(1, min(group_size, batching.max_group_size))
    partitions: dict[str, list[RenderedPrompt]] = {}
    order: list[str] = []
    for prompt in prompts:
        signature = prompt.sampling.signature()
        if signature not in partitions:
            partitions[signature] = []
            order.append(signature)
        partitions[signature].append(prompt)

    batches: list[Batch] = []
    for signature in order:
        group = partitions[signature]
        if batching.sort_by_input_tokens:
            group = sorted(group, key=lambda p: (p.input_tokens_est, p.sample_id))
        for chunk_index in range(0, len(group), effective_size):
            chunk = group[chunk_index : chunk_index + effective_size]
            batches.append(
                Batch(
                    batch_id=f"{prefix}-{signature[:8]}-{chunk_index // effective_size:04d}",
                    prompts=chunk,
                    sampling=chunk[0].sampling,
                )
            )
    return batches


def bisect(batch: Batch) -> list[Batch]:
    """Split a failed batch in half to isolate an offending sample.

    Used when a batch call fails with an invalid-request/context-length error:
    the server rejects the *entire* call because of one bad conversation, so the
    only way to salvage the rest is to narrow down which conversation it was.
    A size-1 batch cannot be split further and is reported as a sample-level
    failure by the caller.
    """
    if batch.size <= 1:
        return []
    middle = batch.size // 2
    halves = [batch.prompts[:middle], batch.prompts[middle:]]
    return [
        Batch(
            batch_id=f"{batch.batch_id}.{index}",
            prompts=half,
            sampling=batch.sampling,
            depth=batch.depth + 1,
            attempt_history=list(batch.attempt_history),
        )
        for index, half in enumerate(halves)
        if half
    ]


def iter_chunks(items: list, size: int) -> Iterator[list]:
    """Yield fixed-size chunks (used by the judge stage)."""
    step = max(1, size)
    for start in range(0, len(items), step):
        yield items[start : start + step]
