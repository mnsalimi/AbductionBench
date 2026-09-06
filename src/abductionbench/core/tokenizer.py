"""Pluggable input-token counting.

Token counts drive the input-size policy (``engine.limits.input_token_budget``)
and the per-model context check.  Which counter is used is configuration:

* ``hf``        -- the served model's own tokenizer (exact, needs ``transformers``);
* ``tiktoken``  -- BPE approximation, no model download;
* ``endpoint``  -- not implemented; asks the caller to use ``hf`` instead
                   rather than quietly degrading to a guess;
* ``heuristic`` -- characters / ``chars_per_token``; dependency-free fallback;
* ``auto``      -- hf → tiktoken → heuristic, whichever is available.

Counting is intentionally conservative: chat-template scaffolding is
approximated with a per-message overhead and a global safety margin, so the
engine errs towards declaring a prompt oversize rather than letting the server
reject it mid-batch (which would fail the whole batch call).
"""

from __future__ import annotations

import logging
from typing import Protocol

from .config import TokenizerConfig
from .types import ChatMessage

logger = logging.getLogger(__name__)

__all__ = ["TokenCounter", "build_token_counter"]


class TokenCounter(Protocol):
    """Counts tokens of a rendered conversation."""

    backend: str
    exact: bool

    def count_text(self, text: str) -> int: ...

    def count_messages(self, messages: list[ChatMessage]) -> int: ...


class _BaseCounter:
    backend = "base"
    #: Approximate counters need the engine to hold back more context.
    exact = False

    def __init__(self, config: TokenizerConfig):
        self._config = config

    def count_text(self, text: str) -> int:  # pragma: no cover - overridden
        raise NotImplementedError

    def count_messages(self, messages: list[ChatMessage]) -> int:
        total = 0
        for message in messages:
            total += self.count_text(message.content)
            total += self.count_text(message.role)
            total += self._config.per_message_overhead
        return total + self._config.safety_margin_tokens


class HeuristicCounter(_BaseCounter):
    """Character-length heuristic; always available."""

    backend = "heuristic"

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        return int(len(text) / self._config.chars_per_token) + 1


class TiktokenCounter(_BaseCounter):
    backend = "tiktoken"

    def __init__(self, config: TokenizerConfig):
        super().__init__(config)
        import tiktoken

        try:
            self._encoding = tiktoken.get_encoding(config.tiktoken_encoding)
        except Exception:  # unknown encoding name -> fall back to a known one
            logger.warning(
                "tiktoken encoding %r unavailable; falling back to cl100k_base",
                config.tiktoken_encoding,
            )
            self._encoding = tiktoken.get_encoding("cl100k_base")

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        return len(self._encoding.encode(text, disallowed_special=()))


class HFCounter(_BaseCounter):
    """Exact counting with the served model's own tokenizer."""

    backend = "hf"

    def __init__(self, config: TokenizerConfig):
        super().__init__(config)
        if not config.hf_model:
            raise ValueError("tokenizer.backend='hf' requires tokenizer.hf_model")
        from transformers import AutoTokenizer  # imported lazily: heavy dependency

        self._tokenizer = AutoTokenizer.from_pretrained(config.hf_model, trust_remote_code=True)

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    #: Whether this counter matches what the server will count.  The engine
    #: reserves less context when the count is exact.
    exact = True

    def count_messages(self, messages: list[ChatMessage]) -> int:
        """Use the real chat template when the tokenizer ships one."""
        try:
            rendered = self._tokenizer.apply_chat_template(
                [m.to_dict() for m in messages], tokenize=True, add_generation_prompt=True
            )
            count = _token_count(rendered)
            if count is None:
                raise TypeError(f"unexpected chat-template result: {type(rendered).__name__}")
            return count + self._config.safety_margin_tokens
        except Exception as exc:  # no chat template / unexpected signature
            logger.debug("chat-template counting unavailable (%s); approximating", exc)
            return super().count_messages(messages)


def _token_count(rendered: object) -> int | None:
    """Number of tokens in whatever ``apply_chat_template`` returned.

    The return type has changed across transformers versions: a flat list of ids
    in 4.x, a ``BatchEncoding`` in 5.x, sometimes a batch of one.  ``len()`` on
    the 5.x shape counts *keys* -- two -- which would tell the engine every
    prompt is two tokens long: no prompt would ever look oversize, and every
    request would ask for a whole context window of output and be rejected. So
    the shape is unwrapped explicitly rather than trusted.
    """
    ids: object = rendered
    if hasattr(ids, "keys") and "input_ids" in ids:  # BatchEncoding / dict
        ids = ids["input_ids"]
    if hasattr(ids, "tolist"):  # tensor / ndarray
        ids = ids.tolist()
    if isinstance(ids, (list, tuple)):
        if ids and isinstance(ids[0], (list, tuple)):  # a batch of one
            return len(ids[0])
        if all(isinstance(item, int) for item in ids):
            return len(ids)
    return None


class CachingCounter:
    """Memoizes counts for repeated strings (few-shot blocks, system prompts)."""

    def __init__(self, inner: TokenCounter, max_entries: int = 4096):
        self._inner = inner
        self._cache: dict[str, int] = {}
        self._max = max_entries
        self.backend = inner.backend
        self.exact = getattr(inner, "exact", False)

    def count_text(self, text: str) -> int:
        if len(text) > 4096:  # long unique bodies are not worth caching
            return self._inner.count_text(text)
        cached = self._cache.get(text)
        if cached is None:
            cached = self._inner.count_text(text)
            if len(self._cache) < self._max:
                self._cache[text] = cached
        return cached

    def count_messages(self, messages: list[ChatMessage]) -> int:
        return self._inner.count_messages(messages)


def build_token_counter(config: TokenizerConfig) -> TokenCounter:
    """Instantiate the configured counter, degrading gracefully.

    ``auto`` prefers exactness but never fails the run over a missing optional
    dependency -- it logs which backend it settled on so the run documentation
    records how token counts were obtained.
    """
    order: list[str]
    if config.backend == "auto":
        order = ["hf", "tiktoken", "heuristic"] if config.hf_model else ["tiktoken", "heuristic"]
    else:
        order = [config.backend]

    last_error: Exception | None = None
    for backend in order:
        try:
            if backend == "hf":
                counter: TokenCounter = HFCounter(config)
            elif backend == "tiktoken":
                counter = TiktokenCounter(config)
            elif backend == "heuristic":
                counter = HeuristicCounter(config)
            elif backend == "endpoint":
                # Documented as exact, but never implemented: it would need an
                # HTTP session the counter does not have. Silently handing back
                # the crudest counter to someone who asked for the exact one is
                # how a run overshoots its context window, so this says no.
                raise ValueError(
                    "tokenizer.backend='endpoint' is not implemented. Use 'hf' for exact "
                    "counts with the model's own tokenizer (set tokenizer.hf_model, or "
                    "leave it empty and the run's first model is used)."
                )
            else:
                raise ValueError(f"unknown tokenizer backend {backend!r}")
            logger.info("token counting backend: %s", counter.backend)
            return CachingCounter(counter)
        except Exception as exc:  # try the next candidate
            last_error = exc
            logger.warning("token counter %r unavailable (%s)", backend, exc)
    raise RuntimeError(f"no usable token counter: {last_error}")
