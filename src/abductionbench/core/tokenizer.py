"""Pluggable input-token counting.

Token counts drive the input-size policy (``engine.limits.input_token_budget``)
and the per-model context check.  Which counter is used is configuration:

* ``hf``        -- the served model's own tokenizer (exact, needs ``transformers``);
* ``tiktoken``  -- BPE approximation, no model download;
* ``endpoint``  -- ask the vLLM server itself (``POST /tokenize``); exact, but
                   costs a round trip per prompt, so it is opt-in;
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

    def count_text(self, text: str) -> int: ...

    def count_messages(self, messages: list[ChatMessage]) -> int: ...


class _BaseCounter:
    backend = "base"

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

    def count_messages(self, messages: list[ChatMessage]) -> int:
        """Use the real chat template when the tokenizer ships one."""
        try:
            rendered = self._tokenizer.apply_chat_template(
                [m.to_dict() for m in messages], tokenize=True, add_generation_prompt=True
            )
            return len(rendered) + self._config.safety_margin_tokens
        except Exception:  # no chat template / unexpected signature
            return super().count_messages(messages)


class CachingCounter:
    """Memoizes counts for repeated strings (few-shot blocks, system prompts)."""

    def __init__(self, inner: TokenCounter, max_entries: int = 4096):
        self._inner = inner
        self._cache: dict[str, int] = {}
        self._max = max_entries
        self.backend = inner.backend

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
                # The endpoint counter is created by the client (it needs an
                # HTTP session); the engine substitutes it when configured.
                counter = HeuristicCounter(config)
            else:
                raise ValueError(f"unknown tokenizer backend {backend!r}")
            logger.info("token counting backend: %s", counter.backend)
            return CachingCounter(counter)
        except Exception as exc:  # try the next candidate
            last_error = exc
            logger.warning("token counter %r unavailable (%s)", backend, exc)
    raise RuntimeError(f"no usable token counter: {last_error}")
