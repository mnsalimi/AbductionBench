"""What fits in a judge's context window, counted in the judge's own tokens.

Both judges used to bound what they were sent in CHARACTERS -- 60,000 for the
exchange, 60,000 for a chain -- sized so the densest text this suite produces
(2.72 characters per token, abd's s-expressions) could never overrun a 65,536
token window. That made the limit safe only by making it small: ordinary prose
runs ~4.6 characters per token, so a 60,000-character answer is ~13,000 tokens
and the judge's window was two-thirds unused on exactly the long answers the
limit turned away. And a character count cannot know the window it guards, so
the same number applied whether the judge had 65,536 tokens or 131,072.

The limit now IS the window: every field is counted with the judge's own
tokenizer, and a request is skipped only when those tokens plus the prompt's
wording plus the room the reply needs would not fit.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Iterable

logger = logging.getLogger(__name__)

#: Characters per token at this suite's densest, measured with gpt-oss's own
#: tokenizer over 13 datasets. Used only when the tokenizer cannot be loaded,
#: and deliberately the worst case: an over-count skips a sample that would
#: have fit, an under-count sends one the server rejects.
DENSEST_CHARS_PER_TOKEN = 2.72

#: Held back from every window for the chat template's own tokens (role
#: markers, the reasoning preamble) and for any disagreement between this
#: count and the server's.
SAFETY_TOKENS = 512


@functools.lru_cache(maxsize=8)
def _tokenizer(model_name: str) -> Any | None:
    """The model's tokenizer from the local Hugging Face cache, or ``None``.

    ``local_files_only``: a judge served here has its tokenizer on disk, and a
    rented one that does not must not turn a size check into a download.
    """
    if not model_name:
        return None
    try:
        from transformers import AutoTokenizer  # heavy; imported only when used

        return AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception as exc:  # noqa: BLE001 - any failure means "estimate instead"
        logger.info(
            "judge budget: no local tokenizer for %s (%s); estimating at %.2f chars/token",
            model_name, type(exc).__name__, DENSEST_CHARS_PER_TOKEN,
        )
        return None


class JudgeWindow:
    """One judge's context window, and a way to ask whether a request fits it."""

    def __init__(self, model_name: str, context_window: int | None):
        self.model_name = model_name
        self.window = int(context_window or 0)
        self._tokenizer = _tokenizer(model_name)
        #: Whether counts come from the judge's own tokenizer.
        self.exact = self._tokenizer is not None

    @classmethod
    def for_client(cls, client: Any, override: int | None = None) -> JudgeWindow:
        """The window of the model behind ``client`` (or ``override``)."""
        model = getattr(client, "model", None)
        name = str(getattr(model, "model_name", "") or "")
        limits = getattr(model, "limits", None)
        window = override or getattr(limits, "context_window", None)
        return cls(name, window)

    def count(self, text: Any) -> int:
        if not text:
            return 0
        text = str(text)
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text, add_special_tokens=False))
        return int(len(text) / DENSEST_CHARS_PER_TOKEN) + 1

    def overrun(self, parts: Iterable[Any], *, reserved: int) -> str | None:
        """Why ``parts`` plus ``reserved`` tokens will not fit, or ``None``.

        ``reserved`` is everything that is not a field: the prompt's own
        wording and the reply's budget. No window configured is no limit.
        """
        if not self.window:
            return None
        used = sum(self.count(part) for part in parts if part) + reserved + SAFETY_TOKENS
        if used > self.window:
            return f"request_needs_{used}_tokens_of_a_{self.window}_token_window"
        return None


def template_tokens(window: JudgeWindow, template: Any) -> int:
    """Tokens of a judge template's own wording, placeholders included."""
    messages = getattr(template, "messages", None) or []
    text = "\n".join(
        str(message.get("content", "") if isinstance(message, dict) else getattr(message, "content", ""))
        for message in messages
    )
    return window.count(text)
