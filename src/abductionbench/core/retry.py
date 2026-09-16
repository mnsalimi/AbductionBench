"""Retry policy and endpoint recovery.

The remote GPU endpoints are reached over Cloudflare quick tunnels, which drop,
rate-limit, and *change URL* whenever the tunnel process restarts.  The policy
here separates three reactions:

1. **Wait and retry** -- transient 5xx, timeouts, connection resets, 429.
2. **Recover the endpoint** -- on a connection-level failure, re-run the model's
   configured discovery command to pick up a rotated tunnel URL, then probe
   ``/v1/models`` until the endpoint answers again.
3. **Do not retry as-is** -- auth failures (abort) and invalid-request /
   context-length failures (isolate the offending sample by splitting the
   batch, since vLLM fails an entire batch call when one conversation is bad).
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from .config import RetryConfig
from .errors import AbenchError, EndpointError, ErrorClass

logger = logging.getLogger(__name__)

T = TypeVar("T")

__all__ = ["RetryPolicy", "RetryOutcome", "with_retry"]


@dataclass(slots=True)
class RetryOutcome:
    """Bookkeeping about how many attempts a call needed."""

    attempts: int = 0
    total_wait_s: float = 0.0
    last_error: BaseException | None = None
    error_class: ErrorClass | None = None


class RetryPolicy:
    """Decides whether and how long to wait before the next attempt."""

    def __init__(self, config: RetryConfig):
        self.config = config
        self._retryable = {ErrorClass(v) for v in config.retry_error_classes}

    def error_class_of(self, exc: BaseException) -> ErrorClass:
        if isinstance(exc, EndpointError):
            return exc.error_class
        if isinstance(exc, AbenchError):
            return exc.error_class
        if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
            return ErrorClass.TRANSIENT
        if isinstance(exc, (ConnectionError, OSError)):
            return ErrorClass.TRANSIENT
        return ErrorClass.UNKNOWN

    def should_retry(self, exc: BaseException, attempt: int) -> bool:
        if attempt >= self.config.max_attempts:
            return False
        return self.error_class_of(exc) in self._retryable

    def backoff_for(self, exc: BaseException, attempt: int) -> float:
        """Exponential backoff with jitter; 429 gets its own, longer base."""
        error_class = self.error_class_of(exc)
        if error_class is ErrorClass.RATE_LIMIT:
            base = self.config.rate_limit_backoff_s
        else:
            base = self.config.initial_backoff_s
        delay = min(
            base * (self.config.backoff_multiplier ** max(0, attempt - 1)),
            self.config.max_backoff_s,
        )
        if self.config.jitter:
            spread = delay * self.config.jitter
            delay = max(0.0, delay + random.uniform(-spread, spread))
        return delay


async def with_retry(
    operation: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy,
    description: str,
    on_recover: Callable[[BaseException], Awaitable[bool]] | None = None,
    on_attempt_failed: Callable[[BaseException, int, float], None] | None = None,
) -> tuple[T, RetryOutcome]:
    """Run ``operation`` under ``policy``.

    Parameters
    ----------
    operation:
        Zero-argument coroutine function; retried as a whole.
    on_recover:
        Called before sleeping when the failure looks connection-level.  Should
        attempt endpoint re-discovery/probing and return ``True`` if the
        endpoint is believed healthy again (in which case the backoff sleep is
        skipped, because recovery already took real time).
    on_attempt_failed:
        Observer hook ``(exc, attempt, sleep_s)`` used for telemetry.

    Returns
    -------
    ``(result, outcome)`` on success; re-raises the last exception when the
    policy is exhausted.
    """
    outcome = RetryOutcome()
    attempt = 0
    while True:
        attempt += 1
        outcome.attempts = attempt
        try:
            result = await operation()
            return result, outcome
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001 - classified below
            outcome.last_error = exc
            outcome.error_class = policy.error_class_of(exc)
            if not policy.should_retry(exc, attempt):
                logger.error(
                    "%s failed permanently after %d attempt(s) [%s]: %s",
                    description,
                    attempt,
                    outcome.error_class.value if outcome.error_class else "?",
                    exc,
                )
                raise

            recovered = False
            if on_recover is not None and outcome.error_class is ErrorClass.TRANSIENT:
                try:
                    recovered = await on_recover(exc)
                except asyncio.CancelledError:
                    raise
                except Exception as recovery_exc:  # recovery is best-effort
                    logger.warning("%s: endpoint recovery failed: %s", description, recovery_exc)

            sleep_s = 0.0 if recovered else policy.backoff_for(exc, attempt)
            outcome.total_wait_s += sleep_s
            if on_attempt_failed is not None:
                on_attempt_failed(exc, attempt, sleep_s)
            logger.warning(
                "%s attempt %d/%d failed [%s]: %s -- retrying in %.1fs",
                description,
                attempt,
                policy.config.max_attempts,
                outcome.error_class.value if outcome.error_class else "?",
                _short(exc),
                sleep_s,
            )
            if sleep_s:
                await asyncio.sleep(sleep_s)


def _short(exc: BaseException, limit: int = 300) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text if len(text) <= limit else text[:limit] + "..."


def rate_limit_sleep(requests_per_minute: float | None) -> float:
    """Minimum spacing between requests implied by an RPM cap."""
    if not requests_per_minute or requests_per_minute <= 0:
        return 0.0
    return 60.0 / float(requests_per_minute)


def summarize(outcome: RetryOutcome) -> dict[str, Any]:
    return {
        "attempts": outcome.attempts,
        "total_wait_s": round(outcome.total_wait_s, 2),
        "error_class": outcome.error_class.value if outcome.error_class else None,
        "last_error": str(outcome.last_error) if outcome.last_error else None,
    }
