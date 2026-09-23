"""Stopping a run without destroying the work that is in flight.

The failure this replaces: an interrupted run closed every HTTP client as soon
as its task gather unwound, while interactive episodes, retries and simulator
calls were still running. Each of those then failed with "Cannot send a
request, as the client has been closed", was retried against the closed client,
and was finally written down as a *model* error -- 150 vivabench episodes and
7,651 seconds of retries on one interrupt, all charged to the model under
evaluation.

A stop is two phases:

1. **Drain.** Nothing new starts: no task, no batch, no retry, no endpoint
   recovery, no simulator retry. Work already in flight -- a request on the
   wire, an episode part-way through -- is allowed to finish, up to
   ``engine.shutdown.drain_timeout_s``. Everything that finishes is recorded
   exactly as it would have been.
2. **Cancel.** Whatever is still running when the drain window closes (or at
   once, on a second signal) is cancelled and *awaited*. It is recorded as
   interrupted, never as a model error, and stays pending in its checkpoint so
   a resume runs it again.

Only then are the clients closed -- each exactly once -- and the artifacts
flushed. Nothing is left running that could reach a closed client.
"""

from __future__ import annotations

import asyncio
import logging
import time

logger = logging.getLogger(__name__)

__all__ = ["RunInterrupted", "Shutdown"]


class RunInterrupted(BaseException):  # noqa: N818 - named for what happened, like CancelledError
    """Work abandoned because the run is stopping. Not a failure of anything.

    A ``BaseException`` for the same reason ``asyncio.CancelledError`` is one:
    the engine is full of ``except Exception`` handlers whose job is to turn a
    failure into a recorded error so one bad sample cannot stop a run. An
    interruption must pass straight through every one of them -- caught, it
    would be written down as exactly the model error this exists to prevent.
    """


class Shutdown:
    """Whether the run has been asked to stop, and how urgently."""

    def __init__(self) -> None:
        self._requested = asyncio.Event()
        self._forced = asyncio.Event()
        self.reason: str | None = None
        self.requested_at: float | None = None

    @property
    def requested(self) -> bool:
        return self._requested.is_set()

    @property
    def forced(self) -> bool:
        return self._forced.is_set()

    def request(self, reason: str) -> None:
        """Begin the drain. A second request skips what is left of it."""
        if not self.requested:
            self.reason = reason
            self.requested_at = time.time()
            self._requested.set()
            logger.warning(
                "shutdown requested (%s): starting nothing new, letting in-flight work "
                "finish; request again to cancel it now",
                reason,
            )
            return
        if not self.forced:
            self._forced.set()
            logger.warning("shutdown forced (%s): cancelling in-flight work now", reason)

    def force(self, reason: str) -> None:
        """Skip the drain entirely."""
        if not self.requested:
            self.request(reason)
        self.request(reason)

    async def wait(self) -> None:
        await self._requested.wait()

    async def wait_forced(self) -> None:
        await self._forced.wait()

    def check(self, what: str) -> None:
        """Raise :class:`RunInterrupted` instead of starting ``what``."""
        if self.requested:
            raise RunInterrupted(f"{what}: not started, the run is stopping ({self.reason})")

    async def sleep(self, seconds: float, what: str) -> None:
        """A backoff that a stop request ends early.

        A retry waiting out a 429 can sleep for minutes; the drain window would
        be spent waiting for it to wake up and then refuse to retry.
        """
        if seconds <= 0:
            self.check(what)
            return
        try:
            await asyncio.wait_for(self._requested.wait(), timeout=seconds)
        except TimeoutError:
            return
        self.check(what)
