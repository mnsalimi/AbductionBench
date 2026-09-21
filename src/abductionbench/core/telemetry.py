"""Structured logging and event telemetry.

Two sinks per run:

* a human-readable log (``engine.log``) -- what a person tails during a run;
* a JSONL event log (``events.jsonl``) -- one object per engine event
  (batch submitted/completed/retried, sample skipped, task finished), which is
  what the run report and any post-hoc analysis read.

Events are appended atomically so a killed run still leaves a valid log.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import orjson

logger = logging.getLogger(__name__)

__all__ = ["setup_logging", "EventLog", "clip"]


def clip(text: Any, limit: int = 400) -> str:
    """Shorten a value for logging."""
    if text is None:
        return ""
    body = text if isinstance(text, str) else repr(text)
    if limit and len(body) > limit:
        return body[:limit] + f"... [+{len(body) - limit} chars]"
    return body


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": record.created,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in getattr(record, "extra_fields", {}).items():
            payload[key] = value
        return orjson.dumps(payload, default=str).decode()


def setup_logging(
    log_dir: Path | None = None,
    *,
    level: str = "INFO",
    json_log: bool = True,
    quiet_console: bool = False,
) -> logging.Logger:
    """Configure root logging for a run.

    Idempotent: repeated calls replace the handlers rather than duplicating
    them (important for the CLI, which may configure logging twice).
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    console = logging.StreamHandler(stream=sys.stderr)
    console.setLevel(logging.WARNING if quiet_console else root.level)
    console.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
    )
    root.addHandler(console)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "engine.log", encoding="utf-8")
        file_handler.setLevel(root.level)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-7s %(name)s [%(threadName)s]: %(message)s"
            )
        )
        root.addHandler(file_handler)

        if json_log:
            json_handler = logging.FileHandler(log_dir / "engine.jsonl", encoding="utf-8")
            json_handler.setLevel(root.level)
            json_handler.setFormatter(_JsonFormatter())
            root.addHandler(json_handler)

    # Third-party noise.
    for noisy in ("httpx", "httpcore", "urllib3", "filelock", "datasets", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return root


class EventLog:
    """Append-only JSONL event sink, safe for concurrent writers."""

    def __init__(self, path: Path | None):
        self._path = Path(path) if path else None
        self._lock = threading.Lock()
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, **fields: Any) -> None:
        if not self._path:
            return
        payload = {"ts": time.time(), "event": event, **fields}
        blob = orjson.dumps(payload, default=str) + b"\n"
        with self._lock:
            # O_APPEND writes of a single small buffer are atomic on POSIX, so
            # concurrent emitters never interleave partial lines.
            fd = os.open(self._path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                os.write(fd, blob)
            finally:
                os.close(fd)


class ScorerWatchdog:
    """Names a scorer that has stopped returning, while it is still running.

    An adapter's ``score()`` runs on a worker thread, and a thread cannot be
    interrupted. If one stops returning -- SymPy failing to terminate on a
    model's equation is the case this was written for -- the run does not
    crash. It goes quiet: pure-Python work holds the GIL, so the event loop
    gets only slivers, batches stop being sent, and the engine starts
    reporting endpoints "unreachable" that are answering in a millisecond.
    Run ``20260921-012522_reasoning`` spent seven hours like that, and nothing
    in the log said which dataset, which sample, or even that a scorer was
    involved.

    This does not fix that -- the bound belongs where the work is, and for the
    symbolic comparisons it is in ``adapters/_symbolic.py``. What it does is
    make the next one legible within minutes: any scorer still running after
    ``warn_after_s`` is named, with its dataset and sample, and named again
    every ``interval_s`` until it finishes.

    It is a plain daemon thread, not an asyncio task, precisely because the
    event loop is the thing that will not be running. CPython releases the GIL
    every few milliseconds even inside a tight Python loop, so the thread
    still gets its slice.
    """

    def __init__(self, *, warn_after_s: float = 300.0, interval_s: float = 300.0) -> None:
        self.warn_after_s = warn_after_s
        self.interval_s = interval_s
        self._lock = threading.Lock()
        #: token -> (started_at, dataset_id, sample_id, times already warned)
        self._live: dict[int, list[Any]] = {}
        self._next = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, name="scorer-watchdog", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    @contextlib.contextmanager
    def watch(self, dataset_id: str, sample_id: str) -> Iterator[None]:
        with self._lock:
            token = self._next
            self._next += 1
            self._live[token] = [time.monotonic(), dataset_id, sample_id, 0]
        try:
            yield
        finally:
            with self._lock:
                self._live.pop(token, None)

    def _loop(self) -> None:
        # Checked often enough to be useful, rarely enough to cost nothing.
        while not self._stop.wait(min(30.0, self.interval_s)):
            now = time.monotonic()
            overdue = []
            with self._lock:
                for entry in self._live.values():
                    elapsed = now - entry[0]
                    due = self.warn_after_s + entry[3] * self.interval_s
                    if elapsed >= due:
                        entry[3] += 1
                        overdue.append((elapsed, entry[1], entry[2]))
            for elapsed, dataset_id, sample_id in overdue:
                logger.warning(
                    "scorer for %s sample %s has been running %.0f minutes and has not "
                    "returned. A scorer cannot be interrupted, so while it runs it holds "
                    "the GIL: batches stall and endpoints may be misreported as "
                    "unreachable. Nothing already scored is affected.",
                    dataset_id, sample_id, elapsed / 60.0,
                )
