"""Structured logging and event telemetry.

Two sinks per run:

* a human-readable log (``engine.log``) -- what a person tails during a run;
* a JSONL event log (``events.jsonl``) -- one object per engine event
  (batch submitted/completed/retried, sample skipped, task finished), which is
  what the run report and any post-hoc analysis read.

Events are appended atomically so a killed run still leaves a valid log.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import orjson

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
