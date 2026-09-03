"""Incremental, crash-safe persistence and resume.

Every completed sample is appended to ``records.jsonl`` as a single line.  The
append uses ``O_APPEND`` with one ``os.write`` call of the whole line, which
POSIX guarantees to be atomic for a single writer of a small buffer -- so a run
killed mid-flight leaves a file with only whole records, never a half line.
A sidecar ``checkpoint.json`` (written tmp+rename, i.e. atomically replaced)
carries the task-level summary used for resume decisions and reporting.

Resume policy (``engine.checkpoint.resume_policy``):

* ``strict``    -- a stored record is reused only if its ``prompt_fingerprint``
                   still matches the freshly rendered prompt.  Changing the
                   prompt template, sampling params or model therefore
                   invalidates it, which is what makes "reproducibility without
                   a reproducibility tax" real: you cannot silently mix results
                   from two different prompts.
* ``sample_id`` -- reuse by sample id regardless of prompt changes (cheap
                   continuation of an interrupted run).
* ``off``       -- ignore existing records.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import orjson

from .types import EvalRecord, ResponseStatus

logger = logging.getLogger(__name__)

__all__ = ["RecordStore", "TaskCheckpoint", "load_records"]

RECORDS_FILENAME = "records.jsonl"
CHECKPOINT_FILENAME = "checkpoint.json"


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def load_records(path: Path) -> list[dict[str, Any]]:
    """Read a records file, tolerating a truncated final line."""
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(orjson.loads(line))
            except orjson.JSONDecodeError:
                logger.warning(
                    "%s:%d is not valid JSON (truncated write?); ignoring that line",
                    path,
                    line_number,
                )
    return records


@dataclass(slots=True)
class TaskCheckpoint:
    """Task-level state summary, refreshed as work completes."""

    task: dict[str, str]
    total_planned: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    batches_submitted: int = 0
    batches_failed: int = 0
    bisections: int = 0
    oversize_replacements: int = 0
    prompt_tokens_total: int = 0
    completion_tokens_total: int = 0
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    finished: bool = False
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "total_planned": self.total_planned,
            "completed": self.completed,
            "failed": self.failed,
            "skipped": self.skipped,
            "batches_submitted": self.batches_submitted,
            "batches_failed": self.batches_failed,
            "bisections": self.bisections,
            "oversize_replacements": self.oversize_replacements,
            "prompt_tokens_total": self.prompt_tokens_total,
            "completion_tokens_total": self.completion_tokens_total,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "finished": self.finished,
            "notes": self.notes,
        }


class RecordStore:
    """Append-only record store for one task directory."""

    def __init__(
        self,
        directory: Path,
        *,
        fsync_every: int = 1,
        store_raw_payloads: bool = True,
        max_raw_payloads: int = 0,
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.records_path = self.directory / RECORDS_FILENAME
        self.checkpoint_path = self.directory / CHECKPOINT_FILENAME
        self.raw_dir = self.directory / "raw"
        self._fsync_every = max(1, fsync_every)
        self._store_raw = store_raw_payloads
        self._max_raw = max_raw_payloads
        self._raw_written = 0
        self._since_fsync = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # reading / resume
    # ------------------------------------------------------------------ #

    def existing(self) -> list[dict[str, Any]]:
        return load_records(self.records_path)

    def completed_keys(self, *, policy: str) -> dict[str, dict[str, Any]]:
        """Map resume-key → stored record for records worth reusing.

        Records whose status is ``error`` are never reused: a run that resumes
        after a tunnel outage must retry what the outage broke.  ``skipped``
        records *are* reused, because a skip is a deliberate, reproducible
        decision (e.g. an over-budget prompt with no replacement).
        """
        if policy == "off":
            return {}
        reusable_status = {ResponseStatus.OK.value, ResponseStatus.EMPTY.value,
                           ResponseStatus.TRUNCATED.value, ResponseStatus.SKIPPED.value}
        out: dict[str, dict[str, Any]] = {}
        for record in self.existing():
            if record.get("status") not in reusable_status:
                continue
            sample_id = record.get("sample_id")
            if not sample_id:
                continue
            if policy == "strict":
                key = f"{sample_id}::{record.get('prompt_fingerprint', '')}"
            else:
                key = str(sample_id)
            out[key] = record
        return out

    # ------------------------------------------------------------------ #
    # writing
    # ------------------------------------------------------------------ #

    def append(self, record: EvalRecord) -> None:
        """Atomically append one record."""
        line = orjson.dumps(record.to_json_dict(), default=str) + b"\n"
        with self._lock:
            fd = os.open(self.records_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                os.write(fd, line)
                self._since_fsync += 1
                if self._since_fsync >= self._fsync_every:
                    os.fsync(fd)
                    self._since_fsync = 0
            finally:
                os.close(fd)

    def append_many(self, records: list[EvalRecord]) -> None:
        """Append several records in one write (batch completion)."""
        if not records:
            return
        blob = b"".join(
            orjson.dumps(record.to_json_dict(), default=str) + b"\n" for record in records
        )
        with self._lock:
            fd = os.open(self.records_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                os.write(fd, blob)
                os.fsync(fd)
                self._since_fsync = 0
            finally:
                os.close(fd)

    def save_checkpoint(self, checkpoint: TaskCheckpoint) -> None:
        checkpoint.updated_at = time.time()
        _atomic_write(self.checkpoint_path, orjson.dumps(checkpoint.to_dict(), default=str))

    def load_checkpoint(self) -> dict[str, Any] | None:
        if not self.checkpoint_path.exists():
            return None
        try:
            return orjson.loads(self.checkpoint_path.read_bytes())
        except orjson.JSONDecodeError:
            logger.warning("checkpoint %s is corrupt; ignoring it", self.checkpoint_path)
            return None

    def save_raw(self, name: str, payload: dict[str, Any]) -> None:
        """Persist a raw request/response pair for auditability."""
        if not self._store_raw:
            return
        if self._max_raw and self._raw_written >= self._max_raw:
            return
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
        _atomic_write(self.raw_dir / f"{safe}.json", orjson.dumps(payload, default=str))
        self._raw_written += 1

    def write_json(self, filename: str, payload: Any) -> Path:
        """Write a JSON sidecar (metrics, documentation data) atomically."""
        path = self.directory / filename
        _atomic_write(path, orjson.dumps(payload, default=str, option=orjson.OPT_INDENT_2))
        return path

    def write_text(self, filename: str, text: str) -> Path:
        path = self.directory / filename
        _atomic_write(path, text.encode("utf-8"))
        return path

    def iter_records(self) -> Iterator[dict[str, Any]]:
        yield from self.existing()
