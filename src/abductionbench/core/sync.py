"""Incremental off-box backup of a run's artifacts, out of the hot path.

A 300-sample sweep runs for hours on a machine whose filesystem may not be
persistent, so results and logs are mirrored to remote storage *while the run
is happening* rather than at the end.

Design constraints, in order of importance:

1. **It must never slow inference down.** The uploader is a daemon thread that
   does nothing but launch ``rclone`` as a subprocess. All the I/O happens in
   that separate process, so the asyncio event loop driving batch calls is never
   blocked, and the GIL is released while waiting.
2. **It must never break a run.** Every failure is caught, counted and logged;
   a network outage or a bad credential degrades the run to "not backed up",
   never to "crashed". Nothing in the engine awaits the uploader.
3. **It must be incremental.** ``rclone copy --update`` transfers only files
   that are new or newer locally, so each tick uploads the handful of
   ``records.jsonl``/log files that actually changed.
4. **It must never delete remote data.** ``copy`` is used rather than ``sync``,
   so a local file disappearing (or a fresh run directory) cannot wipe results
   already backed up.

Any rclone remote works -- Google Drive, S3, another host over SFTP -- and a
plain local path works too, which is how this module is tested without
credentials.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import SyncConfig

logger = logging.getLogger(__name__)

__all__ = ["ArtifactSync", "SyncStats"]


@dataclass(slots=True)
class SyncStats:
    """Counters surfaced in the run report."""

    ticks: int = 0
    successes: int = 0
    failures: int = 0
    last_error: str | None = None
    last_success_ts: float | None = None
    total_seconds: float = 0.0
    bytes_transferred: int = 0
    files_transferred: int = 0
    extra: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "ticks": self.ticks,
            "successes": self.successes,
            "failures": self.failures,
            "last_error": self.last_error,
            "last_success_ts": self.last_success_ts,
            "total_seconds": round(self.total_seconds, 1),
            "files_transferred": self.files_transferred,
            "bytes_transferred": self.bytes_transferred,
            **self.extra,
        }


class ArtifactSync:
    """Mirrors a run directory to a remote destination on an interval.

    Usage::

        syncer = ArtifactSync(config.engine.sync, run_dir, run_id)
        syncer.start()          # returns immediately
        ...                     # the run proceeds, untouched
        syncer.stop()           # one final upload, then the thread exits
    """

    def __init__(
        self,
        config: SyncConfig,
        run_dir: Path,
        run_id: str,
        *,
        on_event=None,
    ):
        self.config = config
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.stats = SyncStats()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._on_event = on_event
        #: Set when preflight rules the destination unusable.  Every later
        #: upload is then skipped: retrying a misconfigured remote for hours
        #: would only bury the original error under rclone noise.
        self._degraded = False

    # ------------------------------------------------------------------ #
    # destination
    # ------------------------------------------------------------------ #

    @property
    def destination(self) -> str:
        """``remote:base/run-id`` (or ``base/run-id`` for a local path)."""
        base = self.config.remote_path.rstrip("/")
        if self.config.per_run_subdir:
            return f"{base}/{self.run_id}"
        return base

    def _command(self) -> list[str]:
        command = [
            self.config.rclone_binary,
            "copy",
            str(self.run_dir),
            self.destination,
            # Only send what changed: skip files already on the remote whose
            # modification time is not older than the local one.
            "--update",
            f"--transfers={self.config.transfers}",
            f"--checkers={self.config.checkers}",
            # A stuck transfer must not pile up behind the next tick.
            f"--timeout={self.config.timeout_s}s",
            "--retries=2",
            "--low-level-retries=3",
            # One listing per directory instead of one per file: far fewer API
            # calls against Drive's rate limits.
            "--fast-list",
            "--stats=0",
        ]
        if self.config.bandwidth_limit:
            command.append(f"--bwlimit={self.config.bandwidth_limit}")
        for pattern in self.config.exclude:
            command.extend(["--exclude", pattern])
        command.extend(self.config.extra_args)
        return command

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #

    def preflight(self) -> str | None:
        """Check the uploader can work; returns an error string, or ``None``.

        Called before the run starts so a misconfiguration is reported once and
        loudly, instead of failing silently on every tick for hours.
        """
        if not self.config.enabled:
            return None
        if shutil.which(self.config.rclone_binary) is None:
            return f"{self.config.rclone_binary!r} is not on PATH"
        if not self.config.remote_path:
            return "engine.sync.remote_path is empty"
        remote, separator, _ = self.config.remote_path.partition(":")
        if separator:
            try:
                listed = subprocess.run(  # noqa: S603 - binary comes from config
                    [self.config.rclone_binary, "listremotes"],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )
            except (subprocess.SubprocessError, OSError) as exc:
                return f"cannot run rclone: {exc}"
            remotes = {line.strip().rstrip(":") for line in listed.stdout.splitlines()}
            if remote not in remotes:
                return (
                    f"rclone remote {remote!r} is not configured "
                    f"(configured: {sorted(remotes) or 'none'}); "
                    "run 'rclone config' or set engine.sync.enabled=false"
                )
        return None

    def start(self) -> None:
        if not self.config.enabled:
            return
        problem = self.preflight()
        if problem:
            self._degraded = True
            self.stats.failures += 1
            self.stats.last_error = problem
            logger.error(
                "artifact sync disabled for this run: %s. The run continues; results stay "
                "on local disk only.",
                problem,
            )
            self._emit("sync_disabled", reason=problem)
            return
        self._thread = threading.Thread(
            target=self._loop, name="artifact-sync", daemon=True
        )
        self._thread.start()
        logger.info(
            "artifact sync started: %s -> %s every %ss (incremental, in a background thread)",
            self.run_dir,
            self.destination,
            self.config.interval_s,
        )
        self._emit("sync_started", destination=self.destination)

    def _loop(self) -> None:
        # Wait one interval first: the very first seconds of a run produce only
        # the resolved config, and there is no point racing the engine's startup.
        while not self._stop.wait(self.config.interval_s):
            self._tick(reason="interval")

    def stop(self, *, final: bool = True) -> SyncStats:
        """Stop the thread and (by default) run one last upload."""
        if not self.config.enabled or self._degraded:
            return self.stats
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if final:
            logger.info("artifact sync: final upload of %s", self.run_dir)
            self._tick(reason="final")
        return self.stats

    def flush(self) -> SyncStats:
        """Upload now (used after the report files are written)."""
        if not self.config.enabled or self._degraded:
            return self.stats
        self._tick(reason="flush")
        return self.stats

    # ------------------------------------------------------------------ #
    # one upload
    # ------------------------------------------------------------------ #

    def _tick(self, *, reason: str) -> None:
        # Serialize ticks: a slow upload must not overlap with the next one, or
        # rclone instances would fight over the same files.
        if not self._lock.acquire(blocking=False):
            logger.debug("artifact sync: previous upload still running; skipping this tick")
            return
        started = time.monotonic()
        try:
            self.stats.ticks += 1
            completed = subprocess.run(  # noqa: S603 - binary and args come from config
                self._command(),
                capture_output=True,
                text=True,
                timeout=self.config.timeout_s + 60,
                check=False,
            )
            elapsed = time.monotonic() - started
            self.stats.total_seconds += elapsed
            if completed.returncode == 0:
                self.stats.successes += 1
                self.stats.last_success_ts = time.time()
                logger.info(
                    "artifact sync ok (%s) in %.1fs -> %s", reason, elapsed, self.destination
                )
                self._emit("sync_ok", reason=reason, seconds=round(elapsed, 1))
            else:
                self.stats.failures += 1
                message = (completed.stderr or completed.stdout or "").strip()[-500:]
                self.stats.last_error = message
                logger.warning(
                    "artifact sync failed (%s, exit %d): %s -- the run continues; will retry "
                    "on the next tick",
                    reason,
                    completed.returncode,
                    message,
                )
                self._emit("sync_failed", reason=reason, error=message)
        except subprocess.TimeoutExpired:
            self.stats.failures += 1
            self.stats.last_error = f"rclone timed out after {self.config.timeout_s}s"
            logger.warning("artifact sync timed out (%s); will retry on the next tick", reason)
            self._emit("sync_failed", reason=reason, error="timeout")
        except Exception as exc:  # noqa: BLE001 - a backup must never break a run
            self.stats.failures += 1
            self.stats.last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("artifact sync error (%s): %s", reason, exc)
            self._emit("sync_failed", reason=reason, error=str(exc)[:300])
        finally:
            self._lock.release()

    def _emit(self, event: str, **fields: object) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event(event, **fields)
        except Exception:  # noqa: BLE001 - telemetry must not break the uploader
            logger.debug("artifact sync: event sink failed", exc_info=True)
