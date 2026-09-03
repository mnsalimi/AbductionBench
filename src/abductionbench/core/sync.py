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
5. **It must upload files that are not changing under it.** Records and logs are
   appended to continuously; uploading them live makes rclone size/hash a file,
   send it, find the remote copy no longer matches, declare the transfer corrupt
   and *delete it from the remote*.  Each pass therefore rsyncs a snapshot
   locally first (cheap) and uploads that.
6. **It must respect the destination's rate limits.** A full run writes ~1,200
   files, 1,096 of them per-batch debug payloads; sending those exhausts Google
   Drive's per-minute request quota for rclone's shared OAuth client, so
   ``raw/`` is excluded by default and API calls are throttled.

Any rclone remote works -- Google Drive, S3, another host over SFTP -- and a
plain local path works too, which is how this module is tested without
credentials.
"""

from __future__ import annotations

import logging
import os
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

    @property
    def stage_dir(self) -> Path:
        """Local snapshot directory uploaded in place of the live run dir."""
        base = self.config.stage_dir or os.path.join(
            os.environ.get("TMPDIR", "/tmp"), "abench_sync_stage"
        )
        return Path(base) / self.run_id

    def _snapshot(self) -> Path:
        """rsync the run directory into the staging dir; returns what to upload.

        Falls back to uploading the live directory if rsync is unavailable, so a
        missing tool degrades the guarantee rather than the backup.
        """
        if not self.config.snapshot_before_upload:
            return self.run_dir
        if shutil.which("rsync") is None:
            logger.warning(
                "rsync not found: uploading the live directory, so a file appended to "
                "mid-upload may be rejected by the remote and retried next pass"
            )
            return self.run_dir
        stage = self.stage_dir
        stage.mkdir(parents=True, exist_ok=True)
        command = ["rsync", "-a", "--delete"]
        for pattern in self.config.exclude:
            command.extend(["--exclude", pattern])
        command.extend([f"{self.run_dir}/", f"{stage}/"])
        subprocess.run(  # noqa: S603 - fixed binary, paths from config
            command, capture_output=True, text=True, timeout=self.config.timeout_s, check=True
        )
        return stage

    def _command(self, source: Path) -> list[str]:
        """rclone arguments for uploading ``source`` to the destination."""
        command = [
            self.config.rclone_binary,
            "copy",
            str(source),
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
        if self.config.tps_limit:
            # Drive's shared OAuth client is rate-limited per project across all
            # rclone users; without this a run's file count trips HTTP 403.
            command.extend(["--tpslimit", str(self.config.tps_limit)])
            command.extend(["--drive-pacer-min-sleep", "100ms"])
        if self.config.bandwidth_limit:
            command.append(f"--bwlimit={self.config.bandwidth_limit}")
        if source == self.run_dir:
            # Not snapshotting, so rclone has to do the filtering itself.
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
            if self.config.preflight_retry_s:
                logger.error(
                    "artifact sync cannot start yet: %s. The run continues (results on local "
                    "disk); the destination is re-checked every %.0fs, so fixing it now is "
                    "enough for this run to start backing up.",
                    problem,
                    self.config.preflight_retry_s,
                )
            else:
                logger.error(
                    "artifact sync disabled for this run: %s. The run continues; results stay "
                    "on local disk only.",
                    problem,
                )
            self._emit("sync_disabled", reason=problem)
            if not self.config.preflight_retry_s:
                return
            # Keep a thread alive purely to watch for the destination becoming
            # usable; it uploads nothing until preflight passes.
        self._thread = threading.Thread(
            target=self._loop, name="artifact-sync", daemon=True
        )
        self._thread.start()
        if self._degraded:
            return
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
        last_probe = time.monotonic()
        while not self._stop.wait(self.config.interval_s):
            if self._degraded:
                if not self.config.preflight_retry_s:
                    return
                if time.monotonic() - last_probe < self.config.preflight_retry_s:
                    continue
                last_probe = time.monotonic()
                problem = self.preflight()
                if problem:
                    logger.debug("artifact sync still unavailable: %s", problem)
                    continue
                self._degraded = False
                self.stats.last_error = None
                logger.warning(
                    "artifact sync is now available (%s); backing up everything produced so "
                    "far, then continuing incrementally",
                    self.destination,
                )
                self._emit("sync_recovered", destination=self.destination)
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
            source = self._snapshot()
            completed = subprocess.run(  # noqa: S603 - binary and args come from config
                self._command(source),
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
