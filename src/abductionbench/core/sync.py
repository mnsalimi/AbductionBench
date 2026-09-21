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
   Drive's per-minute request quota for rclone's shared OAuth client, so API
   calls are throttled (``--tpslimit``, ``--drive-pacer-min-sleep``) and
   ``checkpoint.max_raw_payloads`` caps the payloads where they are written.
   ``exclude`` is empty by default and filters the final pass as well as the
   interval ones, so anything put there never reaches the backup at all.

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

__all__ = ["ArtifactSync", "SyncStats", "restore_run"]


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
    #: Files the remote was missing after a pass and that were re-sent.
    repaired: int = 0
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


def restore_run(config: SyncConfig, run_id: str, destination: Path) -> tuple[bool, str]:
    """Pull a run's artifacts back from the remote.  Returns (restored, detail).

    The point of backing a run up is being able to continue it: this box's
    filesystem is not guaranteed to survive, so the copy on the remote is
    sometimes the only one left.  Continuing needs the records -- they are what
    tells the engine which samples are already answered -- so a resume whose
    local directory is missing looks for it on the remote before giving up.

    Nothing is deleted locally and nothing is written remotely; ``copy`` in one
    direction only, so a restore can never damage either side.
    """
    if not config.remote_path:
        return False, "engine.sync.remote_path is not set"
    if shutil.which(config.rclone_binary) is None:
        return False, f"{config.rclone_binary!r} is not on PATH"
    base = config.remote_path.rstrip("/")
    source = f"{base}/{run_id}" if config.per_run_subdir else base
    command = [
        config.rclone_binary, "copy", source, str(destination),
        f"--transfers={config.transfers}",
        f"--timeout={config.timeout_s}s",
        "--retries=3", "--fast-list", "--stats=0",
    ]
    if config.tps_limit:
        command.extend(["--tpslimit", str(config.tps_limit)])
    try:
        completed = subprocess.run(  # noqa: S603 - binary and args come from config
            command, capture_output=True, text=True, timeout=None, check=False
        )
    except (subprocess.SubprocessError, OSError) as exc:
        return False, f"cannot run rclone: {exc}"
    if completed.returncode != 0:
        return False, (completed.stderr or completed.stdout or "").strip()[-300:]
    records = list(destination.rglob("records.jsonl"))
    if not records:
        return False, f"{source} holds no records.jsonl -- nothing to continue from"
    return True, f"restored {len(records)} task record file(s) from {source}"


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
        #: Set by request_upload() so a caller on the hot path can ask for an
        #: upload without waiting for one. The loop picks it up on its next
        #: turn, which is also what keeps requests from stacking: several
        #: datasets finishing close together coalesce into one pass.
        self._requested = threading.Event()
        self._requested_reason = ""
        #: Monotonic deadline before which no *interval* pass may start. Set
        #: when the destination rate-limits us, so the response to "you are
        #: asking too often" is to ask less often rather than to keep asking.
        self._backoff_until = 0.0
        self._rate_limited_in_a_row = 0
        #: When the last pass finished, so a requested upload keeps a minimum
        #: distance from the previous one however often it is requested.
        self._last_pass_ended = 0.0
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
        # --delete-excluded, not just --delete: plain --delete leaves files the
        # receiver already has when they are *excluded*, so a stage built before
        # a pattern was added keeps serving what that pattern is meant to skip,
        # and rclone dutifully uploads it forever. Tightening `exclude` on a run
        # that has already synced is exactly when this bites.
        command = ["rsync", "-a", "--delete", "--delete-excluded"]
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
        while True:
            # Wake either on the interval or as soon as something asks for an
            # upload, whichever comes first.
            requested = self._requested.wait(timeout=self.config.interval_s)
            if self._stop.is_set():
                return
            reason = "interval"
            if requested:
                self._requested.clear()
                reason = self._requested_reason or "requested"
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
            self._tick(reason=reason)

    def stop(self, *, final: bool = True) -> SyncStats:
        """Stop the thread and (by default) run one last upload."""
        if not self.config.enabled or self._degraded:
            return self.stats
        self._stop.set()
        # The loop waits on the request event, so signal it or stopping would
        # block for up to a whole interval.
        self._requested.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if final:
            logger.info("artifact sync: final upload of %s", self.run_dir)
            self._tick(reason="final")
            if self.config.verify_after_final:
                self.verify_and_repair()
        return self.stats

    def flush(self) -> SyncStats:
        """Upload now, on this thread (used after the report files are written)."""
        if not self.config.enabled or self._degraded:
            return self.stats
        self._tick(reason="flush")
        return self.stats

    def request_upload(self, reason: str = "requested") -> None:
        """Ask the background thread to upload soon, and return immediately.

        For callers on the hot path -- the engine asking for the workbook to go
        up now that a dataset has finished its API calls. It must not block the
        event loop, and it must not turn "one upload per dataset" into "one
        rclone per dataset finishing at the same moment": the loop coalesces
        requests that arrive while a pass is running, and enforces the same
        minimum spacing between passes that the interval gives.
        """
        if not self.config.enabled or self._degraded or self._thread is None:
            return
        self._requested_reason = reason
        self._requested.set()

    # ------------------------------------------------------------------ #
    # one upload
    # ------------------------------------------------------------------ #

    #: Destination responses that mean "you are asking too often". Backing off
    #: is the only correct answer to these: retrying on the usual interval turns
    #: one rate-limited pass into an hour of them, and on Google Drive the quota
    #: is per *project* across every rclone user, so a run that keeps hammering
    #: also starves whatever else is uploading.
    _RATE_LIMITED = (
        "ratelimitexceeded",
        "userratelimitexceeded",
        "quotaexceeded",
        "too many requests",
        "429",
        "403: rate",
    )

    def _may_start(self, reason: str) -> bool:
        """Whether a pass may begin now. A pure predicate; the caller owns the lock.

        Two separate guards, and both matter:

        * **backoff** -- after the destination rate-limits us, interval passes
          stand down for a growing interval instead of retrying every minute.
        * **spacing** -- no pass starts within one interval of the last one
          finishing, whoever asked. This is what keeps "upload after each
          dataset" from becoming a burst when several datasets finish together.

        A ``final`` pass overrides both: the run is ending, and the last upload
        is the one that must not be skipped.  An explicit ``flush`` -- a caller
        that deliberately blocked to wait for an upload -- overrides the
        spacing but not the backoff, because forcing a request at a destination
        that has just refused one only lengthens the refusal.
        """
        if reason == "final":
            return True
        now = time.monotonic()
        if now < self._backoff_until:
            logger.info(
                "artifact sync: standing down for another %.0fs after the destination "
                "rate-limited us (%s pass skipped; nothing is lost, the next pass sends "
                "everything that changed)",
                self._backoff_until - now,
                reason,
            )
            return False
        if reason == "flush":
            return True
        gap = now - self._last_pass_ended
        if self._last_pass_ended and gap < self.config.min_gap_s:
            logger.debug(
                "artifact sync: only %.0fs since the last upload; %s pass deferred",
                gap,
                reason,
            )
            return False
        return True

    def _clear_backoff(self) -> None:
        if self._rate_limited_in_a_row:
            logger.info("artifact sync: destination accepted us again; normal interval resumed")
        self._rate_limited_in_a_row = 0
        self._backoff_until = 0.0

    def _note_failure(self, message: str) -> None:
        """Back off when the failure says we are going too fast."""
        lowered = message.lower()
        if not any(marker in lowered for marker in self._RATE_LIMITED):
            return
        self._rate_limited_in_a_row += 1
        delay = min(
            self.config.interval_s * (2**self._rate_limited_in_a_row),
            self.config.rate_limit_backoff_max_s,
        )
        self._backoff_until = time.monotonic() + delay
        logger.warning(
            "artifact sync: the destination rate-limited us (%d in a row); waiting %.0fs "
            "before the next attempt. Uploads are incremental, so the delay costs nothing "
            "but freshness.%s",
            self._rate_limited_in_a_row,
            delay,
            self._shared_client_hint(),
        )
        self._emit("sync_rate_limited", seconds=round(delay, 1))

    def _shared_client_hint(self) -> str:
        """Say so when the quota being hit is not really ours to pace around.

        A Google Drive remote configured without its own ``client_id`` uses
        rclone's built-in OAuth client, whose per-minute project quota is shared
        by every rclone user in the world. When that is the limit being hit, no
        amount of backing off on this box fixes it, and the fix is a one-off
        configuration change -- so the log says which of the two situations this
        is rather than leaving it to be guessed at from repeated warnings.
        """
        if self._rate_limited_in_a_row < 2:
            return ""
        remote = self.config.remote_path.split(":", 1)[0]
        if not remote or "/" in remote:
            return ""
        try:
            shown = subprocess.run(  # noqa: S603 - fixed binary, remote name from config
                [self.config.rclone_binary, "config", "show", remote],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return ""
        if "type = drive" not in shown or "client_id" in shown:
            return ""
        return (
            f" NOTE: the {remote!r} remote has no client_id of its own, so it is using "
            "rclone's built-in one, whose Drive quota is shared with every other rclone "
            "user. That quota is very likely what is being hit, and pacing on this box "
            "cannot fix it -- create a Google Cloud OAuth client and set it with "
            f"`rclone config update {remote} client_id <id> client_secret <secret>`."
        )

    def _tick(self, *, reason: str) -> None:
        # Serialize ticks: a slow upload must not overlap with the next one, or
        # rclone instances would fight over the same files.
        if not self._lock.acquire(blocking=False):
            logger.debug("artifact sync: previous upload still running; skipping this tick")
            return
        if not self._may_start(reason):
            self._lock.release()
            return
        started = time.monotonic()
        try:
            self.stats.ticks += 1
            source = self._snapshot()
            completed = subprocess.run(  # noqa: S603 - binary and args come from config
                self._command(source),
                capture_output=True,
                text=True,
                # No wall-clock kill by default. rclone creates a destination
                # directory before it uploads into it, so a pass killed part-way
                # leaves an empty remote directory that later passes, seeing the
                # directory already there, never fill in. rclone's own
                # --timeout still bounds an individual stalled transfer.
                timeout=self.config.pass_timeout_s or None,
                check=False,
            )
            elapsed = time.monotonic() - started
            self.stats.total_seconds += elapsed
            self._last_pass_ended = time.monotonic()
            if completed.returncode == 0:
                self.stats.successes += 1
                self.stats.last_success_ts = time.time()
                self._clear_backoff()
                logger.info(
                    "artifact sync ok (%s) in %.1fs -> %s", reason, elapsed, self.destination
                )
                self._emit("sync_ok", reason=reason, seconds=round(elapsed, 1))
            else:
                self.stats.failures += 1
                message = (completed.stderr or completed.stdout or "").strip()[-500:]
                self.stats.last_error = message
                self._note_failure(message)
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

    # ------------------------------------------------------------------ #
    # verification
    # ------------------------------------------------------------------ #

    def verify_and_repair(self) -> list[str]:
        """Re-upload anything the remote is missing.  Returns what is still absent.

        A pass can fail per file and still exit non-zero only once, and Drive
        answers a burst of small uploads with HTTP 403 rate-limit errors that
        rclone reports at the end.  Either way the visible symptom is the same:
        a run whose report uploaded fine sitting next to
        ``datasets/<dataset>/<model>/<template>/`` directories that are empty,
        because the directory is created before the files land in it.

        So the end of a run does not trust the exit code.  It asks the remote
        what it actually has, re-sends what is missing, and asks again --
        because the second attempt is usually enough, and because a backup that
        cannot say what it holds is not a backup.
        """
        if not self.config.enabled or self._degraded:
            return []
        missing: list[str] = []
        for attempt in range(1, self.config.verify_attempts + 1):
            missing = self._missing_files()
            if not missing:
                if attempt > 1:
                    logger.info("artifact sync: remote complete after %d repair pass(es)", attempt - 1)
                self._emit("sync_verified", destination=self.destination, missing=0)
                return []
            logger.warning(
                "artifact sync: %d file(s) missing on %s after upload (%s%s); re-sending",
                len(missing),
                self.destination,
                ", ".join(missing[:3]),
                ", ..." if len(missing) > 3 else "",
            )
            self.stats.repaired += len(missing)
            self._repair(missing)
        remaining = self._missing_files()
        if remaining:
            self.stats.last_error = f"{len(remaining)} file(s) never uploaded"
            logger.error(
                "artifact sync: %d file(s) are still missing from %s after %d repair attempt(s). "
                "They are on local disk; re-run tools/sync_run.sh to finish the upload. "
                "First few: %s",
                len(remaining),
                self.destination,
                self.config.verify_attempts,
                ", ".join(remaining[:5]),
            )
            self._emit("sync_incomplete", destination=self.destination,
                       missing=len(remaining), examples=remaining[:5])
        return remaining

    def _missing_files(self) -> list[str]:
        """Paths present locally but not on the remote, via ``rclone check``."""
        source = self.stage_dir if self.stage_dir.exists() else self.run_dir
        command = [
            self.config.rclone_binary,
            "check",
            str(source),
            self.destination,
            # One-way: extra files on the remote (an older run's leftovers) are
            # not our problem; files we hold and the remote does not are.
            "--one-way",
            "--missing-on-dst",
            "-",
            "--fast-list",
            "--stats=0",
        ]
        if self.config.tps_limit:
            command.extend(["--tpslimit", str(self.config.tps_limit)])
        for pattern in self.config.exclude:
            command.extend(["--exclude", pattern])
        try:
            completed = subprocess.run(  # noqa: S603 - binary and args come from config
                command, capture_output=True, text=True, timeout=self.config.timeout_s * 4,
                check=False,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            logger.warning("artifact sync: cannot verify the remote: %s", exc)
            return []
        # rclone writes the missing paths to the file named by --missing-on-dst,
        # which is stdout here; its own progress goes to stderr.
        return [line.strip() for line in completed.stdout.splitlines() if line.strip()]

    def _repair(self, missing: list[str]) -> None:
        """Re-upload exactly the named files."""
        source = self.stage_dir if self.stage_dir.exists() else self.run_dir
        listing = source.parent / f".{self.run_id}.missing"
        try:
            listing.write_text("\n".join(missing), encoding="utf-8")
            command = [
                self.config.rclone_binary,
                "copy",
                str(source),
                self.destination,
                f"--files-from={listing}",
                f"--transfers={self.config.transfers}",
                f"--timeout={self.config.timeout_s}s",
                # More patience than a normal pass: this is the last chance, and
                # what is being retried is what already failed once.
                "--retries=5",
                "--low-level-retries=20",
                "--stats=0",
            ]
            if self.config.tps_limit:
                command.extend(["--tpslimit", str(self.config.tps_limit)])
                command.extend(["--drive-pacer-min-sleep", "100ms"])
            subprocess.run(  # noqa: S603 - binary and args come from config
                command, capture_output=True, text=True,
                timeout=self.config.pass_timeout_s or None, check=False,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            logger.warning("artifact sync: repair pass failed: %s", exc)
        finally:
            listing.unlink(missing_ok=True)

    def _emit(self, event: str, **fields: object) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event(event, **fields)
        except Exception:  # noqa: BLE001 - telemetry must not break the uploader
            logger.debug("artifact sync: event sink failed", exc_info=True)
