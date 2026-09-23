"""A backup must be able to correct itself.

The sync used `rclone copy --update`, which skips a file whose remote
modification time is not older than the local one. That is a cheap definition
of "changed" and it has a failure mode with no exit: once the remote's
timestamp runs AHEAD of the local file's, the file can never be sent again.
Every later pass compares the two times, decides the remote is current, and
skips -- forever.

Observed, not imagined. The results workbook sat on Drive holding 61,463,134
bytes stamped 11:41 while the real file was 62,063,409 bytes stamped 00:36: a
partial upload that had been given a later timestamp than the content it
replaced. It was missing an entire night's results for two models, and a dry
run confirmed rclone would transfer nothing. 278 files were stale the same
way, and no number of passes would have fixed one of them.

These tests use a local directory as the "remote", which is what rclone does
for any backend, so the timestamp logic under test is the same one Drive hits.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import pytest

from abductionbench.core.sync import ArtifactSync

pytestmark = pytest.mark.skipif(
    shutil.which("rclone") is None, reason="rclone is not installed"
)


def _config(remote: Path):
    from abductionbench.core.config import SyncConfig

    return SyncConfig(
        enabled=True, remote_path=str(remote), per_run_subdir=True, interval_s=3600
    )


def _syncer(tmp_path: Path):
    run = tmp_path / "runs" / "run-1"
    (run / "reports").mkdir(parents=True)
    remote = tmp_path / "remote"
    return ArtifactSync(_config(remote), run, "run-1"), run, remote


def test_a_remote_copy_with_a_newer_timestamp_is_still_replaced(tmp_path):
    """The exact shape that pinned the workbook: newer clock, older content."""
    syncer, run, remote = _syncer(tmp_path)
    book = run / "reports" / "results.xlsx"
    book.write_text("the real results, complete", encoding="utf-8")
    syncer.flush()

    uploaded = remote / "run-1" / "reports" / "results.xlsx"
    assert uploaded.read_text() == "the real results, complete"

    # Now corrupt the remote the way a partial upload did: different content,
    # and a timestamp an hour in the future.
    uploaded.write_text("truncated, missing a night of results", encoding="utf-8")
    future = time.time() + 3600
    os.utime(uploaded, (future, future))
    assert uploaded.stat().st_mtime > book.stat().st_mtime

    syncer.flush()
    assert uploaded.read_text() == "the real results, complete", (
        "a stale remote copy with a future timestamp was never re-sent"
    )


def test_a_file_that_has_not_changed_is_not_reuploaded(tmp_path):
    """Incremental still means incremental -- just decided by content."""
    syncer, run, remote = _syncer(tmp_path)
    (run / "engine.log").write_text("unchanged", encoding="utf-8")
    syncer.flush()

    uploaded = remote / "run-1" / "engine.log"
    first = uploaded.stat().st_mtime_ns
    time.sleep(0.01)
    syncer.flush()
    assert uploaded.stat().st_mtime_ns == first, "an unchanged file was re-sent"


def test_a_changed_file_of_the_same_size_is_sent(tmp_path):
    """Size alone is not a hash; an edit that keeps the length still counts."""
    syncer, run, remote = _syncer(tmp_path)
    path = run / "metrics.json"
    path.write_text('{"accuracy": 0.10}', encoding="utf-8")
    syncer.flush()

    path.write_text('{"accuracy": 0.99}', encoding="utf-8")
    os.utime(path, (0, 0))  # and an ancient timestamp, to be sure it is content
    syncer.flush()
    assert (remote / "run-1" / "metrics.json").read_text() == '{"accuracy": 0.99}'


def test_the_upload_command_decides_on_content_not_on_time(tmp_path):
    """Named explicitly, because the difference is invisible until it bites."""
    syncer, run, _ = _syncer(tmp_path)
    command = syncer._command(run)  # noqa: SLF001
    assert "--checksum" in command
    assert "--update" not in command


def test_verification_frees_paths_that_hold_more_than_one_object(tmp_path, monkeypatch):
    """A duplicated remote path is skipped by every pass, silently.

    Google Drive allows two files with the same name in one folder. rclone will
    not guess between them -- it logs "Duplicate object found in destination -
    ignoring" and skips the path -- so the file freezes at whatever it was, and
    the pass still exits 0 because nothing failed.

    Two writers are all it takes: a manual repair alongside the run's own
    5-minute sync produced 95 such paths, and the results workbook was one of
    them. A whole night of results could not be uploaded while every pass
    reported success. So verification now clears them before it repairs.
    """
    import subprocess

    syncer, run, _ = _syncer(tmp_path)
    (run / "engine.log").write_text("x", encoding="utf-8")

    calls: list[list[str]] = []
    real_run = subprocess.run

    def _spy(command, *args, **kwargs):
        calls.append(list(command))
        if len(command) > 1 and command[1] == "dedupe":
            return subprocess.CompletedProcess(
                command, 0,
                stdout="NOTICE: reports/results.xlsx: Found 2 files with duplicate names\n",
                stderr="",
            )
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", _spy)
    syncer.verify_and_repair()

    dedupe = [c for c in calls if len(c) > 1 and c[1] == "dedupe"]
    assert dedupe, "verification never checked for duplicated remote paths"
    command = dedupe[0]
    assert command.index("--dedupe-mode") + 1 < len(command)
    assert command[command.index("--dedupe-mode") + 1] == "rename", (
        "a backup must not resolve an ambiguity by deleting: rclone already removes "
        "byte-identical copies in any mode, and renaming keeps genuinely differing "
        "ones while freeing the canonical name"
    )


def test_the_duplicate_check_runs_before_the_repair(tmp_path, monkeypatch):
    """Order matters: repairing a duplicated path does nothing at all."""
    import subprocess

    syncer, run, _ = _syncer(tmp_path)
    (run / "engine.log").write_text("x", encoding="utf-8")
    monkeypatch.setattr(syncer, "_missing_files", lambda: ["engine.log"])

    order: list[str] = []
    real_run = subprocess.run

    def _spy(command, *args, **kwargs):
        if len(command) > 1 and command[1] in ("dedupe", "copy"):
            order.append(command[1])
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", _spy)
    syncer.verify_and_repair()
    assert order and order[0] == "dedupe", order


def test_a_dedupe_failure_does_not_fail_the_run(tmp_path, monkeypatch):
    """A backup step must never be the thing that ends a run."""
    import subprocess

    syncer, run, _ = _syncer(tmp_path)
    (run / "engine.log").write_text("x", encoding="utf-8")

    def _boom(command, *args, **kwargs):
        raise OSError("rclone exploded")

    monkeypatch.setattr(subprocess, "run", _boom)
    syncer.verify_and_repair()  # must not raise
