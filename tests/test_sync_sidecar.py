"""The standalone sidecar (`tools/sync_run.sh`).

It runs as a separate process against a live run directory, so its guarantees
are shell-level and are tested by actually running it. Both behaviours below
come from bugs seen in production against Google Drive.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

SIDECAR = Path(__file__).resolve().parents[1] / "tools" / "sync_run.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("rclone") is None or shutil.which("rsync") is None,
    reason="rclone/rsync not installed",
)


def _env(stage: Path, remote: Path) -> dict[str, str]:
    return {
        **os.environ,
        "SYNC_REMOTE": str(remote),
        "SYNC_INTERVAL": "30",
        "SYNC_STAGE": str(stage),
    }


def test_only_one_sidecar_per_run_may_run(tmp_path: Path):
    """Two sidecars upload the same files and Drive stores both copies.

    That is how duplicate objects appeared in the live folder, so a second
    instance for the same run must refuse.
    """
    run = tmp_path / "runs" / "20260101-000000_test"
    run.mkdir(parents=True)
    (run / "engine.log").write_text("x\n", encoding="utf-8")
    stage, remote = tmp_path / "stage", tmp_path / "remote"

    first = subprocess.Popen(
        ["bash", str(SIDECAR), str(run)],
        env=_env(stage, remote),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        # Wait for the first pass, which proves the lock is held.
        deadline = time.time() + 30
        while time.time() < deadline and not (remote / run.name / "engine.log").exists():
            time.sleep(0.3)
        assert (remote / run.name / "engine.log").exists(), "first sidecar never uploaded"

        second = subprocess.run(
            ["bash", str(SIDECAR), str(run)],
            env=_env(stage, remote),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert second.returncode == 3, second.stdout + second.stderr
        assert "already backing up" in second.stderr
    finally:
        first.terminate()
        first.wait(timeout=60)


def test_the_lock_survives_the_rsync_delete(tmp_path: Path):
    """The lock must live outside the staged tree.

    `rsync --delete` mirrors the run directory into the staging directory; a
    lock kept inside it was deleted on every pass, leaving each process holding
    a lock on an unlinked inode -- so two sidecars ran anyway.
    """
    run = tmp_path / "runs" / "20260101-000000_test"
    run.mkdir(parents=True)
    (run / "engine.log").write_text("x\n", encoding="utf-8")
    stage, remote = tmp_path / "stage", tmp_path / "remote"

    process = subprocess.Popen(
        ["bash", str(SIDECAR), str(run)],
        env=_env(stage, remote),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.time() + 30
        while time.time() < deadline and not (remote / run.name / "engine.log").exists():
            time.sleep(0.3)
        lock = stage / f".{run.name}.sidecar.lock"
        assert lock.exists(), "lock is missing after a pass -- it is inside the rsync target"
        assert not (stage / run.name / ".sidecar.lock").exists()
    finally:
        process.terminate()
        process.wait(timeout=60)
