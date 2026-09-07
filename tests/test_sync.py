"""Incremental artifact backup.

rclone copies to a plain local path just as it does to ``gdrive:``, so every
behaviour below is exercised for real -- a subprocess, a filesystem, changed
files -- without needing any credentials.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest

from abductionbench.core.config import SyncConfig
from abductionbench.core.sync import ArtifactSync

pytestmark = pytest.mark.skipif(
    shutil.which("rclone") is None, reason="rclone is not installed"
)


def _run_dir(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    (run / "datasets" / "ds" / "model" / "tpl").mkdir(parents=True)
    (run / "engine.log").write_text("start\n", encoding="utf-8")
    (run / "datasets" / "ds" / "model" / "tpl" / "records.jsonl").write_text(
        '{"sample_id": "a"}\n', encoding="utf-8"
    )
    return run


def _config(destination: Path, **overrides) -> SyncConfig:
    fields = {
        "enabled": True,
        "remote_path": str(destination),
        "per_run_subdir": True,
        "interval_s": 0.2,
    }
    fields.update(overrides)
    return SyncConfig(**fields)


def test_flush_mirrors_the_run_directory(tmp_path: Path):
    run, remote = _run_dir(tmp_path), tmp_path / "remote"
    syncer = ArtifactSync(_config(remote), run, "run-1")
    stats = syncer.flush()

    assert stats.successes == 1 and stats.failures == 0
    # per_run_subdir keeps runs from overwriting one another
    assert (remote / "run-1" / "engine.log").read_text() == "start\n"
    assert (remote / "run-1" / "datasets/ds/model/tpl/records.jsonl").exists()


def test_uploads_are_incremental(tmp_path: Path):
    """A second pass must send only what changed, not the whole directory."""
    run, remote = _run_dir(tmp_path), tmp_path / "remote"
    syncer = ArtifactSync(_config(remote), run, "run-1")
    syncer.flush()

    unchanged = remote / "run-1" / "engine.log"
    first_mtime = unchanged.stat().st_mtime_ns

    records = run / "datasets/ds/model/tpl/records.jsonl"
    time.sleep(1.1)  # filesystem timestamp granularity
    records.write_text('{"sample_id": "a"}\n{"sample_id": "b"}\n', encoding="utf-8")
    syncer.flush()

    # the appended file is updated ...
    assert (
        remote / "run-1" / "datasets/ds/model/tpl/records.jsonl"
    ).read_text().count("sample_id") == 2
    # ... and the untouched one was skipped rather than re-copied
    assert unchanged.stat().st_mtime_ns == first_mtime


def test_background_thread_uploads_without_being_asked(tmp_path: Path):
    run, remote = _run_dir(tmp_path), tmp_path / "remote"
    syncer = ArtifactSync(_config(remote, interval_s=0.2), run, "run-1")
    syncer.start()
    try:
        deadline = time.time() + 15
        while time.time() < deadline and not (remote / "run-1" / "engine.log").exists():
            time.sleep(0.2)
        assert (remote / "run-1" / "engine.log").exists()
        assert syncer.stats.successes >= 1
    finally:
        syncer.stop()


def test_stop_performs_a_final_upload(tmp_path: Path):
    run, remote = _run_dir(tmp_path), tmp_path / "remote"
    syncer = ArtifactSync(_config(remote, interval_s=3600), run, "run-1")
    syncer.start()
    (run / "reports").mkdir()
    (run / "reports" / "results.xlsx").write_text("late artifact", encoding="utf-8")
    syncer.stop()
    # Written after the last interval tick, but the final flush still caught it.
    assert (remote / "run-1" / "reports" / "results.xlsx").exists()


def test_exclude_patterns_are_honoured(tmp_path: Path):
    run, remote = _run_dir(tmp_path), tmp_path / "remote"
    raw = run / "datasets/ds/model/tpl/raw"
    raw.mkdir()
    (raw / "batch-0.json").write_text("{}", encoding="utf-8")
    syncer = ArtifactSync(_config(remote, exclude=["raw/"]), run, "run-1")
    syncer.flush()
    assert (remote / "run-1" / "datasets/ds/model/tpl/records.jsonl").exists()
    assert not (remote / "run-1" / "datasets/ds/model/tpl/raw").exists()


def test_a_broken_destination_is_reported_not_raised(tmp_path: Path):
    """A bad remote must degrade to 'not backed up', never fail the run."""
    run = _run_dir(tmp_path)
    config = SyncConfig(
        enabled=True, remote_path="definitely-not-a-remote:some/path", interval_s=0.2
    )
    syncer = ArtifactSync(config, run, "run-1")

    problem = syncer.preflight()
    assert problem and "not configured" in problem

    syncer.start()  # must not raise
    # It uploads nothing, but keeps a watcher thread so that fixing the remote
    # mid-run starts the backup (see the recovery test below).
    assert syncer._degraded is True  # noqa: SLF001
    assert syncer.stats.failures == 1
    stats = syncer.stop()
    assert stats.successes == 0
    assert stats.last_error and "not configured" in stats.last_error


def test_missing_rclone_binary_is_reported(tmp_path: Path):
    run = _run_dir(tmp_path)
    config = SyncConfig(
        enabled=True, remote_path=str(tmp_path / "remote"), rclone_binary="rclone-does-not-exist"
    )
    syncer = ArtifactSync(config, run, "run-1")
    assert "not on PATH" in (syncer.preflight() or "")


def test_disabled_sync_is_a_no_op(tmp_path: Path):
    run, remote = _run_dir(tmp_path), tmp_path / "remote"
    syncer = ArtifactSync(SyncConfig(enabled=False, remote_path=str(remote)), run, "run-1")
    syncer.start()
    syncer.flush()
    syncer.stop()
    assert not remote.exists()
    assert syncer.stats.ticks == 0


def test_overlapping_ticks_do_not_pile_up(tmp_path: Path):
    """A slow upload must not have the next tick run on top of it."""
    run, remote = _run_dir(tmp_path), tmp_path / "remote"
    syncer = ArtifactSync(_config(remote), run, "run-1")
    syncer._lock.acquire()  # noqa: SLF001 - simulate an upload in progress
    try:
        syncer.flush()  # should return immediately without a second rclone
        assert syncer.stats.ticks == 0
    finally:
        syncer._lock.release()  # noqa: SLF001


def test_engine_run_backs_up_artifacts_end_to_end(
    fake_server, write_run_config, fake_dataset, tmp_path
):
    """A real run with sync on: records land locally *and* on the destination."""
    import asyncio

    from abductionbench.core.config import load_run_config
    from abductionbench.core.engine import EvaluationEngine

    remote = tmp_path / "backup"
    config_path = write_run_config(
        base_url=fake_server.base_url,
        datasets=[fake_dataset("fake", n=4, sample_size=4)],
        engine={
            "sync": {
                "enabled": True,
                "remote_path": str(remote),
                "interval_s": 0.5,
            }
        },
    )
    config = load_run_config(config_path)
    engine = EvaluationEngine(config)
    result = asyncio.run(engine.run())

    backed_up = list((remote / result.run_id).rglob("records.jsonl"))
    assert backed_up, f"nothing under {remote / result.run_id}"
    assert backed_up[0].read_text().count("sample_id") == 4
    assert (remote / result.run_id / "run_config.resolved.yaml").exists()
    assert result.sync_stats["successes"] >= 1
    assert result.sync_stats["failures"] == 0


def test_sync_recovers_when_the_destination_becomes_usable(tmp_path: Path, monkeypatch):
    """Credentials added mid-run must start the backup, not be ignored.

    Preflight fails first (unconfigured remote), then succeeds -- as it would
    after `setup_drive_remote.sh` runs while an evaluation is in flight.
    """
    run, remote = _run_dir(tmp_path), tmp_path / "remote"
    config = SyncConfig(
        enabled=True,
        remote_path=str(remote),
        interval_s=0.2,
        preflight_retry_s=0.2,
    )
    syncer = ArtifactSync(config, run, "run-1")

    calls = {"n": 0}
    real_preflight = syncer.preflight

    def flaky_preflight():
        calls["n"] += 1
        if calls["n"] <= 1:
            return "rclone remote 'gdrive' is not configured (configured: none)"
        return real_preflight()

    monkeypatch.setattr(syncer, "preflight", flaky_preflight)

    syncer.start()
    assert syncer._degraded is True          # noqa: SLF001 - it must not upload yet
    assert syncer._thread is not None        # noqa: SLF001 - but it must keep watching
    try:
        deadline = time.time() + 20
        while time.time() < deadline and not (remote / "run-1" / "engine.log").exists():
            time.sleep(0.2)
        assert (remote / "run-1" / "engine.log").exists(), "did not recover"
        assert syncer._degraded is False     # noqa: SLF001
        assert syncer.stats.successes >= 1
    finally:
        syncer.stop()


def test_preflight_retry_can_be_switched_off(tmp_path: Path):
    """With retries off, a bad destination stays off for the whole run."""
    run = _run_dir(tmp_path)
    config = SyncConfig(
        enabled=True,
        remote_path="definitely-not-a-remote:path",
        interval_s=0.2,
        preflight_retry_s=0,
    )
    syncer = ArtifactSync(config, run, "run-1")
    syncer.start()
    assert syncer._thread is None            # noqa: SLF001 - no watcher thread at all
    assert syncer.stop().successes == 0


def test_uploads_a_snapshot_so_growing_files_are_stable(tmp_path: Path):
    """Records and logs are appended to; uploading them live loses them.

    rclone sizes/hashes a file, uploads it, compares with the remote copy, and
    on a mismatch declares the transfer corrupt and DELETES it remotely -- which
    is what happened to events.jsonl on the first live Drive run. Each pass
    therefore rsyncs a snapshot first and uploads that.
    """
    run, remote = _run_dir(tmp_path), tmp_path / "remote"
    syncer = ArtifactSync(_config(remote, exclude=[]), run, "run-1")
    assert syncer._snapshot() == syncer.stage_dir  # noqa: SLF001
    assert (syncer.stage_dir / "engine.log").exists()

    stats = syncer.flush()
    assert stats.failures == 0
    assert (remote / "run-1" / "engine.log").exists()

    # Opting out uploads the live directory instead, and says so in the command.
    direct = ArtifactSync(_config(remote, snapshot_before_upload=False), run, "run-1")
    assert direct._snapshot() == run                     # noqa: SLF001
    assert "--exclude" in direct._command(run)           # rclone filters instead
    assert "--exclude" not in direct._command(direct.stage_dir)


def test_api_calls_are_throttled(tmp_path: Path):
    """Drive's shared OAuth client is rate-limited per project, not per user."""
    run, remote = _run_dir(tmp_path), tmp_path / "remote"
    syncer = ArtifactSync(_config(remote), run, "run-1")
    command = syncer._command(syncer.stage_dir)  # noqa: SLF001
    assert "--tpslimit" in command
    unthrottled = ArtifactSync(_config(remote, tps_limit=0), run, "run-1")
    assert "--tpslimit" not in unthrottled._command(unthrottled.stage_dir)  # noqa: SLF001


def test_raw_payloads_are_excluded_by_default(tmp_path: Path):
    """1,096 of a run's 1,201 files are debug payloads; results must not starve."""
    run, remote = _run_dir(tmp_path), tmp_path / "remote"
    raw = run / "datasets/ds/model/tpl/raw"
    raw.mkdir()
    for index in range(5):
        (raw / f"batch-{index}.json").write_text("{}", encoding="utf-8")

    syncer = ArtifactSync(_config(remote), run, "run-1")  # default exclude
    syncer.flush()
    assert (remote / "run-1" / "datasets/ds/model/tpl/records.jsonl").exists()
    assert not (remote / "run-1" / "datasets/ds/model/tpl/raw").exists()

    # ... and everything can be backed up when asked for explicitly.
    everything = ArtifactSync(_config(remote, exclude=[]), run, "run-2")
    everything.flush()
    assert (remote / "run-2" / "datasets/ds/model/tpl/raw/batch-0.json").exists()


def test_a_file_growing_during_upload_still_reaches_the_remote(tmp_path: Path):
    """End-to-end version of the bug: a file appended to during the pass lands."""
    run, remote = _run_dir(tmp_path), tmp_path / "remote"
    events = run / "events.jsonl"
    events.write_text("\n".join(f'{{"i": {i}}}' for i in range(2000)) + "\n", encoding="utf-8")

    syncer = ArtifactSync(_config(remote), run, "run-1")
    syncer._tick(reason="interval")  # noqa: SLF001 - a live pass

    assert (remote / "run-1" / "events.jsonl").exists()
    assert syncer.stats.failures == 0


def test_a_run_can_be_continued_from_its_backup_after_the_directory_is_gone(
    fake_server, write_run_config, fake_dataset, tmp_path
):
    """The reason for backing a run up: this box's filesystem may not survive.

    Continuing needs the records -- they are what says which samples are already
    answered -- so a resume whose local directory has vanished pulls it back
    from the remote first.
    """
    import asyncio

    from abductionbench.cli import _resolve_resume
    from abductionbench.core.config import load_run_config
    from abductionbench.core.engine import EvaluationEngine

    def _run(path):
        config = load_run_config(path)
        engine = EvaluationEngine(config)
        return asyncio.run(engine.run()), engine

    remote = tmp_path / "remote"
    remote.mkdir()
    config_path = write_run_config(
        base_url=fake_server.base_url,
        datasets=[fake_dataset("a", n=4, sample_size=4)],
        engine={"sync": {"enabled": True, "remote_path": str(remote), "interval_s": 0.2}},
    )
    result, engine = _run(config_path)
    engine.flush_sync()
    assert result.tasks[0].n_scored == 4

    # The box is recycled: the local run directory is gone, the backup is not.
    shutil.rmtree(result.run_dir)
    assert not result.run_dir.exists()
    assert (remote / result.run_id / "datasets").exists()

    config = load_run_config(config_path)
    restored = _resolve_resume(config, result.run_id)
    assert restored == result.run_dir
    assert list(restored.rglob("records.jsonl")), "the records came back"

    # And the run continues from them without re-asking anything.
    calls_before = fake_server.state.requests
    engine2 = EvaluationEngine(config, run_id=result.run_id, run_dir=restored)
    continued = asyncio.run(engine2.run())
    assert continued.tasks[0].n_reused == 4
    assert continued.tasks[0].n_scored == 4
    assert fake_server.state.requests - calls_before <= 2   # only the endpoint probe
