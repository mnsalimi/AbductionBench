"""Incremental artifact backup.

rclone copies to a plain local path just as it does to ``gdrive:``, so every
behaviour below is exercised for real -- a subprocess, a filesystem, changed
files -- without needing any credentials.
"""

from __future__ import annotations

import logging
import shutil
import threading
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
    """A slow upload must not have the next tick run on top of it.

    An *interval* tick that finds one running simply skips: another pass is
    already sending whatever changed, and there will be another tick after it.
    """
    run, remote = _run_dir(tmp_path), tmp_path / "remote"
    syncer = ArtifactSync(_config(remote), run, "run-1")
    syncer._lock.acquire()  # noqa: SLF001 - simulate an upload in progress
    try:
        syncer._tick(reason="interval")  # noqa: SLF001
        assert syncer.stats.ticks == 0
    finally:
        syncer._lock.release()  # noqa: SLF001


def test_a_closing_pass_waits_for_a_running_one_instead_of_skipping(tmp_path: Path):
    """`final` and `flush` are the end of the run: skipping them loses data.

    They used to take the same non-blocking path as an interval tick and
    return silently at DEBUG level. There is no next tick to catch what they
    drop, so whatever was written last -- in practice the reports, which are
    written *after* the final upload -- stayed on the box. This waits instead.
    """
    run, remote = _run_dir(tmp_path), tmp_path / "remote"
    syncer = ArtifactSync(_config(remote), run, "run-1")
    syncer._CLOSING_LOCK_WAIT_S = 10.0  # noqa: SLF001 - the real one is 15 minutes

    syncer._lock.acquire()  # noqa: SLF001 - a pass is in flight
    released = threading.Event()

    def _release_soon() -> None:
        time.sleep(0.4)
        syncer._lock.release()  # noqa: SLF001
        released.set()

    threading.Thread(target=_release_soon, daemon=True).start()
    syncer.flush()
    assert released.is_set(), "flush returned before the running pass finished"
    assert syncer.stats.ticks == 1, "flush must upload, not skip"
    assert (remote / "run-1" / "engine.log").exists()


def test_a_closing_pass_that_cannot_start_says_so_loudly(tmp_path: Path, caplog):
    """If it really cannot run, that must not be a DEBUG line nobody reads."""
    run, remote = _run_dir(tmp_path), tmp_path / "remote"
    syncer = ArtifactSync(_config(remote), run, "run-1")
    syncer._CLOSING_LOCK_WAIT_S = 0.2  # noqa: SLF001
    syncer._lock.acquire()  # noqa: SLF001 - never released
    try:
        with caplog.at_level(logging.WARNING, logger="abductionbench.core.sync"):
            syncer.flush()
        assert syncer.stats.ticks == 0
        warned = [r.message for r in caplog.records if "could not start" in r.message]
        assert warned, "a dropped closing pass went unreported"
        # And it says how to fix it by hand.
        assert "rclone copy" in warned[0]
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

    # Opting out uploads the live directory instead, and says so in the command:
    # with no snapshot to filter, rclone has to do the excluding itself.
    direct = ArtifactSync(
        _config(remote, snapshot_before_upload=False, exclude=["raw/"]), run, "run-1"
    )
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


def test_raw_payloads_are_backed_up_by_default_and_can_still_be_excluded(tmp_path: Path):
    """They are the sample prompts and responses; a backup without them is thin.

    They used to be excluded by default, because an uncapped run writes ~1,100
    of them and uploading that exhausts a Drive-style request quota. The bound
    now lives where they are *written* (checkpoint.max_raw_payloads), so the
    quota is safe and the payloads are still visible -- which is what anyone
    reviewing a run actually wants from the backup.
    """
    run, remote = _run_dir(tmp_path), tmp_path / "remote"
    raw = run / "datasets/ds/model/tpl/raw"
    raw.mkdir()
    for index in range(3):
        (raw / f"batch-{index}.json").write_text("{}", encoding="utf-8")

    syncer = ArtifactSync(_config(remote), run, "run-1")  # default exclude: none
    syncer.flush()
    assert (remote / "run-1" / "datasets/ds/model/tpl/records.jsonl").exists()
    assert (remote / "run-1" / "datasets/ds/model/tpl/raw/batch-0.json").exists()

    # ...and a run that keeps every payload can still opt out.
    lean = ArtifactSync(_config(remote, exclude=["raw/"]), run, "run-2")
    lean.flush()
    assert (remote / "run-2" / "datasets/ds/model/tpl/records.jsonl").exists()
    assert not (remote / "run-2" / "datasets/ds/model/tpl/raw").exists()


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


# --------------------------------------------------------------------------- #
# pacing: ask less often when told to, and never in a burst
# --------------------------------------------------------------------------- #


def test_a_rate_limited_pass_backs_off_instead_of_retrying_on_the_interval(tmp_path: Path):
    """The only correct answer to "you are asking too often" is to ask less often.

    Google Drive's quota is per *project* across every rclone user, so a run
    that keeps retrying on its usual interval does not merely fail -- it starves
    whatever else is uploading, including its own earlier passes.
    """
    run, remote = _run_dir(tmp_path), tmp_path / "remote"
    syncer = ArtifactSync(_config(remote, interval_s=60, min_gap_s=0), run, "run-1")

    syncer._note_failure(
        "Failed to copy: googleapi: Error 403: Rate Limit Exceeded, rateLimitExceeded"
    )
    assert syncer._rate_limited_in_a_row == 1
    first = syncer._backoff_until
    assert first > 0

    # An interval pass stands down while backed off; `final` never does.
    assert syncer._may_start("interval") is False
    assert syncer._may_start("final") is True

    # Consecutive refusals wait longer, up to the configured ceiling.
    syncer._note_failure("Error 403: rateLimitExceeded")
    assert syncer._backoff_until > first
    syncer._rate_limited_in_a_row = 99
    syncer._note_failure("Error 403: rateLimitExceeded")
    assert syncer._backoff_until - time.monotonic() <= syncer.config.rate_limit_backoff_max_s + 1


def test_an_ordinary_failure_does_not_back_off(tmp_path: Path):
    """Backoff is for being throttled, not for every error there is."""
    run, remote = _run_dir(tmp_path), tmp_path / "remote"
    syncer = ArtifactSync(_config(remote), run, "run-1")
    syncer._note_failure("directory not found")
    assert syncer._rate_limited_in_a_row == 0
    assert syncer._backoff_until == 0.0


def test_the_first_success_clears_the_backoff(tmp_path: Path):
    run, remote = _run_dir(tmp_path), tmp_path / "remote"
    syncer = ArtifactSync(_config(remote, min_gap_s=0), run, "run-1")
    syncer._note_failure("rateLimitExceeded")
    assert syncer._backoff_until > 0
    syncer._backoff_until = 0.0  # pretend the wait elapsed
    syncer.flush()
    assert syncer.stats.successes >= 1
    assert syncer._rate_limited_in_a_row == 0
    assert syncer._backoff_until == 0.0


def test_requested_uploads_keep_a_minimum_distance_from_the_last_one(tmp_path: Path):
    """Several datasets finishing together must produce one upload, not five."""
    run, remote = _run_dir(tmp_path), tmp_path / "remote"
    syncer = ArtifactSync(_config(remote, min_gap_s=600), run, "run-1")
    syncer._last_pass_ended = time.monotonic()

    assert syncer._may_start("dataset:art") is False
    # An explicit flush is a caller that blocked on purpose, and the final pass
    # is the one that must never be skipped.
    assert syncer._may_start("flush") is True
    assert syncer._may_start("final") is True


def test_request_upload_returns_immediately_and_the_thread_does_the_work(tmp_path: Path):
    """It is called from the engine's event loop, so it must never block."""
    run, remote = _run_dir(tmp_path), tmp_path / "remote"
    syncer = ArtifactSync(_config(remote, interval_s=30, min_gap_s=0), run, "run-1")
    syncer.start()
    try:
        started = time.monotonic()
        syncer.request_upload(reason="dataset:art")
        assert time.monotonic() - started < 0.5, "request_upload blocked the caller"
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and not (remote / "run-1" / "engine.log").exists():
            time.sleep(0.1)
        assert (remote / "run-1" / "engine.log").exists(), (
            "the requested upload never happened"
        )
    finally:
        syncer.stop(final=False)


def test_requests_arriving_during_a_pass_coalesce(tmp_path: Path):
    """Ten datasets finishing at once must not queue ten rclone runs."""
    run, remote = _run_dir(tmp_path), tmp_path / "remote"
    syncer = ArtifactSync(_config(remote, interval_s=30, min_gap_s=600), run, "run-1")
    syncer.start()
    try:
        for index in range(10):
            syncer.request_upload(reason=f"dataset:d{index}")
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and syncer.stats.ticks < 1:
            time.sleep(0.1)
        time.sleep(1.0)
        # min_gap_s holds the rest off; one upload covers them all, because a
        # pass sends whatever has changed rather than one dataset's worth.
        assert syncer.stats.successes == 1, syncer.stats
    finally:
        syncer.stop(final=False)


def test_a_finished_dataset_asks_for_the_workbook_to_go_off_box(
    fake_server, write_run_config, fake_dataset, tmp_path
):
    """A dataset finishing its API calls must ask for an upload there and then.

    A full sweep is many hours, and the interim workbook exists precisely so the
    first dataset's results are readable before the last one finishes. Leaving
    it for the next scheduled pass is usually fine and occasionally not: a run
    that dies in between leaves those results only on a filesystem this box does
    not guarantee.

    What is asserted is the *request*, not a file on the remote: the final pass
    uploads everything anyway, so a file proves nothing about when it went. The
    request is also where the timing guarantee lives -- it happens as the
    dataset completes rather than up to an interval later. That the request
    turns into an upload is
    test_request_upload_returns_immediately_and_the_thread_does_the_work.
    """
    import asyncio

    from abductionbench.core.config import load_run_config
    from abductionbench.core.engine import EvaluationEngine

    remote = tmp_path / "backup"
    config_path = write_run_config(
        base_url=fake_server.base_url,
        datasets=[fake_dataset("fake", n=4, sample_size=4)],
        engine={"sync": {"enabled": True, "remote_path": str(remote), "interval_s": 600}},
    )
    config = load_run_config(config_path)
    engine = EvaluationEngine(config)

    asked: list[str] = []
    original = engine.sync.request_upload
    engine.sync.request_upload = lambda reason="requested": (  # type: ignore[method-assign]
        asked.append(reason),
        original(reason=reason),
    )[0]

    result = asyncio.run(engine.run())

    assert "dataset:fake" in asked, f"the finished dataset asked for nothing: {asked}"
    # The workbook is on the remote by the end either way, which is the backstop.
    assert (remote / result.run_id / "reports" / "abductionbench_results.xlsx").exists()


def test_repeated_rate_limiting_names_the_shared_oauth_client(tmp_path: Path, monkeypatch):
    """When the quota is not ours to pace around, say so.

    A Drive remote with no client_id of its own uses rclone's built-in OAuth
    client, whose per-minute project quota is shared with every rclone user
    alive. Backing off on this box cannot fix that, and a run that keeps logging
    "waiting 240s" without saying why is how an afternoon goes into tuning the
    wrong knob.
    """
    import subprocess as sp

    run, remote = _run_dir(tmp_path), tmp_path / "remote"
    syncer = ArtifactSync(_config(remote, remote_path="gdrive:Backups"), run, "run-1")

    def fake_run(command, **kwargs):
        assert command[1:] == ["config", "show", "gdrive"]
        return sp.CompletedProcess(command, 0, stdout="[gdrive]\ntype = drive\ntoken = x\n", stderr="")

    monkeypatch.setattr(sp, "run", fake_run)
    syncer._rate_limited_in_a_row = 2
    hint = syncer._shared_client_hint()
    assert "client_id" in hint and "rclone config update gdrive" in hint

    # A remote that already has one is not nagged about it.
    def fake_run_configured(command, **kwargs):
        return sp.CompletedProcess(
            command, 0, stdout="[gdrive]\ntype = drive\nclient_id = mine\n", stderr=""
        )

    monkeypatch.setattr(sp, "run", fake_run_configured)
    assert syncer._shared_client_hint() == ""

    # Neither is a one-off blip, nor a destination that is not Drive at all.
    syncer._rate_limited_in_a_row = 1
    assert syncer._shared_client_hint() == ""


def test_tightening_exclude_drops_what_the_stage_already_holds(tmp_path: Path):
    """A pattern added later must take effect, not just apply to new files.

    rsync's --delete leaves files the receiver already has when they are
    excluded, so a stage built under `exclude: []` keeps every raw payload and
    rclone re-uploads them on every pass afterwards -- the exact cost the
    pattern was added to avoid.
    """
    run, remote = _run_dir(tmp_path), tmp_path / "remote"
    raw = run / "datasets" / "ds" / "model" / "tpl" / "raw"
    raw.mkdir(parents=True)
    (raw / "batch-0.json").write_text('{"request": {}}', encoding="utf-8")

    # First pass with nothing excluded: the payload is staged and uploaded.
    syncer = ArtifactSync(_config(remote, exclude=[], min_gap_s=0), run, "run-1")
    syncer.flush()
    assert (remote / "run-1" / "datasets/ds/model/tpl/raw/batch-0.json").exists()
    staged = syncer.stage_dir / "datasets/ds/model/tpl/raw/batch-0.json"
    assert staged.exists()

    # Now exclude it. The stage must stop offering it to rclone.
    tightened = ArtifactSync(_config(remote, exclude=["raw/"], min_gap_s=0), run, "run-1")
    tightened.flush()
    assert not staged.exists(), "the stage still holds an excluded payload"
    # copy never deletes from the remote, so what is already there stays -- the
    # point is that it is not re-sent from here on.
    assert (remote / "run-1" / "engine.log").exists()


def test_sample_prompts_and_responses_reach_the_backup(tmp_path: Path):
    """The raw payloads are what "show me what the model was asked" means.

    They were excluded from sync because an uncapped run writes ~1,100 of them
    and uploading that exhausts a Drive-style request quota. Bounding them where
    they are *written* is the fix that keeps them visible; hiding them from the
    upload made the backup useless for reviewing a run.
    """
    run, remote = _run_dir(tmp_path), tmp_path / "remote"
    raw = run / "datasets" / "ds" / "model" / "tpl" / "raw"
    raw.mkdir(parents=True)
    (raw / "batch-0.json").write_text('{"request": {"messages": []}}', encoding="utf-8")

    syncer = ArtifactSync(_config(remote), run, "run-1")   # default exclude
    syncer.flush()
    assert (remote / "run-1" / "datasets/ds/model/tpl/raw/batch-0.json").exists()


def test_an_uncapped_run_with_sync_on_is_warned_about(tmp_path: Path, caplog):
    """The combination that exhausted the quota is easy to reach by halves.

    Setting max_raw_payloads: 0 for a full audit trail is legitimate; forgetting
    to exclude raw/ from the backup at the same time is how a run ends up trying
    to upload ~1,100 files and starving the results behind them.
    """
    import logging

    from abductionbench.core.config import RunConfig

    payload = {
        "name": "t",
        "prompts": {},
        "datasets": [{"id": "d", "impl": "x:Y"}],
        "models": [
            {"id": "m", "model_name": "m", "endpoint": {"base_url": "http://x"}}
        ],
        "engine": {
            "checkpoint": {"max_raw_payloads": 0},
            "sync": {"enabled": True, "remote_path": str(tmp_path), "exclude": []},
        },
    }
    with caplog.at_level(logging.WARNING):
        RunConfig.model_validate(payload)
    assert any("max_raw_payloads" in record.message for record in caplog.records)

    # Capping it, or excluding raw/, is enough on its own.
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        RunConfig.model_validate({**payload, "engine": {
            "checkpoint": {"max_raw_payloads": 3},
            "sync": {"enabled": True, "remote_path": str(tmp_path), "exclude": []},
        }})
    assert not any("max_raw_payloads" in record.message for record in caplog.records)


def test_verification_catches_a_file_the_remote_holds_a_different_copy_of(tmp_path: Path):
    """"Is it there?" is not the same question as "is it right?".

    A pass can report per-file failures and still be counted once, and a
    truncated or half-written upload leaves a file that *exists*. Asking only
    which paths are absent called that backed up. The check now reports both
    what is missing (`+`) and what differs (`*`).
    """
    run, remote = _run_dir(tmp_path), tmp_path / "remote"
    syncer = ArtifactSync(_config(remote), run, "run-1")
    syncer.flush()
    assert syncer._missing_files() == []  # noqa: SLF001

    # The remote holds a copy, but not the copy we sent.
    (remote / "run-1" / "engine.log").write_text("truncated", encoding="utf-8")
    stale = syncer._missing_files()  # noqa: SLF001
    assert "engine.log" in stale, "a differing remote copy was reported as backed up"

    # And the repair pass fixes it rather than leaving it.
    syncer.verify_and_repair()
    assert (remote / "run-1" / "engine.log").read_text() == "start\n"
    assert syncer._missing_files() == []  # noqa: SLF001


def test_the_end_of_a_run_leaves_nothing_behind(tmp_path: Path):
    """stop() -> write reports -> flush(): the whole closing sequence.

    This is the order the CLI uses, and the one that was losing the reports.
    """
    run, remote = _run_dir(tmp_path), tmp_path / "remote"
    syncer = ArtifactSync(_config(remote, interval_s=3600), run, "run-1")
    syncer.start()
    syncer.stop()  # the engine's final upload

    # The CLI writes reports after the engine returns, then flushes.
    (run / "reports").mkdir(exist_ok=True)
    (run / "reports" / "results.xlsx").write_text("the real numbers", encoding="utf-8")
    (run / "datasets" / "ds" / "model" / "tpl" / "run_documentation.md").write_text(
        "written last", encoding="utf-8"
    )
    syncer.flush()

    assert (remote / "run-1" / "reports" / "results.xlsx").read_text() == "the real numbers"
    assert (
        remote / "run-1" / "datasets/ds/model/tpl/run_documentation.md"
    ).read_text() == "written last"
    assert syncer._missing_files() == [], "the remote is not a copy of the run"  # noqa: SLF001
