"""Stopping a run: drain what is in flight, cancel what is left, then close.

What these guard against actually happened: an interrupted run closed every
client while episodes, retries and simulator calls were still running, and each
of them was then retried against the closed client and recorded as a model
error -- 150 vivabench episodes, 7,651 seconds of retries, on one interrupt.

The stop is requested from inside the fake server's request handler, which runs
on its own thread while the request is genuinely on the wire. That is what
makes "in flight" true in these tests rather than a timing guess.
"""

from __future__ import annotations

import asyncio
import os
import signal
import threading
import time
from pathlib import Path

import orjson
import pytest

from abductionbench.core.client import ModelClient
from abductionbench.core.config import RetryConfig, load_run_config
from abductionbench.core.engine import EvaluationEngine
from abductionbench.core.errors import EndpointError, ErrorClass
from abductionbench.core.retry import RetryPolicy, with_retry
from abductionbench.core.shutdown import RunInterrupted, Shutdown

PROBE = "ping"


def _is_probe(conversation) -> bool:
    return any(str(m.get("content", "")) == PROBE for m in conversation)


def _real(respond):
    """The endpoint probe is answered plainly: these tests are about the run."""
    def wrapped(conv, max_tokens):
        return "pong" if _is_probe(conv) else respond(conv, max_tokens)
    return wrapped


def _payload_is_probe(payload) -> bool:
    messages = payload.get("messages") or []
    if messages and isinstance(messages[0], list):
        return all(_is_probe(c) for c in messages)
    return _is_probe(messages)


class _Harness:
    """Runs an engine and lets the server thread reach into its loop."""

    def __init__(self, config_path: Path, **engine_kwargs):
        self.engine = EvaluationEngine(load_run_config(config_path), **engine_kwargs)
        self.loop: asyncio.AbstractEventLoop | None = None
        #: The task running engine.run(): it waits for the close-down, so it is
        #: alive while the clients close and holds nothing that uses them.
        self.main_task: asyncio.Task | None = None
        self._stop_sent = threading.Lock()
        self.stops_sent = 0

    def stop(self, *, force: bool = False) -> None:
        """Request the stop from the server thread, as a signal would arrive."""
        assert self.loop is not None
        if self.loop.is_closed():
            # A request still being answered after its test finished -- one of
            # the batch the run cancelled. There is no run left to stop.
            return
        self.loop.call_soon_threadsafe(
            lambda: self.engine.request_shutdown("test", force=force)
        )
        self.stops_sent += 1

    def wait_until_stopping(self, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        while not self.engine.shutdown.requested:
            if time.monotonic() > deadline:  # pragma: no cover - a hung test says why
                raise AssertionError("the stop request never reached the engine")
            time.sleep(0.005)

    def run(self):
        async def main():
            self.loop = asyncio.get_running_loop()
            self.main_task = asyncio.current_task()
            return await self.engine.run()

        return asyncio.run(main())


def _records(run_dir: Path, dataset: str) -> list[dict]:
    rows = []
    for path in (run_dir / "datasets" / dataset).glob("*/*/records.jsonl"):
        rows.extend(orjson.loads(line) for line in path.read_bytes().splitlines() if line.strip())
    return rows


def _config(write_run_config, fake_server, dataset, *, drain_s=30.0, name="test-run", **extra):
    engine = {"shutdown": {"drain_timeout_s": drain_s}, **extra.pop("engine", {})}
    return write_run_config(
        base_url=fake_server.base_url, datasets=[dataset], engine=engine, name=name, **extra
    )


# --------------------------------------------------------------------------- #
# draining: what is in flight finishes, nothing new starts
# --------------------------------------------------------------------------- #


def test_a_stop_lets_in_flight_batches_finish_and_starts_no_more(
    fake_server, write_run_config, fake_dataset
):
    """16 samples in batches of 4, two batches allowed in flight at once.

    The stop arrives while the first two are on the wire. They finish and are
    recorded exactly as usual; the two still waiting for a slot are not sent.
    """
    harness = _Harness(_config(write_run_config, fake_server, fake_dataset(sample_size=16)))
    real_batches = []
    lock = threading.Lock()

    def respond(conv, max_tokens):
        with lock:
            first = not real_batches
            real_batches.append(conv)
        if first:
            harness.stop()
        harness.wait_until_stopping()   # still in flight when the stop lands
        return f"Answer: echo:{conv[-1]['content'][:80]}"

    fake_server.state.responder = _real(respond)
    result = harness.run()

    assert result.shutdown["drained"] is True
    assert result.shutdown["cancelled_tasks"] == 0
    records = _records(result.run_dir, "fake")
    assert len(records) == 8, "the two in-flight batches, and only those"
    assert all(r["status"] == "ok" for r in records)
    assert len(real_batches) == 8, "the batches still queued were never sent"
    assert [row["stage"] for row in result.interrupted] == ["stopped"]
    assert not result.tasks, "an interrupted task is not reported as a finished one"


def test_what_a_stop_leaves_undone_a_resume_does(fake_server, write_run_config, fake_dataset):
    """Finished work survives the stop, and is reused rather than bought again."""
    config_path = _config(write_run_config, fake_server, fake_dataset(sample_size=16))
    harness = _Harness(config_path)
    seen = []
    lock = threading.Lock()

    def stopping(conv, max_tokens):
        # Two batches are served on two threads; exactly one of them may stop
        # the run, or the second request would force it.
        with lock:
            first = not seen
            seen.append(conv)
        if first:
            harness.stop()
        harness.wait_until_stopping()
        return f"Answer: echo:{conv[-1]['content'][:80]}"

    fake_server.state.responder = _real(stopping)
    first = harness.run()
    assert len(_records(first.run_dir, "fake")) == 8

    fake_server.state.responder = lambda conv, mt: f"Answer: echo:{conv[-1]['content'][:80]}"
    second = _Harness(config_path, run_id=first.run_id, run_dir=first.run_dir).run()
    task = second.tasks[0]
    assert task.n_reused == 8, "what the stopped run finished is kept"
    assert task.n_scored == 16
    assert not second.shutdown and not second.interrupted


# --------------------------------------------------------------------------- #
# retries are new work, and are not started
# --------------------------------------------------------------------------- #


def test_a_failed_attempt_is_not_retried_once_the_run_is_stopping(
    fake_server, write_run_config, fake_dataset
):
    """The in-flight attempt fails after the stop. Normally that is retried;
    now it ends as interrupted -- no retry, and no ERROR record."""
    harness = _Harness(
        _config(write_run_config, fake_server, fake_dataset(sample_size=4))
    )
    attempts = []

    def fail_after_stop(payload):
        if _payload_is_probe(payload):
            return None
        attempts.append(payload)
        harness.stop()
        harness.wait_until_stopping()
        return 503   # transient: retried in any other circumstance

    fake_server.state.fail_request = fail_after_stop
    result = harness.run()

    assert len(attempts) == 1, "the failed attempt was retried during the stop"
    assert _records(result.run_dir, "fake") == [], "an interruption was recorded as an error"
    assert [row["stage"] for row in result.interrupted] == ["stopped"]
    assert "not retried" in result.interrupted[0]["detail"]


def test_a_backoff_is_cut_short_by_a_stop():
    """A retry waiting out a long backoff does not spend the drain window asleep."""
    shutdown = Shutdown()
    policy = RetryPolicy(
        RetryConfig(max_attempts=5, initial_backoff_s=30.0, max_backoff_s=30.0, jitter=0.0),
        shutdown=shutdown,
    )
    calls = []

    async def flaky():
        calls.append(time.monotonic())
        raise EndpointError("unavailable", error_class=ErrorClass.TRANSIENT)

    async def main():
        loop = asyncio.get_running_loop()
        loop.call_later(0.1, shutdown.request, "test")
        with pytest.raises(RunInterrupted):
            await with_retry(flaky, policy=policy, description="op")

    started = time.monotonic()
    asyncio.run(main())
    assert time.monotonic() - started < 5, "the backoff slept through the stop"
    assert len(calls) == 1, "the retry ran after the stop"


# --------------------------------------------------------------------------- #
# the drain window closes: cancel, await, record as interrupted
# --------------------------------------------------------------------------- #


def test_the_drain_window_closing_cancels_what_is_left(
    fake_server, write_run_config, fake_dataset
):
    """A request that will not finish in time is cancelled, not waited on --
    and is not written down as an error, because nothing failed."""
    release = threading.Event()
    harness = _Harness(
        _config(write_run_config, fake_server, fake_dataset(sample_size=4), drain_s=0.3)
    )

    def hang(conv, max_tokens):
        harness.stop()
        release.wait(timeout=20)
        return "Answer: too late"

    fake_server.state.responder = _real(hang)
    started = time.monotonic()
    try:
        result = harness.run()
    finally:
        release.set()
    elapsed = time.monotonic() - started

    assert elapsed < 10, f"the run waited {elapsed:.1f}s past a 0.3s drain window"
    assert result.shutdown["drained"] is False
    assert result.shutdown["cancelled_tasks"] == 1
    assert [row["stage"] for row in result.interrupted] == ["cancelled"]
    assert _records(result.run_dir, "fake") == [], "a cancelled request became a record"


def test_a_second_stop_request_skips_the_drain(fake_server, write_run_config, fake_dataset):
    release = threading.Event()
    harness = _Harness(
        _config(write_run_config, fake_server, fake_dataset(sample_size=4), drain_s=600)
    )

    def hang(conv, max_tokens):
        harness.stop()
        harness.stop()
        release.wait(timeout=20)
        return "Answer: too late"

    fake_server.state.responder = _real(hang)
    started = time.monotonic()
    try:
        result = harness.run()
    finally:
        release.set()
    assert time.monotonic() - started < 10, "a forced stop still waited out the drain"
    assert result.shutdown["forced"] is True
    assert result.shutdown["drained"] is False


# --------------------------------------------------------------------------- #
# interactive episodes
# --------------------------------------------------------------------------- #

EPISODES = {
    "id": "episodes",
    "impl": "test_interactive_engine:EchoEnvironmentAdapter",
    "sample_size": 3,
    "options": {},
}


def test_episodes_in_flight_when_the_stop_comes_run_to_the_end(fake_server, write_run_config):
    """Three episodes of 1, 2 and 3 turns; the stop arrives during turn one.

    All three finish -- turns two and three are the same episodes finishing,
    not new work -- and are scored as if nothing had happened.
    """
    harness = _Harness(_config(write_run_config, fake_server, EPISODES))
    turns = []

    def respond(conv, max_tokens):
        if not turns:
            harness.stop()
        turns.append(conv)
        harness.wait_until_stopping()
        return "Answer: next"

    fake_server.state.responder = _real(respond)
    result = harness.run()

    assert result.shutdown["drained"] is True
    assert not result.interrupted
    task = result.tasks[0]
    assert task.n_scored == 3
    assert task.metrics["final_answer_accuracy"] == 1.0, "an episode was cut short or scored early"
    assert len(turns) == 6          # 1 + 2 + 3


def test_episodes_the_drain_window_cuts_off_are_not_recorded_as_errors(
    fake_server, write_run_config
):
    release = threading.Event()
    harness = _Harness(_config(write_run_config, fake_server, EPISODES, drain_s=0.3))
    turns = []

    def respond(conv, max_tokens):
        turns.append(conv)
        if len(turns) == 4:        # turn two has begun: every episode is mid-flight
            harness.stop()
            release.wait(timeout=20)
        return "Answer: next"

    fake_server.state.responder = _real(respond)
    try:
        result = harness.run()
    finally:
        release.set()

    assert result.shutdown["drained"] is False
    assert [row["stage"] for row in result.interrupted] == ["cancelled"]
    records = _records(result.run_dir, "episodes")
    assert not [r for r in records if r["status"] == "error"], (
        "an episode the stop cut off was recorded as a model error"
    )


# --------------------------------------------------------------------------- #
# clients close last, and once
# --------------------------------------------------------------------------- #


def test_clients_close_once_and_only_after_nothing_is_running(
    fake_server, write_run_config, fake_dataset, monkeypatch
):
    """The ordering is the whole fix: a client closed under a live request is
    how one interrupt became 150 recorded errors."""
    closes: list[tuple[str, list[str]]] = []
    original = ModelClient.aclose

    harness: _Harness

    async def watched(self):
        current = asyncio.current_task()
        alive = [
            t.get_name() for t in asyncio.all_tasks()
            if t is not current and t is not harness.main_task and not t.done()
        ]
        closes.append((self.model.id, alive))
        await original(self)

    monkeypatch.setattr(ModelClient, "aclose", watched)
    release = threading.Event()
    harness = _Harness(
        _config(write_run_config, fake_server, fake_dataset(sample_size=8), drain_s=0.3)
    )

    def hang(conv, max_tokens):
        harness.stop()
        release.wait(timeout=20)
        return "Answer: too late"

    fake_server.state.responder = _real(hang)
    try:
        result = harness.run()
    finally:
        release.set()

    assert result.shutdown["cancelled_tasks"] == 1
    assert [model_id for model_id, _alive in closes] == ["fake-model"], closes
    for model_id, alive in closes:
        assert not alive, f"{model_id} was closed while {alive} were still running"


def test_a_run_that_finishes_normally_closes_each_client_exactly_once(
    fake_server, write_run_config, fake_dataset, monkeypatch
):
    calls: list[str] = []
    original = ModelClient.aclose

    async def counted(self):
        calls.append(self.model.id)
        await original(self)

    monkeypatch.setattr(ModelClient, "aclose", counted)
    result = _Harness(_config(write_run_config, fake_server, fake_dataset(sample_size=4))).run()
    assert result.tasks and not result.shutdown
    assert calls == ["fake-model"]


def test_closing_a_client_twice_closes_it_once():
    from abductionbench.core.config import ModelConfig, TimeoutConfig

    client = ModelClient(
        ModelConfig(id="m", model_name="m", endpoint={"base_url": "http://127.0.0.1:9"}),
        TimeoutConfig(),
    )

    async def main():
        await client.aclose()
        await client.aclose()

    asyncio.run(main())
    assert client.closed


# --------------------------------------------------------------------------- #
# real signals
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM], ids=["SIGINT", "SIGTERM"])
def test_a_signal_starts_the_drain_instead_of_ending_the_process(
    fake_server, write_run_config, fake_dataset, sig
):
    """SIGTERM by default ends the process with no cleanup at all -- no drain,
    no last checkpoint, no final upload. Both signals now begin the drain."""
    harness = _Harness(_config(write_run_config, fake_server, fake_dataset(sample_size=4)))
    sent = []

    def respond(conv, max_tokens):
        if not sent:
            sent.append(sig)
            # Only ever sent while the engine's handler is installed: a stray
            # SIGINT or SIGTERM would take the whole test session down.
            if signal.getsignal(sig) in (signal.SIG_DFL, signal.default_int_handler):
                harness.stop()   # pragma: no cover - fails the assertion below
            else:
                os.kill(os.getpid(), sig)
        harness.wait_until_stopping()
        return f"Answer: echo:{conv[-1]['content'][:80]}"

    fake_server.state.responder = _real(respond)
    result = harness.run()
    assert result.shutdown["reason"] == signal.Signals(sig).name
    assert result.shutdown["drained"] is True
    assert len(_records(result.run_dir, "fake")) == 4
    # And the handler is gone again once the run is over.
    assert signal.getsignal(sig) in (signal.SIG_DFL, signal.default_int_handler)


def test_the_cli_reports_a_stopped_run_and_exits_130(
    fake_server, write_run_config, fake_dataset, monkeypatch
):
    """Reports are written and the backup flushed, and the exit code still
    says the run did not finish."""
    from typer.testing import CliRunner

    from abductionbench.cli import app

    config_path = _config(write_run_config, fake_server, fake_dataset(sample_size=8))
    started = []

    def respond(conv, max_tokens):
        if not started:
            started.append(1)
            os.kill(os.getpid(), signal.SIGINT)
        deadline = time.monotonic() + 10
        while signal.getsignal(signal.SIGINT) is not signal.default_int_handler and (
            time.monotonic() < deadline
        ) and not _stopping():
            time.sleep(0.005)
        return f"Answer: echo:{conv[-1]['content'][:80]}"

    engines: list[EvaluationEngine] = []
    real_init = EvaluationEngine.__init__

    def capture(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        engines.append(self)

    def _stopping() -> bool:
        return bool(engines) and engines[0].shutdown.requested

    monkeypatch.setattr(EvaluationEngine, "__init__", capture)
    fake_server.state.responder = _real(respond)
    outcome = CliRunner().invoke(app, ["run", str(config_path)])

    assert outcome.exit_code == 130, outcome.output
    assert "stopped (SIGINT)" in outcome.output
    run_dir = engines[0].run_dir
    assert (run_dir / "reports").exists(), "a stopped run wrote no reports"


def test_a_stop_during_endpoint_verification_does_not_wait_for_the_probe(
    fake_server, write_run_config, fake_dataset
):
    """Before any task starts there is nothing to drain: a probe against a slow
    endpoint can take a whole read timeout, and a stop does not wait for it."""
    release = threading.Event()
    harness = _Harness(_config(write_run_config, fake_server, fake_dataset(sample_size=4)))
    answered = []

    def respond(conv, max_tokens):
        if _is_probe(conv):
            harness.stop()
            release.wait(timeout=20)
            return "pong"
        answered.append(conv)
        return "Answer: never asked"

    fake_server.state.responder = respond
    started = time.monotonic()
    try:
        result = harness.run()
    finally:
        release.set()
    assert time.monotonic() - started < 10, "the stop waited for the endpoint probe"
    assert result.shutdown["reason"] == "test"
    assert not answered and not result.tasks


def test_clients_close_last_even_when_the_run_itself_is_cancelled(
    fake_server, write_run_config, fake_dataset, monkeypatch
):
    """Not a stop request: the coroutine running the engine is cancelled from
    outside. The task coroutines it started are then nobody's -- and they are
    exactly the requests that would meet a closed client."""
    closes: list[list[str]] = []
    original = ModelClient.aclose

    handle: dict = {}

    async def watched(self):
        current = asyncio.current_task()
        closes.append([
            t.get_name() for t in asyncio.all_tasks()
            # The test's own coroutine, and the engine's task waiting for its
            # close-down to finish: neither holds anything that uses a client.
            if t is not current and t not in (handle["outside"], handle["task"])
            and not t.done()
        ])
        await original(self)

    monkeypatch.setattr(ModelClient, "aclose", watched)
    release = threading.Event()
    engine = EvaluationEngine(load_run_config(
        _config(write_run_config, fake_server, fake_dataset(sample_size=8))
    ))

    def hang(conv, max_tokens):
        # Every conversation of both batches cancels the run: repeated
        # cancellation is part of what this test puts the close-down through.
        if not handle["loop"].is_closed():
            handle["loop"].call_soon_threadsafe(handle["task"].cancel)
        release.wait(timeout=20)
        return "Answer: too late"

    fake_server.state.responder = _real(hang)

    async def main():
        handle["loop"] = asyncio.get_running_loop()
        handle["outside"] = asyncio.current_task()
        handle["task"] = asyncio.ensure_future(engine.run())
        with pytest.raises(asyncio.CancelledError):
            await handle["task"]

    try:
        asyncio.run(main())
    finally:
        release.set()
    assert closes, "the client was never closed"
    assert closes == [[]], f"closed while {closes[0]} were still running"
