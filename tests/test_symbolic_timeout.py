"""SymPy is given a deadline, because on a model's answer it may not finish.

Run ``20260921-012522_reasoning`` stopped at 05:46 on sample 49 of
``synpat/qwen3-5-2b-local/io``: one ``equal_as_zero_set`` call ran for seven
hours. SymPy is pure Python, so it held the GIL the whole time; the event loop
starved, batches stopped, and the engine reported two healthy servers as
unreachable. 203 of 211 tasks were already done and the last 8 never ran.

What these cover:

* the bound actually stops a comparison that does not terminate,
* it does not change any verdict the old code reached,
* the worker survives having been killed, and
* the watchdog names a scorer that has stopped returning.
"""

from __future__ import annotations

import logging
import time

import pytest

from abductionbench.adapters import _mathnorm, _symbolic
from abductionbench.core.telemetry import ScorerWatchdog

#: The symbol set of a real SynPAT case.
SYMBOLS = ["Fc", "Fg", "W", "d1", "d2", "m1", "G", "c",
           "dx1dt", "d2x1dt2", "dx2dt", "d2x2dt2"]
GOLD = "4*d2x1dt2*W - dx1dt*dx2dt*Fg"

#: Answers of the shape a model writes when it "factors" its rearrangement of a
#: six-variable law. Each of these ran past 45 s in the unbounded code and was
#: still going when interrupted.
NON_TERMINATING = [
    "(Fc + Fg + W + d1 + d2 + m1 + G + c)**8 - 1",
    "(Fc*d1 + Fg*d2 + W*m1 + G*c + dx1dt*dx2dt + d2x1dt2*d2x2dt2)**6",
    "((Fc+Fg)*(W+d1)*(d2+m1)*(G+c)*(dx1dt+dx2dt)*(d2x1dt2+d2x2dt2))**4 - W",
]


@pytest.fixture(autouse=True)
def _fresh_worker():
    """Each test starts and ends without a worker, so none leaks between them."""
    _symbolic.shutdown()
    yield
    _symbolic.shutdown()


# --------------------------------------------------------------------------- #
# the bound
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("candidate", NON_TERMINATING)
def test_a_comparison_that_will_not_finish_is_stopped(candidate):
    """The whole point: it returns, and it returns undecidable.

    ``None`` is the value these functions already use for "cannot be decided",
    so a timeout lands in ``symbolic_undecidable`` and is routed to the judge
    where a dataset does that. It is never a wrong answer.
    """
    started = time.monotonic()
    verdict = _mathnorm.equal_as_zero_set(candidate, GOLD, SYMBOLS, timeout_s=3)
    elapsed = time.monotonic() - started
    assert verdict is None
    # Generous: the assertion is "it came back", not "it came back fast".
    assert elapsed < 30, f"the bound did not stop it ({elapsed:.0f}s)"


def test_the_worker_still_works_after_being_killed():
    """A killed worker must not poison the rest of the task.

    Every timeout kills the worker mid-computation, and there are 150 samples
    behind it.
    """
    assert _mathnorm.equal_as_zero_set(NON_TERMINATING[0], GOLD, SYMBOLS, timeout_s=3) is None
    assert _mathnorm.equal_as_zero_set("4*d2x1dt2*W - dx1dt*dx2dt*Fg", GOLD, SYMBOLS) is True
    assert _mathnorm.equal_as_zero_set("8*d2x1dt2*W - 2*dx1dt*dx2dt*Fg", GOLD, SYMBOLS) is True
    assert _mathnorm.equal_as_zero_set("d2x1dt2*W - dx1dt*dx2dt*Fg", GOLD, SYMBOLS) is False


def test_a_timeout_says_so_in_the_log(caplog):
    """A silent undecidable would hide a scorer that had stopped working."""
    with caplog.at_level(logging.WARNING, logger="abductionbench.adapters._symbolic"):
        _mathnorm.equal_as_zero_set(NON_TERMINATING[0], GOLD, SYMBOLS, timeout_s=3)
    assert any("did not finish within" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# it must not change an answer the old code reached
# --------------------------------------------------------------------------- #


CASES = [
    # (candidate, reference, symbols) -- the ordinary path, where the bound
    # must be invisible. Verdicts are compared against the unbounded
    # implementation rather than hard-coded, so this tracks the real function.
    ("x + y", "y + x", ["x", "y"]),
    ("2*x - 2*y", "x - y", ["x", "y"]),
    ("x - y", "x + y", ["x", "y"]),
    ("Fg/Fc - dxdt/c", "c*Fg - dxdt*Fc", ["Fg", "Fc", "dxdt", "c"]),
    ("4*d2x1dt2*W - dx1dt*dx2dt*Fg", GOLD, SYMBOLS),
    ("d2x1dt2*W", GOLD, SYMBOLS),
    ("((( not an equation", "x", ["x"]),
    ("", "x", ["x"]),
]


@pytest.mark.parametrize(("candidate", "reference", "symbols"), CASES)
@pytest.mark.parametrize(
    "op", ["equal_expressions", "equal_up_to_scale", "equal_as_zero_set", "same_monomials"]
)
def test_the_bound_changes_no_verdict(op, candidate, reference, symbols):
    bounded = getattr(_mathnorm, op)(candidate, reference, symbols, timeout_s=60)
    unbounded = getattr(_mathnorm, f"_{op}")(candidate, reference, symbols)
    assert bounded is unbounded


def test_zero_disables_the_bound_and_runs_in_process():
    """The escape hatch, for reproducing a score computed before the fix."""
    assert _mathnorm.equal_expressions("x + y", "y + x", ["x", "y"], timeout_s=0) is True
    assert _symbolic._worker is None, "timeout_s=0 must not start a worker"


def test_the_environment_can_set_the_default(monkeypatch):
    monkeypatch.setenv("ABENCH_SYMBOLIC_TIMEOUT_S", "7.5")
    assert _symbolic._configured_timeout() == 7.5
    monkeypatch.setenv("ABENCH_SYMBOLIC_TIMEOUT_S", "not-a-number")
    assert _symbolic._configured_timeout() == _symbolic.DEFAULT_TIMEOUT_S
    monkeypatch.delenv("ABENCH_SYMBOLIC_TIMEOUT_S")
    assert _symbolic._configured_timeout() == _symbolic.DEFAULT_TIMEOUT_S


def test_synpat_takes_the_timeout_from_its_dataset_options():
    """So an operator can tighten it without editing code."""
    from pathlib import Path

    from abductionbench.core.adapter import AdapterContext
    from abductionbench.core.modes import TaskModes
    from abductionbench.core.registry import resolve_adapter
    from abductionbench.core.types import ModelResponse, ResponseStatus

    cls = resolve_adapter("abductionbench.adapters.synpat:SynPATAdapter")
    adapter = cls(AdapterContext(
        dataset_id="synpat", data_dir=Path("data/synpat"),
        modes=TaskModes(prompt_mode="io"), sample_size=3, seed=20260903,
        offline=True, options={"symbolic_timeout_s": 3},
    ))
    adapter.prepare()
    sample = adapter.build_samples()[0]
    sample.reference["symbols"] = SYMBOLS
    sample.reference["gold"] = GOLD
    response = ModelResponse(
        sample_id=sample.sample_id, model_id="m", status=ResponseStatus.OK,
        content=f"Answer: {NON_TERMINATING[0]}",
    )
    started = time.monotonic()
    score = adapter.score_request(sample, response, output_contract=None)
    assert time.monotonic() - started < 30, "the dataset option was not honoured"
    # Undecidable, not wrong: the scorer already models this.
    assert score.metrics["symbolic_undecidable"] == 1.0
    assert score.metrics["equation_equivalent"] == 0.0


# --------------------------------------------------------------------------- #
# the watchdog
# --------------------------------------------------------------------------- #


def test_the_watchdog_names_a_scorer_that_has_not_returned(caplog):
    watchdog = ScorerWatchdog(warn_after_s=0.5, interval_s=0.5)
    watchdog.start()
    try:
        with caplog.at_level(logging.WARNING, logger="abductionbench.core.telemetry"):
            with watchdog.watch("synpat", "sample-49"):
                time.sleep(1.5)
    finally:
        watchdog.stop()
    named = [r.message for r in caplog.records if "has not" in r.message]
    assert named, "a stuck scorer went unreported"
    assert "synpat" in named[0] and "sample-49" in named[0]


def test_the_watchdog_says_nothing_about_a_scorer_that_returns(caplog):
    watchdog = ScorerWatchdog(warn_after_s=0.5, interval_s=0.5)
    watchdog.start()
    try:
        with caplog.at_level(logging.WARNING, logger="abductionbench.core.telemetry"):
            for _ in range(5):
                with watchdog.watch("synpat", "quick"):
                    pass
            time.sleep(1.2)
    finally:
        watchdog.stop()
    assert not [r for r in caplog.records if "has not" in r.message]
