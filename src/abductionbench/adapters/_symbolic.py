"""A wall-clock bound on symbolic comparison, enforced from outside the process.

SymPy decides whether a model's equation is the reference equation, and on a
model's answer it can fail to terminate.  ``simplify``/``factor_list`` on an
expanded multivariate polynomial is exponential in the worst case, and a model
asked to rearrange a six-variable physical law regularly writes the worst case:
a product of sums raised to a power.

This is not a hypothetical.  Run ``20260921-012522_reasoning`` stopped dead at
05:46 on sample 49 of ``synpat/qwen3-5-2b-local/io``: one comparison ran for
**seven hours** without returning.  Because SymPy is pure Python it held the
GIL throughout, so the whole process starved -- the other scorers, the report
writer, and the asyncio event loop with it.  The engine's endpoint probes could
not even be scheduled, and the run logged "endpoint still unreachable" about
two servers that were answering in a millisecond.  203 of 211 tasks were
already done; the remaining 8 never ran.

**A timeout has to come from outside the interpreter.**  A thread cannot be
interrupted while it is inside a C-level or tight Python loop -- there is no
safe way to stop one in CPython -- so the comparison runs in a *subprocess*
that can simply be killed.  A persistent worker, not one process per call:
importing SymPy costs a second or two, and there are thousands of comparisons
in a run.

Costing nothing in parallelism is what makes this free: SymPy is pure Python,
so the GIL already serialized every comparison in the run.  Moving them all
into one worker process changes when they are interruptible, not how many run
at once.

A comparison that runs out of time comes back ``None`` -- the value these
functions already return for "cannot be decided" -- so it lands in the
``symbolic_undecidable`` metric and, where a dataset routes undecidables to the
judge, gets looked at by one.  It is never reported as a wrong answer.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import subprocess
import sys
import threading
from collections.abc import Sequence
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["decide", "DEFAULT_TIMEOUT_S", "OPS", "shutdown", "main"]

#: Every comparison this module will run out of process.  The names are the
#: public functions of :mod:`_mathnorm`; the worker dispatches on them.
OPS = ("equal_expressions", "equal_up_to_scale", "equal_as_zero_set", "same_monomials")

#: Seconds one comparison may take.  Measured against the real thing: the 48
#: samples that did score in the run above took 0.19-0.20 s each, so 20 s is
#: a hundredfold headroom rather than a budget anything legitimate will spend.
#: ``ABENCH_SYMBOLIC_TIMEOUT_S`` overrides it; ``0`` disables the bound and
#: restores the old in-process behaviour, including its ability to hang.
DEFAULT_TIMEOUT_S = 20.0


def _configured_timeout() -> float:
    raw = os.environ.get("ABENCH_SYMBOLIC_TIMEOUT_S")
    if raw is None:
        return DEFAULT_TIMEOUT_S
    try:
        return max(0.0, float(raw))
    except ValueError:
        logger.warning(
            "ABENCH_SYMBOLIC_TIMEOUT_S=%r is not a number; using %.1fs",
            raw, DEFAULT_TIMEOUT_S,
        )
        return DEFAULT_TIMEOUT_S


# --------------------------------------------------------------------------- #
# the parent's side: one worker, restarted whenever it has to be killed
# --------------------------------------------------------------------------- #

_lock = threading.Lock()
_worker: subprocess.Popen | None = None
#: Set once, after the first failure to start a worker, so a box where the
#: subprocess cannot run says so once instead of on every sample.
_unavailable = False
_atexit_registered = False


def _start() -> subprocess.Popen | None:
    """Launch the worker, or ``None`` if this box cannot run one."""
    global _unavailable, _atexit_registered
    if not _atexit_registered:
        # Registered on first use rather than at import, so a process that
        # never does symbolic work carries no hook. The worker is idle between
        # comparisons and would otherwise outlive the run.
        atexit.register(shutdown)
        _atexit_registered = True
    try:
        return subprocess.Popen(
            [sys.executable, "-u", "-m", "abductionbench.adapters._symbolic"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # The worker's stderr is its own; letting it inherit would
            # interleave tracebacks into the run's log for failures that are
            # already reported through the protocol.
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
        )
    except OSError as exc:
        if not _unavailable:
            logger.warning(
                "symbolic comparisons cannot be bounded on this box (%s); running them "
                "in process, where a pathological expression can hang the run", exc,
            )
            _unavailable = True
        return None


def _kill(proc: subprocess.Popen) -> None:
    """Kill the worker and release its pipes, in that order.

    The order is load-bearing. A reader thread may be blocked inside
    ``stdout.readline()`` holding the buffer's lock, and closing that stream
    first deadlocks against it -- which is exactly what the first version of
    this function did. Killing the process makes the read return EOF, the lock
    is released, and only then is the stream safe to close.
    """
    proc.kill()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:  # pragma: no cover - the OS did not reap it
        logger.warning("symbolic worker %d would not die", proc.pid)
    for stream in (proc.stdin, proc.stdout):
        try:
            if stream is not None:
                stream.close()
        except (OSError, ValueError):
            pass


def shutdown() -> None:
    """Stop the worker.  Safe to call at any time; it restarts on demand."""
    global _worker
    with _lock:
        if _worker is not None:
            _kill(_worker)
            _worker = None


def _inline(op: str, candidate: str, reference: str, symbols: Sequence[str]) -> bool | None:
    """Run the comparison here, unbounded -- the fallback, and the ``0`` path."""
    from . import _mathnorm

    return getattr(_mathnorm, f"_{op}")(candidate, reference, symbols)


def decide(
    op: str,
    candidate: str,
    reference: str,
    symbols: Sequence[str],
    *,
    timeout_s: float | None = None,
) -> bool | None:
    """One symbolic comparison, given at most ``timeout_s`` seconds.

    Returns what the comparison returned, or ``None`` when it could not be
    decided -- unparseable input, SymPy raising, or the time running out. The
    caller cannot tell those apart on purpose: all three mean the same thing to
    a scorer, and all three are already counted as undecidable.
    """
    if op not in OPS:  # pragma: no cover - programming error
        raise ValueError(f"unknown symbolic comparison {op!r}")
    budget = _configured_timeout() if timeout_s is None else max(0.0, float(timeout_s))
    if budget == 0.0 or _unavailable:
        return _inline(op, candidate, reference, symbols)

    request = json.dumps(
        {"op": op, "candidate": candidate, "reference": reference,
         "symbols": [str(s) for s in symbols]},
        ensure_ascii=True,
    )

    global _worker
    # One worker, one comparison at a time. Serializing costs nothing: SymPy is
    # pure Python, so the GIL serialized these anyway.
    with _lock:
        for attempt in (1, 2):
            if _worker is None or _worker.poll() is not None:
                _worker = _start()
                if _worker is None:
                    return _inline(op, candidate, reference, symbols)
            try:
                assert _worker.stdin is not None and _worker.stdout is not None
                _worker.stdin.write(request + "\n")
                _worker.stdin.flush()
            except (BrokenPipeError, ValueError, OSError):
                # The worker died between calls. One restart, then give up on
                # bounding this comparison rather than looping.
                _kill(_worker)
                _worker = None
                if attempt == 2:
                    return _inline(op, candidate, reference, symbols)
                continue

            line = _read_line(_worker, budget)
            if line is None:
                # Out of time, or the worker died mid-comparison. Either way it
                # cannot be trusted to answer the next one, so it is replaced.
                logger.warning(
                    "symbolic comparison %s did not finish within %.0fs; recording it as "
                    "undecidable. candidate=%r", op, budget, candidate[:160],
                )
                _kill(_worker)
                _worker = None
                return None
            try:
                reply = json.loads(line)
            except ValueError:  # pragma: no cover - the worker writes JSON or nothing
                _kill(_worker)
                _worker = None
                return None
            return reply.get("value") if reply.get("ok") else None
    return None  # pragma: no cover - unreachable


def _read_line(proc: subprocess.Popen, budget: float) -> str | None:
    """One reply, or ``None`` if it does not arrive in time.

    The read happens on a helper thread because a blocking ``readline`` cannot
    be given a deadline. The helper is left behind when the deadline passes --
    harmless, because the pipe is closed and the worker killed immediately
    after, which ends it.
    """
    box: list[str] = []

    def _pump() -> None:
        try:
            assert proc.stdout is not None
            line = proc.stdout.readline()
        except (OSError, ValueError):
            return
        if line:
            box.append(line)

    reader = threading.Thread(target=_pump, daemon=True, name="symbolic-read")
    reader.start()
    reader.join(budget)
    if reader.is_alive() or not box:
        return None
    return box[0]


# --------------------------------------------------------------------------- #
# the worker's side
# --------------------------------------------------------------------------- #


def main() -> int:
    """Read one comparison per line, answer one result per line.

    Deliberately dependency-free beyond the suite itself: this is started as
    ``python -m abductionbench.adapters._symbolic`` rather than through
    ``multiprocessing``, which would re-import ``__main__`` -- and ``__main__``
    during a run is the ``abench`` entry point.
    """
    from . import _mathnorm

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            op = request["op"]
            if op not in OPS:
                raise ValueError(f"unknown symbolic comparison {op!r}")
            value = getattr(_mathnorm, f"_{op}")(
                request["candidate"], request["reference"], request["symbols"]
            )
            reply: dict[str, Any] = {"ok": True, "value": value}
        except Exception as exc:  # noqa: BLE001 - reported, never fatal
            reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
