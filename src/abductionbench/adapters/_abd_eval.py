"""Running ABD's Z3 evaluator where it cannot take the run down with it.

The release's evaluator is exact and is the right thing to score with, but it
is a Python wrapper over a native solver, and a native solver can *abort the
process*.  Observed, not hypothesised: on a model answer that quantified over
the free variable, z3-solver 5.1.0 raised

    ASSERTION VIOLATION
    File: .../src/ast/ast.cpp Line: 375
    UNEXPECTED CODE WAS REACHED

which is a C++ ``abort``, not a Python exception.  No ``try``/``except`` around
the call can catch it, and it killed a benchmark run at the scoring step.  The
completion that did it was not among the ones the run had written to disk, so
it is not reproducible on demand and the cause is unconfirmed; ``z3-solver`` is
capped below 5 as a precaution rather than a diagnosed fix (4.16 and 5.1 agree
on every verdict checked).

So evaluation happens in a **separate process**.  One worker is kept alive
across calls, so the import and the solver warm-up are paid once; if it dies --
abort, segfault, or a hang past the deadline -- the failure is reported for
that one answer and a fresh worker is started for the next.  A scorer is not
allowed to end a run.
"""

from __future__ import annotations

import logging
import sys
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor, TimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Wall-clock ceiling for one evaluation, well above the Z3 timeout the
#: evaluator itself applies per query. Reaching this means the worker is stuck,
#: not that the instance is hard.
DEADLINE_S = 180.0


@dataclass(frozen=True)
class EvalOutcome:
    """What the worker got back, flattened so it crosses a process boundary.

    ``crashed`` is the case the wrapper exists for: the evaluator did not
    return a verdict, so the answer has no score rather than a zero -- a
    solver that fell over is not evidence about the model.
    """

    valid: bool = False
    parse_error: str | None = None
    total_cost: int | None = None
    total_opt_cost: int | None = None
    total_gap: int | None = None
    avg_gap: float | None = None
    cost_vs_gold: int | None = None
    forbidden_preds_used: tuple[str, ...] = ()
    trailing_parens_added: int = 0
    crashed: bool = False
    crash_reason: str = ""


def _worker(source: str, problem: dict[str, Any], alpha: str, timeout_ms: int) -> dict[str, Any]:
    """Evaluate one formula. Runs in the child process."""
    if source not in sys.path:
        sys.path.insert(0, source)
    from concept_synth.abduction.evaluate_abd_b1 import evaluate_abd_b1

    result = evaluate_abd_b1({"problem": problem}, alpha, timeout_ms=timeout_ms)
    return {
        "valid": bool(result.valid),
        "parse_error": result.parse_error,
        "total_cost": result.total_cost,
        "total_opt_cost": result.total_opt_cost,
        "total_gap": result.total_gap,
        "avg_gap": result.avg_gap,
        "cost_vs_gold": result.cost_vs_gold,
        "forbidden_preds_used": tuple(result.forbidden_preds_used or ()),
        "trailing_parens_added": int(getattr(result, "trailing_parens_added", 0) or 0),
    }


class IsolatedEvaluator:
    """A restartable one-process pool around the release's evaluator."""

    def __init__(self, release_source: Path, *, deadline_s: float = DEADLINE_S) -> None:
        self.source = str(release_source)
        self.deadline_s = deadline_s
        self._pool: ProcessPoolExecutor | None = None
        #: How many times the worker had to be replaced, reported as a metric
        #: so a run that quietly lost verdicts cannot look like a clean one.
        self.crashes = 0

    def _ensure_pool(self) -> ProcessPoolExecutor:
        if self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=1)
        return self._pool

    def _discard_pool(self) -> None:
        pool, self._pool = self._pool, None
        if pool is not None:
            try:
                pool.shutdown(wait=False, cancel_futures=True)
            except Exception:  # noqa: BLE001 - it is already broken
                pass

    def evaluate(self, problem: dict[str, Any], alpha: str, timeout_ms: int) -> EvalOutcome:
        try:
            future = self._ensure_pool().submit(_worker, self.source, problem, alpha, timeout_ms)
            return EvalOutcome(**future.result(timeout=self.deadline_s))
        except (BrokenExecutor, TimeoutError) as exc:
            # The native solver aborted, or wedged. Replace the worker; the
            # next answer is scored normally.
            self.crashes += 1
            self._discard_pool()
            reason = (
                f"the solver stopped responding after {self.deadline_s:.0f}s"
                if isinstance(exc, TimeoutError)
                else f"the solver process died ({type(exc).__name__})"
            )
            logger.warning("ABD: %s while evaluating %r; worker restarted", reason, alpha[:120])
            return EvalOutcome(crashed=True, crash_reason=reason)
        except Exception as exc:  # noqa: BLE001 - a scorer never ends a run
            logger.warning("ABD: evaluator raised for %r: %s", alpha[:120], exc)
            return EvalOutcome(crashed=True, crash_reason=f"{type(exc).__name__}: {exc}")

    def close(self) -> None:
        self._discard_pool()
