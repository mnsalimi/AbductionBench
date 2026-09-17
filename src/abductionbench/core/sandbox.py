"""Run model-written Python against a test suite, out of process and bounded.

One benchmark in this suite is scored by *executing* what the model wrote:
Alien Abduction asks for a Python function and counts the episode solved only
when that function agrees with the hidden target on a held-out suite.  Nothing
else here executes model output, so the boundary is drawn once, in this module,
and stated honestly rather than implied.

**What this is.**  Each candidate runs in a fresh interpreter started with
``-I`` (isolated: no user site-packages, no ``PYTHONPATH``, no ``PYTHON*``
environment), in its own session and its own empty working directory, under
``RLIMIT_CPU``, ``RLIMIT_AS`` and ``RLIMIT_FSIZE``, with a wall-clock deadline
after which the whole process *group* is killed.  That contains what actually
goes wrong when a language model writes a function: infinite loops, runaway
recursion, allocation blow-ups, a stray ``input()``, and a crash that would
otherwise take the evaluation down with it.  It is the same shape as the
reference implementation shipped with HumanEval, for the same reason.

**What this is not.**  It is not a security boundary.  The candidate is a normal
process with this container's privileges: it can open sockets, read the
filesystem and spend the file descriptors it is given.  The paper's own harness
runs each episode in an ephemeral container, which *is* a boundary; that is not
available here (this is an unprivileged container, so no nested container engine
runs), and pretending otherwise in a docstring would be worse than saying so.
Run this against models you are willing to run code from.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Verification", "verify", "DEFAULT_TIMEOUT_S", "DEFAULT_MEMORY_MB"]

#: Wall-clock ceiling for one candidate against one whole suite.  The suites are
#: a hundred calls to a primitive function, so a candidate that has not finished
#: in this long is looping, not working.
DEFAULT_TIMEOUT_S = 10.0
#: Address-space ceiling.  Generous enough for the interpreter plus the suite,
#: tight enough that ``[0] * 10**12`` fails instead of swapping the box.
DEFAULT_MEMORY_MB = 512
#: How many disagreements are reported back.  The Game Master shows the player a
#: few failing cases when it rejects a submission, as the paper's transcripts do.
_MAX_FAILURES = 5

#: The child prints exactly one line beginning with this, so a candidate that
#: prints on its own cannot be mistaken for the result.
_SENTINEL = "__ABENCH_RESULT__"

_RUNNER = r'''
import hashlib, json, sys

def _agrees(got, expected):
    # bool and int compare equal in Python (True == 1). For a benchmark whose
    # domains are separated by return type, that would credit a Boolean answer
    # to an arithmetic target, so the two are kept apart explicitly.
    if isinstance(expected, bool) != isinstance(got, bool):
        return False
    return got == expected

payload = json.loads(sys.stdin.read())
result = {"status": "error", "passed": 0, "total": len(payload["cases"]),
          "failures": [], "error": "", "digest": ""}
namespace = {"__name__": "__candidate__"}
try:
    exec(compile(payload["source"], "<candidate>", "exec"), namespace)
except BaseException as exc:
    result["error"] = "%s: %s" % (type(exc).__name__, exc)
    print(SENTINEL + json.dumps(result))
    raise SystemExit(0)

fn = namespace.get(payload["entrypoint"])
if not callable(fn):
    candidates = [v for k, v in namespace.items()
                  if callable(v) and getattr(v, "__module__", None) == "__candidate__"
                  and not k.startswith("_")]
    fn = candidates[0] if len(candidates) == 1 else None
if not callable(fn):
    result["status"] = "no_function"
    result["error"] = "no function named %r was defined" % payload["entrypoint"]
    print(SENTINEL + json.dumps(result))
    raise SystemExit(0)

passed = 0
failures = []
behaviour = hashlib.sha1()
for args, expected in payload["cases"]:
    try:
        got = fn(*args)
    except BaseException as exc:
        behaviour.update(b"!" + type(exc).__name__.encode())
        if len(failures) < payload["max_failures"]:
            failures.append({"input": args, "expected": expected,
                             "got": "%s: %s" % (type(exc).__name__, exc), "raised": True})
        continue
    # What the candidate *computed*, not whether it was right: two submissions
    # with the same digest are the same function on this suite, which is what a
    # self-consistency vote over free-written code has to be taken over.
    behaviour.update(b"|" + repr(got).encode("utf-8", "replace"))
    if _agrees(got, expected):
        passed += 1
    elif len(failures) < payload["max_failures"]:
        try:
            shown = json.loads(json.dumps(got))
        except (TypeError, ValueError):
            shown = repr(got)
        failures.append({"input": args, "expected": expected, "got": shown, "raised": False})

result["passed"] = passed
result["digest"] = behaviour.hexdigest()[:16]
result["failures"] = failures[: payload["max_failures"]]
result["status"] = "passed" if passed == result["total"] else "failed"
print(SENTINEL + json.dumps(result))
'''


@dataclass(slots=True)
class Verification:
    """What running one candidate against one suite established."""

    #: ``passed`` | ``failed`` | ``error`` | ``timeout`` | ``no_function`` | ``crashed``
    status: str
    passed: int = 0
    total: int = 0
    failures: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""
    #: A hash of what the candidate returned for every case in the suite, so two
    #: submissions can be compared by what they *compute* rather than by how
    #: they were written.  Empty when nothing ran.
    digest: str = ""

    @property
    def solved(self) -> bool:
        """The paper's criterion: agreement on *every* held-out case."""
        return self.status == "passed" and self.total > 0

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


def _limits(memory_mb: int, timeout_s: float):
    """The child's own first act: put itself in a new session and take its limits."""

    def apply() -> None:  # pragma: no cover - runs in the forked child
        import resource

        os.setsid()  # so a candidate that forks is still killable as one group
        cpu = max(1, int(timeout_s))
        for what, limit in (
            (resource.RLIMIT_CPU, (cpu, cpu + 1)),
            (resource.RLIMIT_AS, (memory_mb * 1024 * 1024,) * 2),
            (resource.RLIMIT_FSIZE, (0, 0)),
            (resource.RLIMIT_CORE, (0, 0)),
        ):
            try:
                resource.setrlimit(what, limit)
            except (ValueError, OSError):
                # A limit the platform will not take is not a reason to skip the
                # ones it will; the wall-clock deadline still applies.
                pass

    return apply


def verify(
    source: str,
    cases: list[tuple[tuple[Any, ...], Any]],
    *,
    entrypoint: str = "solution",
    timeout_s: float = DEFAULT_TIMEOUT_S,
    memory_mb: int = DEFAULT_MEMORY_MB,
    max_failures: int = _MAX_FAILURES,
) -> Verification:
    """Run ``source`` against ``cases`` and report what agreed.

    Never raises for a bad candidate: an unparseable, crashing, looping or
    silent submission comes back as a :class:`Verification` with the reason in
    ``status``, because failing to write a working function is a result of the
    benchmark and not an error of the harness.
    """
    total = len(cases)
    if not source.strip():
        return Verification(status="no_function", total=total, error="no code was submitted")

    payload = json.dumps(
        {
            "source": source,
            "entrypoint": entrypoint,
            "cases": [[list(args), expected] for args, expected in cases],
            "max_failures": max_failures,
        }
    )
    runner = f"SENTINEL = {_SENTINEL!r}\n{_RUNNER}"
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", runner],
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=os.devnull.rsplit("/", 1)[0],  # /dev: present, and nothing to read
            preexec_fn=_limits(memory_mb, timeout_s),  # noqa: PLW1509 - the point of the call
            env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent", "LC_ALL": "C.UTF-8"},
        )
    except subprocess.TimeoutExpired:
        return Verification(
            status="timeout",
            total=total,
            error=f"the candidate did not finish within {timeout_s:g}s",
        )
    except OSError as exc:
        return Verification(status="error", total=total, error=f"could not start a runner: {exc}")

    for line in reversed((completed.stdout or "").splitlines()):
        if line.startswith(_SENTINEL):
            try:
                blob = json.loads(line[len(_SENTINEL) :])
            except json.JSONDecodeError:
                break
            return Verification(
                status=str(blob.get("status", "error")),
                passed=int(blob.get("passed", 0)),
                total=int(blob.get("total", total)),
                failures=list(blob.get("failures", [])),
                error=str(blob.get("error", "")),
                digest=str(blob.get("digest", "")),
            )

    # No result line at all: the interpreter died before it could print one.
    # RLIMIT_CPU usually bites before the wall-clock deadline does, and it
    # arrives as SIGXCPU -- which is a loop that ran out of time, so it is
    # reported as one rather than as an unexplained crash.
    if completed.returncode == -signal.SIGXCPU:
        return Verification(
            status="timeout",
            total=total,
            error=f"the candidate exhausted its {timeout_s:g}s of CPU",
        )
    detail = (completed.stderr or "").strip().splitlines()
    return Verification(
        status="crashed",
        total=total,
        error=detail[-1] if detail else f"the runner exited with code {completed.returncode}",
    )
