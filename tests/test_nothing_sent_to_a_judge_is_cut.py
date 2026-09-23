"""What a judge grades, and what a vote counts, is the model's whole answer.

Adapters used to slice the text they handed a judge -- the candidate to 300-900
characters, the gold and the observation to a few hundred words -- and the
prediction they stored to 300-600. None of those limits had a reason written
down, and together they meant a judge was graded on a cut answer presented as
the model's, and self-consistency merged two different long answers that
happened to share an opening.

The judge's context window is a real constraint, and it is enforced where it
belongs: the judge stages skip a request whose fields TOGETHER would overrun it
(core/judge.py:exceeds_total), so no field has to be cut to make room.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ADAPTERS = sorted(Path("src/abductionbench/adapters").glob("*.py"))


def _constant_slices(node: ast.AST) -> list[int]:
    """Line numbers of every ``x[:N]`` / ``x[-N:]`` with a literal bound."""
    lines = []
    for child in ast.walk(node):
        if isinstance(child, ast.Subscript) and isinstance(child.slice, ast.Slice):
            bounds = (child.slice.lower, child.slice.upper)
            if any(isinstance(b, (ast.Constant, ast.UnaryOp)) for b in bounds if b is not None):
                lines.append(child.lineno)
    return lines


def _clip_calls(node: ast.AST) -> list[int]:
    return [
        child.lineno
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr == "clip_words"
    ]


@pytest.mark.parametrize("path", ADAPTERS, ids=lambda p: p.name)
def test_no_adapter_cuts_what_it_sends_a_judge(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name == "judge_request":
            cut = _constant_slices(fn) + _clip_calls(fn)
            assert not cut, f"{path.name}: judge_request cuts its fields at line(s) {cut}"


@pytest.mark.parametrize("path", ADAPTERS, ids=lambda p: p.name)
def test_no_adapter_stores_a_cut_prediction(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call):
            continue
        for keyword in call.keywords:
            if keyword.arg == "prediction":
                cut = _constant_slices(keyword.value)
                assert not cut, f"{path.name}: prediction= is cut at line(s) {cut}"
