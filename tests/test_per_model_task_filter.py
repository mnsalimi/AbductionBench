"""A model that cannot answer the whole grid is not asked the whole grid.

Most models answer every cell a run plans, so they carry no filter. jev is the
reason this exists: it reaches a decision endpoint that takes the answer
options as structured criteria, so it has nothing to do on a generation task,
and the chain-of-thought instruction that defines a cot cell lives in a
rendered prompt it never receives.

The alternative was a second run config and a second process -- which is what
was being done, and two runs mean two sync threads writing one Drive folder.
Drive allows two files to share a name, so the second writer creates duplicates
that rclone then refuses to touch: 95 such paths, and a night of results left
un-backed-up while every pass reported success. Restricting the model keeps it
to one run, one sync, one writer.
"""

from __future__ import annotations

import pytest

from abductionbench.core.config import ModelTaskFilter, load_layered

SELECTION = {"prompt_mode": "io", "selection_mode": "SCS", "task_kinds": ["selection"]}


def _jev_filter() -> ModelTaskFilter:
    raw = load_layered("configs/models/jev-openrouter.yaml")["model"]["only_tasks"]
    return ModelTaskFilter.model_validate(raw)


def test_an_empty_filter_admits_everything():
    """The default: a model with no restriction answers the whole grid."""
    blank = ModelTaskFilter()
    assert blank.admits(prompt_mode="cot", selection_mode="MCS", task_kinds=["generation"])
    assert blank.admits(prompt_mode="io", selection_mode="n/a", task_kinds=["selection"])


def test_jev_is_asked_the_scs_selection_cell():
    assert _jev_filter().admits(**SELECTION)


@pytest.mark.parametrize(
    ("label", "cell"),
    [
        ("cot", {**SELECTION, "prompt_mode": "cot"}),
        ("MCS", {**SELECTION, "selection_mode": "MCS"}),
        ("BOV", {**SELECTION, "selection_mode": "BOV"}),
        ("no selection mode", {**SELECTION, "selection_mode": "n/a"}),
        ("generation", {**SELECTION, "task_kinds": ["generation"]}),
    ],
)
def test_jev_is_not_asked_the_cells_it_cannot_answer(label, cell):
    assert not _jev_filter().admits(**cell), label


def test_one_axis_can_be_pinned_while_the_others_stay_open():
    """So a filter says only what it means to say."""
    only_io = ModelTaskFilter(prompt_modes=["io"])
    assert only_io.admits(prompt_mode="io", selection_mode="MCS", task_kinds=["generation"])
    assert not only_io.admits(prompt_mode="cot", selection_mode="MCS", task_kinds=["generation"])


def test_a_mixed_kind_task_is_admitted_if_any_of_its_kinds_is_allowed():
    """Dropping a whole task over one stray sample would lose the rest of it,
    and the per-sample kind is on every record either way."""
    selection_only = ModelTaskFilter(task_kinds=["selection"])
    assert selection_only.admits(
        prompt_mode="io", selection_mode="SCS", task_kinds=["generation", "selection"]
    )
    assert not selection_only.admits(
        prompt_mode="io", selection_mode="SCS", task_kinds=["generation"]
    )


def test_the_planner_skips_a_model_outside_its_filter():
    """The filter has to be consulted where tasks are planned, not just parsed."""
    import inspect

    from abductionbench.core.engine import EvaluationEngine

    source = inspect.getsource(EvaluationEngine._plan_tasks)
    assert "only_tasks.admits(" in source
    assert "continue" in source


def test_the_shipped_run_asks_qwen_everything_and_jev_only_scs():
    """The two configs have to agree, and nothing else checks that they do."""
    run = load_layered("configs/runs/generate_only.yaml")
    files = [entry["file"] for entry in run["models"]]
    assert any("qwen3-5-27b-local" in f for f in files)
    assert any("jev-openrouter" in f for f in files)

    qwen = load_layered("configs/models/qwen3-5-27b-local.yaml")["model"]
    assert not qwen.get("only_tasks"), "the local model must answer the whole grid"
