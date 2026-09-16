"""Report construction: what the workbook shows when it cannot show everything."""

from __future__ import annotations


def test_sample_sheet_keeps_every_task_when_it_has_to_truncate():
    """Filling in task order spent the whole cap on the first task.

    The later tasks -- the cot ones, since io runs first -- then had no rows at
    all, which reads as "that task produced nothing" rather than "the sheet is
    capped".
    """
    from abductionbench.core.reporting import _rows_per_task

    # Three tasks, 100 records each, room for 60 rows.
    allowed = _rows_per_task([[{}] * 100, [{}] * 100, [{}] * 100], 60)
    assert sum(allowed) == 60
    assert all(count > 0 for count in allowed), "no task may be left out entirely"

    # A task with less than its share hands the remainder back to the others.
    allowed = _rows_per_task([[{}] * 2, [{}] * 100, [{}] * 100], 60)
    assert allowed[0] == 2
    assert sum(allowed) == 60

    # Under the cap, nothing is dropped.
    assert _rows_per_task([[{}] * 10, [{}] * 10], 100) == [10, 10]
