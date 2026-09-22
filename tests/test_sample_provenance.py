"""Which records ran, and how to run the next 50 without repeating them.

"50 records per dataset" is not a claim anyone can check against a results
sheet. The draw is one seeded shuffle of the whole split whose prefix is the
evaluation set, so records [offset, offset + size) of that same shuffle are a
set with nothing in common with [0, offset) -- which is what makes a second run
EXTEND the first rather than measure it twice.

Two things have to be true for that to work, and both are tested here: the
offset has to produce a disjoint set, and the run has to write down what it
took so the next offset can be read off it rather than remembered.
"""

from __future__ import annotations

import json
from pathlib import Path

from abductionbench.adapters._base import PooledDatasetAdapter
from abductionbench.core.adapter import AdapterContext
from abductionbench.core.types import SampleSpec


class _Adapter(PooledDatasetAdapter):
    dataset_id = "fake"
    adapter_version = "1.0"

    def load_items(self):
        self.split_used = "test"
        return list(range(200))

    def make_sample(self, item, index):
        # Every seventh item declines to build, so "skip N built samples" and
        # "skip N pool positions" are different numbers -- which is the case
        # the offset has to get right.
        if item % 7 == 0:
            return None
        return SampleSpec(sample_id=f"s-{item}", fields={"observation": str(item)})

    def score(self, sample, response):  # pragma: no cover - not exercised here
        raise NotImplementedError

    def documentation(self):  # pragma: no cover - not exercised here
        raise NotImplementedError


def _draw(size: int, offset: int) -> _Adapter:
    adapter = _Adapter(
        AdapterContext(
            dataset_id="fake", data_dir=Path("."), sample_size=size,
            sample_offset=offset, seed=20260903,
        )
    )
    adapter.prepare()
    adapter.build_samples()
    return adapter


def test_the_second_draw_shares_no_record_with_the_first():
    """The whole point: 50 + 50 is 100 distinct records, not 50 measured twice."""
    first = set(_draw(50, 0)._built)
    second = set(_draw(50, 50)._built)
    assert len(first) == 50 and len(second) == 50
    assert first & second == set(), sorted(first & second)[:5]
    assert len(first | second) == 100


def test_a_third_draw_continues_past_both():
    first, second, third = (set(_draw(20, off)._built) for off in (0, 20, 40))
    assert not (first & second) and not (second & third) and not (first & third)
    assert len(first | second | third) == 60


def test_the_offset_counts_built_samples_not_pool_positions():
    """An item the adapter declines to build is skipped by both runs alike.

    Counting pool positions would let the two draws drift into each other
    exactly where `make_sample` returns None.
    """
    first = _draw(50, 0)
    second = _draw(50, 50)
    # The second draw starts at the pool position after the first one ended,
    # having stepped over the same declined items.
    assert second._drawn_pool_indices[0] > first._drawn_pool_indices[-1]


def test_offset_zero_is_the_draw_that_was_always_taken():
    """The default must not move a single existing run's sample set."""
    assert _draw(50, 0)._built == _draw(50, 0)._built
    unoffset = _Adapter(
        AdapterContext(dataset_id="fake", data_dir=Path("."), sample_size=50, seed=20260903)
    )
    unoffset.prepare()
    unoffset.build_samples()
    assert unoffset._built == _draw(50, 0)._built


def test_the_draw_record_says_where_the_next_run_should_start():
    record = _draw(50, 0).draw_record()
    assert record["n_drawn"] == 50
    assert record["sample_offset"] == 0
    assert record["next_offset"] == 50
    assert record["seed"] == 20260903
    assert record["salt"] == "fake"
    assert len(record["sample_ids"]) == 50

    # And following it lands exactly on the disjoint set.
    following = _draw(50, record["next_offset"])
    assert set(following._built).isdisjoint(record["sample_ids"])
    assert following.draw_record()["next_offset"] == 100


def test_a_short_split_reports_what_it_actually_drew():
    """`next_offset` has to follow the draw, not the request.

    Asking for 50 and getting 30 means the next run starts at 30; starting at
    50 would silently skip 20 records that were never evaluated.
    """
    adapter = _draw(500, 0)
    record = adapter.draw_record()
    assert record["n_drawn"] < 500
    assert record["next_offset"] == record["n_drawn"]


def test_the_manifest_is_written_with_the_models_that_saw_the_draw(tmp_path):
    """Every model sees the identical draw -- the prompt set is built once.

    So the models belong on the row rather than as a row apiece, and the row
    says so instead of leaving a reader to assume it.
    """
    from abductionbench.core.engine import EvaluationEngine

    engine = object.__new__(EvaluationEngine)
    engine.run_id = "r1"
    engine.run_dir = tmp_path

    class _Cfg:
        id = "fake"

        @staticmethod
        def evaluated_models():
            return [type("M", (), {"id": "m1"}), type("M", (), {"id": "m2"})]

    engine.config = _Cfg()
    adapter = _draw(10, 0)
    EvaluationEngine._record_draw(engine, adapter, _Cfg(), [])

    rows = [json.loads(line) for line in (tmp_path / "sample_manifest.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["schema"] == "sample_manifest/v1"
    assert row["run_id"] == "r1"
    assert row["models"] == ["m1", "m2"]
    assert row["dataset_id"] == "fake"
    assert len(row["sample_ids"]) == 10
    assert row["next_offset"] == 10
