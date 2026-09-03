"""Record store: atomic appends, resume policies, corrupt-line tolerance."""

from __future__ import annotations

from pathlib import Path

from abductionbench.core.checkpoint import RecordStore, TaskCheckpoint, load_records
from abductionbench.core.types import EvalRecord, ResponseStatus, TaskIdentity


def _identity() -> TaskIdentity:
    return TaskIdentity(
        run_id="r", dataset_id="d", model_id="m", template_id="t", template_version="1.0"
    )


def _record(sample_id: str, status: ResponseStatus, fingerprint: str = "fp") -> EvalRecord:
    return EvalRecord(
        task=_identity(),
        sample_id=sample_id,
        status=status,
        prompt_fingerprint=fingerprint,
        task_kind="generation",
        input_tokens_est=10,
        sampling={"max_tokens": 64},
        response={"content": "x"},
        metrics={"accuracy": 1.0},
    )


def test_append_and_reload(tmp_path: Path):
    store = RecordStore(tmp_path)
    store.append(_record("a", ResponseStatus.OK))
    store.append_many([_record("b", ResponseStatus.OK), _record("c", ResponseStatus.ERROR)])
    records = store.existing()
    assert [r["sample_id"] for r in records] == ["a", "b", "c"]
    assert records[0]["metrics"]["accuracy"] == 1.0


def test_resume_policies(tmp_path: Path):
    store = RecordStore(tmp_path)
    store.append(_record("ok", ResponseStatus.OK, fingerprint="fp1"))
    store.append(_record("err", ResponseStatus.ERROR, fingerprint="fp1"))
    store.append(_record("skip", ResponseStatus.SKIPPED, fingerprint="fp1"))

    strict = store.completed_keys(policy="strict")
    # Errors are never reused (a resumed run must retry them); skips are.
    assert set(strict) == {"ok::fp1", "skip::fp1"}

    by_id = store.completed_keys(policy="sample_id")
    assert set(by_id) == {"ok", "skip"}

    assert store.completed_keys(policy="off") == {}


def test_truncated_final_line_is_tolerated(tmp_path: Path):
    store = RecordStore(tmp_path)
    store.append(_record("a", ResponseStatus.OK))
    with store.records_path.open("ab") as handle:
        handle.write(b'{"sample_id": "partial", "sta')
    records = load_records(store.records_path)
    assert [r["sample_id"] for r in records] == ["a"]


def test_checkpoint_roundtrip_and_raw_payloads(tmp_path: Path):
    store = RecordStore(tmp_path, store_raw_payloads=True)
    checkpoint = TaskCheckpoint(task=_identity().as_dict(), total_planned=5, completed=2)
    store.save_checkpoint(checkpoint)
    loaded = store.load_checkpoint()
    assert loaded["total_planned"] == 5 and loaded["completed"] == 2

    store.save_raw("batch-1", {"request": {"a": 1}, "response": {"b": 2}})
    assert (tmp_path / "raw" / "batch-1.json").exists()

    store.write_json("metrics.json", {"accuracy": 0.5})
    assert (tmp_path / "metrics.json").exists()


def test_raw_payload_cap(tmp_path: Path):
    store = RecordStore(tmp_path, store_raw_payloads=True, max_raw_payloads=2)
    for index in range(5):
        store.save_raw(f"b{index}", {"i": index})
    assert len(list((tmp_path / "raw").glob("*.json"))) == 2
