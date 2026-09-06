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


def test_dedupe_records_keeps_the_newest_per_request(tmp_path: Path):
    """A task re-run into the same directory must not be double-counted."""
    from abductionbench.core.checkpoint import dedupe_records

    store = RecordStore(tmp_path)
    first = _record("a", ResponseStatus.OK)
    first.metrics = {"accuracy": 0.0}
    store.append(first)
    second = _record("a", ResponseStatus.OK)  # same sample and fingerprint
    second.metrics = {"accuracy": 1.0}
    store.append(second)
    store.append(_record("b", ResponseStatus.OK))

    raw = store.existing()
    assert len(raw) == 3  # the file keeps both, by design (append-only)
    deduped = dedupe_records(raw)
    assert len(deduped) == 2
    by_id = {record["sample_id"]: record for record in deduped}
    assert by_id["a"]["metrics"]["accuracy"] == 1.0  # newest wins


def test_rebuilt_report_is_not_inflated_by_a_rerun(tmp_path: Path):
    from abductionbench.core.checkpoint import dedupe_records

    store = RecordStore(tmp_path)
    for _ in range(3):  # three passes over the same two samples
        store.append_many([_record("a", ResponseStatus.OK), _record("b", ResponseStatus.OK)])
    assert len(dedupe_records(store.existing())) == 2


def test_report_does_not_average_a_group_with_its_own_reduction(tmp_path):
    """A rebuilt report must agree with the run that produced it.

    A BOV or self-consistency task writes both the per-request members and the
    folded result. Averaging them together counts the item once per hypothesis
    or once per vote, so a rebuilt workbook would quietly disagree with the run.
    """
    import json

    from abductionbench.core.checkpoint import dedupe_records

    records = [
        # Three BOV members of one item, each scored on its own question...
        {"sample_id": "i1#bov0", "group_id": "i1", "prompt_fingerprint": "a",
         "status": "ok", "metrics": {"set_f1": 0.0}, "metadata": {}},
        {"sample_id": "i1#bov1", "group_id": "i1", "prompt_fingerprint": "b",
         "status": "ok", "metrics": {"set_f1": 0.0}, "metadata": {}},
        {"sample_id": "i1#bov2", "group_id": "i1", "prompt_fingerprint": "c",
         "status": "ok", "metrics": {"set_f1": 0.0}, "metadata": {}},
        # ... and the reduction, which is the item's actual score.
        {"sample_id": "i1", "group_id": "i1", "prompt_fingerprint": "reduced::i1",
         "status": "ok", "metrics": {"set_f1": 1.0}, "metadata": {"reduced": True}},
    ]
    path = tmp_path / "records.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")

    kept = dedupe_records(
        [json.loads(line) for line in path.read_text().splitlines()]
    )
    reduced_groups = {
        str(r.get("sample_id")) for r in kept if (r.get("metadata") or {}).get("reduced")
    }
    scored = [
        r for r in kept
        if (r.get("metadata") or {}).get("reduced")
        or str(r.get("group_id") or "") not in reduced_groups
    ]
    assert len(scored) == 1
    assert scored[0]["metrics"]["set_f1"] == 1.0


def test_a_non_finite_metric_never_reaches_the_records_file():
    """NaN is not JSON: the serializer writes null, and null cannot be read back.

    An adapter that reports NaN for "not measured" would otherwise produce a
    records file that `abench report` crashes on -- which is exactly what
    happened to a live run's BoxingGym episodes.
    """
    import math

    from abductionbench.core.types import EvalRecord, ResponseStatus, TaskIdentity

    identity = TaskIdentity(
        run_id="r", dataset_id="d", model_id="m", template_id="t", template_version="1.0"
    )
    record = EvalRecord(
        task=identity,
        sample_id="s1",
        status=ResponseStatus.OK,
        prompt_fingerprint="f",
        task_kind="generation",
        input_tokens_est=10,
        sampling={},
        response={},
        metrics={
            "good": 0.5,
            "not_measured": float("nan"),
            "diverged": float("inf"),
        },
    )
    payload = record.to_json_dict()
    assert payload["metrics"] == {"good": 0.5}

    # And what survives is round-trippable.
    import json

    reread = json.loads(json.dumps(payload))
    assert all(
        isinstance(v, (int, float)) and math.isfinite(v)
        for v in reread["metrics"].values()
    )
