"""Put gemma-4-E4B/E2B cot replies back where they belong, and re-score them.

vLLM's gemma4 reasoning parser never found the end of the native thought on
these replies, so the WHOLE output -- native thought, then the visible
``<think>...</think><answer>...</answer>`` the cot template asks for -- landed in
``response.reasoning`` with ``content`` null. Every such record was stored as
``status: empty`` and scored as no answer.

Nothing is asked of any model. For each affected record:

* the visible reply starts at the first ``<think>``: ``content`` is from there
  on, ``reasoning`` is what came before (the native thought; None if nothing);
* a reply with no ``<think>`` at all starts at its last ``<answer>``;
* status becomes ``ok`` (``truncated`` if finish_reason is ``length``), and the
  record is re-scored by the dataset's own adapter, with the engine's
  no-answer rule applied exactly as ``EvaluationEngine._score`` applies it.

Each records.jsonl is backed up to ``records.jsonl.before-gemma-repair`` and
replaced atomically. Only the two gemma models' cot task directories are
touched. The engine is built against a scratch directory, never the run's, so
a run that is live in the same folder is not written to. Every record is
checked against the sample it is scored with (same stored ``reference``);
one mismatch in a task and that task is left alone.

    .venv/bin/python tools/repair_gemma_cot_records.py runs/<run>            # dry run
    .venv/bin/python tools/repair_gemma_cot_records.py runs/<run> --apply
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import orjson

from abductionbench.adapters._prompting import missing_answer
from abductionbench.core.config import load_run_config
from abductionbench.core.engine import EvaluationEngine
from abductionbench.core.modes import COT
from abductionbench.core.rejudge import _rendered_prompts
from abductionbench.core.types import ModelResponse, ResponseStatus

MODELS = ("gemma-4-e4b-local", "gemma-4-e2b-local")
CONFIG = "configs/runs/trio_gemma_cot_resume.yaml"
BACKUP_SUFFIX = ".before-gemma-repair"


def _norm(value):
    return orjson.loads(orjson.dumps(value, default=str))


def _broken(record: dict) -> bool:
    response = record.get("response") or {}
    return not (response.get("content") or "").strip() and bool(
        (response.get("reasoning") or "").strip()
    )


def _split(text: str) -> tuple[str, str | None, str]:
    """(content, reasoning, how) for a reply stored whole in ``reasoning``."""
    at = text.find("<think>")
    how = "think"
    if at < 0:
        at = text.rfind("<answer>")
        how = "answer"
    if at < 0:
        return "", text, "none"
    return text[at:], (text[:at].strip() or None), how


def _repair(record: dict, prompt, adapter, model_id: str) -> tuple[dict, str]:
    payload = dict(record.get("response") or {})
    content, reasoning, how = _split(payload.get("reasoning") or "")
    if not content:
        return record, how
    finish = payload.get("finish_reason")
    status = ResponseStatus.TRUNCATED if finish == "length" else ResponseStatus.OK
    payload.update(content=content, reasoning=reasoning, status=status.value)
    response = ModelResponse(
        sample_id=prompt.sample_id,
        model_id=model_id,
        status=status,
        content=content,
        reasoning=reasoning,
        finish_reason=finish,
        attempts=int(payload.get("attempts") or 1),
        latency_s=float(payload.get("latency_s") or 0.0),
        batch_id=payload.get("batch_id"),
        batch_size=payload.get("batch_size"),
        batch_index=payload.get("batch_index"),
        usage=payload.get("usage") or {},
        completion_tokens_est=payload.get("completion_tokens_est"),
    )
    # EvaluationEngine._score, minus the thread and the watchdog.
    unanswered = missing_answer(response.content, response.finish_reason)
    shown = dataclasses.replace(response, content="") if unanswered else response
    scored = adapter.score_request(prompt.sample, shown, output_contract=prompt.output_contract)
    details = dict(scored.details or {})
    parse_ok = scored.parse_ok
    if unanswered:
        parse_ok = False
        details["no_answer"] = unanswered
    repaired = dict(record)
    repaired.update(
        status=status.value,
        response=payload,
        metrics={
            k: v
            for k, v in (scored.metrics or {}).items()
            if isinstance(v, (int, float)) and math.isfinite(v)
        },
        prediction=scored.prediction,
        parse_ok=parse_ok,
        details=details,
    )
    return repaired, how


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()

    config = load_run_config(CONFIG)
    config.modes.prompt_modes = [COT]
    scratch = Path(tempfile.mkdtemp(prefix="gemma-repair-"))
    engine = EvaluationEngine(config, dry_run=True, run_dir=scratch, run_id=run_dir.name)

    totals = collections.Counter()
    for bundle in engine._build_bundles():  # noqa: SLF001
        if bundle.skipped or bundle.adapter is None or bundle.modes.prompt_mode != COT:
            continue
        by_id = {p.sample_id: p for p in _rendered_prompts(engine._build_prompt_set(bundle))}  # noqa: SLF001
        for model_id in MODELS:
            model_dir = run_dir / "datasets" / bundle.config.id / model_id
            if not model_dir.is_dir():
                continue
            for task_dir in sorted(model_dir.iterdir()):
                if not task_dir.name.startswith(f"{bundle.modes.slug}@"):
                    continue
                path = task_dir / "records.jsonl"
                if not path.exists():
                    continue
                lines = path.read_bytes().splitlines()
                out: list[bytes] = []
                counts = collections.Counter()
                problems: list[str] = []
                for line in lines:
                    if not line.strip():
                        out.append(line)
                        continue
                    record = orjson.loads(line)
                    if record.get("prompt_mode") != COT or not _broken(record):
                        counts["untouched"] += 1
                        out.append(line)
                        continue
                    prompt = by_id.get(str(record.get("sample_id")))
                    if prompt is None:
                        problems.append(f"no prompt for {record.get('sample_id')}")
                        break
                    if _norm(prompt.sample.reference) != _norm(record.get("reference")):
                        problems.append(f"reference mismatch on {record.get('sample_id')}")
                        break
                    before = record.get("metrics") or {}
                    repaired, how = _repair(record, prompt, bundle.adapter, model_id)
                    counts[f"split_{how}"] += 1
                    if repaired is record:
                        out.append(line)
                        continue
                    counts["repaired"] += 1
                    counts["answered" if not repaired["details"].get("no_answer") else "still_no_answer"] += 1
                    counts["parse_ok"] += bool(repaired["parse_ok"])
                    if set(before) - set(repaired["metrics"]):
                        counts["metric_keys_dropped"] += 1
                    out.append(orjson.dumps(repaired, default=str))
                label = f"{bundle.config.id}/{model_id}/{task_dir.name}"
                if problems:
                    print(f"SKIP {label}: {problems[0]}")
                    totals["tasks_skipped"] += 1
                    continue
                print(f"{label}: {dict(counts)}")
                for key, value in counts.items():
                    totals[f"{model_id}:{key}"] += value
                if args.apply and counts["repaired"]:
                    backup = path.with_name(path.name + BACKUP_SUFFIX)
                    if not backup.exists():
                        shutil.copy2(path, backup)
                    tmp = path.with_name(path.name + ".repair-tmp")
                    tmp.write_bytes(b"\n".join(out) + b"\n")
                    os.replace(tmp, path)
    print()
    for key in sorted(totals):
        print(f"{key}: {totals[key]}")
    print("APPLIED" if args.apply else "DRY RUN -- nothing written")
    shutil.rmtree(scratch, ignore_errors=True)
    return 1 if totals["tasks_skipped"] else 0


if __name__ == "__main__":
    sys.exit(main())
