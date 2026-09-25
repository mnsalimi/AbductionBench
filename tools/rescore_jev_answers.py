"""Re-score jev's stored answers with the answer tag every scorer reads.

jev's choices were stored as "Answer: X". For letter labels the scorers'
lenient parser read the "A" of "Answer", so jev was scored as answering A on
every letter-labelled dataset (scir, true_detective, diagnosisarena: A on
150/150). The choice itself was right there in the record. This rewrites the
content as "<answer>X</answer>" -- what jev's client now emits -- and scores
it again with the dataset's own adapter. No request is sent.

    .venv/bin/python tools/rescore_jev_answers.py runs/<run>            # dry run: corrected scores, nothing written
    .venv/bin/python tools/rescore_jev_answers.py runs/<run> --apply    # write (backup kept)

After --apply, rerun the jev pass (tools/run_jev_from_worktree.sh): it reuses
every answer, sends nothing, and rebuilds each task's metrics.json.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import orjson

from abductionbench.core.checkpoint import dedupe_records, load_records
from abductionbench.core.config import load_run_config
from abductionbench.core.engine import EvaluationEngine
from abductionbench.core.rejudge import _rendered_prompts
from abductionbench.core.types import ModelResponse, ResponseStatus

CONFIG = "configs/runs/trio_jev_scs.yaml"
MODEL = "jev-openrouter"
OLD = re.compile(r"^\s*Answer:\s*(.+?)\s*$", re.S)


def main() -> int:
    run_dir = Path(sys.argv[1]).resolve()
    apply = "--apply" in sys.argv[2:]
    config = load_run_config(CONFIG)
    engine = EvaluationEngine(config, dry_run=True, run_dir=Path(tempfile.mkdtemp()), run_id=run_dir.name)
    summary = defaultdict(lambda: {"n": 0, "before": 0.0, "after": 0.0, "changed": 0})
    for bundle in engine._build_bundles():  # noqa: SLF001
        if bundle.skipped or bundle.adapter is None:
            continue
        by_id = {p.sample_id: p for p in _rendered_prompts(engine._build_prompt_set(bundle))}  # noqa: SLF001
        model_dir = run_dir / "datasets" / bundle.config.id / MODEL
        if not model_dir.is_dir():
            continue
        for task_dir in sorted(model_dir.iterdir()):
            if not task_dir.name.startswith(f"{bundle.modes.slug}@"):
                continue
            path = task_dir / "records.jsonl"
            rows = dedupe_records(load_records(path))
            out = []
            for row in rows:
                content = (row.get("response") or {}).get("content") or ""
                m = OLD.match(content)
                prompt = by_id.get(str(row.get("sample_id")))
                primary = "accuracy"
                s = summary[f"{bundle.config.id}/{task_dir.name.split('@')[0]}"]
                if not m or prompt is None or row.get("status") != ResponseStatus.OK.value:
                    out.append(row)
                    continue
                label = m.group(1).strip()
                new_content = f"<answer>{label}</answer>"
                response = ModelResponse(sample_id=prompt.sample_id, model_id=MODEL, status=ResponseStatus.OK,
                                         content=new_content, finish_reason="stop")
                score = bundle.adapter.score_request(prompt.sample, response, output_contract=prompt.output_contract)
                s["n"] += 1
                s["before"] += float((row.get("metrics") or {}).get(primary, 0.0))
                s["after"] += float(score.metrics.get(primary, 0.0))
                new = dict(row)
                new["response"] = {**row["response"], "content": new_content}
                new.update(metrics=dict(score.metrics), prediction=score.prediction,
                           parse_ok=score.parse_ok, details=dict(score.details))
                s["changed"] += new["prediction"] != row.get("prediction")
                out.append(new)
            if apply:
                backup = path.with_name(path.name + ".before-jev-rescore")
                if not backup.exists():
                    shutil.copy2(path, backup)
                tmp = path.with_name(path.name + ".tmp")
                tmp.write_bytes(b"".join(orjson.dumps(r, default=str) + b"\n" for r in out))
                os.replace(tmp, path)
    print(f"{'APPLIED' if apply else 'DRY RUN (nothing written)'}")
    print(f"{'task':40} {'n':>4} {'accuracy before':>16} {'after':>7} {'predictions changed':>20}")
    for task, s in sorted(summary.items()):
        if s["n"]:
            print(f"{task:40} {s['n']:>4} {s['before'] / s['n']:16.3f} {s['after'] / s['n']:7.3f} {s['changed']:>20}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
