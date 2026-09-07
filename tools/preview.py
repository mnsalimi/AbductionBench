"""Preview one dataset adapter: sample statistics and a rendered prompt.

    python tools/preview.py configs/runs/pilot.yaml <dataset_id> [sample_index]

Prints how many samples the adapter built, the task-kind mix, per-sample
max_tokens and input-token statistics under the run's configured tokenizer, the
adapter's own documentation, and one fully rendered prompt with its reference --
i.e. everything needed to check an adapter without calling a model.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from abductionbench.core.config import load_run_config  # noqa: E402
from abductionbench.core.engine import EvaluationEngine  # noqa: E402
from abductionbench.core.metrics import summarize_numeric  # noqa: E402
from abductionbench.core.telemetry import setup_logging  # noqa: E402


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    config_path, dataset_id = sys.argv[1], sys.argv[2]
    index = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    setup_logging(level="INFO")
    config = load_run_config(config_path, dataset_filter=[dataset_id])
    engine = EvaluationEngine(config, dry_run=True)
    bundle = engine._build_bundles()[0]  # noqa: SLF001 - intentional reuse of stage A

    if bundle.skipped:
        print(f"\nSKIPPED: {bundle.config.id}: {bundle.skipped_reason}")
        return 1

    samples = bundle.samples
    print(f"\n=== {bundle.config.id}: {len(samples)} sample(s) built ===")
    print("task kinds:", dict(Counter(s.task_kind for s in samples)))
    # The output budget is the model's remaining window, decided per model at
    # run time, so it is not a property of a sample any more.

    # The adapter owns its prompts now, so a prompt set is built from the
    # bundle alone -- there is no template binding to look up.
    prompt_set = engine._build_prompt_set(bundle)  # noqa: SLF001
    tokens = [t for _, _, t in prompt_set.entries]
    print("template:", prompt_set.template.ref)
    print("input tokens:", summarize_numeric(tokens))
    print(
        f"oversize dropped: {len(prompt_set.oversize_dropped)}, "
        f"replacements: {prompt_set.replacements_used}, kept: {prompt_set.size}"
    )
    if prompt_set.unusable_reason:
        print("UNUSABLE:", prompt_set.unusable_reason)

    doc = bundle.documentation
    if doc:
        print("\n--- adapter documentation ---")
        print("split_used:      ", doc.split_used)
        print("abductive_subset:", doc.abductive_subset)
        print("sampling:        ", doc.sampling_procedure)
        print("primary_metric:  ", doc.primary_metric)
        for name, description in doc.metrics_description.items():
            print(f"  metric {name}: {description}")
        for decision in doc.decisions:
            print("  decision:", decision)
        for caveat in doc.caveats:
            print("  caveat:  ", caveat)
        print("statistics:", doc.statistics)

    if index < len(prompt_set.entries):
        sample, messages, token_count = prompt_set.entries[index]
        print(f"\n--- rendered prompt for {sample.sample_id} ({token_count} input tokens) ---")
        for message in messages:
            print(f"\n[{message.role}]\n{message.content}")
        print(f"\n[reference]\n{sample.reference}")
        print(f"\n[metadata] {sample.metadata}")
        print(f"[max_tokens] {sample.max_tokens}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
