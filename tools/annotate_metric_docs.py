"""One-shot: mark every dataset's primary metric and its direction.

Requirement 1 asks each dataset to document which metric is primary and whether
higher or lower is better.  Both facts are already declared on the adapter
(``primary_metric`` / ``primary_metric_by_mode`` and ``higher_is_better``), so
they are stamped onto the metric's own description from there rather than
retyped -- which is what keeps the documentation and the implementation from
drifting apart.

Shared engine metrics (coverage, parse_failure_rate, self_consistency_*,
best_of_n_*, repeat_agreement, ...) are deliberately NOT repeated per dataset:
they mean the same thing everywhere and are documented once in the README.
What is added per dataset is the one sentence that is dataset-specific -- how
its repeats are reduced, given the way it is scored.

Run once:  python tools/annotate_metric_docs.py
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import abductionbench.adapters as adapters_pkg  # noqa: E402
from abductionbench.core.adapter import DatasetAdapter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "src" / "abductionbench" / "adapters"


def primaries(adapter) -> list[str]:
    out = []
    value = getattr(adapter, "primary_metric", "")
    if isinstance(value, str) and value:
        out.append(value)
    by_mode = getattr(adapter, "primary_metric_by_mode", None)
    if isinstance(by_mode, dict):
        out.extend(v for v in by_mode.values() if isinstance(v, str) and v)
    return list(dict.fromkeys(out))


def reduction_note(adapter) -> tuple[str, str]:
    """How this dataset's repeats become one number, and under what name."""
    if adapter.objective_metrics:
        return (
            "self_consistency_<metric>",
            "Every metric also gets a self_consistency_ counterpart: the plurality answer "
            "over modes.repeats samples of the same record, read off those samples rather "
            "than bought again. Available because this dataset's answers are checkable and "
            "so can coincide.",
        )
    return (
        "best_of_n_<metric>",
        "Every metric also gets a best_of_n_ counterpart: per record, the repeat the judge "
        "scored highest. This dataset has no checkable answer, so a plurality is meaningless "
        "-- free-text answers never repeat verbatim -- and Best-of-N replaces it.",
    )


def annotate(source: str, adapter) -> tuple[str, int]:
    direction = "higher is better" if adapter.higher_is_better else "LOWER is better"
    changed = 0
    for metric in primaries(adapter):
        # Prefix the primary's own description, once.
        pattern = re.compile(rf'(^\s*"{re.escape(metric)}": ")(?!\(PRIMARY)', re.M)
        source, count = pattern.subn(rf'\g<1>(PRIMARY, {direction}) ', source, count=1)
        changed += count
    name, note = reduction_note(adapter)
    if "self_consistency_<metric>" not in source and "best_of_n_<metric>" not in source:
        match = re.search(r"(?ms)^(            metrics_description=\{\n)", source)
        if match:
            words = note.split()
            lines, cur = [], ""
            for word in words:
                if len(cur) + len(word) + 1 > 84:
                    lines.append(cur)
                    cur = word
                else:
                    cur = (cur + " " + word).strip()
            lines.append(cur)
            body = "".join(
                f'                "{line} "\n' if index < len(lines) - 1
                else f'                "{line}",\n'
                for index, line in enumerate(lines)
            )
            entry = f'                "{name}": "' + body.lstrip()[len('                "'):]
            entry = f'                "{name}":\n' + body
            source = source[: match.end(1)] + entry + source[match.end(1):]
            changed += 1
    return source, changed


def main() -> int:
    total = 0
    for module_info in sorted(pkgutil.iter_modules(adapters_pkg.__path__), key=lambda m: m.name):
        if module_info.name.startswith("_") or module_info.name == "unavailable":
            continue
        module = importlib.import_module(f"abductionbench.adapters.{module_info.name}")
        for _name, obj in vars(module).items():
            if (
                inspect.isclass(obj)
                and issubclass(obj, DatasetAdapter)
                and obj is not DatasetAdapter
                and obj.__module__ == module.__name__
            ):
                path = ROOT / f"{module_info.name}.py"
                source = path.read_text(encoding="utf-8")
                updated, changed = annotate(source, obj)
                if changed:
                    path.write_text(updated, encoding="utf-8")
                    total += 1
                    print(f"-> {module_info.name}: {changed} annotation(s)")
    print(f"annotated {total} adapter(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
