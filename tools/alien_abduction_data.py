"""Materialize the Alien Abduction benchmark into each dataset's data directory.

Every other dataset in this suite downloads its source into ``data/<id>/``.
Alien Abduction has nothing to download -- arXiv:2608.03388 released no code and
no test instances -- so its 50 targets live in
``abductionbench.adapters._alien_targets`` and its suites are generated.  This
writes that generated benchmark out, so the data directory holds the real thing
before a run starts rather than filling in silently during one.

    python tools/alien_abduction_data.py [--cases 100] [--data-dir data]

The adapters call the same code on ``prepare()``, and regenerate whenever the
generator's fingerprint stops matching what is on disk, so running this is a
convenience and an audit point -- never a prerequisite.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from abductionbench.adapters import _alien_targets as T  # noqa: E402

DATASETS = ("alien_abduction", "alien_abduction_active", "alien_abduction_passive")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=100, help="test cases per target")
    parser.add_argument("--data-dir", default="data", help="root of the data directories")
    parser.add_argument(
        "--datasets", nargs="*", default=list(DATASETS), help="which dataset dirs to write"
    )
    args = parser.parse_args()

    root = Path(args.data_dir)
    print(f"generator {T.GENERATOR_VERSION}, fingerprint {T.fingerprint(args.cases)}")
    for dataset_id in args.datasets:
        target_dir = root / dataset_id
        manifest = T.materialize(target_dir, args.cases)
        written = sorted(p.name for p in target_dir.iterdir() if p.is_file())
        size = sum(p.stat().st_size for p in target_dir.iterdir() if p.is_file())
        print(
            f"  {dataset_id:24s} {manifest['targets']} targets, "
            f"{size / 1024:.0f} KiB  [{', '.join(written)}]"
        )

    # Read one back, so a materialization that cannot be loaded fails here
    # rather than at the start of a run.
    check = T.load_materialized(root / args.datasets[0], args.cases)
    total = sum(len(item["cases"]) for item in check)
    print(f"verified: {len(check)} targets, {total} test cases readable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
