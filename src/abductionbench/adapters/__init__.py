"""Child dataset adapters (Phase 2).

One module per benchmark.  Each is resolved at runtime from a dataset config's
``impl: "abductionbench.adapters.<module>:<Class>"`` -- the core engine never
imports anything from this package, so adapters can be added, changed or removed
without touching the engine.

Shared, dataset-agnostic plumbing (downloads, file readers, split selection,
length estimation) lives in :mod:`abductionbench.adapters._common`.
"""
