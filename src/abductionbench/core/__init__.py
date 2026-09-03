"""Dataset-agnostic evaluation engine (Phase 1).

Public surface used by child adapters:

* :mod:`abductionbench.core.types` -- the data contract (``SampleSpec``,
  ``SampleScore``, ``ModelResponse``, ``AdapterDocumentation`` ...).
* :mod:`abductionbench.core.adapter` -- ``DatasetAdapter`` abstract base class
  and ``AdapterContext``.
* :mod:`abductionbench.core.metrics` -- generic, task-shaped metric primitives.
* :mod:`abductionbench.core.registry` -- adapter registration/resolution.
"""
