"""AbductionBench: a configurable evaluation framework for abductive reasoning.

The package is split into two strictly separated layers:

* :mod:`abductionbench.core` -- the dataset-agnostic evaluation engine (Phase 1).
  It knows how to render prompts from configuration, batch them, send them to
  OpenAI-compatible endpoints, retry/checkpoint, score with adapter-provided
  scorers and report.  It contains no knowledge of any particular dataset.
* :mod:`abductionbench.adapters` -- one child adapter per benchmark (Phase 2).
  Each adapter materializes a dataset, samples from it deterministically,
  produces :class:`~abductionbench.core.types.SampleSpec` objects and scores
  model responses for that dataset's task.

Nothing in ``core`` imports from ``adapters``; adapters are resolved at runtime
from configuration (``impl: "module:ClassName"``).
"""

__version__ = "0.1.0"
