"""A deliberate no-op adapter for datasets that cannot be evaluated.

Phase 2 requires that a dataset which cannot be obtained, cannot be parsed, or
whose abductive portion cannot be identified confidently is **skipped and
reported**, not guessed at.  Pointing a dataset config at this adapter keeps the
dataset visible in the run (it appears in the ``Skipped`` sheet of the workbook
and in ``RUN_REPORT.md`` with its reason) instead of quietly disappearing from
the suite.

::

    dataset:
      id: some_dataset
      impl: abductionbench.adapters.unavailable:UnavailableAdapter
      options:
        reason: "requires credentialed access to MIMIC-IV"
        detail: "Only 4 sample notes are public; the rest is behind PhysioNet."
        source_url: "https://github.com/..."
        name: "DiReCT"
        domain: "Healthcare: Diagnostic Reasoning"
        processing_mode: "Generation"
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..core.adapter import DatasetAdapter, SkippedDataset
from ..core.types import AdapterDocumentation, ModelResponse, SampleScore, SampleSpec


class UnavailableAdapter(DatasetAdapter):
    """Always raises :class:`SkippedDataset` with the configured reason."""

    always_skips = True

    #: Never reached -- the adapter skips before a prompt is ever built -- but
    #: the base class requires it, and an explicit failure beats an empty one.
    def build_messages(self, sample):  # type: ignore[override]
        raise SkippedDataset(str(self.context.option("reason", "not available")))

    adapter_version = "1.0"
    primary_metric = ""

    def prepare(self) -> None:
        reason = str(self.context.option("reason", "not available"))
        detail = str(self.context.option("detail", "")).strip()
        message = f"{reason}" + (f" -- {detail}" if detail else "")
        raise SkippedDataset(message, dataset_id=self.dataset_id)

    def build_samples(self) -> list[SampleSpec]:  # pragma: no cover - never reached
        raise SkippedDataset(str(self.context.option("reason", "not available")))

    def score(
        self,
        sample: SampleSpec,
        response: ModelResponse,
        *,
        output_contract: dict[str, Any] | None = None,
    ) -> SampleScore:  # pragma: no cover - never reached
        return SampleScore(parse_ok=False)

    def aggregate(self, scores: Sequence[SampleScore]) -> dict[str, float]:  # pragma: no cover
        return {}

    def documentation(self) -> AdapterDocumentation:
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name=str(self.context.option("name", self.dataset_id)),
            domain=str(self.context.option("domain", "")),
            source_url=str(self.context.option("source_url", "")),
            processing_mode=str(self.context.option("processing_mode", "")),
            split_used="none -- dataset skipped",
            abductive_subset="n/a",
            sampling_procedure="n/a",
            primary_metric="",
            decisions=[f"Skipped: {self.context.option('reason', 'not available')}"],
            caveats=[str(self.context.option("detail", ""))] if self.context.option("detail") else [],
        )
