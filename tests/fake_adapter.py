"""A synthetic dataset adapter used to test the core engine.

It lives in the test tree on purpose: the engine must be exercisable with no
real dataset present, which is the whole point of the Phase 1 / Phase 2 split.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from abductionbench.core.adapter import AdapterContext, DatasetAdapter, SkippedDataset
from abductionbench.core.metrics import aggregate_mean_metrics, exact_match, extract_answer_span
from abductionbench.core.types import (
    AdapterDocumentation,
    ModelResponse,
    SampleScore,
    SampleSpec,
)


class FakeAdapter(DatasetAdapter):
    """Yields ``n`` trivial generation samples; scores by exact match.

    Options
    -------
    ``n``               how many samples to build (default: ``sample_size``).
    ``long_every``      every k-th sample gets a huge observation (oversize test).
    ``long_chars``      size of that huge observation.
    ``poison_every``    every k-th sample carries the fake server's poison marker.
    ``skip``            raise :class:`SkippedDataset` from ``prepare``.
    ``max_tokens``      per-sample output budget requested.
    ``replacements``    how many replacement samples are available.
    """

    adapter_version = "test-1.0"
    primary_metric = "accuracy"
    # Prompts belong to the adapter now, so even the fake owns one.
    system_prompt = "You are a test adapter. Answer with the observation number."
    objective_metrics = True

    def __init__(self, context: AdapterContext):
        super().__init__(context)
        self._pool: list[SampleSpec] = []

    # -- prompts (owned here, as every adapter's are) -------------------- #

    def build_messages(self, sample):
        from abductionbench.adapters._prompting import PromptParts
        from abductionbench.adapters._prompting import build_messages as compose

        return compose(
            PromptParts(
                system=self.system_prompt_for(sample),
                observation=str(sample.fields.get("observation", "")),
                question=str(sample.fields.get("question", "")),
                options=[str(o) for o in (sample.fields.get("options") or [])],
                option_labels=[str(o) for o in (sample.fields.get("option_labels") or [])],
            ),
            self.context.modes,
        )

    # -- lifecycle ------------------------------------------------------ #

    def prepare(self) -> None:
        if self.context.option("skip"):
            raise SkippedDataset(str(self.context.option("skip")))
        total = int(self.context.option("n", self.context.sample_size))
        extra = int(self.context.option("replacements", 20))
        self._pool = [self._make(index) for index in range(total + extra)]

    def _make(self, index: int) -> SampleSpec:
        long_every = int(self.context.option("long_every", 0) or 0)
        poison_every = int(self.context.option("poison_every", 0) or 0)
        observation = f"observation number {index}"
        if long_every and index % long_every == 0:
            observation += " " + ("x " * int(self.context.option("long_chars", 80000)))
        if poison_every and index % poison_every == 0:
            observation += " POISON"
        return SampleSpec(
            sample_id=f"s{index:04d}",
            fields={"observation": observation, "question": "What explains it?"},
            reference=f"echo:{observation[:80]}".strip(),
            task_kind="generation",
            max_tokens=int(self.context.option("max_tokens", 128)),
            metadata={"index": index},
        )

    def build_samples(self) -> list[SampleSpec]:
        total = int(self.context.option("n", self.context.sample_size))
        return self._pool[:total]

    def replacement_samples(self, count: int, exclude: set[str]) -> list[SampleSpec]:
        return [s for s in self._pool if s.sample_id not in exclude][:count]

    # -- scoring -------------------------------------------------------- #

    def score(
        self,
        sample: SampleSpec,
        response: ModelResponse,
        *,
        output_contract: dict[str, Any] | None = None,
    ) -> SampleScore:
        prediction = extract_answer_span(response.text, output_contract)
        if not prediction:
            return SampleScore(metrics={"accuracy": 0.0}, prediction="", parse_ok=False)
        # The fake server echoes the last user message, which contains the
        # observation; treat a response that mentions the observation index as
        # correct so the metric is deterministic.
        correct = exact_match(
            str(sample.metadata["index"]) in prediction and "echo" in prediction, True
        )
        return SampleScore(
            metrics={"accuracy": float(correct)},
            prediction=prediction[:120],
            parse_ok=True,
        )

    def aggregate(self, scores: Sequence[SampleScore]) -> dict[str, float]:
        return aggregate_mean_metrics([s.metrics for s in scores])

    def documentation(self) -> AdapterDocumentation:
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="Fake dataset",
            domain="Testing",
            source_url="n/a",
            processing_mode="Generation",
            split_used="synthetic",
            abductive_subset="all (synthetic)",
            sampling_procedure=f"first N of a synthetic pool, seed={self.context.seed}",
            metrics_description={"accuracy": "1 if the echoed observation index is present"},
            primary_metric="accuracy",
            decisions=["Synthetic adapter used only by the core-engine test suite."],
        )
