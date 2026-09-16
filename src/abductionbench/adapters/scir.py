"""SciR: multi-paradigm scientific inference -- causal-discovery subset.

Source: https://huggingface.co/datasets/sci-reason/scir

SciR groups tasks by inference paradigm (``causal``, ``deduction``,
``induction``).  Only the **causal** tasks are abductive: a known signalling
network is given, a new protein is observed together with measurement data, and
the model must infer which hidden causal edge explains the observations,
choosing from an explicit option list (including "None").  Deduction tasks
derive consequences from given rules and induction tasks generalize a pattern --
neither is inference to the best explanation, so both are excluded.

Each task ships two phrasings: ``problem_nl`` (real protein names) and
``problem_obfuscated`` (renamed entities).  The obfuscated variant is the
default here, because the real names are from a published network and invite
recall instead of inference; ``options.variant = nl`` switches back.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from ..core.adapter import SkippedDataset
from ..core.metrics import aggregate_mean_metrics
from ..core.types import AdapterDocumentation, ModelResponse, SampleScore, SampleSpec
from . import _common as C
from ._base import PooledDatasetAdapter, selection_score

REPO_ID = "sci-reason/scir"
_OPTIONS_RE = re.compile(r"Options:\s*\n(?P<body>(?:\s*[A-Z]\)[^\n]*\n?)+)", re.MULTILINE)
_OPTION_LINE_RE = re.compile(r"^\s*([A-Z])\)\s*(.+?)\s*$", re.MULTILINE)


class SciRAdapter(PooledDatasetAdapter):
    """Choose the hidden causal edge that explains the observed data."""

    adapter_version = "1.0"

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. You are given observational data "
        "and candidate causal edges. Choose the hidden edge whose presence explains the "
        "pattern in the data."
    )
    data_delivery_mode = "static"

    options_heading = "Answer options:"
    objective_metrics = True
    selection_cardinality = "single"
    primary_metric = "accuracy"

    def load_items(self) -> list[dict[str, Any]]:
        include_scaling = bool(self.context.option("include_difficulty_scaling", False))
        patterns = ["causal/tasks/main/*.json", "*.md"]
        if include_scaling:
            patterns.append("causal/tasks/difficulty_scaling/*.json")
        root = C.ensure_hf_snapshot(
            REPO_ID,
            self.context.data_dir / "hf",
            offline=self.context.offline,
            allow_patterns=patterns,
        )
        files = C.find_files(root / "causal", ["*.json"])
        if not files:
            raise SkippedDataset("no SciR causal task files found")
        items: list[dict[str, Any]] = []
        self._per_file: dict[str, int] = {}
        for path in files:
            payload = C.read_json(path)
            tasks = payload.get("tasks") if isinstance(payload, dict) else None
            if not isinstance(tasks, list):
                continue
            for task in tasks:
                if isinstance(task, dict):
                    items.append({**task, "task_file": path.stem})
            self._per_file[path.stem] = len(tasks)
        if not items:
            raise SkippedDataset("SciR causal files contained no tasks")
        self.split_used = (
            "causal task sets (the release ships evaluation tasks only, no train/test split): "
            + ", ".join(f"{name} ({count})" for name, count in sorted(self._per_file.items()))
        )
        return items

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        variant = str(self.context.option("variant", "obfuscated"))
        problem = C.normalize_whitespace(
            item.get("problem_obfuscated") if variant == "obfuscated" else item.get("problem_nl")
        ) or C.normalize_whitespace(item.get("problem_nl"))
        gold = C.normalize_whitespace(item.get("answer")).upper()
        if not problem or not gold:
            return None
        match = _OPTIONS_RE.search(problem)
        if not match:
            return None  # options are embedded in the prompt text; skip if absent
        options = _OPTION_LINE_RE.findall(match.group("body"))
        if len(options) < 2:
            return None
        labels = [label for label, _ in options]
        if gold not in labels:
            return None
        # Everything before "Options:" is the observation (network + data table).
        body = problem[: match.start()].strip()
        return SampleSpec(
            sample_id=C.stable_id("scir", item.get("task_file"), item.get("id", index)),
            fields={
                "observation": body,
                "question": (
                    "Which causal relationship involving the new entity best explains the "
                    "observed measurements?"
                ),
                "options": [text for _, text in options],
                "option_labels": labels,
                "instructions": (
                    "Answer 'None' if the data support no additional causal relationship."
                ),
            },
            reference={"gold_label": gold},
            task_kind="selection",
            # Reading a data table and reasoning over a graph before one label.
            max_tokens=1024,
            metadata={
                "task_file": item.get("task_file"),
                "domain": item.get("domain"),
                "n_options": len(options),
            },
        )

    def score(
        self,
        sample: SampleSpec,
        response: ModelResponse,
        *,
        output_contract: dict[str, Any] | None = None,
    ) -> SampleScore:
        score = selection_score(
            response,
            labels=sample.fields["option_labels"],
            gold_label=sample.reference["gold_label"],
            output_contract=output_contract,
            metric_name="accuracy",
        )
        task_file = sample.metadata.get("task_file")
        if task_file and "accuracy" in score.metrics:
            score.metrics[f"accuracy_{task_file}"] = score.metrics["accuracy"]
        return score

    def aggregate(self, scores: Sequence[SampleScore]) -> dict[str, float]:
        return aggregate_mean_metrics([score.metrics for score in scores])

    def documentation(self) -> AdapterDocumentation:
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="SciR",
            domain="Scientific Reasoning: Multi-Paradigm Inference",
            source_url=f"https://huggingface.co/datasets/{REPO_ID}",
            processing_mode="Selection",
            split_used=self.split_used,
            abductive_subset=(
                "The causal task family only: infer the hidden causal edge that explains "
                "observed measurements. SciR's deduction tasks (derive consequences) and "
                "induction tasks (generalize a pattern) are different inference paradigms and are "
                "excluded."
            ),
            sampling_procedure=self.sampling_note(),
            metrics_description={
                "self_consistency_<metric>":
                "Every metric also gets a self_consistency_ counterpart: the plurality answer over "
                "modes.repeats samples of the same record, read off those samples rather than bought "
                "again. Available because this dataset's answers are checkable and so can coincide.",
                "accuracy": "(PRIMARY, higher is better) 1 if the selected causal relationship is the gold one (primary); "
                "'None' is one of the options, so a model that abstains correctly is credited",
                "accuracy_<task_file>": "accuracy on one task set (the file names encode "
                "graph size and number of connections, i.e. difficulty)",
            },
            primary_metric="accuracy",
            decisions=[
                "Used the obfuscated phrasing by default so that recall of the published "
                "signalling network cannot substitute for inference; options.variant = nl "
                "switches to the real names.",
                "Parsed the option block out of the problem text (the release embeds it) and "
                "skipped any task whose options could not be parsed, instead of guessing.",
                "Included only causal/tasks/main by default; the difficulty_scaling sets are "
                "available via options.include_difficulty_scaling.",
                "The release ships evaluation tasks without splits, so those are the population.",
            ],
            caveats=[
                "Each problem embeds a numeric measurement table, so prompts are long; items over "
                "the input-token budget are replaced by the engine and the count is reported.",
                "Roughly 10 options including 'None' means chance accuracy is ~10%.",
            ],
            statistics={
                **self.base_statistics(),
                "tasks_per_file": self._per_file,
                "variant": str(self.context.option("variant", "obfuscated")),
            },
        )
