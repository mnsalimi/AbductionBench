"""AgentRx: diagnosing why an AI agent's run failed.

Source: https://github.com/microsoft/AgentRx

The ground-truth files annotate agent trajectories: a ``failure_summary``
describing what went wrong, a list of ``failures`` (each with a step number, a
reason and a ``failure_category``), and a ``root_cause`` naming which of those
failures was decisive.

**The abductive item.** Given the observed failure (and, optionally, an excerpt
of the trajectory), select the category that explains it -- the root-cause
failure's category.  The candidate set is the union of failure categories that
appear in the ground truth, i.e. the dataset's own taxonomy; no categories are
invented.  (The repository notes that a formal taxonomy file is not published,
so deriving it from the labels is the only faithful option.)

Full trajectories run to 100-180 KB of JSON, well over the input budget, so by
default the observation is the failure summary; ``options.include_trajectory``
adds a bounded excerpt of the failing steps.
"""

from __future__ import annotations

from typing import Any, Sequence

from ..core.adapter import SkippedDataset
from ..core.metrics import aggregate_mean_metrics
from ..core.types import AdapterDocumentation, ModelResponse, SampleScore, SampleSpec
from . import _common as C
from ._base import PooledDatasetAdapter, selection_score

REPO_URL = "https://github.com/microsoft/AgentRx"


class AgentRxAdapter(PooledDatasetAdapter):
    """Select the failure category that explains an agent's failed run."""

    adapter_version = "1.0"
    primary_metric = "accuracy"

    def load_items(self) -> list[dict[str, Any]]:
        root = C.ensure_git_repo(
            REPO_URL, self.context.data_dir / "repo", offline=self.context.offline
        )
        files = C.find_files(root / "data" / "ground_truth", ["*.json"])
        if not files:
            raise SkippedDataset("no AgentRx ground-truth files found")
        items: list[dict[str, Any]] = []
        categories: set[str] = set()
        self._per_file: dict[str, int] = {}
        for path in files:
            payload = C.read_json(path)
            rows = payload if isinstance(payload, list) else C.as_list(payload)
            usable = 0
            for row in rows:
                if not isinstance(row, dict):
                    continue
                failures = [f for f in C.as_list(row.get("failures")) if isinstance(f, dict)]
                for failure in failures:
                    category = C.normalize_whitespace(failure.get("failure_category"))
                    if category:
                        categories.add(category)
                root_cause = row.get("root_cause") or {}
                if failures and root_cause.get("failure_id") is not None:
                    items.append({**row, "collection": path.stem})
                    usable += 1
            self._per_file[path.stem] = usable
        if not items:
            raise SkippedDataset("AgentRx ground truth carried no root-caused failures")
        self._categories = sorted(categories)
        self.split_used = (
            "all annotated trajectories with a marked root cause: "
            + ", ".join(f"{name} ({count})" for name, count in self._per_file.items())
            + " -- the release ships no train/test split"
        )
        return items

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        summary = C.normalize_whitespace(item.get("failure_summary"))
        failures = [f for f in C.as_list(item.get("failures")) if isinstance(f, dict)]
        root_cause = item.get("root_cause") or {}
        target_id = root_cause.get("failure_id")
        gold_category = ""
        for failure in failures:
            if failure.get("failure_id") == target_id:
                gold_category = C.normalize_whitespace(failure.get("failure_category"))
                break
        if not summary or not gold_category or gold_category not in self._categories:
            return None

        context_parts = []
        if bool(self.context.option("include_trajectory", False)):
            words = int(self.context.option("trajectory_words", 400))
            steps = [
                f"- step {failure.get('step_number')}: "
                + C.clip_words(C.normalize_whitespace(failure.get("step_reason")), 60)
                for failure in failures
            ]
            context_parts.append(C.clip_words("Observed problems during the run:\n" + "\n".join(steps), words))
        options = list(self._categories)
        labels = C.letter_labels(len(options))
        return SampleSpec(
            sample_id=C.stable_id("agentrx", item.get("collection"), item.get("trajectory_id", index)),
            fields={
                "observation": f"An AI agent's run failed. Observed failure: {summary}",
                "context": "\n\n".join(context_parts),
                "question": (
                    "Which category of failure is the root cause of this run's failure?"
                ),
                "options": options,
                "option_labels": labels,
                "instructions": (
                    "Several things may have gone wrong; choose the category of the failure that "
                    "caused the run to fail, not of a later symptom."
                ),
            },
            reference={"gold_label": labels[options.index(gold_category)], "gold": gold_category},
            task_kind="selection",
            max_tokens=768,
            metadata={
                "collection": item.get("collection"),
                "trajectory_id": item.get("trajectory_id"),
                "n_failures": len(failures),
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
        collection = sample.metadata.get("collection")
        if collection and "accuracy" in score.metrics:
            score.metrics[f"accuracy_{collection}"] = score.metrics["accuracy"]
        return score

    def aggregate(self, scores: Sequence[SampleScore]) -> dict[str, float]:
        return aggregate_mean_metrics([score.metrics for score in scores])

    def documentation(self) -> AdapterDocumentation:
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="AgentRx",
            domain="AI Systems: Agent Failure Diagnosis",
            source_url=REPO_URL,
            processing_mode="Selection",
            split_used=self.split_used,
            abductive_subset=(
                "Root-cause attribution for a failed agent run: which failure category explains "
                "the observed failure. Trajectories without a marked root cause are skipped."
            ),
            sampling_procedure=self.sampling_note(),
            metrics_description={
                "accuracy": "1 if the selected failure category is the root-cause failure's "
                "category (primary)",
                "accuracy_<collection>": "accuracy per ground-truth collection "
                "(magentic-one / tau)",
            },
            primary_metric="accuracy",
            decisions=[
                "Built the candidate set from the union of failure_category values in the ground "
                "truth: the repository publishes no taxonomy file, so its own labels are the only "
                "faithful option set.",
                "Default observation is the failure summary alone, because full trajectories are "
                "100-180 KB of JSON -- far over the 16k-token input budget. "
                "options.include_trajectory adds a bounded excerpt of the failing steps.",
                "Used the root-cause failure's category as gold rather than any of the "
                "co-occurring failures, which is what the dataset's root_cause field marks.",
            ],
            caveats=[
                "The failure summary is a human-written description of what went wrong, so the "
                "task is closer to 'classify this diagnosis' than to 'diagnose from raw evidence'; "
                "include_trajectory=true is the harder, more abductive setting.",
                "Few annotated trajectories exist, so this dataset reports well under 300 items.",
            ],
            statistics={
                **self.base_statistics(),
                "categories": getattr(self, "_categories", []),
                "items_per_collection": self._per_file,
            },
        )
