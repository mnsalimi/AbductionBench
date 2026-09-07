"""HypoBench / HypoGeniC datasets: data-driven hypothesis generation.

Source: https://huggingface.co/datasets/ChicagoHAI/HypoGeniC-datasets

Each task directory holds a classification dataset (train/val/test splits) plus
a ``metadata.json`` describing the task, its features and labels, and -- crucially
for scoring -- ``known_hypotheses``: hypotheses from the literature about what
distinguishes the classes.

**The abductive task.** HypoBench's purpose is generating hypotheses that
*explain labelled data*.  Single-turn, one sample therefore consists of a
deterministic sample of labelled examples from a task's training split, and the
model must state the hypothesis that separates the classes.  Responses are
scored against the task's ``known_hypotheses`` (best-match overlap), and a
judge verdict is available for semantic agreement.

Tasks without ``known_hypotheses`` are skipped -- there would be nothing to
score against -- and the count is reported.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

from ..core.adapter import SkippedDataset
from ..core.metrics import aggregate_mean_metrics, extract_answer_span, rouge_l, token_f1
from ..core.types import AdapterDocumentation, ModelResponse, SampleScore, SampleSpec
from . import _common as C
from ._base import PooledDatasetAdapter, unparsed_score

REPO_ID = "ChicagoHAI/HypoGeniC-datasets"


class HypoBenchAdapter(PooledDatasetAdapter):
    """State the hypothesis that explains a labelled sample of examples."""

    adapter_version = "1.0"

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. You are given a labelled sample "
        "of examples. State the hypothesis that explains the labelling: the rule that "
        "separates the classes and would generalise to unseen examples of the same kind."
    )
    data_delivery_mode = "static"

    answer_format = "one hypothesis"
    answer_constraints = (
        "state one hypothesis, not several",
        "name the feature and the direction of its effect",
        "do not restate the observation",
        "do not use introductory phrases or commentary",
    )
    objective_metrics = False
    selection_cardinality = None
    primary_metric = "hypothesis_judged"

    def load_items(self) -> list[dict[str, Any]]:
        root = C.ensure_hf_snapshot(
            REPO_ID,
            self.context.data_dir / "hf",
            offline=self.context.offline,
            allow_patterns=["real/**", "synthetic/**", "*.md"],
        )
        families = list(self.context.option("families", ["real", "synthetic"]))
        examples_per_sample = int(self.context.option("examples_per_sample", 12))
        samples_per_task = int(self.context.option("samples_per_task", 20))

        items: list[dict[str, Any]] = []
        self._tasks: dict[str, int] = {}
        self._skipped_tasks: list[str] = []
        for family in families:
            family_dir = root / family
            if not family_dir.exists():
                continue
            for task_dir in sorted(p for p in family_dir.iterdir() if p.is_dir()):
                metadata_path = task_dir / "metadata.json"
                if not metadata_path.exists():
                    self._skipped_tasks.append(f"{family}/{task_dir.name} (no metadata.json)")
                    continue
                metadata = C.read_json(metadata_path)
                known = [
                    C.normalize_whitespace(text)
                    for text in C.as_list(metadata.get("known_hypotheses"))
                    if C.normalize_whitespace(text)
                ]
                if not known:
                    self._skipped_tasks.append(f"{family}/{task_dir.name} (no known_hypotheses)")
                    continue
                rows = self._load_examples(task_dir, metadata)
                if len(rows) < examples_per_sample:
                    self._skipped_tasks.append(f"{family}/{task_dir.name} (too few examples)")
                    continue
                # Deterministic, disjoint example blocks per task.
                rng = random.Random(f"{self.context.seed}::hypobench::{family}/{task_dir.name}")
                order = list(range(len(rows)))
                rng.shuffle(order)
                for block in range(samples_per_task):
                    start = block * examples_per_sample
                    if start + examples_per_sample > len(order):
                        break
                    chosen = [rows[index] for index in order[start : start + examples_per_sample]]
                    items.append(
                        {
                            "task": f"{family}/{task_dir.name}",
                            "block": block,
                            "metadata": metadata,
                            "examples": chosen,
                            "known": known,
                        }
                    )
                self._tasks[f"{family}/{task_dir.name}"] = len(rows)
        if not items:
            raise SkippedDataset("no HypoBench task provided both examples and known hypotheses")
        self.split_used = (
            f"train splits of {len(self._tasks)} task(s), in blocks of {examples_per_sample} "
            f"labelled examples ({samples_per_task} blocks per task); scored against each task's "
            "known_hypotheses"
        )
        return items

    @staticmethod
    def _load_examples(task_dir, metadata: dict[str, Any]) -> list[dict[str, str]]:
        """Read a task's train file, which stores parallel column arrays."""
        candidates = [
            path
            for path in C.find_files(task_dir, ["*train*.json"])
            if "ood" not in path.name.lower()
        ]
        if not candidates:
            return []
        payload = C.read_json(candidates[0])
        if not isinstance(payload, dict):
            return []
        feature_names = list((metadata.get("features") or {}).keys())
        label_names = list((metadata.get("labels") or {}).keys())
        columns = {key: value for key, value in payload.items() if isinstance(value, list)}
        if not columns:
            return []
        length = min(len(value) for value in columns.values())
        rows: list[dict[str, str]] = []
        for index in range(length):
            row = {}
            for name in feature_names or list(columns):
                if name in columns:
                    row[name] = C.clip_words(C.normalize_whitespace(columns[name][index]), 80)
            for name in label_names or []:
                if name in columns:
                    row[f"__label__{name}"] = C.normalize_whitespace(columns[name][index])
            if row:
                rows.append(row)
        return rows

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        metadata = item["metadata"]
        description = C.normalize_whitespace(metadata.get("task_description"))
        labels_info = metadata.get("labels") or {}
        label_values = []
        for info in labels_info.values():
            label_values.extend(C.as_list(info.get("values")))
        blocks = []
        for position, example in enumerate(item["examples"], start=1):
            features = {k: v for k, v in example.items() if not k.startswith("__label__")}
            labels = {k.replace("__label__", ""): v for k, v in example.items() if k.startswith("__label__")}
            feature_text = " | ".join(f"{k}: {v}" for k, v in features.items())
            label_text = ", ".join(f"{k}={v}" for k, v in labels.items())
            blocks.append(f"{position}. [{label_text}] {feature_text}")
        if not blocks or not description:
            return None
        return SampleSpec(
            sample_id=C.stable_id("hypobench", item["task"], item["block"]),
            fields={
                "context": f"Task: {description}\nPossible labels: {', '.join(map(str, label_values))}",
                "observation": "Labelled examples:\n" + "\n".join(blocks),
                "question": (
                    "What hypothesis explains why the examples carry the labels they do?"
                ),
                "instructions": (
                    "State one general, testable hypothesis about what distinguishes the classes. "
                    "Name the feature or pattern and the direction of the effect."
                ),
            },
            reference={"gold": item["known"][0], "references": item["known"]},
            task_kind="generation",
            # Reading a dozen labelled examples then generalizing.
            max_tokens=768,
            metadata={"task": item["task"], "block": item["block"], "n_examples": len(blocks)},
        )

    def score(
        self,
        sample: SampleSpec,
        response: ModelResponse,
        *,
        output_contract: dict[str, Any] | None = None,
    ) -> SampleScore:
        answer = extract_answer_span(response.text, output_contract)
        if not answer:
            return unparsed_score(["best_rouge_l", "best_token_f1"], raw=response.text[:300])
        references = sample.reference["references"]
        rouges = [rouge_l(answer, reference)["f"] for reference in references]
        f1s = [token_f1(answer, reference) for reference in references]
        task = sample.metadata.get("task", "unknown").replace("/", "_")
        return SampleScore(
            metrics={
                "best_rouge_l": max(rouges),
                "best_token_f1": max(f1s),
                f"best_rouge_l_{task}": max(rouges),
            },
            prediction=answer[:500],
            details={"n_references": len(references)},
        )

    def aggregate(self, scores: Sequence[SampleScore]) -> dict[str, float]:
        return aggregate_mean_metrics([score.metrics for score in scores])

    def judge_request(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore
    ) -> dict[str, Any] | None:
        if not response.text:
            return None
        return {
            "candidate": score.prediction or response.text[:600],
            "gold": sample.reference["gold"],
            "observation": C.clip_words(sample.fields["observation"], 150),
            "criteria": (
                "Correct if the candidate identifies the same distinguishing pattern as the "
                "reference hypothesis, in the same direction."
            ),
        }

    def apply_judge(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore, verdict: Any
    ) -> SampleScore:
        metrics = dict(score.metrics)
        metrics["hypothesis_judged"] = 1.0 if getattr(verdict, "positive", False) else 0.0
        return SampleScore(
            metrics=metrics,
            prediction=score.prediction,
            parse_ok=score.parse_ok,
            details={**score.details, "judge_label": getattr(verdict, "label", None)},
        )

    def documentation(self) -> AdapterDocumentation:
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="HypoBench (HypoGeniC datasets)",
            domain="Scientific Discovery: Data-Driven Hypotheses",
            source_url=f"https://huggingface.co/datasets/{REPO_ID}",
            processing_mode="Generation",
            split_used=self.split_used,
            abductive_subset=(
                "Hypothesis generation over labelled examples. Tasks without known_hypotheses "
                "are skipped because there would be no reference to score against: "
                f"{len(getattr(self, '_skipped_tasks', []))} task(s) skipped."
            ),
            sampling_procedure=(
                "each task's train examples are shuffled with a task-specific seeded RNG and cut "
                "into disjoint blocks; one block is one sample, so every model sees identical "
                "example sets. Blocks are then drawn by "
                + self.sampling_note()
            ),
            metrics_description={
                "best_rouge_l": "ROUGE-L F against the closest known hypothesis",
                "best_token_f1": "token F1 against the closest known hypothesis",
                "best_rouge_l_<task>": "the primary metric restricted to one task",
                "hypothesis_judged": "(primary) LLM-judge verdict on same-pattern-same-direction (only when "
                "engine.judge.enabled)",
            },
            primary_metric="hypothesis_judged",
            decisions=[
                "Used train examples as the evidence shown to the model (test/val are held out "
                "for the classification task the dataset was built for, and a hypothesis is "
                "generated *from* data, not evaluated on a held-out row).",
                "Composed samples as blocks of 12 labelled examples (configurable), 20 blocks per "
                "task, which is how a hypothesis-generation prompt is normally set up and keeps "
                "the item count meaningful.",
                "Long feature text is clipped to 80 words per example so a block fits the input "
                "budget.",
                "Scored against the literature hypotheses shipped in metadata.json, taking the "
                "best match among them.",
            ],
            caveats=[
                "These overlap numbers are diagnostics, not the evaluation: "
                "character/n-gram similarity punishes a correct paraphrase and "
                "rewards a wrong sentence that reuses the reference's words, so "
                "this dataset is scored by an LLM judge instead.",
                "known_hypotheses are literature claims, not an exhaustive answer key: a model may "
                "state a valid pattern that no reference mentions and be under-credited. The judge "
                "stage mitigates but does not remove this.",
                "Samples from the same task share reference hypotheses, so items are not fully "
                "independent.",
            ],
            statistics={
                **self.base_statistics(),
                "tasks_used": len(getattr(self, "_tasks", {})),
                "tasks_skipped": getattr(self, "_skipped_tasks", []),
                "examples_per_sample": int(self.context.option("examples_per_sample", 12)),
            },
        )
