"""e-CARE: explainable causal reasoning (Du et al., ACL 2022).

Source: https://github.com/Waste-Wood/e-CARE/

**What is abductive here.** e-CARE ships two subtasks:

* ``Causal_Reasoning`` -- a premise plus two hypotheses, with an ``ask-for``
  field that is either ``"cause"`` or ``"effect"``.  Only the **ask-for-cause**
  half is abduction (given an observed effect, pick the hypothesis that best
  explains it); ask-for-effect items are *predictive* (deduction-like) and are
  therefore excluded.
* ``Explanation_Generation`` -- given a cause/effect pair, produce the
  conceptual explanation of why the relation holds.  That is Stage-1
  explanation generation, and is available via ``options.subtask``.

**Split.** The public release has train and dev only (the test labels are held
back for the leaderboard), so the **dev** split is used and this is reported.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..core.adapter import SkippedDataset
from ..core.metrics import aggregate_mean_metrics
from ..core.types import AdapterDocumentation, ModelResponse, SampleScore, SampleSpec
from . import _common as C
from ._base import PooledDatasetAdapter, selection_score, text_match_score

REPO_URL = "https://github.com/Waste-Wood/e-CARE"
LABELS = ["A", "B"]


class ECareAdapter(PooledDatasetAdapter):
    """Abductive half of e-CARE: pick the cause that explains the premise."""

    adapter_version = "1.0"
    primary_metric = "accuracy"

    def load_items(self) -> list[dict[str, Any]]:
        subtask = str(self.context.option("subtask", "cause_selection"))
        root = C.ensure_git_repo(
            REPO_URL, self.context.data_dir / "repo", offline=self.context.offline
        )
        if subtask == "explanation":
            path = root / "dataset" / "Explanation_Generation" / "dev.jsonl"
            self.split_used = "dev (official; test labels are withheld) -- Explanation_Generation"
            rows = C.read_jsonl(self._require(path))
            return [{"kind": "explanation", **row} for row in rows]

        path = root / "dataset" / "Causal_Reasoning" / "dev.jsonl"
        self.split_used = "dev (official; test labels are withheld) -- Causal_Reasoning"
        rows = C.read_jsonl(self._require(path))
        # The abductive selection: only questions that ask for the CAUSE.
        cause_rows = [row for row in rows if str(row.get("ask-for", "")).lower() == "cause"]
        self._ask_for_effect_dropped = len(rows) - len(cause_rows)
        if subtask == "both":
            explanation_rows = C.read_jsonl(
                self._require(root / "dataset" / "Explanation_Generation" / "dev.jsonl")
            )
            keep = {row["index"] for row in cause_rows}
            self.split_used = "dev (official) -- Causal_Reasoning (ask-for=cause) + Explanation_Generation"
            return [{"kind": "selection", **row} for row in cause_rows] + [
                {"kind": "explanation", **row}
                for row in explanation_rows
                if row.get("index") in keep
            ]
        return [{"kind": "selection", **row} for row in cause_rows]

    def _require(self, path: Path) -> Path:
        if not path.exists():
            raise SkippedDataset(f"expected e-CARE file not found: {path}")
        return path

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        if item.get("kind") == "explanation":
            cause = C.normalize_whitespace(item.get("cause"))
            effect = C.normalize_whitespace(item.get("effect"))
            explanation = C.normalize_whitespace(item.get("conceptual_explanation"))
            if not (cause and effect and explanation):
                return None
            return SampleSpec(
                sample_id=C.stable_id("ecare-exp", item.get("index", index)),
                fields={
                    "observation": f"Cause: {cause}\nEffect: {effect}",
                    "question": (
                        "What general conceptual relation explains why this cause "
                        "produces this effect?"
                    ),
                    "instructions": "State the underlying conceptual explanation in one sentence.",
                },
                reference={"gold": explanation},
                task_kind="generation",
                # One-sentence conceptual explanations: a small budget suffices,
                # with headroom for a reasoning model's hidden chain-of-thought.
                max_tokens=384,
                metadata={"subtask": "explanation_generation", "index": item.get("index")},
            )

        premise = C.normalize_whitespace(item.get("premise"))
        hypotheses = [
            C.normalize_whitespace(item.get("hypothesis1")),
            C.normalize_whitespace(item.get("hypothesis2")),
        ]
        label = item.get("label")
        if not premise or not all(hypotheses) or label not in (0, 1, "0", "1"):
            return None
        gold = LABELS[int(label)]
        return SampleSpec(
            sample_id=C.stable_id("ecare-sel", item.get("index", index)),
            fields={
                "observation": premise,
                "question": "Which hypothesis is the more plausible CAUSE of the observation above?",
                "options": hypotheses,
                "option_labels": LABELS,
            },
            reference={"gold_label": gold, "options": hypotheses},
            task_kind="selection",
            # Two short candidates; the answer is a single letter.
            max_tokens=320,
            metadata={"subtask": "cause_selection", "index": item.get("index")},
        )

    def score(
        self,
        sample: SampleSpec,
        response: ModelResponse,
        *,
        output_contract: dict[str, Any] | None = None,
    ) -> SampleScore:
        if sample.task_kind == "selection":
            return selection_score(
                response,
                labels=sample.fields["option_labels"],
                gold_label=sample.reference["gold_label"],
                output_contract=output_contract,
                metric_name="accuracy",
            )
        return text_match_score(
            response,
            gold=sample.reference["gold"],
            output_contract=output_contract,
            primary="explanation_match",
        )

    def aggregate(self, scores: Sequence[SampleScore]) -> dict[str, float]:
        return aggregate_mean_metrics([score.metrics for score in scores])

    def documentation(self) -> AdapterDocumentation:
        subtask = str(self.context.option("subtask", "cause_selection"))
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="e-CARE",
            domain="Commonsense: Causal Reasoning",
            source_url=REPO_URL,
            processing_mode="Selection (default) / Generation",
            split_used=self.split_used,
            abductive_subset=(
                "Causal_Reasoning items with ask-for == 'cause' only -- given the observed "
                "premise, choose the hypothesis that best explains it. ask-for == 'effect' "
                "items are predictive rather than abductive and are excluded "
                f"({getattr(self, '_ask_for_effect_dropped', 0)} dropped from dev). "
                "Explanation_Generation (conceptual explanation of a causal pair) is "
                "Stage-1 explanation generation and is selectable with "
                "options.subtask = explanation | both."
            ),
            sampling_procedure=self.sampling_note(),
            metrics_description={
                "accuracy": "selection: 1 if the chosen hypothesis label matches the gold cause",
                "explanation_match": "generation: answer equals or contains the gold conceptual explanation",
                "exact_match": "generation: normalized string equality with the gold explanation",
                "token_f1": "generation: bag-of-tokens F1 against the gold explanation",
                "rouge_l": "generation: LCS-based F-measure against the gold explanation",
            },
            primary_metric="accuracy" if subtask != "explanation" else "explanation_match",
            decisions=[
                "Used the dev split: e-CARE's test labels are not public.",
                "Restricted the selection subtask to ask-for == 'cause', the abductive half.",
                "options.subtask selects cause_selection (default), explanation, or both.",
                "Selection items get max_tokens=320 and explanation items 384 -- both answers "
                "are one short sentence, but gpt-oss-style reasoning models need budget for "
                "hidden chain-of-thought before the answer line.",
            ],
            caveats=[
                "Conceptual explanations are free text with many valid paraphrases, so "
                "explanation_match/token_f1 under-credit correct rewordings; enable the "
                "LLM-judge stage for a semantic view.",
            ],
            statistics={
                **self.base_statistics(),
                "ask_for_effect_dropped": getattr(self, "_ask_for_effect_dropped", 0),
                "subtask": subtask,
            },
        )
