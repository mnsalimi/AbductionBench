"""True Detective: long-form mystery solving (Del & Fishel, 2023).

Source: https://github.com/TartuNLP/true-detective

191 detective puzzles from 5minutemystery.com.  Each gives a ~1,200-word
mystery text, four named suspects/explanations, the correct one, and the human
solve rate -- an unusually direct difficulty signal, which this adapter also
uses: it reports the correlation between model correctness and human solve rate,
so one can see whether a model finds the same puzzles hard that people do.

The dataset is shipped as ``data/data.zip``, which the adapter extracts on first
use.  There is no official split and only 191 items exist, so the whole set is
used and the shortfall against a 300-sample target is reported.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from ..core.adapter import SkippedDataset
from ..core.metrics import aggregate_mean_metrics, mean, spearman
from ..core.types import AdapterDocumentation, ModelResponse, SampleScore, SampleSpec
from . import _common as C
from ._base import PooledDatasetAdapter, selection_score

REPO_URL = "https://github.com/TartuNLP/true-detective"
#: "(a) Chris Henderson; (b) Dave Perkins; ..."
_OPTION_RE = re.compile(r"\(([a-z])\)\s*([^;]+)")


class TrueDetectiveAdapter(PooledDatasetAdapter):
    """Select the correct culprit/explanation for a long mystery."""

    adapter_version = "1.0"

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. You are given a detective puzzle "
        "in full. Work out which candidate explanation the evidence actually supports; these "
        "puzzles are designed so that the obvious reading is usually wrong."
    )
    data_delivery_mode = "static"
    objective_metrics = True
    selection_cardinality = "single"
    primary_metric = "accuracy"

    def load_items(self) -> list[dict[str, Any]]:
        root = C.ensure_git_repo(
            REPO_URL, self.context.data_dir / "repo", offline=self.context.offline
        )
        archive = root / "data" / "data.zip"
        if not archive.exists():
            raise SkippedDataset(f"True Detective archive not found: {archive}")
        extracted = C.extract_archive(archive, self.context.data_dir / "extracted")
        csv_files = C.find_files(extracted, ["*.csv"])
        if not csv_files:
            raise SkippedDataset("no CSV found inside True Detective's data.zip")
        rows = C.read_csv_rows(csv_files[0])
        self.split_used = (
            f"whole dataset ({len(rows)} puzzles; no official split, and fewer than the "
            "300-sample target)"
        )
        return rows

    @staticmethod
    def _parse_options(raw: str) -> list[tuple[str, str]]:
        return [
            (letter.upper(), C.normalize_whitespace(text))
            for letter, text in _OPTION_RE.findall(raw or "")
        ]

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        mystery = C.normalize_whitespace(item.get("mystery_text"))
        options = self._parse_options(item.get("answer_options", ""))
        answer = C.normalize_whitespace(item.get("answer"))
        if not mystery or len(options) < 2 or not answer:
            return None
        gold_letter = None
        match = _OPTION_RE.match(answer)
        if match:
            gold_letter = match.group(1).upper()
        else:  # fall back to matching the answer text against the options
            for letter, text in options:
                if C.normalize_whitespace(text).lower() == answer.lower():
                    gold_letter = letter
                    break
        if gold_letter is None or gold_letter not in {letter for letter, _ in options}:
            return None

        try:
            solve_rate = float(item.get("solve_rate") or "nan") / 100.0
        except ValueError:
            solve_rate = float("nan")

        return SampleSpec(
            sample_id=C.stable_id("truedet", item.get("case_name") or index),
            fields={
                "observation": mystery,
                "question": (
                    "Who is responsible, and which explanation fits every detail of the case?"
                ),
                "options": [text for _, text in options],
                "option_labels": [letter for letter, _ in options],
            },
            reference={
                "gold_label": gold_letter,
                "human_solve_rate": solve_rate,
                "case_name": item.get("case_name"),
            },
            task_kind="selection",
            # Long multi-clue mysteries genuinely need deliberation before the
            # single-label answer, so the budget is much larger than the answer.
            max_tokens=1536,
            metadata={
                "case_name": item.get("case_name"),
                "human_solve_rate": solve_rate,
                "n_options": len(options),
                "mystery_words": len(mystery.split()),
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
        solve_rate = sample.reference.get("human_solve_rate")
        if solve_rate == solve_rate:  # not NaN
            score.details["human_solve_rate"] = solve_rate
            score.metrics["human_solve_rate"] = float(solve_rate)
        return score

    def aggregate(self, scores: Sequence[SampleScore]) -> dict[str, float]:
        metrics = aggregate_mean_metrics([score.metrics for score in scores])
        paired = [
            (score.metrics.get("accuracy", 0.0), score.metrics["human_solve_rate"])
            for score in scores
            if "human_solve_rate" in score.metrics
        ]
        if len(paired) >= 3:
            metrics["human_agreement_spearman"] = spearman(
                [correct for correct, _ in paired], [rate for _, rate in paired]
            )
            hard = [correct for correct, rate in paired if rate < 0.3]
            easy = [correct for correct, rate in paired if rate >= 0.3]
            if hard:
                metrics["accuracy_human_hard"] = mean(hard)
            if easy:
                metrics["accuracy_human_easy"] = mean(easy)
        return metrics

    def documentation(self) -> AdapterDocumentation:
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="True Detective",
            domain="Narrative Reasoning: Mystery Solving",
            source_url=REPO_URL,
            processing_mode="Selection",
            split_used=self.split_used,
            abductive_subset=(
                "The entire dataset is abductive: the reader must infer the single explanation "
                "consistent with all the clues. No filtering was needed."
            ),
            sampling_procedure=self.sampling_note()
            + "; with only 191 puzzles available the draw covers all of them",
            metrics_description={
                "accuracy": "1 if the selected suspect/explanation is the gold one",
                "human_solve_rate": "mean human solve rate of the sampled puzzles (context, not "
                "model performance)",
                "human_agreement_spearman": "rank correlation between model correctness and human "
                "solve rate -- positive means the model finds the same puzzles hard that humans do",
                "accuracy_human_easy": "accuracy on puzzles humans solve at >= 30%",
                "accuracy_human_hard": "accuracy on puzzles humans solve at < 30%",
            },
            primary_metric="accuracy",
            decisions=[
                "Parsed the '(a) X; (b) Y' option string into labelled options and derived the "
                "gold label from the answer field's own '(x)' prefix, falling back to matching "
                "the answer text.",
                "Kept the human solve rate as a per-sample metric so the aggregate can report "
                "human-difficulty correlation and an easy/hard split at the 30% solve rate "
                "(the dataset provides no difficulty labels, so this threshold is our choice).",
                "max_tokens=1536: mysteries average ~1,200 words and require multi-clue "
                "deliberation before committing.",
            ],
            caveats=[
                "Only 191 items exist, so this dataset reports fewer than the 300-sample target.",
                "These puzzles are public web content and may appear in pretraining corpora; "
                "high scores should not be read as pure reasoning ability.",
            ],
            statistics=self.base_statistics(),
        )
