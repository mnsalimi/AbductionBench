"""XCOPA: multilingual choice of plausible alternatives (Ponti et al., EMNLP 2020).

Source: https://github.com/cambridgeltl/xcopa

**What is abductive here.** Each item has a ``question`` field that is either
``"cause"`` or ``"effect"``.  Only the **cause** questions are abductive (given
an observed premise, choose the alternative that most plausibly *caused* it);
``effect`` questions ask for a consequence, which is prediction rather than
abduction, and are excluded.

**Languages.** XCOPA is the human-translated COPA test set in 11 languages
(et, ht, id, it, qu, sw, ta, th, tr, vi, zh); there is no English directory
because English is COPA itself.  The evaluation set is drawn across *all*
languages from one seeded shuffle, so the sample is naturally spread over them,
and accuracy is reported per language as well as overall.  ``data/`` (human
translations) is used, not ``data-gmt/`` (machine translations).
"""

from __future__ import annotations

from typing import Any

from ..core.adapter import SkippedDataset
from ..core.types import AdapterDocumentation, ModelResponse, SampleScore, SampleSpec
from . import _common as C
from ._base import PooledDatasetAdapter, selection_score

REPO_URL = "https://github.com/cambridgeltl/xcopa"
#: Numbered, from the one house helper, so the label the model sees and
#: the label the gold refers to cannot drift apart.
LABELS = C.choice_labels(2)


class XCopaAdapter(PooledDatasetAdapter):
    """Cause-question half of XCOPA, across all available languages."""

    adapter_version = "1.0"

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. You are given a premise in one of "
        "several languages and two candidate causes. Choose the alternative that is the more "
        "plausible cause of the premise, judging by everyday causal knowledge in that "
        "language's context."
    )
    data_delivery_mode = "static"

    options_heading = "Answer options:"
    objective_metrics = True
    selection_cardinality = "single"
    primary_metric = "accuracy"

    def load_items(self) -> list[dict[str, Any]]:
        root = C.ensure_git_repo(
            REPO_URL, self.context.data_dir / "repo", offline=self.context.offline
        )
        data_dir = root / ("data-gmt" if self.context.option("machine_translated") else "data")
        if not data_dir.exists():
            raise SkippedDataset(f"XCOPA data directory not found: {data_dir}")

        wanted = self.context.option("languages")  # None -> every language present
        items: list[dict[str, Any]] = []
        self._effect_dropped = 0
        self._languages: list[str] = []
        for language_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
            language = language_dir.name
            if wanted and language not in wanted:
                continue
            found = C.pick_split_file(sorted(language_dir.glob("*.jsonl")))
            if not found:
                continue
            path, split = found
            self._languages.append(language)
            for row in C.read_jsonl(path):
                if str(row.get("question", "")).lower() != "cause":
                    self._effect_dropped += 1
                    continue
                items.append({**row, "language": language, "split": split})
        if not items:
            raise SkippedDataset("no cause-question items found in XCOPA")
        self.split_used = (
            f"test (official) for each of {len(self._languages)} languages: "
            f"{', '.join(self._languages)}"
        )
        return items

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        premise = C.normalize_whitespace(item.get("premise"))
        choices = [
            C.normalize_whitespace(item.get("choice1")),
            C.normalize_whitespace(item.get("choice2")),
        ]
        label = item.get("label")
        if not premise or not all(choices) or label not in (0, 1):
            return None
        language = item["language"]
        return SampleSpec(
            sample_id=C.stable_id("xcopa", language, item.get("idx", index)),
            fields={
                "observation": premise,
                "question": (
                    "Which alternative is the more plausible CAUSE of the observation? "
                    f"(The text is in language code '{language}'; answer with a label only.)"
                ),
                "options": choices,
                "option_labels": LABELS,
            },
            reference={"gold_label": LABELS[int(label)], "language": language},
            task_kind="selection",
            # One-sentence alternatives; the answer is a single letter.  Budget is
            # for a reasoning model's hidden chain-of-thought, not for output.
            max_tokens=320,
            metadata={"language": language, "idx": item.get("idx"), "split": item.get("split")},
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
        # Sparse per-language keys: the engine averages each key over only the
        # samples that carry it, which is exactly per-language accuracy.
        language = sample.reference["language"]
        score.metrics[f"accuracy_lang_{language}"] = score.metrics.get("accuracy", 0.0)
        return score

    def documentation(self) -> AdapterDocumentation:
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="XCOPA",
            domain="Commonsense: Multilingual Causality",
            source_url=REPO_URL,
            processing_mode="Selection",
            split_used=self.split_used,
            abductive_subset=(
                "Only items whose question is 'cause' (choose the plausible cause of the "
                f"premise); 'effect' items are predictive and were dropped "
                f"({getattr(self, '_effect_dropped', 0)} items)."
            ),
            sampling_procedure=self.sampling_note()
            + "; the pool pools all languages, so the draw spreads across them",
            metrics_description={
                "accuracy": "1 if the selected alternative is the gold cause",
                "accuracy_lang_<code>": "the same metric restricted to one language",
            },
            primary_metric="accuracy",
            decisions=[
                "Used data/ (human translations) rather than data-gmt/ (machine translations); "
                "options.machine_translated switches to the latter.",
                "Pooled all 11 languages into one 300-item draw instead of 300 per language, "
                "so the dataset contributes one comparable difficulty number while still "
                "reporting per-language accuracy.",
                "No English split exists in XCOPA (English is COPA itself); none was added, to "
                "avoid mixing a different source into the benchmark.",
            ],
            caveats=[
                "Per-language accuracy is computed over roughly 300/11 ≈ 27 items per language, "
                "so per-language numbers are indicative only.",
            ],
            statistics={
                **self.base_statistics(),
                "languages": ",".join(getattr(self, "_languages", [])),
                "effect_items_dropped": getattr(self, "_effect_dropped", 0),
            },
        )
