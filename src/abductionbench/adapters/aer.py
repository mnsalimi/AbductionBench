"""AER: abductive event reasoning (SemEval-2026 Task 12).

Source: https://github.com/sooo66/semeval2026-task12-dataset

Each instance gives an observed real-world event, four candidate explanations,
and retrieved news documents for the event's topic.  The gold answer is the set
of options that are plausible *direct causes* -- often more than one -- so this
is multi-answer hypothesis selection.

**Split.** The competition's ``test_data`` ships **without** ``golden_answer``
(it is the blind evaluation set), so the labelled **dev** split is used; this is
reported rather than silently scoring against nothing.

**Context size.** The retrieved corpus for one topic averages ~240,000
characters (~60k tokens), far over the suite's 16,000-token input budget.  By
default only each document's title and snippet are included (``doc_content:
snippets``), which keeps the retrieved evidence present at ~2k tokens; the full
article text is available via ``options.doc_content = full`` for runs with a
larger budget, and ``none`` measures the task without retrieval.
"""

from __future__ import annotations

import re
from typing import Any

from ..core.adapter import SkippedDataset
from ..core.metrics import extract_answer_span, set_prf
from ..core.types import AdapterDocumentation, ModelResponse, SampleScore, SampleSpec
from . import _common as C
from ._base import PooledDatasetAdapter, unparsed_score

REPO_URL = "https://github.com/sooo66/semeval2026-task12-dataset"
OPTION_KEYS = ("option_A", "option_B", "option_C", "option_D")


class AERAdapter(PooledDatasetAdapter):
    """Multi-answer selection of the direct causes of a news event."""

    adapter_version = "1.0"

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. You are given a news event and a "
        "list of candidate antecedent events. Select every candidate that is a direct cause "
        "of the event -- an event whose occurrence made the target event happen, not one that "
        "merely preceded or accompanied it. Several candidates can be direct causes."
    )
    data_delivery_mode = "static"

    options_heading = "Candidate explanations:"
    objective_metrics = True
    selection_cardinality = "multi"
    primary_metric = "set_f1"

    def load_items(self) -> list[dict[str, Any]]:
        root = C.ensure_git_repo(
            REPO_URL, self.context.data_dir / "repo", offline=self.context.offline
        )
        # test_data has no golden_answer; dev_data does.
        for split in ("dev", "train", "sample"):
            questions = root / f"{split}_data" / "questions.jsonl"
            docs = root / f"{split}_data" / "docs.json"
            if not questions.exists():
                continue
            rows = [row for row in C.read_jsonl(questions) if row.get("golden_answer")]
            if not rows:
                continue
            self._docs = self._index_docs(docs)
            self.split_used = (
                f"{split} ({len(rows)} labelled instances); the official test split ships "
                "without golden_answer and cannot be scored locally"
            )
            return rows
        raise SkippedDataset("no AER split with golden_answer labels was found")

    @staticmethod
    def _index_docs(path) -> dict[int, dict[str, Any]]:
        if not path.exists():
            return {}
        payload = C.read_json(path)
        index: dict[int, dict[str, Any]] = {}
        for entry in payload if isinstance(payload, list) else []:
            topic_id = entry.get("topic_id")
            if topic_id is not None:
                index[int(topic_id)] = entry
        return index

    def _context_for(self, topic_id: Any) -> str:
        mode = str(self.context.option("doc_content", "snippets"))
        if mode == "none" or topic_id is None:
            return ""
        entry = getattr(self, "_docs", {}).get(int(topic_id))
        if not entry:
            return ""
        max_docs = int(self.context.option("max_docs", 12))
        words_per_doc = int(self.context.option("words_per_doc", 120))
        blocks = [f"Topic: {C.normalize_whitespace(entry.get('topic'))}"]
        for document in (entry.get("docs") or [])[:max_docs]:
            title = C.normalize_whitespace(document.get("title"))
            body = C.normalize_whitespace(
                document.get("content") if mode == "full" else document.get("snippet")
            )
            if mode == "full":
                body = C.clip_words(body, words_per_doc)
            blocks.append(f"- {title}: {body}".strip())
        return "\n".join(blocks)

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        event = C.normalize_whitespace(item.get("target_event"))
        # Kept parallel to OPTION_KEYS, because the gold answer names options by
        # the *letter in their key* ("A,C") while the prompt labels them by
        # position ("1", "2", ...).  Dropping an empty option shifts every
        # position after it, so the translation has to go through the key the
        # release used, not through the index in the filtered list.
        present = [
            (key[-1], C.normalize_whitespace(item.get(key)))
            for key in OPTION_KEYS
            if C.normalize_whitespace(item.get(key))
        ]
        options = [text for _letter, text in present]
        labels = C.choice_labels(len(options))
        by_letter = {letter: labels[index] for index, (letter, _t) in enumerate(present)}
        gold_letters = [
            part.strip().upper()
            for part in str(item.get("golden_answer", "")).split(",")
            if part.strip()
        ]
        if not event or len(options) < 2 or not gold_letters:
            return None
        # An unknown letter means the row's gold points at an option the row
        # does not carry; that item cannot be scored, so it is dropped rather
        # than silently scored against a wrong option.
        if not set(gold_letters) <= set(by_letter):
            return None
        gold = [by_letter[letter] for letter in gold_letters]
        return SampleSpec(
            sample_id=C.stable_id("aer", item.get("id", index)),
            fields={
                "observation": event,
                "context": self._context_for(item.get("topic_id")),
                "question": (
                    "Which of the following are plausible DIRECT causes of the observed event?"
                ),
                "options": options,
                "option_labels": labels,
                "instructions": (
                    "More than one option may be a direct cause; list every one that is."
                ),
            },
            reference={"gold_labels": gold, "options": options},
            task_kind="multi_selection",
            # A label list, but the model must weigh four candidates against the
            # retrieved evidence first.
            max_tokens=640,
            metadata={"topic_id": item.get("topic_id"), "n_gold": len(gold)},
        )

    def score(
        self,
        sample: SampleSpec,
        response: ModelResponse,
        *,
        output_contract: dict[str, Any] | None = None,
    ) -> SampleScore:
        labels = sample.fields["option_labels"]
        answer = extract_answer_span(response.text, output_contract)
        # Labels are numbers now, so the token pattern is alphanumeric: a
        # letters-only pattern found nothing and every response parsed as a
        # failure.
        found = {
            token.upper()
            for token in re.findall(r"[A-Za-z0-9]+", answer or "")
            if token.upper() in labels
        }
        if not found:
            return unparsed_score(
                ["set_f1", "exact_set_match", "set_precision", "set_recall"],
                raw=response.text[:300],
            )
        gold = set(sample.reference["gold_labels"])
        prf = set_prf(found, gold)
        return SampleScore(
            metrics={
                "set_f1": prf["f1"],
                "set_precision": prf["precision"],
                "set_recall": prf["recall"],
                "exact_set_match": float(found == gold),
            },
            prediction=",".join(sorted(found)),
            details={"gold": ",".join(sorted(gold))},
        )

    def documentation(self) -> AdapterDocumentation:
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="AER (SemEval-2026 Task 12)",
            domain="Event Reasoning: News Causality",
            source_url=REPO_URL,
            processing_mode="Selection (multi-answer)",
            split_used=self.split_used,
            abductive_subset=(
                "The whole task is abductive: identify the most plausible direct cause(s) of an "
                "observed event from candidate explanations."
            ),
            sampling_procedure=self.sampling_note(),
            metrics_description={
                "set_f1": "F1 between the predicted and gold label sets (primary; the gold answer "
                "is a set, so single-label accuracy would misreport it)",
                "exact_set_match": "1 only if the predicted set equals the gold set exactly",
                "set_precision": "precision of the predicted labels",
                "set_recall": "recall of the gold labels",
            },
            primary_metric="set_f1",
            decisions=[
                "Used the dev split because the official test split ships without golden_answer.",
                "Scored as a set task (F1 plus exact set match) since 38% of dev items have more "
                "than one gold cause.",
                "Included retrieved documents as title + snippet by default: the full corpus per "
                "topic averages ~60k tokens, over the suite's 16k input budget. "
                "options.doc_content = full | none changes this, and max_docs/words_per_doc cap "
                "the full variant.",
                "max_tokens=640 -- the output is a short label list, the budget is for weighing "
                "four candidates against the evidence.",
            ],
            caveats=[
                "With snippets only, a model may lack the detail needed to separate a direct "
                "cause from a correlated event; compare against doc_content=full before drawing "
                "conclusions about retrieval-grounded ability.",
            ],
            statistics={
                **self.base_statistics(),
                "doc_content": str(self.context.option("doc_content", "snippets")),
                "topics_indexed": len(getattr(self, "_docs", {})),
            },
        )
