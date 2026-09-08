"""ResearchBench -- literature-grounded hypothesis discovery.

Source: https://huggingface.co/datasets/ankilok/ResearchBench (gated).

The release ships four files, of which two are abductive and are run as the two
independent tasks the dataset table calls for:

``generation/generation.jsonl``
    A paper's background and research question, with the hypothesis it went on
    to establish as the gold answer -> **generation**.
``ranking/ranking.jsonl``
    Candidate hypotheses for one research question, one of which is the
    published one -> **selection**.

``retrieve/`` (inspiration retrieval over a corpus) and ``papers/`` (the
bibliographic records the other files refer to) are not abductive on their own:
retrieval asks which prior work is relevant, not which hypothesis explains an
observation.  ``papers.jsonl`` is still read, because it is where the background
text lives when the task files reference a paper by id.

**Access.**  The repository is gated with ``gated: auto``: any Hugging Face
account can read it, but only after accepting the dataset's terms once on its
page.  A token alone is not enough -- an un-accepted account gets HTTP 403 on
every file while still being able to read the repo's metadata, which is why this
adapter checks the files rather than the repo when it reports the problem.  Set
``HF_TOKEN`` to a token whose account has accepted the terms.

Field names are resolved from a list of candidates rather than hard-coded, and
an unrecognised schema is reported with the keys that were actually found.  The
gate means the file layout below could not be verified against the real files
before shipping this adapter, and a wrong guess must fail loudly with something
actionable rather than quietly evaluate the wrong column.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from ..core.adapter import SkippedDataset
from ..core.metrics import aggregate_mean_metrics
from ..core.types import AdapterDocumentation, ModelResponse, SampleScore, SampleSpec
from . import _common as C
from ._base import PooledDatasetAdapter, apply_judged_metric, judged_only_score, selection_score, text_match_score

REPO_ID = "ankilok/ResearchBench"
ACCEPT_URL = f"https://huggingface.co/datasets/{REPO_ID}"

#: Candidate key names, most specific first.  ResearchBench's own vocabulary is
#: "background / research question / hypothesis"; the alternatives cover the
#: usual renamings between a paper's release and its HF card.
BACKGROUND_KEYS = ("background", "background_survey", "context", "paper_background", "abstract")
QUESTION_KEYS = ("research_question", "question", "problem", "query", "task")
HYPOTHESIS_KEYS = ("hypothesis", "gold_hypothesis", "ground_truth_hypothesis", "answer",
                   "reference_hypothesis")
CANDIDATE_KEYS = ("candidates", "hypotheses", "options", "candidate_hypotheses", "ranking")
GOLD_INDEX_KEYS = ("gold_index", "label", "answer_index", "correct_index")
PAPER_ID_KEYS = ("paper_id", "paper", "doi", "id", "arxiv_id")


def _first_present(row: dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


class ResearchBenchAdapter(PooledDatasetAdapter):
    """Hypothesis generation (default) or selection among released candidates."""

    adapter_version = "1.0"

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. You are given the background of a "
        "research paper and the question it set out to answer. The hypothesis you want is the "
        "one the paper went on to establish: a specific, mechanistic claim that would explain "
        "the findings and that an experiment could test, not a restatement of the question or "
        "a call for further study."
    )
    data_delivery_mode = "static"

    answer_format = "one testable hypothesis"
    answer_constraints = (
        "state one hypothesis, not several",
        "make it specific enough to be tested",
        "do not describe the method or the expected result",
        "do not use introductory phrases or commentary",
    )
    options_heading = "Candidate hypotheses:"
    # Generation is scored by overlap with one reference hypothesis, which is a
    # proxy rather than a decision procedure, so the reasoning prompt modes are
    # not offered; the selection task is exact and could support them, but the
    # flag is per adapter and the stricter reading is the safe one.
    objective_metrics = False
    selection_cardinality = "single"
    hypothesis_modes = ("generation", "selection")
    hypothesis_mode_options = {
        "generation": {"subtask": "generation"},
        "selection": {"subtask": "selection"},
    }
    table_hypothesis_mode = "Generation / Selection (separate tasks)"

    primary_metric = "hypothesis_judged"

    primary_metric_by_mode = {
        # The two tasks are scored by different things, so each names
        # the metric it actually produces rather than inheriting one.
        "generation": "hypothesis_judged",
        "selection": "accuracy",
    }

    # ------------------------------------------------------------------ #
    # data
    # ------------------------------------------------------------------ #

    @property
    def subtask(self) -> str:
        return str(self.context.option("subtask", "generation")).lower()

    def _root(self):
        if not os.environ.get("HF_TOKEN"):
            raise SkippedDataset(
                f"ResearchBench is a gated Hugging Face dataset and HF_TOKEN is not set. "
                f"Set HF_TOKEN to a token whose account has accepted the terms at {ACCEPT_URL}."
            )
        try:
            return C.ensure_hf_snapshot(
                REPO_ID,
                self.context.data_dir / "hf",
                offline=self.context.offline,
                allow_patterns=["generation/*", "ranking/*", "papers/*", "README.md"],
            )
        except SkippedDataset as exc:
            if "403" in str(exc) or "gated" in str(exc).lower():
                raise SkippedDataset(
                    "ResearchBench downloads return HTTP 403: the HF account behind HF_TOKEN "
                    f"has not accepted the dataset's terms. Accept them once at {ACCEPT_URL} "
                    "(the gate is 'auto', so access is granted immediately) and re-run. The "
                    "adapter needs no other change."
                ) from exc
            raise

    def _papers(self, root) -> dict[str, dict[str, Any]]:
        """Bibliographic records, keyed by whichever id field they carry."""
        path = root / "papers" / "papers.jsonl"
        if not path.exists():
            return {}
        index: dict[str, dict[str, Any]] = {}
        for row in C.read_jsonl(path):
            key = _first_present(row, PAPER_ID_KEYS)
            if key is not None:
                index[str(key)] = row
        return index

    def load_items(self) -> list[dict[str, Any]]:
        root = self._root()
        if self.subtask == "selection":
            path = root / "ranking" / "ranking.jsonl"
            self.split_used = (
                "ranking/ranking.jsonl -- the released hypothesis-ranking task, used as the "
                "selection half of ResearchBench"
            )
        else:
            path = root / "generation" / "generation.jsonl"
            self.split_used = (
                "generation/generation.jsonl -- the released hypothesis-generation task. "
                "retrieve/ is excluded: inspiration retrieval asks which prior work is "
                "relevant, which is not abduction"
            )
        if not path.exists():
            raise SkippedDataset(f"ResearchBench file not found after download: {path}")
        rows = list(C.read_jsonl(path))
        if not rows:
            raise SkippedDataset(f"{path.name} is empty")

        papers = self._papers(root)
        for row in rows:
            paper_id = _first_present(row, PAPER_ID_KEYS)
            if paper_id is not None and str(paper_id) in papers:
                # Merge the paper record underneath the task row, so background
                # text stored once in papers.jsonl is available to both tasks.
                merged = {**papers[str(paper_id)], **row}
                row.clear()
                row.update(merged)

        probe = rows[0]
        if _first_present(probe, HYPOTHESIS_KEYS) is None:
            raise SkippedDataset(
                f"cannot find the gold hypothesis in {path.name}: none of {list(HYPOTHESIS_KEYS)} "
                f"is present. Keys in the file: {sorted(probe)[:20]}. Add the right name to "
                "HYPOTHESIS_KEYS in adapters/researchbench.py rather than guessing a column."
            )
        if self.subtask == "selection" and _first_present(probe, CANDIDATE_KEYS) is None:
            raise SkippedDataset(
                f"cannot find candidate hypotheses in {path.name}: none of {list(CANDIDATE_KEYS)} "
                f"is present. Keys in the file: {sorted(probe)[:20]}."
            )
        return rows

    # ------------------------------------------------------------------ #
    # samples
    # ------------------------------------------------------------------ #

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        background = C.clip_words(str(_first_present(item, BACKGROUND_KEYS) or ""), 900)
        question = C.normalize_whitespace(str(_first_present(item, QUESTION_KEYS) or ""))
        gold = C.normalize_whitespace(str(_first_present(item, HYPOTHESIS_KEYS) or ""))
        if not gold or not (background or question):
            return None
        sample_id = C.stable_id("rbench", _first_present(item, PAPER_ID_KEYS) or index, self.subtask)

        if self.subtask == "selection":
            raw = _first_present(item, CANDIDATE_KEYS) or []
            candidates = [C.normalize_whitespace(str(c)) for c in raw if str(c).strip()]
            if len(candidates) < 2:
                return None
            gold_index = _first_present(item, GOLD_INDEX_KEYS)
            if gold_index is None:
                # No index: the gold hypothesis text has to be one of the
                # candidates, or the item cannot be graded.
                matches = [i for i, c in enumerate(candidates) if c == gold]
                if not matches:
                    return None
                gold_index = matches[0]
            gold_index = int(gold_index)
            if not 0 <= gold_index < len(candidates):
                return None
            labels = C.choice_labels(len(candidates))
            return SampleSpec(
                sample_id=sample_id,
                fields={
                    "context": background,
                    "observation": question or background,
                    "question": (
                        "Which of these hypotheses is the one this paper established?"
                    ),
                    "options": candidates,
                    "option_labels": labels,
                },
                reference={"gold_label": labels[gold_index], "gold": candidates[gold_index],
                           "options": candidates},
                task_kind="selection",
                metadata={"subtask": "selection", "n_candidates": len(candidates)},
            )

        return SampleSpec(
            sample_id=sample_id,
            fields={
                "context": background,
                "observation": question or background,
                "answer_format": "the hypothesis, in one or two sentences",
            },
            reference={"gold": gold},
            task_kind="generation",
            metadata={"subtask": "generation"},
        )

    # ------------------------------------------------------------------ #
    # scoring
    # ------------------------------------------------------------------ #

    def score(
        self,
        sample: SampleSpec,
        response: ModelResponse,
        *,
        output_contract: dict[str, Any] | None = None,
    ) -> SampleScore:
        """Parse only: the judge is what scores this dataset.

        No overlap metric is emitted. The reference here is one
        acceptable explanation among many, so similarity to it measures
        resemblance to one particular wording rather than correctness.
        """
        return judged_only_score(
            response,
            metric="hypothesis_judged",
            output_contract=output_contract,
            details={},
        )

    def aggregate(self, scores: Sequence[SampleScore]) -> dict[str, float]:
        return aggregate_mean_metrics([score.metrics for score in scores])

    def apply_judge(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore, verdict: Any
    ) -> SampleScore:
        """The verdict is the score -- and the score of every stratum of it."""
        return apply_judged_metric(score, verdict, "hypothesis_judged")

    def judge_request(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore
    ) -> dict[str, Any] | None:
        """Overlap is a weak proxy for a hypothesis; offer the judge the pair."""
        if sample.task_kind != "generation" or not response.text:
            return None
        return {
            "candidate": (score.prediction or response.text)[:900],
            "gold": str((sample.reference or {}).get("gold", ""))[:900],
            "observation": C.clip_words(str(sample.fields.get("observation", "")), 150),
            "criteria": (
                "Correct if the candidate states the same hypothesis as the reference: the "
                "same mechanism or relationship, even in different words. A restatement of "
                "the research question, or a proposal to investigate, does not count."
            ),
        }

    # ------------------------------------------------------------------ #
    # documentation
    # ------------------------------------------------------------------ #

    def documentation(self) -> AdapterDocumentation:
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="ResearchBench",
            domain="Scientific Discovery: Literature-Grounded Hypothesis Discovery",
            source_url=ACCEPT_URL,
            processing_mode="Generation / Selection (separate tasks)",
            split_used=self.split_used,
            abductive_subset=(
                "generation/ and ranking/ only. retrieve/ (inspiration retrieval over a paper "
                "corpus) is excluded: finding relevant prior work is a retrieval task, not the "
                "inference of an explanation. papers/ is used only to supply background text."
            ),
            sampling_procedure=self.sampling_note(),
            metrics_description={
                "best_of_n_<metric>":
                "Every metric also gets a best_of_n_ counterpart: per record, the repeat the judge "
                "scored highest. This dataset has no checkable answer, so a plurality is meaningless "
                "-- free-text answers never repeat verbatim -- and Best-of-N replaces it.",
                "hypothesis_judged": "(PRIMARY, higher is better, 0-1) LLM-judge verdict on whether the hypothesis is "
                "the same as the paper's. 1.0 when the judge affirms, 0.0 when it does not or "
                "when the response could not be parsed. The dataset score is the mean over "
                "repeats x records.",
                "best_of_n_hypothesis_judged": "(higher is better, 0-1) Best-of-N: per record, the repeat the judge scored "
                "highest, then averaged over records. Read off the same modes.repeats samples -- "
                "no extra calls. This is what replaces self-consistency here: a plurality needs "
                "answers that can coincide, and free-text hypotheses do not.",
                "hypothesis_judged_repeat_std": "(lower is better) mean within-record standard deviation of the primary metric "
                "across repeats -- how much the same question's answers varied.",
                "repeat_agreement": "(0-1) fraction of records whose repeats all produced the identical prediction. "
                "Near 0 is expected for free-text generation.",
                "parse_failure_rate": "(lower is better, 0-1) fraction of responses no answer could be extracted from. "
                "These score 0 on the primary metric and are counted here separately, so being "
                "unparseable is distinguishable from being wrong.",
            },
            primary_metric=self.primary_metric,
            decisions=[
                "Ran generation and selection as two independent tasks, as the dataset table's "
                "'Generation / Selection (separate tasks)' requires, rather than as one mixed "
                "task: they are separate files in the release with separate gold formats.",
                "Excluded the retrieval file rather than treating retrieved inspirations as "
                "hypotheses.",
                "Resolved field names from a candidate list and fail with the keys actually "
                "found, rather than hard-coding a schema that could not be verified while the "
                "repository was gated.",
            ],
            caveats=[
                "These overlap numbers are diagnostics, not the evaluation: "
                "character/n-gram similarity punishes a correct paraphrase and "
                "rewards a wrong sentence that reuses the reference's words, so "
                "this dataset is scored by an LLM judge instead.",
                "Access is gated: an HF account must accept the dataset's terms once at "
                f"{ACCEPT_URL} before any file downloads. Metadata reads without it, which is "
                "why a missing acceptance looks like a working token.",
                "Generation is scored by overlap with a single reference hypothesis, which "
                "punishes correct paraphrases; enable the judge for a fairer reading.",
            ],
            statistics=self.base_statistics(),
        )
