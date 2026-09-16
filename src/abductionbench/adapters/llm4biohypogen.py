"""LLM4BioHypoGen: propose the hypothesis a biomedical background motivates.

Source: https://github.com/TsinghuaC3I/LLM4BioHypoGen

**The task.**  An item pairs the ``background`` of a published biomedical study
-- what was already known, and the gap the authors identified -- with the
``hypothesis`` that study went on to test.  The model sees only the background
and must state the hypothesis.  This is creative abduction: the hypothesis is
not entailed by the background, it is the conjecture that would account for the
gap and is worth testing.

**Seen and unseen.**  The release splits by whether the source paper predates
the evaluated models' training cut-off (``test_seen``) or follows it
(``test_unseen``).  **Only ``test_unseen`` is used by default**, because a
hypothesis from a paper a model may have read is a memory test rather than an
abduction test.  ``options.split`` can select ``seen`` for a deliberate
leakage comparison, and the choice is recorded in the run documentation.

The release also ships the same items under ``gpt-3.5/`` and ``gpt-4/``
directories.  Those differ only in which model produced the *reference*
extraction of background and hypothesis from the paper; ``gpt-4`` is used, as
the better extraction, and the choice is stated rather than hidden.

**Scoring.**  There is a gold hypothesis, but it is one paper's phrasing of a
conjecture that can be worded many ways, so an LLM judge scores whether the
stated hypothesis is the same conjecture.
"""

from __future__ import annotations

from typing import Any

from ..core.adapter import SkippedDataset
from ..core.metrics import aggregate_mean_metrics
from ..core.types import AdapterDocumentation, ModelResponse, SampleScore, SampleSpec
from . import _common as C
from ._base import PooledDatasetAdapter, apply_judged_metric, judged_only_score

REPO_URL = "https://github.com/TsinghuaC3I/LLM4BioHypoGen.git"
EXTRACTORS = ("gpt-4", "gpt-3.5")
SPLITS = {"unseen": "test_unseen.json", "seen": "test_seen.json"}


class LLM4BioHypoGenAdapter(PooledDatasetAdapter):
    """State the hypothesis a study's background motivates."""

    adapter_version = "1.0"

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. You are given the background of "
        "a biomedical study: what was already known, and the gap the authors identified. "
        "State the hypothesis that study set out to test -- the conjecture that would "
        "account for the gap and is specific enough to be investigated."
    )
    answer_format = "one hypothesis"
    answer_constraints = (
        "state one hypothesis, not several",
        "name the factors involved and the relationship claimed between them",
        "make it specific enough to be tested",
        "do not restate the background",
        "do not describe the method or the expected result",
        "do not use introductory phrases or commentary",
    )
    data_delivery_mode = "static"
    objective_metrics = False
    selection_cardinality = None
    hypothesis_modes = ("generation",)
    hypothesis_mode_options = {"generation": {"subtask": "generation"}}
    table_hypothesis_mode = "Generation"
    primary_metric = "hypothesis_judged"

    def load_items(self) -> list[dict[str, Any]]:
        root = C.ensure_git_repo(
            REPO_URL, self.context.data_dir / "repo", depth=1, offline=self.context.offline
        )
        split = str(self.context.option("split", "unseen")).lower()
        if split not in SPLITS:
            raise SkippedDataset(
                f"options.split={split!r}; expected one of {sorted(SPLITS)}"
            )
        extractor = str(self.context.option("extractor", "gpt-4"))
        if extractor not in EXTRACTORS:
            raise SkippedDataset(
                f"options.extractor={extractor!r}; expected one of {list(EXTRACTORS)}"
            )
        path = root / "data" / extractor / SPLITS[split]
        if not path.is_file():
            raise SkippedDataset(f"{path} is not in the clone of {REPO_URL}")
        rows = C.read_json(path)
        if not isinstance(rows, list) or not rows:
            raise SkippedDataset(f"{path} is not a non-empty list of items")
        self.split_used = (
            f"data/{extractor}/{SPLITS[split]} ({len(rows)} studies; the "
            f"{'seen' if split == 'unseen' else 'unseen'} split is not used)"
        )
        self.split_choice = split
        return rows

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        background = C.normalize_whitespace(item.get("background"))
        gold = C.normalize_whitespace(item.get("hypothesis"))
        if not background or not gold:
            return None
        return SampleSpec(
            sample_id=C.stable_id("llm4biohypogen", self.split_choice, index),
            fields={
                "observation": C.clip_words(background, 700),
                "question": "What hypothesis did this study set out to test?",
            },
            reference={"gold": gold},
            task_kind="generation",
            metadata={"split": self.split_choice},
        )

    def score(
        self,
        sample: SampleSpec,
        response: ModelResponse,
        *,
        output_contract: dict[str, Any] | None = None,
    ) -> SampleScore:
        """Parse only: the judge compares the conjecture to the paper's own."""
        return judged_only_score(
            response,
            metric="hypothesis_judged",
            output_contract=output_contract,
            details={},
        )

    def judge_request(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore
    ) -> dict[str, Any] | None:
        if not response.text:
            return None
        return {
            "candidate": score.prediction or response.text[:900],
            "gold": sample.reference["gold"],
            "observation": C.clip_words(sample.fields["observation"], 250),
            "criteria": (
                "The candidate is correct if it proposes the same relationship between the "
                "same factors as the reference hypothesis, however it is worded. A "
                "hypothesis about different factors, or one that states the relationship in "
                "the opposite direction, is not the same hypothesis. Broader restatements "
                "of the background that commit to no relationship do not count."
            ),
        }

    def apply_judge(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore, verdict: Any
    ) -> SampleScore:
        return apply_judged_metric(score, verdict, "hypothesis_judged")

    def aggregate(self, scores) -> dict[str, float]:
        return aggregate_mean_metrics([score.metrics for score in scores])

    def documentation(self) -> AdapterDocumentation:
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="LLM4BioHypoGen",
            domain="Scientific Discovery: Biomedical Hypothesis Generation",
            source_url="https://github.com/TsinghuaC3I/LLM4BioHypoGen",
            processing_mode="Generation",
            split_used=getattr(self, "split_used", "data/gpt-4/test_unseen.json"),
            abductive_subset=(
                "THE UNSEEN SPLIT ONLY, by default. The release separates studies published "
                "before the evaluated models' training cut-off (test_seen) from those after "
                "it (test_unseen); a hypothesis from a paper the model may have read tests "
                "memory rather than abduction. options.split selects 'seen' for a deliberate "
                "leakage comparison, and whichever was used is recorded here. The task "
                "itself is wholly abductive: the hypothesis is not entailed by the "
                "background, it is the conjecture that would account for the gap."
            ),
            sampling_procedure=self.sampling_note(),
            metrics_description={
                "hypothesis_judged": (
                    "(PRIMARY, higher is better, 0-1) LLM-judge verdict on whether the stated "
                    "hypothesis proposes the same relationship between the same factors as "
                    "the paper's, however worded. 1.0 when the judge affirms."
                ),
                "parse_failure_rate": (
                    "(lower is better, 0-1) fraction of responses no hypothesis could be "
                    "extracted from; these score 0 and are counted separately from being wrong."
                ),
                "best_of_n_<metric>":
                "Every metric also gets a best_of_n_ counterpart: per record, the repeat the "
                "judge scored highest. This dataset has no checkable answer, so a plurality is "
                "meaningless -- free-text hypotheses never repeat verbatim -- and Best-of-N "
                "replaces it.",
            },
            primary_metric="hypothesis_judged",
            decisions=[
                "Used test_unseen by default so the score is not a memory test; the seen "
                "split remains available as a deliberate leakage comparison.",
                "Used the gpt-4 extraction of background and hypothesis rather than the "
                "gpt-3.5 one. Both directories hold the same studies and differ only in "
                "which model extracted the reference text.",
                "Scored by an LLM judge against the paper's own hypothesis: the gold is one "
                "phrasing of a conjecture that can be worded many ways.",
            ],
            caveats=[
                "The reference hypothesis was itself extracted from the paper by a language "
                "model, so it is the release's reading of the study rather than a hand "
                "annotation.",
                "Backgrounds are clipped to 700 words; a few long ones lose their tail.",
                "'Unseen' is relative to the models the release evaluated, not to the model "
                "under test here -- a newer model may still have seen these papers.",
            ],
            statistics={**self.base_statistics(), "split": getattr(self, "split_choice", "")},
        )
