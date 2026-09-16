"""ENWN & EntailmentBank: missing-premise abduction.

Source: https://github.com/Zayne-sprague/Natural_Language_Deduction_with_Incomplete_Information

The repository's ``data/step/abductive/`` directory holds exactly the abductive
step task: the ``.source`` file gives one known premise plus a goal, and the
``.target`` file gives the **missing premise** that would complete the
entailment.  Two corpora are provided:

* ``entailmentbank`` -- science-exam entailment trees (abductive step data ships
  as a train file only);
* ``enwn`` (Everyday Norms: Why Not) -- everyday normative reasoning, with a
  validation file.

Both are pooled and the corpus is recorded per sample.  The repository's other
directories (``wanli``, ``mnli``, deductive step data, ``task_1/2/3`` full
trees) are different tasks and are ignored.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..core.adapter import SkippedDataset
from ..core.metrics import aggregate_mean_metrics
from ..core.types import AdapterDocumentation, ModelResponse, SampleScore, SampleSpec
from . import _common as C
from ._base import PooledDatasetAdapter, apply_judged_metric, judged_only_score

REPO_URL = (
    "https://github.com/Zayne-sprague/Natural_Language_Deduction_with_Incomplete_Information"
)


class EnwnEntailmentBankAdapter(PooledDatasetAdapter):
    """Recover the missing premise of an incomplete entailment step."""

    adapter_version = "1.0"

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. You are given the premises of an "
        "entailment step and its conclusion, with one premise missing. State the missing "
        "premise: the single statement that, together with those given, entails the "
        "conclusion."
    )
    data_delivery_mode = "static"

    answer_format = "one missing premise"
    task_requirements = (
        "do not restate the hypothesis or the given premises",
    )
    #: Measured, not assumed: the missing premise is written in free English
    #: into an unbounded space, and several different premises complete the
    #: same step. "Animals need water to survive" against a gold of "an animal
    #: requires water for survival" is the same premise and fails a string
    #: comparison, so the judge scores it.
    objective_metrics = False
    selection_cardinality = None
    primary_metric = "premise_judged"

    def load_items(self) -> list[dict[str, Any]]:
        root = C.ensure_git_repo(
            REPO_URL, self.context.data_dir / "repo", offline=self.context.offline
        )
        base = root / "data" / "step" / "abductive"
        if not base.exists():
            raise SkippedDataset(f"abductive step data not found at {base}")
        items: list[dict[str, Any]] = []
        self._per_corpus: dict[str, int] = {}
        for corpus_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            pairs = self._source_target_pairs(corpus_dir)
            if not pairs:
                continue
            corpus = "enwn" if "enwn" in corpus_dir.name else "entailmentbank"
            for split_name, source_path, target_path in pairs:
                sources = C.read_lines(source_path)
                targets = C.read_lines(target_path)
                if len(sources) != len(targets):
                    self.log.warning(
                        "%s: %d sources vs %d targets; skipping this pair",
                        corpus_dir.name,
                        len(sources),
                        len(targets),
                    )
                    continue
                for position, (source, target) in enumerate(zip(sources, targets, strict=True)):
                    items.append(
                        {
                            "corpus": corpus,
                            "subdir": corpus_dir.name,
                            "split": split_name,
                            "position": position,
                            "source": source,
                            "target": target,
                        }
                    )
                self._per_corpus[f"{corpus_dir.name}/{split_name}"] = len(sources)
        if not items:
            raise SkippedDataset("no abductive source/target pairs could be read")
        self.split_used = (
            "abductive step data, best split available per corpus: "
            + ", ".join(f"{name} ({count} steps)" for name, count in sorted(self._per_corpus.items()))
            + " -- EntailmentBank's abductive step data ships as a train file only"
        )
        return items

    @staticmethod
    def _source_target_pairs(directory: Path) -> list[tuple[str, Path, Path]]:
        """Pick one (source, target) pair per corpus, preferring test > val > train."""
        sources = {path.stem: path for path in directory.glob("*.source")}
        targets = {path.stem: path for path in directory.glob("*.target")}
        shared = sorted(set(sources) & set(targets))
        if not shared:
            return []
        for preference in ("test", "val", "validation", "dev", "train"):
            for stem in shared:
                if preference in stem.lower():
                    return [(stem, sources[stem], targets[stem])]
        stem = shared[0]
        return [(stem, sources[stem], targets[stem])]

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        source = C.normalize_whitespace(item.get("source"))
        target = C.normalize_whitespace(item.get("target"))
        if not source or not target:
            return None
        # The source line is "<known premise> <goal>"; the goal is its last
        # sentence, so both are shown explicitly rather than as one blob.
        sentences = [part.strip() for part in source.split(". ") if part.strip()]
        if len(sentences) >= 2:
            goal = sentences[-1].rstrip(".") + "."
            known = ". ".join(sentences[:-1]).rstrip(".") + "."
        else:
            goal, known = source, ""
        return SampleSpec(
            sample_id=C.stable_id("enwn", item["subdir"], item["split"], item["position"]),
            fields={
                "context": f"Known premise: {known}" if known else "",
                "observation": goal,
                "instructions": (
                    "One premise is missing. State the single missing premise that, together "
                    "with the known premise, would entail the conclusion."
                ),
            },
            reference={"gold": target},
            task_kind="knowledge_completion",
            max_tokens=320,
            metadata={
                "corpus": item["corpus"],
                "subdir": item["subdir"],
                "split": item["split"],
            },
        )

    def score(
        self,
        sample: SampleSpec,
        response: ModelResponse,
        *,
        output_contract: dict[str, Any] | None = None,
    ) -> SampleScore:
        corpus = sample.metadata.get("corpus", "unknown")
        # The per-corpus stratum is seeded here and filled by the judge along with
        # the base metric: it is the same verdict seen through a filter.
        return judged_only_score(
            response,
            metric="premise_judged",
            output_contract=output_contract,
            extra_metrics={f"premise_judged_{corpus}": 0.0},
            details={"gold": str(sample.reference["gold"])[:300]},
        )

    def judge_request(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore
    ) -> dict[str, Any] | None:
        if not response.text:
            return None
        return {
            "candidate": score.prediction or response.text[:600],
            "gold": sample.reference["gold"],
            "observation": C.clip_words(
                (sample.fields.get("context") or "")
                + "\nConclusion: "
                + sample.fields.get("observation", ""),
                200,
            ),
            "criteria": (
                "The candidate is correct if, taken with the premise already given, it would "
                "license the conclusion in the same way the reference premise does -- however "
                "it is worded. A premise that is about something else, that merely restates "
                "the conclusion or the given premise, or that is too weak to close the gap, "
                "does not count."
            ),
        }

    def apply_judge(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore, verdict: Any
    ) -> SampleScore:
        return apply_judged_metric(score, verdict, "premise_judged")

    def aggregate(self, scores: Sequence[SampleScore]) -> dict[str, float]:
        return aggregate_mean_metrics([score.metrics for score in scores])

    def documentation(self) -> AdapterDocumentation:
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="ENWN & EntailmentBank",
            domain="Formal Reasoning: Missing-Premise Inference",
            source_url=REPO_URL,
            processing_mode="Generation",
            split_used=self.split_used,
            abductive_subset=(
                "Only data/step/abductive/ -- the missing-premise task. The repository's WANLI, "
                "MNLI, deductive-step and full-tree (task_1/2/3) data are different tasks and are "
                "ignored."
            ),
            sampling_procedure=self.sampling_note()
            + "; the two corpora are pooled and reported separately",
            metrics_description={
                "best_of_n_<metric>":
                "Every metric also gets a best_of_n_ counterpart: per record, the repeat the "
                "judge scored highest. A premise is free text with many correct phrasings, so "
                "a plurality over repeats is not meaningful and Best-of-N replaces it.",
                "premise_judged": "(PRIMARY, higher is better, 0-1) LLM-judge verdict on "
                "whether the stated premise closes the same gap as the gold one, however worded. "
                "1.0 when the judge affirms.",
                "premise_judged_<corpus>": "the metric restricted to entailmentbank or enwn",
                "parse_failure_rate": "(lower is better, 0-1) fraction of responses no premise "
                "could be read from; these score 0 and are counted separately from being wrong.",
            },
            primary_metric="premise_judged",
            decisions=[
                "Split each source line into 'known premise' and 'conclusion' at its last "
                "sentence, so the prompt states both roles explicitly instead of pasting one blob.",
                "Preferred a test/val file per corpus and fell back to train where the repository "
                "ships nothing else (EntailmentBank's abductive step data); this is reported.",
                "max_tokens=320: the answer is a single short premise.",
                "Scored by an LLM judge rather than by string overlap. The premise is written in "
                "free English, several different premises complete the same step, and the gold is "
                "one annotator's phrasing of one of them -- exact_match punishes a correct "
                "paraphrase and token overlap rewards a wrong sentence that reuses the words.",
            ],
            caveats=[
                "Several different premises can complete an entailment. The judge is asked "
                "whether the candidate closes the same gap as the gold, not whether it is the "
                "gold, but a correct premise the gold does not anticipate may still be marked "
                "wrong -- the score is a lower bound.",
                "Because EntailmentBank's abductive step data is a train file, models trained on "
                "public EntailmentBank data may have seen these items.",
            ],
            statistics={**self.base_statistics(), "steps_per_file": self._per_corpus},
        )
