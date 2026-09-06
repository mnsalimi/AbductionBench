"""AbductionRules: abduction over natural-language rule bases (Young et al., 2022).

Source: https://github.com/Strong-AI-Lab/AbductionRules

Each instance is a small theory (facts + rules) plus questions.  A question
gives an observation that does **not** follow from the theory, and the label is
the single missing fact which, if added, would make it derivable -- i.e.
knowledge-completion abduction.  Four subsets exist (Animal / Person, each in a
full and a ``Simple`` variant); all four contribute, and the subset is recorded
per sample so difficulty can be broken down.

Question categories (``QCat``) distinguish a plain observation (``0``) from a
negated one (``0_0``); both are abductive and both are kept.
"""

from __future__ import annotations

from typing import Any

from ..core.adapter import SkippedDataset
from ..core.types import AdapterDocumentation, ModelResponse, SampleScore, SampleSpec
from . import _common as C
from ._base import PooledDatasetAdapter, text_match_score

REPO_URL = "https://github.com/Strong-AI-Lab/AbductionRules"
SUBSETS = (
    "Abduction-Animal",
    "Abduction-Animal-Simple",
    "Abduction-Person",
    "Abduction-Person-Simple",
)


class AbductionRulesAdapter(PooledDatasetAdapter):
    """One sample per (theory, observation) pair from the official test splits."""

    adapter_version = "1.0"

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. You are given a small rule base "
        "and an observation that the rules alone do not entail. State the single missing fact "
        "which, added to the rule base, would make the observation derivable. Answer in the "
        "same subject-predicate form the rules use."
    )
    data_delivery_mode = "static"
    objective_metrics = True
    selection_cardinality = None
    primary_metric = "exact_match"

    def load_items(self) -> list[dict[str, Any]]:
        root = C.ensure_git_repo(
            REPO_URL, self.context.data_dir / "repo", offline=self.context.offline
        )
        wanted = self.context.option("subsets") or list(SUBSETS)
        items: list[dict[str, Any]] = []
        self._per_subset: dict[str, int] = {}
        for subset in wanted:
            path = root / "datasets" / subset / "test.jsonl"
            if not path.exists():
                self.log.warning("subset %s has no test.jsonl at %s", subset, path)
                continue
            rows = C.read_jsonl(path)
            count = 0
            for row in rows:
                context_text = C.normalize_whitespace(row.get("context"))
                for question in row.get("questions") or []:
                    if not question.get("text") or not question.get("label"):
                        continue
                    items.append(
                        {
                            "subset": subset,
                            "instance_id": row.get("id"),
                            "context": context_text,
                            "question_id": question.get("id"),
                            "observation": C.normalize_whitespace(question["text"]),
                            "label": C.normalize_whitespace(question["label"]),
                            "qcat": question.get("QCat", ""),
                        }
                    )
                    count += 1
            self._per_subset[subset] = count
        if not items:
            raise SkippedDataset("no AbductionRules test questions could be read")
        self.split_used = (
            "test (official) of "
            + ", ".join(f"{name} ({count} questions)" for name, count in self._per_subset.items())
        )
        return items

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        return SampleSpec(
            sample_id=C.stable_id("abdrules", item.get("question_id") or index),
            fields={
                "context": item["context"],
                "observation": item["observation"],
                "instructions": (
                    "Add the single missing fact that would make the observation derivable "
                    "from the theory. Answer with that one fact, phrased exactly like the "
                    "facts in the theory."
                ),
            },
            reference={"gold": item["label"]},
            task_kind="knowledge_completion",
            # The answer is one short sentence; the theory is the long part.
            max_tokens=384,
            metadata={
                "subset": item["subset"],
                "qcat": item["qcat"],
                "negated_observation": str(item["qcat"]).endswith("_0"),
                "instance_id": item["instance_id"],
            },
        )

    def score(
        self,
        sample: SampleSpec,
        response: ModelResponse,
        *,
        output_contract: dict[str, Any] | None = None,
    ) -> SampleScore:
        score = text_match_score(
            response,
            gold=sample.reference["gold"],
            output_contract=output_contract,
            primary="match",
        )
        subset = sample.metadata.get("subset", "unknown")
        if "exact_match" in score.metrics:
            score.metrics[f"exact_match_{subset}"] = score.metrics["exact_match"]
        return score

    def documentation(self) -> AdapterDocumentation:
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="AbductionRules",
            domain="Formal Reasoning: Natural-Language Logic",
            source_url=REPO_URL,
            processing_mode="Generation",
            split_used=self.split_used,
            abductive_subset=(
                "The whole dataset is abductive by construction: every question asks for the "
                "missing fact that would make an unprovable observation derivable. Both "
                "question categories (plain '0' and negated '0_0') are kept."
            ),
            sampling_procedure=self.sampling_note()
            + "; instances are first flattened into one item per (theory, observation) pair",
            metrics_description={
                "exact_match": "normalized string equality with the gold missing fact (the "
                "dataset's own metric)",
                "match": "gold fact equals or is contained in the answer (lenient variant)",
                "token_f1": "bag-of-tokens F1 against the gold fact",
                "rouge_l": "LCS F-measure against the gold fact",
                "exact_match_<subset>": "exact_match restricted to one of the four subsets",
            },
            primary_metric="exact_match",
            decisions=[
                "Flattened each instance's question list, so a sample is one (theory, "
                "observation) pair rather than a whole instance -- otherwise 300 samples would "
                "conflate several independent abductions into one score.",
                "Kept all four subsets in one pool and report exact_match per subset.",
                "max_tokens=384: the answer is a single fact, but the model must read a "
                "~200-word theory first and reasoning models spend budget before answering.",
            ],
            caveats=[
                "More than one added fact can sometimes make the observation derivable, but the "
                "dataset provides a single gold fact, so exact_match slightly understates "
                "logically valid alternatives.",
            ],
            statistics={**self.base_statistics(), "questions_per_subset": self._per_subset},
        )
