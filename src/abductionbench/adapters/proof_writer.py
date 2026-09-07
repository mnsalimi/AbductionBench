"""ProofWriter: abduction over natural-language rule bases (Tafjord et al., 2021).

Source: https://allenai.org/data/proofwriter
Data:   https://huggingface.co/datasets/arqa39/proofwriter-source (mirror of the
        official ProofWriter release: OWA/CWA x depth, train/dev/test parquet)

**Why the task is constructed here, and how.** ProofWriter's paper defines an
abduction task ("which missing fact would make this statement provable?"), but
the official ``meta-abduct-*`` files are not present in any accessible mirror --
the AI2 S3 link now returns HTTP 403 and every Hugging Face mirror carries only
the QA/proof portions.  Rather than skip the dataset, this adapter reconstructs
the same task **deterministically from the released proofs**:

1. take a test theory and a question whose answer is provable;
2. read its proof and find the ground facts (``triple`` entries) it uses;
3. delete exactly one of those facts from the theory;
4. ask for the fact that must be added back.

The deleted fact is the gold answer.  Nothing is invented: the theory, the
statement and the proof all come from the release, and the construction is
seeded so every model sees the identical items.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Sequence
from typing import Any

from ..core.adapter import SkippedDataset
from ..core.metrics import aggregate_mean_metrics
from ..core.types import AdapterDocumentation, ModelResponse, SampleScore, SampleSpec
from . import _common as C
from ._base import PooledDatasetAdapter, text_match_score

REPO_ID = "arqa39/proofwriter-source"
DEFAULT_CONFIGS = ("OWA-depth-1", "OWA-depth-2", "OWA-depth-3", "OWA-depth-5")
_TRIPLE_RE = re.compile(r"\btriple\d+\b")


class ProofWriterAdapter(PooledDatasetAdapter):
    """Abduce the fact removed from a ProofWriter theory."""

    adapter_version = "1.0"

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. You are given a rule base and a "
        "statement it can almost prove. State the single fact that is missing from the theory "
        "and would complete the proof."
    )
    data_delivery_mode = "static"

    answer_format = "one fact per line, or None"
    answer_constraints = (
        "output only the missing facts",
        "each missing fact must be a single fact, not a rule",
        "if several single facts would each work, output all of them",
        "put each answer on its own line",
        "if there is no valid single missing fact, output exactly: None",
        "do not explain your reasoning",
    )
    objective_metrics = True
    selection_cardinality = None
    primary_metric = "exact_match"

    def load_items(self) -> list[dict[str, Any]]:
        configs = list(self.context.option("configs", DEFAULT_CONFIGS))
        patterns = [f"{config}/test.parquet" for config in configs]
        root = C.ensure_hf_snapshot(
            REPO_ID,
            self.context.data_dir / "hf",
            offline=self.context.offline,
            allow_patterns=patterns + ["*.md"],
        )
        items: list[dict[str, Any]] = []
        self._per_config: dict[str, int] = {}
        for config in configs:
            paths = C.find_files(root / config, ["test.parquet"])
            if not paths:
                self.log.warning("ProofWriter config %s not found in the snapshot", config)
                continue
            rows = C.read_parquet_rows(paths[0])
            for row in rows:
                items.append({**row, "config": config})
            self._per_config[config] = len(rows)
        if not items:
            raise SkippedDataset(
                "no ProofWriter test parquet files could be read from the mirror"
            )
        self.split_used = "test (official) of " + ", ".join(
            f"{name} ({count} theories)" for name, count in self._per_config.items()
        )
        return items

    @staticmethod
    def _as_dict(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        triples = self._as_dict(item.get("triples"))
        rules = self._as_dict(item.get("rules"))
        questions = self._as_dict(item.get("questions"))
        if not triples or not questions:
            return None

        # Deterministic choice of the question and the fact to ablate.
        rng = random.Random(f"{self.context.seed}::proofwriter::{item.get('id')}")
        candidates = []
        for key, question in questions.items():
            if not isinstance(question, dict) or question.get("answer") is not True:
                continue
            proof = str(question.get("proofs") or "")
            used = [name for name in _TRIPLE_RE.findall(proof) if name in triples]
            # Depth >= 1 keeps the task non-trivial: a depth-0 question *is* a fact.
            if used and int(question.get("QDep") or 0) >= 1:
                candidates.append((key, question, used))
        if not candidates:
            return None
        question_key, question, used_triples = rng.choice(candidates)
        removed = rng.choice(used_triples)
        gold = C.normalize_whitespace(triples[removed].get("text"))
        statement = C.normalize_whitespace(question.get("question"))
        if not gold or not statement:
            return None

        remaining_facts = [
            C.normalize_whitespace(value.get("text"))
            for name, value in triples.items()
            if name != removed
        ]
        rule_texts = [C.normalize_whitespace(value.get("text")) for value in rules.values()]
        theory = "\n".join(
            ["Facts:", *(f"- {fact}" for fact in remaining_facts if fact),
             "Rules:", *(f"- {rule}" for rule in rule_texts if rule)]
        )
        depth = int(question.get("QDep") or 0)
        return SampleSpec(
            sample_id=C.stable_id("pw", item.get("id"), question_key, removed),
            fields={
                "context": theory,
                "observation": statement,
                "instructions": (
                    "Exactly one fact is missing from the list of facts. State that missing fact, "
                    "phrased like the other facts. Do not restate the observation itself, and do "
                    "not add a rule."
                ),
            },
            reference={"gold": gold, "removed": removed},
            task_kind="knowledge_completion",
            # One short fact; the theory is the long part.  Deeper proofs need
            # more thinking room, so the budget scales with proof depth.
            max_tokens=self.clamp_max_tokens(384 + 128 * depth, low=384, high=1024),
            metadata={
                "config": item.get("config"),
                "theory_id": item.get("id"),
                "question": question_key,
                "question_depth": depth,
                "n_facts": len(triples),
                "n_rules": len(rules),
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
        depth = sample.metadata.get("question_depth")
        if depth is not None and "exact_match" in score.metrics:
            score.metrics[f"exact_match_depth_{depth}"] = score.metrics["exact_match"]
        config = sample.metadata.get("config")
        if config and "exact_match" in score.metrics:
            score.metrics[f"exact_match_{config}"] = score.metrics["exact_match"]
        return score

    def aggregate(self, scores: Sequence[SampleScore]) -> dict[str, float]:
        return aggregate_mean_metrics([score.metrics for score in scores])

    def documentation(self) -> AdapterDocumentation:
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="ProofWriter",
            domain="Formal Reasoning: Natural-Language Logic",
            source_url="https://allenai.org/data/proofwriter",
            processing_mode="Generation",
            split_used=self.split_used,
            abductive_subset=(
                "Abduction items are constructed from the released proofs: a ground fact used in "
                "a provable statement's proof is deleted from the theory and must be recovered. "
                "Only questions with proof depth >= 1 are used, so the answer is never the "
                "observation restated."
            ),
            sampling_procedure=(
                self.sampling_note()
                + "; within a theory, the question and the ablated fact are chosen by a "
                "theory-id-seeded RNG, so the constructed items are identical across models and runs"
            ),
            metrics_description={
                "exact_match": "normalized equality with the deleted fact (primary)",
                "match": "deleted fact equals or is contained in the answer",
                "token_f1": "bag-of-tokens F1 against the deleted fact",
                "rouge_l": "LCS F-measure against the deleted fact",
                "exact_match_depth_<d>": "exact_match restricted to proof depth d",
                "exact_match_<config>": "exact_match restricted to one OWA/CWA depth configuration",
            },
            primary_metric="exact_match",
            decisions=[
                "The official abduction files are unreachable (AI2 S3 returns 403; HF mirrors "
                "carry only the QA/proof portions), so the abduction task was reconstructed from "
                "the released proofs exactly as the paper defines it. This is a construction, and "
                "it is reported as such.",
                "Used the OWA (open-world) configurations at depths 1, 2, 3 and 5 by default: "
                "open-world semantics is the setting in which a missing fact is meaningful. "
                "options.configs can select CWA or other depths.",
                "Only one fact is ablated per item, and only from questions with depth >= 1.",
                "max_tokens scales with proof depth (384 + 128 per depth level, capped at 1024).",
            ],
            caveats=[
                "More than one addition can restore derivability (including restating the "
                "observation); the prompt forbids the trivial answer and exact_match credits only "
                "the deleted fact, so the metric is a lower bound on logically valid answers.",
                "Because items are constructed here, numbers are not comparable to published "
                "ProofWriter abduction results.",
            ],
            statistics={**self.base_statistics(), "theories_per_config": self._per_config},
        )
