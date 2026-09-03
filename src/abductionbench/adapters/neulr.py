"""NeuLR: content-neutral logical reasoning (Xu et al., 2023).

Source: https://github.com/DeepReasoning/NeuLR

The repository ships three parallel files -- ``deductive_neutral.json``,
``inductive_neutral.json`` and ``abductive_neutral.json``.  Only the
**abductive** file is used here: each item gives a theory of facts and rules
plus a target fact that does not follow from it, and the label is the missing
premise that makes it derivable.  Symbols are deliberately meaningless
(``NPafp1fg``, ``ADPVKYvxg``), so the task isolates abductive inference from
world knowledge.
"""

from __future__ import annotations

from typing import Any

from ..core.adapter import SkippedDataset
from ..core.types import AdapterDocumentation, ModelResponse, SampleScore, SampleSpec
from . import _common as C
from ._base import PooledDatasetAdapter, text_match_score

REPO_URL = "https://github.com/DeepReasoning/NeuLR"


class NeuLRAdapter(PooledDatasetAdapter):
    """The abductive third of NeuLR: name the missing premise."""

    adapter_version = "1.0"
    primary_metric = "exact_match"

    def load_items(self) -> list[dict[str, Any]]:
        root = C.ensure_git_repo(
            REPO_URL, self.context.data_dir / "repo", offline=self.context.offline
        )
        path = root / "abductive_neutral.json"
        if not path.exists():
            raise SkippedDataset(f"NeuLR abductive file not found: {path}")
        payload = C.read_json(path)
        if not isinstance(payload, list):
            raise SkippedDataset("NeuLR abductive_neutral.json is not a list of items")
        # The release is a single unsplit file of 1,000 items.
        self.split_used = f"abductive_neutral.json in full ({len(payload)} items; no official splits)"
        return payload

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        context_text = C.normalize_whitespace(item.get("context"))
        gold = C.normalize_whitespace(item.get("label"))
        if not context_text or not gold:
            return None
        # The item's context already ends with "The fact is: <observation>."
        theory, _, observation = context_text.rpartition("The fact is:")
        if not observation:
            theory, observation = context_text, ""
        return SampleSpec(
            sample_id=C.stable_id("neulr", item.get("id", index)),
            fields={
                "context": theory.strip(),
                "observation": observation.strip() or context_text,
                "instructions": (
                    "State the single missing fact that must be added to the theory so that "
                    "the observation follows. Use exactly the notation of the theory, in the "
                    "form '<entity> is <property>.'"
                ),
            },
            reference={"gold": gold, "proof": item.get("explain")},
            task_kind="knowledge_completion",
            # A one-line symbolic fact, after reading a ~15-line theory.
            max_tokens=384,
            metadata={"id": item.get("id")},
        )

    def score(
        self,
        sample: SampleSpec,
        response: ModelResponse,
        *,
        output_contract: dict[str, Any] | None = None,
    ) -> SampleScore:
        return text_match_score(
            response,
            gold=sample.reference["gold"],
            output_contract=output_contract,
            primary="match",
        )

    def documentation(self) -> AdapterDocumentation:
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="NeuLR",
            domain="Formal Reasoning: Content-Neutral Logic",
            source_url=REPO_URL,
            processing_mode="Generation",
            split_used=self.split_used,
            abductive_subset=(
                "Only abductive_neutral.json; the deductive and inductive files in the same "
                "repository are different inference types and are ignored."
            ),
            sampling_procedure=self.sampling_note(),
            metrics_description={
                "exact_match": "normalized equality with the gold missing premise",
                "match": "gold premise equals or is contained in the answer",
                "token_f1": "bag-of-tokens F1 against the gold premise",
                "rouge_l": "LCS F-measure against the gold premise",
            },
            primary_metric="exact_match",
            decisions=[
                "The release has no train/dev/test split, so the whole 1,000-item file is the "
                "population for the seeded draw; this is reported rather than inventing a split.",
                "Split each item's context at its trailing 'The fact is:' marker so the theory "
                "goes into `context` and the unprovable fact into `observation`, matching the "
                "field contract the prompt templates expect.",
                "max_tokens=384: the answer is one symbolic line; the budget covers a reasoning "
                "model's hidden derivation.",
            ],
            caveats=[
                "Meaningless symbols make tokenization noisy, so a model may reproduce the right "
                "premise with a corrupted symbol; exact_match counts that as wrong while "
                "token_f1 partially credits it.",
            ],
            statistics=self.base_statistics(),
        )
