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
from ._base import PooledDatasetAdapter, apply_judged_metric, judged_only_score

REPO_URL = "https://github.com/DeepReasoning/NeuLR"


class NeuLRAdapter(PooledDatasetAdapter):
    """The abductive third of NeuLR: name the missing premise."""

    adapter_version = "1.0"

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. You are given premises and a "
        "conclusion with one premise missing. Name the missing premise that makes the "
        "argument go through."
    )
    data_delivery_mode = "static"

    answer_format = "one missing fact"
    task_requirements = (
        "the answer must be a single fact, not a rule",
    )
    #: Measured, not assumed: there is no candidate list. The model writes the
    #: missing fact itself, and its form varies -- "NPafp1fg is ADPRl020G",
    #: "NPafp1fg is a ADPRl020G", a trailing full stop, the fact wrapped in a
    #: sentence -- while naming the same fact. More than one missing fact can
    #: also make the same observation derivable. The judge scores the fact, not
    #: the string.
    objective_metrics = False
    selection_cardinality = None
    primary_metric = "premise_judged"

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
        return judged_only_score(
            response,
            metric="premise_judged",
            output_contract=output_contract,
            details={"gold": str(sample.reference["gold"])[:300]},
        )

    def judge_request(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore
    ) -> dict[str, Any] | None:
        if not response.text:
            return None
        return {
            "candidate": score.prediction or response.text[:400],
            "gold": sample.reference["gold"],
            "observation": C.clip_words(
                sample.fields["context"] + "\nThe fact is: " + sample.fields["observation"], 400
            ),
            "criteria": (
                "Symbols here are meaningless strings, so compare them character by character. "
                "The candidate is correct if it asserts the same fact as the reference -- the "
                "same entity having the same property -- regardless of punctuation, articles, "
                "or being wrapped in a sentence. A fact naming a different entity or a "
                "different property, or one whose symbol differs by even one character, is "
                "wrong. A rule ('if ... then ...') rather than a fact is wrong."
            ),
        }

    def apply_judge(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore, verdict: Any
    ) -> SampleScore:
        return apply_judged_metric(score, verdict, "premise_judged")

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
                "best_of_n_<metric>":
                "Every metric also gets a best_of_n_ counterpart: per record, the repeat the "
                "judge scored highest. The answer is written free-form rather than chosen from a "
                "list, so a plurality over repeats is not meaningful and Best-of-N replaces it.",
                "premise_judged": "(PRIMARY, higher is better, 0-1) LLM-judge verdict on whether "
                "the stated fact is the gold missing premise -- same entity, same property -- "
                "whatever the punctuation or sentence frame. 1.0 when the judge affirms.",
                "parse_failure_rate": "(lower is better, 0-1) fraction of responses no fact could "
                "be read from; these score 0 and are counted separately from being wrong.",
            },
            primary_metric="premise_judged",
            decisions=[
                "The release has no train/dev/test split, so the whole 1,000-item file is the "
                "population for the seeded draw; this is reported rather than inventing a split.",
                "Split each item's context at its trailing 'The fact is:' marker so the theory "
                "goes into `context` and the unprovable fact into `observation`, matching the "
                "field contract the prompt templates expect.",
                "max_tokens=384: the answer is one symbolic line; the budget covers a reasoning "
                "model's hidden derivation.",
                "Scored by an LLM judge rather than by exact match. No candidate list is shown, "
                "so the model writes the fact itself and its surface form varies freely around "
                "the same content; the judge is told the symbols are meaningless and must be "
                "compared character by character, so it grades the fact and not the wording.",
            ],
            caveats=[
                "Meaningless symbols make tokenization noisy, so a model may reproduce the "
                "right premise with one character of a symbol corrupted. The judge is instructed "
                "to count that as wrong -- a different symbol is a different entity -- so the "
                "score is strict about symbols and lenient only about phrasing.",
                "More than one missing fact can make the same observation derivable; the judge "
                "is asked about the release's own gold, so an alternative that also works scores "
                "0 and the number is a lower bound.",
            ],
            statistics=self.base_statistics(),
        )
