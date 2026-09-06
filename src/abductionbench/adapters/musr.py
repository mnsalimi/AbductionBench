"""MuSR: multistep soft reasoning (Sprague et al., ICLR 2024).

Source: https://github.com/Zayne-Sprague/MuSR

MuSR ships three domains; only **murder mysteries** are abductive.  Each of the
250 murder-mystery instances gives a ~900-word narrative and asks *"Who is the
most likely murderer?"* with two suspects -- inference to the best explanation
over means, motive and opportunity spread through the story.  The other two
domains are excluded:

* ``object_placements`` asks what a character *believes* about an object's
  location (theory of mind / state tracking, not explanation of an observation);
* ``team_allocation`` asks for the best assignment of people to tasks
  (constrained optimization).

Neither infers a hypothesis that explains an observation, so including them
would dilute what this dataset measures here.
"""

from __future__ import annotations

from typing import Any

from ..core.adapter import SkippedDataset
from ..core.types import AdapterDocumentation, ModelResponse, SampleScore, SampleSpec
from . import _common as C
from ._base import PooledDatasetAdapter, selection_score

REPO_URL = "https://github.com/Zayne-Sprague/MuSR"


class MuSRAdapter(PooledDatasetAdapter):
    """Murder-mystery domain of MuSR: name the murderer."""

    adapter_version = "1.0"

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. You are given a murder mystery. "
        "Name the murderer, and be guided by means, motive and opportunity as the narrative "
        "establishes them -- the culprit is the suspect all three converge on."
    )
    data_delivery_mode = "static"
    objective_metrics = True
    selection_cardinality = "single"
    primary_metric = "accuracy"

    def load_items(self) -> list[dict[str, Any]]:
        root = C.ensure_git_repo(
            REPO_URL, self.context.data_dir / "repo", offline=self.context.offline
        )
        domain = str(self.context.option("domain", "murder_mystery"))
        path = root / "datasets" / f"{domain}.json"
        if not path.exists():
            raise SkippedDataset(f"MuSR dataset file not found: {path}")
        payload = C.read_json(path)
        if not isinstance(payload, list):
            raise SkippedDataset(f"unexpected MuSR structure in {path}")
        items: list[dict[str, Any]] = []
        for instance_index, instance in enumerate(payload):
            context_text = instance.get("context")
            for question_index, question in enumerate(instance.get("questions") or []):
                items.append(
                    {
                        "instance": instance_index,
                        "question_index": question_index,
                        "context": context_text,
                        **{k: v for k, v in question.items() if k.startswith(("question", "choices", "answer"))},
                    }
                )
        self.split_used = (
            f"{domain}.json in full ({len(payload)} instances, {len(items)} questions; "
            "MuSR ships a single evaluation set with no train split)"
        )
        return items

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        narrative = C.normalize_whitespace(item.get("context"))
        choices = [C.normalize_whitespace(choice) for choice in item.get("choices") or []]
        answer = item.get("answer")
        if not narrative or len(choices) < 2 or not isinstance(answer, int):
            return None
        if not 0 <= answer < len(choices):
            return None
        labels = C.letter_labels(len(choices))
        return SampleSpec(
            sample_id=C.stable_id("musr", item["instance"], item["question_index"]),
            fields={
                "observation": narrative,
                "question": C.normalize_whitespace(item.get("question"))
                or "Who is the most likely murderer?",
                "options": choices,
                "option_labels": labels,
                "instructions": (
                    "Weigh each suspect's means, motive and opportunity as described in the "
                    "story, then choose the one the evidence best explains."
                ),
            },
            reference={"gold_label": labels[answer], "options": choices},
            task_kind="selection",
            # Multi-step reasoning over a long narrative before a one-label answer.
            max_tokens=1536,
            metadata={
                "domain": str(self.context.option("domain", "murder_mystery")),
                "instance": item["instance"],
                "narrative_words": len(narrative.split()),
            },
        )

    def score(
        self,
        sample: SampleSpec,
        response: ModelResponse,
        *,
        output_contract: dict[str, Any] | None = None,
    ) -> SampleScore:
        return selection_score(
            response,
            labels=sample.fields["option_labels"],
            gold_label=sample.reference["gold_label"],
            output_contract=output_contract,
            metric_name="accuracy",
        )

    def documentation(self) -> AdapterDocumentation:
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="MuSR",
            domain="Narrative Reasoning: Multistep Soft Reasoning",
            source_url=REPO_URL,
            processing_mode="Selection",
            split_used=self.split_used,
            abductive_subset=(
                "murder_mystery only -- inference to the best explanation of a crime from "
                "means/motive/opportunity evidence. object_placements (theory of mind) and "
                "team_allocation (assignment optimization) are not abductive and are excluded; "
                "options.domain can still point at them explicitly."
            ),
            sampling_procedure=self.sampling_note()
            + "; 250 murder-mystery questions exist, i.e. fewer than the 300 target",
            metrics_description={
                "accuracy": "1 if the selected suspect is the gold murderer (the dataset's own "
                "metric; chance is 50% with two suspects)"
            },
            primary_metric="accuracy",
            decisions=[
                "Excluded MuSR's other two domains as non-abductive (documented above).",
                "Flattened instances to one sample per question (murder mysteries have exactly "
                "one question each, so this is 1:1 here).",
                "max_tokens=1536 for deliberation over ~900-word narratives.",
            ],
            caveats=[
                "Only 250 items exist, so this dataset reports fewer than the 300-sample target.",
                "Two-way choice means chance accuracy is 50%; read scores against that baseline.",
            ],
            statistics=self.base_statistics(),
        )
