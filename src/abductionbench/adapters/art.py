"""ART / alphaNLI + alphaNLG: narrative abduction (Bhagavatula et al., ICLR 2020).

Source: https://github.com/allenai/abductive-commonsense-reasoning
Data:   https://storage.googleapis.com/ai2-mosaic/public/abductive-commonsense-reasoning-iclr2020/{anli,anlg}.zip

Given two observations O1 (earlier) and O2 (later), the task is to infer what
happened in between:

* **alphaNLI** (default, ``options.subtask = selection``) -- choose which of two
  hypotheses better explains the pair.  The official ``test`` split ships with
  its label file, so test is used.
* **alphaNLG** (``options.subtask = generation``) -- write the hypothesis
  yourself; the reference is the hypothesis the annotators marked plausible.

This is abduction in its purest narrative form: both stages of the framework
(generate a hypothesis, or select the better of two) over the same items.
"""

from __future__ import annotations

from typing import Any

from ..core.adapter import SkippedDataset
from ..core.types import AdapterDocumentation, ModelResponse, SampleScore, SampleSpec
from . import _common as C
from ._base import (
    PooledDatasetAdapter,
    apply_judged_metric,
    judged_only_score,
    selection_score,
)

REPO_URL = "https://github.com/allenai/abductive-commonsense-reasoning"
BASE = "https://storage.googleapis.com/ai2-mosaic/public/abductive-commonsense-reasoning-iclr2020"
#: Numbered, from the one house helper, so the label the model sees and
#: the label the gold refers to cannot drift apart.
LABELS = C.choice_labels(2)


class ARTAdapter(PooledDatasetAdapter):
    """alphaNLI selection (default) or alphaNLG generation over ART."""

    adapter_version = "1.0"

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. You are given the first and last "
        "observation of a short everyday story. The hypothesis you want is the event in "
        "between that makes the ending unsurprising given the beginning -- the most plausible "
        "thing to have happened, not the most dramatic."
    )
    data_delivery_mode = "static"

    answer_format = "one short sentence"
    task_requirements = (
        "make it plausible with both observations",
        "keep it plain and concrete",
        "do not restate either observation",
        "do not add extra causes, background detail or consequences",
    )
    options_heading = "Answer options:"
    #: Set per instance in prepare(). alphaNLI picks one of two hypotheses and is
    #: checkable; alphaNLG writes the hypothesis into an open space where the
    #: reference is one annotator's plausible sentence among many that would have
    #: been just as plausible.
    objective_metrics = True
    selection_cardinality = "single"
    hypothesis_modes = ("generation", "selection",)
    hypothesis_mode_options = {
        "generation": {'subtask': 'generation'},
        "selection": {'subtask': 'selection'},
    }
    table_hypothesis_mode = "Generation / Selection (separate tasks)"
    primary_metric = "accuracy"

    primary_metric_by_mode = {
        # The two tasks are scored by different things, so each names
        # the metric it actually produces rather than inheriting one.
        "generation": "hypothesis_judged",
        "selection": "accuracy",
    }

    @property
    def _subtask(self) -> str:
        return str(self.context.option("subtask", "selection"))

    def prepare(self) -> None:
        # alphaNLI is scored against its public label file; alphaNLG is judged.
        self.objective_metrics = self._subtask != "generation"
        super().prepare()

    def load_items(self) -> list[dict[str, Any]]:
        if self._subtask == "generation":
            root = C.extract_archive(
                C.ensure_download(
                    f"{BASE}/anlg.zip",
                    self.context.data_dir / "anlg.zip",
                    offline=self.context.offline,
                ),
                self.context.data_dir / "anlg",
            )
            candidates = C.find_files(root, ["*test*.jsonl", "*dev*.jsonl"])
            found = C.pick_split_file(candidates)
            if not found:
                raise SkippedDataset("no alphaNLG split file found after extraction")
            path, split = found
            rows = C.read_jsonl(path)
            self.split_used = f"alphaNLG {split} ({len(rows)} items) from anlg.zip"
            return [{"kind": "generation", **row} for row in rows]

        root = C.extract_archive(
            C.ensure_download(
                f"{BASE}/anli.zip",
                self.context.data_dir / "anli.zip",
                offline=self.context.offline,
            ),
            self.context.data_dir / "anli",
        )
        # Labels live in a sibling .lst file, one label per line, 1-indexed.
        pairs = []
        for split in ("test", "dev", "train"):
            data = C.find_files(root, [f"{split}.jsonl"])
            labels = C.find_files(root, [f"{split}-labels.lst"])
            if data and labels:
                pairs = [(data[0], labels[0], split)]
                break
        if not pairs:
            raise SkippedDataset("no alphaNLI split with a label file was found")
        data_path, label_path, split = pairs[0]
        rows = C.read_jsonl(data_path)
        labels = [line.strip() for line in C.read_text(label_path).splitlines() if line.strip()]
        if len(labels) != len(rows):
            raise SkippedDataset(
                f"alphaNLI {split}: {len(rows)} items but {len(labels)} labels -- refusing to "
                "align them by guesswork"
            )
        self.split_used = f"alphaNLI {split} ({len(rows)} items, labels from {label_path.name})"
        return [{"kind": "selection", "label": label, **row} for row, label in zip(rows, labels, strict=True)]

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        obs1 = C.normalize_whitespace(item.get("obs1"))
        obs2 = C.normalize_whitespace(item.get("obs2"))
        if not obs1 or not obs2:
            return None
        observation = (
            f"O1 (earlier): {obs1}\nO2 (later): {obs2}"
        )
        if item.get("kind") == "generation":
            label = item.get("label")
            hypotheses = [item.get("hyp1"), item.get("hyp2")]
            try:
                gold = C.normalize_whitespace(hypotheses[int(label) - 1])
            except (TypeError, ValueError, IndexError):
                return None
            if not gold:
                return None
            return SampleSpec(
                sample_id=C.stable_id("art-gen", item.get("story_id", index)),
                fields={
                    "observation": observation,
                    "question": "What most plausibly happened between O1 and O2?",
                    # No `instructions` line. It said "Answer with a single short sentence
                    # describing that event", which is answer SHAPE -- already
                    # carried by answer_format, the constraints and the closing.
                    # Under cot it also landed BEFORE the reasoning instruction, so
                    # the model read "answer with one sentence" and then "work
                    # through the evidence step by step". The question above already
                    # states the task.
                },
                reference={"gold": gold},
                task_kind="generation",
                # A one-sentence hypothesis; the rest of the budget is reasoning room.
                max_tokens=320,
                metadata={"story_id": item.get("story_id"), "subtask": "alphaNLG"},
            )

        hypotheses = [
            C.normalize_whitespace(item.get("hyp1")),
            C.normalize_whitespace(item.get("hyp2")),
        ]
        try:
            gold_index = int(item["label"]) - 1  # label file is 1-indexed
        except (KeyError, TypeError, ValueError):
            return None
        if not all(hypotheses) or not 0 <= gold_index < len(hypotheses):
            return None
        return SampleSpec(
            sample_id=C.stable_id("art-sel", item.get("story_id", index)),
            fields={
                "observation": observation,
                "question": (
                    "Which hypothesis more plausibly explains what happened between O1 and O2?"
                ),
                "options": hypotheses,
                "option_labels": LABELS,
            },
            reference={"gold_label": LABELS[gold_index], "options": hypotheses},
            task_kind="selection",
            max_tokens=320,
            metadata={"story_id": item.get("story_id"), "subtask": "alphaNLI"},
        )

    def score(
        self,
        sample: SampleSpec,
        response: ModelResponse,
        *,
        output_contract: dict[str, Any] | None = None,
    ) -> SampleScore:
        if sample.task_kind == "selection":
            return selection_score(
                response,
                labels=sample.fields["option_labels"],
                gold_label=sample.reference["gold_label"],
                output_contract=output_contract,
                metric_name="accuracy",
            )
        return judged_only_score(
            response,
            metric="hypothesis_judged",
            output_contract=output_contract,
            details={"gold": str(sample.reference["gold"])[:300]},
        )

    def judge_request(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore
    ) -> dict[str, Any] | None:
        if sample.task_kind != "generation" or not response.text:
            return None
        return {
            "candidate": score.prediction or response.text[:400],
            "gold": sample.reference["gold"],
            "observation": sample.fields["observation"],
            "criteria": (
                "The two observations bracket a gap in time, and the reference is one event "
                "annotators judged plausible in that gap. The candidate is correct if it too "
                "is a plausible account of what happened between them -- it need not match the "
                "reference, and a different but equally plausible event counts. An event that "
                "contradicts either observation, that merely restates one of them, or that "
                "would not lead from the first to the second, does not count."
            ),
        }

    def apply_judge(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore, verdict: Any
    ) -> SampleScore:
        return apply_judged_metric(score, verdict, "hypothesis_judged")

    def documentation(self) -> AdapterDocumentation:
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="ART (alphaNLI / alphaNLG)",
            domain="Commonsense: Narrative Abduction",
            source_url=REPO_URL,
            processing_mode="Selection (default) / Generation",
            split_used=self.split_used,
            abductive_subset=(
                "The whole dataset is abductive (infer the event between two observations). "
                "options.subtask picks selection (alphaNLI, default) or generation (alphaNLG)."
            ),
            sampling_procedure=self.sampling_note(),
            metrics_description={
                "self_consistency_<metric>":
                "Selection only: every metric gets a self_consistency_ counterpart, the plurality "
                "answer over modes.repeats samples of the same record, read off those samples "
                "rather than bought again. A label can coincide across repeats; a free-text "
                "hypothesis cannot.",
                "best_of_n_<metric>":
                "Generation only: every metric gets a best_of_n_ counterpart, the repeat the judge "
                "scored highest, which is what replaces a plurality when answers never repeat "
                "verbatim.",
                "accuracy": "(PRIMARY, higher is better) selection: 1 if the chosen hypothesis is the annotated plausible one",
                "hypothesis_judged": "(PRIMARY, higher is better, 0-1) generation: LLM-judge "
                "verdict on whether the written event is a plausible account of what happened "
                "between the two observations. The reference is shown to the judge as one "
                "plausible answer, not as the only one.",
                "parse_failure_rate": "(lower is better, 0-1) fraction of responses nothing could "
                "be read from; these score 0 and are counted separately from being wrong.",
            },
            primary_metric="accuracy" if self._subtask != "generation" else "hypothesis_judged",
            decisions=[
                "Used the official test split for alphaNLI because its label file is public "
                "(test-labels.lst); no guessing of labels was needed.",
                "Refused to proceed if item and label counts disagree, rather than aligning them "
                "heuristically.",
                "For alphaNLG the reference is the hypothesis the label marks plausible; the "
                "COMET predictions shipped alongside are ignored (they are model outputs, not "
                "gold data).",
                "Scored alphaNLG by an LLM judge on PLAUSIBILITY, not on similarity to the "
                "reference. ART's own premise is that many events fit between two observations, "
                "so scoring against the one annotators wrote down would measure agreement with "
                "an arbitrary choice; the reference is given to the judge as an example of a "
                "plausible answer rather than as the answer.",
                "Rendered the two observations as labelled 'O1 (earlier)' / 'O2 (later)' lines so "
                "the temporal order is unambiguous in every prompt template.",
            ],
            caveats=[
                "alphaNLI is a two-way choice: chance accuracy is 50%.",
                "alphaNLG's judge grades plausibility rather than agreement with the "
                "reference, which is the right question but a softer one: a model that writes a "
                "generic event fitting almost any pair of observations can score well.",
                "The judge is required for alphaNLG: with engine.judge disabled that subtask has "
                "no metric and the run fails rather than reporting an overlap number.",
            ],
            statistics={**self.base_statistics(), "subtask": self._subtask},
        )
