"""MedUPS: diagnostic reasoning in uncommon medical cases.

Source: https://huggingface.co/collections/oriel9p/medups
Data:   ``oriel9p/MedUPS_final_diagnosis`` (525 test cases) and
        ``oriel9p/MedUPS_mid_stream`` (mid-stream question answering)

The collection URL is not itself a dataset repository, so the adapter targets
the collection's constituent datasets by id.  Only the **final-diagnosis**
subset is used: it gives a case presentation and the published final diagnosis,
which is the abductive task.  The mid-stream subset contains generated
follow-up questions of many kinds (risk factors, next investigations, prognosis)
and is not abduction-only, so it is excluded rather than guessed at.

**Fields withheld.** ``cot``, ``final_answer``, ``raw_response`` and
``diagnosis_match`` are outputs of the authors' own model; using them as gold
(or showing them) would measure imitation of that model.  The reference is the
published ``final diagnosis`` field.  The five ``distractor*`` columns are real
retrieved ICD candidates and are used only for the optional selection subtask.
"""

from __future__ import annotations

from typing import Any

from ..core.adapter import SkippedDataset
from ..core.metrics import extract_answer_span
from ..core.types import AdapterDocumentation, ModelResponse, SampleScore, SampleSpec
from . import _common as C
from ._base import PooledDatasetAdapter, selection_score, text_match_score

REPO_ID = "oriel9p/MedUPS_final_diagnosis"
COLLECTION_URL = "https://huggingface.co/collections/oriel9p/medups"
DISTRACTOR_KEYS = tuple(f"distractor{index}" for index in range(1, 6))


class MedUPSAdapter(PooledDatasetAdapter):
    """Diagnose an uncommon published case; free text (default) or 6-way choice."""

    adapter_version = "1.0"

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. You are given an uncommon "
        "published case, delivered as a sequence of clinical steps in the order the "
        "clinicians received them. State the diagnosis that explains the case given "
        "everything disclosed so far."
    )
    data_delivery_mode = "sequential"
    objective_metrics = True
    selection_cardinality = "single"
    hypothesis_modes = ("generation", "selection",)
    hypothesis_mode_options = {
        "generation": {'subtask': 'generation'},
        "selection": {'subtask': 'selection'},
    }
    table_hypothesis_mode = "Generation"
    hypothesis_mode_justification = (
        "MedUPS ships a six-option multiple-choice item per case alongside the free-text "
        "diagnosis, so selection among the released candidates is a task the benchmark "
        "defines rather than one imposed here."
    )
    primary_metric = "diagnosis_match"

    @property
    def _subtask(self) -> str:
        return str(self.context.option("subtask", "generation"))

    def load_items(self) -> list[dict[str, Any]]:
        root = C.ensure_hf_snapshot(
            REPO_ID,
            self.context.data_dir / "hf",
            offline=self.context.offline,
            allow_patterns=["*.jsonl", "*.md"],
        )
        files = C.find_files(root, ["*.jsonl"])
        found = C.pick_split_file(files)
        if not found:
            raise SkippedDataset("no MedUPS final-diagnosis split file found")
        path, split = found
        rows = C.read_jsonl(path)
        if not rows:
            raise SkippedDataset(f"{path} contained no rows")
        self.split_used = f"{split} ({len(rows)} cases) of MedUPS_final_diagnosis"
        return rows

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        presentation = C.normalize_whitespace(
            item.get("clean text") or item.get("case_presentation") or item.get("Case presentation")
        )
        diagnosis = C.normalize_whitespace(item.get("final diagnosis"))
        if not presentation or not diagnosis:
            return None
        # Some cases carry figure references without the figures; keep the text.
        case_index = item.get("case_presentation_index", index)

        if self._subtask == "selection":
            distractors = [
                C.normalize_whitespace(item.get(key))
                for key in DISTRACTOR_KEYS
                if C.normalize_whitespace(item.get(key))
            ]
            if len(distractors) < 3:
                return None
            options = sorted({diagnosis, *distractors})
            labels = C.letter_labels(len(options))
            return SampleSpec(
                sample_id=C.stable_id("medups", case_index, item.get("chunk_number", "")),
                fields={
                    "observation": presentation,
                    "question": "Which diagnosis best explains this case?",
                    "options": options,
                    "option_labels": labels,
                },
                reference={"gold_label": labels[options.index(diagnosis)], "gold": diagnosis},
                task_kind="selection",
                max_tokens=1024,
                metadata={"case_index": case_index, "subtask": "selection"},
            )

        return SampleSpec(
            sample_id=C.stable_id("medups", case_index, item.get("chunk_number", "")),
            fields={
                "observation": presentation,
                "question": "What is the final diagnosis for this patient?",
                "instructions": (
                    "Name the specific diagnosis. These are uncommon presentations, so do not "
                    "default to the most common disease that fits loosely."
                ),
            },
            reference={"gold": diagnosis},
            task_kind="generation",
            # Long case narratives; short answer with reasoning headroom.
            max_tokens=1024,
            metadata={"case_index": case_index, "subtask": "generation"},
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
        return text_match_score(
            response,
            gold=sample.reference["gold"],
            output_contract=output_contract,
            primary="diagnosis_match",
        )

    def judge_request(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore
    ) -> dict[str, Any] | None:
        if sample.task_kind != "generation" or not response.text:
            return None
        return {
            "candidate": extract_answer_span(response.text, None)[:600],
            "gold": sample.reference["gold"],
            "observation": C.clip_words(sample.fields["observation"], 200),
            "criteria": "Equivalent disease entities (synonyms, abbreviations) count as correct.",
        }

    def apply_judge(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore, verdict: Any
    ) -> SampleScore:
        metrics = dict(score.metrics)
        metrics["diagnosis_match_judged"] = 1.0 if getattr(verdict, "positive", False) else 0.0
        return SampleScore(
            metrics=metrics,
            prediction=score.prediction,
            parse_ok=score.parse_ok,
            details={**score.details, "judge_label": getattr(verdict, "label", None)},
        )

    def documentation(self) -> AdapterDocumentation:
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="MedUPS",
            domain="Healthcare: Clinical Decision-Making",
            source_url=COLLECTION_URL,
            processing_mode="Generation (default) / Selection",
            split_used=self.split_used,
            abductive_subset=(
                "The final-diagnosis subset only (case presentation -> published diagnosis). The "
                "MedUPS_mid_stream subset asks generated follow-up questions of many kinds (risk "
                "factors, next tests, prognosis) that are not uniformly abductive, so it is "
                "excluded rather than partially guessed at."
            ),
            sampling_procedure=self.sampling_note(),
            metrics_description={
                "diagnosis_match": "1 if the answer equals or contains the published diagnosis "
                "(primary)",
                "exact_match": "strict normalized equality with the published diagnosis",
                "token_f1": "bag-of-tokens F1 against the published diagnosis",
                "rouge_l": "LCS F-measure against the published diagnosis",
                "accuracy": "selection subtask: 1 if the chosen option is the gold diagnosis",
                "diagnosis_match_judged": "LLM-judge equivalence verdict (only when "
                "engine.judge.enabled)",
            },
            primary_metric="accuracy" if self._subtask == "selection" else "diagnosis_match",
            decisions=[
                "The suite's URL points at a Hugging Face *collection*, which is not a loadable "
                "dataset; resolved it to its member datasets and used MedUPS_final_diagnosis.",
                "Used the official test split (525 cases).",
                "Withheld and never scored against cot / final_answer / raw_response / "
                "diagnosis_match: these are the authors' own model outputs, not gold data.",
                "The selection subtask uses the five retrieved ICD distractors shipped with each "
                "case; no distractors are invented.",
            ],
            caveats=[
                "Cases are published reports of uncommon presentations and may be memorized.",
                "Cases can appear more than once with different chunk_number values; the sample "
                "id includes the chunk so duplicates are visible in the records.",
            ],
            statistics={**self.base_statistics(), "subtask": self._subtask},
        )
