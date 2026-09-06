"""DiagnosisArena: clinical diagnosis from published case reports.

Source: https://github.com/SPIRAL-MED/DiagnosisArena
Data:   https://huggingface.co/datasets/SII-SPIRAL-MED/DiagnosisArena

915 cases, each with the presentation, physical examination, diagnostic work-up,
the final diagnosis and four candidate diagnoses with the correct letter.

**Overlap warning (important when reading results).** These are the same 915
case reports that ship inside the EvoClinician repository as the *Med-Inquire*
test file -- DiagnosisArena is the source data that benchmark re-uses.  The two
adapters therefore measure the same items and their scores must not be treated
as two independent difficulty probes.  To keep them from being literally the
same run, this adapter presents the **full** evidence (presentation +
examination + work-up), which is DiagnosisArena's own protocol, while
``med_inquire`` withholds the work-up by default (EvoClinician's premise is that
an agent must *ask* for it).  Both facts are recorded in each adapter's caveats.
"""

from __future__ import annotations

from typing import Any

from ..core.adapter import SkippedDataset
from ..core.metrics import extract_answer_span
from ..core.types import AdapterDocumentation, ModelResponse, SampleScore, SampleSpec
from . import _common as C
from ._base import PooledDatasetAdapter, selection_score, text_match_score

REPO_ID = "SII-SPIRAL-MED/DiagnosisArena"
GITHUB_URL = "https://github.com/SPIRAL-MED/DiagnosisArena"


class DiagnosisArenaAdapter(PooledDatasetAdapter):
    """Free-text diagnosis (default) or the dataset's own four-way choice."""

    adapter_version = "1.0"

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. You are given the full work-up of "
        "a published clinical case: presentation, history, examination and investigations. "
        "State the final diagnosis that explains the case as a whole."
    )
    data_delivery_mode = "static"
    objective_metrics = True
    selection_cardinality = "single"
    hypothesis_modes = ("generation", "selection",)
    hypothesis_mode_options = {
        "generation": {'subtask': 'generation'},
        "selection": {'subtask': 'selection'},
    }
    table_hypothesis_mode = "Generation"
    hypothesis_mode_justification = (
        "Every DiagnosisArena case ships four answer options with one correct diagnosis "
        "alongside the free-text gold diagnosis, so the release itself defines a closed-set "
        "selection task over the same cases as well as the open-ended one."
    )
    primary_metric = "diagnosis_match"

    primary_metric_by_mode = {
        # The two tasks are scored by different things, so each names
        # the metric it actually produces rather than inheriting one.
        "generation": "diagnosis_match",
        "selection": "accuracy",
    }

    @property
    def _subtask(self) -> str:
        return str(self.context.option("subtask", "generation"))

    def load_items(self) -> list[dict[str, Any]]:
        root = C.ensure_hf_snapshot(
            REPO_ID, self.context.data_dir / "hf", offline=self.context.offline
        )
        files = C.find_files(root, ["*.parquet"])
        found = C.pick_split_file(files) or ((files[0], "unsplit") if files else None)
        if not found:
            raise SkippedDataset("no DiagnosisArena parquet file found")
        path, split = found
        rows = C.read_parquet_rows(path)
        self.split_used = f"{split} ({len(rows)} cases) from {path.name}"
        return rows

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        presentation = C.normalize_whitespace(item.get("Case Information"))
        examination = C.normalize_whitespace(item.get("Physical Examination"))
        tests = C.normalize_whitespace(item.get("Diagnostic Tests"))
        diagnosis = C.normalize_whitespace(item.get("Final Diagnosis"))
        if not presentation or not diagnosis:
            return None
        blocks = []
        if examination:
            blocks.append(f"Physical examination:\n{examination}")
        if tests:
            blocks.append(f"Diagnostic work-up:\n{tests}")
        context = "\n\n".join(blocks)

        if self._subtask == "selection":
            raw_options = item.get("Options")
            options_map = raw_options if isinstance(raw_options, dict) else {}
            if not options_map:
                return None
            labels = sorted(str(key).upper() for key in options_map)
            options = [C.normalize_whitespace(options_map[key]) for key in sorted(options_map)]
            gold = C.normalize_whitespace(item.get("Right Option")).upper()
            if gold not in labels or len(options) < 2:
                return None
            return SampleSpec(
                sample_id=C.stable_id("dxarena", item.get("id", index)),
                fields={
                    "observation": presentation,
                    "context": context,
                    "question": "Which diagnosis best explains this presentation?",
                    "options": options,
                    "option_labels": labels,
                },
                reference={"gold_label": gold, "gold": diagnosis},
                task_kind="selection",
                max_tokens=1024,
                metadata={"id": item.get("id"), "subtask": "selection"},
            )

        return SampleSpec(
            sample_id=C.stable_id("dxarena", item.get("id", index)),
            fields={
                "observation": presentation,
                "context": context,
                "question": "What is the single most likely final diagnosis?",
                "instructions": (
                    "Name the specific disease entity, not a category or a differential list."
                ),
            },
            reference={"gold": diagnosis},
            task_kind="generation",
            # Long case with work-up; the answer is a short diagnosis name.
            max_tokens=1024,
            metadata={"id": item.get("id"), "subtask": "generation"},
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
        gold = sample.reference["gold"]
        return text_match_score(
            response,
            gold=gold,
            accepted=_parenthetical_forms(gold),
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
            name="DiagnosisArena",
            domain="Healthcare: Clinical Diagnosis",
            source_url=GITHUB_URL,
            processing_mode="Generation (default) / Selection",
            split_used=self.split_used,
            abductive_subset=(
                "The whole dataset is diagnostic abduction: name the disease that best explains "
                "the presentation, examination and work-up."
            ),
            sampling_procedure=self.sampling_note(),
            metrics_description={
                "diagnosis_match": "1 if the answer equals or contains the gold diagnosis (primary)",
                "exact_match": "strict normalized equality with the gold diagnosis",
                "token_f1": "bag-of-tokens F1 against the gold diagnosis",
                "rouge_l": "LCS F-measure against the gold diagnosis",
                "accuracy": "selection subtask: 1 if the chosen option letter is correct",
                "diagnosis_match_judged": "LLM-judge equivalence verdict (only when "
                "engine.judge.enabled)",
            },
            primary_metric="accuracy" if self._subtask == "selection" else "diagnosis_match",
            decisions=[
                "Default is free-text generation (this dataset's processing mode in the suite); "
                "the four shipped candidates are used only with options.subtask = selection.",
                "Presented the full evidence (presentation + examination + work-up), which is "
                "DiagnosisArena's own setting, and deliberately different from the med_inquire "
                "adapter's default so the two entries are not identical runs.",
                "Accepted the abbreviation the gold string itself parenthesizes; no synonyms are "
                "invented.",
                "Took the data from the authors' Hugging Face release; the GitHub repo is code only.",
            ],
            caveats=[
                "SAME UNDERLYING ITEMS AS med_inquire: EvoClinician ships these 915 cases as its "
                "Med-Inquire test file. Treat the two rows as one dataset viewed two ways, not as "
                "independent evidence.",
                "Published case reports may be memorized by large models.",
            ],
            statistics={**self.base_statistics(), "subtask": self._subtask},
        )


def _parenthetical_forms(gold: str) -> list[str]:
    out: list[str] = []
    if "(" in gold and ")" in gold:
        head = gold.split("(")[0].strip()
        inner = gold[gold.index("(") + 1 : gold.rindex(")")].strip()
        out.extend(part for part in (head, inner) if part)
    return out
