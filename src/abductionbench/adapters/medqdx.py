"""MedQDx: diagnosis under partial information.

Source: https://github.com/MaiWert/MedQDx

MedQDx generates patient vignettes from the Symptom-Disease Prediction Dataset
at three levels of information completeness (100%, 80%, 50% of the disease's
symptoms) and, in its own benchmark, has an LLM "doctor" interrogate an LLM
"patient" for three rounds before diagnosing.

**How it is adapted.** The multi-turn interrogation cannot be reproduced in a
single-turn harness, but the part that makes MedQDx abductive is the diagnosis
itself: infer the disease that best explains an incomplete symptom picture.  Two
modes are provided:

* ``information_levels`` (default) -- each of the 100 vignettes is evaluated at
  all three completeness levels, giving exactly 300 samples and a built-in
  difficulty gradient (accuracy at 100% vs 80% vs 50% information).
* ``inquiry_rounds`` -- the benchmark CSV's 50% vignette plus the recorded
  doctor-patient Q&A (0, 1 or 3 rounds appended as context), which measures
  whether extra inquiry answers help.

**Selection.** The disease label space is closed (the source dataset has 41
diseases, 29 of which occur in the generated cases), so the task is rendered as
selection among the gold diagnosis and distractors drawn from that same closed
label set -- which is what makes this the "Selection" processing mode.  The
distractors are drawn with the run seed, so they are identical across models.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

from ..core.adapter import SkippedDataset
from ..core.metrics import aggregate_mean_metrics
from ..core.types import (
    AdapterDocumentation,
    ChatMessage,
    ModelResponse,
    SampleScore,
    SampleSpec,
)
from . import _common as C
from ._base import PooledDatasetAdapter, selection_score
from ._interactive import EvidenceStore, InteractiveMixin, parse_action

REPO_URL = "https://github.com/MaiWert/MedQDx"
LEVELS = ("100% Case", "80% Case", "50% Case")


class MedQDxAdapter(InteractiveMixin, PooledDatasetAdapter):
    """Closed-set diagnosis from vignettes of varying completeness."""

    adapter_version = "1.0"

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. You are given a clinical vignette "
        "that may be incomplete, and a closed set of candidate diagnoses. Choose the "
        "diagnosis best supported by the information actually present."
    )
    data_delivery_mode = "interactive"
    objective_metrics = True
    selection_cardinality = "single"
    primary_metric = "accuracy"

    def load_items(self) -> list[dict[str, Any]]:
        root = C.ensure_git_repo(
            REPO_URL, self.context.data_dir / "repo", offline=self.context.offline
        )
        mode = str(self.context.option("mode", "information_levels"))
        if mode == "inquiry_rounds":
            path = root / "Benchmark Creation" / "MedQDx_Benchmark.csv"
        else:
            path = root / "EDA and Baseline" / "Patient Cases.csv"
        if not path.exists():
            raise SkippedDataset(f"MedQDx CSV not found: {path}")
        rows = C.read_csv_rows(path)
        rows = [row for row in rows if (row.get("prognosis") or "").strip()]
        if not rows:
            raise SkippedDataset(f"no usable rows in {path}")

        self._label_space = sorted({(row["prognosis"] or "").strip() for row in rows})
        items: list[dict[str, Any]] = []
        if mode == "inquiry_rounds":
            for case_index, row in enumerate(rows):
                for rounds in (0, 1, 3):
                    items.append({"row": row, "case_index": case_index, "rounds": rounds})
            self.split_used = (
                f"MedQDx_Benchmark.csv ({len(rows)} cases) x 3 inquiry conditions "
                "(0/1/3 recorded Q&A rounds); no official split exists"
            )
        else:
            for case_index, row in enumerate(rows):
                for level in LEVELS:
                    if (row.get(level) or "").strip():
                        items.append({"row": row, "case_index": case_index, "level": level})
            self.split_used = (
                f"Patient Cases.csv ({len(rows)} vignettes) x 3 information levels "
                "(100%/80%/50%); no official split exists"
            )
        self._mode = mode
        return items

    def _options_for(self, gold: str, salt: str) -> list[str]:
        """Gold plus deterministic distractors from the dataset's own label set."""
        count = int(self.context.option("n_options", 5))
        rng = random.Random(f"{self.context.seed}::medqdx::{salt}")
        pool = [label for label in self._label_space if label != gold]
        rng.shuffle(pool)
        options = [gold, *pool[: max(1, count - 1)]]
        rng.shuffle(options)
        return options

    # ------------------------------------------------------------------ #
    # the questioning loop -- MedQDx's whole premise
    # ------------------------------------------------------------------ #

    ACTIONS = ("ask", "diagnosis")
    max_turns = 12
    category_limits = {"ask": 8}

    QUESTIONING_PROMPT = (
        "You are a physician assessing a patient from an incomplete presentation. You may ask "
        "the patient about any symptom, one question per turn, and the patient answers only "
        "what you ask. When you can name the condition, commit to it.\n\n"
        "Reply with a single JSON object and nothing else:\n"
        '{"reasoning": "...", "action": "ask", "query": "the symptom you are asking about"}\n'
        'or {"reasoning": "...", "action": "diagnosis", "query": "<label>"}\n\n'
        "Ask about symptoms that would separate the conditions you are considering. Every "
        "question costs a turn, so ask the discriminating one."
    )

    def interactive_start(self, sample: SampleSpec) -> tuple[list[ChatMessage], dict[str, Any]]:
        store = EvidenceStore()
        # The case's symptom list is the patient: a symptom in the list is
        # present, and anything else is absent. MedQDx's cases are built from
        # exactly that list, so nothing has to be invented either way.
        present = sample.metadata.get("_symptoms") or []
        store.categories["ask"] = {
            symptom.replace("_", " "): "yes, that is present" for symptom in present
        }
        options = sample.fields.get("options") or []
        labels = sample.fields.get("option_labels") or []
        listing = "\n".join(f"{label}) {option}" for label, option in zip(labels, options,
                                                                          strict=False))
        opening = (
            f"{sample.fields.get('observation', '')}\n\n"
            f"Possible conditions:\n{listing}\n\n"
            "Ask about symptoms, then give the label of your diagnosis."
        )
        return (
            [
                ChatMessage(role="system", content=self.QUESTIONING_PROMPT),
                ChatMessage(role="user", content=opening),
            ],
            {"evidence": store, "counts": {}},
        )

    def interactive_step(
        self, sample: SampleSpec, state: dict[str, Any], assistant_text: str
    ) -> str | None:
        action = parse_action(assistant_text, actions=self.ACTIONS)
        if action.action == "diagnosis":
            return None
        if not action.action:
            state["parse_errors"] = state.get("parse_errors", 0) + 1
            if state["parse_errors"] > 2:
                return None
            return 'Reply with one JSON object: {"action": "ask"|"diagnosis", "query": "..."}'
        self.bump(state, "ask")
        if self.over_limit(state, "ask"):
            return "No more questions. Give the label of your diagnosis now."
        store: EvidenceStore = state["evidence"]
        found = store.reveal("ask", action.query_text, limit=3)
        answer = found if found else "No, I do not have that."
        return answer + self.limit_note(state, "ask")

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        row = item["row"]
        gold = (row.get("prognosis") or "").strip()
        if not gold:
            return None

        if self._mode == "inquiry_rounds":
            rounds = item["rounds"]
            vignette = C.normalize_whitespace(row.get("50% Case"))
            transcript = []
            for round_index in range(1, rounds + 1):
                question = C.normalize_whitespace(row.get(f"Question_{round_index}"))
                answer = C.normalize_whitespace(row.get(f"Answer_{round_index}"))
                if question and answer:
                    transcript.append(f"Doctor: {question}\nPatient: {answer}")
            context = "\n".join(transcript)
            condition = f"{rounds}_rounds"
            sample_id = C.stable_id("medqdx", item["case_index"], condition)
        else:
            level = item["level"]
            vignette = C.normalize_whitespace(row.get(level))
            context = ""
            condition = level.replace("% Case", "pct").replace(" ", "")
            sample_id = C.stable_id("medqdx", item["case_index"], condition)

        if not vignette:
            return None
        options = self._options_for(gold, salt=sample_id)
        labels = C.letter_labels(len(options))
        return SampleSpec(
            sample_id=sample_id,
            fields={
                "observation": vignette,
                "context": context,
                "question": "Which diagnosis best explains this presentation?",
                "options": options,
                "option_labels": labels,
            },
            reference={"gold_label": labels[options.index(gold)], "gold": gold},
            task_kind="selection",
            # A single diagnosis label; the budget covers differential reasoning.
            max_tokens=512,
            metadata={
                "condition": condition,
                "case_index": item["case_index"],
                "mode": self._mode,
                "gold": gold,
                # The patient, for the interactive form: the case's own symptom
                # list, which is what its vignettes were written from.
                "_symptoms": [
                    part.strip()
                    for part in str(row.get("symptoms") or "").split(",")
                    if part.strip()
                ],
            },
        )

    def score(
        self,
        sample: SampleSpec,
        response: ModelResponse,
        *,
        output_contract: dict[str, Any] | None = None,
    ) -> SampleScore:
        score = selection_score(
            response,
            labels=sample.fields["option_labels"],
            gold_label=sample.reference["gold_label"],
            output_contract=output_contract,
            metric_name="accuracy",
        )
        condition = sample.metadata.get("condition", "unknown")
        score.metrics[f"accuracy_{condition}"] = score.metrics.get("accuracy", 0.0)
        return score

    def aggregate(self, scores: Sequence[SampleScore]) -> dict[str, float]:
        metrics = aggregate_mean_metrics([score.metrics for score in scores])
        # Difficulty gradient: how much does information completeness matter?
        full, half = metrics.get("accuracy_100pct"), metrics.get("accuracy_50pct")
        if full is not None and half is not None:
            metrics["information_sensitivity"] = full - half
        return metrics

    def documentation(self) -> AdapterDocumentation:
        mode = getattr(self, "_mode", str(self.context.option("mode", "information_levels")))
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="MedQDx",
            domain="Healthcare: Interactive Diagnosis",
            source_url=REPO_URL,
            processing_mode="Selection",
            split_used=self.split_used,
            abductive_subset=(
                "The diagnostic step: infer the disease that best explains an incomplete "
                "symptom picture. MedQDx's multi-turn question-asking loop is an interactive "
                "protocol that a single-turn harness cannot reproduce, so it is represented "
                "either as varying information completeness (default) or by appending the "
                "recorded Q&A rounds as context (options.mode = inquiry_rounds)."
            ),
            sampling_procedure=(
                f"all case x condition combinations are enumerated ({self.split_size} items) and "
                f"then drawn by {self.sampling_note()}"
            ),
            metrics_description={
                "self_consistency_<metric>":
                "Every metric also gets a self_consistency_ counterpart: the plurality answer over "
                "modes.repeats samples of the same record, read off those samples rather than bought "
                "again. Available because this dataset's answers are checkable and so can coincide.",
                "accuracy": "(PRIMARY, higher is better) 1 if the selected diagnosis is the gold prognosis",
                "accuracy_100pct/_80pct/_50pct": "accuracy at each information-completeness level",
                "accuracy_<n>_rounds": "accuracy with n recorded inquiry rounds (inquiry_rounds mode)",
                "information_sensitivity": "accuracy at 100% information minus accuracy at 50% -- "
                "how much the model depends on a complete picture",
            },
            primary_metric="accuracy",
            decisions=[
                "Rendered the task as closed-set selection (the processing mode the suite asks "
                "for) using the dataset's own 29-disease label space; distractors are drawn with "
                "the run seed, so every model sees identical options.",
                f"n_options={self.context.option('n_options', 5)} (gold + 4 distractors): enough "
                "to be non-trivial while keeping the prompt short. Configurable.",
                "Chose the 100 patient vignettes x 3 completeness levels as the default, which "
                "yields exactly 300 samples and measures degradation as evidence is withheld.",
                "The recorded Q&A in the benchmark CSV was produced by a specific model and is "
                "often uninformative ('No, I have not noticed that'), so it is not the default.",
            ],
            caveats=[
                "The vignettes are LLM-generated from a symptom-disease table, not real clinical "
                "notes; scores reflect textbook symptom-to-disease mapping.",
                "Only 100 distinct cases exist, so the 300 samples are 3 views of 100 cases -- "
                "items are not statistically independent.",
            ],
            statistics={
                **self.base_statistics(),
                "mode": mode,
                "label_space": len(getattr(self, "_label_space", [])),
                "n_options": int(self.context.option("n_options", 5)),
            },
        )
