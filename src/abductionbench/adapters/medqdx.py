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
from ._base import PooledDatasetAdapter, apply_judged_metric, judged_only_score
from ._interactive import EvidenceStore, InteractiveMixin, parse_action

REPO_URL = "https://github.com/MaiWert/MedQDx"
LEVELS = ("100% Case", "80% Case", "50% Case")


class MedQDxAdapter(InteractiveMixin, PooledDatasetAdapter):
    """Closed-set diagnosis from vignettes of varying completeness."""

    adapter_version = "1.0"

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. You are given a clinical vignette "
        "that may be incomplete. Name the diagnosis best supported by the information "
        "actually present."
    )
    data_delivery_mode = "interactive"
    #: No candidate list is shown -- the release asks for an open diagnosis
    #: ("Output ONLY the name of the disease or condition using correct medical
    #: term"), so the answer is written into an open vocabulary and a correct
    #: answer routinely differs from the gold in wording. The judge scores it.
    objective_metrics = False
    selection_cardinality = None
    primary_metric = "diagnosis_judged"

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

    # ------------------------------------------------------------------ #
    # the questioning loop -- MedQDx's whole premise
    # ------------------------------------------------------------------ #

    ACTIONS = ("ask", "diagnosis")
    max_turns = 12
    category_limits = {"ask": 8}

    #: MedQDx's own prompts, verbatim from the release's benchmark-creation
    #: notebook (``Benchmark Creation/MedQDx_Benchmark_Creation.ipynb``). The
    #: benchmark is a diagnostic interview and these three strings are the
    #: protocol: one question a turn, then a single diagnosis. They are quoted
    #: rather than restated because this is an interactive benchmark -- the
    #: wording is part of what is measured.
    AUTHORS_SYSTEM = "You are a medical doctor."

    AUTHORS_FIRST_QUESTION = (
        "You are a medical doctor conducting a diagnostic interview with a patient.\n"
        "You have received partial information about the case.\n"
        "Your goal is to ask exactly **ONE** relevant, case-specific question that will "
        "enable a more precise diagnosis.\n\n"
        "Rules:\n"
        "- Output exactly one question as a single, complete sentence.\n"
        "- The question must be a single line, ending in \u2018?\u2019\n"
        "- The question must be specific to the case.\n"
        "- Do not include any explanations, reasoning, or additional text.\n"
        "- Do not provide a diagnosis or suggest treatments in the question.\n"
        "- Base the question solely on the provided partial case information.\n\n"
        "Partial case Information:\n{case_text}\n\n"
        "### User: What is the first best diagnostic question you want to ask the patient "
        "(one question)?\n### Assistant:"
    )

    AUTHORS_NEXT_QUESTION = (
        "You are a medical doctor conducting a diagnostic interview with a patient.\n"
        "You have received partial information about the case and the past conversation "
        "with the patient.\n"
        "Your goal is to ask exactly **ONE** new, relevant, case-specific question that "
        "will enable a more precise diagnosis.\n"
        "Ensure that this new question is **word-for-word different** from all previous "
        "questions.\n\n"
        "Rules:\n"
        "- Output **ONLY ONE** question as a single, complete sentence ending with a "
        "question mark.\n"
        "- Do NOT include any explanations, reasoning, or additional text\u2014only the "
        "question itself.\n"
        "- The question must be a single line, ending in \u2018?\u2019\n"
        "- Do NOT provide a diagnosis or suggest treatments.\n"
        "- Base the question on the partial case information and past conversation with "
        "the patient.\n"
        "- ask new question to obtain additional information for better diasnosis.\n"
        "- If the patient responded \"I'm not sure,\" ask a broader or differently phrased "
        "question to elicit new information.\n\n"
        "Partial case Information::\n{case_text}\n\n"
        "Past Conversation with the patient:\n{history}\n\n"
        "{prev_section}\n\n"
        "### User:Next, output one NEW question you would ask the patient:\n### Assistant:"
    )

    AUTHORS_DIAGNOSIS = (
        "***You are a medical doctor***. Your task is to provide a single most likely "
        "diagnosis based on the partial case information and the past conversation with "
        "the patient.\n\n"
        "Rules:\n"
        "- Analyze the case details and patient responses.\n"
        "- Use clinical reasoning to determine the most probable diagnosis.\n"
        "- Output ONLY the name of the disease or condition using correct medical term "
        "(e.g., Pneumonia, Hypoglycemia).\n"
        "- Do not include any notes, explanations, disclaimers, or additional text.\n"
        "- Do not output symbols like ### or other placeholders.\n"
        "- Do not repeat on the case symptoms\n"
        "- Do not repeat on the patient answers\n\n"
        "Case Information:\n{case_text}\n\n"
        "Conversation History:\n{history}\n\n"
        "### User: The patient diagnosis is:\n### Assistant:"
    )

    def _history_text(self, state: dict[str, Any]) -> str:
        turns = state.get("history") or []
        if not turns:
            return "(none yet)"
        return "\n".join(f"Q: {q}\nA: {a}" for q, a in turns)

    def interactive_start(self, sample: SampleSpec) -> tuple[list[ChatMessage], dict[str, Any]]:
        store = EvidenceStore()
        # The case's symptom list is the patient: a symptom in the list is
        # present, and anything else is absent. MedQDx's cases are built from
        # exactly that list, so nothing has to be invented either way.
        present = sample.metadata.get("_symptoms") or []
        store.categories["ask"] = {
            symptom.replace("_", " "): "yes, that is present" for symptom in present
        }
        case_text = sample.fields.get("observation", "")
        state = {"evidence": store, "counts": {}, "history": [], "case_text": case_text}
        return (
            [
                ChatMessage(role="system", content=self.AUTHORS_SYSTEM),
                ChatMessage(
                    role="user",
                    content=self.AUTHORS_FIRST_QUESTION.format(case_text=case_text),
                ),
            ],
            state,
        )

    def interactive_step(
        self, sample: SampleSpec, state: dict[str, Any], assistant_text: str
    ) -> str | None:
        # MedQDx's protocol has no action vocabulary: the model is asked for a
        # question in plain language, and the patient answers it. The turn ends
        # when the question budget is spent, and the release then asks for the
        # diagnosis with its own prompt.
        question = (assistant_text or "").strip().split("\n")[-1].strip()
        if state.get("phase") == "diagnose":
            return None

        self.bump(state, "ask")
        store: EvidenceStore = state["evidence"]
        found = store.reveal("ask", question, limit=3)
        # The release's patient answers from the case, and says so plainly when
        # the symptom is not recorded -- "I'm not sure" is the wording its
        # next-question prompt is written to handle.
        answer = found if found else "I'm not sure."
        state["history"].append((question, answer))

        if self.over_limit(state, "ask"):
            state["phase"] = "diagnose"
            return self.AUTHORS_DIAGNOSIS.format(
                case_text=state["case_text"], history=self._history_text(state)
            )
        return self.AUTHORS_NEXT_QUESTION.format(
            case_text=state["case_text"],
            history=self._history_text(state),
            prev_section="",
        )

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
        return SampleSpec(
            sample_id=sample_id,
            fields={
                "observation": vignette,
                "context": context,
                "question": "Which diagnosis best explains this presentation?",
            },
            reference={"gold": gold},
            task_kind="generation",
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
        # The release asks for the disease NAME ("Output ONLY the name of the
        # disease or condition using correct medical term"), not a label from a
        # list -- it never shows the model a candidate list at all. Scoring a
        # label here would score a different, easier task than MedQDx poses.
        condition = sample.metadata.get("condition", "unknown")
        # The stratum is seeded here and filled by the judge along with the base
        # metric, so each information level is the same verdict seen through a filter.
        return judged_only_score(
            response,
            metric="diagnosis_judged",
            output_contract=output_contract,
            extra_metrics={f"diagnosis_judged_{condition}": 0.0},
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
            "observation": C.clip_words(sample.fields["observation"], 200),
            "criteria": (
                "The candidate is correct if it names the same disease entity as the "
                "reference, however it is written: synonyms, abbreviations, eponyms and "
                "spelling variants all count. A broader category that does not identify "
                "the reference disease, or a different disease that shares symptoms with "
                "it, does not count."
            ),
        }

    def apply_judge(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore, verdict: Any
    ) -> SampleScore:
        return apply_judged_metric(score, verdict, "diagnosis_judged")

    def aggregate(self, scores: Sequence[SampleScore]) -> dict[str, float]:
        metrics = aggregate_mean_metrics([score.metrics for score in scores])
        # Difficulty gradient: how much does information completeness matter?
        full = metrics.get("diagnosis_judged_100pct")
        half = metrics.get("diagnosis_judged_50pct")
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
            processing_mode="Generation",
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
                "best_of_n_<metric>":
                "Every metric also gets a best_of_n_ counterpart: per record, the repeat the "
                "judge scored highest. A diagnosis is free text with many correct surface forms, "
                "so a plurality over repeats is not meaningful and Best-of-N replaces it.",
                "diagnosis_judged": "(PRIMARY, higher is better, 0-1) LLM-judge verdict on whether "
                "the named diagnosis is the same disease entity as the case's prognosis, however "
                "written. 1.0 when the judge affirms.",
                "diagnosis_judged_100pct/_80pct/_50pct": "the same metric at each "
                "information-completeness level",
                "diagnosis_judged_<n>_rounds": "the same metric with n recorded inquiry rounds "
                "(inquiry_rounds mode)",
                "information_sensitivity": "score at 100% information minus score at 50% -- "
                "how much the model depends on a complete picture",
                "parse_failure_rate": "(lower is better, 0-1) fraction of responses no diagnosis "
                "could be read from; these score 0 and are counted separately from being wrong.",
            },
            primary_metric="diagnosis_judged",
            decisions=[
                "Kept the release's own open-vocabulary task -- MedQDx tells the model to "
                "output only the name of the disease and never shows it a candidate list. An "
                "earlier version of this adapter built a 5-way choice from the 29-disease label "
                "space; that is an easier task than the benchmark poses, and it is gone.",
                "Scored by an LLM judge against the case prognosis rather than by string "
                "comparison, which is what an open vocabulary requires.",
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
            },
        )
