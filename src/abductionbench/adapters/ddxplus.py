"""DDXPlus: differential diagnosis from a symptom questionnaire.

Source: https://figshare.com/articles/dataset/DDXPlus_Dataset/20043374

The release ships patient files (train/validate/test) plus two dictionaries:
``release_evidences.json`` (223 questions, their answer vocabularies and whether
they are antecedents) and ``release_conditions.json`` (49 conditions with
English names).  A patient row carries age, sex, the coded ``EVIDENCES`` list,
the ground-truth ``PATHOLOGY`` and a ``DIFFERENTIAL_DIAGNOSIS`` (a ranked list
of candidate conditions with probabilities).

The adapter decodes the coded evidences into readable question/answer pairs via
the evidence dictionary -- otherwise the prompt would be strings like
``E_55_@_V_89`` -- and evaluates two modes:

* ``generation`` (default) -- name the pathology that explains the answers;
* ``selection`` -- choose among the case's own differential candidates.

Both use English surface forms (``question_en``, ``cond-name-eng``); the dataset
is bilingual and the French fields are ignored, which is recorded below.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..core.adapter import SkippedDataset
from ..core.metrics import aggregate_mean_metrics, extract_answer_span
from ..core.types import (
    AdapterDocumentation,
    ChatMessage,
    ModelResponse,
    SampleScore,
    SampleSpec,
)
from . import _common as C
from ._base import PooledDatasetAdapter, selection_score, text_match_score
from ._interactive import EvidenceStore, InteractiveMixin, parse_action

FIGSHARE = "https://ndownloader.figshare.com/files"
FILES = {
    "evidences": f"{FIGSHARE}/40495562",
    "conditions": f"{FIGSHARE}/62657140",
    "test": f"{FIGSHARE}/40495565",
    "validate": f"{FIGSHARE}/40495571",
}
SOURCE_URL = "https://figshare.com/articles/dataset/DDXPlus_Dataset/20043374"


class DDXPlusAdapter(InteractiveMixin, PooledDatasetAdapter):
    """Diagnose a DDXPlus patient from decoded questionnaire answers."""

    adapter_version = "1.0"

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. You are given a patient's answers "
        "to a diagnostic questionnaire, including their age, sex and reported symptoms. State "
        "the diagnosis that explains the whole picture, not merely a condition consistent "
        "with one symptom."
    )
    data_delivery_mode = "interactive"
    objective_metrics = True
    selection_cardinality = "single"
    hypothesis_modes = ("selection",)
    hypothesis_mode_options = {
        "selection": {'subtask': 'selection'},
    }
    table_hypothesis_mode = "Selection"
    hypothesis_mode_justification = (
        "DDXPlus is a SELECTION benchmark and is run as one only. Each patient ships a "
        "DIFFERENTIAL_DIAGNOSIS list, and that list is the candidate set -- the distractors are "
        "the clinician-plausible conditions the case itself raises. Recovering the ground-truth "
        "PATHOLOGY without candidates is a different task than the benchmark poses, so it is "
        "not run."
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
        return str(self.context.option("subtask", "selection"))

    def load_items(self) -> list[dict[str, Any]]:
        data_dir = self.context.data_dir
        evidences_path = C.ensure_download(
            FILES["evidences"], data_dir / "release_evidences.json", offline=self.context.offline
        )
        conditions_path = C.ensure_download(
            FILES["conditions"], data_dir / "release_conditions.json", offline=self.context.offline
        )
        self._evidences = C.read_json(evidences_path)
        self._conditions = C.read_json(conditions_path)

        # Test split preferred; validate is the fallback if the test archive is
        # unavailable (it is the larger download).
        rows: list[dict[str, Any]] = []
        split_name = ""
        for candidate in ("test", "validate"):
            try:
                archive = C.ensure_download(
                    FILES[candidate],
                    data_dir / f"release_{candidate}_patients.zip",
                    offline=self.context.offline,
                )
                extracted = C.extract_archive(archive, data_dir / candidate)
            except SkippedDataset as exc:
                self.log.warning("DDXPlus %s split unavailable: %s", candidate, exc)
                continue
            # The archives contain a single extension-less CSV file
            # ("release_test_patients"), so match on any file and read it as CSV.
            candidates = C.find_files(extracted, ["*.csv", "release_*"])
            if not candidates:
                continue
            rows = C.read_csv_rows(candidates[0], delimiter=",")
            split_name = candidate
            break
        if not rows:
            raise SkippedDataset("no DDXPlus patient file could be downloaded or read")
        self.split_used = f"{split_name} ({len(rows)} patients) from the figshare release"
        return rows

    # -- evidence decoding ---------------------------------------------- #

    def _decode(self, token: str) -> str | None:
        """Turn ``E_55_@_V_89`` into 'question -> answer' in English."""
        token = token.strip().strip("'\"")
        if not token:
            return None
        code, _, value = token.partition("_@_")
        entry = self._evidences.get(code)
        if not entry:
            return None
        question = C.normalize_whitespace(entry.get("question_en") or entry.get("name"))
        if not value:
            return f"{question} -> yes"
        meaning = (entry.get("value_meaning") or {}).get(value)
        if isinstance(meaning, dict):
            answer = C.normalize_whitespace(meaning.get("en") or meaning.get("fr") or value)
        else:
            answer = value.replace("V_", "")
        return f"{question} -> {answer}"

    def _condition_name(self, raw: str) -> str:
        entry = self._conditions.get(raw) or {}
        return C.normalize_whitespace(entry.get("cond-name-eng") or raw)

    # ------------------------------------------------------------------ #
    # the consultation -- DDXPlus is a questionnaire, so the model asks it
    # ------------------------------------------------------------------ #

    ACTIONS = ("ask", "diagnosis")
    #: DDXPlus dialogues in the release average well under 20 questions; the
    #: limit exists so a model that never commits still terminates.
    max_turns = 20
    category_limits = {"ask": 15}

    INTERVIEW_PROMPT = (
        "You are a physician taking a history from a patient. You are given the patient's "
        "age, sex and presenting complaint, and you may ask about any symptom or antecedent "
        "one at a time. The patient answers only what you ask.\n\n"
        "Reply with a single JSON object and nothing else:\n"
        '{"reasoning": "...", "action": "ask", "query": "your question to the patient"}\n'
        "or, once the picture is clear:\n"
        '{"reasoning": "...", "action": "diagnosis", "query": "<the number of your '
        'chosen diagnosis>"}\n\n'
        "Ask about what would discriminate between the diagnoses you are considering, not "
        "about what you already know. Commit as soon as the evidence supports one diagnosis."
    )

    def _question_catalogue(self) -> dict[str, str]:
        """``code -> English question text`` for every evidence in the release."""
        if getattr(self, "_catalogue", None) is None:
            self._catalogue = {
                code: C.normalize_whitespace(entry.get("question_en") or entry.get("name") or "")
                for code, entry in (self._evidences or {}).items()
            }
        return self._catalogue

    def interactive_start(self, sample: SampleSpec) -> tuple[list[ChatMessage], dict[str, Any]]:
        meta = sample.metadata
        # DDXPlus records the presenting complaint as the questionnaire item the
        # patient first answered, e.g. "Do you have a cough? -> yes". It is
        # quoted as that question-and-answer pair rather than reworded into a
        # complaint, so the opening states exactly what the record says.
        initial = str(meta.get("initial") or "")
        question, _, answer = initial.partition(" -> ")
        complaint = f'"{question}" - {answer}' if answer else (question or "unspecified")
        # The candidates are the patient's OWN differential, which is the task
        # DDXPlus poses. They have to be shown: the scorer matches the label
        # chosen, and until this was fixed the prompt asked for a disease name
        # while the scorer looked for a label -- so every single episode ended
        # in a parse failure however good the diagnosis was.
        options = sample.fields.get("options") or []
        labels = sample.fields.get("option_labels") or []
        listing = "\n".join(
            f"{label}. {option}" for label, option in zip(labels, options, strict=False)
        )
        opening = [
            f"Patient: {meta.get('age')}-year-old, sex {meta.get('sex')}.",
            f"Presenting complaint, as the patient first reported it: {complaint}",
            "",
            "Candidate diagnoses:",
            listing,
            "",
            "Take a history, then give the number of your diagnosis.",
        ]
        # The patient's answers: every evidence the record holds, keyed by the
        # question the release asks for it.
        store = EvidenceStore()
        store.categories["ask"] = dict(meta.get("_answers") or {})
        state = {"evidence": store, "counts": {}, "final": None}
        return (
            [
                ChatMessage(role="system", content=self.INTERVIEW_PROMPT),
                ChatMessage(role="user", content="\n".join(opening)),
            ],
            state,
        )

    def interactive_step(
        self, sample: SampleSpec, state: dict[str, Any], assistant_text: str
    ) -> str | None:
        action = parse_action(assistant_text, actions=self.ACTIONS)
        if action.action == "diagnosis":
            state["final"] = action.query_text
            return None
        if not action.action:
            state["parse_errors"] = state.get("parse_errors", 0) + 1
            if state["parse_errors"] > 2:
                return None
            return (
                'Reply with one JSON object: {"reasoning": "...", "action": "ask"|"diagnosis", '
                '"query": "..."}'
            )

        self.bump(state, "ask")
        if self.over_limit(state, "ask"):
            return "That is all the history available. Please give your diagnosis now."
        store: EvidenceStore = state["evidence"]
        answer = store.reveal("ask", action.query_text, limit=3)
        if not answer:
            # The patient record has nothing matching. DDXPlus records a
            # patient's *positive* evidences plus the questionnaire's defaults,
            # so an unmatched question is genuinely a negative answer.
            answer = "No, nothing like that."
        return answer + self.limit_note(state, "ask")

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        pathology = C.normalize_whitespace(item.get("PATHOLOGY"))
        if not pathology:
            return None
        gold = self._condition_name(pathology)
        raw_evidences = item.get("EVIDENCES") or ""
        tokens = [
            token
            for token in str(raw_evidences).strip("[]").replace("'", "").split(",")
            if token.strip()
        ]
        decoded = [line for line in (self._decode(token) for token in tokens) if line]
        if not decoded:
            return None
        initial = self._decode(str(item.get("INITIAL_EVIDENCE") or ""))
        # The patient's answer sheet for the interactive consultation: the
        # release's own question text mapped to what this patient answered.
        answers = {
            line.split(" -> ", 1)[0]: line.split(" -> ", 1)[1]
            for line in decoded
            if " -> " in line
        }
        age, sex = item.get("AGE"), item.get("SEX")
        antecedents = [line for line in decoded if "have you" in line.lower() or "did you" in line.lower()]
        observation = "\n".join(
            part
            for part in (
                f"Patient: {age}-year-old, sex {sex}." if age else "",
                f"Presenting complaint: {initial}" if initial else "",
                "Questionnaire answers:\n" + "\n".join(f"- {line}" for line in decoded),
            )
            if part
        )

        if self._subtask == "selection":
            differential = _parse_differential(item.get("DIFFERENTIAL_DIAGNOSIS"))
            candidates = [self._condition_name(name) for name, _ in differential]
            candidates = [name for name in candidates if name]
            if gold not in candidates:
                candidates.insert(0, gold)
            limit = int(self.context.option("n_options", 6))
            options = sorted(set(candidates[:limit] + [gold]))
            if len(options) < 3:
                return None
            labels = C.letter_labels(len(options))
            return SampleSpec(
                sample_id=C.stable_id("ddxplus", index),
                fields={
                    "observation": observation,
                    "question": "Which condition best explains these findings?",
                    "options": options,
                    "option_labels": labels,
                },
                reference={"gold_label": labels[options.index(gold)], "gold": gold},
                task_kind="selection",
                max_tokens=768,
                metadata={
                    "pathology": pathology,
                    "n_evidences": len(decoded),
                "age": age,
                "sex": sex,
                "initial": initial,
                "_answers": answers,
                },
            )

        return SampleSpec(
            sample_id=C.stable_id("ddxplus", index),
            fields={
                "observation": observation,
                "question": "What is the most likely diagnosis?",
                "instructions": "Name the single most likely condition.",
            },
            reference={"gold": gold},
            task_kind="generation",
            # A condition name after a questionnaire; budget covers a differential.
            max_tokens=768,
            metadata={
                "pathology": pathology,
                "n_evidences": len(decoded),
                "n_antecedents": len(antecedents),
                "age": age,
                "sex": sex,
                "initial": initial,
                "_answers": answers,
            },
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

    def aggregate(self, scores: Sequence[SampleScore]) -> dict[str, float]:
        return aggregate_mean_metrics([score.metrics for score in scores])

    def judge_request(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore
    ) -> dict[str, Any] | None:
        if sample.task_kind != "generation" or not response.text:
            return None
        return {
            "candidate": extract_answer_span(response.text, None)[:400],
            "gold": sample.reference["gold"],
            "criteria": "Equivalent condition names (synonyms, abbreviations) count as correct.",
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
            name="DDXPlus",
            domain="Healthcare: Differential Diagnosis",
            source_url=SOURCE_URL,
            processing_mode="Generation (default) / Selection",
            split_used=self.split_used,
            abductive_subset=(
                "The diagnosis itself: infer the condition that explains the questionnaire "
                "answers. DDXPlus's simulated interview (which question to ask next) is a policy "
                "problem, not abduction, and is not evaluated."
            ),
            sampling_procedure=self.sampling_note(),
            metrics_description={
                "self_consistency_<metric>":
                "Every metric also gets a self_consistency_ counterpart: the plurality answer over "
                "modes.repeats samples of the same record, read off those samples rather than bought "
                "again. Available because this dataset's answers are checkable and so can coincide.",
                "diagnosis_match": "(PRIMARY, higher is better) 1 if the answer names the gold condition (primary)",
                "exact_match": "strict normalized equality with the gold condition name",
                "token_f1": "bag-of-tokens F1 against the gold condition name",
                "rouge_l": "LCS F-measure against the gold condition name",
                "accuracy": "(PRIMARY, higher is better) selection subtask: 1 if the chosen candidate is the gold condition",
                "diagnosis_match_judged": "LLM-judge equivalence verdict (only when "
                "engine.judge.enabled)",
            },
            primary_metric="accuracy" if self._subtask == "selection" else "diagnosis_match",
            decisions=[
                "Decoded the coded evidences through release_evidences.json into English "
                "question/answer lines; presenting raw codes would test cipher-breaking, not "
                "diagnosis.",
                "Used English surface forms only (question_en, cond-name-eng); the dataset is "
                "bilingual French/English and the French fields are ignored.",
                "Used the official test split, falling back to validate only if the test archive "
                "cannot be downloaded (this is reported in split_used).",
                "The selection subtask draws its options from the patient's own "
                "DIFFERENTIAL_DIAGNOSIS list, so distractors are clinically plausible and are "
                "never invented.",
            ],
            caveats=[
                "NO AGENT PROMPT EXISTS TO ADOPT. DDXPlus is a dataset, not an agent "
                "harness: it ships patients, conditions and an evidence bank, and its "
                "published baselines are supervised models rather than a prompted LLM. "
                "What the release does publish is used -- the interview questions are its "
                "own `question_en` strings and the answers are what the patient record "
                "says -- but the system prompt that frames the interview is necessarily "
                "written here, and is therefore NOT the authors'.",
                "Patients are synthesized from a medical knowledge base, so findings are "
                "internally consistent in a way real cases are not.",
                "Answers are a fixed questionnaire, so a diagnosis is often strongly determined; "
                "high accuracy here does not transfer to open-ended clinical abduction.",
            ],
            statistics={**self.base_statistics(), "subtask": self._subtask},
        )


def _parse_differential(raw: Any) -> list[tuple[str, float]]:
    """Parse the ``[[name, prob], ...]`` differential column."""
    if isinstance(raw, list):
        entries = raw
    else:
        text = str(raw or "").strip()
        if not text:
            return []
        import ast

        try:
            entries = ast.literal_eval(text)
        except Exception:  # noqa: BLE001 - malformed cell
            return []
    out: list[tuple[str, float]] = []
    for entry in entries if isinstance(entries, list) else []:
        if isinstance(entry, (list, tuple)) and entry:
            name = str(entry[0])
            try:
                probability = float(entry[1]) if len(entry) > 1 else 0.0
            except (TypeError, ValueError):
                probability = 0.0
            out.append((name, probability))
    out.sort(key=lambda pair: -pair[1])
    return out
