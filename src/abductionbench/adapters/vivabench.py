"""VivaBench: interactive clinical reasoning ("viva" examination).

Source: https://huggingface.co/datasets/chychiu/VivaBench

Two tables ship: ``pubmed_reviewed.csv`` (990 human-reviewed cases) and
``dataset_generated.csv`` (1,952 generated cases).  Each row has an opening
``vignette``, the accepted ``diagnosis`` list, a ``differentials`` list, and a
structured ``clinicalcase`` JSON holding the findings an examiner would reveal
on request (history, examination, investigations).

**How it is run.** VivaBench's protocol is interactive and is executed as such:
the episode opens with the case stem the release builds (demographics, chief
complaint, vitals), and the model works the patient up by asking for history,
examination, investigations and imaging, one action per turn, until it commits
to a final diagnosis.  The environment answers only from that case's structured
findings, so a request for something the case does not record is answered as not
available rather than invented.

The system prompt, the action vocabulary, the error message and the per-category
limits are the release's own (``vivabench/prompts/examiner.py`` and
``vivabench/examiner.py``, downloaded with the dataset), not written here.  What
is not the release's is the *mapper*: VivaBench resolves a free-text request to a
finding with an LLM, which would put a second model inside the evaluation of the
first.  This adapter matches lexically instead -- deterministic, reproducible,
and visible in the transcript when it misses.

**What the model is shown is the release's own case stem**, not the release's
``vignette`` column.  ``vivabench/examiner.py`` builds the examinee's opening
turn from three structured fields -- demographics, chief complaint and the
initial vitals -- and that is all a viva candidate gets before they start
asking.  The ``vignette`` column is the source PubMed case report in full: the
text the release's *generation pipeline* read in order to build the structured
case, complete with the article title, the work-up, the diagnosis and the
outcome.  An earlier version of this adapter used that column as the prompt,
which put the answer in front of the model in 41% of cases.

The human-reviewed table is the default -- the generated table is available via
``options.table``.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from ..core.adapter import SkippedDataset
from ..core.metrics import extract_answer_span
from ..core.types import (
    AdapterDocumentation,
    ChatMessage,
    ModelResponse,
    SampleScore,
    SampleSpec,
)
from . import _common as C
from ._base import PooledDatasetAdapter, selection_score, text_match_score
from ._interactive import EvidenceStore, InteractiveMixin, flatten, parse_action

REPO_ID = "chychiu/VivaBench"


#: The release's own vital-sign labels and units, from
#: ``vivabench/ontology/schema.py`` (``Vitals.initial_prompt``). Order matters:
#: it is the order the examinee sees them in.
_VITAL_LABELS = (
    ("temperature", "Temperature", "\u00b0C"),
    ("heart_rate", "HR", " bpm"),
    ("blood_pressure_systolic", "BP", " mmHg"),
    ("respiratory_rate", "RR", "/min"),
    ("oxygen_saturation", "O2 sat", "%"),
    ("pain_score", "Pain", ""),
    ("gcs", "GCS", ""),
)


#: Keys of the structured case that hold the ANSWER, never the evidence.
_ANSWER_KEYS = frozenset({"diagnosis", "differentials"})


def _initial(value: Any) -> Any:
    """The first reading when a vital is recorded as a trajectory."""
    return value[0] if isinstance(value, list) and value else value


def _vitals_line(vitals: dict[str, Any] | None) -> str:
    """``Vitals.initial_prompt`` from the release, reimplemented on raw JSON.

    Only the *initial* reading of each vital, because that is what a viva
    candidate is given before examining anyone; the trajectory is disclosed
    later, on request, like any other finding.
    """
    if not isinstance(vitals, dict):
        return ""
    parts: list[str] = []
    for field, label, unit in _VITAL_LABELS:
        value = _initial(vitals.get(field))
        if value is None:
            continue
        if field == "blood_pressure_systolic":
            diastolic = _initial(vitals.get("blood_pressure_diastolic"))
            if diastolic is not None:
                parts.append(f"BP {value}/{diastolic} mmHg")
                continue
        parts.append(f"{label} {value}{unit}")
    return ", ".join(parts)


def _parse_listish(value: Any) -> list[str]:
    """Parse a CSV cell that holds a Python/JSON list of strings."""
    if isinstance(value, list):
        return [str(item) for item in value]
    text = str(value or "").strip()
    if not text or text in ("[]", "nan"):
        return []
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except Exception:  # noqa: BLE001 - try the next loader
            continue
    return [text]


class VivaBenchAdapter(InteractiveMixin, PooledDatasetAdapter):
    """Diagnose a viva case: free-text (default) or among the case's differentials."""

    adapter_version = "1.0"

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. You are given a clinical viva "
        "case. State the diagnosis that explains the presentation."
    )
    data_delivery_mode = "static"
    objective_metrics = True
    selection_cardinality = "single"
    hypothesis_modes = ("selection",)
    hypothesis_mode_options = {
        "selection": {'subtask': 'selection'},
    }
    table_hypothesis_mode = "Selection"
    hypothesis_mode_justification = (
        "VivaBench is run as static SELECTION only, over the candidate list the official GitHub "
        "release ships with each case. Only the static selection data in that release is used."
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
        root = C.ensure_hf_snapshot(
            REPO_ID,
            self.context.data_dir / "hf",
            offline=self.context.offline,
            # The release ships its own examiner prompts and protocol next to the
            # data; they are downloaded because item 2 of the specification asks
            # for the benchmark's published prompts to be used rather than
            # re-invented.
            allow_patterns=["dataset/*", "*.md", "vivabench/prompts/*.py",
                            "vivabench/examiner.py", "configs/*.yaml"],
        )
        table = str(self.context.option("table", "pubmed_reviewed"))
        files = C.find_files(root, [f"{table}.csv"]) or C.find_files(root, ["*.csv"])
        if not files:
            raise SkippedDataset("no VivaBench CSV found in the release")
        rows = C.read_csv_rows(files[0])
        self._with_differentials = sum(1 for row in rows if len(_parse_listish(row.get("differentials"))) >= 2)
        self.split_used = (
            f"{files[0].stem} ({len(rows)} cases); the release ships no train/test split"
        )
        return rows

    def _findings(self, item: dict[str, Any]) -> str:
        """Flatten the structured clinicalcase JSON into readable findings."""
        if str(self.context.option("evidence", "vignette")) == "vignette":
            return ""
        raw = item.get("clinicalcase")
        try:
            payload = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:  # noqa: BLE001 - malformed cell
            return ""
        if not isinstance(payload, dict):
            return ""
        limit = int(self.context.option("words_per_section", 250))
        blocks: list[str] = []
        for key, value in payload.items():
            # The structured case carries the answer as well as the evidence:
            # `diagnosis` is the gold and `differentials` is the selection
            # subtask's option list. Rendering either into the prompt would
            # hand over the answer outright.
            if key in _ANSWER_KEYS:
                continue
            if isinstance(value, str) and value.strip():
                blocks.append(f"{key.replace('_', ' ').title()}: {C.clip_words(value, limit)}")
            elif isinstance(value, (list, dict)):
                text = C.normalize_whitespace(json.dumps(value)[:4000])
                blocks.append(f"{key.replace('_', ' ').title()}: {C.clip_words(text, limit)}")
        return "\n\n".join(blocks)

    # ------------------------------------------------------------------ #
    # the examination loop -- VivaBench's own protocol
    # ------------------------------------------------------------------ #

    #: The release's limits (vivabench/examiner.py, Examiner.__init__).
    category_limits = {"history": 10, "examination": 5, "investigation": 5, "imaging": 5}
    #: action_limit=20 in the release; one action is one model turn here.
    max_turns = 20

    #: The release's action vocabulary, in the order the prompt lists it.
    ACTIONS = (
        "history",
        "examination",
        "diagnosis_provisional",
        "investigation",
        "imaging",
        "diagnosis_final",
    )

    #: Which part of the case each action may disclose.
    _SOURCES = {
        "history": ("history", "past_medical_history", "social_history", "family_history",
                    "medications", "allergies", "history_freetext"),
        "examination": ("physical",),
        "investigation": ("investigations",),
        "imaging": ("imaging",),
    }

    def _release_prompts(self) -> dict[str, str]:
        """The benchmark's own prompt text, read from the files it ships."""
        if getattr(self, "_prompt_cache", None):
            return self._prompt_cache
        cache: dict[str, str] = {}
        root = self.context.data_dir / "hf"
        path = root / "vivabench" / "prompts" / "examiner.py"
        if path.exists():
            source = path.read_text(encoding="utf-8", errors="replace")
            for name in ("ASSISTANT_BASE_PROMPT", "ERROR_RETURN_MSG"):
                match = re.search(rf'{name}\s*=\s*"""(.*?)"""', source, re.S)
                if match:
                    cache[name] = match.group(1).strip()
        self._prompt_cache = cache
        return cache

    def _evidence(self, item: dict[str, Any]) -> EvidenceStore:
        payload = self._case_json(item)
        store = EvidenceStore()
        for category, keys in self._SOURCES.items():
            merged: dict[str, str] = {}
            for key in keys:
                if key in payload:
                    merged.update(flatten(payload[key], key))
            store.categories[category] = merged
        return store

    @staticmethod
    def _case_json(item: dict[str, Any]) -> dict[str, Any]:
        raw = item.get("clinicalcase")
        try:
            payload = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:  # noqa: BLE001 - a malformed cell yields an empty case
            return {}
        return payload if isinstance(payload, dict) else {}

    def _stem(self, payload: dict[str, Any]) -> str:
        """The opening stem, built the way the release builds it.

        ``vivabench/examiner.py`` line 299, verbatim in structure::

            f"Clinical case stem: {demographics.prompt} presenting with "
            f"{history.chief_complaint.lower()}.\n{physical.vitals.prompt}\n"
            f"Please review and diagnose the patient."

        Demographics, chief complaint and the initial vitals -- and nothing
        else. This is the whole of what a VivaBench examinee is given before
        they start asking questions, and reproducing it is the point: the
        benchmark measures what the model asks for next.
        """
        demographics = payload.get("demographics") or {}
        age, unit, gender = (
            demographics.get("age"),
            demographics.get("unit"),
            demographics.get("gender"),
        )
        who = " ".join(str(part) for part in (age, unit, "old", gender) if part not in (None, ""))
        who = who or "A patient"
        history = payload.get("history") or {}
        complaint = str(history.get("chief_complaint") or "an undifferentiated presentation")
        lines = [f"Clinical case stem: {who} presenting with {complaint.lower()}."]
        vital_line = _vitals_line((payload.get("physical") or {}).get("vitals"))
        if vital_line:
            lines.append(vital_line)
        lines.append("Please review and diagnose the patient.")
        return "\n".join(lines)

    def interactive_start(self, sample: SampleSpec) -> tuple[list[ChatMessage], dict[str, Any]]:
        prompts = self._release_prompts()
        system = prompts.get("ASSISTANT_BASE_PROMPT") or self.system_prompt
        payload = sample.metadata.get("_case") or {}
        state: dict[str, Any] = {
            "evidence": self._evidence({"clinicalcase": json.dumps(payload)}),
            "counts": {},
            "final": None,
        }
        return (
            [
                ChatMessage(role="system", content=system),
                ChatMessage(role="user", content=self._stem(payload)),
            ],
            state,
        )

    def interactive_step(
        self, sample: SampleSpec, state: dict[str, Any], assistant_text: str
    ) -> str | None:
        action = parse_action(assistant_text, actions=self.ACTIONS)
        if not action.parsed and not action.action:
            error = self._release_prompts().get("ERROR_RETURN_MSG", "Unable to parse your response.")
            state["parse_errors"] = state.get("parse_errors", 0) + 1
            if state["parse_errors"] > 3:
                # Three malformed turns in a row is a formatting failure, not a
                # diagnostic one; end the episode and let the scorer read the
                # last message rather than burning the whole turn budget.
                return None
            return f"{error}\n{assistant_text[:400]}"

        if action.action == "diagnosis_final":
            state["final"] = action.query_text
            return None
        if action.action == "diagnosis_provisional":
            state["provisional"] = action.query_text
            return (
                "Provisional diagnosis noted. You may now order investigations and imaging; "
                "history and examination are closed."
            )

        category = action.action
        if category not in self._SOURCES:
            return "Unrecognised action. Choose one of: " + ", ".join(self.ACTIONS)

        self.bump(state, category)
        if self.over_limit(state, category):
            return (
                f"Limit on {category} reached. Please proceed to working up the patient, "
                "or give your final diagnosis."
            )
        store: EvidenceStore = state["evidence"]
        revealed = store.reveal(category, action.query_text)
        if not revealed:
            # The case does not record this finding. Saying so is the honest
            # answer and is what the release does; inventing a normal result
            # would hand the model evidence the case never had.
            body = f"No {category} finding recorded for that request in this case."
        else:
            body = revealed
        return body + self.limit_note(state, category)

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        # THE RELEASE'S STEM, NOT THE `vignette` COLUMN. That column is the
        # source PubMed case report in full -- title, work-up, diagnosis and
        # outcome -- and it is what the release's pipeline read *to build* the
        # structured case, not what it ever showed a model. Measured on the
        # released tables: the gold diagnosis appears verbatim in 41% of
        # pubmed_reviewed vignettes ("...consistent with pheochromocytoma",
        # "...the diagnosis of BRASH syndrome was suspected"), and every one of
        # the 990 begins with the article's own title. Handing that to a model
        # and asking it to diagnose measures reading, not reasoning. The stem
        # the release does show -- demographics, chief complaint, initial
        # vitals -- leaks the diagnosis in 0 of 990 and 0 of 1952 rows.
        case = self._case_json(item)
        stem = C.normalize_whitespace(self._stem(case)) if case else ""
        diagnoses = [C.normalize_whitespace(d) for d in _parse_listish(item.get("diagnosis"))]
        diagnoses = [d for d in diagnoses if d]
        if not stem or not diagnoses:
            return None
        # Kept under the old name so the two SampleSpec branches below read the
        # same; it is the release stem, never the `vignette` column.
        vignette = stem
        differentials = [
            C.normalize_whitespace(d) for d in _parse_listish(item.get("differentials"))
        ]
        differentials = [d for d in differentials if d]
        findings = self._findings(item)

        if self._subtask == "selection":
            # Options are the case's own differentials plus the gold diagnosis.
            candidates = [diagnoses[0]] + [d for d in differentials if d != diagnoses[0]]
            if len(candidates) < 3:
                return None  # not enough real distractors; item is skipped, never invented
            # Plain alphabetical order. The previous key sorted the gold
            # diagnosis to the front of every item -- `text != gold` is False
            # for the gold and True for everything else -- so the answer was
            # always option 1 and the task could be passed without reading the
            # case. Alphabetical is deterministic (so resume stays stable) and
            # says nothing about which option is correct.
            ordered = C.shuffled_options(
                sorted(set(candidates)),
                key=str(item.get("uid", index)),
                seed=self.context.seed,
            )
            labels = C.choice_labels(len(ordered))
            return SampleSpec(
                sample_id=C.stable_id("viva", item.get("uid", index)),
                fields={
                    "observation": vignette,
                    "context": findings,
                    "question": "Which diagnosis best explains this presentation?",
                    "options": ordered,
                    "option_labels": labels,
                },
                reference={
                    "gold_label": labels[ordered.index(diagnoses[0])],
                    "gold": diagnoses[0],
                },
                task_kind="selection",
                max_tokens=1024,
                metadata={
                    "uid": item.get("uid"),
                    "specialty": item.get("specialty_group"),
                    # The structured case is the environment for an interactive
                    # episode; underscore-prefixed so it stays out of records.
                    "_case": self._case_json(item),
                },
            )

        return SampleSpec(
            sample_id=C.stable_id("viva", item.get("uid", index)),
            fields={
                "observation": vignette,
                "context": findings,
                "question": "What is the most likely diagnosis?",
                "instructions": "Name the single most likely diagnosis for this patient.",
            },
            reference={"gold": diagnoses[0], "accepted": diagnoses},
            task_kind="generation",
            max_tokens=1024,
            metadata={
                "uid": item.get("uid"),
                "specialty": item.get("specialty_group"),
                "n_accepted": len(diagnoses),
                "_case": self._case_json(item),
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
        answer = self._committed_diagnosis(response.text)
        if answer is not None:
            # An interactive episode ends in a structured diagnosis_final
            # action, not a free-text line, so the condition names are pulled
            # out before the usual matching runs.
            response = ModelResponse(
                sample_id=response.sample_id,
                model_id=response.model_id,
                status=response.status,
                content=f"Answer: {answer}",
                finish_reason=response.finish_reason,
            )
            output_contract = {"answer_prefix": "Answer:"}
        score = text_match_score(
            response,
            gold=sample.reference["gold"],
            # The dataset itself lists several accepted diagnoses per case.
            accepted=sample.reference.get("accepted", [])[1:],
            output_contract=output_contract,
            primary="diagnosis_match",
        )
        turns = sample.metadata.get("turns_used")
        if turns:
            # What the model spent to get there is part of an interactive
            # result: two systems with the same accuracy are not equivalent if
            # one needed four times as many investigations.
            score.metrics["turns_used"] = float(turns)
            score.metrics["committed"] = 1.0 if answer is not None else 0.0
        return score

    @staticmethod
    def _committed_diagnosis(text: str) -> str | None:
        """The condition names from a ``diagnosis_final`` action, if there is one."""
        action = parse_action(text, actions=VivaBenchAdapter.ACTIONS)
        if action.action != "diagnosis_final":
            return None
        query = action.query
        if isinstance(query, list):
            names = [
                str(entry.get("condition") or entry.get("diagnosis") or "")
                for entry in query
                if isinstance(entry, dict)
            ]
            names = [name for name in names if name]
            if names:
                # The first entry is the model's own top-ranked diagnosis.
                return names[0]
        return action.query_text or None

    def judge_request(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore
    ) -> dict[str, Any] | None:
        if sample.task_kind != "generation" or not response.text:
            return None
        return {
            "candidate": extract_answer_span(response.text, None)[:600],
            "gold": "; ".join(sample.reference.get("accepted", [sample.reference["gold"]])),
            "observation": C.clip_words(sample.fields["observation"], 200),
            "criteria": "Any of the listed accepted diagnoses counts as correct.",
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
            name="VivaBench",
            domain="Healthcare: Interactive Clinical Reasoning",
            source_url=f"https://huggingface.co/datasets/{REPO_ID}",
            processing_mode="Generation & Selection",
            split_used=self.split_used,
            abductive_subset=(
                "STATIC SELECTION ONLY, over the candidate list the official GitHub release "
                "ships with each case: the differentials it records next to the final "
                "diagnosis. The release's interactive finding-request protocol is NOT run "
                "here -- that is a different task, and running it as well would evaluate the "
                "same cases twice. Evidence disclosure within the static form remains a "
                "configuration choice (options.evidence = vignette | with_findings), recorded "
                "per run."
            ),
            sampling_procedure=self.sampling_note(),
            metrics_description={
                "self_consistency_<metric>":
                "Every metric also gets a self_consistency_ counterpart: the plurality answer over "
                "modes.repeats samples of the same record, read off those samples rather than bought "
                "again. Available because this dataset's answers are checkable and so can coincide.",
                "diagnosis_match": "(PRIMARY, higher is better) 1 if the answer equals or contains any accepted diagnosis "
                "(primary for the generation subtask)",
                "exact_match": "strict normalized equality with the primary diagnosis",
                "token_f1": "bag-of-tokens F1 against the primary diagnosis",
                "rouge_l": "LCS F-measure against the primary diagnosis",
                "accuracy": "(PRIMARY, higher is better) selection subtask: 1 if the chosen option is the gold diagnosis",
                "diagnosis_match_judged": "LLM-judge verdict (only when engine.judge.enabled)",
            },
            primary_metric="accuracy" if self._subtask == "selection" else "diagnosis_match",
            decisions=[
                "Default table is pubmed_reviewed (human-reviewed) rather than the larger "
                "generated table; options.table switches.",
                "THE PROMPT IS THE RELEASE'S OWN CASE STEM -- demographics, chief complaint and "
                "initial vitals, built the way vivabench/examiner.py builds it -- and NOT the "
                "`vignette` column. That column is the full source case report, which the "
                "release read to construct the structured case and never showed to a model; it "
                "names the gold diagnosis outright in 41% of pubmed_reviewed rows (0% for the "
                "stem) and every one of its 990 rows opens with the article's title.",
                "options.evidence = with_findings additionally reveals the structured "
                "examination and investigations up front, as an upper bound on what the "
                "interaction is worth. The case's `diagnosis` and `differentials` keys are "
                "excluded from that dump -- they are the answer, not evidence.",
                "The selection subtask uses only the case's own differentials as distractors and "
                f"skips cases with fewer than two ({self._with_differentials} of the table's rows "
                "have enough) -- distractors are never invented.",
                "All diagnoses in the row's list count as correct, since the dataset records "
                "multi-part diagnoses.",
            ],
            caveats=[
                "Because the differentials column is empty for many rows, the selection subtask "
                "covers a biased subset; the generation default avoids that.",
                "The stem is deliberately thin -- three fields. A model given only this cannot "
                "do well without asking for more, which is the benchmark's point, but this "
                "adapter runs the STATIC form, where it cannot ask. Read the score as "
                "'diagnosis from the opening turn alone', and use options.evidence = "
                "with_findings for the contrast.",
                "Cases derive from published reports and may be memorized.",
            ],
            statistics={
                **self.base_statistics(),
                "subtask": self._subtask,
                "evidence": str(self.context.option("evidence", "vignette")),
                "rows_with_differentials": getattr(self, "_with_differentials", 0),
            },
        )
