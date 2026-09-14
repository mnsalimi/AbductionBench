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

**There is no selection variant.**  The release never offers the examinee a
candidate list; its ``differentials`` column is part of the case's answer.  For
a period this adapter was configured ``data_delivery_mode = "static"`` and run
as selection over that column, with the whole interactive protocol below
present but unreachable -- which measured picking from five rather than working
a patient up.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
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
from ._base import PooledDatasetAdapter, apply_judged_metric, judged_only_score
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

    adapter_version = "2.0"

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. You are given a clinical viva "
        "case. State the diagnosis that explains the presentation."
    )
    #: VivaBench IS the interaction. The release drives a multi-turn loop --
    #: the agent asks for history, examines, orders tests, then commits -- and
    #: what it measures is which findings the model went looking for. Run as a
    #: one-shot it measures something else entirely.
    data_delivery_mode = "interactive"
    #: The committed diagnosis is free text against a list of accepted
    #: diagnoses, so the judge scores it; there is no candidate list to check
    #: a label against.
    objective_metrics = False
    selection_cardinality = None
    hypothesis_modes = ("generation",)
    hypothesis_mode_options = {
        "generation": {},
    }
    table_hypothesis_mode = "Generation (interactive)"
    hypothesis_mode_justification = (
        "VivaBench is run the way it is published: an interactive viva. The agent takes a "
        "history, examines the patient, orders investigations and imaging, and commits to a "
        "diagnosis, under the release's own action vocabulary and limits. There is no "
        "selection variant, because the release never offers the examinee a candidate list -- "
        "the `differentials` column is part of the case's answer, not an option set. An "
        "earlier version of this adapter ran it as static selection over that column, which "
        "is a different and much easier task than the benchmark poses."
    )
    primary_metric = "diagnosis_judged"

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
                            "vivabench/examiner.py", "vivabench/ontology/*.py",
                            "configs/*.yaml"],
        )
        self._apply_release_limits(root)
        table = str(self.context.option("table", "pubmed_reviewed"))
        files = C.find_files(root, [f"{table}.csv"]) or C.find_files(root, ["*.csv"])
        if not files:
            raise SkippedDataset("no VivaBench CSV found in the release")
        rows = C.read_csv_rows(files[0])
        self._with_differentials = sum(
            1 for row in rows if len(_parse_listish(row.get("differentials"))) >= 2
        )
        self.split_used = (
            f"{files[0].stem} ({len(rows)} cases); the release ships no train/test split"
        )
        return rows

    def _apply_release_limits(self, root: Path) -> None:
        """Take the examination limits from the release's own evaluate.yaml.

        ``examination:`` in ``configs/evaluate.yaml`` is where the published
        numbers live (hx 10, phys 5, ix 5, img 5, action 20). Reading them
        beats restating them: if the release revises a limit, this follows.
        A missing or unreadable file leaves the class defaults, which are the
        same numbers from ``Examiner.__init__``.
        """
        path = root / "configs" / "evaluate.yaml"
        if not path.is_file():
            return
        try:
            import yaml

            block = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get(
                "examination"
            ) or {}
        except Exception as exc:  # noqa: BLE001 - a bad file leaves the defaults
            self.log.warning("VivaBench: cannot read the release's evaluate.yaml: %s", exc)
            return
        limits = dict(self.category_limits)
        for category, key in self._LIMIT_KEYS.items():
            value = block.get(key)
            if isinstance(value, int) and value > 0:
                limits[category] = value
        self.category_limits = limits
        action_limit = block.get("action_limit")
        if isinstance(action_limit, int) and action_limit > 0:
            self.max_turns = action_limit
        self.limits_source = (
            f"configs/evaluate.yaml (hx {limits['history']}, phys {limits['examination']}, "
            f"ix {limits['investigation']}, img {limits['imaging']}, actions {self.max_turns})"
        )

    # ------------------------------------------------------------------ #
    # the examination loop -- VivaBench's own protocol
    # ------------------------------------------------------------------ #

    #: Defaults from ``Examiner.__init__``; ``load_items`` overrides them with
    #: whatever the release's own ``configs/evaluate.yaml`` says, so the limits
    #: are the release's rather than a transcription of them.
    category_limits = {"history": 10, "examination": 5, "investigation": 5, "imaging": 5}
    #: ``Examination(turn_limit=20)`` and ``Examiner(action_limit=20)``.
    max_turns = 20

    #: ``Examiner.__init__`` keyword <- ``evaluate.yaml`` key, for the four
    #: request categories plus the overall action budget.
    _LIMIT_KEYS = {
        "history": "hx_limit",
        "examination": "phys_limit",
        "investigation": "ix_limit",
        "imaging": "img_limit",
    }

    #: The examiner's own replies, copied from ``Examiner.process_response``
    #: and ``Examiner.process_*`` so the transcript reads as the release's does.
    #: The wording is part of the protocol: it is what tells the agent a door
    #: has closed, and paraphrasing it would change what the model is told.
    _CLOSED_MSG = (
        "You can no longer review the patient. Please proceed to order any "
        "investigations or imaging to help with diagnosis."
    )
    _PROVISIONAL_MSG = "Thank you. Please proceed to imaging and lab investigations."
    _FINAL_MSG = "Final diagnosis was made."
    _OUT_OF_TIME_MSG = (
        "\nYou have run out of time. Please give your final diagnosis for this patient."
    )
    _LIMIT_MSGS = {
        "history": "\nLimit on history-taking reached. Please proceed to further working up the patient.",
        "examination": "\nLimit on physical examination reached. Please proceed to further working up the patient.",
        "investigation": "\nLimit on ordering investigations reached. Please proceed to further working up the patient.",
        "imaging": "\nLimit on ordering imaging reached. Please proceed to further working up the patient.",
    }
    #: ``RETRY_LIMIT`` in ``vivabench/examiner.py``.
    _RETRY_LIMIT = 2

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
        """Open the viva exactly as ``Examination.__init__`` does.

        System message is the release's ``ASSISTANT_BASE_PROMPT`` -- which is
        where the workflow constraints, the action vocabulary and the required
        JSON shape are all defined -- followed by the case stem as the first
        human turn.
        """
        prompts = self._release_prompts()
        system = prompts.get("ASSISTANT_BASE_PROMPT") or self.system_prompt
        payload = sample.metadata.get("_case") or {}
        state: dict[str, Any] = {
            "evidence": self._evidence({"clinicalcase": json.dumps(payload)}),
            "counts": {},
            "actions": 0,
            "retries": 0,
            # `Examiner.reviewed_patient`: set by the first investigation,
            # imaging or provisional diagnosis, and it closes history and
            # examination for the rest of the episode.
            "reviewed_patient": False,
            "final": None,
            "provisional": None,
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
        """One examiner turn, following ``Examiner.process_response``.

        The routing, the order of the checks and the examiner's replies are the
        release's. The one thing that is not is how a free-text request is
        matched to a finding: the release resolves that with a second LLM (its
        ``LLMMapper``/``LLMParser``, gpt-4.1 by default), and this suite
        matches lexically instead. That is a deliberate, documented deviation
        -- an examiner model inside the evaluation makes two runs of the same
        system disagree because the examiner did -- and it is the only one.
        """
        action = parse_action(assistant_text, actions=self.ACTIONS)
        if not action.parsed and not action.action:
            # `invoke_agent`: a response that will not parse is returned to the
            # agent with ERROR_RETURN_MSG and its own previous message, up to
            # RETRY_LIMIT consecutive times.
            state["retries"] = state.get("retries", 0) + 1
            if state["retries"] > self._RETRY_LIMIT:
                return None
            error = self._release_prompts().get(
                "ERROR_RETURN_MSG", "Unable to parse your response."
            )
            return f"{error}{assistant_text[:400]}"
        state["retries"] = 0

        category = action.action
        if category not in self.ACTIONS:
            return "Unrecognised action. Choose one of: " + ", ".join(self.ACTIONS)

        # `Examiner.process_response` counts EVERY action against the budget,
        # diagnoses included, and appends the out-of-time notice when the last
        # one is spent.
        state["actions"] = state.get("actions", 0) + 1
        out_of_time = state["actions"] >= self.max_turns

        if category == "diagnosis_final":
            state["final"] = action.query_text
            state["final_query"] = action.query
            return None

        if category == "diagnosis_provisional":
            state["provisional"] = action.query_text
            state["reviewed_patient"] = True
            reply = self._PROVISIONAL_MSG
            return reply + (self._OUT_OF_TIME_MSG if out_of_time else "")

        if category in ("history", "examination") and state.get("reviewed_patient"):
            # The workflow gate: once the work-up starts, the patient is no
            # longer available to interview or examine.
            return self._CLOSED_MSG + (self._OUT_OF_TIME_MSG if out_of_time else "")

        if category in ("investigation", "imaging"):
            state["reviewed_patient"] = True

        self.bump(state, category)
        used = state["counts"].get(category, 0)
        limit = self.category_limits.get(category)
        if limit is not None and used > limit:
            # Past the category's budget: the release stops answering that kind
            # of request and says so.
            return self._LIMIT_MSGS[category].lstrip("\n") + (
                self._OUT_OF_TIME_MSG if out_of_time else ""
            )

        store: EvidenceStore = state["evidence"]
        revealed = store.reveal(category, action.query_text)
        if not revealed:
            # The case does not record this finding. Saying so is the honest
            # answer; inventing a normal result would hand the model evidence
            # the case never had.
            body = f"No {category} finding recorded for that request in this case."
        else:
            body = revealed
        if limit is not None and used >= limit:
            body += self._LIMIT_MSGS[category]
        if out_of_time:
            body += self._OUT_OF_TIME_MSG
        return body

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

        return SampleSpec(
            sample_id=C.stable_id("viva", item.get("uid", index)),
            fields={
                # The stem is the episode's opening turn; interactive_start
                # builds the real first message from the structured case. These
                # fields are what a `options.delivery = static` ablation would
                # render, and what the record and the judge see.
                "observation": stem,
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
                # The structured case is the environment for the episode;
                # underscore-prefixed so it stays out of the records.
                "_case": case,
            },
        )

    def score(
        self,
        sample: SampleSpec,
        response: ModelResponse,
        *,
        output_contract: dict[str, Any] | None = None,
    ) -> SampleScore:
        answer = self._committed_diagnosis(response.text)
        if answer is None:
            # No diagnosis_final action: the episode ran out of turns, or ended
            # on a malformed one. The release treats that as no diagnosis at
            # all, and so does this -- the last turn's prose is NOT handed to
            # the judge, because a model that reasoned aloud about the right
            # condition without ever committing to it has not diagnosed the
            # patient, and letting the judge read that text would credit it.
            return SampleScore(
                metrics={"diagnosis_judged": 0.0, "committed": 0.0},
                prediction=None,
                parse_ok=False,
                details={"uncommitted": response.text[:300]},
            )
        if answer is not None:
            # An episode ends in a structured diagnosis_final action, not a
            # free-text line, so the condition name is pulled out of it before
            # anything else looks at the text.
            response = ModelResponse(
                sample_id=response.sample_id,
                model_id=response.model_id,
                status=response.status,
                content=f"Answer: {answer}",
                finish_reason=response.finish_reason,
            )
            output_contract = {"answer_prefix": "Answer:"}
        score = judged_only_score(
            response,
            metric="diagnosis_judged",
            output_contract=output_contract,
            details={"gold": "; ".join(sample.reference.get("accepted", []))[:300]},
        )
        # Committing at all is a separate competence from committing correctly.
        score.metrics["committed"] = 1.0
        turns = sample.metadata.get("turns_used")
        if turns:
            # What the model spent to get there is part of an interactive
            # result: two systems with the same accuracy are not equivalent if
            # one needed four times as many investigations.
            score.metrics["turns_used"] = float(turns)
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
        # Only a committed diagnosis is judged. score() leaves `prediction`
        # empty when the episode ended without a diagnosis_final action, and
        # sending the transcript instead would let the judge find the condition
        # in the model's thinking-aloud and credit a diagnosis never given.
        if not response.text or not score.prediction:
            return None
        return {
            "candidate": str(score.prediction)[:600],
            "gold": "; ".join(sample.reference.get("accepted", [sample.reference["gold"]])),
            "observation": C.clip_words(sample.fields["observation"], 200),
            "criteria": "Any of the listed accepted diagnoses counts as correct.",
        }

    def apply_judge(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore, verdict: Any
    ) -> SampleScore:
        return apply_judged_metric(score, verdict, "diagnosis_judged")

    def documentation(self) -> AdapterDocumentation:
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="VivaBench",
            domain="Healthcare: Interactive Clinical Reasoning",
            source_url=f"https://huggingface.co/datasets/{REPO_ID}",
            processing_mode="Generation (interactive)",
            split_used=self.split_used,
            abductive_subset=(
                "THE WHOLE BENCHMARK, RUN AS PUBLISHED: a multi-turn viva. The agent takes a "
                "history, examines the patient, orders investigations and imaging and commits "
                "to a diagnosis, under the release's own action vocabulary, workflow gate and "
                "limits. Every case is abductive -- the diagnosis has to be inferred from "
                "findings the model chooses to go and get. There is no selection variant: the "
                "release never shows the examinee a candidate list, and the `differentials` "
                "column is part of the case's answer rather than an option set."
            ),
            sampling_procedure=self.sampling_note(),
            metrics_description={
                "diagnosis_judged": (
                    "(PRIMARY, higher is better, 0-1) LLM-judge verdict on whether the "
                    "committed final diagnosis is one of the case's accepted diagnoses, "
                    "however worded. The release compares diagnoses by ICD-10 mapping and "
                    "embedding similarity; neither is shipped in this snapshot, so the judge "
                    "stands in for them."
                ),
                "committed": (
                    "(higher is better, 0-1) fraction of episodes that ended in a "
                    "diagnosis_final action at all. An episode can run out of turns without "
                    "ever committing, which is a different failure from committing wrongly."
                ),
                "turns_used": (
                    "(diagnostic, no direction) model turns the episode took. Two systems with "
                    "the same score are not equivalent if one needed four times the work-up, "
                    "which is most of what this benchmark exists to measure."
                ),
                "parse_failure_rate": (
                    "(lower is better, 0-1) fraction of episodes no diagnosis could be read "
                    "from; these score 0 and are counted separately from being wrong."
                ),
                "best_of_n_<metric>":
                "Every metric also gets a best_of_n_ counterpart: per record, the repeat the "
                "judge scored highest. A diagnosis is free text with many correct surface "
                "forms, so a plurality over repeats is not meaningful and Best-of-N replaces it.",
            },
            primary_metric="diagnosis_judged",
            decisions=[
                "RUN AS AN INTERACTIVE VIVA, which is what the benchmark is. An earlier "
                "version of this adapter ran it as static selection over the case's own "
                "differentials -- a task the release never poses, and a much easier one, "
                "since it replaces 'work out what to ask' with 'pick from five'.",
                "The agent's system prompt is the release's own ASSISTANT_BASE_PROMPT, read "
                "from vivabench/prompts/examiner.py in the snapshot rather than transcribed; "
                "it is where the workflow constraints, the action vocabulary and the required "
                "JSON response shape are defined.",
                "The examiner's replies are the release's own strings -- the closed-patient "
                "notice, the per-category limit notices, the provisional acknowledgement and "
                "the out-of-time warning -- because the wording is what tells the agent a door "
                "has closed.",
                "The workflow gate is the release's: the first investigation, imaging or "
                "provisional diagnosis sets `reviewed_patient`, after which history and "
                "examination are refused.",
                f"Limits are read from the release's own configs/evaluate.yaml -- "
                f"{getattr(self, 'limits_source', 'hx 10, phys 5, ix 5, img 5, actions 20')} -- "
                "rather than restated here, and every action counts against the budget, "
                "diagnoses included, as in Examiner.process_response.",
                "An unparseable turn is returned to the agent with the release's "
                "ERROR_RETURN_MSG and its own previous message, up to RETRY_LIMIT = 2 "
                "consecutive times, after which the episode ends.",
                "THE PROMPT IS THE RELEASE'S OWN CASE STEM -- demographics, chief complaint "
                "and initial vitals, built the way vivabench/examiner.py builds it -- and NOT "
                "the `vignette` column. That column is the full source case report, which the "
                "release read to construct the structured case and never showed to a model; it "
                "names the gold diagnosis outright in 41% of pubmed_reviewed rows (0% for the "
                "stem) and every one of its 990 rows opens with the article's title.",
                "Default table is pubmed_reviewed (human-reviewed) rather than the larger "
                "generated table; options.table switches.",
                "All diagnoses in the row's list count as correct, since the dataset records "
                "multi-part diagnoses.",
            ],
            caveats=[
                "REQUESTS ARE MATCHED TO FINDINGS LEXICALLY, NOT BY A SECOND MODEL. The "
                "release resolves a free-text request with its own LLMMapper/LLMParser "
                "(gpt-4.1 by default); this snapshot ships neither, and putting an examiner "
                "model inside the evaluation would let two runs of the same system disagree "
                "because the examiner did. The cost is that a request phrased unusually can "
                "be answered with 'no finding recorded' when the case does record it -- "
                "visible in the transcript, and the one place this adapter departs from the "
                "release's behaviour.",
                "The release also scores diagnoses by ICD-10 mapping and sentence-embedding "
                "similarity at a 0.8 threshold (configs/evaluate.yaml, metrics:). Those "
                "resources are not in the snapshot either, so the judge stands in, and scores "
                "are not directly comparable to published VivaBench numbers.",
                "The release's optional full-information condition "
                "(diagnose_with_full_information, an upper bound measured by showing the whole "
                "case at once) is not run; options.delivery = static is the nearest analogue "
                "and shows only the stem.",
                "Cases derive from published reports and may be memorized.",
            ],
            statistics={
                **self.base_statistics(),
                "table": str(self.context.option("table", "pubmed_reviewed")),
                "limits": getattr(self, "limits_source", ""),
                "max_turns": self.max_turns,
                "rows_with_differentials": getattr(self, "_with_differentials", 0),
            },
        )
