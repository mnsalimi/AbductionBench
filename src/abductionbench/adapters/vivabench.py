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

**The protocol prompt is this suite's, adapted rather than quoted.**  The six actions, the workflow order, the gate that closes the patient once the work-up begins, the per-category limits (read from the release's own ``configs/evaluate.yaml``) and the diagnosis payload shape are the release's and are reproduced in substance, because they are the benchmark.  Its *reasoning elicitation* is not: the ``"reasoning"`` field of its JSON contract, and its "include a short line of reasoning for your action", are dropped -- including from the parse-error message, which restated the schema and would otherwise have reintroduced the field on a retry.  This dataset runs ``io`` only (``io_only``), and an instruction to explain a choice would contradict the mode instruction sitting in the same prompt -- and, here, the parser as well.  The wording is therefore this harness's, in the same five layers every static prompt uses.

The examiner's own replies -- the closed-patient notice, the limit notices, the
provisional acknowledgement and the out-of-time warning -- are kept verbatim:
they carry no reasoning request, and the wording is what tells the agent a door
has closed.  **The mapper is configurable.**  VivaBench resolves a free-text
request to a finding with an LLM (``LLMMapper``/``LLMParser``, gpt-4.1 by
default).  With ``engine.simulator`` enabled this adapter does the same -- with
whichever model the run names, which need not be gpt-4.1 -- and keeps the
release's split: the model chooses *which* recorded keys the request
resolves to, and the finding itself is still rendered from the case, so it can
decide what is disclosed and never what it says.  With the simulator off, a
lexical matcher stands in -- deterministic and reproducible, and visible in the
transcript when it misses a request the case does record.  Which one ran is in
the run's ``Simulators`` sheet, and it changes the score: mapped by a model,
this dataset is no longer bit-reproducible.

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
from ._prompting import ProtocolParts, build_protocol_messages

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
    #: The protocol prompt below is this suite's, adapted from the release's
    #: workflow rather than quoted from it, so this is False. It still runs one
    #: prompt mode, for the reason io_only states: every turn has to be one
    #: JSON action the examiner can execute.
    authors_prompt = False
    io_only = True
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

    #: What the examiner says to a turn it cannot read. The release's own
    #: ERROR_RETURN_MSG restates its JSON schema *including* the reasoning
    #: field; this one restates the schema this adapter actually asks for, so
    #: the retry cannot reintroduce the field the opening removed.
    _PARSE_ERROR_MSG = (
        "That could not be read as an action. Reply with one JSON object and nothing "
        'else: {"action": "<one of the allowed actions>", "query": "<your request>"}.\n'
        "Your previous message:"
    )

    #: The release resolves a free-text request to the case's own keys with a
    #: model (``LLMMapper``/``LLMParser``, gpt-4.1 by default; this suite names
    #: its own in ``engine.simulator``) and then has the
    #: Examiner read the recorded finding back.  This reproduces that split:
    #: the mapper chooses *which* keys are disclosed, and the finding itself is
    #: still rendered from the case by :meth:`EvidenceStore.reveal_keys`.  So a
    #: simulator can decide what the doctor gets to see, and cannot decide what
    #: it says -- it has no way to invent a result, because it never writes the
    #: reply.
    _MAPPER_BRIEF = (
        "You are the examiner in a clinical viva. A doctor has made one "
        "request about a case. Your only job is to decide which of the case's "
        "recorded findings answer it.\n\n"
        "The findings recorded under '{category}', as key: value:\n"
        "{catalogue}\n\n"
        "Rules you must follow:\n"
        "- Reply with a JSON array of the keys, exactly as written above, "
        "whose findings answer the request. Nothing else.\n"
        "- Include a key when the request names that finding, or names a "
        "region, system or work-up it clearly belongs to.\n"
        "- If nothing recorded answers the request, reply with exactly: []\n"
        "- Never invent a key, and never reply with the finding itself.\n"
        "- At most {limit} keys, the most directly relevant first."
    )
    #: How many findings one request may disclose. ``EvidenceStore.reveal``'s
    #: own default, so the mapper and the matcher are held to the same budget.
    _MAPPER_KEY_LIMIT = 6

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

    #: The release's action vocabulary with the release's own descriptions of
    #: what each one is for, reworded into this suite's voice. The clinical
    #: content is the benchmark's -- which bedside tests count as
    #: investigations rather than imaging is a property of VivaBench, not a
    #: stylistic choice -- and the examiner routes on exactly these names.
    _ACTION_HELP = (
        ("history", "interview the patient. One or two questions at a time; assume average "
                    "medical literacy."),
        ("examination", "perform a physical examination. Name the examination and the sign "
                        "you are looking for."),
        ("diagnosis_provisional", "give your provisional diagnosis, after reviewing the "
                                  "patient and before any investigation or imaging."),
        ("investigation", "order a test that is not imaging -- laboratory tests (name the "
                          "specimen type if it is not serological), bedside tests such as "
                          "ECG, and special tests such as EEG or pulmonary function tests."),
        ("imaging", "order imaging performed by a radiologist, radiographer or nuclear "
                    "medicine physician -- x-ray, ultrasound, CT, MRI, PET, VQ. Name both "
                    "the modality and the anatomical region."),
        ("diagnosis_final", "give your final diagnosis. This ends the consultation."),
    )

    #: The diagnosis payload the scorer reads: a list whose first entry is the
    #: model's own top-ranked condition. Kept exactly as the release defines it,
    #: because `_committed_diagnosis` and the primary metric both depend on it.
    _DIAGNOSIS_SHAPE = (
        '[{"condition": "<name>", "icd_10_name": "<ICD-10 name>", '
        '"icd_10": "<ICD-10 code>", "confidence": <0.0-1.0>}]'
    )

    def _protocol(self, sample: SampleSpec, payload: dict[str, Any]) -> ProtocolParts:
        """This suite's opening prompt for one viva, in the shared five layers."""
        return ProtocolParts(
            system=self.system_prompt,
            observation=self._stem(payload),
            instructions=(
                "Work this patient up and reach the diagnosis that accounts for their "
                "presentation. You decide what to ask, what to examine and what to order."
            ),
            requirements=[
                "take a history and examine the patient before ordering any test or imaging",
                "give a provisional diagnosis after reviewing the patient and before "
                "ordering any investigation or imaging",
                "once you order an investigation or imaging, the patient is no longer "
                "available for history or examination",
                "perform exactly one action per turn",
                "a diagnosis may list up to five conditions, each with its ICD-10 name and "
                "code and a confidence between 0.0 and 1.0; list the one you consider most "
                "likely first",
                "the confidences do not have to sum to 1.0",
                "give the final diagnosis once the evidence supports one",
            ],
            actions=list(self._ACTION_HELP),
            output_format=(
                "Reply with one JSON object and nothing else -- no markdown, no code fence, "
                "no text outside the object:\n"
                '{"action": "<one of the actions above>", "query": "<your request>"}\n'
                "For either diagnosis action, the query is a list instead:\n"
                f"{self._DIAGNOSIS_SHAPE}"
            ),
        )

    def interactive_start(self, sample: SampleSpec) -> tuple[list[ChatMessage], dict[str, Any]]:
        """Open the viva on this suite's own protocol prompt.

        Everything the environment enforces is carried over from the release's
        ``ASSISTANT_BASE_PROMPT``: the six actions, the order they have to come
        in, the gate that closes the patient once the work-up starts, and the
        shape of a diagnosis. What is not carried over is its reasoning
        elicitation -- the ``"reasoning"`` JSON field and "include a short line
        of reasoning for your action" -- because this task runs io only and
        those would contradict the mode instruction in the same prompt.
        """
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
        return build_protocol_messages(self._protocol(sample, payload), self.context.modes), state

    async def interactive_step(
        self, sample: SampleSpec, state: dict[str, Any], assistant_text: str
    ) -> str | None:
        """One examiner turn, following ``Examiner.process_response``.

        The routing, the order of the checks and the examiner's replies are the
        release's, and so is the *shape* of the mapping from a free-text
        request to a finding when ``engine.simulator`` is configured -- though
        not the model doing it, which is configured per run and need not be
        the release's gpt-4.1. Without a simulator the request is matched
        lexically instead: reproducible, and the one place the adapter then
        departs from the release's behaviour altogether.
        """
        action = parse_action(assistant_text, actions=self.ACTIONS)
        if not action.parsed and not action.action:
            # `invoke_agent`: a response that will not parse is returned to the
            # agent with ERROR_RETURN_MSG and its own previous message, up to
            # RETRY_LIMIT consecutive times.
            state["retries"] = state.get("retries", 0) + 1
            if state["retries"] > self._RETRY_LIMIT:
                return None
            return f"{self._PARSE_ERROR_MSG}{assistant_text[:400]}"
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
        # NO HARD REFUSAL. `Examiner.process_history` and its three siblings
        # always map, parse and return the findings, and only *append* the
        # limit notice once the count reaches the limit:
        #
        #     self.hx_count += 1
        #     if self.hx_count >= self.hx_limit:
        #         _prompt += "\nLimit on history-taking reached. ..."
        #     return _prompt
        #
        # Nothing upstream stops answering. The per-category limits are
        # advice; the only hard cap is `action_limit`, which bounds the
        # episode as a whole. This adapter used to refuse every request past
        # the limit -- and said in a comment that the release did too, which
        # it does not -- so an agent that spent eleven turns on history got
        # ten answers and then a wall, instead of eleven answers and a nudge.

        store: EvidenceStore = state["evidence"]
        revealed = await self._reveal(state, store, category, action.query_text)
        if revealed is None:
            # The examiner stopped answering. The engine ends the episode as an
            # error; the model's half-finished work-up is not a diagnosis it
            # chose to give.
            return None
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

    async def _reveal(
        self,
        state: dict[str, Any],
        store: EvidenceStore,
        category: str,
        query: str,
    ) -> str | None:
        """The finding this request discloses, or ``""`` if the case has none.

        ``None`` means the examiner could not be reached, which is not the same
        as a case that records nothing: one ends the episode as an error, the
        other is a legitimate answer the doctor has to work around.
        """
        if self.simulator is None:
            return store.reveal(category, query)
        catalogue = store.categories.get(category) or {}
        if not catalogue:
            return ""
        brief = self._MAPPER_BRIEF.format(
            category=category,
            catalogue="\n".join(f"{key}: {value}" for key, value in catalogue.items()),
            limit=self._MAPPER_KEY_LIMIT,
        )
        # One mapping conversation per category: the release maps each request
        # against the case section it names, and a history request must not be
        # resolved against the investigations the doctor has not ordered yet.
        raw = await self.simulate(
            state, brief=brief, request=query, role=f"mapper:{category}"
        )
        if raw is None:
            return None
        keys = self._parse_keys(raw, set(catalogue))
        return store.reveal_keys(category, keys, limit=self._MAPPER_KEY_LIMIT)

    @staticmethod
    def _parse_keys(raw: str, known: set[str]) -> list[str]:
        """The keys a mapper reply names, in order, ignoring everything else.

        A model that wraps its array in prose or a fence is still understood; a
        model that invents a key is not, and the invention is dropped rather
        than guessed at. Falling back to the lexical matcher here would hide a
        mapper that had stopped working behind plausible answers, so an
        unreadable reply discloses nothing -- which the transcript shows as the
        case recording no such finding.
        """
        found: list[str] = []
        for match in re.finditer(r'"([^"]+)"', raw or ""):
            key = match.group(1)
            if key in known and key not in found:
                found.append(key)
        if not found:
            # Unquoted or bare-word replies. Longest keys first, so a parent is
            # not claimed by a prefix of one of its children -- and on a word
            # boundary, or a short key would be found inside an unrelated word
            # and a reply of pure prose would "name" a finding.
            for key in sorted(known, key=len, reverse=True):
                if key in found:
                    continue
                if re.search(rf"(?<![\w.]){re.escape(key)}(?![\w.])", raw or ""):
                    found.append(key)
        return found

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
            "observation": sample.fields["observation"],
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
                "THE PROTOCOL PROMPT IS LOCALLY ADAPTED, NOT THE AUTHORS'. The six actions, "
                "the workflow order, the gate, the limits and the diagnosis payload shape come "
                "from the release's ASSISTANT_BASE_PROMPT and are reproduced in substance; the "
                "wording is this suite's, in its standard prompt layers. The release's "
                "`reasoning` JSON field and its request for a line of reasoning per action are "
                "dropped, here and in the parse-error retry, because this task runs io only.",
                "The examiner's replies are still the release's own strings -- the "
                "closed-patient notice, the per-category limit notices, the provisional "
                "acknowledgement and the out-of-time warning -- because the wording is what "
                "tells the agent a door has closed, and none of them asks for reasoning. The "
                "one replaced is ERROR_RETURN_MSG, which restated the JSON schema including "
                "its reasoning field.",
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
                "THE EXAMINER IS A SECOND MODEL WHEN ONE IS CONFIGURED, AND THE SCORE IS "
                "THEN NOT BIT-REPRODUCIBLE. The release resolves a free-text request with "
                "its own LLMMapper/LLMParser, gpt-4.1 by default; with engine.simulator "
                "enabled this adapter reproduces that DESIGN but not necessarily that "
                "MODEL -- the mapper is named per run in engine.simulator.by_dataset, and "
                "results are not comparable to published numbers on that axis. The split is "
                "the release's either way: the model picks which recorded keys the request "
                "resolves to, and the finding is still rendered from the case, so it cannot "
                "invent a result. Two runs of the "
                "same system can now disagree because the mapper did; temperature 0 and a "
                "seed narrow that and do not remove it. Which model mapped, and how often it "
                "had to be retried, is in the run's Simulators sheet and in each task's "
                "simulator_calls.jsonl. With no simulator configured the request is matched "
                "lexically instead: reproducible, but a request phrased unusually is answered "
                "'no finding recorded' when the case does record it.",
                "The examiner holds the whole case except its diagnosis and differentials, "
                "which are not disclosable through any action. It is not held back from a "
                "confirmatory investigation that names the condition -- ordering one is how a "
                "viva candidate confirms an answer, and the lexical environment discloses the "
                "same finding to the same request.",
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
