"""Med-Inquire (EvoClinician): diagnosis from a clinical case.

Source: https://github.com/yf-he/EvoClinician

The repository ships ``data/test-00000-of-00001.jsonl`` -- 915 published case
reports, each with the narrative (``case_information``), the physical
examination, the diagnostic work-up, the ``final_diagnosis``, and four candidate
diagnoses with the correct letter.

**What is abductive here.** Naming the diagnosis that best explains the findings.
EvoClinician's own harness wraps this in an interactive inquiry loop where an
agent asks for information before committing, and that loop is what runs here:
the model asks the patient, orders examinations and tests, and commits, with the
environment answering only from the case's recorded work-up.

Because the table asks for **generation**, the default asks the model to name the
diagnosis in free text and scores it against ``final_diagnosis`` (with the
provided options ignored).  ``options.subtask = selection`` uses the four
candidate diagnoses instead, and ``options.evidence`` controls how much of the
work-up is revealed -- the case narrative alone is a much harder abduction than
narrative + examination + tests.

**The protocol prompt is this suite's, adapted rather than quoted.**  The
Actor's three actions and their JSON shape, the budgets, the cost-awareness and
differential-driven task rules, and the instruction to commit once the evidence
supports a diagnosis are the release's and are reproduced in substance.  Its
request that the actor *explain the decision value* of each proposed test is
not: this dataset runs ``io`` only, and asking for an explanation would
contradict the mode instruction in the same prompt.
"""

from __future__ import annotations

import re
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
from ._base import (
    PooledDatasetAdapter,
    apply_judged_metric,
    judged_only_score,
    selection_score,
)
from ._interactive import EvidenceStore, InteractiveMixin, parse_action
from ._prompting import ProtocolParts, build_protocol_messages


def _sentences(text: str, prefix: str = "") -> dict[str, str]:
    """Split a case section into individually disclosable findings.

    Med-Inquire stores its case as prose, so a "finding" is a sentence: the unit
    a patient would answer with, and small enough that answering one question
    does not hand over the whole case file.
    """
    out: dict[str, str] = {}
    for index, chunk in enumerate(re.split(r"(?<=[.;])\s+|\n+", text or "")):
        sentence = chunk.strip(" -\t")
        if len(sentence) < 12:
            continue
        # No dot in the key: a dotted key would make every sentence of one
        # section a child of the same parent, and the reveal would hand back the
        # whole section the first time any of it matched.
        key = f"{prefix}{index}" if prefix else str(index)
        out[key] = sentence
    return out

REPO_URL = "https://github.com/yf-he/EvoClinician"
OPTION_KEYS = ("option_a", "option_b", "option_c", "option_d")


class MedInquireAdapter(InteractiveMixin, PooledDatasetAdapter):
    """Free-text (default) or four-way diagnosis of a published case report."""

    adapter_version = "1.0"

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. You are given a published case "
        "report with the diagnostic work-up withheld. State the diagnosis that best explains "
        "the presenting picture you can see."
    )
    data_delivery_mode = "interactive"
    #: The protocol prompt below is this suite's, adapted from the release's
    #: Actor rather than quoted from it, so this is False. One prompt mode
    #: still, for the reason io_only states.
    authors_prompt = False
    io_only = True
    #: The generation subtask -- the only one run -- asks for a disease name with
    #: no candidate list, so a correct answer routinely differs from the gold in
    #: wording and only the judge can score it. prepare() flips this back to True
    #: if the selection subtask is ever selected, where a label is checkable.
    objective_metrics = False
    #: NOT a selection task: no candidate list is ever shown, the answer is
    #: a disease name, written free-form. Declaring "single" made the engine schedule it as SCS and
    #: append "Select exactly one hypothesis / Answer: 1" to a prompt with no
    #: hypotheses in it -- which taught the model to answer with a number.
    selection_cardinality = None
    hypothesis_modes = ("generation",)
    hypothesis_mode_options = {
        "generation": {'subtask': 'generation'},
    }
    table_hypothesis_mode = "Generation"
    hypothesis_mode_justification = (
        "Med-Inquire is run as interactive GENERATION only: the point of the benchmark is the "
        "questioning -- the model asks for findings before committing to a diagnosis. Its "
        "selection variant is NOT run, and the reason matters: the Med-Inquire test file IS "
        "DiagnosisArena's release, so a selection task here would be DiagnosisArena's task over "
        "DiagnosisArena's cases, scored twice."
    )
    primary_metric = "diagnosis_judged"

    primary_metric_by_mode = {
        # The two tasks are scored by different things, so each names
        # the metric it actually produces rather than inheriting one.
        "generation": "diagnosis_judged",
        "selection": "accuracy",
    }

    @property
    def _subtask(self) -> str:
        return str(self.context.option("subtask", "generation"))

    def prepare(self) -> None:
        # Selection is scored against an answer key; generation is judged.
        self.objective_metrics = self._subtask == "selection"
        super().prepare()

    def load_items(self) -> list[dict[str, Any]]:
        root = C.ensure_git_repo(
            REPO_URL, self.context.data_dir / "repo", offline=self.context.offline
        )
        candidates = C.find_files(root / "data", ["*.jsonl"]) or C.find_files(root, ["*.jsonl"])
        found = C.pick_split_file(candidates)
        if not found:
            raise SkippedDataset("no Med-Inquire split file found in the repository")
        path, split = found
        rows = C.read_jsonl(path)
        if not rows:
            raise SkippedDataset(f"{path} contained no rows")
        self.split_used = f"{split} ({len(rows)} case reports) from {path.name}"
        return rows

    def _evidence(self, item: dict[str, Any]) -> tuple[str, str]:
        """Return ``(observation, context)`` for the configured evidence level."""
        level = str(self.context.option("evidence", "case_and_exam"))
        narrative = C.normalize_whitespace(item.get("case_information"))
        examination = C.normalize_whitespace(item.get("physical_examination"))
        tests = C.normalize_whitespace(item.get("diagnostic_tests"))
        if level == "case_only":
            return narrative, ""
        if level == "case_and_exam":
            context = f"Physical examination:\n{examination}" if examination else ""
            return narrative, context
        parts = []
        if examination:
            parts.append(f"Physical examination:\n{examination}")
        if tests:
            parts.append(f"Diagnostic work-up:\n{tests}")
        return narrative, "\n\n".join(parts)

    # ------------------------------------------------------------------ #
    # the inquiry loop -- EvoClinician's own Med-Inquire protocol
    # ------------------------------------------------------------------ #

    #: The release's action vocabulary (evoclinician/med_inquire/types.py).
    ACTIONS = ("askquestion", "ordertest", "submitdiagnosis")
    max_turns = 12
    category_limits = {"askquestion": 8, "ordertest": 6}

    #: The release's action vocabulary, written out as the Actor's own three
    #: choices. The names are the release's (``evoclinician/med_inquire/types.py``)
    #: because the environment routes on them.
    _ACTION_HELP = (
        ("AskQuestion", "ask the patient one question about their history or symptoms."),
        ("OrderTest", "order one physical examination or diagnostic test. Name it exactly."),
        ("SubmitDiagnosis", "commit to the diagnosis. This ends the consultation."),
    )

    #: The release runs two agents behind the two request kinds: a Patient who
    #: answers from the history, and an Examination agent who returns recorded
    #: findings.  They get separate briefs *and* separate conversations, so a
    #: question to the patient cannot return an imaging report and the patient
    #: never learns what the work-up found.
    _PATIENT_BRIEF = (
        "You are a patient being interviewed by a doctor. Answer in the first "
        "person, in one or two sentences of plain lay language.\n\n"
        "Your history, as you would tell it:\n{history}\n\n"
        "Rules you must follow:\n"
        "- Answer only from your history above, and only what was asked.\n"
        "- If your history does not cover the question, say so plainly in your "
        "own words -- do not invent a symptom, a date or a number.\n"
        "- Never name or guess a diagnosis, a condition or a disease. You do "
        "not know what you have.\n"
        "- You know nothing about examinations, tests or results. If asked "
        "about one, say the doctor would have to check.\n"
        "- Do not use clinical terminology, and do not volunteer what you were "
        "not asked."
    )
    _EXAMINER_BRIEF = (
        "You report the findings recorded for one case to the doctor working "
        "it up. You are not the patient and you do not interpret; you report "
        "what the record holds.\n\n"
        "The recorded examination and work-up:\n{findings}\n\n"
        "Rules you must follow:\n"
        "- The doctor names one examination or test. Report what the record "
        "holds for it, in one or two lines, with the numbers and units as "
        "recorded.\n"
        "- If the record holds nothing for what was named, reply with exactly: "
        "NOT AVAILABLE\n"
        "- Never invent a result, and never report a normal finding the record "
        "does not state.\n"
        "- Report only the test that was named. Do not volunteer the rest of "
        "the work-up.\n"
        "- Never name or suggest a diagnosis."
    )

    def _briefs(self, sample: SampleSpec) -> dict[str, str]:
        """The hidden material each agent may see, and nothing else.

        Built here, from the case, so that ``interactive_step`` never reaches
        into the sample: the split between what the patient knows and what the
        examiner knows is made once, in one place, where it can be tested.
        The final diagnosis is in neither -- it is the answer.
        """
        case = sample.metadata.get("_case") or {}
        history = C.normalize_whitespace(case.get("case_information")) or "(nothing recorded)"
        findings = "\n\n".join(
            part for part in (
                f"Physical examination:\n{C.normalize_whitespace(case.get('physical_examination'))}"
                if case.get("physical_examination") else "",
                f"Diagnostic tests:\n{C.normalize_whitespace(case.get('diagnostic_tests'))}"
                if case.get("diagnostic_tests") else "",
            ) if part
        ) or "(nothing recorded)"
        return {
            "askquestion": self._PATIENT_BRIEF.format(history=history),
            "ordertest": self._EXAMINER_BRIEF.format(findings=findings),
        }

    def interactive_start(self, sample: SampleSpec) -> tuple[list[ChatMessage], dict[str, Any]]:
        case = sample.metadata.get("_case") or {}
        store = EvidenceStore()
        # The Patient answers from the case history; the Examination agent
        # answers test orders from the recorded work-up. Separate stores, so a
        # question to the patient cannot return an imaging report.
        store.categories["askquestion"] = _sentences(case.get("case_information", ""))
        store.categories["ordertest"] = {
            **_sentences(case.get("physical_examination", ""), prefix="exam"),
            **_sentences(case.get("diagnostic_tests", ""), prefix="test"),
        }
        parts = ProtocolParts(
            system=self.system_prompt,
            context="A new patient has presented. You have not yet taken a history.",
            observation=(
                "Opening statement from the patient: "
                f"{sample.fields.get('observation', '')[:600]}"
            ),
            instructions=(
                "Work out what is wrong with this patient. Ask the patient what you need to "
                "know, order the examinations and tests that would settle it, and commit to "
                "a diagnosis."
            ),
            requirements=[
                "take exactly one action per turn",
                "keep a short differential in mind and act on what would separate its "
                "entries",
                "when several actions would help, prefer the one that is quick and "
                "inexpensive -- time and resources are part of the task",
                "rule out the conditions that would be dangerous to miss before the "
                "unlikely ones",
                f"you may ask at most {self.category_limits['askquestion']} questions and "
                f"order at most {self.category_limits['ordertest']} tests",
                "submit the diagnosis as soon as the evidence supports one",
            ],
            actions=list(self._ACTION_HELP),
            output_format=(
                "Reply with one JSON object and nothing else:\n"
                '{"action_type": "AskQuestion" | "OrderTest" | "SubmitDiagnosis", '
                '"action_text": "<your question, test, or diagnosis>"}'
            ),
        )
        return build_protocol_messages(parts, self.context.modes), {
            "evidence": store,
            "counts": {},
            "_briefs": self._briefs(sample),
        }

    async def interactive_step(
        self, sample: SampleSpec, state: dict[str, Any], assistant_text: str
    ) -> str | None:
        action = parse_action(assistant_text, actions=self.ACTIONS)
        # The release's field is action_text, not query; accept both.
        if not action.query:
            match = re.search(r'"action_text"\s*:\s*"([^"]*)"', assistant_text or "")
            if match:
                action.query = match.group(1)
        if action.action == "submitdiagnosis":
            return None
        if not action.action:
            state["parse_errors"] = state.get("parse_errors", 0) + 1
            if state["parse_errors"] > 2:
                return None
            return (
                'Return exactly one JSON object: {"action_type": "AskQuestion"|"OrderTest"|'
                '"SubmitDiagnosis", "action_text": "..."}'
            )

        self.bump(state, action.action)
        if self.over_limit(state, action.action):
            return "No further actions of that kind are available. Please submit your diagnosis."
        if self.simulator is not None:
            # The release's Patient and Examination agents, each answering from
            # its own half of the case.
            return await self.simulate(
                state,
                brief=state["_briefs"][action.action],
                request=action.query_text,
                role=action.action,
            )
        store: EvidenceStore = state["evidence"]
        reply = store.reveal(action.action, action.query_text, limit=3)
        if reply:
            return reply
        if action.action == "ordertest":
            # The release's ExaminationAgent returns exactly this for a test the
            # case does not record.
            return "NOT AVAILABLE"
        return "The patient does not report anything about that."

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        diagnosis = C.normalize_whitespace(item.get("final_diagnosis"))
        observation, context = self._evidence(item)
        if not diagnosis or not observation:
            return None

        if self._subtask == "selection":
            options = [C.normalize_whitespace(item.get(key)) for key in OPTION_KEYS]
            options = [option for option in options if option]
            gold_letter = C.normalize_whitespace(item.get("right_option")).upper()
            labels = C.letter_labels(len(options))
            if len(options) < 2 or gold_letter not in labels:
                return None
            return SampleSpec(
                sample_id=C.stable_id("medinq", item.get("id", index)),
                fields={
                    "observation": observation,
                    "context": context,
                    "question": "Which diagnosis best explains this presentation?",
                    "options": options,
                    "option_labels": labels,
                },
                reference={"gold_label": gold_letter, "gold": diagnosis},
                task_kind="selection",
                max_tokens=768,
                metadata={"id": item.get("id"), "subtask": "selection", "_case": dict(item)},
            )

        return SampleSpec(
            sample_id=C.stable_id("medinq", item.get("id", index)),
            fields={
                "observation": observation,
                "context": context,
                "question": "What is the single most likely diagnosis?",
                "instructions": (
                    "Name the specific diagnosis (the disease entity), not a category or a "
                    "differential list."
                ),
            },
            reference={"gold": diagnosis},
            task_kind="generation",
            # A diagnosis name, after reading a full case: reasoning-heavy input,
            # short output.
            max_tokens=768,
            metadata={"id": item.get("id"), "subtask": "generation", "_case": dict(item)},
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
            metric="diagnosis_judged",
            output_contract=output_contract,
            details={"gold": str(sample.reference["gold"])[:300]},
        )

    def judge_request(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore
    ) -> dict[str, Any] | None:
        """Ask the judge whether a free-text diagnosis names the gold disease."""
        if sample.task_kind != "generation" or not response.text:
            return None
        return {
            "candidate": score.prediction or response.text[:600],
            "gold": sample.reference["gold"],
            "observation": sample.fields["observation"],
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

    def documentation(self) -> AdapterDocumentation:
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="Med-Inquire (EvoClinician)",
            domain="Healthcare: Interactive Diagnosis",
            source_url=REPO_URL,
            processing_mode="Generation",
            split_used=self.split_used,
            abductive_subset=(
                "The diagnostic inference itself: given case narrative, examination and "
                "work-up, name the disease that best explains them. EvoClinician's "
                "information-gathering loop is an interactive protocol and is not reproduced; "
                "what is evaluated is the hypothesis it exists to produce."
            ),
            sampling_procedure=self.sampling_note(),
            metrics_description={
                "best_of_n_<metric>":
                "Every metric also gets a best_of_n_ counterpart: per record, the repeat the "
                "judge scored highest. A diagnosis is free text with many correct surface forms, "
                "so a plurality over repeats is not meaningful and Best-of-N replaces it. The "
                "selection subtask, being checkable, gets self_consistency_ instead.",
                "diagnosis_judged": "(PRIMARY, higher is better, 0-1) LLM-judge verdict on whether "
                "the named diagnosis is the same disease entity as the gold, however written. "
                "1.0 when the judge affirms.",
                "parse_failure_rate": "(lower is better, 0-1) fraction of responses no diagnosis "
                "could be read from; these score 0 and are counted separately from being wrong.",
                "accuracy": "(PRIMARY, higher is better) selection subtask: 1 if the chosen option letter is correct",
            },
            primary_metric="accuracy" if self._subtask == "selection" else "diagnosis_judged",
            decisions=[
                "THE PROTOCOL PROMPT IS LOCALLY ADAPTED, NOT THE AUTHORS'. The Actor's three "
                "actions and their JSON shape, the budgets, the cost-awareness and "
                "differential-driven task rules and the stopping condition come from the "
                "release and are reproduced in substance; the wording is this suite's. Its "
                "request that the actor explain the decision value of each proposed test is "
                "dropped, because this task runs io only.",
                "Default is free-text generation, matching this dataset's processing mode; the "
                "four provided candidates are only used with options.subtask = selection.",
                "Scored by an LLM judge rather than by string comparison: the model names a "
                "disease with no candidate list in front of it, so a correct answer routinely "
                "differs from the gold in wording (synonym, eponym, abbreviation, subtype).",
                "options.evidence controls how much work-up is shown: full (default), "
                "case_and_exam, or case_only, so the same items can probe abduction under "
                "progressively less evidence.",
                "max_tokens=768: cases are long and diagnoses need differential reasoning, but "
                "the answer itself is a short disease name.",
            ],
            caveats=[
                "THE PATIENT AND THE EXAMINATION AGENT ARE A SECOND MODEL WHEN ONE IS "
                "CONFIGURED, AND THE SCORE IS THEN NOT BIT-REPRODUCIBLE. The release runs "
                "both as LLM agents; with engine.simulator enabled this adapter does the "
                "same, on gpt-4o-mini by default. They get separate briefs and separate "
                "conversations -- the patient is given the case history, the examination "
                "agent the recorded examination and work-up -- so a question to the patient "
                "cannot return an imaging report and the patient never learns what the "
                "work-up found. Neither is given final_diagnosis or the release's answer "
                "options. Two runs of the same system can now disagree because an agent "
                "did; temperature 0 and a seed narrow that and do not remove it. Which "
                "model answered is in the run's Simulators sheet and in each task's "
                "simulator_calls.jsonl. With no simulator configured the request is matched "
                "lexically against the case's sentences instead.",
                "A simulator that cannot be reached ends the episode as an ERROR rather "
                "than a wrong answer, so an outage lowers the sample count rather than the "
                "score.",
                "SAME UNDERLYING ITEMS AS diagnosisarena: this test file is the DiagnosisArena "
                "release re-used by EvoClinician, so the two datasets share their cases. They "
                "are NOT independent evidence, and a suite-level average over both counts those "
                "cases twice.",
                "What separates them is the task, and that is why both are kept: diagnosisarena "
                "is run as STATIC SELECTION over the four answer options the release ships, "
                "while med_inquire is run as INTERACTIVE GENERATION -- the model must ask for "
                "history, examination and investigations before naming a diagnosis, with no "
                "candidate list. The pair therefore measures what the questioning buys on the "
                "same cases; it does not measure two datasets.",
                "Neither is run in the other's mode: no open-ended variant of diagnosisarena, "
                "and no selection variant here, precisely because that would be the other "
                "dataset's task over the other dataset's cases.",
                "Case reports are published literature and may be memorized by large models.",
                "The judge is required for this dataset: with engine.judge disabled the "
                "generation subtask has no metric, and the run fails rather than reporting a "
                "string-overlap number that would not mean what it says.",
            ],
            statistics={
                **self.base_statistics(),
                "subtask": self._subtask,
                "evidence": str(self.context.option("evidence", "case_and_exam")),
            },
        )
