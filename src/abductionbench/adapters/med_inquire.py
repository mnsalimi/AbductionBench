"""Med-Inquire (EvoClinician): diagnosis from a clinical case.

Source: https://github.com/yf-he/EvoClinician

The repository ships ``data/test-00000-of-00001.jsonl`` -- 915 published case
reports, each with the narrative (``case_information``), the physical
examination, the diagnostic work-up, the ``final_diagnosis``, and four candidate
diagnoses with the correct letter.

**What is abductive here.** Naming the diagnosis that best explains the findings.
EvoClinician's own harness wraps this in an interactive inquiry loop where an
agent asks for information before committing; the single-turn adaptation gives
the model the case evidence and asks for the diagnosis, which is the abductive
core of that loop.

Because the table asks for **generation**, the default asks the model to name the
diagnosis in free text and scores it against ``final_diagnosis`` (with the
provided options ignored).  ``options.subtask = selection`` uses the four
candidate diagnoses instead, and ``options.evidence`` controls how much of the
work-up is revealed -- the case narrative alone is a much harder abduction than
narrative + examination + tests.
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
from ._base import PooledDatasetAdapter, selection_score, text_match_score
from ._interactive import EvidenceStore, InteractiveMixin, parse_action


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
    objective_metrics = True
    selection_cardinality = "single"
    hypothesis_modes = ("generation", "selection",)
    hypothesis_mode_options = {
        "generation": {'subtask': 'generation'},
        "selection": {'subtask': 'selection'},
    }
    table_hypothesis_mode = "Generation"
    hypothesis_mode_justification = (
        "The Med-Inquire test file is DiagnosisArena's release, in which each case carries "
        "four answer options with one correct diagnosis, so a closed-set selection task is "
        "defined by the data itself."
    )
    primary_metric = "diagnosis_match"

    @property
    def _subtask(self) -> str:
        return str(self.context.option("subtask", "generation"))

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

    def _release_actor_prompt(self) -> str:
        """The Actor system prompt as published, read from the cloned repo."""
        if getattr(self, "_actor_prompt", None):
            return self._actor_prompt
        path = self.context.data_dir / "repo" / "evoclinician" / "agents" / "actor.py"
        prompt = ""
        if path.exists():
            source = path.read_text(encoding="utf-8", errors="replace")
            match = re.search(r"base_prompt: str = \(\n(.*?)\n    \)", source, re.S)
            if match:
                # The prompt is a run of adjacent string literals; evaluate just
                # that expression so the text is the release's, character for
                # character, rather than a transcription of it.
                try:
                    prompt = eval(  # noqa: S307 - a literal from a file we cloned
                        "(" + match.group(1) + ")", {"__builtins__": {}}, {}
                    )
                except Exception:  # noqa: BLE001 - fall back to our own wording
                    prompt = ""
        self._actor_prompt = prompt or self.system_prompt
        return self._actor_prompt

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
        opening = (
            "A new patient has presented. You have not yet taken a history.\n\n"
            f"Opening statement from the patient: {sample.fields.get('observation', '')[:600]}"
        )
        return (
            [
                ChatMessage(role="system", content=self._release_actor_prompt()),
                ChatMessage(role="user", content=opening),
            ],
            {"evidence": store, "counts": {}},
        )

    def interactive_step(
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
        return text_match_score(
            response,
            gold=sample.reference["gold"],
            accepted=_abbreviations(sample.reference["gold"]),
            output_contract=output_contract,
            primary="diagnosis_match",
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
                "diagnosis_match": "1 if the answer equals or contains the gold diagnosis (or a "
                "parenthesized abbreviation the dataset itself gives)",
                "exact_match": "strict normalized equality with the gold diagnosis",
                "token_f1": "bag-of-tokens F1 against the gold diagnosis",
                "rouge_l": "LCS F-measure against the gold diagnosis",
                "diagnosis_match_judged": "LLM-judge verdict on semantic equivalence (only when "
                "engine.judge.enabled)",
                "accuracy": "selection subtask: 1 if the chosen option letter is correct",
            },
            primary_metric="accuracy" if self._subtask == "selection" else "diagnosis_match",
            decisions=[
                "Default is free-text generation, matching this dataset's processing mode; the "
                "four provided candidates are only used with options.subtask = selection.",
                "Accepted the abbreviation the gold string itself parenthesizes (e.g. 'Cutaneous "
                "meningeal heterotopia (CMH)' also accepts 'CMH') -- no synonyms are invented.",
                "options.evidence controls how much work-up is shown: full (default), "
                "case_and_exam, or case_only, so the same items can probe abduction under "
                "progressively less evidence.",
                "max_tokens=768: cases are long and diagnoses need differential reasoning, but "
                "the answer itself is a short disease name.",
            ],
            caveats=[
                "SAME UNDERLYING ITEMS AS diagnosisarena: this test file is the DiagnosisArena "
                "release re-used by EvoClinician. The two rows are one dataset viewed two ways "
                "(here without the diagnostic work-up), not independent evidence.",
                "Case reports are published literature and may be memorized by large models.",
                "String matching under-credits correct paraphrases; enable engine.judge for "
                "diagnosis_match_judged.",
            ],
            statistics={
                **self.base_statistics(),
                "subtask": self._subtask,
                "evidence": str(self.context.option("evidence", "case_and_exam")),
            },
        )


def _abbreviations(gold: str) -> list[str]:
    """Surface forms the gold string itself provides: text before/inside parentheses."""
    out: list[str] = []
    if "(" in gold and ")" in gold:
        head = gold.split("(")[0].strip()
        inner = gold[gold.index("(") + 1 : gold.rindex(")")].strip()
        out.extend(part for part in (head, inner) if part)
    return out
