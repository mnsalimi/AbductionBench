"""AthenaBench: threat-actor attribution as evidence arrives, turn by turn.

Source: ``resources/athena_bench_dataset.json`` (57 cases, shipped with the
code -- there is no public download). Each case is one threat actor (the gold
``gold_label``, e.g. "TraderTraitor", "APT29") and 4-6 turns; each turn adds a
few pieces of evidence (profile, TTPs with ATT&CK ids, incidents) and asks the
same question again: which actor best accounts for the activity so far?

**How it is run -- SEQUENTIAL delivery.** The model is not asked once over the
whole file. Turn 1 shows turn 1's evidence and asks for the culprit; turn 2
shows turn 2's evidence, with the conversation so far -- every earlier piece of
evidence and every earlier prediction -- and asks again; and so on to the
case's last turn. The model cannot ask for anything: the order of the evidence
is the dataset's, which is what makes this sequential and not interactive.

**The prompts are this suite's.** The release ships a question per turn and no
system prompt; the question is used verbatim, the framing around it is ours.

**Scoring -- one judge verdict per turn.** Actor names have aliases (Sednit is
APT28, UNC2452 is APT29's SolarWinds cluster, "Lazarus" and "Lazarus group"),
so no turn is scored by string match: every turn's prediction is judged
against the gold, and every earlier prediction is judged against the final one
(a prediction the same as the final one once normalised is not sent). Metrics
(core/episode_metrics.py):

* ``final_answer_accuracy``           the last turn's prediction is the gold
* ``turns_to_correctness``            first turn judged correct; ``None`` if none
* ``average_sample_accuracy``         mean of the per-turn correctness
* ``turns_to_final_output``           turns the episode took
* ``turns_to_first_final_prediction`` first turn already predicting the final answer
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from ..core.adapter import SkippedDataset
from ..core.metrics import MISSING_METRIC
from ..core.types import AdapterDocumentation, ChatMessage, ModelResponse, SampleScore, SampleSpec
from ._base import PooledDatasetAdapter
from ._interactive import InteractiveMixin
from ._prompting import ANSWER_CLOSE, ANSWER_OPEN, mode_instruction, requirements_block

FILE_NAME = "athena_bench_dataset.json"
PACKAGED = Path(__file__).resolve().parent / "resources" / FILE_NAME

_ANSWER = re.compile(r"(?is).*<answer>\s*(?P<answer>.*?)\s*</answer>")
_NORMALISE = re.compile(r"[^a-z0-9]+")
#: Words that do not tell two actor names apart ("Lazarus" / "Lazarus Group").
_FILLER = frozenset({"the", "group", "apt", "team", "actor", "threat", "cluster"})


def _normalise(name: str | None) -> str:
    words = _NORMALISE.sub(" ", (name or "").lower()).split()
    kept = [w for w in words if w not in _FILLER] or words
    return " ".join(kept)


def parse_prediction(text: str | None) -> str | None:
    """The actor a turn's reply names: its answer block, else its last line."""
    body = (text or "").strip()
    if not body:
        return None
    match = _ANSWER.match(body)
    if match:
        answer = match.group("answer").strip()
    else:
        answer = body.splitlines()[-1].strip()
    answer = answer.strip().strip("*`\"'").strip()
    return answer[:200] or None


class AthenaBenchAdapter(InteractiveMixin, PooledDatasetAdapter):
    """Culprit prediction after every turn of cumulative threat evidence."""

    adapter_version = "1.0"
    data_delivery_mode = "sequential"
    authors_prompt = False
    #: One answer block per turn -- and the suite's rule for every
    #: interactive and sequential dataset: io, native reasoning off.
    io_only = True
    objective_metrics = False
    selection_cardinality = None
    primary_metric = "final_answer_accuracy"
    #: The longest case has 6 turns; the engine stops at this many anyway.
    max_turns = 8

    SYSTEM = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. You are a cyber threat "
        "intelligence analyst attributing an intrusion. Evidence about the activity arrives "
        "over several turns; after each turn, name the threat actor that best accounts for "
        "all of the evidence seen so far."
    )

    _REQUIREMENTS = (
        "name exactly one threat actor (a group or cluster name), not a list",
        "use the name the actor is best known by in threat-intelligence reporting",
        "base the answer on all the evidence given so far, not only this turn's",
        "you may keep or change your previous answer as the evidence warrants",
    )

    _FORMAT = (
        "Reply with exactly one answer block and nothing else:\n"
        f"{ANSWER_OPEN}threat actor name{ANSWER_CLOSE}"
    )

    #: "Only when sure": asked plainly, gpt-oss-120b knows TA571 is not
    #: Sandworm, but told that aliases count it scored the pair 1 five times
    #: in five -- an invitation to match aliases is read as permission to
    #: guess one.
    _CRITERIA = (
        "The candidate is correct if it names the same threat actor as the reference. Spelling "
        "variants and an added or dropped 'group' count. A different name counts ONLY if it is "
        "a well-documented alias of the same group that you are certain of (for example APT28 "
        "/ Fancy Bear / Sednit / Sofacy, or APT29 / Cozy Bear / Midnight Blizzard). Two names "
        "are NOT the same group merely because both are from the same country, share tooling "
        "or targets, or overlap in some reporting. A different group, a sub-group named "
        "separately, or a broader umbrella that does not identify the reference does not "
        "count. If you are not sure the two names are the same group, score 0."
    )

    # ------------------------------------------------------------------ #
    # data
    # ------------------------------------------------------------------ #

    def load_items(self) -> list[dict[str, Any]]:
        # The dataset lives where every dataset does, data/<id>/, and is put
        # there from the copy shipped with the code the first time.
        target = Path(self.context.data_dir) / FILE_NAME
        if not target.exists():
            if not PACKAGED.exists():
                raise SkippedDataset(f"AthenaBench file not found: {target} (nor {PACKAGED})")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(PACKAGED, target)
        items = json.loads(target.read_text(encoding="utf-8"))
        items = [
            item for item in items
            if str(item.get("gold_label") or "").strip() and item.get("turns")
        ]
        if not items:
            raise SkippedDataset(f"no usable cases in {target}")
        self.split_used = f"{FILE_NAME} ({len(items)} cases); no official split exists"
        return items

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        turns = sorted(item["turns"], key=lambda t: int(t.get("turn") or 0))
        turns = [
            {"question": str(t.get("question") or "").strip(),
             "evidences": [str(e).strip() for e in (t.get("evidences") or []) if str(e).strip()]}
            for t in turns
        ]
        turns = [t for t in turns if t["evidences"]]
        if not turns:
            return None
        gold = str(item["gold_label"]).strip()
        return SampleSpec(
            sample_id=f"athena-{item.get('sample_id', index)}",
            fields={
                "observation": self._evidence_block(turns[0]["evidences"]),
                "question": turns[0]["question"],
            },
            reference={"gold": gold},
            task_kind="generation",
            max_tokens=256,
            metadata={
                "case_index": item.get("sample_id", index),
                "n_turns": len(turns),
                "gold": gold,
                # The later turns, for the environment. Underscored: never
                # written into a record as metadata.
                "_turns": turns,
            },
        )

    # ------------------------------------------------------------------ #
    # the sequence
    # ------------------------------------------------------------------ #

    @staticmethod
    def _evidence_block(evidences: list[str]) -> str:
        return "\n".join(f"- {e}" for e in evidences)

    def _turn_message(self, turns: list[dict[str, Any]], index: int) -> str:
        turn = turns[index]
        block = [
            f"Turn {index + 1} of {len(turns)}.",
            f"{'Evidence' if index == 0 else 'New evidence this turn'}:\n"
            f"{self._evidence_block(turn['evidences'])}",
            f"Task: {turn['question']}",
            requirements_block(list(self._REQUIREMENTS)),
            mode_instruction(self.context.modes),
            self._FORMAT,
        ]
        return "\n\n".join(b for b in block if b)

    def interactive_start(self, sample: SampleSpec) -> tuple[list[ChatMessage], dict[str, Any]]:
        turns = sample.metadata["_turns"]
        state = {"turns": turns, "next": 1, "predictions": []}
        messages = [
            ChatMessage(role="system", content=self.SYSTEM),
            ChatMessage(role="user", content=self._turn_message(turns, 0)),
        ]
        return messages, state

    def interactive_step(
        self, sample: SampleSpec, state: dict[str, Any], assistant_text: str
    ) -> str | None:
        state["predictions"].append(parse_prediction(assistant_text))
        index = state["next"]
        if index >= len(state["turns"]):
            return None
        state["next"] = index + 1
        # The history is the conversation itself: every earlier turn's
        # evidence and the model's prediction after it are the messages
        # above this one, sent again on every request.
        return self._turn_message(state["turns"], index)

    # ------------------------------------------------------------------ #
    # scoring
    # ------------------------------------------------------------------ #

    def _predictions(self, sample: SampleSpec, score: SampleScore | None = None) -> list[str | None]:
        stored = (score.details or {}).get("turn_predictions") if score is not None else None
        if stored is not None:
            return list(stored)
        state = sample.metadata.get("_episode_state") or {}
        if state.get("predictions") is not None:
            return list(state["predictions"])
        transcript = sample.metadata.get("_transcript") or []
        return [parse_prediction(m.get("content")) for m in transcript if m.get("role") == "assistant"]

    def score(
        self,
        sample: SampleSpec,
        response: ModelResponse,
        *,
        output_contract: dict[str, Any] | None = None,
    ) -> SampleScore:
        predictions = self._predictions(sample)
        n_turns = int(sample.metadata.get("n_turns") or len(predictions) or 1)
        final = predictions[-1] if predictions else None
        details = {
            "gold": str(sample.reference["gold"]),
            "turn_predictions": predictions,
            "n_turns": n_turns,
            # Filled by the judge; until then every turn counts as wrong.
            "turn_correct": [0] * len(predictions),
            "turn_matches_final": self._matches_final_lexically(predictions),
        }
        metrics = self._metrics(predictions, details["turn_correct"],
                                details["turn_matches_final"], n_turns)
        return SampleScore(
            metrics=metrics,
            prediction=final,
            parse_ok=final is not None,
            details=details,
        )

    @staticmethod
    def _matches_final_lexically(predictions: list[str | None]) -> list[int | None]:
        final = _normalise(predictions[-1]) if predictions and predictions[-1] else ""
        if not final:
            return [None] * len(predictions)
        return [1 if p and _normalise(p) == final else None for p in predictions]

    @staticmethod
    def _metrics(
        predictions: list[str | None],
        correct: list[int],
        matches_final: list[int | None],
        n_turns: int,
    ) -> dict[str, Any]:
        turns_taken = len(predictions)
        first_correct = next((i + 1 for i, c in enumerate(correct) if c), None)
        first_final = next((i + 1 for i, m in enumerate(matches_final) if m), None)
        return {
            "final_answer_accuracy": float(correct[-1]) if correct else 0.0,
            "turns_to_correctness": float(first_correct) if first_correct else MISSING_METRIC,
            # Over the case's turns: a turn the episode never reached is not correct.
            "average_sample_accuracy": sum(correct) / max(n_turns, turns_taken, 1),
            "turns_to_final_output": float(turns_taken),
            "turns_to_first_final_prediction": (
                float(first_final) if first_final else MISSING_METRIC
            ),
        }

    def judge_requests(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore
    ) -> dict[str, dict[str, Any]]:
        """One verdict per turn against the gold, and per earlier turn against
        the final prediction (only where the two differ once normalised)."""
        predictions = self._predictions(sample, score)
        gold = str(sample.reference["gold"])
        requests: dict[str, dict[str, Any]] = {}
        for i, prediction in enumerate(predictions):
            if prediction:
                requests[f"gold:{i}"] = {"candidate": prediction, "gold": gold,
                                         "criteria": self._CRITERIA}
        final = predictions[-1] if predictions else None
        if final:
            for i, prediction in enumerate(predictions[:-1]):
                if prediction and _normalise(prediction) != _normalise(final):
                    requests[f"final:{i}"] = {"candidate": prediction, "gold": final,
                                              "criteria": self._CRITERIA}
        return requests

    def apply_judges(
        self,
        sample: SampleSpec,
        response: ModelResponse,
        score: SampleScore,
        verdicts: dict[str, Any],
    ) -> SampleScore:
        predictions = self._predictions(sample, score)
        details = dict(score.details or {})
        n_turns = int(details.get("n_turns") or sample.metadata.get("n_turns") or len(predictions))
        correct = [
            1 if (v := verdicts.get(f"gold:{i}")) is not None and v.positive else 0
            for i in range(len(predictions))
        ]
        matches = self._matches_final_lexically(predictions)
        for i in range(len(predictions)):
            verdict = verdicts.get(f"final:{i}")
            if verdict is not None:
                matches[i] = 1 if verdict.positive else 0
        judged = sum(1 for k in verdicts if k.startswith("gold:"))
        details.update(
            turn_predictions=predictions,
            turn_correct=correct,
            turn_matches_final=matches,
            turn_verdicts_missing=sum(1 for p in predictions if p) - judged,
        )
        return SampleScore(
            metrics=self._metrics(predictions, correct, matches, n_turns),
            prediction=score.prediction,
            parse_ok=score.parse_ok,
            details=details,
        )

    # judge_request is not used: judge_requests (several per sample) is.
    def judge_request(self, sample, response, score):  # noqa: ANN001
        return None

    def documentation(self) -> AdapterDocumentation:
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="AthenaBench",
            domain="Cybersecurity: Threat-Actor Attribution",
            source_url="local file (resources/athena_bench_dataset.json)",
            processing_mode="Generation",
            split_used=self.split_used,
            abductive_subset=(
                "The attribution step: infer the threat actor that best explains accumulating "
                "intrusion evidence, re-inferred after every new batch of evidence."
            ),
            sampling_procedure=(
                f"all {self.split_size} cases, then drawn by {self.sampling_note()}"
            ),
            metrics_description={
                "final_answer_accuracy": "(PRIMARY, higher is better, 0-1) the last turn's "
                "prediction judged the same actor as the gold",
                "turns_to_correctness": "(lower is better) first turn whose prediction was "
                "judged correct; None when no turn was, and left out of the mean",
                "average_sample_accuracy": "(higher is better, 0-1) mean per-turn correctness "
                "over the case's turns",
                "turns_to_final_output": "turns the episode took",
                "turns_to_first_final_prediction": "(lower is better) first turn whose "
                "prediction was already the final one (judged, aliases count)",
            },
            primary_metric="final_answer_accuracy",
            decisions=[
                "Sequential delivery: each turn's evidence arrives in the dataset's order, with "
                "the earlier evidence and predictions as the conversation history.",
                "The system prompt, the requirements and the answer format are this suite's; "
                "each turn's question is the dataset's, verbatim.",
                "Every turn is judged by the LLM judge against the gold (actor aliases count); "
                "no turn is scored by string match.",
                "Run io only, with the model's native reasoning switched off, like every "
                "interactive and sequential dataset.",
            ],
            caveats=[
                "No candidate list is shown, so the answer space is open; the judge decides "
                "alias equivalence and can err on obscure tracking names.",
            ],
            statistics=self.base_statistics(),
        )
