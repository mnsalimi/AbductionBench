"""UniADILR-HGc: name the hypothesis a body of premises supports.

Source: https://github.com/YuSheng-00/UniADILR

UniADILR generates one dataset covering several reasoning types and ships them
as parallel files -- ``abduction.jsonl``, ``deduction.jsonl`` and
``induction.jsonl`` under ``data/UniADILR-HGc/``.  **Only the abduction file is
used here.**  The other two are the same construction applied to deductive and
inductive inference, and including them would report non-abductive reasoning
under an abduction benchmark.  Every row of the abduction file carries
``reasoning_type: "abduction"``, and rows that do not are dropped, so the
restriction is enforced rather than assumed.

**The task.**  An item gives a numbered pool of sentences (``sent1`` ...
``sentN``), most of which are distractors, and a hypothesis that a small
subset of them supports -- the ``proof`` field names which, e.g.
``sent5 & sent13 -> Sarah is a talented programmer.``  The model sees the pool
and must state the hypothesis those premises point to.  What makes it
constrained abduction rather than deduction is the pool: the supporting
sentences must be found among many irrelevant ones before the hypothesis can be
formed.

**What is asked for.**  Not the claim, but the two statements that support it:
the answer is the pair of sentence numbers, e.g. ``5 13``.  The release's
``proof`` field names them -- ``sent5 & sent13 -> Sarah is a talented
programmer.`` -- so the answer key is the release's own, and the hard part of
the task is untouched: the supporting pair still has to be found among a pool
of mostly irrelevant statements.

**Why that rather than the claim.**  Asking for the claim made the answer free
text that only a judge could grade, and it graded the wrong thing: a model can
paraphrase the hypothesis convincingly from the pool's general subject matter
without ever locating the two statements that entail it.  A pair of numbers is
checkable outright.

**Only the two-premise items are used.**  Of the 500 abductive items, 466 have
exactly two supporting statements, 19 have one and 15 have three.  A prompt
that asks for exactly two numbers is unanswerable on the other 34, so they are
dropped rather than scored against an instruction they cannot satisfy; the
count is reported.

**Scoring.**  Order-independent exact match of the pair against the release's
own ``proof``: ``13 5`` and ``5 13`` are the same answer, and anything that is
not exactly two numbers is a parse failure rather than a wrong answer.  No
judge -- there is nothing here a judge could settle that a set comparison
cannot.
"""

from __future__ import annotations

import re
from typing import Any

from ..core.adapter import SkippedDataset
from ..core.metrics import aggregate_mean_metrics, extract_answer_span
from ..core.types import (
    AdapterDocumentation,
    ModelResponse,
    SampleScore,
    SampleSpec,
)
from . import _common as C
from ._base import PooledDatasetAdapter, unparsed_score

REPO_URL = "https://github.com/YuSheng-00/UniADILR.git"
#: The abductive split, and only it.
ABDUCTION_FILE = "data/UniADILR-HGc/abduction.jsonl"
_SENT_RE = re.compile(r"\bsent(\d+)\b")
#: A bare integer in the model's answer. The prompt asks for "5 13"; this also
#: reads "5, 13" and "sent5 sent13" without accepting prose that merely
#: contains numbers, because the count is checked immediately afterwards.
_NUMBER_RE = re.compile(r"\d+")


def _sent_number(name: str) -> int:
    """``sent13`` -> ``13``; the release's own statement numbering."""
    digits = re.findall(r"\d+", str(name))
    return int(digits[0]) if digits else 0


def _premise_ids(proof: str) -> set[int]:
    """The statement numbers on the LEFT of ``->`` in a proof.

    ``sent5 & sent13 -> Sarah is a talented programmer.`` gives ``{5, 13}``.
    Only the left side: the right side is the claim, and on an item whose claim
    text happened to contain a token like ``sent2`` it would otherwise be read
    as a third premise.
    """
    left = proof.split("->")[0] if "->" in proof else proof
    return {int(value) for value in _SENT_RE.findall(left)}


class UniADILRHGcAdapter(PooledDatasetAdapter):
    """Constrained abductive hypothesis generation over a pool of premises."""

    adapter_version = "1.0"

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. You are given a numbered pool of "
        "statements, most of which are irrelevant. Exactly two of them together support a "
        "single further claim. Identify which two."
    )
    answer_format = "two statement numbers"
    answer_constraints = (
        "give exactly two numbers",
        "separate them with a space, for example: 5 13",
        "output only the two numbers",
        "do not name the claim",
        "do not explain your reasoning",
        "do not use introductory phrases or commentary",
    )
    data_delivery_mode = "static"
    #: The answer is a pair of numbers drawn from the pool the prompt shows, so
    #: it is checkable outright: a set comparison against the release's own
    #: proof field settles it, and repeats can coincide, so self-consistency
    #: applies.
    objective_metrics = True
    selection_cardinality = None
    hypothesis_modes = ("generation",)
    hypothesis_mode_options = {"generation": {"subtask": "generation"}}
    table_hypothesis_mode = "Generation"
    primary_metric = "premise_set_match"

    # ------------------------------------------------------------------ #
    # data
    # ------------------------------------------------------------------ #

    def load_items(self) -> list[dict[str, Any]]:
        root = C.ensure_git_repo(
            REPO_URL, self.context.data_dir / "repo", depth=1, offline=self.context.offline
        )
        path = root / ABDUCTION_FILE
        if not path.is_file():
            raise SkippedDataset(
                f"UniADILR's abductive split is not in the clone: {path} does not exist"
            )
        rows = C.read_jsonl(path)
        if not rows:
            raise SkippedDataset(f"{path} contained no rows")
        # Enforced, not assumed: the deduction and induction files are the same
        # construction over other reasoning types and are not read at all, and a
        # row here that is not marked abduction is dropped.
        abductive = [
            row for row in rows
            if str(row.get("reasoning_type", "abduction")).lower() == "abduction"
        ]
        if not abductive:
            raise SkippedDataset(
                f"{path} has {len(rows)} rows but none are marked reasoning_type=abduction"
            )
        self.dropped_non_abductive = len(rows) - len(abductive)

        # Exactly two supporting statements, because that is what the prompt
        # asks for. 466 of the 500 items are two-premise; asking for two
        # numbers on a one- or three-premise item would make it unanswerable by
        # construction, so those are dropped and counted rather than scored
        # against an instruction they cannot satisfy.
        two_premise = []
        self.dropped_premise_counts: dict[int, int] = {}
        for row in abductive:
            premises = _premise_ids(row.get("proof") or "")
            if len(premises) == 2:
                two_premise.append(row)
            else:
                self.dropped_premise_counts[len(premises)] = (
                    self.dropped_premise_counts.get(len(premises), 0) + 1
                )
        if not two_premise:
            raise SkippedDataset(
                f"{path} has no items whose proof names exactly two supporting statements"
            )
        dropped = ", ".join(
            f"{count} with {n}" for n, count in sorted(self.dropped_premise_counts.items())
        )
        self.split_used = (
            f"{ABDUCTION_FILE}, the two-premise items ({len(two_premise)} of "
            f"{len(abductive)}; dropped {dropped} supporting statement(s)); the deduction "
            f"and induction files are not used"
        )
        return two_premise

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        context = item.get("context")
        gold_claim = C.normalize_whitespace(item.get("hypothesis"))
        proof = C.normalize_whitespace(item.get("proof"))
        premises = _premise_ids(proof)
        if not isinstance(context, dict) or not context or len(premises) != 2:
            return None
        # Numbered as the release numbers them -- `sent5` is statement 5 -- but
        # shown as a bare number, so the number the model reads is the number
        # the answer format asks it to write.
        statements = [
            f"{_sent_number(name)}. {C.normalize_whitespace(text)}"
            for name, text in sorted(context.items(), key=lambda kv: _sent_number(kv[0]))
            if C.normalize_whitespace(text)
        ]
        if len(statements) < 2:
            return None
        return SampleSpec(
            sample_id=C.stable_id("uniadilr", index),
            fields={
                "observation": "\n".join(statements),
                "question": (
                    "Exactly two of these statements together support a single further "
                    "claim. Which two?"
                ),
            },
            reference={
                # The answer key, from the release's own proof field.
                "premises": sorted(premises),
                "proof": proof,
                # The claim itself, kept for inspection; never scored against.
                "claim_unscored": gold_claim,
            },
            task_kind="generation",
            metadata={"n_statements": len(statements)},
        )

    # ------------------------------------------------------------------ #
    # scoring
    # ------------------------------------------------------------------ #

    def score(
        self,
        sample: SampleSpec,
        response: ModelResponse,
        *,
        output_contract: dict[str, Any] | None = None,
    ) -> SampleScore:
        """Exactly two numbers, compared as a set against the release's proof.

        "Exactly" is enforced rather than forgiven: an answer carrying one
        number, or three, has not answered the question that was asked, and
        counting it as a wrong pair would hide a model that cannot follow the
        format inside the score for a model that cannot find the premises.
        Those land in ``parse_failure_rate`` instead.
        """
        answer = extract_answer_span(response.text, output_contract)
        numbers = _NUMBER_RE.findall(answer or "")
        if len(numbers) != 2:
            return unparsed_score(
                ["premise_set_match", "premise_partial"],
                raw=(answer or response.text or "")[:200],
                n_numbers=len(numbers),
            )
        chosen = {int(value) for value in numbers}
        gold = set(sample.reference["premises"])
        # Order-independent: {5, 13} and {13, 5} are the same answer. A pair
        # naming the same statement twice is not a pair, and cannot match a
        # two-element gold.
        exact = 1.0 if chosen == gold else 0.0
        return SampleScore(
            metrics={
                "premise_set_match": exact,
                # Partial credit is reported but is NOT the score: it exists so
                # a 0.0 can be read as "found one of the two" rather than
                # "found neither".
                "premise_partial": len(chosen & gold) / 2.0,
            },
            prediction=" ".join(str(value) for value in sorted(chosen)),
            details={"gold": " ".join(str(value) for value in sorted(gold))},
        )

    def aggregate(self, scores) -> dict[str, float]:
        return aggregate_mean_metrics([score.metrics for score in scores])

    # ------------------------------------------------------------------ #
    # documentation
    # ------------------------------------------------------------------ #

    def documentation(self) -> AdapterDocumentation:
        dropped = getattr(self, "dropped_premise_counts", {})
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="UniADILR-HGc",
            domain="Formal Reasoning: Constrained Abductive Premise Selection",
            source_url="https://github.com/YuSheng-00/UniADILR",
            processing_mode="Generation",
            split_used=getattr(self, "split_used", ABDUCTION_FILE),
            abductive_subset=(
                "THE ABDUCTION FILE ONLY, and within it the two-premise items. UniADILR ships "
                "data/UniADILR-HGc/abduction.jsonl beside deduction.jsonl and induction.jsonl, "
                "which apply the same construction to other reasoning types; those two are "
                "never read, and a row in the abduction file whose reasoning_type is not "
                "'abduction' is dropped. What remains is constrained abduction: which two of "
                "many statements jointly support a further claim. The model names the pair; "
                "finding it among the distractors is the whole difficulty."
            ),
            sampling_procedure=self.sampling_note(),
            metrics_description={
                "premise_set_match": (
                    "(PRIMARY, higher is better, 0-1) 1.0 when the two numbers given are the "
                    "two the release's own proof field names, compared as a SET -- '13 5' and "
                    "'5 13' are the same answer. Checked mechanically; there is no judge."
                ),
                "premise_partial": (
                    "(diagnostic, higher is better, 0-1) how many of the two the model found, "
                    "over two. Reported so a 0.0 on the primary metric can be read as 'found "
                    "one' rather than 'found neither'. It is NOT the score."
                ),
                "parse_failure_rate": (
                    "(lower is better, 0-1) fraction of responses that did not contain exactly "
                    "two numbers. Answering with one number, or three, is a failure to answer "
                    "the question rather than a wrong pair, and is counted here so it cannot "
                    "hide inside the primary metric."
                ),
                "self_consistency_<metric>":
                "Every metric also gets a self_consistency_ counterpart: the plurality answer "
                "over modes.repeats samples of the same record, read off those samples rather "
                "than bought again. Available because the answer is a pair of numbers, which "
                "repeats can agree on.",
            },
            primary_metric="premise_set_match",
            decisions=[
                "THE ANSWER IS THE SUPPORTING PAIR, NOT THE CLAIM. Asking for the claim made "
                "the answer free text that only an LLM judge could grade, and it graded the "
                "wrong thing: a model can paraphrase a plausible hypothesis from the pool's "
                "general subject matter without locating the two statements that entail it. "
                "The pair is checkable outright, and the release's proof field is the key.",
                "Statements are shown as bare numbers ('5. ...') rather than the release's "
                "'sent5' labels, so the number the model reads is the number the answer format "
                "asks it to write.",
                "Read the premises from the LEFT of '->' in the proof only; the right side is "
                "the claim, and parsing the whole string would read a stray token in the claim "
                "text as a third premise.",
                f"Used only the items whose proof names exactly two supporting statements "
                f"({getattr(self, 'split_used', '')!r} records the counts). A prompt that asks "
                "for exactly two numbers cannot be satisfied on a one- or three-premise item, "
                "so those are dropped rather than scored against an impossible instruction: "
                + (", ".join(f"{count} item(s) with {n} premise(s)"
                             for n, count in sorted(dropped.items())) or "none were dropped")
                + ".",
                "Used only the abductive split, and enforced it per row rather than trusting "
                "the filename.",
                "Presented the statements in the release's own numbering, which is what its "
                "proof field refers to.",
            ],
            caveats=[
                "Two numbers out of a pool of N gives a chance rate of 1 / C(N, 2) -- small, "
                "but not zero, and it falls as the pool grows. Read the score against the pool "
                "size in n_statements rather than against zero.",
                "The release's proof is one derivation. If another pair of statements also "
                "entailed the claim, naming it would score 0; the construction makes that "
                "unlikely but nothing here verifies it.",
                "Items are synthetic, so their distractors are drawn from unrelated corpora "
                "and read as obviously unrelated more often than a natural pool would.",
            ],
            statistics={
                **self.base_statistics(),
                "dropped_non_abductive": getattr(self, "dropped_non_abductive", 0),
                "dropped_by_premise_count": dropped,
            },
        )
