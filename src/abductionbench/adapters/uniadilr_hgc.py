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

**How many statements is itself part of the question.**  All 500 abductive
items are used: 466 have two supporting statements, 19 have one and 15 have
three.  The prompt does not say which, so deciding *how many* to name is part
of the task rather than something the instruction gives away -- a model told
"exactly two" gets the cardinality for free on 93% of items.

**Scoring.**  Order-independent exact set match against the release's own
``proof``: ``13 5`` and ``5 13`` are the same answer, and so is any ordering of
a three-statement set.  Naming too many or too few is wrong, which is the point
of not stating the count.  ``premise_f1`` sits beside it for partial credit.
No judge -- there is nothing here a judge could settle that a set comparison
cannot.
"""

from __future__ import annotations

import re
from typing import Any

from ..core.adapter import SkippedDataset
from ..core.metrics import aggregate_mean_metrics, extract_answer_span, set_prf
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
#: A bare integer in the model's answer. Reads "5 13", "5, 13" and
#: "sent5 sent13" alike. Prose that happens to contain a number will be read as
#: an answer, which is the price of not constraining the count; the Requirements
#: block asks for numbers only, and a prose answer scores 0 on the set match.
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
        "statements, most of which are irrelevant. A few of them together support a single "
        "further claim. Identify which ones -- how many there are is part of the question, "
        "not something you are told."
    )
    answer_format = "the statement numbers, separated by spaces (for example: 5 13)"
    task_requirements = (
        "give the number of every supporting statement, and no others",
        "do not name the claim",
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

        # Every item, whatever its premise count. The prompt does not state how
        # many statements support the claim, so an item with one or three is as
        # answerable as one with two -- and working out the cardinality is part
        # of the task. Kept as a statistic because it is the shape of the set
        # the model has to find.
        usable = []
        self.premise_counts: dict[int, int] = {}
        for row in abductive:
            count = len(_premise_ids(row.get("proof") or ""))
            if count < 1:
                continue
            usable.append(row)
            self.premise_counts[count] = self.premise_counts.get(count, 0) + 1
        if not usable:
            raise SkippedDataset(
                f"{path} has no items whose proof names any supporting statement"
            )
        spread = ", ".join(
            f"{count} with {n}" for n, count in sorted(self.premise_counts.items())
        )
        self.split_used = (
            f"{ABDUCTION_FILE} in full ({len(usable)} items: {spread} supporting "
            f"statement(s)); the deduction and induction files are not used"
        )
        return usable

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        context = item.get("context")
        gold_claim = C.normalize_whitespace(item.get("hypothesis"))
        proof = C.normalize_whitespace(item.get("proof"))
        premises = _premise_ids(proof)
        if not isinstance(context, dict) or not context or not premises:
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
                    "Some of these statements together support a single further claim. "
                    "Which ones?"
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
            metadata={
                "n_statements": len(statements),
                # How many the model has to find; the prompt never says.
                "n_premises": len(premises),
            },
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
        """Whatever numbers were given, compared as a set against the proof.

        The count is not fixed by the prompt, so it is not fixed here either:
        naming three statements where the proof names two is a wrong answer,
        not a malformed one. Only an answer with NO numbers in it has failed to
        answer the question, and that is what the parse failure is reserved for.
        """
        answer = extract_answer_span(response.text, output_contract)
        numbers = _NUMBER_RE.findall(answer or "")
        if not numbers:
            return unparsed_score(
                ["premise_set_match", "premise_f1"],
                raw=(answer or response.text or "")[:200],
            )
        chosen = {int(value) for value in numbers}
        gold = set(sample.reference["premises"])
        overlap = set_prf(chosen, gold)
        return SampleScore(
            metrics={
                # Order-independent, and cardinality-sensitive: the set has to
                # be right, not merely overlap.
                "premise_set_match": 1.0 if chosen == gold else 0.0,
                # Partial credit. F1 rather than recall on purpose: recall alone
                # would reward naming every statement in the pool, which is
                # exactly the answer that has understood nothing.
                "premise_f1": overlap["f1"],
                # Did the model work out HOW MANY support the claim? The prompt
                # does not say, so this is a real part of the task and is worth
                # separating from getting the right ones.
                "premise_count_match": 1.0 if len(chosen) == len(gold) else 0.0,
            },
            prediction=" ".join(str(value) for value in sorted(chosen)),
            details={
                "gold": " ".join(str(value) for value in sorted(gold)),
                "n_given": len(chosen),
                "n_gold": len(gold),
            },
        )

    def aggregate(self, scores) -> dict[str, float]:
        return aggregate_mean_metrics([score.metrics for score in scores])

    # ------------------------------------------------------------------ #
    # documentation
    # ------------------------------------------------------------------ #

    def documentation(self) -> AdapterDocumentation:
        counts = getattr(self, "premise_counts", {})
        spread = ", ".join(f"{count} item(s) with {n}" for n, count in sorted(counts.items()))
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="UniADILR-HGc",
            domain="Formal Reasoning: Constrained Abductive Premise Selection",
            source_url="https://github.com/YuSheng-00/UniADILR",
            processing_mode="Generation",
            split_used=getattr(self, "split_used", ABDUCTION_FILE),
            abductive_subset=(
                "THE ABDUCTION FILE ONLY, in full. UniADILR ships "
                "data/UniADILR-HGc/abduction.jsonl beside deduction.jsonl and induction.jsonl, "
                "which apply the same construction to other reasoning types; those two are "
                "never read, and a row in the abduction file whose reasoning_type is not "
                "'abduction' is dropped. What remains is constrained abduction: which of many "
                "statements jointly support a further claim. The model names them; finding "
                "them among the distractors, and working out how many there are, is the whole "
                "difficulty."
            ),
            sampling_procedure=self.sampling_note(),
            metrics_description={
                "premise_set_match": (
                    "(PRIMARY, higher is better, 0-1) 1.0 when the numbers given are exactly "
                    "the ones the release's own proof field names, compared as a SET -- order "
                    "is irrelevant, but cardinality is not: naming three where the proof names "
                    "two is wrong. Checked mechanically; there is no judge."
                ),
                "premise_f1": (
                    "(diagnostic, higher is better, 0-1) F1 of the chosen set against the gold "
                    "set. Reported so a 0.0 on the primary metric can be read as 'found some' "
                    "rather than 'found none'. F1 rather than recall on purpose: recall alone "
                    "would reward naming every statement in the pool, which is the answer that "
                    "has understood nothing. It is NOT the score."
                ),
                "premise_count_match": (
                    "(diagnostic, higher is better, 0-1) whether the model named the right "
                    "NUMBER of statements, regardless of which. The prompt does not say how "
                    "many support the claim, so this is a real part of the task, and it "
                    "separates 'picked the wrong ones' from 'did not work out how many'."
                ),
                "parse_failure_rate": (
                    "(lower is better, 0-1) fraction of responses containing no number at all. "
                    "Only that counts as failing to answer: a wrong count is a wrong answer, "
                    "not a malformed one, and belongs in the primary metric."
                ),
                "self_consistency_<metric>":
                "Every metric also gets a self_consistency_ counterpart: the plurality answer "
                "over modes.repeats samples of the same record, read off those samples rather "
                "than bought again. Available because the answer is a set of numbers, which "
                "repeats can agree on.",
            },
            primary_metric="premise_set_match",
            decisions=[
                "THE ANSWER IS THE SUPPORTING STATEMENTS, NOT THE CLAIM. Asking for the claim "
                "made the answer free text that only an LLM judge could grade, and it graded "
                "the wrong thing: a model can paraphrase a plausible hypothesis from the "
                "pool's general subject matter without locating the statements that entail it. "
                "A set of numbers is checkable outright, and the release's proof field is the "
                "key.",
                "THE PROMPT DOES NOT SAY HOW MANY to name. 466 of the 500 items have exactly "
                "two supporting statements, so an instruction to give two would hand the model "
                "the cardinality on 93% of the set; deciding how many is part of the task. An "
                "earlier version of this adapter did fix the count at two, and dropped the 34 "
                "items that did not fit it -- both are gone.",
                f"Used every abductive item, whatever its premise count ({spread or 'n/a'}).",
                "Statements are shown as bare numbers ('5. ...') rather than the release's "
                "'sent5' labels, so the number the model reads is the number the answer format "
                "asks it to write.",
                "Read the premises from the LEFT of '->' in the proof only; the right side is "
                "the claim, and parsing the whole string would read a stray token in the claim "
                "text as an extra premise.",
                "Used only the abductive split, and enforced it per row rather than trusting "
                "the filename.",
                "Presented the statements in the release's own numbering, which is what its "
                "proof field refers to.",
            ],
            caveats=[
                "Chance is low but not zero, and it now depends on the model guessing the "
                "cardinality as well as the members: for a pool of N there are C(N, k) sets of "
                "size k. Read the score against n_statements rather than against zero.",
                "The release's proof is one derivation. If another set of statements also "
                "entailed the claim, naming it would score 0; the construction makes that "
                "unlikely but nothing here verifies it.",
                "premise_f1 gives partial credit to an over-long answer, so read it beside "
                "premise_count_match -- a model that names half the pool will show a "
                "respectable F1 and a count_match of 0.",
                "Items are synthetic, so their distractors are drawn from unrelated corpora "
                "and read as obviously unrelated more often than a natural pool would.",
            ],
            statistics={
                **self.base_statistics(),
                "dropped_non_abductive": getattr(self, "dropped_non_abductive", 0),
                "items_by_premise_count": counts,
            },
        )
