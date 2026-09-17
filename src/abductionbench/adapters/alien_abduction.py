"""Alien Abduction: recover a hidden Python function from input/output evidence.

Source: "Don't Let Me Ask for It: LLMs Show Deficiencies in Active Multi-Turn
Information Acquisition for Abductive Inference" (arXiv:2608.03388).

A Game Master holds a small pure function and answers questions about it; the
player has to abduce the function and submit it as code.  The paper factors the
game along two axes -- who chooses the probes (active / passive / single-turn)
and what the answers look like (exact outputs / yes-no membership verdicts) --
giving six modes.  **This adapter implements the three output modes only**; the
verdict modes are a different task (falsifying a self-proposed pair) and are not
part of this suite.

The three land on the three delivery modes this harness already has, and the
mapping is exact rather than approximate:

============================  ==================  ====================================
Paper mode                    ``data_delivery``   What the model may do
============================  ==================  ====================================
Active-Output (AO)            ``interactive``     ``TEST: <input>`` or ``SOLVE:``, 15 turns
Passive-Output (PO)           ``sequential``      ``NEXT:`` or ``SOLVE:``, 15 turns
Single-Turn-Output (STO)      ``static``          one shot, from a fixed batch of examples
============================  ==================  ====================================

AO and PO quote the release's own protocol prompt (Figure 13, templates D.1 and
D.3, with the fuller wording of the transcript in Table 6), so they set
``authors_prompt`` and run one prompt mode: reproducing a protocol means sending
the bytes its authors sent.  **STO does not.**  It is a single static prompt with
no protocol to reproduce, so it is written in this suite's own house style and
gets the generic IO and CoT instructions like every other static dataset -- which
is the only way its ``io`` and ``cot`` rows mean what the column says.

**Where the data comes from.**  The paper states that "the source code and test
instances will be released upon acceptance"; no release exists.  Table 3 does
publish the names and signatures of all fifty targets, and Section 4.1 publishes
how their evidence is built, so :mod:`._alien_targets` reconstructs the target
set from the paper's own list rather than inventing a different one -- the names,
the five domains, the ten-per-domain split and the hundred test cases per target
are the paper's; the fifty bodies are this suite's reading of the paper's names,
and the readings that were not forced are recorded per function and surfaced in
the run documentation.  All three modes use the identical fifty targets and the
identical suites, which is the paper's own design ("the same 50 targets are used
across the modes, so differences between modes cannot be confounded by target
difficulty").

**How it is scored.**  The paper's criterion, unchanged: an episode is solved if
and only if the submitted function agrees with the target on the held-out suite.
That means running model-written code, which nothing else in this suite does; it
happens in :mod:`abductionbench.core.sandbox`, whose docstring says exactly what
that sandbox does and does not contain.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Sequence
from typing import Any

from ..core.metrics import aggregate_mean_metrics
from ..core.sandbox import DEFAULT_MEMORY_MB, DEFAULT_TIMEOUT_S, Verification, verify
from ..core.types import AdapterDocumentation, ChatMessage, ModelResponse, SampleScore, SampleSpec
from . import _alien_targets as T
from ._base import PooledDatasetAdapter

PAPER_URL = "https://arxiv.org/abs/2608.03388"

#: The entry point the paper's own transcripts use.
ENTRYPOINT = "solution"
#: Turn budget for the two multi-turn modes (Figure 13: "Turn budget: 15").
TURN_BUDGET = 15
#: How many examples the single-turn mode is shown up front (Table 1: "a fixed
#: batch of ten evidence items").
PRELOADED_EXAMPLES = 10
#: What the Game Master says to an unreadable turn, verbatim from the paper's
#: transcripts (Figure 10).  A parser error costs the turn, which is the point:
#: Table 5 reports parser-error rates as a finding about the models.
PARSE_ERROR_REPLY = "OUTPUT: Parse Error Invalid Input"

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S | re.I)
_SOLVE = re.compile(r"\bSOLVE\s*:", re.I)
_TEST = re.compile(r"\bTEST\s*:(.*)", re.I)
_NEXT = re.compile(r"\bNEXT\s*:?", re.I)
_ANSWER_MARKER = re.compile(r"^\s*Answer\s*:", re.I | re.M)


def extract_code(text: str) -> str:
    """The Python the model submitted, or ``""``.

    Three readings, widening: a fenced block (what both prompts ask for), the
    text after a ``SOLVE:`` or ``Answer:`` marker, and finally the whole reply
    when it plainly contains a definition.  The last one exists because an
    episode costs fifteen turns and throwing it away over a missing fence would
    be measuring formatting rather than inference -- the paper counts that
    separately, as a parser error, and so does this adapter.
    """
    if not text:
        return ""
    fences = _FENCE.findall(text)
    if fences:
        # The last fence: a model that shows a rejected draft first submits last.
        for block in reversed(fences):
            if "def " in block:
                return block.strip()
        return fences[-1].strip()
    tail = text
    for marker in (_SOLVE, _ANSWER_MARKER):
        # The last marker: a model that shows a rejected draft submits last.
        found = list(marker.finditer(text))
        if found:
            tail = text[found[-1].end() :]
            break
    tail = tail.strip().strip("`").strip()
    if "def " in tail:
        # Drop any prose before the first definition; a trailing sentence after
        # the body is left to the compiler, which will say so.
        return tail[tail.index("def ") :].strip()
    return ""


def _render_pair(args: tuple[Any, ...], output: Any) -> str:
    """One example as the Game Master writes it: ``(<input>, <output>)``."""
    shown = args[0] if len(args) == 1 else tuple(args)
    return f"({shown!r}, {output!r})"


class _AlienAbductionBase(PooledDatasetAdapter):
    """Shared target set, sandbox scoring and documentation for the three modes."""

    adapter_version = "1.0"

    #: Correctness is decidable by running the code, so this dataset is graded
    #: mechanically and never by a judge.
    objective_metrics = True
    selection_cardinality = None
    primary_metric = "solved"
    table_hypothesis_mode = "Generation"

    #: Which paper mode this class is, for the documentation and the prompts.
    paper_mode = ""
    #: How many of the suite's cases this mode reveals before the model acts.
    preloaded = 0

    # ------------------------------------------------------------------ #
    # the targets
    # ------------------------------------------------------------------ #

    def load_items(self) -> list[dict[str, Any]]:
        """The fifty targets of Table 3, each with its suite and its reveal order.

        Read from ``data/<dataset_id>/``, which this adapter writes rather than
        downloads: the benchmark's authors released nothing to fetch, so the
        materialized copy *is* the dataset on disk, and it is regenerated
        whenever the generator's fingerprint stops matching it.  The reveal
        order is fixed per target, so the single-turn batch is the prefix of the
        sequential mode's reveals -- which is what makes the paper's "STO vs. PO
        isolates the effect of sequential interaction" true here too.
        """
        count = int(self.context.option("test_cases", 100))
        items = T.load_materialized(self.context.data_dir, count)
        self.split_used = (
            "the 50 targets of Table 3, rebuilt from the paper's published names "
            "(no code or test release exists); 10 per domain across 5 domains"
        )
        return items

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        target: T.Target = item["target"]
        cases = item["cases"]
        order = item["order"]
        revealed = [cases[position] for position in order[: self.preloaded]]
        return SampleSpec(
            sample_id=f"alien-{target.name}",
            fields=self.sample_fields(target, revealed),
            reference={
                "target": target.name,
                "signature": target.signature,
                "source": target.source,
            },
            task_kind="generation",
            # Room for a chain of thought plus a short function. The targets are
            # deliberately primitive; a reply longer than this is reasoning, not
            # code, and the budget is that reasoning's headroom.
            max_tokens=1536,
            metadata={
                "domain": target.domain,
                "target": target.name,
                "ambiguous_reading": T.AMBIGUOUS.get(target.name, ""),
                "_cases": cases,
                "_order": order,
                "_revealed": revealed,
                "_signature": target.signature,
                "_arity": target.arity,
            },
        )

    def sample_fields(
        self, target: T.Target, revealed: list[tuple[tuple[Any, ...], Any]]
    ) -> dict[str, Any]:
        """The prompt fields for one target.  Only the static mode uses these."""
        return {}

    # ------------------------------------------------------------------ #
    # scoring -- the paper's criterion, run in a sandbox
    # ------------------------------------------------------------------ #

    def _sandbox(self, cases: list[tuple[tuple[Any, ...], Any]], code: str) -> Verification:
        return verify(
            code,
            cases,
            entrypoint=ENTRYPOINT,
            timeout_s=float(self.context.option("timeout_s", DEFAULT_TIMEOUT_S)),
            memory_mb=int(self.context.option("memory_mb", DEFAULT_MEMORY_MB)),
        )

    def submitted_code(self, sample: SampleSpec, response: ModelResponse) -> tuple[str, bool]:
        """The code this episode submitted, and whether the model committed at all.

        The multi-turn modes record the submission in the environment state when
        the ``SOLVE`` arrives, so a model that solved on turn 7 and then said
        nothing more is still scored on what it submitted.  Falling back to the
        last message covers the static mode and an episode whose state is gone.
        """
        state = sample.metadata.get("_episode_state") or {}
        if state.get("submitted"):
            return str(state.get("code", "")), True
        code = extract_code(response.text or "")
        return code, bool(code)

    def score(
        self,
        sample: SampleSpec,
        response: ModelResponse,
        *,
        output_contract: dict[str, Any] | None = None,
    ) -> SampleScore:
        cases: list[tuple[tuple[Any, ...], Any]] = sample.metadata.get("_cases") or []
        code, committed = self.submitted_code(sample, response)
        domain = str(sample.metadata.get("domain", "")).lower().replace(" ", "_")

        metrics: dict[str, float] = {"committed": float(committed)}
        details: dict[str, Any] = {
            "target": sample.reference.get("target"),
            "domain": sample.metadata.get("domain"),
            "submitted_code": code,
        }
        turns = sample.metadata.get("turns_used")
        if turns is not None:
            metrics["turns_used"] = float(turns)
        state = sample.metadata.get("_episode_state") or {}
        for key in ("evidence_requests", "parser_errors"):
            if key in state:
                metrics[key] = float(state[key])

        if not committed:
            # In a multi-turn mode this is not a parse failure: an episode
            # that spends its budget without ever committing is one of the
            # paper's own findings (Table 4), and filing it as unparseable would
            # report a result as a formatting fault. In the static mode there is
            # no such distinction to draw -- one turn, and no code came back --
            # so it is exactly what parse_failure_rate is for.
            parse_ok = bool(state)
            metrics.update({"solved": 0.0, f"solved_{domain}": 0.0, "test_pass_rate": 0.0})
            details["outcome"] = "not_committed"
            return SampleScore(
                metrics=metrics, prediction=None, parse_ok=parse_ok, details=details
            )

        result = self._sandbox(cases, code)
        metrics.update(
            {
                "solved": float(result.solved),
                f"solved_{domain}": float(result.solved),
                "test_pass_rate": result.pass_rate,
                "runtime_failure": float(result.status in ("error", "crashed", "timeout")),
            }
        )
        details.update(
            {
                "outcome": result.status,
                "passed": result.passed,
                "total": result.total,
                "failures": result.failures,
                "error": result.error,
            }
        )

        # The paper's Hypothesis Retrodiction Accuracy: how much of the evidence
        # the player actually saw its final hypothesis reproduces. Undefined when
        # an episode saw none, and then it is left out rather than scored zero.
        evidence = self.evidence_seen(sample)
        if evidence:
            retro = self._sandbox(evidence, code)
            metrics["hypothesis_retrodiction"] = retro.pass_rate
            details["evidence_seen"] = len(evidence)

        return SampleScore(
            metrics=metrics,
            # What the submission *computes* over the suite, not how it was
            # written. A self-consistency vote needs answers that can coincide,
            # and two correct solutions are never written the same way twice --
            # but they have the same behaviour, which is what is voted on here.
            prediction=f"behaviour:{result.digest}" if result.digest else f"({result.status})",
            parse_ok=True,
            details=details,
        )

    def evidence_seen(self, sample: SampleSpec) -> list[tuple[tuple[Any, ...], Any]]:
        """The input/output pairs this episode was actually shown."""
        state = sample.metadata.get("_episode_state") or {}
        if state.get("seen"):
            return [(tuple(args), output) for args, output in state["seen"]]
        return list(sample.metadata.get("_revealed") or [])

    def aggregate(self, scores: Sequence[SampleScore]) -> dict[str, float]:
        return aggregate_mean_metrics([score.metrics for score in scores])

    # ------------------------------------------------------------------ #
    # documentation
    # ------------------------------------------------------------------ #

    #: Filled in by each mode.
    mode_decisions: tuple[str, ...] = ()
    mode_caveats: tuple[str, ...] = ()

    def documentation(self) -> AdapterDocumentation:
        per_domain = {
            domain: f"{len(functions)} targets, {signature}"
            for domain, (signature, functions) in T.DOMAINS.items()
        }
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name=f"Alien Abduction ({self.paper_mode})",
            domain="Program Induction: Hidden Function Recovery",
            source_url=PAPER_URL,
            processing_mode="Generation",
            split_used=self.split_used,
            abductive_subset=(
                "All 50 targets of the benchmark. The task is abductive in the paper's own "
                "sense: the model observes effects (input/output pairs) and must infer the "
                "rule that produced them, with the rule underdetermined by any finite sample "
                "of its behaviour."
            ),
            sampling_procedure=(
                "No sampling in the usual sense: the population is the 50 published targets and "
                "all of them are evaluated whenever sample_size allows, so what the draw fixes is "
                "only the order -- "
                + self.sampling_note()
            ),
            metrics_description={
                "solved": "(PRIMARY, higher is better, 0-1) the paper's criterion: the submitted "
                "function agrees with the hidden target on every case of the held-out suite. "
                "Partial agreement is not a partial solve.",
                "solved_<domain>": "the same criterion restricted to one of the five domains; "
                "identical definition, filtered population. The paper reports per-domain results "
                "because the aggregate hides real differences between them.",
                "test_pass_rate": "(higher is better, 0-1) fraction of the held-out suite the "
                "submission agreed on. A diagnostic beside `solved`, not a substitute for it: a "
                "function right on 99 of 100 cases is the wrong function.",
                "committed": "(0-1) whether the model submitted code at all. The paper reports "
                "non-commitment separately (Table 4) because a model that spends its turn budget "
                "without ever answering has failed differently from one that answered wrongly.",
                "hypothesis_retrodiction": "(higher is better, 0-1) the paper's Hypothesis "
                "Retrodiction Accuracy: how much of the evidence this episode actually saw the "
                "submitted function reproduces. Reported only for episodes that saw evidence; "
                "absent, not zero, for the rest.",
                "runtime_failure": "(lower is better, 0-1) fraction of submissions that did not "
                "run: a syntax error, an exception at import, or a function that exhausted its "
                "CPU or memory budget.",
                "turns_used": "(multi-turn modes) how many turns the episode took.",
                "evidence_requests": "(multi-turn modes) how many turns were spent asking for "
                "evidence rather than submitting.",
                "parser_errors": "(multi-turn modes) turns the Game Master could not read as an "
                "action. The paper reports this as a finding about the models (Table 5), so it is "
                "a metric here rather than a silently retried turn.",
                "parse_failure_rate": "(lower is better, 0-1) fraction of responses no submission "
                "could be extracted from. Distinct from `committed`: a model that deliberately "
                "spent its last turn probing did not fail to parse.",
            },
            primary_metric="solved",
            decisions=[
                "Implemented the three output modes only. The verdict modes ask the model to "
                "propose a pair and be told whether it is in the function's graph, which is a "
                "different task (falsifying a self-proposed hypothesis) and is out of scope here.",
                "Rebuilt the 50 targets from Table 3 rather than inventing a different set: the "
                "paper publishes every name and signature, so using them keeps a per-target "
                "comparison against the published results meaningful.",
                "All three modes share one target set and one suite per target, which is the "
                "paper's own design -- otherwise a difference between modes could be a difference "
                "in target difficulty.",
                "Verified the paper's criterion by executing the submission against the held-out "
                "suite, in a resource-bounded subprocess (see core.sandbox). Scoring it any other "
                "way -- string similarity to the reference source, or an LLM judge -- would grade "
                "how the function was written rather than what it computes.",
                "Wrote the generated benchmark into data/<dataset_id>/ (targets.json, "
                "test_cases.jsonl, MANIFEST.json) instead of keeping it only in memory, so this "
                "dataset's data directory holds its source like every other dataset's does. The "
                "manifest carries a fingerprint over the generator version and every target body, "
                "and a copy that no longer matches is regenerated rather than trusted -- otherwise "
                "editing one function would leave a run scoring against the previous suite.",
                *self.mode_decisions,
            ],
            caveats=[
                "The paper's code and test instances are not released ('will be released upon "
                "acceptance'). The names, signatures, domain split and evidence construction here "
                "are the paper's; the 50 function bodies are this suite's reading of those names, "
                "so absolute numbers are not directly comparable to the published ones.",
                "Eight of the fifty names admit more than one honest body; each such reading is "
                "recorded on the function and carried on the sample as `ambiguous_reading`, so a "
                "per-target result can be checked against the interpretation it rests on.",
                "Two Boolean targets (b_without_a, difference_negative) reduce to the same truth "
                "table. On two Boolean inputs only sixteen functions exist and ten published names "
                "must land among them, so this is a property of the name list rather than of the "
                "reconstruction.",
                "The Logic domain's entire input space is four pairs, so its suite is four cases "
                "rather than a hundred, and its evidence batch cannot exceed that. A Logic target "
                "is therefore fully determined by its evidence, which is why the paper's Logic "
                "results are the highest in every mode.",
                "Submitted code is executed. The sandbox bounds CPU, memory and wall clock and "
                "isolates the interpreter, but it is not a security boundary -- the paper's own "
                "harness uses an ephemeral container, which is not available in an unprivileged "
                "one. Run this against models you are willing to run code from.",
                *self.mode_caveats,
            ],
            statistics={
                **self.base_statistics(),
                "targets": len(T.TARGETS),
                "domains": per_domain,
                "cases_per_target": {
                    "Logic": 4,
                    "other domains": int(self.context.option("test_cases", 100)),
                },
                "ambiguous_readings": len(T.AMBIGUOUS),
                "turn_budget": TURN_BUDGET if self.paper_mode != "Single-Turn-Output" else 1,
            },
        )


# --------------------------------------------------------------------------- #
# Single-Turn-Output -> static.  This suite's own prompt, not the paper's.
# --------------------------------------------------------------------------- #


class AlienAbductionAdapter(_AlienAbductionBase):
    """Infer the hidden function from a fixed batch of examples, in one shot.

    The paper's Single-Turn-Output mode.  Its prompt is *not* reproduced here:
    a single-turn task has no protocol to reproduce, only a question, so this is
    written the way every other static dataset in this suite is written and gets
    the shared IO and CoT instructions.  What is kept from the paper is the
    thing that makes it that mode -- one action, from a preloaded batch of ten
    examples and nothing else.
    """

    paper_mode = "Single-Turn-Output"
    data_delivery_mode = "static"
    #: Deliberately False: the prompt below is this suite's, so both prompt
    #: modes are real modes rather than the same bytes under two labels.
    authors_prompt = False
    preloaded = PRELOADED_EXAMPLES

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. You are shown input/output pairs "
        "produced by a hidden Python function. Infer the function that produced them and "
        "write it out."
    )

    answer_format = "a Python code block defining solution"
    answer_is_a_block = True
    answer_block_note = (
        "Put nothing in that block but the code itself, fenced as ```python ... ```."
    )
    task_requirements = (
        "define exactly one function, named solution, with the signature shown",
        "use only the Python standard library",
        "the function must reproduce every example shown",
        "the function must be deterministic and free of side effects",
    )

    mode_decisions = (
        "Wrote this mode's prompt in the suite's own house style rather than quoting the "
        "paper's Single-Turn-Output template. A static dataset's io and cot rows are only "
        "distinguishable when the mode instruction is this harness's to add, and a one-shot "
        "question -- unlike the two multi-turn protocols -- has no interaction rules that "
        "quoting is needed to preserve.",
        "Showed the first ten pairs of the same per-target reveal order the sequential mode "
        "walks, so this mode's evidence is a prefix of that mode's rather than a different "
        "draw. The paper's own comparison (STO vs. PO) isolates sequential interaction, which "
        "it can only do if the evidence is otherwise the same.",
    )
    mode_caveats = (
        "A Logic target's whole input space is four pairs, so its batch is four examples, not "
        "ten -- and those four fully determine the function. Logic is therefore a check on "
        "whether the model can read exhaustive evidence, not on abduction under uncertainty.",
    )

    def sample_fields(
        self, target: T.Target, revealed: list[tuple[tuple[Any, ...], Any]]
    ) -> dict[str, Any]:
        lines = [
            f"{index}. solution({T.render_input(args)}) -> {output!r}"
            for index, (args, output) in enumerate(revealed, start=1)
        ]
        return {
            "context": f"Signature: {target.signature}",
            "observation": "These are all the examples you have:\n" + "\n".join(lines),
            "instructions": (
                "Infer the hidden function that produced these pairs and write it out in "
                "Python. There are no further examples to ask for."
            ),
        }


# --------------------------------------------------------------------------- #
# the two multi-turn modes -- the release's own protocol
# --------------------------------------------------------------------------- #

#: A format illustration per domain: what a pair looks like, in a shape that is
#: not any target's evidence.  The paper's own templates carry one of these
#: (``Visible example format: (-10, 10)``), instantiated for the domain shown.
_FORMAT_EXAMPLE = {
    "Number": "(-10, 10)",
    "Number Pairs": "((2, 3), 5)",
    "String": "('ab', 'BA')",
    "List": "([1, 2], 3)",
    "Logic": "((True, False), False)",
}

#: Section 3: "Alongside each action, the player reports its current hypothesis,
#: a state ..., its rationale ..., and a confidence score. These fields are
#: recorded for analysis but are not processed by the Game Master."
_REPORT_BLOCK = (
    "Alongside your action, report:\n"
    "Hypothesis: <your current guess at the function>\n"
    "State: probing | confirming | uncertain\n"
    "Rationale: <why this action>\n"
    "Confidence: <0.0-1.0>\n"
    "These are recorded but do not affect the Game Master's reply."
)


class _UnreadableType:
    """Sentinel for 'that was not a Python literal', distinct from a literal None."""

    __slots__ = ()


_UNREADABLE = _UnreadableType()


def _parse_test_input(domain: str, raw: str) -> tuple[Any, ...] | None:
    """Read the arguments of a ``TEST:`` turn, or ``None`` if they are not readable.

    Typed against the domain rather than accepted as written: a Number target
    handed a string would raise inside the Game Master's own oracle, and an
    environment that crashes on a malformed request is not an environment.
    """
    text = raw.strip().rstrip(".").strip()
    if not text:
        return None

    def literal(source: str) -> Any:
        try:
            return ast.literal_eval(source)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            return _UNREADABLE

    if domain == "Number":
        value = literal(text)
        return (value,) if isinstance(value, int) and not isinstance(value, bool) else None
    if domain == "Number Pairs":
        value = literal(text if text.startswith("(") else f"({text})")
        if (
            isinstance(value, (tuple, list))
            and len(value) == 2
            and all(isinstance(v, int) and not isinstance(v, bool) for v in value)
        ):
            return tuple(value)
        return None
    if domain == "Logic":
        value = literal(text if text.startswith("(") else f"({text})")
        if isinstance(value, (tuple, list)) and len(value) == 2 and all(
            isinstance(v, bool) for v in value
        ):
            return tuple(value)
        return None
    if domain == "List":
        value = literal(text if text.startswith("[") else f"[{text}]")
        if isinstance(value, (list, tuple)) and all(
            isinstance(v, int) and not isinstance(v, bool) for v in value
        ):
            return (list(value),)
        return None
    if domain == "String":
        value = literal(text)
        if isinstance(value, str):
            return (value,)
        # An unquoted string is the commonest way to write this turn, and
        # rejecting it would score quoting rather than probing.
        return (text,) if value is _UNREADABLE else None
    return None  # pragma: no cover - the five domains above are the whole table


class _MultiTurnAlienAdapter(_AlienAbductionBase):
    """Shared episode machinery for the two modes that quote the paper's protocol."""

    #: The release's own protocol prompt, so there is one prompt set and one
    #: prompt mode (see DatasetAdapter.authors_prompt).
    authors_prompt = True
    max_turns = TURN_BUDGET

    #: The "Mode:" line of the paper's template.
    setup_line = ""
    #: The "Action per turn:" lines and whatever the template says about replies.
    protocol_lines: tuple[str, ...] = ()

    def opening(self, sample: SampleSpec) -> str:
        signature = sample.metadata["_signature"]
        body = [
            "Infer hidden Python function from limited evidence.",
            f"You know: {signature}; You have {self.max_turns} turns total.",
            f"Mode: {self.setup_line}",
            *self.protocol_lines,
            f"Visible example format: {_FORMAT_EXAMPLE[sample.metadata['domain']]}",
            "A SOLVE ends the episode. Submit it as:",
            f"SOLVE: ```python\ndef {ENTRYPOINT}{signature}:\n    ...\n```",
            "",
            _REPORT_BLOCK,
        ]
        return "\n".join(body)

    def interactive_start(self, sample: SampleSpec) -> tuple[list[ChatMessage], dict[str, Any]]:
        return (
            [ChatMessage(role="user", content=self.opening(sample))],
            {
                "seen": [],
                "evidence_requests": 0,
                "parser_errors": 0,
                "submitted": False,
                "code": "",
            },
        )

    def build_messages(self, sample: SampleSpec) -> tuple[list[ChatMessage], dict[str, Any]]:
        """The same opening the episode starts from.

        Overridden rather than left to ``prompt_parts`` so the rendered prompt
        the run logs and the first turn the model actually receives are the same
        bytes; the default would assemble a house-style prompt that this mode
        then never sends.
        """
        messages, _state = self.interactive_start(sample)
        return messages, {"answer_prefix": "SOLVE:", "style": "free_form"}

    def _commit(self, state: dict[str, Any], assistant_text: str) -> str | None:
        """Record a SOLVE and end the episode."""
        state["submitted"] = True
        state["code"] = extract_code(assistant_text)
        return None

    def interactive_step(
        self, sample: SampleSpec, state: dict[str, Any], assistant_text: str
    ) -> str | None:
        text = assistant_text or ""
        if _SOLVE.search(text):
            return self._commit(state, text)
        reply = self.evidence_reply(sample, state, text)
        if reply is None:
            state["parser_errors"] = state.get("parser_errors", 0) + 1
            return PARSE_ERROR_REPLY
        return reply

    def evidence_reply(
        self, sample: SampleSpec, state: dict[str, Any], text: str
    ) -> str | None:
        """The Game Master's answer to an evidence request, or ``None`` if unreadable."""
        raise NotImplementedError


class AlienAbductionActiveAdapter(_MultiTurnAlienAdapter):
    """Active-Output: the model chooses the inputs it wants to see.

    The paper's own protocol prompt (Figure 13, template D.1) and its own reply
    format, unmodified.
    """

    paper_mode = "Active-Output"
    data_delivery_mode = "interactive"
    preloaded = 0

    setup_line = "Active Inputs"
    protocol_lines = (
        "Action per turn: TEST: <inputs>",
        "                 or SOLVE: ```python ... ```",
        "GM reply format: OUTPUT: <value>",
    )

    mode_decisions = (
        "Quoted the release's protocol prompt and reply format, so this mode runs as a single "
        "prompt set (io only). A model's probes are the thing being measured here, and they are "
        "a response to the protocol's exact wording.",
        "Type-checked each TEST against the target's domain and answered an unreadable or "
        "ill-typed request with the paper's own 'OUTPUT: Parse Error Invalid Input', which costs "
        "the turn. The paper reports parser errors as a property of the models (Table 5), so "
        "they are counted rather than silently repaired.",
    )
    mode_caveats = (
        "The model may probe the same input repeatedly; nothing stops it, and the turn is spent "
        "either way. That is the paper's protocol, and evidence_requests reports it.",
    )

    def evidence_reply(
        self, sample: SampleSpec, state: dict[str, Any], text: str
    ) -> str | None:
        match = _TEST.search(text)
        if match is None:
            return None
        args = _parse_test_input(str(sample.metadata["domain"]), match.group(1).splitlines()[0])
        if args is None:
            return None
        target = T.BY_NAME[str(sample.reference["target"])]
        try:
            output = target.fn(*args)
        except Exception:  # noqa: BLE001 - a probe the target cannot take is a bad probe
            return None
        state["evidence_requests"] = state.get("evidence_requests", 0) + 1
        state["seen"].append([list(args), output])
        return f"OUTPUT: {output!r}"


class AlienAbductionPassiveAdapter(_MultiTurnAlienAdapter):
    """Passive-Output: the Game Master chooses, one pair per request.

    The paper's own protocol prompt (Figure 13, template D.3).  The model spends
    a turn on ``NEXT:`` to be shown one more valid pair, or commits.
    """

    paper_mode = "Passive-Output"
    data_delivery_mode = "sequential"
    preloaded = 0

    setup_line = "Passive Examples"
    protocol_lines = (
        "Action per turn: NEXT: or SOLVE: ```python ... ```",
        "The Game Master reveals one valid pair at a time.",
        "GM reply format: OUTPUT: (<input>, <output>)",
    )

    mode_decisions = (
        "Quoted the release's protocol prompt and reply format, so this mode runs as a single "
        "prompt set (io only), for the same reason as the active mode.",
        "Revealed pairs in the same seeded per-target order the single-turn mode takes its batch "
        "from, so the two modes differ in how the evidence arrives rather than in what it is.",
    )
    mode_caveats = (
        "The turn budget caps this mode at fourteen revealed pairs (one turn has to be spent "
        "committing), against ten shown up front in the single-turn mode. The asymmetry is the "
        "paper's own design: sequential access is what the comparison is about.",
        "A Logic target runs out of pairs after four; the Game Master says so rather than "
        "repeating itself, and the model still has turns left to commit in.",
    )

    def evidence_reply(
        self, sample: SampleSpec, state: dict[str, Any], text: str
    ) -> str | None:
        if not _NEXT.search(text):
            return None
        cases = sample.metadata["_cases"]
        order = sample.metadata["_order"]
        index = state.get("evidence_requests", 0)
        if index >= len(order):
            return (
                "OUTPUT: no further pairs remain. Submit your solution with "
                "SOLVE: ```python ... ```"
            )
        args, output = cases[order[index]]
        state["evidence_requests"] = index + 1
        state["seen"].append([list(args), output])
        return f"OUTPUT: {_render_pair(tuple(args), output)}"
