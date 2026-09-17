"""Prompt mechanics shared by child adapters -- content stays with the dataset.

The division of labour here is the point of item 4 of the specification:

* **The core owns nothing.**  There is no universal system prompt anywhere in
  ``abductionbench.core``; the engine renders whatever the adapter hands it.
* **Each dataset owns its wording.**  A child adapter sets ``system_prompt`` (and
  may override :meth:`prompt_parts`) to say what its task is, in the terms its
  own benchmark uses.
* **The modes own the scaffolding.**  How an answer is *elicited* -- answer
  directly, reason first, vote over samples, pick one label, pick several, or
  answer yes/no about one hypothesis at a time -- is identical across datasets,
  so it is written once here.  Two datasets in the same mode therefore differ
  only where they should: in the task description.

Keeping the scaffolding in one place is also what makes the ``prompt_mode`` and
``selection_mode`` columns mean something: a row labelled ``cot`` got the same
reasoning instruction whichever dataset it came from.

**Every string appears in exactly one layer, and only one layer talks about
reasoning.**  A prompt is built from five, in this order:

1. the system prompt -- what this dataset's task is;
2. the evidence -- background, observation, question;
3. ``Task:`` and the ``Requirements:`` block -- what the task asks for, stated
   once, rendered identically in every prompt mode and every selection mode
   because a task requirement is a property of the benchmark, not of how the
   answer is elicited;
4. the mode instruction -- ``Answer directly. Do not explain your reasoning.``
   or ``Work through the evidence step by step...``.  **This is the only place
   in the whole prompt that says whether to reason**, which is what makes an
   ``io`` row and a ``cot`` row differ by exactly one line;
5. the closing -- the answer shape and the ``Answer:`` marker the parser reads.

Requirements therefore hold what the *task* demands ("use only the allowed
predicates", "do not restate the observation").  What the *answer line* looks
like belongs to ``answer_format`` and the closing, and what the *response* may
contain belongs to the mode instruction.  Nothing is said twice, so nothing can
contradict itself: the earlier design mixed all three into one per-dataset
``answer_constraints`` list, which put "do not explain" and "work through the
evidence step by step" into the same prompt on 27 datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from ..core.modes import BOV, COT, IO, MCS, SCS, SELF_CONSISTENCY, TaskModes
from ..core.types import ChatMessage

__all__ = ["PromptParts", "build_messages", "mode_instruction", "option_labels_for", "letters"]

#: The answer marker every scorer looks for.  One marker across the suite means
#: a dataset's scorer keeps working when its prompt mode changes.
ANSWER_PREFIX = "Answer:"

_COT_INSTRUCTION = (
    "Work through the evidence step by step before answering. Consider what each "
    "candidate explanation would have to be true for, and whether it accounts for "
    "everything observed rather than only part of it."
)

#: Chain-of-thought for a BOV request, which has one candidate rather than a
#: list. The general instruction says "consider what *each* candidate would have
#: to be true for", which under BOV points at candidates the model was not shown.
_COT_INSTRUCTION_BOV = (
    "Work through the evidence step by step before answering. Consider what this "
    "hypothesis would have to be true for, and whether it accounts for everything "
    "observed rather than only part of it."
)

_IO_INSTRUCTION = "Answer directly. Do not explain your reasoning."

#: The closing for a dataset whose answer is genuinely several lines.  "On the
#: last line" and "one fact per line" cannot both be obeyed, and the dataset that
#: asks for both was telling the model to do two incompatible things; the parser
#: was never the problem -- ``extract_answer_span`` already returns everything
#: after the final marker, newlines included.
_ANSWER_BLOCK_ONLY = "Put nothing in that block but the answer itself, one item per line."

#: The one sentence that replaces the seventeen per-dataset "output only the
#: diagnosis name" / "output only the formula" clauses.  Said once, in shared
#: wording, and scoped to the marker rather than to the response -- which is
#: what lets it be true in ``io`` and ``cot`` alike, where a per-dataset
#: "output only the fact" had to be either dropped or left to argue with the
#: reasoning instruction.
_ANSWER_LINE_ONLY = "Write the answer itself after that marker and nothing else."


def mode_instruction(modes: TaskModes, *, one_at_a_time: bool = False) -> str:
    """The one line in a prompt that says whether to reason.

    Exposed for the interactive adapters, whose prompts this module does not
    assemble: one of them writes its own interview prompt and so has to carry
    the mode instruction itself, or its io and cot tasks are the same bytes
    under two labels.  The adapters that quote their release's prompt instead
    run a single mode and never call this.
    """
    if modes.prompt_mode in (COT, SELF_CONSISTENCY):
        return _COT_INSTRUCTION_BOV if one_at_a_time else _COT_INSTRUCTION
    return _IO_INSTRUCTION


def _requirements_block(requirements: list[str]) -> str:
    """The task's requirements, worded and ordered the same for every dataset.

    Not mode-scoped, and deliberately so.  These are the benchmark's own
    demands -- how many hypotheses, which symbols, what must not be restated --
    and a demand that changes when the prompt mode changes was never a property
    of the task.  Anything that *is* a property of the mode lives in the mode
    instruction instead, and nothing else in the prompt mentions reasoning.
    """
    if not requirements:
        return ""
    return "Requirements:\n" + "\n".join(f"- {clause}" for clause in requirements)


def letters(count: int, start: str = "A") -> list[str]:
    """``["A", "B", ...]``.

    Only for a dataset whose source data keys its options by letter and whose
    gold answer refers to that key.
    """
    first = ord(start)
    return [chr(first + index) for index in range(count)]


def numbers(count: int) -> list[str]:
    """``["1", "2", ...]`` -- the house labels for a candidate list."""
    return [str(index + 1) for index in range(count)]


def option_labels_for(sample_fields: dict[str, Any]) -> list[str]:
    """The labels the model is expected to answer with for this sample."""
    labels = sample_fields.get("option_labels")
    if labels:
        return [str(label) for label in labels]
    options = sample_fields.get("options") or []
    return numbers(len(options))


@dataclass(slots=True)
class PromptParts:
    """The dataset-owned content of one prompt.

    Everything here is written by the child adapter in its benchmark's own
    terms.  None of it is mode-specific: the same parts are reused across io /
    cot / self-consistency and across SCS / MCS / BOV.
    """

    #: The system instruction for this dataset's task.
    system: str
    #: The observation, case, trace or premise set -- the evidence to explain.
    observation: str
    #: Optional background shown before the observation.
    context: str = ""
    #: The dataset's own question, if it asks one.
    question: str = ""
    #: Extra task instructions from the benchmark (kept verbatim where the
    #: benchmark publishes them).
    instructions: str = ""
    #: What a well-formed answer looks like, in the dataset's terms
    #: (e.g. "a single diagnosis", "one equation in the given symbols").
    answer_format: str = ""
    #: Candidate hypotheses for a selection task.
    options: list[str] = field(default_factory=list)
    #: Labels for those options (defaults to A, B, C...).
    option_labels: list[str] = field(default_factory=list)
    #: Heading above the candidate list, in the dataset's own words
    #: ("Answer options:", "Candidate diagnoses:").  Defaults to
    #: "Candidate hypotheses:".
    options_heading: str = ""
    #: Whether the answer is several lines rather than one.  A dataset whose
    #: answer is a list of facts cannot put it "on the last line", so it closes
    #: with a block instead.  Off by default: one line is the right shape for
    #: every other dataset here, and it is the shape the marker was designed for.
    answer_is_block: bool = False
    #: What the *task* requires, one clause per item, rendered as a
    #: "Requirements:" list inside the task block -- "use only the allowed
    #: predicates", "do not restate the observation", "if no single fact works,
    #: output exactly: None".  Writing them as data rather than prose is what
    #: keeps them comparable across datasets.
    #:
    #: Deliberately NOT the place for the answer's shape (that is
    #: :attr:`answer_format`) or for whether the response may reason (that is
    #: the mode instruction).  A clause here is rendered unchanged in every
    #: mode, so anything that would have to be dropped under ``cot`` does not
    #: belong here.
    requirements: list[str] = field(default_factory=list)
    #: Extra keys merged into the output contract handed to the scorer.
    contract: dict[str, Any] = field(default_factory=dict)


def _observation_block(parts: PromptParts, modes: TaskModes | None = None) -> list[str]:
    block: list[str] = []
    if parts.context:
        block.append(f"Background:\n{parts.context}")
    if parts.observation:
        block.append(f"Observation:\n{parts.observation}")
    if parts.question:
        if modes is not None and modes.selection_mode == BOV and parts.options:
            # A BOV request shows one candidate and no list, so a dataset's own
            # "Which of these candidates...?" would be asking the model to pick
            # from something it was never shown -- the same failure the options
            # guard in _closing exists to prevent, one block higher up. The
            # question is rendered as the standing question this one candidate
            # is being tested against instead of as the ask.
            block.append(f"The question being asked of the candidates:\n{parts.question}")
        else:
            block.append(f"Question: {parts.question}")
    if parts.instructions:
        block.append(f"Task: {parts.instructions}")
    requirements = _requirements_block(parts.requirements)
    if requirements:
        # With the task, not with the closing, and in every mode: these say what
        # the benchmark asks for, which does not change because the answer is
        # being elicited differently. Rendered before the mode instruction so
        # the last thing read before the answer format is whether to reason.
        block.append(requirements)
    return block


def _options_block(parts: PromptParts, labels: list[str]) -> str:
    lines = [f"{label}. {option}" for label, option in zip(labels, parts.options, strict=False)]
    return (parts.options_heading or "Candidate hypotheses:") + "\n" + "\n".join(lines)


def _label_list(labels: list[str]) -> str:
    """``"1, 2 or 3"`` -- how the closing line names the admissible answers."""
    if not labels:
        return "the label"
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + " or " + labels[-1]


def _closing(parts: PromptParts, modes: TaskModes, labels: list[str]) -> tuple[str, dict[str, Any]]:
    """The answer instruction and the contract that parses what it asks for.

    One wording per (selection mode, task shape) across the whole suite.  The
    point is comparability: a dataset should be hard because its data is hard,
    not because its answer instruction was clearer than its neighbour's.  So
    the shape is fixed here -- say what to answer with, name the admissible
    answers, and require the answer on its own last line behind a marker --
    and only the dataset's own nouns vary.

    The ``Answer:`` marker is load-bearing, not decoration.  Numbered labels
    are far easier to confuse with numbers that appear in reasoning than
    letters were, and the marker is what lets the parser take the label the
    model *submitted* rather than the last digit it happened to write.
    """
    contract: dict[str, Any] = {"answer_prefix": ANSWER_PREFIX, "strip_markdown": True}

    # THE SCAFFOLDING FOLLOWS THE CONTENT, NOT THE DECLARATION. A selection
    # closing tells the model to answer with a label from a list; if the prompt
    # never showed a list, the instruction is a lie and the example is worse
    # than useless.
    #
    # Observed on aiops2025, which asks for a root-cause entity by name but
    # declares selection_cardinality = "single", so the engine ran it as SCS.
    # With no options the closing rendered as:
    #
    #     Select exactly one hypothesis.
    #     Answer with only one of: the label, on the last line, as:
    #     Answer: 1
    #
    # -- "the label" being the empty-list fallback of _label_list. The model was
    # shown a numeric example for an answer that is a service name, and duly
    # replied "Answer: 1" instead of "Answer: inventory". The contract was also
    # set to style=single_label with an empty label set, so the parser was
    # configured for a label that could never arrive.
    #
    # A dataset that renders no candidate list gets the free-form closing, which
    # is built from its own answer_format and constraints, whatever selection
    # mode the run is labelled with.
    if not parts.options:
        modes = replace(modes, selection_mode=None)

    if modes.selection_mode == BOV:
        # One hypothesis at a time; the selected set is rebuilt from the yeses.
        #
        # Deliberately NOT "is this the best explanation": the model is shown a
        # single candidate and cannot see the others, so a superlative asks it
        # to rank against an invisible field and the answer would depend on what
        # it imagined the alternatives to be. The question BOV actually poses is
        # whether this hypothesis, judged alone, accounts for the observation.
        contract.update({"style": "binary", "labels_from_field": None, "yes_no": True})
        asks = "answers that question" if parts.question else "explains the observation"
        return (
            "You are shown one candidate hypothesis at a time; the others are not listed "
            "here, so judge this one on its own merits rather than against them.\n"
            f"Decide whether this hypothesis {asks}.\n"
            "Answer with only YES or NO, on the last line, as:\n"
            f"{ANSWER_PREFIX} YES"
        ), contract

    if modes.selection_mode == MCS:
        contract.update({"style": "multi_label", "labels_from_field": "option_labels"})
        example = ", ".join(labels[:2]) if len(labels) > 1 else (labels[0] if labels else "1")
        return (
            "Select every hypothesis that applies -- there may be one or several.\n"
            f"Answer with only their numbers, separated by commas, on the last line, as:\n"
            f"{ANSWER_PREFIX} {example}"
        ), contract

    if modes.selection_mode == SCS:
        contract.update({"style": "single_label", "labels_from_field": "option_labels"})
        return (
            "Select exactly one hypothesis.\n"
            f"Answer with only one of: {_label_list(labels)}, on the last line, as:\n"
            f"{ANSWER_PREFIX} {labels[0] if labels else '1'}"
        ), contract

    contract.update({"style": "free_form"})
    shape = parts.answer_format or "your answer"
    if parts.answer_is_block:
        return (
            f"End your reply with the answer block:\n{ANSWER_PREFIX}\n<{shape}>\n"
            f"{_ANSWER_BLOCK_ONLY}"
        ), contract
    return (
        f"On the last line, give your final answer as:\n{ANSWER_PREFIX} <{shape}>\n"
        f"{_ANSWER_LINE_ONLY}"
    ), contract


def build_messages(
    parts: PromptParts, modes: TaskModes
) -> tuple[list[ChatMessage], dict[str, Any]]:
    """Compose the conversation for one sample in the active modes.

    ``self-consistency`` deliberately renders the *same* text as ``cot``: it is
    chain-of-thought sampled several times with a majority vote, so a different
    prompt would confound the two.  What differs is the decoding temperature and
    the vote, both of which the engine applies.
    """
    labels = parts.option_labels or option_labels_for(
        {"options": parts.options, "option_labels": parts.option_labels}
    )

    body = _observation_block(parts, modes)
    if parts.options and modes.selection_mode == BOV:
        # One hypothesis at a time, and it has to be on the page: the model is
        # being asked about *this* candidate, not about the list it came from.
        body.append(f"Hypothesis under consideration:\n{parts.options[0]}")
    elif parts.options:
        body.append(_options_block(parts, labels))

    if modes.prompt_mode in (COT, SELF_CONSISTENCY):
        one_at_a_time = bool(parts.options) and modes.selection_mode == BOV
        body.append(_COT_INSTRUCTION_BOV if one_at_a_time else _COT_INSTRUCTION)
    elif modes.prompt_mode == IO:
        body.append(_IO_INSTRUCTION)

    closing, contract = _closing(parts, modes, labels)
    body.append(closing)
    contract["option_labels"] = labels
    contract.update(parts.contract)

    messages: list[ChatMessage] = []
    system = (parts.system or "").strip()
    if system:
        messages.append(ChatMessage(role="system", content=system))
    messages.append(ChatMessage(role="user", content="\n\n".join(b for b in body if b).strip()))
    return messages, contract
