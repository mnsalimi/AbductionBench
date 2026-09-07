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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.modes import BOV, COT, IO, MCS, SCS, SELF_CONSISTENCY, TaskModes
from ..core.types import ChatMessage

__all__ = ["PromptParts", "build_messages", "option_labels_for", "letters"]

#: The answer marker every scorer looks for.  One marker across the suite means
#: a dataset's scorer keeps working when its prompt mode changes.
ANSWER_PREFIX = "Answer:"

_COT_INSTRUCTION = (
    "Work through the evidence step by step before answering. Consider what each "
    "candidate explanation would have to be true for, and whether it accounts for "
    "everything observed rather than only part of it."
)

_IO_INSTRUCTION = "Answer directly. Do not explain your reasoning."


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
    #: What a well-formed answer must and must not do, one clause per item,
    #: rendered as a "Requirements:" list.  This is where a generation task
    #: gets its shape -- one sentence, no restating the observation, no
    #: explaining why -- and writing it as data rather than prose is what keeps
    #: those constraints comparable across datasets.
    constraints: list[str] = field(default_factory=list)
    #: Extra keys merged into the output contract handed to the scorer.
    contract: dict[str, Any] = field(default_factory=dict)


def _observation_block(parts: PromptParts) -> list[str]:
    block: list[str] = []
    if parts.context:
        block.append(f"Background:\n{parts.context}")
    if parts.observation:
        block.append(f"Observation:\n{parts.observation}")
    if parts.question:
        block.append(f"Question: {parts.question}")
    if parts.instructions:
        block.append(f"Task: {parts.instructions}")
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

    if modes.selection_mode == BOV:
        # One hypothesis at a time; the selected set is rebuilt from the yeses.
        contract.update({"style": "binary", "labels_from_field": None, "yes_no": True})
        return (
            "Decide whether this is the best explanation of the observation.\n"
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
    lines = []
    if parts.constraints:
        lines.append("Requirements:")
        lines.extend(f"- {clause}" for clause in parts.constraints)
    if lines:
        lines.append("")
    lines.append(f"On the last line, give your final answer as:\n{ANSWER_PREFIX} <{shape}>")
    return "\n".join(lines), contract


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

    body = _observation_block(parts)
    if parts.options and modes.selection_mode == BOV:
        # One hypothesis at a time, and it has to be on the page: the model is
        # being asked about *this* candidate, not about the list it came from.
        body.append(f"Hypothesis under consideration:\n{parts.options[0]}")
    elif parts.options:
        body.append(_options_block(parts, labels))

    if modes.prompt_mode in (COT, SELF_CONSISTENCY):
        body.append(_COT_INSTRUCTION)
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
