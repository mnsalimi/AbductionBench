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

import re

from dataclasses import dataclass, field, replace
from typing import Any

from ..core.modes import BOV, COT, MCS, SCS, SELF_CONSISTENCY, TaskModes
from ..core.types import ChatMessage

__all__ = [
    "PromptParts", "build_messages", "mode_instruction", "option_labels_for", "letters",
    "ProtocolParts", "build_protocol_messages", "requirements_block",
]

#: The answer marker every scorer looks for.  One marker across the suite means
#: a dataset's scorer keeps working when its prompt mode changes.
#: Retained for the interactive protocols and the adapters that quote their
#: release's own prompt, which still ask for a marker. The static closings no
#: longer use it -- see ANSWER_TAG below.
ANSWER_PREFIX = "Answer:"

#: The tags a static reply is built from.
#:
#: WHY TAGS RATHER THAN A MARKER. "Answer:" is a string a model can write in
#: the middle of its reasoning, and under cot it often did -- the parser took
#: the LAST occurrence precisely because an earlier one was usually a false
#: positive. A closing tag cannot be produced by accident in the same way, it
#: delimits the answer at both ends rather than only the front, and it gives
#: the reasoning its own container instead of leaving it as "everything above
#: the marker".
ANSWER_OPEN, ANSWER_CLOSE = "<answer>", "</answer>"
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"

#: Finds the LAST answer block. Greedy prefix, then non-greedy body: a reply
#: that mentions the tag while reasoning still parses to what it finally
#: submitted, which is the same property the marker parser had.
ANSWER_TAG_REGEX = r"(?s).*<answer>\s*(?P<answer>.*?)\s*</answer>"
THINK_TAG_REGEX = r"(?s)<think>\s*(?P<think>.*?)\s*</think>"


def _meaningful(text: str | None) -> str:
    """``text`` stripped, or empty if it holds no word at all.

    A think block of ``.`` -- which gpt-5.6-luna returns when its reasoning
    went to the provider's own field -- is a model filling in a required tag,
    not a chain of one step.
    """
    body = (text or "").strip()
    return body if re.search(r"\w", body) else ""


#: A think block, closed or not. A reply cut off by its token budget opens the
#: block and never closes it, and what it did write is still visible reasoning.
_VISIBLE_COT_REGEX = r"(?s)<think>\s*(?P<think>.*?)\s*(?:</think>|\Z)"

#: Tags that only mean something to this harness: the reply's own channels and
#: the container the step judge reads the trace from. None of them is a word
#: the model reasoned with, so none may reach the judge as text it could cut a
#: step from.
_HARNESS_TAG_REGEX = r"</?\s*(?:think|answer|reasoning_chain|reasoning_trace)\s*>"

#: Where a trace came from, as recorded beside every reasoning metric.
TRACE_VISIBLE, TRACE_NATIVE, TRACE_UNTAGGED, TRACE_NONE = (
    "visible_cot",
    "native_reasoning",
    "untagged_prose",
    "none",
)


@dataclass(frozen=True)
class ReasoningTrace:
    """The one chain a reply's reasoning metrics are measured on, and its source."""

    text: str
    source: str


def _clean_trace(text: str) -> str:
    """The trace without answer blocks or harness tags, placeholders counted empty."""
    text = re.sub(r"(?s)<answer>.*?</answer>", "", text)
    text = re.sub(_HARNESS_TAG_REGEX, "", text, flags=re.IGNORECASE)
    return _meaningful(text)


def reasoning_trace(content: str | None, reasoning: str | None) -> ReasoningTrace:
    """Pick ONE source for a reply's chain of thought -- never both.

    A model can reason in the visible ``<think>`` block, the provider can
    return reasoning in its own field (``message.reasoning`` /
    ``reasoning_content``), a reply can carry both, or neither. They used to be
    concatenated, and where a reply carried both that was usually the same
    inferential process twice -- gemini-3.8-flash's native field is a
    provider-written summary of the thinking the visible block then spells out
    -- so the step judge segmented it twice and every step-based metric was
    inflated. Precedence:

    ===================  =====================================  ================
    native field         content                                trace
    ===================  =====================================  ================
    "R"                  ``<think>T</think><answer>…``          "T" (visible)
    empty                ``<think>T</think><answer>…``          "T" (visible)
    "R"                  ``<answer>…``                          "R" (native)
    empty                ``<answer>…``                          "" (none)
    ===================  =====================================  ================

    Visible CoT wins whenever there is any: it is what the model wrote where it
    was asked to, and the native field is at best a second route to the same
    chain and at worst a summary of it. A placeholder (``<think>.</think>``,
    which gpt-5.6-luna writes when its chain went to the native field) is no
    visible CoT. A reply with no tags at all and no native field -- one written
    before the tagged format -- contributes its prose as a last resort.

    The trace never carries the answer block or any harness tag, so neither
    can come back from the step judge as a step. The reply's content and its
    native field are untouched on the record; this only chooses what is read.
    """
    content = content or ""
    think = re.search(_VISIBLE_COT_REGEX, content)
    visible = _clean_trace(think.group("think")) if think else ""
    if visible:
        return ReasoningTrace(visible, TRACE_VISIBLE)
    native = _clean_trace(reasoning or "")
    if native:
        return ReasoningTrace(native, TRACE_NATIVE)
    if not think:
        prose = _clean_trace(content)
        if prose:
            return ReasoningTrace(prose, TRACE_UNTAGGED)
    return ReasoningTrace("", TRACE_NONE)


def cot_chain(content: str | None, reasoning: str | None) -> str:
    """The chain of thought of a reply -- :func:`reasoning_trace`'s text."""
    return reasoning_trace(content, reasoning).text

_COT_INSTRUCTION = (
    "Reason and explain explicitly by working through the evidence step by step "
    "before answering. Consider what each "
    "candidate explanation would have to be true for, and whether it accounts for "
    "everything observed rather than only part of it."
)

#: Chain-of-thought for a BOV request, which has one candidate rather than a
#: list. The general instruction says "consider what *each* candidate would have
#: to be true for", which under BOV points at candidates the model was not shown.
_COT_INSTRUCTION_BOV = (
    "Reason and explain explicitly by working through the evidence step by step "
    "before answering. Consider what this "
    "hypothesis would have to be true for, and whether it accounts for everything "
    "observed rather than only part of it."
)

_IO_INSTRUCTION = "Answer directly. Do not explain your reasoning."

#: The format block, which ends the system prompt in both modes.
#:
#: It sits there, immediately after the mode instruction, because the two say
#: one thing between them: whether to reason, and where the reasoning and the
#: answer go. Splitting them left the format half in the user turn, next to the
#: options, where it had to be re-stated per selection mode and drifted out of
#: agreement with the mode instruction above it -- io prompts that said "answer
#: directly" and then "on the last line", cot prompts that restricted the whole
#: response.
#:
#: The two modes do NOT share a format any more, and that is the point: an io
#: reply has no reasoning to contain, so naming <think> there would invite one.
_FORMAT_IO = (
    "Reply with exactly one answer block and nothing else:\n"
    f"{ANSWER_OPEN}your answer{ANSWER_CLOSE}\n"
    "Write nothing before it and nothing after it. Do not include a "
    f"{THINK_OPEN} block."
)

_FORMAT_COT = (
    "Put all of your reasoning inside one think block, then give your answer "
    "inside one answer block:\n"
    f"{THINK_OPEN}your reasoning{THINK_CLOSE}\n"
    f"{ANSWER_OPEN}your answer{ANSWER_CLOSE}\n"
    "Write nothing before the think block and nothing after the answer block, "
    "and put the answer only inside the answer block -- not inside the think "
    "block as well."
)


def format_compliance(
    text: str | None, modes: TaskModes, reasoning: str | None = None
) -> bool:
    """Did the reply obey the format block exactly?

    REPORTED SEPARATELY FROM CORRECTNESS, and that separation is the point. A
    model can answer correctly in the wrong shape, and a harness that folds the
    two together cannot tell "wrong about the task" from "wrote it another
    way" -- which is a difference about the model, not about the benchmark.
    The parser stays lenient so a sloppy reply is still scored; this says how
    often it had to be.

    Strict means strict: exactly one answer block, nothing outside the blocks,
    and under cot exactly one think block before it. Whitespace between them is
    allowed and nothing else is.

    UNDER COT, THE PROVIDER'S REASONING FIELD IS THE THINK BLOCK. A provider
    that supports reasoning mode takes the chain out of the content and
    returns it separately, so the content is a bare answer block through no
    choice of the model's -- the same chain :func:`cot_chain` reads from that
    field. Counting it as a violation reported every reasoning model as
    ignoring the format. A reply with neither a think block nor a reasoning
    field did not reason where it was asked to, and still is one.
    """
    body = (text or "").strip()
    if not body:
        return False
    cot = modes.prompt_mode in (COT, SELF_CONSISTENCY)
    if cot and THINK_OPEN not in body and _meaningful(reasoning):
        # Held to the io shape: one answer block and nothing else.
        return format_compliance(body, TaskModes(prompt_mode="io"))
    if cot:
        pattern = (
            rf"\A{THINK_OPEN}(?P<think>.*?){THINK_CLOSE}\s*"
            rf"{ANSWER_OPEN}(?P<answer>.*?){ANSWER_CLOSE}\Z"
        )
    else:
        pattern = rf"\A{ANSWER_OPEN}(?P<answer>.*?){ANSWER_CLOSE}\Z"
    match = re.fullmatch(pattern, body, flags=re.DOTALL)
    if match is None:
        return False
    # One block each: a reply that opens a second answer block inside the first
    # matched the non-greedy body above but is not the shape that was asked for.
    return body.count(ANSWER_OPEN) == 1 and body.count(THINK_OPEN) == (1 if cot else 0)


def format_segment(modes: TaskModes) -> str:
    """The output-format block that ends the system prompt.

    One per mode, deliberately not one shared template: io and cot no longer
    ask for the same shape, and a single block that tried to cover both is what
    produced instructions arguing with each other.
    """
    if modes.prompt_mode in (COT, SELF_CONSISTENCY):
        return _FORMAT_COT
    return _FORMAT_IO


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


def _label_noun(labels: list[str]) -> str:
    """What to call these labels in an instruction: "numbers" or "letters".

    The MCS closing used to say "answer with only their numbers" whatever the
    labels were, so three datasets keyed by letter -- ddxplus, scir,
    true_detective -- were shown A, B, C and told to reply with numbers. The
    noun now follows the labels actually rendered, which is the only thing that
    can keep the two in step as datasets come and go.
    """
    if labels and all(str(label)[:1].isalpha() for label in labels):
        return "letters"
    return "numbers"


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
    # EMITTED AND PARSED FROM ONE PLACE. The system prompt asks for
    # <answer>...</answer>; this is what reads it back. They cannot drift,
    # because nothing else sets either one.
    contract: dict[str, Any] = {
        "answer_regex": ANSWER_TAG_REGEX,
        "strip_markdown": True,
    }

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
            # The admissible answers, and nothing about where they go -- the
            # system prompt's format block owns that now, in both modes. Both
            # options are named rather than one: an example showing only YES on
            # every BOV prompt is a standing nudge toward it, and naming both is
            # safe precisely because it is symmetric.
            "Answer with only YES or NO."
        ), contract

    if modes.selection_mode == MCS:
        contract.update({"style": "multi_label", "labels_from_field": "option_labels"})
        # A placeholder, not the first labels. A selection dataset reuses the
        # same small pool -- A/B/C, 1/2/3 -- on every sample, so an example
        # built from real labels shows the model "Answer: A, B" on all 150 of
        # them. That is a constant, and a constant next to the answer line is
        # a prior. Even an "e.g." would carry it, because the letters would be
        # the same letters every time.
        return (
            "Select every hypothesis that applies -- there may be one or several.\n"
            f"Answer with only their {_label_noun(labels)}, separated by commas."
        ), contract

    if modes.selection_mode == SCS:
        contract.update({"style": "single_label", "labels_from_field": "option_labels"})
        # "Answer with only one of" forbids the explanation cot has just asked
        # for, so under cot the restriction is scoped to the final answer
        # instead of to the whole reply. io keeps the answer-only form.
        # ONE WORDING FOR BOTH MODES NOW. They used to differ because "answer
        # with only one of" forbade the explanation cot had just asked for; with
        # the answer in its own block that conflict is gone -- the restriction is
        # on what goes in the block, which is true in io and cot alike.
        lead = f"Answer with only one of: {_label_list(labels)}."
        # Same reasoning as MCS: `labels[0]` is the *same* label on every
        # sample of the dataset, so it showed "Answer: A" -- or "Answer: 1" --
        # beside every question the model was ever asked.
        return ("Select exactly one hypothesis.\n" + lead), contract

    contract.update({"style": "free_form"})
    shape = parts.answer_format or "your answer"
    if parts.answer_is_block:
        # Same split as the one-line closing: under cot the block genuinely
        # ends a reply that has reasoning above it, under io it is the reply.
        # A multiline answer is still one answer block; what changes is what
        # goes inside it, so this says that and nothing about the reply's shape.
        return f"Give {shape}, one item per line.", contract
    return f"Give {shape}.", contract


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

    closing, contract = _closing(parts, modes, labels)
    body.append(closing)
    contract["option_labels"] = labels
    contract.update(parts.contract)

    # The mode instruction ends the system prompt, in every dataset and every
    # mode. It used to sit in the user turn between the requirements and the
    # closing, which put the one line that says whether to reason in the middle
    # of the task description; at the end of the system prompt it is the last
    # standing instruction before the question, and there is exactly one of it.
    one_at_a_time = bool(parts.options) and modes.selection_mode == BOV
    mode_line = mode_instruction(modes, one_at_a_time=one_at_a_time)
    # The format block comes LAST, straight after the mode instruction, because
    # the two are one statement: whether to reason, and where the reasoning and
    # the answer go. Anywhere else and they drift -- which is what happened when
    # the format lived in the user turn beside the options.
    messages: list[ChatMessage] = []
    system = "\n\n".join(
        part for part in ((parts.system or "").strip(), mode_line, format_segment(modes))
        if part
    )
    if system:
        messages.append(ChatMessage(role="system", content=system))
    messages.append(ChatMessage(role="user", content="\n\n".join(b for b in body if b).strip()))
    return messages, contract


# --------------------------------------------------------------------------- #
# interactive protocols -- the same five layers, for a benchmark that is a loop
# --------------------------------------------------------------------------- #

#: Heading for the action vocabulary.  It sits where a static prompt puts its
#: candidate list: after the task, before the mode instruction, because what the
#: model may *do* is part of the task rather than part of how it answers.
_ACTIONS_HEADING = "Available actions:"


@dataclass(slots=True)
class ProtocolParts:
    """The dataset-owned content of one interactive benchmark's opening turn.

    The static builder above assembles a question; this one assembles a
    *protocol*, and it keeps the same five layers in the same order so that an
    interactive prompt reads like the rest of the suite and can be audited the
    same way:

    1. :attr:`system` -- what this benchmark's task is, in its own terms;
    2. :attr:`context` / :attr:`observation` -- the case, the incident, the stem;
    3. ``Task:`` and ``Requirements:`` -- what the benchmark demands, including
       its workflow gates and stopping conditions, stated once;
    4. the mode instruction -- the *only* line that says whether to reason, and
       it is the shared one, unmodified;
    5. the closing -- :attr:`actions` and :attr:`output_format`, the exact shape
       the environment's parser reads.

    What deliberately has no home here is a reasoning field, a "thought" line or
    a request to explain a choice.  Those are the upstream projects' *reasoning
    elicitation*, not their task definition, and carrying them over would put
    "explain your reasoning" and "Answer directly. Do not explain your
    reasoning." into the same prompt -- the exact contradiction this module
    exists to prevent.  The action vocabulary, the evidence rules, the gates,
    the limits and the submission shape are the benchmark and are all kept.
    """

    #: The system instruction: this benchmark's task, in this suite's voice.
    system: str
    #: The case, incident or stem the episode opens on.
    observation: str = ""
    #: Background shown before the observation.
    context: str = ""
    #: What the benchmark asks for, as a ``Task:`` line.
    instructions: str = ""
    #: The benchmark's own demands: workflow gates, budgets, stopping
    #: conditions.  Rendered as a ``Requirements:`` list, one clause each.
    requirements: list[str] = field(default_factory=list)
    #: ``(name, what it does)`` for every action the environment accepts, in the
    #: order the benchmark lists them.
    actions: list[tuple[str, str]] = field(default_factory=list)
    #: The exact output shape the environment parses -- the closing. This is a
    #: format, never an instruction to reason.
    output_format: str = ""


def requirements_block(requirements: list[str]) -> str:
    """The ``Requirements:`` block, for a turn this module does not assemble.

    An interactive benchmark restates its rules on later turns; rendering them
    through the same function as the opening is what keeps a mid-episode
    message in the same shape as the prompt that started it.
    """
    return _requirements_block(requirements)


def _actions_block(actions: list[tuple[str, str]]) -> str:
    if not actions:
        return ""
    lines = [f"- {name}: {what}" for name, what in actions]
    return _ACTIONS_HEADING + "\n" + "\n".join(lines)


def build_protocol_messages(
    parts: ProtocolParts, modes: TaskModes
) -> list[ChatMessage]:
    """The opening conversation for an interactive benchmark.

    Returns messages only: an interactive adapter owns its own state, and the
    answer contract for these benchmarks is the protocol itself rather than an
    ``Answer:`` marker.
    """
    body: list[str] = []
    if parts.context:
        body.append(f"Background:\n{parts.context}")
    if parts.observation:
        body.append(f"Observation:\n{parts.observation}")
    if parts.instructions:
        body.append(f"Task: {parts.instructions}")
    requirements = _requirements_block(parts.requirements)
    if requirements:
        body.append(requirements)
    actions = _actions_block(parts.actions)
    if actions:
        body.append(actions)
    # The one line in the prompt that talks about reasoning, and it is the
    # suite's own -- unchanged here, exactly as a static prompt gets it.
    body.append(mode_instruction(modes))
    if parts.output_format:
        body.append(parts.output_format)

    messages: list[ChatMessage] = []
    system = (parts.system or "").strip()
    if system:
        messages.append(ChatMessage(role="system", content=system))
    messages.append(ChatMessage(role="user", content="\n\n".join(b for b in body if b).strip()))
    return messages
