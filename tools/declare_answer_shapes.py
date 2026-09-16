"""One-shot, SPENT: gave every static/sequential adapter its answer shape.

.. warning::

   **Do not run this again.**  Its table is a snapshot of the *superseded*
   design, in which one ``answer_constraints`` list per dataset mixed three
   different things: what the task demands, what the answer line looks like,
   and whether the response may reason.  That last one put "do not explain"
   into the same prompt as "work through the evidence step by step" on 27
   datasets.  Those three now live in ``task_requirements``, ``answer_format``
   and the mode instruction respectively (see ``adapters/_prompting.py``), so
   re-running this would reintroduce the conflict it is kept here to document.
   It is retained as the record of how the shapes were first declared, and it
   refuses to write.

Specification item 4 asks for prompts written from scratch for the static and
sequential datasets, in one house structure, so that a difference in score is a
difference in the difficulty of the data rather than in how firmly the
instruction happened to be worded.  Interactive datasets are deliberately
absent: those keep the prompts their own benchmark publishes.

The structure itself lives in ``adapters/_prompting.py`` -- numbered candidate
lists, one closing wording per selection mode, an ``Answer:`` marker on the
last line.  What varies per dataset is only what it is asking for, and that is
this table:

``answer_format``
    What a well-formed answer looks like, in the dataset's own nouns.  It is
    rendered inside the closing line: ``Answer: <one short sentence>``.
``constraints``
    What the answer must and must not do, one clause per line.  Superseded:
    the task-defining clauses of each list became ``task_requirements``, the
    shape clauses were folded into ``answer_format``, and the rest were
    deleted as duplicates of the mode instruction or of the closing.
``options_heading``
    What the candidate list is called where the dataset has one.

Ran once, on the design this replaced; kept for the record.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ADAPTERS = Path(__file__).resolve().parents[1] / "src" / "abductionbench" / "adapters"

# Clauses shared by many datasets, spelled once so they read identically.
ONE_SENTENCE = "write exactly one sentence"
NO_RESTATE = "do not restate the observation"
NO_WHY = "do not explain why"
NO_PREAMBLE = "do not use introductory phrases or commentary"
PLAIN = "keep it short, plain and concrete"
ONLY_ANSWER = "output only the answer, with nothing before or after it"

TABLE: dict[str, dict] = {
    # -- selection-only: the closing is shared, the heading is theirs ------ #
    "agentrx": {"options_heading": "Candidate root causes:"},
    "aiops2025": {"options_heading": "Candidate root causes:",
                  "answer_format": "the root cause",
                  "constraints": [ONE_SENTENCE, "name only the root cause", NO_PREAMBLE]},
    "causalopsbench": {"options_heading": "Candidate faulty components:",
                       "answer_format": "the faulty component",
                       "constraints": ["name only the component", NO_WHY, NO_PREAMBLE]},
    "gear": {"options_heading": "Answer options:"},
    "musr": {"options_heading": "Options:"},
    "scir": {"options_heading": "Answer options:"},
    "true_detective": {"options_heading": "Answer options:"},
    "xcopa": {"options_heading": "Answer options:"},
    "aer": {"options_heading": "Candidate explanations:"},
    "defab": {"options_heading": "Candidate explanations:"},

    # -- both tasks: a heading for selection, a shape for generation ------- #
    "art": {
        "options_heading": "Answer options:",
        "answer_format": "one short sentence",
        "constraints": [ONE_SENTENCE,
                        "output only the missing intermediate event or state",
                        PLAIN,
                        "make it plausible with both observations",
                        "do not restate either observation",
                        "do not add extra causes, background detail or consequences",
                        NO_WHY, NO_PREAMBLE],
    },
    "ecare": {
        "options_heading": "Answer options:",
        "answer_format": "one short factual statement",
        "constraints": [ONE_SENTENCE,
                        "name the underlying property, rule or definition",
                        "use plain factual wording",
                        "do not refer to the cause and effect themselves",
                        NO_WHY, NO_PREAMBLE],
    },
    "diagnosisarena": {
        "options_heading": "Candidate diagnoses:",
        "answer_format": "a single diagnosis",
        "constraints": ["give exactly one diagnosis",
                        "output only the diagnosis name", NO_WHY, NO_PREAMBLE],
    },
    "medups": {
        "options_heading": "Candidate diagnoses:",
        "answer_format": "a single diagnosis",
        "constraints": ["give exactly one diagnosis",
                        "output only the diagnosis name", NO_WHY, NO_PREAMBLE],
    },
    "researchbench": {
        "options_heading": "Candidate hypotheses:",
        "answer_format": "one testable hypothesis",
        "constraints": ["state one hypothesis, not several",
                        "make it specific enough to be tested",
                        "do not describe the method or the expected result",
                        NO_PREAMBLE],
    },

    # -- diagnosis generation --------------------------------------------- #
    "house_md": {"answer_format": "a single diagnosis",
                 "constraints": ["give exactly one diagnosis",
                                 "output only the diagnosis name", NO_WHY, NO_PREAMBLE]},
    "medcasereasoning": {"answer_format": "a single diagnosis",
                         "constraints": ["give exactly one diagnosis",
                                         "output only the diagnosis name", NO_WHY, NO_PREAMBLE]},
    "medr_bench": {"answer_format": "a single diagnosis",
                   "constraints": ["give exactly one diagnosis",
                                   "output only the diagnosis name", NO_WHY, NO_PREAMBLE]},

    # -- formal / symbolic: an exact string is the answer ------------------ #
    "abductionrules": {
        "answer_format": "one fact",
        "constraints": ["output exactly one fact",
                        "output only the fact",
                        "do not output a list",
                        "do not explain", NO_PREAMBLE],
    },
    "proof_writer": {
        "answer_format": "one fact per line, or None",
        "constraints": ["output only the missing facts",
                        "each missing fact must be a single fact, not a rule",
                        "if several single facts would each work, output all of them",
                        "put each answer on its own line",
                        "if there is no valid single missing fact, output exactly: None",
                        "do not explain your reasoning"],
    },
    "neulr": {
        "answer_format": "one missing fact",
        "constraints": ["output exactly one missing fact",
                        "the answer must be a single fact, not a rule",
                        "output only the missing fact",
                        "do not explain your reasoning"],
    },
    "enwn_entailmentbank": {
        "answer_format": "one missing premise",
        "constraints": ["output exactly one premise",
                        "output only the premise",
                        "do not restate the hypothesis or the given premises",
                        "do not explain your reasoning"],
    },
    "synpat": {
        "answer_format": "one equation in the given symbols",
        "constraints": ["output exactly one equation",
                        "use only the symbols given",
                        "output only the equation", NO_PREAMBLE]},
    "abd": {
        "answer_format": "one s-expression formula in the variable x",
        "constraints": ["output exactly one formula",
                        "use only the allowed predicates",
                        "leave x as the only free variable",
                        "output only the formula", NO_PREAMBLE]},
    # -- judged free-text generation -------------------------------------- #
    "uncommonsense": {
        "answer_format": "1 to 3 sentences",
        "constraints": ["write 1 to 3 sentences",
                        "make the outcome more likely, leaving as little of an "
                        "information gap as possible",
                        "do not restate the context or the outcome",
                        "write only the explanation", NO_PREAMBLE]},
    "commonwhy": {
        "answer_format": "one short sentence",
        "constraints": [ONE_SENTENCE, "give the reason itself",
                        NO_RESTATE, NO_PREAMBLE]},
    "crosstrace": {
        "answer_format": "one short hypothesis",
        "constraints": [ONE_SENTENCE,
                        "name the underlying fault, not its symptoms",
                        NO_RESTATE, NO_PREAMBLE]},
    "hypoarena": {
        "answer_format": "one hypothesis",
        "constraints": ["state one hypothesis, not several",
                        "make it specific and checkable against the observations",
                        NO_RESTATE, NO_PREAMBLE]},
    "hypobench": {
        "answer_format": "one hypothesis",
        "constraints": ["state one hypothesis, not several",
                        "name the feature and the direction of its effect",
                        NO_RESTATE, NO_PREAMBLE]},
    "hypogen": {
        "answer_format": "one hypothesis",
        "constraints": ["state one hypothesis, not several",
                        "say which condition changes the outcome and how",
                        NO_RESTATE, NO_PREAMBLE]},
    "matter_to_mechanism": {
        "answer_format": "one mechanism",
        "constraints": ["state one mechanism, not several",
                        "name the physical or chemical process responsible",
                        NO_RESTATE, NO_PREAMBLE]},
    "moose_chem2": {
        "answer_format": "one hypothesis",
        "constraints": ["state one hypothesis, not several",
                        "make it specific enough to be tested experimentally",
                        NO_RESTATE, NO_PREAMBLE]},
}


def _wrap(clauses: list[str], indent: str) -> str:
    return "\n".join(f'{indent}"{clause}",' for clause in clauses)


def block_for(spec: dict) -> str:
    lines: list[str] = []
    if spec.get("answer_format"):
        lines.append(f'    answer_format = "{spec["answer_format"]}"')
    if spec.get("constraints"):
        lines.append("    answer_constraints = (")
        lines.append(_wrap(spec["constraints"], "        "))
        lines.append("    )")
    if spec.get("options_heading"):
        lines.append(f'    options_heading = "{spec["options_heading"]}"')
    return "\n" + "\n".join(lines) + "\n" if lines else ""


def main() -> int:
    print(
        "declare_answer_shapes is spent: its table predates the "
        "task_requirements / answer_format split and would reintroduce the "
        "reasoning conflict it once created. Refusing to write.",
        file=sys.stderr,
    )
    return 1


def _historical_main() -> int:
    changed = 0
    for name, spec in sorted(TABLE.items()):
        path = ADAPTERS / f"{name}.py"
        if not path.exists():
            print(f"!! no adapter module for {name}", file=sys.stderr)
            continue
        source = path.read_text(encoding="utf-8")
        if "    answer_constraints = " in source or "    options_heading = " in source:
            print(f"== {name}: already declared, skipping")
            continue
        match = re.search(r"\n    data_delivery_mode = \"[^\"]+\"\n", source)
        if not match:
            print(f"!! {name}: no data_delivery_mode anchor", file=sys.stderr)
            continue
        source = source[: match.end()] + block_for(spec) + source[match.end():]
        path.write_text(source, encoding="utf-8")
        changed += 1
        print(f"-> {name}")
    print(f"declared {changed} adapter(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
