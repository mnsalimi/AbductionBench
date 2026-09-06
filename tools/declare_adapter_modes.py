"""One-shot: give every child adapter its own prompts and mode declarations.

Specification items 3, 4, 5 and 15 all land in the same place -- the class header
of each adapter -- so they are applied from one table rather than 40 hand edits,
which is also what makes them reviewable side by side.

Per dataset the table records:

``system``
    The dataset's own system prompt.  Written in the benchmark's terms, because
    the core no longer supplies one.
``delivery``
    static | interactive | sequential, taken from the dataset table's "Data
    Delivery Mode" column.
``objective``
    Whether the primary metric is decidable without an LLM judge -- exact match,
    label accuracy, set/graph F1, symbolic equivalence.  Free-text similarity
    (ROUGE against a reference explanation) is not: a paraphrase scores low and a
    fluent wrong answer scores high, so a reasoning mode's effect could not be
    read off it.  Only objective datasets offer cot / self-consistency.
``cardinality``
    single | multi | flexible | None, from the benchmark's own task definition.
``hypothesis``
    Which abductive tasks the benchmark poses as independent evaluations, the
    dataset options that select them, what the published table lists, and -- when
    a mode is introduced beyond the table -- the benchmark formulation that
    justifies it.

Run once:  python tools/declare_adapter_modes.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADAPTERS = ROOT / "src" / "abductionbench" / "adapters"

# --------------------------------------------------------------------------- #
# The table.  system prompts are deliberately concrete: each names the evidence
# the model is given and what a good hypothesis has to do with it.
# --------------------------------------------------------------------------- #

ABDUCTIVE_CORE = (
    "You are an expert at abductive reasoning: inferring the explanation that, "
    "if true, would best account for the evidence you are given."
)

TABLE: dict[str, dict] = {
    "abd": {
        "system": ABDUCTIVE_CORE + " Here the evidence is a default-exception theory "
        "and an observation it does not yet explain. A good answer is the minimal-cost "
        "set of literals that, added to the theory, derives the observation without "
        "contradicting a stated exception. Respect the theory's own predicate "
        "vocabulary: an explanation outside it does not count.",
        "delivery": "static", "objective": True, "cardinality": None,
    },
    "abductionrules": {
        "system": ABDUCTIVE_CORE + " You are given a small rule base and an observation "
        "that the rules alone do not entail. State the single missing fact which, added "
        "to the rule base, would make the observation derivable. Answer in the same "
        "subject-predicate form the rules use.",
        "delivery": "static", "objective": True, "cardinality": None,
    },
    "aer": {
        "system": ABDUCTIVE_CORE + " You are given a news event and a list of candidate "
        "antecedent events. Select every candidate that is a direct cause of the event -- "
        "an event whose occurrence made the target event happen, not one that merely "
        "preceded or accompanied it. Several candidates can be direct causes.",
        "delivery": "static", "objective": True, "cardinality": "multi",
    },
    "agentrx": {
        "system": ABDUCTIVE_CORE + " You are given the trace of a failed run of an AI "
        "agent and a taxonomy of failure categories. Identify the category that explains "
        "why the run failed -- the fault that produced the observed behaviour, not a "
        "downstream symptom of it.",
        "delivery": "static", "objective": True, "cardinality": "single",
    },
    "aiops2025": {
        "system": ABDUCTIVE_CORE + " You are given the alerts, metrics and service "
        "topology of a microservice incident. Name the entity that is the root cause: the "
        "component whose failure explains the whole alert pattern, not every component "
        "that reported an anomaly downstream of it.",
        "delivery": "static", "objective": True, "cardinality": "single",
    },
    "art": {
        "system": ABDUCTIVE_CORE + " You are given the first and last observation of a "
        "short everyday story. The hypothesis you want is the event in between that makes "
        "the ending unsurprising given the beginning -- the most plausible thing to have "
        "happened, not the most dramatic.",
        "delivery": "static", "objective": True, "cardinality": "single",
        "hypothesis": {
            "modes": ("generation", "selection"),
            "options": {"generation": {"subtask": "generation"},
                        "selection": {"subtask": "selection"}},
            "table": "Generation / Selection (separate tasks)",
            "justification": "",
        },
    },
    "causalab": {
        "system": ABDUCTIVE_CORE + " You are given observational data over a set of "
        "variables. Recover the causal structure that would generate it: state the edges "
        "of the causal graph, orienting each from cause to effect. Do not list an edge "
        "that the data does not distinguish from its reverse.",
        "delivery": "interactive", "objective": True, "cardinality": None,
    },
    "causalopsbench": {
        "system": ABDUCTIVE_CORE + " You are given operational telemetry from a running "
        "system. Identify the faulty component and the kind of fault that together "
        "explain the observed symptoms.",
        "delivery": "static", "objective": True, "cardinality": "single",
    },
    "cloud_opsbench": {
        "system": ABDUCTIVE_CORE + " You are given the symptom of a Kubernetes incident "
        "and the evidence collected from the cluster. Name the root cause: the "
        "misconfiguration, resource limit or failure that accounts for the symptom.",
        "delivery": "interactive", "objective": True, "cardinality": None,
    },
    "commonwhy": {
        "system": ABDUCTIVE_CORE + " You are told that an entity could not do something. "
        "Explain why not, using what is commonly known about that kind of entity. The "
        "explanation should be the property that actually rules the action out.",
        "delivery": "static", "objective": False, "cardinality": None,
    },
    "crosstrace": {
        "system": ABDUCTIVE_CORE + " You are given a field's conventional assumption and "
        "the observation that strains it. Propose the hypothesis that would explain the "
        "observation while contradicting the assumption -- state the mechanism, not a "
        "call for further research.",
        "delivery": "static", "objective": False, "cardinality": None,
    },
    "ddxplus": {
        "system": ABDUCTIVE_CORE + " You are given a patient's answers to a diagnostic "
        "questionnaire, including their age, sex and reported symptoms. State the "
        "diagnosis that explains the whole picture, not merely a condition consistent "
        "with one symptom.",
        "delivery": "interactive", "objective": True, "cardinality": "single",
        "hypothesis": {
            "modes": ("generation", "selection"),
            "options": {"generation": {"subtask": "generation"},
                        "selection": {"subtask": "selection"}},
            "table": "Selection",
            "justification": "DDXPlus releases, per patient, both a single GROUND-TRUTH "
            "PATHOLOGY and a DIFFERENTIAL DIAGNOSIS list; the differential defines a "
            "closed candidate set (selection) while the ground-truth pathology is "
            "recoverable without candidates (generation), so the two are separate tasks "
            "over the same records.",
        },
    },
    "defab": {
        "system": ABDUCTIVE_CORE + " You are given a defeasible theory, a target "
        "conclusion it does not currently support, and candidate additions. Select every "
        "candidate whose addition would make the target derivable while respecting the "
        "theory's defeaters. More than one addition can be required.",
        "delivery": "static", "objective": True, "cardinality": "multi",
    },
    "diagnosisarena": {
        "system": ABDUCTIVE_CORE + " You are given the full work-up of a published "
        "clinical case: presentation, history, examination and investigations. State the "
        "final diagnosis that explains the case as a whole.",
        "delivery": "static", "objective": True, "cardinality": "single",
        "hypothesis": {
            "modes": ("generation", "selection"),
            "options": {"generation": {"subtask": "generation"},
                        "selection": {"subtask": "selection"}},
            "table": "Generation",
            "justification": "Every DiagnosisArena case ships four answer options with "
            "one correct diagnosis alongside the free-text gold diagnosis, so the release "
            "itself defines a closed-set selection task over the same cases as well as "
            "the open-ended one.",
        },
    },
    "ecare": {
        "system": ABDUCTIVE_CORE + " You are given an everyday observation. The cause you "
        "want is the one that would ordinarily bring the observation about -- judge "
        "plausibility by common causal knowledge, not by topical similarity of wording.",
        "delivery": "static", "objective": True, "cardinality": "single",
        "hypothesis": {
            "modes": ("generation", "selection"),
            "options": {"generation": {"subtask": "explanation"},
                        "selection": {"subtask": "cause_selection"}},
            "table": "Generation / Selection (separate tasks)",
            "justification": "",
        },
    },
    "enwn_entailmentbank": {
        "system": ABDUCTIVE_CORE + " You are given the premises of an entailment step and "
        "its conclusion, with one premise missing. State the missing premise: the single "
        "statement that, together with those given, entails the conclusion.",
        "delivery": "static", "objective": True, "cardinality": None,
    },
    "gear": {
        "system": ABDUCTIVE_CORE + " You are shown blicket-detector experiments: which "
        "objects were placed on the detector and whether it activated. Decide whether the "
        "queried object activates the detector. Answer 'undetermined' when the "
        "experiments genuinely do not settle it -- guessing is worse than admitting the "
        "evidence is incomplete.",
        "delivery": "static", "objective": True, "cardinality": "single",
    },
    "house_md": {
        "system": ABDUCTIVE_CORE + " You are given a clinical vignette whose presentation "
        "is unusual. Name the underlying disease that explains the combination of "
        "findings -- rare diseases are expected here, so do not default to the common "
        "condition that explains only part of the picture.",
        "delivery": "static", "objective": True, "cardinality": None,
    },
    "hypoarena": {
        "system": ABDUCTIVE_CORE + " You are given an observed situation. Propose the "
        "hypothesis that explains it: a specific, testable claim about what is going on, "
        "not a restatement of the observation.",
        "delivery": "static", "objective": False, "cardinality": None,
    },
    "hypobench": {
        "system": ABDUCTIVE_CORE + " You are given a labelled sample of examples. State "
        "the hypothesis that explains the labelling: the rule that separates the classes "
        "and would generalise to unseen examples of the same kind.",
        "delivery": "static", "objective": False, "cardinality": None,
    },
    "hypogen": {
        "system": ABDUCTIVE_CORE + " You are given the conventional wisdom in a research "
        "area (the 'bit'). Propose the 'flip': the hypothesis that overturns it and would "
        "explain the results the paper reports.",
        "delivery": "static", "objective": False, "cardinality": None,
    },
    "hypospace": {
        "system": ABDUCTIVE_CORE + " You are given perturbation observations over a set "
        "of variables. Several distinct causal graphs can be compatible with them, and "
        "the task is to cover that space: propose the distinct hypotheses that are all "
        "consistent with the observations, rather than committing to one.",
        "delivery": "static", "objective": True, "cardinality": None,
    },
    "matter_to_mechanism": {
        "system": ABDUCTIVE_CORE + " You are given a materials-science research problem "
        "and its observations. Propose the mechanism that explains them at the level of "
        "structure and process -- what is happening in the material, not what should be "
        "measured next.",
        "delivery": "static", "objective": False, "cardinality": None,
    },
    "med_inquire": {
        "system": ABDUCTIVE_CORE + " You are given a published case report with the "
        "diagnostic work-up withheld. State the diagnosis that best explains the "
        "presenting picture you can see.",
        "delivery": "interactive", "objective": True, "cardinality": "single",
        "hypothesis": {
            "modes": ("generation", "selection"),
            "options": {"generation": {"subtask": "generation"},
                        "selection": {"subtask": "selection"}},
            "table": "Generation",
            "justification": "The Med-Inquire test file is DiagnosisArena's release, in "
            "which each case carries four answer options with one correct diagnosis, so a "
            "closed-set selection task is defined by the data itself.",
        },
    },
    "medcasereasoning": {
        "system": ABDUCTIVE_CORE + " You are given a clinical case report. State the "
        "final diagnosis, and let your reasoning follow the diagnostic evidence in the "
        "case rather than prior probability alone.",
        "delivery": "static", "objective": True, "cardinality": None,
    },
    "medqdx": {
        "system": ABDUCTIVE_CORE + " You are given a clinical vignette that may be "
        "incomplete, and a closed set of candidate diagnoses. Choose the diagnosis best "
        "supported by the information actually present.",
        "delivery": "interactive", "objective": True, "cardinality": "single",
    },
    "medr_bench": {
        "system": ABDUCTIVE_CORE + " You are given a case summary. State the diagnosis "
        "that accounts for the findings; where the case is of a rare disease, the common "
        "look-alike is not the answer.",
        "delivery": "static", "objective": True, "cardinality": None,
    },
    "medups": {
        "system": ABDUCTIVE_CORE + " You are given an uncommon published case, delivered "
        "as a sequence of clinical steps in the order the clinicians received them. State "
        "the diagnosis that explains the case given everything disclosed so far.",
        "delivery": "sequential", "objective": True, "cardinality": "single",
        "hypothesis": {
            "modes": ("generation", "selection"),
            "options": {"generation": {"subtask": "generation"},
                        "selection": {"subtask": "selection"}},
            "table": "Generation",
            "justification": "MedUPS ships a six-option multiple-choice item per case "
            "alongside the free-text diagnosis, so selection among the released "
            "candidates is a task the benchmark defines rather than one imposed here.",
        },
    },
    "moose_chem2": {
        "system": ABDUCTIVE_CORE + " You are given the background and research question "
        "of a chemistry paper. Propose the hypothesis it went on to confirm: a specific "
        "mechanism or relationship, stated so that an experiment could test it.",
        "delivery": "static", "objective": False, "cardinality": None,
        "hypothesis": {
            "modes": ("generation",),
            "options": {},
            "table": "Generation & Selection (separate tasks)",
            "justification": "",
        },
    },
    "musr": {
        "system": ABDUCTIVE_CORE + " You are given a murder mystery. Name the murderer, "
        "and be guided by means, motive and opportunity as the narrative establishes "
        "them -- the culprit is the suspect all three converge on.",
        "delivery": "static", "objective": True, "cardinality": "single",
    },
    "neulr": {
        "system": ABDUCTIVE_CORE + " You are given premises and a conclusion with one "
        "premise missing. Name the missing premise that makes the argument go through.",
        "delivery": "static", "objective": True, "cardinality": None,
    },
    "physgym": {
        "system": ABDUCTIVE_CORE + " You are given a physical setup and the quantities it "
        "involves. State the law relating the target quantity to the others, as an "
        "equation in the given symbols. A relation that fits the described behaviour "
        "matters more than one that looks like a familiar formula.",
        "delivery": "interactive", "objective": True, "cardinality": None,
    },
    "proof_writer": {
        "system": ABDUCTIVE_CORE + " You are given a rule base and a statement it can "
        "almost prove. State the single fact that is missing from the theory and would "
        "complete the proof.",
        "delivery": "static", "objective": True, "cardinality": None,
    },
    "scir": {
        "system": ABDUCTIVE_CORE + " You are given observational data and candidate "
        "causal edges. Choose the hidden edge whose presence explains the pattern in the "
        "data.",
        "delivery": "static", "objective": True, "cardinality": "single",
    },
    "synpat": {
        "system": ABDUCTIVE_CORE + " You are given an axiom system of physical equations "
        "and data that contradicts it. Identify the equation that is wrong and state its "
        "corrected form, in the symbols the system uses.",
        "delivery": "static", "objective": True, "cardinality": None,
    },
    "true_detective": {
        "system": ABDUCTIVE_CORE + " You are given a detective puzzle in full. Work out "
        "which candidate explanation the evidence actually supports; these puzzles are "
        "designed so that the obvious reading is usually wrong.",
        "delivery": "static", "objective": True, "cardinality": "single",
    },
    "uncommonsense": {
        "system": ABDUCTIVE_CORE + " You are given a situation with an outcome that is "
        "surprising given the context. Explain how it could plausibly have come about -- "
        "the explanation has to make the uncommon outcome likely, not merely possible.",
        "delivery": "static", "objective": False, "cardinality": None,
    },
    "vivabench": {
        "system": ABDUCTIVE_CORE + " You are given a clinical viva case. State the "
        "diagnosis that explains the presentation.",
        "delivery": "interactive", "objective": True, "cardinality": "single",
        "hypothesis": {
            "modes": ("generation", "selection"),
            "options": {"generation": {"subtask": "generation"},
                        "selection": {"subtask": "selection"}},
            "table": "Generation",
            "justification": "Each VivaBench case ships an explicit differential-diagnosis "
            "list next to its final diagnosis, so choosing among the case's own "
            "differentials is a task the release defines.",
        },
    },
    "xcopa": {
        "system": ABDUCTIVE_CORE + " You are given a premise in one of several languages "
        "and two candidate causes. Choose the alternative that is the more plausible "
        "cause of the premise, judging by everyday causal knowledge in that language's "
        "context.",
        "delivery": "static", "objective": True, "cardinality": "single",
    },
    "open_problems": {
        "system": ABDUCTIVE_CORE + " You are given a research problem that was open as of "
        "your knowledge cutoff. Judge it on the evidence available at the time and say "
        "how confident you are; a calibrated 'probably not' is worth more than a "
        "confident guess.",
        "delivery": "static", "objective": True, "cardinality": None,
    },
}


def _wrap(text: str, indent: str, width: int = 96) -> str:
    """Render a long string as adjacent quoted literals, one per line."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(indent) + len(candidate) + 4 > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    out = []
    for index, line in enumerate(lines):
        suffix = " " if index < len(lines) - 1 else ""
        out.append(f'{indent}"{line}{suffix}"')
    return "\n".join(out)


def block_for(name: str, spec: dict) -> str:
    lines = ["", "    system_prompt = (", _wrap(spec["system"], "        "), "    )"]
    lines.append(f'    data_delivery_mode = "{spec["delivery"]}"')
    lines.append(f"    objective_metrics = {spec['objective']}")
    if spec["cardinality"] is None:
        lines.append("    selection_cardinality = None")
    else:
        lines.append(f'    selection_cardinality = "{spec["cardinality"]}"')
    hypothesis = spec.get("hypothesis")
    if hypothesis:
        modes = ", ".join(f'"{m}"' for m in hypothesis["modes"])
        lines.append(f"    hypothesis_modes = ({modes},)")
        if hypothesis["options"]:
            lines.append("    hypothesis_mode_options = {")
            for mode, options in hypothesis["options"].items():
                lines.append(f'        "{mode}": {options!r},')
            lines.append("    }")
        lines.append(f'    table_hypothesis_mode = "{hypothesis["table"]}"')
        if hypothesis["justification"]:
            lines.append("    hypothesis_mode_justification = (")
            lines.append(_wrap(hypothesis["justification"], "        "))
            lines.append("    )")
    return "\n".join(lines) + "\n"


def main() -> int:
    changed = 0
    for name, spec in sorted(TABLE.items()):
        path = ADAPTERS / f"{name}.py"
        if not path.exists():
            print(f"!! no adapter module for {name}", file=sys.stderr)
            continue
        source = path.read_text(encoding="utf-8")
        if "    data_delivery_mode = " in source:
            print(f"== {name}: already declared, skipping")
            continue
        match = re.search(r"\n    adapter_version = \"[^\"]+\"\n", source)
        if not match:
            print(f"!! {name}: no adapter_version anchor", file=sys.stderr)
            continue
        insert_at = match.end()
        source = source[:insert_at] + block_for(name, spec) + source[insert_at:]
        path.write_text(source, encoding="utf-8")
        changed += 1
        print(f"-> {name}")
    print(f"declared {changed} adapter(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
