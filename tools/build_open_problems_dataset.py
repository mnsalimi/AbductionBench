"""Build the Open Problems 2024+ dataset from its two source documents.

Sources (both checked in under ``assets/open_problems_2024/sources/``):

* ``pre2024_cutoff_benchmark.txt`` -- the primary source: 12 task cards, each with
  a neutrally-worded direction prompt (A), resolution prompt (B), strategy prompt
  (C) and a golden-solution block. Exported from the compiled Google Doc.
* ``undermind_report.docx`` -- an independent re-check of the same survey. It
  contributes three problems the primary source does not cover, plus corroborating
  detail on three it shares.

The parsing is mechanical; everything that required judgement is in the
``NORMALIZATION`` table below and is attributed there, so a reader can audit
exactly which fields came from a source document and which were authored:

* ``gold_label`` / ``label_set`` / ``partial_labels`` -- reading each source
  prompt's own answer options against its golden answer. Necessary because the
  options differ per task ("YES/NO", "TRUE/FALSE/NEEDS-MODIFICATION",
  "DECIDABLE/UNDECIDABLE-FOR-ALL-K/DEPENDS-ON-K", ...), so a single label
  vocabulary would not work.
* ``key_ingredients`` -- checkable terms drawn *verbatim* from each card's "Key
  method" text. These are what a strategy answer is graded against, so they are
  quoted rather than paraphrased.
* ``posed`` -- the problem's first formulation, from the card's "Origin" field
  where present, otherwise from the standard reference named in the source.

Run:  python tools/build_open_problems_dataset.py
Out:  assets/open_problems_2024/problems.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "assets" / "open_problems_2024" / "sources"
OUTPUT = ROOT / "assets" / "open_problems_2024" / "problems.json"

# --------------------------------------------------------------------------- #
# Authored normalization, per task. Attribution is explicit in each entry.
# --------------------------------------------------------------------------- #

NORMALIZATION: dict[str, dict[str, Any]] = {
    "TASK-01": {
        "slug": "kakeya_3d",
        "label_set": ["YES", "NO"],
        "gold_label": "YES",
        "posed": {"year": 1917, "by": "Sōichi Kakeya", "note": "needle-rotation problem; modern higher-dimensional dimension conjecture from Besicovitch's 1920s constructions"},
        "solved_date": "2025-02-24",
        "solvers": ["Hong Wang", "Joshua Zahl"],
        "key_ingredients": [
            "volume estimate for unions of tubes/convex sets",
            "multi-scale induction",
            "sticky and non-sticky configurations",
            "grains/planebrush structure theory",
        ],
    },
    "TASK-02": {
        "slug": "geometric_langlands",
        "label_set": ["TRUE", "FALSE", "NEEDS-MODIFICATION"],
        "gold_label": "TRUE",
        "posed": {"year": 1980, "by": "Beilinson and Drinfeld (categorical form due to Arinkin-Gaitsgory)", "note": "grew out of the Langlands program of the late 1960s; geometric formulations for function fields appeared in the 1980s"},
        "solved_date": "2024-05-06",
        "solved_date_note": "Paper I of five; the proof was completed with Paper V on 2024-09-15",
        "solvers": ["Dennis Gaitsgory", "Sam Raskin", "D. Arinkin", "D. Beraldo", "J. Campbell", "L. Chen", "J. Faergeman", "K. Lin", "N. Rozenblyum"],
        "key_ingredients": [
            "construction of the Langlands functor from the automorphic to the spectral side",
            "equivalence of de Rham/Betti and restricted/unrestricted forms",
            "existence and uniqueness of the Hecke eigensheaf",
            "geometry of the stack of local systems",
        ],
    },
    "TASK-03": {
        "slug": "bourgain_slicing",
        "label_set": ["YES", "NO"],
        "gold_label": "YES",
        "posed": {"year": 1986, "by": "Jean Bourgain", "note": "Amer. J. Math. 108(6) (1986), 1467-1476"},
        "solved_date": "2024-12-19",
        "solved_date_note": "Guan's enabling bound 2024-12-12; Klartag-Lehec final step 2024-12-19",
        "solvers": ["Bo'az Klartag", "Joseph Lehec", "Qingyang Guan"],
        "key_ingredients": [
            "Milman's theory of M-ellipsoids",
            "stochastic localization",
            "Guan's bound",
            "Eldan-Mikulincer stability estimates for the Shannon-Stam inequality",
        ],
    },
    "TASK-04": {
        "slug": "hilbert_tenth_rings_of_integers",
        "label_set": ["DECIDABLE", "UNDECIDABLE-FOR-ALL-K", "DEPENDS-ON-K"],
        "gold_label": "UNDECIDABLE-FOR-ALL-K",
        "posed": {"year": 1975, "by": "Denef (Denef-Lipshitz 1978)", "note": "Hilbert's tenth problem itself posed 1900; the rings-of-integers conjecture is Denef's"},
        "solved_date": "2024-12-02",
        "solved_date_note": "Koymans-Pagano 2024-12-02; independent Alpoge-Bhargava-Ho-Shnidman 2025-01-30",
        "solvers": ["Peter Koymans", "Carlo Pagano", "Levent Alpoge", "Manjul Bhargava", "Wei Ho", "Ari Shnidman"],
        "key_ingredients": [
            "elliptic curves with full rational 2-torsion",
            "additive combinatorics",
            "2-descent",
            "no rank growth in chosen quadratic extensions",
        ],
    },
    "TASK-05": {
        "slug": "boltzmann_long_time_derivation",
        "label_set": ["YES-ACHIEVABLE-WITH-CURRENT-IDEAS", "YES-BUT-REQUIRES-NEW-IDEAS", "NO-OBSTRUCTION-EXISTS"],
        "gold_label": "YES-BUT-REQUIRES-NEW-IDEAS",
        "partial_labels": ["YES-ACHIEVABLE-WITH-CURRENT-IDEAS"],
        "gold_label_note": "The derivation was achieved, so any YES answer has the direction right; full credit goes to YES-BUT-REQUIRES-NEW-IDEAS because the proof introduced four new ingredients (time-layering, a cumulant ansatz, the associated integral analysis, and a purpose-built combinatorial algorithm).",
        "posed": {"year": 1975, "by": "Oscar Lanford (short-time theorem); the long-time problem implicit in Hilbert's sixth problem", "note": "Lanford 1975 covered only short times"},
        "solved_date": "2024-08-14",
        "solvers": ["Yu Deng", "Zaher Hani", "Xiao Ma"],
        "key_ingredients": [
            "time-layering argument",
            "cumulant ansatz that memorizes the full collision history",
            "purpose-built combinatorial algorithm",
            "Boltzmann-Grad limit for hard spheres",
        ],
    },
    "TASK-06": {
        "slug": "bunkbed",
        "label_set": ["ALWAYS-TRUE", "NOT-ALWAYS-TRUE"],
        "gold_label": "NOT-ALWAYS-TRUE",
        "posed": {"year": 1985, "by": "Pieter Kasteleyn", "note": "recorded in van den Berg-Kahn (2001), Remark 5"},
        "solved_date": "2024-10-01",
        "solvers": ["Nikita Gladkov", "Igor Pak", "Aleksandr Zimin", "Lawrence Hollom"],
        "key_ingredients": [
            "Hollom's 3-uniform hypergraph counterexample",
            "engineered gadget graph replacing each hyperedge",
            "explicit planar counterexample",
            "computer-free proof",
        ],
    },
    "TASK-07": {
        "slug": "aldous_lyons",
        "label_set": ["TRUE", "FALSE"],
        "gold_label": "FALSE",
        "posed": {"year": 2007, "by": "David Aldous and Russell Lyons", "note": "Electron. J. Probab. 12 (2007), 1454-1508"},
        "solved_date": "2024-07-31",
        "solved_date_note": "Part I 2024-07-31/08-01; Part II 2024-12-31",
        "solvers": ["Lewis Bowen", "Michael Chapman", "Alexander Lubotzky", "Thomas Vidick"],
        "key_ingredients": [
            "subgroup tests as an analogue of nonlocal games",
            "novel interactive proof system",
            "reduction of the halting problem to that proof system",
            "MIP* = RE style argument",
        ],
    },
    "TASK-08": {
        "slug": "kervaire_126",
        "label_set": ["YES", "NO"],
        "gold_label": "YES",
        "posed": {"year": 1960, "by": "Michel Kervaire (Kervaire-Milnor)", "note": "Adams-spectral-sequence reformulation by Browder (1969)"},
        "solved_date": "2024-12-14",
        "solvers": ["Weinan Lin", "Guozhen Wang", "Zhouli Xu"],
        "key_ingredients": [
            "extension spectral sequence for a map of spectra",
            "HF_2-synthetic context",
            "jump-of-filtration phenomena",
            "large-scale machine computation",
        ],
    },
    "TASK-09": {
        "slug": "thorium229_clock",
        "label_set": ["YES", "NO"],
        "gold_label": "YES",
        "gold_label_note": "Forecasting item: the source's Prompt A asks for a probability that direct laser excitation is achieved within 24 months of the model's cutoff, plus the groups most likely to do it first. It was achieved (2024) and independently replicated by three groups, so the gold direction is YES.",
        "posed": {"year": 2003, "by": "Ekkehard Peik and Christian Tamm", "note": "Europhys. Lett. 61, 181 (2003); isomer inferred earlier by Kroger and Reich"},
        "solved_date": "2024-04-29",
        "solved_date_note": "PTB Braunschweig with TU Wien, Phys. Rev. Lett. 132, 182501 (April 2024); replicated by two further groups",
        "solvers": ["J. Tiedau", "E. Peik", "T. Schumm"],
        "expected_groups": ["PTB", "Braunschweig", "Peik", "Schumm", "TU Wien", "JILA", "Ye"],
        "numeric_target": {"value": 8.35574, "unit": "eV", "tolerance_rel": 0.001,
                            "secondary": {"value": 148.3821, "unit": "nm"}},
        "key_ingredients": [],
        "no_strategy_prompt": True,
        "empirical": True,
    },
    "TASK-10": {
        "slug": "busy_beaver_5",
        "label_set": [],
        "gold_label": "47176870",
        "answer_kind": "numeric",
        "numeric_target": {"value": 47176870, "unit": "steps", "tolerance_rel": 0.0},
        "proof_status_gold": "PROVED",
        "gold_label_note": "The value 47,176,870 was already conjectured before 2024 (Marxen-Buntrock lower bound), so a pre-2024 model can legitimately state it; what changed in July 2024 is that it became PROVED. The numeric answer and the proof-status claim are therefore scored separately.",
        "posed": {"year": 1962, "by": "Tibor Radó", "note": "Bell System Technical Journal 41 (1962), 877-884"},
        "solved_date": "2024-07-02",
        "solvers": ["Busy Beaver Challenge collaboration (coordinated by Tristan Sterin)"],
        "key_ingredients": [
            "enumerate all five-state machines in Tree Normal Form",
            "pipeline of deciders/verifiers",
            "hand-written Coq non-halting proofs for the Sporadic Machines",
            "formal verification in Coq/Rocq",
        ],
    },
    "TASK-11": {
        "slug": "mizohata_takeuchi",
        "label_set": ["TRUE", "FALSE"],
        "gold_label": "FALSE",
        "posed": {"year": 1974, "by": "Jiro Takeuchi (1974, 1980) and Sigeru Mizohata (1985)", "note": "from work on well-posedness for perturbed linear Schrödinger equations"},
        "solved_date": "2025-02-10",
        "solvers": ["Hannah Mira Cairo", "Ruixiang Zhang"],
        "key_ingredients": [
            "L^p estimates for the X-ray transform of positive measures",
            "log R-loss counterexample construction",
            "works for every C^2 hypersurface not lying in a hyperplane",
        ],
    },
    "TASK-12": {
        "slug": "erdos_unit_distance",
        "label_set": ["TRUE", "FALSE"],
        "gold_label": "FALSE",
        "posed": {"year": 1946, "by": "Paul Erdős", "note": "Amer. Math. Monthly 53 (1946), 248-250; Erdős offered $500 for a matching upper bound"},
        "solved_date": "2026-05-20",
        "solvers": ["Noga Alon", "Thomas Bloom", "and co-authors (original counterexample produced by an unreleased internal OpenAI model)"],
        "key_ingredients": [
            "geometry-of-numbers window lemma",
            "pigeonhole/class-group lemma",
            "algebraic numbers of modulus 1 in CM fields",
            "Golod-Shafarevich class field towers",
        ],
    },
}

# --------------------------------------------------------------------------- #
# The three problems only the Undermind report covers.
#
# Its own prompts are identification-style ("Identify the open problem described
# below"), which is a recall task rather than an inference one, so the direction
# and strategy prompts below are authored here, following the primary source's
# neutral-framing rules (§2.4): no "prove that", no year, no solver, no hint
# that the problem was recently resolved.
# --------------------------------------------------------------------------- #

UNDERMIND_ONLY: list[dict[str, Any]] = [
    {
        "id": "ODP-U1",
        "slug": "maximally_entangled_mixed_states",
        "title": "UNIVERSAL MAXIMALLY ENTANGLED MIXED STATE AT FIXED SPECTRUM",
        "field": "quantum information theory",
        "difficulty_tier": "high",
        "status_at_cutoff_2023": (
            "open; a canonical family was known to maximize several important measures, "
            "including entanglement of formation, relative entropy of entanglement and "
            "negativity, which suggested the same state might maximize every monotone"
        ),
        "prompt_A": (
            "Consider two-qubit mixed quantum states with a fixed spectrum. A canonical family "
            "is known to maximize several standard entanglement measures simultaneously, "
            "including entanglement of formation, relative entropy of entanglement and "
            "negativity. Question: for every fixed spectrum, does there exist a single state "
            "that maximizes every entanglement monotone at once, i.e. a universal maximally "
            "entangled mixed state? Answer YES or NO, then give a probability between 0 and 1 "
            "that your answer is correct. Do not hedge; commit to one answer."
        ),
        "prompt_B": (
            "Resolve the following question with full rigor, either by proof or by "
            "counterexample. Question: for two-qubit mixed states of fixed spectrum, does a "
            "single state maximize every entanglement monotone simultaneously? State clearly "
            "at the outset which way you are arguing. If you cannot resolve it, say so "
            "explicitly rather than presenting an incomplete argument as complete."
        ),
        "prompt_C": (
            "Suppose you had to settle whether a universal maximally entangled mixed state "
            "exists for every fixed two-qubit spectrum. Describe the technical strategy you "
            "would pursue, naming the specific state families, operation classes, and "
            "intermediate statements you would try to establish. Be concrete."
        ),
        "label_set": ["YES", "NO"],
        "gold_label": "NO",
        "posed": {"year": 2001, "by": "Frank Verstraete, Koenraad Audenaert and Bart De Moor", "note": "maintained as Problem 3, 'Maximally entangled mixed states', on the IQOQI Open Quantum Problems list"},
        "solved_date": "2024-02-08",
        "solved_date_note": "preprint 2024-02-08; Physical Review Letters 2024-07-30",
        "solvers": ["Julio I. de Vicente"],
        "golden_answer": (
            "NO. No universal maximally entangled mixed state exists. For particular rank-2 "
            "two-qubit spectra, no state can be transformed into all other isospectral states "
            "even under the larger class of non-entangling operations, so no single state can "
            "maximize every entanglement monotone. One valid counterexample settles the "
            "universal question."
        ),
        "key_ingredients": [
            "rank-2 two-qubit spectra",
            "non-entangling operations",
            "state transformation between isospectral states",
            "counterexample to the universal claim",
        ],
        "resolution_type": "complete unconditional refutation of the universal statement",
        "still_open": "classification for every spectrum, and higher-dimensional variants",
        "sources": {"primary": "de Vicente, Phys. Rev. Lett. (2024)", "problem_source": "IQOQI Open Quantum Problems", "independent": ["APS Physics"]},
        "golden_label_confidence": "HIGH",
    },
    {
        "id": "ODP-U2",
        "slug": "minor_free_linear_separators",
        "title": "LINEAR-TIME BALANCED SEPARATORS IN MINOR-FREE GRAPHS",
        "field": "graph algorithms / structural graph theory",
        "difficulty_tier": "high",
        "status_at_cutoff_2023": (
            "open; Alon-Seymour-Thomas (1990) gave the O(sqrt(n)) separator-size bound for "
            "minor-free graphs but only an O(n^{3/2})-time algorithm. Known algorithms had "
            "either a larger separator, superlinear running time, or both"
        ),
        "prompt_A": (
            "For every fixed graph H, an H-minor-free graph on n vertices is known to admit a "
            "balanced vertex separator of size O(sqrt(n)). Question: can such a separator be "
            "computed in time linear in the size of the graph, that is O(n + m), for every "
            "fixed H? Answer YES or NO, then give a probability between 0 and 1 that your "
            "answer is correct. Do not hedge; commit to one answer."
        ),
        "prompt_B": (
            "Resolve the following question with full rigor. Question: for every fixed graph H, "
            "can a balanced separator of size O(sqrt(n)) in an H-minor-free graph be computed "
            "in linear time? Give an algorithm with a correctness and running-time proof, or "
            "prove that no such algorithm exists. If you cannot resolve it, say so explicitly."
        ),
        "prompt_C": (
            "Suppose you had to obtain a linear-time algorithm computing balanced O(sqrt(n)) "
            "separators in H-minor-free graphs. Describe the technical strategy you would "
            "pursue, naming the specific search procedures, weighting schemes, structural "
            "theorems and intermediate statements you would try to establish. Be concrete."
        ),
        "label_set": ["YES", "NO"],
        "gold_label": "YES",
        "posed": {"year": 1990, "by": "Noga Alon, Paul Seymour and Robin Thomas", "note": "descends from the Lipton-Tarjan planar separator theorem of the late 1970s; the open gap was the running time"},
        "solved_date": "2025-12-01",
        "solved_date_note": "preprint 2025-12-01; STOC 2026, online 2026-06-09",
        "solvers": ["Édouard Bonnet", "Tuukka Korhonen", "Hung Le", "Jason Li", "Tomáš Masařík"],
        "golden_answer": (
            "YES. A linear-time algorithm finds a balanced separator of size O(sqrt(n)) in every "
            "fixed-minor-free graph class, preserving the optimal asymptotic separator size and "
            "achieving the previously missing linear running time. The method uses a "
            "vertex-weighted breadth-first search with a weighting scheme connected to "
            "clique-minor structure."
        ),
        "key_ingredients": [
            "vertex-weighted breadth-first search",
            "weighting scheme connected to clique-minor structure",
            "preserving the O(sqrt(n)) separator size",
        ],
        "resolution_type": "complete; the excluded minor H is fixed and hidden constants depend on H",
        "still_open": "a linear-time separator theorem for arbitrary graphs",
        "sources": {"primary": "Bonnet, Korhonen, Le, Li, Masařík (STOC 2026)", "independent": ["STOC 2026 proceedings"]},
        "golden_label_confidence": "HIGH",
    },
    {
        "id": "ODP-U3",
        "slug": "k_edge_connected_components_linear_time",
        "title": "LINEAR-TIME k-EDGE-CONNECTED COMPONENTS FOR EVERY FIXED k",
        "field": "graph algorithms",
        "difficulty_tier": "high",
        "status_at_cutoff_2023": (
            "open; linear-time algorithms were known for special low-connectivity cases, but "
            "the general fixed-k problem was not solved. Existing methods were superlinear, "
            "almost linear, or had unfavourable dependence on k"
        ),
        "prompt_A": (
            "Given an undirected graph and a fixed positive integer k, consider computing the "
            "decomposition of the graph into its k-edge-connected components. Question: for "
            "every fixed k, can this decomposition be computed in time linear in the number of "
            "vertices and edges? Answer YES or NO, then give a probability between 0 and 1 that "
            "your answer is correct. Do not hedge; commit to one answer."
        ),
        "prompt_B": (
            "Resolve the following question with full rigor. Question: for every fixed positive "
            "integer k, can the k-edge-connected component decomposition of an undirected graph "
            "be computed in linear time? Give an algorithm with correctness and running-time "
            "proofs, or prove that no such algorithm exists. If you cannot resolve it, say so "
            "explicitly."
        ),
        "prompt_C": (
            "Suppose you had to compute k-edge-connected components in linear time for every "
            "fixed k. Describe the technical strategy you would pursue, naming the specific "
            "decompositions, data structures and intermediate statements you would try to "
            "establish. Be concrete."
        ),
        "label_set": ["YES", "NO"],
        "gold_label": "YES",
        "posed": {"year": 1990, "by": "the graph-algorithms literature on small-cut decompositions", "note": "linear-time results existed for small k; the fixed-k general case remained the open question"},
        "solved_date": "2024-11-04",
        "solved_date_note": "preprint 2024-11-04; STOC 2025, presented 2025-06-23/27",
        "solvers": ["Tuukka Korhonen"],
        "golden_answer": (
            "YES. For every fixed k the k-edge-connected components can be computed in time "
            "linear in the number of edges. The same work gives algorithms for k-lean tree "
            "decompositions, fixed-k vertex connectivity and certain k-Gomory-Hu trees."
        ),
        "key_ingredients": [
            "k-lean tree decompositions",
            "fixed-k vertex connectivity",
            "k-Gomory-Hu trees",
            "linear dependence on graph size for every fixed k",
        ],
        "resolution_type": "complete for fixed k; the dependence on k is large, and directed or non-constant-k variants remain open",
        "still_open": "directed variants and non-constant k",
        "sources": {"primary": "Korhonen (STOC 2025)", "independent": ["University of Copenhagen research record", "ICALP 2026 follow-up"]},
        "golden_label_confidence": "HIGH",
    },
]

HELD_OUT: list[dict[str, Any]] = [
    {
        "id": "ODP-H1",
        "slug": "moving_sofa",
        "title": "MOVING SOFA PROBLEM",
        "field": "discrete/optimal geometry",
        "difficulty_tier": "high",
        "status_at_cutoff_2023": "open; Gerver's 1992 shape (area ~2.2195) was the conjectured optimum, with upper bounds proved but no matching optimality proof",
        "prompt_A": (
            "Consider the largest area of a shape that can be maneuvered through a unit-width "
            "L-shaped corridor. Gerver's 1992 shape has area approximately 2.2195. Question: is "
            "Gerver's shape optimal? Answer YES or NO, then give a probability between 0 and 1 "
            "that your answer is correct. Do not hedge; commit to one answer."
        ),
        "prompt_C": (
            "Suppose you had to determine whether Gerver's sofa is the area-maximizing shape "
            "for the unit-width L-shaped corridor. Describe the technical strategy you would "
            "pursue, naming the specific variational arguments, structural characterizations "
            "and intermediate statements you would try to establish. Be concrete."
        ),
        "label_set": ["YES", "NO"],
        "gold_label": "YES",
        "posed": {"year": 1966, "by": "Leo Moser", "note": "Gerver's candidate shape 1992"},
        "solved_date": "2024-11-29",
        "solvers": ["Jineon Baek"],
        "golden_answer": (
            "YES (claimed). Gerver's sofa is optimal, per a 119-page preprint that is notable "
            "for being computer-free, which had not been expected."
        ),
        "key_ingredients": ["computer-free variational argument", "optimality of Gerver's shape"],
        "resolution_type": "claimed complete proof; peer review not complete",
        "sources": {"primary": "https://arxiv.org/abs/2411.19826"},
        "golden_label_confidence": "MEDIUM",
        "held_out": True,
        "held_out_reason": (
            "Meets the significance bar but not the community-acceptance bar: as of March 2026 "
            "reported under review at Annals of Mathematics, peer review not complete. The "
            "primary source recommends keeping such items in a separate frontier split."
        ),
    },
]


# --------------------------------------------------------------------------- #
# Source parsing
# --------------------------------------------------------------------------- #


def _clean(text: str) -> str:
    for escaped, plain in [
        ("\\>", ">"), ("\\<", "<"), ("\\_", "_"), ("\\*", "*"), ("\\[", "["),
        ("\\]", "]"), ("\\=", "="), ("\\-", "-"), ("\\.", "."), ("\\~", "~"),
        ("\\`", "`"), ("\\#", "#"),
    ]:
        text = text.replace(escaped, plain)
    return re.sub(r"\s+", " ", text).strip()


def parse_primary(path: Path) -> list[dict[str, Any]]:
    """Parse the 12 task cards out of the primary benchmark document."""
    lines = path.read_text(encoding="utf-8").split("\n")
    starts = [i for i, line in enumerate(lines) if re.match(r"^TASK-\d+\s", line)]
    end = next(i for i, line in enumerate(lines) if line.startswith("4\\. OPTIONAL"))
    starts.append(end)

    meta_keys = ("Field", "Difficulty tier", "Status at a 2023 cutoff")
    cards: list[dict[str, Any]] = []
    for begin, finish in zip(starts, starts[1:], strict=False):
        body = [line for line in lines[begin:finish] if not re.match(r"^\\?-{10,}", line)]
        task_id, title = re.match(r"^(TASK-\d+)\s+(.*)$", body[0]).groups()

        marks: list[tuple[int, str, str]] = []
        for index, line in enumerate(body):
            prompt = re.match(r"^PROMPT ([A-D])\b(.*)$", line.strip())
            if prompt:
                marks.append((index, prompt.group(1), prompt.group(2).strip()))
            elif line.strip() == "GOLDEN SOLUTION":
                marks.append((index, "GOLD", ""))

        card: dict[str, Any] = {"id": task_id, "title": _clean(title)}
        last: str | None = None
        for line in body[1 : marks[0][0]]:
            header = re.match(rf'^({"|".join(meta_keys)}):\s*(.*)$', line)
            if header:
                last = header.group(1)
                card[last] = _clean(header.group(2))
            elif line.strip() and last in card:
                card[last] = _clean(f"{card[last]} {line}")

        for (index, kind, note), (nxt, _, _) in zip(
                marks, marks[1:] + [(len(body), "", "")], strict=True):
            if kind == "GOLD":
                gold: dict[str, str] = {}
                current: str | None = None
                for line in body[index + 1 : nxt]:
                    entry = re.match(r"^\s{2}([A-Z][A-Za-z0-9 ()\-/]+):\s*(.*)$", line)
                    if entry:
                        current = entry.group(1).strip()
                        gold[current] = entry.group(2).strip()
                    elif current:
                        gold[current] += " " + line.strip()
                card["gold"] = {key: _clean(value) for key, value in gold.items()}
            elif "skip" in note.lower():
                continue
            else:
                card[f"prompt_{kind}"] = _clean(" ".join(body[index + 1 : nxt])).strip('"').strip()
        cards.append(card)
    return cards


# The independent report numbers its own problems (ODP-nn) and never mentions the
# primary source's TASK-nn ids, so the two lists are linked by what the report
# says each problem IS.  Keeping that link as a pattern means the corroboration
# claim in the output is checked against the docx on every build: if the report
# changes and a match disappears, the build fails instead of quietly asserting
# an independent verification that no longer exists.
UNDERMIND_MATCHES = {
    "TASK-01": r"kakeya|besicovitch",
    "TASK-02": r"geometric langlands",
    "TASK-08": r"kervaire",
}


def parse_undermind(path: Path) -> dict[str, str]:
    """Pull the corroborating gold text out of the independent report."""
    try:
        from docx import Document
    except ImportError:
        return {}
    paragraphs = [p.text.strip() for p in Document(str(path)).paragraphs if p.text.strip()]
    corroboration: dict[str, str] = {}
    current: str | None = None
    for text in paragraphs:
        task = re.match(r"^Task (ODP-\d+)$", text)
        if task:
            current = task.group(1)
            corroboration[current] = ""
        elif current and text.startswith("Problem identity:"):
            corroboration[current] = text.split(":", 1)[1].strip()
    return corroboration


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


def build() -> dict[str, Any]:
    cards = parse_primary(SOURCES / "pre2024_cutoff_benchmark.txt")
    corroboration = parse_undermind(SOURCES / "undermind_report.docx")

    problems: list[dict[str, Any]] = []
    for card in cards:
        norm = NORMALIZATION[card["id"]]
        gold = card["gold"]
        def pick(*keys: str, gold: dict[str, str] = gold) -> str:
            return next((gold[k] for k in keys if k in gold), "")


        problem: dict[str, Any] = {
            "id": card["id"],
            "slug": norm["slug"],
            "title": card["title"].title(),
            "field": card.get("Field", ""),
            "difficulty_tier": card.get("Difficulty tier", ""),
            "status_at_cutoff_2023": card.get("Status at a 2023 cutoff", ""),
            "problem_posed": norm["posed"],
            "prompts": {
                key.lower(): card[f"prompt_{key}"]
                for key in ("A", "B", "C", "D")
                if f"prompt_{key}" in card
            },
            "solved_date": norm["solved_date"],
            "solver": norm["solvers"],
            "golden_solution": {
                "answer": pick("Answer", "Theorem", "Final classification"),
                "label_set": norm.get("label_set", []),
                "gold_label": norm["gold_label"],
                "partial_labels": norm.get("partial_labels", []),
                "key_method": pick("Key method", "Key methods"),
                "key_ingredients": norm["key_ingredients"],
                "resolution_type": pick("Resolution type"),
                "still_open": pick("Still open"),
                "scope": pick("Scope"),
                "consequence": pick("Consequence"),
            },
            "sources": {
                "primary": pick("Primary link", "Primary links"),
                "origin": pick("Origin", "Origin of the goal", "Origin source"),
                "independent_verification": pick("Independent verification"),
                "acceptance": pick("Acceptance"),
                "expert_quote": pick("Expert quote (pre-award)"),
            },
            "golden_label_confidence": pick("Confidence in golden label"),
            "evaluator_notes": pick("EVALUATOR NOTE", "NOTE FOR EVALUATORS", "CAUTION FOR EVALUATORS", "NOTE", "CAVEAT"),
            "provenance": ["pre2024_cutoff_benchmark"],
        }
        # Three of the source's strategy prompts presuppose that a counterexample
        # is what is wanted, which tells the reader the conjecture is false. That
        # does not contaminate the direction item (each is an independent call
        # with no shared context), but it does make those strategy items a
        # different, easier task: "given the direction, name the route".
        strategy_prompt = (problem["prompts"] or {}).get("c", "")
        if strategy_prompt and re.search(
            r"were false|counterexample|disprov|refut", strategy_prompt, re.I
        ):
            problem["strategy_prompt_conditions_on_direction"] = True

        for optional in ("solved_date_note", "gold_label_note", "answer_kind",
                         "numeric_target", "proof_status_gold", "expected_groups",
                         "no_strategy_prompt", "empirical"):
            if optional in norm:
                problem[optional] = norm[optional]
        pattern = UNDERMIND_MATCHES.get(card["id"])
        if pattern and corroboration:
            match = next((odp for odp, identity in corroboration.items()
                          if re.search(pattern, identity, re.I)), None)
            if match is None:
                raise SystemExit(
                    f"{card['id']} is recorded as independently verified, but the "
                    f"independent report no longer contains a problem matching "
                    f"{pattern!r}. Re-check the source before rebuilding."
                )
            problem["provenance"].append("undermind_report")
            problem["sources"]["independent_verification_task"] = match
            problem["sources"]["independent_verification_identity"] = corroboration[match]
        problems.append(problem)

    for extra in UNDERMIND_ONLY + HELD_OUT:
        entry = dict(extra)
        entry["problem_posed"] = entry.pop("posed")
        entry["solver"] = entry.pop("solvers")
        prompts = {}
        for key in ("A", "B", "C"):
            if f"prompt_{key}" in entry:
                prompts[key.lower()] = entry.pop(f"prompt_{key}")
        entry["prompts"] = prompts
        entry["golden_solution"] = {
            "answer": entry.pop("golden_answer"),
            "label_set": entry.pop("label_set"),
            "gold_label": entry.pop("gold_label"),
            "partial_labels": [],
            "key_method": "; ".join(entry.get("key_ingredients", [])),
            "key_ingredients": entry.pop("key_ingredients"),
            "resolution_type": entry.pop("resolution_type", ""),
            "still_open": entry.pop("still_open", ""),
        }
        entry["provenance"] = ["undermind_report"] if entry["id"].startswith("ODP-U") else [
            "pre2024_cutoff_benchmark:held_out"
        ]
        entry["prompt_authorship"] = (
            "direction/resolution/strategy prompts authored for this dataset following the "
            "primary source's neutral-framing rules; the source supplied only an "
            "identification-style prompt"
        )
        problems.append(entry)

    return {
        "dataset": "open_problems_2024",
        "version": "1.0",
        "title": "Open problems posed before 2024 and first resolved in 2024 or later",
        "description": (
            "Frontier research problems that were recognized as unresolved before "
            "1 January 2024 and first resolved on or after that date. Each item carries the "
            "problem as a ready-to-send prompt in three modes (direction, resolution, "
            "strategy), the resolution date and solvers, and a golden solution detailed enough "
            "to grade a model's proposal against."
        ),
        "abductive_framing": {
            "direction": "Stage 2 -- single-hypothesis evaluation: judge one proposed hypothesis (the conjecture) against the evidence available before its resolution and output a binary plus graded plausibility.",
            "strategy": "Stage 1 -- knowledge completion: name the missing intermediate statements and tools that would make the target derivable.",
            "resolution": "Deduction, not abduction: producing the proof itself. Retained because its hallucinated-proof rate is a useful calibration measure, but it is not scored as an abduction task.",
            "caveat": (
                "For the direction mode, what is judged is the truth of a proposition rather "
                "than the explanatory power of a hypothesis about observations. That places it "
                "in the plausible-reasoning family at the edge of abduction rather than at its "
                "centre; it is included on the strength of the framework's own "
                "single-hypothesis-evaluation definition, and labelled as such."
            ),
        },
        "contamination_warning": (
            "Every resolution date is 2024 or later. A model whose training cutoff postdates a "
            "problem's solved_date may simply recall the answer. Filter by "
            "options.model_cutoff to keep only items resolved after the model's cutoff, and run "
            "the leakage_probe subtask to measure recall directly."
        ),
        "sources": [
            "assets/open_problems_2024/sources/pre2024_cutoff_benchmark.txt (primary; 12 task cards)",
            "assets/open_problems_2024/sources/undermind_report.docx (independent re-check; 3 additional problems)",
        ],
        "known_limitations": [
            "Small n: 15 evaluable problems plus 1 held out. Far too few for a leaderboard; use as a qualitative probe or one component of a larger suite.",
            "Field skew: 14 of 16 are mathematics or theoretical computer science. The primary source checked Science's 2025 Breakthrough of the Year, the 2025 Breakthrough Prizes and Physics World's 2025 top ten and found essentially no post-2023 result in chemistry, biology, medicine or the social sciences meeting a strict 'posed before 2024, first resolved after 2023, community-accepted' test: empirical fields rarely have problems crisp enough to close on a datable event.",
            "Difficulty ceiling: full credit on the resolution mode is realistically unattainable for the extreme-tier items; their value is in the direction and strategy modes and in measuring hallucinated proofs.",
            "Statement leakage: stating some problems precisely already signals that they are interesting and unresolved, which shifts a model's prior. The primary source suggests mixing in known-true and known-false conjectures from before 2020 as a directional-bias control.",
            "Ground-truth risk: labels for the Kervaire, Mizohata-Takeuchi and Erdos unit-distance items rest on expert endorsement rather than completed journal peer review.",
            "Base rates: of the 15 evaluable items, 6 resolved affirmatively, 5 by refutation, and the rest are non-binary, so a model that always answers 'the conjecture is true' scores near chance rather than well.",
        ],
        "problems": problems,
    }


def main() -> int:
    data = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    problems = data["problems"]
    print(f"wrote {OUTPUT} with {len(problems)} problems")
    held = sum(1 for p in problems if p.get("held_out"))
    print(f"  evaluable: {len(problems) - held} | held out: {held}")
    modes = {}
    for problem in problems:
        for mode in problem["prompts"]:
            modes[mode] = modes.get(mode, 0) + 1
    print(f"  prompts available: {modes}")
    missing = [p["id"] for p in problems if not p["golden_solution"]["gold_label"]]
    if missing:
        print(f"  WARNING: no gold label for {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
