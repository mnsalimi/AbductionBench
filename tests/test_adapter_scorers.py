"""Unit tests for the non-trivial scorers in the Phase 2 adapters.

The generic helpers are covered by ``test_metrics.py``; what needs its own
tests is the bespoke logic: symbolic equation comparison, formula
canonicalization, hypothesis-set parsing and label-set extraction.  Each test
constructs the score path directly, without any dataset on disk.
"""

from __future__ import annotations

import pytest

from abductionbench.core.types import ModelResponse, ResponseStatus, SampleScore, SampleSpec


def _response(text: str) -> ModelResponse:
    return ModelResponse(
        sample_id="s", model_id="m", status=ResponseStatus.OK, content=text
    )


# --------------------------------------------------------------------------- #
# PhysGym: symbolic equivalence
# --------------------------------------------------------------------------- #


def test_physgym_symbolic_equivalence():
    from abductionbench.adapters.physgym import _extract_expression, _symbolic_equal

    assert _extract_expression("Answer: v = a*b/c", "v") == "a*b/c"
    assert _extract_expression("v = x\nv = y", "v") == "y"  # last assignment wins
    assert _symbolic_equal("a*b/c", "b*a/c", ["a", "b", "c"]) is True
    assert _symbolic_equal("2*a", "a", ["a"]) is False
    # LaTeX is normalized before parsing, so it is decided rather than skipped.
    assert _symbolic_equal("\\frac{a}{b}", "a/b", ["a", "b"]) is True
    # A missing candidate stays undecidable rather than wrong-by-default.
    assert _symbolic_equal("", "a/b", ["a", "b"]) is None
    assert _symbolic_equal(None, "a/b", ["a", "b"]) is None


# --------------------------------------------------------------------------- #
# SynPAT: equality up to a scalar factor
# --------------------------------------------------------------------------- #


def test_synpat_expression_cleaning_and_scaling():
    from abductionbench.adapters.synpat import _clean_expression, _equal_up_to_scale, _parse_system

    assert _clean_expression("Equation 3: x*y - z = 0") == "x*y - z"
    assert _clean_expression("a = b").startswith("(a) - (")
    assert _clean_expression("x^2 + y") == "x**2 + y"

    symbols = ["m", "a", "F"]
    # Expressions set to zero are invariant under non-zero scaling.
    assert _equal_up_to_scale("F - m*a", "2*F - 2*m*a", symbols) is True
    assert _equal_up_to_scale("F - m*a", "F + m*a", symbols) is False
    assert _equal_up_to_scale("not an expression ((", "F - m*a", symbols) is None

    parsed = _parse_system(
        "Variables: ['a', 'b']\nConstants: ['G']\nDerivatives: ['dx']\n"
        "Equations:\na - b\nG*a\nUnits of Measure of Variables: ['m', 'm']\n"
    )
    assert parsed["variables"] == ["a", "b"]
    assert parsed["constants"] == ["G"]
    assert parsed["equations"] == ["a - b", "G*a"]


# --------------------------------------------------------------------------- #
# ABD: formula canonicalization and predicate compliance
# --------------------------------------------------------------------------- #


def test_abd_formula_helpers():
    from abductionbench.adapters.abd import _canonical, _formula, _predicate_compliance

    assert _formula("The answer is (exists y (S x y)) because ...") == "(exists y (S x y))"
    assert _canonical("(exists y  (S x y))") == _canonical("(EXISTS y (S x y))")
    assert _canonical("(and (P x))") != _canonical("(or (P x))")
    assert _predicate_compliance("(exists y (and (S x y) (P y)))", ["S", "P"]) == 1.0
    assert _predicate_compliance("(exists y (and (S x y) (Ab y)))", ["S", "P"]) == 0.0
    assert _predicate_compliance("", ["S"]) == 0.0


# --------------------------------------------------------------------------- #
# HypoSpace: hypothesis parsing and coverage scoring
# --------------------------------------------------------------------------- #


def test_hypospace_parsing_and_coverage():
    from abductionbench.adapters.hypospace import HypoSpaceAdapter, _parse_hypotheses

    parsed = _parse_hypotheses(
        "Hypothesis 1: A -> B, C -> D\nHypothesis 2: none\nHypothesis 3: B -> Z\nsome prose",
        ["A", "B", "C", "D"],
    )
    assert parsed[0] == frozenset({("A", "B"), ("C", "D")})
    assert parsed[1] == frozenset()
    # Edges naming a variable outside the problem are dropped, so that line
    # contributes no graph rather than an invalid one.
    assert len(parsed) == 2

    sample = SampleSpec(
        sample_id="h",
        fields={},
        reference={
            "compatible": [[["A", "B"]], []],
            "n_compatible": 2,
            "nodes": ["A", "B"],
            "requested": 3,
        },
    )
    score = HypoSpaceAdapter.score(
        object.__new__(HypoSpaceAdapter),
        sample,
        _response("Hypothesis 1: A -> B\nHypothesis 2: none\nHypothesis 3: B -> A"),
    )
    # Two of three proposals are compatible; both achievable ones were found.
    assert score.metrics["validity"] == pytest.approx(2 / 3)
    assert score.metrics["distinct_valid_rate"] == 1.0
    assert score.metrics["any_valid"] == 1.0
    assert score.metrics["duplicate_rate"] == 0.0


def test_hypospace_unparseable_is_not_scored_as_wrong():
    from abductionbench.adapters.hypospace import HypoSpaceAdapter

    sample = SampleSpec(
        sample_id="h",
        fields={},
        reference={"compatible": [[["A", "B"]]], "n_compatible": 1, "nodes": ["A", "B"], "requested": 3},
    )
    score = HypoSpaceAdapter.score(
        object.__new__(HypoSpaceAdapter), sample, _response("I think A causes B.")
    )
    assert score.parse_ok is False
    assert score.metrics["distinct_valid_rate"] == 0.0


# --------------------------------------------------------------------------- #
# AER: multi-answer set scoring
# --------------------------------------------------------------------------- #


def test_aer_set_scoring():
    from abductionbench.adapters.aer import AERAdapter

    sample = SampleSpec(
        sample_id="a",
        fields={"option_labels": ["A", "B", "C", "D"]},
        reference={"gold_labels": ["A", "C"], "options": ["w", "x", "y", "z"]},
        task_kind="multi_selection",
    )
    scorer = object.__new__(AERAdapter)
    perfect = AERAdapter.score(scorer, sample, _response("Answer: A, C"), output_contract={"answer_prefix": "Answer:"})
    assert perfect.metrics["exact_set_match"] == 1.0
    assert perfect.metrics["set_f1"] == 1.0

    partial = AERAdapter.score(scorer, sample, _response("Answer: A"), output_contract={"answer_prefix": "Answer:"})
    assert partial.metrics["exact_set_match"] == 0.0
    assert partial.metrics["set_recall"] == 0.5
    assert partial.metrics["set_precision"] == 1.0

    unparseable = AERAdapter.score(scorer, sample, _response("no idea"))
    assert unparseable.parse_ok is False


# --------------------------------------------------------------------------- #
# CausaLab: edge-set extraction
# --------------------------------------------------------------------------- #


def test_causalab_scores_from_the_releases_own_scorecard():
    """CausaLab is scored by DiscoveryWorld, not by anything defined here.

    The adapter used to compare a stated edge list against a gold graph, which
    was a different (and easier) task than the benchmark poses: its own
    scenario keeps a scorecard, and score/maxScore is what the release reports.
    """
    from abductionbench.adapters.causalab import CausaLabAdapter

    scorer = object.__new__(CausaLabAdapter)
    sample = SampleSpec(
        sample_id="c",
        fields={},
        reference={"config": "3_chain"},
        metadata={
            "graph": "3_chain",
            "_episode_state": {
                "turns": 12,
                "final": [{
                    "score": 9, "maxScore": 15, "scoreNormalized": 0.6,
                    "completedSuccessfully": True,
                }],
            },
        },
    )
    score = CausaLabAdapter.score(scorer, sample, _response("irrelevant"))
    assert score.metrics["score_normalized"] == pytest.approx(0.6)
    assert score.metrics["score_raw"] == 9.0
    assert score.metrics["task_completed"] == 1.0
    assert score.metrics["turns_taken"] == 12.0

    # An episode that produced no scorecard is a parse failure, not a zero
    # dressed up as a real score.
    empty = SampleSpec(
        sample_id="c2", fields={}, reference={"config": "3_chain"},
        metadata={"graph": "3_chain", "_episode_state": {"turns": 40}},
    )
    missing = CausaLabAdapter.score(scorer, empty, _response("x"))
    assert missing.parse_ok is False
    assert missing.metrics["score_normalized"] == 0.0


def test_causalab_reads_the_environments_own_action_json():
    from abductionbench.adapters.causalab import _parse_action_json

    assert _parse_action_json('{"action": "TALK", "arg1": 21559}') == {
        "action": "TALK", "arg1": 21559
    }
    # Dialog selection and value entry carry no "action" key at all.
    assert _parse_action_json('sure: {"chosen_dialog_option_int": 1}') == {
        "chosen_dialog_option_int": 1
    }
    assert _parse_action_json('I will set it. {"value": 1254}') == {"value": 1254}
    assert _parse_action_json("no json here") is None


# --------------------------------------------------------------------------- #
# Shared base helpers
# --------------------------------------------------------------------------- #


def test_shared_scorers():
    from abductionbench.adapters._base import selection_score, text_match_score, unparsed_score

    contract = {"answer_prefix": "Answer:"}
    score = selection_score(
        _response("Answer: B"), labels=["A", "B"], gold_label="B", output_contract=contract
    )
    assert score.metrics["accuracy"] == 1.0 and score.parse_ok

    miss = selection_score(
        _response("nothing here"), labels=["A", "B"], gold_label="B", output_contract=contract
    )
    assert miss.parse_ok is False and miss.metrics["accuracy"] == 0.0

    text = text_match_score(
        _response("Answer: acute pancreatitis"),
        gold="Acute pancreatitis",
        output_contract=contract,
        primary="diagnosis_match",
    )
    assert text.metrics["diagnosis_match"] == 1.0
    assert text.metrics["exact_match"] == 1.0

    lenient = text_match_score(
        _response("Answer: most likely acute pancreatitis given the labs"),
        gold="acute pancreatitis",
        output_contract=contract,
        primary="diagnosis_match",
    )
    assert lenient.metrics["diagnosis_match"] == 1.0  # contained
    assert lenient.metrics["exact_match"] == 0.0      # but not equal

    accepted = text_match_score(
        _response("Answer: CMH"),
        gold="Cutaneous meningeal heterotopia (CMH)",
        accepted=["CMH"],
        output_contract=contract,
        primary="diagnosis_match",
    )
    assert accepted.metrics["diagnosis_match"] == 1.0

    blank = unparsed_score(["x", "y"])
    assert blank.parse_ok is False and blank.metrics == {"x": 0.0, "y": 0.0}


def test_pooled_adapter_draw_is_deterministic_and_replacements_disjoint():
    from abductionbench.adapters._base import PooledDatasetAdapter
    from abductionbench.core.adapter import AdapterContext
    from abductionbench.core.types import AdapterDocumentation

    class Dummy(PooledDatasetAdapter):
        def load_items(self):
            self.split_used = "synthetic"
            return list(range(50))

        def make_sample(self, item, index):
            return SampleSpec(sample_id=f"i{item}", fields={"observation": str(item)})

        def score(self, sample, response, *, output_contract=None):
            return SampleScore()

        def documentation(self):
            return AdapterDocumentation(
                dataset_id="dummy", name="d", domain="", source_url="", processing_mode=""
            )

    def build(seed: int) -> list[str]:
        context = AdapterContext(
            dataset_id="dummy", data_dir=__import__("pathlib").Path("/tmp"), sample_size=10, seed=seed
        )
        adapter = Dummy(context)
        adapter.prepare()
        return [s.sample_id for s in adapter.build_samples()]

    first, again, other = build(1), build(1), build(2)
    assert first == again          # same seed -> same draw
    assert first != other          # different seed -> different draw
    assert len(set(first)) == 10

    context = AdapterContext(
        dataset_id="dummy", data_dir=__import__("pathlib").Path("/tmp"), sample_size=10, seed=1
    )
    adapter = Dummy(context)
    adapter.prepare()
    built = [s.sample_id for s in adapter.build_samples()]
    replacements = adapter.replacement_samples(3, set(built))
    assert len(replacements) == 3
    assert not (set(s.sample_id for s in replacements) & set(built))  # never a re-draw


# --------------------------------------------------------------------------- #
# Math normalization (shared by PhysGym and SynPAT)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("candidate", "reference", "symbols"),
    [
        # LaTeX \dfrac with thin spaces (whose backslashes extractors strip).
        (
            "dfrac{mu G M_m}{R,n,(R_m+h)}).",
            "(mu * G * M_m) / (n * R * (R_m + h))",
            ["mu", "G", "M_m", "R", "n", "R_m", "h"],
        ),
        # Unicode Greek, subscripts, superscripts, minus sign and root.
        (
            "2ε₀E₀ / √[1 − (r/a)²]",
            "2 * epsilon_0 * E_0 / (1 - (r/a)**2)**0.5",
            ["epsilon_0", "E_0", "a", "r"],
        ),
        # \sqrt{...} plus implicit multiplication inside.
        (
            r"\sqrt{k T epsilon_0/(2 e**2 n_0)}",
            "(k * T * epsilon_0 / (2 * e**2 * n_0))**0.5",
            ["k", "T", "epsilon_0", "e", "n_0"],
        ),
        # A plain Python-style answer with a variable assignment.
        ("v = a*b/c", "b*a/c", ["a", "b", "c"]),
    ],
)
def test_math_normalization_makes_real_answers_comparable(candidate, reference, symbols):
    """These are the exact answer shapes gpt-oss-120b produced in a live run.

    Before normalization every one of them was 'undecidable' -- one had 0.95
    token overlap with the reference yet scored zero.
    """
    from abductionbench.adapters._mathnorm import equal_expressions

    assert equal_expressions(candidate, reference, symbols) is True


def test_math_normalization_does_not_make_wrong_answers_right():
    from abductionbench.adapters._mathnorm import equal_expressions, equal_up_to_scale

    assert equal_expressions("2*a", "a", ["a"]) is False
    assert equal_expressions("a+b", "a*b", ["a", "b"]) is False
    # Expressions equal to zero are scale-invariant, plain equality is not.
    assert equal_up_to_scale("2*F - 2*m*a", "F - m*a", ["F", "m", "a"]) is True
    assert equal_expressions("2*F - 2*m*a", "F - m*a", ["F", "m", "a"]) is False
    # Empty input stays undecidable rather than counting as a mismatch.
    assert equal_expressions("", "a", ["a"]) is None


@pytest.mark.parametrize(
    ("candidate", "reference", "symbols"),
    [
        # LaTeX Greek *commands* (not the unicode forms) plus a nested \sqrt
        # inside a \frac, and a trailing math-mode "$".
        (
            r"\frac{\sqrt{3} a^4 \Delta p}{12 \eta l}$.",
            "3**0.5 * a**4 * Delta * p / (12 * eta * l)",
            ["a", "Delta", "p", "eta", "l"],
        ),
        # numpy-flavoured references, which PhysGym ships.
        (
            "np.arctan(np.sqrt(c**4 - d**2*(a**2-c**2))/(a*d))",
            "np.arctan(np.sqrt(c**4 - d**2 * (a**2 - c**2)) / (a * d))",
            ["a", "c", "d"],
        ),
        ("math.atan(c**2/(a**2-d**2))", "np.arctan(c**2 / (a**2 - d**2))", ["a", "c", "d"]),
    ],
)
def test_math_normalization_handles_latex_commands_and_numpy(candidate, reference, symbols):
    from abductionbench.adapters._mathnorm import equal_expressions

    assert equal_expressions(candidate, reference, symbols) is True


def test_math_normalization_distinguishes_unreadable_from_wrong():
    """A prose answer stays undecidable; a parseable answer is decided.

    This is the distinction the PhysGym/SynPAT metrics claim to make: an answer
    the scorer cannot read must not be silently counted as a wrong equation.
    """
    from abductionbench.adapters._mathnorm import equal_expressions

    assert equal_expressions("The Debye radius is the square root of the ratio", "(k*T)**0.5", ["k", "T"]) is None
    # A prime is notation for a related quantity: renamed so the answer parses
    # and is judged, rather than reported as unreadable.
    assert equal_expressions("t'**3", "t**3", ["t"]) is False
    assert equal_expressions("t'**3", "t_prime**3", ["t", "t_prime"]) is True
