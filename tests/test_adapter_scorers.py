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


def test_synpat_zero_set_equivalence_and_structure():
    """The comparison SynPAT is actually scored by.

    An equation is an expression set to zero, so what identifies it is where it
    vanishes. A sign flip, a scalar multiple and a rearrangement by a
    non-constant factor are all the same equation; only the last of those
    defeats a scalar-factor test, which is why the scalar test is not the one
    used.
    """
    from abductionbench.adapters._mathnorm import same_monomials
    from abductionbench.adapters.synpat import _equal_as_zero_set, _equal_up_to_scale

    symbols = ["Fc", "Fg", "c", "dxdt"]
    gold = "c*Fg - dxdt*Fc"

    assert _equal_as_zero_set(gold, gold, symbols) is True
    assert _equal_as_zero_set("dxdt*Fc - c*Fg", gold, symbols) is True       # sign
    assert _equal_as_zero_set("2*c*Fg - 2*dxdt*Fc", gold, symbols) is True   # scalar
    # The case the scalar test gets wrong: dividing through by c*Fc.
    assert _equal_as_zero_set("Fg/Fc - dxdt/c", gold, symbols) is True
    assert _equal_up_to_scale("Fg/Fc - dxdt/c", gold, symbols) is False
    assert _equal_as_zero_set("c*Fg + dxdt*Fc", gold, symbols) is False

    # structure_match ignores a coefficient the theory alone cannot fix, so the
    # gap between it and the primary metric is readable rather than hidden.
    assert same_monomials("4*c*Fg - dxdt*Fc", gold, symbols) is True
    assert same_monomials("c*Fg - dxdt*Fg", gold, symbols) is False


def test_synpat_prompt_carries_no_data_rows():
    """Samples are withheld on purpose: fitting them would be induction."""
    import inspect

    from abductionbench.adapters import synpat

    source = inspect.getsource(synpat.SynPATAdapter.make_sample)
    assert ".dat" not in source
    assert "data_rows" not in source
    assert "noise_level" not in source
    # What replaces them: the quantities' units, which are part of the axiom
    # system rather than a sample drawn from it.
    assert "units_variables" in source


# --------------------------------------------------------------------------- #
# ABD: deciding logical equivalence against the instance's own worlds
# --------------------------------------------------------------------------- #


def test_abd_model_checker_decides_equivalence():
    from abductionbench.adapters._folmodel import extensionally_equal

    # One world: a has an S-successor that is P, d has one that is not. The
    # second individual is what lets a weaker hypothesis be told apart from
    # the gold -- which is also why the adapter scores against every world the
    # instance ships rather than a few of them.
    worlds = [
        {
            "domain": ["a", "b", "c", "d"],
            "predicates": {"S": ["(a, b)", "(d, c)"], "P": ["b"]},
        }
    ]
    gold = "(exists y (and (S x y) (P y)))"

    assert extensionally_equal(gold, gold, worlds) is True
    # Renaming the bound variable and reordering the conjuncts changes the
    # string and not the hypothesis.
    assert extensionally_equal("(exists z (and (P z) (S x z)))", gold, worlds) is True
    # A double negation, likewise.
    assert extensionally_equal(
        "(not (not (exists y (and (S x y) (P y)))))", gold, worlds
    ) is True
    # A genuinely weaker hypothesis separates different individuals.
    assert extensionally_equal("(exists y (S x y))", gold, worlds) is False
    # A predicate the problem does not have is a WRONG answer, not an
    # unreadable one: ABD forbids the exception predicate Ab in a hypothesis,
    # and an answer that uses it is outside the hypothesis space.
    assert extensionally_equal("(and (S x y) (not (Ab x)))", gold, worlds) is False
    assert extensionally_equal("(exists y (Zz x y))", gold, worlds) is False
    # Undecidable, and the only case a judge is asked about: it will not parse.
    assert extensionally_equal("I could not work it out (((", gold, worlds) is None


def test_abd_model_checker_repairs_the_release_var_wrapper():
    """Two of the 600 golds serialise a bound variable as ``Var(y)``."""
    from abductionbench.adapters._folmodel import extensionally_equal

    worlds = [{"domain": ["a", "b"], "predicates": {"R": ["(a, b)"], "Q": ["b"]}}]
    broken = "(exists Var(y) (and (R x y) (Q y)))"
    fixed = "(exists y (and (R x y) (Q y)))"
    assert extensionally_equal(broken, fixed, worlds) is True


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
# Math normalization (SynPAT)
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


def test_researchbench_selection_is_scored_by_its_answer_key():
    """Only one of ResearchBench's two tasks needs a judge.

    Its ranking items list the release's own gold_hypothesis alongside the
    negatives the release wrote against it, so `gold_label` names the right
    option and a label match settles it. A conversion pass once replaced this
    adapter's whole score() with the judged-only path, which left the selection
    task reporting a metric it never produced.
    """
    from abductionbench.adapters.researchbench import ResearchBenchAdapter

    scorer = object.__new__(ResearchBenchAdapter)
    selection = SampleSpec(
        sample_id="r1",
        fields={"option_labels": ["1", "2", "3"]},
        reference={"gold_label": "2", "gold": "the paper's hypothesis"},
        task_kind="selection",
    )
    right = ResearchBenchAdapter.score(scorer, selection, _response("Answer: 2"))
    assert right.metrics["accuracy"] == 1.0
    wrong = ResearchBenchAdapter.score(scorer, selection, _response("Answer: 3"))
    assert wrong.metrics["accuracy"] == 0.0
    assert "hypothesis_judged" not in right.metrics

    # Generation has no answer key, so it stays with the judge.
    generation = SampleSpec(
        sample_id="r2", fields={}, reference={"gold": "the paper's hypothesis"},
        task_kind="generation",
    )
    judged = ResearchBenchAdapter.score(scorer, generation, _response("a hypothesis"))
    assert "hypothesis_judged" in judged.metrics
    assert "accuracy" not in judged.metrics
