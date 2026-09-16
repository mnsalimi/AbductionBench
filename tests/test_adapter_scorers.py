"""Unit tests for the non-trivial scorers in the Phase 2 adapters.

The generic helpers are covered by ``test_metrics.py``; what needs its own
tests is the bespoke logic: symbolic equation comparison, formula
canonicalization, hypothesis-set parsing and label-set extraction.  Each test
constructs the score path directly, without any dataset on disk.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

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
# ABD: scored by the release's own Z3 evaluator, not by matching the gold
# --------------------------------------------------------------------------- #


def test_abd_formula_extraction_and_canonicalization():
    from abductionbench.adapters.abd import _canonical, _formula

    # Balanced-paren extraction, because the release's own regex stops at the
    # first ')' and this suite's prompt does not use its JSON contract.
    assert _formula("The answer is (exists y (S x y)) because ...") == "(exists y (S x y))"
    assert _canonical("(exists y  (S x y))") == _canonical("(EXISTS y (S x y))")
    assert _canonical("(and (P x))") != _canonical("(or (P x))")


def test_abd_scoring_credits_a_valid_non_gold_repair():
    """The defect this scoring replaced.

    A formula that repairs every prompt world is correct even when it is
    nothing like the planted gold, and the old gold-matching metric scored
    exactly that case zero. The scorer is driven here with a stubbed evaluator
    result so the test needs neither the clone nor z3.
    """
    from abductionbench.adapters.abd import ABDAdapter

    class _Result:
        crashed = False
        valid = True
        parse_error = None
        total_cost = 34
        total_opt_cost = 14
        total_gap = 20
        avg_gap = 2.0
        cost_vs_gold = 16
        forbidden_preds_used = None
        trailing_parens_added = 0

    adapter = object.__new__(ABDAdapter)
    adapter._evaluate = lambda instance_id, alpha: _Result()
    sample = SampleSpec(
        sample_id="a",
        fields={},
        reference={
            "gold": "(exists y (exists z (and (S x y) (S x z))))",
            "instance_id": "ABD_FULL_TH10_000",
        },
        metadata={"scenario": "ABD_FULL", "difficulty": "hard"},
    )
    score = ABDAdapter.score(adapter, sample, _response("(exists y (and (S x y) (P y)))"))

    assert score.metrics["valid"] == 1.0           # the release's verdict
    assert score.metrics["optimal"] == 0.0         # valid, but 20 above the optimum
    assert score.metrics["formula_match"] == 0.0   # nothing like the gold, and that is fine
    assert score.metrics["valid_ABD_FULL"] == 1.0
    assert score.metrics["total_cost"] == 34.0
    assert score.metrics["total_gap"] == 20.0


def test_abd_invalid_answers_report_why_and_omit_cost():
    """Cost is undefined for an answer that repairs nothing, so it is absent."""
    from abductionbench.adapters.abd import ABDAdapter

    class _Rejected:
        crashed = False
        valid = False
        parse_error = "Formula uses forbidden predicate 'Ab' (circular definition)"
        total_cost = None
        total_opt_cost = 0
        total_gap = None
        avg_gap = None
        cost_vs_gold = None
        forbidden_preds_used = ["Ab"]
        trailing_parens_added = 0

    adapter = object.__new__(ABDAdapter)
    adapter._evaluate = lambda instance_id, alpha: _Rejected()
    sample = SampleSpec(
        sample_id="a", fields={}, reference={"gold": "(P x)", "instance_id": "i"}, metadata={}
    )
    score = ABDAdapter.score(adapter, sample, _response("(Ab x)"))

    assert score.metrics["valid"] == 0.0
    assert score.metrics["forbidden_predicate_use"] == 1.0
    assert score.metrics["formula_parse_error"] == 0.0   # scope, not syntax
    # Reporting 0 here would make the worst answer look like the cheapest one.
    assert "total_cost" not in score.metrics
    assert "total_gap" not in score.metrics


def test_abd_isolated_evaluator_survives_a_dead_worker():
    """A scorer is not allowed to end a run.

    The release's evaluator wraps a native solver, and a native solver can
    abort the process -- z3-solver 5.1.0 did exactly that on a model answer
    during a live run, with an assertion violation no Python except clause can
    catch. So it runs in a separate process. Here the worker is killed outright,
    which is the same thing from the parent's point of view.
    """
    import os
    import signal
    import time

    from abductionbench.adapters._abd_eval import IsolatedEvaluator

    evaluator = IsolatedEvaluator(Path("/nonexistent"), deadline_s=30)
    try:
        # The first call fails on the import, which is an ordinary exception:
        # reported as a crash, and the worker stays usable.
        first = evaluator.evaluate({}, "(P x)", 5000)
        assert first.crashed and "ModuleNotFoundError" in first.crash_reason
        assert evaluator._pool is not None

        for pid in list(evaluator._pool._processes):
            os.kill(pid, signal.SIGKILL)
        time.sleep(0.3)

        killed = evaluator.evaluate({}, "(P x)", 5000)
        assert killed.crashed
        assert evaluator.crashes >= 1
        # And the next answer is evaluated by a fresh worker rather than
        # inheriting the broken one.
        again = evaluator.evaluate({}, "(P x)", 5000)
        assert again.crashed and "ModuleNotFoundError" in again.crash_reason
    finally:
        evaluator.close()


def test_abd_evaluator_crash_scores_nothing_rather_than_zero():
    """A solver that fell over is not evidence about the model."""
    from abductionbench.adapters._abd_eval import EvalOutcome
    from abductionbench.adapters.abd import ABDAdapter

    adapter = object.__new__(ABDAdapter)
    adapter._evaluate = lambda instance_id, alpha: EvalOutcome(
        crashed=True, crash_reason="the solver process died (BrokenProcessPool)"
    )
    sample = SampleSpec(
        sample_id="a", fields={}, reference={"gold": "(P x)", "instance_id": "i"}, metadata={}
    )
    score = ABDAdapter.score(adapter, sample, _response("(P x)"))

    assert score.metrics == {"evaluator_crashed": 1.0}
    assert "valid" not in score.metrics      # absent, not 0.0
    assert "evaluator_crash" in score.details


def test_abd_excludes_records_whose_own_gold_breaks_the_rules():
    """Detected and named, never repaired.

    Forty of the release's 600 records are unusable: two whose gold its own
    parser rejects, and thirty-eight whose gold uses a predicate the record
    itself forbids -- the evaluator scores those golds invalid. The check is
    driven here with a stubbed release so the test needs no clone.
    """
    from types import SimpleNamespace

    from abductionbench.adapters.abd import ABDAdapter

    class _Parsed:
        ast = object()
        trailing_parens_added = 0

    def _release(*, parses=True, scoped=True, used=("P",)):
        def parse(_text):
            if not parses:
                raise ValueError("Expected variable name after 'exists', got 'Var'")
            return _Parsed()

        return SimpleNamespace(
            evaluator=SimpleNamespace(
                parse_alpha_formula_with_suffix_repair=parse,
                validate_alpha_predicate_scoping=lambda *_a: (
                    (True, None, None) if scoped else (False, "uses forbidden ['Q']", ["Q"])
                ),
            ),
            used_predicates=lambda _ast: set(used),
        )

    adapter = object.__new__(ABDAdapter)
    problem = {"gold": {"alpha": "(P x)"}, "allowedAlphaPreds": ["P", "R", "S"], "theoryId": "TH2"}

    adapter._release = _release()
    assert ABDAdapter._inspect_gold(adapter, problem) == ([], [])

    adapter._release = _release(parses=False)
    defects, _ = ABDAdapter._inspect_gold(adapter, problem)
    assert defects and "parser rejects" in defects[0]

    adapter._release = _release(scoped=False, used=("P", "Q"))
    defects, _ = ABDAdapter._inspect_gold(adapter, problem)
    # Both readings of the same rule fire: the release's own scoping check and
    # the record's own allowedAlphaPreds field.
    assert len(defects) == 2
    assert any("evaluator rejects its gold" in d for d in defects)
    assert any("allowedAlphaPreds" in d for d in defects)


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


# --------------------------------------------------------------------------- #
# UniADILR-HGc: the supporting pair, as a set
# --------------------------------------------------------------------------- #


def test_uniadilr_premise_ids_read_only_the_left_of_the_arrow():
    """The right of `->` is the claim, not a premise."""
    from abductionbench.adapters.uniadilr_hgc import _premise_ids

    assert _premise_ids("sent5 & sent13 -> Sarah is a talented programmer.") == {5, 13}
    # A claim whose own text contains a sent-like token must not become a third
    # premise: this is why the split happens before the search.
    assert _premise_ids("sent2 & sent19 -> sent7 was mentioned in the report.") == {2, 19}
    assert _premise_ids("sent9") == {9}


def test_uniadilr_scores_the_set_order_independently_and_any_size():
    from abductionbench.adapters.uniadilr_hgc import UniADILRHGcAdapter

    adapter = object.__new__(UniADILRHGcAdapter)

    def score(gold, text):
        sample = SampleSpec(
            sample_id="u", fields={}, reference={"premises": list(gold)}, metadata={}
        )
        return UniADILRHGcAdapter.score(adapter, sample, _response(text))

    # two premises
    assert score([5, 13], "5 13").metrics["premise_set_match"] == 1.0
    assert score([5, 13], "13 5").metrics["premise_set_match"] == 1.0      # order-free
    assert score([5, 13], "sent13, sent5").metrics["premise_set_match"] == 1.0
    # one premise, and three: both are real items and both must score
    assert score([4], "4").metrics["premise_set_match"] == 1.0
    assert score([1, 7, 19], "19 1 7").metrics["premise_set_match"] == 1.0
    # cardinality counts -- the prompt does not say how many, so getting the
    # number wrong is a wrong answer
    assert score([5, 13], "5 13 99").metrics["premise_set_match"] == 0.0
    assert score([5, 13], "5").metrics["premise_set_match"] == 0.0
    assert score([1, 7, 19], "1 7").metrics["premise_count_match"] == 0.0
    assert score([1, 7, 19], "1 7 99").metrics["premise_count_match"] == 1.0


def test_uniadilr_partial_credit_does_not_reward_naming_everything():
    """F1, not recall: listing the whole pool must not look like understanding."""
    from abductionbench.adapters.uniadilr_hgc import UniADILRHGcAdapter

    adapter = object.__new__(UniADILRHGcAdapter)
    sample = SampleSpec(
        sample_id="u", fields={}, reference={"premises": [1, 7, 19]}, metadata={}
    )
    shotgun = UniADILRHGcAdapter.score(
        adapter, sample, _response(" ".join(str(i) for i in range(1, 24)))
    )
    assert shotgun.metrics["premise_set_match"] == 0.0
    assert shotgun.metrics["premise_f1"] < 0.3      # recall would have been 1.0
    assert shotgun.metrics["premise_count_match"] == 0.0

    close = UniADILRHGcAdapter.score(adapter, sample, _response("1 7 99"))
    assert close.metrics["premise_f1"] > shotgun.metrics["premise_f1"]


def test_uniadilr_only_an_answer_with_no_numbers_fails_to_parse():
    """A wrong count is a wrong answer, not a malformed one."""
    from abductionbench.adapters.uniadilr_hgc import UniADILRHGcAdapter

    adapter = object.__new__(UniADILRHGcAdapter)
    sample = SampleSpec(
        sample_id="u", fields={}, reference={"premises": [5, 13]}, metadata={}
    )
    for text in ("I cannot tell", ""):
        score = UniADILRHGcAdapter.score(adapter, sample, _response(text))
        assert score.parse_ok is False, text
    # ... whereas these parse fine and are simply scored wrong
    for text in ("5", "5 13 7"):
        score = UniADILRHGcAdapter.score(adapter, sample, _response(text))
        assert score.parse_ok is True, text
        assert score.metrics["premise_set_match"] == 0.0


def test_uniadilr_keeps_items_of_every_premise_count():
    """The prompt no longer fixes the count, so no item is unanswerable."""
    import inspect

    from abductionbench.adapters import uniadilr_hgc

    source = inspect.getsource(uniadilr_hgc.UniADILRHGcAdapter.load_items)
    assert "len(premises) == 2" not in source
    assert "premise_counts" in source
    constraints = " ".join(uniadilr_hgc.UniADILRHGcAdapter.task_requirements)
    assert "exactly two" not in constraints


def test_uniadilr_is_objective_and_has_no_judge():
    """A set comparison settles this; there is nothing for a judge to add."""
    from abductionbench.adapters.uniadilr_hgc import UniADILRHGcAdapter

    assert UniADILRHGcAdapter.objective_metrics is True
    assert UniADILRHGcAdapter.primary_metric == "premise_set_match"
    # judge_request/apply_judge are the base class's no-ops, not overrides.
    assert "judge_request" not in vars(UniADILRHGcAdapter)
    assert "apply_judge" not in vars(UniADILRHGcAdapter)


# --------------------------------------------------------------------------- #
# ABD: a batch is evaluated in parallel, and comes out the same
# --------------------------------------------------------------------------- #


def test_abd_concurrent_scoring_deduplicates_identical_questions():
    """Eight threads asking the same question must pay for one Z3 run.

    The engine scores a batch concurrently, and repeats of a record often
    produce the same formula. Without the in-flight map every one of them would
    start its own solver run, because none is in the cache yet.
    """
    import concurrent.futures
    import threading

    from abductionbench.adapters._abd_eval import EvalOutcome
    from abductionbench.adapters.abd import ABDAdapter

    calls = []
    barrier = threading.Barrier(1)  # unused; kept explicit that no coordination is needed

    class _Isolated:
        deadline_s = 30.0

        def evaluate(self, problem, alpha, timeout_ms):
            calls.append(alpha)
            time.sleep(0.2)          # long enough that the others really overlap
            return EvalOutcome(valid=True, total_cost=3, total_opt_cost=3, total_gap=0,
                               avg_gap=0.0, cost_vs_gold=0)

    adapter = object.__new__(ABDAdapter)
    adapter._problems = {"i": {"scenario": "ABD_FULL"}}
    adapter._isolated = _Isolated()
    adapter.context = SimpleNamespace(option=lambda _name, default=None: default)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: adapter._evaluate("i", "(P x)"), range(8)))

    assert len(calls) == 1, f"expected one evaluation, got {len(calls)}"
    assert all(r is results[0] for r in results)
    del barrier


def test_abd_a_broken_pool_does_not_condemn_the_sibling_that_shared_it():
    """One bad formula takes the whole pool down; the others must survive it.

    A native abort kills every worker, so an innocent answer's future raises
    BrokenProcessPool too. Charging that to the innocent answer would inflate
    evaluator_crashed and lose real verdicts, so it is retried once on the
    fresh pool.
    """
    from concurrent.futures import BrokenExecutor

    from abductionbench.adapters._abd_eval import IsolatedEvaluator

    evaluator = IsolatedEvaluator(Path("/nonexistent"), deadline_s=5)
    attempts = {"n": 0}

    class _Pool:
        def submit(self, *_args, **_kwargs):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise BrokenExecutor("a sibling aborted the pool")

            class _Future:
                @staticmethod
                def result(timeout=None):
                    return {"valid": True, "total_cost": 1, "total_opt_cost": 1,
                            "total_gap": 0, "avg_gap": 0.0, "cost_vs_gold": 0,
                            "parse_error": None, "forbidden_preds_used": (),
                            "trailing_parens_added": 0}

            return _Future()

        def shutdown(self, **_kwargs):
            pass

    evaluator._pool = _Pool()
    evaluator._ensure_pool = lambda: evaluator._pool or _Pool()
    # first submit raises, second (after the restart) succeeds
    evaluator._discard_pool = lambda broken=None: None
    outcome = evaluator.evaluate({}, "(P x)", 5000)

    assert attempts["n"] == 2, "the sibling should have been retried once"
    assert outcome.crashed is False
    assert outcome.valid is True
    assert evaluator.crashes == 1      # the breakage is still counted
