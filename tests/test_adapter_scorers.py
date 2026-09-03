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
    # An unparseable candidate is undecidable, not wrong-by-default.
    assert _symbolic_equal("\\frac{a}{b}", "a/b", ["a", "b"]) is None
    assert _symbolic_equal("", "a/b", ["a", "b"]) is None


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


def test_causalab_edge_scoring():
    from abductionbench.adapters.causalab import CausaLabAdapter

    sample = SampleSpec(
        sample_id="c",
        fields={},
        reference={"edges": ["density->freq", "moisture->freq"], "n_nodes": 3},
        metadata={"variant": "4nodes_golden"},
    )
    scorer = object.__new__(CausaLabAdapter)
    score = CausaLabAdapter.score(
        scorer, sample, _response("density -> freq\nmoisture -> freq")
    )
    assert score.metrics["edge_f1"] == 1.0
    assert score.metrics["exact_graph_match"] == 1.0

    partial = CausaLabAdapter.score(scorer, sample, _response("density -> freq\nfreq -> density"))
    assert partial.metrics["edge_precision"] == pytest.approx(0.5)
    assert partial.metrics["exact_graph_match"] == 0.0


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
    from abductionbench.core.adapter import AdapterContext
    from abductionbench.adapters._base import PooledDatasetAdapter
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
