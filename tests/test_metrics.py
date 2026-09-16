"""Generic metric primitives and answer extraction."""

from __future__ import annotations

import pytest

from abductionbench.core import metrics as M


def test_normalization_and_exact_match():
    assert M.normalize_answer("The  Big-Cat!") == "big cat"
    assert M.exact_match("A cat.", "the cat") == 1.0
    assert M.exact_match("dog", "cat") == 0.0
    assert M.any_exact_match("cat", ["dog", "the cat"]) == 1.0


def test_contains_and_token_f1():
    assert M.contains_match("the diagnosis is acute pancreatitis here", "acute pancreatitis") == 1.0
    assert M.contains_match("nothing", "acute pancreatitis") == 0.0
    assert M.token_f1("a b c", "a b c") == 1.0
    assert M.token_f1("a b c", "x y z") == 0.0
    assert 0 < M.token_f1("a b c", "b c d") < 1
    # empty vs empty is a match; empty vs non-empty is not
    assert M.token_f1("", "") == 1.0
    assert M.token_f1("", "x") == 0.0


def test_rouge_and_bleu_bounds():
    assert M.rouge_l("the cat sat", "the cat sat")["f"] == pytest.approx(1.0)
    assert M.rouge_l("", "x")["f"] == 0.0
    assert M.bleu("the cat sat on the mat", "the cat sat on the mat") == pytest.approx(1.0)
    assert M.bleu("", "x") == 0.0


def test_set_and_ranking_metrics():
    assert M.set_prf({"a", "b"}, {"a", "b"})["f1"] == 1.0
    assert M.set_prf(set(), set())["f1"] == 1.0
    assert M.set_prf({"a"}, {"b"})["f1"] == 0.0
    assert M.jaccard({1, 2}, {2, 3}) == pytest.approx(1 / 3)
    assert M.hits_at_k(["a", "b", "c"], "c", 3) == 1.0
    assert M.hits_at_k(["a", "b", "c"], "c", 2) == 0.0
    assert M.mean_reciprocal_rank(["a", "b"], "b") == 0.5
    assert M.mean_reciprocal_rank(["a"], "z") == 0.0


def test_correlations_and_numeric():
    assert M.spearman([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert M.spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert M.spearman([1], [1]) == 0.0
    assert M.kendall_tau([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert M.numeric_match("10.2", 10.0, rel_tol=0.05) == 1.0
    assert M.numeric_match("20", 10.0, rel_tol=0.05) == 0.0
    assert M.numeric_match("abc", 10.0) == 0.0
    assert M.brier_score(0.9, 1.0) == pytest.approx(0.01)


def test_answer_extraction_variants():
    contract = {"answer_prefix": "Answer:"}
    assert M.extract_answer_span("blah\nAnswer: it rained", contract) == "it rained"
    # last occurrence wins (models often restate the marker)
    assert M.extract_answer_span("Answer: a\nAnswer: b", contract) == "b"
    # no marker -> whole body, so a scorer still gets something
    assert M.extract_answer_span("just prose", contract) == "just prose"
    regex = {"answer_regex": r"\[\[(?P<answer>.+?)\]\]"}
    assert M.extract_answer_span("noise [[the cause]] noise", regex) == "the cause"
    assert M.extract_answer_span(None, contract) == ""
    assert M.extract_answer_span("**bold**", {}) == "bold"


def test_choice_label_extraction():
    labels = ["A", "B", "C", "D"]
    assert M.extract_choice_label("Answer: C", labels, {"answer_prefix": "Answer:"}) == "C"
    assert M.extract_choice_label("I pick (B) because", labels) == "B"
    assert M.extract_choice_label("The correct option is D.", labels) == "D"
    assert M.extract_choice_label("A) rain is likely, so answer A", labels) == "A"
    assert M.extract_choice_label("no idea", labels) is None
    assert M.extract_choice_label("", labels) is None
    # non-letter labels work too
    assert M.extract_choice_label("Answer: hypothesis_2", ["hypothesis_1", "hypothesis_2"],
                                  {"answer_prefix": "Answer:"}) == "hypothesis_2"


def test_aggregation_helpers():
    assert M.mean([]) == 0.0
    assert M.mean([1, 2, 3]) == 2.0
    assert M.std([1, 1, 1]) == 0.0
    summary = M.summarize_numeric([1, 2, 3, 4])
    assert summary["n"] == 4 and summary["mean"] == 2.5
    aggregated = M.aggregate_mean_metrics([{"a": 1.0}, {"a": 0.0, "b": 1.0}])
    assert aggregated == {"a": 0.5, "b": 1.0}  # 'b' missing from sample 1 is skipped, not zeroed
    assert M.macro_average({"g1": [1.0, 1.0], "g2": [0.0]}) == 0.5
