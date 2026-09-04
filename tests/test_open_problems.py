"""The Open Problems 2024+ adapter: dataset integrity and mode-specific scoring.

The dataset ships with the repository, so these tests check the real file rather
than a fixture -- a malformed gold label or a missing prompt is a data bug that
would silently misgrade a run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from abductionbench.adapters.open_problems import (
    OpenProblemsAdapter,
    _all_numbers,
    _parse_confidence,
    _phrase_recall,
    _resolved_after,
)
from abductionbench.core.adapter import AdapterContext
from abductionbench.core.types import ModelResponse, ResponseStatus, SampleSpec

DATASET = Path(__file__).resolve().parents[1] / "assets" / "open_problems_2024" / "problems.json"


def _response(text: str) -> ModelResponse:
    return ModelResponse(sample_id="s", model_id="m", status=ResponseStatus.OK, content=text)


def _adapter(**options) -> OpenProblemsAdapter:
    context = AdapterContext(
        dataset_id="open_problems_2024", data_dir=Path("/tmp"), sample_size=300, seed=1,
        options=options,
    )
    return OpenProblemsAdapter(context)


# --------------------------------------------------------------------------- #
# dataset integrity
# --------------------------------------------------------------------------- #


def test_dataset_is_well_formed():
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    problems = data["problems"]
    assert len(problems) >= 15
    seen_ids, seen_slugs = set(), set()
    for problem in problems:
        assert problem["id"] not in seen_ids
        assert problem["slug"] not in seen_slugs
        seen_ids.add(problem["id"])
        seen_slugs.add(problem["slug"])
        gold = problem["golden_solution"]
        # Every item must be gradeable: a label out of its own label set, or a
        # numeric target.
        assert gold["gold_label"], problem["id"]
        if gold["label_set"]:
            assert gold["gold_label"] in gold["label_set"], problem["id"]
            for partial in gold["partial_labels"]:
                assert partial in gold["label_set"], problem["id"]
        else:
            assert problem.get("numeric_target"), problem["id"]
        # The user-facing keys must be present and non-empty.
        assert problem["prompts"].get("a"), problem["id"]
        assert problem["solved_date"] and len(problem["solved_date"]) == 10
        assert problem["solver"], problem["id"]
        assert problem["problem_posed"]["year"], problem["id"]
        assert problem["field"] and problem["title"]


def test_every_problem_was_resolved_after_2023():
    """The dataset's whole premise: posed before 2024, first resolved after it."""
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    for problem in data["problems"]:
        assert problem["solved_date"] >= "2024-01-01", problem["id"]
        assert int(problem["problem_posed"]["year"]) < 2024, problem["id"]


def test_strategy_items_have_ingredients_to_grade_against():
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    for problem in data["problems"]:
        if problem["prompts"].get("c"):
            assert problem["golden_solution"]["key_ingredients"], problem["id"]


def test_direction_gold_labels_are_balanced_enough_to_be_informative():
    """A model that always answers 'the conjecture holds' must not score well."""
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    affirmative = {"YES", "TRUE", "ALWAYS-TRUE", "YES-BUT-REQUIRES-NEW-IDEAS",
                   "YES-ACHIEVABLE-WITH-CURRENT-IDEAS"}
    labels = [p["golden_solution"]["gold_label"] for p in data["problems"]
              if p["golden_solution"]["label_set"]]
    yes = sum(1 for label in labels if label in affirmative)
    assert 0.3 <= yes / len(labels) <= 0.7, f"gold labels are skewed: {yes}/{len(labels)} affirmative"


# --------------------------------------------------------------------------- #
# sample construction
# --------------------------------------------------------------------------- #


def test_default_modes_are_the_two_abductive_ones():
    adapter = _adapter()
    adapter.prepare()
    samples = adapter.build_samples()
    kinds = {s.task_kind for s in samples}
    assert kinds == {"direction_judgment", "solution_strategy"}
    # The deductive mode must not appear unless it is asked for.
    assert "full_resolution" not in kinds


def test_resolution_mode_is_opt_in():
    adapter = _adapter(subtask="resolution")
    adapter.prepare()
    assert {s.task_kind for s in adapter.build_samples()} == {"full_resolution"}


def test_model_cutoff_filters_to_items_the_model_cannot_know():
    early = _adapter(model_cutoff="2025-06-01")
    early.prepare()
    kept = {s.metadata["problem_id"] for s in early.build_samples()}
    dates = {s.metadata["problem_id"]: s.metadata["solved_date"] for s in early.build_samples()}
    assert kept, "cutoff filtered everything"
    assert all(date > "2025-06-01" for date in dates.values())
    # Without a cutoff, strictly more problems are in play.
    every = _adapter()
    every.prepare()
    assert len({s.metadata["problem_id"] for s in every.build_samples()}) > len(kept)


def test_held_out_problem_is_excluded_by_default():
    default = _adapter()
    default.prepare()
    assert not any(s.metadata["held_out"] for s in default.build_samples())
    included = _adapter(include_held_out=True)
    included.prepare()
    assert any(s.metadata["held_out"] for s in included.build_samples())


def test_unknown_subtask_is_reported_not_guessed():
    from abductionbench.core.adapter import SkippedDataset

    with pytest.raises(SkippedDataset, match="unknown options.subtask"):
        _adapter(subtask="nonsense").prepare()


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #


def _direction_sample(gold="YES", labels=("YES", "NO"), partial=()) -> SampleSpec:
    return SampleSpec(
        sample_id="d", fields={"option_labels": list(labels)},
        reference={"mode": "direction", "gold_label": gold, "label_set": list(labels),
                   "partial_labels": list(partial), "key_ingredients": [], "key_method": "",
                   "answer": "", "numeric_target": None, "proof_status_gold": None,
                   "expected_groups": [], "solvers": [], "solved_date": "2025-01-01"},
        task_kind="direction_judgment",
    )


def test_direction_scoring_and_calibration():
    adapter = _adapter()
    contract = {"answer_prefix": "Answer:"}
    right = adapter.score(_direction_sample(), _response("Answer: YES\nConfidence: 0.9"),
                          output_contract=contract)
    assert right.metrics["direction_accuracy"] == 1.0
    assert right.metrics["brier_score"] == pytest.approx(0.01)
    assert right.metrics["overconfident_wrong_rate"] == 0.0

    wrong = adapter.score(_direction_sample(), _response("Answer: NO\nConfidence: 0.95"),
                          output_contract=contract)
    assert wrong.metrics["direction_accuracy"] == 0.0
    # Confidently wrong is the outcome the source rubric penalises most.
    assert wrong.metrics["overconfident_wrong_rate"] == 1.0
    assert wrong.metrics["brier_score"] == pytest.approx(0.9025)

    hedged = adapter.score(_direction_sample(), _response("Answer: NO\nConfidence: 0.5"),
                           output_contract=contract)
    assert hedged.metrics["overconfident_wrong_rate"] == 0.0


def test_partial_credit_for_right_direction_wrong_strength():
    adapter = _adapter()
    sample = _direction_sample(
        gold="YES-BUT-REQUIRES-NEW-IDEAS",
        labels=("YES-ACHIEVABLE-WITH-CURRENT-IDEAS", "YES-BUT-REQUIRES-NEW-IDEAS",
                "NO-OBSTRUCTION-EXISTS"),
        partial=("YES-ACHIEVABLE-WITH-CURRENT-IDEAS",),
    )
    score = adapter.score(
        sample, _response("Answer: YES-ACHIEVABLE-WITH-CURRENT-IDEAS\nConfidence: 0.6"),
        output_contract={"answer_prefix": "Answer:"},
    )
    assert score.metrics["direction_accuracy"] == 0.5
    wrong = adapter.score(
        sample, _response("Answer: NO-OBSTRUCTION-EXISTS\nConfidence: 0.6"),
        output_contract={"answer_prefix": "Answer:"},
    )
    assert wrong.metrics["direction_accuracy"] == 0.0


def test_no_verdict_is_a_refusal_not_a_wrong_answer():
    adapter = _adapter()
    score = adapter.score(_direction_sample(), _response("This is a deep open problem."),
                          output_contract={"answer_prefix": "Answer:"})
    assert score.parse_ok is False
    assert score.metrics["refusal_rate"] == 1.0
    assert score.metrics["direction_accuracy"] == 0.0


def test_numeric_answer_picks_the_intended_number():
    """"Answer: 47,176,870 steps for a 5-state machine" must not score the 5."""
    adapter = _adapter()
    sample = SampleSpec(
        sample_id="n", fields={"option_labels": []},
        reference={"mode": "direction", "gold_label": "47176870", "label_set": [],
                   "partial_labels": [], "key_ingredients": [], "key_method": "", "answer": "",
                   "numeric_target": {"value": 47176870, "unit": "steps", "tolerance_rel": 0.0},
                   "proof_status_gold": "PROVED", "expected_groups": [], "solvers": [],
                   "solved_date": "2024-07-02"},
        task_kind="direction_judgment",
    )
    good = adapter.score(
        sample, _response("Answer: 47,176,870 steps for a 5-state machine, and it is PROVED"),
        output_contract={"answer_prefix": "Answer:"},
    )
    assert good.metrics["direction_accuracy"] == 1.0
    assert good.metrics["proof_status_correct"] == 1.0
    bad = adapter.score(
        sample, _response("Answer: about 47 million steps"),
        output_contract={"answer_prefix": "Answer:"},
    )
    assert bad.metrics["direction_accuracy"] == 0.0


def test_strategy_scoring_rewards_named_ingredients():
    adapter = _adapter()
    sample = SampleSpec(
        sample_id="s", fields={"observation": "how would you attack it"},
        reference={"mode": "strategy", "gold_label": "YES", "label_set": [], "partial_labels": [],
                   "key_ingredients": ["volume estimate for unions of tubes/convex sets",
                                       "multi-scale induction",
                                       "grains/planebrush structure theory"],
                   "key_method": "a volume estimate for unions of tubes; multi-scale induction",
                   "answer": "", "numeric_target": None, "proof_status_gold": None,
                   "expected_groups": [], "solvers": [], "solved_date": "2025-02-24"},
        task_kind="solution_strategy",
    )
    good = adapter.score(
        sample,
        _response(
            "I would prove a volume estimate for unions of tubes, then run a multi-scale "
            "induction over scales, and develop a grains and planebrush structure theory."
        ),
    )
    # All three ingredients named, in the model's own words.
    assert good.metrics["key_ingredient_recall"] == 1.0
    assert good.metrics["named_any_ingredient"] == 1.0

    generic = adapter.score(sample, _response("I would use induction and try a computer search."))
    assert generic.metrics["key_ingredient_recall"] == 0.0
    assert generic.metrics["named_any_ingredient"] == 0.0


def test_resolution_mode_measures_hallucinated_proofs_not_abduction():
    adapter = _adapter()
    sample = SampleSpec(
        sample_id="r", fields={"option_labels": ["YES", "NO"]},
        reference={"mode": "resolution", "gold_label": "YES", "label_set": ["YES", "NO"],
                   "partial_labels": [], "key_ingredients": [], "key_method": "", "answer": "",
                   "numeric_target": None, "proof_status_gold": None, "expected_groups": [],
                   "solvers": [], "solved_date": "2025-02-24"},
        task_kind="full_resolution",
    )
    fabricated = adapter.score(
        sample, _response("Theorem. ... Combining the lemmas, this completes the proof. Answer: YES"),
        output_contract={"answer_prefix": "Answer:"},
    )
    assert fabricated.metrics["hallucinated_proof_rate"] == 1.0
    assert "abduction_score" not in fabricated.metrics   # deduction is not scored as abduction

    honest = adapter.score(
        sample, _response("I cannot resolve this problem; it is beyond what I can prove. Answer: YES"),
        output_contract={"answer_prefix": "Answer:"},
    )
    assert honest.metrics["honest_abstention_rate"] == 1.0
    assert honest.metrics["hallucinated_proof_rate"] == 0.0


def test_leakage_probe_detects_post_cutoff_knowledge():
    adapter = _adapter(subtask="leakage_probe")
    sample = SampleSpec(
        sample_id="l", fields={},
        reference={"mode": "leakage_probe", "gold_label": "YES", "label_set": [],
                   "partial_labels": [], "key_ingredients": [], "key_method": "", "answer": "",
                   "numeric_target": None, "proof_status_gold": None, "expected_groups": [],
                   "solvers": ["Hong Wang", "Joshua Zahl"], "solved_date": "2025-02-24"},
        task_kind="leakage_probe",
    )
    leaked = adapter.score(
        sample, _response("This was resolved in 2025 by Hong Wang and Joshua Zahl.")
    )
    assert leaked.metrics["leakage_rate"] == 1.0
    assert leaked.metrics["named_solver"] == 1.0

    clean = adapter.score(
        sample, _response("As of my knowledge cutoff this remains open; the best bound is 2.5.")
    )
    assert clean.metrics["leakage_rate"] == 0.0
    assert clean.metrics["claims_resolved"] == 0.0


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def test_helper_functions():
    assert _resolved_after("2025-02-24", "2023-12-31") is True
    assert _resolved_after("2024-01-01", "2025-01-01") is False
    assert _resolved_after("unknown", "2023-12-31") is True   # unparseable dates are kept
    assert _all_numbers("Answer: 47,176,870 steps (5 states)") == [47176870.0, 5.0]
    assert _parse_confidence("Confidence: 0.85") == pytest.approx(0.85)
    assert _parse_confidence("probability 85%") == pytest.approx(0.85)
    assert _parse_confidence("no number here") is None
    assert _phrase_recall("we use stochastic localization", "stochastic localization") == 1.0
    assert _phrase_recall("nothing relevant", "stochastic localization") == 0.0


def test_primary_metric_follows_the_mode():
    """A mode that produces no abduction_score must not advertise it."""
    assert _adapter().primary_metric == "abduction_score"
    assert _adapter(subtask="strategy").primary_metric == "abduction_score"
    assert _adapter(subtask="leakage_probe").primary_metric == "leakage_rate"
    assert _adapter(subtask="resolution").primary_metric == "hallucinated_proof_rate"


def test_repeats_multiply_the_items_with_distinct_ids():
    once = _adapter()
    once.prepare()
    base = once.build_samples()

    thrice = _adapter(repeats=3)
    thrice.prepare()
    repeated = thrice.build_samples()

    assert len(repeated) == 3 * len(base)
    # Distinct sample ids (so the engine treats them as separate observations),
    # identical prompts (so they are repeats and not variations).
    assert len({s.sample_id for s in repeated}) == len(repeated)
    by_problem: dict[tuple, set] = {}
    for sample in repeated:
        key = (sample.metadata["problem_id"], sample.metadata["mode"])
        by_problem.setdefault(key, set()).add(sample.fields["observation"])
    assert all(len(prompts) == 1 for prompts in by_problem.values())
    assert {s.metadata["repeat"] for s in repeated} == {0, 1, 2}
    assert "x 3 repeats" in thrice.split_used


def test_direction_stability_reports_disagreement_between_repeats():
    adapter = _adapter(repeats=3)
    adapter.prepare()
    contract = {"answer_prefix": "Answer:"}

    def score_of(problem_id: str, answer: str):
        sample = _direction_sample()
        sample.metadata["problem_id"] = problem_id
        return adapter.score(sample, _response(f"Answer: {answer}\nConfidence: 0.6"),
                             output_contract=contract)

    # One question answered consistently, one that flips on its third asking.
    stable = [score_of("P1", "YES") for _ in range(3)]
    flipping = [score_of("P2", "YES"), score_of("P2", "YES"), score_of("P2", "NO")]
    metrics = adapter.aggregate(stable + flipping)
    assert metrics["repeats"] == 3.0
    assert metrics["direction_answer_stability"] == 0.5
    # The score itself is the mean over every observation, not over questions.
    assert metrics["direction_accuracy"] == pytest.approx(5 / 6)


def test_stability_is_not_reported_when_nothing_was_repeated():
    adapter = _adapter()
    adapter.prepare()
    sample = _direction_sample()
    sample.metadata["problem_id"] = "P1"
    metrics = adapter.aggregate([
        adapter.score(sample, _response("Answer: YES\nConfidence: 0.6"),
                      output_contract={"answer_prefix": "Answer:"})
    ])
    assert metrics["repeats"] == 1.0
    assert "direction_answer_stability" not in metrics


def test_strategy_stability_is_score_spread_not_string_equality():
    """Two strategy answers are never byte-identical; their scores can still agree."""
    adapter = _adapter(repeats=2)
    adapter.prepare()
    ingredients = ["stochastic localization", "multi-scale induction"]

    def score_of(problem_id: str, text: str):
        sample = SampleSpec(
            sample_id=problem_id, fields={"observation": "how would you attack it"},
            reference={"mode": "strategy", "gold_label": "", "label_set": [],
                       "partial_labels": [], "key_ingredients": ingredients,
                       "key_method": "", "answer": "", "numeric_target": None,
                       "proof_status_gold": None, "expected_groups": [], "solvers": [],
                       "solved_date": "2025-01-01"},
            task_kind="solution_strategy", metadata={"problem_id": problem_id},
        )
        return adapter.score(sample, _response(text))

    metrics = adapter.aggregate([
        # Same one ingredient named both times, in different prose: no spread.
        score_of("P1", "I would apply stochastic localization to the measure."),
        score_of("P1", "The route is stochastic localization, applied carefully."),
        # Two ingredients one time, none the next: the full spread.
        score_of("P2", "Use stochastic localization plus a multi-scale induction."),
        score_of("P2", "Probably just a computer search."),
    ])
    assert metrics["strategy_score_spread"] == pytest.approx(0.5)   # mean of 0.0 and 1.0
    # Free text must not be counted as an unstable verdict.
    assert "direction_answer_stability" not in metrics
