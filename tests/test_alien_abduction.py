"""Alien Abduction: the target set, the sandbox, and the three output modes.

The three modes share one target set and one scorer, so what is tested here is
the shape of each mode's protocol and the properties the paper's design turns
on: identical targets across modes, evidence that is a prefix of the sequential
mode's stream, and a solve criterion that is agreement on the whole suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from abductionbench.adapters import _alien_targets as T
from abductionbench.adapters.alien_abduction import (
    PARSE_ERROR_REPLY,
    PRELOADED_EXAMPLES,
    TURN_BUDGET,
    AlienAbductionActiveAdapter,
    AlienAbductionAdapter,
    AlienAbductionPassiveAdapter,
    _parse_test_input,
    extract_code,
)
from abductionbench.core.adapter import AdapterContext
from abductionbench.core.modes import TaskModes
from abductionbench.core.sandbox import verify
from abductionbench.core.types import ModelResponse, ResponseStatus

ADD_SEVEN = "def solution(x: int) -> int:\n    return x + 7 if x < 0 else x"


def _adapter(cls, delivery, prompt_mode="io", dataset_id="alien"):
    modes = TaskModes(
        prompt_mode=prompt_mode, selection_mode=None, data_delivery_mode=delivery
    )
    adapter = cls(
        AdapterContext(
            dataset_id=dataset_id, data_dir=Path("/tmp"), modes=modes, sample_size=50
        )
    )
    adapter.prepare()
    return adapter, adapter.build_samples()


def _response(text: str) -> ModelResponse:
    return ModelResponse(sample_id="s", model_id="m", status=ResponseStatus.OK, content=text)


def _sample(samples, target):
    return next(s for s in samples if s.reference["target"] == target)


# --------------------------------------------------------------------------- #
# the target set (Table 2 and Table 3)
# --------------------------------------------------------------------------- #


def test_fifty_targets_ten_per_domain_with_the_papers_signatures():
    assert len(T.TARGETS) == 50
    assert len(T.BY_NAME) == 50, "two targets share a name"
    expected = {
        "Number": "(x: int) -> int",
        "Number Pairs": "(a: int, b: int) -> int",
        "String": "(text: str) -> str",
        "List": "(items: List[int]) -> int",
        "Logic": "(a: bool, b: bool) -> bool",
    }
    for domain, signature in expected.items():
        members = [t for t in T.TARGETS if t.domain == domain]
        assert len(members) == 10, domain
        assert {t.signature for t in members} == {signature}


def test_every_target_runs_over_its_whole_suite_and_returns_its_declared_type():
    returns = {"Number": int, "Number Pairs": int, "String": str, "List": int, "Logic": bool}
    for target in T.TARGETS:
        cases = T.test_cases(target)
        assert cases, target.name
        for _args, expected in cases:
            assert isinstance(expected, returns[target.domain]), target.name
            if target.domain != "Logic":
                # bool is an int subclass; an arithmetic target that returns one
                # would be graded against the wrong type by the sandbox.
                assert not isinstance(expected, bool), target.name


def test_suites_are_a_hundred_cases_except_where_the_input_space_is_smaller():
    for target in T.TARGETS:
        cases = T.test_cases(target)
        assert len({repr(args) for args, _ in cases}) == len(cases), target.name
        assert len(cases) == (4 if target.domain == "Logic" else 100), target.name


def test_suites_and_reveal_order_are_fixed_by_the_target_name_alone():
    """The paper's design needs one suite per target, the same in every run."""
    for target in (T.BY_NAME["times_four_mod_nine"], T.BY_NAME["letters_only"]):
        assert T.test_cases(target) == T.test_cases(target)
        assert T.reveal_order(target) == T.reveal_order(target)


def test_reveal_order_is_a_permutation_that_front_loads_edge_cases():
    target = T.BY_NAME["add_seven_if_negative"]
    order = T.reveal_order(target)
    assert sorted(order) == list(range(len(T.test_cases(target))))
    # The edge cases are the head of the pool; a random draw would put about a
    # quarter of them in the first ten, and the weighting exists to beat that.
    edges = len(T._EDGE_INTS)
    assert sum(1 for position in order[:10] if position < edges) >= 4


def test_every_ambiguous_reading_names_a_real_target():
    assert set(T.AMBIGUOUS) <= set(T.BY_NAME)


# --------------------------------------------------------------------------- #
# the sandbox
# --------------------------------------------------------------------------- #


def test_sandbox_grades_agreement_and_never_raises_on_a_bad_candidate():
    cases = [((x,), abs(x)) for x in (-3, 0, 5)]
    assert verify("def solution(x):\n    return abs(x)", cases).solved
    assert not verify("def solution(x):\n    return x", cases).solved
    assert verify("def solution(x)\n", cases).status == "error"
    assert verify("x = 1", cases).status == "no_function"
    assert verify("", cases).status == "no_function"
    # An exception per call is a wrong answer, not a harness failure.
    assert verify("def solution(x):\n    return 1 / 0", cases).status == "failed"


def test_sandbox_stops_a_loop_and_an_allocation_blowup():
    cases = [((1,), 1)]
    assert verify("def solution(x):\n    while True: pass", cases, timeout_s=2).status == "timeout"
    blown = verify("def solution(x):\n    return [0] * 10**12", cases, timeout_s=8)
    assert not blown.solved


def test_sandbox_keeps_bool_and_int_apart():
    """A Boolean answer must not be credited to an arithmetic target."""
    assert verify("def solution(x):\n    return True", [((0,), 1)]).status == "failed"
    assert verify("def solution(x):\n    return 1", [((0,), True)]).status == "failed"


def test_behaviour_digest_identifies_the_function_not_its_source():
    cases = [((x,), abs(x)) for x in (-3, 0, 5)]
    same = verify("def solution(x):\n    return abs(x)", cases).digest
    written_differently = verify("def solution(x):\n    return max(x, -x)", cases).digest
    wrong = verify("def solution(x):\n    return x", cases).digest
    assert same and same == written_differently
    assert same != wrong


def test_the_reference_source_of_every_target_solves_its_own_suite():
    """The suites are computed by running the targets, so this is a real check
    that what the episode reveals at the end is what the Game Master held."""
    for target in T.TARGETS:
        source = target.source.replace(f"def {target.name}(", "def solution(", 1)
        result = verify(source, T.test_cases(target))
        assert result.solved, f"{target.name}: {result.status} {result.error}"


# --------------------------------------------------------------------------- #
# code extraction
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        "```python\n" + ADD_SEVEN + "\n```",
        "SOLVE: ```python\n" + ADD_SEVEN + "\n```",
        "Answer:\n```\n" + ADD_SEVEN + "\n```",
        "Here it is.\nSOLVE:\n" + ADD_SEVEN,
    ],
)
def test_extract_code_reads_the_shapes_a_model_actually_writes(text):
    assert "return x + 7" in extract_code(text)


def test_extract_code_prefers_the_submitted_block_over_a_rejected_draft():
    text = "First try:\n```python\ndef solution(x):\n    return x\n```\nNo. \n```python\n" + ADD_SEVEN + "\n```"
    assert "return x + 7" in extract_code(text)


def test_extract_code_returns_nothing_for_prose():
    assert extract_code("I think it adds seven to negatives.") == ""
    assert extract_code("") == ""


# --------------------------------------------------------------------------- #
# the single-turn (static) mode
# --------------------------------------------------------------------------- #


def test_static_mode_shows_a_preloaded_batch_and_asks_for_nothing_else():
    adapter, samples = _adapter(AlienAbductionAdapter, "static")
    assert len(samples) == 50
    sample = _sample(samples, "add_seven_if_negative")
    _system, user = adapter.build_messages(sample)[0]
    assert "Signature: (x: int) -> int" in user.content
    assert len(sample.metadata["_revealed"]) == PRELOADED_EXAMPLES
    assert "no further examples" in user.content.lower()
    # Nothing in this prompt is the paper's protocol: there is no turn budget,
    # no TEST and no NEXT to offer in a mode that has one action.
    for absent in ("Turn budget", "TEST:", "NEXT:", "Game Master"):
        assert absent not in user.content


def test_static_mode_gets_both_prompt_modes_and_they_differ_by_the_mode_line():
    io_adapter, io_samples = _adapter(AlienAbductionAdapter, "static", "io")
    cot_adapter, cot_samples = _adapter(AlienAbductionAdapter, "static", "cot")
    io_text = io_adapter.build_messages(_sample(io_samples, "letters_only"))[0][1].content
    cot_text = cot_adapter.build_messages(_sample(cot_samples, "letters_only"))[0][1].content
    assert io_text != cot_text
    assert "Do not explain your reasoning" in io_text
    assert "step by step" in cot_text
    assert AlienAbductionAdapter.supports_modes(io_adapter.context.modes) is None
    assert AlienAbductionAdapter.supports_modes(cot_adapter.context.modes) is None


def test_static_mode_scores_the_papers_criterion():
    adapter, samples = _adapter(AlienAbductionAdapter, "static")
    sample = _sample(samples, "add_seven_if_negative")

    solved = adapter.score(sample, _response("Answer:\n```python\n" + ADD_SEVEN + "\n```"))
    assert solved.metrics["solved"] == 1.0
    assert solved.metrics["solved_number"] == 1.0
    assert solved.metrics["hypothesis_retrodiction"] == 1.0
    assert solved.prediction.startswith("behaviour:")

    near = adapter.score(sample, _response("```python\ndef solution(x):\n    return x\n```"))
    assert near.metrics["solved"] == 0.0
    # Agreement on most of the suite is not a partial solve.
    assert 0.0 < near.metrics["test_pass_rate"] < 1.0

    prose = adapter.score(sample, _response("It adds seven."))
    assert prose.parse_ok is False, "a static reply with no code is a parse failure"
    assert prose.metrics["committed"] == 0.0


# --------------------------------------------------------------------------- #
# the two multi-turn modes
# --------------------------------------------------------------------------- #


def _run_episode(adapter, sample, player):
    """The same loop the engine runs, so a protocol test exercises the real path."""
    messages, state = adapter.interactive_start(sample)
    turn, last = 0, ""
    replies = []
    for turn in range(1, adapter.max_turns + 1):
        last = player(turn)
        reply = adapter.interactive_step(sample, state, last)
        if reply is None:
            break
        replies.append(reply)
    sample.metadata["_episode_state"] = state
    sample.metadata["turns_used"] = turn
    return replies, state, _response(last)


def test_both_multi_turn_modes_run_one_prompt_set_on_the_papers_budget():
    for cls, delivery in (
        (AlienAbductionActiveAdapter, "interactive"),
        (AlienAbductionPassiveAdapter, "sequential"),
    ):
        assert cls.authors_prompt is True
        assert cls.max_turns == TURN_BUDGET
        adapter, _ = _adapter(cls, delivery)
        assert cls.supports_modes(adapter.context.modes) is None
        cot = TaskModes(prompt_mode="cot", selection_mode=None, data_delivery_mode=delivery)
        assert cls.supports_modes(cot) is not None, "quoting a protocol means one prompt mode"


def test_active_mode_answers_the_models_own_probes():
    adapter, samples = _adapter(AlienAbductionActiveAdapter, "interactive")
    sample = _sample(samples, "add_seven_if_negative")
    assert sample.metadata["_revealed"] == [], "the active mode reveals nothing up front"

    probes = ["TEST: -5", "TEST: 0", "TEST: not-an-integer", "TEST: 10"]
    replies, state, response = _run_episode(
        adapter,
        sample,
        lambda turn: probes[turn - 1] if turn <= len(probes)
        else "SOLVE: ```python\n" + ADD_SEVEN + "\n```",
    )
    assert replies[:2] == ["OUTPUT: 2", "OUTPUT: 0"]
    assert replies[2] == PARSE_ERROR_REPLY
    assert state["evidence_requests"] == 3 and state["parser_errors"] == 1

    score = adapter.score(sample, response)
    assert score.metrics["solved"] == 1.0
    assert score.metrics["parser_errors"] == 1.0
    assert score.metrics["hypothesis_retrodiction"] == 1.0


def test_active_mode_rejects_a_probe_the_targets_type_cannot_take():
    assert _parse_test_input("Number", "-8") == (-8,)
    assert _parse_test_input("Number", "'eight'") is None
    assert _parse_test_input("Number", "True") is None
    assert _parse_test_input("Number Pairs", "2, 1") == (2, 1)
    assert _parse_test_input("Number Pairs", "(2, 1)") == (2, 1)
    assert _parse_test_input("Number Pairs", "2") is None
    assert _parse_test_input("Logic", "True, False") == (True, False)
    assert _parse_test_input("Logic", "1, 0") is None
    assert _parse_test_input("List", "[1, 2, 3]") == ([1, 2, 3],)
    assert _parse_test_input("List", "1, 2, 3") == ([1, 2, 3],)
    assert _parse_test_input("String", "'ab'") == ("ab",)
    # An unquoted string is the commonest way to write this turn.
    assert _parse_test_input("String", "hello there") == ("hello there",)
    assert _parse_test_input("Number", "") is None


def test_passive_mode_reveals_one_pair_per_turn_in_the_static_modes_order():
    passive, passive_samples = _adapter(AlienAbductionPassiveAdapter, "sequential")
    static, static_samples = _adapter(AlienAbductionAdapter, "static")
    sample = _sample(passive_samples, "sum_mod_three")

    replies, state, response = _run_episode(
        passive,
        sample,
        lambda turn: "NEXT:" if turn <= 10 else "SOLVE: ```python\ndef solution(i):\n    return sum(i) % 3\n```",
    )
    assert len(replies) == 10 and all(r.startswith("OUTPUT: (") for r in replies)
    # The same evidence the single-turn mode is handed, arriving one at a time.
    assert [tuple(a) for a, _ in state["seen"]][:5] == [
        tuple(args) for args, _ in _sample(static_samples, "sum_mod_three").metadata["_revealed"]
    ][:5]
    assert passive.score(sample, response).metrics["solved"] == 1.0


def test_passive_mode_says_when_a_logic_targets_evidence_runs_out():
    adapter, samples = _adapter(AlienAbductionPassiveAdapter, "sequential")
    sample = _sample(samples, "nor_result")
    replies, state, _response_ = _run_episode(adapter, sample, lambda turn: "NEXT:")
    assert state["evidence_requests"] == 4, "the Logic input space is four pairs"
    assert "no further pairs remain" in replies[-1]


def test_an_episode_that_never_commits_is_a_result_not_a_parse_failure():
    adapter, samples = _adapter(AlienAbductionPassiveAdapter, "sequential")
    sample = _sample(samples, "count_odd_values")
    _replies, _state, response = _run_episode(adapter, sample, lambda turn: "NEXT:")
    score = adapter.score(sample, response)
    assert score.metrics["committed"] == 0.0
    assert score.metrics["solved"] == 0.0
    assert score.parse_ok is True, "not committing is one of the paper's own findings"


def test_a_solve_on_an_early_turn_is_what_gets_scored():
    """The model commits on turn 2 and says nothing after; the submission stands."""
    adapter, samples = _adapter(AlienAbductionActiveAdapter, "interactive")
    sample = _sample(samples, "add_seven_if_negative")
    _replies, state, response = _run_episode(
        adapter,
        sample,
        lambda turn: "TEST: -1" if turn == 1 else "SOLVE: ```python\n" + ADD_SEVEN + "\n```",
    )
    assert state["submitted"] is True
    assert adapter.score(sample, response).metrics["solved"] == 1.0


# --------------------------------------------------------------------------- #
# the property the paper's comparison rests on
# --------------------------------------------------------------------------- #


def test_all_three_modes_are_evaluated_on_identical_targets_and_suites():
    _static, static_samples = _adapter(AlienAbductionAdapter, "static")
    _active, active_samples = _adapter(AlienAbductionActiveAdapter, "interactive")
    _passive, passive_samples = _adapter(AlienAbductionPassiveAdapter, "sequential")

    by_target = [
        {s.reference["target"]: s for s in group}
        for group in (static_samples, active_samples, passive_samples)
    ]
    assert by_target[0].keys() == by_target[1].keys() == by_target[2].keys()
    for name in by_target[0]:
        suites = [group[name].metadata["_cases"] for group in by_target]
        assert suites[0] == suites[1] == suites[2], name
        assert by_target[0][name].reference["source"] == by_target[1][name].reference["source"]


def test_documentation_states_the_reconstruction_and_the_execution():
    for cls, delivery in (
        (AlienAbductionAdapter, "static"),
        (AlienAbductionActiveAdapter, "interactive"),
        (AlienAbductionPassiveAdapter, "sequential"),
    ):
        adapter, _ = _adapter(cls, delivery)
        doc = adapter.documentation()
        assert doc.primary_metric == "solved"
        blob = " ".join(doc.caveats).lower()
        assert "not released" in blob, "the reconstruction has to be stated"
        assert "executed" in blob and "security boundary" in blob
        assert doc.statistics["targets"] == 50
