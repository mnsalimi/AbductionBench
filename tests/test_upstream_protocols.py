"""Interactive protocols, checked against the authors' own implementations.

Each assertion here cites the upstream file it comes from. The point is that a
number like "12 turns" is not a design choice this suite gets to make: it is a
property of the benchmark, and getting it wrong measures something the paper
did not.

Upstream sources are vendored under ``data/<dataset>/repo`` where the release
ships code.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from abductionbench.core.adapter import AdapterContext
from abductionbench.core.modes import TaskModes
from abductionbench.core.registry import resolve_adapter
from abductionbench.core.types import ModelResponse, ResponseStatus


def _adapter(dataset_id: str, impl: str, **options):
    cls = resolve_adapter(f"abductionbench.adapters.{impl}")
    adapter = cls(
        AdapterContext(
            dataset_id=dataset_id,
            data_dir=Path("data") / dataset_id,
            modes=TaskModes(prompt_mode="io", data_delivery_mode="interactive"),
            sample_size=3,
            offline=True,
            options=options,
        )
    )
    adapter.prepare()
    return adapter, adapter.build_samples()


def _step(adapter, sample, state, text):
    reply = adapter.interactive_step(sample, state, text)
    if asyncio.iscoroutine(reply):
        return asyncio.run(reply)
    return reply


# --------------------------------------------------------------------------- #
# med_inquire / EvoClinician
# --------------------------------------------------------------------------- #

MED_INQUIRE = "med_inquire:MedInquireAdapter"


def test_med_inquire_budget_is_the_releases_twenty_turns():
    """``EpisodeConfig.max_turns = 20`` -- evoclinician/med_inquire/types.py.

    And no per-kind cap: ``MedInquireEnv.run_episode`` counts every action
    against that one budget. The "at most 8 questions and 6 tests" this adapter
    enforced, and stated in its prompt, appears nowhere upstream.
    """
    from abductionbench.adapters.med_inquire import MedInquireAdapter

    assert MedInquireAdapter.max_turns == 20
    assert MedInquireAdapter.category_limits == {}


def test_med_inquire_tells_the_agent_how_many_turns_are_left():
    """``Actor.decide`` puts "Turns remaining before forced stop: {n}" in the
    user message on every turn. An agent that cannot see the clock cannot
    choose to commit before it runs out."""
    adapter, samples = _adapter("med_inquire", MED_INQUIRE)
    sample = samples[0]
    _messages, state = adapter.interactive_start(sample)
    reply = _step(
        adapter, sample, state,
        '{"action_type": "AskQuestion", "action_text": "What brings you in?"}',
    )
    assert "Turns remaining before forced stop:" in reply
    assert "19" in reply, "one action spent of twenty"

    for _ in range(3):
        reply = _step(
            adapter, sample, state,
            '{"action_type": "AskQuestion", "action_text": "Any fever?"}',
        )
    assert "Turns remaining before forced stop: 16" in reply


def test_med_inquire_forced_diagnosis_is_the_last_actions_content():
    """``final_diagnosis = history[-1].action.content`` on exhaustion.

    The content, not the JSON envelope. An episode that timed out mid-work-up
    used to hand the judge the whole OrderTest object and ask whether it named
    the disease.
    """
    adapter, samples = _adapter("med_inquire", MED_INQUIRE)
    sample = samples[0]
    _messages, state = adapter.interactive_start(sample)
    last = '{"action_type": "OrderTest", "action_text": "chest x-ray"}'
    _step(adapter, sample, state, last)

    # The engine hands the scorer the episode state under this key.
    sample.metadata["_episode_state"] = state
    response = ModelResponse(
        sample_id=sample.sample_id, model_id="m", status=ResponseStatus.OK, content=last,
    )
    score = adapter.score_request(sample, response, output_contract=None)
    assert score.prediction == "chest x-ray", (
        f"the judge would be handed {score.prediction!r} as a diagnosis"
    )
    assert "action_type" not in (score.prediction or "")
    assert score.details.get("forced_diagnosis")


def test_med_inquire_a_real_submission_is_not_overwritten():
    """Only an exhausted episode gets the forced diagnosis."""
    adapter, samples = _adapter("med_inquire", MED_INQUIRE)
    sample = samples[0]
    _messages, state = adapter.interactive_start(sample)
    _step(adapter, sample, state,
          '{"action_type": "AskQuestion", "action_text": "What brings you in?"}')
    submission = '{"action_type": "SubmitDiagnosis", "action_text": "sarcoidosis"}'
    assert _step(adapter, sample, state, submission) is None, "submitting ends the episode"
    assert state.get("submitted") is True

    sample.metadata["_episode_state"] = state
    response = ModelResponse(
        sample_id=sample.sample_id, model_id="m", status=ResponseStatus.OK,
        content=submission,
    )
    score = adapter.score_request(sample, response, output_contract=None)
    assert "forced_diagnosis" not in score.details


def test_med_inquire_does_not_cap_actions_by_kind():
    """Twenty questions in a row is a legitimate strategy upstream."""
    adapter, samples = _adapter("med_inquire", MED_INQUIRE)
    sample = samples[0]
    _messages, state = adapter.interactive_start(sample)
    for _ in range(12):
        reply = _step(
            adapter, sample, state,
            '{"action_type": "AskQuestion", "action_text": "Any fever?"}',
        )
        assert reply is not None
        assert "No further actions of that kind" not in reply
