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


def _adapter(dataset_id: str, impl: str, sample_size: int = 3, **options):
    cls = resolve_adapter(f"abductionbench.adapters.{impl}")
    adapter = cls(
        AdapterContext(
            dataset_id=dataset_id,
            data_dir=Path("data") / dataset_id,
            modes=TaskModes(prompt_mode="io", data_delivery_mode="interactive"),
            sample_size=sample_size,
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


# --------------------------------------------------------------------------- #
# vivabench
# --------------------------------------------------------------------------- #

VIVABENCH = "vivabench:VivaBenchAdapter"


def test_vivabench_limits_are_the_releases_own_config():
    """configs/evaluate.yaml: hx 10, phys 5, ix 5, img 5, action_limit 20."""
    from abductionbench.adapters.vivabench import VivaBenchAdapter

    assert VivaBenchAdapter.category_limits == {
        "history": 10, "examination": 5, "investigation": 5, "imaging": 5,
    }
    assert VivaBenchAdapter.max_turns == 20
    assert VivaBenchAdapter._RETRY_LIMIT == 2  # RETRY_LIMIT in examiner.py


def test_vivabench_category_limits_are_advice_not_a_wall():
    """``Examiner.process_history`` and its siblings always answer.

        self.hx_count += 1
        if self.hx_count >= self.hx_limit:
            _prompt += "\\nLimit on history-taking reached. ..."
        return _prompt

    The findings are returned either way; the notice is appended. Nothing
    upstream refuses a request. Only ``action_limit`` is a hard cap, and it
    bounds the episode rather than a category. This adapter used to answer ten
    history requests and then refuse the eleventh.
    """
    adapter, samples = _adapter("vivabench", VIVABENCH)
    sample = samples[0]
    _messages, state = adapter.interactive_start(sample)

    replies = []
    for _ in range(13):  # past the history limit of 10
        replies.append(_step(
            adapter, sample, state,
            '{"action": "history", "query": "tell me about the pain"}',
        ))

    assert all(r is not None for r in replies), "the episode ended early"
    # The notice appears from the tenth onward ...
    assert "Limit on history-taking reached" in replies[9]
    assert "Limit on history-taking reached" in replies[12]
    # ... but the eleventh and beyond are still answered, not walled off.
    for index in (10, 11, 12):
        body = replies[index].split("Limit on history-taking reached")[0].strip()
        assert body, f"request {index + 1} was refused rather than answered"


def test_vivabench_an_uncommitted_episode_is_not_judged_on_its_prose():
    """``conduct_examination`` returns only on ``diagnosis_final``; running out
    raises TimeoutError. There is no diagnosis, so none is judged.

    We score it 0 with committed=0 rather than raising -- excluding the
    failures would inflate the mean -- but the judge is never shown the last
    turn's text, because reasoning aloud about the right condition without
    committing is not a diagnosis.
    """
    from abductionbench.core.types import ModelResponse, ResponseStatus

    adapter, samples = _adapter("vivabench", VIVABENCH)
    sample = samples[0]
    response = ModelResponse(
        sample_id=sample.sample_id, model_id="m", status=ResponseStatus.OK,
        content='{"action": "history", "query": "I think this is probably sarcoidosis"}',
    )
    score = adapter.score_request(sample, response, output_contract=None)
    assert score.metrics["committed"] == 0.0
    assert score.metrics["diagnosis_judged"] == 0.0
    assert score.prediction is None, "an uncommitted episode reached the judge"


# --------------------------------------------------------------------------- #
# medqdx
# --------------------------------------------------------------------------- #

MEDQDX = "medqdx:MedQDxAdapter"


def test_medqdx_asks_the_releases_three_questions():
    """``for round_num in range(1, 4)`` -- MedQDx_Benchmark_Creation.ipynb.

    And the evaluation notebook reports Similarity_1, Similarity_2,
    Similarity_3, one per round. Eight was this adapter's own number.
    """
    from abductionbench.adapters.medqdx import MedQDxAdapter

    assert MedQDxAdapter.category_limits == {"ask": 3}


def test_medqdx_stops_after_three_and_then_asks_for_the_diagnosis():
    adapter, samples = _adapter("medqdx", MEDQDX)
    sample = samples[0]
    _messages, state = adapter.interactive_start(sample)
    replies = [
        _step(adapter, sample, state, f"Have you noticed symptom {i}?")
        for i in range(1, 4)
    ]
    assert all(r is not None for r in replies)
    # The third answer is followed by the request for a diagnosis, not a fourth
    # question.
    assert "Name the condition" in replies[-1]
    assert _step(adapter, sample, state, "Pneumonia") is None


def test_medqdx_patient_reads_the_full_case_not_a_symptom_list():
    """``build_patient_answer_prompt(Full_case, doctor_question)``.

    The patient gets the FULL case, not the partial vignette the doctor was
    shown. That asymmetry is the point of the interview: at the 50% level the
    doctor is missing things the patient can answer. Briefing the patient with
    the symptom list alone -- as this adapter did -- is strictly less: it can
    confirm or deny a symptom and can say nothing about onset or context.
    """
    adapter, samples = _adapter("medqdx", MEDQDX, sample_size=30)
    by_level = {}
    for sample in samples:
        by_level.setdefault(sample.metadata.get("condition"), sample)

    assert "50pct" in by_level, "no 50% sample to compare against"
    sample = by_level["50pct"]
    brief = adapter._patient_brief(sample)
    doctor_sees = sample.fields["observation"]

    assert len(sample.metadata["_full_case"]) > len(doctor_sees), (
        "the patient knows no more than the doctor, so the interview is empty"
    )
    assert sample.metadata["_full_case"][:60] in brief
    # The release's own wording, and its own no-invention rule.
    assert "Do not add, remove, or invent any details" in brief
    assert 'reply honestly: "No," or "I have not noticed that."' in brief
    # And still never the answer.
    assert str(sample.reference["gold"]).lower() not in brief.lower()


def test_medqdx_unrecorded_details_get_the_releases_answer():
    """The patient says "No," / "I have not noticed that", not "I'm not sure".

    The doctor-side prompt does mention "I'm not sure", which is an
    inconsistency upstream; the patient-side wording is what produced the
    recorded answers, so it is what the environment says.
    """
    adapter, samples = _adapter("medqdx", MEDQDX)
    sample = samples[0]
    _messages, state = adapter.interactive_start(sample)
    reply = _step(adapter, sample, state, "Have you ever been to Antarctica?")
    patient_line = reply.splitlines()[0]
    assert "I have not noticed that" in patient_line
    assert "I'm not sure" not in patient_line
    # The DOCTOR's rules do still mention it, because the release's
    # doctor-side prompt does: "If the patient responded \"I'm not sure,\"
    # ask a broader or differently phrased question". Only the patient's own
    # wording changed.
    assert "I'm not sure" in reply


# --------------------------------------------------------------------------- #
# cloud_opsbench
# --------------------------------------------------------------------------- #

CLOUD_OPSBENCH = "cloud_opsbench:CloudOpsBenchAdapter"


def test_cloud_opsbench_budget_is_the_releases_twenty_steps():
    """``diagnosis.max_iterations: 20`` -- cloudops_agent/configs/
    model_configs.yaml, enforced by AgentRuntime.run as
    ``while not state.finished and state.current_step < state.max_steps``.

    And no separate tool budget: every ReAct step counts against max_steps and
    tool calls are capped nowhere. "At most 12 tool calls" was invented here,
    and was in the prompt.
    """
    from abductionbench.adapters.cloud_opsbench import CloudOpsBenchAdapter

    assert CloudOpsBenchAdapter.max_turns == 20
    assert CloudOpsBenchAdapter.category_limits == {}


def test_cloud_opsbench_shows_the_step_counter_every_turn():
    """``PromptBuilder._build_case_section``:

        Current Step: {state.current_step + 1}
        Budget Steps: {state.max_steps}
    """
    adapter, samples = _adapter("cloud_opsbench", CLOUD_OPSBENCH)
    sample = samples[0]
    _messages, state = adapter.interactive_start(sample)
    # The reply to step N is the context for step N+1, so it announces N+1 --
    # which is what `current_step + 1` is in the prompt built for that step.
    reply = _step(adapter, sample, state, "Action: GetResources\nAction Input: {}")
    assert "Current Step: 2" in reply
    assert "Budget Steps: 20" in reply

    reply = _step(adapter, sample, state, "Action: GetResources\nAction Input: {}")
    assert "Current Step: 3" in reply


def test_cloud_opsbench_forces_the_answer_on_the_last_allowed_step():
    """``PromptBuilder._build_current_step_instruction`` replaces the protocol
    on the last step with "This is the final allowed step. You MUST now stop
    calling tools and output the final diagnosis...".

    This adapter said nothing and simply stopped answering, so an agent that
    ran out was never told to commit.
    """
    adapter, samples = _adapter("cloud_opsbench", CLOUD_OPSBENCH)
    sample = samples[0]
    _messages, state = adapter.interactive_start(sample)
    replies = []
    for _ in range(adapter.max_turns):
        reply = _step(adapter, sample, state, "Action: GetResources\nAction Input: {}")
        assert reply is not None, "the environment stopped answering before the budget"
        replies.append(reply)
    # Not before the last step is in sight ...
    assert "final allowed step" not in replies[adapter.max_turns - 3]
    # ... and on the turn whose next step is the last one.
    assert "final allowed step" in replies[adapter.max_turns - 2]
    assert "output the final diagnosis" in replies[adapter.max_turns - 2]


def test_cloud_opsbench_does_not_cap_tool_calls_separately():
    """Twenty tool calls in a row is within the benchmark's budget."""
    adapter, samples = _adapter("cloud_opsbench", CLOUD_OPSBENCH)
    sample = samples[0]
    _messages, state = adapter.interactive_start(sample)
    for index in range(15):
        reply = _step(adapter, sample, state, "Action: GetResources\nAction Input: {}")
        assert reply is not None
        assert "budget exhausted" not in reply.lower(), f"refused at call {index + 1}"


# --------------------------------------------------------------------------- #
# ddxplus -- a dataset release, not an agent harness
# --------------------------------------------------------------------------- #


def test_ddxplus_uses_the_releases_own_english_fields():
    """There is no upstream interaction code to follow, so what can be checked
    is the data handling: the questions are the release's ``question_en`` and a
    categorical answer is its ``value_meaning[...]["en"]``, not a raw V-code.

    (mila-iqia/ddxplus ships a README, a dialogue PDF and the JSON banks. Its
    published baselines are supervised models, not a prompted agent.)
    """
    import json

    from abductionbench.adapters.ddxplus import DDXPlusAdapter

    evidences = json.loads(Path("data/ddxplus/release_evidences.json").read_text())
    adapter, _samples = _adapter("ddxplus", "ddxplus:DDXPlusAdapter")
    adapter._evidences = evidences  # the bank, as loaded from the release

    # A categorical evidence whose values are V-codes with English meanings.
    code, entry = next(
        (c, e) for c, e in evidences.items()
        if e.get("data_type") == "C" and (e.get("value_meaning") or {})
    )
    value = [v for v in entry["possible-values"] if v != entry.get("default_value")][0]
    rendered = adapter._decode(f"{code}_@_{value}")

    assert entry["question_en"] in rendered, "the question is not the release's English one"
    meaning = entry["value_meaning"][value]["en"]
    assert meaning in rendered, f"the answer is not the release's English meaning ({meaning!r})"
    assert value not in rendered, "a raw V-code reached the model"
    assert isinstance(DDXPlusAdapter.max_turns, int)


def test_ddxplus_numeric_scales_are_rendered_with_their_scale():
    """Six evidences have no ``value_meaning`` and integer possible-values 0-10.

    "How intense is the pain? -> 7" is not an answer anyone can use: seven out
    of what? The range is in the release's own ``possible-values``, so it is
    stated rather than left to be inferred.
    """
    import json

    evidences = json.loads(Path("data/ddxplus/release_evidences.json").read_text())
    adapter, _samples = _adapter("ddxplus", "ddxplus:DDXPlusAdapter")
    adapter._evidences = evidences

    scaled = [
        (c, e) for c, e in evidences.items()
        if e.get("data_type") in ("C", "M") and not (e.get("value_meaning") or {})
    ]
    assert len(scaled) == 6, "the release's numeric evidences changed"
    code, entry = scaled[0]
    rendered = adapter._decode(f"{code}_@_7")
    assert entry["question_en"] in rendered
    assert "7" in rendered
    assert "scale of 0-10" in rendered, f"a bare, uninterpretable number: {rendered!r}"
