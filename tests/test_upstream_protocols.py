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
import json
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


def test_medqdx_uses_the_papers_benchmarking_cap_not_the_construction_loop():
    """Five, not three -- and the difference is the whole point.

    The construction notebook runs ``for round_num in range(1, 4)``: three
    rounds, which produced the reference transcripts (Question_1..3) that the
    static "rounds" condition replays.

    Benchmarking a model under test is a different setting: "Interrogation is
    capped at five questions to standardize evaluation across models; if the
    correct diagnosis is not produced within this limit, the case is marked as
    a failure and assigned the maximum question count" (paper S3.B, MQD).

    The interactive task benchmarks a model, so the cap is five.
    """
    from abductionbench.adapters.medqdx import MedQDxAdapter

    assert MedQDxAdapter.category_limits == {"ask": 5}


def test_medqdx_stops_at_the_cap_and_then_asks_for_the_diagnosis():
    adapter, samples = _adapter("medqdx", MEDQDX)
    cap = adapter.category_limits["ask"]
    sample = samples[0]
    _messages, state = adapter.interactive_start(sample)
    replies = [
        _step(adapter, sample, state, f"Have you noticed symptom {i}?")
        for i in range(1, cap + 1)
    ]
    assert all(r is not None for r in replies)
    # Exactly `cap` questions are answered; the last reply asks for the
    # diagnosis rather than inviting one more.
    assert "Ask your next question" in replies[cap - 2]
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


def test_ddxplus_a_multi_choice_answer_reports_every_selected_value():
    """A multi-choice evidence is SEVERAL tokens sharing one code.

    The paper (§3.2): "we limit to 5 the maximum number of choices associated
    with multi-choice evidences such as pain location" -- so "Characterize your
    pain" can be answered sharp+burning+tugging, encoded as three
    ``E_54_@_V_*`` entries.

    The interactive answer sheet was a dict keyed by question text, so each
    token overwrote the last and the patient kept ONE value. Over 2,000 test
    patients, 79% had a multi-value evidence and 4.5 answers each were lost.
    """
    adapter, samples = _adapter(
        "ddxplus", "ddxplus:DDXPlusAdapter", sample_size=20,
    )
    multi = [
        (sample, question, value)
        for sample in samples
        for question, value in (sample.metadata.get("_answers") or {}).items()
        if ", " in value
    ]
    assert multi, "no multi-value evidence in 20 patients -- expected ~79%"

    sample, question, value = multi[0]
    values = [part.strip() for part in value.split(",")]
    assert len(values) >= 2
    assert len(values) <= 5, "the paper caps a multi-choice selection at 5"
    assert len(set(values)) == len(values), "a value is repeated"

    # And the interview actually reports them all, not just the last.
    _messages, state = adapter.interactive_start(sample)
    asked = question.replace('"', "")
    reply = _step(adapter, sample, state, json.dumps({"action": "ask", "query": asked}))
    for part in values:
        assert part in reply, f"the patient withheld {part!r} from {values}"


def test_vivabench_returns_negatives_not_a_shrug():
    """The paper, §3.2 "Information Retrieval and Parsing":

        "For history and physical examination findings, negative results
         (absent symptoms or normal examination findings) are explicitly
         returned when queried. ... investigations not available in the case
         are explicitly noted as 'not available' to prevent information
         leakage."

    Three different answers, not one. A flat "no finding recorded" collapses
    the distinction: asked about chest pain in a case that does not mention it,
    a candidate should learn the patient DOES NOT have chest pain -- that is
    evidence. Told only that nothing is recorded, they cannot tell a negative
    from a gap in the paperwork.
    """
    from abductionbench.adapters.vivabench import VivaBenchAdapter

    adapter, samples = _adapter("vivabench", VIVABENCH)
    sample = samples[0]
    _messages, state = adapter.interactive_start(sample)

    # A physical examination the case does not record comes back normal ...
    reply = _step(adapter, sample, state,
                  json.dumps({"action": "examination", "query": "cranial nerves ii to xii"}))
    assert "normal" in reply.lower()
    assert "no examination finding recorded" not in reply.lower()

    # ... and an investigation it does not hold is "not available", which is
    # the release's own wording and is deliberately NOT a normal value.
    reply = _step(adapter, sample, state,
                  json.dumps({"action": "imaging", "query": "PET-CT whole body"}))
    assert "not available" in reply.lower()

    assert set(VivaBenchAdapter._NOTHING_RECORDED) == {
        "history", "examination", "investigation", "imaging",
    }
    # History and physical are negatives; investigations are not-available.
    assert "not available" not in VivaBenchAdapter._NOTHING_RECORDED["history"].lower()
    assert "not available" in VivaBenchAdapter._NOTHING_RECORDED["investigation"].lower()


def test_med_inquire_reports_all_three_of_the_benchmarks_axes():
    """The paper, S3.6: "Med-Inquire evaluates an agent along three axes:
    diagnostic grade, interaction length, and resource cost." Table 1 reports
    all three per backbone. We reported only the grade.

    The cost schedule is the release's own ``default_cost_model()``: a base
    turn cost of 1.0, +0.5 for a question, +10.0 for an unlisted test, and a
    per-test table (cbc 5, cmp 6, bmp 4, ct head 120, mri brain 250,
    chest xray 25).
    """
    from abductionbench.core.types import ModelResponse, ResponseStatus

    adapter, samples = _adapter("med_inquire", MED_INQUIRE)
    sample = samples[0]
    _messages, state = adapter.interactive_start(sample)

    for action, query in (
        ("AskQuestion", "what brings you in"),   # 1.0 + 0.5
        ("OrderTest", "cbc"),                    # 1.0 + 5.0
        ("OrderTest", "mri brain"),              # 1.0 + 250.0
        ("SubmitDiagnosis", "sarcoidosis"),      # 1.0 + 0.0
    ):
        _step(adapter, sample, state,
              json.dumps({"action_type": action, "action_text": query}))

    assert state["encounter_cost"] == 259.5

    sample.metadata["_episode_state"] = state
    sample.metadata["turns_used"] = 4
    response = ModelResponse(
        sample_id=sample.sample_id, model_id="m", status=ResponseStatus.OK,
        content="sarcoidosis",
    )
    metrics = adapter.score_request(sample, response, output_contract=None).metrics
    assert metrics["encounter_cost"] == 259.5
    assert metrics["turns_used"] == 4.0
    assert "diagnosis_judged" in metrics


def test_med_inquire_an_unlisted_test_costs_the_default():
    """``self.test_costs.get(name.strip().lower(), default_test_cost)`` --
    an unlisted test costs 10.0, not a guess."""
    from abductionbench.adapters.med_inquire import MedInquireAdapter

    assert MedInquireAdapter._action_cost("ordertest", "CBC") == 6.0      # table, case-insensitive
    assert MedInquireAdapter._action_cost("ordertest", "skin biopsy") == 11.0   # default
    assert MedInquireAdapter._action_cost("askquestion", "anything") == 1.5
    assert MedInquireAdapter._action_cost("submitdiagnosis", "x") == 1.0


def test_cloud_opsbench_primary_metric_is_joint_rca_accuracy():
    """The paper, S4.1.1: "Our primary metric is Joint RCA Accuracy (JRA), the
    fraction of episodes for which R_j = R*_j, equivalently C_j = C*_j and
    F_j = F*_j."

    The outcome ground truth is a PAIR -- faulty component and fault type --
    and naming one without the other is not a diagnosis. We headlined
    `root_cause_judged`, which is the paper's Fault-Type Accuracy alone, so a
    run that located every component and named every mechanism wrongly scored
    the same as one that did the reverse.
    """
    from abductionbench.adapters.cloud_opsbench import CloudOpsBenchAdapter
    from abductionbench.core.types import SampleScore

    assert CloudOpsBenchAdapter.primary_metric == "joint_rca_accuracy"

    adapter = object.__new__(CloudOpsBenchAdapter)

    class _Verdict:
        score, positive, label, details = 1.0, True, "1", {}

    # Component right, fault type right -> JRA 1.
    both = adapter.apply_judge(
        None, None,
        SampleScore(metrics={"root_cause_judged": 0.0, "fault_object_match": 1.0,
                             "joint_rca_accuracy": 0.0}),
        _Verdict(),
    )
    assert both.metrics["joint_rca_accuracy"] == 1.0

    # Fault type right, component wrong -> JRA 0, even though FA is 1.
    half = adapter.apply_judge(
        None, None,
        SampleScore(metrics={"root_cause_judged": 0.0, "fault_object_match": 0.0,
                             "joint_rca_accuracy": 0.0}),
        _Verdict(),
    )
    assert half.metrics["root_cause_judged"] == 1.0
    assert half.metrics["joint_rca_accuracy"] == 0.0


def test_cloud_opsbench_a_cache_miss_is_unsupported_not_an_absence():
    """The paper, S3.2: "A supported query targeting a non-existent object
    returns the captured failure response, such as `Not Found`. A valid request
    outside the enumerated replay coverage returns `UnsupportedQuery` and is not
    interpreted as evidence about the system state."

    Two different answers. A `Not Found` is evidence -- the object genuinely is
    not there -- and lives in the recorded cache, so it arrives as a hit. A
    miss means the snapshot never recorded the call, which says nothing about
    the cluster. "No recorded output ... try a different call" invited the
    model to read a coverage gap as an absence.
    """
    adapter, samples = _adapter("cloud_opsbench", CLOUD_OPSBENCH)
    sample = samples[0]
    _messages, state = adapter.interactive_start(sample)
    reply = _step(
        adapter, sample, state,
        'Action: GetSourceCode\nAction Input: {"app_name": "nope", "file_path": "/nowhere"}',
    )
    assert "UnsupportedQuery" in reply
    assert "not evidence" in reply.lower()
    assert "no recorded output" not in reply.lower()


# --------------------------------------------------------------------------- #
# medups
# --------------------------------------------------------------------------- #

MEDUPS = "medups:MedUPSAdapter"


def _medups(sample_size=6):
    cls = resolve_adapter(f"abductionbench.adapters.{MEDUPS}")
    adapter = cls(
        AdapterContext(
            dataset_id="medups", data_dir=Path("data") / "medups",
            modes=TaskModes(prompt_mode="io", data_delivery_mode="sequential"),
            sample_size=sample_size, seed=20260903, offline=True,
        )
    )
    adapter.prepare()
    return adapter, adapter.build_samples()


def test_medups_asks_the_question_that_was_asked():
    """The task is the NEXT CLINICAL STEP, not the diagnosis.

    MedUPSQA is 21,874 mid-stream decision points, and the paper is explicit
    that "the accompanying free-text final diagnosis is not used anywhere in
    this work" (S3.2). Its own examples span Diagnosis, Management, Workup and
    Pathology (Table 2), and the release ships no question-type column, so the
    questions arrive mixed.

    This adapter closed every prompt with "Answer: <a single diagnosis>",
    underneath questions like "What will be the expected radiographic findings
    to confirm no recurrence of infection?" -- and then judged the reply
    against a gold describing radiographs.
    """
    from abductionbench.adapters.medups import MedUPSAdapter

    assert MedUPSAdapter.answer_format == "the answer to the question asked"
    assert "diagnosis" not in MedUPSAdapter.answer_format

    adapter, samples = _medups()
    messages, _contract = adapter.build_messages(samples[0])
    prompt = messages[-1].content
    assert "Answer: <the answer to the question asked>" in prompt
    assert "<a single diagnosis>" not in prompt
    # The question itself is in the prompt, or there is nothing to answer.
    assert samples[0].fields["question"][:40] in prompt


def test_medups_metric_is_not_named_for_a_task_it_does_not_run():
    """`diagnosis_judged` invited comparison with every other medical
    dataset's diagnosis accuracy. Only one of the four question kinds is a
    diagnosis."""
    from abductionbench.adapters.medups import MedUPSAdapter

    assert MedUPSAdapter.primary_metric == "next_step_judged"
    assert MedUPSAdapter.primary_metric_by_mode["generation"] == "next_step_judged"


def test_medups_judge_is_told_what_was_asked():
    """Grading disease identity is wrong for a question about the next test."""
    from abductionbench.core.types import ModelResponse, ResponseStatus, SampleScore

    adapter, samples = _medups()
    sample = samples[0]
    response = ModelResponse(
        sample_id=sample.sample_id, model_id="m", status=ResponseStatus.OK,
        content="Answer: repeat radiography showing no lucency",
    )
    request = adapter.judge_request(
        sample, response, SampleScore(metrics={}, prediction="repeat radiography"),
    )
    assert sample.fields["question"][:40] in request["context"]
    assert "same clinical step" in request["criteria"]
    assert "not answered by a diagnosis" in request["criteria"]


def test_medups_pool_is_stratified_along_the_trajectory():
    """The paper's pool is "stratified by the number of context chunks
    available at prediction time (1 through 8) so that positions along the
    trajectory are covered" (S4).

    A plain shuffle draws positions in proportion to how common they are, and
    this split has a long tail, so a small draw could contain no early decision
    point at all -- the hardest case, where almost nothing has been revealed.
    """
    from collections import Counter

    adapter, samples = _medups(sample_size=40)
    positions = Counter(s.metadata["answer_chunk"] - 1 for s in samples)
    assert set(positions) == set(range(1, 9)), f"positions covered: {sorted(positions)}"
    # Balanced, not merely present.
    assert max(positions.values()) - min(positions.values()) <= 1

    # And nothing beyond position 8, which the paper's pool excludes.
    adapter, many = _medups(sample_size=200)
    assert all(s.metadata["answer_chunk"] - 1 <= 8 for s in many)
