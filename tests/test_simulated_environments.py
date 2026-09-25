"""The environment played by a model: what it may see, and what it may break.

Putting an LLM behind the patient is what the three interview papers do, and
it introduces two failure modes a lexical matcher could not have:

* **Leakage.** The simulator holds the case, and the case holds the answer. If
  the diagnosis reaches the simulator's brief, or the brief reaches the
  evaluated model's conversation, the benchmark measures nothing.
* **A wrong answer that was never the model's.** A simulator that times out
  leaves the interview half-finished. Scoring whatever the model last said
  would charge an API outage to the model under evaluation.

Everything here is about one of those two, plus the audit trail that makes
either of them provable after the fact.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from conftest import step

from abductionbench.adapters._interactive import EvidenceStore
from abductionbench.core.adapter import AdapterContext
from abductionbench.core.client import BatchResult, RawChoice
from abductionbench.core.config import SimulatorConfig, TimeoutConfig
from abductionbench.core.modes import TaskModes
from abductionbench.core.registry import resolve_adapter
from abductionbench.core.simulator import (
    SIMULATOR_LOG_FILENAME,
    EnvironmentSimulator,
    SimulatorPool,
)
from abductionbench.core.types import TaskIdentity

INTERVIEW_ADAPTERS = [
    ("medqdx", "medqdx:MedQDxAdapter"),
    ("med_inquire", "med_inquire:MedInquireAdapter"),
    ("vivabench", "vivabench:VivaBenchAdapter"),
]

IDENTITY = TaskIdentity(
    run_id="testrun",
    dataset_id="medqdx",
    model_id="fake-model",
    template_id="io_SCS_interactive",
    template_version="1",
)


# --------------------------------------------------------------------------- #
# doubles
# --------------------------------------------------------------------------- #


class _StubClient:
    """A ModelClient's one method, scripted.

    Records every request so a test can read back exactly what the simulator
    was told -- which is the only way to check the information boundary from
    the outside.
    """

    def __init__(self, replies, *, reasoning=None, fail_times=0, model="stub/model"):
        self.replies = list(replies)
        self.reasoning = reasoning
        self.fail_times = fail_times
        self.model = model
        self.requests: list[list[dict]] = []
        self.sampling: list = []

    async def chat_single(self, messages, sampling):
        self.requests.append([m.to_dict() for m in messages])
        self.sampling.append(sampling)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise TimeoutError("the simulator did not respond")
        reply = self.replies.pop(0) if self.replies else ""
        return BatchResult(
            choices=[RawChoice(index=0, content=reply, reasoning=self.reasoning,
                               finish_reason="stop")],
            usage={}, response_id="r1", model=self.model, latency_s=0.01,
            endpoint_url="stub", is_batch=False,
        )


def _simulator(replies, *, run_dir=None, reasoning=None, fail_times=0, **overrides):
    config = SimulatorConfig(
        enabled=True, base_url="https://stub/api", model="stub/model",
        api_key="secret-key-do-not-log", **overrides,
    )
    client = _StubClient(replies, reasoning=reasoning, fail_times=fail_times)
    return EnvironmentSimulator(
        config=config, client=client, dataset_id="medqdx", run_dir=run_dir
    ), client


def _adapter(dataset_id, impl, *, simulator=None, prompt_mode="io"):
    cls = resolve_adapter(f"abductionbench.adapters.{impl}")
    adapter = cls(
        AdapterContext(
            dataset_id=dataset_id,
            data_dir=Path("data") / dataset_id,
            modes=TaskModes(prompt_mode=prompt_mode, data_delivery_mode="interactive"),
            sample_size=3,
            offline=True,
            simulator=simulator,
        )
    )
    adapter.prepare()
    return adapter, adapter.build_samples()


# --------------------------------------------------------------------------- #
# the information boundary
# --------------------------------------------------------------------------- #


async def _drive(adapter, sample, state, turns=3):
    """Ask the environment a few plausible things, and return what it said."""
    asks = {
        "medqdx": ["Do you have a fever?", "How long has this gone on?", "Any cough?"],
        "med_inquire": [
            '{"action_type": "AskQuestion", "action_text": "What brings you in?"}',
            '{"action_type": "OrderTest", "action_text": "full blood count"}',
            '{"action_type": "AskQuestion", "action_text": "Any fever?"}',
        ],
        "vivabench": [
            '{"action": "history", "query": "What brings you in?"}',
            '{"action": "examination", "query": "abdomen"}',
            '{"action": "investigation", "query": "full blood count"}',
        ],
    }[adapter.dataset_id]
    spoken: list[str] = []
    for text in asks[:turns]:
        reply = adapter.interactive_step(sample, state, text)
        if asyncio.iscoroutine(reply):
            reply = await reply
        if reply is None:
            break
        spoken.append(reply)
    return spoken


def test_the_medqdx_patient_is_never_told_the_diagnosis():
    """MedQDx grades a condition name, and the patient is a symptom list.

    So this one is absolute: the label may not appear in the brief at all. It
    is rendered from real samples rather than read off the template, because
    the leak that would matter is the one a particular case introduces.
    """
    simulator, _client = _simulator(["I have a cough."] * 40)
    adapter, samples = _adapter("medqdx", "medqdx:MedQDxAdapter", simulator=simulator)
    checked = 0
    for sample in samples[:3]:
        gold = str(sample.reference.get("gold") or "").strip()
        if len(gold) < 4:
            continue
        brief = adapter._patient_brief(sample)
        assert gold.lower() not in brief.lower(), (
            f"the gold answer {gold!r} reached the patient's brief"
        )
        # And the patient is told not to offer one -- in the release's own
        # words now, which are "Do not volunteer any additional background,
        # diagnosis, or speculation."
        assert "diagnosis, or speculation" in brief.lower()
        assert "do not add, remove, or invent any details" in brief.lower()
        checked += 1
    assert checked, "no sample had a gold label to check against"


def test_the_med_inquire_briefs_exclude_every_answer_field():
    """The case carries the answer four ways; none of them may be handed over.

    ``final_diagnosis`` is the graded label and ``option_a..d`` /
    ``right_option`` are the release's multiple-choice form of it. A patient
    who has read any of them is not a patient.
    """
    simulator, _client = _simulator(["Nothing to add."] * 10)
    adapter, samples = _adapter("med_inquire", "med_inquire:MedInquireAdapter",
                                simulator=simulator)
    for sample in samples[:3]:
        case = sample.metadata.get("_case") or {}
        briefs = adapter._briefs(sample)
        for field in ("final_diagnosis", "option_a", "option_b", "option_c",
                      "option_d", "right_option"):
            value = str(case.get(field) or "").strip()
            if len(value) < 4:
                continue
            for role, brief in briefs.items():
                assert value.lower() not in brief.lower(), (
                    f"the case's {field} reached the {role} brief"
                )
    # The patient knows their history and nothing about the work-up.
    briefs = adapter._briefs(samples[0])
    exam = str((samples[0].metadata.get("_case") or {}).get("physical_examination") or "")
    if len(exam) > 40:
        assert exam[:40].lower() not in briefs["askquestion"].lower()


def test_the_vivabench_examiner_is_never_given_the_diagnosis_sections():
    """The examiner holds the case -- but not the case's answer.

    VivaBench's structured case records ``diagnosis`` and ``differentials``
    alongside the findings, and neither is something a candidate could order.
    Those two sections are simply not among the categories an action can
    disclose, and that is what is checked here.

    What is *not* claimed is that the diagnosis name never appears anywhere in
    a brief: a confirmatory investigation ("molecular testing ... consistent
    with pheochromocytoma") is part of the recorded work-up, and ordering it is
    how a viva candidate confirms an answer. The deterministic environment
    discloses the same finding to the same request -- this is the benchmark,
    not something the simulator introduced.
    """
    from abductionbench.adapters.vivabench import VivaBenchAdapter

    for keys in VivaBenchAdapter._SOURCES.values():
        assert "diagnosis" not in keys and "differentials" not in keys

    adapter, samples = _adapter("vivabench", "vivabench:VivaBenchAdapter")
    payload = samples[0].metadata.get("_case") or {}
    assert "diagnosis" in payload, "this case has no diagnosis section to exclude"
    store = adapter._evidence({"clinicalcase": json.dumps(payload)})
    keys = [key for category in store.categories.values() for key in category]
    assert keys, "the case disclosed nothing at all"
    assert not [key for key in keys if key.startswith(("diagnosis", "differentials"))]


@pytest.mark.parametrize(("dataset_id", "impl"), INTERVIEW_ADAPTERS)
def test_the_environment_says_only_what_the_simulator_gave_it(dataset_id, impl):
    """No turn may carry case material the simulator did not disclose.

    The stub answers every request with one fixed sentence, so anything else
    that turns up in an environment reply came out of the hidden brief by a
    route that is not the protocol -- which is the leak this guards.
    """
    canned = "I feel unwell, and that is all I can tell you."
    simulator, client = _simulator([canned] * 40)
    adapter, samples = _adapter(dataset_id, impl, simulator=simulator)
    sample = samples[0]
    _messages, state = adapter.interactive_start(sample)
    state["_identity"] = IDENTITY
    state["_sample_id"] = sample.sample_id
    replies = asyncio.run(_drive(adapter, sample, state))
    assert replies, f"{dataset_id}: the environment never answered"

    briefs = [req[0]["content"] for req in client.requests if req]
    spoken = " ".join(replies)
    for brief in briefs:
        for line in brief.splitlines():
            line = line.strip().lstrip("- ").strip()
            # Rules and headings are boilerplate; case material is the long,
            # content-bearing lines.
            if len(line) < 40 or line.endswith(":"):
                continue
            if line.lower().startswith(("rules you must", "you are", "reply with",
                                        "the doctor names", "your history",
                                        "what you are")):
                continue
            assert line not in spoken, (
                f"{dataset_id}: hidden case material reached the model unasked: "
                f"{line[:80]!r}"
            )


def test_med_inquire_keeps_the_patient_and_the_examiner_apart():
    """Two agents, two conversations: the patient must not see the work-up.

    The release runs a Patient and an Examination agent. Sharing one transcript
    between them would let a history question be answered out of the lab
    results -- evidence the doctor never ordered.
    """
    simulator, client = _simulator(["Nothing to add."] * 10)
    adapter, samples = _adapter("med_inquire", "med_inquire:MedInquireAdapter",
                                simulator=simulator)
    sample = samples[0]
    _messages, state = adapter.interactive_start(sample)
    asyncio.run(_drive(adapter, sample, state))

    roles = state["_sim_turns"]
    assert set(roles) == {"askquestion", "ordertest"}, roles
    patient_brief = adapter._briefs(sample)["askquestion"]
    examiner_brief = adapter._briefs(sample)["ordertest"]
    assert patient_brief != examiner_brief
    # The patient's conversation holds only the patient's turns.
    patient_texts = " ".join(m.content for m in roles["askquestion"])
    assert "full blood count" not in patient_texts.lower()


# --------------------------------------------------------------------------- #
# parsing: a simulator may choose what is disclosed, never what it says
# --------------------------------------------------------------------------- #


def test_the_vivabench_mapper_cannot_invent_a_finding():
    """The mapper picks keys; the finding is still rendered from the case.

    This is the release's own split (``LLMMapper`` resolves the request, the
    Examiner reads the record back), and it is also the tighter boundary: a
    simulator that tries to write the reply has no channel to do it through.
    """
    store = EvidenceStore()
    store.categories["investigation"] = {
        "investigations.k.name": "Potassium",
        "investigations.k.value": "2.9",
        "investigations.k.units": "mmol/L",
    }
    # A mapper that names a key AND tries to dictate the answer.
    rendered = store.reveal_keys(
        "investigation",
        ["investigations.k.value", "the patient has renal failure"],
    )
    assert "2.9" in rendered and "Potassium" in rendered
    assert "renal failure" not in rendered


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('["a.b"]', ["a.b"]),
        ('```json\n["a.b", "nope"]\n```', ["a.b"]),
        ("[]", []),
        ("I would look at a.b here", ["a.b"]),
        ("complete nonsense", []),
        ('["c", "a.b"]', ["c", "a.b"]),
    ],
)
def test_mapper_replies_are_parsed_or_discarded(raw, expected):
    """Unreadable mapper output discloses nothing rather than guessing.

    Falling back to the lexical matcher here would hide a mapper that had
    stopped working behind plausible-looking answers.
    """
    from abductionbench.adapters.vivabench import VivaBenchAdapter

    assert VivaBenchAdapter._parse_keys(raw, {"a.b", "c"}) == expected


# --------------------------------------------------------------------------- #
# failure handling
# --------------------------------------------------------------------------- #


def test_a_failed_call_is_reported_as_a_failure_not_as_a_shrug():
    simulator, client = _simulator([], fail_times=99, max_retries=2)
    reply = asyncio.run(simulator.ask(hidden_brief="case", conversation=[]))
    assert reply.ok is False
    assert reply.text == ""
    assert "TimeoutError" in (reply.error or "")
    assert reply.attempts == 2, "max_retries attempts, then give up"
    assert simulator.stats == {"calls": 2, "failures": 1, "retries": 2}


def test_an_empty_reply_is_retried_then_failed():
    """An empty turn is a broken call, not the patient saying nothing."""
    simulator, client = _simulator(["", "   ", ""], max_retries=3)
    reply = asyncio.run(simulator.ask(hidden_brief="case", conversation=[]))
    assert reply.ok is False
    assert "empty" in (reply.error or "")
    assert len(client.requests) == 3


def test_a_retry_that_succeeds_is_recorded_as_a_retry():
    simulator, client = _simulator(["I have a cough."], fail_times=1, max_retries=3)
    reply = asyncio.run(simulator.ask(hidden_brief="case", conversation=[]))
    assert reply.ok is True
    assert reply.attempts == 2
    assert simulator.stats["retries"] == 1


@pytest.mark.parametrize(("dataset_id", "impl"), INTERVIEW_ADAPTERS)
def test_an_unreachable_environment_ends_the_episode_as_an_error(dataset_id, impl):
    """A dead simulator must not leave a scoreable answer behind.

    ``interactive_step`` returns ``None`` -- the same signal a finished episode
    gives -- so the distinguishing fact has to be in the state, which is what
    the engine reads to decide between "the model answered" and "the
    environment broke".
    """
    simulator, _client = _simulator([], fail_times=99, max_retries=1)
    adapter, samples = _adapter(dataset_id, impl, simulator=simulator)
    sample = samples[0]
    _messages, state = adapter.interactive_start(sample)
    asyncio.run(_drive(adapter, sample, state, turns=1))
    assert state.get("_environment_failed"), (
        f"{dataset_id}: a dead simulator left no trace in the episode state"
    )
    assert "TimeoutError" in state["_environment_failed"]


def test_the_engine_turns_a_broken_environment_into_an_error_record(
    fake_server, write_run_config, tmp_path, monkeypatch
):
    """End to end: the episode is an ERROR, not a zero.

    The whole point of the state flag is what the engine does with it, so this
    drives the real loop rather than asserting on the flag again.
    """
    import asyncio as _asyncio

    from abductionbench.core.config import load_run_config
    from abductionbench.core.engine import EvaluationEngine

    fake_server.state.responder = (
        lambda conversation, max_tokens: '{"action": "ask", "query": "why?"}'
    )
    adapter_module = tmp_path / "broken_env_adapter.py"
    adapter_module.write_text(
        '''
from typing import Sequence

from abductionbench.core.adapter import DatasetAdapter
from abductionbench.core.metrics import aggregate_mean_metrics
from abductionbench.core.types import AdapterDocumentation, ChatMessage, SampleScore, SampleSpec


class BrokenEnvAdapter(DatasetAdapter):
    primary_metric = "accuracy"
    system_prompt = "You investigate."
    data_delivery_mode = "interactive"
    objective_metrics = True
    max_turns = 4

    def build_samples(self):
        return [SampleSpec(sample_id="ep0", fields={"observation": "case"},
                           reference={"gold": "x"}, max_tokens=64)]

    def build_messages(self, sample):
        return ([ChatMessage(role="system", content=self.system_prompt),
                 ChatMessage(role="user", content="case")], {"answer_prefix": "Answer:"})

    def interactive_start(self, sample):
        return ([ChatMessage(role="system", content=self.system_prompt),
                 ChatMessage(role="user", content="case")], {})

    async def interactive_step(self, sample, state, assistant_text):
        state["_environment_failed"] = "patient: TimeoutError: no answer"
        return None

    def score(self, sample, response, *, output_contract=None):
        raise AssertionError("a broken environment must never reach the scorer")

    def aggregate(self, scores: Sequence[SampleScore]):
        return aggregate_mean_metrics([s.metrics for s in scores])

    def documentation(self):
        return AdapterDocumentation(dataset_id=self.dataset_id, name="broken", domain="t",
                                    source_url="n/a", processing_mode="Generation (interactive)",
                                    primary_metric="accuracy")
''',
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    config_path = write_run_config(
        base_url=fake_server.base_url,
        datasets=[{"id": "brokenenv", "impl": "broken_env_adapter:BrokenEnvAdapter",
                   "sample_size": 1}],
    )
    result = _asyncio.run(EvaluationEngine(load_run_config(config_path)).run())
    task = result.tasks[0]
    records = [json.loads(line) for line in
               (task.output_dir / "records.jsonl").read_text().splitlines() if line.strip()]
    assert records, "the episode produced no record at all"
    assert all(r["status"] == "error" for r in records), records
    assert "environment failed mid-episode" in (records[0]["response"]["error"] or "")


# --------------------------------------------------------------------------- #
# the audit trail
# --------------------------------------------------------------------------- #


def test_every_call_is_written_out_whole(tmp_path):
    run_dir = tmp_path / "run"
    simulator, _client = _simulator(
        ["I have had a cough for three days."], run_dir=run_dir,
        reasoning="the record lists cough",
    )
    from abductionbench.core.types import ChatMessage

    reply = asyncio.run(simulator.ask(
        hidden_brief="HIDDEN: the patient has pneumonia on the record",
        conversation=[ChatMessage(role="user", content="Any cough?")],
        identity=IDENTITY, sample_id="s1", turn=1, role="patient",
    ))
    assert reply.ok

    path = (run_dir / "datasets" / "medqdx" / "fake-model" / "io_SCS_interactive@1"
            / SIMULATOR_LOG_FILENAME)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(rows) == 1
    row = rows[0]
    assert row["schema"] == "simulator_call/v1"
    assert row["run_id"] == "testrun"
    assert row["dataset_id"] == "medqdx"
    assert row["evaluated_model_id"] == "fake-model"
    assert row["sample_id"] == "s1"
    assert row["turn"] == 1
    assert row["role"] == "patient"
    # The submitted prompt, untruncated, the hidden brief included: this file
    # is the record of what the environment was told.
    assert row["request"]["messages"][0]["content"].startswith("HIDDEN:")
    assert row["response"]["content"] == "I have had a cough for three days."
    assert row["response"]["reasoning_trace"] == {
        "available": True, "text": "the record lists cough"
    }
    assert row["response"]["ok"] is True
    assert row["simulator"]["model"] == "stub/model"
    assert row["simulator"]["answered_by"] == "stub/model"
    assert row["simulator"]["seed"] == 20260903
    # The visible half is recorded separately, so the boundary is auditable
    # from the log alone.
    assert row["visible_conversation"] == [{"role": "user", "content": "Any cough?"}]


def test_a_missing_reasoning_trace_is_distinguished_from_an_empty_one(tmp_path):
    run_dir = tmp_path / "run"
    simulator, _client = _simulator(["fine"], run_dir=run_dir, reasoning=None)
    asyncio.run(simulator.ask(hidden_brief="b", conversation=[], identity=IDENTITY))
    path = (run_dir / "datasets" / "medqdx" / "fake-model" / "io_SCS_interactive@1"
            / SIMULATOR_LOG_FILENAME)
    row = json.loads(path.read_text().splitlines()[0])
    assert row["response"]["reasoning_trace"] == {"available": False, "text": None}


def test_failures_and_retries_are_preserved_too(tmp_path):
    run_dir = tmp_path / "run"
    simulator, _client = _simulator([], run_dir=run_dir, fail_times=99, max_retries=2)
    asyncio.run(simulator.ask(hidden_brief="b", conversation=[], identity=IDENTITY))
    path = (run_dir / "datasets" / "medqdx" / "fake-model" / "io_SCS_interactive@1"
            / SIMULATOR_LOG_FILENAME)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(rows) == 1, "a failed call is still one logged call"
    assert rows[0]["response"]["ok"] is False
    assert rows[0]["response"]["attempts"] == 2
    assert "TimeoutError" in rows[0]["response"]["error"]
    assert rows[0]["response"]["content"] is None


def test_the_api_key_is_in_no_record_and_no_log(tmp_path):
    """The key is configuration, not provenance. It leaves no trace."""
    run_dir = tmp_path / "run"
    simulator, _client = _simulator(["ok"], run_dir=run_dir)
    assert simulator.config.api_key == "secret-key-do-not-log"
    asyncio.run(simulator.ask(hidden_brief="b", conversation=[], identity=IDENTITY))
    path = (run_dir / "datasets" / "medqdx" / "fake-model" / "io_SCS_interactive@1"
            / SIMULATOR_LOG_FILENAME)
    assert "secret-key-do-not-log" not in path.read_text()
    assert "secret-key-do-not-log" not in json.dumps(simulator.config.as_record())


def test_logging_is_skipped_rather_than_fatal_without_a_run_dir():
    simulator, _client = _simulator(["ok"], run_dir=None)
    reply = asyncio.run(simulator.ask(hidden_brief="b", conversation=[], identity=IDENTITY))
    assert reply.ok is True


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #


def test_the_simulator_is_off_unless_it_is_configured():
    """No configuration, no second model: an existing run is unchanged."""
    pool = SimulatorPool(SimulatorConfig(), TimeoutConfig())
    assert pool.for_dataset("medqdx") is None
    assert pool.as_record() == {}


def test_an_enabled_simulator_with_nowhere_to_call_is_refused():
    """Silently falling back would produce a run that looks like it followed
    the papers and did not."""
    with pytest.raises(ValueError, match="base_url"):
        SimulatorConfig(enabled=True, model="openai/gpt-4o-mini", api_key="k")
    with pytest.raises(ValueError, match="model"):
        SimulatorConfig(enabled=True, base_url="https://x/api", api_key="k")


def test_a_remote_simulator_without_a_key_is_refused_but_a_local_one_is_not():
    """A forgotten environment variable must not become a dataset of 401s.

    Every episode would be recorded as an error, which is honest but useless.
    Catching it at load costs nothing. A simulator served on this box may
    legitimately need no key, so the rule is about reaching off the machine.
    """
    with pytest.raises(ValueError, match="api_key"):
        SimulatorConfig(enabled=True, model="m", base_url="https://openrouter.ai/api")
    local = SimulatorConfig(enabled=True, model="m", base_url="http://127.0.0.1:18000")
    assert local.api_key is None
    # And a disabled simulator never needs one, so a box with no key can still
    # load every config in the repository.
    assert SimulatorConfig().enabled is False


def test_the_shipped_config_loads_without_the_key_when_the_simulator_is_off(monkeypatch):
    """pilot.yaml is loaded by every run, including on boxes with no key."""
    from abductionbench.core.config import load_run_config

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("ABENCH_API_KEY", "x")
    monkeypatch.setenv("ABENCH_SIMULATOR", "false")
    config = load_run_config(Path("configs/runs/pilot.yaml"))
    assert config.engine.simulator.enabled is False


def test_each_dataset_can_name_its_own_model():
    config = SimulatorConfig(
        enabled=True, base_url="https://x/api", model="openai/gpt-4o-mini", api_key="k",
        by_dataset={"vivabench": {"model": "openai/gpt-4.1", "max_tokens": 256}},
    )
    assert config.for_dataset("medqdx").model == "openai/gpt-4o-mini"
    assert config.for_dataset("med_inquire").model == "openai/gpt-4o-mini"
    viva = config.for_dataset("vivabench")
    assert viva.model == "openai/gpt-4.1"
    assert viva.max_tokens == 256
    # An override must not lose what it did not mention.
    assert viva.base_url == "https://x/api"
    assert viva.seed == 20260903


def test_the_settings_that_shape_a_reply_are_all_recorded():
    """Everything a reader needs to reproduce the environment, and no key."""
    config = SimulatorConfig(enabled=True, base_url="https://x/api",
                             model="openai/gpt-4.1", api_key="sk-secret-9Q7X")
    record = config.as_record()
    assert set(record) == {"model", "base_url", "temperature", "seed",
                           "max_tokens", "max_retries", "extra"}
    assert "sk-secret-9Q7X" not in json.dumps(record)


def test_vendor_fields_reach_the_request():
    """`extra` is how a reasoning model is told not to reason."""
    simulator, client = _simulator(["ok"], extra={"reasoning": {"enabled": False}})
    asyncio.run(simulator.ask(hidden_brief="b", conversation=[]))
    payload = client.sampling[0].to_payload()
    assert payload["reasoning"] == {"enabled": False}


def test_no_simulator_may_reason_or_search(monkeypatch):
    """A simulator reads a record back; there is nothing there to think about.

    A chain would share `max_tokens` with the reply -- and an empty reply is
    retried and then fails the episode -- as well as being slower and landing
    in the audit log as material nobody asked for. Search is worse than
    useless: a simulator that looked something up would be answering from the
    internet instead of from the case, which is not the environment the paper
    describes.
    """
    from abductionbench.core.config import load_run_config

    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("ABENCH_API_KEY", "k")
    config = load_run_config(Path("configs/runs/pilot.yaml")).engine.simulator
    for dataset_id in ("medqdx", "med_inquire", "vivabench"):
        resolved = config.for_dataset(dataset_id)
        assert resolved.extra.get("reasoning") == {"enabled": False}, dataset_id
        assert resolved.extra.get("plugins") == [], dataset_id
        # OpenRouter turns search on with an `:online` model suffix too.
        assert not resolved.model.endswith(":online"), dataset_id
        # And it is recorded, because whether the model reasoned or searched
        # shapes the reply and cannot be reconstructed from the result later.
        record = resolved.as_record()["extra"]
        assert record["reasoning"] == {"enabled": False}
        assert record["plugins"] == []


def test_the_pinned_sampling_is_actually_sent():
    simulator, client = _simulator(["ok"], temperature=0.0, seed=4242, max_tokens=128)
    asyncio.run(simulator.ask(hidden_brief="b", conversation=[]))
    sampling = client.sampling[0]
    assert sampling.temperature == 0.0
    assert sampling.seed == 4242
    assert sampling.max_tokens == 128
    assert sampling.to_payload()["seed"] == 4242


def test_the_run_config_wires_the_two_requested_models():
    """pilot.yaml is what a real run loads, so the wiring is checked there."""
    import os

    from abductionbench.core.config import load_run_config

    env = {"OPENROUTER_API_KEY": "k", "ABENCH_API_KEY": "k"}
    previous = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        config = load_run_config(Path("configs/runs/pilot.yaml")).engine.simulator
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    assert config.enabled is True
    assert config.for_dataset("medqdx").model == "openai/gpt-4o-mini"
    assert config.for_dataset("med_inquire").model == "openai/gpt-4o-mini"
    viva = config.for_dataset("vivabench")
    # gpt-4o-mini since 2026-09-25 (was gpt-5.6-luna), like the two patients.
    assert viva.model == "openai/gpt-4o-mini"
    # A reasoning model shares max_tokens between its chain and its reply; at
    # the 512 default a long chain returns empty content, which is retried and
    # then fails the episode.
    assert viva.max_tokens >= 2048
    assert config.temperature == 0.0


def test_a_pool_shares_one_client_per_endpoint_and_model():
    async def _check():
        pool = SimulatorPool(
            SimulatorConfig(enabled=True, base_url="https://x/api", model="m", api_key="k",
                            by_dataset={"vivabench": {"model": "other"}}),
            TimeoutConfig(),
        )
        first = pool.for_dataset("medqdx")
        second = pool.for_dataset("med_inquire")
        third = pool.for_dataset("vivabench")
        assert first.client is second.client, "same model, same client"
        assert third.client is not first.client, "a different model needs its own client"
        assert first.calls is third.calls, "max_parallel_calls is run-wide"
        await pool.aclose()

    asyncio.run(_check())


def test_the_record_lists_the_simulators_that_played_not_the_ones_configured():
    """A row in the Simulators sheet is a claim about how a run was produced.

    It says this dataset's environment was answered by a model rather than by
    the release's own deterministic logic, and a reader uses it to decide how
    much of a result rests on a simulated patient. A simulator that never made
    a call shaped nothing, and listing it invites that discount to be applied
    to results it had nothing to do with.

    Before this, `for_dataset` was called while building EVERY adapter, so all
    42 datasets registered one, and the sheet credited a patient model against
    abd, aer, art and climate_fever -- none of which has an environment at all.
    """
    async def _check():
        pool = SimulatorPool(
            SimulatorConfig(enabled=True, base_url="https://x/api", model="m", api_key="k"),
            TimeoutConfig(),
        )
        played = pool.for_dataset("medqdx")
        pool.for_dataset("med_inquire")          # registered, never called
        assert set(pool.as_record()) == set(), "nothing has played yet"

        played.stats["calls"] += 1
        assert set(pool.as_record()) == {"medqdx"}

        # A simulator that only ever failed still played: the episodes it
        # broke are part of how the run came out, and hiding it would make an
        # environment failure look like a deterministic one.
        pool.for_dataset("med_inquire").stats["failures"] += 1
        assert set(pool.as_record()) == {"medqdx", "med_inquire"}
        await pool.aclose()

    asyncio.run(_check())


# --------------------------------------------------------------------------- #
# the deterministic environment still works
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("dataset_id", "impl"), INTERVIEW_ADAPTERS)
def test_without_a_simulator_the_lexical_environment_still_answers(dataset_id, impl):
    """Opting out must leave the old, reproducible protocol exactly as it was."""
    adapter, samples = _adapter(dataset_id, impl, simulator=None)
    sample = samples[0]
    _messages, state = adapter.interactive_start(sample)
    asks = {
        "medqdx": "Do you have a fever?",
        "med_inquire": '{"action_type": "AskQuestion", "action_text": "What brings you in?"}',
        "vivabench": '{"action": "history", "query": "What brings you in?"}',
    }[dataset_id]
    reply = step(adapter, sample, state, asks)
    assert isinstance(reply, str) and reply.strip()
    assert not state.get("_environment_failed")
