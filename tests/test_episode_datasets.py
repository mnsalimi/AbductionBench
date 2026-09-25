"""Interactive + sequential datasets: io with reasoning off at 0.7, their own
metrics only, athena_bench's per-turn judging, and ddxplus's BM25 patient."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from abductionbench.adapters._retrieval import BM25, terms
from abductionbench.adapters.athena_bench import AthenaBenchAdapter, parse_prediction
from abductionbench.core import episode_metrics
from abductionbench.core.adapter import AdapterContext
from abductionbench.core.client import _apply_override_extra
from abductionbench.core.config import ModelSamplingConfig
from abductionbench.core.metrics import MISSING_METRIC
from abductionbench.core.modes import COT, IO, TaskModes
from abductionbench.core.types import ModelResponse, ResponseStatus, SampleScore, SamplingParams


def _athena(tmp_path: Path, sample_size: int = 50) -> AthenaBenchAdapter:
    adapter = AthenaBenchAdapter(
        AdapterContext(
            dataset_id="athena_bench",
            data_dir=tmp_path / "athena_bench",
            sample_size=sample_size,
            seed=13,
            options={},
            modes=TaskModes(prompt_mode=IO, data_delivery_mode="sequential"),
        )
    )
    adapter.prepare()
    return adapter


def _verdict(yes: bool):
    return SimpleNamespace(positive=yes)


# --------------------------------------------------------------------------- #
# sampling: temperature 0.7, no seed, native reasoning off


def test_episode_turns_run_at_07_without_a_seed_and_with_reasoning_off():
    from abductionbench.core.engine import EvaluationEngine

    engine = SimpleNamespace()
    model = SimpleNamespace(
        id="m",
        sampling=ModelSamplingConfig(
            temperature=0.0, seed=7,
            extra={"chat_template_kwargs": {"enable_thinking": True}},
            reasoning_off_extra={"chat_template_kwargs": {"enable_thinking": False}},
        ),
    )
    base = SamplingParams(max_tokens=100, temperature=0.0, seed=7,
                          extra=(("chat_template_kwargs", {"enable_thinking": True}),))
    out = EvaluationEngine._episode_sampling(engine, base, model, TaskModes(prompt_mode=IO))
    assert out.temperature == 0.7 and out.seed is None
    payload = {"model": "m", "messages": [], "chat_template_kwargs": {"enable_thinking": True,
                                                                       "other": 1}}
    _apply_override_extra(payload, out)
    # Over the model's own extra, and merged rather than replaced.
    assert payload["chat_template_kwargs"] == {"enable_thinking": False, "other": 1}


def test_a_static_prompt_fingerprint_is_unchanged_by_the_new_field():
    plain = SamplingParams(max_tokens=10)
    assert "override_extra" not in plain.signature()
    assert plain.override_extra() == {}


def test_interactive_and_sequential_datasets_are_io_only():
    assert AthenaBenchAdapter.supports_modes(TaskModes(prompt_mode=COT)) is not None
    assert AthenaBenchAdapter.supports_modes(TaskModes(prompt_mode=IO)) is None


# --------------------------------------------------------------------------- #
# only the listed metrics


def test_interactive_projection_keeps_only_its_three_metrics():
    adapter = SimpleNamespace(final_answer_metric="diagnosis_judged")
    score = SampleScore(
        metrics={"diagnosis_judged": 1.0, "interaction_steps": 4.0, "mqd": 2.0,
                 "interaction_step_relevance_rate": 0.75, "interaction_relevant_steps": 3.0},
        prediction="x",
    )
    out = episode_metrics.project(adapter, "interactive", score)
    assert out.metrics == {"final_answer_accuracy": 1.0, "turns_to_final_output": 4.0,
                           "interaction_step_relevance": 0.75}
    assert out.details["adapter_metrics"]["mqd"] == 2.0
    # Idempotent: a resumed record projects to the same thing.
    again = episode_metrics.project(adapter, "interactive", out)
    assert again.metrics == out.metrics


def test_no_action_to_grade_is_none_not_zero():
    adapter = SimpleNamespace(final_answer_metric="accuracy")
    out = episode_metrics.project(adapter, "interactive",
                                  SampleScore(metrics={"accuracy": 0.0, "interaction_steps": 1.0}))
    assert out.metrics["interaction_step_relevance"] == MISSING_METRIC
    agg = episode_metrics.aggregate("interactive", [out])
    assert "interaction_step_relevance" not in agg and agg["final_answer_accuracy"] == 0.0


# --------------------------------------------------------------------------- #
# athena_bench


def test_athena_draws_50_of_57_and_places_its_file(tmp_path):
    adapter = _athena(tmp_path)
    samples = adapter.build_samples()
    assert len(samples) == 50
    assert (tmp_path / "athena_bench" / "athena_bench_dataset.json").exists()
    assert len({s.sample_id for s in samples}) == 50


def test_athena_sequence_gives_each_turns_evidence_and_ends_after_the_last(tmp_path):
    adapter = _athena(tmp_path)
    sample = adapter.build_samples()[0]
    turns = sample.metadata["_turns"]
    messages, state = adapter.interactive_start(sample)
    assert messages[0].role == "system"
    assert turns[0]["evidences"][0] in messages[1].content
    assert turns[0]["question"] in messages[1].content
    assert "<answer>" in messages[1].content
    replies = []
    for i in range(len(turns)):
        replies.append(adapter.interactive_step(sample, state, f"<answer>Actor {i}</answer>"))
    assert replies[-1] is None
    for i, reply in enumerate(replies[:-1], start=1):
        assert turns[i]["evidences"][0] in reply and f"Turn {i + 1} of {len(turns)}" in reply
    assert state["predictions"] == [f"Actor {i}" for i in range(len(turns))]


def test_athena_metrics_from_per_turn_verdicts(tmp_path):
    adapter = _athena(tmp_path)
    sample = adapter.build_samples()[0]
    sample.metadata["n_turns"] = 5
    sample.metadata["_episode_state"] = {
        "predictions": ["APT41", "Lazarus", "TraderTraitor", "Lazarus Group", "Lazarus"]}
    response = ModelResponse(sample_id=sample.sample_id, model_id="m",
                             status=ResponseStatus.OK, content="<answer>Lazarus</answer>")
    score = adapter.score(sample, response)
    assert score.metrics["final_answer_accuracy"] == 0.0     # unjudged yet
    requests = adapter.judge_requests(sample, response, score)
    # Every turn vs the gold; vs the final only where they differ once normalised.
    assert set(requests) == {"gold:0", "gold:1", "gold:2", "gold:3", "gold:4",
                             "final:0", "final:2"}
    verdicts = {"gold:0": _verdict(False), "gold:1": _verdict(True), "gold:2": _verdict(True),
                "gold:3": _verdict(True), "gold:4": _verdict(True),
                "final:0": _verdict(False), "final:2": _verdict(True)}
    judged = adapter.apply_judges(sample, response, score, verdicts)
    m = judged.metrics
    assert m["final_answer_accuracy"] == 1.0
    assert m["turns_to_correctness"] == 2.0
    assert m["average_sample_accuracy"] == 0.8
    assert m["turns_to_final_output"] == 5.0
    # Turn 2 already said Lazarus; turn 3 (TraderTraitor) judged an alias too.
    assert m["turns_to_first_final_prediction"] == 2.0
    assert judged.details["turn_correct"] == [0, 1, 1, 1, 1]
    # The projection leaves these five alone.
    projected = episode_metrics.project(adapter, "sequential", judged, response)
    assert projected.metrics == m


def test_athena_never_correct_is_none_and_sits_out_the_mean(tmp_path):
    adapter = _athena(tmp_path)
    sample = adapter.build_samples()[0]
    sample.metadata["_episode_state"] = {"predictions": ["A", "B"]}
    response = ModelResponse(sample_id=sample.sample_id, model_id="m",
                             status=ResponseStatus.OK, content="B")
    score = adapter.score(sample, response)
    judged = adapter.apply_judges(sample, response, score,
                                  {"gold:0": _verdict(False), "gold:1": _verdict(False)})
    assert judged.metrics["turns_to_correctness"] == MISSING_METRIC
    agg = episode_metrics.aggregate("sequential", [judged])
    assert "turns_to_correctness" not in agg


def test_answer_block_parsing():
    assert parse_prediction("<answer> APT29 </answer>") == "APT29"
    assert parse_prediction("It is\nKimsuky") == "Kimsuky"
    assert parse_prediction("") is None


# --------------------------------------------------------------------------- #
# ddxplus BM25


def test_bm25_normalises_words_and_applies_the_threshold():
    assert terms("Coughing") == terms("coughs") == ["cough"]
    inventory = ["Do you have a cough?", "Have you been coughing up blood?",
                 "Have you gained weight recently?", "Do you smoke cigarettes?",
                 "Have you traveled out of the country in the last 4 weeks?"]
    bm = BM25(inventory)
    hits = bm.search("are you coughing?", inventory, limit=3)
    assert hits and hits[0][0] in inventory[:2]
    assert all("weight" not in q for q, _s in hits)
    assert len(bm.search("cough", inventory, limit=1)) == 1
    assert bm.search("how old are you?", inventory) == []


def test_ddxplus_answers_with_the_most_similar_recorded_questions(monkeypatch):
    from abductionbench.adapters.ddxplus import DDXPlusAdapter

    adapter = DDXPlusAdapter.__new__(DDXPlusAdapter)
    adapter.context = SimpleNamespace(option=lambda k, d=None: d)
    adapter._catalogue = {"E1": "Do you have a cough?", "E2": "Do you smoke cigarettes?",
                          "E3": "Do you have a fever?", "E4": "Have you been coughing up blood?"}
    state = {"evidence": SimpleNamespace(categories={"ask": {
        "Do you have a cough?": "yes", "Have you been coughing up blood?": "yes"}})}
    reply = adapter._retrieve(state, "Are you coughing a lot?")
    assert reply.startswith("These are the recorded questions most similar")
    assert "Do you have a cough? -> yes" in reply
    assert adapter._retrieve(state, "Do you smoke?") == ""
    assert json.dumps(state["retrievals"])  # logged for the record


# --------------------------------------------------------------------------- #
# end to end: athena through the engine, judged per turn


def test_athena_end_to_end(fake_server, write_run_config, tmp_path):
    import asyncio

    from abductionbench.core.checkpoint import dedupe_records, load_records
    from abductionbench.core.config import load_run_config
    from abductionbench.core.engine import EvaluationEngine
    from abductionbench.core.reporting import build_turns_frame

    model_payloads: list[dict] = []

    def _watch(payload):
        items = payload.get("requests") or [payload]
        for item in items:
            text = json.dumps(item.get("messages", []))
            if "threat actor name" in text:
                model_payloads.append({**payload, **item})
        return None

    fake_server.state.fail_request = _watch
    fake_server.state.responder = lambda conv, max_tokens: (
        "<answer>Lazarus</answer>" if "threat actor name" in json.dumps(conv) else "Score: 1"
    )
    model = {
        "id": "fake-model",
        "model_name": "test/model",
        "endpoint": {"base_url": fake_server.base_url, "api_key": "test-key",
                     "batch": {"enabled": False}},
        "sampling": {"max_tokens_default": 64, "max_tokens_cap": 512, "max_tokens_floor": 16,
                     "temperature": 0.0, "seed": 5,
                     "extra": {"chat_template_kwargs": {"enable_thinking": True}},
                     "reasoning_off_extra": {"chat_template_kwargs": {"enable_thinking": False}}},
        "limits": {"max_parallel_batches": 2, "context_window": 16384},
    }
    config_path = write_run_config(
        base_url=fake_server.base_url,
        models=[model],
        datasets=[{"id": "athena_bench",
                   "impl": "abductionbench.adapters.athena_bench:AthenaBenchAdapter",
                   "sample_size": 3, "primary_metric": "final_answer_accuracy"}],
        engine={"judge": {"enabled": True, "model": "fake-model",
                          "template": "judge_binary_v1", "group_size": 1}},
        modes={"prompt_modes": ["io", "cot"]},
    )
    result = asyncio.run(EvaluationEngine(load_run_config(config_path)).run())
    tasks = [t for t in result.tasks if t.identity.dataset_id == "athena_bench"]
    assert [t.identity.prompt_mode for t in tasks] == ["io"], "episodes run io only"
    task = tasks[0]
    assert set(task.metrics) >= set(episode_metrics.EPISODE_METRICS["sequential"])
    assert task.metrics["final_answer_accuracy"] == 1.0
    assert task.metrics["turns_to_correctness"] == 1.0
    assert task.metrics["average_sample_accuracy"] == 1.0
    assert not any(k.endswith("_strict") or k.startswith("best_of_n") for k in task.metrics)

    records = dedupe_records(load_records(task.output_dir / "records.jsonl"))
    assert len(records) == 3
    for record in records:
        assert set(record["metrics"]) == set(episode_metrics.EPISODE_METRICS["sequential"])
        assert record["metrics"]["turns_to_final_output"] == record["details"]["n_turns"]

    assert model_payloads, "the model was asked"
    for payload in model_payloads:
        assert payload.get("temperature") == 0.7
        assert "seed" not in payload or payload["seed"] is None
        assert payload["chat_template_kwargs"]["enable_thinking"] is False

    frame = build_turns_frame([task.output_dir], clip=2000)
    assert len(frame) == sum(r["details"]["n_turns"] for r in records)
    assert set(frame["turn_prediction"]) == {"Lazarus"}
    assert set(frame["turn_correct"]) == {1}
