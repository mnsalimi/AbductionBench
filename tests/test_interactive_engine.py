"""The engine's multi-turn execution: episodes, not prompts.

An interactive benchmark is only interactive if the environment actually gets
asked and actually answers, so these drive the whole loop through the engine
against the fake server -- batching, turn accounting, and the transcript that
ends up in the record.
"""

from __future__ import annotations

import asyncio

from abductionbench.core.adapter import DatasetAdapter
from abductionbench.core.config import load_run_config
from abductionbench.core.engine import EvaluationEngine
from abductionbench.core.types import ChatMessage, SampleScore, SampleSpec


class EchoEnvironmentAdapter(DatasetAdapter):
    """Three episodes that each need a fixed number of turns to finish.

    The environment answers with a countdown; the item is "solved" when the
    model has been told the secret, which only happens on the last turn. That
    makes the score depend on the interaction having really run.
    """

    adapter_version = "test-1.0"
    primary_metric = "solved"
    system_prompt = "You are talking to a test environment."
    data_delivery_mode = "interactive"
    objective_metrics = True
    max_turns = 6

    def build_samples(self) -> list[SampleSpec]:
        return [
            SampleSpec(
                sample_id=f"e{index}",
                fields={"observation": f"episode {index}"},
                reference={"turns_needed": index + 1},
                task_kind="generation",
                metadata={"turns_needed": index + 1},
            )
            for index in range(3)
        ]

    def build_messages(self, sample):
        return (
            [
                ChatMessage(role="system", content=self.system_prompt),
                ChatMessage(role="user", content=str(sample.fields["observation"])),
            ],
            {"answer_prefix": "Answer:"},
        )

    def interactive_start(self, sample):
        messages, _ = self.build_messages(sample)
        return list(messages), {"turn": 0}

    def interactive_step(self, sample, state, assistant_text):
        state["turn"] += 1
        if state["turn"] >= int(sample.metadata["turns_needed"]):
            return None
        return f"not yet: {sample.metadata['turns_needed'] - state['turn']} to go"

    def score(self, sample, response, *, output_contract=None):
        used = int(sample.metadata.get("turns_used", 0))
        return SampleScore(
            metrics={"solved": 1.0 if used == int(sample.reference["turns_needed"]) else 0.0,
                     "turns": float(used)},
            prediction=used,
        )

    def aggregate(self, scores):
        return {
            "solved": sum(s.metrics["solved"] for s in scores) / max(1, len(scores)),
            "turns": sum(s.metrics["turns"] for s in scores) / max(1, len(scores)),
        }

    def documentation(self):
        from abductionbench.core.types import AdapterDocumentation

        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="Echo environment",
            domain="test",
            source_url="",
            processing_mode="Generation",
            split_used="synthetic",
        )


def _run(config_path):
    config = load_run_config(config_path)
    engine = EvaluationEngine(config)
    return asyncio.run(engine.run())


def test_episodes_run_to_completion_and_are_scored(
    fake_server, write_run_config, tmp_path, monkeypatch
):
    monkeypatch.syspath_prepend(str(tmp_path))
    config_path = write_run_config(
        base_url=fake_server.base_url,
        datasets=[
            {
                "id": "episodes",
                "impl": "test_interactive_engine:EchoEnvironmentAdapter",
                "sample_size": 3,
                "options": {},
            }
        ],
    )
    result = _run(config_path)
    task = result.tasks[0]

    # Delivery is the adapter's, and it reaches the identity and the columns.
    assert task.identity.data_delivery_mode == "interactive"
    assert task.identity.template_mode.endswith("|generation|interactive")

    # Every episode ran for exactly as many turns as its environment demanded.
    assert task.n_scored == 3
    assert task.metrics["solved"] == 1.0
    assert task.metrics["turns"] == 2.0        # (1 + 2 + 3) / 3

    # Six model turns in total (1 + 2 + 3), but each turn of all still-live
    # episodes goes out as one batch, so the run costs three batch calls plus
    # the endpoint probe -- not six.
    assert fake_server.state.requests < 6


def test_a_transcript_is_kept_for_every_episode(fake_server, write_run_config, tmp_path,
                                                monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    config_path = write_run_config(
        base_url=fake_server.base_url,
        datasets=[
            {
                "id": "episodes",
                "impl": "test_interactive_engine:EchoEnvironmentAdapter",
                "sample_size": 3,
                "options": {},
            }
        ],
    )
    result = _run(config_path)
    records_path = result.tasks[0].output_dir / "records.jsonl"
    import json

    rows = [json.loads(line) for line in records_path.read_text().splitlines()]
    by_id = {row["sample_id"]: row for row in rows}
    assert by_id["e2"]["metadata"]["turns_used"] == 3
    assert by_id["e2"]["response"]["usage"]["turns"] == 3


class ChattyEnvironmentAdapter(EchoEnvironmentAdapter):
    """An environment that pads every reply until the window is gone.

    A real interactive benchmark does this on its own: a viva that discloses
    findings for twenty turns eventually leaves no room to answer in. The
    episode has to end at that point rather than send a request the server can
    only reject.
    """

    max_turns = 40

    def interactive_step(self, sample, state, assistant_text):
        state["turn"] = state.get("turn", 0) + 1
        return "finding: " + ("padding " * 400)


def test_an_episode_that_runs_out_of_context_ends_rather_than_failing(
    fake_server, write_run_config, tmp_path, monkeypatch
):
    monkeypatch.syspath_prepend(str(tmp_path))
    config_path = write_run_config(
        base_url=fake_server.base_url,
        datasets=[
            {
                "id": "chatty",
                "impl": "test_interactive_engine:ChattyEnvironmentAdapter",
                "sample_size": 2,
                "options": {},
            }
        ],
    )
    result = _run(config_path)
    task = result.tasks[0]

    # Every episode is accounted for, and none of them failed: running out of
    # room is an outcome, not an error.
    assert task.n_scored == 2
    assert task.n_error == 0

    import json

    rows = [
        json.loads(line)
        for line in (task.output_dir / "records.jsonl").read_text().splitlines()
    ]
    assert all(row["metadata"]["context_exhausted"] for row in rows)
    # It stopped well before the 40-turn limit, because the window ran out first.
    assert all(0 < row["metadata"]["turns_used"] < 40 for row in rows)


def test_two_models_do_not_score_each_others_episodes(
    fake_server, write_run_config, tmp_path, monkeypatch
):
    """The prompt set is shared by every model; per-episode state must not be.

    Both models see the same SampleSpec objects. If an episode writes its
    transcript onto the sample it ran, the second model overwrites the first's
    before the first is scored -- and the first model is graded on the second
    model's episode. With two models running concurrently, silently.
    """
    monkeypatch.syspath_prepend(str(tmp_path))
    endpoint = {
        "base_url": fake_server.base_url,
        "api_key": "test-key",
        "batch": {"enabled": True, "path": "/v1/chat/completions/batch", "group_size": 4},
    }
    models = [
        {
            "id": f"model-{index}",
            "model_name": "test/model",
            "endpoint": endpoint,
            "sampling": {"max_tokens_default": 256, "max_tokens_cap": 256},
            "limits": {"max_parallel_batches": 2, "context_window": 4096},
        }
        for index in (1, 2)
    ]
    config_path = write_run_config(
        base_url=fake_server.base_url,
        models=models,
        datasets=[
            {
                "id": "episodes",
                "impl": "test_interactive_engine:EchoEnvironmentAdapter",
                "sample_size": 3,
                "options": {},
            }
        ],
    )
    result = _run(config_path)
    assert len(result.tasks) == 2
    for task in result.tasks:
        # Each model's episodes ran to their own environment's demands.
        assert task.metrics["solved"] == 1.0, task.identity.model_id
        assert task.metrics["turns"] == 2.0
