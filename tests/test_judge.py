"""The optional LLM-judge stage, end to end against the fake server."""

from __future__ import annotations

import asyncio

from abductionbench.core.config import load_run_config
from abductionbench.core.engine import EvaluationEngine


def test_judge_stage_updates_scores(fake_server, write_run_config, tmp_path, monkeypatch):
    # The judge model always answers "Verdict: yes".
    fake_server.state.responder = lambda conv, max_tokens: "Verdict: yes"

    adapter_module = tmp_path / "judged_adapter.py"
    adapter_module.write_text(
        '''
from typing import Any, Sequence

from abductionbench.core.adapter import DatasetAdapter
from abductionbench.core.metrics import aggregate_mean_metrics
from abductionbench.core.types import AdapterDocumentation, SampleScore, SampleSpec


class JudgedAdapter(DatasetAdapter):
    primary_metric = "judged_accuracy"
    system_prompt = "You explain observations."

    def build_messages(self, sample):
        from abductionbench.core.types import ChatMessage

        return (
            [
                ChatMessage(role="system", content=self.system_prompt),
                ChatMessage(role="user", content=str(sample.fields["observation"])),
            ],
            {"answer_prefix": "Answer:"},
        )

    def build_samples(self):
        return [
            SampleSpec(
                sample_id=f"j{i}",
                fields={"observation": f"obs {i}"},
                reference="gold explanation",
                max_tokens=64,
            )
            for i in range(4)
        ]

    def score(self, sample, response, *, output_contract=None):
        # Deterministic part scores 0; the judge is what can lift it.
        return SampleScore(metrics={"judged_accuracy": 0.0}, prediction=response.text[:50])

    def judge_request(self, sample, response, score):
        return {"candidate": response.text or "", "gold": sample.reference,
                "observation": sample.fields["observation"]}

    def apply_judge(self, sample, response, score, verdict):
        return SampleScore(
            metrics={**score.metrics, "judged_accuracy": 1.0 if verdict.positive else 0.0},
            prediction=score.prediction,
            parse_ok=score.parse_ok,
            details={**score.details, "judge_label": verdict.label},
        )

    def aggregate(self, scores: Sequence[SampleScore]):
        return aggregate_mean_metrics([s.metrics for s in scores])

    def documentation(self):
        return AdapterDocumentation(
            dataset_id=self.dataset_id, name="judged", domain="test",
            source_url="n/a", processing_mode="Generation",
            primary_metric="judged_accuracy",
        )
''',
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    config_path = write_run_config(
        base_url=fake_server.base_url,
        datasets=[{"id": "judged", "impl": "judged_adapter:JudgedAdapter", "sample_size": 4}],
        engine={
            "judge": {
                "enabled": True,
                "model": "fake-model",
                "template": "judge_binary_v1",
                "group_size": 4,
            }
        },
    )
    config = load_run_config(config_path)
    engine = EvaluationEngine(config)
    result = asyncio.run(engine.run())

    task = result.tasks[0]
    assert task.metrics["judged_accuracy"] == 1.0
    # Run-level, not task-level: the same (template, fields) verdict is
    # otherwise re-bought once per task.
    cache = task.output_dir.parents[3] / "judge_cache" / "verdicts.json"
    assert cache.exists()


def test_judge_verdict_parsing():
    from abductionbench.core.judge import JudgeStage
    from abductionbench.core.prompts import PromptRegistry

    registry = PromptRegistry([__import__("pathlib").Path("configs/prompts")])
    binary = registry.get("judge_binary_v1")
    graded = registry.get("judge_graded_v1")

    parse = JudgeStage._parse
    stage = object.__new__(JudgeStage)
    stage.template = binary
    verdict = parse(stage, "some reasoning\nVerdict: no")
    assert verdict.label == "no" and verdict.parsed and not verdict.positive
    verdict = parse(stage, "Verdict: yes")
    assert verdict.positive

    stage.template = graded
    verdict = parse(stage, "Score: 4")
    assert verdict.score == 4 / 5 and verdict.positive
    verdict = parse(stage, "unparseable")
    assert verdict.score is None


def test_the_judge_is_shown_the_whole_exchange(fake_server, write_run_config, tmp_path,
                                               monkeypatch):
    """Everything the model was sent, and everything it produced.

    The judge used to see the adapter's extracted `candidate` and an observation
    clipped to 200 words -- never the prompt, never the rest of the reply. On a
    chain-of-thought answer that is one line out of thousands, with no sight of
    the question it answered or the reasoning that produced it. The stage now
    adds both to every request, so all the judged datasets get it rather than
    the handful whose adapters happened to pass context.
    """
    import asyncio

    from abductionbench.core.config import load_run_config
    from abductionbench.core.engine import EvaluationEngine

    seen: list[list[dict]] = []

    def responder(conversation, max_tokens):
        seen.append(conversation)
        body = " ".join(str(m.get("content", "")) for m in conversation)
        # The judge's own system prompt opens "You grade candidate explanations".
        return "Verdict: yes" if "You grade" in body else "Answer: a sprinkler"

    fake_server.state.responder = responder

    adapter_module = tmp_path / "exchange_adapter.py"
    adapter_module.write_text(
        '''
from typing import Sequence

from abductionbench.core.adapter import DatasetAdapter
from abductionbench.core.metrics import aggregate_mean_metrics
from abductionbench.core.types import AdapterDocumentation, ChatMessage, SampleScore, SampleSpec


class ExchangeAdapter(DatasetAdapter):
    primary_metric = "judged"
    system_prompt = "You explain observations."

    def build_messages(self, sample):
        return ([ChatMessage(role="system", content=self.system_prompt),
                 ChatMessage(role="user", content=str(sample.fields["observation"]))],
                {"answer_prefix": "Answer:"})

    def build_samples(self):
        return [SampleSpec(sample_id="e0", fields={"observation": "the grass is wet"},
                           reference="the sprinkler ran", max_tokens=64)]

    def score(self, sample, response, *, output_contract=None):
        return SampleScore(metrics={"judged": 0.0}, prediction="a sprinkler")

    def judge_request(self, sample, response, score):
        return {"candidate": score.prediction, "gold": sample.reference}

    def apply_judge(self, sample, response, score, verdict):
        return SampleScore(metrics={"judged": 1.0 if verdict.positive else 0.0},
                           prediction=score.prediction)

    def aggregate(self, scores: Sequence[SampleScore]):
        return aggregate_mean_metrics([s.metrics for s in scores])

    def documentation(self):
        return AdapterDocumentation(dataset_id=self.dataset_id, name="x", domain="t",
                                    source_url="n/a", processing_mode="Generation",
                                    primary_metric="judged")
''',
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    config_path = write_run_config(
        base_url=fake_server.base_url,
        datasets=[{"id": "exch", "impl": "exchange_adapter:ExchangeAdapter", "sample_size": 1}],
        engine={"judge": {"enabled": True, "model": "fake-model",
                          "template": "judge_binary_v1", "group_size": 1}},
    )
    asyncio.run(EvaluationEngine(load_run_config(config_path)).run())

    judge_prompts = [
        "".join(str(m.get("content", "")) for m in conversation)
        for conversation in seen
        if "You grade" in "".join(str(m.get("content", "")) for m in conversation)
    ]
    assert judge_prompts, f"the judge was never called ({len(seen)} call(s) seen)"
    body = judge_prompts[-1]
    assert "WHAT THE MODEL WAS ASKED" in body
    assert "the grass is wet" in body, "the judge cannot see the question the model answered"
    assert "WHAT THE MODEL REPLIED, IN FULL" in body
    assert "Answer: a sprinkler" in body, "the judge cannot see what the model actually replied"


def test_the_whole_exchange_is_budgeted_not_unbounded():
    """A request that overruns the judge's window gets no verdict at all."""
    from abductionbench.core.judge import clip_middle

    text = "HEAD" + ("x" * 100_000) + "TAIL"
    clipped = clip_middle(text, 1000)
    assert len(clipped) <= 1100
    assert clipped.startswith("HEAD") and clipped.endswith("TAIL")
    assert "omitted from the middle" in clipped
    # Anything that already fits is passed through untouched.
    assert clip_middle("short", 1000) == "short"
