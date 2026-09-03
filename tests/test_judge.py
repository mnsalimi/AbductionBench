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
    cache = task.output_dir / "judge_cache" / "verdicts.json"
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
