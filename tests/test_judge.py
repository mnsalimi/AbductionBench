"""The optional LLM-judge stage, end to end against the fake server."""

from __future__ import annotations

import asyncio

from abductionbench.core.config import load_run_config
from abductionbench.core.engine import EvaluationEngine


def test_judge_stage_updates_scores(fake_server, write_run_config, tmp_path, monkeypatch):
    # The judge model always accepts: the answer judges are binary now.
    fake_server.state.responder = lambda conv, max_tokens: "Score: 1"

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
    plausibility = registry.get("judge_binary_plausibility_v1")

    parse = JudgeStage._parse
    stage = object.__new__(JudgeStage)
    stage.template = binary
    verdict = parse(stage, "some reasoning\nScore: 0")
    assert verdict.score == 0.0 and verdict.parsed and not verdict.positive
    verdict = parse(stage, "Score: 1")
    assert verdict.score == 1.0 and verdict.positive

    stage.template = plausibility
    verdict = parse(stage, "Score: 1")
    assert verdict.score == 1.0 and verdict.positive
    verdict = parse(stage, "Score: 0")
    assert verdict.score == 0.0 and not verdict.positive
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


def test_the_whole_request_is_budgeted_and_nothing_in_it_is_cut():
    """Every field goes to the judge whole, so the SUM is what is bounded.

    The answer, gold and observation used to be sliced to a few hundred
    characters each, which kept the request small by showing the judge a cut
    answer and calling it the model's. Nothing is sliced now; a request whose
    fields together would overrun the window is skipped and counted instead.
    """
    from abductionbench.core.judge import exceeds_total

    assert exceeds_total(["x" * 40, "y" * 40, None, ""], 100) is None
    reason = exceeds_total(["x" * 60, "y" * 60], 100)
    assert reason and "100" in reason and "120" in reason
    # No limit configured is no limit.
    assert exceeds_total(["x" * 10_000], 0) is None


def test_what_a_judge_may_be_shown_fits_the_window_it_has():
    """The sum of the per-field caps, not each cap on its own.

    Twice now a per-field limit has been set without checking what they add up
    to, and a request that overruns the judge's context window is rejected
    outright -- the sample gets no verdict at all, which is strictly worse than
    one read from a clipped copy. The reasoning judge's four fields at its chain
    limit came to 240,000 characters against a window that holds about 178,000
    at this suite's densest.

    The density is measured, not assumed: tokenised with gpt-oss-20b's own
    tokenizer over real responses from 13 datasets, this suite runs 2.72
    characters per token at its worst (abd's s-expressions) against a 4.57
    median. Using the worst is the point -- a limit that only holds for average
    text is not a limit.
    """
    from abductionbench.core.config import JudgeConfig, ReasoningJudgeConfig

    WINDOW = 65_536          # both judges are served with this context
    DENSEST = 2.72           # measured chars per token, worst dataset
    SCAFFOLD = 6_000         # template wording, criteria, gold, observation

    def tokens(chars: int) -> float:
        return chars / DENSEST

    answer = JudgeConfig()
    used = (
        tokens(answer.max_prompt_chars + answer.max_response_chars)
        + SCAFFOLD
        + answer.max_tokens
    )
    assert used < WINDOW, (
        f"the answer judge can be sent {used:,.0f} tokens against a {WINDOW:,} window; "
        "lower max_prompt_chars/max_response_chars, not max_tokens"
    )

    reasoning = ReasoningJudgeConfig()
    # question + chain at the chain limit, answer + reference at theirs.
    used = (
        tokens(2 * reasoning.max_chain_chars + 2 * reasoning.max_reference_chars)
        + SCAFFOLD
        + reasoning.max_tokens
    )
    assert used < WINDOW, (
        f"the reasoning judge can be sent {used:,.0f} tokens against a {WINDOW:,} window"
    )


def test_output_budget_is_not_the_constraint_on_a_judge():
    """A verdict is short; reserving output tokens only starves the input.

    Measured on the worst-case exchange -- both clips maxed, 80,338 characters --
    the judge spent 62 of its 2,048 completion tokens and finished cleanly. The
    instinct to raise max_tokens when a judge returns nothing is what produced
    the 24,000-token setting that cut the input to 41,536 and had requests
    rejected for length; the cause there was reasoning_effort, not room.
    """
    from abductionbench.core.config import JudgeConfig, ReasoningJudgeConfig

    for config in (JudgeConfig(), ReasoningJudgeConfig()):
        assert config.reasoning_effort == "low", (
            "a judge that thinks without bound is what exhausts an output budget; "
            "the budget is not the fix"
        )
        # Generous against a 62-token verdict, and small against the window.
        assert config.max_tokens <= 8192


def test_an_exchange_too_big_for_the_window_is_skipped_not_clipped():
    """A verdict on a cut-down exchange is not the same measurement.

    Clipping keeps the number and loses the fact that it was computed on less --
    and it is then averaged in beside verdicts read from whole exchanges, with
    nothing downstream able to tell them apart. Skipping loses the number and
    keeps the fact, which is the honest trade: the record is counted and named
    in the coverage report.
    """
    from abductionbench.core.judge import exceeds_budget

    limits = {"full_prompt": 100, "full_response": 100}
    assert exceeds_budget({"full_prompt": "x" * 50, "full_response": "y" * 50}, limits) is None
    reason = exceeds_budget({"full_prompt": "x" * 500, "full_response": "y"}, limits)
    assert reason and "full_prompt" in reason and "500" in reason
    # A missing field is not an oversize field.
    assert exceeds_budget({}, limits) is None
