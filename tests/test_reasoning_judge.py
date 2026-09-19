"""The reasoning-metric judge: what it measures, and what it refuses to."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from abductionbench.core.reasoning_judge import (
    REASONING_METRIC_COLUMNS,
    ReasoningJudgeStage,
    comparison_combinations,
    derive_reasoning_metrics,
)

JUDGE_PROMPTS = Path("configs/prompts/judge")


def _raw(**overrides):
    """A complete, self-consistent set of judge outputs."""
    base = {
        "observation_inventory": {"total_observations": 8},
        # Metrics 1 and 4 in one call: used = redundancy + completeness.
        "evidence": {
            "total_observations": 8,
            "observations_used": 6,
            "redundancy": 2,
            "completeness": 4,
        },
        "steps": {
            "total_steps": 10,
            "useful_steps": 7,
            "useless_steps": 3,
            "backtracking_steps": 2,
        },
        "branchiness_diversity": {"branchiness": 4, "diversity": 1},
        "directionality": {"directionality": 0.5},
        "differential_elimination": {"differential_elimination": 3},
        "uncertainty": {"uncertainty_steps": 3},
        "prior_knowledge": {"prior_knowledge": 1},
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
# the numbers themselves
# --------------------------------------------------------------------------- #


def test_every_raw_and_derived_value_is_reported():
    metrics, errors, inapplicable = derive_reasoning_metrics(
        _raw(), generation_like=False, selection_like=True, option_count=3
    )
    assert not errors
    assert not inapplicable or inapplicable == ["branchiness_diversity:generation_only"]
    # metric 1
    assert metrics["reasoning_observations_total"] == 8
    assert metrics["reasoning_observations_used"] == 6
    assert metrics["reasoning_observation_coverage"] == 6 / 8
    # metric 2: four raw counts and three derived values
    assert metrics["reasoning_total_steps"] == 10
    assert metrics["reasoning_useful_steps"] == 7
    assert metrics["reasoning_useless_steps"] == 3
    assert metrics["reasoning_backtracking_steps"] == 2
    assert metrics["reasoning_useful_step_fraction"] == 0.7
    assert metrics["reasoning_useless_step_fraction"] == 0.3
    assert metrics["reasoning_backtracking_rate"] == 0.2
    # metric 4 normalizes on metric 1's per-sample total, not on its own counts
    assert metrics["reasoning_redundancy_normalized"] == 2 / 8
    assert metrics["reasoning_completeness_normalized"] == 4 / 8
    # metric 5, 6, 7, 8
    assert metrics["reasoning_directionality"] == 0.5
    assert metrics["reasoning_differential_elimination"] == 3
    assert metrics["reasoning_differential_elimination_normalized"] == 3 / 4
    assert metrics["reasoning_uncertainty_steps"] == 3
    assert metrics["reasoning_uncertainty_rate"] == 0.3
    assert metrics["reasoning_prior_knowledge"] == 1


def test_the_combination_count_is_every_subset_of_two_or_more():
    # The specification's own worked example: C(3,2) + C(3,3) = 3 + 1 = 4.
    assert comparison_combinations(3) == 4
    assert comparison_combinations(2) == 1
    assert comparison_combinations(4) == 11
    assert comparison_combinations(1) == 0


def test_generation_gets_branchiness_and_selection_gets_elimination():
    generation, _e, gen_inapplicable = derive_reasoning_metrics(
        _raw(), generation_like=True, selection_like=False, option_count=0
    )
    assert "reasoning_branchiness" in generation
    assert "reasoning_diversity" in generation
    assert "reasoning_differential_elimination" not in generation
    assert gen_inapplicable == ["differential_elimination:selection_and_pipeline_only"]

    selection, _e, sel_inapplicable = derive_reasoning_metrics(
        _raw(), generation_like=False, selection_like=True, option_count=3
    )
    assert "reasoning_differential_elimination" in selection
    assert "reasoning_branchiness" not in selection
    assert sel_inapplicable == ["branchiness_diversity:generation_only"]


def test_a_pipeline_task_gets_both_generation_and_selection_families():
    """A benchmark that generates and selects in one task does both things."""
    metrics, errors, inapplicable = derive_reasoning_metrics(
        _raw(),
        generation_like=False,
        selection_like=False,
        pipeline_like=True,
        option_count=3,
    )
    assert not errors
    assert not inapplicable
    assert "reasoning_branchiness" in metrics
    assert "reasoning_differential_elimination" in metrics


# --------------------------------------------------------------------------- #
# what it refuses: an impossible count is dropped, never coerced
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "gone", "error"),
    [
        # More backtracks than steps is a judge that did not count. One reply of
        # "147" against a 14-step chain moved this metric's mean fifty-fold.
        (
            _raw(steps={"total_steps": 4, "useful_steps": 2, "useless_steps": 2,
                        "backtracking_steps": 9}),
            "reasoning_backtracking_steps",
            "steps:backtracking_exceeds_total_steps",
        ),
        (
            _raw(evidence={"total_observations": 8, "observations_used": 99,
                           "redundancy": 50, "completeness": 49}),
            "reasoning_observations_used",
            "evidence:used_exceeds_inventory_total",
        ),
        (
            _raw(steps={"total_steps": 10, "useful_steps": 7, "useless_steps": 9,
                        "backtracking_steps": 1}),
            "reasoning_total_steps",
            "steps:invalid_counts_or_sum",
        ),
        (
            _raw(uncertainty={"uncertainty_steps": 40}),
            "reasoning_uncertainty_steps",
            "uncertainty:exceeds_total_steps",
        ),
        (
            _raw(directionality={"directionality": 0.7}),
            "reasoning_directionality",
            "directionality:expected_0_0.5_or_1",
        ),
        (
            _raw(evidence={"total_observations": 8, "observations_used": 6,
                           "redundancy": 20, "completeness": 1}),
            "reasoning_redundancy_normalized",
            "evidence:redundancy_plus_completeness_is_not_the_used_count",
        ),
        (
            _raw(prior_knowledge={"prior_knowledge": 7}),
            "reasoning_prior_knowledge",
            "prior_knowledge:invalid_or_missing_output",
        ),
    ],
)
def test_an_impossible_count_is_dropped_with_a_reason(raw, gone, error):
    metrics, errors, _inapplicable = derive_reasoning_metrics(
        raw, generation_like=True, selection_like=False, option_count=0
    )
    assert gone not in metrics
    assert error in errors


def test_more_comparisons_than_combinations_is_rejected():
    metrics, errors, _ = derive_reasoning_metrics(
        _raw(differential_elimination={"differential_elimination": 99}),
        generation_like=False,
        selection_like=True,
        option_count=3,
    )
    assert "reasoning_differential_elimination" not in metrics
    assert "differential_elimination:exceeds_possible_combinations" in errors


def test_a_zero_observation_inventory_blocks_the_ratios_but_keeps_the_counts():
    """Dividing by it would be a made-up number; the raw counts still stand."""
    metrics, errors, _ = derive_reasoning_metrics(
        _raw(
            observation_inventory={"total_observations": 0},
            evidence={"total_observations": 0, "observations_used": 0,
                      "redundancy": 0, "completeness": 0},
        ),
        generation_like=True,
        selection_like=False,
        option_count=0,
    )
    assert metrics["reasoning_observations_total"] == 0
    assert metrics["reasoning_observations_used"] == 0
    assert "reasoning_observation_coverage" not in metrics
    assert "reasoning_redundancy_normalized" not in metrics
    assert "evidence:inventory_total_is_zero" in errors


def test_a_missing_family_costs_only_its_own_metrics():
    metrics, errors, _ = derive_reasoning_metrics(
        {**_raw(), "steps": {}},
        generation_like=True,
        selection_like=False,
        option_count=0,
    )
    assert "reasoning_total_steps" not in metrics
    assert "reasoning_uncertainty_rate" not in metrics  # normalizer is gone
    assert metrics["reasoning_uncertainty_steps"] == 3  # the raw count survives
    assert metrics["reasoning_observation_coverage"] == 6 / 8  # untouched
    assert "steps:invalid_counts_or_sum" in errors


# --------------------------------------------------------------------------- #
# the judge prompts
# --------------------------------------------------------------------------- #


def _judge_prompt_files():
    return sorted(JUDGE_PROMPTS.glob("reasoning_*.yaml"))


def test_one_prompt_per_metric_family_ships():
    from abductionbench.core.config import ReasoningJudgeConfig

    ids = {path.stem for path in _judge_prompt_files()}
    assert set(ReasoningJudgeConfig().templates.values()) == ids
    assert len(ids) == 8


def test_no_judge_prompt_mentions_normalization():
    """Raw counts only: every ratio is computed in code, after the fact.

    A judge asked for a ratio has to do arithmetic on its own counts, and the
    sheet can then disagree with its own columns.
    """
    import re

    forbidden = re.compile(
        r"normali|\bdivide|\bdivided\b|\bratio\b|\bfraction\b|\bpercent|\bper step\b|÷",
        re.IGNORECASE,
    )
    offenders = []
    for path in _judge_prompt_files():
        for number, line in enumerate(path.read_text().splitlines(), 1):
            found = forbidden.search(line)
            if found:
                offenders.append((path.name, number, found.group(0)))
    assert not offenders, f"judge prompt mentions normalization: {offenders}"


def test_every_judge_prompt_asks_for_exactly_its_declared_fields():
    """The contract the parser enforces has to be the one the prompt asks for."""
    from abductionbench.core.config import load_yaml

    for path in _judge_prompt_files():
        blob = load_yaml(path)
        declared = set((blob.get("output_contract") or {}).get("json_fields") or {})
        assert declared, path.name
        body = "\n".join(message["content"] for message in blob["messages"])
        for name in declared:
            assert f'"{name}"' in body, (path.name, name)


def test_the_step_counts_are_one_prompt_and_not_four():
    """Four counts from four calls would be four different segmentations."""
    from abductionbench.core.config import load_yaml

    blob = load_yaml(JUDGE_PROMPTS / "reasoning_steps_v1.yaml")
    assert set((blob.get("output_contract") or {}).get("json_fields")) == {
        "total_steps",
        "useless_steps",
        "useful_steps",
        "backtracking_steps",
    }


# --------------------------------------------------------------------------- #
# parsing a judge that thinks out loud
# --------------------------------------------------------------------------- #


class _Template:
    ref = "t@1"
    output_contract = {"json_fields": {"total_steps": "nonnegative_integer"}}


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ('{"total_steps": 3}', 3),
        ('```json\n{"total_steps": 4}\n```', 4),
        ("Let me think. First... Therefore:\n{\"total_steps\": 5}", 5),
        # An echoed example followed by the real verdict: the last one wins.
        ('{"total_steps": <integer>} ... my answer: {"total_steps": 6}', 6),
        ('I counted {a} steps. {"total_steps": 7}', 7),
    ],
)
def test_the_verdict_is_read_out_of_whatever_the_judge_wrote_around_it(reply, expected):
    parsed = ReasoningJudgeStage._parse_json(reply, _Template())
    assert parsed is not None and parsed["total_steps"] == expected


def test_a_reply_without_the_declared_fields_is_unparsed():
    assert ReasoningJudgeStage._parse_json('{"something_else": 1}', _Template()) is None
    assert ReasoningJudgeStage._parse_json("no json at all", _Template()) is None


# --------------------------------------------------------------------------- #
# the stage as a whole
# --------------------------------------------------------------------------- #


def test_the_column_list_covers_every_metric_the_derivation_can_emit():
    metrics, _e, _i = derive_reasoning_metrics(
        _raw(), generation_like=True, selection_like=True, option_count=3
    )
    assert set(metrics) <= set(REASONING_METRIC_COLUMNS)
    assert len(REASONING_METRIC_COLUMNS) == len(set(REASONING_METRIC_COLUMNS))


def test_io_outputs_are_never_judged(tmp_path):
    """The guard that makes these columns mean what they say.

    An io output has no chain of reasoning in it, so a number measured over one
    would be a number about the answer line.
    """
    from abductionbench.core.config import ReasoningJudgeConfig
    from abductionbench.core.types import (
        ModelResponse,
        ResponseStatus,
        SampleScore,
        SampleSpec,
        TaskIdentity,
    )

    class _Boom:
        """Any call to the judge at all is the failure this test looks for."""

        supports_batch = False

        def chat_single(self, *args, **kwargs):  # pragma: no cover - must not run
            raise AssertionError("the reasoning judge called the model for an io output")

        chat_batch = chat_single

    stage = ReasoningJudgeStage(
        config=ReasoningJudgeConfig(enabled=True, model="m"),
        registry=_FakeRegistry(),
        renderer=None,
        clients={"m": _Boom()},
        retry_policy=None,
        cache_dir=tmp_path,
    )
    sample = SampleSpec(sample_id="s1", fields={"observation": "x"}, task_kind="generation")
    scored = [
        (
            sample,
            ModelResponse(sample_id="s1", model_id="m", status=ResponseStatus.OK,
                          content="Answer: something"),
            SampleScore(metrics={"match": 1.0}),
        )
    ]
    identity = TaskIdentity(
        run_id="r", dataset_id="d", model_id="m", template_id="t", template_version="1.0",
        prompt_mode="io", selection_mode="n/a", data_delivery_mode="static",
        task_kind="generation",
    )
    out = asyncio.run(stage.apply(_FakeAdapter(), identity, [], scored))
    assert out is scored
    assert not any(key.startswith("reasoning_") for key in out[0][2].metrics)


class _FakeRegistry:
    def get(self, template_id):
        class _T:
            ref = f"{template_id}@1.0"
            id = template_id
            version = "1.0"
            required_fields: list[str] = []
            output_contract: dict = {}

        return _T()


class _FakeAdapter:
    dataset_id = "d"

    def documentation(self):
        raise RuntimeError("not needed")


# --------------------------------------------------------------------------- #
# in the engine, end to end
# --------------------------------------------------------------------------- #


def _reasoning_responder(conversation, max_tokens):
    """A judge that answers whichever reasoning prompt it was handed."""
    body = " ".join(str(message.get("content", "")) for message in conversation)
    if "total_observations" in body and "observations_used" not in body:
        return '{"total_observations": 2}'
    if "observations_used" in body:
        # One call for metrics 1 and 4; used must equal redundancy + completeness.
        return (
            '{"total_observations": 2, "observations_used": 2, '
            '"redundancy": 1, "completeness": 1}'
        )
    if "backtracking_steps" in body:
        return (
            '{"total_steps": 4, "useless_steps": 1, "useful_steps": 3, '
            '"backtracking_steps": 1}'
        )
    if "branchiness" in body:
        return '{"branchiness": 2, "diversity": 1}'
    if "directionality" in body:
        return '{"directionality": 1}'
    if "differential_elimination" in body:
        return '{"differential_elimination": 1}'
    if "uncertainty_steps" in body:
        return '{"uncertainty_steps": 2}'
    if "prior_knowledge" in body:
        return '{"prior_knowledge": 0}'
    return "Answer: something"


def test_the_engine_adds_the_columns_to_cot_and_leaves_io_alone(
    fake_server, write_run_config, tmp_path, monkeypatch
):
    """The wiring, not the arithmetic: a real run, both prompt modes, one pass.

    io and cot are planned as separate tasks off the same records, so this also
    pins the guarantee the whole stage rests on -- the reasoning columns exist
    on one and not on the other.
    """
    import asyncio
    import json

    from abductionbench.core.config import load_run_config
    from abductionbench.core.engine import EvaluationEngine

    fake_server.state.responder = _reasoning_responder

    adapter_module = tmp_path / "reasoned_adapter.py"
    adapter_module.write_text(
        '''
from typing import Sequence

from abductionbench.core.adapter import DatasetAdapter
from abductionbench.core.metrics import aggregate_mean_metrics
from abductionbench.core.types import AdapterDocumentation, ChatMessage, SampleScore, SampleSpec


class ReasonedAdapter(DatasetAdapter):
    primary_metric = "accuracy"
    system_prompt = "You explain observations."
    objective_metrics = True

    def build_messages(self, sample):
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
                sample_id=f"r{i}",
                fields={"observation": f"the grass is wet, case {i}"},
                reference="the sprinkler ran",
                max_tokens=64,
            )
            for i in range(2)
        ]

    def score(self, sample, response, *, output_contract=None):
        return SampleScore(metrics={"accuracy": 1.0}, prediction=response.text[:50])

    def aggregate(self, scores: Sequence[SampleScore]):
        return aggregate_mean_metrics([s.metrics for s in scores])

    def documentation(self):
        return AdapterDocumentation(
            dataset_id=self.dataset_id, name="reasoned", domain="test",
            source_url="n/a", processing_mode="Generation", primary_metric="accuracy",
        )
''',
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    config_path = write_run_config(
        base_url=fake_server.base_url,
        datasets=[
            {"id": "reasoned", "impl": "reasoned_adapter:ReasonedAdapter", "sample_size": 2}
        ],
        engine={
            "reasoning_judge": {
                "enabled": True,
                "model": "fake-model",
                "group_size": 4,
                "max_parallel_calls": 4,
                "max_tokens": 64,
            }
        },
        modes={"prompt_modes": ["io", "cot"]},
    )
    config = load_run_config(config_path)
    engine = EvaluationEngine(config)
    result = asyncio.run(engine.run())

    by_mode = {task.identity.prompt_mode: task for task in result.tasks}
    assert set(by_mode) == {"io", "cot"}

    def _metric_names(task):
        records = [
            json.loads(line)
            for line in (task.output_dir / "records.jsonl").read_text().splitlines()
            if line.strip()
        ]
        return {
            name
            for record in records
            for name in (record.get("metrics") or {})
            if name.startswith("reasoning_")
        }

    cot_metrics = _metric_names(by_mode["cot"])
    assert "reasoning_observation_coverage" in cot_metrics
    assert "reasoning_backtracking_rate" in cot_metrics
    assert "reasoning_useful_step_fraction" in cot_metrics
    # io is untouched: no chain, so no columns.
    assert _metric_names(by_mode["io"]) == set()

    # ...and the standalone log lands inside the run directory, which is what
    # engine.sync mirrors off-box.
    log = by_mode["cot"].output_dir.parents[3] / "reasoning_metrics.jsonl"
    assert log.exists()
    lines = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    assert lines and all(entry["prompt_mode"] == "cot" for entry in lines)
    assert lines[0]["raw"]["steps"]["backtracking_steps"] == 1
    assert lines[0]["metrics"]["reasoning_backtracking_rate"] == 0.25


def test_one_question_is_bought_once_even_when_tasks_ask_it_together():
    """Deduplication must not cost the concurrency it is there to protect.

    Holding a lock across the whole inventory step buys each question once, but
    it also queues every task's first wave behind every other task's. This
    checks both halves: the same question is fetched once, and two tasks asking
    *different* questions are in flight at the same time.
    """
    import tempfile

    from abductionbench.core.config import ReasoningJudgeConfig

    fetched: list[tuple[str, ...]] = []
    in_flight = 0
    peak = 0

    class _Stage(ReasoningJudgeStage):
        async def _judge_many(self, family, requests, *, identity=None, context=None):
            nonlocal in_flight, peak
            fetched.append(tuple(sorted(requests)))
            in_flight += 1
            peak = max(peak, in_flight)
            try:
                await asyncio.sleep(0.05)
                return {key: {"total_observations": 1} for key in requests}
            finally:
                in_flight -= 1

    async def run():
        with tempfile.TemporaryDirectory() as directory:
            stage = _Stage(
                config=ReasoningJudgeConfig(enabled=True, model="m"),
                registry=_FakeRegistry(),
                renderer=None,
                clients={"m": object()},
                retry_policy=None,
                cache_dir=Path(directory),
            )
            shared = {"q-shared": {"question": "same"}}
            other = {"q-other": {"question": "different"}}
            return await asyncio.gather(
                stage._inventories(dict(shared)),
                stage._inventories(dict(shared)),
                stage._inventories(dict(other)),
            )

    first, second, third = asyncio.run(run())
    assert first == second == {"q-shared": {"total_observations": 1}}
    assert third == {"q-other": {"total_observations": 1}}
    # One fetch for the shared question, one for the other -- not three.
    assert sorted(fetched) == [("q-other",), ("q-shared",)]
    # ...and they overlapped rather than queueing.
    assert peak == 2


def test_a_failed_inventory_never_leaves_another_task_waiting():
    """A claimed question must be resolved even when its fetch blows up."""
    import tempfile

    from abductionbench.core.config import ReasoningJudgeConfig

    class _Stage(ReasoningJudgeStage):
        calls = 0

        async def _judge_many(self, family, requests, *, identity=None, context=None):
            type(self).calls += 1
            if type(self).calls == 1:
                await asyncio.sleep(0.05)
                raise RuntimeError("judge exploded")
            return {key: {"total_observations": 3} for key in requests}

    async def run():
        with tempfile.TemporaryDirectory() as directory:
            stage = _Stage(
                config=ReasoningJudgeConfig(enabled=True, model="m"),
                registry=_FakeRegistry(),
                renderer=None,
                clients={"m": object()},
                retry_policy=None,
                cache_dir=Path(directory),
            )
            request = {"q": {"question": "same"}}
            owner = asyncio.create_task(stage._inventories(dict(request)))
            await asyncio.sleep(0.01)  # let the owner claim the key
            waiter = asyncio.create_task(stage._inventories(dict(request)))
            results = await asyncio.gather(owner, waiter, return_exceptions=True)
            return results, stage

    results, stage = asyncio.run(asyncio.wait_for(run(), timeout=5))
    assert isinstance(results[0], RuntimeError)
    # The waiter is released with "no inventory" rather than hanging forever.
    assert results[1] == {"q": None}
    assert not stage._inventories_inflight


# --------------------------------------------------------------------------- #
# cost: a verdict is a reading task, not a research budget
# --------------------------------------------------------------------------- #


def test_the_judge_is_told_how_hard_to_think(tmp_path):
    """reasoning_effort rides on every call, and is omitted when unset.

    Measured on this suite's chains at the same budget: default effort 30.2s /
    1,484 completion tokens, low effort 4.0s / 231 -- same verdict. On a long
    chain the default runs to the budget and returns empty content, losing the
    call: 793 of one run's 802 unparseable replies were that.
    """
    from abductionbench.core.config import ReasoningJudgeConfig

    def stage(**overrides):
        return ReasoningJudgeStage(
            config=ReasoningJudgeConfig(enabled=True, model="m", **overrides),
            registry=_FakeRegistry(),
            renderer=None,
            clients={"m": object()},
            retry_policy=None,
            cache_dir=tmp_path,
        )

    assert stage()._sampling_extra() == (("reasoning_effort", "low"),)
    assert stage(reasoning_effort="high")._sampling_extra() == (("reasoning_effort", "high"),)
    assert stage(reasoning_effort=None)._sampling_extra() == ()


def test_an_over_long_chain_is_clipped_from_the_middle(tmp_path):
    """Better to lose the middle of a chain than the whole call.

    A chain plus its question can exceed the judge's own context window -- seen
    at 65,621 tokens against a 65,536 limit -- and the request is then rejected
    outright, so the sample gets no metrics at all. Both ends are kept: the
    opening says what the model set out to do, the close is where it commits,
    and every metric here reads one or both.
    """
    from abductionbench.core.config import ReasoningJudgeConfig

    stage = ReasoningJudgeStage(
        config=ReasoningJudgeConfig(enabled=True, model="m", max_chain_chars=1000),
        registry=_FakeRegistry(),
        renderer=None,
        clients={"m": object()},
        retry_policy=None,
        cache_dir=tmp_path,
    )
    chain = "START" + ("x" * 5000) + "END"
    clipped = stage._clip_chain(chain)
    assert len(clipped) < len(chain)
    assert clipped.startswith("START")
    assert clipped.endswith("END")
    assert "omitted" in clipped, "the cut must be marked, not silent"
    assert stage.stats["clipped"] == 1
    # A chain that fits is returned untouched.
    assert stage._clip_chain("short") == "short"


def test_coverage_counts_what_each_metric_was_computed_over(tmp_path):
    """A mean says what the scored records were worth, not how many there were.

    They are different questions the moment a metric can be absent -- an
    exchange too big to judge, a chain no step count could be read from, a
    metric that does not apply to the task -- and reading the first without the
    second is how a number computed over a third of a dataset gets quoted as
    the dataset's score.
    """
    import json

    from abductionbench.core.engine import RunResult, TaskResult
    from abductionbench.core.reporting import build_coverage_frame
    from abductionbench.core.types import TaskIdentity

    task_dir = tmp_path / "datasets" / "d" / "m" / "cot"
    task_dir.mkdir(parents=True)
    records = [
        {"sample_id": "a", "prompt_fingerprint": "1",
         "metrics": {"accuracy": 1.0, "reasoning_total_steps": 5.0}},
        {"sample_id": "b", "prompt_fingerprint": "1",
         "metrics": {"accuracy": 0.0}},                      # judge skipped this one
        {"sample_id": "c", "prompt_fingerprint": "1",
         "metrics": {"accuracy": 1.0, "reasoning_total_steps": 3.0}},
    ]
    (task_dir / "records.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )
    identity = TaskIdentity(
        run_id="r", dataset_id="d", model_id="m", template_id="cot", template_version="1.0",
        prompt_mode="cot", selection_mode="n/a", task_kind="generation",
        data_delivery_mode="static",
    )
    result = RunResult(run_id="r", run_dir=tmp_path, config=None, started_at=0.0)
    result.tasks = [TaskResult(identity=identity, output_dir=task_dir)]

    frame = build_coverage_frame(result)
    by_metric = {row["metric"]: row for _, row in frame.iterrows()}
    assert by_metric["accuracy"]["kept"] == 3
    assert by_metric["accuracy"]["skipped"] == 0
    assert by_metric["accuracy"]["coverage"] == 1.0
    # The metric one record never produced is visible as a gap, not as a zero.
    assert by_metric["reasoning_total_steps"]["kept"] == 2
    assert by_metric["reasoning_total_steps"]["skipped"] == 1
    assert by_metric["reasoning_total_steps"]["coverage"] == round(2 / 3, 4)


def test_the_evidence_counts_must_partition_the_observations_used():
    """What merging metrics 1 and 4 into one call buys.

    Every observation the chain used is either removable or necessary, so
    redundancy and completeness partition the used count. As two separate calls
    this could not be checked at all: each read the evidence on its own, and a
    used count of 6 sitting beside a redundancy of 2 and a completeness of 5 was
    simply reported, because nothing knew the two readings were meant to be the
    same one.
    """
    metrics, errors, _ = derive_reasoning_metrics(
        _raw(evidence={"total_observations": 8, "observations_used": 6,
                       "redundancy": 2, "completeness": 5}),   # 2 + 5 != 6
        generation_like=True, selection_like=False, option_count=0,
    )
    assert "evidence:redundancy_plus_completeness_is_not_the_used_count" in errors
    assert "reasoning_redundancy" not in metrics
    assert "reasoning_completeness" not in metrics
    # The used count itself still stands: it is not what failed.
    assert metrics["reasoning_observations_used"] == 6

    ok, errors, _ = derive_reasoning_metrics(
        _raw(evidence={"total_observations": 8, "observations_used": 6,
                       "redundancy": 2, "completeness": 4}),
        generation_like=True, selection_like=False, option_count=0,
    )
    assert not errors
    assert ok["reasoning_redundancy"] == 2 and ok["reasoning_completeness"] == 4


# --------------------------------------------------------------------------- #
# audit: why did the judge say that?
# --------------------------------------------------------------------------- #


def _audit_stage(tmp_path, client, **overrides):
    """A stage wired to write its audit log under ``tmp_path``."""
    from abductionbench.core.config import ReasoningJudgeConfig
    from abductionbench.core.types import ChatMessage

    class _Renderer:
        def render(self, sample, template):
            return (
                [
                    ChatMessage(role="system", content="judge system"),
                    ChatMessage(role="user", content=f"CHAIN>>> {sample.fields}"),
                ],
                {},
            )

    return ReasoningJudgeStage(
        config=ReasoningJudgeConfig(enabled=True, model="judge-m", cache=False, **overrides),
        registry=_FakeRegistry(),
        renderer=_Renderer(),
        clients={"judge-m": client},
        retry_policy=None,
        cache_dir=tmp_path / "cache",
        run_dir=tmp_path / "run",
    )


def _identity():
    from abductionbench.core.types import TaskIdentity

    return TaskIdentity(
        run_id="run-1", dataset_id="dset", model_id="model-under-test",
        template_id="cot_n-a_static", template_version="1.0", prompt_mode="cot",
        selection_mode="n/a", data_delivery_mode="static", task_kind="generation",
    )


def _audit_lines(tmp_path):
    import orjson

    path = (
        tmp_path / "run" / "datasets" / "dset" / "model-under-test"
        / "cot_n-a_static@1.0" / ReasoningJudgeStage.AUDIT_FILENAME
    )
    if not path.exists():
        return []
    return [orjson.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


class _Choice:
    def __init__(self, content, reasoning=None, finish_reason="stop"):
        self.content = content
        self.reasoning = reasoning
        self.finish_reason = finish_reason


class _Result:
    def __init__(self, choices, usage=None):
        self.choices = choices
        self.usage = usage or {"completion_tokens": 7}


def test_every_judge_call_is_logged_whole_with_its_identifiers(tmp_path):
    """The prompt, the answer and enough ids to find the generation it graded.

    reasoning_metrics.jsonl records the parsed numbers, and the verdict cache
    keeps a truncated copy of the reply; neither answers "why did the judge say
    14 steps for this chain". This log does, and it is untruncated on purpose.
    """
    class _Client:
        supports_batch = False

        async def chat_single(self, messages, params):
            return _Result([_Choice('{"total_steps": 14}', reasoning="I counted them.")])

    stage = _audit_stage(tmp_path, _Client())
    long_reply = '{"total_steps": 14}' + "x" * 5000

    async def run():
        async def fake_retry(fn, **kwargs):
            return await fn(), None

        import abductionbench.core.reasoning_judge as rj

        original, rj.with_retry = rj.with_retry, fake_retry
        try:
            return await stage._judge_many(
                "steps",
                {"0": {"question": "q", "reasoning_chain": "c"}},
                identity=_identity(),
                context={"0": {"sample_id": "sample-42", "group_id": "g1", "target_index": 0}},
            )
        finally:
            rj.with_retry = original

    asyncio.run(asyncio.wait_for(run(), timeout=5))
    lines = _audit_lines(tmp_path)
    assert len(lines) == 1, lines
    row = lines[0]

    assert row["schema"] == ReasoningJudgeStage.AUDIT_SCHEMA
    # Identifiers that connect the call to its run, dataset, task and sample.
    assert row["run_id"] == "run-1"
    assert row["dataset_id"] == "dset"
    assert row["model_id"] == "model-under-test", "the evaluated model, not the judge"
    assert row["template_id"] == "cot_n-a_static"
    assert row["prompt_mode"] == "cot"
    assert row["sample_id"] == "sample-42"
    assert row["group_id"] == "g1"
    assert row["metric_family"] == "steps"
    assert row["judge_model"] == "judge-m"
    assert row["judge_template"].startswith("reasoning_steps")
    assert row["cache_key"]

    # The exact submitted messages, not a summary of them.
    sent = row["request"]["messages"]
    assert [m["role"] for m in sent] == ["system", "user"]
    assert "CHAIN>>>" in sent[1]["content"]

    # The complete answer and an explicitly-present reasoning trace.
    assert row["response"]["outcome"] == "ok"
    assert row["response"]["content"] == '{"total_steps": 14}'
    assert row["response"]["reasoning_trace"]["available"] is True
    assert row["response"]["reasoning_trace"]["text"] == "I counted them."
    assert row["parse"]["ok"] is True
    assert row["parse"]["values"] == {"total_steps": 14}
    assert row["parse"]["source"] == "content"
    assert long_reply  # (kept for the next test's contrast)


def test_an_absent_reasoning_trace_is_not_an_empty_one(tmp_path):
    """"The API returned no trace" and "the trace was empty" are different facts."""
    class _Client:
        supports_batch = False

        async def chat_single(self, messages, params):
            return _Result([_Choice('{"total_steps": 3}', reasoning=None)])

    stage = _audit_stage(tmp_path, _Client())

    async def run():
        async def fake_retry(fn, **kwargs):
            return await fn(), None

        import abductionbench.core.reasoning_judge as rj

        original, rj.with_retry = rj.with_retry, fake_retry
        try:
            await stage._judge_many(
                "steps", {"0": {"question": "q"}}, identity=_identity(),
                context={"0": {"sample_id": "s"}},
            )
        finally:
            rj.with_retry = original

    asyncio.run(asyncio.wait_for(run(), timeout=5))
    trace = _audit_lines(tmp_path)[0]["response"]["reasoning_trace"]
    assert trace["available"] is False
    assert trace["text"] is None


def test_unparseable_and_failed_calls_are_kept_not_dropped(tmp_path):
    """The replies worth reading back are precisely the ones that did not work."""
    from abductionbench.core.errors import EndpointError

    class _Unparseable:
        supports_batch = False

        async def chat_single(self, messages, params):
            return _Result([_Choice("I think about nine steps, roughly?")])

    class _Dead:
        supports_batch = False

        async def chat_single(self, messages, params):
            raise EndpointError("connection refused")

    for client, outcome, has_content in (
        (_Unparseable(), "unparseable", True),
        (_Dead(), "call_failed", False),
    ):
        target = tmp_path / outcome
        stage = _audit_stage(target, client)

        async def run(stage=stage):
            async def fake_retry(fn, **kwargs):
                return await fn(), None

            import abductionbench.core.reasoning_judge as rj

            original, rj.with_retry = rj.with_retry, fake_retry
            try:
                await stage._judge_many(
                    "steps", {"0": {"question": "q"}}, identity=_identity(),
                    context={"0": {"sample_id": "s"}},
                )
            finally:
                rj.with_retry = original

        asyncio.run(asyncio.wait_for(run(), timeout=5))
        rows = _audit_lines(target)
        assert len(rows) == 1, f"{outcome}: {rows}"
        row = rows[0]
        assert row["response"]["outcome"] == outcome
        assert row["parse"]["ok"] is False
        if has_content:
            # Kept in full: this is the string that has to be read to see why.
            assert row["response"]["content"] == "I think about nine steps, roughly?"
        else:
            assert row["response"]["error"], "a failed call must say why"
        # Even a failure carries the identifiers that locate the generation.
        assert row["sample_id"] == "s" and row["dataset_id"] == "dset"


def test_the_audit_log_carries_no_credentials(tmp_path):
    """It ships to Drive with the run, so it must not carry a key."""
    class _Client:
        supports_batch = False

        async def chat_single(self, messages, params):
            return _Result([_Choice('{"total_steps": 1}')])

    stage = _audit_stage(tmp_path, _Client())

    async def run():
        async def fake_retry(fn, **kwargs):
            return await fn(), None

        import abductionbench.core.reasoning_judge as rj

        original, rj.with_retry = rj.with_retry, fake_retry
        try:
            await stage._judge_many(
                "steps", {"0": {"question": "q"}}, identity=_identity(),
                context={"0": {"sample_id": "s"}},
            )
        finally:
            rj.with_retry = original

    asyncio.run(asyncio.wait_for(run(), timeout=5))
    import orjson

    blob = orjson.dumps(_audit_lines(tmp_path)).decode().lower()
    for secret in ("api_key", "authorization", "bearer", "vllm-", "hf_", "sk-"):
        assert secret not in blob, f"the audit log leaked {secret!r}"


def test_the_audit_lands_inside_the_run_so_sync_ships_it(tmp_path):
    """Beside the task's own records, which is what engine.sync mirrors."""
    stage = _audit_stage(tmp_path, object())
    path = stage._audit_path(_identity())
    assert path is not None
    assert path.parent.name == "cot_n-a_static@1.0"
    assert path.parent.parent.name == "model-under-test"
    assert path.parent.parent.parent.name == "dset"
    assert path.name == "reasoning_judge_calls.jsonl"
    # No run directory (a bare stage in a test or a probe) writes nothing and
    # raises nothing.
    bare = _audit_stage(tmp_path, object())
    bare.run_dir = None
    assert bare._audit_path(_identity()) is None
    bare._append_audit(_identity(), [{"a": 1}])


# --------------------------------------------------------------------------- #
# the reasoning columns in Summary_Long
# --------------------------------------------------------------------------- #


def _summary_task(prompt_mode, metrics):
    from abductionbench.core.engine import TaskResult
    from abductionbench.core.types import TaskIdentity

    identity = TaskIdentity(
        run_id="r", dataset_id="d", model_id="m", template_id="t", template_version="1.0",
        prompt_mode=prompt_mode, selection_mode="n/a", data_delivery_mode="static",
        task_kind="generation",
    )
    return TaskResult(
        identity=identity, output_dir=Path("/tmp"), metrics=metrics,
        primary_metric="hypothesis_judged",
    )


def test_summary_long_carries_every_reasoning_metric():
    """The averages already exist on the task; the sheet now shows them.

    Taken from the task's own aggregate rather than recomputed from the
    per-sample sheet, so the headline row and the sheet it summarises cannot
    drift apart.
    """
    from abductionbench.core.engine import RunResult
    from abductionbench.core.reporting import build_summary_frame

    judged = {"hypothesis_judged": 0.5}
    judged.update({column: float(index) for index, column in enumerate(REASONING_METRIC_COLUMNS)})
    result = RunResult(
        run_id="r", run_dir=Path("/tmp"), config=None,
        tasks=[_summary_task("cot", judged)],
    )
    frame = build_summary_frame(result)
    for index, column in enumerate(REASONING_METRIC_COLUMNS):
        assert column in frame.columns, column
        assert frame.iloc[0][column] == float(index), column


def test_a_task_without_chains_gets_blanks_not_zeros():
    """Every one of these metrics has a meaningful zero.

    No backtracking, no uncertainty marked, nothing branched -- all real
    findings. Writing 0.0 where nothing was measured would report one. An io
    task has no chain in it, so its cells are empty.
    """
    import pandas as pd

    from abductionbench.core.engine import RunResult
    from abductionbench.core.reporting import build_summary_frame

    result = RunResult(
        run_id="r", run_dir=Path("/tmp"), config=None,
        tasks=[
            _summary_task("io", {"hypothesis_judged": 0.4}),
            _summary_task("cot", {"hypothesis_judged": 0.5, "reasoning_total_steps": 0.0}),
        ],
    )
    frame = build_summary_frame(result).set_index("prompt_mode")

    for column in REASONING_METRIC_COLUMNS:
        assert pd.isna(frame.loc["io", column]), f"io must be blank in {column}"

    # A genuine zero on a judged task survives as a zero, not a blank.
    assert frame.loc["cot", "reasoning_total_steps"] == 0.0
    # ...and a column the judge never produced for that task is still blank.
    assert pd.isna(frame.loc["cot", "reasoning_branchiness"])
