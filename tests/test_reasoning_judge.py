"""The reasoning-metric judge: what it measures, and what it refuses to.

Every metric that is about something happening *across* a chain is asked for as
a per-step list rather than a total, so the distribution survives to be plotted
and not just its mean.  Metric 2 segments the chain once and every other list is
indexed by that segmentation, which is why a list of the wrong length is an
error here rather than something to pad.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from abductionbench.core.reasoning_judge import (
    REASONING_LIST_COLUMNS,
    REASONING_METRIC_COLUMNS,
    ReasoningJudgeStage,
    comparison_pairs,
    derive_reasoning_metrics,
)

JUDGE_PROMPTS = Path("configs/prompts/judge")

#: Four steps, so every per-step list below has to have four elements.
STEPS = ["read the rash", "consider measles", "consider rubella", "settle on measles"]


def _raw(**overrides):
    """A complete, self-consistent set of judge outputs for a generation task."""
    base = {
        "observation_inventory": {"observations": ["rash", "fever", "cough"]},
        "steps": {
            "steps": list(STEPS),
            "proof_disproof_counts": [0, 1, 1, 2],
            "backtracking_steps": 1,
        },
        "observation_coverage": {"observations_per_step": [1, 1, 0, 1]},
        "branchiness_generation": {"branchiness_per_step": [0, 1, 1, 0], "diversity": 1},
        "directionality": {"directionality": 0.5},
        "step_directionality": {"directionality_per_step": [1, 0.5, 0.5, 1]},
        "differential_elimination": {"comparisons_per_step": [0, 0, 0, 1]},
        "uncertainty": {"uncertainty_per_step": [0, 1, 1, 0]},
        "prior_knowledge": {"prior_knowledge_per_step": [0, 2, 1, 0]},
        "anchoring_point": {"anchoring_step_index": 1},
        "unresolved_contradiction": {"unresolved_per_step": [0, 0, 1, 0]},
    }
    base.update(overrides)
    return base


def _derive(raw=None, **kwargs):
    kwargs.setdefault("generation_like", True)
    kwargs.setdefault("selection_like", False)
    kwargs.setdefault("option_count", 0)
    return derive_reasoning_metrics(raw if raw is not None else _raw(), **kwargs)


# --------------------------------------------------------------------------- #
# the numbers themselves
# --------------------------------------------------------------------------- #


def test_every_derived_value_follows_from_the_lists():
    """Each normalization is this code's arithmetic, never the judge's."""
    metrics, lists, errors, _inapplicable = _derive()
    assert not errors, errors

    # Metric 2 -- the segmentation, and what follows from the proof list.
    assert lists["reasoning_steps"] == STEPS
    assert lists["reasoning_proof_disproof_per_step"] == [0, 1, 1, 2]
    assert metrics["reasoning_useless_steps"] == 1.0      # one zero in the list
    assert metrics["reasoning_useful_steps"] == 3.0
    assert metrics["reasoning_useful_step_fraction"] == 0.75
    assert metrics["reasoning_useless_step_fraction"] == 0.25
    assert metrics["reasoning_backtracking_rate"] == 0.25  # 1 of 4 steps

    # Metric 1 -- sum of the per-step list over the inventory's size.
    assert metrics["reasoning_observations_total"] == 3.0
    assert metrics["reasoning_observations_used"] == 3.0
    assert metrics["reasoning_observation_coverage"] == 1.0

    # Metrics 3, 5, 6, 8, 9, 11 -- each list's own aggregate.
    assert metrics["reasoning_branchiness_total"] == 2.0
    assert metrics["reasoning_step_directionality_mean"] == 0.75
    assert metrics["reasoning_differential_elimination"] == 1.0
    assert metrics["reasoning_differential_elimination_normalized"] == 0.25
    assert metrics["reasoning_uncertainty_steps"] == 2.0
    assert metrics["reasoning_uncertainty_rate"] == 0.5
    assert metrics["reasoning_prior_knowledge"] == 3.0
    assert metrics["reasoning_prior_knowledge_normalized"] == 0.75
    assert metrics["reasoning_unresolved_contradictions"] == 1.0
    assert metrics["reasoning_unresolved_contradiction_normalized"] == 0.25

    # Metric 10 -- an index into the step list, and where it falls in the chain.
    assert metrics["reasoning_anchoring_point"] == 1.0
    assert metrics["reasoning_anchoring_point_normalized"] == 0.25


def test_the_step_count_is_never_stored_only_the_steps():
    """Its length is the count; a second copy could disagree with the first."""
    metrics, lists, _errors, _inapplicable = _derive()
    assert "reasoning_total_steps" not in metrics
    assert "reasoning_total_steps" not in REASONING_METRIC_COLUMNS
    assert len(lists["reasoning_steps"]) == 4


def test_exhaustiveness_is_measured_against_pairs():
    """C(n, 2): a chain that has weighed every pair has compared exhaustively."""
    assert comparison_pairs(4) == 6
    assert comparison_pairs(3) == 3
    assert comparison_pairs(2) == 1
    # Nothing to pair.
    assert comparison_pairs(1) == 0
    assert comparison_pairs(0) == 0


def test_a_per_step_list_of_the_wrong_length_is_refused():
    """Padding it would file one step's count against another step."""
    metrics, lists, errors, _inapplicable = _derive(
        _raw(uncertainty={"uncertainty_per_step": [0, 1]})
    )
    assert "uncertainty:missing_or_not_one_value_per_step" in errors
    assert not [key for key in metrics if "uncertainty" in key]
    assert "reasoning_uncertainty_per_step" not in lists
    # ...and it costs that metric alone.
    assert metrics["reasoning_prior_knowledge"] == 3.0


def test_without_a_segmentation_nothing_per_step_can_be_asked():
    """Metric 2 is a dependency, so its failure is reported once, not nine times."""
    metrics, lists, errors, inapplicable = _derive(_raw(steps={"backtracking_steps": 0}))
    assert "steps:invalid_or_missing_step_list" in errors
    assert not [key for key in lists if key.endswith("_per_step")]
    assert not [key for key in metrics if "uncertainty" in key]
    # Each dependent family says why it could not be computed.
    assert len([entry for entry in inapplicable if "no_step_list" in entry]) >= 7
    # The one metric that does not need the segmentation still lands.
    assert metrics["reasoning_directionality"] == 0.5


def test_selection_and_generation_ask_different_branchiness_questions():
    """Diversity is the model's candidates; option count is the question's."""
    generation, _l, _e, gen_inapplicable = _derive()
    assert generation["reasoning_diversity"] == 1.0
    assert "reasoning_option_count" not in generation
    assert any("option_count" in entry for entry in gen_inapplicable)

    selection_raw = _raw()
    selection_raw.pop("branchiness_generation")
    selection_raw["branchiness_selection"] = {
        "branchiness_per_step": [1, 1, 0, 0], "option_count": 4
    }
    selection, _l2, errors, sel_inapplicable = _derive(
        selection_raw, generation_like=False, selection_like=True, option_count=4
    )
    assert not errors, errors
    assert selection["reasoning_option_count"] == 4.0
    assert "reasoning_diversity" not in selection
    assert any("diversity" in entry for entry in sel_inapplicable)
    # One comparison against C(4, 2) = 6 possible pairs.
    assert selection["reasoning_comparison_exhaustiveness"] == pytest.approx(1 / 6)


def test_generation_normalizes_exhaustiveness_by_what_the_model_proposed():
    """No options to pair, so the hypotheses the chain raised are the n."""
    raw = _raw(
        branchiness_generation={"branchiness_per_step": [0, 2, 1, 0], "diversity": 1},
        differential_elimination={"comparisons_per_step": [0, 1, 1, 1]},
    )
    metrics, _lists, errors, _inapplicable = _derive(raw)
    assert not errors, errors
    assert metrics["reasoning_branchiness_total"] == 3.0
    # 3 comparisons against C(3, 2) = 3 pairs.
    assert metrics["reasoning_comparison_exhaustiveness"] == 1.0


def test_a_chain_that_never_reaches_the_answer_is_blank_not_zero():
    """Null and step 0 are different findings about where the model landed."""
    metrics, _lists, errors, inapplicable = _derive(
        _raw(anchoring_point={"anchoring_step_index": None})
    )
    assert not errors, errors
    assert "reasoning_anchoring_point" not in metrics
    assert "reasoning_anchoring_point_normalized" not in metrics
    assert any("never_considered" in entry for entry in inapplicable)

    # An index past the end of the chain is a judge error, not a finding.
    _m, _l, bad, _i = _derive(_raw(anchoring_point={"anchoring_step_index": 9}))
    assert "anchoring_point:not_an_index_into_the_step_list" in bad


def test_an_empty_observation_inventory_blocks_the_ratio_but_keeps_the_counts():
    metrics, _lists, errors, _inapplicable = _derive(
        _raw(observation_inventory={"observations": []})
    )
    assert "observation_inventory:invalid_or_missing_observation_list" in errors
    assert metrics["reasoning_observations_used"] == 3.0
    assert "reasoning_observation_coverage" not in metrics


def test_redundancy_and_completeness_are_gone():
    """Removed outright, not left computing quietly."""
    metrics, lists, _errors, _inapplicable = _derive()
    for column in (*REASONING_METRIC_COLUMNS, *REASONING_LIST_COLUMNS, *metrics, *lists):
        assert "redundancy" not in column, column
        assert "completeness" not in column, column


# --------------------------------------------------------------------------- #
# the prompts
# --------------------------------------------------------------------------- #


def test_one_prompt_per_metric_family_ships():
    from abductionbench.core.config import ReasoningJudgeConfig

    for family, template_id in ReasoningJudgeConfig().templates.items():
        assert (JUDGE_PROMPTS / f"{template_id}.yaml").is_file(), family


def test_no_judge_prompt_mentions_normalization():
    """The judge is asked for raw values and never told what they become.

    A judge that knows a count is about to be divided by the step total has a
    reason to shade the count. Every ratio in this stage is computed in code,
    after the raw values are in hand.
    """
    import re

    # Whole words: "sepa-rate" and "va-ria-tion" are not ratios, and a
    # substring match would flag every prompt that says "separate".
    banned = re.compile(
        r"\b(normalis\w*|normaliz\w*|divide[ds]?|dividing|ratio|ratios|percentage|"
        r"per cent|fraction|fractions|rate|rates|proportion|proportions|average|averaged|"
        r"mean of)\b",
        re.I,
    )
    offenders = []
    for path in sorted(JUDGE_PROMPTS.glob("reasoning_*.yaml")):
        for found in banned.finditer(path.read_text(encoding="utf-8")):
            offenders.append((path.name, found.group(0)))
    assert not offenders, f"judge prompts leaking derived values: {offenders}"


def test_only_two_prompts_see_the_raw_chain():
    """Everything else reads metric 2's segmentation instead.

    That is what makes the per-step lists comparable: one segmentation, and
    every list indexed by it. It also means those judges cannot silently
    re-segment the chain their own way.
    """
    import yaml

    from abductionbench.core.config import ReasoningJudgeConfig

    reads_chain, reads_steps = set(), set()
    for family, template_id in ReasoningJudgeConfig().templates.items():
        blob = yaml.safe_load((JUDGE_PROMPTS / f"{template_id}.yaml").read_text())
        fields = set(blob.get("required_fields") or []) | set(blob.get("optional_fields") or [])
        if "reasoning_chain" in fields:
            reads_chain.add(family)
        if "steps" in fields:
            reads_steps.add(family)
    assert reads_chain == {"steps", "directionality"}, reads_chain
    assert "steps" not in reads_steps, "the segmenter cannot consume its own output"
    # And no prompt gets both, which would let it ignore the segmentation.
    assert not (reads_chain & reads_steps)


def test_the_segmentation_and_its_counts_come_from_one_call():
    """One reading of the chain, or the lists could not be aligned to it."""
    import yaml

    blob = yaml.safe_load((JUDGE_PROMPTS / "reasoning_steps_v2.yaml").read_text())
    declared = set(blob["output_contract"]["json_fields"])
    assert declared == {"steps", "proof_disproof_counts", "backtracking_steps"}


def test_every_judge_prompt_asks_for_exactly_its_declared_fields():
    import yaml

    for path in sorted(JUDGE_PROMPTS.glob("reasoning_*.yaml")):
        blob = yaml.safe_load(path.read_text(encoding="utf-8"))
        rendered = " ".join(message["content"] for message in blob["messages"])
        for field in blob["output_contract"]["json_fields"]:
            assert f'"{field}"' in rendered, f"{path.name} never shows {field}"


def test_the_column_lists_cover_every_value_the_derivation_can_emit():
    """A value with no column is a value nobody sees."""
    metrics, lists, _errors, _inapplicable = _derive()
    for name in metrics:
        assert name in REASONING_METRIC_COLUMNS, name
    for name in lists:
        assert name in REASONING_LIST_COLUMNS, name

    selection_raw = _raw()
    selection_raw.pop("branchiness_generation")
    selection_raw["branchiness_selection"] = {
        "branchiness_per_step": [1, 1, 0, 0], "option_count": 4
    }
    sel_metrics, sel_lists, _e, _i = _derive(
        selection_raw, generation_like=False, selection_like=True, option_count=4
    )
    for name in sel_metrics:
        assert name in REASONING_METRIC_COLUMNS, name
    for name in sel_lists:
        assert name in REASONING_LIST_COLUMNS, name


def test_the_list_columns_and_metric_columns_do_not_overlap():
    assert not set(REASONING_LIST_COLUMNS) & set(REASONING_METRIC_COLUMNS)


@pytest.mark.parametrize(
    "reply,expected",
    [
        ('{"directionality": 1}', {"directionality": 1}),
        ('here you go: {"directionality": 1} hope that helps', {"directionality": 1}),
        ('```json\n{"directionality": 1}\n```', {"directionality": 1}),
    ],
)
def test_the_verdict_is_read_out_of_whatever_the_judge_wrote_around_it(reply, expected):
    class _T:
        output_contract = {"json_fields": {"directionality": "x"}}

    assert ReasoningJudgeStage._parse_json(reply, _T()) == expected


def test_a_reply_without_the_declared_fields_is_unparsed():
    class _T:
        output_contract = {"json_fields": {"steps": "x", "backtracking_steps": "y"}}

    assert ReasoningJudgeStage._parse_json('{"steps": ["a"]}', _T()) is None


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
    """A judge that answers whichever reasoning prompt it was handed.

    Routed on the field each prompt asks for, so it exercises the real
    dependency: the segmentation has to come back before any per-step list can
    be asked for, and every list it returns is four long because the
    segmentation it returned was.
    """
    body = " ".join(str(message.get("content", "")) for message in conversation)
    if '"observations"' in body and '"observations_per_step"' not in body:
        return '{"observations": ["first thing", "second thing"]}'
    if '"steps"' in body and '"proof_disproof_counts"' in body:
        return (
            '{"steps": ["one", "two", "three", "four"], '
            '"proof_disproof_counts": [0, 1, 1, 2], "backtracking_steps": 1}'
        )
    if '"observations_per_step"' in body:
        return '{"observations_per_step": [1, 1, 0, 0]}'
    if '"branchiness_per_step"' in body and '"diversity"' in body:
        return '{"branchiness_per_step": [1, 1, 0, 0], "diversity": 1}'
    if '"branchiness_per_step"' in body:
        return '{"branchiness_per_step": [1, 1, 0, 0], "option_count": 3}'
    if '"directionality_per_step"' in body:
        return '{"directionality_per_step": [1, 1, 0.5, 1]}'
    if '"directionality"' in body:
        return '{"directionality": 1}'
    if '"comparisons_per_step"' in body:
        return '{"comparisons_per_step": [0, 0, 1, 0]}'
    if '"uncertainty_per_step"' in body:
        return '{"uncertainty_per_step": [0, 1, 0, 0]}'
    if '"prior_knowledge_per_step"' in body:
        return '{"prior_knowledge_per_step": [0, 0, 1, 0]}'
    if '"anchoring_step_index"' in body:
        return '{"anchoring_step_index": 2}'
    if '"unresolved_per_step"' in body:
        return '{"unresolved_per_step": [0, 0, 0, 0]}'
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
         "metrics": {"accuracy": 1.0, "reasoning_useful_steps": 5.0}},
        {"sample_id": "b", "prompt_fingerprint": "1",
         "metrics": {"accuracy": 0.0}},                      # judge skipped this one
        {"sample_id": "c", "prompt_fingerprint": "1",
         "metrics": {"accuracy": 1.0, "reasoning_useful_steps": 3.0}},
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
    assert by_metric["reasoning_useful_steps"]["kept"] == 2
    assert by_metric["reasoning_useful_steps"]["skipped"] == 1
    assert by_metric["reasoning_useful_steps"]["coverage"] == round(2 / 3, 4)


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
            _summary_task("cot", {"hypothesis_judged": 0.5, "reasoning_useful_steps": 0.0}),
        ],
    )
    frame = build_summary_frame(result).set_index("prompt_mode")

    for column in REASONING_METRIC_COLUMNS:
        assert pd.isna(frame.loc["io", column]), f"io must be blank in {column}"

    # A genuine zero on a judged task survives as a zero, not a blank.
    assert frame.loc["cot", "reasoning_useful_steps"] == 0.0
    # ...and a column the judge never produced for that task is still blank.
    assert pd.isna(frame.loc["cot", "reasoning_branchiness_total"])
