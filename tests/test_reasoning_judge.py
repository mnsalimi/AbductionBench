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

        # The segmentation only segments now. What each step proves, disproves
        # or corrects moved to its own call -- one prompt cutting the chain AND
        # judging it produced a worse cut.
        "steps": {
            "steps": list(STEPS),
            "n_steps": len(STEPS),
        },
        # Wave one: the inventory. Its LENGTH is the total -- the judge lists
        # the observations, it is not asked to count them as well.
        "observation_inventory": {
            "observations": ["the lawn is wet", "the sky is clear", "it is 6am"],
        },
        # Wave two: first-appearance placement against that fixed inventory.
        # The list's own sum is the covered count, so it is not asked twice.
        "observation_coverage": {
            "observations_per_step": [1, 1, 0, 1],
        },
        "branchiness_generation": {"branchiness_per_step": [0, 1, 1, 0], "mentions_per_step": [1, 1, 1, 1], "diversity": 1},
        "directionality": {"directionality": 0.5},
        "step_directionality": {"directionality_per_step": [1, 0.5, 0.5, 1]},
        "differential_elimination": {"comparisons_per_step": [0, 0, 0, 1]},
        "uncertainty": {"uncertainty_per_step": [0, 1, 1, 0]},
        "prior_knowledge": {"prior_knowledge_per_step": [0, 2, 1, 0]},
        "anchoring_point": {"anchoring_step_index": 1,
                            "gold_alive_per_step": [0, 1, 1, 1]},
        # Backtracking lives here now: the two are one judgement split by
        # whether the model noticed its own mistake.
        "unresolved_contradiction": {"unresolved_per_step": [0, 0, 1, 0],
                                     "backtracking_per_step": [0, 0, 1, 0]},
        # Split out of `steps`, which was cutting the chain and judging it in
        # the same call.
        "proof_disproof": {"proof_disproof_per_step": [0, 1, 1, 2]},
        "helpfulness": {"helpfulness_per_step": [0, 1, -1, 1]},
    }
    base.update(overrides)
    return base


def _derive(raw=None, **kwargs):
    kwargs.setdefault("generation_like", True)
    kwargs.setdefault("selection_like", False)
    kwargs.setdefault("option_count", 0)
    return derive_reasoning_metrics(raw if raw is not None else _raw(), **kwargs)


def test_the_normalized_anchoring_point_spans_the_whole_chain():
    """(index + 1) / total_steps: the last step is 1.0, the first is 1 / n.

    The index is 0-based, so index / total_steps topped out at (n - 1) / n --
    a model that settled on the last of 4 steps read 0.75, and no chain could
    ever reach 1.
    """
    n = len(STEPS)
    gold = [0] * (n - 1) + [1]
    last, _l, _e, _i = _derive(_raw(anchoring_point={"anchoring_step_index": n - 1,
                                                     "gold_alive_per_step": gold}))
    first, _l, _e, _i = _derive(_raw(anchoring_point={"anchoring_step_index": 0,
                                                      "gold_alive_per_step": gold}))
    assert last["reasoning_anchoring_point_normalized"] == 1.0
    assert first["reasoning_anchoring_point_normalized"] == 1 / n
    assert last["reasoning_anchoring_point"] == float(n - 1), "the raw index is unchanged"


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
    assert metrics["reasoning_observation_coverage"] == 1.0  # cannot exceed 1

    # Metrics 3, 5, 6, 8, 9, 11 -- each list's own aggregate.
    assert metrics["reasoning_branchiness_total"] == 2.0
    assert metrics["reasoning_step_directionality_mean"] == 0.75
    assert metrics["reasoning_differential_elimination"] == 1.0
    assert metrics["reasoning_differential_elimination_normalized"] == 0.25
    assert metrics["reasoning_uncertainty_steps"] == 2.0
    assert metrics["reasoning_uncertainty_rate"] == 0.5
    assert metrics["reasoning_prior_knowledge"] == 3.0
    # PROPORTION OF STEPS, NOT DENSITY PER STEP. The list is [0, 2, 1, 0]: two
    # of the four steps drew on prior knowledge at all, so 0.5. It used to be
    # sum/steps = 3/4 = 0.75, which counts the step citing two facts twice and
    # is not bounded by 1 -- [1, 4, 0, 6] would have given a "normalized" 2.75.
    # The density is not lost: `reasoning_prior_knowledge` above is the sum.
    assert metrics["reasoning_prior_knowledge_normalized"] == 0.5
    assert metrics["reasoning_unresolved_contradictions"] == 1.0
    assert metrics["reasoning_unresolved_contradiction_normalized"] == 0.25

    # Metric 10 -- an index into the step list, and where it falls in the chain.
    assert metrics["reasoning_anchoring_point"] == 1.0
    # 0-based index 1 of 4 steps: (1 + 1) / 4.
    assert metrics["reasoning_anchoring_point_normalized"] == 0.5


def test_the_step_count_is_reported_but_always_read_off_the_list():
    """A sheet of averages needs a number; a column of JSON cannot be averaged.

    So the count is reported -- but it is written from the length of the step
    list and never from anything the judge said about it, which is what stops
    the two from drifting apart.
    """
    metrics, lists, _errors, _inapplicable = _derive()
    assert metrics["reasoning_total_steps"] == 4.0
    assert metrics["reasoning_total_steps"] == float(len(lists["reasoning_steps"]))
    assert "reasoning_total_steps" in REASONING_METRIC_COLUMNS

    # A judge that also volunteered a total could not override the list.
    raw = _raw()
    raw["steps"]["total_steps"] = 99
    metrics, lists, _e, _i = _derive(raw)
    assert metrics["reasoning_total_steps"] == 4.0


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
    metrics, lists, errors, inapplicable = _derive(_raw(steps={"n_steps": 0}))
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
        "branchiness_per_step": [1, 1, 0, 0],
        "mentions_per_step": [1, 2, 0, 0],
    }
    selection_raw["option_count"] = {"option_count": 4}
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
        branchiness_generation={"branchiness_per_step": [0, 2, 1, 0],
                                "mentions_per_step": [0, 2, 1, 0], "diversity": 1},
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
        _raw(anchoring_point={"anchoring_step_index": None,
                              "gold_alive_per_step": [0, 0, 0, 0]})
    )
    assert not errors, errors
    assert "reasoning_anchoring_point" not in metrics
    assert "reasoning_anchoring_point_normalized" not in metrics
    assert any("never_considered" in entry for entry in inapplicable)

    # An index past the end of the chain is a judge error, not a finding.
    _m, _l, bad, _i = _derive(_raw(anchoring_point={"anchoring_step_index": 9,
                                     "gold_alive_per_step": [0, 0, 0, 1]}))
    assert "anchoring_point:not_an_index_into_the_step_list" in bad


def test_a_question_with_no_observations_blocks_the_ratio():
    """Nothing to cover, so there is no fraction to report."""
    metrics, _lists, errors, _inapplicable = _derive(
        _raw(observation_inventory={"observations": []}, observation_coverage={
            "observations_per_step": [0, 0, 0, 0],
        })
    )
    assert "observation_inventory:invalid_or_missing_observations" in errors
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
    import yaml

    offenders = []
    for path in sorted(JUDGE_PROMPTS.glob("reasoning_*.yaml")):
        blob = yaml.safe_load(path.read_text(encoding="utf-8"))
        # The messages alone: `description` documents the prompt for whoever
        # maintains it and is never sent, so scanning it would forbid saying in
        # a comment what the code is careful not to say to the judge.
        sent = " ".join(message["content"] for message in blob["messages"])
        for found in banned.finditer(sent):
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

    blob = yaml.safe_load((JUDGE_PROMPTS / "reasoning_steps_v4.yaml").read_text())
    declared = set(blob["output_contract"]["json_fields"])
    # The segmentation only segments. What each step proves, disproves or
    # corrects is its own call now: one prompt asked to cut the chain AND judge
    # it produced a worse cut, and `steps` was already the family losing the
    # most replies to truncation.
    # The answer is not returned: it sits behind an <answer> tag, so a regex
    # already has it, and asking a judge to re-extract what a regex can read is
    # a call that can fail for a value that cannot.
    assert declared == {"steps", "n_steps"}

    counts = yaml.safe_load((JUDGE_PROMPTS / "reasoning_proof_disproof_v1.yaml").read_text())
    assert set(counts["output_contract"]["json_fields"]) == {"proof_disproof_per_step"}

    contradiction = yaml.safe_load((JUDGE_PROMPTS / "reasoning_contradiction_v2.yaml").read_text())
    assert set(contradiction["output_contract"]["json_fields"]) == {
        "unresolved_per_step", "backtracking_per_step",
    }


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
        "branchiness_per_step": [1, 1, 0, 0]
    }
    selection_raw["option_count"] = {"option_count": 4}
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
    if '"n_steps"' in body:
        # Cut from the chain the model below wrote -- a segmentation that is
        # not a span of its chain is rejected (test_steps_are_spans_of_the_chain).
        return '{"steps": ["one", "two", "three", "four"], "n_steps": 4}'
    if '"proof_disproof_per_step"' in body:
        return '{"proof_disproof_per_step": [0, 1, 1, 2]}'
    if '"unresolved_per_step"' in body:
        return ('{"unresolved_per_step": [0, 0, 0, 0], '
                '"backtracking_per_step": [0, 0, 1, 0]}')
    if '"helpfulness_per_step"' in body:
        return '{"helpfulness_per_step": [0, 1, -1, 1]}'
    if '"gold_alive_per_step"' in body:
        return ('{"anchoring_step_index": 1, '
                '"gold_alive_per_step": [0, 1, 1, 1]}')
    if '"observations"' in body and '"observations_per_step"' not in body:
        return '{"observations": ["fact one", "fact two", "fact three"]}'
    if '"option_count"' in body:
        return '{"option_count": 3}'
    if '"observations_per_step"' in body:
        return '{"observations_per_step": [1, 1, 0, 0]}'
    if '"branchiness_per_step"' in body and '"diversity"' in body:
        return ('{"branchiness_per_step": [1, 1, 0, 0], '
                '"mentions_per_step": [1, 2, 0, 0], "diversity": 1}')
    if '"branchiness_per_step"' in body:
        return ('{"branchiness_per_step": [1, 1, 0, 0], '
                '"mentions_per_step": [1, 2, 0, 0]}')
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
    return "one two three four\n\nAnswer: something"


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
                # The fake model's own window is 4,096, smaller than the steps
                # call's reply budget alone; a real judge has 65,536 or more.
                "context_window": 65536,
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
    assert lines[0]["raw"]["steps"]["n_steps"] == 4
    assert lines[0]["raw"]["unresolved_contradiction"]["backtracking_per_step"] == [0, 0, 1, 0]
    # One of four steps was a correction -- a proportion of steps, not
    # corrections per step.
    assert lines[0]["metrics"]["reasoning_backtracking_rate"] == 0.25
    # And the new families landed.
    assert lines[0]["metrics"]["reasoning_helpfulness_mean"] == 0.25
    assert lines[0]["metrics"]["reasoning_harmful_steps"] == 1.0
    # Which channel the chain came from is on the log line and on the record;
    # the fake model writes untagged prose and no native field.
    assert lines[0]["reasoning_source"] == "untagged_prose"
    records = [
        json.loads(line)
        for line in (by_mode["cot"].output_dir / "records.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert any((r.get("details") or {}).get("reasoning_source") == "untagged_prose" for r in records)
    audit = (by_mode["cot"].output_dir / ReasoningJudgeStage.AUDIT_FILENAME).read_text().splitlines()
    assert all(json.loads(line)["reasoning_source"] == "untagged_prose" for line in audit if line.strip())


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


def _targets_sent(tmp_path, *, reference: str, answer: str, **limits):
    """The targets the reasoning judge would query, without querying anything."""
    from abductionbench.core.config import ReasoningJudgeConfig
    from abductionbench.core.types import (
        ChatMessage,
        ModelResponse,
        RenderedPrompt,
        ResponseStatus,
        SampleScore,
        SampleSpec,
        SamplingParams,
        TaskIdentity,
    )

    stage = ReasoningJudgeStage(
        config=ReasoningJudgeConfig(enabled=True, model="m", **limits),
        registry=_FakeRegistry(),
        renderer=None,
        clients={"m": object()},
        retry_policy=None,
        cache_dir=tmp_path,
    )
    sent: list = []

    async def capture(targets, identity):
        sent.extend(targets)
        for target in targets:
            target.raw = {}

    stage._evaluate = capture
    sample = SampleSpec(sample_id="s1", fields={"observation": "x"},
                        reference={"gold": reference}, task_kind="generation")
    prompt = RenderedPrompt(
        sample=sample, messages=[ChatMessage(role="user", content="what happened?")],
        template_id="t", template_version="1.0", sampling=SamplingParams(max_tokens=64),
        input_tokens_est=1,
    )
    response = ModelResponse(sample_id="s1", model_id="m", status=ResponseStatus.OK,
                             content=f"<think>step one. step two.</think><answer>{answer}</answer>")
    identity = TaskIdentity(
        run_id="r", dataset_id="d", model_id="m", template_id="t", template_version="1.0",
        prompt_mode="cot", selection_mode="n/a", data_delivery_mode="static",
        task_kind="generation",
    )
    scored = [(sample, response, SampleScore(metrics={}, prediction=answer))]
    out = asyncio.run(stage.apply(_FakeAdapter(), identity, [prompt], scored))
    return sent, out


def test_a_long_reference_and_answer_reach_the_judge_whole(tmp_path):
    """moose_chem2's reference runs to 16,000 characters; it used to be cut at 8,000.

    The judge was then grading steps against the middle-less copy of a gold
    hypothesis -- so a step matching the part that was cut away was graded
    against a standard it did not appear in.
    """
    reference, answer = "R" * 16_000, "A" * 9_000
    sent, _out = _targets_sent(tmp_path, reference=reference, answer=answer)
    assert len(sent) == 1
    assert sent[0].answer == answer
    assert reference in sent[0].reference
    assert "omitted" not in sent[0].reference


def test_a_request_too_big_for_the_window_is_skipped_not_cut(tmp_path):
    """Whole or not at all: an oversize sample is counted, never judged from a cut copy.

    The bound is the judge's window in tokens: the fields, the template's
    wording and the steps call's reply (as long as the chain) must all fit.
    """
    sent, out = _targets_sent(
        tmp_path, reference="R" * 3_000, answer="A" * 3_000, context_window=8_192,
    )
    assert sent == []
    status = out[0][2].details.get("reasoning_metrics_status", "")
    assert "oversize" in status and "token_window" in status, status

    # The same sample fits a real judge's window.
    sent, _out = _targets_sent(
        tmp_path / "wide", reference="R" * 3_000, answer="A" * 3_000, context_window=65_536,
    )
    assert len(sent) == 1


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


def test_the_segmentation_budget_follows_the_chain_it_has_to_segment():
    """It returns the chain's steps, not a handful of integers.

    Every other prompt answers with a short list because it is handed the
    segmentation; this one produces it, so its reply is about as long as the
    chain it read. A fixed budget is the wrong shape: too small truncates a long
    chain mid-JSON, which costs that sample every metric indexed against the
    segmentation, and too large makes every short chain pay for headroom it
    never uses.
    """
    from abductionbench.core.config import ReasoningJudgeConfig
    from abductionbench.core.reasoning_judge import ReasoningJudgeStage

    class _Stage:
        config = ReasoningJudgeConfig(enabled=True, model="m")
        _estimate_tokens = ReasoningJudgeStage._estimate_tokens
        _budget_for = ReasoningJudgeStage._budget_for

    stage = _Stage()

    def chunk(*chains):
        return [(str(i), {"reasoning_chain": c}, "k") for i, c in enumerate(chains)]

    floor = stage.config.max_tokens_by_family["steps"]
    assert stage._budget_for("steps", chunk("x" * 400)) == floor
    # A long chain buys more room, in proportion to itself.
    assert stage._budget_for("steps", chunk("x" * 40_000)) > floor
    assert stage._budget_for("steps", chunk("x" * 60_000)) > stage._budget_for(
        "steps", chunk("x" * 40_000)
    )
    # Bounded, so one runaway chain cannot ask for the whole window.
    assert stage._budget_for("steps", chunk("x" * 4_000_000)) == (
        stage.config.max_tokens_ceiling
    )
    # A batch shares one sampling object, so it sizes to its longest member --
    # sizing to the shortest would truncate every other reply in the group.
    assert stage._budget_for("steps", chunk("x" * 400, "x" * 40_000)) == stage._budget_for(
        "steps", chunk("x" * 40_000)
    )
    # No other family is affected.
    for family in ("uncertainty", "prior_knowledge", "directionality"):
        assert stage._budget_for(family, chunk("x" * 60_000)) == stage.config.max_tokens


def test_coverage_counts_each_observation_once_so_it_cannot_exceed_one():
    """The old reading counted an observation again in every step that used it.

    Sum-over-steps divided by the inventory is then a references-per-observation
    ratio, not a coverage fraction, and it ran above 1 routinely. Counting each
    observation at its first appearance makes the list sum to the distinct
    observations reached, which is what the ratio is supposed to be over.
    """
    metrics, _lists, errors, _inapplicable = _derive(
        _raw(
            observation_inventory={"observations": ["a", "b", "c", "d"]},
            observation_coverage={
                "observations_per_step": [2, 1, 0, 0],
            },
        )
    )
    assert not errors, errors
    assert metrics["reasoning_observations_used"] == 3.0
    assert metrics["reasoning_observation_coverage"] == 0.75
    assert metrics["reasoning_observation_coverage"] <= 1.0


def test_a_chain_cannot_reach_more_observations_than_the_question_gave():
    """The only cross-check left, and the only one that carries information.

    The per-step list's own sum is the covered count, so asking the judge for
    that sum as well would give it a second chance to disagree with itself and
    tell us nothing new. What the total does check is real: a list summing above
    it means one of the two readings is wrong and there is no way to tell which.
    """
    _m, _l, errors, _i = _derive(
        _raw(observation_inventory={"observations": ["only one fact"]},
             observation_coverage={
            "observations_per_step": [1, 1, 0, 1],
        })
    )
    assert "observation_coverage:used_exceeds_the_inventory" in errors
    assert "reasoning_observation_coverage" not in _m




def test_the_segmentation_is_the_only_thing_the_rest_waits_on():
    """One wave for the cut, one for everything indexed by it.

    Every per-step list is positional against the segmentation, so none of them
    can be asked before it returns -- that is a dependency, not a scheduling
    choice. Once it is in hand the rest have no ordering between them and go out
    together.
    """
    import inspect

    from abductionbench.core.reasoning_judge import ReasoningJudgeStage

    source = inspect.getsource(ReasoningJudgeStage._evaluate)
    first_wave = source.index('"steps", targets')
    second_wave = source.index("wave_two = [")
    assert first_wave < second_wave, "the segmentation has to be bought first"

    # And the second wave really is one gather, not a sequence of awaits.
    tail = source[second_wave:]
    assert tail.count("await ") == 1, tail
    assert "asyncio.gather(*wave_two)" in tail


def test_every_judge_call_in_this_stage_runs_at_temperature_zero():
    """A structural measurement must not move because the judge rolled again.

    These numbers are counted, not preferred: a step either proves something or
    it does not. Sampling would make the same chain score differently between
    runs and make a difference between two models unreadable.
    """
    from abductionbench.core.config import EngineConfig, ReasoningJudgeConfig

    assert ReasoningJudgeConfig().temperature == 0.0
    # And the stage sends exactly what it is configured with.
    engine = EngineConfig()
    assert engine.reasoning_judge.temperature == 0.0
    assert engine.judge.temperature == 0.0


def test_the_prompts_may_call_the_step_list_numbered_because_it_is():
    """The wording is only honest because the renderer makes it so.

    Every prompt that consumes the segmentation says "numbered list", and the
    judge is invited to align its i-th value by those numbers. That is true only
    because the step list is rendered through `_numbered` on the way in -- if it
    were ever passed through raw, every one of those prompts would be lying and
    the alignment the per-step metrics rest on would be guesswork.
    """
    import inspect

    from abductionbench.core.reasoning_judge import ReasoningJudgeStage, _numbered

    assert _numbered(["first", "second"]) == "1. first\n2. second"

    source = inspect.getsource(ReasoningJudgeStage._evaluate)
    assert '["steps"] = _numbered(' in source, source

    for path in sorted(JUDGE_PROMPTS.glob("reasoning_*.yaml")):
        text = path.read_text(encoding="utf-8")
        if "{{ steps }}" in text:
            assert "numbered list" in text, f"{path.name} consumes steps but never says numbered"


# --------------------------------------------------------------------------- #
# wave one: three readings, once each, before anything else
# --------------------------------------------------------------------------- #


def test_wave_one_is_three_distinct_calls_and_never_merged():
    """steps, the observation inventory, and the option count are separate.

    Merging any two makes one reading's mistake become the other's: a judge
    that both inventories the observations and places them can place four of
    the three it just found, and nothing downstream can tell which half was
    wrong. Split, the inventory is fixed before anything is measured against
    it, and the coverage prompt is told not to revise it.
    """
    from pathlib import Path

    from abductionbench.core.config import _reasoning_judge_templates
    from abductionbench.core.prompts import PromptRegistry

    registry = PromptRegistry([Path("configs/prompts")])
    families = _reasoning_judge_templates()
    for family in ("steps", "observation_inventory", "option_count"):
        assert family in families, family

    inventory = registry.get(families["observation_inventory"])
    coverage = registry.get(families["observation_coverage"])
    counter = registry.get(families["option_count"])
    branch = registry.get(families["branchiness_selection"])

    # Wave one reads the question (and, for steps, the chain) and nothing else.
    assert set(inventory.required_fields) == {"question"}
    assert set(counter.required_fields) == {"question"}

    # The inventory LISTS; it is not asked to count as well.
    assert inventory.output_contract["json_fields"] == {"observations": "list_of_strings"}
    assert counter.output_contract["json_fields"] == {"option_count": "nonnegative_integer"}

    # Wave two CONSUMES them and re-derives neither.
    assert "observations" in coverage.required_fields
    assert "observations_total" not in coverage.output_contract["json_fields"], (
        "coverage is taking its own inventory again"
    )
    assert "option_count" in branch.required_fields
    assert "option_count" not in branch.output_contract["json_fields"], (
        "branchiness is re-reading the option count"
    )


def test_the_option_count_is_asked_for_selection_tasks_only():
    """A generation task offers no options, so there is nothing to count.

    It is marked inapplicable rather than asked and scored 0 -- a 0 there would
    average in as "this question offered no choices", which is true but is not
    a measurement of the model.
    """
    selection_raw = _raw()
    selection_raw["branchiness_selection"] = {"branchiness_per_step": [1, 1, 0, 0]}
    selection_raw["option_count"] = {"option_count": 4}
    selection_raw.pop("branchiness_generation")
    metrics, _lists, _errors, _inapplicable = _derive(
        selection_raw, generation_like=False, selection_like=True, option_count=4
    )
    assert metrics["reasoning_option_count"] == 4.0

    generation, _l, _e, gen_inapplicable = _derive()
    assert "reasoning_option_count" not in generation
    assert any("option_count" in entry for entry in gen_inapplicable)


def test_the_inventory_is_kept_so_the_ratio_can_be_audited():
    """Coverage is a ratio; a ratio whose denominator is invisible is a number
    nobody can check."""
    metrics, lists, _errors, _inapplicable = _derive()
    assert metrics["reasoning_observations_total"] == 3.0
    assert lists["reasoning_observations"] == [
        "the lawn is wet", "the sky is clear", "it is 6am",
    ]
    assert len(lists["reasoning_observations"]) == metrics["reasoning_observations_total"]
