"""The three project-specific proxy scores, and what keeps them honest.

UNcommonsense, HypoBench and HypoArena all store several references per item and
all used to judge against ``references[0]`` -- so a generation matching any of
the others was marked wrong. These metrics score against the *closest* reference
instead (HypoArena against no reference at all, since its source hypotheses are
one good analysis rather than the only one).

They are not the papers' metrics and the tests below are as concerned with
making that impossible to miss as with the arithmetic.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from abductionbench.adapters._base import PROXY_PREFIX, apply_proxy_score
from abductionbench.core.judge import JudgeStage, JudgeVerdict
from abductionbench.core.prompts import PromptRegistry
from abductionbench.core.registry import resolve_adapter
from abductionbench.core.types import SampleScore

JUDGE_PROMPTS = Path("configs/prompts/judge")

#: dataset id -> (adapter, judge template, primary metric)
PROXY_ADAPTERS = {
    "uncommonsense": (
        "uncommonsense:UncommonsenseAdapter",
        "proxy_closest_explanation_v1",
        "proxy_closest_explanation_score",
    ),
    "hypobench": (
        "hypobench:HypoBenchAdapter",
        "proxy_closest_hypothesis_v1",
        "proxy_closest_hypothesis_score",
    ),
    "hypoarena": (
        "hypoarena:HypoArenaAdapter",
        "proxy_hypothesis_quality_v1",
        "proxy_hypothesis_quality_score",
    ),
}


def _adapter(dataset_id):
    return resolve_adapter(f"abductionbench.adapters.{PROXY_ADAPTERS[dataset_id][0]}")


# --------------------------------------------------------------------------- #
# they cannot be mistaken for the papers' own results
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dataset_id", sorted(PROXY_ADAPTERS))
def test_the_metric_name_says_it_is_a_proxy(dataset_id):
    """The prefix is the first thing a reader sees in a sheet."""
    cls = _adapter(dataset_id)
    _impl, template, metric = PROXY_ADAPTERS[dataset_id]
    assert metric.startswith(PROXY_PREFIX)
    assert cls.primary_metric == metric
    assert cls.judge_template == template


@pytest.mark.parametrize("dataset_id", sorted(PROXY_ADAPTERS))
def test_the_documentation_says_which_paper_metric_this_is_not(dataset_id):
    """A reader has to be able to tell what is reproduced and what is ours.

    These three report no number the papers report, and the caveat says so by
    name -- HypoBench's discovery rates, HypoArena's arena ranking, and
    UNcommonsense's human preference judgements are all absent here.
    """
    import inspect

    source = inspect.getsource(_adapter(dataset_id))
    assert "PROJECT-SPECIFIC PROXY" in source, dataset_id
    assert "IS NOT THE PAPER'S METRIC" in source, dataset_id


def test_no_proxy_prompt_is_used_by_a_dataset_that_claims_a_paper_metric():
    """The proxy templates belong to these three adapters and nowhere else."""
    import glob

    users = set()
    for path in sorted(glob.glob("src/abductionbench/adapters/*.py")):
        text = Path(path).read_text(encoding="utf-8")
        for _impl, template, _metric in PROXY_ADAPTERS.values():
            if template in text:
                users.add(Path(path).stem)
    assert users == set(PROXY_ADAPTERS), users


# --------------------------------------------------------------------------- #
# a failed judgement is blank, never zero
# --------------------------------------------------------------------------- #


def test_an_unusable_verdict_leaves_the_metric_absent():
    """0.0 means the judge looked and found it worthless. That is a finding.

    Recording it for a call that never returned would put fabricated zeros into
    the mean, and on a 0-1 quality scale nothing distinguishes them afterwards.
    """
    seeded = SampleScore(metrics={"proxy_x": 0.0, "proxy_x_domain": 0.0}, prediction="p")
    out = apply_proxy_score(seeded, JudgeVerdict(score=None, parsed=False, raw="hmm"), "proxy_x")

    assert "proxy_x" not in out.metrics
    assert "proxy_x_domain" not in out.metrics, "a stratum cannot survive its base metric"
    assert out.details["proxy_judgement"]["status"] == "unjudged"
    assert out.details["proxy_judgement"]["judge_reply"] == "hmm"


def test_a_genuine_zero_is_kept():
    """The judge read it and rated it 0; that is data, not a missing value."""
    seeded = SampleScore(metrics={"proxy_x": 0.0}, prediction="p")
    out = apply_proxy_score(seeded, JudgeVerdict(score=0.0, parsed=True, raw="Score: 0"), "proxy_x")
    assert out.metrics["proxy_x"] == 0.0
    assert out.details["proxy_judgement"]["status"] == "judged"


def test_the_score_is_graded_and_clamped():
    seeded = SampleScore(metrics={"proxy_x": 0.0, "proxy_x_domain": 0.0}, prediction="p")
    out = apply_proxy_score(seeded, JudgeVerdict(score=0.6, parsed=True, raw="Score: 3"), "proxy_x")
    assert out.metrics["proxy_x"] == 0.6
    assert out.metrics["proxy_x_domain"] == 0.6, "strata follow the base metric"
    # Out-of-range never escapes into the sheet.
    high = apply_proxy_score(seeded, JudgeVerdict(score=1.4, parsed=True, raw=""), "proxy_x")
    assert high.metrics["proxy_x"] == 1.0


def test_the_judges_own_assessment_is_recorded():
    """Auditability: the score, the reply, and what the judge said about it."""
    seeded = SampleScore(metrics={"proxy_x": 0.0}, prediction="p")
    verdict = JudgeVerdict(
        score=0.8, parsed=True, raw="Closest: 2\nScore: 4",
        details={"closest_reference": "2", "direction": "matches"},
    )
    judgement = apply_proxy_score(seeded, verdict, "proxy_x").details["proxy_judgement"]
    assert judgement["score"] == 0.8
    assert judgement["closest_reference"] == "2"
    assert judgement["direction"] == "matches"
    assert "Closest: 2" in judgement["judge_reply"]


# --------------------------------------------------------------------------- #
# the prompts
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dataset_id", sorted(PROXY_ADAPTERS))
def test_each_proxy_template_parses_its_own_reply(dataset_id):
    _impl, template_id, _metric = PROXY_ADAPTERS[dataset_id]
    registry = PromptRegistry([JUDGE_PROMPTS])

    class _Stage:
        template = registry.get(template_id)
        _parse = JudgeStage._parse

    reply = {
        "proxy_closest_explanation_v1": "Closest: 3\nScore: 4",
        "proxy_closest_hypothesis_v1": "Closest: 2\nDirection: reversed\nScore: 2",
        "proxy_hypothesis_quality_v1": "Grounded: 5\nInsight: 3\nTestable: 4\nScore: 4",
    }[template_id]
    verdict = _Stage()._parse(reply)
    assert verdict.parsed
    assert 0.0 <= verdict.score <= 1.0
    assert verdict.details, "the template must record how the score was reached"

    # And a reply it cannot read stays unparsed rather than becoming a zero.
    assert _Stage()._parse("I would rather not say.").score is None


def test_the_closest_reference_templates_are_told_the_references_are_alternatives():
    """The whole point of the change: they are not a set to cover."""
    for template_id in ("proxy_closest_explanation_v1", "proxy_closest_hypothesis_v1"):
        blob = yaml.safe_load((JUDGE_PROMPTS / f"{template_id}.yaml").read_text())
        text = " ".join(m["content"] for m in blob["messages"]).lower()
        assert "alternatives" in text
        assert "closest" in text
        assert "references" in set(blob["required_fields"])


def test_hypobench_scores_the_direction_not_only_the_feature():
    """Naming the right feature backwards is the opposite hypothesis."""
    blob = yaml.safe_load((JUDGE_PROMPTS / "proxy_closest_hypothesis_v1.yaml").read_text())
    text = " ".join(m["content"] for m in blob["messages"]).lower()
    assert "direction" in text
    assert "wrong direction" in text or "reverses the direction" in text


def test_hypoarena_is_told_not_to_treat_the_source_hypotheses_as_the_answer():
    """Its references are calibration, and a different answer can still be right."""
    blob = yaml.safe_load((JUDGE_PROMPTS / "proxy_hypothesis_quality_v1.yaml").read_text())
    text = " ".join(m["content"] for m in blob["messages"])
    assert "THERE IS NO SINGLE RIGHT ANSWER" in text
    assert "calibration" in text.lower() or "yardstick" in text.lower()
    for dimension in ("GROUNDED", "INSIGHT", "TESTABLE"):
        assert dimension in text
    # References are optional here: the score does not depend on having them.
    assert "references" not in set(blob["required_fields"])


# --------------------------------------------------------------------------- #
# the whole path, on real samples
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dataset_id", sorted(PROXY_ADAPTERS))
def test_every_reference_reaches_the_judge_not_just_the_first(dataset_id):
    """The defect these metrics exist to fix.

    All three items store several references and all three judged against
    ``references[0]``, so a generation matching any of the others was marked
    wrong. HypoArena is included because it must still *send* them -- as
    calibration -- even though it scores against none of them.
    """
    from abductionbench.core.adapter import AdapterContext
    from abductionbench.core.modes import TaskModes
    from abductionbench.core.types import ModelResponse, ResponseStatus

    cls = _adapter(dataset_id)
    adapter = cls(
        AdapterContext(
            dataset_id=dataset_id,
            data_dir=Path("data") / dataset_id,
            modes=TaskModes(prompt_mode="io"),
            sample_size=2,
            offline=True,
        )
    )
    try:
        adapter.prepare()
        sample = adapter.build_samples()[0]
    except Exception as exc:  # pragma: no cover - dataset not materialised here
        pytest.skip(f"{dataset_id} unavailable: {type(exc).__name__}")

    response = ModelResponse(
        sample_id="s", model_id="m", status=ResponseStatus.OK, content="Answer: something"
    )
    score = adapter.score(sample, response, output_contract={"answer_prefix": "Answer:"})
    request = adapter.judge_request(sample, response, score)

    stored = [r for r in (sample.reference.get("references") or []) if str(r).strip()]
    sent = [line for line in (request.get("references") or "").splitlines() if line.strip()]
    assert len(sent) == len(stored), dataset_id
    # Numbered, so the judge can name which one it scored against.
    assert sent[0].startswith("1. ")
    # And the single stored gold is no longer what the judge is pointed at.
    assert "gold" not in request, f"{dataset_id} still sends a single gold reference"


@pytest.mark.parametrize("dataset_id", sorted(PROXY_ADAPTERS))
def test_a_failed_judgement_survives_into_the_sample_row(dataset_id):
    """Aggregates must be over what was judged, and the blank must be legible."""
    from abductionbench.core.adapter import AdapterContext
    from abductionbench.core.modes import TaskModes
    from abductionbench.core.types import ModelResponse, ResponseStatus

    cls = _adapter(dataset_id)
    metric = PROXY_ADAPTERS[dataset_id][2]
    adapter = cls(
        AdapterContext(
            dataset_id=dataset_id,
            data_dir=Path("data") / dataset_id,
            modes=TaskModes(prompt_mode="io"),
            sample_size=2,
            offline=True,
        )
    )
    try:
        adapter.prepare()
        sample = adapter.build_samples()[0]
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"{dataset_id} unavailable: {type(exc).__name__}")

    response = ModelResponse(
        sample_id="s", model_id="m", status=ResponseStatus.OK, content="Answer: something"
    )
    score = adapter.score(sample, response, output_contract={"answer_prefix": "Answer:"})
    assert metric in score.metrics, "the metric is seeded before judging"

    failed = adapter.apply_judge(
        sample, response, score, JudgeVerdict(score=None, parsed=False, raw="")
    )
    assert not [key for key in failed.metrics if key.startswith(PROXY_PREFIX)], dataset_id
    assert failed.details["proxy_judgement"]["status"] == "unjudged"

    judged = adapter.apply_judge(
        sample, response, score, JudgeVerdict(score=0.6, parsed=True, raw="Score: 3")
    )
    assert judged.metrics[metric] == 0.6
