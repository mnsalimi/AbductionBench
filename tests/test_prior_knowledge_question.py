"""Prior knowledge is measured against the question, so the judge must see it.

The metric counts facts the model brought in "from outside the question".
That boundary is the whole measurement, and a judge shown only the steps has
to guess where it lies -- so it counts every stated fact, including the ones
the question had already supplied. A given lab value repeated is the model
reading its input; what that value usually indicates is knowledge of its own,
and only the question distinguishes them.

This is not a prompt-wording preference: it is the difference between
measuring what the model knows and measuring how much it restates.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from abductionbench.core.prompts import PromptRegistry
from abductionbench.core.reasoning_judge import ReasoningJudgeStage


def _template():
    return PromptRegistry([Path("configs/prompts")]).get("reasoning_prior_knowledge_v2")


def _flat(template) -> str:
    return " ".join("\n".join(m["content"] for m in template.messages).lower().split())


def test_the_question_is_a_required_field_not_an_optional_one():
    """Optional would render as "" and silently restore the old measurement."""
    template = _template()
    assert "question" in template.required_fields
    assert "question" not in template.optional_fields


def test_the_prompt_renders_the_question_and_says_what_it_is_for():
    template = _template()
    body = "\n".join(m["content"] for m in template.messages)
    assert "{{ question }}" in body
    text = _flat(template)
    assert "the question is the baseline" in text
    # The rule that does the work: given facts are not background knowledge.
    assert "a fact the question already supplied is not background knowledge" in text


def test_the_runner_actually_sends_the_question():
    """The template can require it and the call site can still not pass it.

    `_run_family` builds each request from the named fields only, so a family
    that forgets one is dropped as a missing dependency -- the metric goes
    silently unmeasured rather than measured wrongly, which is safer and still
    wrong.
    """
    source = inspect.getsource(ReasoningJudgeStage._evaluate)
    assert 'run("prior_knowledge", ready, ("question", "steps"))' in source


def test_the_old_contract_is_gone():
    """`steps` alone was the v1 contract; it must not still be what is sent."""
    source = inspect.getsource(ReasoningJudgeStage._evaluate)
    assert 'run("prior_knowledge", ready, ("steps",))' not in source


def test_the_version_was_bumped_so_cached_verdicts_are_not_reused():
    """The cache key is the template ref, and this answers a different question.

    A verdict bought without the question is not a cheaper version of this
    measurement, it is the other one -- so it must not be served from cache.
    """
    assert _template().version == "2.0"
