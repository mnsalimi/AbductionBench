"""The `steps` output budget is sized from the chain, so the estimate must not
be an under-estimate.

`steps` returns the chain segmented into JSON, so its reply is about as long
as the chain it read, and a fixed budget is the wrong shape. The code already
sizes it from the input -- the defect was the conversion: four characters per
token is an English approximation, and this suite's long chains are formal
logic, LaTeX and s-expressions, which tokenize far finer.

Measured on the run that prompted this: 142 of 1,334 `steps` calls came back
with `finish_reason: "length"` and unparseable JSON, while the same run's
judge prompts measured 2.89 characters per token.

Truncating `steps` is not worth one metric. Every per-step family is judged
against ITS segmentation, so the sample loses all of them.
"""

from __future__ import annotations

from abductionbench.core.reasoning_judge import _CHARS_PER_TOKEN, ReasoningJudgeStage


def _estimate(text: str) -> int:
    return ReasoningJudgeStage._estimate_tokens(ReasoningJudgeStage, text)


def test_the_estimate_is_not_the_english_four_characters_per_token():
    """4 ch/tok is what truncated the calls; the suite measures 2.72-2.89."""
    assert _CHARS_PER_TOKEN < 3.0
    text = "x" * 40_000
    assert _estimate(text) > 40_000 // 4


def test_a_dense_chain_is_not_under_estimated():
    """The real case: a chain that tokenizes at ~2.89 characters per token.

    The old ratio told the budget this chain was 25%+ shorter than it is, so
    the reply ran out of room echoing it back.
    """
    chars = 41_776          # the largest prompt among the truncated calls
    real_tokens = int(chars / 2.89)   # what it actually cost, from usage
    assert _estimate("y" * chars) >= real_tokens, (
        "the estimate must cover the real token count, or the reply is cut off"
    )


def test_the_estimate_is_conservative_in_the_right_direction():
    """Over- and under-estimating are not symmetric costs.

    Reserved output tokens that are never generated are never billed. A budget
    too small loses the call.
    """
    text = "z" * 10_000
    assert _estimate(text) > len(text) // 4


def test_an_empty_chain_still_estimates_zero():
    """Zero is the signal `_budget_for` uses to fall back to the fixed budget."""
    assert _estimate("") == 0
