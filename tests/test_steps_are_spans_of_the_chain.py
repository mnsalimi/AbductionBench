"""A step the judge returns has to be a span of the chain it was given.

In openrouter-trio 703 of 5,924 step lists ended in the steps prompt's own
reply instructions -- "Reply with exactly this object and nothing after it:"
and the JSON schema -- because in v3 the chain was followed directly by them in
the same message, with nothing marking where it ended, and the judge had been
told to copy every span and omit nothing. The reply was well-formed, so it was
accepted: the step count went up by one or two, and every per-step metric was
indexed against a step the model never took.

Two fixes, tested here: the prompt fences the chain (v4), and a reply whose
steps are not found in the chain is rejected as unparseable -- not cached, so a
later pass asks again.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import orjson
import yaml

from abductionbench.core.config import PromptConfig, ReasoningJudgeConfig, _reasoning_judge_templates
from abductionbench.core.prompts import PromptRegistry, PromptRenderer
from abductionbench.core.reasoning_judge import ReasoningJudgeStage, _ungrounded_spans
from abductionbench.core.types import ChatMessage, SampleSpec, TaskIdentity

PROMPTS = Path("configs/prompts")
V4 = PROMPTS / "judge" / "reasoning_steps_v4.yaml"

CHAIN = (
    "O1: Bill received an antique mirror from his grandmother.\n"
    "O2: Bill was crushed that the mirror was broken.\n\n"
    "Hypothesis 1 says the mirror fell during the move — that explains the break.\n"
    "Hypothesis 2 does not mention the mirror at all, so it cannot explain O2.\n"
    "Therefore hypothesis 1."
)
HONEST = [
    "O1: Bill received an antique mirror from his grandmother.",
    "O2: Bill was crushed that the mirror was broken.",
    # A copy with the em dash typed as a hyphen and the whitespace re-flowed:
    # still the model's words.
    "Hypothesis 1 says the mirror fell during the move - that explains   the break.",
    "**Hypothesis 2** does not mention the mirror at all, so it cannot explain O2.",
    "Therefore hypothesis 1.",
]
LEAKED = 'Reply with exactly this object and nothing after it:\n{"steps": ["<step 1>", "<step 2>", ...], "n_steps": <integer>}'


# -- the check itself ------------------------------------------------------- #


def test_honest_cuts_are_found_even_with_dashes_markdown_and_whitespace():
    assert _ungrounded_spans(HONEST, CHAIN) == []


def test_the_prompts_own_instructions_are_not_found():
    assert _ungrounded_spans([*HONEST, LEAKED], CHAIN) == [5]
    assert _ungrounded_spans([*HONEST, "Reply with exactly this object and nothing after it:"], CHAIN) == [5]


def test_the_system_prompt_is_not_found_either():
    leaked = "Reply with ONE JSON object and nothing else. No preamble, no explanation."
    assert _ungrounded_spans([HONEST[0], leaked], CHAIN) == [1]


def test_a_short_step_needs_every_word_in_the_chain():
    assert _ungrounded_spans(["Therefore hypothesis 1."], CHAIN) == []
    assert _ungrounded_spans(["Therefore hypothesis 3 wins."], CHAIN) == [0]


def test_a_paraphrase_is_not_a_span():
    paraphrase = "The second hypothesis never talks about the mirror and so fails to account for it."
    assert _ungrounded_spans([paraphrase], CHAIN) == [0]


# -- the prompt ------------------------------------------------------------- #


def test_the_live_steps_template_is_v4_and_declares_its_spans():
    assert _reasoning_judge_templates()["steps"] == "reasoning_steps_v4"
    blob = yaml.safe_load(V4.read_text())
    assert blob["output_contract"]["spans_of"] == {"steps": "reasoning_chain"}


def test_the_chain_is_fenced_and_nothing_follows_it():
    """The closing tag is the last thing the judge reads before it replies."""
    registry = PromptRegistry([PROMPTS])
    renderer = PromptRenderer(registry, PromptConfig())
    messages, _ = renderer.render(
        SampleSpec(
            sample_id="s",
            fields={"question": "Why is Bill crushed?", "options": "", "reasoning_chain": CHAIN},
            task_kind="judge",
        ),
        registry.get("reasoning_steps_v4"),
    )
    user = messages[-1].content
    body = user[user.index("<reasoning_chain>") + len("<reasoning_chain>") : user.index("</reasoning_chain>")]
    assert body.strip() == CHAIN
    assert user.rstrip().endswith("</reasoning_chain>")
    # The reply format comes BEFORE the chain, so no instruction follows it.
    assert "reply with exactly this object" in user.split("<reasoning_chain>")[0]
    assert "No step may contain any text from outside the tags" in messages[0].content


# -- the pipeline ----------------------------------------------------------- #


class _Choice:
    def __init__(self, content):
        self.content, self.reasoning, self.finish_reason = content, None, "stop"


class _Result:
    def __init__(self, choices):
        self.choices, self.usage = choices, {"completion_tokens": 7}


def _judge(tmp_path, reply):
    class _Client:
        supports_batch = False

        async def chat_single(self, messages, params):
            return _Result([_Choice(reply)])

    class _Renderer:
        def render(self, sample, template):
            return [ChatMessage(role="system", content="s"), ChatMessage(role="user", content="u")], {}

    stage = ReasoningJudgeStage(
        config=ReasoningJudgeConfig(enabled=True, model="judge-m", cache=True),
        registry=PromptRegistry([PROMPTS]),
        renderer=_Renderer(),
        clients={"judge-m": _Client()},
        retry_policy=None,
        cache_dir=tmp_path / "cache",
        run_dir=tmp_path / "run",
    )
    identity = TaskIdentity(
        run_id="r", dataset_id="d", model_id="m", template_id="cot_n-a_static",
        template_version="1.0", prompt_mode="cot", selection_mode="n/a",
        data_delivery_mode="static", task_kind="generation",
    )

    async def run():
        async def fake_retry(fn, **kwargs):
            return await fn(), None

        import abductionbench.core.reasoning_judge as rj

        original, rj.with_retry = rj.with_retry, fake_retry
        try:
            return await stage._judge_many(
                "steps",
                {"0": {"question": "q", "options": "", "reasoning_chain": CHAIN}},
                identity=identity,
                context={"0": {"sample_id": "s", "group_id": "g", "target_index": 0}},
            )
        finally:
            rj.with_retry = original

    out = asyncio.run(asyncio.wait_for(run(), timeout=5))
    audit = tmp_path / "run" / "datasets" / "d" / "m" / "cot_n-a_static@1.0" / ReasoningJudgeStage.AUDIT_FILENAME
    rows = [orjson.loads(line) for line in audit.read_bytes().splitlines() if line.strip()]
    return stage, out, rows


def test_a_reply_with_leaked_instructions_is_rejected_and_not_cached(tmp_path):
    reply = orjson.dumps({"steps": [*HONEST, LEAKED], "n_steps": 6}).decode()
    stage, out, rows = _judge(tmp_path, reply)
    assert out["0"] is None, "a leaked step list must not reach the per-step metrics"
    assert stage.stats["ungrounded"] == 1
    assert stage._cache == {}, "cached, the bad segmentation would be served back on every resume"
    assert rows[0]["response"]["outcome"] == "ungrounded"
    # Kept to be read back: what the judge returned is the evidence.
    assert rows[0]["parse"]["values"]["steps"][-1] == LEAKED


def test_an_honest_reply_is_accepted_and_cached(tmp_path):
    reply = orjson.dumps({"steps": HONEST, "n_steps": 5}).decode()
    stage, out, rows = _judge(tmp_path, reply)
    assert out["0"]["steps"] == HONEST
    assert stage.stats.get("ungrounded", 0) == 0
    assert len(stage._cache) == 1
    assert rows[0]["response"]["outcome"] == "ok"
