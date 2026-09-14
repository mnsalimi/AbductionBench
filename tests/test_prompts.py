"""Prompt registry, binding resolution and rendering.

These now cover the *judge* templates: dataset prompts moved into the adapters
(specification item 4), so what remains configurable here is how the harness
prompts a judge model. The templating machinery itself is unchanged, which is
what most of these check.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from abductionbench.core.config import PromptConfig
from abductionbench.core.errors import TemplateError
from abductionbench.core.prompts import PromptRegistry, PromptRenderer
from abductionbench.core.types import SampleSpec


def _registry(prompt_dir: Path) -> PromptRegistry:
    return PromptRegistry([prompt_dir])


def test_shipped_templates_load(prompt_dir: Path):
    registry = _registry(prompt_dir)
    ids = registry.ids()
    # Only judge templates ship now; a dataset's prompts live with its adapter.
    assert set(ids) == {"judge_binary_v1", "judge_graded_v1"}
    template = registry.get("judge_binary_v1")
    assert set(template.required_fields) == {"candidate", "gold"}
    assert template.output_contract["labels"] == ["yes", "no"]


def test_no_dataset_prompt_templates_remain(prompt_dir: Path):
    """Item 4: nothing in configuration may prompt a dataset any more."""
    registry = _registry(prompt_dir)
    for template_id in registry.ids():
        assert registry.get(template_id).task_kinds == ["judge"], template_id


def test_render_a_judge_prompt(prompt_dir: Path):
    config = PromptConfig(template_dirs=[prompt_dir], bindings={"judge": "judge_binary_v1"})
    renderer = PromptRenderer(_registry(prompt_dir), config)

    sample = SampleSpec(
        sample_id="a",
        fields={
            "gold": "the sprinkler ran",
            "candidate": "someone left the sprinkler on",
            "observation": "The lawn is wet.",
        },
        task_kind="judge",
    )
    messages, contract = renderer.render(sample, _registry(prompt_dir).get("judge_binary_v1"))
    assert [m.role for m in messages] == ["system", "user"]
    body = messages[1].content
    assert "the sprinkler ran" in body and "someone left the sprinkler on" in body
    assert "The lawn is wet." in body          # the optional field rendered
    assert contract["labels"] == ["yes", "no"]


def test_missing_required_field_is_an_error(prompt_dir: Path):
    config = PromptConfig(template_dirs=[prompt_dir], bindings={"judge": "judge_binary_v1"})
    renderer = PromptRenderer(_registry(prompt_dir), config)
    sample = SampleSpec(sample_id="x", fields={"gold": "no candidate given"})
    with pytest.raises(TemplateError, match="missing fields"):
        renderer.render(sample, _registry(prompt_dir).get("judge_binary_v1"))


def test_unknown_variable_in_template_fails_loudly(tmp_path: Path):
    directory = tmp_path / "templates"
    directory.mkdir()
    (directory / "bad.yaml").write_text(
        yaml.safe_dump(
            {
                "id": "bad_v1",
                "version": "1.0",
                "required_fields": ["observation"],
                "messages": [{"role": "user", "content": "{{ observation }} {{ nope }}"}],
            }
        ),
        encoding="utf-8",
    )
    registry = PromptRegistry([directory])
    renderer = PromptRenderer(registry, PromptConfig(template_dirs=[directory], bindings={"generation": "bad_v1"}))
    with pytest.raises(TemplateError, match="does not provide"):
        renderer.render(SampleSpec(sample_id="s", fields={"observation": "o"}), registry.get("bad_v1"))


def test_binding_precedence_and_variants(tmp_path: Path):
    directory = _local_templates(tmp_path)
    registry = PromptRegistry([directory])
    config = PromptConfig(
        template_dirs=[directory],
        bindings={"judge": "one_v1"},
        dataset_overrides={"ds": {"judge": "two_v1"}},
        template_variants={"alt": {"judge": "two_v1"}},
    )
    renderer = PromptRenderer(registry, config)

    bindings = renderer.bindings_for_dataset("ds", {})
    assert bindings[0].variant == "default"
    assert bindings[0].template_for("judge") == "two_v1"   # dataset override wins
    assert bindings[1].variant == "alt"

    # An explicit binding beats the run-level dataset override.
    bindings = renderer.bindings_for_dataset("ds", {"judge": "one_v1"})
    assert bindings[0].template_for("judge") == "one_v1"

    other = renderer.bindings_for_dataset("other", {})
    assert other[0].template_for("judge") == "one_v1"


def test_unbound_task_kind_raises(tmp_path: Path):
    directory = _local_templates(tmp_path)
    config = PromptConfig(template_dirs=[directory], bindings={"judge": "one_v1"})
    renderer = PromptRenderer(PromptRegistry([directory]), config)
    binding = renderer.bindings_for_dataset("ds", {})[0]
    with pytest.raises(TemplateError, match="no prompt template bound"):
        binding.template_for("generation")


def _local_templates(tmp_path: Path) -> Path:
    """Two throwaway templates, for the binding-resolution tests."""
    directory = tmp_path / "templates"
    directory.mkdir()
    for name in ("one", "two"):
        (directory / f"{name}.yaml").write_text(
            yaml.safe_dump(
                {
                    "id": f"{name}_v1",
                    "version": "1.0",
                    "task_kinds": ["judge"],
                    "required_fields": ["candidate"],
                    "messages": [{"role": "user", "content": "{{ candidate }}"}],
                }
            ),
            encoding="utf-8",
        )
    return directory


# --------------------------------------------------------------------------- #
# The Requirements block must not contradict the prompt mode
# --------------------------------------------------------------------------- #


def _free_form_parts(constraints):
    from abductionbench.adapters._prompting import PromptParts

    return PromptParts(
        system="You explain things.",
        observation="The grass is wet.",
        answer_format="one sentence",
        constraints=list(constraints),
    )


def test_cot_drops_the_constraints_that_forbid_reasoning():
    """The bug: "reason step by step" and "do not explain" in one prompt.

    `answer_constraints` belong to the dataset, not to the mode, so clauses
    written for io -- where "do not explain" is the whole point -- were also
    rendered into the cot prompt. 27 of 44 datasets carried one.
    """
    from abductionbench.adapters._prompting import build_messages
    from abductionbench.core.modes import TaskModes

    constraints = [
        "output exactly one fact",
        "do not explain",
        "do not use introductory phrases or commentary",
    ]

    cot, _ = build_messages(_free_form_parts(constraints), TaskModes(prompt_mode="cot"))
    text = cot[-1].content
    assert "step by step" in text
    assert "do not explain" not in text
    assert "introductory phrases" not in text
    # The shape constraint survives, scoped to what it actually governs.
    assert "Requirements for the answer line:" in text
    assert "- output exactly one fact" in text


def test_io_keeps_them_because_there_it_is_the_instruction():
    from abductionbench.adapters._prompting import build_messages
    from abductionbench.core.modes import TaskModes

    io, _ = build_messages(
        _free_form_parts(["output exactly one fact", "do not explain"]),
        TaskModes(prompt_mode="io"),
    )
    text = io[-1].content
    assert "Answer directly. Do not explain your reasoning." in text
    assert "- do not explain" in text
    assert text.count("Requirements:") == 1
    assert "answer line" not in text


def test_self_consistency_is_scoped_like_cot():
    """It renders the same text as cot by design; the scoping must follow."""
    from abductionbench.adapters._prompting import build_messages
    from abductionbench.core.modes import TaskModes

    parts = _free_form_parts(["write exactly one sentence", "do not explain your reasoning"])
    sc, _ = build_messages(parts, TaskModes(prompt_mode="self-consistency"))
    cot, _ = build_messages(parts, TaskModes(prompt_mode="cot"))
    assert sc[-1].content == cot[-1].content
    assert "do not explain" not in sc[-1].content


def test_a_requirements_block_of_only_reasoning_clauses_disappears():
    """No empty heading left behind when every clause is dropped."""
    from abductionbench.adapters._prompting import build_messages
    from abductionbench.core.modes import TaskModes

    cot, _ = build_messages(
        _free_form_parts(["do not explain", "do not use introductory phrases or commentary"]),
        TaskModes(prompt_mode="cot"),
    )
    assert "Requirements" not in cot[-1].content
    assert "Answer:" in cot[-1].content


def test_no_shipped_dataset_contradicts_cot():
    """The suite-wide guard: rendered prompts, not declared constraints.

    Catches the conflict wherever it enters the prompt -- an adapter's
    constraints, its system prompt, or a per-sample instruction.
    """
    import re

    from abductionbench.adapters._prompting import build_messages
    from abductionbench.core.modes import TaskModes
    from abductionbench.core.registry import resolve_adapter

    suppress = re.compile(
        r"do not explain|do not use introductory|without explanation|no commentary", re.I
    )
    # A representative spread rather than every dataset: these need no data on
    # disk because the parts are built from the class, not from a sample.
    from abductionbench.adapters._prompting import PromptParts

    for impl in (
        "abductionbench.adapters.abductionrules:AbductionRulesAdapter",
        "abductionbench.adapters.house_md:HouseMDAdapter",
        "abductionbench.adapters.neulr:NeuLRAdapter",
        "abductionbench.adapters.synpat:SynPATAdapter",
        "abductionbench.adapters.uniadilr_hgc:UniADILRHGcAdapter",
    ):
        cls = resolve_adapter(impl)
        parts = PromptParts(
            system=cls.system_prompt,
            observation="an observation",
            answer_format=getattr(cls, "answer_format", "") or "an answer",
            constraints=list(getattr(cls, "answer_constraints", ()) or ()),
        )
        cot, _ = build_messages(parts, TaskModes(prompt_mode="cot"))
        rendered = " ".join(m.content for m in cot)
        assert not suppress.search(rendered), f"{impl} contradicts cot: {rendered[-300:]}"


# --------------------------------------------------------------------------- #
# The answer instruction must match what the prompt actually shows
# --------------------------------------------------------------------------- #


def test_no_selection_scaffolding_without_a_candidate_list():
    """The bug: "Answer with only one of: the label ... Answer: 1", with no list.

    aiops2025 asks for a root-cause entity by name but declared
    selection_cardinality = "single", so the engine ran it as SCS and appended a
    selection closing to a prompt that never showed a candidate. _label_list([])
    rendered as the literal "the label" and the example was "Answer: 1", so the
    model was taught to answer with a number where the answer is a service name.
    """
    from abductionbench.adapters._prompting import PromptParts, build_messages
    from abductionbench.core.modes import TaskModes

    parts = PromptParts(
        system="Find the root cause.",
        observation="A service is alerting.",
        answer_format="the root cause",
        constraints=["name only the root cause"],
    )  # note: no options

    for selection in ("SCS", "MCS", "BOV"):
        messages, contract = build_messages(
            parts, TaskModes(prompt_mode="io", selection_mode=selection)
        )
        text = messages[-1].content
        assert "the label" not in text, selection
        assert "Answer: 1" not in text, selection
        assert "Select exactly one hypothesis" not in text, selection
        assert "Select every hypothesis" not in text, selection
        # it falls back to the dataset's own answer shape
        assert "Answer: <the root cause>" in text, selection
        assert contract["style"] == "free_form", selection


def test_selection_scaffolding_survives_when_there_are_options():
    """The fix must not disarm selection where selection is real."""
    from abductionbench.adapters._prompting import PromptParts, build_messages
    from abductionbench.core.modes import TaskModes

    parts = PromptParts(
        system="Choose.",
        observation="Something happened.",
        options=["it rained", "the sprinkler ran"],
        option_labels=["1", "2"],
    )
    messages, contract = build_messages(
        parts, TaskModes(prompt_mode="io", selection_mode="SCS")
    )
    text = messages[-1].content
    assert "Select exactly one hypothesis" in text
    assert "Answer with only one of: 1 or 2" in text
    assert contract["style"] == "single_label"


def test_the_example_answer_is_always_one_of_the_offered_labels():
    """A numeric example under lettered labels would teach the wrong format."""
    from abductionbench.adapters._prompting import PromptParts, build_messages
    from abductionbench.core.modes import TaskModes

    parts = PromptParts(
        system="Choose.",
        observation="Something happened.",
        options=["alpha", "beta", "gamma"],
        option_labels=["A", "B", "C"],
    )
    text = build_messages(parts, TaskModes(prompt_mode="io", selection_mode="SCS"))[0][-1].content
    assert "Answer: A" in text
    assert "Answer: 1" not in text


def test_no_shipped_dataset_is_told_to_select_from_nothing():
    """Suite-wide guard, over rendered prompts rather than declarations.

    Six datasets hit this at once -- the failure came from the mode declaration,
    not from any one adapter's wording, so the guard has to be over what the
    model actually reads.
    """
    from abductionbench.adapters._prompting import PromptParts, build_messages
    from abductionbench.core.modes import TaskModes
    from abductionbench.core.registry import resolve_adapter

    for impl in (
        "abductionbench.adapters.aiops2025:AIOps2025Adapter",
        "abductionbench.adapters.causalopsbench:CausalOpsBenchAdapter",
        "abductionbench.adapters.med_inquire:MedInquireAdapter",
        "abductionbench.adapters.researchbench:ResearchBenchAdapter",
    ):
        cls = resolve_adapter(impl)
        parts = PromptParts(
            system=cls.system_prompt,
            observation="an observation",
            answer_format=getattr(cls, "answer_format", "") or "an answer",
            constraints=list(getattr(cls, "answer_constraints", ()) or ()),
        )  # no options, as these datasets render none in their generation mode
        for selection in (None, "SCS", "BOV"):
            text = build_messages(
                parts, TaskModes(prompt_mode="io", selection_mode=selection)
            )[0][-1].content
            assert "the label" not in text, (impl, selection)
            assert "Select exactly one hypothesis" not in text, (impl, selection)
