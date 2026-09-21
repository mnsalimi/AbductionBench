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
    ids = set(registry.ids())
    # Only judge templates ship now; a dataset's prompts live with its adapter.
    # Three kinds: the shared answer judges, the reasoning-structure family, and
    # the project-specific proxies, which are named so they cannot be mistaken
    # for a paper's own protocol.
    assert {"judge_binary_v1", "judge_binary_plausibility_v1"} <= ids
    generic = {
        template
        for template in ids
        if not template.startswith(("reasoning_", "proxy_", "interaction_"))
    }
    assert generic == {"judge_binary_v1", "judge_binary_plausibility_v1"}
    assert {template for template in ids if template.startswith("proxy_")} == {
        "proxy_closest_explanation_v1",
        "proxy_closest_hypothesis_v1",
        "proxy_hypothesis_quality_v1",
    }
    template = registry.get("judge_binary_v1")
    assert set(template.required_fields) == {"candidate", "gold"}
    assert template.output_contract["labels"] == ["1", "0", "yes", "no"]


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
    assert contract["labels"] == ["1", "0", "yes", "no"]


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
# One layer per statement: only the mode instruction talks about reasoning
# --------------------------------------------------------------------------- #

#: Response-level suppression: wording that tells the model not to reason *at
#: all*. Written independently of the production strings on purpose -- a guard
#: that reuses the code's own vocabulary can only confirm that the code says
#: what it says. Line-level scoping ("Answer with only one of: 1 or 2") is
#: deliberately not matched: that constrains the answer line, which is exactly
#: where a format belongs. No literal spaces, so these compose into a bigger
#: pattern without needing re.VERBOSE.
_SUPPRESSES_REASONING = (
    r"do\s+not\s+(explain|justify|elaborate|reason|think)"
    r"|don't\s+(explain|justify|reason)"
    r"|without\s+(explanation|justification|reasoning|commenting)"
    r"|no\s+(commentary|explanation|justification|preamble|reasoning)"
    r"|(output|write|give|state|name)\s+only\s+the"
    r"|answer\s+immediately"
    r"|skip\s+the\s+reasoning"
)

#: Response-level elicitation: wording that asks the model to reason before it
#: answers, or to show working alongside the answer.
_ELICITS_REASONING = (
    r"step\s+by\s+step"
    r"|work\s+(through|out)"
    r"|think\s+(it\s+|this\s+)?through"
    r"|show\s+your\s+(work|reasoning)"
    r"|reason\s+(first|through|about)"
    r"|briefly\s+justify"
    r"|then\s+(commit|choose|decide|answer|pick)"
    r"|weigh\s+each"
    r"|give\s+a\s+short\s+differential"
)


def _without_the_mode_instruction(text):
    """The rendered prompt with the one line that is allowed to mention reasoning removed."""
    from abductionbench.adapters._prompting import (
        _COT_INSTRUCTION,
        _COT_INSTRUCTION_BOV,
        _IO_INSTRUCTION,
    )

    for instruction in (_IO_INSTRUCTION, _COT_INSTRUCTION, _COT_INSTRUCTION_BOV):
        text = text.replace(instruction, " ")
    return text


def _free_form_parts(requirements):
    from abductionbench.adapters._prompting import PromptParts

    return PromptParts(
        system="You explain things.",
        observation="The grass is wet.",
        instructions="Name what wet it.",
        answer_format="one sentence",
        requirements=list(requirements),
    )


def test_requirements_are_a_property_of_the_task_not_of_the_mode():
    """The whole point of the split: a requirement reads the same in every mode.

    Before, `answer_constraints` mixed three different things into one list --
    what the task demands, what the answer line looks like, and whether the
    response may reason -- so the list had to be filtered per mode, and 27
    datasets carried a clause that had to be dropped under cot. Requirements now
    hold only the first, so there is nothing to filter and nothing to contradict.
    """
    from abductionbench.adapters._prompting import build_messages
    from abductionbench.core.modes import TaskModes

    requirements = ["use only the allowed predicates", "do not restate the observation"]
    rendered = {
        mode: build_messages(_free_form_parts(requirements), TaskModes(prompt_mode=mode))[0][
            -1
        ].content
        for mode in ("io", "cot", "self-consistency")
    }
    for mode, text in rendered.items():
        assert "Requirements:\n- use only the allowed predicates" in text, mode
        assert "- do not restate the observation" in text, mode
    # ...and there is no second, mode-dependent heading any more.
    for text in rendered.values():
        assert text.count("Requirements") == 1
        assert "answer line" not in text


def test_io_and_cot_prompts_differ_only_where_the_mode_requires():
    """The property that makes an io/cot comparison mean anything.

    Two layers legitimately differ, and only two: the mode instruction, and the
    closing that says where the answer goes. The closing has to, because "on the
    last line" is true of a reply that reasons first and false of one told to
    answer directly -- io gets "Your entire response must be" instead. Every
    other layer, including the task and its requirements, is byte-identical,
    which is what makes a difference in score a difference in elicitation.
    """
    from abductionbench.adapters._prompting import build_messages
    from abductionbench.core.modes import TaskModes

    parts = _free_form_parts(["do not restate the observation"])
    io = build_messages(parts, TaskModes(prompt_mode="io"))[0][-1].content
    cot = build_messages(parts, TaskModes(prompt_mode="cot"))[0][-1].content
    assert io != cot

    # The mode instruction now ends the system prompt, so the user turn's
    # only difference is the closing each mode requires.
    assert "Answer directly" not in io and "step by step" not in cot

    # And the only difference below it is the closing each mode requires.
    assert "Your entire response must be" in io and "on the last line" not in io.lower()
    assert "On the last line" in cot and "Your entire response must be" not in cot


def test_self_consistency_renders_exactly_the_cot_prompt():
    """It is cot sampled k times; a different prompt would confound the two."""
    from abductionbench.adapters._prompting import build_messages
    from abductionbench.core.modes import TaskModes

    parts = _free_form_parts(["do not restate the observation"])
    sc = build_messages(parts, TaskModes(prompt_mode="self-consistency"))[0][-1].content
    cot = build_messages(parts, TaskModes(prompt_mode="cot"))[0][-1].content
    assert sc == cot


def test_the_answer_line_is_scoped_once_in_shared_wording():
    """"output only the <answer>" is said by the closing, not by 17 datasets.

    cot says it as a separate sentence, because its answer sits at the end of a
    longer reply and the marker is what separates the two. io no longer needs
    the sentence at all: "Your entire response must be: Answer: <x>" already
    says nothing else may appear, and repeating it would be the duplication this
    module exists to prevent.
    """
    from abductionbench.adapters._prompting import _ANSWER_LINE_ONLY, build_messages
    from abductionbench.core.modes import TaskModes

    cot = build_messages(_free_form_parts([]), TaskModes(prompt_mode="cot"))[0][-1].content
    assert cot.count(_ANSWER_LINE_ONLY) == 1
    assert cot.rstrip().endswith(_ANSWER_LINE_ONLY)

    io = build_messages(_free_form_parts([]), TaskModes(prompt_mode="io"))[0][-1].content
    assert _ANSWER_LINE_ONLY not in io, "io says it once, in the closing itself"
    assert io.count("Your entire response must be") == 1


def test_requirements_survive_every_selection_mode():
    """They used to be dropped outright: _requirements ran only on the free-form path.

    Four datasets declared requirements and also ran as a selection task, so
    their task demands silently disappeared from SCS, MCS and BOV prompts.
    """
    from abductionbench.adapters._prompting import PromptParts, build_messages
    from abductionbench.core.modes import TaskModes

    parts = PromptParts(
        system="Choose.",
        observation="Something happened.",
        options=["it rained", "the sprinkler ran"],
        option_labels=["1", "2"],
        requirements=["judge each candidate against every observation"],
    )
    for selection in ("SCS", "MCS", "BOV"):
        for mode in ("io", "cot"):
            text = build_messages(
                parts, TaskModes(prompt_mode=mode, selection_mode=selection)
            )[0][-1].content
            assert "- judge each candidate against every observation" in text, (selection, mode)


def test_an_empty_requirements_list_renders_no_heading():
    from abductionbench.adapters._prompting import build_messages
    from abductionbench.core.modes import TaskModes

    text = build_messages(_free_form_parts([]), TaskModes(prompt_mode="cot"))[0][-1].content
    assert "Requirements" not in text
    assert "Answer:" in text


def test_no_shipped_dataset_talks_about_reasoning_outside_the_mode_instruction():
    """The suite-wide guard, in both directions, over every adapter that ships.

    The old version checked one direction (cot must not be told to stop
    reasoning), over five hardcoded adapters, with a regex that mirrored the
    production filter -- so it could not fail for any wording the filter already
    knew, and never looked at io at all. io was where the sharper conflicts
    were: a task line asking for a differential, immediately under "Answer
    directly. Do not explain your reasoning."
    """
    import re

    from abductionbench.adapters._prompting import PromptParts, build_messages
    from abductionbench.core.modes import TaskModes

    suppresses = re.compile(_SUPPRESSES_REASONING, re.I)
    elicits = re.compile(_ELICITS_REASONING, re.I)

    offenders = []
    for dataset_id, cls in _shipped_adapters():
        parts = PromptParts(
            system=cls.system_prompt,
            observation="an observation",
            answer_format=getattr(cls, "answer_format", "") or "an answer",
            requirements=list(getattr(cls, "task_requirements", ()) or ()),
        )
        for mode, forbidden in (("io", elicits), ("cot", suppresses)):
            rendered = " ".join(
                m.content for m in build_messages(parts, TaskModes(prompt_mode=mode))[0]
            )
            found = forbidden.search(_without_the_mode_instruction(rendered))
            if found:
                offenders.append((dataset_id, mode, found.group(0)))
    assert not offenders, f"reasoning discussed outside the mode instruction: {offenders}"


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
        requirements=["name the root cause, not a downstream symptom"],
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


def test_the_example_answer_names_no_concrete_label():
    """The example line is a placeholder, never a real answer.

    It used to show one: SCS rendered ``Answer: A``, MCS ``Answer: A, B``, BOV
    ``Answer: YES``. A selection dataset draws every sample from the same small
    label pool, so that was not an illustrative value -- it was the *same*
    value on all 150 prompts, sitting on the answer line. A constant beside the
    answer line is a prior, and the model can pick it up.

    Checked over both label pools, because the earlier fix here was about
    letters-versus-numbers and this one is about showing a label at all.
    """
    from abductionbench.adapters._prompting import PromptParts, build_messages
    from abductionbench.core.modes import TaskModes

    for pool in (["A", "B", "C"], ["1", "2", "3"]):
        parts = PromptParts(
            system="Choose.",
            observation="Something happened.",
            options=["alpha", "beta", "gamma"],
            option_labels=list(pool),
        )
        for selection, expected in (
            ("SCS", "Answer: <the chosen label>"),
            ("MCS", "Answer: <the applicable labels, separated by commas>"),
            ("BOV", "Answer: <YES or NO>"),
        ):
            for prompt_mode in ("io", "cot"):
                text = build_messages(
                    parts, TaskModes(prompt_mode=prompt_mode, selection_mode=selection)
                )[0][-1].content
                assert expected in text, (pool, selection, prompt_mode)
                for label in pool:
                    assert f"Answer: {label}" not in text, (pool, selection, label)
                assert "Answer: YES\n" not in text and not text.endswith("Answer: YES")


def test_no_shipped_selection_dataset_shows_a_concrete_example_answer():
    """Suite-wide, over rendered prompts: the fix is in `_closing` alone.

    The point of fixing it there is that no adapter has to know about it. This
    is what proves that -- if someone re-adds a concrete example for one
    dataset, or a new selection dataset arrives with its own closing, it fails
    here rather than quietly biasing that dataset's numbers.
    """
    import re

    from abductionbench.adapters._prompting import ANSWER_PREFIX, PromptParts, build_messages
    from abductionbench.core.modes import TaskModes
    from abductionbench.core.registry import resolve_adapter

    #: Every selection dataset in the suite, with the pool it labels with.
    for impl in (
        "abductionbench.adapters.b_copa:BalancedCOPAAdapter",
        "abductionbench.adapters.xcopa:XCopaAdapter",
        "abductionbench.adapters.aer:AERAdapter",
        "abductionbench.adapters.ecare:ECareAdapter",
        "abductionbench.adapters.art:ARTAdapter",
    ):
        cls = resolve_adapter(impl)
        parts = PromptParts(
            system=cls.system_prompt,
            observation="an observation",
            options=["alpha", "beta", "gamma"],
            option_labels=["A", "B", "C"],
        )
        for selection in ("SCS", "MCS", "BOV"):
            for prompt_mode in ("io", "cot"):
                text = build_messages(
                    parts, TaskModes(prompt_mode=prompt_mode, selection_mode=selection)
                )[0][-1].content
                for line in text.splitlines():
                    if not line.startswith(ANSWER_PREFIX):
                        continue
                    value = line[len(ANSWER_PREFIX):].strip()
                    assert value.startswith("<") and value.endswith(">"), (
                        f"{impl} {selection}/{prompt_mode}: the example answer is a "
                        f"concrete value, not a placeholder: {line!r}"
                    )
                    assert not re.fullmatch(r"<[A-Z0-9](,\s*[A-Z0-9])*>", value), (
                        f"{impl} {selection}/{prompt_mode}: placeholder wraps real "
                        f"labels: {line!r}"
                    )


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
            requirements=list(getattr(cls, "task_requirements", ()) or ()),
        )  # no options, as these datasets render none in their generation mode
        for selection in (None, "SCS", "BOV"):
            text = build_messages(
                parts, TaskModes(prompt_mode="io", selection_mode=selection)
            )[0][-1].content
            assert "the label" not in text, (impl, selection)
            assert "Select exactly one hypothesis" not in text, (impl, selection)


def test_no_dataset_states_the_answer_shape_or_reasoning_in_its_task_line():
    """A task line must describe the TASK -- not how to answer, not whether to reason.

    Two failures, both over the per-sample `instructions` field, which no
    mode-aware code path ever inspected:

    * **Answer shape.** The prompt has one place for it -- `answer_format` and
      the closing "On the last line, give your final answer as:". A dataset
      that also puts it in `instructions` says it twice, and under cot it says
      it in the wrong order, since the Task line renders BEFORE the reasoning
      instruction.
    * **Reasoning.** house_md asked for "a short differential, then commit to
      the single most likely diagnosis" and medcasereasoning for "State the
      diagnosis, then briefly justify it" -- in the io prompt, three lines above
      "Answer directly. Do not explain your reasoning." gear and musr each
      opened with a directive to work the evidence out first. Whether to reason
      is the mode instruction's to say, in one line, and nothing else's.
    """
    import re

    import pytest

    from abductionbench.core.adapter import AdapterContext
    from abductionbench.core.config import load_run_config
    from abductionbench.core.modes import TaskModes
    from abductionbench.core.registry import resolve_adapter

    config_path = Path("configs/runs/full.yaml")
    if not config_path.is_file():
        pytest.skip("run configs unavailable")
    try:
        config = load_run_config(config_path)
    except Exception as exc:  # pragma: no cover - needs the run env
        pytest.skip(f"cannot load the run config: {exc}")

    directive = re.compile(
        # answer shape
        r"\b(answer|respond|reply)\b.{0,30}\b(with|in|using|only)\b"
        r"|\bgive (your|the) answer\b|\boutput only\b|\breturn only\b"
        r"|\bin (one|a single) (short )?(sentence|line|word)\b"
        r"|\bone short sentence\b|\bin a few numbered steps\b"
        r"|\b(one|two|three|four|five|1 to 3) (or (two|three) )?sentences?\b"
        r"|\bkeep it (under|to) \w+ sentences\b"
        # whether to reason: the mode instruction's job alone
        + f"|{_ELICITS_REASONING}|{_SUPPRESSES_REASONING}",
        re.IGNORECASE,
    )

    offenders, checked = [], 0
    for dataset in config.datasets:
        try:
            adapter_cls = resolve_adapter(dataset.impl)
        except Exception:  # pragma: no cover - optional dependency
            continue
        for mode in adapter_cls.hypothesis_modes or ("generation",):
            options = dict(dataset.options)
            options.update((adapter_cls.hypothesis_mode_options or {}).get(mode, {}))
            try:
                context = AdapterContext(
                    dataset_id=dataset.id,
                    data_dir=Path(config.engine.data_root) / dataset.id,
                    sample_size=3,
                    seed=config.seed,
                    options=options,
                    offline=True,
                    modes=TaskModes(prompt_mode="cot"),
                )
                adapter = adapter_cls(context)
                adapter.prepare()
                samples = adapter.build_samples()[:3]
            except Exception:  # the dataset is not materialized on this machine
                continue
            for sample in samples:
                checked += 1
                found = directive.search(str(sample.fields.get("instructions") or ""))
                if found:
                    offenders.append((dataset.id, mode, found.group(0)))
                    break

    if not checked:
        pytest.skip("no datasets are materialized on this machine")
    assert not offenders, f"answer shape or reasoning stated in a task line: {offenders}"


# --------------------------------------------------------------------------- #
# The selection-mode rule, checked against every dataset the suite ships
# --------------------------------------------------------------------------- #


def _shipped_adapters():
    """Every adapter class a run config would resolve, without touching data."""
    import os

    from abductionbench.core.config import load_run_config
    from abductionbench.core.registry import resolve_adapter

    # The run config interpolates the model endpoint's key; this test never
    # calls a model, it only resolves adapter classes.
    os.environ.setdefault("ABENCH_API_KEY", "not-a-real-key")
    config = load_run_config(Path("configs/runs/pilot.yaml"))
    for dataset in config.datasets:
        yield dataset.id, resolve_adapter(dataset.impl)


def test_no_multi_answer_dataset_in_the_suite_offers_single_choice():
    """The rule, applied to what actually ships rather than to a stand-in.

    Forcing one choice on an item with several correct answers makes the item
    unanswerable, so the score would measure the constraint instead of the
    model.  The converse is allowed and is checked below.
    """
    from abductionbench.core.modes import TaskModes

    multi = [
        (dataset_id, cls)
        for dataset_id, cls in _shipped_adapters()
        if cls.selection_cardinality == "multi"
    ]
    for dataset_id, cls in multi:
        assert "SCS" not in cls.selection_modes_offered(), dataset_id
        assert cls.supports_modes(TaskModes(selection_mode="SCS")), dataset_id


def test_every_selection_dataset_in_the_suite_offers_mcs_and_bov():
    """Both are available everywhere there is a candidate list to ask about."""
    from abductionbench.core.modes import TaskModes

    selection = [
        (dataset_id, cls)
        for dataset_id, cls in _shipped_adapters()
        if cls.selection_cardinality is not None
    ]
    assert selection, "the suite ships no selection datasets -- the guard is vacuous"
    for dataset_id, cls in selection:
        offered = cls.selection_modes_offered()
        assert "MCS" in offered and "BOV" in offered, f"{dataset_id}: {offered}"
        for mode in ("MCS", "BOV"):
            assert cls.supports_modes(TaskModes(selection_mode=mode)) is None, dataset_id
        if cls.selection_cardinality in ("single", "flexible"):
            assert "SCS" in offered, dataset_id


def test_a_bov_prompt_never_refers_to_a_list_it_does_not_show():
    """BOV shows one candidate; nothing in the prompt may imply a visible list.

    This is the aiops failure one level up: there the *closing* named a list the
    prompt never rendered, here it would be the dataset's own question ("Which
    of these candidates...?") or the shared CoT instruction ("consider what each
    candidate would have to be true for").
    """
    import re

    from abductionbench.adapters._prompting import PromptParts, build_messages
    from abductionbench.core.modes import TaskModes

    parts = PromptParts(
        system="You pick the best explanation.",
        observation="the lawn is wet",
        question="Which candidate, if added to the theory, would explain the observation?",
        options=["it rained"],
        option_labels=["1"],
    )
    plural = re.compile(r"each candidate|these candidates|the candidates below|listed below", re.I)
    for prompt_mode in ("io", "cot"):
        messages, contract = build_messages(
            parts, TaskModes(prompt_mode=prompt_mode, selection_mode="BOV")
        )
        rendered = " ".join(m.content for m in messages)
        assert contract["style"] == "binary"
        assert not plural.search(rendered), f"{prompt_mode}: {rendered}"
        # And it must not ask for a superlative over candidates it cannot see.
        assert "best explanation of the observation" not in rendered
        assert "YES or NO" in rendered


def test_the_dead_binding_fields_stay_inert():
    """They load, and they do nothing -- both halves matter.

    `prompts.bindings`, `dataset_overrides` and `template_variants` date from
    the design in which the harness owned dataset wording. Each dataset's prompt
    lives with its adapter now, so nothing reads them; they are kept only so an
    old run config still loads. The risk is that they quietly acquire a meaning
    again and a stale binding starts steering real prompts, so this pins the
    fact that no caller exists.
    """
    import ast
    from pathlib import Path

    from abductionbench.core.config import PromptConfig

    # They still parse, so an old config is not a hard error.
    config = PromptConfig(
        bindings={"generation": "gone_v1"},
        dataset_overrides={"ecare": {"generation": "gone_v1"}},
        template_variants={"v2": {"generation": "gone_v1"}},
    )
    assert config.bindings == {"generation": "gone_v1"}

    # ...and nothing in the package calls the machinery that would read them.
    callers = []
    for path in Path("src/abductionbench").rglob("*.py"):
        if path.name == "prompts.py":
            continue  # where they are defined
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Attribute) and node.attr in (
                "bindings_for_dataset",
                "template_for",
            ):
                callers.append(f"{path.name}:{node.lineno}")
    assert not callers, (
        f"prompt bindings have acquired a caller ({callers}); either wire them up "
        "properly and document them, or delete them -- they must not half-work"
    )


# --------------------------------------------------------------------------- #
# an interactive benchmark is a protocol: one prompt set means one prompt mode
# --------------------------------------------------------------------------- #


def test_every_interactive_protocol_offers_one_prompt_mode():
    """io and cot must not be the same bytes under two labels.

    Five interactive datasets used to report an io row and a cot row built from
    the identical prompt, because their prompts came from their releases and
    interactive_start never looked at prompt_mode. medqdx was the sharpest: its
    authors' wording says "Do NOT include any explanations, reasoning", and that
    was the row labelled cot.

    The prompts are this suite's now, so none of them quotes a release -- but
    all five still run io only, and for the reason `io_only` states rather than
    the provenance one: a turn-by-turn protocol needs every turn to be a single
    parseable action, which a reasoning instruction contradicts. ddxplus was the
    last to change: its interview prompt being ours does not make "work through
    the evidence step by step" compatible with one question per turn.
    """
    from abductionbench.core.modes import TaskModes
    from abductionbench.core.registry import resolve_adapter

    protocol = ["cloud_opsbench:CloudOpsBenchAdapter", "med_inquire:MedInquireAdapter",
                "medqdx:MedQDxAdapter", "vivabench:VivaBenchAdapter",
                "ddxplus:DDXPlusAdapter"]

    for impl in protocol:
        cls = resolve_adapter(f"abductionbench.adapters.{impl}")
        assert cls.authors_prompt is False, impl
        assert cls.io_only is True, impl
        assert cls.supports_modes(TaskModes(prompt_mode="io")) is None, impl
        for refused in ("cot", "self-consistency"):
            why = cls.supports_modes(TaskModes(prompt_mode=refused))
            assert why and "protocol" in why, f"{impl} would schedule a {refused} task"

    # A static dataset whose prompt this harness writes still gets both modes.
    static = resolve_adapter("abductionbench.adapters.commonwhy:CommonWhyAdapter")
    assert static.io_only is False
    assert static.supports_modes(TaskModes(prompt_mode="io")) is None
    assert static.supports_modes(TaskModes(prompt_mode="cot")) is None


def test_the_shared_mode_instruction_actually_varies_by_mode():
    """The one line that differs between an io prompt and a cot one."""
    from abductionbench.adapters._prompting import mode_instruction
    from abductionbench.core.modes import TaskModes

    io = mode_instruction(TaskModes(prompt_mode="io"))
    cot = mode_instruction(TaskModes(prompt_mode="cot"))
    assert io != cot
    assert "Do not explain" in io
    assert "step by step" in cot
    # self-consistency is cot sampled k times, so it renders the cot line.
    assert mode_instruction(TaskModes(prompt_mode="self-consistency")) == cot


# --------------------------------------------------------------------------- #
# three prompts that contradicted themselves
# --------------------------------------------------------------------------- #


def test_no_dataset_question_dictates_the_answer_shape():
    """The question renders in every mode; the answer's shape is the closing's.

    XCOPA's question carried "answer with a label only", so a cot prompt asked
    the model to reason and, in the same breath, not to. The closing already
    names the admissible labels and where to put them.
    """
    import re

    import pytest

    from abductionbench.core.adapter import AdapterContext
    from abductionbench.core.config import load_run_config
    from abductionbench.core.modes import TaskModes
    from abductionbench.core.registry import resolve_adapter

    shape = re.compile(
        r"answer with a label only|label only|\bone word\b|\ba single letter\b"
        r"|answer with only",
        re.IGNORECASE,
    )
    config_path = Path("configs/runs/full.yaml")
    if not config_path.is_file():
        pytest.skip("run configs unavailable")
    try:
        config = load_run_config(config_path)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"cannot load the run config: {exc}")

    offenders, checked = [], 0
    for dataset in config.datasets:
        try:
            adapter_cls = resolve_adapter(dataset.impl)
        except Exception:  # pragma: no cover - optional dependency
            continue
        try:
            context = AdapterContext(
                dataset_id=dataset.id,
                data_dir=Path(config.engine.data_root) / dataset.id,
                sample_size=3, seed=config.seed, options=dict(dataset.options),
                offline=True, modes=TaskModes(prompt_mode="cot"),
            )
            adapter = adapter_cls(context)
            adapter.prepare()
            samples = adapter.build_samples()[:3]
        except Exception:      # not materialized on this machine
            continue
        for sample in samples:
            checked += 1
            found = shape.search(str(sample.fields.get("question") or ""))
            if found:
                offenders.append((dataset.id, found.group(0)))
                break
    if not checked:
        pytest.skip("no datasets are materialized on this machine")
    assert not offenders, f"a dataset question dictates the answer shape: {offenders}"


def test_commonwhy_does_not_assert_a_polarity_the_data_does_not_have():
    """Most of its questions are not impossibilities.

    Of 400 sampled items, 220 carry no negation at all -- "Why is it likely that
    X celebrates Christmas?", "Why must X understand economic systems?". The
    prompt asserted the model had been told an entity *could not* do something,
    which on the majority of the dataset states the opposite of the question
    asked, and a model obeying it answers a question nobody put.
    """
    import re

    from abductionbench.core.registry import resolve_adapter

    cls = resolve_adapter("abductionbench.adapters.commonwhy:CommonWhyAdapter")
    asserts_negative = re.compile(
        r"could not do something|explain why not|the impossibility|makes it impossible",
        re.IGNORECASE,
    )
    assert not asserts_negative.search(cls.system_prompt), cls.system_prompt


def test_a_multi_line_answer_closes_with_a_block_not_a_last_line():
    """"one fact per line" and "on the last line" cannot both be obeyed."""
    from abductionbench.adapters._prompting import PromptParts, build_messages
    from abductionbench.core.metrics import extract_answer_span
    from abductionbench.core.modes import TaskModes
    from abductionbench.core.registry import resolve_adapter

    assert resolve_adapter(
        "abductionbench.adapters.proof_writer:ProofWriterAdapter"
    ).answer_is_a_block is True

    block = build_messages(
        PromptParts(system="s", observation="o", answer_format="one fact per line, or None",
                    answer_is_block=True),
        TaskModes(prompt_mode="cot"),
    )[0][-1].content
    assert "End your reply with the answer block" in block
    assert "On the last line" not in block

    # Every other dataset keeps the single-line closing.
    one_line = build_messages(
        PromptParts(system="s", observation="o", answer_format="a single diagnosis"),
        TaskModes(prompt_mode="cot"),
    )[0][-1].content
    assert "On the last line" in one_line
    assert "answer block" not in one_line

    # And the parser already handles the block -- it was never the obstacle.
    parsed = extract_answer_span(
        "Reasoning.\nAnswer:\nAnna is kind.\nBob is round.", {"answer_prefix": "Answer:"}
    )
    assert parsed == "Anna is kind.\nBob is round."


# --------------------------------------------------------------------------- #
# An io reply is the answer; a cot reply ends with it
# --------------------------------------------------------------------------- #

#: Wording that only means something if the reply has a body above the answer.
#: True of cot, where the model reasons first. False of io, which was just told
#: "Answer directly. Do not explain your reasoning." -- and then, for a long
#: time, "On the last line, give your final answer as", which describes a reply
#: with something before it. A model reading that could reasonably conclude a
#: preamble was wanted, i.e. that this was the reasoning mode after all.
_IMPLIES_PRECEDING_TEXT = r"on the last line|end your reply with|after your (?:reasoning|analysis)"

#: What each mode's closing says instead.
_IO_CLOSING = "Your entire response must be"


def _closings_for(cls, selection_mode, prompt_mode):
    """One shipped dataset's rendered prompt, in one mode."""
    from abductionbench.adapters._prompting import PromptParts, build_messages
    from abductionbench.core.modes import TaskModes

    options = ["first candidate", "second candidate"] if selection_mode else []
    parts = PromptParts(
        system=cls.system_prompt,
        observation="an observation",
        answer_format=getattr(cls, "answer_format", "") or "an answer",
        requirements=list(getattr(cls, "task_requirements", ()) or ()),
        options=options,
        answer_is_block=bool(getattr(cls, "answer_is_a_block", False)),
    )
    modes = TaskModes(prompt_mode=prompt_mode, selection_mode=selection_mode)
    messages, _contract = build_messages(parts, modes)
    return " ".join(m.content for m in messages)


def test_an_io_prompt_never_describes_a_reply_with_text_before_the_answer():
    """Every shipped dataset, every selection mode, both prompt modes.

    io must ask for the answer and nothing else; cot must keep "on the last
    line", because there the answer genuinely is the last line and the marker is
    what separates it from the reasoning above.
    """
    import re

    from abductionbench.core.modes import BOV, MCS, SCS

    implies = re.compile(_IMPLIES_PRECEDING_TEXT, re.I)
    io_offenders, cot_offenders = [], []
    for dataset_id, cls in _shipped_adapters():
        for selection in (None, SCS, MCS, BOV):
            io = _closings_for(cls, selection, "io")
            found = implies.search(io)
            if found:
                io_offenders.append((dataset_id, selection, found.group(0)))
            if _IO_CLOSING not in io:
                io_offenders.append((dataset_id, selection, "no 'entire response' closing"))

            # The other direction: cot must not have been flattened into io.
            cot = _closings_for(cls, selection, "cot")
            if not implies.search(cot):
                cot_offenders.append((dataset_id, selection, "cot lost its last-line closing"))
            if _IO_CLOSING in cot:
                cot_offenders.append((dataset_id, selection, "cot took io's closing"))

    assert not io_offenders, f"io prompts implying a multi-line reply: {io_offenders[:6]}"
    assert not cot_offenders, f"cot prompts damaged by the io fix: {cot_offenders[:6]}"


def test_neither_mode_can_be_mistaken_for_the_other():
    """The two modes must stay distinguishable at both ends of the prompt.

    Fixing the io closing is only safe if it does not make an io prompt read
    like a cot one anywhere else, or vice versa: the mode instruction and the
    closing have to agree with each other in both modes.
    """
    from abductionbench.adapters._prompting import mode_instruction
    from abductionbench.core.modes import BOV, MCS, SCS, TaskModes

    io_line = mode_instruction(TaskModes(prompt_mode="io"))
    assert io_line != mode_instruction(TaskModes(prompt_mode="cot"))

    for dataset_id, cls in _shipped_adapters():
        for selection in (None, SCS, MCS, BOV):
            # BOV shows one candidate, so its cot instruction is the
            # one-at-a-time variant -- existing behaviour, not a mode leak.
            cot_line = mode_instruction(
                TaskModes(prompt_mode="cot"), one_at_a_time=selection == BOV
            )
            io = _closings_for(cls, selection, "io")
            cot = _closings_for(cls, selection, "cot")
            # Each carries its own mode instruction and not the other's.
            assert io_line in io, dataset_id
            assert cot_line not in io, dataset_id
            assert cot_line in cot, dataset_id
            assert io_line not in cot, dataset_id
            # And the two differ by more than nothing.
            assert io != cot, dataset_id


def test_the_parsing_contract_is_identical_in_both_modes():
    """The wording changed; what the scorer reads must not have.

    The closing is the only thing that moved, and the marker inside it is what
    every scorer keys on. If io and cot stopped agreeing on the contract, the
    fix would have traded a prompt contradiction for a parsing bug.
    """
    from abductionbench.adapters._prompting import ANSWER_PREFIX, PromptParts, build_messages
    from abductionbench.core.modes import BOV, MCS, SCS, TaskModes

    for dataset_id, cls in _shipped_adapters():
        for selection in (None, SCS, MCS, BOV):
            options = ["first candidate", "second candidate"] if selection else []
            parts = PromptParts(
                system=cls.system_prompt,
                observation="an observation",
                answer_format=getattr(cls, "answer_format", "") or "an answer",
                requirements=list(getattr(cls, "task_requirements", ()) or ()),
                options=options,
                answer_is_block=bool(getattr(cls, "answer_is_a_block", False)),
            )
            contracts = {}
            for mode in ("io", "cot"):
                messages, contract = build_messages(
                    parts, TaskModes(prompt_mode=mode, selection_mode=selection)
                )
                contracts[mode] = contract
                assert ANSWER_PREFIX in messages[-1].content, (dataset_id, mode)
            assert contracts["io"] == contracts["cot"], (dataset_id, selection)


def test_no_system_prompt_uses_a_verb_the_mode_instruction_negates():
    """A dataset must not open with the verb the io mode line forbids.

    The io mode instruction is "Answer directly. Do not explain your
    reasoning." A system prompt that opens a sentence with "Explain" uses the
    same verb a few lines above it, with a different object -- explain the
    *outcome* versus explain your *reasoning* -- and a reader has to resolve
    that collision before it can tell which one is meant. uncommonsense was the
    only one, against 33 datasets that say State, Name, Identify, Propose,
    Choose, Select, Decide or Answer.

    The task is untouched: uncommonsense's answer still *is* an explanation,
    named as a noun rather than demanded as an imperative that the next layer
    takes back.
    """
    import re

    # Verbs that name what the *response* should do, where the mode instruction
    # already owns that question.
    collides = re.compile(
        r"(?:^|\.\s+)(Explain|Justify|Reason|Think|Elaborate|Describe your|Show your)\b"
    )
    offenders = []
    for dataset_id, cls in _shipped_adapters():
        text = getattr(cls, "system_prompt", "") or ""
        found = collides.search(text)
        if found:
            offenders.append((dataset_id, found.group(1)))
    assert not offenders, (
        "system prompts using a verb the mode instruction negates "
        f"(say State/Name/Identify/Propose instead): {offenders}"
    )


def test_uncommonsense_still_asks_for_an_explanation():
    """The rewording must not have turned the task into something else.

    Its answer is an explanation and its metric judges plausibility; if the
    prompt stopped asking for one, the fix would have traded a wording
    collision for a changed benchmark.
    """
    from abductionbench.core.registry import resolve_adapter

    cls = resolve_adapter("abductionbench.adapters.uncommonsense:UncommonsenseAdapter")
    prompt = cls.system_prompt
    assert "explanation" in prompt, "the answer is still an explanation"
    assert "likely, not merely possible" in prompt, "the task's own constraint stands"
    assert cls.answer_format == "1 to 3 sentences"
    assert cls.primary_metric == "proxy_closest_explanation_score"


def test_the_mode_instruction_ends_the_system_prompt_and_appears_once():
    """One place for the line that says whether to reason, in every dataset.

    It used to sit in the user turn between the requirements and the closing,
    which put it in the middle of the task description. At the end of the system
    prompt it is the last standing instruction before the question -- and, more
    importantly, there is exactly one of it, so no dataset can end up carrying a
    second copy somewhere else.
    """
    from abductionbench.adapters._prompting import mode_instruction
    from abductionbench.core.modes import BOV, MCS, SCS, TaskModes

    for dataset_id, cls in _shipped_adapters():
        for selection in (None, SCS, MCS, BOV):
            for prompt_mode in ("io", "cot"):
                modes = TaskModes(prompt_mode=prompt_mode, selection_mode=selection)
                text = _closings_for(cls, selection, prompt_mode)
                line = mode_instruction(modes, one_at_a_time=selection == BOV)
                assert text.count(line) == 1, (dataset_id, selection, prompt_mode)


def test_the_multi_select_instruction_names_the_labels_it_shows():
    """Three datasets were shown A, B, C and told to reply with numbers."""
    from abductionbench.adapters._prompting import PromptParts, build_messages
    from abductionbench.core.modes import TaskModes

    for labels, noun in ((["1", "2", "3"], "numbers"), (["A", "B", "C"], "letters")):
        parts = PromptParts(
            system="S", observation="O", options=["x", "y", "z"], option_labels=labels
        )
        text = build_messages(parts, TaskModes(prompt_mode="io", selection_mode="MCS"))[0][
            -1
        ].content
        assert f"their {noun}" in text, labels
        other = "letters" if noun == "numbers" else "numbers"
        assert f"their {other}" not in text, labels


def test_no_shipped_selection_dataset_is_told_the_wrong_label_kind():
    """The suite-wide version, over what actually ships.

    ddxplus, scir and true_detective key their options by letter and were all
    told to answer with numbers under MCS.
    """
    import re

    from abductionbench.core.modes import MCS, SCS

    offenders = []
    for dataset_id, cls in _shipped_adapters():
        if not getattr(cls, "selection_cardinality", None):
            continue
        for selection in (SCS, MCS):
            text = _closings_for(cls, selection, "io")
            lettered = re.search(r"^A\. ", text, re.M) is not None
            if lettered and "their numbers" in text:
                offenders.append((dataset_id, selection))
    assert not offenders, f"lettered options told to answer with numbers: {offenders}"


# --------------------------------------------------------------------------- #
# every answer judge is strictly binary
# --------------------------------------------------------------------------- #

#: The judges that grade an *answer*: one verdict, for the thing the model was
#: asked to produce.
#:
#: Two families are excluded, and for the same reason -- neither returns an
#: answer verdict, so "is the score 1 or 0" is not a question about them:
#:
#: * ``reasoning_*`` measures the shape of a chain of thought (step counts,
#:   coverage, branchiness). Deliberately not binary.
#: * ``interaction_*`` returns one binary verdict PER STEP of an episode, as a
#:   list. Binary per element, but its contract is ``list_of_binary`` rather
#:   than a single ``score_regex``, so the answer-judge shape does not fit it.
_NOT_ANSWER_JUDGES = ("reasoning_", "interaction_")


def _answer_judge_ids(registry) -> list[str]:
    return [t for t in registry.ids() if not t.startswith(_NOT_ANSWER_JUDGES)]


def test_every_answer_judge_asks_for_a_strictly_binary_score(prompt_dir: Path):
    """1 when the answer is acceptable by that dataset's criteria, 0 otherwise.

    A judge free to answer 3/5 turns a rate into an average, and the two are
    read differently: "the judge accepted 42% of answers" is not "the answers
    averaged 0.42 quality". These prompts were a 0-5 rubric until 2026-09-21.
    The guard is on the *contract*, because that is what the parser obeys --
    a prompt that asked for 0-5 while declaring ``[01]`` would silently score
    every 2, 3, 4 and 5 as unparseable.
    """
    registry = _registry(prompt_dir)
    for template_id in _answer_judge_ids(registry):
        contract = registry.get(template_id).output_contract or {}
        assert "score_regex" in contract, f"{template_id}: no score_regex"
        assert "[01]" in contract["score_regex"], (
            f"{template_id}: score_regex accepts values outside 1/0 -- "
            f"{contract['score_regex']!r}"
        )
        # A scale would divide the score: 1/5 is not what "1" means here.
        assert float(contract.get("score_scale", 1.0) or 1.0) == 1.0, (
            f"{template_id}: score_scale must be 1 for a binary verdict"
        )
        assert contract.get("expect_numeric_score") is True, template_id


def test_every_answer_judge_says_binary_in_the_prompt_itself(prompt_dir: Path):
    """The contract is what the parser reads; this is what the judge reads.

    A regex that only accepts 1 and 0 does not stop a model answering 4 -- it
    just makes that answer unparseable, and an unparseable verdict is a lost
    sample rather than a wrong one. The instruction has to be in the prompt.
    """
    registry = _registry(prompt_dir)
    for template_id in _answer_judge_ids(registry):
        text = "\n".join(m["content"] for m in registry.get(template_id).messages).lower()
        assert "strictly binary" in text, f"{template_id}: never says the score is binary"
        assert "score: 1" in text or "<1 or 0>" in text, (
            f"{template_id}: never shows the 1/0 output shape"
        )
        # No rubric may survive that offers anything between the two.
        for banned in ("0-5", "0 to 5", "integer 0-5", "partial credit for"):
            assert banned not in text, f"{template_id}: still offers a scale ({banned!r})"


def test_the_reasoning_judges_were_left_alone(prompt_dir: Path):
    """The binary rule is for answer judges only.

    A step count or a coverage fraction is not a yes/no, and forcing one onto
    them would destroy the metric. This fails if the sweep above ever widens.
    """
    registry = _registry(prompt_dir)
    reasoning = [t for t in registry.ids() if t.startswith("reasoning_")]
    assert len(reasoning) >= 10, "the reasoning family went missing"
    # The interaction judge is per-step and binary per element, but it is not
    # an answer judge either; it must not drift into that guard.
    assert any(t.startswith("interaction_") for t in registry.ids())
    non_binary = [
        t for t in reasoning
        if "[01]" not in str((registry.get(t).output_contract or {}).get("score_regex", ""))
    ]
    assert non_binary, "every reasoning judge became binary -- they must not be"


def test_a_disobedient_judge_is_still_read(prompt_dir: Path):
    """"yes" scores 1, not 0.

    The prompts demand 1 or 0, and a strict contract is right -- but a judge
    that answers "yes" anyway should be understood, not silently counted as a
    rejection. An unparseable verdict costs a sample; a misparsed one costs a
    wrong number, and this is the cheap way to avoid both.
    """
    from abductionbench.core.judge import JudgeStage

    registry = _registry(prompt_dir)
    for template_id in ("judge_binary_v1", "judge_binary_plausibility_v1"):
        stage = object.__new__(JudgeStage)
        stage.template = registry.get(template_id)
        assert JudgeStage._parse(stage, "Score: 1").positive is True
        assert JudgeStage._parse(stage, "Score: 0").positive is False
        assert JudgeStage._parse(stage, "YES").positive is True, template_id
        assert JudgeStage._parse(stage, "no").positive is False, template_id
        # Still nothing between the two: there is no verdict that means 0.5.
        assert JudgeStage._parse(stage, "Score: 3").score in (None, 3.0)
