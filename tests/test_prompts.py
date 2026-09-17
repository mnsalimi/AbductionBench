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
    # Two grade the answer, nine measure the structure of a chain of reasoning.
    assert {"judge_binary_v1", "judge_graded_v1"} <= ids
    assert {template for template in ids if not template.startswith("reasoning_")} == {
        "judge_binary_v1",
        "judge_graded_v1",
    }
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


def test_io_and_cot_prompts_differ_by_exactly_the_mode_instruction():
    """The property that makes an io/cot comparison mean anything."""
    from abductionbench.adapters._prompting import build_messages
    from abductionbench.core.modes import TaskModes

    parts = _free_form_parts(["do not restate the observation"])
    io = build_messages(parts, TaskModes(prompt_mode="io"))[0][-1].content
    cot = build_messages(parts, TaskModes(prompt_mode="cot"))[0][-1].content
    assert io != cot
    assert _without_the_mode_instruction(io) == _without_the_mode_instruction(cot)


def test_self_consistency_renders_exactly_the_cot_prompt():
    """It is cot sampled k times; a different prompt would confound the two."""
    from abductionbench.adapters._prompting import build_messages
    from abductionbench.core.modes import TaskModes

    parts = _free_form_parts(["do not restate the observation"])
    sc = build_messages(parts, TaskModes(prompt_mode="self-consistency"))[0][-1].content
    cot = build_messages(parts, TaskModes(prompt_mode="cot"))[0][-1].content
    assert sc == cot


def test_the_answer_line_is_scoped_once_in_shared_wording():
    """"output only the <answer>" is said by the closing, not by 17 datasets."""
    from abductionbench.adapters._prompting import _ANSWER_LINE_ONLY, build_messages
    from abductionbench.core.modes import TaskModes

    for mode in ("io", "cot"):
        text = build_messages(_free_form_parts([]), TaskModes(prompt_mode=mode))[0][-1].content
        assert text.count(_ANSWER_LINE_ONLY) == 1, mode
        assert text.rstrip().endswith(_ANSWER_LINE_ONLY), mode


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


def test_a_dataset_that_quotes_its_release_offers_one_prompt_mode():
    """io and cot must not be the same bytes under two labels.

    Five interactive datasets used to report an io row and a cot row built from
    the identical prompt, because their prompts come from their releases and
    interactive_start never looked at prompt_mode. medqdx was the sharpest: its
    authors' wording says "Do NOT include any explanations, reasoning", and that
    was the row labelled cot.

    Where the prompt is the release's, there is one prompt set and so one task.
    Where this harness had to write the prompt itself, the mode instruction is
    ours to add and both modes are real.
    """
    from abductionbench.core.modes import TaskModes
    from abductionbench.core.registry import resolve_adapter

    quoted = ["cloud_opsbench:CloudOpsBenchAdapter", "med_inquire:MedInquireAdapter",
              "medqdx:MedQDxAdapter", "vivabench:VivaBenchAdapter"]
    ours = ["ddxplus:DDXPlusAdapter"]

    for impl in quoted:
        cls = resolve_adapter(f"abductionbench.adapters.{impl}")
        assert cls.authors_prompt is True, impl
        assert cls.supports_modes(TaskModes(prompt_mode="io")) is None, impl
        refusal = cls.supports_modes(TaskModes(prompt_mode="cot"))
        assert refusal and "one prompt set" in refusal, impl

    for impl in ours:
        cls = resolve_adapter(f"abductionbench.adapters.{impl}")
        assert cls.authors_prompt is False, impl
        # Both modes are offered, because the prompt is this harness's to vary.
        assert cls.supports_modes(TaskModes(prompt_mode="io")) is None, impl
        assert cls.supports_modes(TaskModes(prompt_mode="cot")) is None, impl


def test_the_harness_written_interview_prompt_actually_varies_by_mode():
    """ddxplus offers both modes, so its two prompts have to differ."""
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

    from abductionbench.core.adapter import AdapterContext
    from abductionbench.core.config import load_run_config
    from abductionbench.core.modes import TaskModes
    from abductionbench.core.registry import resolve_adapter

    import pytest

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
