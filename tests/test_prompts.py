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
