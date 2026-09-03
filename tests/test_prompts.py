"""Prompt registry, binding resolution and rendering."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from abductionbench.core.config import PromptConfig
from abductionbench.core.errors import TemplateError
from abductionbench.core.prompts import PromptRegistry, PromptRenderer
from abductionbench.core.types import ChatMessage, SampleSpec


def _registry(prompt_dir: Path) -> PromptRegistry:
    return PromptRegistry([prompt_dir])


def test_shipped_templates_load(prompt_dir: Path):
    registry = _registry(prompt_dir)
    ids = registry.ids()
    assert {"gen_freeform_v1", "gen_cot_v1", "sel_mcq_letter_v1", "judge_binary_v1"} <= set(ids)
    template = registry.get("gen_freeform_v1")
    assert template.required_fields == ["observation"]
    assert template.output_contract["answer_prefix"] == "Answer:"


def test_render_generation_and_selection(prompt_dir: Path):
    config = PromptConfig(
        template_dirs=[prompt_dir],
        bindings={"generation": "gen_freeform_v1", "selection": "sel_mcq_letter_v1"},
    )
    renderer = PromptRenderer(_registry(prompt_dir), config)

    sample = SampleSpec(
        sample_id="a",
        fields={"observation": "The lawn is wet.", "context": "It is summer."},
        task_kind="generation",
    )
    messages, contract = renderer.render(sample, _registry(prompt_dir).get("gen_freeform_v1"))
    assert [m.role for m in messages] == ["system", "user"]
    assert "The lawn is wet." in messages[1].content
    assert "It is summer." in messages[1].content
    assert contract["answer_prefix"] == "Answer:"

    mcq = SampleSpec(
        sample_id="b",
        fields={"observation": "Wet lawn.", "options": ["rain", "sprinkler"]},
        task_kind="selection",
    )
    messages, _ = renderer.render(mcq, _registry(prompt_dir).get("sel_mcq_letter_v1"))
    body = messages[-1].content
    assert "A) rain" in body and "B) sprinkler" in body


def test_missing_required_field_is_an_error(prompt_dir: Path):
    config = PromptConfig(template_dirs=[prompt_dir], bindings={"generation": "gen_freeform_v1"})
    renderer = PromptRenderer(_registry(prompt_dir), config)
    sample = SampleSpec(sample_id="x", fields={"context": "no observation"})
    with pytest.raises(TemplateError, match="missing fields"):
        renderer.render(sample, _registry(prompt_dir).get("gen_freeform_v1"))


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


def test_binding_precedence_and_variants(prompt_dir: Path):
    config = PromptConfig(
        template_dirs=[prompt_dir],
        bindings={"generation": "gen_freeform_v1"},
        dataset_overrides={"ds": {"generation": "gen_cot_v1"}},
        template_variants={"structured": {"generation": "gen_structured_v1"}},
    )
    renderer = PromptRenderer(_registry(prompt_dir), config)

    bindings = renderer.bindings_for_dataset("ds", {})
    assert bindings[0].variant == "default"
    assert bindings[0].template_for("generation") == "gen_cot_v1"  # dataset override wins
    assert bindings[1].variant == "structured"
    assert bindings[1].template_for("generation") == "gen_structured_v1"

    # A dataset-config binding beats the run-level dataset override.
    bindings = renderer.bindings_for_dataset("ds", {"generation": "gen_freeform_v1"})
    assert bindings[0].template_for("generation") == "gen_freeform_v1"

    other = renderer.bindings_for_dataset("other", {})
    assert other[0].template_for("generation") == "gen_freeform_v1"


def test_unbound_task_kind_raises(prompt_dir: Path):
    config = PromptConfig(template_dirs=[prompt_dir], bindings={"generation": "gen_freeform_v1"})
    renderer = PromptRenderer(_registry(prompt_dir), config)
    binding = renderer.bindings_for_dataset("ds", {})[0]
    with pytest.raises(TemplateError, match="no prompt template bound"):
        binding.template_for("selection")


def test_messages_override_bypasses_templating(prompt_dir: Path):
    config = PromptConfig(template_dirs=[prompt_dir], bindings={"generation": "gen_freeform_v1"})
    renderer = PromptRenderer(_registry(prompt_dir), config)
    sample = SampleSpec(
        sample_id="o",
        fields={},
        messages_override=[ChatMessage("user", "verbatim")],
    )
    messages, _ = renderer.render(sample, _registry(prompt_dir).get("gen_freeform_v1"))
    assert len(messages) == 1 and messages[0].content == "verbatim"


def test_duplicate_template_id_detected(tmp_path: Path):
    directory = tmp_path / "t"
    directory.mkdir()
    for name in ("one.yaml", "two.yaml"):
        (directory / name).write_text(
            yaml.safe_dump({"id": "dup", "messages": [{"role": "user", "content": "x"}]}),
            encoding="utf-8",
        )
    with pytest.raises(TemplateError, match="duplicate prompt template id"):
        PromptRegistry([directory])
