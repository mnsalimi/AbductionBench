"""Versioned prompt templates -- for the **judge**, not for datasets.

Dataset prompts moved out of here: each child adapter owns its own system prompt
and wording (specification item 4), because what makes a good instruction is a
property of the task, and a universal one imposed by the core cannot be right
for forty benchmarks at once.  ``abductionbench.adapters._prompting`` holds the
mode scaffolding those adapters share.

What is left is the LLM judge.  A judge is the harness prompting a model of its
own, not a dataset being evaluated, so its wording stays swappable configuration
-- a YAML file such as ``configs/prompts/judge/judge_binary_v1.yaml``::

    id: judge_binary_v1
    version: "1.0"
    description: Binary same-hypothesis verdict.
    task_kinds: [judge]
    required_fields: [candidate, gold]
    messages:
      - role: system
        content: "You are grading whether two statements say the same thing."
      - role: user
        content: |
          Reference: {{ gold }}
          Candidate: {{ candidate }}
          Answer YES or NO.
    output_contract:
      answer_prefix: "Answer:"

``output_contract`` is handed to whatever parses the verdict, so a template can
change the expected answer format without a code change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jinja2

from .config import PromptConfig, load_yaml
from .errors import TemplateError
from .types import ChatMessage, SampleSpec

__all__ = ["PromptTemplate", "PromptRegistry", "PromptRenderer", "TemplateBinding"]


@dataclass(slots=True)
class PromptTemplate:
    """One versioned prompt template."""

    id: str
    version: str
    messages: list[dict[str, str]]
    description: str = ""
    task_kinds: list[str] = field(default_factory=list)
    required_fields: list[str] = field(default_factory=list)
    optional_fields: list[str] = field(default_factory=list)
    output_contract: dict[str, Any] = field(default_factory=dict)
    sampling: dict[str, Any] = field(default_factory=dict)
    #: Anything else the template author wants to expose to scorers/reports.
    metadata: dict[str, Any] = field(default_factory=dict)
    source_path: str | None = None

    @property
    def ref(self) -> str:
        return f"{self.id}@{self.version}"

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, source_path: str | None = None) -> PromptTemplate:
        missing = [k for k in ("id", "messages") if k not in data]
        if missing:
            raise TemplateError(f"template {source_path or data!r} missing keys: {missing}")
        messages = data["messages"]
        if not isinstance(messages, list) or not messages:
            raise TemplateError(f"template {data['id']!r}: 'messages' must be a non-empty list")
        for message in messages:
            if not isinstance(message, dict) or "role" not in message or "content" not in message:
                raise TemplateError(
                    f"template {data['id']!r}: each message needs 'role' and 'content'"
                )
        return cls(
            id=str(data["id"]),
            version=str(data.get("version", "1.0")),
            messages=[{"role": str(m["role"]), "content": str(m["content"])} for m in messages],
            description=str(data.get("description", "")),
            task_kinds=[str(k) for k in data.get("task_kinds", [])],
            required_fields=[str(k) for k in data.get("required_fields", [])],
            optional_fields=[str(k) for k in data.get("optional_fields", [])],
            output_contract=dict(data.get("output_contract", {})),
            sampling=dict(data.get("sampling", {})),
            metadata=dict(data.get("metadata", {})),
            source_path=source_path,
        )


class PromptRegistry:
    """Loads every template found in the configured directories, by id."""

    def __init__(self, template_dirs: list[Path]):
        self._dirs = [Path(d) for d in template_dirs]
        self._templates: dict[str, PromptTemplate] = {}
        self._load()

    def _load(self) -> None:
        for directory in self._dirs:
            if not directory.exists():
                raise TemplateError(f"prompt template directory not found: {directory}")
            for path in sorted(directory.rglob("*.y*ml")):
                data = load_yaml(path)
                # A file may hold one template or a list under 'templates:'.
                blobs = data.get("templates") if isinstance(data.get("templates"), list) else [data]
                for blob in blobs:
                    template = PromptTemplate.from_dict(blob, source_path=str(path))
                    if template.id in self._templates:
                        existing = self._templates[template.id].source_path
                        raise TemplateError(
                            f"duplicate prompt template id {template.id!r} "
                            f"({existing} and {path})"
                        )
                    self._templates[template.id] = template

    def get(self, template_id: str) -> PromptTemplate:
        try:
            return self._templates[template_id]
        except KeyError:
            raise TemplateError(
                f"unknown prompt template {template_id!r}; available: "
                f"{sorted(self._templates)}"
            ) from None

    def ids(self) -> list[str]:
        return sorted(self._templates)

    def __len__(self) -> int:
        return len(self._templates)


@dataclass(frozen=True, slots=True)
class TemplateBinding:
    """A resolved ``task_kind -> template_id`` mapping for one task.

    ``variant`` names which alternative binding set this came from
    (``"default"`` for the run's base bindings), and becomes part of the task
    identity so several templates can be compared inside one run.
    """

    variant: str
    mapping: dict[str, str]

    def template_for(self, task_kind: str) -> str:
        if task_kind in self.mapping:
            return self.mapping[task_kind]
        if "default" in self.mapping:
            return self.mapping["default"]
        raise TemplateError(
            f"no prompt template bound for task_kind {task_kind!r} in variant "
            f"{self.variant!r}; bound kinds: {sorted(self.mapping)}"
        )


class PromptRenderer:
    """Turns adapter-supplied fields into a chat conversation.

    The Jinja environment uses ``StrictUndefined``, so a template that
    references a variable no adapter supplies fails loudly at render time
    instead of silently emitting an empty prompt.  Declared ``optional_fields``
    are pre-filled with ``""`` so ``{% if question %}`` style guards work.
    """

    def __init__(self, registry: PromptRegistry, prompt_config: PromptConfig):
        self.registry = registry
        self.config = prompt_config
        self._env = jinja2.Environment(
            undefined=jinja2.StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=False,
            autoescape=False,
        )
        self._env.filters["enumerate_options"] = _enumerate_options
        self._compiled: dict[tuple[str, int], jinja2.Template] = {}

    # -- binding resolution ------------------------------------------------ #

    def bindings_for_dataset(self, dataset_id: str, dataset_overrides: dict[str, str]) -> list[
        TemplateBinding
    ]:
        """All template variants that should be evaluated for one dataset.

        Precedence (later wins): run ``prompts.bindings`` → run
        ``prompts.dataset_overrides[dataset_id]`` → the dataset config's own
        ``prompt_bindings`` → the variant's mapping.
        """
        base = dict(self.config.bindings)
        base.update(self.config.dataset_overrides.get(dataset_id, {}))
        base.update(dataset_overrides or {})

        bindings = [TemplateBinding(variant="default", mapping=base)]
        for variant_name, mapping in self.config.template_variants.items():
            merged = dict(base)
            merged.update(mapping)
            bindings.append(TemplateBinding(variant=variant_name, mapping=merged))
        return bindings

    # -- rendering --------------------------------------------------------- #

    def _template(self, template_id: str, index: int, source: str) -> jinja2.Template:
        key = (template_id, index)
        if key not in self._compiled:
            try:
                self._compiled[key] = self._env.from_string(source)
            except jinja2.TemplateSyntaxError as exc:
                raise TemplateError(
                    f"prompt template {template_id!r} message #{index} has a syntax error: {exc}"
                ) from exc
        return self._compiled[key]

    def render(
        self, sample: SampleSpec, template: PromptTemplate
    ) -> tuple[list[ChatMessage], dict[str, Any]]:
        """Render one sample.  Returns ``(messages, output_contract)``.

        An adapter that set ``messages_override`` bypasses templating entirely
        (used only where a dataset's protocol is itself the prompt).
        """
        if sample.messages_override is not None:
            return list(sample.messages_override), dict(template.output_contract)

        missing = [f for f in template.required_fields if f not in sample.fields]
        if missing:
            raise TemplateError(
                f"sample {sample.sample_id!r} is missing fields required by template "
                f"{template.ref}: {missing}"
            )

        context: dict[str, Any] = {}
        context.update(self.config.shared_blocks)
        for name in template.optional_fields:
            context.setdefault(name, "")
        context.update(sample.fields)
        # Read-only extras available to every template.
        context["fields"] = dict(sample.fields)
        context["meta"] = dict(sample.metadata)
        context["sample_id"] = sample.sample_id
        context["task_kind"] = sample.task_kind

        messages: list[ChatMessage] = []
        for index, message in enumerate(template.messages):
            compiled = self._template(template.id, index, message["content"])
            try:
                content = compiled.render(**context).strip()
            except jinja2.UndefinedError as exc:
                raise TemplateError(
                    f"template {template.ref} message #{index} references a variable that "
                    f"sample {sample.sample_id!r} does not provide: {exc}"
                ) from exc
            if not content:
                # Skip empty messages so conditional system prompts can vanish.
                continue
            messages.append(ChatMessage(role=message["role"], content=content))

        if not messages:
            raise TemplateError(
                f"template {template.ref} rendered to an empty conversation for sample "
                f"{sample.sample_id!r}"
            )
        return messages, dict(template.output_contract)


def _enumerate_options(options: Any, start: str = "A") -> str:
    """Jinja filter: render a list of options as ``A) ...`` lines.

    Generic multiple-choice formatting helper -- it makes no assumption about
    which dataset the options came from.
    """
    if not isinstance(options, (list, tuple)):
        return str(options)
    first = ord(start)
    return "\n".join(f"{chr(first + i)}) {option}" for i, option in enumerate(options))
