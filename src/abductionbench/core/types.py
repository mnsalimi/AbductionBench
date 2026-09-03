"""The data contract between the core engine and child dataset adapters.

Everything here is deliberately *structural*: the engine never inspects the
semantic content of :attr:`SampleSpec.fields`, :attr:`SampleSpec.reference` or
:attr:`SampleSpec.metadata`.  Those are opaque payloads owned by the child
adapter.  That is what keeps the engine dataset-agnostic while still being able
to render prompts from swappable templates, batch them, and hand responses back
for dataset-specific scoring.

Design notes
------------
* An adapter yields *fields*, not prompt text.  The wording of the prompt lives
  in versioned template configuration (:mod:`abductionbench.core.prompts`), so
  the same adapter can be re-run against a different prompt template purely by
  changing config.  An adapter that genuinely must control the message list can
  still do so via :attr:`SampleSpec.messages_override`, but that is the
  escape hatch, not the normal path.
* ``task_kind`` is a free-form string (e.g. ``"generation"``, ``"selection"``).
  The engine only uses it as a key to look up which configured template to
  render with -- it carries no built-in behaviour.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "ChatMessage",
    "SamplingParams",
    "SampleSpec",
    "RenderedPrompt",
    "ResponseStatus",
    "ModelResponse",
    "SampleScore",
    "EvalRecord",
    "AdapterDocumentation",
    "TaskIdentity",
    "stable_hash",
]


def stable_hash(payload: Any, length: int = 16) -> str:
    """Deterministic short hash of any JSON-serializable payload.

    Used for prompt fingerprints (resume keys) and deterministic sampling.  The
    payload is serialized with sorted keys so the hash is stable across runs and
    Python versions.
    """
    blob = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:length]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One message of an OpenAI-style chat conversation."""

    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True, slots=True)
class SamplingParams:
    """Decoding parameters for one request.

    vLLM's batch endpoint (``POST /v1/chat/completions/batch``) applies a single
    sampling-parameter set to *every* conversation in the call, so the engine can
    only pack samples that share an identical :meth:`signature` into the same
    batch.  Keep this class small and hashable for exactly that reason.
    """

    max_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int | None = None
    stop: tuple[str, ...] = ()
    extra: tuple[tuple[str, Any], ...] = ()  # frozen dict of vendor-specific knobs

    def signature(self) -> str:
        """Key identifying which samples may share one batch call."""
        return stable_hash(
            {
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "seed": self.seed,
                "stop": list(self.stop),
                "extra": sorted(self.extra),
            },
            length=12,
        )

    def to_payload(self) -> dict[str, Any]:
        """Render as request JSON fields (only non-default values are sent)."""
        payload: dict[str, Any] = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.stop:
            payload["stop"] = list(self.stop)
        payload.update(dict(self.extra))
        return payload

    def merged(self, **overrides: Any) -> SamplingParams:
        """Return a copy with ``overrides`` applied (unknown keys go to ``extra``)."""
        known = {"max_tokens", "temperature", "top_p", "seed", "stop"}
        base = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "stop": self.stop,
            "extra": dict(self.extra),
        }
        for key, value in overrides.items():
            if value is None:
                continue
            if key in known:
                base[key] = tuple(value) if key == "stop" else value
            else:
                base["extra"][key] = value
        base["extra"] = tuple(sorted(base["extra"].items()))
        return SamplingParams(**base)  # type: ignore[arg-type]


@dataclass(slots=True)
class SampleSpec:
    """One evaluation item, as produced by a child adapter.

    Parameters
    ----------
    sample_id:
        Stable identifier, unique within the dataset.  Used as the resume key,
        so it must be derived from the data (not from iteration order).
    fields:
        Template variables for prompt rendering.  Opaque to the engine.
    reference:
        Gold payload used by the adapter's own scorer.  Opaque to the engine.
    task_kind:
        Which configured prompt template family to render with, e.g.
        ``"generation"`` or ``"selection"``.
    max_tokens:
        Adapter's own estimate of the output budget this item needs.  ``None``
        falls back to the model/run default.  The engine may quantize it
        upwards (see ``engine.batching.max_tokens_quantum``) so that similar
        items can share a batch call.
    sampling_overrides:
        Per-sample decoding overrides (rarely needed; every distinct signature
        costs batch packing efficiency).
    metadata:
        Free-form provenance for logs/reports (split, original index, subtask,
        difficulty, ...).
    messages_override:
        Escape hatch: a fully pre-rendered conversation.  When set, the engine
        skips template rendering for this sample and records
        ``template_id="__adapter_override__"``.  Prefer ``fields``.
    group_id:
        Optional grouping key for metrics that aggregate over several samples
        (e.g. several probes of one clinical case).  Carried through to records.
    """

    sample_id: str
    fields: dict[str, Any]
    reference: Any = None
    task_kind: str = "generation"
    max_tokens: int | None = None
    sampling_overrides: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    messages_override: list[ChatMessage] | None = None
    group_id: str | None = None


@dataclass(slots=True)
class RenderedPrompt:
    """A :class:`SampleSpec` after template rendering and budget checking."""

    sample: SampleSpec
    messages: list[ChatMessage]
    template_id: str
    template_version: str
    sampling: SamplingParams
    input_tokens_est: int
    output_contract: dict[str, Any] = field(default_factory=dict)

    @property
    def sample_id(self) -> str:
        return self.sample.sample_id

    def fingerprint(self, model_id: str) -> str:
        """Identity of *this exact request*, used for strict resume.

        Changing the template, the sampling params or the model invalidates
        previously stored records for the sample.
        """
        return stable_hash(
            {
                "model": model_id,
                "template": f"{self.template_id}@{self.template_version}",
                "messages": [m.to_dict() for m in self.messages],
                "sampling": self.sampling.to_payload(),
            }
        )


class ResponseStatus(str, Enum):
    """Outcome of a single sample's inference."""

    OK = "ok"
    #: Server produced no usable ``content`` (e.g. reasoning consumed the whole
    #: ``max_tokens`` budget, so ``content`` came back ``null``).
    EMPTY = "empty"
    #: Output was cut off by the token budget but some content exists.
    TRUNCATED = "truncated"
    #: Request failed permanently after the retry policy was exhausted.
    ERROR = "error"
    #: Skipped before inference (e.g. prompt over the input-token budget and no
    #: replacement was available).
    SKIPPED = "skipped"


@dataclass(slots=True)
class ModelResponse:
    """Normalized model output for one sample."""

    sample_id: str
    model_id: str
    status: ResponseStatus
    content: str | None = None
    reasoning: str | None = None
    finish_reason: str | None = None
    error: str | None = None
    error_class: str | None = None
    attempts: int = 1
    latency_s: float = 0.0
    batch_id: str | None = None
    batch_size: int | None = None
    batch_index: int | None = None
    #: Usage as reported by the server.  For batch calls vLLM reports usage for
    #: the *whole* call, so the engine stores the batch aggregate plus an
    #: even-split estimate; see ``usage_is_batch_aggregate``.
    usage: dict[str, Any] = field(default_factory=dict)
    usage_is_batch_aggregate: bool = False
    completion_tokens_est: int | None = None
    timestamp: float = field(default_factory=time.time)

    @property
    def text(self) -> str:
        """Content, or empty string.  Convenience for scorers."""
        return self.content or ""


@dataclass(slots=True)
class SampleScore:
    """Adapter's judgement of one response.

    Parameters
    ----------
    metrics:
        Numeric per-sample metrics.  Keys must be stable across samples of the
        same dataset; they become columns in the detailed grid and are what
        :meth:`~abductionbench.core.adapter.DatasetAdapter.aggregate` reduces.
    prediction:
        The parsed prediction (string/loggable form), for auditability.
    parse_ok:
        ``False`` when the response could not be parsed into a prediction at
        all.  The engine tracks this separately as ``parse_failure_rate`` so a
        low score caused by formatting can be told apart from a wrong answer.
    details:
        Free-form extras for the run log (matched span, chosen option, judge
        rationale, ...).
    """

    metrics: dict[str, float] = field(default_factory=dict)
    prediction: Any = None
    parse_ok: bool = True
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TaskIdentity:
    """Identifies one (dataset, model, prompt-template) evaluation unit."""

    run_id: str
    dataset_id: str
    model_id: str
    template_id: str
    template_version: str

    @property
    def slug(self) -> str:
        return f"{self.dataset_id}__{self.model_id}__{self.template_id}"

    def as_dict(self) -> dict[str, str]:
        return {
            "run_id": self.run_id,
            "dataset_id": self.dataset_id,
            "model_id": self.model_id,
            "template_id": self.template_id,
            "template_version": self.template_version,
        }


@dataclass(slots=True)
class EvalRecord:
    """One row of the unified result grid: a sample, its response and its score.

    Serialized as one JSON object per line (``records.jsonl``) with atomic
    appends, so a crashed or disconnected run loses at most the in-flight batch.
    """

    task: TaskIdentity
    sample_id: str
    status: ResponseStatus
    prompt_fingerprint: str
    task_kind: str
    input_tokens_est: int
    sampling: dict[str, Any]
    response: dict[str, Any]
    metrics: dict[str, float]
    prediction: Any = None
    parse_ok: bool = True
    reference: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    group_id: str | None = None
    created_at: float = field(default_factory=time.time)

    def to_json_dict(self) -> dict[str, Any]:
        payload = {
            **self.task.as_dict(),
            "sample_id": self.sample_id,
            "group_id": self.group_id,
            "status": self.status.value,
            "prompt_fingerprint": self.prompt_fingerprint,
            "task_kind": self.task_kind,
            "input_tokens_est": self.input_tokens_est,
            "sampling": self.sampling,
            "response": self.response,
            "metrics": self.metrics,
            "prediction": self.prediction,
            "parse_ok": self.parse_ok,
            "reference": self.reference,
            "metadata": self.metadata,
            "details": self.details,
            "created_at": self.created_at,
        }
        return payload


@dataclass(slots=True)
class AdapterDocumentation:
    """Self-description a child adapter must provide for the run documentation.

    This is how the "document what you decided" requirement is enforced
    mechanically: the engine writes these fields into every run's
    ``run_documentation.md``, so an adapter cannot ship undocumented choices.
    """

    dataset_id: str
    name: str
    domain: str
    source_url: str
    processing_mode: str
    #: Which split was used and why (e.g. "test; official test split available").
    split_used: str = ""
    #: How the abductive subset was identified, if the dataset is not
    #: abduction-only.
    abductive_subset: str = ""
    #: Sampling procedure, including the seed.
    sampling_procedure: str = ""
    #: Metric definitions and how each is computed.
    metrics_description: dict[str, str] = field(default_factory=dict)
    #: The metric used for the headline table.
    primary_metric: str = ""
    #: Decisions the adapter author made that were not specified upstream.
    decisions: list[str] = field(default_factory=list)
    #: Known caveats/limitations.
    caveats: list[str] = field(default_factory=list)
    #: Populated at prepare() time: how many items the split had, how many were
    #: sampled, how many were replaced for exceeding the input-token budget.
    statistics: dict[str, Any] = field(default_factory=dict)
