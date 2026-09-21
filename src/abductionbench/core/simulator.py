"""The model that plays the environment in an interactive benchmark.

Three datasets here are interviews -- MedQDx, Med-Inquire and VivaBench -- and
what the evaluated model interviewed used to be a lexical matcher over the case
file.  That was deterministic and reproducible, and it was also not what the
papers do: their patient answers in natural language, and a question phrased in
a way the matcher missed came back "I'm not sure" while the case plainly held
the answer.

Putting a real model there follows the papers.  It also changes what a score
means, and this module is written around that fact rather than around hiding
it:

* **The evaluated model never sees the case.**  The hidden record goes into the
  *simulator's* system prompt and nowhere else; the evaluated model's
  conversation holds only what the simulator chose to disclose in answer to
  something it asked.  :meth:`EnvironmentSimulator.ask` takes the hidden
  material and the visible conversation as separate arguments precisely so that
  the boundary is a parameter rather than a convention, and so a test can
  assert it.

* **Every call is on the record.**  Request, reply, the vendor's reasoning
  channel when it returns one, error, retry, latency, and which model answered
  -- written per task beside the records, because a transcript produced by a
  second model is evidence and not an implementation detail.

* **A simulator failure is not a wrong answer.**  If the patient cannot be
  reached, the evaluated model was never given the chance to answer, and
  scoring that episode zero would put the environment's outage into the model's
  number.  The episode is abandoned and reported as an error instead.

* **Sampling is pinned as hard as the vendor allows** -- temperature 0 and a
  seed where one is supported -- and both are *recorded* rather than presented
  as a guarantee, because a hosted model is not bit-stable and two runs of the
  same system can differ because the simulator differed.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import orjson

from .client import ModelClient
from .config import (
    EndpointConfig,
    ModelConfig,
    ModelLimitsConfig,
    ModelSamplingConfig,
    SimulatorConfig,
    TimeoutConfig,
)
from .types import ChatMessage, SamplingParams, TaskIdentity

logger = logging.getLogger(__name__)

__all__ = [
    "EnvironmentSimulator",
    "SimulatorReply",
    "SimulatorPool",
    "SIMULATOR_LOG_FILENAME",
    "SIMULATOR_SCHEMA",
]

#: Written beside each task's records, as the reasoning judge's audit log is.
SIMULATOR_LOG_FILENAME = "simulator_calls.jsonl"

#: Version tag on every audit record, so a reader can tell the shape.
SIMULATOR_SCHEMA = "simulator_call/v1"


@dataclass(slots=True)
class SimulatorReply:
    """One turn of the environment -- and whether it happened at all.

    ``ok=False`` is the case the whole design turns on: it means the
    environment did not answer, not that it answered "I don't know".  Only the
    caller can tell those apart, so the distinction is carried in the type.
    """

    ok: bool
    text: str = ""
    #: The vendor's separate reasoning channel, when it returns one. ``None``
    #: means none was returned, which is a different fact from an empty string.
    reasoning: str | None = None
    error: str | None = None
    attempts: int = 0
    #: The model string the *server* reported, which can differ from the one
    #: requested when a provider routes or versions behind the name.
    model: str | None = None
    latency_s: float = 0.0


@dataclass
class EnvironmentSimulator:
    """Plays a patient, an examiner, or any other environment, over an API.

    One instance per dataset, because the model is chosen per dataset: a
    patient answering from their own experience and an examiner reading a
    structured case file are not the same job.
    """

    config: SimulatorConfig
    client: ModelClient
    dataset_id: str = ""
    run_dir: Path | None = None
    #: Shared across datasets, so ``max_parallel_calls`` bounds the whole run
    #: and not each dataset separately.
    calls: asyncio.Semaphore | None = None
    stats: dict[str, int] = field(
        default_factory=lambda: {"calls": 0, "failures": 0, "retries": 0}
    )

    def __post_init__(self) -> None:
        if self.calls is None:
            self.calls = asyncio.Semaphore(max(1, self.config.max_parallel_calls))

    # ------------------------------------------------------------------ #
    # the one call
    # ------------------------------------------------------------------ #

    async def ask(
        self,
        *,
        hidden_brief: str,
        conversation: list[ChatMessage],
        identity: TaskIdentity | None = None,
        sample_id: str | None = None,
        turn: int = 0,
        role: str = "environment",
    ) -> SimulatorReply:
        """One environment turn.

        ``hidden_brief`` is the material the evaluated model must not see -- the
        case file, the recorded findings -- and becomes the simulator's system
        prompt.  ``conversation`` is the visible half: what the evaluated model
        asked, and what the environment has already disclosed.  Keeping the two
        apart in the signature is what makes the information boundary
        checkable.

        Never raises.  A simulator that cannot be reached comes back with
        ``ok=False`` and the caller abandons the episode.
        """
        messages = [ChatMessage(role="system", content=hidden_brief), *conversation]
        sampling = SamplingParams(
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            seed=self.config.seed,
            # Vendor fields, passed through as written -- `reasoning` among
            # them. Sorted so the sampling signature is stable.
            extra=tuple(sorted(self.config.extra.items())),
        )

        last_error: str | None = None
        attempts = 0
        assert self.calls is not None
        async with self.calls:
            for attempt in range(1, max(1, self.config.max_retries) + 1):
                attempts = attempt
                self.stats["calls"] += 1
                try:
                    result = await self.client.chat_single(messages, sampling)
                except Exception as exc:  # noqa: BLE001 - reported, never raised
                    last_error = f"{type(exc).__name__}: {exc}"
                else:
                    choice = result.choices[0] if result.choices else None
                    text = ((choice.content if choice else None) or "").strip()
                    if text:
                        reply = SimulatorReply(
                            ok=True,
                            text=text,
                            reasoning=choice.reasoning if choice else None,
                            attempts=attempt,
                            model=result.model or self.config.model,
                            latency_s=result.latency_s,
                        )
                        self._log(identity, sample_id, turn, role, messages, conversation, reply)
                        return reply
                    # An empty turn is not something the environment said; it
                    # is a call that did not work, and retrying is right.
                    last_error = "the simulator returned an empty reply"
                self.stats["retries"] += 1
                logger.warning(
                    "simulator %s (%s): attempt %d/%d failed: %s",
                    self.config.model, self.dataset_id or "?",
                    attempt, self.config.max_retries, last_error,
                )

        self.stats["failures"] += 1
        reply = SimulatorReply(
            ok=False, error=last_error, attempts=attempts, model=self.config.model
        )
        self._log(identity, sample_id, turn, role, messages, conversation, reply)
        return reply

    # ------------------------------------------------------------------ #
    # the audit trail
    # ------------------------------------------------------------------ #

    def _log_path(self, identity: TaskIdentity | None) -> Path | None:
        if self.run_dir is None or identity is None:
            return None
        return (
            self.run_dir
            / "datasets"
            / identity.dataset_id
            / identity.model_id
            / f"{identity.template_id}@{identity.template_version}"
            / SIMULATOR_LOG_FILENAME
        )

    def _log(
        self,
        identity: TaskIdentity | None,
        sample_id: str | None,
        turn: int,
        role: str,
        messages: list[ChatMessage],
        visible: list[ChatMessage],
        reply: SimulatorReply,
    ) -> None:
        """Write one call out whole.  Never raises: logging must not end a run."""
        path = self._log_path(identity)
        if path is None:
            return
        record = {
            "schema": SIMULATOR_SCHEMA,
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": getattr(identity, "run_id", None),
            "dataset_id": getattr(identity, "dataset_id", None),
            # The model under evaluation, not the simulator, so a reader can
            # tell whose episode this turn belongs to.
            "evaluated_model_id": getattr(identity, "model_id", None),
            "template_id": getattr(identity, "template_id", None),
            "template_mode": getattr(identity, "template_mode", None),
            "sample_id": sample_id,
            "turn": turn,
            "role": role,
            # Which model produced this turn of the environment and on what
            # settings, recorded per call rather than once per run: a resumed
            # pass can enter the same directory under a different config.
            "simulator": {**self.config.as_record(), "answered_by": reply.model},
            # Exactly what was submitted, untruncated -- the hidden brief
            # included, because this file is the record of what the environment
            # was told.
            "request": {"messages": [m.to_dict() for m in messages]},
            # The evaluated model's half of the conversation, so the boundary
            # is auditable: the brief sits in `request` and must never turn up
            # here.
            "visible_conversation": [m.to_dict() for m in visible],
            "response": {
                "ok": reply.ok,
                "content": reply.text or None,
                # An unavailable trace and an empty one are different facts.
                "reasoning_trace": {
                    "available": reply.reasoning is not None,
                    "text": reply.reasoning,
                },
                "error": reply.error,
                "attempts": reply.attempts,
                "latency_s": round(reply.latency_s, 3),
            },
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("ab") as handle:
                handle.write(orjson.dumps(record, default=str) + b"\n")
        except OSError as exc:  # pragma: no cover - disk full, permissions
            logger.warning("simulator: could not write the audit log %s: %s", path, exc)


class SimulatorPool:
    """Builds and owns one :class:`EnvironmentSimulator` per dataset.

    The clients are pooled by endpoint and model, so ten datasets on the same
    provider share one connection pool, and the run-wide semaphore is shared by
    every simulator so ``max_parallel_calls`` means what it says.
    """

    def __init__(
        self,
        config: SimulatorConfig,
        timeouts: TimeoutConfig,
        run_dir: Path | None = None,
    ) -> None:
        self.config = config
        self.timeouts = timeouts
        self.run_dir = run_dir
        self._calls = asyncio.Semaphore(max(1, config.max_parallel_calls))
        self._clients: dict[tuple[str, str], ModelClient] = {}
        self._simulators: dict[str, EnvironmentSimulator] = {}

    def for_dataset(self, dataset_id: str) -> EnvironmentSimulator | None:
        """The simulator this dataset should use, or ``None`` if it has none.

        ``None`` is the signal the adapters fall back on: no simulator
        configured means the deterministic environment, which keeps every run
        that does not opt in exactly as it was.
        """
        if dataset_id in self._simulators:
            return self._simulators[dataset_id]
        resolved = self.config.for_dataset(dataset_id)
        if not resolved.enabled or not resolved.model or not resolved.base_url:
            return None
        key = (resolved.base_url, resolved.model)
        client = self._clients.get(key)
        if client is None:
            client = ModelClient(
                ModelConfig(
                    id=f"simulator:{resolved.model}",
                    model_name=resolved.model,
                    description=f"environment simulator for {dataset_id}",
                    endpoint=EndpointConfig(
                        base_url=resolved.base_url,
                        api_key=resolved.api_key,
                    ),
                    sampling=ModelSamplingConfig(
                        temperature=resolved.temperature,
                        seed=resolved.seed,
                        max_tokens_default=resolved.max_tokens,
                        max_tokens_cap=resolved.max_tokens,
                    ),
                    limits=ModelLimitsConfig(
                        max_parallel_batches=resolved.max_parallel_calls,
                    ),
                ),
                self.timeouts,
                # A hosted API is not something this harness can restart, and
                # rediscovery probes /v1/models on a provider that may bill for
                # it. Retries are the simulator's own, bounded by max_retries.
                rediscover=False,
            )
            self._clients[key] = client
        simulator = EnvironmentSimulator(
            config=resolved,
            client=client,
            dataset_id=dataset_id,
            run_dir=self.run_dir,
            calls=self._calls,
        )
        self._simulators[dataset_id] = simulator
        return simulator

    def as_record(self) -> dict[str, Any]:
        """What the run's results should say about the simulators used."""
        return {
            dataset_id: {
                **simulator.config.as_record(),
                "calls": simulator.stats["calls"],
                "retries": simulator.stats["retries"],
                "failures": simulator.stats["failures"],
            }
            for dataset_id, simulator in sorted(self._simulators.items())
        }

    async def aclose(self) -> None:
        for client in self._clients.values():
            await client.aclose()
        self._clients.clear()
