"""OpenAI-compatible inference client with native batch support.

Network topology this implements (see ``docs/architecture.md``):

* **Discovery and single calls** go to the model's ``endpoint.base_url``
  (typically the shared LiteLLM gateway): ``GET /v1/models``,
  ``POST /v1/chat/completions``.
* **Batched calls** go to ``endpoint.batch.base_url + endpoint.batch.path``,
  which is a *different* URL: vLLM's ``POST /v1/chat/completions/batch`` is not
  an OpenAI route, so it exists only on a model's own tunnel or on a dedicated
  gateway pass-through path.  Both are expressible in config.

Batch-call semantics that the rest of the engine depends on:

* one request body carries ``messages: [conversation, conversation, ...]`` and a
  **single shared sampling-parameter set** -- hence the batching module groups
  samples by sampling signature;
* the response is one ``chat.completion`` object whose ``choices[i]`` maps to
  input conversation ``i`` (the server sorts by index; we re-validate);
* ``usage`` is the **aggregate over the whole call**, not per conversation;
* if any single conversation is invalid (e.g. too long) the whole call fails --
  which is why the executor bisects a failed batch instead of retrying it whole.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import EndpointConfig, ModelConfig, TimeoutConfig
from .errors import (
    AuthError,
    BatchProtocolError,
    EndpointError,
    TransientEndpointError,
    classify_status_code,
)
from .types import ChatMessage, SamplingParams

logger = logging.getLogger(__name__)

__all__ = ["RawChoice", "BatchResult", "ModelClient"]


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _apply_override_extra(payload: dict[str, Any], sampling: SamplingParams) -> None:
    """Lay ``sampling.override_extra`` over the payload, the model's extra included."""
    over = sampling.override_extra()
    if over:
        payload.update(_deep_merge(payload, over))


@dataclass(slots=True)
class RawChoice:
    """One normalized choice out of a (batched or single) completion."""

    index: int
    content: str | None
    reasoning: str | None
    finish_reason: str | None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BatchResult:
    """Normalized result of one batch (or single) call."""

    choices: list[RawChoice]
    usage: dict[str, Any]
    response_id: str | None
    model: str | None
    latency_s: float
    endpoint_url: str
    is_batch: bool
    raw: dict[str, Any] = field(default_factory=dict)


def _extract_message(choice: dict[str, Any]) -> tuple[str | None, str | None]:
    """Pull ``(content, reasoning)`` out of a choice, across server variants.

    vLLM 0.28 returns the chain-of-thought of a reasoning model under
    ``message.reasoning``; other builds/proxies use ``reasoning_content``.
    ``content`` legitimately comes back ``null`` when the reasoning consumed the
    whole ``max_tokens`` budget, so both fields are captured.
    """
    message = choice.get("message") or {}
    content = message.get("content")
    if isinstance(content, list):  # content-parts form
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        ) or None
    reasoning = message.get("reasoning")
    if reasoning is None:
        reasoning = message.get("reasoning_content")
    return content, reasoning


class ModelClient:
    """Async client for one model, including endpoint recovery.

    A single instance is shared by every task using that model, so its
    connection pool, rate limiter and recovery state are shared too.
    """

    def __init__(
        self,
        model: ModelConfig,
        timeouts: TimeoutConfig,
        *,
        rediscover: bool = True,
    ):
        self.model = model
        self.endpoint: EndpointConfig = model.endpoint
        self._timeouts = timeouts
        self._rediscover = rediscover
        #: Set by the first aclose(); a closed client is never reopened.
        self.closed = False
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=timeouts.connect_s,
                read=timeouts.read_s,
                write=timeouts.write_s,
                pool=timeouts.pool_s,
            ),
            verify=self.endpoint.verify_tls,
            follow_redirects=True,
            # Sized from the model's own config, because the right number is
            # not a property of HTTP: it is how many requests this run can aim
            # at this endpoint at once. A pool smaller than that does not slow
            # the surplus down, it kills it at `pool_s` -- see
            # ModelLimitsConfig.max_connections.
            limits=httpx.Limits(
                max_connections=model.limits.max_connections,
                max_keepalive_connections=max(16, model.limits.max_connections // 4),
            ),
        )
        self._base_url = self.endpoint.base_url
        self._batch_base_url = self.endpoint.resolved_batch_base_url()
        self._recovery_lock = asyncio.Lock()
        self._last_recovery_ts = 0.0
        self._rate_lock = asyncio.Lock()
        self._next_allowed_ts = 0.0
        self._min_spacing_s = (
            60.0 / model.limits.requests_per_minute if model.limits.requests_per_minute else 0.0
        )

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #

    async def aclose(self) -> None:
        """Close the connection pool. Idempotent: only the first call closes."""
        if self.closed:
            return
        self.closed = True
        await self._client.aclose()

    async def __aenter__(self) -> ModelClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    # ------------------------------------------------------------------ #
    # URLs and headers
    # ------------------------------------------------------------------ #

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def batch_url(self) -> str:
        return f"{self._batch_base_url}{self.endpoint.batch.path}"

    @property
    def supports_batch(self) -> bool:
        # A decision endpoint answers one question per request; there is no
        # batch route and no group size to fill.
        return self.endpoint.batch.enabled and not self.speaks_systemone

    @property
    def speaks_systemone(self) -> bool:
        """Is this a decision endpoint rather than a chat one?"""
        return self.endpoint.protocol == "systemone"

    def _headers(self, *, batch: bool) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        headers.update(self.endpoint.headers)
        if batch:
            headers.update(self.endpoint.batch.headers)
        key = self.endpoint.resolved_batch_api_key() if batch else self.endpoint.api_key
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    # ------------------------------------------------------------------ #
    # rate limiting
    # ------------------------------------------------------------------ #

    async def _throttle(self) -> None:
        if not self._min_spacing_s:
            return
        async with self._rate_lock:
            now = time.monotonic()
            wait = self._next_allowed_ts - now
            if wait > 0:
                await asyncio.sleep(wait)
                now = time.monotonic()
            self._next_allowed_ts = now + self._min_spacing_s

    # ------------------------------------------------------------------ #
    # requests
    # ------------------------------------------------------------------ #

    async def _post(self, url: str, payload: dict[str, Any], *, batch: bool) -> dict[str, Any]:
        await self._throttle()
        try:
            response = await self._client.post(url, json=payload, headers=self._headers(batch=batch))
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise TransientEndpointError(f"cannot connect to {url}: {exc}") from exc
        except (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout) as exc:
            raise TransientEndpointError(f"timeout talking to {url}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise TransientEndpointError(f"HTTP error talking to {url}: {exc}") from exc

        if response.status_code >= 400:
            raise classify_status_code(response.status_code, response.text)
        try:
            return response.json()
        except ValueError as exc:
            raise BatchProtocolError(
                f"non-JSON response from {url}: {response.text[:500]}",
                status_code=response.status_code,
            ) from exc

    async def list_models(self) -> list[str]:
        """Model discovery through the gateway."""
        url = f"{self._base_url}{self.endpoint.models_path}"
        try:
            response = await self._client.get(url, headers=self._headers(batch=False))
        except httpx.HTTPError as exc:
            raise TransientEndpointError(f"cannot reach {url}: {exc}") from exc
        if response.status_code >= 400:
            raise classify_status_code(response.status_code, response.text)
        data = response.json().get("data", [])
        return [entry.get("id", "") for entry in data]

    async def health_check(self) -> bool:
        """Cheap liveness probe used by ``abench doctor`` and by recovery."""
        try:
            models = await self.list_models()
        except EndpointError:
            return False
        return bool(models)

    async def decide(
        self, prompts: list[Any], sampling: SamplingParams
    ) -> BatchResult:
        """Answer selection prompts through a decision endpoint.

        NOT A CHAT CALL, and the difference is the point. A decision endpoint
        takes the evidence as ``state`` and the answer options as named
        ``criteria``, and returns which key it chose with a probability over
        all of them. It never sees the rendered prompt, so the framing the
        other models get -- the system instruction, the answer-format contract,
        the chain-of-thought instruction -- does not exist here, and the model
        cannot produce free text to be parsed. That is why this is restricted
        to selection tasks: there is nothing for it to do on a generation one.

        The sample's own fields are the source, not the rendered messages.
        Parsing options back out of a prompt this project wrote would be a
        second, silently divergent definition of what the options are.

        The reply is turned into the answer line the dataset's scorer already
        expects, so a jev result is scored by exactly the same code as every
        other model's -- which is what makes the columns comparable. The
        probabilities and confidence are kept on the raw record: they are the
        one thing this endpoint gives that a chat model does not, and throwing
        them away would waste the reason for using it.
        """
        started = time.monotonic()
        choices: list[RawChoice] = []
        usage_total: dict[str, Any] = {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}
        url = f"{self._base_url.rstrip('/')}{self.endpoint.systemone_path}"
        model_reported: str | None = None
        response_id: str | None = None

        for index, prompt in enumerate(prompts):
            fields = dict(getattr(prompt.sample, "fields", {}) or {})
            labels = [str(label) for label in (fields.get("option_labels") or [])]
            options = [str(option) for option in (fields.get("options") or [])]
            if not labels or len(labels) != len(options):
                raise BatchProtocolError(
                    f"{url}: sample {prompt.sample_id!r} has {len(labels)} option "
                    f"label(s) and {len(options)} option(s); a decision endpoint needs "
                    f"one criterion per option"
                )
            # STATE IS THE CONTENT TO EVALUATE. The question does not belong
            # here: TypeSafe's reference says state is "the content to
            # evaluate" and instructions are "what the model should decide".
            # Putting the question in the state made it one more piece of
            # evidence rather than the thing being asked.
            state = "\n\n".join(
                part for part in (
                    str(fields.get("observation") or "").strip(),
                    str(fields.get("context") or "").strip(),
                ) if part
            )
            payload = {
                "model": self.model.model_name,
                "state": state,
                "questions": {
                    "answer": {
                        "type": "choice",
                        # INSTRUCTIONS ARE WHAT TO DECIDE, so the question
                        # lives here with the dataset's own framing after it.
                        # Written as a decision and not as a reasoning
                        # instruction: this is not a language model, it scores
                        # each option against its rubric in parallel and in
                        # isolation, so "think step by step" would be a
                        # sentence with nothing to act on it.
                        "instructions": "\n".join(
                            part for part in (
                                str(fields.get("question") or "").strip(),
                                str(fields.get("instructions") or "").strip(),
                            ) if part
                        ) or "Choose the option that best explains the evidence.",
                        # Keyed by the dataset's own labels, so the chosen key
                        # IS the answer and nothing has to be matched back by
                        # text -- which would be ambiguous here anyway: agentrx
                        # offers both "Intent Not Supported" and "Intent not
                        # supported" as separate options.
                        "criteria": dict(zip(labels, options, strict=True)),
                    }
                },
            }
            data = await self._post(url, payload, batch=False)
            answer = ((data.get("answers") or {}).get("answer") or {})
            chosen = answer.get("choice")
            model_reported = model_reported or data.get("model")
            response_id = response_id or data.get("id")
            for key, value in (data.get("usage") or {}).items():
                if isinstance(value, (int, float)):
                    usage_total[key] = usage_total.get(key, 0) + value
            choices.append(
                RawChoice(
                    index=index,
                    # The answer line the dataset's own scorer reads.
                    content=(f"Answer: {chosen}" if chosen is not None else None),
                    reasoning=None,
                    finish_reason="stop" if chosen is not None else "error",
                    raw={
                        "choice": chosen,
                        "probabilities": answer.get("probabilities"),
                        "confidence": answer.get("confidence"),
                        "provider": data.get("provider"),
                    },
                )
            )
        return BatchResult(
            choices=choices,
            usage=usage_total,
            response_id=response_id,
            model=model_reported,
            latency_s=time.monotonic() - started,
            endpoint_url=url,
            is_batch=len(prompts) > 1,
        )

    async def chat_single(
        self, messages: list[ChatMessage], sampling: SamplingParams
    ) -> BatchResult:
        """One non-batched completion (fallback path and judge calls)."""
        url = f"{self._base_url}{self.endpoint.chat_path}"
        payload: dict[str, Any] = {
            "model": self.model.model_name,
            "messages": [m.to_dict() for m in messages],
            **sampling.to_payload(),
            **self.model.sampling.extra,
        }
        _apply_override_extra(payload, sampling)
        started = time.monotonic()
        data = await self._post(url, payload, batch=False)
        latency = time.monotonic() - started
        choices = data.get("choices") or []
        if not choices:
            raise BatchProtocolError(f"{url} returned no choices: {str(data)[:400]}")
        if choices[0].get("finish_reason") == "error":
            # THE PROVIDER FAILED MID-GENERATION, NOT THE MODEL. OpenRouter
            # reports an upstream failure as a 200 whose choice finishes with
            # "error": content null or cut off, usage all zeros. Taken as an
            # answer it became an EMPTY record scored as wrong -- and EMPTY is
            # reusable on resume, so it could never be retried either. Raised
            # as transient, the retry loop gets it like any other outage.
            detail = choices[0].get("error") or (choices[0].get("message") or {}).get("error")
            raise TransientEndpointError(
                f"{url} choice finished with 'error' (provider failure): {str(detail)[:300]}",
                body=str(data)[:2000],
            )
        content, reasoning = _extract_message(choices[0])
        return BatchResult(
            choices=[
                RawChoice(
                    index=0,
                    content=content,
                    reasoning=reasoning,
                    finish_reason=choices[0].get("finish_reason"),
                    raw=choices[0],
                )
            ],
            usage=data.get("usage") or {},
            response_id=data.get("id"),
            model=data.get("model"),
            latency_s=latency,
            endpoint_url=url,
            is_batch=False,
            raw=data,
        )

    async def chat_batch(
        self, conversations: list[list[ChatMessage]], sampling: SamplingParams
    ) -> BatchResult:
        """Native batch call: N conversations, one request, one shared param set.

        Raises :class:`BatchProtocolError` if the server returns a choice count
        that does not match the request, since silently mis-aligning responses
        to samples would corrupt every downstream metric.
        """
        if not conversations:
            raise ValueError("chat_batch called with no conversations")
        url = self.batch_url
        payload: dict[str, Any] = {
            "model": self.model.model_name,
            "messages": [[m.to_dict() for m in conv] for conv in conversations],
            **sampling.to_payload(),
            **self.model.sampling.extra,
        }
        _apply_override_extra(payload, sampling)
        started = time.monotonic()
        data = await self._post(url, payload, batch=True)
        latency = time.monotonic() - started

        raw_choices = data.get("choices")
        if not isinstance(raw_choices, list):
            raise BatchProtocolError(f"batch response from {url} has no choices list")
        if len(raw_choices) != len(conversations):
            raise BatchProtocolError(
                f"batch response from {url} returned {len(raw_choices)} choices for "
                f"{len(conversations)} conversations"
            )

        choices: list[RawChoice] = []
        for position, choice in enumerate(raw_choices):
            index = choice.get("index", position)
            if not isinstance(index, int) or not 0 <= index < len(conversations):
                raise BatchProtocolError(
                    f"batch response from {url} has out-of-range choice index {index!r}"
                )
            content, reasoning = _extract_message(choice)
            choices.append(
                RawChoice(
                    index=index,
                    content=content,
                    reasoning=reasoning,
                    finish_reason=choice.get("finish_reason"),
                    raw=choice,
                )
            )
        seen = {c.index for c in choices}
        if len(seen) != len(conversations):
            raise BatchProtocolError(
                f"batch response from {url} has duplicate/missing choice indices: {sorted(seen)}"
            )
        choices.sort(key=lambda c: c.index)

        return BatchResult(
            choices=choices,
            usage=data.get("usage") or {},
            response_id=data.get("id"),
            model=data.get("model"),
            latency_s=latency,
            endpoint_url=url,
            is_batch=True,
            raw={k: v for k, v in data.items() if k != "choices"},
        )

    # ------------------------------------------------------------------ #
    # recovery
    # ------------------------------------------------------------------ #

    def _run_discovery(self, command: str) -> str | None:
        try:
            proc = subprocess.run(  # noqa: S602 - command comes from trusted local config
                command, shell=True, capture_output=True, text=True, timeout=60, check=True
            )
        except subprocess.SubprocessError as exc:
            logger.warning("discovery command failed (%s): %s", command, exc)
            return None
        url = proc.stdout.strip().splitlines()[-1].strip() if proc.stdout.strip() else ""
        if not url.startswith("http"):
            logger.warning("discovery command %r produced no URL (%r)", command, url[:120])
            return None
        return url.rstrip("/")

    async def rediscover(self) -> bool:
        """Re-resolve base URLs from the configured discovery commands.

        Returns ``True`` when at least one URL changed -- meaning the tunnel had
        rotated and the next attempt should use the new address.
        """
        if not self._rediscover:
            return False
        discovery = self.endpoint.discovery
        changed = False
        loop = asyncio.get_running_loop()
        if discovery.base_url_command:
            url = await loop.run_in_executor(
                None, self._run_discovery, discovery.base_url_command
            )
            if url and url != self._base_url:
                logger.warning(
                    "model %s: base URL rotated %s -> %s", self.model.id, self._base_url, url
                )
                self._base_url = url
                changed = True
        if discovery.batch_base_url_command:
            url = await loop.run_in_executor(
                None, self._run_discovery, discovery.batch_base_url_command
            )
            if url and url != self._batch_base_url:
                logger.warning(
                    "model %s: batch URL rotated %s -> %s",
                    self.model.id,
                    self._batch_base_url,
                    url,
                )
                self._batch_base_url = url
                changed = True
        elif changed and not self.endpoint.batch.base_url:
            # Batch URL defaults to the chat base URL; keep them in sync.
            self._batch_base_url = self._base_url
        return changed

    async def recover(
        self,
        *,
        probe_interval_s: float,
        max_probe_attempts: int,
        rediscover_on_connection_error: bool = True,
    ) -> bool:
        """Bring the endpoint back: re-discover the URL, then probe until alive.

        Serialized by a lock so a burst of concurrent batch failures triggers
        one recovery, not one per failed call.  Recent successful recoveries are
        not repeated (a 30s cooldown), so callers that failed on an
        already-recovered endpoint just retry.
        """
        async with self._recovery_lock:
            if time.monotonic() - self._last_recovery_ts < 30.0:
                return True
            if rediscover_on_connection_error:
                await self.rediscover()
            for attempt in range(1, max_probe_attempts + 1):
                if await self.health_check():
                    self._last_recovery_ts = time.monotonic()
                    logger.info(
                        "model %s: endpoint healthy again after %d probe(s)",
                        self.model.id,
                        attempt,
                    )
                    return True
                logger.warning(
                    "model %s: endpoint still unreachable (probe %d/%d); waiting %.0fs",
                    self.model.id,
                    attempt,
                    max_probe_attempts,
                    probe_interval_s,
                )
                if rediscover_on_connection_error and attempt % 3 == 0:
                    await self.rediscover()
                await asyncio.sleep(probe_interval_s)
            return False

    async def verify(self, *, probe_max_tokens: int | None = None) -> dict[str, Any]:
        """Startup validation: is the model served, and is batch mode usable?

        Returns a diagnostics dict (also written into the run documentation).
        Never raises for a *missing batch route* -- the engine can fall back to
        single calls -- but does raise on auth failures, which are fatal.
        """
        report: dict[str, Any] = {
            "model_id": self.model.id,
            "model_name": self.model.model_name,
            "base_url": self._base_url,
            "batch_url": self.batch_url if self.supports_batch else None,
        }
        try:
            served = await self.list_models()
            report["served_models"] = served
            report["model_available"] = self.model.model_name in served
        except AuthError:
            raise
        except EndpointError as exc:
            report["served_models"] = []
            report["model_available"] = False
            report["discovery_error"] = str(exc)

        if self.supports_batch:
            try:
                probe = await self.chat_batch(
                    [[ChatMessage("user", "ping")], [ChatMessage("user", "ping")]],
                    SamplingParams(max_tokens=probe_max_tokens or self.model.sampling.max_tokens_floor),
                )
                report["batch_ok"] = True
                report["batch_probe_latency_s"] = round(probe.latency_s, 2)
            except AuthError:
                raise
            except EndpointError as exc:
                # The error class matters: a 404/400 means this deployment has no
                # batch route (fall back to single calls), whereas a timeout or
                # 5xx says nothing about whether the route exists -- the engine
                # must not downgrade a whole run over a momentary hiccup.
                report["batch_ok"] = False
                report["batch_error"] = str(exc)[:500]
                report["batch_error_class"] = exc.error_class.value
        else:
            report["batch_ok"] = False
            report["batch_error"] = "batch disabled in configuration"
            report["batch_error_class"] = "config"
        return report
