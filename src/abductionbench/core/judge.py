"""Optional LLM-as-judge scoring stage.

Several abductive tasks are graded by *semantic* equivalence rather than string
overlap (e.g. "is this generated explanation the same diagnosis as the gold
one?").  Deterministic overlap metrics under-credit correct paraphrases there,
so adapters may declare a judge request per sample and the engine will run a
second, batched inference pass to obtain a verdict.

The stage is off unless ``engine.judge.enabled`` is true, and the judge prompt
is an ordinary versioned template -- so a judged metric is exactly as
configurable and as auditable as the main prompts.  Verdicts are cached on disk
keyed by (template, fields), so re-scoring a run costs nothing.

Adapters opt in by overriding
:meth:`~abductionbench.core.adapter.DatasetAdapter.judge_request` and
:meth:`~abductionbench.core.adapter.DatasetAdapter.apply_judge`.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import orjson

from .adapter import DatasetAdapter
from .batching import iter_chunks
from .client import ModelClient
from .config import JudgeConfig
from .errors import ConfigError, EndpointError
from .metrics import extract_first_number
from .prompts import PromptRegistry, PromptRenderer
from .retry import RetryPolicy, with_retry
from .types import ModelResponse, SampleScore, SampleSpec, SamplingParams, stable_hash

logger = logging.getLogger(__name__)

__all__ = ["JudgeVerdict", "JudgeStage", "clip_middle", "exceeds_budget"]


def exceeds_budget(parts: dict[str, str], limits: dict[str, int]) -> str | None:
    """Why this exchange is too big to judge, or ``None`` if it fits.

    Skipping beats clipping. A clipped exchange still gets a verdict, and that
    verdict is reported beside verdicts read from complete ones as though the
    two were the same measurement -- a judge shown the middle of a chain removed
    is being asked a different question, and nothing downstream can tell. A
    skipped record has no verdict, is counted, and is named in the coverage
    report, so the gap is visible instead of silently averaged in.
    """
    for name, limit in limits.items():
        text = parts.get(name) or ""
        if limit and len(text) > limit:
            return f"{name}_exceeds_{limit}_chars_at_{len(text)}"
    return None


def clip_middle(text: str, limit: int) -> str:
    """Keep ``text`` under ``limit`` characters, losing the middle if it must.

    Shared by both judge stages, because both face the same problem: a judge
    should be shown everything the model was given and everything it produced,
    and on the longest items that does not fit in the judge's own context
    window -- a request that overruns is rejected outright, so the sample gets
    no verdict rather than a slightly thinner one.

    The middle goes rather than the tail. The opening of a prompt says what the
    task is and the opening of a response says what the model set out to do;
    the close of a response is where it commits to an answer. Truncating from
    the end throws away the answer, which is the one part every judge needs.
    The cut is marked, so a judge is never silently shown a doctored document.
    """
    if limit <= 0 or len(text) <= limit:
        return text
    half = max(1, (limit - 80) // 2)
    return (
        text[:half]
        + f"\n\n[... {len(text) - 2 * half} characters omitted from the middle ...]\n\n"
        + text[-half:]
    )


@dataclass(slots=True)
class JudgeVerdict:
    """Parsed judge output."""

    label: str | None = None
    score: float | None = None
    raw: str = ""
    parsed: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def positive(self) -> bool:
        """Convenience: did the judge affirm?"""
        if self.score is not None:
            return self.score >= 0.5
        return (self.label or "").strip().lower() in {"yes", "true", "correct", "match", "1"}


class JudgeStage:
    """Runs the judge model over samples an adapter asked to have judged."""

    def __init__(
        self,
        *,
        config: JudgeConfig,
        registry: PromptRegistry,
        renderer: PromptRenderer,
        clients: dict[str, ModelClient],
        retry_policy: RetryPolicy,
        cache_dir: Path,
        batch_disabled: set[str] | None = None,
        calls: asyncio.Semaphore | None = None,
    ):
        self.config = config
        self.registry = registry
        self.renderer = renderer
        self.retry_policy = retry_policy
        self.cache_dir = Path(cache_dir)
        #: Live view of the models whose batch route the engine found unusable.
        #: Held by reference, because the engine discovers this after the stage
        #: is built.
        self._batch_disabled = batch_disabled if batch_disabled is not None else set()
        #: Shared with the reasoning judge and with every other task, because
        #: they all queue on the same judge server. Without it this stage issued
        #: its batches strictly one after another while 29 of the 44 datasets
        #: waited on it.
        self._calls = calls or asyncio.Semaphore(config.max_parallel_calls)
        if config.model not in clients:
            raise ConfigError(
                f"engine.judge.model={config.model!r} is not one of the run's models "
                f"({sorted(clients)}); add it to the run's model list"
            )
        self.client = clients[config.model]
        self.registry = registry
        self.default_template = registry.get(config.template)
        #: Resolved per adapter: a dataset whose task has its own grading
        #: criteria gets its own judge prompt, because "is this the same
        #: hypothesis?" and "does this explanation make the outcome less
        #: surprising?" are different questions and a single generic judge
        #: answers neither well.  Adapters that declare nothing keep the
        #: configured default.
        self.template = self.default_template
        #: Verdicts that were asked for and never obtained, because the judge
        #: endpoint failed permanently.  Read by the engine after `apply`: for
        #: a dataset with no verifiable answer the judge *is* the score, so a
        #: missing verdict has to be a task failure rather than a zero.
        self.unavailable: int = 0
        #: Samples not judged because the exchange was too big for the judge's
        #: window. Reported, never averaged in as a zero.
        self.skipped_oversize: int = 0
        self.last_error: str = ""
        self._cache: dict[str, dict[str, Any]] = {}
        self._cache_path = self.cache_dir / "verdicts.json"
        if config.cache and self._cache_path.exists():
            try:
                self._cache = orjson.loads(self._cache_path.read_bytes())
            except orjson.JSONDecodeError:
                logger.warning("judge cache %s is corrupt; starting fresh", self._cache_path)

    # ------------------------------------------------------------------ #

    def _template_for(self, adapter: DatasetAdapter):
        """The judge prompt this dataset is graded with.

        An unknown id is a configuration mistake in the adapter, not a reason
        to abandon the judged metric, so it falls back to the configured
        default and says so once.
        """
        wanted = getattr(adapter, "judge_template", None)
        if not wanted or wanted == self.default_template.id:
            return self.default_template
        try:
            return self.registry.get(wanted)
        except Exception as exc:  # noqa: BLE001 - a bad id must not lose the metric
            logger.warning(
                "adapter %s asks for judge template %r which is unavailable (%s); "
                "grading with %s instead",
                adapter.dataset_id, wanted, exc, self.default_template.id,
            )
            return self.default_template

    async def apply(
        self,
        adapter: DatasetAdapter,
        scored: list[tuple[SampleSpec, ModelResponse, SampleScore]],
        prompts: list[Any] | None = None,
    ) -> list[tuple[SampleSpec, ModelResponse, SampleScore]]:
        """Judge what the adapter asks to be judged; return updated scores.

        Every request also carries the *whole* exchange -- the conversation the
        model was sent and the whole of what it replied -- added here rather
        than by each adapter, so all 29 judged datasets get it and none can
        forget to. An adapter's own fields stay authoritative: they name the
        candidate and the reference, which is what the verdict is about. These
        two are the context for reading them, and the difference matters on a
        chain-of-thought answer, where the adapter's `candidate` is one line and
        the reasoning that produced it is the rest of the response.
        """
        self.template = self._template_for(adapter)
        by_id = {p.sample_id: p for p in (prompts or [])}
        pending: list[tuple[int, dict[str, Any], str]] = []
        for index, (sample, response, score) in enumerate(scored):
            try:
                request = adapter.judge_request(sample, response, score)
            except Exception as exc:  # noqa: BLE001
                logger.warning("adapter %s: judge_request failed: %s", adapter.dataset_id, exc)
                continue
            if not request:
                continue
            exchange = self._whole_exchange(by_id.get(sample.sample_id), response)
            too_big = exceeds_budget(
                exchange,
                {
                    "full_prompt": self.config.max_prompt_chars,
                    "full_response": self.config.max_response_chars,
                },
            )
            if too_big:
                # No verdict rather than a verdict on a cut-down exchange: the
                # two would be reported as the same measurement.
                self.skipped_oversize += 1
                score.details.setdefault("judge_skipped", too_big)
                logger.debug("judge: skipping %s -- %s", sample.sample_id, too_big)
                continue
            request = {**exchange, **request}
            key = stable_hash({"template": self.template.ref, "fields": request})
            pending.append((index, request, key))

        if not pending:
            return scored

        to_call = [item for item in pending if item[2] not in self._cache]
        logger.info(
            "judge stage: %d sample(s) to judge (%d cached) with %s via %s",
            len(to_call),
            len(pending) - len(to_call),
            self.template.ref,
            self.config.model,
        )

        # One conversation per call when the batch route is unusable:
        # chat_single answers only the first conversation, so a group sent
        # without batching would lose every other item in it.
        can_batch = self.client.supports_batch and self.config.model not in self._batch_disabled
        chunk_size = self.config.group_size if can_batch else 1

        async def run_chunk(chunk: list[tuple[int, dict[str, Any], str]]) -> None:
            conversations = []
            for _index, fields, _key in chunk:
                spec = SampleSpec(
                    sample_id=f"judge::{_key}", fields=fields, task_kind="judge"
                )
                messages, _ = self.renderer.render(spec, self.template)
                conversations.append(messages)
            sampling = SamplingParams(
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                # A verdict is a short classification; an unbounded reasoning
                # chain on one costs time and buys nothing.
                extra=(
                    (("reasoning_effort", self.config.reasoning_effort),)
                    if self.config.reasoning_effort
                    else ()
                ),
            )
            async with self._calls:
                try:
                    result, _ = await with_retry(
                        lambda convs=conversations, s=sampling: (
                            self.client.chat_batch(convs, s)
                            if can_batch and len(convs) > 1
                            else self.client.chat_single(convs[0], s)
                        ),
                        policy=self.retry_policy,
                        description=f"judge batch ({len(conversations)} item(s))",
                    )
                except EndpointError as exc:
                    # Counted, not just logged. Swallowing this is what let an
                    # unreachable judge produce a full set of plausible-looking
                    # zeros: every sample kept its seeded 0.0 and the run
                    # reported it as if the model had answered and been wrong.
                    self.unavailable += len(chunk)
                    self.last_error = f"{type(exc).__name__}: {exc}"
                    logger.warning(
                        "judge batch failed permanently, %d verdict(s) unavailable: %s",
                        len(chunk), exc,
                    )
                    return
            if len(result.choices) != len(chunk):
                # Misalignment would file one sample's verdict against another.
                self.unavailable += len(chunk)
                self.last_error = (
                    f"{len(result.choices)} verdict(s) for {len(chunk)} request(s)"
                )
                logger.warning("judge batch: %s; dropping the group", self.last_error)
                return
            for (_index, _fields, key), choice in zip(chunk, result.choices, strict=True):
                verdict = self._parse(choice.content or "")
                self._cache[key] = {
                    "label": verdict.label,
                    "score": verdict.score,
                    "raw": verdict.raw[:2000],
                    "parsed": verdict.parsed,
                }

        # The batches go out together, throttled only by the shared semaphore.
        await asyncio.gather(
            *(run_chunk(chunk) for chunk in iter_chunks(to_call, chunk_size))
        )

        if self.config.cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_bytes(orjson.dumps(self._cache, option=orjson.OPT_INDENT_2))

        updated = list(scored)
        for index, _fields, key in pending:
            cached = self._cache.get(key)
            if not cached:
                continue
            verdict = JudgeVerdict(
                label=cached.get("label"),
                score=cached.get("score"),
                raw=cached.get("raw", ""),
                parsed=bool(cached.get("parsed")),
            )
            sample, response, score = updated[index]
            try:
                new_score = adapter.apply_judge(sample, response, score, verdict)
            except Exception as exc:  # noqa: BLE001
                logger.warning("adapter %s: apply_judge failed: %s", adapter.dataset_id, exc)
                continue
            updated[index] = (sample, response, new_score or score)
        return updated

    # ------------------------------------------------------------------ #

    def _whole_exchange(self, prompt: Any, response: ModelResponse) -> dict[str, Any]:
        """The conversation sent and the reply received, both in full.

        Budgeted rather than unbounded: the two together can exceed the judge's
        own context window on the longest items, and a request that overruns is
        rejected outright -- no verdict at all, which is strictly worse than a
        verdict read from a marked, middle-clipped copy.
        """
        fields: dict[str, Any] = {}
        if prompt is not None:
            fields["full_prompt"] = "\n\n".join(
                f"[{message.role.upper()}]\n{message.content}" for message in prompt.messages
            )
        whole = response.content or ""
        if response.reasoning:
            # A reasoning model's chain arrives in its own channel; it is part
            # of what the model produced and the judge should see it.
            whole = f"[REASONING]\n{response.reasoning}\n\n[ANSWER]\n{whole}"
        if whole:
            fields["full_response"] = whole
        return fields

    def _parse(self, text: str) -> JudgeVerdict:
        """Parse a judge response using the template's ``output_contract``.

        Recognized contract keys: ``verdict_regex`` (group ``label``),
        ``score_regex`` (group ``score``), ``labels`` (accepted label values).
        Falls back to looking for yes/no and for the first number in the text.
        """
        contract = self.template.output_contract or {}
        raw = (text or "").strip()
        verdict = JudgeVerdict(raw=raw)

        verdict_regex = contract.get("verdict_regex")
        if verdict_regex:
            match = re.search(verdict_regex, raw, flags=re.IGNORECASE | re.DOTALL)
            if match:
                groups = match.groupdict()
                verdict.label = (groups.get("label") or match.group(1) if match.groups() else None)
                verdict.parsed = True

        raw_score: float | None = None
        score_regex = contract.get("score_regex")
        if score_regex:
            match = re.search(score_regex, raw, flags=re.IGNORECASE | re.DOTALL)
            if match:
                groups = match.groupdict()
                candidate = groups.get("score") or (match.group(1) if match.groups() else None)
                raw_score = extract_first_number(candidate or "")

        if verdict.label is None:
            labels = [str(label) for label in contract.get("labels", ["yes", "no"])]
            escaped = "|".join(re.escape(label) for label in labels)
            found = re.findall(rf"\b({escaped})\b", raw, flags=re.IGNORECASE)
            if found:
                verdict.label = found[-1].lower()
                verdict.parsed = True

        if raw_score is None and contract.get("expect_numeric_score"):
            raw_score = extract_first_number(raw)

        if raw_score is not None:
            # Graded templates declare their scale; normalize to [0, 1] so the
            # same verdict object is comparable across judge templates.
            scale = float(contract.get("score_scale", 1.0) or 1.0)
            verdict.score = raw_score / scale if scale else raw_score
            verdict.parsed = True
        return verdict
