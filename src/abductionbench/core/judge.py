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

__all__ = ["JudgeVerdict", "JudgeStage"]


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
    ):
        self.config = config
        self.registry = registry
        self.renderer = renderer
        self.retry_policy = retry_policy
        self.cache_dir = Path(cache_dir)
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
    ) -> list[tuple[SampleSpec, ModelResponse, SampleScore]]:
        """Judge what the adapter asks to be judged; return updated scores."""
        self.template = self._template_for(adapter)
        pending: list[tuple[int, dict[str, Any], str]] = []
        for index, (sample, response, score) in enumerate(scored):
            try:
                request = adapter.judge_request(sample, response, score)
            except Exception as exc:  # noqa: BLE001
                logger.warning("adapter %s: judge_request failed: %s", adapter.dataset_id, exc)
                continue
            if not request:
                continue
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

        for chunk in iter_chunks(to_call, self.config.group_size):
            conversations = []
            for _index, fields, _key in chunk:
                spec = SampleSpec(
                    sample_id=f"judge::{_key}", fields=fields, task_kind="judge"
                )
                messages, _ = self.renderer.render(spec, self.template)
                conversations.append(messages)
            sampling = SamplingParams(
                max_tokens=self.config.max_tokens, temperature=self.config.temperature
            )
            try:
                result, _ = await with_retry(
                    lambda convs=conversations, s=sampling: (
                        self.client.chat_batch(convs, s)
                        if self.client.supports_batch and len(convs) > 1
                        else self.client.chat_single(convs[0], s)
                    ),
                    policy=self.retry_policy,
                    description=f"judge batch ({len(conversations)} item(s))",
                )
            except EndpointError as exc:
                # Counted, not just logged. Swallowing this is what let an
                # unreachable judge produce a full set of plausible-looking
                # zeros: every sample kept its seeded 0.0 and the run reported
                # it as if the model had answered and been wrong.
                self.unavailable += len(chunk)
                self.last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "judge batch failed permanently, %d verdict(s) unavailable: %s",
                    len(chunk), exc,
                )
                continue
            for (_index, _fields, key), choice in zip(chunk, result.choices, strict=True):
                verdict = self._parse(choice.content or "")
                self._cache[key] = {
                    "label": verdict.label,
                    "score": verdict.score,
                    "raw": verdict.raw[:2000],
                    "parsed": verdict.parsed,
                }

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
