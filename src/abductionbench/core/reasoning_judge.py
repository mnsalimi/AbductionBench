"""LLM-judged structural metrics for chains of thought.

The stage is deliberately independent of answer-quality judging.  It runs only
for prompt modes that request a chain of reasoning (``cot`` and
``self-consistency``), caches the question-only observation inventory across
models/repeats, obtains each requested metric family with exactly one prompt,
and computes shared normalizations locally from the parsed raw counts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import orjson

from .adapter import DatasetAdapter
from .batching import iter_chunks
from .client import ModelClient
from .config import ReasoningJudgeConfig
from .errors import ConfigError, EndpointError
from .modes import BOV, COT, SELF_CONSISTENCY
from .prompts import PromptRegistry, PromptRenderer, PromptTemplate
from .retry import RetryPolicy, with_retry
from .types import (
    ModelResponse,
    RenderedPrompt,
    SampleScore,
    SampleSpec,
    SamplingParams,
    TaskIdentity,
    stable_hash,
)

logger = logging.getLogger(__name__)

__all__ = ["ReasoningJudgeStage", "derive_reasoning_metrics"]

_REQUIRED_TEMPLATES = {
    "observation_inventory",
    "observation_coverage",
    "branchiness_diversity",
    "density",
    "redundancy_completeness",
    "directionality",
    "backtracking",
    "differential_elimination",
    "prior_knowledge",
    "uncertainty",
}


@dataclass(slots=True)
class _Target:
    index: int
    sample: SampleSpec
    response: ModelResponse
    score: SampleScore
    prompt: RenderedPrompt
    question: str
    reasoning: str
    answer: str
    reference: str
    options: list[str]
    generation_like: bool
    selection_like: bool
    pipeline_like: bool
    bov: bool
    raw: dict[str, dict[str, Any]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    inapplicable: list[str] = field(default_factory=list)


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, float) and value >= 0 and value.is_integer():
        return int(value)
    return None


def _nonnegative_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number >= 0 and math.isfinite(number) else None


def _binary(value: Any) -> int | None:
    parsed = _nonnegative_int(value)
    return parsed if parsed in (0, 1) else None


def derive_reasoning_metrics(
    raw: dict[str, dict[str, Any]],
    *,
    generation_like: bool,
    selection_like: bool,
    option_count: int,
    pipeline_like: bool = False,
    bov: bool = False,
) -> tuple[dict[str, float], list[str], list[str]]:
    """Validate judge outputs and calculate every shared/derived value.

    Invalid or undefined quantities are omitted, never silently coerced to
    zero.  The caller writes ``errors`` and ``inapplicable`` into the sample
    sheet so a missing value always has an auditable explanation.
    """
    metrics: dict[str, float] = {}
    errors: list[str] = []
    inapplicable: list[str] = []

    inventory = raw.get("observation_inventory") or {}
    total_observations = _nonnegative_int(inventory.get("total_observations"))
    if total_observations is None:
        errors.append("observation_inventory:invalid_or_missing_total")
    else:
        metrics["reasoning_observations_total"] = float(total_observations)

    coverage = raw.get("observation_coverage") or {}
    coverage_total = _nonnegative_int(coverage.get("total_observations"))
    observations_used = _nonnegative_int(coverage.get("observations_used"))
    if total_observations is None:
        errors.append("observation_coverage:missing_cached_total")
    elif coverage_total != total_observations:
        errors.append("observation_coverage:total_mismatch")
    elif observations_used is None or observations_used > total_observations:
        errors.append("observation_coverage:invalid_used_count")
    else:
        metrics["reasoning_observations_used"] = float(observations_used)
        if total_observations > 0:
            metrics["reasoning_observation_coverage"] = observations_used / total_observations
        else:
            errors.append("observation_coverage:normalization_total_is_zero")

    branchiness: int | None = None
    if generation_like or pipeline_like:
        branch = raw.get("branchiness_diversity") or {}
        branchiness = _nonnegative_int(branch.get("branchiness"))
        diversity = _binary(branch.get("diversity"))
        if branchiness is None or diversity is None:
            errors.append("branchiness_diversity:invalid_or_missing_output")
        else:
            metrics["reasoning_branchiness"] = float(branchiness)
            metrics["reasoning_diversity"] = float(diversity)
    else:
        inapplicable.append("branchiness_diversity:not_generation_or_pipeline")

    total_steps: int | None = None
    density_value: float | None = None
    density = raw.get("density") or {}
    total_steps = _nonnegative_int(density.get("total_steps"))
    useless_steps = _nonnegative_int(density.get("useless_steps"))
    useful_steps = _nonnegative_int(density.get("useful_steps"))
    density_value = _nonnegative_number(density.get("reasoning_density"))
    if (
        total_steps is None
        or useless_steps is None
        or useful_steps is None
        or density_value is None
        or useful_steps + useless_steps != total_steps
    ):
        errors.append("density:invalid_counts_or_sum")
        total_steps = None
        density_value = None
    else:
        metrics.update(
            {
                "reasoning_total_steps": float(total_steps),
                "reasoning_useless_steps": float(useless_steps),
                "reasoning_useful_steps": float(useful_steps),
                "reasoning_density": density_value,
            }
        )
        if total_steps > 0:
            metrics["reasoning_useless_step_fraction"] = useless_steps / total_steps
            metrics["reasoning_useful_step_fraction"] = useful_steps / total_steps
        else:
            errors.append("density:step_fraction_total_is_zero")

        if pipeline_like:
            if branchiness is not None and branchiness > 0:
                metrics["reasoning_density_normalized"] = density_value / branchiness
            else:
                errors.append("density:pipeline_branchiness_is_zero_or_unavailable")
        elif selection_like and not bov:
            if option_count > 0:
                metrics["reasoning_density_normalized"] = density_value / option_count
            else:
                errors.append("density:selection_option_count_unavailable")
        elif generation_like:
            if branchiness is not None and branchiness > 0:
                metrics["reasoning_density_normalized"] = density_value / branchiness
            else:
                errors.append("density:generation_branchiness_is_zero_or_unavailable")
        elif bov:
            errors.append("density:bov_full_option_count_not_visible_in_each_chain")

    redundancy = raw.get("redundancy_completeness") or {}
    redundant = _nonnegative_int(redundancy.get("redundancy"))
    complete = _nonnegative_int(redundancy.get("completeness"))
    if redundant is None or complete is None:
        errors.append("redundancy_completeness:invalid_or_missing_output")
    else:
        metrics["reasoning_redundancy"] = float(redundant)
        metrics["reasoning_completeness"] = float(complete)
        if total_observations is None:
            errors.append("redundancy_completeness:missing_observation_normalizer")
        elif total_observations == 0:
            errors.append("redundancy_completeness:normalization_total_is_zero")
        elif redundant > total_observations or complete > total_observations:
            errors.append("redundancy_completeness:count_exceeds_total_observations")
        else:
            metrics["reasoning_redundancy_normalized"] = redundant / total_observations
            metrics["reasoning_completeness_normalized"] = complete / total_observations

    directionality = _nonnegative_number((raw.get("directionality") or {}).get("directionality"))
    if directionality not in (0.0, 0.5, 1.0):
        errors.append("directionality:expected_0_0.5_or_1")
    else:
        metrics["reasoning_directionality"] = directionality

    backtracking = _nonnegative_int((raw.get("backtracking") or {}).get("backtracking"))
    if backtracking is None:
        errors.append("backtracking:invalid_or_missing_output")
    else:
        metrics["reasoning_backtracking"] = float(backtracking)
        if total_steps is not None and total_steps > 0 and backtracking <= total_steps:
            metrics["reasoning_backtracking_normalized"] = backtracking / total_steps
        else:
            errors.append("backtracking:invalid_or_missing_step_normalizer")

    if selection_like or pipeline_like:
        if bov:
            inapplicable.append("differential_elimination:bov_shows_one_option_per_chain")
        elif option_count < 2:
            errors.append("differential_elimination:fewer_than_two_visible_options")
        else:
            differential = _nonnegative_int(
                (raw.get("differential_elimination") or {}).get("differential_elimination")
            )
            combinations = (2**option_count) - option_count - 1
            if differential is None or differential > combinations:
                errors.append("differential_elimination:invalid_or_exceeds_possible")
            else:
                metrics["reasoning_differential_elimination"] = float(differential)
                metrics["reasoning_differential_elimination_normalized"] = (
                    differential / combinations
                )
    else:
        inapplicable.append("differential_elimination:not_selection_or_pipeline")

    prior = _binary((raw.get("prior_knowledge") or {}).get("prior_knowledge"))
    if prior is None:
        errors.append("prior_knowledge:invalid_or_missing_output")
    else:
        metrics["reasoning_prior_knowledge"] = float(prior)

    uncertainty = _nonnegative_int(
        (raw.get("uncertainty") or {}).get("uncertainty_steps")
    )
    if uncertainty is None:
        errors.append("uncertainty:invalid_or_missing_output")
    else:
        metrics["reasoning_uncertainty_steps"] = float(uncertainty)
        if total_steps is not None and total_steps > 0 and uncertainty <= total_steps:
            metrics["reasoning_uncertainty_normalized"] = uncertainty / total_steps
        else:
            errors.append("uncertainty:invalid_or_missing_step_normalizer")

    return metrics, errors, inapplicable


class ReasoningJudgeStage:
    """Run and cache all reasoning-chain judge prompts."""

    def __init__(
        self,
        *,
        config: ReasoningJudgeConfig,
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
                f"engine.reasoning_judge.model={config.model!r} is not one of the run's "
                f"models ({sorted(clients)}); add it to the run's model list"
            )
        missing = _REQUIRED_TEMPLATES - set(config.templates)
        if missing:
            raise ConfigError(
                "engine.reasoning_judge.templates is missing: " + ", ".join(sorted(missing))
            )
        self.client = clients[config.model]
        self.templates: dict[str, PromptTemplate] = {
            name: registry.get(template_id) for name, template_id in config.templates.items()
        }
        self._cache_path = self.cache_dir / "verdicts.json"
        self._cache: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        if config.cache and self._cache_path.exists():
            try:
                payload = orjson.loads(self._cache_path.read_bytes())
                if isinstance(payload, dict):
                    self._cache = payload
            except orjson.JSONDecodeError:
                logger.warning(
                    "reasoning judge cache %s is corrupt; starting fresh", self._cache_path
                )

    async def apply(
        self,
        adapter: DatasetAdapter,
        identity: TaskIdentity,
        prompts: list[RenderedPrompt],
        scored: list[tuple[SampleSpec, ModelResponse, SampleScore]],
    ) -> list[tuple[SampleSpec, ModelResponse, SampleScore]]:
        """Add reasoning metrics to COT samples and leave every IO score untouched."""
        if identity.prompt_mode not in (COT, SELF_CONSISTENCY) or not scored:
            return scored

        prompt_by_id = {prompt.sample_id: prompt for prompt in prompts}
        try:
            processing_mode = adapter.documentation().processing_mode.lower()
        except Exception:  # pragma: no cover - adapters already document during preparation
            processing_mode = ""
        pipeline = "generation & selection" in processing_mode and "separate" not in processing_mode

        targets: list[_Target] = []
        updated = list(scored)
        for index, (sample, response, score) in enumerate(scored):
            prompt = prompt_by_id.get(sample.sample_id)
            if prompt is None:
                score.details = {
                    **score.details,
                    "reasoning_metrics_status": "not_applicable:prompt_unavailable",
                }
                continue
            question, reasoning = self._question_and_reasoning(prompt, response)
            if not reasoning.strip():
                score.details = {
                    **score.details,
                    "reasoning_metrics_status": "not_applicable:no_reasoning_chain",
                }
                continue
            root = self._root_sample(sample)
            options = [str(value) for value in (root.fields.get("options") or [])]
            selection_like = sample.task_kind == "selection"
            generation_like = sample.task_kind == "generation"
            targets.append(
                _Target(
                    index=index,
                    sample=sample,
                    response=response,
                    score=score,
                    prompt=prompt,
                    question=question,
                    reasoning=reasoning,
                    answer=response.text,
                    reference=json.dumps(sample.reference, ensure_ascii=False, default=str),
                    options=options,
                    generation_like=generation_like,
                    selection_like=selection_like,
                    pipeline_like=pipeline,
                    bov=identity.selection_mode == BOV,
                )
            )

        if not targets:
            return updated

        # Serializing apply() prevents two concurrently-running model tasks from
        # buying the same question-only observation inventory before either has
        # put it into the shared run cache.
        async with self._lock:
            await self._evaluate(targets)
            self._save_cache()

        for target in targets:
            metrics, errors, inapplicable = derive_reasoning_metrics(
                target.raw,
                generation_like=target.generation_like,
                selection_like=target.selection_like,
                pipeline_like=target.pipeline_like,
                option_count=len(target.options),
                bov=target.bov,
            )
            errors = [*target.errors, *errors]
            details = dict(target.score.details)
            details["reasoning_metrics_status"] = "ok" if not errors else "partial"
            if errors:
                details["reasoning_judge_errors"] = errors
            if inapplicable:
                details["reasoning_metrics_inapplicable"] = inapplicable
            new_score = SampleScore(
                metrics={**target.score.metrics, **metrics},
                prediction=target.score.prediction,
                parse_ok=target.score.parse_ok,
                details=details,
            )
            updated[target.index] = (target.sample, target.response, new_score)
        return updated

    async def _evaluate(self, targets: list[_Target]) -> None:
        inventory_requests: dict[str, dict[str, Any]] = {}
        question_keys: dict[int, str] = {}
        for target in targets:
            question_key = stable_hash({"question": target.question}, length=32)
            question_keys[target.index] = question_key
            inventory_requests.setdefault(question_key, {"question": target.question})
        inventories = await self._judge_many("observation_inventory", inventory_requests)

        common: dict[int, dict[str, Any]] = {}
        for target in targets:
            inventory = inventories.get(question_keys[target.index])
            if inventory is None:
                target.errors.append("observation_inventory:judge_failed_or_unparseable")
                inventory = {}
            target.raw["observation_inventory"] = inventory
            total = _nonnegative_int(inventory.get("total_observations"))
            common[target.index] = {
                "question": target.question,
                "reasoning_chain": target.reasoning,
                "model_answer": target.answer,
                "reference_answer": target.reference,
                "options": target.options,
                "option_count": len(target.options),
                "total_observations": total,
            }

        async def run_family(
            family: str,
            selected: list[_Target],
            field_names: tuple[str, ...],
        ) -> None:
            requests = {
                str(target.index): {
                    key: common[target.index][key]
                    for key in field_names
                    if common[target.index].get(key) is not None
                }
                for target in selected
            }
            # A missing required dependency is reported rather than rendering a
            # made-up zero into the prompt.
            template = self.templates[family]
            callable_requests = {
                key: fields
                for key, fields in requests.items()
                if all(required in fields for required in template.required_fields)
            }
            results = await self._judge_many(family, callable_requests)
            for target in selected:
                key = str(target.index)
                if key not in callable_requests:
                    target.errors.append(f"{family}:missing_required_dependency")
                    target.raw[family] = {}
                elif results.get(key) is None:
                    target.errors.append(f"{family}:judge_failed_or_unparseable")
                    target.raw[family] = {}
                else:
                    target.raw[family] = results[key]

        await run_family(
            "observation_coverage",
            targets,
            ("question", "reasoning_chain", "total_observations"),
        )
        generation = [
            target for target in targets if target.generation_like or target.pipeline_like
        ]
        if generation:
            await run_family(
                "branchiness_diversity", generation, ("question", "reasoning_chain")
            )
        await run_family("density", targets, ("question", "reasoning_chain", "options"))
        await run_family(
            "redundancy_completeness",
            targets,
            ("question", "reasoning_chain", "model_answer", "reference_answer"),
        )
        await run_family("directionality", targets, ("question", "reasoning_chain"))
        await run_family("prior_knowledge", targets, ("question", "reasoning_chain"))

        # These two prompts use the density judge's step count as the shared
        # normalizer and to keep their step segmentation aligned.
        for target in targets:
            common[target.index]["total_steps"] = _nonnegative_int(
                target.raw.get("density", {}).get("total_steps")
            )
        await run_family(
            "backtracking", targets, ("question", "reasoning_chain", "total_steps")
        )
        await run_family(
            "uncertainty", targets, ("question", "reasoning_chain", "total_steps")
        )

        differential = [
            target
            for target in targets
            if (target.selection_like or target.pipeline_like)
            and not target.bov
            and len(target.options) >= 2
        ]
        if differential:
            await run_family(
                "differential_elimination",
                differential,
                ("question", "reasoning_chain", "options", "option_count"),
            )

    async def _judge_many(
        self, family: str, requests: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any] | None]:
        template = self.templates[family]
        keyed: list[tuple[str, dict[str, Any], str]] = []
        out: dict[str, dict[str, Any] | None] = {}
        for request_id, fields in requests.items():
            key = stable_hash({"template": template.ref, "fields": fields}, length=32)
            cached = self._cache.get(key)
            if cached is not None:
                values = cached.get("values")
                out[request_id] = values if isinstance(values, dict) else None
            else:
                keyed.append((request_id, fields, key))

        for chunk in iter_chunks(keyed, self.config.group_size):
            conversations = []
            for _request_id, fields, key in chunk:
                sample = SampleSpec(sample_id=f"reasoning-judge::{key}", fields=fields,
                                    task_kind="judge")
                messages, _contract = self.renderer.render(sample, template)
                conversations.append(messages)
            sampling = SamplingParams(
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
            try:
                result, _outcome = await with_retry(
                    lambda convs=conversations, params=sampling: (
                        self.client.chat_batch(convs, params)
                        if self.client.supports_batch and len(convs) > 1
                        else self.client.chat_single(convs[0], params)
                    ),
                    policy=self.retry_policy,
                    description=f"reasoning judge {family} ({len(conversations)} item(s))",
                )
            except EndpointError as exc:
                logger.warning("reasoning judge %s failed permanently: %s", family, exc)
                for request_id, _fields, _key in chunk:
                    out[request_id] = None
                continue
            for (request_id, _fields, key), choice in zip(
                chunk, result.choices, strict=True
            ):
                raw = choice.content or ""
                values = self._parse_json(raw, template)
                self._cache[key] = {
                    "template": template.ref,
                    "values": values,
                    "raw": raw[:2000],
                    "parsed": values is not None,
                }
                out[request_id] = values
        return out

    @staticmethod
    def _parse_json(text: str, template: PromptTemplate) -> dict[str, Any] | None:
        raw = (text or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
        try:
            parsed = orjson.loads(raw)
        except orjson.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if not match:
                return None
            try:
                parsed = orjson.loads(match.group(0))
            except orjson.JSONDecodeError:
                return None
        if not isinstance(parsed, dict):
            return None
        fields = (template.output_contract or {}).get("json_fields") or {}
        if any(name not in parsed for name in fields):
            return None
        return parsed

    def _save_cache(self) -> None:
        if not self.config.cache:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        temp = self._cache_path.with_suffix(".json.tmp")
        temp.write_bytes(orjson.dumps(self._cache, option=orjson.OPT_INDENT_2))
        temp.replace(self._cache_path)

    @staticmethod
    def _root_sample(sample: SampleSpec) -> SampleSpec:
        root = sample
        seen: set[int] = set()
        while id(root) not in seen:
            seen.add(id(root))
            parent = root.metadata.get("_parent_sample")
            if not isinstance(parent, SampleSpec):
                break
            root = parent
        return root

    @staticmethod
    def _question_and_reasoning(
        prompt: RenderedPrompt, response: ModelResponse
    ) -> tuple[str, str]:
        transcript = prompt.sample.metadata.get("_transcript")
        if isinstance(transcript, list) and transcript:
            question_parts: list[str] = []
            reasoning_parts: list[str] = []
            for message in transcript:
                if not isinstance(message, dict):
                    continue
                role = str(message.get("role", "unknown"))
                content = str(message.get("content", ""))
                if role == "assistant":
                    if content:
                        reasoning_parts.append(content)
                else:
                    question_parts.append(f"[{role.upper()}]\n{content}")
            if response.reasoning:
                reasoning_parts.append(response.reasoning)
            return "\n\n".join(question_parts), "\n\n".join(reasoning_parts)

        question = "\n\n".join(
            f"[{message.role.upper()}]\n{message.content}" for message in prompt.messages
        )
        reasoning = response.reasoning or response.content or ""
        return question, reasoning
