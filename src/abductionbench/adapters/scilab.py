"""SciLab (LLM-AutoSciLab): run experiments on a hidden law, then state it.

Source: https://github.com/scientific-discovery/LLM-AutoSciLab

Like BoxingGym, SciLab ships no item file, because an item is a *laboratory*: the
release vendors NewtonBench's twelve physics domains, each of which hides a law
the agent has to discover by experiment.  The adapter runs those oracles.

**Why the laws are not the textbook ones.**  NewtonBench deliberately modifies
them -- gravity is ``F = C·m₁·m₂/r^1.5`` at easy, ``C·(m₁+m₂)²/r^1.5`` at hard --
so a model cannot succeed by recalling physics.  It has to infer the mechanism
from what the apparatus actually returns, which is what makes this abduction
rather than recall, and it is why an episode's evidence has to come from
experiments the model chose.

**The loop.**  The model sets the domain's parameters and receives a
measurement, up to ``options.experiments`` times; then it is asked to predict
the measurement at settings it has not seen, and to state the law it inferred.

**Scoring.**  Predictive error on the held-out settings, as a median relative
error -- objective, and comparable across domains whose measurements differ by
orders of magnitude.  The stated law is recorded next to it for inspection but
is not scored by string comparison: the release grades a law with an LLM judge,
and using one here would put a second model inside the evaluation of the first.
Prediction is the consequence the hypothesis is answerable to.
"""

from __future__ import annotations

import json
import math
import random
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..core.adapter import SkippedDataset
from ..core.metrics import aggregate_mean_metrics, mean
from ..core.types import (
    AdapterDocumentation,
    ChatMessage,
    ModelResponse,
    SampleScore,
    SampleSpec,
)
from . import _common as C
from ._base import PooledDatasetAdapter
from ._interactive import InteractiveMixin

REPO_URL = "https://github.com/scientific-discovery/LLM-AutoSciLab"

#: NewtonBench's difficulty tiers; each hides a differently modified law.
DIFFICULTIES = ("easy", "medium", "hard")

_EXPERIMENT_RE = re.compile(r"<experiment>\s*(.*?)\s*</experiment>", re.S | re.I)
_PREDICT_RE = re.compile(r"<predict>\s*(.*?)\s*</predict>", re.S | re.I)
_LAW_RE = re.compile(r"<law>\s*(.*?)\s*</law>", re.S | re.I)


def _setting_from(body: str, names: Sequence[str]) -> dict[str, float] | None:
    """Read a parameter setting out of one turn, by name where possible.

    JSON first, because that is what the prompt asks for and it says which value
    is which. Falling back to positional numbers needs the parameter names
    struck out of the text first: NewtonBench's parameters are called ``mass1``
    and ``mass2``, and a naive scan reads the 1 and the 2 as measurements.
    """
    try:
        blob = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        blob = None
    if isinstance(blob, dict):
        lowered = {str(key).lower(): value for key, value in blob.items()}
        setting: dict[str, float] = {}
        for name in names:
            value = lowered.get(name.lower())
            if value is None:
                return None
            try:
                setting[name] = float(value)
            except (TypeError, ValueError):
                return None
        return setting

    stripped = body
    for name in sorted(names, key=len, reverse=True):
        stripped = re.sub(re.escape(name), " ", stripped, flags=re.I)
    values = _numbers(stripped)
    if len(values) < len(names):
        return None
    return dict(zip(names, values[: len(names)], strict=True))


def _numbers(text: str) -> list[float]:
    out: list[float] = []
    for token in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", text or ""):
        try:
            out.append(float(token))
        except ValueError:
            continue
    return out


class SciLabAdapter(InteractiveMixin, PooledDatasetAdapter):
    """One episode per (domain, difficulty, law version): experiment, then explain."""

    adapter_version = "1.0"

    system_prompt = (
        "You are running an automated laboratory. A hidden law governs the apparatus, and "
        "the only way to learn it is to set the inputs and read the measurement. The law is "
        "NOT necessarily the textbook one for this quantity -- infer it from what the "
        "apparatus actually returns. Vary one parameter at a time where you can: that is how "
        "an exponent becomes visible."
    )
    data_delivery_mode = "interactive"
    objective_metrics = True
    selection_cardinality = None
    # The dataset table marks SciLab "Generation & Selection": an ampersand is a
    # combined pipeline, so it stays one task rather than splitting in two.
    hypothesis_modes = ("generation",)
    table_hypothesis_mode = "Generation & Selection"
    primary_metric = "prediction_error"
    higher_is_better = False

    max_turns = 24

    # ------------------------------------------------------------------ #
    # data: the released oracles
    # ------------------------------------------------------------------ #

    def _registry(self):
        root = C.ensure_git_repo(
            REPO_URL, self.context.data_dir / "repo", offline=self.context.offline
        )
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        try:
            module = __import__("autoscilab.oracle.newtonbench", fromlist=["*"])
        except Exception as exc:  # noqa: BLE001 - a missing dependency is a skip
            raise SkippedDataset(
                f"cannot import SciLab's NewtonBench oracles: {exc}. They need numpy, scipy "
                "and the release's own vendored modules (the openai/dotenv imports are only "
                "for its LLM law judge, which this adapter does not use)."
            ) from exc
        return module.DOMAIN_REGISTRY, module.NewtonBenchOracle, Path(root)

    def load_items(self) -> list[dict[str, Any]]:
        registry, oracle_class, _root = self._registry()
        self._oracle_class = oracle_class
        wanted = self.context.option("domains") or list(registry)
        difficulties = self.context.option("difficulties") or list(DIFFICULTIES)
        items = [
            {"domain": domain, "difficulty": difficulty, "config": registry[domain]}
            for domain in wanted
            if domain in registry
            for difficulty in difficulties
        ]
        if not items:
            raise SkippedDataset("no NewtonBench domain matched options.domains")
        self.split_used = (
            f"{len({i['domain'] for i in items})} vendored NewtonBench domains x "
            f"{len(difficulties)} difficulty tiers = {len(items)} laboratories"
        )
        return items

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        config = item["config"]
        names = list(config.get("param_names") or [])
        if not names:
            return None
        return SampleSpec(
            sample_id=C.stable_id("scilab", item["domain"], item["difficulty"]),
            fields={
                "observation": (
                    f"Laboratory: {item['domain']} ({item['difficulty']} setting). "
                    f"Parameters you can set: {', '.join(names)}."
                ),
            },
            reference={"domain": item["domain"], "difficulty": item["difficulty"]},
            task_kind="generation",
            metadata={
                "domain": item["domain"],
                "difficulty": item["difficulty"],
                "n_params": len(names),
                "_spec": item,
            },
        )

    # ------------------------------------------------------------------ #
    # the episode
    # ------------------------------------------------------------------ #

    def _held_out(self, config: dict[str, Any], rng: random.Random, count: int):
        """Settings the model will be asked to predict, drawn inside the bounds."""
        bounds = config.get("bounds") or {}
        names = list(config.get("param_names") or [])
        settings = []
        for _ in range(count):
            setting = {}
            for name in names:
                low, high = bounds.get(name, (1.0, 10.0))
                # Log-uniform, as the release samples: these bounds span orders
                # of magnitude, and uniform draws would sit at the top of them.
                setting[name] = math.exp(rng.uniform(math.log(low), math.log(high)))
            settings.append(setting)
        return settings

    def interactive_start(self, sample: SampleSpec) -> tuple[list[ChatMessage], dict[str, Any]]:
        spec = sample.metadata["_spec"]
        config = spec["config"]
        names = list(config.get("param_names") or [])
        bounds = config.get("bounds") or {}
        rng = random.Random(f"{self.context.seed}::{sample.sample_id}")
        try:
            oracle = self._oracle_class(spec["domain"], difficulty=spec["difficulty"])
        except TypeError:
            oracle = self._oracle_class(spec["domain"])
        except Exception as exc:  # noqa: BLE001 - one dead lab is not the run
            self.log.warning("scilab: cannot build oracle for %s: %s", spec["domain"], exc)
            oracle = None

        budget = int(self.context.option("experiments", 12))
        questions = int(self.context.option("questions", 5))
        bound_lines = "\n".join(
            f"- {name}: between {bounds.get(name, (1.0, 10.0))[0]:g} and "
            f"{bounds.get(name, (1.0, 10.0))[1]:g}"
            for name in names
        )
        opening = (
            f"{sample.fields['observation']}\n\n"
            f"Allowed ranges:\n{bound_lines}\n\n"
            f"You may run {budget} experiments. Each turn, either run one:\n"
            "<experiment>{" + ", ".join(f'"{name}": <value>' for name in names) + "}</experiment>\n"
            "or stop early and go to the questions:\n<experiment>done</experiment>\n\n"
            f"Then you will be asked to predict {questions} measurements at settings you have "
            "not tried, and to state the law you inferred."
        )
        return (
            [
                ChatMessage(role="system", content=self.system_prompt),
                ChatMessage(role="user", content=opening),
            ],
            {
                "oracle": oracle,
                "names": names,
                "budget": budget,
                "used": 0,
                "phase": "experiment",
                "questions": self._held_out(config, rng, questions),
                "asked": 0,
                "predictions": [],
                "truths": [],
                "law": "",
                "experiments": [],
            },
        )

    def interactive_step(
        self, sample: SampleSpec, state: dict[str, Any], assistant_text: str
    ) -> str | None:
        if state["oracle"] is None:
            return None
        if state["phase"] == "experiment":
            return self._experiment_turn(state, assistant_text)
        return self._prediction_turn(state, assistant_text)

    def _experiment_turn(self, state: dict[str, Any], text: str) -> str | None:
        match = _EXPERIMENT_RE.search(text or "")
        if match is None:
            state["parse_errors"] = state.get("parse_errors", 0) + 1
            if state["parse_errors"] > 2:
                return self._begin_questions(state)
            return (
                "Reply with <experiment>{...}</experiment> containing every parameter, "
                "or <experiment>done</experiment>."
            )
        body = match.group(1).strip()
        if body.lower().startswith("done"):
            return self._begin_questions(state)

        names = state["names"]
        setting = _setting_from(body, names)
        if setting is None:
            return f"That setting is incomplete: give a value for each of {', '.join(names)}."
        state["used"] += 1
        try:
            measurement = state["oracle"].run(setting).measurement
        except Exception as exc:  # noqa: BLE001 - a rejected setting is an observation
            return f"The apparatus rejected those settings: {type(exc).__name__}: {exc}"
        state["experiments"].append({"setting": setting, "measurement": measurement})
        readable = ", ".join(f"{name}={value:g}" for name, value in setting.items())
        if state["used"] >= state["budget"]:
            return f"Measured {readable}: {measurement:g}\n\n{self._begin_questions(state)}"
        return (
            f"Measured {readable}: {measurement:g}\n"
            f"({state['budget'] - state['used']} experiment(s) left.)"
        )

    def _begin_questions(self, state: dict[str, Any]) -> str:
        state["phase"] = "predict"
        return "Experiments closed.\n\n" + self._next_question(state)

    def _next_question(self, state: dict[str, Any]) -> str:
        setting = state["questions"][state["asked"]]
        state["asked"] += 1
        readable = ", ".join(f"{name}={value:g}" for name, value in setting.items())
        tail = ""
        if state["asked"] == len(state["questions"]):
            tail = (
                "\nThis is the last one, so also state the law you inferred, as "
                "<law>your equation</law>."
            )
        return (
            f"Question {state['asked']} of {len(state['questions'])}: what will the apparatus "
            f"measure at {readable}?\nAnswer as <predict>value</predict>.{tail}"
        )

    def _prediction_turn(self, state: dict[str, Any], text: str) -> str | None:
        match = _PREDICT_RE.search(text or "")
        numbers = _numbers(match.group(1)) if match else _numbers(text or "")
        state["predictions"].append(numbers[0] if numbers else None)
        setting = state["questions"][state["asked"] - 1]
        try:
            state["truths"].append(state["oracle"].run(setting).measurement)
        except Exception:  # noqa: BLE001 - an unrunnable setting cannot be graded
            state["truths"].append(None)
        law = _LAW_RE.search(text or "")
        if law:
            state["law"] = law.group(1).strip()
        if state["asked"] >= len(state["questions"]):
            return None
        return self._next_question(state)

    # ------------------------------------------------------------------ #
    # scoring
    # ------------------------------------------------------------------ #

    def score(
        self,
        sample: SampleSpec,
        response: ModelResponse,
        *,
        output_contract: dict[str, Any] | None = None,
    ) -> SampleScore:
        state = sample.metadata.get("_episode_state") or {}
        pairs = [
            (prediction, truth)
            for prediction, truth in zip(
                state.get("predictions") or [], state.get("truths") or [], strict=False
            )
            if prediction is not None and truth is not None
        ]
        if not pairs:
            return SampleScore(
                # No prediction_error key: the episode was not measured.
                metrics={"answered_rate": 0.0, "law_stated": 0.0},
                parse_ok=False,
                details={"reason": "the episode produced no usable predictions"},
            )
        errors = [
            abs(prediction - truth) / max(abs(truth), 1e-12) for prediction, truth in pairs
        ]
        errors.sort()
        middle = len(errors) // 2
        median = errors[middle] if len(errors) % 2 else (errors[middle - 1] + errors[middle]) / 2
        return SampleScore(
            metrics={
                # Median relative error: a single wild answer should not decide
                # an episode, and the domains differ by orders of magnitude.
                "prediction_error": median,
                "within_10pct": mean([1.0 if error <= 0.1 else 0.0 for error in errors]),
                "experiments_used": float(state.get("used", 0)),
                "answered_rate": len(pairs) / max(1, len(state.get("questions") or [1])),
                "law_stated": 1.0 if state.get("law") else 0.0,
            },
            prediction=state.get("law") or [p for p, _ in pairs],
            details={
                "law": state.get("law", ""),
                "experiments": state.get("experiments", [])[:20],
            },
        )

    def aggregate(self, scores: Sequence[SampleScore]) -> dict[str, float]:
        # An episode that could not be measured omits the metric rather than
        # reporting NaN, so "scored" is simply the episodes that have one.
        metrics = aggregate_mean_metrics([score.metrics for score in scores])
        errors = [
            score.metrics["prediction_error"]
            for score in scores
            if "prediction_error" in score.metrics
        ]
        metrics["episodes_scored"] = float(len(errors))
        if errors:
            metrics["prediction_error"] = mean(errors)
        return metrics

    # ------------------------------------------------------------------ #
    # documentation
    # ------------------------------------------------------------------ #

    def documentation(self) -> AdapterDocumentation:
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="SciLab (LLM-AutoSciLab)",
            domain="Scientific Discovery: Active Mechanism Discovery",
            source_url=REPO_URL,
            processing_mode="Generation & Selection",
            split_used=self.split_used,
            abductive_subset=(
                "All twelve vendored NewtonBench domains. The hidden law is the hypothesis: "
                "it is never shown, the textbook law is deliberately not it, and the only "
                "evidence is measurements the model chose to take."
            ),
            sampling_procedure=(
                "No sampling from a pool: an item is a laboratory (domain x difficulty). "
                f"Held-out prediction settings are drawn log-uniformly inside the release's "
                f"own bounds, seeded from the run seed ({self.context.seed}) and the item id."
            ),
            metrics_description={
                "self_consistency_<metric>":
                "Every metric also gets a self_consistency_ counterpart: the plurality answer over "
                "modes.repeats samples of the same record, read off those samples rather than bought "
                "again. Available because this dataset's answers are checkable and so can coincide.",
                "prediction_error": "(PRIMARY, LOWER is better) LOWER IS BETTER (primary). Median relative error of the "
                "model's predictions at settings it never tried -- the consequence its "
                "inferred law is answerable to",
                "within_10pct": "fraction of predictions within 10% of the true measurement",
                "experiments_used": "experiments the episode spent before predicting",
                "answered_rate": "fraction of the held-out questions that got a number",
                "law_stated": "whether the model stated a law at all, recorded for inspection",
                "episodes_scored": "episodes that produced usable predictions",
            },
            primary_metric=self.primary_metric,
            decisions=[
                "Ran the release's own oracles rather than substituting static data: SciLab "
                "ships laboratories, not items.",
                "Scored by prediction on held-out settings rather than by comparing the stated "
                "law to a reference string: the release grades laws with an LLM judge, and a "
                "judge inside the evaluation would make the score depend on a second model.",
                "Kept all three difficulty tiers, because they are three different modified "
                "laws over the same apparatus and the gap between them is the point.",
                "Did not use the vendored modules' evaluate_law (its openai/dotenv imports are "
                "satisfied only so the simulator can be imported; the judge is never called).",
            ],
            caveats=[
                "Lower is better here: the primary metric is an error.",
                "Measurements carry the release's own noise, so a perfect law still leaves a "
                "small residual error.",
                "An episode is many turns (experiments plus questions), so this dataset costs "
                "more per item than a static one.",
            ],
            statistics=self.base_statistics(),
        )
