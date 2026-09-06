"""BoxingGym: design experiments, then predict what the system will do.

Source: https://github.com/kanishkg/boxing-gym

BoxingGym has no item file, and cannot have one: an item *is* an environment.
The release ships those environments as runnable Python -- a generative model
(``DeathProcess``, ``LotkaVolterra``, ``IRT``, ``Peregrines``, ...) paired with a
``Goal`` that states what the agent is supposed to end up able to predict.  So
the adapter runs them: it imports the released modules, instantiates one
(environment, goal) pair per item with its own seed, and drives the benchmark's
own loop.

**The loop, as the benchmark defines it.**  An episode has two phases:

1. *Experimentation*.  The model observes at inputs of its choosing, in the
   release's own ``<observe>x</observe>`` syntax, and the environment returns the
   true measurement.  The budget is ``options.experiments``.
2. *Prediction*.  The environment then asks the goal's own evaluation questions
   (``Goal.get_goal_eval_question``) and collects the answers.

Scoring is the release's ``Goal.evaluate_predictions``, which returns the error
of those predictions.  Lower is better, so this is the one dataset in the suite
whose primary metric is an error: ``standardized_error`` divides by the goal's
own published normalisation constants (``norm_mu``/``norm_sigma``), which is how
BoxingGym makes errors comparable across environments.

**Why this is abduction.**  The hypothesis is the model of the system: nothing
in the observations states it, and the only way to be right about the
predictions is to have inferred a mechanism that would produce what was seen.
The benchmark grades that inference by its consequences, which is what makes it
scorable at all.

**What is excluded.**  ``emotion`` and ``moral_machines`` build their
environment out of an LLM (they import ``openai``), which would put a second
model inside the evaluation of the first and make the score depend on it; they
are skipped and reported.  ``options.environments`` overrides the selection.
"""

from __future__ import annotations

import importlib
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

REPO_URL = "https://github.com/kanishkg/boxing-gym"

#: Environment modules that need no LLM of their own.  ``emotion`` and
#: ``moral_machines`` are deliberately absent: see the module docstring.
USABLE_MODULES = (
    "death_process",
    "dugongs",
    "hyperbolic_temporal_discount",
    "irt",
    "location_finding",
    "lotka_volterra",
    "peregrines",
    "survival_analysis",
)

#: Goals whose question is "predict the system's output", which is the family
#: the benchmark's headline numbers use.  ``*Naive`` variants are the same goal
#: without the domain prior and are kept: the pair measures how much the prior
#: is worth, which is one of the paper's own questions.
_OBSERVE_RE = re.compile(r"<observe>\s*(.*?)\s*</observe>", re.S | re.I)
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S | re.I)


class BoxingGymAdapter(InteractiveMixin, PooledDatasetAdapter):
    """One episode per (environment, goal, seed): experiment, then predict."""

    adapter_version = "1.0"

    system_prompt = (
        "You are a scientist studying a system you cannot see inside. You learn about it "
        "only by running experiments, and you are judged on whether you can then predict "
        "what it will do. Choose experiments that would distinguish the mechanisms you are "
        "considering, not experiments that confirm the one you already prefer."
    )
    data_delivery_mode = "interactive"
    # Scored by the release's own evaluate_predictions: numeric error against
    # measurements the environment produced. No judge is involved.
    objective_metrics = True
    selection_cardinality = None
    primary_metric = "standardized_error"
    higher_is_better = False

    max_turns = 25

    # ------------------------------------------------------------------ #
    # data: the released environments themselves
    # ------------------------------------------------------------------ #

    @staticmethod
    def _patch_pymc_compat() -> None:
        """Let the release's environments run on a current PyMC.

        BoxingGym calls ``pm.sample_prior_predictive(samples=1)``; PyMC renamed
        that argument to ``draws`` in 6.x, so on a current install 21 of the 66
        episodes raise ``TypeError`` the moment their goal is constructed.  The
        environments are otherwise unchanged and correct, so the fix is to
        accept the old spelling rather than to pin the whole project to a PyMC
        from before the rename.
        """
        try:
            import pymc
        except ImportError:  # pragma: no cover - reported by load_items instead
            return
        if getattr(pymc.sample_prior_predictive, "_abench_samples_shim", False):
            return
        original = pymc.sample_prior_predictive
        try:
            import inspect

            if "samples" in inspect.signature(original).parameters:
                return  # an older PyMC: nothing to translate
        except (TypeError, ValueError):  # pragma: no cover - unsignatured callable
            return

        def sample_prior_predictive(*args, **kwargs):
            if "samples" in kwargs:
                kwargs["draws"] = kwargs.pop("samples")
            return original(*args, **kwargs)

        sample_prior_predictive._abench_samples_shim = True
        pymc.sample_prior_predictive = sample_prior_predictive

    def _source_root(self) -> Path:
        root = C.ensure_git_repo(
            REPO_URL, self.context.data_dir / "repo", offline=self.context.offline
        )
        source = root / "src"
        if not (source / "boxing_gym" / "envs").exists():
            raise SkippedDataset(f"BoxingGym environments not found under {source}")
        if str(source) not in sys.path:
            # The release is a package that is not installed; importing it from
            # its own checkout is how its own scripts run it too.
            sys.path.insert(0, str(source))
        return source

    def load_items(self) -> list[dict[str, Any]]:
        self._source_root()
        self._patch_pymc_compat()
        try:
            goal_module = importlib.import_module("boxing_gym.envs.goal")
        except Exception as exc:  # noqa: BLE001 - a missing dependency is a skip, not a crash
            raise SkippedDataset(
                f"cannot import BoxingGym's environment package: {exc}. "
                "Its environments need numpy, scipy, pymc and matplotlib."
            ) from exc
        goal_base = goal_module.Goal

        wanted = self.context.option("environments") or USABLE_MODULES
        episodes_per_goal = int(self.context.option("episodes_per_goal", 3))
        experiments = int(self.context.option("experiments", 10))

        items: list[dict[str, Any]] = []
        unusable: list[str] = []
        for name in wanted:
            try:
                module = importlib.import_module(f"boxing_gym.envs.{name}")
            except Exception as exc:  # noqa: BLE001 - report and carry on
                unusable.append(f"{name} ({type(exc).__name__}: {exc})")
                continue
            env_classes, goal_classes = [], []
            for attribute, obj in vars(module).items():
                if not isinstance(obj, type) or obj.__module__ != module.__name__:
                    continue
                if issubclass(obj, goal_base):
                    goal_classes.append((attribute, obj))
                else:
                    env_classes.append((attribute, obj))
            if not env_classes or not goal_classes:
                unusable.append(f"{name} (no environment/goal pair)")
                continue
            env_name, env_class = env_classes[0]
            for goal_name, goal_class in goal_classes:
                problem = self._briefing_problem(env_class, goal_class, goal_name)
                if problem:
                    # A goal whose briefing the release cannot produce would have
                    # to be run with a prompt written here, which would measure a
                    # different task than the one the benchmark defines.
                    unusable.append(f"{name}.{goal_name} ({problem})")
                    continue
                for episode in range(episodes_per_goal):
                    items.append(
                        {
                            "module": name,
                            "env_name": env_name,
                            "env_class": env_class,
                            "goal_name": goal_name,
                            "goal_class": goal_class,
                            "episode": episode,
                            "experiments": experiments,
                        }
                    )
        if not items:
            raise SkippedDataset(
                "no usable BoxingGym environment could be imported: " + "; ".join(unusable)
            )
        self._unusable = unusable
        self.split_used = (
            f"{len({i['module'] for i in items})} released environment modules x their goals "
            f"x {episodes_per_goal} seeded episodes = {len(items)} episodes"
            + (f"; excluded: {', '.join(unusable)}" if unusable else "")
        )
        return items

    @staticmethod
    def _briefing_problem(env_class, goal_class, goal_name: str) -> str | None:
        """``None`` if this goal can state its own task, else why it cannot.

        Three of IRT's five goals ask their environment for a system message it
        does not implement (``'IRT' object has no attribute
        'get_system_message'``). That is a gap in the release, not something to
        paper over: the briefing is what tells the model what it is predicting.
        """
        try:
            env = env_class()
            goal = goal_class(env)
            message = goal.get_system_message(not goal_name.endswith("Naive"))
        except Exception as exc:  # noqa: BLE001 - the reason is what we want
            return f"{type(exc).__name__}: {exc}"
        if not str(message or "").strip():
            return "the goal's system message is empty"
        return None

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        sample_id = C.stable_id(
            "boxgym", item["module"], item["goal_name"], item["episode"]
        )
        return SampleSpec(
            sample_id=sample_id,
            fields={
                "observation": (
                    f"{item['env_name']} ({item['module']}), goal {item['goal_name']}"
                ),
            },
            reference={"module": item["module"], "goal": item["goal_name"]},
            task_kind="generation",
            metadata={
                "module": item["module"],
                "environment": item["env_name"],
                "goal": item["goal_name"],
                "episode": item["episode"],
                "_spec": item,
            },
        )

    # ------------------------------------------------------------------ #
    # the episode: experiment, then predict
    # ------------------------------------------------------------------ #

    def _instantiate(self, spec: dict[str, Any], seed: int):
        """Build one (environment, goal) pair, seeded so an episode repeats."""
        random.seed(seed)
        try:
            import numpy as np

            np.random.seed(seed)
        except Exception:  # noqa: BLE001 - seeding is best-effort
            pass
        env = spec["env_class"]()
        goal = spec["goal_class"](env)
        if hasattr(env, "reset"):
            env.reset()
        return env, goal

    def interactive_start(self, sample: SampleSpec) -> tuple[list[ChatMessage], dict[str, Any]]:
        spec = sample.metadata["_spec"]
        seed = self.context.seed + int(spec["episode"])
        env, goal = self._instantiate(spec, seed)
        include_prior = not spec["goal_name"].endswith("Naive")
        # Not caught: load_items already excluded every goal that cannot brief
        # itself, so a failure here is a real breakage and the engine records the
        # episode as an error rather than scoring a substitute prompt.
        briefing = goal.get_system_message(include_prior)
        state = {
            "env": env,
            "goal": goal,
            "include_prior": include_prior,
            "budget": int(spec["experiments"]),
            "used": 0,
            "phase": "experiment",
            "predictions": [],
            "truths": [],
            "asked": 0,
            "n_questions": int(self.context.option("questions", 5)),
        }
        return (
            [
                # The briefing is the benchmark's own, including its
                # <observe> syntax; only the phase instruction is added.
                ChatMessage(role="system", content=self.system_prompt),
                ChatMessage(
                    role="user",
                    content=(
                        f"{briefing}\n\n"
                        f"You may run {state['budget']} experiments. After that you will be "
                        "asked to make predictions, so spend them on learning the system."
                    ),
                ),
            ],
            state,
        )

    def interactive_step(
        self, sample: SampleSpec, state: dict[str, Any], assistant_text: str
    ) -> str | None:
        if state["phase"] == "experiment":
            return self._experiment_turn(state, assistant_text)
        return self._prediction_turn(state, assistant_text)

    def _experiment_turn(self, state: dict[str, Any], text: str) -> str | None:
        match = _OBSERVE_RE.search(text or "")
        if match is None:
            state["parse_errors"] = state.get("parse_errors", 0) + 1
            if state["parse_errors"] > 2:
                return self._begin_predictions(state)
            return "Reply with your observation in the required form: <observe>value</observe>"

        state["used"] += 1
        try:
            result, ok = state["env"].run_experiment(match.group(1))
        except Exception as exc:  # noqa: BLE001 - a rejected setting is an observation
            result, ok = f"Error: {type(exc).__name__}: {exc}", False
        reply = f"Result: {result}" if ok else str(result)
        if state["used"] >= state["budget"]:
            return f"{reply}\n\n{self._begin_predictions(state)}"
        remaining = state["budget"] - state["used"]
        return f"{reply}\n({remaining} experiment(s) left.)"

    def _begin_predictions(self, state: dict[str, Any]) -> str:
        """Close the experiment phase and ask the goal's own first question."""
        state["phase"] = "predict"
        return "No more experiments.\n\n" + self._next_question(state)

    def _next_question(self, state: dict[str, Any]) -> str:
        goal = state["goal"]
        question, truth = goal.get_goal_eval_question(state["include_prior"])
        state["truths"].append(truth)
        state["asked"] += 1
        return (
            f"Question {state['asked']} of {state['n_questions']}: {question}\n"
            "Answer inside <answer></answer> tags."
        )

    def _prediction_turn(self, state: dict[str, Any], text: str) -> str | None:
        match = _ANSWER_RE.search(text or "")
        answer = match.group(1) if match else (text or "").strip().split("\n")[-1]
        state["predictions"].append(answer)
        if state["asked"] >= state["n_questions"]:
            return None
        return self._next_question(state)

    # ------------------------------------------------------------------ #
    # scoring: the release's own evaluate_predictions
    # ------------------------------------------------------------------ #

    def score(
        self,
        sample: SampleSpec,
        response: ModelResponse,
        *,
        output_contract: dict[str, Any] | None = None,
    ) -> SampleScore:
        state = sample.metadata.get("_episode_state") or {}
        predictions = state.get("predictions") or []
        truths = state.get("truths") or []
        pairs = min(len(predictions), len(truths))
        if not pairs:
            # No standardized_error key at all: the episode was not measured,
            # and an absent metric is what the aggregator reads as "skip me".
            return SampleScore(
                metrics={"answered_rate": 0.0},
                parse_ok=False,
                details={"reason": "the episode produced no predictions"},
            )
        goal = state.get("goal")
        try:
            error, _spread = goal.evaluate_predictions(predictions[:pairs], truths[:pairs])
        except Exception as exc:  # noqa: BLE001 - an unparseable answer is a wrong one
            return SampleScore(
                metrics={"answered_rate": pairs / max(1, len(truths))},
                parse_ok=False,
                details={"reason": f"predictions could not be scored: {exc}"},
            )
        # The goal publishes the constants that make errors comparable across
        # environments; without them a hard environment would dominate the mean.
        norm_mu = float(getattr(goal, "norm_mu", 0.0) or 0.0)
        norm_sigma = float(getattr(goal, "norm_sigma", 0.0) or 0.0)
        standardized = (float(error) - norm_mu) / norm_sigma if norm_sigma else float(error)
        return SampleScore(
            metrics={
                "standardized_error": standardized,
                "raw_error": float(error),
                "experiments_used": float(state.get("used", 0)),
                "answered_rate": pairs / max(1, len(truths)),
            },
            prediction=predictions[:pairs],
            details={"truths": [str(t) for t in truths[:pairs]]},
        )

    def aggregate(self, scores: Sequence[SampleScore]) -> dict[str, float]:
        # An episode that could not be measured omits the metric rather than
        # reporting NaN, so "scored" is simply the episodes that have one.
        metrics = aggregate_mean_metrics([score.metrics for score in scores])
        errors = [
            score.metrics["standardized_error"]
            for score in scores
            if "standardized_error" in score.metrics
        ]
        metrics["episodes_scored"] = float(len(errors))
        if errors:
            metrics["standardized_error"] = mean(errors)
        return metrics

    # ------------------------------------------------------------------ #
    # documentation
    # ------------------------------------------------------------------ #

    def documentation(self) -> AdapterDocumentation:
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="BoxingGym",
            domain="Scientific Discovery: Experimental Design and Model Discovery",
            source_url=REPO_URL,
            processing_mode="Generation",
            split_used=self.split_used,
            abductive_subset=(
                "Every released environment whose simulator is self-contained. The hypothesis "
                "is the model of the system: it is never stated in the observations, and the "
                "predictions can only be right if it was inferred correctly."
            ),
            sampling_procedure=(
                "No sampling from a pool: each item is an environment instance, seeded from "
                f"the run seed ({self.context.seed}) plus its episode index, so an episode is "
                "reproducible. options.episodes_per_goal sets how many instances per goal."
            ),
            metrics_description={
                "standardized_error": "LOWER IS BETTER (primary). The release's own "
                "evaluate_predictions error, standardized by the goal's published norm_mu and "
                "norm_sigma so environments of different scales are comparable",
                "raw_error": "the same error before standardization",
                "experiments_used": "how many experiments the episode spent before predicting",
                "answered_rate": "fraction of the goal's evaluation questions that got an answer",
                "episodes_scored": "episodes that produced scorable predictions",
            },
            primary_metric=self.primary_metric,
            decisions=[
                "Ran the benchmark's real environments rather than substituting static data: "
                "BoxingGym ships its simulators as code and has no item file, so a static form "
                "would have had to invent one.",
                "Used the release's own system message per goal (Goal.get_system_message), its "
                "<observe>/<answer> syntax, and its evaluate_predictions for scoring.",
                "Kept the *Naive goal variants alongside their prior-carrying counterparts: the "
                "pair measures what the domain prior is worth, which is one of the paper's own "
                "questions.",
                "Seeded each episode from the run seed plus the episode index, so a re-run "
                "reproduces the same environment instances.",
            ],
            caveats=[
                "The emotion and moral_machines environments are excluded: they build the "
                "environment out of an LLM (they import openai), which would put a second model "
                "inside the evaluation of the first.",
                "Three of IRT's goals (BestStudent, DifficultQuestion, DiscriminatingQuestion) "
                "are excluded because the released IRT environment implements no "
                "get_system_message, so those goals cannot state their own task. They are "
                "listed in statistics.unusable_modules rather than run with a substitute "
                "prompt written here.",
                "Lower is better here, unlike every other dataset in the suite. The Summary "
                "sheet shows the value as reported; read it as an error.",
                "An episode's cost is its experiment budget plus its questions, so this dataset "
                "is many turns per item even though it is few items.",
            ],
            statistics={
                **self.base_statistics(),
                "unusable_modules": getattr(self, "_unusable", []),
            },
        )
