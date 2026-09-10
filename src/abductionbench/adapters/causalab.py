"""CausaLab: discover a causal structure inside the release's own game world.

Source: https://github.com/allenai/causalab (vendored DiscoveryWorld included)

**The task, as its authors pose it.**  The agent stands in the Causal Discovery
Lab on Planet X.  Quantum crystals have properties -- temperature, moisture,
and others depending on the graph -- wired to each other by a hidden linear
DAG, and a resonance frequency that no one can set directly.  The agent works a
*Property Manipulator* through dialog menus to adjust one property at a time,
watches what else moves, infers the structure, and finally sets the *Crystal
Reactor* to the frequency the target crystal must have.  A limited number of
adjustments is the experiment budget, and using the reactor ends the
experimentation phase for good.

**This adapter drives that world, not a substitute for it.**  An earlier
version simulated the release's ``causal_graph_configs`` as a linear structural
causal model with ``intervene``/``observe``/``answer`` actions.  That was a
different task under a prompt written here, and its numbers were not CausaLab's.
What runs now is ``DiscoveryWorldAPI`` itself, vendored in the repository:

* the prompt is the environment's own ``taskDescription``, which the release
  renders from ``agents/recoma/prompts/reactor_task_causal_*.txt`` -- it is
  never written here, only read;
* the action space is DiscoveryWorld's (``TELEPORT_TO_OBJECT``, ``TALK``,
  ``READ``, ``ACTIVATE``, dialog selections and value entry), passed through
  verbatim as the JSON the API already accepts;
* the score is the release's own scorecard, ``score / maxScore``, plus whether
  it recorded a successful completion.

**Headless.**  DiscoveryWorld draws with pygame; ``SDL_VIDEODRIVER=dummy`` (set
below before the import) runs it with no display, which is how it runs here.
``pygame``, ``pathfinding`` and ``termcolor`` are its runtime dependencies.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from ..core.adapter import SkippedDataset
from ..core.metrics import aggregate_mean_metrics
from ..core.types import (
    AdapterDocumentation,
    ChatMessage,
    ModelResponse,
    SampleScore,
    SampleSpec,
)
from . import _common as C
from ._base import PooledDatasetAdapter

#: The release's own scenario entry, from discoveryworld/ScenarioMaker.py.
SCENARIO_NAME = "Reactor Lab Causal"
DIFFICULTY = "Causal"

#: How the model is told to act.  This is *not* a task prompt -- the task
#: prompt is the environment's, read from its scorecard.  This only states the
#: JSON envelope, because the harness has to receive one action per turn and
#: DiscoveryWorld's own agent gets the same envelope from its ReAct scaffold.
_ACTION_ENVELOPE = (
    "Reply with exactly one JSON object and nothing else. It is passed straight "
    "to the environment.\n"
    'Examples: {"action": "TELEPORT_TO_OBJECT", "arg1": <uuid>}  '
    '{"action": "TALK", "arg1": <uuid>}  '
    '{"action": "READ", "arg1": <uuid>}  '
    '{"action": "ACTIVATE", "arg1": <uuid>}\n'
    'In a dialog, choose an option with {"chosen_dialog_option_int": <n>}, and in '
    'value-entry mode give {"value": <number>}.'
)


def _ensure_headless() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


class CausaLabAdapter(PooledDatasetAdapter):
    """One episode per causal-graph configuration, in the real environment."""

    adapter_version = "2.0"

    #: Empty by design: CausaLab's task prompt is the environment's own
    #: taskDescription, which the release renders from its prompt templates.
    system_prompt = ""
    data_delivery_mode = "interactive"
    objective_metrics = True
    selection_cardinality = None
    hypothesis_modes = ("generation",)
    hypothesis_mode_options = {"generation": {"subtask": "generation"}}
    table_hypothesis_mode = "Generation"
    primary_metric = "score_normalized"
    higher_is_better = True
    #: The release's agent runs a long episode; the budget is the number of
    #: model turns, not of property adjustments (the environment enforces that
    #: one itself).
    max_turns = 40

    # ------------------------------------------------------------------ #
    # setup
    # ------------------------------------------------------------------ #

    @property
    def _repo(self) -> Path:
        configured = self.context.option("repo_dir", None)
        return Path(str(configured)) if configured else self.context.data_dir / "repo"

    def prepare(self) -> None:
        _ensure_headless()
        repo = self._repo
        if not (repo / "discoveryworld").is_dir():
            raise SkippedDataset(
                f"CausaLab's vendored DiscoveryWorld is not in the clone at {repo}. "
                f"SETUP: git clone https://github.com/allenai/causalab {repo}"
            )
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        try:
            from discoveryworld.DiscoveryWorldAPI import DiscoveryWorldAPI  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            raise SkippedDataset(
                f"CausaLab's environment could not be imported ({type(exc).__name__}: "
                f"{exc}). It is pygame-based and runs headless here. SETUP: "
                f"pip install pygame pathfinding termcolor"
            ) from exc
        super().prepare()

    # ------------------------------------------------------------------ #
    # items: one per causal-graph configuration the release ships
    # ------------------------------------------------------------------ #

    def load_items(self) -> list[dict[str, Any]]:
        configs = self._repo / "causal_graph_configs"
        if not configs.is_dir():
            raise SkippedDataset(f"{configs} does not exist in the clone")
        wanted = self.context.option("configs", None)
        files = sorted(p for p in configs.glob("*.json") if p.is_file())
        if wanted:
            names = {str(w) for w in wanted}
            files = [p for p in files if p.name in names or p.stem in names]
        if not files:
            raise SkippedDataset(f"no causal graph configs found under {configs}")
        seeds = [int(s) for s in (self.context.option("seeds", None) or [1])]
        items = [
            {"config": str(path), "name": path.stem, "seed": seed}
            for path in files
            for seed in seeds
        ]
        self.split_used = (
            f"{len(files)} causal-graph configuration(s) x {len(seeds)} seed(s) "
            f"in {SCENARIO_NAME}"
        )
        return items

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        return SampleSpec(
            sample_id=C.stable_id("causalab", item["name"], item["seed"]),
            fields={"observation": f"Causal graph configuration: {item['name']}"},
            reference={"config": item["name"]},
            task_kind="generation",
            metadata={
                "graph": item["name"],
                "seed": item["seed"],
                "_config_path": item["config"],
            },
        )

    # ------------------------------------------------------------------ #
    # the episode: DiscoveryWorld, driven turn by turn
    # ------------------------------------------------------------------ #

    def _open_world(self, sample: SampleSpec):
        """Load the scenario for one episode.

        The causal structure is read from an environment variable by the
        release's own ``mkReactorLabCausal``; that is how it is configured, so
        it is how it is configured here.
        """
        _ensure_headless()
        if str(self._repo) not in sys.path:
            sys.path.insert(0, str(self._repo))
        from discoveryworld.DiscoveryWorldAPI import DiscoveryWorldAPI

        os.environ["CAUSAL_GRAPH_CONFIG"] = sample.metadata["_config_path"]
        seed = int(sample.metadata["seed"])
        # A distinct threadID per episode: DiscoveryWorld keys its scratch
        # directories on it, and two episodes sharing one would overwrite each
        # other's frames.
        thread = 5000 + (abs(hash(sample.sample_id)) % 900) * 10 + seed
        api = DiscoveryWorldAPI(threadID=thread)
        if not api.loadScenario(
            scenarioName=SCENARIO_NAME,
            difficultyStr=DIFFICULTY,
            randomSeed=seed,
            numUserAgents=1,
        ):
            raise RuntimeError(f"loadScenario failed for {sample.metadata['graph']}")
        return api

    def _render(self, api) -> str:
        """What the agent can see, in the environment's own terms."""
        obs = api.getAgentObservation(agentIdx=0)
        ui = obs.get("ui", {}) or {}
        dialog = ui.get("dialog_box") or {}
        lines: list[str] = []
        if dialog.get("is_in_dialog"):
            lines.append("You are in a dialog.")
            if dialog.get("dialogText"):
                lines.append(str(dialog["dialogText"]))
            options = dialog.get("dialogOptions") or {}
            if options:
                lines.append(
                    "Options: "
                    + "; ".join(f"{k}. {v}" for k, v in options.items())
                )
        accessible = ui.get("accessibleEnvironmentObjects") or []
        if accessible:
            seen: dict[Any, str] = {}
            for o in accessible:
                seen.setdefault(o.get("uuid"), str(o.get("name")))
            lines.append(
                "Accessible objects: "
                + ", ".join(f"{name} (uuid {uuid})" for uuid, name in seen.items())
            )
        nearby = ui.get("nearbyObjects") or {}
        near_list = nearby.get("objects") if isinstance(nearby, dict) else nearby
        if isinstance(near_list, list) and near_list:
            names = []
            for o in near_list[:12]:
                if isinstance(o, dict):
                    names.append(f"{o.get('name')} (uuid {o.get('uuid')})")
            if names:
                lines.append("Nearby: " + ", ".join(names))
        inventory = ui.get("inventoryObjects") or []
        if inventory:
            lines.append(
                "Inventory: "
                + ", ".join(f"{o.get('name')} (uuid {o.get('uuid')})" for o in inventory)
            )
        if ui.get("lastActionMessage"):
            lines.append(f"Last action: {ui['lastActionMessage']}")
        return "\n".join(lines) if lines else "(nothing visible)"

    def interactive_start(self, sample: SampleSpec) -> tuple[list[ChatMessage], dict[str, Any]]:
        api = self._open_world(sample)
        card = (api.getTaskScorecard() or [{}])[0]
        # The environment's own task description: the release renders it from
        # agents/recoma/prompts/reactor_task_causal_*.txt. Read, never written.
        briefing = str(card.get("taskDescription") or "").strip()
        if not briefing:
            raise RuntimeError("the scenario produced no task description")
        state = {"api": api, "turns": 0, "final": None}
        return (
            [
                ChatMessage(role="system", content=briefing),
                ChatMessage(
                    role="user",
                    content=f"{self._render(api)}\n\n{_ACTION_ENVELOPE}",
                ),
            ],
            state,
        )

    def interactive_step(
        self, sample: SampleSpec, state: dict[str, Any], assistant_text: str
    ) -> str | None:
        api = state.get("api")
        if api is None:
            return None
        action = _parse_action_json(assistant_text)
        if action is None:
            state["parse_errors"] = state.get("parse_errors", 0) + 1
            if state["parse_errors"] > 3:
                return None
            return (
                "That was not a single JSON object I could pass to the environment.\n"
                + _ACTION_ENVELOPE
            )
        state["parse_errors"] = 0
        state["turns"] += 1
        try:
            result = api.performAgentAction(agentIdx=0, actionJSON=action)
            api.tick()
        except Exception as exc:  # noqa: BLE001 - a refused action is an observation
            return f"The environment refused that action: {type(exc).__name__}: {exc}"

        errors = (result or {}).get("errors") or []
        head = "; ".join(str(e) for e in errors) if errors else "OK"
        if api.areTasksComplete():
            state["final"] = api.getTaskScorecard()
            return None
        return f"{head}\n\n{self._render(api)}"

    # ------------------------------------------------------------------ #
    # scoring: the release's own scorecard
    # ------------------------------------------------------------------ #

    def score(
        self,
        sample: SampleSpec,
        response: ModelResponse,
        *,
        output_contract: dict[str, Any] | None = None,
    ) -> SampleScore:
        state = sample.metadata.get("_episode_state") or {}
        api = state.get("api")
        card_list = state.get("final")
        if card_list is None and api is not None:
            try:
                card_list = api.getTaskScorecard()
            except Exception:  # noqa: BLE001
                card_list = None
        card = (card_list or [{}])[0]
        raw = card.get("score")
        maximum = card.get("maxScore") or 0
        if raw is None or not maximum:
            return SampleScore(
                metrics={
                    "score_normalized": 0.0,
                    "task_completed": 0.0,
                    "turns_taken": float(state.get("turns", 0)),
                },
                prediction=None,
                parse_ok=False,
                details={"reason": "the episode produced no scorecard"},
            )
        normalized = card.get("scoreNormalized")
        if not isinstance(normalized, (int, float)):
            normalized = float(raw) / float(maximum)
        return SampleScore(
            metrics={
                "score_normalized": max(0.0, min(1.0, float(normalized))),
                "score_raw": float(raw),
                "task_completed": 1.0 if card.get("completedSuccessfully") else 0.0,
                "turns_taken": float(state.get("turns", 0)),
            },
            prediction=f"score {raw}/{maximum}",
            details={"graph": sample.metadata.get("graph")},
        )

    def aggregate(self, scores) -> dict[str, float]:
        return aggregate_mean_metrics([score.metrics for score in scores])

    # ------------------------------------------------------------------ #
    # documentation
    # ------------------------------------------------------------------ #

    def documentation(self) -> AdapterDocumentation:
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="CausaLab",
            domain="Causal Science: Causal Discovery",
            source_url="https://github.com/allenai/causalab",
            processing_mode="Generation (interactive)",
            split_used=getattr(self, "split_used", "causal graph configurations"),
            abductive_subset=(
                "The whole benchmark. A hidden linear DAG relates the crystals' properties, "
                "and the agent must infer it from interventions it chooses before setting the "
                "reactor to the frequency the structure implies."
            ),
            sampling_procedure=(
                "One episode per causal-graph configuration the release ships, at the seeds "
                "configured. The configurations are the benchmark's own."
            ),
            metrics_description={
                "score_normalized": (
                    "(PRIMARY, higher is better, 0-1) the release's own scorecard score over "
                    "its maximum. DiscoveryWorld computes it; nothing here judges the episode."
                ),
                "score_raw": "the same scorecard's unnormalised points",
                "task_completed": (
                    "(higher is better, 0-1) fraction of episodes the environment recorded as "
                    "completed successfully"
                ),
                "turns_taken": "mean model turns spent in the world before the episode ended",
                "self_consistency_<metric>":
                "Every metric also gets a self_consistency_ counterpart: the plurality answer "
                "over modes.repeats samples of the same record, read off those samples rather "
                "than bought again. Available because this dataset's answers are checkable and "
                "so can coincide.",
            },
            primary_metric="score_normalized",
            decisions=[
                "Ran the release's vendored DiscoveryWorld rather than simulating its causal "
                "graphs: the benchmark is the game world, and an SCM stand-in measures a "
                "different, easier task.",
                "Used the environment's own taskDescription as the prompt. The release renders "
                "it from agents/recoma/prompts/reactor_task_causal_*.txt, so it is read from "
                "the running scenario rather than restated here.",
                "Passed DiscoveryWorld's action JSON through unchanged, so the action space is "
                "the environment's (TELEPORT_TO_OBJECT, TALK, dialog selection, value entry).",
                "Scored with the release's own scorecard (score / maxScore), not a metric "
                "defined here.",
            ],
            caveats=[
                "Runs headless via SDL_VIDEODRIVER=dummy and needs pygame, pathfinding and "
                "termcolor. It draws no window, but it is still a full game world per episode, "
                "so episodes are slower than a text-only benchmark.",
                "The agent must navigate before it can act: the Property Manipulator is only "
                "reachable after a TELEPORT_TO_OBJECT, exactly as in the release. A model that "
                "never navigates scores zero, which is a real outcome of this benchmark.",
                "The causal structure is passed to the scenario through the CAUSAL_GRAPH_CONFIG "
                "environment variable, which is how the release configures it. Episodes "
                "therefore set it per episode and cannot be interleaved within one process.",
            ],
            statistics={**self.base_statistics()},
        )


def _parse_action_json(text: str | None) -> dict[str, Any] | None:
    """The last JSON object in the response, which is the action."""
    if not text:
        return None
    for match in reversed(list(re.finditer(r"\{.*?\}", text, flags=re.S))):
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and (
            "action" in parsed or "chosen_dialog_option_int" in parsed or "value" in parsed
        ):
            return parsed
    return None
