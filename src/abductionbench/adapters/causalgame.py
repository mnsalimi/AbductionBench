"""CausalGame: infer a hidden causal structure by experiment, then commit once.

Source: https://github.com/viewsetting/CausalGame

**The task.** A drone fleet is destroyed in a canyon and the model controls the
armour (DEF) of seven components.  The rules are hidden: HP is fixed and
invisible, the environment varies, and the scenario name says what the trap is
-- ``antenna_trap`` makes an irrelevant component *look* decisive through
selection bias, ``*_simpsons_paradox`` reverses the sign of an effect when the
data is pooled, ``*_local_optima`` rewards a design that is good but not
winning.  The model runs experiments, infers which components actually matter,
and then submits one design that is scored on a 1,000-drone fleet against a
win threshold.  That is abduction with a checkable answer: the simulator, not a
judge, says whether the inferred structure was right.

**How it is run.**  Interactive, against the project's *own* backend:

    uvicorn api.app:app --port 8000

started from a clone of the repository.  The environment is not a static item
file; it is that FastAPI service, and the adapter drives its published API
(``/api/v2/mission_status``, ``/api/v2/action_space``, ``/api/v2/deploy_drone``,
``/api/v2/evaluate_final_design``, ``/api/admin/experiment/switch``).  Each
episode is one scenario: up to ``stage1_deployment_budget`` deployments out of a
``total_drone_budget`` of drones, then exactly one submission.

The prompt is the release's own -- ``experiments/<scenario>/prompt.md``, rendered
with the mission's numbers -- because an interactive benchmark's prompt is part
of the benchmark.  What is *not* the release's is the action syntax: the project
drives its API through generated Python in a sandbox, which would put a code
interpreter inside the evaluation, so this adapter asks for one JSON action per
turn and calls the API itself.  The actions are the API's, one for one.

**Setup.**  Two things must exist, and if either is missing the dataset is
skipped with the exact command to fix it rather than being called unavailable:

1. a clone of the repository, at ``options.repo_dir`` (default
   ``data/causalgame/CausalGame``), for the scenario prompts;
2. the backend reachable at ``options.base_url``.
"""

from __future__ import annotations

import json
import os
import re
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

REPO_URL = "https://github.com/viewsetting/CausalGame.git"

#: One episode per scenario.  The variants are the point of the benchmark --
#: each hides the causal structure a different way -- so all of them are run.
SCENARIOS = (
    "antenna_trap",
    "antenna_trap_high_def",
    "antenna_trap_local_optima",
    "antenna_trap_no_history",
    "antenna_trap_no_selection_bias",
    "antenna_trap_simpsons_paradox",
    "deployment_zone_trap_categorical",
    "deployment_zone_trap_categorical_high_def",
    "deployment_zone_trap_categorical_local_optima",
    "deployment_zone_trap_categorical_no_history",
    "deployment_zone_trap_categorical_no_selection_bias",
    "deployment_zone_trap_categorical_simpsons_paradox",
    "deployment_zone_trap_env_shift",
    "weather_noise",
)

_ACTION_HELP = (
    "Reply with exactly one JSON object and nothing else.\n"
    "To run an experiment:\n"
    '  {"action": "deploy", "design": {"engine_def": 20, "cockpit_def": 20, '
    '"wing_def": 15, "body_def": 15, "antenna_def": 10, "camera_def": 5, '
    '"gun_def": 5}, "count": 20}\n'
    "To read the recorded history again:\n"
    '  {"action": "history"}\n'
    "To commit your final design (once, irreversible):\n"
    '  {"action": "submit", "design": {"engine_def": 20, "...": 0}}'
)


class CausalGameAdapter(PooledDatasetAdapter):
    """One episode per scenario against the project's local simulation backend."""

    adapter_version = "1.0"

    system_prompt = ""  # the release's per-scenario prompt.md is the system prompt
    data_delivery_mode = "interactive"
    objective_metrics = True
    selection_cardinality = None
    hypothesis_modes = ("generation",)
    hypothesis_mode_options = {"generation": {"subtask": "generation"}}
    table_hypothesis_mode = "Generation"
    primary_metric = "victory_rate"
    higher_is_better = True
    max_turns = 24

    # ------------------------------------------------------------------ #
    # setup
    # ------------------------------------------------------------------ #

    @property
    def _base_url(self) -> str:
        return str(self.context.option("base_url", "http://127.0.0.1:8000")).rstrip("/")

    def prepare(self) -> None:
        # The clone and the backend probe come first: load_items() reads the
        # scenario prompts out of the clone, and super().prepare() is what
        # calls it and fills the pool.
        self._repo = self._ensure_repo()
        self._probe_backend()
        super().prepare()

    def _ensure_repo(self):
        configured = self.context.option("repo_dir", None)
        root = (
            Path(str(configured))
            if configured
            else self.context.data_dir / "CausalGame"
        )
        if (root / "experiments").is_dir():
            return root
        try:
            root = C.ensure_git_repo(REPO_URL, root, depth=1, offline=self.context.offline)
        except Exception as exc:  # noqa: BLE001
            raise SkippedDataset(
                f"CausalGame's scenario prompts are not present and could not be fetched "
                f"({type(exc).__name__}: {exc}). SETUP: "
                f"git clone --depth 1 {REPO_URL} {root}"
            ) from exc
        if not (root / "experiments").is_dir():
            raise SkippedDataset(
                f"{root} has no experiments/ directory. SETUP: "
                f"git clone --depth 1 {REPO_URL} {root}"
            )
        return root

    def _probe_backend(self) -> None:
        """The environment is a service; say exactly how to start it if it is down."""
        try:
            health = self._get("/api/health")
        except Exception as exc:  # noqa: BLE001
            raise SkippedDataset(
                f"CausalGame's simulation backend is not reachable at {self._base_url} "
                f"({type(exc).__name__}: {exc}). The environment IS this service, and it "
                f"is part of the release, so this is a setup step rather than a missing "
                f"benchmark. SETUP:\n"
                f"  cd {self._repo}\n"
                f"  python -m venv .venv && .venv/bin/pip install -r api/requirements.txt\n"
                f"  .venv/bin/uvicorn api.app:app --host 127.0.0.1 --port 8000\n"
                f"Then set options.base_url if it is not http://127.0.0.1:8000."
            ) from exc
        if str(health.get("status")) != "healthy":
            raise SkippedDataset(
                f"CausalGame backend at {self._base_url} answered /api/health with "
                f"{health!r} rather than status=healthy"
            )

    # -- HTTP ----------------------------------------------------------- #

    def _admin_headers(self) -> dict[str, str]:
        """``X-Admin-Token``, which every ``/api/admin/*`` route requires.

        The backend mints the token at startup and appends it to ``<repo>/.env``
        (``api/security.py``), so the usual case needs no configuration: read it
        out of the clone the server was started from. ``options.admin_token``
        and ``ABENCH_CAUSALGAME_ADMIN_TOKEN`` override that, for a server whose
        repository is somewhere else.
        """
        token = self.context.option("admin_token", None) or os.environ.get(
            "ABENCH_CAUSALGAME_ADMIN_TOKEN"
        )
        if not token:
            token = self._token_from_env_file()
        if not token:
            raise SkippedDataset(
                "CausalGame's /api/admin/* routes need an X-Admin-Token and none was found. "
                "The backend writes one to <repo>/.env when it starts, so either point "
                "options.repo_dir at the clone the running server was started from, or set "
                "ABENCH_CAUSALGAME_ADMIN_TOKEN to the ADMIN_TOKEN value in that file."
            )
        return {"X-Admin-Token": str(token)}

    def _token_from_env_file(self) -> str:
        """The LAST ``ADMIN_TOKEN`` in ``<repo>/.env``, which is the live one.

        ``api/security.py`` *appends* a freshly minted token to that file every
        time the server starts without one in its environment, so the file
        accumulates them and the newest wins -- the same way a shell sourcing it
        would resolve the variable. Reading the first line instead sent a stale
        token and every episode died on a 403 from
        ``/api/admin/experiment/switch``.
        """
        env_file = self._repo / ".env"
        if not env_file.is_file():
            return ""
        found = ""
        for line in C.read_text(env_file).splitlines():
            if line.strip().startswith("ADMIN_TOKEN="):
                found = line.split("=", 1)[1].strip()
        return found

    def _get(self, path: str) -> Any:
        return C.http_json("GET", f"{self._base_url}{path}", timeout=120)

    def _post(self, path: str, payload: dict[str, Any] | None = None) -> Any:
        admin = path.startswith("/api/admin/")
        headers = self._admin_headers() if admin else None
        try:
            return C.http_json(
                "POST", f"{self._base_url}{path}", json_body=payload, headers=headers, timeout=300
            )
        except Exception as exc:  # noqa: BLE001
            # A backend restarted mid-run mints a new token and appends it, so
            # the one cached at prepare() time is now stale. Re-read and retry
            # once rather than failing every remaining episode.
            if not admin or "403" not in str(exc):
                raise
            self._cached_admin_token = ""
            fresh = self._token_from_env_file()
            if not fresh:
                raise
            self.log.warning(
                "CausalGame refused the admin token (403); re-read %s and retrying",
                self._repo / ".env",
            )
            return C.http_json(
                "POST",
                f"{self._base_url}{path}",
                json_body=payload,
                headers={"X-Admin-Token": fresh},
                timeout=300,
            )

    def _switch(self, scenario: str) -> dict[str, Any]:
        """Point the backend at one scenario and reset it.

        The backend holds a single global experiment, so episodes cannot be
        interleaved: each is switched in, played, and scored before the next
        starts.  The engine batches turns across episodes, so this adapter
        keeps episodes to one scenario at a time by making the switch part of
        the episode's own first turn.
        """
        self._post(f"/api/admin/experiment/switch?experiment_name={scenario}")
        self._post("/api/reset")
        return self._get("/api/v2/mission_status")

    # ------------------------------------------------------------------ #
    # items
    # ------------------------------------------------------------------ #

    def load_items(self) -> list[dict[str, Any]]:
        wanted = self.context.option("scenarios", None) or list(SCENARIOS)
        available = []
        for scenario in wanted:
            prompt = self._scenario_prompt_path(scenario)
            if prompt is None:
                self.log.warning("scenario %s has no prompt.md in the clone; skipped", scenario)
                continue
            available.append({"scenario": scenario, "prompt_path": str(prompt)})
        if not available:
            raise SkippedDataset(
                f"none of the requested scenarios exist under {self._repo}/experiments"
            )
        return available

    def _scenario_prompt_path(self, scenario: str):
        directory = self._repo / "experiments" / scenario
        for name in ("prompt.md", "no_tool_def_prompt.md"):
            candidate = directory / name
            if candidate.is_file():
                return candidate
        return None

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        scenario = item["scenario"]
        return SampleSpec(
            sample_id=C.stable_id("causalgame", scenario),
            fields={"observation": f"Scenario: {scenario}"},
            reference={"scenario": scenario},
            task_kind="generation",
            metadata={"scenario": scenario, "prompt_path": item["prompt_path"]},
        )

    # ------------------------------------------------------------------ #
    # the episode
    # ------------------------------------------------------------------ #

    def _render_release_prompt(self, path: str, status: dict[str, Any]) -> str:
        """The release's prompt.md, with its ``{{variable}}`` slots filled.

        The template is Mustache-like and the loader in ``agent/prompt_loader.py``
        renders it; only the substitutions the mission actually provides are
        needed here, and an unfilled conditional block is dropped rather than
        shown to the model as markup.
        """
        text = C.read_text(Path(path))
        values = {
            "total_drones": status.get("total_drone_budget", status.get("total_drones", 200)),
            "deployment_budget": status.get("stage1_deployment_budget", 10),
            "stage2_fleet_size": status.get("stage2_fleet_size", 1000),
            "victory_threshold": status.get("victory_threshold", 0.75),
            "history_count": status.get("history_count", 0),
        }
        # {{#if x}}...{{/if}} -- kept when the value is truthy, dropped otherwise.
        def _conditional(match: re.Match[str]) -> str:
            key, body = match.group(1), match.group(2)
            return body if values.get(key) else ""

        text = re.sub(r"\{\{#if (\w+)\}\}(.*?)\{\{/if\}\}", _conditional, text, flags=re.S)
        for key, value in values.items():
            text = text.replace(f"{{{{{key}}}}}", str(value))
        # Any slot the mission does not provide is removed rather than left as
        # literal braces in the prompt.
        return re.sub(r"\{\{[^}]*\}\}", "", text).strip()

    def interactive_start(self, sample: SampleSpec) -> tuple[list[ChatMessage], dict[str, Any]]:
        scenario = sample.reference["scenario"]
        status = self._switch(scenario)
        action_space = self._get("/api/v2/action_space").get("action_space", {})
        history = self._get("/api/mission_data")

        system = self._render_release_prompt(sample.metadata["prompt_path"], status)
        opening = [
            "Mission status:",
            json.dumps(status, indent=1),
            "",
            "Action space:",
            json.dumps(action_space, indent=1),
        ]
        if history:
            shown = history[: int(self.context.option("history_shown", 50))]
            opening += ["", f"Recorded history ({len(shown)} deployments):",
                        json.dumps(shown, indent=1)]
        opening += ["", _ACTION_HELP]
        state = {
            "scenario": scenario,
            "turn": 0,
            "deployments": 0,
            "deployment_budget": int(status.get("stage1_deployment_budget", 10)),
            "submitted": False,
            "result": None,
        }
        return (
            [
                ChatMessage(role="system", content=system),
                ChatMessage(role="user", content="\n".join(opening)),
            ],
            state,
        )

    def interactive_step(
        self, sample: SampleSpec, state: dict[str, Any], assistant_text: str
    ) -> str | None:
        if state.get("submitted"):
            return None
        state["turn"] = int(state.get("turn", 0)) + 1
        action = _parse_action(assistant_text)
        if action is None:
            return (
                "That was not a single JSON object I could read.\n" + _ACTION_HELP
            )
        kind = str(action.get("action", "")).lower()

        if kind == "history":
            return "Recorded history:\n" + json.dumps(self._get("/api/mission_data")[:50], indent=1)

        if kind == "deploy":
            if state["deployments"] >= state["deployment_budget"]:
                return (
                    f"Deployment budget exhausted ({state['deployment_budget']} used). "
                    "Submit your final design now.\n" + _ACTION_HELP
                )
            design = action.get("design")
            if not isinstance(design, dict) or not design:
                return "A deploy action needs a `design` object.\n" + _ACTION_HELP
            payload = {"design": design, "count": int(action.get("count", 20) or 20)}
            try:
                result = self._post("/api/v2/deploy_drone", payload)
            except Exception as exc:  # noqa: BLE001 - the env's own refusal is the reply
                return f"The simulator refused that deployment: {exc}\n{_ACTION_HELP}"
            state["deployments"] += 1
            remaining = state["deployment_budget"] - state["deployments"]
            return (
                json.dumps(result, indent=1)
                + f"\n\nDeployments left: {remaining}."
                + ("\nThis was your last one; submit next." if remaining == 0 else "")
            )

        if kind == "submit":
            design = action.get("design")
            if not isinstance(design, dict) or not design:
                return "A submit action needs a `design` object.\n" + _ACTION_HELP
            try:
                result = self._post("/api/v2/evaluate_final_design", {"design": design})
            except Exception as exc:  # noqa: BLE001
                return f"The simulator refused that submission: {exc}\n{_ACTION_HELP}"
            state["submitted"] = True
            state["result"] = result
            return None

        return f"Unknown action {kind!r}.\n" + _ACTION_HELP

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
        """The simulator's own verdict; nothing here is a judgement call.

        An episode that never submitted scores zero on both metrics and is
        recorded as such -- running out of turns without committing is a real
        outcome of the benchmark, not a scoring failure.
        """
        state = sample.metadata.get("_episode_state") or {}
        result = state.get("result") or {}
        if not result:
            return SampleScore(
                metrics={"victory_rate": 0.0, "final_score": 0.0, "submitted": 0.0},
                prediction=None,
                parse_ok=False,
                details={"reason": "episode ended without a final submission"},
            )
        return SampleScore(
            metrics={
                "victory_rate": 1.0 if result.get("victory") else 0.0,
                "final_score": _percent(result.get("scoring", {}).get("final_score")),
                "survival": _percent(result.get("scoring", {}).get("survival_component")),
                "efficiency": _percent(result.get("scoring", {}).get("efficiency_component")),
                "submitted": 1.0,
                "deployments_used": float(state.get("deployments", 0)),
            },
            prediction=json.dumps(result.get("design", {}), sort_keys=True),
            details={
                "scenario": state.get("scenario"),
                "victory_threshold": result.get("victory_threshold"),
            },
        )

    def aggregate(self, scores) -> dict[str, float]:
        return aggregate_mean_metrics([score.metrics for score in scores])

    # ------------------------------------------------------------------ #
    # documentation
    # ------------------------------------------------------------------ #

    def documentation(self) -> AdapterDocumentation:
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="CausalGame",
            domain="Causal Science: Experimental Reasoning",
            source_url="https://github.com/viewsetting/CausalGame",
            processing_mode="Generation (interactive)",
            split_used=f"{len(SCENARIOS)} released scenarios, one episode each",
            abductive_subset=(
                "The whole benchmark. The model must infer a hidden causal structure -- which "
                "components actually drive survival -- from experiments it chooses, then commit "
                "to one design. The scenario variants each hide that structure differently "
                "(selection bias, Simpson's paradox, a local optimum, environment shift)."
            ),
            sampling_procedure=(
                "One episode per released scenario, in the release's own order. Not sampled: "
                "the 14 scenarios are the benchmark."
            ),
            metrics_description={
                "self_consistency_<metric>":
                "Every metric also gets a self_consistency_ counterpart: the plurality answer over "
                "modes.repeats samples of the same record, read off those samples rather than bought "
                "again. Available because this dataset's answers are checkable and so can coincide.",
                "victory_rate": (
                    "(PRIMARY, higher is better, 0-1) fraction of episodes whose submitted design "
                    "beat the scenario's win threshold on the 1,000-drone evaluation fleet. The "
                    "simulator decides this, not a judge."
                ),
                "final_score": (
                    "(higher is better, 0-1) the simulator's own composite score for the "
                    "submitted design, averaged over episodes. victory_rate is this thresholded, "
                    "so final_score shows how close a loss was."
                ),
                "survival": "(higher is better, 0-1) the survival component of final_score",
                "efficiency": "(higher is better, 0-1) the efficiency component of final_score",
                "submitted": (
                    "(higher is better, 0-1) fraction of episodes that committed a design at all. "
                    "Below 1.0 means episodes ran out of turns first, and those score 0."
                ),
                "deployments_used": (
                    "mean experiments run before committing, out of the scenario's budget -- how "
                    "much evidence the model gathered."
                ),
                "self_consistency_victory_rate": (
                    "(higher is better) the same metric over the plurality answer of "
                    "modes.repeats episodes of each scenario. Available because the outcome is "
                    "checkable: no judge is involved."
                ),
            },
            primary_metric="victory_rate",
            decisions=[
                "Ran the project's own FastAPI backend (uvicorn api.app:app) rather than the "
                "public server: the environment is released, so it is run locally and "
                "reproducibly.",
                "Used the release's per-scenario experiments/<scenario>/prompt.md as the system "
                "prompt, rendered with the mission's numbers, because an interactive "
                "benchmark's prompt is part of the benchmark.",
                "Replaced the project's sandboxed-Python action channel with one JSON action per "
                "turn, calling the same API endpoints. Generated code in a sandbox would put a "
                "code interpreter inside the evaluation of the model.",
                "All 14 released scenarios are run, one episode each: the variants are the "
                "benchmark's design, not a sample of it.",
            ],
            caveats=[
                "The backend holds ONE global experiment, so episodes cannot be interleaved: "
                "each scenario is switched in and reset at the start of its own episode. Two "
                "runs sharing one backend would corrupt each other -- give each run its own "
                "port.",
                "An episode that never submits scores 0. That is the benchmark's rule (the "
                "submission is irreversible and mandatory), not a parsing artefact.",
                "The simulator is stochastic: repeats of one scenario differ, which is what "
                "makes repeats and self-consistency meaningful here.",
            ],
            statistics={**self.base_statistics(), "scenarios": len(SCENARIOS)},
        )


def _percent(value: Any) -> float:
    """``"42.3%"`` -> ``0.423``; the API reports these as formatted strings."""
    if value is None:
        return 0.0
    text = str(value).strip().rstrip("%")
    try:
        number = float(text)
    except ValueError:
        return 0.0
    return number / 100.0 if str(value).strip().endswith("%") else number


def _parse_action(text: str | None) -> dict[str, Any] | None:
    """The last JSON object in the response, which is the action."""
    if not text:
        return None
    for match in reversed(list(re.finditer(r"\{.*\}", text, flags=re.S))):
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "action" in parsed:
            return parsed
    return None
