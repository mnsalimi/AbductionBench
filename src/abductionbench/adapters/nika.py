"""NIKA: diagnose a real network incident through MCP tools, then name the cause.

Source: https://github.com/sands-lab/nika  ·  https://sands-lab.github.io/nika/

**The task.** A fault is injected into an emulated network -- a BGP ASN
misconfiguration, an ARP ACL block, a poisoned ARP cache -- and the model is put
on call. It probes the live network through MCP tools (``ping_pair``,
``traceroute``, ``frr_show_ip_route``, ``run_pingmesh_snapshot``, packet
capture), forms a hypothesis, and then commits: ``submit`` with
``is_anomaly`` and ``root_causes: [{resource_id, fault_type}, ...]``.  The
release scores that against its own ground truth and its leaderboard metric is
``rca_f1``, with ``judge_allowed: false`` -- so this is abduction with a
checkable answer and no judge anywhere in it.

**The items and their gold answers ship with the repository**
(``benchmark/releases/<version>/{dev,test}.yaml``): scenario, injected problem,
and the ``root_causes`` list.  What does *not* ship is the observations: they
exist only while the emulator is running, because the whole point is that the
agent measures a live network rather than reading a recorded trace.

**What running it needs, specifically.**  NIKA emulates with Kathará, which
drives one container per network device and rewires host networking. That needs

* a container runtime -- Kathará uses Docker (or Megalos on Kubernetes), and
* ``CAP_NET_ADMIN`` plus network-namespace creation, for the virtual links.

Neither is available inside this instance: it is an unprivileged container with
no Docker, ``unshare --net`` refused, and ``cap_net_admin``/``cap_sys_admin``
dropped from its capability set.  That is a property of *this host*, not of the
benchmark, and it is not something a different adapter could work around.

So this adapter is written against NIKA's MCP gateway and reaches a deployment
wherever one is running:

    # on a host with Docker and NET_ADMIN
    nika env run dc-clos-bgp
    # then, here
    export ABENCH_NIKA_GATEWAY_URL=http://<that-host>:<gateway-port>
    export ABENCH_NIKA_SESSION_ID=<session id>

With those set, the dataset runs.  Without them it is skipped with this exact
requirement, rather than reported as an unavailable benchmark.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from ..core.adapter import SkippedDataset
from ..core.metrics import aggregate_mean_metrics, set_prf
from ..core.types import (
    AdapterDocumentation,
    ChatMessage,
    ModelResponse,
    SampleScore,
    SampleSpec,
)
from . import _common as C
from ._base import PooledDatasetAdapter, unparsed_score

REPO_URL = "https://github.com/sands-lab/nika.git"

#: Tools the release mounts on every diagnosis session, with the arguments the
#: docs give.  Advertised to the model verbatim so the action space is the
#: benchmark's, not one invented here.
_ALWAYS_ON_TOOLS = {
    "kathara_base_mcp_server": (
        "ping_pair", "traceroute", "get_host_net_config", "get_tc_statistics",
        "netstat", "ip_addr_statistics", "ethtool", "curl_web_test",
        "iperf_test", "active_tcp_probe", "cat_file", "exec_shell",
    ),
    "pingmesh_mcp_server": ("run_pingmesh_snapshot",),
    "packet_capture_mcp_server": (
        "packet_capture_start", "packet_capture_stop", "packet_capture_inspect",
    ),
}

_ACTION_HELP = (
    "Reply with exactly one JSON object and nothing else.\n"
    "To call a diagnostic tool:\n"
    '  {"action": "tool", "server": "kathara_base_mcp_server", '
    '"tool": "ping_pair", "arguments": {"source": "leaf0", "destination": "spine0"}}\n'
    "To list what a server exposes:\n"
    '  {"action": "tools", "server": "kathara_frr_mcp_server"}\n'
    "To commit your diagnosis (once, irreversible):\n"
    '  {"action": "submit", "is_anomaly": true, "diagnosis_report": "...", '
    '"root_causes": [{"resource_id": "leaf0", "fault_type": "bgp_asn_misconfig"}]}'
)


class NikaAdapter(PooledDatasetAdapter):
    """One episode per released incident, diagnosed through NIKA's MCP gateway."""

    adapter_version = "1.0"

    system_prompt = ""  # the release's session prompt is used when reachable
    data_delivery_mode = "interactive"
    objective_metrics = True          # scored by the release's rule-based rca_f1
    selection_cardinality = None
    hypothesis_modes = ("generation",)
    hypothesis_mode_options = {"generation": {"subtask": "generation"}}
    table_hypothesis_mode = "Generation"
    primary_metric = "rca_f1"
    higher_is_better = True
    max_turns = 30

    # ------------------------------------------------------------------ #
    # setup
    # ------------------------------------------------------------------ #

    @property
    def _gateway(self) -> str:
        url = self.context.option("gateway_url", None) or os.environ.get(
            "ABENCH_NIKA_GATEWAY_URL", ""
        )
        return str(url).rstrip("/")

    @property
    def _session_id(self) -> str:
        return str(
            self.context.option("session_id", None)
            or os.environ.get("ABENCH_NIKA_SESSION_ID", "")
        )

    def prepare(self) -> None:
        self._repo = self._ensure_repo()
        self._require_gateway()
        super().prepare()

    def _ensure_repo(self) -> Path:
        configured = self.context.option("repo_dir", None)
        root = Path(str(configured)) if configured else self.context.data_dir / "nika"
        if (root / "benchmark").is_dir():
            return root
        try:
            root = C.ensure_git_repo(REPO_URL, root, depth=1, offline=self.context.offline)
        except Exception as exc:  # noqa: BLE001
            raise SkippedDataset(
                f"NIKA's case files (which carry the ground truth) are not present and "
                f"could not be fetched ({type(exc).__name__}: {exc}). SETUP: "
                f"git clone --depth 1 {REPO_URL} {root}"
            ) from exc
        return root

    def _require_gateway(self) -> None:
        """The observations are a live network; say exactly what is missing."""
        if self._gateway and self._session_id:
            try:
                C.http_json("GET", f"{self._gateway}/healthz", timeout=30)
                return
            except Exception:  # noqa: BLE001 - some gateways expose no /healthz
                return  # the first tool call will surface a real failure
        raise SkippedDataset(
            "NIKA needs a running incident to observe: its cases ship with the repository "
            "(including their root-cause ground truth) but the telemetry exists only while "
            "the emulator is up, which is the point of the benchmark.\n"
            "\n"
            "WHY THIS INSTANCE CANNOT RUN IT, precisely:\n"
            "  * NIKA emulates with Kathara, which runs one container per network device "
            "(Docker, or Megalos on Kubernetes) and wires them with veth pairs.\n"
            "  * No container runtime can even be installed here. Every one of them -- "
            "rootful Docker, rootless Docker, podman, containerd -- needs mount and user "
            "namespaces, and this container is refused all three namespace types: "
            "`unshare -m`, `unshare -U` and `unshare -n` all return EPERM.\n"
            "  * Kathara additionally needs CAP_NET_ADMIN for the virtual links, and the "
            "effective capability set here holds neither CAP_NET_ADMIN nor CAP_SYS_ADMIN.\n"
            "  * BEING ROOT DOES NOT HELP, and this is the decisive point. Root's power is "
            "its capabilities, and the container's *bounding set* is a hard ceiling: "
            "CapBnd == CapEff == 0xa80405fb here, which excludes CAP_SYS_ADMIN, "
            "CAP_NET_ADMIN, CAP_NET_RAW and CAP_SYS_MODULE. A process cannot add a "
            "capability absent from its own bounding set -- that is the kernel's rule, not "
            "a permission anything can escalate past. The host dropped them when the "
            "container was created.\n"
            "  * Measured, not assumed: `ip netns add` and `ip link add ... type veth` both "
            "return EPERM; no host docker socket is mounted; /dev/kvm does not exist; and "
            "podman -- the most permissive, rootless-capable runtime -- fails before it "
            "starts with `cannot clone: Operation not permitted`.\n"
            "  * There is nothing to replay offline either. The release ships case lists and "
            "root-cause ground truth, but no recorded telemetry: its only test fixture is "
            "another 4 KB case list. A static variant would mean inventing observations, "
            "which would be a different benchmark wearing NIKA's name.\n"
            "\n"
            "TWO WAYS TO RUN IT, both supported without code changes:\n"
            "  A. A Vast.ai VM instance (rather than a container instance) has the kernel "
            "access Kathara needs; install NIKA there and run it locally.\n"
            "  B. Leave this instance as it is and run NIKA on any host that has Docker "
            "and NET_ADMIN, then point this run at it over the network:\n"
            "\n"
            "SETUP on that host:\n"
            "  git clone https://github.com/sands-lab/nika && cd nika\n"
            "  nika env run dc-clos-bgp        # stand up a scenario\n"
            "  nika agent run --agent <yours>  # starts a session and its MCP gateway\n"
            "then point this run at it:\n"
            "  export ABENCH_NIKA_GATEWAY_URL=http://<host>:<gateway-port>\n"
            "  export ABENCH_NIKA_SESSION_ID=<session id>\n"
            "The adapter drives the session's own MCP tools and the release's `submit`, "
            "and scores rca_f1 against the shipped ground truth."
        )

    # -- MCP over the session gateway ----------------------------------- #

    def _mcp(self, server: str, method: str, params: dict[str, Any]) -> Any:
        """One JSON-RPC call to a mounted MCP server.

        The gateway mounts each server at ``/mcp/<name>`` and serves streamable
        HTTP under ``/mcp``, so a server's endpoint is ``/mcp/<name>/mcp``.
        """
        self._rpc_id = getattr(self, "_rpc_id", 0) + 1
        body = {"jsonrpc": "2.0", "id": self._rpc_id, "method": method, "params": params}
        payload = C.http_json(
            "POST",
            f"{self._gateway}/mcp/{server}/mcp",
            json_body=body,
            headers={
                "Accept": "application/json, text/event-stream",
                "X-Nika-Session-Id": self._session_id,
            },
            timeout=300,
        )
        if isinstance(payload, dict) and payload.get("error"):
            raise RuntimeError(str(payload["error"])[:400])
        return (payload or {}).get("result", payload)

    def _call_tool(self, server: str, tool: str, arguments: dict[str, Any]) -> Any:
        return self._mcp(server, "tools/call", {"name": tool, "arguments": arguments or {}})

    # ------------------------------------------------------------------ #
    # items -- from the release, with their ground truth
    # ------------------------------------------------------------------ #

    def load_items(self) -> list[dict[str, Any]]:
        version = str(self.context.option("release", "") or "").strip()
        releases = self._repo / "benchmark" / "releases"
        if not releases.is_dir():
            raise SkippedDataset(f"{releases} does not exist in the clone")
        if version:
            release_dir = releases / version
        else:
            # Newest released version, by version-sorted directory name.
            candidates = sorted(
                (p for p in releases.iterdir() if p.is_dir()),
                key=lambda p: [int(part) if part.isdigit() else 0
                               for part in re.split(r"\D+", p.name) if part != ""],
            )
            if not candidates:
                raise SkippedDataset(f"no release directories under {releases}")
            release_dir = candidates[-1]
        manifest = release_dir / "RELEASE.yaml"
        split = str(self.context.option("split", "") or "").strip()
        if not split and manifest.is_file():
            meta = C.read_yaml(manifest) or {}
            split = str(meta.get("default_split_for_release", "test"))
        split = split or "test"
        cases_file = release_dir / f"{split}.yaml"
        if not cases_file.is_file():
            raise SkippedDataset(f"{cases_file} does not exist")
        cases = (C.read_yaml(cases_file) or {}).get("cases") or []
        if not cases:
            raise SkippedDataset(f"{cases_file} lists no cases")
        self.split_used = f"{split} ({len(cases)} incidents) of NIKA {release_dir.name}"
        self._release_dir = release_dir
        return list(cases)

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        scenario = str(item.get("scenario") or "").strip()
        problem = str(item.get("problem") or "").strip()
        gold = _gold_causes(item.get("root_causes"))
        if not scenario or not problem or not gold:
            return None
        return SampleSpec(
            sample_id=C.stable_id("nika", scenario, problem, item.get("topo_size", "")),
            fields={
                "observation": (
                    f"Network scenario: {scenario}\n"
                    f"Topology size: {item.get('topo_size', 'default')}\n"
                    "Something is wrong. Diagnose it with the tools available."
                )
            },
            reference={"root_causes": sorted(gold), "problem": problem, "scenario": scenario},
            task_kind="generation",
            metadata={
                "scenario": scenario,
                "problem": problem,
                "topo_size": item.get("topo_size"),
                # The injection point is the answer; kept out of records.
                "_inject": item.get("inject"),
            },
        )

    # ------------------------------------------------------------------ #
    # the episode
    # ------------------------------------------------------------------ #

    def interactive_start(self, sample: SampleSpec) -> tuple[list[ChatMessage], dict[str, Any]]:
        catalogue = ["Available MCP servers and tools:"]
        for server, tools in _ALWAYS_ON_TOOLS.items():
            catalogue.append(f"- {server}: {', '.join(tools)}")
        catalogue.append(
            "- kathara_frr_mcp_server (routing scenarios): frr_get_bgp_conf, "
            "frr_show_running_config, frr_show_ip_route, frr_get_ospf_conf, frr_exec"
        )
        system = (
            "You are on call for a network incident. A fault has been injected into a live "
            "emulated network and you must find it. Probe the network with the tools "
            "available, then commit to a root cause: the resource it is on and the type of "
            "fault it is. You are scored on whether that root cause is right, not on how "
            "many tools you called."
        )
        opening = [
            sample.fields["observation"],
            "",
            "\n".join(catalogue),
            "",
            _ACTION_HELP,
        ]
        return (
            [
                ChatMessage(role="system", content=system),
                ChatMessage(role="user", content="\n".join(opening)),
            ],
            {"turn": 0, "calls": 0, "submitted": False, "submission": None},
        )

    def interactive_step(
        self, sample: SampleSpec, state: dict[str, Any], assistant_text: str
    ) -> str | None:
        if state.get("submitted"):
            return None
        state["turn"] = int(state.get("turn", 0)) + 1
        action = _parse_action(assistant_text)
        if action is None:
            return "That was not a single JSON object I could read.\n" + _ACTION_HELP
        kind = str(action.get("action", "")).lower()

        if kind == "tools":
            server = str(action.get("server", "kathara_base_mcp_server"))
            try:
                return json.dumps(self._mcp(server, "tools/list", {}), indent=1)[:4000]
            except Exception as exc:  # noqa: BLE001
                return f"{server} could not be listed: {exc}\n{_ACTION_HELP}"

        if kind == "tool":
            server = str(action.get("server", ""))
            tool = str(action.get("tool", ""))
            if not server or not tool:
                return "A tool action needs `server` and `tool`.\n" + _ACTION_HELP
            state["calls"] = int(state.get("calls", 0)) + 1
            try:
                result = self._call_tool(server, tool, action.get("arguments") or {})
            except Exception as exc:  # noqa: BLE001 - the tool's refusal is the reply
                return f"{server}.{tool} failed: {exc}\n{_ACTION_HELP}"
            return json.dumps(result, indent=1)[:6000]

        if kind == "submit":
            causes = action.get("root_causes")
            if not isinstance(causes, list) or not causes:
                return (
                    "A submit action needs `root_causes`, a list of "
                    '{"resource_id": ..., "fault_type": ...}.\n' + _ACTION_HELP
                )
            report = str(action.get("diagnosis_report", ""))[:4000]
            try:
                # The release gates submission behind a phase advance, then
                # accepts exactly one submit.
                self._call_tool(
                    "task_mcp_server",
                    "begin_submission_mcp_phase",
                    {"session_id": self._session_id, "diagnosis_report": report},
                )
                submitted = self._call_tool(
                    "task_mcp_server",
                    "submit",
                    {
                        "is_anomaly": bool(action.get("is_anomaly", True)),
                        "root_causes": causes,
                    },
                )
            except Exception as exc:  # noqa: BLE001
                # A gateway failure must not become a silent wrong answer: the
                # claim is recorded and scored locally against the release's
                # ground truth, and the failure is noted in the transcript.
                submitted = {"error": str(exc)[:300]}
            state["submitted"] = True
            state["submission"] = {"root_causes": causes, "server_reply": submitted}
            return None

        return f"Unknown action {kind!r}.\n" + _ACTION_HELP

    # ------------------------------------------------------------------ #
    # scoring -- the release's rule-based rca_f1, no judge
    # ------------------------------------------------------------------ #

    def score(
        self,
        sample: SampleSpec,
        response: ModelResponse,
        *,
        output_contract: dict[str, Any] | None = None,
    ) -> SampleScore:
        state = sample.metadata.get("_episode_state") or {}
        submission = state.get("submission") or {}
        claimed = _claimed_causes(submission.get("root_causes"))
        if not claimed:
            return unparsed_score(
                ["rca_f1", "rca_precision", "rca_recall", "rca_exact", "submitted"],
                reason="episode ended without a submitted root cause",
            )
        gold = set(sample.reference["root_causes"])
        prf = set_prf(claimed, gold)
        return SampleScore(
            metrics={
                "rca_f1": prf["f1"],
                "rca_precision": prf["precision"],
                "rca_recall": prf["recall"],
                "rca_exact": float(claimed == gold),
                "submitted": 1.0,
                "tool_calls": float(state.get("calls", 0)),
            },
            prediction="; ".join(sorted(claimed)),
            details={"gold": "; ".join(sorted(gold)), "problem": sample.reference["problem"]},
        )

    def aggregate(self, scores) -> dict[str, float]:
        return aggregate_mean_metrics([score.metrics for score in scores])

    # ------------------------------------------------------------------ #
    # documentation
    # ------------------------------------------------------------------ #

    def documentation(self) -> AdapterDocumentation:
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="NIKA",
            domain="Computing Systems: Network Troubleshooting",
            source_url="https://github.com/sands-lab/nika",
            processing_mode="Generation (interactive)",
            split_used=getattr(self, "split_used", "release split"),
            abductive_subset=(
                "The whole benchmark. A fault is injected into a live emulated network and the "
                "model must infer which resource carries which fault from the symptoms it "
                "measures -- reachability, routes, captures -- then commit to that cause."
            ),
            sampling_procedure=(
                "The release's own split file, in its order, one episode per incident. The "
                "cases and their root-cause ground truth ship with the repository."
            ),
            metrics_description={
                "self_consistency_<metric>":
                "Every metric also gets a self_consistency_ counterpart: the plurality answer over "
                "modes.repeats samples of the same record, read off those samples rather than bought "
                "again. Available because this dataset's answers are checkable and so can coincide.",
                "rca_f1": (
                    "(PRIMARY, higher is better, 0-1) F1 between the submitted "
                    "{resource_id, fault_type} pairs and the release's root_causes. This is "
                    "NIKA's own leaderboard_primary, and its RELEASE.yaml sets "
                    "judge_allowed: false -- the answer is checked, never judged."
                ),
                "rca_precision": "(higher is better, 0-1) of the causes claimed, how many were real",
                "rca_recall": "(higher is better, 0-1) of the real causes, how many were found",
                "rca_exact": (
                    "(higher is better, 0-1) 1.0 only when the claimed set equals the gold set "
                    "exactly -- a stricter reading than rca_f1"
                ),
                "submitted": (
                    "(higher is better, 0-1) fraction of episodes that committed a cause at "
                    "all; below 1.0 means episodes ran out of turns, and those score 0"
                ),
                "tool_calls": "mean diagnostic tool calls before committing -- evidence gathered",
                "self_consistency_rca_f1": (
                    "(higher is better) rca_f1 of the plurality answer over modes.repeats "
                    "episodes of one incident. Available because the answer is checkable."
                ),
            },
            primary_metric="rca_f1",
            decisions=[
                "Took the cases and their ground truth from the release's own split file, so "
                "the evaluation set is the benchmark's rather than one assembled here.",
                "Scored with rca_f1 over {resource_id, fault_type} pairs, which is the "
                "release's leaderboard_primary; no LLM judge is used, because the release "
                "sets judge_allowed: false.",
                "Drove the session's published MCP tools through the gateway, one JSON action "
                "per turn, rather than embedding a coding agent: NIKA's own CLI harness runs "
                "an agent framework, which would evaluate that framework as much as the model.",
            ],
            caveats=[
                "REQUIRES A RUNNING NIKA DEPLOYMENT, and this container cannot host one at "
                "all: Kathara needs a container runtime, and no runtime -- not even a "
                "rootless one -- can be installed where `unshare -m`, `unshare -U` and "
                "`unshare -n` all return EPERM and the capability set holds neither "
                "CAP_NET_ADMIN nor CAP_SYS_ADMIN. This is the container's seccomp profile, "
                "not the kernel, so only the host operator can lift it. Either use a VM "
                "instance, or point ABENCH_NIKA_GATEWAY_URL and ABENCH_NIKA_SESSION_ID at a "
                "deployment on a host that has both -- the adapter needs no change either way.",
                "The observations are live, so two runs of one incident are not identical; "
                "that is what makes repeats and self-consistency meaningful here.",
                "One session serves one incident at a time. Episodes are not interleaved "
                "across incidents on a single gateway.",
            ],
            statistics={**self.base_statistics()},
        )


def _gold_causes(raw: Any) -> set[str]:
    """``{"resource": {"node": "leaf0"}, "fault_type": "x"}`` -> ``{"leaf0::x"}``."""
    out: set[str] = set()
    for entry in raw or []:
        if not isinstance(entry, dict):
            continue
        resource = entry.get("resource") or {}
        name = ""
        if isinstance(resource, dict):
            name = str(
                resource.get("node")
                or resource.get("name")
                or resource.get("resource_id")
                or ""
            ).strip()
        else:
            name = str(resource).strip()
        fault = str(entry.get("fault_type") or "").strip()
        if name and fault:
            out.add(f"{name}::{fault}")
    return out


def _claimed_causes(raw: Any) -> set[str]:
    """The same key shape, from what the model submitted."""
    out: set[str] = set()
    for entry in raw or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("resource_id") or entry.get("resource") or "").strip()
        fault = str(entry.get("fault_type") or "").strip()
        if name and fault:
            out.add(f"{name}::{fault}")
    return out


def _parse_action(text: str | None) -> dict[str, Any] | None:
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
