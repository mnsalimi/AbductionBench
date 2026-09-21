"""Cloud-OpsBench: root-cause analysis for Kubernetes incidents.

Source: https://github.com/LLM4Ops/Cloud-OpsBench

Each case directory holds ``metadata.json`` (the reported symptom in ``query``,
a ``difficulty`` label, and the ground truth: ``fault_taxonomy``,
``fault_object``, ``root_cause``) plus ``tool_cache.json`` -- cached outputs of
the diagnostic tools an agent would have called (pod listings, describes, logs,
events).

**How it is run.** The tool cache is what makes this benchmark interactive
without a cluster: it is a recording of the diagnostic calls an agent would
make, so the episode replays it.  The model is given the symptom and the tool
list, issues one call per turn (``Action`` / ``Action Input``, the release's own
syntax), and receives that call's recorded output -- or a miss, when it asks for
something the recording does not hold.  It ends by finalising a diagnosis.

That is also the only way this dataset *fits*: one case's cache runs to hundreds
of thousands of characters, far beyond any input budget, so a single-turn form
could only ever show an arbitrary slice of the evidence.  Letting the model
choose which slice to look at is both the benchmark's task and the thing that
makes it tractable.  ``options.delivery = static`` keeps the old bounded-slice
form for comparison.

**The protocol prompt is this suite's, adapted rather than quoted.**  The tool vocabulary, the ``Action`` / ``Action Input`` syntax, the single-fault constraint, the evidence rule ("do not conclude from an assumed cluster state") and the pivot rule are the release's and are reproduced in substance, because they are the benchmark.  Its *reasoning elicitation* is not: the release's ``Thought:`` line, its "provide a clear reasoning chain" instruction and its "limit your internal reasoning to a few concise sentences" are all dropped.  This dataset runs ``io`` only (``io_only``), and an instruction to explain a choice would contradict the mode instruction sitting in the same prompt -- and, here, the parser as well.  The wording is therefore this harness's, in the same five layers every static prompt uses.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..core.adapter import SkippedDataset
from ..core.metrics import aggregate_mean_metrics, contains_match, extract_answer_span
from ..core.types import (
    AdapterDocumentation,
    ChatMessage,
    ModelResponse,
    SampleScore,
    SampleSpec,
)
from . import _common as C
from ._base import PooledDatasetAdapter, apply_judged_metric, unparsed_score
from ._interactive import InteractiveMixin
from ._prompting import ProtocolParts, build_protocol_messages

REPO_URL = "https://github.com/LLM4Ops/Cloud-OpsBench"
#: Tool-cache keys are of the form ``ToolName:{json args}``; these come first
#: because they describe cluster state compactly.
_PREFERRED_TOOLS = ("GetResources", "DescribeResource", "GetEvents", "GetLogs")


def _parse_tool_call(text: str) -> tuple[str | None, dict[str, Any]]:
    """Read the release's ``Action`` / ``Action Input`` pair out of a turn."""
    action = re.search(r"Action\s*:\s*([A-Za-z_]+)", text or "")
    if not action:
        return None, {}
    arguments: dict[str, Any] = {}
    blob = re.search(r"Action\s*Input\s*:\s*(\{.*?\})", text or "", re.S)
    if blob:
        try:
            parsed = json.loads(blob.group(1))
            if isinstance(parsed, dict):
                arguments = parsed
        except json.JSONDecodeError:
            arguments = {}
    return action.group(1), arguments


def _lookup(cache: dict[str, Any], action: str, arguments: dict[str, Any]) -> tuple[Any, bool]:
    """Find a recorded call, exactly if possible and by closest arguments if not.

    Exact first, because the cache is keyed by the precise argument JSON the
    reference agent used. Then the same tool with the most argument values in
    common, so a model that asks for the right pod with one extra flag still
    gets the recording rather than a miss it cannot learn anything from.
    """
    exact = f"{action}:{json.dumps(arguments, separators=(',', ':'), sort_keys=False)}"
    if exact in cache:
        return cache[exact], True
    wanted = {str(value).lower() for value in arguments.values() if value not in (None, "")}
    best: tuple[int, str] | None = None
    for key in cache:
        name, _, raw = key.partition(":")
        if name != action:
            continue
        try:
            recorded = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            continue
        values = {str(value).lower() for value in recorded.values() if value not in (None, "")}
        if wanted and not (wanted & values):
            continue
        overlap = len(wanted & values) - abs(len(values) - len(wanted)) * 0.01
        if best is None or overlap > best[0]:
            best = (overlap, key)
    if best is not None:
        return cache[best[1]], True
    return None, False


class CloudOpsBenchAdapter(InteractiveMixin, PooledDatasetAdapter):
    """Name the root cause of a Kubernetes incident from symptom + evidence."""

    adapter_version = "1.0"

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. You are given the symptom of a "
        "Kubernetes incident and the evidence collected from the cluster. Name the root "
        "cause: the misconfiguration, resource limit or failure that accounts for the "
        "symptom."
    )
    data_delivery_mode = "interactive"
    #: The protocol prompt below is this suite's, adapted from the release's
    #: agent prompt rather than quoted from it, so this is False. One prompt
    #: mode still, for the reason io_only states.
    authors_prompt = False
    io_only = True
    #: Measured, not assumed: the root cause is a short technical label the model
    #: writes itself -- no candidate list is shown, and the release's own golds
    #: ("missing_service_account", "cpu_throttling") are one naming convention
    #: among many that describe the same fault. A string test here punishes "the
    #: deployment references a ServiceAccount that does not exist" for not being
    #: the gold's wording. The affected object, by contrast, is named in the tool
    #: output, so fault_object_match stays a mechanical check beside the verdict.
    objective_metrics = False
    selection_cardinality = None
    primary_metric = "root_cause_judged"

    # ------------------------------------------------------------------ #
    # the investigation -- replayed from the release's own tool cache
    # ------------------------------------------------------------------ #

    #: ``diagnosis.max_iterations: 20`` in the release's
    #: ``cloudops_agent/configs/model_configs.yaml``, passed to
    #: ``init_case_state(max_steps=...)`` and enforced by ``AgentRuntime.run``
    #: as ``while not state.finished and state.current_step < state.max_steps``.
    #: 15 was this adapter's own number.
    max_turns = 20
    #: NONE. The release counts every ReAct step against ``max_steps`` and caps
    #: tool calls nowhere -- there is no tool budget separate from the step
    #: budget. "You may make at most 12 tool calls" was invented here and, worse,
    #: stated in the prompt, so the agent was told a rule the benchmark does not
    #: have.
    category_limits: dict[str, int] = {}

    @staticmethod
    def _tool_signature(key: str) -> tuple[str, str]:
        name, _, arguments = key.partition(":")
        return name, arguments

    def interactive_start(self, sample: SampleSpec) -> tuple[list[ChatMessage], dict[str, Any]]:
        cache = sample.metadata.get("_tool_cache") or {}
        tools = sorted({self._tool_signature(key)[0] for key in cache if ":" in key})
        parts = ProtocolParts(
            system=self.system_prompt,
            context=(
                f"Reported symptom: {sample.metadata.get('query', '')}\n"
                f"Namespace: {sample.metadata.get('namespace', '')}"
            ),
            observation="Diagnostic tools available on this cluster: " + ", ".join(tools),
            instructions=(
                "Find the root cause of this incident. You choose which tools to run and in "
                "what order, then finalise with the fault you have established."
            ),
            requirements=[
                "this scenario contains exactly one primary fault",
                "issue exactly one tool call per turn",
                "your conclusion must be supported by output a tool actually returned, not "
                "by an assumed cluster state",
                "if a tool returns no anomaly, drop that hypothesis and investigate a "
                "different path",
                "reach the fault in as few tool calls as you can",
                "finalise as soon as the evidence establishes the fault",
            ],
            actions=[
                ("<ToolName>", "run one of the diagnostic tools listed above."),
                ("Finalize", "submit the root cause. This ends the investigation."),
            ],
            output_format=(
                "Reply in exactly this form and nothing else:\n"
                "Action: <ToolName>\n"
                'Action Input: {"key": "value"}\n'
                "Action Input is mandatory; use {} for a tool that takes no parameters.\n\n"
                "To finalise:\n"
                "Action: Finalize\n"
                'Action Input: {"root_cause": "<the fault>", "fault_object": "kind/name"}'
            ),
        )
        return (
            build_protocol_messages(parts, self.context.modes),
            {"cache": cache, "counts": {}, "calls": []},
        )

    def interactive_step(
        self, sample: SampleSpec, state: dict[str, Any], assistant_text: str
    ) -> str | None:
        action, arguments = _parse_tool_call(assistant_text)
        if action is None:
            state["parse_errors"] = state.get("parse_errors", 0) + 1
            if state["parse_errors"] > 2:
                return None
            return (
                "Could not read a tool call. Reply with:\nAction: <ToolName>\n"
                'Action Input: {"key": "value"}'
            )
        if action.lower() in ("finalize", "finalise", "final_answer", "answer"):
            return None

        self.bump(state, "tool")
        cache = state["cache"]
        output, matched = _lookup(cache, action, arguments)
        state["calls"].append({"action": action, "arguments": arguments, "hit": matched})
        if not matched:
            # The recording holds only the calls the reference agent made. Saying
            # so is honest: inventing a plausible kubectl output would be
            # fabricating cluster state, and the model would reason from it.
            return self._with_budget(
                state,
                f"{action}: no recorded output for those arguments in this case. "
                "Try a different call.",
            )
        body = C.clip_words(str(output), int(self.context.option("words_per_tool", 400)))
        return self._with_budget(state, body)

    def _with_budget(self, state: dict[str, Any], body: str) -> str:
        """The step counter, and the release's forced final step.

        ``PromptBuilder._build_case_section`` puts

            Current Step: {state.current_step + 1}
            Budget Steps: {state.max_steps}

        in the prompt on every step, and
        ``PromptBuilder._build_current_step_instruction`` replaces the normal
        protocol on the last one with

            This is the final allowed step. You MUST now stop calling tools and
            output the final diagnosis JSON only, using the best evidence
            collected so far.

        This adapter showed neither, so the agent could not see the clock and
        was never told to commit -- it simply stopped getting answers.
        """
        used = (state.get("counts") or {}).get("tool", 0)
        lines = [body, f"Current Step: {used + 1}", f"Budget Steps: {self.max_turns}"]
        if used + 1 >= self.max_turns:
            lines.append(
                "This is the final allowed step. You MUST now stop calling tools and "
                "output the final diagnosis, using the best evidence collected so far."
            )
        return "\n\n".join(lines)

    def load_items(self) -> list[dict[str, Any]]:
        root = C.ensure_git_repo(
            REPO_URL, self.context.data_dir / "repo", offline=self.context.offline
        )
        metadata_files = C.find_files(root / "benchmark", ["metadata.json"])
        if not metadata_files:
            raise SkippedDataset("no Cloud-OpsBench case metadata found")
        items: list[dict[str, Any]] = []
        self._namespaces: dict[str, int] = {}
        for path in metadata_files:
            metadata = C.read_json(path)
            result = metadata.get("result") or {}
            if not result.get("root_cause"):
                continue
            namespace = str(metadata.get("namespace") or path.parents[2].name)
            items.append(
                {
                    "case_dir": str(path.parent),
                    "case_id": f"{namespace}/{path.parents[1].name}/{path.parent.name}",
                    "metadata": metadata,
                }
            )
            self._namespaces[namespace] = self._namespaces.get(namespace, 0) + 1
        if not items:
            raise SkippedDataset("Cloud-OpsBench cases carry no root-cause ground truth")
        self.split_used = (
            f"all released cases with ground truth ({len(items)} across "
            f"{len(self._namespaces)} namespaces); the benchmark ships no train/test split"
        )
        return items

    def _evidence(self, case_dir: str) -> str:
        mode = str(self.context.option("evidence", "tool_cache"))
        if mode == "symptom_only":
            return ""
        from pathlib import Path

        path = Path(case_dir) / "tool_cache.json"
        if not path.exists():
            return ""
        try:
            cache = C.read_json(path)
        except Exception:  # noqa: BLE001 - a malformed cache must not break the run
            return ""
        if not isinstance(cache, dict):
            return ""
        max_entries = int(self.context.option("max_evidence_entries", 6))
        words_per_entry = int(self.context.option("words_per_entry", 220))
        chosen: list[tuple[str, Any]] = []
        for tool in _PREFERRED_TOOLS:
            for key, value in cache.items():
                if key.startswith(tool) and len(chosen) < max_entries:
                    chosen.append((key, value))
        blocks = []
        for key, value in chosen:
            body = value if isinstance(value, str) else str(value)
            blocks.append(
                f"$ {key.split(':')[0]} {key.split(':', 1)[1][:120] if ':' in key else ''}\n"
                + C.clip_words(C.normalize_whitespace(body), words_per_entry)
            )
        return "\n\n".join(blocks)

    def _tool_cache(self, case_dir) -> dict[str, Any]:
        """The case's recorded tool calls, or an empty recording."""
        if self.context.modes.data_delivery_mode != "interactive":
            return {}
        path = Path(case_dir) / "tool_cache.json"
        if not path.exists():
            return {}
        try:
            payload = C.read_json(path)
        except Exception:  # noqa: BLE001 - a corrupt cache means a static episode
            return {}
        return payload if isinstance(payload, dict) else {}

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        metadata = item["metadata"]
        result = metadata.get("result") or {}
        symptom = C.normalize_whitespace(metadata.get("query"))
        root_cause = C.normalize_whitespace(result.get("root_cause"))
        if not symptom or not root_cause:
            return None
        evidence = self._evidence(item["case_dir"])
        return SampleSpec(
            sample_id=C.stable_id("cloudops", item["case_id"]),
            fields={
                "observation": f"Reported symptom in namespace '{metadata.get('namespace')}': {symptom}",
                "context": (f"Diagnostic tool output:\n\n{evidence}" if evidence else ""),
                "question": "What is the root cause of this incident?",
                "instructions": (
                    "Name the root cause as a short technical label (for example "
                    "'missing_service_account'), and also name the affected object "
                    "(for example 'app/adservice')."
                ),
            },
            reference={
                "root_cause": root_cause,
                "fault_object": C.normalize_whitespace(result.get("fault_object")),
                "fault_taxonomy": C.normalize_whitespace(result.get("fault_taxonomy")),
            },
            task_kind="generation",
            # Reading tool output then naming a cause and an object.
            max_tokens=1024,
            metadata={
                "case_id": item["case_id"],
                "namespace": metadata.get("namespace"),
                "difficulty": metadata.get("difficulty"),
                "query": symptom,
                "evidence_included": bool(evidence),
                # The recorded cluster, for the interactive episode. Loaded
                # lazily: the caches are large and only the episodes that run
                # need them.
                "_tool_cache": self._tool_cache(item["case_dir"]),
            },
        )

    def score(
        self,
        sample: SampleSpec,
        response: ModelResponse,
        *,
        output_contract: dict[str, Any] | None = None,
    ) -> SampleScore:
        text = response.text
        if not text.strip():
            return unparsed_score(
                ["root_cause_judged", "fault_object_match", "taxonomy_match"], raw=text[:200]
            )
        answer = extract_answer_span(text, output_contract) or text
        reference = sample.reference
        # Underscored labels are matched loosely: a model may write
        # "missing service account" for "missing_service_account".
        object_variants = _variants(reference["fault_object"])
        taxonomy_variants = _variants(reference["fault_taxonomy"])
        metrics = {
            # Seeded at 0 and filled by the judge, along with the difficulty
            # stratum, which is the same verdict seen through a filter.
            "root_cause_judged": 0.0,
            # These two stay mechanical on purpose: the object is named in the
            # evidence the model was shown and the taxonomy is a small closed
            # vocabulary, so a string test asks a question with a right answer.
            "fault_object_match": _any_contains(text, object_variants),
            "taxonomy_match": _any_contains(text, taxonomy_variants),
        }
        difficulty = sample.metadata.get("difficulty")
        if difficulty:
            metrics[f"root_cause_judged_{difficulty}"] = 0.0
        return SampleScore(
            metrics=metrics,
            prediction=answer[:400],
            details={"gold_root_cause": reference["root_cause"]},
        )

    def judge_request(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore
    ) -> dict[str, Any] | None:
        if not response.text:
            return None
        return {
            "candidate": score.prediction or response.text[:800],
            "gold": sample.reference["root_cause"],
            "observation": C.clip_words(sample.fields.get("observation", ""), 150),
            "criteria": (
                "Root causes are written as short technical labels. The candidate is correct "
                "if it identifies the same underlying fault as the reference, in whatever "
                "words: 'missing_service_account', 'missing service account' and 'the "
                "deployment references a ServiceAccount that does not exist' are all the same "
                "root cause. Naming a downstream symptom of the fault, or a different fault "
                "that would produce a similar symptom, is not the same root cause."
            ),
        }

    def apply_judge(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore, verdict: Any
    ) -> SampleScore:
        return apply_judged_metric(score, verdict, "root_cause_judged")

    def aggregate(self, scores: Sequence[SampleScore]) -> dict[str, float]:
        metrics = aggregate_mean_metrics([score.metrics for score in scores])
        # A full diagnosis names both the cause and the object it applies to.
        pairs = [
            1.0
            if score.metrics.get("root_cause_judged") and score.metrics.get("fault_object_match")
            else 0.0
            for score in scores
            if "root_cause_judged" in score.metrics
        ]
        if pairs:
            metrics["full_diagnosis_rate"] = sum(pairs) / len(pairs)
        return metrics

    def documentation(self) -> AdapterDocumentation:
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="Cloud-OpsBench",
            domain="Computing Systems: Cloud RCA",
            source_url=REPO_URL,
            processing_mode="Generation",
            split_used=self.split_used,
            abductive_subset=(
                "Root-cause identification: infer the fault that explains an observed symptom. "
                "The benchmark's agentic tool-calling loop is replaced by a bounded slice of its "
                "own cached tool output, so the inference is evidence-grounded but single-turn."
            ),
            sampling_procedure=self.sampling_note(),
            metrics_description={
                "best_of_n_<metric>":
                "Every metric also gets a best_of_n_ counterpart: per record, the repeat the "
                "judge scored highest. The root cause is written free-form rather than chosen "
                "from a list, so a plurality over repeats is not meaningful and Best-of-N "
                "replaces it.",
                "root_cause_judged": "(PRIMARY, higher is better, 0-1) LLM-judge verdict on "
                "whether the stated root cause is the same underlying fault as the release's "
                "gold label, in whatever words. 1.0 when the judge affirms.",
                "root_cause_judged_<difficulty>": "the primary metric per difficulty label",
                "fault_object_match": "1 if the response names the affected object. Mechanical: "
                "the object is named in the evidence the model was shown, so its spelling is "
                "fixed and a string test is the right check.",
                "taxonomy_match": "1 if the response names the gold fault taxonomy class -- also "
                "mechanical, the taxonomy being a small closed vocabulary.",
                "full_diagnosis_rate": "fraction of cases where both the cause and the object are "
                "named -- the operationally useful outcome",
                "parse_failure_rate": "(lower is better, 0-1) fraction of responses no diagnosis "
                "could be read from; these score 0 and are counted separately from being wrong.",
            },
            primary_metric="root_cause_judged",
            decisions=[
                "THE PROTOCOL PROMPT IS LOCALLY ADAPTED, NOT THE AUTHORS'. The tool "
                "vocabulary, the Action/Action Input syntax, the one-primary-fault "
                "constraint, the evidence rule and the pivot rule come from the release's "
                "agent prompt and are reproduced in substance; the wording is this suite's, "
                "in its standard prompt layers. The release's Thought: line, its 'provide a "
                "clear reasoning chain' instruction and its 'limit your internal reasoning' "
                "clause are dropped, because this task runs io only and they would "
                "contradict the mode instruction in the same prompt.",
                "Included at most 6 cached tool outputs per case, each clipped to 220 words "
                "(configurable): the full tool cache is ~300 KB and the raw logs tens of MB, both "
                "far over the 16k-token input budget. How much was included is recorded per sample.",
                "Preferred resource listings and describe/event output over raw logs, because they "
                "state cluster state compactly.",
                "options.evidence = symptom_only gives the no-telemetry contrast condition.",
                "Matched the fault object and taxonomy loosely across underscore/space/hyphen "
                "variants, since those gold strings are machine labels rather than prose.",
                "Scored the root cause itself by an LLM judge rather than by string matching. "
                "No candidate list is shown, so the model writes the cause in its own words; "
                "the release's gold is one naming convention for the fault, not the only one."
            ],
            caveats=[
                "With a truncated evidence window a model may be unable to see the decisive "
                "signal; compare against symptom_only before concluding that a model cannot do RCA.",
                "Gold root causes are short machine labels and the judge is asked whether the "
                "candidate names the same underlying fault, so a correct diagnosis phrased in "
                "prose is credited. The cost is that the verdict is a model's, not a string "
                "test's: it can be wrong in both directions.",
            ],
            statistics={**self.base_statistics(), "cases_per_namespace": self._namespaces},
        )


def _variants(label: str) -> list[str]:
    """Surface variants of a machine label ('a_b' -> 'a b', 'a-b')."""
    if not label:
        return []
    base = label.strip()
    return list({base, base.replace("_", " "), base.replace("_", "-"), base.replace("-", " ")})


def _any_contains(text: str, variants: Sequence[str]) -> float:
    return float(any(contains_match(text, variant) for variant in variants if variant))
