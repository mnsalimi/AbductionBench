"""AIOps2025 / RCA100: incident root-cause identification.

Source: https://www.aiops.cn/gitlab/aiops-live-benchmark/agenticopseval

The repository ships two collections:

* ``RCA100/`` -- 103 incident cases, each with ``task.json`` (the firing alert,
  its window and the observability modalities available) and ``topology.json``
  (the service/entity graph for that window), plus an ``answer_key`` naming the
  ``root_cause_entities``.  This is the abductive item set used here.
* ``AIOps2025/groundtruth.jsonl`` -- fault records for the live competition,
  without the per-case observability context needed to reason about them, so it
  is not used as the item source (it is, however, the source of the fault-type
  vocabulary reported in the statistics).

The task: from the alert and the service topology, name the entity whose failure
explains the alert.  The benchmark's own agentic setting also allows querying
metrics/logs/traces; those queries are not reproducible offline, so the case's
own alert text and topology are the evidence.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..core.adapter import SkippedDataset
from ..core.metrics import aggregate_mean_metrics, contains_match, extract_answer_span
from ..core.types import AdapterDocumentation, ModelResponse, SampleScore, SampleSpec
from . import _common as C
from ._base import PooledDatasetAdapter, unparsed_score

REPO_URL = "https://www.aiops.cn/gitlab/aiops-live-benchmark/agenticopseval"


class AIOps2025Adapter(PooledDatasetAdapter):
    """Name the root-cause entity of an incident from its alert and topology."""

    adapter_version = "1.0"

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. You are given the alerts, metrics "
        "and service topology of a microservice incident. Name the entity that is the root "
        "cause: the component whose failure explains the whole alert pattern, not every "
        "component that reported an anomaly downstream of it."
    )
    data_delivery_mode = "static"

    answer_format = "the root cause"
    answer_constraints = (
        "write exactly one sentence",
        "name only the root cause",
        "do not use introductory phrases or commentary",
    )
    options_heading = "Candidate root causes:"
    objective_metrics = True
    selection_cardinality = "single"
    primary_metric = "root_cause_match"

    def load_items(self) -> list[dict[str, Any]]:
        root = C.ensure_git_repo(
            REPO_URL, self.context.data_dir / "repo", offline=self.context.offline
        )
        cases_dir = root / "RCA100" / "cases"
        answers_dir = root / "RCA100" / "answer_key"
        if not cases_dir.exists() or not answers_dir.exists():
            raise SkippedDataset("RCA100 cases/answer_key not found in the repository")
        items: list[dict[str, Any]] = []
        for case_dir in sorted(p for p in cases_dir.iterdir() if p.is_dir()):
            task_path = case_dir / "task.json"
            answer_path = answers_dir / f"{case_dir.name}.gt.json"
            if not task_path.exists() or not answer_path.exists():
                continue
            answer = C.read_json(answer_path)
            entities = [
                C.normalize_whitespace(entity)
                for entity in C.as_list(answer.get("root_cause_entities"))
                if C.normalize_whitespace(entity)
            ]
            if not entities:
                continue
            items.append(
                {
                    "case_id": case_dir.name,
                    "task": C.read_json(task_path),
                    "topology_path": case_dir / "topology.json",
                    "root_cause_entities": entities,
                }
            )
        if not items:
            raise SkippedDataset("no RCA100 case had a usable answer key")
        self.split_used = (
            f"RCA100 ({len(items)} cases with answer keys); the collection is a single evaluation "
            "set and is smaller than the 300-sample target"
        )
        return items

    def _topology_text(self, path) -> tuple[str, list[str]]:
        if not path.exists():
            return "", []
        try:
            payload = C.read_json(path)
        except Exception:  # noqa: BLE001
            return "", []
        entities = payload.get("entities") or []
        edges = payload.get("edges") or []
        max_entities = int(self.context.option("max_entities", 40))
        max_edges = int(self.context.option("max_edges", 60))
        names: list[str] = []
        lines: list[str] = []
        for entity in entities[:max_entities]:
            name = C.normalize_whitespace(entity.get("name") or entity.get("entity_name"))
            kind = C.normalize_whitespace(entity.get("type") or entity.get("entity_type"))
            if name:
                names.append(name)
                lines.append(f"- {name}" + (f" ({kind})" if kind else ""))
        edge_lines = []
        for edge in edges[:max_edges]:
            source = C.normalize_whitespace(edge.get("source") or edge.get("from"))
            target = C.normalize_whitespace(edge.get("target") or edge.get("to"))
            if source and target:
                edge_lines.append(f"- {source} -> {target}")
        blocks = []
        if lines:
            blocks.append("Entities in the incident window:\n" + "\n".join(lines))
        if edge_lines:
            blocks.append("Call/dependency edges:\n" + "\n".join(edge_lines))
        return "\n\n".join(blocks), names

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        task = item["task"]
        alert_title = C.normalize_whitespace(task.get("alert_title"))
        alert_entity = (task.get("alert_entity") or {}).get("entity_name")
        window = task.get("alert_window") or {}
        prompt_text = C.clip_words(
            C.normalize_whitespace(task.get("prompt_text")),
            int(self.context.option("alert_words", 400)),
        )
        topology_text, entity_names = self._topology_text(item["topology_path"])
        if not alert_title and not prompt_text:
            return None
        observation = "\n".join(
            part
            for part in (
                f"Firing alert: {alert_title}" if alert_title else "",
                f"Alerting entity: {alert_entity}" if alert_entity else "",
                f"Window: {window.get('start')} to {window.get('end')}" if window else "",
                f"Alert detail:\n{prompt_text}" if prompt_text else "",
            )
            if part
        )
        return SampleSpec(
            sample_id=C.stable_id("aiops", item["case_id"]),
            fields={
                "observation": observation,
                "context": topology_text,
                "question": (
                    "Which entity's failure is the root cause of this alert?"
                ),
                "instructions": (
                    "Name the single root-cause entity (a service or component name) exactly as it "
                    "appears in the topology. The alerting entity is often a downstream victim, "
                    "not the cause."
                ),
            },
            reference={"entities": item["root_cause_entities"]},
            task_kind="generation",
            # Reading an alert plus a topology, then naming one entity.
            max_tokens=1024,
            metadata={
                "case_id": item["case_id"],
                "modalities": C.as_list(task.get("available_modalities")),
                "n_entities": len(entity_names),
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
            return unparsed_score(["root_cause_match"], raw=text[:200])
        gold = sample.reference["entities"]
        answer = extract_answer_span(text, output_contract) or text
        # The answer line is checked first (a strict reading), then the whole
        # response (lenient), and both are reported.
        strict = float(any(contains_match(answer, entity) for entity in gold))
        lenient = float(any(contains_match(text, entity) for entity in gold))
        return SampleScore(
            metrics={
                "root_cause_match": strict,
                "root_cause_mentioned": lenient,
            },
            prediction=answer[:300],
            details={"gold": ", ".join(gold)},
        )

    def aggregate(self, scores: Sequence[SampleScore]) -> dict[str, float]:
        return aggregate_mean_metrics([score.metrics for score in scores])

    def documentation(self) -> AdapterDocumentation:
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="AIOps2025 / RCA100",
            domain="Computing Systems: Incident Response",
            source_url=REPO_URL,
            processing_mode="Generation",
            split_used=self.split_used,
            abductive_subset=(
                "RCA100's incident cases: infer which entity's failure explains an alert. The "
                "AIOps2025 live-competition groundtruth file is not used as the item source "
                "because it ships fault records without the per-case observability context "
                "needed to reason about them."
            ),
            sampling_procedure=self.sampling_note()
            + "; only 103 cases exist, so the draw covers all of them",
            metrics_description={
                "self_consistency_<metric>":
                "Every metric also gets a self_consistency_ counterpart: the plurality answer over "
                "modes.repeats samples of the same record, read off those samples rather than bought "
                "again. Available because this dataset's answers are checkable and so can coincide.",
                "root_cause_match": "(PRIMARY, higher is better) 1 if the answer line names a gold root-cause entity (primary)",
                "root_cause_mentioned": "1 if any part of the response names it -- the difference "
                "from the primary metric shows how often the right entity was considered but not "
                "committed to",
            },
            primary_metric="root_cause_match",
            decisions=[
                "Used the alert text and the case topology as evidence; the benchmark's live "
                "metric/log/trace queries are not reproducible offline, and this limitation is "
                "recorded rather than worked around.",
                "Clipped the alert detail to 400 words and the topology to 40 entities / 60 edges "
                "(configurable) to stay inside the input budget.",
                "Reported both a strict (answer-line) and a lenient (whole-response) match, since "
                "these responses tend to discuss several candidate services.",
                "The repository notes that the official output contract and fault taxonomy will "
                "be published later, so no taxonomy-based scoring was attempted.",
            ],
            caveats=[
                "Only 103 cases exist, so this dataset reports fewer than the 300-sample target.",
                "Alert titles are partly in Chinese; the task is therefore mildly multilingual.",
                "Without live telemetry a model must reason from the alert and topology alone, so "
                "scores are a lower bound on agentic RCA ability.",
            ],
            statistics=self.base_statistics(),
        )
