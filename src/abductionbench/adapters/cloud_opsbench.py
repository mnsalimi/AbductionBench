"""Cloud-OpsBench: root-cause analysis for Kubernetes incidents.

Source: https://github.com/LLM4Ops/Cloud-OpsBench

Each case directory holds ``metadata.json`` (the reported symptom in ``query``,
a ``difficulty`` label, and the ground truth: ``fault_taxonomy``,
``fault_object``, ``root_cause``) plus ``tool_cache.json`` -- cached outputs of
the diagnostic tools an agent would have called (pod listings, describes, logs,
events).

**Evidence budget.** ``tool_cache.json`` runs to hundreds of thousands of
characters per case (and the raw log dumps to tens of megabytes), far beyond the
16,000-token input budget.  The adapter therefore includes a bounded slice of
the cached evidence -- the resource listings and describe/event outputs first,
each clipped to a word budget -- and reports how much was included.  With
``options.evidence = symptom_only`` the task becomes symptom-to-root-cause
abduction with no telemetry, which is a useful contrast.
"""

from __future__ import annotations

from typing import Any, Sequence

from ..core.adapter import SkippedDataset
from ..core.metrics import aggregate_mean_metrics, contains_match, extract_answer_span, token_f1
from ..core.types import AdapterDocumentation, ModelResponse, SampleScore, SampleSpec
from . import _common as C
from ._base import PooledDatasetAdapter, unparsed_score

REPO_URL = "https://github.com/LLM4Ops/Cloud-OpsBench"
#: Tool-cache keys are of the form ``ToolName:{json args}``; these come first
#: because they describe cluster state compactly.
_PREFERRED_TOOLS = ("GetResources", "DescribeResource", "GetEvents", "GetLogs")


class CloudOpsBenchAdapter(PooledDatasetAdapter):
    """Name the root cause of a Kubernetes incident from symptom + evidence."""

    adapter_version = "1.0"
    primary_metric = "root_cause_match"

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
                "evidence_included": bool(evidence),
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
                ["root_cause_match", "fault_object_match", "taxonomy_match"], raw=text[:200]
            )
        answer = extract_answer_span(text, output_contract) or text
        reference = sample.reference
        # Underscored labels are matched loosely: a model may write
        # "missing service account" for "missing_service_account".
        root_cause_variants = _variants(reference["root_cause"])
        object_variants = _variants(reference["fault_object"])
        taxonomy_variants = _variants(reference["fault_taxonomy"])
        metrics = {
            "root_cause_match": _any_contains(text, root_cause_variants),
            "fault_object_match": _any_contains(text, object_variants),
            "taxonomy_match": _any_contains(text, taxonomy_variants),
            "root_cause_token_f1": token_f1(answer, reference["root_cause"]),
        }
        difficulty = sample.metadata.get("difficulty")
        if difficulty:
            metrics[f"root_cause_match_{difficulty}"] = metrics["root_cause_match"]
        return SampleScore(
            metrics=metrics,
            prediction=answer[:400],
            details={"gold_root_cause": reference["root_cause"]},
        )

    def aggregate(self, scores: Sequence[SampleScore]) -> dict[str, float]:
        metrics = aggregate_mean_metrics([score.metrics for score in scores])
        # A full diagnosis names both the cause and the object it applies to.
        pairs = [
            1.0
            if score.metrics.get("root_cause_match") and score.metrics.get("fault_object_match")
            else 0.0
            for score in scores
            if "root_cause_match" in score.metrics
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
                "root_cause_match": "1 if the response names the gold root cause (underscore/space "
                "variants accepted) -- primary",
                "fault_object_match": "1 if the response names the affected object",
                "taxonomy_match": "1 if the response names the gold fault taxonomy class",
                "root_cause_token_f1": "token F1 between the answer span and the gold root cause",
                "full_diagnosis_rate": "fraction of cases where both the cause and the object are "
                "named -- the operationally useful outcome",
                "root_cause_match_<difficulty>": "the primary metric per difficulty label",
            },
            primary_metric="root_cause_match",
            decisions=[
                "Included at most 6 cached tool outputs per case, each clipped to 220 words "
                "(configurable): the full tool cache is ~300 KB and the raw logs tens of MB, both "
                "far over the 16k-token input budget. How much was included is recorded per sample.",
                "Preferred resource listings and describe/event output over raw logs, because they "
                "state cluster state compactly.",
                "options.evidence = symptom_only gives the no-telemetry contrast condition.",
                "Matched the gold labels loosely across underscore/space/hyphen variants, since "
                "the gold strings are machine labels rather than prose.",
            ],
            caveats=[
                "With a truncated evidence window a model may be unable to see the decisive "
                "signal; compare against symptom_only before concluding that a model cannot do RCA.",
                "Gold root causes are short machine labels, so a correct diagnosis phrased "
                "differently can be scored as a miss; root_cause_token_f1 gives partial credit.",
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
