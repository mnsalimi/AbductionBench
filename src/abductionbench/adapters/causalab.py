"""CausaLab: causal-structure discovery from observational data.

Source: https://github.com/DylanZSZ/CausaLab-Benchmark

Each released graph (``release/causalab_dataset/data/*.jsonl``) describes a set
of variables with human-readable names, the true ``edges``, and
``bootstrap_past_data``: observations of every variable across many runs.  The
benchmark's own protocol lets an agent spend an intervention ``budget``; this
adapter uses the **static bootstrap observations**, which is the single-turn
form of the same abduction -- infer the causal structure that explains the
observed co-variation.

The graph file names encode the variant (node count, hidden nodes, frequent
parents, quadratic hardness), and accuracy is reported per variant.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from ..core.adapter import SkippedDataset
from ..core.metrics import aggregate_mean_metrics, extract_answer_span, set_prf
from ..core.types import AdapterDocumentation, ModelResponse, SampleScore, SampleSpec
from . import _common as C
from ._base import PooledDatasetAdapter, unparsed_score

REPO_URL = "https://github.com/DylanZSZ/CausaLab-Benchmark"
_EDGE_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_ ]*?)\s*(?:->|→|causes|to)\s*([A-Za-z_][A-Za-z0-9_ ]*)")


class CausaLabAdapter(PooledDatasetAdapter):
    """Recover the causal edge set that explains observational data."""

    adapter_version = "1.0"
    primary_metric = "edge_f1"

    def load_items(self) -> list[dict[str, Any]]:
        root = C.ensure_git_repo(
            REPO_URL, self.context.data_dir / "repo", offline=self.context.offline
        )
        data_dir = root / "release" / "causalab_dataset" / "data"
        if not data_dir.exists():
            data_dir = root / "causal_graph_configs"
        files = C.find_files(data_dir, ["*.jsonl"])
        if not files:
            raise SkippedDataset("no CausaLab graph files found")
        variants = self.context.option("variants")
        items: list[dict[str, Any]] = []
        self._per_variant: dict[str, int] = {}
        for path in files:
            variant = path.stem
            if variants and variant not in variants:
                continue
            rows = C.read_jsonl(path)
            # Only graphs that ship bootstrap observations can be evaluated
            # single-turn; the rest would leave nothing to abduce from.
            usable = [row for row in rows if row.get("bootstrap_past_data")]
            for row in usable:
                items.append({**row, "variant": variant})
            self._per_variant[variant] = len(usable)
            self._graphs_seen = getattr(self, "_graphs_seen", 0) + len(rows)
        if not items:
            raise SkippedDataset("CausaLab graph files contained no graphs")
        self.split_used = (
            f"released graph configurations that include bootstrap observations: {len(items)} of "
            f"{getattr(self, '_graphs_seen', len(items))} graphs across "
            f"{len([v for v in self._per_variant.values() if v])} variants; the release ships no "
            "train/test split"
        )
        return items

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        nodes = item.get("nodes") or {}
        edges = item.get("edges") or []
        observations = item.get("bootstrap_past_data") or []
        if not nodes or not edges or not observations:
            return None
        limit = int(self.context.option("max_observations", 20))
        variable_lines = []
        for name, info in nodes.items():
            display = (info or {}).get("display_name") or name
            controllable = (info or {}).get("is_controllable")
            variable_lines.append(
                f"- {name} ({display})" + (" [controllable]" if controllable else "")
            )
        observation_lines = []
        for record in observations[:limit]:
            props = record.get("props") or {}
            extras = {k: v for k, v in record.items() if k not in ("id", "props")}
            values = ", ".join(f"{key}={value}" for key, value in props.items())
            extra_text = ", ".join(f"{key}={value}" for key, value in extras.items())
            observation_lines.append(f"- {values}" + (f", {extra_text}" if extra_text else ""))
        gold_edges = {
            f"{edge.get('from')}->{edge.get('to')}"
            for edge in edges
            if edge.get("from") and edge.get("to")
        }
        if not gold_edges:
            return None
        return SampleSpec(
            sample_id=C.stable_id("causalab", item.get("variant"), item.get("graph_id", index)),
            fields={
                "context": "Variables:\n" + "\n".join(variable_lines),
                "observation": (
                    f"{len(observation_lines)} observations of these variables:\n"
                    + "\n".join(observation_lines)
                ),
                "question": (
                    "Which direct causal relationships between these variables best explain the "
                    "observations?"
                ),
                "instructions": (
                    "List every direct causal edge, one per line, in the form 'source -> target', "
                    "using the exact variable names given. List only edges you can justify."
                ),
            },
            reference={"edges": sorted(gold_edges), "n_nodes": len(nodes)},
            task_kind="generation",
            # Reasoning over a table of observations, then a short edge list.
            max_tokens=self.clamp_max_tokens(512 + 128 * len(nodes), low=640, high=1536),
            metadata={
                "variant": item.get("variant"),
                "graph_id": item.get("graph_id"),
                "n_nodes": len(nodes),
                "n_edges": len(gold_edges),
            },
        )

    def score(
        self,
        sample: SampleSpec,
        response: ModelResponse,
        *,
        output_contract: dict[str, Any] | None = None,
    ) -> SampleScore:
        text = extract_answer_span(response.text, output_contract) or response.text
        if not text:
            return unparsed_score(
                ["edge_f1", "edge_precision", "edge_recall", "exact_graph_match"],
                raw=response.text[:300],
            )
        predicted = {
            f"{source.strip()}->{target.strip()}"
            for source, target in _EDGE_RE.findall(text)
        }
        if not predicted:
            return unparsed_score(
                ["edge_f1", "edge_precision", "edge_recall", "exact_graph_match"],
                raw=response.text[:300],
            )
        gold = set(sample.reference["edges"])
        prf = set_prf(predicted, gold)
        variant = sample.metadata.get("variant", "unknown")
        return SampleScore(
            metrics={
                "edge_f1": prf["f1"],
                "edge_precision": prf["precision"],
                "edge_recall": prf["recall"],
                "exact_graph_match": float(predicted == gold),
                f"edge_f1_{variant}": prf["f1"],
            },
            prediction=", ".join(sorted(predicted))[:400],
            details={"gold": ", ".join(sorted(gold))[:300]},
        )

    def aggregate(self, scores: Sequence[SampleScore]) -> dict[str, float]:
        return aggregate_mean_metrics([score.metrics for score in scores])

    def documentation(self) -> AdapterDocumentation:
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="CausaLab",
            domain="Causal Science: Causal Discovery",
            source_url=REPO_URL,
            processing_mode="Generation",
            split_used=self.split_used,
            abductive_subset=(
                "Recovering the causal graph that explains observed co-variation. CausaLab's "
                "interactive intervention budget is not reproducible single-turn, so the released "
                "bootstrap observations are the evidence and the true edge set is the reference."
            ),
            sampling_procedure=self.sampling_note(),
            metrics_description={
                "edge_f1": "F1 between predicted and true directed edge sets (primary)",
                "edge_precision": "precision of the predicted edges -- penalizes over-claiming",
                "edge_recall": "recall of the true edges",
                "exact_graph_match": "1 only if the predicted edge set is exactly the true one",
                "edge_f1_<variant>": "edge F1 per released graph variant (node count, hidden "
                "nodes, frequent parents, ...)",
            },
            primary_metric="edge_f1",
            decisions=[
                "Used bootstrap_past_data as the observations (capped at 20 records by default, "
                "options.max_observations) so prompts stay inside the input budget while still "
                "showing real co-variation.",
                "Scored as a set task with F1 plus precision/recall, because a model that lists "
                "every possible edge would score well on recall alone.",
                "Parsed edges with a permissive 'A -> B' / 'A causes B' pattern so formatting "
                "differences do not count as errors.",
                "max_tokens scales with node count (512 + 128 per node).",
            ],
            caveats=[
                "Only the graph variants that ship bootstrap observations are usable; the "
                "remaining released configurations are intervention-only and are reported as "
                "excluded in the statistics.",
                "With only 20 observations, some edges are genuinely underdetermined; edge_f1 is "
                "therefore a measure of plausible-structure inference, not of asymptotic "
                "identifiability.",
                "Interventional data (the benchmark's own protocol) is not used, so scores are not "
                "comparable to published CausaLab results.",
            ],
            statistics={
                **self.base_statistics(),
                "graphs_per_variant": {k: v for k, v in self._per_variant.items() if v},
                "graphs_total_seen": getattr(self, "_graphs_seen", 0),
                "graphs_without_observations": getattr(self, "_graphs_seen", 0) - self.split_size,
            },
        )
