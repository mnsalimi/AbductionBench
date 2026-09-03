"""HypoSpace: hypothesis-space exploration under underdetermination.

Source: https://github.com/CTT-Pavilion/_HypoSpace

The repository ships generators rather than files: ``causal/generate_causal_dataset.py``
enumerates, for every set of perturbation observations, **all** causal DAGs
compatible with them.  The generator is pure Python (numpy + networkx), takes a
``--seed`` and writes a JSON file, so the adapter runs it once and caches the
result -- the data is therefore reproducible from the release itself.

**Why this dataset is scored differently.** Its point is that observations
*underdetermine* the hypothesis: an item with 48 compatible graphs has 48 right
answers.  Scoring a single "gold" answer would misrepresent that, so the model
is asked for several distinct hypotheses and credited for how many are genuinely
compatible:

* ``validity`` -- share of proposed graphs that are in the compatible set;
* ``distinct_valid_rate`` -- distinct compatible graphs found, over how many
  could have been found (primary; this is hypothesis-space *coverage*);
* ``any_valid`` -- did the model produce at least one compatible hypothesis.
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Sequence
from typing import Any

from ..core.adapter import SkippedDataset
from ..core.metrics import aggregate_mean_metrics, extract_answer_span
from ..core.types import AdapterDocumentation, ModelResponse, SampleScore, SampleSpec
from . import _common as C
from ._base import PooledDatasetAdapter, unparsed_score

REPO_URL = "https://github.com/CTT-Pavilion/_HypoSpace"
_EDGE_RE = re.compile(r"\b([A-Z])\s*(?:->|→)\s*([A-Z])\b")


class HypoSpaceAdapter(PooledDatasetAdapter):
    """Propose several causal graphs compatible with perturbation observations."""

    adapter_version = "1.0"
    primary_metric = "distinct_valid_rate"

    def load_items(self) -> list[dict[str, Any]]:
        root = C.ensure_git_repo(
            REPO_URL, self.context.data_dir / "repo", offline=self.context.offline
        )
        generator = root / "causal" / "generate_causal_dataset.py"
        if not generator.exists():
            raise SkippedDataset(f"HypoSpace causal generator not found at {generator}")
        node_counts = [int(n) for n in C.as_list(self.context.option("node_counts", [3, 4]))]
        seed = int(self.context.option("generator_seed") or self.context.seed)
        items: list[dict[str, Any]] = []
        self._generated: dict[str, int] = {}
        for nodes in node_counts:
            output = (self.context.data_dir / f"causal_{nodes}nodes_seed{seed}.json").resolve()
            if not output.exists():
                if self.context.offline:
                    raise SkippedDataset(
                        f"offline mode: {output} has not been generated yet"
                    )
                self.log.info("running HypoSpace generator for %d nodes", nodes)
                try:
                    subprocess.run(  # noqa: S603 - the release's own generator
                        [
                            sys.executable,
                            str(generator.name),
                            "--nodes",
                            str(nodes),
                            "--seed",
                            str(seed),
                            "--output",
                            str(output),
                        ],
                        cwd=str(generator.parent),
                        check=True,
                        capture_output=True,
                        timeout=1800,
                    )
                except (subprocess.SubprocessError, OSError) as exc:
                    stderr = getattr(exc, "stderr", b"") or b""
                    raise SkippedDataset(
                        f"HypoSpace generator failed for {nodes} nodes: "
                        f"{stderr[-300:].decode('utf-8', 'replace') if stderr else exc}"
                    ) from exc
            payload = C.read_json(output)
            by_count = payload.get("datasets_by_n_observations") or {}
            count = 0
            for _count, entries in sorted(by_count.items(), key=lambda kv: int(kv[0])):
                for entry in entries:
                    items.append({**entry, "graph_nodes": nodes})
                    count += 1
            self._generated[f"{nodes}nodes"] = count
        if not items:
            raise SkippedDataset("HypoSpace generator produced no observation sets")
        self.split_used = (
            "generated from the release's own generator (deterministic, seed="
            f"{seed}): " + ", ".join(f"{k}: {v} observation sets" for k, v in self._generated.items())
        )
        return items

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        nodes = C.as_list(item.get("nodes"))
        observations = C.as_list(item.get("observations"))
        compatible = C.as_list(item.get("ground_truth_graphs"))
        if not nodes or not observations or not compatible:
            return None
        wanted = int(self.context.option("hypotheses_requested", 3))
        observation_lines = [
            f"- {C.normalize_whitespace(observation.get('string'))}"
            if observation.get("string")
            else f"- perturb {observation.get('perturbed_node')} -> "
            + ", ".join(f"{k}:{v}" for k, v in (observation.get("effects") or {}).items())
            for observation in observations
        ]
        compatible_sets = [
            frozenset(
                (str(edge[0]), str(edge[1]))
                for edge in C.as_list(graph.get("edges"))
                if isinstance(edge, (list, tuple)) and len(edge) >= 2
            )
            for graph in compatible
        ]
        return SampleSpec(
            sample_id=C.stable_id("hypospace", item.get("graph_nodes"), item.get("observation_set_id")),
            fields={
                "context": (
                    f"Variables: {', '.join(str(node) for node in nodes)}\n"
                    "Each observation perturbs one variable and reports which variables changed "
                    "(1 = changed, 0 = unchanged). Causal influence flows along directed edges."
                ),
                "observation": "Observations:\n" + "\n".join(observation_lines),
                "question": (
                    "Which causal graphs are consistent with these observations?"
                ),
                "instructions": (
                    f"Propose up to {wanted} DIFFERENT causal graphs that are each fully "
                    "consistent with every observation. Write one per line as "
                    "'Hypothesis k: A -> B, C -> D' (write 'Hypothesis k: none' for the empty "
                    "graph). Do not propose a graph you cannot justify."
                ),
            },
            reference={
                "compatible": [sorted(edges) for edges in compatible_sets],
                "n_compatible": int(item.get("n_compatible_graphs") or len(compatible_sets)),
                "nodes": [str(node) for node in nodes],
                "requested": wanted,
            },
            task_kind="generation",
            # Several hypotheses per answer, each a short edge list.
            max_tokens=self.clamp_max_tokens(512 + 256 * wanted, low=768, high=1536),
            metadata={
                "graph_nodes": item.get("graph_nodes"),
                "n_observations": item.get("n_observations"),
                "n_compatible_graphs": item.get("n_compatible_graphs"),
                "observation_set_id": item.get("observation_set_id"),
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
                ["distinct_valid_rate", "validity", "any_valid"], raw=text[:200]
            )
        proposed = _parse_hypotheses(text, sample.reference["nodes"])
        if not proposed:
            return unparsed_score(
                ["distinct_valid_rate", "validity", "any_valid"], raw=text[:300]
            )
        compatible = {frozenset(tuple(edge) for edge in edges) for edges in sample.reference["compatible"]}
        valid = [graph for graph in proposed if graph in compatible]
        distinct_valid = len(set(valid))
        achievable = min(int(sample.reference["requested"]), sample.reference["n_compatible"])
        metrics = {
            "validity": len(valid) / len(proposed),
            "distinct_valid_rate": distinct_valid / achievable if achievable else 0.0,
            "any_valid": float(bool(valid)),
            "n_proposed": float(len(proposed)),
            "duplicate_rate": 1.0 - (len(set(proposed)) / len(proposed)),
        }
        nodes = sample.metadata.get("graph_nodes")
        if nodes:
            metrics[f"distinct_valid_rate_{nodes}nodes"] = metrics["distinct_valid_rate"]
        return SampleScore(
            metrics=metrics,
            prediction="; ".join(
                ", ".join(f"{a}->{b}" for a, b in sorted(graph)) or "none" for graph in proposed
            )[:400],
            details={
                "n_compatible": sample.reference["n_compatible"],
                "n_valid_proposed": distinct_valid,
            },
        )

    def aggregate(self, scores: Sequence[SampleScore]) -> dict[str, float]:
        return aggregate_mean_metrics([score.metrics for score in scores])

    def documentation(self) -> AdapterDocumentation:
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="HypoSpace",
            domain="General: Hypothesis-Space Exploration",
            source_url=REPO_URL,
            processing_mode="Generation",
            split_used=self.split_used,
            abductive_subset=(
                "The causal task family. Observations underdetermine the graph, and the generator "
                "enumerates every compatible graph, which is what makes this a hypothesis-space "
                "task rather than a single-answer one. The repository's boolean and 3d families "
                "use the same idea and could be added the same way; only causal is enabled here."
            ),
            sampling_procedure=(
                "the release's generator is run once per configured node count with a fixed seed "
                "(deterministic), then " + self.sampling_note()
            ),
            metrics_description={
                "distinct_valid_rate": "distinct compatible graphs proposed, divided by how many "
                "could have been proposed (min(requested, number compatible)) -- primary; this is "
                "hypothesis-space coverage",
                "validity": "share of proposed graphs that are compatible with all observations",
                "any_valid": "1 if at least one proposed graph is compatible",
                "duplicate_rate": "share of proposed graphs that repeat an earlier one -- a model "
                "that cannot find genuinely different hypotheses shows up here",
                "n_proposed": "how many hypotheses the model actually produced",
                "distinct_valid_rate_<n>nodes": "the primary metric per graph size",
            },
            primary_metric="distinct_valid_rate",
            decisions=[
                "The repository publishes generators, not data files; the causal generator is "
                "pure Python and seeded, so the adapter runs it once and caches the JSON -- the "
                "items are reproducible from the release rather than invented.",
                "Node counts 3 and 4 by default (options.node_counts); larger counts make the "
                "enumeration explode.",
                "Asked for 3 hypotheses (options.hypotheses_requested) and scored coverage rather "
                "than single-answer accuracy, because the observations genuinely admit many "
                "graphs -- often dozens.",
                "Credited only graphs that are exactly in the enumerated compatible set, so "
                "validity is decided by the generator, not by our own reasoning.",
            ],
            caveats=[
                "distinct_valid_rate depends on how many hypotheses were requested; comparisons "
                "are only valid at a fixed hypotheses_requested.",
                "Parsing relies on the requested 'A -> B' format; a model that ignores the format "
                "is recorded as a parse failure rather than as wrong.",
            ],
            statistics={
                **self.base_statistics(),
                "observation_sets_generated": self._generated,
                "hypotheses_requested": int(self.context.option("hypotheses_requested", 3)),
            },
        )


def _parse_hypotheses(text: str, nodes: Sequence[str]) -> list[frozenset[tuple[str, str]]]:
    """Parse 'Hypothesis k: A -> B, C -> D' lines into edge sets."""
    allowed = {str(node) for node in nodes}
    graphs: list[frozenset[tuple[str, str]]] = []
    body = extract_answer_span(text, None) or text
    for line in body.splitlines():
        stripped = line.strip()
        if not re.match(r"^\**\s*(hypothesis|graph|option)\b", stripped, re.IGNORECASE):
            continue
        if re.search(r":\s*none\b", stripped, re.IGNORECASE):
            graphs.append(frozenset())
            continue
        edges = {
            (source, target)
            for source, target in _EDGE_RE.findall(stripped)
            if source in allowed and target in allowed
        }
        if edges:
            graphs.append(frozenset(edges))
    return graphs
