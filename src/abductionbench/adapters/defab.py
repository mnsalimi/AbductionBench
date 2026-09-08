"""DeFAb: defeasible abduction over ontology-derived theories.

Source: https://huggingface.co/datasets/PatrickAllenCooper/DeFAb

Each instance gives a ``target`` (an observation that does not follow from the
theory), a list of ``candidates`` (facts and defeasible rules) and the ``gold``
subset of candidates whose addition would explain the target.  Instances come
from several ontologies (BabelNet, FrameNet, Gene Ontology, SUMO, UMLS,
Wikidata, YAGO) plus a synthetic tier, and carry a ``level`` (inference depth)
and the type of the ablated element.

**Selection, not generation.** The suite's table lists DeFAb as *Generation*,
but the release ships explicit candidate sets with a gold subset, and the gold
strings are internal rule identifiers plus formulas (e.g.
``df_is_gene(rwdd3):  => is_gene(rwdd3)``).  Asking a model to reproduce those
identifiers verbatim would measure format mimicry, not abduction, so the task is
rendered as selecting the explanatory candidate(s) -- which is exactly how the
release is structured.  ``options.subtask = generation`` is available for anyone
who wants the free-text variant, and the deviation is recorded here.

Instances whose gold set has more than one element become ``multi_selection``
samples scored with set F1; single-gold instances are ordinary selection.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from ..core.adapter import SkippedDataset
from ..core.metrics import (
    aggregate_mean_metrics,
    extract_answer_span,
    set_prf,
)
from ..core.types import AdapterDocumentation, ModelResponse, SampleScore, SampleSpec
from . import _common as C
from ._base import PooledDatasetAdapter, selection_score, text_match_score, unparsed_score

REPO_ID = "PatrickAllenCooper/DeFAb"


class DeFAbAdapter(PooledDatasetAdapter):
    """Pick the candidate rule/fact whose addition explains the target."""

    adapter_version = "1.0"

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. You are given a defeasible "
        "theory, a target conclusion it does not currently support, and candidate additions. "
        "Select every candidate whose addition would make the target derivable while "
        "respecting the theory's defeaters. More than one addition can be required."
    )
    data_delivery_mode = "static"

    options_heading = "Candidate explanations:"
    objective_metrics = True
    selection_cardinality = "multi"
    primary_metric = "accuracy"

    @property
    def _subtask(self) -> str:
        return str(self.context.option("subtask", "selection"))

    def load_items(self) -> list[dict[str, Any]]:
        root = C.ensure_hf_snapshot(
            REPO_ID,
            self.context.data_dir / "hf",
            offline=self.context.offline,
            allow_patterns=["instances/**", "*.md", "MANIFEST.txt"],
        )
        files = [
            path
            for path in C.find_files(root / "instances", ["*.json"])
            if "instances" in path.name or "instances" in path.parent.name
        ]
        if not files:
            raise SkippedDataset("no DeFAb instance files found in the release")
        wanted = self.context.option("domains")
        items: list[dict[str, Any]] = []
        self._per_source: dict[str, int] = {}
        for path in files:
            payload = C.read_json(path)
            rows = payload if isinstance(payload, list) else []
            source = path.parent.name if path.name == "instances.json" else path.stem
            if wanted and source not in wanted:
                continue
            for row in rows:
                if isinstance(row, dict):
                    items.append({**row, "source": source})
            self._per_source[source] = len(rows)
        if not items:
            raise SkippedDataset("DeFAb instance files contained no usable rows")
        self.split_used = (
            "all released instance sets pooled: "
            + ", ".join(f"{name} ({count})" for name, count in sorted(self._per_source.items()))
            + " -- the release ships no train/test split"
        )
        return items

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        target = C.normalize_whitespace(item.get("target"))
        candidates = [C.normalize_whitespace(text) for text in C.as_list(item.get("candidates"))]
        candidates = [text for text in candidates if text]
        gold = [C.normalize_whitespace(text) for text in C.as_list(item.get("gold"))]
        gold = [text for text in gold if text]
        if not target or len(candidates) < 2 or not gold:
            return None
        if not set(gold) <= set(candidates):
            return None  # cannot align gold to candidates; skip rather than guess

        level = item.get("level")
        source = item.get("source", "unknown")
        sample_id = C.stable_id("defab", source, item.get("target", index), index)

        if self._subtask == "generation":
            return SampleSpec(
                sample_id=sample_id,
                fields={
                    "observation": target,
                    "instructions": (
                        "State the fact or defeasible rule that, if added to the theory, would "
                        "explain the observation. Use the same notation as the observation."
                    ),
                },
                reference={"gold": gold[0], "gold_set": gold},
                task_kind="knowledge_completion",
                max_tokens=384,
                metadata={"source": source, "level": level, "n_candidates": len(candidates)},
            )

        ordered = sorted(set(candidates))
        labels = C.choice_labels(len(ordered))
        gold_labels = [labels[ordered.index(text)] for text in gold if text in ordered]
        if not gold_labels:
            return None
        multi = len(gold_labels) > 1
        return SampleSpec(
            sample_id=sample_id,
            fields={
                "observation": target,
                "question": (
                    "Which candidate(s), if added to the theory, would explain the observation?"
                    if multi
                    else "Which candidate, if added to the theory, would explain the observation?"
                ),
                "options": ordered,
                "option_labels": labels,
                "instructions": (
                    "Candidates are facts and defeasible rules in logical notation; a defeasible "
                    "rule may be overridden by a more specific one."
                ),
            },
            reference={"gold_labels": gold_labels, "gold_label": gold_labels[0]},
            # Mixed task kinds inside one dataset: the engine binds each to its
            # own template.
            task_kind="multi_selection" if multi else "selection",
            # Formal candidates over theories of up to a few hundred axioms; the
            # answer is a label, the budget is deliberation room.
            max_tokens=self.clamp_max_tokens(384 + 128 * int(level or 1), low=384, high=1024),
            metadata={
                "source": source,
                "level": level,
                "n_candidates": len(ordered),
                "n_gold": len(gold_labels),
                "ablated_type": (item.get("metadata") or {}).get("ablated_type"),
            },
        )

    def score(
        self,
        sample: SampleSpec,
        response: ModelResponse,
        *,
        output_contract: dict[str, Any] | None = None,
    ) -> SampleScore:
        if sample.task_kind == "knowledge_completion":
            return text_match_score(
                response,
                gold=sample.reference["gold"],
                accepted=sample.reference.get("gold_set", [])[1:],
                output_contract=output_contract,
                primary="match",
            )
        labels = sample.fields["option_labels"]
        if sample.task_kind == "selection":
            score = selection_score(
                response,
                labels=labels,
                gold_label=sample.reference["gold_label"],
                output_contract=output_contract,
                metric_name="accuracy",
            )
        else:
            answer = extract_answer_span(response.text, output_contract)
            found = {
                token.upper()
                for token in re.findall(r"[A-Za-z]+", answer or "")
                if token.upper() in labels
            }
            if not found:
                score = unparsed_score(["accuracy", "set_f1"], raw=response.text[:300])
            else:
                gold = set(sample.reference["gold_labels"])
                prf = set_prf(found, gold)
                score = SampleScore(
                    metrics={"set_f1": prf["f1"], "accuracy": float(found == gold)},
                    prediction=",".join(sorted(found)),
                    details={"gold": ",".join(sorted(gold))},
                )
        level = sample.metadata.get("level")
        if level is not None and "accuracy" in score.metrics:
            score.metrics[f"accuracy_level_{level}"] = score.metrics["accuracy"]
        source = sample.metadata.get("source")
        if source and "accuracy" in score.metrics:
            score.metrics[f"accuracy_{source}"] = score.metrics["accuracy"]
        return score

    def aggregate(self, scores: Sequence[SampleScore]) -> dict[str, float]:
        return aggregate_mean_metrics([score.metrics for score in scores])

    def documentation(self) -> AdapterDocumentation:
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="DeFAb",
            domain="Formal Reasoning: Defeasible Abduction",
            source_url=f"https://huggingface.co/datasets/{REPO_ID}",
            processing_mode="Selection (release structure) -- generation available by option",
            split_used=self.split_used,
            abductive_subset=(
                "Every instance is a defeasible-abduction problem. All released instance sets "
                "(seven ontologies plus the synthetic and tier0 sets) are pooled; "
                "options.domains can restrict them. The repository's evaluation/ and rules/ "
                "directories hold baseline results and source theories, not evaluation items, "
                "and are ignored."
            ),
            sampling_procedure=self.sampling_note(),
            metrics_description={
                "self_consistency_<metric>":
                "Every metric also gets a self_consistency_ counterpart: the plurality answer over "
                "modes.repeats samples of the same record, read off those samples rather than bought "
                "again. Available because this dataset's answers are checkable and so can coincide.",
                "accuracy": "(PRIMARY, higher is better) 1 if the selected candidate is the gold one (for multi-gold items, "
                "1 only if the predicted set equals the gold set) -- primary",
                "set_f1": "F1 between predicted and gold label sets, on multi-gold items",
                "accuracy_level_<n>": "accuracy at inference depth n",
                "accuracy_<source>": "accuracy on one ontology/instance set",
                "match": "generation subtask: gold rule equals or is contained in the answer",
            },
            primary_metric="accuracy",
            decisions=[
                "Rendered as selection rather than the table's 'Generation', because the release "
                "ships candidate sets with a gold subset and the gold strings are internal rule "
                "identifiers -- reproducing those verbatim would measure format mimicry. "
                "options.subtask = generation gives the free-text variant.",
                "Instances with several gold candidates are treated as multi-answer items and "
                "scored with set F1 (accuracy then requires an exact set match).",
                "Instances whose gold strings are not a subset of their candidate list are "
                "skipped rather than aligned by guesswork.",
                "max_tokens scales with the instance's own `level` (depth).",
            ],
            caveats=[
                "Candidate lists are short (4-6), so chance accuracy is high; read scores against "
                "a 1/n baseline.",
                "The full theories are not shown -- only the target and the candidates -- because "
                "theories run to hundreds of axioms and would exceed the input budget. The task "
                "is therefore 'which candidate is the plausible explainer', not full theory "
                "reasoning.",
            ],
            statistics={
                **self.base_statistics(),
                "instances_per_source": self._per_source,
                "subtask": self._subtask,
            },
        )
