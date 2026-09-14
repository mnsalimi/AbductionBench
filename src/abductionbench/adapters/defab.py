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

**Single-answer, not multi-answer.**  This adapter declared
``selection_cardinality = "multi"`` until 2026-09-14, which is wrong on both the
paper's account and the data's.  The paper (arXiv:2606.18557, §3) says the
pipeline ablates one critical element and injects "k = 5 syntactically similar
distractors", so an instance is one gold among six candidates, and states
outright: *"Both levels have a unique gold hypothesis (|H*| = 1) of minimal size
(min |h| = 1), so accuracy is unambiguous."*  The release agrees without
exception -- all 419,475 instances across every shipped file carry exactly one
gold.  DeFAb is therefore an SCS benchmark, and being SCS it is also run under
MCS and BOV: same pool, same single right answer, harder framings of the ask.
An instance with any other number of golds is skipped and counted rather than
reshaped, because the benchmark does not define one.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..core.adapter import SkippedDataset
from ..core.metrics import (
    aggregate_mean_metrics,
)
from ..core.types import AdapterDocumentation, ModelResponse, SampleScore, SampleSpec
from . import _common as C
from ._base import PooledDatasetAdapter, selection_score, text_match_score

REPO_ID = "PatrickAllenCooper/DeFAb"


class DeFAbAdapter(PooledDatasetAdapter):
    """Pick the candidate rule/fact whose addition explains the target."""

    adapter_version = "1.0"

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. You are given a defeasible "
        "theory, a target conclusion it does not currently support, and candidate additions. "
        "Identify the candidate whose addition would make the target derivable while "
        "respecting the theory's defeaters. Exactly one candidate does."
    )
    data_delivery_mode = "static"

    options_heading = "Candidate explanations:"
    objective_metrics = True
    # One gold among six candidates, in every instance the release ships and by
    # the paper's own statement (|H*| = 1). See the module docstring.
    selection_cardinality = "single"
    primary_metric = "accuracy"

    @property
    def _subtask(self) -> str:
        return str(self.context.option("subtask", "selection"))

    #: Files under ``instances/`` that hold generation statistics, not items.
    _NOT_INSTANCES = {"generation_summary.json", "manifest.json"}

    #: Filled by :meth:`load_items`; declared here so documentation() is safe to
    #: call on an adapter that has not been prepared.
    _per_source: dict[str, int] = {}
    _duplicates_dropped: int = 0

    @staticmethod
    def _rows_of(payload: Any) -> list[dict[str, Any]]:
        """The instance rows in a release file, whichever shape it ships in.

        The multitier and tier1/tier2 files are bare JSON arrays.  The tier0 and
        synthetic files are objects that carry their metadata alongside the
        items, as ``{"metadata": {...}, "instances": [...]}``.  This adapter
        read only the array form until 2026-09-14, so every tier0 and synthetic
        file -- 818 instances, including the Level 3 set the paper's hardest
        results are reported on -- was silently dropped: no error, no count, just
        absent from the pool.
        """
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            rows = payload.get("instances") or []
        else:
            rows = []
        return [row for row in rows if isinstance(row, dict)]

    @staticmethod
    def _normalize(row: dict[str, Any], source: str) -> dict[str, Any] | None:
        """One release row in the shape the rest of this adapter expects.

        The two shapes differ in more than nesting: tier1/tier2/multitier name
        the observation ``target`` and ship ``gold`` as a one-element list, while
        tier0 names it ``anomaly`` and ships ``gold`` as a bare string.  Both mean
        the same thing, so both are read.
        """
        target = C.normalize_whitespace(row.get("target") or row.get("anomaly"))
        raw_gold = row.get("gold")
        gold = [raw_gold] if isinstance(raw_gold, str) else list(C.as_list(raw_gold))
        candidates = [C.normalize_whitespace(text) for text in C.as_list(row.get("candidates"))]
        if not candidates and row.get("distractors"):
            # tier0 ships the pool as gold + distractors as well as pre-joined.
            candidates = [C.normalize_whitespace(text)
                          for text in [*gold, *C.as_list(row.get("distractors"))]]
        gold = [C.normalize_whitespace(text) for text in gold]
        candidates = [text for text in candidates if text]
        gold = [text for text in gold if text]
        if not target or not candidates or not gold:
            return None
        return {
            **row,
            "target": target,
            "gold": gold,
            "candidates": candidates,
            "level": row.get("level"),
            "source": source,
        }

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
            if path.name not in self._NOT_INSTANCES
        ]
        if not files:
            raise SkippedDataset("no DeFAb instance files found in the release")
        wanted = {str(name) for name in (self.context.option("domains") or [])}
        items: list[dict[str, Any]] = []
        self._per_source: dict[str, int] = {}
        self._duplicates_dropped = 0
        seen: set[tuple[str, str, str]] = set()
        for path in sorted(files):
            try:
                relative = path.relative_to(root / "instances")
            except ValueError:  # pragma: no cover - find_files searched that root
                relative = path
            # Tier-qualified, because the ontology name alone is ambiguous:
            # multitier/framenet and tier2/framenet are different paths and, as
            # it turns out, byte-identical files.
            source = str(relative.parent) if path.name == "instances.json" else (
                f"{relative.parent}/{path.stem.removesuffix('_instances')}"
            )
            if wanted and source not in wanted and relative.parent.name not in wanted:
                continue
            kept = 0
            for row in self._rows_of(C.read_json(path)):
                item = self._normalize(row, source)
                if item is None:
                    continue
                # multitier/<x> and tier2/<x> ship the same instances twice, so
                # an undeduplicated pool can draw the same item into one
                # evaluation set twice and count it as two observations.
                key = (item["target"], "|".join(item["candidates"]), "|".join(item["gold"]))
                if key in seen:
                    self._duplicates_dropped += 1
                    continue
                seen.add(key)
                items.append(item)
                kept += 1
            if kept:
                self._per_source[source] = kept
        if not items:
            raise SkippedDataset("DeFAb instance files contained no usable rows")
        self.split_used = (
            "all released instance sets pooled: "
            + ", ".join(f"{name} ({count})" for name, count in sorted(self._per_source.items()))
            + f" -- the release ships no train/test split; {self._duplicates_dropped} instances "
            "that appear under more than one tier were kept once"
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
        if len(set(gold)) != 1:
            # The benchmark defines |H*| = 1 and every released instance obeys
            # it. Anything else is a malformed row, not a multi-answer item, and
            # inventing a set-valued task for it would report a number the
            # benchmark does not define.
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
        if len(gold_labels) != 1:
            return None
        return SampleSpec(
            sample_id=sample_id,
            fields={
                "observation": target,
                # Deliberately not "which candidate(s)": the number of correct
                # answers is a fact about the benchmark, and under MCS the whole
                # point is that the model has to work out that the answer is one
                # rather than being told how many to name.
                "question": (
                    "Which candidate, if added to the theory, would explain the observation?"
                ),
                "options": ordered,
                "option_labels": labels,
                "instructions": (
                    "Candidates are facts and defeasible rules in logical notation; a defeasible "
                    "rule may be overridden by a more specific one."
                ),
            },
            reference={"gold_labels": gold_labels, "gold_label": gold_labels[0]},
            task_kind="selection",
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
        # One scorer for all three selection modes: SCS reads a single label,
        # MCS reads a set, and BOV hands back a set rebuilt from its per-
        # hypothesis yes/no answers. `accuracy` is an exact match against the
        # one gold label in every case, which is what makes the three columns
        # comparable; `set_f1` and `n_selected` come along under MCS and BOV.
        score = selection_score(
            response,
            labels=sample.fields["option_labels"],
            gold_label=sample.reference["gold_label"],
            output_contract=output_contract,
            metric_name="accuracy",
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
                "that ship evaluation items are pooled -- seven ontologies under multitier, the "
                "five tier1 domains, and the four tier0 sets including the Level 3 one -- keyed "
                "by tier so options.domains can restrict them either by tier-qualified name "
                "('tier0/level3') or by ontology ('framenet'). tier2 duplicates multitier "
                "exactly and is therefore folded into it. instances/synthetic holds "
                "contamination-control *theories* with structural-match statistics, not items: "
                "its rows carry no target, candidates or gold, so there is nothing to ask. The "
                "repository's evaluation/ and rules/ directories hold baseline results and "
                "source theories, not evaluation items, and are ignored."
            ),
            sampling_procedure=self.sampling_note(),
            metrics_description={
                "self_consistency_<metric>":
                "Every metric also gets a self_consistency_ counterpart: the plurality answer over "
                "modes.repeats samples of the same record, read off those samples rather than bought "
                "again. Available because this dataset's answers are checkable and so can coincide.",
                "accuracy": "(PRIMARY, higher is better) 1 if the model named the gold candidate "
                "and nothing else. Every instance has exactly one gold, so under MCS and BOV -- "
                "where the model may name several -- this stays an exact set match and the three "
                "selection modes report the same quantity",
                "set_f1": "MCS/BOV only: F1 between the selected label set and the one-element "
                "gold set, i.e. the partial credit `accuracy` withholds",
                "set_precision": "MCS/BOV only: share of selected labels that are gold",
                "set_recall": "MCS/BOV only: share of gold labels selected (0 or 1 here)",
                "n_selected": "MCS/BOV only: how many candidates the model committed to. The gold "
                "count is 1, so a mean above 1 means the score is being propped up by hedging",
                "bov_yes_rate": "BOV only: share of the per-candidate yes/no questions answered "
                "yes. A model that says yes to everything selects everything",
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
                "Single-answer (SCS), on the paper's own statement -- 'Both levels have a unique "
                "gold hypothesis (|H*| = 1) of minimal size (min |h| = 1), so accuracy is "
                "unambiguous' (arXiv:2606.18557) -- and on the release, in which all 419,475 "
                "instances carry exactly one gold. Being single-answer, it is also run under MCS "
                "and BOV: the pool and the right answer are unchanged, only how the model is "
                "asked to commit. It was declared multi-answer until 2026-09-14, which offered "
                "MCS but withheld SCS, the benchmark's own framing.",
                "Instances whose gold count is not exactly 1 are skipped: the benchmark defines "
                "no such item, so scoring one would report a number it does not define.",
                "Instances whose gold strings are not a subset of their candidate list are "
                "skipped rather than aligned by guesswork.",
                "The tier0 and synthetic instance files store their items under an `instances` "
                "key rather than as a bare array. They were read as empty until 2026-09-14, "
                "which dropped 818 instances -- including the Level 3 set -- without a word.",
                "Instances that appear under more than one tier (multitier/<x> and tier2/<x> are "
                "byte-identical) are pooled once, so one item cannot be drawn twice into the same "
                "evaluation set and counted as two observations.",
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
                "cross_tier_duplicates_dropped": self._duplicates_dropped,
                "subtask": self._subtask,
            },
        )
