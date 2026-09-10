"""UniADILR-HGc: name the hypothesis a body of premises supports.

Source: https://github.com/YuSheng-00/UniADILR

UniADILR generates one dataset covering several reasoning types and ships them
as parallel files -- ``abduction.jsonl``, ``deduction.jsonl`` and
``induction.jsonl`` under ``data/UniADILR-HGc/``.  **Only the abduction file is
used here.**  The other two are the same construction applied to deductive and
inductive inference, and including them would report non-abductive reasoning
under an abduction benchmark.  Every row of the abduction file carries
``reasoning_type: "abduction"``, and rows that do not are dropped, so the
restriction is enforced rather than assumed.

**The task.**  An item gives a numbered pool of sentences (``sent1`` ...
``sentN``), most of which are distractors, and a hypothesis that a small
subset of them supports -- the ``proof`` field names which, e.g.
``sent5 & sent13 -> Sarah is a talented programmer.``  The model sees the pool
and must state the hypothesis those premises point to.  What makes it
constrained abduction rather than deduction is the pool: the supporting
sentences must be found among many irrelevant ones before the hypothesis can be
formed.

**Scoring.**  There is a gold hypothesis, but it is one sentence's worth of
wording for a claim that can be phrased many ways, so an exact match would
punish a correct paraphrase.  Following the suite's rule for *unverifiable
output with a gold available*, an LLM judge scores similarity to the gold.
"""

from __future__ import annotations

import re
from typing import Any

from ..core.adapter import SkippedDataset
from ..core.metrics import aggregate_mean_metrics
from ..core.types import (
    AdapterDocumentation,
    ModelResponse,
    SampleScore,
    SampleSpec,
)
from . import _common as C
from ._base import PooledDatasetAdapter, apply_judged_metric, judged_only_score

REPO_URL = "https://github.com/YuSheng-00/UniADILR.git"
#: The abductive split, and only it.
ABDUCTION_FILE = "data/UniADILR-HGc/abduction.jsonl"
_SENT_RE = re.compile(r"\bsent\d+\b")


class UniADILRHGcAdapter(PooledDatasetAdapter):
    """Constrained abductive hypothesis generation over a pool of premises."""

    adapter_version = "1.0"

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. You are given a numbered pool of "
        "statements, most of which are irrelevant. A small number of them together support a "
        "single further claim. Find those statements and state the claim they support."
    )
    answer_format = "one sentence"
    answer_constraints = (
        "write exactly one sentence",
        "state the claim the supporting statements point to",
        "do not cite the statement numbers",
        "do not explain your reasoning",
        "do not use introductory phrases or commentary",
    )
    data_delivery_mode = "static"
    #: The gold is one phrasing of a claim that can be worded many ways, so the
    #: judge scores it rather than a string comparison.
    objective_metrics = False
    selection_cardinality = None
    hypothesis_modes = ("generation",)
    hypothesis_mode_options = {"generation": {"subtask": "generation"}}
    table_hypothesis_mode = "Generation"
    primary_metric = "hypothesis_judged"

    # ------------------------------------------------------------------ #
    # data
    # ------------------------------------------------------------------ #

    def load_items(self) -> list[dict[str, Any]]:
        root = C.ensure_git_repo(
            REPO_URL, self.context.data_dir / "repo", depth=1, offline=self.context.offline
        )
        path = root / ABDUCTION_FILE
        if not path.is_file():
            raise SkippedDataset(
                f"UniADILR's abductive split is not in the clone: {path} does not exist"
            )
        rows = C.read_jsonl(path)
        if not rows:
            raise SkippedDataset(f"{path} contained no rows")
        # Enforced, not assumed: the deduction and induction files are the same
        # construction over other reasoning types and are not read at all, and a
        # row here that is not marked abduction is dropped.
        abductive = [
            row for row in rows
            if str(row.get("reasoning_type", "abduction")).lower() == "abduction"
        ]
        if not abductive:
            raise SkippedDataset(
                f"{path} has {len(rows)} rows but none are marked reasoning_type=abduction"
            )
        self.dropped_non_abductive = len(rows) - len(abductive)
        self.split_used = (
            f"{ABDUCTION_FILE} in full ({len(abductive)} items; the deduction and "
            f"induction files are not used)"
        )
        return abductive

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        context = item.get("context")
        gold = C.normalize_whitespace(item.get("hypothesis"))
        if not isinstance(context, dict) or not context or not gold:
            return None
        # Presented in the release's own order, which is the numbering its
        # proofs refer to.
        def _key(name: str) -> int:
            digits = re.findall(r"\d+", name)
            return int(digits[0]) if digits else 0

        statements = [
            f"{name}. {C.normalize_whitespace(text)}"
            for name, text in sorted(context.items(), key=lambda kv: _key(kv[0]))
            if C.normalize_whitespace(text)
        ]
        if len(statements) < 2:
            return None
        proof = C.normalize_whitespace(item.get("proof"))
        return SampleSpec(
            sample_id=C.stable_id("uniadilr", index),
            fields={
                "observation": "\n".join(statements),
                "question": "Which claim do a few of these statements together support?",
            },
            reference={"gold": gold, "proof": proof},
            task_kind="generation",
            metadata={
                "n_statements": len(statements),
                # How many premises the proof uses: the difficulty of finding
                # them among the distractors.
                "n_premises": len(set(_SENT_RE.findall(proof))),
            },
        )

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
        """Parse only: the judge compares the claim to the gold one.

        No overlap metric is emitted. The gold is one phrasing of the claim,
        and similarity to that phrasing is not the same thing as naming the
        right claim.
        """
        return judged_only_score(
            response,
            metric="hypothesis_judged",
            output_contract=output_contract,
            details={},
        )

    def judge_request(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore
    ) -> dict[str, Any] | None:
        if not response.text:
            return None
        return {
            "candidate": score.prediction or response.text[:600],
            "gold": sample.reference["gold"],
            "observation": C.clip_words(sample.fields["observation"], 250),
            "criteria": (
                "The candidate is correct if it states the same claim as the reference, "
                "however it is worded. A claim about a different subject, or a weaker or "
                "stronger claim than the reference makes, is not the same claim."
            ),
        }

    def apply_judge(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore, verdict: Any
    ) -> SampleScore:
        return apply_judged_metric(score, verdict, "hypothesis_judged")

    def aggregate(self, scores) -> dict[str, float]:
        return aggregate_mean_metrics([score.metrics for score in scores])

    # ------------------------------------------------------------------ #
    # documentation
    # ------------------------------------------------------------------ #

    def documentation(self) -> AdapterDocumentation:
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="UniADILR-HGc",
            domain="Formal Reasoning: Constrained Abductive Hypothesis Generation",
            source_url="https://github.com/YuSheng-00/UniADILR",
            processing_mode="Generation",
            split_used=getattr(self, "split_used", ABDUCTION_FILE),
            abductive_subset=(
                "THE ABDUCTION FILE ONLY. UniADILR ships data/UniADILR-HGc/abduction.jsonl "
                "beside deduction.jsonl and induction.jsonl, which apply the same "
                "construction to other reasoning types; those two are never read, and a row "
                "in the abduction file whose reasoning_type is not 'abduction' is dropped. "
                "What remains is constrained abduction: the supporting premises must be "
                "found among many distractors before the hypothesis can be formed."
            ),
            sampling_procedure=self.sampling_note(),
            metrics_description={
                "hypothesis_judged": (
                    "(PRIMARY, higher is better, 0-1) LLM-judge verdict on whether the stated "
                    "claim is the same claim as the reference, however worded. 1.0 when the "
                    "judge affirms, 0.0 otherwise or when nothing could be parsed."
                ),
                "parse_failure_rate": (
                    "(lower is better, 0-1) fraction of responses no claim could be extracted "
                    "from; these score 0 and are counted here separately from being wrong."
                ),
                "best_of_n_<metric>":
                "Every metric also gets a best_of_n_ counterpart: per record, the repeat the "
                "judge scored highest. This dataset has no checkable answer, so a plurality is "
                "meaningless -- free-text claims never repeat verbatim -- and Best-of-N "
                "replaces it.",
            },
            primary_metric="hypothesis_judged",
            decisions=[
                "Used only the abductive split, and enforced it per row rather than trusting "
                "the filename.",
                "Presented the statements in the release's own numbering, which is what its "
                "proof field refers to.",
                "Scored by an LLM judge against the gold claim rather than by string overlap: "
                "the gold is one phrasing of a claim that can be worded many ways.",
            ],
            caveats=[
                "The proof field names the supporting sentences and is withheld from the "
                "prompt -- it would give away which statements matter, which is the "
                "difficulty of the task.",
                "Items are synthetic, so their distractors are drawn from unrelated corpora "
                "and read as obviously unrelated more often than a natural pool would.",
            ],
            statistics={
                **self.base_statistics(),
                "dropped_non_abductive": getattr(self, "dropped_non_abductive", 0),
            },
        )
