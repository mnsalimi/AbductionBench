"""HypoArena: prospective hypothesis discovery across six domains.

Source: https://huggingface.co/datasets/HypoArena/HypoData

830 items spanning biomedical science, machine learning, IT operations, social
science, safety investigation and financial analysis.  Each gives a ``context``
describing what is observed or puzzling, and one or more reference
``hypotheses`` -- each with the ``evidence`` that would test it and, for some
domains, a ``category`` (``causal``, ``latent``, ``intervention``, ...).

The abductive task: read the context and propose the hypothesis.  Where an item
has several reference hypotheses, all of them are acceptable and the closest is
credited.

**No selection mode.** The release contains reference hypotheses only -- there
are no labelled candidate sets or distractors -- so the "Generation & Selection"
mode in the suite's table is covered here as generation only.  Building
distractors would mean inventing data, which this adapter does not do.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..core.adapter import SkippedDataset
from ..core.metrics import aggregate_mean_metrics
from ..core.types import AdapterDocumentation, ModelResponse, SampleScore, SampleSpec
from . import _common as C
from ._base import PooledDatasetAdapter, apply_proxy_score, judged_only_score

REPO_ID = "HypoArena/HypoData"


def _numbered_references(references: Any) -> str:
    """The item's reference texts, numbered so a judge can name the one it used.

    Numbered rather than bulleted because the judge is asked which it scored
    against, and an index is the only handle that survives into the record.
    """
    items = [str(item).strip() for item in (references or []) if str(item).strip()]
    return "\n".join(f"{index}. {item}" for index, item in enumerate(items, start=1))


class HypoArenaAdapter(PooledDatasetAdapter):
    """Propose a hypothesis for an observed situation; multi-reference scoring."""

    adapter_version = "2.0"

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. You are given an observed "
        "situation. Propose the hypothesis that explains it: a specific, testable claim about "
        "what is going on, not a restatement of the observation."
    )
    data_delivery_mode = "static"

    answer_format = "one hypothesis, in at most five sentences"
    task_requirements = (
        "make it specific and checkable against the observations",
        "do not restate the observation",
    )
    objective_metrics = False
    selection_cardinality = None
    #: A PROJECT-SPECIFIC PROXY, not this benchmark's own protocol.
    #: The `proxy_` prefix is load-bearing: these numbers must never be
    #: read as the paper's metric, and the prefix is what a reader sees
    #: first in a sheet.
    judge_template = "proxy_hypothesis_quality_v1"
    primary_metric = "proxy_hypothesis_quality_score"

    def load_items(self) -> list[dict[str, Any]]:
        root = C.ensure_hf_snapshot(
            REPO_ID,
            self.context.data_dir / "hf",
            offline=self.context.offline,
            allow_patterns=["hypodata.json", "*.md"],
        )
        files = C.find_files(root, ["hypodata.json"]) or C.find_files(root, ["*.json"])
        if not files:
            raise SkippedDataset("hypodata.json not found in the HypoArena release")
        payload = C.read_json(files[0])
        if not isinstance(payload, list):
            raise SkippedDataset("unexpected HypoArena structure (expected a list)")
        domains = self.context.option("domains")
        rows = [
            row
            for row in payload
            if not domains or row.get("domain") in domains
        ]
        self.split_used = (
            f"whole release ({len(rows)} items across "
            f"{len({row.get('domain') for row in rows})} domains); no official split"
        )
        return rows

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        context_text = C.normalize_whitespace(item.get("context"))
        hypotheses = [
            C.normalize_whitespace(entry.get("hypothesis"))
            for entry in item.get("hypotheses") or []
            if isinstance(entry, dict) and C.normalize_whitespace(entry.get("hypothesis"))
        ]
        if not context_text or not hypotheses:
            return None
        domain = str(item.get("domain") or "unknown")
        return SampleSpec(
            sample_id=C.stable_id("hypoarena", item.get("id", index)),
            fields={
                "observation": context_text,
                "question": (
                    "What is the most plausible hypothesis that would account for this situation?"
                ),
                "instructions": (
                    "State one specific, testable hypothesis, and name the evidence that "
                    "would test it."
                ),
            },
            reference={"gold": hypotheses[0], "references": hypotheses},
            task_kind="generation",
            # Research-grade hypotheses with a testing plan: a real writing budget.
            max_tokens=1024,
            metadata={
                "domain": domain,
                "n_references": len(hypotheses),
                "categories": [
                    entry.get("category")
                    for entry in item.get("hypotheses") or []
                    if isinstance(entry, dict)
                ],
            },
        )

    def score(
        self,
        sample: SampleSpec,
        response: ModelResponse,
        *,
        output_contract: dict[str, Any] | None = None,
    ) -> SampleScore:
        """Parse only: the judge is what scores this dataset.

        No overlap metric is emitted. The reference here is one
        acceptable explanation among many, so similarity to it measures
        resemblance to one particular wording rather than correctness.
        """
        stratum = sample.metadata.get("domain", "unknown")
        extra = {f"proxy_hypothesis_quality_score_{stratum}": 0.0}
        return judged_only_score(
            response,
            metric="proxy_hypothesis_quality_score",
            output_contract=output_contract,
            extra_metrics=extra,
            details={"n_references": len(sample.reference["references"])},
        )

    def aggregate(self, scores: Sequence[SampleScore]) -> dict[str, float]:
        return aggregate_mean_metrics([score.metrics for score in scores])

    def judge_request(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore
    ) -> dict[str, Any] | None:
        if not response.text:
            return None
        return {
            "candidate": score.prediction or response.text,
            # Passed for calibration, explicitly not as the answer: HypoArena's
            # source-derived hypotheses are one good analysis of the case, not
            # the only one, and scoring resemblance to them marked a better
            # hypothesis wrong for being different.
            "references": _numbered_references(sample.reference.get("references")),
            "observation": sample.fields["observation"],
        }

    def apply_judge(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore, verdict: Any
    ) -> SampleScore:
        """The verdict is the score -- and the score of every stratum of it."""
        return apply_proxy_score(score, verdict, "proxy_hypothesis_quality_score")

    def documentation(self) -> AdapterDocumentation:
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="HypoArena",
            domain="General: Prospective Hypothesis Discovery",
            source_url=f"https://huggingface.co/datasets/{REPO_ID}",
            processing_mode="Generation (selection not supported by the release)",
            split_used=self.split_used,
            abductive_subset=(
                "All items are hypothesis-discovery tasks. Every domain is kept; "
                "options.domains can restrict them."
            ),
            sampling_procedure=self.sampling_note(),
            metrics_description={
                "best_of_n_<metric>":
                "Every metric also gets a best_of_n_ counterpart: per record, the repeat the judge "
                "scored highest. This dataset has no checkable answer, so a plurality is meaningless "
                "-- free-text answers never repeat verbatim -- and Best-of-N replaces it.",
                "proxy_hypothesis_quality_score": "(PRIMARY, PROJECT-SPECIFIC PROXY -- not HypoArena's own evaluation. higher is better, 0-1) The fraction of records whose generated hypothesis or hypothesis set an LLM judge accepted. The verdict is strictly BINARY, 1 or 0, and 1 requires all three of: GROUNDED in the case, offering a real INSIGHT rather than a restatement, and TESTABLE. Any one of them failing scores 0 -- there is no partial credit for two out of three. The source-derived hypotheses are shown to the judge for calibration only and are explicitly not the answer, so a candidate that differs from all of them can still score 1. The three dimensions are recorded per sample as 1/0. Blank, never 0.0, when the judge call failed.",
                "proxy_hypothesis_quality_score_<domain>": "the same verdict restricted to one domain; identical definition, filtered "
                "population",
                "best_of_n_proxy_hypothesis_quality_score": "(higher is better, 0-1) Best-of-N: per record, the repeat the judge scored "
                "highest, then averaged over records. Read off the same modes.repeats samples -- "
                "no extra calls. This is what replaces self-consistency here: a plurality needs "
                "answers that can coincide, and free-text hypotheses do not.",
                "proxy_hypothesis_quality_score_repeat_std": "(lower is better) mean within-record standard deviation of the primary metric "
                "across repeats -- how much the same question's answers varied.",
                "repeat_agreement": "(0-1) fraction of records whose repeats all produced the identical prediction. "
                "Near 0 is expected for free-text generation.",
                "parse_failure_rate": "(lower is better, 0-1) fraction of responses no answer could be extracted from. "
                "These score 0 on the primary metric and are counted here separately, so being "
                "unparseable is distinguishable from being wrong.",
            },
            primary_metric="proxy_hypothesis_quality_score",
            decisions=[
                "No official split exists, so the whole release is the population.",
                "Multi-reference items credit the closest reference; all listed hypotheses are "
                "legitimate answers.",
                "Implemented generation only and reported why: the release ships reference "
                "hypotheses without candidate sets, so a selection mode would require inventing "
                "distractors.",
                "max_tokens=1024, because the task asks for a hypothesis plus the evidence that "
                "would test it.",
            ],
            caveats=[
                "proxy_hypothesis_quality_score IS NOT THE PAPER'S METRIC. HypoArena ranks systems by pairwise comparison in an arena; this is an absolute LLM-judge quality score defined by this suite, and a run's number here says nothing about where it would place in that arena. It replaced equivalence-scoring against hypotheses[0], which treated one source-derived hypothesis as the only right answer and marked a better, different one wrong.",
                "These overlap numbers are diagnostics, not the evaluation: "
                "character/n-gram similarity punishes a correct paraphrase and "
                "rewards a wrong sentence that reuses the reference's words, so "
                "this dataset is scored by an LLM judge instead.",
                "Reference hypotheses are long, domain-specific prose; overlap metrics are a weak "
                "proxy here and the judge stage is strongly recommended.",
            ],
            statistics=self.base_statistics(),
        )
