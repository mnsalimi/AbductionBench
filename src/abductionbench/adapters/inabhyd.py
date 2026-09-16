"""InAbHyD: explain observations under an incomplete world model.

Source: https://github.com/byrantwithyou/inabhyd

**The task.**  An item gives a small theory -- rules and facts over a synthetic
vocabulary ("Every gomper is blue.", "Sarpers are gompers.") -- and an
observation the theory does not yet account for ("Charles is angry. Charles is
blue. Charles is not muffled.").  The world model is deliberately incomplete,
and the model must supply the hypothesis that closes the gap ("Charles is a
sarper").  Three things can be missing, and the release generates each:

``membership``  which class an individual belongs to
``ontology``    which class is a subclass of which
``property``    which property a class carries

All three are abductive -- each asks for the assumption that would make the
observation follow -- so all three are generated, and ``options.recover``
selects a subset.

**The items are generated, not downloaded.**  InAbHyD ships a generator rather
than a fixed file: ``Ontology(OntologyConfig(...))`` builds a theory, its
observations and its gold hypothesis together.  This adapter calls that
generator with a fixed seed, so the same items come back every run.  The
``data/*.pkl`` files in the repository are recorded *replies* from the authors'
own experiments, not the benchmark items, and are not read.

**Scoring.**  An LLM judge, comparing the stated hypothesis to the gold one.
A string comparison looked adequate -- the vocabulary is synthetic, so there is
no world knowledge to paraphrase around -- and measurement showed it is not:
the model answered "A kurpor is windy." against a gold of "Every kurpor is
windy", which is the same hypothesis and scored zero.  The quantifier and the
article vary while the claim does not, so this is an unverifiable output *with*
a gold available, and the judge scores similarity to it.

**Prompt.**  Static delivery, so the suite's standardised prompt structure is
used rather than the wording in the release's ``prompt.py`` -- the point of a
static dataset here is to measure the difficulty of its data, not its authors'
prompt engineering.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

from ..core.adapter import SkippedDataset
from ..core.types import AdapterDocumentation, ModelResponse, SampleScore, SampleSpec
from . import _common as C
from ._base import PooledDatasetAdapter, apply_judged_metric, judged_only_score

REPO_URL = "https://github.com/byrantwithyou/inabhyd.git"

#: The three things the release can hide, each a different abductive question.
RECOVER_MODES = ("membership", "ontology", "property")
#: Reasoning depth: how many rule applications separate the hypothesis from the
#: observation. The release's own MIN_HOP/MAX_HOP.
DEFAULT_HOPS = (1, 2, 3, 4)


class InAbHyDAdapter(PooledDatasetAdapter):
    """Generate the hypothesis that completes an incomplete world model."""

    adapter_version = "1.0"

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. You are given a theory -- a set "
        "of rules and facts -- and an observation the theory does not yet explain. The "
        "theory is incomplete: exactly one further statement, consistent with it, would make "
        "the observation follow. Name that statement."
    )
    answer_format = "one statement"
    answer_constraints = (
        "output exactly one statement",
        "use the same vocabulary and phrasing style as the theory",
        "state only the missing assumption, not the reasoning that follows from it",
        "do not restate the observation",
        "do not explain your reasoning",
    )
    data_delivery_mode = "static"
    #: Measured, not assumed: "A kurpor is windy." against a gold of "Every
    #: kurpor is windy" is the same hypothesis and fails an exact match. The
    #: claim is fixed but its phrasing is not, so the judge scores it.
    objective_metrics = False
    selection_cardinality = None
    hypothesis_modes = ("generation",)
    hypothesis_mode_options = {"generation": {"subtask": "generation"}}
    table_hypothesis_mode = "Generation"
    primary_metric = "hypothesis_judged"

    # ------------------------------------------------------------------ #
    # generation
    # ------------------------------------------------------------------ #

    def _generator(self):
        """Import the release's own ontology generator from the clone."""
        root = C.ensure_git_repo(
            REPO_URL, self.context.data_dir / "repo", depth=1, offline=self.context.offline
        )
        if not (root / "ontology.py").is_file():
            raise SkippedDataset(
                f"InAbHyD's generator (ontology.py) is not in the clone at {root}. "
                f"SETUP: git clone --depth 1 {REPO_URL} {root}"
            )
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        try:
            from ontology import Ontology, OntologyConfig  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            raise SkippedDataset(
                f"InAbHyD's generator could not be imported ({type(exc).__name__}: {exc})"
            ) from exc
        return Ontology, OntologyConfig

    def load_items(self) -> list[dict[str, Any]]:
        Ontology, OntologyConfig = self._generator()
        modes = [
            m for m in (self.context.option("recover", None) or RECOVER_MODES)
            if m in RECOVER_MODES
        ]
        if not modes:
            raise SkippedDataset(
                f"options.recover selected none of {list(RECOVER_MODES)}"
            )
        hops = [int(h) for h in (self.context.option("hops", None) or DEFAULT_HOPS)]
        per_cell = int(self.context.option("per_cell", 40))

        items: list[dict[str, Any]] = []
        failures = 0
        for mode in modes:
            for hop in hops:
                for index in range(per_cell):
                    # Seeded per cell and index, so the same theory comes back
                    # on a resumed run and across sample sizes.
                    random.seed(f"{self.context.seed}::inabhyd::{mode}::{hop}::{index}")
                    config = OntologyConfig(hops=hop, **{f"recover_{mode}": True})
                    try:
                        ontology = Ontology(config)
                    except Exception:  # noqa: BLE001 - the generator rejects some draws
                        failures += 1
                        continue
                    items.append({
                        "theories": str(ontology.theories),
                        "observations": str(ontology.observations),
                        "hypotheses": str(ontology.hypotheses),
                        "recover": mode,
                        "hops": hop,
                        "index": index,
                    })
        if not items:
            raise SkippedDataset("InAbHyD's generator produced no usable theories")
        self.generator_failures = failures
        self.split_used = (
            f"generated: {len(items)} items over recover={modes}, hops={hops} "
            f"(seeded; the release ships a generator, not a fixed file)"
        )
        return items

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        theory = C.normalize_whitespace(item.get("theories"))
        observation = C.normalize_whitespace(item.get("observations"))
        gold = C.normalize_whitespace(item.get("hypotheses"))
        if not theory or not observation or not gold:
            return None
        return SampleSpec(
            sample_id=C.stable_id("inabhyd", item["recover"], item["hops"], item["index"]),
            fields={
                "context": f"Theory:\n{theory}",
                "observation": observation,
                "question": (
                    "Which single further statement, consistent with the theory, would "
                    "make this observation follow?"
                ),
            },
            reference={"gold": gold},
            task_kind="generation",
            metadata={"recover": item["recover"], "hops": item["hops"]},
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
        """Parse only: the judge compares the statement to the hidden one.

        The two axes the benchmark varies are seeded here as strata so a single
        mean cannot hide that one recovery type or one depth carries the score;
        apply_judged_metric fills them from the same verdict.
        """
        extra = {
            f"hypothesis_judged_{sample.metadata['recover']}": 0.0,
            f"hypothesis_judged_{sample.metadata['hops']}hop": 0.0,
        }
        return judged_only_score(
            response,
            metric="hypothesis_judged",
            output_contract=output_contract,
            extra_metrics=extra,
            details={},
        )

    def judge_request(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore
    ) -> dict[str, Any] | None:
        if not response.text:
            return None
        return {
            "candidate": score.prediction or response.text[:400],
            "gold": sample.reference["gold"],
            "observation": C.clip_words(sample.fields["observation"], 120),
            "criteria": (
                "The candidate is correct if it states the same assumption as the "
                "reference. Differences of quantifier phrasing that do not change the "
                "claim ('Every X is Y' / 'All X are Y' / 'A X is Y') count as the same. "
                "Naming a different individual, class or property does not."
            ),
        }

    def apply_judge(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore, verdict: Any
    ) -> SampleScore:
        return apply_judged_metric(score, verdict, "hypothesis_judged")

    # ------------------------------------------------------------------ #
    # documentation
    # ------------------------------------------------------------------ #

    def documentation(self) -> AdapterDocumentation:
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="InAbHyD",
            domain="Formal Reasoning: Inductive and Abductive Hypothesis Generation",
            source_url="https://github.com/byrantwithyou/inabhyd",
            processing_mode="Generation",
            split_used=getattr(self, "split_used", "generated"),
            abductive_subset=(
                "The abductive recovery tasks, which is what the generator produces here: "
                "membership, ontology and property recovery each hide one statement from an "
                "otherwise complete theory and ask for the assumption that would make the "
                "observation follow. All three are abductive, so all three are generated; "
                "options.recover narrows to a subset."
            ),
            sampling_procedure=(
                "Generated, not sampled from a file: the release ships a generator, and it is "
                "called with a per-item seed so the same theories return every run. Items are "
                "spread evenly over the recovery types and hop depths."
            ),
            metrics_description={
                "hypothesis_judged": (
                    "(PRIMARY, higher is better, 0-1) LLM-judge verdict on whether the stated "
                    "assumption is the same one the theory is missing, however it is "
                    "quantified. 1.0 when the judge affirms, 0.0 otherwise or when nothing "
                    "could be parsed."
                ),
                "hypothesis_judged_<recover>": (
                    "the same verdict per recovery type (membership / ontology / property) -- "
                    "which kind of missing statement the model can supply"
                ),
                "hypothesis_judged_<n>hop": (
                    "the same verdict per reasoning depth -- how far the hypothesis sits from "
                    "the observation"
                ),
                "parse_failure_rate": (
                    "(lower is better, 0-1) fraction of responses no statement could be "
                    "extracted from; these score 0 and are counted separately from being wrong."
                ),
                "best_of_n_<metric>":
                "Every metric also gets a best_of_n_ counterpart: per record, the repeat the "
                "judge scored highest. The claim is fixed but its phrasing is not, so a "
                "plurality over exact strings would be meaningless and Best-of-N replaces it.",
            },
            primary_metric="hypothesis_match",
            decisions=[
                "Generated the items with the release's own Ontology/OntologyConfig rather "
                "than reading data/*.pkl: those files are recorded replies from the authors' "
                "experiments, not benchmark items.",
                "Seeded per item so the generated set is stable across runs, resumes and "
                "sample sizes.",
                "Used the suite's standardised static prompt rather than the release's "
                "prompt.py, so the score reflects the difficulty of the theories rather than "
                "the authors' prompt wording.",
                "Reported per recovery type and per hop depth, which are the two axes the "
                "benchmark varies.",
                "Scored by an LLM judge rather than by string match. An exact comparison was "
                "tried first and measured: the model's \"A kurpor is windy.\" against a gold "
                "of \"Every kurpor is windy\" is the same hypothesis and scored zero.",
            ],
            caveats=[
                "The vocabulary is synthetic nonsense by design ('gomper', 'sarper'), which "
                "removes world knowledge as a confound but also makes the surface form "
                "unfamiliar to a model.",
                "The generator rejects some draws internally and is retried; the count of "
                "rejected draws is reported as generator_failures.",
                "The judge is what makes a differently-quantified but identical claim count, "
                "so this dataset cannot be run without one.",
            ],
            statistics={
                **self.base_statistics(),
                "generator_failures": getattr(self, "generator_failures", 0),
            },
        )
