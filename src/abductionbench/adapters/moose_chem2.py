"""MOOSE-Chem2: fine-grained chemistry hypothesis discovery.

Source: https://github.com/ZonglinY/MOOSE-Chem2

``Data/chem_research_2024_finegrained.xlsx`` distils 51 chemistry papers into a
background survey, the research question they answer, the paper's ``Main
hypothesis`` and a much longer ``Finegrained Hypothesis`` (the methodology-level
version), plus the inspirations and the reasoning chain that led there.

The abductive task: from the background survey and the question, propose the
hypothesis.  Everything that reveals it -- ``Main Inspiration``, the inspiration
paper titles and relations, ``Reasoning Process``, ``Experiments to Verify``,
both hypothesis columns and the paper ``Title`` -- is withheld from the prompt.

The Google-Drive archive linked from the repository holds the authors' run
checkpoints and analysis outputs (not gold data) and is deliberately not used.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..core.adapter import SkippedDataset
from ..core.metrics import aggregate_mean_metrics, extract_answer_span, rouge_l, token_f1
from ..core.types import AdapterDocumentation, ModelResponse, SampleScore, SampleSpec
from . import _common as C
from ._base import PooledDatasetAdapter, unparsed_score

REPO_URL = "https://github.com/ZonglinY/MOOSE-Chem2"


class MooseChem2Adapter(PooledDatasetAdapter):
    """Propose the paper's hypothesis from its background and question."""

    adapter_version = "1.0"

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. You are given the background and "
        "research question of a chemistry paper. Propose the hypothesis it went on to "
        "confirm: a specific mechanism or relationship, stated so that an experiment could "
        "test it."
    )
    data_delivery_mode = "static"
    objective_metrics = False
    selection_cardinality = None
    hypothesis_modes = ("generation",)
    table_hypothesis_mode = "Generation & Selection (separate tasks)"
    primary_metric = "hypothesis_judged"

    def load_items(self) -> list[dict[str, Any]]:
        root = C.ensure_git_repo(
            REPO_URL, self.context.data_dir / "repo", offline=self.context.offline
        )
        workbooks = C.find_files(root / "Data", ["*finegrained*.xlsx"]) or C.find_files(
            root, ["*.xlsx"]
        )
        if not workbooks:
            raise SkippedDataset("MOOSE-Chem2 fine-grained workbook not found")
        import pandas as pd

        frame = pd.read_excel(workbooks[0])
        rows = [
            record
            for record in frame.to_dict(orient="records")
            if C.normalize_whitespace(record.get("Background Question"))
        ]
        if not rows:
            raise SkippedDataset("no usable rows in the MOOSE-Chem2 workbook")
        self.split_used = (
            f"{workbooks[0].name} in full ({len(rows)} papers); the release ships a single table "
            "and is smaller than the 300-sample target"
        )
        return rows

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        question = C.normalize_whitespace(item.get("Background Question"))
        survey = C.normalize_whitespace(
            item.get("Background Little Survey") or item.get("Background Little Survey (strict)")
        )
        coarse = C.normalize_whitespace(item.get("Main hypothesis"))
        fine = C.normalize_whitespace(item.get("Finegrained Hypothesis"))
        if not question or not (coarse or fine):
            return None
        granularity = str(self.context.option("granularity", "finegrained"))
        gold = (fine or coarse) if granularity == "finegrained" else (coarse or fine)
        survey_words = int(self.context.option("survey_words", 350))
        return SampleSpec(
            sample_id=C.stable_id("moose2", item.get("No", index)),
            fields={
                "context": (
                    "Background survey:\n" + C.clip_words(survey, survey_words) if survey else ""
                ),
                "observation": question,
                "question": (
                    "What research hypothesis would answer this question, given the state of the "
                    "field?"
                ),
                "instructions": (
                    "State the hypothesis at methodology level: what to make or do, with which "
                    "materials or mechanism, and what outcome it should achieve."
                    if granularity == "finegrained"
                    else "State the core hypothesis in two or three sentences."
                ),
            },
            reference={"gold": gold, "coarse": coarse, "fine": fine},
            task_kind="generation",
            # Fine-grained hypotheses are long (methodology level), so the budget
            # is set from the granularity actually being scored.
            max_tokens=1536 if granularity == "finegrained" else 768,
            metadata={
                "paper_no": item.get("No"),
                "publisher": item.get("Publisher"),
                "granularity": granularity,
            },
        )

    def score(
        self,
        sample: SampleSpec,
        response: ModelResponse,
        *,
        output_contract: dict[str, Any] | None = None,
    ) -> SampleScore:
        answer = extract_answer_span(response.text, output_contract) or response.text
        if not answer.strip():
            return unparsed_score(
                ["hypothesis_rouge_l", "hypothesis_token_f1"], raw=response.text[:200]
            )
        gold = sample.reference["gold"]
        metrics = {
            "hypothesis_rouge_l": rouge_l(answer, gold)["f"],
            "hypothesis_token_f1": token_f1(answer, gold),
        }
        # Report against both granularities when both exist, since a model may
        # match the coarse idea without the methodology detail.
        if sample.reference.get("coarse") and sample.reference.get("fine"):
            metrics["coarse_rouge_l"] = rouge_l(answer, sample.reference["coarse"])["f"]
            metrics["fine_rouge_l"] = rouge_l(answer, sample.reference["fine"])["f"]
        return SampleScore(
            metrics=metrics, prediction=answer[:600], details={"gold": gold[:300]}
        )

    def aggregate(self, scores: Sequence[SampleScore]) -> dict[str, float]:
        return aggregate_mean_metrics([score.metrics for score in scores])

    def judge_request(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore
    ) -> dict[str, Any] | None:
        if not response.text:
            return None
        return {
            "candidate": (score.prediction or response.text)[:900],
            "gold": sample.reference.get("coarse") or sample.reference["gold"][:600],
            "observation": sample.fields["observation"],
            "criteria": (
                "Correct if the candidate proposes the same core mechanism/material strategy as "
                "the reference, even with less detail."
            ),
        }

    def apply_judge(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore, verdict: Any
    ) -> SampleScore:
        metrics = dict(score.metrics)
        metrics["hypothesis_judged"] = 1.0 if getattr(verdict, "positive", False) else 0.0
        return SampleScore(
            metrics=metrics,
            prediction=score.prediction,
            parse_ok=score.parse_ok,
            details={**score.details, "judge_label": getattr(verdict, "label", None)},
        )

    def documentation(self) -> AdapterDocumentation:
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="MOOSE-Chem2",
            domain="Scientific Discovery: Chemistry",
            source_url=REPO_URL,
            processing_mode="Generation (selection not supported by the release)",
            split_used=self.split_used,
            abductive_subset=(
                "Background + question -> hypothesis. Withheld from the prompt: paper Title, "
                "Main Inspiration, all inspiration-paper titles and relations, Reasoning Process, "
                "Experiments to Verify, and both hypothesis columns."
            ),
            sampling_procedure=self.sampling_note(),
            metrics_description={
                "hypothesis_rouge_l": "ROUGE-L F against the scored granularity's gold hypothesis "
                "(primary)",
                "hypothesis_token_f1": "token F1 against that gold hypothesis",
                "coarse_rouge_l": "ROUGE-L against the paper's Main hypothesis",
                "fine_rouge_l": "ROUGE-L against the Finegrained Hypothesis -- the gap between the "
                "two shows whether a model got the idea but not the methodology",
                "hypothesis_judged": "LLM-judge verdict on same-core-mechanism (only when "
                "engine.judge.enabled)",
            },
            primary_metric="hypothesis_rouge_l",
            decisions=[
                "Took the data from the repository's own workbook; the linked Google-Drive archive "
                "contains the authors' run checkpoints and analysis outputs, not gold data.",
                "Default granularity is finegrained (the benchmark's own emphasis); "
                "options.granularity = coarse scores against Main hypothesis instead.",
                "Clipped the background survey to 350 words (configurable) to stay inside the "
                "input budget.",
                "No selection mode: the release provides no candidate hypothesis sets.",
            ],
            caveats=[
                "Only 51 papers, so this dataset reports far fewer than the 300-sample target.",
                "Fine-grained references are long paragraphs; ROUGE-L rewards verbosity, so read "
                "it together with the judged metric.",
                "All papers are from 2024 and may fall inside a model's pretraining window.",
            ],
            statistics=self.base_statistics(),
        )
