"""VivaBench: interactive clinical reasoning ("viva" examination).

Source: https://huggingface.co/datasets/chychiu/VivaBench

Two tables ship: ``pubmed_reviewed.csv`` (990 human-reviewed cases) and
``dataset_generated.csv`` (1,952 generated cases).  Each row has an opening
``vignette``, the accepted ``diagnosis`` list, a ``differentials`` list, and a
structured ``clinicalcase`` JSON holding the findings an examiner would reveal
on request (history, examination, investigations).

**How it is adapted.** VivaBench's own protocol is interactive: the model asks
for findings before committing.  A single-turn harness cannot ask, so evidence
disclosure becomes a configuration choice (``options.evidence``): the vignette
alone (hardest, closest to the first turn of a viva) or the vignette plus the
structured findings (an upper bound in which nothing has to be requested).

The human-reviewed table is the default -- the generated table is available via
``options.table``.
"""

from __future__ import annotations

import ast
import json
from typing import Any

from ..core.adapter import SkippedDataset
from ..core.metrics import extract_answer_span
from ..core.types import AdapterDocumentation, ModelResponse, SampleScore, SampleSpec
from . import _common as C
from ._base import PooledDatasetAdapter, selection_score, text_match_score

REPO_ID = "chychiu/VivaBench"


def _parse_listish(value: Any) -> list[str]:
    """Parse a CSV cell that holds a Python/JSON list of strings."""
    if isinstance(value, list):
        return [str(item) for item in value]
    text = str(value or "").strip()
    if not text or text in ("[]", "nan"):
        return []
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except Exception:  # noqa: BLE001 - try the next loader
            continue
    return [text]


class VivaBenchAdapter(PooledDatasetAdapter):
    """Diagnose a viva case: free-text (default) or among the case's differentials."""

    adapter_version = "1.0"
    primary_metric = "diagnosis_match"

    @property
    def _subtask(self) -> str:
        return str(self.context.option("subtask", "generation"))

    def load_items(self) -> list[dict[str, Any]]:
        root = C.ensure_hf_snapshot(
            REPO_ID,
            self.context.data_dir / "hf",
            offline=self.context.offline,
            allow_patterns=["dataset/*", "*.md"],
        )
        table = str(self.context.option("table", "pubmed_reviewed"))
        files = C.find_files(root, [f"{table}.csv"]) or C.find_files(root, ["*.csv"])
        if not files:
            raise SkippedDataset("no VivaBench CSV found in the release")
        rows = C.read_csv_rows(files[0])
        self._with_differentials = sum(1 for row in rows if len(_parse_listish(row.get("differentials"))) >= 2)
        self.split_used = (
            f"{files[0].stem} ({len(rows)} cases); the release ships no train/test split"
        )
        return rows

    def _findings(self, item: dict[str, Any]) -> str:
        """Flatten the structured clinicalcase JSON into readable findings."""
        if str(self.context.option("evidence", "vignette")) == "vignette":
            return ""
        raw = item.get("clinicalcase")
        try:
            payload = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:  # noqa: BLE001 - malformed cell
            return ""
        if not isinstance(payload, dict):
            return ""
        limit = int(self.context.option("words_per_section", 250))
        blocks: list[str] = []
        for key, value in payload.items():
            if isinstance(value, str) and value.strip():
                blocks.append(f"{key.replace('_', ' ').title()}: {C.clip_words(value, limit)}")
            elif isinstance(value, (list, dict)):
                text = C.normalize_whitespace(json.dumps(value)[:4000])
                blocks.append(f"{key.replace('_', ' ').title()}: {C.clip_words(text, limit)}")
        return "\n\n".join(blocks)

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        vignette = C.normalize_whitespace(item.get("vignette"))
        diagnoses = [C.normalize_whitespace(d) for d in _parse_listish(item.get("diagnosis"))]
        diagnoses = [d for d in diagnoses if d]
        if not vignette or not diagnoses:
            return None
        differentials = [
            C.normalize_whitespace(d) for d in _parse_listish(item.get("differentials"))
        ]
        differentials = [d for d in differentials if d]
        findings = self._findings(item)

        if self._subtask == "selection":
            # Options are the case's own differentials plus the gold diagnosis.
            candidates = [diagnoses[0]] + [d for d in differentials if d != diagnoses[0]]
            if len(candidates) < 3:
                return None  # not enough real distractors; item is skipped, never invented
            ordered = sorted(set(candidates), key=lambda text: (text != diagnoses[0], text))
            labels = C.letter_labels(len(ordered))
            return SampleSpec(
                sample_id=C.stable_id("viva", item.get("uid", index)),
                fields={
                    "observation": vignette,
                    "context": findings,
                    "question": "Which diagnosis best explains this presentation?",
                    "options": ordered,
                    "option_labels": labels,
                },
                reference={
                    "gold_label": labels[ordered.index(diagnoses[0])],
                    "gold": diagnoses[0],
                },
                task_kind="selection",
                max_tokens=1024,
                metadata={"uid": item.get("uid"), "specialty": item.get("specialty_group")},
            )

        return SampleSpec(
            sample_id=C.stable_id("viva", item.get("uid", index)),
            fields={
                "observation": vignette,
                "context": findings,
                "question": "What is the most likely diagnosis?",
                "instructions": "Name the single most likely diagnosis for this patient.",
            },
            reference={"gold": diagnoses[0], "accepted": diagnoses},
            task_kind="generation",
            max_tokens=1024,
            metadata={
                "uid": item.get("uid"),
                "specialty": item.get("specialty_group"),
                "n_accepted": len(diagnoses),
            },
        )

    def score(
        self,
        sample: SampleSpec,
        response: ModelResponse,
        *,
        output_contract: dict[str, Any] | None = None,
    ) -> SampleScore:
        if sample.task_kind == "selection":
            return selection_score(
                response,
                labels=sample.fields["option_labels"],
                gold_label=sample.reference["gold_label"],
                output_contract=output_contract,
                metric_name="accuracy",
            )
        return text_match_score(
            response,
            gold=sample.reference["gold"],
            # The dataset itself lists several accepted diagnoses per case.
            accepted=sample.reference.get("accepted", [])[1:],
            output_contract=output_contract,
            primary="diagnosis_match",
        )

    def judge_request(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore
    ) -> dict[str, Any] | None:
        if sample.task_kind != "generation" or not response.text:
            return None
        return {
            "candidate": extract_answer_span(response.text, None)[:600],
            "gold": "; ".join(sample.reference.get("accepted", [sample.reference["gold"]])),
            "observation": C.clip_words(sample.fields["observation"], 200),
            "criteria": "Any of the listed accepted diagnoses counts as correct.",
        }

    def apply_judge(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore, verdict: Any
    ) -> SampleScore:
        metrics = dict(score.metrics)
        metrics["diagnosis_match_judged"] = 1.0 if getattr(verdict, "positive", False) else 0.0
        return SampleScore(
            metrics=metrics,
            prediction=score.prediction,
            parse_ok=score.parse_ok,
            details={**score.details, "judge_label": getattr(verdict, "label", None)},
        )

    def documentation(self) -> AdapterDocumentation:
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="VivaBench",
            domain="Healthcare: Interactive Clinical Reasoning",
            source_url=f"https://huggingface.co/datasets/{REPO_ID}",
            processing_mode="Generation & Selection",
            split_used=self.split_used,
            abductive_subset=(
                "The diagnostic hypothesis. VivaBench's interactive finding-request protocol is "
                "not reproducible single-turn, so evidence disclosure is a configuration choice "
                "(options.evidence = vignette | with_findings) and is recorded per run."
            ),
            sampling_procedure=self.sampling_note(),
            metrics_description={
                "diagnosis_match": "1 if the answer equals or contains any accepted diagnosis "
                "(primary for the generation subtask)",
                "exact_match": "strict normalized equality with the primary diagnosis",
                "token_f1": "bag-of-tokens F1 against the primary diagnosis",
                "rouge_l": "LCS F-measure against the primary diagnosis",
                "accuracy": "selection subtask: 1 if the chosen option is the gold diagnosis",
                "diagnosis_match_judged": "LLM-judge verdict (only when engine.judge.enabled)",
            },
            primary_metric="accuracy" if self._subtask == "selection" else "diagnosis_match",
            decisions=[
                "Default table is pubmed_reviewed (human-reviewed) rather than the larger "
                "generated table; options.table switches.",
                "Default evidence is the vignette alone, which is the closest single-turn analogue "
                "of a viva's opening turn; with_findings reveals the structured examination and "
                "investigations as an upper bound.",
                "The selection subtask uses only the case's own differentials as distractors and "
                f"skips cases with fewer than two ({self._with_differentials} of the table's rows "
                "have enough) -- distractors are never invented.",
                "All diagnoses in the row's list count as correct, since the dataset records "
                "multi-part diagnoses.",
            ],
            caveats=[
                "Because the differentials column is empty for many rows, the selection subtask "
                "covers a biased subset; the generation default avoids that.",
                "Cases derive from published reports and may be memorized.",
            ],
            statistics={
                **self.base_statistics(),
                "subtask": self._subtask,
                "evidence": str(self.context.option("evidence", "vignette")),
                "rows_with_differentials": getattr(self, "_with_differentials", 0),
            },
        )
