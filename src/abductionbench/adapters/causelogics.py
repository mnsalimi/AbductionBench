"""CauseLogics: decide whether a proposed cause actually explains a phenomenon.

Source: https://github.com/sternstude/CauseJudger (the ``CauseLogics/`` tree)

**The task.**  An item gives a set of premises about individuals, a set of
if-then rules, an observed ``Phenomenon`` ("Anne is trustworthy"), and one
``PossibleCause`` ("Anne is excited").  The model decides whether that cause,
taken together with the premises and rules, would in fact produce the
phenomenon.  It is abductive validation rather than abductive search: the
hypothesis is supplied and its explanatory adequacy is what is judged.

**Difficulty levels.**  The release ships four, and all four are the same task
at increasing depth -- more rules to chain, more distracting premises -- so all
four are used and results are reported per level.  ``options.levels`` narrows
to a subset.

**Scoring.**  A binary decision against the release's own ``Label``, so
accuracy, checked mechanically.  Chance is 0.5, and because a model that always
answers "yes" would score 0.5 without reasoning at all, the rate at which it
says yes is reported beside the score.
"""

from __future__ import annotations

from typing import Any

from ..core.adapter import SkippedDataset
from ..core.metrics import aggregate_mean_metrics
from ..core.types import AdapterDocumentation, ModelResponse, SampleScore, SampleSpec
from . import _common as C
from ._base import PooledDatasetAdapter, selection_score

REPO_URL = "https://github.com/sternstude/CauseJudger.git"
DEFAULT_LEVELS = (1, 2, 3, 4)
#: The two answers, in a fixed order so the gold's position is not a function
#: of the gold.
OPTIONS = ("Yes, that cause would produce the phenomenon.",
           "No, that cause would not produce the phenomenon.")


class CauseLogicsAdapter(PooledDatasetAdapter):
    """Validate a proposed cause against a set of premises and rules."""

    adapter_version = "1.0"

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. Here the explanation is already "
        "proposed, and your job is to test it: decide whether the proposed cause, taken "
        "together with the premises and the rules, would actually produce the observed "
        "phenomenon. A cause that is merely consistent with the phenomenon, but does not "
        "lead to it through the rules, does not count."
    )
    options_heading = "Answer options:"
    data_delivery_mode = "static"
    objective_metrics = True
    selection_cardinality = "single"
    hypothesis_modes = ("selection",)
    hypothesis_mode_options = {"selection": {"subtask": "selection"}}
    table_hypothesis_mode = "Selection"
    primary_metric = "accuracy"

    def load_items(self) -> list[dict[str, Any]]:
        root = C.ensure_git_repo(
            REPO_URL, self.context.data_dir / "repo", depth=1, offline=self.context.offline
        )
        base = root / "CauseLogics"
        if not base.is_dir():
            raise SkippedDataset(f"{base} is not in the clone of {REPO_URL}")
        levels = [int(x) for x in (self.context.option("levels", None) or DEFAULT_LEVELS)]
        items: list[dict[str, Any]] = []
        per_level: dict[int, int] = {}
        for level in levels:
            directory = base / f"Level {level}"
            if not directory.is_dir():
                self.log.warning("CauseLogics has no %s", directory)
                continue
            before = len(items)
            for path in sorted(directory.glob("*.jsonl")):
                for row in C.read_jsonl(path):
                    items.append({**row, "level": level})
            per_level[level] = len(items) - before
        if not items:
            raise SkippedDataset(f"no CauseLogics rows found under {base}")
        self.split_used = "CauseLogics " + ", ".join(
            f"Level {level} ({count} items)" for level, count in sorted(per_level.items())
        )
        return items

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        premises = [C.normalize_whitespace(p) for p in (item.get("Premises") or [])]
        rules = [C.normalize_whitespace(r) for r in (item.get("Rules") or [])]
        phenomenon = C.normalize_whitespace(item.get("Phenomenon"))
        cause = C.normalize_whitespace(item.get("PossibleCause"))
        label = item.get("Label")
        if not premises or not rules or not phenomenon or not cause:
            return None
        if isinstance(label, str):
            label = label.strip().lower() in {"true", "yes", "1"}
        if not isinstance(label, bool):
            return None
        labels = C.choice_labels(len(OPTIONS))
        gold_index = 0 if label else 1
        return SampleSpec(
            sample_id=C.stable_id("causelogics", item["level"], index),
            fields={
                "context": (
                    "Premises:\n" + "\n".join(f"- {p}" for p in premises)
                    + "\n\nRules:\n" + "\n".join(f"- {r}" for r in rules)
                ),
                "observation": f"Observed phenomenon: {phenomenon}",
                "question": f"Would this cause produce it? Proposed cause: {cause}",
                "options": list(OPTIONS),
                "option_labels": labels,
            },
            reference={"gold_label": labels[gold_index], "gold": OPTIONS[gold_index]},
            task_kind="selection",
            metadata={"level": item["level"], "label": label},
        )

    def score(
        self,
        sample: SampleSpec,
        response: ModelResponse,
        *,
        output_contract: dict[str, Any] | None = None,
    ) -> SampleScore:
        score = selection_score(
            response,
            labels=sample.fields["option_labels"],
            gold_label=sample.reference["gold_label"],
            output_contract=output_contract,
            metric_name="accuracy",
        )
        score.metrics[f"accuracy_level{sample.metadata['level']}"] = score.metrics.get(
            "accuracy", 0.0
        )
        # A model that always answers yes scores 0.5 without reasoning, so how
        # often it says yes sits next to the score rather than inside it.
        chosen = str(score.prediction or "")
        score.metrics["said_yes_rate"] = 1.0 if chosen == sample.fields["option_labels"][0] else 0.0
        return score

    def aggregate(self, scores) -> dict[str, float]:
        return aggregate_mean_metrics([score.metrics for score in scores])

    def documentation(self) -> AdapterDocumentation:
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="CauseLogics",
            domain="Formal Reasoning: Abductive Cause Validation",
            source_url="https://github.com/sternstude/CauseJudger",
            processing_mode="Selection",
            split_used=getattr(self, "split_used", "CauseLogics"),
            abductive_subset=(
                "The whole CauseLogics tree. Every item is abductive validation: a "
                "phenomenon, a proposed cause, and the question of whether that cause "
                "would produce it under the given premises and rules. All four difficulty "
                "levels are the same task at increasing rule depth, so all four are used "
                "and reported separately; options.levels narrows to a subset."
            ),
            sampling_procedure=self.sampling_note(),
            metrics_description={
                "accuracy": (
                    "(PRIMARY, higher is better, 0-1) 1.0 when the yes/no decision matches "
                    "the release's Label. Two options, so chance is 0.5."
                ),
                "accuracy_level<n>": (
                    "the same metric per difficulty level -- how the score falls as more "
                    "rules have to be chained"
                ),
                "said_yes_rate": (
                    "(diagnostic, no direction) how often the model accepted the proposed "
                    "cause. A model answering yes to everything scores 0.5 on accuracy "
                    "without reasoning; this is what makes that visible."
                ),
                "parse_failure_rate": (
                    "(lower is better, 0-1) fraction of responses no option could be read "
                    "from; these score 0 and are counted separately from being wrong."
                ),
                "self_consistency_<metric>":
                "Every metric also gets a self_consistency_ counterpart: the plurality answer "
                "over modes.repeats samples of the same record, read off those samples rather "
                "than bought again. Available because this dataset's answers are checkable and "
                "so can coincide.",
            },
            primary_metric="accuracy",
            decisions=[
                "Kept the yes/no decision as a two-option selection with the answers in a "
                "fixed order, so the gold's position never depends on what the gold is.",
                "Used all four difficulty levels and reported each, since they differ only "
                "in rule depth and a pooled mean would hide where the score comes from.",
                "Reported said_yes_rate beside accuracy, because a constant answer scores "
                "chance on a balanced binary task.",
            ],
            caveats=[
                "Two options means chance is about 0.5; read accuracy against that floor.",
                "Names and properties are synthetic, which removes world knowledge as a "
                "confound but makes the surface unfamiliar.",
            ],
            statistics={**self.base_statistics()},
        )
