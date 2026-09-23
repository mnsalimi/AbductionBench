"""Reusable adapter bases and scoring helpers for child adapters.

Two things every adapter in the suite needs, factored once:

* :class:`PooledDatasetAdapter` -- the deterministic-draw idiom. A subclass says
  how to *load* raw items and how to turn one into a
  :class:`~abductionbench.core.types.SampleSpec`; the base class handles the
  seeded shuffle, taking the first N, and serving replacements from the same
  ordered pool when the engine has to drop an oversize sample.  Replacements are
  therefore never re-draws of an item already seen.
* scoring helpers -- ``selection_score`` and ``text_match_score`` implement the
  two shapes almost every benchmark here reduces to (pick a label; produce text
  that should name the gold hypothesis), including the "unparseable is not the
  same as wrong" rule.

Dataset-specific logic stays in the dataset's own module.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from collections.abc import Sequence
from typing import Any

from ..core.adapter import DatasetAdapter, SkippedDataset
from ..core.metrics import (
    aggregate_mean_metrics,
    any_exact_match,
    contains_match,
    exact_match,
    extract_answer_span,
    extract_choice_label,
    extract_choice_labels,
    rouge_l,
    set_prf,
    token_f1,
)
from ..core.types import ChatMessage, ModelResponse, SampleScore, SampleSpec
from ._prompting import PromptParts, build_messages

logger = logging.getLogger(__name__)

__all__ = [
    "PooledDatasetAdapter",
    "PromptParts",
    "multi_selection_score",
    "selection_score",
    "text_match_score",
    "unparsed_score",
]


class PooledDatasetAdapter(DatasetAdapter):
    """Adapter over a materialized list of raw items, drawn deterministically."""

    #: Set by :meth:`load_items` implementations for the run documentation.
    split_used: str = ""
    #: Total items available in the chosen split, before sampling.
    split_size: int = 0

    def __init__(self, context: Any):
        super().__init__(context)
        self._pool: list[Any] = []
        self._built: list[str] = []

    # -- subclass hooks ------------------------------------------------- #

    @abstractmethod
    def load_items(self) -> list[Any]:
        """Materialize and return the raw items of the chosen split.

        Implementations should set :attr:`split_used` and may raise
        :class:`~abductionbench.core.adapter.SkippedDataset`.
        """

    @abstractmethod
    def make_sample(self, item: Any, index: int) -> SampleSpec | None:
        """Convert one raw item into a sample, or ``None`` to skip it."""

    # -- lifecycle ------------------------------------------------------ #

    def prepare(self) -> None:
        items = self.load_items()
        if not items:
            raise SkippedDataset("no items found in the chosen split")
        self.split_size = len(items)
        # One seeded shuffle defines both the evaluation set (its prefix) and the
        # replacement order (the rest), so the draw is fully reproducible.
        self._pool = self.ordered_pool(items)

    def build_samples(self) -> list[SampleSpec]:
        samples: list[SampleSpec] = []
        skipped = 0
        offset = max(0, int(getattr(self.context, "sample_offset", 0) or 0))
        #: Pool positions of the samples this draw actually took, so the
        #: manifest can say which records were used rather than only how many.
        self._drawn_pool_indices: list[int] = []
        for index, item in enumerate(self._pool):
            if len(samples) >= self.context.sample_size:
                break
            sample = self._safe_make(item, index)
            if sample is None:
                continue
            if skipped < offset:
                # Counted as built and then dropped, which is what makes the
                # offset line up with an earlier run: that run consumed exactly
                # these, in exactly this order, so skipping the same number of
                # BUILT samples leaves a set disjoint from it. Counting pool
                # positions instead would drift whenever an item fails to build.
                skipped += 1
                continue
            samples.append(sample)
            self._drawn_pool_indices.append(index)
        self._built = [s.sample_id for s in samples]
        return samples

    def draw_record(self) -> dict[str, Any]:
        """Exactly which records this draw took, and how to take the next set.

        Written to the run's ``sample_manifest.jsonl``. The ids are the point:
        "50 records" is not a claim anyone can check, and a later run that means
        to extend this one rather than repeat it has to be able to see what this
        one used.
        """
        offset = max(0, int(getattr(self.context, "sample_offset", 0) or 0))
        return {
            "dataset_id": self.dataset_id,
            "split": self.split_used,
            "split_size": self.split_size,
            "seed": self.context.seed,
            "salt": self.dataset_id,
            "sample_offset": offset,
            "sample_size": self.context.sample_size,
            "n_drawn": len(self._built),
            # Where a run that must not overlap this one should start.
            "next_offset": offset + len(self._built),
            "adapter_version": self.adapter_version,
            "sample_ids": list(self._built),
            "pool_indices": list(getattr(self, "_drawn_pool_indices", [])),
        }

    def replacement_samples(self, count: int, exclude: set[str]) -> list[SampleSpec]:
        """Replacements for oversize items, expanded for the active modes.

        The engine expands the main sample set once, before planning; a
        replacement arrives after that, so it expands itself.
        """
        out: list[SampleSpec] = []
        for index, item in enumerate(self._pool):
            if len(out) >= count:
                break
            sample = self._safe_make(item, index)
            if sample is None or sample.sample_id in exclude:
                continue
            out.append(sample)
        return self.expand_for_modes(out)

    # -- prompts: mechanics here, wording in the dataset's own adapter --- #

    #: What a well-formed answer looks like for this dataset, in its own terms
    #: ("one short sentence", "a single diagnosis").  Rendered into the closing
    #: line.  A sample may override it with an ``answer_format`` field.
    answer_format: str = ""

    #: What the *task* requires, one clause per item, rendered as a
    #: "Requirements:" list inside the task block.  Declared per dataset because
    #: the requirements are the dataset's ("use only the allowed predicates",
    #: "do not restate the observation"), and declared as *data* because that is
    #: what keeps them comparable: every generation task in the suite states its
    #: demands the same way, so a difference in score is a difference in
    #: difficulty rather than in how firmly the instruction was worded.
    #:
    #: Three things that used to live here do not any more, because each has
    #: exactly one home now: the answer's shape belongs to :attr:`answer_format`,
    #: "output only the <answer>" is said once by the closing for every dataset,
    #: and whether the response may reason is said only by the mode instruction.
    #: A clause here is rendered unchanged in every prompt mode and every
    #: selection mode, so anything mode-specific does not belong in it.
    task_requirements: tuple[str, ...] = ()

    #: Set by a dataset whose answer is several lines (see
    #: PromptParts.answer_is_block); it then closes with an answer block rather
    #: than "on the last line", which such an answer cannot satisfy.
    answer_is_a_block: bool = False

    #: Heading above the candidate list, in the dataset's own words.
    options_heading: str = ""

    def prompt_parts(self, sample: SampleSpec) -> PromptParts:
        """The dataset-owned content of this sample's prompt.

        The default reads the fields adapters already produce, falling back to
        the class-level answer shape.  An adapter whose prompt is published by
        its own benchmark -- every interactive one -- overrides this and returns
        that wording instead.
        """
        fields = sample.fields
        requirements = fields.get("requirements")
        return PromptParts(
            system=self.system_prompt_for(sample),
            observation=str(fields.get("observation", "") or ""),
            context=str(fields.get("context", "") or ""),
            question=str(fields.get("question", "") or ""),
            instructions=str(fields.get("instructions", "") or ""),
            answer_format=str(fields.get("answer_format") or self.answer_format or ""),
            options=[str(o) for o in (fields.get("options") or [])],
            option_labels=[str(o) for o in (fields.get("option_labels") or [])],
            options_heading=str(fields.get("options_heading") or self.options_heading or ""),
            requirements=[str(c) for c in (requirements or self.task_requirements)],
            answer_is_block=self.answer_is_a_block,
        )

    def build_messages(self, sample: SampleSpec) -> tuple[list[ChatMessage], dict[str, Any]]:
        if sample.messages_override is not None:
            return list(sample.messages_override), {}
        return build_messages(self.prompt_parts(sample), self.context.modes)

    def _safe_make(self, item: Any, index: int) -> SampleSpec | None:
        try:
            return self.make_sample(item, index)
        except Exception as exc:  # noqa: BLE001 - one malformed row must not stop a run
            self.log.warning("skipping item %d (%s): %s", index, type(exc).__name__, exc)
            return None

    # -- default aggregation -------------------------------------------- #

    def aggregate(self, scores: Sequence[SampleScore]) -> dict[str, float]:
        """Mean of every per-sample metric (override for anything else)."""
        return aggregate_mean_metrics([score.metrics for score in scores])

    # -- documentation helpers ------------------------------------------ #

    def base_statistics(self) -> dict[str, Any]:
        return {
            "split_size": self.split_size,
            "requested_sample_size": self.context.sample_size,
            "seed": self.context.seed,
            "adapter_version": self.adapter_version,
        }

    def sampling_note(self) -> str:
        return (
            f"seeded shuffle of the whole split (seed={self.context.seed}, "
            f"salt='{self.dataset_id}'), then the first {self.context.sample_size} items; "
            "oversize items are replaced by the next unused item of the same shuffle"
        )


# --------------------------------------------------------------------------- #
# scoring helpers
# --------------------------------------------------------------------------- #


def unparsed_score(metric_names: Sequence[str], **details: Any) -> SampleScore:
    """A score for a response no prediction could be extracted from.

    Metrics are zeroed *and* ``parse_ok`` is false, so the engine can report
    ``parse_failure_rate`` separately from being wrong.
    """
    return SampleScore(
        metrics={name: 0.0 for name in metric_names},
        prediction=None,
        parse_ok=False,
        details=details,
    )


def judged_only_score(
    response: ModelResponse,
    *,
    metric: str,
    output_contract: dict[str, Any] | None = None,
    extra_metrics: dict[str, float] | None = None,
    details: dict[str, Any] | None = None,
) -> SampleScore:
    """Score for a generation task that only an LLM judge can grade.

    These datasets have no answer key: the reference is one human-written
    explanation among many that would have been just as good.  Character and
    n-gram overlap with it is not a weak measure of correctness, it is a
    measure of something else -- it punishes a correct paraphrase and rewards a
    wrong sentence that reuses the reference's words -- so none is emitted.

    What this produces is the *parse*: the extracted answer as ``prediction``,
    which is what the judge is handed, and ``metric`` seeded at 0.0 so the
    number always exists.  :meth:`DatasetAdapter.apply_judge` overwrites it
    with the verdict.  Seeding rather than omitting matters: an empty response
    is never sent to the judge, and it should score zero rather than vanish
    from the mean.
    """
    answer = extract_answer_span(response.text, output_contract)
    if not answer:
        return unparsed_score([metric], **(details or {}))
    metrics = {metric: 0.0}
    metrics.update(extra_metrics or {})
    return SampleScore(metrics=metrics, prediction=answer, details=details or {})


def apply_judged_metric(score: SampleScore, verdict: Any, metric: str) -> SampleScore:
    """Fold a judge verdict into ``metric`` and every stratum of it.

    A dataset that reports its score per domain, per task or per popularity
    stratum seeds those alongside the base metric (``hypothesis_judged`` and
    ``hypothesis_judged_chemistry``).  They are the same verdict viewed through
    a filter, so one rule sets them all rather than each adapter remembering
    which strata it declared.
    """
    metrics = dict(score.metrics)
    details = {**score.details, "judge_label": getattr(verdict, "label", None)}

    value = getattr(verdict, "score", None)
    if value is None and not getattr(verdict, "parsed", False):
        # NOT JUDGED IS NOT WRONG.
        #
        # This used to fall back to `verdict.positive`, which for a verdict
        # that parsed nothing is False, so an unreadable judge reply was
        # written down as a confident 0 -- indistinguishable from a judge that
        # read the answer and rejected it. On a dataset whose only metric is
        # the judge's verdict that is the whole score.
        #
        # Dropping the metric is the sanctioned way to say "no measurement
        # here": `aggregate_mean_metrics` skips keys a sample does not carry
        # rather than averaging them as zero, which is what its own docstring
        # promises. The sample still appears, with the reason on it.
        for name in list(metrics):
            if name == metric or name.startswith(f"{metric}_"):
                metrics.pop(name)
        details["judge_unparsed"] = True
        raw = getattr(verdict, "raw", "") or ""
        if raw:
            details["judge_unparsed_raw"] = raw
        return SampleScore(
            metrics=metrics,
            prediction=score.prediction,
            parse_ok=score.parse_ok,
            details=details,
        )

    value = float(value) if value is not None else (
        1.0 if getattr(verdict, "positive", False) else 0.0
    )
    for name in list(metrics):
        if name == metric or name.startswith(f"{metric}_"):
            metrics[name] = value
    metrics[metric] = value
    return SampleScore(
        metrics=metrics,
        prediction=score.prediction,
        parse_ok=score.parse_ok,
        details=details,
    )


def selection_score(
    response: ModelResponse,
    *,
    labels: Sequence[str],
    gold_label: str,
    output_contract: dict[str, Any] | None = None,
    metric_name: str = "accuracy",
    extra_metrics: dict[str, float] | None = None,
) -> SampleScore:
    """Score a multiple-choice (hypothesis selection) response.

    The same function scores all three selection modes, which is what makes
    their numbers comparable:

    * ``SCS`` -- one label is read and compared with the gold label.
    * ``MCS`` -- a *set* of labels is read, and ``metric_name`` is 1.0 only when
      that set is exactly the gold one.  On a single-answer benchmark that means
      naming the gold hypothesis **and nothing else**: a model that hedges by
      selecting three options has not answered the question, and scoring it as
      correct because the gold label was among them would measure recall while
      claiming to measure accuracy.  ``set_f1`` is reported beside it for the
      partial-credit view.
    * ``BOV`` -- the engine rebuilds a selected set from the per-hypothesis
      yes/no answers and hands it here as a synthetic multi-label response, so a
      BOV score and an MCS score come out of this same code path.

    The mode is read from the prompt's own ``output_contract`` rather than from
    a flag, so an adapter does not have to know which mode it is being run in.
    """
    if (output_contract or {}).get("style") == "multi_label":
        return multi_selection_score(
            response,
            labels=labels,
            gold_labels=[gold_label],
            output_contract=output_contract,
            metric_name=metric_name,
            extra_metrics=extra_metrics,
        )
    chosen = extract_choice_label(response.text, labels, output_contract)
    if chosen is None:
        score = unparsed_score([metric_name], raw=response.text[:300])
        score.metrics.update(extra_metrics or {})
        return score
    correct = float(str(chosen).strip().upper() == str(gold_label).strip().upper())
    metrics = {metric_name: correct}
    metrics.update(extra_metrics or {})
    return SampleScore(metrics=metrics, prediction=chosen, details={"gold": gold_label})


def multi_selection_score(
    response: ModelResponse,
    *,
    labels: Sequence[str],
    gold_labels: Sequence[str],
    output_contract: dict[str, Any] | None = None,
    metric_name: str = "accuracy",
    extra_metrics: dict[str, float] | None = None,
) -> SampleScore:
    """Score a selection answered as a set (MCS, or a rebuilt BOV set)."""
    chosen = extract_choice_labels(response.text, labels, output_contract)
    if chosen is None:
        score = unparsed_score(
            [metric_name, "set_exact_match", "set_f1"], raw=response.text[:300]
        )
        score.metrics.update(extra_metrics or {})
        return score
    selected = {str(label).strip().upper() for label in chosen}
    gold = {str(label).strip().upper() for label in gold_labels if str(label).strip()}
    prf = set_prf(selected, gold)
    exact = float(selected == gold)
    metrics = {
        metric_name: exact,
        # The same number under the name that says what it is. Under MCS the
        # primary metric is an exact *set* match, not the per-item accuracy the
        # word suggests, and a sheet that shows "accuracy 0.41" beside
        # "set_f1 0.68" invites the reader to think the first is a stricter view
        # of the second rather than a different measure entirely.
        "set_exact_match": exact,
        "set_f1": prf["f1"],
        "set_precision": prf["precision"],
        "set_recall": prf["recall"],
        # How many hypotheses the model committed to. Against a single-answer
        # benchmark this is the tell for hedging: the gold count is 1, so a mean
        # well above 1 says the score is being propped up by recall.
        "n_selected": float(len(selected)),
    }
    metrics.update(extra_metrics or {})
    return SampleScore(
        metrics=metrics,
        prediction=",".join(sorted(selected)) or "none",
        details={"gold": ",".join(sorted(gold)), "n_gold": len(gold)},
    )


def text_match_score(
    response: ModelResponse,
    *,
    gold: str,
    accepted: Sequence[str] | None = None,
    output_contract: dict[str, Any] | None = None,
    primary: str = "match",
    extra_metrics: dict[str, float] | None = None,
) -> SampleScore:
    """Score a free-form (hypothesis generation) response against a gold string.

    Emits several complementary views, because a single number cannot describe
    free-form abduction:

    ``match``       1.0 if the answer equals, or contains, an accepted gold string
    ``exact_match`` strict normalized equality with the gold string
    ``token_f1``    bag-of-tokens overlap with the gold string
    ``rouge_l``     longest-common-subsequence F-measure with the gold string

    ``accepted`` lists alternative gold surface forms (abbreviations, synonyms)
    that the dataset itself provides; it never invents them.
    """
    answer = extract_answer_span(response.text, output_contract)
    if not answer:
        return unparsed_score(
            [primary, "exact_match", "token_f1", "rouge_l"], raw=response.text[:300]
        )
    candidates = [gold, *(accepted or [])]
    strict = any_exact_match(answer, candidates)
    loose = max(
        [strict] + [contains_match(answer, candidate) for candidate in candidates if candidate]
    )
    metrics = {
        primary: float(loose),
        "exact_match": float(exact_match(answer, gold)),
        "token_f1": token_f1(answer, gold),
        "rouge_l": rouge_l(answer, gold)["f"],
    }
    metrics.update(extra_metrics or {})
    return SampleScore(
        metrics=metrics,
        prediction=answer,
        details={"gold": str(gold)[:300]},
    )


#: Prefix every project-specific proxy score carries.
#:
#: These are not the papers' own evaluation protocols and must never be read as
#: though they were.  A reader skimming a sheet sees the prefix before the name,
#: and every one of them is documented as a proxy in its adapter's caveats.
PROXY_PREFIX = "proxy_"


def apply_proxy_score(
    score: SampleScore,
    verdict: Any,
    metric: str,
    *,
    detail_key: str = "proxy_judgement",
) -> SampleScore:
    """Fold a judge verdict into a proxy metric, or record why it is absent.

    The verdict is **binary**: the proxy templates return 1 or 0 and nothing
    between, so the metric is a rate -- the fraction of records the judge
    accepted -- rather than an average quality. (It was a 0-5 rubric until
    2026-09-21; the clamp below is what remains of that, and it now only
    guards against a malformed verdict.)

    What still makes this different from :func:`apply_judged_metric` is what
    happens when there is **no** verdict: the metric is left **absent** rather
    than seeded 0.0. A 0 here is a judgement the judge actually made -- it
    read the answer and rejected it -- and recording one for a call that never
    came back would put fabricated rejections into the rate. The sample is
    marked instead, so a blank is legible as unjudged and the coverage of
    these metrics can be counted.

    The judge's own reply is kept on the record either way: a proxy score that
    cannot be traced back to what the judge said is not auditable, and these are
    the scores most in need of auditing.
    """
    metrics = dict(score.metrics)
    details = dict(score.details or {})
    value = getattr(verdict, "score", None)
    raw = getattr(verdict, "raw", "") or ""
    judge_details = dict(getattr(verdict, "details", {}) or {})

    if value is None or not getattr(verdict, "parsed", False):
        # Absent, not zero. Every stratum of it goes too, or the strata would
        # report a score the base metric does not have.
        for name in list(metrics):
            if name == metric or name.startswith(f"{metric}_"):
                metrics.pop(name)
        details[detail_key] = {
            "status": "unjudged",
            "reason": "the judge returned no usable score",
            "judge_reply": raw,
            **judge_details,
        }
        return SampleScore(
            metrics=metrics,
            prediction=score.prediction,
            parse_ok=score.parse_ok,
            details=details,
        )

    value = max(0.0, min(1.0, float(value)))
    for name in list(metrics):
        if name == metric or name.startswith(f"{metric}_"):
            metrics[name] = value
    metrics[metric] = value
    details[detail_key] = {
        "status": "judged",
        "score": value,
        "judge_reply": raw,
        **judge_details,
    }
    return SampleScore(
        metrics=metrics,
        prediction=score.prediction,
        parse_ok=score.parse_ok,
        details=details,
    )
