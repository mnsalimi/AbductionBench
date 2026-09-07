"""Open Problems 2024+: frontier research problems resolved after a 2023 cutoff.

Source: ``assets/open_problems_2024/problems.json``, built by
``tools/build_open_problems_dataset.py`` from two documents checked in beside it.
Unlike every other adapter in the suite this dataset is *authored here* rather
than downloaded, so the data ships with the repository.

**What is abductive here, and what is not.** The dataset carries three prompt
modes per problem, and they are not equally abductive:

* ``direction`` -- "is this conjecture true? answer plus a probability."  This is
  Stage-2 **single-hypothesis evaluation** in the framework's own terms: judge one
  proposed hypothesis (the conjecture) against the evidence available before its
  resolution and return a binary plus a graded plausibility.  Defeasible, since a
  new partial result can overturn the judgement; non-monotonic; ampliative,
  because the answer is not entailed by the evidence -- which is precisely why it
  is a guess and not a proof.  The honest caveat is that what is judged is the
  truth of a proposition rather than the explanatory power of a hypothesis about
  observations, which places it in the plausible-reasoning family at the edge of
  abduction rather than at its centre.
* ``strategy`` -- "name the tools, dichotomies and intermediate statements you
  would establish."  Stage-1 **knowledge completion**: which missing statements,
  once available, would make the target derivable.  This is the closest fit of
  the three and the primary source calls it the most discriminating signal.
* ``resolution`` -- "produce the proof."  **Deduction, not abduction.**  Kept
  because its hallucinated-proof rate is a valuable calibration measure, and
  excluded from the abduction score.

A fourth mode, ``leakage_probe``, asks a model what it knows about the problem's
status.  It measures contamination rather than reasoning, and it matters here:
every resolution date is 2024 or later, so a model whose cutoff postdates a
problem can simply recall the answer.  Use ``options.model_cutoff`` to keep only
items resolved after the model under test was trained, and run the probe to
measure what leaked anyway.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

from ..core.adapter import SkippedDataset
from ..core.metrics import (
    aggregate_mean_metrics,
    brier_score,
    contains_match,
    extract_answer_span,
    extract_choice_label,
    mean,
    rouge_l,
)
from ..core.types import (
    AdapterDocumentation,
    ChatMessage,
    ModelResponse,
    SampleScore,
    SampleSpec,
)
from . import _common as C
from ._base import PooledDatasetAdapter, unparsed_score

#: Mode -> (prompt key in the dataset, task_kind bound to a template).
MODES: dict[str, tuple[str, str]] = {
    "direction": ("a", "direction_judgment"),
    "resolution": ("b", "full_resolution"),
    "strategy": ("c", "solution_strategy"),
    "numeric_forecast": ("d", "direction_judgment"),
}
#: Modes included by each ``options.subtask`` setting.
SUBTASKS: dict[str, tuple[str, ...]] = {
    "both": ("direction", "strategy"),
    "direction": ("direction",),
    "strategy": ("strategy",),
    "resolution": ("resolution",),
    "all": ("direction", "strategy", "resolution"),
    "leakage_probe": ("leakage_probe",),
}
#: Phrases that mark a response as claiming a finished proof.
_COMPLETION_CLAIMS = (
    "this completes the proof", "we have proved", "we have shown", "q.e.d", "qed",
    "the proof is complete", "this proves the conjecture", "hence the conjecture holds",
    "therefore the conjecture is proved", "this establishes the conjecture",
)
#: Phrases that mark an honest refusal to resolve.
_ABSTENTIONS = (
    "i cannot resolve", "cannot be resolved", "i am unable to", "this remains open",
    "i do not have a proof", "i can not resolve", "no complete proof",
    "beyond current", "i cannot provide a complete", "this is still open",
)


class OpenProblemsAdapter(PooledDatasetAdapter):
    """Frontier open problems, judged for direction and for solution strategy."""

    adapter_version = "1.0"

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. You are given a research problem "
        "that was open as of your knowledge cutoff. Judge it on the evidence available at the "
        "time and say how confident you are; a calibrated 'probably not' is worth more than a "
        "confident guess."
    )
    data_delivery_mode = "static"

    answer_constraints = (
        "output only the answer, with nothing before or after it",
        "do not use introductory phrases or commentary",
    )
    objective_metrics = True
    selection_cardinality = None
    _repeats = 1

    @property
    def primary_metric(self) -> str:
        """The headline metric depends on which mode is being run.

        Only the direction and strategy modes are abductive, so only they
        produce ``abduction_score``; reporting it as the primary metric for the
        other two would print a NaN in every table.  The substitutes are the
        metrics those modes exist to measure.
        """
        subtask = str(self.context.option("subtask", "both")).lower()
        return {
            "leakage_probe": "leakage_rate",
            "resolution": "hallucinated_proof_rate",
        }.get(subtask, "abduction_score")

    # ------------------------------------------------------------------ #
    # prompts -- this dataset's own, one per mode
    # ------------------------------------------------------------------ #

    #: A different instruction per mode, because the four modes are four
    #: different jobs.  These were previously YAML templates bound by the
    #: engine; they live here now, with the adapter that knows what they mean.
    SYSTEM_PROMPTS = {
        "direction": (
            "You are a research mathematician and scientist assessing whether a stated "
            "conjecture holds, using only the evidence available to you: known partial "
            "results, heuristics, analogous cases, and the structure of the problem. You "
            "will not be able to prove your answer. Commit to the most plausible verdict "
            "anyway, and report how confident you actually are -- a well-judged 0.55 is "
            "more useful than a reflexive 0.95."
        ),
        "strategy": (
            "You are a research mathematician and scientist planning an attack on an open "
            "problem. Name specific machinery, not general advice: the theorems you would "
            "invoke, the structural dichotomies you would set up, the intermediate "
            "statements you would try to establish first, and where you expect the "
            "difficulty to concentrate. \"Use induction\" or \"try a computer search\" is "
            "not an answer."
        ),
        "resolution": (
            "You are a research mathematician and scientist asked to resolve an open "
            "problem outright. Give a complete argument if you have one. If you do not, "
            "say so plainly: an honest statement that you cannot resolve the problem is "
            "worth more than a proof-shaped answer with a gap in it."
        ),
        "leakage_probe": (
            "Report what you know about the status of a research problem as of your "
            "knowledge cutoff. If it is unresolved, say so. If you believe it has been "
            "resolved, say by whom and when."
        ),
    }

    def system_prompt_for(self, sample: SampleSpec) -> str:
        mode = (sample.reference or {}).get("mode", "direction")
        if mode == "numeric_forecast":
            mode = "direction"
        return self.SYSTEM_PROMPTS.get(mode, self.system_prompt)

    def build_messages(self, sample: SampleSpec) -> tuple[list[ChatMessage], dict[str, Any]]:
        """The dataset's own wording, kept verbatim; the modes add only a contract.

        The problem statements are the sources' own and are deliberately neutral
        about which way each conjecture resolved, so nothing here rephrases them:
        the observation goes in as written, and only the answer format is added.
        """
        mode = (sample.reference or {}).get("mode", "direction")
        body = [str(sample.fields.get("observation", ""))]
        contract: dict[str, Any] = {"answer_prefix": "Answer:", "strip_markdown": True}
        answer_format = str(sample.fields.get("answer_format", "your answer"))

        if mode in ("direction", "numeric_forecast"):
            body.append(
                "Finish with exactly these two lines and nothing after them:\n"
                f"Answer: <{answer_format}>\n"
                "Confidence: <a probability between 0 and 1>"
            )
            contract.update(
                {"confidence_prefix": "Confidence:", "style": "single_label_with_confidence"}
            )
        elif mode == "strategy":
            body.append(
                "Then close with a single line listing the ingredients your route "
                f"depends on:\nAnswer: <{answer_format}>"
            )
            contract["style"] = "free_form"
        elif mode == "resolution":
            body.append(
                "If you can resolve the problem, give the argument and then state your "
                f"conclusion as:\nAnswer: <{answer_format}>\n"
                "If you cannot, say so explicitly instead of producing an incomplete proof."
            )
            contract["style"] = "free_form"
        else:  # leakage_probe
            contract["style"] = "free_form"

        labels = sample.fields.get("option_labels") or []
        if labels:
            contract["option_labels"] = list(labels)
        return (
            [
                ChatMessage(role="system", content=self.system_prompt_for(sample)),
                ChatMessage(role="user", content="\n\n".join(part for part in body if part)),
            ],
            contract,
        )

    # ------------------------------------------------------------------ #
    # data
    # ------------------------------------------------------------------ #

    def _dataset_path(self) -> Path:
        configured = self.context.option("dataset_path")
        if configured:
            return Path(str(configured))
        # The dataset ships with the repository rather than being downloaded.
        packaged = (
            Path(__file__).resolve().parents[3]
            / "assets"
            / "open_problems_2024"
            / "problems.json"
        )
        return packaged

    def load_items(self) -> list[dict[str, Any]]:
        path = self._dataset_path()
        if not path.exists():
            raise SkippedDataset(
                f"bundled dataset not found at {path}; run "
                "'python tools/build_open_problems_dataset.py' to rebuild it"
            )
        payload = C.read_json(path)
        problems = payload.get("problems") or []
        if not problems:
            raise SkippedDataset(f"{path} contains no problems")
        self._meta = {k: v for k, v in payload.items() if k != "problems"}

        include_held_out = bool(self.context.option("include_held_out", False))
        cutoff = self.context.option("model_cutoff")
        subtask = str(self.context.option("subtask", "both"))
        if subtask not in SUBTASKS:
            raise SkippedDataset(
                f"unknown options.subtask {subtask!r}; expected one of {sorted(SUBTASKS)}"
            )
        modes = SUBTASKS[subtask]

        self._excluded_by_cutoff: list[str] = []
        self._held_out_skipped = 0
        # Repeats exist because this dataset is small enough for server-side
        # nondeterminism to dominate it: two temperature-0, fixed-seed runs of
        # the 15 direction items disagreed on 5 of them, because vLLM's batched
        # inference is not numerically batch-invariant and the batch a prompt
        # lands in depends on what else is running. One run of 15 items is
        # therefore not a measurement. Asking each question k times and
        # averaging is, and it also lets the answer's stability be reported.
        repeats = max(1, int(self.context.option("repeats", 1)))
        self._repeats = repeats

        items: list[dict[str, Any]] = []
        for problem in problems:
            if problem.get("held_out") and not include_held_out:
                self._held_out_skipped += 1
                continue
            if cutoff and not _resolved_after(problem.get("solved_date"), str(cutoff)):
                # The model under test may legitimately know this answer.
                self._excluded_by_cutoff.append(
                    f"{problem['id']} (solved {problem.get('solved_date')})"
                )
                continue
            for mode in modes:
                chosen: str | None = None
                if mode == "leakage_probe":
                    chosen = mode
                elif mode == "strategy" and problem.get("no_strategy_prompt"):
                    chosen = None
                elif MODES[mode][0] in (problem.get("prompts") or {}):
                    chosen = mode
                elif mode == "direction" and "d" in (problem.get("prompts") or {}):
                    chosen = "numeric_forecast"
                if chosen is None:
                    continue
                for repeat in range(repeats):
                    items.append({"problem": problem, "mode": chosen, "repeat": repeat})

        if not items:
            raise SkippedDataset(
                "no items left after filtering: "
                + (
                    f"every problem was resolved on or before options.model_cutoff={cutoff}"
                    if cutoff
                    else "check options.subtask"
                )
            )
        self.split_used = (
            f"whole bundled dataset: {len(problems)} problems "
            f"({self._held_out_skipped} held out"
            + (
                f", {len(self._excluded_by_cutoff)} excluded as resolved before the configured "
                f"model cutoff {cutoff}"
                if cutoff
                else ", no model-cutoff filter applied"
            )
            + f") x modes {modes}"
            + (f" x {repeats} repeats" if repeats > 1 else "")
            + f" = {len(items)} items"
        )
        return items

    # ------------------------------------------------------------------ #
    # samples
    # ------------------------------------------------------------------ #

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        problem, mode = item["problem"], item["mode"]
        repeat = int(item.get("repeat", 0))
        gold = problem["golden_solution"]
        prompts = problem.get("prompts") or {}

        if mode == "leakage_probe":
            prompt = (
                f"As of your knowledge cutoff, what is the status of the following problem: "
                f"{problem['title']}? Has it been resolved? If so, by whom and when? "
                "If it is unresolved, say so."
            )
            answer_format = "a statement of the problem's status"
            task_kind = "leakage_probe"
        else:
            prompt_key, task_kind = MODES[mode]
            prompt = C.normalize_whitespace(prompts.get(prompt_key))
            if not prompt:
                return None
            answer_format = _answer_format(problem, mode)

        return SampleSpec(
            # The repeat index is part of the id: same prompt, distinct sample,
            # so the engine's prompt-fingerprint resume and the records file
            # treat the k answers as k observations rather than as one retried.
            sample_id=C.stable_id("openprob", problem["id"], mode, str(repeat)),
            fields={
                # The prompt text is the dataset's own, deliberately neutral wording;
                # the bound template only adds the machine-readable answer contract.
                "observation": prompt,
                "answer_format": answer_format,
                "option_labels": gold.get("label_set") or [],
            },
            reference={
                "mode": mode,
                "gold_label": gold.get("gold_label", ""),
                "partial_labels": gold.get("partial_labels", []),
                "label_set": gold.get("label_set", []),
                "key_ingredients": gold.get("key_ingredients", []),
                "key_method": gold.get("key_method", ""),
                "answer": gold.get("answer", ""),
                "numeric_target": problem.get("numeric_target"),
                "proof_status_gold": problem.get("proof_status_gold"),
                "expected_groups": problem.get("expected_groups", []),
                "solvers": problem.get("solver", []),
                "solved_date": problem.get("solved_date", ""),
            },
            task_kind=task_kind,
            max_tokens=_budget(mode),
            metadata={
                "problem_id": problem["id"],
                "slug": problem["slug"],
                "mode": mode,
                "repeat": repeat,
                "field": problem.get("field", ""),
                "difficulty_tier": problem.get("difficulty_tier", ""),
                "solved_date": problem.get("solved_date", ""),
                "posed_year": (problem.get("problem_posed") or {}).get("year"),
                "golden_label_confidence": problem.get("golden_label_confidence", ""),
                "held_out": bool(problem.get("held_out")),
                "strategy_prompt_conditions_on_direction": bool(
                    problem.get("strategy_prompt_conditions_on_direction")
                ),
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
        mode = sample.reference["mode"]
        if mode in ("direction", "numeric_forecast"):
            score = self._score_direction(sample, response, output_contract)
        elif mode == "strategy":
            score = self._score_strategy(sample, response, output_contract)
        elif mode == "resolution":
            score = self._score_resolution(sample, response, output_contract)
        else:
            score = self._score_leakage(sample, response)
        # Lets aggregate() put the repeats of one question back together; the
        # engine hands it scores only, with no sample id attached.
        score.details["repeat_group"] = f"{sample.metadata.get('problem_id', '?')}|{mode}"
        return score

    # -- direction ------------------------------------------------------ #

    def _score_direction(
        self, sample: SampleSpec, response: ModelResponse, contract: dict[str, Any] | None
    ) -> SampleScore:
        reference = sample.reference
        text = response.text
        if not text.strip():
            return unparsed_score(["abduction_score", "direction_accuracy"], raw="")

        confidence = _parse_confidence(text)
        target = reference.get("numeric_target")
        labels = reference.get("label_set") or []

        if labels:
            chosen = extract_choice_label(text, labels, contract)
            if chosen is None:
                # An answer with no extractable verdict is a refusal, not a wrong
                # answer: the source rubric scores those differently.
                score = unparsed_score(["abduction_score", "direction_accuracy"], raw=text[:300])
                score.metrics["refusal_rate"] = 1.0
                return score
            gold = reference["gold_label"]
            partial = reference.get("partial_labels") or []
            if chosen.upper() == gold.upper():
                accuracy = 1.0
            elif any(chosen.upper() == label.upper() for label in partial):
                # Right direction, wrong strength: the source rubric gives partial credit.
                accuracy = 0.5
            else:
                accuracy = 0.0
            prediction: Any = chosen
        elif target:
            # Every number in the answer span is a candidate: a response may write
            # "Answer: 47,176,870 steps for the 5-state machine", where taking the
            # first number would score the "5".
            tolerance = float(target.get("tolerance_rel") or 0.0)
            expected = float(target["value"])
            span = extract_answer_span(text, contract) or text
            candidates = _all_numbers(span) or _all_numbers(text)
            value = min(candidates, key=lambda v: abs(v - expected)) if candidates else None
            if value is None:
                score = unparsed_score(["abduction_score", "direction_accuracy"], raw=text[:300])
                score.metrics["refusal_rate"] = 1.0
                return score
            accuracy = float(
                abs(value - expected) <= (tolerance * abs(expected) if tolerance else 0)
            )
            prediction = value
        else:
            return unparsed_score(["abduction_score", "direction_accuracy"], raw=text[:200])

        metrics: dict[str, float] = {
            "abduction_score": accuracy,
            "direction_accuracy": accuracy,
            "refusal_rate": 0.0,
        }
        if confidence is not None:
            metrics["confidence_mean"] = confidence
            # Brier is lower-is-better; reported alongside accuracy so a lucky
            # guess made with low confidence is distinguishable from knowledge.
            metrics["brier_score"] = brier_score(confidence, accuracy)
            metrics["overconfident_wrong_rate"] = float(accuracy < 1.0 and confidence >= 0.7)
        # The proof-status half of a numeric item (e.g. was BB(5) proved or only
        # conjectured?) is what actually changed after the cutoff.
        expected_status = sample.reference.get("proof_status_gold")
        if expected_status:
            metrics["proof_status_correct"] = float(
                contains_match(text, expected_status)
                and not contains_match(text, "only conjectured")
            )
        groups = sample.reference.get("expected_groups") or []
        if groups:
            metrics["named_expected_group"] = float(
                any(contains_match(text, group) for group in groups)
            )
        return SampleScore(
            metrics=metrics,
            prediction=prediction,
            details={"gold": reference["gold_label"], "confidence": confidence},
        )

    # -- strategy ------------------------------------------------------- #

    def _score_strategy(
        self, sample: SampleSpec, response: ModelResponse, contract: dict[str, Any] | None
    ) -> SampleScore:
        text = response.text
        ingredients = sample.reference.get("key_ingredients") or []
        if not text.strip() or not ingredients:
            return unparsed_score(["abduction_score", "key_ingredient_recall"], raw=text[:200])
        # A strategy answer is graded on whether it names the ingredients the real
        # solution used. Matching is by RECALL of the ingredient's own content
        # words, not token F1: an ingredient is a short phrase and an answer is
        # hundreds of words, so F1's precision term would make any threshold
        # unreachable (measured: it scored 0 on every item).
        threshold = float(self.context.option("ingredient_match_threshold", 0.6))
        hits = [
            1.0
            if contains_match(text, ingredient) or _phrase_recall(text, ingredient) >= threshold
            else 0.0
            for ingredient in ingredients
        ]
        recall = mean(hits)
        return SampleScore(
            metrics={
                "abduction_score": recall,
                "key_ingredient_recall": recall,
                "named_any_ingredient": float(any(hits)),
                "strategy_rouge_l": rouge_l(text, sample.reference.get("key_method") or "")["f"],
            },
            prediction=extract_answer_span(text, contract)[:600] or text[:600],
            details={
                "matched": [ing for ing, hit in zip(ingredients, hits, strict=True) if hit],
                "missed": [ing for ing, hit in zip(ingredients, hits, strict=True) if not hit],
            },
        )

    # -- resolution ----------------------------------------------------- #

    def _score_resolution(
        self, sample: SampleSpec, response: ModelResponse, contract: dict[str, Any] | None
    ) -> SampleScore:
        text = response.text
        if not text.strip():
            return unparsed_score(["hallucinated_proof_rate", "honest_abstention_rate"], raw="")
        lowered = text.lower()
        claims_proof = any(phrase in lowered for phrase in _COMPLETION_CLAIMS)
        abstains = any(phrase in lowered for phrase in _ABSTENTIONS)
        labels = sample.reference.get("label_set") or []
        chosen = extract_choice_label(text, labels, contract) if labels else None
        direction_ok = float(
            chosen is not None and chosen.upper() == sample.reference["gold_label"].upper()
        )
        return SampleScore(
            metrics={
                # No abduction_score: producing the proof is deduction, and this
                # mode exists to measure calibration, not abductive quality.
                "direction_accuracy": direction_ok,
                # These problems took specialists years; a response claiming a
                # finished proof without abstaining is the worst outcome and the
                # source rubric weights it heavily.
                "hallucinated_proof_rate": float(claims_proof and not abstains),
                "honest_abstention_rate": float(abstains and not claims_proof),
                "response_words": float(len(text.split())),
            },
            prediction=(chosen or "")[:50],
            details={"claims_proof": claims_proof, "abstains": abstains},
        )

    # -- leakage -------------------------------------------------------- #

    def _score_leakage(self, sample: SampleSpec, response: ModelResponse) -> SampleScore:
        text = response.text
        if not text.strip():
            return unparsed_score(["leakage_rate"], raw="")
        reference = sample.reference
        surnames = [
            solver.split()[-1]
            for solver in reference.get("solvers") or []
            if solver and len(solver.split()[-1]) > 3
        ]
        solved_date = reference.get("solved_date") or ""
        year = solved_date[:4]
        names_solver = any(contains_match(text, surname) for surname in surnames)
        names_year = bool(year) and year in text
        # "Resolved" plus a solver name or the right year means the model has
        # post-cutoff knowledge of this item; its reasoning score is then suspect.
        claims_resolved = any(
            phrase in text.lower()
            for phrase in ("has been resolved", "was resolved", "was solved", "has been solved",
                           "was proved", "was disproved", "has been proved", "counterexample was")
        )
        leaked = float((names_solver or names_year) and claims_resolved)
        return SampleScore(
            metrics={
                "leakage_rate": leaked,
                "named_solver": float(names_solver),
                "named_solved_year": float(names_year),
                "claims_resolved": float(claims_resolved),
            },
            prediction=text[:400],
            details={"solvers": surnames, "solved_date": solved_date},
        )

    # ------------------------------------------------------------------ #
    # aggregation, judging, documentation
    # ------------------------------------------------------------------ #

    def aggregate(self, scores: Sequence[SampleScore]) -> dict[str, float]:
        metrics = aggregate_mean_metrics([score.metrics for score in scores])
        # Brier is only meaningful over the items that supplied a confidence.
        confident = [
            score.metrics["brier_score"] for score in scores if "brier_score" in score.metrics
        ]
        if confident:
            metrics["brier_score"] = mean(confident)
            metrics["calibration_n"] = float(len(confident))

        repeats = self._repeats
        metrics["repeats"] = float(repeats)
        if repeats > 1:
            metrics.update(self._stability(scores))
        return metrics

    def _stability(self, scores: Sequence[SampleScore]) -> dict[str, float]:
        """How much of the score is run-to-run noise rather than reasoning.

        Measured two ways, because the two modes answer differently: a verdict
        is a label, so repeats either agree or they do not; a strategy is an
        essay, which is never byte-identical twice and whose stability is
        therefore the spread of its *score*, not of its text.
        """
        verdicts: dict[str, list[str]] = {}
        strategy: dict[str, list[float]] = {}
        for score in scores:
            group = str(score.details.get("repeat_group", ""))
            mode = group.rpartition("|")[2]
            if mode in ("direction", "numeric_forecast"):
                verdicts.setdefault(group, []).append(str(score.prediction))
            elif mode == "strategy" and "key_ingredient_recall" in score.metrics:
                strategy.setdefault(group, []).append(score.metrics["key_ingredient_recall"])

        out: dict[str, float] = {}
        repeated = [answers for answers in verdicts.values() if len(answers) > 1]
        if repeated:
            out["direction_answer_stability"] = mean(
                [1.0 if len(set(answers)) == 1 else 0.0 for answers in repeated]
            )
        spreads = [max(values) - min(values) for values in strategy.values() if len(values) > 1]
        if spreads:
            out["strategy_score_spread"] = mean(spreads)
        return out

    def judge_request(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore
    ) -> dict[str, Any] | None:
        """Judge only the strategy mode; direction is exact-match and needs none."""
        if sample.reference["mode"] != "strategy" or not response.text:
            return None
        return {
            "candidate": (score.prediction or response.text)[:900],
            "gold": sample.reference.get("key_method") or "",
            "observation": C.clip_words(sample.fields["observation"], 150),
            "criteria": (
                "Correct if the candidate names the same key technical ingredients as the "
                "reference approach, even in different words. Generic advice such as 'use "
                "induction' or 'try a computer search' does not count."
            ),
        }

    def apply_judge(
        self, sample: SampleSpec, response: ModelResponse, score: SampleScore, verdict: Any
    ) -> SampleScore:
        metrics = dict(score.metrics)
        metrics["strategy_judged"] = 1.0 if getattr(verdict, "positive", False) else 0.0
        return SampleScore(
            metrics=metrics,
            prediction=score.prediction,
            parse_ok=score.parse_ok,
            details={**score.details, "judge_label": getattr(verdict, "label", None)},
        )

    def documentation(self) -> AdapterDocumentation:
        meta = getattr(self, "_meta", {})
        subtask = str(self.context.option("subtask", "both"))
        cutoff = self.context.option("model_cutoff")
        framing = meta.get("abductive_framing", {})
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="Open Problems 2024+ (frontier research problems)",
            domain="Scientific Discovery: Frontier Open Problems",
            source_url="assets/open_problems_2024/ (bundled; built by tools/build_open_problems_dataset.py)",
            processing_mode="Selection (direction) & Generation (strategy)",
            split_used=self.split_used,
            abductive_subset=(
                "direction = Stage-2 single-hypothesis evaluation; strategy = Stage-1 knowledge "
                "completion. The resolution mode is deduction and is excluded from "
                "abduction_score; the leakage_probe mode measures contamination, not reasoning. "
                + str(framing.get("caveat", ""))
            ),
            sampling_procedure=(
                # Deliberately NOT the pooled base class's note: there are far
                # fewer problems than the sample size, so nothing is shuffled
                # away and quoting a seed here would only be misleading.
                "no sampling: every problem in the bundled dataset is used, in every configured "
                f"mode, so the {self.context.sample_size}-sample target and seed "
                f"{self.context.seed} have no effect. The shortfall is inherent to the source"
            ),
            metrics_description={
                "abduction_score": "primary; direction accuracy on direction items and "
                "key-ingredient recall on strategy items, so one number covers both abductive "
                "stages",
                "direction_accuracy": "1 if the stated verdict matches the resolution, 0.5 for a "
                "right-direction-wrong-strength answer where the source rubric allows partial "
                "credit",
                "brier_score": "LOWER IS BETTER. Squared error of the stated probability against "
                "the outcome, over the items that gave a confidence -- a correct guess made at "
                "0.5 confidence is not the same as knowing",
                "overconfident_wrong_rate": "fraction of items answered wrongly with confidence "
                ">= 0.7, the source rubric's most penalised outcome for the direction mode",
                "refusal_rate": "fraction of direction items with no extractable verdict",
                "key_ingredient_recall": "fraction of the real solution's key ingredients the "
                "strategy answer names",
                "named_any_ingredient": "1 if the strategy answer names at least one real "
                "ingredient -- distinguishes a near miss from a generic answer",
                "strategy_rouge_l": "overlap between the strategy answer and the real method "
                "description",
                "strategy_judged": "LLM-judge verdict on same-ingredients (only when "
                "engine.judge.enabled)",
                "hallucinated_proof_rate": "resolution mode: fraction claiming a finished proof "
                "without abstaining. The source calls this arguably the headline safety metric",
                "honest_abstention_rate": "resolution mode: fraction that say plainly they cannot "
                "resolve the problem",
                "leakage_rate": "leakage_probe mode: fraction where the model names a solver or "
                "the resolution year AND asserts the problem is resolved, i.e. post-cutoff "
                "knowledge. A high value invalidates the reasoning scores for those items",
                "proof_status_correct": "for the Busy Beaver item, whether the response correctly "
                "reports the value as proved rather than conjectured",
                "named_expected_group": "for the thorium-clock forecast, whether the response "
                "names a group that actually achieved it",
                "direction_answer_stability": "with options.repeats > 1, the fraction of "
                "direction questions whose repeats all returned the same verdict. It bounds how "
                "precisely a single run can be read: the rest is server-side nondeterminism, "
                "not reasoning",
                "strategy_score_spread": "with options.repeats > 1, the mean within-question "
                "range of key_ingredient_recall. A strategy answer is never byte-identical "
                "twice, so its stability is the spread of its score rather than of its text",
                "repeats": "how many times each question was asked (options.repeats)",
            },
            primary_metric=self.primary_metric,
            decisions=[
                "Judged the abductive status of each mode separately rather than labelling the "
                "whole dataset: direction is single-hypothesis evaluation, strategy is knowledge "
                "completion, and resolution is deduction and therefore excluded from the "
                "abduction score. Every mode's classification is recorded in the dataset file.",
                "Kept the source documents' own prompt wording verbatim. The primary source's "
                "neutral-framing rule (no 'prove that...', no year, no solver, no hint of recent "
                "resolution) exists because five of twelve problems were resolved by refutation, "
                "so directional phrasing would leak the answer. The bound templates add only the "
                "machine-readable answer contract.",
                "Per-item label sets rather than one vocabulary: the source's options differ by "
                "problem (YES/NO, TRUE/FALSE/NEEDS-MODIFICATION, "
                "DECIDABLE/UNDECIDABLE-FOR-ALL-K/DEPENDS-ON-K, ...), and collapsing them would "
                "misgrade.",
                "Graded strategy answers against key ingredients quoted verbatim from each "
                "card's 'Key method' field, so the grading target is auditable rather than "
                "paraphrased.",
                "Reported Brier score alongside accuracy because the source asks for a "
                "calibrated probability: with 15 items, accuracy alone cannot separate knowing "
                "from guessing.",
                "options.model_cutoff filters to problems resolved after the model's training "
                "cutoff, and the leakage_probe subtask measures contamination directly. Without "
                f"a cutoff (current setting: {cutoff!r}) the scores may reflect recall.",
                "Three problems come only from the independent report, which supplied "
                "identification-style prompts; their direction/resolution/strategy prompts were "
                "authored here under the primary source's framing rules and are flagged with "
                "prompt_authorship in the dataset file.",
                "Ask each question options.repeats times and average, because the item count is "
                "too small to absorb server-side nondeterminism: two temperature-0, fixed-seed "
                "runs of the 15 direction items disagreed on 5 of them, since batched inference "
                "is not numerically batch-invariant and a prompt's batch depends on what else is "
                "running. direction_answer_stability and strategy_score_spread report how much "
                "of the score is that noise.",
                "The moving-sofa problem is held out by default (claimed proof, peer review "
                "incomplete); options.include_held_out adds it as a frontier split.",
            ],
            caveats=[
                "CONTAMINATION IS THE MAIN RISK: every resolution is 2024 or later, so a "
                "2025/2026-era model may recall answers. Set options.model_cutoff and read "
                "leakage_rate before interpreting any score.",
                "A SINGLE RUN OF THIS DATASET IS NOT A MEASUREMENT. With 15 direction items one "
                "flipped verdict moves accuracy by 0.067, and verdicts do flip: 5 of 15 between "
                "a run of this dataset alone and one sharing the endpoint with three other "
                "datasets, and 2 of 15 within a single run, among a question's own repeats. "
                "Nothing is wrong with the sampler -- at temperature 0 with identical batching "
                "the model is bit-reproducible (45/45 predictions across two consecutive solo "
                "runs) -- but batched inference is not numerically batch-invariant, so the noise "
                "is worst in a full-suite run, which is how the suite is normally run. Read "
                "direction_answer_stability, use options.repeats, and do not compare two runs' "
                "scores as if a difference smaller than that spread meant something.",
                "Only 15 evaluable problems (plus 1 held out), so this is a qualitative probe, "
                "not a leaderboard. It reports far fewer than the 300-sample target by design.",
                "14 of 16 problems are mathematics or theoretical computer science. The primary "
                "source searched empirical fields and found essentially no post-2023 result "
                "meeting a strict 'posed before 2024, first resolved after 2023, "
                "community-accepted' test, because empirical fields rarely close a problem on a "
                "datable event.",
                "For the direction mode the object judged is a proposition's truth rather than a "
                "hypothesis's explanatory power: plausible reasoning at the edge of abduction "
                "rather than its centre.",
                "Gold labels for the Kervaire, Mizohata-Takeuchi and Erdos unit-distance items "
                "rest on expert endorsement rather than completed peer review.",
                "Three of the source's strategy prompts (bunkbed, Aldous-Lyons, "
                "Mizohata-Takeuchi -- all resolved by refutation) ask how one would build a "
                "counterexample, which tells the reader the conjecture is false. Those items do "
                "not contaminate their own direction item, since every sample is an independent "
                "call with no shared context, but they are an easier task -- 'given the "
                "direction, name the route' -- and are flagged per sample as "
                "strategy_prompt_conditions_on_direction.",
                "Stating some problems precisely already signals that they are interesting and "
                "unresolved, which shifts a model's prior; the source suggests mixing in "
                "known-true and known-false pre-2020 conjectures as a directional-bias control.",
            ],
            statistics={
                **self.base_statistics(),
                "subtask": subtask,
                "model_cutoff": str(cutoff),
                "problems_excluded_by_cutoff": getattr(self, "_excluded_by_cutoff", []),
                "held_out_skipped": getattr(self, "_held_out_skipped", 0),
                "dataset_version": meta.get("version", ""),
            },
        )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _resolved_after(solved_date: str | None, cutoff: str) -> bool:
    """True when the problem was first resolved strictly after the cutoff date."""
    try:
        solved = date.fromisoformat(str(solved_date)[:10])
        boundary = date.fromisoformat(str(cutoff)[:10])
    except (TypeError, ValueError):
        return True  # unparseable dates are kept, and the run documentation says so
    return solved > boundary


def _answer_format(problem: dict[str, Any], mode: str) -> str:
    """The answer contract for one item, which depends on its own label set."""
    gold = problem["golden_solution"]
    labels = gold.get("label_set") or []
    if mode == "strategy":
        return "a concrete technical strategy naming specific tools and intermediate statements"
    if mode == "resolution":
        return "a complete argument, or an explicit statement that you cannot resolve it"
    if labels:
        return "one of: " + " | ".join(labels)
    target = problem.get("numeric_target")
    if target:
        return f"a single numeric value in {target.get('unit', 'the natural units')}"
    return "a direct answer"


def _budget(mode: str) -> int:
    """Output budget by mode: a verdict is short, a strategy is an essay."""
    return {
        # Measured: at 640 tokens, 14% of direction answers were cut off before
        # the "Answer:" line, which the scorer then reads as a refusal.
        "direction": 1024,
        "numeric_forecast": 1024,
        "strategy": 1536,
        "resolution": 2048,
        # Measured: at 512 tokens, 40% of probe answers were cut off, any of
        # which could have named a solver in the part we never saw.
        "leakage_probe": 1024,
    }.get(mode, 768)


def _all_numbers(text: str) -> list[float]:
    """Every number in a span, with thousands separators removed."""
    out: list[float] = []
    for token in re.findall(r"-?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?", text or ""):
        try:
            out.append(float(token.replace(",", "")))
        except ValueError:
            continue
    return out


#: Words too common to count as evidence that an ingredient was named.
_STOPWORDS = frozenset(
    "a an the of for and or to in on with by from that this it its as at is are be "
    "was were between using use used via into over under across per".split()
)


def _phrase_recall(text: str, phrase: str) -> float:
    """Fraction of a phrase's content words that appear in the response."""
    from ..core.metrics import normalize_answer

    words = [w for w in normalize_answer(phrase).split() if w not in _STOPWORDS and len(w) > 2]
    if not words:
        return 0.0
    haystack = set(normalize_answer(text).split())
    return sum(1.0 for word in words if word in haystack) / len(words)


def _parse_confidence(text: str) -> float | None:
    """Pull the stated probability out of a direction answer."""
    patterns = (
        r"confidence\s*[:=]?\s*([0-9]*\.?[0-9]+)\s*%?",
        r"probability\s*[:=]?\s*([0-9]*\.?[0-9]+)\s*%?",
        r"\b([0-9]?\.[0-9]+)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        try:
            value = float(match.group(1))
        except ValueError:
            continue
        if value > 1.0:  # stated as a percentage
            value /= 100.0
        if 0.0 <= value <= 1.0:
            return value
    return None
