"""ABD: default-exception abduction in finite first-order worlds.

Source: https://github.com/SerafimBatzoglou/concept-synth
        (``benchmarks/abduction/``, release "abduction-v1.2")

Each instance gives a first-order theory containing a **default rule with an
exception predicate** ``Ab(x)``, several finite worlds, and the question: find
one formula ``alpha(x)`` that, substituted for ``Ab``, repairs the theory in
every world at once.  Instances come in three observation regimes
(``ABD_FULL``, ``ABD_PARTIAL``, ``ABD_SKEPTICAL``) which differ in how unknown
atoms are treated, and carry a difficulty label.

**Scored by the release's own Z3 evaluator, not by comparison with the gold.**
This is the whole point of the benchmark: ABD is *solver-checkable*, and its
``concept_synth.abduction.evaluate_abd_b1`` decides exactly whether a formula
repairs every prompt world and at what cost.  Comparing the answer with the
planted gold instead -- as this adapter previously did -- gets the task wrong,
because the gold is one valid repair among many and is not even the cheapest
one.  Measured on the release's own instance 0: the gold ``alpha`` is valid at
cost 18 against a solver optimum of 14, and a completely different formula,
``(exists y (and (S x y) (P y)))``, is *also* valid, at cost 34.  Any
gold-matching metric scores that second formula 0; the official evaluator
scores it valid and not optimal, which is what it is.

So the metrics here are the release's:

``valid``    the formula repairs every prompt world -- **primary**
``optimal``  it does so with zero gap above the solver's lower bound
``total_cost`` / ``total_gap`` / ``avg_gap``  parsimony, over valid answers
``formula_match``  exact text equality with the planted gold, kept only as a
             diagnostic: how often a model reproduces the gold rather than
             finding some valid repair

Validity is the headline rather than optimality because the benchmark says so
("Primary objective: **Validity**", "Secondary objective: **Parsimony**"), and
because optimality is a bar the planted golds themselves usually miss.

**The prompt is the release's.**  ABD's task specification is not framing that
can be paraphrased: the substitution semantics, the closed-world rule, the
per-regime completion semantics, the cost objective and the formula grammar are
all things the evaluator checks.  So the prompt body is built with the
release's own template and formatters, and only its final JSON output contract
is replaced by this suite's answer contract, which is what the harness parses.
"""

from __future__ import annotations

import gzip
import re
import sys
import threading
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ..core.adapter import SkippedDataset
from ..core.metrics import aggregate_mean_metrics, extract_answer_span
from ..core.types import AdapterDocumentation, ModelResponse, SampleScore, SampleSpec
from . import _common as C
from ._abd_eval import IsolatedEvaluator
from ._base import PooledDatasetAdapter, unparsed_score

REPO_URL = "https://github.com/SerafimBatzoglou/concept-synth"

#: The release's own default Z3 timeout, in milliseconds. Not tuned here: a
#: different timeout is a different benchmark.
DEFAULT_TIMEOUT_MS = 5000


def _load_release(repo_root: Path) -> SimpleNamespace:
    """Import the release's evaluator and prompt builder out of the clone.

    Vendored copies drift. The clone is already required to get the data, so
    the code that defines what a correct answer *is* comes from the same place
    the data does, at the same version.
    """
    source = repo_root / "src"
    if not (source / "concept_synth").is_dir():
        raise SkippedDataset(
            f"the ABD release's package is not in the clone: {source / 'concept_synth'} "
            "does not exist, so its evaluator cannot be used"
        )
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    try:
        from concept_synth.abduction import abd_b1_prompt as prompts
        from concept_synth.abduction import evaluate_abd_b1 as evaluator
        from concept_synth.abduction.abd_formula_utils import get_used_predicates
    except ImportError as exc:  # z3-solver missing, or a release layout change
        raise SkippedDataset(
            f"the ABD release's Z3 evaluator could not be imported ({exc}). It needs "
            "z3-solver; install this project's 'adapters' extra."
        ) from exc
    return SimpleNamespace(
        evaluator=evaluator,
        prompts=prompts,
        used_predicates=get_used_predicates,
        templates={
            "ABD_FULL": (prompts.load_abd_full_templates, prompts.format_world_full),
            "ABD_PARTIAL": (prompts.load_abd_partial_templates, prompts.format_world_partial),
            "ABD_SKEPTICAL": (
                prompts.load_abd_skeptical_templates,
                prompts.format_world_partial,
            ),
        },
    )


class ABDAdapter(PooledDatasetAdapter):
    """Synthesize the abnormality rule that repairs a default theory."""

    adapter_version = "2.0"

    system_prompt = (
        "You are an expert at abductive reasoning: inferring the explanation that, if true, "
        "would best account for the evidence you are given. Here the evidence is a default "
        "theory whose exception predicate Ab(x) is undefined, and several finite worlds in "
        "which the theory must hold. Your hypothesis is the definition of Ab: one formula "
        "alpha(x), shared across every world, that makes the theory true everywhere while "
        "calling as few objects abnormal as possible."
    )
    data_delivery_mode = "static"

    answer_format = "one s-expression formula in the variable x"
    answer_constraints = (
        "output exactly one formula",
        "use only the allowed predicates",
        "leave x as the only free variable",
        "output only the formula",
        "do not use introductory phrases or commentary",
    )
    #: Solver-checkable end to end: Z3 decides validity and cost exactly, and an
    #: answer it cannot parse is not undecidable but invalid -- a formula that
    #: does not parse repairs nothing. There is no judge here and no use for
    #: one; self-consistency over the repeats is available as usual.
    objective_metrics = True
    selection_cardinality = None
    primary_metric = "valid"

    # ------------------------------------------------------------------ #
    # data
    # ------------------------------------------------------------------ #

    def load_items(self) -> list[dict[str, Any]]:
        root = C.ensure_git_repo(
            REPO_URL, self.context.data_dir / "repo", depth=1, offline=self.context.offline
        )
        self._release = _load_release(root)
        # Parsing, scoping and prompt building are pure Python and run here;
        # only the solver runs behind a process boundary, because it can abort.
        self._isolated = IsolatedEvaluator(root / "src")
        archives = C.find_files(root / "benchmarks" / "abduction" / "data", ["*instances*.yaml.gz"])
        if not archives:
            raise SkippedDataset("ABD instance archive not found in the repository")
        import yaml

        with gzip.open(archives[0], "rt", encoding="utf-8", errors="replace") as handle:
            payload = yaml.safe_load(handle)
        rows = payload if isinstance(payload, list) else []
        scenarios = self.context.option("scenarios")

        items: list[dict[str, Any]] = []
        self._problems: dict[str, dict[str, Any]] = {}
        self._per_scenario: dict[str, int] = {}
        #: Canonical records this adapter refuses to score, with the reason.
        #: Reported, never repaired: a benchmark item whose own gold the release
        #: cannot read is a defect in the release, and quietly patching it would
        #: change what the published numbers mean.
        self.invalid_items: list[dict[str, str]] = []
        #: Anomalies worth naming that are not grounds for exclusion.
        self.gold_anomalies: list[dict[str, str]] = []

        for row in rows:
            problem = (row or {}).get("problem") or {}
            instance_id = str(problem.get("instanceId") or "")
            gold = (problem.get("gold") or {}).get("alpha")
            scenario = str(problem.get("scenario") or "unknown")
            if not gold or not instance_id:
                continue
            if scenarios and scenario not in scenarios:
                continue
            defects, anomalies = self._inspect_gold(problem)
            if anomalies:
                self.gold_anomalies.append(
                    {"instance_id": instance_id, "reason": "; ".join(anomalies)}
                )
            if defects:
                self.invalid_items.append(
                    {
                        "instance_id": instance_id,
                        "scenario": scenario,
                        "reason": "; ".join(defects),
                    }
                )
                continue
            items.append(row)
            self._problems[instance_id] = problem
            self._per_scenario[scenario] = self._per_scenario.get(scenario, 0) + 1

        if not items:
            raise SkippedDataset("ABD instances carry no usable gold alpha formulas")
        for entry in self.invalid_items:
            self.log.warning(
                "ABD %s excluded as an invalid benchmark item: %s",
                entry["instance_id"],
                entry["reason"],
            )
        # The prompt body is the release's, minus its own output contract. If
        # that contract cannot be found to remove, the model would be told to
        # answer in a format this harness does not parse, so the dataset refuses
        # rather than mis-prompting.
        for scenario in sorted(self._per_scenario):
            if scenario not in self._release.templates:
                raise SkippedDataset(f"ABD scenario {scenario!r} has no release prompt template")

        self.split_used = (
            f"{archives[0].name} in full ({len(items)} instances: "
            + ", ".join(f"{name} {count}" for name, count in sorted(self._per_scenario.items()))
            + "); the release ships a single evaluation set"
            + (
                f"; {len(self.invalid_items)} record(s) excluded as invalid benchmark items"
                if self.invalid_items
                else ""
            )
        )
        return items

    def _inspect_gold(self, problem: dict[str, Any]) -> tuple[list[str], list[str]]:
        """Check a canonical record's own gold against the release's own rules.

        Returns ``(defects, anomalies)``. A *defect* disqualifies the record:
        the release's parser cannot read its gold, or the gold uses predicates
        the record's own ``allowedAlphaPreds`` forbids -- either way the item
        asks for something its answer key does not satisfy. An *anomaly* is
        worth reporting but not disqualifying.
        """
        release = self._release
        gold = str((problem.get("gold") or {}).get("alpha") or "")
        allowed = {str(name) for name in C.as_list(problem.get("allowedAlphaPreds"))}
        try:
            parsed = release.evaluator.parse_alpha_formula_with_suffix_repair(gold)
        except Exception as exc:  # noqa: BLE001 - any parser failure disqualifies
            return [f"the release's own parser rejects its gold formula ({exc})"], []

        defects: list[str] = []
        anomalies: list[str] = []
        if getattr(parsed, "trailing_parens_added", 0):
            anomalies.append(
                f"the gold formula is short {parsed.trailing_parens_added} closing "
                "parenthesis/es and was completed by the release's own suffix repair"
            )
        # Two independent readings of the same rule: the release's own scoping
        # check (which the evaluator applies to a model's answer) and the
        # record's own allowedAlphaPreds field. Both are consulted, so a record
        # is disqualified whichever of the two it breaks, and a disagreement
        # between them would show up as a defect naming only one.
        try:
            scoped_ok, scoping_error, _ = release.evaluator.validate_alpha_predicate_scoping(
                parsed.ast, str(problem.get("theoryId") or "")
            )
        except Exception as exc:  # noqa: BLE001
            return [f"the release's own scoping check cannot read its gold formula ({exc})"], anomalies
        if not scoped_ok:
            defects.append(
                "the release's own evaluator rejects its gold formula: "
                f"{scoping_error}"
            )
        try:
            used = {str(name) for name in release.used_predicates(parsed.ast)}
        except Exception as exc:  # noqa: BLE001
            return [f"the release's own checker cannot read its gold formula ({exc})"], anomalies
        outside = sorted(used - allowed)
        if allowed and outside:
            defects.append(
                f"its gold formula uses {outside}, which its own allowedAlphaPreds "
                f"({sorted(allowed)}) does not permit -- and the prompt tells the model so"
            )
        return defects, anomalies

    # ------------------------------------------------------------------ #
    # prompt
    # ------------------------------------------------------------------ #

    def _prompt_body(self, problem: dict[str, Any], worlds: list[dict[str, Any]]) -> str:
        """The release's own prompt, minus the JSON output contract it ends with.

        Every other section is load-bearing -- the Ab substitution, the
        closed-world rule, the regime's completion semantics, the cost
        objective, the formula grammar, the forbidden predicates -- because the
        evaluator checks all of them. Rewriting any of it in this suite's own
        words would risk grading a model on a rule it was never told.
        """
        release = self._release
        scenario = str(problem.get("scenario") or "ABD_FULL")
        load_templates, format_world = release.templates[scenario]
        task_template, _suffix = load_templates()
        allowed, forbidden = release.prompts.get_predicate_scope_from_problem(problem)
        theory = f"**Theory ID**: {problem.get('theoryId', 'Unknown')}\n\n**Axioms**:\n"
        theory += release.prompts.format_axioms(C.as_list(problem.get("axioms")))
        worlds_text = "\n\n".join(
            format_world(world, index) for index, world in enumerate(worlds)
        )
        return (
            task_template
            + release.prompts.format_predicate_scope(allowed, forbidden)
            + "\n\n"
            + theory
            + "\n\n## Training Worlds\n\n"
            + worlds_text
        )

    def make_sample(self, item: dict[str, Any], index: int) -> SampleSpec | None:
        problem = item.get("problem") or {}
        description = item.get("problemDescription") or {}
        gold = problem.get("gold") or {}
        alpha = C.normalize_whitespace(gold.get("alpha"))
        worlds = C.as_list(problem.get("trainWorlds"))
        instance_id = str(problem.get("instanceId") or "")
        if not alpha or not C.as_list(problem.get("axioms")) or not worlds or not instance_id:
            return None

        # Every world by default. The release's own prompts show all of them,
        # and its evaluator scores against all of them, so anything less would
        # grade a model on evidence it was not shown. options.max_worlds
        # truncates for a deliberate partial-evidence ablation, and the
        # evaluator is then given the same truncated set.
        max_worlds = self.context.option("max_worlds", None)
        shown = worlds[: int(max_worlds)] if max_worlds else list(worlds)
        scored_problem = {**problem, "trainWorlds": shown}
        self._problems[instance_id] = scored_problem

        try:
            body = self._prompt_body(scored_problem, shown)
        except Exception as exc:  # noqa: BLE001 - a release layout change
            self.log.warning("ABD %s: cannot build the release prompt: %s", instance_id, exc)
            return None

        return SampleSpec(
            sample_id=C.stable_id("abd", instance_id),
            fields={
                "context": body,
                "observation": (
                    "Find the formula alpha(x) that defines Ab and makes every axiom true in "
                    "all of the worlds above, marking as few objects abnormal as possible."
                ),
            },
            reference={
                "gold": alpha,
                "description": C.normalize_whitespace(gold.get("description")),
                "allowed": [str(name) for name in C.as_list(problem.get("allowedAlphaPreds"))],
                # The worlds themselves are not stored per record: they are
                # ~11 kB each and the adapter already holds them, keyed by
                # instance id, for the whole run.
                "instance_id": instance_id,
            },
            task_kind="knowledge_completion",
            # Formal hypothesis synthesis over several worlds: room to reason.
            max_tokens=self.clamp_max_tokens(
                768 + 128 * int(description.get("alphaTier") or 1), low=768, high=1536
            ),
            metadata={
                "instance_id": instance_id,
                "scenario": problem.get("scenario"),
                "difficulty": description.get("difficulty"),
                "alpha_tier": description.get("alphaTier"),
                "n_worlds": len(shown),
            },
        )

    # ------------------------------------------------------------------ #
    # scoring -- the release's evaluator, and nothing else
    # ------------------------------------------------------------------ #

    def _evaluate(self, instance_id: str, alpha: str) -> Any | None:
        """Run the official evaluator, once per distinct (instance, formula).

        Repeats of a record often produce the same formula, and a Z3 run costs
        seconds, so the same question is never asked twice.

        The engine scores a batch concurrently, so this runs on several threads
        at once. The lock covers only the bookkeeping -- deciding who evaluates
        a given key -- and is released before the solver is called, so distinct
        formulas still run in parallel across the worker pool. Threads that
        want a key someone else is already evaluating wait for that one result
        rather than paying for a second identical Z3 run.
        """
        problem = getattr(self, "_problems", {}).get(instance_id)
        if problem is None:
            return None
        cache = self.__dict__.setdefault("_eval_cache", {})
        inflight = self.__dict__.setdefault("_eval_inflight", {})
        lock = self.__dict__.setdefault("_eval_lock", threading.Lock())
        key = (instance_id, alpha)

        with lock:
            if key in cache:
                return cache[key]
            event = inflight.get(key)
            mine = event is None
            if mine:
                event = threading.Event()
                inflight[key] = event

        if not mine:
            # Someone else is already asking this exact question.
            event.wait(timeout=self._isolated.deadline_s + 30)
            return cache.get(key)

        result = None
        try:
            timeout = int(self.context.option("z3_timeout_ms", DEFAULT_TIMEOUT_MS))
            result = self._isolated.evaluate(problem, alpha, timeout)
            cache[key] = result
        finally:
            # Always release the waiters, even if the evaluator itself threw:
            # they read the cache, find nothing, and are told the evaluator was
            # unavailable rather than blocking until the deadline.
            with lock:
                inflight.pop(key, None)
            event.set()
        return result

    def score(
        self,
        sample: SampleSpec,
        response: ModelResponse,
        *,
        output_contract: dict[str, Any] | None = None,
    ) -> SampleScore:
        headline = ["valid", "optimal", "formula_match"]
        answer = extract_answer_span(response.text, output_contract)
        if not answer:
            return unparsed_score(headline, raw=response.text[:200])
        candidate = _formula(answer)
        gold = sample.reference["gold"]
        result = self._evaluate(sample.reference["instance_id"], candidate)
        if result is None:
            # The instance is not in hand -- a re-score outside a prepared run.
            return unparsed_score(headline, raw=f"evaluator unavailable: {candidate[:160]}")
        if result.crashed:
            # The solver fell over on this answer. That is not evidence about
            # the model, so it contributes to no mean: only the crash is
            # recorded, and `valid` is absent rather than zero.
            return SampleScore(
                metrics={"evaluator_crashed": 1.0},
                prediction=candidate[:400],
                details={"gold": gold[:300], "evaluator_crash": result.crash_reason},
            )

        forbidden = list(result.forbidden_preds_used or [])
        parse_error = result.parse_error
        valid = 1.0 if result.valid else 0.0
        metrics: dict[str, float] = {
            "valid": valid,
            "evaluator_crashed": 0.0,
            "optimal": 1.0 if result.valid and result.total_gap == 0 else 0.0,
            # Kept as a diagnostic only: it answers "did the model reproduce the
            # planted gold", which is a different and much narrower question
            # than "did the model solve the instance".
            "formula_match": float(_canonical(candidate) == _canonical(gold)),
            "forbidden_predicate_use": 1.0 if forbidden else 0.0,
            "formula_parse_error": 1.0 if (parse_error and not forbidden) else 0.0,
            "trailing_parens_repaired": float(getattr(result, "trailing_parens_added", 0) or 0),
        }
        # Cost is undefined for an answer that does not repair the worlds, so
        # these keys are simply absent there and their means are over valid
        # answers only -- reporting 0 would make an invalid answer look cheapest.
        if result.valid:
            for key, value in (
                ("total_cost", result.total_cost),
                # The instance's own optimum, reported only alongside a valid
                # answer so the two means are over the same set of instances.
                ("total_opt_cost", result.total_opt_cost),
                ("total_gap", result.total_gap),
                ("avg_gap", result.avg_gap),
                ("cost_vs_gold", result.cost_vs_gold),
            ):
                if value is not None:
                    metrics[key] = float(value)
        for key in ("scenario", "difficulty"):
            stratum = sample.metadata.get(key)
            if stratum:
                metrics[f"valid_{stratum}"] = valid
        return SampleScore(
            metrics=metrics,
            prediction=candidate[:400],
            details={
                "gold": gold[:300],
                "valid": bool(result.valid),
                "total_cost": result.total_cost,
                "total_opt_cost": result.total_opt_cost,
                "parse_error": parse_error,
                "forbidden_preds_used": forbidden or None,
            },
        )

    def aggregate(self, scores: Sequence[SampleScore]) -> dict[str, float]:
        metrics = aggregate_mean_metrics([score.metrics for score in scores])
        # How many answers the cost metrics are averaged over, so a low mean
        # gap cannot be read as good when it rests on three valid answers.
        valid = sum(1 for score in scores if score.metrics.get("valid"))
        metrics["n_valid"] = float(valid)
        crashes = getattr(getattr(self, "_isolated", None), "crashes", 0)
        if crashes:
            metrics["evaluator_worker_restarts"] = float(crashes)
        return metrics

    # ------------------------------------------------------------------ #
    # documentation
    # ------------------------------------------------------------------ #

    def documentation(self) -> AdapterDocumentation:
        invalid = getattr(self, "invalid_items", [])
        anomalies = getattr(self, "gold_anomalies", [])
        caveats = [
            "Validity, not optimality, is the headline. That is the release's own ordering "
            "('Primary objective: Validity; Secondary objective: Parsimony'), and it is also "
            "the only defensible one here: the planted golds are themselves usually "
            "sub-optimal, so a suite reporting `optimal` as the score would report that the "
            "benchmark's own answer key mostly fails.",
            "A trivially true formula such as (or (P x) (not (P x))) marks every object "
            "abnormal and IS valid -- measured cost 101 against an optimum of 14 on instance "
            "0. Read `valid` beside `avg_gap`, or a degenerate answer looks like a solved one.",
            "Cost metrics are means over VALID answers only. A model that is valid three times "
            "out of a hundred can show a small mean gap; `n_valid` says how many answers the "
            "mean rests on.",
            "Equivalence is decided against the worlds in the prompt. A formula that repairs "
            "those and would fail on a fresh world from the same generator is counted correct; "
            "the release ships holdout worlds for exactly that question, and they are not used "
            "here.",
            "Each evaluation is a Z3 run of roughly one to thirty seconds. Identical answers "
            "to the same instance are evaluated once and cached, but a full sweep of this "
            "dataset spends real CPU on scoring.",
        ]
        if invalid:
            # Every excluded record, not a truncated sample: a caveat that says
            # "40 excluded, here are 6" is not documentation, it's a promise
            # about the other 34 that nothing checks. The full list is what
            # "reported here" in the closing sentence below actually means.
            caveats.insert(
                0,
                f"{len(invalid)} canonical record(s) are EXCLUDED as invalid benchmark items, "
                "not silently repaired: "
                + "; ".join(f"{row['instance_id']} -- {row['reason']}" for row in invalid)
                + ". Each is a record whose own gold the release's parser cannot read, or whose "
                "gold uses predicates the record itself forbids. They are reported here and in "
                "the run statistics rather than patched, because patching a released answer key "
                "would change what a published number means.",
            )
        if anomalies:
            caveats.insert(
                1 if invalid else 0,
                f"{len(anomalies)} gold formula(s) are malformed but recoverable by the "
                "release's own suffix repair, and are kept on that basis: "
                + "; ".join(f"{row['instance_id']}" for row in anomalies[:8])
                + ("; ..." if len(anomalies) > 8 else "")
                + ".",
            )
        return AdapterDocumentation(
            dataset_id=self.dataset_id,
            name="ABD",
            domain="Formal Reasoning: Default-Exception Logic",
            source_url=REPO_URL,
            processing_mode="Generation",
            split_used=getattr(self, "split_used", "abd_instances_v1.yaml.gz"),
            abductive_subset=(
                "The abduction benchmark only (benchmarks/abduction/). The repository's "
                "induction benchmark is a different inference type, and its eval/ and "
                "predictions/ directories hold model outputs, which are never read. Every "
                "instance is abductive: a theory that does not fit its worlds, and the "
                "question of what hypothesis about Ab would make it fit."
            ),
            sampling_procedure=self.sampling_note(),
            metrics_description={
                "valid": (
                    "(PRIMARY, higher is better, 0-1) 1 when the release's own Z3 evaluator "
                    "proves the formula repairs EVERY prompt world under that instance's "
                    "observation regime -- a satisfying completion exists for ABD-Partial, "
                    "every completion satisfies for ABD-Skeptical. It does not matter whether "
                    "the formula resembles the gold."
                ),
                "optimal": (
                    "(higher is better, 0-1) 1 when the answer is valid AND its total cost "
                    "equals the solver's lower bound, i.e. total_gap is 0. A strict bar: the "
                    "planted golds usually do not clear it."
                ),
                "total_cost": (
                    "(lower is better) abnormal objects summed over the prompt worlds. Mean "
                    "over VALID answers only; undefined otherwise."
                ),
                "total_opt_cost": (
                    "the solver's lower bound, averaged over the same instances total_cost "
                    "is averaged over -- a property of the sampled instances, not of the model"
                ),
                "total_gap": (
                    "(lower is better) total_cost minus the solver's optimum. Zero means "
                    "optimal. Mean over valid answers only."
                ),
                "avg_gap": (
                    "(lower is better) total_gap divided by the number of prompt worlds -- the "
                    "per-world excess, comparable across instances with different world counts"
                ),
                "cost_vs_gold": (
                    "(lower is better; negative is better than the gold) total_cost minus the "
                    "planted gold's cost. Negative values are real and expected: the gold is "
                    "one valid repair, not the cheapest."
                ),
                "formula_match": (
                    "(DIAGNOSTIC, higher is better, 0-1) exact text equality with the planted "
                    "gold after canonicalising whitespace, parentheses and case. Deliberately "
                    "NOT the score: it answers whether the model reproduced the answer key, "
                    "which a valid and cheaper repair does not have to do."
                ),
                "forbidden_predicate_use": (
                    "(lower is better, 0-1) fraction of answers the evaluator rejected for "
                    "using a predicate outside the instance's allowed set -- Ab itself most "
                    "often, which would be a circular definition. Such answers are invalid."
                ),
                "formula_parse_error": (
                    "(lower is better, 0-1) fraction the release's parser could not read, or "
                    "that failed evaluation for any reason other than predicate scope. These "
                    "are invalid, not undecidable: a formula that does not parse repairs "
                    "nothing, and there is no judge here to ask about it."
                ),
                "trailing_parens_repaired": (
                    "(diagnostic) closing parentheses the release's own suffix repair added to "
                    "the model's answer before evaluating it. The repair is the release's "
                    "behaviour, kept rather than overridden, and counted rather than hidden."
                ),
                "valid_<scenario>": "the primary metric per regime (FULL / PARTIAL / SKEPTICAL)",
                "valid_<difficulty>": "the primary metric per difficulty label",
                "n_valid": "how many answers the cost means are computed over",
                "evaluator_crashed": (
                    "(lower is better, 0-1) fraction of answers on which the native solver "
                    "aborted or hung. Those answers are scored on NOTHING -- `valid` is absent "
                    "rather than 0, because a solver that fell over says nothing about the "
                    "model -- so this number must be read beside the rest."
                ),
                "evaluator_worker_restarts": (
                    "how many times the isolated solver process had to be replaced during the "
                    "task; present only when it happened"
                ),
                "parse_failure_rate": (
                    "(lower is better, 0-1) fraction of responses no answer span could be "
                    "extracted from at all, before the release's parser ever saw one"
                ),
                "self_consistency_<metric>":
                "Every metric also gets a self_consistency_ counterpart: the plurality answer "
                "over modes.repeats samples of the same record, read off those samples rather "
                "than bought again. Available because this dataset's answers are checkable. It "
                "votes on the formula as written, so two different valid repairs do not pool "
                "their votes -- read it as a floor.",
            },
            primary_metric="valid",
            decisions=[
                "SCORED WITH THE RELEASE'S OWN Z3 EVALUATOR (concept_synth.abduction."
                "evaluate_abd_b1), imported from the same clone the data comes from rather "
                "than vendored, so the code that decides what is correct cannot drift from the "
                "data it decides about.",
                "Replaced gold-matching as the headline metric. It was wrong, not merely "
                "strict: on the release's own instance 0 the formula "
                "'(exists y (and (S x y) (P y)))' is a valid repair at cost 34, and any "
                "gold-matching metric scores it 0. formula_match is kept as a diagnostic.",
                "Made `valid` primary rather than `optimal`, following the benchmark's own "
                "stated objective ordering; `optimal` is reported beside it.",
                "Built the prompt from the release's own task template and world formatters, "
                "replacing only its closing JSON output contract with this suite's answer "
                "contract. The substitution semantics, closed-world rule, completion semantics "
                "per regime, cost objective and formula grammar are all things the evaluator "
                "checks, so none of them can be left out or paraphrased.",
                "Showed every training world by default, and scored against exactly the worlds "
                "shown; options.max_worlds truncates both together for a partial-evidence "
                "ablation.",
                "Kept the release's default 5000 ms Z3 timeout (options.z3_timeout_ms): a "
                "different timeout is a different benchmark.",
                "Excluded canonical records whose own gold the release's parser rejects or "
                "whose gold breaks the record's own allowedAlphaPreds, and named them, rather "
                "than repairing the released data.",
                "Dropped the LLM judge entirely. The evaluator decides validity exactly, "
                "including for answers that do not parse, and a judge could only add noise to "
                "a solver's verdict.",
            ],
            caveats=caveats,
            statistics={
                **self.base_statistics(),
                "instances_per_scenario": getattr(self, "_per_scenario", {}),
                "invalid_benchmark_items": invalid,
                "gold_anomalies": anomalies,
                "z3_timeout_ms": int(self.context.option("z3_timeout_ms", DEFAULT_TIMEOUT_MS)),
            },
        )


def _formula(text: str) -> str:
    """Extract the first balanced s-expression from a model answer, else the text.

    The release's own ``extract_alpha_from_response`` looks for its JSON output
    contract, which this suite replaces with its answer contract, and its
    fallback regex stops at the first ``)``. So extraction is done here, where
    the prompt is known, and only *evaluation* is the release's.
    """
    body = text.strip().strip("`")
    start = body.find("(")
    if start < 0:
        return body
    depth = 0
    for position in range(start, len(body)):
        if body[position] == "(":
            depth += 1
        elif body[position] == ")":
            depth -= 1
            if depth == 0:
                return body[start : position + 1]
    return body[start:]


def _canonical(formula: str) -> str:
    """Whitespace/parenthesis/case-insensitive canonical form."""
    compact = re.sub(r"\s+", " ", (formula or "").strip().lower())
    return re.sub(r"\s*([()])\s*", r"\1", compact)
