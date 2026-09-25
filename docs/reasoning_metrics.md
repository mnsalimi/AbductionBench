# Reasoning metrics -- definitions

What every `reasoning_*` column means, how it is computed, and where it comes
from. Source of truth: `src/abductionbench/core/reasoning_judge.py`
(`derive_reasoning_metrics`, `FAMILY_METRICS`) and the judge prompts in
`configs/prompts/judge/reasoning_*.yaml`. Written 2026-09-25 against
`reasoning_steps_v4`, `reasoning_anchoring_point_v3@1.1` and the other
templates named in `core/config.py::_reasoning_judge_templates`.

## How the metrics are produced

* **Only cot samples are measured.** The judge reads the model's reasoning
  trace: the visible chain of thought if there is one, otherwise the native
  reasoning (`details.reasoning_source` = `visible_cot` / `native`). A reply
  that never gave an answer is not judged (`not_applicable:no_answer:*`).
* **The judge (gpt-oss-120b, reasoning effort low, temperature 0) is asked for
  counts and per-step lists only.** Every ratio, fraction and normalization is
  computed afterwards in code, so no judge ever sees a denominator.
* **One segmentation governs everything.** The `steps` judge cuts the chain
  into steps once (spans validated against the chain text). Every per-step
  list from the other judges must have exactly `reasoning_total_steps`
  elements, or it is rejected (the metric is left blank and the reason goes to
  `details.reasoning_judge_errors`) -- never padded or truncated.
* **Blank is not zero.** A metric that does not apply to a task shape is
  blank with a reason in `details.reasoning_metrics_inapplicable`; one the
  judge could not produce is blank with a reason in
  `details.reasoning_judge_errors`. `reasoning_anchoring_point(_normalized)`
  is written as the string `"None"` when the model never considered its own
  final answer, and averages skip it.
* **Two waves of calls.** Wave one: `steps`, `observation_inventory` (the
  question's observations -- the coverage denominator), `option_count`
  (selection tasks). Wave two, given the step list: every other family.
* **Task shapes.** *Selection* (single/multi choice from given options) vs
  *generation* (free-form answer). A few metrics exist for one shape only.

## The metrics

`n` = `reasoning_total_steps`. "Per step" lists live in
`details.reasoning_lists` under the name shown.

| # | Column | Judge family (prompt) | Computed as | Range | Meaning |
|---|---|---|---|---|---|
| 1 | `reasoning_observations_total` | observation_inventory | number of observations the judge lists from the question | ≥1 | the facts any answer must account for |
| 1 | `reasoning_observations_used` | observation_coverage | sum of `reasoning_observations_per_step` (each observation counted once, at its first appearance) | 0..total | how many of those facts the chain actually draws on |
| 1 | `reasoning_observation_coverage` | observation_coverage | used / total | 0..1 | share of the evidence the reasoning uses. Left blank (error) if used > total |
| 2 | `reasoning_total_steps` | steps (`reasoning_steps_v4`) | length of the step list | ≥1 | chain length in inferential moves; blank if the chain has nothing to segment |
| 2 | `reasoning_useful_steps` / `_useless_steps` | proof_disproof | steps whose proof/disproof count is >0 / =0 | 0..n | steps that do or do not prove/disprove any hypothesis |
| 2 | `reasoning_useful_step_fraction` / `reasoning_useless_step_fraction` | proof_disproof | useful/n, useless/n | 0..1 | |
| 2 | `reasoning_backtracking_steps` | unresolved_contradiction (`reasoning_contradiction_v2`) | sum of `reasoning_backtracking_per_step` (1 = the model notices an earlier error/contradiction and corrects or acknowledges it) | 0..n | self-correction |
| 2 | `reasoning_backtracking_rate` | same | backtracking steps / n | 0..1 | |
| 3 | `reasoning_branchiness_total` | branchiness_selection / _generation | sum of `reasoning_branchiness_per_step` (NEW distinct hypotheses raised in each step; revisits do not count) | ≥0 | how many different candidates the chain considered |
| 3 | `reasoning_diversity` | branchiness_generation | judge's 0/1: are the explanations raised meaningfully different (different mechanisms), not rewordings | 0/1 | generation only |
| 3 | `reasoning_option_count` | option_count | judge's count of answer options in the question | ≥0 | selection only; the question's own option count is preferred where known |
| 4 | `reasoning_directionality` | directionality | judge's single value for the whole chain: 1 = reasons from evidence toward an explanation; 0 = assumes/guesses an explanation first and justifies it forward; 0.5 = mixed | 0 / 0.5 / 1 | abductive (evidence-first) vs confirmatory (guess-first) |
| 5 | `reasoning_step_directionality_mean` | step_directionality | mean of per-step values (same 1 / 0 / 0.5 scale) | 0..1 | the same question asked step by step |
| 6 | `reasoning_differential_elimination` | differential_elimination | sum of `reasoning_comparisons_per_step` (COMPARISON EVENTS; one event comparing 3+ hypotheses counts 1) | ≥0 | how often hypotheses are weighed against each other |
| 6 | `reasoning_differential_elimination_normalized` | same | share of steps with at least one comparison | 0..1 | |
| 7 | `reasoning_comparison_exhaustiveness` | same | comparisons / C(h,2), h = option count (selection) or distinct hypotheses (generation) | ≥0 (can exceed 1) | how many of the possible pairs were compared; blank if h < 2 |
| 8 | `reasoning_uncertainty_steps` | uncertainty | sum of per-step counts of uncertainty MARKERS ("might", "possibly", ...) | ≥0 | hedging |
| 8 | `reasoning_uncertainty_rate` | same | share of steps with at least one marker | 0..1 | |
| 9 | `reasoning_prior_knowledge` | prior_knowledge | sum of per-step counts of non-trivial background facts NOT in the question that the reasoning leans on | ≥0 | reliance on world knowledge |
| 9 | `reasoning_prior_knowledge_normalized` | same | share of steps with at least one such reference | 0..1 | |
| 10 | `reasoning_anchoring_point` | anchoring_point (`v3@1.1`) | 1-based index of the FIRST step in which the model considers ITS OWN FINAL ANSWER as a candidate; `"None"` if it never does before the answer | 1..n or None | when the eventual answer entered consideration |
| 10 | `reasoning_anchoring_point_normalized` | same | index / n | (0,1] or None | 1.0 = the answer first appears at the last step |
| 10 | `reasoning_gold_alive_steps` | same | sum of `reasoning_gold_alive_per_step` (1 = the CORRECT answer is in play in that step -- raised, weighed, argued for or against) | 0..n | engagement with the right answer |
| 10 | `reasoning_gold_alive_rate` | same | gold-alive steps / n | 0..1 | |
| 11 | `reasoning_unresolved_contradictions` | unresolved_contradiction | sum of `reasoning_unresolved_per_step` (1 = the step contradicts an earlier one and the model never corrects or acknowledges it) | 0..n | incoherence left standing |
| 11 | `reasoning_unresolved_contradiction_normalized` | same | unresolved / n | 0..1 | |
| 12 | `reasoning_proof_disproof_total` | proof_disproof | sum of per-step counts of hypotheses proved or disproved | ≥0 | |
| 13 | `reasoning_mentions_total` | branchiness_* | sum of `reasoning_mentions_per_step` (EVERY reference to a hypothesis, new or not) | ≥ branchiness | revisiting |
| 13 | `reasoning_mention_ratio` | same | mentions / branchiness | ≥1 | how often each hypothesis is revisited |
| 14 | `reasoning_helpfulness_mean` | helpfulness | mean of per-step values: 1 = HELPFUL (moves toward the reference answer; a step that catches and fixes an error is 1), 0 = neutral, -1 = HARMFUL | -1..1 | the only signed metric |
| 14 | `reasoning_helpful_steps` / `_neutral_steps` / `_harmful_steps` | same | counts of 1 / 0 / -1 | 0..n | |
| 14 | `reasoning_helpful_fraction` / `_neutral_fraction` / `_harmful_fraction` | same | counts / n | 0..1 | |

### Invariants the code guarantees (useful when looking for bugs)

* every per-step list has exactly `n` elements;
* `useful + useless = n`; `helpful + neutral + harmful = n`;
* `observations_used ≤ observations_total` (otherwise coverage is blank);
* `1 ≤ anchoring_point ≤ n`;
* `mentions_total ≥ branchiness_total` is expected but NOT enforced;
* `directionality` (one call over the whole chain) and
  `step_directionality_mean` (one call per step) are separate judgements and
  can disagree.

### Known caveats

* Anchoring values produced before 2026-09-25 (`anchoring_point_v3@1.0`)
  count steps inconsistently (0- or 1-based); they are re-asked with v1.1.
* Before the per-family replacement fix, a re-judge could leave a metric from
  an earlier round beside newer ones (~10.7% of trio cot samples). The latest
  round per sample is in `reasoning_metrics.jsonl`.
* The inapplicable reason `anchoring_point:correct_answer_never_considered`
  is misnamed: a null anchoring point means the model never considered ITS
  OWN final answer during the chain, not the correct one.
