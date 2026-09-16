# Temporary implementation handoff: COT reasoning-chain metrics

Date: 2026-09-16
Branch: `main`
Purpose: temporary handoff for the next maintainer; this file may be deleted
after review.

## Requested change

Add LLM-judged metrics for the quality and structure of chains of reasoning.
They must run on COT outputs only, never IO outputs; grouped metrics must use one
judge prompt/call; shared normalizers must be reused; every raw and normalized
value must appear per sample in the result sheet; and genuinely inapplicable or
failed measurements must be reported rather than silently set to zero.

## What was implemented

- Added the independent `engine.reasoning_judge` configuration block. It is
  separate from the existing semantic-answer `engine.judge` stage, can use a
  different configured model, and is disabled by default because it incurs
  model calls.
- Added `src/abductionbench/core/reasoning_judge.py`.
  - Runs only for `cot` and `self-consistency` prompt modes.
  - Returns IO scores unchanged and makes no reasoning-judge calls for IO.
  - Uses the full rendered static question. For interactive/sequential tasks it
    separates non-assistant transcript messages as the supplied question/evidence
    and assistant messages as the produced reasoning chain.
  - Uses `response.reasoning` when the endpoint exposes a separate reasoning
    field; otherwise it analyzes the response content requested by the COT prompt.
  - Strictly parses JSON, validates types/ranges and the invariant
    `useful_steps + useless_steps == total_steps`, and omits invalid/undefined
    values instead of coercing them.
  - Caches the question-only observation inventory by exact rendered question
    across models and repeats. All other verdicts are also cached by versioned
    template plus exact fields.
  - The stage is shared across concurrent tasks and serialized around cache
    population so two model tasks cannot buy the same observation inventory at
    the same time.
  - Resumed COT records are reconstructed and judged too. A scoring-code change
    does not alter the inference fingerprint, so reusable model answers from an
    older checkpoint no longer remain blank merely because they predate the new
    metrics.
- Corrected post-judge persistence: judge-updated `SampleScore` objects are
  appended as replacement records. Record de-duplication selects the newest
  fingerprint, so per-sample metrics now reach CSV/Excel sheets rather than
  existing only in task aggregates.
- Added three sample-sheet audit columns:
  - `reasoning_metrics_status`
  - `reasoning_metrics_inapplicable`
  - `reasoning_judge_errors`
- Added documentation to `README.md` and `configs/prompts/README.md`.

## Judge prompts and call grouping

All prompts are versioned YAML under `configs/prompts/judge/` and return one
strict JSON object with no free-form fields.

| Family | Template | Outputs in its single call |
|---|---|---|
| one-time observation inventory | `reasoning_observation_inventory_v1` | `total_observations` |
| observation coverage | `reasoning_observation_coverage_v1` | `total_observations`, `observations_used` |
| branchiness/diversity | `reasoning_branchiness_diversity_v1` | `branchiness`, `diversity` |
| density | `reasoning_density_v1` | `total_steps`, `useless_steps`, `useful_steps`, `reasoning_density` |
| redundancy/completeness | `reasoning_redundancy_completeness_v1` | `redundancy`, `completeness` |
| directionality | `reasoning_directionality_v1` | `directionality` |
| backtracking | `reasoning_backtracking_v1` | `backtracking` |
| differential elimination | `reasoning_differential_elimination_v1` | `differential_elimination` |
| prior knowledge | `reasoning_prior_knowledge_v1` | `prior_knowledge` |
| uncertainty marking | `reasoning_uncertainty_v1` | `uncertainty_steps` |

The separate inventory call is the explicit optimization allowed in the task:
coverage still returns its required two outputs, but copies the cached total and
computes only the output-dependent used count.

## Per-sample metric columns

The sample sheet prefixes numeric metrics with `metric.`. The underlying metric
names are:

- `reasoning_observations_total`
- `reasoning_observations_used`
- `reasoning_observation_coverage`
- `reasoning_branchiness`
- `reasoning_diversity`
- `reasoning_total_steps`
- `reasoning_useless_steps`
- `reasoning_useful_steps`
- `reasoning_density`
- `reasoning_useless_step_fraction`
- `reasoning_useful_step_fraction`
- `reasoning_density_normalized`
- `reasoning_redundancy`
- `reasoning_completeness`
- `reasoning_redundancy_normalized`
- `reasoning_completeness_normalized`
- `reasoning_directionality`
- `reasoning_backtracking`
- `reasoning_backtracking_normalized`
- `reasoning_differential_elimination`
- `reasoning_differential_elimination_normalized`
- `reasoning_prior_knowledge`
- `reasoning_uncertainty_steps`
- `reasoning_uncertainty_normalized`

Shared denominators are calculated locally, not re-judged:

- coverage, redundancy and completeness use the cached observation total;
- backtracking and uncertainty use the density prompt's total-step count;
- generation and pipeline density use Branchiness;
- selection density uses the number of visible answer options;
- differential elimination uses `2**n - n - 1`, equivalent to the requested
  sum of unordered combinations `C(n,k)` for `k=2..n`.

Pipeline tasks are detected from adapter documentation that declares
`Generation & Selection`; they receive generation branchiness and use it for
density normalization, while differential elimination runs only when a real
selection-stage option list is available. BOV is explicitly reported as
inapplicable for differential elimination because each BOV chain sees one
option rather than the full differential.

## Configuration example

```yaml
engine:
  reasoning_judge:
    enabled: true
    model: gpt-oss-120b   # must also appear in this run's models list
    max_tokens: 2048
    temperature: 0.0
    group_size: 8
    cache: true
```

The resolved run configuration records all ten template IDs, so a metric is
auditable against the exact prompt version that produced it.

## Tests added and validation performed

Added `tests/test_reasoning_judge.py`, covering:

- exact raw/derived values for generation;
- different selection versus pipeline density normalizers;
- the unordered-combination denominator for differential elimination;
- invalid step-count invariants being reported rather than forced;
- COT integration through the real engine and local fake model server;
- one-time observation inventory reuse across repeated outputs;
- persistence of every expected sample-sheet column;
- resume behavior and verdict-cache reuse;
- the hard guarantee that IO outputs never call this judge and never receive
  reasoning metric keys.

Validation results:

- `.venv/bin/ruff check src tests tools` — passed.
- Focused judge/config/prompt suite — **28 passed**.
- Full Python/fake-server suite excluding the shell sidecar tests —
  **191 passed**.
- The two `tests/test_sync_sidecar.py` tests were not rerun successfully on this
  macOS host because the external `flock` executable is absent. This is unrelated
  to the reasoning-metric code; the rest of the sync tests passed.
- Python AST parsing and all YAML parsing passed.

The existing test suite additionally requires optional `sympy` and `tabulate`
packages; they were installed only in the ignored local `.venv` for validation.
No dependency manifest was changed as part of this feature.

## Live-model limitation

No real model endpoint or API key was available. The configured temporary
Cloudflare tunnel URLs no longer resolved and no local vLLM port was listening.
Therefore the implementation was exercised end-to-end with the repository's
fake OpenAI-compatible batch server, including actual judge calls, parsing,
caching, record replacement, aggregation, and sample-sheet construction, but
the wording has not yet been calibrated against a live judge model.

Recommended first live check: run two samples from one generation adapter and
one selection adapter with `prompt_modes: [io, cot]`, verify that only the COT
rows contain `metric.reasoning_*` columns, then manually compare each strict
JSON verdict with the stored chain before launching the full suite.

## Repository hygiene

The pre-existing user modification to `.gitignore` and untracked `.DS_Store` /
`.history` files were deliberately left out of this feature commit.


## Review pass over the reasoning-metric stage

Six defects found reading the stage back, each with a test:

1. **A model with no usable batch route lost every reasoning metric.**
   `chat_single` answers only the first conversation of a group, so a group of
   eight came back with one answer and the strict `zip` raised; the engine
   caught it and the whole task's metrics were dropped. Groups are now one item
   wide whenever batching is unavailable, and a count mismatch drops the group
   with a warning instead of the run's metrics. The same defect was in the
   answer judge and is fixed there too.
2. **The batch fallback the engine decides at startup was invisible to both
   judges.** `engine._batch_disabled` is now passed to them by reference, so a
   model whose batch probe failed structurally is judged with single calls
   rather than through a route already known to be dead.
3. **`multi_selection` and `knowledge_completion` were judged as neither shape.**
   Eight datasets use those kinds. They now map to selection and generation
   respectively; a kind that maps to neither is skipped with a reason, and a
   normalized density that no branch defines is reported as an error instead of
   quietly missing. `test_every_shipped_task_kind_has_a_reasoning_shape` fails
   if a new kind is added without a shape.
4. **An unparseable verdict was cached as a permanent failure.** Only parsed
   verdicts are cached now, so continuing a run retries what it could not read.
5. **`max_tokens: 512` is too small for a reasoning judge.** gpt-oss-120b
   spends the budget on its hidden chain and returns empty content. The default
   is 2048; it is a ceiling, so a terse judge still costs what it costs.
6. **Rewriting a checkpointed record duplicated it under a non-strict resume
   policy.** Records dedupe on (sample_id, fingerprint), and the rewrite
   recomputed the fingerprint; it now keeps the record's own.

Also made the verdict parser read the last JSON object in a reply rather than
one greedy span from the first brace to the last, so a judge that reasons in
the open, echoes the example object, or writes a stray brace is still read.

Validation: `ruff check src tests` passed, full suite **201 passed**.


## First live calibration (2026-09-16), and what it changed

Two datasets, 6 records, 1 repeat, judged by gpt-oss-20b on this box. The stage
worked: 53 of 54 verdicts parsed on the first CoT task, and the metrics were
coherent. Two failures, both real, both now fixed.

**The judge's budget was the binding constraint, measured not guessed.** Against
defab's longest chain (18,239 characters) the density verdict took **5,220
completion tokens**; at 2,048 the call came back `finish_reason=length` with
empty content, which is the empty-reply warning seen eight times during the run.
Redundancy/completeness landed at 1,974 -- inside 2,048 by 74 tokens, which is
why it failed only sometimes. The default is now **8,192**. Worth recording:
vLLM does not split this model's analysis into `reasoning_content`, so those
thousands of tokens are counted against the budget and then discarded, and the
JSON arrives inline -- which is what makes reading the *last* JSON object out of
a reply load-bearing rather than defensive.

**A judge counted more evidence than the question contains.** One eCARE sample
came back `completeness: 3` against a 2-observation inventory; the validator
caught it and dropped the value rather than recording it. The cause is that the
prompt never showed the judge the inventory its counts are normalized against.
`reasoning_redundancy_completeness_v2` supplies the total and states the bound
(neither count exceeds it, and their sum does not either, because an observation
is redundant or necessary and never both). The same sample now returns
`completeness: 1`. v1 is left in place; a template that has produced a number is
not edited underneath it.

Validation: `ruff check src tests` clean for the files touched, full suite
**229 passed, 1 skipped**.


## Nine defects from the code-review pass (2026-09-16)

1. **Ctrl-C was retried.** `with_retry` caught `BaseException` and the classifier
   filed `KeyboardInterrupt` as `unknown`, which is retryable by default, so an
   interrupt re-issued the batch four more times over ~30 s of backoff before it
   got out. It now catches `Exception`, with `KeyboardInterrupt`/`SystemExit`
   re-raised explicitly next to `CancelledError`.
2. **An out-of-budget reply was filed as empty.** `finish_reason=length` is now
   checked *before* the content test, so a reasoning model that spends its whole
   budget thinking is `truncated`, not `empty` -- the only cause a larger budget
   fixes is no longer indistinguishable from a server that answered with
   nothing. The case keeps its own metric, `empty_after_truncation_rate`, and a
   warning naming the budget. Resume gained `_outgrew_its_budget`: a truncated
   record is re-asked when the run now allows more tokens than produced it,
   which is what keeps the `sample_id` policy honest (`strict` already had the
   budget in its fingerprint).
3. **The sample sheet truncated silently and unevenly.** It filled in task order
   and stopped at `max_sample_rows_per_sheet`, so the cap was spent on the first
   task and the later ones -- the cot tasks, since io runs first -- had no rows
   at all. `_rows_per_task` now gives every task an equal share, hands back what
   a small task does not use, and the drop is logged.
4. **`update_scores` matched on `sample_id` alone** while records dedupe on
   `(sample_id, prompt_fingerprint)`, so it patched every fingerprint variant of
   a sample rather than the judged one.
5. **A free-text dataset could not report a parse failure.**
   `extract_answer_span` falls back to the whole response, so the unparsed
   branch never fired and `parse_failure_rate` read 0.0 by construction. The new
   `answer_span_is_marked` says whether the template's declared marker was
   actually there; `text_match_score` still scores the fallback text -- the
   answer may well be in it -- but sets `parse_ok=False` and records
   `answer_marker_missing`.
6. **Two worst-case values that looked like measurements.** `mean([])` returned
   0.0 and `brier_score` returned 1.0 for an unreadable probability. Both are
   `nan` now, which is what the aggregator and the record serializer already
   read as "not measured".
7. **`numeric_match` used its relative tolerance as an absolute one** against a
   reference of zero. Same default, but it is named `abs_tol` and can be set.
8. **Two mechanisms for one job.** The answer judge rewrote records in place
   while the reasoning judge appended and let de-duplication retire the old row;
   the append path already covered the answer judge's records, so the in-place
   rewrite was redundant as well as mis-keyed, and is gone. The answer judge's
   verdict cache moved from per-task to per-run, matching the reasoning judge's.
18. **`kendall_tau` claimed tau-a and computed neither tau.** Tied pairs were
   dropped from both counts *and* the normalizer, which reports 1.0 for data
   whose ranks mostly coincide by being tied. It is tau-b now.

Deliberately not changed: an empty or truncated answer is still *scored* rather
than counted as an error. Reclassifying it would move every published number in
the suite, and the signal is already there in `parse_failure_rate`,
`truncation_rate` and now `empty_after_truncation_rate`.

Validation: `ruff check src tests` shows the same 24 pre-existing errors as
before the pass and no new ones; full suite **236 passed, 1 skipped**.


## Making the reasoning judge fast (2026-09-16), measured

**Why it was slow.** The stage issued roughly **171 sequential round-trips per
task**: nine metric families awaited one after another, and inside each family
150 samples split into 19 chunks of 8, also awaited one after another. Every
round-trip is a gpt-oss call that generates thousands of hidden reasoning tokens
before its one-line JSON. A run-wide lock then serialized the whole stage across
tasks, so at `max_parallel_tasks: 3` only one task could judge at a time. The
judge server offers 64 scheduler slots; the stage used 8.

**What changed.** Families now run in two dependency waves -- only coverage and
redundancy need the inventory total, and only backtracking and uncertainty need
density's step count; the other six waited for nothing. Chunks within a family
go out together. A single run-wide semaphore bounds the whole stage at
`group_size x max_parallel_calls = 8 x 8 = 64`, the judge's `--max-num-seqs`
exactly. The lock now covers only the observation-inventory purchase, which is
the one genuinely shared buy.

**Measured**, ecare, cot only, same budget and prompts on both sides:

| sample size | before | after | |
|---|---|---|---|
| 6 records, 2 datasets | 378 s | 193 s | **1.96x** |
| 32 records | 801 s | 257 s | **3.12x** |

The gain grows with sample size because chunk parallelism does not engage below
`group_size`. At the suite's 150 samples per task there are ~19 chunks per
family rather than 1, so 3.1x is a floor rather than a ceiling.

**No OOM risk, by construction.** vLLM preallocates its KV cache at startup from
`GPU_MEMORY_UTILIZATION` (0.40 + 0.36 = 71 GB of 98 GB here, 27 GB free).
Client-side concurrency allocates no GPU memory; requests beyond
`--max-num-seqs` queue in the scheduler. The concurrency was sized to land on 64
so nothing even queues.

**`group_size` stays at 8.** Raising it to 16 was tried and reverted: vLLM at
temperature 0 is not bit-stable across batch sizes, and the density judge's step
counts moved by a quarter. The speed comes from the concurrency, so the batch
size is held where earlier runs had it and their numbers stay comparable.

### What the measurement turned up

Running the **identical code twice** moved `reasoning_backtracking` from 0.032
to 1.568 -- more than any difference the speed work caused. The cause was one
sample: the judge answered **147 backtracks for a 14-step chain**, and
`derive_reasoning_metrics` recorded that raw 147 while withholding only the
ratio. A backtrack is a step, so a count above the step count is not a value
missing its normalizer; it is a judge that did not count. Both backtracking and
uncertainty now drop the raw value too, with `:exceeds_total_steps` recorded.

That fixes the outlier. It does not fix the underlying noise: `total_steps` for
one sample read 43 in one run and 4 in the next, and 28 of 94 samples changed
their step count between identical runs. The step-segmentation prompt is not
reproducible enough for a per-sample number to be trusted, and that is a prompt
problem rather than a code one. Aggregate means over 150 samples are far
steadier than the per-sample values, but `reasoning_total_steps` and everything
derived from it should be read with that in mind.

Validation: full suite **237 passed, 1 skipped**.


## The answer judge was sequential too (2026-09-16)

The reasoning judge was parallelized; the answer judge was not, and it scores
**29 of the 44 datasets** -- the ones with no verifiable answer, where its
verdict *is* the metric. It issued its batches strictly one after another, the
same pattern that had just been fixed next door. It now runs bounded concurrent
chunks like the reasoning judge does.

**The first attempt made things slower, and the reason is worth recording.**
`JudgeStage` is constructed inside `_run_task`, once per task, so a semaphore
held on the stage bounds a *task* rather than the run: at
`max_parallel_tasks: 3` that is 3 x 8 calls of 8 conversations = 192 sequences,
on top of the reasoning judge's 64, against a server with `--max-num-seqs 64`.
Oversubscribing vLLM that far pushes it into KV-cache preemption and recompute,
which costs more than the concurrency buys. The budget is now one semaphore
created by the engine and shared by both stages and every task, so in-flight
sequences stay at `group_size x max_parallel_calls` no matter which stage or
which task issues them -- the right unit is the judge *server*, not the stage.

**No clean speed number for this one.** The two measurements taken after the
change ran while a full production run was using both GPUs, and read 674 s and
517 s against 178-319 s for the same configuration on a quiet machine. Those
numbers say nothing about the change; they say a benchmark was run on a busy
box. The earlier 3.12x for the reasoning judge was measured before that run
started and stands.

### `reasoning_effort: low` -- measured, and not adopted

gpt-oss accepts `reasoning_effort`, and it is a large lever: **4.78x fewer
completion tokens** across 50 judge calls over 10 real chains. It was rejected
on the evidence, because it moves the verdicts:

| family | tokens default | tokens low | verdicts unchanged |
|---|---|---|---|
| density | 6778 | 1143 | 0/10 |
| backtracking | 3476 | 94 | 4/10 |
| observation_coverage | 1068 | 103 | 6/10 |
| directionality | 213 | 72 | 10/10 |
| uncertainty | 3426 | 1718 | 9/10 |

The control matters as much as the result: running the **default twice** agrees
only **34/50**, against 29/50 for default-vs-low. So `low` is a real but modest
degradation on top of a judge that is already unreliable at counting --
`density` agrees with itself 2 times in 10. `directionality` and `uncertainty`
are unaffected either way and could take `low` safely, which would need
per-family effort settings that do not exist yet.

Two further levers found and left alone: prefix caching is enabled for the model
under test but not for the judge server, and the reasoning chain -- up to 30k
tokens -- is re-sent to all nine families behind a family-specific system
prompt, so none of it is a shared prefix. Moving the family instruction to the
end of the user message would make the chain cacheable across all nine calls,
but it changes the prompts, so it changes the measurements.
