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
    max_tokens: 512
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
