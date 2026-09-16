# Temporary implementation handoff: COT reasoning-chain metrics

Date: 2026-09-16
Published branch for the current handoff: `codex/coworker-ready`
Purpose: temporary handoff for the next maintainer; this file may be deleted
after review.

## Coworker startup follow-up

The original metrics branch was based on an older project snapshot. The current
handoff was built in an isolated worktree from `origin/main` at `ff10bb4`, with
the reasoning-metrics commit integrated as `d0fa4ee`. Current run configs,
adapters, judge-only models, and prompt changes were preserved. The original
dirty user workspace was not switched or overwritten.

New startup artifacts:

- `docs/coworker_setup.md`: complete fresh-clone guide, including Linux/macOS,
  Python dependencies, correct branch, model URLs/SSH forwards, private keys,
  existing rclone JSON import, doctor/smoke/full run, verification, and resume.
- `.env.coworker.example`: placeholders only; correct model and judge localhost
  ports (18001 and 18003) and correctly named key variables.
- `configs/runs/coworker.yaml`: current full suite with COT reasoning analysis
  enabled using the independently served `gpt-oss-20b-local` judge.
- `configs/runs/coworker_smoke.yaml`: two generation and two selection examples,
  IO and COT, one repeat, no dataset downloads, and a bounded model output budget.

Startup/sync corrections:

- Existing Drive JSON can be imported via `--token-stdin` or `--token-file`,
  avoiding literal secrets in shell history. A durable token must include a
  refresh token. The script uses private file permissions, preserves unrelated
  remotes, makes unique backups, and fails rather than claiming ready if read
  access fails.
- New full runs and the sidecar use `gdrive:` because the setup already pins the
  remote to the shared folder. The previous `gdrive:AbductionBench` base created
  an extra nested directory. Existing nested backups are left unchanged; the
  guide documents explicitly selecting their old base when restoring.
- `raw/` debug payloads are excluded by default, consistently with the documented
  Drive quota behavior; records, responses, metrics, logs, and reports are included.
- Failed remote checks cannot emit `sync_verified`. Authentication/listing errors
  and unexplained check failures increment failure stats and emit
  `sync_verification_failed`. The final post-report flush is verified too.
- The diagnostic smoke adapter now implements the current abstract prompt API
  and separates generation/selection tasks. Previously it could not instantiate.
- Added the missing `tabulate` report dependency to both manifests.
- `.env` variants and rclone configuration/backups are ignored. **No live model
  key, OAuth token, or rclone credential file is committed.**

Added private-import/config planning tests and a shipped-config end-to-end test
using fake model calls plus a temporary local rclone destination. Full validation
results are also summarized in `run_report.md`.

Drive OAuth and a tiny upload/read-back were verified on the owner's Mac, but
authentication must still be imported on the coworker's execution machine. Eight
existing run folders were found under the old nested directory, through
`20260915-004605_full`. No live model calibration was possible: configured public
tunnels did not resolve and this Mac had no local vLLM listeners. The coworker
must supply current endpoints or run/forward the GPU host's ports.

rclone warned that its shared Google OAuth client is being retired during 2026.
The guide links the personal-client setup and notes that tokens must match the
OAuth client used to generate them.

### Final coworker-ready validation

- Full Python suite against this worktree: **225 passed, 7 skipped in 42.24s**.
  Tests use fake model endpoints and temporary local sync destinations, without
  live credentials. Skips reflect unavailable external dataset/shell prerequisites.
- `bash -n` passed for `setup_drive_remote.sh`, `connect_drive.sh`, and
  `sync_run.sh`.
- Ruff passed for all changed Python modules and tests. A whole-repository lint
  run found 24 existing upstream findings in unrelated adapters/tests, which
  were left unchanged.
- The shipped smoke configuration was exercised end-to-end: eight model outputs,
  COT-only metric values, observation-cache reuse, workbook generation, and
  local rclone backup. Failed verification regression tests also passed.
- `git diff --check` passed. The published setup files contain placeholders,
  never the owner's live keys or OAuth configuration.

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

Original feature validation results (before the coworker-ready integration):

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
