# AbductionBench

A configurable evaluation framework for **abductive reasoning** — inference to
the most plausible explanation — across a curated suite of benchmarks.

Everything that defines a run is configuration: endpoints, per-model batch group
sizes, retry policy, token budgets, sampling, which datasets, how many samples,
and **the prompt templates themselves**. The engine knows nothing about any
particular dataset.

```
configs/          engine defaults · models · prompt templates · datasets · runs
src/abductionbench/
  core/           the dataset-agnostic engine (Phase 1)
  adapters/       one adapter per benchmark  (Phase 2)
docs/             architecture · adapter-authoring contract
tools/            diagnostics (live smoke test)
runs/             one directory per run: records, metrics, logs, reports
data/             materialized datasets and caches
```

## Conceptual frame

Abduction is *the inference of the most plausible explanation for an
observation*. Following the two-stage decomposition the suite is organized
around, every task falls into:

* **Stage 1 — hypothesis generation**: produce candidate explanations
  (free-form *explanation generation*, or *knowledge completion* that adds the
  missing facts/rules making an observation derivable). → `task_kind: generation`
* **Stage 2 — hypothesis selection**: evaluate candidates on explanatory
  virtues and pick the best (*scoring and selection*), or judge a single
  hypothesis's plausibility (*single-hypothesis evaluation*).
  → `task_kind: selection`

Abductive conclusions are **defeasible**, **non-monotonic** and **ampliative**;
the shipped prompt templates are written to elicit exactly that (commit to the
best explanation given current evidence, rather than prove a theorem), and the
metrics report parse failures and coverage separately so a tentative-but-correct
answer is never confused with an unparseable one.

## Install

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[adapters,dev]"
cp .env.example .env    # then fill in ABENCH_API_KEY and the URLs
```

## Run the whole suite

```bash
set -a; . ./.env; set +a                       # or: export ABENCH_API_KEY=...

abench doctor configs/runs/full.yaml           # 30 s: endpoint + batch route reachable
abench run    configs/runs/full.yaml           # 300 samples x 40 datasets x every model

# interrupted? continue exactly where it stopped (per-sample checkpoints):
abench run configs/runs/full.yaml --resume runs/<run-id>
```

`configs/runs/full.yaml` is the single place that decides scope: it picks up
every dataset config automatically and lists the models. Adding a model is one
line there. One model over the whole suite is 9,882 prompts (~1,270 native batch
calls at group size 8) and takes roughly 2-3 h on an idle GPU; selection
datasets cost ~0.02 s/sample, long-form physics ~4 s/sample.

## Backing up a run (incremental, off the hot path)

A long run on a box whose filesystem is not persistent should not keep its only
copy locally. `engine.sync` mirrors the run directory to any rclone destination
**while the run proceeds**:

```yaml
engine:
  sync:
    enabled: true
    remote_path: gdrive:AbductionBench   # or s3:bucket/prefix, or a local path
    per_run_subdir: true                 # -> <remote_path>/<run-id>/
    interval_s: 60
    exclude: []                          # e.g. ["raw/**"] on a slow uplink
```

How it behaves, and why:

* **It cannot slow inference down.** A daemon thread does nothing but launch
  `rclone` as a subprocess, so all network I/O happens in a separate process and
  the asyncio loop driving batch calls is never blocked.
* **It cannot break a run.** Every failure is caught, counted and retried on the
  next tick; a bad credential or a dead link degrades the run to "not backed
  up", never to "crashed", and `RUN_REPORT.md` says which happened.
* **It is incremental.** `rclone copy --update` sends only the files that
  changed, so a tick uploads the few `records.jsonl` and log files that grew.
* **It never deletes remote data** (`copy`, not `sync`), so a fresh run
  directory cannot wipe results already backed up.
* **It uploads a snapshot, not the live directory.** Records and logs are
  appended to continuously; uploading them live makes rclone size/hash a file,
  send it, find the remote copy no longer matches, declare the transfer corrupt
  and *delete it remotely*. Each pass rsyncs locally first (a moment) and
  uploads that, so every transfer is stable and verifiable.
* **It respects the destination's rate limits.** A full run writes ~1,200 files,
  1,096 of them per-batch debug payloads under `raw/`; sending those exhausts
  Google Drive's per-minute quota for rclone's shared OAuth client (HTTP 403)
  and starves the files that matter. `raw/` is excluded by default and API calls
  are throttled (`exclude: []` backs up everything).
* A final pass runs after the workbook and run documentation are written.

The Drive layout mirrors the local one, one folder per run:

```
AbductionBench/                        <- the folder you shared
└── 20260903-190409_full/              <- run id: UTC timestamp + run name
    ├── RUN_REPORT.md                  <- headline table, skipped datasets, reliability
    ├── run_config.resolved.yaml       <- exactly what was run (API keys redacted)
    ├── engine.log · engine.jsonl · events.jsonl
    ├── reports/
    │   ├── abductionbench_results.xlsx
    │   └── summary.csv · metrics_long.csv
    └── datasets/<dataset>/<model>/<template@version>/
        ├── records.jsonl              <- one JSON object per sample
        ├── metrics.json · checkpoint.json
        └── run_documentation.md
```

### One-time Google Drive setup

```bash
bash tools/connect_drive.sh
```

That is the whole thing. It prints an authorization link; **Ctrl+click it** --
if you are attached over VS Code Remote SSH, VS Code forwards the port
automatically and the link opens in your own browser, so nothing needs to be
installed locally. Approve the account that owns the target folder and the
script writes the remote, verifies read *and* write access, and attaches the
backup to a run that is already in progress.

The remote is pinned with `root_folder_id`, so nothing can be written anywhere
else in the Drive. If the link does not become clickable: VS Code -> **PORTS**
tab -> Forward a Port -> `53682`, then open the link.

Two alternatives, if you prefer them:

* `bash tools/setup_drive_remote.sh '<token-json>'` -- when you already have a
  token from running `rclone authorize "drive"` on another machine.
* A service account, for a **Shared** Drive. `setup_drive_remote.sh`'s header
  explains why that does not work for a folder in a personal My Drive (the
  service account would own the files and has no storage quota).

### A run that is already in progress

A running process cannot grow the feature mid-flight, so attach the sidecar --
same job, same rclone flags, its own process:

```bash
nohup bash tools/sync_run.sh runs/<run-id> > /tmp/abench_sync.log 2>&1 &
```

## Other commands

```bash
abench validate configs/runs/pilot.yaml        # config, templates, adapter imports
abench doctor   configs/runs/pilot.yaml        # probe endpoints incl. the batch route
abench prepare  configs/runs/pilot.yaml        # materialize datasets, no inference
abench run      configs/runs/pilot.yaml        # the evaluation
abench run      configs/runs/pilot.yaml --dry-run          # plan + render prompts only
abench run      configs/runs/pilot.yaml -d ecare -m gpt-oss-120b
abench run      configs/runs/pilot.yaml --resume runs/<run-id>   # continue after a drop
abench report   runs/<run-id>                  # rebuild the workbook from records
abench templates configs/runs/pilot.yaml --show gen_cot_v1
```

Any setting can be overridden per invocation:

```bash
abench run configs/runs/pilot.yaml \
  -s engine.limits.input_token_budget=8000 \
  -s prompts.bindings.generation=gen_cot_v1
```

## Swapping prompt templates

Templates are versioned YAML in `configs/prompts/` (see its README). To compare
several against the same data in one run:

```yaml
prompts:
  bindings:            {generation: gen_freeform_v1, selection: sel_mcq_letter_v1}
  template_variants:
    cot:               {generation: gen_cot_v1, selection: sel_mcq_cot_v1}
  dataset_overrides:
    proof_writer:      {generation: gen_structured_v1}
```

Each variant becomes its own task, its own output directory
(`datasets/<ds>/<model>/<template@version>/`) and its own row in the result
grid. Because a stored record's resume key includes the prompt fingerprint,
results from two different templates can never be silently mixed.

## Models and batching

Each model is one file in `configs/models/`, with **its own batch group size**:

```yaml
model:
  id: gemma-4-e4b
  model_name: google/gemma-4-E4B-it
  endpoint:
    base_url: ${env:ABENCH_GATEWAY_URL}       # discovery + single calls
    api_key:  ${env:ABENCH_API_KEY}
    batch:
      base_url: ${env:ABENCH_BATCH_URL}       # the native batch route lives here
      path: /gemma-4-e4b/v1/chat/completions/batch
      group_size: 8                           # 8 samples per batch call, server-side
  limits:
    context_window: 16384                     # this model's MAX_MODEL_LEN
```

`configs/models/` ships one file per model and per access path — gateway
pass-through (`gemma-4-e4b.yaml`), the model's own tunnel
(`gemma-4-e4b-direct.yaml`), and tunnel-free local/SSH-forwarded
(`gemma-4-e4b-local.yaml`) — plus the same three for `gpt-oss-120b`, which stay
valid for whenever that model is brought back up. Add every model you want in
one run to the run config's `models:` list; each keeps its own group size,
sampling and limits.

"Batch of 8" means **one HTTP request carrying 8 conversations** to vLLM's
`POST /v1/chat/completions/batch`, executed as a batch server-side — not 8
concurrent single requests. Different models in the same run can use different
group sizes. See `docs/architecture.md` for the endpoint semantics this relies
on and how failures are handled.

## Outputs

```
runs/<run-id>/
├── reports/abductionbench_results.xlsx   Summary · Metrics · Tasks · Datasets · Skipped · Models · per-dataset samples
├── reports/summary.csv · metrics_long.csv
├── RUN_REPORT.md                         headline table, failures, skipped datasets, reliability
├── datasets/<dataset>/<model>/<template@version>/
│   ├── records.jsonl                     one JSON object per sample, appended atomically
│   ├── metrics.json · checkpoint.json · run_documentation.md · raw/
└── run_config.resolved.yaml · engine.log · engine.jsonl · events.jsonl
```

Reliability is reported next to every score: `coverage`,
`parse_failure_rate`, `truncation_rate`, `empty_response_rate`,
`output_budget_clamped_rate`, and `<primary>_strict` (the primary metric with
unscored samples counted as zero), so a number produced under endpoint trouble,
a squeezed token budget or a model that never answered cannot look like a clean
result.

## Adding a dataset

See `docs/adapter_contract.md`. In short: implement `prepare`,
`build_samples`, `score`, `aggregate`, `documentation`; point a dataset config
at it with `impl: "module:Class"`. A dataset that cannot be obtained, parsed, or
whose abductive subset is unclear must raise `SkippedDataset("reason")` — it is
then reported as skipped rather than guessed at.

## Development

```bash
python -m pytest -q            # 73 tests: config, prompts, batching, metrics,
                               # checkpointing, judge, adapter scorers, and
                               # end-to-end engine runs against a fake vLLM
                               # server over real HTTP
abench run configs/runs/smoke_local.yaml    # engine smoke test, ~30 s
abench run configs/runs/pilot_smoke.yaml \
  -d ecare,gear,hypospace,aer,abductionrules,medcasereasoning   # live, real adapters
python tools/preview.py configs/runs/pilot.yaml <dataset_id>    # inspect one adapter
python tools/dataset_catalogue.py                               # regenerate docs/datasets.md
```

---

## Dataset coverage (Phase 2)

`docs/datasets.md` is the catalogue — **generated from the adapters themselves**
(`python tools/dataset_catalogue.py`), so its numbers, splits and stated
decisions cannot drift from the code. Current state: **49 datasets configured,
40 evaluable, 9 reported as skipped with a reason** — the 48 from the original
table, plus `open_problems_2024`, built here (see below).

Policies applied uniformly, and recorded per dataset:

* **Split**: test → validation → train, and the fallback is always reported
  (e.g. e-CARE's test labels are withheld upstream, so dev is used).
* **Abductive subset only**: `ask-for=cause` in e-CARE, `question=cause` in
  XCOPA, murder mysteries in MuSR, the causal family in SciR, the diagnosis
  collection in MedR-Bench, ACRE in GEAR, and so on — with the excluded portion
  and the reason named.
* **Answer leakage**: fields that contain the answer (worked solutions,
  reasoning traces, inspirations, model-generated columns) are withheld from
  prompts and listed in each adapter's decisions.
* **Per-sample `max_tokens`** from that item's complexity — proof depth, node
  count, reasoning-step count, or output shape — never one flat value.
* **300 samples** per dataset by a seeded shuffle; oversize prompts are replaced
  from the same shuffle rather than dropped, and shortfalls (e.g. HypoGen's 50
  test items) are reported.
* **Skipped, never guessed**: a dataset whose data is unobtainable, whose gold is
  withheld, or that only exists as an interactive environment is declared with
  `UnavailableAdapter` and appears in every run's `Skipped` sheet with its
  reason.

Three datasets required constructing items from released material rather than
using a shipped item file; both say so explicitly and are excluded from
comparison with published numbers: **ProofWriter** (the official abduction files
are no longer downloadable, so abduction items are rebuilt from the released
proofs) and **SynPAT** (items assembled from the release's own replacement files
plus true-system data). **HypoSpace** runs the release's own seeded generator.

### `open_problems_2024` — the one dataset built here

Not from the original table: a scientific-discovery set built in this repository
from two independent surveys of research problems that were **open at a 2023
model cutoff and resolved in 2024 or later** (both surveys are committed under
`assets/open_problems_2024/sources/`, and
`tools/build_open_problems_dataset.py` rebuilds `problems.json` from them, so
every field is traceable to a source line). 16 problems, 15 evaluable, 1 held
out.

It is deliberately split into modes, because only some of them are abduction:

| mode | task | abductive? |
|---|---|---|
| `direction` | judge which way an open conjecture resolved, with a confidence | **yes** — Stage 2, single-hypothesis evaluation |
| `strategy` | say what a solution would have to use, before seeing one | **yes** — Stage 1, knowledge completion |
| `resolution` | actually resolve the problem | no — deduction; excluded from `abduction_score` |
| `leakage_probe` | ask what the model knows about the problem's status | not scored as reasoning; it is the contamination control |

The scores are only meaningful if the model has not read the answer, so the
probe is part of the dataset rather than an afterthought: it reports
`leakage_rate` per item, and a high value invalidates that item's reasoning
score. Measured on gemma-4-E4B (15 items, `subtask=leakage_probe`):
`leakage_rate = 0.00` — not one item where the model both asserts the problem is
resolved and names the solver or the year.

**Read `direction_answer_stability` before reading any score.** 15 items is
small enough that server-side nondeterminism dominates the number. What is
actually going on, measured rather than assumed:

* Under *identical* batching the model is bit-reproducible — two consecutive
  runs of this dataset alone agreed on **45/45** direction predictions and gave
  the same `abduction_score` to four decimals.
* Change the batching and verdicts flip. Within one run, 2 of 15 questions got
  different answers from their own three repeats. Between a solo run and one
  sharing the endpoint with three other datasets, **5 of 15** flipped, moving
  accuracy from 0.33 to 0.60.

The cause is vLLM's batched inference not being numerically batch-invariant: a
prompt's logits depend on what else is in its batch, so the noise is *worst* in
a full-suite run, where 39 other datasets share the endpoint. That is not
fixable from here, so it is measured instead. `options.repeats` (3 by default)
asks each question k times as k distinct samples; the score is the mean over all
observations, `direction_answer_stability` is the fraction of questions whose
repeats all agreed, and `strategy_score_spread` is the mean within-question
range of `key_ingredient_recall` — a strategy answer is never byte-identical
twice (1/39 were), so its stability is the spread of its score, not of its text.

gemma-4-E4B, alone, 3 repeats: direction accuracy **0.378** over 45
observations at stability **0.867**, `key_ingredient_recall` 0.289 at spread
0.109, Brier 0.278, `overconfident_wrong_rate` 0.179. Treat a difference between
two runs as noise unless it exceeds that spread.

```bash
abench run configs/runs/pilot_smoke.yaml -d open_problems_2024                 # direction + strategy
abench run configs/runs/pilot_smoke.yaml -d open_problems_2024 \
  -s dataset_force.options.subtask=leakage_probe                               # contamination control
abench run configs/runs/pilot_smoke.yaml -d open_problems_2024 \
  -s dataset_force.options.model_cutoff=2025-06-01                             # only post-cutoff items
```

### Metrics beyond accuracy

Where a dataset's task makes a single accuracy number misleading, the adapter
reports what the task actually measures:

| dataset | metric | why |
|---|---|---|
| `hypospace` | `distinct_valid_rate` | observations admit dozens of valid graphs; coverage of the hypothesis space is the point |
| `gear` | `undetermined_recall`, `overcaution_rate` | separates admitting underdetermined evidence from over-hedging |
| `physgym`, `synpat` | `symbolic_match` (SymPy) | physical laws have many algebraically equal forms |
| `causalab` | `edge_f1` + precision/recall | listing every possible edge must not score well |
| `aer`, `defab` | `set_f1`, `exact_set_match` | the gold answer is a set |
| `true_detective` | `human_agreement_spearman` | do models find the same puzzles hard that people do |
| `medr_bench` | `rare_disease_gap` | the benchmark's headline claim is about rare disease |
| `commonwhy` | `popularity_gap` | head vs long-tail entities |
| `medqdx` | `information_sensitivity` | accuracy at 100% vs 50% of the symptom picture |
| `abd` | `predicate_compliance` | respecting the hypothesis space is its own competence |
| `house_md` | `diagnosis_in_differential` | separates recall of the disease from committing to it |
