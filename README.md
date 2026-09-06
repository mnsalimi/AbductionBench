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
abench run    configs/runs/full.yaml           # 300 samples x 42 datasets x every model

# interrupted? continue exactly where it stopped (per-sample checkpoints):
abench run configs/runs/full.yaml --resume runs/<run-id>
```

`configs/runs/full.yaml` is the single place that decides scope: it picks up
every dataset config automatically, lists the models, and sets the execution
modes. Adding a model is one line there; adding a prompt mode is one entry under
`modes:` and doubles the tasks rather than replacing them.

The dataset table's "Generation / Selection (separate tasks)" datasets each plan
two tasks, and interactive benchmarks plan one episode per item rather than one
prompt, so a full run is more than one task per dataset — `abench run ...
--dry-run` prints the exact plan before anything is sent.

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
* **It checks what actually arrived.** A pass can fail per file — Drive answers
  a burst of small uploads with HTTP 403 — and because rclone creates a
  destination directory before it puts files in it, a failed pass leaves an
  *empty* `datasets/<dataset>/<model>/<mode>/` next to a report that uploaded
  fine. So the final pass does not trust the exit code: it asks the remote what
  it is missing (`rclone check --one-way`), re-sends exactly those files, and
  asks again, up to `verify_attempts` times. What is still missing after that is
  named in the log rather than left to be discovered later.
* **A pass is never killed part-way.** `timeout_s` is rclone's per-transfer
  timeout; the whole pass is unbounded by default (`pass_timeout_s: 0`), because
  a killed pass is exactly what produces those empty directories.
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
    └── datasets/<dataset>/<model>/<mode-slug@version>/
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

It verifies and repairs the same way the engine's own sync does: on exit it asks
the remote what it is missing and re-sends it, and prints what is still absent.
Run it again after a finished run to repair a backup that ended up incomplete.

## Other commands

```bash
abench validate configs/runs/pilot.yaml        # config, modes, adapter imports
abench doctor   configs/runs/pilot.yaml        # probe endpoints incl. the batch route
abench prepare  configs/runs/pilot.yaml        # materialize datasets, no inference
abench run      configs/runs/pilot.yaml        # the evaluation
abench run      configs/runs/pilot.yaml --dry-run          # plan + render prompts only
abench run      configs/runs/pilot.yaml -d ecare -m gpt-oss-120b
abench run      configs/runs/pilot.yaml --resume runs/<run-id>   # continue after a drop
abench report   runs/<run-id>                  # rebuild the workbook from records
abench templates configs/runs/pilot.yaml        # the judge templates
abench run      configs/runs/pilot.yaml -s modes.prompt_modes='[io,cot]'   # compare modes
```

Any setting can be overridden per invocation:

```bash
abench run configs/runs/pilot.yaml \
  -s engine.limits.input_token_budget=8000 \
  -s prompts.bindings.generation=gen_cot_v1
```

## Execution modes

What used to be "which prompt template is bound" is now four independent axes.
Each combination a dataset admits becomes its own task, its own output
directory and its own row, identified by `template_mode`.

| axis | values | who decides |
|---|---|---|
| `prompt_mode` | `io`, `cot`, `self-consistency` | the run config |
| `selection_mode` | `SCS`, `MCS`, `BOV` | the run config, within what the benchmark allows |
| `hypothesis_mode` | `generation`, `selection` | the benchmark's task definition |
| `data_delivery_mode` | `static`, `interactive`, `sequential` | the benchmark — not a choice |

```yaml
modes:
  prompt_modes:     [io, cot]          # crossed with:
  selection_modes:  [SCS, BOV]         # ... for datasets whose task permits each
  hypothesis_modes: [generation, selection]
  self_consistency_n: 5
  self_consistency_temperature: 0.7
```

**`io` / `cot` / `self-consistency`.** `io` asks for the answer and nothing
else; `cot` asks the model to work through the evidence first; `self-consistency`
sends the `cot` prompt *k* times at a non-zero temperature and takes the
plurality answer, reporting `self_consistency_agreement` alongside the score.
These are offered **only for datasets whose metrics are objectively verifiable**
— exact match, label accuracy, set or graph F1, symbolic equivalence. A dataset
scored by text overlap against one reference answer declines them, with a
reason in the run's skipped-modes list, because a reasoning mode's effect
cannot be read off a similarity score.

**`SCS` / `MCS` / `BOV`.** Single-choice asks for exactly one hypothesis;
multi-choice asks for every one that applies; **binary option verification**
presents the hypotheses *one at a time* and asks whether each is the best
explanation, then rebuilds the selected set from the answers that were a direct
yes. A dataset whose task definition requires several selections (AER, DeFAb)
never offers `SCS`; one whose items have exactly one correct answer never offers
`MCS`. BOV is graded by the dataset's own scorer on the reconstructed set, so a
BOV score and an MCS score are comparable, and `bov_yes_rate` reports how choosy
the model was.

**`generation` / `selection`.** Where the dataset table says "Generation /
Selection (separate tasks)", the two run as independent evaluations with
independent scores. Where a benchmark ships the material for both but the table
lists one, the extra mode still runs — and the run log records that it was
introduced and quotes the benchmark's own formulation that justifies it
(`RUN_REPORT.md`, "Additional hypothesis modes"). Where the table says
"Generation **&** Selection", the ampersand is a pipeline, not two tasks, and it
stays one.

**Prompts belong to the dataset.** There is no universal system prompt: the core
does not have one to impose. Each adapter owns its `system_prompt` and the
wording of its task; `adapters/_prompting.py` supplies only the mode
scaffolding, so two datasets in `cot` mode differ in what they ask, never in how
the reasoning is elicited. Interactive benchmarks go further and use the prompts
their own release publishes, read from the cloned repository — VivaBench's
examiner prompt, EvoClinician's actor prompt, Cloud-OpsBench's RCA prompt.

`configs/prompts/` now holds only the **judge** templates: an LLM judge is the
harness prompting a model of its own, not a dataset being evaluated, so its
wording stays swappable configuration.

## Interactive and sequential benchmarks

A benchmark marked `interactive` in the dataset table is executed as a loop, not
squeezed into one prompt. The engine drives episodes turn by turn and batches
**turn *t* of every still-live episode into one batch call**, so a multi-turn
benchmark costs turns × episodes conversations but only a handful of requests.
An episode ends when the adapter's environment says it is done, when the model
commits to an answer, or at the adapter's `max_turns`. `turns_used` is recorded
per sample, because two systems with the same accuracy are not equivalent if one
needed four times as many investigations.

| dataset | what the model does | where the environment comes from |
|---|---|---|
| VivaBench | asks for history, examination, investigations, imaging; commits to a diagnosis | the case's own structured findings; release's action vocabulary and limits |
| DDXPlus | interviews the patient one symptom at a time | the patient record's evidence set, keyed by the release's own question text |
| Med-Inquire | asks the patient, orders tests, submits a diagnosis | the case file's sections; `NOT AVAILABLE` for tests it does not record |
| MedQDx | asks about symptoms, then names the condition | the case's symptom list |
| Cloud-OpsBench | issues `kubectl`-style tool calls, then finalises a root cause | the release's recorded `tool_cache.json` — a real cluster, replayed |
| PhysGym | sets the inputs, reads the output, states the law | the release's own `env_function`, executed |
| CausaLab | intervenes on controllable variables, observes the rest | the released structural model, simulated under intervention |
| BoxingGym | designs experiments, then predicts what the system will do | the release's own simulators, run; scored by its own `evaluate_predictions` |
| SciLab | runs a laboratory whose law is hidden and *not* the textbook one | the release's vendored NewtonBench oracles, run |

Requests are matched to findings **lexically**, not by a second model. Several
of these benchmarks resolve a free-text request with an LLM mapper; doing that
here would put a second model inside the evaluation of the first, so two runs of
the same system could disagree because the examiner did. A deterministic matcher
is reproducible and its misses are visible in the transcript. Each adapter says
so in its own caveats.

`options.delivery: static` runs an interactive benchmark in its single-turn form
as an ablation — useful for asking what the interaction actually buys.

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
├── reports/abductionbench_results.xlsx   Summary · Summary_Long · Metrics · Tasks · Datasets · Skipped · Models · Samples:<ds>
├── reports/summary.csv · metrics_long.csv
├── RUN_REPORT.md                         headline table, failures, skipped datasets and modes, reliability
├── datasets/<dataset>/<model>/<mode-slug@version>/
│   ├── records.jsonl                     one JSON object per sample, appended atomically
│   ├── metrics.json · checkpoint.json · run_documentation.md · raw/
└── run_config.resolved.yaml · engine.log · engine.jsonl · events.jsonl
```

The task directory is named for the **execution mode**, not a template:
`io_SCS_selection_static@1.0` is io prompting, single-choice selection, the
selection task, static delivery, adapter version 1.0.

---

# Field reference

Every metric in a result, every column in a sample-level log, every column in an
evaluation sheet. Each field is defined once; later sections point back rather
than repeat.

## 1. Identity columns

These name *what was run*, and appear in the sample log and in every aggregate
sheet.

| column | meaning |
|---|---|
| `run_id` | the run this row belongs to (`<timestamp>_<run name>`) |
| `dataset_id` | dataset key from `configs/datasets/` |
| `model_id` | model key from `configs/models/` |
| `prompt_mode` | `io`, `cot` or `self-consistency` — see [Execution modes](#execution-modes) |
| `selection_mode` | `SCS`, `MCS`, `BOV`, or `n/a` for a task that is not a selection |
| `data_delivery_mode` | `static`, `interactive` or `sequential`, from the benchmark |
| `task_kind` | what the item asks for, in the adapter's own vocabulary (`generation`, `selection`, `knowledge_completion`, `multi_selection`, `direction_judgment`, ...); `mixed` on a task whose samples span kinds |
| `template_mode` | **the run-version identity**, and nothing else: `prompt_mode\|selection_mode\|task_kind\|data_delivery_mode`, e.g. `cot\|BOV\|selection\|static`. Two rows with the same `template_mode` were produced the same way |

## 2. Sample-level run log

`records.jsonl` per task, and the `Samples:<dataset>` sheets. One row per
request, plus one reduced row per item in modes that ask an item more than once.

**Identity** — every column from [§1](#1-identity-columns), plus:

| column | meaning |
|---|---|
| `sample_id` | stable id derived from the data, not from iteration order; the resume key |
| `group_id` | the evaluation item a request belongs to. Set when one item is asked as several requests: `#bov0`, `#bov1`, ... for BOV, `#sc0`, `#sc1`, ... for self-consistency |
| `reduced` | `True` on the single folded row that carries the item's score; the member rows are kept for inspection but are not counted twice |

**Request and response**

| column | meaning |
|---|---|
| `status` | `ok`, `empty` (server returned no content), `truncated` (stopped at the token budget), `error` (failed after retries), `skipped` (never sent — e.g. over the input-token budget) |
| `parse_ok` | whether a prediction could be extracted at all. Distinguishes *wrong* from *unreadable* |
| `input_tokens_est` | prompt tokens as counted by the configured tokenizer |
| `max_tokens` | the output budget this request was given: `min(32000, context_window - input_tokens_est)`, rounded down onto the batching grid |
| `finish_reason` | the server's own reason for stopping (`stop`, `length`, ...) |
| `prediction` | what the scorer parsed out of the response |
| `reference` | the gold payload the adapter scored against |
| `response` | the model's output, **complete** up to 32,000 characters (Excel's per-cell ceiling); a value that reaches it is marked `...[truncated at cell limit]` |
| `reasoning` | separate reasoning content, where the server returns it |
| `error` | error text for a failed request |
| `batch_id`, `batch_size` | which batch call carried this request, and how many conversations it held |
| `latency_s` | wall-clock for the batch call this request was part of |
| `metric.<name>` | one column per per-sample metric — see [§4](#4-metrics) |

**Interactive rows** additionally carry `metadata.turns_used` (model turns the
episode took) and `response.usage.turns`; the full transcript is in the record's
metadata.

## 3. Evaluation sheets

All share the identity columns of [§1](#1-identity-columns).

**`Summary`** — the headline matrix: datasets down, `model (mode)` across, each
cell the dataset's primary metric.

**`Summary_Long` / `summary.csv`** — one row per task:

| column | meaning |
|---|---|
| `primary_metric` | which metric heads this dataset (the adapter decides) |
| `value` | that metric's value |
| `value_strict` | the same metric with unscored samples counted as zero |
| `coverage`, `n_scored`, `n_planned` | see [§4](#4-reliability-metrics) |
| `failure` | why the task produced nothing, when it did |

**`Metrics` / `metrics_long.csv`** — one row per (task, metric): `metric`,
`value`, `is_primary`, plus the counts.

**`Tasks`** — one row per task, for reading a run's health:
`n_error`, `n_skipped`, `n_reused` (records reused from a checkpoint),
`batch_mode`, `group_size`, `batches_submitted`, `batches_failed`, `bisections`
(batches split to isolate a bad conversation), `prompt_tokens_total`,
`completion_tokens_total`, `duration_s`, `endpoint`, `output_dir`.

**`Datasets`** — each adapter's self-documentation: source, split used,
abductive subset, sampling procedure, decisions, caveats, statistics.

**`Skipped`** — datasets not evaluated, with the reason; and mode combinations a
dataset declined, with the reason.

**`Models`** — endpoint, batch route, verification result per model.

## 4. Metrics

### Reliability metrics (every dataset)

Reported next to every score, so a number produced under endpoint trouble or a
squeezed budget cannot look like a clean result.

| metric | meaning |
|---|---|
| `coverage` | scored samples ÷ planned samples |
| `n_planned`, `n_scored`, `n_error`, `n_skipped` | request counts behind the score |
| `parse_failure_rate` | fraction whose response yielded no prediction |
| `truncation_rate` | fraction that stopped at the token budget. **Not retried** — a truncated answer is a result, and the budget is already the whole remaining context window |
| `empty_response_rate` | fraction that returned no content at all |
| `output_budget_clamped_rate` | fraction whose budget was limited by the context window rather than the 32,000-token ceiling — i.e. where the prompt crowded out the answer |
| `batch_latency_s_mean`, `completion_tokens_mean` | cost and length, per sample |
| `<primary>_strict` | the primary metric with unscored samples counted as zero |

### Mode metrics

| metric | meaning |
|---|---|
| `self_consistency_agreement` | fraction of the *k* votes that agreed with the winning answer. 1.0 is unanimous; 1/k means every sample differed |
| `bov_yes_rate` | fraction of hypotheses a BOV run said yes to. A model that says yes to everything scores well on recall alone, so this sits next to the score rather than inside it |
| `turns_used` | model turns an interactive episode took |
| `repeats` | how many times each question was asked (`options.repeats`) |

### Answer-shape metrics

The general shapes almost every dataset reduces to. A dataset's own metric names
are these with a dataset-specific prefix (`diagnosis_match`, `root_cause_match`,
`hypothesis_rouge_l`, `flip_token_f1`, ...), and mean the same thing about that
dataset's answer.

| metric | meaning |
|---|---|
| `accuracy` | fraction of selection items where the chosen label was the gold label |
| `exact_match` | normalized string equality with the gold answer |
| `match` | equality *or* containment of an accepted gold form — the lenient view |
| `token_f1` | bag-of-tokens F1 against the gold answer |
| `rouge_l` | longest-common-subsequence F-measure against the gold answer |
| `set_f1`, `set_precision`, `set_recall` | for answers that are a *set* (multi-selection, causal edge sets) |
| `exact_set_match` | the whole set exactly right |
| `symbolic_match` | SymPy proves the answer equivalent to the reference expression; `symbolic_match_decidable` restricts to the items SymPy could compare, and `symbolic_undecidable` reports the rest |
| `<metric>_judged` | the same judgement made by an LLM judge, only when `engine.judge.enabled` |

### Dataset-specific metrics

Where a single accuracy number would mislead, the adapter reports what the task
actually measures. Each is defined in that dataset's entry in
[`docs/datasets.md`](docs/datasets.md), which is generated from the adapters
themselves; the ones worth knowing about here:

| dataset | metric | why |
|---|---|---|
| `hypospace` | `distinct_valid_rate` | observations admit dozens of valid graphs; covering the space is the point |
| `gear` | `undetermined_recall`, `overcaution_rate` | separates admitting underdetermined evidence from over-hedging |
| `causalab` | `edge_f1` + precision/recall | listing every possible edge must not score well |
| `true_detective` | `human_agreement_spearman`, `human_solve_rate` | do models find the same puzzles hard that people do |
| `medr_bench` | `rare_disease_gap` | the benchmark's headline claim is about rare disease |
| `commonwhy` | `popularity_gap` | head vs long-tail entities |
| `medqdx` | `information_sensitivity` | accuracy at 100% vs 50% of the symptom picture |
| `abd` | `predicate_compliance` | respecting the hypothesis space is its own competence |
| `house_md` | `diagnosis_in_differential` | separates recall of the disease from committing to it |
| `medcasereasoning` | `reasoning_recall`, `reasoning_overlap` | how much of the clinician's reasoning the answer recovers |
| `open_problems_2024` | `direction_accuracy`, `brier_score`, `leakage_rate`, `direction_answer_stability` | a calibrated verdict, and whether the model had simply read the answer |

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
42 evaluable, 6 declared unavailable with a reason, 1 (`researchbench`) waiting
on a Hugging Face gate** — the 48 from the original table plus
`open_problems_2024`, built here.

Nine of the interactive benchmarks run their real environment; see
[Interactive and sequential benchmarks](#interactive-and-sequential-benchmarks).
Interactivity is no longer a reason to skip anything. What is still skipped is
skipped for reasons that have nothing to do with it: **CausalGame**'s
environment is not distributed (its harness plays against the authors' server),
**NIKA** records no telemetry to replay and needs privileged networking to
generate any, **BioVerge** ships its items only inside an 11.9 GB corpus
archive, and **DiReCT**, **DiscoveryBench** and **RLF-KG** are blocked on
credentialed access, withheld gold, and an unreachable download respectively.

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
* **Output budget** is not guessed per item: every request gets
  `min(32000, context_window - input_tokens_est)`, so a truncated answer means
  the model used the whole window rather than an estimate that was too small.
* **300 samples** per dataset by a seeded shuffle; oversize prompts are replaced
  from the same shuffle rather than dropped, and shortfalls (e.g. HypoGen's 50
  test items) are reported.
* **Skipped, never guessed**: a dataset whose data is unobtainable or whose gold
  is withheld is declared with `UnavailableAdapter` and appears in every run's
  `Skipped` sheet with its reason. "Interactive" is no longer a reason to skip:
  a benchmark whose environment ships with it is executed.

Three datasets required constructing items from released material rather than
using a shipped item file; both say so explicitly and are excluded from
comparison with published numbers: **ProofWriter** (the official abduction files
are no longer downloadable, so abduction items are rebuilt from the released
proofs) and **SynPAT** (items assembled from the release's own replacement files
plus true-system data). **HypoSpace** runs the release's own seeded generator.

### ResearchBench: one manual step

`researchbench` is fully implemented — both of its tasks, generation from
`generation/generation.jsonl` and selection from `ranking/ranking.jsonl` — but
its Hugging Face repository is **gated**. `gated: auto` means access is granted
the moment an account accepts the dataset's terms, and a token alone is not
enough: repository *metadata* reads for anyone, while every file returns HTTP
403 until the account behind the token has accepted.

```bash
# 1. sign in as the account whose token you will use, open
#    https://huggingface.co/datasets/ankilok/ResearchBench
#    and accept the terms (name, affiliation, intended use, two checkboxes)
# 2. then simply:
export HF_TOKEN=hf_...
abench run configs/runs/full.yaml -d researchbench
```

Until then the adapter reports itself skipped with that URL and the account name
in the reason, rather than failing obscurely. Nothing else needs changing.

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

Where a dataset's task makes a single accuracy number misleading, its adapter
reports what the task actually measures. Those metrics are defined in
[Field reference §4](#4-metrics) and, per dataset, in
[`docs/datasets.md`](docs/datasets.md).

