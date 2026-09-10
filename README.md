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
abench run    configs/runs/full.yaml           # 200 records x 42 datasets x every model

# interrupted? continue it by name -- the folder name, the same one on Drive:
abench run configs/runs/full.yaml --resume 20260907-101530_full
```

`configs/runs/full.yaml` is the single place that decides scope: it picks up
every dataset config automatically, lists the models, and sets the execution
modes. Adding a model is one line there; adding a prompt mode is one entry under
`modes:` and doubles the tasks rather than replacing them.

The dataset table's "Generation / Selection (separate tasks)" datasets each plan
two tasks, and interactive benchmarks plan one episode per item rather than one
prompt, so a full run is more than one task per dataset — `abench run ...
--dry-run` prints the exact plan before anything is sent.

## Continuing a run

```bash
abench run configs/runs/full.yaml --resume 20260907-101530_full
```

The argument is the **run id** — the name of the directory under `runs/`, which
is also the name of the folder on Drive, which is also what the run printed when
it started. A path works too, for a run kept somewhere else.

What continuing does:

* **Already-answered samples are reused, not paid for again.** Every record's
  answer is checkpointed with a fingerprint of the exact request that produced
  it (model, mode, messages, sampling), so a sample is only reused when the same
  question would be asked again.
* **Anything new in the config is run.** Add a dataset, a model, a prompt mode,
  raise `repeats` — the new work has no checkpoint and runs in full while
  everything else is skipped. This is the normal way to grow a run: launch a
  small one, then continue it wider.
* **Changing a config does not erase the record of what already ran.** The
  original `run_config.resolved.yaml` stays; each later pass that differs is
  written next to it as `run_config.resolved.2.yaml` and so on, so the records on
  disk are always explained by a config that is still there.
* **A missing directory is restored from the backup first.** If `runs/<id>` is
  not here but the run was mirrored to `engine.sync.remote_path`, it is pulled
  back before continuing — which is the point of the backup: this box's
  filesystem is not guaranteed to survive, and continuing needs the records.
  Nothing is written to the remote and nothing local is deleted.
* An unknown id lists the runs that *are* available rather than starting a new
  run under that name.

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

Orthogonal to all four is **`repeats`**: how many times each record is asked.

```yaml
modes:
  prompt_modes:     [io, cot]          # crossed with:
  selection_modes:  [SCS, BOV]         # ... for datasets whose task permits each
  hypothesis_modes: [generation, selection]
  self_consistency_n: 5
  self_consistency_temperature: 0.7
  repeats: 3                           # each record asked 3 times
  repeats_by_delivery: {interactive: 1, sequential: 1}
  repeat_temperature: 0.7
```

**`repeats`.** Each record is put to the model `repeats` times as separate API
calls, and **each answer is scored on its own**. A dataset's score becomes the
mean over `repeats × records` observations rather than over records, and two
metrics report whether that mean is worth anything:

| metric | meaning |
|---|---|
| `repeat_agreement` | fraction of records whose repeats all gave the same answer. 1.0 means the model is deterministic here; a low value means the mean is an average of disagreement |
| `<primary>_repeat_std` | typical spread of the primary metric within one record |

Repeats are drawn at `repeat_temperature` with the model's fixed seed dropped —
asking the same question five times at temperature 0 measures the server's
determinism, not the model. `n_scored` is five times `records`, and the sample
sheet has one row per call, with `record_id` and `repeat_index` to group them.

**Self-consistency comes out of these same calls, free.** A vote is a plurality
over *k* samples of one question, and the repeats *are* *k* samples of one
question — so the voted answer is read off them rather than bought again. Every
metric gains a `self_consistency_` counterpart:

```
accuracy                      0.60   ← what one sample is worth
self_consistency_accuracy     1.00   ← what a vote over five is worth
repeat_agreement              0.00   ← they never all agreed
accuracy_repeat_std           0.49   ← and this is how much they moved
```

So `prompt_modes: [io, cot]` with `repeats: 3` gives you four numbers per
dataset — io, cot, and the voted version of each — for two tasks' worth of
calls. There is a standalone `self-consistency` prompt mode, but with repeats on
it buys nothing the io and cot tasks do not already report.

**Only where a vote means something.** A plurality needs answers that can
coincide: a label, a set, an equation. The nine datasets graded by an LLM judge
or by overlap with a reference (`crosstrace`, `hypogen`, `uncommonsense`,
`moose_chem2`, `hypoarena`, `hypobench`, `matter_to_mechanism`, `commonwhy`,
`researchbench`) never produce the same free-text hypothesis twice, so every
sample would be its own plurality of one and the "voted" score would be whichever
sample came first. Those datasets report the spread of their repeats and no
`self_consistency_` metrics. It is the same rule that keeps `cot` off them:
a reasoning mode is only readable where correctness is decidable.

**Cost.** Repeats multiply a run by `repeats`, and an interactive record is
already ten to twenty calls, so it multiplies five times a lot rather than five
times a little. Measured on this suite: repeating the interactive
datasets is what costs: at 200 records, `repeats: 3` on the static datasets plus
1 on the interactive ones is ~37,000 calls, while repeating the interactive ones
too would add tens of thousands of *episodes* — tens of hours against about an
hour for every static dataset combined. Hence
`repeats_by_delivery: {interactive: 1}` by default. Raise it deliberately.

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

## Prompts: one structure, so the data is what differs

**Static and sequential datasets get prompts written for this suite**, not the
ones their papers published. The reason is comparability: if each dataset
inherited its authors' wording, a score gap could just as easily be a gap in how
firmly the instruction was phrased. So the structure is fixed once in
`adapters/_prompting.py` and only the dataset's own nouns vary.

**Interactive datasets are the exception** and keep the prompts their own
benchmark publishes -- an environment's action grammar is part of the benchmark,
not a style choice. That is why `ddxplus`, `med_inquire`, `medqdx` and
`vivabench` still label their options `A`, `B`, `C`.

Every static/sequential prompt is assembled the same way:

```
[system]   what this dataset's task is, in its own terms

[user]     Observation:  <the evidence>
           Question:     <the dataset's question, if it asks one>
           Answer options:                     <-- numbered, never lettered
           1. ...
           2. ...
           Answer directly. Do not explain your reasoning.     <-- io
             (or: Work through the evidence step by step...)   <-- cot
           Requirements:                        <-- generation tasks
           - write exactly one sentence
           - do not restate the observation
           Select exactly one hypothesis.       <-- selection tasks
           Answer with only one of: 1, 2 or 3, on the last line, as:
           Answer: 1
```

Three pieces are dataset-owned and declared as **data**, so the same constraint
reads identically everywhere it applies (`tools/declare_answer_shapes.py` is the
one-shot table that set all 34):

| attribute | what it says |
|---|---|
| `answer_format` | what a well-formed answer is (`"one short sentence"`), rendered into the closing line |
| `answer_constraints` | what the answer must and must not do, one clause each, rendered as `Requirements:` |
| `options_heading` | what the candidate list is called (`"Answer options:"`, `"Candidate diagnoses:"`) |

**Options are numbered.** One helper (`_common.choice_labels`) produces both the
labels shown and the labels the gold refers to, so they cannot drift apart --
which is what a per-adapter `LABELS = ["A", "B"]` constant risked.

**The `Answer:` marker is load-bearing, not decoration.** A numbered label is
much easier to confuse with a number that appears in reasoning than a letter
was, and the marker is what lets the parser take the label the model
*submitted* rather than the last digit it happened to write. Measured across
xcopa, ecare, musr, true_detective and aer in both io and cot:
`parse_failure_rate` 0.0 on nine of ten tasks and 0.1 on the tenth.

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

## Making a run faster without touching quality

Measured on this suite (Qwen3.5-2B, 200 records, io + cot, repeats 3):

| | |
|---|---|
| output tokens | 37.0 M |
| input tokens | 8.7 M |
| | **4.2x more decode than prefill** |
| calls whose prompt is a byte-identical re-send | **80%** (the repeats) |

So the run is decode-bound, and four fifths of its prefill work is duplicated.
That points at server flags and nothing about the prompts:

```bash
vllm serve Qwen/Qwen3.5-2B --host 127.0.0.1 --port 18001 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --max-num-seqs 64 \             # was 8: decode 64 sequences per step, not 8
  --enable-prefix-caching \       # the 80% of duplicate prompts prefill for free
  --gpu-memory-utilization 0.35 \ # was 0.12: KV cache is what buys concurrency
  --api-key "$ABENCH_API_KEY"      # CUDA graphs on (no --enforce-eager)

export ABENCH_GROUP_SIZE=64        # a batch should fill the scheduler exactly
```

`ABENCH_GROUP_SIZE` must track `--max-num-seqs`: a larger batch only queues
inside vLLM, a smaller one leaves the GPU idle.  On this box the flags above are
set in `/workspace/vllm_serving/qwen3.5-2b/.env` (`MAX_NUM_SEQS`,
`ENABLE_PREFIX_CACHING`, `GPU_MEMORY_UTILIZATION`, `ENFORCE_EAGER=0`); 0.35
utilisation is 2.1 M KV tokens, exactly 64 x 32,768, so the cache matches the
scheduler instead of over-reserving the GPU.

**Measured, same task before and after** (`aer`, cot, 24 records, one repeat):

| | wall clock | set_f1 | scored |
|---|---|---|---|
| `--max-num-seqs 8`, no prefix cache, util 0.12 | 275.5 s | 0.6510 | 24/24 |
| `--max-num-seqs 64`, prefix cache, util 0.35 | **157.1 s** | 0.6694 | 24/24 |

**1.75x**, and the freed headroom is larger than the number suggests: with the
same server, tripling the task to 72 prompts (repeats 3) cost 161.2 s -- 3x the
work for 3% more wall clock.

**Two levers that were measured and rejected.** Raising
`engine.concurrency.max_parallel_tasks` from 2 to 6 on six tasks made the run
*slower* (532 s -> 621 s): the GPU was already the constraint at ~48 concurrent
sequences, so extra tasks only shared decode slots. Raising the per-model
`max_parallel_batches` has the same ceiling. The client-side concurrency
defaults are therefore left alone.

**What this does and does not change.** Prefix caching is exact KV reuse -- the
same arithmetic, so the same distribution. Raising `--max-num-seqs` changes
which sequences share a decode step, and batched inference is not numerically
batch-invariant, so individual answers can come out differently; nothing is
degraded, and at the temperature repeats are drawn at the answers already vary
between runs. Neither touches a prompt, a token budget, or a stop condition.

**What is *not* free**, and is therefore not done here: capping `max_tokens`
below the context window. It is the largest single speedup available -- 93% of
BoxingGym's answers and 21% of `abd`'s CoT answers run to the 32,000-token cap
and are cut off -- but of the answers that *do* finish on their own, 9% exceed
8,192 tokens and 16% exceed 2,048. A cap would truncate real answers, so it is a
trade to make deliberately, with those numbers in view, not a free win.

## The judge is a different model

Datasets with no answer key are graded by an LLM judge, not by overlap with a
reference. Two models are served on the one GPU:

| | model | port | window | role |
|---|---|---|---|---|
| under test | `Qwen/Qwen3.5-2B` | 18001 | 65,536 | answers |
| judge | `openai/gpt-oss-120b` | 18004 | 65,536 | grades |

**Why two.** Grading a model's answers with that same model scores its own
reasoning. The judge is therefore a 117B-parameter model and the model under
test is a 2B one, and they are separate servers.

**Why the judge is in `models:`.** The run needs a client to reach it. It is
marked `judge_only: true` in `configs/models/gpt-oss-120b-local.yaml`, so
`evaluated_models()` leaves it out of task planning -- otherwise it would double
the run and report a column nobody asked for. `--models` also never filters it
out: that flag narrows what is *measured*, and dropping the judge with it would
silently switch off the judged metric of every dataset that has no answer key.

**Sharing one 95.6 GiB card.** gpt-oss-120b's weights are 60.8 GiB in MXFP4, so
the split is deliberate and the start order matters -- restart Qwen first so it
shrinks, then start the judge, which needs its whole budget free to pass vLLM's
check:

```
qwen3.5-2b     GPU_MEMORY_UTILIZATION=0.20  ->  19 GiB  (~10 GiB KV = 1.04M tokens)
gpt-oss-120b   GPU_MEMORY_UTILIZATION=0.78  ->  75 GiB  (~11 GiB KV =  158k tokens)
```

`MAX_NUM_BATCHED_TOKENS=8192` on the judge is load-bearing: without it vLLM
profiles a forward pass at the full 65,536 tokens, and that activation peak
consumed the entire budget -- startup died with *"No available memory for the
cache blocks"* at 0.78 utilisation with 60.8 GiB of weights.

Both `.env` files under `/workspace/vllm_serving/` carry these numbers and the
reasoning; timestamped backups sit beside them.

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
├── reports/abductionbench_results.xlsx   Summary · Summary_Long · Metrics · Tasks · Datasets · Skipped · Skipped_modes · Introduced_modes · Models · S_<ds>
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

### Reports appear as the run goes, not at the end

A full run is many hours of API calls, so the sheets are **rewritten every time
a dataset finishes**, each pass covering every task completed so far. The
workbook therefore grows a dataset at a time and exists from the first one
onward; the write at the end of the run is simply the last of them. It is also
why an interrupted run still has readable reports.

Each pass writes the complete set -- `Summary`, `Metrics`, `Tasks`, the
per-sample `S_<ds>` sheets, the CSVs and `RUN_REPORT.md` -- to a temp file and
renames it into place, so the backup's upload pass, or a person opening the
workbook, always sees a whole file rather than half of one. Building it costs
about 9 s at 22,000 records and runs in a worker thread, off the event loop that
is driving the batches.

Set `engine.reporting.interim_after_each_dataset: false` to go back to writing
the reports once, after the last dataset.

`abench report <run-dir>` still rebuilds every report from `records.jsonl`
alone, which is the way to regenerate sheets for a run that was killed outright.

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

`records.jsonl` per task, and the `S_<dataset>` sheets. One row per
request, plus one reduced row per item in modes that ask an item more than once.

**Identity** — every column from [§1](#1-identity-columns), plus:

| column | meaning |
|---|---|
| `sample_id` | stable id derived from the data, not from iteration order; the resume key. With `repeats > 1` it is unique per *call*, not per record |
| `record_id` | the record the call came from, through every expansion — group repeats, BOV questions and votes back to the item they belong to |
| `repeat_index` | which repeat of that record this call is (`0 … repeats-1`) |
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
| `output_budget_clamped_rate` | fraction whose budget was limited by the context window rather than the dataset's output-token ceiling — i.e. where the prompt crowded out the answer. The ceiling is 32,000 by default and 64,000 for `abd` and `boxinggym` (`max_output_tokens` in their dataset files) |
| `batch_latency_s_mean`, `completion_tokens_mean` | cost and length, per sample |
| `<primary>_strict` | the primary metric with unscored samples counted as zero |

### Mode metrics

| metric | meaning |
|---|---|
| `self_consistency_agreement` | fraction of the *k* votes that agreed with the winning answer. 1.0 is unanimous; 1/k means every sample differed |
| `bov_yes_rate` | fraction of hypotheses a BOV run said yes to. A model that says yes to everything scores well on recall alone, so this sits next to the score rather than inside it |
| `turns_used` | model turns an interactive episode took |
| `repeats` | how many times each record was asked (`modes.repeats`) |
| `repeat_agreement` | fraction of records whose repeats all agreed |
| `<primary>_repeat_std` | typical spread of the primary metric within one record |
| `self_consistency_<metric>` | **verifiable datasets.** The metric recomputed on the plurality answer over the `repeats` samples of each record. Read off those samples — no extra calls. The winning answer's score *is* the score of any repeat that produced it, so nothing is re-scored |
| `best_of_n_<metric>` | **unverifiable datasets.** The metric of the repeat the judge scored highest, averaged over records. Also read off the same samples. Every metric of the winning repeat travels with it, including the ones that were worse there — that is what makes it best-of-*n* rather than the best value of each metric separately |
| `best_of_n_n`, `best_of_n_n_records` | how many repeats each record had, and how many records contributed |

**Which of the two a dataset gets is not a setting.** A plurality needs answers
that can coincide, and free-text hypotheses never repeat verbatim, so a vote
there would be a plurality of one. Datasets whose answers are checkable
(`objective_metrics = True`) report `self_consistency_`; datasets scored by the
judge report `best_of_n_`. Both come out of the same `modes.repeats` calls.

### Answer-shape metrics

The general shapes almost every dataset reduces to. A dataset's own metric names
are these with a dataset-specific prefix (`diagnosis_match`, `root_cause_match`,
`formula_match`, ...), and mean the same thing about that dataset's answer.

| metric | meaning |
|---|---|
| `accuracy` | fraction of selection items where the chosen label was the gold label |
| `exact_match` | normalized string equality with the gold answer |
| `match` | equality *or* containment of an accepted gold form — the lenient view |
| `token_f1` | bag-of-tokens F1 against the gold answer. A **diagnostic**, never a primary metric, and not emitted at all by the datasets with no answer key |
| `set_f1`, `set_precision`, `set_recall` | for answers that are a *set* (multi-selection, causal edge sets) |
| `exact_set_match` | the whole set exactly right |
| `symbolic_match` | SymPy proves the answer equivalent to the reference expression; `symbolic_match_decidable` restricts to the items SymPy could compare, and `symbolic_undecidable` reports the rest |
| `<metric>_judged` | the same judgement made by an LLM judge. For a dataset with **no answer key this is the score**, not an extra view of it, and the run refuses to start without a configured judge. For a dataset that *has* an answer key (the diagnosis sets, `abd`) the judge is a synonym matcher over a real gold string, and the mechanical metric stands on its own |

### Per-dataset metric documentation

Every adapter documents its own metrics — what each measures, which is
**primary**, and whether **higher or lower is better** — in its
`documentation()`, and those descriptions are what the run writes into
`Datasets` sheet and `docs/datasets.md`. They are generated from the adapters,
so they cannot drift from the code. The shared metrics above are documented
here and deliberately not repeated in each of the 45 adapters.

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
| `commonwhy` | `popularity_gap` | head vs long-tail entities (`explanation_judged_head` − `explanation_judged_longtail`) |
| `medqdx` | `information_sensitivity` | accuracy at 100% vs 50% of the symptom picture |
| `abd` | `predicate_compliance` | respecting the hypothesis space is its own competence |
| `house_md` | `diagnosis_in_differential` | separates recall of the disease from committing to it |
| `medcasereasoning` | `reasoning_recall`, `reasoning_overlap` | how much of the clinician's reasoning the answer recovers |
| `open_problems_2024` | `direction_accuracy`, `brier_score`, `leakage_rate`, `direction_answer_stability` | a calibrated verdict, and whether the model had simply read the answer |
| `causalgame` | `victory_rate`, `final_score`, `deployments_used` | the simulator's own verdict against the scenario's win threshold, how close a loss was, and how much evidence was gathered first |
| `vivabench` | `accuracy` | selection over the release's candidate list — note the gold option is no longer sorted first, which it was until this was fixed |

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
decisions cannot drift from the code. Current state: **44 datasets configured, all 44
evaluable** — `researchbench` needs only `HF_TOKEN` from an account that has
accepted its gate — the 48 from the original table plus
`open_problems_2024`, built here.

Ten of the interactive benchmarks run their real environment; see
[Interactive and sequential benchmarks](#interactive-and-sequential-benchmarks).
Interactivity is no longer a reason to skip anything.

**CausalGame** now runs. Its environment *is* distributed -- the earlier note
here was wrong. The repository ships the simulator as a FastAPI service
(`uvicorn api.app:app`), so the adapter starts it locally and drives its
published API across all 14 released scenarios, scoring `victory_rate` from the
simulator's own verdict on a 1,000-drone fleet.

**Five datasets have been removed from the suite entirely** rather than carried
as permanent skips: **BioVerge** (items ship only inside an 11.9 GB corpus
archive), **DiReCT** (notes need credentialed MIMIC-IV access), **DiscoveryBench**
(gold hypotheses withheld on every scorable split), **RLF-KG** (sampled query
data is not downloadable) and **NIKA** (its emulator needs a container runtime
and `CAP_NET_ADMIN`, and this container's capability bounding set excludes both,
so no runtime can even be installed). Each added a row to every report without
ever adding a number.

NIKA's adapter was written and its scoring verified against release 0.2.0's 85
incidents before it was removed, so `git log -- src/abductionbench/adapters/nika.py`
recovers a working implementation -- along with `docs/nika-remote-lab.md`, which
describes running its emulator on a separate Docker host via the release's own
remote lab-host mode. The mechanism that reported these datasets --
`UnavailableAdapter`, which turns a dataset into a documented skip rather than a
silent absence -- stays: it is how a *newly* unobtainable dataset gets
reported.

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
* **200 records** per dataset by a seeded shuffle (`dataset_defaults.sample_size`
  in the run config -- one place for the whole suite); oversize prompts are replaced
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

**Verified working** once an account has accepted: the gate check passes, the
release's 20 files are readable, and all four tasks plan (io/cot x
generation/selection, 600 items each). Keep the token in the environment — it is
a credential and does not belong in a config file in this repository.

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

