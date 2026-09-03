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
abench run    configs/runs/full.yaml           # 300 samples x 39 datasets x every model

# interrupted? continue exactly where it stopped (per-sample checkpoints):
abench run configs/runs/full.yaml --resume runs/<run-id>
```

`configs/runs/full.yaml` is the single place that decides scope: it picks up
every dataset config automatically and lists the models. Adding a model is one
line there. One model over the whole suite is 9,882 prompts (~1,270 native batch
calls at group size 8) and takes roughly 2-3 h on an idle GPU; selection
datasets cost ~0.02 s/sample, long-form physics ~4 s/sample.

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
decisions cannot drift from the code. Current state: **48 datasets configured,
39 evaluable, 9 reported as skipped with a reason.**

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

Two datasets required constructing items from released material rather than
using a shipped item file; both say so explicitly and are excluded from
comparison with published numbers: **ProofWriter** (the official abduction files
are no longer downloadable, so abduction items are rebuilt from the released
proofs) and **SynPAT** (items assembled from the release's own replacement files
plus true-system data). **HypoSpace** runs the release's own seeded generator.

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
