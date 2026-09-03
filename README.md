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

## Use

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
  id: gpt-oss-120b
  model_name: openai/gpt-oss-120b
  endpoint:
    base_url: ${env:ABENCH_GATEWAY_URL}       # discovery + single calls
    api_key:  ${env:ABENCH_API_KEY}
    batch:
      base_url: ${env:ABENCH_BATCH_URL}       # the native batch route lives here
      path: /gpt-oss-120b/v1/chat/completions/batch
      group_size: 8                           # 8 samples per batch call, server-side
```

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
`parse_failure_rate`, `truncation_rate`, `empty_response_rate`, and
`<primary>_strict` (the primary metric with unscored samples counted as zero),
so a number produced under endpoint trouble cannot look like a clean result.

## Adding a dataset

See `docs/adapter_contract.md`. In short: implement `prepare`,
`build_samples`, `score`, `aggregate`, `documentation`; point a dataset config
at it with `impl: "module:Class"`. A dataset that cannot be obtained, parsed, or
whose abductive subset is unclear must raise `SkippedDataset("reason")` — it is
then reported as skipped rather than guessed at.

## Development

```bash
python -m pytest -q            # 57 tests: config, prompts, batching, metrics,
                               # checkpointing, judge, and end-to-end engine runs
                               # against a fake vLLM server (real HTTP)
abench run configs/runs/smoke_local.yaml     # live smoke test, ~30 s
```
