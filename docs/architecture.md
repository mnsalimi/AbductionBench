# Architecture

AbductionBench is split into two layers with a single, narrow seam between them.

```
                    ┌───────────────────────────────────────────────┐
 configs/           │                  CORE ENGINE                  │
 ├── engine/        │  (dataset-agnostic — Phase 1)                 │
 ├── models/        │                                               │
 ├── prompts/  ────▶│  prompts.py    render fields → conversation    │
 ├── datasets/      │  tokenizer.py  count input tokens             │
 └── runs/     ────▶│  batching.py   pack into native batch calls   │
                    │  client.py     gateway + direct batch route   │
                    │  retry.py      classify · backoff · recover   │
                    │  engine.py     orchestrate · score · persist  │
                    │  checkpoint.py atomic appends · resume        │
                    │  judge.py      optional LLM-judge stage       │
                    │  metrics.py    generic metric primitives      │
                    │  reporting.py  Excel grid · CSV · run docs    │
                    └───────────────────────┬───────────────────────┘
                                            │  DatasetAdapter ABC
                    ┌───────────────────────┴───────────────────────┐
                    │              CHILD ADAPTERS                   │
                    │  (one per benchmark — Phase 2)                │
                    │  prepare → build_samples → score → aggregate  │
                    │           → documentation                     │
                    └───────────────────────────────────────────────┘
```

## The seam

The engine and an adapter exchange exactly four kinds of object
(`core/types.py`):

| object                 | direction        | meaning |
|------------------------|------------------|---------|
| `SampleSpec`           | adapter → engine | one item: prompt **fields**, gold `reference`, `task_kind`, `max_tokens`, metadata |
| `RenderedPrompt`       | engine internal  | a sample after template rendering, budgeting and sampling resolution |
| `ModelResponse`        | engine → adapter | normalized model output plus status, finish reason, usage, latency |
| `SampleScore`          | adapter → engine | per-sample metrics, parsed prediction, `parse_ok`, details |

Plus `AdapterDocumentation`, which the engine writes verbatim into the run
documentation — an adapter cannot ship undocumented choices.

The engine never looks *inside* `fields`, `reference` or `metadata`. That is
what keeps it dataset-agnostic while still owning prompt rendering, batching,
retries and reporting.

## Three execution stages

1. **Dataset bundles** (once per dataset). Resolve the adapter from
   `impl: "module:Class"`, `prepare()` it, take its deterministic sample. An
   adapter that raises `SkippedDataset` is recorded and the run continues.
2. **Prompt sets** (once per dataset × template variant). Render through the
   bound template, count input tokens, replace oversize items with fresh draws
   from the same split. Deliberately **model-independent**, so every model sees
   the identical prompt set and cross-model numbers are comparable.
3. **Tasks** (dataset × variant × model). Resolve sampling params against that
   model's limits, pack into native batch calls of that model's own group size,
   submit with retry/recovery, score, append to `records.jsonl`.

## Network topology

```
 this host (datasets + engine)                    GPU server
 ┌──────────────────────────┐               ┌────────────────────────┐
 │ abench run               │  /v1/models   │  LiteLLM gateway       │
 │  ├─ discovery ───────────┼──────────────▶│   (model registry,     │
 │  ├─ single calls ────────┼──────────────▶│    per-user keys)      │
 │  │                       │               │        │               │
 │  └─ BATCH calls ─────────┼──────────────▶│  ┌─────▼─────────────┐ │
 │      POST /v1/chat/      │  (own tunnel  │  │ vLLM  :18004      │ │
 │      completions/batch   │   or gateway  │  │ /v1/chat/         │ │
 └──────────────────────────┘   passthrough)│  │  completions/batch│ │
                                            │  └───────────────────┘ │
                                            └────────────────────────┘
```

`endpoint.base_url` (discovery + single calls) and
`endpoint.batch.base_url` + `endpoint.batch.path` (batch calls) are separate
config fields precisely because the batch route is **not** an OpenAI route: it
exists only on a model's own address or on a dedicated gateway pass-through
path. Both topologies — and a tunnel-free local/SSH-forwarded one — are
expressible without touching code (`configs/models/*.yaml`).

## Batch-endpoint semantics the engine is built around

Established by reading vLLM 0.28's
`entrypoints/openai/chat_completion/batch_serving.py` and by probing the live
server:

1. **One shared sampling-parameter set per call.** `messages` is a list of
   conversations, but `max_tokens`/`temperature`/… apply to all of them. The
   engine therefore groups samples by sampling signature and quantizes
   adapter-supplied `max_tokens` upward onto a common grid
   (`engine.batching.max_tokens_quantum`) so per-sample budgets do not shatter
   batches into singletons. Rounding up only ever grants headroom.
2. **`choices[i]` maps to conversation `i`** (server sorts by index; the client
   re-validates count and indices, and refuses to guess if they disagree —
   mis-aligning responses to samples would corrupt every metric).
3. **`usage` is the aggregate over the whole call**, not per conversation. Each
   record stores the batch aggregate and flags `usage_is_batch_aggregate`,
   plus a per-sample `completion_tokens_est` counted locally.
4. **One bad conversation fails the entire call.** Hence bisection: on an
   invalid-request/context-length failure the batch is split to isolate the
   offender and salvage the rest.
5. **`content` can be `null`** on a reasoning model whose hidden
   chain-of-thought consumed `max_tokens` (the text lands in `message.reasoning`
   / `reasoning_content`). That is its own status (`empty`), not an error, and
   it is reported as `empty_response_rate`.
6. **Proxies impose a request ceiling.** Cloudflare quick tunnels cut requests
   off at ~100 s with HTTP 524. A batch whose retries are exhausted by
   timeouts is also split (`engine.batching.bisect_on_timeout`), which both
   shortens each call and salvages samples.
7. **An empty or cut-off response is an under-budgeted request, not a
   failure.** When a model returns `content: null` with
   `finish_reason="length"`, the engine re-asks *only that sample* with a
   multiplied budget (`engine.retry.escalate_empty_responses`, on by default).
   Measured on gpt-oss-120b over AbductionRules: 16.7% of samples came back
   empty at a 768-token floor, and escalation took that to 0% at the cost of one
   extra call per affected sample -- far cheaper than raising the budget for all
   300. `escalate_truncated_responses` (off by default, since a verbose model
   would double the cost of every long-form dataset) extends the same treatment
   to answers that were cut off mid-way; on gemma-4-E4B over PhysGym it took the
   truncation rate from 50% to 0%.
8. **A small context window silently shortens answers.** With
   `limits.context_window` set, the engine clamps each sample's output budget to
   what is left after its prompt, counts how often it had to, and reports
   `output_budget_clamped_rate` plus a warning naming the worst sample. On a
   16k-window model (gemma-4-E4B) the longest prompt any adapter currently
   produces is ~8.1k tokens, so nothing is clamped -- but the metric is there so
   a tighter model cannot quietly turn a scoring result into a budget artifact.

## Failure handling summary

| failure | classification | reaction |
|---|---|---|
| connection refused / reset, timeout, 5xx, 524 | `transient` | backoff + retry; endpoint recovery; split batch when retries are exhausted |
| HTTP 429 | `rate_limit` | dedicated longer backoff, then retry |
| HTTP 401/403 | `auth` | abort the task immediately (retrying cannot help) |
| HTTP 400/422 | `invalid_request` | bisect to isolate the sample, then fail just that one |
| "maximum context length …" | `context_length` | bisect, then mark that sample `skipped` |
| wrong choice count/index | `protocol` | retry, then fail the batch's samples |
| adapter `score()` raises | — | logged, sample recorded with `parse_ok=false` |
| adapter `prepare()` raises | — | dataset skipped and reported; run continues |
| `content: null` at the budget | — | that sample re-asked with a ×N budget, then reported as `empty` if still empty |

**Endpoint recovery** is what makes a rotating tunnel survivable: on a
connection-level failure the engine re-runs the model's configured
`discovery.*_command` (e.g. `grep -o 'https://…trycloudflare.com'
/var/log/portal/gateway-tunnel.log | tail -1`), picks up the new URL, and probes
`/v1/models` until the endpoint answers — serialized so a burst of failures
triggers one recovery, not one per call.

## Output layout

```
runs/<run-id>/
├── run_config.resolved.yaml      fully resolved config (API keys redacted)
├── engine.log · engine.jsonl     human + JSON logs
├── events.jsonl                  structured telemetry (batch/sample/task events)
├── RUN_REPORT.md                 human summary of the whole run
├── datasets/<dataset>/<model>/<template@version>/
│   ├── records.jsonl             one JSON object per sample (atomic appends)
│   ├── metrics.json              aggregated metrics + diagnostics
│   ├── checkpoint.json           resume state and batch statistics
│   ├── run_documentation.md      what was run, and the adapter's decisions
│   ├── judge_cache/              judge verdicts (when the judge stage ran)
│   └── raw/<batch-id>.json       raw request/response payloads
└── reports/
    ├── abductionbench_results.xlsx   the unified grid
    ├── summary.csv · metrics_long.csv
```

The hierarchy is `dataset / model / template@version`, so a run that compares
two prompt templates on the same dataset keeps them in separate directories and
separate rows of the grid.

## Resume semantics

`records.jsonl` is append-only; each record carries a `prompt_fingerprint` =
hash(model, template@version, messages, sampling). Under the default
`resume_policy: strict` a stored record is reused only if that fingerprint still
matches, so changing a prompt template can never silently mix results from two
different prompts. `error` records are always retried; `skipped` ones are
reproducible decisions and are kept.
