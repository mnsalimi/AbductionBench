# Writing a child adapter (Phase 2)

A child adapter teaches the engine about **one** benchmark. It does four things
and nothing else:

```python
from abductionbench.core.adapter import AdapterContext, DatasetAdapter, SkippedDataset
from abductionbench.core.metrics import aggregate_mean_metrics, extract_choice_label
from abductionbench.core.types import AdapterDocumentation, SampleScore, SampleSpec


class MyAdapter(DatasetAdapter):
    adapter_version = "1.0"        # bump when a change invalidates old records
    primary_metric = "accuracy"    # heads the summary table

    def prepare(self) -> None: ...            # 1. materialize the data
    def build_samples(self) -> list[SampleSpec]: ...   # 2. deterministic sample
    def score(self, sample, response, *, output_contract=None) -> SampleScore: ...  # 3.
    def aggregate(self, scores) -> dict[str, float]: ...
    def documentation(self) -> AdapterDocumentation: ...   # 4. document decisions
```

Register it in config; nothing needs to import it:

```yaml
# configs/datasets/my_dataset.yaml
dataset:
  id: my_dataset
  impl: abductionbench.adapters.my_dataset:MyAdapter
  sample_size: 300
  options: {}          # anything adapter-specific, passed through untouched
```

## What the engine gives you: `AdapterContext`

| field                 | use |
|-----------------------|-----|
| `data_dir`            | per-dataset directory for downloads/caches (already created) |
| `sample_size`         | how many items to evaluate (300 by default) |
| `seed`                | determinism seed — use it, never the global RNG |
| `options`             | your `options:` block from config (`context.option("key", default)`) |
| `input_token_budget`  | the input ceiling the engine will enforce |
| `offline`             | when true, do not touch the network |
| `logger`              | pre-namespaced logger (`adapter.<dataset_id>`) |

Helpers on the base class: `self.draw(pool, n)` and `self.ordered_pool(pool)`
(seeded, reproducible shuffles), `self.clamp_max_tokens(v)`.

## 1. `prepare()` — materialize, and choose a split

Idempotent; cache under `context.data_dir`. Split preference is
**test → validation → train**; when you fall back, say so in
`documentation().split_used`. If the data cannot be obtained or read, or if you
cannot confidently identify which part of it is abductive, raise
`SkippedDataset("why")` — the dataset is then recorded as skipped with that
reason and the run continues with the others. **Never guess.**

## 2. `build_samples()` — a deterministic pseudo-random draw

Return at most `context.sample_size` `SampleSpec` objects:

```python
SampleSpec(
    sample_id="split-000123",        # derived from the data, never from iteration order
    fields={"observation": ..., "context": ..., "options": [...]},
    reference={...},                  # whatever your scorer needs; opaque to the engine
    task_kind="generation",           # or "selection" — picks the prompt template family
    max_tokens=512,                   # YOUR estimate of this item's output need
    metadata={"split": "test", "subtask": ...},   # provenance for the reports
    group_id=None,                    # optional: groups several probes of one case
)
```

Rules that matter:

* **`fields` are content, not wording.** Use the conventional names below so any
  template works with your adapter. Prompt phrasing lives in
  `configs/prompts/*.yaml`.
* **`sample_id` must be stable** across runs — it is the resume key.
* **`max_tokens` is per sample**, chosen from that item's complexity (a one-word
  label needs far less than a multi-step derivation). The engine quantizes it
  upward so similar items can share a batch call, and clamps it to the model's
  floor/cap.
* If the chosen split has fewer than `sample_size` items, return them all and
  record the shortfall in `documentation().statistics`.

### Conventional field names

| field          | meaning |
|----------------|---------|
| `observation`  | the puzzling fact(s) needing explanation **(required by every shipped template)** |
| `context`      | background: case history, rule base, logs, dialogue, theory |
| `question`     | an explicit question, when the dataset poses one |
| `options`      | candidate hypotheses (selection tasks) |
| `option_labels`| the labels for those options (usually `["A", "B", ...]`) |
| `instructions` | dataset-specific task constraints |
| `answer_format`| the answer shape, when the dataset dictates one |

### `replacement_samples(count, exclude)`

The engine calls this when a rendered prompt exceeds the input-token budget and
`engine.limits.on_oversize == "resample"`: instead of shrinking the evaluation
set, it asks for a replacement from the same split. Implement it for any dataset
with long items — the easy pattern is

```python
def prepare(self):
    self._pool = self.ordered_pool(all_items)      # one seeded shuffle
def build_samples(self):
    return [self._spec(x) for x in self._pool[: self.context.sample_size]]
def replacement_samples(self, count, exclude):
    return [self._spec(x) for x in self._pool if self._id(x) not in exclude][:count]
```

so a replacement is the next unused item of the *same* draw, never a re-draw.

## 3. `score()` and `aggregate()`

```python
def score(self, sample, response, *, output_contract=None) -> SampleScore:
    prediction = extract_choice_label(response.text, sample.fields["option_labels"], output_contract)
    if prediction is None:
        return SampleScore(metrics={"accuracy": 0.0}, parse_ok=False)   # unparseable ≠ wrong
    return SampleScore(
        metrics={"accuracy": float(prediction == sample.reference["gold"])},
        prediction=prediction,
    )
```

* **Never raise.** A malformed response returns `parse_ok=False` with zeroed
  metrics; the engine tracks `parse_failure_rate` separately so a formatting
  problem is distinguishable from a wrong answer. (A scorer that raises anyway
  is caught and recorded, but you lose the metric.)
* **Honour `output_contract`** — it is the active template's declared answer
  format (`answer_prefix`, `answer_regex`, …). Use the helpers in
  `abductionbench.core.metrics` (`extract_answer_span`, `extract_choice_label`)
  and your adapter keeps working when the prompt template is swapped.
* `response.status` may be `empty` (a reasoning model spent its whole budget on
  hidden chain-of-thought and returned `content: null`) or `truncated`.
  `response.text` is `""` in the empty case.
* `aggregate()` receives the scored samples only. The engine adds `coverage`,
  `parse_failure_rate`, `<primary>_strict`, `truncation_rate`,
  `empty_response_rate` and token/latency statistics on top.

### Metrics are yours to define

Each dataset's task defines its own metric; `core/metrics.py` supplies the
primitives (normalized exact match, token F1, ROUGE-L, BLEU, set P/R/F1,
hits@k, MRR, Spearman/Kendall, numeric tolerance, Brier). Compose them, and
describe every metric you emit in `documentation().metrics_description`.

### Optional: LLM-as-judge

For free-form tasks where paraphrases are correct answers, override
`judge_request()` / `apply_judge()`. The judge runs as a second batched pass
using a configured judge template, is cached on disk, and is inert unless
`engine.judge.enabled` is true — so deterministic metrics stay the default.

## 4. `documentation()` — mandatory, and it ends up in the report

```python
AdapterDocumentation(
    dataset_id=self.dataset_id,
    name="...", domain="...", source_url="...", processing_mode="Generation",
    split_used="test (official); 1,200 items",
    abductive_subset="only the 'why' questions whose gold answer is a cause",
    sampling_procedure="seeded shuffle (seed=<seed>), first 300 items",
    metrics_description={"accuracy": "..."},
    primary_metric="accuracy",
    decisions=["Chose X over Y because ..."],
    caveats=["..."],
)
```

Every field lands in `run_documentation.md` and in the workbook's `Datasets`
sheet. The engine additionally fills `statistics` with the split size, sample
size, oversize/replacement counts, and input-token statistics.

## Checklist

- [ ] Data cached under `context.data_dir`; `prepare()` is idempotent.
- [ ] Split preference test → validation → train, and it is documented.
- [ ] Only the abductive portion is used (or the dataset is skipped, with a reason).
- [ ] `sample_id` derived from the data; sampling seeded from `context.seed`.
- [ ] `max_tokens` reflects per-item complexity.
- [ ] `replacement_samples()` implemented if items can be long.
- [ ] `score()` never raises and honours `output_contract`.
- [ ] Every emitted metric is described in `documentation()`.
- [ ] `SkippedDataset` used instead of guessing.
