# Prompt templates

Prompt wording is configuration, not code. Dataset prompts live in their
adapters; the versioned templates here control semantic answer judging and COT
reasoning-structure judging.

## The field contract

Adapters supply *fields*, the template supplies *wording*.  The conventional
field names (see `docs/adapter_contract.md`) are:

| field          | meaning                                                        |
|----------------|----------------------------------------------------------------|
| `observation`  | the puzzling fact(s) that need explaining  **(required)**       |
| `context`      | background: case history, rule base, logs, theory, dialogue     |
| `question`     | an explicit question, when the dataset poses one                |
| `options`      | candidate hypotheses to choose between (selection tasks)        |
| `instructions` | dataset-specific task constraints (what to produce, not how to word it) |
| `answer_format`| the shape the answer must take, when the dataset dictates one    |

A template must declare every field it reads under `required_fields` or
`optional_fields`; rendering uses `StrictUndefined`, so a typo fails loudly
instead of producing a silently empty prompt.

## Swapping templates

* Change one binding for a whole run: `prompts.bindings.generation: gen_cot_v1`.
* Change it for one dataset only: `prompts.dataset_overrides.<dataset_id>`.
* Compare several templates in a single run: `prompts.template_variants`, which
  turns each variant into its own task, its own output directory and its own
  row in the result grid.

## `output_contract`

Whatever a template promises about the answer's shape is declared in
`output_contract` and handed to the adapter's scorer, so a scorer follows the
active template rather than hardcoding one prompt's format.  Recognized keys
used by the generic helpers in `abductionbench.core.metrics`:
`answer_prefix`, `answer_regex`, `strip_markdown`, and for judge templates
`verdict_regex`, `score_regex`, `labels`, `expect_numeric_score`, `score_scale`.

## COT reasoning metrics

`judge/reasoning_*.yaml` contains one strict-JSON prompt per requested metric
family. Multi-output definitions stay grouped in one call: observation
coverage, branchiness/diversity, density/step counts, and
redundancy/completeness. The question-only observation inventory is the sole
extra prompt; it is cached by exact rendered question across models and repeats.

Enable the stage independently of semantic answer judging:

```yaml
engine:
  reasoning_judge:
    enabled: true
    model: judge-model-id   # one of the run's configured models
```

The stage runs only for `cot` and `self-consistency` prompt modes. It never
calls a judge or adds reasoning metrics to `io` records. Raw judge outputs are
validated, normalized values are calculated locally from shared counts, and
undefined/inapplicable values are named in the sample sheet instead of being
forced to zero.
