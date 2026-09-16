# Prompt templates

Prompt wording is configuration, not code.  Each YAML file here is one
*versioned* template that the engine binds to a `task_kind` declared by a
dataset adapter (`generation`, `selection`, `judge`, ...).

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
