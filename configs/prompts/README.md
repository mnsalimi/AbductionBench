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

## The reasoning-metric judges (`judge/reasoning_*.yaml`)

Nine templates measure the *structure of a chain of reasoning* rather than
whether an answer is right.  They are used only for `cot` and
`self-consistency` outputs; the stage that runs them refuses an `io` output,
because there is no chain in one to measure.

One template per metric family, and the grouping is the measurement's rather
than a convenience.  `reasoning_steps_v1` returns four counts -- total, useful,
useless and backtracking steps -- in one reply precisely because they have to
come from one segmentation of one chain; asking four times would segment it
four ways and the counts would not add up.

Their `output_contract` uses a key the answer judges do not:

```yaml
output_contract:
  json_fields:
    total_steps: nonnegative_integer
    backtracking_steps: nonnegative_integer
```

`json_fields` names the keys the reply must contain and the shape each value
has to take (`nonnegative_integer`, `nonnegative_number`, `binary_integer`,
`directionality`).  A reply missing any of them is not parsed, is not cached,
and is reported as an error rather than filled in with a zero.  The parser
tolerates a judge that thinks out loud: it takes the *last* balanced JSON object
in the reply, so an echoed example followed by the real verdict resolves to the
verdict.

**Two rules these templates must keep.**

1. **Raw counts only.**  No template may mention normalizing, dividing, a ratio,
   a fraction or a percentage.  Every derived value is computed in
   `derive_reasoning_metrics` after the raw values come back -- a judge asked for
   a ratio has to do arithmetic on its own counts, and the sheet can then
   disagree with its own columns.  `tests/test_reasoning_judge.py` greps every
   shipped template for that vocabulary and fails on a hit.
2. **Declared fields must be asked for.**  Every name under `json_fields` has to
   appear in the prompt body, so the contract the parser enforces is the one the
   judge was actually given.  There is a test for that too.
