# How the child adapters were built (Phase 2 process record)

This is the audit trail for Phase 2: how each dataset in the suite was located,
what was inspected, and the rules applied when the specification left a choice
open. The per-dataset outcomes live in `docs/datasets.md` (generated from the
adapters); this document records the *process*.

## 1. Reconnaissance

Every source in the table was fetched before any adapter was written: 32 GitHub
repositories cloned, 10 Hugging Face datasets probed via the Hub API, and the
remaining sources resolved by hand:

| source in the table | what it actually was |
|---|---|
| GEAR (`UNVERIFIED` in the table) | confirmed from arXiv:2509.24096 → `github.com/KaiyuHe998/GEAR-Abduction_evaluation` |
| `bit.ly/4p9ltW8` (House M.D.) | a Kaggle dataset, downloadable through Kaggle's public API without credentials |
| `huggingface.co/collections/oriel9p/medups` | a *collection*, not a dataset repo → resolved to `oriel9p/MedUPS_final_diagnosis` and `oriel9p/MedUPS_mid_stream` |
| ProofWriter (`allenai.org/data/proofwriter`) | the AI2 S3 archive now returns HTTP 403; the abduction files exist in no accessible mirror |
| `causalgame.github.io` | an interactive game harness (`github.com/CausalGame/CausalGame`), no released items |
| `aiops.cn/gitlab/...` | clonable GitLab repo; the usable item set is `RCA100/` |
| DDXPlus (figshare) | the figshare API exposes direct file URLs; the patient archives hold one extension-less CSV each |
| MOOSE-Chem2 / BioVerge Google Drive links | MOOSE's archive holds run checkpoints (the gold data is in the repo's own workbook); BioVerge's is an 11.9 GB corpus |

## 2. Rules applied when the specification was open

**Split.** test → validation → train, always reported. Where a benchmark's test
labels are withheld upstream, the labelled split is used and that is stated
(e-CARE dev, AER dev, UNcommonsense validation, ProofWriter's own test files).

**Which part is abductive.** Only the abductive portion is evaluated, and both
the kept and the discarded part are named. Concretely: e-CARE
`ask-for == "cause"` (1,088 of 2,131 dev items); XCOPA `question == "cause"`
(3,096 effect items dropped); MuSR murder mysteries only (object placement =
theory of mind, team allocation = optimization); SciR's causal family only
(deduction/induction are other paradigms); MedR-Bench's diagnosis collection
(treatment planning is decision-making); GEAR's ACRE family (ARC-AGI is program
induction); NeuLR's `abductive_neutral.json` only; ENWN/EntailmentBank's
`data/step/abductive/` only; the abduction benchmark of concept-synth (not its
induction benchmark). Where it was *not* clear, the dataset was skipped rather
than guessed at.

**Processing mode.** The table's mode is honoured unless the release makes it
impossible, in which case the deviation is recorded in the adapter's decisions:
DeFAb is rendered as selection because the release ships candidate sets and its
gold strings are internal rule identifiers; HypoArena, MedR-Bench and
MOOSE-Chem2 are generation-only because no candidate sets exist. Datasets marked
"Generation / Selection" default to the more clearly abductive mode with the
other available through `options.subtask`.

**Answer leakage.** Every field that reveals the answer is withheld from the
prompt and listed in the adapter's decisions — PhysGym's `solution`,
MedCaseReasoning's `diagnostic_reasoning`, HypoGen's title/abstract/spark,
MOOSE-Chem2's inspirations and reasoning chain, CommonWhy's `rule`,
Matter-to-Mechanism's mechanism/intervention columns, MedUPS's `cot`/`final_answer`.
Model-generated columns are never used as gold (UNcommonsense's
`gpt4_explanations`, House M.D.'s per-model answers, MedUPS's `raw_response`,
CausalOpsBench's `results/*/predictions/`).

**`max_tokens` per sample.** Derived from the item, not fixed per dataset:
ProofWriter scales with proof depth, ABD with `alphaTier`, CausaLab with node
count, Matter-to-Mechanism with the dataset's own `num_reasoning_steps`,
HypoSpace with how many hypotheses were requested. Short-answer selection tasks
still get a few hundred tokens because reasoning models spend budget on hidden
chain-of-thought before emitting an answer.

**Input-token budget.** Enforced by the engine at 16,000 input tokens with
replacement from the same seeded shuffle. Datasets whose raw evidence dwarfs
that budget include a bounded slice, and the bound is configurable *and*
reported: AER (retrieved corpus ~60k tokens/topic → title+snippet),
Cloud-OpsBench (tool cache ~300 KB → 6 entries × 220 words), AgentRx
(trajectories 100–180 KB → failure summary, trajectory optional), CausalOpsBench
(60-point series → 12 timepoints), AIOps2025 (alert 400 words, topology 40
entities/60 edges), CausaLab (20 bootstrap observations).

## 3. Constructed items (and why that is disclosed)

Three datasets could not be evaluated from a shipped item file. In each case the
construction uses only released material, is seeded, and is flagged in the
adapter's decisions and caveats so its numbers are never compared to published
results:

* **ProofWriter** — the official abduction files are unreachable (AI2 S3 403; HF
  mirrors carry only QA/proof data). Abduction items are rebuilt exactly as the
  paper defines the task: delete one ground fact used in a provable statement's
  proof, ask for it back. Only depth ≥ 1 questions are used, so the answer is
  never the observation restated.
* **SynPAT** — the release ships a true system, five single-equation
  corruptions, and data files. An item pairs a corrupted system with the true
  system's data and asks for the corrected equation.
* **HypoSpace** — the release ships generators, not data. The causal generator
  is pure Python and seeded, so the adapter runs it once and caches its JSON.

A fourth dataset, `open_problems_2024`, is not from the specification's table at
all: it was assembled here from two independent surveys of research problems
open at a 2023 cutoff and resolved in 2024 or later. It is covered in section 6.

## 4. Skipped datasets

Nine datasets are declared with `UnavailableAdapter` and appear in every run's
`Skipped` sheet with a full reason. Grouped by cause:

* **Interactive environments with no released items** — BoxingGym, SciLab,
  CausalGame, NIKA (its YAMLs specify fault injection; telemetry only exists once
  the emulator runs).
* **Data behind access control** — ResearchBench (gated HF repo, HTTP 401),
  DiReCT (annotated MIMIC-IV notes need PhysioNet credentials; 4 public samples).
* **Data not obtainable at a sane cost** — RLF-KG (sampled queries on a personal
  OneDrive share returning 404), BioVerge (items only inside an 11.9 GB corpus
  archive).
* **Gold withheld upstream** — DiscoveryBench (`hypotheses.main` empty in 0/144
  real-test and 0/200 synth-test files; only 14 train files carry hypotheses, and
  the task needs code execution over CSVs to be meaningful).

## 5. Overlap found between two table entries

`diagnosisarena` and `med_inquire` turned out to be the **same 915 case
reports** — EvoClinician ships DiagnosisArena's release as its Med-Inquire test
file. Rather than report them as two independent difficulty probes, both
adapters carry the overlap in their caveats, and they are differentiated by
evidence level: DiagnosisArena shows the full work-up (its own protocol), while
Med-Inquire withholds it (EvoClinician's premise is that an agent must ask).
Read together, the pair measures the value of the diagnostic work-up.

## 6. The one dataset built from scratch: `open_problems_2024`

Requested separately, after the other 48, as a scientific-discovery probe. It is
documented here because it is the only dataset whose *items* originate in this
repository rather than in a release.

**Whether it is abduction — decided per mode, not per dataset.** The sources
supply up to four prompts per problem, and they are not the same task:

| mode | what the model is asked | classification |
|---|---|---|
| `direction` | which way an open conjecture resolved, plus a probability | Stage 2, single-hypothesis evaluation — **abductive** |
| `strategy` | what a solution would have to use, before seeing one | Stage 1, knowledge completion — **abductive** |
| `resolution` | resolve the problem | deduction — **not** abductive; excluded from `abduction_score` |
| `leakage_probe` | what the model knows about the problem's status | not reasoning at all; a contamination control |

The honest caveat, recorded in the adapter's own caveats and therefore in
`docs/datasets.md`: in `direction` mode the object being judged is a
proposition's truth, not a hypothesis's explanatory power over observations.
That is plausible reasoning at the edge of abduction rather than at its centre.
It is included on the strength of the framework's single-hypothesis-evaluation
definition, and labelled as such rather than quietly counted.

**Provenance is mechanical.** Both sources are committed under
`assets/open_problems_2024/sources/`, and
`tools/build_open_problems_dataset.py` regenerates `problems.json` from them.
All judgement lives in one auditable `NORMALIZATION` table keyed by task id —
label sets, gold labels, partial-credit labels, posing year, solve date,
solvers, and key ingredients quoted verbatim from each card's "Key method"
field. Prompts are the sources' own wording, unedited; the three problems that
exist only in the independent report had no direction prompt, so theirs were
written here under the primary source's neutral-framing rules and carry
`prompt_authorship` in the output. The corroboration between the two surveys is
matched at build time against the report's own words, so if the report changes
the build fails rather than asserting a verification that no longer holds.

**Contamination is the real threat, so it is measured, not assumed.** Every
resolution is 2024 or later. `options.model_cutoff` drops problems resolved
before a model's training cutoff, and the `leakage_probe` subtask asks directly
what the model knows, scoring `leakage_rate` per item — a model that both
asserts the problem is resolved and names the solver or the year has read the
answer, and that item's reasoning score means nothing. On gemma-4-E4B:
`leakage_rate = 0.00` over 15 items.

**Nondeterminism, which on a dataset this small is a first-order problem.** Two
temperature-0, fixed-seed runs of the 15 direction items disagreed on 5 of them.
The cause is not the adapter and not the sampler: vLLM's batched inference is not
numerically batch-invariant, so a prompt's logits depend on what shares its
batch. The evidence is clean in both directions — two consecutive runs of this
dataset *alone* agreed on 45/45 direction predictions and reproduced
`abduction_score` to four decimals, while the run that disagreed on 5 items had
three other 300-item datasets in flight. Even inside a single run, 2 of 15
questions got different verdicts from their own three repeats, which sit at
different positions in different batches. One run of 15 items is therefore not a
measurement. `options.repeats`
(3 by default here) asks each question k times as k distinct samples; the score
is the mean over all observations, and two metrics report how much of it is
noise — `direction_answer_stability` (fraction of questions whose repeats all
returned the same verdict; 0.87 running alone) and `strategy_score_spread` (mean
within-question range of `key_ingredient_recall`, since a strategy answer is
never byte-identical twice and its text stability is meaningless). Averaged over
3 repeats, gemma-4-E4B scores direction accuracy 0.38 and
`key_ingredient_recall` 0.29.

**What it cannot be.** 15 evaluable problems (plus one held out pending peer
review) is a qualitative probe, not a leaderboard, and the shortfall against the
300-sample target is reported rather than padded. 14 of 16 are mathematics or
theoretical computer science: the primary source searched empirical fields and
found essentially nothing meeting a strict "posed before 2024, first resolved
after 2023, community-accepted" test, because empirical fields rarely close a
problem on a datable event.

## 7. Verification performed

* `abench prepare configs/runs/pilot.yaml` materializes all 49 configured
  datasets: 40 build their samples, 9 report their skip reason. No dataset
  errors out.
* `tools/preview.py` was run per adapter to check the rendered prompt, the
  reference payload, per-sample `max_tokens` and input-token statistics.
* `tests/test_adapter_scorers.py` unit-tests the bespoke scorers (symbolic
  equation comparison, formula canonicalization, hypothesis-set parsing,
  multi-label set scoring, edge-set F1, and the seeded-draw/replacement
  contract).
* A live run against `openai/gpt-oss-120b` exercised every task kind
  (selection, generation, knowledge completion, multi-answer selection) through
  the real batch endpoint.
* `tests/test_open_problems.py` checks the bundled dataset itself, not just the
  code: every item gradeable against its own label set, every problem posed
  before 2024 and resolved after it, no strategy item without ingredients to
  grade against, and gold labels balanced enough that always answering "the
  conjecture holds" cannot score well.
* `open_problems_2024` was run live beside `ecare`, `abductionrules` and
  `medcasereasoning` in one invocation, to confirm a 28-item dataset with four
  prompt templates and its own primary metric does not disturb the 300-item
  tasks sharing the batch endpoint with it.
