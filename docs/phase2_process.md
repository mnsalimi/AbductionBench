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

## 6. Verification performed

* `abench prepare configs/runs/pilot.yaml` materializes all 48 configured
  datasets: 39 build their samples, 9 report their skip reason. No dataset
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
