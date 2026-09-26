"""The paper's main results table (LaTeX), every number read from the records.

    PYTHONPATH=src .venv/bin/python tools/main_table_latex.py OUT.tex

Static delivery: runs/20260924-002252_openrouter-trio (50 records x 3 repeats
at T=0.7; score = mean over every scored sample). Interactive and sequential:
the two episode runs (50 records each, asked once). A score is the mean of the
record-level metric -- never a report's cached value.

A cell is "--" when the run does not exist by design (CoT for the io-only
interactive / sequential protocols) and "--$^\\dagger$" when it was started but
covers fewer than 90 % of the planned samples (Qwen3.5-27B's CoT pass stopped
early on several static datasets).
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import orjson

STATIC_RUN = "/workspace/AbductionBench/runs/20260924-002252_openrouter-trio"
EPISODE_RUNS = ["/workspace/AbductionBench-interseq/runs/20260925-210104_episode-trio50",
                "/workspace/AbductionBench/runs/20260925-225607_episode-qwen27-local4"]
MODELS = [("gpt-5.6-luna-openrouter", "GPT-5.6-Luna"), ("gemini-3.8-flash-openrouter", "Gemini-3.8-Flash"),
          ("gemma-4-31b-it-openrouter", "Gemma-4-31B"), ("qwen3-5-27b-openrouter", "Qwen-3.5-27B"),
          ("gemma-4-e4b-local", "Gemma-4-E4B"), ("gemma-4-e2b-local", "Gemma-4-E2B"),
          ("qwen3-5-4b-local", "Qwen-3.5-4B"), ("qwen3-5-2b-local", "Qwen-3.5-2B")]
MIN_COVERAGE = 0.9
STATIC_PLANNED = 150
EPISODE_PLANNED = 50

# (dataset, [(regime label, template prefix, metric key, metric label), ...])
STATIC_SELECTION = [
    ("art", [("MCS", "MCS", "set_exact_match", "EM"), ("", "MCS", "set_f1", "F1"),
             ("SCS", "SCS", "accuracy", "ACC"), ("Gen", "n-a_generation", "hypothesis_judged", "LLM-J")]),
    ("ecare", [("MCS", "MCS", "set_exact_match", "EM"), ("", "MCS", "set_f1", "F1"),
               ("SCS", "SCS", "accuracy", "ACC"), ("Gen", "n-a_generation", "explanation_judged", "LLM-J")]),
    ("climate_fever", [("MCS", "MCS", "set_exact_match", "EM"), ("", "MCS", "set_f1", "F1"),
                       ("SCS", "SCS", "accuracy", "ACC")]),
    ("true_detective", [("MCS", "MCS", "set_exact_match", "EM"), ("", "MCS", "set_f1", "F1"),
                        ("SCS", "SCS", "accuracy", "ACC")]),
    ("diagnosisarena", [("MCS", "MCS", "set_exact_match", "EM"), ("", "MCS", "set_f1", "F1"),
                        ("SCS", "SCS", "accuracy", "ACC")]),
    ("agentrx", [("MCS", "MCS", "set_exact_match", "EM"), ("", "MCS", "set_f1", "F1"),
                 ("SCS", "SCS", "accuracy", "ACC")]),
    ("aer", [("MCS", "MCS", "exact_set_match", "EM"), ("", "MCS", "set_f1", "F1")]),
]
STATIC_GENERATION = [
    ("aiops2025", "root_cause_match", "RC"), ("commonwhy", "explanation_judged", "LLM-J"),
    ("crosstrace", "insight_judged", "LLM-J"), ("house_md", "diagnosis_judged", "LLM-J"),
    ("hypogen", "flip_judged", "LLM-J"), ("llm4biohypogen", "hypothesis_judged", "LLM-J"),
    ("matter_to_mechanism", "hypothesis_judged", "LLM-J"), ("medcasereasoning", "diagnosis_judged", "LLM-J"),
    ("medr_bench", "diagnosis_judged", "LLM-J"), ("uncommonsense", "proxy_closest_explanation_score", "Proxy"),
    ("uniadilr_hgc", "premise_set_match", "Set-EM"),
]
STATIC_COMPLETION = [
    ("enwn_entailmentbank", "premise_judged", "LLM-J"), ("neulr", "premise_judged", "LLM-J"),
    ("proof_writer", "exact_match", "EM"),
]
# (dataset, [(regime, template prefix, metric label)]) -- final_answer_accuracy
INTERACTIVE = [
    ("medqdx", [("Gen", "io_n-a_interactive", "LLM-J")]),
    ("med_inquire", [("Gen", "io_n-a_generation_interactive", "LLM-J")]),
    ("vivabench", [("Gen", "io_n-a_generation_interactive", "LLM-J")]),
    ("ddxplus", [("SCS", "io_SCS_selection_interactive", "ACC"), ("MCS", "io_MCS_selection_interactive", "ACC")]),
    ("cloud_opsbench", [("Gen", "io_n-a_interactive", "JRA")]),
]
SEQUENTIAL = [("athena_bench", [("Gen", "io_n-a_sequential", "LLM-J")])]

_cache: dict[str, list[dict]] = {}


def _records(path: str) -> list[dict]:
    if path not in _cache:
        latest = {}
        with open(path, "rb") as fh:
            for line in fh:
                if line.strip():
                    try:
                        r = orjson.loads(line)
                    except orjson.JSONDecodeError:
                        continue
                    latest[(r.get("sample_id"), r.get("prompt_fingerprint"))] = r
        _cache[path] = [r for r in latest.values() if not (r.get("metadata") or {}).get("reduced")]
    return _cache[path]


def score(run: str, dataset: str, model: str, template_glob: str, key: str, planned: int) -> str:
    paths = glob.glob(f"{run}/datasets/{dataset}/{model}/{template_glob}@*/records.jsonl")
    if not paths:
        return None
    recs = [r for r in _records(paths[0]) if r.get("status") in ("ok", "empty", "truncated")]
    vals = [(r.get("metrics") or {}).get(key) for r in recs]
    vals = [v for v in vals if isinstance(v, (int, float))]
    if len(recs) < MIN_COVERAGE * planned or not vals:
        return "--$^\\dagger$"
    return f"{100 * np.mean(vals):.1f}"


def static_cell(dataset, model, mode, prefix, key):
    # A dataset run as ONE task kind has no kind in its folder name
    # (io_n-a_static); one with two kinds has (io_n-a_generation_static).
    variants = {"MCS": ["MCS", "MCS_selection"], "SCS": ["SCS", "SCS_selection"],
                "n-a": ["n-a", "n-a_generation"]}.get(prefix, [prefix])
    for v in variants:
        s = score(STATIC_RUN, dataset, model, f"{mode}_{v}_static", key, STATIC_PLANNED)
        if s is not None:
            return s
    return "--"


def episode_cell(dataset, model, tprefix):
    for run in EPISODE_RUNS:
        s = score(run, dataset, model, tprefix, "final_answer_accuracy", EPISODE_PLANNED)
        if s is not None:
            return s
    return "--"


def tt(name: str) -> str:
    return "\\texttt{" + name.replace("_", "\\_") + "}"


def row(first, regime, metric, cells, grouped=False):
    """One table row. `grouped`: the dataset has several regime rows, joined by
    a solid vertical line on the left of the Regime column."""
    reg = f"\\abRegime{{{regime}}}" if grouped else regime
    return f"{first} & {reg} & {metric} & " + " & ".join(cells) + " \\\\"


# Vertical guide lines drawn inside the cells, one row high each (the table's own
# strut height), so consecutive rows join into one continuous line and the
# \addlinespace between datasets breaks it: a solid line groups a dataset's
# regimes, a dotted one groups the two metrics (EM, F1) of its MCS regime.
MACROS = r"""\makeatletter
\providecommand{\abRegime}[1]{\rlap{\hspace{-\tabcolsep}\vrule width 0.5pt height \ht\@arstrutbox depth \dp\@arstrutbox}#1}
\providecommand{\abMetric}[1]{\rlap{\hspace{-\tabcolsep}\lower\dp\@arstrutbox\vbox to \dimexpr\ht\@arstrutbox+\dp\@arstrutbox\relax{\cleaders\vbox to 1.2pt{\vss\hbox{\vrule width 0.55pt height 0.55pt}}\vfill}}#1}
\makeatother"""


def main() -> int:
    out = Path(sys.argv[1])
    L = [MACROS]
    L.append(r"\begin{table*}[t]")
    L.append(r"\centering")
    L.append(r"\scriptsize")
    L.append(r"\renewcommand{\arraystretch}{0.82}")
    L.append(r"\setlength{\tabcolsep}{3pt}")
    L.append(r"\caption{\textbf{Main results by delivery protocol, task format and selection regime.} All scores "
             r"are percentages (\%). \textbf{Static}: 50 records per dataset, each asked three times at "
             r"temperature 0.7, score averaged over all 150 answers; IO = direct answer, CoT = reasoning "
             r"before answering. \textbf{Interactive} and \textbf{Sequential}: 50 records per dataset, one "
             r"episode each; these protocols run in IO only (native reasoning off), so CoT is \texttt{--}; the "
             r"score is final-answer accuracy. Regimes: SCS = single choice, MCS = multiple choice allowed "
             r"(EM = exact set match, F1 = set F1), Gen = free-form generation, Comp.\ = knowledge completion. "
             r"Metrics: ACC = accuracy, LLM-J = LLM-judge (gpt-oss-120b) equivalence with the reference, "
             r"RC = root-cause match, Proxy = judge-scored closest explanation, Set-EM = exact premise-set match, "
             r"JRA = joint root-cause accuracy (fault type and component). \texttt{--}$^\dagger$: run covers "
             r"fewer than 90\% of the planned samples (not reported).}")
    L.append(r"\label{tab:main_results_all_models}")
    L.append(r"\resizebox{\linewidth}{!}{%")
    L.append(r"\begin{tabular}{l l l " + "c" * (2 * len(MODELS)) + "}")
    L.append(r"\toprule")
    L.append(r"\textbf{Benchmark / Dataset} & \textbf{Regime} & \textbf{Metric} & "
             + " & ".join(f"\\multicolumn{{2}}{{c}}{{\\textbf{{{n}}}}}" for _, n in MODELS) + r" \\")
    L.append(" ".join(f"\\cmidrule(lr){{{4 + 2 * i}-{5 + 2 * i}}}" for i in range(len(MODELS))))
    L.append("& & & " + " & ".join(r"\textbf{IO} & \textbf{CoT}" for _ in MODELS) + r" \\")
    ncol = 3 + 2 * len(MODELS)

    def section(title):
        L.append(r"\midrule")
        L.append(f"\\multicolumn{{{ncol}}}{{l}}{{\\textit{{\\textbf{{{title}}}}}}} \\\\")
        L.append(r"\midrule")

    def sub(title):
        L.append(f"\\multicolumn{{{ncol}}}{{l}}{{\\textit{{{title}}}}} \\\\")

    section("1. Static Delivery")
    sub("1a. Selection")
    for ds, regimes in STATIC_SELECTION:
        for i, (reg, prefix, key, lab) in enumerate(regimes):
            cells = []
            for mid, _ in MODELS:
                cells += [static_cell(ds, mid, "io", prefix, key), static_cell(ds, mid, "cot", prefix, key)]
            metric = f"\\abMetric{{{lab}}}" if prefix == "MCS" else lab
            L.append(row(tt(ds) if i == 0 else "", reg, metric, cells, grouped=len(regimes) > 1))
        L.append(r"\addlinespace[1.2pt]")
    sub("1b. Generation")
    for ds, key, lab in STATIC_GENERATION:
        cells = []
        for mid, _ in MODELS:
            cells += [static_cell(ds, mid, "io", "n-a", key), static_cell(ds, mid, "cot", "n-a", key)]
        L.append(row(tt(ds), "Gen", lab, cells))
    L.append(r"\addlinespace[1.2pt]")
    sub("1c. Knowledge completion")
    for ds, key, lab in STATIC_COMPLETION:
        cells = []
        for mid, _ in MODELS:
            cells += [static_cell(ds, mid, "io", "n-a", key), static_cell(ds, mid, "cot", "n-a", key)]
        L.append(row(tt(ds), "Comp.", lab, cells))

    for title, block in (("2. Interactive Delivery", INTERACTIVE), ("3. Sequential Delivery", SEQUENTIAL)):
        section(title)
        for ds, regimes in block:
            for i, (reg, tprefix, lab) in enumerate(regimes):
                cells = []
                for mid, _ in MODELS:
                    cells += [episode_cell(ds, mid, tprefix), "--"]
                L.append(row(tt(ds) if i == 0 else "", reg, lab, cells, grouped=len(regimes) > 1))
            if block is INTERACTIVE:
                L.append(r"\addlinespace[1.2pt]")

    L.append(r"\bottomrule")
    L.append(r"\end{tabular}%")
    L.append(r"}")
    L.append(r"\end{table*}")
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
