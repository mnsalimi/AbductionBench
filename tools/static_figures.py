"""Paper figures for the static (single-prompt) datasets of a run (one PDF each).

    PYTHONPATH=src .venv/bin/python tools/static_figures.py RUN_DIR OUT_DIR

Reads the run's reports/summary.csv (task-level scores) and streams every
records.jsonl once for the per-sample figures (newest line per sample, folded
records dropped). Vector PDFs for a 5.5 in text width, TrueType fonts embedded,
a PNG preview of each.

What is left out, and why -- every figure caption should say it:
  * the datasets disabled for the run (hypobench, musr, scir, xcopa) and the
    non-LLM baseline (jev-openrouter);
  * nothing for low coverage: Qwen3.5 27B's cot pass stopped early on nine
    datasets, and those cells are scored on the samples it completed (7-101 of
    150) -- say so in the caption;
  * per-sample reasoning metrics exist only where the reasoning judge produced
    them (for Qwen3.5 2B / 4B the step list often failed to parse); those
    figures say how many samples they rest on.
Selection datasets are scored in single-choice mode (SCS) unless a figure is
about the selection mode itself.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import orjson  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from episode_figures import (  # noqa: E402  (shared palette, model order, helpers)
    COLOR, FULL, GRID, HALF, INK, INK2, LABEL, ORDER, PALETTE, API, end_labels, save, wilson,
)

DROP_DATASETS = {"hypobench", "musr", "scir", "xcopa"}
DROP_MODELS = {"jev-openrouter"}
MIN_COVERAGE = 0.9
PRETTY = {
    "aer": "AER", "agentrx": "AgentRx", "aiops2025": "AIOps2025", "art": "ART", "climate_fever": "Climate-FEVER",
    "commonwhy": "CommonWhy", "crosstrace": "CrossTrace", "diagnosisarena": "DiagnosisArena", "ecare": "e-CARE",
    "enwn_entailmentbank": "EntailmentBank", "house_md": "House-MD", "hypogen": "HypoGen",
    "llm4biohypogen": "LLM4BioHypoGen", "matter_to_mechanism": "Matter-to-Mech.",
    "medcasereasoning": "MedCaseReasoning", "medr_bench": "MedR-Bench", "neulr": "NeuLR",
    "proof_writer": "ProofWriter", "true_detective": "True Detective", "uncommonsense": "UNcommonsense",
    "uniadilr_hgc": "UniADILR",
}
KIND = {"generation": "gen", "selection": "sel", "multi_selection": "sel", "knowledge_completion": "comp"}


def task_label(ds: str, kind: str, both: set[str]) -> str:
    name = PRETTY.get(ds, ds)
    return f"{name} ({'gen.' if kind == 'generation' else 'sel.'})" if ds in both else name


def load_summary(run: str) -> pd.DataFrame:
    s = pd.read_csv(Path(run) / "reports" / "summary.csv")
    s = s[~s.model_id.isin(DROP_MODELS) & ~s.dataset_id.isin(DROP_DATASETS) & s.model_id.isin(LABEL)].copy()
    s["model"] = s.model_id.map(LABEL)
    s["sel"] = s.selection_mode.fillna("-")
    s["partial"] = s.coverage < MIN_COVERAGE
    kinds = s.groupby("dataset_id").task_kind.nunique()
    both = set(kinds[kinds > 1].index)
    s["task"] = [task_label(d, k, both) for d, k in zip(s.dataset_id, s.task_kind)]
    return s


def load_samples(run: str, summary: pd.DataFrame) -> pd.DataFrame:
    prim = {(r.dataset_id, r.task_kind): r.primary_metric for r in summary.itertuples()}
    rows = []
    for path in sorted(glob.glob(run + "/datasets/*/*/*/records.jsonl")):
        p = path.split("/")
        ds, model = p[-4], p[-3]
        if ds in DROP_DATASETS or model in DROP_MODELS or model not in LABEL:
            continue
        latest = {}
        with open(path, "rb") as fh:
            for line in fh:
                if line.strip():
                    try:
                        r = orjson.loads(line)
                    except orjson.JSONDecodeError:
                        continue
                    latest[(r.get("sample_id"), r.get("prompt_fingerprint"))] = r
        for r in latest.values():
            if (r.get("metadata") or {}).get("reduced"):
                continue
            m = r.get("metrics") or {}
            pm = prim.get((ds, r.get("task_kind")))
            u = (r.get("response") or {}).get("usage") or {}
            row = dict(dataset=ds, model=LABEL[model], prompt_mode=r.get("prompt_mode"),
                       sel=r.get("selection_mode") or "-", task_kind=r.get("task_kind"), status=r.get("status"),
                       correct=m.get(pm) if pm else None,
                       tokens=u.get("completion_tokens") or (r.get("response") or {}).get("completion_tokens_est"))
            row.update({k: v for k, v in m.items() if k.startswith("reasoning_") and not k.endswith("_per_step")})
            rows.append(row)
    df = pd.DataFrame(rows)
    for c in df.columns:
        if c.startswith("reasoning_") or c in ("correct", "tokens"):
            df[c] = pd.to_numeric(df[c].replace("None", None), errors="coerce")
    return df


def rescore_from_records(s: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Each task's score recomputed from its records (mean of the primary metric
    over every scored sample), so the figures never depend on a stale report."""
    scored = df[df.status.isin(["ok", "empty", "truncated"])]
    means = scored.groupby(["dataset", "model", "prompt_mode", "sel", "task_kind"]).correct.mean()
    s = s.copy()
    s["value"] = [means.get((d, m, pmode, sel if sel != "-" else "-", k), np.nan)
                  for d, m, pmode, sel, k in zip(s.dataset_id, s.model, s.prompt_mode, s.sel, s.task_kind)]
    return s


def headline(s: pd.DataFrame) -> pd.DataFrame:
    """One score per (task, model, prompt mode): SCS for selection. A task with
    low coverage (Qwen3.5 27B's cot pass stopped early on some datasets) is
    scored on the samples it did complete -- the caption has to say so."""
    return s[s.sel.isin(["-", "SCS"])].copy()


# ----------------------------------------------------------------------------- figures
def _heat(ax, piv, cmap, vmin, vmax, fmt, dark, partial=None):
    im = ax.imshow(piv.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(piv.shape[1]), piv.columns, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(piv.shape[0]), piv.index)
    ax.tick_params(length=0)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if partial is not None and partial.values[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="////",
                                           edgecolor="#b9b8b2", lw=0))
                ax.text(j, i, "partial", ha="center", va="center", fontsize=4.8, color=INK2)
            elif not np.isnan(v):
                ax.text(j, i, fmt(v), ha="center", va="center", fontsize=5.6,
                        color="white" if dark(v) else INK)
    return im


def fig_accuracy_heatmap(s, out):
    h = headline(s)
    for mode, name in (("io", "figS1a_accuracy_heatmap_io"), ("cot", "figS1b_accuracy_heatmap_cot")):
        d = h[h.prompt_mode == mode]
        piv = d.pivot_table(index="task", columns="model", values="value", aggfunc="first")[ORDER]
        part = d.pivot_table(index="task", columns="model", values="partial", aggfunc="first")[ORDER]
        order = piv.mean(axis=1).sort_values(ascending=False).index
        piv, part = piv.loc[order], part.loc[order].fillna(False).astype(bool)
        fig, ax = plt.subplots(figsize=(FULL, 5.2))
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("b", ["#f4f8fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
        im = _heat(ax, piv, cmap, 0, 1, lambda v: f"{v:.2f}", lambda v: v > 0.6)
        ax.axvline(3.5, color="white", lw=2.5)
        ax.set_title(f"{'Direct answer (io)' if mode == 'io' else 'Chain-of-thought (cot)'} — primary metric per task",
                     fontsize=7.5, loc="left", color=INK)
        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.015)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=6.2, length=2)
        save(fig, out, name)


def fig_cot_gain_heatmap(s, out):
    h = headline(s)
    p = h.pivot_table(index=["task", "model"], columns="prompt_mode", values="value", aggfunc="first")
    gain = (p["cot"] - p["io"]).unstack("model")[ORDER]
    part = h[h.prompt_mode == "cot"].pivot_table(index="task", columns="model", values="partial",
                                                 aggfunc="first")[ORDER].reindex(gain.index).fillna(False).astype(bool)
    order = gain.mean(axis=1).sort_values(ascending=False).index
    gain, part = gain.loc[order], part.loc[order]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", ["#b9302f", "#e34948", "#f3b5b4", "#f0efec",
                                                     "#9ec5f4", "#2a78d6", "#1c5cab"])
    fig, ax = plt.subplots(figsize=(FULL, 5.2))
    im = _heat(ax, gain, cmap, -0.4, 0.4, lambda v: f"{v:+.2f}", lambda v: abs(v) > 0.25)
    ax.axvline(3.5, color="white", lw=2.5)
    ax.set_title("Chain-of-thought minus direct answer (primary metric; blue = CoT helps, red = CoT hurts)",
                 fontsize=7.3, loc="left", color=INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.015)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6.2, length=2)
    save(fig, out, "figS2_cot_gain_heatmap")


def fig_cot_gain_by_model(s, df, out):
    """Mean CoT gain per model (with a 95 % interval over tasks) against its CoT length."""
    h = headline(s)
    p = h.pivot_table(index=["task", "model"], columns="prompt_mode", values="value", aggfunc="first").dropna()
    p["gain"] = p.cot - p.io
    g = p.groupby("model").gain.agg(["mean", "std", "count"]).reindex(ORDER)
    g["ci"] = 1.96 * g["std"] / np.sqrt(g["count"])
    tok = df[df.prompt_mode == "cot"].groupby("model").tokens.median().reindex(ORDER)
    fig, ax = plt.subplots(figsize=(HALF + 0.8, 2.6))
    ax.axhline(0, color=INK2, lw=0.7)
    for m in ORDER:
        ax.errorbar(tok[m], g.loc[m, "mean"], yerr=g.loc[m, "ci"], fmt="none", ecolor=COLOR[m], elinewidth=1,
                    capsize=2, alpha=0.8)
        ax.scatter(tok[m], g.loc[m, "mean"], s=34, color=COLOR[m], marker="o" if m in API else "s",
                   edgecolor="white", linewidth=0.8, zorder=3)
    offs = {"Gemini 3.8 Flash": (6, 5), "GPT-5.6 Luna": (6, 5), "Gemma 4 31B": (6, -9),
            "Qwen3.5 27B": (-5, -10), "Qwen3.5 4B": (6, 3), "Gemma 4 E4B": (6, -8), "Qwen3.5 2B": (6, 5),
            "Gemma 4 E2B": (-6, -10)}
    for m in ORDER:
        dx, dy = offs[m]
        ax.annotate(m, (tok[m], g.loc[m, "mean"]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=5.8, ha="left" if dx > 0 else "right", color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("Median CoT reply length (tokens, log scale)")
    ax.set_ylabel("Mean gain of CoT over io")
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 100:+.0f} pts"))
    ax.set_xlim(150, 40000)
    save(fig, out, "figS3_cot_gain_vs_length")


def fig_overthinking(df, out):
    """Accuracy by reasoning-length quintile, within each (dataset, model)."""
    c = df[(df.prompt_mode == "cot") & df.correct.isin([0, 1]) & df.tokens.notna()].copy()
    c["q"] = c.groupby(["dataset", "model", "sel"]).tokens.transform(
        lambda x: pd.qcut(x.rank(method="first"), 5, labels=False))
    fig, ax = plt.subplots(figsize=(FULL * 0.62, 2.8))
    ends = []
    for m in ORDER:
        g = c[c.model == m].groupby("q").correct.agg(["sum", "count"])
        y = g["sum"] / g["count"]
        lo, hi = wilson(g["sum"], g["count"])
        ax.fill_between(g.index + 1, lo, hi, color=COLOR[m], alpha=0.10, lw=0)
        ax.plot(g.index + 1, y, color=COLOR[m], marker="o" if m in API else "s", ms=2.8, lw=1.5)
        ends.append((5, y.iloc[-1], m, COLOR[m]))
    allg = c.groupby("q").correct.mean()
    ax.plot(allg.index + 1, allg, color=INK, lw=2.4, ls=(0, (1, 1)), zorder=5)
    ax.text(0.03, 0.04, f"dotted: all models, {allg.iloc[0]:.0%} → {allg.iloc[-1]:.0%}",
            transform=ax.transAxes, fontsize=6.2, color=INK)
    ax.set_xticks(range(1, 6), ["1\nshortest", "2", "3", "4", "5\nlongest"])
    ax.set_xlabel("Reasoning-length quintile (within dataset and model)")
    ax.set_ylabel("Accuracy (primary metric)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlim(0.8, 5.2)
    ax.set_ylim(0, 0.95)
    end_labels(ax, ends, x_pad=0.3, min_gap=0.04)
    save(fig, out, "figS4_accuracy_vs_reasoning_length")


def fig_process_markers(df, out):
    """Standardized difference (correct - wrong) of each reasoning measure, within (dataset, model)."""
    c = df[(df.prompt_mode == "cot") & df.correct.isin([0, 1])]
    free = [("reasoning_comparison_exhaustiveness", "Comparison exhaustiveness"),
            ("reasoning_useful_step_fraction", "Useful-step fraction"),
            ("reasoning_directionality", "Directionality toward an answer"),
            ("reasoning_mention_ratio", "Observation mention ratio"),
            ("reasoning_observation_coverage", "Observation coverage"),
            ("reasoning_differential_elimination_normalized", "Differential elimination"),
            ("reasoning_unresolved_contradiction_normalized", "Unresolved contradictions"),
            ("reasoning_total_steps", "Number of steps"),
            ("reasoning_prior_knowledge_normalized", "Prior-knowledge use"),
            ("reasoning_backtracking_rate", "Backtracking rate"),
            ("reasoning_anchoring_point_normalized", "Anchoring point (later = higher)"),
            ("reasoning_uncertainty_rate", "Uncertainty rate"),
            ("reasoning_branchiness_total", "Branchiness"),
            ("tokens", "Reply length (tokens)")]
    gold = [("reasoning_helpful_fraction", "Helpful-step fraction"),
            ("reasoning_gold_alive_rate", "Gold hypothesis kept alive"),
            ("reasoning_harmful_fraction", "Harmful-step fraction")]
    rows = []
    for group, items in (("gold-free", free), ("gold-referenced", gold)):
        for col, label in items:
            ds = []
            for _, g in c.groupby(["dataset", "model", "sel"]):
                g = g.dropna(subset=[col])
                a, b = g[g.correct == 1][col], g[g.correct == 0][col]
                sd = g[col].std()
                if len(a) >= 10 and len(b) >= 10 and sd > 0:
                    ds.append((a.mean() - b.mean()) / sd)
            ds = np.array(ds)
            rows.append((group, label, ds.mean(), 1.96 * ds.std() / np.sqrt(len(ds)), len(ds)))
    r = pd.DataFrame(rows, columns=["group", "label", "d", "ci", "cells"])
    r = pd.concat([r[r.group == "gold-free"].sort_values("d"), r[r.group == "gold-referenced"].sort_values("d")])
    fig, ax = plt.subplots(figsize=(FULL * 0.72, 3.4))
    y = np.arange(len(r))[::-1]
    for yi, (_, row) in zip(y, r.iterrows()):
        col = PALETTE[0] if row.d > 0 else PALETTE[7]
        ax.errorbar(row.d, yi, xerr=row.ci, fmt="o", color=col, ms=3.6, elinewidth=1.1, capsize=1.8,
                    mfc=col if row.group == "gold-free" else "white")
    ax.axvline(0, color=INK2, lw=0.8)
    n_free = int((r.group == "gold-free").sum())
    split = y[n_free - 1] - 0.5
    ax.axhline(split, color=GRID, lw=1)
    ax.text(1.01, y[0], "gold-free\nmeasures", transform=ax.get_yaxis_transform(), fontsize=6.2,
            color=INK2, va="top")
    ax.text(1.01, y[n_free], "judged against\nthe gold answer", transform=ax.get_yaxis_transform(),
            fontsize=6.2, color=INK2, va="top")
    ax.set_yticks(y, r.label)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Standardized difference, correct − wrong answers (Cohen's d)")
    ax.set_xlim(-1.6, 1.6)
    ax.text(0.01, 0.01, "← more in wrong answers", transform=ax.transAxes, fontsize=6, color=PALETTE[7])
    ax.text(0.99, 0.01, "more in correct answers →", transform=ax.transAxes, fontsize=6, color=PALETTE[0],
            ha="right")
    save(fig, out, "figS5_reasoning_markers_of_success")


def fig_selection_mode(s, out):
    q = s[s.sel.isin(["SCS", "MCS"]) & ~s.partial]
    p = q.pivot_table(index=["dataset_id", "model", "prompt_mode"], columns="sel", values="value").dropna()
    g = p.groupby("model")[["SCS", "MCS"]].mean().reindex(ORDER)
    g = g.loc[(g.SCS - g.MCS).sort_values().index]
    fig, ax = plt.subplots(figsize=(HALF + 0.7, 2.5))
    for i, (m, row) in enumerate(g.iterrows()):
        ax.plot([row.MCS, row.SCS], [i, i], color="#cfcec9", lw=2.2, solid_capstyle="round", zorder=1)
        ax.scatter(row.SCS, i, s=30, color=COLOR[m], edgecolor="white", linewidth=0.8, zorder=3)
        ax.scatter(row.MCS, i, s=26, color="white", edgecolor=COLOR[m], linewidth=1.4, zorder=3)
        ax.text(row.SCS + 0.015, i, f"−{(row.SCS - row.MCS) * 100:.0f} pts", va="center", fontsize=6.2,
                color=INK2)
    ax.set_yticks(range(len(g)), g.index)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Mean accuracy on the selection datasets")
    ax.set_xlim(0.1, 0.85)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc=INK2, mec="white", label="single choice (SCS)"),
                       Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, label="multiple choice allowed (MCS)")],
              loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, frameon=False, handletextpad=0.2, fontsize=6.2)
    save(fig, out, "figS6_single_vs_multiple_choice")


def main() -> int:
    run, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    s = load_summary(run)
    df = load_samples(run, s)
    df["sel"] = df["sel"].replace({"n/a": "-"})
    s = rescore_from_records(s, df)
    for f, args in ((fig_accuracy_heatmap, (s,)), (fig_cot_gain_heatmap, (s,)), (fig_cot_gain_by_model, (s, df)),
                    (fig_overthinking, (df,)), (fig_process_markers, (df,)), (fig_selection_mode, (s,))):
        f(*args, out)
        print("wrote", f.__name__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
