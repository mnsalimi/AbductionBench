"""Paper figures for the interactive + sequential datasets (one PDF each).

    PYTHONPATH=src .venv/bin/python tools/episode_figures.py OUT_DIR RUN_DIR [RUN_DIR ...]

Vector PDFs sized for a 5.5 in (ICLR) text width, TrueType fonts embedded, plus
a PNG preview of each. Colours: the validated 8-slot categorical palette, one
fixed colour per model (API models slots 1-4, local models 5-8), every line
also labelled directly so identity never rests on colour alone.

MedQDx alternates a question with an interim diagnosis, and the step-relevance
judge grades both; the step curves use its QUESTION steps only (the interim
diagnoses score far lower and would draw a zig-zag that is about the protocol,
not the model). Step curves show a step only where >= MIN_N episodes reach it.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from abductionbench.core.checkpoint import dedupe_records, load_records  # noqa: E402

# ----------------------------------------------------------------------------- style
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MODELS = [  # (id, label, API?)
    ("gemini-3.8-flash-openrouter", "Gemini 3.8 Flash", True),
    ("gpt-5.6-luna-openrouter", "GPT-5.6 Luna", True),
    ("gemma-4-31b-it-openrouter", "Gemma 4 31B", True),
    ("qwen3-5-27b-openrouter", "Qwen3.5 27B", True),
    ("qwen3-5-4b-local", "Qwen3.5 4B", False),
    ("gemma-4-e4b-local", "Gemma 4 E4B", False),
    ("qwen3-5-2b-local", "Qwen3.5 2B", False),
    ("gemma-4-e2b-local", "Gemma 4 E2B", False),
]
LABEL = {m: n for m, n, _ in MODELS}
ORDER = [n for _, n, _ in MODELS]
COLOR = {n: PALETTE[i] for i, n in enumerate(ORDER)}
API = {n for _, n, api in MODELS if api}
TASKS_INTER = ["DDXPlus (SCS)", "DDXPlus (MCS)", "VivaBench", "Med-Inquire", "MedQDx", "CloudOpsBench"]
TASK_COLOR = {t: PALETTE[i] for i, t in enumerate(TASKS_INTER)}
TICK = {"DDXPlus (SCS)": "DDXPlus\n(SCS)", "DDXPlus (MCS)": "DDXPlus\n(MCS)", "VivaBench": "Viva-\nBench",
        "Med-Inquire": "Med-\nInquire", "MedQDx": "MedQDx", "CloudOpsBench": "CloudOps-\nBench",
        "AthenaBench": "Athena-\nBench", "Mean": "Mean", "All": "All"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FULL, HALF = 5.5, 2.7
MIN_N = 30

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 7.5,
    "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "lines.linewidth": 1.6, "savefig.dpi": 300, "figure.dpi": 150,
})


def task_name(ds: str, tdir: str) -> str:
    if ds == "ddxplus":
        return "DDXPlus (SCS)" if "SCS" in tdir else "DDXPlus (MCS)"
    return {"medqdx": "MedQDx", "med_inquire": "Med-Inquire", "vivabench": "VivaBench",
            "cloud_opsbench": "CloudOpsBench", "athena_bench": "AthenaBench"}[ds]


def wilson(k, n, z=1.96):
    k, n = np.asarray(k, float), np.asarray(n, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = k / n
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def load(runs: list[str]) -> pd.DataFrame:
    rows = []
    for run in runs:
        for path in glob.glob(run + "/datasets/*/*/*/records.jsonl"):
            p = path.split("/")
            ds, model, tdir = p[-4], p[-3], p[-2]
            if model not in LABEL:
                continue
            for r in dedupe_records(load_records(Path(path))):
                m, d = r.get("metrics") or {}, r.get("details") or {}
                steps = d.get("interaction_step_relevance_per_step")
                if ds == "medqdx" and isinstance(steps, list):
                    steps = steps[0::2]          # questions only (see the docstring)
                rows.append(dict(model=LABEL[model], dataset=ds, task=task_name(ds, tdir),
                                 acc=m.get("final_answer_accuracy"), turns=m.get("turns_to_final_output"),
                                 rel=m.get("interaction_step_relevance"), steps=steps,
                                 turn_correct=d.get("turn_correct"), turn_preds=d.get("turn_predictions")))
    df = pd.DataFrame(rows)
    for c in ("acc", "turns", "rel"):
        df[c] = pd.to_numeric(df[c].replace("None", None), errors="coerce")
    return df


def save(fig, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)


def end_labels(ax, items, x_pad=0.25, min_gap=0.035):
    """Direct labels at line ends, nudged apart so none collide."""
    items = sorted(items, key=lambda t: t[1])
    placed = []
    lo, hi = ax.get_ylim()
    gap = min_gap * (hi - lo)
    for x, y, text, color in items:
        if placed and y - placed[-1] < gap:
            y = placed[-1] + gap
        placed.append(y)
        ax.annotate(text, (x, y), xytext=(x + x_pad, y), color=INK, fontsize=6.3, va="center",
                    annotation_clip=False)
        ax.plot([x + 0.05, x + x_pad * 0.85], [y, y], color=color, lw=1.2, clip_on=False)


# ----------------------------------------------------------------------------- figures
def fig_accuracy_heatmap(df, out):
    tasks = TASKS_INTER + ["AthenaBench"]
    piv = df.pivot_table(index="model", columns="task", values="acc", aggfunc="mean").reindex(ORDER)[tasks]
    piv["Mean"] = piv.mean(axis=1)
    fig, ax = plt.subplots(figsize=(FULL, 2.6))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
    im = ax.imshow(piv.values, cmap=cmap, vmin=0, vmax=0.6, aspect="auto")
    ax.set_xticks(range(piv.shape[1]), [TICK[c] for c in piv.columns])
    ax.set_yticks(range(len(ORDER)), ORDER)
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.6,
                    color="white" if v > 0.36 else INK, fontweight="bold" if j == piv.shape[1] - 1 else None)
    ax.axvline(piv.shape[1] - 1.5, color="white", lw=2.5)
    ax.axvline(len(TASKS_INTER) - 0.5, color="white", lw=1.2)
    ax.axhline(3.5, color="white", lw=2.5)
    ax.text(-0.02, 1.02, "Interactive", transform=ax.transAxes, fontsize=7, color=INK2, ha="left")
    ax.text((len(TASKS_INTER) + 0.5) / piv.shape[1], 1.02, "Sequential", transform=ax.transAxes,
            fontsize=7, color=INK2, ha="center")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cb.set_label("Final-answer accuracy", fontsize=7)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6.5, length=2)
    save(fig, out, "fig1_accuracy_heatmap")


def _step_table(df, key, min_n=MIN_N):
    rows = []
    for _, r in df.iterrows():
        if isinstance(r.steps, list):
            for k, v in enumerate(r.steps, 1):
                rows.append((r[key], k, v))
    s = pd.DataFrame(rows, columns=[key, "k", "v"])
    g = s.groupby([key, "k"]).v.agg(["sum", "count"]).reset_index()
    g["mean"] = g["sum"] / g["count"]
    g["lo"], g["hi"] = wilson(g["sum"], g["count"])
    return g[g["count"] >= min_n]


def fig_relevance_by_step_datasets(df, out):
    it = df[df.task.isin(TASKS_INTER)]
    g = _step_table(it, "task", min_n=50)
    fig, ax = plt.subplots(figsize=(FULL, 2.7))
    ends = []
    for t in TASKS_INTER:
        d = g[g.task == t]
        if d.empty:
            continue
        ax.fill_between(d.k, d.lo, d.hi, color=TASK_COLOR[t], alpha=0.08, lw=0)
        ax.plot(d.k, d["mean"], color=TASK_COLOR[t], marker="o", ms=2.6, lw=1.5)
        if t == "MedQDx":
            # Labelled below its line: its end sits on the Med-Inquire curve.
            x, y = d.k.iloc[-1], d["mean"].iloc[-1]
            ax.annotate("MedQDx (questions only)", (x, y), xytext=(x + 0.25, y - 0.045), fontsize=6.3,
                        color=INK, va="center")
            continue
        ends.append((d.k.iloc[-1], d["mean"].iloc[-1], t, TASK_COLOR[t]))
    ax.set_xlabel("Step of the interaction (action index)")
    ax.set_ylabel("Share of steps judged relevant")
    ax.set_ylim(0, 0.52)
    ax.set_xlim(0.6, g.k.max() + 0.4)
    ax.set_xticks(range(1, int(g.k.max()) + 1, 2))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    end_labels(ax, ends, x_pad=0.5)
    ax.text(0.99, 0.97, "all 8 models pooled; 95% Wilson bands; steps reached by $\\geq$50 episodes",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.2, color=INK2)
    save(fig, out, "fig2_relevance_by_step_per_dataset")


def fig_relevance_by_step_models(df, out):
    it = df[df.task.isin(TASKS_INTER)]
    g = _step_table(it, "model")
    fig, axes = plt.subplots(2, 4, figsize=(FULL, 2.9), sharex=True, sharey=True)
    for ax, m in zip(axes.flat, ORDER):
        for other in ORDER:
            d = g[g.model == other]
            ax.plot(d.k, d["mean"], color="#cfcec9", lw=0.8, zorder=1)
        d = g[g.model == m]
        ax.fill_between(d.k, d.lo, d.hi, color=COLOR[m], alpha=0.18, lw=0, zorder=2)
        ax.plot(d.k, d["mean"], color=COLOR[m], lw=1.7, zorder=3)
        ax.set_title(m, fontsize=7.3, pad=3, color=INK)
        ax.set_ylim(0, 0.62)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        first, last = d["mean"].iloc[0], d["mean"].iloc[-1]
        ax.text(0.97, 0.93, f"{first:.0%} → {last:.0%}", transform=ax.transAxes, ha="right", va="top",
                fontsize=6.2, color=INK2)
    for ax in axes[1]:
        ax.set_xlabel("Step")
    for ax in axes[:, 0]:
        ax.set_ylabel("Relevant steps")
    fig.tight_layout(pad=0.3, w_pad=0.5, h_pad=0.6)
    save(fig, out, "fig3_relevance_by_step_per_model")


def fig_relevance_success_gap(df, out):
    it = df[df.task.isin(TASKS_INTER)].dropna(subset=["rel"])
    g = it.groupby(["model", "acc"]).rel.mean().unstack().reindex(ORDER)
    g = g.sort_values(1.0)
    fig, ax = plt.subplots(figsize=(HALF + 0.6, 2.5))
    y = np.arange(len(g))
    for i, (m, row) in enumerate(g.iterrows()):
        ax.plot([row[0.0], row[1.0]], [i, i], color="#cfcec9", lw=2.2, solid_capstyle="round", zorder=1)
        ax.scatter(row[0.0], i, s=26, color="white", edgecolor=COLOR[m], linewidth=1.4, zorder=3)
        ax.scatter(row[1.0], i, s=30, color=COLOR[m], edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(row[1.0] + 0.015, i, f"+{(row[1.0] - row[0.0]) * 100:.0f} pts", va="center",
                fontsize=6.2, color=INK2)
    ax.set_yticks(y, g.index)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0.1, 0.72)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Mean step relevance of the episode")
    ax.grid(axis="y", visible=False)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, label="final answer wrong"),
                       Line2D([], [], marker="o", ls="", mfc=INK2, mec="white", label="final answer correct")],
              loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, frameon=False, handletextpad=0.2)
    save(fig, out, "fig4_relevance_correct_vs_wrong")


def fig_first_question(df, out):
    it = df[df.task.isin(TASKS_INTER)].copy()
    it["first"] = it.steps.apply(lambda s: s[0] if isinstance(s, list) and s else np.nan)
    it = it.dropna(subset=["first", "acc"])
    rows = []
    for t in TASKS_INTER + ["All"]:
        d = it if t == "All" else it[it.task == t]
        for f in (0, 1):
            x = d[d["first"] == f].acc
            lo, hi = wilson(x.sum(), len(x))
            rows.append((t, f, x.mean(), lo, hi, len(x)))
    r = pd.DataFrame(rows, columns=["task", "first", "acc", "lo", "hi", "n"])
    fig, ax = plt.subplots(figsize=(FULL, 2.3))
    names = TASKS_INTER + ["All"]
    x = np.arange(len(names))
    w = 0.36
    for f, col, lab, off in ((0, "#b9b8b2", "first question irrelevant", -w / 2 - 0.01),
                             (1, PALETTE[0], "first question relevant", w / 2 + 0.01)):
        d = r[r["first"] == f].set_index("task").reindex(names)
        ax.bar(x + off, d.acc, w, color=col, label=lab, zorder=2)
        ax.errorbar(x + off, d.acc, yerr=[d.acc - d.lo, d.hi - d.acc], fmt="none", ecolor=INK2,
                    elinewidth=0.7, capsize=1.6, zorder=3)
    for i, t in enumerate(names):
        d = r[r.task == t].set_index("first")
        if d.loc[0, "acc"] > 0:
            ax.text(i, max(d.loc[1, "hi"], d.loc[0, "hi"]) + 0.03, f"×{d.loc[1, 'acc'] / d.loc[0, 'acc']:.1f}",
                    ha="center", fontsize=6.3, color=INK)
    ax.axvline(len(names) - 1.5, color=GRID, lw=1)
    ax.set_xticks(x, [TICK[n] for n in names])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Final-answer accuracy")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, 0.85)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", frameon=False, ncol=2)
    save(fig, out, "fig5_first_question_predicts_success")


def fig_first_question_by_model(df, out):
    """fig5 per model: accuracy when the first question was relevant vs not,
    for every interactive task and all of them together, one panel per model."""
    it = df[df.task.isin(TASKS_INTER)].copy()
    it["first"] = it.steps.apply(lambda s: s[0] if isinstance(s, list) and s else np.nan)
    it = it.dropna(subset=["first", "acc"])
    names = TASKS_INTER + ["All"]
    short = {"DDXPlus (SCS)": "DDX\nSCS", "DDXPlus (MCS)": "DDX\nMCS", "VivaBench": "Viva",
             "Med-Inquire": "Med-\nInq.", "MedQDx": "Med-\nQDx", "CloudOpsBench": "Cloud-\nOps", "All": "All"}
    fig, axes = plt.subplots(4, 2, figsize=(FULL, 6.4), sharey=True)
    w = 0.38
    x = np.arange(len(names))
    for ax, m in zip(axes.flat, ORDER):
        d = it[it.model == m]
        for f, col, off in ((0, "#c4c3bd", -w / 2 - 0.01), (1, COLOR[m], w / 2 + 0.01)):
            accs, los, his, ns = [], [], [], []
            for t in names:
                sub = d if t == "All" else d[d.task == t]
                v = sub[sub["first"] == f].acc
                n = len(v)
                accs.append(v.mean() if n else np.nan)
                lo, hi = wilson(v.sum(), n) if n else (np.nan, np.nan)
                los.append(lo)
                his.append(hi)
                ns.append(n)
            accs, los, his = map(np.array, (accs, los, his))
            ax.bar(x + off, accs, w, color=col, zorder=2)
            ok = ~np.isnan(accs)
            ax.errorbar((x + off)[ok], accs[ok], yerr=[np.clip(accs - los, 0, None)[ok],
                                                      np.clip(his - accs, 0, None)[ok]], fmt="none",
                        ecolor=INK2, elinewidth=0.5, capsize=1.0, zorder=3)
            for xi, n in zip(x + off, ns):
                if n == 0:
                    ax.text(xi, 0.01, "–", ha="center", va="bottom", fontsize=5.5, color=INK2)
        # the ratio over the pooled pair only: per-task cells hold ~50 episodes split in two
        sub = d
        a0 = sub[sub["first"] == 0].acc.mean()
        a1 = sub[sub["first"] == 1].acc.mean()
        ax.text(len(names) - 1, np.nanmax([a0, a1]) + 0.12, f"×{a1 / a0:.1f}" if a0 else "",
                ha="center", fontsize=6.4, color=INK, fontweight="bold")
        ax.axvline(len(names) - 1.5, color=GRID, lw=0.9)
        ax.set_title(m, fontsize=7.4, pad=3)
        ax.set_xticks(x, [short[n] for n in names])
        ax.tick_params(axis="x", length=0, labelsize=5.9)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(axis="x", visible=False)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    for ax in axes[:, 0]:
        ax.set_ylabel("Accuracy", fontsize=7)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color="#c4c3bd", label="first question irrelevant"),
                        Patch(color=INK2, label="first question relevant (model colour)")],
               loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=6.8)
    fig.tight_layout(pad=0.3, w_pad=0.8, h_pad=0.9)
    fig.subplots_adjust(top=0.935)
    save(fig, out, "fig5b_first_question_per_model")


def fig_accuracy_by_length(df, out):
    it = df[df.task.isin(TASKS_INTER) & (df.task != "MedQDx")].dropna(subset=["acc", "turns"]).copy()
    bins = [0, 3, 5, 8, 12, 16, 20]
    labels = ["1–3", "4–5", "6–8", "9–12", "13–16", "17–20"]
    it["bin"] = pd.cut(it.turns, bins, labels=labels)
    fig, ax = plt.subplots(figsize=(HALF + 0.5, 2.4))
    g = it.groupby("bin", observed=False).acc.agg(["sum", "count"])
    lo, hi = wilson(g["sum"], g["count"])
    xs = np.arange(len(labels))
    for t in [t for t in TASKS_INTER if t != "MedQDx"]:
        d = it[it.task == t].groupby("bin", observed=False).acc.agg(["mean", "count"])
        d = d.where(d["count"] >= 15)
        ax.plot(xs, d["mean"], color=TASK_COLOR[t], lw=0.9, alpha=0.55, marker="o", ms=1.8)
    ax.fill_between(xs, lo, hi, color=INK, alpha=0.08, lw=0)
    ax.plot(xs, g["sum"] / g["count"], color=INK, lw=2.0, marker="o", ms=3.2, label="all tasks")
    for i, (a, n) in enumerate(zip(g["sum"] / g["count"], g["count"])):
        ax.text(i, a + 0.035, f"{a:.0%}", ha="center", fontsize=6.2, color=INK)
    ax.set_xticks(xs, labels)
    ax.set_xlabel("Episode length (turns)")
    ax.set_ylabel("Final-answer accuracy")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, 0.7)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=INK, lw=2, marker="o", ms=3, label="all tasks (95% band)")] + [
        Line2D([], [], color=TASK_COLOR[t], lw=0.9, label=t) for t in TASKS_INTER if t != "MedQDx"]
    ax.legend(handles=handles, frameon=False, fontsize=5.8, loc="upper right", ncol=1, handlelength=1.4)
    save(fig, out, "fig6_accuracy_vs_episode_length")


def fig_efficiency(df, out):
    it = df[df.task.isin(TASKS_INTER)]
    g = it.groupby("model").agg(acc=("acc", "mean"), turns=("turns", "mean"), rel=("rel", "mean")).reindex(ORDER)
    fig, ax = plt.subplots(figsize=(HALF + 0.6, 2.5))
    for m, r in g.iterrows():
        ax.scatter(r.turns, r.acc, s=18 + 260 * r.rel, color=COLOR[m], alpha=0.9,
                   marker="o" if m in API else "s", edgecolor="white", linewidth=0.8, zorder=3)
    offsets = {"Gemini 3.8 Flash": (7, 3), "GPT-5.6 Luna": (7, -9), "Gemma 4 31B": (-7, -10),
               "Qwen3.5 27B": (0, 9), "Qwen3.5 4B": (7, -2), "Gemma 4 E4B": (7, -7),
               "Qwen3.5 2B": (-7, 7), "Gemma 4 E2B": (7, 3)}
    for m, r in g.iterrows():
        dx, dy = offsets.get(m, (5, 3))
        ax.annotate(m, (r.turns, r.acc), xytext=(dx, dy), textcoords="offset points", fontsize=6.2,
                    ha="left" if dx > 0 else ("right" if dx < 0 else "center"), color=INK)
    ax.set_xlabel("Mean turns to final answer")
    ax.set_ylabel("Mean final-answer accuracy")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlim(6.5, 15.5)
    ax.set_ylim(0.05, 0.38)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=INK2, label="API model"),
                       Line2D([], [], marker="s", ls="", color=INK2, label="local model"),
                       Line2D([], [], marker="o", ls="", mfc="none", mec=INK2, ms=4, label="size = step relevance")],
              frameon=False, loc="lower left", fontsize=6)
    save(fig, out, "fig7_accuracy_vs_turns")


def fig_athena_by_turn(df, out):
    at = df[df.task == "AthenaBench"]
    fig, ax = plt.subplots(figsize=(HALF + 0.9, 2.5))
    ends = []
    for m in ORDER:
        d = at[at.model == m]
        per = {}
        for tc in d.turn_correct:
            for k, c in enumerate(tc or [], 1):
                per.setdefault(k, []).append(c)
        ks = [k for k in sorted(per) if len(per[k]) >= MIN_N]
        ys = [np.mean(per[k]) for k in ks]
        ax.plot(ks, ys, color=COLOR[m], lw=1.5, marker="o" if m in API else "s", ms=2.6)
        ends.append((ks[-1], ys[-1], m, COLOR[m]))
    ax.set_xlabel("Turn (evidence revealed so far)")
    ax.set_ylabel("Culprit identified correctly")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(-0.02, 0.55)
    ax.set_xticks(range(1, 6))
    ax.set_xlim(0.8, 5.2)
    end_labels(ax, ends, x_pad=0.35, min_gap=0.05)
    save(fig, out, "fig8_athena_accuracy_by_turn")


def fig_athena_revision(df, out):
    at = df[df.task == "AthenaBench"]
    rows = []
    for m in ORDER:
        d = at[at.model == m]
        keep_r = keep_w = w2r = r2w = changed_same = 0
        for tp, tc in zip(d.turn_preds, d.turn_correct):
            tp = [(x or "").strip().lower() for x in (tp or [])]
            tc = tc or []
            changed = any(a != b for a, b in zip(tp, tp[1:]))
            first, last = (tc[0], tc[-1]) if tc else (0, 0)
            if not changed:
                keep_r += first == 1
                keep_w += first == 0
            elif first == 0 and last == 1:
                w2r += 1
            elif first == 1 and last == 0:
                r2w += 1
            else:
                changed_same += 1
        n = len(d)
        rows.append((m, keep_r / n, keep_w / n, changed_same / n, w2r / n, r2w / n))
    r = pd.DataFrame(rows, columns=["model", "never changed · right", "never changed · wrong",
                                    "changed · outcome unchanged", "changed · wrong → right",
                                    "changed · right → wrong"]).set_index("model")
    r = r.loc[r.iloc[:, :2].sum(axis=1).sort_values().index]
    cols = ["#1c5cab", "#9ec5f4", "#d9d8d3", "#1baf7a", "#e34948"]
    fig, ax = plt.subplots(figsize=(FULL, 2.4))
    left = np.zeros(len(r))
    for c, col in zip(r.columns, cols):
        ax.barh(r.index, r[c], left=left, color=col, height=0.68, edgecolor="white", linewidth=0.8, label=c)
        for i, v in enumerate(r[c]):
            if v >= 0.06:
                ax.text(left[i] + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=6,
                        color="white" if col in ("#1c5cab", "#1baf7a", "#e34948") else INK)
        left += r[c].values
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Share of AthenaBench episodes (50 per model)")
    ax.tick_params(axis="y", length=0)
    ax.grid(False)
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.45, 1.25), fontsize=6.2,
              handlelength=1.1, columnspacing=1.0)
    save(fig, out, "fig9_athena_belief_revision")


def main() -> int:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    df = load(sys.argv[2:])
    for f in (fig_accuracy_heatmap, fig_relevance_by_step_datasets, fig_relevance_by_step_models,
              fig_relevance_success_gap, fig_first_question, fig_first_question_by_model,
              fig_accuracy_by_length, fig_efficiency,
              fig_athena_by_turn, fig_athena_revision):
        f(df, out)
        print("wrote", f.__name__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
