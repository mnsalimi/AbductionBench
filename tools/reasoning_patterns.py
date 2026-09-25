"""Find surprising and suspicious samples in a run's reasoning metrics.

READ-ONLY on the run: it reads records.jsonl, metrics.json and
reasoning_metrics.jsonl and writes only into runs/<run>/analysis/reasoning_patterns/.
Metric definitions: docs/reasoning_metrics.md.

Reasoning metrics come from the LATEST judge round per sample
(reasoning_metrics.jsonl), not from records.jsonl -- a record can still carry a
value an earlier round left behind (the per-family replacement bug), and that
value would make a sample look surprising for the wrong reason. Where a sample
has no row in the log, the record's own values are used and the row says so
(`metrics_source = records`). Correctness is the task's primary metric on the
record: 1 = correct, 0 = wrong, anything between = partial.

Only models whose cot samples carry reasoning metrics are analysed; the rest
are listed and skipped.

    .venv/bin/python tools/reasoning_patterns.py runs/<run>

Outputs (runs/<run>/analysis/reasoning_patterns/):
  summary.md                 every filter, why it is surprising, counts, examples
  interesting.csv            one row per (filter, sample): manuscript candidates
  problems.csv               one row per (filter, sample): likely judge/pipeline bugs
  metric_vs_correctness.csv  per model x metric: mean when right / wrong, AUC
  repeat_divergence.csv      same question, same model, repeats disagree
  cross_model.csv            questions one model always solves and the others never
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from abductionbench.core.checkpoint import dedupe_records, load_records  # noqa: E402
from abductionbench.core.reasoning_judge import REASONING_METRIC_COLUMNS  # noqa: E402

ANSWERED = {"ok", "empty", "truncated"}
REPEAT = re.compile(r"#r\d+$")
NAN = float("nan")


def num(value):
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return NAN


def isnan(x):
    return isinstance(x, float) and math.isnan(x)


# ---------------------------------------------------------------- loading ---
def load(run_dir: Path) -> tuple[pd.DataFrame, dict]:
    latest: dict[tuple, dict] = {}
    log = run_dir / "reasoning_metrics.jsonl"
    if log.exists():
        for line in log.open():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["dataset_id"], row["model_id"], row.get("selection_mode"),
                   row.get("task_kind"), row["sample_id"])
            latest[key] = row
    info = {"log_rows": len(latest), "models_skipped": Counter(), "stale": Counter(),
            "errors": Counter(), "records_without_metrics": Counter()}
    rows = []
    for task_dir in sorted(run_dir.glob("datasets/*/*/cot*")):
        model, dataset = task_dir.parent.name, task_dir.parent.parent.name
        mfile = task_dir / "metrics.json"
        primary = (json.loads(mfile.read_text()).get("primary_metric") if mfile.exists() else None) or ""
        for rec in dedupe_records(load_records(task_dir / "records.jsonl")):
            if rec.get("status") not in ANSWERED:
                continue
            key = (dataset, model, rec.get("selection_mode"), rec.get("task_kind"), rec["sample_id"])
            logged = latest.get(key)
            rec_metrics = rec.get("metrics") or {}
            rec_rm = {k: v for k, v in rec_metrics.items() if k in REASONING_METRIC_COLUMNS}
            if logged is None and not rec_rm:
                info["records_without_metrics"][model] += 1
                continue
            rm = (logged or {}).get("metrics") or rec_rm
            lists = (logged or {}).get("lists") or ((rec.get("details") or {}).get("reasoning_lists") or {})
            if logged:
                for k, v in rm.items():
                    if k in rec_rm and not (num(v) == num(rec_rm[k]) or (isnan(num(v)) and isnan(num(rec_rm[k])))):
                        info["stale"][model] += 1
                        break
                for e in logged.get("errors") or []:
                    info["errors"][e.split(":")[0] + ":" + e.split(":")[1] if ":" in e else e] += 1
            value = num(rec_metrics.get(primary))
            correct = "unknown" if isnan(value) else ("correct" if value >= 0.999 else ("wrong" if value <= 0.001 else "partial"))
            response = rec.get("response") or {}
            row = {
                "dataset": dataset, "model": model, "task": task_dir.name.split("@")[0],
                "sample_id": rec["sample_id"], "question": REPEAT.sub("", rec["sample_id"]),
                "task_kind": rec.get("task_kind"), "primary_metric": primary, "primary_value": value,
                "correct": correct, "metrics_source": "log" if logged else "records",
                "reasoning_source": (logged or {}).get("reasoning_source") or (rec.get("details") or {}).get("reasoning_source"),
                "judge_status": (logged or {}).get("status"),
                "judge_errors": ";".join((logged or {}).get("errors") or []),
                "option_count": (logged or {}).get("option_count"),
                "anchoring_none": str(rm.get("reasoning_anchoring_point")) == "None",
                "trace_chars": len(response.get("content") or "") + len(response.get("reasoning") or ""),
                "lists": lists,
            }
            for column in REASONING_METRIC_COLUMNS:
                row[column] = num(rm.get(column))
            rows.append(row)
    df = pd.DataFrame(rows)
    return df, info


# ---------------------------------------------------------------- filters ---
def g(r, name):
    return r.get("reasoning_" + name, NAN)


def L(r, name):
    value = (r.get("lists") or {}).get("reasoning_" + name)
    return value if isinstance(value, list) else None


def n_steps(r):
    return g(r, "total_steps")


def first_one(values):
    for i, v in enumerate(values or [], start=1):
        if v == 1:
            return i
    return None


def right(r):
    return r["correct"] == "correct"


def wrong(r):
    return r["correct"] == "wrong"


def sel(r):
    return r.get("task_kind") in ("selection", "multi_selection")


INTERESTING = [
    ("all_helpful_but_wrong",
     "Every step (≥3) was judged to move toward the reference answer, yet the final answer is wrong: "
     "step-level helpfulness does not add up to a correct conclusion -- the failure is in the last leap.",
     lambda r: g(r, "helpful_fraction") == 1 and n_steps(r) >= 3 and wrong(r)),
    ("mostly_harmful_but_correct",
     "At least half the steps were judged to push AWAY from the reference answer, yet the answer is right: "
     "right for the wrong reasons, or a conclusion unrelated to its own chain.",
     lambda r: g(r, "harmful_fraction") >= 0.5 and right(r)),
    ("gold_in_play_most_steps_but_wrong",
     "The correct answer was in play in at least half the steps, and the model still chose something else: "
     "it had the right candidate and talked itself out of it.",
     lambda r: g(r, "gold_alive_rate") >= 0.5 and n_steps(r) >= 3 and wrong(r)),
    ("gold_dropped_then_wrong",
     "The correct answer was considered early but never again after the first half of the chain, and the "
     "answer is wrong: a considered-and-abandoned hypothesis, visible step by step.",
     lambda r: (lambda a: a is not None and len(a) >= 4 and 1 in a[: len(a) // 2]
                and 1 not in a[len(a) // 2:] and wrong(r))(L(r, "gold_alive_per_step"))),
    ("answer_appears_only_at_last_step_correct",
     "The model's own final answer first enters consideration at the LAST step of a ≥4-step chain, and it is "
     "correct: the conclusion was not built up by the chain before it.",
     lambda r: g(r, "anchoring_point_normalized") == 1 and n_steps(r) >= 4 and right(r)),
    ("guess_first_anchored_at_step1_wrong",
     "Whole chain judged guess-first (directionality 0), the eventual answer present from step 1, and wrong: "
     "classic confirmation of an initial guess.",
     lambda r: g(r, "directionality") == 0 and g(r, "anchoring_point") == 1 and n_steps(r) >= 3 and wrong(r)),
    ("evidence_first_full_coverage_but_wrong",
     "Reasoned from the evidence (directionality 1) and used EVERY observation, yet wrong: textbook process, "
     "wrong outcome -- the error is in inference, not in attention.",
     lambda r: g(r, "directionality") == 1 and g(r, "observation_coverage") == 1 and wrong(r)),
    ("correct_without_using_any_observation",
     "Correct while drawing on none of the question's observations: answered from priors or pattern-matching.",
     lambda r: g(r, "observations_used") == 0 and right(r)),
    ("world_knowledge_not_evidence_correct",
     "Background knowledge in ≥80% of steps while using ≤25% of the observations, and correct: the question "
     "was solved by recall, not abduction from what it states.",
     lambda r: g(r, "prior_knowledge_normalized") >= 0.8 and g(r, "observation_coverage") <= 0.25 and right(r)),
    ("incoherent_but_correct",
     "Two or more contradictions left standing, yet the answer is right.",
     lambda r: g(r, "unresolved_contradictions") >= 2 and right(r)),
    ("confidently_wrong",
     "A chain of ≥5 steps with not one uncertainty marker, and wrong.",
     lambda r: g(r, "uncertainty_steps") == 0 and n_steps(r) >= 5 and wrong(r)),
    ("hedges_everywhere_but_correct",
     "Uncertainty marked in ≥80% of steps, yet correct: hedging is not a sign of being wrong here.",
     lambda r: g(r, "uncertainty_rate") >= 0.8 and n_steps(r) >= 3 and right(r)),
    ("recovered_after_harm",
     "At least one harmful step AND a self-correction, ending correct: a visible recovery.",
     lambda r: g(r, "harmful_steps") >= 1 and g(r, "backtracking_steps") >= 1 and right(r)),
    ("exhaustive_comparison_but_wrong",
     "Compared at least as many pairs as the options allow, and still wrong: thorough elimination is no guarantee.",
     lambda r: sel(r) and g(r, "comparison_exhaustiveness") >= 1 and wrong(r)),
    ("tunnel_vision",
     "A selection question with ≥4 options where the chain raised at most one candidate: no alternatives considered "
     "(listed with the outcome -- the correct ones are lucky, the wrong ones are the cost).",
     lambda r: sel(r) and (r.get("option_count") or 0) >= 4 and g(r, "branchiness_total") <= 1),
    ("rumination",
     "Each hypothesis revisited ≥5 times on average (mention ratio): circling, not progressing.",
     lambda r: g(r, "mention_ratio") >= 5),
    ("late_derailment",
     "Helpful steps first and a HARMFUL final step, ending wrong: on the right track until the end.",
     lambda r: (lambda h: h is not None and len(h) >= 3 and h[-1] == -1 and 1 in h[:-1] and wrong(r))(L(r, "helpfulness_per_step"))),
    ("filler_but_correct",
     "≤25% of ≥4 steps prove or disprove anything, yet correct: most of the chain is not doing work.",
     lambda r: g(r, "useful_step_fraction") <= 0.25 and n_steps(r) >= 4 and right(r)),
]


def _overthinking(df: pd.DataFrame) -> set[int]:
    """Longest 5% of chains within (dataset, model), wrong."""
    out = set()
    for _key, part in df.groupby(["dataset", "model"]):
        steps = part["reasoning_total_steps"].dropna()
        if len(steps) < 20:
            continue
        cut = steps.quantile(0.95)
        out.update(part.index[(part["reasoning_total_steps"] >= cut) & (part["correct"] == "wrong")])
    return out


PROBLEMS = [
    ("coverage_used_exceeds_inventory",
     "The coverage judge placed more observations than the inventory holds (the metric was blanked; the sample "
     "shows where the two judges disagree).",
     lambda r: "used_exceeds_the_inventory" in (r.get("judge_errors") or "")
     or (lambda c: c is not None and not isnan(g(r, "observations_total")) and sum(c) > g(r, "observations_total"))(L(r, "observations_per_step"))),
    ("list_length_mismatch",
     "A stored per-step list does not have one value per step -- the invariant every metric rests on.",
     lambda r: not isnan(n_steps(r)) and any(isinstance(v, list) and len(v) != int(n_steps(r))
            for k, v in (r.get("lists") or {}).items() if k.endswith("_per_step"))),
    ("anchoring_out_of_range",
     "Anchoring index below 1 or above the step count -- a 0-based value from the old prompt, or a bad index.",
     lambda r: not isnan(g(r, "anchoring_point")) and (g(r, "anchoring_point") < 1 or g(r, "anchoring_point") > n_steps(r))),
    ("correct_but_gold_never_in_play",
     "The final answer is correct, but the judge says the correct answer was in play in NO step: the correct "
     "answer can only have appeared after the chain -- or the gold-alive judge missed it.",
     lambda r: right(r) and g(r, "gold_alive_steps") == 0 and n_steps(r) >= 2),
    ("anchoring_contradicts_gold_alive",
     "Correct answer, so the model's final answer IS the gold. The same judge call says gold was in play at step k "
     "but that the model's own answer first appeared later (or never). One of its two outputs is wrong.",
     lambda r: right(r) and sel(r) and (lambda k: k is not None and (r["anchoring_none"] or
            (not isnan(g(r, "anchoring_point")) and g(r, "anchoring_point") > k)))(first_one(L(r, "gold_alive_per_step")))),
    ("counts_do_not_sum",
     "helpful + neutral + harmful, or useful + useless, does not equal the step count.",
     lambda r: not isnan(n_steps(r)) and (
         (not isnan(g(r, "helpful_steps")) and g(r, "helpful_steps") + g(r, "neutral_steps") + g(r, "harmful_steps") != n_steps(r))
         or (not isnan(g(r, "useful_steps")) and g(r, "useful_steps") + g(r, "useless_steps") != n_steps(r)))),
    ("mentions_below_branchiness",
     "Fewer references to hypotheses than distinct hypotheses -- impossible, since every new one is a mention.",
     lambda r: not isnan(g(r, "mentions_total")) and not isnan(g(r, "branchiness_total")) and g(r, "mentions_total") < g(r, "branchiness_total")),
    ("more_candidates_than_options",
     "A selection chain raising more DISTINCT candidates than the question has options: the branchiness judge "
     "counted non-options (or split one option into several).",
     lambda r: sel(r) and (r.get("option_count") or 0) >= 2 and g(r, "branchiness_total") > (r.get("option_count") or 0)),
    ("directionality_judges_disagree",
     "Whole-chain directionality and the mean of the per-step directionality differ by ≥0.9: the two judges read "
     "the same chain in opposite directions.",
     lambda r: not isnan(g(r, "directionality")) and not isnan(g(r, "step_directionality_mean"))
     and abs(g(r, "directionality") - g(r, "step_directionality_mean")) >= 0.9 and n_steps(r) >= 3),
    ("helpful_but_proves_nothing",
     "Every step judged HELPFUL, yet not one step proves or disproves anything: two judges disagree about "
     "whether the steps do work.",
     lambda r: g(r, "helpful_fraction") == 1 and g(r, "useful_step_fraction") == 0 and n_steps(r) >= 3),
    ("under_segmented",
     "A long trace cut into a single step (or >4,000 characters per step): the segmentation did not segment.",
     lambda r: not isnan(n_steps(r)) and r["trace_chars"] > 3000 and (n_steps(r) == 1 or r["trace_chars"] / n_steps(r) > 4000)),
    ("over_segmented",
     "80 or more steps: likely one step per sentence, which inflates every per-step count.",
     lambda r: n_steps(r) >= 80),
    ("metrics_from_records_only",
     "No row in the latest-round log; the values come from records.jsonl and may include leftovers from an "
     "earlier round.",
     lambda r: r["metrics_source"] == "records"),
]


def run_filters(df: pd.DataFrame, filters, extra: dict[str, tuple[str, set]] | None = None):
    hits, why = [], {}
    records = df.to_dict("index")
    for name, reason, fn in filters:
        why[name] = reason
        for idx, r in records.items():
            try:
                if fn(r):
                    hits.append((name, idx))
            except Exception:  # noqa: BLE001 - a filter must not stop the pass
                continue
    for name, (reason, idxs) in (extra or {}).items():
        why[name] = reason
        hits.extend((name, i) for i in idxs)
    return hits, why


KEY_COLS = ["dataset", "task", "model", "sample_id", "correct", "primary_metric", "primary_value",
            "reasoning_total_steps", "reasoning_helpful_fraction", "reasoning_harmful_fraction",
            "reasoning_gold_alive_rate", "reasoning_anchoring_point", "reasoning_directionality",
            "reasoning_observation_coverage", "reasoning_uncertainty_steps",
            "reasoning_unresolved_contradictions", "reasoning_backtracking_steps",
            "reasoning_branchiness_total", "reasoning_useful_step_fraction", "metrics_source",
            "reasoning_source", "judge_errors"]


def hits_frame(df, hits):
    rows = []
    for name, idx in hits:
        r = df.loc[idx]
        steps = (r["lists"] or {}).get("reasoning_steps") or []
        rows.append({"filter": name, **{c: r.get(c) for c in KEY_COLS},
                     "first_steps": " | ".join(str(s)[:160] for s in steps[:3])})
    return pd.DataFrame(rows)


# --------------------------------------------------------- other analyses ---
def auc(pos, neg):
    """P(metric of a correct sample > metric of a wrong one); ties count half."""
    pos, neg = [x for x in pos if not isnan(x)], [x for x in neg if not isnan(x)]
    if len(pos) < 20 or len(neg) < 20:
        return NAN
    ranks = pd.Series(pos + neg).rank()
    r_pos = ranks[: len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def metric_vs_correctness(df):
    rows = []
    cols = [c for c in REASONING_METRIC_COLUMNS if df[c].notna().sum() > 50]
    for model, part in df.groupby("model"):
        good, bad = part[part.correct == "correct"], part[part.correct == "wrong"]
        for c in cols:
            a = auc(list(good[c]), list(bad[c]))
            rows.append({"model": model, "metric": c, "mean_correct": good[c].mean(),
                         "mean_wrong": bad[c].mean(), "auc_correct_higher": a,
                         "n_correct": int(good[c].notna().sum()), "n_wrong": int(bad[c].notna().sum())})
    out = pd.DataFrame(rows)
    if not out.empty:
        piv = out.pivot(index="metric", columns="model", values="auc_correct_higher")
        flips = [m for m, row in piv.iterrows()
                 if row.notna().sum() >= 2 and (row > 0.58).any() and (row < 0.42).any()]
        out["direction_flips_across_models"] = out.metric.isin(flips)
    return out


def repeat_divergence(df):
    rows = []
    for (ds, task, model, q), part in df.groupby(["dataset", "task", "model", "question"]):
        kinds = set(part.correct)
        if len(part) >= 2 and "correct" in kinds and "wrong" in kinds:
            ok, bad = part[part.correct == "correct"], part[part.correct == "wrong"]
            rows.append({"dataset": ds, "task": task, "model": model, "question": q,
                         "repeats": len(part), "correct_repeats": ";".join(ok.sample_id),
                         "wrong_repeats": ";".join(bad.sample_id),
                         "steps_correct": ok.reasoning_total_steps.mean(), "steps_wrong": bad.reasoning_total_steps.mean(),
                         "helpful_correct": ok.reasoning_helpful_fraction.mean(), "helpful_wrong": bad.reasoning_helpful_fraction.mean(),
                         "gold_alive_correct": ok.reasoning_gold_alive_rate.mean(), "gold_alive_wrong": bad.reasoning_gold_alive_rate.mean()})
    return pd.DataFrame(rows)


def cross_model(df):
    rows = []
    score = df[df.correct.isin(["correct", "wrong"])].assign(ok=lambda d: (d.correct == "correct").astype(float))
    for (ds, task, q), part in score.groupby(["dataset", "task", "question"]):
        by_model = part.groupby("model").ok.mean()
        if len(by_model) < 3:
            continue
        solvers = [m for m, v in by_model.items() if v == 1.0]
        failers = [m for m, v in by_model.items() if v == 0.0]
        if len(solvers) == 1 and len(failers) == len(by_model) - 1:
            rows.append({"dataset": ds, "task": task, "question": q, "only_solver": solvers[0],
                         "always_wrong": ";".join(failers)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- report ----
def main() -> int:
    run_dir = Path(sys.argv[1]).resolve()
    out = run_dir / "analysis" / "reasoning_patterns"
    out.mkdir(parents=True, exist_ok=True)
    df, info = load(run_dir)
    if df.empty:
        print("no samples with reasoning metrics")
        return 1
    models = sorted(df.model.unique())
    all_models = sorted({p.parent.name for p in run_dir.glob("datasets/*/*/cot*")})
    skipped = [m for m in all_models if m not in models]

    extra = {"overthinking_wrong": ("Among the longest 5% of chains for its dataset and model, and wrong: more "
                                    "reasoning bought nothing.", _overthinking(df))}
    ihits, iwhy = run_filters(df, INTERESTING, extra)
    phits, pwhy = run_filters(df, PROBLEMS)
    idf, pdf = hits_frame(df, ihits), hits_frame(df, phits)
    idf.to_csv(out / "interesting.csv", index=False)
    pdf.to_csv(out / "problems.csv", index=False)
    mvc = metric_vs_correctness(df)
    mvc.to_csv(out / "metric_vs_correctness.csv", index=False)
    rep = repeat_divergence(df)
    rep.to_csv(out / "repeat_divergence.csv", index=False)
    cross = cross_model(df)
    cross.to_csv(out / "cross_model.csv", index=False)

    lines = [f"# Reasoning-metric patterns -- {run_dir.name}", "",
             f"Samples analysed: **{len(df)}** cot samples with reasoning metrics, models: {', '.join(models)}.",
             f"Skipped (no reasoning metrics yet): {', '.join(skipped) or 'none'}.",
             f"Metric source: latest judge round for {int((df.metrics_source == 'log').sum())}, "
             f"records.jsonl for {int((df.metrics_source == 'records').sum())}.",
             f"Correctness: {dict(Counter(df.correct))}.",
             f"Records whose stored reasoning metrics differ from the latest round (stale leftovers): {dict(info['stale'])}.",
             ""]

    def section(title, hits_df, why):
        lines.extend([f"## {title}", "", "| filter | samples | by model | why it is unexpected |", "|---|---|---|---|"])
        counts = hits_df.groupby("filter").size() if not hits_df.empty else pd.Series(dtype=int)
        for name in why:
            n = int(counts.get(name, 0))
            bym = hits_df[hits_df["filter"] == name].groupby("model").size().to_dict() if n else {}
            short = {m.split("-openrouter")[0]: v for m, v in bym.items()}
            lines.append(f"| `{name}` | {n} | {short or '-'} | {why[name]} |")
        lines.append("")
        for name in why:
            sub = hits_df[hits_df["filter"] == name] if not hits_df.empty else hits_df
            if sub.empty:
                continue
            lines.append(f"**{name}** -- examples:")
            for _, r in sub.head(3).iterrows():
                lines.append(f"- `{r['dataset']}/{r['task']}` {r['model']} `{r['sample_id']}` ({r['correct']}; "
                             f"{int(r['reasoning_total_steps']) if not pd.isna(r['reasoning_total_steps']) else '?'} steps): "
                             f"{str(r['first_steps'])[:220]}")
            lines.append("")

    section("Interesting samples (manuscript candidates)", idf, iwhy)
    section("Problem samples (likely judge or pipeline bugs)", pdf, pwhy)

    lines += ["## Judge errors in the latest round", ""]
    lines += [f"- `{k}`: {v}" for k, v in info["errors"].most_common(12)] + [""]
    if not mvc.empty:
        lines += ["## Which metrics separate right from wrong answers",
                  "", "AUC = probability that a correct sample scores higher than a wrong one (0.5 = no signal).", ""]
        piv = mvc.pivot(index="metric", columns="model", values="auc_correct_higher").round(2)
        piv["spread"] = (piv.max(axis=1) - piv.min(axis=1)).round(2)
        top = piv.assign(strength=(piv.drop(columns="spread") - 0.5).abs().max(axis=1)).sort_values("strength", ascending=False)
        try:
            lines.append(top.drop(columns="strength").head(15).to_markdown())
        except ImportError:
            lines.append("```\n" + top.drop(columns="strength").head(15).to_string() + "\n```")
        flips = sorted(set(mvc[mvc.direction_flips_across_models].metric))
        lines += ["", f"Metrics whose direction FLIPS between models (AUC > 0.58 for one, < 0.42 for another): "
                  f"{', '.join(flips) or 'none'}.", ""]
    lines += ["## Same question, same model, repeats disagree", "",
              f"{len(rep)} (question, model, task) groups have at least one correct and one wrong repeat "
              f"(file `repeat_divergence.csv`)."]
    if not rep.empty:
        lines.append(f"Across them, mean steps correct vs wrong: {rep.steps_correct.mean():.1f} vs {rep.steps_wrong.mean():.1f}; "
                     f"helpful fraction {rep.helpful_correct.mean():.2f} vs {rep.helpful_wrong.mean():.2f}; "
                     f"gold-alive rate {rep.gold_alive_correct.mean():.2f} vs {rep.gold_alive_wrong.mean():.2f}.")
    lines += ["", "## Questions exactly one model solves", "",
              f"{len(cross)} questions where one model is right in every repeat and every other model wrong in every "
              f"repeat (file `cross_model.csv`)."]
    if not cross.empty:
        lines.append(f"Only solver: {dict(Counter(cross.only_solver))}.")
    by_source = df.groupby(["model", "reasoning_source"]).agg(
        samples=("sample_id", "size"), primary_mean=("primary_value", "mean"),
        steps_median=("reasoning_total_steps", "median"),
        helpful_fraction=("reasoning_helpful_fraction", "mean"),
        observation_coverage=("reasoning_observation_coverage", "mean"),
        uncertainty_rate=("reasoning_uncertainty_rate", "mean"),
        gold_alive_rate=("reasoning_gold_alive_rate", "mean")).round(2)
    by_source.to_csv(out / "by_reasoning_source.csv")
    lines += ["", "## The same metric depends on WHERE the chain came from", "",
              "Visible chain of thought vs native reasoning vs untagged prose, per model -- compare "
              "models within one source, not across sources (file `by_reasoning_source.csv`).", "",
              "```", by_source.to_string(), "```"]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:8]))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
