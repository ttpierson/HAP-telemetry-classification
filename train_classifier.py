#!/usr/bin/env python3
"""Train and evaluate classifiers on collected telemetry traces.

    # workload type: binary (training vs rest) and three-way
    python train_classifier.py workload --data data/workload --out models/workload

    # which LLM is running, with and without memory-derived features
    python train_classifier.py model_id --data data/model_id --out models/model_id

    # score traces with a saved bundle
    python train_classifier.py score --bundle models/workload_200w/threeway.joblib TRACE.parquet ...

Scores come from grouped cross-validation, reported two ways:

  grouped by run       no window straddles train and test, but other runs of
                       the same config are in training: "recognise a known
                       workload".
  grouped by workload  a whole config is held out: "generalise to an unseen
                       workload". This is the number to quote.

Train on one power regime at a time. A 200 W-capped and an uncapped GPU differ
by ~150 W on the most heavily weighted signals, so a pooled classifier learns
which machine a trace came from.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gputel import features as F  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)


def rf(trees: int) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=trees, min_samples_leaf=2,
                                  max_features="sqrt", class_weight="balanced",
                                  random_state=42, n_jobs=-1)


def evaluate(X, y, groups, trees, max_folds=5, splitter=None) -> dict | None:
    """Grouped CV. None when the smallest class has fewer than two groups."""
    if splitter is None:
        per_class = pd.DataFrame({"y": y, "g": groups}).groupby("y")["g"].nunique()
        k = int(min(max_folds, per_class.min()))
        if k < 2:
            return None
        splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=42)
    yt, yp, folds = [], [], 0
    for tr, te in splitter.split(X, y, groups):
        yt.extend(y[te])
        yp.extend(rf(trees).fit(X[tr], y[tr]).predict(X[te]))
        folds += 1
    labels = sorted(set(yt) | set(yp))
    return dict(folds=folds, accuracy=float(accuracy_score(yt, yp)),
                macro_f1=float(f1_score(yt, yp, average="macro", zero_division=0)),
                labels=labels, confusion=confusion_matrix(yt, yp, labels=labels).tolist())


def confusion_md(s: dict) -> list[str]:
    labs = s["labels"]
    lines = ["| true \\ pred | " + " | ".join(labs) + " |",
             "|---" * (len(labs) + 1) + "|"]
    lines += [f"| {l} | " + " | ".join(map(str, row)) + " |"
              for l, row in zip(labs, s["confusion"])]
    return lines


def fmt(s: dict | None) -> str:
    if s is None:
        return "n/a (fewer than two groups in some class)"
    return f"accuracy **{s['accuracy']:.3f}**, macro-F1 {s['macro_f1']:.3f} ({s['folds']} folds)"


def provenance(raw: pd.DataFrame) -> dict:
    per_run_peak = raw.groupby("run_id")["power_draw_w"].max()
    p = {"gpu_names": sorted(raw["gpu_name"].dropna().unique().tolist())
         if "gpu_name" in raw else [],
         "peak_power_w_median_run": round(float(per_run_peak.median()), 1),
         "peak_power_w_max": round(float(per_run_peak.max()), 1)}
    if "power_limit_w" in raw:
        p["power_limits_w"] = sorted(raw["power_limit_w"].dropna().unique().tolist())
    return p


def save(out: Path, name: str, model, cols, a, **extra):
    joblib.dump({"model": model, "feature_cols": cols, "classes": list(model.classes_),
                 "window_sec": a.window, "stride_sec": a.stride, **extra},
                out / f"{name}.joblib")


def load_windows(a):
    raw = F.load_traces(a.data, min_samples=a.min_samples)
    if a.holdout_config:
        held = raw["workload_label"].isin(a.holdout_config)
        print(f"holding out {raw.loc[held, 'run_id'].nunique()} run(s) of "
              f"{a.holdout_config}")
        raw = raw[~held]
    print(f"{raw['run_id'].nunique()} runs, {raw['workload_label'].nunique()} configs; "
          f"windowing at {a.window}s / {a.stride}s ...")
    win = F.sliding_windows(raw, a.window, a.stride)
    cols = F.feature_cols(win)
    print(f"{len(win)} windows x {len(cols)} features")
    return raw, win, cols


def base_meta(a, raw, win, cols) -> dict:
    return {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "data_dir": str(a.data), "window_sec": a.window, "stride_sec": a.stride,
            "n_windows": len(win), "n_runs": int(win["run_id"].nunique()),
            "n_configs": int(win["workload_label"].nunique()),
            "holdout_configs": a.holdout_config, "provenance": provenance(raw),
            "train_run_ids": sorted(win["run_id"].unique().tolist()),
            "train_workload_labels": sorted(win["workload_label"].unique().tolist()),
            "feature_cols": cols, "scores": {}}


# ── tasks ────────────────────────────────────────────────────────────────────

def train_workload(a) -> None:
    raw, win, cols = load_windows(a)
    X = F.to_matrix(win, cols)
    runs, configs = win["run_id"].to_numpy(), win["workload_label"].to_numpy()
    y3 = win["threeway_label"].to_numpy()
    y2 = np.where(y3 == "ml_training", "ml_training", "rest")

    meta = base_meta(a, raw, win, cols)
    per_run = win.groupby("run_id")[["threeway_label", "workload_label"]].first()
    meta["runs_per_class"] = per_run["threeway_label"].value_counts().to_dict()
    meta["configs_per_class"] = (per_run.drop_duplicates("workload_label")
                                 ["threeway_label"].value_counts().to_dict())

    report = [f"# Workload classifier — {a.data}", "",
              f"{meta['n_runs']} runs, {meta['n_configs']} configs, "
              f"{len(win)} windows x {len(cols)} features.", "",
              "| class | runs | configs |", "|---|---|---|"]
    report += [f"| {c} | {meta['runs_per_class'][c]} | {meta['configs_per_class'][c]} |"
               for c in sorted(meta["runs_per_class"])]
    report.append("")

    for task, y in (("binary", y2), ("threeway", y3)):
        by_run = evaluate(X, y, runs, a.trees)
        by_wl = evaluate(X, y, configs, a.trees)
        meta["scores"][task] = {"grouped_by_run": by_run, "grouped_by_workload": by_wl}
        report += [f"## {task}", "",
                   f"- grouped by run: {fmt(by_run)}",
                   f"- grouped by workload: {fmt(by_wl)}", ""]
        if by_wl:
            report += ["Grouped-by-workload confusion:", ""] + confusion_md(by_wl) + [""]
        model = rf(a.trees).fit(X, y)
        save(a.out, task, model, cols, a, task=task)
        print(f"{task:9s} by run {fmt(by_run)}\n{'':9s} by workload {fmt(by_wl)}")

        if task == "binary":
            imp = pd.Series(model.feature_importances_, index=cols).sort_values(ascending=False)
            report += ["Top 15 features (binary):", "", "| feature | importance |", "|---|---|"]
            report += [f"| `{k}` | {v:.4f} |" for k, v in imp.head(15).items()] + [""]

    finish(a, meta, report)


def train_model_id(a) -> None:
    raw, win, cols = load_windows(a)
    win["model"] = win["workload_label"].map(F.model_label)
    X = F.to_matrix(win, cols)
    y, runs = win["model"].to_numpy(), win["run_id"].to_numpy()
    nomem = np.array([not c.startswith(F.MEMORY_PREFIXES) for c in cols])
    mem_only = np.array([c == "mem_used_mb_mean" for c in cols])

    batches = win["workload_label"].map(F.batch_size)
    sweep = batches.notna().all() and batches.nunique() > 1

    meta = base_meta(a, raw, win, cols)
    meta["n_models"] = int(len(set(y)))
    meta["memory_footprint_mb"] = {m: round(float(v)) for m, v in
                                   win.groupby("model")["mem_used_mb_mean"].mean().items()}
    report = [f"# Model identification — {a.data}", "",
              f"{meta['n_models']} models, {meta['n_runs']} runs, {len(win)} windows.", "",
              "Memory footprint is close to a readout of parameter count, so the "
              "classifier is also scored with every memory-derived feature removed, "
              "and against a control that sees only `mem_used_mb_mean`.", ""]

    for name, mask in (("with_memory", np.ones(len(cols), bool)),
                       ("no_memory", nomem), ("control_mem_used_mean_only", mem_only)):
        s = {"grouped_by_run": evaluate(X[:, mask], y, runs, a.trees, max_folds=3)}
        if sweep:
            # Hold out a whole batch size: does model identity survive a change
            # in serving configuration?
            s["leave_one_batch_size_out"] = evaluate(
                X[:, mask], y, batches.to_numpy(), a.trees, splitter=LeaveOneGroupOut())
        meta["scores"][name] = s
        report += [f"## {name} ({int(mask.sum())} features)", "",
                   f"- grouped by run: {fmt(s['grouped_by_run'])}"]
        if sweep:
            report.append(f"- leave one batch size out: {fmt(s['leave_one_batch_size_out'])}")
        report.append("")
        print(f"{name:28s} {fmt(s['grouped_by_run'])}")
        if not name.startswith("control"):
            used = [c for c, m in zip(cols, mask) if m]
            save(a.out, f"model_id_{name}", rf(a.trees).fit(X[:, mask], y), used, a,
                 task=f"model_id_{name}", memory_features_removed=(name == "no_memory"))

    finish(a, meta, report)


def finish(a, meta: dict, report: list[str]) -> None:
    report += ["## Caveats", "",
               "- Small corpus: a perfect score means no errors in these windows, "
               "not a measured error rate.",
               "- One GPU model and one power regime per corpus; do not pool regimes.",
               "- No adversarial workloads: nothing here was tested against "
               "deliberate evasion.", ""]
    (a.out / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    (a.out / "report.md").write_text("\n".join(report))
    print(f"\nwrote bundles, metadata.json and report.md to {a.out}")


def score(a) -> None:
    b = joblib.load(a.bundle)
    for trace in a.traces:
        raw = F.load_trace(trace)
        win = F.sliding_windows(raw, b["window_sec"], b["stride_sec"])
        if win.empty:
            print(f"{Path(trace).name}: no windows (shorter than one window?)\n")
            continue
        proba = b["model"].predict_proba(F.to_matrix(win, b["feature_cols"]))
        pred = pd.Series(b["model"].classes_[proba.argmax(1)])
        label = raw["workload_label"].iloc[0]

        print(f"trace   : {Path(trace).name}")
        print(f"label   : {label} ({F.threeway_label(label)})")
        print(f"windows : {len(win)} @ {b['window_sec']}s")
        for cls, n in pred.value_counts().items():
            print(f"  {cls:16s} {n:4d} windows ({100 * n / len(pred):5.1f}%)")
        print(f"majority: {pred.value_counts().index[0]}   "
              f"mean confidence {proba.max(1).mean():.3f}\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("workload", train_workload), ("model_id", train_model_id)):
        p = sub.add_parser(name)
        p.add_argument("--data", type=Path, required=True,
                       help="directory of .parquet traces (searched recursively)")
        p.add_argument("--out", type=Path, required=True, help="output directory")
        p.add_argument("--window", type=int, default=30, help="window length, s")
        p.add_argument("--stride", type=int, default=15, help="window stride, s")
        p.add_argument("--min-samples", type=int, default=200,
                       help="drop runs shorter than this (truncated / crashed)")
        p.add_argument("--trees", type=int, default=400)
        p.add_argument("--holdout-config", action="append", default=[],
                       help="exclude a workload label from training entirely; repeat")
        p.set_defaults(fn=fn)
    p = sub.add_parser("score")
    p.add_argument("--bundle", required=True)
    p.add_argument("traces", nargs="+", metavar="TRACE")
    p.set_defaults(fn=score)

    a = ap.parse_args(argv)
    if a.cmd != "score":
        a.out.mkdir(parents=True, exist_ok=True)
    a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
