#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F1a Capacity-Only Non-Duplicate Collector

Collects single-factor capacity-reduction runs produced by official 02_src/07_train.py.

This version intentionally excludes:
- dropout-only
- weight-decay-only
because user already tried regularization-only and it was ineffective.

No tree.
No pair head.
No reranker.
No prototype frozen CLS.
No center loss.
No K/preprocessing changes.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd


BASE_REFERENCE = {
    "name": "official_D3_baseline_F0",
    "train_macro_f1": 0.910253,
    "val_macro_f1": 0.810094,
    "gap_macro_f1": 0.100158,
    "note": "From F0 official D3 audit; used as reference.",
}


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def repo_root_from_here() -> Path:
    p = Path(__file__).resolve()
    if p.parent.name == "05_test":
        return p.parents[1]
    return Path.cwd().resolve()


def resolve_path(p: str | Path, root: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (root / p).resolve()


def maybe_get(d: Dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def read_run(run_dir: Path) -> Dict[str, Any] | None:
    diag_path = run_dir / "diagnosis_summary.json"
    cfg_path = run_dir / "config.json"
    hist_path = run_dir / "history.csv"
    val_report_path = run_dir / "val_classification_report_best.json"
    train_report_path = run_dir / "train_classification_report_best.json"

    if not diag_path.exists():
        return None

    diag = load_json(diag_path)
    cfg = load_json(cfg_path) if cfg_path.exists() else {}
    val_report = load_json(val_report_path) if val_report_path.exists() else {}
    train_report = load_json(train_report_path) if train_report_path.exists() else {}

    model_cfg = cfg.get("model", cfg.get("model_config", {}))
    if not isinstance(model_cfg, dict):
        model_cfg = {}

    row = {
        "variant": run_dir.name,
        "run_dir": str(run_dir),
        "best_epoch": diag.get("best_epoch"),
        "epochs_ran": None,
        "train_macro_f1": maybe_get(diag, "train", "macro_f1"),
        "val_macro_f1": maybe_get(diag, "val", "macro_f1"),
        "gap_macro_f1": diag.get("generalization_gap_macro_f1"),
        "train_acc": maybe_get(diag, "train", "accuracy"),
        "val_acc": maybe_get(diag, "val", "accuracy"),
        "train_weighted_f1": maybe_get(diag, "train", "weighted_f1"),
        "val_weighted_f1": maybe_get(diag, "val", "weighted_f1"),
        "train_malware_avg_f1": maybe_get(diag, "train", "malware_only_avg_f1"),
        "val_malware_avg_f1": maybe_get(diag, "val", "malware_only_avg_f1"),
        "hidden_dim": model_cfg.get("hidden_dim"),
        "num_layers": model_cfg.get("num_layers"),
        "num_heads": model_cfg.get("num_heads"),
        "dropout": model_cfg.get("dropout"),
        "classifier_hidden_dim": model_cfg.get("classifier_hidden_dim"),
        "classifier_dropout": model_cfg.get("classifier_dropout"),
        "lr": cfg.get("lr"),
        "weight_decay": cfg.get("weight_decay"),
        "batch_size": cfg.get("batch_size"),
    }

    if hist_path.exists():
        hist = pd.read_csv(hist_path)
        row["epochs_ran"] = int(hist["epoch"].max()) if "epoch" in hist and len(hist) else len(hist)
        if "val_macro_f1" in hist and len(hist):
            best_i = hist["val_macro_f1"].idxmax()
            row["history_best_val_macro_f1"] = float(hist.loc[best_i, "val_macro_f1"])
            row["history_best_train_macro_f1"] = float(hist.loc[best_i, "train_macro_f1"]) if "train_macro_f1" in hist else None
            row["history_best_gap"] = float(hist.loc[best_i, "macro_f1_gap_train_minus_val"]) if "macro_f1_gap_train_minus_val" in hist else None

    for split, rep in [("train", train_report), ("val", val_report)]:
        per = rep.get("per_class", {})
        if isinstance(per, dict):
            for label, metrics in per.items():
                if isinstance(metrics, dict) and "f1" in metrics:
                    safe_label = str(label).replace(" ", "_")
                    row[f"{split}_f1_{safe_label}"] = float(metrics["f1"])

    if row["val_macro_f1"] is not None:
        row["delta_val_vs_base"] = float(row["val_macro_f1"] - BASE_REFERENCE["val_macro_f1"])
    if row["train_macro_f1"] is not None:
        row["delta_train_vs_base"] = float(row["train_macro_f1"] - BASE_REFERENCE["train_macro_f1"])
    if row["gap_macro_f1"] is not None:
        row["delta_gap_vs_base"] = float(row["gap_macro_f1"] - BASE_REFERENCE["gap_macro_f1"])

    row["val_improved"] = bool(row.get("delta_val_vs_base", -999) > 0)
    row["gap_reduced"] = bool(row.get("delta_gap_vs_base", 999) < 0)

    if row["val_improved"] and row["gap_reduced"]:
        row["capacity_diagnosis"] = "good_capacity_signal"
    elif (not row["val_improved"]) and row["gap_reduced"]:
        row["capacity_diagnosis"] = "weakened_but_underfit_or_no_gain"
    elif row["val_improved"] and (not row["gap_reduced"]):
        row["capacity_diagnosis"] = "val_gain_without_gap_reduction_check_noise"
    else:
        row["capacity_diagnosis"] = "no_capacity_fix_signal"

    return row


def write_markdown(out_dir: Path, df: pd.DataFrame) -> None:
    lines = []
    lines.append("# F1a Capacity-Only Non-Duplicate Summary\n")
    lines.append("This excludes regularization-only runs because those were already tried and ineffective.\n")
    lines.append("## Baseline reference\n")
    lines.append("```text")
    lines.append(f"official D3 train macro-F1 = {BASE_REFERENCE['train_macro_f1']:.6f}")
    lines.append(f"official D3 val macro-F1   = {BASE_REFERENCE['val_macro_f1']:.6f}")
    lines.append(f"official D3 gap            = {BASE_REFERENCE['gap_macro_f1']:.6f}")
    lines.append("```")
    lines.append("\n## Variants\n")
    if len(df) == 0:
        lines.append("No completed F1a run dirs found.")
    else:
        cols = [
            "variant", "best_epoch", "train_macro_f1", "val_macro_f1", "gap_macro_f1",
            "delta_train_vs_base", "delta_val_vs_base", "delta_gap_vs_base", "capacity_diagnosis",
        ]
        lines.append(df[cols].sort_values("val_macro_f1", ascending=False).to_markdown(index=False))
        best = df.sort_values("val_macro_f1", ascending=False).iloc[0]
        lines.append("\n## Best by val macro-F1\n")
        lines.append("```text")
        lines.append(f"variant       = {best['variant']}")
        lines.append(f"val_macro_f1  = {best['val_macro_f1']:.6f}")
        lines.append(f"train_macro_f1= {best['train_macro_f1']:.6f}")
        lines.append(f"gap           = {best['gap_macro_f1']:.6f}")
        lines.append(f"delta_train   = {best['delta_train_vs_base']:+.6f}")
        lines.append(f"delta_val     = {best['delta_val_vs_base']:+.6f}")
        lines.append(f"delta_gap     = {best['delta_gap_vs_base']:+.6f}")
        lines.append(f"diagnosis     = {best['capacity_diagnosis']}")
        lines.append("```")

    lines.append("\n## Decision rule\n")
    lines.append("```text")
    lines.append("If reduced capacity lowers train macro-F1 and raises val macro-F1:")
    lines.append("  capacity overfit confirmed.")
    lines.append("")
    lines.append("If reduced capacity lowers train and val:")
    lines.append("  model needs capacity; go F1b branch/fusion ablation.")
    lines.append("")
    lines.append("If only classifier bottleneck helps:")
    lines.append("  overfit likely at classifier/fusion output, not attention stack.")
    lines.append("")
    lines.append("If layer/head/hidden reductions help:")
    lines.append("  overfit likely in attention interaction capacity.")
    lines.append("```")

    (out_dir / "F1a_capacity_only_summary.md").write_text("\n".join(lines), encoding="utf-8")


def zip_dir(src_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in src_dir.rglob("*"):
            if p.is_file() and p != zip_path:
                z.write(p, p.relative_to(src_dir.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="05_test/outputs/F1a_capacity_only")
    ap.add_argument("--out-dir", default="05_test/outputs/F1a_capacity_only_summary")
    ap.add_argument("--make-zip", action="store_true", default=True)
    ap.add_argument("--no-zip", dest="make_zip", action="store_false")
    args = ap.parse_args()

    root = repo_root_from_here()
    runs_root = resolve_path(args.runs_root, root)
    out_dir = resolve_path(args.out_dir, root)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    if runs_root.exists():
        for child in sorted(runs_root.iterdir()):
            if child.is_dir():
                row = read_run(child)
                if row is not None:
                    rows.append(row)

    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["val_macro_f1", "gap_macro_f1"], ascending=[False, True])
    df.to_csv(out_dir / "F1a_capacity_only_summary.csv", index=False)
    (out_dir / "F1a_baseline_reference.json").write_text(json.dumps(BASE_REFERENCE, indent=2), encoding="utf-8")
    write_markdown(out_dir, df)

    if args.make_zip:
        zip_path = out_dir.with_suffix(".zip")
        zip_dir(out_dir, zip_path)
        print(f"[F1a collect] zip: {zip_path}")
    print(f"[F1a collect] rows={len(df)}")
    if len(df):
        print(df[["variant", "train_macro_f1", "val_macro_f1", "gap_macro_f1", "delta_val_vs_base", "delta_gap_vs_base", "capacity_diagnosis"]].to_string(index=False))


if __name__ == "__main__":
    main()
