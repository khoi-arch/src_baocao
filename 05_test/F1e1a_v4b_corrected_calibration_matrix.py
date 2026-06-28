#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F1e1a_v4b Corrected Calibration-Derived Family Smoothing Matrix

Purpose
-------
No training.
No validation usage.
No fake data.

This script reads outputs from F1e1a_v4 and re-derives a corrected locked
family-aware smoothing matrix.

Why v4b?
--------
F1e1a_v4 was methodologically cleaner than validation-derived smoothing, but its
eps rule was too coarse:

    eps_family = min(eps_cap, other_malware_prob_mass_mean)

In the observed v4 output this saturated all malware families at eps=0.20.
That loses the key family-specific strength signal.

v4b fixes only the eps strength rule and keeps the calibration-only protocol.

Corrected eps rule
------------------
For each family f:

    error_gap_f = max(0, calibration_error_rate_f - train_inner_error_rate_f)

    other_gap_f = max(
        0,
        calibration_other_malware_prob_mass_mean_f
        - train_inner_other_malware_prob_mass_mean_f
    )

    eps_raw_f = 0.5 * error_gap_f + 0.5 * other_gap_f

    eps_used_f = min(max_eps, eps_raw_f)

This uses only train_inner/calibration outputs, not validation.

Direction allocation
--------------------
The direction of smoothing mass still comes from calibration family behavior:

    score(target other malware class k) =
        0.55 * mean_prob_k
      + 0.30 * top2_rate_k
      + 0.15 * pred_rate_k

Then eps is allocated across other malware classes proportional to score.

Inputs
------
Default input dir:
    05_test/outputs/F1e1a_v4_clean_calibration_family_smoothing

Required:
    F1e1a_v4_family_summary_by_split.csv

Optional:
    F1e1a_v4_locked_family_smoothing_matrix_CALIBRATION_DERIVED.csv

Outputs
-------
- F1e1a_v4b_family_gap_evidence.csv
- F1e1a_v4b_locked_family_smoothing_matrix_CORRECTED_CALIBRATION_DERIVED.csv
- F1e1a_v4b_locked_family_smoothing_targets.json
- F1e1a_v4b_report.md
- F1e1a_v4b_leakage_policy.md
- combined zip
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


DEFAULT_CLASSES = ["Benign", "Ransomware", "Spyware", "Trojan"]
DEFAULT_MALWARE = ["Ransomware", "Spyware", "Trojan"]


def log(msg: str):
    print(f"[F1e1a_v4b] {msg}", flush=True)


def repo_root_from_here() -> Path:
    p = Path(__file__).resolve()
    if p.parent.name == "05_test":
        return p.parents[1]
    return Path.cwd().resolve()


def resolve_path(p: str | Path, root: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (root / p).resolve()


def clean(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def parse_list(s: str) -> List[str]:
    return [clean(x) for x in s.split(",") if clean(x)]


def require_cols(df: pd.DataFrame, cols: List[str], name: str):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise KeyError(f"{name} missing required columns: {miss}")


def load_v4_summary(v4_dir: Path) -> pd.DataFrame:
    p = v4_dir / "F1e1a_v4_family_summary_by_split.csv"
    if not p.exists():
        raise FileNotFoundError(f"missing {p}")
    df = pd.read_csv(p)
    require_cols(df, [
        "split", "true_L2", "true_L3", "n", "accuracy", "error_rate",
        "other_malware_prob_mass_mean",
    ], "F1e1a_v4_family_summary_by_split.csv")
    df["split"] = df["split"].map(clean)
    df["true_L2"] = df["true_L2"].map(clean)
    df["true_L3"] = df["true_L3"].map(clean)
    return df


def build_gap_evidence(fam: pd.DataFrame, class_names: List[str], malware_classes: List[str]) -> pd.DataFrame:
    train = fam[fam["split"] == "train_inner"].copy()
    cal = fam[fam["split"] == "calibration"].copy()
    if len(train) == 0 or len(cal) == 0:
        raise ValueError("Need both train_inner and calibration rows in family summary.")

    keys = ["true_L2", "true_L3"]
    tcols = keys + [
        "n", "accuracy", "error_rate", "true_in_top2_rate", "top2_gap_mean",
        "true_prob_mean", "other_malware_prob_mass_mean",
    ]
    ccols = tcols.copy()

    for c in class_names:
        for base in [f"mean_prob_{c}", f"pred_rate_{c}", f"top2_rate_{c}"]:
            if base in train.columns and base not in tcols:
                tcols.append(base)
            if base in cal.columns and base not in ccols:
                ccols.append(base)

    t = train[tcols].rename(columns={c: f"train_inner_{c}" for c in tcols if c not in keys})
    c = cal[ccols].rename(columns={c: f"calibration_{c}" for c in ccols if c not in keys})
    ev = pd.merge(c, t, on=keys, how="outer", validate="one_to_one")

    for col in [
        "calibration_error_rate", "train_inner_error_rate",
        "calibration_other_malware_prob_mass_mean", "train_inner_other_malware_prob_mass_mean",
        "calibration_true_prob_mean", "train_inner_true_prob_mean",
        "calibration_n", "train_inner_n",
    ]:
        if col in ev.columns:
            ev[col] = pd.to_numeric(ev[col], errors="coerce")

    ev["error_gap"] = (ev["calibration_error_rate"] - ev["train_inner_error_rate"]).clip(lower=0).fillna(0)
    ev["other_malware_mass_gap"] = (
        ev["calibration_other_malware_prob_mass_mean"]
        - ev["train_inner_other_malware_prob_mass_mean"]
    ).clip(lower=0).fillna(0)
    ev["true_prob_drop"] = (
        ev["train_inner_true_prob_mean"]
        - ev["calibration_true_prob_mean"]
    ).clip(lower=0).fillna(0) if "train_inner_true_prob_mean" in ev.columns and "calibration_true_prob_mean" in ev.columns else 0.0

    ev["eps_raw_gap_avg"] = 0.5 * ev["error_gap"] + 0.5 * ev["other_malware_mass_gap"]
    ev["is_malware_family"] = ev["true_L2"].isin(malware_classes)
    ev["has_required_calibration_support"] = ev["calibration_n"].fillna(0) > 0
    return ev


def derive_corrected_matrix(
    ev: pd.DataFrame,
    class_names: List[str],
    malware_classes: List[str],
    max_eps: float,
    min_family_support: int,
) -> pd.DataFrame:
    rows = []
    for _, r in ev.iterrows():
        true_l2 = clean(r["true_L2"])
        true_l3 = clean(r["true_L3"])
        ncal = int(r["calibration_n"]) if pd.notna(r.get("calibration_n", np.nan)) else 0
        row = {
            "true_L2": true_l2,
            "true_L3": true_l3,
            "n_calibration": ncal,
            "n_train_inner": int(r["train_inner_n"]) if pd.notna(r.get("train_inner_n", np.nan)) else 0,
            "reliable_support": bool(ncal >= min_family_support),
            "eps_rule": "0.5*max(0,cal_error-train_error)+0.5*max(0,cal_other_mass-train_other_mass)",
            "max_eps": float(max_eps),
            "error_gap": float(r.get("error_gap", 0.0)),
            "other_malware_mass_gap": float(r.get("other_malware_mass_gap", 0.0)),
            "true_prob_drop": float(r.get("true_prob_drop", 0.0)),
            "eps_raw": float(r.get("eps_raw_gap_avg", 0.0)),
            "eps_used": 0.0,
            "cap_active": False,
            "source": "",
        }
        for c in class_names:
            row[f"target_{c}"] = 0.0

        if true_l2 not in malware_classes:
            if true_l2 in class_names:
                row[f"target_{true_l2}"] = 1.0
            elif "Benign" in class_names:
                row["target_Benign"] = 1.0
            else:
                row[f"target_{class_names[0]}"] = 1.0
            row["source"] = "non_malware_or_benign_one_hot"
            row["target_sum"] = sum(row[f"target_{c}"] for c in class_names)
            rows.append(row)
            continue

        if ncal < min_family_support:
            row[f"target_{true_l2}"] = 1.0
            row["source"] = "support_below_min_keep_one_hot"
            row["target_sum"] = sum(row[f"target_{c}"] for c in class_names)
            rows.append(row)
            continue

        eps_raw = max(0.0, float(r.get("eps_raw_gap_avg", 0.0)))
        eps = min(float(max_eps), eps_raw)
        row["eps_used"] = eps
        row["cap_active"] = bool(eps_raw > max_eps + 1e-12)

        other_scores = {}
        for c in malware_classes:
            if c == true_l2:
                continue
            mean_prob = float(r.get(f"calibration_mean_prob_{c}", 0.0))
            top2_rate = float(r.get(f"calibration_top2_rate_{c}", 0.0))
            pred_rate = float(r.get(f"calibration_pred_rate_{c}", 0.0))
            score = 0.55 * mean_prob + 0.30 * top2_rate + 0.15 * pred_rate
            other_scores[c] = max(0.0, score)

        total_score = sum(other_scores.values())
        if eps <= 0 or total_score <= 0:
            row[f"target_{true_l2}"] = 1.0
            row["eps_used"] = 0.0
            row["source"] = "no_positive_gap_or_no_direction_signal_keep_one_hot"
        else:
            row[f"target_{true_l2}"] = 1.0 - eps
            for c, sc in other_scores.items():
                row[f"target_{c}"] = eps * sc / total_score
            if "Benign" in class_names and true_l2 != "Benign":
                row["target_Benign"] = 0.0
            row["source"] = "CORRECTED_CALIBRATION_GAP_DERIVED_family_prob_top2_pred_weighted"

        row["target_sum"] = sum(row[f"target_{c}"] for c in class_names)
        rows.append(row)
    return pd.DataFrame(rows)


def attach_old_matrix_compare(matrix: pd.DataFrame, v4_dir: Path) -> pd.DataFrame:
    oldp = v4_dir / "F1e1a_v4_locked_family_smoothing_matrix_CALIBRATION_DERIVED.csv"
    out = matrix.copy()
    if not oldp.exists():
        out["old_v4_eps_family"] = np.nan
        out["eps_delta_vs_old_v4"] = np.nan
        return out
    old = pd.read_csv(oldp)
    keys = ["true_L2", "true_L3"]
    eps_col = "eps_family" if "eps_family" in old.columns else ("eps_used" if "eps_used" in old.columns else None)
    if eps_col is None:
        out["old_v4_eps_family"] = np.nan
        out["eps_delta_vs_old_v4"] = np.nan
        return out
    old = old[keys + [eps_col]].rename(columns={eps_col: "old_v4_eps_family"})
    out = pd.merge(out, old, on=keys, how="left")
    out["eps_delta_vs_old_v4"] = out["eps_used"] - out["old_v4_eps_family"]
    return out


def write_policy(out_dir: Path):
    txt = """# F1e1a_v4b Leakage Policy

## What v4b does

v4b does not train and does not use validation.

It reads only F1e1a_v4 train_inner/calibration outputs and derives a corrected
family smoothing matrix from calibration-vs-train_inner gap.

## Why this is cleaner than validation tuning

The validation split is not used for:
- eps strength
- family direction allocation
- matrix selection
- final model choice

Therefore, the resulting matrix can be locked before F1e1b validation evaluation.

## Remaining caveat

F1e1a_v4 itself reused the existing preprocessed token dataset.
A fully nested strict version would rebuild preprocessing/tokenization using
train_inner only. v4b does not make this worse; it only re-derives the matrix
from already existing clean calibration outputs.

## How to use

Use:

    F1e1a_v4b_locked_family_smoothing_matrix_CORRECTED_CALIBRATION_DERIVED.csv

for F1e1b.

Do not tune this matrix using validation results.
"""
    (out_dir / "F1e1a_v4b_leakage_policy.md").write_text(txt, encoding="utf-8")


def write_report(out_dir: Path, ev: pd.DataFrame, matrix: pd.DataFrame, class_names: List[str], malware_classes: List[str]):
    lines = []
    lines.append("# F1e1a_v4b Corrected Calibration-Derived Family Smoothing Matrix\n")
    lines.append("## Purpose\n")
    lines.append("```text")
    lines.append("No training.")
    lines.append("No validation usage.")
    lines.append("Re-derive corrected family-aware matrix from F1e1a_v4 train_inner/calibration outputs.")
    lines.append("```")

    lines.append("\n## Why v4b was needed\n")
    lines.append("```text")
    lines.append("F1e1a_v4 used eps = min(0.20, other_malware_prob_mass_mean).")
    lines.append("That saturated all malware families at eps=0.20, so family-specific strength was lost.")
    lines.append("v4b uses calibration-vs-train_inner gap instead.")
    lines.append("```")

    lines.append("\n## Corrected eps rule\n")
    lines.append("```text")
    lines.append("error_gap = max(0, calibration_error_rate - train_inner_error_rate)")
    lines.append("other_gap = max(0, calibration_other_malware_prob_mass_mean - train_inner_other_malware_prob_mass_mean)")
    lines.append("eps_raw = 0.5 * error_gap + 0.5 * other_gap")
    lines.append("eps_used = min(max_eps, eps_raw)")
    lines.append("```")

    lines.append("\n## Family gap evidence\n")
    show_cols = [c for c in [
        "true_L2", "true_L3",
        "calibration_n", "train_inner_n",
        "calibration_error_rate", "train_inner_error_rate", "error_gap",
        "calibration_other_malware_prob_mass_mean", "train_inner_other_malware_prob_mass_mean", "other_malware_mass_gap",
        "true_prob_drop", "eps_raw_gap_avg",
    ] if c in ev.columns]
    ev_show = ev[ev["true_L2"].isin(malware_classes)].sort_values("eps_raw_gap_avg", ascending=False)
    lines.append(ev_show[show_cols].to_markdown(index=False))

    lines.append("\n## Corrected locked matrix\n")
    matrix_show_cols = [c for c in [
        "true_L2", "true_L3", "n_calibration", "eps_raw", "eps_used", "old_v4_eps_family",
        "eps_delta_vs_old_v4", "target_Benign", "target_Ransomware", "target_Spyware", "target_Trojan", "source"
    ] if c in matrix.columns]
    lines.append(matrix[matrix_show_cols].to_markdown(index=False))

    lines.append("\n## Sanity summary\n")
    mal = matrix[matrix["true_L2"].isin(malware_classes)].copy()
    if len(mal):
        lines.append("```text")
        lines.append(f"malware_family_count = {len(mal)}")
        lines.append(f"eps_min = {mal['eps_used'].min():.6f}")
        lines.append(f"eps_median = {mal['eps_used'].median():.6f}")
        lines.append(f"eps_mean = {mal['eps_used'].mean():.6f}")
        lines.append(f"eps_max = {mal['eps_used'].max():.6f}")
        lines.append(f"cap_active_count = {int(mal['cap_active'].sum())}")
        lines.append("```")

    lines.append("\n## Next step\n")
    lines.append("```text")
    lines.append("F1e1b: train full original train with this locked corrected family matrix.")
    lines.append("Evaluate validation once.")
    lines.append("Do not tune matrix using validation.")
    lines.append("```")
    (out_dir / "F1e1a_v4b_report.md").write_text("\n".join(lines), encoding="utf-8")


def zip_dir(src: Path, dst: Path):
    if dst.exists():
        dst.unlink()
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for p in src.rglob("*"):
            if p.is_file() and p != dst:
                z.write(p, p.relative_to(src.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v4-dir", default="05_test/outputs/F1e1a_v4_clean_calibration_family_smoothing")
    ap.add_argument("--out-dir", default="05_test/outputs/F1e1a_v4b_corrected_calibration_family_matrix")
    ap.add_argument("--combined-zip", default="05_test/outputs/F1e1a_v4b_corrected_calibration_family_matrix.zip")
    ap.add_argument("--class-names", default="Benign,Ransomware,Spyware,Trojan")
    ap.add_argument("--malware-classes", default="Ransomware,Spyware,Trojan")
    ap.add_argument("--max-eps", type=float, default=0.30)
    ap.add_argument("--min-family-support", type=int, default=30)
    args = ap.parse_args()

    root = repo_root_from_here()
    v4_dir = resolve_path(args.v4_dir, root)
    out_dir = resolve_path(args.out_dir, root)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = resolve_path(args.combined_zip, root)

    class_names = parse_list(args.class_names) or DEFAULT_CLASSES
    malware_classes = parse_list(args.malware_classes) or DEFAULT_MALWARE

    log(f"root={root}")
    log(f"v4_dir={v4_dir}")
    log(f"out_dir={out_dir}")
    log("No training. No validation. Re-derive corrected calibration matrix only.")

    fam = load_v4_summary(v4_dir)
    ev = build_gap_evidence(fam, class_names, malware_classes)
    ev.to_csv(out_dir / "F1e1a_v4b_family_gap_evidence.csv", index=False)

    matrix = derive_corrected_matrix(
        ev=ev,
        class_names=class_names,
        malware_classes=malware_classes,
        max_eps=float(args.max_eps),
        min_family_support=int(args.min_family_support),
    )
    matrix = attach_old_matrix_compare(matrix, v4_dir)
    matrix.to_csv(out_dir / "F1e1a_v4b_locked_family_smoothing_matrix_CORRECTED_CALIBRATION_DERIVED.csv", index=False)

    targets = {
        "usage": "Use this locked corrected calibration-derived matrix for F1e1b.",
        "validation_used": False,
        "training_done_in_v4b": False,
        "fake_data_used": False,
        "source_v4_dir": str(v4_dir),
        "class_names": class_names,
        "malware_classes": malware_classes,
        "eps_rule": "eps_raw = 0.5*max(0,cal_error-train_error)+0.5*max(0,cal_other_malware_mass-train_other_malware_mass)",
        "direction_rule": "0.55*calibration_mean_prob_other + 0.30*calibration_top2_rate_other + 0.15*calibration_pred_rate_other",
        "max_eps": float(args.max_eps),
        "min_family_support": int(args.min_family_support),
        "matrix_file": "F1e1a_v4b_locked_family_smoothing_matrix_CORRECTED_CALIBRATION_DERIVED.csv",
        "matrix": matrix.to_dict(orient="records"),
    }
    (out_dir / "F1e1a_v4b_locked_family_smoothing_targets.json").write_text(json.dumps(targets, indent=2), encoding="utf-8")

    write_policy(out_dir)
    write_report(out_dir, ev, matrix, class_names, malware_classes)
    zip_dir(out_dir, zip_path)

    mal = matrix[matrix["true_L2"].isin(malware_classes)].copy()
    log("Corrected eps summary:")
    if len(mal):
        print(mal[["true_L2", "true_L3", "eps_raw", "eps_used", "old_v4_eps_family"]].sort_values("eps_used", ascending=False).to_string(index=False), flush=True)
    log(f"zip={zip_path}")
    log("DONE")


if __name__ == "__main__":
    main()
