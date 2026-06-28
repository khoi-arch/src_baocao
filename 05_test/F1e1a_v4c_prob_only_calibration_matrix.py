#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F1e1a_v4c Prob-Only Corrected Calibration Matrix

Purpose
-------
No training.
No validation usage.
No fake data.

This script re-derives the locked family-aware smoothing matrix from F1e1a_v4
outputs, but removes the heuristic direction weights used in v4b:

    old v4b direction score =
        0.55 * mean_prob_other
      + 0.30 * top2_rate_other
      + 0.15 * pred_rate_other

That formula does NOT leak validation, but the constants are hand-chosen
hyperparameters. v4c is stricter:

    direction score(other class k) = calibration_mean_prob_k

This uses the model's calibrated/soft belief distribution on the calibration
split directly. top2_rate and pred_rate remain diagnostics only, not matrix
parameters.

Eps strength is still calibration-gap based:

    error_gap = max(0, cal_error - train_inner_error)
    other_gap = max(0, cal_other_malware_mass - train_inner_other_malware_mass)
    eps_raw = 0.5 * error_gap + 0.5 * other_gap
    eps_used = min(max_eps, eps_raw)

Inputs
------
F1e1a_v4 output dir:
    F1e1a_v4_family_summary_by_split.csv

Outputs
-------
- F1e1a_v4c_family_gap_evidence.csv
- F1e1a_v4c_locked_family_smoothing_matrix_PROB_ONLY_CALIBRATION_DERIVED.csv
- F1e1a_v4c_locked_family_smoothing_targets.json
- F1e1a_v4c_report.md
- F1e1a_v4c_leakage_policy.md
- combined zip
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


DEFAULT_CLASSES = ["Benign", "Ransomware", "Spyware", "Trojan"]
DEFAULT_MALWARE = ["Ransomware", "Spyware", "Trojan"]


def log(msg: str):
    print(f"[F1e1a_v4c] {msg}", flush=True)


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


def load_family_summary(v4_dir: Path) -> pd.DataFrame:
    p = v4_dir / "F1e1a_v4_family_summary_by_split.csv"
    if not p.exists():
        raise FileNotFoundError(f"missing {p}")
    df = pd.read_csv(p)
    required = [
        "split", "true_L2", "true_L3", "n", "error_rate",
        "other_malware_prob_mass_mean"
    ]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise KeyError(f"family summary missing columns: {miss}")
    df["split"] = df["split"].map(clean)
    df["true_L2"] = df["true_L2"].map(clean)
    df["true_L3"] = df["true_L3"].map(clean)
    return df


def build_gap_evidence(fam: pd.DataFrame, class_names: List[str]) -> pd.DataFrame:
    train = fam[fam["split"] == "train_inner"].copy()
    cal = fam[fam["split"] == "calibration"].copy()
    if len(train) == 0 or len(cal) == 0:
        raise ValueError("Need both train_inner and calibration rows.")

    keys = ["true_L2", "true_L3"]
    base_cols = [
        "n", "accuracy", "error_rate", "true_in_top2_rate", "top2_gap_mean",
        "true_prob_mean", "other_malware_prob_mass_mean",
    ]
    cols = keys + [c for c in base_cols if c in fam.columns]
    for c in class_names:
        for x in [f"mean_prob_{c}", f"pred_rate_{c}", f"top2_rate_{c}"]:
            if x in fam.columns and x not in cols:
                cols.append(x)

    t = train[cols].rename(columns={c: f"train_inner_{c}" for c in cols if c not in keys})
    c = cal[cols].rename(columns={c: f"calibration_{c}" for c in cols if c not in keys})
    ev = pd.merge(c, t, on=keys, how="outer", validate="one_to_one")

    numeric_cols = [c for c in ev.columns if c not in keys]
    for c in numeric_cols:
        ev[c] = pd.to_numeric(ev[c], errors="coerce")

    ev["error_gap"] = (ev["calibration_error_rate"] - ev["train_inner_error_rate"]).clip(lower=0).fillna(0)
    ev["other_malware_mass_gap"] = (
        ev["calibration_other_malware_prob_mass_mean"]
        - ev["train_inner_other_malware_prob_mass_mean"]
    ).clip(lower=0).fillna(0)
    if "train_inner_true_prob_mean" in ev.columns and "calibration_true_prob_mean" in ev.columns:
        ev["true_prob_drop"] = (ev["train_inner_true_prob_mean"] - ev["calibration_true_prob_mean"]).clip(lower=0).fillna(0)
    else:
        ev["true_prob_drop"] = 0.0
    ev["eps_raw_gap_avg"] = 0.5 * ev["error_gap"] + 0.5 * ev["other_malware_mass_gap"]
    return ev


def derive_matrix_prob_only(ev: pd.DataFrame, class_names: List[str], malware_classes: List[str], max_eps: float, min_family_support: int) -> pd.DataFrame:
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
            "eps_rule": "0.5*gap(error_rate)+0.5*gap(other_malware_prob_mass)",
            "direction_rule": "prob_only: target allocation proportional to calibration mean_prob_other",
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
            other_scores[c] = max(0.0, float(r.get(f"calibration_mean_prob_{c}", 0.0)))

        total = sum(other_scores.values())
        if eps <= 0 or total <= 0:
            row[f"target_{true_l2}"] = 1.0
            row["eps_used"] = 0.0
            row["source"] = "no_positive_gap_or_no_prob_direction_keep_one_hot"
        else:
            row[f"target_{true_l2}"] = 1.0 - eps
            for c, sc in other_scores.items():
                row[f"target_{c}"] = eps * sc / total
            if "Benign" in class_names and true_l2 != "Benign":
                row["target_Benign"] = 0.0
            row["source"] = "PROB_ONLY_CORRECTED_CALIBRATION_DERIVED"

        row["target_sum"] = sum(row[f"target_{c}"] for c in class_names)
        rows.append(row)

    return pd.DataFrame(rows)


def compare_old(matrix: pd.DataFrame, old_path: Path, old_label: str, old_col_name: str) -> pd.DataFrame:
    out = matrix.copy()
    if not old_path.exists():
        out[old_col_name] = np.nan
        out[f"eps_delta_vs_{old_label}"] = np.nan
        return out
    old = pd.read_csv(old_path)
    eps_col = "eps_family" if "eps_family" in old.columns else ("eps_used" if "eps_used" in old.columns else None)
    if eps_col is None:
        out[old_col_name] = np.nan
        out[f"eps_delta_vs_{old_label}"] = np.nan
        return out
    old = old[["true_L2", "true_L3", eps_col]].rename(columns={eps_col: old_col_name})
    out = pd.merge(out, old, on=["true_L2", "true_L3"], how="left")
    out[f"eps_delta_vs_{old_label}"] = out["eps_used"] - out[old_col_name]
    return out


def write_policy(out_dir: Path):
    txt = """# F1e1a_v4c Leakage / Cleanliness Policy

v4c does not train and does not use validation.

It reads F1e1a_v4 train_inner/calibration outputs only.

Compared with v4b, v4c removes hand-chosen direction weights
0.55/0.30/0.15. Direction allocation uses only calibration mean probabilities:

    target allocation among other malware classes ∝ mean_prob_other

This is cleaner because no direction-weight hyperparameters are introduced.

Top2 and hard pred rates remain useful diagnostics but are not used to set the
matrix.

Remaining caveat:
F1e1a_v4 reused existing preprocessed train tokens. A fully nested experiment
would rebuild preprocessing/tokenization on train_inner only.
"""
    (out_dir / "F1e1a_v4c_leakage_policy.md").write_text(txt, encoding="utf-8")


def write_report(out_dir: Path, ev: pd.DataFrame, matrix: pd.DataFrame, malware_classes: List[str]):
    lines = []
    lines.append("# F1e1a_v4c Prob-Only Corrected Calibration Matrix\n")
    lines.append("## Cleanliness decision\n")
    lines.append("```text")
    lines.append("The old weighted direction score 0.55/0.30/0.15 did not leak validation,")
    lines.append("but it introduced hand-chosen direction hyperparameters.")
    lines.append("v4c removes those weights and uses calibration mean probability only.")
    lines.append("```")

    lines.append("\n## Eps rule\n")
    lines.append("```text")
    lines.append("eps_raw = 0.5 * max(0, cal_error - train_error)")
    lines.append("        + 0.5 * max(0, cal_other_mass - train_other_mass)")
    lines.append("eps_used = min(max_eps, eps_raw)")
    lines.append("```")

    lines.append("\n## Direction rule\n")
    lines.append("```text")
    lines.append("For other malware class k:")
    lines.append("score_k = calibration_mean_prob_k")
    lines.append("target_k = eps_used * score_k / sum(score_other_malware)")
    lines.append("```")

    lines.append("\n## Family gap evidence\n")
    cols = [c for c in [
        "true_L2", "true_L3", "calibration_n", "train_inner_n",
        "calibration_error_rate", "train_inner_error_rate", "error_gap",
        "calibration_other_malware_prob_mass_mean", "train_inner_other_malware_prob_mass_mean", "other_malware_mass_gap",
        "eps_raw_gap_avg",
    ] if c in ev.columns]
    lines.append(ev[ev["true_L2"].isin(malware_classes)].sort_values("eps_raw_gap_avg", ascending=False)[cols].to_markdown(index=False))

    lines.append("\n## Locked matrix\n")
    cols2 = [c for c in [
        "true_L2", "true_L3", "n_calibration", "eps_raw", "eps_used",
        "old_v4_eps_family", "old_v4b_eps_used",
        "target_Benign", "target_Ransomware", "target_Spyware", "target_Trojan",
        "source",
    ] if c in matrix.columns]
    lines.append(matrix[cols2].to_markdown(index=False))

    mal = matrix[matrix["true_L2"].isin(malware_classes)].copy()
    lines.append("\n## Sanity summary\n")
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
    lines.append("Use this v4c matrix for F1e1b if sanity checks pass.")
    lines.append("Do not tune using validation.")
    lines.append("```")
    (out_dir / "F1e1a_v4c_report.md").write_text("\n".join(lines), encoding="utf-8")


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
    ap.add_argument("--v4b-dir", default="05_test/outputs/F1e1a_v4b_corrected_calibration_family_matrix")
    ap.add_argument("--out-dir", default="05_test/outputs/F1e1a_v4c_prob_only_calibration_family_matrix")
    ap.add_argument("--combined-zip", default="05_test/outputs/F1e1a_v4c_prob_only_calibration_family_matrix.zip")
    ap.add_argument("--class-names", default="Benign,Ransomware,Spyware,Trojan")
    ap.add_argument("--malware-classes", default="Ransomware,Spyware,Trojan")
    ap.add_argument("--max-eps", type=float, default=0.30)
    ap.add_argument("--min-family-support", type=int, default=30)
    args = ap.parse_args()

    root = repo_root_from_here()
    v4_dir = resolve_path(args.v4_dir, root)
    v4b_dir = resolve_path(args.v4b_dir, root)
    out_dir = resolve_path(args.out_dir, root)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = resolve_path(args.combined_zip, root)

    class_names = parse_list(args.class_names) or DEFAULT_CLASSES
    malware_classes = parse_list(args.malware_classes) or DEFAULT_MALWARE

    log(f"root={root}")
    log(f"v4_dir={v4_dir}")
    log(f"out_dir={out_dir}")
    log("No training. No validation. Direction allocation = calibration mean probabilities only.")

    fam = load_family_summary(v4_dir)
    ev = build_gap_evidence(fam, class_names)
    ev.to_csv(out_dir / "F1e1a_v4c_family_gap_evidence.csv", index=False)

    matrix = derive_matrix_prob_only(ev, class_names, malware_classes, args.max_eps, args.min_family_support)

    old_v4 = v4_dir / "F1e1a_v4_locked_family_smoothing_matrix_CALIBRATION_DERIVED.csv"
    matrix = compare_old(matrix, old_v4, "old_v4", "old_v4_eps_family")

    old_v4b = v4b_dir / "F1e1a_v4b_locked_family_smoothing_matrix_CORRECTED_CALIBRATION_DERIVED.csv"
    matrix = compare_old(matrix, old_v4b, "old_v4b", "old_v4b_eps_used")

    matrix.to_csv(out_dir / "F1e1a_v4c_locked_family_smoothing_matrix_PROB_ONLY_CALIBRATION_DERIVED.csv", index=False)

    targets = {
        "usage": "Use this locked prob-only calibration-derived matrix for F1e1b if sanity checks pass.",
        "validation_used": False,
        "training_done_in_v4c": False,
        "fake_data_used": False,
        "source_v4_dir": str(v4_dir),
        "class_names": class_names,
        "malware_classes": malware_classes,
        "eps_rule": "0.5*gap(error_rate)+0.5*gap(other_malware_prob_mass)",
        "direction_rule": "prob_only: allocation proportional to calibration mean_prob_other",
        "max_eps": float(args.max_eps),
        "min_family_support": int(args.min_family_support),
        "matrix_file": "F1e1a_v4c_locked_family_smoothing_matrix_PROB_ONLY_CALIBRATION_DERIVED.csv",
        "matrix": matrix.to_dict(orient="records"),
    }
    (out_dir / "F1e1a_v4c_locked_family_smoothing_targets.json").write_text(json.dumps(targets, indent=2), encoding="utf-8")

    write_policy(out_dir)
    write_report(out_dir, ev, matrix, malware_classes)
    zip_dir(out_dir, zip_path)

    mal = matrix[matrix["true_L2"].isin(malware_classes)].copy()
    if len(mal):
        print(mal[["true_L2", "true_L3", "eps_raw", "eps_used", "target_Ransomware", "target_Spyware", "target_Trojan"]].sort_values("eps_used", ascending=False).to_string(index=False), flush=True)
    log(f"zip={zip_path}")
    log("DONE")


if __name__ == "__main__":
    main()
