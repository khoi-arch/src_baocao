#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F1e1a Audit-to-Smoothing Feasibility

Purpose
-------
Do NOT train.

Use existing overlap/root-cause audit outputs to decide whether a
malware-subtype label-smoothing target is justified, and if yes, derive a
candidate directed pair-aware smoothing matrix.

This script intentionally does NOT choose smoothing from raw aggregate counts only.

It combines:
- train-vs-val directed wrong-pair gap
- CLS kNN pull toward predicted class
- centroid pull toward predicted class
- centroid margin true-minus-pred
- confidence of wrong predictions
- optional raw/token context as diagnostic evidence

Outputs
-------
- F1e1a_pair_evidence_table.csv
- F1e1a_pair_severity_scores.csv
- F1e1a_class_feasibility_summary.csv
- F1e1a_smoothing_matrix_fixed_eps.csv
- F1e1a_smoothing_matrix_adaptive_eps.csv
- F1e1a_smoothing_targets_json.json
- F1e1a_report.md
- combined zip

Interpretation
--------------
This is a feasibility/derivation step.

If the derived L2 pair-aware matrix is high confidence:
    proceed to F1e1b train with this fixed matrix.

If pair evidence is mixed or low-confidence:
    do NOT train smoothing yet; run L3/family-level audit first.
"""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_MALWARE = ["Ransomware", "Spyware", "Trojan"]
DEFAULT_BENIGN = "Benign"


def log(msg: str) -> None:
    print(f"[F1e1a] {msg}", flush=True)


def repo_root_from_here() -> Path:
    p = Path(__file__).resolve()
    if p.parent.name == "05_test":
        return p.parents[1]
    return Path.cwd().resolve()


def resolve_path(p: str | Path, root: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (root / p).resolve()


def clean_class(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def clean_df_classes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["true_class", "pred_class", "class"]:
        if c in df.columns:
            df[c] = df[c].map(clean_class)
    return df


def find_file(audit_dir: Path, filename: str) -> Optional[Path]:
    direct = audit_dir / filename
    if direct.exists():
        return direct
    hits = sorted(audit_dir.rglob(filename))
    return hits[0] if hits else None


def read_csv_if_exists(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None or not path.exists():
        return None
    df = pd.read_csv(path)
    return clean_df_classes(df)


def safe_num(s, default=0.0):
    try:
        if pd.isna(s):
            return default
        return float(s)
    except Exception:
        return default


def positive_norm(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower=0.0)
    mx = float(x.max())
    if mx <= 0:
        return pd.Series(np.zeros(len(x)), index=series.index)
    return x / mx


def minmax_norm(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").fillna(0.0)
    mn, mx = float(x.min()), float(x.max())
    if abs(mx - mn) < 1e-12:
        return pd.Series(np.zeros(len(x)), index=series.index)
    return (x - mn) / (mx - mn)


def log_support_norm(series: pd.Series) -> pd.Series:
    x = np.log1p(pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower=0.0))
    mx = float(x.max())
    if mx <= 0:
        return pd.Series(np.zeros(len(x)), index=series.index)
    return x / mx


def load_inputs(audit_dir: Path) -> Dict[str, object]:
    files = {
        "wrong_pair_centroid_summary": find_file(audit_dir, "wrong_pair_centroid_summary.csv"),
        "class_centroid_shift": find_file(audit_dir, "class_centroid_shift.csv"),
        "wrong_pair_cross_space_rootcause_summary": find_file(audit_dir, "wrong_pair_cross_space_rootcause_summary.csv"),
        "val_cross_space_rootcause_per_sample": find_file(audit_dir, "val_cross_space_rootcause_per_sample.csv"),
        "val_predictions": find_file(audit_dir, "val_predictions.csv"),
        "train_predictions": find_file(audit_dir, "train_predictions.csv"),
        "summary_md": find_file(audit_dir, "summary.md"),
        "cls_capture_info": find_file(audit_dir, "cls_capture_info.json"),
    }
    dfs = {k: read_csv_if_exists(v) for k, v in files.items() if k not in {"summary_md", "cls_capture_info"}}
    meta = {"files": {k: (str(v) if v else None) for k, v in files.items()}}
    return {"files": files, "dfs": dfs, "meta": meta}


def pair_rates_from_predictions(train_pred: Optional[pd.DataFrame], val_pred: Optional[pd.DataFrame],
                                malware_classes: List[str], benign_class: str) -> pd.DataFrame:
    classes = [benign_class] + [c for c in malware_classes if c != benign_class]
    rows = []

    def prep(df: pd.DataFrame, split: str) -> pd.DataFrame:
        d = df.copy()
        d["true_class"] = d["true_class"].map(clean_class)
        d["pred_class"] = d["pred_class"].map(clean_class)
        d["split"] = split
        return d

    if train_pred is None and val_pred is None:
        return pd.DataFrame()

    all_dfs = []
    if train_pred is not None:
        all_dfs.append(prep(train_pred, "train"))
    if val_pred is not None:
        all_dfs.append(prep(val_pred, "val"))

    for d in all_dfs:
        split = d["split"].iloc[0]
        supports = d.groupby("true_class").size().to_dict()
        wrong = d[d["true_class"] != d["pred_class"]]
        grp = wrong.groupby(["true_class", "pred_class"]).size().reset_index(name=f"{split}_n")
        for _, r in grp.iterrows():
            true_c = clean_class(r["true_class"])
            pred_c = clean_class(r["pred_class"])
            n = int(r[f"{split}_n"])
            support = int(supports.get(true_c, 0))
            rows.append({
                "split": split,
                "true_class": true_c,
                "pred_class": pred_c,
                f"{split}_n": n,
                f"{split}_support_true": support,
                f"{split}_rate_within_true": n / support if support else 0.0,
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Merge train and val views.
    keys = ["true_class", "pred_class"]
    train = df[df["split"] == "train"].drop(columns=["split"], errors="ignore")
    val = df[df["split"] == "val"].drop(columns=["split"], errors="ignore")
    out = pd.merge(train, val, on=keys, how="outer")
    for c in ["train_n", "train_support_true", "train_rate_within_true", "val_n", "val_support_true", "val_rate_within_true"]:
        if c not in out.columns:
            out[c] = 0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    out["val_minus_train_rate"] = out["val_rate_within_true"] - out["train_rate_within_true"]
    out["is_malware_pair"] = out["true_class"].isin(malware_classes) & out["pred_class"].isin(malware_classes) & (out["true_class"] != out["pred_class"])
    out["is_benign_involved"] = (out["true_class"] == benign_class) | (out["pred_class"] == benign_class)
    return out


def merge_evidence(pair_rate: pd.DataFrame, cross: Optional[pd.DataFrame], centroid: Optional[pd.DataFrame],
                   malware_classes: List[str]) -> pd.DataFrame:
    if pair_rate is None or len(pair_rate) == 0:
        # fallback from cross-space summary
        if cross is not None:
            pair_rate = cross[["true_class", "pred_class", "n"]].copy()
            pair_rate["val_n"] = pair_rate["n"]
            pair_rate["train_n"] = np.nan
            pair_rate["val_support_true"] = np.nan
            pair_rate["train_support_true"] = np.nan
            pair_rate["val_rate_within_true"] = np.nan
            pair_rate["train_rate_within_true"] = np.nan
            pair_rate["val_minus_train_rate"] = np.nan
        elif centroid is not None:
            pair_rate = centroid[["true_class", "pred_class", "n"]].copy()
            pair_rate["val_n"] = pair_rate["n"]
            pair_rate["train_n"] = np.nan
            pair_rate["val_support_true"] = np.nan
            pair_rate["train_support_true"] = np.nan
            pair_rate["val_rate_within_true"] = np.nan
            pair_rate["train_rate_within_true"] = np.nan
            pair_rate["val_minus_train_rate"] = np.nan
        else:
            return pd.DataFrame()

    out = pair_rate.copy()
    keys = ["true_class", "pred_class"]

    if cross is not None and len(cross):
        cols = [c for c in cross.columns if c not in {"n"}]
        out = pd.merge(out, cross[cols], on=keys, how="left", suffixes=("", "_cross"))

    if centroid is not None and len(centroid):
        cols = [c for c in centroid.columns if c not in {"n"}]
        out = pd.merge(out, centroid[cols], on=keys, how="left", suffixes=("", "_centroid"))

    # Only directed malware wrong pairs for target derivation.
    out["is_directed_malware_pair"] = (
        out["true_class"].isin(malware_classes)
        & out["pred_class"].isin(malware_classes)
        & (out["true_class"] != out["pred_class"])
    )

    # Fill key columns.
    for c in [
        "val_n", "train_n", "val_minus_train_rate", "val_rate_within_true", "train_rate_within_true",
        "confidence_mean",
        "cls_classifier_input_feature_space_overlap_with_pred_class_rate",
        "cls_classifier_input_model_boundary_failure_knn_true_neighbors_rate",
        "cls_classifier_input_OOD_or_distribution_shift_rate",
        "cls_classifier_input_mixed_neighbors_ambiguous_rate",
        "cls_classifier_input_true_frac_mean",
        "cls_classifier_input_pred_frac_mean",
        "raw_scaled_feature_space_overlap_with_pred_class_rate",
        "token_bin_offset_feature_space_overlap_with_pred_class_rate",
        "pred_closer_than_true_centroid_rate",
        "centroid_margin_true_minus_pred_mean",
    ]:
        if c not in out.columns:
            out[c] = np.nan

    return out


def compute_severity(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    m = d["is_directed_malware_pair"].fillna(False)
    work = d[m].copy()

    if len(work) == 0:
        d["severity_score"] = 0.0
        return d

    work["support_score"] = log_support_norm(work["val_n"])
    # If train gap unavailable, fallback to val rate, but mark it.
    if work["val_minus_train_rate"].notna().any():
        work["val_gap_score"] = positive_norm(work["val_minus_train_rate"])
    else:
        work["val_gap_score"] = positive_norm(work["val_rate_within_true"])

    cls_overlap = pd.to_numeric(work["cls_classifier_input_feature_space_overlap_with_pred_class_rate"], errors="coerce")
    cls_pred_frac = pd.to_numeric(work["cls_classifier_input_pred_frac_mean"], errors="coerce")
    cls_true_frac = pd.to_numeric(work["cls_classifier_input_true_frac_mean"], errors="coerce")
    cls_pred_minus_true = (cls_pred_frac - cls_true_frac).clip(lower=0.0)
    work["cls_overlap_score"] = cls_overlap.fillna(0.0).clip(0, 1)
    work["cls_pred_frac_score"] = cls_pred_frac.fillna(0.0).clip(0, 1)
    work["cls_pred_minus_true_score"] = positive_norm(cls_pred_minus_true)
    work["cls_pull_score"] = (
        0.45 * work["cls_overlap_score"]
        + 0.35 * work["cls_pred_frac_score"]
        + 0.20 * work["cls_pred_minus_true_score"]
    ).clip(0, 1)

    work["centroid_pred_closer_score"] = pd.to_numeric(work["pred_closer_than_true_centroid_rate"], errors="coerce").fillna(0.0).clip(0, 1)
    work["centroid_margin_score"] = positive_norm(work["centroid_margin_true_minus_pred_mean"])

    conf = pd.to_numeric(work["confidence_mean"], errors="coerce").fillna(0.5)
    work["wrong_confidence_score"] = ((conf - 0.5) / 0.5).clip(0, 1)

    raw_overlap = pd.to_numeric(work["raw_scaled_feature_space_overlap_with_pred_class_rate"], errors="coerce").fillna(0.0).clip(0, 1)
    tok_overlap = pd.to_numeric(work["token_bin_offset_feature_space_overlap_with_pred_class_rate"], errors="coerce").fillna(0.0).clip(0, 1)
    work["input_overlap_context_score"] = (0.5 * raw_overlap + 0.5 * tok_overlap).clip(0, 1)

    # Main severity: not just counts. Representation pull and train-val gap matter.
    work["severity_score"] = (
        0.18 * work["support_score"]
        + 0.22 * work["val_gap_score"]
        + 0.25 * work["cls_pull_score"]
        + 0.18 * work["centroid_pred_closer_score"]
        + 0.12 * work["centroid_margin_score"]
        + 0.05 * work["wrong_confidence_score"]
    )

    # A stricter multiplicative overfit-boundary evidence score.
    work["boundary_overfit_evidence_score"] = (
        (0.5 + 0.5 * work["support_score"])
        * (0.5 + 0.5 * work["val_gap_score"])
        * (0.5 + 0.5 * work["cls_pull_score"])
        * (0.5 + 0.5 * work["centroid_pred_closer_score"])
        * (0.5 + 0.5 * work["centroid_margin_score"])
    )

    # Merge back.
    for c in [
        "support_score", "val_gap_score", "cls_overlap_score", "cls_pred_frac_score",
        "cls_pred_minus_true_score", "cls_pull_score", "centroid_pred_closer_score",
        "centroid_margin_score", "wrong_confidence_score", "input_overlap_context_score",
        "severity_score", "boundary_overfit_evidence_score",
    ]:
        d[c] = np.nan
        d.loc[work.index, c] = work[c]

    d["severity_score"] = d["severity_score"].fillna(0.0)
    d["boundary_overfit_evidence_score"] = d["boundary_overfit_evidence_score"].fillna(0.0)
    return d


def entropy_norm(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    w = w[w > 0]
    if len(w) <= 1:
        return 0.0
    p = w / w.sum()
    h = -float(np.sum(p * np.log(p)))
    return h / math.log(len(p))


def derive_matrices(sev: pd.DataFrame, malware_classes: List[str], benign_class: str,
                    fixed_eps: float, adaptive_min_eps: float, adaptive_max_eps: float):
    # Class-level evidence based on positive train-val gap mass and severity mass.
    rows_class = []
    for true_c in malware_classes:
        sub = sev[(sev["true_class"] == true_c) & (sev["pred_class"].isin(malware_classes)) & (sev["pred_class"] != true_c)].copy()
        total_sev = float(sub["severity_score"].sum()) if len(sub) else 0.0
        gap_sum = float(pd.to_numeric(sub["val_minus_train_rate"], errors="coerce").fillna(0.0).clip(lower=0.0).sum()) if len(sub) else 0.0
        val_wrong_rate_sum = float(pd.to_numeric(sub["val_rate_within_true"], errors="coerce").fillna(0.0).sum()) if len(sub) else 0.0
        weights = sub["severity_score"].to_numpy(dtype=float) if len(sub) else np.array([])
        if weights.sum() > 0:
            top_share = float(weights.max() / weights.sum())
            ent = entropy_norm(weights)
        else:
            top_share, ent = 0.0, 0.0
        top_pair = ""
        if len(sub):
            top = sub.sort_values("severity_score", ascending=False).iloc[0]
            top_pair = f"{top['true_class']}->{top['pred_class']}"
        rows_class.append({
            "true_class": true_c,
            "severity_sum": total_sev,
            "positive_val_minus_train_gap_sum": gap_sum,
            "val_wrong_rate_to_other_malware_sum": val_wrong_rate_sum,
            "top_pair": top_pair,
            "top_pair_share_of_severity": top_share,
            "severity_entropy_norm": ent,
            "direction_specificity": 1.0 - ent,
            "n_candidate_pairs": int(len(sub)),
        })

    class_df = pd.DataFrame(rows_class)
    max_gap_sum = float(class_df["positive_val_minus_train_gap_sum"].max()) if len(class_df) else 0.0
    if max_gap_sum <= 1e-12:
        class_df["adaptive_eps"] = adaptive_min_eps
    else:
        class_df["adaptive_eps"] = adaptive_min_eps + (adaptive_max_eps - adaptive_min_eps) * (class_df["positive_val_minus_train_gap_sum"] / max_gap_sum)
    class_df["adaptive_eps"] = class_df["adaptive_eps"].clip(adaptive_min_eps, adaptive_max_eps)

    all_classes = [benign_class] + malware_classes

    def build_matrix(mode: str) -> pd.DataFrame:
        rows = []
        for true_c in all_classes:
            row = {"true_class": true_c, "mode": mode}
            for c in all_classes:
                row[f"target_{c}"] = 0.0

            if true_c == benign_class:
                row[f"target_{benign_class}"] = 1.0
                row["eps_used"] = 0.0
                row["source"] = "benign_one_hot_no_smoothing"
                rows.append(row)
                continue

            sub = sev[(sev["true_class"] == true_c) & (sev["pred_class"].isin(malware_classes)) & (sev["pred_class"] != true_c)].copy()
            if mode == "fixed_eps":
                eps = fixed_eps
            else:
                eps = float(class_df.loc[class_df["true_class"] == true_c, "adaptive_eps"].iloc[0])

            if len(sub) == 0 or float(sub["severity_score"].sum()) <= 0:
                row[f"target_{true_c}"] = 1.0
                row["eps_used"] = 0.0
                row["source"] = "no_pair_evidence"
            else:
                total = float(sub["severity_score"].sum())
                row[f"target_{true_c}"] = 1.0 - eps
                for _, r in sub.iterrows():
                    pred_c = clean_class(r["pred_class"])
                    row[f"target_{pred_c}"] = eps * float(r["severity_score"]) / total
                row[f"target_{benign_class}"] = 0.0
                row["eps_used"] = eps
                row["source"] = "directed_pair_severity_weighted"
            rows.append(row)

        mat = pd.DataFrame(rows)
        target_cols = [f"target_{c}" for c in all_classes]
        mat["target_sum"] = mat[target_cols].sum(axis=1)
        return mat

    fixed = build_matrix("fixed_eps")
    adaptive = build_matrix("adaptive_eps")
    return class_df, fixed, adaptive


def feasibility_tags(sev: pd.DataFrame, class_df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    df = class_df.copy()
    tags = []
    for _, r in df.iterrows():
        reasons = []
        if r["severity_sum"] <= 0:
            tag = "no_evidence"
            reasons.append("no directed pair severity")
        elif r["top_pair_share_of_severity"] >= 0.65:
            tag = "direction_specific_pair"
            reasons.append("one pair dominates severity")
        elif r["severity_entropy_norm"] >= 0.85:
            tag = "diffuse_pairs_need_L3_or_prob_audit"
            reasons.append("pair evidence is diffuse; family-level audit recommended")
        else:
            tag = "moderately_feasible_pair_aware"
            reasons.append("directed pair evidence exists but not sharply concentrated")
        if r["positive_val_minus_train_gap_sum"] <= 0.02:
            reasons.append("low positive val-train pair gap; smoothing strength should be conservative")
        tags.append((tag, "; ".join(reasons)))
    df["feasibility_tag"] = [x[0] for x in tags]
    df["feasibility_reason"] = [x[1] for x in tags]

    if (df["feasibility_tag"] == "diffuse_pairs_need_L3_or_prob_audit").any():
        overall = "L2_pair_matrix_possible_but_low_confidence_due_to_diffuse_pairs_need_L3_or_prob_audit"
    elif (df["feasibility_tag"] == "no_evidence").any():
        overall = "insufficient_evidence_for_some_classes"
    else:
        overall = "L2_pair_aware_smoothing_feasible_as_next_fixed_hypothesis"
    return df, overall


def write_report(out_dir: Path, meta: Dict[str, object], sev: pd.DataFrame, class_df: pd.DataFrame,
                 fixed: pd.DataFrame, adaptive: pd.DataFrame, overall: str,
                 malware_classes: List[str], benign_class: str):
    lines = []
    lines.append("# F1e1a Audit-to-Smoothing Feasibility Report\n")
    lines.append("## Purpose\n")
    lines.append("```text")
    lines.append("This step does not train a model.")
    lines.append("It uses existing root-cause/overlap audits to derive whether pair-aware malware label smoothing is justified.")
    lines.append("It explicitly avoids choosing smoothing from aggregate counts alone.")
    lines.append("```")

    lines.append("\n## Loaded files\n")
    lines.append("```json")
    lines.append(json.dumps(meta.get("files", {}), indent=2))
    lines.append("```")

    lines.append("\n## Directed malware pair severity\n")
    show_cols = [c for c in [
        "true_class", "pred_class", "val_n", "train_n",
        "val_rate_within_true", "train_rate_within_true", "val_minus_train_rate",
        "confidence_mean",
        "cls_classifier_input_feature_space_overlap_with_pred_class_rate",
        "cls_classifier_input_pred_frac_mean",
        "cls_classifier_input_true_frac_mean",
        "pred_closer_than_true_centroid_rate",
        "centroid_margin_true_minus_pred_mean",
        "support_score", "val_gap_score", "cls_pull_score",
        "centroid_pred_closer_score", "centroid_margin_score",
        "severity_score", "boundary_overfit_evidence_score",
    ] if c in sev.columns]
    malware_sev = sev[sev["is_directed_malware_pair"]].sort_values("severity_score", ascending=False)
    if len(malware_sev):
        lines.append(malware_sev[show_cols].to_markdown(index=False))
    else:
        lines.append("No directed malware-pair evidence found.")

    lines.append("\n## Class-level feasibility\n")
    lines.append(class_df.to_markdown(index=False))

    lines.append("\n## Candidate smoothing matrix: fixed total eps\n")
    lines.append(fixed.to_markdown(index=False))

    lines.append("\n## Candidate smoothing matrix: adaptive eps from val-train gap mass\n")
    lines.append(adaptive.to_markdown(index=False))

    lines.append("\n## Overall call\n")
    lines.append("```text")
    lines.append(f"overall_feasibility = {overall}")
    lines.append("```")

    lines.append("\n## How to use this\n")
    lines.append("```text")
    lines.append("If overall feasibility is high:")
    lines.append("  Use the fixed_eps or adaptive_eps matrix as a single fixed hypothesis in F1e1b.")
    lines.append("")
    lines.append("If pair evidence is diffuse:")
    lines.append("  Do not claim the matrix is precise. Run L3/family-level or probability/top2 audit first.")
    lines.append("")
    lines.append("This output is allowed to choose a direction and a candidate matrix.")
    lines.append("It is not proof that the smoothing hyperparameters are optimal.")
    lines.append("```")

    lines.append("\n## Important limitation\n")
    lines.append("```text")
    lines.append("These audits are L2 directed-pair and CLS/root-cause audits.")
    lines.append("They do not fully replace L3/family-level prediction-probability audit.")
    lines.append("If the goal is the strongest scientific target design, family-aware smoothing should be derived from L3/family evidence.")
    lines.append("```")

    (out_dir / "F1e1a_report.md").write_text("\n".join(lines), encoding="utf-8")


def zip_dir(src_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in src_dir.rglob("*"):
            if p.is_file() and p != zip_path:
                z.write(p, p.relative_to(src_dir.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-dir", default="03_outputs/00_data_for_test/03_audit_rootcause")
    ap.add_argument("--out-dir", default="05_test/outputs/F1e1a_audit_to_smoothing_feasibility")
    ap.add_argument("--combined-zip", default="05_test/outputs/F1e1a_audit_to_smoothing_feasibility.zip")
    ap.add_argument("--malware-classes", default="Ransomware,Spyware,Trojan")
    ap.add_argument("--benign-class", default="Benign")
    ap.add_argument("--fixed-eps", type=float, default=0.10)
    ap.add_argument("--adaptive-min-eps", type=float, default=0.03)
    ap.add_argument("--adaptive-max-eps", type=float, default=0.12)
    args = ap.parse_args()

    root = repo_root_from_here()
    audit_dir = resolve_path(args.audit_dir, root)
    out_dir = resolve_path(args.out_dir, root)
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_zip = resolve_path(args.combined_zip, root)

    malware_classes = [clean_class(x) for x in args.malware_classes.split(",") if clean_class(x)]
    benign_class = clean_class(args.benign_class)

    log(f"root={root}")
    log(f"audit_dir={audit_dir}")
    log(f"out_dir={out_dir}")
    log(f"malware_classes={malware_classes}, benign_class={benign_class}")
    log("No training. Deriving feasibility and candidate smoothing matrices from audit files.")

    if not audit_dir.exists():
        raise FileNotFoundError(f"audit dir not found: {audit_dir}")

    loaded = load_inputs(audit_dir)
    dfs = loaded["dfs"]
    meta = loaded["meta"]

    # Save file inventory.
    (out_dir / "F1e1a_loaded_files.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    pair_rate = pair_rates_from_predictions(
        train_pred=dfs.get("train_predictions"),
        val_pred=dfs.get("val_predictions"),
        malware_classes=malware_classes,
        benign_class=benign_class,
    )
    pair_rate.to_csv(out_dir / "F1e1a_pair_rates_from_predictions.csv", index=False)

    evidence = merge_evidence(
        pair_rate=pair_rate,
        cross=dfs.get("wrong_pair_cross_space_rootcause_summary"),
        centroid=dfs.get("wrong_pair_centroid_summary"),
        malware_classes=malware_classes,
    )
    evidence.to_csv(out_dir / "F1e1a_pair_evidence_table.csv", index=False)

    sev = compute_severity(evidence)
    sev_sorted = sev.sort_values("severity_score", ascending=False)
    sev_sorted.to_csv(out_dir / "F1e1a_pair_severity_scores.csv", index=False)

    class_df, fixed, adaptive = derive_matrices(
        sev=sev,
        malware_classes=malware_classes,
        benign_class=benign_class,
        fixed_eps=float(args.fixed_eps),
        adaptive_min_eps=float(args.adaptive_min_eps),
        adaptive_max_eps=float(args.adaptive_max_eps),
    )
    class_df, overall = feasibility_tags(sev, class_df)
    class_df.to_csv(out_dir / "F1e1a_class_feasibility_summary.csv", index=False)
    fixed.to_csv(out_dir / "F1e1a_smoothing_matrix_fixed_eps.csv", index=False)
    adaptive.to_csv(out_dir / "F1e1a_smoothing_matrix_adaptive_eps.csv", index=False)

    targets = {
        "overall_feasibility": overall,
        "classes": [benign_class] + malware_classes,
        "fixed_eps": {
            "eps": float(args.fixed_eps),
            "matrix": fixed.to_dict(orient="records"),
        },
        "adaptive_eps": {
            "min_eps": float(args.adaptive_min_eps),
            "max_eps": float(args.adaptive_max_eps),
            "matrix": adaptive.to_dict(orient="records"),
        },
        "class_feasibility": class_df.to_dict(orient="records"),
        "severity_formula": {
            "severity_score": "0.18 support + 0.22 val_gap + 0.25 cls_pull + 0.18 centroid_pred_closer + 0.12 centroid_margin + 0.05 wrong_confidence",
            "cls_pull_score": "0.45 cls_overlap_rate + 0.35 cls_pred_frac + 0.20 normalized(cls_pred_frac-true_frac)",
            "note": "The matrix is derived from directed evidence, not raw aggregate counts alone.",
        },
    }
    (out_dir / "F1e1a_smoothing_targets_json.json").write_text(json.dumps(targets, indent=2), encoding="utf-8")

    write_report(
        out_dir=out_dir,
        meta=meta,
        sev=sev_sorted,
        class_df=class_df,
        fixed=fixed,
        adaptive=adaptive,
        overall=overall,
        malware_classes=malware_classes,
        benign_class=benign_class,
    )

    zip_dir(out_dir, combined_zip)

    log("Top directed malware pairs:")
    cols = [c for c in ["true_class", "pred_class", "val_n", "val_minus_train_rate", "severity_score", "boundary_overfit_evidence_score"] if c in sev_sorted.columns]
    print(sev_sorted[sev_sorted["is_directed_malware_pair"]][cols].head(20).to_string(index=False), flush=True)
    log(f"overall_feasibility={overall}")
    log(f"zip={combined_zip}")
    log("DONE")


if __name__ == "__main__":
    main()
