#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F1e1a_v2 L1 Audit-to-Smoothing Feasibility

Purpose
-------
Do NOT train.

Derive a candidate malware-subtype smoothing matrix from the CURRENT anti-overfit
candidate, L1, not from the older base L=3 audit.

Why v2?
-------
If F1e1b will train on top of L1, the smoothing matrix must be based on L1's
behavior:
- L1 train/val confusion gap
- L1 val predicted probabilities/confidence
- optional F1e0 L1 FFN-delta evidence by original pair

This replaces the earlier F1e1a that read the old base root-cause audit by default.

Main evidence
-------------
For each directed malware pair true -> pred:
1. L1 val wrong count and rate within true class
2. L1 train wrong count and rate within true class
3. excess validation error:
       excess_gap = max(0, val_rate - train_rate)
4. L1 probability evidence if available:
       mean_prob_pred, mean_prob_true, mean_pred_minus_true
5. Optional F1e0 evidence:
       FFN delta margin on val wrong pair
       whether FFN tends to improve/hurt true-class margin

Candidate eps
-------------
For each true malware class c:
    eps_c = sum_j max(0, val_rate(c->j) - train_rate(c->j))

This directly ties smoothing strength to the L1 train-val excess pair error.
No fixed eps=0.10 is assumed.

Pair allocation inside eps_c is based on severity, not raw count alone.

Outputs
-------
- F1e1a_v2_l1_pair_evidence.csv
- F1e1a_v2_l1_pair_severity.csv
- F1e1a_v2_l1_class_eps_summary.csv
- F1e1a_v2_smoothing_matrix_excess_gap_eps.csv
- F1e1a_v2_smoothing_targets_json.json
- F1e1a_v2_report.md
- combined zip
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


DEFAULT_CLASS_NAMES = ["Benign", "Ransomware", "Spyware", "Trojan"]


def log(msg: str) -> None:
    print(f"[F1e1a_v2] {msg}", flush=True)


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


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def find_first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def detect_class_names_from_report(run_dir: Path) -> List[str]:
    for fn in ["val_classification_report_best.json", "train_classification_report_best.json"]:
        p = run_dir / fn
        if p.exists():
            try:
                rep = read_json(p)
                names = []
                for k, v in rep.items():
                    if isinstance(v, dict) and "support" in v and k not in {"macro avg", "weighted avg"}:
                        # Skip accuracy if represented weirdly
                        if k.lower() != "accuracy":
                            names.append(k)
                if names:
                    # Standard expected order if all present
                    std = [c for c in DEFAULT_CLASS_NAMES if c in names]
                    if len(std) == len(names):
                        return std
                    return names
            except Exception:
                pass
    return DEFAULT_CLASS_NAMES


def normalize_predictions(df: pd.DataFrame, names: List[str]) -> pd.DataFrame:
    """
    Robustly normalize prediction CSVs.

    Handles both formats:
      - numeric labels: y_true/y_pred = 0,1,2,3
      - class-name labels: y_true/y_pred or true_class/pred_class = Benign/Ransomware/...
      - mixed output files that already contain true_name/pred_name
    """
    d = df.copy()
    name_to_id = {str(n): i for i, n in enumerate(names)}

    def series_to_label_id(s: pd.Series, fallback_name_series: Optional[pd.Series] = None, colname: str = "") -> pd.Series:
        # First try numeric conversion.
        num = pd.to_numeric(s, errors="coerce")
        if num.notna().mean() >= 0.95:
            return num

        # Then try mapping class-name strings from the same column.
        mapped = s.map(lambda x: name_to_id.get(clean_class(x), np.nan))
        if mapped.notna().mean() >= 0.95:
            return mapped

        # Then try fallback name column.
        if fallback_name_series is not None:
            mapped2 = fallback_name_series.map(lambda x: name_to_id.get(clean_class(x), np.nan))
            if mapped2.notna().mean() >= 0.95:
                return mapped2

        # Last resort: if values look like floats stored as strings with spaces, numeric again after strip.
        stripped = s.map(lambda x: clean_class(x))
        num2 = pd.to_numeric(stripped, errors="coerce")
        if num2.notna().mean() >= 0.95:
            return num2

        bad_preview = s.head(10).tolist()
        raise ValueError(
            f"Cannot normalize label column {colname}. "
            f"Known names={names}. Preview={bad_preview}. Columns={list(d.columns)}"
        )

    def label_id_to_name(x):
        if pd.isna(x):
            return ""
        ix = int(x)
        return names[ix] if 0 <= ix < len(names) else str(ix)

    # Detect true/pred source columns.
    true_candidates = ["y_true", "true_label", "label", "target", "true"]
    pred_candidates = ["y_pred", "pred_label", "prediction", "pred", "predicted"]

    true_col = next((c for c in true_candidates if c in d.columns), None)
    pred_col = next((c for c in pred_candidates if c in d.columns), None)

    # Some files use true_class/pred_class names only.
    fallback_true_name = d["true_class"] if "true_class" in d.columns else (d["true_name"] if "true_name" in d.columns else None)
    fallback_pred_name = d["pred_class"] if "pred_class" in d.columns else (d["pred_name"] if "pred_name" in d.columns else None)

    if true_col is None:
        if fallback_true_name is None:
            raise ValueError(f"Cannot detect true label/name column in predictions: {list(d.columns)}")
        d["y_true"] = series_to_label_id(fallback_true_name, colname="true_name/fallback")
    else:
        d["y_true"] = series_to_label_id(d[true_col], fallback_true_name, colname=true_col)

    if pred_col is None:
        if fallback_pred_name is None:
            raise ValueError(f"Cannot detect pred label/name column in predictions: {list(d.columns)}")
        d["y_pred"] = series_to_label_id(fallback_pred_name, colname="pred_name/fallback")
    else:
        d["y_pred"] = series_to_label_id(d[pred_col], fallback_pred_name, colname=pred_col)

    # At this point no NaNs should remain.
    if d["y_true"].isna().any() or d["y_pred"].isna().any():
        bad = d[d["y_true"].isna() | d["y_pred"].isna()].head(10)
        raise ValueError(
            "NaN labels remain after normalization. "
            f"Bad preview={bad.to_dict(orient='records')}"
        )

    d["y_true"] = d["y_true"].astype(int)
    d["y_pred"] = d["y_pred"].astype(int)

    # Preserve existing class-name columns if valid, otherwise derive from ids.
    if "true_name" in d.columns:
        d["true_name"] = d["true_name"].map(clean_class)
        # If existing names are empty/non-standard, derive from ids.
        bad_true_names = ~d["true_name"].isin(names)
        if bad_true_names.mean() > 0.05:
            d["true_name"] = d["y_true"].map(label_id_to_name)
    elif "true_class" in d.columns:
        d["true_name"] = d["true_class"].map(clean_class)
        bad_true_names = ~d["true_name"].isin(names)
        if bad_true_names.mean() > 0.05:
            d["true_name"] = d["y_true"].map(label_id_to_name)
    else:
        d["true_name"] = d["y_true"].map(label_id_to_name)

    if "pred_name" in d.columns:
        d["pred_name"] = d["pred_name"].map(clean_class)
        bad_pred_names = ~d["pred_name"].isin(names)
        if bad_pred_names.mean() > 0.05:
            d["pred_name"] = d["y_pred"].map(label_id_to_name)
    elif "pred_class" in d.columns:
        d["pred_name"] = d["pred_class"].map(clean_class)
        bad_pred_names = ~d["pred_name"].isin(names)
        if bad_pred_names.mean() > 0.05:
            d["pred_name"] = d["y_pred"].map(label_id_to_name)
    else:
        d["pred_name"] = d["y_pred"].map(label_id_to_name)

    d["correct"] = d["y_true"] == d["y_pred"]

    # Probability columns.
    prob_cols = {}
    for i, name in enumerate(names):
        candidates = [
            f"prob_{name}",
            f"prob_{i}",
            f"p_{name}",
            f"p{i}",
            f"prob_class_{i}",
        ]
        c = next((x for x in candidates if x in d.columns), None)
        if c is not None:
            prob_cols[name] = c
    d.attrs["prob_cols"] = prob_cols

    if "confidence" not in d.columns:
        if prob_cols:
            d["confidence"] = d[list(prob_cols.values())].max(axis=1)
        else:
            d["confidence"] = np.nan
    else:
        d["confidence"] = pd.to_numeric(d["confidence"], errors="coerce")

    return d

def load_confusion_matrix(path: Optional[Path], names: List[str]) -> Optional[pd.DataFrame]:
    if path is None or not path.exists():
        return None
    df = pd.read_csv(path, index_col=0)
    # Normalize index/columns.
    df.index = [clean_class(x) for x in df.index]
    df.columns = [clean_class(x) for x in df.columns]
    # Some CSVs may use numeric labels.
    if not set(names).issubset(set(df.index)):
        try:
            df.index = [names[int(x)] if str(x).isdigit() and int(x) < len(names) else str(x) for x in df.index]
        except Exception:
            pass
    if not set(names).issubset(set(df.columns)):
        try:
            df.columns = [names[int(x)] if str(x).isdigit() and int(x) < len(names) else str(x) for x in df.columns]
        except Exception:
            pass
    # Reindex to known names where possible.
    existing_names = [n for n in names if n in df.index and n in df.columns]
    if existing_names:
        df = df.reindex(index=names, columns=names, fill_value=0)
    return df.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)


def confusion_from_predictions(d: pd.DataFrame, names: List[str]) -> pd.DataFrame:
    mat = pd.crosstab(d["true_name"], d["pred_name"])
    return mat.reindex(index=names, columns=names, fill_value=0).astype(int)


def pair_rates_from_confusions(train_cm: Optional[pd.DataFrame], val_cm: pd.DataFrame, names: List[str]) -> pd.DataFrame:
    rows = []
    for true_c in names:
        val_support = int(val_cm.loc[true_c].sum()) if true_c in val_cm.index else 0
        train_support = int(train_cm.loc[true_c].sum()) if train_cm is not None and true_c in train_cm.index else np.nan
        for pred_c in names:
            if true_c == pred_c:
                continue
            val_n = int(val_cm.loc[true_c, pred_c]) if true_c in val_cm.index and pred_c in val_cm.columns else 0
            train_n = int(train_cm.loc[true_c, pred_c]) if train_cm is not None and true_c in train_cm.index and pred_c in train_cm.columns else np.nan
            val_rate = val_n / val_support if val_support else 0.0
            train_rate = train_n / train_support if train_cm is not None and train_support and not pd.isna(train_support) else np.nan
            rows.append({
                "true_class": true_c,
                "pred_class": pred_c,
                "val_n": val_n,
                "val_support_true": val_support,
                "val_rate_within_true": val_rate,
                "train_n": train_n,
                "train_support_true": train_support,
                "train_rate_within_true": train_rate,
                "val_minus_train_rate": val_rate - train_rate if not pd.isna(train_rate) else np.nan,
                "excess_val_over_train_rate": max(0.0, val_rate - train_rate) if not pd.isna(train_rate) else max(0.0, val_rate),
            })
    return pd.DataFrame(rows)


def add_probability_evidence(pair_df: pd.DataFrame, val_pred: pd.DataFrame, names: List[str]) -> pd.DataFrame:
    d = pair_df.copy()
    prob_cols = val_pred.attrs.get("prob_cols", {})
    rows = []
    for _, r in d.iterrows():
        true_c = clean_class(r["true_class"])
        pred_c = clean_class(r["pred_class"])
        mask = (val_pred["true_name"] == true_c) & (val_pred["pred_name"] == pred_c)
        sub = val_pred[mask]
        row = {}
        if len(sub):
            row["val_pair_confidence_mean"] = float(pd.to_numeric(sub["confidence"], errors="coerce").mean())
            row["val_pair_confidence_median"] = float(pd.to_numeric(sub["confidence"], errors="coerce").median())
            if true_c in prob_cols and pred_c in prob_cols:
                pt = pd.to_numeric(sub[prob_cols[true_c]], errors="coerce")
                pp = pd.to_numeric(sub[prob_cols[pred_c]], errors="coerce")
                row["val_pair_prob_true_mean"] = float(pt.mean())
                row["val_pair_prob_pred_mean"] = float(pp.mean())
                row["val_pair_prob_pred_minus_true_mean"] = float((pp - pt).mean())
                # top2 check if all probs exist
                if len(prob_cols) == len(names):
                    probs = sub[[prob_cols[n] for n in names]].to_numpy(dtype=float)
                    order = np.argsort(-probs, axis=1)
                    true_idx = names.index(true_c)
                    pred_idx = names.index(pred_c)
                    row["val_pair_true_in_top2_rate"] = float(np.mean(order[:, :2] == true_idx))
                    row["val_pair_pred_top1_rate"] = float(np.mean(order[:, 0] == pred_idx))
                    row["val_pair_top2_gap_mean"] = float(np.mean(probs[np.arange(len(probs)), order[:, 0]] - probs[np.arange(len(probs)), order[:, 1]]))
            else:
                row["val_pair_prob_true_mean"] = np.nan
                row["val_pair_prob_pred_mean"] = np.nan
                row["val_pair_prob_pred_minus_true_mean"] = np.nan
        else:
            row["val_pair_confidence_mean"] = np.nan
            row["val_pair_confidence_median"] = np.nan
            row["val_pair_prob_true_mean"] = np.nan
            row["val_pair_prob_pred_mean"] = np.nan
            row["val_pair_prob_pred_minus_true_mean"] = np.nan
        rows.append(row)

    ev = pd.DataFrame(rows)
    return pd.concat([d.reset_index(drop=True), ev], axis=1)


def add_f1e0_evidence(pair_df: pd.DataFrame, f1e0_dir: Optional[Path]) -> pd.DataFrame:
    d = pair_df.copy()
    if f1e0_dir is None or not f1e0_dir.exists():
        d["f1e0_available"] = False
        return d

    p = f1e0_dir / "F1e0_val_ffn_delta_by_original_pair.csv"
    if not p.exists():
        d["f1e0_available"] = False
        return d

    f = pd.read_csv(p)
    # Expected columns: true_name, orig_pred_name, orig_correct, n, delta_margin_mean, frac_ffn_improves_true_margin, ...
    if "true_name" not in f.columns or "orig_pred_name" not in f.columns:
        d["f1e0_available"] = False
        return d
    f["true_class"] = f["true_name"].map(clean_class)
    f["pred_class"] = f["orig_pred_name"].map(clean_class)
    if "orig_correct" in f.columns:
        f = f[f["orig_correct"] == False].copy()

    keep = ["true_class", "pred_class"]
    for c in [
        "n", "delta_margin_mean", "delta_margin_median", "frac_ffn_improves_true_margin",
        "attn_margin_mean", "layerout_margin_mean",
    ]:
        if c in f.columns:
            keep.append(c)
    f = f[keep].copy()
    rename = {
        "n": "f1e0_pair_n",
        "delta_margin_mean": "f1e0_delta_margin_mean",
        "delta_margin_median": "f1e0_delta_margin_median",
        "frac_ffn_improves_true_margin": "f1e0_frac_ffn_improves_true_margin",
        "attn_margin_mean": "f1e0_attn_margin_mean",
        "layerout_margin_mean": "f1e0_layerout_margin_mean",
    }
    f = f.rename(columns=rename)
    out = pd.merge(d, f, on=["true_class", "pred_class"], how="left")
    out["f1e0_available"] = True
    return out


def positive_norm(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").fillna(0.0).clip(lower=0.0)
    mx = float(x.max())
    if mx <= 1e-12:
        return pd.Series(np.zeros(len(x)), index=s.index)
    return x / mx


def log_norm(s: pd.Series) -> pd.Series:
    x = np.log1p(pd.to_numeric(s, errors="coerce").fillna(0.0).clip(lower=0.0))
    mx = float(x.max())
    if mx <= 1e-12:
        return pd.Series(np.zeros(len(x)), index=s.index)
    return x / mx


def compute_l1_severity(pair_df: pd.DataFrame, malware_classes: List[str], benign_class: str) -> pd.DataFrame:
    d = pair_df.copy()
    d["is_directed_malware_pair"] = (
        d["true_class"].isin(malware_classes)
        & d["pred_class"].isin(malware_classes)
        & (d["true_class"] != d["pred_class"])
    )

    m = d["is_directed_malware_pair"]
    work = d[m].copy()
    if len(work) == 0:
        d["severity_score"] = 0.0
        return d

    work["support_score"] = log_norm(work["val_n"])
    work["val_rate_score"] = positive_norm(work["val_rate_within_true"])
    work["excess_gap_score"] = positive_norm(work["excess_val_over_train_rate"])

    # Probability/confidence boundary strength.
    if "val_pair_prob_pred_minus_true_mean" in work.columns and work["val_pair_prob_pred_minus_true_mean"].notna().any():
        work["prob_boundary_score"] = positive_norm(work["val_pair_prob_pred_minus_true_mean"])
    else:
        conf = pd.to_numeric(work.get("val_pair_confidence_mean", pd.Series(index=work.index, dtype=float)), errors="coerce").fillna(0.5)
        work["prob_boundary_score"] = ((conf - 0.5) / 0.5).clip(0, 1)

    # F1e0: if delta margin is negative for a wrong pair, FFN/layerout pushes true margin worse.
    if "f1e0_delta_margin_mean" in work.columns and work["f1e0_delta_margin_mean"].notna().any():
        work["ffn_wrong_pull_score"] = positive_norm(-pd.to_numeric(work["f1e0_delta_margin_mean"], errors="coerce").fillna(0.0))
        if "f1e0_frac_ffn_improves_true_margin" in work.columns:
            hurt_frac = 1.0 - pd.to_numeric(work["f1e0_frac_ffn_improves_true_margin"], errors="coerce").fillna(0.5)
            work["ffn_hurt_frac_score"] = hurt_frac.clip(0, 1)
        else:
            work["ffn_hurt_frac_score"] = 0.0
        work["ffn_evidence_score"] = (0.70 * work["ffn_wrong_pull_score"] + 0.30 * work["ffn_hurt_frac_score"]).clip(0, 1)
    else:
        work["ffn_wrong_pull_score"] = 0.0
        work["ffn_hurt_frac_score"] = 0.0
        work["ffn_evidence_score"] = 0.0

    # Current L1-target matrix should mainly follow L1 excess train-val gap.
    # Support and probability/evidence break ties.
    work["severity_score"] = (
        0.35 * work["excess_gap_score"]
        + 0.20 * work["val_rate_score"]
        + 0.15 * work["support_score"]
        + 0.15 * work["prob_boundary_score"]
        + 0.15 * work["ffn_evidence_score"]
    )

    work["severity_score"] = work["severity_score"].clip(lower=0.0)
    # Multiplicative evidence for ranking confidence.
    work["boundary_overfit_evidence_score"] = (
        (0.5 + 0.5 * work["excess_gap_score"])
        * (0.5 + 0.5 * work["val_rate_score"])
        * (0.5 + 0.5 * work["prob_boundary_score"])
        * (0.5 + 0.5 * work["ffn_evidence_score"])
    )

    for c in [
        "support_score", "val_rate_score", "excess_gap_score", "prob_boundary_score",
        "ffn_wrong_pull_score", "ffn_hurt_frac_score", "ffn_evidence_score",
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


def derive_excess_gap_matrix(sev: pd.DataFrame, names: List[str], malware_classes: List[str], benign_class: str,
                             eps_cap: float, min_pair_severity: float):
    rows_class = []
    for true_c in malware_classes:
        sub = sev[(sev["true_class"] == true_c) & (sev["pred_class"].isin(malware_classes)) & (sev["pred_class"] != true_c)].copy()
        sub_pos = sub[sub["excess_val_over_train_rate"].fillna(0) > 0].copy()
        eps_raw = float(sub_pos["excess_val_over_train_rate"].sum()) if len(sub_pos) else 0.0
        eps_used = min(float(eps_cap), eps_raw)
        severity_sum = float(sub_pos["severity_score"].sum()) if len(sub_pos) else 0.0
        if severity_sum > 0:
            top = sub_pos.sort_values("severity_score", ascending=False).iloc[0]
            top_pair = f"{top['true_class']}->{top['pred_class']}"
            weights = sub_pos["severity_score"].to_numpy(dtype=float)
            top_share = float(weights.max() / weights.sum())
            ent = entropy_norm(weights)
        else:
            top_pair, top_share, ent = "", 0.0, 0.0
        rows_class.append({
            "true_class": true_c,
            "eps_raw_sum_excess_val_minus_train": eps_raw,
            "eps_cap": eps_cap,
            "eps_used": eps_used,
            "severity_sum_positive_excess": severity_sum,
            "top_pair": top_pair,
            "top_pair_share_of_severity": top_share,
            "severity_entropy_norm": ent,
            "direction_specificity": 1.0 - ent,
            "n_positive_excess_pairs": int(len(sub_pos)),
        })

    class_df = pd.DataFrame(rows_class)

    rows = []
    for true_c in names:
        row = {"true_class": true_c, "source": ""}
        for c in names:
            row[f"target_{c}"] = 0.0

        if true_c == benign_class:
            row[f"target_{benign_class}"] = 1.0
            row["eps_used"] = 0.0
            row["source"] = "benign_one_hot_no_smoothing"
            rows.append(row)
            continue

        if true_c not in malware_classes:
            row[f"target_{true_c}"] = 1.0
            row["eps_used"] = 0.0
            row["source"] = "non_malware_or_unknown_one_hot"
            rows.append(row)
            continue

        eps_used = float(class_df.loc[class_df["true_class"] == true_c, "eps_used"].iloc[0])
        sub = sev[
            (sev["true_class"] == true_c)
            & (sev["pred_class"].isin(malware_classes))
            & (sev["pred_class"] != true_c)
            & (sev["excess_val_over_train_rate"].fillna(0) > 0)
            & (sev["severity_score"].fillna(0) >= min_pair_severity)
        ].copy()

        if eps_used <= 0 or len(sub) == 0 or float(sub["severity_score"].sum()) <= 0:
            row[f"target_{true_c}"] = 1.0
            row["eps_used"] = 0.0
            row["source"] = "no_positive_l1_excess_gap_pair_evidence"
        else:
            total = float(sub["severity_score"].sum())
            row[f"target_{true_c}"] = 1.0 - eps_used
            for _, r in sub.iterrows():
                pred_c = clean_class(r["pred_class"])
                row[f"target_{pred_c}"] = eps_used * float(r["severity_score"]) / total
            row[f"target_{benign_class}"] = 0.0
            row["eps_used"] = eps_used
            row["source"] = "L1_excess_val_minus_train_gap_eps_and_L1_pair_severity_allocation"
        rows.append(row)

    mat = pd.DataFrame(rows)
    target_cols = [f"target_{c}" for c in names]
    mat["target_sum"] = mat[target_cols].sum(axis=1)
    return class_df, mat


def add_feasibility_tags(class_df: pd.DataFrame, sev: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    d = class_df.copy()
    tags = []
    for _, r in d.iterrows():
        if r["eps_used"] <= 1e-9:
            tag = "no_L1_excess_gap_no_smoothing"
            reason = "no positive val-train excess gap for this class"
        elif r["eps_used"] > 0.15:
            tag = "eps_high_check_risk"
            reason = "derived eps is high; cap or family audit may be safer"
        elif r["severity_entropy_norm"] >= 0.85:
            tag = "diffuse_pair_evidence_L3_recommended"
            reason = "severity spread across pairs; L3/family audit recommended before strong claim"
        elif r["top_pair_share_of_severity"] >= 0.65:
            tag = "direction_specific_smoothing_feasible"
            reason = "one directed pair dominates L1 excess gap/severity"
        else:
            tag = "moderately_feasible_pair_aware"
            reason = "L1 excess gap evidence exists but not fully concentrated"
        tags.append((tag, reason))
    d["feasibility_tag"] = [t for t, _ in tags]
    d["feasibility_reason"] = [r for _, r in tags]

    if (d["feasibility_tag"] == "eps_high_check_risk").any():
        overall = "L1_pair_aware_smoothing_possible_but_eps_high_check_cap_or_L3"
    elif (d["feasibility_tag"] == "diffuse_pair_evidence_L3_recommended").any():
        overall = "L1_pair_aware_smoothing_possible_but_L3_recommended_due_to_diffuse_pairs"
    elif (d["eps_used"] > 0).any():
        overall = "L1_pair_aware_smoothing_feasible_as_next_fixed_hypothesis"
    else:
        overall = "no_L1_excess_gap_evidence_for_smoothing"
    return d, overall


def write_report(out_dir: Path, meta: Dict, sev: pd.DataFrame, class_df: pd.DataFrame, mat: pd.DataFrame, overall: str,
                 names: List[str], malware_classes: List[str], benign_class: str):
    lines = []
    lines.append("# F1e1a_v2 L1 Audit-to-Smoothing Feasibility Report\n")
    lines.append("## Purpose\n")
    lines.append("```text")
    lines.append("This step does not train a model.")
    lines.append("It derives a candidate smoothing matrix from L1 outputs, because F1e1b would train on top of L1.")
    lines.append("It does not use the old base L=3 root-cause audit as the primary matrix source.")
    lines.append("```")

    lines.append("\n## Loaded files\n")
    lines.append("```json")
    lines.append(json.dumps(meta, indent=2))
    lines.append("```")

    lines.append("\n## Why L1 source matters\n")
    lines.append("```text")
    lines.append("Base L=3 audit can justify the general direction: malware subtype boundary overfit.")
    lines.append("But smoothing targets must reflect the current candidate model's errors.")
    lines.append("Therefore v2 uses L1 train/val confusion and L1 val prediction behavior.")
    lines.append("```")

    lines.append("\n## L1 directed malware pair severity\n")
    show_cols = [c for c in [
        "true_class", "pred_class",
        "val_n", "train_n",
        "val_rate_within_true", "train_rate_within_true",
        "val_minus_train_rate", "excess_val_over_train_rate",
        "val_pair_confidence_mean",
        "val_pair_prob_true_mean", "val_pair_prob_pred_mean", "val_pair_prob_pred_minus_true_mean",
        "f1e0_delta_margin_mean", "f1e0_frac_ffn_improves_true_margin",
        "support_score", "val_rate_score", "excess_gap_score", "prob_boundary_score", "ffn_evidence_score",
        "severity_score", "boundary_overfit_evidence_score",
    ] if c in sev.columns]
    sub = sev[sev["is_directed_malware_pair"]].sort_values("severity_score", ascending=False)
    if len(sub):
        lines.append(sub[show_cols].to_markdown(index=False))
    else:
        lines.append("No directed malware-pair evidence found.")

    lines.append("\n## Class eps derived from L1 excess validation error\n")
    lines.append("```text")
    lines.append("eps_c = sum_j max(0, val_rate_L1(c->j) - train_rate_L1(c->j))")
    lines.append("j only ranges over other malware classes.")
    lines.append("This ties smoothing strength directly to L1's generalization gap per true class.")
    lines.append("```")
    lines.append(class_df.to_markdown(index=False))

    lines.append("\n## Candidate smoothing matrix from L1 evidence\n")
    lines.append(mat.to_markdown(index=False))

    lines.append("\n## Overall call\n")
    lines.append("```text")
    lines.append(f"overall_feasibility = {overall}")
    lines.append("```")

    lines.append("\n## How to use this\n")
    lines.append("```text")
    lines.append("If overall feasibility is acceptable:")
    lines.append("  Train F1e1b with exactly this matrix as one fixed hypothesis.")
    lines.append("")
    lines.append("If eps is high or evidence diffuse:")
    lines.append("  Do not claim this is optimal. Run L3/family or full-probability audit first.")
    lines.append("")
    lines.append("This matrix is derived from L1 behavior, not chosen as uniform eps=0.10.")
    lines.append("```")

    lines.append("\n## Limitation\n")
    lines.append("```text")
    lines.append("This is still L2 pair-aware, not family-aware.")
    lines.append("If label_L3/family is available, family-aware target design is scientifically stronger.")
    lines.append("```")

    (out_dir / "F1e1a_v2_report.md").write_text("\n".join(lines), encoding="utf-8")


def zip_dir(src_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in src_dir.rglob("*"):
            if p.is_file() and p != zip_path:
                z.write(p, p.relative_to(src_dir.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l1-run-dir", default="05_test/outputs/F1a2_stage2_depth_classifier/Keff512/F1a2_L1_reduce_num_layers_strong")
    ap.add_argument("--f1e0-dir", default="05_test/outputs/F1e0_l1_ffn_contribution_audit")
    ap.add_argument("--out-dir", default="05_test/outputs/F1e1a_v2_l1_audit_to_smoothing_feasibility")
    ap.add_argument("--combined-zip", default="05_test/outputs/F1e1a_v2_l1_audit_to_smoothing_feasibility.zip")
    ap.add_argument("--class-names", default="Benign,Ransomware,Spyware,Trojan")
    ap.add_argument("--malware-classes", default="Ransomware,Spyware,Trojan")
    ap.add_argument("--benign-class", default="Benign")
    ap.add_argument("--eps-cap", type=float, default=0.20)
    ap.add_argument("--min-pair-severity", type=float, default=0.0)
    args = ap.parse_args()

    root = repo_root_from_here()
    l1_dir = resolve_path(args.l1_run_dir, root)
    f1e0_dir = resolve_path(args.f1e0_dir, root) if args.f1e0_dir else None
    out_dir = resolve_path(args.out_dir, root)
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_zip = resolve_path(args.combined_zip, root)

    if not l1_dir.exists():
        raise FileNotFoundError(f"L1 run dir not found: {l1_dir}")

    explicit_names = [clean_class(x) for x in args.class_names.split(",") if clean_class(x)]
    names = explicit_names or detect_class_names_from_report(l1_dir)
    malware_classes = [clean_class(x) for x in args.malware_classes.split(",") if clean_class(x)]
    benign_class = clean_class(args.benign_class)

    log(f"root={root}")
    log(f"l1_dir={l1_dir}")
    log(f"f1e0_dir={f1e0_dir}")
    log(f"out_dir={out_dir}")
    log(f"names={names}, malware_classes={malware_classes}, benign_class={benign_class}")
    log("No training. Deriving smoothing matrix from L1 outputs.")

    # Load L1 val predictions.
    val_pred_path = find_first_existing([
        l1_dir / "val_predictions_best.csv",
        l1_dir / "predictions" / "val_predictions.csv",
        l1_dir / "val_predictions.csv",
    ])
    if val_pred_path is None:
        raise FileNotFoundError(f"No L1 val predictions found in {l1_dir}")

    val_pred = normalize_predictions(pd.read_csv(val_pred_path), names)

    # Load train predictions if available; otherwise train confusion matrix.
    train_pred_path = find_first_existing([
        l1_dir / "train_predictions_best.csv",
        l1_dir / "predictions" / "train_predictions.csv",
        l1_dir / "train_predictions.csv",
    ])
    train_pred = normalize_predictions(pd.read_csv(train_pred_path), names) if train_pred_path is not None else None

    train_cm_path = find_first_existing([
        l1_dir / "train_confusion_matrix_best.csv",
        l1_dir / "reports" / "train_confusion_matrix.csv",
        l1_dir / "train_confusion_matrix.csv",
    ])
    val_cm_path = find_first_existing([
        l1_dir / "val_confusion_matrix_best.csv",
        l1_dir / "reports" / "val_confusion_matrix.csv",
        l1_dir / "val_confusion_matrix.csv",
    ])

    val_cm = load_confusion_matrix(val_cm_path, names) if val_cm_path is not None else confusion_from_predictions(val_pred, names)
    if train_pred is not None:
        train_cm = confusion_from_predictions(train_pred, names)
    else:
        train_cm = load_confusion_matrix(train_cm_path, names)

    if train_cm is None:
        log("WARNING: train confusion not found. train-rate gap will fallback to val rate only.")
    meta = {
        "l1_run_dir": str(l1_dir),
        "f1e0_dir": str(f1e0_dir) if f1e0_dir else None,
        "val_predictions": str(val_pred_path),
        "train_predictions": str(train_pred_path) if train_pred_path else None,
        "val_confusion_matrix": str(val_cm_path) if val_cm_path else "computed_from_val_predictions",
        "train_confusion_matrix": str(train_cm_path) if train_cm_path else ("computed_from_train_predictions" if train_pred is not None else None),
        "classes": names,
        "eps_rule": "eps_c = sum_j max(0, val_rate_L1(c->j) - train_rate_L1(c->j)); j is other malware classes",
        "eps_cap": float(args.eps_cap),
    }
    (out_dir / "F1e1a_v2_loaded_files.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Save normalized CMs.
    val_cm.to_csv(out_dir / "F1e1a_v2_l1_val_confusion_matrix_used.csv")
    if train_cm is not None:
        train_cm.to_csv(out_dir / "F1e1a_v2_l1_train_confusion_matrix_used.csv")

    pair_df = pair_rates_from_confusions(train_cm, val_cm, names)
    pair_df = add_probability_evidence(pair_df, val_pred, names)
    pair_df = add_f1e0_evidence(pair_df, f1e0_dir)
    pair_df.to_csv(out_dir / "F1e1a_v2_l1_pair_evidence.csv", index=False)

    sev = compute_l1_severity(pair_df, malware_classes, benign_class).sort_values("severity_score", ascending=False)
    sev.to_csv(out_dir / "F1e1a_v2_l1_pair_severity.csv", index=False)

    class_df, mat = derive_excess_gap_matrix(
        sev=sev,
        names=names,
        malware_classes=malware_classes,
        benign_class=benign_class,
        eps_cap=float(args.eps_cap),
        min_pair_severity=float(args.min_pair_severity),
    )
    class_df, overall = add_feasibility_tags(class_df, sev)
    class_df.to_csv(out_dir / "F1e1a_v2_l1_class_eps_summary.csv", index=False)
    mat.to_csv(out_dir / "F1e1a_v2_smoothing_matrix_excess_gap_eps.csv", index=False)

    targets = {
        "overall_feasibility": overall,
        "classes": names,
        "malware_classes": malware_classes,
        "benign_class": benign_class,
        "eps_rule": meta["eps_rule"],
        "eps_cap": float(args.eps_cap),
        "matrix": mat.to_dict(orient="records"),
        "class_eps_summary": class_df.to_dict(orient="records"),
        "severity_formula": {
            "severity_score": "0.35 excess_gap + 0.20 val_rate + 0.15 support + 0.15 prob_boundary + 0.15 f1e0_ffn_evidence",
            "excess_gap": "max(0, val_rate_L1 - train_rate_L1)",
            "prob_boundary": "mean(prob_pred - prob_true) if available; else confidence proxy",
            "f1e0_ffn_evidence": "optional wrong-pair FFN delta margin evidence from F1e0; 0 if unavailable",
        },
        "important_note": "This is L1-derived and L2 pair-aware. It is not family/L3-aware.",
    }
    (out_dir / "F1e1a_v2_smoothing_targets_json.json").write_text(json.dumps(targets, indent=2), encoding="utf-8")

    write_report(out_dir, meta, sev, class_df, mat, overall, names, malware_classes, benign_class)

    zip_dir(out_dir, combined_zip)

    log("Top L1 directed malware pairs:")
    cols = [c for c in [
        "true_class", "pred_class", "val_n", "train_n", "val_rate_within_true",
        "train_rate_within_true", "excess_val_over_train_rate", "severity_score"
    ] if c in sev.columns]
    print(sev[sev["is_directed_malware_pair"]][cols].head(20).to_string(index=False), flush=True)
    log(f"overall_feasibility={overall}")
    log(f"zip={combined_zip}")
    log("DONE")


if __name__ == "__main__":
    main()
