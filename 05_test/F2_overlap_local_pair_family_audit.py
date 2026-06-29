#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F2 Overlap Audit: official L1 local/pair/family overlap diagnostics.

No training. No validation-based hyperparameter selection.

Inputs:
  - official L1 model output directory containing val_predictions_best.csv
  - train_raw.csv, val_raw.csv
  - dataset.npz for token-space local-neighbor audit

Outputs:
  - per-family difficulty
  - malware pair confusion/margin/top2 audit
  - raw-space and token-space kNN overlap audit
  - pair feature transfer audit train -> val
  - centroid/local signal audit on stable pair features
  - markdown summary

This is diagnostic only. It may inspect official validation to understand error
structure, but it does not select or tune a model parameter.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


CLASS_NAMES_DEFAULT = ["Benign", "Ransomware", "Spyware", "Trojan"]
MALWARE_CLASSES_DEFAULT = ["Ransomware", "Spyware", "Trojan"]
LABEL_COL_CANDIDATES = [
    "label_L1", "label_L2", "label_L3", "Label_L1", "Label_L2", "Label_L3",
    "Class", "Category", "Family", "class", "category", "family",
    "MalwareFamily", "malware_family", "label", "target",
]


def log(msg: str):
    print(f"[F2audit] {msg}", flush=True)


def repo_root_from_here() -> Path:
    p = Path(__file__).resolve()
    if p.parent.name == "05_test":
        return p.parents[1]
    return Path.cwd().resolve()


def resolve_path(p: str | Path, root: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (root / p).resolve()


def parse_list(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def clean_label(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def find_label_col(df: pd.DataFrame, level: str) -> Optional[str]:
    if level == "L2":
        cands = ["label_L2", "Label_L2", "l2", "L2", "Category", "category", "Class", "class"]
    elif level == "L3":
        cands = ["label_L3", "Label_L3", "l3", "L3", "Family", "family", "MalwareFamily", "malware_family"]
    else:
        cands = LABEL_COL_CANDIDATES
    for c in cands:
        if c in df.columns:
            return c
    return None


def make_id_maps(class_names: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    name_to_id = {c: i for i, c in enumerate(class_names)}
    id_to_name = {i: c for i, c in enumerate(class_names)}
    return name_to_id, id_to_name


def label_to_id_series(s: pd.Series, class_names: List[str]) -> pd.Series:
    name_to_id, _ = make_id_maps(class_names)

    def one(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, np.integer)):
            return int(x)
        if isinstance(x, float) and x.is_integer():
            return int(x)
        t = str(x).strip()
        if t in name_to_id:
            return name_to_id[t]
        try:
            f = float(t)
            if f.is_integer():
                return int(f)
        except Exception:
            pass
        # Loose lower-case match.
        tl = t.lower()
        for k, v in name_to_id.items():
            if k.lower() == tl:
                return v
        return np.nan

    return s.map(one)


def infer_prob_cols(df: pd.DataFrame, class_names: List[str]) -> Tuple[List[str], str]:
    n = len(class_names)
    patterns = []
    patterns.append(([f"prob_{i}" for i in range(n)], "prob_i"))
    patterns.append(([f"proba_{i}" for i in range(n)], "proba_i"))
    patterns.append(([f"p{i}" for i in range(n)], "p_i"))
    patterns.append(([f"p_{i}" for i in range(n)], "p_i2"))
    patterns.append(([f"logit_{i}" for i in range(n)], "logit_i"))
    patterns.append(([f"prob_{c}" for c in class_names], "prob_class"))
    patterns.append(([f"proba_{c}" for c in class_names], "proba_class"))
    patterns.append(([f"p_{c}" for c in class_names], "p_class"))
    patterns.append(([f"{c}_prob" for c in class_names], "class_prob"))
    patterns.append(([f"{c}_proba" for c in class_names], "class_proba"))
    patterns.append(([f"{c}_logit" for c in class_names], "class_logit"))

    for cols, kind in patterns:
        if all(c in df.columns for c in cols):
            return cols, kind

    # Fuzzy: exactly n columns containing "prob" and matching class substrings.
    cols = []
    for cname in class_names:
        found = None
        for c in df.columns:
            cl = c.lower()
            if "prob" in cl or "proba" in cl or cl.startswith("p_"):
                if cname.lower() in cl:
                    found = c
                    break
        if found:
            cols.append(found)
    if len(cols) == n:
        return cols, "fuzzy_prob_class"

    return [], "none"


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    x = x - np.nanmax(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)


def load_predictions(pred_path: Path, raw_df: pd.DataFrame, class_names: List[str], split_name: str) -> pd.DataFrame:
    df = pd.read_csv(pred_path)
    out = pd.DataFrame({"row_id": np.arange(len(df), dtype=np.int64)})

    true_cols = ["y_true", "true", "true_id", "target", "target_id", "label", "label_id", "true_label"]
    pred_cols = ["y_pred", "pred", "pred_id", "prediction", "prediction_id", "pred_label"]

    true_col = next((c for c in true_cols if c in df.columns), None)
    pred_col = next((c for c in pred_cols if c in df.columns), None)

    if true_col is not None:
        out["true_id"] = label_to_id_series(df[true_col], class_names)
    else:
        l2 = find_label_col(raw_df, "L2")
        if l2 is None:
            raise ValueError(f"Cannot infer true labels for {split_name}: no prediction true column and no L2 label column in raw.")
        out["true_id"] = label_to_id_series(raw_df[l2].iloc[:len(df)].reset_index(drop=True), class_names)

    prob_cols, prob_kind = infer_prob_cols(df, class_names)
    probs = None
    if prob_cols:
        arr = df[prob_cols].to_numpy(dtype=float)
        if "logit" in prob_kind:
            probs = softmax_np(arr)
        else:
            # If values do not look normalized, softmax as fallback.
            row_sums = np.nansum(arr, axis=1)
            if np.nanmedian(row_sums) < 0.5 or np.nanmedian(row_sums) > 1.5 or np.nanmin(arr) < -1e-6:
                probs = softmax_np(arr)
            else:
                probs = arr / np.maximum(row_sums[:, None], 1e-12)
        for i, c in enumerate(class_names):
            out[f"prob_{c}"] = probs[:, i]

    if pred_col is not None:
        out["pred_id"] = label_to_id_series(df[pred_col], class_names)
    elif probs is not None:
        out["pred_id"] = np.nanargmax(probs, axis=1)
    else:
        raise ValueError(f"Cannot infer predicted labels for {split_name}: no pred column and no probabilities.")

    _, id_to_name = make_id_maps(class_names)
    out["true_id"] = out["true_id"].astype(int)
    out["pred_id"] = out["pred_id"].astype(int)
    out["true_label"] = out["true_id"].map(id_to_name)
    out["pred_label"] = out["pred_id"].map(id_to_name)

    if probs is not None:
        top_order = np.argsort(-probs, axis=1)
        out["top1_id"] = top_order[:, 0]
        out["top2_id"] = top_order[:, 1]
        out["top1_label"] = out["top1_id"].map(id_to_name)
        out["top2_label"] = out["top2_id"].map(id_to_name)
        out["top1_prob"] = probs[np.arange(len(out)), top_order[:, 0]]
        out["top2_prob"] = probs[np.arange(len(out)), top_order[:, 1]]
        out["margin_top1_top2"] = out["top1_prob"] - out["top2_prob"]
        out["true_prob"] = probs[np.arange(len(out)), out["true_id"].to_numpy()]
        out["pred_prob"] = probs[np.arange(len(out)), out["pred_id"].to_numpy()]
        out["true_rank"] = [int(np.where(top_order[i] == out.loc[i, "true_id"])[0][0]) + 1 for i in range(len(out))]
        out["true_in_top2"] = out["true_rank"] <= 2

    # Preserve columns from original predictions if useful for debugging.
    out.attrs["prediction_source"] = str(pred_path)
    out.attrs["prob_cols"] = prob_cols
    out.attrs["prob_kind"] = prob_kind
    return out


def find_model_dir(search_root: Path, preferred_keywords: List[str]) -> Optional[Path]:
    if not search_root.exists():
        return None
    candidates = []
    for p in search_root.rglob("val_predictions_best.csv"):
        d = p.parent
        score = 0
        name = str(d).lower()
        for kw in preferred_keywords:
            if kw.lower() in name:
                score += 10
        # Prefer base L1, avoid SAM/smoothing/lambda if possible.
        if "l1" in name:
            score += 5
        if "f1e2a" in name or "reproduce" in name or "base" in name:
            score += 4
        for bad in ["sam", "rho", "smoothing", "soft", "lambda", "subce", "overlap"]:
            if bad in name:
                score -= 4
        # Prefer dirs with config/checkpoint.
        if (d / "config.json").exists():
            score += 1
        if (d / "best_model.pt").exists():
            score += 1
        candidates.append((score, d))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def add_raw_labels(pred: pd.DataFrame, raw: pd.DataFrame, class_names: List[str]) -> pd.DataFrame:
    out = pred.copy()
    n = len(out)
    l2 = find_label_col(raw, "L2")
    l3 = find_label_col(raw, "L3")
    if l2:
        out["raw_L2"] = raw[l2].iloc[:n].map(clean_label).to_numpy()
    else:
        out["raw_L2"] = out["true_label"]
    if l3:
        out["raw_L3"] = raw[l3].iloc[:n].map(clean_label).to_numpy()
    else:
        out["raw_L3"] = out["raw_L2"]
    # Keep true_label authoritative from model/prediction.
    out["family"] = out["raw_L3"].where(out["raw_L3"].astype(str).str.len() > 0, out["true_label"])
    return out


def get_numeric_features(train_raw: pd.DataFrame, val_raw: pd.DataFrame, max_missing: float = 0.5) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    drop_cols = set(LABEL_COL_CANDIDATES)
    feature_cols = []
    train_num = {}
    val_num = {}
    for c in train_raw.columns:
        if c in drop_cols:
            continue
        tr = pd.to_numeric(train_raw[c], errors="coerce")
        va = pd.to_numeric(val_raw[c], errors="coerce") if c in val_raw.columns else None
        if va is None:
            continue
        miss = float(tr.isna().mean())
        if miss > max_missing:
            continue
        if tr.notna().sum() == 0:
            continue
        feature_cols.append(c)
        med = float(tr.median()) if tr.notna().any() else 0.0
        train_num[c] = tr.fillna(med).astype(float)
        val_num[c] = va.fillna(med).astype(float)
    return pd.DataFrame(train_num), pd.DataFrame(val_num), feature_cols


def basic_metrics(pred: pd.DataFrame, class_names: List[str], out_dir: Path) -> Dict[str, Any]:
    y = pred["true_label"].to_numpy()
    yh = pred["pred_label"].to_numpy()
    report = classification_report(y, yh, labels=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, yh, labels=class_names)
    cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in class_names], columns=[f"pred_{c}" for c in class_names])
    cm_df.to_csv(out_dir / "00_val_confusion_matrix.csv")
    pd.DataFrame(report).T.to_csv(out_dir / "00_val_classification_report.csv")
    return {
        "accuracy": float(accuracy_score(y, yh)),
        "macro_f1": float(f1_score(y, yh, labels=class_names, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, yh, labels=class_names, average="weighted", zero_division=0)),
        "support": int(len(pred)),
    }


def family_difficulty(pred: pd.DataFrame, class_names: List[str], out_dir: Path) -> pd.DataFrame:
    rows = []
    has_prob = "true_prob" in pred.columns
    for fam, g in pred.groupby("family", dropna=False):
        true_label_mode = g["true_label"].mode().iloc[0] if not g.empty else ""
        acc = float((g["true_label"] == g["pred_label"]).mean()) if len(g) else np.nan
        top_pred = g["pred_label"].value_counts().idxmax() if len(g) else ""
        row = {
            "family": fam,
            "true_label_mode": true_label_mode,
            "support": int(len(g)),
            "accuracy": acc,
            "error_rate": 1.0 - acc if not math.isnan(acc) else np.nan,
            "top_pred": top_pred,
            "top_pred_count": int(g["pred_label"].value_counts().max()) if len(g) else 0,
        }
        if has_prob:
            row["mean_true_prob"] = float(g["true_prob"].mean())
            row["mean_margin_top1_top2"] = float(g["margin_top1_top2"].mean())
            row["true_in_top2_rate"] = float(g["true_in_top2"].mean())
        rows.append(row)
    res = pd.DataFrame(rows).sort_values(["error_rate", "support"], ascending=[False, False])
    res.to_csv(out_dir / "01_family_difficulty.csv", index=False)
    return res


def pair_confusion(pred: pd.DataFrame, malware_classes: List[str], out_dir: Path) -> pd.DataFrame:
    rows = []
    has_prob = "true_prob" in pred.columns
    for true in malware_classes:
        gtrue = pred[pred["true_label"] == true]
        for pred_lab in malware_classes:
            if pred_lab == true:
                continue
            g = gtrue[gtrue["pred_label"] == pred_lab]
            row = {
                "true_label": true,
                "pred_label": pred_lab,
                "count": int(len(g)),
                "true_support": int(len(gtrue)),
                "rate_within_true": float(len(g) / max(len(gtrue), 1)),
            }
            if len(g):
                row["top_family"] = g["family"].value_counts().idxmax()
                row["top_family_count"] = int(g["family"].value_counts().max())
            else:
                row["top_family"] = ""
                row["top_family_count"] = 0
            if has_prob and len(g):
                row["mean_true_prob"] = float(g["true_prob"].mean())
                row["mean_pred_prob"] = float(g["pred_prob"].mean())
                row["mean_margin_top1_top2"] = float(g["margin_top1_top2"].mean())
                row["true_in_top2_rate"] = float(g["true_in_top2"].mean())
            rows.append(row)
    res = pd.DataFrame(rows).sort_values(["count"], ascending=False)
    res.to_csv(out_dir / "02_malware_pair_confusion.csv", index=False)
    return res


def margin_audit(pred: pd.DataFrame, malware_classes: List[str], out_dir: Path) -> pd.DataFrame:
    if "true_prob" not in pred.columns:
        empty = pd.DataFrame()
        empty.to_csv(out_dir / "03_margin_audit.csv", index=False)
        return empty
    rows = []
    pred = pred.copy()
    pred["correct"] = pred["true_label"] == pred["pred_label"]
    pred["is_malware"] = pred["true_label"].isin(malware_classes)
    for keys, g in pred.groupby(["true_label", "pred_label", "correct"], dropna=False):
        true, pred_lab, correct = keys
        row = {
            "true_label": true,
            "pred_label": pred_lab,
            "correct": bool(correct),
            "count": int(len(g)),
            "mean_true_prob": float(g["true_prob"].mean()) if len(g) else np.nan,
            "mean_pred_prob": float(g["pred_prob"].mean()) if len(g) else np.nan,
            "mean_margin_top1_top2": float(g["margin_top1_top2"].mean()) if len(g) else np.nan,
            "median_margin_top1_top2": float(g["margin_top1_top2"].median()) if len(g) else np.nan,
            "true_in_top2_rate": float(g["true_in_top2"].mean()) if len(g) else np.nan,
        }
        rows.append(row)
    res = pd.DataFrame(rows).sort_values(["correct", "count"], ascending=[True, False])
    res.to_csv(out_dir / "03_margin_audit.csv", index=False)
    return res


def knn_overlap(
    *,
    train_X: np.ndarray,
    val_X: np.ndarray,
    train_labels: np.ndarray,
    val_pred: pd.DataFrame,
    malware_classes: List[str],
    out_path: Path,
    k: int,
    max_query: int,
    seed: int,
    space_name: str,
) -> pd.DataFrame:
    mask_train_mw = np.isin(train_labels, malware_classes)
    Xtr = train_X[mask_train_mw]
    ytr = train_labels[mask_train_mw]
    if len(Xtr) == 0:
        raise ValueError(f"No malware train samples for {space_name} kNN audit")

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)

    # Query all malware val errors plus a comparable sample of correct malware.
    val_mw = val_pred[val_pred["true_label"].isin(malware_classes)].copy()
    val_mw["correct"] = val_mw["true_label"] == val_mw["pred_label"]
    wrong = val_mw[~val_mw["correct"]]
    correct = val_mw[val_mw["correct"]]

    rng = np.random.default_rng(seed)
    query_df = wrong.copy()
    remaining = max(0, int(max_query) - len(query_df))
    if remaining > 0 and len(correct) > 0:
        take = min(remaining, len(correct))
        idx = rng.choice(correct.index.to_numpy(), size=take, replace=False)
        query_df = pd.concat([query_df, correct.loc[idx]], axis=0)
    if len(query_df) > max_query:
        idx = rng.choice(query_df.index.to_numpy(), size=max_query, replace=False)
        query_df = query_df.loc[idx]
    query_df = query_df.sort_values("row_id").reset_index(drop=True)

    Xq = val_X[query_df["row_id"].to_numpy()]
    Xq_s = scaler.transform(Xq)

    nn = NearestNeighbors(n_neighbors=min(k, len(Xtr_s)), metric="euclidean", algorithm="auto")
    nn.fit(Xtr_s)
    dists, inds = nn.kneighbors(Xq_s)

    rows = []
    for i, (_, r) in enumerate(query_df.iterrows()):
        labs = ytr[inds[i]]
        counts = pd.Series(labs).value_counts().to_dict()
        total = len(labs)
        true_lab = r["true_label"]
        pred_lab = r["pred_label"]
        probs = {c: counts.get(c, 0) / total for c in malware_classes}
        entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs.values())
        row = {
            "space": space_name,
            "row_id": int(r["row_id"]),
            "true_label": true_lab,
            "pred_label": pred_lab,
            "correct": bool(true_lab == pred_lab),
            "family": r.get("family", ""),
            "nn_k": int(total),
            "nearest_label": str(labs[0]),
            "nearest_distance": float(dists[i][0]),
            "mean_distance": float(np.mean(dists[i])),
            "true_nn_frac": float(probs.get(true_lab, 0.0)),
            "pred_nn_frac": float(probs.get(pred_lab, 0.0)),
            "neighbor_entropy": float(entropy),
        }
        for c in malware_classes:
            row[f"nn_frac_{c}"] = float(probs.get(c, 0.0))
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(out_path, index=False)

    summary = []
    for correct_val, g in res.groupby("correct"):
        summary.append({
            "space": space_name,
            "correct": bool(correct_val),
            "n": int(len(g)),
            "mean_true_nn_frac": float(g["true_nn_frac"].mean()),
            "mean_pred_nn_frac": float(g["pred_nn_frac"].mean()),
            "mean_entropy": float(g["neighbor_entropy"].mean()),
            "mean_nearest_distance": float(g["nearest_distance"].mean()),
        })
    pd.DataFrame(summary).to_csv(out_path.with_name(out_path.stem + "_summary.csv"), index=False)
    return res


def auc_safe(x: np.ndarray, y: np.ndarray) -> float:
    try:
        if len(np.unique(y)) < 2:
            return np.nan
        return float(roc_auc_score(y, x))
    except Exception:
        return np.nan


def pair_feature_transfer(
    *,
    train_raw_num: pd.DataFrame,
    val_raw_num: pd.DataFrame,
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    feature_cols: List[str],
    malware_classes: List[str],
    out_dir: Path,
    top_n_per_pair: int,
) -> pd.DataFrame:
    rows = []
    pairs = []
    for i in range(len(malware_classes)):
        for j in range(i + 1, len(malware_classes)):
            pairs.append((malware_classes[i], malware_classes[j]))

    Xtr = train_raw_num[feature_cols]
    Xva = val_raw_num[feature_cols]

    # Standardize for mean differences.
    scaler = StandardScaler()
    Xtr_z = pd.DataFrame(scaler.fit_transform(Xtr), columns=feature_cols)
    Xva_z = pd.DataFrame(scaler.transform(Xva), columns=feature_cols)

    for a, b in pairs:
        tr_mask = np.isin(train_labels, [a, b])
        va_mask = np.isin(val_labels, [a, b])
        ytr = (train_labels[tr_mask] == b).astype(int)
        yva = (val_labels[va_mask] == b).astype(int)
        for feat in feature_cols:
            xtr = Xtr.loc[tr_mask, feat].to_numpy(dtype=float)
            xva = Xva.loc[va_mask, feat].to_numpy(dtype=float)
            auc_tr = auc_safe(xtr, ytr)
            auc_va = auc_safe(xva, yva)
            if np.isnan(auc_tr) or np.isnan(auc_va):
                continue
            inverted = auc_tr < 0.5
            auc_tr_or = 1.0 - auc_tr if inverted else auc_tr
            auc_va_or = 1.0 - auc_va if inverted else auc_va

            ztr = Xtr_z.loc[tr_mask, feat].to_numpy(dtype=float)
            zva = Xva_z.loc[va_mask, feat].to_numpy(dtype=float)
            tr_a = ztr[train_labels[tr_mask] == a]
            tr_b = ztr[train_labels[tr_mask] == b]
            va_a = zva[val_labels[va_mask] == a]
            va_b = zva[val_labels[va_mask] == b]
            train_delta = float(np.mean(tr_b) - np.mean(tr_a))
            val_delta = float(np.mean(va_b) - np.mean(va_a))
            if inverted:
                train_delta *= -1
                val_delta *= -1

            rows.append({
                "pair": f"{a}_vs_{b}",
                "class_a": a,
                "class_b": b,
                "feature": feat,
                "train_auc_oriented": float(auc_tr_or),
                "val_auc_same_orientation": float(auc_va_or),
                "auc_transfer_gap": float(auc_tr_or - auc_va_or),
                "min_train_val_auc": float(min(auc_tr_or, auc_va_or)),
                "train_delta_z_oriented": float(train_delta),
                "val_delta_z_same_orientation": float(val_delta),
                "delta_sign_consistent": bool(np.sign(train_delta) == np.sign(val_delta) or abs(train_delta) < 1e-9 or abs(val_delta) < 1e-9),
                "train_support": int(tr_mask.sum()),
                "val_support": int(va_mask.sum()),
            })

    res = pd.DataFrame(rows)
    if len(res) == 0:
        res.to_csv(out_dir / "05_pair_feature_transfer.csv", index=False)
        return res

    res = res.sort_values(["pair", "min_train_val_auc", "val_auc_same_orientation"], ascending=[True, False, False])
    res.to_csv(out_dir / "05_pair_feature_transfer.csv", index=False)

    top = res.groupby("pair", group_keys=False).head(top_n_per_pair).reset_index(drop=True)
    top.to_csv(out_dir / "05_pair_feature_transfer_top.csv", index=False)
    return res


def centroid_signal_audit(
    *,
    feature_transfer: pd.DataFrame,
    train_raw_num: pd.DataFrame,
    val_raw_num: pd.DataFrame,
    train_labels: np.ndarray,
    val_pred: pd.DataFrame,
    malware_classes: List[str],
    out_dir: Path,
    top_n_features: int,
) -> pd.DataFrame:
    if feature_transfer.empty:
        empty = pd.DataFrame()
        empty.to_csv(out_dir / "06_centroid_local_signal_audit.csv", index=False)
        return empty

    scaler = StandardScaler()
    Xtr_z = pd.DataFrame(scaler.fit_transform(train_raw_num), columns=train_raw_num.columns)
    Xva_z = pd.DataFrame(scaler.transform(val_raw_num), columns=val_raw_num.columns)

    rows = []
    # Evaluate each actual true/pred malware pair.
    for true_lab in malware_classes:
        for other_lab in malware_classes:
            if true_lab == other_lab:
                continue
            pair_name_1 = f"{true_lab}_vs_{other_lab}"
            pair_name_2 = f"{other_lab}_vs_{true_lab}"
            ft = feature_transfer[(feature_transfer["pair"] == pair_name_1) | (feature_transfer["pair"] == pair_name_2)].copy()
            if ft.empty:
                continue
            ft = ft.sort_values(["min_train_val_auc", "val_auc_same_orientation"], ascending=False).head(top_n_features)
            feats = [f for f in ft["feature"].tolist() if f in Xtr_z.columns]
            if not feats:
                continue

            tr_true = Xtr_z.loc[train_labels == true_lab, feats]
            tr_other = Xtr_z.loc[train_labels == other_lab, feats]
            if len(tr_true) == 0 or len(tr_other) == 0:
                continue
            c_true = tr_true.mean(axis=0).to_numpy(dtype=float)
            c_other = tr_other.mean(axis=0).to_numpy(dtype=float)

            q = val_pred[(val_pred["true_label"] == true_lab) & (val_pred["pred_label"].isin([true_lab, other_lab]))].copy()
            for _, r in q.iterrows():
                x = Xva_z.loc[int(r["row_id"]), feats].to_numpy(dtype=float)
                d_true = float(np.linalg.norm(x - c_true))
                d_other = float(np.linalg.norm(x - c_other))
                rows.append({
                    "true_label": true_lab,
                    "other_label": other_lab,
                    "pred_label": r["pred_label"],
                    "correct": bool(r["pred_label"] == true_lab),
                    "row_id": int(r["row_id"]),
                    "family": r.get("family", ""),
                    "n_features": int(len(feats)),
                    "dist_to_true_centroid": d_true,
                    "dist_to_other_centroid": d_other,
                    "closer_to_true": bool(d_true < d_other),
                    "distance_margin_other_minus_true": float(d_other - d_true),
                    "feature_set": "|".join(feats),
                })

    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "06_centroid_local_signal_audit.csv", index=False)
    if len(res):
        summary = res.groupby(["true_label", "other_label", "correct"]).agg(
            n=("row_id", "count"),
            closer_to_true_rate=("closer_to_true", "mean"),
            mean_margin_other_minus_true=("distance_margin_other_minus_true", "mean"),
        ).reset_index()
        summary.to_csv(out_dir / "06_centroid_local_signal_summary.csv", index=False)
    return res


def load_npz_token_arrays(dataset_npz: Path) -> Tuple[np.ndarray, np.ndarray]:
    z = np.load(dataset_npz, allow_pickle=True)
    needed = ["X_train_bin", "X_train_offset", "X_val_bin", "X_val_offset"]
    missing = [k for k in needed if k not in z.files]
    if missing:
        raise KeyError(f"dataset npz missing token arrays: {missing}")
    Xtr = np.asarray(z["X_train_bin"], dtype=float) + np.asarray(z["X_train_offset"], dtype=float)
    Xva = np.asarray(z["X_val_bin"], dtype=float) + np.asarray(z["X_val_offset"], dtype=float)
    return Xtr, Xva


def safe_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or len(df) == 0:
        return "_empty_"
    try:
        return df.head(max_rows).to_markdown(index=False)
    except Exception:
        return df.head(max_rows).to_string(index=False)


def write_report(
    *,
    out_dir: Path,
    model_dir: Path,
    metrics: Dict[str, Any],
    family_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    margin_df: pd.DataFrame,
    raw_knn: Optional[pd.DataFrame],
    token_knn: Optional[pd.DataFrame],
    feature_transfer: pd.DataFrame,
    centroid_audit: pd.DataFrame,
    config: Dict[str, Any],
):
    lines = []
    lines.append("# F2 local overlap audit\n")
    lines.append("## Scope\n")
    lines.append("```text")
    lines.append("Diagnostic only. No training. No hyperparameter selection.")
    lines.append("Goal: identify whether malware subtype errors look like true raw/token overlap, representation/boundary issue, family concentration, or unstable local feature signal.")
    lines.append("```")
    lines.append("\n## Model directory\n")
    lines.append(f"`{model_dir}`\n")
    lines.append("\n## Base validation metrics\n")
    lines.append("```json")
    lines.append(json.dumps(metrics, indent=2))
    lines.append("```")
    lines.append("\n## Hardest families\n")
    lines.append(safe_to_markdown(family_df.sort_values(["error_rate", "support"], ascending=[False, False]), 15))
    lines.append("\n## Malware pair confusions\n")
    lines.append(safe_to_markdown(pair_df.sort_values("count", ascending=False), 10))

    if raw_knn is not None and len(raw_knn):
        p = out_dir / "04_raw_knn_overlap_summary.csv"
        if p.exists():
            lines.append("\n## Raw-space kNN overlap summary\n")
            lines.append(safe_to_markdown(pd.read_csv(p), 20))
    if token_knn is not None and len(token_knn):
        p = out_dir / "04_token_knn_overlap_summary.csv"
        if p.exists():
            lines.append("\n## Token-space kNN overlap summary\n")
            lines.append(safe_to_markdown(pd.read_csv(p), 20))

    lines.append("\n## Top stable pair features\n")
    top_path = out_dir / "05_pair_feature_transfer_top.csv"
    if top_path.exists():
        lines.append(safe_to_markdown(pd.read_csv(top_path), 30))
    else:
        lines.append("_missing_")

    if len(centroid_audit):
        sum_path = out_dir / "06_centroid_local_signal_summary.csv"
        if sum_path.exists():
            lines.append("\n## Centroid local signal summary\n")
            lines.append(safe_to_markdown(pd.read_csv(sum_path), 30))

    lines.append("\n## How to read\n")
    lines.append("```text")
    lines.append("If wrong samples have high pred_nn_frac and low true_nn_frac in raw/token space, the split itself has local overlap.")
    lines.append("If raw/token kNN favors true class but model predicts another class, the learned representation/boundary is likely distorting local signal.")
    lines.append("If pair feature transfer has strong train AUC but weak val AUC, feature rules do not transfer cleanly.")
    lines.append("If errors concentrate in a few L3 families, a family-aware/local method may be needed instead of global regularization.")
    lines.append("```")

    lines.append("\n## Config\n")
    lines.append("```json")
    lines.append(json.dumps(config, indent=2, default=str))
    lines.append("```")

    (out_dir / "F2_overlap_audit_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="", help="Directory containing val_predictions_best.csv. If omitted, auto-search.")
    ap.add_argument("--search-root", default="05_test/outputs", help="Root used to auto-find model-dir.")
    ap.add_argument("--preferred-keywords", default="F1e2a,L1,base,reproduce", help="Auto-find preference keywords.")
    ap.add_argument("--dataset-npz", default="03_outputs/05_dataset/dataset.npz")
    ap.add_argument("--train-raw", default="01_split/train_raw.csv")
    ap.add_argument("--val-raw", default="01_split/val_raw.csv")
    ap.add_argument("--out-dir", default="05_test/outputs/F2_overlap_local_pair_family_audit")
    ap.add_argument("--combined-zip", default="05_test/outputs/F2_overlap_local_pair_family_audit.zip")
    ap.add_argument("--class-names", default="Benign,Ransomware,Spyware,Trojan")
    ap.add_argument("--malware-classes", default="Ransomware,Spyware,Trojan")
    ap.add_argument("--knn-k", type=int, default=31)
    ap.add_argument("--max-knn-query", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-n-features-per-pair", type=int, default=20)
    ap.add_argument("--centroid-top-n-features", type=int, default=15)
    ap.add_argument("--skip-raw-knn", action="store_true")
    ap.add_argument("--skip-token-knn", action="store_true")
    args = ap.parse_args()

    root = repo_root_from_here()
    dataset_npz = resolve_path(args.dataset_npz, root)
    train_raw_path = resolve_path(args.train_raw, root)
    val_raw_path = resolve_path(args.val_raw, root)
    out_dir = resolve_path(args.out_dir, root)
    zip_path = resolve_path(args.combined_zip, root)

    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = parse_list(args.class_names) or CLASS_NAMES_DEFAULT
    malware_classes = parse_list(args.malware_classes) or MALWARE_CLASSES_DEFAULT

    if args.model_dir.strip():
        model_dir = resolve_path(args.model_dir, root)
    else:
        model_dir = find_model_dir(resolve_path(args.search_root, root), parse_list(args.preferred_keywords))
        if model_dir is None:
            raise FileNotFoundError(
                "Could not auto-find model dir containing val_predictions_best.csv. "
                "Pass --model-dir explicitly."
            )

    pred_path = model_dir / "val_predictions_best.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing val_predictions_best.csv in model dir: {model_dir}")

    train_raw = pd.read_csv(train_raw_path)
    val_raw = pd.read_csv(val_raw_path)

    val_pred = load_predictions(pred_path, val_raw, class_names, "val")
    val_pred = add_raw_labels(val_pred, val_raw, class_names)
    val_pred.to_csv(out_dir / "00_val_predictions_normalized.csv", index=False)

    train_l2_col = find_label_col(train_raw, "L2")
    val_l2_col = find_label_col(val_raw, "L2")
    if train_l2_col is None or val_l2_col is None:
        raise ValueError("Need L2 labels in train_raw and val_raw for overlap audit.")
    train_labels = train_raw[train_l2_col].map(clean_label).to_numpy()
    val_labels = val_raw[val_l2_col].map(clean_label).to_numpy()

    metrics = basic_metrics(val_pred, class_names, out_dir)
    family_df = family_difficulty(val_pred, class_names, out_dir)
    pair_df = pair_confusion(val_pred, malware_classes, out_dir)
    margin_df = margin_audit(val_pred, malware_classes, out_dir)

    train_raw_num, val_raw_num, feature_cols = get_numeric_features(train_raw, val_raw)
    train_raw_num.to_csv(out_dir / "_debug_train_numeric_head.csv", index=False)
    pd.DataFrame({"feature": feature_cols}).to_csv(out_dir / "00_numeric_features_used.csv", index=False)

    raw_knn = None
    token_knn = None
    if not args.skip_raw_knn:
        log("Running raw-space kNN overlap audit")
        raw_knn = knn_overlap(
            train_X=train_raw_num.to_numpy(dtype=float),
            val_X=val_raw_num.to_numpy(dtype=float),
            train_labels=train_labels,
            val_pred=val_pred,
            malware_classes=malware_classes,
            out_path=out_dir / "04_raw_knn_overlap.csv",
            k=int(args.knn_k),
            max_query=int(args.max_knn_query),
            seed=int(args.seed),
            space_name="raw_numeric_z",
        )

    if not args.skip_token_knn:
        try:
            log("Running token-space kNN overlap audit")
            Xtr_tok, Xva_tok = load_npz_token_arrays(dataset_npz)
            token_knn = knn_overlap(
                train_X=Xtr_tok,
                val_X=Xva_tok,
                train_labels=train_labels,
                val_pred=val_pred,
                malware_classes=malware_classes,
                out_path=out_dir / "04_token_knn_overlap.csv",
                k=int(args.knn_k),
                max_query=int(args.max_knn_query),
                seed=int(args.seed),
                space_name="token_bin_plus_offset_z",
            )
        except Exception as e:
            (out_dir / "04_token_knn_error.txt").write_text(str(e), encoding="utf-8")
            log(f"Token kNN skipped due to error: {e}")

    log("Running pair feature transfer audit")
    feature_transfer = pair_feature_transfer(
        train_raw_num=train_raw_num,
        val_raw_num=val_raw_num,
        train_labels=train_labels,
        val_labels=val_labels,
        feature_cols=feature_cols,
        malware_classes=malware_classes,
        out_dir=out_dir,
        top_n_per_pair=int(args.top_n_features_per_pair),
    )

    log("Running centroid local signal audit")
    centroid_audit = centroid_signal_audit(
        feature_transfer=feature_transfer,
        train_raw_num=train_raw_num,
        val_raw_num=val_raw_num,
        train_labels=train_labels,
        val_pred=val_pred,
        malware_classes=malware_classes,
        out_dir=out_dir,
        top_n_features=int(args.centroid_top_n_features),
    )

    config = {
        "experiment": "F2_overlap_local_pair_family_audit",
        "diagnostic_only": True,
        "training_performed": False,
        "hyperparameter_selection_performed": False,
        "model_dir": str(model_dir),
        "prediction_path": str(pred_path),
        "dataset_npz": str(dataset_npz),
        "train_raw": str(train_raw_path),
        "val_raw": str(val_raw_path),
        "class_names": class_names,
        "malware_classes": malware_classes,
        "knn_k": int(args.knn_k),
        "max_knn_query": int(args.max_knn_query),
        "n_numeric_features": int(len(feature_cols)),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

    write_report(
        out_dir=out_dir,
        model_dir=model_dir,
        metrics=metrics,
        family_df=family_df,
        pair_df=pair_df,
        margin_df=margin_df,
        raw_knn=raw_knn,
        token_knn=token_knn,
        feature_transfer=feature_transfer,
        centroid_audit=centroid_audit,
        config=config,
    )

    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out_dir.rglob("*"):
            if p.is_file() and p != zip_path:
                z.write(p, p.relative_to(out_dir.parent))

    log(f"metrics: {metrics}")
    log(f"out_dir={out_dir}")
    log(f"zip={zip_path}")
    log("DONE")


if __name__ == "__main__":
    main()
