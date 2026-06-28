#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C3_external_pairwise_reranker_diagnostic.py

Purpose
-------
Phase C3 diagnostic only.

Train external pairwise rerankers with out-of-fold (OOF) validation prediction
to test whether CLS/log-probability features can improve hard malware subtype
decisions WITHOUT changing the official model.

This script explicitly measures both:
  1. wrong -> correct improvements
  2. correct -> wrong damage

It also reports hard-pair breakdowns so we can see which confused pairs improve
and which already-correct pairs are damaged.

Important guardrail
-------------------
This is NOT an official final model and NOT a deployable result:
  - It uses validation-set OOF diagnostic training.
  - It does not modify 02_src or official baseline outputs.
  - It is only used to decide whether a Phase D model-side change is worth trying.

Inputs
------
Recommended:
  --pred-csv 05_test/outputs/B1_cls_pairwise_signal/val_cls_predictions_with_probs.csv
  --cls-npz  05_test/outputs/B1_cls_pairwise_signal/val_cls_embeddings.npz

Outputs
-------
  C3_summary.md
  C3_policy_metrics.csv
  C3_policy_per_class_f1.csv
  C3_pairwise_cv_metrics.csv
  C3_transition_summary.csv
  C3_pair_fix_damage_summary.csv
  C3_best_confusion_matrix.csv
  C3_original_confusion_matrix.csv
  C3_gate_decision.json
  C3_external_pairwise_reranker_output.zip
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_HARD_PAIRS = [
    ("Ransomware", "Spyware"),
    ("Ransomware", "Trojan"),
    ("Spyware", "Trojan"),
]

DEFAULT_LABEL_ORDER = ["Benign", "Ransomware", "Spyware", "Trojan"]
DEFAULT_CONF_THRESHOLDS = [0.50, 0.55, 0.60, 0.70, 0.80]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="C3 external OOF pairwise reranker diagnostic.")
    p.add_argument("--repo-root", default=".")
    p.add_argument("--pred-csv", default="05_test/outputs/B1_cls_pairwise_signal/val_cls_predictions_with_probs.csv")
    p.add_argument("--cls-npz", default="05_test/outputs/B1_cls_pairwise_signal/val_cls_embeddings.npz")
    p.add_argument("--out-dir", default="05_test/outputs/C3_external_pairwise_reranker")
    p.add_argument("--score-prefix", default="prob_")
    p.add_argument("--label-order", nargs="*", default=DEFAULT_LABEL_ORDER)
    p.add_argument("--pairs", nargs="*", default=None,
                   help='Optional hard pairs as "A:B", e.g. "Ransomware:Trojan". Default malware pairs.')
    p.add_argument("--feature-sets", nargs="*", default=None,
                   help="Optional feature sets. Defaults to cls, logprobs, probs, cls__logprobs, cls__probs.")
    p.add_argument("--confidence-thresholds", nargs="*", type=float, default=DEFAULT_CONF_THRESHOLDS,
                   help="Only apply a reranker flip if pairwise confidence >= threshold.")
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--max-iter", type=int, default=5000)
    return p.parse_args()


def repo_path(repo_root: Path, path_like: str | Path) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    return repo_root / p


def normalize_label(x: Any) -> str:
    return str(x).strip()


def parse_pairs(pair_args: List[str] | None) -> List[Tuple[str, str]]:
    if not pair_args:
        return DEFAULT_HARD_PAIRS
    pairs: List[Tuple[str, str]] = []
    for item in pair_args:
        if ":" not in item:
            raise ValueError(f"Invalid pair format {item!r}; expected A:B")
        a, b = item.split(":", 1)
        pairs.append((normalize_label(a), normalize_label(b)))
    return pairs


def zip_outputs(out_dir: Path, zip_name: str = "C3_external_pairwise_reranker_output.zip") -> Path:
    out_zip = out_dir / zip_name
    if out_zip.exists():
        out_zip.unlink()

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fp in out_dir.rglob("*"):
            if fp.is_file() and fp != out_zip:
                z.write(fp, fp.relative_to(out_dir))
    return out_zip


def safe_label_col(label: str) -> str:
    return normalize_label(label).replace(" ", "_")


def prob_col_for_label(df: pd.DataFrame, label: str, score_prefix: str) -> str:
    candidates = [
        f"{score_prefix}{label}",
        f"{score_prefix}{safe_label_col(label)}",
    ]
    for c in candidates:
        if c in df.columns:
            return c

    target = safe_label_col(label).lower()
    for c in df.columns:
        if not c.startswith(score_prefix):
            continue
        tail = c[len(score_prefix):]
        if safe_label_col(tail).lower() == target:
            return c

    raise KeyError(f"Cannot find probability column for label={label!r}; tried {candidates}")


def infer_label_order(df: pd.DataFrame, user_order: List[str], score_prefix: str) -> List[str]:
    observed: List[str] = []
    for c in ["true_label", "pred_label", "top1_label", "top2_label", "computed_top1_label", "computed_top2_label"]:
        if c in df.columns:
            for x in df[c].dropna().map(normalize_label).unique().tolist():
                if x not in observed:
                    observed.append(x)

    prob_labels: List[str] = []
    for c in df.columns:
        if c.startswith(score_prefix):
            label = c[len(score_prefix):].replace("_", " ")
            if label not in prob_labels:
                prob_labels.append(label)

    labels: List[str] = []
    for x in user_order:
        x = normalize_label(x)
        if x in observed or x in prob_labels:
            labels.append(x)

    for x in observed + prob_labels:
        if x not in labels:
            labels.append(x)

    return labels


def load_cls_embeddings(path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CLS npz: {path}")

    with np.load(path, allow_pickle=True) as z:
        keys = list(z.files)
        cls_key = None
        for k in ["cls_embeddings", "val_cls_embeddings", "cls", "X_cls", "embeddings"]:
            if k in z.files:
                cls_key = k
                break
        if cls_key is None:
            raise KeyError(f"Cannot find CLS embeddings in {path}; available keys={keys}")

        cls = z[cls_key].astype(np.float32)
        info: Dict[str, Any] = {
            "path": str(path),
            "keys": keys,
            "cls_key": cls_key,
            "cls_shape": list(cls.shape),
        }

        for k in ["label_names", "labels"]:
            if k in z.files:
                try:
                    info[k] = [str(x) for x in z[k].tolist()]
                except Exception:
                    info[k] = str(z[k])

    if cls.ndim != 2:
        raise ValueError(f"CLS embeddings must be 2D, got shape={cls.shape}")

    return cls, info


def build_augmented_predictions(df0: pd.DataFrame, labels: List[str], score_prefix: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    required = ["true_label", "pred_label"]
    missing = [c for c in required if c not in df0.columns]
    if missing:
        raise KeyError(f"Prediction CSV missing required columns: {missing}")

    df = df0.copy()
    df["true_label"] = df["true_label"].map(normalize_label)
    df["pred_label"] = df["pred_label"].map(normalize_label)

    if "sample_index" not in df.columns:
        df["sample_index"] = np.arange(len(df))

    prob_cols = {label: prob_col_for_label(df, label, score_prefix) for label in labels}
    probs = df[[prob_cols[label] for label in labels]].to_numpy(dtype=np.float64)

    if not np.isfinite(probs).all():
        raise ValueError("Probability matrix contains NaN/Inf")

    label_to_id = {label: i for i, label in enumerate(labels)}
    true_id = np.array([label_to_id[x] for x in df["true_label"]], dtype=int)
    pred_id = np.array([label_to_id[x] for x in df["pred_label"]], dtype=int)

    order = np.argsort(-probs, axis=1)
    top1_id = order[:, 0]
    top2_id = order[:, 1]
    idx = np.arange(len(df))

    df["true_id"] = true_id
    df["pred_id"] = pred_id
    df["top1_id"] = top1_id
    df["top2_id"] = top2_id
    df["top1_label"] = [labels[i] for i in top1_id]
    df["top2_label"] = [labels[i] for i in top2_id]
    df["top1_score"] = probs[idx, top1_id]
    df["top2_score"] = probs[idx, top2_id]
    df["top12_margin"] = df["top1_score"] - df["top2_score"]
    df["true_prob"] = probs[idx, true_id]
    df["pred_prob"] = probs[idx, pred_id]
    df["pred_minus_true_prob"] = df["pred_prob"] - df["true_prob"]
    df["original_correct"] = df["true_id"] == df["pred_id"]
    df["true_in_top2"] = (df["true_id"] == top1_id) | (df["true_id"] == top2_id)

    # Robustness check: pred_label should be the top-1 label.
    mismatch = (df["pred_label"].to_numpy() != df["top1_label"].to_numpy())
    if bool(np.any(mismatch)):
        # Keep going, but record; some exporters may have different label order.
        df["pred_top1_mismatch"] = mismatch
    else:
        df["pred_top1_mismatch"] = False

    return df, probs.astype(np.float32), np.log(np.clip(probs, 1e-12, 1.0)).astype(np.float32)


def make_feature_sets(cls: np.ndarray, probs: np.ndarray, logprobs: np.ndarray, requested: List[str] | None) -> Dict[str, np.ndarray]:
    all_sets: Dict[str, np.ndarray] = {
        "cls": cls.astype(np.float32),
        "probs": probs.astype(np.float32),
        "logprobs": logprobs.astype(np.float32),
        "cls__probs": np.concatenate([cls.astype(np.float32), probs.astype(np.float32)], axis=1),
        "cls__logprobs": np.concatenate([cls.astype(np.float32), logprobs.astype(np.float32)], axis=1),
    }

    if requested:
        missing = [x for x in requested if x not in all_sets]
        if missing:
            raise KeyError(f"Unknown feature sets: {missing}; available={sorted(all_sets)}")
        return {k: all_sets[k] for k in requested}

    return all_sets


def compute_pairwise_oof(
    X: np.ndarray,
    y_binary: np.ndarray,
    *,
    cv_folds: int,
    random_state: int,
    max_iter: int,
) -> Dict[str, Any]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
        from sklearn.model_selection import StratifiedKFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("C3 requires scikit-learn. Install with: pip install scikit-learn") from e

    counts = np.bincount(y_binary.astype(int), minlength=2)
    min_count = int(counts.min())
    k = int(min(cv_folds, min_count))
    if k < 2:
        raise ValueError(f"Cannot run CV: class counts={counts.tolist()}, folds={k}")

    oof_proba_b = np.zeros(len(y_binary), dtype=np.float64)
    oof_pred = np.zeros(len(y_binary), dtype=int)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)

    for tr, te in skf.split(X, y_binary):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=max_iter,
                class_weight="balanced",
                solver="lbfgs",
                random_state=random_state,
            ),
        )
        clf.fit(X[tr], y_binary[tr])
        oof_proba_b[te] = clf.predict_proba(X[te])[:, 1]
        oof_pred[te] = (oof_proba_b[te] >= 0.5).astype(int)

    metrics = {
        "cv_folds": k,
        "n_samples": int(len(y_binary)),
        "class0_count": int(counts[0]),
        "class1_count": int(counts[1]),
        "accuracy": float(accuracy_score(y_binary, oof_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_binary, oof_pred)),
        "macro_f1": float(f1_score(y_binary, oof_pred, average="macro")),
        "auc": float(roc_auc_score(y_binary, oof_proba_b)),
    }

    return {
        "oof_pred": oof_pred,
        "oof_proba_b": oof_proba_b,
        "metrics": metrics,
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> Tuple[Dict[str, Any], Dict[str, float]]:
    try:
        from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("C3 requires scikit-learn. Install with: pip install scikit-learn") from e

    metric = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
    }

    p, r, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    per_class = {f"f1_{label}": float(v) for label, v in zip(labels, f1)}
    per_class.update({f"recall_{label}": float(v) for label, v in zip(labels, r)})
    per_class.update({f"precision_{label}": float(v) for label, v in zip(labels, p)})
    per_class.update({f"support_{label}": int(v) for label, v in zip(labels, support)})

    return metric, per_class


def confusion_df(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> pd.DataFrame:
    try:
        from sklearn.metrics import confusion_matrix
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("C3 requires scikit-learn. Install with: pip install scikit-learn") from e

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=[f"true_{x}" for x in labels], columns=[f"pred_{x}" for x in labels])


def pair_key_from_top2(row: pd.Series) -> str:
    a = normalize_label(row["top1_label"])
    b = normalize_label(row["top2_label"])
    return "<->".join(sorted([a, b]))


def simulate_rerank_policy(
    df: pd.DataFrame,
    labels: List[str],
    pairs: List[Tuple[str, str]],
    pair_oof: Dict[Tuple[str, str], Dict[str, np.ndarray]],
    *,
    confidence_threshold: float,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Apply pairwise OOF reranker only on samples whose true label is inside the hard pair
    and whose original top-1/top-2 set exactly matches that pair.

    This estimates fix/damage on hard-pair validation samples with OOF predictions.
    """
    reranked = df["pred_label"].copy()
    changed_rows: List[Dict[str, Any]] = []

    label_to_id = {label: i for i, label in enumerate(labels)}
    n = len(df)

    # Maps row index to pairwise prediction metadata.
    for a, b in pairs:
        aid, bid = label_to_id[a], label_to_id[b]
        true_pair_mask = (df["true_label"] == a) | (df["true_label"] == b)
        pair_candidate_mask = (
            ((df["top1_label"] == a) & (df["top2_label"] == b)) |
            ((df["top1_label"] == b) & (df["top2_label"] == a))
        )
        apply_base_mask = true_pair_mask & pair_candidate_mask

        pair_result = pair_oof[(a, b)]
        pair_indices = np.where(true_pair_mask.to_numpy(dtype=bool))[0]

        # pair_result arrays are aligned to pair_indices.
        local_to_global = pair_indices
        oof_pred = pair_result["oof_pred"]
        oof_proba_b = pair_result["oof_proba_b"]

        for local_i, global_i in enumerate(local_to_global):
            if not bool(apply_base_mask.iloc[global_i]):
                continue

            p_b = float(oof_proba_b[local_i])
            pred_bin = int(oof_pred[local_i])
            confidence = max(p_b, 1.0 - p_b)
            pair_pred_label = b if pred_bin == 1 else a

            original_label = normalize_label(df.iloc[global_i]["pred_label"])
            true_label = normalize_label(df.iloc[global_i]["true_label"])

            if confidence < confidence_threshold:
                continue
            if pair_pred_label == original_label:
                continue

            old_correct = original_label == true_label
            new_correct = pair_pred_label == true_label

            reranked.iloc[global_i] = pair_pred_label
            changed_rows.append({
                "row_index": int(global_i),
                "sample_index": int(df.iloc[global_i]["sample_index"]),
                "pair": f"{a}<->{b}",
                "true_label": true_label,
                "original_pred": original_label,
                "reranked_pred": pair_pred_label,
                "top1_label": normalize_label(df.iloc[global_i]["top1_label"]),
                "top2_label": normalize_label(df.iloc[global_i]["top2_label"]),
                "top12_margin": float(df.iloc[global_i]["top12_margin"]),
                "pairwise_confidence": confidence,
                "pairwise_proba_second_label": p_b,
                "old_correct": bool(old_correct),
                "new_correct": bool(new_correct),
                "transition_type": (
                    "wrong_to_correct" if (not old_correct and new_correct)
                    else "correct_to_wrong" if (old_correct and not new_correct)
                    else "wrong_to_wrong" if (not old_correct and not new_correct)
                    else "correct_to_correct"
                ),
            })

    return reranked, pd.DataFrame(changed_rows)


def summarize_transitions(df: pd.DataFrame, reranked_pred: pd.Series, changed_df: pd.DataFrame, policy_name: str) -> Dict[str, Any]:
    y_true = df["true_label"].to_numpy()
    y_orig = df["pred_label"].to_numpy()
    y_new = reranked_pred.to_numpy()

    orig_correct = y_orig == y_true
    new_correct = y_new == y_true
    changed = y_orig != y_new

    wrong_to_correct = (~orig_correct) & new_correct
    correct_to_wrong = orig_correct & (~new_correct)
    wrong_to_wrong_changed = (~orig_correct) & (~new_correct) & changed
    correct_to_correct_changed = orig_correct & new_correct & changed

    return {
        "policy": policy_name,
        "n_changed": int(changed.sum()),
        "wrong_to_correct_n": int(wrong_to_correct.sum()),
        "correct_to_wrong_n": int(correct_to_wrong.sum()),
        "wrong_to_wrong_changed_n": int(wrong_to_wrong_changed.sum()),
        "correct_to_correct_changed_n": int(correct_to_correct_changed.sum()),
        "net_gain_n": int(wrong_to_correct.sum() - correct_to_wrong.sum()),
        "damage_ratio_correct_to_wrong_over_wrong_to_correct": (
            float(correct_to_wrong.sum() / wrong_to_correct.sum()) if int(wrong_to_correct.sum()) else float("nan")
        ),
        "n_changed_detail_rows": int(len(changed_df)),
    }


def pair_fix_damage_summary(
    df: pd.DataFrame,
    reranked_pred: pd.Series,
    pairs: List[Tuple[str, str]],
    policy_name: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    y_true = df["true_label"]
    y_orig = df["pred_label"]
    y_new = reranked_pred

    orig_correct = y_orig == y_true
    new_correct = y_new == y_true
    changed = y_orig != y_new

    for a, b in pairs:
        pair_candidate = (
            ((df["top1_label"] == a) & (df["top2_label"] == b)) |
            ((df["top1_label"] == b) & (df["top2_label"] == a))
        )
        true_pair = y_true.isin([a, b])
        candidate_true_pair = pair_candidate & true_pair

        direct_wrong = (
            ((y_true == a) & (y_orig == b)) |
            ((y_true == b) & (y_orig == a))
        )
        direct_wrong_candidate = direct_wrong & pair_candidate

        fixed = direct_wrong & new_correct
        damaged = orig_correct & (~new_correct) & candidate_true_pair
        changed_wrong_to_wrong = (~orig_correct) & (~new_correct) & changed & candidate_true_pair

        row = {
            "policy": policy_name,
            "pair": f"{a}<->{b}",
            "direction": "BIDIRECTIONAL",
            "candidate_true_pair_n": int(candidate_true_pair.sum()),
            "original_direct_wrong_n": int(direct_wrong.sum()),
            "original_direct_wrong_candidate_n": int(direct_wrong_candidate.sum()),
            "fixed_direct_wrong_n": int(fixed.sum()),
            "fix_rate_among_direct_wrong": float(fixed.sum() / direct_wrong.sum()) if int(direct_wrong.sum()) else float("nan"),
            "original_correct_candidate_n": int((orig_correct & candidate_true_pair).sum()),
            "correct_to_wrong_damage_n": int(damaged.sum()),
            "damage_rate_among_correct_candidates": (
                float(damaged.sum() / (orig_correct & candidate_true_pair).sum()) if int((orig_correct & candidate_true_pair).sum()) else float("nan")
            ),
            "wrong_to_wrong_changed_n": int(changed_wrong_to_wrong.sum()),
            "net_pair_gain_n": int(fixed.sum() - damaged.sum()),
        }
        rows.append(row)

        for true_c, pred_c in [(a, b), (b, a)]:
            direct_dir = (y_true == true_c) & (y_orig == pred_c)
            fixed_dir = direct_dir & new_correct

            # damage in opposite direction: true_c originally correct, top2 pred_c, reranker flips to pred_c.
            damage_dir = (
                (y_true == true_c)
                & (y_orig == true_c)
                & (df["top2_label"] == pred_c)
                & (y_new == pred_c)
            )

            rows.append({
                "policy": policy_name,
                "pair": f"{a}<->{b}",
                "direction": f"{true_c}->{pred_c}",
                "candidate_true_pair_n": int(((y_true == true_c) & pair_candidate).sum()),
                "original_direct_wrong_n": int(direct_dir.sum()),
                "original_direct_wrong_candidate_n": int((direct_dir & pair_candidate).sum()),
                "fixed_direct_wrong_n": int(fixed_dir.sum()),
                "fix_rate_among_direct_wrong": float(fixed_dir.sum() / direct_dir.sum()) if int(direct_dir.sum()) else float("nan"),
                "original_correct_candidate_n": int(((y_true == true_c) & orig_correct & pair_candidate).sum()),
                "correct_to_wrong_damage_n": int(damage_dir.sum()),
                "damage_rate_among_correct_candidates": (
                    float(damage_dir.sum() / ((y_true == true_c) & orig_correct & pair_candidate).sum())
                    if int(((y_true == true_c) & orig_correct & pair_candidate).sum()) else float("nan")
                ),
                "wrong_to_wrong_changed_n": int(((y_true == true_c) & (~orig_correct) & (~new_correct) & changed & pair_candidate).sum()),
                "net_pair_gain_n": int(fixed_dir.sum() - damage_dir.sum()),
            })

    return pd.DataFrame(rows)


def make_policy_tables(
    df: pd.DataFrame,
    labels: List[str],
    pairs: List[Tuple[str, str]],
    feature_sets: Dict[str, np.ndarray],
    *,
    confidence_thresholds: List[float],
    cv_folds: int,
    random_state: int,
    max_iter: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, pd.Series], Dict[str, pd.DataFrame]]:
    y_true = df["true_label"].to_numpy()
    y_orig = df["pred_label"].to_numpy()
    orig_metrics, orig_per_class = compute_metrics(y_true, y_orig, labels)

    policy_rows: List[Dict[str, Any]] = []
    per_class_rows: List[Dict[str, Any]] = []
    pairwise_cv_rows: List[Dict[str, Any]] = []
    transition_rows: List[Dict[str, Any]] = []
    pair_damage_frames: List[pd.DataFrame] = []

    reranked_by_policy: Dict[str, pd.Series] = {}
    changed_by_policy: Dict[str, pd.DataFrame] = {}

    # Original policy.
    original_policy = "original"
    transition_rows.append({
        "policy": original_policy,
        "n_changed": 0,
        "wrong_to_correct_n": 0,
        "correct_to_wrong_n": 0,
        "wrong_to_wrong_changed_n": 0,
        "correct_to_correct_changed_n": 0,
        "net_gain_n": 0,
        "damage_ratio_correct_to_wrong_over_wrong_to_correct": float("nan"),
        "n_changed_detail_rows": 0,
    })
    policy_rows.append({
        "policy": original_policy,
        "feature_set": "none",
        "confidence_threshold": None,
        "n_changed": 0,
        "wrong_to_correct_n": 0,
        "correct_to_wrong_n": 0,
        "net_gain_n": 0,
        **orig_metrics,
        "delta_accuracy": 0.0,
        "delta_macro_f1": 0.0,
        "delta_weighted_f1": 0.0,
    })
    per_class_rows.append({
        "policy": original_policy,
        "feature_set": "none",
        "confidence_threshold": None,
        **orig_per_class,
        **{f"delta_f1_{label}": 0.0 for label in labels},
        **{f"delta_recall_{label}": 0.0 for label in labels},
    })
    reranked_by_policy[original_policy] = df["pred_label"].copy()
    changed_by_policy[original_policy] = pd.DataFrame()

    label_to_id = {label: i for i, label in enumerate(labels)}

    for feat_name, X_all in feature_sets.items():
        pair_oof: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}

        for a, b in pairs:
            true_pair_mask = ((df["true_label"] == a) | (df["true_label"] == b)).to_numpy(dtype=bool)
            X_pair = X_all[true_pair_mask]
            y_pair = (df.loc[true_pair_mask, "true_label"].to_numpy() == b).astype(int)

            oof = compute_pairwise_oof(
                X_pair,
                y_pair,
                cv_folds=cv_folds,
                random_state=random_state,
                max_iter=max_iter,
            )
            pair_oof[(a, b)] = oof

            row = {
                "feature_set": feat_name,
                "pair": f"{a}<->{b}",
                "class0": a,
                "class1": b,
                "dim": int(X_all.shape[1]),
                **oof["metrics"],
            }
            pairwise_cv_rows.append(row)

        for thr in confidence_thresholds:
            policy_name = f"{feat_name}__rerank_top2_pair_conf_ge_{thr:g}"

            reranked_pred, changed_df = simulate_rerank_policy(
                df,
                labels,
                pairs,
                pair_oof,
                confidence_threshold=float(thr),
            )

            metrics, per_class = compute_metrics(y_true, reranked_pred.to_numpy(), labels)
            trans = summarize_transitions(df, reranked_pred, changed_df, policy_name)
            transition_rows.append(trans)

            policy_rows.append({
                "policy": policy_name,
                "feature_set": feat_name,
                "confidence_threshold": float(thr),
                "n_changed": trans["n_changed"],
                "wrong_to_correct_n": trans["wrong_to_correct_n"],
                "correct_to_wrong_n": trans["correct_to_wrong_n"],
                "net_gain_n": trans["net_gain_n"],
                "damage_ratio": trans["damage_ratio_correct_to_wrong_over_wrong_to_correct"],
                **metrics,
                "delta_accuracy": float(metrics["accuracy"] - orig_metrics["accuracy"]),
                "delta_macro_f1": float(metrics["macro_f1"] - orig_metrics["macro_f1"]),
                "delta_weighted_f1": float(metrics["weighted_f1"] - orig_metrics["weighted_f1"]),
            })

            pc_row = {
                "policy": policy_name,
                "feature_set": feat_name,
                "confidence_threshold": float(thr),
                **per_class,
            }
            for label in labels:
                pc_row[f"delta_f1_{label}"] = float(per_class[f"f1_{label}"] - orig_per_class[f"f1_{label}"])
                pc_row[f"delta_recall_{label}"] = float(per_class[f"recall_{label}"] - orig_per_class[f"recall_{label}"])
            per_class_rows.append(pc_row)

            pfd = pair_fix_damage_summary(df, reranked_pred, pairs, policy_name)
            pair_damage_frames.append(pfd)

            reranked_by_policy[policy_name] = reranked_pred
            changed_by_policy[policy_name] = changed_df

    policy_df = pd.DataFrame(policy_rows)
    per_class_df = pd.DataFrame(per_class_rows)
    pairwise_cv_df = pd.DataFrame(pairwise_cv_rows)
    transition_df = pd.DataFrame(transition_rows)
    pair_damage_df = pd.concat(pair_damage_frames, ignore_index=True) if pair_damage_frames else pd.DataFrame()

    return policy_df, per_class_df, pairwise_cv_df, transition_df, pair_damage_df, reranked_by_policy, changed_by_policy


def choose_best_policy(policy_df: pd.DataFrame) -> pd.Series:
    candidates = policy_df[policy_df["policy"] != "original"].copy()
    if len(candidates) == 0:
        return policy_df.iloc[0]

    # Prefer macro-F1, then net gain, then lower damage, then fewer changed samples.
    candidates = candidates.sort_values(
        by=["delta_macro_f1", "net_gain_n", "correct_to_wrong_n", "n_changed"],
        ascending=[False, False, True, True],
    )
    return candidates.iloc[0]


def make_gate(best: pd.Series, policy_df: pd.DataFrame) -> Dict[str, Any]:
    delta_macro = float(best["delta_macro_f1"])
    wrong_to_correct = int(best["wrong_to_correct_n"])
    correct_to_wrong = int(best["correct_to_wrong_n"])
    net_gain = int(best["net_gain_n"])
    damage_ratio = float(correct_to_wrong / wrong_to_correct) if wrong_to_correct else float("nan")

    if delta_macro >= 0.02 and net_gain > 0 and (not np.isfinite(damage_ratio) or damage_ratio <= 0.5):
        result = "PASS — external pairwise reranker shows useful net gain with bounded damage"
        reason = (
            f"Best policy `{best['policy']}` improves macro-F1 by {delta_macro:.4f}, "
            f"fixes {wrong_to_correct} wrong samples, damages {correct_to_wrong} correct samples, "
            f"net_gain={net_gain}."
        )
        recommendation = (
            "Proceed to Phase D only as an isolated model-side test, such as auxiliary pairwise head or margin objective. "
            "Do not treat this validation OOF reranker as final."
        )
    elif delta_macro > 0 and net_gain > 0:
        result = "MIXED — reranker improves metrics but damage or gain size needs caution"
        reason = (
            f"Best policy `{best['policy']}` has positive delta_macro_f1={delta_macro:.4f}, "
            f"wrong_to_correct={wrong_to_correct}, correct_to_wrong={correct_to_wrong}, net_gain={net_gain}. "
            "The improvement exists but may not be stable enough for direct adoption."
        )
        recommendation = (
            "Inspect pair-level damage before deciding Phase D. Prefer safer constraints or a training-time auxiliary head."
        )
    else:
        result = "FAIL — external pairwise reranker does not produce useful net gain"
        reason = (
            f"Best policy `{best['policy']}` delta_macro_f1={delta_macro:.4f}, "
            f"wrong_to_correct={wrong_to_correct}, correct_to_wrong={correct_to_wrong}, net_gain={net_gain}."
        )
        recommendation = (
            "Do not move to reranker-like Phase D yet. Revisit features/objective or consider other diagnosis."
        )

    return {
        "result": result,
        "reason": reason,
        "recommendation": recommendation,
        "best_policy": str(best["policy"]),
        "best_feature_set": str(best["feature_set"]),
        "best_confidence_threshold": None if pd.isna(best["confidence_threshold"]) else float(best["confidence_threshold"]),
        "best_accuracy": float(best["accuracy"]),
        "best_macro_f1": float(best["macro_f1"]),
        "best_weighted_f1": float(best["weighted_f1"]),
        "best_delta_accuracy": float(best["delta_accuracy"]),
        "best_delta_macro_f1": float(best["delta_macro_f1"]),
        "best_delta_weighted_f1": float(best["delta_weighted_f1"]),
        "best_wrong_to_correct_n": wrong_to_correct,
        "best_correct_to_wrong_n": correct_to_wrong,
        "best_net_gain_n": net_gain,
        "best_damage_ratio": damage_ratio,
        "guardrail": (
            "C3 is a validation OOF diagnostic. It measures whether a pairwise correction direction is promising, "
            "but it is not an official final model and not a train/val/test-proven solution."
        ),
    }


def to_md(df: pd.DataFrame, index: bool = False) -> str:
    try:
        return df.to_markdown(index=index)
    except Exception:
        return df.to_string(index=index)


def make_markdown(
    *,
    labels: List[str],
    cls_info: Dict[str, Any],
    policy_df: pd.DataFrame,
    per_class_df: pd.DataFrame,
    pairwise_cv_df: pd.DataFrame,
    transition_df: pd.DataFrame,
    pair_damage_df: pd.DataFrame,
    best_policy: str,
    gate: Dict[str, Any],
    out_files: List[Path],
) -> str:
    lines: List[str] = []
    lines.append("# C3 — External pairwise reranker diagnostic")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("Test whether external pairwise CLS/log-probability reranking can fix hard subtype mistakes without damaging too many already-correct samples.")
    lines.append("")
    lines.append("## Guardrail")
    lines.append("")
    lines.append(gate["guardrail"])
    lines.append("")
    lines.append("## Interpretation gate")
    lines.append("")
    lines.append(f"- Result: **{gate['result']}**")
    lines.append(f"- Reason: {gate['reason']}")
    lines.append(f"- Recommendation: {gate['recommendation']}")
    lines.append("")
    lines.append("## Best policy")
    lines.append("")
    for k in [
        "best_policy", "best_feature_set", "best_confidence_threshold",
        "best_accuracy", "best_macro_f1", "best_delta_macro_f1",
        "best_weighted_f1", "best_delta_weighted_f1",
        "best_wrong_to_correct_n", "best_correct_to_wrong_n",
        "best_net_gain_n", "best_damage_ratio",
    ]:
        lines.append(f"- `{k}`: {gate[k]}")
    lines.append("")
    lines.append("## Top policies by macro-F1 delta")
    lines.append("")
    top_policies = policy_df.sort_values(
        by=["delta_macro_f1", "net_gain_n", "correct_to_wrong_n"],
        ascending=[False, False, True],
    ).head(12)
    show_cols = [
        "policy", "feature_set", "confidence_threshold",
        "accuracy", "delta_accuracy",
        "macro_f1", "delta_macro_f1",
        "weighted_f1", "delta_weighted_f1",
        "n_changed", "wrong_to_correct_n", "correct_to_wrong_n", "net_gain_n", "damage_ratio",
    ]
    show_cols = [c for c in show_cols if c in top_policies.columns]
    lines.append(to_md(top_policies[show_cols], index=False))
    lines.append("")
    lines.append("## Best policy per-class F1")
    lines.append("")
    pc = per_class_df[per_class_df["policy"].isin(["original", best_policy])].copy()
    pc_cols = ["policy"] + [f"f1_{label}" for label in labels] + [f"delta_f1_{label}" for label in labels]
    pc_cols = [c for c in pc_cols if c in pc.columns]
    lines.append(to_md(pc[pc_cols], index=False))
    lines.append("")
    lines.append("## Best policy transition summary")
    lines.append("")
    ts = transition_df[transition_df["policy"].isin(["original", best_policy])].copy()
    lines.append(to_md(ts, index=False))
    lines.append("")
    lines.append("## Best policy pair-level fix/damage summary")
    lines.append("")
    pdmg = pair_damage_df[pair_damage_df["policy"] == best_policy].copy()
    dmg_cols = [
        "pair", "direction",
        "candidate_true_pair_n",
        "original_direct_wrong_n",
        "fixed_direct_wrong_n",
        "fix_rate_among_direct_wrong",
        "original_correct_candidate_n",
        "correct_to_wrong_damage_n",
        "damage_rate_among_correct_candidates",
        "wrong_to_wrong_changed_n",
        "net_pair_gain_n",
    ]
    dmg_cols = [c for c in dmg_cols if c in pdmg.columns]
    lines.append(to_md(pdmg[dmg_cols], index=False))
    lines.append("")
    lines.append("## Pairwise classifier OOF metrics")
    lines.append("")
    cv_summary = pairwise_cv_df.copy()
    cv_summary = cv_summary.sort_values(["feature_set", "pair"])
    cv_cols = ["feature_set", "pair", "dim", "accuracy", "balanced_accuracy", "macro_f1", "auc"]
    cv_cols = [c for c in cv_cols if c in cv_summary.columns]
    lines.append(to_md(cv_summary[cv_cols], index=False))
    lines.append("")
    lines.append("## How to read this")
    lines.append("")
    lines.append("- `wrong_to_correct_n`: original baseline was wrong, reranker makes it correct.")
    lines.append("- `correct_to_wrong_n`: original baseline was correct, reranker breaks it.")
    lines.append("- `net_gain_n = wrong_to_correct_n - correct_to_wrong_n`.")
    lines.append("- A good direction must improve hard-pair errors without a large correct-to-wrong cost.")
    lines.append("- Pair-level rows show whether each hard pair improves and whether already-correct samples in that pair are damaged.")
    lines.append("")
    lines.append("## CLS input info")
    lines.append("")
    lines.append(f"- `{cls_info}`")
    lines.append("")
    lines.append("## Generated files")
    lines.append("")
    for p in out_files:
        lines.append(f"- `{p}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    pred_csv = repo_path(repo_root, args.pred_csv)
    cls_npz = repo_path(repo_root, args.cls_npz)
    out_dir = repo_path(repo_root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pred_csv.exists():
        raise FileNotFoundError(f"Missing pred csv: {pred_csv}")

    pairs = parse_pairs(args.pairs)
    thresholds = sorted(set(float(x) for x in args.confidence_thresholds))

    df0 = pd.read_csv(pred_csv)
    labels = infer_label_order(df0, args.label_order, args.score_prefix)
    df, probs, logprobs = build_augmented_predictions(df0, labels, args.score_prefix)

    cls, cls_info = load_cls_embeddings(cls_npz)
    if len(cls) != len(df):
        raise ValueError(f"CLS rows != pred CSV rows: {len(cls)} vs {len(df)}")

    feature_sets = make_feature_sets(cls, probs, logprobs, args.feature_sets)

    policy_df, per_class_df, pairwise_cv_df, transition_df, pair_damage_df, reranked_by_policy, changed_by_policy = make_policy_tables(
        df,
        labels,
        pairs,
        feature_sets,
        confidence_thresholds=thresholds,
        cv_folds=int(args.cv_folds),
        random_state=int(args.random_state),
        max_iter=int(args.max_iter),
    )

    best = choose_best_policy(policy_df)
    best_policy = str(best["policy"])
    gate = make_gate(best, policy_df)

    # Output paths.
    summary_path = out_dir / "C3_summary.md"
    policy_path = out_dir / "C3_policy_metrics.csv"
    per_class_path = out_dir / "C3_policy_per_class_f1.csv"
    pairwise_cv_path = out_dir / "C3_pairwise_cv_metrics.csv"
    transition_path = out_dir / "C3_transition_summary.csv"
    pair_damage_path = out_dir / "C3_pair_fix_damage_summary.csv"
    gate_path = out_dir / "C3_gate_decision.json"
    manifest_path = out_dir / "C3_run_manifest.json"
    best_changed_path = out_dir / "C3_best_changed_samples.csv"
    original_cm_path = out_dir / "C3_original_confusion_matrix.csv"
    best_cm_path = out_dir / "C3_best_confusion_matrix.csv"

    policy_df.to_csv(policy_path, index=False)
    per_class_df.to_csv(per_class_path, index=False)
    pairwise_cv_df.to_csv(pairwise_cv_path, index=False)
    transition_df.to_csv(transition_path, index=False)
    pair_damage_df.to_csv(pair_damage_path, index=False)
    gate_path.write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")

    changed_best = changed_by_policy.get(best_policy, pd.DataFrame())
    changed_best.to_csv(best_changed_path, index=False)

    original_cm = confusion_df(df["true_label"].to_numpy(), df["pred_label"].to_numpy(), labels)
    best_cm = confusion_df(df["true_label"].to_numpy(), reranked_by_policy[best_policy].to_numpy(), labels)
    original_cm.to_csv(original_cm_path)
    best_cm.to_csv(best_cm_path)

    manifest = {
        "stage": "C3_external_pairwise_reranker_diagnostic",
        "purpose": "Validation OOF diagnostic for pairwise reranking direction; not official final model.",
        "inputs": {
            "pred_csv": str(pred_csv),
            "cls_npz": str(cls_npz),
            "score_prefix": args.score_prefix,
        },
        "labels": labels,
        "hard_pairs": [f"{a}<->{b}" for a, b in pairs],
        "feature_sets": list(feature_sets.keys()),
        "confidence_thresholds": thresholds,
        "cv_folds": int(args.cv_folds),
        "random_state": int(args.random_state),
        "cls_info": cls_info,
        "gate": gate,
        "baseline_guardrail": "This diagnostic should be run on 03_outputs/06_model baseline 0.81, not the separate 0.817 model until the best direction is chosen.",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    out_files = [
        summary_path,
        policy_path,
        per_class_path,
        pairwise_cv_path,
        transition_path,
        pair_damage_path,
        gate_path,
        manifest_path,
        best_changed_path,
        original_cm_path,
        best_cm_path,
    ]

    summary = make_markdown(
        labels=labels,
        cls_info=cls_info,
        policy_df=policy_df,
        per_class_df=per_class_df,
        pairwise_cv_df=pairwise_cv_df,
        transition_df=transition_df,
        pair_damage_df=pair_damage_df,
        best_policy=best_policy,
        gate=gate,
        out_files=out_files,
    )
    summary_path.write_text(summary, encoding="utf-8")

    out_zip = zip_outputs(out_dir)

    print("===== C3 external pairwise reranker diagnostic done =====")
    print("summary:", summary_path)
    print("policy_metrics:", policy_path)
    print("gate:", gate_path)
    print("zip:", out_zip)
    print("result:", gate["result"])
    print("best_policy:", gate["best_policy"])
    print("best_macro_f1:", gate["best_macro_f1"])
    print("best_delta_macro_f1:", gate["best_delta_macro_f1"])
    print("best_wrong_to_correct_n:", gate["best_wrong_to_correct_n"])
    print("best_correct_to_wrong_n:", gate["best_correct_to_wrong_n"])
    print("best_net_gain_n:", gate["best_net_gain_n"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
