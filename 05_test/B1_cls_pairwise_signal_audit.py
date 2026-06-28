#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B1_cls_pairwise_signal_audit.py

Purpose
-------
Audit whether the fresh CLS embedding space from the official C2+D3 model still
contains pairwise signal for hard malware subtype boundaries.

This is a diagnostic-only test under 05_test. It does NOT modify official source
or training files.

Inputs
------
Default:
  05_test/outputs/B1_cls_pairwise_signal/val_cls_embeddings.npz
  05_test/outputs/B1_cls_pairwise_signal/val_cls_predictions_with_probs.csv

Outputs
-------
  B1_summary.md
  B1_pairwise_logreg_cv_metrics.csv
  B1_centroid_distance.csv
  B1_wrong_direction_centroid_behavior.csv
  B1_hard_pair_summary.csv
  B1_gate_decision.json
  B1_cls_pairwise_signal_output.zip

Method
------
For each hard pair:
  1. Pairwise linear separability in CLS space:
     StandardScaler + LogisticRegression(class_weight="balanced"), stratified CV.

  2. Centroid behavior:
     Centroids are computed from correctly predicted validation samples for each
     class. Wrong samples are checked for whether they are closer to their true
     class centroid or predicted class centroid.

Interpretation
--------------
If pairwise linear CV metrics are strong while wrong true-in-top2 is high, then
CLS still contains pairwise boundary signal. This supports testing reranking or
pairwise auxiliary heads next.

If pairwise CV metrics are weak and wrong samples are mostly closer to predicted
centroids, then the representation itself is mixed; B2 raw/token/offset signal
audit should come before model changes.
"""

from __future__ import annotations

import argparse
import json
import math
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="B1 CLS pairwise signal audit.")
    p.add_argument("--repo-root", default=".")
    p.add_argument("--cls-npz", default="05_test/outputs/B1_cls_pairwise_signal/val_cls_embeddings.npz")
    p.add_argument("--pred-csv", default="05_test/outputs/B1_cls_pairwise_signal/val_cls_predictions_with_probs.csv")
    p.add_argument("--out-dir", default="05_test/outputs/B1_cls_pairwise_signal")
    p.add_argument("--pairs", nargs="*", default=None,
                   help='Optional hard pairs as "A:B", e.g. "Ransomware:Trojan". Default malware pairs.')
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--min-class-count", type=int, default=20)
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


def load_cls_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CLS npz: {path}")
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def require_npz_keys(data: Dict[str, np.ndarray], keys: Iterable[str]) -> None:
    missing = [k for k in keys if k not in data]
    if missing:
        raise KeyError(f"Missing keys in CLS npz: {missing}. Available: {sorted(data.keys())}")


def label_names_from_npz(data: Dict[str, np.ndarray]) -> List[str]:
    if "label_names" not in data:
        y = data["y_true"].astype(int)
        n = int(y.max()) + 1
        return [f"class_{i}" for i in range(n)]
    arr = data["label_names"]
    return [normalize_label(x) for x in arr.tolist()]


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b.reshape(1, -1), axis=1)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = np.linalg.norm(a, axis=1)
    bn = np.linalg.norm(b)
    denom = np.maximum(an * bn, 1e-12)
    sim = (a @ b) / denom
    return 1.0 - sim


def zip_outputs(out_dir: Path, zip_name: str = "B1_cls_pairwise_signal_output.zip") -> Path:
    out_zip = out_dir / zip_name
    if out_zip.exists():
        out_zip.unlink()

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fp in out_dir.rglob("*"):
            if fp.is_file() and fp != out_zip:
                z.write(fp, fp.relative_to(out_dir))
    return out_zip


def compute_pairwise_logreg_cv(
    X: np.ndarray,
    y_binary: np.ndarray,
    *,
    cv_folds: int,
    random_state: int,
    min_class_count: int,
) -> Dict[str, Any]:
    """
    Return stratified-CV logistic regression metrics.
    Requires scikit-learn; error message tells user how to install it.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            f1_score,
            roc_auc_score,
        )
        from sklearn.model_selection import StratifiedKFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "B1 pairwise audit requires scikit-learn. Install with: pip install scikit-learn"
        ) from e

    counts = np.bincount(y_binary.astype(int), minlength=2)
    min_count = int(counts.min())

    result: Dict[str, Any] = {
        "n_samples": int(len(y_binary)),
        "class0_count": int(counts[0]),
        "class1_count": int(counts[1]),
        "min_class_count": min_count,
        "cv_status": "ok",
    }

    if min_count < min_class_count:
        result.update({
            "cv_status": f"skipped_min_class_count_lt_{min_class_count}",
            "cv_folds": 0,
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "macro_f1": float("nan"),
            "auc": float("nan"),
            "coef_l2_norm": float("nan"),
        })
        return result

    k = int(min(cv_folds, min_count))
    if k < 2:
        result.update({
            "cv_status": "skipped_less_than_2_folds_possible",
            "cv_folds": k,
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "macro_f1": float("nan"),
            "auc": float("nan"),
            "coef_l2_norm": float("nan"),
        })
        return result

    oof_pred = np.zeros_like(y_binary, dtype=int)
    oof_score = np.zeros_like(y_binary, dtype=float)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    for train_idx, test_idx in skf.split(X, y_binary):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=random_state,
            ),
        )
        clf.fit(X[train_idx], y_binary[train_idx])
        oof_pred[test_idx] = clf.predict(X[test_idx])
        oof_score[test_idx] = clf.predict_proba(X[test_idx])[:, 1]

    full_clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=random_state,
        ),
    )
    full_clf.fit(X, y_binary)
    lr = full_clf.named_steps["logisticregression"]
    coef_l2 = float(np.linalg.norm(lr.coef_))

    result.update({
        "cv_folds": k,
        "accuracy": float(accuracy_score(y_binary, oof_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_binary, oof_pred)),
        "macro_f1": float(f1_score(y_binary, oof_pred, average="macro")),
        "auc": float(roc_auc_score(y_binary, oof_score)),
        "coef_l2_norm": coef_l2,
    })
    return result


def make_centroids(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str],
) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict[str, Any]]]:
    centroids: Dict[int, np.ndarray] = {}
    info: Dict[int, Dict[str, Any]] = {}

    for cid, label in enumerate(label_names):
        correct_mask = (y_true == cid) & (y_pred == cid)
        all_true_mask = y_true == cid

        if int(correct_mask.sum()) >= 2:
            use_mask = correct_mask
            source = "correct_val_samples"
        elif int(all_true_mask.sum()) >= 1:
            use_mask = all_true_mask
            source = "all_true_val_samples_fallback"
        else:
            continue

        centroids[cid] = X[use_mask].mean(axis=0)
        info[cid] = {
            "class_id": int(cid),
            "class_label": label,
            "source": source,
            "n_used": int(use_mask.sum()),
            "n_true_total": int(all_true_mask.sum()),
            "n_correct": int(correct_mask.sum()),
        }

    return centroids, info


def pair_centroid_rows(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_to_id: Dict[str, int],
    pairs: List[Tuple[str, str]],
    centroids: Dict[int, np.ndarray],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for a, b in pairs:
        aid, bid = label_to_id[a], label_to_id[b]
        if aid not in centroids or bid not in centroids:
            rows.append({
                "pair": f"{a}<->{b}",
                "class_a": a,
                "class_b": b,
                "status": "missing_centroid",
                "centroid_distance_euclidean": float("nan"),
                "centroid_distance_cosine": float("nan"),
            })
            continue

        ca, cb = centroids[aid], centroids[bid]
        rows.append({
            "pair": f"{a}<->{b}",
            "class_a": a,
            "class_b": b,
            "status": "ok",
            "centroid_distance_euclidean": float(np.linalg.norm(ca - cb)),
            "centroid_distance_cosine": float(1.0 - ((ca @ cb) / max(np.linalg.norm(ca) * np.linalg.norm(cb), 1e-12))),
        })

    return rows


def direction_centroid_behavior(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    top2_hit: np.ndarray,
    label_to_id: Dict[str, int],
    pairs: List[Tuple[str, str]],
    centroids: Dict[int, np.ndarray],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for a, b in pairs:
        for true_label, pred_label in [(a, b), (b, a)]:
            tid = label_to_id[true_label]
            pid = label_to_id[pred_label]
            mask = (y_true == tid) & (y_pred == pid)

            row_base = {
                "pair": f"{a}<->{b}",
                "direction": f"{true_label}->{pred_label}",
                "true_class": true_label,
                "pred_class": pred_label,
                "n_wrong": int(mask.sum()),
            }

            if int(mask.sum()) == 0:
                rows.append({
                    **row_base,
                    "status": "no_wrong_samples",
                    "wrong_true_in_top2": 0,
                    "wrong_true_in_top2_rate": float("nan"),
                    "true_centroid_source_available": tid in centroids,
                    "pred_centroid_source_available": pid in centroids,
                    "true_closer_than_pred_n": 0,
                    "true_closer_than_pred_rate": float("nan"),
                    "mean_dist_to_true_centroid": float("nan"),
                    "mean_dist_to_pred_centroid": float("nan"),
                    "mean_distance_margin_pred_minus_true": float("nan"),
                    "median_distance_margin_pred_minus_true": float("nan"),
                    "mean_cosdist_to_true_centroid": float("nan"),
                    "mean_cosdist_to_pred_centroid": float("nan"),
                    "mean_cosdist_margin_pred_minus_true": float("nan"),
                })
                continue

            true_in_top2_n = int(top2_hit[mask].sum())
            if tid not in centroids or pid not in centroids:
                rows.append({
                    **row_base,
                    "status": "missing_centroid",
                    "wrong_true_in_top2": true_in_top2_n,
                    "wrong_true_in_top2_rate": safe_div(true_in_top2_n, int(mask.sum())),
                    "true_centroid_source_available": tid in centroids,
                    "pred_centroid_source_available": pid in centroids,
                    "true_closer_than_pred_n": 0,
                    "true_closer_than_pred_rate": float("nan"),
                    "mean_dist_to_true_centroid": float("nan"),
                    "mean_dist_to_pred_centroid": float("nan"),
                    "mean_distance_margin_pred_minus_true": float("nan"),
                    "median_distance_margin_pred_minus_true": float("nan"),
                    "mean_cosdist_to_true_centroid": float("nan"),
                    "mean_cosdist_to_pred_centroid": float("nan"),
                    "mean_cosdist_margin_pred_minus_true": float("nan"),
                })
                continue

            Xw = X[mask]
            d_true = euclidean_distance(Xw, centroids[tid])
            d_pred = euclidean_distance(Xw, centroids[pid])
            margin = d_pred - d_true  # positive means closer to true than pred

            cd_true = cosine_distance(Xw, centroids[tid])
            cd_pred = cosine_distance(Xw, centroids[pid])
            cmargin = cd_pred - cd_true

            true_closer = d_true < d_pred

            rows.append({
                **row_base,
                "status": "ok",
                "wrong_true_in_top2": true_in_top2_n,
                "wrong_true_in_top2_rate": safe_div(true_in_top2_n, int(mask.sum())),
                "true_centroid_source_available": True,
                "pred_centroid_source_available": True,
                "true_closer_than_pred_n": int(true_closer.sum()),
                "true_closer_than_pred_rate": safe_div(int(true_closer.sum()), int(mask.sum())),
                "mean_dist_to_true_centroid": float(d_true.mean()),
                "mean_dist_to_pred_centroid": float(d_pred.mean()),
                "mean_distance_margin_pred_minus_true": float(margin.mean()),
                "median_distance_margin_pred_minus_true": float(np.median(margin)),
                "mean_cosdist_to_true_centroid": float(cd_true.mean()),
                "mean_cosdist_to_pred_centroid": float(cd_pred.mean()),
                "mean_cosdist_margin_pred_minus_true": float(cmargin.mean()),
            })

    return rows


def official_pair_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    top2_hit: np.ndarray,
    label_to_id: Dict[str, int],
    pairs: List[Tuple[str, str]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for a, b in pairs:
        aid, bid = label_to_id[a], label_to_id[b]
        pair_true_mask = (y_true == aid) | (y_true == bid)
        n = int(pair_true_mask.sum())
        correct = (y_true == y_pred) & pair_true_mask
        pair_confused = (
            ((y_true == aid) & (y_pred == bid)) |
            ((y_true == bid) & (y_pred == aid))
        )
        pred_outside_pair = pair_true_mask & ~((y_pred == aid) | (y_pred == bid))

        rows.append({
            "pair": f"{a}<->{b}",
            "class_a": a,
            "class_b": b,
            "n_true_pair": n,
            "official_correct_n": int(correct.sum()),
            "official_correct_rate": safe_div(int(correct.sum()), n),
            "official_pair_confusion_n": int(pair_confused.sum()),
            "official_pair_confusion_rate": safe_div(int(pair_confused.sum()), n),
            "official_pred_outside_pair_n": int(pred_outside_pair.sum()),
            "official_pred_outside_pair_rate": safe_div(int(pred_outside_pair.sum()), n),
            "true_in_top2_n": int((top2_hit & pair_true_mask).sum()),
            "true_in_top2_rate": safe_div(int((top2_hit & pair_true_mask).sum()), n),
        })

    return rows


def make_markdown_summary(
    *,
    main_metrics: Dict[str, Any],
    pair_logreg_df: pd.DataFrame,
    centroid_df: pd.DataFrame,
    direction_df: pd.DataFrame,
    official_pair_df: pd.DataFrame,
    gate: Dict[str, Any],
    out_files: List[Path],
) -> str:
    lines: List[str] = []
    lines.append("# B1 — CLS pairwise signal audit")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("Check whether the fresh CLS embedding space from the official C2+D3 model still contains pairwise signal for hard malware subtype boundaries.")
    lines.append("")
    lines.append("## Main metrics")
    lines.append("")
    for k, v in main_metrics.items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("## Interpretation gate")
    lines.append("")
    lines.append(f"- Result: **{gate['result']}**")
    lines.append(f"- Reason: {gate['reason']}")
    lines.append("")
    lines.append("## Official model behavior inside each hard pair")
    lines.append("")
    lines.append(official_pair_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Pairwise linear separability in CLS space")
    lines.append("")
    show_cols = [
        "pair", "n_samples", "class0_count", "class1_count", "cv_folds",
        "accuracy", "balanced_accuracy", "macro_f1", "auc", "cv_status",
    ]
    lines.append(pair_logreg_df[show_cols].to_markdown(index=False))
    lines.append("")
    lines.append("## Correct-sample centroid distances")
    lines.append("")
    lines.append(centroid_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Wrong-direction centroid behavior")
    lines.append("")
    show_dir_cols = [
        "pair", "direction", "n_wrong", "wrong_true_in_top2_rate",
        "true_closer_than_pred_rate", "mean_distance_margin_pred_minus_true",
        "median_distance_margin_pred_minus_true", "status",
    ]
    lines.append(direction_df[show_dir_cols].to_markdown(index=False))
    lines.append("")
    lines.append("## How to read centroid margin")
    lines.append("")
    lines.append("- `mean_distance_margin_pred_minus_true = dist_to_pred_centroid - dist_to_true_centroid`.")
    lines.append("- Positive value: wrong samples are closer to their true class centroid than predicted class centroid.")
    lines.append("- Negative value: wrong samples are closer to the predicted class centroid, suggesting representation-level pull toward the wrong class.")
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
    cls_npz = repo_path(repo_root, args.cls_npz)
    pred_csv = repo_path(repo_root, args.pred_csv)
    out_dir = repo_path(repo_root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pred_csv.exists():
        raise FileNotFoundError(f"Missing prediction CSV: {pred_csv}")

    pairs = parse_pairs(args.pairs)
    data = load_cls_npz(cls_npz)
    require_npz_keys(data, ["cls_embeddings", "y_true", "y_pred", "top1_id", "top2_id"])

    X = data["cls_embeddings"].astype(np.float32)
    y_true = data["y_true"].astype(np.int64)
    y_pred = data["y_pred"].astype(np.int64)
    top1 = data["top1_id"].astype(np.int64)
    top2 = data["top2_id"].astype(np.int64)

    label_names = label_names_from_npz(data)
    label_to_id = {name: i for i, name in enumerate(label_names)}

    unknown = [x for pair in pairs for x in pair if x not in label_to_id]
    if unknown:
        raise KeyError(f"Unknown labels in pairs: {unknown}; available label_names={label_names}")

    pred_df = pd.read_csv(pred_csv)
    # Prefer explicit CSV true_in_top2 if present; otherwise compute from NPZ.
    if "true_in_top2" in pred_df.columns:
        csv_top2 = pred_df["true_in_top2"].astype(bool).to_numpy()
        if len(csv_top2) == len(y_true):
            top2_hit = csv_top2
        else:
            top2_hit = (top1 == y_true) | (top2 == y_true)
    else:
        top2_hit = (top1 == y_true) | (top2 == y_true)

    correct = y_true == y_pred
    wrong = ~correct

    main_metrics = {
        "n_total": int(len(y_true)),
        "cls_dim": int(X.shape[1]),
        "n_correct": int(correct.sum()),
        "n_wrong": int(wrong.sum()),
        "accuracy_from_cls_export": round(float(correct.mean()), 10),
        "top2_accuracy_from_cls_export": round(float(top2_hit.mean()), 10),
        "wrong_true_in_top2": int((wrong & top2_hit).sum()),
        "wrong_true_in_top2_rate": round(safe_div(int((wrong & top2_hit).sum()), int(wrong.sum())), 10),
        "pairs": [f"{a}<->{b}" for a, b in pairs],
    }

    # Official pair behavior.
    official_pair_df = pd.DataFrame(
        official_pair_rows(y_true, y_pred, top2_hit, label_to_id, pairs)
    )

    # Pairwise logreg.
    pair_rows: List[Dict[str, Any]] = []
    for a, b in pairs:
        aid, bid = label_to_id[a], label_to_id[b]
        mask = (y_true == aid) | (y_true == bid)
        X_pair = X[mask]
        y_pair = (y_true[mask] == bid).astype(int)

        metrics = compute_pairwise_logreg_cv(
            X_pair,
            y_pair,
            cv_folds=int(args.cv_folds),
            random_state=int(args.random_state),
            min_class_count=int(args.min_class_count),
        )
        pair_rows.append({
            "pair": f"{a}<->{b}",
            "class0": a,
            "class1": b,
            **metrics,
        })

    pair_logreg_df = pd.DataFrame(pair_rows)

    # Centroids and wrong behavior.
    centroids, centroid_info = make_centroids(X, y_true, y_pred, label_names)

    centroid_info_df = pd.DataFrame(list(centroid_info.values()))
    centroid_info_path = out_dir / "B1_centroid_sources.csv"
    centroid_info_df.to_csv(centroid_info_path, index=False)

    centroid_df = pd.DataFrame(
        pair_centroid_rows(X, y_true, y_pred, label_to_id, pairs, centroids)
    )

    direction_df = pd.DataFrame(
        direction_centroid_behavior(X, y_true, y_pred, top2_hit, label_to_id, pairs, centroids)
    )

    hard_pair_df = direction_df.merge(
        pair_logreg_df[["pair", "balanced_accuracy", "macro_f1", "auc", "cv_status"]],
        on="pair",
        how="left",
        suffixes=("", "_pair_logreg"),
    )

    # Gate decision.
    ok_logreg = pair_logreg_df["cv_status"].eq("ok")
    mean_macro_f1 = float(pair_logreg_df.loc[ok_logreg, "macro_f1"].mean()) if ok_logreg.any() else float("nan")
    mean_auc = float(pair_logreg_df.loc[ok_logreg, "auc"].mean()) if ok_logreg.any() else float("nan")
    min_macro_f1 = float(pair_logreg_df.loc[ok_logreg, "macro_f1"].min()) if ok_logreg.any() else float("nan")
    mean_wrong_top2 = float(direction_df.loc[direction_df["status"].eq("ok"), "wrong_true_in_top2_rate"].mean())

    # Conservative diagnostic threshold. These are gates, not official metrics.
    if ok_logreg.any() and mean_macro_f1 >= 0.80 and mean_auc >= 0.88 and mean_wrong_top2 >= 0.65:
        result = "PASS — CLS has usable pairwise signal"
        reason = (
            f"Mean pairwise LogisticRegression macro-F1={mean_macro_f1:.4f}, "
            f"mean AUC={mean_auc:.4f}, and mean wrong-direction true-in-top2={mean_wrong_top2:.4f}. "
            "This supports testing reranking or auxiliary pairwise heads."
        )
    elif ok_logreg.any() and mean_macro_f1 >= 0.72 and mean_auc >= 0.80:
        result = "MIXED — CLS has some pairwise signal but boundary is not clearly strong"
        reason = (
            f"Mean pairwise LogisticRegression macro-F1={mean_macro_f1:.4f}, "
            f"mean AUC={mean_auc:.4f}. Test B2 raw/token/offset before committing to model changes."
        )
    else:
        result = "FAIL — CLS pairwise signal appears weak"
        reason = (
            f"Mean pairwise LogisticRegression macro-F1={mean_macro_f1:.4f}, "
            f"mean AUC={mean_auc:.4f}. Representation may be mixed; run B2 input-space signal audit next."
        )

    gate = {
        "result": result,
        "reason": reason,
        "mean_pairwise_logreg_macro_f1": mean_macro_f1,
        "min_pairwise_logreg_macro_f1": min_macro_f1,
        "mean_pairwise_logreg_auc": mean_auc,
        "mean_wrong_direction_true_in_top2_rate": mean_wrong_top2,
        "thresholds": {
            "pass_mean_macro_f1": 0.80,
            "pass_mean_auc": 0.88,
            "pass_mean_wrong_top2": 0.65,
            "mixed_mean_macro_f1": 0.72,
            "mixed_mean_auc": 0.80,
        },
        "note": "Gate is diagnostic only, not an official validation metric.",
    }

    # Write files.
    pair_logreg_path = out_dir / "B1_pairwise_logreg_cv_metrics.csv"
    centroid_path = out_dir / "B1_centroid_distance.csv"
    direction_path = out_dir / "B1_wrong_direction_centroid_behavior.csv"
    official_pair_path = out_dir / "B1_official_pair_behavior.csv"
    hard_pair_path = out_dir / "B1_hard_pair_summary.csv"
    gate_path = out_dir / "B1_gate_decision.json"
    metrics_path = out_dir / "B1_metrics.json"

    pair_logreg_df.to_csv(pair_logreg_path, index=False)
    centroid_df.to_csv(centroid_path, index=False)
    direction_df.to_csv(direction_path, index=False)
    official_pair_df.to_csv(official_pair_path, index=False)
    hard_pair_df.to_csv(hard_pair_path, index=False)
    gate_path.write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics = {
        "main_metrics": main_metrics,
        "gate": gate,
        "label_names": label_names,
        "pairs": pairs,
        "inputs": {
            "cls_npz": str(cls_npz),
            "pred_csv": str(pred_csv),
        },
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_path = out_dir / "B1_summary.md"
    out_files = [
        summary_path,
        metrics_path,
        pair_logreg_path,
        centroid_path,
        centroid_info_path,
        direction_path,
        official_pair_path,
        hard_pair_path,
        gate_path,
    ]
    summary_md = make_markdown_summary(
        main_metrics=main_metrics,
        pair_logreg_df=pair_logreg_df,
        centroid_df=centroid_df,
        direction_df=direction_df,
        official_pair_df=official_pair_df,
        gate=gate,
        out_files=out_files,
    )
    summary_path.write_text(summary_md, encoding="utf-8")

    out_zip = zip_outputs(out_dir)

    print("===== B1 CLS pairwise signal audit done =====")
    print("summary:", summary_path)
    print("gate:", gate_path)
    print("zip:", out_zip)
    print("result:", gate["result"])
    print("mean_pairwise_logreg_macro_f1:", gate["mean_pairwise_logreg_macro_f1"])
    print("mean_pairwise_logreg_auc:", gate["mean_pairwise_logreg_auc"])
    print("mean_wrong_direction_true_in_top2_rate:", gate["mean_wrong_direction_true_in_top2_rate"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
