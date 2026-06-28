#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B2_input_pairwise_signal_audit.py

Purpose
-------
Audit whether raw/token/offset input spaces still contain pairwise signal for
hard malware subtype boundaries.

This is Phase B diagnostic only:
  - no model training
  - no baseline source modification
  - no reranking/model-head solution
  - all outputs go under 05_test/outputs

Representations tested by default
---------------------------------
For each validation sample:
  1. raw_scaled
     - raw features scaled by train-only minmax, val clipped to [0,1]
  2. bin_norm
     - X_val_bin normalized by (num_bins - 1)
  3. offset
     - X_val_offset
  4. bin_plus_offset_norm
     - (X_val_bin + X_val_offset) / num_bins
  5. bin_norm__offset
     - concat(bin_norm, offset)
  6. raw_scaled__bin_plus_offset_norm
     - concat(raw_scaled, bin_plus_offset_norm)
  7. raw_scaled__bin_norm__offset
     - concat(raw_scaled, bin_norm, offset)
  8. d3_scalar_input
     - concat(bin_norm, offset, raw_scaled, mask)

Method
------
For each hard pair:
  - Filter validation samples whose true label is one of the two classes.
  - Fit StandardScaler + LogisticRegression(class_weight="balanced").
  - Evaluate with Stratified K-fold CV.
  - Report accuracy, balanced accuracy, macro-F1, AUC.

Interpretation
--------------
B2 answers: "Does pairwise signal still exist in input spaces?"
It does not propose or apply a fix.

If input spaces are strong, then the pipeline still has usable information before
the Transformer/classifier. Later B3 can compare this with CLS to locate whether
representation learning improves or weakens pairwise boundaries.
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="B2 raw/token/offset pairwise signal audit.")
    p.add_argument("--repo-root", default=".")
    p.add_argument("--dataset-npz", default="03_outputs/05_dataset/dataset.npz")
    p.add_argument("--metadata-json", default="03_outputs/05_dataset/metadata.json")
    p.add_argument("--train-raw", default="01_split/train_raw.csv")
    p.add_argument("--val-raw", default="01_split/val_raw.csv")
    p.add_argument("--pred-csv", default="05_test/outputs/B1_cls_pairwise_signal/val_cls_predictions_with_probs.csv",
                   help="Optional model prediction CSV for official-pair behavior context.")
    p.add_argument("--out-dir", default="05_test/outputs/B2_input_pairwise_signal")
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


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_dataset_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset npz: {path}")
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def get_required_array(data: Dict[str, np.ndarray], key: str) -> np.ndarray:
    if key not in data:
        raise KeyError(f"dataset npz missing required array: {key}. Available keys: {sorted(data.keys())}")
    return data[key]


def infer_num_bins(meta: Dict[str, Any], data: Dict[str, np.ndarray]) -> int:
    for key in ("num_bins", "K", "effective_token_budget", "K_artifact"):
        if key in meta:
            try:
                return int(meta[key])
            except Exception:
                pass
    if "X_val_bin" in data:
        return int(np.max(data["X_val_bin"])) + 1
    return 512


def label_names_from_meta(meta: Dict[str, Any], y_val: np.ndarray) -> List[str]:
    mapping = meta.get("label_mapping")
    if isinstance(mapping, dict) and mapping:
        inv = {int(v): normalize_label(k) for k, v in mapping.items()}
        n = max(max(inv.keys()) + 1, int(y_val.max()) + 1)
        if all(i in inv for i in range(n)):
            return [inv[i] for i in range(n)]

    names = meta.get("label_names")
    if isinstance(names, list):
        return [normalize_label(x) for x in names]

    n = int(y_val.max()) + 1
    return [f"class_{i}" for i in range(n)]


def compute_raw_scaled_continuous(
    meta: Dict[str, Any],
    train_raw_path: Path,
    val_raw_path: Path,
    expected_shape: Tuple[int, int],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if "feature_names" not in meta:
        raise KeyError("metadata missing feature_names; cannot compute raw_scaled")

    feature_names = [str(x) for x in meta["feature_names"]]
    if not train_raw_path.exists():
        raise FileNotFoundError(f"train_raw not found: {train_raw_path}")
    if not val_raw_path.exists():
        raise FileNotFoundError(f"val_raw not found: {val_raw_path}")

    train_df = pd.read_csv(train_raw_path)
    val_df = pd.read_csv(val_raw_path)

    missing_train = [f for f in feature_names if f not in train_df.columns]
    missing_val = [f for f in feature_names if f not in val_df.columns]
    if missing_train:
        raise KeyError(f"train_raw missing features: {missing_train[:20]}")
    if missing_val:
        raise KeyError(f"val_raw missing features: {missing_val[:20]}")

    X_train_raw = train_df.loc[:, feature_names].to_numpy(dtype=np.float64)
    X_val_raw = val_df.loc[:, feature_names].to_numpy(dtype=np.float64)

    if X_val_raw.shape != expected_shape:
        raise ValueError(f"val_raw feature shape mismatch: {X_val_raw.shape} vs expected {expected_shape}")

    if not np.isfinite(X_train_raw).all():
        raise ValueError("train_raw selected features contain NaN/Inf")
    if not np.isfinite(X_val_raw).all():
        raise ValueError("val_raw selected features contain NaN/Inf")

    mn = X_train_raw.min(axis=0)
    mx = X_train_raw.max(axis=0)
    denom = mx - mn
    constant = np.isclose(denom, 0.0)
    denom_safe = denom.copy()
    denom_safe[constant] = 1.0

    X_val_scaled = (X_val_raw - mn) / denom_safe
    X_val_scaled[:, constant] = 0.5
    X_val_scaled = np.clip(X_val_scaled, 0.0, 1.0).astype(np.float32)

    info = {
        "source": "raw_scaled_from_raw_csv",
        "scale": "train_only_minmax_linear_clip_val",
        "train_raw": str(train_raw_path),
        "val_raw": str(val_raw_path),
        "n_constant_features": int(constant.sum()),
        "constant_features": [feature_names[i] for i, flag in enumerate(constant) if flag],
        "val_min": float(X_val_scaled.min()),
        "val_max": float(X_val_scaled.max()),
    }
    return X_val_scaled, info


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def zip_outputs(out_dir: Path, zip_name: str = "B2_input_pairwise_signal_output.zip") -> Path:
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
            "B2 input pairwise audit requires scikit-learn. Install with: pip install scikit-learn"
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


def build_representations(
    *,
    X_val_bin: np.ndarray,
    X_val_offset: np.ndarray,
    X_val_raw_scaled: np.ndarray,
    X_val_mask: np.ndarray,
    num_bins: int,
) -> Dict[str, np.ndarray]:
    denom_bin = max(int(num_bins) - 1, 1)
    denom_plus = max(int(num_bins), 1)

    bin_norm = np.clip(X_val_bin.astype(np.float32) / float(denom_bin), 0.0, 1.0)
    offset = np.clip(X_val_offset.astype(np.float32), 0.0, 1.0)
    raw_scaled = np.clip(X_val_raw_scaled.astype(np.float32), 0.0, 1.0)
    mask = np.clip(X_val_mask.astype(np.float32), 0.0, 1.0)

    bin_plus_offset_norm = np.clip(
        (X_val_bin.astype(np.float32) + offset) / float(denom_plus),
        0.0,
        1.0,
    )

    reps = {
        "raw_scaled": raw_scaled,
        "bin_norm": bin_norm,
        "offset": offset,
        "bin_plus_offset_norm": bin_plus_offset_norm,
        "bin_norm__offset": np.concatenate([bin_norm, offset], axis=1),
        "raw_scaled__bin_plus_offset_norm": np.concatenate([raw_scaled, bin_plus_offset_norm], axis=1),
        "raw_scaled__bin_norm__offset": np.concatenate([raw_scaled, bin_norm, offset], axis=1),
        "d3_scalar_input": np.concatenate([bin_norm, offset, raw_scaled, mask], axis=1),
    }

    # Ensure contiguous float32 for sklearn speed/consistency.
    return {k: np.ascontiguousarray(v.astype(np.float32)) for k, v in reps.items()}


def official_pair_rows_from_pred(
    pred_csv: Path,
    pairs: List[Tuple[str, str]],
) -> pd.DataFrame:
    if not pred_csv.exists():
        return pd.DataFrame([{
            "pair": f"{a}<->{b}",
            "status": f"missing_pred_csv:{pred_csv}",
        } for a, b in pairs])

    df = pd.read_csv(pred_csv)
    if "true_label" not in df.columns or "pred_label" not in df.columns:
        return pd.DataFrame([{
            "pair": f"{a}<->{b}",
            "status": "pred_csv_missing_true_label_or_pred_label",
        } for a, b in pairs])

    true_label = df["true_label"].map(normalize_label)
    pred_label = df["pred_label"].map(normalize_label)

    if "true_in_top2" in df.columns:
        true_in_top2 = df["true_in_top2"].astype(bool)
    elif "top1_label" in df.columns and "top2_label" in df.columns:
        top1 = df["top1_label"].map(normalize_label)
        top2 = df["top2_label"].map(normalize_label)
        true_in_top2 = (true_label == top1) | (true_label == top2)
    else:
        true_in_top2 = pd.Series([False] * len(df))

    rows: List[Dict[str, Any]] = []
    for a, b in pairs:
        pair_true = true_label.isin([a, b])
        n = int(pair_true.sum())
        correct = pair_true & (true_label == pred_label)
        pair_confused = (
            ((true_label == a) & (pred_label == b)) |
            ((true_label == b) & (pred_label == a))
        )
        pred_outside = pair_true & (~pred_label.isin([a, b]))

        rows.append({
            "pair": f"{a}<->{b}",
            "status": "ok",
            "n_true_pair": n,
            "official_correct_n": int(correct.sum()),
            "official_correct_rate": safe_div(int(correct.sum()), n),
            "official_pair_confusion_n": int(pair_confused.sum()),
            "official_pair_confusion_rate": safe_div(int(pair_confused.sum()), n),
            "official_pred_outside_pair_n": int(pred_outside.sum()),
            "official_pred_outside_pair_rate": safe_div(int(pred_outside.sum()), n),
            "true_in_top2_n": int((true_in_top2 & pair_true).sum()),
            "true_in_top2_rate": safe_div(int((true_in_top2 & pair_true).sum()), n),
        })

    return pd.DataFrame(rows)


def make_markdown_summary(
    *,
    main_metrics: Dict[str, Any],
    representation_summary_df: pd.DataFrame,
    best_by_pair_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    official_pair_df: pd.DataFrame,
    gate: Dict[str, Any],
    out_files: List[Path],
) -> str:
    lines: List[str] = []
    lines.append("# B2 — Raw/token/offset pairwise signal audit")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("Check whether pairwise malware-subtype signal exists in input spaces before the Transformer CLS representation.")
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
    lines.append("## Representation summary")
    lines.append("")
    show_rep_cols = [
        "representation", "dim", "mean_macro_f1", "mean_auc",
        "min_macro_f1", "max_macro_f1", "mean_balanced_accuracy",
    ]
    lines.append(representation_summary_df[show_rep_cols].to_markdown(index=False))
    lines.append("")
    lines.append("## Best input representation per pair")
    lines.append("")
    lines.append(best_by_pair_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Full pairwise metrics")
    lines.append("")
    show_metric_cols = [
        "representation", "pair", "dim", "n_samples", "class0_count", "class1_count",
        "accuracy", "balanced_accuracy", "macro_f1", "auc", "cv_status",
    ]
    lines.append(metrics_df[show_metric_cols].to_markdown(index=False))
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- B2 is diagnostic only. It does not apply reranking or model changes.")
    lines.append("- Strong input-space signal means useful class information still exists before the model representation.")
    lines.append("- B3 should compare these input-space metrics against B1 CLS metrics before deciding where the bottleneck is.")
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

    dataset_npz = repo_path(repo_root, args.dataset_npz)
    metadata_json = repo_path(repo_root, args.metadata_json)
    train_raw = repo_path(repo_root, args.train_raw)
    val_raw = repo_path(repo_root, args.val_raw)
    pred_csv = repo_path(repo_root, args.pred_csv)
    out_dir = repo_path(repo_root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    required = {
        "dataset_npz": dataset_npz,
        "metadata_json": metadata_json,
        "train_raw": train_raw,
        "val_raw": val_raw,
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    pairs = parse_pairs(args.pairs)

    meta = load_json(metadata_json)
    data = load_dataset_npz(dataset_npz)

    X_val_bin = get_required_array(data, "X_val_bin").astype(np.int64)
    X_val_offset = get_required_array(data, "X_val_offset").astype(np.float32)
    y_val = get_required_array(data, "y_val").astype(np.int64)

    if X_val_bin.shape != X_val_offset.shape:
        raise ValueError(f"X_val_bin and X_val_offset shape mismatch: {X_val_bin.shape} vs {X_val_offset.shape}")
    if len(y_val) != X_val_bin.shape[0]:
        raise ValueError(f"y_val length mismatch: {len(y_val)} vs rows {X_val_bin.shape[0]}")

    num_bins = infer_num_bins(meta, data)
    label_names = label_names_from_meta(meta, y_val)
    label_to_id = {name: i for i, name in enumerate(label_names)}

    unknown = [x for pair in pairs for x in pair if x not in label_to_id]
    if unknown:
        raise KeyError(f"Unknown labels in pairs: {unknown}; available label_names={label_names}")

    if "X_val_continuous" in data:
        X_val_raw_scaled = data["X_val_continuous"].astype(np.float32)
        raw_info = {
            "source": "X_val_continuous_from_dataset_npz",
            "val_min": float(X_val_raw_scaled.min()),
            "val_max": float(X_val_raw_scaled.max()),
        }
    else:
        X_val_raw_scaled, raw_info = compute_raw_scaled_continuous(
            meta=meta,
            train_raw_path=train_raw,
            val_raw_path=val_raw,
            expected_shape=X_val_bin.shape,
        )

    if "X_val_mask" in data:
        X_val_mask = data["X_val_mask"].astype(np.float32)
    else:
        X_val_mask = np.ones_like(X_val_offset, dtype=np.float32)

    reps = build_representations(
        X_val_bin=X_val_bin,
        X_val_offset=X_val_offset,
        X_val_raw_scaled=X_val_raw_scaled,
        X_val_mask=X_val_mask,
        num_bins=num_bins,
    )

    metric_rows: List[Dict[str, Any]] = []
    total_jobs = len(reps) * len(pairs)
    job_i = 0
    print(f"[B2] start pairwise audit: {len(reps)} representations x {len(pairs)} pairs = {total_jobs} jobs", flush=True)

    for rep_name, X_rep in reps.items():
        print(f"[B2] representation={rep_name} shape={X_rep.shape}", flush=True)

        for a, b in pairs:
            job_i += 1
            print(f"[B2] job {job_i}/{total_jobs}: {rep_name} | {a}<->{b}", flush=True)

            aid, bid = label_to_id[a], label_to_id[b]
            mask = (y_val == aid) | (y_val == bid)
            X_pair = X_rep[mask]
            y_pair = (y_val[mask] == bid).astype(int)

            metrics = compute_pairwise_logreg_cv(
                X_pair,
                y_pair,
                cv_folds=int(args.cv_folds),
                random_state=int(args.random_state),
                min_class_count=int(args.min_class_count),
            )

            print(
                f"[B2] done {rep_name} | {a}<->{b}: "
                f"macro_f1={metrics.get('macro_f1')} auc={metrics.get('auc')} status={metrics.get('cv_status')}",
                flush=True,
            )

            metric_rows.append({
                "representation": rep_name,
                "pair": f"{a}<->{b}",
                "class0": a,
                "class1": b,
                "dim": int(X_rep.shape[1]),
                **metrics,
            })

    metrics_df = pd.DataFrame(metric_rows)

    ok_df = metrics_df[metrics_df["cv_status"].eq("ok")].copy()
    rep_summary_rows: List[Dict[str, Any]] = []
    for rep_name, group in metrics_df.groupby("representation", sort=False):
        ok = group[group["cv_status"].eq("ok")]
        dim = int(group["dim"].iloc[0])
        if len(ok) == 0:
            rep_summary_rows.append({
                "representation": rep_name,
                "dim": dim,
                "n_pairs_ok": 0,
                "mean_macro_f1": float("nan"),
                "mean_auc": float("nan"),
                "min_macro_f1": float("nan"),
                "max_macro_f1": float("nan"),
                "mean_balanced_accuracy": float("nan"),
            })
        else:
            rep_summary_rows.append({
                "representation": rep_name,
                "dim": dim,
                "n_pairs_ok": int(len(ok)),
                "mean_macro_f1": float(ok["macro_f1"].mean()),
                "mean_auc": float(ok["auc"].mean()),
                "min_macro_f1": float(ok["macro_f1"].min()),
                "max_macro_f1": float(ok["macro_f1"].max()),
                "mean_balanced_accuracy": float(ok["balanced_accuracy"].mean()),
            })

    representation_summary_df = pd.DataFrame(rep_summary_rows)
    representation_summary_df = representation_summary_df.sort_values(
        by=["mean_macro_f1", "mean_auc"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)

    best_by_pair_rows: List[Dict[str, Any]] = []
    for pair, group in ok_df.groupby("pair"):
        best = group.sort_values(by=["macro_f1", "auc"], ascending=[False, False]).iloc[0]
        best_by_pair_rows.append({
            "pair": pair,
            "best_representation": best["representation"],
            "dim": int(best["dim"]),
            "best_macro_f1": float(best["macro_f1"]),
            "best_auc": float(best["auc"]),
            "best_balanced_accuracy": float(best["balanced_accuracy"]),
            "n_samples": int(best["n_samples"]),
        })
    best_by_pair_df = pd.DataFrame(best_by_pair_rows).sort_values("pair").reset_index(drop=True)

    official_pair_df = official_pair_rows_from_pred(pred_csv, pairs)

    # Gate.
    best_rep = representation_summary_df.iloc[0].to_dict()
    mean_macro_f1 = float(best_rep["mean_macro_f1"])
    mean_auc = float(best_rep["mean_auc"])
    min_macro_f1 = float(best_rep["min_macro_f1"])
    best_rep_name = str(best_rep["representation"])

    if mean_macro_f1 >= 0.80 and mean_auc >= 0.88 and min_macro_f1 >= 0.75:
        result = "PASS — input spaces contain strong pairwise signal"
        reason = (
            f"Best representation `{best_rep_name}` has mean macro-F1={mean_macro_f1:.4f}, "
            f"min macro-F1={min_macro_f1:.4f}, mean AUC={mean_auc:.4f}. "
            "This means pairwise subtype information exists before CLS/model decision."
        )
    elif mean_macro_f1 >= 0.72 and mean_auc >= 0.80:
        result = "MIXED — input spaces contain moderate pairwise signal"
        reason = (
            f"Best representation `{best_rep_name}` has mean macro-F1={mean_macro_f1:.4f}, "
            f"mean AUC={mean_auc:.4f}. Signal exists, but it may not be strong enough alone."
        )
    else:
        result = "FAIL — input-space pairwise signal appears weak"
        reason = (
            f"Best representation `{best_rep_name}` has mean macro-F1={mean_macro_f1:.4f}, "
            f"mean AUC={mean_auc:.4f}. This suggests the subtype overlap is already severe before CLS."
        )

    gate = {
        "result": result,
        "reason": reason,
        "best_representation": best_rep_name,
        "best_mean_macro_f1": mean_macro_f1,
        "best_min_macro_f1": min_macro_f1,
        "best_mean_auc": mean_auc,
        "thresholds": {
            "pass_mean_macro_f1": 0.80,
            "pass_min_macro_f1": 0.75,
            "pass_mean_auc": 0.88,
            "mixed_mean_macro_f1": 0.72,
            "mixed_mean_auc": 0.80,
        },
        "note": "Gate is diagnostic only, not an official validation metric.",
    }

    main_metrics = {
        "n_total": int(len(y_val)),
        "n_features": int(X_val_bin.shape[1]),
        "num_bins": int(num_bins),
        "label_names": label_names,
        "pairs": [f"{a}<->{b}" for a, b in pairs],
        "representations_tested": list(reps.keys()),
        "raw_scaled_source": raw_info.get("source"),
        "cv_folds": int(args.cv_folds),
        "random_state": int(args.random_state),
    }

    # Write outputs.
    metrics_path = out_dir / "B2_pairwise_signal_metrics.csv"
    rep_summary_path = out_dir / "B2_representation_summary.csv"
    best_by_pair_path = out_dir / "B2_best_input_representation_by_pair.csv"
    official_pair_path = out_dir / "B2_official_pair_behavior.csv"
    gate_path = out_dir / "B2_gate_decision.json"
    main_metrics_path = out_dir / "B2_metrics.json"
    raw_info_path = out_dir / "B2_raw_scaled_info.json"
    summary_path = out_dir / "B2_summary.md"

    metrics_df.to_csv(metrics_path, index=False)
    representation_summary_df.to_csv(rep_summary_path, index=False)
    best_by_pair_df.to_csv(best_by_pair_path, index=False)
    official_pair_df.to_csv(official_pair_path, index=False)
    gate_path.write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    raw_info_path.write_text(json.dumps(raw_info, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics_json = {
        "main_metrics": main_metrics,
        "gate": gate,
        "inputs": {
            "dataset_npz": str(dataset_npz),
            "metadata_json": str(metadata_json),
            "train_raw": str(train_raw),
            "val_raw": str(val_raw),
            "pred_csv": str(pred_csv),
        },
    }
    main_metrics_path.write_text(json.dumps(metrics_json, ensure_ascii=False, indent=2), encoding="utf-8")

    out_files = [
        summary_path,
        main_metrics_path,
        metrics_path,
        rep_summary_path,
        best_by_pair_path,
        official_pair_path,
        gate_path,
        raw_info_path,
    ]

    summary_md = make_markdown_summary(
        main_metrics=main_metrics,
        representation_summary_df=representation_summary_df,
        best_by_pair_df=best_by_pair_df,
        metrics_df=metrics_df,
        official_pair_df=official_pair_df,
        gate=gate,
        out_files=out_files,
    )
    summary_path.write_text(summary_md, encoding="utf-8")

    out_zip = zip_outputs(out_dir)

    print("===== B2 input pairwise signal audit done =====")
    print("summary:", summary_path)
    print("gate:", gate_path)
    print("zip:", out_zip)
    print("result:", gate["result"])
    print("best_representation:", gate["best_representation"])
    print("best_mean_macro_f1:", gate["best_mean_macro_f1"])
    print("best_mean_auc:", gate["best_mean_auc"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
