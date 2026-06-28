#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E0 pair-specific input separation audit for C2+D3.

Goal
----
Do not train. Do not modify official files.
Audit whether hard malware subtype pairs are separable at input/feature level:
  - raw_scaled continuous
  - C2 bin/token
  - C2 offset
  - D3 scalar proxy = bin_norm + offset/(K-1)
  - small pair-specific interactions among top features

This is intended after D1a/D1b/D1c showed that adding heads on the same
representation mostly shifts the boundary. E0 asks a different question:
  "Is there an input/feature transformation that actually separates overlap?"

Default expected paths from repo root:
  01_split/train_raw.csv
  01_split/val_raw.csv
  03_outputs/05_dataset/dataset.npz
  03_outputs/05_dataset/metadata.json
  03_outputs/06_model/val_predictions_best.csv
  05_test/outputs/D1b_official_fork_lam0p01/val_predictions_best.csv
  05_test/outputs/D1c_audit_D1b_lam0p01/*.csv
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import roc_auc_score
    from sklearn.feature_selection import mutual_info_classif
except Exception as e:
    raise RuntimeError("E0 requires scikit-learn. Install with: pip install scikit-learn") from e


HARD_PAIRS = [
    ("Ransomware", "Spyware"),
    ("Ransomware", "Trojan"),
    ("Spyware", "Trojan"),
]

IMPORTANT_DIRECTIONS = [
    "Ransomware->Spyware",
    "Spyware->Ransomware",
    "Ransomware->Trojan",
    "Trojan->Ransomware",
    "Spyware->Trojan",
    "Trojan->Spyware",
]


def repo_root_from_here() -> Path:
    p = Path(__file__).resolve()
    if p.parent.name == "05_test":
        return p.parents[1]
    return Path.cwd().resolve()


def resolve_path(path_like: str | Path, repo_root: Path) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    return (repo_root / p).resolve()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def import_official_train(train_script: Path):
    if not train_script.exists():
        raise FileNotFoundError(f"official 07_train.py not found: {train_script}")

    src_dir = train_script.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    spec = importlib.util.spec_from_file_location("official_07_train_for_e0", str(train_script))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import official train script: {train_script}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def normalize_label_names(meta: dict) -> Tuple[List[str], Dict[str, int]]:
    label_mapping = meta.get("label_mapping")
    if not isinstance(label_mapping, dict):
        raise ValueError("metadata.json missing label_mapping dict")
    label_names = [str(label) for label, _ in sorted(label_mapping.items(), key=lambda kv: kv[1])]
    label_to_id = {name.strip(): int(i) for i, name in enumerate(label_names)}
    return label_names, label_to_id


def load_official_inputs(args, repo_root: Path):
    """
    Reuse official 07_train.py for:
      - load_dataset()
      - RUN_SPECS["D3"]
      - load_continuous_for_run()
    This avoids reimplementing raw_scaled continuous logic.
    """
    train_mod = import_official_train(resolve_path(args.official_train, repo_root))

    dataset_npz = resolve_path(args.dataset_npz, repo_root)
    metadata_json = resolve_path(args.metadata_json, repo_root)

    data, meta = train_mod.load_dataset(dataset_npz, metadata_json)
    label_names, label_to_id = normalize_label_names(meta)
    feature_names = [str(x) for x in meta["feature_names"]]

    X_train_bin = data["X_train_bin"].astype(np.int64)
    X_train_offset = data["X_train_offset"].astype(np.float32)
    y_train = data["y_train"].astype(np.int64)

    X_val_bin = data["X_val_bin"].astype(np.int64)
    X_val_offset = data["X_val_offset"].astype(np.float32)
    y_val = data["y_val"].astype(np.int64)

    spec = train_mod.RUN_SPECS["D3"]
    raw_args = SimpleNamespace(
        train_raw=str(resolve_path(args.train_raw, repo_root)),
        val_raw=str(resolve_path(args.val_raw, repo_root)),
    )
    X_train_raw, X_val_raw, continuous_info = train_mod.load_continuous_for_run(
        spec=spec,
        meta=meta,
        args=raw_args,
        train_shape=X_train_bin.shape,
        val_shape=X_val_bin.shape,
    )

    # Infer K/num_bins from metadata/config or max bin.
    num_bins = int(meta.get("num_bins", 0) or meta.get("K", 0) or (max(int(X_train_bin.max()), int(X_val_bin.max())) + 1))
    denom = max(1, num_bins - 1)

    X_train_bin_norm = X_train_bin.astype(np.float32) / float(denom)
    X_val_bin_norm = X_val_bin.astype(np.float32) / float(denom)

    X_train_d3_scalar = X_train_bin_norm + (X_train_offset.astype(np.float32) / float(denom))
    X_val_d3_scalar = X_val_bin_norm + (X_val_offset.astype(np.float32) / float(denom))

    reps_train = {
        "raw_scaled": X_train_raw.astype(np.float32),
        "bin_norm": X_train_bin_norm.astype(np.float32),
        "offset": X_train_offset.astype(np.float32),
        "d3_scalar": X_train_d3_scalar.astype(np.float32),
    }
    reps_val = {
        "raw_scaled": X_val_raw.astype(np.float32),
        "bin_norm": X_val_bin_norm.astype(np.float32),
        "offset": X_val_offset.astype(np.float32),
        "d3_scalar": X_val_d3_scalar.astype(np.float32),
    }

    return {
        "train_mod": train_mod,
        "meta": meta,
        "label_names": label_names,
        "label_to_id": label_to_id,
        "feature_names": feature_names,
        "num_bins": num_bins,
        "continuous_info": continuous_info,
        "X_train_bin": X_train_bin,
        "X_val_bin": X_val_bin,
        "X_train_offset": X_train_offset,
        "X_val_offset": X_val_offset,
        "y_train": y_train,
        "y_val": y_val,
        "reps_train": reps_train,
        "reps_val": reps_val,
    }


def normalize_pred_df(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = df.copy()
    if "sample_index" not in df.columns:
        df["sample_index"] = np.arange(len(df), dtype=int)

    needed = ["sample_index", "true_id", "true_label", "pred_id", "pred_label", "correct"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{prefix} pred missing {missing}; columns={list(df.columns)}")

    out = df[needed].copy()
    out = out.rename(columns={
        "true_id": f"{prefix}_true_id",
        "true_label": f"{prefix}_true_label",
        "pred_id": f"{prefix}_pred_id",
        "pred_label": f"{prefix}_pred_label",
        "correct": f"{prefix}_correct",
    })
    out["sample_index"] = out["sample_index"].astype(int)
    out[f"{prefix}_correct"] = out[f"{prefix}_correct"].astype(bool)
    return out


def load_transition_context(args, repo_root: Path, y_val: np.ndarray) -> pd.DataFrame:
    base_path = resolve_path(args.baseline_pred, repo_root)
    if not base_path.exists():
        raise FileNotFoundError(f"baseline prediction not found: {base_path}")

    base = normalize_pred_df(pd.read_csv(base_path), "base")

    # D1b prediction is optional. If absent, still produce baseline-direction audit.
    d1b_path = resolve_path(args.d1b_pred, repo_root) if args.d1b_pred else None
    if d1b_path is not None and d1b_path.exists():
        d1b = normalize_pred_df(pd.read_csv(d1b_path), "d1b")
        df = base.merge(d1b, on="sample_index", how="inner")
        if len(df) != len(base) or len(df) != len(d1b):
            raise ValueError(f"Prediction alignment mismatch: base={len(base)} d1b={len(d1b)} merged={len(df)}")
        if not (df["base_true_id"].to_numpy() == df["d1b_true_id"].to_numpy()).all():
            raise ValueError("baseline and D1b true_id mismatch")
        df["d1b_direction"] = df["base_true_label"].astype(str) + "->" + df["d1b_pred_label"].astype(str)
        df["transition"] = "both_wrong"
        df.loc[df["base_correct"] & df["d1b_correct"], "transition"] = "both_correct"
        df.loc[(~df["base_correct"]) & df["d1b_correct"], "transition"] = "fixed"
        df.loc[df["base_correct"] & (~df["d1b_correct"]), "transition"] = "damaged"
        df["pred_changed"] = df["base_pred_id"].astype(int) != df["d1b_pred_id"].astype(int)
    else:
        df = base.copy()
        df["d1b_pred_id"] = np.nan
        df["d1b_pred_label"] = ""
        df["d1b_correct"] = False
        df["d1b_direction"] = ""
        df["transition"] = np.where(df["base_correct"], "base_correct", "base_wrong")
        df["pred_changed"] = False

    df["true_id"] = df["base_true_id"].astype(int)
    df["true_label"] = df["base_true_label"].astype(str)
    df["base_direction"] = df["true_label"].astype(str) + "->" + df["base_pred_label"].astype(str)

    # Sanity with y_val order.
    if len(df) != len(y_val):
        raise ValueError(f"prediction rows {len(df)} != y_val rows {len(y_val)}")
    if not (df.sort_values("sample_index")["true_id"].to_numpy() == y_val).all():
        print("[WARN] y_val differs from prediction true_id after sorting by sample_index. Continuing with prediction labels.")
    return df.sort_values("sample_index").reset_index(drop=True)


def safe_auc_binary(x: np.ndarray, y_binary: np.ndarray) -> Tuple[float, int]:
    """
    Return orientation-invariant AUC and direction sign.
      auc_best >= 0.5
      direction = +1 means larger x predicts class 1, -1 means smaller x predicts class 1.
    """
    if len(np.unique(y_binary)) < 2:
        return float("nan"), 0
    try:
        auc = float(roc_auc_score(y_binary, x))
    except Exception:
        return float("nan"), 0
    if auc >= 0.5:
        return auc, +1
    return 1.0 - auc, -1


def cohens_d(x0: np.ndarray, x1: np.ndarray) -> float:
    if len(x0) < 2 or len(x1) < 2:
        return float("nan")
    m0, m1 = float(np.mean(x0)), float(np.mean(x1))
    v0, v1 = float(np.var(x0, ddof=1)), float(np.var(x1, ddof=1))
    pooled = math.sqrt(max(1e-12, ((len(x0)-1)*v0 + (len(x1)-1)*v1) / max(1, len(x0)+len(x1)-2)))
    return (m1 - m0) / pooled


def ks_statistic(x0: np.ndarray, x1: np.ndarray) -> float:
    if len(x0) == 0 or len(x1) == 0:
        return float("nan")
    xs = np.sort(np.unique(np.concatenate([x0, x1])))
    if len(xs) == 0:
        return float("nan")
    cdf0 = np.searchsorted(np.sort(x0), xs, side="right") / len(x0)
    cdf1 = np.searchsorted(np.sort(x1), xs, side="right") / len(x1)
    return float(np.max(np.abs(cdf0 - cdf1)))


def overlap_iqr_ratio(x0: np.ndarray, x1: np.ndarray) -> float:
    """
    1 = heavy IQR overlap, 0 = no IQR overlap.
    """
    if len(x0) == 0 or len(x1) == 0:
        return float("nan")
    a0, b0 = np.quantile(x0, [0.25, 0.75])
    a1, b1 = np.quantile(x1, [0.25, 0.75])
    inter = max(0.0, min(b0, b1) - max(a0, a1))
    union = max(b0, b1) - min(a0, a1)
    if union <= 1e-12:
        return 1.0
    return float(inter / union)


def median_gap_norm(x0: np.ndarray, x1: np.ndarray) -> float:
    if len(x0) == 0 or len(x1) == 0:
        return float("nan")
    med_gap = abs(float(np.median(x1)) - float(np.median(x0)))
    q1, q3 = np.quantile(np.concatenate([x0, x1]), [0.25, 0.75])
    scale = max(1e-12, float(q3 - q1))
    return float(med_gap / scale)


def pair_feature_metrics_for_rep(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    label_to_id: Dict[str, int],
    pair: Tuple[str, str],
    rep_name: str,
) -> pd.DataFrame:
    a, b = pair
    ida, idb = label_to_id[a], label_to_id[b]
    mask = (y == ida) | (y == idb)
    Xp = X[mask]
    yp_raw = y[mask]
    yb = (yp_raw == idb).astype(int)

    rows = []
    for j, fname in enumerate(feature_names):
        x = Xp[:, j].astype(float)
        x0 = x[yb == 0]
        x1 = x[yb == 1]
        auc, direction = safe_auc_binary(x, yb)
        d = cohens_d(x0, x1)
        ks = ks_statistic(x0, x1)
        iqr_overlap = overlap_iqr_ratio(x0, x1)
        med_gap = median_gap_norm(x0, x1)
        rows.append({
            "pair": f"{a}<->{b}",
            "label_a": a,
            "label_b": b,
            "rep": rep_name,
            "feature_index": int(j),
            "feature": fname,
            "n_pair": int(len(yb)),
            "n_a": int((yb == 0).sum()),
            "n_b": int((yb == 1).sum()),
            "auc_best": float(auc),
            "auc_direction_larger_means_label_b": int(direction),
            "abs_auc_minus_0p5": float(abs(auc - 0.5)) if not np.isnan(auc) else float("nan"),
            "cohens_d_b_minus_a": float(d),
            "abs_cohens_d": float(abs(d)) if not np.isnan(d) else float("nan"),
            "ks_stat": float(ks),
            "iqr_overlap_ratio": float(iqr_overlap),
            "median_gap_norm_iqr": float(med_gap),
            "mean_a": float(np.mean(x0)) if len(x0) else float("nan"),
            "mean_b": float(np.mean(x1)) if len(x1) else float("nan"),
            "median_a": float(np.median(x0)) if len(x0) else float("nan"),
            "median_b": float(np.median(x1)) if len(x1) else float("nan"),
        })
    return pd.DataFrame(rows)


def compute_mutual_info_for_top(
    X: np.ndarray,
    y: np.ndarray,
    feature_rank_df: pd.DataFrame,
    label_to_id: Dict[str, int],
    rep_name: str,
    topn: int,
    random_state: int,
) -> pd.DataFrame:
    rows = []
    for pair_str, g in feature_rank_df[feature_rank_df["rep"] == rep_name].groupby("pair"):
        a, b = pair_str.split("<->")
        ida, idb = label_to_id[a], label_to_id[b]
        mask = (y == ida) | (y == idb)
        yb = (y[mask] == idb).astype(int)
        top_features = (
            g.sort_values(["auc_best", "ks_stat", "abs_cohens_d"], ascending=[False, False, False])
             .head(topn)["feature_index"].astype(int).tolist()
        )
        if not top_features:
            continue
        Xsub = X[mask][:, top_features]
        try:
            mi = mutual_info_classif(Xsub, yb, random_state=random_state, discrete_features=False)
        except Exception:
            mi = np.full(len(top_features), np.nan)
        for feat_idx, v in zip(top_features, mi):
            rows.append({
                "pair": pair_str,
                "rep": rep_name,
                "feature_index": int(feat_idx),
                "mutual_info": float(v),
            })
    return pd.DataFrame(rows)


def create_feature_rank(reps_val, y_val, feature_names, label_to_id, random_state: int) -> pd.DataFrame:
    all_rows = []
    for rep_name, X in reps_val.items():
        for pair in HARD_PAIRS:
            all_rows.append(pair_feature_metrics_for_rep(X, y_val, feature_names, label_to_id, pair, rep_name))
    df = pd.concat(all_rows, ignore_index=True)

    # Add MI for selected top raw_scaled/d3_scalar/bin/offset features.
    mi_parts = []
    for rep_name, X in reps_val.items():
        mi_parts.append(compute_mutual_info_for_top(
            X=X, y=y_val, feature_rank_df=df, label_to_id=label_to_id,
            rep_name=rep_name, topn=30, random_state=random_state
        ))
    mi_df = pd.concat(mi_parts, ignore_index=True) if mi_parts else pd.DataFrame()
    if not mi_df.empty:
        df = df.merge(mi_df, on=["pair", "rep", "feature_index"], how="left")
    else:
        df["mutual_info"] = np.nan

    # Composite separability score.
    df["separation_score"] = (
        df["abs_auc_minus_0p5"].fillna(0.0) * 2.0
        + df["ks_stat"].fillna(0.0)
        + np.minimum(df["abs_cohens_d"].fillna(0.0), 3.0) / 3.0
        + df["median_gap_norm_iqr"].fillna(0.0).clip(0, 3) / 3.0
        + df["mutual_info"].fillna(0.0).clip(0, 1)
        - df["iqr_overlap_ratio"].fillna(1.0) * 0.25
    )
    return df.sort_values(["pair", "rep", "separation_score"], ascending=[True, True, False]).reset_index(drop=True)


def create_overlap_stats(feature_rank: pd.DataFrame, topn: int) -> pd.DataFrame:
    rows = []
    for (pair, rep), g in feature_rank.groupby(["pair", "rep"]):
        top = g.sort_values("separation_score", ascending=False).head(topn)
        rows.append({
            "pair": pair,
            "rep": rep,
            "topn": int(len(top)),
            "mean_auc_best": float(top["auc_best"].mean()),
            "max_auc_best": float(top["auc_best"].max()),
            "mean_ks": float(top["ks_stat"].mean()),
            "max_ks": float(top["ks_stat"].max()),
            "mean_abs_cohens_d": float(top["abs_cohens_d"].mean()),
            "mean_iqr_overlap_ratio": float(top["iqr_overlap_ratio"].mean()),
            "min_iqr_overlap_ratio": float(top["iqr_overlap_ratio"].min()),
            "mean_separation_score": float(top["separation_score"].mean()),
            "top_features": ", ".join(top["feature"].astype(str).head(10).tolist()),
        })
    return pd.DataFrame(rows).sort_values(["pair", "mean_separation_score"], ascending=[True, False])


def class_centroids_by_pair(X: np.ndarray, y: np.ndarray, label_to_id: Dict[str, int], pair: Tuple[str, str]):
    a, b = pair
    ida, idb = label_to_id[a], label_to_id[b]
    ca = np.median(X[y == ida], axis=0)
    cb = np.median(X[y == idb], axis=0)
    return ca, cb


def create_direction_shift(
    X: np.ndarray,
    y: np.ndarray,
    pred_ctx: pd.DataFrame,
    feature_names: List[str],
    label_to_id: Dict[str, int],
    feature_rank: pd.DataFrame,
    rep_name: str,
    topn_features: int,
) -> pd.DataFrame:
    """
    For baseline error directions, compare wrong samples to true/pred class medians.
    Also reports top feature shifts in the selected rep.
    """
    rows = []
    for pair in HARD_PAIRS:
        a, b = pair
        pair_str = f"{a}<->{b}"
        ca, cb = class_centroids_by_pair(X, y, label_to_id, pair)
        # top features for this pair/rep.
        top_feats = (
            feature_rank[(feature_rank["pair"] == pair_str) & (feature_rank["rep"] == rep_name)]
            .sort_values("separation_score", ascending=False)
            .head(topn_features)["feature_index"].astype(int).tolist()
        )
        if not top_feats:
            continue

        for direction in [f"{a}->{b}", f"{b}->{a}"]:
            true_label, pred_label = direction.split("->")
            true_cent = ca if true_label == a else cb
            pred_cent = cb if pred_label == b else ca

            mask = (pred_ctx["base_direction"].to_numpy().astype(str) == direction)
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue
            Xw = X[idx]

            dist_true = np.linalg.norm(Xw[:, top_feats] - true_cent[top_feats], axis=1)
            dist_pred = np.linalg.norm(Xw[:, top_feats] - pred_cent[top_feats], axis=1)

            feature_shift = []
            for j in top_feats[:10]:
                vals = Xw[:, j]
                feature_shift.append({
                    "feature_index": int(j),
                    "feature": feature_names[int(j)],
                    "wrong_mean": float(np.mean(vals)),
                    "true_class_median": float(true_cent[j]),
                    "pred_class_median": float(pred_cent[j]),
                    "closer_to_pred_by_abs": bool(abs(np.mean(vals) - pred_cent[j]) < abs(np.mean(vals) - true_cent[j])),
                })

            rows.append({
                "rep": rep_name,
                "pair": pair_str,
                "direction": direction,
                "n_wrong": int(len(idx)),
                "topn_features_used": int(len(top_feats)),
                "mean_dist_to_true_centroid": float(np.mean(dist_true)),
                "mean_dist_to_pred_centroid": float(np.mean(dist_pred)),
                "median_dist_to_true_centroid": float(np.median(dist_true)),
                "median_dist_to_pred_centroid": float(np.median(dist_pred)),
                "closer_to_true_rate": float(np.mean(dist_true < dist_pred)),
                "feature_shift_json": json.dumps(feature_shift, ensure_ascii=False),
            })
    return pd.DataFrame(rows)


def get_top_feature_indices(feature_rank: pd.DataFrame, pair_str: str, rep_name: str, topn: int) -> List[int]:
    return (
        feature_rank[(feature_rank["pair"] == pair_str) & (feature_rank["rep"] == rep_name)]
        .sort_values("separation_score", ascending=False)
        .head(topn)["feature_index"].astype(int).tolist()
    )


def candidate_interactions(X: np.ndarray, y: np.ndarray, feature_rank: pd.DataFrame, feature_names: List[str], label_to_id: Dict[str, int], rep_name: str, topn_base: int, max_pairs: int) -> pd.DataFrame:
    """
    Build simple interactions among top pair features only:
      diff = xi - xj
      absdiff = |xi-xj|
      prod = xi*xj
      ratio = xi/(xj+eps)
    Evaluate AUC/KS/effect.
    """
    rows = []
    eps = 1e-6
    for pair in HARD_PAIRS:
        a, b = pair
        pair_str = f"{a}<->{b}"
        ida, idb = label_to_id[a], label_to_id[b]
        mask = (y == ida) | (y == idb)
        Xp = X[mask]
        yb = (y[mask] == idb).astype(int)

        top_idx = get_top_feature_indices(feature_rank, pair_str, rep_name, topn_base)
        pairs = []
        for ix, i in enumerate(top_idx):
            for j in top_idx[ix+1:]:
                pairs.append((i, j))
        pairs = pairs[:max_pairs]

        for i, j in pairs:
            xi, xj = Xp[:, i].astype(float), Xp[:, j].astype(float)
            candidates = {
                "diff_i_minus_j": xi - xj,
                "absdiff": np.abs(xi - xj),
                "prod": xi * xj,
                "ratio_i_over_j": xi / (xj + eps),
            }
            for kind, z in candidates.items():
                if np.allclose(z, z[0]):
                    continue
                z0, z1 = z[yb == 0], z[yb == 1]
                auc, direction = safe_auc_binary(z, yb)
                d = cohens_d(z0, z1)
                ks = ks_statistic(z0, z1)
                rows.append({
                    "pair": pair_str,
                    "rep": rep_name,
                    "interaction": kind,
                    "feature_i_index": int(i),
                    "feature_i": feature_names[int(i)],
                    "feature_j_index": int(j),
                    "feature_j": feature_names[int(j)],
                    "auc_best": float(auc),
                    "auc_direction_larger_means_label_b": int(direction),
                    "ks_stat": float(ks),
                    "abs_cohens_d": float(abs(d)) if not np.isnan(d) else float("nan"),
                    "separation_score": float((abs(auc - 0.5) * 2 if not np.isnan(auc) else 0.0) + (ks if not np.isnan(ks) else 0.0) + min(abs(d) if not np.isnan(d) else 0.0, 3.0)/3.0),
                })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["pair", "separation_score"], ascending=[True, False]).reset_index(drop=True)


def create_d1c_attention_vs_signal(feature_rank: pd.DataFrame, args, repo_root: Path) -> pd.DataFrame:
    d1c_path = resolve_path(args.d1c_top_features_by_transition, repo_root) if args.d1c_top_features_by_transition else None
    if d1c_path is None or not d1c_path.exists():
        return pd.DataFrame([{"note": "D1c top_features_by_transition not found; skipped."}])

    d1c = pd.read_csv(d1c_path)
    if not {"pair_key", "transition", "feature", "mean_attention"}.issubset(d1c.columns):
        return pd.DataFrame([{"note": f"D1c file missing expected columns: {d1c_path}"}])

    key_to_pair = {
        "Ransomware__vs__Spyware": "Ransomware<->Spyware",
        "Ransomware__vs__Trojan": "Ransomware<->Trojan",
        "Spyware__vs__Trojan": "Spyware<->Trojan",
    }
    d1c["pair"] = d1c["pair_key"].map(key_to_pair).fillna(d1c["pair_key"])

    # Use raw_scaled and d3_scalar as most meaningful signal references.
    sig = feature_rank[feature_rank["rep"].isin(["raw_scaled", "d3_scalar"])].copy()
    sig = sig.sort_values("separation_score", ascending=False)
    sig = sig.drop_duplicates(["pair", "feature", "rep"], keep="first")

    wide = sig.pivot_table(
        index=["pair", "feature"],
        columns="rep",
        values=["auc_best", "ks_stat", "separation_score", "iqr_overlap_ratio"],
        aggfunc="max"
    )
    wide.columns = ["_".join([str(a), str(b)]) for a, b in wide.columns]
    wide = wide.reset_index()

    out = d1c.merge(wide, on=["pair", "feature"], how="left")
    # Rank within pair/transition by attention.
    out["attention_rank"] = out.groupby(["pair", "transition"])["mean_attention"].rank(ascending=False, method="first")
    # A simple mismatch flag: high attention but weak raw/d3 separability.
    sep_cols = [c for c in out.columns if c.startswith("separation_score_")]
    if sep_cols:
        out["max_input_separation_score"] = out[sep_cols].max(axis=1)
        out["attention_high_but_signal_weak"] = (out["attention_rank"] <= 10) & (out["max_input_separation_score"].fillna(0) < 0.75)
    return out.sort_values(["pair", "transition", "attention_rank"])


def write_readme(out_dir: Path, summary: dict) -> None:
    text = f"""# E0 Pair Input Separation Audit Summary

## Purpose

E0 audits whether hard malware subtype pairs can be separated at input/feature level,
instead of adding another head on the same representation.

## Inputs

- dataset: `{summary['inputs']['dataset_npz']}`
- metadata: `{summary['inputs']['metadata_json']}`
- train_raw: `{summary['inputs']['train_raw']}`
- val_raw: `{summary['inputs']['val_raw']}`
- baseline_pred: `{summary['inputs']['baseline_pred']}`
- d1b_pred: `{summary['inputs'].get('d1b_pred')}`

## Key output files

- `E0_summary.json`
- `E0_pair_feature_rank.csv`
- `E0_pair_feature_overlap_stats.csv`
- `E0_pair_direction_feature_shift.csv`
- `E0_pair_interaction_candidates.csv`
- `E0_d1c_attention_vs_feature_signal.csv`

## How to read

1. Start with `E0_pair_feature_overlap_stats.csv`.
   - Look for pair/rep rows with high `max_auc_best`, high `mean_ks`, low `mean_iqr_overlap_ratio`.
2. Then open `E0_pair_feature_rank.csv`.
   - For each pair, inspect top features in `raw_scaled` and `d3_scalar`.
3. Then open `E0_pair_interaction_candidates.csv`.
   - If interactions have much higher score than single features, E1 should build pair-specific transformed inputs.
4. Then open `E0_d1c_attention_vs_feature_signal.csv`.
   - If D1c attention focuses on weak-separation features, D1b attention is not reliable for expert decisions.

## Decision logic

- If clear pair-specific features/interactions exist:
  proceed to E1/E2 expert with NEW pair-specific input.
- If not:
  overlap is likely too strong under current features; more heads on same representation are unlikely to help.
"""
    (out_dir / "E0_readme_summary.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="E0 pair-specific input separation audit")
    parser.add_argument("--dataset-npz", default="03_outputs/05_dataset/dataset.npz")
    parser.add_argument("--metadata-json", default="03_outputs/05_dataset/metadata.json")
    parser.add_argument("--train-raw", default="01_split/train_raw.csv")
    parser.add_argument("--val-raw", default="01_split/val_raw.csv")
    parser.add_argument("--baseline-pred", default="03_outputs/06_model/val_predictions_best.csv")
    parser.add_argument("--d1b-pred", default="05_test/outputs/D1b_official_fork_lam0p01/val_predictions_best.csv")
    parser.add_argument("--d1c-top-features-by-transition", default="05_test/outputs/D1c_audit_D1b_lam0p01/d1c_top_features_by_transition.csv")
    parser.add_argument("--official-train", default="02_src/07_train.py")
    parser.add_argument("--out-dir", default="05_test/outputs/E0_pair_input_separation_audit")
    parser.add_argument("--topn-overlap", type=int, default=20)
    parser.add_argument("--topn-direction", type=int, default=20)
    parser.add_argument("--interaction-rep", default="raw_scaled", choices=["raw_scaled", "bin_norm", "offset", "d3_scalar"])
    parser.add_argument("--interaction-topn-base", type=int, default=12)
    parser.add_argument("--interaction-max-pairs", type=int, default=66)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    repo_root = repo_root_from_here()
    out_dir = resolve_path(args.out_dir, repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    inp = load_official_inputs(args, repo_root)
    y_val = inp["y_val"]
    feature_names = inp["feature_names"]
    label_to_id = inp["label_to_id"]

    pred_ctx = load_transition_context(args, repo_root, y_val)

    print(f"[E0] repo_root={repo_root}")
    print(f"[E0] n_val={len(y_val)} n_features={len(feature_names)} num_bins={inp['num_bins']}")
    print("[E0] computing feature rank...")

    feature_rank = create_feature_rank(
        reps_val=inp["reps_val"],
        y_val=y_val,
        feature_names=feature_names,
        label_to_id=label_to_id,
        random_state=int(args.random_state),
    )
    feature_rank.to_csv(out_dir / "E0_pair_feature_rank.csv", index=False)

    overlap_stats = create_overlap_stats(feature_rank, topn=int(args.topn_overlap))
    overlap_stats.to_csv(out_dir / "E0_pair_feature_overlap_stats.csv", index=False)

    print("[E0] computing direction shift...")
    direction_parts = []
    for rep_name in ["raw_scaled", "d3_scalar", "bin_norm", "offset"]:
        direction_parts.append(create_direction_shift(
            X=inp["reps_val"][rep_name],
            y=y_val,
            pred_ctx=pred_ctx,
            feature_names=feature_names,
            label_to_id=label_to_id,
            feature_rank=feature_rank,
            rep_name=rep_name,
            topn_features=int(args.topn_direction),
        ))
    direction_df = pd.concat(direction_parts, ignore_index=True) if direction_parts else pd.DataFrame()
    direction_df.to_csv(out_dir / "E0_pair_direction_feature_shift.csv", index=False)

    print("[E0] computing interaction candidates...")
    inter_df = candidate_interactions(
        X=inp["reps_val"][args.interaction_rep],
        y=y_val,
        feature_rank=feature_rank,
        feature_names=feature_names,
        label_to_id=label_to_id,
        rep_name=args.interaction_rep,
        topn_base=int(args.interaction_topn_base),
        max_pairs=int(args.interaction_max_pairs),
    )
    inter_df.to_csv(out_dir / "E0_pair_interaction_candidates.csv", index=False)

    print("[E0] comparing D1c attention vs feature signal...")
    d1c_vs = create_d1c_attention_vs_signal(feature_rank, args, repo_root)
    d1c_vs.to_csv(out_dir / "E0_d1c_attention_vs_feature_signal.csv", index=False)

    # Transition compact.
    transition_counts = pred_ctx["transition"].value_counts().to_dict()
    pred_ctx.to_csv(out_dir / "E0_prediction_context.csv", index=False)

    # Summary top lines.
    best_by_pair_rep = []
    for (pair, rep), g in overlap_stats.groupby(["pair", "rep"]):
        row = g.iloc[0].to_dict()
        best_by_pair_rep.append(row)

    best_rep_per_pair = (
        overlap_stats.sort_values(["pair", "mean_separation_score"], ascending=[True, False])
        .groupby("pair")
        .head(1)
        .to_dict(orient="records")
    )

    summary = {
        "stage": "E0_pair_input_separation_audit",
        "purpose": "Audit pair-specific input/feature separability without training.",
        "inputs": {
            "dataset_npz": str(resolve_path(args.dataset_npz, repo_root)),
            "metadata_json": str(resolve_path(args.metadata_json, repo_root)),
            "train_raw": str(resolve_path(args.train_raw, repo_root)),
            "val_raw": str(resolve_path(args.val_raw, repo_root)),
            "baseline_pred": str(resolve_path(args.baseline_pred, repo_root)),
            "d1b_pred": str(resolve_path(args.d1b_pred, repo_root)) if args.d1b_pred else None,
            "official_train": str(resolve_path(args.official_train, repo_root)),
            "d1c_top_features_by_transition": str(resolve_path(args.d1c_top_features_by_transition, repo_root)) if args.d1c_top_features_by_transition else None,
        },
        "label_names": inp["label_names"],
        "n_val": int(len(y_val)),
        "n_features": int(len(feature_names)),
        "num_bins": int(inp["num_bins"]),
        "representations_audited": list(inp["reps_val"].keys()),
        "hard_pairs": [f"{a}<->{b}" for a, b in HARD_PAIRS],
        "transition_counts": {str(k): int(v) for k, v in transition_counts.items()},
        "best_rep_per_pair_by_mean_separation_score": best_rep_per_pair,
        "outputs": {
            "feature_rank": str(out_dir / "E0_pair_feature_rank.csv"),
            "overlap_stats": str(out_dir / "E0_pair_feature_overlap_stats.csv"),
            "direction_shift": str(out_dir / "E0_pair_direction_feature_shift.csv"),
            "interaction_candidates": str(out_dir / "E0_pair_interaction_candidates.csv"),
            "d1c_attention_vs_feature_signal": str(out_dir / "E0_d1c_attention_vs_feature_signal.csv"),
            "prediction_context": str(out_dir / "E0_prediction_context.csv"),
            "readme": str(out_dir / "E0_readme_summary.md"),
        },
        "interpretation_hint": {
            "strong_single_feature_signal": "AUC_best high, KS high, IQR overlap low for raw_scaled/d3_scalar.",
            "strong_interaction_signal": "Interaction candidates outperform best single features for the same pair.",
            "weak_signal": "All reps have AUC close to 0.5, low KS, high IQR overlap.",
            "d1c_mismatch": "D1c attention high on features with low input separation suggests attention is not a clean expert signal.",
        },
    }
    save_json(out_dir / "E0_summary.json", summary)
    write_readme(out_dir, summary)

    print("[E0] done. Wrote:")
    for k, v in summary["outputs"].items():
        print(f"  - {k}: {v}")
    print(f"  - summary: {out_dir / 'E0_summary.json'}")


if __name__ == "__main__":
    main()
