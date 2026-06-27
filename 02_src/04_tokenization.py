#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build mixed quantile/uniform bin + offset dataset.

No training.
No model change.
No overwrite baseline token artifacts.

Input:
  - train_preprocessed_K{K}.csv
  - val_preprocessed_K{K}.csv
  - preprocess_policy_K{K}.json
  - quantile_vs_uniform_bin_diag_K{K}_B{B}.json

Output:
  03_outputs/token/K{K}_B{B}/
    dataset.npz
    metadata.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

import config as CFG


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build mixed quantile/uniform bin + offset dataset.")
    p.add_argument("--K", type=int, default=int(CFG.TOKEN_K))
    p.add_argument("--num-bins", type=int, default=int(getattr(CFG, "VALUE_NUM_BINS", 128)))
    p.add_argument("--train-preprocessed", default="")
    p.add_argument("--val-preprocessed", default="")
    p.add_argument("--policy-json", default="")
    p.add_argument("--diag-json", default="")
    p.add_argument("--out-root", default=str(CFG.OUTPUT_ROOT / "04_token"))
    p.add_argument("--label-col", default=str(CFG.DEFAULT_LABEL_COL))

    # Mixed-selection rule.
    p.add_argument("--min-unique-for-quantile", type=int, default=128)
    p.add_argument("--min-dominant-reduction", type=float, default=0.02)
    p.add_argument("--min-entropy-delta", type=float, default=0.05)
    return p.parse_args()


def load_feature_order(policy_path: Path) -> List[str]:
    obj = json.loads(policy_path.read_text(encoding="utf-8"))

    meta_features = obj.get("metadata", {}).get("feature_order")
    if meta_features:
        return [str(x) for x in meta_features]

    policies = obj.get("policies", [])
    if not policies:
        raise ValueError(f"No policies found in: {policy_path}")

    return [str(p["feature"]) for p in policies]


def validate_split(name: str, df: pd.DataFrame, features: Sequence[str], label_col: str) -> None:
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"{name} missing features: {missing[:10]}")
    if label_col not in df.columns:
        raise ValueError(f"{name} missing label column: {label_col}")

    arr = df.loc[:, list(features)].to_numpy(dtype=np.float64)
    if np.isnan(arr).any() or np.isinf(arr).any():
        raise ValueError(f"{name} contains NaN/Inf in features.")

    mn = float(np.min(arr)) if arr.size else 0.0
    mx = float(np.max(arr)) if arr.size else 0.0
    if mn < -1e-6 or mx > 1.0 + 1e-6:
        raise ValueError(f"{name} expected preprocessed features in [0,1], got min={mn}, max={mx}")


def build_label_mapping(train_labels: pd.Series) -> Dict[str, int]:
    labels = sorted([str(x) for x in train_labels.dropna().unique().tolist()])
    return {label: idx for idx, label in enumerate(labels)}


def encode_labels(labels: pd.Series, mapping: Dict[str, int], split_name: str) -> np.ndarray:
    out = []
    unknown = []
    for x in labels.tolist():
        key = str(x)
        if key not in mapping:
            unknown.append(key)
            out.append(-1)
        else:
            out.append(mapping[key])
    if unknown:
        raise ValueError(f"{split_name} contains unseen labels: {sorted(set(unknown))[:10]}")
    return np.asarray(out, dtype=np.int64)


def load_diag(diag_path: Path) -> Dict[str, Dict[str, object]]:
    obj = json.loads(diag_path.read_text(encoding="utf-8"))
    rows = obj.get("features", [])
    return {str(r["feature"]): r for r in rows}


def choose_strategy(
    *,
    feature: str,
    z_train: np.ndarray,
    diag_row: Dict[str, object] | None,
    min_unique_for_quantile: int,
    min_dominant_reduction: float,
    min_entropy_delta: float,
) -> Tuple[str, str]:
    unique_n = int(np.unique(z_train).size)

    if unique_n <= 1:
        return "constant", "constant_or_all_same"

    if diag_row is None:
        return "uniform_offset", "no_diag_row"

    dom_red = float(diag_row.get("dominant_ratio_reduction", 0.0))
    ent_delta = float(diag_row.get("delta_entropy_norm", 0.0))
    u_dom = float(diag_row.get("uniform_dominant_bin_ratio", 1.0))
    q_dom = float(diag_row.get("quantile_dominant_bin_ratio", 1.0))

    if unique_n < int(min_unique_for_quantile):
        return "uniform_offset", f"low_unique_{unique_n}"

    if dom_red >= float(min_dominant_reduction) and ent_delta >= float(min_entropy_delta) and q_dom < u_dom:
        return "quantile_offset", (
            f"quantile_improves_dom_{dom_red:.4f}_entropy_{ent_delta:.4f}_unique_{unique_n}"
        )

    return "uniform_offset", (
        f"quantile_not_enough_dom_{dom_red:.4f}_entropy_{ent_delta:.4f}_unique_{unique_n}"
    )


def make_uniform_edges(num_bins: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, int(num_bins) + 1, dtype=np.float64)


def make_quantile_edges(values: np.ndarray, num_bins: int) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    qs = np.linspace(0.0, 1.0, int(num_bins) + 1)
    try:
        edges = np.quantile(v, qs, method="linear")
    except TypeError:
        edges = np.quantile(v, qs, interpolation="linear")
    edges[0] = float(np.min(v))
    edges[-1] = float(np.max(v))
    return edges.astype(np.float64)


def assign_bin_offset(values: np.ndarray, edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    v = np.asarray(values, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    num_bins = int(len(edges) - 1)

    if num_bins <= 0:
        raise ValueError("edges must have at least 2 values.")

    if np.isclose(edges[0], edges[-1]):
        return (
            np.zeros(v.shape, dtype=np.int64),
            np.full(v.shape, 0.5, dtype=np.float32),
        )

    # Clip only to learned train range for this representation.
    # For preprocessed z, normal range should be [0,1].
    v_clip = np.clip(v, edges[0], edges[-1])

    internal = edges[1:-1]
    bin_id = np.searchsorted(internal, v_clip, side="right")
    bin_id = np.clip(bin_id, 0, num_bins - 1).astype(np.int64)

    left = edges[bin_id]
    right = edges[bin_id + 1]
    width = right - left

    offset = np.zeros(v.shape, dtype=np.float32)
    good = ~np.isclose(width, 0.0)
    offset[good] = ((v_clip[good] - left[good]) / width[good]).astype(np.float32)
    offset[~good] = 0.5
    offset = np.clip(offset, 0.0, 1.0).astype(np.float32)
    return bin_id, offset


def _main_build_current_mixed() -> None:
    args = parse_args()
    K = int(args.K)
    B = int(args.num_bins)

    train_path = Path(args.train_preprocessed) if args.train_preprocessed else CFG.preprocess_train_csv_path(K)
    val_path = Path(args.val_preprocessed) if args.val_preprocessed else CFG.preprocess_val_csv_path(K)
    policy_path = Path(args.policy_json) if args.policy_json else CFG.preprocess_policy_json_path(K)
    diag_path = (
        Path(args.diag_json)
        if args.diag_json
        else CFG.OUTPUT_ROOT / "03_bin_diag" / f"quantile_vs_uniform_bin_diag_K{K}_B{B}.json"
    )

    for p in [train_path, val_path, policy_path, diag_path]:
        if not p.exists():
            raise FileNotFoundError(str(p))

    features = load_feature_order(policy_path)
    diag = load_diag(diag_path)

    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    label_col = str(args.label_col)

    validate_split("train", train, features, label_col)
    validate_split("val", val, features, label_col)

    n_train = len(train)
    n_val = len(val)
    n_feat = len(features)

    X_train_bin = np.zeros((n_train, n_feat), dtype=np.int64)
    X_train_offset = np.zeros((n_train, n_feat), dtype=np.float32)
    X_val_bin = np.zeros((n_val, n_feat), dtype=np.int64)
    X_val_offset = np.zeros((n_val, n_feat), dtype=np.float32)

    feature_meta = []
    strategy_counts: Dict[str, int] = {}

    for j, f in enumerate(features):
        z_train = np.clip(train[f].to_numpy(dtype=np.float64), 0.0, 1.0)
        z_val = np.clip(val[f].to_numpy(dtype=np.float64), 0.0, 1.0)

        strategy, reason = choose_strategy(
            feature=f,
            z_train=z_train,
            diag_row=diag.get(f),
            min_unique_for_quantile=int(args.min_unique_for_quantile),
            min_dominant_reduction=float(args.min_dominant_reduction),
            min_entropy_delta=float(args.min_entropy_delta),
        )

        if strategy == "constant":
            edges = np.asarray([0.0, 0.0], dtype=np.float64)
            tr_bin = np.zeros(n_train, dtype=np.int64)
            tr_off = np.full(n_train, 0.5, dtype=np.float32)
            va_bin = np.zeros(n_val, dtype=np.int64)
            va_off = np.full(n_val, 0.5, dtype=np.float32)
        elif strategy == "quantile_offset":
            edges = make_quantile_edges(z_train, B)
            tr_bin, tr_off = assign_bin_offset(z_train, edges)
            va_bin, va_off = assign_bin_offset(z_val, edges)
        else:
            edges = make_uniform_edges(B)
            tr_bin, tr_off = assign_bin_offset(z_train, edges)
            va_bin, va_off = assign_bin_offset(z_val, edges)

        X_train_bin[:, j] = tr_bin
        X_train_offset[:, j] = tr_off
        X_val_bin[:, j] = va_bin
        X_val_offset[:, j] = va_off

        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        drow = diag.get(f, {})
        feature_meta.append({
            "feature": f,
            "strategy": strategy,
            "reason": reason,
            "z_train_unique": int(np.unique(z_train).size),
            "train_bin_used": int(np.unique(tr_bin).size),
            "val_bin_used": int(np.unique(va_bin).size),
            "train_offset_min": float(np.min(tr_off)),
            "train_offset_max": float(np.max(tr_off)),
            "train_offset_mean": float(np.mean(tr_off)),
            "val_offset_min": float(np.min(va_off)),
            "val_offset_max": float(np.max(va_off)),
            "val_offset_mean": float(np.mean(va_off)),
            "diag_uniform_dominant_bin_ratio": drow.get("uniform_dominant_bin_ratio"),
            "diag_quantile_dominant_bin_ratio": drow.get("quantile_dominant_bin_ratio"),
            "diag_dominant_ratio_reduction": drow.get("dominant_ratio_reduction"),
            "diag_uniform_entropy_norm": drow.get("uniform_entropy_norm"),
            "diag_quantile_entropy_norm": drow.get("quantile_entropy_norm"),
            "diag_delta_entropy_norm": drow.get("delta_entropy_norm"),
            "edge_min": float(edges[0]),
            "edge_max": float(edges[-1]),
            "edge_unique_count": int(np.unique(edges).size),
            "zero_width_edge_count": int(np.sum(np.isclose(np.diff(edges), 0.0))),
            "edges": [float(x) for x in edges.tolist()],
        })

    label_mapping = build_label_mapping(train[label_col])
    y_train = encode_labels(train[label_col], label_mapping, "train")
    y_val = encode_labels(val[label_col], label_mapping, "val")

    out_dir = Path(args.out_root) / f"K{K}_B{B}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_npz = out_dir / "dataset.npz"
    out_meta = out_dir / "metadata.json"

    np.savez_compressed(
        out_npz,
        X_train_bin=X_train_bin,
        X_train_offset=X_train_offset,
        y_train=y_train,
        X_val_bin=X_val_bin,
        X_val_offset=X_val_offset,
        y_val=y_val,
        feature_names=np.asarray(features, dtype=object),
        label_names=np.asarray([k for k, _ in sorted(label_mapping.items(), key=lambda kv: kv[1])], dtype=object),
        K=np.asarray([K], dtype=np.int64),
        num_bins=np.asarray([B], dtype=np.int64),
    )

    metadata = {
        "stage": "build_mixed_quantile_offset",
        "K": K,
        "num_bins": B,
        "input": {
            "train_preprocessed": str(train_path),
            "val_preprocessed": str(val_path),
            "policy_json": str(policy_path),
            "diag_json": str(diag_path),
        },
        "selection_rule": {
            "min_unique_for_quantile": int(args.min_unique_for_quantile),
            "min_dominant_reduction": float(args.min_dominant_reduction),
            "min_entropy_delta": float(args.min_entropy_delta),
        },
        "label_col": label_col,
        "label_mapping": label_mapping,
        "n_features": n_feat,
        "feature_names": features,
        "strategy_counts": strategy_counts,
        "splits": {
            "train": {
                "n_rows": n_train,
                "X_bin_shape": list(X_train_bin.shape),
                "X_offset_shape": list(X_train_offset.shape),
                "bin_min": int(X_train_bin.min()),
                "bin_max": int(X_train_bin.max()),
                "offset_min": float(X_train_offset.min()),
                "offset_max": float(X_train_offset.max()),
            },
            "val": {
                "n_rows": n_val,
                "X_bin_shape": list(X_val_bin.shape),
                "X_offset_shape": list(X_val_offset.shape),
                "bin_min": int(X_val_bin.min()),
                "bin_max": int(X_val_bin.max()),
                "offset_min": float(X_val_offset.min()),
                "offset_max": float(X_val_offset.max()),
            },
        },
        "feature_meta": feature_meta,
        "outputs": {
            "dataset_npz": str(out_npz),
            "metadata_json": str(out_meta),
        },
        "note": "Mixed representation. Quantile+offset only for features where diagnostic improves; others use uniform+offset or constant.",
    }

    out_meta.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print("===== build mixed quantile offset done =====")
    print(f"K={K}, B={B}")
    print(f"features={n_feat}")
    print(f"strategy_counts={strategy_counts}")
    print(f"dataset={out_npz}")
    print(f"metadata={out_meta}")



# ===== integrated rank-uniform-only source B =====

def _rank_as_str_list(arr):
    out = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return out


def _rank_entropy_norm_from_counts(counts):
    counts = np.asarray(counts, dtype=np.float64)
    counts = counts[counts > 0]
    if counts.size <= 1:
        return 0.0
    p = counts / counts.sum()
    h = -np.sum(p * np.log(p + 1e-12))
    return float(h / np.log(counts.size))


def _rank_uniform_bin_offset(z, num_bins):
    z = np.asarray(z, dtype=np.float64)
    z = np.nan_to_num(z, nan=0.0, posinf=1.0, neginf=0.0)
    z = np.clip(z, 0.0, 1.0)

    scaled = z * float(num_bins)
    b = np.floor(scaled).astype(np.int64)
    b = np.clip(b, 0, num_bins - 1)

    off = scaled - b.astype(np.float64)
    off[z >= 1.0] = 1.0

    return b.astype(np.int64), np.clip(off, 0.0, 1.0).astype(np.float32)


def _rank_piecewise_unique_rank_fit_transform(train_values, val_values):
    train_values = np.asarray(train_values, dtype=np.float64)
    val_values = np.asarray(val_values, dtype=np.float64)

    finite = train_values[np.isfinite(train_values)]
    if finite.size == 0:
        train_values = np.zeros_like(train_values, dtype=np.float64)
        val_values = np.zeros_like(val_values, dtype=np.float64)
    else:
        fill = float(np.median(finite))
        train_values = np.nan_to_num(
            train_values,
            nan=fill,
            posinf=float(finite.max()),
            neginf=float(finite.min()),
        )
        val_values = np.nan_to_num(
            val_values,
            nan=fill,
            posinf=float(finite.max()),
            neginf=float(finite.min()),
        )

    uniq = np.unique(train_values)

    if uniq.size <= 1:
        return np.zeros_like(train_values, dtype=np.float64), np.zeros_like(val_values, dtype=np.float64), uniq

    ranks = np.linspace(0.0, 1.0, uniq.size, dtype=np.float64)

    idx = np.searchsorted(uniq, train_values, side="left")
    idx = np.clip(idx, 0, uniq.size - 1)
    z_train = ranks[idx]

    z_val = np.interp(val_values, uniq, ranks, left=0.0, right=1.0)

    return z_train, z_val, uniq


def _rank_feature_diag(feature, raw_train, z_train, bin_ids, num_bins):
    counts = np.bincount(np.asarray(bin_ids, dtype=np.int64), minlength=num_bins)
    used = int(np.count_nonzero(counts))
    n = int(len(raw_train))
    raw_unique = int(np.unique(raw_train[np.isfinite(raw_train)]).size) if n else 0

    rare_1 = int(np.sum(counts == 1))
    rare_5 = int(np.sum((counts > 0) & (counts <= 5)))
    rare_10 = int(np.sum((counts > 0) & (counts <= 10)))

    return {
        "feature": feature,
        "strategy": "rank_uniform_offset",
        "n": n,
        "raw_unique": raw_unique,
        "bins_used": used,
        "empty_bins": int(num_bins - used),
        "empty_bin_ratio": float((num_bins - used) / max(num_bins, 1)),
        "rare_bins_count_eq_1": rare_1,
        "rare_bins_count_le_5": rare_5,
        "rare_bins_count_le_10": rare_10,
        "rare_used_bin_ratio_le_5": float(rare_5 / max(used, 1)),
        "rare_used_bin_ratio_le_10": float(rare_10 / max(used, 1)),
        "dominant_bin_ratio": float(counts.max() / max(n, 1)),
        "entropy_norm": _rank_entropy_norm_from_counts(counts),
        "compression_factor": float(raw_unique / max(used, 1)),
        "z_min": float(np.min(z_train)) if n else None,
        "z_max": float(np.max(z_train)) if n else None,
        "uniform_transformed_bin_width": float(1.0 / num_bins),
    }


def _build_rank_uniform_only_source_B(*, K: int, B: int, out_root: Path) -> None:
    """
    Integrated B-source builder.

    This is not final C2. It creates:
      03_outputs/04_token/K512_B512_rank_uniform_only/dataset.npz
      03_outputs/04_token/K512_B512_rank_uniform_only/metadata.json

    05_build_dataset.py later consumes A+B to build C2.
    """
    out_root = Path(out_root)
    if not out_root.is_absolute():
        out_root = CFG.ROOT_DIR / out_root

    current_dir = out_root / f"K{K}_B{B}"
    out_dir = out_root / f"K{K}_B{B}_rank_uniform_only"

    template_npz_path = current_dir / "dataset.npz"
    template_meta_path = current_dir / "metadata.json"

    train_raw_path = Path(CFG.TRAIN_RAW_CSV)
    val_raw_path = Path(CFG.VAL_RAW_CSV)

    required = [train_raw_path, val_raw_path, template_npz_path, template_meta_path]
    for fp in required:
        if not fp.exists():
            raise FileNotFoundError(fp)

    train_df = pd.read_csv(train_raw_path)
    val_df = pd.read_csv(val_raw_path)

    with np.load(template_npz_path, allow_pickle=True) as data:
        template = {k: data[k] for k in data.files}
        if "feature_names" in data.files:
            feature_names = _rank_as_str_list(data["feature_names"])
        else:
            feature_names = [
                c for c in train_df.columns
                if c not in set(CFG.TARGET_COLS) and pd.api.types.is_numeric_dtype(train_df[c])
            ]

    n_train = len(train_df)
    n_val = len(val_df)
    n_features = len(feature_names)

    X_train_bin = np.zeros((n_train, n_features), dtype=np.int64)
    X_val_bin = np.zeros((n_val, n_features), dtype=np.int64)
    X_train_offset = np.zeros((n_train, n_features), dtype=np.float32)
    X_val_offset = np.zeros((n_val, n_features), dtype=np.float32)

    strategies = {}
    rows = []
    constant_features = []

    for j, feat in enumerate(feature_names):
        tr_raw = train_df[feat].to_numpy(dtype=np.float64)
        va_raw = val_df[feat].to_numpy(dtype=np.float64)

        finite = tr_raw[np.isfinite(tr_raw)]
        raw_unique = int(np.unique(finite).size) if finite.size else 1

        if raw_unique <= 1:
            strategies[feat] = "constant"
            constant_features.append(feat)

            X_train_bin[:, j] = 0
            X_val_bin[:, j] = 0
            X_train_offset[:, j] = 0.0
            X_val_offset[:, j] = 0.0

            rows.append({
                "feature": feat,
                "strategy": "constant",
                "n": int(n_train),
                "raw_unique": int(raw_unique),
                "bins_used": 1,
                "empty_bins": int(B - 1),
                "empty_bin_ratio": float((B - 1) / B),
                "rare_bins_count_eq_1": 0,
                "rare_bins_count_le_5": 0,
                "rare_bins_count_le_10": 0,
                "rare_used_bin_ratio_le_5": 0.0,
                "rare_used_bin_ratio_le_10": 0.0,
                "dominant_bin_ratio": 1.0,
                "entropy_norm": 0.0,
                "compression_factor": float(raw_unique),
                "z_min": 0.0,
                "z_max": 0.0,
                "uniform_transformed_bin_width": float(1.0 / B),
            })
            continue

        strategies[feat] = "rank_uniform_offset"

        z_tr, z_va, _uniq = _rank_piecewise_unique_rank_fit_transform(tr_raw, va_raw)

        bt, ot = _rank_uniform_bin_offset(z_tr, B)
        bv, ov = _rank_uniform_bin_offset(z_va, B)

        X_train_bin[:, j] = bt
        X_val_bin[:, j] = bv
        X_train_offset[:, j] = ot
        X_val_offset[:, j] = ov

        rows.append(_rank_feature_diag(feat, tr_raw, z_tr, bt, B))

    out_dir.mkdir(parents=True, exist_ok=True)

    out_arrays = dict(template)
    out_arrays["X_train_bin"] = X_train_bin
    out_arrays["X_val_bin"] = X_val_bin
    out_arrays["X_train_offset"] = X_train_offset
    out_arrays["X_val_offset"] = X_val_offset

    np.savez_compressed(out_dir / "dataset.npz", **out_arrays)

    current_meta = json.loads(template_meta_path.read_text(encoding="utf-8"))

    meta = dict(current_meta)
    meta["stage"] = "rank_uniform_policy_ablation"
    meta["policy_name"] = "rank_uniform_only"
    meta["K"] = K
    meta["num_bins"] = B
    meta["source_raw_train"] = str(train_raw_path)
    meta["source_raw_val"] = str(val_raw_path)
    meta["source_template_npz"] = str(template_npz_path)
    meta["strategy_counts"] = {
        "rank_uniform_offset": int(n_features - len(constant_features)),
        "uniform_offset": 0,
        "quantile_offset": 0,
        "constant": int(len(constant_features)),
    }
    meta["constant_features"] = constant_features
    meta["feature_strategies"] = strategies
    meta["splits"] = {
        "train": {
            "n_rows": int(n_train),
            "X_bin_shape": list(X_train_bin.shape),
            "X_offset_shape": list(X_train_offset.shape),
            "bin_min": int(X_train_bin.min()),
            "bin_max": int(X_train_bin.max()),
            "offset_min": float(X_train_offset.min()),
            "offset_max": float(X_train_offset.max()),
        },
        "val": {
            "n_rows": int(n_val),
            "X_bin_shape": list(X_val_bin.shape),
            "X_offset_shape": list(X_val_offset.shape),
            "bin_min": int(X_val_bin.min()),
            "bin_max": int(X_val_bin.max()),
            "offset_min": float(X_val_offset.min()),
            "offset_max": float(X_val_offset.max()),
        },
    }

    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    diag_dir = Path(CFG.BIN_DIAG_DIR)
    diag_dir.mkdir(parents=True, exist_ok=True)

    diag_csv = diag_dir / f"rank_uniform_token_diag_K{K}_B{B}.csv"
    diag_json = diag_dir / f"rank_uniform_token_diag_K{K}_B{B}.json"

    df_diag = pd.DataFrame(rows)
    df_diag.to_csv(diag_csv, index=False)

    nonconst = df_diag[df_diag["strategy"] != "constant"]

    summary = {
        "stage": "rank_uniform_token_diag",
        "policy_name": "rank_uniform_only",
        "K": K,
        "num_bins": B,
        "n_features": int(n_features),
        "n_constant": int(len(constant_features)),
        "n_nonconstant": int(len(nonconst)),
        "mean_bins_used_nonconstant": float(nonconst["bins_used"].mean()) if len(nonconst) else 0.0,
        "median_bins_used_nonconstant": float(nonconst["bins_used"].median()) if len(nonconst) else 0.0,
        "mean_empty_bin_ratio_nonconstant": float(nonconst["empty_bin_ratio"].mean()) if len(nonconst) else 0.0,
        "mean_compression_nonconstant": float(nonconst["compression_factor"].mean()) if len(nonconst) else 0.0,
        "median_compression_nonconstant": float(nonconst["compression_factor"].median()) if len(nonconst) else 0.0,
        "mean_entropy_nonconstant": float(nonconst["entropy_norm"].mean()) if len(nonconst) else 0.0,
        "mean_dominant_bin_ratio_nonconstant": float(nonconst["dominant_bin_ratio"].mean()) if len(nonconst) else 0.0,
        "features_full_512_bins": int((nonconst["bins_used"] == B).sum()) if len(nonconst) else 0,
        "features_bins_used_ge_400": int((nonconst["bins_used"] >= 400).sum()) if len(nonconst) else 0,
        "features_bins_used_lt_128": int((nonconst["bins_used"] < 128).sum()) if len(nonconst) else 0,
        "features_rare_ratio_le_5_gt_0_2": int((nonconst["rare_used_bin_ratio_le_5"] > 0.2).sum()) if len(nonconst) else 0,
    }

    diag_json.write_text(json.dumps({"summary": summary, "features": rows}, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_path = out_root / f"K{K}_A_current_vs_B_rank_uniform_summary.json"
    summary_path.write_text(json.dumps({
        "A_current_mixed_artifact": str(current_dir),
        "B_rank_uniform_artifact": str(out_dir),
        "B_rank_uniform_diag_csv": str(diag_csv),
        "B_rank_uniform_diag_json": str(diag_json),
        "B_summary": summary,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print("===== integrated rank-uniform source B done =====")
    print("A current:", current_dir)
    print("B rank-uniform:", out_dir)
    print("B diag:", diag_json)


def main() -> None:
    args = parse_args()
    _main_build_current_mixed()
    _build_rank_uniform_only_source_B(
        K=int(args.K),
        B=int(args.num_bins),
        out_root=Path(args.out_root),
    )


if __name__ == "__main__":
    main()
