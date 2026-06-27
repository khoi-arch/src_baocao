#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_build_dataset.py

Build the official C2 final dataset from token source artifacts A+B.

Inputs:
  A current mixed token source:
    03_outputs/04_token/K{K}_B{B}/token_artifact.npz
  B rank-uniform token source:
    03_outputs/04_token/K{K}_B{B}_rank_uniform_only/token_artifact.npz
  Raw train/val CSV for discrete compact decisions.

Output:
  03_outputs/05_dataset/dataset.npz
  03_outputs/05_dataset/metadata.json
  03_outputs/05_dataset/K{K}_C2_hybrid_policy_diag.csv/json
  03_outputs/05_dataset/K{K}_C2_hybrid_summary.json

This file must not write anything under 03_outputs/04_token.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import config as CFG


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build official C2 final dataset.")
    p.add_argument("--K", type=int, default=int(CFG.TOKEN_K))
    p.add_argument("--num-bins", type=int, default=int(CFG.VALUE_NUM_BINS))
    p.add_argument("--train-raw", default=str(CFG.TRAIN_RAW_CSV))
    p.add_argument("--val-raw", default=str(CFG.VAL_RAW_CSV))
    p.add_argument("--A-dir", default="")
    p.add_argument("--B-dir", default="")
    p.add_argument("--out-dir", default=str(CFG.DATASET_DIR))
    return p.parse_args()


def as_str_list(arr) -> List[str]:
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in arr]


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def entropy_norm_from_counts(counts):
    counts = np.asarray(counts, dtype=np.float64)
    counts = counts[counts > 0]
    if counts.size <= 1:
        return 0.0
    p = counts / counts.sum()
    h = -np.sum(p * np.log(p + 1e-12))
    return float(h / np.log(counts.size))


def bin_stats(bin_ids, raw_unique: int, num_bins: int) -> Dict[str, object]:
    counts = np.bincount(np.asarray(bin_ids, dtype=np.int64), minlength=int(num_bins))
    used = int(np.count_nonzero(counts))
    rare_le5 = int(np.sum((counts > 0) & (counts <= 5)))
    rare_le10 = int(np.sum((counts > 0) & (counts <= 10)))
    return {
        "bins_used": used,
        "empty_bins": int(num_bins - used),
        "empty_ratio": float((num_bins - used) / max(num_bins, 1)),
        "dominant_bin_ratio": float(counts.max() / max(counts.sum(), 1)),
        "entropy_norm": entropy_norm_from_counts(counts),
        "compression_factor": float(raw_unique / max(used, 1)),
        "rare_bins_le5": rare_le5,
        "rare_bins_le10": rare_le10,
        "rare_used_bin_ratio_le5": float(rare_le5 / max(used, 1)),
        "rare_used_bin_ratio_le10": float(rare_le10 / max(used, 1)),
    }


def nearest_unique_index(values, uniq):
    values = np.asarray(values, dtype=np.float64)
    uniq = np.asarray(uniq, dtype=np.float64)
    idx = np.searchsorted(uniq, values, side="left")
    idx = np.clip(idx, 0, len(uniq) - 1)
    left_idx = np.clip(idx - 1, 0, len(uniq) - 1)
    right_idx = idx
    left_dist = np.abs(values - uniq[left_idx])
    right_dist = np.abs(values - uniq[right_idx])
    choose_left = left_dist < right_dist
    return np.where(choose_left, left_idx, right_idx).astype(np.int64)


def make_discrete_compact(train_values, val_values, num_bins: int):
    train_values = np.asarray(train_values, dtype=np.float64)
    val_values = np.asarray(val_values, dtype=np.float64)
    finite = train_values[np.isfinite(train_values)]
    if finite.size == 0:
        train_values = np.zeros_like(train_values, dtype=np.float64)
        val_values = np.zeros_like(val_values, dtype=np.float64)
    else:
        fill = float(np.median(finite))
        train_values = np.nan_to_num(train_values, nan=fill, posinf=float(finite.max()), neginf=float(finite.min()))
        val_values = np.nan_to_num(val_values, nan=fill, posinf=float(finite.max()), neginf=float(finite.min()))
    uniq = np.unique(train_values)
    if uniq.size > int(num_bins):
        raise ValueError(f"discrete_compact requires unique <= num_bins, got {uniq.size}")
    tr_idx = nearest_unique_index(train_values, uniq)
    va_idx = nearest_unique_index(val_values, uniq)
    tr_off = np.zeros_like(tr_idx, dtype=np.float32)
    va_off = np.zeros_like(va_idx, dtype=np.float32)
    return tr_idx.astype(np.int64), tr_off, va_idx.astype(np.int64), va_off, uniq


def choose_c2_strategy(raw_unique: int, A_stats: Dict[str, object], B_stats: Dict[str, object]) -> tuple[str, Dict[str, bool]]:
    is_constant = raw_unique <= 1
    rank_candidate = (
        raw_unique >= int(CFG.C2_RANK_MIN_RAW_UNIQUE)
        and (
            A_stats["compression_factor"] >= float(CFG.C2_RANK_MIN_COMPRESSION)
            or A_stats["dominant_bin_ratio"] >= float(CFG.C2_RANK_MIN_DOMINANT)
            or A_stats["entropy_norm"] < float(CFG.C2_RANK_MAX_ENTROPY)
        )
        and B_stats["bins_used"] >= int(CFG.C2_RANK_MIN_B_BINS_USED)
        and B_stats["rare_used_bin_ratio_le5"] <= float(CFG.C2_RANK_MAX_B_RARE_LE5_RATIO)
    )
    compact_allowed = (
        raw_unique <= int(CFG.C2_COMPACT_MAX_UNIQUE)
        or (
            raw_unique <= int(CFG.C2_COMPACT_MAX_UNIQUE_WITH_LOW_BINS)
            and A_stats["bins_used"] <= int(CFG.C2_COMPACT_MAX_USED_BINS)
            and B_stats["bins_used"] <= int(CFG.C2_COMPACT_MAX_USED_BINS)
        )
    )
    low_unique_discrete_signal = raw_unique <= int(CFG.C2_COMPACT_MAX_UNIQUE) or A_stats["bins_used"] <= int(CFG.C2_COMPACT_MAX_USED_BINS) or B_stats["bins_used"] <= int(CFG.C2_COMPACT_MAX_USED_BINS)
    if is_constant:
        strategy = "constant"
    elif rank_candidate:
        strategy = "rank_uniform_offset"
    elif compact_allowed:
        strategy = "discrete_compact_offset0"
    else:
        strategy = "keep_current"
    return strategy, {
        "is_constant": bool(is_constant),
        "rank_candidate": bool(rank_candidate),
        "compact_allowed": bool(compact_allowed),
        "low_unique_discrete_signal": bool(low_unique_discrete_signal),
    }


def main() -> None:
    args = parse_args()
    K = int(args.K)
    B = int(args.num_bins)
    train_raw_path = Path(args.train_raw)
    val_raw_path = Path(args.val_raw)
    A_dir = Path(args.A_dir) if args.A_dir else CFG.build_mixed_dir(K, B)
    B_dir = Path(args.B_dir) if args.B_dir else CFG.build_rank_uniform_dir(K, B)
    out_dir = Path(args.out_dir)

    A_npz_path = A_dir / "token_artifact.npz"
    B_npz_path = B_dir / "token_artifact.npz"
    A_meta_path = A_dir / "metadata.json"
    B_meta_path = B_dir / "metadata.json"
    required = [train_raw_path, val_raw_path, A_npz_path, B_npz_path, A_meta_path, B_meta_path]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(str(p))

    train_df = pd.read_csv(train_raw_path)
    val_df = pd.read_csv(val_raw_path)
    with np.load(A_npz_path, allow_pickle=True) as A_data, np.load(B_npz_path, allow_pickle=True) as B_data:
        A = {k: A_data[k] for k in A_data.files}
        Bdata = {k: B_data[k] for k in B_data.files}
        feature_names = as_str_list(A_data["feature_names"])

    for k in ["X_train_bin", "X_train_offset", "X_val_bin", "X_val_offset", "y_train", "y_val"]:
        if k not in A:
            raise ValueError(f"A token artifact missing array: {k}")
        if k in ["X_train_bin", "X_train_offset", "X_val_bin", "X_val_offset"] and k not in Bdata:
            raise ValueError(f"B token artifact missing array: {k}")
    if A["X_train_bin"].shape != Bdata["X_train_bin"].shape:
        raise ValueError(f"A/B train shape mismatch: {A['X_train_bin'].shape} vs {Bdata['X_train_bin'].shape}")
    if A["X_val_bin"].shape != Bdata["X_val_bin"].shape:
        raise ValueError(f"A/B val shape mismatch: {A['X_val_bin'].shape} vs {Bdata['X_val_bin'].shape}")

    C2 = {k: np.array(v, copy=True) for k, v in A.items()}
    rows = []
    strategies: Dict[str, str] = {}
    compact_unique_values = {}

    for j, feat in enumerate(feature_names):
        if feat not in train_df.columns or feat not in val_df.columns:
            raise ValueError(f"raw split missing feature: {feat}")
        tr_raw = train_df[feat].to_numpy(dtype=np.float64)
        va_raw = val_df[feat].to_numpy(dtype=np.float64)
        finite = tr_raw[np.isfinite(tr_raw)]
        raw_unique = int(np.unique(finite).size) if finite.size else 1
        A_stats = bin_stats(A["X_train_bin"][:, j], raw_unique, B)
        B_stats = bin_stats(Bdata["X_train_bin"][:, j], raw_unique, B)
        c2_strategy, flags = choose_c2_strategy(raw_unique, A_stats, B_stats)

        if c2_strategy == "rank_uniform_offset":
            C2["X_train_bin"][:, j] = Bdata["X_train_bin"][:, j]
            C2["X_val_bin"][:, j] = Bdata["X_val_bin"][:, j]
            C2["X_train_offset"][:, j] = Bdata["X_train_offset"][:, j]
            C2["X_val_offset"][:, j] = Bdata["X_val_offset"][:, j]
        elif c2_strategy == "discrete_compact_offset0":
            tr_b, tr_o, va_b, va_o, uniq = make_discrete_compact(tr_raw, va_raw, B)
            C2["X_train_bin"][:, j] = tr_b
            C2["X_val_bin"][:, j] = va_b
            C2["X_train_offset"][:, j] = tr_o
            C2["X_val_offset"][:, j] = va_o
            compact_unique_values[feat] = {"n_unique": int(len(uniq)), "unique_values_preview": [float(x) for x in uniq[:50].tolist()]}
        # constant and keep_current keep A as source.

        strategies[feat] = c2_strategy
        rows.append({
            "feature": feat,
            "raw_unique": raw_unique,
            "A_bins_used": A_stats["bins_used"],
            "A_empty_ratio": A_stats["empty_ratio"],
            "A_compression_factor": A_stats["compression_factor"],
            "A_dominant_bin_ratio": A_stats["dominant_bin_ratio"],
            "A_entropy_norm": A_stats["entropy_norm"],
            "A_rare_used_bin_ratio_le5": A_stats["rare_used_bin_ratio_le5"],
            "B_bins_used": B_stats["bins_used"],
            "B_empty_ratio": B_stats["empty_ratio"],
            "B_compression_factor": B_stats["compression_factor"],
            "B_dominant_bin_ratio": B_stats["dominant_bin_ratio"],
            "B_entropy_norm": B_stats["entropy_norm"],
            "B_rare_used_bin_ratio_le5": B_stats["rare_used_bin_ratio_le5"],
            **flags,
            "C2_strategy": c2_strategy,
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / "dataset.npz"
    out_meta = out_dir / "metadata.json"
    np.savez_compressed(out_npz, **C2)

    df = pd.DataFrame(rows)
    diag_csv = out_dir / f"K{K}_C2_hybrid_policy_diag.csv"
    diag_json = out_dir / f"K{K}_C2_hybrid_policy_diag.json"
    df.to_csv(diag_csv, index=False)

    def counts_for(col):
        return {str(k): int(v) for k, v in df[col].value_counts().to_dict().items()}

    summary = {
        "stage": "05_build_C2_final_dataset",
        "policy_name": "C2_selective_rank_discrete_compact",
        "K": K,
        "num_bins": B,
        "n_features": int(len(feature_names)),
        "C2_strategy_counts": counts_for("C2_strategy"),
        "n_rank_candidates": int(df["rank_candidate"].sum()),
        "n_low_unique_discrete_signal": int(df["low_unique_discrete_signal"].sum()),
        "n_compact_allowed": int(df["compact_allowed"].sum()),
        "rank_candidate_features": df[df["rank_candidate"]]["feature"].tolist(),
        "C2_discrete_compact_features": df[df["C2_strategy"] == "discrete_compact_offset0"]["feature"].tolist(),
        "C2_keep_current_features": df[df["C2_strategy"] == "keep_current"]["feature"].tolist(),
        "thresholds": {
            "constant": "raw_unique <= 1",
            "compact_allowed": f"raw_unique <= {CFG.C2_COMPACT_MAX_UNIQUE} OR (raw_unique <= {CFG.C2_COMPACT_MAX_UNIQUE_WITH_LOW_BINS} AND A_bins_used <= {CFG.C2_COMPACT_MAX_USED_BINS} AND B_bins_used <= {CFG.C2_COMPACT_MAX_USED_BINS})",
            "rank_candidate": f"raw_unique >= {CFG.C2_RANK_MIN_RAW_UNIQUE} AND (A_compression >= {CFG.C2_RANK_MIN_COMPRESSION} OR A_dominant >= {CFG.C2_RANK_MIN_DOMINANT} OR A_entropy < {CFG.C2_RANK_MAX_ENTROPY}) AND B_bins_used >= {CFG.C2_RANK_MIN_B_BINS_USED} AND B_rare_used_bin_ratio_le5 <= {CFG.C2_RANK_MAX_B_RARE_LE5_RATIO}",
        },
    }
    save_json({"summary": summary, "features": rows}, diag_json)

    A_meta = load_json(A_meta_path)
    B_meta = load_json(B_meta_path)
    label_mapping = A_meta.get("label_mapping", {})
    label_names = as_str_list(A.get("label_names", np.asarray([]))) if "label_names" in A else [k for k, _ in sorted(label_mapping.items(), key=lambda kv: kv[1])]

    meta = dict(A_meta)
    meta.update({
        "stage": "05_build_C2_final_dataset",
        "policy_name": "C2_selective_rank_discrete_compact",
        "K": K,
        "num_bins": B,
        "source_A_current_mixed": str(A_dir),
        "source_A_token_artifact": str(A_npz_path),
        "source_B_rank_uniform": str(B_dir),
        "source_B_token_artifact": str(B_npz_path),
        "source_raw_train": str(train_raw_path),
        "source_raw_val": str(val_raw_path),
        "source_note": "rank candidates use B rank-uniform; compact_allowed low/discrete features use compact token ids with offset=0; rest keep A current mixed.",
        "strategy_counts": summary["C2_strategy_counts"],
        "feature_strategies": strategies,
        "compact_unique_values_preview": compact_unique_values,
        "policy_diag_csv": str(diag_csv),
        "policy_diag_json": str(diag_json),
        "thresholds": summary["thresholds"],
        "label_mapping": label_mapping,
        "label_names": label_names,
        "n_features": int(len(feature_names)),
        "feature_names": feature_names,
        "splits": {
            "train": {"n_rows": int(C2["X_train_bin"].shape[0]), "X_bin_shape": list(C2["X_train_bin"].shape), "X_offset_shape": list(C2["X_train_offset"].shape), "bin_min": int(C2["X_train_bin"].min()), "bin_max": int(C2["X_train_bin"].max()), "offset_min": float(C2["X_train_offset"].min()), "offset_max": float(C2["X_train_offset"].max())},
            "val": {"n_rows": int(C2["X_val_bin"].shape[0]), "X_bin_shape": list(C2["X_val_bin"].shape), "X_offset_shape": list(C2["X_val_offset"].shape), "bin_min": int(C2["X_val_bin"].min()), "bin_max": int(C2["X_val_bin"].max()), "offset_min": float(C2["X_val_offset"].min()), "offset_max": float(C2["X_val_offset"].max())},
        },
        "outputs": {"dataset_npz": str(out_npz), "metadata_json": str(out_meta)},
    })
    save_json(meta, out_meta)

    comparison_path = out_dir / f"K{K}_C2_hybrid_summary.json"
    save_json({"A_current_mixed": str(A_dir), "B_rank_uniform_all": str(B_dir), "C2_selective_rank_discrete_compact": str(out_dir), "summary": summary}, comparison_path)

    print("===== C2 final dataset build done =====")
    print(f"dataset:  {out_npz}")
    print(f"metadata: {out_meta}")
    print(f"diag:     {diag_json}")
    print(json.dumps(summary, indent=2, ensure_ascii=False)[:6000])


if __name__ == "__main__":
    main()
