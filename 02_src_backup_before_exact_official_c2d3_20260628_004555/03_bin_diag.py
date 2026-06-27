#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_bin_diag.py

Diagnostic only. No labels, no training, no model change.

Purpose:
  Compare uniform+offset and quantile+offset using the exact same bin assignment
  rule that 04_tokenization.py uses. This avoids deciding quantile strategy from
  a different tokenization rule.
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
    p = argparse.ArgumentParser(description="Compare uniform+offset vs quantile+offset on preprocessed z.")
    p.add_argument("--K", type=int, default=int(CFG.TOKEN_K))
    p.add_argument("--num-bins", type=int, default=int(CFG.VALUE_NUM_BINS))
    p.add_argument("--train-preprocessed", default="")
    p.add_argument("--policy-json", default="")
    p.add_argument("--out-dir", default=str(CFG.BIN_DIAG_DIR))
    return p.parse_args()


def load_feature_order(policy_path: Path) -> List[str]:
    if not policy_path.exists():
        raise FileNotFoundError(f"policy json not found: {policy_path}")
    obj = json.loads(policy_path.read_text(encoding="utf-8"))
    meta_features = obj.get("metadata", {}).get("feature_order")
    if meta_features:
        return [str(x) for x in meta_features]
    policies = obj.get("policies", [])
    if not policies:
        raise ValueError(f"No policies found in {policy_path}")
    return [str(p["feature"]) for p in policies]


def entropy_norm_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    total = float(counts.sum())
    used = int(np.count_nonzero(counts))
    if total <= 0.0 or used <= 1:
        return 0.0
    p = counts[counts > 0] / total
    ent = -float(np.sum(p * np.log(p + 1e-12)))
    return float(ent / np.log(used))


def bin_stats(bin_ids: np.ndarray, num_bins: int) -> Dict[str, object]:
    bin_ids = np.asarray(bin_ids, dtype=np.int64)
    counts = np.bincount(bin_ids, minlength=int(num_bins)).astype(np.int64)
    used = int(np.count_nonzero(counts))
    n = int(counts.sum())
    max_count = int(counts.max()) if counts.size else 0
    return {
        "bins_used": used,
        "empty_bins": int(num_bins - used),
        "max_bin_count": max_count,
        "dominant_bin_ratio": float(max_count / max(n, 1)),
        "entropy_norm": entropy_norm_from_counts(counts),
        "mean_nonempty_bin_count": float(counts[counts > 0].mean()) if used else 0.0,
        "rare_bins_le5": int(np.sum((counts > 0) & (counts <= 5))),
        "rare_bins_le10": int(np.sum((counts > 0) & (counts <= 10))),
    }


def uniform_edges(num_bins: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, int(num_bins) + 1, dtype=np.float64)


def quantile_edges(values: np.ndarray, num_bins: int) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    qs = np.linspace(0.0, 1.0, int(num_bins) + 1)
    try:
        edges = np.quantile(v, qs, method="linear")
    except TypeError:
        edges = np.quantile(v, qs, interpolation="linear")
    edges[0] = float(np.min(v))
    edges[-1] = float(np.max(v))
    return edges.astype(np.float64)


def assign_bin_offset(values: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    v = np.asarray(values, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    num_bins = int(len(edges) - 1)
    if num_bins <= 0:
        raise ValueError("edges must have at least 2 values")
    if np.isclose(edges[0], edges[-1]):
        b = np.zeros(v.shape, dtype=np.int64)
        off = np.full(v.shape, 0.5, dtype=np.float32)
        diag = {
            "edge_count": int(len(edges)),
            "unique_edge_count": int(len(np.unique(edges))),
            "duplicate_edge_count": int(len(edges) - len(np.unique(edges))),
            "zero_width_bin_count": int(num_bins),
            "zero_width_sample_ratio": 1.0 if v.size else 0.0,
            "offset_min": 0.5,
            "offset_max": 0.5,
            "offset_mean": 0.5,
        }
        return b, off, diag

    v_clip = np.clip(v, edges[0], edges[-1])
    internal = edges[1:-1]
    bin_id = np.searchsorted(internal, v_clip, side="right")
    bin_id = np.clip(bin_id, 0, num_bins - 1).astype(np.int64)

    left = edges[bin_id]
    right = edges[bin_id + 1]
    width = right - left
    zero_width = np.isclose(width, 0.0)
    offset = np.zeros(v.shape, dtype=np.float32)
    offset[~zero_width] = ((v_clip[~zero_width] - left[~zero_width]) / width[~zero_width]).astype(np.float32)
    offset[zero_width] = 0.5
    offset = np.clip(offset, 0.0, 1.0).astype(np.float32)

    diag = {
        "edge_count": int(len(edges)),
        "unique_edge_count": int(len(np.unique(edges))),
        "duplicate_edge_count": int(len(edges) - len(np.unique(edges))),
        "zero_width_bin_count": int(np.sum(np.isclose(np.diff(edges), 0.0))),
        "zero_width_sample_ratio": float(np.mean(zero_width)) if v.size else 0.0,
        "offset_min": float(offset.min()) if offset.size else None,
        "offset_max": float(offset.max()) if offset.size else None,
        "offset_mean": float(offset.mean()) if offset.size else None,
    }
    return bin_id, offset, diag


def main() -> None:
    args = parse_args()
    K = int(args.K)
    B = int(args.num_bins)
    if B <= 1:
        raise ValueError("num-bins must be > 1")

    train_pre_path = Path(args.train_preprocessed) if args.train_preprocessed else CFG.preprocess_train_csv_path(K)
    policy_path = Path(args.policy_json) if args.policy_json else CFG.preprocess_policy_json_path(K)
    if not train_pre_path.exists():
        raise FileNotFoundError(f"train_preprocessed not found: {train_pre_path}")
    if not policy_path.exists():
        raise FileNotFoundError(f"policy json not found: {policy_path}")

    features = load_feature_order(policy_path)
    df = pd.read_csv(train_pre_path)
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"train_preprocessed missing features: {missing[:10]}")

    rows: List[Dict[str, object]] = []
    for f in features:
        z = np.clip(df[f].to_numpy(dtype=np.float64), 0.0, 1.0)

        u_edges = uniform_edges(B)
        u_bin, u_offset, u_diag = assign_bin_offset(z, u_edges)
        u_stats = bin_stats(u_bin, B)

        q_edges = quantile_edges(z, B)
        q_bin, q_offset, q_diag = assign_bin_offset(z, q_edges)
        q_stats = bin_stats(q_bin, B)

        rows.append({
            "feature": f,
            "n": int(len(z)),
            "num_bins": B,
            "z_min": float(z.min()) if z.size else None,
            "z_max": float(z.max()) if z.size else None,
            "z_num_unique": int(np.unique(z).size),
            "uniform_bins_used": u_stats["bins_used"],
            "quantile_bins_used": q_stats["bins_used"],
            "delta_bins_used": int(q_stats["bins_used"] - u_stats["bins_used"]),
            "uniform_empty_bins": u_stats["empty_bins"],
            "quantile_empty_bins": q_stats["empty_bins"],
            "uniform_dominant_bin_ratio": u_stats["dominant_bin_ratio"],
            "quantile_dominant_bin_ratio": q_stats["dominant_bin_ratio"],
            "delta_dominant_bin_ratio": float(q_stats["dominant_bin_ratio"] - u_stats["dominant_bin_ratio"]),
            "dominant_ratio_reduction": float(u_stats["dominant_bin_ratio"] - q_stats["dominant_bin_ratio"]),
            "uniform_max_bin_count": u_stats["max_bin_count"],
            "quantile_max_bin_count": q_stats["max_bin_count"],
            "uniform_entropy_norm": u_stats["entropy_norm"],
            "quantile_entropy_norm": q_stats["entropy_norm"],
            "delta_entropy_norm": float(q_stats["entropy_norm"] - u_stats["entropy_norm"]),
            "uniform_offset_min": u_diag["offset_min"],
            "uniform_offset_max": u_diag["offset_max"],
            "uniform_offset_mean": u_diag["offset_mean"],
            "quantile_duplicate_edge_count": q_diag["duplicate_edge_count"],
            "quantile_zero_width_bin_count": q_diag["zero_width_bin_count"],
            "quantile_zero_width_sample_ratio": q_diag["zero_width_sample_ratio"],
            "quantile_offset_min": q_diag["offset_min"],
            "quantile_offset_max": q_diag["offset_max"],
            "quantile_offset_mean": q_diag["offset_mean"],
        })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"quantile_vs_uniform_bin_diag_K{K}_B{B}.csv"
    out_json = out_dir / f"quantile_vs_uniform_bin_diag_K{K}_B{B}.json"

    result_df = pd.DataFrame(rows)
    result_df.to_csv(out_csv, index=False)
    improved_dom = result_df[result_df["dominant_ratio_reduction"] > 1e-12]
    worsened_dom = result_df[result_df["dominant_ratio_reduction"] < -1e-12]
    improved_entropy = result_df[result_df["delta_entropy_norm"] > 1e-12]
    summary = {
        "stage": "quantile_vs_uniform_bin_diag",
        "bin_rule": "shared_edge_searchsorted_offset_rule_used_by_04_tokenization",
        "K": K,
        "num_bins": B,
        "input_train_preprocessed": str(train_pre_path),
        "policy_json": str(policy_path),
        "n_features": int(len(rows)),
        "mean_uniform_dominant_bin_ratio": float(result_df["uniform_dominant_bin_ratio"].mean()),
        "mean_quantile_dominant_bin_ratio": float(result_df["quantile_dominant_bin_ratio"].mean()),
        "mean_dominant_ratio_reduction": float(result_df["dominant_ratio_reduction"].mean()),
        "mean_uniform_entropy_norm": float(result_df["uniform_entropy_norm"].mean()),
        "mean_quantile_entropy_norm": float(result_df["quantile_entropy_norm"].mean()),
        "mean_delta_entropy_norm": float(result_df["delta_entropy_norm"].mean()),
        "features_dominant_ratio_improved": int(len(improved_dom)),
        "features_dominant_ratio_worsened": int(len(worsened_dom)),
        "features_entropy_improved": int(len(improved_entropy)),
        "features_with_quantile_zero_width_bins": int((result_df["quantile_zero_width_bin_count"] > 0).sum()),
        "top_dominant_ratio_reductions": result_df.sort_values("dominant_ratio_reduction", ascending=False).head(20).to_dict(orient="records"),
        "top_entropy_improvements": result_df.sort_values("delta_entropy_norm", ascending=False).head(20).to_dict(orient="records"),
        "top_quantile_degenerate_features": result_df.sort_values(["quantile_zero_width_bin_count", "quantile_zero_width_sample_ratio"], ascending=False).head(20).to_dict(orient="records"),
        "outputs": {"csv": str(out_csv), "json": str(out_json)},
    }
    out_json.write_text(json.dumps({"summary": summary, "features": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    print("===== quantile vs uniform bin diagnostic done =====")
    print(f"K={K}, B={B}, features={len(rows)}")
    print(f"csv:  {out_csv}")
    print(f"json: {out_json}")


if __name__ == "__main__":
    main()
