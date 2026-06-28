#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D1c attention audit for D1b pair-specific attention branch.

Purpose
-------
This script DOES NOT train.
It loads an already-trained D1b model checkpoint, forwards the validation set,
extracts pair-specific attention weights for RS/RT/ST, and compares attention
patterns across:
  - fixed samples:    baseline wrong -> D1b correct
  - damaged samples:  baseline correct -> D1b wrong
  - both_correct
  - both_wrong
  - important confusion directions

Expected usage from repo root:
  python 05_test/D1c_audit_pair_attention.py \
    --d1b-run-dir 05_test/outputs/D1b_official_fork_lam0p01 \
    --baseline-pred 03_outputs/06_model/val_predictions_best.csv \
    --dataset-npz 03_outputs/05_dataset/dataset.npz \
    --metadata-json 03_outputs/05_dataset/metadata.json \
    --train-raw 01_split/train_raw.csv \
    --val-raw 01_split/val_raw.csv \
    --d1b-script 05_test/D1b_train_official_fork.py \
    --out-dir 05_test/outputs/D1c_audit_D1b_lam0p01 \
    --device cuda \
    --batch-size 512
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
import torch
from torch.utils.data import DataLoader


PAIR_ORDER = [
    "Ransomware__vs__Spyware",
    "Ransomware__vs__Trojan",
    "Spyware__vs__Trojan",
]

PAIR_SHORT = {
    "Ransomware__vs__Spyware": "RS",
    "Ransomware__vs__Trojan": "RT",
    "Spyware__vs__Trojan": "ST",
}


def repo_root_from_here() -> Path:
    # If this file is at 05_test/D1c_*.py, parents[1] is repo root.
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


def import_d1b_module(d1b_script: Path):
    if not d1b_script.exists():
        raise FileNotFoundError(f"D1b script not found: {d1b_script}")

    # Make official 02_src imports visible too.
    repo_root = d1b_script.resolve().parents[1] if d1b_script.parent.name == "05_test" else Path.cwd().resolve()
    src_dir = repo_root / "02_src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    if str(d1b_script.parent) not in sys.path:
        sys.path.insert(0, str(d1b_script.parent))

    spec = importlib.util.spec_from_file_location("d1b_train_official_fork_for_audit", str(d1b_script))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import D1b script from {d1b_script}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def normalize_pred_df(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Normalize prediction CSV columns. Official files normally have:
      sample_index,true_id,true_label,pred_id,pred_label,correct,confidence
    """
    df = df.copy()

    if "sample_index" not in df.columns:
        df["sample_index"] = np.arange(len(df), dtype=int)

    needed = ["sample_index", "true_id", "true_label", "pred_id", "pred_label", "correct"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{prefix} prediction missing columns {missing}; available={list(df.columns)}")

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


def build_transition_df(baseline_pred: Path, d1b_pred: Path) -> pd.DataFrame:
    base = normalize_pred_df(pd.read_csv(baseline_pred), "base")
    d1b = normalize_pred_df(pd.read_csv(d1b_pred), "d1b")

    merged = base.merge(d1b, on="sample_index", how="inner")
    if len(merged) != len(base) or len(merged) != len(d1b):
        raise ValueError(
            f"Prediction alignment mismatch: base={len(base)} d1b={len(d1b)} merged={len(merged)}"
        )

    # Sanity check true labels are the same.
    if not (merged["base_true_id"].to_numpy() == merged["d1b_true_id"].to_numpy()).all():
        bad = merged.loc[merged["base_true_id"] != merged["d1b_true_id"], ["sample_index", "base_true_id", "d1b_true_id"]].head()
        raise ValueError(f"Baseline/D1b true label mismatch, examples:\n{bad}")

    merged["true_id"] = merged["base_true_id"]
    merged["true_label"] = merged["base_true_label"]
    merged["base_direction"] = merged["true_label"].astype(str) + "->" + merged["base_pred_label"].astype(str)
    merged["d1b_direction"] = merged["true_label"].astype(str) + "->" + merged["d1b_pred_label"].astype(str)

    merged["transition"] = "both_wrong"
    merged.loc[merged["base_correct"] & merged["d1b_correct"], "transition"] = "both_correct"
    merged.loc[(~merged["base_correct"]) & merged["d1b_correct"], "transition"] = "fixed"
    merged.loc[merged["base_correct"] & (~merged["d1b_correct"]), "transition"] = "damaged"

    merged["pred_changed"] = merged["base_pred_id"].astype(int) != merged["d1b_pred_id"].astype(int)
    return merged


def make_val_loader(d1b, args, repo_root: Path):
    dataset_npz = resolve_path(args.dataset_npz, repo_root)
    metadata_json = resolve_path(args.metadata_json, repo_root)
    data, meta = d1b.load_dataset(dataset_npz, metadata_json)

    X_train = data["X_train_bin"].astype(np.int64)
    O_train = data["X_train_offset"].astype(np.float32)
    y_train = data["y_train"].astype(np.int64)

    X_val = data["X_val_bin"].astype(np.int64)
    O_val = data["X_val_offset"].astype(np.float32)
    y_val = data["y_val"].astype(np.int64)

    # D3 official spec.
    spec = d1b.RUN_SPECS["D3"]

    raw_args = SimpleNamespace(
        train_raw=str(resolve_path(args.train_raw, repo_root)),
        val_raw=str(resolve_path(args.val_raw, repo_root)),
    )

    X_train_cont, X_val_cont, continuous_info = d1b.load_continuous_for_run(
        spec=spec,
        meta=meta,
        args=raw_args,
        train_shape=X_train.shape,
        val_shape=X_val.shape,
    )

    M_val = np.ones_like(X_val, dtype=np.float32)
    val_ds = d1b.FusionAblationDataset(X_val, O_val, X_val_cont, M_val, y_val)
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(torch.cuda.is_available() and str(args.device).startswith("cuda")),
    )

    label_mapping = meta["label_mapping"]
    label_names = [label for label, _ in sorted(label_mapping.items(), key=lambda kv: kv[1])]
    feature_names = [str(x) for x in meta["feature_names"]]

    return val_loader, y_val, label_names, feature_names, meta, continuous_info


def build_model_and_load(d1b, args, repo_root: Path, label_names: List[str], n_features: int):
    run_dir = resolve_path(args.d1b_run_dir, repo_root)
    config_path = run_dir / "config.json"
    ckpt_path = run_dir / "best_model.pt"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing D1b config.json: {config_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing D1b best_model.pt: {ckpt_path}")

    cfg = load_json(config_path)
    model_cfg = cfg.get("model", {})
    d1b_cfg = cfg.get("d1b", {})

    num_bins = int(cfg.get("num_bins", cfg.get("effective_token_budget", 512)))
    num_classes = int(cfg.get("num_classes", len(label_names)))

    device = pick_device(args.device)

    model = d1b.FusionAblationTransformer(
        run_id="D3",
        num_bins=num_bins,
        n_features=int(n_features),
        num_classes=num_classes,
        value_dim=int(model_cfg.get("value_dim", 32)),
        feature_dim=int(model_cfg.get("feature_dim", 32)),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        num_layers=int(model_cfg.get("num_layers", 3)),
        num_heads=int(model_cfg.get("num_heads", 4)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        classifier_hidden_dim=int(model_cfg.get("classifier_hidden_dim", 128)),
        classifier_dropout=float(model_cfg.get("classifier_dropout", 0.1)),
        norm_first=bool(model_cfg.get("norm_first", True)),
        gate_init=float(model_cfg.get("gate_init", 0.0)),
        activation=str(d1b.cfg("MODEL_ACTIVATION", "gelu")) if hasattr(d1b, "cfg") else "gelu",
    ).to(device)

    # Attach D1b attention branch before loading checkpoint.
    aux_hidden_dim = int(d1b_cfg.get("aux_hidden_dim", 0))
    aux_dropout = float(d1b_cfg.get("aux_dropout", 0.1))
    if hasattr(d1b, "attach_d1b_aux_heads"):
        d1b.attach_d1b_aux_heads(
            model,
            label_names=label_names,
            hidden_dim=int(model_cfg.get("hidden_dim", 128)),
            aux_hidden_dim=aux_hidden_dim,
            aux_dropout=aux_dropout,
        )
        model.to(device)
    else:
        raise RuntimeError("D1b script does not expose attach_d1b_aux_heads(). Use the fixed D1b_train_official_fork.py.")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # strict=False is used only to produce a clearer diagnostic; fail if important keys are missing/unexpected.
    bad_missing = [k for k in missing if not k.startswith("_")]
    bad_unexpected = [k for k in unexpected if not k.startswith("_")]
    if bad_missing or bad_unexpected:
        print("[WARN] load_state_dict non-strict mismatch")
        print("  missing:", bad_missing[:20])
        print("  unexpected:", bad_unexpected[:20])
        # For D1c, missing aux branch is fatal because attention would be random.
        aux_missing = [k for k in bad_missing if "d1b_" in k]
        if aux_missing:
            raise RuntimeError(f"D1b aux keys missing from checkpoint; cannot audit trained attention: {aux_missing[:20]}")

    model.eval()
    return model, device, cfg, ckpt_path


def pair_attention_alpha(model, token_out: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Returns attention weights alpha per pair:
      key -> [B,F]
    This mirrors D1b pair attention without calling the aux classifier.
    """
    out = {}
    scale = float(token_out.shape[-1]) ** 0.5
    for key, *_ in model.d1b_pair_id_names:
        q = model.d1b_pair_queries[key].to(device=token_out.device, dtype=token_out.dtype)
        scores = torch.einsum("bfh,h->bf", token_out, q) / scale
        out[key] = torch.softmax(scores, dim=1)
    return out


def collect_attention(model, loader, device: torch.device, feature_count: int):
    rows = []
    # Store dense alpha per pair as float32 [N,F]; N=11720,F=55 -> fine.
    alpha_all = {key: [] for key in PAIR_ORDER}

    offset = 0
    with torch.no_grad():
        for tokens, values, y in loader:
            tokens = tokens.to(device, non_blocking=True)
            values = values.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits, cls_out, token_out = model(tokens, z_values=values, return_pair_tokens=True)
            pred = logits.argmax(dim=1)
            alphas = pair_attention_alpha(model, token_out)

            bsz = int(tokens.shape[0])
            for key in PAIR_ORDER:
                if key not in alphas:
                    continue
                a = alphas[key].detach().cpu().numpy().astype(np.float32)
                alpha_all[key].append(a)

            offset += bsz

    alpha_np = {key: np.concatenate(chunks, axis=0) for key, chunks in alpha_all.items() if chunks}
    for key, arr in alpha_np.items():
        if arr.ndim != 2 or arr.shape[1] != feature_count:
            raise ValueError(f"Alpha shape mismatch for {key}: {arr.shape}, expected [N,{feature_count}]")
    return alpha_np


def entropy_norm(alpha: np.ndarray) -> np.ndarray:
    eps = 1e-12
    ent = -(alpha * np.log(alpha + eps)).sum(axis=1)
    return ent / math.log(alpha.shape[1])


def top_features_from_mean(mean_alpha: np.ndarray, feature_names: List[str], topk: int) -> List[dict]:
    idx = np.argsort(-mean_alpha)[:topk]
    return [
        {"rank": int(r + 1), "feature_index": int(i), "feature": feature_names[int(i)], "mean_attention": float(mean_alpha[int(i)])}
        for r, i in enumerate(idx)
    ]


def make_top_feature_rows(
    *,
    alpha_np: Dict[str, np.ndarray],
    transition_df: pd.DataFrame,
    feature_names: List[str],
    topk: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      top_by_pair
      top_by_transition
      direction_summary
      changed_sample_topk
    """
    n = len(transition_df)
    sample_index = transition_df["sample_index"].to_numpy(dtype=int)

    top_pair_rows = []
    trans_rows = []
    direction_rows = []
    sample_rows = []

    transitions = ["fixed", "damaged", "both_correct", "both_wrong"]
    important_dirs = [
        "Ransomware->Spyware",
        "Spyware->Ransomware",
        "Ransomware->Trojan",
        "Trojan->Ransomware",
        "Spyware->Trojan",
        "Trojan->Spyware",
    ]

    for key, alpha in alpha_np.items():
        if alpha.shape[0] != n:
            raise ValueError(f"alpha rows != predictions rows for {key}: {alpha.shape[0]} vs {n}")

        ent = entropy_norm(alpha)
        mean_all = alpha.mean(axis=0)
        for item in top_features_from_mean(mean_all, feature_names, topk):
            top_pair_rows.append({
                "pair_key": key,
                "pair_short": PAIR_SHORT.get(key, key),
                "group": "all_val",
                "n": int(n),
                "mean_entropy_norm": float(ent.mean()),
                **item,
            })

        for tr in transitions:
            mask = (transition_df["transition"].to_numpy() == tr)
            cnt = int(mask.sum())
            if cnt == 0:
                continue
            mean_alpha = alpha[mask].mean(axis=0)
            mean_ent = float(ent[mask].mean())
            for item in top_features_from_mean(mean_alpha, feature_names, topk):
                trans_rows.append({
                    "pair_key": key,
                    "pair_short": PAIR_SHORT.get(key, key),
                    "transition": tr,
                    "n": cnt,
                    "mean_entropy_norm": mean_ent,
                    **item,
                })

        # Fixed: use baseline error direction.
        fixed_mask_all = transition_df["transition"].to_numpy() == "fixed"
        damaged_mask_all = transition_df["transition"].to_numpy() == "damaged"
        d1b_wrong_mask_all = (~transition_df["d1b_correct"].to_numpy())

        for source_name, dir_col, base_mask in [
            ("fixed_baseline_error_direction", "base_direction", fixed_mask_all),
            ("damaged_d1b_error_direction", "d1b_direction", damaged_mask_all),
            ("all_d1b_error_direction", "d1b_direction", d1b_wrong_mask_all),
        ]:
            for direction in important_dirs:
                mask = base_mask & (transition_df[dir_col].to_numpy().astype(str) == direction)
                cnt = int(mask.sum())
                if cnt == 0:
                    continue
                mean_alpha = alpha[mask].mean(axis=0)
                top_list = top_features_from_mean(mean_alpha, feature_names, min(topk, 10))
                direction_rows.append({
                    "pair_key": key,
                    "pair_short": PAIR_SHORT.get(key, key),
                    "source": source_name,
                    "direction": direction,
                    "n": cnt,
                    "mean_entropy_norm": float(ent[mask].mean()),
                    "top_features_json": json.dumps(top_list, ensure_ascii=False),
                })

        # Per-sample topk only for changed/fixed/damaged samples to keep file manageable.
        changed_mask = transition_df["pred_changed"].to_numpy() | transition_df["transition"].isin(["fixed", "damaged"]).to_numpy()
        changed_indices = np.where(changed_mask)[0]
        for row_i in changed_indices:
            a = alpha[row_i]
            idx = np.argsort(-a)[: min(5, topk)]
            for rank, feat_i in enumerate(idx, start=1):
                sample_rows.append({
                    "sample_index": int(sample_index[row_i]),
                    "pair_key": key,
                    "pair_short": PAIR_SHORT.get(key, key),
                    "transition": str(transition_df.iloc[row_i]["transition"]),
                    "true_label": str(transition_df.iloc[row_i]["true_label"]),
                    "base_pred_label": str(transition_df.iloc[row_i]["base_pred_label"]),
                    "d1b_pred_label": str(transition_df.iloc[row_i]["d1b_pred_label"]),
                    "rank": int(rank),
                    "feature_index": int(feat_i),
                    "feature": feature_names[int(feat_i)],
                    "attention": float(a[int(feat_i)]),
                    "entropy_norm": float(ent[row_i]),
                })

    return (
        pd.DataFrame(top_pair_rows),
        pd.DataFrame(trans_rows),
        pd.DataFrame(direction_rows),
        pd.DataFrame(sample_rows),
    )


def make_transition_summary(df: pd.DataFrame) -> dict:
    counts = df["transition"].value_counts().to_dict()
    wrong_to_correct = int(counts.get("fixed", 0))
    correct_to_wrong = int(counts.get("damaged", 0))
    return {
        "n": int(len(df)),
        "baseline_correct": int(df["base_correct"].sum()),
        "d1b_correct": int(df["d1b_correct"].sum()),
        "wrong_to_correct": wrong_to_correct,
        "correct_to_wrong": correct_to_wrong,
        "net_gain": int(wrong_to_correct - correct_to_wrong),
        "damage_ratio": float(correct_to_wrong / wrong_to_correct) if wrong_to_correct else None,
        "changed_pred_n": int(df["pred_changed"].sum()),
        "transition_counts": {str(k): int(v) for k, v in counts.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="D1c audit for D1b pair-specific attention")
    parser.add_argument("--d1b-run-dir", default="05_test/outputs/D1b_official_fork_lam0p01")
    parser.add_argument("--baseline-pred", default="03_outputs/06_model/val_predictions_best.csv")
    parser.add_argument("--dataset-npz", default="03_outputs/05_dataset/dataset.npz")
    parser.add_argument("--metadata-json", default="03_outputs/05_dataset/metadata.json")
    parser.add_argument("--train-raw", default="01_split/train_raw.csv")
    parser.add_argument("--val-raw", default="01_split/val_raw.csv")
    parser.add_argument("--d1b-script", default="05_test/D1b_train_official_fork.py")
    parser.add_argument("--out-dir", default="05_test/outputs/D1c_audit_D1b_lam0p01")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--topk", type=int, default=15)
    args = parser.parse_args()

    repo_root = repo_root_from_here()

    d1b_run_dir = resolve_path(args.d1b_run_dir, repo_root)
    d1b_pred = d1b_run_dir / "val_predictions_best.csv"
    if not d1b_pred.exists():
        raise FileNotFoundError(f"Missing D1b prediction file: {d1b_pred}")

    baseline_pred = resolve_path(args.baseline_pred, repo_root)
    out_dir = resolve_path(args.out_dir, repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    d1b_script = resolve_path(args.d1b_script, repo_root)
    d1b = import_d1b_module(d1b_script)

    transition_df = build_transition_df(baseline_pred, d1b_pred)
    transition_summary = make_transition_summary(transition_df)

    val_loader, y_val, label_names, feature_names, meta, continuous_info = make_val_loader(d1b, args, repo_root)
    model, device, d1b_config, ckpt_path = build_model_and_load(
        d1b=d1b,
        args=args,
        repo_root=repo_root,
        label_names=label_names,
        n_features=len(feature_names),
    )

    print(f"[D1c] repo_root={repo_root}")
    print(f"[D1c] device={device}")
    print(f"[D1c] checkpoint={ckpt_path}")
    print(f"[D1c] val_n={len(transition_df)} features={len(feature_names)}")
    print(f"[D1c] transition_summary={transition_summary}")

    alpha_np = collect_attention(model, val_loader, device, feature_count=len(feature_names))

    top_pair_df, trans_df, direction_df, sample_df = make_top_feature_rows(
        alpha_np=alpha_np,
        transition_df=transition_df,
        feature_names=feature_names,
        topk=int(args.topk),
    )

    # Save outputs.
    transition_df.to_csv(out_dir / "d1c_transition_per_sample.csv", index=False)
    top_pair_df.to_csv(out_dir / "d1c_top_features_by_pair.csv", index=False)
    trans_df.to_csv(out_dir / "d1c_top_features_by_transition.csv", index=False)
    direction_df.to_csv(out_dir / "d1c_direction_attention_summary.csv", index=False)
    sample_df.to_csv(out_dir / "d1c_changed_samples_top_features.csv", index=False)

    # Additional compact stats by pair/transition.
    pair_stats_rows = []
    for key, alpha in alpha_np.items():
        ent = entropy_norm(alpha)
        for tr in ["fixed", "damaged", "both_correct", "both_wrong"]:
            mask = transition_df["transition"].to_numpy() == tr
            if int(mask.sum()) == 0:
                continue
            max_att = alpha[mask].max(axis=1)
            pair_stats_rows.append({
                "pair_key": key,
                "pair_short": PAIR_SHORT.get(key, key),
                "transition": tr,
                "n": int(mask.sum()),
                "mean_entropy_norm": float(ent[mask].mean()),
                "mean_max_attention": float(max_att.mean()),
                "median_max_attention": float(np.median(max_att)),
            })
    pair_stats_df = pd.DataFrame(pair_stats_rows)
    pair_stats_df.to_csv(out_dir / "d1c_pair_attention_stats.csv", index=False)

    summary = {
        "stage": "D1c_pair_attention_audit",
        "purpose": "Audit D1b pair-specific attention branch; no training.",
        "inputs": {
            "d1b_run_dir": str(d1b_run_dir),
            "checkpoint": str(ckpt_path),
            "baseline_pred": str(baseline_pred),
            "d1b_pred": str(d1b_pred),
            "dataset_npz": str(resolve_path(args.dataset_npz, repo_root)),
            "metadata_json": str(resolve_path(args.metadata_json, repo_root)),
            "train_raw": str(resolve_path(args.train_raw, repo_root)),
            "val_raw": str(resolve_path(args.val_raw, repo_root)),
            "d1b_script": str(d1b_script),
        },
        "label_names": label_names,
        "n_features": int(len(feature_names)),
        "feature_names": feature_names,
        "transition_summary": transition_summary,
        "d1b_config_d1b": d1b_config.get("d1b", {}),
        "continuous_info": continuous_info,
        "outputs": {
            "transition_per_sample": str(out_dir / "d1c_transition_per_sample.csv"),
            "top_features_by_pair": str(out_dir / "d1c_top_features_by_pair.csv"),
            "top_features_by_transition": str(out_dir / "d1c_top_features_by_transition.csv"),
            "direction_attention_summary": str(out_dir / "d1c_direction_attention_summary.csv"),
            "changed_samples_top_features": str(out_dir / "d1c_changed_samples_top_features.csv"),
            "pair_attention_stats": str(out_dir / "d1c_pair_attention_stats.csv"),
        },
        "interpretation_hint": {
            "good_sign": "fixed samples and damaged samples attend to clearly different features/patterns; useful for selective decision rule design.",
            "bad_sign": "fixed and damaged samples attend to nearly the same features; pair branch is not clean enough and mostly shifts boundary.",
        },
    }
    save_json(out_dir / "d1c_summary.json", summary)

    print("[D1c] wrote:")
    for k, v in summary["outputs"].items():
        print(f"  - {k}: {v}")
    print(f"  - summary: {out_dir / 'd1c_summary.json'}")


if __name__ == "__main__":
    main()
