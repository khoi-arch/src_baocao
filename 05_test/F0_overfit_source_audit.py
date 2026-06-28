#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F0 Overfit Source Audit for CIC-MalMem D3.

Purpose
-------
Stop adding heads blindly. Diagnose where the train->val gap comes from.

Questions answered:
1. How large is train-vs-val generalization gap for official D3?
2. Is tokenization/K creating sparse/rare/unseen token shortcuts?
3. Do token-label rules work on train but fail on val?
4. Do raw features drift train-vs-val, globally and within class?
5. Are val errors correlated with rare/unseen tokens?
6. Which branch should be tested next:
   - K/tokenization
   - raw feature drift
   - fusion/model capacity
   - class split/subtype distribution shift

No training.
No tree.
No solution test.
Only audit.

Outputs:
  F0_summary.json
  F0_summary.md
  F0_train_val_metrics.json
  F0_per_class_gap.csv
  F0_token_sparsity_audit.csv
  F0_token_shortcut_audit.csv
  F0_raw_feature_drift.csv
  F0_class_conditional_raw_drift.csv
  F0_raw_quantile_shortcut_audit.csv
  F0_val_sample_risk.csv
  F0_error_risk_summary.csv
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import math
import random
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


LABEL_COLS = {"Class", "Category", "label_L1", "label_L2", "label_L3", "Label", "label", "target"}


def strip_label(x: Any) -> str:
    return str(x).strip()


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


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available; falling back to CPU.", flush=True)
        return torch.device("cpu")
    return torch.device(device_arg)


def import_official_train(train_script: Path):
    if not train_script.exists():
        raise FileNotFoundError(f"official 07_train.py not found: {train_script}")
    src_dir = train_script.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    spec = importlib.util.spec_from_file_location("official_07_train_for_f0", str(train_script))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import official train script: {train_script}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def normalize_label_mapping(meta: dict) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    label_mapping = meta.get("label_mapping")
    if not isinstance(label_mapping, dict):
        raise ValueError("metadata.json missing label_mapping dict")
    pairs = sorted([(strip_label(label), int(idx)) for label, idx in label_mapping.items()], key=lambda x: x[1])
    label_names = [p[0] for p in pairs]
    label_to_id = {label: idx for label, idx in pairs}
    id_to_label = {idx: label for label, idx in pairs}
    return label_names, label_to_id, id_to_label


def load_official_inputs(args, repo_root: Path) -> dict:
    train_mod = import_official_train(resolve_path(args.official_train, repo_root))
    dataset_npz = resolve_path(args.dataset_npz, repo_root)
    metadata_json = resolve_path(args.metadata_json, repo_root)

    data, meta = train_mod.load_dataset(dataset_npz, metadata_json)
    label_names, label_to_id, id_to_label = normalize_label_mapping(meta)
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
    X_train_cont, X_val_cont, continuous_info = train_mod.load_continuous_for_run(
        spec=spec,
        meta=meta,
        args=raw_args,
        train_shape=X_train_bin.shape,
        val_shape=X_val_bin.shape,
    )

    M_train = np.ones_like(X_train_bin, dtype=np.float32)
    M_val = np.ones_like(X_val_bin, dtype=np.float32)

    train_ds = train_mod.FusionAblationDataset(
        X_train_bin, X_train_offset, X_train_cont.astype(np.float32), M_train, y_train
    )
    val_ds = train_mod.FusionAblationDataset(
        X_val_bin, X_val_offset, X_val_cont.astype(np.float32), M_val, y_val
    )

    num_bins = int(meta.get("num_bins", 0) or meta.get("K", 0) or (max(int(X_train_bin.max()), int(X_val_bin.max())) + 1))

    return {
        "train_mod": train_mod,
        "meta": meta,
        "feature_names": feature_names,
        "label_names": label_names,
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "num_bins": num_bins,
        "continuous_info": continuous_info,
        "train_ds": train_ds,
        "val_ds": val_ds,
        "X_train_bin": X_train_bin,
        "X_val_bin": X_val_bin,
        "X_train_offset": X_train_offset,
        "X_val_offset": X_val_offset,
        "X_train_cont": X_train_cont.astype(np.float32),
        "X_val_cont": X_val_cont.astype(np.float32),
        "y_train": y_train,
        "y_val": y_val,
    }


def get_baseline_model_cfg(args, repo_root: Path) -> dict:
    cfg = {}
    path = resolve_path(args.baseline_config, repo_root)
    if path.exists():
        loaded = load_json(path)
        if isinstance(loaded.get("model"), dict):
            cfg.update(loaded["model"])
        if isinstance(loaded.get("model_config"), dict):
            cfg.update({k: v for k, v in loaded["model_config"].items() if k not in cfg})
        for k in [
            "value_dim", "feature_dim", "hidden_dim", "num_layers", "num_heads",
            "dropout", "classifier_hidden_dim", "classifier_dropout", "norm_first",
            "gate_init", "activation", "num_bins", "effective_token_budget",
        ]:
            if k in loaded and k not in cfg:
                cfg[k] = loaded[k]

    defaults = {
        "value_dim": 32,
        "feature_dim": 32,
        "hidden_dim": 128,
        "num_layers": 3,
        "num_heads": 4,
        "dropout": 0.1,
        "classifier_hidden_dim": 128,
        "classifier_dropout": 0.1,
        "norm_first": True,
        "gate_init": 0.0,
        "activation": "gelu",
    }
    defaults.update(cfg)
    return defaults


def build_official_d3_model(train_mod, model_cfg: dict, n_features: int, num_bins: int, num_classes: int, device: torch.device):
    cls = train_mod.FusionAblationTransformer
    sig = inspect.signature(cls)
    kwargs = {
        "run_id": "D3",
        "num_bins": int(num_bins),
        "n_features": int(n_features),
        "num_classes": int(num_classes),
        "value_dim": int(model_cfg.get("value_dim", 32)),
        "feature_dim": int(model_cfg.get("feature_dim", 32)),
        "hidden_dim": int(model_cfg.get("hidden_dim", 128)),
        "num_layers": int(model_cfg.get("num_layers", 3)),
        "num_heads": int(model_cfg.get("num_heads", 4)),
        "dropout": float(model_cfg.get("dropout", 0.1)),
        "classifier_hidden_dim": int(model_cfg.get("classifier_hidden_dim", 128)),
        "classifier_dropout": float(model_cfg.get("classifier_dropout", 0.1)),
        "norm_first": bool(model_cfg.get("norm_first", True)),
        "gate_init": float(model_cfg.get("gate_init", 0.0)),
        "activation": str(model_cfg.get("activation", "gelu")),
    }
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    model = cls(**filtered).to(device)
    return model, filtered


def safe_torch_load_checkpoint(ckpt_path: Path, device: torch.device):
    try:
        return torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(ckpt_path, map_location=device)


def extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for key in ["model_state_dict", "state_dict", "model"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        if all(isinstance(k, str) for k in ckpt.keys()) and any(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt
    raise ValueError("Cannot extract model state dict from checkpoint")


def load_checkpoint_into_model(model: nn.Module, ckpt_path: Path, device: torch.device, strict: bool = True) -> dict:
    ckpt = safe_torch_load_checkpoint(ckpt_path, device)
    sd = extract_state_dict(ckpt)
    cleaned = {}
    for k, v in sd.items():
        nk = k
        for prefix in ["module.", "backbone.", "model."]:
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        cleaned[nk] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=strict)
    return {
        "path": str(ckpt_path),
        "strict": bool(strict),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


def model_forward_logits(model: nn.Module, tokens: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    try:
        out = model(tokens, z_values=values)
    except TypeError:
        out = model(tokens, values)
    if isinstance(out, dict):
        if "logits" in out:
            out = out["logits"]
        elif "final_logits" in out:
            out = out["final_logits"]
        else:
            raise ValueError(f"Dict output has no logits key: {list(out.keys())}")
    if isinstance(out, tuple):
        out = out[0]
    return out


def make_loader(ds, batch_size: int, shuffle: bool, seed: int, num_workers: int, device: torch.device):
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=gen if shuffle else None,
        num_workers=int(num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )


@torch.no_grad()
def predict_model(model, loader, device):
    model.eval()
    ys, preds, probs_all, logits_all = [], [], [], []
    for tokens, values, y in loader:
        tokens = tokens.to(device, non_blocking=True)
        values = values.to(device, non_blocking=True)
        logits = model_forward_logits(model, tokens, values)
        probs = torch.softmax(logits, dim=1)
        ys.append(y.cpu().numpy().astype(int))
        preds.append(logits.argmax(dim=1).cpu().numpy().astype(int))
        probs_all.append(probs.cpu().numpy().astype(np.float32))
        logits_all.append(logits.cpu().numpy().astype(np.float32))
    y = np.concatenate(ys)
    pred = np.concatenate(preds)
    probs = np.concatenate(probs_all)
    logits = np.concatenate(logits_all)
    return y, pred, probs, logits


def classification_metrics(y, pred, label_names):
    labels = list(range(len(label_names)))
    p, r, f1, sup = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    per = pd.DataFrame([
        {"class_id": i, "label": label_names[i], "precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i]), "support": int(sup[i])}
        for i in labels
    ])
    overall = {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "weighted_f1": float(f1_score(y, pred, average="weighted")),
    }
    cm = pd.DataFrame(confusion_matrix(y, pred, labels=labels), index=label_names, columns=label_names)
    return overall, per, cm


def top2_metrics(y, logits):
    top2 = np.argsort(-logits, axis=1)[:, :2]
    top1 = top2[:, 0]
    second = top2[:, 1]
    return {
        "top1_acc": float((top1 == y).mean()),
        "top2_acc": float(((top2 == y[:, None]).any(axis=1)).mean()),
        "wrong_true_in_top2_rate": float((((top2 == y[:, None]).any(axis=1)) & (top1 != y)).sum() / max(1, (top1 != y).sum())),
        "wrong_total": int((top1 != y).sum()),
        "wrong_true_in_top2": int((((top2 == y[:, None]).any(axis=1)) & (top1 != y)).sum()),
    }


def ks_statistic(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    x = np.sort(x)
    y = np.sort(y)
    data = np.concatenate([x, y])
    cdf_x = np.searchsorted(x, data, side="right") / len(x)
    cdf_y = np.searchsorted(y, data, side="right") / len(y)
    return float(np.max(np.abs(cdf_x - cdf_y)))


def psi_statistic(train, val, n_bins=10):
    train = np.asarray(train, dtype=float)
    val = np.asarray(val, dtype=float)
    train = train[np.isfinite(train)]
    val = val[np.isfinite(val)]
    if len(train) == 0 or len(val) == 0:
        return np.nan
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(train, qs))
    if len(edges) < 3:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    t_counts, _ = np.histogram(train, bins=edges)
    v_counts, _ = np.histogram(val, bins=edges)
    t = t_counts / max(1, t_counts.sum())
    v = v_counts / max(1, v_counts.sum())
    eps = 1e-6
    t = np.clip(t, eps, None)
    v = np.clip(v, eps, None)
    return float(np.sum((v - t) * np.log(v / t)))


def corr_safe(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan
    if np.std(a[mask]) < 1e-12 or np.std(b[mask]) < 1e-12:
        return np.nan
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def load_raw_feature_frames(args, repo_root, feature_names):
    train_raw = pd.read_csv(resolve_path(args.train_raw, repo_root))
    val_raw = pd.read_csv(resolve_path(args.val_raw, repo_root))
    common = [c for c in feature_names if c in train_raw.columns and c in val_raw.columns]
    if not common:
        # fallback numeric non-label columns
        common = [c for c in train_raw.columns if c in val_raw.columns and c not in LABEL_COLS and pd.api.types.is_numeric_dtype(train_raw[c])]
    train = train_raw[common].apply(pd.to_numeric, errors="coerce")
    val = val_raw[common].apply(pd.to_numeric, errors="coerce")
    return train, val, common


def raw_drift_audit(train_df, val_df, y_train, y_val, label_names, max_features_class_drift=999999):
    rows = []
    for c in train_df.columns:
        tr = train_df[c].to_numpy(dtype=float)
        va = val_df[c].to_numpy(dtype=float)
        tr_f = tr[np.isfinite(tr)]
        va_f = va[np.isfinite(va)]
        std = np.nanstd(tr_f) if len(tr_f) else np.nan
        row = {
            "feature": c,
            "train_mean": float(np.nanmean(tr_f)) if len(tr_f) else np.nan,
            "val_mean": float(np.nanmean(va_f)) if len(va_f) else np.nan,
            "train_std": float(std) if np.isfinite(std) else np.nan,
            "val_std": float(np.nanstd(va_f)) if len(va_f) else np.nan,
            "mean_shift_abs": float(abs(np.nanmean(va_f) - np.nanmean(tr_f))) if len(tr_f) and len(va_f) else np.nan,
            "mean_shift_std_units": float(abs(np.nanmean(va_f) - np.nanmean(tr_f)) / (std + 1e-12)) if len(tr_f) and len(va_f) and np.isfinite(std) else np.nan,
            "ks": ks_statistic(tr, va),
            "psi": psi_statistic(tr, va, n_bins=10),
            "train_q05": float(np.nanquantile(tr_f, 0.05)) if len(tr_f) else np.nan,
            "val_q05": float(np.nanquantile(va_f, 0.05)) if len(va_f) else np.nan,
            "train_q50": float(np.nanquantile(tr_f, 0.50)) if len(tr_f) else np.nan,
            "val_q50": float(np.nanquantile(va_f, 0.50)) if len(va_f) else np.nan,
            "train_q95": float(np.nanquantile(tr_f, 0.95)) if len(tr_f) else np.nan,
            "val_q95": float(np.nanquantile(va_f, 0.95)) if len(va_f) else np.nan,
        }
        rows.append(row)
    drift = pd.DataFrame(rows).sort_values(["psi", "ks"], ascending=False)

    class_rows = []
    for c in train_df.columns[:max_features_class_drift]:
        tr_all = train_df[c].to_numpy(dtype=float)
        va_all = val_df[c].to_numpy(dtype=float)
        for cid, lname in enumerate(label_names):
            tr = tr_all[y_train == cid]
            va = va_all[y_val == cid]
            if len(tr) == 0 or len(va) == 0:
                continue
            std = np.nanstd(tr)
            class_rows.append({
                "feature": c,
                "class_id": int(cid),
                "label": lname,
                "n_train": int(len(tr)),
                "n_val": int(len(va)),
                "train_mean": float(np.nanmean(tr)),
                "val_mean": float(np.nanmean(va)),
                "mean_shift_std_units": float(abs(np.nanmean(va)-np.nanmean(tr))/(std+1e-12)),
                "ks": ks_statistic(tr, va),
                "psi": psi_statistic(tr, va),
            })
    class_drift = pd.DataFrame(class_rows)
    if len(class_drift):
        class_drift = class_drift.sort_values(["psi", "ks"], ascending=False)
    return drift, class_drift


def majority_rule_acc_from_bins(train_bins, val_bins, y_train, y_val, num_classes, min_support=1):
    max_token = int(max(np.max(train_bins), np.max(val_bins))) if len(train_bins) and len(val_bins) else 0
    counts = np.zeros((max_token + 1, num_classes), dtype=np.int64)
    np.add.at(counts, (train_bins, y_train), 1)
    support = counts.sum(axis=1)
    maj = counts.argmax(axis=1)
    purity = np.divide(counts.max(axis=1), np.maximum(1, support), dtype=float)
    seen_train = support[train_bins] >= min_support
    seen_val = support[val_bins] >= min_support
    train_pred = maj[train_bins]
    val_pred = maj[np.clip(val_bins, 0, max_token)]
    train_acc = float((train_pred[seen_train] == y_train[seen_train]).mean()) if seen_train.any() else np.nan
    val_acc = float((val_pred[seen_val] == y_val[seen_val]).mean()) if seen_val.any() else np.nan
    val_coverage = float(seen_val.mean())
    train_weighted_purity = float(np.average(purity[train_bins], weights=np.ones_like(train_bins))) if len(train_bins) else np.nan
    val_seen_weighted_purity = float(np.average(purity[np.clip(val_bins[seen_val], 0, max_token)], weights=np.ones(seen_val.sum()))) if seen_val.any() else np.nan
    return train_acc, val_acc, val_coverage, train_weighted_purity, val_seen_weighted_purity


def token_audit(X_train_bin, X_val_bin, y_train, y_val, label_names, pred_val=None, rare_thresholds=(1,2,5,10)):
    n_train, n_features = X_train_bin.shape
    n_val = X_val_bin.shape[0]
    num_classes = len(label_names)
    sparsity_rows = []
    shortcut_rows = []

    val_wrong = None
    if pred_val is not None:
        val_wrong = pred_val != y_val

    sample_unseen_counts = np.zeros(n_val, dtype=np.float32)
    sample_rare5_counts = np.zeros(n_val, dtype=np.float32)
    sample_mean_train_count = np.zeros(n_val, dtype=np.float32)

    for j in range(n_features):
        tr = X_train_bin[:, j].astype(int)
        va = X_val_bin[:, j].astype(int)
        max_tok = int(max(tr.max(), va.max()))
        train_counts = np.bincount(tr, minlength=max_tok+1)
        val_counts = np.bincount(va, minlength=max_tok+1)
        val_train_count = train_counts[va]

        unseen = val_train_count == 0
        rare = {thr: val_train_count <= thr for thr in rare_thresholds}

        sample_unseen_counts += unseen.astype(np.float32)
        sample_rare5_counts += rare[5].astype(np.float32)
        sample_mean_train_count += val_train_count.astype(np.float32)

        row = {
            "feature_idx": int(j),
            "n_unique_train": int((train_counts > 0).sum()),
            "n_unique_val": int((val_counts > 0).sum()),
            "n_unseen_val_tokens": int(((val_counts > 0) & (train_counts == 0)).sum()),
            "val_unseen_sample_rate": float(unseen.mean()),
            "val_mean_train_token_count": float(val_train_count.mean()),
            "val_median_train_token_count": float(np.median(val_train_count)),
            "train_singleton_token_count": int((train_counts == 1).sum()),
            "train_rare_le5_token_count": int(((train_counts > 0) & (train_counts <= 5)).sum()),
        }
        for thr in rare_thresholds:
            row[f"val_rare_le{thr}_sample_rate"] = float(rare[thr].mean())
        if val_wrong is not None:
            row["wrong_val_unseen_rate"] = float(unseen[val_wrong].mean()) if val_wrong.any() else np.nan
            row["correct_val_unseen_rate"] = float(unseen[~val_wrong].mean()) if (~val_wrong).any() else np.nan
            row["wrong_minus_correct_unseen"] = row["wrong_val_unseen_rate"] - row["correct_val_unseen_rate"] if np.isfinite(row["wrong_val_unseen_rate"]) and np.isfinite(row["correct_val_unseen_rate"]) else np.nan
            row["wrong_val_rare_le5_rate"] = float(rare[5][val_wrong].mean()) if val_wrong.any() else np.nan
            row["correct_val_rare_le5_rate"] = float(rare[5][~val_wrong].mean()) if (~val_wrong).any() else np.nan
            row["wrong_minus_correct_rare_le5"] = row["wrong_val_rare_le5_rate"] - row["correct_val_rare_le5_rate"] if np.isfinite(row["wrong_val_rare_le5_rate"]) and np.isfinite(row["correct_val_rare_le5_rate"]) else np.nan
        sparsity_rows.append(row)

        train_acc, val_acc, val_cov, tr_purity, va_purity = majority_rule_acc_from_bins(
            tr, va, y_train, y_val, num_classes, min_support=1
        )
        shortcut_rows.append({
            "feature_idx": int(j),
            "train_token_majority_acc": train_acc,
            "val_token_majority_acc_seen": val_acc,
            "val_seen_coverage": val_cov,
            "train_minus_val_token_acc": float(train_acc - val_acc) if np.isfinite(train_acc) and np.isfinite(val_acc) else np.nan,
            "train_weighted_token_purity": tr_purity,
            "val_seen_weighted_train_purity": va_purity,
            "purity_transfer_gap": float(tr_purity - va_purity) if np.isfinite(tr_purity) and np.isfinite(va_purity) else np.nan,
        })

    sample_unseen_frac = sample_unseen_counts / n_features
    sample_rare5_frac = sample_rare5_counts / n_features
    sample_mean_train_count = sample_mean_train_count / n_features

    sparsity = pd.DataFrame(sparsity_rows)
    shortcut = pd.DataFrame(shortcut_rows)
    return sparsity, shortcut, sample_unseen_frac, sample_rare5_frac, sample_mean_train_count


def raw_quantile_shortcut_audit(train_df, val_df, y_train, y_val, num_classes, n_bins=20):
    rows = []
    for c in train_df.columns:
        tr = train_df[c].to_numpy(dtype=float)
        va = val_df[c].to_numpy(dtype=float)
        finite = np.isfinite(tr)
        if finite.sum() < 10:
            continue
        edges = np.unique(np.nanquantile(tr[finite], np.linspace(0, 1, n_bins+1)))
        if len(edges) < 3:
            continue
        edges[0] = -np.inf
        edges[-1] = np.inf
        tr_bins = np.digitize(tr, edges[1:-1], right=False)
        va_bins = np.digitize(va, edges[1:-1], right=False)
        train_acc, val_acc, val_cov, tr_purity, va_purity = majority_rule_acc_from_bins(
            tr_bins.astype(int), va_bins.astype(int), y_train, y_val, num_classes, min_support=1
        )
        rows.append({
            "feature": c,
            "n_bins": int(len(edges)-1),
            "train_rawbin_majority_acc": train_acc,
            "val_rawbin_majority_acc": val_acc,
            "train_minus_val_rawbin_acc": float(train_acc-val_acc) if np.isfinite(train_acc) and np.isfinite(val_acc) else np.nan,
            "train_weighted_rawbin_purity": tr_purity,
            "val_seen_weighted_train_rawbin_purity": va_purity,
            "rawbin_purity_transfer_gap": float(tr_purity-va_purity) if np.isfinite(tr_purity) and np.isfinite(va_purity) else np.nan,
        })
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["train_minus_val_rawbin_acc", "rawbin_purity_transfer_gap"], ascending=False)
    return df


def write_summary_md(out_dir: Path, summary: dict):
    text = f"""# F0 Overfit Source Audit

## Core result

```text
train macro-F1 = {summary['metrics']['train']['overall']['macro_f1']:.6f}
val macro-F1   = {summary['metrics']['val']['overall']['macro_f1']:.6f}
gap            = {summary['metrics']['gap']['macro_f1']:.6f}

train acc      = {summary['metrics']['train']['overall']['accuracy']:.6f}
val acc        = {summary['metrics']['val']['overall']['accuracy']:.6f}
acc gap        = {summary['metrics']['gap']['accuracy']:.6f}
```

## Top-2 headroom

```text
train top2 acc = {summary['top2']['train']['top2_acc']:.6f}
val top2 acc   = {summary['top2']['val']['top2_acc']:.6f}

val wrong total        = {summary['top2']['val']['wrong_total']}
val wrong true in top2 = {summary['top2']['val']['wrong_true_in_top2']}
```

## Token/K risk signals

```text
mean val unseen token sample rate      = {summary['token_risk']['mean_val_unseen_sample_rate']:.6f}
mean val rare<=5 token sample rate     = {summary['token_risk']['mean_val_rare_le5_sample_rate']:.6f}
mean token shortcut train-val acc gap  = {summary['token_risk']['mean_token_shortcut_gap']:.6f}
max token shortcut train-val acc gap   = {summary['token_risk']['max_token_shortcut_gap']:.6f}
corr(val wrong, rare<=5 frac)          = {summary['error_risk']['corr_wrong_rare_le5_frac']}
corr(val wrong, unseen frac)           = {summary['error_risk']['corr_wrong_unseen_frac']}
```

## Raw feature drift risk signals

```text
max raw PSI = {summary['raw_drift_risk']['max_psi']:.6f}
max raw KS  = {summary['raw_drift_risk']['max_ks']:.6f}
mean raw PSI = {summary['raw_drift_risk']['mean_psi']:.6f}
```

## Preliminary diagnosis

{summary['diagnosis_text']}

## Key files

- `F0_train_val_metrics.json`
- `F0_per_class_gap.csv`
- `F0_token_sparsity_audit.csv`
- `F0_token_shortcut_audit.csv`
- `F0_raw_feature_drift.csv`
- `F0_class_conditional_raw_drift.csv`
- `F0_raw_quantile_shortcut_audit.csv`
- `F0_val_sample_risk.csv`
- `F0_error_risk_summary.csv`

## How to use this

Do not jump to a fix from one number.

Use this order:

```text
1. If token shortcut gap / rare-token error correlation is high:
   test K/rare-bin/token dropout.

2. If raw drift and class-conditional drift are high:
   test stability-aware feature filtering/dropout.

3. If all branch inputs show gap later:
   test capacity/regularization.

4. Before any solution test:
   compare with previous E1-E5 experiments to avoid duplicates.
```
"""
    (out_dir / "F0_summary.md").write_text(text, encoding="utf-8")


def zip_dir(src_dir: Path, zip_path: Path):
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(src_dir.rglob("*")):
            if p.is_file() and p != zip_path:
                z.write(p, p.relative_to(src_dir.parent))


def main():
    parser = argparse.ArgumentParser(description="F0 overfit source audit for official D3")
    parser.add_argument("--dataset-npz", default="03_outputs/05_dataset/dataset.npz")
    parser.add_argument("--metadata-json", default="03_outputs/05_dataset/metadata.json")
    parser.add_argument("--train-raw", default="01_split/train_raw.csv")
    parser.add_argument("--val-raw", default="01_split/val_raw.csv")
    parser.add_argument("--official-train", default="02_src/07_train.py")
    parser.add_argument("--baseline-config", default="03_outputs/06_model/config.json")
    parser.add_argument("--baseline-checkpoint", default="03_outputs/06_model/best_model.pt")
    parser.add_argument("--out-dir", default="05_test/outputs/F0_overfit_source_audit")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--make-zip", action="store_true", default=True)
    parser.add_argument("--no-zip", dest="make_zip", action="store_false")
    args = parser.parse_args()

    repo_root = repo_root_from_here()
    out_dir = resolve_path(args.out_dir, repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = pick_device(args.device)

    print(f"[F0] repo_root={repo_root}", flush=True)
    print(f"[F0] out_dir={out_dir}", flush=True)
    print(f"[F0] device={device}", flush=True)
    print("[F0] audit only. no training. no tree.", flush=True)

    inp = load_official_inputs(args, repo_root)
    train_loader = make_loader(inp["train_ds"], args.batch_size, False, args.seed, args.num_workers, device)
    val_loader = make_loader(inp["val_ds"], args.batch_size, False, args.seed, args.num_workers, device)

    model_cfg = get_baseline_model_cfg(args, repo_root)
    model, model_kwargs = build_official_d3_model(
        inp["train_mod"], model_cfg, len(inp["feature_names"]), inp["num_bins"], len(inp["label_names"]), device
    )
    ckpt_info = load_checkpoint_into_model(model, resolve_path(args.baseline_checkpoint, repo_root), device, strict=True)
    print(f"[F0] checkpoint loaded: {ckpt_info}", flush=True)

    print("[F0] predicting train...", flush=True)
    ytr, pred_tr, probs_tr, logits_tr = predict_model(model, train_loader, device)
    print("[F0] predicting val...", flush=True)
    yva, pred_va, probs_va, logits_va = predict_model(model, val_loader, device)

    train_overall, train_per, train_cm = classification_metrics(ytr, pred_tr, inp["label_names"])
    val_overall, val_per, val_cm = classification_metrics(yva, pred_va, inp["label_names"])
    per_gap = train_per.merge(val_per, on=["class_id", "label"], suffixes=("_train", "_val"))
    for m in ["precision", "recall", "f1"]:
        per_gap[f"{m}_gap_train_minus_val"] = per_gap[f"{m}_train"] - per_gap[f"{m}_val"]

    train_per.to_csv(out_dir / "F0_train_per_class.csv", index=False)
    val_per.to_csv(out_dir / "F0_val_per_class.csv", index=False)
    per_gap.to_csv(out_dir / "F0_per_class_gap.csv", index=False)
    train_cm.to_csv(out_dir / "F0_train_confusion_matrix.csv")
    val_cm.to_csv(out_dir / "F0_val_confusion_matrix.csv")

    metrics = {
        "train": {"overall": train_overall},
        "val": {"overall": val_overall},
        "gap": {
            "accuracy": float(train_overall["accuracy"] - val_overall["accuracy"]),
            "macro_f1": float(train_overall["macro_f1"] - val_overall["macro_f1"]),
            "weighted_f1": float(train_overall["weighted_f1"] - val_overall["weighted_f1"]),
        },
    }
    save_json(out_dir / "F0_train_val_metrics.json", metrics)

    top2 = {
        "train": top2_metrics(ytr, logits_tr),
        "val": top2_metrics(yva, logits_va),
    }

    print("[F0] token audit...", flush=True)
    token_sparsity, token_shortcut, unseen_frac, rare5_frac, mean_train_count = token_audit(
        inp["X_train_bin"], inp["X_val_bin"], inp["y_train"], inp["y_val"], inp["label_names"], pred_val=pred_va
    )
    token_sparsity["feature"] = [inp["feature_names"][i] for i in token_sparsity["feature_idx"]]
    token_shortcut["feature"] = [inp["feature_names"][i] for i in token_shortcut["feature_idx"]]
    token_sparsity = token_sparsity.sort_values(
        ["wrong_minus_correct_rare_le5", "val_rare_le5_sample_rate", "val_unseen_sample_rate"],
        ascending=False,
    )
    token_shortcut = token_shortcut.sort_values(["train_minus_val_token_acc", "purity_transfer_gap"], ascending=False)
    token_sparsity.to_csv(out_dir / "F0_token_sparsity_audit.csv", index=False)
    token_shortcut.to_csv(out_dir / "F0_token_shortcut_audit.csv", index=False)

    val_wrong = pred_va != yva
    val_sample = pd.DataFrame({
        "sample_index": np.arange(len(yva)),
        "true_id": yva,
        "true_label": [inp["id_to_label"][int(i)] for i in yva],
        "pred_id": pred_va,
        "pred_label": [inp["id_to_label"][int(i)] for i in pred_va],
        "correct": pred_va == yva,
        "top1_conf": probs_va.max(axis=1),
        "top2_conf": np.sort(probs_va, axis=1)[:, -2],
        "margin_prob": probs_va.max(axis=1) - np.sort(probs_va, axis=1)[:, -2],
        "unseen_token_frac": unseen_frac,
        "rare_le5_token_frac": rare5_frac,
        "mean_train_token_count": mean_train_count,
    })
    val_sample.to_csv(out_dir / "F0_val_sample_risk.csv", index=False)

    error_risk = pd.DataFrame([{
        "metric": "unseen_token_frac",
        "wrong_mean": float(val_sample.loc[~val_sample["correct"], "unseen_token_frac"].mean()),
        "correct_mean": float(val_sample.loc[val_sample["correct"], "unseen_token_frac"].mean()),
        "wrong_minus_correct": float(val_sample.loc[~val_sample["correct"], "unseen_token_frac"].mean() - val_sample.loc[val_sample["correct"], "unseen_token_frac"].mean()),
        "corr_with_wrong": corr_safe((~val_sample["correct"]).astype(int), val_sample["unseen_token_frac"]),
    }, {
        "metric": "rare_le5_token_frac",
        "wrong_mean": float(val_sample.loc[~val_sample["correct"], "rare_le5_token_frac"].mean()),
        "correct_mean": float(val_sample.loc[val_sample["correct"], "rare_le5_token_frac"].mean()),
        "wrong_minus_correct": float(val_sample.loc[~val_sample["correct"], "rare_le5_token_frac"].mean() - val_sample.loc[val_sample["correct"], "rare_le5_token_frac"].mean()),
        "corr_with_wrong": corr_safe((~val_sample["correct"]).astype(int), val_sample["rare_le5_token_frac"]),
    }, {
        "metric": "mean_train_token_count",
        "wrong_mean": float(val_sample.loc[~val_sample["correct"], "mean_train_token_count"].mean()),
        "correct_mean": float(val_sample.loc[val_sample["correct"], "mean_train_token_count"].mean()),
        "wrong_minus_correct": float(val_sample.loc[~val_sample["correct"], "mean_train_token_count"].mean() - val_sample.loc[val_sample["correct"], "mean_train_token_count"].mean()),
        "corr_with_wrong": corr_safe((~val_sample["correct"]).astype(int), val_sample["mean_train_token_count"]),
    }, {
        "metric": "margin_prob",
        "wrong_mean": float(val_sample.loc[~val_sample["correct"], "margin_prob"].mean()),
        "correct_mean": float(val_sample.loc[val_sample["correct"], "margin_prob"].mean()),
        "wrong_minus_correct": float(val_sample.loc[~val_sample["correct"], "margin_prob"].mean() - val_sample.loc[val_sample["correct"], "margin_prob"].mean()),
        "corr_with_wrong": corr_safe((~val_sample["correct"]).astype(int), val_sample["margin_prob"]),
    }])
    error_risk.to_csv(out_dir / "F0_error_risk_summary.csv", index=False)

    print("[F0] raw feature drift audit...", flush=True)
    train_raw_df, val_raw_df, raw_features = load_raw_feature_frames(args, repo_root, inp["feature_names"])
    raw_drift, class_raw_drift = raw_drift_audit(train_raw_df, val_raw_df, inp["y_train"], inp["y_val"], inp["label_names"])
    raw_drift.to_csv(out_dir / "F0_raw_feature_drift.csv", index=False)
    class_raw_drift.to_csv(out_dir / "F0_class_conditional_raw_drift.csv", index=False)

    print("[F0] raw quantile shortcut audit...", flush=True)
    raw_shortcut = raw_quantile_shortcut_audit(train_raw_df, val_raw_df, inp["y_train"], inp["y_val"], len(inp["label_names"]), n_bins=20)
    raw_shortcut.to_csv(out_dir / "F0_raw_quantile_shortcut_audit.csv", index=False)

    # Preliminary rule-based diagnosis. This is intentionally non-final.
    token_risk = {
        "mean_val_unseen_sample_rate": float(token_sparsity["val_unseen_sample_rate"].mean()),
        "mean_val_rare_le5_sample_rate": float(token_sparsity["val_rare_le5_sample_rate"].mean()),
        "mean_token_shortcut_gap": float(token_shortcut["train_minus_val_token_acc"].mean()),
        "max_token_shortcut_gap": float(token_shortcut["train_minus_val_token_acc"].max()),
        "top_token_shortcut_features": token_shortcut.head(10)[["feature", "train_token_majority_acc", "val_token_majority_acc_seen", "train_minus_val_token_acc"]].to_dict(orient="records"),
        "top_token_sparse_error_features": token_sparsity.head(10)[["feature", "val_rare_le5_sample_rate", "wrong_minus_correct_rare_le5", "val_unseen_sample_rate"]].to_dict(orient="records"),
    }
    raw_drift_risk = {
        "max_psi": float(raw_drift["psi"].max()) if len(raw_drift) else np.nan,
        "max_ks": float(raw_drift["ks"].max()) if len(raw_drift) else np.nan,
        "mean_psi": float(raw_drift["psi"].mean()) if len(raw_drift) else np.nan,
        "top_raw_drift_features": raw_drift.head(10)[["feature", "psi", "ks", "mean_shift_std_units"]].to_dict(orient="records") if len(raw_drift) else [],
        "top_class_conditional_drift": class_raw_drift.head(15)[["feature", "label", "psi", "ks", "mean_shift_std_units"]].to_dict(orient="records") if len(class_raw_drift) else [],
    }
    error_risk_summary = {
        "corr_wrong_rare_le5_frac": float(error_risk.loc[error_risk["metric"]=="rare_le5_token_frac", "corr_with_wrong"].iloc[0]),
        "corr_wrong_unseen_frac": float(error_risk.loc[error_risk["metric"]=="unseen_token_frac", "corr_with_wrong"].iloc[0]),
        "corr_wrong_margin_prob": float(error_risk.loc[error_risk["metric"]=="margin_prob", "corr_with_wrong"].iloc[0]),
    }

    diagnosis_lines = []
    gap = metrics["gap"]["macro_f1"]
    if gap > 0.05:
        diagnosis_lines.append(f"- Strong generalization gap detected: train-val macro-F1 gap = {gap:.4f}.")
    if token_risk["max_token_shortcut_gap"] > 0.05:
        diagnosis_lines.append("- Token shortcut risk is non-trivial: at least one feature has high train-vs-val token-majority gap.")
    if abs(error_risk_summary["corr_wrong_rare_le5_frac"]) > 0.05:
        diagnosis_lines.append("- Val errors correlate with rare-token exposure; K/bin sparsity should be tested.")
    if raw_drift_risk["max_psi"] > 0.2 or raw_drift_risk["max_ks"] > 0.1:
        diagnosis_lines.append("- Raw train-vs-val drift is present; inspect top drift and class-conditional drift features.")
    if not diagnosis_lines:
        diagnosis_lines.append("- No single dominant audit signal crossed the coarse thresholds; use the CSV rankings for manual inspection.")
    diagnosis_text = "\n".join(diagnosis_lines)

    summary = {
        "stage": "F0_overfit_source_audit",
        "tree_usage": "none",
        "training": "none",
        "checkpoint_load": ckpt_info,
        "model_kwargs": model_kwargs,
        "paths": {
            "dataset_npz": str(resolve_path(args.dataset_npz, repo_root)),
            "metadata_json": str(resolve_path(args.metadata_json, repo_root)),
            "train_raw": str(resolve_path(args.train_raw, repo_root)),
            "val_raw": str(resolve_path(args.val_raw, repo_root)),
            "baseline_checkpoint": str(resolve_path(args.baseline_checkpoint, repo_root)),
        },
        "n_features": int(len(inp["feature_names"])),
        "raw_features_found": int(len(raw_features)),
        "num_bins": int(inp["num_bins"]),
        "metrics": metrics,
        "top2": top2,
        "token_risk": token_risk,
        "raw_drift_risk": raw_drift_risk,
        "error_risk": error_risk_summary,
        "diagnosis_text": diagnosis_text,
        "next_decision_rule": {
            "if_token_risk_high": "Run K/rare-bin/token-dropout tests, not pair heads.",
            "if_raw_drift_high": "Run stability-aware feature filtering/dropout tests.",
            "if_both_low_but_gap_high": "Run capacity/regularization ablations.",
            "before_solution_tests": "Compare against E1-E5 and previous K/adaptive-K tests to avoid duplication.",
        },
    }
    save_json(out_dir / "F0_summary.json", summary)
    write_summary_md(out_dir, summary)

    if args.make_zip:
        zip_path = out_dir.with_suffix(".zip")
        zip_dir(out_dir, zip_path)
        print(f"[F0] zipped outputs: {zip_path}", flush=True)

    print("[F0] done.", flush=True)
    print(f"[F0] train_macro_f1={train_overall['macro_f1']:.6f}", flush=True)
    print(f"[F0] val_macro_f1={val_overall['macro_f1']:.6f}", flush=True)
    print(f"[F0] gap_macro_f1={metrics['gap']['macro_f1']:.6f}", flush=True)


if __name__ == "__main__":
    main()
