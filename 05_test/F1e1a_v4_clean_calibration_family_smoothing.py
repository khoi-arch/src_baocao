#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F1e1a_v4 Clean Calibration-Derived Family Smoothing Matrix

Purpose
-------
No final model training. No validation-derived hyperparameter selection.

This script creates a clean-ish calibration procedure from the TRAIN split:

1. Split original train into:
      train_inner + calibration
   stratified by L2/L3 where possible.

2. Train an L1 model on train_inner only.

3. Infer calibration with full probability/top2/L3.

4. Derive a locked family-aware malware smoothing target matrix from calibration.

5. Save the locked matrix for the next step F1e1b.

Validation set is NOT used for deriving the smoothing matrix.

Leakage note
------------
This avoids validation leakage/tuning-to-val.

However, because dataset.npz token bins/offsets were already built by the
existing preprocessing pipeline, this is clean for validation usage but not a
fully strict nested-preprocessing experiment. A fully strict version would rebuild
tokenization/preprocessing using train_inner only. For the current project, this
script is intended to avoid the major issue: choosing parameters from validation.

Outputs
-------
- F1e1a_v4_split_indices.npz
- F1e1a_v4_split_distribution.csv
- history.csv
- best_calibration_model.pt / last_calibration_model.pt
- F1e1a_v4_calibration_predictions_with_probs_top2_l3.csv
- F1e1a_v4_train_inner_predictions_with_probs_top2_l3.csv
- F1e1a_v4_family_summary_by_split.csv
- F1e1a_v4_locked_family_smoothing_matrix_CALIBRATION_DERIVED.csv
- F1e1a_v4_locked_family_smoothing_targets.json
- F1e1a_v4_leakage_policy.md
- F1e1a_v4_report.md
- combined zip
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import math
import random
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader, Subset


DEFAULT_CLASS_NAMES = ["Benign", "Ransomware", "Spyware", "Trojan"]
DEFAULT_MALWARE = ["Ransomware", "Spyware", "Trojan"]
L1_REF = {"train_macro_f1": 0.911431, "val_macro_f1": 0.814224, "gap_macro_f1": 0.097207}


def log(msg: str) -> None:
    print(f"[F1e1a_v4] {msg}", flush=True)


def repo_root_from_here() -> Path:
    p = Path(__file__).resolve()
    if p.parent.name == "05_test":
        return p.parents[1]
    return Path.cwd().resolve()


def resolve_path(p: str | Path, root: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (root / p).resolve()


def clean(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_json(p: Path) -> Dict[str, Any]:
    try:
        return load_json(p) if p.exists() else {}
    except Exception:
        return {}


def cfg_get(cfg: Dict[str, Any], k: str, default):
    if k in cfg:
        return cfg[k]
    for dname in ["model", "model_config", "training", "args"]:
        d = cfg.get(dname, {})
        if isinstance(d, dict) and k in d:
            return d[k]
    return default


def load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def class_names_from_arg(s: str) -> List[str]:
    out = [clean(x) for x in s.split(",") if clean(x)]
    return out or DEFAULT_CLASS_NAMES


def find_label_col(df: pd.DataFrame, level: str) -> Optional[str]:
    if level == "L2":
        cands = ["label_L2", "Label_L2", "l2", "L2", "Category", "category"]
    else:
        cands = ["label_L3", "Label_L3", "l3", "L3", "Family", "family", "MalwareFamily", "malware_family"]
    for c in cands:
        if c in df.columns:
            return c
    return None


def load_dataset_train_only(dataset_npz: Path, train_raw: Path, class_names: List[str]) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    data = np.load(dataset_npz, allow_pickle=True)
    required = ["X_train_bin", "X_train_offset", "y_train"]
    missing = [k for k in required if k not in data.files]
    if missing:
        raise KeyError(f"dataset npz missing train keys: {missing}")

    X_bin = np.asarray(data["X_train_bin"], dtype=np.int64)
    X_off = np.asarray(data["X_train_offset"], dtype=np.float32)
    y = np.asarray(data["y_train"], dtype=np.int64).reshape(-1)
    feature_names = [str(x) for x in np.asarray(data["feature_names"]).tolist()] if "feature_names" in data.files else [f"f{i}" for i in range(X_bin.shape[1])]
    num_bins = int(np.asarray(data["num_bins"]).reshape(-1)[0]) if "num_bins" in data.files else 512

    if not train_raw.exists():
        raise FileNotFoundError(f"train_raw not found: {train_raw}")
    raw_df = pd.read_csv(train_raw)
    if len(raw_df) != len(y):
        raise ValueError(f"train_raw rows={len(raw_df)} but y_train={len(y)}; cannot align L3 labels safely")

    feat_cols = [c for c in feature_names if c in raw_df.columns]
    if len(feat_cols) != len(feature_names):
        exclude = {"label_L1", "label_L2", "label_L3", "Label_L1", "Label_L2", "Label_L3", "Class", "Category", "class", "category", "Family", "family"}
        feat_cols = [c for c in raw_df.columns if c not in exclude and pd.api.types.is_numeric_dtype(raw_df[c])][:len(feature_names)]
    if len(feat_cols) != len(feature_names):
        raise ValueError(f"raw feature mismatch: got {len(feat_cols)}, expected {len(feature_names)}")

    R = raw_df[feat_cols].to_numpy(dtype=np.float32)
    # This scaling uses the original train only. For a fully strict internal-calibration
    # study one would fit this on train_inner only and rebuild tokens too; see policy.
    mn = np.nanmin(R, axis=0, keepdims=True)
    mx = np.nanmax(R, axis=0, keepdims=True)
    den = mx - mn
    den[den < 1e-8] = 1.0
    X_raw = np.clip((R - mn) / den, 0.0, 1.0).astype(np.float32)

    values = np.stack([X_off.astype(np.float32), X_raw.astype(np.float32), np.ones_like(X_off, dtype=np.float32)], axis=-1).astype(np.float32)

    l2_col = find_label_col(raw_df, "L2")
    l3_col = find_label_col(raw_df, "L3")

    label_L2 = raw_df[l2_col].map(clean).to_numpy() if l2_col else np.array([""] * len(y), dtype=object)
    label_L3 = raw_df[l3_col].map(clean).to_numpy() if l3_col else np.array([""] * len(y), dtype=object)
    y_to_l2 = np.array([class_names[int(i)] if int(i) < len(class_names) else str(i) for i in y], dtype=object)

    # If raw L2 absent/not matching, use y-derived L2.
    if pd.Series(label_L2).isin(class_names).mean() < 0.80:
        label_L2 = y_to_l2
    # If L3 absent, fallback to L2 but mark in info.
    l3_missing = (pd.Series(label_L3).map(clean) == "").mean() > 0.80
    if l3_missing:
        label_L3 = label_L2.copy()

    ds = {
        "tokens": X_bin,
        "values": values,
        "y": y,
        "label_L2": label_L2,
        "label_L3": label_L3,
    }
    info = {
        "dataset_npz": str(dataset_npz),
        "train_raw": str(train_raw),
        "keys": list(data.files),
        "n_train": int(len(y)),
        "n_features": int(X_bin.shape[1]),
        "num_bins": int(num_bins),
        "raw_feature_cols_preview": feat_cols[:10],
        "l2_col": l2_col,
        "l3_col": l3_col,
        "l3_missing_fallback_to_L2": bool(l3_missing),
        "label_L3_unique_count": int(pd.Series(label_L3).nunique()),
        "label_L3_unique_preview": sorted(pd.Series(label_L3).dropna().astype(str).unique().tolist())[:40],
        "values_candidate": "offset_raw_one",
        "method_limitation": "No validation used. Existing dataset tokens/offsets are reused from prior preprocessing; fully strict nested preprocessing would rebuild tokens using train_inner only.",
    }
    return ds, info


class TrainDataset(Dataset):
    def __init__(self, ds: Dict[str, np.ndarray]):
        self.tokens = ds["tokens"]
        self.values = ds["values"]
        self.y = ds["y"]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int):
        return (
            torch.as_tensor(self.tokens[idx], dtype=torch.long),
            torch.as_tensor(self.values[idx], dtype=torch.float32),
            torch.as_tensor(self.y[idx], dtype=torch.long),
        )


def make_calibration_split(ds: Dict[str, np.ndarray], calib_size: float, seed: int, class_names: List[str]) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    y = ds["y"]
    l2 = pd.Series(ds["label_L2"]).map(clean)
    l3 = pd.Series(ds["label_L3"]).map(clean)
    strat = (l2 + "::" + l3).astype(str)

    counts = strat.value_counts()
    # Rare strata with <2 samples cannot be stratified; fallback those to L2.
    rare = strat.map(counts) < 2
    strat_safe = strat.copy()
    strat_safe[rare] = l2[rare].astype(str)

    # If still rare, fallback to y.
    counts2 = strat_safe.value_counts()
    rare2 = strat_safe.map(counts2) < 2
    if rare2.any():
        strat_safe[rare2] = pd.Series(y[rare2.to_numpy()]).map(lambda i: class_names[int(i)] if int(i) < len(class_names) else str(i)).to_numpy()

    # If still impossible, use y only.
    if (strat_safe.value_counts() < 2).any():
        strat_safe = pd.Series(y).map(lambda i: class_names[int(i)] if int(i) < len(class_names) else str(i))

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=calib_size, random_state=seed)
    all_idx = np.arange(len(y))
    train_inner_idx, calib_idx = next(splitter.split(all_idx, strat_safe))

    def dist_rows(name: str, idx: np.ndarray):
        d = pd.DataFrame({
            "split": name,
            "label_L2": l2.iloc[idx].to_numpy(),
            "label_L3": l3.iloc[idx].to_numpy(),
            "y": y[idx],
            "strat_key": strat.iloc[idx].to_numpy(),
        })
        g = d.groupby(["split", "label_L2", "label_L3"]).size().reset_index(name="n")
        return g

    dist = pd.concat([dist_rows("train_inner", train_inner_idx), dist_rows("calibration", calib_idx)], ignore_index=True)
    return np.sort(train_inner_idx), np.sort(calib_idx), dist


def infer_model_config(cfg: Dict[str, Any], ds_info: Dict[str, Any], class_names: List[str], args) -> Dict[str, Any]:
    return {
        "num_bins": int(cfg_get(cfg, "num_bins", cfg_get(cfg, "K", ds_info["num_bins"]))),
        "n_features": int(cfg_get(cfg, "n_features", cfg_get(cfg, "num_features", ds_info["n_features"]))),
        "num_classes": int(cfg_get(cfg, "num_classes", len(class_names))),
        "value_dim": int(cfg_get(cfg, "value_dim", 32)),
        "feature_dim": int(cfg_get(cfg, "feature_dim", 32)),
        "hidden_dim": int(args.hidden_dim if args.hidden_dim is not None else cfg_get(cfg, "hidden_dim", 128)),
        "num_layers": 1,
        "num_heads": int(args.num_heads if args.num_heads is not None else cfg_get(cfg, "num_heads", 4)),
        "dropout": float(args.dropout if args.dropout is not None else cfg_get(cfg, "dropout", 0.1)),
        "classifier_hidden_dim": int(args.classifier_hidden_dim if args.classifier_hidden_dim is not None else cfg_get(cfg, "classifier_hidden_dim", 128)),
        "classifier_dropout": float(args.classifier_dropout if args.classifier_dropout is not None else cfg_get(cfg, "classifier_dropout", 0.1)),
        "gate_init": float(cfg_get(cfg, "gate_init", 0.0)),
    }


def build_model(root: Path, model_config: Dict[str, Any]) -> nn.Module:
    mod = load_module_from_path("_f1e1a_v4_model_06_model", root / "02_src" / "06_model.py")
    cls = getattr(mod, "D3C2D3Transformer", None)
    if cls is None:
        raise RuntimeError("D3C2D3Transformer not found in 02_src/06_model.py")
    kwargs = {k: v for k, v in model_config.items() if k in inspect.signature(cls).parameters}
    return cls(**kwargs)


def compute_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y.astype(int), minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = len(y) / (num_classes * counts)
    return torch.as_tensor(w, dtype=torch.float32)


def warmup_cosine_lr(epoch: int, base_lr: float, epochs: int, warmup_epochs: int, min_lr_ratio: float) -> float:
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return base_lr * epoch / warmup_epochs
    if epochs <= warmup_epochs:
        return base_lr
    progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1 - min_lr_ratio) * cosine)


def set_lr(opt, lr: float):
    for g in opt.param_groups:
        g["lr"] = lr


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    ys, probs_all = [], []
    for tokens, values, y in loader:
        tokens = tokens.to(device, non_blocking=True)
        values = values.to(device, non_blocking=True)
        logits = model(tokens, values)
        if not torch.is_tensor(logits):
            if isinstance(logits, dict):
                logits = logits.get("logits", next(iter(logits.values())))
            elif isinstance(logits, (tuple, list)):
                logits = logits[0]
        probs = torch.softmax(logits.detach().float(), dim=1)
        ys.append(y.cpu().numpy())
        probs_all.append(probs.cpu().numpy())
    y = np.concatenate(ys)
    probs = np.concatenate(probs_all, axis=0)
    pred = probs.argmax(axis=1)
    return y, pred, probs


def metrics(y, pred) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
    }


def make_pred_df(indices: np.ndarray, ds: Dict[str, np.ndarray], y: np.ndarray, pred: np.ndarray, probs: np.ndarray, class_names: List[str]) -> pd.DataFrame:
    order = np.argsort(-probs, axis=1)
    top1 = order[:, 0]
    top2 = order[:, 1]
    rows = pd.DataFrame({
        "global_train_idx": indices,
        "y_true": y,
        "true_L2": [class_names[int(i)] for i in y],
        "true_L3": ds["label_L3"][indices],
        "y_pred": pred,
        "pred_L2": [class_names[int(i)] for i in pred],
        "correct": pred == y,
        "top1_L2": [class_names[int(i)] for i in top1],
        "top1_prob": probs[np.arange(len(y)), top1],
        "top2_L2": [class_names[int(i)] for i in top2],
        "top2_prob": probs[np.arange(len(y)), top2],
        "top2_gap": probs[np.arange(len(y)), top1] - probs[np.arange(len(y)), top2],
        "true_prob": probs[np.arange(len(y)), y],
        "true_in_top2": np.array([yy in order[i, :2] for i, yy in enumerate(y)], dtype=bool),
    })
    for i, name in enumerate(class_names):
        rows[f"prob_{name}"] = probs[:, i]
    malware = [c for c in DEFAULT_MALWARE if c in class_names]
    other_mass = []
    for _, row in rows.iterrows():
        mass = 0.0
        for m in malware:
            if m != row["true_L2"] and f"prob_{m}" in rows.columns:
                mass += float(row[f"prob_{m}"])
        other_mass.append(mass)
    rows["other_malware_prob_mass"] = other_mass
    return rows


def entropy_norm(vals: np.ndarray) -> float:
    v = np.asarray(vals, dtype=float)
    v = v[v > 0]
    if len(v) <= 1:
        return 0.0
    p = v / v.sum()
    h = -float(np.sum(p * np.log(p)))
    return h / math.log(len(p))


def summarize_family(pred_df: pd.DataFrame, class_names: List[str], split: str, min_family_support: int) -> pd.DataFrame:
    rows = []
    for (true_l2, true_l3), g in pred_df.groupby(["true_L2", "true_L3"], dropna=False):
        n = int(len(g))
        pred_rates = g["pred_L2"].value_counts().reindex(class_names, fill_value=0) / n
        top2_rates = g["top2_L2"].value_counts().reindex(class_names, fill_value=0) / n
        row = {
            "split": split,
            "true_L2": clean(true_l2),
            "true_L3": clean(true_l3),
            "n": n,
            "audit_reliable_support": bool(n >= min_family_support),
            "accuracy": float(g["correct"].mean()),
            "error_rate": float(1 - g["correct"].mean()),
            "true_in_top2_rate": float(g["true_in_top2"].mean()),
            "top2_gap_mean": float(g["top2_gap"].mean()),
            "top2_gap_median": float(g["top2_gap"].median()),
            "top1_prob_mean": float(g["top1_prob"].mean()),
            "true_prob_mean": float(g["true_prob"].mean()),
            "other_malware_prob_mass_mean": float(g["other_malware_prob_mass"].mean()),
            "pred_distribution_entropy_norm": entropy_norm(pred_rates.to_numpy()),
            "top2_distribution_entropy_norm": entropy_norm(top2_rates.to_numpy()),
        }
        for c in class_names:
            row[f"mean_prob_{c}"] = float(g[f"prob_{c}"].mean()) if f"prob_{c}" in g.columns else np.nan
            row[f"pred_rate_{c}"] = float(pred_rates[c])
            row[f"top2_rate_{c}"] = float(top2_rates[c])
        rows.append(row)
    return pd.DataFrame(rows)


def derive_locked_family_matrix(fam: pd.DataFrame, class_names: List[str], malware_classes: List[str],
                                eps_cap: float, min_family_support: int) -> pd.DataFrame:
    rows = []
    sub = fam[fam["split"] == "calibration"].copy()
    for _, r in sub.iterrows():
        true_l2 = clean(r["true_L2"])
        true_l3 = clean(r["true_L3"])
        n = int(r["n"])
        row = {
            "true_L2": true_l2,
            "true_L3": true_l3,
            "n_calibration": n,
            "reliable_support": bool(n >= min_family_support),
            "eps_cap": float(eps_cap),
            "eps_family": 0.0,
            "source": "",
        }
        for c in class_names:
            row[f"target_{c}"] = 0.0

        if true_l2 not in malware_classes:
            row[f"target_{true_l2}"] = 1.0 if true_l2 in class_names else 0.0
            if true_l2 not in class_names and class_names:
                row[f"target_{class_names[0]}"] = 1.0
            row["source"] = "non_malware_or_benign_one_hot"
            row["target_sum"] = sum(row[f"target_{c}"] for c in class_names)
            rows.append(row)
            continue

        if n < min_family_support:
            row[f"target_{true_l2}"] = 1.0
            row["source"] = "support_below_min_keep_one_hot"
            row["target_sum"] = sum(row[f"target_{c}"] for c in class_names)
            rows.append(row)
            continue

        eps = min(float(eps_cap), max(0.0, float(r.get("other_malware_prob_mass_mean", 0.0))))
        other_scores = {}
        for c in malware_classes:
            if c == true_l2:
                continue
            mean_prob = float(r.get(f"mean_prob_{c}", 0.0))
            top2_rate = float(r.get(f"top2_rate_{c}", 0.0))
            pred_rate = float(r.get(f"pred_rate_{c}", 0.0))
            # probability carries most weight; top2 shows boundary competition; hard pred shows actual errors.
            other_scores[c] = 0.55 * mean_prob + 0.30 * top2_rate + 0.15 * pred_rate

        total = sum(max(0.0, v) for v in other_scores.values())
        if eps <= 0 or total <= 0:
            row[f"target_{true_l2}"] = 1.0
            row["source"] = "no_other_malware_overlap_signal_keep_one_hot"
        else:
            row[f"target_{true_l2}"] = 1.0 - eps
            for c, sc in other_scores.items():
                row[f"target_{c}"] = eps * max(0.0, sc) / total
            if "Benign" in class_names:
                row["target_Benign"] = 0.0
            row["eps_family"] = eps
            row["source"] = "CALIBRATION_DERIVED_family_prob_top2_pred_weighted"
        row["target_sum"] = sum(row[f"target_{c}"] for c in class_names)
        rows.append(row)
    return pd.DataFrame(rows)


def write_policy(out_dir: Path):
    text = """# F1e1a_v4 Clean Calibration Policy

## What this script is allowed to do

- Split original train into train_inner and calibration.
- Train a temporary L1 model on train_inner only.
- Use calibration predictions to derive a smoothing matrix.
- Save that matrix as a locked candidate for F1e1b.
- Keep validation untouched for later evaluation.

## What this script must not do

- It must not derive smoothing from validation labels or validation predictions.
- It must not tune parameters on validation.
- It must not claim calibration result as final model performance.
- It must not create fake family labels or fake samples.

## Data leakage status

This avoids the major leakage issue: validation-derived hyperparameter selection.

Caveat:
The dataset tokens/offsets are reused from the existing preprocessing output. A fully nested strict study would rebuild preprocessing/tokenization using train_inner only, then derive calibration predictions. That is more expensive and should be noted if required by the report.

## Next clean step

F1e1b should train on the original full train split with the locked matrix from:

    F1e1a_v4_locked_family_smoothing_matrix_CALIBRATION_DERIVED.csv

Then evaluate once on validation. If F1e1b is bad, do not repeatedly tune using validation without creating a new calibration procedure.
"""
    (out_dir / "F1e1a_v4_leakage_policy.md").write_text(text, encoding="utf-8")


def write_report(out_dir: Path, ds_info: Dict[str, Any], config: Dict[str, Any], split_metrics: Dict[str, Any],
                 fam: pd.DataFrame, locked: pd.DataFrame):
    lines = []
    lines.append("# F1e1a_v4 Clean Calibration-Derived Family Smoothing Matrix\n")
    lines.append("## Purpose\n")
    lines.append("```text")
    lines.append("No final model training.")
    lines.append("No validation-derived hyperparameter selection.")
    lines.append("Train temporary L1 on train_inner, infer calibration, derive locked family-aware smoothing matrix.")
    lines.append("```")

    lines.append("\n## Leakage status\n")
    lines.append("```text")
    lines.append("Validation is not used to derive the matrix.")
    lines.append("The locked matrix can be used in F1e1b, then validation can be evaluated once.")
    lines.append("Caveat: existing tokenized dataset is reused; fully nested preprocessing would rebuild tokens on train_inner only.")
    lines.append("```")

    lines.append("\n## Dataset / split info\n")
    lines.append("```json")
    lines.append(json.dumps(ds_info, indent=2, default=str)[:5000])
    lines.append("```")

    lines.append("\n## Training config\n")
    lines.append("```json")
    lines.append(json.dumps(config, indent=2, default=str)[:5000])
    lines.append("```")

    lines.append("\n## Temporary calibration model metrics\n")
    lines.append(pd.DataFrame([{"split": k, **v} for k, v in split_metrics.items()]).to_markdown(index=False))

    lines.append("\n## Calibration family summary: hardest families\n")
    cal = fam[fam["split"] == "calibration"].copy()
    if len(cal):
        cols = [c for c in [
            "true_L2", "true_L3", "n", "audit_reliable_support", "accuracy", "error_rate",
            "true_in_top2_rate", "top2_gap_mean", "true_prob_mean", "other_malware_prob_mass_mean",
            "mean_prob_Ransomware", "mean_prob_Spyware", "mean_prob_Trojan",
            "pred_rate_Ransomware", "pred_rate_Spyware", "pred_rate_Trojan",
            "top2_rate_Ransomware", "top2_rate_Spyware", "top2_rate_Trojan",
        ] if c in cal.columns]
        lines.append(cal.sort_values(["audit_reliable_support", "error_rate", "other_malware_prob_mass_mean"], ascending=[False, False, False]).head(60)[cols].to_markdown(index=False))
    else:
        lines.append("No calibration family summary.")

    lines.append("\n## Locked family smoothing matrix\n")
    if len(locked):
        lines.append(locked.to_markdown(index=False))
    else:
        lines.append("No locked matrix produced.")

    lines.append("\n## Next step\n")
    lines.append("```text")
    lines.append("F1e1b: train L1 + locked family-aware smoothing matrix on full original train.")
    lines.append("Then evaluate once on validation.")
    lines.append("Do not tune matrix using validation result unless you create a new calibration protocol.")
    lines.append("```")

    (out_dir / "F1e1a_v4_report.md").write_text("\n".join(lines), encoding="utf-8")


def zip_dir(src: Path, dst: Path):
    if dst.exists():
        dst.unlink()
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for p in src.rglob("*"):
            if p.is_file() and p != dst:
                z.write(p, p.relative_to(src.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-npz", default="03_outputs/05_dataset/dataset.npz")
    ap.add_argument("--train-raw", default="01_split/train_raw.csv")
    ap.add_argument("--base-config", default="03_outputs/06_model/config.json")
    ap.add_argument("--out-dir", default="05_test/outputs/F1e1a_v4_clean_calibration_family_smoothing")
    ap.add_argument("--combined-zip", default="05_test/outputs/F1e1a_v4_clean_calibration_family_smoothing.zip")
    ap.add_argument("--class-names", default="Benign,Ransomware,Spyware,Trojan")
    ap.add_argument("--malware-classes", default="Ransomware,Spyware,Trojan")

    ap.add_argument("--calib-size", type=float, default=0.20)
    ap.add_argument("--eps-cap", type=float, default=0.20)
    ap.add_argument("--min-family-support", type=int, default=30)

    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-epochs", type=int, default=8)
    ap.add_argument("--min-lr-ratio", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--grad-clip-norm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")

    ap.add_argument("--hidden-dim", type=int, default=None)
    ap.add_argument("--num-heads", type=int, default=None)
    ap.add_argument("--classifier-hidden-dim", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--classifier-dropout", type=float, default=None)

    args = ap.parse_args()

    root = repo_root_from_here()
    out_dir = resolve_path(args.out_dir, root)
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_zip = resolve_path(args.combined_zip, root)

    seed_everything(args.seed)
    class_names = class_names_from_arg(args.class_names)
    malware_classes = [clean(x) for x in args.malware_classes.split(",") if clean(x)]

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    torch.backends.cudnn.benchmark = True

    log(f"root={root}")
    log(f"out_dir={out_dir}")
    log(f"device={device}")
    log("Clean calibration derivation: validation is not used.")

    ds, ds_info = load_dataset_train_only(resolve_path(args.dataset_npz, root), resolve_path(args.train_raw, root), class_names)
    train_inner_idx, calib_idx, split_dist = make_calibration_split(ds, args.calib_size, args.seed, class_names)

    np.savez(out_dir / "F1e1a_v4_split_indices.npz", train_inner_idx=train_inner_idx, calibration_idx=calib_idx)
    split_dist.to_csv(out_dir / "F1e1a_v4_split_distribution.csv", index=False)
    (out_dir / "F1e1a_v4_dataset_info.json").write_text(json.dumps(ds_info, indent=2, default=str), encoding="utf-8")

    cfg = safe_json(resolve_path(args.base_config, root))
    model_config = infer_model_config(cfg, ds_info, class_names, args)
    model = build_model(root, model_config).to(device)

    full_dataset = TrainDataset(ds)
    train_loader = DataLoader(
        Subset(full_dataset, train_inner_idx.tolist()),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )
    train_eval_loader = DataLoader(
        Subset(full_dataset, train_inner_idx.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )
    calib_loader = DataLoader(
        Subset(full_dataset, calib_idx.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    class_weights = compute_class_weights(ds["y"][train_inner_idx], len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and args.amp))

    config = {
        "experiment": "F1e1a_v4_clean_calibration_family_smoothing",
        "role": "derive locked family smoothing matrix from train_inner/calibration; no validation used",
        "class_names": class_names,
        "malware_classes": malware_classes,
        "model_config": model_config,
        "training": vars(args),
        "split": {
            "n_train_inner": int(len(train_inner_idx)),
            "n_calibration": int(len(calib_idx)),
            "calib_size": float(args.calib_size),
            "stratification": "label_L2::label_L3 fallback to L2/y for rare strata",
        },
        "loss": {"name": "CrossEntropyLoss", "class_weights": class_weights.detach().cpu().numpy().tolist()},
        "methodology": {
            "validation_used_for_matrix": False,
            "fake_data_used": False,
            "known_caveat": ds_info["method_limitation"],
        },
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

    best_calib = -1.0
    best_epoch = -1
    no_improve = 0
    history = []
    split_metrics = {}
    best_state = None

    for epoch in range(1, args.epochs + 1):
        lr = warmup_cosine_lr(epoch, args.lr, args.epochs, args.warmup_epochs, args.min_lr_ratio)
        set_lr(optimizer, lr)
        model.train()
        total_loss = 0.0
        n_seen = 0
        t0 = time.time()

        for tokens, values, y in train_loader:
            tokens = tokens.to(device, non_blocking=True)
            values = values.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda" and args.amp)):
                logits = model(tokens, values)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            if args.grad_clip_norm and args.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            bs = int(y.shape[0])
            total_loss += float(loss.detach().cpu().item()) * bs
            n_seen += bs

        y_tr, pred_tr, prob_tr = predict(model, train_eval_loader, device)
        y_ca, pred_ca, prob_ca = predict(model, calib_loader, device)
        m_tr = metrics(y_tr, pred_tr)
        m_ca = metrics(y_ca, pred_ca)
        row = {
            "epoch": epoch,
            "lr": lr,
            "loss": total_loss / max(1, n_seen),
            "train_inner_accuracy": m_tr["accuracy"],
            "train_inner_macro_f1": m_tr["macro_f1"],
            "train_inner_weighted_f1": m_tr["weighted_f1"],
            "calibration_accuracy": m_ca["accuracy"],
            "calibration_macro_f1": m_ca["macro_f1"],
            "calibration_weighted_f1": m_ca["weighted_f1"],
            "gap_train_inner_minus_calib_macro_f1": m_tr["macro_f1"] - m_ca["macro_f1"],
            "seconds": time.time() - t0,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

        if m_ca["macro_f1"] > best_calib + args.min_delta:
            best_calib = m_ca["macro_f1"]
            best_epoch = epoch
            no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_calibration_macro_f1": best_calib,
                "model_config": model_config,
                "config": config,
            }, out_dir / "best_calibration_model.pt")
        else:
            no_improve += 1

        log(
            f"epoch {epoch:03d} loss={row['loss']:.5f} "
            f"train_inner_f1={m_tr['macro_f1']:.6f} calib_f1={m_ca['macro_f1']:.6f} "
            f"gap={row['gap_train_inner_minus_calib_macro_f1']:.6f} best={best_calib:.6f}@{best_epoch}"
        )

        if no_improve >= args.patience:
            log(f"early stopping at epoch={epoch}; best_epoch={best_epoch}; best_calib={best_calib:.6f}")
            break

    torch.save({
        "epoch": history[-1]["epoch"] if history else None,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": model_config,
        "config": config,
    }, out_dir / "last_calibration_model.pt")

    if best_state is None:
        raise RuntimeError("No best state captured.")
    model.load_state_dict(best_state)
    model.eval()

    # Final inference using best calibration model.
    y_tr, pred_tr, prob_tr = predict(model, train_eval_loader, device)
    y_ca, pred_ca, prob_ca = predict(model, calib_loader, device)
    split_metrics = {
        "train_inner": metrics(y_tr, pred_tr),
        "calibration": metrics(y_ca, pred_ca),
    }
    split_metrics["train_inner"]["n"] = int(len(train_inner_idx))
    split_metrics["calibration"]["n"] = int(len(calib_idx))
    split_metrics["gap_train_inner_minus_calib_macro_f1"] = split_metrics["train_inner"]["macro_f1"] - split_metrics["calibration"]["macro_f1"]
    (out_dir / "F1e1a_v4_split_metrics.json").write_text(json.dumps(split_metrics, indent=2), encoding="utf-8")

    train_pred_df = make_pred_df(train_inner_idx, ds, y_tr, pred_tr, prob_tr, class_names)
    cal_pred_df = make_pred_df(calib_idx, ds, y_ca, pred_ca, prob_ca, class_names)
    train_pred_df["split"] = "train_inner"
    cal_pred_df["split"] = "calibration"
    train_pred_df.to_csv(out_dir / "F1e1a_v4_train_inner_predictions_with_probs_top2_l3.csv", index=False)
    cal_pred_df.to_csv(out_dir / "F1e1a_v4_calibration_predictions_with_probs_top2_l3.csv", index=False)

    for name, yx, px in [("train_inner", y_tr, pred_tr), ("calibration", y_ca, pred_ca)]:
        rep = classification_report(yx, px, labels=list(range(len(class_names))), target_names=class_names, output_dict=True, zero_division=0)
        (out_dir / f"F1e1a_v4_{name}_classification_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
        cm = confusion_matrix(yx, px, labels=list(range(len(class_names))))
        pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(out_dir / f"F1e1a_v4_{name}_confusion_matrix.csv")

    fam_train = summarize_family(train_pred_df, class_names, "train_inner", args.min_family_support)
    fam_cal = summarize_family(cal_pred_df, class_names, "calibration", args.min_family_support)
    fam = pd.concat([fam_train, fam_cal], ignore_index=True)
    fam.to_csv(out_dir / "F1e1a_v4_family_summary_by_split.csv", index=False)

    locked = derive_locked_family_matrix(fam, class_names, malware_classes, args.eps_cap, args.min_family_support)
    locked.to_csv(out_dir / "F1e1a_v4_locked_family_smoothing_matrix_CALIBRATION_DERIVED.csv", index=False)

    targets = {
        "usage": "Use this locked matrix in F1e1b. It is derived from calibration split only; validation not used.",
        "validation_used": False,
        "fake_data_used": False,
        "class_names": class_names,
        "malware_classes": malware_classes,
        "eps_cap": float(args.eps_cap),
        "min_family_support": int(args.min_family_support),
        "matrix_file": "F1e1a_v4_locked_family_smoothing_matrix_CALIBRATION_DERIVED.csv",
        "matrix": locked.to_dict(orient="records"),
        "split_metrics": split_metrics,
        "method_caveat": ds_info["method_limitation"],
    }
    (out_dir / "F1e1a_v4_locked_family_smoothing_targets.json").write_text(json.dumps(targets, indent=2), encoding="utf-8")

    write_policy(out_dir)
    write_report(out_dir, ds_info, config, split_metrics, fam, locked)
    zip_dir(out_dir, combined_zip)

    log("Split metrics:")
    print(pd.DataFrame([{"split": k, **v} for k, v in split_metrics.items() if isinstance(v, dict)]).to_string(index=False), flush=True)
    log(f"zip={combined_zip}")
    log("DONE")


if __name__ == "__main__":
    main()
