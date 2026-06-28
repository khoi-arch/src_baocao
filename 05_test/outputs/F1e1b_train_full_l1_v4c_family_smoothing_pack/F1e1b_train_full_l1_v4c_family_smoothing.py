#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F1e1b Train Full L1 with Locked v4c Family Smoothing Matrix

Purpose
-------
Train the real experiment model on the full original train split using the
locked family-aware smoothing matrix derived by F1e1a_v4c.

Clean methodology
-----------------
- The smoothing matrix is loaded from F1e1a_v4c.
- Validation is NOT used to derive the matrix.
- Validation is NOT used for early stopping or hyperparameter tuning.
- The number of epochs is fixed before validation evaluation.
  By default, it can be read from F1e1a_v4 history best calibration epoch,
  which comes from train_inner/calibration, not validation.
- Validation is evaluated once at the end.

Important caveat
----------------
This script reuses the existing preprocessed dataset.npz. A fully nested strict
experiment would rebuild tokenization/preprocessing under the chosen training
protocol. This script avoids the major leakage issue: validation-derived matrix
selection.

Outputs
-------
- config.json
- history.csv
- final_model.pt
- train_classification_report_final.json
- val_classification_report_final.json
- train_confusion_matrix_final.csv
- val_confusion_matrix_final.csv
- train_predictions_final.csv
- val_predictions_final.csv
- F1e1b_report.md
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
from torch.utils.data import Dataset, DataLoader


DEFAULT_CLASS_NAMES = ["Benign", "Ransomware", "Spyware", "Trojan"]
DEFAULT_MALWARE = ["Ransomware", "Spyware", "Trojan"]


def log(msg: str):
    print(f"[F1e1b] {msg}", flush=True)


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


def parse_list(s: str) -> List[str]:
    return [clean(x) for x in s.split(",") if clean(x)]


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


def find_label_col(df: pd.DataFrame, level: str) -> Optional[str]:
    if level == "L2":
        candidates = ["label_L2", "Label_L2", "l2", "L2", "Category", "category"]
    else:
        candidates = ["label_L3", "Label_L3", "l3", "L3", "Family", "family", "MalwareFamily", "malware_family"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_dataset(dataset_npz: Path, train_raw: Path, val_raw: Path, class_names: List[str]):
    data = np.load(dataset_npz, allow_pickle=True)
    required = ["X_train_bin", "X_train_offset", "y_train", "X_val_bin", "X_val_offset", "y_val"]
    missing = [k for k in required if k not in data.files]
    if missing:
        raise KeyError(f"dataset npz missing keys: {missing}")

    feature_names = [str(x) for x in np.asarray(data["feature_names"]).tolist()] if "feature_names" in data.files else [f"f{i}" for i in range(np.asarray(data['X_train_bin']).shape[1])]
    num_bins = int(np.asarray(data["num_bins"]).reshape(-1)[0]) if "num_bins" in data.files else 512

    Xtr_bin = np.asarray(data["X_train_bin"], dtype=np.int64)
    Xtr_off = np.asarray(data["X_train_offset"], dtype=np.float32)
    ytr = np.asarray(data["y_train"], dtype=np.int64).reshape(-1)
    Xva_bin = np.asarray(data["X_val_bin"], dtype=np.int64)
    Xva_off = np.asarray(data["X_val_offset"], dtype=np.float32)
    yva = np.asarray(data["y_val"], dtype=np.int64).reshape(-1)

    def load_raw(path: Path, expected: int, split: str):
        if not path.exists():
            raise FileNotFoundError(f"{split} raw csv not found: {path}")
        df = pd.read_csv(path)
        if len(df) != expected:
            raise ValueError(f"{split} raw rows={len(df)} expected={expected}; cannot align labels safely")

        feat_cols = [c for c in feature_names if c in df.columns]
        if len(feat_cols) != len(feature_names):
            exclude = {"label_L1","label_L2","label_L3","Label_L1","Label_L2","Label_L3","Class","Category","class","category","Family","family"}
            feat_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])][:len(feature_names)]
        if len(feat_cols) != len(feature_names):
            raise ValueError(f"{split} raw feature mismatch: got {len(feat_cols)}, expected {len(feature_names)}")
        R = df[feat_cols].to_numpy(dtype=np.float32)

        l2_col = find_label_col(df, "L2")
        l3_col = find_label_col(df, "L3")
        label_L2 = df[l2_col].map(clean).to_numpy() if l2_col else np.array([""] * expected, dtype=object)
        label_L3 = df[l3_col].map(clean).to_numpy() if l3_col else np.array([""] * expected, dtype=object)
        return R, label_L2, label_L3, {"l2_col": l2_col, "l3_col": l3_col, "raw_feature_cols_preview": feat_cols[:10]}

    Rtr, l2tr, l3tr, infotr = load_raw(train_raw, len(ytr), "train")
    Rva, l2va, l3va, infova = load_raw(val_raw, len(yva), "val")

    ytr_l2 = np.array([class_names[int(i)] if int(i) < len(class_names) else str(i) for i in ytr], dtype=object)
    yva_l2 = np.array([class_names[int(i)] if int(i) < len(class_names) else str(i) for i in yva], dtype=object)

    if pd.Series(l2tr).isin(class_names).mean() < 0.80:
        l2tr = ytr_l2
    if pd.Series(l2va).isin(class_names).mean() < 0.80:
        l2va = yva_l2
    if (pd.Series(l3tr).map(clean) == "").mean() > 0.80:
        l3tr = l2tr.copy()
    if (pd.Series(l3va).map(clean) == "").mean() > 0.80:
        l3va = l2va.copy()

    # Values candidate = offset_raw_one.
    # Scale raw values using train raw range only.
    mn = np.nanmin(Rtr, axis=0, keepdims=True)
    mx = np.nanmax(Rtr, axis=0, keepdims=True)
    den = mx - mn
    den[den < 1e-8] = 1.0
    Xtr_raw = np.clip((Rtr - mn) / den, 0.0, 1.0).astype(np.float32)
    Xva_raw = np.clip((Rva - mn) / den, 0.0, 1.0).astype(np.float32)

    def make_values(off, raw):
        return np.stack([off.astype(np.float32), raw.astype(np.float32), np.ones_like(off, dtype=np.float32)], axis=-1).astype(np.float32)

    ds = {
        "train": {
            "tokens": Xtr_bin,
            "values": make_values(Xtr_off, Xtr_raw),
            "y": ytr,
            "label_L2": l2tr,
            "label_L3": l3tr,
        },
        "val": {
            "tokens": Xva_bin,
            "values": make_values(Xva_off, Xva_raw),
            "y": yva,
            "label_L2": l2va,
            "label_L3": l3va,
        },
    }
    info = {
        "dataset_npz": str(dataset_npz),
        "train_raw": str(train_raw),
        "val_raw": str(val_raw),
        "keys": list(data.files),
        "n_train": int(len(ytr)),
        "n_val": int(len(yva)),
        "n_features": int(Xtr_bin.shape[1]),
        "num_bins": int(num_bins),
        "train_label_cols": infotr,
        "val_label_cols": infova,
        "train_L3_unique_count": int(pd.Series(l3tr).nunique()),
        "val_L3_unique_count": int(pd.Series(l3va).nunique()),
        "values_candidate": "offset_raw_one",
    }
    return ds, info


def load_family_matrix(path: Path, class_names: List[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"family smoothing matrix not found: {path}")
    mat = pd.read_csv(path)
    required = ["true_L2", "true_L3"] + [f"target_{c}" for c in class_names]
    missing = [c for c in required if c not in mat.columns]
    if missing:
        raise KeyError(f"matrix missing columns: {missing}")
    mat["true_L2"] = mat["true_L2"].map(clean)
    mat["true_L3"] = mat["true_L3"].map(clean)

    target_cols = [f"target_{c}" for c in class_names]
    for c in target_cols:
        mat[c] = pd.to_numeric(mat[c], errors="coerce").fillna(0.0)
    mat["target_sum_check"] = mat[target_cols].sum(axis=1)
    bad = np.abs(mat["target_sum_check"] - 1.0) > 1e-4
    if bad.any():
        raise ValueError("target rows not summing to 1:\n" + mat[bad][["true_L2", "true_L3", "target_sum_check"]].to_string(index=False))
    return mat


def make_soft_targets(ds_split: Dict[str, np.ndarray], mat: pd.DataFrame, class_names: List[str]):
    target_cols = [f"target_{c}" for c in class_names]
    key_to_target = {}
    for _, r in mat.iterrows():
        key_to_target[(clean(r["true_L2"]), clean(r["true_L3"]))] = np.array([float(r[c]) for c in target_cols], dtype=np.float32)

    n = len(ds_split["y"])
    out = np.zeros((n, len(class_names)), dtype=np.float32)
    matched = 0
    fallback_l2 = 0
    fallback_onehot = 0
    missing_keys = {}
    for i in range(n):
        l2 = clean(ds_split["label_L2"][i])
        l3 = clean(ds_split["label_L3"][i])
        y = int(ds_split["y"][i])
        key = (l2, l3)
        if key in key_to_target:
            out[i] = key_to_target[key]
            matched += 1
        elif (l2, l2) in key_to_target:
            out[i] = key_to_target[(l2, l2)]
            fallback_l2 += 1
        else:
            out[i, y] = 1.0
            fallback_onehot += 1
            missing_keys[key] = missing_keys.get(key, 0) + 1
    info = {
        "n": n,
        "matched_L2_L3": matched,
        "fallback_L2_L2": fallback_l2,
        "fallback_onehot": fallback_onehot,
        "missing_keys_top": sorted(missing_keys.items(), key=lambda kv: kv[1], reverse=True)[:20],
    }
    return out, info


class SoftTargetDataset(Dataset):
    def __init__(self, split: Dict[str, np.ndarray], soft_targets: Optional[np.ndarray] = None):
        self.tokens = split["tokens"]
        self.values = split["values"]
        self.y = split["y"]
        self.soft_targets = soft_targets

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        tokens = torch.as_tensor(self.tokens[idx], dtype=torch.long)
        values = torch.as_tensor(self.values[idx], dtype=torch.float32)
        y = torch.as_tensor(self.y[idx], dtype=torch.long)
        if self.soft_targets is None:
            return tokens, values, y
        t = torch.as_tensor(self.soft_targets[idx], dtype=torch.float32)
        return tokens, values, y, t


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
    mod = load_module_from_path("_f1e1b_model_06_model", root / "02_src" / "06_model.py")
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


class WeightedSoftTargetCE(nn.Module):
    def __init__(self, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        if class_weights is None:
            self.register_buffer("class_weights", None)
        else:
            self.register_buffer("class_weights", class_weights.float())

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        logp = torch.log_softmax(logits, dim=1)
        if self.class_weights is not None:
            target = target * self.class_weights.view(1, -1)
        loss = -(target * logp).sum(dim=1)
        return loss.mean()


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
    for batch in loader:
        if len(batch) == 4:
            tokens, values, y, _ = batch
        else:
            tokens, values, y = batch
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


def make_pred_df(split: Dict[str, np.ndarray], y: np.ndarray, pred: np.ndarray, probs: np.ndarray, class_names: List[str]):
    order = np.argsort(-probs, axis=1)
    top1 = order[:, 0]
    top2 = order[:, 1]
    df = pd.DataFrame({
        "sample_idx": np.arange(len(y)),
        "y_true": y,
        "true_L2": [class_names[int(i)] for i in y],
        "true_L3": split["label_L3"],
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
        df[f"prob_{name}"] = probs[:, i]
    return df


def find_default_epochs(v4_dir: Path, fallback: int) -> Tuple[int, str]:
    hist = v4_dir / "history.csv"
    if hist.exists():
        try:
            df = pd.read_csv(hist)
            if "calibration_macro_f1" in df.columns and "epoch" in df.columns:
                row = df.sort_values("calibration_macro_f1", ascending=False).iloc[0]
                return int(row["epoch"]), f"best calibration epoch from {hist}"
        except Exception:
            pass
    return fallback, "fallback fixed epochs"


def write_report(out_dir: Path, config: Dict[str, Any], target_info: Dict[str, Any], final_metrics: Dict[str, Any], class_names: List[str]):
    lines = []
    lines.append("# F1e1b Full Train L1 + Locked v4c Family Smoothing\n")
    lines.append("## Method\n")
    lines.append("```text")
    lines.append("Train on full original train split.")
    lines.append("Use locked family-aware smoothing matrix derived by F1e1a_v4c.")
    lines.append("Validation is evaluated once at the end.")
    lines.append("Validation is not used for early stopping or matrix selection.")
    lines.append("```")

    lines.append("\n## Target mapping info\n")
    lines.append("```json")
    lines.append(json.dumps(target_info, indent=2, default=str))
    lines.append("```")

    lines.append("\n## Final metrics\n")
    rows = []
    for split, m in final_metrics.items():
        if isinstance(m, dict):
            rows.append({"split": split, **m})
    if rows:
        lines.append(pd.DataFrame(rows).to_markdown(index=False))

    lines.append("\n## Config summary\n")
    lines.append("```json")
    lines.append(json.dumps(config, indent=2, default=str)[:6000])
    lines.append("```")
    (out_dir / "F1e1b_report.md").write_text("\n".join(lines), encoding="utf-8")


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
    ap.add_argument("--val-raw", default="01_split/val_raw.csv")
    ap.add_argument("--base-config", default="03_outputs/06_model/config.json")
    ap.add_argument("--v4-dir", default="05_test/outputs/F1e1a_v4_clean_calibration_family_smoothing")
    ap.add_argument("--matrix-csv", default="05_test/outputs/F1e1a_v4c_prob_only_calibration_family_matrix/F1e1a_v4c_locked_family_smoothing_matrix_PROB_ONLY_CALIBRATION_DERIVED.csv")
    ap.add_argument("--out-dir", default="05_test/outputs/F1e1b_full_l1_v4c_family_smoothing")
    ap.add_argument("--combined-zip", default="05_test/outputs/F1e1b_full_l1_v4c_family_smoothing.zip")
    ap.add_argument("--class-names", default="Benign,Ransomware,Spyware,Trojan")

    ap.add_argument("--epochs", type=int, default=-1, help="if -1, use best calibration epoch from v4 history")
    ap.add_argument("--fallback-epochs", type=int, default=49)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-epochs", type=int, default=8)
    ap.add_argument("--min-lr-ratio", type=float, default=0.05)
    ap.add_argument("--grad-clip-norm", type=float, default=1.0)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--no-class-weights", action="store_true")

    ap.add_argument("--hidden-dim", type=int, default=None)
    ap.add_argument("--num-heads", type=int, default=None)
    ap.add_argument("--classifier-hidden-dim", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--classifier-dropout", type=float, default=None)
    args = ap.parse_args()

    root = repo_root_from_here()
    out_dir = resolve_path(args.out_dir, root)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = resolve_path(args.combined_zip, root)

    seed_everything(args.seed)
    class_names = parse_list(args.class_names) or DEFAULT_CLASS_NAMES
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    torch.backends.cudnn.benchmark = True

    v4_dir = resolve_path(args.v4_dir, root)
    if args.epochs == -1:
        epochs, epoch_source = find_default_epochs(v4_dir, args.fallback_epochs)
    else:
        epochs, epoch_source = int(args.epochs), "explicit command line"

    log(f"root={root}")
    log(f"out_dir={out_dir}")
    log(f"device={device}")
    log(f"epochs={epochs} ({epoch_source})")
    log("Validation is not used during training. It will be evaluated once at the end.")

    ds, ds_info = load_dataset(
        resolve_path(args.dataset_npz, root),
        resolve_path(args.train_raw, root),
        resolve_path(args.val_raw, root),
        class_names,
    )
    mat = load_family_matrix(resolve_path(args.matrix_csv, root), class_names)
    soft_train, target_info = make_soft_targets(ds["train"], mat, class_names)
    (out_dir / "F1e1b_target_mapping_info.json").write_text(json.dumps(target_info, indent=2, default=str), encoding="utf-8")

    cfg = safe_json(resolve_path(args.base_config, root))
    model_config = infer_model_config(cfg, ds_info, class_names, args)
    model = build_model(root, model_config).to(device)

    train_dataset = SoftTargetDataset(ds["train"], soft_train)
    train_eval_dataset = SoftTargetDataset(ds["train"], None)
    val_dataset = SoftTargetDataset(ds["val"], None)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device=="cuda"), drop_last=False)
    train_eval_loader = DataLoader(train_eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device=="cuda"))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device=="cuda"))

    class_weights = None if args.no_class_weights else compute_class_weights(ds["train"]["y"], len(class_names)).to(device)
    criterion = WeightedSoftTargetCE(class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and args.amp))

    config = {
        "experiment": "F1e1b_full_l1_v4c_family_smoothing",
        "methodology": {
            "train_split": "full original train",
            "validation_used_for_training_or_matrix": False,
            "validation_early_stopping": False,
            "validation_eval": "once_at_end",
            "matrix_source": str(resolve_path(args.matrix_csv, root)),
            "epoch_source": epoch_source,
            "fake_data_used": False,
        },
        "dataset_info": ds_info,
        "model_config": model_config,
        "training_args": vars(args) | {"resolved_epochs": epochs, "epoch_source": epoch_source},
        "loss": {
            "name": "WeightedSoftTargetCE",
            "class_weights": None if class_weights is None else class_weights.detach().cpu().numpy().tolist(),
        },
        "target_mapping": target_info,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

    history = []
    for epoch in range(1, epochs + 1):
        lr = warmup_cosine_lr(epoch, args.lr, epochs, args.warmup_epochs, args.min_lr_ratio)
        for g in optimizer.param_groups:
            g["lr"] = lr
        model.train()
        total_loss, n_seen = 0.0, 0
        t0 = time.time()
        for tokens, values, y, targets in train_loader:
            tokens = tokens.to(device, non_blocking=True)
            values = values.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda" and args.amp)):
                logits = model(tokens, values)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            if args.grad_clip_norm and args.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            bs = int(tokens.shape[0])
            total_loss += float(loss.detach().cpu().item()) * bs
            n_seen += bs

        # Only train metrics during training. No val during training.
        ytr, ptr, _ = predict(model, train_eval_loader, device)
        mt = metrics(ytr, ptr)
        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": total_loss / max(1, n_seen),
            "train_accuracy": mt["accuracy"],
            "train_macro_f1": mt["macro_f1"],
            "train_weighted_f1": mt["weighted_f1"],
            "seconds": time.time() - t0,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        log(f"epoch {epoch:03d}/{epochs} loss={row['train_loss']:.5f} train_f1={row['train_macro_f1']:.6f}")

    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": model_config,
        "config": config,
    }, out_dir / "final_model.pt")

    # Final train and val evaluation.
    ytr, ptr, probtr = predict(model, train_eval_loader, device)
    yva, pva, probva = predict(model, val_loader, device)

    train_metrics = metrics(ytr, ptr)
    val_metrics = metrics(yva, pva)
    final_metrics = {"train": train_metrics, "val": val_metrics, "gap_train_minus_val_macro_f1": train_metrics["macro_f1"] - val_metrics["macro_f1"]}
    (out_dir / "final_metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")

    for split_name, y, pred, probs, split_ds in [
        ("train", ytr, ptr, probtr, ds["train"]),
        ("val", yva, pva, probva, ds["val"]),
    ]:
        rep = classification_report(y, pred, labels=list(range(len(class_names))), target_names=class_names, output_dict=True, zero_division=0)
        (out_dir / f"{split_name}_classification_report_final.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
        cm = confusion_matrix(y, pred, labels=list(range(len(class_names))))
        pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(out_dir / f"{split_name}_confusion_matrix_final.csv")
        make_pred_df(split_ds, y, pred, probs, class_names).to_csv(out_dir / f"{split_name}_predictions_final.csv", index=False)

    write_report(out_dir, config, target_info, final_metrics, class_names)
    zip_dir(out_dir, zip_path)

    log("Final metrics:")
    print(json.dumps(final_metrics, indent=2), flush=True)
    log(f"zip={zip_path}")
    log("DONE")


if __name__ == "__main__":
    main()
