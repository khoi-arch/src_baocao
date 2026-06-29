#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F1e1c Calibration-Only Gamma Search for v4c Family Smoothing

Purpose
-------
Search a scalar gamma for the locked v4c family smoothing matrix using only
train_inner/calibration data.

No validation usage.

Protocol
--------
For gamma in a fixed grid:

    target_gamma = one_hot + gamma * (target_v4c - one_hot)

Train temporary L1 model from scratch on train_inner.
Evaluate calibration once after fixed epochs.
Choose gamma by calibration macro-F1.
Export locked scaled matrix for the next full-train experiment.

Why this is needed
------------------
F1e1b with gamma=1.0 reduced train-val gap but lowered validation macro-F1,
indicating the smoothing strength may be too strong. F1e1c selects the strength
using calibration only, not validation.

Outputs
-------
- F1e1c_gamma_results.csv
- F1e1c_best_gamma.json
- F1e1c_locked_scaled_matrix_CALIBRATION_SELECTED.csv
- F1e1c_report.md
- per-gamma train/calibration reports and confusion matrices
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


def log(msg: str):
    print(f"[F1e1c] {msg}", flush=True)


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


def parse_float_grid(s: str) -> List[float]:
    out = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(float(x))
    # unique, preserve sorted increasing for readability
    return sorted(set(out))


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_json(p: Path) -> Dict[str, Any]:
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
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
    required = ["X_train_bin", "X_train_offset", "y_train"]
    missing = [k for k in required if k not in data.files]
    if missing:
        raise KeyError(f"dataset npz missing keys: {missing}")

    feature_names = [str(x) for x in np.asarray(data["feature_names"]).tolist()] if "feature_names" in data.files else [f"f{i}" for i in range(np.asarray(data['X_train_bin']).shape[1])]
    num_bins = int(np.asarray(data["num_bins"]).reshape(-1)[0]) if "num_bins" in data.files else 512

    Xtr_bin = np.asarray(data["X_train_bin"], dtype=np.int64)
    Xtr_off = np.asarray(data["X_train_offset"], dtype=np.float32)
    ytr = np.asarray(data["y_train"], dtype=np.int64).reshape(-1)

    # val is not used for gamma selection, but keep metadata only if present.
    n_train = len(ytr)
    if not train_raw.exists():
        raise FileNotFoundError(f"train raw csv not found: {train_raw}")
    df = pd.read_csv(train_raw)
    if len(df) != n_train:
        raise ValueError(f"train raw rows={len(df)} expected={n_train}; cannot align labels safely")

    feat_cols = [c for c in feature_names if c in df.columns]
    if len(feat_cols) != len(feature_names):
        exclude = {"label_L1","label_L2","label_L3","Label_L1","Label_L2","Label_L3","Class","Category","class","category","Family","family"}
        feat_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])][:len(feature_names)]
    if len(feat_cols) != len(feature_names):
        raise ValueError(f"raw feature mismatch: got {len(feat_cols)}, expected {len(feature_names)}")
    R = df[feat_cols].to_numpy(dtype=np.float32)

    l2_col = find_label_col(df, "L2")
    l3_col = find_label_col(df, "L3")
    l2 = df[l2_col].map(clean).to_numpy() if l2_col else np.array([""] * n_train, dtype=object)
    l3 = df[l3_col].map(clean).to_numpy() if l3_col else np.array([""] * n_train, dtype=object)
    y_l2 = np.array([class_names[int(i)] if int(i) < len(class_names) else str(i) for i in ytr], dtype=object)
    if pd.Series(l2).isin(class_names).mean() < 0.80:
        l2 = y_l2
    if (pd.Series(l3).map(clean) == "").mean() > 0.80:
        l3 = l2.copy()

    mn = np.nanmin(R, axis=0, keepdims=True)
    mx = np.nanmax(R, axis=0, keepdims=True)
    den = mx - mn
    den[den < 1e-8] = 1.0
    Xtr_raw = np.clip((R - mn) / den, 0.0, 1.0).astype(np.float32)

    values = np.stack([Xtr_off.astype(np.float32), Xtr_raw.astype(np.float32), np.ones_like(Xtr_off, dtype=np.float32)], axis=-1).astype(np.float32)

    ds = {
        "tokens": Xtr_bin,
        "values": values,
        "y": ytr,
        "label_L2": l2,
        "label_L3": l3,
    }
    info = {
        "dataset_npz": str(dataset_npz),
        "train_raw": str(train_raw),
        "val_raw_not_used": str(val_raw),
        "keys": list(data.files),
        "n_train": int(n_train),
        "n_features": int(Xtr_bin.shape[1]),
        "num_bins": int(num_bins),
        "label_L2_col": l2_col,
        "label_L3_col": l3_col,
        "L3_unique_count": int(pd.Series(l3).nunique()),
        "values_candidate": "offset_raw_one",
    }
    return ds, info


def load_or_create_split(v4_dir: Path, ds: Dict[str, np.ndarray], calib_size: float, seed: int):
    p = v4_dir / "F1e1a_v4_split_indices.npz"
    if p.exists():
        z = np.load(p, allow_pickle=True)
        keys = set(z.files)
        train_key = None
        calib_key = None
        for k in ["train_inner_idx", "train_inner_indices", "inner_train_idx", "train_idx"]:
            if k in keys:
                train_key = k
                break
        for k in ["calibration_idx", "calibration_indices", "calib_idx", "calib_indices", "val_idx"]:
            if k in keys:
                calib_key = k
                break
        if train_key and calib_key:
            train_idx = np.asarray(z[train_key], dtype=np.int64)
            calib_idx = np.asarray(z[calib_key], dtype=np.int64)
            return train_idx, calib_idx, {
                "source": str(p),
                "train_key": train_key,
                "calib_key": calib_key,
                "n_train_inner": int(len(train_idx)),
                "n_calibration": int(len(calib_idx)),
            }
        else:
            log(f"split file found but keys not recognized: {z.files}; fallback to stratified split")

    labels = np.array([f"{clean(a)}::{clean(b)}" for a, b in zip(ds["label_L2"], ds["label_L3"])], dtype=object)
    counts = pd.Series(labels).value_counts()
    if counts.min() < 2:
        labels = ds["label_L2"]
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=calib_size, random_state=seed)
    train_idx, calib_idx = next(splitter.split(np.zeros(len(ds["y"])), labels))
    return train_idx.astype(np.int64), calib_idx.astype(np.int64), {
        "source": "fallback StratifiedShuffleSplit inside F1e1c",
        "calib_size": float(calib_size),
        "seed": int(seed),
        "n_train_inner": int(len(train_idx)),
        "n_calibration": int(len(calib_idx)),
    }


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
    for c in [f"target_{x}" for x in class_names]:
        mat[c] = pd.to_numeric(mat[c], errors="coerce").fillna(0.0)
    mat["target_sum_check"] = mat[[f"target_{c}" for c in class_names]].sum(axis=1)
    bad = np.abs(mat["target_sum_check"] - 1.0) > 1e-4
    if bad.any():
        raise ValueError("matrix targets do not sum to 1:\n" + mat[bad][["true_L2","true_L3","target_sum_check"]].to_string(index=False))
    return mat


def make_soft_targets(ds: Dict[str, np.ndarray], mat: pd.DataFrame, class_names: List[str], gamma: float):
    target_cols = [f"target_{c}" for c in class_names]
    key_to_target = {}
    for _, r in mat.iterrows():
        key_to_target[(clean(r["true_L2"]), clean(r["true_L3"]))] = np.array([float(r[c]) for c in target_cols], dtype=np.float32)

    n = len(ds["y"])
    out = np.zeros((n, len(class_names)), dtype=np.float32)
    matched = 0
    fallback_onehot = 0
    missing = {}
    for i in range(n):
        y = int(ds["y"][i])
        one = np.zeros(len(class_names), dtype=np.float32)
        one[y] = 1.0
        key = (clean(ds["label_L2"][i]), clean(ds["label_L3"][i]))
        if key in key_to_target:
            base = key_to_target[key]
            matched += 1
        else:
            base = one
            fallback_onehot += 1
            missing[key] = missing.get(key, 0) + 1
        tgt = one + float(gamma) * (base - one)
        # numeric safety
        tgt = np.clip(tgt, 0.0, 1.0)
        s = tgt.sum()
        if s <= 0:
            tgt = one
        else:
            tgt = tgt / s
        out[i] = tgt
    return out, {
        "gamma": float(gamma),
        "matched": int(matched),
        "fallback_onehot": int(fallback_onehot),
        "missing_top": sorted(missing.items(), key=lambda kv: kv[1], reverse=True)[:10],
    }


class ArrayDataset(Dataset):
    def __init__(self, ds: Dict[str, np.ndarray], soft_targets: Optional[np.ndarray] = None):
        self.tokens = ds["tokens"]
        self.values = ds["values"]
        self.y = ds["y"]
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


def subset_dict(ds: Dict[str, np.ndarray], idx: np.ndarray) -> Dict[str, np.ndarray]:
    return {k: np.asarray(v)[idx] for k, v in ds.items()}


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
    mod = load_module_from_path(f"_f1e1c_model_{time.time_ns()}", root / "02_src" / "06_model.py")
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
            self.class_weights = None
        else:
            self.register_buffer("class_weights", class_weights.float())

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        logp = torch.log_softmax(logits, dim=1)
        if self.class_weights is not None:
            target = target * self.class_weights.view(1, -1)
        return -(target * logp).sum(dim=1).mean()


def warmup_cosine_lr(epoch: int, base_lr: float, epochs: int, warmup_epochs: int, min_lr_ratio: float) -> float:
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return base_lr * epoch / warmup_epochs
    if epochs <= warmup_epochs:
        return base_lr
    progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1 - min_lr_ratio) * cosine)


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


def train_one_gamma(
    gamma: float,
    root: Path,
    out_dir: Path,
    cfg: Dict[str, Any],
    ds_info: Dict[str, Any],
    class_names: List[str],
    train_inner: Dict[str, np.ndarray],
    calibration: Dict[str, np.ndarray],
    train_soft: np.ndarray,
    args,
    device: str,
):
    seed_everything(args.seed)
    model_config = infer_model_config(cfg, ds_info, class_names, args)
    model = build_model(root, model_config).to(device)

    train_dataset = ArrayDataset(train_inner, train_soft)
    train_eval_dataset = ArrayDataset(train_inner, None)
    calib_dataset = ArrayDataset(calibration, None)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device=="cuda"), drop_last=False)
    train_eval_loader = DataLoader(train_eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device=="cuda"))
    calib_loader = DataLoader(calib_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device=="cuda"))

    cw = None if args.no_class_weights else compute_class_weights(train_inner["y"], len(class_names)).to(device)
    criterion = WeightedSoftTargetCE(cw)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and args.amp))

    rows = []
    gamma_tag = str(gamma).replace(".", "p")
    for epoch in range(1, args.epochs + 1):
        lr = warmup_cosine_lr(epoch, args.lr, args.epochs, args.warmup_epochs, args.min_lr_ratio)
        for g in optimizer.param_groups:
            g["lr"] = lr
        model.train()
        total_loss, n_seen = 0.0, 0
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

        # train_inner metrics only during training for progress; calibration only at final by default.
        ytr, ptr, _ = predict(model, train_eval_loader, device)
        mt = metrics(ytr, ptr)
        row = {
            "gamma": gamma,
            "epoch": epoch,
            "lr": lr,
            "train_loss": total_loss / max(1, n_seen),
            "train_inner_accuracy": mt["accuracy"],
            "train_inner_macro_f1": mt["macro_f1"],
            "train_inner_weighted_f1": mt["weighted_f1"],
        }
        rows.append(row)
        log(f"gamma={gamma:.4f} epoch {epoch:03d}/{args.epochs} loss={row['train_loss']:.5f} train_inner_f1={row['train_inner_macro_f1']:.6f}")

    ytr, ptr, probtr = predict(model, train_eval_loader, device)
    yc, pc, probc = predict(model, calib_loader, device)
    mt = metrics(ytr, ptr)
    mc = metrics(yc, pc)

    # save detailed reports for each gamma
    gdir = out_dir / f"gamma_{gamma_tag}"
    gdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(gdir / "history.csv", index=False)

    for split_name, y, pred in [("train_inner", ytr, ptr), ("calibration", yc, pc)]:
        rep = classification_report(y, pred, labels=list(range(len(class_names))), target_names=class_names, output_dict=True, zero_division=0)
        (gdir / f"{split_name}_classification_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
        cm = confusion_matrix(y, pred, labels=list(range(len(class_names))))
        pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(gdir / f"{split_name}_confusion_matrix.csv")

    # Do not save all model checkpoints by default to keep zip small.
    if args.save_models:
        torch.save({
            "gamma": gamma,
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
        }, gdir / "model.pt")

    result = {
        "gamma": float(gamma),
        "train_inner_accuracy": mt["accuracy"],
        "train_inner_macro_f1": mt["macro_f1"],
        "train_inner_weighted_f1": mt["weighted_f1"],
        "calibration_accuracy": mc["accuracy"],
        "calibration_macro_f1": mc["macro_f1"],
        "calibration_weighted_f1": mc["weighted_f1"],
        "gap_train_inner_minus_calibration_macro_f1": mt["macro_f1"] - mc["macro_f1"],
    }
    return result


def scale_matrix(mat: pd.DataFrame, gamma: float, class_names: List[str]) -> pd.DataFrame:
    out = mat.copy()
    target_cols = [f"target_{c}" for c in class_names]
    for idx, r in out.iterrows():
        true_l2 = clean(r["true_L2"])
        if true_l2 not in class_names:
            continue
        one = np.zeros(len(class_names), dtype=np.float64)
        one[class_names.index(true_l2)] = 1.0
        base = np.array([float(r[c]) for c in target_cols], dtype=np.float64)
        scaled = one + float(gamma) * (base - one)
        scaled = np.clip(scaled, 0.0, 1.0)
        scaled = scaled / max(1e-12, scaled.sum())
        for c, v in zip(target_cols, scaled):
            out.loc[idx, c] = float(v)
    out["selected_gamma"] = float(gamma)
    out["target_sum_check"] = out[target_cols].sum(axis=1)
    return out


def write_report(out_dir: Path, config: Dict[str, Any], results: pd.DataFrame, best: Dict[str, Any]):
    lines = []
    lines.append("# F1e1c Calibration-Only Gamma Search\n")
    lines.append("## Protocol\n")
    lines.append("```text")
    lines.append("For each gamma, train temporary L1 from scratch on train_inner.")
    lines.append("Evaluate calibration after fixed epochs.")
    lines.append("Choose gamma by calibration macro-F1.")
    lines.append("Validation is not used.")
    lines.append("```")

    lines.append("\n## Gamma results\n")
    lines.append(results.to_markdown(index=False))

    lines.append("\n## Selected gamma\n")
    lines.append("```json")
    lines.append(json.dumps(best, indent=2))
    lines.append("```")

    lines.append("\n## Config summary\n")
    lines.append("```json")
    lines.append(json.dumps(config, indent=2, default=str)[:8000])
    lines.append("```")

    lines.append("\n## Next step\n")
    lines.append("```text")
    lines.append("If selected gamma is > 0 and calibration improves over gamma=0,")
    lines.append("run F1e1d/F1e1b-style full-train experiment using the locked scaled matrix.")
    lines.append("If gamma=0 wins, family smoothing should be rejected for now.")
    lines.append("```")
    (out_dir / "F1e1c_report.md").write_text("\n".join(lines), encoding="utf-8")


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
    ap.add_argument("--out-dir", default="05_test/outputs/F1e1c_calibration_gamma_search")
    ap.add_argument("--combined-zip", default="05_test/outputs/F1e1c_calibration_gamma_search.zip")
    ap.add_argument("--class-names", default="Benign,Ransomware,Spyware,Trojan")
    ap.add_argument("--gamma-grid", default="0.0,0.125,0.25,0.5,0.75,1.0")

    ap.add_argument("--epochs", type=int, default=49)
    ap.add_argument("--calib-size", type=float, default=0.20)
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
    ap.add_argument("--save-models", action="store_true")

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
    gammas = parse_float_grid(args.gamma_grid)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    torch.backends.cudnn.benchmark = True

    v4_dir = resolve_path(args.v4_dir, root)

    log(f"root={root}")
    log(f"out_dir={out_dir}")
    log(f"device={device}")
    log(f"gamma_grid={gammas}")
    log("Validation is not used in F1e1c.")

    ds, ds_info = load_dataset(resolve_path(args.dataset_npz, root), resolve_path(args.train_raw, root), resolve_path(args.val_raw, root), class_names)
    train_idx, calib_idx, split_info = load_or_create_split(v4_dir, ds, args.calib_size, args.seed)
    train_inner = subset_dict(ds, train_idx)
    calibration = subset_dict(ds, calib_idx)

    mat = load_family_matrix(resolve_path(args.matrix_csv, root), class_names)
    cfg = safe_json(resolve_path(args.base_config, root))

    config = {
        "experiment": "F1e1c_calibration_gamma_search",
        "methodology": {
            "validation_used": False,
            "gamma_selected_by": "calibration_macro_f1",
            "epochs_fixed": int(args.epochs),
            "matrix_source": str(resolve_path(args.matrix_csv, root)),
        },
        "dataset_info": ds_info,
        "split_info": split_info,
        "gamma_grid": gammas,
        "args": vars(args),
        "model_config": infer_model_config(cfg, ds_info, class_names, args),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

    pd.DataFrame({
        "split": ["train_inner", "calibration"],
        "n": [len(train_idx), len(calib_idx)],
    }).to_csv(out_dir / "split_summary.csv", index=False)

    results = []
    target_infos = {}
    for gamma in gammas:
        full_soft, target_info = make_soft_targets(ds, mat, class_names, gamma)
        target_infos[str(gamma)] = target_info
        train_soft = full_soft[train_idx]
        res = train_one_gamma(gamma, root, out_dir, cfg, ds_info, class_names, train_inner, calibration, train_soft, args, device)
        results.append(res)
        pd.DataFrame(results).to_csv(out_dir / "F1e1c_gamma_results_partial.csv", index=False)
        log(f"gamma={gamma:.4f} final calibration_macro_f1={res['calibration_macro_f1']:.6f} gap={res['gap_train_inner_minus_calibration_macro_f1']:.6f}")

    results_df = pd.DataFrame(results).sort_values(["calibration_macro_f1", "calibration_accuracy"], ascending=False).reset_index(drop=True)
    results_df.to_csv(out_dir / "F1e1c_gamma_results.csv", index=False)

    best_row = results_df.iloc[0].to_dict()
    best_gamma = float(best_row["gamma"])
    scaled = scale_matrix(mat, best_gamma, class_names)
    scaled.to_csv(out_dir / "F1e1c_locked_scaled_matrix_CALIBRATION_SELECTED.csv", index=False)

    best = {
        "selected_gamma": best_gamma,
        "selection_metric": "calibration_macro_f1",
        "selected_row": best_row,
        "validation_used": False,
        "target_info_by_gamma": target_infos,
        "locked_matrix_file": "F1e1c_locked_scaled_matrix_CALIBRATION_SELECTED.csv",
    }
    (out_dir / "F1e1c_best_gamma.json").write_text(json.dumps(best, indent=2, default=str), encoding="utf-8")
    (out_dir / "F1e1c_leakage_policy.md").write_text(
        "# F1e1c Leakage Policy\n\n"
        "F1e1c does not use validation. It uses train_inner/calibration only.\n"
        "Gamma is selected by calibration macro-F1. The selected scaled matrix is locked before any full-train validation evaluation.\n",
        encoding="utf-8",
    )

    write_report(out_dir, config, results_df, best)
    zip_dir(out_dir, zip_path)

    log("Gamma results:")
    print(results_df.to_string(index=False), flush=True)
    log(f"selected_gamma={best_gamma}")
    log(f"zip={zip_path}")
    log("DONE")


if __name__ == "__main__":
    main()
