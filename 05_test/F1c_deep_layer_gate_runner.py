#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F1c Deep-Layer Residual Gate Runner

Purpose
-------
Audit F1b located the overfit locus in deeper Transformer interaction layers:
- embedding/input_proj gap is small
- encoder.layers.0 gap is still small/moderate
- encoder.layers.1 and encoder.layers.2 create most of the train/val gap
- classifier/logit boundary is not the main locus

F1c is NOT a hyperparameter sweep.
It keeps the original 3-layer D3 architecture but controls only the deeper layers:
    encoder.layers.1
    encoder.layers.2

Mechanism
---------
Wrap deeper layer l with:

    y = x + alpha_l * (Layer_l(x) - x)

where alpha_l is a learnable gate in [0,1].

This keeps the high-ceiling depth path available, but prevents deeper layers from
dominating immediately and overfitting train-specific subtype interactions.

Two diagnostic variants are run:
1) conservative: alpha_init=0.05, gate_reg=1e-3
   - starts close to L1 behavior
   - allows deeper layers only if useful

2) moderate: alpha_init=0.25, gate_reg=5e-4
   - preserves more depth signal
   - tests whether higher representation ceiling can be kept without gap explosion

Outputs
-------
- full train/val metrics every epoch
- classification reports/confusion matrices/predictions
- gate values per epoch
- summary CSV/MD
- raw runs zip + logs zip + summary zip + ALL zip

This file is self-contained and does not modify official 02_src files.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import sys
import threading
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


BASE512_REF = {
    "name": "official_D3_base512",
    "train_macro_f1": 0.910253,
    "val_macro_f1": 0.810094,
    "gap_macro_f1": 0.100158,
}

L1_REF = {
    "name": "F1a2_L1_anchor",
    "train_macro_f1": 0.911431,
    "val_macro_f1": 0.814224,
    "gap_macro_f1": 0.097207,
}

VARIANTS = [
    {
        "name": "F1c_Gate12_conservative_alpha005_reg1e3",
        "description": "3-layer D3; gate encoder.layers.1/2; alpha_init=0.05; gate_reg=1e-3",
        "gate_layers": "1,2",
        "gate_alpha_init": 0.05,
        "gate_reg_lambda": 1e-3,
    },
    {
        "name": "F1c_Gate12_moderate_alpha025_reg5e4",
        "description": "3-layer D3; gate encoder.layers.1/2; alpha_init=0.25; gate_reg=5e-4",
        "gate_layers": "1,2",
        "gate_alpha_init": 0.25,
        "gate_reg_lambda": 5e-4,
    },
]


def log(msg: str) -> None:
    print(f"[F1c] {msg}", flush=True)


def repo_root_from_here() -> Path:
    p = Path(__file__).resolve()
    if p.parent.name == "05_test":
        return p.parents[1]
    return Path.cwd().resolve()


def resolve_path(p: str | Path, root: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (root / p).resolve()


def seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def cfg_get(cfg: Dict[str, Any], k: str, default):
    if k in cfg:
        return cfg[k]
    for dname in ["model", "model_config", "training", "args"]:
        d = cfg.get(dname, {})
        if isinstance(d, dict) and k in d:
            return d[k]
    return default


def infer_model_config_from_base(base_config_path: Path, dataset_info: Dict[str, Any], num_classes: int) -> Dict[str, Any]:
    cfg = load_json(base_config_path) if base_config_path.exists() else {}
    return {
        "num_bins": int(cfg_get(cfg, "num_bins", cfg_get(cfg, "K", dataset_info["num_bins"]))),
        "n_features": int(cfg_get(cfg, "n_features", cfg_get(cfg, "num_features", dataset_info["n_features"]))),
        "num_classes": int(cfg_get(cfg, "num_classes", num_classes)),
        "value_dim": int(cfg_get(cfg, "value_dim", 32)),
        "feature_dim": int(cfg_get(cfg, "feature_dim", 32)),
        "hidden_dim": int(cfg_get(cfg, "hidden_dim", 128)),
        "num_layers": int(cfg_get(cfg, "num_layers", 3)),
        "num_heads": int(cfg_get(cfg, "num_heads", 4)),
        "dropout": float(cfg_get(cfg, "dropout", 0.1)),
        "classifier_hidden_dim": int(cfg_get(cfg, "classifier_hidden_dim", 128)),
        "classifier_dropout": float(cfg_get(cfg, "classifier_dropout", 0.1)),
        "gate_init": float(cfg_get(cfg, "gate_init", 0.0)),
    }


def load_dataset(dataset_npz: Path, train_raw: Path, val_raw: Path) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Any]]:
    data = np.load(dataset_npz, allow_pickle=True)
    required = ["X_train_bin", "X_train_offset", "y_train", "X_val_bin", "X_val_offset", "y_val"]
    missing = [k for k in required if k not in data.files]
    if missing:
        raise KeyError(f"dataset npz missing keys: {missing}")

    Xtr_bin = np.asarray(data["X_train_bin"], dtype=np.int64)
    Xtr_off = np.asarray(data["X_train_offset"], dtype=np.float32)
    ytr = np.asarray(data["y_train"], dtype=np.int64).reshape(-1)

    Xva_bin = np.asarray(data["X_val_bin"], dtype=np.int64)
    Xva_off = np.asarray(data["X_val_offset"], dtype=np.float32)
    yva = np.asarray(data["y_val"], dtype=np.int64).reshape(-1)

    feature_names = [str(x) for x in np.asarray(data["feature_names"]).tolist()] if "feature_names" in data.files else [f"f{i}" for i in range(Xtr_bin.shape[1])]
    num_bins = int(np.asarray(data["num_bins"]).reshape(-1)[0]) if "num_bins" in data.files else 512

    def load_raw_csv(path: Path) -> Tuple[Optional[np.ndarray], List[str]]:
        if not path.exists():
            return None, []
        df = pd.read_csv(path)
        cols = [c for c in feature_names if c in df.columns]
        if len(cols) != len(feature_names):
            exclude = {
                "label", "Label", "Class", "Category", "class", "category",
                "label_L1", "label_L2", "label_L3", "label_l1", "label_l2", "label_l3",
            }
            cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])][:len(feature_names)]
        if len(cols) != len(feature_names):
            raise ValueError(f"raw csv feature columns mismatch: got {len(cols)}, expected {len(feature_names)}")
        return df[cols].to_numpy(dtype=np.float32), cols

    raw_tr, raw_cols = load_raw_csv(train_raw)
    raw_va, _ = load_raw_csv(val_raw)

    if raw_tr is not None and raw_va is not None:
        mn = np.nanmin(raw_tr, axis=0, keepdims=True)
        mx = np.nanmax(raw_tr, axis=0, keepdims=True)
        den = mx - mn
        den[den < 1e-8] = 1.0
        Xtr_raw = np.clip((raw_tr - mn) / den, 0.0, 1.0).astype(np.float32)
        Xva_raw = np.clip((raw_va - mn) / den, 0.0, 1.0).astype(np.float32)
        raw_info = {"source": "train_minmax_raw_csv", "available": True, "columns_preview": raw_cols[:10], "num_columns": len(raw_cols)}
    else:
        # Fallback only. The audit showed official-compatible candidate uses raw_scaled.
        # If raw CSV is absent, this run is still possible but not final-quality.
        Xtr_raw = Xtr_off.astype(np.float32)
        Xva_raw = Xva_off.astype(np.float32)
        raw_info = {"source": "fallback_offset_as_raw_scaled", "available": False}

    def make_values(bin_arr: np.ndarray, off_arr: np.ndarray, raw_arr: np.ndarray) -> np.ndarray:
        # F1b V3 selected `offset_raw_one`, reproducing official val macro-F1 within +0.000239.
        mask = np.ones_like(off_arr, dtype=np.float32)
        return np.stack([off_arr.astype(np.float32), raw_arr.astype(np.float32), mask], axis=-1).astype(np.float32)

    ds = {
        "train": {
            "tokens": Xtr_bin,
            "values": make_values(Xtr_bin, Xtr_off, Xtr_raw),
            "y": ytr,
        },
        "val": {
            "tokens": Xva_bin,
            "values": make_values(Xva_bin, Xva_off, Xva_raw),
            "y": yva,
        },
    }

    info = {
        "dataset_npz": str(dataset_npz),
        "keys": list(data.files),
        "n_train": int(len(ytr)),
        "n_val": int(len(yva)),
        "n_features": int(Xtr_bin.shape[1]),
        "num_bins": int(num_bins),
        "num_classes": int(len(np.unique(ytr))),
        "feature_names_preview": feature_names[:10],
        "values_candidate": "offset_raw_one",
        "raw_info": raw_info,
    }
    return ds, info


class TokenValueDataset(Dataset):
    def __init__(self, split_data: Dict[str, np.ndarray]):
        self.tokens = split_data["tokens"]
        self.values = split_data["values"]
        self.y = split_data["y"]

    def __len__(self) -> int:
        return int(len(self.y))

    def __getitem__(self, idx: int):
        return (
            torch.as_tensor(self.tokens[idx], dtype=torch.long),
            torch.as_tensor(self.values[idx], dtype=torch.float32),
            torch.as_tensor(self.y[idx], dtype=torch.long),
        )


def alpha_to_logit(alpha: float) -> float:
    alpha = float(np.clip(alpha, 1e-4, 1 - 1e-4))
    return math.log(alpha / (1 - alpha))


class ResidualGatedLayer(nn.Module):
    """
    Wrap a TransformerEncoderLayer-like module:
        y_raw = layer(x)
        y = x + sigmoid(gate_logit) * (y_raw - x)

    This gates the full layer transformation, not a classifier or embedding change.
    """

    def __init__(self, layer: nn.Module, alpha_init: float, name: str):
        super().__init__()
        self.layer = layer
        self.gate_logit = nn.Parameter(torch.tensor(alpha_to_logit(alpha_init), dtype=torch.float32))
        self.name = name

    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logit)

    def forward(self, src, *args, **kwargs):
        y = self.layer(src, *args, **kwargs)
        a = self.alpha().to(dtype=y.dtype, device=y.device)
        return src + a * (y - src)


def apply_deep_layer_gates(model: nn.Module, gate_layers: List[int], alpha_init: float) -> None:
    if not hasattr(model, "encoder") or not hasattr(model.encoder, "layers"):
        raise AttributeError("model.encoder.layers not found; cannot apply F1c deep-layer gates")
    layers = model.encoder.layers
    for idx in gate_layers:
        if idx < 0 or idx >= len(layers):
            raise IndexError(f"gate layer index {idx} out of range for encoder.layers length {len(layers)}")
        layers[idx] = ResidualGatedLayer(layers[idx], alpha_init=alpha_init, name=f"encoder.layers.{idx}")


def gate_values(model: nn.Module) -> Dict[str, float]:
    out = {}
    for name, m in model.named_modules():
        if isinstance(m, ResidualGatedLayer):
            out[name] = float(m.alpha().detach().cpu().item())
    return out


def gate_penalty(model: nn.Module) -> torch.Tensor:
    vals = []
    for m in model.modules():
        if isinstance(m, ResidualGatedLayer):
            vals.append(m.alpha() ** 2)
    if not vals:
        # Device will be fixed by caller if needed.
        return torch.tensor(0.0)
    return torch.stack(vals).sum()


def build_model(root: Path, model_config: Dict[str, Any], gate_layers: List[int], alpha_init: float) -> nn.Module:
    mod = load_module_from_path("_f1c_model_06_model", root / "02_src" / "06_model.py")
    cls = getattr(mod, "D3C2D3Transformer", None)
    if cls is None:
        raise RuntimeError("D3C2D3Transformer not found in 02_src/06_model.py")
    import inspect
    kwargs = {k: v for k, v in model_config.items() if k in inspect.signature(cls).parameters}
    model = cls(**kwargs)
    apply_deep_layer_gates(model, gate_layers=gate_layers, alpha_init=alpha_init)
    return model


def class_names_for(num_classes: int) -> List[str]:
    if num_classes == 4:
        return ["Benign", "Ransomware", "Spyware", "Trojan"]
    return [str(i) for i in range(num_classes)]


def compute_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y.astype(int), minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = len(y) / (num_classes * counts)
    return torch.as_tensor(weights, dtype=torch.float32)


def warmup_cosine_lr(epoch: int, base_lr: float, epochs: int, warmup_epochs: int, min_lr_ratio: float) -> float:
    # epoch is 1-indexed
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return base_lr * epoch / warmup_epochs
    if epochs <= warmup_epochs:
        return base_lr
    progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def set_optimizer_lr(optimizer, lr: float):
    for g in optimizer.param_groups:
        g["lr"] = lr


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    ys, preds, probs = [], [], []
    for tokens, values, y in loader:
        tokens = tokens.to(device, non_blocking=True)
        values = values.to(device, non_blocking=True)
        logits = model(tokens, values)
        p = torch.softmax(logits, dim=1)
        pred = p.argmax(dim=1)
        ys.append(y.cpu().numpy())
        preds.append(pred.detach().cpu().numpy())
        probs.append(p.detach().cpu().numpy())
    return np.concatenate(ys), np.concatenate(preds), np.concatenate(probs)


def metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def report_dict(y_true: np.ndarray, y_pred: np.ndarray, names: List[str]) -> Dict[str, Any]:
    labels = list(range(len(names)))
    return classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=names,
        output_dict=True,
        zero_division=0,
    )


def save_predictions(path: Path, y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray, names: List[str]):
    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "true_name": [names[int(x)] if int(x) < len(names) else str(x) for x in y_true],
        "pred_name": [names[int(x)] if int(x) < len(names) else str(x) for x in y_pred],
    })
    for i, name in enumerate(names):
        df[f"prob_{name}"] = probs[:, i]
    df.to_csv(path, index=False)


def train_one_variant(args, variant: Dict[str, Any]) -> None:
    root = repo_root_from_here()
    out_root = resolve_path(args.out_root, root)
    run_dir = out_root / f"Keff{args.K}" / variant["name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    torch.backends.cudnn.benchmark = True

    ds, ds_info = load_dataset(
        resolve_path(args.dataset_npz, root),
        resolve_path(args.train_raw, root),
        resolve_path(args.val_raw, root),
    )

    num_classes = int(ds_info["num_classes"])
    names = class_names_for(num_classes)

    model_config = infer_model_config_from_base(
        resolve_path(args.base_config, root),
        ds_info,
        num_classes=num_classes,
    )
    # Force official/base architecture except deep gates.
    model_config["num_layers"] = 3
    model_config["hidden_dim"] = int(args.hidden_dim)
    model_config["num_heads"] = int(args.num_heads)
    model_config["classifier_hidden_dim"] = int(args.classifier_hidden_dim)
    model_config["dropout"] = float(args.dropout)
    model_config["classifier_dropout"] = float(args.classifier_dropout)

    gate_layers = [int(x) for x in str(variant["gate_layers"]).split(",") if str(x).strip()]
    model = build_model(root, model_config, gate_layers=gate_layers, alpha_init=float(variant["gate_alpha_init"]))
    model.to(device)

    train_ds = TokenValueDataset(ds["train"])
    val_ds = TokenValueDataset(ds["val"])
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )
    train_eval_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )

    weights = compute_class_weights(ds["train"]["y"], num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and args.amp))

    best_val = -1.0
    best_epoch = -1
    best_row = None
    no_improve = 0
    history = []

    config = {
        "experiment": "F1c_deep_layer_gate",
        "variant": variant,
        "references": {"base512": BASE512_REF, "L1": L1_REF},
        "model_config": model_config,
        "dataset_info": ds_info,
        "training": vars(args),
        "gate_layers": gate_layers,
        "initial_gate_values": gate_values(model),
        "note": "3-layer D3 with residual gates on deeper Transformer layers; official files are not modified.",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

    log(f"[{variant['name']}] run_dir={run_dir}")
    log(f"[{variant['name']}] model_config={model_config}")
    log(f"[{variant['name']}] initial gates={gate_values(model)}")

    for epoch in range(1, args.epochs + 1):
        lr = warmup_cosine_lr(epoch, args.lr, args.epochs, args.warmup_epochs, args.min_lr_ratio)
        set_optimizer_lr(optimizer, lr)
        model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_gate = 0.0
        n_seen = 0

        t0 = time.time()
        for tokens, values, y in train_loader:
            tokens = tokens.to(device, non_blocking=True)
            values = values.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda" and args.amp)):
                logits = model(tokens, values)
                ce = criterion(logits, y)
                gp = gate_penalty(model).to(device)
                loss = ce + float(variant["gate_reg_lambda"]) * gp

            scaler.scale(loss).backward()
            if args.grad_clip_norm and args.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            bs = int(y.shape[0])
            total_loss += float(loss.detach().cpu().item()) * bs
            total_ce += float(ce.detach().cpu().item()) * bs
            total_gate += float(gp.detach().cpu().item()) * bs
            n_seen += bs

        train_y, train_pred, train_prob = predict(model, train_eval_loader, device)
        val_y, val_pred, val_prob = predict(model, val_loader, device)

        train_m = metrics_from_predictions(train_y, train_pred)
        val_m = metrics_from_predictions(val_y, val_pred)
        gap = train_m["macro_f1"] - val_m["macro_f1"]

        row = {
            "epoch": epoch,
            "lr": lr,
            "loss": total_loss / max(1, n_seen),
            "ce_loss": total_ce / max(1, n_seen),
            "gate_penalty": total_gate / max(1, n_seen),
            "train_accuracy": train_m["accuracy"],
            "train_macro_f1": train_m["macro_f1"],
            "train_weighted_f1": train_m["weighted_f1"],
            "val_accuracy": val_m["accuracy"],
            "val_macro_f1": val_m["macro_f1"],
            "val_weighted_f1": val_m["weighted_f1"],
            "macro_f1_gap_train_minus_val": gap,
            "seconds": time.time() - t0,
            **{f"gate_{k}": v for k, v in gate_values(model).items()},
        }
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

        improved = val_m["macro_f1"] > best_val + args.min_delta
        if improved:
            best_val = val_m["macro_f1"]
            best_epoch = epoch
            best_row = row
            no_improve = 0

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_macro_f1": best_val,
                "model_config": model_config,
                "variant": variant,
                "gate_values": gate_values(model),
            }, run_dir / "best_model.pt")

            # Save best reports/preds.
            (run_dir / "train_classification_report_best.json").write_text(
                json.dumps(report_dict(train_y, train_pred, names), indent=2), encoding="utf-8"
            )
            (run_dir / "val_classification_report_best.json").write_text(
                json.dumps(report_dict(val_y, val_pred, names), indent=2), encoding="utf-8"
            )
            pd.DataFrame(confusion_matrix(train_y, train_pred, labels=list(range(num_classes))), index=names, columns=names).to_csv(
                run_dir / "train_confusion_matrix_best.csv"
            )
            pd.DataFrame(confusion_matrix(val_y, val_pred, labels=list(range(num_classes))), index=names, columns=names).to_csv(
                run_dir / "val_confusion_matrix_best.csv"
            )
            save_predictions(run_dir / "train_predictions_best.csv", train_y, train_pred, train_prob, names)
            save_predictions(run_dir / "val_predictions_best.csv", val_y, val_pred, val_prob, names)

            diagnosis = {
                "best_epoch": best_epoch,
                "variant": variant["name"],
                "train": train_m,
                "val": val_m,
                "generalization_gap_macro_f1": gap,
                "gate_values": gate_values(model),
                "delta_vs_base512": {
                    "val_macro_f1": val_m["macro_f1"] - BASE512_REF["val_macro_f1"],
                    "gap_macro_f1": gap - BASE512_REF["gap_macro_f1"],
                },
                "delta_vs_L1_anchor": {
                    "val_macro_f1": val_m["macro_f1"] - L1_REF["val_macro_f1"],
                    "gap_macro_f1": gap - L1_REF["gap_macro_f1"],
                },
                "interpretation_hint": "If val > L1 and gap is not worse, gate keeps depth ceiling while controlling layer-overfit.",
            }
            (run_dir / "diagnosis_summary.json").write_text(json.dumps(diagnosis, indent=2), encoding="utf-8")
        else:
            no_improve += 1

        gate_str = " ".join([f"{k}={v:.4f}" for k, v in gate_values(model).items()])
        log(
            f"[{variant['name']}] epoch {epoch:03d} "
            f"loss={row['loss']:.5f} ce={row['ce_loss']:.5f} "
            f"train_f1={train_m['macro_f1']:.6f} val_f1={val_m['macro_f1']:.6f} "
            f"gap={gap:.6f} best={best_val:.6f}@{best_epoch} "
            f"lr={lr:.6g} {gate_str}"
        )

        if no_improve >= args.patience:
            log(f"[{variant['name']}] early stopping at epoch {epoch}, best_epoch={best_epoch}, best_val={best_val:.6f}")
            break

    # Make sure final summary exists.
    if best_row is None:
        raise RuntimeError("No best epoch recorded.")
    log(f"[{variant['name']}] DONE best_epoch={best_epoch} best_val={best_val:.6f}")


def read_diag(run_dir: Path) -> Optional[Dict[str, Any]]:
    p = run_dir / "diagnosis_summary.json"
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_summary(root: Path, out_root: Path, summary_out: Path, K: int) -> pd.DataFrame:
    summary_out.mkdir(parents=True, exist_ok=True)
    rows = []
    for diag_path in sorted(out_root.rglob("diagnosis_summary.json")):
        run_dir = diag_path.parent
        diag = read_diag(run_dir)
        if not diag:
            continue
        variant = diag.get("variant", run_dir.name)
        row = {
            "variant": variant,
            "run_dir": str(run_dir),
            "relative_run_dir": str(run_dir.relative_to(out_root)) if str(run_dir).startswith(str(out_root)) else run_dir.name,
            "best_epoch": diag.get("best_epoch"),
            "train_macro_f1": diag.get("train", {}).get("macro_f1"),
            "val_macro_f1": diag.get("val", {}).get("macro_f1"),
            "gap_macro_f1": diag.get("generalization_gap_macro_f1"),
            "train_accuracy": diag.get("train", {}).get("accuracy"),
            "val_accuracy": diag.get("val", {}).get("accuracy"),
            "train_weighted_f1": diag.get("train", {}).get("weighted_f1"),
            "val_weighted_f1": diag.get("val", {}).get("weighted_f1"),
            "delta_val_vs_base512": diag.get("delta_vs_base512", {}).get("val_macro_f1"),
            "delta_gap_vs_base512": diag.get("delta_vs_base512", {}).get("gap_macro_f1"),
            "delta_val_vs_L1": diag.get("delta_vs_L1_anchor", {}).get("val_macro_f1"),
            "delta_gap_vs_L1": diag.get("delta_vs_L1_anchor", {}).get("gap_macro_f1"),
        }
        for k, v in diag.get("gate_values", {}).items():
            row[f"gate_{k}"] = v
        # Per-class val f1.
        rep = {}
        rp = run_dir / "val_classification_report_best.json"
        if rp.exists():
            rep = json.loads(rp.read_text(encoding="utf-8"))
            for label, metrics in rep.items():
                if isinstance(metrics, dict) and "f1-score" in metrics:
                    row[f"val_f1_{label}"] = metrics["f1-score"]
        if row["val_macro_f1"] is not None:
            if row["val_macro_f1"] > L1_REF["val_macro_f1"] and row["gap_macro_f1"] <= L1_REF["gap_macro_f1"] + 0.005:
                row["decision_tag"] = "beats_L1_with_controlled_gap"
            elif row["val_macro_f1"] > L1_REF["val_macro_f1"]:
                row["decision_tag"] = "beats_L1_but_gap_check"
            elif row["val_macro_f1"] > BASE512_REF["val_macro_f1"] and row["gap_macro_f1"] <= BASE512_REF["gap_macro_f1"]:
                row["decision_tag"] = "beats_base_not_L1"
            else:
                row["decision_tag"] = "not_better_than_current_anchor"
        rows.append(row)

    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["val_macro_f1", "gap_macro_f1"], ascending=[False, True])
    df.to_csv(summary_out / "F1c_deep_layer_gate_summary.csv", index=False)

    lines = []
    lines.append("# F1c Deep-Layer Residual Gate Summary\n")
    lines.append("## Goal\n")
    lines.append("```text")
    lines.append("F1b audit located overfit in encoder.layers.1/2.")
    lines.append("F1c keeps 3 layers but gates only layers 1/2 to preserve depth ceiling while reducing gap.")
    lines.append("This is not a hidden/classifier/head sweep.")
    lines.append("```")
    lines.append("\n## References\n")
    lines.append("```text")
    lines.append(f"Base512 val macro-F1 = {BASE512_REF['val_macro_f1']:.6f}, gap = {BASE512_REF['gap_macro_f1']:.6f}")
    lines.append(f"L1 anchor val macro-F1 = {L1_REF['val_macro_f1']:.6f}, gap = {L1_REF['gap_macro_f1']:.6f}")
    lines.append("```")
    lines.append("\n## Results\n")
    if len(df):
        cols = [c for c in [
            "variant", "best_epoch", "train_macro_f1", "val_macro_f1", "gap_macro_f1",
            "delta_val_vs_base512", "delta_gap_vs_base512", "delta_val_vs_L1", "delta_gap_vs_L1",
            "decision_tag"
        ] if c in df.columns]
        lines.append(df[cols].to_markdown(index=False))
        lines.append("\n## Gate values at best epoch\n")
        gate_cols = [c for c in df.columns if c.startswith("gate_")]
        if gate_cols:
            lines.append(df[["variant"] + gate_cols].to_markdown(index=False))
        best = df.iloc[0]
        lines.append("\n## Best F1c by val macro-F1\n")
        lines.append("```text")
        lines.append(f"variant = {best['variant']}")
        lines.append(f"val_macro_f1 = {best['val_macro_f1']:.6f}")
        lines.append(f"gap_macro_f1 = {best['gap_macro_f1']:.6f}")
        lines.append(f"delta_val_vs_L1 = {best['delta_val_vs_L1']:+.6f}")
        lines.append(f"delta_gap_vs_L1 = {best['delta_gap_vs_L1']:+.6f}")
        lines.append(f"decision_tag = {best['decision_tag']}")
        lines.append("```")
    else:
        lines.append("No completed F1c runs found.")

    lines.append("\n## Decision rule\n")
    lines.append("```text")
    lines.append("If F1c > L1 and gap <= L1 gap + 0.005:")
    lines.append("  F1c becomes stronger anti-overfit candidate than hard L1.")
    lines.append("")
    lines.append("If F1c > L1 but gap grows a lot:")
    lines.append("  treat as architecture gain, not clean anti-overfit.")
    lines.append("")
    lines.append("If F1c <= L1:")
    lines.append("  keep L1 as current anti-overfit candidate.")
    lines.append("```")
    (summary_out / "F1c_deep_layer_gate_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return df


def zip_dir(src_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in src_dir.rglob("*"):
            if p.is_file() and p != zip_path:
                z.write(p, p.relative_to(src_dir.parent))


def make_combined_zip(paths: List[Tuple[Path, str]], zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for path, prefix in paths:
            if not path.exists():
                continue
            if path.is_file():
                z.write(path, Path(prefix) / path.name)
            else:
                for p in path.rglob("*"):
                    if p.is_file() and p != zip_path:
                        z.write(p, Path(prefix) / p.relative_to(path))


def _stream_process_output(proc: subprocess.Popen, log_file, prefix: str) -> None:
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            log_file.write(line + "\n")
            log_file.flush()
            print(f"[{prefix}] {line}", flush=True)
    except Exception as e:
        print(f"[F1c][stream-error][{prefix}] {e}", flush=True)


def launch_variant(args, variant: Dict[str, Any], gpu_id: str, root: Path) -> subprocess.Popen:
    log_dir = resolve_path(args.log_dir, root)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{variant['name']}.log"

    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--single-run",
        "--variant-name", variant["name"],
        "--out-root", args.out_root,
        "--dataset-npz", args.dataset_npz,
        "--train-raw", args.train_raw,
        "--val-raw", args.val_raw,
        "--base-config", args.base_config,
        "--device", "cuda",
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--warmup-epochs", str(args.warmup_epochs),
        "--min-lr-ratio", str(args.min_lr_ratio),
        "--patience", str(args.patience),
        "--min-delta", str(args.min_delta),
        "--num-workers", str(args.num_workers),
        "--grad-clip-norm", str(args.grad_clip_norm),
        "--seed", str(args.seed),
        "--K", str(args.K),
        "--hidden-dim", str(args.hidden_dim),
        "--num-heads", str(args.num_heads),
        "--classifier-hidden-dim", str(args.classifier_hidden_dim),
        "--dropout", str(args.dropout),
        "--classifier-dropout", str(args.classifier_dropout),
    ]
    if args.amp:
        cmd.append("--amp")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print("=" * 100, flush=True)
    print(f"[F1c] launch {variant['name']} on physical GPU {gpu_id}", flush=True)
    print(f"[F1c] log: {log_path}", flush=True)
    print("[F1c] CMD:", " ".join(cmd), flush=True)

    f = log_path.open("w", encoding="utf-8")
    f.write(f"[F1c] variant={variant['name']}\n")
    f.write(f"[F1c] gpu={gpu_id}\n")
    f.write(f"[F1c] cmd={' '.join(cmd)}\n\n")
    f.flush()

    p = subprocess.Popen(
        cmd,
        cwd=str(root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    p._f1c_log_file = f  # type: ignore[attr-defined]
    p._f1c_variant_name = variant["name"]  # type: ignore[attr-defined]
    p._f1c_gpu_id = gpu_id  # type: ignore[attr-defined]
    th = threading.Thread(target=_stream_process_output, args=(p, f, variant["name"]), daemon=True)
    p._f1c_stream_thread = th  # type: ignore[attr-defined]
    th.start()
    return p


def run_parallel(args) -> None:
    root = repo_root_from_here()
    gpu_ids = [x.strip() for x in args.gpu_ids.split(",") if x.strip()]
    if not gpu_ids:
        gpu_ids = ["0"]
    max_parallel = max(1, min(args.max_parallel, len(gpu_ids)))

    queue = list(VARIANTS)
    running: List[subprocess.Popen] = []

    out_root = resolve_path(args.out_root, root)
    log(f"root={root}")
    log(f"out_root={out_root}")
    log(f"parallel GPUs={gpu_ids}, max_parallel={max_parallel}")
    log("F1c is a stage-specific test: gates only encoder.layers.1/2.")

    while queue or running:
        used = {getattr(p, "_f1c_gpu_id") for p in running}
        free = [g for g in gpu_ids if g not in used]
        while queue and len(running) < max_parallel and free:
            v = queue.pop(0)
            rd = out_root / f"Keff{args.K}" / v["name"]
            if args.skip_existing and (rd / "diagnosis_summary.json").exists():
                log(f"skip existing {v['name']}")
                continue
            running.append(launch_variant(args, v, free.pop(0), root))

        time.sleep(args.poll_seconds)
        still = []
        for p in running:
            ret = p.poll()
            if ret is None:
                still.append(p)
            else:
                name = getattr(p, "_f1c_variant_name", "unknown")
                gpu = getattr(p, "_f1c_gpu_id", "?")
                th = getattr(p, "_f1c_stream_thread", None)
                if th is not None:
                    th.join(timeout=5)
                lf = getattr(p, "_f1c_log_file", None)
                if lf is not None:
                    lf.flush()
                    lf.close()
                print(f"[F1c] finished {name} on GPU {gpu} returncode={ret}", flush=True)
                if ret != 0:
                    raise RuntimeError(f"F1c variant failed: {name}, returncode={ret}")
        running = still

    summary_out = resolve_path(args.summary_out, root)
    df = collect_summary(root, out_root, summary_out, args.K)

    if args.make_zip:
        raw_zip = out_root.with_suffix(".zip")
        summary_zip = summary_out.with_suffix(".zip")
        log_dir = resolve_path(args.log_dir, root)
        log_zip = log_dir.with_suffix(".zip")
        combined_zip = resolve_path(args.combined_zip, root)

        zip_dir(out_root, raw_zip)
        zip_dir(summary_out, summary_zip)
        zip_dir(log_dir, log_zip)
        make_combined_zip(
            [(summary_out, "summary"), (out_root, "raw_runs"), (log_dir, "logs")],
            combined_zip,
        )
        log(f"raw zip={raw_zip}")
        log(f"summary zip={summary_zip}")
        log(f"logs zip={log_zip}")
        log(f"combined zip={combined_zip}")

    log("summary:")
    if len(df):
        cols = [c for c in ["variant", "best_epoch", "train_macro_f1", "val_macro_f1", "gap_macro_f1", "delta_val_vs_L1", "delta_gap_vs_L1", "decision_tag"] if c in df.columns]
        print(df[cols].to_string(index=False), flush=True)
    log("DONE")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--single-run", action="store_true")
    ap.add_argument("--variant-name", default=None)

    # Paths.
    ap.add_argument("--dataset-npz", default="03_outputs/05_dataset/dataset.npz")
    ap.add_argument("--train-raw", default="01_split/train_raw.csv")
    ap.add_argument("--val-raw", default="01_split/val_raw.csv")
    ap.add_argument("--base-config", default="03_outputs/06_model/config.json")
    ap.add_argument("--out-root", default="05_test/outputs/F1c_deep_layer_gate")
    ap.add_argument("--summary-out", default="05_test/outputs/F1c_deep_layer_gate_summary")
    ap.add_argument("--log-dir", default="05_test/outputs/F1c_deep_layer_gate_logs")
    ap.add_argument("--combined-zip", default="05_test/outputs/F1c_deep_layer_gate_ALL.zip")

    # Official base-compatible training.
    ap.add_argument("--K", type=int, default=512)
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

    # Keep base model dimensions fixed unless explicitly changed.
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--classifier-hidden-dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--classifier-dropout", type=float, default=0.1)

    # Parallel.
    ap.add_argument("--gpu-ids", default="0,1")
    ap.add_argument("--max-parallel", type=int, default=2)
    ap.add_argument("--poll-seconds", type=float, default=10.0)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.add_argument("--make-zip", action="store_true", default=True)
    ap.add_argument("--no-zip", dest="make_zip", action="store_false")

    args = ap.parse_args()

    if args.single_run:
        if not args.variant_name:
            raise ValueError("--variant-name required with --single-run")
        variant = next((v for v in VARIANTS if v["name"] == args.variant_name), None)
        if variant is None:
            raise ValueError(f"unknown variant {args.variant_name}; available {[v['name'] for v in VARIANTS]}")
        train_one_variant(args, variant)
    else:
        run_parallel(args)


if __name__ == "__main__":
    main()
