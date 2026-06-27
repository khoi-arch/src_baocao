#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_train.py

Official D3 trainer for the final C2 dataset.

Reads:
  03_outputs/05_dataset/dataset.npz
  03_outputs/05_dataset/metadata.json
  01_split/train_raw.csv
  01_split/val_raw.csv

Trains:
  D3 = offset interpolation + raw_scaled FiLM/multiply fusion

Writes:
  03_outputs/06_model/
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import config as CFG
import train_utils as TU

_model_path = Path(__file__).resolve().with_name("06_model.py")
_model_spec = importlib.util.spec_from_file_location("_src_baocao_06_model", _model_path)
_model_mod = importlib.util.module_from_spec(_model_spec)
assert _model_spec is not None and _model_spec.loader is not None
_model_spec.loader.exec_module(_model_mod)
D3C2D3Transformer = _model_mod.D3C2D3Transformer


def cfg(name: str, default):
    return getattr(CFG, name, default)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train official D3 model on official C2 dataset.")
    p.add_argument("--run-id", choices=["D3"], default="D3")
    p.add_argument("--K", type=int, default=int(CFG.TOKEN_K))
    p.add_argument("--num-bins", type=int, default=int(CFG.VALUE_NUM_BINS))
    p.add_argument("--dataset-npz", default=str(CFG.DATASET_NPZ))
    p.add_argument("--metadata-json", default=str(CFG.DATASET_METADATA))
    p.add_argument("--train-raw", default=str(CFG.TRAIN_RAW_CSV))
    p.add_argument("--val-raw", default=str(CFG.VAL_RAW_CSV))
    p.add_argument("--out-root", default=str(CFG.MODEL_DIR))
    p.add_argument("--seed", type=int, default=int(CFG.TRAIN_SEED))
    p.add_argument("--device", default=str(CFG.TRAIN_DEVICE))
    p.add_argument("--epochs", type=int, default=int(CFG.TRAIN_EPOCHS))
    p.add_argument("--batch-size", type=int, default=int(CFG.TRAIN_BATCH_SIZE))
    p.add_argument("--lr", type=float, default=float(CFG.TRAIN_LR))
    p.add_argument("--weight-decay", type=float, default=float(CFG.TRAIN_WEIGHT_DECAY))
    p.add_argument("--scheduler", choices=["none", "warmup_cosine"], default=str(CFG.TRAIN_SCHEDULER))
    p.add_argument("--warmup-epochs", type=int, default=int(CFG.TRAIN_WARMUP_EPOCHS))
    p.add_argument("--min-lr-ratio", type=float, default=float(CFG.TRAIN_MIN_LR_RATIO))
    p.add_argument("--patience", type=int, default=int(CFG.TRAIN_PATIENCE))
    p.add_argument("--min-delta", type=float, default=float(CFG.TRAIN_MIN_DELTA))
    p.add_argument("--num-workers", type=int, default=int(CFG.TRAIN_NUM_WORKERS))
    p.add_argument("--grad-clip-norm", type=float, default=float(CFG.TRAIN_GRAD_CLIP_NORM))
    p.add_argument("--use-class-weights", action=argparse.BooleanOptionalAction, default=bool(CFG.USE_CLASS_WEIGHTS))
    p.add_argument("--value-dim", type=int, default=int(CFG.VALUE_EMBED_DIM))
    p.add_argument("--feature-dim", type=int, default=int(CFG.FEATURE_EMBED_DIM))
    p.add_argument("--hidden-dim", type=int, default=int(CFG.MODEL_HIDDEN_DIM))
    p.add_argument("--num-layers", type=int, default=int(CFG.MODEL_NUM_LAYERS))
    p.add_argument("--num-heads", type=int, default=int(CFG.MODEL_NUM_HEADS))
    p.add_argument("--dropout", type=float, default=float(CFG.MODEL_DROPOUT))
    p.add_argument("--classifier-hidden-dim", type=int, default=int(CFG.CLASSIFIER_HIDDEN_DIM))
    p.add_argument("--classifier-dropout", type=float, default=float(CFG.CLASSIFIER_DROPOUT))
    p.add_argument("--norm-first", action=argparse.BooleanOptionalAction, default=bool(CFG.TRANSFORMER_NORM_FIRST))
    p.add_argument("--gate-init", type=float, default=float(CFG.GATE_INIT))
    p.add_argument("--diagnostic-only", action="store_true", help="Build data/model diagnostics and exit before training.")
    return p.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path_like: str | Path) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    return repo_root() / p


def as_str_list(arr) -> List[str]:
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in arr]


def load_dataset(dataset_path: Path, metadata_path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    if not dataset_path.exists():
        raise FileNotFoundError(str(dataset_path))
    if not metadata_path.exists():
        raise FileNotFoundError(str(metadata_path))
    data = dict(np.load(dataset_path, allow_pickle=True))
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    required = ["X_train_bin", "X_train_offset", "y_train", "X_val_bin", "X_val_offset", "y_val"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"dataset missing arrays: {missing}")
    return data, meta


def resolve_raw_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    train_path = resolve_path(args.train_raw)
    val_path = resolve_path(args.val_raw)
    if not train_path.exists():
        raise FileNotFoundError(str(train_path))
    if not val_path.exists():
        raise FileNotFoundError(str(val_path))
    return train_path, val_path


def load_raw_scaled(meta: Dict[str, object], args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    feature_names = [str(x) for x in meta["feature_names"]]
    train_path, val_path = resolve_raw_paths(args)
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    missing_train = [f for f in feature_names if f not in train.columns]
    missing_val = [f for f in feature_names if f not in val.columns]
    if missing_train:
        raise ValueError(f"train_raw missing features: {missing_train[:10]}")
    if missing_val:
        raise ValueError(f"val_raw missing features: {missing_val[:10]}")
    X_train_raw = train.loc[:, feature_names].to_numpy(dtype=np.float64)
    X_val_raw = val.loc[:, feature_names].to_numpy(dtype=np.float64)
    if np.isnan(X_train_raw).any() or np.isinf(X_train_raw).any():
        raise ValueError("train_raw contains NaN/Inf in selected features")
    if np.isnan(X_val_raw).any() or np.isinf(X_val_raw).any():
        raise ValueError("val_raw contains NaN/Inf in selected features")
    mn = X_train_raw.min(axis=0)
    mx = X_train_raw.max(axis=0)
    denom = mx - mn
    constant = np.isclose(denom, 0.0)
    denom_safe = denom.copy()
    denom_safe[constant] = 1.0
    X_train = (X_train_raw - mn) / denom_safe
    X_val = (X_val_raw - mn) / denom_safe
    X_train[:, constant] = 0.5
    X_val[:, constant] = 0.5
    X_train = np.clip(X_train, 0.0, 1.0).astype(np.float32)
    X_val = np.clip(X_val, 0.0, 1.0).astype(np.float32)
    info = {
        "source": "raw_scaled",
        "train_path": str(train_path),
        "val_path": str(val_path),
        "scale": "train_only_minmax_linear_clip_val",
        "n_constant_features": int(constant.sum()),
        "constant_features": [feature_names[i] for i, c in enumerate(constant) if bool(c)],
        "train_min": float(X_train.min()),
        "train_max": float(X_train.max()),
        "val_min": float(X_val.min()),
        "val_max": float(X_val.max()),
    }
    return X_train, X_val, info


class C2D3Dataset(Dataset):
    def __init__(self, X_bin: np.ndarray, X_offset: np.ndarray, X_cont: np.ndarray, X_mask: np.ndarray, y: np.ndarray):
        if X_bin.ndim != 2:
            raise ValueError(f"X_bin must be [N,F], got {X_bin.shape}")
        if X_offset.shape != X_bin.shape or X_cont.shape != X_bin.shape or X_mask.shape != X_bin.shape:
            raise ValueError(f"shape mismatch: bin={X_bin.shape}, offset={X_offset.shape}, cont={X_cont.shape}, mask={X_mask.shape}")
        if y.ndim != 1 or y.shape[0] != X_bin.shape[0]:
            raise ValueError(f"y mismatch: {y.shape} vs {X_bin.shape}")
        vals = np.stack([X_offset.astype(np.float32), X_cont.astype(np.float32), X_mask.astype(np.float32)], axis=-1)
        self.X = torch.as_tensor(X_bin, dtype=torch.long)
        self.V = torch.as_tensor(vals, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.V[idx], self.y[idx]


def strip_eval(report: Dict[str, object]) -> Dict[str, object]:
    return {k: v for k, v in report.items() if k not in {"y_true", "y_pred", "confidence", "probs"}}


def malware_avg_f1(report: Dict[str, object]) -> float:
    vals = []
    for label, m in report["per_class"].items():
        if str(label).strip().lower() != "benign":
            vals.append(float(m["f1"]))
    return float(np.mean(vals)) if vals else 0.0


def make_boundary_bin_diagnostics(X_bin: np.ndarray, X_offset: np.ndarray, *, num_bins: int, feature_names: List[str], split_name: str) -> Dict[str, object]:
    Xb = np.asarray(X_bin, dtype=np.int64)
    Off = np.asarray(X_offset, dtype=np.float32)
    last_bin = int(num_bins) - 1
    mask = Xb == last_bin
    n_rows, n_features = Xb.shape
    per_feature = []
    for j, name in enumerate(feature_names):
        mj = mask[:, j]
        cnt = int(mj.sum())
        if cnt:
            offsets = Off[mj, j]
            offset_min = float(offsets.min())
            offset_max = float(offsets.max())
            offset_mean = float(offsets.mean())
            offset_std = float(offsets.std())
        else:
            offset_min = offset_max = offset_mean = offset_std = None
        per_feature.append({"feature": str(name), "last_bin_count": cnt, "last_bin_ratio": float(cnt / max(n_rows, 1)), "last_bin_offset_min": offset_min, "last_bin_offset_max": offset_max, "last_bin_offset_mean": offset_mean, "last_bin_offset_std": offset_std})
    return {
        "split": split_name,
        "num_bins": int(num_bins),
        "last_bin_id": int(last_bin),
        "n_rows": int(n_rows),
        "n_features": int(n_features),
        "last_bin_cells": int(mask.sum()),
        "last_bin_cell_ratio": float(mask.sum() / max(mask.size, 1)),
        "features_with_last_bin": int(sum(1 for r in per_feature if r["last_bin_count"] > 0)),
        "top_features_by_last_bin_ratio": sorted(per_feature, key=lambda r: r["last_bin_ratio"], reverse=True)[:15],
    }


@torch.no_grad()
def make_embedding_runtime_diagnostics(model, *, X_bin: np.ndarray, X_offset: np.ndarray, X_cont: np.ndarray, X_mask: np.ndarray, device: torch.device, max_rows: int = 2048) -> Dict[str, object]:
    n = int(min(max_rows, X_bin.shape[0]))
    X = torch.as_tensor(X_bin[:n], dtype=torch.long, device=device)
    V = torch.as_tensor(np.stack([X_offset[:n].astype(np.float32), X_cont[:n].astype(np.float32), X_mask[:n].astype(np.float32)], axis=-1), dtype=torch.float32, device=device)
    emb = model.embedding
    vals = V.to(dtype=torch.float32, device=device)
    offset = vals[..., 0:1].clamp(0.0, 1.0)
    cont = vals[..., 1:2].clamp(0.0, 1.0)
    mask = vals[..., 2:3].clamp(0.0, 1.0)
    out = {
        "n_rows_checked": n,
        "embedding_class": emb.__class__.__name__,
        "offset_mean": float(offset.mean().detach().cpu()),
        "offset_std": float(offset.std(unbiased=False).detach().cpu()),
        "continuous_mean": float(cont.mean().detach().cpu()),
        "continuous_std": float(cont.std(unbiased=False).detach().cpu()),
        "continuous_mask_mean": float(mask.mean().detach().cpu()),
    }
    local = emb.local_interp(X, offset)
    out["local_abs_mean"] = float(local.abs().mean().detach().cpu())
    out["local_std"] = float(local.std(unbiased=False).detach().cpu())
    out["local_l2_mean_per_cell"] = float(torch.linalg.vector_norm(local, dim=-1).mean().detach().cpu())
    b_last = torch.full((1, 1), int(emb.num_bins) - 1, dtype=torch.long, device=device)
    off0 = torch.zeros((1, 1, 1), dtype=torch.float32, device=device)
    off1 = torch.ones((1, 1, 1), dtype=torch.float32, device=device)
    diff = torch.linalg.vector_norm(emb.local_interp(b_last, off1) - emb.local_interp(b_last, off0), dim=-1)
    out["last_bin_offset_sensitivity_l2"] = float(diff.item())
    out["last_bin_boundary_collapse_fixed"] = bool(float(diff.item()) > 1e-12)
    gamma = torch.tanh(emb.gamma_proj(cont))
    beta = emb.beta_proj(cont)
    gate = torch.sigmoid(emb.cont_gate_logit).to(device=device).unsqueeze(0).expand(X.shape[0], X.shape[1], 1)
    g = mask * gate
    delta = (local * (g * gamma)) + (g * beta)
    out["film_gamma_abs_mean"] = float(gamma.abs().mean().detach().cpu())
    out["film_beta_abs_mean"] = float(beta.abs().mean().detach().cpu())
    out["film_gate_mean"] = float(g.mean().detach().cpu())
    out["film_delta_l2_mean_per_cell"] = float(torch.linalg.vector_norm(delta, dim=-1).mean().detach().cpu())
    out["film_delta_over_local_l2_mean"] = float(out["film_delta_l2_mean_per_cell"] / max(out["local_l2_mean_per_cell"], 1e-12))
    return out


def make_diagnosis_summary(*, args, best_epoch: int, best_train: Dict[str, object], best_val: Dict[str, object], metadata: Dict[str, object], continuous_info: Dict[str, object], boundary_data_diagnostics: Dict[str, object], embedding_runtime_diagnostics: Dict[str, object], model) -> Dict[str, object]:
    train_macro = float(best_train["macro_f1"])
    val_macro = float(best_val["macro_f1"])
    worst_classes = sorted([
        {"label": str(label).strip(), "f1": float(m["f1"]), "precision": float(m["precision"]), "recall": float(m["recall"]), "support": int(m["support"])}
        for label, m in best_val["per_class"].items()
    ], key=lambda x: (x["f1"], -x["support"]))[:10]
    return {
        "phase": "Official C2/D3 training",
        "run_id": "D3",
        "best_epoch": int(best_epoch),
        "representation": "D3_offset_interpolation_raw_FiLM_on_C2_dataset",
        "local": "offset_interpolation",
        "continuous_source": "raw_scaled",
        "fusion": "raw_film",
        "train": {"loss": float(best_train["loss"]), "accuracy": float(best_train["accuracy"]), "macro_f1": train_macro, "weighted_f1": float(best_train["weighted_f1"]), "malware_only_avg_f1": malware_avg_f1(best_train)},
        "val": {"loss": float(best_val["loss"]), "accuracy": float(best_val["accuracy"]), "macro_f1": val_macro, "weighted_f1": float(best_val["weighted_f1"]), "malware_only_avg_f1": malware_avg_f1(best_val)},
        "generalization_gap_macro_f1": float(train_macro - val_macro),
        "worst_val_classes_by_f1": worst_classes,
        "model_config": {"value_dim": int(args.value_dim), "feature_dim": int(args.feature_dim), "cell_dim": int(args.value_dim + args.feature_dim), "num_bins": int(args.num_bins), "hidden_dim": int(args.hidden_dim), "num_layers": int(args.num_layers), "num_heads": int(args.num_heads), "dropout": float(args.dropout), "classifier_hidden_dim": int(args.classifier_hidden_dim), "classifier_dropout": float(args.classifier_dropout), "norm_first": bool(args.norm_first), "use_class_weights": bool(args.use_class_weights), "scheduler": str(args.scheduler), "gate_init": float(args.gate_init)},
        "continuous_info": continuous_info,
        "boundary_data_diagnostics": boundary_data_diagnostics,
        "embedding_runtime_diagnostics": embedding_runtime_diagnostics,
        "embedding_extra_summary": model.embedding_extra_summary(),
        "c2_policy_name": metadata.get("policy_name"),
        "c2_strategy_counts": metadata.get("strategy_counts", {}),
    }


def main() -> None:
    args = parse_args()
    args.run_id = "D3"
    TU.set_seed(int(args.seed))
    dataset_path = resolve_path(args.dataset_npz)
    metadata_path = resolve_path(args.metadata_json)
    data, meta = load_dataset(dataset_path, metadata_path)

    X_train = data["X_train_bin"].astype(np.int64)
    O_train = data["X_train_offset"].astype(np.float32)
    y_train = data["y_train"].astype(np.int64)
    X_val = data["X_val_bin"].astype(np.int64)
    O_val = data["X_val_offset"].astype(np.float32)
    y_val = data["y_val"].astype(np.int64)

    X_train_cont, X_val_cont, continuous_info = load_raw_scaled(meta, args)
    if X_train_cont.shape != X_train.shape:
        raise ValueError(f"train continuous shape mismatch: {X_train_cont.shape} vs {X_train.shape}")
    if X_val_cont.shape != X_val.shape:
        raise ValueError(f"val continuous shape mismatch: {X_val_cont.shape} vs {X_val.shape}")
    M_train = np.ones_like(X_train, dtype=np.float32)
    M_val = np.ones_like(X_val, dtype=np.float32)

    feature_names = [str(x) for x in meta["feature_names"]]
    B = int(args.num_bins)
    K_artifact = int(args.K)
    boundary_data_diagnostics = {
        "train": make_boundary_bin_diagnostics(X_train, O_train, num_bins=B, feature_names=feature_names, split_name="train"),
        "val": make_boundary_bin_diagnostics(X_val, O_val, num_bins=B, feature_names=feature_names, split_name="val"),
    }
    label_mapping = meta.get("label_mapping", {})
    if label_mapping:
        label_names = [label for label, _ in sorted(label_mapping.items(), key=lambda kv: kv[1])]
    elif "label_names" in data:
        label_names = as_str_list(data["label_names"])
    else:
        label_names = list(CFG.LABEL_NAMES)
    num_classes = int(len(label_names))
    n_features = int(meta.get("n_features", X_train.shape[1]))

    out_dir = resolve_path(args.out_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = out_dir / "reports"
    pred_dir = out_dir / "predictions"
    reports_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    device = TU.pick_device(str(args.device))

    config_obj = {
        "stage": "07_train_D3_official",
        "phase": "official_C2D3_training",
        "run_id": "D3",
        "K_artifact": K_artifact,
        "num_bins": B,
        "dataset_npz": str(dataset_path),
        "metadata_json": str(metadata_path),
        "out_dir": str(out_dir),
        "continuous_info": continuous_info,
        "seed": int(args.seed),
        "device": str(device),
        "epochs": int(args.epochs),
        "diagnostic_only": bool(args.diagnostic_only),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "scheduler": str(args.scheduler),
        "warmup_epochs": int(args.warmup_epochs),
        "min_lr_ratio": float(args.min_lr_ratio),
        "patience": int(args.patience),
        "min_delta": float(args.min_delta),
        "grad_clip_norm": float(args.grad_clip_norm),
        "use_class_weights": bool(args.use_class_weights),
        "n_features": n_features,
        "num_classes": num_classes,
        "label_names": label_names,
        "representation": {"local": "offset_interpolation", "continuous_source": "raw_scaled", "fusion": "raw_film", "description": "D3 offset interpolation + raw FiLM/multiply fusion", "c2_policy_name": meta.get("policy_name"), "strategy_counts": meta.get("strategy_counts", {})},
        "model": {"value_dim": int(args.value_dim), "feature_dim": int(args.feature_dim), "cell_dim": int(args.value_dim + args.feature_dim), "hidden_dim": int(args.hidden_dim), "num_layers": int(args.num_layers), "num_heads": int(args.num_heads), "dropout": float(args.dropout), "classifier_hidden_dim": int(args.classifier_hidden_dim), "classifier_dropout": float(args.classifier_dropout), "norm_first": bool(args.norm_first), "gate_init": float(args.gate_init)},
        "boundary_data_diagnostics": boundary_data_diagnostics,
        "data_shapes": {"X_train_bin": list(X_train.shape), "X_train_offset": list(O_train.shape), "X_train_continuous": list(X_train_cont.shape), "X_train_mask": list(M_train.shape), "y_train": list(y_train.shape), "X_val_bin": list(X_val.shape), "X_val_offset": list(O_val.shape), "X_val_continuous": list(X_val_cont.shape), "X_val_mask": list(M_val.shape), "y_val": list(y_val.shape)},
        "torch_version": torch.__version__,
    }
    TU.save_json(out_dir / "config.json", config_obj)

    train_ds = C2D3Dataset(X_train, O_train, X_train_cont, M_train, y_train)
    val_ds = C2D3Dataset(X_val, O_val, X_val_cont, M_val, y_val)
    generator = torch.Generator(); generator.manual_seed(int(args.seed))
    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, num_workers=int(args.num_workers), pin_memory=(device.type == "cuda"), generator=generator)
    train_eval_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers), pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers), pin_memory=(device.type == "cuda"))

    model = D3C2D3Transformer(
        num_bins=B,
        n_features=n_features,
        num_classes=num_classes,
        value_dim=int(args.value_dim),
        feature_dim=int(args.feature_dim),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        num_heads=int(args.num_heads),
        dropout=float(args.dropout),
        classifier_hidden_dim=int(args.classifier_hidden_dim),
        classifier_dropout=float(args.classifier_dropout),
        norm_first=bool(args.norm_first),
        gate_init=float(args.gate_init),
        activation=str(CFG.MODEL_ACTIVATION),
    ).to(device)

    embedding_runtime_diagnostics = make_embedding_runtime_diagnostics(model, X_bin=X_train, X_offset=O_train, X_cont=X_train_cont, X_mask=M_train, device=device)
    config_obj["embedding_runtime_diagnostics"] = embedding_runtime_diagnostics
    TU.save_json(out_dir / "config.json", config_obj)

    if args.use_class_weights:
        weights = TU.compute_class_weights(y_train, num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        class_weights_log = weights.detach().cpu().numpy().tolist()
    else:
        criterion = nn.CrossEntropyLoss()
        class_weights_log = None
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    TU.save_json(out_dir / "class_weights.json", {"use_class_weights": bool(args.use_class_weights), "class_weights": class_weights_log, "label_names": label_names, "train_counts": np.bincount(y_train, minlength=num_classes).astype(int).tolist(), "val_counts": np.bincount(y_val, minlength=num_classes).astype(int).tolist()})

    print("===== official D3 training start =====")
    print(f"dataset: {dataset_path}")
    print(f"metadata: {metadata_path}")
    print(f"out_dir: {out_dir}")
    print(f"device: {device}")
    print(f"classes: {num_classes} {label_names}")
    print(f"C2 strategy_counts: {meta.get('strategy_counts', {})}")
    print(f"boundary collapse fixed: {embedding_runtime_diagnostics.get('last_bin_boundary_collapse_fixed')}")

    if bool(args.diagnostic_only):
        TU.save_json(out_dir / "diagnostics_only.json", {"run_id": "D3", "dataset_npz": str(dataset_path), "metadata_json": str(metadata_path), "boundary_data_diagnostics": boundary_data_diagnostics, "embedding_runtime_diagnostics": embedding_runtime_diagnostics, "continuous_info": continuous_info, "data_shapes": config_obj["data_shapes"], "model_config": config_obj["model"], "note": "Diagnostic-only run exited before training."})
        print("diagnostic-only mode: exiting before training")
        print(f"diagnostics_only: {out_dir / 'diagnostics_only.json'}")
        return

    history: List[Dict[str, object]] = []
    best_metric = -math.inf
    best_epoch = -1
    best_train_eval = None
    best_val_eval = None
    bad_epochs = 0

    for epoch in range(1, int(args.epochs) + 1):
        t0 = time.time()
        lr_epoch = TU.compute_epoch_lr(base_lr=float(args.lr), epoch=epoch, total_epochs=int(args.epochs), scheduler_name=str(args.scheduler), warmup_epochs=int(args.warmup_epochs), min_lr_ratio=float(args.min_lr_ratio))
        TU.set_optimizer_lr(optimizer, lr_epoch)
        train_step_loss = TU.train_one_epoch(model=model, loader=train_loader, criterion=criterion, optimizer=optimizer, device=device, grad_clip_norm=float(args.grad_clip_norm))
        train_eval = TU.evaluate(model, train_eval_loader, criterion, device, num_classes, label_names)
        val_eval = TU.evaluate(model, val_loader, criterion, device, num_classes, label_names)
        row = {"epoch": epoch, "lr": float(lr_epoch), "epoch_seconds": round(time.time() - t0, 3), "train_step_loss": float(train_step_loss), "train_loss": float(train_eval["loss"]), "train_acc": float(train_eval["accuracy"]), "train_macro_f1": float(train_eval["macro_f1"]), "train_weighted_f1": float(train_eval["weighted_f1"]), "train_malware_avg_f1": malware_avg_f1(train_eval), "val_loss": float(val_eval["loss"]), "val_acc": float(val_eval["accuracy"]), "val_macro_f1": float(val_eval["macro_f1"]), "val_weighted_f1": float(val_eval["weighted_f1"]), "val_malware_avg_f1": malware_avg_f1(val_eval), "macro_f1_gap_train_minus_val": float(train_eval["macro_f1"] - val_eval["macro_f1"])}
        row.update(model.embedding_extra_summary())
        history.append(row)
        TU.write_history_csv(out_dir / "history.csv", history)
        metric = float(val_eval["macro_f1"])
        improved = metric > best_metric + float(args.min_delta)
        if improved:
            best_metric = metric
            best_epoch = epoch
            best_train_eval = train_eval
            best_val_eval = val_eval
            bad_epochs = 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "best_val_macro_f1": best_metric, "config": config_obj, "metadata": meta, "continuous_info": continuous_info}, out_dir / "best_model.pt")
        else:
            bad_epochs += 1
        print(f"[D3 epoch {epoch:03d}] lr={lr_epoch:.6g} train_macro={row['train_macro_f1']:.4f} train_malware={row['train_malware_avg_f1']:.4f} val_macro={row['val_macro_f1']:.4f} val_malware={row['val_malware_avg_f1']:.4f} gap={row['macro_f1_gap_train_minus_val']:.4f} best={best_metric:.4f}@{best_epoch}")
        if bad_epochs >= int(args.patience):
            print(f"early stop: bad_epochs={bad_epochs}, patience={args.patience}")
            break

    torch.save({"model_state_dict": model.state_dict(), "epoch": int(history[-1]["epoch"]) if history else -1, "config": config_obj, "metadata": meta, "continuous_info": continuous_info}, out_dir / "last_model.pt")
    if best_train_eval is None or best_val_eval is None:
        raise RuntimeError("No best eval was recorded")

    train_report = strip_eval(best_train_eval)
    val_report = strip_eval(best_val_eval)
    TU.save_json(reports_dir / "train_classification_report_best.json", train_report)
    TU.save_json(reports_dir / "val_classification_report_best.json", val_report)
    TU.write_confusion_outputs(reports_dir, "train", best_train_eval, label_names)
    TU.write_confusion_outputs(reports_dir, "val", best_val_eval, label_names)
    TU.write_predictions_csv(pred_dir / "train_predictions_best.csv", best_train_eval["y_true"], best_train_eval["y_pred"], best_train_eval["confidence"], label_names)
    TU.write_predictions_csv(pred_dir / "val_predictions_best.csv", best_val_eval["y_true"], best_val_eval["y_pred"], best_val_eval["confidence"], label_names)
    diagnosis = make_diagnosis_summary(args=args, best_epoch=best_epoch, best_train=best_train_eval, best_val=best_val_eval, metadata=meta, continuous_info=continuous_info, boundary_data_diagnostics=boundary_data_diagnostics, embedding_runtime_diagnostics=embedding_runtime_diagnostics, model=model)
    TU.save_json(out_dir / "diagnosis_summary.json", diagnosis)
    print("===== official D3 training done =====")
    print(f"best_epoch:           {best_epoch}")
    print(f"train_macro_f1:       {diagnosis['train']['macro_f1']:.6f}")
    print(f"val_macro_f1:         {diagnosis['val']['macro_f1']:.6f}")
    print(f"val_malware_avg_f1:   {diagnosis['val']['malware_only_avg_f1']:.6f}")
    print(f"gap:                  {diagnosis['generalization_gap_macro_f1']:.6f}")
    print(f"diagnosis:            {out_dir / 'diagnosis_summary.json'}")


if __name__ == "__main__":
    main()
