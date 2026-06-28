#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E1a1 Binary D3 Attention Expert.

This script trains 3 standalone binary D3/Transformer experts:

  RS: Ransomware vs Spyware
  RT: Ransomware vs Trojan
  ST: Spyware vs Trojan

Important design
----------------
- This is NOT an auxiliary head added to the 4-class model.
- This is NOT training on baseline CLS/logits.
- Each expert is a separate D3 Transformer model trained from scratch on only
  the two classes of its pair.
- Input is the same official D3 input source as the 4-class baseline:
    dataset.npz: X_bin + X_offset + y
    raw CSV: official raw_scaled continuous via 07_train.load_continuous_for_run()
    values per cell: [offset, raw_scaled_continuous, mask]
- Model/data components are imported from 02_src/07_train.py:
    load_dataset
    load_continuous_for_run
    FusionAblationDataset
    FusionAblationTransformer
    RUN_SPECS["D3"]

Why this exists
---------------
E1a0 showed that full-feature binary experts can improve final top-2 decisions.
E1a1 tests whether the same idea works with the D3 attention architecture,
using baseline-equivalent input.

Default output:
  05_test/outputs/E1a1_binary_d3_attention_expert/
  05_test/outputs/E1a1_binary_d3_attention_expert.zip
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import inspect
import json
import math
import random
import sys
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )
except Exception as e:
    raise RuntimeError("E1a1 requires scikit-learn. Install with: pip install scikit-learn") from e


HARD_PAIRS = [
    ("Ransomware", "Spyware"),
    ("Ransomware", "Trojan"),
    ("Spyware", "Trojan"),
]
PAIR_KEY = {
    ("Ransomware", "Spyware"): "RS",
    ("Ransomware", "Trojan"): "RT",
    ("Spyware", "Trojan"): "ST",
}
PAIR_FROM_KEY = {v: k for k, v in PAIR_KEY.items()}
MALWARE_LABELS = {"Ransomware", "Spyware", "Trojan"}


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


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def import_official_train(train_script: Path):
    if not train_script.exists():
        raise FileNotFoundError(f"official 07_train.py not found: {train_script}")

    src_dir = train_script.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    spec = importlib.util.spec_from_file_location("official_07_train_for_e1a1", str(train_script))
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


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available; falling back to CPU.", flush=True)
        return torch.device("cpu")
    return torch.device(device_arg)


def load_official_inputs(args, repo_root: Path):
    """
    Reuse official 07_train.py to avoid input/preprocessing drift.
    """
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

    # Official D3 mask is ones unless D5/selective variant is used.
    M_train = np.ones_like(X_train_bin, dtype=np.float32)
    M_val = np.ones_like(X_val_bin, dtype=np.float32)

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
        "X_train_bin": X_train_bin,
        "X_train_offset": X_train_offset,
        "X_train_cont": X_train_cont.astype(np.float32),
        "M_train": M_train,
        "y_train": y_train,
        "X_val_bin": X_val_bin,
        "X_val_offset": X_val_offset,
        "X_val_cont": X_val_cont.astype(np.float32),
        "M_val": M_val,
        "y_val": y_val,
    }


def filter_pair_arrays(inp: dict, pair_key: str, split: str, max_per_class: int = 0):
    a, b = PAIR_FROM_KEY[pair_key]
    ida, idb = inp["label_to_id"][a], inp["label_to_id"][b]
    if split == "train":
        X_bin = inp["X_train_bin"]
        X_offset = inp["X_train_offset"]
        X_cont = inp["X_train_cont"]
        M = inp["M_train"]
        y = inp["y_train"]
    elif split == "val":
        X_bin = inp["X_val_bin"]
        X_offset = inp["X_val_offset"]
        X_cont = inp["X_val_cont"]
        M = inp["M_val"]
        y = inp["y_val"]
    else:
        raise ValueError(f"unknown split={split}")

    idx_a = np.where(y == ida)[0]
    idx_b = np.where(y == idb)[0]
    if max_per_class and max_per_class > 0:
        idx_a = idx_a[:max_per_class]
        idx_b = idx_b[:max_per_class]
    idx = np.concatenate([idx_a, idx_b])
    # Keep deterministic but mix classes.
    rng = np.random.default_rng(12345 if split == "train" else 23456)
    idx = rng.permutation(idx)

    y_bin = (y[idx] == idb).astype(np.int64)
    return {
        "indices": idx.astype(np.int64),
        "X_bin": X_bin[idx],
        "X_offset": X_offset[idx],
        "X_cont": X_cont[idx],
        "M": M[idx],
        "y_bin": y_bin,
        "label_a": a,
        "label_b": b,
        "id_a": int(ida),
        "id_b": int(idb),
    }


def make_official_dataset(train_mod, arrs: dict):
    return train_mod.FusionAblationDataset(
        arrs["X_bin"],
        arrs["X_offset"],
        arrs["X_cont"],
        arrs["M"],
        arrs["y_bin"],
    )


def make_loader(ds, batch_size: int, shuffle: bool, seed: int, num_workers: int, device: torch.device):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )


def get_baseline_model_cfg(args, repo_root: Path) -> dict:
    """
    Use baseline config if available. Fallback to official known D3 defaults.
    """
    cfg = {}
    path = resolve_path(args.baseline_config, repo_root)
    if path.exists():
        try:
            loaded = load_json(path)
            # Common layouts:
            # - {"model": {...}}
            # - top-level fields
            cfg.update(loaded.get("model", {}))
            for k in [
                "value_dim", "feature_dim", "hidden_dim", "num_layers", "num_heads",
                "dropout", "classifier_hidden_dim", "classifier_dropout", "norm_first", "gate_init"
            ]:
                if k in loaded and k not in cfg:
                    cfg[k] = loaded[k]
        except Exception as e:
            print(f"[WARN] failed to read baseline config {path}: {e}", flush=True)

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


def build_binary_d3_model(train_mod, args, model_cfg: dict, n_features: int, num_bins: int, device: torch.device):
    """
    Build official FusionAblationTransformer with num_classes=2.
    Tries to pass only arguments supported by the imported official class.
    """
    cls = train_mod.FusionAblationTransformer
    sig = inspect.signature(cls)

    kwargs = {
        "run_id": "D3",
        "num_bins": int(num_bins),
        "n_features": int(n_features),
        "num_classes": 2,
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


def model_forward_logits(model: nn.Module, tokens: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """
    Official model variants may use positional values or z_values keyword.
    """
    try:
        out = model(tokens, z_values=values)
    except TypeError:
        out = model(tokens, values)
    if isinstance(out, tuple):
        out = out[0]
    return out


def class_weight_tensor(y_bin: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(y_bin.astype(int), minlength=2).astype(np.float64)
    weights = counts.sum() / np.maximum(1.0, 2.0 * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def compute_lr(epoch: int, base_lr: float, epochs: int, warmup_epochs: int, min_lr_ratio: float, scheduler: str) -> float:
    if scheduler == "none":
        return float(base_lr)
    if scheduler != "warmup_cosine":
        return float(base_lr)
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return float(base_lr) * float(epoch) / float(warmup_epochs)
    if epochs <= warmup_epochs:
        return float(base_lr)
    progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(base_lr) * (float(min_lr_ratio) + (1.0 - float(min_lr_ratio)) * cosine)


def set_optimizer_lr(optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def train_one_epoch_binary(model, loader, criterion, optimizer, device, grad_clip_norm: float = 1.0) -> dict:
    model.train()
    total_loss = 0.0
    ys = []
    ps = []
    preds = []
    n = 0
    for tokens, values, y in loader:
        tokens = tokens.to(device, non_blocking=True)
        values = values.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)
        logits = model_forward_logits(model, tokens, values)
        loss = criterion(logits, y)
        loss.backward()
        if grad_clip_norm and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
        optimizer.step()

        with torch.no_grad():
            prob = torch.softmax(logits, dim=1)[:, 1]
            pred = logits.argmax(dim=1)
        bs = int(y.shape[0])
        total_loss += float(loss.item()) * bs
        n += bs
        ys.append(y.detach().cpu().numpy())
        ps.append(prob.detach().cpu().numpy())
        preds.append(pred.detach().cpu().numpy())

    y_np = np.concatenate(ys)
    p_np = np.concatenate(ps)
    pred_np = np.concatenate(preds)
    return binary_metric_dict(y_np, pred_np, p_np, loss=total_loss / max(1, n))


@torch.no_grad()
def evaluate_binary(model, loader, criterion, device) -> Tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    ys = []
    ps = []
    preds = []
    n = 0
    for tokens, values, y in loader:
        tokens = tokens.to(device, non_blocking=True)
        values = values.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()
        logits = model_forward_logits(model, tokens, values)
        loss = criterion(logits, y)
        prob = torch.softmax(logits, dim=1)[:, 1]
        pred = logits.argmax(dim=1)
        bs = int(y.shape[0])
        total_loss += float(loss.item()) * bs
        n += bs
        ys.append(y.detach().cpu().numpy())
        ps.append(prob.detach().cpu().numpy())
        preds.append(pred.detach().cpu().numpy())

    y_np = np.concatenate(ys)
    p_np = np.concatenate(ps)
    pred_np = np.concatenate(preds)
    metrics = binary_metric_dict(y_np, pred_np, p_np, loss=total_loss / max(1, n))
    return metrics, y_np, pred_np, p_np


def binary_metric_dict(y_true: np.ndarray, y_pred: np.ndarray, p_pos: np.ndarray, loss: float) -> dict:
    out = {
        "loss": float(loss),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }
    try:
        out["auc"] = float(roc_auc_score(y_true, p_pos))
    except Exception:
        out["auc"] = float("nan")
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    out.update({
        "precision_0": float(prec[0]), "recall_0": float(rec[0]), "f1_0": float(f1[0]), "support_0": int(sup[0]),
        "precision_1": float(prec[1]), "recall_1": float(rec[1]), "f1_1": float(f1[1]), "support_1": int(sup[1]),
    })
    return out


def train_pair_expert(pair_key: str, inp: dict, args, repo_root: Path, out_dir: Path, device: torch.device, model_cfg: dict) -> dict:
    train_mod = inp["train_mod"]
    pair_dir = out_dir / pair_key
    pair_dir.mkdir(parents=True, exist_ok=True)

    tr = filter_pair_arrays(inp, pair_key, "train", max_per_class=int(args.max_train_per_class))
    va = filter_pair_arrays(inp, pair_key, "val", max_per_class=int(args.max_val_per_class))

    train_ds = make_official_dataset(train_mod, tr)
    val_ds = make_official_dataset(train_mod, va)

    train_loader = make_loader(train_ds, int(args.batch_size), True, int(args.seed), int(args.num_workers), device)
    val_loader = make_loader(val_ds, int(args.batch_size), False, int(args.seed), int(args.num_workers), device)

    model, model_kwargs = build_binary_d3_model(
        train_mod=train_mod,
        args=args,
        model_cfg=model_cfg,
        n_features=len(inp["feature_names"]),
        num_bins=inp["num_bins"],
        device=device,
    )

    weight = class_weight_tensor(tr["y_bin"], device) if args.use_class_weights else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    best_score = -1.0
    best_epoch = -1
    best_state = None
    no_improve = 0
    history_rows = []

    print(
        f"[E1a1:{pair_key}] train_n={len(tr['y_bin'])} val_n={len(va['y_bin'])} "
        f"labels={tr['label_a']}<->{tr['label_b']} device={device}",
        flush=True,
    )

    for epoch in range(1, int(args.epochs) + 1):
        lr_epoch = compute_lr(
            epoch=epoch,
            base_lr=float(args.lr),
            epochs=int(args.epochs),
            warmup_epochs=int(args.warmup_epochs),
            min_lr_ratio=float(args.min_lr_ratio),
            scheduler=str(args.scheduler),
        )
        set_optimizer_lr(optimizer, lr_epoch)

        t0 = time.time()
        train_metrics = train_one_epoch_binary(
            model, train_loader, criterion, optimizer, device, grad_clip_norm=float(args.grad_clip_norm)
        )
        val_metrics, yv, predv, pv = evaluate_binary(model, val_loader, criterion, device)
        dt = time.time() - t0

        row = {
            "pair_key": pair_key,
            "epoch": int(epoch),
            "lr": float(lr_epoch),
            "seconds": float(dt),
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history_rows.append(row)

        improved = val_metrics["macro_f1"] > best_score + float(args.min_delta)
        if improved:
            best_score = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0

            torch.save({
                "model_state_dict": best_state,
                "pair_key": pair_key,
                "label_a": tr["label_a"],
                "label_b": tr["label_b"],
                "id_a": tr["id_a"],
                "id_b": tr["id_b"],
                "model_kwargs": model_kwargs,
                "model_cfg": model_cfg,
                "epoch": int(epoch),
                "val_metrics": val_metrics,
                "args": vars(args),
            }, pair_dir / "best_model.pt")
        else:
            no_improve += 1

        if epoch == 1 or epoch % int(args.log_every) == 0 or improved:
            print(
                f"[E1a1:{pair_key}] ep={epoch:03d} lr={lr_epoch:.3e} "
                f"train_f1={train_metrics['macro_f1']:.4f} val_f1={val_metrics['macro_f1']:.4f} "
                f"val_auc={val_metrics['auc']:.4f} best={best_score:.4f}@{best_epoch} "
                f"noimp={no_improve} sec={dt:.1f}",
                flush=True,
            )

        if no_improve >= int(args.patience):
            print(f"[E1a1:{pair_key}] early stop at epoch {epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError(f"No best state for {pair_key}")

    model.load_state_dict(best_state)
    best_metrics, yv, predv, pv = evaluate_binary(model, val_loader, criterion, device)

    hist_df = pd.DataFrame(history_rows)
    hist_df.to_csv(pair_dir / "history.csv", index=False)

    pred_df = pd.DataFrame({
        "pair_val_row": np.arange(len(yv), dtype=int),
        "original_sample_index": va["indices"].astype(int),
        "true_binary": yv.astype(int),
        "pred_binary": predv.astype(int),
        "prob_label_b": pv.astype(float),
        "true_label": [tr["label_b"] if int(x) == 1 else tr["label_a"] for x in yv],
        "pred_label": [tr["label_b"] if int(x) == 1 else tr["label_a"] for x in predv],
    })
    pred_df.to_csv(pair_dir / "val_pair_predictions.csv", index=False)

    cm = confusion_matrix(yv, predv, labels=[0, 1])
    pd.DataFrame(cm, index=[tr["label_a"], tr["label_b"]], columns=[tr["label_a"], tr["label_b"]]).to_csv(pair_dir / "val_pair_confusion_matrix.csv")

    summary = {
        "pair_key": pair_key,
        "pair": f"{tr['label_a']}<->{tr['label_b']}",
        "label_a": tr["label_a"],
        "label_b": tr["label_b"],
        "id_a": tr["id_a"],
        "id_b": tr["id_b"],
        "train_n": int(len(tr["y_bin"])),
        "val_n": int(len(va["y_bin"])),
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_score),
        "best_metrics": best_metrics,
        "model_kwargs": model_kwargs,
        "checkpoint": str(pair_dir / "best_model.pt"),
    }
    save_json(pair_dir / "pair_summary.json", summary)

    return {
        "pair_key": pair_key,
        "summary": summary,
        "model": model,
        "val_indices": va["indices"],
    }


def normalize_pred_df(df: pd.DataFrame, label_to_id: Dict[str, int], id_to_label: Dict[int, str]) -> pd.DataFrame:
    df = df.copy()
    if "sample_index" not in df.columns:
        df["sample_index"] = np.arange(len(df), dtype=int)
    for c in ["true_label", "pred_label"]:
        if c in df.columns:
            df[c] = df[c].map(strip_label)
    if "true_id" not in df.columns and "true_label" in df.columns:
        df["true_id"] = df["true_label"].map(label_to_id)
    if "pred_id" not in df.columns and "pred_label" in df.columns:
        df["pred_id"] = df["pred_label"].map(label_to_id)
    if "pred_label" not in df.columns and "pred_id" in df.columns:
        df["pred_label"] = df["pred_id"].astype(int).map(id_to_label)
    if "true_label" not in df.columns and "true_id" in df.columns:
        df["true_label"] = df["true_id"].astype(int).map(id_to_label)
    if "correct" not in df.columns:
        df["correct"] = df["true_id"].astype(int) == df["pred_id"].astype(int)
    needed = ["sample_index", "true_id", "true_label", "pred_id", "pred_label", "correct"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"prediction file missing columns {missing}; available={list(df.columns)}")
    df["sample_index"] = df["sample_index"].astype(int)
    df["true_id"] = df["true_id"].astype(int)
    df["pred_id"] = df["pred_id"].astype(int)
    df["true_label"] = df["true_label"].map(strip_label)
    df["pred_label"] = df["pred_label"].map(strip_label)
    df["correct"] = df["correct"].astype(bool)
    return df


def find_prob_columns(df: pd.DataFrame, label_names: List[str]) -> Optional[List[str]]:
    cols = list(df.columns)
    by_label = []
    for i, lab in enumerate(label_names):
        lab = strip_label(lab)
        possible = [
            f"prob_{lab}", f"p_{lab}", f"proba_{lab}",
            f"prob_{i}", f"p_{i}", f"proba_{i}",
        ]
        found = None
        for c in possible:
            if c in cols:
                found = c
                break
        if found is None:
            by_label = []
            break
        by_label.append(found)
    if by_label:
        return by_label
    prob_cols = [c for c in cols if str(c).startswith("prob_")]
    if len(prob_cols) == len(label_names):
        def key(c):
            tail = str(c).replace("prob_", "")
            return int(tail) if tail.isdigit() else str(c)
        return sorted(prob_cols, key=key)
    return None


def add_top2(df: pd.DataFrame, label_names: List[str], label_to_id: Dict[str, int], id_to_label: Dict[int, str]) -> pd.DataFrame:
    df = df.copy()
    if "top2_label" in df.columns:
        df["top1_id"] = df["pred_id"].astype(int)
        df["top1_label"] = df["pred_label"].map(strip_label)
        df["top2_label"] = df["top2_label"].map(strip_label)
        df["top2_id"] = df["top2_label"].map(label_to_id).astype(int)
        return df
    if "top2_id" in df.columns:
        df["top1_id"] = df["pred_id"].astype(int)
        df["top1_label"] = df["pred_label"].map(strip_label)
        df["top2_id"] = df["top2_id"].astype(int)
        df["top2_label"] = df["top2_id"].map(id_to_label)
        return df
    prob_cols = find_prob_columns(df, label_names)
    if prob_cols is None:
        raise ValueError("Cannot infer top2; provide --baseline-probs with prob_* columns.")
    prob = df[prob_cols].to_numpy(dtype=float)
    order = np.argsort(-prob, axis=1)
    top1 = order[:, 0]
    top2 = order[:, 1]
    df["top1_id"] = top1.astype(int)
    df["top2_id"] = top2.astype(int)
    df["top1_label"] = [id_to_label[int(i)] for i in top1]
    df["top2_label"] = [id_to_label[int(i)] for i in top2]
    df["top1_prob"] = prob[np.arange(len(df)), top1]
    df["top2_prob"] = prob[np.arange(len(df)), top2]
    return df


def load_baseline_with_top2(args, repo_root: Path, inp: dict) -> pd.DataFrame:
    probs_path = resolve_path(args.baseline_probs, repo_root)
    pred_path = resolve_path(args.baseline_pred, repo_root)
    path = probs_path if probs_path.exists() else pred_path
    if not path.exists():
        raise FileNotFoundError(f"Missing baseline probabilities/predictions: {probs_path}, {pred_path}")
    df = pd.read_csv(path)
    df = normalize_pred_df(df, inp["label_to_id"], inp["id_to_label"])
    df = add_top2(df, inp["label_names"], inp["label_to_id"], inp["id_to_label"])
    df = df.sort_values("sample_index").reset_index(drop=True)
    if len(df) != len(inp["y_val"]):
        raise ValueError(f"baseline rows {len(df)} != y_val {len(inp['y_val'])}")
    return df


@torch.no_grad()
def predict_expert_all_val(model: nn.Module, inp: dict, device: torch.device, args) -> np.ndarray:
    ds = inp["train_mod"].FusionAblationDataset(
        inp["X_val_bin"], inp["X_val_offset"], inp["X_val_cont"], inp["M_val"], np.zeros_like(inp["y_val"], dtype=np.int64)
    )
    loader = make_loader(ds, int(args.batch_size), False, int(args.seed), int(args.num_workers), device)
    probs = []
    model.eval()
    for tokens, values, y in loader:
        tokens = tokens.to(device, non_blocking=True)
        values = values.to(device, non_blocking=True)
        logits = model_forward_logits(model, tokens, values)
        p = torch.softmax(logits, dim=1)[:, 1]
        probs.append(p.detach().cpu().numpy())
    return np.concatenate(probs).astype(np.float32)


def hard_pair_key_from_labels(a: str, b: str) -> Optional[str]:
    s = frozenset([strip_label(a), strip_label(b)])
    for pair, key in PAIR_KEY.items():
        if s == frozenset(pair):
            return key
    return None


def baseline_top2_hardpair_mask(base: pd.DataFrame) -> np.ndarray:
    top1 = base["top1_label"].map(strip_label).to_numpy()
    top2 = base["top2_label"].map(strip_label).to_numpy()
    mask = []
    for a, b in zip(top1, top2):
        mask.append((a in MALWARE_LABELS) and (b in MALWARE_LABELS) and (hard_pair_key_from_labels(a, b) is not None))
    return np.asarray(mask, dtype=bool)


def apply_policy(base: pd.DataFrame, expert_probs: Dict[str, np.ndarray], pair_summaries: Dict[str, dict], inp: dict, threshold: float):
    y_true = inp["y_val"].astype(int)
    base_pred = base["pred_id"].to_numpy(dtype=int)
    new_pred = base_pred.copy()
    rows = []
    hard_mask = baseline_top2_hardpair_mask(base)

    for i in np.where(hard_mask)[0]:
        top1 = strip_label(base.at[int(i), "top1_label"])
        top2 = strip_label(base.at[int(i), "top2_label"])
        pk = hard_pair_key_from_labels(top1, top2)
        if pk is None or pk not in expert_probs:
            continue
        summ = pair_summaries[pk]
        p_b = float(expert_probs[pk][i])
        conf = max(p_b, 1.0 - p_b)
        if conf < threshold:
            continue
        chosen_label = summ["label_b"] if p_b >= 0.5 else summ["label_a"]
        chosen_id = int(inp["label_to_id"][chosen_label])
        old_id = int(new_pred[i])
        new_pred[i] = chosen_id
        rows.append({
            "sample_index": int(base.at[int(i), "sample_index"]),
            "row_index": int(i),
            "true_label": strip_label(base.at[int(i), "true_label"]),
            "base_pred_label": strip_label(base.at[int(i), "pred_label"]),
            "top1_label": top1,
            "top2_label": top2,
            "pair_key": pk,
            "pair": f"{summ['label_a']}<->{summ['label_b']}",
            "threshold": float(threshold),
            "expert_prob_label_b": p_b,
            "expert_conf": conf,
            "expert_label": chosen_label,
            "old_pred_id": old_id,
            "new_pred_id": chosen_id,
            "old_pred_label": inp["id_to_label"][old_id],
            "new_pred_label": inp["id_to_label"][chosen_id],
            "changed": bool(old_id != chosen_id),
        })
    return new_pred, pd.DataFrame(rows)


def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray, label_names: List[str]):
    labels = list(range(len(label_names)))
    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro")
    weighted = f1_score(y_true, y_pred, average="weighted")
    prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    per = pd.DataFrame([
        {"class_id": i, "label": label_names[i], "precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]), "support": int(sup[i])}
        for i in labels
    ])
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels), index=label_names, columns=label_names)
    return {"accuracy": float(acc), "macro_f1": float(macro), "weighted_f1": float(weighted)}, per, cm


def transition_stats(y_true: np.ndarray, base_pred: np.ndarray, new_pred: np.ndarray) -> dict:
    base_correct = base_pred == y_true
    new_correct = new_pred == y_true
    fixed = (~base_correct) & new_correct
    damaged = base_correct & (~new_correct)
    changed = base_pred != new_pred
    return {
        "wrong_to_correct": int(fixed.sum()),
        "correct_to_wrong": int(damaged.sum()),
        "net_gain": int(fixed.sum() - damaged.sum()),
        "damage_ratio": float(damaged.sum() / fixed.sum()) if int(fixed.sum()) else None,
        "changed_pred_n": int(changed.sum()),
        "baseline_correct": int(base_correct.sum()),
        "new_correct": int(new_correct.sum()),
    }


def pair_fix_damage(y_true: np.ndarray, base_pred: np.ndarray, new_pred: np.ndarray, inp: dict) -> pd.DataFrame:
    rows = []
    base_correct = base_pred == y_true
    new_correct = new_pred == y_true
    for a, b in HARD_PAIRS:
        ida, idb = inp["label_to_id"][a], inp["label_to_id"][b]
        pair_mask = (y_true == ida) | (y_true == idb)
        fixed = pair_mask & (~base_correct) & new_correct
        damaged = pair_mask & base_correct & (~new_correct)
        rows.append({
            "scope": "pair_true_labels",
            "pair": f"{a}<->{b}",
            "direction": "BIDIR",
            "n_true_pair": int(pair_mask.sum()),
            "fixed": int(fixed.sum()),
            "damaged": int(damaged.sum()),
            "net": int(fixed.sum() - damaged.sum()),
            "damage_ratio": float(damaged.sum()/fixed.sum()) if int(fixed.sum()) else None,
        })
        for true_label, other_label in [(a, b), (b, a)]:
            tid = inp["label_to_id"][true_label]
            oid = inp["label_to_id"][other_label]
            dir_mask = y_true == tid
            base_conf = dir_mask & (base_pred == oid)
            new_conf = dir_mask & (new_pred == oid)
            fixed_dir = base_conf & (new_pred == tid)
            damaged_dir = dir_mask & (base_pred == tid) & (new_pred == oid)
            rows.append({
                "scope": "hard_direction",
                "pair": f"{a}<->{b}",
                "direction": f"{true_label}->{other_label}",
                "n_true": int(dir_mask.sum()),
                "baseline_confusion_count": int(base_conf.sum()),
                "new_confusion_count": int(new_conf.sum()),
                "confusion_delta_new_minus_base": int(new_conf.sum() - base_conf.sum()),
                "fixed": int(fixed_dir.sum()),
                "damaged": int(damaged_dir.sum()),
                "net": int(fixed_dir.sum() - damaged_dir.sum()),
                "damage_ratio": float(damaged_dir.sum()/fixed_dir.sum()) if int(fixed_dir.sum()) else None,
            })
    return pd.DataFrame(rows)


def evaluate_policies(inp: dict, base: pd.DataFrame, expert_probs: Dict[str, np.ndarray], pair_summaries: Dict[str, dict], args, out_dir: Path) -> dict:
    y_true = inp["y_val"].astype(int)
    base_pred = base["pred_id"].to_numpy(dtype=int)
    base_metrics, base_per, base_cm = multiclass_metrics(y_true, base_pred, inp["label_names"])
    base_per.to_csv(out_dir / "E1a1_baseline_per_class_f1.csv", index=False)
    base_cm.to_csv(out_dir / "E1a1_baseline_confusion_matrix.csv")

    rows = []
    per_rows = []
    pred_by_policy = {}
    applied_by_policy = {}

    for thr in [float(x) for x in str(args.thresholds).split(",") if str(x).strip()]:
        new_pred, applied = apply_policy(base, expert_probs, pair_summaries, inp, threshold=thr)
        met, per, cm = multiclass_metrics(y_true, new_pred, inp["label_names"])
        trans = transition_stats(y_true, base_pred, new_pred)
        policy = f"E1a1_binary_D3_top2_hardpair_thr_{thr:g}"
        rows.append({
            "policy": policy,
            "threshold": float(thr),
            "applied_n": int(len(applied)),
            "applied_changed_n": int(applied["changed"].sum()) if len(applied) else 0,
            **met,
            "delta_accuracy": met["accuracy"] - base_metrics["accuracy"],
            "delta_macro_f1": met["macro_f1"] - base_metrics["macro_f1"],
            "delta_weighted_f1": met["weighted_f1"] - base_metrics["weighted_f1"],
            **trans,
        })
        per = per.copy()
        per["policy"] = policy
        per["threshold"] = float(thr)
        per_rows.append(per)
        pred_by_policy[policy] = new_pred
        applied_by_policy[policy] = applied

    policy_df = pd.DataFrame(rows).sort_values(["macro_f1", "net_gain", "damage_ratio"], ascending=[False, False, True]).reset_index(drop=True)
    base_row = {
        "policy": "BASELINE",
        "threshold": np.nan,
        "applied_n": 0,
        "applied_changed_n": 0,
        **base_metrics,
        "delta_accuracy": 0.0,
        "delta_macro_f1": 0.0,
        "delta_weighted_f1": 0.0,
        **transition_stats(y_true, base_pred, base_pred),
    }
    policy_out = pd.concat([pd.DataFrame([base_row]), policy_df], ignore_index=True)
    policy_out.to_csv(out_dir / "E1a1_policy_metrics.csv", index=False)
    pd.concat(per_rows, ignore_index=True).to_csv(out_dir / "E1a1_policy_per_class_f1.csv", index=False)

    best_policy = str(policy_df.iloc[0]["policy"]) if len(policy_df) else "BASELINE"
    best_pred = pred_by_policy.get(best_policy, base_pred)
    best_applied = applied_by_policy.get(best_policy, pd.DataFrame())

    best_metrics, best_per, best_cm = multiclass_metrics(y_true, best_pred, inp["label_names"])
    best_trans = transition_stats(y_true, base_pred, best_pred)
    best_pair = pair_fix_damage(y_true, base_pred, best_pred, inp)

    pred_df = base[["sample_index", "true_id", "true_label", "pred_id", "pred_label", "correct", "top1_label", "top2_label"]].copy()
    pred_df = pred_df.rename(columns={"pred_id": "base_pred_id", "pred_label": "base_pred_label", "correct": "base_correct"})
    pred_df["e1a1_pred_id"] = best_pred
    pred_df["e1a1_pred_label"] = [inp["id_to_label"][int(i)] for i in best_pred]
    pred_df["e1a1_correct"] = best_pred == y_true
    pred_df["transition"] = "both_wrong"
    pred_df.loc[pred_df["base_correct"] & pred_df["e1a1_correct"], "transition"] = "both_correct"
    pred_df.loc[(~pred_df["base_correct"]) & pred_df["e1a1_correct"], "transition"] = "fixed"
    pred_df.loc[pred_df["base_correct"] & (~pred_df["e1a1_correct"]), "transition"] = "damaged"

    pred_df.to_csv(out_dir / "E1a1_best_policy_predictions.csv", index=False)
    best_cm.to_csv(out_dir / "E1a1_best_policy_confusion_matrix.csv")
    best_pair.to_csv(out_dir / "E1a1_best_policy_pair_fix_damage.csv", index=False)
    best_applied.to_csv(out_dir / "E1a1_best_policy_applied_samples.csv", index=False)
    best_per.to_csv(out_dir / "E1a1_best_policy_per_class_f1.csv", index=False)

    return {
        "baseline_metrics": base_metrics,
        "best_policy": best_policy,
        "best_metrics": best_metrics,
        "best_transition": best_trans,
        "best_policy_row": policy_df.iloc[0].to_dict() if len(policy_df) else {},
    }


def zip_dir(src_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(src_dir.rglob("*")):
            if p.is_file() and p != zip_path:
                z.write(p, p.relative_to(src_dir.parent))


def write_summary_md(out_dir: Path, summary: dict) -> None:
    text = f"""# E1a1 Binary D3 Attention Expert

## Design

- Separate binary D3 models, not auxiliary heads.
- Input source is same as official baseline D3:
  - `dataset.npz`: bin/token + offset
  - `train_raw.csv/val_raw.csv`: official raw_scaled continuous
  - values: `[offset, raw_scaled, mask]`
- Model class imported from `02_src/07_train.py`: `FusionAblationTransformer`.
- Each expert is trained only on samples of its two classes.

## Baseline

```text
accuracy = {summary['baseline_metrics']['accuracy']:.6f}
macro_f1 = {summary['baseline_metrics']['macro_f1']:.6f}
weighted = {summary['baseline_metrics']['weighted_f1']:.6f}
```

## Best E1a1 policy

```text
policy   = {summary['best_policy']}
accuracy = {summary['best_metrics']['accuracy']:.6f}
macro_f1 = {summary['best_metrics']['macro_f1']:.6f}
weighted = {summary['best_metrics']['weighted_f1']:.6f}
```

## Transition

```text
wrong_to_correct = {summary['best_transition']['wrong_to_correct']}
correct_to_wrong = {summary['best_transition']['correct_to_wrong']}
net_gain         = {summary['best_transition']['net_gain']}
damage_ratio     = {summary['best_transition']['damage_ratio']}
changed_pred_n   = {summary['best_transition']['changed_pred_n']}
```

## Key files

- `E1a1_binary_expert_metrics.csv`
- `E1a1_policy_metrics.csv`
- `E1a1_best_policy_predictions.csv`
- `E1a1_best_policy_pair_fix_damage.csv`
- `RS/best_model.pt`, `RT/best_model.pt`, `ST/best_model.pt`
"""
    (out_dir / "E1a1_summary.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="E1a1 binary D3 attention expert")
    parser.add_argument("--dataset-npz", default="03_outputs/05_dataset/dataset.npz")
    parser.add_argument("--metadata-json", default="03_outputs/05_dataset/metadata.json")
    parser.add_argument("--train-raw", default="01_split/train_raw.csv")
    parser.add_argument("--val-raw", default="01_split/val_raw.csv")
    parser.add_argument("--official-train", default="02_src/07_train.py")
    parser.add_argument("--baseline-config", default="03_outputs/06_model/config.json")
    parser.add_argument("--baseline-pred", default="03_outputs/06_model/val_predictions_best.csv")
    parser.add_argument("--baseline-probs", default="05_test/outputs/B0_wrong_top2_audit/val_predictions_with_probs.csv")
    parser.add_argument("--out-dir", default="05_test/outputs/E1a1_binary_d3_attention_expert")
    parser.add_argument("--pairs", default="RS,RT,ST", help="Comma-separated from RS,RT,ST")
    parser.add_argument("--thresholds", default="0.5,0.55,0.6,0.65,0.7,0.75")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", default="warmup_cosine", choices=["none", "warmup_cosine"])
    parser.add_argument("--warmup-epochs", type=int, default=8)
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--use-class-weights", action="store_true", default=True)
    parser.add_argument("--no-class-weights", dest="use_class_weights", action="store_false")
    parser.add_argument("--max-train-per-class", type=int, default=0, help="Smoke-test limit. 0 means all.")
    parser.add_argument("--max-val-per-class", type=int, default=0, help="Smoke-test limit. 0 means all.")
    parser.add_argument("--make-zip", action="store_true", default=True)
    parser.add_argument("--no-zip", dest="make_zip", action="store_false")
    args = parser.parse_args()

    repo_root = repo_root_from_here()
    out_dir = resolve_path(args.out_dir, repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(args.seed))
    device = pick_device(args.device)

    print(f"[E1a1] repo_root={repo_root}", flush=True)
    print(f"[E1a1] out_dir={out_dir}", flush=True)
    print(f"[E1a1] device={device}", flush=True)

    inp = load_official_inputs(args, repo_root)
    model_cfg = get_baseline_model_cfg(args, repo_root)

    # Save run config early.
    run_config = {
        "args": vars(args),
        "device": str(device),
        "model_cfg": model_cfg,
        "label_names": inp["label_names"],
        "num_bins": int(inp["num_bins"]),
        "continuous_info": inp["continuous_info"],
        "design": "Separate binary D3 Transformer experts; official input/data/model components; not auxiliary heads.",
    }
    save_json(out_dir / "E1a1_run_config.json", run_config)

    selected_pairs = [p.strip().upper() for p in args.pairs.split(",") if p.strip()]
    for pk in selected_pairs:
        if pk not in PAIR_FROM_KEY:
            raise ValueError(f"Unknown pair key {pk}; use RS,RT,ST")

    pair_results = {}
    pair_summaries = {}
    metrics_rows = []

    for pk in selected_pairs:
        res = train_pair_expert(pk, inp, args, repo_root, out_dir, device, model_cfg)
        pair_results[pk] = res
        pair_summaries[pk] = res["summary"]
        row = {
            "pair_key": pk,
            "pair": res["summary"]["pair"],
            "best_epoch": res["summary"]["best_epoch"],
            "train_n": res["summary"]["train_n"],
            "val_n": res["summary"]["val_n"],
            **res["summary"]["best_metrics"],
        }
        metrics_rows.append(row)

    expert_metrics = pd.DataFrame(metrics_rows)
    expert_metrics.to_csv(out_dir / "E1a1_binary_expert_metrics.csv", index=False)

    print("[E1a1] loading baseline top2/probs and predicting all-val expert probabilities...", flush=True)
    base = load_baseline_with_top2(args, repo_root, inp)
    base.to_csv(out_dir / "E1a1_baseline_top2_context.csv", index=False)

    expert_probs = {}
    for pk, res in pair_results.items():
        expert_probs[pk] = predict_expert_all_val(res["model"], inp, device, args)
        np.save(out_dir / pk / "all_val_prob_label_b.npy", expert_probs[pk])
        print(f"[E1a1:{pk}] all-val prob saved", flush=True)

    evals = evaluate_policies(inp, base, expert_probs, pair_summaries, args, out_dir)

    summary = {
        "stage": "E1a1_binary_d3_attention_expert",
        "purpose": "Train separate binary D3 attention experts and test top-2 hard malware intervention.",
        "input_equivalence": {
            "same_source_as_baseline": True,
            "dataset_npz": str(resolve_path(args.dataset_npz, repo_root)),
            "metadata_json": str(resolve_path(args.metadata_json, repo_root)),
            "train_raw": str(resolve_path(args.train_raw, repo_root)),
            "val_raw": str(resolve_path(args.val_raw, repo_root)),
            "values": "[offset, raw_scaled_continuous, mask]",
            "model_component": "FusionAblationTransformer imported from 02_src/07_train.py",
            "not_used_as_input": "baseline CLS/logits/checkpoint",
        },
        "not_auxiliary_head": True,
        "model_type": "3 separate binary D3 Transformer experts",
        "pairs_trained": selected_pairs,
        "baseline_metrics": evals["baseline_metrics"],
        "best_policy": evals["best_policy"],
        "best_metrics": evals["best_metrics"],
        "best_transition": evals["best_transition"],
        "best_policy_row": evals["best_policy_row"],
        "binary_expert_metrics": expert_metrics.to_dict(orient="records"),
        "outputs": {
            "binary_expert_metrics": str(out_dir / "E1a1_binary_expert_metrics.csv"),
            "policy_metrics": str(out_dir / "E1a1_policy_metrics.csv"),
            "best_policy_predictions": str(out_dir / "E1a1_best_policy_predictions.csv"),
            "best_policy_pair_fix_damage": str(out_dir / "E1a1_best_policy_pair_fix_damage.csv"),
            "summary_md": str(out_dir / "E1a1_summary.md"),
            "run_config": str(out_dir / "E1a1_run_config.json"),
        },
        "guardrail": "This is a diagnostic/validation-direction experiment. Best threshold is selected on validation; use for research decision, not final unbiased test score.",
    }
    save_json(out_dir / "E1a1_summary.json", summary)
    write_summary_md(out_dir, summary)

    if args.make_zip:
        zip_path = out_dir.with_suffix(".zip")
        zip_dir(out_dir, zip_path)
        print(f"[E1a1] zipped outputs: {zip_path}", flush=True)

    print("[E1a1] done.", flush=True)
    print(f"[E1a1] baseline_macro_f1={evals['baseline_metrics']['macro_f1']:.6f}", flush=True)
    print(f"[E1a1] best_policy={evals['best_policy']}", flush=True)
    print(f"[E1a1] best_macro_f1={evals['best_metrics']['macro_f1']:.6f}", flush=True)
    print(f"[E1a1] best_transition={evals['best_transition']}", flush=True)


if __name__ == "__main__":
    main()
