#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_utils.py

Exact helper subset copied from old official 05_train.py for D3 reproduction.
This file intentionally keeps old training/evaluation semantics.
"""
from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Reproducible enough for debugging.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def pick_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)

def compute_class_weights(y_train: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y_train.astype(int), minlength=num_classes).astype(np.float64)
    total = float(counts.sum())
    weights = np.zeros(num_classes, dtype=np.float32)
    for i, c in enumerate(counts):
        if c <= 0:
            weights[i] = 0.0
        else:
            weights[i] = total / (num_classes * c)
    return torch.as_tensor(weights, dtype=torch.float32)

def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm

def metrics_from_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    label_names: List[str],
) -> Dict[str, object]:
    cm = confusion_matrix_np(y_true, y_pred, num_classes)
    total = int(cm.sum())
    correct = int(np.trace(cm))
    acc = float(correct / total) if total else 0.0

    per_class: Dict[str, Dict[str, float]] = {}
    f1s = []
    weights = []

    for i in range(num_classes):
        tp = float(cm[i, i])
        fp = float(cm[:, i].sum() - cm[i, i])
        fn = float(cm[i, :].sum() - cm[i, i])
        support = int(cm[i, :].sum())

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        name = label_names[i] if i < len(label_names) else str(i)
        per_class[name] = {
            "class_id": int(i),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

        f1s.append(f1)
        weights.append(support)

    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    weighted_f1 = float(np.average(f1s, weights=weights)) if sum(weights) > 0 else 0.0

    true_counts = {label_names[i] if i < len(label_names) else str(i): int(cm[i, :].sum()) for i in range(num_classes)}
    pred_counts = {label_names[i] if i < len(label_names) else str(i): int(cm[:, i].sum()) for i in range(num_classes)}

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "true_counts": true_counts,
        "pred_counts": pred_counts,
        "confusion_matrix": cm.tolist(),
    }

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    label_names: List[str],
    collect_probs: bool = False,
) -> Dict[str, object]:
    model.eval()

    losses = []
    all_true = []
    all_pred = []
    all_conf = []
    all_prob = []

    for X, Z, y in loader:
        X = X.to(device, non_blocking=True)
        Z = Z.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(X, z_values=Z)
        loss = criterion(logits, y)

        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)

        losses.append(float(loss.item()) * int(X.size(0)))
        all_true.append(y.detach().cpu().numpy())
        all_pred.append(pred.detach().cpu().numpy())
        all_conf.append(conf.detach().cpu().numpy())
        if collect_probs:
            all_prob.append(probs.detach().cpu().numpy())

    y_true = np.concatenate(all_true) if all_true else np.asarray([], dtype=np.int64)
    y_pred = np.concatenate(all_pred) if all_pred else np.asarray([], dtype=np.int64)
    conf = np.concatenate(all_conf) if all_conf else np.asarray([], dtype=np.float32)

    metrics = metrics_from_predictions(y_true, y_pred, num_classes, label_names)
    avg_loss = float(sum(losses) / max(len(y_true), 1))

    out = {
        "loss": avg_loss,
        "y_true": y_true,
        "y_pred": y_pred,
        "confidence": conf,
        **metrics,
    }
    if collect_probs:
        out["probs"] = np.concatenate(all_prob) if all_prob else np.zeros((0, num_classes), dtype=np.float32)
    return out

def compute_epoch_lr(
    *,
    base_lr: float,
    epoch: int,
    total_epochs: int,
    scheduler_name: str,
    warmup_epochs: int,
    min_lr_ratio: float,
) -> float:
    """
    Epoch-level LR schedule.

    none:
        lr = base_lr

    warmup_cosine:
        - warmup: linearly increase LR from base_lr/warmup_epochs to base_lr
        - cosine: decay from base_lr to base_lr * min_lr_ratio

    epoch is 1-indexed.
    """
    if scheduler_name == "none":
        return float(base_lr)

    if scheduler_name != "warmup_cosine":
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    total_epochs = max(int(total_epochs), 1)
    warmup_epochs = max(int(warmup_epochs), 0)
    min_lr_ratio = float(min(max(min_lr_ratio, 0.0), 1.0))

    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return float(base_lr) * float(epoch) / float(warmup_epochs)

    if total_epochs <= warmup_epochs:
        return float(base_lr)

    progress = float(epoch - warmup_epochs) / float(max(total_epochs - warmup_epochs, 1))
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    lr_ratio = min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return float(base_lr) * float(lr_ratio)

def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0

    for X, Z, y in loader:
        X = X.to(device, non_blocking=True)
        Z = Z.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(X, z_values=Z)
        loss = criterion(logits, y)
        loss.backward()

        if grad_clip_norm and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))

        optimizer.step()

        n = int(X.size(0))
        total_loss += float(loss.item()) * n
        total_n += n

    return float(total_loss / max(total_n, 1))

def write_history_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

def write_confusion_outputs(
    out_dir: Path,
    prefix: str,
    report: Dict[str, object],
    label_names: List[str],
) -> None:
    cm = np.asarray(report["confusion_matrix"], dtype=np.int64)

    json_path = out_dir / f"{prefix}_confusion_matrix_best.json"
    csv_path = out_dir / f"{prefix}_confusion_matrix_best.csv"

    json_path.write_text(json.dumps({
        "labels": label_names,
        "matrix": cm.tolist(),
        "note": "rows=true labels, columns=predicted labels",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + label_names)
        for i, row in enumerate(cm.tolist()):
            writer.writerow([label_names[i] if i < len(label_names) else str(i)] + row)

def write_predictions_csv(
    path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    label_names: List[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_index", "true_id", "true_label", "pred_id", "pred_label", "correct", "confidence"])
        for idx, (t, p, c) in enumerate(zip(y_true.astype(int), y_pred.astype(int), confidence.astype(float))):
            true_label = label_names[t] if 0 <= t < len(label_names) else str(t)
            pred_label = label_names[p] if 0 <= p < len(label_names) else str(p)
            writer.writerow([idx, int(t), true_label, int(p), pred_label, int(t == p), float(c)])

def save_json(path: Path, obj: Dict[str, object]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
