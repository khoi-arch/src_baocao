#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch


def set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def pick_device(device: str) -> torch.device:
    device = str(device).lower().strip()

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device == "cuda":
        if not torch.cuda.is_available():
            print("[WARN] cuda requested but not available; fallback to cpu")
            return torch.device("cpu")
        return torch.device("cuda")

    return torch.device("cpu")


def save_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def compute_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    y = np.asarray(y, dtype=np.int64)
    counts = np.bincount(y, minlength=int(num_classes)).astype(np.float64)
    counts = np.maximum(counts, 1.0)

    total = float(counts.sum())
    weights = total / (float(num_classes) * counts)
    weights = weights / np.mean(weights)

    return torch.tensor(weights, dtype=torch.float32)


def compute_epoch_lr(
    *,
    base_lr: float,
    epoch: int,
    total_epochs: int,
    scheduler_name: str,
    warmup_epochs: int,
    min_lr_ratio: float,
) -> float:
    scheduler_name = str(scheduler_name).lower()

    if scheduler_name == "none":
        return float(base_lr)

    if scheduler_name != "warmup_cosine":
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    epoch = int(epoch)
    total_epochs = max(int(total_epochs), 1)
    warmup_epochs = max(int(warmup_epochs), 0)
    min_lr_ratio = float(min_lr_ratio)

    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return float(base_lr) * float(epoch) / float(warmup_epochs)

    denom = max(total_epochs - warmup_epochs, 1)
    progress = float(epoch - warmup_epochs) / float(denom)
    progress = min(max(progress, 0.0), 1.0)

    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    ratio = min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return float(base_lr) * float(ratio)


def set_optimizer_lr(optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def train_one_epoch(
    *,
    model,
    loader,
    criterion,
    optimizer,
    device,
    grad_clip_norm: float,
) -> float:
    model.train()

    total_loss = 0.0
    total_n = 0

    for X, V, y in loader:
        X = X.to(device, non_blocking=True)
        V = V.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(X, V)
        loss = criterion(logits, y)

        loss.backward()

        if grad_clip_norm is not None and float(grad_clip_norm) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))

        optimizer.step()

        bs = int(y.shape[0])
        total_loss += float(loss.detach().cpu()) * bs
        total_n += bs

    return float(total_loss / max(total_n, 1))


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def _classification_metrics(y_true, y_pred, num_classes: int, label_names: List[str]) -> Dict[str, object]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= int(t) < num_classes and 0 <= int(p) < num_classes:
            cm[int(t), int(p)] += 1

    per_class = {}
    f1s = []
    supports = []

    for i in range(num_classes):
        tp = float(cm[i, i])
        fp = float(cm[:, i].sum() - cm[i, i])
        fn = float(cm[i, :].sum() - cm[i, i])
        support = int(cm[i, :].sum())

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2.0 * precision * recall, precision + recall)

        label = str(label_names[i]) if i < len(label_names) else str(i)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

        f1s.append(f1)
        supports.append(support)

    accuracy = float(np.mean(y_true == y_pred)) if len(y_true) else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    supports_arr = np.asarray(supports, dtype=np.float64)
    f1s_arr = np.asarray(f1s, dtype=np.float64)
    weighted_f1 = float(np.sum(f1s_arr * supports_arr) / max(float(supports_arr.sum()), 1.0))

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes: int, label_names: List[str]) -> Dict[str, object]:
    model.eval()

    total_loss = 0.0
    total_n = 0

    all_true = []
    all_pred = []
    all_conf = []
    all_probs = []

    for X, V, y in loader:
        X = X.to(device, non_blocking=True)
        V = V.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(X, V)
        loss = criterion(logits, y)

        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)

        bs = int(y.shape[0])
        total_loss += float(loss.detach().cpu()) * bs
        total_n += bs

        all_true.append(y.detach().cpu().numpy())
        all_pred.append(pred.detach().cpu().numpy())
        all_conf.append(conf.detach().cpu().numpy())
        all_probs.append(probs.detach().cpu().numpy())

    y_true = np.concatenate(all_true) if all_true else np.asarray([], dtype=np.int64)
    y_pred = np.concatenate(all_pred) if all_pred else np.asarray([], dtype=np.int64)
    confidence = np.concatenate(all_conf) if all_conf else np.asarray([], dtype=np.float32)
    probs = np.concatenate(all_probs) if all_probs else np.asarray([], dtype=np.float32)

    metrics = _classification_metrics(y_true, y_pred, int(num_classes), list(label_names))
    metrics["loss"] = float(total_loss / max(total_n, 1))
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred
    metrics["confidence"] = confidence
    metrics["probs"] = probs

    return metrics


def write_history_csv(path, history: List[Dict[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not history:
        return

    fieldnames = list(history[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def write_confusion_outputs(out_dir, split: str, eval_result: Dict[str, object], label_names: List[str]) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cm = np.asarray(eval_result["confusion_matrix"], dtype=np.int64)

    save_json(
        out_dir / f"{split}_confusion_matrix.json",
        {
            "labels": list(label_names),
            "matrix": cm.tolist(),
        },
    )

    csv_path = out_dir / f"{split}_confusion_matrix.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + list(label_names))
        for label, row in zip(label_names, cm.tolist()):
            writer.writerow([label] + row)


def write_predictions_csv(path, y_true, y_pred, confidence, label_names: List[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    confidence = np.asarray(confidence, dtype=np.float64)

    def label_of(i: int) -> str:
        if 0 <= int(i) < len(label_names):
            return str(label_names[int(i)])
        return str(i)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_index",
            "true_id",
            "true_label",
            "pred_id",
            "pred_label",
            "correct",
            "confidence",
        ])

        for idx, (t, p, c) in enumerate(zip(y_true, y_pred, confidence)):
            writer.writerow([
                idx,
                int(t),
                label_of(int(t)),
                int(p),
                label_of(int(p)),
                int(t) == int(p),
                float(c),
            ])
