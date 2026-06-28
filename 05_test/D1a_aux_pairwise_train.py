#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D1a_aux_pairwise_train.py

D1a — Auxiliary pairwise heads on captured CLS representation.

Purpose
-------
Train the official C2+D3 baseline with an auxiliary pairwise supervision loss:

  main 4-class loss
  + lambda_pair * auxiliary BCE loss over hard malware pairs:
      Ransomware vs Spyware
      Ransomware vs Trojan
      Spyware vs Trojan

Important:
  - Inference/evaluation still uses ONLY the main 4-class logits.
  - Pairwise heads are auxiliary training signals, not post-hoc rerankers.
  - This script does NOT modify 02_src or 03_outputs/06_model.
  - It imports the official D3C2D3Transformer and captures the representation
    passed into its classifier via a forward pre-hook.

Why hook instead of editing 02_src?
-----------------------------------
D1a1c showed official forward works as:
  forward(tokens, values, return_info=False)
and return_info only exposes shapes, not the actual CLS tensor.
So this script captures classifier input without editing the official source.

Recommended workflow
--------------------
Local smoke test first:
  python 05_test/D1a_aux_pairwise_train.py --repo-root . --smoke-test ...

Then full Kaggle GPU run:
  python 05_test/D1a_aux_pairwise_train.py --repo-root . --epochs 80 --device cuda ...

Outputs
-------
  D1a_summary.md
  D1a_history.csv
  D1a_val_classification_report_best.json
  D1a_val_confusion_matrix_best.csv
  D1a_val_predictions_best.csv
  D1a_transition_summary.json
  D1a_pair_fix_damage_summary.csv
  D1a_aux_pair_metrics_best.csv
  D1a_config.json
  best_model.pt
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_LABEL_ORDER = ["Benign", "Ransomware", "Spyware", "Trojan"]
DEFAULT_HARD_PAIRS = [
    ("Ransomware", "Spyware"),
    ("Ransomware", "Trojan"),
    ("Spyware", "Trojan"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="D1a auxiliary pairwise-head training.")

    # Paths
    p.add_argument("--repo-root", default=".")
    p.add_argument("--model-py", default="02_src/06_model.py")
    p.add_argument("--dataset-npz", default="03_outputs/05_dataset/dataset.npz")
    p.add_argument("--metadata-json", default="03_outputs/05_dataset/metadata.json")
    p.add_argument("--baseline-config-json", default="03_outputs/06_model/config.json")
    p.add_argument("--baseline-pred-csv", default="05_test/outputs/B0_wrong_top2_audit/val_predictions_with_probs.csv")
    p.add_argument("--train-raw-csv", default="01_split/train_raw.csv")
    p.add_argument("--val-raw-csv", default="01_split/val_raw.csv")
    p.add_argument("--out-dir", default="05_test/outputs/D1a_aux_pairwise_lam0p10_smoke")

    # Train
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lambda-pair", type=float, default=0.10)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--use-class-weights", action="store_true", default=True)
    p.add_argument("--no-class-weights", dest="use_class_weights", action="store_false")
    p.add_argument("--grad-clip", type=float, default=1.0)

    # Smoke
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--max-train-rows", type=int, default=None)
    p.add_argument("--max-val-rows", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=0)

    # Model override, defaults read from config
    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--aux-hidden-dim", type=int, default=0,
                   help="0 means Linear(hidden_dim->1). >0 means Linear-GELU-Dropout-Linear.")
    p.add_argument("--aux-dropout", type=float, default=0.1)

    # Input value channels
    p.add_argument("--value-order", default="offset_cont_mask",
                   choices=["offset_cont_mask"],
                   help="Official D3 expected values [offset, raw_scaled_continuous, mask].")

    return p.parse_args()


def repo_path(root: Path, p: str | Path) -> Path:
    q = Path(p)
    return q if q.is_absolute() else root / q


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def cfg(config: Dict[str, Any], key: str, default: Any) -> Any:
    if key in config:
        return config[key]
    model_config = config.get("model_config")
    if isinstance(model_config, dict) and key in model_config:
        return model_config[key]
    training_config = config.get("training_config")
    if isinstance(training_config, dict) and key in training_config:
        return training_config[key]
    return default


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str):
    import torch

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("[WARN] cuda requested but unavailable; using cpu")
        return torch.device("cpu")
    return torch.device(device_arg)


def import_model_class(model_py: Path):
    spec = importlib.util.spec_from_file_location("d1a_official_model", model_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import model from {model_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "D3C2D3Transformer")


def instantiate_official_model(cls, config: Dict[str, Any], *, n_features: int, num_classes: int, num_bins: int):
    kwargs = dict(
        num_bins=int(cfg(config, "num_bins", num_bins)),
        n_features=int(cfg(config, "n_features", n_features)),
        num_classes=int(cfg(config, "num_classes", num_classes)),
        value_dim=int(cfg(config, "value_dim", 32)),
        feature_dim=int(cfg(config, "feature_dim", 32)),
        hidden_dim=int(cfg(config, "hidden_dim", 128)),
        num_layers=int(cfg(config, "num_layers", 3)),
        num_heads=int(cfg(config, "num_heads", 4)),
        dropout=float(cfg(config, "dropout", 0.1)),
        classifier_hidden_dim=int(cfg(config, "classifier_hidden_dim", 128)),
        classifier_dropout=float(cfg(config, "classifier_dropout", 0.1)),
        norm_first=bool(cfg(config, "norm_first", True)),
        gate_init=float(cfg(config, "gate_init", 0.0)),
    )
    return cls(**kwargs), kwargs


def compute_raw_scaled_from_csv(train_raw_csv: Path, val_raw_csv: Path, feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Recompute train-only minmax raw_scaled continuous. Constants are set to 0.5.
    This mirrors the prior B0/B2 reconstruction approach when dataset.npz lacks X_*_continuous.
    """
    if not train_raw_csv.exists() or not val_raw_csv.exists():
        raise FileNotFoundError(
            f"Missing raw CSVs for continuous reconstruction: {train_raw_csv}, {val_raw_csv}"
        )

    train_df = pd.read_csv(train_raw_csv)
    val_df = pd.read_csv(val_raw_csv)

    missing_train = [f for f in feature_names if f not in train_df.columns]
    missing_val = [f for f in feature_names if f not in val_df.columns]
    if missing_train or missing_val:
        raise KeyError(f"Missing feature columns. train={missing_train[:10]}, val={missing_val[:10]}")

    Xtr = train_df[feature_names].to_numpy(dtype=np.float32)
    Xva = val_df[feature_names].to_numpy(dtype=np.float32)

    mn = np.nanmin(Xtr, axis=0)
    mx = np.nanmax(Xtr, axis=0)
    denom = mx - mn
    const = denom <= 1e-12
    denom_safe = denom.copy()
    denom_safe[const] = 1.0

    Xtr_s = (Xtr - mn) / denom_safe
    Xva_s = (Xva - mn) / denom_safe
    Xtr_s[:, const] = 0.5
    Xva_s[:, const] = 0.5

    Xtr_s = np.clip(Xtr_s, 0.0, 1.0).astype(np.float32)
    Xva_s = np.clip(Xva_s, 0.0, 1.0).astype(np.float32)

    info = {
        "source": "raw_scaled_from_train_val_raw_csv",
        "train_raw_csv": str(train_raw_csv),
        "val_raw_csv": str(val_raw_csv),
        "n_constant_features": int(const.sum()),
        "constant_features": [feature_names[i] for i in np.where(const)[0].tolist()],
        "scale": "train_only_minmax_linear_clip_val_constants_0p5",
    }
    return Xtr_s, Xva_s, info


def load_arrays(args: argparse.Namespace, repo_root: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    dataset_npz = repo_path(repo_root, args.dataset_npz)
    train_raw_csv = repo_path(repo_root, args.train_raw_csv)
    val_raw_csv = repo_path(repo_root, args.val_raw_csv)

    with np.load(dataset_npz, allow_pickle=True) as z:
        X_train_bin = z["X_train_bin"].astype(np.int64)
        X_train_offset = z["X_train_offset"].astype(np.float32)
        y_train = z["y_train"].astype(np.int64)

        X_val_bin = z["X_val_bin"].astype(np.int64)
        X_val_offset = z["X_val_offset"].astype(np.float32)
        y_val = z["y_val"].astype(np.int64)

        feature_names = [str(x) for x in z["feature_names"].tolist()]
        label_names = [str(x) for x in z["label_names"].tolist()]
        num_bins = int(z["num_bins"][0]) if "num_bins" in z.files else int(z["K"][0])

        has_cont = "X_train_continuous" in z.files and "X_val_continuous" in z.files
        if has_cont:
            X_train_cont = z["X_train_continuous"].astype(np.float32)
            X_val_cont = z["X_val_continuous"].astype(np.float32)
            cont_info = {"source": "dataset_npz_X_continuous"}
        else:
            X_train_cont, X_val_cont, cont_info = compute_raw_scaled_from_csv(
                train_raw_csv,
                val_raw_csv,
                feature_names,
            )

        has_mask = "X_train_mask" in z.files and "X_val_mask" in z.files
        if has_mask:
            X_train_mask = z["X_train_mask"].astype(np.float32)
            X_val_mask = z["X_val_mask"].astype(np.float32)
            mask_info = {"source": "dataset_npz_X_mask"}
        else:
            X_train_mask = np.ones_like(X_train_offset, dtype=np.float32)
            X_val_mask = np.ones_like(X_val_offset, dtype=np.float32)
            mask_info = {"source": "ones_like_offset"}

    # Apply smoke/max rows after reconstructing full arrays.
    if args.smoke_test:
        if args.max_train_rows is None:
            args.max_train_rows = 2048
        if args.max_val_rows is None:
            args.max_val_rows = 1024
        if args.epochs < 1:
            args.epochs = 1

    if args.max_train_rows is not None:
        n = int(args.max_train_rows)
        X_train_bin = X_train_bin[:n]
        X_train_offset = X_train_offset[:n]
        X_train_cont = X_train_cont[:n]
        X_train_mask = X_train_mask[:n]
        y_train = y_train[:n]

    if args.max_val_rows is not None:
        n = int(args.max_val_rows)
        X_val_bin = X_val_bin[:n]
        X_val_offset = X_val_offset[:n]
        X_val_cont = X_val_cont[:n]
        X_val_mask = X_val_mask[:n]
        y_val = y_val[:n]

    arrays = {
        "X_train_bin": X_train_bin,
        "X_train_offset": X_train_offset,
        "X_train_cont": X_train_cont,
        "X_train_mask": X_train_mask,
        "y_train": y_train,
        "X_val_bin": X_val_bin,
        "X_val_offset": X_val_offset,
        "X_val_cont": X_val_cont,
        "X_val_mask": X_val_mask,
        "y_val": y_val,
    }

    info = {
        "dataset_npz": str(dataset_npz),
        "feature_names": feature_names,
        "label_names": label_names,
        "num_bins": num_bins,
        "n_features": int(X_train_bin.shape[1]),
        "num_classes": int(len(label_names)),
        "continuous_info": cont_info,
        "mask_info": mask_info,
        "shapes": {k: list(v.shape) for k, v in arrays.items()},
    }
    return arrays, info


def make_values(offset: np.ndarray, cont: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.stack([offset.astype(np.float32), cont.astype(np.float32), mask.astype(np.float32)], axis=-1)


class PackedDataset:
    def __init__(self, tokens: np.ndarray, values: np.ndarray, y: np.ndarray):
        self.tokens = tokens.astype(np.int64)
        self.values = values.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self) -> int:
        return int(len(self.y))

    def __getitem__(self, idx: int):
        import torch
        return (
            torch.as_tensor(self.tokens[idx], dtype=torch.long),
            torch.as_tensor(self.values[idx], dtype=torch.float32),
            torch.as_tensor(self.y[idx], dtype=torch.long),
        )


class FocalLoss:
    def __init__(self, gamma: float = 2.0, weight=None, label_smoothing: float = 0.0):
        import torch
        import torch.nn.functional as F

        self.gamma = float(gamma)
        self.weight = weight
        self.label_smoothing = float(label_smoothing)
        self.F = F
        self.torch = torch

    def __call__(self, logits, target):
        ce = self.F.cross_entropy(
            logits,
            target,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        if self.gamma <= 0:
            return ce.mean()
        pt = self.torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()


def class_weights_from_y(y: np.ndarray, num_classes: int, device):
    import torch
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    weights = len(y) / (num_classes * np.maximum(counts, 1.0))
    weights = weights / weights.mean()
    return torch.as_tensor(weights, dtype=torch.float32, device=device), counts.tolist()


def make_aux_head(hidden_dim: int, aux_hidden_dim: int, aux_dropout: float):
    import torch.nn as nn

    if aux_hidden_dim and aux_hidden_dim > 0:
        return nn.Sequential(
            nn.Linear(hidden_dim, aux_hidden_dim),
            nn.GELU(),
            nn.Dropout(aux_dropout),
            nn.Linear(aux_hidden_dim, 1),
        )
    return nn.Linear(hidden_dim, 1)


class D1aAuxPairwiseWrapper:
    """
    Wrapper around official model.
    Captures classifier input via forward pre-hook as CLS representation.
    """
    def __init__(
        self,
        base_model,
        *,
        hidden_dim: int,
        label_names: List[str],
        hard_pairs: List[Tuple[str, str]],
        aux_hidden_dim: int = 0,
        aux_dropout: float = 0.1,
    ):
        import torch
        import torch.nn as nn

        class _Module(nn.Module):
            pass

        self.module = _Module()
        self.module.base = base_model
        self.module.label_names = label_names
        self.module.hard_pairs = hard_pairs
        self.module.label_to_id = {name: i for i, name in enumerate(label_names)}
        self.module.pair_id_names = []
        self.module.aux_heads = nn.ModuleDict()

        for a, b in hard_pairs:
            if a not in self.module.label_to_id or b not in self.module.label_to_id:
                raise KeyError(f"Pair labels not in label_names: {a}, {b}, label_names={label_names}")
            key = f"{a}__vs__{b}"
            self.module.pair_id_names.append((key, self.module.label_to_id[a], self.module.label_to_id[b], a, b))
            self.module.aux_heads[key] = make_aux_head(hidden_dim, aux_hidden_dim, aux_dropout)

        self.module._captured_cls = None
        self.module._capture_module_name = None

        def pre_hook(mod, inputs):
            if not inputs:
                return
            x = inputs[0]
            try:
                if x is not None and getattr(x, "ndim", None) == 2:
                    self.module._captured_cls = x
            except Exception:
                pass

        # Prefer the official classifier module input.
        if hasattr(base_model, "classifier"):
            target = getattr(base_model, "classifier")
            self.module._capture_module_name = "base.classifier"
            self.module._capture_handle = target.register_forward_pre_hook(pre_hook)
        else:
            # Fallback: hook last module whose name contains classifier/head.
            target_name = None
            target_mod = None
            for name, mod in base_model.named_modules():
                low = name.lower()
                if "classifier" in low or low.endswith("head"):
                    target_name, target_mod = name, mod
            if target_mod is None:
                raise RuntimeError("Cannot locate classifier module to capture CLS.")
            self.module._capture_module_name = "base." + target_name
            self.module._capture_handle = target_mod.register_forward_pre_hook(pre_hook)

        def forward(tokens, values, return_aux: bool = True):
            self.module._captured_cls = None
            logits = self.module.base(tokens, values)
            if isinstance(logits, tuple):
                logits = logits[0]
            cls = self.module._captured_cls
            if cls is None:
                raise RuntimeError(
                    "Failed to capture CLS/classifier input. "
                    f"capture_module={self.module._capture_module_name}"
                )
            aux_logits = {}
            if return_aux:
                for key, _, _, _, _ in self.module.pair_id_names:
                    aux_logits[key] = self.module.aux_heads[key](cls).squeeze(-1)
            return logits, aux_logits, cls

        self.module.forward = forward


def aux_pairwise_loss(aux_logits: Dict[str, Any], y, pair_id_names: List[Tuple[str, int, int, str, str]]):
    import torch
    import torch.nn.functional as F

    losses = []
    details = {}
    for key, ida, idb, a, b in pair_id_names:
        mask = (y == ida) | (y == idb)
        n = int(mask.sum().item())
        if n == 0:
            details[key] = {"n": 0, "loss": None}
            continue
        target = (y[mask] == idb).float()
        loss = F.binary_cross_entropy_with_logits(aux_logits[key][mask], target)
        losses.append(loss)
        details[key] = {"n": n, "loss": float(loss.detach().cpu().item())}

    if not losses:
        return torch.zeros((), dtype=torch.float32, device=y.device), details
    return torch.stack(losses).mean(), details


def sklearn_metrics(y_true: np.ndarray, y_pred: np.ndarray, label_names: List[str]) -> Tuple[Dict[str, Any], Dict[str, Any], pd.DataFrame]:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    labels = list(range(len(label_names)))
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
    }
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true_{x}" for x in label_names], columns=[f"pred_{x}" for x in label_names])
    return metrics, report, cm_df


def evaluate_model(model, loader, device, label_names: List[str], pair_id_names):
    import torch
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    model.eval()
    all_y = []
    all_pred = []
    all_logits = []
    aux_collect: Dict[str, Dict[str, List[float]]] = {
        key: {"y": [], "score": []}
        for key, *_ in pair_id_names
    }

    with torch.no_grad():
        for tokens, values, y in loader:
            tokens = tokens.to(device)
            values = values.to(device)
            y = y.to(device)

            logits, aux_logits, _cls = model(tokens, values, return_aux=True)
            pred = logits.argmax(dim=1)

            all_y.append(y.detach().cpu().numpy())
            all_pred.append(pred.detach().cpu().numpy())
            all_logits.append(logits.detach().cpu().numpy())

            for key, ida, idb, a, b in pair_id_names:
                mask = (y == ida) | (y == idb)
                if int(mask.sum().item()) == 0:
                    continue
                yy = (y[mask] == idb).long().detach().cpu().numpy()
                ss = aux_logits[key][mask].detach().cpu().numpy()
                aux_collect[key]["y"].extend(yy.tolist())
                aux_collect[key]["score"].extend(ss.tolist())

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred)
    logits_np = np.concatenate(all_logits)

    metrics, report, cm_df = sklearn_metrics(y_true, y_pred, label_names)

    aux_rows = []
    for key, data in aux_collect.items():
        yy = np.asarray(data["y"], dtype=int)
        ss = np.asarray(data["score"], dtype=float)
        if len(yy) == 0:
            aux_rows.append({"pair_key": key, "n": 0})
            continue
        pp = (ss >= 0).astype(int)
        row = {
            "pair_key": key,
            "n": int(len(yy)),
            "accuracy": float(accuracy_score(yy, pp)),
            "macro_f1": float(f1_score(yy, pp, average="macro", zero_division=0)),
        }
        try:
            row["auc"] = float(roc_auc_score(yy, ss))
        except Exception:
            row["auc"] = float("nan")
        aux_rows.append(row)

    aux_df = pd.DataFrame(aux_rows)
    return metrics, report, cm_df, y_true, y_pred, logits_np, aux_df


def load_baseline_preds(path: Path, label_names: List[str], n_val: int) -> Optional[np.ndarray]:
    if not path.exists():
        return None

    df = pd.read_csv(path)
    if len(df) < n_val:
        return None
    df = df.iloc[:n_val].copy()

    if "pred_id" in df.columns:
        return df["pred_id"].to_numpy(dtype=int)

    if "pred_label" in df.columns:
        label_to_id = {x: i for i, x in enumerate(label_names)}
        return df["pred_label"].map(lambda x: label_to_id[str(x)]).to_numpy(dtype=int)

    return None


def transition_summary(y_true: np.ndarray, baseline_pred: Optional[np.ndarray], new_pred: np.ndarray) -> Dict[str, Any]:
    if baseline_pred is None:
        return {"available": False, "reason": "baseline predictions not found or incompatible"}

    base_correct = baseline_pred == y_true
    new_correct = new_pred == y_true

    wrong_to_correct = (~base_correct) & new_correct
    correct_to_wrong = base_correct & (~new_correct)
    wrong_to_wrong = (~base_correct) & (~new_correct)
    correct_to_correct = base_correct & new_correct

    return {
        "available": True,
        "n": int(len(y_true)),
        "baseline_correct": int(base_correct.sum()),
        "new_correct": int(new_correct.sum()),
        "wrong_to_correct_n": int(wrong_to_correct.sum()),
        "correct_to_wrong_n": int(correct_to_wrong.sum()),
        "wrong_to_wrong_n": int(wrong_to_wrong.sum()),
        "correct_to_correct_n": int(correct_to_correct.sum()),
        "net_gain_n": int(wrong_to_correct.sum() - correct_to_wrong.sum()),
        "damage_ratio": float(correct_to_wrong.sum() / wrong_to_correct.sum()) if int(wrong_to_correct.sum()) else float("nan"),
    }


def pair_fix_damage_summary(
    y_true: np.ndarray,
    baseline_pred: Optional[np.ndarray],
    new_pred: np.ndarray,
    label_names: List[str],
    hard_pairs: List[Tuple[str, str]],
) -> pd.DataFrame:
    if baseline_pred is None:
        return pd.DataFrame([{"available": False, "reason": "baseline predictions unavailable"}])

    label_to_id = {x: i for i, x in enumerate(label_names)}
    rows = []

    for a, b in hard_pairs:
        ida, idb = label_to_id[a], label_to_id[b]
        pair_true = (y_true == ida) | (y_true == idb)
        base_correct = baseline_pred == y_true
        new_correct = new_pred == y_true

        direct_wrong = (
            ((y_true == ida) & (baseline_pred == idb)) |
            ((y_true == idb) & (baseline_pred == ida))
        )
        fixed = direct_wrong & new_correct
        damage_pair = pair_true & base_correct & (~new_correct)

        rows.append({
            "pair": f"{a}<->{b}",
            "direction": "BIDIRECTIONAL",
            "pair_true_n": int(pair_true.sum()),
            "baseline_direct_wrong_n": int(direct_wrong.sum()),
            "fixed_direct_wrong_n": int(fixed.sum()),
            "fix_rate_among_direct_wrong": float(fixed.sum() / direct_wrong.sum()) if int(direct_wrong.sum()) else float("nan"),
            "baseline_correct_pair_n": int((pair_true & base_correct).sum()),
            "correct_to_wrong_damage_n": int(damage_pair.sum()),
            "damage_rate_among_pair_correct": float(damage_pair.sum() / (pair_true & base_correct).sum()) if int((pair_true & base_correct).sum()) else float("nan"),
            "net_pair_gain_n": int(fixed.sum() - damage_pair.sum()),
        })

        for true_id, pred_id, true_name, pred_name in [(ida, idb, a, b), (idb, ida, b, a)]:
            direct_dir = (y_true == true_id) & (baseline_pred == pred_id)
            fixed_dir = direct_dir & new_correct
            damage_dir = (y_true == true_id) & base_correct & (~new_correct)

            rows.append({
                "pair": f"{a}<->{b}",
                "direction": f"{true_name}->{pred_name}",
                "pair_true_n": int((y_true == true_id).sum()),
                "baseline_direct_wrong_n": int(direct_dir.sum()),
                "fixed_direct_wrong_n": int(fixed_dir.sum()),
                "fix_rate_among_direct_wrong": float(fixed_dir.sum() / direct_dir.sum()) if int(direct_dir.sum()) else float("nan"),
                "baseline_correct_pair_n": int(((y_true == true_id) & base_correct).sum()),
                "correct_to_wrong_damage_n": int(damage_dir.sum()),
                "damage_rate_among_pair_correct": float(damage_dir.sum() / ((y_true == true_id) & base_correct).sum()) if int(((y_true == true_id) & base_correct).sum()) else float("nan"),
                "net_pair_gain_n": int(fixed_dir.sum() - damage_dir.sum()),
            })

    return pd.DataFrame(rows)


def make_summary_md(
    *,
    config: Dict[str, Any],
    best_metrics: Dict[str, Any],
    transition: Dict[str, Any],
    pair_df: pd.DataFrame,
    aux_df: pd.DataFrame,
    out_files: List[Path],
) -> str:
    lines = []
    lines.append("# D1a — Auxiliary pairwise head training")
    lines.append("")
    lines.append("## Guardrail")
    lines.append("")
    lines.append("- This is an isolated D1a test under `05_test`.")
    lines.append("- It does not modify `02_src`.")
    lines.append("- It does not modify `03_outputs/06_model`.")
    lines.append("- Inference uses only the main 4-class logits; auxiliary pair heads are training losses, not post-hoc rerankers.")
    lines.append("")
    lines.append("## Config")
    lines.append("")
    for k in [
        "smoke_test", "epochs", "batch_size", "lr", "weight_decay",
        "lambda_pair", "focal_gamma", "label_smoothing", "device",
        "max_train_rows", "max_val_rows", "capture_module_name",
    ]:
        lines.append(f"- `{k}`: {config.get(k)}")
    lines.append("")
    lines.append("## Best validation metrics")
    lines.append("")
    for k, v in best_metrics.items():
        if k not in ("report",):
            lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("## Baseline-vs-D1a transition")
    lines.append("")
    for k, v in transition.items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("## Pair-level fix/damage")
    lines.append("")
    try:
        lines.append(pair_df.to_markdown(index=False))
    except Exception:
        lines.append(pair_df.to_string(index=False))
    lines.append("")
    lines.append("## Auxiliary pair head metrics at best epoch")
    lines.append("")
    try:
        lines.append(aux_df.to_markdown(index=False))
    except Exception:
        lines.append(aux_df.to_string(index=False))
    lines.append("")
    lines.append("## How to judge")
    lines.append("")
    lines.append("- Good D1a must show `wrong_to_correct_n > correct_to_wrong_n`, clear positive net gain, and at least 2/3 hard pairs with positive `net_pair_gain_n`.")
    lines.append("- If macro-F1 improves but correct-to-wrong damage is high, this direction is not safe.")
    lines.append("- Smoke-test results only verify code path; full conclusion requires Kaggle/full run.")
    lines.append("")
    lines.append("## Generated files")
    lines.append("")
    for p in out_files:
        lines.append(f"- `{p}`")
    lines.append("")
    return "\n".join(lines)


def zip_outputs(out_dir: Path) -> Path:
    out_zip = out_dir / "D1a_aux_pairwise_train_output.zip"
    if out_zip.exists():
        out_zip.unlink()
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fp in out_dir.rglob("*"):
            if fp.is_file() and fp != out_zip:
                z.write(fp, fp.relative_to(out_dir))
    return out_zip


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    out_dir = repo_path(repo_root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from torch.utils.data import DataLoader

    set_seed(args.seed)
    device = resolve_device(args.device)

    arrays, data_info = load_arrays(args, repo_root)
    label_names = data_info["label_names"]
    num_classes = data_info["num_classes"]
    hidden_dim = int(args.hidden_dim or cfg(read_json(repo_path(repo_root, args.baseline_config_json)), "hidden_dim", 128))

    train_values = make_values(arrays["X_train_offset"], arrays["X_train_cont"], arrays["X_train_mask"])
    val_values = make_values(arrays["X_val_offset"], arrays["X_val_cont"], arrays["X_val_mask"])

    train_ds = PackedDataset(arrays["X_train_bin"], train_values, arrays["y_train"])
    val_ds = PackedDataset(arrays["X_val_bin"], val_values, arrays["y_val"])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    baseline_config = read_json(repo_path(repo_root, args.baseline_config_json))
    model_cls = import_model_class(repo_path(repo_root, args.model_py))
    base_model, model_kwargs = instantiate_official_model(
        model_cls,
        baseline_config,
        n_features=data_info["n_features"],
        num_classes=num_classes,
        num_bins=data_info["num_bins"],
    )

    wrapper = D1aAuxPairwiseWrapper(
        base_model,
        hidden_dim=hidden_dim,
        label_names=label_names,
        hard_pairs=DEFAULT_HARD_PAIRS,
        aux_hidden_dim=args.aux_hidden_dim,
        aux_dropout=args.aux_dropout,
    )
    model = wrapper.module.to(device)

    weight_tensor = None
    class_counts = None
    if args.use_class_weights:
        weight_tensor, class_counts = class_weights_from_y(arrays["y_train"], num_classes, device)
    criterion = FocalLoss(
        gamma=args.focal_gamma,
        weight=weight_tensor,
        label_smoothing=args.label_smoothing,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_macro = -1.0
    best_state = None
    best_epoch = None
    best_payload = None

    start_time = time.time()
    print(f"[D1a] device={device} train={len(train_ds)} val={len(val_ds)} epochs={args.epochs}")
    print(f"[D1a] capture_module={model._capture_module_name} lambda_pair={args.lambda_pair}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_main = 0.0
        total_pair = 0.0
        total_n = 0

        for tokens, values, y in train_loader:
            tokens = tokens.to(device)
            values = values.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits, aux_logits, _cls = model(tokens, values, return_aux=True)

            main_loss = criterion(logits, y)
            pair_loss, pair_details = aux_pairwise_loss(aux_logits, y, model.pair_id_names)
            loss = main_loss + float(args.lambda_pair) * pair_loss

            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            bs = int(y.numel())
            total_n += bs
            total_loss += float(loss.detach().cpu().item()) * bs
            total_main += float(main_loss.detach().cpu().item()) * bs
            total_pair += float(pair_loss.detach().cpu().item()) * bs

        val_metrics, val_report, val_cm_df, y_true, y_pred, logits_np, aux_df = evaluate_model(
            model,
            val_loader,
            device,
            label_names,
            model.pair_id_names,
        )

        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_n, 1),
            "train_main_loss": total_main / max(total_n, 1),
            "train_pair_loss": total_pair / max(total_n, 1),
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
        }
        history.append(row)
        print(
            f"[D1a][epoch {epoch:03d}] "
            f"loss={row['train_loss']:.5f} main={row['train_main_loss']:.5f} pair={row['train_pair_loss']:.5f} "
            f"val_acc={row['val_accuracy']:.5f} val_macro={row['val_macro_f1']:.5f}"
        )

        if val_metrics["macro_f1"] > best_macro:
            best_macro = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {
                "model_state_dict": model.state_dict(),
                "model_kwargs": model_kwargs,
                "label_names": label_names,
                "hard_pairs": DEFAULT_HARD_PAIRS,
                "hidden_dim": hidden_dim,
                "aux_hidden_dim": args.aux_hidden_dim,
                "aux_dropout": args.aux_dropout,
                "lambda_pair": args.lambda_pair,
                "epoch": epoch,
                "val_metrics": val_metrics,
            }
            best_payload = {
                "metrics": val_metrics,
                "report": val_report,
                "cm_df": val_cm_df,
                "y_true": y_true,
                "y_pred": y_pred,
                "logits": logits_np,
                "aux_df": aux_df,
            }

    assert best_payload is not None and best_state is not None

    # Baseline comparison.
    baseline_preds = load_baseline_preds(repo_path(repo_root, args.baseline_pred_csv), label_names, len(best_payload["y_true"]))
    trans = transition_summary(best_payload["y_true"], baseline_preds, best_payload["y_pred"])
    pair_df = pair_fix_damage_summary(
        best_payload["y_true"],
        baseline_preds,
        best_payload["y_pred"],
        label_names,
        DEFAULT_HARD_PAIRS,
    )

    # Save outputs.
    history_df = pd.DataFrame(history)
    history_path = out_dir / "D1a_history.csv"
    report_path = out_dir / "D1a_val_classification_report_best.json"
    cm_path = out_dir / "D1a_val_confusion_matrix_best.csv"
    preds_path = out_dir / "D1a_val_predictions_best.csv"
    aux_path = out_dir / "D1a_aux_pair_metrics_best.csv"
    trans_path = out_dir / "D1a_transition_summary.json"
    pair_path = out_dir / "D1a_pair_fix_damage_summary.csv"
    config_path = out_dir / "D1a_config.json"
    model_path = out_dir / "best_model.pt"
    summary_path = out_dir / "D1a_summary.md"

    history_df.to_csv(history_path, index=False)
    write_json(report_path, best_payload["report"])
    best_payload["cm_df"].to_csv(cm_path)
    best_payload["aux_df"].to_csv(aux_path, index=False)
    write_json(trans_path, trans)
    pair_df.to_csv(pair_path, index=False)

    pred_df = pd.DataFrame({
        "sample_index": np.arange(len(best_payload["y_true"])),
        "true_id": best_payload["y_true"],
        "true_label": [label_names[i] for i in best_payload["y_true"]],
        "pred_id": best_payload["y_pred"],
        "pred_label": [label_names[i] for i in best_payload["y_pred"]],
        "correct": best_payload["y_true"] == best_payload["y_pred"],
    })
    for i, name in enumerate(label_names):
        pred_df[f"logit_{name}"] = best_payload["logits"][:, i]
    pred_df.to_csv(preds_path, index=False)

    torch.save(best_state, model_path)

    run_config = {
        "stage": "D1a_aux_pairwise_head_train",
        "smoke_test": bool(args.smoke_test),
        "repo_root": str(repo_root),
        "out_dir": str(out_dir),
        "epochs": int(args.epochs),
        "best_epoch": int(best_epoch),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "lambda_pair": float(args.lambda_pair),
        "focal_gamma": float(args.focal_gamma),
        "label_smoothing": float(args.label_smoothing),
        "use_class_weights": bool(args.use_class_weights),
        "class_counts": class_counts,
        "device": str(device),
        "seed": int(args.seed),
        "max_train_rows": args.max_train_rows,
        "max_val_rows": args.max_val_rows,
        "value_order": args.value_order,
        "capture_module_name": model._capture_module_name,
        "model_kwargs": model_kwargs,
        "data_info": data_info,
        "baseline_pred_csv": str(repo_path(repo_root, args.baseline_pred_csv)),
        "transition_summary": trans,
    }
    write_json(config_path, run_config)

    best_metrics = {
        "best_epoch": int(best_epoch),
        **best_payload["metrics"],
        "elapsed_seconds": float(time.time() - start_time),
    }

    out_files = [
        summary_path,
        history_path,
        report_path,
        cm_path,
        preds_path,
        aux_path,
        trans_path,
        pair_path,
        config_path,
        model_path,
    ]

    summary_md = make_summary_md(
        config=run_config,
        best_metrics=best_metrics,
        transition=trans,
        pair_df=pair_df,
        aux_df=best_payload["aux_df"],
        out_files=out_files,
    )
    summary_path.write_text(summary_md, encoding="utf-8")
    out_zip = zip_outputs(out_dir)

    print("===== D1a auxiliary pairwise train done =====")
    print("summary:", summary_path)
    print("best_model:", model_path)
    print("zip:", out_zip)
    print("best_epoch:", best_epoch)
    print("best_macro_f1:", best_metrics["macro_f1"])
    print("transition:", trans)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
