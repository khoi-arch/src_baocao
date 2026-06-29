#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_train.py

Final-pipeline D3 trainer for the C2 dataset.

Default clean workflow:
  input : 03_outputs/05_dataset/dataset.npz
          03_outputs/05_dataset/metadata.json
  raw   : 01_split/train_raw.csv, 01_split/val_raw.csv
  output: 03_outputs/06_model/

Compatibility workflow:
  pass --out-root and --run-name to write old-style folders such as
  03_outputs/train_runs_fusion_ablation_D0_D7/Keff512/<run-name>/

The model/training behavior is copied from the old official D3 path:
  10_train_fusion_ablation_D0_D7.py + 05_train.py semantics,
  but locked to D3 only.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import config as CFG
import train_utils as _train_mod


def cfg(name: str, default):
    return getattr(CFG, name, default)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train official D3 only on final C2 dataset. Exact old D3 behavior, no reimplementation.")

    p.add_argument("--run-id", choices=["D3"], default="D3")
    p.add_argument("--K", type=int, default=int(cfg("TOKEN_K", 1000)))
    p.add_argument("--num-bins", type=int, default=int(cfg("VALUE_NUM_BINS", 128)))
    p.add_argument("--dataset-npz", default="")
    p.add_argument("--metadata-json", default="")
    # Default clean pipeline output. Leave --run-name empty so `python 02_src/07_train.py`
    # writes directly to 03_outputs/06_model instead of old train_runs folders.
    p.add_argument("--out-root", default="")
    p.add_argument("--run-name", default="")

    p.add_argument("--train-raw", default="")
    p.add_argument("--val-raw", default="")

    # D5 selective continuous parameters; distribution-only, no labels.
    p.add_argument("--tail-frac", type=float, default=0.10)
    p.add_argument("--wide-quantile", type=float, default=0.90)

    p.add_argument("--seed", type=int, default=int(cfg("TRAIN_SEED", 42)))
    p.add_argument("--device", default=str(cfg("TRAIN_DEVICE", "auto")))

    p.add_argument("--epochs", type=int, default=int(cfg("TRAIN_EPOCHS", 80)))
    p.add_argument("--batch-size", type=int, default=int(cfg("TRAIN_BATCH_SIZE", 256)))
    p.add_argument("--lr", type=float, default=float(cfg("TRAIN_LR", 1e-3)))
    p.add_argument("--weight-decay", type=float, default=float(cfg("TRAIN_WEIGHT_DECAY", 1e-4)))
    p.add_argument("--scheduler", choices=["none", "warmup_cosine"], default=str(cfg("TRAIN_SCHEDULER", "warmup_cosine")))
    p.add_argument("--warmup-epochs", type=int, default=int(cfg("TRAIN_WARMUP_EPOCHS", 8)))
    p.add_argument("--min-lr-ratio", type=float, default=float(cfg("TRAIN_MIN_LR_RATIO", 0.05)))
    p.add_argument("--patience", type=int, default=int(cfg("TRAIN_PATIENCE", 12)))
    p.add_argument("--min-delta", type=float, default=float(cfg("TRAIN_MIN_DELTA", 1e-4)))
    p.add_argument("--num-workers", type=int, default=int(cfg("TRAIN_NUM_WORKERS", 0)))
    p.add_argument("--grad-clip-norm", type=float, default=float(cfg("TRAIN_GRAD_CLIP_NORM", 1.0)))
    p.add_argument("--use-class-weights", action=argparse.BooleanOptionalAction, default=bool(cfg("USE_CLASS_WEIGHTS", True)))

    # Optional soft targets for clean calibration-derived family-aware smoothing.
    # If not provided, trainer behavior remains the official hard-label path.
    p.add_argument("--soft-target-matrix", default="", help="CSV with true_L2,true_L3,target_<class> columns.")
    p.add_argument("--soft-target-gamma", type=float, default=1.0, help="Scale matrix toward one-hot: one_hot + gamma*(matrix-one_hot).")
    p.add_argument("--soft-target-report-name", default="soft_target_info.json")

    # SAM/ASAM optimizer wrapper. rho=0.0 keeps official AdamW behavior.
    p.add_argument("--sam-rho", type=float, default=0.0, help="SAM rho. 0 disables SAM and keeps official trainer behavior.")
    p.add_argument("--sam-adaptive", action=argparse.BooleanOptionalAction, default=False, help="Use ASAM-style adaptive perturbation.")
    p.add_argument("--sam-eps", type=float, default=1e-12, help="Numerical epsilon for SAM norm.")

    # F3e: learned difficulty-gated adaptive hard-negative separation.
    # No fixed hard-pair list, no L3/family training signal, no fixed margin.
    p.add_argument("--gahn-enabled", action=argparse.BooleanOptionalAction, default=False, help="Enable learned difficulty-gated adaptive hard-negative loss.")
    p.add_argument("--gahn-temperature", type=float, default=1.0, help="Softmax temperature over wrong logits.")
    p.add_argument("--gahn-gate-temperature", type=float, default=1.0, help="Temperature for hardness target sigmoid(max_wrong_logit - true_logit).")
    p.add_argument("--gahn-scope", default="all", choices=["all", "malware"], help="all = all classes; malware = only true/confuser in Ransomware/Spyware/Trojan.")
    p.add_argument("--gahn-gate-loss-weight", type=float, default=1.0, help="Weight for BCE gate supervision. Hard-negative scale is learned gate, not a global alpha.")

    p.add_argument("--value-dim", type=int, default=int(cfg("VALUE_EMBED_DIM", 32)))
    p.add_argument("--feature-dim", type=int, default=int(cfg("FEATURE_EMBED_DIM", 32)))
    p.add_argument("--hidden-dim", type=int, default=int(cfg("MODEL_HIDDEN_DIM", 128)))
    p.add_argument("--num-layers", type=int, default=int(cfg("MODEL_NUM_LAYERS", 3)))
    p.add_argument("--num-heads", type=int, default=int(cfg("MODEL_NUM_HEADS", 4)))
    p.add_argument("--dropout", type=float, default=float(cfg("MODEL_DROPOUT", 0.1)))
    p.add_argument("--classifier-hidden-dim", type=int, default=int(cfg("CLASSIFIER_HIDDEN_DIM", 128)))
    p.add_argument("--classifier-dropout", type=float, default=float(cfg("CLASSIFIER_DROPOUT", 0.1)))
    p.add_argument("--norm-first", action=argparse.BooleanOptionalAction, default=bool(cfg("TRANSFORMER_NORM_FIRST", True)))

    # sigmoid(0)=0.5. Conservative default; model can move it up/down per feature.
    p.add_argument("--gate-init", type=float, default=0.0)
    p.add_argument(
        "--diagnostic-only",
        action="store_true",
        help="Build data/model diagnostics and exit before training. Useful for checking boundary collapse and scale diagnostics.",
    )

    return p.parse_args()


RUN_SPECS: Dict[str, Dict[str, str]] = {
    "D0": {
        "local": "offset_interpolation",
        "continuous_source": "none",
        "fusion": "none",
        "description": "offset interpolation only",
    },
    "D1": {
        "local": "offset_interpolation",
        "continuous_source": "raw_scaled",
        "fusion": "raw_scalar_concat",
        "description": "offset interpolation + raw scalar concat",
    },
    "D2": {
        "local": "offset_interpolation",
        "continuous_source": "raw_scaled",
        "fusion": "raw_project_gate_concat",
        "description": "offset interpolation + raw projected gated concat-fusion",
    },
    "D3": {
        "local": "offset_interpolation",
        "continuous_source": "raw_scaled",
        "fusion": "raw_film",
        "description": "offset interpolation + raw FiLM/multiply fusion",
    },
    "D4": {
        "local": "offset_interpolation",
        "continuous_source": "z_preprocessed",
        "fusion": "z_project_gate_concat",
        "description": "offset interpolation + z projected gated concat-fusion",
    },
    "D5": {
        "local": "offset_interpolation",
        "continuous_source": "raw_scaled",
        "fusion": "raw_selective_project_gate_concat",
        "description": "offset interpolation + raw selective wide/tail projected gated concat-fusion",
    },
    "D6": {
        "local": "projected_offset_add",
        "continuous_source": "none",
        "fusion": "none",
        "description": "bin embedding + projected offset add",
    },
    "D7": {
        "local": "projected_offset_concat",
        "continuous_source": "none",
        "fusion": "none",
        "description": "bin embedding + projected offset concat-fusion",
    },
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path_like: str | Path) -> Path:
    """Resolve relative paths from repo root, not from the caller's cwd."""
    p = Path(path_like)
    if p.is_absolute():
        return p
    return repo_root() / p


def default_model_dir() -> Path:
    if hasattr(CFG, "MODEL_DIR"):
        return resolve_path(Path(CFG.MODEL_DIR))
    return resolve_path(cfg("OUTPUT_ROOT", Path("03_outputs")) / "06_model")


def default_dataset_path(K: int, B: int) -> Path:
    # Clean pipeline default: train on the final C2 dataset produced by 05_build_dataset.py.
    # Fallback preserves old official layout.
    if hasattr(CFG, "DATASET_NPZ"):
        return Path(CFG.DATASET_NPZ)
    return cfg("OUTPUT_ROOT", Path("03_outputs")) / "build_mixed_quantile_offset" / f"K{K}_B{B}_C2_selective_rank_discrete_compact" / "mixed_quantile_offset_dataset.npz"


def default_metadata_path(K: int, B: int) -> Path:
    if hasattr(CFG, "DATASET_METADATA"):
        return Path(CFG.DATASET_METADATA)
    return cfg("OUTPUT_ROOT", Path("03_outputs")) / "build_mixed_quantile_offset" / f"K{K}_B{B}_C2_selective_rank_discrete_compact" / "mixed_quantile_offset_metadata.json"


def resolve_repo_path(path_from_meta: str | Path, fallback_relative: str) -> Path:
    p = Path(path_from_meta)
    if p.exists():
        return p

    root = repo_root()
    fallback = root / fallback_relative
    if fallback.exists():
        return fallback

    parts = list(p.parts)
    for anchor in ["03_outputs", "01_split", "00_raw_dataset"]:
        if anchor in parts:
            idx = parts.index(anchor)
            rel = Path(*parts[idx:])
            candidate = root / rel
            if candidate.exists():
                return candidate

    raise FileNotFoundError(f"Could not resolve path. metadata_path={p}, fallback={fallback}")


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


def load_z_preprocessed(meta: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    feature_names = [str(x) for x in meta["feature_names"]]

    train_path = resolve_repo_path(
        meta["input"]["train_preprocessed"],
        "03_outputs/preprocessing/train_preprocessed_K1000.csv",
    )
    val_path = resolve_repo_path(
        meta["input"]["val_preprocessed"],
        "03_outputs/preprocessing/val_preprocessed_K1000.csv",
    )

    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)

    missing_train = [f for f in feature_names if f not in train.columns]
    missing_val = [f for f in feature_names if f not in val.columns]
    if missing_train:
        raise ValueError(f"train_preprocessed missing features: {missing_train[:10]}")
    if missing_val:
        raise ValueError(f"val_preprocessed missing features: {missing_val[:10]}")

    X_train = train.loc[:, feature_names].to_numpy(dtype=np.float32)
    X_val = val.loc[:, feature_names].to_numpy(dtype=np.float32)

    X_train = np.clip(X_train, 0.0, 1.0).astype(np.float32)
    X_val = np.clip(X_val, 0.0, 1.0).astype(np.float32)

    info = {
        "source": "z_preprocessed",
        "train_path": str(train_path),
        "val_path": str(val_path),
        "scale": "already_preprocessed_to_0_1",
        "train_min": float(X_train.min()),
        "train_max": float(X_train.max()),
        "val_min": float(X_val.min()),
        "val_max": float(X_val.max()),
    }
    return X_train, X_val, info


def resolve_raw_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    train_path = Path(args.train_raw) if args.train_raw else repo_root() / "01_split" / "train_raw.csv"
    val_path = Path(args.val_raw) if args.val_raw else repo_root() / "01_split" / "val_raw.csv"

    if not train_path.exists():
        train_path = resolve_repo_path(train_path, "01_split/train_raw.csv")
    if not val_path.exists():
        val_path = resolve_repo_path(val_path, "01_split/val_raw.csv")

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


def load_continuous_for_run(
    spec: Dict[str, str],
    meta: Dict[str, object],
    args: argparse.Namespace,
    train_shape: Tuple[int, int],
    val_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    source = spec["continuous_source"]
    if source == "none":
        return (
            np.zeros(train_shape, dtype=np.float32),
            np.zeros(val_shape, dtype=np.float32),
            {"source": "none", "scale": "none"},
        )
    if source == "raw_scaled":
        return load_raw_scaled(meta, args)
    if source == "z_preprocessed":
        return load_z_preprocessed(meta)
    raise ValueError(f"unknown continuous source: {source}")


def build_selective_mask(
    X_train_bin: np.ndarray,
    X_val_bin: np.ndarray,
    X_train_cont: np.ndarray,
    *,
    num_bins: int,
    tail_frac: float,
    wide_quantile: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """
    Distribution-only selective mask.
    Active if:
      - bin index is in lower/upper tail by bin index, OR
      - empirical train width of continuous values inside feature/bin is high.

    No labels are used.
    """
    N, F = X_train_bin.shape
    tail_bins = max(1, int(round(num_bins * float(tail_frac))))

    active_table = np.zeros((F, num_bins), dtype=bool)
    width_table = np.zeros((F, num_bins), dtype=np.float32)

    tail_bin_mask = np.zeros(num_bins, dtype=bool)
    tail_bin_mask[:tail_bins] = True
    tail_bin_mask[num_bins - tail_bins:] = True

    wide_counts = []
    thresholds = []

    for j in range(F):
        bcol = np.clip(X_train_bin[:, j].astype(np.int64), 0, num_bins - 1)
        xcol = X_train_cont[:, j].astype(np.float32)

        widths = np.zeros(num_bins, dtype=np.float32)
        for b in np.unique(bcol):
            vals = xcol[bcol == b]
            if vals.size >= 2:
                widths[b] = float(vals.max() - vals.min())

        width_table[j] = widths
        nonzero = widths[widths > 0.0]
        if nonzero.size:
            thr = float(np.quantile(nonzero, float(wide_quantile)))
            wide_mask = widths >= thr
        else:
            thr = None
            wide_mask = np.zeros(num_bins, dtype=bool)

        active_table[j] = tail_bin_mask | wide_mask
        wide_counts.append(int(wide_mask.sum()))
        thresholds.append(thr)

    def sample_mask(X_bin: np.ndarray) -> np.ndarray:
        Xb = np.clip(X_bin.astype(np.int64), 0, num_bins - 1)
        out = np.zeros(Xb.shape, dtype=np.float32)
        for j in range(F):
            out[:, j] = active_table[j, Xb[:, j]].astype(np.float32)
        return out

    train_mask = sample_mask(X_train_bin)
    val_mask = sample_mask(X_val_bin)

    info = {
        "type": "wide_or_tail_distribution_only",
        "tail_frac": float(tail_frac),
        "tail_bins_each_side": int(tail_bins),
        "wide_quantile": float(wide_quantile),
        "train_active_ratio": float(train_mask.mean()),
        "val_active_ratio": float(val_mask.mean()),
        "active_bins_mean_per_feature": float(active_table.sum(axis=1).mean()),
        "active_bins_min_per_feature": int(active_table.sum(axis=1).min()),
        "active_bins_max_per_feature": int(active_table.sum(axis=1).max()),
        "wide_bins_mean_per_feature": float(np.mean(wide_counts)),
        "wide_bins_min_per_feature": int(np.min(wide_counts)),
        "wide_bins_max_per_feature": int(np.max(wide_counts)),
        "thresholds_available": int(sum(t is not None for t in thresholds)),
    }
    return train_mask, val_mask, info


class FusionAblationDataset(Dataset):
    """
    Returns:
      X_bin: [F]
      values: [F, 3]
        values[...,0] = offset
        values[...,1] = continuous
        values[...,2] = continuous_mask
      y
      optional soft target [C] for training only
    """
    def __init__(
        self,
        X_bin: np.ndarray,
        X_offset: np.ndarray,
        X_cont: np.ndarray,
        X_mask: np.ndarray,
        y: np.ndarray,
        soft_targets: np.ndarray | None = None,
    ):
        if X_bin.ndim != 2:
            raise ValueError(f"X_bin must be [N,F], got {X_bin.shape}")
        if X_offset.shape != X_bin.shape or X_cont.shape != X_bin.shape or X_mask.shape != X_bin.shape:
            raise ValueError(f"shape mismatch: bin={X_bin.shape}, offset={X_offset.shape}, cont={X_cont.shape}, mask={X_mask.shape}")
        if y.ndim != 1 or y.shape[0] != X_bin.shape[0]:
            raise ValueError(f"y mismatch: {y.shape} vs {X_bin.shape}")
        if soft_targets is not None:
            soft_targets = np.asarray(soft_targets, dtype=np.float32)
            if soft_targets.ndim != 2 or soft_targets.shape[0] != X_bin.shape[0]:
                raise ValueError(f"soft_targets mismatch: {soft_targets.shape} vs rows={X_bin.shape[0]}")

        vals = np.stack([X_offset.astype(np.float32), X_cont.astype(np.float32), X_mask.astype(np.float32)], axis=-1)

        self.X = torch.as_tensor(X_bin, dtype=torch.long)
        self.V = torch.as_tensor(vals, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.soft_targets = None if soft_targets is None else torch.as_tensor(soft_targets, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        if self.soft_targets is None:
            return self.X[idx], self.V[idx], self.y[idx]
        return self.X[idx], self.V[idx], self.y[idx], self.soft_targets[idx]


class BaseValueEmbedding(nn.Module):
    def __init__(self, *, num_bins: int, n_features: int, value_dim: int, feature_dim: int, init_std: float = 0.02):
        super().__init__()
        self.num_bins = int(num_bins)
        self.n_features = int(n_features)
        self.value_dim = int(value_dim)
        self.feature_dim = int(feature_dim)
        self.cell_dim = int(value_dim + feature_dim)
        self.init_std = float(init_std)

        self.feature_embedding = nn.Embedding(self.n_features, self.feature_dim)
        nn.init.normal_(self.feature_embedding.weight, mean=0.0, std=self.init_std)
        self.register_buffer("default_feature_ids", torch.arange(self.n_features, dtype=torch.long), persistent=False)

    def add_feature_embedding(self, value_emb: torch.Tensor) -> torch.Tensor:
        B, F, _ = value_emb.shape
        fid = self.default_feature_ids.unsqueeze(0).expand(B, F)
        feature_emb = self.feature_embedding(fid)
        return torch.cat([value_emb, feature_emb], dim=-1)

    def gate_summary(self) -> Dict[str, float]:
        return {}


class D0OffsetInterpolationEmbedding(BaseValueEmbedding):
    def __init__(self, *, num_bins: int, n_features: int, value_dim: int, feature_dim: int):
        super().__init__(num_bins=num_bins, n_features=n_features, value_dim=value_dim, feature_dim=feature_dim)
        self.bin_embedding = nn.Embedding(self.num_bins + 1, self.value_dim)
        nn.init.normal_(self.bin_embedding.weight, mean=0.0, std=self.init_std)

    def local_interp(self, bin_ids: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        b0 = bin_ids.long().clamp(0, self.num_bins - 1)
        b1 = b0 + 1
        e0 = self.bin_embedding(b0)
        e1 = self.bin_embedding(b1)
        return (1.0 - offset) * e0 + offset * e1

    def forward(self, bin_ids: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        offset = values[..., 0:1].to(dtype=torch.float32, device=bin_ids.device).clamp(0.0, 1.0)
        value_emb = self.local_interp(bin_ids, offset)
        return self.add_feature_embedding(value_emb)


class D1InterpRawScalarConcatEmbedding(BaseValueEmbedding):
    def __init__(self, *, num_bins: int, n_features: int, value_dim: int, feature_dim: int):
        super().__init__(num_bins=num_bins, n_features=n_features, value_dim=value_dim, feature_dim=feature_dim)
        if self.value_dim < 2:
            raise ValueError("value_dim must be >= 2")
        self.local_dim = self.value_dim - 1
        self.bin_embedding = nn.Embedding(self.num_bins + 1, self.local_dim)
        nn.init.normal_(self.bin_embedding.weight, mean=0.0, std=self.init_std)

    def local_interp(self, bin_ids: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        b0 = bin_ids.long().clamp(0, self.num_bins - 1)
        b1 = b0 + 1
        e0 = self.bin_embedding(b0)
        e1 = self.bin_embedding(b1)
        return (1.0 - offset) * e0 + offset * e1

    def forward(self, bin_ids: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        vals = values.to(dtype=torch.float32, device=bin_ids.device)
        offset = vals[..., 0:1].clamp(0.0, 1.0)
        cont = vals[..., 1:2].clamp(0.0, 1.0)
        local = self.local_interp(bin_ids, offset)
        value_emb = torch.cat([local, cont], dim=-1)
        return self.add_feature_embedding(value_emb)


class D2D4D5InterpProjectGateConcatEmbedding(BaseValueEmbedding):
    """
    Proper concat-fusion:
      local_emb  = offset interpolation in value_dim
      cont_emb   = Project(continuous) in value_dim
      cont_emb   = sample_mask * sigmoid(gate_feature) * cont_emb
      value_emb  = Linear(concat([local_emb, cont_emb])) -> value_dim

    This keeps local bin geometry and global continuous as separate branches before fusion.
    """
    def __init__(self, *, num_bins: int, n_features: int, value_dim: int, feature_dim: int, gate_init: float):
        super().__init__(num_bins=num_bins, n_features=n_features, value_dim=value_dim, feature_dim=feature_dim)

        self.bin_embedding = nn.Embedding(self.num_bins + 1, self.value_dim)
        self.cont_proj = nn.Sequential(
            nn.Linear(1, self.value_dim),
            nn.GELU(),
            nn.Linear(self.value_dim, self.value_dim),
        )
        self.fuse = nn.Sequential(
            nn.LayerNorm(self.value_dim * 2),
            nn.Linear(self.value_dim * 2, self.value_dim),
        )
        self.cont_gate_logit = nn.Parameter(torch.full((self.n_features, 1), float(gate_init)))

        nn.init.normal_(self.bin_embedding.weight, mean=0.0, std=self.init_std)
        for m in self.cont_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.fuse[1].weight)
        nn.init.zeros_(self.fuse[1].bias)

    def local_interp(self, bin_ids: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        b0 = bin_ids.long().clamp(0, self.num_bins - 1)
        b1 = b0 + 1
        e0 = self.bin_embedding(b0)
        e1 = self.bin_embedding(b1)
        return (1.0 - offset) * e0 + offset * e1

    def forward(self, bin_ids: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        vals = values.to(dtype=torch.float32, device=bin_ids.device)
        offset = vals[..., 0:1].clamp(0.0, 1.0)
        cont = vals[..., 1:2].clamp(0.0, 1.0)
        mask = vals[..., 2:3].clamp(0.0, 1.0)

        B, F = bin_ids.shape
        local = self.local_interp(bin_ids, offset)
        cont_emb = self.cont_proj(cont)

        gate = torch.sigmoid(self.cont_gate_logit).to(device=bin_ids.device).unsqueeze(0).expand(B, F, 1)
        cont_emb = mask * gate * cont_emb

        value_emb = self.fuse(torch.cat([local, cont_emb], dim=-1))
        return self.add_feature_embedding(value_emb)

    def gate_summary(self) -> Dict[str, float]:
        with torch.no_grad():
            g = torch.sigmoid(self.cont_gate_logit.detach().cpu()).numpy()
        return {
            "cont_gate_min": float(g.min()),
            "cont_gate_max": float(g.max()),
            "cont_gate_mean": float(g.mean()),
            "cont_gate_std": float(g.std()),
        }


class D3InterpRawFiLMEmbedding(BaseValueEmbedding):
    """
    FiLM/multiply fusion:
      local_emb = offset interpolation
      gamma,beta = Project(continuous)
      value_emb = local_emb * (1 + mask * gate * tanh(gamma)) + mask * gate * beta
    """
    def __init__(self, *, num_bins: int, n_features: int, value_dim: int, feature_dim: int, gate_init: float):
        super().__init__(num_bins=num_bins, n_features=n_features, value_dim=value_dim, feature_dim=feature_dim)

        self.bin_embedding = nn.Embedding(self.num_bins + 1, self.value_dim)
        self.gamma_proj = nn.Sequential(nn.Linear(1, self.value_dim), nn.GELU(), nn.Linear(self.value_dim, self.value_dim))
        self.beta_proj = nn.Sequential(nn.Linear(1, self.value_dim), nn.GELU(), nn.Linear(self.value_dim, self.value_dim))
        self.cont_gate_logit = nn.Parameter(torch.full((self.n_features, 1), float(gate_init)))

        nn.init.normal_(self.bin_embedding.weight, mean=0.0, std=self.init_std)
        for net in [self.gamma_proj, self.beta_proj]:
            for m in net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

    def local_interp(self, bin_ids: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        b0 = bin_ids.long().clamp(0, self.num_bins - 1)
        b1 = b0 + 1
        e0 = self.bin_embedding(b0)
        e1 = self.bin_embedding(b1)
        return (1.0 - offset) * e0 + offset * e1

    def forward(self, bin_ids: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        vals = values.to(dtype=torch.float32, device=bin_ids.device)
        offset = vals[..., 0:1].clamp(0.0, 1.0)
        cont = vals[..., 1:2].clamp(0.0, 1.0)
        mask = vals[..., 2:3].clamp(0.0, 1.0)

        B, F = bin_ids.shape
        local = self.local_interp(bin_ids, offset)

        gamma = torch.tanh(self.gamma_proj(cont))
        beta = self.beta_proj(cont)

        gate = torch.sigmoid(self.cont_gate_logit).to(device=bin_ids.device).unsqueeze(0).expand(B, F, 1)
        g = mask * gate

        value_emb = local * (1.0 + g * gamma) + g * beta
        return self.add_feature_embedding(value_emb)

    def gate_summary(self) -> Dict[str, float]:
        with torch.no_grad():
            g = torch.sigmoid(self.cont_gate_logit.detach().cpu()).numpy()
        return {
            "cont_gate_min": float(g.min()),
            "cont_gate_max": float(g.max()),
            "cont_gate_mean": float(g.mean()),
            "cont_gate_std": float(g.std()),
        }


class D6ProjectedOffsetAddEmbedding(BaseValueEmbedding):
    def __init__(self, *, num_bins: int, n_features: int, value_dim: int, feature_dim: int):
        super().__init__(num_bins=num_bins, n_features=n_features, value_dim=value_dim, feature_dim=feature_dim)
        self.bin_embedding = nn.Embedding(self.num_bins, self.value_dim)
        self.offset_proj = nn.Sequential(nn.Linear(1, self.value_dim), nn.GELU(), nn.Linear(self.value_dim, self.value_dim))

        nn.init.normal_(self.bin_embedding.weight, mean=0.0, std=self.init_std)
        for m in self.offset_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, bin_ids: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        offset = values[..., 0:1].to(dtype=torch.float32, device=bin_ids.device).clamp(0.0, 1.0)
        b = bin_ids.long().clamp(0, self.num_bins - 1)
        value_emb = self.bin_embedding(b) + self.offset_proj(offset)
        return self.add_feature_embedding(value_emb)


class D7ProjectedOffsetConcatEmbedding(BaseValueEmbedding):
    def __init__(self, *, num_bins: int, n_features: int, value_dim: int, feature_dim: int):
        super().__init__(num_bins=num_bins, n_features=n_features, value_dim=value_dim, feature_dim=feature_dim)
        self.bin_embedding = nn.Embedding(self.num_bins, self.value_dim)
        self.offset_proj = nn.Sequential(nn.Linear(1, self.value_dim), nn.GELU(), nn.Linear(self.value_dim, self.value_dim))
        self.fuse = nn.Sequential(nn.LayerNorm(self.value_dim * 2), nn.Linear(self.value_dim * 2, self.value_dim))

        nn.init.normal_(self.bin_embedding.weight, mean=0.0, std=self.init_std)
        for m in self.offset_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.fuse[1].weight)
        nn.init.zeros_(self.fuse[1].bias)

    def forward(self, bin_ids: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        offset = values[..., 0:1].to(dtype=torch.float32, device=bin_ids.device).clamp(0.0, 1.0)
        b = bin_ids.long().clamp(0, self.num_bins - 1)
        bin_emb = self.bin_embedding(b)
        offset_emb = self.offset_proj(offset)
        value_emb = self.fuse(torch.cat([bin_emb, offset_emb], dim=-1))
        return self.add_feature_embedding(value_emb)


class FusionAblationTransformer(nn.Module):
    def __init__(
        self,
        *,
        run_id: str,
        num_bins: int,
        n_features: int,
        num_classes: int,
        value_dim: int,
        feature_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        classifier_hidden_dim: int,
        classifier_dropout: float,
        norm_first: bool,
        gate_init: float,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim must be divisible by num_heads: {hidden_dim}/{num_heads}")

        self.run_id = str(run_id)
        self.spec = RUN_SPECS[self.run_id]
        self.num_bins = int(num_bins)
        self.n_features = int(n_features)
        self.num_classes = int(num_classes)
        self.value_dim = int(value_dim)
        self.feature_dim = int(feature_dim)
        self.cell_dim = int(value_dim + feature_dim)
        self.hidden_dim = int(hidden_dim)

        if run_id == "D0":
            self.embedding = D0OffsetInterpolationEmbedding(num_bins=num_bins, n_features=n_features, value_dim=value_dim, feature_dim=feature_dim)
        elif run_id == "D1":
            self.embedding = D1InterpRawScalarConcatEmbedding(num_bins=num_bins, n_features=n_features, value_dim=value_dim, feature_dim=feature_dim)
        elif run_id in {"D2", "D4", "D5"}:
            self.embedding = D2D4D5InterpProjectGateConcatEmbedding(num_bins=num_bins, n_features=n_features, value_dim=value_dim, feature_dim=feature_dim, gate_init=gate_init)
        elif run_id == "D3":
            self.embedding = D3InterpRawFiLMEmbedding(num_bins=num_bins, n_features=n_features, value_dim=value_dim, feature_dim=feature_dim, gate_init=gate_init)
        elif run_id == "D6":
            self.embedding = D6ProjectedOffsetAddEmbedding(num_bins=num_bins, n_features=n_features, value_dim=value_dim, feature_dim=feature_dim)
        elif run_id == "D7":
            self.embedding = D7ProjectedOffsetConcatEmbedding(num_bins=num_bins, n_features=n_features, value_dim=value_dim, feature_dim=feature_dim)
        else:
            raise ValueError(f"unknown run_id: {run_id}")

        self.input_proj = nn.Sequential(
            nn.LayerNorm(self.cell_dim),
            nn.Linear(self.cell_dim, self.hidden_dim),
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=float(dropout),
            activation=activation,
            batch_first=True,
            norm_first=bool(norm_first),
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(self.hidden_dim),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, int(classifier_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(classifier_dropout)),
            nn.Linear(int(classifier_hidden_dim), int(num_classes)),
        )

        # F3e difficulty gate is initialized AFTER the original classifier so that
        # alpha/gate-off sanity keeps the original classifier RNG initialization.
        gate_hidden = max(16, self.hidden_dim // 2)
        self.difficulty_gate = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
        )

    def forward(self, tokens: torch.Tensor, z_values: torch.Tensor | None = None, *, return_info: bool = False, return_cls: bool = False):
        if z_values is None:
            raise ValueError("z_values must contain [offset, continuous, mask].")

        cell_emb = self.embedding(tokens, z_values)
        x = self.input_proj(cell_emb)

        B = x.shape[0]
        cls = self.cls_token.expand(B, 1, self.hidden_dim)
        x = torch.cat([cls, x], dim=1)

        encoded = self.encoder(x)
        cls_out = encoded[:, 0, :]
        logits = self.classifier(cls_out)

        if return_cls:
            return logits, cls_out
        if return_info:
            return logits, {
                "cell_emb_shape": tuple(cell_emb.shape),
                "encoded_shape": tuple(encoded.shape),
                "cls_out_shape": tuple(cls_out.shape),
                "cell_dim": self.cell_dim,
                "num_bins": self.num_bins,
                "run_id": self.run_id,
                "spec": self.spec,
            }
        return logits

    def embedding_extra_summary(self) -> Dict[str, float]:
        if hasattr(self.embedding, "gate_summary"):
            return self.embedding.gate_summary()
        return {}



class GatedAdaptiveHardNegativeLoss(nn.Module):
    """
    CE + detached_gate(x) * adaptive_hard_negative_loss
       + gate_loss_weight * BCE(gate(x), stopgrad(hardness_target)).

    Important:
      - No fixed hard-pair list.
      - No L3/family label is used.
      - No fixed margin threshold.
      - gate is learned from CLS, but detached when scaling hard-negative loss
        so it cannot learn the trivial solution gate=0 to escape that loss.

    hard_negative part:
      all wrong classes are considered.
      wrong classes with higher logits get larger softmax weights.
    """
    def __init__(
        self,
        base_criterion: nn.Module,
        *,
        temperature: float,
        gate_temperature: float,
        scope: str,
        gate_loss_weight: float,
        label_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.base_criterion = base_criterion
        self.temperature = float(max(temperature, 1e-6))
        self.gate_temperature = float(max(gate_temperature, 1e-6))
        self.scope = str(scope)
        self.gate_loss_weight = float(gate_loss_weight)
        self.label_names = list(label_names or [])
        if self.scope not in {"all", "malware"}:
            raise ValueError(f"bad gahn scope: {self.scope}")

        malware = {"Ransomware", "Spyware", "Trojan"}
        if self.label_names:
            self.malware_ids = [i for i, name in enumerate(self.label_names) if str(name) in malware]
        else:
            self.malware_ids = []

    def forward(
        self,
        *,
        logits: torch.Tensor,
        targets: torch.Tensor,
        gate_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        base_loss = self.base_criterion(logits, targets)

        B, C = logits.shape
        true_logits = logits.gather(1, targets.view(-1, 1)).squeeze(1)

        wrong_logits = logits.clone()
        wrong_logits.scatter_(1, targets.view(-1, 1), -1e9)
        valid_mask = torch.ones((B, C), dtype=torch.bool, device=logits.device)
        valid_mask.scatter_(1, targets.view(-1, 1), False)

        if self.scope == "malware":
            if not self.malware_ids:
                raise ValueError("gahn-scope=malware requires label_names containing Ransomware/Spyware/Trojan.")
            malware_ids = torch.as_tensor(self.malware_ids, dtype=torch.long, device=logits.device)
            class_is_malware = torch.zeros(C, dtype=torch.bool, device=logits.device)
            class_is_malware[malware_ids] = True
            true_is_active = class_is_malware[targets]
            valid_mask = valid_mask & class_is_malware.view(1, C)
            wrong_logits = wrong_logits.masked_fill(~valid_mask, -1e9)
        else:
            true_is_active = torch.ones(B, dtype=torch.bool, device=logits.device)
            wrong_logits = wrong_logits.masked_fill(~valid_mask, -1e9)

        # wrong class weights are adaptive and sample-specific.
        wrong_weights = torch.softmax(wrong_logits / self.temperature, dim=1)
        sep_penalty = F.softplus(logits - true_logits.view(-1, 1))
        sep_penalty = sep_penalty.masked_fill(~valid_mask, 0.0)
        hardneg_per_sample = (wrong_weights * sep_penalty).sum(dim=1)

        max_wrong_logits, hard_neg_ids = wrong_logits.max(dim=1)
        hardness_target = torch.sigmoid((max_wrong_logits - true_logits) / self.gate_temperature)
        hardness_target = hardness_target * true_is_active.to(dtype=hardness_target.dtype)

        gate_logits = gate_logits.view(-1)
        gate = torch.sigmoid(gate_logits) * true_is_active.to(dtype=logits.dtype)

        # Detach gate in this multiplication to avoid trivial "turn the gate off".
        hardneg_loss = (gate.detach() * hardneg_per_sample).mean()
        gate_loss = F.binary_cross_entropy_with_logits(gate_logits, hardness_target.detach(), reduction="none")
        gate_loss = (gate_loss * true_is_active.to(dtype=gate_loss.dtype)).mean()

        total_loss = base_loss + hardneg_loss + float(self.gate_loss_weight) * gate_loss
        active_rate = (true_is_active & (hardneg_per_sample > 0)).float().mean()

        return total_loss, {
            "base_loss": float(base_loss.detach().cpu()),
            "hardneg_loss": float(hardneg_loss.detach().cpu()),
            "gate_loss": float(gate_loss.detach().cpu()),
            "gate_mean": float(gate.detach().mean().cpu()),
            "hardness_target_mean": float(hardness_target.detach().mean().cpu()),
            "hardneg_per_sample_mean": float(hardneg_per_sample.detach().mean().cpu()),
            "active_rate": float(active_rate.detach().cpu()),
        }


def train_one_epoch_gated_adaptive_hardneg(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion: GatedAdaptiveHardNegativeLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0

    for batch in loader:
        if len(batch) == 4:
            tokens, values, y, _unused = batch
        else:
            tokens, values, y = batch

        tokens = tokens.to(device, non_blocking=True)
        values = values.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits, cls_out = model(tokens, values, return_cls=True)
        gate_logits = model.difficulty_gate(cls_out).view(-1)
        loss, _info = criterion(logits=logits, targets=y, gate_logits=gate_logits)
        loss.backward()
        if grad_clip_norm and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
        optimizer.step()

        bs = int(tokens.shape[0])
        total_loss += float(loss.detach().cpu()) * bs
        n += bs

    return float(total_loss / max(n, 1))



class WeightedSoftTargetCrossEntropyLoss(nn.Module):
    """Soft-target CE compatible with per-class weights.

    For one-hot targets, this reduces to CrossEntropyLoss(weight=class_weights)
    up to the same per-sample weighting semantics used here.
    """
    def __init__(self, class_weights: torch.Tensor | None = None):
        super().__init__()
        if class_weights is None:
            self.class_weights = None
        else:
            self.register_buffer("class_weights", class_weights.float())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.ndim != 2:
            raise ValueError(f"soft targets must be [B,C], got {tuple(targets.shape)}")
        logp = torch.log_softmax(logits, dim=1)
        t = targets.to(dtype=logp.dtype, device=logp.device)
        if self.class_weights is not None:
            t = t * self.class_weights.view(1, -1).to(dtype=logp.dtype, device=logp.device)
        return -(t * logp).sum(dim=1).mean()


def _find_label_col(df: pd.DataFrame, level: str) -> str | None:
    if level == "L2":
        candidates = ["label_L2", "Label_L2", "l2", "L2", "Category", "category"]
    else:
        candidates = ["label_L3", "Label_L3", "l3", "L3", "Family", "family", "MalwareFamily", "malware_family"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_soft_targets_from_matrix(
    *,
    args: argparse.Namespace,
    y_train: np.ndarray,
    label_names: List[str],
) -> Tuple[np.ndarray | None, Dict[str, object]]:
    """Build per-sample soft targets from a locked family matrix.

    This uses only train_raw labels to align each training row with its L2/L3
    family and does not look at validation labels or validation predictions.
    """
    if not str(getattr(args, "soft_target_matrix", "")).strip():
        return None, {
            "enabled": False,
            "reason": "no --soft-target-matrix provided",
        }

    matrix_path = resolve_path(str(args.soft_target_matrix))
    if not matrix_path.exists():
        raise FileNotFoundError(f"soft target matrix not found: {matrix_path}")

    train_path, _ = resolve_raw_paths(args)
    raw = pd.read_csv(train_path)
    if len(raw) != len(y_train):
        raise ValueError(f"train_raw rows={len(raw)} but y_train rows={len(y_train)}; cannot align soft targets safely")

    l2_col = _find_label_col(raw, "L2")
    l3_col = _find_label_col(raw, "L3")
    y_l2 = np.array([label_names[int(i)] for i in y_train], dtype=object)

    if l2_col is None:
        label_l2 = y_l2
    else:
        label_l2 = raw[l2_col].map(lambda x: "" if pd.isna(x) else str(x).strip()).to_numpy()
        if pd.Series(label_l2).isin(label_names).mean() < 0.80:
            label_l2 = y_l2

    if l3_col is None:
        label_l3 = label_l2.copy()
    else:
        label_l3 = raw[l3_col].map(lambda x: "" if pd.isna(x) else str(x).strip()).to_numpy()
        if (pd.Series(label_l3).map(lambda x: str(x).strip()) == "").mean() > 0.80:
            label_l3 = label_l2.copy()

    mat = pd.read_csv(matrix_path)
    required = ["true_L2", "true_L3"] + [f"target_{c}" for c in label_names]
    missing = [c for c in required if c not in mat.columns]
    if missing:
        raise ValueError(f"soft target matrix missing columns: {missing}")

    mat["true_L2"] = mat["true_L2"].map(lambda x: "" if pd.isna(x) else str(x).strip())
    mat["true_L3"] = mat["true_L3"].map(lambda x: "" if pd.isna(x) else str(x).strip())
    target_cols = [f"target_{c}" for c in label_names]
    for c in target_cols:
        mat[c] = pd.to_numeric(mat[c], errors="coerce").fillna(0.0)

    key_to_target: Dict[Tuple[str, str], np.ndarray] = {}
    for _, r in mat.iterrows():
        key_to_target[(str(r["true_L2"]).strip(), str(r["true_L3"]).strip())] = np.asarray([float(r[c]) for c in target_cols], dtype=np.float32)

    gamma = float(args.soft_target_gamma)
    targets = np.zeros((len(y_train), len(label_names)), dtype=np.float32)
    matched = 0
    fallback_onehot = 0
    missing_keys: Dict[str, int] = {}

    for i, yy in enumerate(y_train):
        y = int(yy)
        one = np.zeros(len(label_names), dtype=np.float32)
        one[y] = 1.0
        key = (str(label_l2[i]).strip(), str(label_l3[i]).strip())
        base = key_to_target.get(key)
        if base is None:
            base = one
            fallback_onehot += 1
            missing_keys[f"{key[0]}::{key[1]}"] = missing_keys.get(f"{key[0]}::{key[1]}", 0) + 1
        else:
            matched += 1

        tgt = one + gamma * (base.astype(np.float32) - one)
        tgt = np.clip(tgt, 0.0, 1.0)
        s = float(tgt.sum())
        targets[i] = one if s <= 0 else (tgt / s).astype(np.float32)

    info = {
        "enabled": True,
        "matrix_path": str(matrix_path),
        "train_raw_path": str(train_path),
        "gamma": gamma,
        "l2_col": l2_col,
        "l3_col": l3_col,
        "n": int(len(y_train)),
        "matched_L2_L3": int(matched),
        "fallback_onehot": int(fallback_onehot),
        "missing_keys_top": sorted(missing_keys.items(), key=lambda kv: kv[1], reverse=True)[:20],
        "target_min": float(targets.min()),
        "target_max": float(targets.max()),
        "target_sum_min": float(targets.sum(axis=1).min()),
        "target_sum_max": float(targets.sum(axis=1).max()),
    }
    return targets, info


def train_one_epoch_soft(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        if len(batch) != 4:
            raise ValueError("soft train loader must return (tokens, values, y, soft_targets)")
        tokens, values, _y, soft_targets = batch
        tokens = tokens.to(device, non_blocking=True)
        values = values.to(device, non_blocking=True)
        soft_targets = soft_targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens, values)
        loss = criterion(logits, soft_targets)
        loss.backward()
        if grad_clip_norm and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
        optimizer.step()

        bs = int(tokens.shape[0])
        total_loss += float(loss.detach().cpu()) * bs
        n += bs
    return float(total_loss / max(n, 1))


def _sam_grad_norm(model: nn.Module, *, adaptive: bool, eps: float, device: torch.device) -> torch.Tensor:
    norms = []
    for p in model.parameters():
        if p.grad is None:
            continue
        grad = p.grad.detach()
        if adaptive:
            grad = grad * (p.detach().abs() + float(eps))
        norms.append(torch.norm(grad, p=2))
    if not norms:
        return torch.tensor(0.0, device=device)
    return torch.norm(torch.stack(norms), p=2)


def _sam_perturb(model: nn.Module, *, rho: float, adaptive: bool, eps: float, device: torch.device):
    grad_norm = _sam_grad_norm(model, adaptive=adaptive, eps=eps, device=device)
    scale = float(rho) / (grad_norm + float(eps))
    perturbations = []
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is None:
                continue
            if adaptive:
                e_w = (p.detach().abs() + float(eps)) * p.grad * scale
            else:
                e_w = p.grad * scale
            p.add_(e_w)
            perturbations.append((p, e_w))
    return perturbations, float(grad_norm.detach().cpu())


def _sam_restore(perturbations):
    with torch.no_grad():
        for p, e_w in perturbations:
            p.sub_(e_w)


def train_one_epoch_sam(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float,
    rho: float,
    adaptive: bool,
    eps: float,
    soft_mode: bool,
) -> float:
    """
    SAM/ASAM training step.

    This wraps the official loss/model path:
    - soft_mode=False: batch = (tokens, values, y), criterion(logits, y)
    - soft_mode=True:  batch = (tokens, values, y, soft_targets), criterion(logits, soft_targets)

    rho=0 should not call this function; use official train_one_epoch path instead.
    """
    model.train()
    total_loss = 0.0
    n = 0
    grad_norms = []

    for batch in loader:
        if soft_mode:
            if len(batch) != 4:
                raise ValueError("soft SAM loader must return (tokens, values, y, soft_targets)")
            tokens, values, _y, target = batch
            target = target.to(device, non_blocking=True)
        else:
            if len(batch) == 4:
                tokens, values, y, _soft_targets = batch
            else:
                tokens, values, y = batch
            target = y.to(device, non_blocking=True)

        tokens = tokens.to(device, non_blocking=True)
        values = values.to(device, non_blocking=True)

        # First forward/backward at current weights.
        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens, values)
        loss = criterion(logits, target)
        loss.backward()

        # Perturb in gradient direction.
        perturbations, grad_norm = _sam_perturb(
            model,
            rho=float(rho),
            adaptive=bool(adaptive),
            eps=float(eps),
            device=device,
        )
        grad_norms.append(grad_norm)

        # Second forward/backward at perturbed weights.
        optimizer.zero_grad(set_to_none=True)
        logits_perturbed = model(tokens, values)
        loss_perturbed = criterion(logits_perturbed, target)
        loss_perturbed.backward()

        # Restore weights, then apply optimizer step using perturbed gradients.
        _sam_restore(perturbations)

        if grad_clip_norm and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
        optimizer.step()

        bs = int(tokens.shape[0])
        total_loss += float(loss_perturbed.detach().cpu()) * bs
        n += bs

    return float(total_loss / max(n, 1))



def strip_eval(report: Dict[str, object]) -> Dict[str, object]:
    skip = {"y_true", "y_pred", "confidence", "probs"}
    return {k: v for k, v in report.items() if k not in skip}


def malware_avg_f1(report: Dict[str, object]) -> float:
    vals = []
    for label, m in report["per_class"].items():
        if str(label).strip().lower() != "benign":
            vals.append(float(m["f1"]))
    return float(np.mean(vals)) if vals else 0.0




def make_boundary_bin_diagnostics(
    X_bin: np.ndarray,
    X_offset: np.ndarray,
    *,
    num_bins: int,
    feature_names: List[str],
    split_name: str,
) -> Dict[str, object]:
    """Measure how much data falls into the last bin affected by old boundary collapse."""
    Xb = np.asarray(X_bin, dtype=np.int64)
    Off = np.asarray(X_offset, dtype=np.float32)
    if Xb.shape != Off.shape:
        raise ValueError(f"boundary diag shape mismatch: X_bin={Xb.shape}, X_offset={Off.shape}")

    last_bin = int(num_bins) - 1
    mask = Xb == last_bin
    n_rows, n_features = Xb.shape
    total_cells = int(mask.size)
    last_cells = int(mask.sum())

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
        per_feature.append({
            "feature": str(name),
            "last_bin_count": cnt,
            "last_bin_ratio": float(cnt / max(n_rows, 1)),
            "last_bin_offset_min": offset_min,
            "last_bin_offset_max": offset_max,
            "last_bin_offset_mean": offset_mean,
            "last_bin_offset_std": offset_std,
        })

    top = sorted(per_feature, key=lambda r: r["last_bin_ratio"], reverse=True)[:15]
    return {
        "split": split_name,
        "num_bins": int(num_bins),
        "last_bin_id": int(last_bin),
        "n_rows": int(n_rows),
        "n_features": int(n_features),
        "total_cells": total_cells,
        "last_bin_cells": last_cells,
        "last_bin_cell_ratio": float(last_cells / max(total_cells, 1)),
        "features_with_last_bin": int(sum(1 for r in per_feature if r["last_bin_count"] > 0)),
        "max_feature_last_bin_ratio": float(max((r["last_bin_ratio"] for r in per_feature), default=0.0)),
        "mean_feature_last_bin_ratio": float(np.mean([r["last_bin_ratio"] for r in per_feature])) if per_feature else 0.0,
        "top_features_by_last_bin_ratio": top,
    }


def make_embedding_runtime_diagnostics(
    model: "FusionAblationTransformer",
    *,
    X_bin: np.ndarray,
    X_offset: np.ndarray,
    X_cont: np.ndarray,
    X_mask: np.ndarray,
    device: torch.device,
    max_rows: int = 2048,
) -> Dict[str, object]:
    """Lightweight forward-scale checks before training.

    This verifies that endpoint interpolation is not collapsed at the final bin,
    and records activation scales for local embedding vs continuous branch.
    """
    emb = model.embedding
    n = int(min(max_rows, X_bin.shape[0]))
    X = torch.as_tensor(X_bin[:n], dtype=torch.long, device=device)
    V = torch.as_tensor(
        np.stack([
            X_offset[:n].astype(np.float32),
            X_cont[:n].astype(np.float32),
            X_mask[:n].astype(np.float32),
        ], axis=-1),
        dtype=torch.float32,
        device=device,
    )
    out: Dict[str, object] = {
        "n_rows_checked": n,
        "has_local_interp": bool(hasattr(emb, "local_interp")),
        "embedding_class": emb.__class__.__name__,
    }

    with torch.no_grad():
        vals = V.to(dtype=torch.float32, device=device)
        offset = vals[..., 0:1].clamp(0.0, 1.0)
        cont = vals[..., 1:2].clamp(0.0, 1.0)
        mask = vals[..., 2:3].clamp(0.0, 1.0)
        out["offset_mean"] = float(offset.mean().detach().cpu())
        out["offset_std"] = float(offset.std(unbiased=False).detach().cpu())
        out["continuous_mean"] = float(cont.mean().detach().cpu())
        out["continuous_std"] = float(cont.std(unbiased=False).detach().cpu())
        out["continuous_mask_mean"] = float(mask.mean().detach().cpu())

        if hasattr(emb, "bin_embedding"):
            out["bin_embedding_num_embeddings"] = int(emb.bin_embedding.num_embeddings)
            out["bin_embedding_dim"] = int(emb.bin_embedding.embedding_dim)
            out["bin_embedding_weight_std"] = float(emb.bin_embedding.weight.detach().std(unbiased=False).cpu())

        if hasattr(emb, "local_interp"):
            local = emb.local_interp(X, offset)
            out["local_abs_mean"] = float(local.abs().mean().detach().cpu())
            out["local_std"] = float(local.std(unbiased=False).detach().cpu())
            out["local_l2_mean_per_cell"] = float(torch.linalg.vector_norm(local, dim=-1).mean().detach().cpu())

            # Direct boundary sensitivity check: old implementation gives zero here.
            b_last = torch.full((1, 1), int(emb.num_bins) - 1, dtype=torch.long, device=device)
            off0 = torch.zeros((1, 1, 1), dtype=torch.float32, device=device)
            off1 = torch.ones((1, 1, 1), dtype=torch.float32, device=device)
            y0 = emb.local_interp(b_last, off0)
            y1 = emb.local_interp(b_last, off1)
            diff = torch.linalg.vector_norm(y1 - y0, dim=-1)
            out["last_bin_offset_sensitivity_l2"] = float(diff.item())
            out["last_bin_boundary_collapse_fixed"] = bool(float(diff.item()) > 1e-12)

            # Scale-ratio check most relevant for D1 raw_scalar_concat.
            local_abs = float(local.abs().mean().detach().cpu())
            cont_abs = float(cont.abs().mean().detach().cpu())
            out["continuous_abs_mean_over_local_abs_mean"] = float(cont_abs / max(local_abs, 1e-12))

        if hasattr(emb, "gamma_proj") and hasattr(emb, "beta_proj"):
            gamma = torch.tanh(emb.gamma_proj(cont))
            beta = emb.beta_proj(cont)
            if hasattr(emb, "cont_gate_logit"):
                Bsz, F = X.shape
                gate = torch.sigmoid(emb.cont_gate_logit).to(device=device).unsqueeze(0).expand(Bsz, F, 1)
                g = mask * gate
            else:
                g = mask
            out["film_gamma_abs_mean"] = float(gamma.abs().mean().detach().cpu())
            out["film_beta_abs_mean"] = float(beta.abs().mean().detach().cpu())
            out["film_gate_mean"] = float(g.mean().detach().cpu())
            if hasattr(emb, "local_interp"):
                delta = (local * (g * gamma)) + (g * beta)
                out["film_delta_l2_mean_per_cell"] = float(torch.linalg.vector_norm(delta, dim=-1).mean().detach().cpu())
                out["film_delta_over_local_l2_mean"] = float(
                    out["film_delta_l2_mean_per_cell"] / max(out.get("local_l2_mean_per_cell", 0.0), 1e-12)
                )

    return out

def make_diagnosis_summary(
    *,
    args: argparse.Namespace,
    spec: Dict[str, str],
    best_epoch: int,
    best_train: Dict[str, object],
    best_val: Dict[str, object],
    metadata: Dict[str, object],
    continuous_info: Dict[str, object],
    selective_info: Dict[str, object],
    boundary_data_diagnostics: Dict[str, object],
    embedding_runtime_diagnostics: Dict[str, object],
    model: FusionAblationTransformer,
) -> Dict[str, object]:
    train_macro = float(best_train["macro_f1"])
    val_macro = float(best_val["macro_f1"])

    worst_classes = sorted(
        [
            {
                "label": str(label).strip(),
                "f1": float(m["f1"]),
                "precision": float(m["precision"]),
                "recall": float(m["recall"]),
                "support": int(m["support"]),
            }
            for label, m in best_val["per_class"].items()
        ],
        key=lambda x: (x["f1"], -x["support"]),
    )[:10]

    return {
        "phase": "Final pipeline C2 D3",
        "run_id": args.run_id,
        "best_epoch": int(best_epoch),
        "representation": f"{args.run_id}_{spec['description']}",
        "local": spec["local"],
        "continuous_source": spec["continuous_source"],
        "fusion": spec["fusion"],
        "train": {
            "loss": float(best_train["loss"]),
            "accuracy": float(best_train["accuracy"]),
            "macro_f1": train_macro,
            "weighted_f1": float(best_train["weighted_f1"]),
            "malware_only_avg_f1": malware_avg_f1(best_train),
        },
        "val": {
            "loss": float(best_val["loss"]),
            "accuracy": float(best_val["accuracy"]),
            "macro_f1": val_macro,
            "weighted_f1": float(best_val["weighted_f1"]),
            "malware_only_avg_f1": malware_avg_f1(best_val),
        },
        "generalization_gap_macro_f1": float(train_macro - val_macro),
        "worst_val_classes_by_f1": worst_classes,
        "model_config": {
            "value_dim": int(args.value_dim),
            "feature_dim": int(args.feature_dim),
            "cell_dim": int(args.value_dim + args.feature_dim),
            "num_bins": int(args.num_bins),
            "effective_token_budget": int(args.num_bins),
            "hidden_dim": int(args.hidden_dim),
            "num_layers": int(args.num_layers),
            "num_heads": int(args.num_heads),
            "dropout": float(args.dropout),
            "classifier_hidden_dim": int(args.classifier_hidden_dim),
            "classifier_dropout": float(args.classifier_dropout),
            "norm_first": bool(args.norm_first),
            "use_class_weights": bool(args.use_class_weights),
            "scheduler": str(args.scheduler),
            "gate_init": float(args.gate_init),
        },
        "continuous_info": continuous_info,
        "selective_info": selective_info,
        "boundary_data_diagnostics": boundary_data_diagnostics,
        "embedding_runtime_diagnostics": embedding_runtime_diagnostics,
        "embedding_extra_summary": model.embedding_extra_summary(),
        "strategy_counts": metadata.get("strategy_counts", {}),
        "baseline_to_compare": {
            "C0_mixed_bin_scalar_offset_no_continuous": {
                "train_macro_f1": 0.9321426589148848,
                "val_macro_f1": 0.8133554660345561,
                "gap": 0.11878719288032868,
            },
            "C2_scalar_offset_raw_scalar_concat": {
                "train_macro_f1": 0.9265573540092596,
                "val_macro_f1": 0.809009713024029,
                "gap": 0.11754764098523063,
            },
            "C3_scalar_offset_z_project_gate_add": {
                "train_macro_f1": 0.890405910135817,
                "val_macro_f1": 0.8087819336025909,
                "gap": 0.08162397653322617,
            },
            "C4_scalar_offset_raw_project_gate_add": {
                "train_macro_f1": 0.9110189147125328,
                "val_macro_f1": 0.807190454742599,
                "gap": 0.10382845996993384,
            },
        },
    }


def main() -> None:
    args = parse_args()
    args.run_id = "D3"
    _train_mod.set_seed(int(args.seed))

    spec = RUN_SPECS[args.run_id]
    K_artifact = int(args.K)
    B = int(args.num_bins)

    dataset_path = resolve_path(args.dataset_npz) if args.dataset_npz else resolve_path(default_dataset_path(K_artifact, B))
    metadata_path = resolve_path(args.metadata_json) if args.metadata_json else resolve_path(default_metadata_path(K_artifact, B))

    data, meta = load_dataset(dataset_path, metadata_path)

    X_train = data["X_train_bin"].astype(np.int64)
    O_train = data["X_train_offset"].astype(np.float32)
    y_train = data["y_train"].astype(np.int64)

    X_val = data["X_val_bin"].astype(np.int64)
    O_val = data["X_val_offset"].astype(np.float32)
    y_val = data["y_val"].astype(np.int64)

    X_train_cont, X_val_cont, continuous_info = load_continuous_for_run(
        spec=spec,
        meta=meta,
        args=args,
        train_shape=X_train.shape,
        val_shape=X_val.shape,
    )

    if X_train_cont.shape != X_train.shape:
        raise ValueError(f"train continuous shape mismatch: {X_train_cont.shape} vs {X_train.shape}")
    if X_val_cont.shape != X_val.shape:
        raise ValueError(f"val continuous shape mismatch: {X_val_cont.shape} vs {X_val.shape}")

    selective_info: Dict[str, object] = {"type": "none"}
    if args.run_id == "D5":
        M_train, M_val, selective_info = build_selective_mask(
            X_train_bin=X_train,
            X_val_bin=X_val,
            X_train_cont=X_train_cont,
            num_bins=B,
            tail_frac=float(args.tail_frac),
            wide_quantile=float(args.wide_quantile),
        )
    else:
        M_train = np.ones_like(X_train, dtype=np.float32)
        M_val = np.ones_like(X_val, dtype=np.float32)

    feature_names_for_diag = [str(x) for x in meta["feature_names"]]
    boundary_data_diagnostics = {
        "train": make_boundary_bin_diagnostics(
            X_train, O_train, num_bins=B, feature_names=feature_names_for_diag, split_name="train"
        ),
        "val": make_boundary_bin_diagnostics(
            X_val, O_val, num_bins=B, feature_names=feature_names_for_diag, split_name="val"
        ),
    }

    label_mapping = meta["label_mapping"]
    label_names = [label for label, _ in sorted(label_mapping.items(), key=lambda kv: kv[1])]
    num_classes = int(len(label_names))
    n_features = int(meta["n_features"])

    soft_targets_train, soft_target_info = load_soft_targets_from_matrix(
        args=args,
        y_train=y_train,
        label_names=label_names,
    )

    # Clean default: python 02_src/07_train.py writes directly to 03_outputs/06_model.
    # Compatibility mode: if --run-name is provided, keep old-style <out-root>/KeffB/<run-name>.
    if args.run_name:
        out_root = resolve_path(args.out_root) if args.out_root else resolve_path(cfg("OUTPUT_ROOT", Path("03_outputs")) / "train_runs_fusion_ablation_D0_D7")
        out_dir = out_root / f"Keff{B}" / str(args.run_name)
    else:
        out_dir = default_model_dir() if not args.out_root else resolve_path(args.out_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _train_mod.pick_device(str(args.device))

    config_obj = {
        "stage": "train_official_d3_final_pipeline",
        "phase": "C2_D3_final_pipeline_reproduction",
        "run_id": args.run_id,
        "run_spec": spec,
        "K_artifact": K_artifact,
        "effective_token_budget": B,
        "num_bins": B,
        "dataset_npz": str(dataset_path),
        "metadata_json": str(metadata_path),
        "out_dir": str(out_dir),
        "continuous_info": continuous_info,
        "selective_info": selective_info,
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
        "soft_targets": soft_target_info,
        "n_features": n_features,
        "num_classes": num_classes,
        "label_names": label_names,
        "representation": {
            "local": spec["local"],
            "continuous_source": spec["continuous_source"],
            "fusion": spec["fusion"],
            "description": spec["description"],
            "strategy_counts": meta.get("strategy_counts", {}),
        },
        "model": {
            "value_dim": int(args.value_dim),
            "feature_dim": int(args.feature_dim),
            "cell_dim": int(args.value_dim + args.feature_dim),
            "hidden_dim": int(args.hidden_dim),
            "num_layers": int(args.num_layers),
            "num_heads": int(args.num_heads),
            "dropout": float(args.dropout),
            "classifier_hidden_dim": int(args.classifier_hidden_dim),
            "classifier_dropout": float(args.classifier_dropout),
            "norm_first": bool(args.norm_first),
            "gate_init": float(args.gate_init),
        },
        "boundary_data_diagnostics": boundary_data_diagnostics,
        "data_shapes": {
            "X_train_bin": list(X_train.shape),
            "X_train_offset": list(O_train.shape),
            "X_train_continuous": list(X_train_cont.shape),
            "X_train_mask": list(M_train.shape),
            "y_train": list(y_train.shape),
            "X_val_bin": list(X_val.shape),
            "X_val_offset": list(O_val.shape),
            "X_val_continuous": list(X_val_cont.shape),
            "X_val_mask": list(M_val.shape),
            "y_val": list(y_val.shape),
        },
        "torch_version": torch.__version__,
    }
    _train_mod.save_json(out_dir / "config.json", config_obj)

    gated_hardneg_info = {
        "enabled": bool(args.gahn_enabled),
        "temperature": float(args.gahn_temperature),
        "gate_temperature": float(args.gahn_gate_temperature),
        "scope": str(args.gahn_scope),
        "gate_loss_weight": float(args.gahn_gate_loss_weight),
        "uses_fixed_hard_pairs": False,
        "uses_l3_or_family_labels_for_training": False,
        "uses_validation_to_select_pairs": False,
        "uses_fixed_margin": False,
        "hardneg_scale": "learned difficulty gate from CLS; gate detached when scaling hardneg loss",
        "note": "CE + gate_detached(x)*adaptive_hardneg + gate_loss_weight*BCE(gate(x), stopgrad(sigmoid(max_wrong_logit-true_logit))).",
    }
    config_obj["gated_adaptive_hard_negative"] = gated_hardneg_info
    _train_mod.save_json(out_dir / "gated_hardneg_info.json", gated_hardneg_info)
    _train_mod.save_json(out_dir / "config.json", config_obj)

    train_ds = FusionAblationDataset(X_train, O_train, X_train_cont, M_train, y_train, soft_targets=soft_targets_train)
    train_eval_ds = FusionAblationDataset(X_train, O_train, X_train_cont, M_train, y_train)
    val_ds = FusionAblationDataset(X_val, O_val, X_val_cont, M_val, y_val)

    generator = torch.Generator()
    generator.manual_seed(int(args.seed))

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        generator=generator,
    )
    train_eval_loader = DataLoader(
        train_eval_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    model = FusionAblationTransformer(
        run_id=args.run_id,
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
        activation=str(cfg("MODEL_ACTIVATION", "gelu")),
    ).to(device)

    embedding_runtime_diagnostics = make_embedding_runtime_diagnostics(
        model,
        X_bin=X_train,
        X_offset=O_train,
        X_cont=X_train_cont,
        X_mask=M_train,
        device=device,
    )
    config_obj["embedding_runtime_diagnostics"] = embedding_runtime_diagnostics
    _train_mod.save_json(out_dir / "config.json", config_obj)

    if args.use_class_weights:
        weights = _train_mod.compute_class_weights(y_train, num_classes).to(device)
        eval_criterion = nn.CrossEntropyLoss(weight=weights)
        class_weights_log = weights.detach().cpu().numpy().tolist()
    else:
        weights = None
        eval_criterion = nn.CrossEntropyLoss()
        class_weights_log = None

    if soft_targets_train is not None:
        criterion = WeightedSoftTargetCrossEntropyLoss(weights)
    elif bool(args.gahn_enabled):
        criterion = GatedAdaptiveHardNegativeLoss(
            eval_criterion,
            temperature=float(args.gahn_temperature),
            gate_temperature=float(args.gahn_gate_temperature),
            scope=str(args.gahn_scope),
            gate_loss_weight=float(args.gahn_gate_loss_weight),
            label_names=label_names,
        )
    else:
        criterion = eval_criterion

    _train_mod.save_json(out_dir / str(args.soft_target_report_name), soft_target_info)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    sam_info = {
        "sam_rho": float(args.sam_rho),
        "sam_adaptive": bool(args.sam_adaptive),
        "sam_eps": float(args.sam_eps),
        "enabled": bool(float(args.sam_rho) > 0.0),
    }
    config_obj["sam"] = sam_info
    _train_mod.save_json(out_dir / "config.json", config_obj)

    _train_mod.save_json(out_dir / "class_weights.json", {
        "use_class_weights": bool(args.use_class_weights),
        "class_weights": class_weights_log,
        "label_names": label_names,
        "train_counts": np.bincount(y_train, minlength=num_classes).astype(int).tolist(),
        "val_counts": np.bincount(y_val, minlength=num_classes).astype(int).tolist(),
    })

    print("===== fusion ablation training start =====")
    print(f"run_id: {args.run_id}")
    print(f"spec: {spec}")
    print(f"run_dir: {out_dir}")
    print(f"device: {device}")
    print(f"dataset: {dataset_path}")
    print(f"continuous_info: {continuous_info}")
    print(f"selective_info: {selective_info}")
    print(f"train bin/offset/continuous/mask shape: {X_train.shape}/{O_train.shape}/{X_train_cont.shape}/{M_train.shape}")
    print(f"val bin/offset/continuous/mask shape:   {X_val.shape}/{O_val.shape}/{X_val_cont.shape}/{M_val.shape}")
    print(f"classes: {num_classes} {label_names}")
    print(f"strategy_counts: {meta.get('strategy_counts', {})}")
    print(f"boundary last-bin train cell ratio: {boundary_data_diagnostics['train']['last_bin_cell_ratio']:.6f}")
    print(f"boundary last-bin val cell ratio:   {boundary_data_diagnostics['val']['last_bin_cell_ratio']:.6f}")
    print(f"last-bin offset sensitivity L2:     {embedding_runtime_diagnostics.get('last_bin_offset_sensitivity_l2')}")
    print(f"boundary collapse fixed:            {embedding_runtime_diagnostics.get('last_bin_boundary_collapse_fixed')}")
    print(f"SAM enabled:                       {float(args.sam_rho) > 0.0} rho={float(args.sam_rho)} adaptive={bool(args.sam_adaptive)}")
    print(f"gated adaptive hardneg:            {bool(args.gahn_enabled)} temp={float(args.gahn_temperature)} gate_temp={float(args.gahn_gate_temperature)} scope={str(args.gahn_scope)} gate_loss_weight={float(args.gahn_gate_loss_weight)}")

    if bool(args.diagnostic_only):
        diagnostic_only_obj = {
            "run_id": args.run_id,
            "spec": spec,
            "dataset_npz": str(dataset_path),
            "metadata_json": str(metadata_path),
            "boundary_data_diagnostics": boundary_data_diagnostics,
            "embedding_runtime_diagnostics": embedding_runtime_diagnostics,
            "continuous_info": continuous_info,
            "selective_info": selective_info,
            "data_shapes": config_obj["data_shapes"],
            "model_config": config_obj["model"],
            "note": "Diagnostic-only run exited before training.",
        }
        _train_mod.save_json(out_dir / "diagnostics_only.json", diagnostic_only_obj)
        print("diagnostic-only mode: exiting before training")
        print(f"diagnostics_only:       {out_dir / 'diagnostics_only.json'}")
        return

    history: List[Dict[str, object]] = []
    best_metric = -math.inf
    best_epoch = -1
    best_train_eval = None
    best_val_eval = None
    bad_epochs = 0

    for epoch in range(1, int(args.epochs) + 1):
        t0 = time.time()

        lr_epoch = _train_mod.compute_epoch_lr(
            base_lr=float(args.lr),
            epoch=epoch,
            total_epochs=int(args.epochs),
            scheduler_name=str(args.scheduler),
            warmup_epochs=int(args.warmup_epochs),
            min_lr_ratio=float(args.min_lr_ratio),
        )
        _train_mod.set_optimizer_lr(optimizer, lr_epoch)

        if bool(args.gahn_enabled):
            if soft_targets_train is not None:
                raise ValueError("Do not combine soft targets with gated adaptive hard-negative loss in this experiment.")
            if float(args.sam_rho) > 0.0:
                raise ValueError("Do not combine SAM with gated adaptive hard-negative loss in this experiment.")
            train_step_loss = train_one_epoch_gated_adaptive_hardneg(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                grad_clip_norm=float(args.grad_clip_norm),
            )
        elif float(args.sam_rho) > 0.0:
            train_step_loss = train_one_epoch_sam(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                grad_clip_norm=float(args.grad_clip_norm),
                rho=float(args.sam_rho),
                adaptive=bool(args.sam_adaptive),
                eps=float(args.sam_eps),
                soft_mode=soft_targets_train is not None,
            )
        elif soft_targets_train is not None:
            train_step_loss = train_one_epoch_soft(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                grad_clip_norm=float(args.grad_clip_norm),
            )
        else:
            train_step_loss = _train_mod.train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                grad_clip_norm=float(args.grad_clip_norm),
            )

        train_eval = _train_mod.evaluate(model, train_eval_loader, eval_criterion, device, num_classes, label_names)
        val_eval = _train_mod.evaluate(model, val_loader, eval_criterion, device, num_classes, label_names)

        row = {
            "epoch": epoch,
            "lr": float(lr_epoch),
            "epoch_seconds": round(time.time() - t0, 3),
            "train_step_loss": float(train_step_loss),
            "train_loss": float(train_eval["loss"]),
            "train_acc": float(train_eval["accuracy"]),
            "train_macro_f1": float(train_eval["macro_f1"]),
            "train_weighted_f1": float(train_eval["weighted_f1"]),
            "train_malware_avg_f1": malware_avg_f1(train_eval),
            "val_loss": float(val_eval["loss"]),
            "val_acc": float(val_eval["accuracy"]),
            "val_macro_f1": float(val_eval["macro_f1"]),
            "val_weighted_f1": float(val_eval["weighted_f1"]),
            "val_malware_avg_f1": malware_avg_f1(val_eval),
            "macro_f1_gap_train_minus_val": float(train_eval["macro_f1"] - val_eval["macro_f1"]),
        }
        row.update(model.embedding_extra_summary())
        history.append(row)
        _train_mod.write_history_csv(out_dir / "history.csv", history)

        metric = float(val_eval["macro_f1"])
        improved = metric > best_metric + float(args.min_delta)

        if improved:
            best_metric = metric
            best_epoch = epoch
            best_train_eval = train_eval
            best_val_eval = val_eval
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_val_macro_f1": best_metric,
                    "config": config_obj,
                    "metadata": meta,
                    "continuous_info": continuous_info,
                    "selective_info": selective_info,
                },
                out_dir / "best_model.pt",
            )
        else:
            bad_epochs += 1

        print(
            f"[{args.run_id} epoch {epoch:03d}] "
            f"lr={lr_epoch:.6g} "
            f"train_macro={row['train_macro_f1']:.4f} "
            f"train_malware={row['train_malware_avg_f1']:.4f} "
            f"val_macro={row['val_macro_f1']:.4f} "
            f"val_malware={row['val_malware_avg_f1']:.4f} "
            f"gap={row['macro_f1_gap_train_minus_val']:.4f} "
            f"best={best_metric:.4f}@{best_epoch}"
        )

        if bad_epochs >= int(args.patience):
            print(f"early stop: bad_epochs={bad_epochs}, patience={args.patience}")
            break

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": int(history[-1]["epoch"]) if history else -1,
            "config": config_obj,
            "metadata": meta,
            "continuous_info": continuous_info,
            "selective_info": selective_info,
        },
        out_dir / "last_model.pt",
    )

    if best_train_eval is None or best_val_eval is None:
        raise RuntimeError("No best eval was recorded.")

    train_report = strip_eval(best_train_eval)
    val_report = strip_eval(best_val_eval)

    _train_mod.save_json(out_dir / "train_classification_report_best.json", train_report)
    _train_mod.save_json(out_dir / "val_classification_report_best.json", val_report)

    _train_mod.write_confusion_outputs(out_dir, "train", best_train_eval, label_names)
    _train_mod.write_confusion_outputs(out_dir, "val", best_val_eval, label_names)

    _train_mod.write_predictions_csv(
        out_dir / "val_predictions_best.csv",
        best_val_eval["y_true"],
        best_val_eval["y_pred"],
        best_val_eval["confidence"],
        label_names,
    )

    diagnosis = make_diagnosis_summary(
        args=args,
        spec=spec,
        best_epoch=best_epoch,
        best_train=best_train_eval,
        best_val=best_val_eval,
        metadata=meta,
        continuous_info=continuous_info,
        selective_info=selective_info,
        boundary_data_diagnostics=boundary_data_diagnostics,
        embedding_runtime_diagnostics=embedding_runtime_diagnostics,
        model=model,
    )
    _train_mod.save_json(out_dir / "diagnosis_summary.json", diagnosis)

    print("===== fusion ablation training done =====")
    print(f"run_id:                {args.run_id}")
    print(f"representation:        {diagnosis['representation']}")
    print(f"best_epoch:            {best_epoch}")
    print(f"train_macro_f1:        {diagnosis['train']['macro_f1']:.6f}")
    print(f"train_malware_avg_f1: {diagnosis['train']['malware_only_avg_f1']:.6f}")
    print(f"val_macro_f1:          {diagnosis['val']['macro_f1']:.6f}")
    print(f"val_malware_avg_f1:   {diagnosis['val']['malware_only_avg_f1']:.6f}")
    print(f"gap:                   {diagnosis['generalization_gap_macro_f1']:.6f}")
    print(f"diagnosis:             {out_dir / 'diagnosis_summary.json'}")


if __name__ == "__main__":
    main()
