#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B0_export_06_model_probs.py

Purpose
-------
Export full probability vector / top-2 candidates for the newly trained official
C2+D3 model under:

    03_outputs/06_model/

This file is intentionally standalone and lives under 05_test. It does NOT
modify or depend on training symbols from 02_src/07_train.py.

Inputs by default
-----------------
    03_outputs/06_model/best_model.pt
    03_outputs/06_model/config.json
    03_outputs/05_dataset/dataset.npz
    03_outputs/05_dataset/metadata.json

Output by default
-----------------
    05_test/outputs/B0_wrong_top2_audit/val_predictions_with_probs.csv
    05_test/outputs/B0_wrong_top2_audit/B0_export_06_model_probs_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import zipfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------
# Model definition: exact official D3 needed for inference
# ---------------------------------------------------------------------
class BaseValueEmbedding(nn.Module):
    def __init__(self, *, num_bins: int, n_features: int, value_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.num_bins = int(num_bins)
        self.n_features = int(n_features)
        self.value_dim = int(value_dim)
        self.feature_dim = int(feature_dim)
        self.cell_dim = int(value_dim + feature_dim)
        self.init_std = 0.02

        self.feature_embedding = nn.Embedding(self.n_features, self.feature_dim)
        nn.init.normal_(self.feature_embedding.weight, mean=0.0, std=self.init_std)

        self.register_buffer(
            "default_feature_ids",
            torch.arange(self.n_features, dtype=torch.long),
            persistent=False,
        )

    def add_feature_embedding(self, value_emb: torch.Tensor) -> torch.Tensor:
        if value_emb.ndim != 3:
            raise ValueError(f"value_emb must be [B,F,D], got {tuple(value_emb.shape)}")
        B, F, _ = value_emb.shape
        if F != self.n_features:
            raise ValueError(f"Expected F={self.n_features}, got {F}")
        fid = self.default_feature_ids.to(value_emb.device).unsqueeze(0).expand(B, F)
        feat_emb = self.feature_embedding(fid)
        return torch.cat([value_emb, feat_emb], dim=-1)


class D3InterpRawFiLMEmbedding(BaseValueEmbedding):
    """
    Official D3 embedding:
      local = (1-offset) * Emb(bin) + offset * Emb(bin+1)
      gamma,beta = Project(raw_scaled_continuous)
      value_emb = local * (1 + mask * gate * tanh(gamma)) + mask * gate * beta
      cell_emb = concat(value_emb, feature_embedding)
    """

    def __init__(
        self,
        *,
        num_bins: int,
        n_features: int,
        value_dim: int,
        feature_dim: int,
        gate_init: float,
    ) -> None:
        super().__init__(
            num_bins=num_bins,
            n_features=n_features,
            value_dim=value_dim,
            feature_dim=feature_dim,
        )

        self.bin_embedding = nn.Embedding(self.num_bins + 1, self.value_dim)
        self.gamma_proj = nn.Sequential(
            nn.Linear(1, self.value_dim),
            nn.GELU(),
            nn.Linear(self.value_dim, self.value_dim),
        )
        self.beta_proj = nn.Sequential(
            nn.Linear(1, self.value_dim),
            nn.GELU(),
            nn.Linear(self.value_dim, self.value_dim),
        )
        self.cont_gate_logit = nn.Parameter(
            torch.full((self.n_features, 1), float(gate_init))
        )

        nn.init.normal_(self.bin_embedding.weight, mean=0.0, std=self.init_std)
        for net in (self.gamma_proj, self.beta_proj):
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
        if values.ndim != 3 or values.shape[-1] < 3:
            raise ValueError(
                "D3 values must be [B,F,3] = [offset, continuous, mask], "
                f"got {tuple(values.shape)}"
            )

        vals = values.to(dtype=torch.float32, device=bin_ids.device)
        offset = vals[..., 0:1].clamp(0.0, 1.0)
        cont = vals[..., 1:2].clamp(0.0, 1.0)
        mask = vals[..., 2:3].clamp(0.0, 1.0)

        B, F = bin_ids.shape
        local = self.local_interp(bin_ids, offset)

        gamma = torch.tanh(self.gamma_proj(cont))
        beta = self.beta_proj(cont)

        gate = torch.sigmoid(self.cont_gate_logit).to(device=bin_ids.device)
        gate = gate.unsqueeze(0).expand(B, F, 1)
        g = mask * gate

        value_emb = local * (1.0 + g * gamma) + g * beta
        return self.add_feature_embedding(value_emb)


class FusionAblationTransformer(nn.Module):
    """
    Minimal D3-only reconstruction of the official fusion ablation model.
    """

    def __init__(
        self,
        *,
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

        self.num_bins = int(num_bins)
        self.n_features = int(n_features)
        self.num_classes = int(num_classes)
        self.value_dim = int(value_dim)
        self.feature_dim = int(feature_dim)
        self.cell_dim = int(value_dim + feature_dim)
        self.hidden_dim = int(hidden_dim)

        self.embedding = D3InterpRawFiLMEmbedding(
            num_bins=num_bins,
            n_features=n_features,
            value_dim=value_dim,
            feature_dim=feature_dim,
            gate_init=gate_init,
        )

        self.input_proj = nn.Sequential(
            nn.LayerNorm(self.cell_dim),
            nn.Linear(self.cell_dim, self.hidden_dim),
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=self.hidden_dim * 4,
            dropout=float(dropout),
            activation=str(activation),
            batch_first=True,
            norm_first=bool(norm_first),
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=int(num_layers),
            norm=nn.LayerNorm(self.hidden_dim),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, int(classifier_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(classifier_dropout)),
            nn.Linear(int(classifier_hidden_dim), int(num_classes)),
        )

    def forward(self, tokens: torch.Tensor, z_values: torch.Tensor) -> torch.Tensor:
        cell_emb = self.embedding(tokens, z_values)
        x = self.input_proj(cell_emb)

        B = x.shape[0]
        cls = self.cls_token.expand(B, 1, self.hidden_dim)
        x = torch.cat([cls, x], dim=1)

        encoded = self.encoder(x)
        cls_out = encoded[:, 0, :]
        logits = self.classifier(cls_out)
        return logits


# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export full probabilities from official 03_outputs/06_model D3 checkpoint.")
    p.add_argument("--repo-root", default=".", help="Path to src_baocao repo root.")
    p.add_argument("--model-pt", default="03_outputs/06_model/best_model.pt")
    p.add_argument("--config-json", default="03_outputs/06_model/config.json")
    p.add_argument("--dataset-npz", default="03_outputs/05_dataset/dataset.npz")
    p.add_argument("--metadata-json", default="03_outputs/05_dataset/metadata.json")
    p.add_argument("--train-raw", default="01_split/train_raw.csv")
    p.add_argument("--val-raw", default="01_split/val_raw.csv")
    p.add_argument("--out-csv", default="05_test/outputs/B0_wrong_top2_audit/val_predictions_with_probs.csv")
    p.add_argument("--out-dir", default="05_test/outputs/B0_wrong_top2_audit")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--strict-load", action="store_true", help="Use strict load_state_dict. Default is robust strict first, then fail with details.")
    return p.parse_args()


def repo_path(repo_root: Path, path_like: str | Path) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    return repo_root / p


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("[WARN] cuda requested but torch.cuda.is_available() is False; using CPU.", file=sys.stderr)
        return torch.device("cpu")
    return torch.device(device_arg)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_label_name(x: Any) -> str:
    return str(x).strip()


def label_names_from_config_meta(config: Dict[str, Any], meta: Dict[str, Any], num_classes: int) -> List[str]:
    if isinstance(config.get("label_names"), list) and len(config["label_names"]) == num_classes:
        return [normalize_label_name(x) for x in config["label_names"]]

    mapping = meta.get("label_mapping")
    if isinstance(mapping, dict) and mapping:
        inv = {int(v): normalize_label_name(k) for k, v in mapping.items()}
        if all(i in inv for i in range(num_classes)):
            return [inv[i] for i in range(num_classes)]

    if "label_names" in meta:
        names = meta["label_names"]
        if isinstance(names, list) and len(names) == num_classes:
            return [normalize_label_name(x) for x in names]

    return [f"class_{i}" for i in range(num_classes)]


def get_required_array(data: Dict[str, np.ndarray], key: str) -> np.ndarray:
    if key not in data:
        raise KeyError(f"dataset npz missing required array: {key}. Available keys: {sorted(data.keys())}")
    return data[key]


def load_dataset_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def compute_raw_scaled_continuous(
    meta: Dict[str, Any],
    train_raw_path: Path,
    val_raw_path: Path,
    expected_shape: Tuple[int, int],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Recompute official D3 raw_scaled branch:
      train-only minmax on raw features, clip val to [0,1],
      constant train features -> 0.5.
    """
    if "feature_names" not in meta:
        raise KeyError("metadata missing feature_names; cannot recompute raw_scaled continuous branch")

    feature_names = [str(x) for x in meta["feature_names"]]

    if not train_raw_path.exists():
        raise FileNotFoundError(f"train_raw not found: {train_raw_path}")
    if not val_raw_path.exists():
        raise FileNotFoundError(f"val_raw not found: {val_raw_path}")

    train_df = pd.read_csv(train_raw_path)
    val_df = pd.read_csv(val_raw_path)

    missing_train = [f for f in feature_names if f not in train_df.columns]
    missing_val = [f for f in feature_names if f not in val_df.columns]
    if missing_train:
        raise KeyError(f"train_raw missing features: {missing_train[:20]}")
    if missing_val:
        raise KeyError(f"val_raw missing features: {missing_val[:20]}")

    X_train_raw = train_df.loc[:, feature_names].to_numpy(dtype=np.float64)
    X_val_raw = val_df.loc[:, feature_names].to_numpy(dtype=np.float64)

    if X_val_raw.shape != expected_shape:
        raise ValueError(f"val_raw feature shape mismatch: {X_val_raw.shape} vs expected {expected_shape}")

    if not np.isfinite(X_train_raw).all():
        raise ValueError("train_raw selected features contain NaN/Inf")
    if not np.isfinite(X_val_raw).all():
        raise ValueError("val_raw selected features contain NaN/Inf")

    mn = X_train_raw.min(axis=0)
    mx = X_train_raw.max(axis=0)
    denom = mx - mn
    constant = np.isclose(denom, 0.0)
    denom_safe = denom.copy()
    denom_safe[constant] = 1.0

    X_val_cont = (X_val_raw - mn) / denom_safe
    X_val_cont[:, constant] = 0.5
    X_val_cont = np.clip(X_val_cont, 0.0, 1.0).astype(np.float32)

    info = {
        "source": "raw_scaled",
        "scale": "train_only_minmax_linear_clip_val",
        "train_raw": str(train_raw_path),
        "val_raw": str(val_raw_path),
        "n_constant_features": int(constant.sum()),
        "constant_features": [feature_names[i] for i, flag in enumerate(constant) if flag],
        "val_min": float(X_val_cont.min()),
        "val_max": float(X_val_cont.max()),
    }
    return X_val_cont, info


def build_val_values(
    data: Dict[str, np.ndarray],
    meta: Dict[str, Any],
    train_raw_path: Path,
    val_raw_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    X_val_bin = get_required_array(data, "X_val_bin").astype(np.int64)
    y_val = get_required_array(data, "y_val").astype(np.int64)

    if "X_val_offset" not in data:
        raise KeyError("dataset npz missing X_val_offset; D3 requires offset.")
    X_val_offset = data["X_val_offset"].astype(np.float32)

    continuous_info: Dict[str, Any]
    if "X_val_continuous" in data:
        X_val_continuous = data["X_val_continuous"].astype(np.float32)
        continuous_info = {
            "source": "X_val_continuous_from_dataset_npz",
            "val_min": float(X_val_continuous.min()),
            "val_max": float(X_val_continuous.max()),
        }
    else:
        X_val_continuous, continuous_info = compute_raw_scaled_continuous(
            meta=meta,
            train_raw_path=train_raw_path,
            val_raw_path=val_raw_path,
            expected_shape=X_val_bin.shape,
        )

    if "X_val_mask" in data:
        X_val_mask = data["X_val_mask"].astype(np.float32)
    else:
        X_val_mask = np.ones_like(X_val_offset, dtype=np.float32)

    if X_val_offset.shape != X_val_bin.shape:
        raise ValueError(f"X_val_offset shape mismatch: {X_val_offset.shape} vs {X_val_bin.shape}")
    if X_val_continuous.shape != X_val_bin.shape:
        raise ValueError(f"X_val_continuous shape mismatch: {X_val_continuous.shape} vs {X_val_bin.shape}")
    if X_val_mask.shape != X_val_bin.shape:
        raise ValueError(f"X_val_mask shape mismatch: {X_val_mask.shape} vs {X_val_bin.shape}")

    V_val = np.stack(
        [
            np.clip(X_val_offset, 0.0, 1.0),
            np.clip(X_val_continuous, 0.0, 1.0),
            np.clip(X_val_mask, 0.0, 1.0),
        ],
        axis=-1,
    ).astype(np.float32)

    return X_val_bin, V_val, y_val, X_val_continuous, continuous_info


def clean_state_dict_keys(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # Support checkpoints saved from DataParallel or wrapper modules.
    cleaned = {}
    for k, v in state.items():
        nk = k
        for prefix in ("module.", "model."):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        cleaned[nk] = v
    return cleaned


def extract_model_state(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return clean_state_dict_keys(ckpt[key])
        # Maybe checkpoint is already a raw state_dict.
        if all(torch.is_tensor(v) for v in ckpt.values()):
            return clean_state_dict_keys(ckpt)
    raise ValueError(
        "Unsupported checkpoint format. Expected dict with model_state_dict/state_dict/model or raw state_dict."
    )


def load_model_state(model: nn.Module, model_pt: Path, device: torch.device) -> Dict[str, Any]:
    # PyTorch >=2.6 defaults weights_only=True, which can fail on checkpoints
    # that store metadata such as TorchVersion. This checkpoint is produced by
    # our own training run, so loading with weights_only=False is acceptable here.
    ckpt = torch.load(model_pt, map_location=device, weights_only=False)
    state = extract_model_state(ckpt)

    try:
        model.load_state_dict(state, strict=True)
        return {"strict": True, "missing_keys": [], "unexpected_keys": []}
    except RuntimeError as e:
        # Fail loudly with useful diagnostics. Do not silently load a wrong architecture.
        missing, unexpected = model.load_state_dict(state, strict=False)
        msg = [
            "Strict checkpoint loading failed.",
            f"Missing keys count: {len(missing)}",
            f"Unexpected keys count: {len(unexpected)}",
            f"First missing keys: {list(missing)[:20]}",
            f"First unexpected keys: {list(unexpected)[:20]}",
            "",
            "Original strict error:",
            str(e),
        ]
        raise RuntimeError("\n".join(msg))


def make_model(config: Dict[str, Any], meta: Dict[str, Any], data: Dict[str, np.ndarray]) -> FusionAblationTransformer:
    model_cfg = config.get("model", config)
    num_bins = int(config.get("num_bins", config.get("effective_token_budget", config.get("K_artifact", 512))))

    if "n_features" in config:
        n_features = int(config["n_features"])
    elif "n_features" in meta:
        n_features = int(meta["n_features"])
    else:
        n_features = int(get_required_array(data, "X_val_bin").shape[1])

    if "num_classes" in config:
        num_classes = int(config["num_classes"])
    else:
        y_val = get_required_array(data, "y_val")
        num_classes = int(np.max(y_val)) + 1

    return FusionAblationTransformer(
        num_bins=num_bins,
        n_features=n_features,
        num_classes=num_classes,
        value_dim=int(model_cfg.get("value_dim", 32)),
        feature_dim=int(model_cfg.get("feature_dim", 32)),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        num_layers=int(model_cfg.get("num_layers", 3)),
        num_heads=int(model_cfg.get("num_heads", 4)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        classifier_hidden_dim=int(model_cfg.get("classifier_hidden_dim", 128)),
        classifier_dropout=float(model_cfg.get("classifier_dropout", 0.1)),
        norm_first=bool(model_cfg.get("norm_first", True)),
        gate_init=float(model_cfg.get("gate_init", 0.0)),
        activation=str(model_cfg.get("activation", "gelu")),
    )


def zip_outputs(out_dir: Path, zip_name: str = "B0_export_06_model_probs_output.zip") -> Path:
    out_zip = out_dir / zip_name
    if out_zip.exists():
        out_zip.unlink()

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fp in out_dir.rglob("*"):
            if fp.is_file() and fp != out_zip:
                z.write(fp, fp.relative_to(out_dir))
    return out_zip


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()

    model_pt = repo_path(repo_root, args.model_pt)
    config_json = repo_path(repo_root, args.config_json)
    dataset_npz = repo_path(repo_root, args.dataset_npz)
    metadata_json = repo_path(repo_root, args.metadata_json)
    train_raw = repo_path(repo_root, args.train_raw)
    val_raw = repo_path(repo_root, args.val_raw)
    out_csv = repo_path(repo_root, args.out_csv)
    out_dir = repo_path(repo_root, args.out_dir)

    required = {
        "model_pt": model_pt,
        "config_json": config_json,
        "dataset_npz": dataset_npz,
        "metadata_json": metadata_json,
        "train_raw": train_raw,
        "val_raw": val_raw,
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    config = load_json(config_json)
    meta = load_json(metadata_json)
    data = load_dataset_npz(dataset_npz)

    X_val_bin, V_val, y_val, X_val_continuous, continuous_info = build_val_values(
        data=data,
        meta=meta,
        train_raw_path=train_raw,
        val_raw_path=val_raw,
    )
    n_val, n_features = X_val_bin.shape

    device = pick_device(args.device)

    model = make_model(config, meta, data).to(device)
    load_info = load_model_state(model, model_pt, device)
    model.eval()

    num_classes = int(model.num_classes)
    label_names = label_names_from_config_meta(config, meta, num_classes)

    ds = TensorDataset(
        torch.as_tensor(X_val_bin, dtype=torch.long),
        torch.as_tensor(V_val, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=int(args.batch_size), shuffle=False)

    all_probs: List[np.ndarray] = []
    all_logits: List[np.ndarray] = []
    with torch.no_grad():
        for xb, vb in loader:
            xb = xb.to(device)
            vb = vb.to(device)
            logits = model(xb, vb)
            probs = torch.softmax(logits, dim=1)
            all_logits.append(logits.detach().cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())

    logits_np = np.concatenate(all_logits, axis=0)
    probs_np = np.concatenate(all_probs, axis=0)

    if probs_np.shape != (n_val, num_classes):
        raise ValueError(f"Unexpected probs shape: {probs_np.shape}, expected {(n_val, num_classes)}")

    top_order = np.argsort(-probs_np, axis=1)
    top1 = top_order[:, 0]
    top2 = top_order[:, 1]
    pred = top1

    idx = np.arange(n_val)
    confidence = probs_np[idx, pred]
    correct = pred == y_val

    rows = {
        "sample_index": idx.astype(int),
        "true_id": y_val.astype(int),
        "true_label": [label_names[int(i)] for i in y_val],
        "pred_id": pred.astype(int),
        "pred_label": [label_names[int(i)] for i in pred],
        "correct": correct.astype(bool),
        "confidence": confidence.astype(float),
        "top1_id": top1.astype(int),
        "top1_label": [label_names[int(i)] for i in top1],
        "top1_score": probs_np[idx, top1].astype(float),
        "top2_id": top2.astype(int),
        "top2_label": [label_names[int(i)] for i in top2],
        "top2_score": probs_np[idx, top2].astype(float),
        "top12_margin": (probs_np[idx, top1] - probs_np[idx, top2]).astype(float),
        "true_in_top2": ((top1 == y_val) | (top2 == y_val)).astype(bool),
    }

    out = pd.DataFrame(rows)

    for i, name in enumerate(label_names):
        safe = normalize_label_name(name).replace(" ", "_")
        out[f"prob_{safe}"] = probs_np[:, i].astype(float)
        out[f"logit_{safe}"] = logits_np[:, i].astype(float)

    out.to_csv(out_csv, index=False)

    acc = float(correct.mean())
    top2_acc = float(((top1 == y_val) | (top2 == y_val)).mean())
    wrong = ~correct
    wrong_total = int(wrong.sum())
    wrong_true_in_top2 = int((((top1 == y_val) | (top2 == y_val)) & wrong).sum())
    wrong_true_in_top2_rate = float(wrong_true_in_top2 / wrong_total) if wrong_total else 0.0

    manifest = {
        "stage": "B0_export_06_model_probs",
        "purpose": "Export full probs/top2 for B0 wrong-sample top2 audit.",
        "inputs": {k: str(v) for k, v in required.items()},
        "outputs": {"out_csv": str(out_csv)},
        "config_summary": {
            "run_id": config.get("run_id"),
            "run_spec": config.get("run_spec"),
            "num_bins": int(model.num_bins),
            "n_features": int(model.n_features),
            "num_classes": int(model.num_classes),
            "label_names": label_names,
        },
        "data_shapes": {
            "X_val_bin": list(X_val_bin.shape),
            "V_val": list(V_val.shape),
            "y_val": list(y_val.shape),
        },
        "continuous_info": continuous_info,
        "load_info": load_info,
        "metrics_from_export": {
            "accuracy": acc,
            "top2_accuracy": top2_acc,
            "wrong_total": wrong_total,
            "wrong_true_in_top2": wrong_true_in_top2,
            "wrong_true_in_top2_rate": wrong_true_in_top2_rate,
        },
        "device": str(device),
        "batch_size": int(args.batch_size),
    }
    manifest_path = out_dir / "B0_export_06_model_probs_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_path = out_dir / "B0_export_06_model_probs_summary.md"
    summary_path.write_text(
        "\n".join([
            "# B0 Export 06 Model Probabilities",
            "",
            f"- output_csv: `{out_csv}`",
            f"- n_val: `{n_val}`",
            f"- n_features: `{n_features}`",
            f"- label_names: `{label_names}`",
            f"- accuracy_from_export: `{acc:.10f}`",
            f"- top2_accuracy_from_export: `{top2_acc:.10f}`",
            f"- wrong_total: `{wrong_total}`",
            f"- wrong_true_in_top2: `{wrong_true_in_top2}`",
            f"- wrong_true_in_top2_rate: `{wrong_true_in_top2_rate:.10f}`",
            "",
            "This is an export-only step. It does not modify the official baseline files.",
            "",
        ]),
        encoding="utf-8",
    )

    out_zip = zip_outputs(out_dir)

    print("===== B0 export 06_model probabilities done =====")
    print("model_pt:", model_pt)
    print("config_json:", config_json)
    print("dataset_npz:", dataset_npz)
    print("metadata_json:", metadata_json)
    print("out_csv:", out_csv)
    print("manifest:", manifest_path)
    print("zip:", out_zip)
    print("accuracy:", acc)
    print("top2_accuracy:", top2_acc)
    print("wrong_total:", wrong_total)
    print("wrong_true_in_top2:", wrong_true_in_top2)
    print("wrong_true_in_top2_rate:", wrong_true_in_top2_rate)


if __name__ == "__main__":
    main()
