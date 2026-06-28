#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E2d Baseline-compatible Official D3 Representation Correction.

Why E2d
-------
E2c0 proved that the wrapper can reproduce the official D3 checkpoint.
E2c1 trained correction from main_logits only and was almost useless:
it only recalibrated the already-made 4-class logits.

E2d keeps baseline compatibility but gives the correction head richer
attention representation:
  official D3 backbone/checkpoint
  hook representation right before the official classifier
  pair-correction head reads that representation
  final_logits = main_logits + correction_scale * correction_logits

No saved baseline logits are used as final output.
No tree is used in inference.

This is still attention-based:
  representation comes from the official D3 Transformer attention model.

Modes
-----
1) eval_zero
   Load official D3 checkpoint.
   correction_scale = 0.
   No training.
   Must reproduce official D3 baseline ≈ 0.810094 / E2c0 ≈ 0.810215.

2) freeze_rep_correction
   Load official D3 checkpoint.
   Freeze official D3 backbone.
   Train only representation-based correction heads.

3) finetune_all
   Load official D3 checkpoint.
   Train correction heads and unfreeze official D3 with small LR.

Default output:
  05_test/outputs/E2d_official_d3_rep_correction/
  05_test/outputs/E2d_official_d3_rep_correction.zip
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


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


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available; falling back to CPU.", flush=True)
        return torch.device("cpu")
    return torch.device(device_arg)


def import_official_train(train_script: Path):
    if not train_script.exists():
        raise FileNotFoundError(f"official 07_train.py not found: {train_script}")
    src_dir = train_script.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    spec = importlib.util.spec_from_file_location("official_07_train_for_e2d", str(train_script))
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


def load_official_inputs(args, repo_root: Path) -> dict:
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

    M_train = np.ones_like(X_train_bin, dtype=np.float32)
    M_val = np.ones_like(X_val_bin, dtype=np.float32)

    train_ds = train_mod.FusionAblationDataset(
        X_train_bin, X_train_offset, X_train_cont.astype(np.float32), M_train, y_train
    )
    val_ds = train_mod.FusionAblationDataset(
        X_val_bin, X_val_offset, X_val_cont.astype(np.float32), M_val, y_val
    )

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
        "train_ds": train_ds,
        "val_ds": val_ds,
        "y_train": y_train,
        "y_val": y_val,
    }


def get_baseline_model_cfg(args, repo_root: Path) -> dict:
    cfg = {}
    path = resolve_path(args.baseline_config, repo_root)
    if path.exists():
        loaded = load_json(path)
        if isinstance(loaded.get("model"), dict):
            cfg.update(loaded["model"])
        if isinstance(loaded.get("model_config"), dict):
            cfg.update({k: v for k, v in loaded["model_config"].items() if k not in cfg})
        for k in [
            "value_dim", "feature_dim", "hidden_dim", "num_layers", "num_heads",
            "dropout", "classifier_hidden_dim", "classifier_dropout", "norm_first",
            "gate_init", "activation", "num_bins", "effective_token_budget",
        ]:
            if k in loaded and k not in cfg:
                cfg[k] = loaded[k]

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


def build_official_d3_model(train_mod, model_cfg: dict, n_features: int, num_bins: int, num_classes: int, device: torch.device):
    cls = train_mod.FusionAblationTransformer
    sig = inspect.signature(cls)
    kwargs = {
        "run_id": "D3",
        "num_bins": int(num_bins),
        "n_features": int(n_features),
        "num_classes": int(num_classes),
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


def safe_torch_load_checkpoint(ckpt_path: Path, device: torch.device):
    # PyTorch >=2.6 defaults torch.load(weights_only=True), which can fail on older
    # trusted local checkpoints that include metadata such as TorchVersion.
    try:
        return torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(ckpt_path, map_location=device)


def extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for key in ["model_state_dict", "state_dict", "model"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        if all(isinstance(k, str) for k in ckpt.keys()) and any(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt
    raise ValueError("Cannot extract model state dict from checkpoint")


def load_checkpoint_into_model(model: nn.Module, ckpt_path: Path, device: torch.device, strict: bool = True) -> dict:
    ckpt = safe_torch_load_checkpoint(ckpt_path, device)
    sd = extract_state_dict(ckpt)
    cleaned = {}
    for k, v in sd.items():
        nk = k
        for prefix in ["module.", "backbone.", "model."]:
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        cleaned[nk] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=strict)
    return {
        "path": str(ckpt_path),
        "strict": bool(strict),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


def model_forward_logits(model: nn.Module, tokens: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    try:
        out = model(tokens, z_values=values)
    except TypeError:
        out = model(tokens, values)
    if isinstance(out, dict):
        if "logits" in out:
            out = out["logits"]
        elif "final_logits" in out:
            out = out["final_logits"]
        else:
            raise ValueError(f"Dict output has no logits key: {list(out.keys())}")
    if isinstance(out, tuple):
        out = out[0]
    return out


class RepresentationTap:
    def __init__(self, model: nn.Module, module_name: str = "auto"):
        self.model = model
        self.module_name = self.resolve_module_name(module_name)
        self.module = dict(model.named_modules())[self.module_name]
        self.last_rep: Optional[torch.Tensor] = None
        self.handle = self.module.register_forward_pre_hook(self._hook)

    def resolve_module_name(self, requested: str) -> str:
        modules = dict(self.model.named_modules())
        if requested and requested != "auto":
            if requested not in modules:
                raise ValueError(f"Requested tap module '{requested}' not found. Available examples: {list(modules.keys())[:30]}")
            return requested

        # Preferred: hook before classifier/head to capture the representation that feeds the official classifier.
        preferred_attrs = ["classifier", "classification_head", "head", "mlp_head", "fc"]
        for name in preferred_attrs:
            if name in modules and name != "":
                return name

        # Prefer module names containing classifier/head.
        candidates = []
        for name, mod in self.model.named_modules():
            lname = name.lower()
            if name and ("classifier" in lname or lname.endswith("head") or ".head" in lname):
                candidates.append((name, mod))
        if candidates:
            # Shortest name is usually the outer classifier block.
            candidates = sorted(candidates, key=lambda x: (len(x[0].split(".")), len(x[0])))
            return candidates[0][0]

        # Fallback: last Linear module. Not ideal, but still representation before final linear.
        linear_names = [name for name, mod in self.model.named_modules() if isinstance(mod, nn.Linear) and name]
        if linear_names:
            return linear_names[-1]

        raise RuntimeError("Cannot find a classifier/head/linear module to tap representation.")

    def _hook(self, module, inputs):
        if not inputs:
            self.last_rep = None
            return
        rep = inputs[0]
        if isinstance(rep, (list, tuple)):
            rep = rep[0]
        self.last_rep = rep

    def clear(self):
        self.last_rep = None

    def close(self):
        self.handle.remove()

    def get_rep(self) -> torch.Tensor:
        if self.last_rep is None:
            raise RuntimeError(f"Representation tap '{self.module_name}' did not capture anything.")
        rep = self.last_rep
        if rep.dim() == 3:
            # If sequence representation is accidentally captured, use first token/CLS.
            rep = rep[:, 0, :]
        elif rep.dim() > 3:
            rep = rep.view(rep.shape[0], -1)
        return rep


def infer_rep_dim(backbone: nn.Module, loader: DataLoader, device: torch.device, tap_module: str):
    tap = RepresentationTap(backbone, tap_module)
    backbone.eval()
    with torch.no_grad():
        tokens, values, y = next(iter(loader))
        tokens = tokens.to(device)
        values = values.to(device)
        tap.clear()
        logits = model_forward_logits(backbone, tokens, values)
        rep = tap.get_rep()
    rep_dim = int(rep.shape[-1])
    module_name = tap.module_name
    tap.close()
    return rep_dim, module_name


class OfficialD3RepresentationCorrection(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        tap_module_name: str,
        rep_dim: int,
        num_classes: int,
        pair_class_ids: Dict[str, Tuple[int, int]],
        correction_hidden_dim: int = 64,
        correction_dropout: float = 0.1,
        correction_scale_init: float = 0.05,
        fixed_correction_scale: bool = False,
        detach_representation: bool = True,
        correction_input: str = "rep",
    ):
        super().__init__()
        self.backbone = backbone
        self.tap = RepresentationTap(self.backbone, tap_module_name)
        self.rep_dim = int(rep_dim)
        self.num_classes = int(num_classes)
        self.pair_keys = ["RS", "RT", "ST"]
        self.pair_class_ids = {k: tuple(v) for k, v in pair_class_ids.items()}
        self.detach_representation = bool(detach_representation)
        self.correction_input = correction_input

        if correction_input == "rep":
            in_dim = self.rep_dim
        elif correction_input == "rep_logits":
            in_dim = self.rep_dim + self.num_classes
        else:
            raise ValueError("--correction-input must be rep or rep_logits")

        self.shared = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, correction_hidden_dim),
            nn.GELU(),
            nn.Dropout(correction_dropout),
        )
        self.pair_heads = nn.ModuleDict({
            pk: nn.Linear(correction_hidden_dim, 2)
            for pk in self.pair_keys
        })

        if fixed_correction_scale:
            self.register_buffer("correction_scale", torch.tensor(float(correction_scale_init), dtype=torch.float32))
        else:
            self.correction_scale = nn.Parameter(torch.tensor(float(correction_scale_init), dtype=torch.float32))

    def set_backbone_requires_grad(self, requires_grad: bool):
        for p in self.backbone.parameters():
            p.requires_grad = bool(requires_grad)

    def forward(self, tokens: torch.Tensor, values: torch.Tensor):
        self.tap.clear()
        main_logits = model_forward_logits(self.backbone, tokens, values)
        rep = self.tap.get_rep()
        if self.detach_representation:
            rep = rep.detach()
        if self.correction_input == "rep_logits":
            logits_in = main_logits.detach() if self.detach_representation else main_logits
            x = torch.cat([rep, logits_in], dim=1)
        else:
            x = rep

        h = self.shared(x)
        correction = torch.zeros_like(main_logits)
        pair_logits = {}
        for pk in self.pair_keys:
            logits = self.pair_heads[pk](h)
            pair_logits[pk] = logits
            ida, idb = self.pair_class_ids[pk]
            delta = logits[:, 1] - logits[:, 0]
            correction[:, ida] = correction[:, ida] - 0.5 * delta
            correction[:, idb] = correction[:, idb] + 0.5 * delta

        final_logits = main_logits + self.correction_scale * correction
        return {
            "main_logits": main_logits,
            "final_logits": final_logits,
            "pair_logits": pair_logits,
            "correction_scale": self.correction_scale,
            "rep": rep,
        }


def make_loader(ds, batch_size: int, shuffle: bool, seed: int, num_workers: int, device: torch.device):
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=gen if shuffle else None,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )


def compute_class_weights(y: np.ndarray, num_classes: int, device: torch.device):
    counts = np.bincount(y.astype(int), minlength=num_classes).astype(np.float64)
    weights = counts.sum() / np.maximum(1.0, num_classes * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_pair_weights(y: np.ndarray, pair_class_ids: Dict[str, Tuple[int, int]], device: torch.device):
    out = {}
    for pk, (ida, idb) in pair_class_ids.items():
        mask = (y == ida) | (y == idb)
        yb = (y[mask] == idb).astype(int)
        counts = np.bincount(yb, minlength=2).astype(np.float64)
        weights = counts.sum() / np.maximum(1.0, 2.0 * counts)
        weights = weights / weights.mean()
        out[pk] = torch.tensor(weights, dtype=torch.float32, device=device)
    return out


def pair_aux_loss(pair_logits: Dict[str, torch.Tensor], y: torch.Tensor, pair_class_ids: Dict[str, Tuple[int, int]], pair_weights: Dict[str, torch.Tensor]):
    losses = []
    logs = {}
    for pk, logits in pair_logits.items():
        ida, idb = pair_class_ids[pk]
        mask = (y == ida) | (y == idb)
        if mask.sum() == 0:
            continue
        target = (y[mask] == idb).long()
        weight = pair_weights.get(pk)
        loss = nn.functional.cross_entropy(logits[mask], target, weight=weight)
        losses.append(loss)
        logs[f"{pk}_loss"] = float(loss.detach().cpu().item())
    if not losses:
        return torch.tensor(0.0, device=y.device), logs
    return torch.stack(losses).mean(), logs


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, loss: float) -> dict:
    return {
        "loss": float(loss),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }


def per_class_and_cm(y_true: np.ndarray, y_pred: np.ndarray, label_names: List[str]):
    labels = list(range(len(label_names)))
    prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    per = pd.DataFrame([
        {"class_id": i, "label": label_names[i], "precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]), "support": int(sup[i])}
        for i in labels
    ])
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels), index=label_names, columns=label_names)
    return per, cm


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


def pair_fix_damage(y_true: np.ndarray, base_pred: np.ndarray, new_pred: np.ndarray, label_to_id: Dict[str, int]) -> pd.DataFrame:
    base_correct = base_pred == y_true
    new_correct = new_pred == y_true
    rows = []
    for a, b in HARD_PAIRS:
        ida, idb = label_to_id[a], label_to_id[b]
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
            tid = label_to_id[true_label]
            oid = label_to_id[other_label]
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


def compute_lr(epoch: int, args) -> float:
    base_lr = float(args.lr)
    if args.scheduler == "none":
        return base_lr
    warm = int(args.warmup_epochs)
    epochs = int(args.epochs)
    if warm > 0 and epoch <= warm:
        return base_lr * epoch / warm
    progress = (epoch - warm) / max(1, epochs - warm)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (float(args.min_lr_ratio) + (1.0 - float(args.min_lr_ratio)) * cosine)


def set_optimizer_lr(optimizer, lr: float):
    for g in optimizer.param_groups:
        g["lr"] = lr


def set_train_state(model: OfficialD3RepresentationCorrection, mode: str):
    if mode == "freeze_rep_correction":
        model.train()
        model.backbone.eval()
    else:
        model.train()


def train_one_epoch(model, loader, optimizer, ce, ce_main, pair_weights, pair_class_ids, device, args):
    set_train_state(model, args.mode)
    ys, pred_finals, pred_mains = [], [], []
    total_loss = total_final = total_main = total_pair = total_l2 = 0.0
    n = 0
    for tokens, values, y in loader:
        tokens = tokens.to(device, non_blocking=True)
        values = values.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)
        out = model(tokens, values)
        final_logits = out["final_logits"]
        main_logits = out["main_logits"]

        loss_final = ce(final_logits, y)
        loss_main = ce_main(main_logits, y)
        loss_pair, _ = pair_aux_loss(out["pair_logits"], y, pair_class_ids, pair_weights)
        loss_l2 = model.correction_scale.pow(2)

        loss = (
            loss_final
            + float(args.main_aux_weight) * loss_main
            + float(args.pair_aux_weight) * loss_pair
            + float(args.correction_l2) * loss_l2
        )
        loss.backward()
        if float(args.grad_clip_norm) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip_norm))
        optimizer.step()

        bs = int(y.shape[0])
        n += bs
        total_loss += float(loss.item()) * bs
        total_final += float(loss_final.item()) * bs
        total_main += float(loss_main.item()) * bs
        total_pair += float(loss_pair.item()) * bs
        total_l2 += float(loss_l2.item()) * bs
        ys.append(y.detach().cpu().numpy())
        pred_finals.append(final_logits.argmax(dim=1).detach().cpu().numpy())
        pred_mains.append(main_logits.argmax(dim=1).detach().cpu().numpy())

    y_np = np.concatenate(ys)
    pf = np.concatenate(pred_finals)
    pm = np.concatenate(pred_mains)
    met = metric_dict(y_np, pf, total_loss / max(1, n))
    met["final_ce_loss"] = float(total_final / max(1, n))
    met["main_ce_loss"] = float(total_main / max(1, n))
    met["pair_loss"] = float(total_pair / max(1, n))
    met["correction_l2"] = float(total_l2 / max(1, n))
    met["main_macro_f1"] = float(f1_score(y_np, pm, average="macro"))
    met["correction_scale"] = float(model.correction_scale.detach().cpu().item())
    return met


@torch.no_grad()
def evaluate_model(model, loader, ce, ce_main, pair_weights, pair_class_ids, device, args):
    model.eval()
    ys, pred_finals, pred_mains = [], [], []
    total_loss = total_final = total_main = total_pair = 0.0
    n = 0
    for tokens, values, y in loader:
        tokens = tokens.to(device, non_blocking=True)
        values = values.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()

        out = model(tokens, values)
        final_logits = out["final_logits"]
        main_logits = out["main_logits"]

        loss_final = ce(final_logits, y)
        loss_main = ce_main(main_logits, y)
        loss_pair, _ = pair_aux_loss(out["pair_logits"], y, pair_class_ids, pair_weights)
        loss = loss_final + float(args.main_aux_weight) * loss_main + float(args.pair_aux_weight) * loss_pair

        bs = int(y.shape[0])
        n += bs
        total_loss += float(loss.item()) * bs
        total_final += float(loss_final.item()) * bs
        total_main += float(loss_main.item()) * bs
        total_pair += float(loss_pair.item()) * bs
        ys.append(y.detach().cpu().numpy())
        pred_finals.append(final_logits.argmax(dim=1).detach().cpu().numpy())
        pred_mains.append(main_logits.argmax(dim=1).detach().cpu().numpy())

    y_np = np.concatenate(ys)
    pf = np.concatenate(pred_finals)
    pm = np.concatenate(pred_mains)
    met = metric_dict(y_np, pf, total_loss / max(1, n))
    met["final_ce_loss"] = float(total_final / max(1, n))
    met["main_ce_loss"] = float(total_main / max(1, n))
    met["pair_loss"] = float(total_pair / max(1, n))
    met["main_macro_f1"] = float(f1_score(y_np, pm, average="macro"))
    met["correction_scale"] = float(model.correction_scale.detach().cpu().item())
    return met, y_np, pf, pm


def load_baseline_predictions(args, repo_root: Path, inp: dict) -> Optional[pd.DataFrame]:
    path = resolve_path(args.baseline_pred, repo_root)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "sample_index" not in df.columns:
        df["sample_index"] = np.arange(len(df))
    for c in ["true_label", "pred_label"]:
        if c in df.columns:
            df[c] = df[c].map(strip_label)
    if "true_id" not in df.columns and "true_label" in df.columns:
        df["true_id"] = df["true_label"].map(inp["label_to_id"])
    if "pred_id" not in df.columns and "pred_label" in df.columns:
        df["pred_id"] = df["pred_label"].map(inp["label_to_id"])
    if "pred_label" not in df.columns and "pred_id" in df.columns:
        df["pred_label"] = df["pred_id"].astype(int).map(inp["id_to_label"])
    if "true_label" not in df.columns and "true_id" in df.columns:
        df["true_label"] = df["true_id"].astype(int).map(inp["id_to_label"])
    if "correct" not in df.columns:
        df["correct"] = df["true_id"].astype(int) == df["pred_id"].astype(int)
    return df.sort_values("sample_index").reset_index(drop=True)


def save_outputs(out_dir: Path, inp: dict, args, repo_root: Path, final_met: dict, y_val, pred_final, pred_main, history: List[dict], ckpt_info: dict, best_epoch: int, rep_dim: int, tap_module_name: str):
    pd.DataFrame(history).to_csv(out_dir / "E2d_history.csv", index=False)
    per, cm = per_class_and_cm(y_val, pred_final, inp["label_names"])
    per_main, cm_main = per_class_and_cm(y_val, pred_main, inp["label_names"])
    per.to_csv(out_dir / "E2d_best_per_class_f1.csv", index=False)
    cm.to_csv(out_dir / "E2d_best_confusion_matrix.csv")
    per_main.to_csv(out_dir / "E2d_best_main_only_per_class_f1.csv", index=False)
    cm_main.to_csv(out_dir / "E2d_best_main_only_confusion_matrix.csv")

    base = load_baseline_predictions(args, repo_root, inp)
    trans = None
    pred_df = pd.DataFrame({
        "sample_index": np.arange(len(y_val), dtype=int),
        "true_id": y_val.astype(int),
        "true_label": [inp["id_to_label"][int(i)] for i in y_val],
        "e2d_pred_id": pred_final.astype(int),
        "e2d_pred_label": [inp["id_to_label"][int(i)] for i in pred_final],
        "e2d_correct": pred_final == y_val,
        "main_only_pred_id": pred_main.astype(int),
        "main_only_pred_label": [inp["id_to_label"][int(i)] for i in pred_main],
        "main_only_correct": pred_main == y_val,
    })
    if base is not None and len(base) == len(y_val):
        base_pred = base["pred_id"].to_numpy(dtype=int)
        trans = transition_stats(y_val, base_pred, pred_final)
        pair_fd = pair_fix_damage(y_val, base_pred, pred_final, inp["label_to_id"])
        pair_fd.to_csv(out_dir / "E2d_best_pair_fix_damage.csv", index=False)

        pred_df["base_pred_id"] = base_pred
        pred_df["base_pred_label"] = [inp["id_to_label"][int(i)] for i in base_pred]
        pred_df["base_correct"] = base_pred == y_val
        pred_df["transition"] = "both_wrong"
        pred_df.loc[pred_df["base_correct"] & pred_df["e2d_correct"], "transition"] = "both_correct"
        pred_df.loc[(~pred_df["base_correct"]) & pred_df["e2d_correct"], "transition"] = "fixed"
        pred_df.loc[pred_df["base_correct"] & (~pred_df["e2d_correct"]), "transition"] = "damaged"
    pred_df.to_csv(out_dir / "E2d_val_predictions_best.csv", index=False)

    summary = {
        "stage": "E2d_official_d3_rep_correction",
        "mode": args.mode,
        "research_position": "official D3 attention representation + trainable pair correction head",
        "baseline_compatibility_rule": "eval_zero with correction=0 must reproduce official D3 baseline macro-F1 ≈ 0.810094 / E2c0 ≈ 0.810215",
        "references": {
            "official_D3_baseline_macro_f1": 0.810094,
            "E2c0_eval_zero_macro_f1": 0.810215,
            "E2c1_freeze_logits_correction_macro_f1": 0.810253,
            "E2a_tree_guided_distill_macro_f1": 0.817847,
            "E1b_tree_expert_macro_f1": 0.829387,
        },
        "best_epoch": int(best_epoch),
        "best_metrics": final_met,
        "transition_vs_official_baseline": trans,
        "checkpoint_load": ckpt_info,
        "tap_module_name": tap_module_name,
        "rep_dim": int(rep_dim),
        "correction_input": args.correction_input,
        "outputs": {
            "history": str(out_dir / "E2d_history.csv"),
            "best_model": str(out_dir / "E2d_best_model.pt"),
            "per_class": str(out_dir / "E2d_best_per_class_f1.csv"),
            "confusion_matrix": str(out_dir / "E2d_best_confusion_matrix.csv"),
            "val_predictions": str(out_dir / "E2d_val_predictions_best.csv"),
            "pair_fix_damage": str(out_dir / "E2d_best_pair_fix_damage.csv"),
        },
        "guardrail": "If eval_zero does not reproduce baseline, do not trust training modes.",
    }
    save_json(out_dir / "E2d_summary.json", summary)
    write_summary_md(out_dir, summary)


def zip_dir(src_dir: Path, zip_path: Path):
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(src_dir.rglob("*")):
            if p.is_file() and p != zip_path:
                z.write(p, p.relative_to(src_dir.parent))


def write_summary_md(out_dir: Path, summary: dict):
    trans = summary.get("transition_vs_official_baseline") or {}
    text = f"""# E2d Official D3 Representation Correction

## Mode

```text
mode = {summary['mode']}
```

## Research position

This is still attention-based:

```text
official D3 Transformer attention backbone
representation tapped from: {summary['tap_module_name']}
rep_dim = {summary['rep_dim']}
+ trainable pair-correction head
```

No saved baseline logits are used as final output.

## Baseline compatibility rule

```text
eval_zero + fixed correction=0
must reproduce official D3 baseline macro-F1 ≈ 0.810094 / E2c0 ≈ 0.810215
```

## References

```text
D3 official baseline = 0.810094
E2c0 eval_zero       = 0.810215
E2c1 logits-correct  = 0.810253
E2a distillation     = 0.817847
E1b tree expert      = 0.829387
```

## E2d result

```text
best_epoch = {summary['best_epoch']}
accuracy   = {summary['best_metrics']['accuracy']:.6f}
macro-F1   = {summary['best_metrics']['macro_f1']:.6f}
weighted   = {summary['best_metrics']['weighted_f1']:.6f}
main_macro = {summary['best_metrics']['main_macro_f1']:.6f}
corr_scale = {summary['best_metrics']['correction_scale']:.6f}
```

## Transition vs official baseline

```text
wrong_to_correct = {trans.get('wrong_to_correct')}
correct_to_wrong = {trans.get('correct_to_wrong')}
net_gain         = {trans.get('net_gain')}
damage_ratio     = {trans.get('damage_ratio')}
changed_pred_n   = {trans.get('changed_pred_n')}
```

## Key files

- `E2d_summary.json`
- `E2d_history.csv`
- `E2d_best_model.pt`
- `E2d_best_per_class_f1.csv`
- `E2d_best_confusion_matrix.csv`
- `E2d_val_predictions_best.csv`
- `E2d_best_pair_fix_damage.csv`
"""
    (out_dir / "E2d_summary.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="E2d official D3 representation correction")
    parser.add_argument("--dataset-npz", default="03_outputs/05_dataset/dataset.npz")
    parser.add_argument("--metadata-json", default="03_outputs/05_dataset/metadata.json")
    parser.add_argument("--train-raw", default="01_split/train_raw.csv")
    parser.add_argument("--val-raw", default="01_split/val_raw.csv")
    parser.add_argument("--official-train", default="02_src/07_train.py")
    parser.add_argument("--baseline-config", default="03_outputs/06_model/config.json")
    parser.add_argument("--baseline-checkpoint", default="03_outputs/06_model/best_model.pt")
    parser.add_argument("--baseline-pred", default="03_outputs/06_model/val_predictions_best.csv")
    parser.add_argument("--out-dir", default="05_test/outputs/E2d_official_d3_rep_correction")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--mode", default="eval_zero", choices=["eval_zero", "freeze_rep_correction", "finetune_all"])

    # Representation tap / correction.
    parser.add_argument("--tap-module", default="auto", help="Module name to hook. auto = classifier/head before official classifier.")
    parser.add_argument("--correction-input", default="rep", choices=["rep", "rep_logits"])
    parser.add_argument("--correction-hidden-dim", type=int, default=64)
    parser.add_argument("--correction-dropout", type=float, default=0.1)
    parser.add_argument("--correction-scale-init", type=float, default=0.05)
    parser.add_argument("--fixed-correction-scale", action="store_true", default=False)
    parser.add_argument("--detach-representation", action="store_true", default=False)

    # Train.
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", default="warmup_cosine", choices=["none", "warmup_cosine"])
    parser.add_argument("--warmup-epochs", type=int, default=4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--main-aux-weight", type=float, default=0.0)
    parser.add_argument("--pair-aux-weight", type=float, default=0.30)
    parser.add_argument("--correction-l2", type=float, default=0.0)
    parser.add_argument("--no-class-weights", dest="use_class_weights", action="store_false")
    parser.set_defaults(use_class_weights=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)

    parser.add_argument("--make-zip", action="store_true", default=True)
    parser.add_argument("--no-zip", dest="make_zip", action="store_false")
    args = parser.parse_args()

    if args.mode == "eval_zero":
        args.correction_scale_init = 0.0
        args.fixed_correction_scale = True
        args.epochs = 0
        args.main_aux_weight = 0.0
        args.pair_aux_weight = 0.0
        args.detach_representation = True
    elif args.mode == "freeze_rep_correction":
        args.detach_representation = True

    repo_root = repo_root_from_here()
    out_dir = resolve_path(args.out_dir, repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(args.seed))
    device = pick_device(args.device)

    print(f"[E2d] repo_root={repo_root}", flush=True)
    print(f"[E2d] out_dir={out_dir}", flush=True)
    print(f"[E2d] mode={args.mode}", flush=True)
    print(f"[E2d] device={device}", flush=True)

    inp = load_official_inputs(args, repo_root)
    pair_class_ids = {
        pk: (int(inp["label_to_id"][a]), int(inp["label_to_id"][b]))
        for pk, (a, b) in PAIR_FROM_KEY.items()
    }

    train_loader = make_loader(inp["train_ds"], int(args.batch_size), True, int(args.seed), int(args.num_workers), device)
    val_loader = make_loader(inp["val_ds"], int(args.batch_size), False, int(args.seed), int(args.num_workers), device)

    model_cfg = get_baseline_model_cfg(args, repo_root)
    backbone, backbone_kwargs = build_official_d3_model(
        inp["train_mod"], model_cfg, len(inp["feature_names"]), inp["num_bins"], len(inp["label_names"]), device
    )
    ckpt_info = load_checkpoint_into_model(
        backbone,
        resolve_path(args.baseline_checkpoint, repo_root),
        device,
        strict=True,
    )
    print(f"[E2d] checkpoint loaded: {ckpt_info}", flush=True)

    rep_dim, tap_module_name = infer_rep_dim(backbone, val_loader, device, args.tap_module)
    print(f"[E2d] tap_module={tap_module_name} rep_dim={rep_dim}", flush=True)

    model = OfficialD3RepresentationCorrection(
        backbone=backbone,
        tap_module_name=tap_module_name,
        rep_dim=rep_dim,
        num_classes=len(inp["label_names"]),
        pair_class_ids=pair_class_ids,
        correction_hidden_dim=int(args.correction_hidden_dim),
        correction_dropout=float(args.correction_dropout),
        correction_scale_init=float(args.correction_scale_init),
        fixed_correction_scale=bool(args.fixed_correction_scale),
        detach_representation=bool(args.detach_representation),
        correction_input=str(args.correction_input),
    ).to(device)

    if args.mode == "freeze_rep_correction":
        model.set_backbone_requires_grad(False)
    elif args.mode == "finetune_all":
        model.set_backbone_requires_grad(True)

    class_weight = compute_class_weights(inp["y_train"], len(inp["label_names"]), device) if args.use_class_weights else None
    ce = nn.CrossEntropyLoss(weight=class_weight)
    ce_main = nn.CrossEntropyLoss(weight=class_weight)
    pair_weights = make_pair_weights(inp["y_train"], pair_class_ids, device) if args.use_class_weights else {}

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=float(args.lr), weight_decay=float(args.weight_decay)) if trainable_params else None

    save_json(out_dir / "E2d_run_config.json", {
        "stage": "E2d_official_d3_rep_correction",
        "mode": args.mode,
        "args": vars(args),
        "device": str(device),
        "label_names": inp["label_names"],
        "pair_class_ids": pair_class_ids,
        "num_bins": int(inp["num_bins"]),
        "continuous_info": inp["continuous_info"],
        "backbone_kwargs": backbone_kwargs,
        "checkpoint_load": ckpt_info,
        "tap_module_name": tap_module_name,
        "rep_dim": int(rep_dim),
        "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "total_params": int(sum(p.numel() for p in model.parameters())),
    })

    history = []
    best_score = -1.0
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0

    if args.mode == "eval_zero":
        print("[E2d] eval_zero: no training; evaluating checkpoint with correction=0.", flush=True)
    else:
        print(
            f"[E2d] training mode={args.mode} trainable_params={sum(p.numel() for p in model.parameters() if p.requires_grad):,} "
            f"total_params={sum(p.numel() for p in model.parameters()):,}",
            flush=True,
        )

    for epoch in range(1, int(args.epochs) + 1):
        lr_epoch = compute_lr(epoch, args)
        set_optimizer_lr(optimizer, lr_epoch)

        t0 = time.time()
        train_met = train_one_epoch(model, train_loader, optimizer, ce, ce_main, pair_weights, pair_class_ids, device, args)
        val_met, yv, pred_final, pred_main = evaluate_model(model, val_loader, ce, ce_main, pair_weights, pair_class_ids, device, args)
        dt = time.time() - t0

        row = {
            "epoch": int(epoch),
            "lr": float(lr_epoch),
            "seconds": float(dt),
            **{f"train_{k}": v for k, v in train_met.items()},
            **{f"val_{k}": v for k, v in val_met.items()},
        }
        history.append(row)

        score = val_met["macro_f1"]
        improved = score > best_score + float(args.min_delta)
        if improved:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            torch.save({
                "model_state_dict": best_state,
                "epoch": int(epoch),
                "val_metrics": val_met,
                "args": vars(args),
                "pair_class_ids": pair_class_ids,
                "label_names": inp["label_names"],
                "backbone_kwargs": backbone_kwargs,
                "tap_module_name": tap_module_name,
                "rep_dim": int(rep_dim),
            }, out_dir / "E2d_best_model.pt")
        else:
            no_improve += 1

        if epoch == 1 or improved or epoch % int(args.log_every) == 0:
            print(
                f"[E2d] ep={epoch:03d} lr={lr_epoch:.3e} "
                f"train_f1={train_met['macro_f1']:.6f} val_f1={val_met['macro_f1']:.6f} "
                f"val_main_f1={val_met['main_macro_f1']:.6f} corr={val_met['correction_scale']:.6f} "
                f"best={best_score:.6f}@{best_epoch} noimp={no_improve} sec={dt:.1f}",
                flush=True,
            )

        if no_improve >= int(args.patience):
            print(f"[E2d] early stop at epoch {epoch}", flush=True)
            break

    model.load_state_dict(best_state)
    final_met, y_val, pred_final, pred_main = evaluate_model(model, val_loader, ce, ce_main, pair_weights, pair_class_ids, device, args)

    if args.mode == "eval_zero":
        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": 0,
            "val_metrics": final_met,
            "args": vars(args),
            "pair_class_ids": pair_class_ids,
            "label_names": inp["label_names"],
            "backbone_kwargs": backbone_kwargs,
            "tap_module_name": tap_module_name,
            "rep_dim": int(rep_dim),
        }, out_dir / "E2d_best_model.pt")
        history.append({
            "epoch": 0,
            "lr": 0.0,
            "seconds": 0.0,
            **{f"val_{k}": v for k, v in final_met.items()},
        })

    save_outputs(out_dir, inp, args, repo_root, final_met, y_val, pred_final, pred_main, history, ckpt_info, best_epoch, rep_dim, tap_module_name)

    if args.make_zip:
        zip_path = out_dir.with_suffix(".zip")
        zip_dir(out_dir, zip_path)
        print(f"[E2d] zipped outputs: {zip_path}", flush=True)

    print("[E2d] done.", flush=True)
    print(f"[E2d] mode={args.mode}", flush=True)
    print(f"[E2d] macro_f1={final_met['macro_f1']:.6f}", flush=True)
    print(f"[E2d] main_macro_f1={final_met['main_macro_f1']:.6f}", flush=True)
    print(f"[E2d] correction_scale={final_met['correction_scale']:.6f}", flush=True)


if __name__ == "__main__":
    main()
