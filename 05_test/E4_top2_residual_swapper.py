#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E4 Top-2 Residual KEEP/SWAP Swapper.

Purpose
-------
Fix the actual decision error of the official D3 baseline.

Prior failures:
  E1a1 asked: "within pair RS, is this Ransomware or Spyware?"
  E2c/E2d asked: "can a small correction head change logits safely?"
  E3 asked: "can representation regularization improve subtype boundary?"

But the real deployment question is:

  Baseline D3 already selected top1 and top2.
  Should we KEEP top1, or SWAP to top2?

E4 directly trains this residual action.

No tree.
No teacher.
No reclassification from scratch.
Final prediction = top1 or top2 of official D3.

Training labels
---------------
For a hard top-2 malware pair:
  if y == top1: action = KEEP
  if y == top2: action = SWAP
  if y not in {top1, top2}: sample is not useful for action training
                          because both keep and swap are wrong.

This is not the same as pair classification. It is baseline-error correction.

Modes
-----
1) eval_zero:
   load official D3 checkpoint, no swaps, reproduce baseline.

2) train_swapper:
   freeze official D3 by default.
   train a KEEP/SWAP head from:
     - official D3 representation before classifier
     - official D3 logits/probs/margin/entropy
     - top1/top2 label embeddings

Outputs
-------
  E4_summary.json
  E4_history.csv
  E4_threshold_sweep_train.csv
  E4_threshold_sweep_val.csv
  E4_best_per_class_f1.csv
  E4_best_confusion_matrix.csv
  E4_val_predictions_best.csv
  E4_best_pair_fix_damage.csv
  E4_action_metrics.json
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
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
    spec = importlib.util.spec_from_file_location("official_07_train_for_e4", str(train_script))
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
                raise ValueError(f"Requested tap module '{requested}' not found.")
            return requested

        for name in ["classifier", "classification_head", "head", "mlp_head", "fc"]:
            if name in modules and name != "":
                return name

        candidates = []
        for name, mod in self.model.named_modules():
            lname = name.lower()
            if name and ("classifier" in lname or lname.endswith("head") or ".head" in lname):
                candidates.append((name, mod))
        if candidates:
            candidates = sorted(candidates, key=lambda x: (len(x[0].split(".")), len(x[0])))
            return candidates[0][0]

        linear_names = [name for name, mod in self.model.named_modules() if isinstance(mod, nn.Linear) and name]
        if linear_names:
            return linear_names[-1]

        raise RuntimeError("Cannot find classifier/head/linear module for representation tap.")

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
        _ = model_forward_logits(backbone, tokens, values)
        rep = tap.get_rep()
    rep_dim = int(rep.shape[-1])
    module_name = tap.module_name
    tap.close()
    return rep_dim, module_name


def entropy_from_probs(p: torch.Tensor) -> torch.Tensor:
    ent = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=1)
    return ent / math.log(p.shape[1])


class E4Top2Swapper(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        tap_module_name: str,
        rep_dim: int,
        num_classes: int,
        label_emb_dim: int = 8,
        hidden_dim: int = 96,
        dropout: float = 0.1,
        detach_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.tap = RepresentationTap(self.backbone, tap_module_name)
        self.rep_dim = int(rep_dim)
        self.num_classes = int(num_classes)
        self.detach_backbone = bool(detach_backbone)

        self.label_emb = nn.Embedding(num_classes, label_emb_dim)

        # scalar features:
        # top1_logit, top2_logit, logit_margin,
        # top1_prob, top2_prob, prob_margin,
        # entropy, top2_over_top1_prob, abs_margin
        scalar_dim = 9
        in_dim = rep_dim + 2 * label_emb_dim + scalar_dim

        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, tokens: torch.Tensor, values: torch.Tensor):
        self.tap.clear()
        if self.detach_backbone:
            with torch.no_grad():
                logits = model_forward_logits(self.backbone, tokens, values)
                rep = self.tap.get_rep()
        else:
            logits = model_forward_logits(self.backbone, tokens, values)
            rep = self.tap.get_rep()

        probs = F.softmax(logits, dim=1)
        top_logits, top_ids = torch.topk(logits, k=2, dim=1)
        top_probs = probs.gather(1, top_ids)

        top1 = top_ids[:, 0]
        top2 = top_ids[:, 1]
        l1 = top_logits[:, 0]
        l2 = top_logits[:, 1]
        p1 = top_probs[:, 0]
        p2 = top_probs[:, 1]
        margin = l1 - l2
        pmargin = p1 - p2
        ent = entropy_from_probs(probs)
        ratio = p2 / p1.clamp_min(1e-12)

        scalars = torch.stack([
            l1,
            l2,
            margin,
            p1,
            p2,
            pmargin,
            ent,
            ratio,
            margin.abs(),
        ], dim=1)

        e1 = self.label_emb(top1)
        e2 = self.label_emb(top2)
        x = torch.cat([rep, e1, e2, scalars], dim=1)
        swap_logit = self.head(x).squeeze(1)

        return {
            "logits": logits,
            "probs": probs,
            "rep": rep,
            "top1": top1,
            "top2": top2,
            "top1_logit": l1,
            "top2_logit": l2,
            "top1_prob": p1,
            "top2_prob": p2,
            "margin": margin,
            "entropy": ent,
            "swap_logit": swap_logit,
            "swap_prob": torch.sigmoid(swap_logit),
        }


def make_loader(ds, batch_size: int, shuffle: bool, seed: int, num_workers: int, device: torch.device):
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=gen if shuffle else None,
        num_workers=int(num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )


def hard_pair_candidate_np(top1: np.ndarray, top2: np.ndarray, pair_ids: List[Tuple[int, int]]) -> np.ndarray:
    mask = np.zeros_like(top1, dtype=bool)
    for a, b in pair_ids:
        mask |= ((top1 == a) & (top2 == b)) | ((top1 == b) & (top2 == a))
    return mask


def hard_pair_candidate_torch(top1: torch.Tensor, top2: torch.Tensor, pair_ids: List[Tuple[int, int]]) -> torch.Tensor:
    mask = torch.zeros_like(top1, dtype=torch.bool)
    for a, b in pair_ids:
        mask |= ((top1 == int(a)) & (top2 == int(b))) | ((top1 == int(b)) & (top2 == int(a)))
    return mask


def supervised_action_mask_torch(y: torch.Tensor, top1: torch.Tensor, top2: torch.Tensor, pair_ids: List[Tuple[int, int]]):
    hard = hard_pair_candidate_torch(top1, top2, pair_ids)
    keep = hard & (y == top1)
    swap = hard & (y == top2)
    sup = keep | swap
    target = swap.float()
    return sup, target, keep, swap, hard


def supervised_action_mask_np(y: np.ndarray, top1: np.ndarray, top2: np.ndarray, pair_ids: List[Tuple[int, int]]):
    hard = hard_pair_candidate_np(top1, top2, pair_ids)
    keep = hard & (y == top1)
    swap = hard & (y == top2)
    sup = keep | swap
    target = swap.astype(np.int64)
    return sup, target, keep, swap, hard


def compute_action_weights_from_arrays(y: np.ndarray, top1: np.ndarray, top2: np.ndarray, pair_ids: List[Tuple[int, int]], keep_cost: float, swap_cost: float):
    sup, target, keep, swap, hard = supervised_action_mask_np(y, top1, top2, pair_ids)
    n_keep = int((sup & (target == 0)).sum())
    n_swap = int((sup & (target == 1)).sum())
    n = max(1, n_keep + n_swap)
    if n_keep == 0 or n_swap == 0:
        return float(keep_cost), float(swap_cost), {"n_keep": n_keep, "n_swap": n_swap, "method": "fallback_cost_only"}
    w_keep = (n / (2.0 * n_keep)) * float(keep_cost)
    w_swap = (n / (2.0 * n_swap)) * float(swap_cost)
    return float(w_keep), float(w_swap), {"n_keep": n_keep, "n_swap": n_swap, "method": "balanced_times_cost"}


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, loss: float = 0.0) -> dict:
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


def action_metrics(y: np.ndarray, top1: np.ndarray, top2: np.ndarray, swap_prob: np.ndarray, pair_ids: List[Tuple[int, int]], threshold: float = 0.5) -> dict:
    sup, target, keep, swap, hard = supervised_action_mask_np(y, top1, top2, pair_ids)
    out = {
        "hard_top2_pair_n": int(hard.sum()),
        "supervised_action_n": int(sup.sum()),
        "keep_label_n": int((sup & (target == 0)).sum()),
        "swap_label_n": int((sup & (target == 1)).sum()),
        "unfixable_hard_pair_n": int((hard & (~sup)).sum()),
    }
    if int(sup.sum()) == 0:
        return out
    yt = target[sup].astype(int)
    ps = swap_prob[sup]
    pa = (ps >= float(threshold)).astype(int)
    out.update({
        "action_threshold": float(threshold),
        "action_accuracy": float(accuracy_score(yt, pa)),
        "action_macro_f1": float(f1_score(yt, pa, average="macro", zero_division=0)),
        "swap_precision": float(precision_recall_fscore_support(yt, pa, labels=[1], zero_division=0)[0][0]),
        "swap_recall": float(precision_recall_fscore_support(yt, pa, labels=[1], zero_division=0)[1][0]),
        "predicted_swap_n": int(pa.sum()),
    })
    try:
        out["action_auc"] = float(roc_auc_score(yt, ps))
    except Exception:
        out["action_auc"] = None
    try:
        out["action_ap"] = float(average_precision_score(yt, ps))
    except Exception:
        out["action_ap"] = None
    return out


def make_swapped_pred(top1: np.ndarray, top2: np.ndarray, swap_prob: np.ndarray, candidate: np.ndarray, threshold: float, margin: Optional[np.ndarray] = None, margin_cap: Optional[float] = None):
    do_swap = candidate & (swap_prob >= float(threshold))
    if margin is not None and margin_cap is not None:
        do_swap &= (margin <= float(margin_cap))
    pred = top1.copy()
    pred[do_swap] = top2[do_swap]
    return pred, do_swap


def threshold_sweep(arr: dict, label_names: List[str], label_to_id: Dict[str, int], thresholds: np.ndarray, margin_cap: Optional[float] = None) -> pd.DataFrame:
    y = arr["y"]
    top1 = arr["top1"]
    top2 = arr["top2"]
    base_pred = top1
    swap_prob = arr["swap_prob"]
    candidate = arr["hard_candidate"]
    margin = arr.get("margin")

    base_macro = float(f1_score(y, base_pred, average="macro"))
    rows = []
    for th in thresholds:
        pred, do_swap = make_swapped_pred(top1, top2, swap_prob, candidate, float(th), margin, margin_cap)
        met = metric_dict(y, pred)
        tr = transition_stats(y, base_pred, pred)
        rows.append({
            "threshold": float(th),
            "margin_cap": margin_cap,
            "macro_f1": met["macro_f1"],
            "accuracy": met["accuracy"],
            "weighted_f1": met["weighted_f1"],
            "delta_macro_vs_base": met["macro_f1"] - base_macro,
            "swap_n": int(do_swap.sum()),
            **tr,
        })
    return pd.DataFrame(rows)


def select_threshold(sweep_df: pd.DataFrame, args) -> dict:
    df = sweep_df.copy()
    df["damage_ratio_filled"] = df["damage_ratio"].fillna(0.0)
    eligible = df.copy()

    if args.threshold_require_net_positive:
        eligible = eligible[eligible["net_gain"] > 0]
    if args.threshold_max_damage_ratio is not None:
        eligible = eligible[eligible["damage_ratio_filled"] <= float(args.threshold_max_damage_ratio)]
    if args.threshold_min_swaps > 0:
        eligible = eligible[eligible["swap_n"] >= int(args.threshold_min_swaps)]

    if len(eligible) == 0:
        eligible = df.copy()
        fallback = True
    else:
        fallback = False

    obj = args.threshold_objective
    if obj == "macro_f1":
        idx = eligible["macro_f1"].idxmax()
    elif obj == "net_gain":
        idx = eligible["net_gain"].idxmax()
    elif obj == "damage_aware":
        # Prefer net gain, but penalize damaged predictions heavily.
        score = eligible["net_gain"] - float(args.damage_penalty) * eligible["correct_to_wrong"]
        idx = score.idxmax()
    else:
        raise ValueError(f"unknown threshold objective: {obj}")

    row = df.loc[idx].to_dict()
    row["fallback_no_constraint_match"] = bool(fallback)
    row["threshold_objective"] = obj
    return row


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


def train_one_epoch(model, loader, optimizer, pair_ids, device, args, keep_w: float, swap_w: float):
    model.train()
    total_loss = 0.0
    total_sup = 0
    total_batches = 0
    skipped = 0
    for tokens, values, y in loader:
        tokens = tokens.to(device, non_blocking=True)
        values = values.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)
        out = model(tokens, values)
        sup, target, keep, swap, hard = supervised_action_mask_torch(y, out["top1"], out["top2"], pair_ids)

        if int(sup.sum().item()) == 0:
            skipped += 1
            continue

        logits = out["swap_logit"][sup]
        target_sup = target[sup]
        weights = torch.where(target_sup > 0.5, torch.tensor(float(swap_w), device=device), torch.tensor(float(keep_w), device=device))
        loss_raw = F.binary_cross_entropy_with_logits(logits, target_sup, reduction="none")
        loss = (loss_raw * weights).mean()

        if not model.detach_backbone and float(args.main_ce_weight) > 0:
            loss = loss + float(args.main_ce_weight) * F.cross_entropy(out["logits"], y)

        loss.backward()
        if float(args.grad_clip_norm) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip_norm))
        optimizer.step()

        n = int(sup.sum().item())
        total_sup += n
        total_batches += 1
        total_loss += float(loss.item()) * n

    return {
        "train_action_loss": float(total_loss / max(1, total_sup)),
        "train_supervised_action_n": int(total_sup),
        "train_used_batches": int(total_batches),
        "train_skipped_batches": int(skipped),
    }


@torch.no_grad()
def collect_arrays(model, loader, pair_ids, device) -> dict:
    model.eval()
    chunks = {
        "y": [],
        "top1": [],
        "top2": [],
        "swap_prob": [],
        "swap_logit": [],
        "margin": [],
        "top1_prob": [],
        "top2_prob": [],
        "entropy": [],
    }
    for tokens, values, y in loader:
        tokens = tokens.to(device, non_blocking=True)
        values = values.to(device, non_blocking=True)
        out = model(tokens, values)
        chunks["y"].append(y.cpu().numpy().astype(int))
        for k in ["top1", "top2"]:
            chunks[k].append(out[k].detach().cpu().numpy().astype(int))
        for k in ["swap_prob", "swap_logit", "margin", "top1_prob", "top2_prob", "entropy"]:
            chunks[k].append(out[k].detach().cpu().numpy().astype(float))

    arr = {k: np.concatenate(v, axis=0) for k, v in chunks.items()}
    arr["hard_candidate"] = hard_pair_candidate_np(arr["top1"], arr["top2"], pair_ids)
    sup, target, keep, swap, hard = supervised_action_mask_np(arr["y"], arr["top1"], arr["top2"], pair_ids)
    arr["supervised_action"] = sup
    arr["action_target"] = target
    arr["baseline_correct"] = arr["top1"] == arr["y"]
    arr["top2_correct"] = arr["top2"] == arr["y"]
    return arr


def eval_epoch_summary(model, train_eval_loader, val_loader, pair_ids, device, args, thresholds):
    val_arr = collect_arrays(model, val_loader, pair_ids, device)
    train_arr = collect_arrays(model, train_eval_loader, pair_ids, device) if args.epoch_eval_train else None
    am = action_metrics(val_arr["y"], val_arr["top1"], val_arr["top2"], val_arr["swap_prob"], pair_ids, threshold=0.5)
    val_sweep = threshold_sweep(val_arr, [], {}, thresholds, margin_cap=args.margin_cap)
    best_val_macro = float(val_sweep["macro_f1"].max())
    best_val_net = int(val_sweep.sort_values("macro_f1", ascending=False).iloc[0]["net_gain"])
    out = {
        "val_base_macro_f1": float(f1_score(val_arr["y"], val_arr["top1"], average="macro")),
        "val_action_auc": am.get("action_auc"),
        "val_action_ap": am.get("action_ap"),
        "val_action_macro_f1_at_05": am.get("action_macro_f1"),
        "val_best_sweep_macro_f1": best_val_macro,
        "val_best_sweep_net_gain": best_val_net,
    }
    if train_arr is not None:
        tam = action_metrics(train_arr["y"], train_arr["top1"], train_arr["top2"], train_arr["swap_prob"], pair_ids, threshold=0.5)
        out["train_action_auc"] = tam.get("action_auc")
        out["train_action_ap"] = tam.get("action_ap")
    return out


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


def save_final_outputs(out_dir, inp, args, repo_root, model, train_eval_loader, val_loader, pair_ids, ckpt_info, rep_dim, tap_module_name, history, best_epoch):
    thresholds = np.linspace(float(args.threshold_min), float(args.threshold_max), int(args.threshold_steps))
    train_arr = collect_arrays(model, train_eval_loader, pair_ids, args._device)
    val_arr = collect_arrays(model, val_loader, pair_ids, args._device)

    train_sweep = threshold_sweep(train_arr, inp["label_names"], inp["label_to_id"], thresholds, margin_cap=args.margin_cap)
    val_sweep = threshold_sweep(val_arr, inp["label_names"], inp["label_to_id"], thresholds, margin_cap=args.margin_cap)

    train_sweep.to_csv(out_dir / "E4_threshold_sweep_train.csv", index=False)
    val_sweep.to_csv(out_dir / "E4_threshold_sweep_val.csv", index=False)

    if args.threshold_select == "train":
        selected = select_threshold(train_sweep, args)
        selected_source = "train"
    elif args.threshold_select == "val":
        selected = select_threshold(val_sweep, args)
        selected_source = "val"
    else:
        selected = {"threshold": float(args.fixed_threshold), "threshold_objective": "fixed", "fallback_no_constraint_match": False}
        selected_source = "fixed"

    th = float(selected["threshold"])
    pred_val, do_swap_val = make_swapped_pred(
        val_arr["top1"], val_arr["top2"], val_arr["swap_prob"],
        val_arr["hard_candidate"], th,
        margin=val_arr["margin"],
        margin_cap=args.margin_cap,
    )
    pred_train, do_swap_train = make_swapped_pred(
        train_arr["top1"], train_arr["top2"], train_arr["swap_prob"],
        train_arr["hard_candidate"], th,
        margin=train_arr["margin"],
        margin_cap=args.margin_cap,
    )

    y_val = val_arr["y"]
    y_train = train_arr["y"]
    base_val = val_arr["top1"]
    base_train = train_arr["top1"]

    final_metrics = metric_dict(y_val, pred_val)
    base_metrics = metric_dict(y_val, base_val)
    train_final_metrics = metric_dict(y_train, pred_train)
    train_base_metrics = metric_dict(y_train, base_train)

    trans_val = transition_stats(y_val, base_val, pred_val)
    trans_train = transition_stats(y_train, base_train, pred_train)

    per, cm = per_class_and_cm(y_val, pred_val, inp["label_names"])
    base_per, base_cm = per_class_and_cm(y_val, base_val, inp["label_names"])
    per.to_csv(out_dir / "E4_best_per_class_f1.csv", index=False)
    cm.to_csv(out_dir / "E4_best_confusion_matrix.csv")
    base_per.to_csv(out_dir / "E4_baseline_per_class_f1.csv", index=False)
    base_cm.to_csv(out_dir / "E4_baseline_confusion_matrix.csv")

    pair_fd = pair_fix_damage(y_val, base_val, pred_val, inp["label_to_id"])
    pair_fd.to_csv(out_dir / "E4_best_pair_fix_damage.csv", index=False)

    action_val = action_metrics(y_val, val_arr["top1"], val_arr["top2"], val_arr["swap_prob"], pair_ids, threshold=th)
    action_train = action_metrics(y_train, train_arr["top1"], train_arr["top2"], train_arr["swap_prob"], pair_ids, threshold=th)
    save_json(out_dir / "E4_action_metrics.json", {"train": action_train, "val": action_val})

    pred_df = pd.DataFrame({
        "sample_index": np.arange(len(y_val), dtype=int),
        "true_id": y_val.astype(int),
        "true_label": [inp["id_to_label"][int(i)] for i in y_val],
        "base_top1_id": val_arr["top1"].astype(int),
        "base_top1_label": [inp["id_to_label"][int(i)] for i in val_arr["top1"]],
        "base_top2_id": val_arr["top2"].astype(int),
        "base_top2_label": [inp["id_to_label"][int(i)] for i in val_arr["top2"]],
        "base_correct": base_val == y_val,
        "top2_correct": val_arr["top2"] == y_val,
        "hard_candidate": val_arr["hard_candidate"],
        "supervised_action": val_arr["supervised_action"],
        "action_target_swap": val_arr["action_target"].astype(int),
        "swap_prob": val_arr["swap_prob"],
        "swap_logit": val_arr["swap_logit"],
        "margin": val_arr["margin"],
        "top1_prob": val_arr["top1_prob"],
        "top2_prob": val_arr["top2_prob"],
        "entropy": val_arr["entropy"],
        "do_swap": do_swap_val,
        "e4_pred_id": pred_val.astype(int),
        "e4_pred_label": [inp["id_to_label"][int(i)] for i in pred_val],
        "e4_correct": pred_val == y_val,
    })
    pred_df["transition"] = "both_wrong"
    pred_df.loc[pred_df["base_correct"] & pred_df["e4_correct"], "transition"] = "both_correct"
    pred_df.loc[(~pred_df["base_correct"]) & pred_df["e4_correct"], "transition"] = "fixed"
    pred_df.loc[pred_df["base_correct"] & (~pred_df["e4_correct"]), "transition"] = "damaged"
    pred_df.to_csv(out_dir / "E4_val_predictions_best.csv", index=False)

    pd.DataFrame(history).to_csv(out_dir / "E4_history.csv", index=False)

    summary = {
        "stage": "E4_top2_residual_swapper",
        "mode": args.mode,
        "research_position": "attention-only residual KEEP/SWAP decision over official D3 top-2",
        "tree_usage": "none",
        "references": {
            "official_D3_baseline_macro_f1": 0.810094,
            "E2c0_eval_zero_macro_f1": 0.810215,
            "E2d1_rep_correction_macro_f1": 0.810390,
            "E3_best_no_tree_macro_f1": 0.813577,
            "E2a_tree_distill_macro_f1": 0.817847,
            "E1b_tree_expert_macro_f1": 0.829387,
        },
        "best_epoch": int(best_epoch),
        "selected_threshold_source": selected_source,
        "selected_threshold": selected,
        "base_val_metrics": base_metrics,
        "final_val_metrics": final_metrics,
        "delta_val": {
            "macro_f1": final_metrics["macro_f1"] - base_metrics["macro_f1"],
            "accuracy": final_metrics["accuracy"] - base_metrics["accuracy"],
            "weighted_f1": final_metrics["weighted_f1"] - base_metrics["weighted_f1"],
        },
        "transition_val_vs_base": trans_val,
        "base_train_metrics": train_base_metrics,
        "final_train_metrics": train_final_metrics,
        "transition_train_vs_base": trans_train,
        "action_metrics": {"train": action_train, "val": action_val},
        "checkpoint_load": ckpt_info,
        "tap_module_name": tap_module_name,
        "rep_dim": int(rep_dim),
        "loss_design": {
            "train_label": "KEEP if true==top1, SWAP if true==top2, ignore if true not in top2",
            "hard_pair_only": True,
            "keep_weight": float(args._keep_weight),
            "swap_weight": float(args._swap_weight),
            "auto_action_weights": bool(args.auto_action_weights),
            "keep_cost": float(args.keep_cost),
            "swap_cost": float(args.swap_cost),
            "freeze_backbone": bool(args.freeze_backbone),
        },
        "outputs": {
            "history": str(out_dir / "E4_history.csv"),
            "best_model": str(out_dir / "E4_best_model.pt"),
            "threshold_sweep_train": str(out_dir / "E4_threshold_sweep_train.csv"),
            "threshold_sweep_val": str(out_dir / "E4_threshold_sweep_val.csv"),
            "per_class": str(out_dir / "E4_best_per_class_f1.csv"),
            "confusion_matrix": str(out_dir / "E4_best_confusion_matrix.csv"),
            "val_predictions": str(out_dir / "E4_val_predictions_best.csv"),
            "pair_fix_damage": str(out_dir / "E4_best_pair_fix_damage.csv"),
            "action_metrics": str(out_dir / "E4_action_metrics.json"),
        },
    }
    save_json(out_dir / "E4_summary.json", summary)
    write_summary_md(out_dir, summary)


def zip_dir(src_dir: Path, zip_path: Path):
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(src_dir.rglob("*")):
            if p.is_file() and p != zip_path:
                z.write(p, p.relative_to(src_dir.parent))


def write_summary_md(out_dir: Path, summary: dict):
    tr = summary["transition_val_vs_base"]
    text = f"""# E4 Top-2 Residual KEEP/SWAP Swapper

## Position

```text
attention-only
no tree in training
no tree in inference
residual action: KEEP top1 or SWAP to top2
```

## Why E4

E4 does not ask "which class is this sample?" again.

It asks the actual correction question:

```text
official D3 top1/top2 are known
should we trust top1 or swap to top2?
```

## Selected threshold

```text
source    = {summary['selected_threshold_source']}
threshold = {summary['selected_threshold'].get('threshold')}
objective = {summary['selected_threshold'].get('threshold_objective')}
```

## Validation result

```text
base macro-F1  = {summary['base_val_metrics']['macro_f1']:.6f}
E4 macro-F1    = {summary['final_val_metrics']['macro_f1']:.6f}
delta macro-F1 = {summary['delta_val']['macro_f1']:+.6f}

base accuracy  = {summary['base_val_metrics']['accuracy']:.6f}
E4 accuracy    = {summary['final_val_metrics']['accuracy']:.6f}
```

## Transition vs official D3 top1

```text
wrong_to_correct = {tr['wrong_to_correct']}
correct_to_wrong = {tr['correct_to_wrong']}
net_gain         = {tr['net_gain']}
damage_ratio     = {tr['damage_ratio']}
changed_pred_n   = {tr['changed_pred_n']}
```

## Action metrics

```text
val supervised_action_n = {summary['action_metrics']['val'].get('supervised_action_n')}
val keep_label_n        = {summary['action_metrics']['val'].get('keep_label_n')}
val swap_label_n        = {summary['action_metrics']['val'].get('swap_label_n')}
val action_auc          = {summary['action_metrics']['val'].get('action_auc')}
val action_ap           = {summary['action_metrics']['val'].get('action_ap')}
```

## Key files

- `E4_summary.json`
- `E4_history.csv`
- `E4_threshold_sweep_train.csv`
- `E4_threshold_sweep_val.csv`
- `E4_best_per_class_f1.csv`
- `E4_best_confusion_matrix.csv`
- `E4_val_predictions_best.csv`
- `E4_best_pair_fix_damage.csv`
- `E4_action_metrics.json`
"""
    (out_dir / "E4_summary.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="E4 top2 residual KEEP/SWAP swapper")
    parser.add_argument("--dataset-npz", default="03_outputs/05_dataset/dataset.npz")
    parser.add_argument("--metadata-json", default="03_outputs/05_dataset/metadata.json")
    parser.add_argument("--train-raw", default="01_split/train_raw.csv")
    parser.add_argument("--val-raw", default="01_split/val_raw.csv")
    parser.add_argument("--official-train", default="02_src/07_train.py")
    parser.add_argument("--baseline-config", default="03_outputs/06_model/config.json")
    parser.add_argument("--baseline-checkpoint", default="03_outputs/06_model/best_model.pt")
    parser.add_argument("--baseline-pred", default="03_outputs/06_model/val_predictions_best.csv")
    parser.add_argument("--out-dir", default="05_test/outputs/E4_top2_residual_swapper")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--mode", default="eval_zero", choices=["eval_zero", "train_swapper"])

    parser.add_argument("--tap-module", default="auto")
    parser.add_argument("--label-emb-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--freeze-backbone", action="store_true", default=True)
    parser.add_argument("--unfreeze-backbone", dest="freeze_backbone", action="store_false")

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", default="warmup_cosine", choices=["none", "warmup_cosine"])
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--main-ce-weight", type=float, default=0.0)

    parser.add_argument("--auto-action-weights", action="store_true", default=True)
    parser.add_argument("--manual-action-weights", dest="auto_action_weights", action="store_false")
    parser.add_argument("--keep-weight", type=float, default=2.0)
    parser.add_argument("--swap-weight", type=float, default=1.0)
    parser.add_argument("--keep-cost", type=float, default=2.0)
    parser.add_argument("--swap-cost", type=float, default=1.0)

    parser.add_argument("--threshold-select", default="train", choices=["train", "val", "fixed"])
    parser.add_argument("--fixed-threshold", type=float, default=0.5)
    parser.add_argument("--threshold-objective", default="damage_aware", choices=["macro_f1", "net_gain", "damage_aware"])
    parser.add_argument("--damage-penalty", type=float, default=1.5)
    parser.add_argument("--threshold-max-damage-ratio", type=float, default=0.75)
    parser.add_argument("--threshold-require-net-positive", action="store_true", default=True)
    parser.add_argument("--threshold-min-swaps", type=int, default=1)
    parser.add_argument("--threshold-min", type=float, default=0.01)
    parser.add_argument("--threshold-max", type=float, default=0.99)
    parser.add_argument("--threshold-steps", type=int, default=99)
    parser.add_argument("--margin-cap", type=float, default=None)

    parser.add_argument("--selection-metric", default="val_action_auc", choices=["val_action_auc", "val_action_ap", "val_best_sweep_macro_f1", "neg_val_action_loss"])
    parser.add_argument("--epoch-eval-train", action="store_true", default=False)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--make-zip", action="store_true", default=True)
    parser.add_argument("--no-zip", dest="make_zip", action="store_false")
    args = parser.parse_args()

    if args.mode == "eval_zero":
        args.epochs = 0
        args.threshold_select = "fixed"
        args.fixed_threshold = 1.1
        args.threshold_min = 1.1
        args.threshold_max = 1.1
        args.threshold_steps = 1

    repo_root = repo_root_from_here()
    out_dir = resolve_path(args.out_dir, repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(args.seed))
    device = pick_device(args.device)
    args._device = device

    print(f"[E4] repo_root={repo_root}", flush=True)
    print(f"[E4] out_dir={out_dir}", flush=True)
    print(f"[E4] mode={args.mode}", flush=True)
    print(f"[E4] device={device}", flush=True)
    print("[E4] no tree. residual action = KEEP top1 or SWAP top1->top2.", flush=True)

    inp = load_official_inputs(args, repo_root)
    pair_ids = [(int(inp["label_to_id"][a]), int(inp["label_to_id"][b])) for a, b in HARD_PAIRS]

    train_loader = make_loader(inp["train_ds"], int(args.batch_size), True, int(args.seed), int(args.num_workers), device)
    train_eval_loader = make_loader(inp["train_ds"], int(args.batch_size), False, int(args.seed), int(args.num_workers), device)
    val_loader = make_loader(inp["val_ds"], int(args.batch_size), False, int(args.seed), int(args.num_workers), device)

    model_cfg = get_baseline_model_cfg(args, repo_root)
    backbone, backbone_kwargs = build_official_d3_model(
        inp["train_mod"], model_cfg, len(inp["feature_names"]), inp["num_bins"], len(inp["label_names"]), device
    )
    ckpt_info = load_checkpoint_into_model(backbone, resolve_path(args.baseline_checkpoint, repo_root), device, strict=True)
    print(f"[E4] checkpoint loaded: {ckpt_info}", flush=True)

    rep_dim, tap_module_name = infer_rep_dim(backbone, val_loader, device, args.tap_module)
    print(f"[E4] tap_module={tap_module_name} rep_dim={rep_dim}", flush=True)

    model = E4Top2Swapper(
        backbone=backbone,
        tap_module_name=tap_module_name,
        rep_dim=rep_dim,
        num_classes=len(inp["label_names"]),
        label_emb_dim=int(args.label_emb_dim),
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.head_dropout),
        detach_backbone=bool(args.freeze_backbone),
    ).to(device)

    if args.freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False

    # Compute action weights from initial official D3 predictions on train.
    init_train_arr = collect_arrays(model, train_eval_loader, pair_ids, device)
    if args.auto_action_weights:
        keep_w, swap_w, weight_info = compute_action_weights_from_arrays(
            init_train_arr["y"], init_train_arr["top1"], init_train_arr["top2"], pair_ids,
            keep_cost=float(args.keep_cost),
            swap_cost=float(args.swap_cost),
        )
    else:
        keep_w, swap_w = float(args.keep_weight), float(args.swap_weight)
        weight_info = {"method": "manual", "n_keep": None, "n_swap": None}
    args._keep_weight = keep_w
    args._swap_weight = swap_w
    print(f"[E4] action_weights keep={keep_w:.6f} swap={swap_w:.6f} info={weight_info}", flush=True)

    save_json(out_dir / "E4_run_config.json", {
        "stage": "E4_top2_residual_swapper",
        "mode": args.mode,
        "tree_usage": "none",
        "args": {k: (str(v) if k == "_device" else v) for k, v in vars(args).items()},
        "device": str(device),
        "label_names": inp["label_names"],
        "pair_ids": pair_ids,
        "num_bins": int(inp["num_bins"]),
        "continuous_info": inp["continuous_info"],
        "backbone_kwargs": backbone_kwargs,
        "checkpoint_load": ckpt_info,
        "tap_module_name": tap_module_name,
        "rep_dim": int(rep_dim),
        "action_weight_info": weight_info,
        "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "total_params": int(sum(p.numel() for p in model.parameters())),
    })

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(args.lr), weight_decay=float(args.weight_decay))

    history = []
    best_score = -1e18
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0
    thresholds_for_epoch = np.linspace(float(args.threshold_min), float(args.threshold_max), min(int(args.threshold_steps), 41))

    if args.mode == "eval_zero":
        print("[E4] eval_zero: no training, threshold fixed >1 so no swaps.", flush=True)
    else:
        print(
            f"[E4] training trainable_params={sum(p.numel() for p in model.parameters() if p.requires_grad):,} "
            f"total_params={sum(p.numel() for p in model.parameters()):,}",
            flush=True,
        )

    for epoch in range(1, int(args.epochs) + 1):
        lr_epoch = compute_lr(epoch, args)
        set_optimizer_lr(optimizer, lr_epoch)
        t0 = time.time()
        train_met = train_one_epoch(model, train_loader, optimizer, pair_ids, device, args, keep_w, swap_w)
        eval_met = eval_epoch_summary(model, train_eval_loader, val_loader, pair_ids, device, args, thresholds_for_epoch)
        dt = time.time() - t0

        if args.selection_metric == "val_action_auc":
            score_val = eval_met.get("val_action_auc")
        elif args.selection_metric == "val_action_ap":
            score_val = eval_met.get("val_action_ap")
        elif args.selection_metric == "val_best_sweep_macro_f1":
            score_val = eval_met.get("val_best_sweep_macro_f1")
        elif args.selection_metric == "neg_val_action_loss":
            score_val = -float(train_met["train_action_loss"])
        else:
            score_val = eval_met.get("val_action_auc")
        if score_val is None or (isinstance(score_val, float) and np.isnan(score_val)):
            score_val = -1e18

        row = {
            "epoch": int(epoch),
            "lr": float(lr_epoch),
            "seconds": float(dt),
            "selection_score": float(score_val),
            **train_met,
            **eval_met,
        }
        history.append(row)

        improved = float(score_val) > best_score + float(args.min_delta)
        if improved:
            best_score = float(score_val)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            torch.save({
                "model_state_dict": best_state,
                "epoch": int(epoch),
                "selection_score": float(score_val),
                "args": {k: (str(v) if k == "_device" else v) for k, v in vars(args).items()},
                "pair_ids": pair_ids,
                "label_names": inp["label_names"],
                "backbone_kwargs": backbone_kwargs,
                "tap_module_name": tap_module_name,
                "rep_dim": int(rep_dim),
                "tree_usage": "none",
            }, out_dir / "E4_best_model.pt")
        else:
            no_improve += 1

        if epoch == 1 or improved or epoch % int(args.log_every) == 0:
            print(
                f"[E4] ep={epoch:03d} lr={lr_epoch:.3e} "
                f"loss={train_met['train_action_loss']:.5f} sup_n={train_met['train_supervised_action_n']} "
                f"val_auc={eval_met.get('val_action_auc')} val_ap={eval_met.get('val_action_ap')} "
                f"val_best_macro={eval_met.get('val_best_sweep_macro_f1'):.6f} "
                f"best_score={best_score:.6f}@{best_epoch} noimp={no_improve} sec={dt:.1f}",
                flush=True,
            )

        if no_improve >= int(args.patience):
            print(f"[E4] early stop at epoch {epoch}", flush=True)
            break

    model.load_state_dict(best_state)
    if args.mode == "eval_zero":
        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": 0,
            "selection_score": 0.0,
            "args": {k: (str(v) if k == "_device" else v) for k, v in vars(args).items()},
            "pair_ids": pair_ids,
            "label_names": inp["label_names"],
            "backbone_kwargs": backbone_kwargs,
            "tap_module_name": tap_module_name,
            "rep_dim": int(rep_dim),
            "tree_usage": "none",
        }, out_dir / "E4_best_model.pt")
        best_epoch = 0
        history.append({"epoch": 0, "selection_score": 0.0})

    save_final_outputs(out_dir, inp, args, repo_root, model, train_eval_loader, val_loader, pair_ids, ckpt_info, rep_dim, tap_module_name, history, best_epoch)

    if args.make_zip:
        zip_path = out_dir.with_suffix(".zip")
        zip_dir(out_dir, zip_path)
        print(f"[E4] zipped outputs: {zip_path}", flush=True)

    summary = load_json(out_dir / "E4_summary.json")
    print("[E4] done.", flush=True)
    print(f"[E4] mode={args.mode}", flush=True)
    print(f"[E4] threshold={summary['selected_threshold'].get('threshold')}", flush=True)
    print(f"[E4] base_macro_f1={summary['base_val_metrics']['macro_f1']:.6f}", flush=True)
    print(f"[E4] macro_f1={summary['final_val_metrics']['macro_f1']:.6f}", flush=True)
    print(f"[E4] net_gain={summary['transition_val_vs_base']['net_gain']}", flush=True)


if __name__ == "__main__":
    main()
