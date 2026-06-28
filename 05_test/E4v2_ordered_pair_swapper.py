#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E4v2 Ordered-pair-specific Top-2 KEEP/SWAP Swapper.

Fix over E4
-----------
E4 used one shared KEEP/SWAP head conditioned by label embeddings.
That is better than blind KEEP/SWAP, but still mixes different ordered
decision problems.

E4v2 makes the action problem explicit:

  Ransomware -> Spyware
  Spyware    -> Ransomware
  Ransomware -> Trojan
  Trojan     -> Ransomware
  Spyware    -> Trojan
  Trojan     -> Spyware

Each ordered direction has:
  - its own KEEP/SWAP head
  - its own action weights
  - its own threshold
  - optional auto-disable if train sweep shows no safe net gain

No tree.
No teacher.
No full re-classification.
Final prediction is still either official D3 top1 or official D3 top2.

Training label
--------------
For a sample routed to ordered pair top1=A, top2=B:
  true == A => KEEP
  true == B => SWAP
  otherwise ignore for action training.

This directly asks:
  "Given D3 chose A over B, should we trust A or swap to B?"

Outputs
-------
  E4v2_summary.json
  E4v2_history.csv
  E4v2_ordered_pair_thresholds.csv
  E4v2_ordered_pair_sweep_train.csv
  E4v2_ordered_pair_sweep_val.csv
  E4v2_best_per_class_f1.csv
  E4v2_best_confusion_matrix.csv
  E4v2_val_predictions_best.csv
  E4v2_best_pair_fix_damage.csv
  E4v2_action_metrics.json
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


HARD_LABELS = ["Ransomware", "Spyware", "Trojan"]
ORDERED_DIRECTIONS = [
    ("Ransomware", "Spyware"),
    ("Spyware", "Ransomware"),
    ("Ransomware", "Trojan"),
    ("Trojan", "Ransomware"),
    ("Spyware", "Trojan"),
    ("Trojan", "Spyware"),
]
UNORDERED_PAIRS = [
    ("Ransomware", "Spyware"),
    ("Ransomware", "Trojan"),
    ("Spyware", "Trojan"),
]


def strip_label(x: Any) -> str:
    return str(x).strip()


def sanitize_key(a: str, b: str) -> str:
    return f"{a}_TO_{b}".replace("-", "_").replace("/", "_").replace(" ", "_")


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
    spec = importlib.util.spec_from_file_location("official_07_train_for_e4v2", str(train_script))
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


class OrderedPairSwapper(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        tap_module_name: str,
        rep_dim: int,
        num_classes: int,
        ordered_pair_ids: List[Tuple[int, int]],
        ordered_pair_names: List[str],
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

        self.ordered_pair_ids = [(int(a), int(b)) for a, b in ordered_pair_ids]
        self.ordered_pair_names = list(ordered_pair_names)
        self.num_pairs = len(self.ordered_pair_ids)
        self.register_buffer("pair_top1_ids", torch.tensor([a for a, b in self.ordered_pair_ids], dtype=torch.long))
        self.register_buffer("pair_top2_ids", torch.tensor([b for a, b in self.ordered_pair_ids], dtype=torch.long))

        scalar_dim = 9
        in_dim = rep_dim + scalar_dim

        self.shared_norm = nn.LayerNorm(in_dim)
        self.heads = nn.ModuleDict()
        for name in self.ordered_pair_names:
            self.heads[name] = nn.Sequential(
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

        x = torch.cat([rep, scalars], dim=1)
        x = self.shared_norm(x)

        all_logits = torch.stack([self.heads[name](x).squeeze(1) for name in self.ordered_pair_names], dim=1)
        matches = (top1[:, None] == self.pair_top1_ids[None, :]) & (top2[:, None] == self.pair_top2_ids[None, :])
        valid_pair = matches.any(dim=1)
        pair_idx = matches.float().argmax(dim=1)
        selected_logit = all_logits.gather(1, pair_idx[:, None]).squeeze(1)
        selected_logit = torch.where(valid_pair, selected_logit, torch.full_like(selected_logit, -30.0))

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
            "pair_idx": pair_idx,
            "valid_pair": valid_pair,
            "swap_logit": selected_logit,
            "swap_prob": torch.sigmoid(selected_logit),
            "all_pair_logits": all_logits,
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


def supervised_action_mask_torch(y: torch.Tensor, top1: torch.Tensor, top2: torch.Tensor, valid_pair: torch.Tensor):
    keep = valid_pair & (y == top1)
    swap = valid_pair & (y == top2)
    sup = keep | swap
    target = swap.float()
    return sup, target, keep, swap


def supervised_action_mask_np(y: np.ndarray, top1: np.ndarray, top2: np.ndarray, valid_pair: np.ndarray):
    keep = valid_pair & (y == top1)
    swap = valid_pair & (y == top2)
    sup = keep | swap
    target = swap.astype(np.int64)
    return sup, target, keep, swap


def compute_pair_action_weights(arr: dict, num_pairs: int, keep_cost: float, swap_cost: float, min_count: int = 5):
    sup, target, keep, swap = supervised_action_mask_np(arr["y"], arr["top1"], arr["top2"], arr["valid_pair"])
    rows = []
    keep_w = np.ones(num_pairs, dtype=np.float32) * float(keep_cost)
    swap_w = np.ones(num_pairs, dtype=np.float32) * float(swap_cost)
    for i in range(num_pairs):
        m = sup & (arr["pair_idx"] == i)
        n_keep = int((m & (target == 0)).sum())
        n_swap = int((m & (target == 1)).sum())
        n = n_keep + n_swap
        if n_keep >= min_count and n_swap >= min_count:
            kw = (n / (2.0 * n_keep)) * float(keep_cost)
            sw = (n / (2.0 * n_swap)) * float(swap_cost)
            method = "balanced_times_cost"
        else:
            kw = float(keep_cost)
            sw = float(swap_cost)
            method = "fallback_cost_only"
        keep_w[i] = kw
        swap_w[i] = sw
        rows.append({
            "pair_idx": i,
            "n_keep": n_keep,
            "n_swap": n_swap,
            "n_supervised": n,
            "keep_weight": float(kw),
            "swap_weight": float(sw),
            "method": method,
        })
    return keep_w, swap_w, pd.DataFrame(rows)


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
    for a, b in UNORDERED_PAIRS:
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


@torch.no_grad()
def collect_arrays(model, loader, device) -> dict:
    model.eval()
    chunks = {k: [] for k in [
        "y", "top1", "top2", "pair_idx", "valid_pair", "swap_prob", "swap_logit",
        "margin", "top1_prob", "top2_prob", "entropy"
    ]}
    for tokens, values, y in loader:
        tokens = tokens.to(device, non_blocking=True)
        values = values.to(device, non_blocking=True)
        out = model(tokens, values)
        chunks["y"].append(y.cpu().numpy().astype(int))
        for k in ["top1", "top2", "pair_idx"]:
            chunks[k].append(out[k].detach().cpu().numpy().astype(int))
        chunks["valid_pair"].append(out["valid_pair"].detach().cpu().numpy().astype(bool))
        for k in ["swap_prob", "swap_logit", "margin", "top1_prob", "top2_prob", "entropy"]:
            chunks[k].append(out[k].detach().cpu().numpy().astype(float))
    arr = {k: np.concatenate(v, axis=0) for k, v in chunks.items()}
    sup, target, keep, swap = supervised_action_mask_np(arr["y"], arr["top1"], arr["top2"], arr["valid_pair"])
    arr["supervised_action"] = sup
    arr["action_target"] = target
    arr["baseline_correct"] = arr["top1"] == arr["y"]
    arr["top2_correct"] = arr["top2"] == arr["y"]
    return arr


def make_swapped_pred_ordered(arr: dict, thresholds: np.ndarray, margin_cap: Optional[float] = None):
    pred = arr["top1"].copy()
    do_swap = np.zeros_like(pred, dtype=bool)
    for i, th in enumerate(thresholds):
        candidate = arr["valid_pair"] & (arr["pair_idx"] == i)
        m = candidate & (arr["swap_prob"] >= float(th))
        if margin_cap is not None:
            m &= (arr["margin"] <= float(margin_cap))
        do_swap |= m
    pred[do_swap] = arr["top2"][do_swap]
    return pred, do_swap


def sweep_one_ordered_pair(arr: dict, pair_idx: int, thresholds: np.ndarray, margin_cap: Optional[float] = None) -> pd.DataFrame:
    y = arr["y"]
    base = arr["top1"]
    candidate = arr["valid_pair"] & (arr["pair_idx"] == int(pair_idx))
    rows = []
    for th in thresholds:
        pred = base.copy()
        do_swap = candidate & (arr["swap_prob"] >= float(th))
        if margin_cap is not None:
            do_swap &= arr["margin"] <= float(margin_cap)
        pred[do_swap] = arr["top2"][do_swap]
        met = metric_dict(y, pred)
        tr = transition_stats(y, base, pred)
        rows.append({
            "pair_idx": int(pair_idx),
            "threshold": float(th),
            "margin_cap": margin_cap,
            "candidate_n": int(candidate.sum()),
            "swap_n": int(do_swap.sum()),
            "macro_f1": met["macro_f1"],
            "accuracy": met["accuracy"],
            "weighted_f1": met["weighted_f1"],
            **tr,
        })
    return pd.DataFrame(rows)


def ordered_pair_sweep(arr: dict, num_pairs: int, thresholds: np.ndarray, pair_names: List[str], pair_labels: List[str], margin_cap: Optional[float] = None):
    dfs = []
    for i in range(num_pairs):
        df = sweep_one_ordered_pair(arr, i, thresholds, margin_cap)
        df["ordered_pair_name"] = pair_names[i]
        df["ordered_pair_label"] = pair_labels[i]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def select_threshold_for_pair(df_pair: pd.DataFrame, args) -> dict:
    df = df_pair.copy()
    df["damage_ratio_filled"] = df["damage_ratio"].fillna(0.0)
    elig = df.copy()

    if args.threshold_require_net_positive:
        elig = elig[elig["net_gain"] > 0]
    if args.threshold_max_damage_ratio is not None:
        elig = elig[elig["damage_ratio_filled"] <= float(args.threshold_max_damage_ratio)]
    if args.threshold_min_swaps > 0:
        elig = elig[elig["swap_n"] >= int(args.threshold_min_swaps)]

    if len(elig) == 0:
        # Disable unsafe direction.
        best = df.sort_values(["net_gain", "macro_f1"], ascending=False).iloc[0].to_dict()
        best["threshold"] = 1.1
        best["enabled"] = False
        best["selection_reason"] = "disabled_no_safe_positive_train_threshold"
        return best

    obj = args.threshold_objective
    if obj == "macro_f1":
        idx = elig["macro_f1"].idxmax()
    elif obj == "net_gain":
        idx = elig["net_gain"].idxmax()
    elif obj == "damage_aware":
        score = elig["net_gain"] - float(args.damage_penalty) * elig["correct_to_wrong"]
        idx = score.idxmax()
    else:
        raise ValueError(f"unknown threshold objective: {obj}")

    best = df.loc[idx].to_dict()
    best["enabled"] = True
    best["selection_reason"] = f"selected_by_{obj}"
    return best


def select_ordered_thresholds(train_sweep: pd.DataFrame, num_pairs: int, args) -> pd.DataFrame:
    rows = []
    for i in range(num_pairs):
        sel = select_threshold_for_pair(train_sweep[train_sweep["pair_idx"] == i], args)
        rows.append(sel)
    return pd.DataFrame(rows)


def evaluate_ordered_thresholds(arr: dict, thresholds: np.ndarray, label_names: List[str], label_to_id: Dict[str, int], margin_cap: Optional[float] = None):
    pred, do_swap = make_swapped_pred_ordered(arr, thresholds, margin_cap)
    base = arr["top1"]
    y = arr["y"]
    return {
        "metrics": metric_dict(y, pred),
        "base_metrics": metric_dict(y, base),
        "transition": transition_stats(y, base, pred),
        "pred": pred,
        "do_swap": do_swap,
        "per_class": per_class_and_cm(y, pred, label_names)[0],
        "confusion": per_class_and_cm(y, pred, label_names)[1],
        "pair_fix_damage": pair_fix_damage(y, base, pred, label_to_id),
    }


def action_metrics(arr: dict, thresholds: np.ndarray):
    sup, target, keep, swap = supervised_action_mask_np(arr["y"], arr["top1"], arr["top2"], arr["valid_pair"])
    out = {
        "valid_ordered_pair_n": int(arr["valid_pair"].sum()),
        "supervised_action_n": int(sup.sum()),
        "keep_label_n": int((sup & (target == 0)).sum()),
        "swap_label_n": int((sup & (target == 1)).sum()),
        "unfixable_valid_pair_n": int((arr["valid_pair"] & (~sup)).sum()),
    }
    if int(sup.sum()) > 0:
        yt = target[sup].astype(int)
        ps = arr["swap_prob"][sup]
        pred_action = np.zeros_like(yt)
        global_idx = np.where(sup)[0]
        for j, idx in enumerate(global_idx):
            pred_action[j] = int(arr["swap_prob"][idx] >= thresholds[arr["pair_idx"][idx]])
        out.update({
            "action_accuracy": float(accuracy_score(yt, pred_action)),
            "action_macro_f1": float(f1_score(yt, pred_action, average="macro", zero_division=0)),
            "predicted_swap_n": int(pred_action.sum()),
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


def train_one_epoch(model, loader, optimizer, device, args, keep_weights_t: torch.Tensor, swap_weights_t: torch.Tensor):
    model.train()
    total_loss = 0.0
    total_sup = 0
    used_batches = 0
    skipped_batches = 0

    for tokens, values, y in loader:
        tokens = tokens.to(device, non_blocking=True)
        values = values.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)
        out = model(tokens, values)
        sup, target, keep, swap = supervised_action_mask_torch(y, out["top1"], out["top2"], out["valid_pair"])

        if int(sup.sum().item()) == 0:
            skipped_batches += 1
            continue

        logits = out["swap_logit"][sup]
        target_sup = target[sup]
        pair_idx_sup = out["pair_idx"][sup]

        kw = keep_weights_t[pair_idx_sup]
        sw = swap_weights_t[pair_idx_sup]
        weights = torch.where(target_sup > 0.5, sw, kw)

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
        used_batches += 1
        total_loss += float(loss.item()) * n

    return {
        "train_action_loss": float(total_loss / max(1, total_sup)),
        "train_supervised_action_n": int(total_sup),
        "train_used_batches": int(used_batches),
        "train_skipped_batches": int(skipped_batches),
    }


def epoch_eval(model, val_loader, device):
    arr = collect_arrays(model, val_loader, device)
    sup = arr["supervised_action"]
    out = {
        "val_supervised_action_n": int(sup.sum()),
        "val_base_macro_f1": float(f1_score(arr["y"], arr["top1"], average="macro")),
    }
    if int(sup.sum()) > 0:
        yt = arr["action_target"][sup]
        ps = arr["swap_prob"][sup]
        try:
            out["val_action_auc"] = float(roc_auc_score(yt, ps))
        except Exception:
            out["val_action_auc"] = None
        try:
            out["val_action_ap"] = float(average_precision_score(yt, ps))
        except Exception:
            out["val_action_ap"] = None
    return out


def save_outputs(out_dir, inp, args, model, train_eval_loader, val_loader, ordered_pair_names, ordered_pair_labels, ckpt_info, rep_dim, tap_module_name, history, best_epoch, device):
    thresholds = np.linspace(float(args.threshold_min), float(args.threshold_max), int(args.threshold_steps))
    train_arr = collect_arrays(model, train_eval_loader, device)
    val_arr = collect_arrays(model, val_loader, device)

    train_sweep = ordered_pair_sweep(train_arr, len(ordered_pair_names), thresholds, ordered_pair_names, ordered_pair_labels, args.margin_cap)
    val_sweep = ordered_pair_sweep(val_arr, len(ordered_pair_names), thresholds, ordered_pair_names, ordered_pair_labels, args.margin_cap)
    train_sweep.to_csv(out_dir / "E4v2_ordered_pair_sweep_train.csv", index=False)
    val_sweep.to_csv(out_dir / "E4v2_ordered_pair_sweep_val.csv", index=False)

    if args.mode == "eval_zero":
        th_df = pd.DataFrame([{
            "pair_idx": i,
            "ordered_pair_name": ordered_pair_names[i],
            "ordered_pair_label": ordered_pair_labels[i],
            "threshold": 1.1,
            "enabled": False,
            "selection_reason": "eval_zero_no_swaps",
        } for i in range(len(ordered_pair_names))])
    else:
        th_df = select_ordered_thresholds(train_sweep, len(ordered_pair_names), args)
        # Make sure names are attached even when disabled dict changed threshold.
        for i in range(len(ordered_pair_names)):
            th_df.loc[th_df["pair_idx"].astype(int) == i, "ordered_pair_name"] = ordered_pair_names[i]
            th_df.loc[th_df["pair_idx"].astype(int) == i, "ordered_pair_label"] = ordered_pair_labels[i]

    th_df = th_df.sort_values("pair_idx").reset_index(drop=True)
    th_df.to_csv(out_dir / "E4v2_ordered_pair_thresholds.csv", index=False)
    selected_thresholds = th_df["threshold"].astype(float).to_numpy()

    train_eval = evaluate_ordered_thresholds(train_arr, selected_thresholds, inp["label_names"], inp["label_to_id"], args.margin_cap)
    val_eval = evaluate_ordered_thresholds(val_arr, selected_thresholds, inp["label_names"], inp["label_to_id"], args.margin_cap)

    val_eval["per_class"].to_csv(out_dir / "E4v2_best_per_class_f1.csv", index=False)
    val_eval["confusion"].to_csv(out_dir / "E4v2_best_confusion_matrix.csv")
    val_eval["pair_fix_damage"].to_csv(out_dir / "E4v2_best_pair_fix_damage.csv", index=False)

    base_per, base_cm = per_class_and_cm(val_arr["y"], val_arr["top1"], inp["label_names"])
    base_per.to_csv(out_dir / "E4v2_baseline_per_class_f1.csv", index=False)
    base_cm.to_csv(out_dir / "E4v2_baseline_confusion_matrix.csv")

    action_train = action_metrics(train_arr, selected_thresholds)
    action_val = action_metrics(val_arr, selected_thresholds)
    save_json(out_dir / "E4v2_action_metrics.json", {"train": action_train, "val": action_val})

    pred = val_eval["pred"]
    do_swap = val_eval["do_swap"]
    pred_df = pd.DataFrame({
        "sample_index": np.arange(len(val_arr["y"]), dtype=int),
        "true_id": val_arr["y"].astype(int),
        "true_label": [inp["id_to_label"][int(i)] for i in val_arr["y"]],
        "base_top1_id": val_arr["top1"].astype(int),
        "base_top1_label": [inp["id_to_label"][int(i)] for i in val_arr["top1"]],
        "base_top2_id": val_arr["top2"].astype(int),
        "base_top2_label": [inp["id_to_label"][int(i)] for i in val_arr["top2"]],
        "ordered_pair_idx": val_arr["pair_idx"].astype(int),
        "ordered_pair_label": [ordered_pair_labels[int(i)] if v else "" for i, v in zip(val_arr["pair_idx"], val_arr["valid_pair"])],
        "valid_ordered_pair": val_arr["valid_pair"],
        "supervised_action": val_arr["supervised_action"],
        "action_target_swap": val_arr["action_target"].astype(int),
        "swap_prob": val_arr["swap_prob"],
        "swap_logit": val_arr["swap_logit"],
        "selected_threshold_for_pair": [selected_thresholds[int(i)] if v else np.nan for i, v in zip(val_arr["pair_idx"], val_arr["valid_pair"])],
        "margin": val_arr["margin"],
        "top1_prob": val_arr["top1_prob"],
        "top2_prob": val_arr["top2_prob"],
        "entropy": val_arr["entropy"],
        "base_correct": val_arr["top1"] == val_arr["y"],
        "top2_correct": val_arr["top2"] == val_arr["y"],
        "do_swap": do_swap,
        "e4v2_pred_id": pred.astype(int),
        "e4v2_pred_label": [inp["id_to_label"][int(i)] for i in pred],
        "e4v2_correct": pred == val_arr["y"],
    })
    pred_df["transition"] = "both_wrong"
    pred_df.loc[pred_df["base_correct"] & pred_df["e4v2_correct"], "transition"] = "both_correct"
    pred_df.loc[(~pred_df["base_correct"]) & pred_df["e4v2_correct"], "transition"] = "fixed"
    pred_df.loc[pred_df["base_correct"] & (~pred_df["e4v2_correct"]), "transition"] = "damaged"
    pred_df.to_csv(out_dir / "E4v2_val_predictions_best.csv", index=False)

    pd.DataFrame(history).to_csv(out_dir / "E4v2_history.csv", index=False)

    summary = {
        "stage": "E4v2_ordered_pair_top2_swapper",
        "mode": args.mode,
        "research_position": "attention-only residual KEEP/SWAP with ordered-pair-specific heads and thresholds",
        "tree_usage": "none",
        "fix_over_E4": "E4v2 separates the six ordered top1->top2 decisions instead of using one shared swapper head.",
        "references": {
            "E4a_damage_aware_macro_f1": 0.810348,
            "E4b_recall_macro_f1": 0.808011,
            "E3_best_no_tree_macro_f1": 0.813577,
            "E2a_tree_distill_macro_f1": 0.817847,
            "E1b_tree_expert_macro_f1": 0.829387,
        },
        "best_epoch": int(best_epoch),
        "base_val_metrics": val_eval["base_metrics"],
        "final_val_metrics": val_eval["metrics"],
        "delta_val": {
            "macro_f1": val_eval["metrics"]["macro_f1"] - val_eval["base_metrics"]["macro_f1"],
            "accuracy": val_eval["metrics"]["accuracy"] - val_eval["base_metrics"]["accuracy"],
            "weighted_f1": val_eval["metrics"]["weighted_f1"] - val_eval["base_metrics"]["weighted_f1"],
        },
        "transition_val_vs_base": val_eval["transition"],
        "base_train_metrics": train_eval["base_metrics"],
        "final_train_metrics": train_eval["metrics"],
        "transition_train_vs_base": train_eval["transition"],
        "action_metrics": {"train": action_train, "val": action_val},
        "ordered_pair_thresholds": th_df.to_dict(orient="records"),
        "checkpoint_load": ckpt_info,
        "tap_module_name": tap_module_name,
        "rep_dim": int(rep_dim),
        "loss_design": {
            "train_label": "KEEP if true==top1, SWAP if true==top2, ignore if true not in top2",
            "ordered_pair_specific_heads": True,
            "ordered_pair_specific_thresholds": True,
            "freeze_backbone": bool(args.freeze_backbone),
            "keep_cost": float(args.keep_cost),
            "swap_cost": float(args.swap_cost),
            "threshold_objective": args.threshold_objective,
            "threshold_max_damage_ratio": args.threshold_max_damage_ratio,
            "disable_unsafe_direction": True,
        },
        "outputs": {
            "history": str(out_dir / "E4v2_history.csv"),
            "best_model": str(out_dir / "E4v2_best_model.pt"),
            "thresholds": str(out_dir / "E4v2_ordered_pair_thresholds.csv"),
            "sweep_train": str(out_dir / "E4v2_ordered_pair_sweep_train.csv"),
            "sweep_val": str(out_dir / "E4v2_ordered_pair_sweep_val.csv"),
            "per_class": str(out_dir / "E4v2_best_per_class_f1.csv"),
            "confusion_matrix": str(out_dir / "E4v2_best_confusion_matrix.csv"),
            "val_predictions": str(out_dir / "E4v2_val_predictions_best.csv"),
            "pair_fix_damage": str(out_dir / "E4v2_best_pair_fix_damage.csv"),
            "action_metrics": str(out_dir / "E4v2_action_metrics.json"),
        },
    }
    save_json(out_dir / "E4v2_summary.json", summary)
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
    text = f"""# E4v2 Ordered-pair-specific Top-2 Swapper

## Position

```text
attention-only
no tree in training
no tree in inference
ordered-pair-specific KEEP/SWAP
```

## Fix over E4

E4 had one shared swapper. E4v2 has separate heads and thresholds for:

```text
Ransomware -> Spyware
Spyware -> Ransomware
Ransomware -> Trojan
Trojan -> Ransomware
Spyware -> Trojan
Trojan -> Spyware
```

## Validation result

```text
base macro-F1  = {summary['base_val_metrics']['macro_f1']:.6f}
E4v2 macro-F1  = {summary['final_val_metrics']['macro_f1']:.6f}
delta macro-F1 = {summary['delta_val']['macro_f1']:+.6f}

base accuracy  = {summary['base_val_metrics']['accuracy']:.6f}
E4v2 accuracy  = {summary['final_val_metrics']['accuracy']:.6f}
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
val valid_ordered_pair_n = {summary['action_metrics']['val'].get('valid_ordered_pair_n')}
val supervised_action_n  = {summary['action_metrics']['val'].get('supervised_action_n')}
val keep_label_n         = {summary['action_metrics']['val'].get('keep_label_n')}
val swap_label_n         = {summary['action_metrics']['val'].get('swap_label_n')}
val action_auc           = {summary['action_metrics']['val'].get('action_auc')}
val action_ap            = {summary['action_metrics']['val'].get('action_ap')}
```

## Key files

- `E4v2_summary.json`
- `E4v2_history.csv`
- `E4v2_ordered_pair_thresholds.csv`
- `E4v2_ordered_pair_sweep_train.csv`
- `E4v2_ordered_pair_sweep_val.csv`
- `E4v2_best_per_class_f1.csv`
- `E4v2_best_confusion_matrix.csv`
- `E4v2_val_predictions_best.csv`
- `E4v2_best_pair_fix_damage.csv`
- `E4v2_action_metrics.json`
"""
    (out_dir / "E4v2_summary.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="E4v2 ordered-pair-specific top2 residual swapper")
    parser.add_argument("--dataset-npz", default="03_outputs/05_dataset/dataset.npz")
    parser.add_argument("--metadata-json", default="03_outputs/05_dataset/metadata.json")
    parser.add_argument("--train-raw", default="01_split/train_raw.csv")
    parser.add_argument("--val-raw", default="01_split/val_raw.csv")
    parser.add_argument("--official-train", default="02_src/07_train.py")
    parser.add_argument("--baseline-config", default="03_outputs/06_model/config.json")
    parser.add_argument("--baseline-checkpoint", default="03_outputs/06_model/best_model.pt")
    parser.add_argument("--baseline-pred", default="03_outputs/06_model/val_predictions_best.csv")
    parser.add_argument("--out-dir", default="05_test/outputs/E4v2_ordered_pair_swapper")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--mode", default="eval_zero", choices=["eval_zero", "train_ordered_swapper"])

    parser.add_argument("--tap-module", default="auto")
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

    parser.add_argument("--keep-cost", type=float, default=2.0)
    parser.add_argument("--swap-cost", type=float, default=1.0)
    parser.add_argument("--min-action-count-per-direction", type=int, default=5)

    parser.add_argument("--threshold-objective", default="damage_aware", choices=["macro_f1", "net_gain", "damage_aware"])
    parser.add_argument("--damage-penalty", type=float, default=1.5)
    parser.add_argument("--threshold-max-damage-ratio", type=float, default=0.75)
    parser.add_argument("--threshold-require-net-positive", action="store_true", default=True)
    parser.add_argument("--threshold-min-swaps", type=int, default=1)
    parser.add_argument("--threshold-min", type=float, default=0.01)
    parser.add_argument("--threshold-max", type=float, default=0.99)
    parser.add_argument("--threshold-steps", type=int, default=99)
    parser.add_argument("--margin-cap", type=float, default=None)

    parser.add_argument("--selection-metric", default="val_action_auc", choices=["val_action_auc", "val_action_ap", "neg_train_loss"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--make-zip", action="store_true", default=True)
    parser.add_argument("--no-zip", dest="make_zip", action="store_false")
    args = parser.parse_args()

    if args.mode == "eval_zero":
        args.epochs = 0
        args.threshold_min = 1.1
        args.threshold_max = 1.1
        args.threshold_steps = 1

    repo_root = repo_root_from_here()
    out_dir = resolve_path(args.out_dir, repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(args.seed))
    device = pick_device(args.device)

    print(f"[E4v2] repo_root={repo_root}", flush=True)
    print(f"[E4v2] out_dir={out_dir}", flush=True)
    print(f"[E4v2] mode={args.mode}", flush=True)
    print(f"[E4v2] device={device}", flush=True)
    print("[E4v2] no tree. ordered-pair-specific residual action.", flush=True)

    inp = load_official_inputs(args, repo_root)

    ordered_pair_ids = [(int(inp["label_to_id"][a]), int(inp["label_to_id"][b])) for a, b in ORDERED_DIRECTIONS]
    ordered_pair_names = [sanitize_key(a, b) for a, b in ORDERED_DIRECTIONS]
    ordered_pair_labels = [f"{a}->{b}" for a, b in ORDERED_DIRECTIONS]

    train_loader = make_loader(inp["train_ds"], int(args.batch_size), True, int(args.seed), int(args.num_workers), device)
    train_eval_loader = make_loader(inp["train_ds"], int(args.batch_size), False, int(args.seed), int(args.num_workers), device)
    val_loader = make_loader(inp["val_ds"], int(args.batch_size), False, int(args.seed), int(args.num_workers), device)

    model_cfg = get_baseline_model_cfg(args, repo_root)
    backbone, backbone_kwargs = build_official_d3_model(
        inp["train_mod"], model_cfg, len(inp["feature_names"]), inp["num_bins"], len(inp["label_names"]), device
    )
    ckpt_info = load_checkpoint_into_model(backbone, resolve_path(args.baseline_checkpoint, repo_root), device, strict=True)
    print(f"[E4v2] checkpoint loaded: {ckpt_info}", flush=True)

    rep_dim, tap_module_name = infer_rep_dim(backbone, val_loader, device, args.tap_module)
    print(f"[E4v2] tap_module={tap_module_name} rep_dim={rep_dim}", flush=True)

    model = OrderedPairSwapper(
        backbone=backbone,
        tap_module_name=tap_module_name,
        rep_dim=rep_dim,
        num_classes=len(inp["label_names"]),
        ordered_pair_ids=ordered_pair_ids,
        ordered_pair_names=ordered_pair_names,
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.head_dropout),
        detach_backbone=bool(args.freeze_backbone),
    ).to(device)

    if args.freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False

    init_train_arr = collect_arrays(model, train_eval_loader, device)
    keep_w, swap_w, weight_df = compute_pair_action_weights(
        init_train_arr,
        len(ordered_pair_names),
        keep_cost=float(args.keep_cost),
        swap_cost=float(args.swap_cost),
        min_count=int(args.min_action_count_per_direction),
    )
    weight_df["ordered_pair_name"] = ordered_pair_names
    weight_df["ordered_pair_label"] = ordered_pair_labels
    weight_df.to_csv(out_dir / "E4v2_ordered_pair_action_weights.csv", index=False)
    print("[E4v2] ordered action weights:", weight_df[["ordered_pair_label","n_keep","n_swap","keep_weight","swap_weight","method"]].to_dict(orient="records"), flush=True)

    keep_w_t = torch.tensor(keep_w, dtype=torch.float32, device=device)
    swap_w_t = torch.tensor(swap_w, dtype=torch.float32, device=device)

    save_json(out_dir / "E4v2_run_config.json", {
        "stage": "E4v2_ordered_pair_top2_swapper",
        "mode": args.mode,
        "tree_usage": "none",
        "args": vars(args),
        "device": str(device),
        "label_names": inp["label_names"],
        "ordered_pair_ids": ordered_pair_ids,
        "ordered_pair_names": ordered_pair_names,
        "ordered_pair_labels": ordered_pair_labels,
        "num_bins": int(inp["num_bins"]),
        "continuous_info": inp["continuous_info"],
        "backbone_kwargs": backbone_kwargs,
        "checkpoint_load": ckpt_info,
        "tap_module_name": tap_module_name,
        "rep_dim": int(rep_dim),
        "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "total_params": int(sum(p.numel() for p in model.parameters())),
    })

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(args.lr), weight_decay=float(args.weight_decay))

    history = []
    best_score = -1e18
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0

    if args.mode == "eval_zero":
        print("[E4v2] eval_zero: no training, all thresholds disabled.", flush=True)
    else:
        print(
            f"[E4v2] training trainable_params={sum(p.numel() for p in model.parameters() if p.requires_grad):,} "
            f"total_params={sum(p.numel() for p in model.parameters()):,}",
            flush=True,
        )

    for epoch in range(1, int(args.epochs) + 1):
        lr_epoch = compute_lr(epoch, args)
        set_optimizer_lr(optimizer, lr_epoch)
        t0 = time.time()
        train_met = train_one_epoch(model, train_loader, optimizer, device, args, keep_w_t, swap_w_t)
        val_met = epoch_eval(model, val_loader, device)
        dt = time.time() - t0

        if args.selection_metric == "val_action_auc":
            score = val_met.get("val_action_auc")
        elif args.selection_metric == "val_action_ap":
            score = val_met.get("val_action_ap")
        else:
            score = -float(train_met["train_action_loss"])
        if score is None or (isinstance(score, float) and np.isnan(score)):
            score = -1e18

        row = {
            "epoch": int(epoch),
            "lr": float(lr_epoch),
            "seconds": float(dt),
            "selection_score": float(score),
            **train_met,
            **val_met,
        }
        history.append(row)

        improved = float(score) > best_score + float(args.min_delta)
        if improved:
            best_score = float(score)
            best_epoch = int(epoch)
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            torch.save({
                "model_state_dict": best_state,
                "epoch": int(epoch),
                "selection_score": float(score),
                "args": vars(args),
                "ordered_pair_ids": ordered_pair_ids,
                "ordered_pair_names": ordered_pair_names,
                "ordered_pair_labels": ordered_pair_labels,
                "label_names": inp["label_names"],
                "backbone_kwargs": backbone_kwargs,
                "tap_module_name": tap_module_name,
                "rep_dim": int(rep_dim),
                "tree_usage": "none",
            }, out_dir / "E4v2_best_model.pt")
        else:
            no_improve += 1

        if epoch == 1 or improved or epoch % int(args.log_every) == 0:
            print(
                f"[E4v2] ep={epoch:03d} lr={lr_epoch:.3e} "
                f"loss={train_met['train_action_loss']:.5f} sup_n={train_met['train_supervised_action_n']} "
                f"val_auc={val_met.get('val_action_auc')} val_ap={val_met.get('val_action_ap')} "
                f"best_score={best_score:.6f}@{best_epoch} noimp={no_improve} sec={dt:.1f}",
                flush=True,
            )

        if no_improve >= int(args.patience):
            print(f"[E4v2] early stop at epoch {epoch}", flush=True)
            break

    model.load_state_dict(best_state)
    if args.mode == "eval_zero":
        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": 0,
            "selection_score": 0.0,
            "args": vars(args),
            "ordered_pair_ids": ordered_pair_ids,
            "ordered_pair_names": ordered_pair_names,
            "ordered_pair_labels": ordered_pair_labels,
            "label_names": inp["label_names"],
            "backbone_kwargs": backbone_kwargs,
            "tap_module_name": tap_module_name,
            "rep_dim": int(rep_dim),
            "tree_usage": "none",
        }, out_dir / "E4v2_best_model.pt")
        best_epoch = 0
        history.append({"epoch": 0, "selection_score": 0.0})

    save_outputs(out_dir, inp, args, model, train_eval_loader, val_loader, ordered_pair_names, ordered_pair_labels, ckpt_info, rep_dim, tap_module_name, history, best_epoch, device)

    if args.make_zip:
        zip_path = out_dir.with_suffix(".zip")
        zip_dir(out_dir, zip_path)
        print(f"[E4v2] zipped outputs: {zip_path}", flush=True)

    summary = load_json(out_dir / "E4v2_summary.json")
    print("[E4v2] done.", flush=True)
    print(f"[E4v2] mode={args.mode}", flush=True)
    print(f"[E4v2] base_macro_f1={summary['base_val_metrics']['macro_f1']:.6f}", flush=True)
    print(f"[E4v2] macro_f1={summary['final_val_metrics']['macro_f1']:.6f}", flush=True)
    print(f"[E4v2] net_gain={summary['transition_val_vs_base']['net_gain']}", flush=True)


if __name__ == "__main__":
    main()
