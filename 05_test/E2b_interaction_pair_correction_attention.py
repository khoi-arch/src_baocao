#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E2b Interaction-token Pair-correction Attention.

Research goal
-------------
Stay attention-based.

E1b tree experts showed that subtype signal exists in pairwise/interacting
tabular features. E2a tree-guided distillation transferred only part of that
signal into D3 attention.

E2b moves the signal inside the attention model:
  official D3 feature tokens
  + selected interaction tokens
  + pair-aware attention correction logits
  -> final 4-class logits

No tree is used in final inference.

Model
-----
Input tokens:
  Base feature tokens:
    [bin_norm, offset_norm, raw_scaled, mask]
  Interaction tokens:
    [interaction_value, 0, interaction_value, 1]

Transformer:
  CLS + all tokens -> TransformerEncoder

Heads:
  main logits from CLS: 4 classes
  pair query attention heads: RS, RT, ST
  pair logits are converted to correction deltas and added to main logits

Loss:
  CE(final_logits, y)
  + main_aux_weight * CE(main_logits, y)
  + pair_aux_weight * CE(pair_logits, pair_label) for true pair samples

Default output:
  05_test/outputs/E2b_interaction_pair_correction_attention/
  05_test/outputs/E2b_interaction_pair_correction_attention.zip
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
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

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
PAIR_CLASS_IDS = {}  # filled after loading label mapping


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
    spec = importlib.util.spec_from_file_location("official_07_train_for_e2b", str(train_script))
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

    num_bins = int(meta.get("num_bins", 0) or meta.get("K", 0) or (max(int(X_train_bin.max()), int(X_val_bin.max())) + 1))
    denom = max(1, num_bins - 1)

    X_train_bin_norm = X_train_bin.astype(np.float32) / float(denom)
    X_val_bin_norm = X_val_bin.astype(np.float32) / float(denom)
    X_train_offset_norm = X_train_offset.astype(np.float32) / float(denom)
    X_val_offset_norm = X_val_offset.astype(np.float32) / float(denom)

    M_train = np.ones_like(X_train_bin_norm, dtype=np.float32)
    M_val = np.ones_like(X_val_bin_norm, dtype=np.float32)

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
        "X_val_bin": X_val_bin,
        "X_train_bin_norm": X_train_bin_norm,
        "X_val_bin_norm": X_val_bin_norm,
        "X_train_offset_norm": X_train_offset_norm,
        "X_val_offset_norm": X_val_offset_norm,
        "X_train_cont": X_train_cont.astype(np.float32),
        "X_val_cont": X_val_cont.astype(np.float32),
        "M_train": M_train,
        "M_val": M_val,
        "y_train": y_train,
        "y_val": y_val,
    }


@dataclass
class InteractionSpec:
    name: str
    pair_key: str
    op: str
    feat_a: str
    feat_b: str
    idx_a: int
    idx_b: int
    found_a: str
    found_b: str


DEFAULT_INTERACTIONS = [
    # From E0: best RS interaction.
    ("RS_absdiff_psxview_not_in_csrss_handles_false_avg__pslist_nproc", "RS", "absdiff", "psxview.not_in_csrss_handles_false_avg", "pslist.nproc"),
    ("RS_ratio_psxview_not_in_csrss_handles_false_avg__pslist_nproc", "RS", "ratio", "psxview.not_in_csrss_handles_false_avg", "pslist.nproc"),
    # From E0: strongest RT interaction.
    ("RT_ratio_handles_nfile__ldrmodules_not_in_mem", "RT", "ratio", "handles.nfile", "ldrmodules.not_in_mem"),
    ("RT_absdiff_handles_nfile__ldrmodules_not_in_mem", "RT", "absdiff", "handles.nfile", "ldrmodules.not_in_mem"),
    # From E0: strongest ST interaction.
    ("ST_absdiff_handles_nfile__malfind_ninjections", "ST", "absdiff", "handles.nfile", "malfind.ninjections"),
    ("ST_ratio_handles_nfile__malfind_ninjections", "ST", "ratio", "handles.nfile", "malfind.ninjections"),
    # Additional simple interactions around repeated attention/audit features.
    ("ALL_product_callbacks_ngeneric__pslist_avg_threads", "ALL", "product", "callbacks.ngeneric", "pslist.avg_threads"),
    ("ALL_absdiff_svcscan_interactive_process_services__handles_nevent", "ALL", "absdiff", "svcscan.interactive_process_services", "handles.nevent"),
]


def normalize_feat_name(s: str) -> str:
    return str(s).lower().replace("_", "").replace(".", "").replace("-", "").replace(" ", "")


def resolve_feature(feature_names: List[str], query: str) -> Tuple[Optional[int], Optional[str]]:
    q = str(query).strip()
    if q in feature_names:
        return feature_names.index(q), q

    qn = normalize_feat_name(q)
    normalized = [normalize_feat_name(x) for x in feature_names]

    # exact normalized
    for i, fn in enumerate(normalized):
        if fn == qn:
            return i, feature_names[i]

    # suffix normalized
    for i, fn in enumerate(normalized):
        if fn.endswith(qn) or qn.endswith(fn):
            return i, feature_names[i]

    # contains normalized
    matches = [(i, feature_names[i]) for i, fn in enumerate(normalized) if qn in fn or fn in qn]
    if matches:
        return matches[0]

    return None, None


def parse_interactions(args, feature_names: List[str]) -> List[InteractionSpec]:
    raw = []
    if args.interaction_specs and Path(args.interaction_specs).exists():
        df = pd.read_csv(args.interaction_specs)
        for r in df.itertuples(index=False):
            raw.append((str(r.name), str(r.pair_key), str(r.op), str(r.feat_a), str(r.feat_b)))
    elif args.interaction_specs:
        # Format:
        # name|pair|op|feat_a|feat_b;name|pair|op|feat_a|feat_b
        chunks = [c.strip() for c in str(args.interaction_specs).split(";") if c.strip()]
        for c in chunks:
            parts = [p.strip() for p in c.split("|")]
            if len(parts) != 5:
                raise ValueError(f"Bad interaction spec: {c}")
            raw.append(tuple(parts))
    else:
        raw = list(DEFAULT_INTERACTIONS)

    specs: List[InteractionSpec] = []
    skipped = []
    for name, pair_key, op, fa, fb in raw:
        ia, fa_found = resolve_feature(feature_names, fa)
        ib, fb_found = resolve_feature(feature_names, fb)
        if ia is None or ib is None:
            skipped.append({
                "name": name, "pair_key": pair_key, "op": op, "feat_a": fa, "feat_b": fb,
                "found_a": fa_found, "found_b": fb_found,
            })
            continue
        specs.append(InteractionSpec(
            name=str(name),
            pair_key=str(pair_key).upper(),
            op=str(op).lower(),
            feat_a=str(fa),
            feat_b=str(fb),
            idx_a=int(ia),
            idx_b=int(ib),
            found_a=str(fa_found),
            found_b=str(fb_found),
        ))
    if skipped:
        print(f"[E2b] skipped {len(skipped)} interaction specs because features were missing:", flush=True)
        for s in skipped:
            print(f"  - {s}", flush=True)
    if not specs:
        raise RuntimeError("No valid interaction specs resolved. Check feature names.")
    return specs


def compute_raw_interaction(X: np.ndarray, spec: InteractionSpec, eps: float = 1e-3) -> np.ndarray:
    a = X[:, spec.idx_a].astype(np.float32)
    b = X[:, spec.idx_b].astype(np.float32)
    op = spec.op
    if op == "absdiff":
        v = np.abs(a - b)
    elif op == "ratio":
        v = a / (np.abs(b) + eps)
    elif op == "logratio":
        v = np.log1p(np.abs(a)) - np.log1p(np.abs(b))
    elif op == "product":
        v = a * b
    elif op == "sum":
        v = 0.5 * (a + b)
    elif op == "max":
        v = np.maximum(a, b)
    elif op == "min":
        v = np.minimum(a, b)
    else:
        raise ValueError(f"Unknown interaction op {op}")
    return v.astype(np.float32)


def build_interaction_matrix(inp: dict, specs: List[InteractionSpec], args, out_dir: Path):
    train_vals = []
    val_vals = []
    rows = []
    for spec in specs:
        tr_raw = compute_raw_interaction(inp["X_train_cont"], spec, eps=float(args.ratio_eps))
        va_raw = compute_raw_interaction(inp["X_val_cont"], spec, eps=float(args.ratio_eps))

        lo = float(np.quantile(tr_raw[np.isfinite(tr_raw)], float(args.interaction_clip_low)))
        hi = float(np.quantile(tr_raw[np.isfinite(tr_raw)], float(args.interaction_clip_high)))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(np.nanmin(tr_raw))
            hi = float(np.nanmax(tr_raw))
        if hi <= lo:
            tr = np.zeros_like(tr_raw, dtype=np.float32)
            va = np.zeros_like(va_raw, dtype=np.float32)
        else:
            tr = np.clip(tr_raw, lo, hi)
            va = np.clip(va_raw, lo, hi)
            tr = (tr - lo) / (hi - lo)
            va = (va - lo) / (hi - lo)
            tr = np.clip(tr, 0.0, 1.0)
            va = np.clip(va, 0.0, 1.0)

        train_vals.append(tr.astype(np.float32))
        val_vals.append(va.astype(np.float32))
        rows.append({
            "interaction_name": spec.name,
            "pair_key": spec.pair_key,
            "op": spec.op,
            "feat_a_query": spec.feat_a,
            "feat_b_query": spec.feat_b,
            "feat_a_found": spec.found_a,
            "feat_b_found": spec.found_b,
            "idx_a": spec.idx_a,
            "idx_b": spec.idx_b,
            "clip_low": lo,
            "clip_high": hi,
            "train_mean": float(np.mean(tr)),
            "train_std": float(np.std(tr)),
            "val_mean": float(np.mean(va)),
            "val_std": float(np.std(va)),
        })
    Xtr = np.stack(train_vals, axis=1).astype(np.float32)
    Xva = np.stack(val_vals, axis=1).astype(np.float32)
    meta = pd.DataFrame(rows)
    meta.to_csv(out_dir / "E2b_interaction_specs.csv", index=False)
    np.save(out_dir / "E2b_X_train_interactions.npy", Xtr)
    np.save(out_dir / "E2b_X_val_interactions.npy", Xva)
    return Xtr, Xva, meta


class E2bDataset(Dataset):
    def __init__(self, bin_norm, offset_norm, cont, mask, interactions, y):
        self.bin_norm = np.asarray(bin_norm, dtype=np.float32)
        self.offset_norm = np.asarray(offset_norm, dtype=np.float32)
        self.cont = np.asarray(cont, dtype=np.float32)
        self.mask = np.asarray(mask, dtype=np.float32)
        self.interactions = np.asarray(interactions, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        base = np.stack([
            self.bin_norm[idx],
            self.offset_norm[idx],
            self.cont[idx],
            self.mask[idx],
        ], axis=1).astype(np.float32)  # [F,4]

        inter_v = self.interactions[idx].astype(np.float32)
        if inter_v.ndim == 0:
            inter_v = inter_v[None]
        inter = np.stack([
            inter_v,
            np.zeros_like(inter_v, dtype=np.float32),
            inter_v,
            np.ones_like(inter_v, dtype=np.float32),
        ], axis=1).astype(np.float32)  # [I,4]

        token_values = np.concatenate([base, inter], axis=0)
        return torch.from_numpy(token_values), torch.tensor(int(self.y[idx]), dtype=torch.long)


class E2bInteractionPairCorrectionTransformer(nn.Module):
    def __init__(
        self,
        n_base_features: int,
        n_interactions: int,
        num_classes: int,
        pair_class_ids: Dict[str, Tuple[int, int]],
        d_model: int = 128,
        value_hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        classifier_hidden_dim: int = 128,
        classifier_dropout: float = 0.1,
        norm_first: bool = True,
        correction_scale_init: float = 0.25,
        learnable_correction_scale: bool = True,
    ):
        super().__init__()
        self.n_base_features = int(n_base_features)
        self.n_interactions = int(n_interactions)
        self.n_tokens = self.n_base_features + self.n_interactions
        self.num_classes = int(num_classes)
        self.pair_keys = ["RS", "RT", "ST"]
        self.pair_class_ids = {k: tuple(v) for k, v in pair_class_ids.items()}

        self.value_encoder = nn.Sequential(
            nn.Linear(4, value_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(value_hidden_dim, d_model),
        )
        self.token_type_embedding = nn.Embedding(2, d_model)
        self.feature_embedding = nn.Embedding(self.n_tokens, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        self.main_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden_dim, num_classes),
        )

        self.pair_queries = nn.Parameter(torch.randn(len(self.pair_keys), d_model) * 0.02)
        self.pair_heads = nn.ModuleDict({
            pk: nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, classifier_hidden_dim),
                nn.GELU(),
                nn.Dropout(classifier_dropout),
                nn.Linear(classifier_hidden_dim, 2),
            ) for pk in self.pair_keys
        })

        if learnable_correction_scale:
            self.correction_scale = nn.Parameter(torch.tensor(float(correction_scale_init), dtype=torch.float32))
        else:
            self.register_buffer("correction_scale", torch.tensor(float(correction_scale_init), dtype=torch.float32))

        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, token_values: torch.Tensor):
        # token_values: [B, F+I, 4]
        bsz, n_tokens, _ = token_values.shape
        if n_tokens != self.n_tokens:
            raise ValueError(f"Expected {self.n_tokens} tokens, got {n_tokens}")

        x = self.value_encoder(token_values)

        ids = torch.arange(self.n_tokens, device=token_values.device).unsqueeze(0).expand(bsz, -1)
        type_ids = torch.zeros(self.n_tokens, device=token_values.device, dtype=torch.long)
        if self.n_interactions > 0:
            type_ids[self.n_base_features:] = 1
        type_ids = type_ids.unsqueeze(0).expand(bsz, -1)

        x = x + self.feature_embedding(ids) + self.token_type_embedding(type_ids)

        cls = self.cls_token.expand(bsz, -1, -1)
        h = torch.cat([cls, x], dim=1)
        h = self.encoder(h)
        h = self.norm(h)

        cls_out = h[:, 0]
        token_out = h[:, 1:]

        main_logits = self.main_head(cls_out)
        correction = torch.zeros_like(main_logits)
        pair_logits = {}
        pair_attn = {}

        for qi, pk in enumerate(self.pair_keys):
            q = self.pair_queries[qi].view(1, 1, -1)
            score = (token_out * q).sum(dim=-1) / math.sqrt(token_out.shape[-1])
            attn = torch.softmax(score, dim=1)
            rep = torch.sum(token_out * attn.unsqueeze(-1), dim=1)
            logits = self.pair_heads[pk](rep)
            pair_logits[pk] = logits
            pair_attn[pk] = attn

            ida, idb = self.pair_class_ids[pk]
            delta = logits[:, 1] - logits[:, 0]
            correction[:, ida] = correction[:, ida] - 0.5 * delta
            correction[:, idb] = correction[:, idb] + 0.5 * delta

        final_logits = main_logits + self.correction_scale * correction

        return {
            "main_logits": main_logits,
            "final_logits": final_logits,
            "pair_logits": pair_logits,
            "pair_attn": pair_attn,
            "correction_scale": self.correction_scale,
        }


def make_loaders(inp: dict, Xtr_int: np.ndarray, Xva_int: np.ndarray, args, device: torch.device):
    train_ds = E2bDataset(
        inp["X_train_bin_norm"], inp["X_train_offset_norm"], inp["X_train_cont"], inp["M_train"], Xtr_int, inp["y_train"]
    )
    val_ds = E2bDataset(
        inp["X_val_bin_norm"], inp["X_val_offset_norm"], inp["X_val_cont"], inp["M_val"], Xva_int, inp["y_val"]
    )

    gen = torch.Generator()
    gen.manual_seed(int(args.seed))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        generator=gen,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    return train_loader, val_loader


def compute_class_weights(y: np.ndarray, num_classes: int, device: torch.device):
    counts = np.bincount(y.astype(int), minlength=num_classes).astype(np.float64)
    weights = counts.sum() / np.maximum(1.0, num_classes * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def compute_pair_loss(pair_logits: Dict[str, torch.Tensor], y: torch.Tensor, pair_class_ids: Dict[str, Tuple[int, int]], pair_weights: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
    losses = []
    logs = {}
    for pk, logits in pair_logits.items():
        ida, idb = pair_class_ids[pk]
        mask = (y == ida) | (y == idb)
        if mask.sum() == 0:
            continue
        target = (y[mask] == idb).long()
        weight = pair_weights.get(pk)
        loss = F.cross_entropy(logits[mask], target, weight=weight)
        losses.append(loss)
        logs[f"pair_{pk}_loss"] = float(loss.detach().cpu().item())
    if not losses:
        return torch.tensor(0.0, device=y.device), logs
    return torch.stack(losses).mean(), logs


def make_pair_weights(y_train: np.ndarray, pair_class_ids: Dict[str, Tuple[int, int]], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for pk, (ida, idb) in pair_class_ids.items():
        mask = (y_train == ida) | (y_train == idb)
        yb = (y_train[mask] == idb).astype(int)
        counts = np.bincount(yb, minlength=2).astype(np.float64)
        w = counts.sum() / np.maximum(1.0, 2.0 * counts)
        w = w / w.mean()
        out[pk] = torch.tensor(w, dtype=torch.float32, device=device)
    return out


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
    for group in optimizer.param_groups:
        group["lr"] = lr


def train_one_epoch(model, loader, optimizer, ce_main, ce_aux, pair_weights, pair_class_ids, device, args):
    model.train()
    ys = []
    preds_final = []
    preds_main = []
    total_loss = total_final = total_main = total_pair = 0.0
    n = 0
    for token_values, y in loader:
        token_values = token_values.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        out = model(token_values)
        final_logits = out["final_logits"]
        main_logits = out["main_logits"]

        final_loss = ce_main(final_logits, y)
        main_loss = ce_aux(main_logits, y)
        pair_loss, pair_logs = compute_pair_loss(out["pair_logits"], y, pair_class_ids, pair_weights)

        loss = final_loss + float(args.main_aux_weight) * main_loss + float(args.pair_aux_weight) * pair_loss
        if float(args.correction_l2) > 0:
            loss = loss + float(args.correction_l2) * (model.correction_scale ** 2)

        loss.backward()
        if float(args.grad_clip_norm) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip_norm))
        optimizer.step()

        bs = int(y.shape[0])
        n += bs
        total_loss += float(loss.item()) * bs
        total_final += float(final_loss.item()) * bs
        total_main += float(main_loss.item()) * bs
        total_pair += float(pair_loss.item()) * bs
        ys.append(y.detach().cpu().numpy())
        preds_final.append(final_logits.argmax(dim=1).detach().cpu().numpy())
        preds_main.append(main_logits.argmax(dim=1).detach().cpu().numpy())

    y_np = np.concatenate(ys)
    pf = np.concatenate(preds_final)
    pm = np.concatenate(preds_main)
    met = metric_dict(y_np, pf, total_loss / max(1, n))
    met["final_ce_loss"] = float(total_final / max(1, n))
    met["main_ce_loss"] = float(total_main / max(1, n))
    met["pair_loss"] = float(total_pair / max(1, n))
    met["main_macro_f1"] = float(f1_score(y_np, pm, average="macro"))
    met["correction_scale"] = float(model.correction_scale.detach().cpu().item())
    return met


@torch.no_grad()
def evaluate_model(model, loader, ce_main, ce_aux, pair_weights, pair_class_ids, device, args, collect_attention: bool = False):
    model.eval()
    ys = []
    preds_final = []
    preds_main = []
    total_loss = total_final = total_main = total_pair = 0.0
    n = 0
    pair_attn_chunks = {pk: [] for pk in ["RS", "RT", "ST"]}

    for token_values, y in loader:
        token_values = token_values.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(token_values)
        final_logits = out["final_logits"]
        main_logits = out["main_logits"]

        final_loss = ce_main(final_logits, y)
        main_loss = ce_aux(main_logits, y)
        pair_loss, _ = compute_pair_loss(out["pair_logits"], y, pair_class_ids, pair_weights)
        loss = final_loss + float(args.main_aux_weight) * main_loss + float(args.pair_aux_weight) * pair_loss

        bs = int(y.shape[0])
        n += bs
        total_loss += float(loss.item()) * bs
        total_final += float(final_loss.item()) * bs
        total_main += float(main_loss.item()) * bs
        total_pair += float(pair_loss.item()) * bs

        ys.append(y.detach().cpu().numpy())
        preds_final.append(final_logits.argmax(dim=1).detach().cpu().numpy())
        preds_main.append(main_logits.argmax(dim=1).detach().cpu().numpy())

        if collect_attention:
            for pk in pair_attn_chunks:
                pair_attn_chunks[pk].append(out["pair_attn"][pk].detach().cpu().numpy())

    y_np = np.concatenate(ys)
    pf = np.concatenate(preds_final)
    pm = np.concatenate(preds_main)
    met = metric_dict(y_np, pf, total_loss / max(1, n))
    met["final_ce_loss"] = float(total_final / max(1, n))
    met["main_ce_loss"] = float(total_main / max(1, n))
    met["pair_loss"] = float(total_pair / max(1, n))
    met["main_macro_f1"] = float(f1_score(y_np, pm, average="macro"))
    met["correction_scale"] = float(model.correction_scale.detach().cpu().item())

    attn = None
    if collect_attention:
        attn = {pk: np.concatenate(chunks, axis=0) for pk, chunks in pair_attn_chunks.items()}
    return met, y_np, pf, pm, attn


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
    df["sample_index"] = df["sample_index"].astype(int)
    df["true_id"] = df["true_id"].astype(int)
    df["pred_id"] = df["pred_id"].astype(int)
    df["true_label"] = df["true_label"].map(strip_label)
    df["pred_label"] = df["pred_label"].map(strip_label)
    df["correct"] = df["correct"].astype(bool)
    return df


def load_baseline(args, repo_root: Path, inp: dict) -> pd.DataFrame:
    pred_path = resolve_path(args.baseline_pred, repo_root)
    if not pred_path.exists():
        raise FileNotFoundError(f"baseline prediction file not found: {pred_path}")
    df = pd.read_csv(pred_path)
    df = normalize_pred_df(df, inp["label_to_id"], inp["id_to_label"])
    if len(df) != len(inp["y_val"]):
        raise ValueError(f"baseline rows {len(df)} != val rows {len(inp['y_val'])}")
    return df.sort_values("sample_index").reset_index(drop=True)


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
    base_correct = base_pred == y_true
    new_correct = new_pred == y_true
    rows = []
    for a, b in HARD_PAIRS:
        ida = inp["label_to_id"][a]
        idb = inp["label_to_id"][b]
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


def save_attention_top_tokens(attn: Dict[str, np.ndarray], token_names: List[str], out_dir: Path, topk: int):
    rows = []
    for pk, a in attn.items():
        mean_attn = a.mean(axis=0)
        order = np.argsort(-mean_attn)
        for rank, idx in enumerate(order[:int(topk)], start=1):
            rows.append({
                "pair_key": pk,
                "rank": rank,
                "token_index": int(idx),
                "token_name": token_names[int(idx)],
                "mean_attention": float(mean_attn[int(idx)]),
                "is_interaction_token": bool(int(idx) >= (len(token_names) - sum(name.startswith("INTER::") for name in token_names))),
            })
    pd.DataFrame(rows).to_csv(out_dir / "E2b_pair_attention_top_tokens.csv", index=False)


def zip_dir(src_dir: Path, zip_path: Path):
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(src_dir.rglob("*")):
            if p.is_file() and p != zip_path:
                z.write(p, p.relative_to(src_dir.parent))


def write_summary_md(out_dir: Path, summary: dict):
    text = f"""# E2b Interaction-token Pair-correction Attention

## Research position

E2b is an attention-based final model. Tree is not used in inference.

## Model

```text
D3 feature tokens
+ interaction tokens
+ Transformer encoder
+ CLS main 4-class logits
+ RS/RT/ST pair-query attention correction logits
```

## Baseline reference

```text
D3 official baseline macro-F1 ≈ 0.810094
E2a tree-guided distill macro-F1 = 0.817847
E1b tree expert macro-F1 ≈ 0.829387
Target macro-F1 = 0.900000
```

## E2b best epoch

```text
best_epoch = {summary['best_epoch']}
accuracy   = {summary['best_metrics']['accuracy']:.6f}
macro-F1   = {summary['best_metrics']['macro_f1']:.6f}
weighted   = {summary['best_metrics']['weighted_f1']:.6f}
main_macro = {summary['best_metrics']['main_macro_f1']:.6f}
corr_scale = {summary['best_metrics']['correction_scale']:.6f}
```

## Transition vs official D3 baseline

```text
wrong_to_correct = {summary['transition_vs_baseline']['wrong_to_correct']}
correct_to_wrong = {summary['transition_vs_baseline']['correct_to_wrong']}
net_gain         = {summary['transition_vs_baseline']['net_gain']}
damage_ratio     = {summary['transition_vs_baseline']['damage_ratio']}
changed_pred_n   = {summary['transition_vs_baseline']['changed_pred_n']}
```

## Interaction tokens

`E2b_interaction_specs.csv` records which interaction tokens were used.

`E2b_pair_attention_top_tokens.csv` shows whether RS/RT/ST pair-correction heads actually attend to interaction tokens.

## Key files

- `E2b_summary.json`
- `E2b_history.csv`
- `E2b_best_model.pt`
- `E2b_val_predictions_best.csv`
- `E2b_best_per_class_f1.csv`
- `E2b_best_confusion_matrix.csv`
- `E2b_best_pair_fix_damage.csv`
- `E2b_pair_attention_top_tokens.csv`
"""
    (out_dir / "E2b_summary.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="E2b interaction-token pair-correction attention")
    parser.add_argument("--dataset-npz", default="03_outputs/05_dataset/dataset.npz")
    parser.add_argument("--metadata-json", default="03_outputs/05_dataset/metadata.json")
    parser.add_argument("--train-raw", default="01_split/train_raw.csv")
    parser.add_argument("--val-raw", default="01_split/val_raw.csv")
    parser.add_argument("--official-train", default="02_src/07_train.py")
    parser.add_argument("--baseline-pred", default="03_outputs/06_model/val_predictions_best.csv")
    parser.add_argument("--out-dir", default="05_test/outputs/E2b_interaction_pair_correction_attention")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    # Interaction specs.
    parser.add_argument("--interaction-specs", default="", help="CSV path or 'name|pair|op|feat_a|feat_b;...'")
    parser.add_argument("--ratio-eps", type=float, default=1e-3)
    parser.add_argument("--interaction-clip-low", type=float, default=0.01)
    parser.add_argument("--interaction-clip-high", type=float, default=0.99)

    # Model.
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--value-hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--classifier-hidden-dim", type=int, default=128)
    parser.add_argument("--classifier-dropout", type=float, default=0.1)
    parser.add_argument("--correction-scale-init", type=float, default=0.25)
    parser.add_argument("--fixed-correction-scale", action="store_true", default=False)

    # Train.
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
    parser.add_argument("--main-aux-weight", type=float, default=0.25)
    parser.add_argument("--pair-aux-weight", type=float, default=0.40)
    parser.add_argument("--correction-l2", type=float, default=0.0)
    parser.add_argument("--no-class-weights", dest="use_class_weights", action="store_false")
    parser.set_defaults(use_class_weights=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--attention-topk", type=int, default=20)

    parser.add_argument("--make-zip", action="store_true", default=True)
    parser.add_argument("--no-zip", dest="make_zip", action="store_false")
    args = parser.parse_args()

    repo_root = repo_root_from_here()
    out_dir = resolve_path(args.out_dir, repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(args.seed))
    device = pick_device(args.device)

    print(f"[E2b] repo_root={repo_root}", flush=True)
    print(f"[E2b] out_dir={out_dir}", flush=True)
    print(f"[E2b] device={device}", flush=True)

    inp = load_official_inputs(args, repo_root)

    pair_class_ids = {}
    for pk, (a, b) in PAIR_FROM_KEY.items():
        pair_class_ids[pk] = (int(inp["label_to_id"][a]), int(inp["label_to_id"][b]))

    specs = parse_interactions(args, inp["feature_names"])
    Xtr_int, Xva_int, interaction_meta = build_interaction_matrix(inp, specs, args, out_dir)
    interaction_names = [f"INTER::{x}" for x in interaction_meta["interaction_name"].tolist()]
    token_names = [f"FEAT::{x}" for x in inp["feature_names"]] + interaction_names

    train_loader, val_loader = make_loaders(inp, Xtr_int, Xva_int, args, device)

    model = E2bInteractionPairCorrectionTransformer(
        n_base_features=len(inp["feature_names"]),
        n_interactions=Xtr_int.shape[1],
        num_classes=len(inp["label_names"]),
        pair_class_ids=pair_class_ids,
        d_model=int(args.d_model),
        value_hidden_dim=int(args.value_hidden_dim),
        num_layers=int(args.num_layers),
        num_heads=int(args.num_heads),
        dropout=float(args.dropout),
        classifier_hidden_dim=int(args.classifier_hidden_dim),
        classifier_dropout=float(args.classifier_dropout),
        correction_scale_init=float(args.correction_scale_init),
        learnable_correction_scale=not bool(args.fixed_correction_scale),
    ).to(device)

    class_weight = compute_class_weights(inp["y_train"], len(inp["label_names"]), device) if args.use_class_weights else None
    ce_main = nn.CrossEntropyLoss(weight=class_weight)
    ce_aux = nn.CrossEntropyLoss(weight=class_weight)
    pair_weights = make_pair_weights(inp["y_train"], pair_class_ids, device) if args.use_class_weights else {}

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    save_json(out_dir / "E2b_run_config.json", {
        "stage": "E2b_interaction_pair_correction_attention",
        "target_macro_f1": 0.90,
        "args": vars(args),
        "device": str(device),
        "label_names": inp["label_names"],
        "num_bins": int(inp["num_bins"]),
        "continuous_info": inp["continuous_info"],
        "pair_class_ids": pair_class_ids,
        "n_base_features": len(inp["feature_names"]),
        "n_interactions": int(Xtr_int.shape[1]),
        "interaction_names": interaction_names,
        "model_role": "attention final model; tree not used in inference",
    })

    best_score = -1.0
    best_epoch = -1
    best_state = None
    no_improve = 0
    history = []

    print(
        f"[E2b] train_n={len(inp['y_train'])} val_n={len(inp['y_val'])} "
        f"features={len(inp['feature_names'])} interactions={Xtr_int.shape[1]} "
        f"params={sum(p.numel() for p in model.parameters()):,}",
        flush=True,
    )

    for epoch in range(1, int(args.epochs) + 1):
        lr_epoch = compute_lr(epoch, args)
        set_optimizer_lr(optimizer, lr_epoch)

        t0 = time.time()
        train_met = train_one_epoch(model, train_loader, optimizer, ce_main, ce_aux, pair_weights, pair_class_ids, device, args)
        val_met, yv, pred_final, pred_main, _ = evaluate_model(model, val_loader, ce_main, ce_aux, pair_weights, pair_class_ids, device, args, collect_attention=False)
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
                "feature_names": inp["feature_names"],
                "interaction_names": interaction_names,
                "label_names": inp["label_names"],
            }, out_dir / "E2b_best_model.pt")
        else:
            no_improve += 1

        if epoch == 1 or improved or epoch % int(args.log_every) == 0:
            print(
                f"[E2b] ep={epoch:03d} lr={lr_epoch:.3e} "
                f"train_f1={train_met['macro_f1']:.4f} train_main_f1={train_met['main_macro_f1']:.4f} "
                f"val_f1={val_met['macro_f1']:.4f} val_main_f1={val_met['main_macro_f1']:.4f} "
                f"corr={val_met['correction_scale']:.4f} best={best_score:.4f}@{best_epoch} "
                f"noimp={no_improve} sec={dt:.1f}",
                flush=True,
            )

        if no_improve >= int(args.patience):
            print(f"[E2b] early stop at epoch {epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("No best checkpoint saved")

    model.load_state_dict(best_state)
    final_met, y_val, pred_final, pred_main, attn = evaluate_model(
        model, val_loader, ce_main, ce_aux, pair_weights, pair_class_ids, device, args, collect_attention=True
    )
    pd.DataFrame(history).to_csv(out_dir / "E2b_history.csv", index=False)

    per, cm = per_class_and_cm(y_val, pred_final, inp["label_names"])
    per_main, cm_main = per_class_and_cm(y_val, pred_main, inp["label_names"])
    per.to_csv(out_dir / "E2b_best_per_class_f1.csv", index=False)
    cm.to_csv(out_dir / "E2b_best_confusion_matrix.csv")
    per_main.to_csv(out_dir / "E2b_best_main_only_per_class_f1.csv", index=False)
    cm_main.to_csv(out_dir / "E2b_best_main_only_confusion_matrix.csv")

    base = load_baseline(args, repo_root, inp)
    base_pred = base["pred_id"].to_numpy(dtype=int)
    trans = transition_stats(y_val, base_pred, pred_final)
    pair_fd = pair_fix_damage(y_val, base_pred, pred_final, inp)
    pair_fd.to_csv(out_dir / "E2b_best_pair_fix_damage.csv", index=False)

    pred_df = base[["sample_index", "true_id", "true_label", "pred_id", "pred_label", "correct"]].copy()
    pred_df = pred_df.rename(columns={"pred_id": "base_pred_id", "pred_label": "base_pred_label", "correct": "base_correct"})
    pred_df["e2b_pred_id"] = pred_final.astype(int)
    pred_df["e2b_pred_label"] = [inp["id_to_label"][int(i)] for i in pred_final]
    pred_df["e2b_correct"] = pred_final == y_val
    pred_df["main_only_pred_id"] = pred_main.astype(int)
    pred_df["main_only_pred_label"] = [inp["id_to_label"][int(i)] for i in pred_main]
    pred_df["main_only_correct"] = pred_main == y_val
    pred_df["transition"] = "both_wrong"
    pred_df.loc[pred_df["base_correct"] & pred_df["e2b_correct"], "transition"] = "both_correct"
    pred_df.loc[(~pred_df["base_correct"]) & pred_df["e2b_correct"], "transition"] = "fixed"
    pred_df.loc[pred_df["base_correct"] & (~pred_df["e2b_correct"]), "transition"] = "damaged"
    pred_df.to_csv(out_dir / "E2b_val_predictions_best.csv", index=False)

    save_attention_top_tokens(attn, token_names, out_dir, int(args.attention_topk))

    summary = {
        "stage": "E2b_interaction_pair_correction_attention",
        "target_macro_f1": 0.90,
        "research_position": "attention final model; tree not used in inference",
        "references": {
            "D3_official_baseline_macro_f1": 0.810094,
            "E2a_distill_macro_f1": 0.817847,
            "E1b_tree_expert_macro_f1": 0.829387,
        },
        "best_epoch": int(best_epoch),
        "best_metrics": final_met,
        "transition_vs_baseline": trans,
        "model_params": int(sum(p.numel() for p in model.parameters())),
        "n_base_features": int(len(inp["feature_names"])),
        "n_interactions": int(Xtr_int.shape[1]),
        "interaction_names": interaction_names,
        "outputs": {
            "run_config": str(out_dir / "E2b_run_config.json"),
            "history": str(out_dir / "E2b_history.csv"),
            "best_model": str(out_dir / "E2b_best_model.pt"),
            "val_predictions": str(out_dir / "E2b_val_predictions_best.csv"),
            "per_class": str(out_dir / "E2b_best_per_class_f1.csv"),
            "confusion_matrix": str(out_dir / "E2b_best_confusion_matrix.csv"),
            "pair_fix_damage": str(out_dir / "E2b_best_pair_fix_damage.csv"),
            "attention_top_tokens": str(out_dir / "E2b_pair_attention_top_tokens.csv"),
            "interaction_specs": str(out_dir / "E2b_interaction_specs.csv"),
        },
        "guardrail": "Validation-set diagnostic. If promising, rerun with seeds and batch256 official comparison.",
    }
    save_json(out_dir / "E2b_summary.json", summary)
    write_summary_md(out_dir, summary)

    if args.make_zip:
        zip_path = out_dir.with_suffix(".zip")
        zip_dir(out_dir, zip_path)
        print(f"[E2b] zipped outputs: {zip_path}", flush=True)

    print("[E2b] done.", flush=True)
    print(f"[E2b] best_epoch={best_epoch}", flush=True)
    print(f"[E2b] best_macro_f1={final_met['macro_f1']:.6f}", flush=True)
    print(f"[E2b] main_only_macro_f1={final_met['main_macro_f1']:.6f}", flush=True)
    print(f"[E2b] transition={trans}", flush=True)


if __name__ == "__main__":
    main()
