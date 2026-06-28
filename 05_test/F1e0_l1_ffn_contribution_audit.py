#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F1e0 L1 FFN Contribution Audit

Question
--------
F1d showed that L1 residual gap is low at self-attention, but grows around
linear2/layer0 output:

    self_attn gap      low
    linear2 gap        high
    layer0 output gap  high

But this only proves FFN is related to the remaining gap.
It does NOT prove we should shrink/control FFN.

F1e0 audits what FFN is actually doing before training any FFN-control model.

No training is performed.

Main audits
-----------
1) Frozen FFN alpha sweep on the trained L1 checkpoint:
       alpha = 0.00, 0.25, 0.50, 0.75, 1.00, 1.25
   This answers:
       If we reduce FFN contribution after training, do train/val/gap improve or collapse?

2) FFN effect on train vs val:
       attention-side representation -> final layer0 representation
   using centroid margins:
       Does FFN delta improve true-class margin?
       Does it hurt val-wrong samples?

3) Fix/damage analysis:
       Relative to original alpha=1.00, for each alpha:
       - correct -> wrong damage
       - wrong -> correct fix
       - wrong -> wrong transition
       - per-class and per-pair effects

Interpretation
--------------
Only if alpha < 1 improves val and keeps train high do we have evidence that
an FFN gate/control model is worth training. Otherwise, do not gate FFN blindly.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import math
import sys
import traceback
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


L1_REF = {
    "train_macro_f1": 0.911431,
    "val_macro_f1": 0.814224,
    "gap_macro_f1": 0.097207,
}


def log(msg: str) -> None:
    print(f"[F1e0] {msg}", flush=True)


def repo_root_from_here() -> Path:
    p = Path(__file__).resolve()
    if p.parent.name == "05_test":
        return p.parents[1]
    return Path.cwd().resolve()


def resolve_path(p: str | Path, root: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (root / p).resolve()


def load_json(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_json(p: Path) -> Dict[str, Any]:
    try:
        return load_json(p) if p.exists() else {}
    except Exception:
        return {}


def cfg_get(cfg: Dict[str, Any], k: str, default):
    if k in cfg:
        return cfg[k]
    for dname in ["model", "model_config", "training", "args"]:
        d = cfg.get(dname, {})
        if isinstance(d, dict) and k in d:
            return d[k]
    return default


def find_checkpoint(run_dir: Path) -> Optional[Path]:
    for n in ["best_model.pt", "model_best.pt", "checkpoint_best.pt", "best.pt", "checkpoint.pt", "model.pt"]:
        p = run_dir / n
        if p.exists():
            return p
    pts = sorted(run_dir.glob("*.pt"))
    return pts[0] if pts else None


def load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_checkpoint(path: Path, device: str, trust: bool):
    if trust:
        return torch.load(path, map_location=device, weights_only=False)
    try:
        return torch.load(path, map_location=device)
    except Exception:
        from torch.serialization import safe_globals
        from torch.torch_version import TorchVersion
        with safe_globals([TorchVersion]):
            return torch.load(path, map_location=device)


def state_dict_from_ckpt(ckpt):
    if isinstance(ckpt, dict):
        for k in ["model_state_dict", "state_dict", "net_state_dict"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k], k
        if all(isinstance(k, str) for k in ckpt.keys()) and any(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt, "checkpoint_is_state_dict"
    if isinstance(ckpt, nn.Module):
        return ckpt.state_dict(), "module_state_dict"
    return None, "not_found"


def infer_model_config(cfg: Dict[str, Any], ds_info: Dict[str, Any], y_train: np.ndarray) -> Dict[str, Any]:
    return {
        "num_bins": int(cfg_get(cfg, "num_bins", cfg_get(cfg, "K", ds_info["num_bins"]))),
        "n_features": int(cfg_get(cfg, "n_features", cfg_get(cfg, "num_features", ds_info["n_features"]))),
        "num_classes": int(cfg_get(cfg, "num_classes", len(np.unique(y_train)))),
        "value_dim": int(cfg_get(cfg, "value_dim", 32)),
        "feature_dim": int(cfg_get(cfg, "feature_dim", 32)),
        "hidden_dim": int(cfg_get(cfg, "hidden_dim", 128)),
        "num_layers": int(cfg_get(cfg, "num_layers", 1)),
        "num_heads": int(cfg_get(cfg, "num_heads", 4)),
        "dropout": float(cfg_get(cfg, "dropout", 0.1)),
        "classifier_hidden_dim": int(cfg_get(cfg, "classifier_hidden_dim", 128)),
        "classifier_dropout": float(cfg_get(cfg, "classifier_dropout", 0.1)),
        "gate_init": float(cfg_get(cfg, "gate_init", 0.0)),
    }


def build_model(root: Path, run_dir: Path, ds_info: Dict[str, Any], y_train: np.ndarray, device: str, trust: bool):
    ckpt_path = find_checkpoint(run_dir)
    if ckpt_path is None:
        raise FileNotFoundError(f"no checkpoint in {run_dir}")
    ckpt = load_checkpoint(ckpt_path, device, trust)
    sd, sd_mode = state_dict_from_ckpt(ckpt)
    if sd is None:
        raise RuntimeError("state_dict not found")

    cfg = safe_json(run_dir / "config.json")
    mcfg = infer_model_config(cfg, ds_info, y_train)

    mod = load_module_from_path("_f1e0_model_06_model", root / "02_src" / "06_model.py")
    cls = getattr(mod, "D3C2D3Transformer", None)
    if cls is None:
        raise RuntimeError("D3C2D3Transformer not found in 02_src/06_model.py")

    kwargs = {k: v for k, v in mcfg.items() if k in inspect.signature(cls).parameters}
    model = cls(**kwargs)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    info = {
        "checkpoint_path": str(ckpt_path),
        "state_dict_mode": sd_mode,
        "class": "D3C2D3Transformer",
        "kwargs": kwargs,
        "missing": len(missing),
        "unexpected": len(unexpected),
        "first_missing": list(missing)[:20],
        "first_unexpected": list(unexpected)[:20],
    }
    if missing or unexpected:
        raise RuntimeError("checkpoint load mismatch:\n" + json.dumps(info, indent=2, default=str))
    model.to(device).eval()
    return model, info


def load_dataset(dataset_npz: Path, train_raw: Path, val_raw: Path):
    data = np.load(dataset_npz, allow_pickle=True)
    Xtr_bin = np.asarray(data["X_train_bin"], dtype=np.int64)
    Xtr_off = np.asarray(data["X_train_offset"], dtype=np.float32)
    ytr = np.asarray(data["y_train"], dtype=np.int64).reshape(-1)
    Xva_bin = np.asarray(data["X_val_bin"], dtype=np.int64)
    Xva_off = np.asarray(data["X_val_offset"], dtype=np.float32)
    yva = np.asarray(data["y_val"], dtype=np.int64).reshape(-1)
    feature_names = [str(x) for x in np.asarray(data["feature_names"]).tolist()] if "feature_names" in data.files else [f"f{i}" for i in range(Xtr_bin.shape[1])]
    num_bins = int(np.asarray(data["num_bins"]).reshape(-1)[0]) if "num_bins" in data.files else 512

    def load_raw(path: Path):
        df = pd.read_csv(path)
        cols = [c for c in feature_names if c in df.columns]
        if len(cols) != len(feature_names):
            exclude = {"label", "Label", "Class", "Category", "class", "category", "label_L1", "label_L2", "label_L3", "label_l1", "label_l2", "label_l3"}
            cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])][:len(feature_names)]
        if len(cols) != len(feature_names):
            raise ValueError(f"raw feature mismatch: got {len(cols)}, expected {len(feature_names)}")
        return df[cols].to_numpy(dtype=np.float32), cols

    raw_available = train_raw.exists() and val_raw.exists()
    if raw_available:
        Rtr, cols = load_raw(train_raw)
        Rva, _ = load_raw(val_raw)
        mn = np.nanmin(Rtr, axis=0, keepdims=True)
        mx = np.nanmax(Rtr, axis=0, keepdims=True)
        den = mx - mn
        den[den < 1e-8] = 1.0
        Xtr_raw = np.clip((Rtr - mn) / den, 0.0, 1.0).astype(np.float32)
        Xva_raw = np.clip((Rva - mn) / den, 0.0, 1.0).astype(np.float32)
    else:
        Xtr_raw = Xtr_off.copy()
        Xva_raw = Xva_off.copy()
        cols = []

    ds = {
        "train": {"bin": Xtr_bin, "off": Xtr_off, "raw": Xtr_raw, "y": ytr},
        "val": {"bin": Xva_bin, "off": Xva_off, "raw": Xva_raw, "y": yva},
    }
    info = {
        "dataset_npz": str(dataset_npz),
        "keys": list(data.files),
        "n_train": int(len(ytr)),
        "n_val": int(len(yva)),
        "n_features": int(Xtr_bin.shape[1]),
        "num_bins": int(num_bins),
        "raw_available": bool(raw_available),
        "raw_cols_preview": cols[:10],
        "feature_names_preview": feature_names[:10],
    }
    return ds, info


def make_values(split: Dict[str, np.ndarray], name: str, num_bins: int):
    off = split["off"].astype(np.float32)
    raw = split["raw"].astype(np.float32)
    bin_norm = np.clip(split["bin"].astype(np.float32) / max(1, num_bins), 0.0, 1.0)
    z = np.zeros_like(off, dtype=np.float32)
    o = np.ones_like(off, dtype=np.float32)
    return {
        "offset_raw_one": np.stack([off, raw, o], axis=-1),
        "offset_raw_zero": np.stack([off, raw, z], axis=-1),
        "offset_raw_bin_norm": np.stack([off, raw, bin_norm], axis=-1),
        "offset_raw_offset": np.stack([off, raw, off], axis=-1),
        "raw_offset_zero": np.stack([raw, off, z], axis=-1),
        "bin_norm_raw_offset": np.stack([bin_norm, raw, off], axis=-1),
        "offset_offset_raw": np.stack([off, off, raw], axis=-1),
        "raw_raw_offset": np.stack([raw, raw, off], axis=-1),
    }[name].astype(np.float32)


def output_to_logits(out):
    if torch.is_tensor(out):
        return out if out.ndim == 2 else out.reshape(out.shape[0], -1)
    if isinstance(out, dict):
        for k in ["logits", "output", "outputs", "pred"]:
            if k in out:
                got = output_to_logits(out[k])
                if got is not None:
                    return got
        for v in out.values():
            got = output_to_logits(v)
            if got is not None:
                return got
    if isinstance(out, (tuple, list)):
        for v in out:
            got = output_to_logits(v)
            if got is not None:
                return got
    return None


def macro_f1(y, p): return float(f1_score(y, p, average="macro", zero_division=0))
def weighted_f1(y, p): return float(f1_score(y, p, average="weighted", zero_division=0))
def acc(y, p): return float(accuracy_score(y, p))


def get_l1_layer(model: nn.Module):
    if not hasattr(model, "encoder") or not hasattr(model.encoder, "layers"):
        raise AttributeError("model.encoder.layers not found")
    layers = model.encoder.layers
    if len(layers) < 1:
        raise RuntimeError("encoder.layers empty")
    return layers[0]


def activation_forward(layer, x):
    act = getattr(layer, "activation", None)
    if act is None:
        return F.relu(x)
    return act(x)


def _canonical_mask(mask, mask_name, other_type, other_name, target_type, check_other=True):
    # Use torch internal helper if available. Otherwise return original mask.
    try:
        from torch.nn.functional import _canonical_mask as cm
        return cm(mask=mask, mask_name=mask_name, other_type=other_type, other_name=other_name, target_type=target_type, check_other=check_other)
    except Exception:
        return mask


class FfnAlphaPatch:
    """
    Runtime-only patch for a PyTorch-like TransformerEncoderLayer.

    It exposes:
      - set_alpha(a)
      - capture keys from the most recent forward:
          attn_side
          ffn_raw
          ffn_scaled_delta
          layer_out

    For norm_first=False:
      attn_side = norm1(x + self_attn(x))
      ffn_raw = linear2(dropout(activation(linear1(attn_side))))
      layer_out = norm2(attn_side + alpha * dropout2(ffn_raw))

    For norm_first=True:
      attn_side = x + self_attn(norm1(x))
      ffn_raw = linear2(dropout(activation(linear1(norm2(attn_side)))))
      layer_out = attn_side + alpha * dropout2(ffn_raw)

    In eval mode dropout is inactive.
    """

    def __init__(self, layer: nn.Module, alpha: float = 1.0):
        self.layer = layer
        self.alpha = float(alpha)
        self.original_forward = layer.forward
        self.last = {}
        self.patch()

    def set_alpha(self, alpha: float):
        self.alpha = float(alpha)

    def patch(self):
        layer = self.layer
        patcher = self

        def fwd(src, src_mask=None, src_key_padding_mask=None, is_causal=False):
            x = src

            src_key_padding_mask2 = _canonical_mask(
                mask=src_key_padding_mask,
                mask_name="src_key_padding_mask",
                other_type=None if src_mask is None else src_mask.dtype,
                other_name="src_mask",
                target_type=x.dtype,
            )
            src_mask2 = _canonical_mask(
                mask=src_mask,
                mask_name="src_mask",
                other_type=None,
                other_name="",
                target_type=x.dtype,
                check_other=False,
            )

            norm_first = bool(getattr(layer, "norm_first", False))
            # PyTorch self_attn accepts is_causal in recent versions, but not all.
            def self_attn_block(q):
                try:
                    out = layer.self_attn(
                        q, q, q,
                        attn_mask=src_mask2,
                        key_padding_mask=src_key_padding_mask2,
                        need_weights=False,
                        is_causal=is_causal,
                    )[0]
                except TypeError:
                    out = layer.self_attn(
                        q, q, q,
                        attn_mask=src_mask2,
                        key_padding_mask=src_key_padding_mask2,
                        need_weights=False,
                    )[0]
                return layer.dropout1(out)

            if norm_first:
                sa = self_attn_block(layer.norm1(x))
                attn_side = x + sa
                ffn_in = layer.norm2(attn_side)
                ffn_raw = layer.linear2(layer.dropout(activation_forward(layer, layer.linear1(ffn_in))))
                ffn_delta = layer.dropout2(ffn_raw)
                out = attn_side + patcher.alpha * ffn_delta
            else:
                sa = self_attn_block(x)
                attn_side = layer.norm1(x + sa)
                ffn_raw = layer.linear2(layer.dropout(activation_forward(layer, layer.linear1(attn_side))))
                ffn_delta = layer.dropout2(ffn_raw)
                out = layer.norm2(attn_side + patcher.alpha * ffn_delta)

            # capture detached tensors only when requested by caller
            if getattr(patcher, "capture_enabled", False):
                patcher.last = {
                    "attn_side": attn_side.detach(),
                    "ffn_raw": ffn_raw.detach(),
                    "ffn_scaled_delta": (patcher.alpha * ffn_delta).detach(),
                    "ffn_unscaled_delta": ffn_delta.detach(),
                    "layer_out": out.detach(),
                }
            else:
                patcher.last = {}
            return out

        layer.forward = fwd

    def restore(self):
        self.layer.forward = self.original_forward


def tensor_to_cls_np(t: torch.Tensor) -> np.ndarray:
    t = t.detach().float()
    if t.ndim == 3:
        return t[:, 0, :].cpu().numpy()
    if t.ndim == 2:
        return t.cpu().numpy()
    if t.ndim == 1:
        return t.reshape(-1, 1).cpu().numpy()
    b = t.shape[0]
    return t.reshape(b, -1).mean(dim=1, keepdim=True).cpu().numpy()


@torch.no_grad()
def forward_predict(model, ds_split, values, device, batch_size) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    logits_all = []
    n = len(ds_split["y"])
    for st in range(0, n, batch_size):
        ed = min(n, st + batch_size)
        t = torch.as_tensor(ds_split["bin"][st:ed], dtype=torch.long, device=device)
        v = torch.as_tensor(values[st:ed], dtype=torch.float32, device=device)
        out = model(t, v)
        logits = output_to_logits(out)
        if logits is None:
            raise RuntimeError("model output did not contain logits")
        logits_all.append(logits.detach().float().cpu().numpy())
    logits = np.concatenate(logits_all, axis=0)
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    pred = probs.argmax(axis=1)
    return logits, probs, pred


def eval_metrics(y, pred, probs=None) -> Dict[str, float]:
    out = {
        "accuracy": acc(y, pred),
        "macro_f1": macro_f1(y, pred),
        "weighted_f1": weighted_f1(y, pred),
    }
    if probs is not None:
        out["mean_confidence"] = float(np.max(probs, axis=1).mean())
    return out


def class_names(num_classes: int) -> List[str]:
    if num_classes == 4:
        return ["Benign", "Ransomware", "Spyware", "Trojan"]
    return [str(i) for i in range(num_classes)]


def report_and_cm(out_dir: Path, prefix: str, y, pred, probs, names):
    rep = classification_report(
        y, pred,
        labels=list(range(len(names))),
        target_names=names,
        output_dict=True,
        zero_division=0,
    )
    (out_dir / f"{prefix}_classification_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    pd.DataFrame(confusion_matrix(y, pred, labels=list(range(len(names)))), index=names, columns=names).to_csv(
        out_dir / f"{prefix}_confusion_matrix.csv"
    )
    df = pd.DataFrame({
        "sample_idx": np.arange(len(y)),
        "y_true": y,
        "y_pred": pred,
        "true_name": [names[int(i)] for i in y],
        "pred_name": [names[int(i)] for i in pred],
    })
    for i, name in enumerate(names):
        df[f"prob_{name}"] = probs[:, i]
    df.to_csv(out_dir / f"{prefix}_predictions.csv", index=False)


def choose_value_candidate(model, ds, num_bins, device, batch_size, expected_val_macro: Optional[float], max_eval_samples: int, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(ds["val"]["y"]))
    if max_eval_samples and len(idx) > max_eval_samples:
        idx = np.sort(rng.choice(idx, max_eval_samples, replace=False))
    cand_names = [
        "offset_raw_one",
        "offset_raw_zero",
        "offset_raw_bin_norm",
        "offset_raw_offset",
        "raw_offset_zero",
        "bin_norm_raw_offset",
        "offset_offset_raw",
        "raw_raw_offset",
    ]
    rows = []
    best = None
    best_score = None
    for name in cand_names:
        try:
            vals = make_values(ds["val"], name, num_bins)
            logits, probs, pred = forward_predict(
                model,
                {"bin": ds["val"]["bin"][idx], "y": ds["val"]["y"][idx]},
                vals[idx],
                device,
                batch_size,
            )
            mf = macro_f1(ds["val"]["y"][idx], pred)
            ac = acc(ds["val"]["y"][idx], pred)
            score = abs(mf - expected_val_macro) if expected_val_macro is not None else -mf
            row = {"candidate": name, "ok": True, "val_macro_f1_sample": mf, "val_acc_sample": ac, "score_abs_diff_expected": score, "n_eval": len(idx)}
            if best_score is None or score < best_score:
                best_score = score
                best = name
        except Exception as e:
            row = {"candidate": name, "ok": False, "error": repr(e)}
        rows.append(row)
    if best is None:
        raise RuntimeError("No value candidate worked: " + json.dumps(rows, indent=2, default=str))
    return best, pd.DataFrame(rows)


def parse_alphas(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def alpha_sweep(model, patcher: FfnAlphaPatch, ds, value_name: str, num_bins: int, alphas: List[float], device: str, batch_size: int, out_dir: Path, names: List[str]):
    vals = {
        "train": make_values(ds["train"], value_name, num_bins),
        "val": make_values(ds["val"], value_name, num_bins),
    }

    rows = []
    preds_by_alpha = {}
    probs_by_alpha = {}

    for a in alphas:
        patcher.set_alpha(a)
        log(f"alpha sweep {a}")
        row = {"alpha": a}
        for split in ["train", "val"]:
            logits, probs, pred = forward_predict(model, ds[split], vals[split], device, batch_size)
            m = eval_metrics(ds[split]["y"], pred, probs)
            for k, v in m.items():
                row[f"{split}_{k}"] = v
            preds_by_alpha[(split, a)] = pred
            probs_by_alpha[(split, a)] = probs
            report_and_cm(out_dir, f"alpha_{a:g}_{split}", ds[split]["y"], pred, probs, names)
        row["gap_macro_f1"] = row["train_macro_f1"] - row["val_macro_f1"]
        row["delta_val_macro_f1_vs_alpha1"] = np.nan
        row["delta_gap_vs_alpha1"] = np.nan
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("alpha")
    if any(abs(df["alpha"] - 1.0) < 1e-9):
        base = df.loc[np.isclose(df["alpha"], 1.0)].iloc[0]
        df["delta_val_macro_f1_vs_alpha1"] = df["val_macro_f1"] - float(base["val_macro_f1"])
        df["delta_train_macro_f1_vs_alpha1"] = df["train_macro_f1"] - float(base["train_macro_f1"])
        df["delta_gap_vs_alpha1"] = df["gap_macro_f1"] - float(base["gap_macro_f1"])
        df["keeps_train_high_ge_0p900"] = df["train_macro_f1"] >= 0.900
        df["beats_alpha1_val"] = df["val_macro_f1"] > float(base["val_macro_f1"])
        df["reduces_gap"] = df["gap_macro_f1"] < float(base["gap_macro_f1"])
    df.to_csv(out_dir / "F1e0_frozen_ffn_alpha_sweep_metrics.csv", index=False)

    # Per-class F1 table.
    per_rows = []
    for a in alphas:
        for split in ["train", "val"]:
            y = ds[split]["y"]
            p = preds_by_alpha[(split, a)]
            rep = classification_report(y, p, labels=list(range(len(names))), target_names=names, output_dict=True, zero_division=0)
            for cls in names:
                per_rows.append({
                    "alpha": a,
                    "split": split,
                    "class": cls,
                    "precision": rep[cls]["precision"],
                    "recall": rep[cls]["recall"],
                    "f1": rep[cls]["f1-score"],
                    "support": rep[cls]["support"],
                })
    per_df = pd.DataFrame(per_rows)
    per_df.to_csv(out_dir / "F1e0_frozen_ffn_alpha_sweep_per_class.csv", index=False)

    return df, preds_by_alpha, probs_by_alpha


def fix_damage_analysis(ds, preds_by_alpha, probs_by_alpha, alphas, out_dir: Path, names: List[str]):
    if ("val", 1.0) not in preds_by_alpha:
        log("alpha=1.0 not found; skip fix/damage")
        return pd.DataFrame(), pd.DataFrame()

    base_pred = preds_by_alpha[("val", 1.0)]
    base_probs = probs_by_alpha[("val", 1.0)]
    y = ds["val"]["y"]

    rows = []
    pair_rows = []
    changed_samples = []

    base_correct = base_pred == y
    for a in alphas:
        pred = preds_by_alpha[("val", a)]
        probs = probs_by_alpha[("val", a)]
        correct = pred == y
        changed = pred != base_pred

        status = np.full(len(y), "unchanged", dtype=object)
        status[(~base_correct) & correct] = "fix_wrong_to_correct"
        status[base_correct & (~correct)] = "damage_correct_to_wrong"
        status[(~base_correct) & (~correct) & changed] = "wrong_to_wrong_changed"
        status[base_correct & correct & changed] = "correct_to_correct_changed"

        rows.append({
            "alpha": a,
            "n": int(len(y)),
            "changed": int(changed.sum()),
            "changed_rate": float(changed.mean()),
            "fix_wrong_to_correct": int((status == "fix_wrong_to_correct").sum()),
            "damage_correct_to_wrong": int((status == "damage_correct_to_wrong").sum()),
            "wrong_to_wrong_changed": int((status == "wrong_to_wrong_changed").sum()),
            "correct_to_correct_changed": int((status == "correct_to_correct_changed").sum()),
            "net_fix_minus_damage": int((status == "fix_wrong_to_correct").sum() - (status == "damage_correct_to_wrong").sum()),
            "damage_ratio_damage_over_fix": float(((status == "damage_correct_to_wrong").sum()) / max(1, (status == "fix_wrong_to_correct").sum())),
        })

        for true_i in range(len(names)):
            for base_j in range(len(names)):
                mask_pair = (y == true_i) & (base_pred == base_j)
                if not mask_pair.any():
                    continue
                for new_k in range(len(names)):
                    m = mask_pair & (pred == new_k)
                    if m.any():
                        pair_rows.append({
                            "alpha": a,
                            "true": names[true_i],
                            "base_pred": names[base_j],
                            "new_pred": names[new_k],
                            "count": int(m.sum()),
                            "base_correct": bool(true_i == base_j),
                            "new_correct": bool(true_i == new_k),
                        })

        # Save changed samples for non-alpha1 only.
        if abs(a - 1.0) > 1e-9:
            ids = np.where(changed)[0]
            for idx in ids[:20000]:
                changed_samples.append({
                    "alpha": a,
                    "sample_idx": int(idx),
                    "y_true": int(y[idx]),
                    "true_name": names[int(y[idx])],
                    "base_pred": int(base_pred[idx]),
                    "base_pred_name": names[int(base_pred[idx])],
                    "new_pred": int(pred[idx]),
                    "new_pred_name": names[int(pred[idx])],
                    "status": str(status[idx]),
                    "base_conf": float(base_probs[idx].max()),
                    "new_conf": float(probs[idx].max()),
                    "base_prob_true": float(base_probs[idx, int(y[idx])]),
                    "new_prob_true": float(probs[idx, int(y[idx])]),
                    "delta_prob_true": float(probs[idx, int(y[idx])] - base_probs[idx, int(y[idx])]),
                })

    fix_df = pd.DataFrame(rows)
    pair_df = pd.DataFrame(pair_rows)
    changed_df = pd.DataFrame(changed_samples)
    fix_df.to_csv(out_dir / "F1e0_val_fix_damage_by_alpha.csv", index=False)
    pair_df.to_csv(out_dir / "F1e0_val_transition_by_alpha_pair.csv", index=False)
    changed_df.to_csv(out_dir / "F1e0_val_changed_samples_by_alpha.csv", index=False)
    return fix_df, pair_df


@torch.no_grad()
def extract_attn_ffn_reps(model, patcher: FfnAlphaPatch, ds, value_name: str, num_bins: int, device: str, batch_size: int, max_samples: int, seed: int):
    # Always extract at alpha=1.0 because we are auditing trained L1's actual FFN contribution.
    patcher.set_alpha(1.0)
    patcher.capture_enabled = True

    rng = np.random.default_rng(seed)
    reps = {}
    for split in ["train", "val"]:
        n = len(ds[split]["y"])
        idx = np.arange(n)
        if max_samples and n > max_samples:
            idx = np.sort(rng.choice(idx, max_samples, replace=False))

        vals = make_values(ds[split], value_name, num_bins)
        store = {"attn_side": [], "ffn_raw": [], "ffn_unscaled_delta": [], "layer_out": [], "logits": []}

        for st in range(0, len(idx), batch_size):
            ids = idx[st:st+batch_size]
            t = torch.as_tensor(ds[split]["bin"][ids], dtype=torch.long, device=device)
            v = torch.as_tensor(vals[ids], dtype=torch.float32, device=device)
            out = model(t, v)
            logits = output_to_logits(out)
            if logits is not None:
                store["logits"].append(logits.detach().float().cpu().numpy())
            for k in ["attn_side", "ffn_raw", "ffn_unscaled_delta", "layer_out"]:
                if k not in patcher.last:
                    raise RuntimeError(f"patch did not capture {k}")
                store[k].append(tensor_to_cls_np(patcher.last[k]))
            log(f"{split} reps {min(st+batch_size, len(idx))}/{len(idx)}")

        reps[split] = {
            "idx": idx,
            "y": ds[split]["y"][idx],
            **{k: np.concatenate(v, axis=0) for k, v in store.items()},
        }

    patcher.capture_enabled = False
    return reps


def standardize_fit(X):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return mu, sd


def standardize_apply(X, mu, sd):
    return (X - mu) / sd


def centroid_info(Xtr, ytr):
    classes = np.unique(ytr)
    C = np.stack([Xtr[ytr == c].mean(axis=0) for c in classes], axis=0)
    return classes, C


def centroid_pred_and_margin(classes, C, X, y):
    d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
    pred = classes[d.argmin(axis=1)]
    pos = np.array([np.where(classes == yy)[0][0] for yy in y])
    td = d[np.arange(len(y)), pos]
    other = d.copy()
    other[np.arange(len(y)), pos] = np.inf
    margin = other.min(axis=1) - td
    nearest_other = classes[other.argmin(axis=1)]
    return pred, margin, nearest_other


def ffn_margin_audit(reps, preds_by_alpha, out_dir: Path, names: List[str]):
    # Compare attention-side vs layer_out in standardized train-fitted spaces.
    rows = []
    sample_rows = []
    stage_pair_rows = []

    for stage in ["attn_side", "layer_out"]:
        Xtr = reps["train"][stage].astype(np.float32)
        Xva = reps["val"][stage].astype(np.float32)
        ytr = reps["train"]["y"].astype(int)
        yva = reps["val"]["y"].astype(int)
        mu, sd = standardize_fit(Xtr)
        Xtr_s = standardize_apply(Xtr, mu, sd)
        Xva_s = standardize_apply(Xva, mu, sd)
        classes, C = centroid_info(Xtr_s, ytr)
        for split, X, y in [("train", Xtr_s, ytr), ("val", Xva_s, yva)]:
            pred, margin, nearest_other = centroid_pred_and_margin(classes, C, X, y)
            rows.append({
                "stage": stage,
                "split": split,
                "centroid_macro_f1": macro_f1(y, pred),
                "centroid_accuracy": acc(y, pred),
                "margin_mean": float(np.mean(margin)),
                "margin_median": float(np.median(margin)),
                "margin_p10": float(np.quantile(margin, 0.10)),
                "margin_frac_negative": float(np.mean(margin < 0)),
            })
            for cls_i, cls_name in enumerate(names):
                m = y == cls_i
                if m.any():
                    stage_pair_rows.append({
                        "stage": stage,
                        "split": split,
                        "group": "true_class",
                        "true": cls_name,
                        "pred_or_pair": "",
                        "n": int(m.sum()),
                        "margin_mean": float(np.mean(margin[m])),
                        "margin_median": float(np.median(margin[m])),
                        "margin_frac_negative": float(np.mean(margin[m] < 0)),
                    })

    metric_df = pd.DataFrame(rows)
    metric_df.to_csv(out_dir / "F1e0_attn_vs_layerout_centroid_margin_metrics.csv", index=False)

    # Delta: layer_out margin - attn_side margin, using separate stage standardizations.
    delta_rows = []
    split_arrays = {}
    for split in ["train", "val"]:
        y = reps[split]["y"].astype(int)
        per_stage_margin = {}
        per_stage_pred = {}
        for stage in ["attn_side", "layer_out"]:
            Xtr = reps["train"][stage].astype(np.float32)
            X = reps[split][stage].astype(np.float32)
            ytr = reps["train"]["y"].astype(int)
            mu, sd = standardize_fit(Xtr)
            Xtr_s = standardize_apply(Xtr, mu, sd)
            X_s = standardize_apply(X, mu, sd)
            classes, C = centroid_info(Xtr_s, ytr)
            pred, margin, nearest_other = centroid_pred_and_margin(classes, C, X_s, y)
            per_stage_margin[stage] = margin
            per_stage_pred[stage] = pred
        dmargin = per_stage_margin["layer_out"] - per_stage_margin["attn_side"]
        split_arrays[split] = {
            "delta_margin_layerout_minus_attn": dmargin,
            "attn_margin": per_stage_margin["attn_side"],
            "layerout_margin": per_stage_margin["layer_out"],
            "attn_centroid_pred": per_stage_pred["attn_side"],
            "layerout_centroid_pred": per_stage_pred["layer_out"],
        }
        delta_rows.append({
            "split": split,
            "n": int(len(y)),
            "delta_margin_mean": float(np.mean(dmargin)),
            "delta_margin_median": float(np.median(dmargin)),
            "delta_margin_p10": float(np.quantile(dmargin, 0.10)),
            "frac_ffn_improves_margin": float(np.mean(dmargin > 0)),
            "frac_ffn_hurts_margin": float(np.mean(dmargin < 0)),
            "attn_centroid_macro_f1": macro_f1(y, per_stage_pred["attn_side"]),
            "layerout_centroid_macro_f1": macro_f1(y, per_stage_pred["layer_out"]),
            "delta_centroid_macro_f1": macro_f1(y, per_stage_pred["layer_out"]) - macro_f1(y, per_stage_pred["attn_side"]),
        })

        # By true class.
        for cls_i, cls_name in enumerate(names):
            m = y == cls_i
            if m.any():
                delta_rows.append({
                    "split": split,
                    "group": f"class:{cls_name}",
                    "n": int(m.sum()),
                    "delta_margin_mean": float(np.mean(dmargin[m])),
                    "delta_margin_median": float(np.median(dmargin[m])),
                    "delta_margin_p10": float(np.quantile(dmargin[m], 0.10)),
                    "frac_ffn_improves_margin": float(np.mean(dmargin[m] > 0)),
                    "frac_ffn_hurts_margin": float(np.mean(dmargin[m] < 0)),
                    "attn_centroid_macro_f1": np.nan,
                    "layerout_centroid_macro_f1": np.nan,
                    "delta_centroid_macro_f1": np.nan,
                })

    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(out_dir / "F1e0_ffn_delta_margin_summary.csv", index=False)

    # Val sample-level: use original alpha=1 predictions to label correct/wrong/pair.
    if ("val", 1.0) in preds_by_alpha:
        yv = reps["val"]["y"].astype(int)
        p_orig = preds_by_alpha[("val", 1.0)][reps["val"]["idx"]]
        correct = p_orig == yv
        arr = split_arrays["val"]
        for i in range(len(yv)):
            sample_rows.append({
                "sample_idx": int(reps["val"]["idx"][i]),
                "y_true": int(yv[i]),
                "true_name": names[int(yv[i])],
                "orig_pred": int(p_orig[i]),
                "orig_pred_name": names[int(p_orig[i])],
                "orig_correct": bool(correct[i]),
                "attn_margin": float(arr["attn_margin"][i]),
                "layerout_margin": float(arr["layerout_margin"][i]),
                "delta_margin_layerout_minus_attn": float(arr["delta_margin_layerout_minus_attn"][i]),
                "ffn_improves_true_margin": bool(arr["delta_margin_layerout_minus_attn"][i] > 0),
                "attn_centroid_pred": int(arr["attn_centroid_pred"][i]),
                "attn_centroid_pred_name": names[int(arr["attn_centroid_pred"][i])],
                "layerout_centroid_pred": int(arr["layerout_centroid_pred"][i]),
                "layerout_centroid_pred_name": names[int(arr["layerout_centroid_pred"][i])],
            })
        pd.DataFrame(sample_rows).to_csv(out_dir / "F1e0_val_ffn_delta_margin_per_sample.csv", index=False)

        # By original confusion pair.
        pr = pd.DataFrame(sample_rows)
        pair_sum = pr.groupby(["true_name", "orig_pred_name", "orig_correct"]).agg(
            n=("sample_idx", "count"),
            delta_margin_mean=("delta_margin_layerout_minus_attn", "mean"),
            delta_margin_median=("delta_margin_layerout_minus_attn", "median"),
            frac_ffn_improves_true_margin=("ffn_improves_true_margin", "mean"),
            attn_margin_mean=("attn_margin", "mean"),
            layerout_margin_mean=("layerout_margin", "mean"),
        ).reset_index().sort_values(["orig_correct", "n"], ascending=[True, False])
        pair_sum.to_csv(out_dir / "F1e0_val_ffn_delta_by_original_pair.csv", index=False)

    return metric_df, delta_df


def ffn_norm_audit(reps, out_dir: Path, names: List[str]):
    rows = []
    for split in ["train", "val"]:
        y = reps[split]["y"].astype(int)
        attn = reps[split]["attn_side"].astype(np.float32)
        delta = reps[split]["ffn_unscaled_delta"].astype(np.float32)
        out = reps[split]["layer_out"].astype(np.float32)

        delta_norm = np.linalg.norm(delta, axis=1)
        attn_norm = np.linalg.norm(attn, axis=1)
        out_norm = np.linalg.norm(out, axis=1)
        ratio = delta_norm / np.maximum(attn_norm, 1e-8)
        cos = np.sum(attn * delta, axis=1) / np.maximum(np.linalg.norm(attn, axis=1) * np.linalg.norm(delta, axis=1), 1e-8)

        def add(group, mask):
            if mask.any():
                rows.append({
                    "split": split,
                    "group": group,
                    "n": int(mask.sum()),
                    "delta_norm_mean": float(delta_norm[mask].mean()),
                    "delta_norm_median": float(np.median(delta_norm[mask])),
                    "attn_norm_mean": float(attn_norm[mask].mean()),
                    "out_norm_mean": float(out_norm[mask].mean()),
                    "delta_to_attn_norm_ratio_mean": float(ratio[mask].mean()),
                    "cos_attn_delta_mean": float(cos[mask].mean()),
                })

        add("all", np.ones(len(y), dtype=bool))
        for i, name in enumerate(names):
            add(f"class:{name}", y == i)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "F1e0_ffn_delta_norm_summary.csv", index=False)
    return df


def write_report(out_dir: Path, model_info: Dict[str, Any], candidate_df: pd.DataFrame, value_name: str,
                 sweep_df: pd.DataFrame, fix_df: pd.DataFrame, pair_df: pd.DataFrame,
                 margin_df: pd.DataFrame, delta_df: pd.DataFrame, norm_df: pd.DataFrame):
    lines = []
    lines.append("# F1e0 L1 FFN Contribution Audit Report\n")
    lines.append("## Question\n")
    lines.append("```text")
    lines.append("F1d showed L1 gap grows around FFN/linear2/layer0.")
    lines.append("F1e0 asks whether FFN is actually harmful, helpful, or mixed before training any FFN-control model.")
    lines.append("No model is trained here.")
    lines.append("```")

    lines.append("\n## Checkpoint and value candidate\n")
    lines.append("```text")
    lines.append(f"checkpoint = {model_info.get('checkpoint_path')}")
    lines.append(f"model kwargs = {model_info.get('kwargs')}")
    lines.append(f"selected value candidate = {value_name}")
    lines.append("```")
    lines.append("\nCandidate reproduction:")
    lines.append(candidate_df.to_markdown(index=False))

    lines.append("\n## Frozen FFN alpha sweep\n")
    cols = [c for c in [
        "alpha",
        "train_macro_f1", "val_macro_f1", "gap_macro_f1",
        "delta_train_macro_f1_vs_alpha1", "delta_val_macro_f1_vs_alpha1", "delta_gap_vs_alpha1",
        "train_accuracy", "val_accuracy",
        "keeps_train_high_ge_0p900", "beats_alpha1_val", "reduces_gap",
    ] if c in sweep_df.columns]
    lines.append(sweep_df[cols].to_markdown(index=False))

    best = None
    if len(sweep_df):
        best = sweep_df.sort_values(["val_macro_f1", "gap_macro_f1"], ascending=[False, True]).iloc[0]
        lines.append("\nBest alpha by val macro-F1:")
        lines.append("```text")
        lines.append(f"alpha = {best['alpha']}")
        lines.append(f"train_macro_f1 = {best['train_macro_f1']:.6f}")
        lines.append(f"val_macro_f1 = {best['val_macro_f1']:.6f}")
        lines.append(f"gap_macro_f1 = {best['gap_macro_f1']:.6f}")
        lines.append(f"delta_val_vs_alpha1 = {best.get('delta_val_macro_f1_vs_alpha1', np.nan):+.6f}")
        lines.append(f"delta_gap_vs_alpha1 = {best.get('delta_gap_vs_alpha1', np.nan):+.6f}")
        lines.append("```")

    lines.append("\n## Val fix/damage relative to alpha=1.0\n")
    if len(fix_df):
        lines.append(fix_df.to_markdown(index=False))
    else:
        lines.append("No fix/damage table produced.")

    lines.append("\n## Attention-side vs layerout centroid margin\n")
    if len(margin_df):
        lines.append(margin_df.to_markdown(index=False))
    else:
        lines.append("No margin metrics produced.")

    lines.append("\n## FFN delta margin summary\n")
    if len(delta_df):
        lines.append(delta_df.to_markdown(index=False))
    else:
        lines.append("No FFN delta margin table produced.")

    lines.append("\n## FFN delta norm summary\n")
    if len(norm_df):
        show = norm_df.head(20)
        lines.append(show.to_markdown(index=False))
    else:
        lines.append("No norm table produced.")

    lines.append("\n## Decision logic\n")
    lines.append("```text")
    lines.append("If alpha < 1 increases val and keeps train high:")
    lines.append("  There is evidence FFN contribution is too strong; then train a gated/controlled FFN model.")
    lines.append("")
    lines.append("If alpha < 1 decreases val:")
    lines.append("  FFN carries necessary signal; do not globally shrink FFN.")
    lines.append("")
    lines.append("If alpha < 1 fixes some pairs but damages others:")
    lines.append("  Global FFN gate is too coarse; inspect pair-specific/conditional behavior.")
    lines.append("")
    lines.append("If attention->layerout improves train much more than val:")
    lines.append("  FFN is partially train-specific; consider softer representation control only if alpha sweep supports it.")
    lines.append("```")

    lines.append("\n## Automatic call\n")
    call = "inconclusive"
    if best is not None and abs(float(best["alpha"]) - 1.0) > 1e-9:
        if float(best.get("val_macro_f1", -1)) > float(sweep_df.loc[np.isclose(sweep_df["alpha"], 1.0), "val_macro_f1"].iloc[0]) and float(best.get("train_macro_f1", 0)) >= 0.900:
            call = "alpha_less_or_greater_than_1_improves_val_on_frozen_checkpoint_check_direction_before_training"
    if len(sweep_df) and any(np.isclose(sweep_df["alpha"], 1.0)):
        base = sweep_df.loc[np.isclose(sweep_df["alpha"], 1.0)].iloc[0]
        lower = sweep_df[sweep_df["alpha"] < 1.0]
        if len(lower):
            best_lower = lower.sort_values(["val_macro_f1", "gap_macro_f1"], ascending=[False, True]).iloc[0]
            if best_lower["val_macro_f1"] > base["val_macro_f1"] and best_lower["train_macro_f1"] >= 0.900:
                call = "evidence_for_ffn_shrink_or_gate"
            elif best_lower["val_macro_f1"] < base["val_macro_f1"]:
                call = "no_evidence_for_global_ffn_shrink_alpha_lower_than_1_hurts_val"
            elif best_lower["train_macro_f1"] < 0.900:
                call = "ffn_shrink_underfits_train"
    lines.append("```text")
    lines.append(f"automatic_call = {call}")
    lines.append("```")

    (out_dir / "F1e0_l1_ffn_contribution_report.md").write_text("\n".join(lines), encoding="utf-8")


def zip_dir(src: Path, dst: Path):
    if dst.exists():
        dst.unlink()
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for p in src.rglob("*"):
            if p.is_file() and p != dst:
                z.write(p, p.relative_to(src.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l1-run-dir", default="05_test/outputs/F1a2_stage2_depth_classifier/Keff512/F1a2_L1_reduce_num_layers_strong")
    ap.add_argument("--dataset-npz", default="03_outputs/05_dataset/dataset.npz")
    ap.add_argument("--train-raw", default="01_split/train_raw.csv")
    ap.add_argument("--val-raw", default="01_split/val_raw.csv")
    ap.add_argument("--out-dir", default="05_test/outputs/F1e0_l1_ffn_contribution_audit")
    ap.add_argument("--combined-zip", default="05_test/outputs/F1e0_l1_ffn_contribution_audit.zip")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--alphas", default="0,0.25,0.5,0.75,1.0,1.25")
    ap.add_argument("--candidate-eval-samples", type=int, default=0)
    ap.add_argument("--max-rep-samples-per-split", type=int, default=0, help="0=full train/val")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trust-local-checkpoint", action="store_true")
    args = ap.parse_args()

    root = repo_root_from_here()
    out_dir = resolve_path(args.out_dir, root)
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_zip = resolve_path(args.combined_zip, root)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    l1_dir = resolve_path(args.l1_run_dir, root)
    if not l1_dir.exists():
        raise FileNotFoundError(f"L1 run dir not found: {l1_dir}")
    if find_checkpoint(l1_dir) is None:
        raise FileNotFoundError(f"L1 checkpoint not found in: {l1_dir}")

    log(f"root={root}")
    log(f"device={device}")
    log(f"l1_dir={l1_dir}")
    log(f"out_dir={out_dir}")
    log("No training. Frozen checkpoint FFN contribution audit only.")

    ds, ds_info = load_dataset(
        resolve_path(args.dataset_npz, root),
        resolve_path(args.train_raw, root),
        resolve_path(args.val_raw, root),
    )
    (out_dir / "dataset_info_f1e0.json").write_text(json.dumps(ds_info, indent=2, default=str), encoding="utf-8")

    model, model_info = build_model(root, l1_dir, ds_info, ds["train"]["y"], device, args.trust_local_checkpoint)
    (out_dir / "model_load_info_f1e0.json").write_text(json.dumps(model_info, indent=2, default=str), encoding="utf-8")

    expected = None
    diag = safe_json(l1_dir / "diagnosis_summary.json")
    try:
        expected = float(diag.get("val", {}).get("macro_f1"))
    except Exception:
        expected = L1_REF["val_macro_f1"]

    value_name, cand_df = choose_value_candidate(
        model=model,
        ds=ds,
        num_bins=ds_info["num_bins"],
        device=device,
        batch_size=args.batch_size,
        expected_val_macro=expected,
        max_eval_samples=args.candidate_eval_samples,
        seed=args.seed,
    )
    cand_df.to_csv(out_dir / "F1e0_value_candidate_selection.csv", index=False)
    log(f"selected value candidate={value_name}")

    layer0 = get_l1_layer(model)
    patcher = FfnAlphaPatch(layer0, alpha=1.0)

    alphas = parse_alphas(args.alphas)
    if not any(abs(a - 1.0) < 1e-9 for a in alphas):
        alphas.append(1.0)
        alphas = sorted(alphas)

    names = class_names(int(len(np.unique(ds["train"]["y"]))))

    try:
        sweep_df, preds_by_alpha, probs_by_alpha = alpha_sweep(
            model=model,
            patcher=patcher,
            ds=ds,
            value_name=value_name,
            num_bins=ds_info["num_bins"],
            alphas=alphas,
            device=device,
            batch_size=args.batch_size,
            out_dir=out_dir,
            names=names,
        )

        fix_df, pair_df = fix_damage_analysis(ds, preds_by_alpha, probs_by_alpha, alphas, out_dir, names)

        reps = extract_attn_ffn_reps(
            model=model,
            patcher=patcher,
            ds=ds,
            value_name=value_name,
            num_bins=ds_info["num_bins"],
            device=device,
            batch_size=args.batch_size,
            max_samples=args.max_rep_samples_per_split,
            seed=args.seed,
        )
        # Save compact representation metadata only.
        rep_info = {
            split: {k: (list(v.shape) if hasattr(v, "shape") else None) for k, v in d.items() if k not in {"y", "idx"}}
            for split, d in reps.items()
        }
        (out_dir / "F1e0_rep_capture_info.json").write_text(json.dumps(rep_info, indent=2), encoding="utf-8")

        margin_df, delta_df = ffn_margin_audit(reps, preds_by_alpha, out_dir, names)
        norm_df = ffn_norm_audit(reps, out_dir, names)

        write_report(
            out_dir=out_dir,
            model_info=model_info,
            candidate_df=cand_df,
            value_name=value_name,
            sweep_df=sweep_df,
            fix_df=fix_df,
            pair_df=pair_df,
            margin_df=margin_df,
            delta_df=delta_df,
            norm_df=norm_df,
        )
    finally:
        patcher.restore()

    zip_dir(out_dir, combined_zip)
    log(f"zip={combined_zip}")
    log("DONE")


if __name__ == "__main__":
    main()
