#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F1d L1 Residual Overfit Audit

Question
--------
After F1b showed that base L=3 develops a large gap in deeper Transformer layers,
L1 (num_layers=1) improved val and reduced gap slightly. But L1 still has:

    train macro-F1 ≈ 0.911431
    val macro-F1   ≈ 0.814224
    gap            ≈ 0.097207

So the correct next question is:

    Where does the remaining L1 gap come from?

This script audits base L=3 and L1 with the SAME stage-probe method:
- embedding
- input_proj
- encoder.layers.0
- CLASSIFIER_PRE_INPUT_CLS
- MODEL_OUTPUT_LOGITS

It does not train any new variant.

Important
---------
For L1 audit to work, a L1 run directory with checkpoint must exist.
By default the script auto-searches 05_test/outputs for a run whose config has
num_layers=1 and has a best_model.pt/checkpoint. You can also pass --l1-run-dir.

Outputs
-------
- static_run_summary_f1d.csv
- stage_probe_metrics_f1d.csv
- stage_centroid_margin_metrics_f1d.csv
- *_value_candidate_selection_f1d.csv
- *_final_direct_report_f1d.json
- F1d_l1_residual_overfit_report.md
- combined zip
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
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


BASE512_REF = {"train_macro_f1": 0.910253, "val_macro_f1": 0.810094, "gap_macro_f1": 0.100158}
L1_REF = {"train_macro_f1": 0.911431, "val_macro_f1": 0.814224, "gap_macro_f1": 0.097207}


def log(msg: str) -> None:
    print(f"[F1d] {msg}", flush=True)


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


def maybe_get(d: Dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def find_checkpoint(run_dir: Path) -> Optional[Path]:
    for n in ["best_model.pt", "model_best.pt", "checkpoint_best.pt", "best.pt", "checkpoint.pt", "model.pt"]:
        p = run_dir / n
        if p.exists():
            return p
    pts = sorted(run_dir.glob("*.pt"))
    return pts[0] if pts else None


def cfg_get(cfg: Dict[str, Any], k: str, default):
    if k in cfg:
        return cfg[k]
    for dname in ["model", "model_config", "training", "args"]:
        d = cfg.get(dname, {})
        if isinstance(d, dict) and k in d:
            return d[k]
    return default


def cfg_num_layers(cfg: Dict[str, Any]) -> Optional[int]:
    for dname in ["model", "model_config"]:
        d = cfg.get(dname, {})
        if isinstance(d, dict) and "num_layers" in d:
            try:
                return int(d["num_layers"])
            except Exception:
                pass
    if "num_layers" in cfg:
        try:
            return int(cfg["num_layers"])
        except Exception:
            pass
    return None


def auto_find_l1_run(root: Path, search_root: Path) -> Optional[Path]:
    candidates = []
    if not search_root.exists():
        return None

    for diag_path in search_root.rglob("diagnosis_summary.json"):
        run_dir = diag_path.parent
        ckpt = find_checkpoint(run_dir)
        if ckpt is None:
            continue
        cfg = safe_json(run_dir / "config.json")
        diag = safe_json(diag_path)

        nl = cfg_num_layers(cfg)
        name_score = 0
        text = " ".join([
            run_dir.name,
            str(diag.get("variant", "")),
            str(diag.get("run_name", "")),
            str(cfg.get("run_name", "")),
        ]).lower()
        if "l1" in text:
            name_score += 10
        if "num_layers" in text and "1" in text:
            name_score += 2

        is_l1 = (nl == 1) or ("l1" in text and "layer" in text)
        if not is_l1:
            continue

        val = maybe_get(diag, "val", "macro_f1", default=None)
        try:
            val = float(val) if val is not None else -1.0
        except Exception:
            val = -1.0
        # Prefer exact known F1a2 L1, then best val.
        known_score = 100 if "F1a2_L1_reduce_num_layers_strong" in str(run_dir) else 0
        candidates.append((known_score + name_score, val, run_dir))

    if not candidates:
        return None
    candidates.sort(key=lambda x: (-x[0], -x[1], str(x[2])))
    return candidates[0][2]


def flatten_f1(prefix: str, rep: Dict[str, Any]) -> Dict[str, float]:
    out = {}
    if not isinstance(rep, dict):
        return out
    items = []
    if "per_class" in rep and isinstance(rep["per_class"], dict):
        items = list(rep["per_class"].items())
    else:
        for k, v in rep.items():
            if isinstance(v, dict) and ("f1" in v or "f1-score" in v):
                if str(k).lower() in {"accuracy", "macro avg", "weighted avg"}:
                    continue
                items.append((k, v))
    for k, v in items:
        f1 = v.get("f1", v.get("f1-score"))
        if f1 is not None:
            out[f"{prefix}_f1_{str(k).replace(' ', '_')}"] = float(f1)
    return out


def static_run(alias: str, run_dir: Path) -> Dict[str, Any]:
    diag = safe_json(run_dir / "diagnosis_summary.json")
    cfg = safe_json(run_dir / "config.json")
    train_rep = safe_json(run_dir / "train_classification_report_best.json")
    val_rep = safe_json(run_dir / "val_classification_report_best.json")
    mc = cfg.get("model", cfg.get("model_config", {}))
    if not isinstance(mc, dict):
        mc = {}
    row = {
        "alias": alias,
        "run_dir": str(run_dir),
        "exists": run_dir.exists(),
        "has_checkpoint": find_checkpoint(run_dir) is not None,
        "checkpoint": None if find_checkpoint(run_dir) is None else str(find_checkpoint(run_dir)),
        "best_epoch": diag.get("best_epoch"),
        "train_macro_f1": maybe_get(diag, "train", "macro_f1"),
        "val_macro_f1": maybe_get(diag, "val", "macro_f1"),
        "gap_macro_f1": diag.get("generalization_gap_macro_f1"),
        "train_acc": maybe_get(diag, "train", "accuracy"),
        "val_acc": maybe_get(diag, "val", "accuracy"),
        "num_layers": mc.get("num_layers", cfg.get("num_layers")),
        "hidden_dim": mc.get("hidden_dim", cfg.get("hidden_dim")),
        "num_heads": mc.get("num_heads", cfg.get("num_heads")),
        "classifier_hidden_dim": mc.get("classifier_hidden_dim", cfg.get("classifier_hidden_dim")),
        "dropout": mc.get("dropout", cfg.get("dropout")),
        "classifier_dropout": mc.get("classifier_dropout", cfg.get("classifier_dropout")),
    }
    row.update(flatten_f1("train", train_rep))
    row.update(flatten_f1("val", val_rep))
    return row


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


def infer_model_config(cfg: Dict[str, Any], ds_info: Dict[str, Any], y_train: np.ndarray):
    return {
        "num_bins": int(cfg_get(cfg, "num_bins", cfg_get(cfg, "K", ds_info["num_bins"]))),
        "n_features": int(cfg_get(cfg, "n_features", cfg_get(cfg, "num_features", ds_info["n_features"]))),
        "num_classes": int(cfg_get(cfg, "num_classes", len(np.unique(y_train)))),
        "value_dim": int(cfg_get(cfg, "value_dim", 32)),
        "feature_dim": int(cfg_get(cfg, "feature_dim", 32)),
        "hidden_dim": int(cfg_get(cfg, "hidden_dim", 128)),
        "num_layers": int(cfg_get(cfg, "num_layers", 3)),
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

    mod = load_module_from_path(f"_f1d_model_06_model_{abs(hash(str(run_dir)))%100000}", root / "02_src" / "06_model.py")
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


def forward_logits(model, tokens_np, values_np, idx, device, batch_size):
    outs = []
    for st in range(0, len(idx), batch_size):
        ids = idx[st:st+batch_size]
        t = torch.as_tensor(tokens_np[ids], dtype=torch.long, device=device)
        v = torch.as_tensor(values_np[ids], dtype=torch.float32, device=device)
        with torch.no_grad():
            out = model(t, v)
        logits = output_to_logits(out)
        if logits is None:
            raise RuntimeError("model output did not contain logits")
        outs.append(logits.detach().float().cpu().numpy())
    return np.concatenate(outs, axis=0)


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
            logits = forward_logits(model, ds["val"]["bin"], vals, idx, device, batch_size)
            pred = logits.argmax(axis=1)
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


def model_modules(model):
    rows = []
    for name, m in model.named_modules():
        if name:
            rows.append({
                "name": name,
                "class": m.__class__.__name__,
                "children": len(list(m.children())),
                "params_direct": sum(p.numel() for p in m.parameters(recurse=False)),
                "params_recursive": sum(p.numel() for p in m.parameters(recurse=True)),
            })
    return pd.DataFrame(rows)


def choose_hooks(model, max_hooks: int):
    priority = []
    for name, m in model.named_modules():
        if not name:
            continue
        lname = name.lower()
        score = 0
        if name == "embedding": score += 200
        if name == "input_proj": score += 190
        if name.startswith("encoder.layers."):
            # Hook full layer modules first, not every nested linear unless room remains.
            parts = name.split(".")
            if len(parts) == 3:
                score += 180
            elif "self_attn" in lname:
                score += 80
            elif "linear1" in lname or "linear2" in lname:
                score += 20
        if name == "encoder.norm": score += 170
        if name == "classifier": score += 160
        if "dropout" in lname or "activation" in lname:
            score -= 100
        if score > 0:
            priority.append((score, name, m))
    priority.sort(key=lambda x: (-x[0], x[1]))
    return [(n, m) for _, n, m in priority[:max_hooks]]


def rep_from_output(x):
    if isinstance(x, (tuple, list)):
        for v in x:
            got = rep_from_output(v)
            if got is not None:
                return got
        return None
    if isinstance(x, dict):
        for v in x.values():
            got = rep_from_output(v)
            if got is not None:
                return got
        return None
    if not torch.is_tensor(x):
        return None
    t = x.detach().float()
    if t.ndim == 2:
        return t.cpu().numpy()
    if t.ndim == 3:
        # model is batch_first in this pipeline; CLS token at index 0.
        return t[:, 0, :].cpu().numpy()
    if t.ndim == 1:
        return t.reshape(-1, 1).cpu().numpy()
    b = t.shape[0]
    return t.reshape(b, -1).mean(dim=1, keepdim=True).cpu().numpy()


def extract_reps(model, ds, value_name, num_bins, device, batch_size, max_samples, max_hooks, seed):
    rng = np.random.default_rng(seed)
    idxs = {}
    for split in ["train", "val"]:
        n = len(ds[split]["y"])
        idx = np.arange(n)
        if max_samples and n > max_samples:
            idx = np.sort(rng.choice(idx, max_samples, replace=False))
        idxs[split] = idx

    hooks = choose_hooks(model, max_hooks)
    captured = {}
    handles = []

    def mk_hook(name):
        def hook(mod, inp, out):
            captured[name] = out
        return hook

    def cls_pre_hook(mod, inp):
        if inp and torch.is_tensor(inp[0]):
            captured["CLASSIFIER_PRE_INPUT_CLS"] = inp[0]

    for name, m in hooks:
        try:
            handles.append(m.register_forward_hook(mk_hook(name)))
        except Exception:
            pass
    if hasattr(model, "classifier"):
        try:
            handles.append(model.classifier.register_forward_pre_hook(cls_pre_hook))
        except Exception:
            pass

    store = {}
    for split in ["train", "val"]:
        vals = make_values(ds[split], value_name, num_bins)
        idx = idxs[split]
        for st in range(0, len(idx), batch_size):
            ids = idx[st:st+batch_size]
            captured.clear()
            t = torch.as_tensor(ds[split]["bin"][ids], dtype=torch.long, device=device)
            v = torch.as_tensor(vals[ids], dtype=torch.float32, device=device)
            with torch.no_grad():
                out = model(t, v)
            logits = output_to_logits(out)
            if logits is not None:
                store.setdefault("MODEL_OUTPUT_LOGITS", {}).setdefault(split, []).append(logits.detach().float().cpu().numpy())
            for name, obj in list(captured.items()):
                arr = rep_from_output(obj)
                if arr is not None and arr.shape[0] == len(ids):
                    store.setdefault(name, {}).setdefault(split, []).append(arr)
            log(f"{split} extracted {min(st+batch_size, len(idx))}/{len(idx)}")

    for h in handles:
        try:
            h.remove()
        except Exception:
            pass

    reps = {}
    for stage, d in store.items():
        if "train" in d and "val" in d:
            reps[stage] = {
                "Xtr": np.concatenate(d["train"], axis=0),
                "Xva": np.concatenate(d["val"], axis=0),
                "ytr": ds["train"]["y"][idxs["train"]],
                "yva": ds["val"]["y"][idxs["val"]],
            }
    info = {
        "value_candidate": value_name,
        "hooked_modules": [{"name": n, "class": m.__class__.__name__} for n, m in hooks],
        "n_train_used": int(len(idxs["train"])),
        "n_val_used": int(len(idxs["val"])),
        "num_stages": int(len(reps)),
    }
    return reps, info


def standardize(Xtr, Xva):
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return (Xtr - mu) / sd, (Xva - mu) / sd


def centroid_pred(Xtr, ytr, X):
    classes = np.unique(ytr)
    C = np.stack([Xtr[ytr == c].mean(axis=0) for c in classes], axis=0)
    d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
    return classes[d.argmin(axis=1)]


def centroid_margin(Xtr, ytr, X, y):
    classes = np.unique(ytr)
    C = np.stack([Xtr[ytr == c].mean(axis=0) for c in classes], axis=0)
    d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
    pos = np.array([np.where(classes == yy)[0][0] for yy in y])
    td = d[np.arange(len(y)), pos]
    other = d.copy()
    other[np.arange(len(y)), pos] = np.inf
    margin = other.min(axis=1) - td
    return {
        "margin_mean": float(np.mean(margin)),
        "margin_median": float(np.median(margin)),
        "margin_p10": float(np.quantile(margin, 0.10)),
        "margin_frac_negative": float(np.mean(margin < 0)),
    }


def eval_stage(alias: str, stage: str, rep: Dict[str, np.ndarray], seed: int, max_probe_train: int):
    Xtr = rep["Xtr"].reshape(len(rep["ytr"]), -1).astype(np.float32)
    Xva = rep["Xva"].reshape(len(rep["yva"]), -1).astype(np.float32)
    ytr = rep["ytr"].astype(int)
    yva = rep["yva"].astype(int)
    Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=1e6, neginf=-1e6)
    Xva = np.nan_to_num(Xva, nan=0.0, posinf=1e6, neginf=-1e6)
    Xtr_s, Xva_s = standardize(Xtr, Xva)

    row = {"run_alias": alias, "stage": stage, "dim": int(Xtr_s.shape[1]), "n_train": len(ytr), "n_val": len(yva)}
    if "LOGITS" in stage.upper():
        ptr = Xtr.argmax(axis=1)
        pva = Xva.argmax(axis=1)
        row.update({
            "direct_train_macro_f1": macro_f1(ytr, ptr),
            "direct_val_macro_f1": macro_f1(yva, pva),
            "direct_gap_macro_f1": macro_f1(ytr, ptr) - macro_f1(yva, pva),
            "direct_train_accuracy": acc(ytr, ptr),
            "direct_val_accuracy": acc(yva, pva),
            "direct_train_weighted_f1": weighted_f1(ytr, ptr),
            "direct_val_weighted_f1": weighted_f1(yva, pva),
        })

    try:
        ptr = centroid_pred(Xtr_s, ytr, Xtr_s)
        pva = centroid_pred(Xtr_s, ytr, Xva_s)
        row.update({
            "centroid_train_macro_f1": macro_f1(ytr, ptr),
            "centroid_val_macro_f1": macro_f1(yva, pva),
            "centroid_gap_macro_f1": macro_f1(ytr, ptr) - macro_f1(yva, pva),
            "centroid_train_accuracy": acc(ytr, ptr),
            "centroid_val_accuracy": acc(yva, pva),
        })
    except Exception:
        row["centroid_error"] = traceback.format_exc()

    margin = {"run_alias": alias, "stage": stage}
    try:
        mt = centroid_margin(Xtr_s, ytr, Xtr_s, ytr)
        mv = centroid_margin(Xtr_s, ytr, Xva_s, yva)
        margin.update({f"train_{k}": v for k, v in mt.items()})
        margin.update({f"val_{k}": v for k, v in mv.items()})
        margin["margin_mean_gap_train_minus_val"] = margin["train_margin_mean"] - margin["val_margin_mean"]
        margin["margin_neg_frac_gap_val_minus_train"] = margin["val_margin_frac_negative"] - margin["train_margin_frac_negative"]
    except Exception:
        margin["margin_error"] = traceback.format_exc()

    try:
        rng = np.random.default_rng(seed)
        ids = np.arange(len(ytr))
        if max_probe_train and len(ids) > max_probe_train:
            ids = rng.choice(ids, max_probe_train, replace=False)
        clf = make_pipeline(
            StandardScaler(),
            SGDClassifier(
                loss="log_loss",
                alpha=1e-4,
                class_weight="balanced",
                max_iter=1000,
                tol=1e-4,
                early_stopping=True,
                random_state=seed,
                n_jobs=-1,
            )
        )
        clf.fit(Xtr[ids], ytr[ids])
        ptr = clf.predict(Xtr)
        pva = clf.predict(Xva)
        row.update({
            "linear_train_macro_f1": macro_f1(ytr, ptr),
            "linear_val_macro_f1": macro_f1(yva, pva),
            "linear_gap_macro_f1": macro_f1(ytr, ptr) - macro_f1(yva, pva),
            "linear_train_accuracy": acc(ytr, ptr),
            "linear_val_accuracy": acc(yva, pva),
            "linear_train_weighted_f1": weighted_f1(ytr, ptr),
            "linear_val_weighted_f1": weighted_f1(yva, pva),
        })
    except Exception:
        row["linear_error"] = traceback.format_exc()

    return row, margin


def save_direct_final_report(out_dir: Path, alias: str, y_true: np.ndarray, probs: np.ndarray):
    pred = probs.argmax(axis=1)
    num_classes = probs.shape[1]
    names = ["Benign", "Ransomware", "Spyware", "Trojan"] if num_classes == 4 else [str(i) for i in range(num_classes)]
    rep = classification_report(
        y_true, pred,
        labels=list(range(num_classes)),
        target_names=names,
        output_dict=True,
        zero_division=0,
    )
    (out_dir / f"{alias}_final_direct_report_f1d.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    pd.DataFrame(confusion_matrix(y_true, pred, labels=list(range(num_classes))), index=names, columns=names).to_csv(
        out_dir / f"{alias}_final_direct_confusion_f1d.csv"
    )


def stage_metric(df: pd.DataFrame, alias: str, pattern: str, metric: str) -> float:
    g = df[df["run_alias"] == alias]
    m = g["stage"].str.contains(pattern, case=False, regex=True, na=False)
    sub = g[m]
    if len(sub) == 0 or metric not in sub.columns or sub[metric].dropna().empty:
        return np.nan
    return float(sub[metric].dropna().max())


def write_report(out_dir: Path, static_df: pd.DataFrame, stage_df: pd.DataFrame, margin_df: pd.DataFrame, infos: List[Dict[str, Any]]):
    lines = []
    lines.append("# F1d L1 Residual Overfit Audit Report\n")
    lines.append("## Question\n")
    lines.append("```text")
    lines.append("L1 removed deeper layers, but L1 still has a large train-val gap.")
    lines.append("This audit checks where the remaining L1 gap appears.")
    lines.append("```")

    lines.append("\n## References\n")
    lines.append("```text")
    lines.append(f"Base512 train macro-F1 = {BASE512_REF['train_macro_f1']:.6f}")
    lines.append(f"Base512 val macro-F1   = {BASE512_REF['val_macro_f1']:.6f}")
    lines.append(f"Base512 gap            = {BASE512_REF['gap_macro_f1']:.6f}")
    lines.append(f"L1 train macro-F1      = {L1_REF['train_macro_f1']:.6f}")
    lines.append(f"L1 val macro-F1        = {L1_REF['val_macro_f1']:.6f}")
    lines.append(f"L1 gap                 = {L1_REF['gap_macro_f1']:.6f}")
    lines.append("```")

    lines.append("\n## Static run summary\n")
    if len(static_df):
        cols = [c for c in ["alias", "exists", "has_checkpoint", "train_macro_f1", "val_macro_f1", "gap_macro_f1", "num_layers", "hidden_dim", "num_heads", "classifier_hidden_dim", "run_dir"] if c in static_df.columns]
        lines.append(static_df[cols].to_markdown(index=False))

    lines.append("\n## Value candidate reproduction check\n")
    for info in infos:
        alias = info.get("alias")
        if info.get("error"):
            lines.append(f"- {alias}: ERROR `{info.get('error')}`")
        else:
            lines.append(f"- {alias}: selected `{info.get('selected_value_candidate')}`")
            lines.append(f"  - expected val macro-F1: {info.get('expected_val_macro_f1')}")
            lines.append(f"  - candidate table: `{info.get('candidate_table')}`")

    if len(stage_df) == 0:
        lines.append("\n## Dynamic stage audit unavailable\n")
        lines.append("No stage metrics were produced. Read `dynamic_audit_info_f1d.json`.")
        (out_dir / "F1d_l1_residual_overfit_report.md").write_text("\n".join(lines), encoding="utf-8")
        return

    lines.append("\n## Stage probe metrics\n")
    show = [c for c in [
        "run_alias", "stage", "dim",
        "direct_train_macro_f1", "direct_val_macro_f1", "direct_gap_macro_f1",
        "linear_train_macro_f1", "linear_val_macro_f1", "linear_gap_macro_f1",
        "centroid_train_macro_f1", "centroid_val_macro_f1", "centroid_gap_macro_f1",
    ] if c in stage_df.columns]
    lines.append(stage_df[show].to_markdown(index=False))

    lines.append("\n## L1 residual-overfit diagnostic table\n")
    lines.append("| alias | embedding/input linear gap | layer0 linear gap | CLS linear gap | logits direct gap | classifier added gap |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    aliases = list(stage_df["run_alias"].unique())
    for alias in aliases:
        metric = "linear_gap_macro_f1"
        early = np.nanmax([
            stage_metric(stage_df, alias, r"embedding", metric),
            stage_metric(stage_df, alias, r"input_proj", metric),
        ])
        layer0 = stage_metric(stage_df, alias, r"encoder\.layers\.0$", metric)
        cls = stage_metric(stage_df, alias, r"CLASSIFIER_PRE_INPUT_CLS", metric)
        logit = stage_metric(stage_df, alias, r"MODEL_OUTPUT_LOGITS", "direct_gap_macro_f1")
        added = logit - cls if not np.isnan(logit) and not np.isnan(cls) else np.nan
        def fmt(x):
            return "NaN" if np.isnan(x) else f"{x:.6f}"
        lines.append(f"| {alias} | {fmt(early)} | {fmt(layer0)} | {fmt(cls)} | {fmt(logit)} | {fmt(added)} |")

    lines.append("\n## Interpretation rules\n")
    lines.append("```text")
    lines.append("For L1:")
    lines.append("- If embedding/input gap is already high: remaining overfit starts before attention.")
    lines.append("- If layer0/CLS gap is high: the single attention layer still learns train-specific subtype boundary.")
    lines.append("- If CLS gap is low but logits gap is high: classifier/logit boundary is the remaining locus.")
    lines.append("- If all representation gaps are moderate but logits direct val remains low: subtype overlap/boundary is likely the bottleneck.")
    lines.append("```")

    lines.append("\n## Automatic heuristic call\n")
    for alias in aliases:
        if alias.lower() != "l1":
            continue
        metric = "linear_gap_macro_f1"
        early = np.nanmax([
            stage_metric(stage_df, alias, r"embedding", metric),
            stage_metric(stage_df, alias, r"input_proj", metric),
        ])
        layer0 = stage_metric(stage_df, alias, r"encoder\.layers\.0$", metric)
        cls = stage_metric(stage_df, alias, r"CLASSIFIER_PRE_INPUT_CLS", metric)
        logit = stage_metric(stage_df, alias, r"MODEL_OUTPUT_LOGITS", "direct_gap_macro_f1")
        added = logit - cls if not np.isnan(logit) and not np.isnan(cls) else np.nan

        call = "inconclusive"
        if not np.isnan(early) and early > 0.06:
            call = "remaining_gap_starts_at_embedding_or_input_projection"
        if not np.isnan(layer0) and layer0 > max(0.06, early + 0.03 if not np.isnan(early) else 0.06):
            call = "remaining_gap_mainly_in_single_attention_layer_or_cls"
        if not np.isnan(added) and added > 0.03:
            call = "remaining_gap_mainly_added_by_classifier_logit_boundary"
        if (not np.isnan(layer0) and layer0 > 0.07) and (not np.isnan(added) and abs(added) <= 0.03):
            call = "L1_gap_is_not_classifier_only_single_layer_representation_already_overfits"
        lines.append("```text")
        lines.append(f"L1 early gap = {early}")
        lines.append(f"L1 layer0 gap = {layer0}")
        lines.append(f"L1 CLS gap = {cls}")
        lines.append(f"L1 logits direct gap = {logit}")
        lines.append(f"L1 classifier added gap = {added}")
        lines.append(f"heuristic_call = {call}")
        lines.append("```")

    lines.append("\n## Next decision\n")
    lines.append("```text")
    lines.append("Do not run F1c/gates or more architecture edits until reading this L1 audit.")
    lines.append("If L1 residual gap is classifier-driven, fix classifier boundary.")
    lines.append("If L1 residual gap is layer0/CLS-driven, inspect subtype overlap and representation shaping.")
    lines.append("If L1 residual gap starts early, inspect embedding/fusion/raw branch.")
    lines.append("```")

    (out_dir / "F1d_l1_residual_overfit_report.md").write_text("\n".join(lines), encoding="utf-8")


def zip_dir(src: Path, dst: Path):
    if dst.exists():
        dst.unlink()
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for p in src.rglob("*"):
            if p.is_file() and p != dst:
                z.write(p, p.relative_to(src.parent))


def source_scan(root: Path, out_dir: Path):
    rows = []
    for fp in [root / "02_src" / "06_model.py", root / "02_src" / "02_embedding.py", root / "02_src" / "07_train.py"]:
        if not fp.exists():
            continue
        txt = fp.read_text(encoding="utf-8", errors="ignore")
        lines = []
        for i, line in enumerate(txt.splitlines(), 1):
            if any(k in line for k in ["class ", "def forward", "Embedding", "Transformer", "classifier", "fusion", "raw_scaled", "values"]):
                lines.append(f"{i}: {line.rstrip()}")
        rows.append({"file": str(fp), "preview": "\n".join(lines[:250])})
    pd.DataFrame(rows).to_csv(out_dir / "source_scan_f1d.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-run-dir", default="03_outputs/06_model")
    ap.add_argument("--l1-run-dir", default="")
    ap.add_argument("--auto-find-l1", action="store_true", default=True)
    ap.add_argument("--no-auto-find-l1", dest="auto_find_l1", action="store_false")
    ap.add_argument("--l1-search-root", default="05_test/outputs")
    ap.add_argument("--allow-base-only", action="store_true", help="Do not fail if L1 checkpoint is missing.")
    ap.add_argument("--dataset-npz", default="03_outputs/05_dataset/dataset.npz")
    ap.add_argument("--train-raw", default="01_split/train_raw.csv")
    ap.add_argument("--val-raw", default="01_split/val_raw.csv")
    ap.add_argument("--out-dir", default="05_test/outputs/F1d_l1_residual_overfit_audit")
    ap.add_argument("--combined-zip", default="05_test/outputs/F1d_l1_residual_overfit_audit.zip")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--max-hooks", type=int, default=40)
    ap.add_argument("--max-samples-per-split", type=int, default=0)
    ap.add_argument("--candidate-eval-samples", type=int, default=0, help="0 = full val candidate check")
    ap.add_argument("--max-probe-train", type=int, default=30000)
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

    log(f"root={root}")
    log(f"device={device}")
    log(f"out_dir={out_dir}")
    log("Audit only; no model variant is trained.")

    base_dir = resolve_path(args.base_run_dir, root)
    l1_dir = resolve_path(args.l1_run_dir, root) if args.l1_run_dir else None
    if l1_dir is None and args.auto_find_l1:
        l1_dir = auto_find_l1_run(root, resolve_path(args.l1_search_root, root))
        if l1_dir is not None:
            log(f"auto-found L1 run: {l1_dir}")

    if l1_dir is None and not args.allow_base_only:
        msg = (
            "L1 run with checkpoint was not found. "
            "Pass --l1-run-dir <path> or rerun L1 before auditing residual L1 gap. "
            "Use --allow-base-only only for debugging."
        )
        (out_dir / "L1_NOT_FOUND.txt").write_text(msg, encoding="utf-8")
        zip_dir(out_dir, combined_zip)
        raise FileNotFoundError(msg)

    run_dirs = {"base512": base_dir}
    if l1_dir is not None:
        run_dirs["L1"] = l1_dir

    static_df = pd.DataFrame([static_run(a, p) for a, p in run_dirs.items()])
    static_df.to_csv(out_dir / "static_run_summary_f1d.csv", index=False)
    source_scan(root, out_dir)

    ds, ds_info = load_dataset(resolve_path(args.dataset_npz, root), resolve_path(args.train_raw, root), resolve_path(args.val_raw, root))
    (out_dir / "dataset_info_f1d.json").write_text(json.dumps(ds_info, indent=2, default=str), encoding="utf-8")

    all_stage = []
    all_margin = []
    infos = []

    for alias, rd in run_dirs.items():
        info = {"alias": alias, "run_dir": str(rd)}
        try:
            log(f"dynamic audit {alias}")
            model, minfo = build_model(root, rd, ds_info, ds["train"]["y"], device, args.trust_local_checkpoint)
            info["model_info"] = minfo
            model_modules(model).to_csv(out_dir / f"{alias}_model_modules_f1d.csv", index=False)

            expected = None
            srow = static_df[static_df["alias"] == alias]
            if len(srow) and "val_macro_f1" in srow and pd.notnull(srow.iloc[0]["val_macro_f1"]):
                expected = float(srow.iloc[0]["val_macro_f1"])
            info["expected_val_macro_f1"] = expected

            selected, cand_df = choose_value_candidate(
                model=model,
                ds=ds,
                num_bins=ds_info["num_bins"],
                device=device,
                batch_size=args.batch_size,
                expected_val_macro=expected,
                max_eval_samples=args.candidate_eval_samples,
                seed=args.seed,
            )
            info["selected_value_candidate"] = selected
            info["candidate_table"] = f"{alias}_value_candidate_selection_f1d.csv"
            cand_df.to_csv(out_dir / f"{alias}_value_candidate_selection_f1d.csv", index=False)
            log(f"{alias} selected value candidate: {selected}")

            # Save direct final report for selected candidate.
            vals = make_values(ds["val"], selected, ds_info["num_bins"])
            idx_full = np.arange(len(ds["val"]["y"]))
            logits = forward_logits(model, ds["val"]["bin"], vals, idx_full, device, args.batch_size)
            probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
            save_direct_final_report(out_dir, alias, ds["val"]["y"], probs)

            reps, einfo = extract_reps(
                model=model,
                ds=ds,
                value_name=selected,
                num_bins=ds_info["num_bins"],
                device=device,
                batch_size=args.batch_size,
                max_samples=args.max_samples_per_split,
                max_hooks=args.max_hooks,
                seed=args.seed,
            )
            info["extract_info"] = einfo

            rows = []
            mrows = []
            for i, (stage, rep) in enumerate(reps.items(), 1):
                log(f"{alias} eval stage {i}/{len(reps)} {stage}")
                row, margin = eval_stage(alias, stage, rep, args.seed, args.max_probe_train)
                rows.append(row)
                mrows.append(margin)

            sdf = pd.DataFrame(rows)
            mdf = pd.DataFrame(mrows)
            sdf.to_csv(out_dir / f"{alias}_stage_probe_metrics_f1d.csv", index=False)
            mdf.to_csv(out_dir / f"{alias}_stage_centroid_margin_f1d.csv", index=False)
            all_stage.append(sdf)
            all_margin.append(mdf)

        except Exception:
            info["error"] = "dynamic_exception"
            info["traceback"] = traceback.format_exc()
        infos.append(info)

    (out_dir / "dynamic_audit_info_f1d.json").write_text(json.dumps(infos, indent=2, default=str), encoding="utf-8")

    stage_df = pd.concat(all_stage, ignore_index=True) if all_stage else pd.DataFrame()
    margin_df = pd.concat(all_margin, ignore_index=True) if all_margin else pd.DataFrame()
    if len(stage_df):
        stage_df.to_csv(out_dir / "stage_probe_metrics_f1d.csv", index=False)
    else:
        (out_dir / "stage_probe_metrics_f1d_EMPTY.txt").write_text("No stage metrics produced.\n", encoding="utf-8")
    if len(margin_df):
        margin_df.to_csv(out_dir / "stage_centroid_margin_metrics_f1d.csv", index=False)

    write_report(out_dir, static_df, stage_df, margin_df, infos)

    log("static summary:")
    print(static_df.to_string(index=False), flush=True)
    if len(stage_df):
        cols = [c for c in ["run_alias", "stage", "linear_val_macro_f1", "linear_gap_macro_f1", "centroid_val_macro_f1", "centroid_gap_macro_f1", "direct_val_macro_f1", "direct_gap_macro_f1"] if c in stage_df.columns]
        print(stage_df[cols].to_string(index=False), flush=True)

    zip_dir(out_dir, combined_zip)
    log(f"zip={combined_zip}")
    log("DONE")


if __name__ == "__main__":
    main()
