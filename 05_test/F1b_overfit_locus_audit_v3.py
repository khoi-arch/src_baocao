#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F1b Overfit Locus Audit V3

Fix over V2:
- V2 loaded model/checkpoint correctly, but failed forward because D3 forward is:
    forward(tokens, values)
  and embedding requires values shape [B, F, 3].
- V3 constructs candidate [B,F,3] value tensors from:
    X_*_offset and raw_scaled
  then chooses the candidate that best reproduces the existing validation macro-F1.

This is still an audit, not a new training experiment.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import sys
import traceback
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


BASE512_REF = {"train_macro_f1": 0.910253, "val_macro_f1": 0.810094, "gap_macro_f1": 0.100158}
L1_REF = {"train_macro_f1": 0.911431, "val_macro_f1": 0.814224, "gap_macro_f1": 0.097207}


def log(msg: str) -> None:
    print(f"[F1bV3] {msg}", flush=True)


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


def parse_run_dirs(s: str, root: Path) -> Dict[str, Path]:
    out = {}
    for part in [x.strip() for x in s.split(",") if x.strip()]:
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = resolve_path(v.strip(), root)
        else:
            out[Path(part).name] = resolve_path(part, root)
    return out


def find_checkpoint(run_dir: Path) -> Optional[Path]:
    for n in ["best_model.pt", "model_best.pt", "checkpoint_best.pt", "best.pt", "checkpoint.pt", "model.pt"]:
        p = run_dir / n
        if p.exists():
            return p
    pts = sorted(run_dir.glob("*.pt"))
    return pts[0] if pts else None


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


def state_dict_from_ckpt(ckpt) -> Tuple[Optional[Dict[str, Any]], str]:
    if isinstance(ckpt, dict):
        for k in ["model_state_dict", "state_dict", "net_state_dict"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k], k
        if all(isinstance(k, str) for k in ckpt.keys()) and any(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt, "checkpoint_is_state_dict"
    if isinstance(ckpt, nn.Module):
        return ckpt.state_dict(), "module_state_dict"
    return None, "not_found"


def cfg_get(cfg: Dict[str, Any], k: str, default):
    if k in cfg:
        return cfg[k]
    for dname in ["model", "model_config", "training", "args"]:
        d = cfg.get(dname, {})
        if isinstance(d, dict) and k in d:
            return d[k]
    return default


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

    mod = load_module_from_path("_f1b_v3_model_06_model", root / "02_src" / "06_model.py")
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
        raise RuntimeError("checkpoint did not load cleanly: " + json.dumps(info, indent=2, default=str))
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
            exclude = {"label", "Class", "Category", "label_L1", "label_L2", "label_L3", "Label", "class", "category"}
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
        Xtr_raw = Xtr_off.copy().astype(np.float32)
        Xva_raw = Xva_off.copy().astype(np.float32)
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
    candidates = {
        # Most likely according to 02_embedding.py comment:
        # values[...,1] = raw_scaled continuous value.
        "offset_raw_zero": np.stack([off, raw, z], axis=-1),
        "offset_raw_one": np.stack([off, raw, o], axis=-1),
        "offset_raw_bin_norm": np.stack([off, raw, bin_norm], axis=-1),
        "offset_raw_offset": np.stack([off, raw, off], axis=-1),

        # Robust fallbacks in case channel order differs.
        "raw_offset_zero": np.stack([raw, off, z], axis=-1),
        "bin_norm_raw_offset": np.stack([bin_norm, raw, off], axis=-1),
        "offset_offset_raw": np.stack([off, off, raw], axis=-1),
        "raw_raw_offset": np.stack([raw, raw, off], axis=-1),
    }
    return candidates[name]


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
    val_n = len(ds["val"]["y"])
    idx = np.arange(val_n)
    if max_eval_samples and val_n > max_eval_samples:
        idx = np.sort(rng.choice(idx, max_eval_samples, replace=False))

    names = list(make_values(ds["train"], "offset_raw_zero", num_bins).shape for _ in [])  # no-op
    cand_names = [
        "offset_raw_zero",
        "offset_raw_one",
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
        raise RuntimeError("No values candidate worked: " + json.dumps(rows, indent=2))
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
        cls = m.__class__.__name__.lower()
        score = 0
        if name == "embedding": score += 200
        if name == "input_proj": score += 190
        if name.startswith("encoder.layers."): score += 180
        if name == "encoder.norm": score += 170
        if name == "classifier": score += 160
        if "self_attn" in lname: score += 80
        if "linear1" in lname or "linear2" in lname: score += 20
        if "dropout" in lname or "gelu" in cls: score -= 100
        if score > 0:
            priority.append((score, name, m))
    priority.sort(key=lambda x: (-x[0], x[1]))
    return [(n, m) for _, n, m in priority[:max_hooks]]


def rep_from_output(x):
    if isinstance(x, (tuple, list)):
        for v in x:
            got = rep_from_output(v)
            if got is not None: return got
        return None
    if isinstance(x, dict):
        for v in x.values():
            got = rep_from_output(v)
            if got is not None: return got
        return None
    if not torch.is_tensor(x):
        return None
    t = x.detach().float()
    if t.ndim == 2:
        return t.cpu().numpy()
    if t.ndim == 3:
        # batch_first expected here.
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
        handles.append(m.register_forward_hook(mk_hook(name)))
    if hasattr(model, "classifier"):
        handles.append(model.classifier.register_forward_pre_hook(cls_pre_hook))

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
        try: h.remove()
        except Exception: pass

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
    return (Xtr-mu)/sd, (Xva-mu)/sd


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
    ytr = rep["ytr"].astype(int); yva = rep["yva"].astype(int)
    Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=1e6, neginf=-1e6)
    Xva = np.nan_to_num(Xva, nan=0.0, posinf=1e6, neginf=-1e6)
    Xtr_s, Xva_s = standardize(Xtr, Xva)

    row = {"run_alias": alias, "stage": stage, "dim": int(Xtr_s.shape[1]), "n_train": len(ytr), "n_val": len(yva)}
    if "LOGITS" in stage.upper():
        ptr = Xtr.argmax(axis=1); pva = Xva.argmax(axis=1)
        row.update({
            "direct_train_macro_f1": macro_f1(ytr, ptr),
            "direct_val_macro_f1": macro_f1(yva, pva),
            "direct_gap_macro_f1": macro_f1(ytr, ptr) - macro_f1(yva, pva),
            "direct_train_acc": acc(ytr, ptr),
            "direct_val_acc": acc(yva, pva),
        })

    try:
        ptr = centroid_pred(Xtr_s, ytr, Xtr_s)
        pva = centroid_pred(Xtr_s, ytr, Xva_s)
        row.update({
            "centroid_train_macro_f1": macro_f1(ytr, ptr),
            "centroid_val_macro_f1": macro_f1(yva, pva),
            "centroid_gap_macro_f1": macro_f1(ytr, ptr) - macro_f1(yva, pva),
            "centroid_train_acc": acc(ytr, ptr),
            "centroid_val_acc": acc(yva, pva),
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
                loss="log_loss", alpha=1e-4, class_weight="balanced",
                max_iter=1000, tol=1e-4, early_stopping=True,
                random_state=seed, n_jobs=-1
            )
        )
        clf.fit(Xtr[ids], ytr[ids])
        ptr = clf.predict(Xtr); pva = clf.predict(Xva)
        row.update({
            "linear_train_macro_f1": macro_f1(ytr, ptr),
            "linear_val_macro_f1": macro_f1(yva, pva),
            "linear_gap_macro_f1": macro_f1(ytr, ptr) - macro_f1(yva, pva),
            "linear_train_acc": acc(ytr, ptr),
            "linear_val_acc": acc(yva, pva),
        })
    except Exception:
        row["linear_error"] = traceback.format_exc()

    return row, margin


def write_report(out_dir: Path, static_df: pd.DataFrame, stage_df: pd.DataFrame, margin_df: pd.DataFrame, infos: List[Dict[str, Any]]):
    lines = []
    lines.append("# F1b Overfit Locus Audit V3 Report\n")
    lines.append("## Reference\n")
    lines.append("```text")
    lines.append(f"Base512 val macro-F1 = {BASE512_REF['val_macro_f1']:.6f}, gap = {BASE512_REF['gap_macro_f1']:.6f}")
    lines.append(f"L1 anchor val macro-F1 = {L1_REF['val_macro_f1']:.6f}, gap = {L1_REF['gap_macro_f1']:.6f}")
    lines.append("```")

    lines.append("\n## Static summary\n")
    if len(static_df):
        cols = [c for c in ["alias","exists","has_checkpoint","train_macro_f1","val_macro_f1","gap_macro_f1","num_layers","hidden_dim","num_heads","classifier_hidden_dim"] if c in static_df.columns]
        lines.append(static_df[cols].to_markdown(index=False))

    lines.append("\n## Value candidate selection\n")
    for info in infos:
        alias = info.get("alias")
        if "selected_value_candidate" in info:
            lines.append(f"- {alias}: selected `{info['selected_value_candidate']}` because it best reproduced validation macro-F1.")
            if "candidate_table_path" in info:
                lines.append(f"  - table: `{info['candidate_table_path']}`")
        elif "error" in info:
            lines.append(f"- {alias}: dynamic error `{info['error']}`")

    if len(stage_df) == 0:
        lines.append("\n## Dynamic audit failed\n")
        lines.append("No stage metrics produced. Read `dynamic_audit_info_v3.json`.")
        (out_dir / "F1b_overfit_locus_report_v3.md").write_text("\n".join(lines), encoding="utf-8")
        return

    lines.append("\n## Stage probe metrics\n")
    show = [c for c in [
        "run_alias","stage","dim",
        "direct_train_macro_f1","direct_val_macro_f1","direct_gap_macro_f1",
        "linear_train_macro_f1","linear_val_macro_f1","linear_gap_macro_f1",
        "centroid_train_macro_f1","centroid_val_macro_f1","centroid_gap_macro_f1",
    ] if c in stage_df.columns]
    lines.append(stage_df[show].to_markdown(index=False))

    lines.append("\n## Heuristic locus call\n")
    for alias, g in stage_df.groupby("run_alias"):
        lines.append(f"\n### {alias}")
        lines.append("```text")
        gap_metric = "linear_gap_macro_f1" if "linear_gap_macro_f1" in g and g["linear_gap_macro_f1"].notna().any() else "centroid_gap_macro_f1"
        val_metric = "linear_val_macro_f1" if "linear_val_macro_f1" in g and g["linear_val_macro_f1"].notna().any() else "centroid_val_macro_f1"

        def subset(pattern):
            return g[g["stage"].str.contains(pattern, case=False, regex=True, na=False)]

        def maxv(df, col):
            if len(df)==0 or col not in df or df[col].dropna().empty:
                return np.nan
            return float(df[col].dropna().max())

        early = subset("embedding|input_proj")
        enc0 = subset("encoder\\.layers\\.0$")
        enc1 = subset("encoder\\.layers\\.1$")
        enc2 = subset("encoder\\.layers\\.2$")
        cls = subset("CLASSIFIER_PRE_INPUT_CLS")
        logits = subset("MODEL_OUTPUT_LOGITS")

        lines.append(f"best_val_probe = {maxv(g, val_metric):.6f}")
        lines.append(f"max_gap_probe = {maxv(g, gap_metric):.6f}")
        lines.append(f"early_max_gap = {maxv(early, gap_metric):.6f}")
        lines.append(f"encoder_layer0_gap = {maxv(enc0, gap_metric):.6f}")
        lines.append(f"encoder_layer1_gap = {maxv(enc1, gap_metric):.6f}")
        lines.append(f"encoder_layer2_gap = {maxv(enc2, gap_metric):.6f}")
        lines.append(f"cls_input_gap = {maxv(cls, gap_metric):.6f}")
        if len(logits) and "direct_gap_macro_f1" in logits and logits["direct_gap_macro_f1"].notna().any():
            lines.append(f"logits_direct_val = {float(logits['direct_val_macro_f1'].dropna().iloc[0]):.6f}")
            lines.append(f"logits_direct_gap = {float(logits['direct_gap_macro_f1'].dropna().iloc[0]):.6f}")

        early_gap = maxv(early, gap_metric)
        enc_gap = np.nanmax([maxv(enc0,gap_metric), maxv(enc1,gap_metric), maxv(enc2,gap_metric)])
        cls_gap = maxv(cls, gap_metric)
        log_gap = np.nan
        if len(logits) and "direct_gap_macro_f1" in logits and logits["direct_gap_macro_f1"].notna().any():
            log_gap = float(logits["direct_gap_macro_f1"].dropna().iloc[0])

        call = "inconclusive"
        if not np.isnan(early_gap) and early_gap > 0.08:
            call = "early_embedding_or_fusion_gap_already_large"
        if not np.isnan(enc_gap) and not np.isnan(early_gap) and (enc_gap - early_gap) > 0.03:
            call = "gap_grows_in_transformer_attention_layers"
        if not np.isnan(log_gap) and not np.isnan(cls_gap) and (log_gap - cls_gap) > 0.03:
            call = "classifier_or_logit_boundary_adds_overfit"
        lines.append(f"heuristic_call = {call}")
        lines.append("```")

    lines.append("\n## Decision rule\n")
    lines.append("```text")
    lines.append("If early_max_gap is already high: fix embedding/fusion/input representation.")
    lines.append("If encoder layer gap grows: depth/attention interaction is the overfit locus.")
    lines.append("If CLS gap is okay but logits gap is much worse: classifier boundary is locus.")
    lines.append("If none is extreme and subtype remains weak: overlap is likely the next bottleneck.")
    lines.append("```")
    (out_dir / "F1b_overfit_locus_report_v3.md").write_text("\n".join(lines), encoding="utf-8")


def source_scan(root: Path, out_dir: Path):
    rows = []
    for fp in [root/"02_src"/"06_model.py", root/"02_src"/"02_embedding.py", root/"02_src"/"07_train.py"]:
        if not fp.exists(): continue
        txt = fp.read_text(encoding="utf-8", errors="ignore")
        lines = []
        for i, line in enumerate(txt.splitlines(), 1):
            if any(k in line for k in ["class ", "def forward", "Embedding", "Transformer", "classifier", "fusion", "raw_scaled", "values"]):
                lines.append(f"{i}: {line.rstrip()}")
        rows.append({"file": str(fp), "preview": "\n".join(lines[:250])})
    pd.DataFrame(rows).to_csv(out_dir/"source_scan_v3.csv", index=False)


def zip_dir(src: Path, dst: Path):
    if dst.exists():
        dst.unlink()
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for p in src.rglob("*"):
            if p.is_file() and p != dst:
                z.write(p, p.relative_to(src.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", default="base512=03_outputs/06_model")
    ap.add_argument("--dataset-npz", default="03_outputs/05_dataset/dataset.npz")
    ap.add_argument("--train-raw", default="01_split/train_raw.csv")
    ap.add_argument("--val-raw", default="01_split/val_raw.csv")
    ap.add_argument("--out-dir", default="05_test/outputs/F1b_overfit_locus_audit_v3")
    ap.add_argument("--combined-zip", default="05_test/outputs/F1b_overfit_locus_audit_v3.zip")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--max-hooks", type=int, default=40)
    ap.add_argument("--max-samples-per-split", type=int, default=0)
    ap.add_argument("--candidate-eval-samples", type=int, default=0, help="0 = full val for candidate selection")
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

    run_dirs = parse_run_dirs(args.run_dirs, root)
    static_df = pd.DataFrame([static_run(a, p) for a, p in run_dirs.items()])
    static_df.to_csv(out_dir/"static_run_summary_v3.csv", index=False)
    source_scan(root, out_dir)

    ds, ds_info = load_dataset(resolve_path(args.dataset_npz, root), resolve_path(args.train_raw, root), resolve_path(args.val_raw, root))
    (out_dir/"dataset_info_v3.json").write_text(json.dumps(ds_info, indent=2, default=str), encoding="utf-8")

    all_stage = []
    all_margin = []
    infos = []

    for alias, rd in run_dirs.items():
        info = {"alias": alias, "run_dir": str(rd)}
        try:
            log(f"dynamic audit {alias}")
            model, minfo = build_model(root, rd, ds_info, ds["train"]["y"], device, args.trust_local_checkpoint)
            info["model_info"] = minfo
            model_modules(model).to_csv(out_dir/f"{alias}_model_modules_v3.csv", index=False)

            expected = None
            srow = static_df[static_df["alias"] == alias]
            if len(srow) and "val_macro_f1" in srow and pd.notnull(srow.iloc[0]["val_macro_f1"]):
                expected = float(srow.iloc[0]["val_macro_f1"])

            selected, cand_df = choose_value_candidate(
                model=model, ds=ds, num_bins=ds_info["num_bins"], device=device,
                batch_size=args.batch_size, expected_val_macro=expected,
                max_eval_samples=args.candidate_eval_samples, seed=args.seed,
            )
            info["selected_value_candidate"] = selected
            info["candidate_table_path"] = f"{alias}_value_candidate_selection_v3.csv"
            cand_df.to_csv(out_dir/f"{alias}_value_candidate_selection_v3.csv", index=False)
            log(f"{alias} selected value candidate: {selected}")

            reps, einfo = extract_reps(
                model=model, ds=ds, value_name=selected, num_bins=ds_info["num_bins"],
                device=device, batch_size=args.batch_size,
                max_samples=args.max_samples_per_split,
                max_hooks=args.max_hooks, seed=args.seed,
            )
            info["extract_info"] = einfo

            rows = []
            mrows = []
            for i, (stage, rep) in enumerate(reps.items(), 1):
                log(f"{alias} eval stage {i}/{len(reps)} {stage}")
                row, margin = eval_stage(alias, stage, rep, args.seed, args.max_probe_train)
                rows.append(row); mrows.append(margin)

            sdf = pd.DataFrame(rows)
            mdf = pd.DataFrame(mrows)
            sdf.to_csv(out_dir/f"{alias}_stage_probe_metrics_v3.csv", index=False)
            mdf.to_csv(out_dir/f"{alias}_stage_centroid_margin_v3.csv", index=False)
            all_stage.append(sdf)
            all_margin.append(mdf)

        except Exception:
            info["error"] = "dynamic_exception"
            info["traceback"] = traceback.format_exc()
        infos.append(info)

    (out_dir/"dynamic_audit_info_v3.json").write_text(json.dumps(infos, indent=2, default=str), encoding="utf-8")
    stage_df = pd.concat(all_stage, ignore_index=True) if all_stage else pd.DataFrame()
    margin_df = pd.concat(all_margin, ignore_index=True) if all_margin else pd.DataFrame()
    if len(stage_df):
        stage_df.to_csv(out_dir/"stage_probe_metrics_v3.csv", index=False)
    else:
        (out_dir/"stage_probe_metrics_v3_EMPTY.txt").write_text("No stage metrics produced.\n", encoding="utf-8")
    if len(margin_df):
        margin_df.to_csv(out_dir/"stage_centroid_margin_metrics_v3.csv", index=False)

    write_report(out_dir, static_df, stage_df, margin_df, infos)

    log("static summary:")
    print(static_df.to_string(index=False), flush=True)
    if len(stage_df):
        cols = [c for c in ["run_alias","stage","linear_val_macro_f1","linear_gap_macro_f1","centroid_val_macro_f1","centroid_gap_macro_f1","direct_val_macro_f1","direct_gap_macro_f1"] if c in stage_df.columns]
        print(stage_df[cols].to_string(index=False), flush=True)

    zip_dir(out_dir, combined_zip)
    log(f"zip={combined_zip}")
    log("DONE")


if __name__ == "__main__":
    main()
