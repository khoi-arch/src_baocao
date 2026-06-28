#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F1b Overfit Locus Audit

Goal
----
Do NOT train new architecture variants.
Do NOT brute-force hyperparameters.

Instead, locate where the train/val overfit gap appears in the D3 pipeline:

raw/token inputs
 -> embedding / fusion
 -> transformer interaction layers
 -> CLS representation
 -> classifier MLP
 -> logits

What it does
------------
For each supplied run directory:
1) Static audit from existing outputs:
   - diagnosis_summary.json
   - history.csv
   - classification reports
   - confusion matrices
   - prediction CSVs if present

2) Dynamic stage audit if possible:
   - loads best checkpoint
   - reconstructs model when possible
   - loads dataset.npz
   - discovers module names
   - registers hooks on embedding/fusion/transformer/layer/classifier-like modules
   - extracts compact representations per stage
   - trains lightweight diagnostic probes, not final models:
       a) nearest-centroid probe
       b) linear SGD probe
   - computes train/val macro-F1, gap, per-class behavior, centroid margins

3) Writes a decision-oriented report:
   - Does gap already exist at early embedding/fusion?
   - Does gap grow across transformer layers?
   - Is representation okay but classifier/logits worse?
   - Is this more overlap than overfit?

Important
---------
This script is intentionally defensive. If it cannot reconstruct the model from
your repo/checkpoint, it will still produce a discovery report with:
- checkpoint keys
- config keys
- model source classes
- module names if any model can be built
- exact failure messages

That prevents guessing.
"""

from __future__ import annotations

import argparse
import ast as pyast
import inspect
import importlib
import json
import math
import os
import random
import sys
import traceback
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Iterable

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
except Exception as e:
    torch = None
    nn = None

try:
    from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
except Exception as e:
    f1_score = None
    accuracy_score = None
    classification_report = None
    confusion_matrix = None
    SGDClassifier = None
    StandardScaler = None
    make_pipeline = None


BASE512_REF = {
    "train_macro_f1": 0.910253,
    "val_macro_f1": 0.810094,
    "gap_macro_f1": 0.100158,
}

L1_REF = {
    "train_macro_f1": 0.911431,
    "val_macro_f1": 0.814224,
    "gap_macro_f1": 0.097207,
}


def log(msg: str) -> None:
    print(f"[F1b] {msg}", flush=True)


def repo_root_from_here() -> Path:
    p = Path(__file__).resolve()
    if p.parent.name == "05_test":
        return p.parents[1]
    return Path.cwd().resolve()


def resolve_path(p: str | Path, root: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (root / p).resolve()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_read_json(path: Path) -> Dict[str, Any]:
    try:
        return load_json(path) if path.exists() else {}
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
    """
    Format:
      alias=path,alias2=path2
    If no alias, use path basename.
    """
    out: Dict[str, Path] = {}
    for part in [x.strip() for x in s.split(",") if x.strip()]:
        if "=" in part:
            alias, path = part.split("=", 1)
            alias = alias.strip()
            path = path.strip()
        else:
            path = part
            alias = Path(part).name
        out[alias] = resolve_path(path, root)
    return out


def flatten_report_f1(prefix: str, rep: Dict[str, Any]) -> Dict[str, float]:
    row = {}
    if not isinstance(rep, dict):
        return row
    if "per_class" in rep and isinstance(rep["per_class"], dict):
        items = rep["per_class"].items()
    else:
        items = []
        for k, v in rep.items():
            if isinstance(v, dict) and ("f1" in v or "f1-score" in v):
                if str(k).lower() in {"accuracy", "macro avg", "weighted avg"}:
                    continue
                items.append((k, v))
    for label, metrics in items:
        f1 = metrics.get("f1", metrics.get("f1-score"))
        if f1 is not None:
            safe = str(label).replace(" ", "_").replace("/", "_")
            row[f"{prefix}_f1_{safe}"] = float(f1)
    return row


def static_audit_run(alias: str, run_dir: Path) -> Dict[str, Any]:
    diag = safe_read_json(run_dir / "diagnosis_summary.json")
    cfg = safe_read_json(run_dir / "config.json")
    val_rep = safe_read_json(run_dir / "val_classification_report_best.json")
    train_rep = safe_read_json(run_dir / "train_classification_report_best.json")

    row: Dict[str, Any] = {
        "alias": alias,
        "run_dir": str(run_dir),
        "exists": run_dir.exists(),
        "has_best_model": any((run_dir / n).exists() for n in ["best_model.pt", "model_best.pt", "checkpoint.pt", "best.pt"]),
        "best_epoch": diag.get("best_epoch"),
        "train_macro_f1": maybe_get(diag, "train", "macro_f1"),
        "val_macro_f1": maybe_get(diag, "val", "macro_f1"),
        "gap_macro_f1": diag.get("generalization_gap_macro_f1"),
        "train_acc": maybe_get(diag, "train", "accuracy"),
        "val_acc": maybe_get(diag, "val", "accuracy"),
        "train_weighted_f1": maybe_get(diag, "train", "weighted_f1"),
        "val_weighted_f1": maybe_get(diag, "val", "weighted_f1"),
    }

    model_cfg = cfg.get("model", cfg.get("model_config", {}))
    if isinstance(model_cfg, dict):
        for k in ["num_layers", "hidden_dim", "num_heads", "classifier_hidden_dim", "dropout", "classifier_dropout"]:
            row[k] = model_cfg.get(k, cfg.get(k))
    for k, v in flatten_report_f1("train", train_rep).items():
        row[k] = v
    for k, v in flatten_report_f1("val", val_rep).items():
        row[k] = v

    hist_path = run_dir / "history.csv"
    if hist_path.exists():
        try:
            hist = pd.read_csv(hist_path)
            row["history_epochs"] = len(hist)
            if "val_macro_f1" in hist.columns and len(hist):
                i = hist["val_macro_f1"].idxmax()
                row["history_best_val_macro_f1"] = float(hist.loc[i, "val_macro_f1"])
                if "train_macro_f1" in hist.columns:
                    row["history_train_at_best_val"] = float(hist.loc[i, "train_macro_f1"])
                    row["history_gap_at_best_val"] = row["history_train_at_best_val"] - row["history_best_val_macro_f1"]
        except Exception as e:
            row["history_read_error"] = str(e)

    return row


def find_checkpoint(run_dir: Path) -> Optional[Path]:
    names = [
        "best_model.pt",
        "model_best.pt",
        "checkpoint_best.pt",
        "best.pt",
        "checkpoint.pt",
        "model.pt",
    ]
    for n in names:
        p = run_dir / n
        if p.exists():
            return p
    pts = sorted(run_dir.glob("*.pt"))
    return pts[0] if pts else None


def add_src_to_path(root: Path) -> None:
    for rel in ["02_src", "src", "."]:
        p = root / rel
        if p.exists():
            sp = str(p.resolve())
            if sp not in sys.path:
                sys.path.insert(0, sp)


def checkpoint_summary(ckpt: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "type": str(type(ckpt)),
    }
    if isinstance(ckpt, dict):
        out["dict_keys"] = list(ckpt.keys())[:100]
        for k in ["epoch", "best_epoch", "val_macro_f1", "best_val_macro_f1"]:
            if k in ckpt:
                out[k] = ckpt[k]
        for k in ["model_state_dict", "state_dict"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                out[f"{k}_num_tensors"] = len(ckpt[k])
                out[f"{k}_first_keys"] = list(ckpt[k].keys())[:20]
    return out


def extract_state_dict(ckpt: Any) -> Tuple[Optional[Dict[str, Any]], str]:
    if torch is not None and isinstance(ckpt, nn.Module):
        return None, "full_module"
    if not isinstance(ckpt, dict):
        return None, "unknown_checkpoint_type"

    for k in ["model_state_dict", "state_dict", "net_state_dict"]:
        if k in ckpt and isinstance(ckpt[k], dict):
            return ckpt[k], k

    # Sometimes checkpoint itself is a state_dict.
    if all(isinstance(k, str) for k in ckpt.keys()) and any(isinstance(v, torch.Tensor) for v in ckpt.values()):
        tensor_ratio = sum(isinstance(v, torch.Tensor) for v in ckpt.values()) / max(1, len(ckpt))
        if tensor_ratio > 0.7:
            return ckpt, "checkpoint_is_state_dict"

    return None, "no_state_dict_found"


def discover_model_classes(root: Path) -> List[Tuple[str, Any]]:
    classes = []
    add_src_to_path(root)
    for modname in ["model", "models", "architecture", "net", "network"]:
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            try:
                if nn is not None and issubclass(obj, nn.Module) and obj is not nn.Module:
                    classes.append((f"{modname}.{name}", obj))
            except Exception:
                pass
    return classes


def merged_config(cfg: Dict[str, Any], y_train: Optional[np.ndarray], feature_map: Optional[Dict[str, np.ndarray]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    def rec(prefix, d):
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, dict):
                    rec(k, v)
                else:
                    merged[k] = v
    rec("", cfg)

    if y_train is not None:
        try:
            merged.setdefault("num_classes", int(len(np.unique(y_train))))
            merged.setdefault("n_classes", int(len(np.unique(y_train))))
            merged.setdefault("num_labels", int(len(np.unique(y_train))))
        except Exception:
            pass

    if feature_map:
        # Best-effort dims.
        first = next(iter(feature_map.values()))
        if hasattr(first, "shape") and len(first.shape) >= 2:
            merged.setdefault("num_features", int(first.shape[1]))
            merged.setdefault("input_dim", int(first.shape[1]))
            merged.setdefault("n_features", int(first.shape[1]))
            merged.setdefault("seq_len", int(first.shape[1]))
    return merged


def instantiate_model_from_state(
    root: Path,
    cfg: Dict[str, Any],
    state_dict: Optional[Dict[str, Any]],
    y_train: Optional[np.ndarray],
    feature_map: Optional[Dict[str, np.ndarray]],
    device: str,
) -> Tuple[Optional[nn.Module], Dict[str, Any]]:
    info: Dict[str, Any] = {"attempts": []}
    classes = discover_model_classes(root)
    info["candidate_classes"] = [name for name, _ in classes]

    if state_dict is None:
        info["error"] = "No state_dict to instantiate from."
        return None, info

    values = merged_config(cfg, y_train, feature_map)

    best_model = None
    best_score = None
    best_info = None

    for cname, cls in classes:
        attempts_for_class = []
        constructors = []

        # 1) No-arg
        constructors.append(("no_args", (), {}))

        # 2) config dict as first positional arg
        constructors.append(("config_positional", (cfg,), {}))

        # 3) kwargs from signature
        try:
            sig = inspect.signature(cls)
            kwargs = {}
            missing = []
            for pname, p in sig.parameters.items():
                if pname == "self":
                    continue
                if pname in values:
                    kwargs[pname] = values[pname]
                elif p.default is inspect._empty and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY):
                    missing.append(pname)
            if not missing:
                constructors.append(("signature_kwargs", (), kwargs))
            else:
                attempts_for_class.append({"constructor": "signature_kwargs", "skipped_missing": missing})
        except Exception as e:
            attempts_for_class.append({"constructor": "signature_inspect", "error": str(e)})

        for label, args, kwargs in constructors:
            try:
                model = cls(*args, **kwargs)
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                score = len(missing_keys) + len(unexpected_keys)
                attempts_for_class.append({
                    "constructor": label,
                    "ok": True,
                    "missing": len(missing_keys),
                    "unexpected": len(unexpected_keys),
                    "first_missing": list(missing_keys)[:10],
                    "first_unexpected": list(unexpected_keys)[:10],
                })
                if best_score is None or score < best_score:
                    best_score = score
                    best_model = model
                    best_info = {
                        "class": cname,
                        "constructor": label,
                        "missing": len(missing_keys),
                        "unexpected": len(unexpected_keys),
                        "first_missing": list(missing_keys)[:20],
                        "first_unexpected": list(unexpected_keys)[:20],
                    }
            except Exception as e:
                attempts_for_class.append({
                    "constructor": label,
                    "ok": False,
                    "error": repr(e),
                })

        info["attempts"].append({"class": cname, "attempts": attempts_for_class})

    info["best"] = best_info
    if best_model is not None:
        best_model.to(device)
        best_model.eval()
    return best_model, info


def scan_source_architecture(root: Path) -> pd.DataFrame:
    rows = []
    src = root / "02_src"
    if not src.exists():
        src = root
    for path in list(src.glob("*.py")) + list((src / "models").glob("*.py") if (src / "models").exists() else []):
        try:
            txt = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        interesting = []
        for i, line in enumerate(txt.splitlines(), 1):
            l = line.strip()
            if any(tok in l for tok in [
                "Transformer", "MultiheadAttention", "Embedding", "Linear", "LayerNorm",
                "classifier", "fusion", "FiLM", "gate", "CLS", "cls"
            ]):
                interesting.append(f"{i}: {l}")
        if interesting:
            rows.append({
                "file": str(path),
                "num_interesting_lines": len(interesting),
                "interesting_lines_preview": "\n".join(interesting[:80]),
            })
    return pd.DataFrame(rows)


def load_dataset_npz(dataset_path: Path) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[Dict[str, np.ndarray]], Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    info: Dict[str, Any] = {"dataset_path": str(dataset_path), "exists": dataset_path.exists()}
    if not dataset_path.exists():
        info["error"] = "dataset_npz_not_found"
        return None, None, None, None, info

    data = np.load(dataset_path, allow_pickle=True)
    keys = list(data.files)
    info["keys"] = keys
    info["shapes"] = {k: list(data[k].shape) for k in keys if hasattr(data[k], "shape")}

    def norm_base(k: str, split: str) -> str:
        lk = k.lower()
        for pat in [f"{split}_", f"{split}", f"_{split}", f"{split}x", f"x_{split}"]:
            if lk.startswith(pat):
                lk = lk[len(pat):]
        lk = lk.replace("valid", "val")
        # common
        for pref in ["x_", "features_", "feature_", "data_"]:
            if lk.startswith(pref):
                lk = lk[len(pref):]
        return lk.strip("_")

    y_train = None
    y_val = None
    train_feats: Dict[str, np.ndarray] = {}
    val_feats: Dict[str, np.ndarray] = {}

    for k in keys:
        arr = data[k]
        lk = k.lower()
        if not hasattr(arr, "shape") or len(arr.shape) == 0:
            continue

        is_label = (
            lk in {"y_train", "train_y", "train_labels", "labels_train", "label_train"}
            or lk in {"y_val", "val_y", "valid_y", "val_labels", "valid_labels", "labels_val", "label_val"}
            or ("label" in lk and ("train" in lk or "val" in lk or "valid" in lk))
            or (lk.endswith("_y") and ("train" in lk or "val" in lk or "valid" in lk))
        )

        if "train" in lk:
            if is_label and y_train is None:
                y_train = np.asarray(arr).reshape(-1)
            elif not is_label:
                train_feats[norm_base(k, "train")] = np.asarray(arr)
        elif "val" in lk or "valid" in lk:
            if is_label and y_val is None:
                y_val = np.asarray(arr).reshape(-1)
            elif not is_label:
                val_feats[norm_base(k.replace("valid", "val"), "val")] = np.asarray(arr)

    # Fallback direct names
    if y_train is None:
        for k in ["y_train", "train_y", "train_labels"]:
            if k in data.files:
                y_train = np.asarray(data[k]).reshape(-1)
    if y_val is None:
        for k in ["y_val", "val_y", "valid_y", "val_labels"]:
            if k in data.files:
                y_val = np.asarray(data[k]).reshape(-1)

    # Fallback X_train/X_val
    if not train_feats:
        for k in ["X_train", "x_train", "train_X", "train_x"]:
            if k in data.files:
                train_feats["x"] = np.asarray(data[k])
    if not val_feats:
        for k in ["X_val", "x_val", "val_X", "val_x", "X_valid", "valid_X"]:
            if k in data.files:
                val_feats["x"] = np.asarray(data[k])

    # Keep only feature keys with both train and val and same base.
    common = sorted(set(train_feats.keys()) & set(val_feats.keys()))
    train_feats = {k: train_feats[k] for k in common}
    val_feats = {k: val_feats[k] for k in common}

    info["detected_feature_keys"] = common
    info["y_train_shape"] = None if y_train is None else list(y_train.shape)
    info["y_val_shape"] = None if y_val is None else list(y_val.shape)

    return train_feats, val_feats, y_train, y_val, info


def tensorize_array(name: str, arr: np.ndarray, device: str):
    if torch is None:
        raise RuntimeError("torch not available")
    if np.issubdtype(arr.dtype, np.integer):
        # Token/id-like arrays should be long. For binary/continuous int arrays,
        # long is still accepted by embedding but not by Linear. Forward trials handle failure.
        t = torch.as_tensor(arr, dtype=torch.long, device=device)
    else:
        t = torch.as_tensor(arr, dtype=torch.float32, device=device)
    return t


def make_batch_dict(feats: Dict[str, np.ndarray], idx: np.ndarray, device: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, arr in feats.items():
        base = k.lower()
        out[base] = tensorize_array(base, arr[idx], device)

        # Add aliases but do not duplicate object too much.
        if base in {"x", "features", "tokens", "token", "input", "inputs"}:
            out.setdefault("x", out[base])
            out.setdefault("inputs", out[base])
        if "token" in base or base in {"x", "ids"}:
            out.setdefault("tokens", out[base])
            out.setdefault("token_ids", out[base])
            out.setdefault("input_ids", out[base])
        if "raw" in base or "cont" in base or "continuous" in base:
            out.setdefault("raw", out[base])
            out.setdefault("raw_features", out[base])
            out.setdefault("continuous", out[base])
            out.setdefault("x_raw", out[base])
    return out


@dataclass
class ForwardMode:
    name: str
    kind: str
    keys: List[str]


def filter_kwargs_for_forward(model: nn.Module, batch: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sig = inspect.signature(model.forward)
        params = sig.parameters
        if any(p.kind == p.VAR_KEYWORD for p in params.values()):
            return batch
        return {k: v for k, v in batch.items() if k in params}
    except Exception:
        return batch


def output_to_logits(out: Any) -> Optional[Any]:
    if torch is None:
        return None
    if isinstance(out, torch.Tensor):
        if out.ndim == 2:
            return out
        if out.ndim > 2:
            return out.reshape(out.shape[0], -1)
    if isinstance(out, dict):
        # Prioritize common logit keys
        for k in ["logits", "output", "outputs", "pred", "prediction"]:
            if k in out and isinstance(out[k], torch.Tensor):
                return output_to_logits(out[k])
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


def find_forward_mode(model: nn.Module, train_feats: Dict[str, np.ndarray], device: str) -> Tuple[Optional[ForwardMode], Dict[str, Any]]:
    info: Dict[str, Any] = {"attempts": []}
    n = min(8, len(next(iter(train_feats.values()))))
    idx = np.arange(n)
    batch = make_batch_dict(train_feats, idx, device)

    # Candidate 1: filtered kwargs
    kwargs = filter_kwargs_for_forward(model, batch)
    candidates: List[Tuple[str, str, List[str], Any]] = []
    if kwargs:
        candidates.append(("filtered_kwargs", "kwargs", list(kwargs.keys()), kwargs))
    candidates.append(("all_kwargs", "kwargs", list(batch.keys()), batch))
    candidates.append(("batch_dict_positional", "dict", list(batch.keys()), batch))

    # Single/positional candidates
    original_keys = list(train_feats.keys())
    for k in original_keys:
        candidates.append((f"single_{k}", "single", [k], batch[k.lower()]))
    if len(original_keys) > 1:
        vals = [batch[k.lower()] for k in original_keys if k.lower() in batch]
        candidates.append(("positional_original_feature_order", "positional", original_keys, vals))

    model.eval()
    with torch.no_grad():
        for name, kind, keys, payload in candidates:
            try:
                if kind == "kwargs":
                    out = model(**payload)
                elif kind == "dict":
                    out = model(payload)
                elif kind == "single":
                    out = model(payload)
                elif kind == "positional":
                    out = model(*payload)
                else:
                    continue
                logits = output_to_logits(out)
                ok = logits is not None and isinstance(logits, torch.Tensor) and logits.shape[0] == n
                info["attempts"].append({
                    "name": name,
                    "kind": kind,
                    "keys": keys,
                    "ok": bool(ok),
                    "out_type": str(type(out)),
                    "logits_shape": None if logits is None else list(logits.shape),
                })
                if ok:
                    return ForwardMode(name=name, kind=kind, keys=keys), info
            except Exception as e:
                info["attempts"].append({
                    "name": name,
                    "kind": kind,
                    "keys": keys,
                    "ok": False,
                    "error": repr(e),
                })
    return None, info


def call_forward(model: nn.Module, mode: ForwardMode, feats: Dict[str, np.ndarray], idx: np.ndarray, device: str):
    batch = make_batch_dict(feats, idx, device)
    if mode.kind == "kwargs":
        if mode.name == "filtered_kwargs":
            payload = filter_kwargs_for_forward(model, batch)
        else:
            payload = {k: batch[k] for k in mode.keys if k in batch}
        return model(**payload)
    if mode.kind == "dict":
        return model({k: batch[k] for k in mode.keys if k in batch})
    if mode.kind == "single":
        return model(batch[mode.keys[0].lower()])
    if mode.kind == "positional":
        return model(*[batch[k.lower()] for k in mode.keys if k.lower() in batch])
    raise RuntimeError(f"Unknown forward mode {mode}")


def should_hook_module(name: str, module: nn.Module, include_patterns: List[str], exclude_patterns: List[str]) -> bool:
    lname = name.lower()
    cname = module.__class__.__name__.lower()
    if not name:
        return False
    if any(p in lname or p in cname for p in exclude_patterns):
        return False
    if any(p in lname or p in cname for p in include_patterns):
        return True
    # PyTorch core modules
    if "transformerencoderlayer" in cname or "multiheadattention" in cname:
        return True
    return False


def module_discovery(model: nn.Module) -> pd.DataFrame:
    rows = []
    for name, module in model.named_modules():
        if not name:
            continue
        rows.append({
            "name": name,
            "class": module.__class__.__name__,
            "num_children": len(list(module.children())),
            "num_params": sum(p.numel() for p in module.parameters(recurse=False)),
            "total_params_recursive": sum(p.numel() for p in module.parameters(recurse=True)),
        })
    return pd.DataFrame(rows)


def rep_from_tensor(x: Any) -> Optional[Dict[str, np.ndarray]]:
    if torch is None:
        return None
    if isinstance(x, (tuple, list)):
        for v in x:
            got = rep_from_tensor(v)
            if got is not None:
                return got
        return None
    if isinstance(x, dict):
        for k in ["logits", "hidden_states", "last_hidden_state", "output"]:
            if k in x:
                got = rep_from_tensor(x[k])
                if got is not None:
                    return got
        for v in x.values():
            got = rep_from_tensor(v)
            if got is not None:
                return got
        return None
    if not isinstance(x, torch.Tensor):
        return None

    t = x.detach()
    if t.ndim == 0:
        return None
    # Ensure batch first.
    if t.ndim == 1:
        arr = t.float().reshape(-1, 1).cpu().numpy()
        return {"flat": arr}
    if t.ndim == 2:
        return {"flat": t.float().cpu().numpy()}
    if t.ndim == 3:
        # Assume [B, T, D] or [T, B, D]. Heuristic: batch dimension is smaller/current batch likely first.
        # Most repo models use batch_first=True, but handle common non-batch-first.
        if t.shape[0] <= 4096:
            cls = t[:, 0, :].float().cpu().numpy()
            mean = t.mean(dim=1).float().cpu().numpy()
        else:
            cls = t[0, :, :].float().cpu().numpy()
            mean = t.mean(dim=0).float().cpu().numpy()
        return {"cls": cls, "mean": mean}
    # Higher dims: global flatten if small; otherwise global mean over non-batch dims.
    b = t.shape[0]
    flat_dim = int(np.prod(t.shape[1:]))
    if flat_dim <= 2048:
        return {"flat": t.reshape(b, -1).float().cpu().numpy()}
    return {"mean": t.reshape(b, -1).mean(dim=1, keepdim=True).float().cpu().numpy()}


def extract_representations(
    model: nn.Module,
    mode: ForwardMode,
    train_feats: Dict[str, np.ndarray],
    val_feats: Dict[str, np.ndarray],
    y_train: np.ndarray,
    y_val: np.ndarray,
    device: str,
    batch_size: int,
    include_patterns: List[str],
    exclude_patterns: List[str],
    max_hooks: int,
    max_samples_per_split: int,
    seed: int,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Any]]:
    rng = np.random.default_rng(seed)

    n_train = len(y_train)
    n_val = len(y_val)
    train_idx = np.arange(n_train)
    val_idx = np.arange(n_val)
    if max_samples_per_split and n_train > max_samples_per_split:
        train_idx = rng.choice(train_idx, size=max_samples_per_split, replace=False)
    if max_samples_per_split and n_val > max_samples_per_split:
        val_idx = rng.choice(val_idx, size=max_samples_per_split, replace=False)
    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)

    # Candidate hook modules.
    candidates = []
    for name, module in model.named_modules():
        if should_hook_module(name, module, include_patterns, exclude_patterns):
            candidates.append((name, module.__class__.__name__, module))
    # Avoid hooking too many leaf Linear modules from classifier internals unless named useful.
    if len(candidates) > max_hooks:
        priority = []
        for name, cls, module in candidates:
            lname = name.lower()
            score = 0
            if "embedding" in lname or "embed" in lname:
                score += 50
            if "fusion" in lname or "film" in lname or "gate" in lname:
                score += 45
            if "transformer" in lname or "encoder" in lname:
                score += 40
            if "layers" in lname or "layer" in lname:
                score += 35
            if "classifier" in lname or "head" in lname:
                score += 30
            if "linear" in cls.lower():
                score -= 10
            priority.append((score, name, cls, module))
        priority.sort(key=lambda x: (-x[0], x[1]))
        candidates = [(n, c, m) for _, n, c, m in priority[:max_hooks]]

    captured: Dict[str, Any] = {}
    handles = []

    def make_hook(hname):
        def hook(module, inp, out):
            captured[hname] = out
        return hook

    for name, cls, module in candidates:
        try:
            handles.append(module.register_forward_hook(make_hook(name)))
        except Exception:
            pass

    info = {
        "num_hook_candidates": len(candidates),
        "hooked_modules": [{"name": n, "class": c} for n, c, _ in candidates],
        "train_samples_used": len(train_idx),
        "val_samples_used": len(val_idx),
    }

    reps: Dict[str, Dict[str, List[np.ndarray]]] = {}
    logits_store = {"train": [], "val": []}
    y_store = {"train": y_train[train_idx], "val": y_val[val_idx]}

    def process_split(split: str, feats: Dict[str, np.ndarray], indices: np.ndarray):
        model.eval()
        for start in range(0, len(indices), batch_size):
            idx = indices[start:start + batch_size]
            captured.clear()
            with torch.no_grad():
                out = call_forward(model, mode, feats, idx, device)
            logits = output_to_logits(out)
            if logits is not None:
                logits_store[split].append(logits.detach().float().cpu().numpy())

            # Include model output/logits as a pseudo-stage.
            if logits is not None:
                reps.setdefault("MODEL_OUTPUT_LOGITS", {}).setdefault(split, []).append(
                    logits.detach().float().cpu().numpy()
                )

            for hname, hval in captured.items():
                rd = rep_from_tensor(hval)
                if not rd:
                    continue
                for kind, arr in rd.items():
                    stage = f"{hname}::{kind}"
                    reps.setdefault(stage, {}).setdefault(split, []).append(arr)

            log(f"{split} extracted {min(start + batch_size, len(indices))}/{len(indices)}")

    process_split("train", train_feats, train_idx)
    process_split("val", val_feats, val_idx)

    for h in handles:
        try:
            h.remove()
        except Exception:
            pass

    final_reps: Dict[str, Dict[str, np.ndarray]] = {}
    for stage, d in reps.items():
        if "train" in d and "val" in d:
            try:
                final_reps[stage] = {
                    "train": np.concatenate(d["train"], axis=0),
                    "val": np.concatenate(d["val"], axis=0),
                    "y_train": y_store["train"],
                    "y_val": y_store["val"],
                }
            except Exception as e:
                info.setdefault("concat_errors", []).append({"stage": stage, "error": repr(e)})

    if logits_store["train"] and logits_store["val"]:
        info["model_logits_train_shape"] = list(np.concatenate(logits_store["train"], axis=0).shape)
        info["model_logits_val_shape"] = list(np.concatenate(logits_store["val"], axis=0).shape)

    return final_reps, info


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def weighted_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def acc(y_true, y_pred) -> float:
    return float(accuracy_score(y_true, y_pred))


def standardize_train_val(xtr: np.ndarray, xva: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = xtr.mean(axis=0, keepdims=True)
    sd = xtr.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return (xtr - mu) / sd, (xva - mu) / sd, mu, sd


def centroid_predict(xtr: np.ndarray, ytr: np.ndarray, x: np.ndarray) -> np.ndarray:
    classes = np.unique(ytr)
    centroids = []
    for c in classes:
        centroids.append(xtr[ytr == c].mean(axis=0))
    C = np.stack(centroids, axis=0)
    # squared euclidean
    d = ((x[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
    return classes[d.argmin(axis=1)]


def centroid_margin_stats(xtr: np.ndarray, ytr: np.ndarray, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    classes = np.unique(ytr)
    centroids = np.stack([xtr[ytr == c].mean(axis=0) for c in classes], axis=0)
    d = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    true_pos = np.array([np.where(classes == yy)[0][0] for yy in y])
    true_d = d[np.arange(len(y)), true_pos]
    d_other = d.copy()
    d_other[np.arange(len(y)), true_pos] = np.inf
    nearest_other = np.min(d_other, axis=1)
    # positive margin means closer to true centroid than nearest other.
    margin = nearest_other - true_d
    return {
        "centroid_margin_mean": float(np.mean(margin)),
        "centroid_margin_median": float(np.median(margin)),
        "centroid_margin_p10": float(np.quantile(margin, 0.10)),
        "centroid_margin_frac_negative": float(np.mean(margin < 0)),
    }


def evaluate_stage(stage: str, rep: Dict[str, np.ndarray], seed: int, max_probe_train: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    xtr = np.asarray(rep["train"])
    xva = np.asarray(rep["val"])
    ytr = np.asarray(rep["y_train"]).reshape(-1)
    yva = np.asarray(rep["y_val"]).reshape(-1)

    # Clean dims.
    xtr = xtr.reshape(xtr.shape[0], -1).astype(np.float32)
    xva = xva.reshape(xva.shape[0], -1).astype(np.float32)

    # Replace non-finite.
    xtr = np.nan_to_num(xtr, nan=0.0, posinf=1e6, neginf=-1e6)
    xva = np.nan_to_num(xva, nan=0.0, posinf=1e6, neginf=-1e6)

    xtr_s, xva_s, _, _ = standardize_train_val(xtr, xva)

    row: Dict[str, Any] = {
        "stage": stage,
        "dim": int(xtr_s.shape[1]),
        "n_train": int(xtr_s.shape[0]),
        "n_val": int(xva_s.shape[0]),
    }

    # Original logits stage: direct argmax is meaningful.
    if "MODEL_OUTPUT_LOGITS" in stage or "logit" in stage.lower():
        ytr_pred = xtr_s.argmax(axis=1)
        yva_pred = xva_s.argmax(axis=1)
        row.update({
            "direct_train_macro_f1": macro_f1(ytr, ytr_pred),
            "direct_val_macro_f1": macro_f1(yva, yva_pred),
            "direct_gap_macro_f1": macro_f1(ytr, ytr_pred) - macro_f1(yva, yva_pred),
            "direct_train_acc": acc(ytr, ytr_pred),
            "direct_val_acc": acc(yva, yva_pred),
        })

    # Centroid probe.
    try:
        ytr_c = centroid_predict(xtr_s, ytr, xtr_s)
        yva_c = centroid_predict(xtr_s, ytr, xva_s)
        row.update({
            "centroid_train_macro_f1": macro_f1(ytr, ytr_c),
            "centroid_val_macro_f1": macro_f1(yva, yva_c),
            "centroid_gap_macro_f1": macro_f1(ytr, ytr_c) - macro_f1(yva, yva_c),
            "centroid_train_acc": acc(ytr, ytr_c),
            "centroid_val_acc": acc(yva, yva_c),
        })
    except Exception as e:
        row["centroid_error"] = repr(e)

    margins: Dict[str, Any] = {}
    try:
        mt = centroid_margin_stats(xtr_s, ytr, xtr_s, ytr)
        mv = centroid_margin_stats(xtr_s, ytr, xva_s, yva)
        margins = {
            "stage": stage,
            **{f"train_{k}": v for k, v in mt.items()},
            **{f"val_{k}": v for k, v in mv.items()},
        }
        margins["margin_mean_gap_train_minus_val"] = margins["train_centroid_margin_mean"] - margins["val_centroid_margin_mean"]
        margins["margin_neg_frac_gap_val_minus_train"] = margins["val_centroid_margin_frac_negative"] - margins["train_centroid_margin_frac_negative"]
    except Exception as e:
        margins = {"stage": stage, "margin_error": repr(e)}

    # Linear probe: use SGD to avoid heavy LogisticRegression.
    if SGDClassifier is not None:
        try:
            rng = np.random.default_rng(seed)
            train_idx = np.arange(len(ytr))
            if max_probe_train and len(train_idx) > max_probe_train:
                train_idx = rng.choice(train_idx, size=max_probe_train, replace=False)
            clf = make_pipeline(
                StandardScaler(with_mean=True, with_std=True),
                SGDClassifier(
                    loss="log_loss",
                    penalty="l2",
                    alpha=1e-4,
                    class_weight="balanced",
                    max_iter=1000,
                    tol=1e-4,
                    early_stopping=True,
                    random_state=seed,
                    n_jobs=-1,
                )
            )
            clf.fit(xtr[train_idx], ytr[train_idx])
            ytr_p = clf.predict(xtr)
            yva_p = clf.predict(xva)
            row.update({
                "linear_train_macro_f1": macro_f1(ytr, ytr_p),
                "linear_val_macro_f1": macro_f1(yva, yva_p),
                "linear_gap_macro_f1": macro_f1(ytr, ytr_p) - macro_f1(yva, yva_p),
                "linear_train_acc": acc(ytr, ytr_p),
                "linear_val_acc": acc(yva, yva_p),
            })
        except Exception as e:
            row["linear_probe_error"] = repr(e)

    return row, margins


def infer_locus(stage_df: pd.DataFrame, static_df: pd.DataFrame) -> str:
    lines = []
    lines.append("# F1b Overfit Locus Interpretation\n")
    lines.append("## Reference\n")
    lines.append("```text")
    lines.append(f"Base512 train macro-F1 = {BASE512_REF['train_macro_f1']:.6f}")
    lines.append(f"Base512 val macro-F1   = {BASE512_REF['val_macro_f1']:.6f}")
    lines.append(f"Base512 gap            = {BASE512_REF['gap_macro_f1']:.6f}")
    lines.append(f"L1 anchor val macro-F1 = {L1_REF['val_macro_f1']:.6f}")
    lines.append(f"L1 anchor gap          = {L1_REF['gap_macro_f1']:.6f}")
    lines.append("```")

    if len(static_df):
        lines.append("\n## Static run summary\n")
        cols = [c for c in [
            "alias", "train_macro_f1", "val_macro_f1", "gap_macro_f1",
            "num_layers", "hidden_dim", "num_heads", "classifier_hidden_dim",
        ] if c in static_df.columns]
        lines.append(static_df[cols].to_markdown(index=False))

    if stage_df is None or len(stage_df) == 0:
        lines.append("\n## Dynamic stage audit unavailable\n")
        lines.append("The script could not extract layer/stage representations. Use `model_discovery.csv`, `dynamic_failures.json`, and `source_architecture_scan.csv` to patch hook/model loading exactly instead of guessing.")
        return "\n".join(lines)

    # Only consider stages with linear probes first; fallback centroid.
    metric = "linear_val_macro_f1" if "linear_val_macro_f1" in stage_df.columns else "centroid_val_macro_f1"
    gap_metric = "linear_gap_macro_f1" if "linear_gap_macro_f1" in stage_df.columns else "centroid_gap_macro_f1"

    df = stage_df.copy()
    if metric not in df.columns or gap_metric not in df.columns:
        lines.append("\n## Dynamic stage audit incomplete\n")
        lines.append("No valid probe metrics found.")
        return "\n".join(lines)

    df = df.dropna(subset=[metric, gap_metric])
    if len(df) == 0:
        lines.append("\n## Dynamic stage audit incomplete\n")
        lines.append("Probe metrics are empty after dropping NaN.")
        return "\n".join(lines)

    lines.append("\n## Stage probe summary\n")
    show_cols = [c for c in [
        "run_alias", "stage", "dim",
        "linear_train_macro_f1", "linear_val_macro_f1", "linear_gap_macro_f1",
        "centroid_train_macro_f1", "centroid_val_macro_f1", "centroid_gap_macro_f1",
        "direct_train_macro_f1", "direct_val_macro_f1", "direct_gap_macro_f1",
    ] if c in df.columns]
    lines.append(df[show_cols].sort_values(["run_alias", metric], ascending=[True, False]).head(30).to_markdown(index=False))

    lines.append("\n## How to read this\n")
    lines.append("```text")
    lines.append("If early embedding/fusion stages already show a large train-val probe gap:")
    lines.append("  overfit starts before Transformer; inspect embedding/fusion/raw branch.")
    lines.append("")
    lines.append("If gap is small early but grows at Transformer layer outputs:")
    lines.append("  attention interaction depth is the likely locus; L1 has a causal explanation.")
    lines.append("")
    lines.append("If final representation has acceptable probe gap but MODEL_OUTPUT_LOGITS/direct gap is much larger:")
    lines.append("  classifier/logit boundary is overfitting.")
    lines.append("")
    lines.append("If all stages have modest gap but subtype F1 remains low:")
    lines.append("  the bottleneck is likely intrinsic malware-family overlap, not overfit locus.")
    lines.append("```")

    # Heuristic per run.
    lines.append("\n## Heuristic locus call per run\n")
    for alias, g in df.groupby("run_alias"):
        gg = g.copy()
        # Sort by original extraction order if present.
        gg = gg.reset_index(drop=True)
        best_val = gg[metric].max()
        worst_gap = gg[gap_metric].max()
        first = gg.iloc[0]
        last = gg.iloc[-1]
        logits = gg[gg["stage"].str.contains("MODEL_OUTPUT_LOGITS", case=False, na=False)]
        cls_like = gg[~gg["stage"].str.contains("MODEL_OUTPUT_LOGITS", case=False, na=False)]
        lines.append(f"\n### {alias}")
        lines.append("```text")
        lines.append(f"best_stage_val_probe = {best_val:.6f}")
        lines.append(f"max_stage_gap_probe  = {worst_gap:.6f}")
        if len(logits):
            lr = logits.iloc[0]
            lines.append(f"logits_val_probe/direct = {lr.get(metric, np.nan):.6f}")
            lines.append(f"logits_gap_probe/direct = {lr.get(gap_metric, np.nan):.6f}")
        # Large first-stage gap.
        first_gap = first.get(gap_metric, np.nan)
        last_gap = last.get(gap_metric, np.nan)
        lines.append(f"first_stage = {first['stage']}")
        lines.append(f"first_stage_gap = {first_gap:.6f}" if pd.notnull(first_gap) else "first_stage_gap = NaN")
        lines.append(f"last_stage = {last['stage']}")
        lines.append(f"last_stage_gap = {last_gap:.6f}" if pd.notnull(last_gap) else "last_stage_gap = NaN")

        call = "inconclusive"
        if pd.notnull(first_gap) and first_gap > 0.08:
            call = "early_embedding_or_fusion_gap_possible"
        if pd.notnull(last_gap) and pd.notnull(first_gap) and (last_gap - first_gap) > 0.03:
            call = "gap_grows_in_deeper_representation_attention_or_cls"
        if len(logits) and "direct_gap_macro_f1" in logits.columns:
            dg = logits.iloc[0].get("direct_gap_macro_f1", np.nan)
            if pd.notnull(dg) and pd.notnull(last_gap) and (dg - last_gap) > 0.03:
                call = "classifier_or_logit_boundary_overfit_possible"
        lines.append(f"heuristic_call = {call}")
        lines.append("```")

    return "\n".join(lines)


def run_dynamic_audit_for_run(
    alias: str,
    run_dir: Path,
    root: Path,
    dataset_path: Path,
    out_dir: Path,
    device: str,
    batch_size: int,
    include_patterns: List[str],
    exclude_patterns: List[str],
    max_hooks: int,
    max_samples_per_split: int,
    max_probe_train: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if torch is None:
        return pd.DataFrame(), pd.DataFrame(), {"alias": alias, "error": "torch_not_available"}

    dyn_info: Dict[str, Any] = {"alias": alias, "run_dir": str(run_dir)}

    train_feats, val_feats, y_train, y_val, data_info = load_dataset_npz(dataset_path)
    dyn_info["dataset_info"] = data_info
    if train_feats is None or val_feats is None or y_train is None or y_val is None:
        dyn_info["error"] = "dataset_load_failed_or_labels_missing"
        return pd.DataFrame(), pd.DataFrame(), dyn_info

    cfg = safe_read_json(run_dir / "config.json")
    ckpt_path = find_checkpoint(run_dir)
    dyn_info["checkpoint_path"] = None if ckpt_path is None else str(ckpt_path)
    if ckpt_path is None:
        dyn_info["error"] = "checkpoint_not_found"
        return pd.DataFrame(), pd.DataFrame(), dyn_info

    try:
        ckpt = torch.load(ckpt_path, map_location=device)
    except Exception as e:
        dyn_info["error"] = "checkpoint_load_failed"
        dyn_info["exception"] = traceback.format_exc()
        return pd.DataFrame(), pd.DataFrame(), dyn_info

    dyn_info["checkpoint_summary"] = checkpoint_summary(ckpt)

    model = None
    if isinstance(ckpt, nn.Module):
        model = ckpt
        model.to(device).eval()
        dyn_info["model_load_mode"] = "full_module_checkpoint"
    elif isinstance(ckpt, dict) and isinstance(ckpt.get("model"), nn.Module):
        model = ckpt["model"]
        model.to(device).eval()
        dyn_info["model_load_mode"] = "dict_model_module"
    else:
        state, state_mode = extract_state_dict(ckpt)
        dyn_info["state_dict_mode"] = state_mode
        model, inst_info = instantiate_model_from_state(root, cfg, state, y_train, train_feats, device)
        dyn_info["instantiate_info"] = inst_info

    if model is None:
        dyn_info["error"] = "model_reconstruction_failed"
        return pd.DataFrame(), pd.DataFrame(), dyn_info

    # Save module discovery.
    mod_df = module_discovery(model)
    mod_df.insert(0, "run_alias", alias)
    mod_df.to_csv(out_dir / f"{alias}_model_modules.csv", index=False)

    mode, fwd_info = find_forward_mode(model, train_feats, device)
    dyn_info["forward_info"] = fwd_info
    if mode is None:
        dyn_info["error"] = "forward_mode_not_found"
        return pd.DataFrame(), pd.DataFrame(), dyn_info
    dyn_info["forward_mode"] = mode.__dict__

    reps, extract_info = extract_representations(
        model=model,
        mode=mode,
        train_feats=train_feats,
        val_feats=val_feats,
        y_train=y_train,
        y_val=y_val,
        device=device,
        batch_size=batch_size,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        max_hooks=max_hooks,
        max_samples_per_split=max_samples_per_split,
        seed=seed,
    )
    dyn_info["extract_info"] = extract_info
    dyn_info["num_stages_extracted"] = len(reps)

    rows = []
    margin_rows = []
    for i, (stage, rep) in enumerate(reps.items(), 1):
        log(f"[{alias}] evaluating stage {i}/{len(reps)}: {stage}")
        try:
            row, margin = evaluate_stage(stage, rep, seed=seed, max_probe_train=max_probe_train)
            row["run_alias"] = alias
            margin["run_alias"] = alias
            rows.append(row)
            margin_rows.append(margin)
        except Exception as e:
            rows.append({"run_alias": alias, "stage": stage, "eval_error": traceback.format_exc()})

    stage_df = pd.DataFrame(rows)
    margin_df = pd.DataFrame(margin_rows)
    return stage_df, margin_df, dyn_info


def zip_dir(src_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in src_dir.rglob("*"):
            if p.is_file() and p != zip_path:
                z.write(p, p.relative_to(src_dir.parent))


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--run-dirs", default="base512=03_outputs/06_model,L1=05_test/outputs/F1a2_stage2_depth_classifier/Keff512/F1a2_L1_reduce_num_layers_strong")
    ap.add_argument("--dataset-npz", default="03_outputs/05_dataset/dataset.npz")
    ap.add_argument("--out-dir", default="05_test/outputs/F1b_overfit_locus_audit")
    ap.add_argument("--combined-zip", default="05_test/outputs/F1b_overfit_locus_audit.zip")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--max-hooks", type=int, default=30)
    ap.add_argument("--max-samples-per-split", type=int, default=0, help="0 = use full split. Use e.g. 20000 for faster diagnostic.")
    ap.add_argument("--max-probe-train", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-dynamic", action="store_true")
    ap.add_argument("--hook-include", default="embedding,embed,fusion,film,gate,transformer,encoder,layers,layer,classifier,head,mlp")
    ap.add_argument("--hook-exclude", default="dropout,activation,relu,gelu,softmax,loss")

    args = ap.parse_args()

    root = repo_root_from_here()
    out_dir = resolve_path(args.out_dir, root)
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_zip = resolve_path(args.combined_zip, root)

    # Device fallback.
    if torch is None:
        args.device = "cpu"
    elif args.device == "cuda" and not torch.cuda.is_available():
        log("CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"

    log(f"root={root}")
    log(f"out_dir={out_dir}")
    log(f"device={args.device}")
    log("This is an audit, not a new training experiment.")

    run_dirs = parse_run_dirs(args.run_dirs, root)
    dataset_path = resolve_path(args.dataset_npz, root)

    # Static audit.
    static_rows = []
    for alias, rd in run_dirs.items():
        log(f"static audit: {alias} -> {rd}")
        static_rows.append(static_audit_run(alias, rd))
    static_df = pd.DataFrame(static_rows)
    static_df.to_csv(out_dir / "static_run_summary.csv", index=False)

    # Source architecture scan, useful even if dynamic fails.
    try:
        src_df = scan_source_architecture(root)
        src_df.to_csv(out_dir / "source_architecture_scan.csv", index=False)
    except Exception as e:
        (out_dir / "source_architecture_scan_error.txt").write_text(traceback.format_exc(), encoding="utf-8")

    # Dataset discovery.
    try:
        train_feats, val_feats, y_train, y_val, data_info = load_dataset_npz(dataset_path)
        (out_dir / "dataset_discovery.json").write_text(json.dumps(data_info, indent=2), encoding="utf-8")
    except Exception:
        (out_dir / "dataset_discovery_error.txt").write_text(traceback.format_exc(), encoding="utf-8")

    all_stage = []
    all_margin = []
    dyn_infos = []

    include_patterns = [x.strip().lower() for x in args.hook_include.split(",") if x.strip()]
    exclude_patterns = [x.strip().lower() for x in args.hook_exclude.split(",") if x.strip()]

    if not args.skip_dynamic:
        for alias, rd in run_dirs.items():
            if not rd.exists():
                dyn_infos.append({"alias": alias, "run_dir": str(rd), "error": "run_dir_not_found"})
                continue
            log(f"dynamic audit: {alias}")
            try:
                stage_df, margin_df, info = run_dynamic_audit_for_run(
                    alias=alias,
                    run_dir=rd,
                    root=root,
                    dataset_path=dataset_path,
                    out_dir=out_dir,
                    device=args.device,
                    batch_size=args.batch_size,
                    include_patterns=include_patterns,
                    exclude_patterns=exclude_patterns,
                    max_hooks=args.max_hooks,
                    max_samples_per_split=args.max_samples_per_split,
                    max_probe_train=args.max_probe_train,
                    seed=args.seed,
                )
                dyn_infos.append(info)
                if len(stage_df):
                    all_stage.append(stage_df)
                if len(margin_df):
                    all_margin.append(margin_df)
            except Exception:
                dyn_infos.append({"alias": alias, "run_dir": str(rd), "error": "dynamic_exception", "traceback": traceback.format_exc()})
    else:
        dyn_infos.append({"note": "dynamic audit skipped by --skip-dynamic"})

    (out_dir / "dynamic_audit_info.json").write_text(json.dumps(dyn_infos, indent=2, default=str), encoding="utf-8")

    stage_df = pd.concat(all_stage, ignore_index=True) if all_stage else pd.DataFrame()
    margin_df = pd.concat(all_margin, ignore_index=True) if all_margin else pd.DataFrame()

    if len(stage_df):
        stage_df.to_csv(out_dir / "stage_probe_metrics.csv", index=False)
    else:
        (out_dir / "stage_probe_metrics_EMPTY.txt").write_text(
            "No dynamic stage metrics were produced. Check dynamic_audit_info.json and *_model_modules.csv if present.\n",
            encoding="utf-8",
        )

    if len(margin_df):
        margin_df.to_csv(out_dir / "stage_centroid_margin_metrics.csv", index=False)

    report = infer_locus(stage_df, static_df)
    (out_dir / "F1b_overfit_locus_report.md").write_text(report, encoding="utf-8")

    # Quick stdout summary.
    log("static summary:")
    if len(static_df):
        cols = [c for c in ["alias", "train_macro_f1", "val_macro_f1", "gap_macro_f1", "num_layers", "hidden_dim", "num_heads", "classifier_hidden_dim"] if c in static_df.columns]
        print(static_df[cols].to_string(index=False), flush=True)
    if len(stage_df):
        log("top dynamic stage metrics:")
        metric = "linear_val_macro_f1" if "linear_val_macro_f1" in stage_df.columns else "centroid_val_macro_f1"
        cols = [c for c in ["run_alias", "stage", "dim", "linear_val_macro_f1", "linear_gap_macro_f1", "centroid_val_macro_f1", "centroid_gap_macro_f1", "direct_val_macro_f1", "direct_gap_macro_f1", "diagnosis"] if c in stage_df.columns]
        print(stage_df.sort_values(["run_alias", metric], ascending=[True, False])[cols].head(20).to_string(index=False), flush=True)
    else:
        log("No dynamic stage metrics. This is not a conclusion; inspect discovery outputs.")

    zip_dir(out_dir, combined_zip)
    log(f"zip={combined_zip}")
    log("DONE")


if __name__ == "__main__":
    main()
