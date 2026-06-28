#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F1b Overfit Locus Audit V2

Fixes v1 issues seen in Kaggle:
1) PyTorch 2.6 changed torch.load default weights_only=True; local training checkpoint
   can fail with TorchVersion unpickling. V2 has --trust-local-checkpoint to load
   user-generated local checkpoints with weights_only=False.
2) Model source is 02_src/06_model.py, whose module name cannot be imported by
   plain import because it starts with a digit. V2 loads it via importlib.util.
3) D3 forward likely needs bin_ids, offsets, and raw_scaled. V2 reconstructs raw_scaled
   from 01_split/train_raw.csv and val_raw.csv using train min/max on feature_names.

Goal:
- Locate overfit stage, not train new variants.
- Hook embedding/fusion, projection, transformer layers, CLS classifier input, classifier output/logits.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import sys
import traceback
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None

try:
    from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
except Exception:
    f1_score = None
    accuracy_score = None
    SGDClassifier = None
    StandardScaler = None
    make_pipeline = None


BASE512_REF = {"train_macro_f1": 0.910253, "val_macro_f1": 0.810094, "gap_macro_f1": 0.100158}
L1_REF = {"train_macro_f1": 0.911431, "val_macro_f1": 0.814224, "gap_macro_f1": 0.097207}


def log(msg: str) -> None:
    print(f"[F1bV2] {msg}", flush=True)


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


def safe_json(path: Path) -> Dict[str, Any]:
    try:
        return load_json(path) if path.exists() else {}
    except Exception:
        return {}


def parse_run_dirs(s: str, root: Path) -> Dict[str, Path]:
    out = {}
    for part in [x.strip() for x in s.split(",") if x.strip()]:
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = resolve_path(v.strip(), root)
        else:
            out[Path(part).name] = resolve_path(part, root)
    return out


def maybe_get(d: Dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def flatten_f1(prefix: str, rep: Dict[str, Any]) -> Dict[str, float]:
    out = {}
    if not isinstance(rep, dict):
        return out
    if "per_class" in rep and isinstance(rep["per_class"], dict):
        items = rep["per_class"].items()
    else:
        items = []
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
    hp = run_dir / "history.csv"
    if hp.exists():
        try:
            h = pd.read_csv(hp)
            row["history_epochs"] = len(h)
            if "val_macro_f1" in h.columns and len(h):
                i = h["val_macro_f1"].idxmax()
                row["history_best_epoch"] = int(h.loc[i, "epoch"]) if "epoch" in h.columns else int(i)
                row["history_best_val_macro_f1"] = float(h.loc[i, "val_macro_f1"])
                if "train_macro_f1" in h.columns:
                    row["history_train_at_best_val"] = float(h.loc[i, "train_macro_f1"])
                    row["history_gap_at_best_val"] = row["history_train_at_best_val"] - row["history_best_val_macro_f1"]
        except Exception as e:
            row["history_error"] = repr(e)
    return row


def find_checkpoint(run_dir: Path) -> Optional[Path]:
    for n in ["best_model.pt", "model_best.pt", "checkpoint_best.pt", "best.pt", "checkpoint.pt", "model.pt"]:
        p = run_dir / n
        if p.exists():
            return p
    pts = sorted(run_dir.glob("*.pt"))
    return pts[0] if pts else None


def load_torch_checkpoint(path: Path, device: str, trust: bool) -> Any:
    if torch is None:
        raise RuntimeError("torch unavailable")
    if trust:
        return torch.load(path, map_location=device, weights_only=False)
    # Try safe default first.
    try:
        return torch.load(path, map_location=device)
    except Exception as e:
        # Try allowlisting TorchVersion only; still safe-ish for common PyTorch metadata.
        try:
            from torch.serialization import safe_globals
            from torch.torch_version import TorchVersion
            with safe_globals([TorchVersion]):
                return torch.load(path, map_location=device)
        except Exception:
            raise e


def state_dict_from_ckpt(ckpt: Any) -> Tuple[Optional[Dict[str, Any]], str]:
    if isinstance(ckpt, dict):
        for k in ["model_state_dict", "state_dict", "net_state_dict"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k], k
        if all(isinstance(k, str) for k in ckpt.keys()) and any(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt, "checkpoint_is_state_dict"
    if nn is not None and isinstance(ckpt, nn.Module):
        return ckpt.state_dict(), "module_state_dict"
    return None, "not_found"


def load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def infer_model_config(cfg: Dict[str, Any], ds_info: Dict[str, Any], y_train: np.ndarray) -> Dict[str, Any]:
    def cfg_get(k, default):
        if k in cfg:
            return cfg[k]
        for dname in ["model", "model_config", "training", "args"]:
            d = cfg.get(dname, {})
            if isinstance(d, dict) and k in d:
                return d[k]
        return default

    n_features = int(ds_info.get("n_features", 55))
    num_bins = int(ds_info.get("num_bins", 512))
    n_classes = int(len(np.unique(y_train)))
    return {
        "num_bins": int(cfg_get("num_bins", cfg_get("K", num_bins))),
        "n_features": int(cfg_get("n_features", cfg_get("num_features", n_features))),
        "num_classes": int(cfg_get("num_classes", n_classes)),
        "value_dim": int(cfg_get("value_dim", 32)),
        "feature_dim": int(cfg_get("feature_dim", 32)),
        "hidden_dim": int(cfg_get("hidden_dim", 128)),
        "num_layers": int(cfg_get("num_layers", 3)),
        "num_heads": int(cfg_get("num_heads", 4)),
        "dropout": float(cfg_get("dropout", 0.1)),
        "classifier_hidden_dim": int(cfg_get("classifier_hidden_dim", 128)),
        "classifier_dropout": float(cfg_get("classifier_dropout", 0.1)),
        "gate_init": float(cfg_get("gate_init", 0.0)),
    }


def build_model(root: Path, run_dir: Path, dataset_info: Dict[str, Any], y_train: np.ndarray, device: str, trust: bool) -> Tuple[nn.Module, Dict[str, Any]]:
    ckpt_path = find_checkpoint(run_dir)
    if ckpt_path is None:
        raise FileNotFoundError(f"no checkpoint in {run_dir}")
    ckpt = load_torch_checkpoint(ckpt_path, device, trust)
    sd, sd_mode = state_dict_from_ckpt(ckpt)
    if sd is None:
        raise RuntimeError(f"cannot extract state_dict from {ckpt_path}")

    cfg = safe_json(run_dir / "config.json")
    mcfg = infer_model_config(cfg, dataset_info, y_train)

    # Try 06_model.py first.
    mod_paths = [root / "02_src" / "06_model.py", root / "02_src" / "07_train.py"]
    errors = []
    best = None
    best_score = None
    best_info = None

    for mp in mod_paths:
        if not mp.exists():
            continue
        try:
            mod = load_module_from_path(f"_f1b_model_{mp.stem}_{abs(hash(str(mp)))%100000}", mp)
        except Exception as e:
            errors.append({"module_path": str(mp), "load_error": traceback.format_exc()})
            continue

        for cname in ["D3C2D3Transformer", "FusionAblationTransformer"]:
            cls = getattr(mod, cname, None)
            if cls is None:
                continue
            attempts = []
            kw_candidates = [
                mcfg,
                {k: v for k, v in mcfg.items() if k in inspect.signature(cls).parameters},
            ]
            # Some train-time class uses fusion_id/run_id.
            tmp = dict(mcfg)
            tmp.setdefault("fusion_id", "D3")
            tmp.setdefault("run_id", "D3")
            kw_candidates.append({k: v for k, v in tmp.items() if k in inspect.signature(cls).parameters})
            for kwargs in kw_candidates:
                try:
                    model = cls(**kwargs)
                    missing, unexpected = model.load_state_dict(sd, strict=False)
                    score = len(missing) + len(unexpected)
                    attempts.append({
                        "class": cname,
                        "kwargs": kwargs,
                        "missing": len(missing),
                        "unexpected": len(unexpected),
                        "first_missing": list(missing)[:10],
                        "first_unexpected": list(unexpected)[:10],
                    })
                    if best_score is None or score < best_score:
                        best_score = score
                        best = model
                        best_info = attempts[-1] | {"module_path": str(mp), "state_dict_mode": sd_mode, "checkpoint_path": str(ckpt_path)}
                except Exception:
                    attempts.append({"class": cname, "kwargs": kwargs, "error": traceback.format_exc()})
            errors.append({"module_path": str(mp), "class": cname, "attempts": attempts})

    if best is None:
        raise RuntimeError("model reconstruction failed:\n" + json.dumps(errors, indent=2, default=str))

    best.to(device)
    best.eval()
    return best, {"best": best_info, "attempts": errors, "model_config": mcfg}


def load_dataset(dataset_npz: Path, train_raw: Path, val_raw: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    data = np.load(dataset_npz, allow_pickle=True)
    need = ["X_train_bin", "X_train_offset", "y_train", "X_val_bin", "X_val_offset", "y_val"]
    missing = [k for k in need if k not in data.files]
    if missing:
        raise KeyError(f"dataset missing keys: {missing}")

    Xtr_bin = np.asarray(data["X_train_bin"])
    Xtr_off = np.asarray(data["X_train_offset"])
    ytr = np.asarray(data["y_train"]).reshape(-1)
    Xva_bin = np.asarray(data["X_val_bin"])
    Xva_off = np.asarray(data["X_val_offset"])
    yva = np.asarray(data["y_val"]).reshape(-1)
    feature_names = [str(x) for x in np.asarray(data["feature_names"]).tolist()] if "feature_names" in data.files else [f"f{i}" for i in range(Xtr_bin.shape[1])]
    num_bins = int(np.asarray(data["num_bins"]).reshape(-1)[0]) if "num_bins" in data.files else 512

    def raw_scaled(csv_path: Path, train_ref: Optional[pd.DataFrame] = None):
        if not csv_path.exists():
            return None
        df = pd.read_csv(csv_path)
        cols = [c for c in feature_names if c in df.columns]
        if len(cols) != len(feature_names):
            # fallback: first n numeric non-label columns
            exclude = {"label", "class", "category", "label_l1", "label_l2", "label_l3", "Class", "Category"}
            num_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
            cols = num_cols[:len(feature_names)]
        x = df[cols].to_numpy(dtype=np.float32)
        return x, cols

    raw_tr = raw_scaled(train_raw)
    raw_va = raw_scaled(val_raw)
    if raw_tr is not None and raw_va is not None:
        Xtr_raw, cols = raw_tr
        Xva_raw, cols2 = raw_va
        mn = np.nanmin(Xtr_raw, axis=0, keepdims=True)
        mx = np.nanmax(Xtr_raw, axis=0, keepdims=True)
        den = mx - mn
        den[den < 1e-8] = 1.0
        Xtr_cont = np.clip((Xtr_raw - mn) / den, 0.0, 1.0).astype(np.float32)
        Xva_cont = np.clip((Xva_raw - mn) / den, 0.0, 1.0).astype(np.float32)
        raw_info = {"available": True, "columns_used": cols[:10], "num_columns": len(cols)}
    else:
        # fallback: use offset in [0,1] as a weak continuous proxy
        Xtr_cont = np.asarray(Xtr_off, dtype=np.float32)
        Xva_cont = np.asarray(Xva_off, dtype=np.float32)
        raw_info = {"available": False, "fallback": "offset_as_continuous_proxy"}

    ds = {
        "Xtr_bin": Xtr_bin.astype(np.int64),
        "Xtr_off": Xtr_off.astype(np.float32),
        "Xtr_cont": Xtr_cont.astype(np.float32),
        "ytr": ytr.astype(np.int64),
        "Xva_bin": Xva_bin.astype(np.int64),
        "Xva_off": Xva_off.astype(np.float32),
        "Xva_cont": Xva_cont.astype(np.float32),
        "yva": yva.astype(np.int64),
    }
    info = {
        "dataset_path": str(dataset_npz),
        "keys": list(data.files),
        "n_train": int(len(ytr)),
        "n_val": int(len(yva)),
        "n_features": int(Xtr_bin.shape[1]),
        "num_bins": int(num_bins),
        "feature_names_preview": feature_names[:10],
        "raw_info": raw_info,
    }
    return ds, info


def tensor_batch(arr: np.ndarray, idx: np.ndarray, device: str, dtype: str):
    if dtype == "long":
        return torch.as_tensor(arr[idx], dtype=torch.long, device=device)
    return torch.as_tensor(arr[idx], dtype=torch.float32, device=device)


def try_forward_modes(model: nn.Module, ds: Dict[str, np.ndarray], device: str) -> Tuple[str, Dict[str, Any]]:
    idx = np.arange(min(8, len(ds["ytr"])))
    b = tensor_batch(ds["Xtr_bin"], idx, device, "long")
    o = tensor_batch(ds["Xtr_off"], idx, device, "float")
    c = tensor_batch(ds["Xtr_cont"], idx, device, "float")

    modes = [
        ("pos_bin_offset_raw", lambda: model(b, o, c)),
        ("pos_bin_offset", lambda: model(b, o)),
        ("kw_bin_offset_raw", lambda: model(bin_ids=b, offsets=o, raw_scaled=c)),
        ("kw_tokens_offset_raw", lambda: model(tokens=b, offsets=o, raw_scaled=c)),
        ("kw_xbin_xoff_xraw", lambda: model(x_bin=b, x_offset=o, x_cont=c)),
        ("kw_input_ids_offsets_raw", lambda: model(input_ids=b, offsets=o, raw_scaled=c)),
    ]
    attempts = []
    with torch.no_grad():
        for name, fn in modes:
            try:
                out = fn()
                logits = output_to_logits(out)
                ok = logits is not None and logits.shape[0] == len(idx)
                attempts.append({"mode": name, "ok": bool(ok), "out_type": str(type(out)), "logits_shape": None if logits is None else list(logits.shape)})
                if ok:
                    return name, {"attempts": attempts}
            except Exception as e:
                attempts.append({"mode": name, "ok": False, "error": repr(e)})
    raise RuntimeError("no forward mode worked:\n" + json.dumps(attempts, indent=2))


def call_model(model: nn.Module, mode: str, b, o, c):
    if mode == "pos_bin_offset_raw":
        return model(b, o, c)
    if mode == "pos_bin_offset":
        return model(b, o)
    if mode == "kw_bin_offset_raw":
        return model(bin_ids=b, offsets=o, raw_scaled=c)
    if mode == "kw_tokens_offset_raw":
        return model(tokens=b, offsets=o, raw_scaled=c)
    if mode == "kw_xbin_xoff_xraw":
        return model(x_bin=b, x_offset=o, x_cont=c)
    if mode == "kw_input_ids_offsets_raw":
        return model(input_ids=b, offsets=o, raw_scaled=c)
    raise ValueError(mode)


def output_to_logits(out: Any):
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


def module_table(model: nn.Module) -> pd.DataFrame:
    rows = []
    for name, m in model.named_modules():
        if not name:
            continue
        rows.append({
            "name": name,
            "class": m.__class__.__name__,
            "children": len(list(m.children())),
            "params_direct": sum(p.numel() for p in m.parameters(recurse=False)),
            "params_recursive": sum(p.numel() for p in m.parameters(recurse=True)),
        })
    return pd.DataFrame(rows)


def choose_hook_modules(model: nn.Module, max_hooks: int) -> List[Tuple[str, nn.Module]]:
    rows = []
    for name, m in model.named_modules():
        if not name:
            continue
        lname = name.lower()
        cls = m.__class__.__name__.lower()
        score = 0
        if "embedding" in lname or "embed" in lname:
            score += 100
        if "input" in lname or "projection" in lname or "proj" in lname:
            score += 80
        if "encoder.layers." in lname:
            score += 90
        if "transformerencoderlayer" in cls:
            score += 90
        if "encoder" == lname or lname.endswith("encoder"):
            score += 50
        if "classifier" in lname:
            score += 70
        if "multiheadattention" in cls or "self_attn" in lname:
            score += 60
        if "dropout" in lname or "activation" in lname:
            score -= 100
        if score > 0:
            rows.append((score, name, m))
    rows.sort(key=lambda x: (-x[0], x[1]))
    # Ensure unique and not too nested noisy.
    return [(n, m) for _, n, m in rows[:max_hooks]]


def rep_from_output(x: Any) -> Optional[np.ndarray]:
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
    if t.ndim == 1:
        return t.reshape(-1, 1).cpu().numpy()
    if t.ndim == 2:
        return t.cpu().numpy()
    if t.ndim == 3:
        # batch_first usually true in this repo.
        return t[:, 0, :].cpu().numpy()
    b = t.shape[0]
    return t.reshape(b, -1).mean(dim=1, keepdim=True).cpu().numpy()


def extract_stages(model: nn.Module, mode: str, ds: Dict[str, np.ndarray], device: str, batch_size: int, max_hooks: int, max_samples: int, seed: int):
    rng = np.random.default_rng(seed)
    idx_tr = np.arange(len(ds["ytr"]))
    idx_va = np.arange(len(ds["yva"]))
    if max_samples and len(idx_tr) > max_samples:
        idx_tr = np.sort(rng.choice(idx_tr, max_samples, replace=False))
    if max_samples and len(idx_va) > max_samples:
        idx_va = np.sort(rng.choice(idx_va, max_samples, replace=False))

    modules = choose_hook_modules(model, max_hooks)
    captured = {}
    handles = []

    # classifier input is especially important: CLS before MLP.
    def cls_pre_hook(mod, inp):
        if inp and torch.is_tensor(inp[0]):
            captured["CLASSIFIER_PRE_INPUT_CLS"] = inp[0]

    def mk_hook(name):
        def hook(mod, inp, out):
            captured[name] = out
        return hook

    for name, mod in modules:
        try:
            handles.append(mod.register_forward_hook(mk_hook(name)))
        except Exception:
            pass
    if hasattr(model, "classifier"):
        try:
            handles.append(model.classifier.register_forward_pre_hook(cls_pre_hook))
        except Exception:
            pass

    store = {}

    def process(split, indices):
        is_train = split == "train"
        for st in range(0, len(indices), batch_size):
            ids = indices[st:st+batch_size]
            captured.clear()
            Xb = ds["Xtr_bin"] if is_train else ds["Xva_bin"]
            Xo = ds["Xtr_off"] if is_train else ds["Xva_off"]
            Xc = ds["Xtr_cont"] if is_train else ds["Xva_cont"]
            b = tensor_batch(Xb, ids, device, "long")
            o = tensor_batch(Xo, ids, device, "float")
            c = tensor_batch(Xc, ids, device, "float")
            with torch.no_grad():
                out = call_model(model, mode, b, o, c)
            logits = output_to_logits(out)
            if logits is not None:
                store.setdefault("MODEL_OUTPUT_LOGITS", {}).setdefault(split, []).append(logits.detach().float().cpu().numpy())
            for name, val in list(captured.items()):
                arr = rep_from_output(val)
                if arr is not None and arr.shape[0] == len(ids):
                    store.setdefault(name, {}).setdefault(split, []).append(arr)
            log(f"{split} {min(st+batch_size, len(indices))}/{len(indices)}")
    process("train", idx_tr)
    process("val", idx_va)

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
                "ytr": ds["ytr"][idx_tr],
                "yva": ds["yva"][idx_va],
            }
    info = {
        "hooked_modules": [{"name": n, "class": m.__class__.__name__} for n, m in modules],
        "n_train_used": int(len(idx_tr)),
        "n_val_used": int(len(idx_va)),
        "num_stages": int(len(reps)),
    }
    return reps, info


def mf1(y, p): return float(f1_score(y, p, average="macro", zero_division=0))
def wf1(y, p): return float(f1_score(y, p, average="weighted", zero_division=0))
def acc(y, p): return float(accuracy_score(y, p))


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
    true_d = d[np.arange(len(y)), pos]
    other = d.copy()
    other[np.arange(len(y)), pos] = np.inf
    margin = np.min(other, axis=1) - true_d
    return {
        "margin_mean": float(np.mean(margin)),
        "margin_median": float(np.median(margin)),
        "margin_p10": float(np.quantile(margin, 0.10)),
        "margin_frac_negative": float(np.mean(margin < 0)),
    }


def eval_stage(alias: str, stage: str, rep: Dict[str, np.ndarray], seed: int, max_probe_train: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    Xtr = np.asarray(rep["Xtr"]).reshape(len(rep["ytr"]), -1).astype(np.float32)
    Xva = np.asarray(rep["Xva"]).reshape(len(rep["yva"]), -1).astype(np.float32)
    ytr = np.asarray(rep["ytr"]).astype(int)
    yva = np.asarray(rep["yva"]).astype(int)
    Xtr = np.nan_to_num(Xtr, nan=0, posinf=1e6, neginf=-1e6)
    Xva = np.nan_to_num(Xva, nan=0, posinf=1e6, neginf=-1e6)
    Xtr_s, Xva_s = standardize(Xtr, Xva)

    row = {"run_alias": alias, "stage": stage, "dim": int(Xtr_s.shape[1]), "n_train": len(ytr), "n_val": len(yva)}
    if "LOGITS" in stage.upper():
        ptr = Xtr.argmax(axis=1)
        pva = Xva.argmax(axis=1)
        row.update({
            "direct_train_macro_f1": mf1(ytr, ptr),
            "direct_val_macro_f1": mf1(yva, pva),
            "direct_gap_macro_f1": mf1(ytr, ptr) - mf1(yva, pva),
            "direct_train_acc": acc(ytr, ptr),
            "direct_val_acc": acc(yva, pva),
        })

    try:
        ptr = centroid_pred(Xtr_s, ytr, Xtr_s)
        pva = centroid_pred(Xtr_s, ytr, Xva_s)
        row.update({
            "centroid_train_macro_f1": mf1(ytr, ptr),
            "centroid_val_macro_f1": mf1(yva, pva),
            "centroid_gap_macro_f1": mf1(ytr, ptr) - mf1(yva, pva),
            "centroid_train_acc": acc(ytr, ptr),
            "centroid_val_acc": acc(yva, pva),
        })
    except Exception:
        row["centroid_error"] = traceback.format_exc()

    margin_row = {"run_alias": alias, "stage": stage}
    try:
        mt = centroid_margin(Xtr_s, ytr, Xtr_s, ytr)
        mv = centroid_margin(Xtr_s, ytr, Xva_s, yva)
        margin_row.update({f"train_{k}": v for k, v in mt.items()})
        margin_row.update({f"val_{k}": v for k, v in mv.items()})
        margin_row["margin_mean_gap_train_minus_val"] = margin_row["train_margin_mean"] - margin_row["val_margin_mean"]
        margin_row["margin_neg_frac_gap_val_minus_train"] = margin_row["val_margin_frac_negative"] - margin_row["train_margin_frac_negative"]
    except Exception:
        margin_row["margin_error"] = traceback.format_exc()

    if make_pipeline is not None:
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
                "linear_train_macro_f1": mf1(ytr, ptr),
                "linear_val_macro_f1": mf1(yva, pva),
                "linear_gap_macro_f1": mf1(ytr, ptr) - mf1(yva, pva),
                "linear_train_acc": acc(ytr, ptr),
                "linear_val_acc": acc(yva, pva),
            })
        except Exception:
            row["linear_error"] = traceback.format_exc()
    return row, margin_row


def write_report(out_dir: Path, static_df: pd.DataFrame, stage_df: pd.DataFrame, margin_df: pd.DataFrame, infos: List[Dict[str, Any]]):
    lines = []
    lines.append("# F1b Overfit Locus Audit V2 Report\n")
    lines.append("## Reference\n")
    lines.append("```text")
    lines.append(f"Base512 val macro-F1 = {BASE512_REF['val_macro_f1']:.6f}, gap = {BASE512_REF['gap_macro_f1']:.6f}")
    lines.append(f"L1 anchor val macro-F1 = {L1_REF['val_macro_f1']:.6f}, gap = {L1_REF['gap_macro_f1']:.6f}")
    lines.append("```")
    lines.append("\n## Static summary\n")
    if len(static_df):
        cols = [c for c in ["alias","exists","has_checkpoint","train_macro_f1","val_macro_f1","gap_macro_f1","num_layers","hidden_dim","num_heads","classifier_hidden_dim"] if c in static_df.columns]
        lines.append(static_df[cols].to_markdown(index=False))
    if len(stage_df) == 0:
        lines.append("\n## Dynamic audit failed or unavailable\n")
        lines.append("No stage metrics were produced. Read `dynamic_audit_info.json`, `model_modules.csv`, and source scan outputs.")
        (out_dir / "F1b_overfit_locus_report_v2.md").write_text("\n".join(lines), encoding="utf-8")
        return

    lines.append("\n## Stage probe metrics, sorted by pipeline-ish order\n")
    show = [c for c in ["run_alias","stage","dim","direct_train_macro_f1","direct_val_macro_f1","direct_gap_macro_f1","linear_train_macro_f1","linear_val_macro_f1","linear_gap_macro_f1","centroid_train_macro_f1","centroid_val_macro_f1","centroid_gap_macro_f1"] if c in stage_df.columns]
    lines.append(stage_df[show].to_markdown(index=False))

    lines.append("\n## Interpretation rule\n")
    lines.append("```text")
    lines.append("Large gap already at embedding/projection => overfit starts early in representation/fusion.")
    lines.append("Gap grows at encoder.layers.* => attention interaction/depth is locus.")
    lines.append("CLASSIFIER_PRE_INPUT_CLS okay but MODEL_OUTPUT_LOGITS/direct much worse => classifier/logit boundary locus.")
    lines.append("No stage has extreme gap but subtype F1 remains low => overlap issue dominates.")
    lines.append("```")

    lines.append("\n## Heuristic call\n")
    for alias, g in stage_df.groupby("run_alias"):
        lines.append(f"\n### {alias}")
        lines.append("```text")
        # Extract stages.
        def find_stage(patterns):
            mask = False
            for p in patterns:
                m = g["stage"].str.contains(p, case=False, regex=True, na=False)
                mask = m if isinstance(mask, bool) else (mask | m)
            return g[mask] if not isinstance(mask, bool) else g.iloc[[]]

        metric_gap = "linear_gap_macro_f1" if "linear_gap_macro_f1" in g.columns and g["linear_gap_macro_f1"].notna().any() else "centroid_gap_macro_f1"
        metric_val = "linear_val_macro_f1" if "linear_val_macro_f1" in g.columns and g["linear_val_macro_f1"].notna().any() else "centroid_val_macro_f1"
        emb = find_stage(["embedding", "embed", "input", "proj"])
        enc = find_stage(["encoder\\.layers", "transformerencoderlayer"])
        cls = find_stage(["CLASSIFIER_PRE_INPUT_CLS"])
        logits = find_stage(["MODEL_OUTPUT_LOGITS"])
        def best(df, col):
            if len(df) == 0 or col not in df or df[col].dropna().empty:
                return np.nan
            return float(df[col].dropna().max())
        lines.append(f"best_val_probe = {best(g, metric_val):.6f}")
        lines.append(f"max_gap_probe = {best(g, metric_gap):.6f}")
        lines.append(f"early_embedding_max_gap = {best(emb, metric_gap):.6f}")
        lines.append(f"encoder_layer_max_gap = {best(enc, metric_gap):.6f}")
        lines.append(f"cls_input_gap = {best(cls, metric_gap):.6f}")
        if len(logits) and "direct_gap_macro_f1" in logits.columns and logits["direct_gap_macro_f1"].notna().any():
            lines.append(f"logits_direct_gap = {float(logits['direct_gap_macro_f1'].dropna().iloc[0]):.6f}")
            lines.append(f"logits_direct_val = {float(logits['direct_val_macro_f1'].dropna().iloc[0]):.6f}")
        call = "inconclusive"
        emb_gap = best(emb, metric_gap)
        enc_gap = best(enc, metric_gap)
        cls_gap = best(cls, metric_gap)
        log_gap = np.nan
        if len(logits) and "direct_gap_macro_f1" in logits.columns and logits["direct_gap_macro_f1"].notna().any():
            log_gap = float(logits["direct_gap_macro_f1"].dropna().iloc[0])
        if not np.isnan(emb_gap) and emb_gap > 0.08:
            call = "early_embedding_or_fusion_overfit_possible"
        if not np.isnan(enc_gap) and (np.isnan(emb_gap) or enc_gap - emb_gap > 0.03):
            call = "attention_interaction_layer_gap_growth"
        if not np.isnan(log_gap) and not np.isnan(cls_gap) and log_gap - cls_gap > 0.03:
            call = "classifier_logit_boundary_overfit"
        lines.append(f"heuristic_call = {call}")
        lines.append("```")

    lines.append("\n## Next action\n")
    lines.append("Do not test more hyperparameter combinations until this report identifies the locus. If the dynamic audit is incomplete, patch the hook/load path first.")
    (out_dir / "F1b_overfit_locus_report_v2.md").write_text("\n".join(lines), encoding="utf-8")


def zip_dir(src: Path, dst: Path):
    if dst.exists():
        dst.unlink()
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for p in src.rglob("*"):
            if p.is_file() and p != dst:
                z.write(p, p.relative_to(src.parent))


def source_scan(root: Path, out_dir: Path):
    rows = []
    for fp in [root/"02_src"/"06_model.py", root/"02_src"/"02_embedding.py", root/"02_src"/"07_train.py"]:
        if not fp.exists(): continue
        txt = fp.read_text(encoding="utf-8", errors="ignore")
        interesting = []
        for i, line in enumerate(txt.splitlines(), 1):
            if any(k in line for k in ["class ", "def forward", "Embedding", "Transformer", "classifier", "fusion", "FiLM", "cls", "raw_scaled"]):
                interesting.append(f"{i}: {line.rstrip()}")
        rows.append({"file": str(fp), "preview": "\n".join(interesting[:200])})
    pd.DataFrame(rows).to_csv(out_dir/"source_scan_v2.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", default="base512=03_outputs/06_model,L1=05_test/outputs/F1a2_stage2_depth_classifier/Keff512/F1a2_L1_reduce_num_layers_strong")
    ap.add_argument("--dataset-npz", default="03_outputs/05_dataset/dataset.npz")
    ap.add_argument("--train-raw", default="01_split/train_raw.csv")
    ap.add_argument("--val-raw", default="01_split/val_raw.csv")
    ap.add_argument("--out-dir", default="05_test/outputs/F1b_overfit_locus_audit_v2")
    ap.add_argument("--combined-zip", default="05_test/outputs/F1b_overfit_locus_audit_v2.zip")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--max-hooks", type=int, default=40)
    ap.add_argument("--max-samples-per-split", type=int, default=0)
    ap.add_argument("--max-probe-train", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trust-local-checkpoint", action="store_true", help="Use torch.load(weights_only=False) for user-generated local checkpoints.")
    ap.add_argument("--skip-dynamic", action="store_true")
    args = ap.parse_args()

    root = repo_root_from_here()
    out_dir = resolve_path(args.out_dir, root)
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_zip = resolve_path(args.combined_zip, root)

    if torch is None:
        args.device = "cpu"
    elif args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    log(f"root={root}")
    log(f"device={args.device}")
    log(f"out_dir={out_dir}")
    log("Audit only; no new model variants are trained.")

    run_dirs = parse_run_dirs(args.run_dirs, root)

    static = []
    for alias, rd in run_dirs.items():
        static.append(static_run(alias, rd))
    static_df = pd.DataFrame(static)
    static_df.to_csv(out_dir/"static_run_summary_v2.csv", index=False)

    source_scan(root, out_dir)

    ds = None
    ds_info = {}
    try:
        ds, ds_info = load_dataset(resolve_path(args.dataset_npz, root), resolve_path(args.train_raw, root), resolve_path(args.val_raw, root))
        (out_dir/"dataset_info_v2.json").write_text(json.dumps(ds_info, indent=2, default=str), encoding="utf-8")
    except Exception:
        (out_dir/"dataset_error_v2.txt").write_text(traceback.format_exc(), encoding="utf-8")

    all_stage = []
    all_margin = []
    infos = []

    if not args.skip_dynamic and ds is not None:
        for alias, rd in run_dirs.items():
            info = {"alias": alias, "run_dir": str(rd)}
            if not rd.exists():
                info["error"] = "run_dir_not_found"
                infos.append(info)
                continue
            try:
                log(f"dynamic audit {alias}")
                model, minfo = build_model(root, rd, ds_info, ds["ytr"], args.device, args.trust_local_checkpoint)
                info["model_info"] = minfo
                mt = module_table(model)
                mt.to_csv(out_dir/f"{alias}_model_modules_v2.csv", index=False)
                mode, finfo = try_forward_modes(model, ds, args.device)
                info["forward_mode"] = mode
                info["forward_info"] = finfo
                reps, einfo = extract_stages(model, mode, ds, args.device, args.batch_size, args.max_hooks, args.max_samples_per_split, args.seed)
                info["extract_info"] = einfo
                rows = []
                mrows = []
                for i, (stage, rep) in enumerate(reps.items(), 1):
                    log(f"{alias} eval stage {i}/{len(reps)} {stage}")
                    row, mr = eval_stage(alias, stage, rep, args.seed, args.max_probe_train)
                    rows.append(row); mrows.append(mr)
                if rows:
                    sdf = pd.DataFrame(rows)
                    all_stage.append(sdf)
                    sdf.to_csv(out_dir/f"{alias}_stage_probe_metrics_v2.csv", index=False)
                if mrows:
                    mdf = pd.DataFrame(mrows)
                    all_margin.append(mdf)
                    mdf.to_csv(out_dir/f"{alias}_stage_centroid_margin_v2.csv", index=False)
            except Exception:
                info["error"] = "dynamic_exception"
                info["traceback"] = traceback.format_exc()
            infos.append(info)

    (out_dir/"dynamic_audit_info_v2.json").write_text(json.dumps(infos, indent=2, default=str), encoding="utf-8")
    stage_df = pd.concat(all_stage, ignore_index=True) if all_stage else pd.DataFrame()
    margin_df = pd.concat(all_margin, ignore_index=True) if all_margin else pd.DataFrame()
    if len(stage_df):
        stage_df.to_csv(out_dir/"stage_probe_metrics_v2.csv", index=False)
    else:
        (out_dir/"stage_probe_metrics_v2_EMPTY.txt").write_text("No stage metrics produced.\n", encoding="utf-8")
    if len(margin_df):
        margin_df.to_csv(out_dir/"stage_centroid_margin_metrics_v2.csv", index=False)

    write_report(out_dir, static_df, stage_df, margin_df, infos)
    zip_dir(out_dir, combined_zip)
    log(f"zip={combined_zip}")
    log("DONE")


if __name__ == "__main__":
    main()
