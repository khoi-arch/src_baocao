#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F1e1a_v3 L1 L3/Family + Top2/Probability Audit

Purpose
-------
No training.

Audit the CURRENT L1 checkpoint at L3/family level:
- full probability vector per sample
- top1/top2 class and top2 gap
- family -> predicted L2 distribution
- family -> mean probability mass over L2 classes
- family ambiguity / overlap signals
- optional diagnostic family-aware smoothing candidates

Data leakage policy
-------------------
If validation output from this script is used to choose smoothing hyperparameters
and the same validation set is later reported as the final performance, that is
validation leakage / tuning-to-val.

Therefore this script separates:
1) DIAGNOSTIC VAL AUDIT:
   Useful to understand root cause and choose broad research direction.
   Not clean for final hyperparameter selection unless you later evaluate on a
   separate untouched test set.

2) TRAIN IN-SAMPLE AUDIT:
   Does not leak validation labels, but is biased because the L1 checkpoint was
   trained on this train set. Useful as sanity, not ideal for parameter design.

Best clean option:
   Generate out-of-fold L1 predictions on the training set or create an internal
   train/calibration split. Then derive smoothing from calibration/OOF, lock it,
   retrain on train, and evaluate once on validation/test.

Inputs
------
- L1 run dir with best_model.pt and config.json
- dataset.npz
- train_raw.csv / val_raw.csv with label_L2/label_L3 or equivalent columns

Outputs
-------
- F1e1a_v3_train_predictions_with_probs_top2_l3.csv
- F1e1a_v3_val_predictions_with_probs_top2_l3.csv
- F1e1a_v3_family_summary_by_split.csv
- F1e1a_v3_family_pred_distribution.csv
- F1e1a_v3_family_top2_distribution.csv
- F1e1a_v3_val_family_smoothing_candidate_DIAGNOSTIC_ONLY.csv
- F1e1a_v3_train_in_sample_family_smoothing_candidate_BIASED.csv
- F1e1a_v3_leakage_policy.md
- F1e1a_v3_report.md
- combined zip
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import math
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


DEFAULT_CLASS_NAMES = ["Benign", "Ransomware", "Spyware", "Trojan"]
DEFAULT_MALWARE = ["Ransomware", "Spyware", "Trojan"]
DEFAULT_LABEL_CANDIDATES_L2 = ["label_L2", "Label_L2", "l2", "L2", "Category", "category", "Class", "class"]
DEFAULT_LABEL_CANDIDATES_L3 = ["label_L3", "Label_L3", "l3", "L3", "Family", "family", "MalwareFamily", "malware_family", "Class", "class"]


def log(msg: str) -> None:
    print(f"[F1e1a_v3] {msg}", flush=True)


def repo_root_from_here() -> Path:
    p = Path(__file__).resolve()
    if p.parent.name == "05_test":
        return p.parents[1]
    return Path.cwd().resolve()


def resolve_path(p: str | Path, root: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (root / p).resolve()


def clean(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


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


def find_checkpoint(run_dir: Path) -> Optional[Path]:
    for n in ["best_model.pt", "model_best.pt", "checkpoint_best.pt", "best.pt", "checkpoint.pt", "model.pt"]:
        p = run_dir / n
        if p.exists():
            return p
    pts = sorted(run_dir.glob("*.pt"))
    return pts[0] if pts else None


def detect_class_names_from_report(run_dir: Path) -> List[str]:
    for fn in ["val_classification_report_best.json", "train_classification_report_best.json"]:
        p = run_dir / fn
        if p.exists():
            try:
                rep = load_json(p)
                names = []
                for k, v in rep.items():
                    if isinstance(v, dict) and "support" in v and k not in {"macro avg", "weighted avg"}:
                        if k.lower() != "accuracy":
                            names.append(k)
                if names:
                    std = [c for c in DEFAULT_CLASS_NAMES if c in names]
                    if len(std) == len(names):
                        return std
                    return names
            except Exception:
                pass
    return DEFAULT_CLASS_NAMES


def load_dataset(dataset_npz: Path, train_raw: Path, val_raw: Path, class_names: List[str]):
    data = np.load(dataset_npz, allow_pickle=True)
    required = ["X_train_bin", "X_train_offset", "y_train", "X_val_bin", "X_val_offset", "y_val"]
    missing = [k for k in required if k not in data.files]
    if missing:
        raise KeyError(f"dataset npz missing keys: {missing}")

    Xtr_bin = np.asarray(data["X_train_bin"], dtype=np.int64)
    Xtr_off = np.asarray(data["X_train_offset"], dtype=np.float32)
    ytr = np.asarray(data["y_train"], dtype=np.int64).reshape(-1)

    Xva_bin = np.asarray(data["X_val_bin"], dtype=np.int64)
    Xva_off = np.asarray(data["X_val_offset"], dtype=np.float32)
    yva = np.asarray(data["y_val"], dtype=np.int64).reshape(-1)

    feature_names = [str(x) for x in np.asarray(data["feature_names"]).tolist()] if "feature_names" in data.files else [f"f{i}" for i in range(Xtr_bin.shape[1])]
    num_bins = int(np.asarray(data["num_bins"]).reshape(-1)[0]) if "num_bins" in data.files else 512

    def load_raw_features_and_labels(path: Path, expected_n: int):
        if not path.exists():
            raise FileNotFoundError(f"raw csv not found: {path}")
        df = pd.read_csv(path)
        if len(df) != expected_n:
            raise ValueError(f"{path} rows={len(df)} but expected {expected_n}; cannot safely align labels")

        # Feature cols for raw scaled values.
        feat_cols = [c for c in feature_names if c in df.columns]
        if len(feat_cols) != len(feature_names):
            exclude = set(DEFAULT_LABEL_CANDIDATES_L2 + DEFAULT_LABEL_CANDIDATES_L3 + [
                "label", "Label", "Class", "Category", "label_L1", "Label_L1", "class", "category"
            ])
            feat_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])][:len(feature_names)]
        if len(feat_cols) != len(feature_names):
            raise ValueError(f"raw feature mismatch in {path}: got {len(feat_cols)}, expected {len(feature_names)}")
        X_raw = df[feat_cols].to_numpy(dtype=np.float32)

        l2_col = next((c for c in DEFAULT_LABEL_CANDIDATES_L2 if c in df.columns), None)
        l3_col = next((c for c in DEFAULT_LABEL_CANDIDATES_L3 if c in df.columns), None)

        # Avoid selecting Class as L3 if Class is just L2. If label_L3 exists, it wins.
        if "label_L3" in df.columns:
            l3_col = "label_L3"
        elif "Label_L3" in df.columns:
            l3_col = "Label_L3"

        labels = {
            "raw_df_columns": list(df.columns),
            "l2_col": l2_col,
            "l3_col": l3_col,
            "label_L2_raw": df[l2_col].map(clean).to_numpy() if l2_col else np.array([""] * len(df), dtype=object),
            "label_L3_raw": df[l3_col].map(clean).to_numpy() if l3_col else np.array([""] * len(df), dtype=object),
        }
        return X_raw, feat_cols, labels

    Rtr, raw_cols, lab_tr = load_raw_features_and_labels(train_raw, len(ytr))
    Rva, _, lab_va = load_raw_features_and_labels(val_raw, len(yva))

    mn = np.nanmin(Rtr, axis=0, keepdims=True)
    mx = np.nanmax(Rtr, axis=0, keepdims=True)
    den = mx - mn
    den[den < 1e-8] = 1.0
    Xtr_raw = np.clip((Rtr - mn) / den, 0.0, 1.0).astype(np.float32)
    Xva_raw = np.clip((Rva - mn) / den, 0.0, 1.0).astype(np.float32)

    def make_values(off_arr, raw_arr):
        mask = np.ones_like(off_arr, dtype=np.float32)
        return np.stack([off_arr.astype(np.float32), raw_arr.astype(np.float32), mask], axis=-1).astype(np.float32)

    def numeric_l2_from_y(y):
        return np.array([class_names[int(i)] if int(i) < len(class_names) else str(i) for i in y], dtype=object)

    ds = {
        "train": {
            "tokens": Xtr_bin,
            "values": make_values(Xtr_off, Xtr_raw),
            "y": ytr,
            "label_L2": lab_tr["label_L2_raw"],
            "label_L3": lab_tr["label_L3_raw"],
            "label_L2_from_y": numeric_l2_from_y(ytr),
        },
        "val": {
            "tokens": Xva_bin,
            "values": make_values(Xva_off, Xva_raw),
            "y": yva,
            "label_L2": lab_va["label_L2_raw"],
            "label_L3": lab_va["label_L3_raw"],
            "label_L2_from_y": numeric_l2_from_y(yva),
        },
    }

    # If raw L2 missing or not matching class names, use y-derived L2.
    for split in ["train", "val"]:
        raw_l2 = pd.Series(ds[split]["label_L2"]).map(clean)
        match_rate = raw_l2.isin(class_names).mean() if len(raw_l2) else 0.0
        if match_rate < 0.80:
            ds[split]["label_L2"] = ds[split]["label_L2_from_y"]
        # If L3 missing, fallback to raw L2 plus numeric y, but mark in info.
        raw_l3 = pd.Series(ds[split]["label_L3"]).map(clean)
        missing_l3_rate = (raw_l3 == "").mean()
        if missing_l3_rate > 0.80:
            ds[split]["label_L3"] = ds[split]["label_L2"]

    info = {
        "dataset_npz": str(dataset_npz),
        "train_raw": str(train_raw),
        "val_raw": str(val_raw),
        "keys": list(data.files),
        "n_train": int(len(ytr)),
        "n_val": int(len(yva)),
        "n_features": int(Xtr_bin.shape[1]),
        "num_bins": int(num_bins),
        "raw_feature_cols_preview": raw_cols[:10],
        "train_l2_col": lab_tr["l2_col"],
        "train_l3_col": lab_tr["l3_col"],
        "val_l2_col": lab_va["l2_col"],
        "val_l3_col": lab_va["l3_col"],
        "train_l3_unique_preview": sorted(pd.Series(ds["train"]["label_L3"]).dropna().astype(str).unique().tolist())[:30],
        "val_l3_unique_preview": sorted(pd.Series(ds["val"]["label_L3"]).dropna().astype(str).unique().tolist())[:30],
        "values_candidate": "offset_raw_one",
    }
    return ds, info


def infer_model_config(cfg: Dict[str, Any], ds_info: Dict[str, Any], num_classes: int) -> Dict[str, Any]:
    return {
        "num_bins": int(cfg_get(cfg, "num_bins", cfg_get(cfg, "K", ds_info["num_bins"]))),
        "n_features": int(cfg_get(cfg, "n_features", cfg_get(cfg, "num_features", ds_info["n_features"]))),
        "num_classes": int(cfg_get(cfg, "num_classes", num_classes)),
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


def build_model(root: Path, run_dir: Path, ds_info: Dict[str, Any], class_names: List[str], device: str, trust: bool):
    ckpt_path = find_checkpoint(run_dir)
    if ckpt_path is None:
        raise FileNotFoundError(f"checkpoint not found in {run_dir}")
    cfg = safe_json(run_dir / "config.json")
    mcfg = infer_model_config(cfg, ds_info, len(class_names))

    mod = load_module_from_path("_f1e1a_v3_model_06_model", root / "02_src" / "06_model.py")
    cls = getattr(mod, "D3C2D3Transformer", None)
    if cls is None:
        raise RuntimeError("D3C2D3Transformer not found in 02_src/06_model.py")
    kwargs = {k: v for k, v in mcfg.items() if k in inspect.signature(cls).parameters}
    model = cls(**kwargs)

    ckpt = load_checkpoint(ckpt_path, device, trust)
    sd, sd_mode = state_dict_from_ckpt(ckpt)
    if sd is None:
        raise RuntimeError(f"state_dict not found in checkpoint: {ckpt_path}")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    info = {
        "checkpoint_path": str(ckpt_path),
        "state_dict_mode": sd_mode,
        "model_kwargs": kwargs,
        "missing": list(missing),
        "unexpected": list(unexpected),
    }
    if missing or unexpected:
        raise RuntimeError("checkpoint load mismatch:\n" + json.dumps(info, indent=2, default=str))
    model.to(device).eval()
    return model, info


@torch.no_grad()
def infer_split(model: nn.Module, split: Dict[str, np.ndarray], class_names: List[str], device: str, batch_size: int) -> pd.DataFrame:
    logits_all = []
    n = len(split["y"])
    for st in range(0, n, batch_size):
        ed = min(n, st + batch_size)
        tokens = torch.as_tensor(split["tokens"][st:ed], dtype=torch.long, device=device)
        values = torch.as_tensor(split["values"][st:ed], dtype=torch.float32, device=device)
        logits = model(tokens, values)
        if not torch.is_tensor(logits):
            if isinstance(logits, dict):
                logits = logits.get("logits", next(iter(logits.values())))
            elif isinstance(logits, (tuple, list)):
                logits = logits[0]
        logits_all.append(logits.detach().float().cpu())
        log(f"infer {st}-{ed}/{n}")

    logits = torch.cat(logits_all, dim=0)
    probs = torch.softmax(logits, dim=1).numpy()
    order = np.argsort(-probs, axis=1)
    top1 = order[:, 0]
    top2 = order[:, 1]
    y = split["y"].astype(int)
    df = pd.DataFrame({
        "sample_idx": np.arange(n),
        "y_true": y,
        "true_L2": [class_names[int(i)] for i in y],
        "true_L2_raw": split["label_L2"],
        "true_L3": split["label_L3"],
        "y_pred": top1,
        "pred_L2": [class_names[int(i)] for i in top1],
        "correct": top1 == y,
        "top1_L2": [class_names[int(i)] for i in top1],
        "top1_prob": probs[np.arange(n), top1],
        "top2_L2": [class_names[int(i)] for i in top2],
        "top2_prob": probs[np.arange(n), top2],
        "top2_gap": probs[np.arange(n), top1] - probs[np.arange(n), top2],
        "true_prob": probs[np.arange(n), y],
        "true_in_top2": np.array([yy in order[i, :2] for i, yy in enumerate(y)], dtype=bool),
    })
    for i, name in enumerate(class_names):
        df[f"prob_{name}"] = probs[:, i]
    malware = [c for c in DEFAULT_MALWARE if c in class_names]
    other_mass = []
    for i, row in df.iterrows():
        true_name = row["true_L2"]
        mass = 0.0
        for m in malware:
            if m != true_name and f"prob_{m}" in df.columns:
                mass += float(row[f"prob_{m}"])
        other_mass.append(mass)
    df["other_malware_prob_mass"] = other_mass
    return df


def entropy_probs(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    if len(p) <= 1:
        return 0.0
    h = -float(np.sum(p * np.log(p)))
    return h / math.log(len(p))


def summarize_family(pred_df: pd.DataFrame, class_names: List[str], malware_classes: List[str], split: str,
                     min_family_support: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    pred_rows = []
    top2_rows = []

    group_cols = ["true_L2", "true_L3"]
    for (true_l2, true_l3), g in pred_df.groupby(group_cols, dropna=False):
        n = int(len(g))
        if n == 0:
            continue
        pred_counts = g["pred_L2"].value_counts().reindex(class_names, fill_value=0)
        pred_rates = pred_counts / n
        top2_counts = g["top2_L2"].value_counts().reindex(class_names, fill_value=0)
        top2_rates = top2_counts / n

        prob_means = {f"mean_prob_{c}": float(g[f"prob_{c}"].mean()) if f"prob_{c}" in g.columns else np.nan for c in class_names}
        pred_dist = pred_rates.to_numpy(dtype=float)
        top2_dist = top2_rates.to_numpy(dtype=float)

        row = {
            "split": split,
            "true_L2": true_l2,
            "true_L3": true_l3,
            "n": n,
            "audit_reliable_support": bool(n >= min_family_support),
            "accuracy": float(g["correct"].mean()),
            "error_rate": float(1.0 - g["correct"].mean()),
            "true_in_top2_rate": float(g["true_in_top2"].mean()),
            "top2_gap_mean": float(g["top2_gap"].mean()),
            "top2_gap_median": float(g["top2_gap"].median()),
            "top1_prob_mean": float(g["top1_prob"].mean()),
            "true_prob_mean": float(g["true_prob"].mean()),
            "other_malware_prob_mass_mean": float(g["other_malware_prob_mass"].mean()),
            "pred_distribution_entropy_norm": entropy_probs(pred_dist),
            "top2_distribution_entropy_norm": entropy_probs(top2_dist),
        }
        row.update(prob_means)
        for c in class_names:
            row[f"pred_rate_{c}"] = float(pred_rates[c])
            row[f"top2_rate_{c}"] = float(top2_rates[c])
        rows.append(row)

        for c in class_names:
            pred_rows.append({
                "split": split,
                "true_L2": true_l2,
                "true_L3": true_l3,
                "n_family": n,
                "pred_L2": c,
                "pred_count": int(pred_counts[c]),
                "pred_rate": float(pred_rates[c]),
            })
            top2_rows.append({
                "split": split,
                "true_L2": true_l2,
                "true_L3": true_l3,
                "n_family": n,
                "top2_L2": c,
                "top2_count": int(top2_counts[c]),
                "top2_rate": float(top2_rates[c]),
            })

    return pd.DataFrame(rows), pd.DataFrame(pred_rows), pd.DataFrame(top2_rows)


def family_candidate_from_summary(summary: pd.DataFrame, class_names: List[str], malware_classes: List[str],
                                  split: str, eps_cap: float, min_support: int, tag: str) -> pd.DataFrame:
    rows = []
    sub = summary[(summary["split"] == split) & (summary["true_L2"].isin(malware_classes))].copy()
    for _, r in sub.iterrows():
        true_l2 = clean(r["true_L2"])
        true_l3 = clean(r["true_L3"])
        n = int(r["n"])
        row = {
            "split_source": split,
            "usage_tag": tag,
            "true_L2": true_l2,
            "true_L3": true_l3,
            "n": n,
            "reliable_support": bool(n >= min_support),
            "eps_rule": "eps_family = min(eps_cap, mean probability mass assigned to other malware classes)",
            "eps_cap": float(eps_cap),
            "eps_family": 0.0,
        }
        for c in class_names:
            row[f"target_{c}"] = 0.0

        if n < min_support:
            row[f"target_{true_l2}"] = 1.0
            row["eps_family"] = 0.0
            row["source_note"] = "support_below_min_keep_one_hot"
            rows.append(row)
            continue

        # eps from mean probability mass assigned to other malware classes.
        eps = min(float(eps_cap), max(0.0, float(r.get("other_malware_prob_mass_mean", 0.0))))
        other_scores = {}
        for c in malware_classes:
            if c == true_l2:
                continue
            # combine mean prob and top2/pred evidence.
            mean_prob = float(r.get(f"mean_prob_{c}", 0.0))
            top2_rate = float(r.get(f"top2_rate_{c}", 0.0))
            pred_rate = float(r.get(f"pred_rate_{c}", 0.0))
            other_scores[c] = 0.55 * mean_prob + 0.30 * top2_rate + 0.15 * pred_rate

        total_score = sum(max(0.0, v) for v in other_scores.values())
        if eps <= 0 or total_score <= 0:
            row[f"target_{true_l2}"] = 1.0
            row["eps_family"] = 0.0
            row["source_note"] = "no_other_malware_overlap_signal_keep_one_hot"
        else:
            row[f"target_{true_l2}"] = 1.0 - eps
            for c, sc in other_scores.items():
                row[f"target_{c}"] = eps * max(0.0, sc) / total_score
            row["eps_family"] = eps
            row["source_note"] = "family_probability_top2_weighted_candidate"
        if DEFAULT_CLASS_NAMES[0] in class_names:
            row[f"target_{DEFAULT_CLASS_NAMES[0]}"] = 0.0 if true_l2 != DEFAULT_CLASS_NAMES[0] else row.get(f"target_{DEFAULT_CLASS_NAMES[0]}", 1.0)
        row["target_sum"] = sum(row[f"target_{c}"] for c in class_names)
        rows.append(row)
    return pd.DataFrame(rows)


def write_leakage_policy(out_dir: Path):
    txt = """# F1e1a_v3 Leakage Policy

## Main answer

Using validation L3/top2/probability audit to choose smoothing hyperparameters and then reporting the same validation score as final is validation leakage / tuning-to-val.

It is not train-label leakage, and it is not a bug in inference, but it makes the validation result optimistic because the validation set influenced model design.

## Allowed uses of validation audit

Validation audit is acceptable for:
- diagnosing failure mode
- deciding broad research direction
- explaining why subtype boundary is hard
- generating hypotheses

But if a hyperparameter/matrix is chosen from validation audit, the resulting validation score should be described as model-selection validation, not final unbiased performance.

## Clean ways to avoid leakage

Best:
1. Create internal split from original train:
   train_inner + calibration
2. Train L1 on train_inner
3. Generate calibration predictions
4. Derive smoothing matrix from calibration
5. Retrain final model on original train with locked matrix
6. Evaluate once on original validation or separate test

Better:
- K-fold out-of-fold predictions on the training set, derive matrix from OOF.

Acceptable but weaker:
- Use validation audit only to decide direction.
- Choose a very small fixed candidate before seeing final result.
- Report honestly that validation was used in model selection.

Not clean:
- Tune matrix repeatedly on validation and report the best validation score as final.
"""
    (out_dir / "F1e1a_v3_leakage_policy.md").write_text(txt, encoding="utf-8")


def write_report(out_dir: Path, ds_info: Dict[str, Any], model_info: Dict[str, Any],
                 split_metrics: Dict[str, Dict[str, float]], family_summary: pd.DataFrame,
                 val_candidate: pd.DataFrame, train_candidate: pd.DataFrame,
                 class_names: List[str], malware_classes: List[str], min_support: int):
    lines = []
    lines.append("# F1e1a_v3 L1 L3/Family + Top2/Probability Audit Report\n")
    lines.append("## Purpose\n")
    lines.append("```text")
    lines.append("No training.")
    lines.append("Audit L1 behavior at L3/family level with full probability vector and top2 information.")
    lines.append("This is meant to diagnose whether family-aware smoothing is justified.")
    lines.append("```")

    lines.append("\n## Leakage answer\n")
    lines.append("```text")
    lines.append("If validation family/probability audit is used to choose smoothing hyperparameters,")
    lines.append("then evaluating/reporting on that same validation set is validation leakage / tuning-to-val.")
    lines.append("")
    lines.append("Use val audit for diagnosis/hypothesis.")
    lines.append("For clean hyperparameter choice, derive matrix from train OOF or an internal calibration split.")
    lines.append("```")

    lines.append("\n## Loaded data/model\n")
    lines.append("```json")
    lines.append(json.dumps({"dataset": ds_info, "model": model_info}, indent=2, default=str)[:6000])
    lines.append("```")

    lines.append("\n## Split metrics from L1 checkpoint inference\n")
    lines.append(pd.DataFrame([
        {"split": k, **v} for k, v in split_metrics.items()
    ]).to_markdown(index=False))

    lines.append("\n## Family summary: worst validation families by error / ambiguity\n")
    valfam = family_summary[family_summary["split"] == "val"].copy()
    if len(valfam):
        show_cols = [c for c in [
            "true_L2", "true_L3", "n", "audit_reliable_support", "accuracy", "error_rate",
            "true_in_top2_rate", "top2_gap_mean", "true_prob_mean", "other_malware_prob_mass_mean",
            "pred_distribution_entropy_norm", "top2_distribution_entropy_norm",
        ] if c in valfam.columns]
        val_show = valfam.sort_values(["audit_reliable_support", "error_rate", "other_malware_prob_mass_mean"], ascending=[False, False, False]).head(40)
        lines.append(val_show[show_cols].to_markdown(index=False))
    else:
        lines.append("No validation family summary produced.")

    lines.append("\n## Diagnostic validation family-aware smoothing candidate")
    lines.append("```text")
    lines.append("This candidate is DIAGNOSTIC_ONLY.")
    lines.append("Do not use it to claim final unbiased val performance unless a separate final test exists.")
    lines.append("```")
    if len(val_candidate):
        lines.append(val_candidate.head(60).to_markdown(index=False))
    else:
        lines.append("No validation candidate produced.")

    lines.append("\n## Train in-sample family-aware candidate")
    lines.append("```text")
    lines.append("This avoids validation labels, but it is biased because L1 was trained on train.")
    lines.append("For clean choice, use OOF or calibration predictions.")
    lines.append("```")
    if len(train_candidate):
        lines.append(train_candidate.head(60).to_markdown(index=False))
    else:
        lines.append("No train candidate produced.")

    lines.append("\n## Decision")
    lines.append("```text")
    lines.append("If many families have diffuse top2/probability mass across malware classes:")
    lines.append("  family-aware smoothing is more justified than global L2 smoothing.")
    lines.append("")
    lines.append("If only a few families dominate errors:")
    lines.append("  design should focus on those families, not whole L2 classes.")
    lines.append("")
    lines.append("If label_L3 is missing or equals L2:")
    lines.append("  this audit cannot answer family-level behavior; fix raw label source first.")
    lines.append("```")

    (out_dir / "F1e1a_v3_report.md").write_text("\n".join(lines), encoding="utf-8")


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
    ap.add_argument("--out-dir", default="05_test/outputs/F1e1a_v3_l1_family_top2_prob_audit")
    ap.add_argument("--combined-zip", default="05_test/outputs/F1e1a_v3_l1_family_top2_prob_audit.zip")
    ap.add_argument("--class-names", default="Benign,Ransomware,Spyware,Trojan")
    ap.add_argument("--malware-classes", default="Ransomware,Spyware,Trojan")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--eps-cap", type=float, default=0.20)
    ap.add_argument("--min-family-support", type=int, default=30)
    ap.add_argument("--trust-local-checkpoint", action="store_true")
    args = ap.parse_args()

    root = repo_root_from_here()
    l1_dir = resolve_path(args.l1_run_dir, root)
    out_dir = resolve_path(args.out_dir, root)
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_zip = resolve_path(args.combined_zip, root)

    if not l1_dir.exists():
        raise FileNotFoundError(f"L1 run dir not found: {l1_dir}")

    class_names = [clean(x) for x in args.class_names.split(",") if clean(x)]
    if not class_names:
        class_names = detect_class_names_from_report(l1_dir)
    malware_classes = [clean(x) for x in args.malware_classes.split(",") if clean(x)]

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    log(f"root={root}")
    log(f"l1_dir={l1_dir}")
    log(f"out_dir={out_dir}")
    log(f"classes={class_names}; malware_classes={malware_classes}")
    log("No training. L1 family/top2/probability audit only.")

    ds, ds_info = load_dataset(
        resolve_path(args.dataset_npz, root),
        resolve_path(args.train_raw, root),
        resolve_path(args.val_raw, root),
        class_names,
    )
    (out_dir / "F1e1a_v3_dataset_info.json").write_text(json.dumps(ds_info, indent=2, default=str), encoding="utf-8")

    model, model_info = build_model(root, l1_dir, ds_info, class_names, device, args.trust_local_checkpoint)
    (out_dir / "F1e1a_v3_model_load_info.json").write_text(json.dumps(model_info, indent=2, default=str), encoding="utf-8")

    split_metrics = {}
    all_family_summary = []
    all_pred_dist = []
    all_top2_dist = []
    pred_dfs = {}

    for split in ["train", "val"]:
        log(f"inference split={split}")
        pred = infer_split(model, ds[split], class_names, device, args.batch_size)
        pred["split"] = split
        pred_dfs[split] = pred
        pred.to_csv(out_dir / f"F1e1a_v3_{split}_predictions_with_probs_top2_l3.csv", index=False)

        y = pred["y_true"].to_numpy()
        yp = pred["y_pred"].to_numpy()
        split_metrics[split] = {
            "n": int(len(pred)),
            "accuracy": float(accuracy_score(y, yp)),
            "macro_f1": float(f1_score(y, yp, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y, yp, average="weighted", zero_division=0)),
            "mean_confidence": float(pred["top1_prob"].mean()),
            "mean_top2_gap": float(pred["top2_gap"].mean()),
            "true_in_top2_rate": float(pred["true_in_top2"].mean()),
        }

        rep = classification_report(y, yp, labels=list(range(len(class_names))), target_names=class_names, output_dict=True, zero_division=0)
        (out_dir / f"F1e1a_v3_{split}_classification_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
        cm = confusion_matrix(y, yp, labels=list(range(len(class_names))))
        pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
            out_dir / f"F1e1a_v3_{split}_confusion_matrix.csv"
        )

        fam, pdist, tdist = summarize_family(pred, class_names, malware_classes, split, args.min_family_support)
        all_family_summary.append(fam)
        all_pred_dist.append(pdist)
        all_top2_dist.append(tdist)

    family_summary = pd.concat(all_family_summary, ignore_index=True) if all_family_summary else pd.DataFrame()
    pred_dist = pd.concat(all_pred_dist, ignore_index=True) if all_pred_dist else pd.DataFrame()
    top2_dist = pd.concat(all_top2_dist, ignore_index=True) if all_top2_dist else pd.DataFrame()

    family_summary.to_csv(out_dir / "F1e1a_v3_family_summary_by_split.csv", index=False)
    pred_dist.to_csv(out_dir / "F1e1a_v3_family_pred_distribution.csv", index=False)
    top2_dist.to_csv(out_dir / "F1e1a_v3_family_top2_distribution.csv", index=False)

    val_candidate = family_candidate_from_summary(
        family_summary, class_names, malware_classes, "val",
        eps_cap=args.eps_cap,
        min_support=args.min_family_support,
        tag="DIAGNOSTIC_ONLY_validation_used_not_clean_for_final_val_claim",
    )
    train_candidate = family_candidate_from_summary(
        family_summary, class_names, malware_classes, "train",
        eps_cap=args.eps_cap,
        min_support=args.min_family_support,
        tag="TRAIN_IN_SAMPLE_BIASED_not_validation_leak_but_overfit_biased",
    )
    val_candidate.to_csv(out_dir / "F1e1a_v3_val_family_smoothing_candidate_DIAGNOSTIC_ONLY.csv", index=False)
    train_candidate.to_csv(out_dir / "F1e1a_v3_train_in_sample_family_smoothing_candidate_BIASED.csv", index=False)

    targets = {
        "leakage_warning": "Validation-derived family candidate is diagnostic only; using it to select hyperparameters and report same val is validation leakage/tuning-to-val.",
        "clean_recommendation": "Derive smoothing matrix from train out-of-fold or internal calibration split, lock it, then evaluate on validation/test.",
        "class_names": class_names,
        "malware_classes": malware_classes,
        "eps_cap": float(args.eps_cap),
        "min_family_support": int(args.min_family_support),
        "split_metrics": split_metrics,
        "val_candidate_file": "F1e1a_v3_val_family_smoothing_candidate_DIAGNOSTIC_ONLY.csv",
        "train_in_sample_candidate_file": "F1e1a_v3_train_in_sample_family_smoothing_candidate_BIASED.csv",
    }
    (out_dir / "F1e1a_v3_targets_and_policy.json").write_text(json.dumps(targets, indent=2), encoding="utf-8")

    write_leakage_policy(out_dir)
    write_report(out_dir, ds_info, model_info, split_metrics, family_summary, val_candidate, train_candidate,
                 class_names, malware_classes, args.min_family_support)

    zip_dir(out_dir, combined_zip)

    log("Split metrics:")
    print(pd.DataFrame([{"split": k, **v} for k, v in split_metrics.items()]).to_string(index=False), flush=True)
    log(f"zip={combined_zip}")
    log("DONE")


if __name__ == "__main__":
    main()
