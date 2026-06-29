#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F2b Export logits/probs/CLS and deep local-overlap audit for official L1.

No training. Diagnostic only.

Questions answered:
  1. Does official L1 checkpoint reproduce expected validation macro-F1?
  2. For malware pair errors, does CLS-space favor true or predicted class?
  3. Does raw/token neighborhood disagree with CLS neighborhood?
  4. For hard groups like Trojan-Zeus -> Ransomware, is failure mostly:
       - inherent raw/token local overlap,
       - representation/CLS distortion,
       - classifier/logit boundary over CLS?
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


CLASS_NAMES_DEFAULT = ["Benign", "Ransomware", "Spyware", "Trojan"]
MALWARE_CLASSES_DEFAULT = ["Ransomware", "Spyware", "Trojan"]
LABEL_COL_CANDIDATES = [
    "label_L1", "label_L2", "label_L3", "Label_L1", "Label_L2", "Label_L3",
    "Class", "Category", "Family", "class", "category", "family",
    "MalwareFamily", "malware_family", "label", "target",
]


def log(msg: str):
    print(f"[F2b] {msg}", flush=True)


def repo_root_from_here() -> Path:
    p = Path(__file__).resolve()
    if p.parent.name == "05_test":
        return p.parents[1]
    return Path.cwd().resolve()


def resolve_path(p: str | Path, root: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (root / p).resolve()


def parse_list(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def clean_label(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def find_label_col(df: pd.DataFrame, level: str) -> Optional[str]:
    if level == "L2":
        cands = ["label_L2", "Label_L2", "l2", "L2", "Category", "category", "Class", "class"]
    elif level == "L3":
        cands = ["label_L3", "Label_L3", "l3", "L3", "Family", "family", "MalwareFamily", "malware_family"]
    else:
        cands = LABEL_COL_CANDIDATES
    for c in cands:
        if c in df.columns:
            return c
    return None


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - np.nanmax(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)


def import_module_from_path(path: Path, name: str = "official_trainer_module"):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def auto_find_model_dir(search_root: Path, preferred_keywords: List[str]) -> Optional[Path]:
    # First reuse previous F2 audit config if available.
    cfg_path = search_root / "F2_overlap_local_pair_family_audit" / "config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            d = Path(cfg.get("model_dir", ""))
            if d.exists() and (d / "best_model.pt").exists():
                return d
        except Exception:
            pass

    candidates = []
    for p in search_root.rglob("best_model.pt"):
        d = p.parent
        score = 0
        name = str(d).lower()
        for kw in preferred_keywords:
            if kw.lower() in name:
                score += 10
        if "l1" in name:
            score += 6
        if "f1a2" in name or "f1e2a" in name or "base" in name or "reproduce" in name:
            score += 5
        if (d / "val_predictions_best.csv").exists():
            score += 2
        if (d / "config.json").exists():
            score += 2
        for bad in ["sam", "rho", "smoothing", "soft", "lambda", "subce", "overlap", "f1f1"]:
            if bad in name:
                score -= 5
        candidates.append((score, d))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def find_trainer_module(root: Path, explicit: str) -> Path:
    if explicit.strip():
        p = resolve_path(explicit, root)
        if p.exists():
            return p
        raise FileNotFoundError(f"trainer module not found: {p}")
    candidates = [
        root / "02_src" / "07_train_sam.py",
        root / "02_src" / "07_train_overlap.py",
        root / "02_src" / "07_train.py",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Cannot find trainer module. Expected one of 02_src/07_train_sam.py, 07_train_overlap.py, 07_train.py")


def load_checkpoint(model_dir: Path, device: torch.device) -> Dict[str, Any]:
    ckpt_path = model_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing best_model.pt: {ckpt_path}")
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unexpected checkpoint type: {type(ckpt)}")
    if "model_state_dict" not in ckpt:
        # Some saves are raw state_dict.
        ckpt = {"model_state_dict": ckpt, "config": {}}
    return ckpt


def make_runtime_args(
    *,
    config: Dict[str, Any],
    dataset_npz: Path,
    metadata_json: Path,
    train_raw: Path,
    val_raw: Path,
    device: str,
    batch_size: int,
    num_workers: int,
) -> SimpleNamespace:
    # Supply only fields used by trainer helper functions / dataset path.
    model_cfg = config.get("model", {})
    return SimpleNamespace(
        run_id=str(config.get("run_id", "D3")),
        K=int(config.get("K_artifact", config.get("effective_token_budget", 512))),
        num_bins=int(config.get("num_bins", 512)),
        dataset_npz=str(dataset_npz),
        metadata_json=str(metadata_json),
        train_raw=str(train_raw),
        val_raw=str(val_raw),
        device=str(device),
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        # model defaults
        value_dim=int(model_cfg.get("value_dim", 32)),
        feature_dim=int(model_cfg.get("feature_dim", 32)),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        num_layers=int(model_cfg.get("num_layers", 1)),
        num_heads=int(model_cfg.get("num_heads", 4)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        classifier_hidden_dim=int(model_cfg.get("classifier_hidden_dim", 128)),
        classifier_dropout=float(model_cfg.get("classifier_dropout", 0.1)),
        norm_first=bool(model_cfg.get("norm_first", True)),
        gate_init=float(model_cfg.get("gate_init", 0.0)),
        tail_frac=0.02,
        wide_quantile=0.90,
    )


def build_arrays_and_model(
    *,
    trainer,
    ckpt: Dict[str, Any],
    dataset_npz: Path,
    metadata_json: Path,
    train_raw: Path,
    val_raw: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
):
    config = ckpt.get("config") or {}
    if not config:
        cfg_path = Path(ckpt.get("config_path", ""))
        if cfg_path.exists():
            config = json.loads(cfg_path.read_text(encoding="utf-8"))

    args = make_runtime_args(
        config=config,
        dataset_npz=dataset_npz,
        metadata_json=metadata_json,
        train_raw=train_raw,
        val_raw=val_raw,
        device=str(device),
        batch_size=batch_size,
        num_workers=num_workers,
    )

    data, meta = trainer.load_dataset(dataset_npz, metadata_json)

    X_train = data["X_train_bin"].astype(np.int64)
    O_train = data["X_train_offset"].astype(np.float32)
    y_train = data["y_train"].astype(np.int64)
    X_val = data["X_val_bin"].astype(np.int64)
    O_val = data["X_val_offset"].astype(np.float32)
    y_val = data["y_val"].astype(np.int64)

    spec = trainer.RUN_SPECS[str(args.run_id)]
    X_train_cont, X_val_cont, continuous_info = trainer.load_continuous_for_run(
        spec=spec,
        meta=meta,
        args=args,
        train_shape=X_train.shape,
        val_shape=X_val.shape,
    )

    if str(args.run_id) == "D5":
        M_train, M_val, selective_info = trainer.build_selective_mask(
            X_train_bin=X_train,
            X_val_bin=X_val,
            X_train_cont=X_train_cont,
            num_bins=int(args.num_bins),
            tail_frac=float(args.tail_frac),
            wide_quantile=float(args.wide_quantile),
        )
    else:
        M_train = np.ones_like(X_train, dtype=np.float32)
        M_val = np.ones_like(X_val, dtype=np.float32)
        selective_info = {"type": "none"}

    label_mapping = meta.get("label_mapping", {"Benign": 0, "Ransomware": 1, "Spyware": 2, "Trojan": 3})
    label_names = [label for label, _ in sorted(label_mapping.items(), key=lambda kv: kv[1])]
    num_classes = int(len(label_names))
    n_features = int(meta.get("n_features", X_train.shape[1]))

    model = trainer.FusionAblationTransformer(
        run_id=str(args.run_id),
        num_bins=int(args.num_bins),
        n_features=n_features,
        num_classes=num_classes,
        value_dim=int(args.value_dim),
        feature_dim=int(args.feature_dim),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        num_heads=int(args.num_heads),
        dropout=float(args.dropout),
        classifier_hidden_dim=int(args.classifier_hidden_dim),
        classifier_dropout=float(args.classifier_dropout),
        norm_first=bool(args.norm_first),
        gate_init=float(args.gate_init),
    ).to(device)

    state = ckpt["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()

    train_ds = trainer.FusionAblationDataset(X_train, O_train, X_train_cont, M_train, y_train)
    val_ds = trainer.FusionAblationDataset(X_val, O_val, X_val_cont, M_val, y_val)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    return {
        "args": args,
        "meta": meta,
        "config": config,
        "label_names": label_names,
        "num_classes": num_classes,
        "n_features": n_features,
        "model": model,
        "load_state_missing_keys": list(missing),
        "load_state_unexpected_keys": list(unexpected),
        "train_loader": train_loader,
        "val_loader": val_loader,
        "arrays": {
            "X_train_bin": X_train,
            "O_train": O_train,
            "X_train_cont": X_train_cont,
            "M_train": M_train,
            "y_train": y_train,
            "X_val_bin": X_val,
            "O_val": O_val,
            "X_val_cont": X_val_cont,
            "M_val": M_val,
            "y_val": y_val,
        },
        "continuous_info": continuous_info,
        "selective_info": selective_info,
    }


@torch.no_grad()
def extract_split(model, loader, device: torch.device, split_name: str) -> Dict[str, np.ndarray]:
    logits_list = []
    y_list = []
    cls_list = []

    captured = []

    def pre_hook(module, inputs):
        # classifier input is cls_out [B,H]
        if inputs and torch.is_tensor(inputs[0]):
            captured.append(inputs[0].detach().cpu())

    handle = model.classifier.register_forward_pre_hook(pre_hook)
    try:
        for batch in loader:
            if len(batch) == 4:
                tokens, values, y, _soft = batch
            else:
                tokens, values, y = batch
            tokens = tokens.to(device, non_blocking=True)
            values = values.to(device, non_blocking=True)
            logits = model(tokens, values)
            logits_list.append(logits.detach().cpu())
            y_list.append(y.detach().cpu())
    finally:
        handle.remove()

    if not captured:
        raise RuntimeError(f"Failed to capture CLS outputs for {split_name}. classifier hook did not fire.")

    logits = torch.cat(logits_list, dim=0).numpy()
    y = torch.cat(y_list, dim=0).numpy().astype(np.int64)
    cls = torch.cat(captured, dim=0).numpy().astype(np.float32)
    probs = softmax_np(logits).astype(np.float32)
    pred = np.argmax(probs, axis=1).astype(np.int64)
    top_order = np.argsort(-probs, axis=1).astype(np.int64)

    return {
        "logits": logits.astype(np.float32),
        "probs": probs,
        "y": y,
        "pred": pred,
        "cls": cls,
        "top_order": top_order,
    }


def add_labels(df: pd.DataFrame, raw: pd.DataFrame, label_names: List[str]) -> pd.DataFrame:
    out = df.copy()
    id_to_name = {i: c for i, c in enumerate(label_names)}
    out["true_label"] = out["y"].map(id_to_name)
    out["pred_label"] = out["pred"].map(id_to_name)
    l2 = find_label_col(raw, "L2")
    l3 = find_label_col(raw, "L3")
    if l2:
        out["raw_L2"] = raw[l2].iloc[:len(out)].map(clean_label).to_numpy()
    else:
        out["raw_L2"] = out["true_label"]
    if l3:
        out["family"] = raw[l3].iloc[:len(out)].map(clean_label).to_numpy()
    else:
        out["family"] = out["true_label"]
    out["family"] = out["family"].where(out["family"].astype(str).str.len() > 0, out["true_label"])
    return out


def prediction_frame(split: Dict[str, np.ndarray], raw: pd.DataFrame, label_names: List[str]) -> pd.DataFrame:
    probs = split["probs"]
    logits = split["logits"]
    y = split["y"]
    pred = split["pred"]
    top_order = split["top_order"]
    n = len(y)
    rows = {
        "row_id": np.arange(n, dtype=np.int64),
        "y": y,
        "pred": pred,
        "correct": y == pred,
        "top1_id": top_order[:, 0],
        "top2_id": top_order[:, 1],
        "top1_prob": probs[np.arange(n), top_order[:, 0]],
        "top2_prob": probs[np.arange(n), top_order[:, 1]],
        "margin_top1_top2": probs[np.arange(n), top_order[:, 0]] - probs[np.arange(n), top_order[:, 1]],
        "true_prob": probs[np.arange(n), y],
        "pred_prob": probs[np.arange(n), pred],
    }
    for i, c in enumerate(label_names):
        rows[f"prob_{c}"] = probs[:, i]
        rows[f"logit_{c}"] = logits[:, i]
    df = pd.DataFrame(rows)
    df["true_rank"] = [int(np.where(top_order[i] == y[i])[0][0]) + 1 for i in range(n)]
    df["true_in_top2"] = df["true_rank"] <= 2
    for i, c in enumerate(label_names):
        df[f"logit_margin_true_minus_{c}"] = logits[np.arange(n), y] - logits[:, i]
    df = add_labels(df, raw, label_names)
    return df


def metrics_and_confusion(val_df: pd.DataFrame, label_names: List[str], out_dir: Path) -> Dict[str, Any]:
    report = classification_report(
        val_df["true_label"], val_df["pred_label"],
        labels=label_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(val_df["true_label"], val_df["pred_label"], labels=label_names)
    pd.DataFrame(report).T.to_csv(out_dir / "00_val_classification_report_recomputed.csv")
    pd.DataFrame(cm, index=[f"true_{c}" for c in label_names], columns=[f"pred_{c}" for c in label_names]).to_csv(
        out_dir / "00_val_confusion_matrix_recomputed.csv"
    )
    return {
        "accuracy": float(accuracy_score(val_df["true_label"], val_df["pred_label"])),
        "macro_f1": float(f1_score(val_df["true_label"], val_df["pred_label"], labels=label_names, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(val_df["true_label"], val_df["pred_label"], labels=label_names, average="weighted", zero_division=0)),
        "support": int(len(val_df)),
    }


def standardize_fit_transform(train_X: np.ndarray, val_X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    sc = StandardScaler()
    tr = sc.fit_transform(train_X)
    va = sc.transform(val_X)
    return tr.astype(np.float32), va.astype(np.float32)


def get_numeric_features(train_raw: pd.DataFrame, val_raw: pd.DataFrame, feature_names: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    drop_cols = set(LABEL_COL_CANDIDATES)
    if feature_names:
        cols = [c for c in feature_names if c in train_raw.columns and c in val_raw.columns]
    else:
        cols = [c for c in train_raw.columns if c not in drop_cols and c in val_raw.columns]
    feat_cols = []
    train_cols = []
    val_cols = []
    for c in cols:
        tr = pd.to_numeric(train_raw[c], errors="coerce")
        va = pd.to_numeric(val_raw[c], errors="coerce")
        if tr.notna().sum() == 0:
            continue
        med = float(tr.median())
        feat_cols.append(c)
        train_cols.append(tr.fillna(med).to_numpy(dtype=np.float32))
        val_cols.append(va.fillna(med).to_numpy(dtype=np.float32))
    if not feat_cols:
        raise ValueError("No numeric features found")
    return np.stack(train_cols, axis=1), np.stack(val_cols, axis=1), feat_cols


def knn_fractions(
    *,
    train_X: np.ndarray,
    val_X: np.ndarray,
    train_labels: np.ndarray,
    val_df: pd.DataFrame,
    malware_classes: List[str],
    k: int,
    space: str,
) -> pd.DataFrame:
    tr_mask = np.isin(train_labels, malware_classes)
    Xtr = train_X[tr_mask]
    ytr = train_labels[tr_mask]
    Xva = val_X
    nn = NearestNeighbors(n_neighbors=min(k, len(Xtr)), metric="euclidean")
    nn.fit(Xtr)
    q = val_df[val_df["true_label"].isin(malware_classes)].copy()
    dists, inds = nn.kneighbors(Xva[q["row_id"].to_numpy()])
    rows = []
    for i, (_, r) in enumerate(q.iterrows()):
        labs = ytr[inds[i]]
        counts = pd.Series(labs).value_counts().to_dict()
        total = len(labs)
        true_lab = r["true_label"]
        pred_lab = r["pred_label"]
        probs = {c: counts.get(c, 0) / total for c in malware_classes}
        entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs.values())
        row = {
            "row_id": int(r["row_id"]),
            "space": space,
            "true_label": true_lab,
            "pred_label": pred_lab,
            "family": r["family"],
            "correct": bool(r["correct"]),
            "true_nn_frac": float(probs.get(true_lab, 0.0)),
            "pred_nn_frac": float(probs.get(pred_lab, 0.0)),
            "neighbor_entropy": float(entropy),
            "nearest_distance": float(dists[i][0]),
            "mean_distance": float(dists[i].mean()),
        }
        for c in malware_classes:
            row[f"nn_frac_{c}"] = float(probs.get(c, 0.0))
        rows.append(row)
    return pd.DataFrame(rows)


def centroid_distance_rows(
    train_X: np.ndarray,
    val_X: np.ndarray,
    train_labels: np.ndarray,
    val_df: pd.DataFrame,
    class_names: List[str],
    malware_classes: List[str],
    space: str,
) -> pd.DataFrame:
    centroids = {}
    for c in class_names:
        m = train_labels == c
        if m.sum() > 0:
            centroids[c] = train_X[m].mean(axis=0)
    rows = []
    q = val_df[val_df["true_label"].isin(malware_classes)].copy()
    for _, r in q.iterrows():
        x = val_X[int(r["row_id"])]
        true_lab = r["true_label"]
        pred_lab = r["pred_label"]
        d_true = float(np.linalg.norm(x - centroids[true_lab])) if true_lab in centroids else np.nan
        d_pred = float(np.linalg.norm(x - centroids[pred_lab])) if pred_lab in centroids else np.nan
        d_all = {c: float(np.linalg.norm(x - centroids[c])) for c in centroids}
        nearest = min(d_all, key=d_all.get)
        rows.append({
            "row_id": int(r["row_id"]),
            "space": space,
            "true_label": true_lab,
            "pred_label": pred_lab,
            "family": r["family"],
            "correct": bool(r["correct"]),
            "dist_true_centroid": d_true,
            "dist_pred_centroid": d_pred,
            "centroid_margin_pred_minus_true": d_pred - d_true if not (np.isnan(d_pred) or np.isnan(d_true)) else np.nan,
            "nearest_centroid_label": nearest,
            "nearest_centroid_is_true": nearest == true_lab,
            "nearest_centroid_is_pred": nearest == pred_lab,
        })
    return pd.DataFrame(rows)


def merge_space_signals(val_df: pd.DataFrame, dfs: List[pd.DataFrame]) -> pd.DataFrame:
    base_cols = [
        "row_id", "true_label", "pred_label", "family", "correct",
        "top1_prob", "top2_prob", "margin_top1_top2", "true_prob", "pred_prob",
        "true_rank", "true_in_top2",
    ]
    keep = val_df[base_cols].copy()
    for c in val_df.columns:
        if c.startswith("prob_") or c.startswith("logit_"):
            keep[c] = val_df[c]
    out = keep
    for d in dfs:
        if d.empty:
            continue
        space = str(d["space"].iloc[0])
        rename = {}
        for c in d.columns:
            if c in {"row_id", "space", "true_label", "pred_label", "family", "correct"}:
                continue
            rename[c] = f"{space}_{c}"
        out = out.merge(d[["row_id"] + list(rename.keys())].rename(columns=rename), on="row_id", how="left")
    return out


def classify_failure_modes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Heuristic tags using available raw/token/cls kNN and cls centroid.
    def one(r):
        tags = []
        if not bool(r.get("correct", False)):
            raw_true = r.get("raw_true_nn_frac", np.nan)
            raw_pred = r.get("raw_pred_nn_frac", np.nan)
            tok_true = r.get("token_true_nn_frac", np.nan)
            tok_pred = r.get("token_pred_nn_frac", np.nan)
            cls_true = r.get("cls_true_nn_frac", np.nan)
            cls_pred = r.get("cls_pred_nn_frac", np.nan)
            cls_centroid_margin = r.get("cls_centroid_margin_pred_minus_true", np.nan)

            raw_favors_true = (not np.isnan(raw_true)) and (raw_true > raw_pred + 0.05)
            raw_favors_pred = (not np.isnan(raw_true)) and (raw_pred > raw_true + 0.05)
            tok_favors_true = (not np.isnan(tok_true)) and (tok_true > tok_pred + 0.05)
            tok_favors_pred = (not np.isnan(tok_true)) and (tok_pred > tok_true + 0.05)
            cls_favors_true = (not np.isnan(cls_true)) and (cls_true > cls_pred + 0.05)
            cls_favors_pred = (not np.isnan(cls_true)) and (cls_pred > cls_true + 0.05)
            centroid_true = (not np.isnan(cls_centroid_margin)) and (cls_centroid_margin > 0)
            centroid_pred = (not np.isnan(cls_centroid_margin)) and (cls_centroid_margin < 0)

            if (raw_favors_pred or tok_favors_pred) and cls_favors_pred:
                tags.append("local_overlap_transfers_to_cls")
            if (raw_favors_true or tok_favors_true) and cls_favors_pred:
                tags.append("representation_distortion_raw_true_cls_pred")
            if cls_favors_true and centroid_true:
                tags.append("cls_still_true_like_but_logit_wrong")
            if centroid_true and not cls_favors_pred:
                tags.append("classifier_boundary_suspect")
            if centroid_pred or cls_favors_pred:
                tags.append("representation_boundary_suspect")
            if not tags:
                tags.append("mixed_ambiguous_overlap")
        else:
            tags.append("correct")
        return "|".join(tags)
    out["failure_mode_tags"] = out.apply(one, axis=1)
    return out


def group_summaries(deep: pd.DataFrame, malware_classes: List[str], out_dir: Path):
    wrong = deep[(deep["correct"] == False) & (deep["true_label"].isin(malware_classes))].copy()

    # true/pred/family groups.
    agg_cols = {
        "row_id": "count",
        "true_prob": "mean",
        "pred_prob": "mean",
        "margin_top1_top2": "mean",
        "true_in_top2": "mean",
    }
    for prefix in ["raw", "token", "cls"]:
        for col in [f"{prefix}_true_nn_frac", f"{prefix}_pred_nn_frac", f"{prefix}_neighbor_entropy", f"{prefix}_centroid_margin_pred_minus_true"]:
            if col in wrong.columns:
                agg_cols[col] = "mean"

    group = wrong.groupby(["true_label", "pred_label", "family"], dropna=False).agg(agg_cols).reset_index()
    group = group.rename(columns={"row_id": "count"})
    group = group.sort_values("count", ascending=False)
    group.to_csv(out_dir / "03_wrong_pair_family_deep_summary.csv", index=False)

    pair = wrong.groupby(["true_label", "pred_label"], dropna=False).agg(agg_cols).reset_index().rename(columns={"row_id": "count"})
    pair = pair.sort_values("count", ascending=False)
    pair.to_csv(out_dir / "04_wrong_pair_deep_summary.csv", index=False)

    # failure mode counts.
    mode_rows = []
    for tagstr, g in wrong.groupby("failure_mode_tags", dropna=False):
        for tag in str(tagstr).split("|"):
            mode_rows.append({"failure_mode_tag": tag, "count": int(len(g))})
    mode_df = pd.DataFrame(mode_rows).groupby("failure_mode_tag").agg(count=("count", "sum")).reset_index().sort_values("count", ascending=False)
    mode_df.to_csv(out_dir / "05_failure_mode_tag_counts.csv", index=False)

    # Focus groups.
    focus = []
    if "family" in wrong.columns:
        focus.append(wrong[(wrong["true_label"] == "Trojan") & (wrong["family"].astype(str).str.contains("Zeus", case=False, na=False))])
        focus.append(wrong[(wrong["true_label"] == "Trojan") & (wrong["pred_label"] == "Ransomware")])
        focus.append(wrong[(wrong["true_label"] == "Trojan") & (wrong["pred_label"] == "Spyware")])
        focus.append(wrong[(wrong["true_label"] == "Spyware") & (wrong["family"].astype(str).str.contains("180", case=False, na=False))])
    focus_df = pd.concat([x for x in focus if len(x)], axis=0).drop_duplicates("row_id") if focus else pd.DataFrame()
    focus_df.to_csv(out_dir / "06_focus_hard_groups_rows.csv", index=False)

    return group, pair, mode_df, focus_df


def safe_md(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or len(df) == 0:
        return "_empty_"
    try:
        return df.head(max_rows).to_markdown(index=False)
    except Exception:
        return df.head(max_rows).to_string(index=False)


def write_report(
    *,
    out_dir: Path,
    model_dir: Path,
    trainer_path: Path,
    metrics: Dict[str, Any],
    reproduction: Dict[str, Any],
    group: pd.DataFrame,
    pair: pd.DataFrame,
    mode_df: pd.DataFrame,
    config: Dict[str, Any],
):
    lines = []
    lines.append("# F2b logits/CLS deep overlap audit\n")
    lines.append("## Scope\n")
    lines.append("```text")
    lines.append("No training. Diagnostic only.")
    lines.append("Exports logits/probabilities/CLS from official L1 checkpoint.")
    lines.append("Compares raw/token/CLS local neighborhoods and CLS centroids for malware subtype errors.")
    lines.append("```")
    lines.append("\n## Model and trainer\n")
    lines.append(f"- model_dir: `{model_dir}`")
    lines.append(f"- trainer_module: `{trainer_path}`")
    lines.append("\n## Reproduction check\n")
    lines.append("```json")
    lines.append(json.dumps(reproduction, indent=2, default=str))
    lines.append("```")
    lines.append("\n## Recomputed validation metrics\n")
    lines.append("```json")
    lines.append(json.dumps(metrics, indent=2, default=str))
    lines.append("```")
    lines.append("\n## Wrong pair/family deep summary\n")
    lines.append(safe_md(group, 20))
    lines.append("\n## Wrong pair summary\n")
    lines.append(safe_md(pair, 12))
    lines.append("\n## Failure-mode tag counts\n")
    lines.append(safe_md(mode_df, 20))
    lines.append("\n## Interpretation guide\n")
    lines.append("```text")
    lines.append("raw/token true_nn_frac > pred_nn_frac but cls pred_nn_frac > true_nn_frac:")
    lines.append("  representation likely distorts raw/token local signal.")
    lines.append("")
    lines.append("raw/token pred_nn_frac > true_nn_frac and cls pred_nn_frac > true_nn_frac:")
    lines.append("  overlap likely exists already in input/token space and transfers to representation.")
    lines.append("")
    lines.append("CLS centroid closer to true but logits predict pred:")
    lines.append("  classifier/head boundary is suspect.")
    lines.append("")
    lines.append("CLS kNN and centroid both favor pred:")
    lines.append("  representation/boundary has already moved sample to pred region.")
    lines.append("```")
    lines.append("\n## Config\n")
    lines.append("```json")
    lines.append(json.dumps(config, indent=2, default=str))
    lines.append("```")
    (out_dir / "F2b_logits_cls_deep_overlap_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="", help="Official L1 model output dir containing best_model.pt. If omitted, auto-find.")
    ap.add_argument("--search-root", default="05_test/outputs")
    ap.add_argument("--preferred-keywords", default="F1a2,L1,base,reproduce")
    ap.add_argument("--trainer-module", default="", help="Path to official trainer module with FusionAblationTransformer.")
    ap.add_argument("--dataset-npz", default="03_outputs/05_dataset/dataset.npz")
    ap.add_argument("--metadata-json", default="03_outputs/05_dataset/metadata.json")
    ap.add_argument("--train-raw", default="01_split/train_raw.csv")
    ap.add_argument("--val-raw", default="01_split/val_raw.csv")
    ap.add_argument("--out-dir", default="05_test/outputs/F2b_logits_cls_deep_overlap_audit")
    ap.add_argument("--combined-zip", default="05_test/outputs/F2b_logits_cls_deep_overlap_audit.zip")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--class-names", default="Benign,Ransomware,Spyware,Trojan")
    ap.add_argument("--malware-classes", default="Ransomware,Spyware,Trojan")
    ap.add_argument("--knn-k", type=int, default=31)
    args = ap.parse_args()

    root = repo_root_from_here()
    search_root = resolve_path(args.search_root, root)
    out_dir = resolve_path(args.out_dir, root)
    zip_path = resolve_path(args.combined_zip, root)
    dataset_npz = resolve_path(args.dataset_npz, root)
    metadata_json = resolve_path(args.metadata_json, root)
    train_raw_path = resolve_path(args.train_raw, root)
    val_raw_path = resolve_path(args.val_raw, root)
    trainer_path = find_trainer_module(root, args.trainer_module)

    if args.model_dir.strip():
        model_dir = resolve_path(args.model_dir, root)
    else:
        model_dir = auto_find_model_dir(search_root, parse_list(args.preferred_keywords))
        if model_dir is None:
            raise FileNotFoundError("Could not auto-find official L1 model dir. Pass --model-dir.")

    if not (model_dir / "best_model.pt").exists():
        raise FileNotFoundError(f"model_dir missing best_model.pt: {model_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    class_names = parse_list(args.class_names) or CLASS_NAMES_DEFAULT
    malware_classes = parse_list(args.malware_classes) or MALWARE_CLASSES_DEFAULT

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    log(f"model_dir={model_dir}")
    log(f"trainer_module={trainer_path}")
    log(f"device={device}")

    trainer = import_module_from_path(trainer_path)
    ckpt = load_checkpoint(model_dir, device)

    built = build_arrays_and_model(
        trainer=trainer,
        ckpt=ckpt,
        dataset_npz=dataset_npz,
        metadata_json=metadata_json,
        train_raw=train_raw_path,
        val_raw=val_raw_path,
        device=device,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
    )
    model = built["model"]
    label_names = built["label_names"]
    arrays = built["arrays"]

    train_raw = pd.read_csv(train_raw_path)
    val_raw = pd.read_csv(val_raw_path)

    log("Extracting train logits/probs/CLS")
    train_split = extract_split(model, built["train_loader"], device, "train")
    log("Extracting val logits/probs/CLS")
    val_split = extract_split(model, built["val_loader"], device, "val")

    train_df = prediction_frame(train_split, train_raw, label_names)
    val_df = prediction_frame(val_split, val_raw, label_names)

    train_df.to_csv(out_dir / "01_train_predictions_logits_probs.csv", index=False)
    val_df.to_csv(out_dir / "01_val_predictions_logits_probs.csv", index=False)

    np.savez_compressed(
        out_dir / "02_logits_probs_cls_arrays.npz",
        train_logits=train_split["logits"],
        train_probs=train_split["probs"],
        train_cls=train_split["cls"],
        train_y=train_split["y"],
        train_pred=train_split["pred"],
        val_logits=val_split["logits"],
        val_probs=val_split["probs"],
        val_cls=val_split["cls"],
        val_y=val_split["y"],
        val_pred=val_split["pred"],
    )

    metrics = metrics_and_confusion(val_df, label_names, out_dir)

    expected = None
    if "best_val_macro_f1" in ckpt:
        try:
            expected = float(ckpt["best_val_macro_f1"])
        except Exception:
            expected = None
    reproduction = {
        "checkpoint_epoch": ckpt.get("epoch", None),
        "checkpoint_best_val_macro_f1": expected,
        "recomputed_val_macro_f1": metrics["macro_f1"],
        "abs_diff": abs(metrics["macro_f1"] - expected) if expected is not None else None,
        "load_state_missing_keys": built["load_state_missing_keys"],
        "load_state_unexpected_keys": built["load_state_unexpected_keys"],
    }
    (out_dir / "00_reproduction_check.json").write_text(json.dumps(reproduction, indent=2, default=str), encoding="utf-8")

    # Labels for train.
    l2_train = find_label_col(train_raw, "L2")
    if l2_train is None:
        train_labels = train_df["true_label"].to_numpy()
    else:
        train_labels = train_raw[l2_train].map(clean_label).to_numpy()

    # Build spaces.
    feature_names = [str(x) for x in built["meta"].get("feature_names", [])]
    Xraw_tr, Xraw_va, raw_features = get_numeric_features(train_raw, val_raw, feature_names=feature_names)
    Xraw_tr_z, Xraw_va_z = standardize_fit_transform(Xraw_tr, Xraw_va)

    Xtok_tr = arrays["X_train_bin"].astype(np.float32) + arrays["O_train"].astype(np.float32)
    Xtok_va = arrays["X_val_bin"].astype(np.float32) + arrays["O_val"].astype(np.float32)
    Xtok_tr_z, Xtok_va_z = standardize_fit_transform(Xtok_tr, Xtok_va)

    Xcls_tr_z, Xcls_va_z = standardize_fit_transform(train_split["cls"], val_split["cls"])

    # kNN fractions in spaces.
    log("Running raw/token/CLS kNN audits")
    raw_knn = knn_fractions(
        train_X=Xraw_tr_z,
        val_X=Xraw_va_z,
        train_labels=train_labels,
        val_df=val_df,
        malware_classes=malware_classes,
        k=int(args.knn_k),
        space="raw",
    )
    token_knn = knn_fractions(
        train_X=Xtok_tr_z,
        val_X=Xtok_va_z,
        train_labels=train_labels,
        val_df=val_df,
        malware_classes=malware_classes,
        k=int(args.knn_k),
        space="token",
    )
    cls_knn = knn_fractions(
        train_X=Xcls_tr_z,
        val_X=Xcls_va_z,
        train_labels=train_labels,
        val_df=val_df,
        malware_classes=malware_classes,
        k=int(args.knn_k),
        space="cls",
    )
    raw_knn.to_csv(out_dir / "02_raw_knn_overlap_with_probs.csv", index=False)
    token_knn.to_csv(out_dir / "02_token_knn_overlap_with_probs.csv", index=False)
    cls_knn.to_csv(out_dir / "02_cls_knn_overlap_with_probs.csv", index=False)

    # Centroids.
    log("Running raw/token/CLS centroid audits")
    raw_cent = centroid_distance_rows(Xraw_tr_z, Xraw_va_z, train_labels, val_df, label_names, malware_classes, "raw")
    tok_cent = centroid_distance_rows(Xtok_tr_z, Xtok_va_z, train_labels, val_df, label_names, malware_classes, "token")
    cls_cent = centroid_distance_rows(Xcls_tr_z, Xcls_va_z, train_labels, val_df, label_names, malware_classes, "cls")
    raw_cent.to_csv(out_dir / "02_raw_centroid_audit.csv", index=False)
    tok_cent.to_csv(out_dir / "02_token_centroid_audit.csv", index=False)
    cls_cent.to_csv(out_dir / "02_cls_centroid_audit.csv", index=False)

    deep = merge_space_signals(val_df, [raw_knn, token_knn, cls_knn, raw_cent, tok_cent, cls_cent])
    deep = classify_failure_modes(deep)
    deep.to_csv(out_dir / "03_val_deep_signal_table.csv", index=False)

    group, pair, mode_df, focus_df = group_summaries(deep, malware_classes, out_dir)

    # Additional summaries.
    summary_rows = []
    for space in ["raw", "token", "cls"]:
        for correct_val, g in deep[deep["true_label"].isin(malware_classes)].groupby("correct"):
            summary_rows.append({
                "space": space,
                "correct": bool(correct_val),
                "n": int(len(g)),
                "mean_true_nn_frac": float(g.get(f"{space}_true_nn_frac", pd.Series(dtype=float)).mean()),
                "mean_pred_nn_frac": float(g.get(f"{space}_pred_nn_frac", pd.Series(dtype=float)).mean()),
                "mean_centroid_margin_pred_minus_true": float(g.get(f"{space}_centroid_margin_pred_minus_true", pd.Series(dtype=float)).mean()),
            })
    pd.DataFrame(summary_rows).to_csv(out_dir / "04_space_summary_by_correct.csv", index=False)

    config = {
        "experiment": "F2b_logits_cls_deep_overlap_audit",
        "diagnostic_only": True,
        "training_performed": False,
        "model_dir": str(model_dir),
        "trainer_module": str(trainer_path),
        "dataset_npz": str(dataset_npz),
        "metadata_json": str(metadata_json),
        "train_raw": str(train_raw_path),
        "val_raw": str(val_raw_path),
        "device": str(device),
        "class_names": label_names,
        "malware_classes": malware_classes,
        "knn_k": int(args.knn_k),
        "raw_numeric_features": int(len(raw_features)),
        "continuous_info": built["continuous_info"],
        "selective_info": built["selective_info"],
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

    write_report(
        out_dir=out_dir,
        model_dir=model_dir,
        trainer_path=trainer_path,
        metrics=metrics,
        reproduction=reproduction,
        group=group,
        pair=pair,
        mode_df=mode_df,
        config=config,
    )

    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out_dir.rglob("*"):
            if p.is_file() and p != zip_path:
                z.write(p, p.relative_to(out_dir.parent))

    log(f"metrics={metrics}")
    log(f"zip={zip_path}")
    log("DONE")


if __name__ == "__main__":
    main()
