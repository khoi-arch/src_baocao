#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F3a1 train/export one clean OOF fold.

Clean nested protocol for one outer fold:
  1. outer_oof_idx = rows with fold == fold_id
  2. train_pool_idx = rows with fold != fold_id
  3. split train_pool into inner_train / inner_val
  4. train official L1 on inner_train, early stop on inner_val
  5. export logits/probs/CLS for outer_oof_idx only

The outer fold is NOT used for:
  - training
  - early stopping
  - checkpoint selection

This script runs one fold only by default. Use fold_id=0 first.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import subprocess
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit


LABEL_COL_CANDIDATES = [
    "label_L1", "label_L2", "label_L3", "Label_L1", "Label_L2", "Label_L3",
    "Class", "Category", "Family", "class", "category", "family",
    "MalwareFamily", "malware_family", "label", "target",
]


def log(msg: str):
    print(f"[F3a1] {msg}", flush=True)


def repo_root_from_here() -> Path:
    p = Path(__file__).resolve()
    if p.parent.name == "05_test":
        return p.parents[1]
    return Path.cwd().resolve()


def resolve_path(p: str | Path, root: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (root / p).resolve()


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


def import_module_from_path(path: Path, name: str = "official_trainer_module"):
    path = Path(path).resolve()
    repo_root = path.parents[1] if path.parent.name == "02_src" else Path.cwd().resolve()
    for p in [str(path.parent), str(repo_root)]:
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - np.nanmax(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)


def choose_inner_split(train_pool_idx: np.ndarray, raw: pd.DataFrame, inner_val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    l2_col = find_label_col(raw, "L2")
    l3_col = find_label_col(raw, "L3")
    if l2_col is None:
        raise ValueError("Cannot find L2 label column in train_raw")

    l2 = raw[l2_col].iloc[train_pool_idx].map(clean_label).reset_index(drop=True)
    if l3_col:
        l3 = raw[l3_col].iloc[train_pool_idx].map(clean_label).reset_index(drop=True)
        l3 = l3.where(l3.astype(str).str.len() > 0, l2)
    else:
        l3 = l2

    combo = l2.astype(str) + "::" + l3.astype(str)
    combo_counts = combo.value_counts()
    l2_counts = l2.value_counts()
    if len(combo_counts) and int(combo_counts.min()) >= 2:
        strata = combo
        mode = "L2_plus_L3"
    elif len(l2_counts) and int(l2_counts.min()) >= 2:
        strata = l2
        mode = "L2_only_due_to_rare_L3"
    else:
        raise ValueError("Cannot make inner stratified split")

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=float(inner_val_frac), random_state=int(seed))
    local_train, local_val = next(splitter.split(np.zeros(len(train_pool_idx)), strata.to_numpy()))
    inner_train_idx = train_pool_idx[local_train]
    inner_val_idx = train_pool_idx[local_val]
    info = {
        "inner_stratification_mode": mode,
        "inner_val_frac": float(inner_val_frac),
        "inner_train_n": int(len(inner_train_idx)),
        "inner_val_n": int(len(inner_val_idx)),
        "inner_seed": int(seed),
        "min_combo_count_train_pool": int(combo_counts.min()) if len(combo_counts) else None,
    }
    return inner_train_idx.astype(np.int64), inner_val_idx.astype(np.int64), info


def make_subset_dataset_and_raw(
    *,
    dataset_npz: Path,
    train_raw: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    out_dir: Path,
    prefix: str,
) -> Tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    z = np.load(dataset_npz, allow_pickle=True)
    needed = ["X_train_bin", "X_train_offset", "y_train"]
    missing = [k for k in needed if k not in z.files]
    if missing:
        raise KeyError(f"dataset npz missing keys: {missing}")

    arrays = {
        "X_train_bin": np.asarray(z["X_train_bin"])[train_idx],
        "X_train_offset": np.asarray(z["X_train_offset"])[train_idx],
        "y_train": np.asarray(z["y_train"])[train_idx],
        "X_val_bin": np.asarray(z["X_train_bin"])[val_idx],
        "X_val_offset": np.asarray(z["X_train_offset"])[val_idx],
        "y_val": np.asarray(z["y_train"])[val_idx],
    }
    ds_path = out_dir / f"{prefix}_dataset.npz"
    np.savez_compressed(ds_path, **arrays)

    train_raw_path = out_dir / f"{prefix}_train_raw.csv"
    val_raw_path = out_dir / f"{prefix}_val_raw.csv"
    train_raw.iloc[train_idx].reset_index(drop=True).to_csv(train_raw_path, index=False)
    train_raw.iloc[val_idx].reset_index(drop=True).to_csv(val_raw_path, index=False)
    return ds_path, train_raw_path, val_raw_path


def find_run_dir(out_root: Path, run_name: str) -> Path:
    cands = [out_root / "Keff512" / run_name, out_root / run_name]
    cands += [p for p in out_root.rglob(run_name) if p.is_dir()]
    for c in cands:
        if (c / "best_model.pt").exists() and (c / "history.csv").exists():
            return c
    raise FileNotFoundError(f"Cannot find completed run dir for {run_name} under {out_root}")


def load_checkpoint(model_dir: Path, device: torch.device):
    ckpt_path = model_dir / "best_model.pt"
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        ckpt = {"model_state_dict": ckpt, "config": {}}
    return ckpt


def make_runtime_args(config: Dict[str, Any], dataset_npz: Path, metadata_json: Path, train_raw: Path, val_raw: Path, device: str, batch_size: int, num_workers: int) -> SimpleNamespace:
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


def build_model_and_export_loader(
    *,
    trainer,
    ckpt,
    export_dataset_npz: Path,
    metadata_json: Path,
    export_train_raw: Path,
    export_oof_raw: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
):
    config = ckpt.get("config") or {}
    args = make_runtime_args(config, export_dataset_npz, metadata_json, export_train_raw, export_oof_raw, str(device), batch_size, num_workers)

    data, meta = trainer.load_dataset(export_dataset_npz, metadata_json)

    X_train = data["X_train_bin"].astype(np.int64)
    O_train = data["X_train_offset"].astype(np.float32)
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

    M_val = np.ones_like(X_val, dtype=np.float32)
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
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    val_ds = trainer.FusionAblationDataset(X_val, O_val, X_val_cont, M_val, y_val)
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    return model, val_loader, label_names, {"continuous_info": continuous_info, "missing": list(missing), "unexpected": list(unexpected)}


@torch.no_grad()
def export_logits_probs_cls(model, loader, device: torch.device):
    logits_list = []
    y_list = []
    cls_list = []
    captured = []

    def pre_hook(module, inputs):
        if inputs and torch.is_tensor(inputs[0]):
            captured.append(inputs[0].detach().cpu())

    handle = model.classifier.register_forward_pre_hook(pre_hook)
    try:
        for batch in loader:
            if len(batch) == 4:
                tokens, values, y, _ = batch
            else:
                tokens, values, y = batch
            tokens = tokens.to(device, non_blocking=True)
            values = values.to(device, non_blocking=True)
            logits = model(tokens, values)
            logits_list.append(logits.detach().cpu())
            y_list.append(y.detach().cpu())
    finally:
        handle.remove()

    logits = torch.cat(logits_list, dim=0).numpy().astype(np.float32)
    y = torch.cat(y_list, dim=0).numpy().astype(np.int64)
    cls = torch.cat(captured, dim=0).numpy().astype(np.float32)
    probs = softmax_np(logits).astype(np.float32)
    pred = np.argmax(probs, axis=1).astype(np.int64)
    top_order = np.argsort(-probs, axis=1).astype(np.int64)
    return logits, probs, cls, y, pred, top_order


def make_prediction_df(
    *,
    oof_idx: np.ndarray,
    logits: np.ndarray,
    probs: np.ndarray,
    cls: np.ndarray,
    y: np.ndarray,
    pred: np.ndarray,
    top_order: np.ndarray,
    oof_raw: pd.DataFrame,
    label_names: List[str],
) -> pd.DataFrame:
    id_to_name = {i: c for i, c in enumerate(label_names)}
    n = len(y)
    df = pd.DataFrame({
        "original_row_id": oof_idx.astype(np.int64),
        "oof_local_row_id": np.arange(n, dtype=np.int64),
        "y": y,
        "pred": pred,
        "true_label": [id_to_name[int(i)] for i in y],
        "pred_label": [id_to_name[int(i)] for i in pred],
        "correct": y == pred,
        "top1_id": top_order[:, 0],
        "top2_id": top_order[:, 1],
        "top1_prob": probs[np.arange(n), top_order[:, 0]],
        "top2_prob": probs[np.arange(n), top_order[:, 1]],
        "true_prob": probs[np.arange(n), y],
        "pred_prob": probs[np.arange(n), pred],
        "margin_top1_top2": probs[np.arange(n), top_order[:, 0]] - probs[np.arange(n), top_order[:, 1]],
        "true_rank": [int(np.where(top_order[i] == y[i])[0][0]) + 1 for i in range(n)],
    })
    df["true_in_top2"] = df["true_rank"] <= 2
    l2_col = find_label_col(oof_raw, "L2")
    l3_col = find_label_col(oof_raw, "L3")
    if l2_col:
        df["raw_L2"] = oof_raw[l2_col].map(clean_label).to_numpy()
    else:
        df["raw_L2"] = df["true_label"]
    if l3_col:
        df["family"] = oof_raw[l3_col].map(clean_label).to_numpy()
    else:
        df["family"] = df["true_label"]

    for i, c in enumerate(label_names):
        df[f"prob_{c}"] = probs[:, i]
        df[f"logit_{c}"] = logits[:, i]
    return df


def safe_md(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or len(df) == 0:
        return "_empty_"
    try:
        return df.head(max_rows).to_markdown(index=False)
    except Exception:
        return df.head(max_rows).to_string(index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-id", type=int, default=0)
    ap.add_argument("--fold-assignments", default="05_test/outputs/F3a0_oof_fold_split_audit/F3a0_oof_fold_assignments.csv")
    ap.add_argument("--dataset-npz", default="03_outputs/05_dataset/dataset.npz")
    ap.add_argument("--metadata-json", default="03_outputs/05_dataset/metadata.json")
    ap.add_argument("--train-raw", default="01_split/train_raw.csv")
    ap.add_argument("--trainer", default="02_src/07_train_sam.py")
    ap.add_argument("--out-dir", default="05_test/outputs/F3a1_oof_fold0_train_export")
    ap.add_argument("--combined-zip", default="05_test/outputs/F3a1_oof_fold0_train_export.zip")
    ap.add_argument("--inner-val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)

    # official L1 training args
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    root = repo_root_from_here()
    fold_assignments = resolve_path(args.fold_assignments, root)
    dataset_npz = resolve_path(args.dataset_npz, root)
    metadata_json = resolve_path(args.metadata_json, root)
    train_raw_path = resolve_path(args.train_raw, root)
    trainer_path = resolve_path(args.trainer, root)
    out_dir = resolve_path(args.out_dir, root)
    zip_path = resolve_path(args.combined_zip, root)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not trainer_path.exists():
        raise FileNotFoundError(f"trainer not found: {trainer_path}")

    raw = pd.read_csv(train_raw_path)
    assign = pd.read_csv(fold_assignments)
    if "row_id" not in assign.columns or "fold" not in assign.columns:
        raise ValueError("fold assignments must contain row_id and fold")

    fold_id = int(args.fold_id)
    oof_idx = assign.loc[assign["fold"] == fold_id, "row_id"].to_numpy(dtype=np.int64)
    train_pool_idx = assign.loc[assign["fold"] != fold_id, "row_id"].to_numpy(dtype=np.int64)
    if len(oof_idx) == 0:
        raise ValueError(f"No rows for fold_id={fold_id}")

    inner_train_idx, inner_val_idx, inner_info = choose_inner_split(
        train_pool_idx=train_pool_idx,
        raw=raw,
        inner_val_frac=float(args.inner_val_frac),
        seed=int(args.seed) + fold_id,
    )

    artifact_dir = out_dir / "_fold_artifacts"
    train_ds_path, train_inner_raw_path, inner_val_raw_path = make_subset_dataset_and_raw(
        dataset_npz=dataset_npz,
        train_raw=raw,
        train_idx=inner_train_idx,
        val_idx=inner_val_idx,
        out_dir=artifact_dir,
        prefix=f"fold{fold_id}_train_inner_val",
    )

    # For export, train side must be the same inner_train used by the trained model
    # so continuous raw scaling matches training.
    export_ds_path, export_train_raw_path, export_oof_raw_path = make_subset_dataset_and_raw(
        dataset_npz=dataset_npz,
        train_raw=raw,
        train_idx=inner_train_idx,
        val_idx=oof_idx,
        out_dir=artifact_dir,
        prefix=f"fold{fold_id}_export_oof",
    )

    run_name = f"fold_{fold_id}_innertrain"
    run_root = out_dir / "fold_runs"

    cmd = [
        "python", str(trainer_path.relative_to(root) if trainer_path.is_relative_to(root) else trainer_path),
        "--run-id", "D3",
        "--K", "512",
        "--num-bins", "512",
        "--dataset-npz", str(train_ds_path.relative_to(root) if train_ds_path.is_relative_to(root) else train_ds_path),
        "--metadata-json", str(metadata_json.relative_to(root) if metadata_json.is_relative_to(root) else metadata_json),
        "--train-raw", str(train_inner_raw_path.relative_to(root) if train_inner_raw_path.is_relative_to(root) else train_inner_raw_path),
        "--val-raw", str(inner_val_raw_path.relative_to(root) if inner_val_raw_path.is_relative_to(root) else inner_val_raw_path),
        "--out-root", str(run_root.relative_to(root) if run_root.is_relative_to(root) else run_root),
        "--run-name", run_name,
        "--num-layers", "1",
        "--epochs", str(int(args.epochs)),
        "--batch-size", str(int(args.batch_size)),
        "--patience", str(int(args.patience)),
        "--device", str(args.device),
        "--num-workers", str(int(args.num_workers)),
    ]

    # 07_train_sam.py accepts SAM args; official 07_train.py may not.
    if "07_train_sam" in trainer_path.name or "07_train_overlap" in trainer_path.name:
        cmd += ["--sam-rho", "0.0"]

    log(f"Training fold={fold_id} with clean inner split")
    subprocess.run(cmd, cwd=root, check=True)
    model_dir = find_run_dir(run_root, run_name)

    # Export OOF logits/probs/CLS.
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    trainer = import_module_from_path(trainer_path)
    ckpt = load_checkpoint(model_dir, device)

    model, export_loader, label_names, export_info = build_model_and_export_loader(
        trainer=trainer,
        ckpt=ckpt,
        export_dataset_npz=export_ds_path,
        metadata_json=metadata_json,
        export_train_raw=export_train_raw_path,
        export_oof_raw=export_oof_raw_path,
        device=device,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
    )
    logits, probs, cls, y, pred, top_order = export_logits_probs_cls(model, export_loader, device)
    oof_raw_df = pd.read_csv(export_oof_raw_path)
    pred_df = make_prediction_df(
        oof_idx=oof_idx,
        logits=logits,
        probs=probs,
        cls=cls,
        y=y,
        pred=pred,
        top_order=top_order,
        oof_raw=oof_raw_df,
        label_names=label_names,
    )
    pred_df.to_csv(out_dir / f"F3a1_fold{fold_id}_oof_predictions_logits_probs.csv", index=False)
    np.savez_compressed(
        out_dir / f"F3a1_fold{fold_id}_oof_arrays.npz",
        oof_original_row_id=oof_idx,
        oof_logits=logits,
        oof_probs=probs,
        oof_cls=cls,
        oof_y=y,
        oof_pred=pred,
    )

    report = classification_report(pred_df["true_label"], pred_df["pred_label"], labels=label_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(pred_df["true_label"], pred_df["pred_label"], labels=label_names)
    pd.DataFrame(report).T.to_csv(out_dir / f"F3a1_fold{fold_id}_oof_classification_report.csv")
    pd.DataFrame(cm, index=[f"true_{c}" for c in label_names], columns=[f"pred_{c}" for c in label_names]).to_csv(
        out_dir / f"F3a1_fold{fold_id}_oof_confusion_matrix.csv"
    )
    metrics = {
        "fold_id": fold_id,
        "oof_n": int(len(oof_idx)),
        "inner_train_n": int(len(inner_train_idx)),
        "inner_val_n": int(len(inner_val_idx)),
        "train_pool_n": int(len(train_pool_idx)),
        "oof_accuracy": float(accuracy_score(pred_df["true_label"], pred_df["pred_label"])),
        "oof_macro_f1": float(f1_score(pred_df["true_label"], pred_df["pred_label"], labels=label_names, average="macro", zero_division=0)),
        "oof_weighted_f1": float(f1_score(pred_df["true_label"], pred_df["pred_label"], labels=label_names, average="weighted", zero_division=0)),
    }

    # Hard group summary on this fold.
    wrong = pred_df[(pred_df["correct"] == False) & (pred_df["true_label"].isin(["Ransomware", "Spyware", "Trojan"]))].copy()
    if len(wrong):
        hard = wrong.groupby(["true_label", "pred_label", "family"]).agg(
            count=("original_row_id", "count"),
            mean_true_prob=("true_prob", "mean"),
            mean_pred_prob=("pred_prob", "mean"),
            true_in_top2_rate=("true_in_top2", "mean"),
        ).reset_index().sort_values("count", ascending=False)
    else:
        hard = pd.DataFrame()
    hard.to_csv(out_dir / f"F3a1_fold{fold_id}_hard_pair_family_summary.csv", index=False)

    config = {
        "experiment": "F3a1_one_clean_oof_fold_train_export",
        "training_performed": True,
        "official_validation_used": False,
        "outer_fold_used_for_training": False,
        "outer_fold_used_for_early_stopping": False,
        "fold_id": fold_id,
        "protocol": "outer fold is pure OOF holdout; inner_val from train_pool is used for early stopping",
        "fold_assignments": str(fold_assignments),
        "dataset_npz": str(dataset_npz),
        "metadata_json": str(metadata_json),
        "train_raw": str(train_raw_path),
        "trainer": str(trainer_path),
        "model_dir": str(model_dir),
        "inner_split_info": inner_info,
        "metrics": metrics,
        "export_info": export_info,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    (out_dir / f"F3a1_fold{fold_id}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    lines = []
    lines.append(f"# F3a1 fold {fold_id} clean OOF train/export\n")
    lines.append("## Protocol\n")
    lines.append("```text")
    lines.append("outer_oof_idx = fold == fold_id")
    lines.append("train_pool_idx = fold != fold_id")
    lines.append("inner_train/inner_val split is made only from train_pool")
    lines.append("official L1 trains on inner_train and early-stops on inner_val")
    lines.append("outer_oof_idx is used only for final OOF prediction export")
    lines.append("```")
    lines.append("\n## Metrics\n")
    lines.append("```json")
    lines.append(json.dumps(metrics, indent=2))
    lines.append("```")
    lines.append("\n## Hard pair/family summary\n")
    try:
        lines.append(hard.head(20).to_markdown(index=False))
    except Exception:
        lines.append(hard.head(20).to_string(index=False))
    lines.append("\n## Next\n")
    lines.append("```text")
    lines.append("If fold 0 output looks sane, run folds 1-4 with the same script.")
    lines.append("Then concatenate all fold OOF predictions for F3a2 OOF overlap reproduction audit.")
    lines.append("```")
    (out_dir / f"F3a1_fold{fold_id}_report.md").write_text("\n".join(lines), encoding="utf-8")

    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out_dir.rglob("*"):
            if p.is_file() and p != zip_path:
                z.write(p, p.relative_to(out_dir.parent))

    log(f"model_dir={model_dir}")
    log(f"metrics={metrics}")
    log(f"zip={zip_path}")
    log("DONE")


if __name__ == "__main__":
    main()
