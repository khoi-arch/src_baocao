#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F3e1 Gated hard-negative boundary-effect audit.

No training.

Input:
  Existing F3e0 output directory:
    05_test/outputs/F3e0_gated_hardneg_calibration

It loads:
  - baseline_ce best_model.pt
  - gated_adaptive_hardneg best_model.pt
  - the same train_inner/calibration subset dataset used by F3e0

It exports logits/probs/CLS/gate on calibration and audits:
  - reproduction metrics
  - fix/damage
  - boundary margin shifts
  - whether learned gate focuses on hard samples
  - class/family damage pattern

This audit does not use official validation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


CLASS_NAMES_DEFAULT = ["Benign", "Ransomware", "Spyware", "Trojan"]
MALWARE_CLASSES_DEFAULT = ["Ransomware", "Spyware", "Trojan"]


def log(msg: str):
    print(f"[F3e1] {msg}", flush=True)


def repo_root_from_here() -> Path:
    p = Path(__file__).resolve()
    if p.parent.name == "05_test":
        return p.parents[1]
    return Path.cwd().resolve()


def resolve_path(p: str | Path, root: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (root / p).resolve()


def import_module_from_path(path: Path, name: str = "f3e_trainer_module"):
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
    return (e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)).astype(np.float32)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


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
        cands = list(df.columns)
    for c in cands:
        if c in df.columns:
            return c
    return None


def load_checkpoint(model_dir: Path, device: torch.device):
    ckpt_path = model_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        ckpt = {"model_state_dict": ckpt, "config": {}}
    return ckpt


def make_runtime_args(
    config: Dict[str, Any],
    dataset_npz: Path,
    metadata_json: Path,
    train_raw: Path,
    val_raw: Path,
    device: str,
    batch_size: int,
    num_workers: int,
) -> SimpleNamespace:
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


def build_model_and_loader(
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
    args = make_runtime_args(config, dataset_npz, metadata_json, train_raw, val_raw, str(device), batch_size, num_workers)

    data, meta = trainer.load_dataset(dataset_npz, metadata_json)

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
    return model, val_loader, label_names, y_val, {
        "continuous_info": continuous_info,
        "missing": list(missing),
        "unexpected": list(unexpected),
    }


@torch.no_grad()
def export_model_outputs(model, loader, device: torch.device) -> Dict[str, np.ndarray]:
    logits_list = []
    y_list = []
    cls_list = []
    gate_logits_list = []

    for batch in loader:
        if len(batch) == 4:
            tokens, values, y, _unused = batch
        else:
            tokens, values, y = batch

        tokens = tokens.to(device, non_blocking=True)
        values = values.to(device, non_blocking=True)

        try:
            logits, cls = model(tokens, values, return_cls=True)
        except TypeError:
            # fallback: capture classifier input using hook
            captured = []
            def pre_hook(module, inputs):
                if inputs and torch.is_tensor(inputs[0]):
                    captured.append(inputs[0].detach())
            h = model.classifier.register_forward_pre_hook(pre_hook)
            logits = model(tokens, values)
            h.remove()
            cls = captured[0]

        if hasattr(model, "difficulty_gate"):
            gate_logits = model.difficulty_gate(cls).view(-1)
        else:
            gate_logits = torch.full((logits.shape[0],), np.nan, device=logits.device)

        logits_list.append(logits.detach().cpu())
        cls_list.append(cls.detach().cpu())
        gate_logits_list.append(gate_logits.detach().cpu())
        y_list.append(y.detach().cpu())

    logits = torch.cat(logits_list, dim=0).numpy().astype(np.float32)
    cls = torch.cat(cls_list, dim=0).numpy().astype(np.float32)
    gate_logits = torch.cat(gate_logits_list, dim=0).numpy().astype(np.float32)
    y = torch.cat(y_list, dim=0).numpy().astype(np.int64)
    probs = softmax_np(logits)
    pred = np.argmax(probs, axis=1).astype(np.int64)
    order = np.argsort(-probs, axis=1).astype(np.int64)
    return {
        "logits": logits,
        "probs": probs,
        "cls": cls,
        "gate_logits": gate_logits,
        "gate": sigmoid_np(gate_logits),
        "y": y,
        "pred": pred,
        "top_order": order,
    }


def metrics_from_outputs(y: np.ndarray, pred: np.ndarray, label_names: List[str]) -> Dict[str, Any]:
    labels = np.arange(len(label_names))
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, labels=labels, average="weighted", zero_division=0)),
    }


def make_compare_df(
    base: Dict[str, np.ndarray],
    gated: Dict[str, np.ndarray],
    label_names: List[str],
    calib_raw: pd.DataFrame,
) -> pd.DataFrame:
    y = base["y"]
    assert np.array_equal(y, gated["y"])
    n = len(y)
    id_to_label = {i: name for i, name in enumerate(label_names)}
    true_label = [id_to_label[int(i)] for i in y]
    base_pred = base["pred"]
    gated_pred = gated["pred"]

    df = pd.DataFrame({
        "row_id": np.arange(n, dtype=np.int64),
        "y": y,
        "true_label": true_label,
        "base_pred": base_pred,
        "base_pred_label": [id_to_label[int(i)] for i in base_pred],
        "gated_pred": gated_pred,
        "gated_pred_label": [id_to_label[int(i)] for i in gated_pred],
        "base_correct": base_pred == y,
        "gated_correct": gated_pred == y,
        "base_true_prob": base["probs"][np.arange(n), y],
        "gated_true_prob": gated["probs"][np.arange(n), y],
        "delta_true_prob": gated["probs"][np.arange(n), y] - base["probs"][np.arange(n), y],
        "base_top1_prob": base["probs"][np.arange(n), base_pred],
        "gated_top1_prob": gated["probs"][np.arange(n), gated_pred],
        "base_gate": base["gate"],
        "gated_gate": gated["gate"],
        "gated_gate_logit": gated["gate_logits"],
    })

    base_top_order = base["top_order"]
    gated_top_order = gated["top_order"]
    df["base_true_rank"] = [int(np.where(base_top_order[i] == y[i])[0][0]) + 1 for i in range(n)]
    df["gated_true_rank"] = [int(np.where(gated_top_order[i] == y[i])[0][0]) + 1 for i in range(n)]
    df["base_true_in_top2"] = df["base_true_rank"] <= 2
    df["gated_true_in_top2"] = df["gated_true_rank"] <= 2

    # Margins against all class labels.
    for j, name in enumerate(label_names):
        df[f"base_logit_{name}"] = base["logits"][:, j]
        df[f"gated_logit_{name}"] = gated["logits"][:, j]
        df[f"base_prob_{name}"] = base["probs"][:, j]
        df[f"gated_prob_{name}"] = gated["probs"][:, j]
        df[f"base_margin_true_minus_{name}"] = base["logits"][np.arange(n), y] - base["logits"][:, j]
        df[f"gated_margin_true_minus_{name}"] = gated["logits"][np.arange(n), y] - gated["logits"][:, j]
        df[f"delta_margin_true_minus_{name}"] = df[f"gated_margin_true_minus_{name}"] - df[f"base_margin_true_minus_{name}"]

    l2_col = find_label_col(calib_raw, "L2")
    l3_col = find_label_col(calib_raw, "L3")
    if l2_col:
        df["raw_L2"] = calib_raw[l2_col].map(clean_label).to_numpy()
    else:
        df["raw_L2"] = df["true_label"]
    if l3_col:
        df["family"] = calib_raw[l3_col].map(clean_label).to_numpy()
    else:
        df["family"] = df["true_label"]

    conditions = [
        df["base_correct"] & df["gated_correct"],
        (~df["base_correct"]) & df["gated_correct"],
        df["base_correct"] & (~df["gated_correct"]),
        (~df["base_correct"]) & (~df["gated_correct"]) & (df["base_pred"] == df["gated_pred"]),
        (~df["base_correct"]) & (~df["gated_correct"]) & (df["base_pred"] != df["gated_pred"]),
    ]
    choices = ["both_correct", "fixed", "damaged", "both_wrong_same", "both_wrong_changed"]
    df["switch_type"] = np.select(conditions, choices, default="unknown")

    return df


def summarize_switches(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    out = {}
    out["switch_counts"] = df["switch_type"].value_counts().rename_axis("switch_type").reset_index(name="count")

    out["switch_by_true_label"] = df.groupby(["true_label", "switch_type"]).agg(
        count=("row_id", "count"),
        mean_gated_gate=("gated_gate", "mean"),
        mean_delta_true_prob=("delta_true_prob", "mean"),
    ).reset_index().sort_values(["true_label", "count"], ascending=[True, False])

    out["switch_by_family"] = df.groupby(["true_label", "family", "switch_type"]).agg(
        count=("row_id", "count"),
        mean_gated_gate=("gated_gate", "mean"),
        mean_delta_true_prob=("delta_true_prob", "mean"),
    ).reset_index().sort_values("count", ascending=False)

    out["transition_summary"] = df.groupby(["true_label", "base_pred_label", "gated_pred_label"]).agg(
        count=("row_id", "count"),
        mean_gated_gate=("gated_gate", "mean"),
        mean_delta_true_prob=("delta_true_prob", "mean"),
    ).reset_index().sort_values("count", ascending=False)

    return out


def pair_margin_summary(df: pd.DataFrame, label_names: List[str]) -> pd.DataFrame:
    rows = []
    for true_label in label_names:
        gtrue = df[df["true_label"] == true_label]
        if len(gtrue) == 0:
            continue
        for conf in label_names:
            if conf == true_label:
                continue
            base_col = f"base_margin_true_minus_{conf}"
            gated_col = f"gated_margin_true_minus_{conf}"
            delta_col = f"delta_margin_true_minus_{conf}"
            rows.append({
                "true_label": true_label,
                "confuser": conf,
                "support": int(len(gtrue)),
                "base_margin_mean": float(gtrue[base_col].mean()),
                "gated_margin_mean": float(gtrue[gated_col].mean()),
                "delta_margin_mean": float(gtrue[delta_col].mean()),
                "base_margin_median": float(gtrue[base_col].median()),
                "gated_margin_median": float(gtrue[gated_col].median()),
                "delta_margin_median": float(gtrue[delta_col].median()),
                "base_correct_rate": float(gtrue["base_correct"].mean()),
                "gated_correct_rate": float(gtrue["gated_correct"].mean()),
                "delta_correct_rate": float(gtrue["gated_correct"].mean() - gtrue["base_correct"].mean()),
                "fix_count": int(((gtrue["switch_type"] == "fixed")).sum()),
                "damage_count": int(((gtrue["switch_type"] == "damaged")).sum()),
            })
    return pd.DataFrame(rows).sort_values(["delta_correct_rate", "support"], ascending=[True, False])


def gate_diagnostics(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    out = {}
    out["gate_by_switch_type"] = df.groupby("switch_type").agg(
        count=("row_id", "count"),
        gate_mean=("gated_gate", "mean"),
        gate_median=("gated_gate", "median"),
        gate_std=("gated_gate", "std"),
        base_true_prob_mean=("base_true_prob", "mean"),
        gated_true_prob_mean=("gated_true_prob", "mean"),
        delta_true_prob_mean=("delta_true_prob", "mean"),
    ).reset_index().sort_values("count", ascending=False)

    out["gate_by_true_label"] = df.groupby("true_label").agg(
        count=("row_id", "count"),
        gate_mean=("gated_gate", "mean"),
        gate_median=("gated_gate", "median"),
        base_correct_rate=("base_correct", "mean"),
        gated_correct_rate=("gated_correct", "mean"),
        delta_correct_rate=("gated_correct", lambda s: np.nan), # filled below
    ).reset_index()
    out["gate_by_true_label"]["delta_correct_rate"] = out["gate_by_true_label"].apply(
        lambda r: float(
            df[df["true_label"] == r["true_label"]]["gated_correct"].mean()
            - df[df["true_label"] == r["true_label"]]["base_correct"].mean()
        ),
        axis=1,
    )

    out["gate_by_family"] = df.groupby(["true_label", "family"]).agg(
        count=("row_id", "count"),
        gate_mean=("gated_gate", "mean"),
        gate_median=("gated_gate", "median"),
        base_correct_rate=("base_correct", "mean"),
        gated_correct_rate=("gated_correct", "mean"),
    ).reset_index()
    out["gate_by_family"]["delta_correct_rate"] = out["gate_by_family"]["gated_correct_rate"] - out["gate_by_family"]["base_correct_rate"]
    out["gate_by_family"] = out["gate_by_family"].sort_values(["count"], ascending=False)

    return out


def write_report(
    out_dir: Path,
    metrics: Dict[str, Any],
    switch_summary: Dict[str, pd.DataFrame],
    pair_summary: pd.DataFrame,
    gate_summary: Dict[str, pd.DataFrame],
    flags: Dict[str, Any],
):
    def md(df, n=20):
        if df is None or len(df) == 0:
            return "_empty_"
        try:
            return df.head(n).to_markdown(index=False)
        except Exception:
            return df.head(n).to_string(index=False)

    lines = []
    lines.append("# F3e1 gated hard-negative boundary-effect audit\n")
    lines.append("## Scope\n")
    lines.append("```text")
    lines.append("No training. No official validation.")
    lines.append("Compare F3e0 baseline_ce vs gated_adaptive_hardneg on the same calibration split.")
    lines.append("Goal: explain why gated loss fixed Trojan but damaged Ransomware.")
    lines.append("```")
    lines.append("\n## Metrics\n")
    lines.append("```json")
    lines.append(json.dumps(metrics, indent=2, default=str))
    lines.append("```")
    lines.append("\n## Switch counts\n")
    lines.append(md(switch_summary["switch_counts"]))
    lines.append("\n## Switch by true label\n")
    lines.append(md(switch_summary["switch_by_true_label"], 30))
    lines.append("\n## Major prediction transitions\n")
    lines.append(md(switch_summary["transition_summary"], 30))
    lines.append("\n## Pair margin shifts\n")
    lines.append(md(pair_summary, 30))
    lines.append("\n## Gate by switch type\n")
    lines.append(md(gate_summary["gate_by_switch_type"], 20))
    lines.append("\n## Gate by family\n")
    lines.append(md(gate_summary["gate_by_family"], 30))
    lines.append("\n## Flags\n")
    lines.append("```json")
    lines.append(json.dumps(flags, indent=2, default=str))
    lines.append("```")
    (out_dir / "F3e1_boundary_effect_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f3e0-dir", default="05_test/outputs/F3e0_gated_hardneg_calibration")
    ap.add_argument("--trainer", default="02_src/07_train_gated_hardneg.py")
    ap.add_argument("--metadata-json", default="03_outputs/05_dataset/metadata.json")
    ap.add_argument("--out-dir", default="05_test/outputs/F3e1_gated_boundary_effect_audit")
    ap.add_argument("--combined-zip", default="05_test/outputs/F3e1_gated_boundary_effect_audit.zip")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    root = repo_root_from_here()
    f3e0_dir = resolve_path(args.f3e0_dir, root)
    trainer_path = resolve_path(args.trainer, root)
    metadata_json = resolve_path(args.metadata_json, root)
    out_dir = resolve_path(args.out_dir, root)
    zip_path = resolve_path(args.combined_zip, root)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not f3e0_dir.exists():
        raise FileNotFoundError(f"F3e0 dir not found: {f3e0_dir}")
    if not trainer_path.exists():
        raise FileNotFoundError(f"trainer not found: {trainer_path}")

    split_dir = f3e0_dir / "_split_artifacts"
    dataset_npz = split_dir / "dataset_train_inner_calibration.npz"
    train_raw = split_dir / "train_inner_raw.csv"
    calib_raw_path = split_dir / "calibration_raw.csv"
    calib_raw = pd.read_csv(calib_raw_path)

    base_dir = f3e0_dir / "config_runs" / "Keff512" / "baseline_ce"
    gated_dir = f3e0_dir / "config_runs" / "Keff512" / "gated_adaptive_hardneg"
    if not base_dir.exists() or not gated_dir.exists():
        raise FileNotFoundError(f"Missing baseline/gated model dirs under {f3e0_dir}/config_runs/Keff512")

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    trainer = import_module_from_path(trainer_path)

    outputs = {}
    build_infos = {}
    label_names = None
    for name, model_dir in [("baseline", base_dir), ("gated", gated_dir)]:
        ckpt = load_checkpoint(model_dir, device)
        model, loader, labels, y_val, info = build_model_and_loader(
            trainer=trainer,
            ckpt=ckpt,
            dataset_npz=dataset_npz,
            metadata_json=metadata_json,
            train_raw=train_raw,
            val_raw=calib_raw_path,
            device=device,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
        )
        label_names = labels
        outputs[name] = export_model_outputs(model, loader, device)
        build_infos[name] = info
        log(f"exported {name}: n={len(outputs[name]['y'])}")

    assert label_names is not None
    compare = make_compare_df(outputs["baseline"], outputs["gated"], label_names, calib_raw)
    compare.to_csv(out_dir / "F3e1_calibration_compare_predictions_logits_probs_gate.csv", index=False)

    # Save arrays too.
    np.savez_compressed(
        out_dir / "F3e1_exported_arrays.npz",
        baseline_logits=outputs["baseline"]["logits"],
        baseline_probs=outputs["baseline"]["probs"],
        baseline_cls=outputs["baseline"]["cls"],
        baseline_gate=outputs["baseline"]["gate"],
        gated_logits=outputs["gated"]["logits"],
        gated_probs=outputs["gated"]["probs"],
        gated_cls=outputs["gated"]["cls"],
        gated_gate=outputs["gated"]["gate"],
        y=outputs["baseline"]["y"],
    )

    metrics = {
        "baseline": metrics_from_outputs(outputs["baseline"]["y"], outputs["baseline"]["pred"], label_names),
        "gated": metrics_from_outputs(outputs["gated"]["y"], outputs["gated"]["pred"], label_names),
    }
    metrics["delta"] = {
        "accuracy": metrics["gated"]["accuracy"] - metrics["baseline"]["accuracy"],
        "macro_f1": metrics["gated"]["macro_f1"] - metrics["baseline"]["macro_f1"],
        "weighted_f1": metrics["gated"]["weighted_f1"] - metrics["baseline"]["weighted_f1"],
    }
    (out_dir / "F3e1_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Classification reports and confusion.
    for name in ["baseline", "gated"]:
        y = outputs[name]["y"]
        pred = outputs[name]["pred"]
        rep = classification_report(
            y,
            pred,
            labels=np.arange(len(label_names)),
            target_names=label_names,
            output_dict=True,
            zero_division=0,
        )
        pd.DataFrame(rep).T.to_csv(out_dir / f"F3e1_{name}_classification_report.csv")
        cm = confusion_matrix(y, pred, labels=np.arange(len(label_names)))
        pd.DataFrame(cm, index=[f"true_{x}" for x in label_names], columns=[f"pred_{x}" for x in label_names]).to_csv(
            out_dir / f"F3e1_{name}_confusion_matrix.csv"
        )

    switch_summary = summarize_switches(compare)
    for k, v in switch_summary.items():
        v.to_csv(out_dir / f"F3e1_{k}.csv", index=False)

    pair_summary = pair_margin_summary(compare, label_names)
    pair_summary.to_csv(out_dir / "F3e1_pair_margin_shift_summary.csv", index=False)

    gate_summary = gate_diagnostics(compare)
    for k, v in gate_summary.items():
        v.to_csv(out_dir / f"F3e1_{k}.csv", index=False)

    # High-value focus groups.
    focus = compare[
        (
            ((compare["true_label"] == "Trojan") & (compare["base_pred_label"] == "Ransomware"))
            | ((compare["true_label"] == "Ransomware") & (compare["gated_pred_label"].isin(["Trojan", "Spyware"])))
            | (compare["switch_type"].isin(["fixed", "damaged"]))
        )
    ].copy()
    focus.to_csv(out_dir / "F3e1_focus_rows_fixed_damaged_TR_RT.csv", index=False)

    # Flags: concise interpretation signals.
    switch_counts = compare["switch_type"].value_counts().to_dict()
    per_class = compare.groupby("true_label").agg(
        support=("row_id", "count"),
        base_acc=("base_correct", "mean"),
        gated_acc=("gated_correct", "mean"),
        fix=("switch_type", lambda s: int((s == "fixed").sum())),
        damage=("switch_type", lambda s: int((s == "damaged").sum())),
        gate_mean=("gated_gate", "mean"),
    ).reset_index()
    per_class["delta_acc"] = per_class["gated_acc"] - per_class["base_acc"]

    flags = {
        "fix_count": int(switch_counts.get("fixed", 0)),
        "damage_count": int(switch_counts.get("damaged", 0)),
        "net_fix_minus_damage": int(switch_counts.get("fixed", 0) - switch_counts.get("damaged", 0)),
        "delta_macro_f1": float(metrics["delta"]["macro_f1"]),
        "per_class_delta_acc": per_class.to_dict(orient="records"),
        "gate_mean_by_switch": gate_summary["gate_by_switch_type"].to_dict(orient="records"),
        "build_infos": build_infos,
        "interpretation_hint": (
            "If gated gate is not much higher on fixed/hard samples than damaged/easy samples, "
            "the learned gate is not selective enough. If Ransomware margins vs Trojan/Spyware fall while Trojan vs Ransomware rises, "
            "boundary moved asymmetrically and overcorrected."
        ),
    }
    (out_dir / "F3e1_interpretation_flags.json").write_text(json.dumps(flags, indent=2, default=str), encoding="utf-8")
    per_class.to_csv(out_dir / "F3e1_per_class_fix_damage.csv", index=False)

    config = {
        "experiment": "F3e1_gated_boundary_effect_audit",
        "training_performed": False,
        "official_validation_used": False,
        "f3e0_dir": str(f3e0_dir),
        "trainer": str(trainer_path),
        "dataset_npz": str(dataset_npz),
        "train_raw": str(train_raw),
        "calibration_raw": str(calib_raw_path),
        "metrics": metrics,
        "class_names": label_names,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

    write_report(out_dir, metrics, switch_summary, pair_summary, gate_summary, flags)

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
