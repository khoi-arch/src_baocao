#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B1_export_06_model_cls.py

Purpose
-------
Export fresh validation CLS embeddings from the newly trained official C2+D3
model under:

    03_outputs/06_model/

This is Step B1.0 only: export representation. It does NOT perform pairwise
audit yet. Pairwise audit should run after this export is verified.

Design
------
This script reuses the already-working B0 exporter module for:
  - model reconstruction
  - checkpoint loading
  - dataset loading
  - raw_scaled continuous recomputation

It then forwards the model until the Transformer CLS vector and writes:
  - val_cls_embeddings.npz
  - val_cls_predictions_with_probs.csv
  - B1_export_06_model_cls_summary.md
  - B1_export_06_model_cls_manifest.json
  - B1_export_06_model_cls_output.zip

It lives under 05_test and does not modify official source files.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export CLS embeddings from official 03_outputs/06_model D3 checkpoint.")
    p.add_argument("--repo-root", default=".", help="Path to src_baocao repo root.")
    p.add_argument("--b0-exporter", default="05_test/B0_export_06_model_probs.py")
    p.add_argument("--model-pt", default="03_outputs/06_model/best_model.pt")
    p.add_argument("--config-json", default="03_outputs/06_model/config.json")
    p.add_argument("--dataset-npz", default="03_outputs/05_dataset/dataset.npz")
    p.add_argument("--metadata-json", default="03_outputs/05_dataset/metadata.json")
    p.add_argument("--train-raw", default="01_split/train_raw.csv")
    p.add_argument("--val-raw", default="01_split/val_raw.csv")
    p.add_argument("--out-dir", default="05_test/outputs/B1_cls_pairwise_signal")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


def repo_path(repo_root: Path, path_like: str | Path) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    return repo_root / p


def import_b0_exporter(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing B0 exporter module: {path}\n"
            "B1_export_06_model_cls.py reuses B0_export_06_model_probs.py because that file "
            "has already been verified to reconstruct and load the official D3 checkpoint."
        )
    spec = importlib.util.spec_from_file_location("_b0_export_06_model_probs", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def normalize_label_name(x: Any) -> str:
    return str(x).strip()


def forward_cls_and_logits(model: torch.nn.Module, tokens: torch.Tensor, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Mirror model.forward(), but return the Transformer CLS vector too.

    Expected model structure comes from B0_export_06_model_probs.py:
      model.embedding
      model.input_proj
      model.cls_token
      model.encoder
      model.classifier
      model.hidden_dim
    """
    required_attrs = ["embedding", "input_proj", "cls_token", "encoder", "classifier", "hidden_dim"]
    missing = [a for a in required_attrs if not hasattr(model, a)]
    if missing:
        raise AttributeError(f"Model missing required attributes for CLS extraction: {missing}")

    cell_emb = model.embedding(tokens, values)
    x = model.input_proj(cell_emb)

    B = x.shape[0]
    cls = model.cls_token.expand(B, 1, int(model.hidden_dim))
    x = torch.cat([cls, x], dim=1)

    encoded = model.encoder(x)
    cls_vec = encoded[:, 0, :]
    logits = model.classifier(cls_vec)
    return cls_vec, logits


def zip_outputs(out_dir: Path, zip_name: str = "B1_export_06_model_cls_output.zip") -> Path:
    out_zip = out_dir / zip_name
    if out_zip.exists():
        out_zip.unlink()

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fp in out_dir.rglob("*"):
            if fp.is_file() and fp != out_zip:
                z.write(fp, fp.relative_to(out_dir))
    return out_zip


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()

    b0_exporter = repo_path(repo_root, args.b0_exporter)
    model_pt = repo_path(repo_root, args.model_pt)
    config_json = repo_path(repo_root, args.config_json)
    dataset_npz = repo_path(repo_root, args.dataset_npz)
    metadata_json = repo_path(repo_root, args.metadata_json)
    train_raw = repo_path(repo_root, args.train_raw)
    val_raw = repo_path(repo_root, args.val_raw)
    out_dir = repo_path(repo_root, args.out_dir)

    required = {
        "b0_exporter": b0_exporter,
        "model_pt": model_pt,
        "config_json": config_json,
        "dataset_npz": dataset_npz,
        "metadata_json": metadata_json,
        "train_raw": train_raw,
        "val_raw": val_raw,
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    out_dir.mkdir(parents=True, exist_ok=True)

    b0 = import_b0_exporter(b0_exporter)

    config: Dict[str, Any] = b0.load_json(config_json)
    meta: Dict[str, Any] = b0.load_json(metadata_json)
    data: Dict[str, np.ndarray] = b0.load_dataset_npz(dataset_npz)

    X_val_bin, V_val, y_val, X_val_continuous, continuous_info = b0.build_val_values(
        data=data,
        meta=meta,
        train_raw_path=train_raw,
        val_raw_path=val_raw,
    )

    device = b0.pick_device(args.device)

    model = b0.make_model(config, meta, data).to(device)
    load_info = b0.load_model_state(model, model_pt, device)
    model.eval()

    num_classes = int(model.num_classes)
    hidden_dim = int(model.hidden_dim)
    n_val, n_features = X_val_bin.shape
    label_names: List[str] = b0.label_names_from_config_meta(config, meta, num_classes)

    ds = TensorDataset(
        torch.as_tensor(X_val_bin, dtype=torch.long),
        torch.as_tensor(V_val, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=int(args.batch_size), shuffle=False)

    all_cls: List[np.ndarray] = []
    all_logits: List[np.ndarray] = []

    with torch.no_grad():
        for xb, vb in loader:
            xb = xb.to(device)
            vb = vb.to(device)
            cls_vec, logits = forward_cls_and_logits(model, xb, vb)
            all_cls.append(cls_vec.detach().cpu().numpy())
            all_logits.append(logits.detach().cpu().numpy())

    cls_np = np.concatenate(all_cls, axis=0).astype(np.float32)
    logits_np = np.concatenate(all_logits, axis=0).astype(np.float32)

    if cls_np.shape != (n_val, hidden_dim):
        raise ValueError(f"Unexpected CLS shape: {cls_np.shape}, expected {(n_val, hidden_dim)}")
    if logits_np.shape != (n_val, num_classes):
        raise ValueError(f"Unexpected logits shape: {logits_np.shape}, expected {(n_val, num_classes)}")

    # Softmax in numpy, stable.
    z = logits_np - logits_np.max(axis=1, keepdims=True)
    exp_z = np.exp(z)
    probs_np = exp_z / exp_z.sum(axis=1, keepdims=True)

    top_order = np.argsort(-probs_np, axis=1)
    top1 = top_order[:, 0]
    top2 = top_order[:, 1]
    pred = top1

    idx = np.arange(n_val)
    confidence = probs_np[idx, pred]
    correct = pred == y_val
    top2_hit = (top1 == y_val) | (top2 == y_val)
    wrong = ~correct

    # Save NPZ for downstream B1 audit.
    npz_path = out_dir / "val_cls_embeddings.npz"
    np.savez_compressed(
        npz_path,
        cls_embeddings=cls_np,
        logits=logits_np,
        probs=probs_np.astype(np.float32),
        y_true=y_val.astype(np.int64),
        y_pred=pred.astype(np.int64),
        top1_id=top1.astype(np.int64),
        top2_id=top2.astype(np.int64),
        top1_score=probs_np[idx, top1].astype(np.float32),
        top2_score=probs_np[idx, top2].astype(np.float32),
        sample_index=idx.astype(np.int64),
        label_names=np.array(label_names, dtype=object),
    )

    # Save CSV mirror for easy inspection and consistency with B0.
    rows = {
        "sample_index": idx.astype(int),
        "true_id": y_val.astype(int),
        "true_label": [label_names[int(i)] for i in y_val],
        "pred_id": pred.astype(int),
        "pred_label": [label_names[int(i)] for i in pred],
        "correct": correct.astype(bool),
        "confidence": confidence.astype(float),
        "top1_id": top1.astype(int),
        "top1_label": [label_names[int(i)] for i in top1],
        "top1_score": probs_np[idx, top1].astype(float),
        "top2_id": top2.astype(int),
        "top2_label": [label_names[int(i)] for i in top2],
        "top2_score": probs_np[idx, top2].astype(float),
        "top12_margin": (probs_np[idx, top1] - probs_np[idx, top2]).astype(float),
        "true_in_top2": top2_hit.astype(bool),
    }
    pred_csv = out_dir / "val_cls_predictions_with_probs.csv"
    pred_df = pd.DataFrame(rows)

    for i, name in enumerate(label_names):
        safe = normalize_label_name(name).replace(" ", "_")
        pred_df[f"prob_{safe}"] = probs_np[:, i].astype(float)
        pred_df[f"logit_{safe}"] = logits_np[:, i].astype(float)

    pred_df.to_csv(pred_csv, index=False)

    accuracy = float(correct.mean())
    top2_accuracy = float(top2_hit.mean())
    wrong_total = int(wrong.sum())
    wrong_true_in_top2 = int((wrong & top2_hit).sum())
    wrong_true_in_top2_rate = float(wrong_true_in_top2 / wrong_total) if wrong_total else 0.0

    manifest = {
        "stage": "B1_export_06_model_cls",
        "purpose": "Export fresh CLS embeddings from 03_outputs/06_model best checkpoint for B1 pairwise signal audit.",
        "inputs": {k: str(v) for k, v in required.items()},
        "outputs": {
            "val_cls_embeddings_npz": str(npz_path),
            "val_cls_predictions_with_probs_csv": str(pred_csv),
        },
        "config_summary": {
            "run_id": config.get("run_id"),
            "run_spec": config.get("run_spec"),
            "num_bins": int(model.num_bins),
            "n_features": int(model.n_features),
            "hidden_dim": hidden_dim,
            "num_classes": num_classes,
            "label_names": label_names,
        },
        "data_shapes": {
            "X_val_bin": list(X_val_bin.shape),
            "V_val": list(V_val.shape),
            "y_val": list(y_val.shape),
            "cls_embeddings": list(cls_np.shape),
            "logits": list(logits_np.shape),
            "probs": list(probs_np.shape),
        },
        "continuous_info": continuous_info,
        "load_info": load_info,
        "metrics_from_export": {
            "accuracy": accuracy,
            "top2_accuracy": top2_accuracy,
            "wrong_total": wrong_total,
            "wrong_true_in_top2": wrong_true_in_top2,
            "wrong_true_in_top2_rate": wrong_true_in_top2_rate,
        },
        "device": str(device),
        "batch_size": int(args.batch_size),
    }

    manifest_path = out_dir / "B1_export_06_model_cls_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_path = out_dir / "B1_export_06_model_cls_summary.md"
    summary_path.write_text(
        "\n".join([
            "# B1 Export 06 Model CLS",
            "",
            "This step exports fresh validation CLS embeddings from the newly trained official C2+D3 model.",
            "",
            "## Outputs",
            "",
            f"- `val_cls_embeddings.npz`: `{npz_path}`",
            f"- `val_cls_predictions_with_probs.csv`: `{pred_csv}`",
            f"- `manifest`: `{manifest_path}`",
            "",
            "## Shape checks",
            "",
            f"- n_val: `{n_val}`",
            f"- n_features: `{n_features}`",
            f"- hidden_dim / CLS dim: `{hidden_dim}`",
            f"- cls_embeddings shape: `{tuple(cls_np.shape)}`",
            f"- logits shape: `{tuple(logits_np.shape)}`",
            f"- probs shape: `{tuple(probs_np.shape)}`",
            "",
            "## Prediction consistency metrics",
            "",
            f"- accuracy_from_export: `{accuracy:.10f}`",
            f"- top2_accuracy_from_export: `{top2_accuracy:.10f}`",
            f"- wrong_total: `{wrong_total}`",
            f"- wrong_true_in_top2: `{wrong_true_in_top2}`",
            f"- wrong_true_in_top2_rate: `{wrong_true_in_top2_rate:.10f}`",
            "",
            "## Notes",
            "",
            "- This is export-only. It does not modify official baseline files.",
            "- B1 pairwise audit should use `val_cls_embeddings.npz` from this directory.",
            "",
        ]),
        encoding="utf-8",
    )

    out_zip = zip_outputs(out_dir)

    print("===== B1 export 06_model CLS done =====")
    print("model_pt:", model_pt)
    print("dataset_npz:", dataset_npz)
    print("metadata_json:", metadata_json)
    print("npz:", npz_path)
    print("pred_csv:", pred_csv)
    print("summary:", summary_path)
    print("manifest:", manifest_path)
    print("zip:", out_zip)
    print("cls_shape:", cls_np.shape)
    print("accuracy:", accuracy)
    print("top2_accuracy:", top2_accuracy)
    print("wrong_total:", wrong_total)
    print("wrong_true_in_top2:", wrong_true_in_top2)
    print("wrong_true_in_top2_rate:", wrong_true_in_top2_rate)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
