#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


def import_train_module(repo_root: Path):
    """
    Historical official D3 was trained by the old fusion-ablation code.
    In the cleaned repo, 02_src/07_train.py no longer exposes those classes.
    So this exporter loads the module that still contains the historical
    FusionAblationTransformer definition, usually audit.py or audit_rootcause.py.
    """
    src_dir = repo_root / "02_src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    candidates = [
        src_dir / "audit.py",
        src_dir / "audit_rootcause.py",
        src_dir / "06_audit_best.py",
        src_dir / "07_audit_rootcause.py",
        src_dir / "07_train.py",
    ]

    last_err = None
    for module_path in candidates:
        if not module_path.exists():
            continue
        try:
            spec = importlib.util.spec_from_file_location("_official_d3_model_mod", module_path)
            mod = importlib.util.module_from_spec(spec)
            assert spec is not None and spec.loader is not None
            spec.loader.exec_module(mod)
            if hasattr(mod, "FusionAblationTransformer"):
                print(f"[INFO] loaded FusionAblationTransformer from {module_path}")
                return mod
            last_err = f"{module_path} loaded but has no FusionAblationTransformer"
        except Exception as e:
            last_err = f"{module_path}: {type(e).__name__}: {e}"

    raise AttributeError(
        "Could not find FusionAblationTransformer in 02_src/audit.py, "
        "audit_rootcause.py, 06_audit_best.py, 07_audit_rootcause.py, or 07_train.py. "
        f"Last error: {last_err}"
    )


def resolve_existing(repo_root: Path, p: str | Path, fallback: str) -> Path:
    p = Path(p)
    if p.exists():
        return p
    if not p.is_absolute() and (repo_root / p).exists():
        return repo_root / p

    parts = list(p.parts)
    for anchor in ["03_outputs", "01_split", "00_raw_dataset"]:
        if anchor in parts:
            idx = parts.index(anchor)
            cand = repo_root / Path(*parts[idx:])
            if cand.exists():
                return cand

    cand = repo_root / fallback
    if cand.exists():
        return cand

    raise FileNotFoundError(f"cannot resolve path: {p}, fallback={cand}")


def pick_device(device: str) -> torch.device:
    device = str(device).lower().strip()
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] cuda requested but not available; fallback cpu")
        return torch.device("cpu")
    return torch.device(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument(
        "--run-dir",
        default="03_outputs/train_runs_fusion_ablation_D0_D7/Keff512/D3_P1_K512_B512_C2_selective_rank_discrete_compact",
    )
    ap.add_argument(
        "--dataset-npz",
        default="03_outputs/build_mixed_quantile_offset/K512_B512_C2_selective_rank_discrete_compact/mixed_quantile_offset_dataset.npz",
    )
    ap.add_argument(
        "--metadata-json",
        default="03_outputs/build_mixed_quantile_offset/K512_B512_C2_selective_rank_discrete_compact/mixed_quantile_offset_metadata.json",
    )
    ap.add_argument("--train-raw", default="01_split/train_raw.csv")
    ap.add_argument("--val-raw", default="01_split/val_raw.csv")
    ap.add_argument("--out-csv", default="05_test/outputs/B0_inputs/val_predictions_with_probs.csv")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    run_dir = resolve_existing(repo_root, args.run_dir, args.run_dir)
    best_model = run_dir / "best_model.pt"
    config_json = run_dir / "config.json"

    dataset_npz = resolve_existing(repo_root, args.dataset_npz, args.dataset_npz)
    metadata_json = resolve_existing(repo_root, args.metadata_json, args.metadata_json)
    train_raw = resolve_existing(repo_root, args.train_raw, "01_split/train_raw.csv")
    val_raw = resolve_existing(repo_root, args.val_raw, "01_split/val_raw.csv")

    if not best_model.exists():
        raise FileNotFoundError(f"best_model.pt not found: {best_model}")
    if not config_json.exists():
        raise FileNotFoundError(f"config.json not found: {config_json}")

    tr = import_train_module(repo_root)
    device = pick_device(args.device)

    with np.load(dataset_npz, allow_pickle=True) as z:
        data = {k: z[k] for k in z.files}
    meta = json.loads(metadata_json.read_text(encoding="utf-8"))

    X_train = data["X_train_bin"].astype(np.int64)
    O_train = data["X_train_offset"].astype(np.float32)
    y_train = data["y_train"].astype(np.int64)

    X_val = data["X_val_bin"].astype(np.int64)
    O_val = data["X_val_offset"].astype(np.float32)
    y_val = data["y_val"].astype(np.int64)

    # Official D3 continuous branch = raw_scaled.
    # Reimplement locally so this 05_test script does not depend on helper
    # functions being exposed by 02_src/07_train.py.
    feature_names = [str(x) for x in meta["feature_names"]]

    train_df = pd.read_csv(train_raw)
    val_df = pd.read_csv(val_raw)

    missing_train = [f for f in feature_names if f not in train_df.columns]
    missing_val = [f for f in feature_names if f not in val_df.columns]
    if missing_train:
        raise ValueError(f"train_raw missing features: {missing_train[:10]}")
    if missing_val:
        raise ValueError(f"val_raw missing features: {missing_val[:10]}")

    X_train_raw = train_df.loc[:, feature_names].to_numpy(dtype=np.float64)
    X_val_raw = val_df.loc[:, feature_names].to_numpy(dtype=np.float64)

    if np.isnan(X_train_raw).any() or np.isinf(X_train_raw).any():
        raise ValueError("train_raw contains NaN/Inf in selected features")
    if np.isnan(X_val_raw).any() or np.isinf(X_val_raw).any():
        raise ValueError("val_raw contains NaN/Inf in selected features")

    mn = X_train_raw.min(axis=0)
    mx = X_train_raw.max(axis=0)
    denom = mx - mn
    constant = np.isclose(denom, 0.0)
    denom_safe = denom.copy()
    denom_safe[constant] = 1.0

    X_train_cont = (X_train_raw - mn) / denom_safe
    X_val_cont = (X_val_raw - mn) / denom_safe
    X_train_cont[:, constant] = 0.5
    X_val_cont[:, constant] = 0.5

    X_train_cont = np.clip(X_train_cont, 0.0, 1.0).astype(np.float32)
    X_val_cont = np.clip(X_val_cont, 0.0, 1.0).astype(np.float32)

    if X_train_cont.shape != X_train.shape:
        raise ValueError(f"X_train_cont shape mismatch: {X_train_cont.shape} vs {X_train.shape}")
    if X_val_cont.shape != X_val.shape:
        raise ValueError(f"X_val_cont shape mismatch: {X_val_cont.shape} vs {X_val.shape}")

    continuous_info = {
        "source": "raw_scaled",
        "train_path": str(train_raw),
        "val_path": str(val_raw),
        "scale": "train_only_minmax_linear_clip_val",
        "n_constant_features": int(constant.sum()),
        "train_min": float(X_train_cont.min()),
        "train_max": float(X_train_cont.max()),
        "val_min": float(X_val_cont.min()),
        "val_max": float(X_val_cont.max()),
    }

    M_val = np.ones_like(X_val, dtype=np.float32)

    label_mapping = meta["label_mapping"]
    label_names = [label for label, _ in sorted(label_mapping.items(), key=lambda kv: kv[1])]
    num_classes = len(label_names)
    n_features = int(meta["n_features"])

    run_cfg = json.loads(config_json.read_text(encoding="utf-8"))

    # Config can be either {"model": {...}} or flat depending on exported run.
    model_cfg = run_cfg.get("model", run_cfg)
    get = model_cfg.get

    model = tr.FusionAblationTransformer(
        run_id="D3",
        num_bins=int(run_cfg.get("num_bins", run_cfg.get("effective_token_budget", meta.get("num_bins", 512)))),
        n_features=n_features,
        num_classes=num_classes,
        value_dim=int(get("value_dim", 32)),
        feature_dim=int(get("feature_dim", 32)),
        hidden_dim=int(get("hidden_dim", 128)),
        num_layers=int(get("num_layers", 3)),
        num_heads=int(get("num_heads", 4)),
        dropout=float(get("dropout", 0.1)),
        classifier_hidden_dim=int(get("classifier_hidden_dim", 128)),
        classifier_dropout=float(get("classifier_dropout", 0.1)),
        norm_first=bool(get("norm_first", True)),
        gate_init=float(get("gate_init", 0.0)),
    ).to(device)

    ckpt = torch.load(best_model, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    V_val = np.stack([
        O_val.astype(np.float32),
        X_val_cont.astype(np.float32),
        M_val.astype(np.float32),
    ], axis=-1)

    val_ds = torch.utils.data.TensorDataset(
        torch.as_tensor(X_val, dtype=torch.long),
        torch.as_tensor(V_val, dtype=torch.float32),
    )
    loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False)

    all_probs = []
    all_pred = []

    with torch.no_grad():
        for X, V in loader:
            X = X.to(device)
            V = V.to(device)
            logits = model(X, V)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1)
            all_probs.append(probs.detach().cpu().numpy())
            all_pred.append(pred.detach().cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    pred = np.concatenate(all_pred, axis=0)

    top_order = np.argsort(-probs, axis=1)
    top1 = top_order[:, 0]
    top2 = top_order[:, 1]

    out = pd.DataFrame({
        "sample_index": np.arange(len(y_val), dtype=int),
        "true_id": y_val.astype(int),
        "true_label": [label_names[i] for i in y_val],
        "pred_id": pred.astype(int),
        "pred_label": [label_names[i] for i in pred],
        "correct": (pred == y_val),
        "confidence": probs[np.arange(len(pred)), pred],
        "top1_label": [label_names[i] for i in top1],
        "top2_label": [label_names[i] for i in top2],
        "top1_score": probs[np.arange(len(pred)), top1],
        "top2_score": probs[np.arange(len(pred)), top2],
        "top12_margin": probs[np.arange(len(pred)), top1] - probs[np.arange(len(pred)), top2],
    })

    for i, name in enumerate(label_names):
        safe = str(name).strip().replace(" ", "_")
        out[f"prob_{safe}"] = probs[:, i]

    out_csv = repo_root / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    acc = float((pred == y_val).mean())
    print("===== exported official D3 val predictions with probs =====")
    print("run_dir:", run_dir)
    print("best_model:", best_model)
    print("dataset:", dataset_npz)
    print("metadata:", metadata_json)
    print("continuous_info:", continuous_info)
    print("n:", len(out))
    print("acc:", acc)
    print("out_csv:", out_csv)
    print("columns:", list(out.columns))


if __name__ == "__main__":
    main()
