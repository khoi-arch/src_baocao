#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F1f0 Official SAM/ASAM rho search using train_inner/calibration only.

This runner does NOT train on validation and does NOT evaluate validation.

It creates a temporary train_inner/calibration dataset from the original train
split, then calls the official trainer with SAM for each rho.

Selection:
    selected_rho = argmax calibration macro-F1

Next step after this search:
    train full original train with selected rho, evaluate validation once.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


CLASS_NAMES_DEFAULT = ["Benign", "Ransomware", "Spyware", "Trojan"]


def log(msg: str):
    print(f"[F1f0] {msg}", flush=True)


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


def parse_list(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def parse_float_grid(s: str) -> List[float]:
    return sorted(set(float(x.strip()) for x in str(s).split(",") if x.strip()))


def find_label_col(df: pd.DataFrame, level: str):
    if level == "L2":
        cands = ["label_L2", "Label_L2", "l2", "L2", "Category", "category"]
    else:
        cands = ["label_L3", "Label_L3", "l3", "L3", "Family", "family", "MalwareFamily", "malware_family"]
    for c in cands:
        if c in df.columns:
            return c
    return None


def load_v4_split(v4_dir: Path) -> Tuple[np.ndarray | None, np.ndarray | None, Dict[str, Any]]:
    p = v4_dir / "F1e1a_v4_split_indices.npz"
    if not p.exists():
        return None, None, {"source": "none", "reason": f"missing {p}"}
    z = np.load(p, allow_pickle=True)
    keys = set(z.files)
    train_key = None
    calib_key = None
    for k in ["train_inner_idx", "train_inner_indices", "inner_train_idx", "train_idx"]:
        if k in keys:
            train_key = k
            break
    for k in ["calibration_idx", "calibration_indices", "calib_idx", "calib_indices", "val_idx"]:
        if k in keys:
            calib_key = k
            break
    if train_key is None or calib_key is None:
        return None, None, {"source": str(p), "reason": f"unrecognized keys {list(z.files)}"}
    return np.asarray(z[train_key], dtype=np.int64), np.asarray(z[calib_key], dtype=np.int64), {
        "source": str(p),
        "train_key": train_key,
        "calib_key": calib_key,
    }


def make_split(train_raw: Path, y_train: np.ndarray, class_names: List[str], v4_dir: Path, calib_size: float, seed: int):
    train_idx, calib_idx, info = load_v4_split(v4_dir)
    if train_idx is not None and calib_idx is not None:
        info["n_train_inner"] = int(len(train_idx))
        info["n_calibration"] = int(len(calib_idx))
        info["method"] = "reuse_F1e1a_v4_split"
        return train_idx, calib_idx, info

    raw = pd.read_csv(train_raw)
    l2_col = find_label_col(raw, "L2")
    l3_col = find_label_col(raw, "L3")
    y_l2 = np.asarray([class_names[int(i)] for i in y_train], dtype=object)
    if l2_col:
        l2 = raw[l2_col].map(clean).to_numpy()
        if pd.Series(l2).isin(class_names).mean() < 0.80:
            l2 = y_l2
    else:
        l2 = y_l2
    if l3_col:
        l3 = raw[l3_col].map(clean).to_numpy()
        if (pd.Series(l3).map(clean) == "").mean() > 0.80:
            l3 = l2
    else:
        l3 = l2

    strat = np.asarray([f"{a}::{b}" for a, b in zip(l2, l3)], dtype=object)
    counts = pd.Series(strat).value_counts()
    if counts.min() < 2:
        strat = l2
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=float(calib_size), random_state=int(seed))
    train_idx, calib_idx = next(splitter.split(np.zeros(len(y_train)), strat))
    return train_idx.astype(np.int64), calib_idx.astype(np.int64), {
        "source": "fallback StratifiedShuffleSplit",
        "calib_size": float(calib_size),
        "seed": int(seed),
        "n_train_inner": int(len(train_idx)),
        "n_calibration": int(len(calib_idx)),
    }


def make_subset_npz_and_raw(dataset_npz: Path, train_raw: Path, out_dir: Path, train_idx: np.ndarray, calib_idx: np.ndarray):
    out_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(dataset_npz, allow_pickle=True)

    req = ["X_train_bin", "X_train_offset", "y_train"]
    missing = [k for k in req if k not in data.files]
    if missing:
        raise KeyError(f"dataset missing required train keys: {missing}")

    Xb = np.asarray(data["X_train_bin"])
    Xo = np.asarray(data["X_train_offset"])
    y = np.asarray(data["y_train"])

    subset_path = out_dir / "dataset_train_inner_calibration.npz"
    arrays = {
        "X_train_bin": Xb[train_idx],
        "X_train_offset": Xo[train_idx],
        "y_train": y[train_idx],
        "X_val_bin": Xb[calib_idx],
        "X_val_offset": Xo[calib_idx],
        "y_val": y[calib_idx],
    }
    np.savez_compressed(subset_path, **arrays)

    raw = pd.read_csv(train_raw)
    train_raw_out = out_dir / "train_inner_raw.csv"
    calib_raw_out = out_dir / "calibration_raw.csv"
    raw.iloc[train_idx].reset_index(drop=True).to_csv(train_raw_out, index=False)
    raw.iloc[calib_idx].reset_index(drop=True).to_csv(calib_raw_out, index=False)

    return subset_path, train_raw_out, calib_raw_out


def read_metrics(run_dir: Path, label_names: List[str]) -> Dict[str, Any]:
    train_rep = json.loads((run_dir / "train_classification_report_best.json").read_text(encoding="utf-8"))
    val_rep = json.loads((run_dir / "val_classification_report_best.json").read_text(encoding="utf-8"))
    hist = pd.read_csv(run_dir / "history.csv")
    best_epoch = int(hist.sort_values("val_macro_f1", ascending=False).iloc[0]["epoch"]) if "val_macro_f1" in hist.columns else -1

    def get_macro(rep):
        if "macro_avg" in rep:
            return float(rep["macro_avg"]["f1"])
        if "macro avg" in rep:
            return float(rep["macro avg"]["f1-score"])
        return float(rep.get("macro_f1", np.nan))

    def get_weighted(rep):
        if "weighted_avg" in rep:
            return float(rep["weighted_avg"]["f1"])
        if "weighted avg" in rep:
            return float(rep["weighted avg"]["f1-score"])
        return float(rep.get("weighted_f1", np.nan))

    def get_acc(rep):
        return float(rep.get("accuracy", np.nan))

    row = {
        "best_epoch": best_epoch,
        "train_macro_f1": get_macro(train_rep),
        "calibration_macro_f1": get_macro(val_rep),
        "train_weighted_f1": get_weighted(train_rep),
        "calibration_weighted_f1": get_weighted(val_rep),
        "train_accuracy": get_acc(train_rep),
        "calibration_accuracy": get_acc(val_rep),
    }
    row["gap_train_minus_calibration_macro_f1"] = row["train_macro_f1"] - row["calibration_macro_f1"]
    for cls in label_names:
        # support both formats
        if cls in val_rep:
            d = val_rep[cls]
            row[f"calib_f1_{cls}"] = float(d.get("f1", d.get("f1-score", np.nan)))
    return row


def zip_dir(src: Path, dst: Path):
    if dst.exists():
        dst.unlink()
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for p in src.rglob("*"):
            if p.is_file() and p != dst:
                z.write(p, p.relative_to(src.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-npz", default="03_outputs/05_dataset/dataset.npz")
    ap.add_argument("--metadata-json", default="03_outputs/05_dataset/metadata.json")
    ap.add_argument("--train-raw", default="01_split/train_raw.csv")
    ap.add_argument("--trainer", default="02_src/07_train_sam.py")
    ap.add_argument("--v4-dir", default="05_test/outputs/F1e1a_v4_clean_calibration_family_smoothing")
    ap.add_argument("--out-dir", default="05_test/outputs/F1f0_official_sam_rho_search")
    ap.add_argument("--combined-zip", default="05_test/outputs/F1f0_official_sam_rho_search.zip")
    ap.add_argument("--rho-grid", default="0.0,0.02,0.05,0.10")
    ap.add_argument("--class-names", default="Benign,Ransomware,Spyware,Trojan")
    ap.add_argument("--calib-size", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)

    # official trainer args
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--sam-adaptive", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--sam-eps", type=float, default=1e-12)
    args = ap.parse_args()

    root = repo_root_from_here()
    dataset_npz = resolve_path(args.dataset_npz, root)
    metadata_json = resolve_path(args.metadata_json, root)
    train_raw = resolve_path(args.train_raw, root)
    trainer = resolve_path(args.trainer, root)
    v4_dir = resolve_path(args.v4_dir, root)
    out_dir = resolve_path(args.out_dir, root)
    zip_path = resolve_path(args.combined_zip, root)

    out_dir.mkdir(parents=True, exist_ok=True)
    label_names = parse_list(args.class_names) or CLASS_NAMES_DEFAULT
    rhos = parse_float_grid(args.rho_grid)

    if not trainer.exists():
        raise FileNotFoundError(f"trainer not found: {trainer}")

    data = np.load(dataset_npz, allow_pickle=True)
    y_train = np.asarray(data["y_train"], dtype=np.int64)
    train_idx, calib_idx, split_info = make_split(train_raw, y_train, label_names, v4_dir, args.calib_size, args.seed)

    split_dir = out_dir / "_split_artifacts"
    subset_npz, train_inner_raw, calib_raw = make_subset_npz_and_raw(dataset_npz, train_raw, split_dir, train_idx, calib_idx)

    config = {
        "experiment": "F1f0_official_sam_rho_search",
        "validation_used": False,
        "selection_split": "calibration derived from original train",
        "rho_grid": rhos,
        "split_info": split_info,
        "dataset_subset_npz": str(subset_npz),
        "train_inner_raw": str(train_inner_raw),
        "calibration_raw": str(calib_raw),
        "trainer": str(trainer),
        "official_trainer_args": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "patience": int(args.patience),
            "sam_adaptive": bool(args.sam_adaptive),
        },
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    results = []
    runs_root = out_dir / "rho_runs"
    for rho in rhos:
        tag = str(rho).replace(".", "p")
        run_name = f"rho_{tag}"
        log(f"Running rho={rho} ({run_name})")

        cmd = [
            "python", str(trainer.relative_to(root) if trainer.is_relative_to(root) else trainer),
            "--run-id", "D3",
            "--K", "512",
            "--num-bins", "512",
            "--dataset-npz", str(subset_npz.relative_to(root) if subset_npz.is_relative_to(root) else subset_npz),
            "--metadata-json", str(metadata_json.relative_to(root) if metadata_json.is_relative_to(root) else metadata_json),
            "--train-raw", str(train_inner_raw.relative_to(root) if train_inner_raw.is_relative_to(root) else train_inner_raw),
            "--val-raw", str(calib_raw.relative_to(root) if calib_raw.is_relative_to(root) else calib_raw),
            "--out-root", str(runs_root.relative_to(root) if runs_root.is_relative_to(root) else runs_root),
            "--run-name", run_name,
            "--num-layers", "1",
            "--epochs", str(int(args.epochs)),
            "--batch-size", str(int(args.batch_size)),
            "--patience", str(int(args.patience)),
            "--device", str(args.device),
            "--num-workers", str(int(args.num_workers)),
            "--sam-rho", str(float(rho)),
            "--sam-eps", str(float(args.sam_eps)),
        ]
        if bool(args.sam_adaptive):
            cmd.append("--sam-adaptive")
        else:
            cmd.append("--no-sam-adaptive")

        subprocess.run(cmd, cwd=root, check=True)

        run_dir = runs_root / run_name
        row = read_metrics(run_dir, label_names)
        row["rho"] = float(rho)
        row["sam_adaptive"] = bool(args.sam_adaptive)
        row["run_dir"] = str(run_dir)
        results.append(row)
        pd.DataFrame(results).to_csv(out_dir / "F1f0_rho_results_partial.csv", index=False)
        log(f"rho={rho} calibration_macro_f1={row['calibration_macro_f1']:.6f} best_epoch={row['best_epoch']}")

    res = pd.DataFrame(results).sort_values(["calibration_macro_f1", "calibration_accuracy"], ascending=False).reset_index(drop=True)
    res.to_csv(out_dir / "F1f0_rho_results.csv", index=False)

    best = res.iloc[0].to_dict()
    (out_dir / "F1f0_best_rho.json").write_text(json.dumps({
        "selected_rho": float(best["rho"]),
        "selection_metric": "calibration_macro_f1",
        "validation_used": False,
        "selected_row": best,
        "rho_grid": rhos,
    }, indent=2, default=str), encoding="utf-8")

    lines = []
    lines.append("# F1f0 Official SAM rho search\n")
    lines.append("## Protocol\n")
    lines.append("```text")
    lines.append("Create train_inner/calibration from original train.")
    lines.append("For each rho, run official trainer with L1 and SAM.")
    lines.append("Select rho by calibration macro-F1.")
    lines.append("Validation is not used.")
    lines.append("```")
    lines.append("\n## Results\n")
    lines.append(res.to_markdown(index=False))
    lines.append("\n## Selected rho\n")
    lines.append("```json")
    lines.append(json.dumps({"selected_rho": float(best["rho"]), "selected_row": best}, indent=2, default=str))
    lines.append("```")
    lines.append("\n## Next\n")
    lines.append("```text")
    lines.append("If selected_rho > 0 and beats rho=0 on calibration, train full original train with selected rho and evaluate validation once.")
    lines.append("If rho=0 wins, reject SAM.")
    lines.append("```")
    (out_dir / "F1f0_report.md").write_text("\n".join(lines), encoding="utf-8")

    (out_dir / "F1f0_leakage_policy.md").write_text(
        "# F1f0 Leakage Policy\n\n"
        "F1f0 does not use the official validation split. It uses only a train_inner/calibration split created from the original training split. Rho is selected by calibration macro-F1.\n",
        encoding="utf-8",
    )

    zip_dir(out_dir, zip_path)

    log("Results:")
    print(res.to_string(index=False), flush=True)
    log(f"selected_rho={float(best['rho'])}")
    log(f"zip={zip_path}")
    log("DONE")


if __name__ == "__main__":
    main()
