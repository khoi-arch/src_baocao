#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F3d0 Adaptive hard-negative separation calibration search.

Purpose:
    Use audit insight without hardcoding pairs.

Loss:
    CE + alpha * sum_wrong softmax(logit_wrong/T) * softplus(logit_wrong - logit_true)

Properties:
    - no fixed hard-pair list
    - no L3/family labels for training
    - no fixed margin
    - per-sample hard-negative strength is adaptive from current logits
    - only small global alpha is searched against lambda=0 baseline on train_inner/calibration
"""

from __future__ import annotations

import argparse
import json
import subprocess
import zipfile
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


CLASS_NAMES_DEFAULT = ["Benign", "Ransomware", "Spyware", "Trojan"]


def log(msg: str):
    print(f"[F3d0] {msg}", flush=True)


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


def parse_float_grid(s: str) -> List[float]:
    vals = [float(x.strip()) for x in str(s).split(",") if x.strip()]
    out = []
    seen = set()
    for v in vals:
        key = round(v, 10)
        if key not in seen:
            seen.add(key)
            out.append(v)
    return out


def make_l2_split(y_train: np.ndarray, calib_size: float, seed: int):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=float(calib_size), random_state=int(seed))
    train_idx, calib_idx = next(splitter.split(np.zeros(len(y_train)), y_train))
    return train_idx.astype(np.int64), calib_idx.astype(np.int64), {
        "split_mode": "L2_only",
        "uses_l3_or_family_for_split": False,
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
    np.savez_compressed(
        subset_path,
        X_train_bin=Xb[train_idx],
        X_train_offset=Xo[train_idx],
        y_train=y[train_idx],
        X_val_bin=Xb[calib_idx],
        X_val_offset=Xo[calib_idx],
        y_val=y[calib_idx],
    )

    raw = pd.read_csv(train_raw)
    train_raw_out = out_dir / "train_inner_raw.csv"
    calib_raw_out = out_dir / "calibration_raw.csv"
    raw.iloc[train_idx].reset_index(drop=True).to_csv(train_raw_out, index=False)
    raw.iloc[calib_idx].reset_index(drop=True).to_csv(calib_raw_out, index=False)
    return subset_path, train_raw_out, calib_raw_out


def read_metrics(run_dir: Path, label_names: List[str]) -> Dict[str, Any]:
    hist = pd.read_csv(run_dir / "history.csv")
    if "val_macro_f1" in hist.columns:
        best_hist_row = hist.sort_values("val_macro_f1", ascending=False).iloc[0]
        best_epoch = int(best_hist_row["epoch"])
    else:
        best_epoch = -1
        best_hist_row = hist.iloc[-1]

    diag_path = run_dir / "diagnosis_summary.json"
    if diag_path.exists():
        diag = json.loads(diag_path.read_text(encoding="utf-8"))
        row = {
            "best_epoch": int(diag.get("best_epoch", best_epoch)),
            "train_macro_f1": float(diag.get("train_macro_f1", best_hist_row.get("train_macro_f1", np.nan))),
            "calibration_macro_f1": float(diag.get("val_macro_f1", best_hist_row.get("val_macro_f1", np.nan))),
            "train_weighted_f1": float(diag.get("train_weighted_f1", best_hist_row.get("train_weighted_f1", np.nan))),
            "calibration_weighted_f1": float(diag.get("val_weighted_f1", best_hist_row.get("val_weighted_f1", np.nan))),
            "train_accuracy": float(diag.get("train_accuracy", best_hist_row.get("train_accuracy", np.nan))),
            "calibration_accuracy": float(diag.get("val_accuracy", best_hist_row.get("val_accuracy", np.nan))),
            "train_loss": float(diag.get("train_loss", best_hist_row.get("train_loss", np.nan))),
            "calibration_loss": float(diag.get("val_loss", best_hist_row.get("val_loss", np.nan))),
        }
    else:
        row = {
            "best_epoch": best_epoch,
            "train_macro_f1": float(best_hist_row.get("train_macro_f1", np.nan)),
            "calibration_macro_f1": float(best_hist_row.get("val_macro_f1", np.nan)),
            "train_weighted_f1": float(best_hist_row.get("train_weighted_f1", np.nan)),
            "calibration_weighted_f1": float(best_hist_row.get("val_weighted_f1", np.nan)),
            "train_accuracy": float(best_hist_row.get("train_accuracy", np.nan)),
            "calibration_accuracy": float(best_hist_row.get("val_accuracy", np.nan)),
            "train_loss": float(best_hist_row.get("train_loss", np.nan)),
            "calibration_loss": float(best_hist_row.get("val_loss", np.nan)),
        }

    val_report = run_dir / "val_classification_report_best.json"
    if val_report.exists():
        rep = json.loads(val_report.read_text(encoding="utf-8"))
        for cls in label_names:
            if cls in rep:
                d = rep[cls]
                row[f"calib_f1_{cls}"] = float(d.get("f1", d.get("f1-score", np.nan)))
                row[f"calib_precision_{cls}"] = float(d.get("precision", np.nan))
                row[f"calib_recall_{cls}"] = float(d.get("recall", np.nan))

    row["gap_train_minus_calibration_macro_f1"] = row["train_macro_f1"] - row["calibration_macro_f1"]

    info_path = run_dir / "adaptive_hardneg_info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        row["ahn_enabled"] = bool(info.get("enabled", False))
        row["ahn_temperature"] = float(info.get("temperature", np.nan))
        row["ahn_scope"] = str(info.get("scope", ""))
        row["ahn_detach_weights"] = bool(info.get("detach_weights", True))

    return row


def find_completed_run(runs_root: Path, run_name: str) -> Path | None:
    candidate_dirs = [runs_root / "Keff512" / run_name, runs_root / run_name]
    candidate_dirs += [p for p in runs_root.rglob(run_name) if p.is_dir()]
    seen = set()
    for cand in candidate_dirs:
        if cand in seen:
            continue
        seen.add(cand)
        if (cand / "history.csv").exists() and (cand / "diagnosis_summary.json").exists():
            return cand
    return None


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
    ap.add_argument("--trainer", default="02_src/07_train_adaptive_hardneg.py")
    ap.add_argument("--out-dir", default="05_test/outputs/F3d0_adaptive_hardneg_calibration_search")
    ap.add_argument("--combined-zip", default="05_test/outputs/F3d0_adaptive_hardneg_calibration_search.zip")
    ap.add_argument("--alpha-grid", default="0.0,0.05,0.1")
    ap.add_argument("--ahn-temperature", type=float, default=1.0)
    ap.add_argument("--ahn-scope", default="all", choices=["all", "malware"])
    ap.add_argument("--class-names", default="Benign,Ransomware,Spyware,Trojan")
    ap.add_argument("--calib-size", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    root = repo_root_from_here()
    dataset_npz = resolve_path(args.dataset_npz, root)
    metadata_json = resolve_path(args.metadata_json, root)
    train_raw = resolve_path(args.train_raw, root)
    trainer = resolve_path(args.trainer, root)
    out_dir = resolve_path(args.out_dir, root)
    zip_path = resolve_path(args.combined_zip, root)

    out_dir.mkdir(parents=True, exist_ok=True)
    label_names = parse_list(args.class_names) or CLASS_NAMES_DEFAULT
    alphas = parse_float_grid(args.alpha_grid)

    if not trainer.exists():
        raise FileNotFoundError(f"trainer not found: {trainer}")

    data = np.load(dataset_npz, allow_pickle=True)
    y_train = np.asarray(data["y_train"], dtype=np.int64)
    train_idx, calib_idx, split_info = make_l2_split(y_train, args.calib_size, args.seed)
    subset_npz, train_inner_raw, calib_raw = make_subset_npz_and_raw(dataset_npz, train_raw, out_dir / "_split_artifacts", train_idx, calib_idx)

    config = {
        "experiment": "F3d0_adaptive_hard_negative_separation_calibration_search",
        "official_validation_used": False,
        "uses_fixed_hard_pairs": False,
        "uses_l3_or_family_labels_for_training": False,
        "uses_l3_or_family_for_split": False,
        "has_fixed_margin": False,
        "method": "CE + alpha * sum_wrong softmax(logit_wrong/T) * softplus(logit_wrong - logit_true)",
        "selection_source": "train_inner/calibration only",
        "audit_role": "F2/F2b/F3a2 justify hard-negative phenomenon only; they do not provide rules for this loss.",
        "alphas": alphas,
        "ahn_temperature": float(args.ahn_temperature),
        "ahn_scope": str(args.ahn_scope),
        "split_info": split_info,
        "dataset_subset_npz": str(subset_npz),
        "train_inner_raw": str(train_inner_raw),
        "calibration_raw": str(calib_raw),
        "trainer": str(trainer),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    results = []
    runs_root = out_dir / "config_runs"
    for alpha in alphas:
        run_name = f"alpha_{str(alpha).replace('.', 'p')}"
        log(f"Running alpha={alpha} ({run_name})")

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
            "--sam-rho", "0.0",
            "--ahn-alpha", str(float(alpha)),
            "--ahn-temperature", str(float(args.ahn_temperature)),
            "--ahn-scope", str(args.ahn_scope),
            "--ahn-detach-weights",
        ]

        run_dir = find_completed_run(runs_root, run_name)
        if run_dir is not None:
            log(f"Reusing completed run: {run_dir}")
        else:
            subprocess.run(cmd, cwd=root, check=True)
            run_dir = find_completed_run(runs_root, run_name)

        if run_dir is None:
            raise FileNotFoundError(f"Cannot locate completed run for {run_name}")

        row = read_metrics(run_dir, label_names)
        row["alpha"] = float(alpha)
        row["run_dir"] = str(run_dir)
        results.append(row)
        pd.DataFrame(results).to_csv(out_dir / "F3d0_results_partial.csv", index=False)
        log(f"alpha={alpha} calib_macro_f1={row['calibration_macro_f1']:.6f}")

    res = pd.DataFrame(results).sort_values(["calibration_macro_f1", "calibration_accuracy"], ascending=False).reset_index(drop=True)
    res.to_csv(out_dir / "F3d0_results.csv", index=False)

    best = res.iloc[0].to_dict()
    baseline = res[(res["alpha"] == 0.0)].iloc[0].to_dict() if (res["alpha"] == 0.0).any() else {}
    delta = float(best["calibration_macro_f1"] - baseline.get("calibration_macro_f1", np.nan)) if baseline else np.nan

    decision = {
        "selected_alpha": float(best["alpha"]),
        "selected_temperature": float(args.ahn_temperature),
        "selected_scope": str(args.ahn_scope),
        "selection_metric": "calibration_macro_f1",
        "official_validation_used": False,
        "uses_fixed_hard_pairs": False,
        "uses_l3_or_family_labels_for_training": False,
        "has_fixed_margin": False,
        "selected_row": best,
        "baseline_row": baseline,
        "delta_vs_alpha0_calibration_macro_f1": delta,
        "recommendation": "run_full_train_official_val_once" if float(best["alpha"]) > 0 and delta > 0 else "reject_adaptive_hardneg_or_rethink",
    }
    (out_dir / "F3d0_best_config.json").write_text(json.dumps(decision, indent=2, default=str), encoding="utf-8")

    lines = []
    lines.append("# F3d0 Adaptive hard-negative separation calibration search\n")
    lines.append("## Protocol\n")
    lines.append("```text")
    lines.append("No fixed hard-pair list.")
    lines.append("No L3/family label used for training.")
    lines.append("No L3/family label used for split.")
    lines.append("No fixed margin.")
    lines.append("For each sample, wrong-class weights are computed from current logits.")
    lines.append("Loss = CE + alpha * sum_wrong softmax(logit_wrong/T) * softplus(logit_wrong - logit_true).")
    lines.append("```")
    lines.append("\n## Results\n")
    lines.append(res.to_markdown(index=False))
    lines.append("\n## Decision\n")
    lines.append("```json")
    lines.append(json.dumps(decision, indent=2, default=str))
    lines.append("```")
    lines.append("\n## Interpretation\n")
    lines.append("```text")
    lines.append("If alpha>0 wins calibration, this supports a general adaptive hard-negative boundary objective.")
    lines.append("If alpha=0 wins, this adaptive loss did not help under the clean calibration protocol.")
    lines.append("```")
    (out_dir / "F3d0_report.md").write_text("\n".join(lines), encoding="utf-8")

    zip_dir(out_dir, zip_path)
    log("Results:")
    print(res.to_string(index=False), flush=True)
    log(f"decision={decision}")
    log(f"zip={zip_path}")
    log("DONE")


if __name__ == "__main__":
    main()
