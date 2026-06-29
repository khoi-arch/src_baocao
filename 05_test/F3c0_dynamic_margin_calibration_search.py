#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F3c0 Dynamic hard-negative margin calibration search.

Purpose:
    Test a non-hardcoded objective based on the audit finding:
      hard negatives exist and CLS/logit boundary amplifies them.

    But during training, no fixed hard-pair list is used.
    For each sample, the hard negative is selected dynamically from current logits:
      max_wrong_logit = max_{c != y} logit_c

Loss:
    CE + lambda * ReLU(margin + max_wrong_logit - true_logit)

No official validation is used for selection.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import zipfile
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


CLASS_NAMES_DEFAULT = ["Benign", "Ransomware", "Spyware", "Trojan"]


def log(msg: str):
    print(f"[F3c0] {msg}", flush=True)


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
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def parse_config_grid(lambda_grid: str, margin_grid: str) -> List[Tuple[float, float]]:
    lambdas = parse_float_grid(lambda_grid)
    margins = parse_float_grid(margin_grid)
    cfgs = []
    if 0.0 in lambdas:
        cfgs.append((0.0, 0.0))
    for lam in lambdas:
        if lam <= 0:
            continue
        for m in margins:
            cfgs.append((float(lam), float(m)))
    out = []
    seen = set()
    for x in cfgs:
        key = (round(x[0], 10), round(x[1], 10))
        if key not in seen:
            seen.add(key)
            out.append(x)
    return out


def find_label_col(df: pd.DataFrame, level: str):
    if level == "L2":
        cands = ["label_L2", "Label_L2", "l2", "L2", "Category", "category", "Class", "class"]
    else:
        cands = ["label_L3", "Label_L3", "l3", "L3", "Family", "family", "MalwareFamily", "malware_family"]
    for c in cands:
        if c in df.columns:
            return c
    return None


def make_split(train_raw: Path, y_train: np.ndarray, class_names: List[str], calib_size: float, seed: int):
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

    # L3 is used only for stratifying calibration split, not for training signal.
    # This is okay as split bookkeeping, but not passed to the model/loss.
    if l3_col:
        l3 = raw[l3_col].map(clean).to_numpy()
        if (pd.Series(l3).map(clean) == "").mean() > 0.80:
            l3 = l2
    else:
        l3 = l2

    strat = np.asarray([f"{a}::{b}" for a, b in zip(l2, l3)], dtype=object)
    counts = pd.Series(strat).value_counts()
    mode = "L2_plus_L3_for_split_balance_only"
    if counts.min() < 2:
        strat = l2
        mode = "L2_only_due_to_rare_L3"

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=float(calib_size), random_state=int(seed))
    train_idx, calib_idx = next(splitter.split(np.zeros(len(y_train)), strat))
    return train_idx.astype(np.int64), calib_idx.astype(np.int64), {
        "split_mode": mode,
        "calib_size": float(calib_size),
        "seed": int(seed),
        "n_train_inner": int(len(train_idx)),
        "n_calibration": int(len(calib_idx)),
        "note": "L3 may be used only for balanced split stratification; dynamic margin training uses no L3/family label.",
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

    dyn_info_path = run_dir / "dynamic_margin_info.json"
    if dyn_info_path.exists():
        info = json.loads(dyn_info_path.read_text(encoding="utf-8"))
        row["dyn_margin_enabled"] = bool(info.get("enabled", False))
        row["dyn_margin_scope"] = str(info.get("scope", ""))
        row["dyn_margin_reduction"] = str(info.get("reduction", ""))

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
    ap.add_argument("--trainer", default="02_src/07_train_dynmargin.py")
    ap.add_argument("--out-dir", default="05_test/outputs/F3c0_dynamic_margin_calibration_search")
    ap.add_argument("--combined-zip", default="05_test/outputs/F3c0_dynamic_margin_calibration_search.zip")
    ap.add_argument("--lambda-grid", default="0.0,0.05,0.1,0.2")
    ap.add_argument("--margin-grid", default="0.1,0.2,0.3")
    ap.add_argument("--dyn-margin-scope", default="all", choices=["all", "malware"])
    ap.add_argument("--dyn-margin-reduction", default="mean_all", choices=["mean_all", "mean_active"])
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
    cfgs = parse_config_grid(args.lambda_grid, args.margin_grid)

    if not trainer.exists():
        raise FileNotFoundError(f"trainer not found: {trainer}")

    data = np.load(dataset_npz, allow_pickle=True)
    y_train = np.asarray(data["y_train"], dtype=np.int64)
    train_idx, calib_idx, split_info = make_split(train_raw, y_train, label_names, args.calib_size, args.seed)
    subset_npz, train_inner_raw, calib_raw = make_subset_npz_and_raw(dataset_npz, train_raw, out_dir / "_split_artifacts", train_idx, calib_idx)

    config = {
        "experiment": "F3c0_dynamic_hard_negative_margin_calibration_search",
        "official_validation_used": False,
        "uses_fixed_hard_pairs": False,
        "uses_l3_or_family_labels_for_training": False,
        "method": "CE + lambda * ReLU(margin + max_wrong_logit - true_logit)",
        "selection_source": "train_inner/calibration only",
        "audit_role": "F2/F2b/F3a2 justify hard-negative phenomenon; they do not provide fixed pair rules for this loss.",
        "configs": [{"lambda": l, "margin": m} for l, m in cfgs],
        "dyn_margin_scope": str(args.dyn_margin_scope),
        "dyn_margin_reduction": str(args.dyn_margin_reduction),
        "split_info": split_info,
        "dataset_subset_npz": str(subset_npz),
        "train_inner_raw": str(train_inner_raw),
        "calibration_raw": str(calib_raw),
        "trainer": str(trainer),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    results = []
    runs_root = out_dir / "config_runs"
    for lam, margin in cfgs:
        run_name = f"lambda_{str(lam).replace('.', 'p')}_margin_{str(margin).replace('.', 'p')}"
        log(f"Running lambda={lam}, margin={margin} ({run_name})")

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
            "--dyn-margin-lambda", str(float(lam)),
            "--dyn-margin-margin", str(float(margin)),
            "--dyn-margin-scope", str(args.dyn_margin_scope),
            "--dyn-margin-reduction", str(args.dyn_margin_reduction),
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
        row["lambda"] = float(lam)
        row["margin"] = float(margin)
        row["run_dir"] = str(run_dir)
        results.append(row)
        pd.DataFrame(results).to_csv(out_dir / "F3c0_results_partial.csv", index=False)
        log(f"lambda={lam} margin={margin} calib_macro_f1={row['calibration_macro_f1']:.6f}")

    res = pd.DataFrame(results).sort_values(["calibration_macro_f1", "calibration_accuracy"], ascending=False).reset_index(drop=True)
    res.to_csv(out_dir / "F3c0_results.csv", index=False)

    best = res.iloc[0].to_dict()
    baseline = res[(res["lambda"] == 0.0)].iloc[0].to_dict() if (res["lambda"] == 0.0).any() else {}
    delta = float(best["calibration_macro_f1"] - baseline.get("calibration_macro_f1", np.nan)) if baseline else np.nan
    decision = {
        "selected_lambda": float(best["lambda"]),
        "selected_margin": float(best["margin"]),
        "selected_scope": str(args.dyn_margin_scope),
        "selected_reduction": str(args.dyn_margin_reduction),
        "selection_metric": "calibration_macro_f1",
        "official_validation_used": False,
        "uses_fixed_hard_pairs": False,
        "uses_l3_or_family_labels_for_training": False,
        "selected_row": best,
        "baseline_row": baseline,
        "delta_vs_lambda0_calibration_macro_f1": delta,
        "recommendation": "proceed_to_F3c1_full_train" if float(best["lambda"]) > 0 and delta > 0 else "reject_dynamic_margin_or_rethink",
    }
    (out_dir / "F3c0_best_config.json").write_text(json.dumps(decision, indent=2, default=str), encoding="utf-8")

    lines = []
    lines.append("# F3c0 Dynamic hard-negative margin calibration search\n")
    lines.append("## Protocol\n")
    lines.append("```text")
    lines.append("No fixed hard-pair list.")
    lines.append("No L3/family label used for training.")
    lines.append("No official validation used for selection.")
    lines.append("For each sample, hard negative = current max wrong logit.")
    lines.append("Loss = CE + lambda * ReLU(margin + max_wrong_logit - true_logit).")
    lines.append("```")
    lines.append("\n## Results\n")
    lines.append(res.to_markdown(index=False))
    lines.append("\n## Decision\n")
    lines.append("```json")
    lines.append(json.dumps(decision, indent=2, default=str))
    lines.append("```")
    lines.append("\n## Interpretation\n")
    lines.append("```text")
    lines.append("If lambda>0 wins calibration, this supports a general hard-negative boundary objective.")
    lines.append("If lambda=0 wins, dynamic margin did not help under this calibration protocol.")
    lines.append("```")
    (out_dir / "F3c0_report.md").write_text("\n".join(lines), encoding="utf-8")

    zip_dir(out_dir, zip_path)
    log("Results:")
    print(res.to_string(index=False), flush=True)
    log(f"decision={decision}")
    log(f"zip={zip_path}")
    log("DONE")


if __name__ == "__main__":
    main()
