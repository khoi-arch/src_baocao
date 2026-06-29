#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F4a1 teacher-protected correction full official validation.

Run only if F4a0 calibration wins.

Protocol:
  1) Train baseline CE teacher on full train, early-stop on official val.
  2) Train F4a student on full train, same official val, using teacher checkpoint.
  3) Compare baseline vs student on official val.

This uses official val once for final verification, not for tuning multiple configs.
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
    print(f"[F4a] {msg}", flush=True)


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

    # L3 is only for stratified split balance, not training signal.
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
        "note": "L3 only for split balance; not used by F4a training loss.",
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


def read_metrics(run_dir: Path, label_names: List[str], val_prefix: str = "val") -> Dict[str, Any]:
    hist = pd.read_csv(run_dir / "history.csv")
    best_hist_row = hist.sort_values("val_macro_f1", ascending=False).iloc[0]
    best_epoch = int(best_hist_row["epoch"])

    diag_path = run_dir / "diagnosis_summary.json"
    if diag_path.exists():
        diag = json.loads(diag_path.read_text(encoding="utf-8"))
        row = {
            "best_epoch": int(diag.get("best_epoch", best_epoch)),
            "train_macro_f1": float(diag.get("train_macro_f1", best_hist_row.get("train_macro_f1", np.nan))),
            f"{val_prefix}_macro_f1": float(diag.get("val_macro_f1", best_hist_row.get("val_macro_f1", np.nan))),
            "train_weighted_f1": float(diag.get("train_weighted_f1", best_hist_row.get("train_weighted_f1", np.nan))),
            f"{val_prefix}_weighted_f1": float(diag.get("val_weighted_f1", best_hist_row.get("val_weighted_f1", np.nan))),
            "train_accuracy": float(diag.get("train_accuracy", best_hist_row.get("train_accuracy", np.nan))),
            f"{val_prefix}_accuracy": float(diag.get("val_accuracy", best_hist_row.get("val_accuracy", np.nan))),
            "train_loss": float(diag.get("train_loss", best_hist_row.get("train_loss", np.nan))),
            f"{val_prefix}_loss": float(diag.get("val_loss", best_hist_row.get("val_loss", np.nan))),
        }
    else:
        row = {
            "best_epoch": best_epoch,
            "train_macro_f1": float(best_hist_row.get("train_macro_f1", np.nan)),
            f"{val_prefix}_macro_f1": float(best_hist_row.get("val_macro_f1", np.nan)),
            "train_weighted_f1": float(best_hist_row.get("train_weighted_f1", np.nan)),
            f"{val_prefix}_weighted_f1": float(best_hist_row.get("val_weighted_f1", np.nan)),
            "train_accuracy": float(best_hist_row.get("train_accuracy", np.nan)),
            f"{val_prefix}_accuracy": float(best_hist_row.get("val_accuracy", np.nan)),
            "train_loss": float(best_hist_row.get("train_loss", np.nan)),
            f"{val_prefix}_loss": float(best_hist_row.get("val_loss", np.nan)),
        }

    val_report = run_dir / "val_classification_report_best.json"
    if val_report.exists():
        rep = json.loads(val_report.read_text(encoding="utf-8"))
        for cls in label_names:
            if cls in rep:
                d = rep[cls]
                row[f"{val_prefix}_f1_{cls}"] = float(d.get("f1", d.get("f1-score", np.nan)))
                row[f"{val_prefix}_precision_{cls}"] = float(d.get("precision", np.nan))
                row[f"{val_prefix}_recall_{cls}"] = float(d.get("recall", np.nan))
    row[f"gap_train_minus_{val_prefix}_macro_f1"] = row["train_macro_f1"] - row[f"{val_prefix}_macro_f1"]
    return row


def find_completed_run(runs_root: Path, run_name: str) -> Path | None:
    candidate_dirs = [runs_root / "Keff512" / run_name, runs_root / run_name]
    candidate_dirs += [p for p in runs_root.rglob(run_name) if p.is_dir()]
    seen = set()
    for cand in candidate_dirs:
        if cand in seen:
            continue
        seen.add(cand)
        if (cand / "history.csv").exists() and (cand / "diagnosis_summary.json").exists() and (cand / "best_model.pt").exists():
            return cand
    return None


def run_train(
    *,
    root: Path,
    trainer: Path,
    runs_root: Path,
    run_name: str,
    dataset_npz: Path,
    metadata_json: Path,
    train_raw: Path,
    val_raw: Path,
    args,
    tpc_enabled: bool,
    teacher_ckpt: Path | None = None,
):
    cmd = [
        "python", str(trainer.relative_to(root) if trainer.is_relative_to(root) else trainer),
        "--run-id", "D3",
        "--K", "512",
        "--num-bins", "512",
        "--dataset-npz", str(dataset_npz.relative_to(root) if dataset_npz.is_relative_to(root) else dataset_npz),
        "--metadata-json", str(metadata_json.relative_to(root) if metadata_json.is_relative_to(root) else metadata_json),
        "--train-raw", str(train_raw.relative_to(root) if train_raw.is_relative_to(root) else train_raw),
        "--val-raw", str(val_raw.relative_to(root) if val_raw.is_relative_to(root) else val_raw),
        "--out-root", str(runs_root.relative_to(root) if runs_root.is_relative_to(root) else runs_root),
        "--run-name", run_name,
        "--num-layers", "1",
        "--epochs", str(int(args.epochs)),
        "--batch-size", str(int(args.batch_size)),
        "--patience", str(int(args.patience)),
        "--device", str(args.device),
        "--num-workers", str(int(args.num_workers)),
        "--sam-rho", "0.0",
        "--tpc-temperature", str(float(args.tpc_temperature)),
    ]
    if tpc_enabled:
        if teacher_ckpt is None:
            raise ValueError("teacher_ckpt required for tpc")
        cmd += ["--tpc-enabled", "--tpc-teacher-checkpoint", str(teacher_ckpt)]
    else:
        cmd += ["--no-tpc-enabled"]

    run_dir = find_completed_run(runs_root, run_name)
    if run_dir is not None:
        log(f"Reusing completed run: {run_dir}")
        return run_dir
    subprocess.run(cmd, cwd=root, check=True)
    run_dir = find_completed_run(runs_root, run_name)
    if run_dir is None:
        raise FileNotFoundError(f"Cannot find completed run: {run_name}")
    return run_dir


def read_pred_csv(run_dir: Path) -> pd.DataFrame:
    p = run_dir / "val_predictions_best.csv"
    if not p.exists():
        raise FileNotFoundError(f"missing predictions: {p}")
    df = pd.read_csv(p)
    if "true_id" in df.columns:
        df = df.rename(columns={"true_id": "y_true"})
    if "pred_id" in df.columns:
        df = df.rename(columns={"pred_id": "y_pred"})
    if "sample_index" not in df.columns:
        df["sample_index"] = np.arange(len(df))
    if "y_true" not in df.columns or "y_pred" not in df.columns:
        raise ValueError(f"Cannot parse prediction columns from {p}: {df.columns.tolist()}")
    return df.sort_values("sample_index").reset_index(drop=True)


def compute_fix_damage(base_dir: Path, method_dir: Path, label_names: List[str], out_dir: Path, prefix: str) -> Dict[str, Any]:
    base = read_pred_csv(base_dir)
    method = read_pred_csv(method_dir)
    if len(base) != len(method):
        raise ValueError(f"prediction length mismatch: baseline={len(base)} method={len(method)}")
    if not np.array_equal(base["sample_index"].to_numpy(), method["sample_index"].to_numpy()):
        raise ValueError("sample_index mismatch between baseline and method")
    if not np.array_equal(base["y_true"].to_numpy(), method["y_true"].to_numpy()):
        raise ValueError("y_true mismatch between baseline and method")

    y = base["y_true"].to_numpy()
    base_pred = base["y_pred"].to_numpy()
    method_pred = method["y_pred"].to_numpy()
    base_correct = base_pred == y
    method_correct = method_pred == y
    id_to_label = {i: label_names[i] for i in range(len(label_names))}

    comp = pd.DataFrame({
        "sample_index": base["sample_index"],
        "y_true": y,
        "true_label": [id_to_label.get(int(i), str(i)) for i in y],
        "base_pred": base_pred,
        "base_pred_label": [id_to_label.get(int(i), str(i)) for i in base_pred],
        "method_pred": method_pred,
        "method_pred_label": [id_to_label.get(int(i), str(i)) for i in method_pred],
        "base_correct": base_correct,
        "method_correct": method_correct,
    })
    conditions = [
        base_correct & method_correct,
        (~base_correct) & method_correct,
        base_correct & (~method_correct),
        (~base_correct) & (~method_correct) & (base_pred == method_pred),
        (~base_correct) & (~method_correct) & (base_pred != method_pred),
    ]
    choices = ["both_correct", "fixed", "damaged", "both_wrong_same", "both_wrong_changed"]
    comp["switch_type"] = np.select(conditions, choices, default="unknown")
    comp.to_csv(out_dir / f"{prefix}_fix_damage_rows.csv", index=False)

    switch_counts = comp["switch_type"].value_counts().rename_axis("switch_type").reset_index(name="count")
    switch_counts.to_csv(out_dir / f"{prefix}_fix_damage_counts.csv", index=False)

    by_class = comp.groupby(["true_label", "switch_type"]).agg(count=("sample_index", "count")).reset_index()
    by_class.to_csv(out_dir / f"{prefix}_fix_damage_by_true_class.csv", index=False)

    transitions = comp.groupby(["true_label", "base_pred_label", "method_pred_label"]).agg(count=("sample_index", "count")).reset_index().sort_values("count", ascending=False)
    transitions.to_csv(out_dir / f"{prefix}_prediction_transition_summary.csv", index=False)

    per_class_net = []
    for cls in label_names:
        sub = comp[comp["true_label"] == cls]
        per_class_net.append({
            "true_label": cls,
            "support": int(len(sub)),
            "base_acc": float(sub["base_correct"].mean()) if len(sub) else float("nan"),
            "method_acc": float(sub["method_correct"].mean()) if len(sub) else float("nan"),
            "delta_acc": float(sub["method_correct"].mean() - sub["base_correct"].mean()) if len(sub) else float("nan"),
            "fixed": int((sub["switch_type"] == "fixed").sum()),
            "damaged": int((sub["switch_type"] == "damaged").sum()),
            "net": int((sub["switch_type"] == "fixed").sum() - (sub["switch_type"] == "damaged").sum()),
        })
    pd.DataFrame(per_class_net).to_csv(out_dir / f"{prefix}_per_class_fix_damage_net.csv", index=False)

    def cnt(name: str) -> int:
        return int((comp["switch_type"] == name).sum())

    summary = {
        "fixed": cnt("fixed"),
        "damaged": cnt("damaged"),
        "net_fixed_minus_damaged": cnt("fixed") - cnt("damaged"),
        "both_correct": cnt("both_correct"),
        "both_wrong_same": cnt("both_wrong_same"),
        "both_wrong_changed": cnt("both_wrong_changed"),
        "damage_over_fix": float(cnt("damaged") / max(cnt("fixed"), 1)),
        "per_class_net": per_class_net,
    }
    (out_dir / f"{prefix}_fix_damage_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


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
    ap.add_argument("--val-raw", default="01_split/val_raw.csv")
    ap.add_argument("--trainer", default="02_src/07_train_teacher_protected_correction.py")
    ap.add_argument("--out-dir", default="05_test/outputs/F4a1_teacher_protected_correction_full_val")
    ap.add_argument("--combined-zip", default="05_test/outputs/F4a1_teacher_protected_correction_full_val.zip")
    ap.add_argument("--class-names", default="Benign,Ransomware,Spyware,Trojan")
    ap.add_argument("--tpc-temperature", type=float, default=1.0)
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
    val_raw = resolve_path(args.val_raw, root)
    trainer = resolve_path(args.trainer, root)
    out_dir = resolve_path(args.out_dir, root)
    zip_path = resolve_path(args.combined_zip, root)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_names = parse_list(args.class_names) or CLASS_NAMES_DEFAULT

    config = {
        "experiment": "F4a1_teacher_protected_correction_full_official_val",
        "official_validation_used": True,
        "note": "Run only after F4a0 calibration wins. This is final verification, not grid tuning.",
        "uses_fixed_hard_pairs": False,
        "uses_l3_or_family_labels_for_training": False,
        "method": "Train baseline full teacher; student CE + teacher-wrong zero-violation correction + teacher-correct KL protection.",
        "tpc_temperature": float(args.tpc_temperature),
        "dataset_npz": str(dataset_npz),
        "train_raw": str(train_raw),
        "val_raw": str(val_raw),
        "trainer": str(trainer),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    runs_root = out_dir / "config_runs"
    teacher_dir = run_train(
        root=root, trainer=trainer, runs_root=runs_root, run_name="baseline_ce_teacher_full",
        dataset_npz=dataset_npz, metadata_json=metadata_json, train_raw=train_raw, val_raw=val_raw,
        args=args, tpc_enabled=False,
    )
    teacher_ckpt = teacher_dir / "best_model.pt"
    student_dir = run_train(
        root=root, trainer=trainer, runs_root=runs_root, run_name="teacher_protected_correction_student_full",
        dataset_npz=dataset_npz, metadata_json=metadata_json, train_raw=train_raw, val_raw=val_raw,
        args=args, tpc_enabled=True, teacher_ckpt=teacher_ckpt,
    )

    rows = []
    for run_name, run_dir in [("baseline_ce_teacher_full", teacher_dir), ("teacher_protected_correction_student_full", student_dir)]:
        row = read_metrics(run_dir, label_names, val_prefix="val")
        row["run_name"] = run_name
        row["run_dir"] = str(run_dir)
        rows.append(row)
    res = pd.DataFrame(rows).sort_values(["val_macro_f1", "val_accuracy"], ascending=False).reset_index(drop=True)
    res.to_csv(out_dir / "F4a1_results.csv", index=False)

    base_row = res[res["run_name"] == "baseline_ce_teacher_full"].iloc[0].to_dict()
    method_row = res[res["run_name"] == "teacher_protected_correction_student_full"].iloc[0].to_dict()
    delta = float(method_row["val_macro_f1"] - base_row["val_macro_f1"])

    fix_damage = compute_fix_damage(teacher_dir, student_dir, label_names, out_dir, "F4a1")
    accept = (
        delta > 0
        and int(fix_damage.get("net_fixed_minus_damaged", -999999)) > 0
        and float(fix_damage.get("damage_over_fix", 999999.0)) <= 1.0
    )
    decision = {
        "official_validation_used": True,
        "baseline_row": base_row,
        "teacher_protected_row": method_row,
        "delta_method_minus_baseline_val_macro_f1": delta,
        "fix_damage_summary": fix_damage,
        "accept_as_final_pipeline": bool(accept),
        "recommendation": "accept_F4a_as_final" if accept else "reject_F4a_and_keep_L1_batch512_final",
    }
    (out_dir / "F4a1_decision.json").write_text(json.dumps(decision, indent=2, default=str), encoding="utf-8")

    lines = []
    lines.append("# F4a1 teacher-protected correction full official validation\n")
    lines.append("## Results\n")
    lines.append(res.to_markdown(index=False))
    lines.append("\n## Fix/damage summary\n")
    lines.append("```json\n" + json.dumps(fix_damage, indent=2, default=str) + "\n```")
    lines.append("\n## Decision\n")
    lines.append("```json\n" + json.dumps(decision, indent=2, default=str) + "\n```")
    (out_dir / "F4a1_report.md").write_text("\n".join(lines), encoding="utf-8")

    zip_dir(out_dir, zip_path)
    log("Results:")
    print(res.to_string(index=False), flush=True)
    log(f"decision={decision}")
    log(f"zip={zip_path}")
    log("DONE")


if __name__ == "__main__":
    main()
