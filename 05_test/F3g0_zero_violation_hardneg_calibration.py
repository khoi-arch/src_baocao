#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F3g0 zero-violation online hard-negative calibration.

Runs two configs on train_inner/calibration:
  - baseline CE
  - CE + zero-violation online hard-negative violation

No official validation.
No fixed hard-pair list.
No L3/family training signal.
No global alpha/margin.
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
    print(f"[F3g0] {msg}", flush=True)


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

    # L3 is for balanced split bookkeeping only, not training signal.
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
        "note": "L3 only for balanced split; not used by training loss.",
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
    best_hist_row = hist.sort_values("val_macro_f1", ascending=False).iloc[0]
    best_epoch = int(best_hist_row["epoch"])

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



def read_pred_csv(run_dir: Path) -> pd.DataFrame:
    p = run_dir / "val_predictions_best.csv"
    if not p.exists():
        raise FileNotFoundError(f"missing predictions: {p}")
    df = pd.read_csv(p)
    rename = {}
    if "true_id" in df.columns:
        rename["true_id"] = "y_true"
    if "pred_id" in df.columns:
        rename["pred_id"] = "y_pred"
    if "true_label" in df.columns:
        rename["true_label"] = "true_label"
    if "pred_label" in df.columns:
        rename["pred_label"] = "pred_label"
    df = df.rename(columns=rename)
    if "sample_index" not in df.columns:
        df["sample_index"] = np.arange(len(df))
    if "y_true" not in df.columns or "y_pred" not in df.columns:
        raise ValueError(f"Cannot parse prediction columns from {p}: {df.columns.tolist()}")
    df = df.sort_values("sample_index").reset_index(drop=True)
    return df


def compute_fix_damage(base_dir: Path, method_dir: Path, label_names: List[str], out_dir: Path) -> Dict[str, Any]:
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
    if "confidence" in base.columns:
        comp["base_confidence"] = base["confidence"]
    if "confidence" in method.columns:
        comp["method_confidence"] = method["confidence"]

    conditions = [
        base_correct & method_correct,
        (~base_correct) & method_correct,
        base_correct & (~method_correct),
        (~base_correct) & (~method_correct) & (base_pred == method_pred),
        (~base_correct) & (~method_correct) & (base_pred != method_pred),
    ]
    choices = ["both_correct", "fixed", "damaged", "both_wrong_same", "both_wrong_changed"]
    comp["switch_type"] = np.select(conditions, choices, default="unknown")

    comp.to_csv(out_dir / "F3g0_fix_damage_rows.csv", index=False)

    switch_counts = comp["switch_type"].value_counts().rename_axis("switch_type").reset_index(name="count")
    switch_counts.to_csv(out_dir / "F3g0_fix_damage_counts.csv", index=False)

    by_class = comp.groupby(["true_label", "switch_type"]).agg(
        count=("sample_index", "count")
    ).reset_index().sort_values(["true_label", "count"], ascending=[True, False])
    by_class.to_csv(out_dir / "F3g0_fix_damage_by_true_class.csv", index=False)

    transitions = comp.groupby(["true_label", "base_pred_label", "method_pred_label"]).agg(
        count=("sample_index", "count")
    ).reset_index().sort_values("count", ascending=False)
    transitions.to_csv(out_dir / "F3g0_prediction_transition_summary.csv", index=False)

    def cnt(name: str) -> int:
        return int((comp["switch_type"] == name).sum())

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
    per_class_net_df = pd.DataFrame(per_class_net)
    per_class_net_df.to_csv(out_dir / "F3g0_per_class_fix_damage_net.csv", index=False)

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
    (out_dir / "F3g0_fix_damage_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def zip_dir(src: Path, dst: Path):
    if dst.exists():
        dst.unlink()
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for p in src.rglob("*"):
            if p.is_file() and p != dst:
                z.write(p, p.relative_to(src.parent))


def run_one(*, root: Path, trainer: Path, runs_root: Path, run_name: str, enabled: bool, subset_npz: Path, metadata_json: Path, train_inner_raw: Path, calib_raw: Path, args):
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
    ]
    cmd.append("--zmh-enabled" if enabled else "--no-zmh-enabled")

    run_dir = find_completed_run(runs_root, run_name)
    if run_dir is not None:
        log(f"Reusing completed run: {run_dir}")
        return run_dir
    subprocess.run(cmd, cwd=root, check=True)
    run_dir = find_completed_run(runs_root, run_name)
    if run_dir is None:
        raise FileNotFoundError(f"Cannot find completed run: {run_name}")
    return run_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-npz", default="03_outputs/05_dataset/dataset.npz")
    ap.add_argument("--metadata-json", default="03_outputs/05_dataset/metadata.json")
    ap.add_argument("--train-raw", default="01_split/train_raw.csv")
    ap.add_argument("--trainer", default="02_src/07_train_zero_violation_hardneg.py")
    ap.add_argument("--out-dir", default="05_test/outputs/F3g0_zero_violation_hardneg_calibration")
    ap.add_argument("--combined-zip", default="05_test/outputs/F3g0_zero_violation_hardneg_calibration.zip")
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

    if not trainer.exists():
        raise FileNotFoundError(f"trainer not found: {trainer}")

    label_names = parse_list(args.class_names) or CLASS_NAMES_DEFAULT

    data = np.load(dataset_npz, allow_pickle=True)
    y_train = np.asarray(data["y_train"], dtype=np.int64)
    train_idx, calib_idx, split_info = make_split(train_raw, y_train, label_names, args.calib_size, args.seed)
    subset_npz, train_inner_raw, calib_raw = make_subset_npz_and_raw(dataset_npz, train_raw, out_dir / "_split_artifacts", train_idx, calib_idx)

    config = {
        "experiment": "F3g0_zero_violation_online_hard_negative_calibration",
        "official_validation_used": False,
        "uses_fixed_hard_pairs": False,
        "uses_l3_or_family_labels_for_training": False,
        "uses_fixed_margin": False,
        "uses_global_aux_alpha": False,
        "method": "CE + mean(sum_wrong softmax(wrong_logits) * ReLU(logit_wrong - logit_true))",
        "configs": ["baseline_CE", "zero_margin_online_hardneg"],
        "split_info": split_info,
        "dataset_subset_npz": str(subset_npz),
        "train_inner_raw": str(train_inner_raw),
        "calibration_raw": str(calib_raw),
        "trainer": str(trainer),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    runs_root = out_dir / "config_runs"
    results = []
    for run_name, enabled in [("baseline_ce", False), ("zero_margin_online_hardneg", True)]:
        log(f"Running {run_name} enabled={enabled}")
        run_dir = run_one(
            root=root,
            trainer=trainer,
            runs_root=runs_root,
            run_name=run_name,
            enabled=enabled,
            subset_npz=subset_npz,
            metadata_json=metadata_json,
            train_inner_raw=train_inner_raw,
            calib_raw=calib_raw,
            args=args,
        )
        row = read_metrics(run_dir, label_names)
        row["run_name"] = run_name
        row["zvh_enabled"] = bool(enabled)
        row["run_dir"] = str(run_dir)
        results.append(row)
        pd.DataFrame(results).to_csv(out_dir / "F3g0_results_partial.csv", index=False)
        log(f"{run_name}: calibration_macro_f1={row['calibration_macro_f1']:.6f}")

    res = pd.DataFrame(results).sort_values(["calibration_macro_f1", "calibration_accuracy"], ascending=False).reset_index(drop=True)
    res.to_csv(out_dir / "F3g0_results.csv", index=False)

    base_row = res[res["run_name"] == "baseline_ce"].iloc[0].to_dict()
    zvh_row = res[res["run_name"] == "zero_margin_online_hardneg"].iloc[0].to_dict()
    delta = float(zvh_row["calibration_macro_f1"] - base_row["calibration_macro_f1"])

    base_run_dir = Path(base_row["run_dir"])
    method_run_dir = Path(zvh_row["run_dir"])
    fix_damage_summary = compute_fix_damage(base_run_dir, method_run_dir, label_names, out_dir)

    proceed = (
        delta > 0
        and int(fix_damage_summary.get("net_fixed_minus_damaged", -999999)) > 0
        and float(fix_damage_summary.get("damage_over_fix", 999999.0)) <= 1.0
    )

    decision = {
        "official_validation_used": False,
        "uses_fixed_hard_pairs": False,
        "uses_l3_or_family_labels_for_training": False,
        "uses_fixed_margin": False,
        "uses_global_aux_alpha": False,
        "baseline_row": base_row,
        "zero_violation_hardneg_row": zvh_row,
        "delta_zvh_minus_baseline_calibration_macro_f1": delta,
        "fix_damage_summary": fix_damage_summary,
        "selection_rule": "Proceed only if calibration macro-F1 improves, net fixed-damaged > 0, and damage/fix <= 1.0.",
        "recommendation": "proceed_to_full_train_once" if proceed else "reject_zero_violation_hardneg",
    }
    (out_dir / "F3g0_decision.json").write_text(json.dumps(decision, indent=2, default=str), encoding="utf-8")

    lines = []
    lines.append("# F3g0 Zero-violation online hard-negative calibration\n")
    lines.append("## Protocol\n")
    lines.append("```text")
    lines.append("No fixed hard-pair list.")
    lines.append("No L3/family label used for training.")
    lines.append("No official validation used.")
    lines.append("No fixed margin threshold.")
    lines.append("No global auxiliary alpha.")
    lines.append("Auxiliary loss is zero for correct samples whose true logit exceeds all wrong logits.")
    lines.append("```")
    lines.append("\n## Results\n")
    lines.append(res.to_markdown(index=False))
    lines.append("\n## Fix/damage summary\n")
    lines.append("```json")
    lines.append(json.dumps(fix_damage_summary, indent=2, default=str))
    lines.append("```")
    lines.append("\n## Decision\n")
    lines.append("```json")
    lines.append(json.dumps(decision, indent=2, default=str))
    lines.append("```")
    (out_dir / "F3g0_report.md").write_text("\n".join(lines), encoding="utf-8")

    zip_dir(out_dir, zip_path)
    log("Results:")
    print(res.to_string(index=False), flush=True)
    log(f"decision={decision}")
    log(f"zip={zip_path}")
    log("DONE")


if __name__ == "__main__":
    main()
