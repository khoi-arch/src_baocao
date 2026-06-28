#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F1a Stage2 Depth + Classifier Bottleneck Runner

Purpose
-------
After F1a screening:
- L2 (num_layers 3->2) was best val but gap increased.
- CH64 (classifier_hidden 128->64) had tiny positive signal.
- H96/H64 reduced hidden_dim were worse.

Stage2 therefore fine-tunes only the promising/non-duplicate axes:
1) depth boundary:
   - num_layers = 1
2) classifier bottleneck:
   - classifier_hidden_dim = 96
   - classifier_hidden_dim = 32

No pair head.
No swapper.
No tree teacher.
No SupCon.
No regularization-only.
No K/global preprocessing change.
No rare token merge.
No confidence-only/rerank/prototype.

This script:
- runs all selected variants sequentially
- recursively collects runs under Keff512/<run_name>
- writes summary CSV/MD
- zips summary
- zips raw run outputs
- zips one combined archive so user does not need to find folders manually
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


BASE_REFERENCE = {
    "name": "official_D3_baseline_F0",
    "train_macro_f1": 0.910253,
    "val_macro_f1": 0.810094,
    "gap_macro_f1": 0.100158,
    "note": "From F0 official D3 audit.",
}

STAGE1_REFERENCE = {
    "F1a_L2_reduce_num_layers": {
        "train_macro_f1": 0.952964,
        "val_macro_f1": 0.813096,
        "gap_macro_f1": 0.139869,
        "note": "Stage1 best val, but larger gap.",
    },
    "F1a_CH64_reduce_classifier_hidden": {
        "train_macro_f1": 0.911808,
        "val_macro_f1": 0.810913,
        "gap_macro_f1": 0.100894,
        "note": "Stage1 classifier bottleneck mild positive signal.",
    },
    "F1a_H96_reduce_hidden_dim": {
        "train_macro_f1": 0.918719,
        "val_macro_f1": 0.807574,
        "gap_macro_f1": 0.111145,
        "note": "Rejected: hidden_dim reduction hurt val.",
    },
    "F1a_H64_reduce_hidden_dim_strong": {
        "train_macro_f1": 0.927552,
        "val_macro_f1": 0.803150,
        "gap_macro_f1": 0.124402,
        "note": "Rejected: stronger hidden_dim reduction hurt val.",
    },
}


VARIANTS = [
    {
        "name": "F1a2_L1_reduce_num_layers_strong",
        "description": "depth fine-tune: num_layers 3 -> 1; only run after L2 gave val gain",
        "overrides": ["--num-layers", "1"],
        "axis": "depth",
    },
    {
        "name": "F1a2_CH96_reduce_classifier_hidden_light",
        "description": "classifier bottleneck fine-tune: classifier_hidden_dim 128 -> 96",
        "overrides": ["--classifier-hidden-dim", "96"],
        "axis": "classifier_bottleneck",
    },
    {
        "name": "F1a2_CH32_reduce_classifier_hidden_strong",
        "description": "classifier bottleneck fine-tune: classifier_hidden_dim 128 -> 32",
        "overrides": ["--classifier-hidden-dim", "32"],
        "axis": "classifier_bottleneck",
    },
]


def repo_root_from_here() -> Path:
    p = Path(__file__).resolve()
    if p.parent.name == "05_test":
        return p.parents[1]
    return Path.cwd().resolve()


def resolve_path(p: str | Path, root: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (root / p).resolve()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def maybe_get(d: Dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def flatten_report_f1(row: Dict[str, Any], split: str, rep: Dict[str, Any]) -> None:
    if not isinstance(rep, dict):
        return

    if "per_class" in rep and isinstance(rep["per_class"], dict):
        items = rep["per_class"].items()
    else:
        items = []
        for k, v in rep.items():
            if isinstance(v, dict) and ("f1" in v or "f1-score" in v):
                if k.lower() in {"accuracy", "macro avg", "weighted avg"}:
                    continue
                items.append((k, v))

    for label, metrics in items:
        f1 = metrics.get("f1", metrics.get("f1-score"))
        if f1 is not None:
            safe = str(label).replace(" ", "_").replace("/", "_")
            row[f"{split}_f1_{safe}"] = float(f1)


def read_run(run_dir: Path, runs_root: Path) -> Dict[str, Any] | None:
    diag_path = run_dir / "diagnosis_summary.json"
    if not diag_path.exists():
        return None

    cfg_path = run_dir / "config.json"
    hist_path = run_dir / "history.csv"
    val_report_path = run_dir / "val_classification_report_best.json"
    train_report_path = run_dir / "train_classification_report_best.json"

    diag = load_json(diag_path)
    cfg = load_json(cfg_path) if cfg_path.exists() else {}
    val_report = load_json(val_report_path) if val_report_path.exists() else {}
    train_report = load_json(train_report_path) if train_report_path.exists() else {}

    model_cfg = cfg.get("model", cfg.get("model_config", {}))
    if not isinstance(model_cfg, dict):
        model_cfg = {}

    try:
        rel = str(run_dir.relative_to(runs_root))
    except ValueError:
        rel = run_dir.name

    row = {
        "variant": run_dir.name,
        "relative_run_dir": rel,
        "run_dir": str(run_dir),
        "best_epoch": diag.get("best_epoch"),
        "epochs_ran": None,
        "train_macro_f1": maybe_get(diag, "train", "macro_f1"),
        "val_macro_f1": maybe_get(diag, "val", "macro_f1"),
        "gap_macro_f1": diag.get("generalization_gap_macro_f1"),
        "train_acc": maybe_get(diag, "train", "accuracy"),
        "val_acc": maybe_get(diag, "val", "accuracy"),
        "train_weighted_f1": maybe_get(diag, "train", "weighted_f1"),
        "val_weighted_f1": maybe_get(diag, "val", "weighted_f1"),
        "train_malware_avg_f1": maybe_get(diag, "train", "malware_only_avg_f1"),
        "val_malware_avg_f1": maybe_get(diag, "val", "malware_only_avg_f1"),
        "hidden_dim": model_cfg.get("hidden_dim", cfg.get("hidden_dim")),
        "num_layers": model_cfg.get("num_layers", cfg.get("num_layers")),
        "num_heads": model_cfg.get("num_heads", cfg.get("num_heads")),
        "dropout": model_cfg.get("dropout", cfg.get("dropout")),
        "classifier_hidden_dim": model_cfg.get("classifier_hidden_dim", cfg.get("classifier_hidden_dim")),
        "classifier_dropout": model_cfg.get("classifier_dropout", cfg.get("classifier_dropout")),
        "lr": cfg.get("lr", maybe_get(cfg, "training", "lr")),
        "weight_decay": cfg.get("weight_decay", maybe_get(cfg, "training", "weight_decay")),
        "batch_size": cfg.get("batch_size", maybe_get(cfg, "training", "batch_size")),
    }

    if hist_path.exists():
        hist = pd.read_csv(hist_path)
        row["epochs_ran"] = int(hist["epoch"].max()) if "epoch" in hist and len(hist) else len(hist)
        if "val_macro_f1" in hist and len(hist):
            best_i = hist["val_macro_f1"].idxmax()
            row["history_best_val_macro_f1"] = float(hist.loc[best_i, "val_macro_f1"])
            row["history_best_train_macro_f1"] = float(hist.loc[best_i, "train_macro_f1"]) if "train_macro_f1" in hist else None
            row["history_best_gap"] = float(hist.loc[best_i, "macro_f1_gap_train_minus_val"]) if "macro_f1_gap_train_minus_val" in hist else None

    flatten_report_f1(row, "train", train_report)
    flatten_report_f1(row, "val", val_report)

    if row["val_macro_f1"] is not None:
        row["delta_val_vs_base"] = float(row["val_macro_f1"] - BASE_REFERENCE["val_macro_f1"])
        row["delta_val_vs_stage1_L2"] = float(row["val_macro_f1"] - STAGE1_REFERENCE["F1a_L2_reduce_num_layers"]["val_macro_f1"])
        row["delta_val_vs_stage1_CH64"] = float(row["val_macro_f1"] - STAGE1_REFERENCE["F1a_CH64_reduce_classifier_hidden"]["val_macro_f1"])
    if row["train_macro_f1"] is not None:
        row["delta_train_vs_base"] = float(row["train_macro_f1"] - BASE_REFERENCE["train_macro_f1"])
    if row["gap_macro_f1"] is not None:
        row["delta_gap_vs_base"] = float(row["gap_macro_f1"] - BASE_REFERENCE["gap_macro_f1"])

    row["val_improved_vs_base"] = bool(row.get("delta_val_vs_base", -999) > 0)
    row["gap_reduced_vs_base"] = bool(row.get("delta_gap_vs_base", 999) < 0)

    if row["val_improved_vs_base"] and row["gap_reduced_vs_base"]:
        row["diagnosis"] = "good_generalization_signal"
    elif row["val_improved_vs_base"] and not row["gap_reduced_vs_base"]:
        row["diagnosis"] = "val_gain_but_gap_not_fixed"
    elif (not row["val_improved_vs_base"]) and row["gap_reduced_vs_base"]:
        row["diagnosis"] = "gap_reduced_but_underfit_or_no_val_gain"
    else:
        row["diagnosis"] = "no_stage2_signal"

    return row


def build_common_args(args) -> List[str]:
    return [
        "--run-id", "D3",
        "--K", str(args.K),
        "--num-bins", str(args.num_bins),
        "--dataset-npz", args.dataset_npz,
        "--metadata-json", args.metadata_json,
        "--train-raw", args.train_raw,
        "--val-raw", args.val_raw,
        "--device", args.device,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--scheduler", args.scheduler,
        "--warmup-epochs", str(args.warmup_epochs),
        "--min-lr-ratio", str(args.min_lr_ratio),
        "--patience", str(args.patience),
        "--min-delta", str(args.min_delta),
        "--num-workers", str(args.num_workers),
        "--grad-clip-norm", str(args.grad_clip_norm),
        "--use-class-weights",
        "--value-dim", str(args.value_dim),
        "--feature-dim", str(args.feature_dim),
        "--hidden-dim", str(args.hidden_dim),
        "--num-layers", str(args.num_layers),
        "--num-heads", str(args.num_heads),
        "--dropout", str(args.dropout),
        "--classifier-hidden-dim", str(args.classifier_hidden_dim),
        "--classifier-dropout", str(args.classifier_dropout),
        "--norm-first",
        "--gate-init", str(args.gate_init),
    ]


def run_variants(args, root: Path) -> None:
    common = build_common_args(args)
    train_script = resolve_path(args.train_script, root)

    for i, v in enumerate(VARIANTS, 1):
        variant_dir_guess = resolve_path(args.out_root, root) / f"Keff{args.K}" / v["name"]
        if args.skip_existing and (variant_dir_guess / "diagnosis_summary.json").exists():
            print(f"[F1a2] skip existing {i}/{len(VARIANTS)}: {v['name']}", flush=True)
            continue

        print("=" * 100, flush=True)
        print(f"[F1a2] {i}/{len(VARIANTS)} {v['name']}", flush=True)
        print(f"[F1a2] axis: {v['axis']}", flush=True)
        print(f"[F1a2] description: {v['description']}", flush=True)
        cmd = [
            sys.executable,
            str(train_script),
            *common,
            *v["overrides"],
            "--out-root", args.out_root,
            "--run-name", v["name"],
        ]
        print("[F1a2] CMD:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=str(root))


def collect_runs(runs_root: Path, summary_out: Path) -> pd.DataFrame:
    summary_out.mkdir(parents=True, exist_ok=True)
    diag_paths = sorted(runs_root.rglob("diagnosis_summary.json")) if runs_root.exists() else []

    rows: List[Dict[str, Any]] = []
    for diag in diag_paths:
        row = read_run(diag.parent, runs_root)
        if row is not None:
            rows.append(row)

    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["val_macro_f1", "gap_macro_f1"], ascending=[False, True])

    df.to_csv(summary_out / "F1a2_stage2_summary.csv", index=False)
    (summary_out / "F1a2_baseline_reference.json").write_text(json.dumps(BASE_REFERENCE, indent=2), encoding="utf-8")
    (summary_out / "F1a2_stage1_reference.json").write_text(json.dumps(STAGE1_REFERENCE, indent=2), encoding="utf-8")
    (summary_out / "F1a2_found_diagnosis_paths.txt").write_text(
        "\n".join(str(p) for p in diag_paths), encoding="utf-8"
    )
    write_summary_md(summary_out, df, diag_paths)
    return df


def write_summary_md(out_dir: Path, df: pd.DataFrame, diag_paths: List[Path]) -> None:
    lines = []
    lines.append("# F1a2 Stage2 Depth + Classifier Bottleneck Summary\n")
    lines.append("## What this stage tests\n")
    lines.append("```text")
    lines.append("depth fine-tune: num_layers=1")
    lines.append("classifier bottleneck fine-tune: classifier_hidden_dim=96 and 32")
    lines.append("No hidden_dim test here because H96/H64 hurt validation in Stage1.")
    lines.append("No dropout/WD/K/rare-merge/pair/swapper/SupCon/tree/reranker/prototype.")
    lines.append("```")
    lines.append("\n## References\n")
    lines.append("```text")
    lines.append(f"Base official D3 val macro-F1 = {BASE_REFERENCE['val_macro_f1']:.6f}, gap = {BASE_REFERENCE['gap_macro_f1']:.6f}")
    lines.append(f"Stage1 L2 val macro-F1       = {STAGE1_REFERENCE['F1a_L2_reduce_num_layers']['val_macro_f1']:.6f}, gap = {STAGE1_REFERENCE['F1a_L2_reduce_num_layers']['gap_macro_f1']:.6f}")
    lines.append(f"Stage1 CH64 val macro-F1     = {STAGE1_REFERENCE['F1a_CH64_reduce_classifier_hidden']['val_macro_f1']:.6f}, gap = {STAGE1_REFERENCE['F1a_CH64_reduce_classifier_hidden']['gap_macro_f1']:.6f}")
    lines.append("```")
    lines.append("\n## Found runs\n")
    lines.append("```text")
    lines.append(f"diagnosis_summary.json files found = {len(diag_paths)}")
    for p in diag_paths:
        lines.append(str(p))
    lines.append("```")

    lines.append("\n## Stage2 variants\n")
    if len(df) == 0:
        lines.append("No completed runs found.")
    else:
        cols = [
            "variant", "best_epoch", "train_macro_f1", "val_macro_f1", "gap_macro_f1",
            "delta_train_vs_base", "delta_val_vs_base", "delta_gap_vs_base",
            "delta_val_vs_stage1_L2", "delta_val_vs_stage1_CH64", "diagnosis",
        ]
        existing = [c for c in cols if c in df.columns]
        lines.append(df[existing].sort_values("val_macro_f1", ascending=False).to_markdown(index=False))

        best = df.sort_values("val_macro_f1", ascending=False).iloc[0]
        lines.append("\n## Best Stage2 run\n")
        lines.append("```text")
        lines.append(f"variant             = {best['variant']}")
        lines.append(f"val_macro_f1        = {best['val_macro_f1']:.6f}")
        lines.append(f"train_macro_f1      = {best['train_macro_f1']:.6f}")
        lines.append(f"gap                 = {best['gap_macro_f1']:.6f}")
        lines.append(f"delta_val_vs_base   = {best['delta_val_vs_base']:+.6f}")
        lines.append(f"delta_gap_vs_base   = {best['delta_gap_vs_base']:+.6f}")
        lines.append(f"delta_val_vs_L2     = {best['delta_val_vs_stage1_L2']:+.6f}")
        lines.append(f"delta_val_vs_CH64   = {best['delta_val_vs_stage1_CH64']:+.6f}")
        lines.append(f"diagnosis           = {best['diagnosis']}")
        lines.append("```")

    lines.append("\n## Next decision rule\n")
    lines.append("```text")
    lines.append("If L1 beats L2:")
    lines.append("  depth best is likely 1; combo candidate uses L1.")
    lines.append("")
    lines.append("If L1 underfits or val lower than L2:")
    lines.append("  depth best remains L2.")
    lines.append("")
    lines.append("If CH96/CH32 beat CH64:")
    lines.append("  classifier best changes to that value.")
    lines.append("")
    lines.append("If CH64 still best among classifier values:")
    lines.append("  classifier best remains 64.")
    lines.append("")
    lines.append("Only after this:")
    lines.append("  test combo = best_depth + best_classifier_hidden.")
    lines.append("```")

    (out_dir / "F1a2_stage2_summary.md").write_text("\n".join(lines), encoding="utf-8")


def zip_dir(src_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in src_dir.rglob("*"):
            if p.is_file() and p != zip_path:
                z.write(p, p.relative_to(src_dir.parent))


def make_combined_zip(paths: List[Tuple[Path, str]], zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for path, prefix in paths:
            if not path.exists():
                continue
            if path.is_file():
                z.write(path, Path(prefix) / path.name)
            else:
                for p in path.rglob("*"):
                    if p.is_file() and p != zip_path:
                        z.write(p, Path(prefix) / p.relative_to(path))


def main():
    ap = argparse.ArgumentParser()

    # Paths
    ap.add_argument("--train-script", default="02_src/07_train.py")
    ap.add_argument("--dataset-npz", default="03_outputs/05_dataset/dataset.npz")
    ap.add_argument("--metadata-json", default="03_outputs/05_dataset/metadata.json")
    ap.add_argument("--train-raw", default="01_split/train_raw.csv")
    ap.add_argument("--val-raw", default="01_split/val_raw.csv")
    ap.add_argument("--out-root", default="05_test/outputs/F1a2_stage2_depth_classifier")
    ap.add_argument("--summary-out", default="05_test/outputs/F1a2_stage2_depth_classifier_summary")
    ap.add_argument("--combined-zip", default="05_test/outputs/F1a2_stage2_depth_classifier_ALL.zip")

    # Official D3/common training config
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--num-bins", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--weight-decay", type=float, default=0.0001)
    ap.add_argument("--scheduler", default="warmup_cosine")
    ap.add_argument("--warmup-epochs", type=int, default=8)
    ap.add_argument("--min-lr-ratio", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--min-delta", type=float, default=0.0001)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--grad-clip-norm", type=float, default=1.0)
    ap.add_argument("--value-dim", type=int, default=32)
    ap.add_argument("--feature-dim", type=int, default=32)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=3)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--classifier-hidden-dim", type=int, default=128)
    ap.add_argument("--classifier-dropout", type=float, default=0.1)
    ap.add_argument("--gate-init", type=float, default=0.0)

    # Execution
    ap.add_argument("--collect-only", action="store_true")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.add_argument("--make-zip", action="store_true", default=True)
    ap.add_argument("--no-zip", dest="make_zip", action="store_false")

    args = ap.parse_args()

    root = repo_root_from_here()
    runs_root = resolve_path(args.out_root, root)
    summary_out = resolve_path(args.summary_out, root)
    combined_zip = resolve_path(args.combined_zip, root)

    print(f"[F1a2] root={root}", flush=True)
    print(f"[F1a2] runs_root={runs_root}", flush=True)
    print(f"[F1a2] summary_out={summary_out}", flush=True)
    print("[F1a2] No duplicate failed directions: no reg-only, no K, no rare merge, no pair/swapper/SupCon/tree.", flush=True)

    if not args.collect_only:
        run_variants(args, root)

    df = collect_runs(runs_root, summary_out)

    if args.make_zip:
        summary_zip = summary_out.with_suffix(".zip")
        raw_zip = runs_root.with_suffix(".zip")
        zip_dir(summary_out, summary_zip)
        zip_dir(runs_root, raw_zip)
        make_combined_zip(
            [(summary_out, "summary"), (runs_root, "raw_runs")],
            combined_zip,
        )
        print(f"[F1a2] summary zip: {summary_zip}", flush=True)
        print(f"[F1a2] raw runs zip: {raw_zip}", flush=True)
        print(f"[F1a2] combined zip: {combined_zip}", flush=True)

    print(f"[F1a2] collected rows={len(df)}", flush=True)
    if len(df):
        cols = ["variant", "train_macro_f1", "val_macro_f1", "gap_macro_f1", "delta_val_vs_base", "delta_val_vs_stage1_L2", "diagnosis"]
        print(df[cols].to_string(index=False), flush=True)
    print("[F1a2] DONE", flush=True)
    print("Upload this combined zip:", flush=True)
    print(str(combined_zip), flush=True)


if __name__ == "__main__":
    main()
