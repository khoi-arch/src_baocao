#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F1a4 L1 Compensation / Capacity Reallocation Parallel Runner

Purpose
-------
Previous anti-overfit search mostly tested capacity reductions:
- depth down: L2, L1
- hidden_dim down: H96/H64
- classifier_hidden down: CH96/CH64/CH32
- reduction combos: L1+CH32, L1+CH64, L2+CH32

User correctly noted this does not cover asymmetric interaction:
- reduce the overfitting part, then compensate capacity elsewhere
- e.g. depth down but classifier/width/head partition up

F1a4 is a controlled non-bruteforce interaction screen around the current best anti-overfit anchor:
- anchor = L1, num_layers=1
- test whether compensating classifier / width / head partition improves L1

Important non-overlap design
----------------------------
These variants are deliberately NOT duplicates of:
- official base: layers=3, hidden=128, classifier=128, heads=4
- F1a stage1: L2, H96, H64, CH64
- F1a2 stage2: L1, CH96, CH32
- F1a3 combos: L1+CH32, L1+CH64, L2+CH32

All F1a4 variants use num_layers=1 plus exactly one compensation axis:
1. L1 + classifier_hidden_dim=192
2. L1 + classifier_hidden_dim=256
3. L1 + hidden_dim=160
4. L1 + hidden_dim=192
5. L1 + num_heads=8

Execution
---------
- Uses up to 2 GPUs by default, because Kaggle T4x2 has 2 GPUs.
- Runs 2 jobs concurrently, then schedules the remaining jobs when GPUs free up.
- Streams child epoch logs live to notebook and also saves per-run log files.
- Collects recursively under Keff512/<run_name>.
- Writes summary CSV/MD and zips raw runs + summary + logs into one ALL zip.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import threading
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


BASE_REFERENCE = {
    "official_D3_base512": {
        "train_macro_f1": 0.910253,
        "val_macro_f1": 0.810094,
        "gap_macro_f1": 0.100158,
        "config": {
            "batch_size": 512,
            "num_layers": 3,
            "hidden_dim": 128,
            "classifier_hidden_dim": 128,
            "num_heads": 4,
        },
        "source": "F0 official D3 audit, batch512.",
    },
}

STAGE_REFERENCES = {
    "F1a2_L1_single_current_anchor": {
        "train_macro_f1": 0.911431,
        "val_macro_f1": 0.814224,
        "gap_macro_f1": 0.097207,
        "config": {
            "num_layers": 1,
            "hidden_dim": 128,
            "classifier_hidden_dim": 128,
            "num_heads": 4,
        },
        "note": "Current best anti-overfit single candidate before F1a4.",
    },
    "F1a2_CH32_single": {
        "train_macro_f1": 0.906852,
        "val_macro_f1": 0.812559,
        "gap_macro_f1": 0.094293,
        "note": "Best classifier bottleneck single, but weaker val than L1.",
    },
    "F1a3_L1_CH32_combo": {
        "train_macro_f1": 0.894431,
        "val_macro_f1": 0.811858,
        "gap_macro_f1": 0.082573,
        "note": "Best F1a3 combo, reduced gap but lower val than L1.",
    },
}

# Non-overlap checked manually:
# - no base: base is layers=3, hidden=128, classifier=128, heads=4
# - no pure L1: layers=1, hidden=128, classifier=128, heads=4
# - no L1+CH32/64, no L2+CH32
# - no H96/H64 or CH96/64/32 alone
VARIANTS = [
    {
        "name": "F1a4_L1_CH192_compensate_classifier_light",
        "description": "L1 anchor + increase classifier_hidden_dim 128 -> 192",
        "axis": "l1_plus_classifier_compensation",
        "overrides": ["--num-layers", "1", "--classifier-hidden-dim", "192"],
    },
    {
        "name": "F1a4_L1_CH256_compensate_classifier_strong",
        "description": "L1 anchor + increase classifier_hidden_dim 128 -> 256",
        "axis": "l1_plus_classifier_compensation",
        "overrides": ["--num-layers", "1", "--classifier-hidden-dim", "256"],
    },
    {
        "name": "F1a4_L1_H160_compensate_hidden_light",
        "description": "L1 anchor + increase hidden_dim 128 -> 160",
        "axis": "l1_plus_hidden_compensation",
        "overrides": ["--num-layers", "1", "--hidden-dim", "160"],
    },
    {
        "name": "F1a4_L1_H192_compensate_hidden_strong",
        "description": "L1 anchor + increase hidden_dim 128 -> 192",
        "axis": "l1_plus_hidden_compensation",
        "overrides": ["--num-layers", "1", "--hidden-dim", "192"],
    },
    {
        "name": "F1a4_L1_HEADS8_head_partition",
        "description": "L1 anchor + change num_heads 4 -> 8, hidden_dim stays 128",
        "axis": "l1_plus_head_partition",
        "overrides": ["--num-layers", "1", "--num-heads", "8"],
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
                if str(k).lower() in {"accuracy", "macro avg", "weighted avg"}:
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

    base = BASE_REFERENCE["official_D3_base512"]
    l1 = STAGE_REFERENCES["F1a2_L1_single_current_anchor"]
    combo = STAGE_REFERENCES["F1a3_L1_CH32_combo"]

    if row["val_macro_f1"] is not None:
        row["delta_val_vs_base"] = float(row["val_macro_f1"] - base["val_macro_f1"])
        row["delta_val_vs_L1_anchor"] = float(row["val_macro_f1"] - l1["val_macro_f1"])
        row["delta_val_vs_best_combo_L1_CH32"] = float(row["val_macro_f1"] - combo["val_macro_f1"])
    if row["train_macro_f1"] is not None:
        row["delta_train_vs_base"] = float(row["train_macro_f1"] - base["train_macro_f1"])
        row["delta_train_vs_L1_anchor"] = float(row["train_macro_f1"] - l1["train_macro_f1"])
    if row["gap_macro_f1"] is not None:
        row["delta_gap_vs_base"] = float(row["gap_macro_f1"] - base["gap_macro_f1"])
        row["delta_gap_vs_L1_anchor"] = float(row["gap_macro_f1"] - l1["gap_macro_f1"])
        row["delta_gap_vs_best_combo_L1_CH32"] = float(row["gap_macro_f1"] - combo["gap_macro_f1"])

    row["val_improved_vs_L1_anchor"] = bool(row.get("delta_val_vs_L1_anchor", -999) > 0)
    row["gap_not_worse_than_base"] = bool(row.get("gap_macro_f1", 999) <= base["gap_macro_f1"])
    row["gap_not_much_worse_than_L1"] = bool(row.get("delta_gap_vs_L1_anchor", 999) <= 0.005)

    if row["val_improved_vs_L1_anchor"] and row["gap_not_much_worse_than_L1"]:
        row["diagnosis"] = "strong_compensation_candidate"
    elif row["val_improved_vs_L1_anchor"] and row["gap_not_worse_than_base"]:
        row["diagnosis"] = "val_gain_compensation_candidate_check_gap"
    elif row["val_improved_vs_L1_anchor"]:
        row["diagnosis"] = "val_gain_but_gap_too_worse"
    elif row.get("delta_val_vs_base", -999) > 0 and row["gap_not_worse_than_base"]:
        row["diagnosis"] = "beats_base_but_not_L1"
    else:
        row["diagnosis"] = "no_compensation_signal"

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
        "--device", "cuda",
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


def _stream_process_output(proc: subprocess.Popen, log_file, prefix: str) -> None:
    """Stream child stdout both to notebook and to per-run log file."""
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            log_file.write(line + "\n")
            log_file.flush()
            print(f"[{prefix}] {line}", flush=True)
    except Exception as e:
        print(f"[F1a4][stream-error][{prefix}] {e}", flush=True)


def launch_job(args, root: Path, variant: Dict[str, Any], gpu_id: str) -> subprocess.Popen:
    train_script = resolve_path(args.train_script, root)
    log_dir = resolve_path(args.log_dir, root)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{variant['name']}.log"

    common = build_common_args(args)
    cmd = [
        sys.executable,
        str(train_script),
        *common,
        *variant["overrides"],
        "--out-root", args.out_root,
        "--run-name", variant["name"],
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id

    print("=" * 100, flush=True)
    print(f"[F1a4] launch {variant['name']} on physical GPU {gpu_id}", flush=True)
    print(f"[F1a4] axis: {variant['axis']}", flush=True)
    print(f"[F1a4] desc: {variant['description']}", flush=True)
    print(f"[F1a4] log: {log_path}", flush=True)
    print("[F1a4] CMD:", " ".join(cmd), flush=True)

    f = log_path.open("w", encoding="utf-8")
    f.write(f"[F1a4] variant={variant['name']}\n")
    f.write(f"[F1a4] gpu={gpu_id}\n")
    f.write(f"[F1a4] axis={variant['axis']}\n")
    f.write(f"[F1a4] description={variant['description']}\n")
    f.write("[F1a4] cmd=" + " ".join(cmd) + "\n\n")
    f.flush()

    p = subprocess.Popen(
        cmd,
        cwd=str(root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    p._f1a4_log_file = f  # type: ignore[attr-defined]
    p._f1a4_variant_name = variant["name"]  # type: ignore[attr-defined]
    p._f1a4_gpu_id = gpu_id  # type: ignore[attr-defined]
    p._f1a4_stream_thread = threading.Thread(
        target=_stream_process_output,
        args=(p, f, variant["name"]),
        daemon=True,
    )  # type: ignore[attr-defined]
    p._f1a4_stream_thread.start()  # type: ignore[attr-defined]
    return p


def expected_run_dir(args, root: Path, variant_name: str) -> Path:
    return resolve_path(args.out_root, root) / f"Keff{args.K}" / variant_name


def run_parallel(args, root: Path) -> None:
    gpu_ids = [x.strip() for x in args.gpu_ids.split(",") if x.strip()]
    if not gpu_ids:
        gpu_ids = ["0"]
    max_parallel = min(args.max_parallel, len(gpu_ids))
    if max_parallel < 1:
        max_parallel = 1

    queue = list(VARIANTS)
    running: List[subprocess.Popen] = []

    print(f"[F1a4] GPUs={gpu_ids} max_parallel={max_parallel}", flush=True)
    print("[F1a4] T4x2 mode: two jobs concurrently; remaining jobs wait for a GPU.", flush=True)
    print("[F1a4] Non-overlap check: no base, no L1 single, no F1a/F1a2/F1a3 duplicates.", flush=True)

    while queue or running:
        used_gpus = {getattr(p, "_f1a4_gpu_id") for p in running}
        free_gpus = [g for g in gpu_ids if g not in used_gpus]
        while queue and len(running) < max_parallel and free_gpus:
            variant = queue.pop(0)
            rd = expected_run_dir(args, root, variant["name"])
            if args.skip_existing and (rd / "diagnosis_summary.json").exists():
                print(f"[F1a4] skip existing: {variant['name']}", flush=True)
                continue
            gpu = free_gpus.pop(0)
            running.append(launch_job(args, root, variant, gpu))

        time.sleep(args.poll_seconds)
        still_running: List[subprocess.Popen] = []
        for p in running:
            ret = p.poll()
            if ret is None:
                still_running.append(p)
            else:
                name = getattr(p, "_f1a4_variant_name", "unknown")
                gpu = getattr(p, "_f1a4_gpu_id", "?")
                thread = getattr(p, "_f1a4_stream_thread", None)
                if thread is not None:
                    thread.join(timeout=5)
                logf = getattr(p, "_f1a4_log_file", None)
                if logf is not None:
                    logf.flush()
                    logf.close()
                print(f"[F1a4] finished {name} on GPU {gpu} with returncode={ret}", flush=True)
                if ret != 0:
                    raise RuntimeError(f"variant failed: {name}, returncode={ret}. Check logs in {args.log_dir}")
        running = still_running


def collect_runs(runs_root: Path, summary_out: Path) -> pd.DataFrame:
    summary_out.mkdir(parents=True, exist_ok=True)
    diag_paths = sorted(runs_root.rglob("diagnosis_summary.json")) if runs_root.exists() else []

    rows = []
    for diag in diag_paths:
        row = read_run(diag.parent, runs_root)
        if row is not None:
            rows.append(row)

    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["val_macro_f1", "gap_macro_f1"], ascending=[False, True])

    df.to_csv(summary_out / "F1a4_l1_compensation_summary.csv", index=False)
    (summary_out / "F1a4_base_reference.json").write_text(json.dumps(BASE_REFERENCE, indent=2), encoding="utf-8")
    (summary_out / "F1a4_stage_references.json").write_text(json.dumps(STAGE_REFERENCES, indent=2), encoding="utf-8")
    (summary_out / "F1a4_found_diagnosis_paths.txt").write_text("\n".join(str(p) for p in diag_paths), encoding="utf-8")
    write_summary_md(summary_out, df, diag_paths)
    return df


def write_summary_md(out_dir: Path, df: pd.DataFrame, diag_paths: List[Path]) -> None:
    lines = []
    lines.append("# F1a4 L1 Compensation / Capacity Reallocation Summary\n")
    lines.append("## Design goal\n")
    lines.append("```text")
    lines.append("Question: after reducing depth to L1, can we compensate capacity elsewhere?")
    lines.append("This is not reduction+reduction; this tests asymmetric capacity reallocation.")
    lines.append("All variants are non-overlap with official base, L1 single, F1a, F1a2, and F1a3.")
    lines.append("```")

    lines.append("\n## Tested variants\n")
    lines.append("```text")
    for v in VARIANTS:
        lines.append(f"{v['name']}: {v['description']}")
    lines.append("```")

    lines.append("\n## References\n")
    lines.append("```text")
    base = BASE_REFERENCE["official_D3_base512"]
    l1 = STAGE_REFERENCES["F1a2_L1_single_current_anchor"]
    combo = STAGE_REFERENCES["F1a3_L1_CH32_combo"]
    lines.append(f"Base512 val macro-F1 = {base['val_macro_f1']:.6f}, gap = {base['gap_macro_f1']:.6f}")
    lines.append(f"L1 anchor val macro-F1 = {l1['val_macro_f1']:.6f}, gap = {l1['gap_macro_f1']:.6f}")
    lines.append(f"Best F1a3 combo L1+CH32 val = {combo['val_macro_f1']:.6f}, gap = {combo['gap_macro_f1']:.6f}")
    lines.append("```")

    lines.append("\n## Found runs\n")
    lines.append("```text")
    lines.append(f"diagnosis_summary.json files found = {len(diag_paths)}")
    for p in diag_paths:
        lines.append(str(p))
    lines.append("```")

    lines.append("\n## F1a4 results\n")
    if len(df) == 0:
        lines.append("No completed F1a4 runs found.")
    else:
        cols = [
            "variant", "best_epoch", "train_macro_f1", "val_macro_f1", "gap_macro_f1",
            "delta_train_vs_base", "delta_val_vs_base", "delta_gap_vs_base",
            "delta_val_vs_L1_anchor", "delta_gap_vs_L1_anchor", "diagnosis",
        ]
        existing = [c for c in cols if c in df.columns]
        lines.append(df[existing].sort_values("val_macro_f1", ascending=False).to_markdown(index=False))

        best = df.sort_values("val_macro_f1", ascending=False).iloc[0]
        lines.append("\n## Best F1a4 by val macro-F1\n")
        lines.append("```text")
        lines.append(f"variant              = {best['variant']}")
        lines.append(f"val_macro_f1         = {best['val_macro_f1']:.6f}")
        lines.append(f"train_macro_f1       = {best['train_macro_f1']:.6f}")
        lines.append(f"gap                  = {best['gap_macro_f1']:.6f}")
        lines.append(f"delta_val_vs_base    = {best['delta_val_vs_base']:+.6f}")
        lines.append(f"delta_gap_vs_base    = {best['delta_gap_vs_base']:+.6f}")
        lines.append(f"delta_val_vs_L1      = {best['delta_val_vs_L1_anchor']:+.6f}")
        lines.append(f"delta_gap_vs_L1      = {best['delta_gap_vs_L1_anchor']:+.6f}")
        lines.append(f"diagnosis            = {best['diagnosis']}")
        lines.append("```")

    lines.append("\n## Decision rule\n")
    lines.append("```text")
    lines.append("If any F1a4 variant beats L1 and keeps gap close to or below L1:")
    lines.append("  select that compensation variant as anti-overfit candidate.")
    lines.append("")
    lines.append("If a variant beats L1 but gap grows too much:")
    lines.append("  treat it as architecture gain, not clean anti-overfit.")
    lines.append("")
    lines.append("If no variant beats L1:")
    lines.append("  L1 remains anti-overfit candidate.")
    lines.append("")
    lines.append("After this, do not open brute-force combos unless the error audit gives a specific reason.")
    lines.append("```")

    (out_dir / "F1a4_l1_compensation_summary.md").write_text("\n".join(lines), encoding="utf-8")


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
    ap.add_argument("--out-root", default="05_test/outputs/F1a4_l1_compensation_parallel")
    ap.add_argument("--summary-out", default="05_test/outputs/F1a4_l1_compensation_parallel_summary")
    ap.add_argument("--log-dir", default="05_test/outputs/F1a4_l1_compensation_parallel_logs")
    ap.add_argument("--combined-zip", default="05_test/outputs/F1a4_l1_compensation_parallel_ALL.zip")

    # D3 training config, base512 compatible
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--num-bins", type=int, default=512)
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

    # Parallel execution
    ap.add_argument("--gpu-ids", default="0,1", help="Physical GPU ids to use, e.g. 0,1 for Kaggle T4x2.")
    ap.add_argument("--max-parallel", type=int, default=2)
    ap.add_argument("--poll-seconds", type=float, default=10.0)
    ap.add_argument("--collect-only", action="store_true")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.add_argument("--make-zip", action="store_true", default=True)
    ap.add_argument("--no-zip", dest="make_zip", action="store_false")

    args = ap.parse_args()

    root = repo_root_from_here()
    runs_root = resolve_path(args.out_root, root)
    summary_out = resolve_path(args.summary_out, root)
    log_dir = resolve_path(args.log_dir, root)
    combined_zip = resolve_path(args.combined_zip, root)

    print(f"[F1a4] root={root}", flush=True)
    print(f"[F1a4] runs_root={runs_root}", flush=True)
    print(f"[F1a4] summary_out={summary_out}", flush=True)
    print(f"[F1a4] log_dir={log_dir}", flush=True)
    print("[F1a4] Running only L1 compensation variants; no duplicate base/stage variants.", flush=True)

    if not args.collect_only:
        run_parallel(args, root)

    df = collect_runs(runs_root, summary_out)

    if args.make_zip:
        summary_zip = summary_out.with_suffix(".zip")
        raw_zip = runs_root.with_suffix(".zip")
        log_zip = log_dir.with_suffix(".zip")
        zip_dir(summary_out, summary_zip)
        zip_dir(runs_root, raw_zip)
        zip_dir(log_dir, log_zip)
        make_combined_zip(
            [(summary_out, "summary"), (runs_root, "raw_runs"), (log_dir, "logs")],
            combined_zip,
        )
        print(f"[F1a4] summary zip: {summary_zip}", flush=True)
        print(f"[F1a4] raw runs zip: {raw_zip}", flush=True)
        print(f"[F1a4] logs zip: {log_zip}", flush=True)
        print(f"[F1a4] combined zip: {combined_zip}", flush=True)

    print(f"[F1a4] collected rows={len(df)}", flush=True)
    if len(df):
        cols = ["variant", "train_macro_f1", "val_macro_f1", "gap_macro_f1", "delta_val_vs_L1_anchor", "delta_gap_vs_L1_anchor", "diagnosis"]
        print(df[cols].to_string(index=False), flush=True)
    print("[F1a4] DONE", flush=True)
    print("Upload this combined zip:", flush=True)
    print(str(combined_zip), flush=True)


if __name__ == "__main__":
    main()
