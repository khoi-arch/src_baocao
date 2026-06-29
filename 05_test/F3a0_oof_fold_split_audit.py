#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F3a0 OOF fold split audit.

Purpose:
    Step 1 for clean OOF overlap workflow.

This script does NOT train a model.

It creates deterministic K-fold assignments from the original training split,
preferably stratified by L2+L3 family labels.

Why:
    Later OOF training will train K models:
        fold i model trains on train rows where fold != i
        predicts rows where fold == i
    Therefore every OOF prediction is made by a model that did not train on
    that sample.

Outputs:
    - F3a0_oof_fold_assignments.csv
    - F3a0_fold_l2_summary.csv
    - F3a0_fold_l3_summary.csv
    - F3a0_strata_summary.csv
    - F3a0_report.md
    - config.json
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


LABEL_COL_CANDIDATES = [
    "label_L1", "label_L2", "label_L3", "Label_L1", "Label_L2", "Label_L3",
    "Class", "Category", "Family", "class", "category", "family",
    "MalwareFamily", "malware_family", "label", "target",
]


def log(msg: str):
    print(f"[F3a0] {msg}", flush=True)


def repo_root_from_here() -> Path:
    p = Path(__file__).resolve()
    if p.parent.name == "05_test":
        return p.parents[1]
    return Path.cwd().resolve()


def resolve_path(p: str | Path, root: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (root / p).resolve()


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
        cands = LABEL_COL_CANDIDATES
    for c in cands:
        if c in df.columns:
            return c
    return None


def safe_md(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df is None or len(df) == 0:
        return "_empty_"
    try:
        return df.head(max_rows).to_markdown(index=False)
    except Exception:
        return df.head(max_rows).to_string(index=False)


def choose_strata(l2: pd.Series, l3: pd.Series, n_splits: int) -> tuple[pd.Series, str, pd.DataFrame]:
    l2 = l2.map(clean_label)
    l3 = l3.map(clean_label)
    l3 = l3.where(l3.astype(str).str.len() > 0, l2)
    combo = l2.astype(str) + "::" + l3.astype(str)

    combo_counts = combo.value_counts().sort_values()
    l2_counts = l2.value_counts().sort_values()

    if len(combo_counts) > 0 and int(combo_counts.min()) >= int(n_splits):
        return combo, "L2_plus_L3", combo_counts.rename_axis("stratum").reset_index(name="count")

    if len(l2_counts) > 0 and int(l2_counts.min()) >= int(n_splits):
        # Rare L3 strata exist; fold on L2 but still report L3.
        return l2, "L2_only_due_to_rare_L3", combo_counts.rename_axis("stratum").reset_index(name="count")

    raise ValueError(
        f"Cannot make {n_splits} stratified folds: min L2 count={int(l2_counts.min()) if len(l2_counts) else None}, "
        f"min L2+L3 count={int(combo_counts.min()) if len(combo_counts) else None}"
    )


def fold_summary(assign: pd.DataFrame, label_col: str) -> pd.DataFrame:
    rows = []
    total_counts = assign[label_col].value_counts().to_dict()
    for fold in sorted(assign["fold"].unique()):
        g = assign[assign["fold"] == fold]
        counts = g[label_col].value_counts().to_dict()
        for label, total in sorted(total_counts.items(), key=lambda kv: str(kv[0])):
            rows.append({
                "fold": int(fold),
                "label_col": label_col,
                "label": label,
                "count": int(counts.get(label, 0)),
                "fold_size": int(len(g)),
                "fold_pct": float(counts.get(label, 0) / max(len(g), 1)),
                "global_count": int(total),
                "global_pct": float(total / max(len(assign), 1)),
                "abs_pct_diff": float(abs(counts.get(label, 0) / max(len(g), 1) - total / max(len(assign), 1))),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-raw", default="01_split/train_raw.csv")
    ap.add_argument("--dataset-npz", default="03_outputs/05_dataset/dataset.npz")
    ap.add_argument("--out-dir", default="05_test/outputs/F3a0_oof_fold_split_audit")
    ap.add_argument("--combined-zip", default="05_test/outputs/F3a0_oof_fold_split_audit.zip")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = repo_root_from_here()
    train_raw_path = resolve_path(args.train_raw, root)
    dataset_npz_path = resolve_path(args.dataset_npz, root)
    out_dir = resolve_path(args.out_dir, root)
    zip_path = resolve_path(args.combined_zip, root)

    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(train_raw_path)
    n = len(raw)

    # Optional sanity check against dataset y_train length.
    dataset_train_len = None
    if dataset_npz_path.exists():
        z = np.load(dataset_npz_path, allow_pickle=True)
        if "y_train" in z.files:
            dataset_train_len = int(len(z["y_train"]))
            if dataset_train_len != n:
                raise ValueError(f"train_raw length {n} != dataset y_train length {dataset_train_len}")

    l2_col = find_label_col(raw, "L2")
    l3_col = find_label_col(raw, "L3")
    if l2_col is None:
        raise ValueError("Cannot find L2 label column in train_raw.csv")
    if l3_col is None:
        # Fallback to L2 as family if no L3 column.
        raw["_F3a0_L3_fallback"] = raw[l2_col]
        l3_col = "_F3a0_L3_fallback"

    l2 = raw[l2_col].map(clean_label)
    l3 = raw[l3_col].map(clean_label)
    strata, stratification_mode, strata_summary = choose_strata(l2, l3, int(args.n_splits))

    skf = StratifiedKFold(n_splits=int(args.n_splits), shuffle=True, random_state=int(args.seed))
    fold = np.full(n, -1, dtype=np.int64)
    for fold_id, (_, val_idx) in enumerate(skf.split(np.zeros(n), strata.to_numpy())):
        fold[val_idx] = int(fold_id)

    if np.any(fold < 0):
        raise RuntimeError("Some rows were not assigned to a fold")

    assign = pd.DataFrame({
        "row_id": np.arange(n, dtype=np.int64),
        "fold": fold,
        "label_L2": l2.to_numpy(),
        "label_L3": l3.to_numpy(),
        "stratum": strata.to_numpy(),
    })
    assign.to_csv(out_dir / "F3a0_oof_fold_assignments.csv", index=False)

    l2_summary = fold_summary(assign, "label_L2")
    l3_summary = fold_summary(assign, "label_L3")
    l2_summary.to_csv(out_dir / "F3a0_fold_l2_summary.csv", index=False)
    l3_summary.to_csv(out_dir / "F3a0_fold_l3_summary.csv", index=False)
    strata_summary.to_csv(out_dir / "F3a0_strata_summary.csv", index=False)

    fold_sizes = assign["fold"].value_counts().sort_index()
    fold_size_df = fold_sizes.rename_axis("fold").reset_index(name="fold_size")
    fold_size_df["pct"] = fold_size_df["fold_size"] / n
    fold_size_df.to_csv(out_dir / "F3a0_fold_sizes.csv", index=False)

    # Integrity checks.
    integrity = {
        "n_rows": int(n),
        "dataset_y_train_len": dataset_train_len,
        "n_splits": int(args.n_splits),
        "seed": int(args.seed),
        "l2_col": l2_col,
        "l3_col": l3_col,
        "stratification_mode": stratification_mode,
        "fold_sizes": {str(int(k)): int(v) for k, v in fold_sizes.to_dict().items()},
        "min_fold_size": int(fold_sizes.min()),
        "max_fold_size": int(fold_sizes.max()),
        "max_abs_l2_pct_diff": float(l2_summary["abs_pct_diff"].max()),
        "max_abs_l3_pct_diff": float(l3_summary["abs_pct_diff"].max()),
        "row_id_unique": bool(assign["row_id"].is_unique),
        "all_rows_assigned_once": bool(len(assign) == n and assign["row_id"].nunique() == n and not (assign["fold"] < 0).any()),
    }

    config = {
        "experiment": "F3a0_oof_fold_split_audit",
        "training_performed": False,
        "purpose": "Create deterministic train-only K-fold assignments for later OOF overlap audit.",
        "train_raw": str(train_raw_path),
        "dataset_npz": str(dataset_npz_path),
        **integrity,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

    lines = []
    lines.append("# F3a0 OOF fold split audit\n")
    lines.append("## Purpose\n")
    lines.append("```text")
    lines.append("Create deterministic K-fold assignments from original training split.")
    lines.append("No training. No validation usage. This is the clean base for OOF mining.")
    lines.append("```")
    lines.append("\n## Integrity\n")
    lines.append("```json")
    lines.append(json.dumps(integrity, indent=2, default=str))
    lines.append("```")
    lines.append("\n## Fold sizes\n")
    lines.append(safe_md(fold_size_df, 10))
    lines.append("\n## L2 fold distribution\n")
    lines.append(safe_md(l2_summary.sort_values(["label", "fold"]), 30))
    lines.append("\n## Hardest/rarest strata\n")
    lines.append(safe_md(strata_summary.sort_values("count"), 30))
    lines.append("\n## Next step\n")
    lines.append("```text")
    lines.append("F3a1 will train one official L1 model per fold:")
    lines.append("  train_idx = fold != i")
    lines.append("  oof_idx   = fold == i")
    lines.append("Then export OOF logits/probs/CLS for train-only hard-pair mining.")
    lines.append("```")
    (out_dir / "F3a0_report.md").write_text("\n".join(lines), encoding="utf-8")

    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out_dir.rglob("*"):
            if p.is_file() and p != zip_path:
                z.write(p, p.relative_to(out_dir.parent))

    log(f"rows={n}")
    log(f"n_splits={int(args.n_splits)}")
    log(f"stratification_mode={stratification_mode}")
    log(f"fold_sizes={fold_sizes.to_dict()}")
    log(f"max_abs_l2_pct_diff={integrity['max_abs_l2_pct_diff']:.6f}")
    log(f"max_abs_l3_pct_diff={integrity['max_abs_l3_pct_diff']:.6f}")
    log(f"zip={zip_path}")
    log("DONE")


if __name__ == "__main__":
    main()
