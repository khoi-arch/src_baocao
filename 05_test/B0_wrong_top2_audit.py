#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B0_wrong_top2_audit.py

Phase B0 for official C2+D3 class-overlap workflow.

Goal:
    Among wrong validation predictions, check whether the true label is often
    still inside the model's top-2 candidates.

Default input path follows the verified src_baocao tree:
    03_outputs/00_data_for_test/01_model/predictions/val_predictions.csv

Default output:
    05_test/outputs/B0_wrong_top2_audit/

This script does not train, does not modify baseline files, and writes only to
05_test/outputs/B0_wrong_top2_audit.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_PRED_CSV = "03_outputs/00_data_for_test/01_model/predictions/val_predictions.csv"
DEFAULT_OUT_DIR = "05_test/outputs/B0_wrong_top2_audit"
DEFAULT_MODEL_CONFIG = "03_outputs/00_data_for_test/01_model/config.json"
DEFAULT_DATASET_METADATA = "03_outputs/05_dataset/metadata.json"

TRUE_COL_CANDIDATES = [
    "y_true", "true", "true_label", "label", "label_name", "target", "gt",
    "ground_truth", "actual", "actual_label", "y", "class_true", "class_label",
]

PRED_COL_CANDIDATES = [
    "y_pred", "pred", "pred_label", "prediction", "predicted", "predicted_label",
    "class_pred", "pred_class", "model_pred", "top1_pred", "top1_label",
]

TOP1_COL_CANDIDATES = [
    "top1", "top1_label", "top_1", "top_1_label", "pred_top1", "pred_top1_label",
    "top1_class", "top_1_class", "rank1", "rank1_label",
]

TOP2_COL_CANDIDATES = [
    "top2", "top2_label", "top_2", "top_2_label", "pred_top2", "pred_top2_label",
    "top2_class", "top_2_class", "rank2", "rank2_label",
]

SCORE_PREFIXES = [
    "prob_", "proba_", "probability_", "p_", "logit_", "score_", "conf_",
    "confidence_",
]

SCORE_SUFFIXES = [
    "_prob", "_proba", "_probability", "_logit", "_score", "_conf",
    "_confidence",
]

HARD_MALWARE_PAIRS = [
    ("Ransomware", "Spyware"),
    ("Ransomware", "Trojan"),
    ("Spyware", "Trojan"),
]

NON_SCORE_COL_HINTS = {
    "true", "pred", "label", "target", "gt", "actual", "index", "sample", "id",
    "correct", "wrong", "rank", "top", "margin", "loss",
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def norm_col(name: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def norm_label(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        # helpful for encoded integer labels loaded as floats
        try:
            s = str(int(float(s)))
        except Exception:
            pass
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def as_label(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        try:
            return str(int(float(s)))
        except Exception:
            return s
    return s


def read_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def extract_label_mapping(*objs: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Return encoded-id string -> label-name string when metadata/config exposes it."""
    mapping: Dict[str, str] = {}

    def visit(obj: Any) -> None:
        if isinstance(obj, dict):
            # common format: {"Benign": 0, ...}
            if "label_mapping" in obj and isinstance(obj["label_mapping"], dict):
                lm = obj["label_mapping"]
                for k, v in lm.items():
                    if isinstance(v, (int, np.integer)) or str(v).isdigit():
                        mapping[str(int(v))] = str(k)
                    else:
                        mapping[str(k)] = str(v)
            # common format: ["Benign", "Ransomware", ...]
            for key in ["label_names", "class_names", "classes", "target_names"]:
                if key in obj and isinstance(obj[key], list):
                    for i, name in enumerate(obj[key]):
                        mapping[str(i)] = str(name)
            for v in obj.values():
                visit(v)
        elif isinstance(obj, list):
            for v in obj:
                visit(v)

    for obj in objs:
        visit(obj)
    return mapping


def maybe_decode_labels(series: pd.Series, id_to_label: Dict[str, str]) -> pd.Series:
    if not id_to_label:
        return series.astype(str).map(as_label)
    decoded = []
    changed = False
    for x in series.tolist():
        raw = as_label(x)
        key = raw
        if key in id_to_label:
            decoded.append(id_to_label[key])
            changed = True
        else:
            decoded.append(raw)
    if changed:
        return pd.Series(decoded, index=series.index, dtype="object")
    return series.astype(str).map(as_label)


def find_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    cand_norm = [norm_col(c) for c in candidates]
    col_norm = {norm_col(c): c for c in df.columns}
    for cn in cand_norm:
        if cn in col_norm:
            return col_norm[cn]

    # soft contains matching, but avoid score columns
    for c in df.columns:
        nc = norm_col(c)
        for cn in cand_norm:
            if nc == cn or nc.endswith(cn) or nc.startswith(cn):
                if not any(nc.startswith(norm_col(p)) for p in SCORE_PREFIXES):
                    return c
    return None


def strip_score_label(col: str, mode: str, token: str) -> str:
    s = str(col)
    if mode == "prefix":
        return s[len(token):]
    if mode == "suffix":
        return s[:-len(token)]
    return s


def numeric_score_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def candidate_score_groups(df: pd.DataFrame, known_labels: Sequence[str]) -> List[Dict[str, Any]]:
    numeric_cols = set(numeric_score_columns(df))
    groups: List[Dict[str, Any]] = []
    known_norms = {norm_label(x) for x in known_labels if norm_label(x)}

    for pref in SCORE_PREFIXES:
        cols = [c for c in df.columns if c in numeric_cols and str(c).lower().startswith(pref)]
        if len(cols) >= 2:
            labels = [strip_score_label(c, "prefix", pref) for c in cols]
            overlap = len({norm_label(x) for x in labels} & known_norms) if known_norms else 0
            groups.append({
                "kind": "prefix",
                "token": pref,
                "columns": cols,
                "labels": labels,
                "overlap": overlap,
            })

    for suff in SCORE_SUFFIXES:
        cols = [c for c in df.columns if c in numeric_cols and str(c).lower().endswith(suff)]
        if len(cols) >= 2:
            labels = [strip_score_label(c, "suffix", suff) for c in cols]
            overlap = len({norm_label(x) for x in labels} & known_norms) if known_norms else 0
            groups.append({
                "kind": "suffix",
                "token": suff,
                "columns": cols,
                "labels": labels,
                "overlap": overlap,
            })

    # Last-resort heuristic: all numeric columns whose names resemble class labels.
    # Avoid using obvious non-score metrics.
    if known_norms:
        cols = []
        labels = []
        for c in numeric_cols:
            nc = norm_col(c)
            if any(h in nc for h in NON_SCORE_COL_HINTS):
                continue
            if nc in known_norms:
                cols.append(c)
                labels.append(str(c))
        if len(cols) >= 2:
            groups.append({
                "kind": "bare_label_numeric",
                "token": "",
                "columns": cols,
                "labels": labels,
                "overlap": len({norm_label(x) for x in labels} & known_norms),
            })

    # De-duplicate same column set.
    uniq = []
    seen = set()
    for g in groups:
        key = tuple(g["columns"])
        if key not in seen:
            uniq.append(g)
            seen.add(key)

    # Prefer groups with known-label overlap, then number of columns.
    uniq.sort(key=lambda g: (g["overlap"], len(g["columns"])), reverse=True)
    return uniq


def compute_top2_from_scores(df: pd.DataFrame, score_cols: Sequence[str], score_labels: Sequence[str]) -> pd.DataFrame:
    scores = df.loc[:, list(score_cols)].to_numpy(dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError("Need at least two score columns to compute top-2.")
    labels = [as_label(x) for x in score_labels]
    order = np.argsort(-scores, axis=1)
    top1_idx = order[:, 0]
    top2_idx = order[:, 1]

    return pd.DataFrame({
        "top1_label": [labels[i] for i in top1_idx],
        "top2_label": [labels[i] for i in top2_idx],
        "top1_score": scores[np.arange(scores.shape[0]), top1_idx],
        "top2_score": scores[np.arange(scores.shape[0]), top2_idx],
        "top12_margin": scores[np.arange(scores.shape[0]), top1_idx] - scores[np.arange(scores.shape[0]), top2_idx],
    }, index=df.index)


def compute_top2_from_columns(df: pd.DataFrame, top1_col: str, top2_col: str) -> pd.DataFrame:
    out = pd.DataFrame({
        "top1_label": df[top1_col].map(as_label),
        "top2_label": df[top2_col].map(as_label),
    }, index=df.index)
    out["top1_score"] = np.nan
    out["top2_score"] = np.nan
    out["top12_margin"] = np.nan
    return out


def safe_rate(num: int, den: int) -> float:
    if den <= 0:
        return float("nan")
    return float(num / den)


def find_existing_index_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ["sample_id", "id", "index", "row_id", "idx", "original_index"]
    return find_column(df, candidates)


def summarize_by_true(audit_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for true_label, g in audit_df.groupby("true_label", dropna=False):
        n_total = int(len(g))
        n_correct = int(g["is_correct"].sum())
        wrong = g[~g["is_correct"]]
        n_wrong = int(len(wrong))
        wrong_true_in_top2 = int(wrong["true_in_top2"].sum())
        rows.append({
            "true_class": true_label,
            "n_total": n_total,
            "n_correct": n_correct,
            "n_wrong": n_wrong,
            "wrong_true_in_top2": wrong_true_in_top2,
            "wrong_true_in_top2_rate": safe_rate(wrong_true_in_top2, n_wrong),
            "class_error_rate": safe_rate(n_wrong, n_total),
        })
    return pd.DataFrame(rows).sort_values(["n_wrong", "true_class"], ascending=[False, True])


def summarize_by_confusion_pair(audit_df: pd.DataFrame) -> pd.DataFrame:
    wrong = audit_df[~audit_df["is_correct"]].copy()
    if wrong.empty:
        return pd.DataFrame(columns=[
            "true_class", "pred_class", "n_wrong", "wrong_true_in_top2", "wrong_true_in_top2_rate"
        ])
    rows = []
    for (true_label, pred_label), g in wrong.groupby(["true_label", "pred_label"], dropna=False):
        n_wrong = int(len(g))
        true_in_top2 = int(g["true_in_top2"].sum())
        rows.append({
            "true_class": true_label,
            "pred_class": pred_label,
            "n_wrong": n_wrong,
            "wrong_true_in_top2": true_in_top2,
            "wrong_true_in_top2_rate": safe_rate(true_in_top2, n_wrong),
        })
    return pd.DataFrame(rows).sort_values(["n_wrong", "true_class", "pred_class"], ascending=[False, True, True])


def resolve_class_label(labels: Iterable[str], target: str) -> Optional[str]:
    target_norm = norm_label(target)
    labels_list = [as_label(x) for x in labels]
    for x in labels_list:
        if norm_label(x) == target_norm:
            return x
    for x in labels_list:
        nx = norm_label(x)
        if target_norm in nx or nx in target_norm:
            return x
    return None


def summarize_hard_pairs(audit_df: pd.DataFrame) -> pd.DataFrame:
    all_labels = sorted(set(audit_df["true_label"].map(as_label)) | set(audit_df["pred_label"].map(as_label)))
    resolved = {target: resolve_class_label(all_labels, target) for pair in HARD_MALWARE_PAIRS for target in pair}

    rows = []
    wrong = audit_df[~audit_df["is_correct"]].copy()
    for a, b in HARD_MALWARE_PAIRS:
        la = resolved.get(a)
        lb = resolved.get(b)
        pair_name = f"{a}<->{b}"
        for true_target, pred_target, true_label, pred_label in [
            (a, b, la, lb),
            (b, a, lb, la),
        ]:
            if true_label is None or pred_label is None:
                rows.append({
                    "pair": pair_name,
                    "direction": f"{true_target}->{pred_target}",
                    "resolved_true_label": true_label,
                    "resolved_pred_label": pred_label,
                    "n_wrong": 0,
                    "wrong_true_in_top2": 0,
                    "wrong_true_in_top2_rate": np.nan,
                    "note": "missing_resolved_label_in_predictions",
                })
                continue
            g = wrong[(wrong["true_label_norm"] == norm_label(true_label)) & (wrong["pred_label_norm"] == norm_label(pred_label))]
            n_wrong = int(len(g))
            true_in_top2 = int(g["true_in_top2"].sum())
            rows.append({
                "pair": pair_name,
                "direction": f"{true_label}->{pred_label}",
                "resolved_true_label": true_label,
                "resolved_pred_label": pred_label,
                "n_wrong": n_wrong,
                "wrong_true_in_top2": true_in_top2,
                "wrong_true_in_top2_rate": safe_rate(true_in_top2, n_wrong),
                "note": "ok",
            })
    return pd.DataFrame(rows)


def write_json(obj: Dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def make_markdown_summary(
    *,
    metrics: Dict[str, Any],
    by_true: pd.DataFrame,
    by_pair: pd.DataFrame,
    hard_pairs: pd.DataFrame,
    manifest: Dict[str, Any],
) -> str:
    lines = []
    lines.append("# B0 — Wrong-sample top-2 coverage audit")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("Check whether the true label is still inside the model's top-2 candidates among validation samples predicted incorrectly by the official C2+D3 baseline.")
    lines.append("")
    lines.append("## Input")
    lines.append("")
    lines.append(f"- prediction CSV: `{manifest.get('pred_csv')}`")
    lines.append(f"- true label column: `{manifest.get('true_col')}`")
    lines.append(f"- predicted label column: `{manifest.get('pred_col')}`")
    lines.append(f"- top-2 source: `{manifest.get('top2_source')}`")
    if manifest.get("score_columns"):
        lines.append(f"- score columns: `{manifest.get('score_columns')}`")
    lines.append("")
    lines.append("## Main metrics")
    lines.append("")
    for key in [
        "n_total", "n_correct", "n_wrong", "accuracy_from_predictions",
        "wrong_true_in_top2", "wrong_true_in_top2_rate",
    ]:
        val = metrics.get(key)
        if isinstance(val, float):
            lines.append(f"- `{key}`: {val:.6f}")
        else:
            lines.append(f"- `{key}`: {val}")
    lines.append("")
    lines.append("## Interpretation gate")
    lines.append("")
    rate = metrics.get("wrong_true_in_top2_rate")
    try:
        rate_f = float(rate)
    except Exception:
        rate_f = float("nan")
    if not math.isnan(rate_f):
        if rate_f >= 0.60:
            lines.append("- Result: high wrong-sample top-2 coverage. Pairwise/reranking/auxiliary-head directions are worth testing next.")
        elif rate_f >= 0.35:
            lines.append("- Result: moderate wrong-sample top-2 coverage. Pairwise methods may help only on selected confusion pairs; inspect pair-level output before continuing.")
        else:
            lines.append("- Result: low wrong-sample top-2 coverage. Pure top-2 reranking is unlikely to solve the main error; representation/training changes should be prioritized.")
    else:
        lines.append("- Result: cannot compute gate because top-2 coverage is NaN.")
    lines.append("")
    lines.append("## Wrong top-2 coverage by true class")
    lines.append("")
    lines.append(by_true.to_markdown(index=False))
    lines.append("")
    lines.append("## Top confusion pairs")
    lines.append("")
    top_pair = by_pair.head(20) if not by_pair.empty else by_pair
    lines.append(top_pair.to_markdown(index=False))
    lines.append("")
    lines.append("## Hard malware pairs")
    lines.append("")
    lines.append(hard_pairs.to_markdown(index=False))
    lines.append("")
    lines.append("## Generated files")
    lines.append("")
    for p in manifest.get("outputs", []):
        lines.append(f"- `{p}`")
    lines.append("")
    return "\n".join(lines)


def zip_output(out_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(out_dir.rglob("*")):
            if fp.is_file() and fp.resolve() != zip_path.resolve():
                zf.write(fp, fp.relative_to(out_dir.parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="B0 wrong-sample top-2 audit for official C2+D3 baseline.")
    p.add_argument("--repo-root", default=".", help="Repository root. Default: current directory.")
    p.add_argument("--pred-csv", default=DEFAULT_PRED_CSV, help="Validation prediction CSV path, relative to repo root unless absolute.")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory, relative to repo root unless absolute.")
    p.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG, help="Optional model config JSON for label mapping.")
    p.add_argument("--dataset-metadata", default=DEFAULT_DATASET_METADATA, help="Optional dataset metadata JSON for label mapping.")
    p.add_argument("--true-col", default="", help="Override true-label column name.")
    p.add_argument("--pred-col", default="", help="Override predicted-label column name.")
    p.add_argument("--top1-col", default="", help="Override top-1 label column name.")
    p.add_argument("--top2-col", default="", help="Override top-2 label column name.")
    p.add_argument("--score-prefix", default="", help="Force score/prob/logit columns by prefix, e.g. prob_.")
    return p.parse_args()


def resolve_path(repo_root: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return repo_root / p


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    pred_csv = resolve_path(repo_root, args.pred_csv)
    out_dir = resolve_path(repo_root, args.out_dir)
    model_config_path = resolve_path(repo_root, args.model_config)
    dataset_metadata_path = resolve_path(repo_root, args.dataset_metadata)

    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "stage": "B0_wrong_top2_audit",
        "created_at": now_iso(),
        "repo_root": str(repo_root),
        "pred_csv": str(pred_csv),
        "out_dir": str(out_dir),
        "script": str(Path(__file__).resolve()),
        "status": "started",
    }

    if not pred_csv.exists():
        manifest["status"] = "failed"
        manifest["error"] = f"Prediction CSV not found: {pred_csv}"
        write_json(manifest, out_dir / "B0_run_manifest.json")
        print(manifest["error"], file=sys.stderr)
        return 2

    df = pd.read_csv(pred_csv)
    manifest["n_rows_loaded"] = int(len(df))
    manifest["columns"] = [str(c) for c in df.columns]

    model_config = read_json_if_exists(model_config_path)
    dataset_metadata = read_json_if_exists(dataset_metadata_path)
    id_to_label = extract_label_mapping(model_config, dataset_metadata)
    manifest["label_id_to_name_mapping_detected"] = id_to_label

    true_col = args.true_col or find_column(df, TRUE_COL_CANDIDATES)
    if not true_col:
        manifest["status"] = "failed"
        manifest["error"] = "Could not detect true-label column. Use --true-col."
        write_json(manifest, out_dir / "B0_run_manifest.json")
        print(manifest["error"], file=sys.stderr)
        return 2

    pred_col = args.pred_col or find_column(df, PRED_COL_CANDIDATES)

    y_true = maybe_decode_labels(df[true_col], id_to_label)
    known_labels = sorted(set(y_true.map(as_label).tolist()))
    if pred_col:
        y_pred_initial = maybe_decode_labels(df[pred_col], id_to_label)
        known_labels = sorted(set(known_labels) | set(y_pred_initial.map(as_label).tolist()))
    else:
        y_pred_initial = None

    # Top-2 detection order:
    # 1) explicit top1/top2 columns
    # 2) score/prob/logit columns
    top2_info: Optional[pd.DataFrame] = None
    top2_source = ""
    score_columns: List[str] = []
    score_labels: List[str] = []

    top1_col = args.top1_col or find_column(df, TOP1_COL_CANDIDATES)
    top2_col = args.top2_col or find_column(df, TOP2_COL_CANDIDATES)
    if top1_col and top2_col and top1_col != top2_col:
        top2_info = compute_top2_from_columns(df, top1_col, top2_col)
        top2_source = f"explicit_top_columns:{top1_col},{top2_col}"
    else:
        groups = []
        if args.score_prefix:
            forced_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and str(c).lower().startswith(args.score_prefix.lower())]
            if len(forced_cols) >= 2:
                groups = [{
                    "kind": "forced_prefix",
                    "token": args.score_prefix,
                    "columns": forced_cols,
                    "labels": [strip_score_label(c, "prefix", args.score_prefix) for c in forced_cols],
                    "overlap": 999,
                }]
        if not groups:
            groups = candidate_score_groups(df, known_labels)
        if groups:
            g = groups[0]
            score_columns = [str(c) for c in g["columns"]]
            score_labels = [as_label(x) for x in g["labels"]]
            top2_info = compute_top2_from_scores(df, score_columns, score_labels)
            top2_source = f"score_columns:{g['kind']}:{g['token']}"
            manifest["candidate_score_groups"] = [
                {k: v for k, v in gg.items() if k in ["kind", "token", "columns", "labels", "overlap"]}
                for gg in groups[:5]
            ]

    if top2_info is None:
        manifest["status"] = "failed"
        manifest["true_col"] = true_col
        manifest["pred_col"] = pred_col
        manifest["top2_source"] = "not_found"
        manifest["error"] = (
            "Could not detect top-2 information. Need probability/logit/score columns "
            "or explicit top1/top2 label columns. Use --top1-col/--top2-col or --score-prefix."
        )
        write_json(manifest, out_dir / "B0_run_manifest.json")
        print(manifest["error"], file=sys.stderr)
        return 2

    if pred_col and y_pred_initial is not None:
        y_pred = y_pred_initial
    else:
        y_pred = maybe_decode_labels(top2_info["top1_label"], id_to_label)
        pred_col = "<computed_from_top1>"

    audit_df = pd.DataFrame(index=df.index)
    idx_col = find_existing_index_column(df)
    if idx_col:
        audit_df["sample_id"] = df[idx_col]
    else:
        audit_df["sample_id"] = df.index.astype(int)

    audit_df["row_index"] = df.index.astype(int)
    audit_df["true_label"] = y_true.map(as_label)
    audit_df["pred_label"] = y_pred.map(as_label)
    audit_df["top1_label"] = top2_info["top1_label"].map(as_label)
    audit_df["top2_label"] = top2_info["top2_label"].map(as_label)
    audit_df["top1_score"] = top2_info["top1_score"]
    audit_df["top2_score"] = top2_info["top2_score"]
    audit_df["top12_margin"] = top2_info["top12_margin"]

    audit_df["true_label_norm"] = audit_df["true_label"].map(norm_label)
    audit_df["pred_label_norm"] = audit_df["pred_label"].map(norm_label)
    audit_df["top1_label_norm"] = audit_df["top1_label"].map(norm_label)
    audit_df["top2_label_norm"] = audit_df["top2_label"].map(norm_label)

    audit_df["is_correct"] = audit_df["true_label_norm"] == audit_df["pred_label_norm"]
    audit_df["true_in_top2"] = (
        (audit_df["true_label_norm"] == audit_df["top1_label_norm"]) |
        (audit_df["true_label_norm"] == audit_df["top2_label_norm"])
    )
    audit_df["wrong_true_in_top2"] = (~audit_df["is_correct"]) & audit_df["true_in_top2"]

    n_total = int(len(audit_df))
    n_correct = int(audit_df["is_correct"].sum())
    n_wrong = int(n_total - n_correct)
    wrong_true_in_top2 = int(audit_df["wrong_true_in_top2"].sum())

    metrics: Dict[str, Any] = {
        "n_total": n_total,
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "accuracy_from_predictions": safe_rate(n_correct, n_total),
        "wrong_true_in_top2": wrong_true_in_top2,
        "wrong_true_in_top2_rate": safe_rate(wrong_true_in_top2, n_wrong),
        "top2_source": top2_source,
        "true_col": true_col,
        "pred_col": pred_col,
    }

    by_true = summarize_by_true(audit_df)
    by_pair = summarize_by_confusion_pair(audit_df)
    hard_pairs = summarize_hard_pairs(audit_df)
    wrong_samples = audit_df[~audit_df["is_correct"]].copy()
    wrong_samples = wrong_samples.sort_values(["wrong_true_in_top2", "true_label", "pred_label", "row_index"], ascending=[False, True, True, True])

    outputs = {
        "summary_md": out_dir / "B0_summary.md",
        "metrics_json": out_dir / "B0_metrics.json",
        "wrong_by_true_class_csv": out_dir / "B0_wrong_by_true_class.csv",
        "wrong_by_confusion_pair_csv": out_dir / "B0_wrong_by_confusion_pair.csv",
        "hard_malware_pairs_csv": out_dir / "B0_hard_malware_pairs.csv",
        "wrong_samples_top2_csv": out_dir / "B0_wrong_samples_top2.csv",
        "run_manifest_json": out_dir / "B0_run_manifest.json",
        "zip": out_dir / "B0_wrong_top2_audit_output.zip",
    }

    manifest.update({
        "status": "success",
        "true_col": true_col,
        "pred_col": pred_col,
        "top2_source": top2_source,
        "score_columns": score_columns,
        "score_labels": score_labels,
        "outputs": [str(p) for k, p in outputs.items() if k != "zip"],
    })

    write_json(metrics, outputs["metrics_json"])
    by_true.to_csv(outputs["wrong_by_true_class_csv"], index=False)
    by_pair.to_csv(outputs["wrong_by_confusion_pair_csv"], index=False)
    hard_pairs.to_csv(outputs["hard_malware_pairs_csv"], index=False)
    wrong_samples.to_csv(outputs["wrong_samples_top2_csv"], index=False)

    summary_md = make_markdown_summary(
        metrics=metrics,
        by_true=by_true,
        by_pair=by_pair,
        hard_pairs=hard_pairs,
        manifest=manifest,
    )
    outputs["summary_md"].write_text(summary_md, encoding="utf-8")
    write_json(manifest, outputs["run_manifest_json"])
    zip_output(out_dir, outputs["zip"])

    print("===== B0 wrong-sample top-2 audit done =====")
    print(f"prediction_csv: {pred_csv}")
    print(f"true_col: {true_col}")
    print(f"pred_col: {pred_col}")
    print(f"top2_source: {top2_source}")
    print(f"n_total: {n_total}")
    print(f"n_wrong: {n_wrong}")
    print(f"wrong_true_in_top2: {wrong_true_in_top2}")
    print(f"wrong_true_in_top2_rate: {metrics['wrong_true_in_top2_rate']:.6f}")
    print(f"output_dir: {out_dir}")
    print(f"zip: {outputs['zip']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
