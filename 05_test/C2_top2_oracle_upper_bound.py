#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C2_top2_oracle_upper_bound.py

Purpose
-------
Phase C2 diagnostic only.

Estimate upper bounds for top-2 based correction on the official C2+D3 baseline:

  C2.1 Oracle top-2 upper bound:
       If every wrong sample whose true label is in top-2 were fixed perfectly,
       how much would validation metrics improve?

  C2.2 Margin-bounded oracle:
       If only wrong-top2 samples with pred-vs-true probability margin <= T were
       fixed perfectly, how much would validation metrics improve?

This script DOES NOT train a reranker and DOES NOT modify model outputs used by
the official baseline. It only creates simulated diagnostic predictions under
oracle assumptions.

Recommended input
-----------------
Use the probability CSV exported from the baseline under analysis:

  05_test/outputs/B0_wrong_top2_audit/val_predictions_with_probs.csv

Outputs
-------
  C2_summary.md
  C2_policy_metrics.csv
  C2_policy_per_class_f1.csv
  C2_margin_threshold_by_pair.csv
  C2_gate_decision.json
  C2_top2_oracle_upper_bound_output.zip
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_HARD_PAIRS = [
    ("Ransomware", "Spyware"),
    ("Ransomware", "Trojan"),
    ("Spyware", "Trojan"),
]

DEFAULT_THRESHOLDS = [0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00]
DEFAULT_LABEL_ORDER = ["Benign", "Ransomware", "Spyware", "Trojan"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="C2 top-2 oracle and margin-bounded upper-bound audit.")
    p.add_argument("--repo-root", default=".")
    p.add_argument("--pred-csv", default="05_test/outputs/B0_wrong_top2_audit/val_predictions_with_probs.csv")
    p.add_argument("--out-dir", default="05_test/outputs/C2_top2_oracle_upper_bound")
    p.add_argument("--score-prefix", default="prob_")
    p.add_argument("--thresholds", nargs="*", type=float, default=DEFAULT_THRESHOLDS)
    p.add_argument("--label-order", nargs="*", default=DEFAULT_LABEL_ORDER)
    p.add_argument("--pairs", nargs="*", default=None,
                   help='Optional hard pairs as "A:B", e.g. "Ransomware:Trojan". Default malware pairs.')
    return p.parse_args()


def repo_path(repo_root: Path, path_like: str | Path) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    return repo_root / p


def normalize_label(x: Any) -> str:
    return str(x).strip()


def parse_pairs(pair_args: List[str] | None) -> List[Tuple[str, str]]:
    if not pair_args:
        return DEFAULT_HARD_PAIRS
    pairs: List[Tuple[str, str]] = []
    for item in pair_args:
        if ":" not in item:
            raise ValueError(f"Invalid pair format {item!r}; expected A:B")
        a, b = item.split(":", 1)
        pairs.append((normalize_label(a), normalize_label(b)))
    return pairs


def zip_outputs(out_dir: Path, zip_name: str = "C2_top2_oracle_upper_bound_output.zip") -> Path:
    out_zip = out_dir / zip_name
    if out_zip.exists():
        out_zip.unlink()

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fp in out_dir.rglob("*"):
            if fp.is_file() and fp != out_zip:
                z.write(fp, fp.relative_to(out_dir))
    return out_zip


def ensure_base_columns(df: pd.DataFrame) -> None:
    required = ["true_label", "pred_label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Prediction CSV missing required columns: {missing}")


def safe_label_col(label: str) -> str:
    return normalize_label(label).replace(" ", "_")


def prob_col_for_label(df: pd.DataFrame, label: str, score_prefix: str) -> str:
    candidates = [
        f"{score_prefix}{label}",
        f"{score_prefix}{safe_label_col(label)}",
    ]
    for c in candidates:
        if c in df.columns:
            return c

    target = safe_label_col(label).lower()
    for c in df.columns:
        if not c.startswith(score_prefix):
            continue
        tail = c[len(score_prefix):]
        if safe_label_col(tail).lower() == target:
            return c

    raise KeyError(f"Cannot find probability column for label={label!r}; tried {candidates}")


def infer_label_order(df: pd.DataFrame, user_order: List[str], score_prefix: str) -> List[str]:
    observed = []
    for c in ["true_label", "pred_label", "top1_label", "top2_label"]:
        if c in df.columns:
            for x in df[c].dropna().map(normalize_label).unique().tolist():
                if x not in observed:
                    observed.append(x)

    prob_labels = []
    for c in df.columns:
        if c.startswith(score_prefix):
            label = c[len(score_prefix):].replace("_", " ")
            if label not in prob_labels:
                prob_labels.append(label)

    labels: List[str] = []
    for x in user_order:
        x = normalize_label(x)
        if x in observed or x in prob_labels:
            labels.append(x)

    for x in observed + prob_labels:
        if x not in labels:
            labels.append(x)

    return labels


def build_augmented_df(df0: pd.DataFrame, labels: List[str], score_prefix: str) -> pd.DataFrame:
    df = df0.copy()
    df["true_label"] = df["true_label"].map(normalize_label)
    df["pred_label"] = df["pred_label"].map(normalize_label)

    prob_cols = {label: prob_col_for_label(df, label, score_prefix) for label in labels}
    probs = df[[prob_cols[label] for label in labels]].to_numpy(dtype=np.float64)

    if not np.isfinite(probs).all():
        raise ValueError("Probability matrix contains NaN/Inf")

    label_to_id = {label: i for i, label in enumerate(labels)}
    true_id = np.array([label_to_id[x] for x in df["true_label"]], dtype=int)
    pred_id = np.array([label_to_id[x] for x in df["pred_label"]], dtype=int)

    order = np.argsort(-probs, axis=1)
    top1_id = order[:, 0]
    top2_id = order[:, 1]
    idx = np.arange(len(df))

    top1_score = probs[idx, top1_id]
    top2_score = probs[idx, top2_id]
    true_prob = probs[idx, true_id]
    pred_prob = probs[idx, pred_id]

    correct = true_id == pred_id
    true_in_top2 = (true_id == top1_id) | (true_id == top2_id)

    # Use existing sample_index if present, otherwise create it.
    if "sample_index" not in df.columns:
        df["sample_index"] = idx

    df["true_id_computed"] = true_id
    df["pred_id_computed"] = pred_id
    df["top1_id_computed"] = top1_id
    df["top2_id_computed"] = top2_id
    df["top1_label_computed"] = [labels[i] for i in top1_id]
    df["top2_label_computed"] = [labels[i] for i in top2_id]
    df["top1_score_computed"] = top1_score
    df["top2_score_computed"] = top2_score
    df["top12_margin"] = top1_score - top2_score
    df["true_prob"] = true_prob
    df["pred_prob"] = pred_prob
    df["pred_minus_true_prob"] = pred_prob - true_prob
    df["correct_computed"] = correct
    df["wrong"] = ~correct
    df["true_in_top2_computed"] = true_in_top2
    df["wrong_true_in_top2"] = (~correct) & true_in_top2

    # true rank.
    true_rank = np.empty(len(df), dtype=int)
    for i in range(len(df)):
        true_rank[i] = int(np.where(order[i] == true_id[i])[0][0]) + 1
    df["true_rank"] = true_rank

    return df


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> Tuple[Dict[str, Any], Dict[str, float]]:
    try:
        from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("C2 requires scikit-learn. Install with: pip install scikit-learn") from e

    metric = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
    }

    p, r, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    per_class = {f"f1_{label}": float(v) for label, v in zip(labels, f1)}
    per_class.update({f"recall_{label}": float(v) for label, v in zip(labels, r)})
    per_class.update({f"precision_{label}": float(v) for label, v in zip(labels, p)})
    per_class.update({f"support_{label}": int(v) for label, v in zip(labels, support)})

    return metric, per_class


def simulate_policy(df: pd.DataFrame, policy_name: str, correction_mask: np.ndarray) -> pd.Series:
    pred = df["pred_label"].copy()
    pred.loc[correction_mask] = df.loc[correction_mask, "true_label"]
    pred.name = policy_name
    return pred


def make_policy_metrics(df: pd.DataFrame, labels: List[str], thresholds: List[float], pairs: List[Tuple[str, str]]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y_true = df["true_label"].to_numpy()
    original_pred = df["pred_label"].to_numpy()

    policy_rows: List[Dict[str, Any]] = []
    per_class_rows: List[Dict[str, Any]] = []

    original_metrics, original_per_class = compute_metrics(y_true, original_pred, labels)
    original_correct = y_true == original_pred

    def add_policy(policy_name: str, corrected_pred: pd.Series, correction_mask: np.ndarray, policy_type: str, threshold: float | None = None):
        y_pred = corrected_pred.to_numpy()
        metrics, per_class = compute_metrics(y_true, y_pred, labels)
        correct = y_true == y_pred

        row: Dict[str, Any] = {
            "policy": policy_name,
            "policy_type": policy_type,
            "margin_threshold": threshold,
            "n_corrected_samples": int(correction_mask.sum()),
            "n_total": int(len(df)),
            **metrics,
            "delta_accuracy": float(metrics["accuracy"] - original_metrics["accuracy"]),
            "delta_macro_f1": float(metrics["macro_f1"] - original_metrics["macro_f1"]),
            "delta_weighted_f1": float(metrics["weighted_f1"] - original_metrics["weighted_f1"]),
            "newly_correct_n": int((correct & ~original_correct).sum()),
            "newly_wrong_n": int((~correct & original_correct).sum()),
        }
        policy_rows.append(row)

        class_row = {
            "policy": policy_name,
            "policy_type": policy_type,
            "margin_threshold": threshold,
            **per_class,
        }
        for label in labels:
            class_row[f"delta_f1_{label}"] = float(per_class[f"f1_{label}"] - original_per_class[f"f1_{label}"])
            class_row[f"delta_recall_{label}"] = float(per_class[f"recall_{label}"] - original_per_class[f"recall_{label}"])
        per_class_rows.append(class_row)

    # Original.
    add_policy(
        "original",
        df["pred_label"],
        np.zeros(len(df), dtype=bool),
        "baseline",
        None,
    )

    # Oracle all wrong top2.
    all_top2_mask = df["wrong_true_in_top2"].to_numpy(dtype=bool)
    add_policy(
        "oracle_all_wrong_true_in_top2",
        simulate_policy(df, "oracle_all_wrong_true_in_top2", all_top2_mask),
        all_top2_mask,
        "top2_oracle",
        None,
    )

    # Margin-bounded oracle.
    for t in thresholds:
        mask = (df["wrong_true_in_top2"].to_numpy(dtype=bool)) & (df["pred_minus_true_prob"].to_numpy(dtype=float) <= float(t))
        add_policy(
            f"oracle_wrong_top2_margin_le_{t:g}",
            simulate_policy(df, f"oracle_wrong_top2_margin_le_{t:g}", mask),
            mask,
            "margin_bounded_top2_oracle",
            float(t),
        )

    policy_df = pd.DataFrame(policy_rows)
    per_class_df = pd.DataFrame(per_class_rows)

    # Pair-level corrected sample counts for oracle/margin policies.
    pair_rows: List[Dict[str, Any]] = []
    for a, b in pairs:
        for direction_true, direction_pred in [(a, b), (b, a)]:
            base_mask = (df["true_label"] == direction_true) & (df["pred_label"] == direction_pred)
            wt2_mask = base_mask & df["wrong_true_in_top2"]

            row = {
                "pair": f"{a}<->{b}",
                "direction": f"{direction_true}->{direction_pred}",
                "n_wrong": int(base_mask.sum()),
                "n_wrong_true_in_top2": int(wt2_mask.sum()),
                "wrong_true_in_top2_rate": float(wt2_mask.sum() / base_mask.sum()) if int(base_mask.sum()) else float("nan"),
                "oracle_all_correctable_n": int(wt2_mask.sum()),
            }

            for t in thresholds:
                tm = wt2_mask & (df["pred_minus_true_prob"] <= float(t))
                row[f"correctable_margin_le_{t:g}_n"] = int(tm.sum())
                row[f"correctable_margin_le_{t:g}_rate_among_wrong"] = float(tm.sum() / base_mask.sum()) if int(base_mask.sum()) else float("nan")
                row[f"correctable_margin_le_{t:g}_rate_among_wrong_top2"] = float(tm.sum() / wt2_mask.sum()) if int(wt2_mask.sum()) else float("nan")

            pair_rows.append(row)

        direct_mask = (
            ((df["true_label"] == a) & (df["pred_label"] == b)) |
            ((df["true_label"] == b) & (df["pred_label"] == a))
        )
        direct_wt2 = direct_mask & df["wrong_true_in_top2"]
        row = {
            "pair": f"{a}<->{b}",
            "direction": "BIDIRECTIONAL_DIRECT_CONFUSION",
            "n_wrong": int(direct_mask.sum()),
            "n_wrong_true_in_top2": int(direct_wt2.sum()),
            "wrong_true_in_top2_rate": float(direct_wt2.sum() / direct_mask.sum()) if int(direct_mask.sum()) else float("nan"),
            "oracle_all_correctable_n": int(direct_wt2.sum()),
        }
        for t in thresholds:
            tm = direct_wt2 & (df["pred_minus_true_prob"] <= float(t))
            row[f"correctable_margin_le_{t:g}_n"] = int(tm.sum())
            row[f"correctable_margin_le_{t:g}_rate_among_wrong"] = float(tm.sum() / direct_mask.sum()) if int(direct_mask.sum()) else float("nan")
            row[f"correctable_margin_le_{t:g}_rate_among_wrong_top2"] = float(tm.sum() / direct_wt2.sum()) if int(direct_wt2.sum()) else float("nan")
        pair_rows.append(row)

    pair_df = pd.DataFrame(pair_rows)

    return policy_df, per_class_df, pair_df


def make_gate(policy_df: pd.DataFrame) -> Dict[str, Any]:
    original = policy_df[policy_df["policy"].eq("original")].iloc[0]
    oracle = policy_df[policy_df["policy"].eq("oracle_all_wrong_true_in_top2")].iloc[0]

    margin_policies = policy_df[policy_df["policy_type"].eq("margin_bounded_top2_oracle")].copy()
    # Important rows if present.
    row_02 = margin_policies[margin_policies["margin_threshold"].eq(0.2)]
    row_03 = margin_policies[margin_policies["margin_threshold"].eq(0.3)]
    row_05 = margin_policies[margin_policies["margin_threshold"].eq(0.5)]

    delta_oracle_macro = float(oracle["delta_macro_f1"])
    oracle_macro = float(oracle["macro_f1"])
    oracle_acc = float(oracle["accuracy"])

    delta_02 = float(row_02.iloc[0]["delta_macro_f1"]) if len(row_02) else float("nan")
    delta_03 = float(row_03.iloc[0]["delta_macro_f1"]) if len(row_03) else float("nan")
    delta_05 = float(row_05.iloc[0]["delta_macro_f1"]) if len(row_05) else float("nan")

    # Diagnostic gate.
    if delta_oracle_macro >= 0.08 and (np.isnan(delta_02) or delta_02 < delta_oracle_macro * 0.45):
        result = "MIXED — large oracle headroom, but simple margin-bounded correction captures limited gain"
        reason = (
            f"All-top2 oracle macro-F1={oracle_macro:.4f} (delta={delta_oracle_macro:.4f}), "
            f"but margin<=0.2 oracle delta_macro_f1={delta_02:.4f}. "
            "This means recoverable errors exist, but many require more than a small-margin rule."
        )
        recommendation = (
            "Proceed to C3 external CLS/logit pairwise reranker diagnostic. Do not rely on a simple margin-threshold rule."
        )
    elif delta_oracle_macro >= 0.08:
        result = "PASS — top-2 correction has strong oracle headroom"
        reason = (
            f"All-top2 oracle macro-F1={oracle_macro:.4f} (delta={delta_oracle_macro:.4f}), "
            f"and margin-bounded oracle captures a meaningful portion."
        )
        recommendation = "Proceed to C3 external reranker diagnostic."
    else:
        result = "FAIL — top-2 correction headroom is limited"
        reason = (
            f"All-top2 oracle delta_macro_f1={delta_oracle_macro:.4f}, which is small. "
            "Top-2 correction is unlikely to be the main path."
        )
        recommendation = "Consider different diagnosis before reranker work."

    return {
        "result": result,
        "reason": reason,
        "recommendation": recommendation,
        "original_accuracy": float(original["accuracy"]),
        "original_macro_f1": float(original["macro_f1"]),
        "oracle_all_top2_accuracy": oracle_acc,
        "oracle_all_top2_macro_f1": oracle_macro,
        "oracle_all_top2_delta_accuracy": float(oracle["delta_accuracy"]),
        "oracle_all_top2_delta_macro_f1": delta_oracle_macro,
        "margin_le_0.2_delta_macro_f1": delta_02,
        "margin_le_0.3_delta_macro_f1": delta_03,
        "margin_le_0.5_delta_macro_f1": delta_05,
        "guardrail": "C2 is an oracle simulation only; it is not a deployable reranker and does not change baseline files.",
    }


def to_md(df: pd.DataFrame, index: bool = False) -> str:
    try:
        return df.to_markdown(index=index)
    except Exception:
        return df.to_string(index=index)


def make_markdown(
    *,
    labels: List[str],
    policy_df: pd.DataFrame,
    per_class_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    gate: Dict[str, Any],
    out_files: List[Path],
) -> str:
    lines: List[str] = []
    lines.append("# C2 — Top-2 oracle upper-bound audit")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("Estimate upper bounds for top-2 correction before training any reranker or changing the model.")
    lines.append("")
    lines.append("## Interpretation gate")
    lines.append("")
    lines.append(f"- Result: **{gate['result']}**")
    lines.append(f"- Reason: {gate['reason']}")
    lines.append(f"- Recommendation: {gate['recommendation']}")
    lines.append("")
    lines.append("## Key baseline/oracle metrics")
    lines.append("")
    key_rows = policy_df[
        policy_df["policy"].isin([
            "original",
            "oracle_all_wrong_true_in_top2",
            "oracle_wrong_top2_margin_le_0.1",
            "oracle_wrong_top2_margin_le_0.2",
            "oracle_wrong_top2_margin_le_0.3",
            "oracle_wrong_top2_margin_le_0.5",
        ])
    ].copy()
    show_cols = [
        "policy", "n_corrected_samples",
        "accuracy", "delta_accuracy",
        "macro_f1", "delta_macro_f1",
        "weighted_f1", "delta_weighted_f1",
        "newly_correct_n", "newly_wrong_n",
    ]
    show_cols = [c for c in show_cols if c in key_rows.columns]
    lines.append(to_md(key_rows[show_cols], index=False))
    lines.append("")
    lines.append("## Per-class F1 under important policies")
    lines.append("")
    pc = per_class_df[
        per_class_df["policy"].isin([
            "original",
            "oracle_all_wrong_true_in_top2",
            "oracle_wrong_top2_margin_le_0.2",
            "oracle_wrong_top2_margin_le_0.5",
        ])
    ].copy()
    pc_cols = ["policy"] + [f"f1_{label}" for label in labels] + [f"delta_f1_{label}" for label in labels]
    pc_cols = [c for c in pc_cols if c in pc.columns]
    lines.append(to_md(pc[pc_cols], index=False))
    lines.append("")
    lines.append("## Margin-bounded correctable counts by hard pair")
    lines.append("")
    pair_cols = [
        "pair", "direction", "n_wrong", "n_wrong_true_in_top2", "wrong_true_in_top2_rate",
        "correctable_margin_le_0.1_n", "correctable_margin_le_0.2_n",
        "correctable_margin_le_0.3_n", "correctable_margin_le_0.5_n",
    ]
    pair_cols = [c for c in pair_cols if c in pair_df.columns]
    lines.append(to_md(pair_df[pair_cols], index=False))
    lines.append("")
    lines.append("## How to read this")
    lines.append("")
    lines.append("- `oracle_all_wrong_true_in_top2` is a theoretical upper bound: it assumes a perfect mechanism fixes every wrong sample whose true class is already in top-2.")
    lines.append("- `oracle_wrong_top2_margin_le_T` is a margin-bounded upper bound: it assumes perfect correction only when the probability gap is at most `T`.")
    lines.append("- If full top-2 oracle is high but small-margin oracle is low, a simple confidence/margin rule is probably too weak.")
    lines.append("- This is still diagnostic only; it does not prove a real reranker will achieve these numbers.")
    lines.append("")
    lines.append("## Generated files")
    lines.append("")
    for p in out_files:
        lines.append(f"- `{p}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    pred_csv = repo_path(repo_root, args.pred_csv)
    out_dir = repo_path(repo_root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pred_csv.exists():
        raise FileNotFoundError(f"Missing prediction CSV: {pred_csv}")

    pairs = parse_pairs(args.pairs)
    thresholds = sorted(set(float(x) for x in args.thresholds))

    df0 = pd.read_csv(pred_csv)
    ensure_base_columns(df0)

    labels = infer_label_order(df0, args.label_order, args.score_prefix)
    df = build_augmented_df(df0, labels, args.score_prefix)

    policy_df, per_class_df, pair_df = make_policy_metrics(df, labels, thresholds, pairs)
    gate = make_gate(policy_df)

    # Save corrected sample index lists for transparency.
    corrected_lists: Dict[str, List[int]] = {}
    for t in thresholds:
        mask = df["wrong_true_in_top2"] & (df["pred_minus_true_prob"] <= t)
        corrected_lists[f"margin_le_{t:g}"] = [int(x) for x in df.loc[mask, "sample_index"].tolist()]
    corrected_lists["oracle_all_wrong_true_in_top2"] = [int(x) for x in df.loc[df["wrong_true_in_top2"], "sample_index"].tolist()]

    # Output paths.
    summary_path = out_dir / "C2_summary.md"
    metrics_path = out_dir / "C2_policy_metrics.csv"
    per_class_path = out_dir / "C2_policy_per_class_f1.csv"
    pair_path = out_dir / "C2_margin_threshold_by_pair.csv"
    gate_path = out_dir / "C2_gate_decision.json"
    manifest_path = out_dir / "C2_run_manifest.json"
    corrected_lists_path = out_dir / "C2_corrected_sample_indices.json"

    policy_df.to_csv(metrics_path, index=False)
    per_class_df.to_csv(per_class_path, index=False)
    pair_df.to_csv(pair_path, index=False)
    gate_path.write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    corrected_lists_path.write_text(json.dumps(corrected_lists, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "stage": "C2_top2_oracle_upper_bound",
        "purpose": "Oracle upper-bound simulation for top-2 correction; diagnostic only.",
        "inputs": {
            "pred_csv": str(pred_csv),
            "score_prefix": args.score_prefix,
        },
        "labels": labels,
        "hard_pairs": [f"{a}<->{b}" for a, b in pairs],
        "thresholds": thresholds,
        "gate": gate,
        "baseline_guardrail": "This run uses the current 03_outputs/06_model baseline under analysis, not the separate 0.817 model.",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    out_files = [
        summary_path,
        metrics_path,
        per_class_path,
        pair_path,
        gate_path,
        manifest_path,
        corrected_lists_path,
    ]

    summary = make_markdown(
        labels=labels,
        policy_df=policy_df,
        per_class_df=per_class_df,
        pair_df=pair_df,
        gate=gate,
        out_files=out_files,
    )
    summary_path.write_text(summary, encoding="utf-8")

    out_zip = zip_outputs(out_dir)

    print("===== C2 top-2 oracle upper-bound audit done =====")
    print("summary:", summary_path)
    print("metrics:", metrics_path)
    print("gate:", gate_path)
    print("zip:", out_zip)
    print("result:", gate["result"])
    print("original_macro_f1:", gate["original_macro_f1"])
    print("oracle_all_top2_macro_f1:", gate["oracle_all_top2_macro_f1"])
    print("oracle_all_top2_delta_macro_f1:", gate["oracle_all_top2_delta_macro_f1"])
    print("margin_le_0.2_delta_macro_f1:", gate["margin_le_0.2_delta_macro_f1"])
    print("margin_le_0.5_delta_macro_f1:", gate["margin_le_0.5_delta_macro_f1"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
