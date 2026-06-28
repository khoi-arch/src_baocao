#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C1_logit_top2_margin_audit.py

Purpose
-------
Phase C1 diagnostic only.

Audit the official C2+D3 baseline's probability/logit decision margins:
  - Are wrong samples close top-1/top-2 decisions?
  - Among wrong samples where the true label is still top-2, how large is the
    pred-vs-true probability margin?
  - Which hard malware pairs have small/large wrong top-2 margins?

This script does NOT rerank, retrain, or modify any official baseline files.
It only reads exported validation probabilities and writes diagnostic outputs.

Recommended input
-----------------
Use the probability CSV exported from the baseline currently under analysis:

  05_test/outputs/B0_wrong_top2_audit/val_predictions_with_probs.csv

or the equivalent B1 export:

  05_test/outputs/B1_cls_pairwise_signal/val_cls_predictions_with_probs.csv

Outputs
-------
  C1_summary.md
  C1_margin_overall.csv
  C1_wrong_top2_by_true_class.csv
  C1_wrong_top2_by_confusion_pair.csv
  C1_hard_pair_margin_summary.csv
  C1_wrong_top2_samples.csv
  C1_gate_decision.json
  C1_logit_top2_margin_output.zip
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


MARGIN_THRESHOLDS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="C1 logit/top-2 margin audit.")
    p.add_argument("--repo-root", default=".")
    p.add_argument("--pred-csv", default="05_test/outputs/B0_wrong_top2_audit/val_predictions_with_probs.csv")
    p.add_argument("--out-dir", default="05_test/outputs/C1_logit_top2_margin_audit")
    p.add_argument("--score-prefix", default="prob_")
    p.add_argument("--pairs", nargs="*", default=None,
                   help='Optional hard pairs as "A:B", e.g. "Ransomware:Trojan". Default malware pairs.')
    p.add_argument("--small-margin-threshold", type=float, default=0.10,
                   help="Diagnostic threshold for a small pred-vs-true/top1-top2 margin.")
    p.add_argument("--moderate-margin-threshold", type=float, default=0.20,
                   help="Diagnostic threshold for a moderate pred-vs-true/top1-top2 margin.")
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

    # Fallback: match after normalizing spaces/underscores/case.
    target = safe_label_col(label).lower()
    for c in df.columns:
        if not c.startswith(score_prefix):
            continue
        tail = c[len(score_prefix):]
        if safe_label_col(tail).lower() == target:
            return c

    raise KeyError(f"Cannot find probability column for label={label!r}; tried {candidates}")


def infer_prob_columns(df: pd.DataFrame, score_prefix: str) -> List[str]:
    cols = [c for c in df.columns if c.startswith(score_prefix)]
    if not cols:
        raise KeyError(f"No probability columns found with prefix {score_prefix!r}")
    return cols


def infer_label_names(df: pd.DataFrame, prob_cols: List[str], score_prefix: str) -> List[str]:
    # Prefer labels visible in true/pred/top columns to preserve original naming.
    labels: List[str] = []
    for col in ["true_label", "pred_label", "top1_label", "top2_label"]:
        if col in df.columns:
            for x in df[col].dropna().map(normalize_label).unique().tolist():
                if x not in labels:
                    labels.append(x)

    # Append any prob-only labels.
    for c in prob_cols:
        label = c[len(score_prefix):]
        label = label.replace("_", " ")
        if label not in labels:
            labels.append(label)

    return labels


def get_prob_matrix(df: pd.DataFrame, label_names: List[str], score_prefix: str) -> Tuple[np.ndarray, Dict[str, str]]:
    col_map: Dict[str, str] = {}
    for label in label_names:
        col_map[label] = prob_col_for_label(df, label, score_prefix)

    probs = df[[col_map[label] for label in label_names]].to_numpy(dtype=np.float64)
    if not np.isfinite(probs).all():
        raise ValueError("Probability matrix contains NaN/Inf")
    return probs, col_map


def ensure_base_columns(df: pd.DataFrame) -> None:
    required = ["true_label", "pred_label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Prediction CSV missing required columns: {missing}")


def add_margin_columns(df: pd.DataFrame, label_names: List[str], probs: np.ndarray) -> pd.DataFrame:
    out = df.copy()

    true_label = out["true_label"].map(normalize_label).to_numpy()
    pred_label = out["pred_label"].map(normalize_label).to_numpy()
    label_to_id = {label: i for i, label in enumerate(label_names)}

    unknown_true = sorted(set(true_label) - set(label_to_id))
    unknown_pred = sorted(set(pred_label) - set(label_to_id))
    if unknown_true:
        raise KeyError(f"Unknown true labels not in probability labels: {unknown_true}")
    if unknown_pred:
        raise KeyError(f"Unknown pred labels not in probability labels: {unknown_pred}")

    true_id = np.array([label_to_id[x] for x in true_label], dtype=int)
    pred_id = np.array([label_to_id[x] for x in pred_label], dtype=int)

    order = np.argsort(-probs, axis=1)
    top1_id = order[:, 0]
    top2_id = order[:, 1]
    top3_id = order[:, 2] if probs.shape[1] >= 3 else order[:, 1]

    idx = np.arange(len(out))

    top1_score = probs[idx, top1_id]
    top2_score = probs[idx, top2_id]
    top3_score = probs[idx, top3_id]
    true_prob = probs[idx, true_id]
    pred_prob = probs[idx, pred_id]

    correct = pred_id == true_id
    true_in_top2 = (true_id == top1_id) | (true_id == top2_id)
    true_rank = np.empty(len(out), dtype=int)
    for i in range(len(out)):
        true_rank[i] = int(np.where(order[i] == true_id[i])[0][0]) + 1

    out["true_label"] = true_label
    out["pred_label"] = pred_label
    out["computed_correct"] = correct
    out["computed_top1_label"] = [label_names[i] for i in top1_id]
    out["computed_top2_label"] = [label_names[i] for i in top2_id]
    out["computed_top3_label"] = [label_names[i] for i in top3_id]
    out["computed_top1_score"] = top1_score
    out["computed_top2_score"] = top2_score
    out["computed_top3_score"] = top3_score
    out["computed_top12_margin"] = top1_score - top2_score
    out["computed_top23_margin"] = top2_score - top3_score
    out["true_prob"] = true_prob
    out["pred_prob"] = pred_prob
    out["pred_minus_true_prob"] = pred_prob - true_prob
    out["true_minus_pred_prob"] = true_prob - pred_prob
    out["true_rank"] = true_rank
    out["computed_true_in_top2"] = true_in_top2

    # For correct samples, pred_minus_true_prob is 0. For wrong samples with true in top2,
    # pred_minus_true_prob equals top1_score - top2_score.
    out["wrong"] = ~correct
    out["wrong_true_in_top2"] = (~correct) & true_in_top2

    return out


def quantile_summary(values: np.ndarray, prefix: str) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_q05": float("nan"),
            f"{prefix}_q10": float("nan"),
            f"{prefix}_q25": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_q75": float("nan"),
            f"{prefix}_q90": float("nan"),
            f"{prefix}_q95": float("nan"),
            f"{prefix}_max": float("nan"),
        }

    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_q05": float(np.quantile(values, 0.05)),
        f"{prefix}_q10": float(np.quantile(values, 0.10)),
        f"{prefix}_q25": float(np.quantile(values, 0.25)),
        f"{prefix}_median": float(np.quantile(values, 0.50)),
        f"{prefix}_q75": float(np.quantile(values, 0.75)),
        f"{prefix}_q90": float(np.quantile(values, 0.90)),
        f"{prefix}_q95": float(np.quantile(values, 0.95)),
        f"{prefix}_max": float(np.max(values)),
    }


def threshold_rates(values: np.ndarray, prefix: str, thresholds: Iterable[float]) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    out: Dict[str, float] = {}
    if len(values) == 0:
        for t in thresholds:
            out[f"{prefix}_le_{t:g}"] = float("nan")
        return out
    for t in thresholds:
        out[f"{prefix}_le_{t:g}"] = float(np.mean(values <= t))
    return out


def group_margin_summary(df: pd.DataFrame, group_cols: List[str], *, subset_col: str | None = None) -> pd.DataFrame:
    if subset_col is not None:
        d = df[df[subset_col].astype(bool)].copy()
    else:
        d = df.copy()

    rows: List[Dict[str, Any]] = []
    if len(d) == 0:
        return pd.DataFrame()

    for keys, g in d.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row: Dict[str, Any] = {col: key for col, key in zip(group_cols, keys)}
        row["n"] = int(len(g))
        row["n_correct"] = int((~g["wrong"]).sum())
        row["n_wrong"] = int(g["wrong"].sum())
        row["n_wrong_true_in_top2"] = int(g["wrong_true_in_top2"].sum())
        row["wrong_true_in_top2_rate_among_wrong"] = (
            float(row["n_wrong_true_in_top2"] / row["n_wrong"]) if row["n_wrong"] else float("nan")
        )

        # Overall top12 margin in this group.
        row.update(quantile_summary(g["computed_top12_margin"].to_numpy(), "top12_margin"))
        row.update(threshold_rates(g["computed_top12_margin"].to_numpy(), "top12_margin_rate", MARGIN_THRESHOLDS))

        # For wrong-true-in-top2, this is the actual margin the true class needs to overcome.
        wt2 = g[g["wrong_true_in_top2"]]
        row.update(quantile_summary(wt2["pred_minus_true_prob"].to_numpy(), "wrong_top2_pred_minus_true"))
        row.update(threshold_rates(wt2["pred_minus_true_prob"].to_numpy(), "wrong_top2_pred_minus_true_rate", MARGIN_THRESHOLDS))

        rows.append(row)

    return pd.DataFrame(rows)


def hard_pair_summary(df: pd.DataFrame, pairs: List[Tuple[str, str]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for a, b in pairs:
        pair_true = df["true_label"].isin([a, b])
        pair_direct_wrong = (
            ((df["true_label"] == a) & (df["pred_label"] == b)) |
            ((df["true_label"] == b) & (df["pred_label"] == a))
        )
        pair_wrong_top2 = pair_direct_wrong & df["wrong_true_in_top2"]

        for direction_true, direction_pred in [(a, b), (b, a)]:
            dmask = (df["true_label"] == direction_true) & (df["pred_label"] == direction_pred)
            d = df[dmask]
            d_wt2 = d[d["wrong_true_in_top2"]]

            row: Dict[str, Any] = {
                "pair": f"{a}<->{b}",
                "direction": f"{direction_true}->{direction_pred}",
                "true_class": direction_true,
                "pred_class": direction_pred,
                "n_wrong": int(len(d)),
                "n_wrong_true_in_top2": int(len(d_wt2)),
                "wrong_true_in_top2_rate": float(len(d_wt2) / len(d)) if len(d) else float("nan"),
            }
            row.update(quantile_summary(d_wt2["pred_minus_true_prob"].to_numpy(), "pred_minus_true"))
            row.update(threshold_rates(d_wt2["pred_minus_true_prob"].to_numpy(), "pred_minus_true_rate", MARGIN_THRESHOLDS))
            rows.append(row)

        all_pair = df[pair_true]
        direct = df[pair_direct_wrong]
        wt2 = df[pair_wrong_top2]

        row = {
            "pair": f"{a}<->{b}",
            "direction": "BIDIRECTIONAL_DIRECT_CONFUSION",
            "true_class": a,
            "pred_class": b,
            "n_wrong": int(len(direct)),
            "n_wrong_true_in_top2": int(len(wt2)),
            "wrong_true_in_top2_rate": float(len(wt2) / len(direct)) if len(direct) else float("nan"),
        }
        row.update(quantile_summary(wt2["pred_minus_true_prob"].to_numpy(), "pred_minus_true"))
        row.update(threshold_rates(wt2["pred_minus_true_prob"].to_numpy(), "pred_minus_true_rate", MARGIN_THRESHOLDS))
        rows.append(row)

    return pd.DataFrame(rows)


def zip_outputs(out_dir: Path, zip_name: str = "C1_logit_top2_margin_output.zip") -> Path:
    out_zip = out_dir / zip_name
    if out_zip.exists():
        out_zip.unlink()

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fp in out_dir.rglob("*"):
            if fp.is_file() and fp != out_zip:
                z.write(fp, fp.relative_to(out_dir))
    return out_zip


def make_gate_decision(
    *,
    n_wrong: int,
    wrong_true_in_top2_rate: float,
    wrong_top2_median_margin: float,
    wrong_top2_rate_le_small: float,
    wrong_top2_rate_le_moderate: float,
    small_threshold: float,
    moderate_threshold: float,
) -> Dict[str, Any]:
    if (
        wrong_true_in_top2_rate >= 0.65
        and (
            wrong_top2_median_margin <= moderate_threshold
            or wrong_top2_rate_le_moderate >= 0.50
        )
    ):
        result = "PASS — many wrong samples are top-2 recoverable with non-extreme margins"
        reason = (
            f"wrong_true_in_top2_rate={wrong_true_in_top2_rate:.4f}; "
            f"median pred-vs-true margin among wrong-top2={wrong_top2_median_margin:.4f}; "
            f"rate margin<={moderate_threshold:g} is {wrong_top2_rate_le_moderate:.4f}. "
            "This supports testing isolated Phase C pairwise/rerank diagnostics."
        )
    elif wrong_true_in_top2_rate >= 0.65:
        result = "MIXED — true label is often top-2 but margins may be large"
        reason = (
            f"wrong_true_in_top2_rate={wrong_true_in_top2_rate:.4f}, but median margin "
            f"among wrong-top2={wrong_top2_median_margin:.4f}. "
            "A simple rerank may be hard; Phase C should first test diagnostic upper/lower-bound rerank."
        )
    else:
        result = "FAIL — not enough wrong samples are top-2 recoverable"
        reason = (
            f"wrong_true_in_top2_rate={wrong_true_in_top2_rate:.4f}. "
            "Top-2 based correction is unlikely to address most errors."
        )

    return {
        "result": result,
        "reason": reason,
        "n_wrong": int(n_wrong),
        "wrong_true_in_top2_rate": float(wrong_true_in_top2_rate),
        "wrong_top2_median_pred_minus_true_margin": float(wrong_top2_median_margin),
        "wrong_top2_rate_margin_le_small_threshold": float(wrong_top2_rate_le_small),
        "wrong_top2_rate_margin_le_moderate_threshold": float(wrong_top2_rate_le_moderate),
        "small_margin_threshold": float(small_threshold),
        "moderate_margin_threshold": float(moderate_threshold),
        "guardrail": "C1 is diagnostic only. It does not perform reranking or modify model outputs.",
    }


def to_md(df: pd.DataFrame, index: bool = False) -> str:
    try:
        return df.to_markdown(index=index)
    except Exception:
        return df.to_string(index=index)


def make_markdown_summary(
    *,
    main_metrics: Dict[str, Any],
    gate: Dict[str, Any],
    overall_df: pd.DataFrame,
    by_true_df: pd.DataFrame,
    by_pair_df: pd.DataFrame,
    hard_pair_df: pd.DataFrame,
    out_files: List[Path],
) -> str:
    lines: List[str] = []
    lines.append("# C1 — Logit/top-2 margin audit")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("Audit whether wrong top-2 samples are close-margin decisions or high-confidence wrong decisions.")
    lines.append("")
    lines.append("## Main metrics")
    lines.append("")
    for k, v in main_metrics.items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("## Interpretation gate")
    lines.append("")
    lines.append(f"- Result: **{gate['result']}**")
    lines.append(f"- Reason: {gate['reason']}")
    lines.append("")
    lines.append("## Overall margin distribution")
    lines.append("")
    lines.append(to_md(overall_df, index=False))
    lines.append("")
    lines.append("## Wrong top-2 margin by true class")
    lines.append("")
    show_true_cols = [
        "true_label", "n", "n_wrong", "n_wrong_true_in_top2",
        "wrong_true_in_top2_rate_among_wrong",
        "wrong_top2_pred_minus_true_median",
        "wrong_top2_pred_minus_true_q75",
        "wrong_top2_pred_minus_true_rate_le_0.1",
        "wrong_top2_pred_minus_true_rate_le_0.2",
    ]
    show_true_cols = [c for c in show_true_cols if c in by_true_df.columns]
    lines.append(to_md(by_true_df[show_true_cols], index=False))
    lines.append("")
    lines.append("## Wrong top-2 margin by confusion pair")
    lines.append("")
    show_pair_cols = [
        "true_label", "pred_label", "n", "n_wrong_true_in_top2",
        "wrong_top2_pred_minus_true_median",
        "wrong_top2_pred_minus_true_q75",
        "wrong_top2_pred_minus_true_rate_le_0.1",
        "wrong_top2_pred_minus_true_rate_le_0.2",
    ]
    show_pair_cols = [c for c in show_pair_cols if c in by_pair_df.columns]
    lines.append(to_md(by_pair_df[show_pair_cols], index=False))
    lines.append("")
    lines.append("## Hard malware pair margin summary")
    lines.append("")
    show_hard_cols = [
        "pair", "direction", "n_wrong", "n_wrong_true_in_top2",
        "wrong_true_in_top2_rate",
        "pred_minus_true_median",
        "pred_minus_true_q75",
        "pred_minus_true_rate_le_0.1",
        "pred_minus_true_rate_le_0.2",
    ]
    show_hard_cols = [c for c in show_hard_cols if c in hard_pair_df.columns]
    lines.append(to_md(hard_pair_df[show_hard_cols], index=False))
    lines.append("")
    lines.append("## How to read the margin")
    lines.append("")
    lines.append("- `pred_minus_true_prob = probability(predicted class) - probability(true class)`.")
    lines.append("- For wrong samples where the true class is top-2, this is the probability gap that a reranker would need to overcome.")
    lines.append("- Small margin means the model is uncertain between the wrong top-1 and the true top-2.")
    lines.append("- Large margin means the model is confidently wrong, so a simple rerank is less likely to be enough.")
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
    df0 = pd.read_csv(pred_csv)
    ensure_base_columns(df0)

    prob_cols = infer_prob_columns(df0, args.score_prefix)
    label_names = infer_label_names(df0, prob_cols, args.score_prefix)
    probs, prob_col_map = get_prob_matrix(df0, label_names, args.score_prefix)

    df = add_margin_columns(df0, label_names, probs)

    n_total = int(len(df))
    n_correct = int((~df["wrong"]).sum())
    n_wrong = int(df["wrong"].sum())
    n_wrong_true_in_top2 = int(df["wrong_true_in_top2"].sum())
    wrong_true_in_top2_rate = float(n_wrong_true_in_top2 / n_wrong) if n_wrong else float("nan")
    top2_accuracy = float(df["computed_true_in_top2"].mean())

    wrong_top2 = df[df["wrong_true_in_top2"]].copy()
    wrong_top2_margins = wrong_top2["pred_minus_true_prob"].to_numpy(dtype=float)

    wrong_top2_margin_summary = quantile_summary(wrong_top2_margins, "wrong_top2_pred_minus_true")
    wrong_top2_thresholds = threshold_rates(wrong_top2_margins, "wrong_top2_pred_minus_true_rate", MARGIN_THRESHOLDS)

    main_metrics = {
        "n_total": n_total,
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "accuracy_from_predictions": round(float(n_correct / n_total), 10),
        "top2_accuracy": round(top2_accuracy, 10),
        "wrong_true_in_top2": n_wrong_true_in_top2,
        "wrong_true_in_top2_rate": round(wrong_true_in_top2_rate, 10),
        "wrong_top2_pred_minus_true_median": round(float(wrong_top2_margin_summary["wrong_top2_pred_minus_true_median"]), 10),
        "wrong_top2_pred_minus_true_q75": round(float(wrong_top2_margin_summary["wrong_top2_pred_minus_true_q75"]), 10),
        f"wrong_top2_pred_minus_true_rate_le_{args.small_margin_threshold:g}": round(
            float(np.mean(wrong_top2_margins <= args.small_margin_threshold)) if len(wrong_top2_margins) else float("nan"), 10
        ),
        f"wrong_top2_pred_minus_true_rate_le_{args.moderate_margin_threshold:g}": round(
            float(np.mean(wrong_top2_margins <= args.moderate_margin_threshold)) if len(wrong_top2_margins) else float("nan"), 10
        ),
        "label_names": label_names,
        "probability_columns": prob_col_map,
        "hard_pairs": [f"{a}<->{b}" for a, b in pairs],
    }

    small_rate = float(np.mean(wrong_top2_margins <= args.small_margin_threshold)) if len(wrong_top2_margins) else float("nan")
    moderate_rate = float(np.mean(wrong_top2_margins <= args.moderate_margin_threshold)) if len(wrong_top2_margins) else float("nan")

    gate = make_gate_decision(
        n_wrong=n_wrong,
        wrong_true_in_top2_rate=wrong_true_in_top2_rate,
        wrong_top2_median_margin=float(wrong_top2_margin_summary["wrong_top2_pred_minus_true_median"]),
        wrong_top2_rate_le_small=small_rate,
        wrong_top2_rate_le_moderate=moderate_rate,
        small_threshold=float(args.small_margin_threshold),
        moderate_threshold=float(args.moderate_margin_threshold),
    )

    # Overall distribution rows.
    subsets = [
        ("all_samples", df),
        ("correct_samples", df[~df["wrong"]]),
        ("wrong_samples", df[df["wrong"]]),
        ("wrong_true_in_top2", wrong_top2),
        ("wrong_true_not_in_top2", df[df["wrong"] & (~df["computed_true_in_top2"])]),
    ]
    overall_rows: List[Dict[str, Any]] = []
    for name, sub in subsets:
        row: Dict[str, Any] = {
            "subset": name,
            "n": int(len(sub)),
        }
        row.update(quantile_summary(sub["computed_top12_margin"].to_numpy(), "top12_margin"))
        row.update(threshold_rates(sub["computed_top12_margin"].to_numpy(), "top12_margin_rate", MARGIN_THRESHOLDS))
        if name == "wrong_true_in_top2":
            row.update(quantile_summary(sub["pred_minus_true_prob"].to_numpy(), "pred_minus_true"))
            row.update(threshold_rates(sub["pred_minus_true_prob"].to_numpy(), "pred_minus_true_rate", MARGIN_THRESHOLDS))
        overall_rows.append(row)
    overall_df = pd.DataFrame(overall_rows)

    by_true_df = group_margin_summary(df, ["true_label"])
    by_true_df = by_true_df.sort_values(["n_wrong", "true_label"], ascending=[False, True]).reset_index(drop=True)

    by_pair_df = group_margin_summary(df[df["wrong"]], ["true_label", "pred_label"])
    if len(by_pair_df):
        by_pair_df = by_pair_df.sort_values(["n", "true_label", "pred_label"], ascending=[False, True, True]).reset_index(drop=True)

    hard_pair_df = hard_pair_summary(df, pairs)

    # Save detailed wrong top2 samples.
    sample_cols = [
        "sample_index", "true_label", "pred_label",
        "computed_top1_label", "computed_top1_score",
        "computed_top2_label", "computed_top2_score",
        "computed_top12_margin", "true_prob", "pred_prob",
        "pred_minus_true_prob", "true_rank",
    ]
    sample_cols = [c for c in sample_cols if c in wrong_top2.columns]
    wrong_top2_samples = wrong_top2[sample_cols].sort_values(
        ["pred_minus_true_prob"], ascending=True
    ).reset_index(drop=True)

    # Write outputs.
    summary_path = out_dir / "C1_summary.md"
    metrics_path = out_dir / "C1_metrics.json"
    overall_path = out_dir / "C1_margin_overall.csv"
    by_true_path = out_dir / "C1_wrong_top2_by_true_class.csv"
    by_pair_path = out_dir / "C1_wrong_top2_by_confusion_pair.csv"
    hard_pair_path = out_dir / "C1_hard_pair_margin_summary.csv"
    wrong_samples_path = out_dir / "C1_wrong_top2_samples.csv"
    gate_path = out_dir / "C1_gate_decision.json"

    overall_df.to_csv(overall_path, index=False)
    by_true_df.to_csv(by_true_path, index=False)
    by_pair_df.to_csv(by_pair_path, index=False)
    hard_pair_df.to_csv(hard_pair_path, index=False)
    wrong_top2_samples.to_csv(wrong_samples_path, index=False)
    gate_path.write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics = {
        "main_metrics": main_metrics,
        "gate": gate,
        "inputs": {
            "pred_csv": str(pred_csv),
            "score_prefix": args.score_prefix,
        },
        "notes": {
            "phase": "C1 diagnostic only",
            "baseline": "Use 03_outputs/06_model baseline currently under analysis; not the 0.817 model unless explicitly rerun later.",
        },
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    out_files = [
        summary_path,
        metrics_path,
        overall_path,
        by_true_path,
        by_pair_path,
        hard_pair_path,
        wrong_samples_path,
        gate_path,
    ]

    summary = make_markdown_summary(
        main_metrics=main_metrics,
        gate=gate,
        overall_df=overall_df,
        by_true_df=by_true_df,
        by_pair_df=by_pair_df,
        hard_pair_df=hard_pair_df,
        out_files=out_files,
    )
    summary_path.write_text(summary, encoding="utf-8")

    out_zip = zip_outputs(out_dir)

    print("===== C1 logit/top-2 margin audit done =====")
    print("summary:", summary_path)
    print("gate:", gate_path)
    print("zip:", out_zip)
    print("result:", gate["result"])
    print("wrong_true_in_top2_rate:", gate["wrong_true_in_top2_rate"])
    print("wrong_top2_median_pred_minus_true_margin:", gate["wrong_top2_median_pred_minus_true_margin"])
    print("wrong_top2_rate_margin_le_small_threshold:", gate["wrong_top2_rate_margin_le_small_threshold"])
    print("wrong_top2_rate_margin_le_moderate_threshold:", gate["wrong_top2_rate_margin_le_moderate_threshold"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
