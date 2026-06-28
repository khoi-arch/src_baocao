#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B3_compare_phase_b_signal.py

Purpose
-------
Summarize Phase B diagnostics by comparing:

  B0: wrong-sample top-2 coverage
  B1: CLS pairwise signal
  B2: raw/token/offset input-space pairwise signal

This script does NOT train, rerank, or modify official baseline files.
It only reads Phase B outputs and writes a diagnosis summary.

Default inputs
--------------
  05_test/outputs/B0_wrong_top2_audit/B0_metrics.json
  05_test/outputs/B1_cls_pairwise_signal/B1_pairwise_logreg_cv_metrics.csv
  05_test/outputs/B1_cls_pairwise_signal/B1_gate_decision.json
  05_test/outputs/B2_input_pairwise_signal/B2_pairwise_signal_metrics.csv
  05_test/outputs/B2_input_pairwise_signal/B2_gate_decision.json

Outputs
-------
  05_test/outputs/B3_phase_b_signal_comparison/B3_summary.md
  05_test/outputs/B3_phase_b_signal_comparison/B3_cls_vs_input_by_pair.csv
  05_test/outputs/B3_phase_b_signal_comparison/B3_phase_b_decision.json
  05_test/outputs/B3_phase_b_signal_comparison/B3_phase_b_signal_comparison_output.zip
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="B3 compare Phase B signal diagnostics.")
    p.add_argument("--repo-root", default=".")
    p.add_argument("--b0-metrics", default="05_test/outputs/B0_wrong_top2_audit/B0_metrics.json")
    p.add_argument("--b1-metrics-csv", default="05_test/outputs/B1_cls_pairwise_signal/B1_pairwise_logreg_cv_metrics.csv")
    p.add_argument("--b1-gate-json", default="05_test/outputs/B1_cls_pairwise_signal/B1_gate_decision.json")
    p.add_argument("--b2-metrics-csv", default="05_test/outputs/B2_input_pairwise_signal/B2_pairwise_signal_metrics.csv")
    p.add_argument("--b2-gate-json", default="05_test/outputs/B2_input_pairwise_signal/B2_gate_decision.json")
    p.add_argument("--out-dir", default="05_test/outputs/B3_phase_b_signal_comparison")
    return p.parse_args()


def repo_path(repo_root: Path, path_like: str | Path) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    return repo_root / p


def load_json_or_empty(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def require_file(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")


def zip_outputs(out_dir: Path, zip_name: str = "B3_phase_b_signal_comparison_output.zip") -> Path:
    out_zip = out_dir / zip_name
    if out_zip.exists():
        out_zip.unlink()

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fp in out_dir.rglob("*"):
            if fp.is_file() and fp != out_zip:
                z.write(fp, fp.relative_to(out_dir))
    return out_zip


def get_b0_main_metrics(b0: Dict[str, Any]) -> Dict[str, Any]:
    """
    Be tolerant because B0_metrics.json schema may differ.
    """
    if not b0:
        return {
            "available": False,
            "n_wrong": None,
            "wrong_true_in_top2": None,
            "wrong_true_in_top2_rate": None,
            "top2_accuracy": None,
        }

    candidates = [b0]
    for k in ("main_metrics", "metrics", "summary", "overall"):
        if isinstance(b0.get(k), dict):
            candidates.append(b0[k])

    def find_key(keys: List[str]):
        for d in candidates:
            for key in keys:
                if key in d:
                    return d[key]
        return None

    n_wrong = find_key(["n_wrong", "wrong_total"])
    wrong_true_in_top2 = find_key(["wrong_true_in_top2"])
    wrong_true_in_top2_rate = find_key(["wrong_true_in_top2_rate"])
    top2_accuracy = find_key(["top2_accuracy", "top2_accuracy_from_export"])

    return {
        "available": True,
        "n_wrong": n_wrong,
        "wrong_true_in_top2": wrong_true_in_top2,
        "wrong_true_in_top2_rate": wrong_true_in_top2_rate,
        "top2_accuracy": top2_accuracy,
    }


def make_cls_vs_input_table(b1_df: pd.DataFrame, b2_df: pd.DataFrame) -> pd.DataFrame:
    required_b1 = {"pair", "macro_f1", "auc", "balanced_accuracy"}
    required_b2 = {"pair", "representation", "macro_f1", "auc", "balanced_accuracy", "dim"}

    miss_b1 = required_b1 - set(b1_df.columns)
    miss_b2 = required_b2 - set(b2_df.columns)
    if miss_b1:
        raise KeyError(f"B1 metrics missing columns: {sorted(miss_b1)}")
    if miss_b2:
        raise KeyError(f"B2 metrics missing columns: {sorted(miss_b2)}")

    b1 = b1_df.copy()
    b1 = b1[b1.get("cv_status", "ok").eq("ok")] if "cv_status" in b1.columns else b1

    b2 = b2_df.copy()
    b2 = b2[b2.get("cv_status", "ok").eq("ok")] if "cv_status" in b2.columns else b2

    rows: List[Dict[str, Any]] = []
    for pair in sorted(set(b1["pair"]).intersection(set(b2["pair"]))):
        cls_row = b1[b1["pair"] == pair].sort_values(["macro_f1", "auc"], ascending=[False, False]).iloc[0]
        input_row = b2[b2["pair"] == pair].sort_values(["macro_f1", "auc"], ascending=[False, False]).iloc[0]

        delta_f1 = float(cls_row["macro_f1"] - input_row["macro_f1"])
        delta_auc = float(cls_row["auc"] - input_row["auc"])
        delta_bacc = float(cls_row["balanced_accuracy"] - input_row["balanced_accuracy"])

        if delta_f1 >= 0.10 and delta_auc >= 0.10:
            interpretation = "CLS much stronger than input-linear"
        elif delta_f1 >= 0.05 and delta_auc >= 0.05:
            interpretation = "CLS moderately stronger than input-linear"
        elif delta_f1 > -0.02:
            interpretation = "CLS roughly comparable to input-linear"
        else:
            interpretation = "Input-linear stronger than CLS"

        rows.append({
            "pair": pair,
            "cls_macro_f1": float(cls_row["macro_f1"]),
            "best_input_representation": str(input_row["representation"]),
            "best_input_dim": int(input_row["dim"]),
            "best_input_macro_f1": float(input_row["macro_f1"]),
            "delta_cls_minus_input_macro_f1": delta_f1,
            "cls_auc": float(cls_row["auc"]),
            "best_input_auc": float(input_row["auc"]),
            "delta_cls_minus_input_auc": delta_auc,
            "cls_balanced_accuracy": float(cls_row["balanced_accuracy"]),
            "best_input_balanced_accuracy": float(input_row["balanced_accuracy"]),
            "delta_cls_minus_input_balanced_accuracy": delta_bacc,
            "interpretation": interpretation,
        })

    return pd.DataFrame(rows)


def make_decision(
    b0_main: Dict[str, Any],
    b1_gate: Dict[str, Any],
    b2_gate: Dict[str, Any],
    cmp_df: pd.DataFrame,
) -> Dict[str, Any]:
    mean_delta_f1 = float(cmp_df["delta_cls_minus_input_macro_f1"].mean())
    min_delta_f1 = float(cmp_df["delta_cls_minus_input_macro_f1"].min())
    mean_delta_auc = float(cmp_df["delta_cls_minus_input_auc"].mean())
    min_delta_auc = float(cmp_df["delta_cls_minus_input_auc"].min())

    b1_result = str(b1_gate.get("result", "unknown"))
    b2_result = str(b2_gate.get("result", "unknown"))

    wrong_top2_rate = b0_main.get("wrong_true_in_top2_rate")
    try:
        wrong_top2_rate_f = float(wrong_top2_rate)
    except Exception:
        wrong_top2_rate_f = float("nan")

    # Phase-B diagnosis.
    if (
        "PASS" in b1_result
        and ("FAIL" in b2_result or "MIXED" in b2_result)
        and mean_delta_f1 >= 0.10
        and mean_delta_auc >= 0.10
        and (np.isnan(wrong_top2_rate_f) or wrong_top2_rate_f >= 0.65)
    ):
        result = "PHASE_B_PASS — bottleneck is not primarily preprocessing/input-linear separability"
        diagnosis = (
            "B0 shows high wrong-sample true-in-top2 coverage, B1 shows usable CLS pairwise signal, "
            "and B2 shows weak input-space linear signal. CLS is substantially stronger than raw/token/offset "
            "linear baselines. Therefore the current evidence points to ambiguous subtype ranking / hard-pair "
            "decision in CLS/logit space, not a first-priority preprocessing/tokenization failure."
        )
        next_phase = "Proceed to Phase C diagnostic rerank or logit/CLS hard-pair correction tests before Phase D model changes."
    elif "PASS" in b1_result and mean_delta_f1 >= 0.05:
        result = "PHASE_B_MIXED — CLS improves over input, but diagnosis is not fully decisive"
        diagnosis = (
            "CLS appears stronger than input-space linear baselines, but the margin is not large enough under "
            "the configured gate to fully rule out input/preprocessing issues."
        )
        next_phase = "Run additional B2 variants or B3 review before choosing Phase C."
    else:
        result = "PHASE_B_FAIL_OR_INCONCLUSIVE — input/representation bottleneck still unclear"
        diagnosis = (
            "The comparison does not clearly show CLS outperforming input baselines. More input-space or "
            "representation-space diagnosis is needed before trying correction methods."
        )
        next_phase = "Do not start Phase C/D yet; inspect B1/B2 assumptions."

    return {
        "result": result,
        "diagnosis": diagnosis,
        "recommended_next_phase": next_phase,
        "b0": b0_main,
        "b1_gate_result": b1_result,
        "b2_gate_result": b2_result,
        "comparison_stats": {
            "mean_delta_cls_minus_input_macro_f1": mean_delta_f1,
            "min_delta_cls_minus_input_macro_f1": min_delta_f1,
            "mean_delta_cls_minus_input_auc": mean_delta_auc,
            "min_delta_cls_minus_input_auc": min_delta_auc,
        },
        "important_guardrail": (
            "Phase B is diagnostic only. It does not prove a final fix. "
            "Any rerank/auxiliary-head/margin-loss idea must still be validated as an isolated Phase C/D test."
        ),
    }


def make_markdown(
    *,
    b0_main: Dict[str, Any],
    b1_gate: Dict[str, Any],
    b2_gate: Dict[str, Any],
    cmp_df: pd.DataFrame,
    decision: Dict[str, Any],
    out_files: List[Path],
) -> str:
    lines: List[str] = []
    lines.append("# B3 — Phase B signal comparison")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("Compare B0/B1/B2 to locate where malware-subtype signal exists before trying any solution.")
    lines.append("")
    lines.append("## Inputs summarized")
    lines.append("")
    lines.append("### B0 — wrong-sample top-2")
    lines.append("")
    for k, v in b0_main.items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("### B1 gate")
    lines.append("")
    lines.append(f"- `{b1_gate.get('result', 'unknown')}`")
    if b1_gate.get("reason"):
        lines.append(f"- Reason: {b1_gate.get('reason')}")
    lines.append("")
    lines.append("### B2 gate")
    lines.append("")
    lines.append(f"- `{b2_gate.get('result', 'unknown')}`")
    if b2_gate.get("reason"):
        lines.append(f"- Reason: {b2_gate.get('reason')}")
    lines.append("")
    lines.append("## CLS vs best input representation by pair")
    lines.append("")
    show_cols = [
        "pair",
        "cls_macro_f1",
        "best_input_representation",
        "best_input_macro_f1",
        "delta_cls_minus_input_macro_f1",
        "cls_auc",
        "best_input_auc",
        "delta_cls_minus_input_auc",
        "interpretation",
    ]
    lines.append(cmp_df[show_cols].to_markdown(index=False))
    lines.append("")
    lines.append("## Phase B diagnosis")
    lines.append("")
    lines.append(f"- Result: **{decision['result']}**")
    lines.append(f"- Diagnosis: {decision['diagnosis']}")
    lines.append(f"- Recommended next phase: {decision['recommended_next_phase']}")
    lines.append("")
    lines.append("## Comparison stats")
    lines.append("")
    for k, v in decision["comparison_stats"].items():
        lines.append(f"- `{k}`: {v:.6f}")
    lines.append("")
    lines.append("## Guardrail")
    lines.append("")
    lines.append(decision["important_guardrail"])
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

    b0_metrics = repo_path(repo_root, args.b0_metrics)
    b1_metrics_csv = repo_path(repo_root, args.b1_metrics_csv)
    b1_gate_json = repo_path(repo_root, args.b1_gate_json)
    b2_metrics_csv = repo_path(repo_root, args.b2_metrics_csv)
    b2_gate_json = repo_path(repo_root, args.b2_gate_json)
    out_dir = repo_path(repo_root, args.out_dir)

    require_file(b1_metrics_csv, "B1 metrics CSV")
    require_file(b2_metrics_csv, "B2 metrics CSV")
    require_file(b1_gate_json, "B1 gate JSON")
    require_file(b2_gate_json, "B2 gate JSON")

    out_dir.mkdir(parents=True, exist_ok=True)

    b0 = load_json_or_empty(b0_metrics)
    b0_main = get_b0_main_metrics(b0)
    b1_gate = load_json_or_empty(b1_gate_json)
    b2_gate = load_json_or_empty(b2_gate_json)

    b1_df = pd.read_csv(b1_metrics_csv)
    b2_df = pd.read_csv(b2_metrics_csv)

    cmp_df = make_cls_vs_input_table(b1_df, b2_df)
    decision = make_decision(b0_main, b1_gate, b2_gate, cmp_df)

    cmp_path = out_dir / "B3_cls_vs_input_by_pair.csv"
    decision_path = out_dir / "B3_phase_b_decision.json"
    summary_path = out_dir / "B3_summary.md"

    cmp_df.to_csv(cmp_path, index=False)
    decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

    out_files = [summary_path, cmp_path, decision_path]
    summary = make_markdown(
        b0_main=b0_main,
        b1_gate=b1_gate,
        b2_gate=b2_gate,
        cmp_df=cmp_df,
        decision=decision,
        out_files=out_files,
    )
    summary_path.write_text(summary, encoding="utf-8")

    out_zip = zip_outputs(out_dir)

    print("===== B3 Phase B signal comparison done =====")
    print("summary:", summary_path)
    print("comparison:", cmp_path)
    print("decision:", decision_path)
    print("zip:", out_zip)
    print("result:", decision["result"])
    print("recommended_next_phase:", decision["recommended_next_phase"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
