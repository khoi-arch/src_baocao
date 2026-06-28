#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E1a2 micro-signal / complementarity audit.

Audit-only. No training.

Question
--------
Do E1a1 binary D3 experts learn useful micro-signal different from baseline,
or do they mostly learn the same boundary and flip correct/wrong samples at
roughly equal rates?

Inputs
------
Default repo paths:
  05_test/outputs/E1a1_binary_d3_attention_expert/
  05_test/outputs/E1a0_full_feature_binary_expert/   optional
  05_test/outputs/B0_wrong_top2_audit/val_predictions_with_probs.csv optional

Required E1a1 files:
  E1a1_baseline_top2_context.csv
  RS/all_val_prob_label_b.npy
  RT/all_val_prob_label_b.npy
  ST/all_val_prob_label_b.npy
  RS/pair_summary.json
  RT/pair_summary.json
  ST/pair_summary.json
  E1a1_best_policy_predictions.csv
  E1a1_policy_metrics.csv

Main outputs:
  E1a2_pair_complementarity.csv
  E1a2_disagreement_samples.csv
  E1a2_fix_damage_confidence_stats.csv
  E1a2_score_separation_auc.csv
  E1a2_global_threshold_policy.csv
  E1a2_pair_threshold_grid.csv
  E1a2_e1a0_vs_e1a1_overlap.csv
  E1a2_summary.json
  E1a2_summary.md
  E1a2_micro_signal_complementarity_audit.zip
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


HARD_PAIRS = [
    ("Ransomware", "Spyware"),
    ("Ransomware", "Trojan"),
    ("Spyware", "Trojan"),
]
PAIR_KEY = {
    ("Ransomware", "Spyware"): "RS",
    ("Ransomware", "Trojan"): "RT",
    ("Spyware", "Trojan"): "ST",
}
PAIR_FROM_KEY = {v: k for k, v in PAIR_KEY.items()}
MALWARE_LABELS = {"Ransomware", "Spyware", "Trojan"}


def strip_label(x: Any) -> str:
    return str(x).strip()


def repo_root_from_here() -> Path:
    p = Path(__file__).resolve()
    if p.parent.name == "05_test":
        return p.parents[1]
    return Path.cwd().resolve()


def resolve_path(path_like: str | Path, repo_root: Path) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    return (repo_root / p).resolve()


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def prepare_input_path(path: Path, expected_name_contains: str = "") -> Tuple[Path, Optional[tempfile.TemporaryDirectory]]:
    """
    Accept either a directory or a zip file. Return the actual root dir containing output files.
    """
    tmp = None
    if path.suffix.lower() == ".zip":
        tmp = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(path, "r") as z:
            z.extractall(tmp.name)
        root = Path(tmp.name)
    else:
        root = path

    # If root directly contains files, use it. Else descend into a single matching subdir.
    if (root / "E1a1_baseline_top2_context.csv").exists() or (root / "E1a0_best_policy_predictions.csv").exists():
        return root, tmp

    subdirs = [p for p in root.iterdir() if p.is_dir()] if root.exists() else []
    if len(subdirs) == 1:
        return subdirs[0], tmp

    if expected_name_contains:
        matches = [p for p in subdirs if expected_name_contains in p.name]
        if matches:
            return matches[0], tmp

    return root, tmp


def prob_col(df: pd.DataFrame, label: str) -> Optional[str]:
    for c in [f"prob_{label}", f"p_{label}", f"proba_{label}"]:
        if c in df.columns:
            return c
    return None


def infer_label_mapping(base: pd.DataFrame) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    pairs = (
        base[["true_id", "true_label"]]
        .drop_duplicates()
        .sort_values("true_id")
        .assign(true_label=lambda d: d["true_label"].map(strip_label))
    )
    id_to_label = {int(r.true_id): str(r.true_label) for r in pairs.itertuples()}
    label_to_id = {v: k for k, v in id_to_label.items()}
    label_names = [id_to_label[i] for i in sorted(id_to_label)]
    return label_names, label_to_id, id_to_label


def hard_pair_key_from_labels(a: str, b: str) -> Optional[str]:
    s = frozenset([strip_label(a), strip_label(b)])
    for pair, key in PAIR_KEY.items():
        if s == frozenset(pair):
            return key
    return None


def macro_metrics(y_true: np.ndarray, y_pred: np.ndarray, label_names: List[str]) -> dict:
    labels = list(range(len(label_names)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }


def transition_stats(y_true: np.ndarray, base_pred: np.ndarray, new_pred: np.ndarray) -> dict:
    base_correct = base_pred == y_true
    new_correct = new_pred == y_true
    fixed = (~base_correct) & new_correct
    damaged = base_correct & (~new_correct)
    changed = base_pred != new_pred
    return {
        "wrong_to_correct": int(fixed.sum()),
        "correct_to_wrong": int(damaged.sum()),
        "net_gain": int(fixed.sum() - damaged.sum()),
        "damage_ratio": float(damaged.sum() / fixed.sum()) if int(fixed.sum()) else None,
        "changed_pred_n": int(changed.sum()),
        "baseline_correct": int(base_correct.sum()),
        "new_correct": int(new_correct.sum()),
    }


def safe_auc_binary(y: np.ndarray, score: np.ndarray) -> Optional[float]:
    y = np.asarray(y)
    score = np.asarray(score, dtype=float)
    mask = np.isfinite(score)
    y = y[mask]
    score = score[mask]
    if len(np.unique(y)) < 2:
        return None
    try:
        return float(roc_auc_score(y, score))
    except Exception:
        return None


def corr_stats(x: np.ndarray, y: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return None, None
    xs = pd.Series(x[mask])
    ys = pd.Series(y[mask])
    pearson = float(xs.corr(ys, method="pearson"))
    spearman = float(xs.rank().corr(ys.rank(), method="pearson"))
    return pearson, spearman


def load_e1a1(e1a1_dir: Path) -> dict:
    base = pd.read_csv(e1a1_dir / "E1a1_baseline_top2_context.csv")
    for c in ["true_label", "pred_label", "top1_label", "top2_label"]:
        if c in base.columns:
            base[c] = base[c].map(strip_label)
    label_names, label_to_id, id_to_label = infer_label_mapping(base)

    pair_summaries = {}
    probs = {}
    for pk in ["RS", "RT", "ST"]:
        pair_dir = e1a1_dir / pk
        summ_path = pair_dir / "pair_summary.json"
        if summ_path.exists():
            summ = load_json(summ_path)
        else:
            a, b = PAIR_FROM_KEY[pk]
            summ = {
                "pair_key": pk,
                "label_a": a,
                "label_b": b,
                "id_a": label_to_id[a],
                "id_b": label_to_id[b],
                "pair": f"{a}<->{b}",
            }
        pair_summaries[pk] = summ
        probs[pk] = np.load(pair_dir / "all_val_prob_label_b.npy").astype(float)

    best_pred = None
    best_pred_path = e1a1_dir / "E1a1_best_policy_predictions.csv"
    if best_pred_path.exists():
        best_pred = pd.read_csv(best_pred_path)
        for c in best_pred.columns:
            if "label" in c:
                best_pred[c] = best_pred[c].map(strip_label)

    return {
        "base": base,
        "label_names": label_names,
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "pair_summaries": pair_summaries,
        "probs": probs,
        "best_pred": best_pred,
    }


def add_expert_columns(base: pd.DataFrame, data: dict) -> pd.DataFrame:
    df = base.copy()
    for pk, summ in data["pair_summaries"].items():
        a = strip_label(summ["label_a"])
        b = strip_label(summ["label_b"])
        p = data["probs"][pk]
        df[f"{pk}_expert_prob_b"] = p
        df[f"{pk}_expert_conf"] = np.maximum(p, 1.0 - p)
        df[f"{pk}_expert_margin"] = np.abs(p - 0.5)
        df[f"{pk}_expert_label"] = np.where(p >= 0.5, b, a)

        col_a = prob_col(df, a)
        col_b = prob_col(df, b)
        if col_a is not None and col_b is not None:
            pa = df[col_a].to_numpy(dtype=float)
            pb = df[col_b].to_numpy(dtype=float)
            denom = np.maximum(1e-12, pa + pb)
            pair_prob_b = pb / denom
            df[f"{pk}_baseline_pair_prob_b"] = pair_prob_b
            df[f"{pk}_baseline_pair_conf"] = np.maximum(pair_prob_b, 1.0 - pair_prob_b)
            df[f"{pk}_baseline_pair_label"] = np.where(pair_prob_b >= 0.5, b, a)
            df[f"{pk}_expert_minus_baseline_pair_conf"] = df[f"{pk}_expert_conf"] - df[f"{pk}_baseline_pair_conf"]
        else:
            df[f"{pk}_baseline_pair_prob_b"] = np.nan
            df[f"{pk}_baseline_pair_conf"] = np.nan
            df[f"{pk}_baseline_pair_label"] = ""
            df[f"{pk}_expert_minus_baseline_pair_conf"] = np.nan

    return df


def pair_mask_top2(df: pd.DataFrame, a: str, b: str) -> np.ndarray:
    top1 = df["top1_label"].map(strip_label).to_numpy()
    top2 = df["top2_label"].map(strip_label).to_numpy()
    s = frozenset([a, b])
    return np.array([frozenset([x, y]) == s for x, y in zip(top1, top2)], dtype=bool)


def pair_complementarity(df: pd.DataFrame, data: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    dis_rows = []
    conf_rows = []

    for pk, summ in data["pair_summaries"].items():
        a = strip_label(summ["label_a"])
        b = strip_label(summ["label_b"])
        p = data["probs"][pk]
        true = df["true_label"].map(strip_label).to_numpy()
        base_pred = df["pred_label"].map(strip_label).to_numpy()
        expert_label = df[f"{pk}_expert_label"].astype(str).to_numpy()

        true_pair = np.isin(true, [a, b])
        hard_pair = pair_mask_top2(df, a, b)
        hard_true_in_top2 = hard_pair & df["true_in_top2"].astype(bool).to_numpy() if "true_in_top2" in df.columns else hard_pair & true_pair

        subsets = {
            "true_pair": true_pair,
            "hard_top2_pair": hard_pair,
            "hard_top2_pair_true_in_top2": hard_true_in_top2,
        }

        for subset_name, mask in subsets.items():
            n = int(mask.sum())
            if n == 0:
                continue
            base_correct = base_pred[mask] == true[mask]
            expert_correct = expert_label[mask] == true[mask]
            disagree = expert_label[mask] != base_pred[mask]

            fix = (~base_correct) & expert_correct
            damage = base_correct & (~expert_correct)
            both_correct = base_correct & expert_correct
            both_wrong = (~base_correct) & (~expert_correct)

            pearson, spearman = corr_stats(
                df.loc[mask, f"{pk}_baseline_pair_prob_b"].to_numpy(dtype=float),
                df.loc[mask, f"{pk}_expert_prob_b"].to_numpy(dtype=float),
            )

            rows.append({
                "pair_key": pk,
                "pair": f"{a}<->{b}",
                "subset": subset_name,
                "n": n,
                "true_in_pair_rate": float(true_pair[mask].mean()),
                "base_correct_rate": float(base_correct.mean()),
                "expert_correct_rate": float(expert_correct.mean()),
                "agreement_rate": float((~disagree).mean()),
                "disagreement_n": int(disagree.sum()),
                "disagreement_rate": float(disagree.mean()),
                "fix_n": int(fix.sum()),
                "damage_n": int(damage.sum()),
                "net_fix_minus_damage": int(fix.sum() - damage.sum()),
                "damage_ratio": float(damage.sum() / fix.sum()) if int(fix.sum()) else None,
                "both_correct_n": int(both_correct.sum()),
                "both_wrong_n": int(both_wrong.sum()),
                "baseline_expert_prob_pearson": pearson,
                "baseline_expert_prob_spearman": spearman,
                "expert_conf_mean": float(df.loc[mask, f"{pk}_expert_conf"].mean()),
                "baseline_pair_conf_mean": float(df.loc[mask, f"{pk}_baseline_pair_conf"].mean()),
                "top12_margin_mean": float(df.loc[mask, "top12_margin"].mean()) if "top12_margin" in df.columns else None,
            })

        # disagreement samples in hard pair route
        mask = hard_pair
        base_correct_all = base_pred == true
        expert_correct_all = expert_label == true
        disagree_all = expert_label != base_pred
        selected = mask & disagree_all
        for i in np.where(selected)[0]:
            if (not base_correct_all[i]) and expert_correct_all[i]:
                trans = "fix"
            elif base_correct_all[i] and (not expert_correct_all[i]):
                trans = "damage"
            elif base_correct_all[i] and expert_correct_all[i]:
                trans = "both_correct_disagree_impossible"
            else:
                trans = "both_wrong_changed"
            dis_rows.append({
                "sample_index": int(df.at[i, "sample_index"]),
                "row_index": int(i),
                "pair_key": pk,
                "pair": f"{a}<->{b}",
                "true_label": true[i],
                "baseline_pred_label": base_pred[i],
                "expert_label": expert_label[i],
                "base_correct": bool(base_correct_all[i]),
                "expert_correct": bool(expert_correct_all[i]),
                "transition_if_override": trans,
                "true_in_top2": bool(df.at[i, "true_in_top2"]) if "true_in_top2" in df.columns else bool(true[i] in [a, b]),
                "top1_label": strip_label(df.at[i, "top1_label"]),
                "top2_label": strip_label(df.at[i, "top2_label"]),
                "top1_score": float(df.at[i, "top1_score"]) if "top1_score" in df.columns else np.nan,
                "top2_score": float(df.at[i, "top2_score"]) if "top2_score" in df.columns else np.nan,
                "top12_margin": float(df.at[i, "top12_margin"]) if "top12_margin" in df.columns else np.nan,
                "expert_prob_b": float(df.at[i, f"{pk}_expert_prob_b"]),
                "expert_conf": float(df.at[i, f"{pk}_expert_conf"]),
                "expert_margin": float(df.at[i, f"{pk}_expert_margin"]),
                "baseline_pair_prob_b": float(df.at[i, f"{pk}_baseline_pair_prob_b"]) if np.isfinite(df.at[i, f"{pk}_baseline_pair_prob_b"]) else np.nan,
                "baseline_pair_conf": float(df.at[i, f"{pk}_baseline_pair_conf"]) if np.isfinite(df.at[i, f"{pk}_baseline_pair_conf"]) else np.nan,
                "expert_minus_baseline_pair_conf": float(df.at[i, f"{pk}_expert_minus_baseline_pair_conf"]) if np.isfinite(df.at[i, f"{pk}_expert_minus_baseline_pair_conf"]) else np.nan,
            })

    dis_df = pd.DataFrame(dis_rows)

    if not dis_df.empty:
        group_cols = ["pair_key", "transition_if_override"]
        score_cols = ["expert_conf", "expert_margin", "top12_margin", "baseline_pair_conf", "expert_minus_baseline_pair_conf"]
        conf_rows = []
        for keys, g in dis_df.groupby(group_cols):
            pk, trans = keys
            row = {"pair_key": pk, "transition_if_override": trans, "n": int(len(g))}
            for c in score_cols:
                if c in g.columns:
                    row[f"{c}_mean"] = float(g[c].mean())
                    row[f"{c}_median"] = float(g[c].median())
                    row[f"{c}_p25"] = float(g[c].quantile(0.25))
                    row[f"{c}_p75"] = float(g[c].quantile(0.75))
            conf_rows.append(row)

    return pd.DataFrame(rows), dis_df, pd.DataFrame(conf_rows)


def score_separation_auc(dis_df: pd.DataFrame) -> pd.DataFrame:
    if dis_df.empty:
        return pd.DataFrame()
    rows = []
    score_cols = ["expert_conf", "expert_margin", "top12_margin", "baseline_pair_conf", "expert_minus_baseline_pair_conf"]
    for pk, g in dis_df.groupby("pair_key"):
        fd = g[g["transition_if_override"].isin(["fix", "damage"])].copy()
        if fd.empty:
            continue
        y = (fd["transition_if_override"] == "fix").astype(int).to_numpy()
        for c in score_cols:
            if c not in fd.columns:
                continue
            score = fd[c].to_numpy(dtype=float)
            auc = safe_auc_binary(y, score)
            rows.append({
                "pair_key": pk,
                "score": c,
                "n_fix_damage": int(len(fd)),
                "fix_n": int((y == 1).sum()),
                "damage_n": int((y == 0).sum()),
                "auc_for_fix": auc,
                "auc_abs_from_0p5": abs(auc - 0.5) if auc is not None else None,
                "direction_hint": "higher_score_more_fix" if auc is not None and auc >= 0.5 else "lower_score_more_fix" if auc is not None else "not_available",
            })
    return pd.DataFrame(rows)


def apply_policy(df: pd.DataFrame, data: dict, thresholds_by_pair: Dict[str, float]) -> np.ndarray:
    """Vectorized policy application for threshold-grid audit."""
    label_to_id = data["label_to_id"]
    base_pred = df["pred_id"].to_numpy(dtype=int)
    new_pred = base_pred.copy()

    top1 = df["top1_label"].map(strip_label).to_numpy()
    top2 = df["top2_label"].map(strip_label).to_numpy()

    for pk, thr in thresholds_by_pair.items():
        if pk not in PAIR_FROM_KEY:
            continue
        a, b = PAIR_FROM_KEY[pk]
        # hard pair mask without Python-level row iteration
        pair_mask = ((top1 == a) & (top2 == b)) | ((top1 == b) & (top2 == a))
        if not np.any(pair_mask):
            continue
        conf = df[f"{pk}_expert_conf"].to_numpy(dtype=float)
        apply_mask = pair_mask & (conf >= float(thr))
        if not np.any(apply_mask):
            continue
        expert_labels = df[f"{pk}_expert_label"].map(strip_label).to_numpy()
        expert_ids = np.array([label_to_id[str(x)] for x in expert_labels[apply_mask]], dtype=int)
        new_pred[apply_mask] = expert_ids

    return new_pred

def threshold_policy_search(df: pd.DataFrame, data: dict, thresholds: List[float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    y_true = df["true_id"].to_numpy(dtype=int)
    base_pred = df["pred_id"].to_numpy(dtype=int)
    label_names = data["label_names"]
    base_metrics = macro_metrics(y_true, base_pred, label_names)

    global_rows = []
    for thr in thresholds:
        pred = apply_policy(df, data, {"RS": thr, "RT": thr, "ST": thr})
        met = macro_metrics(y_true, pred, label_names)
        trans = transition_stats(y_true, base_pred, pred)
        global_rows.append({
            "policy": f"global_thr_{thr:g}",
            "RS_thr": thr,
            "RT_thr": thr,
            "ST_thr": thr,
            **met,
            "delta_macro_f1": met["macro_f1"] - base_metrics["macro_f1"],
            "delta_accuracy": met["accuracy"] - base_metrics["accuracy"],
            "delta_weighted_f1": met["weighted_f1"] - base_metrics["weighted_f1"],
            **trans,
        })

    grid_rows = []
    for rs, rt, st in itertools.product(thresholds, thresholds, thresholds):
        pred = apply_policy(df, data, {"RS": rs, "RT": rt, "ST": st})
        met = macro_metrics(y_true, pred, label_names)
        trans = transition_stats(y_true, base_pred, pred)
        grid_rows.append({
            "policy": f"pair_thr_RS{rs:g}_RT{rt:g}_ST{st:g}",
            "RS_thr": rs,
            "RT_thr": rt,
            "ST_thr": st,
            **met,
            "delta_macro_f1": met["macro_f1"] - base_metrics["macro_f1"],
            "delta_accuracy": met["accuracy"] - base_metrics["accuracy"],
            "delta_weighted_f1": met["weighted_f1"] - base_metrics["weighted_f1"],
            **trans,
        })

    global_df = pd.DataFrame(global_rows).sort_values(["macro_f1", "net_gain"], ascending=[False, False])
    grid_df = pd.DataFrame(grid_rows).sort_values(["macro_f1", "net_gain"], ascending=[False, False])
    return global_df, grid_df


def transition_sets(pred_df: pd.DataFrame, prefix: str) -> dict:
    if pred_df is None:
        return {}
    if "transition" not in pred_df.columns:
        return {}
    return {
        "fixed": set(pred_df.loc[pred_df["transition"] == "fixed", "sample_index"].astype(int)),
        "damaged": set(pred_df.loc[pred_df["transition"] == "damaged", "sample_index"].astype(int)),
        "changed": set(pred_df.loc[pred_df[f"{prefix}_pred_label"] != pred_df["base_pred_label"], "sample_index"].astype(int))
            if f"{prefix}_pred_label" in pred_df.columns and "base_pred_label" in pred_df.columns else set(),
    }


def e1a0_overlap(e1a0_dir: Optional[Path], e1a1_best: Optional[pd.DataFrame]) -> pd.DataFrame:
    if e1a0_dir is None or not e1a0_dir.exists() or e1a1_best is None:
        return pd.DataFrame()
    e1a0_pred_path = e1a0_dir / "E1a0_best_policy_predictions.csv"
    if not e1a0_pred_path.exists():
        return pd.DataFrame()
    e1a0 = pd.read_csv(e1a0_pred_path)
    e1a0_sets = transition_sets(e1a0, "e1a0")
    e1a1_sets = transition_sets(e1a1_best, "e1a1")
    rows = []
    for name in ["fixed", "damaged", "changed"]:
        a = e1a0_sets.get(name, set())
        b = e1a1_sets.get(name, set())
        inter = a & b
        union = a | b
        rows.append({
            "set_name": name,
            "e1a0_n": len(a),
            "e1a1_n": len(b),
            "overlap_n": len(inter),
            "e1a0_only_n": len(a - b),
            "e1a1_only_n": len(b - a),
            "jaccard": float(len(inter) / len(union)) if len(union) else None,
        })
    return pd.DataFrame(rows)


def write_summary_md(out_dir: Path, summary: dict) -> None:
    best_global = summary.get("best_global_threshold_policy", {})
    best_pair = summary.get("best_pair_threshold_policy", {})
    text = f"""# E1a2 Micro-signal / Complementarity Audit

## Goal

Check whether E1a1 binary D3 experts learn useful micro-signal different from baseline,
or only learn a similar boundary that causes fix and damage to cancel out.

## Key result

```text
E1a1 best original policy:
macro-F1 = {summary.get('e1a1_best_original_macro_f1')}
net_gain = {summary.get('e1a1_best_original_net_gain')}

Best global threshold re-audit:
policy   = {best_global.get('policy')}
macro-F1 = {best_global.get('macro_f1')}
net_gain = {best_global.get('net_gain')}
damage_ratio = {best_global.get('damage_ratio')}

Best per-pair threshold grid:
policy   = {best_pair.get('policy')}
macro-F1 = {best_pair.get('macro_f1')}
net_gain = {best_pair.get('net_gain')}
damage_ratio = {best_pair.get('damage_ratio')}
```

## Interpretation guide

- If disagreement fix <= damage, expert does not have enough complementary signal.
- If confidence scores cannot separate fix from damage, gating is unlikely to rescue much.
- If per-pair thresholds improve materially, the problem is partly calibration/gating.
- If E1a0 fixes many samples E1a1 misses, tree/tabular expert likely captures interactions D3 attention does not.

## Main files

- `E1a2_pair_complementarity.csv`
- `E1a2_disagreement_samples.csv`
- `E1a2_fix_damage_confidence_stats.csv`
- `E1a2_score_separation_auc.csv`
- `E1a2_global_threshold_policy.csv`
- `E1a2_pair_threshold_grid.csv`
- `E1a2_e1a0_vs_e1a1_overlap.csv`
"""
    (out_dir / "E1a2_summary.md").write_text(text, encoding="utf-8")


def zip_dir(src_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(src_dir.rglob("*")):
            if p.is_file() and p != zip_path:
                z.write(p, p.relative_to(src_dir.parent))


def main():
    parser = argparse.ArgumentParser(description="E1a2 micro-signal complementarity audit")
    parser.add_argument("--e1a1-dir", default="05_test/outputs/E1a1_binary_d3_attention_expert")
    parser.add_argument("--e1a0-dir", default="05_test/outputs/E1a0_full_feature_binary_expert")
    parser.add_argument("--out-dir", default="05_test/outputs/E1a2_micro_signal_complementarity_audit")
    parser.add_argument("--thresholds", default="0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95")
    parser.add_argument("--make-zip", action="store_true", default=True)
    parser.add_argument("--no-zip", dest="make_zip", action="store_false")
    args = parser.parse_args()

    repo_root = repo_root_from_here()
    e1a1_path = resolve_path(args.e1a1_dir, repo_root)
    e1a0_path = resolve_path(args.e1a0_dir, repo_root) if args.e1a0_dir else None
    out_dir = resolve_path(args.out_dir, repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    e1a1_dir, tmp1 = prepare_input_path(e1a1_path, expected_name_contains="E1a1")
    e1a0_dir = None
    tmp0 = None
    if e1a0_path and e1a0_path.exists():
        e1a0_dir, tmp0 = prepare_input_path(e1a0_path, expected_name_contains="E1a0")

    print(f"[E1a2] e1a1_dir={e1a1_dir}", flush=True)
    print(f"[E1a2] e1a0_dir={e1a0_dir}", flush=True)
    print(f"[E1a2] out_dir={out_dir}", flush=True)

    data = load_e1a1(e1a1_dir)
    df = add_expert_columns(data["base"], data)

    # Save enriched per-sample context for later debugging.
    df.to_csv(out_dir / "E1a2_enriched_baseline_expert_context.csv", index=False)

    pair_df, dis_df, conf_stats = pair_complementarity(df, data)
    pair_df.to_csv(out_dir / "E1a2_pair_complementarity.csv", index=False)
    dis_df.to_csv(out_dir / "E1a2_disagreement_samples.csv", index=False)
    conf_stats.to_csv(out_dir / "E1a2_fix_damage_confidence_stats.csv", index=False)

    sep_auc = score_separation_auc(dis_df)
    sep_auc.to_csv(out_dir / "E1a2_score_separation_auc.csv", index=False)

    thresholds = [float(x) for x in str(args.thresholds).split(",") if str(x).strip()]
    global_df, grid_df = threshold_policy_search(df, data, thresholds)
    global_df.to_csv(out_dir / "E1a2_global_threshold_policy.csv", index=False)
    grid_df.to_csv(out_dir / "E1a2_pair_threshold_grid.csv", index=False)

    overlap_df = e1a0_overlap(e1a0_dir, data["best_pred"])
    overlap_df.to_csv(out_dir / "E1a2_e1a0_vs_e1a1_overlap.csv", index=False)

    # Optional concise pivot for disagreement transition counts.
    if not dis_df.empty:
        piv = (
            dis_df.pivot_table(
                index=["pair_key"],
                columns="transition_if_override",
                values="sample_index",
                aggfunc="count",
                fill_value=0,
            )
            .reset_index()
        )
        piv.to_csv(out_dir / "E1a2_disagreement_transition_counts.csv", index=False)

    # Original best policy from E1a1 summary if available.
    orig_macro = None
    orig_net = None
    summ_path = e1a1_dir / "E1a1_summary.json"
    if summ_path.exists():
        summ = load_json(summ_path)
        orig_macro = summ.get("best_metrics", {}).get("macro_f1")
        orig_net = summ.get("best_transition", {}).get("net_gain")

    best_global = global_df.iloc[0].to_dict() if len(global_df) else {}
    best_pair = grid_df.iloc[0].to_dict() if len(grid_df) else {}

    hard_top2_total = int(sum(pair_mask_top2(df, *PAIR_FROM_KEY[pk]).sum() for pk in ["RS", "RT", "ST"]))
    true_in_top2_rate = float(df["true_in_top2"].mean()) if "true_in_top2" in df.columns else None
    wrong_mask = ~(df["correct"].astype(bool).to_numpy())
    wrong_true_in_top2_rate = float(df.loc[wrong_mask, "true_in_top2"].mean()) if "true_in_top2" in df.columns and wrong_mask.sum() else None

    # Complementarity headline from hard_top2_pair rows.
    hard_rows = pair_df[pair_df["subset"] == "hard_top2_pair"].copy()
    comp_headline = {}
    if not hard_rows.empty:
        comp_headline = {
            "hard_top2_pair_fix_n_sum": int(hard_rows["fix_n"].sum()),
            "hard_top2_pair_damage_n_sum": int(hard_rows["damage_n"].sum()),
            "hard_top2_pair_net_sum": int(hard_rows["net_fix_minus_damage"].sum()),
            "hard_top2_pair_disagreement_n_sum": int(hard_rows["disagreement_n"].sum()),
            "hard_top2_pair_expert_correct_rate_weighted": float(np.average(hard_rows["expert_correct_rate"], weights=hard_rows["n"])),
            "hard_top2_pair_base_correct_rate_weighted": float(np.average(hard_rows["base_correct_rate"], weights=hard_rows["n"])),
        }

    summary = {
        "stage": "E1a2_micro_signal_complementarity_audit",
        "purpose": "Audit whether E1a1 experts provide complementary micro-signal versus baseline.",
        "e1a1_dir": str(e1a1_dir),
        "e1a0_dir": str(e1a0_dir) if e1a0_dir else None,
        "n_val": int(len(df)),
        "true_in_top2_rate_all": true_in_top2_rate,
        "true_in_top2_rate_baseline_wrong": wrong_true_in_top2_rate,
        "hard_top2_total_count_summed_by_pair": hard_top2_total,
        "e1a1_best_original_macro_f1": orig_macro,
        "e1a1_best_original_net_gain": orig_net,
        "complementarity_headline": comp_headline,
        "best_global_threshold_policy": best_global,
        "best_pair_threshold_policy": best_pair,
        "score_separation_best_rows": sep_auc.sort_values("auc_abs_from_0p5", ascending=False).head(10).to_dict(orient="records") if len(sep_auc) else [],
        "e1a0_vs_e1a1_overlap": overlap_df.to_dict(orient="records") if len(overlap_df) else [],
        "outputs": {
            "enriched_context": str(out_dir / "E1a2_enriched_baseline_expert_context.csv"),
            "pair_complementarity": str(out_dir / "E1a2_pair_complementarity.csv"),
            "disagreement_samples": str(out_dir / "E1a2_disagreement_samples.csv"),
            "confidence_stats": str(out_dir / "E1a2_fix_damage_confidence_stats.csv"),
            "score_separation_auc": str(out_dir / "E1a2_score_separation_auc.csv"),
            "global_threshold_policy": str(out_dir / "E1a2_global_threshold_policy.csv"),
            "pair_threshold_grid": str(out_dir / "E1a2_pair_threshold_grid.csv"),
            "e1a0_vs_e1a1_overlap": str(out_dir / "E1a2_e1a0_vs_e1a1_overlap.csv"),
        },
    }
    save_json(out_dir / "E1a2_summary.json", summary)
    write_summary_md(out_dir, summary)

    if args.make_zip:
        zip_path = out_dir.with_suffix(".zip")
        zip_dir(out_dir, zip_path)
        print(f"[E1a2] zipped outputs: {zip_path}", flush=True)

    print("[E1a2] done.", flush=True)
    print(f"[E1a2] hard complementarity={comp_headline}", flush=True)
    print(f"[E1a2] best_global={best_global}", flush=True)
    print(f"[E1a2] best_pair_threshold={best_pair}", flush=True)

    if tmp1 is not None:
        tmp1.cleanup()
    if tmp0 is not None:
        tmp0.cleanup()


if __name__ == "__main__":
    main()
