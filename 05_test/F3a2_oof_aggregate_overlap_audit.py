#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F3a2 Aggregate OOF overlap reproduction audit.

Input:
  F3a1 fold outputs for folds 0..4.

Purpose:
  Combine clean OOF predictions for the whole original train split and check:
    - integrity: every original train row appears once
    - OOF macro-F1 / confusion
    - hard malware pairs
    - hard L3 families
    - fold stability of hard patterns
    - train-only selected hard groups for later clean repair

This is still diagnostic/mining only. It does NOT use official validation.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score


CLASS_NAMES_DEFAULT = ["Benign", "Ransomware", "Spyware", "Trojan"]
MALWARE_CLASSES_DEFAULT = ["Ransomware", "Spyware", "Trojan"]


def log(msg: str):
    print(f"[F3a2] {msg}", flush=True)


def repo_root_from_here() -> Path:
    p = Path(__file__).resolve()
    if p.parent.name == "05_test":
        return p.parents[1]
    return Path.cwd().resolve()


def resolve_path(p: str | Path, root: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (root / p).resolve()


def parse_list(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def safe_md(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df is None or len(df) == 0:
        return "_empty_"
    try:
        return df.head(max_rows).to_markdown(index=False)
    except Exception:
        return df.head(max_rows).to_string(index=False)


def read_text_from_zip(zpath: Path, suffix: str) -> Optional[bytes]:
    if not zpath.exists():
        return None
    with zipfile.ZipFile(zpath) as z:
        matches = [n for n in z.namelist() if n.endswith(suffix)]
        if not matches:
            return None
        return z.read(matches[0])


def load_fold_output(base_dir: Path, fold_id: int) -> tuple[pd.DataFrame, Dict[str, Any]]:
    # Prefer extracted dir.
    d = base_dir / f"F3a1_oof_fold{fold_id}_train_export"
    pred_suffix = f"F3a1_fold{fold_id}_oof_predictions_logits_probs.csv"
    metrics_suffix = f"F3a1_fold{fold_id}_metrics.json"
    cfg_suffix = "config.json"

    if d.exists():
        pred_path = d / pred_suffix
        metrics_path = d / metrics_suffix
        cfg_path = d / cfg_suffix
        if not pred_path.exists():
            # fallback recursive search
            matches = list(d.rglob(pred_suffix))
            if not matches:
                raise FileNotFoundError(f"Missing prediction file for fold {fold_id} under {d}")
            pred_path = matches[0]
        pred = pd.read_csv(pred_path)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        meta = {"metrics": metrics, "config": cfg, "source": str(d)}
        pred["fold"] = int(fold_id)
        return pred, meta

    # Fallback zip.
    zpath = base_dir / f"F3a1_oof_fold{fold_id}_train_export.zip"
    if not zpath.exists():
        raise FileNotFoundError(f"Missing fold {fold_id} output dir or zip: {d} / {zpath}")

    raw_pred = read_text_from_zip(zpath, pred_suffix)
    if raw_pred is None:
        raise FileNotFoundError(f"Missing {pred_suffix} in {zpath}")
    import io
    pred = pd.read_csv(io.BytesIO(raw_pred))

    raw_metrics = read_text_from_zip(zpath, metrics_suffix)
    metrics = json.loads(raw_metrics.decode("utf-8")) if raw_metrics else {}

    raw_cfg = read_text_from_zip(zpath, cfg_suffix)
    cfg = json.loads(raw_cfg.decode("utf-8")) if raw_cfg else {}

    pred["fold"] = int(fold_id)
    return pred, {"metrics": metrics, "config": cfg, "source": str(zpath)}


def compute_metrics(df: pd.DataFrame, class_names: List[str]) -> Dict[str, Any]:
    return {
        "n": int(len(df)),
        "accuracy": float(accuracy_score(df["true_label"], df["pred_label"])),
        "macro_f1": float(f1_score(df["true_label"], df["pred_label"], labels=class_names, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(df["true_label"], df["pred_label"], labels=class_names, average="weighted", zero_division=0)),
    }


def family_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fam, g in df.groupby("family", dropna=False):
        true_label = g["true_label"].mode().iloc[0] if len(g) else ""
        pred_counts = g["pred_label"].value_counts()
        rows.append({
            "family": fam,
            "true_label": true_label,
            "support": int(len(g)),
            "correct": int(g["correct"].sum()),
            "accuracy": float(g["correct"].mean()),
            "error_rate": float(1.0 - g["correct"].mean()),
            "top_pred": pred_counts.index[0] if len(pred_counts) else "",
            "top_pred_count": int(pred_counts.iloc[0]) if len(pred_counts) else 0,
            "mean_true_prob": float(g["true_prob"].mean()) if "true_prob" in g else np.nan,
            "mean_pred_prob": float(g["pred_prob"].mean()) if "pred_prob" in g else np.nan,
            "true_in_top2_rate": float(g["true_in_top2"].mean()) if "true_in_top2" in g else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["error_rate", "support"], ascending=[False, False])


def pair_confusion(df: pd.DataFrame, malware_classes: List[str]) -> pd.DataFrame:
    wrong = df[(df["correct"] == False) & (df["true_label"].isin(malware_classes))].copy()
    rows = []
    for true in malware_classes:
        gtrue = df[df["true_label"] == true]
        for pred in malware_classes:
            if pred == true:
                continue
            g = wrong[(wrong["true_label"] == true) & (wrong["pred_label"] == pred)]
            rows.append({
                "true_label": true,
                "pred_label": pred,
                "count": int(len(g)),
                "true_support": int(len(gtrue)),
                "rate_within_true": float(len(g) / max(len(gtrue), 1)),
                "mean_true_prob": float(g["true_prob"].mean()) if len(g) and "true_prob" in g else np.nan,
                "mean_pred_prob": float(g["pred_prob"].mean()) if len(g) and "pred_prob" in g else np.nan,
                "true_in_top2_rate": float(g["true_in_top2"].mean()) if len(g) and "true_in_top2" in g else np.nan,
                "top_family": g["family"].value_counts().index[0] if len(g) else "",
                "top_family_count": int(g["family"].value_counts().iloc[0]) if len(g) else 0,
            })
    return pd.DataFrame(rows).sort_values("count", ascending=False)


def hard_pair_family(df: pd.DataFrame, malware_classes: List[str]) -> pd.DataFrame:
    wrong = df[(df["correct"] == False) & (df["true_label"].isin(malware_classes))].copy()
    if len(wrong) == 0:
        return pd.DataFrame()
    res = wrong.groupby(["true_label", "pred_label", "family"], dropna=False).agg(
        count=("original_row_id", "count"),
        mean_true_prob=("true_prob", "mean"),
        mean_pred_prob=("pred_prob", "mean"),
        true_in_top2_rate=("true_in_top2", "mean"),
        folds_present=("fold", lambda s: int(s.nunique())),
    ).reset_index().sort_values("count", ascending=False)
    return res


def fold_stability_family(df: pd.DataFrame, families: List[str]) -> pd.DataFrame:
    rows = []
    for fam in families:
        gf = df[df["family"] == fam]
        for fold, g in gf.groupby("fold"):
            rows.append({
                "family": fam,
                "fold": int(fold),
                "support": int(len(g)),
                "accuracy": float(g["correct"].mean()) if len(g) else np.nan,
                "error_rate": float(1.0 - g["correct"].mean()) if len(g) else np.nan,
                "top_pred": g["pred_label"].value_counts().index[0] if len(g) else "",
                "top_pred_count": int(g["pred_label"].value_counts().iloc[0]) if len(g) else 0,
            })
    return pd.DataFrame(rows).sort_values(["family", "fold"])


def fold_stability_pair(df: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    wrong = df[df["correct"] == False].copy()
    for _, r in pairs.iterrows():
        true = r["true_label"]
        pred = r["pred_label"]
        for fold, gfold in df[df["true_label"] == true].groupby("fold"):
            g = wrong[(wrong["fold"] == fold) & (wrong["true_label"] == true) & (wrong["pred_label"] == pred)]
            rows.append({
                "true_label": true,
                "pred_label": pred,
                "fold": int(fold),
                "count": int(len(g)),
                "true_support_fold": int(len(gfold)),
                "rate_within_true_fold": float(len(g) / max(len(gfold), 1)),
            })
    return pd.DataFrame(rows).sort_values(["true_label", "pred_label", "fold"])


def select_train_only_hard_groups(
    *,
    family_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    pair_family_df: pd.DataFrame,
    min_family_support: int,
    min_family_error_rate: float,
    min_pair_count: int,
    min_pair_rate: float,
    min_pair_family_count: int,
) -> Dict[str, Any]:
    hard_families = family_df[
        (family_df["true_label"] != "Benign")
        & (family_df["support"] >= min_family_support)
        & (family_df["error_rate"] >= min_family_error_rate)
    ].copy()

    hard_pairs = pair_df[
        (pair_df["count"] >= min_pair_count)
        & (pair_df["rate_within_true"] >= min_pair_rate)
    ].copy()

    hard_pair_families = pair_family_df[pair_family_df["count"] >= min_pair_family_count].copy()

    return {
        "selection_source": "train_only_clean_OOF",
        "validation_used": False,
        "criteria": {
            "min_family_support": int(min_family_support),
            "min_family_error_rate": float(min_family_error_rate),
            "min_pair_count": int(min_pair_count),
            "min_pair_rate": float(min_pair_rate),
            "min_pair_family_count": int(min_pair_family_count),
        },
        "hard_families": hard_families[[
            "family", "true_label", "support", "accuracy", "error_rate", "top_pred", "top_pred_count", "true_in_top2_rate"
        ]].to_dict(orient="records"),
        "hard_pairs": hard_pairs[[
            "true_label", "pred_label", "count", "true_support", "rate_within_true", "mean_true_prob", "mean_pred_prob", "true_in_top2_rate", "top_family", "top_family_count"
        ]].to_dict(orient="records"),
        "hard_pair_families": hard_pair_families[[
            "true_label", "pred_label", "family", "count", "mean_true_prob", "mean_pred_prob", "true_in_top2_rate", "folds_present"
        ]].to_dict(orient="records"),
    }


def write_report(
    out_dir: Path,
    metrics: Dict[str, Any],
    fold_metrics: pd.DataFrame,
    family_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    pair_family_df: pd.DataFrame,
    family_stability: pd.DataFrame,
    pair_stability: pd.DataFrame,
    selected: Dict[str, Any],
    integrity: Dict[str, Any],
):
    lines = []
    lines.append("# F3a2 Aggregate OOF overlap reproduction audit\n")
    lines.append("## Scope\n")
    lines.append("```text")
    lines.append("Train-only OOF audit. Official validation is not used.")
    lines.append("Each row is predicted once by a model that did not train on that row.")
    lines.append("Outer fold is not used for early stopping in F3a1.")
    lines.append("```")
    lines.append("\n## Integrity\n")
    lines.append("```json")
    lines.append(json.dumps(integrity, indent=2, default=str))
    lines.append("```")
    lines.append("\n## Overall OOF metrics\n")
    lines.append("```json")
    lines.append(json.dumps(metrics, indent=2, default=str))
    lines.append("```")
    lines.append("\n## Fold metrics\n")
    lines.append(safe_md(fold_metrics, 10))
    lines.append("\n## Hardest families from train-only OOF\n")
    lines.append(safe_md(family_df, 20))
    lines.append("\n## Malware pair confusion from train-only OOF\n")
    lines.append(safe_md(pair_df, 12))
    lines.append("\n## Hard pair-family groups\n")
    lines.append(safe_md(pair_family_df, 20))
    lines.append("\n## Selected hard groups for next clean method design\n")
    lines.append("```json")
    # Avoid huge report.
    short_selected = {
        **selected,
        "hard_families": selected["hard_families"][:15],
        "hard_pairs": selected["hard_pairs"][:10],
        "hard_pair_families": selected["hard_pair_families"][:20],
    }
    lines.append(json.dumps(short_selected, indent=2, default=str))
    lines.append("```")
    lines.append("\n## Interpretation\n")
    lines.append("```text")
    lines.append("If these OOF hard families/pairs match the earlier validation audit, then the pattern is not a validation-only artifact.")
    lines.append("The selected hard groups may be used to design a clean train-only repair method.")
    lines.append("Do not use official validation to add/remove hard groups or tune thresholds.")
    lines.append("```")
    (out_dir / "F3a2_oof_aggregate_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-base-dir", default="05_test/outputs")
    ap.add_argument("--out-dir", default="05_test/outputs/F3a2_oof_aggregate_overlap_audit")
    ap.add_argument("--combined-zip", default="05_test/outputs/F3a2_oof_aggregate_overlap_audit.zip")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--class-names", default="Benign,Ransomware,Spyware,Trojan")
    ap.add_argument("--malware-classes", default="Ransomware,Spyware,Trojan")
    ap.add_argument("--expected-n", type=int, default=46876)
    ap.add_argument("--min-family-support", type=int, default=500)
    ap.add_argument("--min-family-error-rate", type=float, default=0.30)
    ap.add_argument("--min-pair-count", type=int, default=700)
    ap.add_argument("--min-pair-rate", type=float, default=0.08)
    ap.add_argument("--min-pair-family-count", type=int, default=150)
    args = ap.parse_args()

    root = repo_root_from_here()
    fold_base_dir = resolve_path(args.fold_base_dir, root)
    out_dir = resolve_path(args.out_dir, root)
    zip_path = resolve_path(args.combined_zip, root)
    out_dir.mkdir(parents=True, exist_ok=True)

    folds = [int(x.strip()) for x in args.folds.split(",") if x.strip()]
    class_names = parse_list(args.class_names) or CLASS_NAMES_DEFAULT
    malware_classes = parse_list(args.malware_classes) or MALWARE_CLASSES_DEFAULT

    dfs = []
    metas = []
    for fold_id in folds:
        df, meta = load_fold_output(fold_base_dir, fold_id)
        df["fold"] = int(fold_id)
        dfs.append(df)
        metas.append({"fold": fold_id, **meta})
        log(f"loaded fold {fold_id}: n={len(df)} source={meta.get('source')}")

    all_oof = pd.concat(dfs, ignore_index=True)
    all_oof = all_oof.sort_values("original_row_id").reset_index(drop=True)
    all_oof.to_csv(out_dir / "F3a2_all_oof_predictions_logits_probs.csv", index=False)

    # Integrity.
    n = len(all_oof)
    unique_n = int(all_oof["original_row_id"].nunique())
    dup_n = int(all_oof["original_row_id"].duplicated().sum())
    missing = []
    if args.expected_n > 0:
        expected_set = set(range(int(args.expected_n)))
        got_set = set(int(x) for x in all_oof["original_row_id"].tolist())
        missing = sorted(expected_set - got_set)
    integrity = {
        "n_rows": int(n),
        "unique_original_row_id": unique_n,
        "duplicate_original_row_id_count": dup_n,
        "expected_n": int(args.expected_n),
        "missing_original_row_id_count": int(len(missing)),
        "min_original_row_id": int(all_oof["original_row_id"].min()) if n else None,
        "max_original_row_id": int(all_oof["original_row_id"].max()) if n else None,
        "folds": folds,
        "fold_counts": {str(int(k)): int(v) for k, v in all_oof["fold"].value_counts().sort_index().to_dict().items()},
        "pass": bool(dup_n == 0 and (args.expected_n <= 0 or (n == args.expected_n and unique_n == args.expected_n and len(missing) == 0))),
    }
    (out_dir / "F3a2_integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")

    metrics = compute_metrics(all_oof, class_names)
    (out_dir / "F3a2_overall_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    report = classification_report(all_oof["true_label"], all_oof["pred_label"], labels=class_names, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(out_dir / "F3a2_oof_classification_report.csv")
    cm = confusion_matrix(all_oof["true_label"], all_oof["pred_label"], labels=class_names)
    pd.DataFrame(cm, index=[f"true_{c}" for c in class_names], columns=[f"pred_{c}" for c in class_names]).to_csv(
        out_dir / "F3a2_oof_confusion_matrix.csv"
    )

    fold_metric_rows = []
    for fold_id, g in all_oof.groupby("fold"):
        row = compute_metrics(g, class_names)
        row["fold"] = int(fold_id)
        row["n"] = int(len(g))
        fold_metric_rows.append(row)
    fold_metrics = pd.DataFrame(fold_metric_rows).sort_values("fold")
    fold_metrics.to_csv(out_dir / "F3a2_fold_metrics.csv", index=False)

    family_df = family_difficulty(all_oof)
    pair_df = pair_confusion(all_oof, malware_classes)
    pair_family_df = hard_pair_family(all_oof, malware_classes)

    family_df.to_csv(out_dir / "F3a2_family_difficulty.csv", index=False)
    pair_df.to_csv(out_dir / "F3a2_malware_pair_confusion.csv", index=False)
    pair_family_df.to_csv(out_dir / "F3a2_hard_pair_family_summary.csv", index=False)

    top_families = family_df[family_df["true_label"] != "Benign"].head(10)["family"].tolist()
    top_pairs = pair_df.head(6)
    family_stability = fold_stability_family(all_oof, top_families)
    pair_stability = fold_stability_pair(all_oof, top_pairs)
    family_stability.to_csv(out_dir / "F3a2_top_family_fold_stability.csv", index=False)
    pair_stability.to_csv(out_dir / "F3a2_top_pair_fold_stability.csv", index=False)

    selected = select_train_only_hard_groups(
        family_df=family_df,
        pair_df=pair_df,
        pair_family_df=pair_family_df,
        min_family_support=int(args.min_family_support),
        min_family_error_rate=float(args.min_family_error_rate),
        min_pair_count=int(args.min_pair_count),
        min_pair_rate=float(args.min_pair_rate),
        min_pair_family_count=int(args.min_pair_family_count),
    )
    (out_dir / "F3a2_train_only_selected_hard_groups.json").write_text(json.dumps(selected, indent=2, default=str), encoding="utf-8")

    config = {
        "experiment": "F3a2_oof_aggregate_overlap_audit",
        "official_validation_used": False,
        "training_performed": False,
        "fold_base_dir": str(fold_base_dir),
        "folds": folds,
        "class_names": class_names,
        "malware_classes": malware_classes,
        "integrity": integrity,
        "metrics": metrics,
        "fold_sources": metas,
        "selection_source": "train_only_clean_OOF",
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

    write_report(
        out_dir=out_dir,
        metrics=metrics,
        fold_metrics=fold_metrics,
        family_df=family_df,
        pair_df=pair_df,
        pair_family_df=pair_family_df,
        family_stability=family_stability,
        pair_stability=pair_stability,
        selected=selected,
        integrity=integrity,
    )

    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out_dir.rglob("*"):
            if p.is_file() and p != zip_path:
                z.write(p, p.relative_to(out_dir.parent))

    log(f"integrity_pass={integrity['pass']}")
    log(f"metrics={metrics}")
    log(f"zip={zip_path}")
    log("DONE")


if __name__ == "__main__":
    main()
