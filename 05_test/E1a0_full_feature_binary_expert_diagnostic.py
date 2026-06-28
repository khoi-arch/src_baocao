#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E1a0 full-feature binary expert diagnostic v2-fast-policy.

Purpose
-------
Do NOT modify official source files.
Do NOT train a new 4-class backbone.
Train 3 standalone binary experts using FULL features, one pair at a time:
  - Ransomware vs Spyware
  - Ransomware vs Trojan
  - Spyware vs Trojan

Then test a top-2 intervention policy:
  main baseline predicts first;
  if baseline top1/top2 are both malware and form one hard pair;
  call the corresponding binary expert;
  expert chooses between those two classes only.

This answers the user's core question:
  "If the input is the same/full, but the task is forced to 2-class,
   does a binary expert beat the 4-class baseline enough?"

Default expected repo structure:
  02_src/07_train.py
  03_outputs/05_dataset/dataset.npz
  03_outputs/05_dataset/metadata.json
  01_split/train_raw.csv
  01_split/val_raw.csv
  03_outputs/06_model/val_predictions_best.csv
  05_test/outputs/B0_wrong_top2_audit/val_predictions_with_probs.csv

v2 fix: vectorized top-2 policy evaluation; avoids per-sample predict_proba slowdown.

Outputs:
  05_test/outputs/E1a0_full_feature_binary_expert/
  05_test/outputs/E1a0_full_feature_binary_expert.zip
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except Exception as e:
    raise RuntimeError("E1a0 requires scikit-learn. Install with: pip install scikit-learn") from e


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

MALWARE_LABELS = {"Ransomware", "Spyware", "Trojan"}


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


def strip_label(x) -> str:
    return str(x).strip()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def import_official_train(train_script: Path):
    if not train_script.exists():
        raise FileNotFoundError(f"official 07_train.py not found: {train_script}")

    src_dir = train_script.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    spec = importlib.util.spec_from_file_location("official_07_train_for_e1a0", str(train_script))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import official train script: {train_script}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def normalize_label_mapping(meta: dict) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    label_mapping = meta.get("label_mapping")
    if not isinstance(label_mapping, dict):
        raise ValueError("metadata.json missing label_mapping dict")
    pairs = sorted([(strip_label(label), int(idx)) for label, idx in label_mapping.items()], key=lambda x: x[1])
    label_names = [p[0] for p in pairs]
    label_to_id = {label: idx for label, idx in pairs}
    id_to_label = {idx: label for label, idx in pairs}
    return label_names, label_to_id, id_to_label


def load_official_inputs(args, repo_root: Path):
    """
    Reuse official 07_train.py to avoid reimplementing dataset/raw_scaled logic.
    """
    train_mod = import_official_train(resolve_path(args.official_train, repo_root))
    dataset_npz = resolve_path(args.dataset_npz, repo_root)
    metadata_json = resolve_path(args.metadata_json, repo_root)

    data, meta = train_mod.load_dataset(dataset_npz, metadata_json)
    label_names, label_to_id, id_to_label = normalize_label_mapping(meta)
    feature_names = [str(x) for x in meta["feature_names"]]

    X_train_bin = data["X_train_bin"].astype(np.int64)
    X_train_offset = data["X_train_offset"].astype(np.float32)
    y_train = data["y_train"].astype(np.int64)

    X_val_bin = data["X_val_bin"].astype(np.int64)
    X_val_offset = data["X_val_offset"].astype(np.float32)
    y_val = data["y_val"].astype(np.int64)

    spec = train_mod.RUN_SPECS["D3"]
    raw_args = SimpleNamespace(
        train_raw=str(resolve_path(args.train_raw, repo_root)),
        val_raw=str(resolve_path(args.val_raw, repo_root)),
    )
    X_train_raw, X_val_raw, continuous_info = train_mod.load_continuous_for_run(
        spec=spec,
        meta=meta,
        args=raw_args,
        train_shape=X_train_bin.shape,
        val_shape=X_val_bin.shape,
    )

    num_bins = int(meta.get("num_bins", 0) or meta.get("K", 0) or (max(int(X_train_bin.max()), int(X_val_bin.max())) + 1))
    denom = max(1, num_bins - 1)

    X_train_bin_norm = X_train_bin.astype(np.float32) / float(denom)
    X_val_bin_norm = X_val_bin.astype(np.float32) / float(denom)

    X_train_d3_scalar = X_train_bin_norm + (X_train_offset.astype(np.float32) / float(denom))
    X_val_d3_scalar = X_val_bin_norm + (X_val_offset.astype(np.float32) / float(denom))

    reps_train = {
        "raw_scaled": X_train_raw.astype(np.float32),
        "bin_norm": X_train_bin_norm.astype(np.float32),
        "offset": X_train_offset.astype(np.float32),
        "d3_scalar": X_train_d3_scalar.astype(np.float32),
    }
    reps_val = {
        "raw_scaled": X_val_raw.astype(np.float32),
        "bin_norm": X_val_bin_norm.astype(np.float32),
        "offset": X_val_offset.astype(np.float32),
        "d3_scalar": X_val_d3_scalar.astype(np.float32),
    }

    if args.include_concat:
        reps_train["concat_all"] = np.concatenate(
            [reps_train["raw_scaled"], reps_train["bin_norm"], reps_train["offset"], reps_train["d3_scalar"]],
            axis=1,
        ).astype(np.float32)
        reps_val["concat_all"] = np.concatenate(
            [reps_val["raw_scaled"], reps_val["bin_norm"], reps_val["offset"], reps_val["d3_scalar"]],
            axis=1,
        ).astype(np.float32)

    return {
        "meta": meta,
        "feature_names": feature_names,
        "label_names": label_names,
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "num_bins": num_bins,
        "continuous_info": continuous_info,
        "y_train": y_train,
        "y_val": y_val,
        "reps_train": reps_train,
        "reps_val": reps_val,
    }


def normalize_pred_df(df: pd.DataFrame, label_to_id: Dict[str, int], id_to_label: Dict[int, str]) -> pd.DataFrame:
    df = df.copy()
    if "sample_index" not in df.columns:
        df["sample_index"] = np.arange(len(df), dtype=int)

    for c in ["true_label", "pred_label"]:
        if c in df.columns:
            df[c] = df[c].map(strip_label)

    if "true_id" not in df.columns and "true_label" in df.columns:
        df["true_id"] = df["true_label"].map(label_to_id)
    if "pred_id" not in df.columns and "pred_label" in df.columns:
        df["pred_id"] = df["pred_label"].map(label_to_id)

    if "pred_label" not in df.columns and "pred_id" in df.columns:
        df["pred_label"] = df["pred_id"].astype(int).map(id_to_label)
    if "true_label" not in df.columns and "true_id" in df.columns:
        df["true_label"] = df["true_id"].astype(int).map(id_to_label)

    if "correct" not in df.columns:
        df["correct"] = df["true_id"].astype(int) == df["pred_id"].astype(int)

    needed = ["sample_index", "true_id", "true_label", "pred_id", "pred_label", "correct"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"prediction file missing columns {missing}; available={list(df.columns)}")

    df["sample_index"] = df["sample_index"].astype(int)
    df["true_id"] = df["true_id"].astype(int)
    df["pred_id"] = df["pred_id"].astype(int)
    df["true_label"] = df["true_label"].map(strip_label)
    df["pred_label"] = df["pred_label"].map(strip_label)
    df["correct"] = df["correct"].astype(bool)
    return df


def find_prob_columns(df: pd.DataFrame, label_names: List[str]) -> Optional[List[str]]:
    """
    Try common probability column formats:
      prob_Benign, prob_Ransomware, ...
      prob_0, prob_1, ...
      p0, p1, ...
      logit/prob names are not assumed unless obvious.
    """
    cols = list(df.columns)
    candidates_by_label = []
    for i, lab in enumerate(label_names):
        lab_clean = strip_label(lab)
        possible = [
            f"prob_{lab_clean}",
            f"prob_{lab_clean.strip()}",
            f"p_{lab_clean}",
            f"proba_{lab_clean}",
            f"prob_{i}",
            f"p_{i}",
            f"proba_{i}",
        ]
        found = None
        for p in possible:
            if p in cols:
                found = p
                break
        if found is None:
            candidates_by_label = []
            break
        candidates_by_label.append(found)
    if candidates_by_label:
        return candidates_by_label

    # Fallback: all columns starting with prob_, if count matches labels.
    prob_cols = [c for c in cols if str(c).startswith("prob_")]
    if len(prob_cols) == len(label_names):
        # Prefer numeric order if prob_0 style.
        def key(c):
            tail = str(c).replace("prob_", "")
            return int(tail) if tail.isdigit() else str(c)
        return sorted(prob_cols, key=key)

    return None


def add_top2_from_probs_or_columns(df: pd.DataFrame, label_names: List[str], label_to_id: Dict[str, int], id_to_label: Dict[int, str]) -> pd.DataFrame:
    df = df.copy()

    # Existing top columns.
    top2_label_candidates = ["top2_label", "second_label", "top_2_label"]
    top2_id_candidates = ["top2_id", "second_id", "top_2_id"]
    existing_top2_label = next((c for c in top2_label_candidates if c in df.columns), None)
    existing_top2_id = next((c for c in top2_id_candidates if c in df.columns), None)

    if existing_top2_label is not None:
        df["top1_label"] = df["pred_label"].map(strip_label)
        df["top1_id"] = df["pred_id"].astype(int)
        df["top2_label"] = df[existing_top2_label].map(strip_label)
        df["top2_id"] = df["top2_label"].map(label_to_id).astype(int)
        return df

    if existing_top2_id is not None:
        df["top1_label"] = df["pred_label"].map(strip_label)
        df["top1_id"] = df["pred_id"].astype(int)
        df["top2_id"] = df[existing_top2_id].astype(int)
        df["top2_label"] = df["top2_id"].map(id_to_label)
        return df

    prob_cols = find_prob_columns(df, label_names)
    if prob_cols is None:
        raise ValueError(
            "Cannot infer top2. Provide --baseline-probs path to B0 val_predictions_with_probs.csv "
            "with prob_* columns or top2_label/top2_id columns."
        )

    prob_mat = df[prob_cols].to_numpy(dtype=float)
    order = np.argsort(-prob_mat, axis=1)
    top1 = order[:, 0]
    top2 = order[:, 1]
    df["top1_id"] = top1.astype(int)
    df["top2_id"] = top2.astype(int)
    df["top1_label"] = [id_to_label[int(i)] for i in top1]
    df["top2_label"] = [id_to_label[int(i)] for i in top2]
    df["top1_prob"] = prob_mat[np.arange(len(df)), top1]
    df["top2_prob"] = prob_mat[np.arange(len(df)), top2]
    return df


def load_baseline_with_top2(args, repo_root: Path, label_names: List[str], label_to_id: Dict[str, int], id_to_label: Dict[int, str], y_val: np.ndarray) -> pd.DataFrame:
    """
    Use B0 probabilities for top2 if available; official baseline pred for correctness if needed.
    """
    probs_path = resolve_path(args.baseline_probs, repo_root)
    pred_path = resolve_path(args.baseline_pred, repo_root)

    if probs_path.exists():
        df = pd.read_csv(probs_path)
    elif pred_path.exists():
        df = pd.read_csv(pred_path)
    else:
        raise FileNotFoundError(f"Neither baseline_probs nor baseline_pred exists: {probs_path}, {pred_path}")

    df = normalize_pred_df(df, label_to_id, id_to_label)
    df = add_top2_from_probs_or_columns(df, label_names, label_to_id, id_to_label)

    df = df.sort_values("sample_index").reset_index(drop=True)
    if len(df) != len(y_val):
        raise ValueError(f"baseline rows {len(df)} != y_val rows {len(y_val)}")
    if not (df["true_id"].to_numpy(dtype=int) == y_val.astype(int)).all():
        print("[WARN] baseline true_id != y_val after sorting. Continuing with baseline file true labels.")

    return df


def make_binary_model(model_name: str, args):
    if model_name == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=int(args.logreg_max_iter),
                class_weight="balanced",
                solver="lbfgs",
                random_state=int(args.seed),
            ),
        )
    if model_name == "extratrees":
        return ExtraTreesClassifier(
            n_estimators=int(args.n_estimators),
            max_depth=None if int(args.max_depth) <= 0 else int(args.max_depth),
            min_samples_leaf=int(args.min_samples_leaf),
            class_weight="balanced",
            random_state=int(args.seed),
            n_jobs=int(args.n_jobs),
        )
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=int(args.n_estimators),
            max_depth=None if int(args.max_depth) <= 0 else int(args.max_depth),
            min_samples_leaf=int(args.min_samples_leaf),
            class_weight="balanced",
            random_state=int(args.seed),
            n_jobs=int(args.n_jobs),
        )
    raise ValueError(f"Unknown model_name={model_name}")


def get_proba_positive(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if p.shape[1] == 2:
            return p[:, 1].astype(float)
    if hasattr(model, "decision_function"):
        z = model.decision_function(X)
        return (1.0 / (1.0 + np.exp(-z))).astype(float)
    pred = model.predict(X)
    return pred.astype(float)


def binary_metrics(y_true_bin: np.ndarray, p_pos: np.ndarray) -> dict:
    y_pred = (p_pos >= 0.5).astype(int)
    acc = accuracy_score(y_true_bin, y_pred)
    macro_f1 = f1_score(y_true_bin, y_pred, average="macro")
    weighted_f1 = f1_score(y_true_bin, y_pred, average="weighted")
    try:
        auc = roc_auc_score(y_true_bin, p_pos)
    except Exception:
        auc = float("nan")
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true_bin, y_pred, labels=[0, 1], zero_division=0
    )
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "auc": float(auc),
        "precision_0": float(prec[0]),
        "recall_0": float(rec[0]),
        "f1_0": float(f1[0]),
        "support_0": int(sup[0]),
        "precision_1": float(prec[1]),
        "recall_1": float(rec[1]),
        "f1_1": float(f1[1]),
        "support_1": int(sup[1]),
    }


def train_all_experts(inp: dict, args) -> Tuple[Dict[Tuple[str, str, str], object], pd.DataFrame, Dict[Tuple[str, str, str], dict]]:
    """
    Returns:
      experts[(rep, model, pair_key)] = fitted model
      metrics_df
      pair_meta[(rep, model, pair_key)] = {label_a, label_b, id_a, id_b}
    """
    y_train = inp["y_train"]
    y_val = inp["y_val"]
    label_to_id = inp["label_to_id"]
    id_to_label = inp["id_to_label"]
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]

    experts = {}
    pair_meta = {}
    rows = []

    for rep_name in args.reps.split(","):
        rep_name = rep_name.strip()
        if not rep_name:
            continue
        if rep_name not in inp["reps_train"]:
            raise ValueError(f"Unknown rep={rep_name}; available={list(inp['reps_train'])}")

        Xtr_all = inp["reps_train"][rep_name]
        Xva_all = inp["reps_val"][rep_name]

        for model_name in model_names:
            for a, b in HARD_PAIRS:
                id_a, id_b = label_to_id[a], label_to_id[b]
                pair_key = PAIR_KEY[(a, b)]

                tr_mask = (y_train == id_a) | (y_train == id_b)
                va_mask = (y_val == id_a) | (y_val == id_b)

                Xtr = Xtr_all[tr_mask]
                ytr = (y_train[tr_mask] == id_b).astype(int)
                Xva = Xva_all[va_mask]
                yva = (y_val[va_mask] == id_b).astype(int)

                model = make_binary_model(model_name, args)
                model.fit(Xtr, ytr)

                pva = get_proba_positive(model, Xva)
                met = binary_metrics(yva, pva)

                key = (rep_name, model_name, pair_key)
                experts[key] = model
                pair_meta[key] = {
                    "pair_key": pair_key,
                    "label_a": a,
                    "label_b": b,
                    "id_a": int(id_a),
                    "id_b": int(id_b),
                    "positive_label": b,
                    "negative_label": a,
                }

                rows.append({
                    "rep": rep_name,
                    "model": model_name,
                    "pair_key": pair_key,
                    "pair": f"{a}<->{b}",
                    "label_0": a,
                    "label_1": b,
                    "n_train": int(len(ytr)),
                    "n_val": int(len(yva)),
                    **met,
                })

                print(f"[E1a0] trained {rep_name}/{model_name}/{pair_key}: macro_f1={met['macro_f1']:.4f} auc={met['auc']:.4f}")

    return experts, pd.DataFrame(rows), pair_meta


def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray, label_names: List[str]) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro")
    weighted = f1_score(y_true, y_pred, average="weighted")
    labels = list(range(len(label_names)))
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_rows = []
    for i, name in enumerate(label_names):
        per_rows.append({
            "class_id": int(i),
            "label": name,
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(sup[i]),
        })
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
    return (
        {"accuracy": float(acc), "macro_f1": float(macro), "weighted_f1": float(weighted)},
        pd.DataFrame(per_rows),
        cm_df,
    )


def baseline_top2_hard_pair_mask(base_df: pd.DataFrame) -> np.ndarray:
    top1 = base_df["top1_label"].map(strip_label).to_numpy()
    top2 = base_df["top2_label"].map(strip_label).to_numpy()
    out = []
    hard_sets = {frozenset(p) for p in HARD_PAIRS}
    for a, b in zip(top1, top2):
        out.append((a in MALWARE_LABELS) and (b in MALWARE_LABELS) and (frozenset([a, b]) in hard_sets))
    return np.asarray(out, dtype=bool)


def find_pair_for_labels(label1: str, label2: str) -> Optional[Tuple[str, str, str]]:
    s = frozenset([strip_label(label1), strip_label(label2)])
    for a, b in HARD_PAIRS:
        if s == frozenset([a, b]):
            return a, b, PAIR_KEY[(a, b)]
    return None


def apply_expert_policy(
    *,
    rep_name: str,
    model_name: str,
    threshold: float,
    experts: Dict[Tuple[str, str, str], object],
    pair_meta: Dict[Tuple[str, str, str], dict],
    X_val_rep: np.ndarray,
    base_df: pd.DataFrame,
    label_to_id: Dict[str, int],
    id_to_label: Dict[int, str],
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Vectorized top-2 expert intervention.

    v1 was slow because it called predict_proba() one sample at a time.
    v2 groups validation rows by pair and calls predict_proba() once per pair/policy.
    """
    base_pred = base_df["pred_id"].to_numpy(dtype=int).copy()
    new_pred = base_pred.copy()

    hard_mask = baseline_top2_hard_pair_mask(base_df)
    hard_indices_all = np.where(hard_mask)[0]
    print(
        f"[E1a0] policy {rep_name}/{model_name}/thr={threshold:g}: "
        f"hard_top2_candidates={len(hard_indices_all)}",
        flush=True,
    )

    applied_rows = []

    if len(hard_indices_all) == 0:
        return new_pred, pd.DataFrame(applied_rows)

    top1_arr = base_df["top1_label"].map(strip_label).to_numpy()
    top2_arr = base_df["top2_label"].map(strip_label).to_numpy()

    for a, b in HARD_PAIRS:
        pair_key = PAIR_KEY[(a, b)]
        key = (rep_name, model_name, pair_key)
        if key not in experts:
            continue

        # Select rows whose baseline top1/top2 are exactly this hard pair.
        pair_set = frozenset([a, b])
        idx = []
        for i in hard_indices_all:
            if frozenset([top1_arr[i], top2_arr[i]]) == pair_set:
                idx.append(int(i))
        if not idx:
            continue

        idx = np.asarray(idx, dtype=int)
        model = experts[key]
        meta = pair_meta[key]

        p_b_all = get_proba_positive(model, X_val_rep[idx])
        expert_is_b = p_b_all >= 0.5
        conf_all = np.maximum(p_b_all, 1.0 - p_b_all)
        apply_mask = conf_all >= float(threshold)

        if not np.any(apply_mask):
            continue

        idx_apply = idx[apply_mask]
        p_apply = p_b_all[apply_mask]
        conf_apply = conf_all[apply_mask]
        is_b_apply = expert_is_b[apply_mask]

        labels_apply = np.where(is_b_apply, meta["label_b"], meta["label_a"])
        new_ids_apply = np.array([label_to_id[str(x)] for x in labels_apply], dtype=int)

        old_ids = new_pred[idx_apply].copy()
        new_pred[idx_apply] = new_ids_apply

        for local_k, i in enumerate(idx_apply):
            old_id = int(old_ids[local_k])
            new_id = int(new_ids_apply[local_k])
            applied_rows.append({
                "sample_index": int(base_df.at[int(i), "sample_index"]),
                "row_index": int(i),
                "true_label": strip_label(base_df.at[int(i), "true_label"]),
                "base_pred_label": strip_label(base_df.at[int(i), "pred_label"]),
                "top1_label": strip_label(base_df.at[int(i), "top1_label"]),
                "top2_label": strip_label(base_df.at[int(i), "top2_label"]),
                "pair_key": pair_key,
                "pair": f"{a}<->{b}",
                "rep": rep_name,
                "model": model_name,
                "threshold": float(threshold),
                "expert_p_label_b": float(p_apply[local_k]),
                "expert_conf": float(conf_apply[local_k]),
                "expert_label": str(labels_apply[local_k]),
                "old_pred_id": old_id,
                "new_pred_id": new_id,
                "old_pred_label": id_to_label[old_id],
                "new_pred_label": id_to_label[new_id],
                "changed": bool(old_id != new_id),
            })

    return new_pred, pd.DataFrame(applied_rows)


def transition_stats(y_true: np.ndarray, base_pred: np.ndarray, new_pred: np.ndarray) -> dict:
    base_correct = base_pred == y_true
    new_correct = new_pred == y_true
    fixed = (~base_correct) & new_correct
    damaged = base_correct & (~new_correct)
    changed = base_pred != new_pred
    wrong_to_wrong_changed = (~base_correct) & (~new_correct) & changed
    return {
        "wrong_to_correct": int(fixed.sum()),
        "correct_to_wrong": int(damaged.sum()),
        "net_gain": int(fixed.sum() - damaged.sum()),
        "damage_ratio": float(damaged.sum() / fixed.sum()) if int(fixed.sum()) > 0 else None,
        "changed_pred_n": int(changed.sum()),
        "wrong_to_wrong_changed": int(wrong_to_wrong_changed.sum()),
        "baseline_correct": int(base_correct.sum()),
        "new_correct": int(new_correct.sum()),
    }


def pair_fix_damage(
    y_true: np.ndarray,
    base_pred: np.ndarray,
    new_pred: np.ndarray,
    label_to_id: Dict[str, int],
    id_to_label: Dict[int, str],
) -> pd.DataFrame:
    rows = []
    base_correct = base_pred == y_true
    new_correct = new_pred == y_true

    for a, b in HARD_PAIRS:
        ida, idb = label_to_id[a], label_to_id[b]
        pair_mask = (y_true == ida) | (y_true == idb)
        fixed = pair_mask & (~base_correct) & new_correct
        damaged = pair_mask & base_correct & (~new_correct)

        rows.append({
            "scope": "pair_true_labels",
            "pair": f"{a}<->{b}",
            "direction": "BIDIR",
            "n_true_pair": int(pair_mask.sum()),
            "fixed": int(fixed.sum()),
            "damaged": int(damaged.sum()),
            "net": int(fixed.sum() - damaged.sum()),
            "damage_ratio": float(damaged.sum()/fixed.sum()) if int(fixed.sum()) else None,
        })

        for true_label, other_label in [(a, b), (b, a)]:
            tid = label_to_id[true_label]
            oid = label_to_id[other_label]
            dir_mask = y_true == tid

            base_conf = dir_mask & (base_pred == oid)
            new_conf = dir_mask & (new_pred == oid)

            fixed_dir = base_conf & (new_pred == tid)
            damaged_dir = dir_mask & (base_pred == tid) & (new_pred == oid)

            rows.append({
                "scope": "hard_direction",
                "pair": f"{a}<->{b}",
                "direction": f"{true_label}->{other_label}",
                "n_true_pair": int(dir_mask.sum()),
                "baseline_confusion_count": int(base_conf.sum()),
                "new_confusion_count": int(new_conf.sum()),
                "confusion_delta_new_minus_base": int(new_conf.sum() - base_conf.sum()),
                "fixed": int(fixed_dir.sum()),
                "damaged": int(damaged_dir.sum()),
                "net": int(fixed_dir.sum() - damaged_dir.sum()),
                "damage_ratio": float(damaged_dir.sum()/fixed_dir.sum()) if int(fixed_dir.sum()) else None,
            })

    return pd.DataFrame(rows)


def evaluate_policies(inp: dict, base_df: pd.DataFrame, experts, pair_meta, args) -> dict:
    y_true = inp["y_val"].astype(int)
    label_names = inp["label_names"]
    label_to_id = inp["label_to_id"]
    id_to_label = inp["id_to_label"]

    base_pred = base_df["pred_id"].to_numpy(dtype=int)
    base_metrics, base_per_class, base_cm = multiclass_metrics(y_true, base_pred, label_names)
    thresholds = [float(x) for x in args.thresholds.split(",") if str(x).strip()]

    policy_rows = []
    per_class_rows = []
    all_applied = {}
    all_new_preds = {}

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    rep_names = [r.strip() for r in args.reps.split(",") if r.strip()]

    for rep_name in rep_names:
        X_val_rep = inp["reps_val"][rep_name]
        for model_name in model_names:
            for thr in thresholds:
                new_pred, applied_df = apply_expert_policy(
                    rep_name=rep_name,
                    model_name=model_name,
                    threshold=thr,
                    experts=experts,
                    pair_meta=pair_meta,
                    X_val_rep=X_val_rep,
                    base_df=base_df,
                    label_to_id=label_to_id,
                    id_to_label=id_to_label,
                )
                met, per_cls, cm_df = multiclass_metrics(y_true, new_pred, label_names)
                trans = transition_stats(y_true, base_pred, new_pred)

                policy_name = f"{rep_name}__{model_name}__top2_hardpair_thr_{thr:g}"
                all_applied[policy_name] = applied_df
                all_new_preds[policy_name] = new_pred

                policy_rows.append({
                    "policy": policy_name,
                    "rep": rep_name,
                    "model": model_name,
                    "threshold": float(thr),
                    "applied_n": int(len(applied_df)),
                    "applied_changed_n": int(applied_df["changed"].sum()) if len(applied_df) else 0,
                    "accuracy": met["accuracy"],
                    "macro_f1": met["macro_f1"],
                    "weighted_f1": met["weighted_f1"],
                    "delta_accuracy": met["accuracy"] - base_metrics["accuracy"],
                    "delta_macro_f1": met["macro_f1"] - base_metrics["macro_f1"],
                    "delta_weighted_f1": met["weighted_f1"] - base_metrics["weighted_f1"],
                    **trans,
                })

                per_cls = per_cls.copy()
                per_cls["policy"] = policy_name
                per_cls["rep"] = rep_name
                per_cls["model"] = model_name
                per_cls["threshold"] = float(thr)
                per_class_rows.append(per_cls)

    policy_df = pd.DataFrame(policy_rows).sort_values(
        ["macro_f1", "net_gain", "damage_ratio"], ascending=[False, False, True]
    ).reset_index(drop=True)
    per_class_df = pd.concat(per_class_rows, ignore_index=True) if per_class_rows else pd.DataFrame()

    # Baseline row for easier comparison.
    base_row = {
        "policy": "BASELINE",
        "rep": "",
        "model": "",
        "threshold": np.nan,
        "applied_n": 0,
        "applied_changed_n": 0,
        "accuracy": base_metrics["accuracy"],
        "macro_f1": base_metrics["macro_f1"],
        "weighted_f1": base_metrics["weighted_f1"],
        "delta_accuracy": 0.0,
        "delta_macro_f1": 0.0,
        "delta_weighted_f1": 0.0,
        **transition_stats(y_true, base_pred, base_pred),
    }
    policy_df_with_base = pd.concat([pd.DataFrame([base_row]), policy_df], ignore_index=True)

    best_policy = policy_df.iloc[0]["policy"] if len(policy_df) else "BASELINE"
    best_pred = all_new_preds[best_policy] if best_policy in all_new_preds else base_pred
    best_applied = all_applied.get(best_policy, pd.DataFrame())

    best_metrics, best_per_class, best_cm = multiclass_metrics(y_true, best_pred, label_names)
    best_pair_fd = pair_fix_damage(y_true, base_pred, best_pred, label_to_id, id_to_label)

    # Build best predictions CSV.
    best_pred_df = base_df[["sample_index", "true_id", "true_label"]].copy()
    best_pred_df["base_pred_id"] = base_pred
    best_pred_df["base_pred_label"] = [id_to_label[int(i)] for i in base_pred]
    best_pred_df["e1a0_pred_id"] = best_pred
    best_pred_df["e1a0_pred_label"] = [id_to_label[int(i)] for i in best_pred]
    best_pred_df["base_correct"] = base_pred == y_true
    best_pred_df["e1a0_correct"] = best_pred == y_true
    best_pred_df["transition"] = "both_wrong"
    best_pred_df.loc[best_pred_df["base_correct"] & best_pred_df["e1a0_correct"], "transition"] = "both_correct"
    best_pred_df.loc[(~best_pred_df["base_correct"]) & best_pred_df["e1a0_correct"], "transition"] = "fixed"
    best_pred_df.loc[best_pred_df["base_correct"] & (~best_pred_df["e1a0_correct"]), "transition"] = "damaged"

    return {
        "baseline_metrics": base_metrics,
        "baseline_per_class": base_per_class,
        "baseline_cm": base_cm,
        "policy_df": policy_df_with_base,
        "per_class_df": per_class_df,
        "best_policy": best_policy,
        "best_metrics": best_metrics,
        "best_per_class": best_per_class,
        "best_cm": best_cm,
        "best_pair_fix_damage": best_pair_fd,
        "best_applied": best_applied,
        "best_pred_df": best_pred_df,
    }


def zip_dir(src_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(src_dir.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(src_dir.parent))


def write_summary_md(out_dir: Path, summary: dict) -> None:
    text = f"""# E1a0 Full-Feature Binary Expert Diagnostic

## Purpose

Train 3 standalone binary experts on full features:
- Ransomware vs Spyware
- Ransomware vs Trojan
- Spyware vs Trojan

Then intervene only when baseline top1/top2 are both malware and form a hard pair.

This tests whether a pure 2-class expert using the same full input can improve the final 4-class decision.

## Best policy

```text
{summary.get('best_policy')}
```

## Baseline

```text
accuracy  = {summary['baseline_metrics']['accuracy']:.6f}
macro_f1  = {summary['baseline_metrics']['macro_f1']:.6f}
weighted  = {summary['baseline_metrics']['weighted_f1']:.6f}
```

## Best E1a0 policy

```text
accuracy  = {summary['best_metrics']['accuracy']:.6f}
macro_f1  = {summary['best_metrics']['macro_f1']:.6f}
weighted  = {summary['best_metrics']['weighted_f1']:.6f}
```

## Best transition

```text
wrong_to_correct = {summary['best_transition']['wrong_to_correct']}
correct_to_wrong = {summary['best_transition']['correct_to_wrong']}
net_gain         = {summary['best_transition']['net_gain']}
damage_ratio     = {summary['best_transition']['damage_ratio']}
changed_pred_n   = {summary['best_transition']['changed_pred_n']}
```

## Output files

- `E1a0_summary.json`
- `E1a0_binary_expert_metrics.csv`
- `E1a0_policy_metrics.csv`
- `E1a0_policy_per_class_f1.csv`
- `E1a0_best_policy_predictions.csv`
- `E1a0_best_policy_confusion_matrix.csv`
- `E1a0_best_policy_pair_fix_damage.csv`
- `E1a0_best_policy_applied_samples.csv`

## Interpretation

If binary expert metrics are high but policy net_gain is small/damage_ratio high,
then the binary expert is not safe enough for final top-2 intervention.
"""
    (out_dir / "E1a0_summary.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="E1a0 full-feature binary expert diagnostic")
    parser.add_argument("--dataset-npz", default="03_outputs/05_dataset/dataset.npz")
    parser.add_argument("--metadata-json", default="03_outputs/05_dataset/metadata.json")
    parser.add_argument("--train-raw", default="01_split/train_raw.csv")
    parser.add_argument("--val-raw", default="01_split/val_raw.csv")
    parser.add_argument("--official-train", default="02_src/07_train.py")
    parser.add_argument("--baseline-pred", default="03_outputs/06_model/val_predictions_best.csv")
    parser.add_argument("--baseline-probs", default="05_test/outputs/B0_wrong_top2_audit/val_predictions_with_probs.csv")
    parser.add_argument("--out-dir", default="05_test/outputs/E1a0_full_feature_binary_expert")
    parser.add_argument("--reps", default="raw_scaled,bin_norm,offset,d3_scalar,concat_all")
    parser.add_argument("--models", default="logreg,extratrees")
    parser.add_argument("--thresholds", default="0.5,0.55,0.6,0.65,0.7,0.75")
    parser.add_argument("--include-concat", action="store_true", default=True)
    parser.add_argument("--no-concat", dest="include_concat", action="store_false")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=0, help="<=0 means no max_depth for tree models")
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--logreg-max-iter", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--make-zip", action="store_true", default=True)
    parser.add_argument("--no-zip", dest="make_zip", action="store_false")
    args = parser.parse_args()

    repo_root = repo_root_from_here()
    out_dir = resolve_path(args.out_dir, repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[E1a0] repo_root={repo_root}")
    print(f"[E1a0] out_dir={out_dir}")

    inp = load_official_inputs(args, repo_root)
    base_df = load_baseline_with_top2(
        args=args,
        repo_root=repo_root,
        label_names=inp["label_names"],
        label_to_id=inp["label_to_id"],
        id_to_label=inp["id_to_label"],
        y_val=inp["y_val"],
    )

    # Save baseline/top2 context for debugging.
    base_df.to_csv(out_dir / "E1a0_baseline_top2_context.csv", index=False)

    print("[E1a0] training standalone full-feature binary experts...")
    experts, expert_metrics_df, pair_meta = train_all_experts(inp, args)
    expert_metrics_df.to_csv(out_dir / "E1a0_binary_expert_metrics.csv", index=False)

    print("[E1a0] evaluating top-2 intervention policies...")
    evals = evaluate_policies(inp, base_df, experts, pair_meta, args)
    evals["policy_df"].to_csv(out_dir / "E1a0_policy_metrics.csv", index=False)
    evals["per_class_df"].to_csv(out_dir / "E1a0_policy_per_class_f1.csv", index=False)
    evals["best_pred_df"].to_csv(out_dir / "E1a0_best_policy_predictions.csv", index=False)
    evals["best_cm"].to_csv(out_dir / "E1a0_best_policy_confusion_matrix.csv")
    evals["best_pair_fix_damage"].to_csv(out_dir / "E1a0_best_policy_pair_fix_damage.csv", index=False)
    evals["best_applied"].to_csv(out_dir / "E1a0_best_policy_applied_samples.csv", index=False)
    evals["baseline_per_class"].to_csv(out_dir / "E1a0_baseline_per_class_f1.csv", index=False)
    evals["baseline_cm"].to_csv(out_dir / "E1a0_baseline_confusion_matrix.csv")

    best_transition = transition_stats(
        inp["y_val"].astype(int),
        base_df["pred_id"].to_numpy(dtype=int),
        evals["best_pred_df"]["e1a0_pred_id"].to_numpy(dtype=int),
    )

    summary = {
        "stage": "E1a0_full_feature_binary_expert_diagnostic",
        "purpose": "Train standalone binary experts with full features and test top-2 hard malware intervention.",
        "inputs": {
            "dataset_npz": str(resolve_path(args.dataset_npz, repo_root)),
            "metadata_json": str(resolve_path(args.metadata_json, repo_root)),
            "train_raw": str(resolve_path(args.train_raw, repo_root)),
            "val_raw": str(resolve_path(args.val_raw, repo_root)),
            "official_train": str(resolve_path(args.official_train, repo_root)),
            "baseline_pred": str(resolve_path(args.baseline_pred, repo_root)),
            "baseline_probs": str(resolve_path(args.baseline_probs, repo_root)),
        },
        "label_names": inp["label_names"],
        "num_bins": int(inp["num_bins"]),
        "n_train": int(len(inp["y_train"])),
        "n_val": int(len(inp["y_val"])),
        "representations": [r.strip() for r in args.reps.split(",") if r.strip()],
        "models": [m.strip() for m in args.models.split(",") if m.strip()],
        "thresholds": [float(x) for x in args.thresholds.split(",") if str(x).strip()],
        "baseline_metrics": evals["baseline_metrics"],
        "best_policy": evals["best_policy"],
        "best_metrics": evals["best_metrics"],
        "best_transition": best_transition,
        "best_policy_row": evals["policy_df"][evals["policy_df"]["policy"] == evals["best_policy"]].iloc[0].to_dict()
            if evals["best_policy"] in set(evals["policy_df"]["policy"]) else {},
        "best_binary_expert_rows_by_pair": (
            expert_metrics_df.sort_values(["pair_key", "macro_f1", "auc"], ascending=[True, False, False])
            .groupby("pair_key").head(3).to_dict(orient="records")
        ),
        "outputs": {
            "baseline_top2_context": str(out_dir / "E1a0_baseline_top2_context.csv"),
            "binary_expert_metrics": str(out_dir / "E1a0_binary_expert_metrics.csv"),
            "policy_metrics": str(out_dir / "E1a0_policy_metrics.csv"),
            "policy_per_class_f1": str(out_dir / "E1a0_policy_per_class_f1.csv"),
            "best_policy_predictions": str(out_dir / "E1a0_best_policy_predictions.csv"),
            "best_policy_confusion_matrix": str(out_dir / "E1a0_best_policy_confusion_matrix.csv"),
            "best_policy_pair_fix_damage": str(out_dir / "E1a0_best_policy_pair_fix_damage.csv"),
            "best_policy_applied_samples": str(out_dir / "E1a0_best_policy_applied_samples.csv"),
            "summary_md": str(out_dir / "E1a0_summary.md"),
        },
        "guardrail": "This is a diagnostic. Policies are selected on validation metrics, so use results to decide research direction, not as final unbiased score.",
    }
    save_json(out_dir / "E1a0_summary.json", summary)
    write_summary_md(out_dir, summary)

    if args.make_zip:
        zip_path = out_dir.with_suffix(".zip")
        zip_dir(out_dir, zip_path)
        print(f"[E1a0] zipped outputs: {zip_path}")

    print("[E1a0] done.")
    print(f"[E1a0] best_policy={evals['best_policy']}")
    print(f"[E1a0] baseline_macro_f1={evals['baseline_metrics']['macro_f1']:.6f}")
    print(f"[E1a0] best_macro_f1={evals['best_metrics']['macro_f1']:.6f}")
    print(f"[E1a0] best_transition={best_transition}")


if __name__ == "__main__":
    main()
