#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E1b Stronger Tabular Pair Experts + Gating v2-fast-verbose.

Audit/diagnostic experiment. Does NOT modify official source/output.

Purpose
-------
E1a0 showed the current best direction:
  concat_all + ExtraTrees binary pair experts + top2 hardpair gating
  macro-F1 ≈ 0.8268 on batch512 baseline.

E1a1 showed D3 attention binary experts do not improve final decision.

E1b continues from E1a0:
  - same official input/preprocess source
  - full feature input, default concat_all
  - stronger tabular binary experts
  - verbose before/after logs for each candidate
  - xgboost is optional, not default, because it may look like a hang
  - pair-specific thresholds
  - optional baseline top1-top2 margin cap
  - optional mixed model selection per pair

Main question
-------------
Can stronger tabular experts + safer gating improve over E1a0 and move closer
to macro-F1 target 0.90?

Input source equivalence
------------------------
Same as E1a0:
  dataset.npz:
    X_train_bin / X_val_bin
    X_train_offset / X_val_offset
    y_train / y_val

  train_raw.csv / val_raw.csv:
    official 07_train.py rebuilds raw_scaled continuous

Representations:
  raw_scaled
  bin_norm
  offset
  d3_scalar
  concat_all = raw_scaled + bin_norm + offset + d3_scalar

v2 changes:
  - default models exclude xgboost: extratrees,extratrees_deep,histgb
  - print START before each candidate train
  - smaller default policy grid
  - optional --models ...,xgboost still supported

Default output:
  05_test/outputs/E1b_stronger_tabular_pair_experts/
  05_test/outputs/E1b_stronger_tabular_pair_experts.zip
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import sys
import time
import warnings
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
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
from sklearn.calibration import CalibratedClassifierCV


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


def import_official_train(train_script: Path):
    if not train_script.exists():
        raise FileNotFoundError(f"official 07_train.py not found: {train_script}")
    src_dir = train_script.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    spec = importlib.util.spec_from_file_location("official_07_train_for_e1b", str(train_script))
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


def load_official_inputs(args, repo_root: Path) -> dict:
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
    cols = list(df.columns)
    by_label = []
    for i, lab in enumerate(label_names):
        lab = strip_label(lab)
        possible = [
            f"prob_{lab}", f"p_{lab}", f"proba_{lab}",
            f"prob_{i}", f"p_{i}", f"proba_{i}",
        ]
        found = None
        for c in possible:
            if c in cols:
                found = c
                break
        if found is None:
            by_label = []
            break
        by_label.append(found)
    if by_label:
        return by_label

    prob_cols = [c for c in cols if str(c).startswith("prob_")]
    if len(prob_cols) == len(label_names):
        def key(c):
            tail = str(c).replace("prob_", "")
            return int(tail) if tail.isdigit() else str(c)
        return sorted(prob_cols, key=key)

    return None


def add_top2(df: pd.DataFrame, label_names: List[str], label_to_id: Dict[str, int], id_to_label: Dict[int, str]) -> pd.DataFrame:
    df = df.copy()
    if "top2_label" in df.columns:
        df["top1_id"] = df["pred_id"].astype(int)
        df["top1_label"] = df["pred_label"].map(strip_label)
        df["top2_label"] = df["top2_label"].map(strip_label)
        df["top2_id"] = df["top2_label"].map(label_to_id).astype(int)
    elif "top2_id" in df.columns:
        df["top1_id"] = df["pred_id"].astype(int)
        df["top1_label"] = df["pred_label"].map(strip_label)
        df["top2_id"] = df["top2_id"].astype(int)
        df["top2_label"] = df["top2_id"].map(id_to_label)
    else:
        prob_cols = find_prob_columns(df, label_names)
        if prob_cols is None:
            raise ValueError("Cannot infer top2; provide baseline probabilities with prob_* columns.")
        prob = df[prob_cols].to_numpy(dtype=float)
        order = np.argsort(-prob, axis=1)
        top1 = order[:, 0]
        top2 = order[:, 1]
        df["top1_id"] = top1.astype(int)
        df["top2_id"] = top2.astype(int)
        df["top1_label"] = [id_to_label[int(i)] for i in top1]
        df["top2_label"] = [id_to_label[int(i)] for i in top2]
        df["top1_score"] = prob[np.arange(len(df)), top1]
        df["top2_score"] = prob[np.arange(len(df)), top2]
        df["top12_margin"] = df["top1_score"] - df["top2_score"]

    if "top1_score" not in df.columns or "top2_score" not in df.columns:
        prob_cols = find_prob_columns(df, label_names)
        if prob_cols is not None:
            prob = df[prob_cols].to_numpy(dtype=float)
            df["top1_score"] = prob[np.arange(len(df)), df["top1_id"].to_numpy(dtype=int)]
            df["top2_score"] = prob[np.arange(len(df)), df["top2_id"].to_numpy(dtype=int)]
            df["top12_margin"] = df["top1_score"] - df["top2_score"]
        else:
            df["top1_score"] = np.nan
            df["top2_score"] = np.nan
            df["top12_margin"] = np.nan

    df["true_in_top2"] = (df["true_id"] == df["top1_id"]) | (df["true_id"] == df["top2_id"])
    return df


def load_baseline(args, repo_root: Path, inp: dict) -> pd.DataFrame:
    probs_path = resolve_path(args.baseline_probs, repo_root)
    pred_path = resolve_path(args.baseline_pred, repo_root)
    if probs_path.exists():
        path = probs_path
    elif pred_path.exists():
        path = pred_path
    else:
        raise FileNotFoundError(f"Missing baseline files: {probs_path}, {pred_path}")

    df = pd.read_csv(path)
    df = normalize_pred_df(df, inp["label_to_id"], inp["id_to_label"])
    df = add_top2(df, inp["label_names"], inp["label_to_id"], inp["id_to_label"])
    df = df.sort_values("sample_index").reset_index(drop=True)
    if len(df) != len(inp["y_val"]):
        raise ValueError(f"baseline rows {len(df)} != y_val rows {len(inp['y_val'])}")
    return df


def make_model(model_name: str, args):
    name = model_name.lower()

    if name == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=int(args.logreg_max_iter),
                class_weight="balanced",
                solver="lbfgs",
                random_state=int(args.seed),
            )
        )

    if name == "extratrees":
        return ExtraTreesClassifier(
            n_estimators=int(args.et_n_estimators),
            max_depth=None if int(args.et_max_depth) <= 0 else int(args.et_max_depth),
            min_samples_leaf=int(args.et_min_samples_leaf),
            max_features=args.et_max_features,
            class_weight="balanced",
            bootstrap=False,
            random_state=int(args.seed),
            n_jobs=int(args.n_jobs),
        )

    if name == "extratrees_deep":
        return ExtraTreesClassifier(
            n_estimators=int(args.et_deep_n_estimators),
            max_depth=None,
            min_samples_leaf=1,
            max_features=args.et_deep_max_features,
            class_weight="balanced",
            bootstrap=False,
            random_state=int(args.seed),
            n_jobs=int(args.n_jobs),
        )

    if name == "rf":
        return RandomForestClassifier(
            n_estimators=int(args.rf_n_estimators),
            max_depth=None if int(args.rf_max_depth) <= 0 else int(args.rf_max_depth),
            min_samples_leaf=int(args.rf_min_samples_leaf),
            max_features=args.rf_max_features,
            class_weight="balanced",
            random_state=int(args.seed),
            n_jobs=int(args.n_jobs),
        )

    if name == "histgb":
        return HistGradientBoostingClassifier(
            max_iter=int(args.hgb_max_iter),
            learning_rate=float(args.hgb_learning_rate),
            max_leaf_nodes=int(args.hgb_max_leaf_nodes),
            l2_regularization=float(args.hgb_l2),
            early_stopping=True,
            validation_fraction=0.15,
            random_state=int(args.seed),
        )

    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except Exception as e:
            raise RuntimeError("xgboost not installed") from e
        return XGBClassifier(
            n_estimators=int(args.xgb_n_estimators),
            max_depth=int(args.xgb_max_depth),
            learning_rate=float(args.xgb_learning_rate),
            subsample=float(args.xgb_subsample),
            colsample_bytree=float(args.xgb_colsample_bytree),
            reg_lambda=float(args.xgb_reg_lambda),
            reg_alpha=float(args.xgb_reg_alpha),
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method=args.xgb_tree_method,
            verbosity=int(args.xgb_verbosity),
            random_state=int(args.seed),
            n_jobs=int(args.n_jobs),
        )

    if name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except Exception as e:
            raise RuntimeError("lightgbm not installed") from e
        return LGBMClassifier(
            n_estimators=int(args.lgb_n_estimators),
            learning_rate=float(args.lgb_learning_rate),
            num_leaves=int(args.lgb_num_leaves),
            subsample=float(args.lgb_subsample),
            colsample_bytree=float(args.lgb_colsample_bytree),
            reg_lambda=float(args.lgb_reg_lambda),
            class_weight="balanced",
            random_state=int(args.seed),
            n_jobs=int(args.n_jobs),
            verbose=-1,
        )

    raise ValueError(f"Unknown model: {model_name}")


def maybe_calibrate(model, X: np.ndarray, y: np.ndarray, args):
    if args.calibration == "none":
        return model

    # Some optional libraries already calibrated enough; but this is diagnostic.
    # Ensemble=False keeps one calibrated wrapper over CV clones.
    method = args.calibration
    try:
        cal = CalibratedClassifierCV(
            estimator=model,
            method=method,
            cv=int(args.calibration_cv),
            ensemble=True,
        )
    except TypeError:
        # Older sklearn compatibility
        cal = CalibratedClassifierCV(
            base_estimator=model,
            method=method,
            cv=int(args.calibration_cv),
        )
    cal.fit(X, y)
    return cal


def get_p_pos(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if p.ndim == 2 and p.shape[1] >= 2:
            return p[:, 1].astype(float)
    if hasattr(model, "decision_function"):
        z = model.decision_function(X)
        return (1.0 / (1.0 + np.exp(-z))).astype(float)
    pred = model.predict(X)
    return pred.astype(float)


def binary_metrics(y_true: np.ndarray, p_pos: np.ndarray) -> dict:
    y_pred = (p_pos >= 0.5).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }
    try:
        out["auc"] = float(roc_auc_score(y_true, p_pos))
    except Exception:
        out["auc"] = float("nan")
    prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    out.update({
        "precision_0": float(prec[0]),
        "recall_0": float(rec[0]),
        "f1_0": float(f1[0]),
        "support_0": int(sup[0]),
        "precision_1": float(prec[1]),
        "recall_1": float(rec[1]),
        "f1_1": float(f1[1]),
        "support_1": int(sup[1]),
    })
    return out


def train_candidates(inp: dict, args, out_dir: Path) -> Tuple[pd.DataFrame, Dict[Tuple[str, str, str], dict]]:
    y_train = inp["y_train"]
    y_val = inp["y_val"]
    label_to_id = inp["label_to_id"]
    reps = [r.strip() for r in args.reps.split(",") if r.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    candidates: Dict[Tuple[str, str, str], dict] = {}
    rows = []

    for rep in reps:
        if rep not in inp["reps_train"]:
            raise ValueError(f"Unknown representation {rep}; available={list(inp['reps_train'])}")
        Xtr_all = inp["reps_train"][rep]
        Xva_all = inp["reps_val"][rep]

        for model_name in models:
            for a, b in HARD_PAIRS:
                pk = PAIR_KEY[(a, b)]
                ida, idb = label_to_id[a], label_to_id[b]
                tr_mask = (y_train == ida) | (y_train == idb)
                va_mask = (y_val == ida) | (y_val == idb)

                Xtr = Xtr_all[tr_mask]
                ytr = (y_train[tr_mask] == idb).astype(int)
                Xva_pair = Xva_all[va_mask]
                yva = (y_val[va_mask] == idb).astype(int)

                print(
                    f"[E1b] START {rep}/{model_name}/{pk} "
                    f"train_n={int(len(ytr))} val_pair_n={int(len(yva))}",
                    flush=True,
                )
                t0 = time.time()
                try:
                    base_model = make_model(model_name, args)
                    if args.calibration != "none":
                        model = maybe_calibrate(base_model, Xtr, ytr, args)
                    else:
                        model = base_model
                        model.fit(Xtr, ytr)
                    p_pair = get_p_pos(model, Xva_pair)
                    p_all = get_p_pos(model, Xva_all)
                    status = "ok"
                    err = ""
                except Exception as e:
                    warnings.warn(f"skip candidate {rep}/{model_name}/{pk}: {e}")
                    status = "failed"
                    err = repr(e)
                    model = None
                    p_pair = np.full(len(yva), np.nan)
                    p_all = np.full(len(Xva_all), np.nan)

                elapsed = time.time() - t0
                met = binary_metrics(yva, p_pair) if status == "ok" else {
                    "accuracy": np.nan, "macro_f1": np.nan, "weighted_f1": np.nan, "auc": np.nan,
                    "precision_0": np.nan, "recall_0": np.nan, "f1_0": np.nan, "support_0": int((yva == 0).sum()),
                    "precision_1": np.nan, "recall_1": np.nan, "f1_1": np.nan, "support_1": int((yva == 1).sum()),
                }

                key = (rep, model_name, pk)
                candidates[key] = {
                    "rep": rep,
                    "model": model_name,
                    "pair_key": pk,
                    "label_a": a,
                    "label_b": b,
                    "id_a": int(ida),
                    "id_b": int(idb),
                    "p_all": p_all.astype(float),
                    "status": status,
                    "error": err,
                    "metrics": met,
                }

                npy_dir = out_dir / "candidate_probs" / rep / model_name
                npy_dir.mkdir(parents=True, exist_ok=True)
                if status == "ok":
                    np.save(npy_dir / f"{pk}_all_val_prob_label_b.npy", p_all.astype(np.float32))

                row = {
                    "rep": rep,
                    "model": model_name,
                    "pair_key": pk,
                    "pair": f"{a}<->{b}",
                    "status": status,
                    "error": err,
                    "seconds": float(elapsed),
                    "n_train": int(len(ytr)),
                    "n_val_pair": int(len(yva)),
                    "calibration": args.calibration,
                    **met,
                }
                rows.append(row)
                print(
                    f"[E1b] {rep}/{model_name}/{pk} status={status} "
                    f"f1={row['macro_f1'] if np.isfinite(row['macro_f1']) else np.nan:.4f} "
                    f"auc={row['auc'] if np.isfinite(row['auc']) else np.nan:.4f} sec={elapsed:.1f}",
                    flush=True,
                )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out_dir / "E1b_binary_expert_metrics.csv", index=False)
    return metrics_df, candidates


def hard_pair_key_from_labels(a: str, b: str) -> Optional[str]:
    s = frozenset([strip_label(a), strip_label(b)])
    for pair, key in PAIR_KEY.items():
        if s == frozenset(pair):
            return key
    return None


def macro_metrics(y_true: np.ndarray, y_pred: np.ndarray, label_names: List[str]) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    labels = list(range(len(label_names)))
    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro")
    weighted = f1_score(y_true, y_pred, average="weighted")
    prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    per = pd.DataFrame([
        {"class_id": i, "label": label_names[i], "precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]), "support": int(sup[i])}
        for i in labels
    ])
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels), index=label_names, columns=label_names)
    return {"accuracy": float(acc), "macro_f1": float(macro), "weighted_f1": float(weighted)}, per, cm


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


def pair_fix_damage(y_true: np.ndarray, base_pred: np.ndarray, new_pred: np.ndarray, label_to_id: Dict[str, int]) -> pd.DataFrame:
    base_correct = base_pred == y_true
    new_correct = new_pred == y_true
    rows = []
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
                "n_true": int(dir_mask.sum()),
                "baseline_confusion_count": int(base_conf.sum()),
                "new_confusion_count": int(new_conf.sum()),
                "confusion_delta_new_minus_base": int(new_conf.sum() - base_conf.sum()),
                "fixed": int(fixed_dir.sum()),
                "damaged": int(damaged_dir.sum()),
                "net": int(fixed_dir.sum() - damaged_dir.sum()),
                "damage_ratio": float(damaged_dir.sum()/fixed_dir.sum()) if int(fixed_dir.sum()) else None,
            })
    return pd.DataFrame(rows)


def apply_policy(
    base: pd.DataFrame,
    candidates: Dict[Tuple[str, str, str], dict],
    selected: Dict[str, Tuple[str, str, str]],
    thresholds: Dict[str, float],
    margin_cap: float,
    label_to_id: Dict[str, int],
) -> Tuple[np.ndarray, pd.DataFrame]:
    base_pred = base["pred_id"].to_numpy(dtype=int)
    new_pred = base_pred.copy()
    rows = []

    top12_margin = base["top12_margin"].to_numpy(dtype=float) if "top12_margin" in base.columns else np.full(len(base), np.nan)

    for i, r in base.iterrows():
        top1 = strip_label(r["top1_label"])
        top2 = strip_label(r["top2_label"])
        pk = hard_pair_key_from_labels(top1, top2)
        if pk is None:
            continue
        if pk not in selected:
            continue
        if np.isfinite(top12_margin[i]) and margin_cap < 1e8 and top12_margin[i] > margin_cap:
            continue

        cand_key = selected[pk]
        cand = candidates[cand_key]
        if cand["status"] != "ok":
            continue

        p_b = float(cand["p_all"][i])
        conf = max(p_b, 1.0 - p_b)
        if conf < thresholds[pk]:
            continue

        label = cand["label_b"] if p_b >= 0.5 else cand["label_a"]
        new_id = int(label_to_id[label])
        old_id = int(new_pred[i])
        new_pred[i] = new_id
        rows.append({
            "sample_index": int(r["sample_index"]),
            "row_index": int(i),
            "true_label": strip_label(r["true_label"]),
            "base_pred_label": strip_label(r["pred_label"]),
            "top1_label": top1,
            "top2_label": top2,
            "pair_key": pk,
            "selected_rep": cand["rep"],
            "selected_model": cand["model"],
            "threshold": float(thresholds[pk]),
            "margin_cap": float(margin_cap),
            "expert_prob_label_b": p_b,
            "expert_conf": conf,
            "expert_label": label,
            "old_pred_id": old_id,
            "new_pred_id": new_id,
            "changed": bool(old_id != new_id),
        })

    return new_pred, pd.DataFrame(rows)


def evaluate_policy(
    *,
    policy_name: str,
    selected: Dict[str, Tuple[str, str, str]],
    thresholds: Dict[str, float],
    margin_cap: float,
    base: pd.DataFrame,
    candidates: Dict[Tuple[str, str, str], dict],
    inp: dict,
    base_metrics: dict,
) -> Tuple[dict, np.ndarray, pd.DataFrame]:
    y_true = inp["y_val"].astype(int)
    base_pred = base["pred_id"].to_numpy(dtype=int)
    new_pred, applied = apply_policy(
        base=base,
        candidates=candidates,
        selected=selected,
        thresholds=thresholds,
        margin_cap=margin_cap,
        label_to_id=inp["label_to_id"],
    )
    met, _, _ = macro_metrics(y_true, new_pred, inp["label_names"])
    trans = transition_stats(y_true, base_pred, new_pred)

    row = {
        "policy": policy_name,
        "RS_candidate": "__".join(selected["RS"]),
        "RT_candidate": "__".join(selected["RT"]),
        "ST_candidate": "__".join(selected["ST"]),
        "RS_thr": float(thresholds["RS"]),
        "RT_thr": float(thresholds["RT"]),
        "ST_thr": float(thresholds["ST"]),
        "margin_cap": float(margin_cap),
        "applied_n": int(len(applied)),
        "applied_changed_n": int(applied["changed"].sum()) if len(applied) else 0,
        **met,
        "delta_accuracy": met["accuracy"] - base_metrics["accuracy"],
        "delta_macro_f1": met["macro_f1"] - base_metrics["macro_f1"],
        "delta_weighted_f1": met["weighted_f1"] - base_metrics["weighted_f1"],
        **trans,
    }
    return row, new_pred, applied


def candidate_families(metrics_df: pd.DataFrame, candidates: Dict[Tuple[str, str, str], dict], args) -> Dict[str, Dict[str, Tuple[str, str, str]]]:
    families = {}

    # Same rep/model across all pairs.
    ok = metrics_df[metrics_df["status"] == "ok"].copy()
    for (rep, model), g in ok.groupby(["rep", "model"]):
        have = set(g["pair_key"])
        if {"RS", "RT", "ST"}.issubset(have):
            families[f"same__{rep}__{model}"] = {
                pk: (rep, model, pk) for pk in ["RS", "RT", "ST"]
            }

    # Mixed best per binary macro-F1.
    mixed = {}
    for pk in ["RS", "RT", "ST"]:
        g = ok[ok["pair_key"] == pk].sort_values(["macro_f1", "auc"], ascending=[False, False])
        if len(g):
            r = g.iloc[0]
            mixed[pk] = (str(r["rep"]), str(r["model"]), pk)
    if set(mixed) == {"RS", "RT", "ST"}:
        families["mixed_best_binary_macro_f1"] = mixed

    # Mixed best per AUC.
    mixed_auc = {}
    for pk in ["RS", "RT", "ST"]:
        g = ok[ok["pair_key"] == pk].sort_values(["auc", "macro_f1"], ascending=[False, False])
        if len(g):
            r = g.iloc[0]
            mixed_auc[pk] = (str(r["rep"]), str(r["model"]), pk)
    if set(mixed_auc) == {"RS", "RT", "ST"}:
        families["mixed_best_binary_auc"] = mixed_auc

    return families


def evaluate_families(inp: dict, base: pd.DataFrame, candidates: dict, metrics_df: pd.DataFrame, args, out_dir: Path) -> Tuple[pd.DataFrame, dict, np.ndarray, pd.DataFrame]:
    y_true = inp["y_val"].astype(int)
    base_pred = base["pred_id"].to_numpy(dtype=int)
    base_metrics, base_per, base_cm = macro_metrics(y_true, base_pred, inp["label_names"])
    base_per.to_csv(out_dir / "E1b_baseline_per_class_f1.csv", index=False)
    base_cm.to_csv(out_dir / "E1b_baseline_confusion_matrix.csv")

    thresholds = [float(x) for x in str(args.thresholds).split(",") if str(x).strip()]
    margin_caps = [float(x) for x in str(args.margin_caps).split(",") if str(x).strip()]
    families = candidate_families(metrics_df, candidates, args)

    rows = []
    pred_by_policy = {}
    applied_by_policy = {}

    for fam_name, selected in families.items():
        print(f"[E1b] EVAL family={fam_name}", flush=True)
        for margin_cap in margin_caps:
            print(f"[E1b] EVAL family={fam_name} margin_cap={margin_cap:g}", flush=True)
            # Global threshold.
            for thr in thresholds:
                th = {"RS": thr, "RT": thr, "ST": thr}
                policy = f"{fam_name}__global_thr_{thr:g}__margin_cap_{margin_cap:g}"
                row, pred, applied = evaluate_policy(
                    policy_name=policy,
                    selected=selected,
                    thresholds=th,
                    margin_cap=margin_cap,
                    base=base,
                    candidates=candidates,
                    inp=inp,
                    base_metrics=base_metrics,
                )
                rows.append(row)
                pred_by_policy[policy] = pred
                applied_by_policy[policy] = applied

            # Pair-specific thresholds.
            for rs, rt, st in itertools.product(thresholds, thresholds, thresholds):
                th = {"RS": rs, "RT": rt, "ST": st}
                policy = f"{fam_name}__pair_thr_RS{rs:g}_RT{rt:g}_ST{st:g}__margin_cap_{margin_cap:g}"
                row, pred, applied = evaluate_policy(
                    policy_name=policy,
                    selected=selected,
                    thresholds=th,
                    margin_cap=margin_cap,
                    base=base,
                    candidates=candidates,
                    inp=inp,
                    base_metrics=base_metrics,
                )
                rows.append(row)
                pred_by_policy[policy] = pred
                applied_by_policy[policy] = applied

    policy_df = pd.DataFrame(rows)
    if len(policy_df):
        policy_df = policy_df.sort_values(["macro_f1", "net_gain", "damage_ratio"], ascending=[False, False, True]).reset_index(drop=True)

    base_row = {
        "policy": "BASELINE",
        "RS_candidate": "",
        "RT_candidate": "",
        "ST_candidate": "",
        "RS_thr": np.nan,
        "RT_thr": np.nan,
        "ST_thr": np.nan,
        "margin_cap": np.nan,
        "applied_n": 0,
        "applied_changed_n": 0,
        **base_metrics,
        "delta_accuracy": 0.0,
        "delta_macro_f1": 0.0,
        "delta_weighted_f1": 0.0,
        **transition_stats(y_true, base_pred, base_pred),
    }
    policy_with_base = pd.concat([pd.DataFrame([base_row]), policy_df], ignore_index=True)
    policy_with_base.to_csv(out_dir / "E1b_policy_metrics.csv", index=False)

    if len(policy_df):
        best_policy = str(policy_df.iloc[0]["policy"])
        best_pred = pred_by_policy[best_policy]
        best_applied = applied_by_policy[best_policy]
    else:
        best_policy = "BASELINE"
        best_pred = base_pred.copy()
        best_applied = pd.DataFrame()

    return policy_with_base, {"baseline_metrics": base_metrics, "best_policy": best_policy}, best_pred, best_applied


def save_best_outputs(inp: dict, base: pd.DataFrame, best_pred: np.ndarray, best_applied: pd.DataFrame, out_dir: Path) -> dict:
    y_true = inp["y_val"].astype(int)
    base_pred = base["pred_id"].to_numpy(dtype=int)

    best_metrics, best_per, best_cm = macro_metrics(y_true, best_pred, inp["label_names"])
    best_trans = transition_stats(y_true, base_pred, best_pred)
    pair_fd = pair_fix_damage(y_true, base_pred, best_pred, inp["label_to_id"])

    pred_df = base[["sample_index", "true_id", "true_label", "pred_id", "pred_label", "correct", "top1_label", "top2_label", "top1_score", "top2_score", "top12_margin", "true_in_top2"]].copy()
    pred_df = pred_df.rename(columns={"pred_id": "base_pred_id", "pred_label": "base_pred_label", "correct": "base_correct"})
    pred_df["e1b_pred_id"] = best_pred.astype(int)
    pred_df["e1b_pred_label"] = [inp["id_to_label"][int(i)] for i in best_pred]
    pred_df["e1b_correct"] = best_pred == y_true
    pred_df["transition"] = "both_wrong"
    pred_df.loc[pred_df["base_correct"] & pred_df["e1b_correct"], "transition"] = "both_correct"
    pred_df.loc[(~pred_df["base_correct"]) & pred_df["e1b_correct"], "transition"] = "fixed"
    pred_df.loc[pred_df["base_correct"] & (~pred_df["e1b_correct"]), "transition"] = "damaged"

    pred_df.to_csv(out_dir / "E1b_best_policy_predictions.csv", index=False)
    best_applied.to_csv(out_dir / "E1b_best_policy_applied_samples.csv", index=False)
    best_per.to_csv(out_dir / "E1b_best_policy_per_class_f1.csv", index=False)
    best_cm.to_csv(out_dir / "E1b_best_policy_confusion_matrix.csv")
    pair_fd.to_csv(out_dir / "E1b_best_policy_pair_fix_damage.csv", index=False)

    return {
        "best_metrics": best_metrics,
        "best_transition": best_trans,
    }


def zip_dir(src_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(src_dir.rglob("*")):
            if p.is_file() and p != zip_path:
                z.write(p, p.relative_to(src_dir.parent))


def write_summary_md(out_dir: Path, summary: dict) -> None:
    text = f"""# E1b Stronger Tabular Pair Experts

## Goal

Continue from E1a0, current best macro-F1 ≈ 0.826835, toward target macro-F1 0.90.

## Design

- Full-feature tabular binary experts.
- Default representation: `concat_all`.
- Pair experts: RS, RT, ST.
- Stronger models tried: {', '.join(summary.get('models', []))}
- Calibration: {summary.get('calibration')}
- Pair-specific threshold and baseline margin-cap search.

## Baseline

```text
accuracy = {summary['baseline_metrics']['accuracy']:.6f}
macro_f1 = {summary['baseline_metrics']['macro_f1']:.6f}
weighted = {summary['baseline_metrics']['weighted_f1']:.6f}
```

## Best E1b

```text
policy   = {summary['best_policy']}
accuracy = {summary['best_metrics']['accuracy']:.6f}
macro_f1 = {summary['best_metrics']['macro_f1']:.6f}
weighted = {summary['best_metrics']['weighted_f1']:.6f}
```

## Transition

```text
wrong_to_correct = {summary['best_transition']['wrong_to_correct']}
correct_to_wrong = {summary['best_transition']['correct_to_wrong']}
net_gain         = {summary['best_transition']['net_gain']}
damage_ratio     = {summary['best_transition']['damage_ratio']}
changed_pred_n   = {summary['best_transition']['changed_pred_n']}
```

## Key files

- `E1b_binary_expert_metrics.csv`
- `E1b_policy_metrics.csv`
- `E1b_best_policy_predictions.csv`
- `E1b_best_policy_pair_fix_damage.csv`
- `E1b_best_policy_applied_samples.csv`
"""
    (out_dir / "E1b_summary.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="E1b stronger tabular pair experts + gating")
    parser.add_argument("--dataset-npz", default="03_outputs/05_dataset/dataset.npz")
    parser.add_argument("--metadata-json", default="03_outputs/05_dataset/metadata.json")
    parser.add_argument("--train-raw", default="01_split/train_raw.csv")
    parser.add_argument("--val-raw", default="01_split/val_raw.csv")
    parser.add_argument("--official-train", default="02_src/07_train.py")
    parser.add_argument("--baseline-pred", default="03_outputs/06_model/val_predictions_best.csv")
    parser.add_argument("--baseline-probs", default="05_test/outputs/B0_wrong_top2_audit/val_predictions_with_probs.csv")
    parser.add_argument("--out-dir", default="05_test/outputs/E1b_stronger_tabular_pair_experts")

    parser.add_argument("--reps", default="concat_all")
    parser.add_argument("--models", default="extratrees,extratrees_deep,histgb")
    parser.add_argument("--thresholds", default="0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95")
    parser.add_argument("--margin-caps", default="1000000000,0.4,0.25", help="1e9 means no cap")
    parser.add_argument("--calibration", default="none", choices=["none", "sigmoid", "isotonic"])
    parser.add_argument("--calibration-cv", type=int, default=3)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)

    # ExtraTrees default near E1a0.
    parser.add_argument("--et-n-estimators", type=int, default=500)
    parser.add_argument("--et-max-depth", type=int, default=0)
    parser.add_argument("--et-min-samples-leaf", type=int, default=2)
    parser.add_argument("--et-max-features", default="sqrt")

    # ExtraTrees stronger/deeper.
    parser.add_argument("--et-deep-n-estimators", type=int, default=900)
    parser.add_argument("--et-deep-max-features", default="sqrt")

    # RandomForest optional.
    parser.add_argument("--rf-n-estimators", type=int, default=500)
    parser.add_argument("--rf-max-depth", type=int, default=0)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=2)
    parser.add_argument("--rf-max-features", default="sqrt")

    # HistGradientBoosting.
    parser.add_argument("--hgb-max-iter", type=int, default=400)
    parser.add_argument("--hgb-learning-rate", type=float, default=0.04)
    parser.add_argument("--hgb-max-leaf-nodes", type=int, default=31)
    parser.add_argument("--hgb-l2", type=float, default=0.01)

    # XGBoost optional.
    parser.add_argument("--xgb-n-estimators", type=int, default=300)
    parser.add_argument("--xgb-max-depth", type=int, default=4)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.03)
    parser.add_argument("--xgb-subsample", type=float, default=0.9)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.9)
    parser.add_argument("--xgb-reg-lambda", type=float, default=2.0)
    parser.add_argument("--xgb-reg-alpha", type=float, default=0.0)
    parser.add_argument("--xgb-tree-method", default="hist")
    parser.add_argument("--xgb-verbosity", type=int, default=1)

    # LightGBM optional.
    parser.add_argument("--lgb-n-estimators", type=int, default=700)
    parser.add_argument("--lgb-learning-rate", type=float, default=0.03)
    parser.add_argument("--lgb-num-leaves", type=int, default=31)
    parser.add_argument("--lgb-subsample", type=float, default=0.9)
    parser.add_argument("--lgb-colsample-bytree", type=float, default=0.9)
    parser.add_argument("--lgb-reg-lambda", type=float, default=2.0)

    # Logreg.
    parser.add_argument("--logreg-max-iter", type=int, default=3000)

    parser.add_argument("--make-zip", action="store_true", default=True)
    parser.add_argument("--no-zip", dest="make_zip", action="store_false")
    args = parser.parse_args()

    repo_root = repo_root_from_here()
    out_dir = resolve_path(args.out_dir, repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[E1b] repo_root={repo_root}", flush=True)
    print(f"[E1b] out_dir={out_dir}", flush=True)
    print(f"[E1b] target_macro_f1=0.90", flush=True)

    inp = load_official_inputs(args, repo_root)
    base = load_baseline(args, repo_root, inp)
    base.to_csv(out_dir / "E1b_baseline_top2_context.csv", index=False)

    run_cfg = {
        "stage": "E1b_stronger_tabular_pair_experts",
        "target_macro_f1": 0.90,
        "args": vars(args),
        "input_source": {
            "dataset_npz": str(resolve_path(args.dataset_npz, repo_root)),
            "metadata_json": str(resolve_path(args.metadata_json, repo_root)),
            "train_raw": str(resolve_path(args.train_raw, repo_root)),
            "val_raw": str(resolve_path(args.val_raw, repo_root)),
            "official_train": str(resolve_path(args.official_train, repo_root)),
        },
        "label_names": inp["label_names"],
        "num_bins": int(inp["num_bins"]),
        "continuous_info": inp["continuous_info"],
    }
    save_json(out_dir / "E1b_run_config.json", run_cfg)

    print("[E1b] training candidate binary experts...", flush=True)
    metrics_df, candidates = train_candidates(inp, args, out_dir)

    print("[E1b] evaluating policies...", flush=True)
    policy_df, context, best_pred, best_applied = evaluate_families(inp, base, candidates, metrics_df, args, out_dir)
    best_outputs = save_best_outputs(inp, base, best_pred, best_applied, out_dir)

    best_row = policy_df.iloc[1].to_dict() if len(policy_df) > 1 else {}
    # policy_df row 0 is baseline due concat; true best is first non-baseline after sorting because baseline prepended.
    if len(policy_df) > 1:
        non_base = policy_df[policy_df["policy"] != "BASELINE"].copy()
        if len(non_base):
            best_row = non_base.sort_values(["macro_f1", "net_gain", "damage_ratio"], ascending=[False, False, True]).iloc[0].to_dict()

    summary = {
        "stage": "E1b_stronger_tabular_pair_experts",
        "target_macro_f1": 0.90,
        "current_reference_best_before_E1b": {
            "name": "E1a0 concat_all + ExtraTrees + threshold 0.55",
            "macro_f1": 0.826835,
            "accuracy": 0.884642,
            "net_gain": 126,
            "damage_ratio": 0.665,
        },
        "baseline_metrics": context["baseline_metrics"],
        "best_policy": context["best_policy"],
        "best_policy_row": best_row,
        "best_metrics": best_outputs["best_metrics"],
        "best_transition": best_outputs["best_transition"],
        "models": [m.strip() for m in args.models.split(",") if m.strip()],
        "representations": [r.strip() for r in args.reps.split(",") if r.strip()],
        "calibration": args.calibration,
        "binary_expert_metrics_top": (
            metrics_df[metrics_df["status"] == "ok"]
            .sort_values(["pair_key", "macro_f1", "auc"], ascending=[True, False, False])
            .groupby("pair_key")
            .head(5)
            .to_dict(orient="records")
        ),
        "outputs": {
            "run_config": str(out_dir / "E1b_run_config.json"),
            "binary_expert_metrics": str(out_dir / "E1b_binary_expert_metrics.csv"),
            "policy_metrics": str(out_dir / "E1b_policy_metrics.csv"),
            "best_policy_predictions": str(out_dir / "E1b_best_policy_predictions.csv"),
            "best_policy_pair_fix_damage": str(out_dir / "E1b_best_policy_pair_fix_damage.csv"),
            "best_policy_applied_samples": str(out_dir / "E1b_best_policy_applied_samples.csv"),
            "summary_md": str(out_dir / "E1b_summary.md"),
        },
        "guardrail": "Diagnostic validation-set policy search. Use to decide direction, not as unbiased final test score.",
    }
    save_json(out_dir / "E1b_summary.json", summary)
    write_summary_md(out_dir, summary)

    if args.make_zip:
        zip_path = out_dir.with_suffix(".zip")
        zip_dir(out_dir, zip_path)
        print(f"[E1b] zipped outputs: {zip_path}", flush=True)

    print("[E1b] done.", flush=True)
    print(f"[E1b] baseline_macro_f1={context['baseline_metrics']['macro_f1']:.6f}", flush=True)
    print(f"[E1b] best_policy={context['best_policy']}", flush=True)
    print(f"[E1b] best_macro_f1={best_outputs['best_metrics']['macro_f1']:.6f}", flush=True)
    print(f"[E1b] best_transition={best_outputs['best_transition']}", flush=True)


if __name__ == "__main__":
    main()
