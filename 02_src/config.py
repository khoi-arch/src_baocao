#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.py

Unified config for cleaned src_baocao C2/D3 baseline repo.

Repo layout:
  01_split/
    train_raw.csv
    val_raw.csv

  02_src/
    00_token_diag.py
    01_preprocessing.py
    04_tokenization.py
    05_build_dataset.py
    06_train.py
    07_audit_best.py
    08_audit_rootcause.py

  03_outputs/
    00_dataset/
      dataset.npz
      metadata.json

    01_model/
      best_model.pt
      last_model.pt
      config.json
      diagnosis_summary.json
      reports/

    02_audit_best/
    03_audit_rootcause/

Main official baseline:
  C2 dataset policy:
    C2_selective_rank_discrete_compact

  D3 model:
    offset interpolation + raw FiLM/multiply fusion

  Official val macro-F1:
    0.8171466447478825
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


# ============================================================
# Root / base paths
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

RAW_DATASET_DIR = ROOT_DIR / "00_raw_dataset"
RAW_DATASET_CSV = RAW_DATASET_DIR / "Obfuscated-MalMem2022.csv"

SPLIT_DIR = ROOT_DIR / "01_split"
TRAIN_RAW_CSV = SPLIT_DIR / "train_raw.csv"
VAL_RAW_CSV = SPLIT_DIR / "val_raw.csv"

OUTPUT_ROOT = ROOT_DIR / "03_outputs"
OUTPUT_DIR = OUTPUT_ROOT

TOKEN_DIAG_DIR = OUTPUT_ROOT / "01_token_diag"
PREPROCESS_DIR = OUTPUT_ROOT / "02_preprocessing"
BIN_DIAG_DIR = OUTPUT_ROOT / "03_bin_diag"
BUILD_ROOT = OUTPUT_ROOT / "04_token"

DATASET_DIR = OUTPUT_ROOT / "00_dataset"
MODEL_DIR = OUTPUT_ROOT / "01_model"
AUDIT_BEST_DIR = OUTPUT_ROOT / "02_audit_best"
AUDIT_ROOTCAUSE_DIR = OUTPUT_ROOT / "03_audit_rootcause"

MODEL_REPORTS_DIR = MODEL_DIR / "reports"
MODEL_PREDICTIONS_DIR = MODEL_DIR / "predictions"


# ============================================================
# Official C2/D3 artifact paths
# ============================================================

DATASET_NPZ = DATASET_DIR / "dataset.npz"
DATASET_METADATA = DATASET_DIR / "metadata.json"
METADATA_JSON = DATASET_METADATA

BEST_MODEL = MODEL_DIR / "best_model.pt"
LAST_MODEL = MODEL_DIR / "last_model.pt"
MODEL_CONFIG = MODEL_DIR / "config.json"
CLASS_WEIGHTS_JSON = MODEL_DIR / "class_weights.json"
DIAGNOSIS_SUMMARY = MODEL_DIR / "diagnosis_summary.json"
HISTORY_CSV = MODEL_DIR / "history.csv"

VAL_PREDICTIONS = MODEL_PREDICTIONS_DIR / "val_predictions_best.csv"
TRAIN_PREDICTIONS = MODEL_PREDICTIONS_DIR / "train_predictions_best.csv"


# ============================================================
# Backward-compatible path aliases
# ============================================================

TRAIN_CSV = TRAIN_RAW_CSV
VAL_CSV = VAL_RAW_CSV

TRAIN_RAW = TRAIN_RAW_CSV
VAL_RAW = VAL_RAW_CSV

DATASET_PATH = DATASET_NPZ
METADATA_PATH = DATASET_METADATA
CHECKPOINT_PATH = BEST_MODEL

RUN_DIR = MODEL_DIR
OUT_DIR = MODEL_DIR

CHECKPOINT = BEST_MODEL
BEST_CHECKPOINT = BEST_MODEL


# ============================================================
# Labels / columns
# ============================================================

DEFAULT_LABEL_COL = "label_L2"
LABEL_COL = DEFAULT_LABEL_COL

LABEL_NAMES = ["Benign", "Ransomware", "Spyware", "Trojan"]
CLASS_NAMES = LABEL_NAMES
NUM_CLASSES = 4

# Các cột label/leakage không được dùng làm feature.
TARGET_COLS = [
    "label_L1",
    "label_L2",
    "label_L3",
    "Class",
    "Category",
]

# DROP_COLS để trống vì TARGET_COLS đã loại label.
# Nếu sau này có thêm ID/hash/path/timestamp thì thêm vào đây.
DROP_COLS: List[str] = []

# Mapping official trong artifact cũ có thể có Benign bị padding space.
# Code mới nên strip label khi cần, nhưng giữ alias này để tương thích.
LABEL_MAPPING = {
    "Benign": 0,
    "Ransomware": 1,
    "Spyware": 2,
    "Trojan": 3,
}

ID_TO_LABEL = {v: k for k, v in LABEL_MAPPING.items()}


# ============================================================
# Token / preprocessing settings
# ============================================================

TOKEN_K = 512
K = TOKEN_K

VALUE_NUM_BINS = 512
NUM_BINS = VALUE_NUM_BINS
EFFECTIVE_TOKEN_BUDGET = 512

# 00_token_diag.py thresholds
UNIQUE_PRESERVE_THRESHOLD = 0.95
COMPRESSION_FACTOR_THRESHOLD = 8.0
TOKEN_ENTROPY_NORM_THRESHOLD = 0.75
TOKEN_QUANTILES = [0, 10, 25, 75, 90, 100]

# 01_preprocessing.py candidates
PREPROCESS_BLEND_ALPHAS = "0.25,0.50,0.75"

# 04_tokenization.py default selection rule
MIN_UNIQUE_FOR_QUANTILE = 128
MIN_DOMINANT_REDUCTION = 0.02
MIN_ENTROPY_DELTA = 0.05


# ============================================================
# Intermediate pipeline paths
# ============================================================

def token_diag_json_path(K: int = TOKEN_K) -> Path:
    return TOKEN_DIAG_DIR / f"token_diag_train_B{int(K)}.json"


def token_diag_summary_csv_path(K: int = TOKEN_K) -> Path:
    return TOKEN_DIAG_DIR / f"token_diag_summary_B{int(K)}.csv"


def preprocess_train_csv_path(K: int = TOKEN_K) -> Path:
    return PREPROCESS_DIR / f"train_preprocessed_K{int(K)}.csv"


def preprocess_val_csv_path(K: int = TOKEN_K) -> Path:
    return PREPROCESS_DIR / f"val_preprocessed_K{int(K)}.csv"


def preprocess_policy_json_path(K: int = TOKEN_K) -> Path:
    return PREPROCESS_DIR / f"preprocess_policy_K{int(K)}.json"


def preprocess_report_json_path(K: int = TOKEN_K) -> Path:
    return PREPROCESS_DIR / f"preprocess_report_K{int(K)}.json"


def bin_diag_json_path(K: int = TOKEN_K, B: int = VALUE_NUM_BINS) -> Path:
    return BIN_DIAG_DIR / f"quantile_vs_uniform_bin_diag_K{int(K)}_B{int(B)}.json"


def bin_diag_csv_path(K: int = TOKEN_K, B: int = VALUE_NUM_BINS) -> Path:
    return BIN_DIAG_DIR / f"quantile_vs_uniform_bin_diag_K{int(K)}_B{int(B)}.csv"


def build_mixed_dir(K: int = TOKEN_K, B: int = VALUE_NUM_BINS) -> Path:
    return BUILD_ROOT / f"K{int(K)}_B{int(B)}"


def build_rank_uniform_dir(K: int = TOKEN_K, B: int = VALUE_NUM_BINS) -> Path:
    return BUILD_ROOT / f"K{int(K)}_B{int(B)}_rank_uniform_only"


def build_c2_dir(K: int = TOKEN_K, B: int = VALUE_NUM_BINS) -> Path:
    # Clean repo stores official C2 final dataset here.
    return DATASET_DIR


# Static aliases for old scripts.
TOKEN_DIAG_JSON = token_diag_json_path(TOKEN_K)
TOKEN_DIAG_SUMMARY_CSV = token_diag_summary_csv_path(TOKEN_K)

TRAIN_PREPROCESSED_CSV = preprocess_train_csv_path(TOKEN_K)
VAL_PREPROCESSED_CSV = preprocess_val_csv_path(TOKEN_K)
PREPROCESS_POLICY_JSON = preprocess_policy_json_path(TOKEN_K)
PREPROCESS_REPORT_JSON = preprocess_report_json_path(TOKEN_K)

BIN_DIAG_JSON = bin_diag_json_path(TOKEN_K, VALUE_NUM_BINS)
BIN_DIAG_CSV = bin_diag_csv_path(TOKEN_K, VALUE_NUM_BINS)

A_CURRENT_MIXED_DIR = build_mixed_dir(TOKEN_K, VALUE_NUM_BINS)
B_RANK_UNIFORM_DIR = build_rank_uniform_dir(TOKEN_K, VALUE_NUM_BINS)
C2_DATASET_DIR = DATASET_DIR


# ============================================================
# D3 model settings
# ============================================================

RUN_ID = "D3"
RUN_NAME = "D3_C2D3_official"

REPRESENTATION = "D3_offset interpolation + raw FiLM/multiply fusion"
LOCAL = "offset_interpolation"
CONTINUOUS_SOURCE = "raw_scaled"
FUSION = "raw_film"

VALUE_EMBED_DIM = 32
FEATURE_EMBED_DIM = 32
VALUE_DIM = VALUE_EMBED_DIM
FEATURE_DIM = FEATURE_EMBED_DIM
CELL_DIM = VALUE_DIM + FEATURE_DIM

MODEL_HIDDEN_DIM = 128
MODEL_NUM_LAYERS = 3
MODEL_NUM_HEADS = 4
MODEL_DROPOUT = 0.1
MODEL_ACTIVATION = "gelu"
TRANSFORMER_NORM_FIRST = True

CLASSIFIER_HIDDEN_DIM = 128
CLASSIFIER_DROPOUT = 0.1

GATE_INIT = 0.0


# ============================================================
# Training settings
# ============================================================

SEED = 42
TRAIN_SEED = SEED

DEVICE = "auto"
TRAIN_DEVICE = DEVICE

EPOCHS = 80
TRAIN_EPOCHS = EPOCHS

BATCH_SIZE = 512
TRAIN_BATCH_SIZE = BATCH_SIZE

LR = 1e-3
TRAIN_LR = LR

WEIGHT_DECAY = 1e-4
TRAIN_WEIGHT_DECAY = WEIGHT_DECAY

SCHEDULER = "warmup_cosine"
TRAIN_SCHEDULER = SCHEDULER

WARMUP_EPOCHS = 8
TRAIN_WARMUP_EPOCHS = WARMUP_EPOCHS

MIN_LR_RATIO = 0.05
TRAIN_MIN_LR_RATIO = MIN_LR_RATIO

PATIENCE = 12
TRAIN_PATIENCE = PATIENCE

MIN_DELTA = 1e-4
TRAIN_MIN_DELTA = MIN_DELTA

GRAD_CLIP_NORM = 1.0
TRAIN_GRAD_CLIP_NORM = GRAD_CLIP_NORM

NUM_WORKERS = 0
TRAIN_NUM_WORKERS = NUM_WORKERS

USE_CLASS_WEIGHTS = True


# ============================================================
# Official expected metrics
# ============================================================

EXPECTED_N_TRAIN = 46876
EXPECTED_N_VAL = 11720

EXPECTED_VAL_MACRO_F1 = 0.8171466447478825
EXPECTED_VAL_ACCURACY = 0.8784982935153584


# ============================================================
# Utility functions
# ============================================================

def cfg(name: str, default: Any = None) -> Any:
    """
    Backward-compatible helper for older scripts.
    Example:
      cfg("OUTPUT_ROOT", Path("03_outputs"))
    """
    return globals().get(name, default)


get = cfg


def ensure_dirs() -> None:
    for p in [
        OUTPUT_ROOT,
        TOKEN_DIAG_DIR,
        PREPROCESS_DIR,
        BIN_DIAG_DIR,
        BUILD_ROOT,
        DATASET_DIR,
        MODEL_DIR,
        MODEL_REPORTS_DIR,
        MODEL_PREDICTIONS_DIR,
        AUDIT_BEST_DIR,
        AUDIT_ROOTCAUSE_DIR,
    ]:
        Path(p).mkdir(parents=True, exist_ok=True)


def as_posix(p: Path | str) -> str:
    return str(Path(p).as_posix())


def repo_summary() -> Dict[str, Any]:
    return {
        "ROOT_DIR": str(ROOT_DIR),
        "TRAIN_RAW_CSV": str(TRAIN_RAW_CSV),
        "VAL_RAW_CSV": str(VAL_RAW_CSV),
        "DATASET_NPZ": str(DATASET_NPZ),
        "DATASET_METADATA": str(DATASET_METADATA),
        "MODEL_DIR": str(MODEL_DIR),
        "BEST_MODEL": str(BEST_MODEL),
        "RUN_ID": RUN_ID,
        "TOKEN_K": TOKEN_K,
        "VALUE_NUM_BINS": VALUE_NUM_BINS,
        "DEFAULT_LABEL_COL": DEFAULT_LABEL_COL,
        "EXPECTED_VAL_MACRO_F1": EXPECTED_VAL_MACRO_F1,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(repo_summary(), indent=2, ensure_ascii=False))
