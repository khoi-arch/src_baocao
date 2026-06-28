#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.py

Official cleaned src_baocao C2/D3 pipeline config.

Pipeline contract:
  00_token_diag.py
      raw train -> 03_outputs/01_token_diag

  01_preprocessing.py
      raw train/val + token diag -> 03_outputs/02_preprocessing

  03_bin_diag.py
      preprocessed train + policy -> 03_outputs/03_bin_diag

  04_tokenization.py
      preprocessed train/val + bin diag -> token source artifacts:
        A: 03_outputs/04_token/K512_B512/token_artifact.npz
        B: 03_outputs/04_token/K512_B512_rank_uniform_only/token_artifact.npz

  05_build_dataset.py
      A+B token sources + raw train/val -> official C2 final dataset:
        03_outputs/05_dataset/dataset.npz
        03_outputs/05_dataset/metadata.json

  07_train.py
      C2 final dataset + raw train/val -> official D3 model:
        03_outputs/06_model/best_model.pt

Official baseline:
  C2 = selective_rank_discrete_compact dataset policy
  D3 = offset interpolation + raw_scaled FiLM/multiply fusion
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

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
TOKEN_ROOT = OUTPUT_ROOT / "04_token"
BUILD_ROOT = TOKEN_ROOT
DATASET_DIR = OUTPUT_ROOT / "05_dataset"
MODEL_DIR = OUTPUT_ROOT / "06_model"
AUDIT_BEST_DIR = OUTPUT_ROOT / "07_audit_best"
AUDIT_ROOTCAUSE_DIR = OUTPUT_ROOT / "08_audit_rootcause"

MODEL_REPORTS_DIR = MODEL_DIR / "reports"
MODEL_PREDICTIONS_DIR = MODEL_DIR / "predictions"

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

# Backward-compatible aliases.
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

DEFAULT_LABEL_COL = "label_L2"
LABEL_COL = DEFAULT_LABEL_COL
LABEL_NAMES = ["Benign", "Ransomware", "Spyware", "Trojan"]
CLASS_NAMES = LABEL_NAMES
NUM_CLASSES = 4

TARGET_COLS = ["label_L1", "label_L2", "label_L3", "Class", "Category"]
DROP_COLS: List[str] = []

LABEL_MAPPING = {"Benign": 0, "Ransomware": 1, "Spyware": 2, "Trojan": 3}
ID_TO_LABEL = {v: k for k, v in LABEL_MAPPING.items()}

TOKEN_K = 512
K = TOKEN_K
VALUE_NUM_BINS = 512
NUM_BINS = VALUE_NUM_BINS
EFFECTIVE_TOKEN_BUDGET = VALUE_NUM_BINS

UNIQUE_PRESERVE_THRESHOLD = 0.95
COMPRESSION_FACTOR_THRESHOLD = 8.0
TOKEN_ENTROPY_NORM_THRESHOLD = 0.75
TOKEN_QUANTILES = [0, 10, 25, 75, 90, 100]

PREPROCESS_BLEND_ALPHAS = "0.25,0.50,0.75"

MIN_UNIQUE_FOR_QUANTILE = 128
MIN_DOMINANT_REDUCTION = 0.02
MIN_ENTROPY_DELTA = 0.05

# C2 policy thresholds.
C2_RANK_MIN_RAW_UNIQUE = 512
C2_RANK_MIN_COMPRESSION = 8.0
C2_RANK_MIN_DOMINANT = 0.10
C2_RANK_MAX_ENTROPY = 0.75
C2_RANK_MIN_B_BINS_USED = 256
C2_RANK_MAX_B_RARE_LE5_RATIO = 0.30
C2_COMPACT_MAX_UNIQUE = 128
C2_COMPACT_MAX_UNIQUE_WITH_LOW_BINS = 512
C2_COMPACT_MAX_USED_BINS = 128

RUN_ID = "D3"
RUN_NAME = "D3_C2D3_official"
REPRESENTATION = "D3_offset_interpolation_raw_film"
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

SEED = 42
TRAIN_SEED = SEED
DEVICE = "auto"
TRAIN_DEVICE = DEVICE
EPOCHS = 80
TRAIN_EPOCHS = EPOCHS
BATCH_SIZE = 256
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

EXPECTED_N_TRAIN = 46876
EXPECTED_N_VAL = 11720
EXPECTED_VAL_MACRO_F1 = 0.8171466447478825
EXPECTED_VAL_ACCURACY = 0.8784982935153584

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

def rank_uniform_diag_json_path(K: int = TOKEN_K, B: int = VALUE_NUM_BINS) -> Path:
    return BIN_DIAG_DIR / f"rank_uniform_token_diag_K{int(K)}_B{int(B)}.json"

def rank_uniform_diag_csv_path(K: int = TOKEN_K, B: int = VALUE_NUM_BINS) -> Path:
    return BIN_DIAG_DIR / f"rank_uniform_token_diag_K{int(K)}_B{int(B)}.csv"

def build_mixed_dir(K: int = TOKEN_K, B: int = VALUE_NUM_BINS) -> Path:
    return TOKEN_ROOT / f"K{int(K)}_B{int(B)}"

def build_rank_uniform_dir(K: int = TOKEN_K, B: int = VALUE_NUM_BINS) -> Path:
    return TOKEN_ROOT / f"K{int(K)}_B{int(B)}_rank_uniform_only"

def token_artifact_npz_path(K: int = TOKEN_K, B: int = VALUE_NUM_BINS) -> Path:
    return build_mixed_dir(K, B) / "token_artifact.npz"

def rank_uniform_token_artifact_npz_path(K: int = TOKEN_K, B: int = VALUE_NUM_BINS) -> Path:
    return build_rank_uniform_dir(K, B) / "token_artifact.npz"

def build_c2_dir(K: int = TOKEN_K, B: int = VALUE_NUM_BINS) -> Path:
    return DATASET_DIR

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

def cfg(name: str, default: Any = None) -> Any:
    return globals().get(name, default)

get = cfg

def ensure_dirs() -> None:
    for p in [
        OUTPUT_ROOT,
        TOKEN_DIAG_DIR,
        PREPROCESS_DIR,
        BIN_DIAG_DIR,
        TOKEN_ROOT,
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
        "A_TOKEN_ARTIFACT": str(token_artifact_npz_path()),
        "B_TOKEN_ARTIFACT": str(rank_uniform_token_artifact_npz_path()),
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
