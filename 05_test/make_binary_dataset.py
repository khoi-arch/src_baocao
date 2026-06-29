import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


SRC_DIR = Path("03_outputs/05_dataset")
DST_DIR = Path("03_outputs/05_dataset_binary")

TRAIN_RAW = Path("01_split/train_raw.csv")
VAL_RAW = Path("01_split/val_raw.csv")


def find_binary_labels(df: pd.DataFrame):
    # Ưu tiên cột binary nếu có
    candidates = ["Class", "label_L1", "Label", "label"]
    for c in candidates:
        if c in df.columns:
            vals = df[c].astype(str).str.lower()
            uniq = sorted(vals.unique())
            print(f"[INFO] candidate {c}: {uniq[:20]}")
            if any("benign" in x for x in uniq):
                y = np.where(vals.str.contains("benign"), 0, 1).astype(np.int64)
                return c, y

    # Nếu không có Class/label_L1 thì dùng label_L2/Category: Benign = 0, còn lại Malware = 1
    fallback = ["label_L2", "Category"]
    for c in fallback:
        if c in df.columns:
            vals = df[c].astype(str).str.lower()
            uniq = sorted(vals.unique())
            print(f"[INFO] fallback {c}: {uniq[:20]}")
            if any("benign" in x for x in uniq):
                y = np.where(vals.str.contains("benign"), 0, 1).astype(np.int64)
                return c, y

    raise RuntimeError(
        "Không tìm thấy cột nhãn binary. Hãy kiểm tra các cột Class/label_L1/label_L2/Category."
    )


def replace_label_arrays(npz_data, y_train_bin, y_val_bin):
    out = {}
    n_train = len(y_train_bin)
    n_val = len(y_val_bin)
    n_all = n_train + n_val

    replaced = []

    for k in npz_data.files:
        arr = npz_data[k]

        # Thay các key label phổ biến
        lower = k.lower()
        if lower in ["y_train", "train_y", "labels_train", "train_labels"]:
            out[k] = y_train_bin
            replaced.append(k)
            continue

        if lower in ["y_val", "val_y", "y_valid", "valid_y", "labels_val", "val_labels"]:
            out[k] = y_val_bin
            replaced.append(k)
            continue

        if lower in ["y", "labels", "target", "targets"] and arr.ndim == 1 and len(arr) == n_all:
            out[k] = np.concatenate([y_train_bin, y_val_bin])
            replaced.append(k)
            continue

        # Heuristic: mảng 1D integer, length bằng train/val, unique ít -> có thể là label
        if arr.ndim == 1 and np.issubdtype(arr.dtype, np.integer):
            uniq = np.unique(arr)
            if len(uniq) <= 20:
                if len(arr) == n_train and ("train" in lower or lower.startswith("y")):
                    out[k] = y_train_bin
                    replaced.append(k)
                    continue
                if len(arr) == n_val and ("val" in lower or "valid" in lower):
                    out[k] = y_val_bin
                    replaced.append(k)
                    continue

        out[k] = arr

    return out, replaced


def main():
    if not SRC_DIR.exists():
        raise FileNotFoundError(SRC_DIR)

    train_df = pd.read_csv(TRAIN_RAW)
    val_df = pd.read_csv(VAL_RAW)

    train_col, y_train_bin = find_binary_labels(train_df)
    val_col, y_val_bin = find_binary_labels(val_df)

    print("[INFO] train label source:", train_col)
    print("[INFO] val label source:", val_col)
    print("[INFO] train binary counts:", dict(zip(*np.unique(y_train_bin, return_counts=True))))
    print("[INFO] val binary counts:", dict(zip(*np.unique(y_val_bin, return_counts=True))))

    if DST_DIR.exists():
        shutil.rmtree(DST_DIR)
    DST_DIR.mkdir(parents=True, exist_ok=True)

    # Copy toàn bộ file phụ trong dataset folder
    for p in SRC_DIR.iterdir():
        if p.name not in ["dataset.npz", "metadata.json"]:
            if p.is_file():
                shutil.copy2(p, DST_DIR / p.name)
            elif p.is_dir():
                shutil.copytree(p, DST_DIR / p.name)

    npz = np.load(SRC_DIR / "dataset.npz", allow_pickle=True)
    new_data, replaced = replace_label_arrays(npz, y_train_bin, y_val_bin)

    if not replaced:
        raise RuntimeError(
            "Không tìm thấy key label trong dataset.npz để thay. "
            "Chạy lệnh inspect npz rồi gửi output cho ChatGPT."
        )

    np.savez_compressed(DST_DIR / "dataset.npz", **new_data)

    meta_path = SRC_DIR / "metadata.json"
    if meta_path.exists():
        meta = json.load(open(meta_path, "r", encoding="utf-8"))
    else:
        meta = {}

    # Ghi đè các metadata phổ biến
    meta["task"] = "binary_benign_vs_malware"
    meta["target"] = "Benign_vs_Malware"
    meta["num_classes"] = 2
    meta["class_names"] = ["Benign", "Malware"]
    meta["label_encoder_classes"] = ["Benign", "Malware"]
    meta["label_mapping"] = {"Benign": 0, "Malware": 1}
    meta["source_label_columns"] = {"train": train_col, "val": val_col}
    meta["note"] = (
        "Binary dataset derived from original 4-class dataset. "
        "Features unchanged; labels collapsed to Benign=0, Malware=1."
    )

    json.dump(meta, open(DST_DIR / "metadata.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print("[OK] Created:", DST_DIR)
    print("[OK] Replaced label arrays:", replaced)
    print("[OK] Binary metadata written:", DST_DIR / "metadata.json")


if __name__ == "__main__":
    main()
