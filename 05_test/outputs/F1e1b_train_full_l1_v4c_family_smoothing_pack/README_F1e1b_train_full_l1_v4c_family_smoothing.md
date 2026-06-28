# F1e1b Full Train L1 + Locked v4c Family Smoothing

## Mục tiêu

Train model thật trên full original train với matrix sạch từ F1e1a_v4c.

```text
Không dùng val để derive matrix.
Không dùng val để early stop.
Không tune theo val.
Evaluate val một lần sau khi train xong.
```

## Input cần có

Cần output v4c:

```text
05_test/outputs/F1e1a_v4c_prob_only_calibration_family_matrix/
  F1e1a_v4c_locked_family_smoothing_matrix_PROB_ONLY_CALIBRATION_DERIVED.csv
```

## Copy local

```bash
cd ~/Documents/src_baocao

mkdir -p /tmp/f1e1b_pack
unzip -o ~/Downloads/F1e1b_train_full_l1_v4c_family_smoothing_pack.zip -d /tmp/f1e1b_pack

cp /tmp/f1e1b_pack/F1e1b_train_full_l1_v4c_family_smoothing.py \
   05_test/F1e1b_train_full_l1_v4c_family_smoothing.py

chmod +x 05_test/F1e1b_train_full_l1_v4c_family_smoothing.py
python -m py_compile 05_test/F1e1b_train_full_l1_v4c_family_smoothing.py

git add 05_test/F1e1b_train_full_l1_v4c_family_smoothing.py
git commit -m "add F1e1b full L1 v4c family smoothing"
git push
```

## Chạy Kaggle

Nếu Kaggle session mới không có v4c output, cần restore/unzip v4c zip trước.

Sau đó chạy:

```python
import os, subprocess, shutil
from pathlib import Path

repo = Path("/kaggle/working/src_baocao")
os.chdir(repo)

subprocess.run(["git", "pull", "origin", "main"], check=True)

out = repo / "05_test/outputs/F1e1b_full_l1_v4c_family_smoothing"
zip_path = repo / "05_test/outputs/F1e1b_full_l1_v4c_family_smoothing.zip"

shutil.rmtree(out, ignore_errors=True)
if zip_path.exists():
    zip_path.unlink()

subprocess.run([
    "python", "05_test/F1e1b_train_full_l1_v4c_family_smoothing.py",
    "--dataset-npz", "03_outputs/05_dataset/dataset.npz",
    "--train-raw", "01_split/train_raw.csv",
    "--val-raw", "01_split/val_raw.csv",
    "--base-config", "03_outputs/06_model/config.json",
    "--v4-dir", "05_test/outputs/F1e1a_v4_clean_calibration_family_smoothing",
    "--matrix-csv", "05_test/outputs/F1e1a_v4c_prob_only_calibration_family_matrix/F1e1a_v4c_locked_family_smoothing_matrix_PROB_ONLY_CALIBRATION_DERIVED.csv",
    "--out-dir", "05_test/outputs/F1e1b_full_l1_v4c_family_smoothing",
    "--combined-zip", "05_test/outputs/F1e1b_full_l1_v4c_family_smoothing.zip",
    "--device", "cuda",
    "--batch-size", "512",
    "--num-workers", "2",
], check=True)

print("DONE:", zip_path)
```

## Output chính

```text
F1e1b_report.md
config.json
history.csv
final_model.pt
final_metrics.json
train_classification_report_final.json
val_classification_report_final.json
train_confusion_matrix_final.csv
val_confusion_matrix_final.csv
train_predictions_final.csv
val_predictions_final.csv
```

## Cách đọc

So với L1 official:

```text
L1 val macro-F1 ≈ 0.814224
```

Nếu F1e1b > L1 và gap không tăng quá mạnh, smoothing có tín hiệu.

Nếu F1e1b giảm, không tune bằng val ngay. Khi đó cần quay lại calibration/OOF rule.
