# F3b0 OOF-selected Hard-Pair/Family Margin Loss

## Fix

```text
loss = CE_4class
     + lambda * ReLU(margin - (logit_true - logit_confuser))
```

Áp dụng chỉ cho sample thuộc hard pair-family chọn từ train-only OOF:

```text
F3a2_train_only_selected_hard_groups.json
```

Ví dụ:

```text
Trojan-Zeus -> Ransomware
=> ép logit_Trojan - logit_Ransomware >= margin
```

## Copy local

```bash
cd ~/Documents/src_baocao

mkdir -p /tmp/f3b0_pack
unzip -o ~/Downloads/F3b0_hard_margin_calibration_search_pack.zip -d /tmp/f3b0_pack

cp /tmp/f3b0_pack/02_src/07_train_hardmargin.py 02_src/07_train_hardmargin.py
cp /tmp/f3b0_pack/05_test/F3b0_hard_margin_calibration_search.py 05_test/F3b0_hard_margin_calibration_search.py

python -m py_compile 02_src/07_train_hardmargin.py
python -m py_compile 05_test/F3b0_hard_margin_calibration_search.py

git add 02_src/07_train_hardmargin.py 05_test/F3b0_hard_margin_calibration_search.py
git commit -m "add F3b0 OOF hard margin calibration search"
git push
```

## Chạy Kaggle F3b0

```python
import os, subprocess, shutil
from pathlib import Path

repo = Path("/kaggle/working/src_baocao")
os.chdir(repo)
subprocess.run(["git", "pull", "origin", "main"], check=True)

out = repo / "05_test/outputs/F3b0_hard_margin_calibration_search"
zip_path = repo / "05_test/outputs/F3b0_hard_margin_calibration_search.zip"

shutil.rmtree(out, ignore_errors=True)
if zip_path.exists():
    zip_path.unlink()

subprocess.run([
    "python", "05_test/F3b0_hard_margin_calibration_search.py",
    "--dataset-npz", "03_outputs/05_dataset/dataset.npz",
    "--metadata-json", "03_outputs/05_dataset/metadata.json",
    "--train-raw", "01_split/train_raw.csv",
    "--trainer", "02_src/07_train_hardmargin.py",
    "--selected-hard-groups-json", "05_test/outputs/F3a2_oof_aggregate_overlap_audit/F3a2_train_only_selected_hard_groups.json",
    "--out-dir", "05_test/outputs/F3b0_hard_margin_calibration_search",
    "--combined-zip", "05_test/outputs/F3b0_hard_margin_calibration_search.zip",
    "--lambda-grid", "0.0,0.3,0.5",
    "--margin-grid", "0.2,0.5",
    "--hard-margin-mode", "pair_family",
    "--calib-size", "0.20",
    "--epochs", "80",
    "--batch-size", "512",
    "--patience", "12",
    "--device", "cuda",
    "--num-workers", "2",
], check=True)

print("DONE:", zip_path)
```

Upload:

```text
05_test/outputs/F3b0_hard_margin_calibration_search.zip
```

## Sau F3b0

Nếu `selected_lambda > 0` và delta calibration > 0, chạy F3b1 full train + official val một lần.
