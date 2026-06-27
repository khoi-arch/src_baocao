# Exact D3 train fix

This replaces the refactored D3 trainer with an exact-D3-only version derived from old `10_train_fusion_ablation_D0_D7.py`, and replaces `train_utils.py` with the old helper subset from `05_train.py` without importing `04_model.py`.

Run:

```bash
cd ~/Documents/src_baocao
bash /path/to/APPLY_EXACT_D3_TRAIN.sh "$PWD"
python 02_src/07_train.py --device auto
```

To verify C2 artifact equality separately, compare arrays between `03_outputs/05_dataset/dataset.npz` and old C2 artifact.
