# E0 Pair Input Separation Audit Summary

## Purpose

E0 audits whether hard malware subtype pairs can be separated at input/feature level,
instead of adding another head on the same representation.

## Inputs

- dataset: `/home/pak/Documents/src_baocao/03_outputs/05_dataset/dataset.npz`
- metadata: `/home/pak/Documents/src_baocao/03_outputs/05_dataset/metadata.json`
- train_raw: `/home/pak/Documents/src_baocao/01_split/train_raw.csv`
- val_raw: `/home/pak/Documents/src_baocao/01_split/val_raw.csv`
- baseline_pred: `/home/pak/Documents/src_baocao/03_outputs/06_model/val_predictions_best.csv`
- d1b_pred: `/home/pak/Documents/src_baocao/05_test/outputs/D1b_official_fork_lam0p01/val_predictions_best.csv`

## Key output files

- `E0_summary.json`
- `E0_pair_feature_rank.csv`
- `E0_pair_feature_overlap_stats.csv`
- `E0_pair_direction_feature_shift.csv`
- `E0_pair_interaction_candidates.csv`
- `E0_d1c_attention_vs_feature_signal.csv`

## How to read

1. Start with `E0_pair_feature_overlap_stats.csv`.
   - Look for pair/rep rows with high `max_auc_best`, high `mean_ks`, low `mean_iqr_overlap_ratio`.
2. Then open `E0_pair_feature_rank.csv`.
   - For each pair, inspect top features in `raw_scaled` and `d3_scalar`.
3. Then open `E0_pair_interaction_candidates.csv`.
   - If interactions have much higher score than single features, E1 should build pair-specific transformed inputs.
4. Then open `E0_d1c_attention_vs_feature_signal.csv`.
   - If D1c attention focuses on weak-separation features, D1b attention is not reliable for expert decisions.

## Decision logic

- If clear pair-specific features/interactions exist:
  proceed to E1/E2 expert with NEW pair-specific input.
- If not:
  overlap is likely too strong under current features; more heads on same representation are unlikely to help.
