# E1b Stronger Tabular Pair Experts

## Goal

Continue from E1a0, current best macro-F1 ≈ 0.826835, toward target macro-F1 0.90.

## Design

- Full-feature tabular binary experts.
- Default representation: `concat_all`.
- Pair experts: RS, RT, ST.
- Stronger models tried: extratrees, extratrees_deep, histgb
- Calibration: none
- Pair-specific threshold and baseline margin-cap search.

## Baseline

```text
accuracy = 0.873891
macro_f1 = 0.810094
weighted = 0.873474
```

## Best E1b

```text
policy   = mixed_best_binary_auc__global_thr_0.55__margin_cap_1e+09
accuracy = 0.886348
macro_f1 = 0.829387
weighted = 0.886286
```

## Transition

```text
wrong_to_correct = 396
correct_to_wrong = 250
net_gain         = 146
damage_ratio     = 0.6313131313131313
changed_pred_n   = 727
```

## Key files

- `E1b_binary_expert_metrics.csv`
- `E1b_policy_metrics.csv`
- `E1b_best_policy_predictions.csv`
- `E1b_best_policy_pair_fix_damage.csv`
- `E1b_best_policy_applied_samples.csv`
