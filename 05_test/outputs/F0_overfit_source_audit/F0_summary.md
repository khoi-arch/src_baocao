# F0 Overfit Source Audit

## Core result

```text
train macro-F1 = 0.910253
val macro-F1   = 0.810094
gap            = 0.100158

train acc      = 0.940652
val acc        = 0.873891
acc gap        = 0.066761
```

## Top-2 headroom

```text
train top2 acc = 0.991382
val top2 acc   = 0.968003

val wrong total        = 1478
val wrong true in top2 = 1103
```

## Token/K risk signals

```text
mean val unseen token sample rate      = 0.000093
mean val rare<=5 token sample rate     = 0.001778
mean token shortcut train-val acc gap  = 0.002882
max token shortcut train-val acc gap   = 0.012666
corr(val wrong, rare<=5 frac)          = 0.01442611516459573
corr(val wrong, unseen frac)           = 0.026286314446045827
```

## Raw feature drift risk signals

```text
max raw PSI = 0.001543
max raw KS  = 0.012368
mean raw PSI = 0.000512
```

## Preliminary diagnosis

- Strong generalization gap detected: train-val macro-F1 gap = 0.1002.

## Key files

- `F0_train_val_metrics.json`
- `F0_per_class_gap.csv`
- `F0_token_sparsity_audit.csv`
- `F0_token_shortcut_audit.csv`
- `F0_raw_feature_drift.csv`
- `F0_class_conditional_raw_drift.csv`
- `F0_raw_quantile_shortcut_audit.csv`
- `F0_val_sample_risk.csv`
- `F0_error_risk_summary.csv`

## How to use this

Do not jump to a fix from one number.

Use this order:

```text
1. If token shortcut gap / rare-token error correlation is high:
   test K/rare-bin/token dropout.

2. If raw drift and class-conditional drift are high:
   test stability-aware feature filtering/dropout.

3. If all branch inputs show gap later:
   test capacity/regularization.

4. Before any solution test:
   compare with previous E1-E5 experiments to avoid duplicates.
```
