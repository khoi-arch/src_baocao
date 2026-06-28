# E1a0 Full-Feature Binary Expert Diagnostic

## Purpose

Train 3 standalone binary experts on full features:
- Ransomware vs Spyware
- Ransomware vs Trojan
- Spyware vs Trojan

Then intervene only when baseline top1/top2 are both malware and form a hard pair.

This tests whether a pure 2-class expert using the same full input can improve the final 4-class decision.

## Best policy

```text
concat_all__extratrees__top2_hardpair_thr_0.55
```

## Baseline

```text
accuracy  = 0.873891
macro_f1  = 0.810094
weighted  = 0.873474
```

## Best E1a0 policy

```text
accuracy  = 0.884642
macro_f1  = 0.826835
weighted  = 0.884562
```

## Best transition

```text
wrong_to_correct = 376
correct_to_wrong = 250
net_gain         = 126
damage_ratio     = 0.6648936170212766
changed_pred_n   = 705
```

## Output files

- `E1a0_summary.json`
- `E1a0_binary_expert_metrics.csv`
- `E1a0_policy_metrics.csv`
- `E1a0_policy_per_class_f1.csv`
- `E1a0_best_policy_predictions.csv`
- `E1a0_best_policy_confusion_matrix.csv`
- `E1a0_best_policy_pair_fix_damage.csv`
- `E1a0_best_policy_applied_samples.csv`

## Interpretation

If binary expert metrics are high but policy net_gain is small/damage_ratio high,
then the binary expert is not safe enough for final top-2 intervention.
