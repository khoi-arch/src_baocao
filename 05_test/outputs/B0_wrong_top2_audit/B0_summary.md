# B0 — Wrong-sample top-2 coverage audit

## Purpose

Check whether the true label is still inside the model's top-2 candidates among validation samples predicted incorrectly by the official C2+D3 baseline.

## Input

- prediction CSV: `/home/pak/Documents/src_baocao/05_test/outputs/B0_wrong_top2_audit/val_predictions_with_probs.csv`
- true label column: `true_label`
- predicted label column: `pred_label`
- top-2 source: `explicit_top_columns:top1_label,top2_label`

## Main metrics

- `n_total`: 11720
- `n_correct`: 10242
- `n_wrong`: 1478
- `accuracy_from_predictions`: 0.873891
- `wrong_true_in_top2`: 1103
- `wrong_true_in_top2_rate`: 0.746279

## Interpretation gate

- Result: high wrong-sample top-2 coverage. Pairwise/reranking/auxiliary-head directions are worth testing next.

## Wrong top-2 coverage by true class

| true_class   |   n_total |   n_correct |   n_wrong |   wrong_true_in_top2 |   wrong_true_in_top2_rate |   class_error_rate |
|:-------------|----------:|------------:|----------:|---------------------:|--------------------------:|-------------------:|
| Ransomware   |      1959 |        1356 |       603 |                  454 |                  0.752902 |        0.30781     |
| Trojan       |      1897 |        1389 |       508 |                  392 |                  0.771654 |        0.267791    |
| Spyware      |      2004 |        1641 |       363 |                  257 |                  0.707989 |        0.181138    |
| Benign       |      5860 |        5856 |         4 |                    0 |                  0        |        0.000682594 |

## Top confusion pairs

| true_class   | pred_class   |   n_wrong |   wrong_true_in_top2 |   wrong_true_in_top2_rate |
|:-------------|:-------------|----------:|---------------------:|--------------------------:|
| Ransomware   | Spyware      |       302 |                  210 |                  0.695364 |
| Ransomware   | Trojan       |       301 |                  244 |                  0.810631 |
| Trojan       | Ransomware   |       295 |                  256 |                  0.867797 |
| Spyware      | Trojan       |       217 |                  163 |                  0.751152 |
| Trojan       | Spyware      |       213 |                  136 |                  0.638498 |
| Spyware      | Ransomware   |       144 |                   94 |                  0.652778 |
| Benign       | Ransomware   |         3 |                    0 |                  0        |
| Spyware      | Benign       |         2 |                    0 |                  0        |
| Benign       | Spyware      |         1 |                    0 |                  0        |

## Hard malware pairs

| pair                 | direction           | resolved_true_label   | resolved_pred_label   |   n_wrong |   wrong_true_in_top2 |   wrong_true_in_top2_rate | note   |
|:---------------------|:--------------------|:----------------------|:----------------------|----------:|---------------------:|--------------------------:|:-------|
| Ransomware<->Spyware | Ransomware->Spyware | Ransomware            | Spyware               |       302 |                  210 |                  0.695364 | ok     |
| Ransomware<->Spyware | Spyware->Ransomware | Spyware               | Ransomware            |       144 |                   94 |                  0.652778 | ok     |
| Ransomware<->Trojan  | Ransomware->Trojan  | Ransomware            | Trojan                |       301 |                  244 |                  0.810631 | ok     |
| Ransomware<->Trojan  | Trojan->Ransomware  | Trojan                | Ransomware            |       295 |                  256 |                  0.867797 | ok     |
| Spyware<->Trojan     | Spyware->Trojan     | Spyware               | Trojan                |       217 |                  163 |                  0.751152 | ok     |
| Spyware<->Trojan     | Trojan->Spyware     | Trojan                | Spyware               |       213 |                  136 |                  0.638498 | ok     |

## Generated files

- `/home/pak/Documents/src_baocao/05_test/outputs/B0_wrong_top2_audit/B0_summary.md`
- `/home/pak/Documents/src_baocao/05_test/outputs/B0_wrong_top2_audit/B0_metrics.json`
- `/home/pak/Documents/src_baocao/05_test/outputs/B0_wrong_top2_audit/B0_wrong_by_true_class.csv`
- `/home/pak/Documents/src_baocao/05_test/outputs/B0_wrong_top2_audit/B0_wrong_by_confusion_pair.csv`
- `/home/pak/Documents/src_baocao/05_test/outputs/B0_wrong_top2_audit/B0_hard_malware_pairs.csv`
- `/home/pak/Documents/src_baocao/05_test/outputs/B0_wrong_top2_audit/B0_wrong_samples_top2.csv`
- `/home/pak/Documents/src_baocao/05_test/outputs/B0_wrong_top2_audit/B0_run_manifest.json`
