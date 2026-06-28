# C2 — Top-2 oracle upper-bound audit

## Purpose

Estimate upper bounds for top-2 correction before training any reranker or changing the model.

## Interpretation gate

- Result: **MIXED — large oracle headroom, but simple margin-bounded correction captures limited gain**
- Reason: All-top2 oracle macro-F1=0.9523 (delta=0.1423), but margin<=0.2 oracle delta_macro_f1=0.0378. This means recoverable errors exist, but many require more than a small-margin rule.
- Recommendation: Proceed to C3 external CLS/logit pairwise reranker diagnostic. Do not rely on a simple margin-threshold rule.

## Key baseline/oracle metrics

| policy                          |   n_corrected_samples |   accuracy |   delta_accuracy |   macro_f1 |   delta_macro_f1 |   weighted_f1 |   delta_weighted_f1 |   newly_correct_n |   newly_wrong_n |
|:--------------------------------|----------------------:|-----------:|-----------------:|-----------:|-----------------:|--------------:|--------------------:|------------------:|----------------:|
| original                        |                     0 |   0.873891 |        0         |   0.810094 |        0         |      0.873474 |           0         |                 0 |               0 |
| oracle_all_wrong_true_in_top2   |                  1103 |   0.968003 |        0.0941126 |   0.952345 |        0.14225   |      0.968024 |           0.0945497 |              1103 |               0 |
| oracle_wrong_top2_margin_le_0.1 |                   150 |   0.886689 |        0.0127986 |   0.829569 |        0.0194751 |      0.88638  |           0.0129053 |               150 |               0 |
| oracle_wrong_top2_margin_le_0.2 |                   292 |   0.898805 |        0.0249147 |   0.847896 |        0.0378016 |      0.898542 |           0.0250675 |               292 |               0 |
| oracle_wrong_top2_margin_le_0.3 |                   384 |   0.906655 |        0.0327645 |   0.859791 |        0.0496968 |      0.906429 |           0.0329544 |               384 |               0 |
| oracle_wrong_top2_margin_le_0.5 |                   547 |   0.920563 |        0.0466724 |   0.880886 |        0.0707915 |      0.92044  |           0.0469654 |               547 |               0 |

## Per-class F1 under important policies

| policy                          |   f1_Benign |   f1_Ransomware |   f1_Spyware |   f1_Trojan |   delta_f1_Benign |   delta_f1_Ransomware |   delta_f1_Spyware |   delta_f1_Trojan |
|:--------------------------------|------------:|----------------:|-------------:|------------:|------------------:|----------------------:|-------------------:|------------------:|
| original                        |    0.999488 |        0.721853 |     0.788753 |    0.730284 |                 0 |             0         |          0         |         0         |
| oracle_all_wrong_true_in_top2   |    0.999488 |        0.937581 |     0.93222  |    0.94009  |                 0 |             0.215728  |          0.143467  |         0.209806  |
| oracle_wrong_top2_margin_le_0.2 |    0.999488 |        0.777925 |     0.82124  |    0.792931 |                 0 |             0.0560726 |          0.0324871 |         0.0626467 |
| oracle_wrong_top2_margin_le_0.5 |    0.999488 |        0.827696 |     0.851423 |    0.844937 |                 0 |             0.105843  |          0.0626704 |         0.114653  |

## Margin-bounded correctable counts by hard pair

| pair                 | direction                      |   n_wrong |   n_wrong_true_in_top2 |   wrong_true_in_top2_rate |   correctable_margin_le_0.1_n |   correctable_margin_le_0.2_n |   correctable_margin_le_0.3_n |   correctable_margin_le_0.5_n |
|:---------------------|:-------------------------------|----------:|-----------------------:|--------------------------:|------------------------------:|------------------------------:|------------------------------:|------------------------------:|
| Ransomware<->Spyware | Ransomware->Spyware            |       302 |                    210 |                  0.695364 |                            14 |                            30 |                            39 |                            69 |
| Ransomware<->Spyware | Spyware->Ransomware            |       144 |                     94 |                  0.652778 |                            14 |                            21 |                            30 |                            40 |
| Ransomware<->Spyware | BIDIRECTIONAL_DIRECT_CONFUSION |       446 |                    304 |                  0.681614 |                            28 |                            51 |                            69 |                           109 |
| Ransomware<->Trojan  | Ransomware->Trojan             |       301 |                    244 |                  0.810631 |                            44 |                            80 |                           100 |                           141 |
| Ransomware<->Trojan  | Trojan->Ransomware             |       295 |                    256 |                  0.867797 |                            47 |                            77 |                           108 |                           143 |
| Ransomware<->Trojan  | BIDIRECTIONAL_DIRECT_CONFUSION |       596 |                    500 |                  0.838926 |                            91 |                           157 |                           208 |                           284 |
| Spyware<->Trojan     | Spyware->Trojan                |       217 |                    163 |                  0.751152 |                            16 |                            47 |                            61 |                            84 |
| Spyware<->Trojan     | Trojan->Spyware                |       213 |                    136 |                  0.638498 |                            15 |                            37 |                            46 |                            70 |
| Spyware<->Trojan     | BIDIRECTIONAL_DIRECT_CONFUSION |       430 |                    299 |                  0.695349 |                            31 |                            84 |                           107 |                           154 |

## How to read this

- `oracle_all_wrong_true_in_top2` is a theoretical upper bound: it assumes a perfect mechanism fixes every wrong sample whose true class is already in top-2.
- `oracle_wrong_top2_margin_le_T` is a margin-bounded upper bound: it assumes perfect correction only when the probability gap is at most `T`.
- If full top-2 oracle is high but small-margin oracle is low, a simple confidence/margin rule is probably too weak.
- This is still diagnostic only; it does not prove a real reranker will achieve these numbers.

## Generated files

- `/home/pak/Documents/src_baocao/05_test/outputs/C2_top2_oracle_upper_bound/C2_summary.md`
- `/home/pak/Documents/src_baocao/05_test/outputs/C2_top2_oracle_upper_bound/C2_policy_metrics.csv`
- `/home/pak/Documents/src_baocao/05_test/outputs/C2_top2_oracle_upper_bound/C2_policy_per_class_f1.csv`
- `/home/pak/Documents/src_baocao/05_test/outputs/C2_top2_oracle_upper_bound/C2_margin_threshold_by_pair.csv`
- `/home/pak/Documents/src_baocao/05_test/outputs/C2_top2_oracle_upper_bound/C2_gate_decision.json`
- `/home/pak/Documents/src_baocao/05_test/outputs/C2_top2_oracle_upper_bound/C2_run_manifest.json`
- `/home/pak/Documents/src_baocao/05_test/outputs/C2_top2_oracle_upper_bound/C2_corrected_sample_indices.json`
