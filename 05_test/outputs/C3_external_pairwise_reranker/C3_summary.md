# C3 — External pairwise reranker diagnostic

## Purpose

Test whether external pairwise CLS/log-probability reranking can fix hard subtype mistakes without damaging too many already-correct samples.

## Guardrail

C3 is a validation OOF diagnostic. It measures whether a pairwise correction direction is promising, but it is not an official final model and not a train/val/test-proven solution.

## Interpretation gate

- Result: **MIXED — reranker improves metrics but damage or gain size needs caution**
- Reason: Best policy `probs__rerank_top2_pair_conf_ge_0.5` has positive delta_macro_f1=0.0014, wrong_to_correct=34, correct_to_wrong=24, net_gain=10. The improvement exists but may not be stable enough for direct adoption.
- Recommendation: Inspect pair-level damage before deciding Phase D. Prefer safer constraints or a training-time auxiliary head.

## Best policy

- `best_policy`: probs__rerank_top2_pair_conf_ge_0.5
- `best_feature_set`: probs
- `best_confidence_threshold`: 0.5
- `best_accuracy`: 0.8747440273037542
- `best_macro_f1`: 0.8115292617320812
- `best_delta_macro_f1`: 0.001434980606937697
- `best_weighted_f1`: 0.8744455686679126
- `best_delta_weighted_f1`: 0.0009713693332468187
- `best_wrong_to_correct_n`: 34
- `best_correct_to_wrong_n`: 24
- `best_net_gain_n`: 10
- `best_damage_ratio`: 0.7058823529411765

## Top policies by macro-F1 delta

| policy                                      | feature_set   |   confidence_threshold |   accuracy |   delta_accuracy |   macro_f1 |   delta_macro_f1 |   weighted_f1 |   delta_weighted_f1 |   n_changed |   wrong_to_correct_n |   correct_to_wrong_n |   net_gain_n |   damage_ratio |
|:--------------------------------------------|:--------------|-----------------------:|-----------:|-----------------:|-----------:|-----------------:|--------------:|--------------------:|------------:|---------------------:|---------------------:|-------------:|---------------:|
| probs__rerank_top2_pair_conf_ge_0.5         | probs         |                   0.5  |   0.874744 |      0.000853242 |   0.811529 |      0.00143498  |      0.874446 |         0.000971369 |          58 |                   34 |                   24 |           10 |       0.705882 |
| probs__rerank_top2_pair_conf_ge_0.55        | probs         |                   0.55 |   0.874317 |      0.000426621 |   0.810779 |      0.00068468  |      0.873935 |         0.000460643 |           9 |                    7 |                    2 |            5 |       0.285714 |
| logprobs__rerank_top2_pair_conf_ge_0.5      | logprobs      |                   0.5  |   0.874488 |      0.00059727  |   0.810579 |      0.00048462  |      0.873873 |         0.000399088 |         225 |                  116 |                  109 |            7 |       0.939655 |
| cls__rerank_top2_pair_conf_ge_0.6           | cls           |                   0.6  |   0.874061 |      0.000170648 |   0.810517 |      0.000423198 |      0.873752 |         0.000278117 |         224 |                  113 |                  111 |            2 |       0.982301 |
| cls__logprobs__rerank_top2_pair_conf_ge_0.8 | cls__logprobs |                   0.8  |   0.874147 |      0.000255973 |   0.81045  |      0.000355846 |      0.873721 |         0.000246586 |          29 |                   16 |                   13 |            3 |       0.8125   |
| cls__logprobs__rerank_top2_pair_conf_ge_0.6 | cls__logprobs |                   0.6  |   0.873976 |      8.53242e-05 |   0.810362 |      0.000267487 |      0.873652 |         0.000177816 |         229 |                  115 |                  114 |            1 |       0.991304 |
| cls__probs__rerank_top2_pair_conf_ge_0.6    | cls__probs    |                   0.6  |   0.873891 |      0           |   0.810287 |      0.000192895 |      0.873595 |         0.000121187 |         226 |                  113 |                  113 |            0 |       1        |
| logprobs__rerank_top2_pair_conf_ge_0.6      | logprobs      |                   0.6  |   0.873976 |      8.53242e-05 |   0.810227 |      0.000133036 |      0.873562 |         8.79318e-05 |           1 |                    1 |                    0 |            1 |       0        |
| original                                    | none          |                 nan    |   0.873891 |      0           |   0.810094 |      0           |      0.873474 |         0           |           0 |                    0 |                    0 |            0 |     nan        |
| probs__rerank_top2_pair_conf_ge_0.6         | probs         |                   0.6  |   0.873891 |      0           |   0.810094 |      0           |      0.873474 |         0           |           0 |                    0 |                    0 |            0 |     nan        |
| probs__rerank_top2_pair_conf_ge_0.7         | probs         |                   0.7  |   0.873891 |      0           |   0.810094 |      0           |      0.873474 |         0           |           0 |                    0 |                    0 |            0 |     nan        |
| probs__rerank_top2_pair_conf_ge_0.8         | probs         |                   0.8  |   0.873891 |      0           |   0.810094 |      0           |      0.873474 |         0           |           0 |                    0 |                    0 |            0 |     nan        |

## Best policy per-class F1

| policy                              |   f1_Benign |   f1_Ransomware |   f1_Spyware |   f1_Trojan |   delta_f1_Benign |   delta_f1_Ransomware |   delta_f1_Spyware |   delta_f1_Trojan |
|:------------------------------------|------------:|----------------:|-------------:|------------:|------------------:|----------------------:|-------------------:|------------------:|
| original                            |    0.999488 |        0.721853 |     0.788753 |    0.730284 |                 0 |            0          |          0         |        0          |
| probs__rerank_top2_pair_conf_ge_0.5 |    0.999488 |        0.727989 |     0.789831 |    0.728809 |                 0 |            0.00613695 |          0.0010778 |       -0.00147483 |

## Best policy transition summary

| policy                              |   n_changed |   wrong_to_correct_n |   correct_to_wrong_n |   wrong_to_wrong_changed_n |   correct_to_correct_changed_n |   net_gain_n |   damage_ratio_correct_to_wrong_over_wrong_to_correct |   n_changed_detail_rows |
|:------------------------------------|------------:|---------------------:|---------------------:|---------------------------:|-------------------------------:|-------------:|------------------------------------------------------:|------------------------:|
| original                            |           0 |                    0 |                    0 |                          0 |                              0 |            0 |                                            nan        |                       0 |
| probs__rerank_top2_pair_conf_ge_0.5 |          58 |                   34 |                   24 |                          0 |                              0 |           10 |                                              0.705882 |                      58 |

## Best policy pair-level fix/damage summary

| pair                 | direction           |   candidate_true_pair_n |   original_direct_wrong_n |   fixed_direct_wrong_n |   fix_rate_among_direct_wrong |   original_correct_candidate_n |   correct_to_wrong_damage_n |   damage_rate_among_correct_candidates |   wrong_to_wrong_changed_n |   net_pair_gain_n |
|:---------------------|:--------------------|------------------------:|--------------------------:|-----------------------:|------------------------------:|-------------------------------:|----------------------------:|---------------------------------------:|---------------------------:|------------------:|
| Ransomware<->Spyware | BIDIRECTIONAL       |                    1666 |                       446 |                     22 |                    0.0493274  |                           1362 |                          11 |                             0.00807636 |                          0 |                11 |
| Ransomware<->Spyware | Ransomware->Spyware |                     565 |                       302 |                     22 |                    0.0728477  |                            355 |                           0 |                             0          |                          0 |                22 |
| Ransomware<->Spyware | Spyware->Ransomware |                    1101 |                       144 |                      0 |                    0          |                           1007 |                          11 |                             0.0109235  |                          0 |               -11 |
| Ransomware<->Trojan  | BIDIRECTIONAL       |                    2191 |                       596 |                      7 |                    0.011745   |                           1691 |                           8 |                             0.00473093 |                          0 |                -1 |
| Ransomware<->Trojan  | Ransomware->Trojan  |                    1245 |                       301 |                      7 |                    0.0232558  |                           1001 |                           0 |                             0          |                          0 |                 7 |
| Ransomware<->Trojan  | Trojan->Ransomware  |                     946 |                       295 |                      0 |                    0          |                            690 |                           8 |                             0.0115942  |                          0 |                -8 |
| Spyware<->Trojan     | BIDIRECTIONAL       |                    1605 |                       430 |                      5 |                    0.0116279  |                           1306 |                           5 |                             0.00382848 |                          0 |                 0 |
| Spyware<->Trojan     | Spyware->Trojan     |                     778 |                       217 |                      3 |                    0.0138249  |                            615 |                           2 |                             0.00325203 |                          0 |                 1 |
| Spyware<->Trojan     | Trojan->Spyware     |                     827 |                       213 |                      2 |                    0.00938967 |                            691 |                           3 |                             0.00434153 |                          0 |                -1 |

## Pairwise classifier OOF metrics

| feature_set   | pair                 |   dim |   accuracy |   balanced_accuracy |   macro_f1 |      auc |
|:--------------|:---------------------|------:|-----------:|--------------------:|-----------:|---------:|
| cls           | Ransomware<->Spyware |   128 |   0.849861 |            0.849564 |   0.849668 | 0.919618 |
| cls           | Ransomware<->Trojan  |   128 |   0.798755 |            0.798461 |   0.79856  | 0.883855 |
| cls           | Spyware<->Trojan     |   128 |   0.859523 |            0.859544 |   0.85945  | 0.935063 |
| cls__logprobs | Ransomware<->Spyware |   132 |   0.852385 |            0.852013 |   0.852119 | 0.918716 |
| cls__logprobs | Ransomware<->Trojan  |   132 |   0.801867 |            0.801516 |   0.801628 | 0.884731 |
| cls__logprobs | Spyware<->Trojan     |   132 |   0.861318 |            0.861305 |   0.861237 | 0.935547 |
| cls__probs    | Ransomware<->Spyware |   132 |   0.849861 |            0.849593 |   0.849694 | 0.919972 |
| cls__probs    | Ransomware<->Trojan  |   132 |   0.800052 |            0.799746 |   0.799848 | 0.883212 |
| cls__probs    | Spyware<->Trojan     |   132 |   0.861318 |            0.861361 |   0.86125  | 0.934868 |
| logprobs      | Ransomware<->Spyware |     4 |   0.859198 |            0.858704 |   0.858797 | 0.923554 |
| logprobs      | Ransomware<->Trojan  |     4 |   0.803423 |            0.802772 |   0.802861 | 0.889744 |
| logprobs      | Spyware<->Trojan     |     4 |   0.866701 |            0.866418 |   0.866546 | 0.937318 |
| probs         | Ransomware<->Spyware |     4 |   0.851375 |            0.851296 |   0.851334 | 0.910131 |
| probs         | Ransomware<->Trojan  |     4 |   0.799533 |            0.798927 |   0.799019 | 0.876654 |
| probs         | Spyware<->Trojan     |     4 |   0.858498 |            0.859081 |   0.858497 | 0.925782 |

## How to read this

- `wrong_to_correct_n`: original baseline was wrong, reranker makes it correct.
- `correct_to_wrong_n`: original baseline was correct, reranker breaks it.
- `net_gain_n = wrong_to_correct_n - correct_to_wrong_n`.
- A good direction must improve hard-pair errors without a large correct-to-wrong cost.
- Pair-level rows show whether each hard pair improves and whether already-correct samples in that pair are damaged.

## CLS input info

- `{'path': '/home/pak/Documents/src_baocao/05_test/outputs/B1_cls_pairwise_signal/val_cls_embeddings.npz', 'keys': ['cls_embeddings', 'logits', 'probs', 'y_true', 'y_pred', 'top1_id', 'top2_id', 'top1_score', 'top2_score', 'sample_index', 'label_names'], 'cls_key': 'cls_embeddings', 'cls_shape': [11720, 128], 'label_names': ['Benign', 'Ransomware', 'Spyware', 'Trojan']}`

## Generated files

- `/home/pak/Documents/src_baocao/05_test/outputs/C3_external_pairwise_reranker/C3_summary.md`
- `/home/pak/Documents/src_baocao/05_test/outputs/C3_external_pairwise_reranker/C3_policy_metrics.csv`
- `/home/pak/Documents/src_baocao/05_test/outputs/C3_external_pairwise_reranker/C3_policy_per_class_f1.csv`
- `/home/pak/Documents/src_baocao/05_test/outputs/C3_external_pairwise_reranker/C3_pairwise_cv_metrics.csv`
- `/home/pak/Documents/src_baocao/05_test/outputs/C3_external_pairwise_reranker/C3_transition_summary.csv`
- `/home/pak/Documents/src_baocao/05_test/outputs/C3_external_pairwise_reranker/C3_pair_fix_damage_summary.csv`
- `/home/pak/Documents/src_baocao/05_test/outputs/C3_external_pairwise_reranker/C3_gate_decision.json`
- `/home/pak/Documents/src_baocao/05_test/outputs/C3_external_pairwise_reranker/C3_run_manifest.json`
- `/home/pak/Documents/src_baocao/05_test/outputs/C3_external_pairwise_reranker/C3_best_changed_samples.csv`
- `/home/pak/Documents/src_baocao/05_test/outputs/C3_external_pairwise_reranker/C3_original_confusion_matrix.csv`
- `/home/pak/Documents/src_baocao/05_test/outputs/C3_external_pairwise_reranker/C3_best_confusion_matrix.csv`
