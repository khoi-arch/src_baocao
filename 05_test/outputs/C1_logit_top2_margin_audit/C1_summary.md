# C1 — Logit/top-2 margin audit

## Purpose

Audit whether wrong top-2 samples are close-margin decisions or high-confidence wrong decisions.

## Main metrics

- `n_total`: 11720
- `n_correct`: 10242
- `n_wrong`: 1478
- `accuracy_from_predictions`: 0.873890785
- `top2_accuracy`: 0.968003413
- `wrong_true_in_top2`: 1103
- `wrong_true_in_top2_rate`: 0.7462787551
- `wrong_top2_pred_minus_true_median`: 0.5041737407
- `wrong_top2_pred_minus_true_q75`: 0.8927814923
- `wrong_top2_pred_minus_true_rate_le_0.1`: 0.1359927471
- `wrong_top2_pred_minus_true_rate_le_0.2`: 0.2647325476
- `label_names`: ['Spyware', 'Benign', 'Trojan', 'Ransomware']
- `probability_columns`: {'Spyware': 'prob_Spyware', 'Benign': 'prob_Benign', 'Trojan': 'prob_Trojan', 'Ransomware': 'prob_Ransomware'}
- `hard_pairs`: ['Ransomware<->Spyware', 'Ransomware<->Trojan', 'Spyware<->Trojan']

## Interpretation gate

- Result: **MIXED — true label is often top-2 but margins may be large**
- Reason: wrong_true_in_top2_rate=0.7463, but median margin among wrong-top2=0.5042. A simple rerank may be hard; Phase C should first test diagnostic upper/lower-bound rerank.

## Overall margin distribution

| subset                 |     n |   top12_margin_mean |   top12_margin_std |   top12_margin_min |   top12_margin_q05 |   top12_margin_q10 |   top12_margin_q25 |   top12_margin_median |   top12_margin_q75 |   top12_margin_q90 |   top12_margin_q95 |   top12_margin_max |   top12_margin_rate_le_0.01 |   top12_margin_rate_le_0.02 |   top12_margin_rate_le_0.05 |   top12_margin_rate_le_0.1 |   top12_margin_rate_le_0.2 |   top12_margin_rate_le_0.3 |   top12_margin_rate_le_0.5 |   pred_minus_true_mean |   pred_minus_true_std |   pred_minus_true_min |   pred_minus_true_q05 |   pred_minus_true_q10 |   pred_minus_true_q25 |   pred_minus_true_median |   pred_minus_true_q75 |   pred_minus_true_q90 |   pred_minus_true_q95 |   pred_minus_true_max |   pred_minus_true_rate_le_0.01 |   pred_minus_true_rate_le_0.02 |   pred_minus_true_rate_le_0.05 |   pred_minus_true_rate_le_0.1 |   pred_minus_true_rate_le_0.2 |   pred_minus_true_rate_le_0.3 |   pred_minus_true_rate_le_0.5 |
|:-----------------------|------:|--------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|----------------------:|-------------------:|-------------------:|-------------------:|-------------------:|----------------------------:|----------------------------:|----------------------------:|---------------------------:|---------------------------:|---------------------------:|---------------------------:|-----------------------:|----------------------:|----------------------:|----------------------:|----------------------:|----------------------:|-------------------------:|----------------------:|----------------------:|----------------------:|----------------------:|-------------------------------:|-------------------------------:|-------------------------------:|------------------------------:|------------------------------:|------------------------------:|------------------------------:|
| all_samples            | 11720 |            0.865979 |           0.264287 |        0.000130445 |          0.16325   |          0.384788  |           0.905582 |              0.999955 |           1        |           1        |           1        |           1        |                  0.00230375 |                  0.00537543 |                  0.0144198  |                  0.0288396 |                  0.0604096 |                  0.0820819 |                  0.124147  |             nan        |            nan        |         nan           |           nan         |           nan         |            nan        |               nan        |            nan        |            nan        |            nan        |            nan        |                   nan          |                    nan         |                    nan         |                    nan        |                    nan        |                    nan        |                     nan       |
| correct_samples        | 10242 |            0.91258  |           0.214178 |        0.0014497   |          0.317466  |          0.640793  |           0.983806 |              0.999999 |           1        |           1        |           1        |           1        |                  0.00146456 |                  0.00312439 |                  0.00781097 |                  0.0151338 |                  0.0338801 |                  0.0478422 |                  0.0749854 |             nan        |            nan        |         nan           |           nan         |           nan         |            nan        |               nan        |            nan        |            nan        |            nan        |            nan        |                   nan          |                    nan         |                    nan         |                    nan        |                    nan        |                    nan        |                     nan       |
| wrong_samples          |  1478 |            0.543053 |           0.341555 |        0.000130445 |          0.0417435 |          0.0786212 |           0.205246 |              0.538952 |           0.899995 |           0.988218 |           0.996213 |           0.999998 |                  0.00811908 |                  0.0209743  |                  0.0602165  |                  0.123816  |                  0.244249  |                  0.31935   |                  0.464817  |             nan        |            nan        |         nan           |           nan         |           nan         |            nan        |               nan        |            nan        |            nan        |            nan        |            nan        |                   nan          |                    nan         |                    nan         |                    nan        |                    nan        |                    nan        |                     nan       |
| wrong_true_in_top2     |  1103 |            0.524947 |           0.343901 |        0.000130445 |          0.0388294 |          0.0741565 |           0.187553 |              0.504174 |           0.892781 |           0.984084 |           0.995265 |           0.999936 |                  0.00815956 |                  0.0208522  |                  0.0625567  |                  0.135993  |                  0.264733  |                  0.348141  |                  0.49592   |               0.524947 |              0.343901 |           0.000130445 |             0.0388294 |             0.0741565 |              0.187553 |                 0.504174 |              0.892781 |              0.984084 |              0.995265 |              0.999936 |                     0.00815956 |                      0.0208522 |                      0.0625567 |                      0.135993 |                      0.264733 |                      0.348141 |                       0.49592 |
| wrong_true_not_in_top2 |   375 |            0.59631  |           0.32883  |        0.0021221   |          0.0467674 |          0.113516  |           0.321577 |              0.644812 |           0.917724 |           0.994203 |           0.998123 |           0.999998 |                  0.008      |                  0.0213333  |                  0.0533333  |                  0.088     |                  0.184     |                  0.234667  |                  0.373333  |             nan        |            nan        |         nan           |           nan         |           nan         |            nan        |               nan        |            nan        |            nan        |            nan        |            nan        |                   nan          |                    nan         |                    nan         |                    nan        |                    nan        |                    nan        |                     nan       |

## Wrong top-2 margin by true class

| true_label   |    n |   n_wrong |   n_wrong_true_in_top2 |   wrong_true_in_top2_rate_among_wrong |   wrong_top2_pred_minus_true_median |   wrong_top2_pred_minus_true_q75 |   wrong_top2_pred_minus_true_rate_le_0.1 |   wrong_top2_pred_minus_true_rate_le_0.2 |
|:-------------|-----:|----------:|-----------------------:|--------------------------------------:|------------------------------------:|---------------------------------:|-----------------------------------------:|-----------------------------------------:|
| Ransomware   | 1959 |       603 |                    454 |                              0.752902 |                            0.553913 |                         0.915057 |                                 0.127753 |                                 0.242291 |
| Trojan       | 1897 |       508 |                    392 |                              0.771654 |                            0.424365 |                         0.804966 |                                 0.158163 |                                 0.290816 |
| Spyware      | 2004 |       363 |                    257 |                              0.707989 |                            0.541818 |                         0.915564 |                                 0.116732 |                                 0.264591 |
| Benign       | 5860 |         4 |                      0 |                              0        |                          nan        |                       nan        |                               nan        |                               nan        |

## Wrong top-2 margin by confusion pair

| true_label   | pred_label   |   n |   n_wrong_true_in_top2 |   wrong_top2_pred_minus_true_median |   wrong_top2_pred_minus_true_q75 |   wrong_top2_pred_minus_true_rate_le_0.1 |   wrong_top2_pred_minus_true_rate_le_0.2 |
|:-------------|:-------------|----:|-----------------------:|------------------------------------:|---------------------------------:|-----------------------------------------:|-----------------------------------------:|
| Ransomware   | Spyware      | 302 |                    210 |                            0.759072 |                         0.92706  |                                0.0666667 |                                 0.142857 |
| Ransomware   | Trojan       | 301 |                    244 |                            0.388667 |                         0.858569 |                                0.180328  |                                 0.327869 |
| Trojan       | Ransomware   | 295 |                    256 |                            0.403533 |                         0.811208 |                                0.183594  |                                 0.300781 |
| Spyware      | Trojan       | 217 |                    163 |                            0.451292 |                         0.915005 |                                0.0981595 |                                 0.288344 |
| Trojan       | Spyware      | 213 |                    136 |                            0.482308 |                         0.790956 |                                0.110294  |                                 0.272059 |
| Spyware      | Ransomware   | 144 |                     94 |                            0.610226 |                         0.916886 |                                0.148936  |                                 0.223404 |
| Benign       | Ransomware   |   3 |                      0 |                          nan        |                       nan        |                              nan         |                               nan        |
| Spyware      | Benign       |   2 |                      0 |                          nan        |                       nan        |                              nan         |                               nan        |
| Benign       | Spyware      |   1 |                      0 |                          nan        |                       nan        |                              nan         |                               nan        |

## Hard malware pair margin summary

| pair                 | direction                      |   n_wrong |   n_wrong_true_in_top2 |   wrong_true_in_top2_rate |   pred_minus_true_median |   pred_minus_true_q75 |   pred_minus_true_rate_le_0.1 |   pred_minus_true_rate_le_0.2 |
|:---------------------|:-------------------------------|----------:|-----------------------:|--------------------------:|-------------------------:|----------------------:|------------------------------:|------------------------------:|
| Ransomware<->Spyware | Ransomware->Spyware            |       302 |                    210 |                  0.695364 |                 0.759072 |              0.92706  |                     0.0666667 |                      0.142857 |
| Ransomware<->Spyware | Spyware->Ransomware            |       144 |                     94 |                  0.652778 |                 0.610226 |              0.916886 |                     0.148936  |                      0.223404 |
| Ransomware<->Spyware | BIDIRECTIONAL_DIRECT_CONFUSION |       446 |                    304 |                  0.681614 |                 0.711105 |              0.92543  |                     0.0921053 |                      0.167763 |
| Ransomware<->Trojan  | Ransomware->Trojan             |       301 |                    244 |                  0.810631 |                 0.388667 |              0.858569 |                     0.180328  |                      0.327869 |
| Ransomware<->Trojan  | Trojan->Ransomware             |       295 |                    256 |                  0.867797 |                 0.403533 |              0.811208 |                     0.183594  |                      0.300781 |
| Ransomware<->Trojan  | BIDIRECTIONAL_DIRECT_CONFUSION |       596 |                    500 |                  0.838926 |                 0.392799 |              0.833481 |                     0.182     |                      0.314    |
| Spyware<->Trojan     | Spyware->Trojan                |       217 |                    163 |                  0.751152 |                 0.451292 |              0.915005 |                     0.0981595 |                      0.288344 |
| Spyware<->Trojan     | Trojan->Spyware                |       213 |                    136 |                  0.638498 |                 0.482308 |              0.790956 |                     0.110294  |                      0.272059 |
| Spyware<->Trojan     | BIDIRECTIONAL_DIRECT_CONFUSION |       430 |                    299 |                  0.695349 |                 0.47991  |              0.863706 |                     0.103679  |                      0.280936 |

## How to read the margin

- `pred_minus_true_prob = probability(predicted class) - probability(true class)`.
- For wrong samples where the true class is top-2, this is the probability gap that a reranker would need to overcome.
- Small margin means the model is uncertain between the wrong top-1 and the true top-2.
- Large margin means the model is confidently wrong, so a simple rerank is less likely to be enough.

## Generated files

- `/home/pak/Documents/src_baocao/05_test/outputs/C1_logit_top2_margin_audit/C1_summary.md`
- `/home/pak/Documents/src_baocao/05_test/outputs/C1_logit_top2_margin_audit/C1_metrics.json`
- `/home/pak/Documents/src_baocao/05_test/outputs/C1_logit_top2_margin_audit/C1_margin_overall.csv`
- `/home/pak/Documents/src_baocao/05_test/outputs/C1_logit_top2_margin_audit/C1_wrong_top2_by_true_class.csv`
- `/home/pak/Documents/src_baocao/05_test/outputs/C1_logit_top2_margin_audit/C1_wrong_top2_by_confusion_pair.csv`
- `/home/pak/Documents/src_baocao/05_test/outputs/C1_logit_top2_margin_audit/C1_hard_pair_margin_summary.csv`
- `/home/pak/Documents/src_baocao/05_test/outputs/C1_logit_top2_margin_audit/C1_wrong_top2_samples.csv`
- `/home/pak/Documents/src_baocao/05_test/outputs/C1_logit_top2_margin_audit/C1_gate_decision.json`
