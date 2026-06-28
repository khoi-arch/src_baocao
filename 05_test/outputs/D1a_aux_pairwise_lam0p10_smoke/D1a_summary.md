# D1a — Auxiliary pairwise head training

## Guardrail

- This is an isolated D1a test under `05_test`.
- It does not modify `02_src`.
- It does not modify `03_outputs/06_model`.
- Inference uses only the main 4-class logits; auxiliary pair heads are training losses, not post-hoc rerankers.

## Config

- `smoke_test`: True
- `epochs`: 1
- `batch_size`: 128
- `lr`: 0.001
- `weight_decay`: 0.0001
- `lambda_pair`: 0.1
- `focal_gamma`: 2.0
- `label_smoothing`: 0.0
- `device`: cpu
- `max_train_rows`: 2048
- `max_val_rows`: 1024
- `capture_module_name`: base.classifier

## Best validation metrics

- `best_epoch`: 1
- `accuracy`: 0.1533203125
- `macro_f1`: 0.06646909398814564
- `weighted_f1`: 0.04076424904741745
- `elapsed_seconds`: 6.298909425735474

## Baseline-vs-D1a transition

- `available`: True
- `n`: 1024
- `baseline_correct`: 902
- `new_correct`: 157
- `wrong_to_correct_n`: 43
- `correct_to_wrong_n`: 788
- `wrong_to_wrong_n`: 79
- `correct_to_correct_n`: 114
- `net_gain_n`: -745
- `damage_ratio`: 18.325581395348838

## Pair-level fix/damage

| pair                 | direction           |   pair_true_n |   baseline_direct_wrong_n |   fixed_direct_wrong_n |   fix_rate_among_direct_wrong |   baseline_correct_pair_n |   correct_to_wrong_damage_n |   damage_rate_among_pair_correct |   net_pair_gain_n |
|:---------------------|:--------------------|--------------:|--------------------------:|-----------------------:|------------------------------:|--------------------------:|----------------------------:|---------------------------------:|------------------:|
| Ransomware<->Spyware | BIDIRECTIONAL       |           354 |                        37 |                      0 |                      0        |                       275 |                         275 |                          1       |              -275 |
| Ransomware<->Spyware | Ransomware->Spyware |           177 |                        28 |                      0 |                      0        |                       126 |                         126 |                          1       |              -126 |
| Ransomware<->Spyware | Spyware->Ransomware |           177 |                         9 |                      0 |                      0        |                       149 |                         149 |                          1       |              -149 |
| Ransomware<->Trojan  | BIDIRECTIONAL       |           334 |                        48 |                     25 |                      0.520833 |                       240 |                         126 |                          0.525   |              -101 |
| Ransomware<->Trojan  | Ransomware->Trojan  |           177 |                        23 |                      0 |                      0        |                       126 |                         126 |                          1       |              -126 |
| Ransomware<->Trojan  | Trojan->Ransomware  |           157 |                        25 |                     25 |                      1        |                       114 |                           0 |                          0       |                25 |
| Spyware<->Trojan     | BIDIRECTIONAL       |           334 |                        37 |                     18 |                      0.486486 |                       263 |                         149 |                          0.56654 |              -131 |
| Spyware<->Trojan     | Spyware->Trojan     |           177 |                        19 |                      0 |                      0        |                       149 |                         149 |                          1       |              -149 |
| Spyware<->Trojan     | Trojan->Spyware     |           157 |                        18 |                     18 |                      1        |                       114 |                           0 |                          0       |                18 |

## Auxiliary pair head metrics at best epoch

| pair_key                |   n |   accuracy |   macro_f1 |      auc |
|:------------------------|----:|-----------:|-----------:|---------:|
| Ransomware__vs__Spyware | 354 |   0.5      |   0.333333 | 0.543777 |
| Ransomware__vs__Trojan  | 334 |   0.511976 |   0.354425 | 0.541293 |
| Spyware__vs__Trojan     | 334 |   0.523952 |   0.509672 | 0.607219 |

## How to judge

- Good D1a must show `wrong_to_correct_n > correct_to_wrong_n`, clear positive net gain, and at least 2/3 hard pairs with positive `net_pair_gain_n`.
- If macro-F1 improves but correct-to-wrong damage is high, this direction is not safe.
- Smoke-test results only verify code path; full conclusion requires Kaggle/full run.

## Generated files

- `/home/pak/Documents/src_baocao/05_test/outputs/D1a_aux_pairwise_lam0p10_smoke/D1a_summary.md`
- `/home/pak/Documents/src_baocao/05_test/outputs/D1a_aux_pairwise_lam0p10_smoke/D1a_history.csv`
- `/home/pak/Documents/src_baocao/05_test/outputs/D1a_aux_pairwise_lam0p10_smoke/D1a_val_classification_report_best.json`
- `/home/pak/Documents/src_baocao/05_test/outputs/D1a_aux_pairwise_lam0p10_smoke/D1a_val_confusion_matrix_best.csv`
- `/home/pak/Documents/src_baocao/05_test/outputs/D1a_aux_pairwise_lam0p10_smoke/D1a_val_predictions_best.csv`
- `/home/pak/Documents/src_baocao/05_test/outputs/D1a_aux_pairwise_lam0p10_smoke/D1a_aux_pair_metrics_best.csv`
- `/home/pak/Documents/src_baocao/05_test/outputs/D1a_aux_pairwise_lam0p10_smoke/D1a_transition_summary.json`
- `/home/pak/Documents/src_baocao/05_test/outputs/D1a_aux_pairwise_lam0p10_smoke/D1a_pair_fix_damage_summary.csv`
- `/home/pak/Documents/src_baocao/05_test/outputs/D1a_aux_pairwise_lam0p10_smoke/D1a_config.json`
- `/home/pak/Documents/src_baocao/05_test/outputs/D1a_aux_pairwise_lam0p10_smoke/best_model.pt`
