# B2 — Raw/token/offset pairwise signal audit

## Purpose

Check whether pairwise malware-subtype signal exists in input spaces before the Transformer CLS representation.

## Main metrics

- `n_total`: 11720
- `n_features`: 55
- `num_bins`: 512
- `label_names`: ['Benign', 'Ransomware', 'Spyware', 'Trojan']
- `pairs`: ['Ransomware<->Spyware', 'Ransomware<->Trojan', 'Spyware<->Trojan']
- `representations_tested`: ['raw_scaled', 'bin_norm', 'offset', 'bin_plus_offset_norm', 'bin_norm__offset', 'raw_scaled__bin_plus_offset_norm', 'raw_scaled__bin_norm__offset', 'd3_scalar_input']
- `raw_scaled_source`: raw_scaled_from_raw_csv
- `cv_folds`: 5
- `random_state`: 42

## Interpretation gate

- Result: **FAIL — input-space pairwise signal appears weak**
- Reason: Best representation `raw_scaled__bin_norm__offset` has mean macro-F1=0.6695, mean AUC=0.7425. This suggests the subtype overlap is already severe before CLS.

## Official model behavior inside each hard pair

| pair                 | status   |   n_true_pair |   official_correct_n |   official_correct_rate |   official_pair_confusion_n |   official_pair_confusion_rate |   official_pred_outside_pair_n |   official_pred_outside_pair_rate |   true_in_top2_n |   true_in_top2_rate |
|:---------------------|:---------|--------------:|---------------------:|------------------------:|----------------------------:|-------------------------------:|-------------------------------:|----------------------------------:|-----------------:|--------------------:|
| Ransomware<->Spyware | ok       |          3963 |                 2997 |                0.756245 |                         446 |                       0.112541 |                            520 |                          0.131214 |             3708 |            0.935655 |
| Ransomware<->Trojan  | ok       |          3856 |                 2745 |                0.711878 |                         596 |                       0.154564 |                            515 |                          0.133558 |             3591 |            0.931276 |
| Spyware<->Trojan     | ok       |          3901 |                 3030 |                0.776724 |                         430 |                       0.110228 |                            441 |                          0.113048 |             3679 |            0.943092 |

## Representation summary

| representation                   |   dim |   mean_macro_f1 |   mean_auc |   min_macro_f1 |   max_macro_f1 |   mean_balanced_accuracy |
|:---------------------------------|------:|----------------:|-----------:|---------------:|---------------:|-------------------------:|
| raw_scaled__bin_norm__offset     |   165 |        0.669473 |   0.742474 |       0.662262 |       0.681834 |                 0.671225 |
| d3_scalar_input                  |   220 |        0.66938  |   0.742411 |       0.661742 |       0.681817 |                 0.671138 |
| raw_scaled__bin_plus_offset_norm |   110 |        0.658881 |   0.727004 |       0.647654 |       0.671196 |                 0.662298 |
| bin_norm__offset                 |   110 |        0.658797 |   0.724863 |       0.647483 |       0.665286 |                 0.660405 |
| bin_plus_offset_norm             |    55 |        0.645083 |   0.706959 |       0.634528 |       0.661023 |                 0.647966 |
| bin_norm                         |    55 |        0.644035 |   0.70705  |       0.635271 |       0.659385 |                 0.646969 |
| raw_scaled                       |    55 |        0.630029 |   0.692109 |       0.606536 |       0.659939 |                 0.635629 |
| offset                           |    55 |        0.588025 |   0.622467 |       0.571258 |       0.612355 |                 0.589301 |

## Best input representation per pair

| pair                 | best_representation          |   dim |   best_macro_f1 |   best_auc |   best_balanced_accuracy |   n_samples |
|:---------------------|:-----------------------------|------:|----------------:|-----------:|-------------------------:|------------:|
| Ransomware<->Spyware | d3_scalar_input              |   220 |        0.664581 |   0.728156 |                 0.664894 |        3963 |
| Ransomware<->Trojan  | raw_scaled__bin_norm__offset |   165 |        0.662262 |   0.732371 |                 0.664479 |        3856 |
| Spyware<->Trojan     | raw_scaled__bin_norm__offset |   165 |        0.681834 |   0.766964 |                 0.684546 |        3901 |

## Full pairwise metrics

| representation                   | pair                 |   dim |   n_samples |   class0_count |   class1_count |   accuracy |   balanced_accuracy |   macro_f1 |      auc | cv_status   |
|:---------------------------------|:---------------------|------:|------------:|---------------:|---------------:|-----------:|--------------------:|-----------:|---------:|:------------|
| raw_scaled                       | Ransomware<->Spyware |    55 |        3963 |           1959 |           2004 |   0.624022 |            0.624478 |   0.623611 | 0.681289 | ok          |
| raw_scaled                       | Ransomware<->Trojan  |    55 |        3856 |           1959 |           1897 |   0.610218 |            0.611931 |   0.606536 | 0.675008 | ok          |
| raw_scaled                       | Spyware<->Trojan     |    55 |        3901 |           2004 |           1897 |   0.666239 |            0.670477 |   0.659939 | 0.720029 | ok          |
| bin_norm                         | Ransomware<->Spyware |    55 |        3963 |           1959 |           2004 |   0.637648 |            0.637997 |   0.637449 | 0.699783 | ok          |
| bin_norm                         | Ransomware<->Trojan  |    55 |        3856 |           1959 |           1897 |   0.636929 |            0.638179 |   0.635271 | 0.699162 | ok          |
| bin_norm                         | Spyware<->Trojan     |    55 |        3901 |           2004 |           1897 |   0.661882 |            0.66473  |   0.659385 | 0.722205 | ok          |
| offset                           | Ransomware<->Spyware |    55 |        3963 |           1959 |           2004 |   0.571537 |            0.571321 |   0.571258 | 0.590719 | ok          |
| offset                           | Ransomware<->Trojan  |    55 |        3856 |           1959 |           1897 |   0.58221  |            0.583399 |   0.58046  | 0.622177 | ok          |
| offset                           | Spyware<->Trojan     |    55 |        3901 |           2004 |           1897 |   0.612407 |            0.613185 |   0.612355 | 0.654504 | ok          |
| bin_plus_offset_norm             | Ransomware<->Spyware |    55 |        3963 |           1959 |           2004 |   0.639919 |            0.640283 |   0.639699 | 0.700119 | ok          |
| bin_plus_offset_norm             | Ransomware<->Trojan  |    55 |        3856 |           1959 |           1897 |   0.636151 |            0.637388 |   0.634528 | 0.698856 | ok          |
| bin_plus_offset_norm             | Spyware<->Trojan     |    55 |        3901 |           2004 |           1897 |   0.66342  |            0.666227 |   0.661023 | 0.721902 | ok          |
| bin_norm__offset                 | Ransomware<->Spyware |   110 |        3963 |           1959 |           2004 |   0.663639 |            0.663804 |   0.663622 | 0.717834 | ok          |
| bin_norm__offset                 | Ransomware<->Trojan  |   110 |        3856 |           1959 |           1897 |   0.6486   |            0.649672 |   0.647483 | 0.709976 | ok          |
| bin_norm__offset                 | Spyware<->Trojan     |   110 |        3901 |           2004 |           1897 |   0.665983 |            0.667737 |   0.665286 | 0.746779 | ok          |
| raw_scaled__bin_plus_offset_norm | Ransomware<->Spyware |   110 |        3963 |           1959 |           2004 |   0.648499 |            0.649138 |   0.647654 | 0.710328 | ok          |
| raw_scaled__bin_plus_offset_norm | Ransomware<->Trojan  |   110 |        3856 |           1959 |           1897 |   0.659232 |            0.660446 |   0.657794 | 0.727085 | ok          |
| raw_scaled__bin_plus_offset_norm | Spyware<->Trojan     |   110 |        3901 |           2004 |           1897 |   0.674186 |            0.677311 |   0.671196 | 0.743599 | ok          |
| raw_scaled__bin_norm__offset     | Ransomware<->Spyware |   165 |        3963 |           1959 |           2004 |   0.664396 |            0.66465  |   0.664321 | 0.728086 | ok          |
| raw_scaled__bin_norm__offset     | Ransomware<->Trojan  |   165 |        3856 |           1959 |           1897 |   0.663382 |            0.664479 |   0.662262 | 0.732371 | ok          |
| raw_scaled__bin_norm__offset     | Spyware<->Trojan     |   165 |        3901 |           2004 |           1897 |   0.682645 |            0.684546 |   0.681834 | 0.766964 | ok          |
| d3_scalar_input                  | Ransomware<->Spyware |   220 |        3963 |           1959 |           2004 |   0.664648 |            0.664894 |   0.664581 | 0.728156 | ok          |
| d3_scalar_input                  | Ransomware<->Trojan  |   220 |        3856 |           1959 |           1897 |   0.662863 |            0.66396  |   0.661742 | 0.732148 | ok          |
| d3_scalar_input                  | Spyware<->Trojan     |   220 |        3901 |           2004 |           1897 |   0.682645 |            0.68456  |   0.681817 | 0.766928 | ok          |

## Notes

- B2 is diagnostic only. It does not apply reranking or model changes.
- Strong input-space signal means useful class information still exists before the model representation.
- B3 should compare these input-space metrics against B1 CLS metrics before deciding where the bottleneck is.

## Generated files

- `/home/pak/Documents/src_baocao/05_test/outputs/B2_input_pairwise_signal/B2_summary.md`
- `/home/pak/Documents/src_baocao/05_test/outputs/B2_input_pairwise_signal/B2_metrics.json`
- `/home/pak/Documents/src_baocao/05_test/outputs/B2_input_pairwise_signal/B2_pairwise_signal_metrics.csv`
- `/home/pak/Documents/src_baocao/05_test/outputs/B2_input_pairwise_signal/B2_representation_summary.csv`
- `/home/pak/Documents/src_baocao/05_test/outputs/B2_input_pairwise_signal/B2_best_input_representation_by_pair.csv`
- `/home/pak/Documents/src_baocao/05_test/outputs/B2_input_pairwise_signal/B2_official_pair_behavior.csv`
- `/home/pak/Documents/src_baocao/05_test/outputs/B2_input_pairwise_signal/B2_gate_decision.json`
- `/home/pak/Documents/src_baocao/05_test/outputs/B2_input_pairwise_signal/B2_raw_scaled_info.json`
