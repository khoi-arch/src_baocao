# B1 — CLS pairwise signal audit

## Purpose

Check whether the fresh CLS embedding space from the official C2+D3 model still contains pairwise signal for hard malware subtype boundaries.

## Main metrics

- `n_total`: 11720
- `cls_dim`: 128
- `n_correct`: 10242
- `n_wrong`: 1478
- `accuracy_from_cls_export`: 0.873890785
- `top2_accuracy_from_cls_export`: 0.968003413
- `wrong_true_in_top2`: 1103
- `wrong_true_in_top2_rate`: 0.7462787551
- `pairs`: ['Ransomware<->Spyware', 'Ransomware<->Trojan', 'Spyware<->Trojan']

## Interpretation gate

- Result: **PASS — CLS has usable pairwise signal**
- Reason: Mean pairwise LogisticRegression macro-F1=0.8359, mean AUC=0.9128, and mean wrong-direction true-in-top2=0.7360. This supports testing reranking or auxiliary pairwise heads.

## Official model behavior inside each hard pair

| pair                 | class_a    | class_b   |   n_true_pair |   official_correct_n |   official_correct_rate |   official_pair_confusion_n |   official_pair_confusion_rate |   official_pred_outside_pair_n |   official_pred_outside_pair_rate |   true_in_top2_n |   true_in_top2_rate |
|:---------------------|:-----------|:----------|--------------:|---------------------:|------------------------:|----------------------------:|-------------------------------:|-------------------------------:|----------------------------------:|-----------------:|--------------------:|
| Ransomware<->Spyware | Ransomware | Spyware   |          3963 |                 2997 |                0.756245 |                         446 |                       0.112541 |                            520 |                          0.131214 |             3708 |            0.935655 |
| Ransomware<->Trojan  | Ransomware | Trojan    |          3856 |                 2745 |                0.711878 |                         596 |                       0.154564 |                            515 |                          0.133558 |             3591 |            0.931276 |
| Spyware<->Trojan     | Spyware    | Trojan    |          3901 |                 3030 |                0.776724 |                         430 |                       0.110228 |                            441 |                          0.113048 |             3679 |            0.943092 |

## Pairwise linear separability in CLS space

| pair                 |   n_samples |   class0_count |   class1_count |   cv_folds |   accuracy |   balanced_accuracy |   macro_f1 |      auc | cv_status   |
|:---------------------|------------:|---------------:|---------------:|-----------:|-----------:|--------------------:|-----------:|---------:|:------------|
| Ransomware<->Spyware |        3963 |           1959 |           2004 |          5 |   0.849861 |            0.849564 |   0.849668 | 0.919618 | ok          |
| Ransomware<->Trojan  |        3856 |           1959 |           1897 |          5 |   0.798755 |            0.798461 |   0.79856  | 0.883855 | ok          |
| Spyware<->Trojan     |        3901 |           2004 |           1897 |          5 |   0.859523 |            0.859544 |   0.85945  | 0.935063 | ok          |

## Correct-sample centroid distances

| pair                 | class_a    | class_b   | status   |   centroid_distance_euclidean |   centroid_distance_cosine |
|:---------------------|:-----------|:----------|:---------|------------------------------:|---------------------------:|
| Ransomware<->Spyware | Ransomware | Spyware   | ok       |                       12.4273 |                    1.23104 |
| Ransomware<->Trojan  | Ransomware | Trojan    | ok       |                       11.1514 |                    1.01845 |
| Spyware<->Trojan     | Spyware    | Trojan    | ok       |                       11.4822 |                    1.07096 |

## Wrong-direction centroid behavior

| pair                 | direction           |   n_wrong |   wrong_true_in_top2_rate |   true_closer_than_pred_rate |   mean_distance_margin_pred_minus_true |   median_distance_margin_pred_minus_true | status   |
|:---------------------|:--------------------|----------:|--------------------------:|-----------------------------:|---------------------------------------:|-----------------------------------------:|:---------|
| Ransomware<->Spyware | Ransomware->Spyware |       302 |                  0.695364 |                    0.0298013 |                               -4.57481 |                                 -4.434   | ok       |
| Ransomware<->Spyware | Spyware->Ransomware |       144 |                  0.652778 |                    0.0833333 |                               -3.16346 |                                 -2.88765 | ok       |
| Ransomware<->Trojan  | Ransomware->Trojan  |       301 |                  0.810631 |                    0.282392  |                               -1.88549 |                                 -1.43274 | ok       |
| Ransomware<->Trojan  | Trojan->Ransomware  |       295 |                  0.867797 |                    0.0440678 |                               -3.75234 |                                 -3.49637 | ok       |
| Spyware<->Trojan     | Spyware->Trojan     |       217 |                  0.751152 |                    0.110599  |                               -2.90337 |                                 -2.66863 | ok       |
| Spyware<->Trojan     | Trojan->Spyware     |       213 |                  0.638498 |                    0.0516432 |                               -3.4226  |                                 -3.25354 | ok       |

## How to read centroid margin

- `mean_distance_margin_pred_minus_true = dist_to_pred_centroid - dist_to_true_centroid`.
- Positive value: wrong samples are closer to their true class centroid than predicted class centroid.
- Negative value: wrong samples are closer to the predicted class centroid, suggesting representation-level pull toward the wrong class.

## Generated files

- `/home/pak/Documents/src_baocao/05_test/outputs/B1_cls_pairwise_signal/B1_summary.md`
- `/home/pak/Documents/src_baocao/05_test/outputs/B1_cls_pairwise_signal/B1_metrics.json`
- `/home/pak/Documents/src_baocao/05_test/outputs/B1_cls_pairwise_signal/B1_pairwise_logreg_cv_metrics.csv`
- `/home/pak/Documents/src_baocao/05_test/outputs/B1_cls_pairwise_signal/B1_centroid_distance.csv`
- `/home/pak/Documents/src_baocao/05_test/outputs/B1_cls_pairwise_signal/B1_centroid_sources.csv`
- `/home/pak/Documents/src_baocao/05_test/outputs/B1_cls_pairwise_signal/B1_wrong_direction_centroid_behavior.csv`
- `/home/pak/Documents/src_baocao/05_test/outputs/B1_cls_pairwise_signal/B1_official_pair_behavior.csv`
- `/home/pak/Documents/src_baocao/05_test/outputs/B1_cls_pairwise_signal/B1_hard_pair_summary.csv`
- `/home/pak/Documents/src_baocao/05_test/outputs/B1_cls_pairwise_signal/B1_gate_decision.json`
