# F1e1a_v3 L1 L3/Family + Top2/Probability Audit Report

## Purpose

```text
No training.
Audit L1 behavior at L3/family level with full probability vector and top2 information.
This is meant to diagnose whether family-aware smoothing is justified.
```

## Leakage answer

```text
If validation family/probability audit is used to choose smoothing hyperparameters,
then evaluating/reporting on that same validation set is validation leakage / tuning-to-val.

Use val audit for diagnosis/hypothesis.
For clean hyperparameter choice, derive matrix from train OOF or an internal calibration split.
```

## Loaded data/model

```json
{
  "dataset": {
    "dataset_npz": "/kaggle/working/src_baocao/03_outputs/05_dataset/dataset.npz",
    "train_raw": "/kaggle/working/src_baocao/01_split/train_raw.csv",
    "val_raw": "/kaggle/working/src_baocao/01_split/val_raw.csv",
    "keys": [
      "X_train_bin",
      "X_train_offset",
      "y_train",
      "X_val_bin",
      "X_val_offset",
      "y_val",
      "feature_names",
      "label_names",
      "K",
      "num_bins"
    ],
    "n_train": 46876,
    "n_val": 11720,
    "n_features": 55,
    "num_bins": 512,
    "raw_feature_cols_preview": [
      "pslist.nproc",
      "pslist.nppid",
      "pslist.avg_threads",
      "pslist.nprocs64bit",
      "pslist.avg_handlers",
      "dlllist.ndlls",
      "dlllist.avg_dlls_per_proc",
      "handles.nhandles",
      "handles.avg_handles_per_proc",
      "handles.nport"
    ],
    "train_l2_col": "label_L2",
    "train_l3_col": "label_L3",
    "val_l2_col": "label_L2",
    "val_l3_col": "label_L3",
    "train_l3_unique_preview": [
      "Benign",
      "Ransomware-Ako",
      "Ransomware-Conti",
      "Ransomware-Maze",
      "Ransomware-Pysa",
      "Ransomware-Shade",
      "Spyware-180solutions",
      "Spyware-CWS",
      "Spyware-Gator",
      "Spyware-TIBS",
      "Spyware-Transponder",
      "Trojan-Emotet",
      "Trojan-Reconyc",
      "Trojan-Refroso",
      "Trojan-Scar",
      "Trojan-Zeus"
    ],
    "val_l3_unique_preview": [
      "Benign",
      "Ransomware-Ako",
      "Ransomware-Conti",
      "Ransomware-Maze",
      "Ransomware-Pysa",
      "Ransomware-Shade",
      "Spyware-180solutions",
      "Spyware-CWS",
      "Spyware-Gator",
      "Spyware-TIBS",
      "Spyware-Transponder",
      "Trojan-Emotet",
      "Trojan-Reconyc",
      "Trojan-Refroso",
      "Trojan-Scar",
      "Trojan-Zeus"
    ],
    "values_candidate": "offset_raw_one"
  },
  "model": {
    "checkpoint_path": "/kaggle/working/src_baocao/05_test/outputs/F1a2_stage2_depth_classifier/Keff512/F1a2_L1_reduce_num_layers_strong/best_model.pt",
    "state_dict_mode": "model_state_dict",
    "model_kwargs": {
      "num_bins": 512,
      "n_features": 55,
      "num_classes": 4,
      "value_dim": 32,
      "feature_dim": 32,
      "hidden_dim": 128,
      "num_layers": 1,
      "num_heads": 4,
      "dropout": 0.1,
      "classifier_hidden_dim": 128,
      "classifier_dropout": 0.1,
      "gate_init": 0.0
    },
    "missing": [],
    "unexpected": []
  }
}
```

## Split metrics from L1 checkpoint inference

| split   |     n |   accuracy |   macro_f1 |   weighted_f1 |   mean_confidence |   mean_top2_gap |   true_in_top2_rate |
|:--------|------:|-----------:|-----------:|--------------:|------------------:|----------------:|--------------------:|
| train   | 46876 |   0.941527 |   0.911531 |      0.94122  |          0.931266 |        0.873533 |            0.992192 |
| val     | 11720 |   0.876451 |   0.814233 |      0.876186 |          0.920654 |        0.853548 |            0.969625 |

## Family summary: worst validation families by error / ambiguity

| true_L2    | true_L3              |    n | audit_reliable_support   |   accuracy |   error_rate |   true_in_top2_rate |   top2_gap_mean |   true_prob_mean |   other_malware_prob_mass_mean |   pred_distribution_entropy_norm |   top2_distribution_entropy_norm |
|:-----------|:---------------------|-----:|:-------------------------|-----------:|-------------:|--------------------:|----------------:|-----------------:|-------------------------------:|---------------------------------:|---------------------------------:|
| Trojan     | Trojan-Zeus          |  390 | True                     |   0.376923 |  0.623077    |            0.866667 |        0.478434 |         0.444243 |                    0.554606    |                       0.925161   |                         0.755772 |
| Spyware    | Spyware-180solutions |  400 | True                     |   0.62     |  0.38        |            0.87     |        0.659289 |         0.576674 |                    0.423247    |                       0.833286   |                         0.805579 |
| Trojan     | Trojan-Emotet        |  393 | True                     |   0.70229  |  0.29771     |            0.933842 |        0.650335 |         0.663953 |                    0.335666    |                       0.732463   |                         0.802837 |
| Ransomware | Ransomware-Pysa      |  343 | True                     |   0.705539 |  0.294461    |            0.956268 |        0.620894 |         0.634317 |                    0.365507    |                       0.735885   |                         0.722736 |
| Ransomware | Ransomware-Conti     |  398 | True                     |   0.723618 |  0.276382    |            0.927136 |        0.649334 |         0.681901 |                    0.31805     |                       0.710801   |                         0.888123 |
| Ransomware | Ransomware-Shade     |  426 | True                     |   0.734742 |  0.265258    |            0.922535 |        0.679092 |         0.676532 |                    0.321192    |                       0.558314   |                         0.672259 |
| Ransomware | Ransomware-Ako       |  400 | True                     |   0.7425   |  0.2575      |            0.94     |        0.728203 |         0.681156 |                    0.318759    |                       0.67354    |                         0.932717 |
| Trojan     | Trojan-Scar          |  400 | True                     |   0.7575   |  0.2425      |            0.945    |        0.738572 |         0.72678  |                    0.271877    |                       0.656929   |                         0.780935 |
| Spyware    | Spyware-CWS          |  400 | True                     |   0.7875   |  0.2125      |            0.9275   |        0.728867 |         0.719951 |                    0.277164    |                       0.487879   |                         0.721028 |
| Ransomware | Ransomware-Maze      |  392 | True                     |   0.80102  |  0.19898     |            0.959184 |        0.731928 |         0.76321  |                    0.23676     |                       0.572435   |                         0.880193 |
| Trojan     | Trojan-Reconyc       |  314 | True                     |   0.821656 |  0.178344    |            0.964968 |        0.772971 |         0.799414 |                    0.20057     |                       0.538895   |                         0.747127 |
| Spyware    | Spyware-TIBS         |  282 | True                     |   0.847518 |  0.152482    |            0.957447 |        0.796856 |         0.794826 |                    0.20516     |                       0.484531   |                         0.702384 |
| Spyware    | Spyware-Transponder  |  482 | True                     |   0.858921 |  0.141079    |            0.958506 |        0.803197 |         0.798426 |                    0.197506    |                       0.369473   |                         0.68265  |
| Trojan     | Trojan-Refroso       |  400 | True                     |   0.88     |  0.12        |            0.98     |        0.818215 |         0.841043 |                    0.158951    |                       0.409321   |                         0.848154 |
| Spyware    | Spyware-Gator        |  440 | True                     |   0.934091 |  0.0659091   |            0.988636 |        0.753339 |         0.822391 |                    0.177599    |                       0.262669   |                         0.592193 |
| Benign     | Benign               | 5860 | True                     |   0.999147 |  0.000853242 |            0.999659 |        0.999745 |         0.999233 |                    0.000766739 |                       0.00678723 |                         0.606334 |

## Diagnostic validation family-aware smoothing candidate
```text
This candidate is DIAGNOSTIC_ONLY.
Do not use it to claim final unbiased val performance unless a separate final test exists.
```
| split_source   | usage_tag                                                     | true_L2    | true_L3              |   n | reliable_support   | eps_rule                                                                           |   eps_cap |   eps_family |   target_Benign |   target_Ransomware |   target_Spyware |   target_Trojan | source_note                                |   target_sum |
|:---------------|:--------------------------------------------------------------|:-----------|:---------------------|----:|:-------------------|:-----------------------------------------------------------------------------------|----------:|-------------:|----------------:|--------------------:|-----------------:|----------------:|:-------------------------------------------|-------------:|
| val            | DIAGNOSTIC_ONLY_validation_used_not_clean_for_final_val_claim | Ransomware | Ransomware-Ako       | 400 | True               | eps_family = min(eps_cap, mean probability mass assigned to other malware classes) |       0.2 |     0.2      |               0 |           0.8       |        0.0902262 |       0.109774  | family_probability_top2_weighted_candidate |            1 |
| val            | DIAGNOSTIC_ONLY_validation_used_not_clean_for_final_val_claim | Ransomware | Ransomware-Conti     | 398 | True               | eps_family = min(eps_cap, mean probability mass assigned to other malware classes) |       0.2 |     0.2      |               0 |           0.8       |        0.0686635 |       0.131336  | family_probability_top2_weighted_candidate |            1 |
| val            | DIAGNOSTIC_ONLY_validation_used_not_clean_for_final_val_claim | Ransomware | Ransomware-Maze      | 392 | True               | eps_family = min(eps_cap, mean probability mass assigned to other malware classes) |       0.2 |     0.2      |               0 |           0.8       |        0.0792418 |       0.120758  | family_probability_top2_weighted_candidate |            1 |
| val            | DIAGNOSTIC_ONLY_validation_used_not_clean_for_final_val_claim | Ransomware | Ransomware-Pysa      | 343 | True               | eps_family = min(eps_cap, mean probability mass assigned to other malware classes) |       0.2 |     0.2      |               0 |           0.8       |        0.0701794 |       0.129821  | family_probability_top2_weighted_candidate |            1 |
| val            | DIAGNOSTIC_ONLY_validation_used_not_clean_for_final_val_claim | Ransomware | Ransomware-Shade     | 426 | True               | eps_family = min(eps_cap, mean probability mass assigned to other malware classes) |       0.2 |     0.2      |               0 |           0.8       |        0.0643937 |       0.135606  | family_probability_top2_weighted_candidate |            1 |
| val            | DIAGNOSTIC_ONLY_validation_used_not_clean_for_final_val_claim | Spyware    | Spyware-180solutions | 400 | True               | eps_family = min(eps_cap, mean probability mass assigned to other malware classes) |       0.2 |     0.2      |               0 |           0.0930927 |        0.8       |       0.106907  | family_probability_top2_weighted_candidate |            1 |
| val            | DIAGNOSTIC_ONLY_validation_used_not_clean_for_final_val_claim | Spyware    | Spyware-CWS          | 400 | True               | eps_family = min(eps_cap, mean probability mass assigned to other malware classes) |       0.2 |     0.2      |               0 |           0.128479  |        0.8       |       0.0715208 | family_probability_top2_weighted_candidate |            1 |
| val            | DIAGNOSTIC_ONLY_validation_used_not_clean_for_final_val_claim | Spyware    | Spyware-Gator        | 440 | True               | eps_family = min(eps_cap, mean probability mass assigned to other malware classes) |       0.2 |     0.177599 |               0 |           0.112754  |        0.822401  |       0.0648451 | family_probability_top2_weighted_candidate |            1 |
| val            | DIAGNOSTIC_ONLY_validation_used_not_clean_for_final_val_claim | Spyware    | Spyware-TIBS         | 282 | True               | eps_family = min(eps_cap, mean probability mass assigned to other malware classes) |       0.2 |     0.2      |               0 |           0.108993  |        0.8       |       0.0910072 | family_probability_top2_weighted_candidate |            1 |
| val            | DIAGNOSTIC_ONLY_validation_used_not_clean_for_final_val_claim | Spyware    | Spyware-Transponder  | 482 | True               | eps_family = min(eps_cap, mean probability mass assigned to other malware classes) |       0.2 |     0.197506 |               0 |           0.128809  |        0.802494  |       0.0686971 | family_probability_top2_weighted_candidate |            1 |
| val            | DIAGNOSTIC_ONLY_validation_used_not_clean_for_final_val_claim | Trojan     | Trojan-Emotet        | 393 | True               | eps_family = min(eps_cap, mean probability mass assigned to other malware classes) |       0.2 |     0.2      |               0 |           0.128608  |        0.0713922 |       0.8       | family_probability_top2_weighted_candidate |            1 |
| val            | DIAGNOSTIC_ONLY_validation_used_not_clean_for_final_val_claim | Trojan     | Trojan-Reconyc       | 314 | True               | eps_family = min(eps_cap, mean probability mass assigned to other malware classes) |       0.2 |     0.2      |               0 |           0.0968263 |        0.103174  |       0.8       | family_probability_top2_weighted_candidate |            1 |
| val            | DIAGNOSTIC_ONLY_validation_used_not_clean_for_final_val_claim | Trojan     | Trojan-Refroso       | 400 | True               | eps_family = min(eps_cap, mean probability mass assigned to other malware classes) |       0.2 |     0.158951 |               0 |           0.0910641 |        0.0678865 |       0.841049  | family_probability_top2_weighted_candidate |            1 |
| val            | DIAGNOSTIC_ONLY_validation_used_not_clean_for_final_val_claim | Trojan     | Trojan-Scar          | 400 | True               | eps_family = min(eps_cap, mean probability mass assigned to other malware classes) |       0.2 |     0.2      |               0 |           0.0993044 |        0.100696  |       0.8       | family_probability_top2_weighted_candidate |            1 |
| val            | DIAGNOSTIC_ONLY_validation_used_not_clean_for_final_val_claim | Trojan     | Trojan-Zeus          | 390 | True               | eps_family = min(eps_cap, mean probability mass assigned to other malware classes) |       0.2 |     0.2      |               0 |           0.150358  |        0.0496417 |       0.8       | family_probability_top2_weighted_candidate |            1 |

## Train in-sample family-aware candidate
```text
This avoids validation labels, but it is biased because L1 was trained on train.
For clean choice, use OOF or calibration predictions.
```
| split_source   | usage_tag                                                     | true_L2    | true_L3              |    n | reliable_support   | eps_rule                                                                           |   eps_cap |   eps_family |   target_Benign |   target_Ransomware |   target_Spyware |   target_Trojan | source_note                                |   target_sum |
|:---------------|:--------------------------------------------------------------|:-----------|:---------------------|-----:|:-------------------|:-----------------------------------------------------------------------------------|----------:|-------------:|----------------:|--------------------:|-----------------:|----------------:|:-------------------------------------------|-------------:|
| train          | TRAIN_IN_SAMPLE_BIASED_not_validation_leak_but_overfit_biased | Ransomware | Ransomware-Ako       | 1600 | True               | eps_family = min(eps_cap, mean probability mass assigned to other malware classes) |       0.2 |    0.1987    |               0 |           0.8013    |        0.0833379 |       0.115362  | family_probability_top2_weighted_candidate |            1 |
| train          | TRAIN_IN_SAMPLE_BIASED_not_validation_leak_but_overfit_biased | Ransomware | Ransomware-Conti     | 1590 | True               | eps_family = min(eps_cap, mean probability mass assigned to other malware classes) |       0.2 |    0.19136   |               0 |           0.80864   |        0.0606179 |       0.130743  | family_probability_top2_weighted_candidate |            1 |
| train          | TRAIN_IN_SAMPLE_BIASED_not_validation_leak_but_overfit_biased | Ransomware | Ransomware-Maze      | 1566 | True               | eps_family = min(eps_cap, mean probability mass assigned to other malware classes) |       0.2 |    0.168021  |               0 |           0.831979  |        0.0562019 |       0.111819  | family_probability_top2_weighted_candidate |            1 |
| train          | TRAIN_IN_SAMPLE_BIASED_not_validation_leak_but_overfit_biased | Ransomware | Ransomware-Pysa      | 1374 | True               | eps_family = min(eps_cap, mean probability mass assigned to other malware classes) |       0.2 |    0.198705  |               0 |           0.801295  |        0.059179  |       0.139526  | family_probability_top2_weighted_candidate |            1 |
| train          | TRAIN_IN_SAMPLE_BIASED_not_validation_leak_but_overfit_biased | Ransomware | Ransomware-Shade     | 1702 | True               | eps_family = min(eps_cap, mean probability mass assigned to other malware classes) |       0.2 |    0.181555  |               0 |           0.818445  |        0.0529512 |       0.128604  | family_probability_top2_weighted_candidate |            1 |
| train          | TRAIN_IN_SAMPLE_BIASED_not_validation_leak_but_overfit_biased | Spyware    | Spyware-180solutions | 1600 | True               | eps_family = min(eps_cap, mean probability mass assigned to other malware classes) |       0.2 |    0.2       |               0 |           0.0930058 |        0.8       |       0.106994  | family_probability_top2_weighted_candidate |            1 |
| train          | TRAIN_IN_SAMPLE_BIASED_not_validation_leak_but_overfit_biased | Spyware    | Spyware-CWS          | 1600 | True               | eps_family = min(eps_cap, mean probability mass assigned to other malware classes) |       0.2 |    0.129498  |               0 |           0.0859197 |        0.870502  |       0.0435788 | family_probability_top2_weighted_candidate |            1 |
| train          | TRAIN_IN_SAMPLE_BIASED_not_validation_leak_but_overfit_biased | Spyware    | Spyware-Gator        | 1760 | True               | eps_family = min(eps_cap, mean probability mass assigned to other malware classes) |       0.2 |    0.147767  |               0 |           0.0938671 |        0.852233  |       0.0538999 | family_probability_top2_weighted_candidate |            1 |
| train          | TRAIN_IN_SAMPLE_BIASED_not_validation_leak_but_overfit_biased | Spyware    | Spyware-TIBS         | 1128 | True               | eps_family = min(eps_cap, mean probability mass assigned to other malware classes) |       0.2 |    0.104495  |               0 |           0.0576322 |        0.895505  |       0.0468629 | family_probability_top2_weighted_candidate |            1 |
| train          | TRAIN_IN_SAMPLE_BIASED_not_validation_leak_but_overfit_biased | Spyware    | Spyware-Transponder  | 1928 | True               | eps_family = min(eps_cap, mean probability mass assigned to other malware classes) |       0.2 |    0.109394  |               0 |           0.0749465 |        0.890606  |       0.0344476 | family_probability_top2_weighted_candidate |            1 |
| train          | TRAIN_IN_SAMPLE_BIASED_not_validation_leak_but_overfit_biased | Trojan     | Trojan-Emotet        | 1574 | True               | eps_family = min(eps_cap, mean probability mass assigned to other malware classes) |       0.2 |    0.2       |               0 |           0.128654  |        0.071346  |       0.8       | family_probability_top2_weighted_candidate |            1 |
| train          | TRAIN_IN_SAMPLE_BIASED_not_validation_leak_but_overfit_biased | Trojan     | Trojan-Reconyc       | 1256 | True               | eps_family = min(eps_cap, mean probability mass assigned to other malware classes) |       0.2 |    0.0899693 |               0 |           0.0378648 |        0.0521045 |       0.910031  | family_probability_top2_weighted_candidate |            1 |
| train          | TRAIN_IN_SAMPLE_BIASED_not_validation_leak_but_overfit_biased | Trojan     | Trojan-Refroso       | 1600 | True               | eps_family = min(eps_cap, mean probability mass assigned to other malware classes) |       0.2 |    0.106898  |               0 |           0.060433  |        0.0464649 |       0.893102  | family_probability_top2_weighted_candidate |            1 |
| train          | TRAIN_IN_SAMPLE_BIASED_not_validation_leak_but_overfit_biased | Trojan     | Trojan-Scar          | 1600 | True               | eps_family = min(eps_cap, mean probability mass assigned to other malware classes) |       0.2 |    0.155987  |               0 |           0.0754975 |        0.0804897 |       0.844013  | family_probability_top2_weighted_candidate |            1 |
| train          | TRAIN_IN_SAMPLE_BIASED_not_validation_leak_but_overfit_biased | Trojan     | Trojan-Zeus          | 1560 | True               | eps_family = min(eps_cap, mean probability mass assigned to other malware classes) |       0.2 |    0.2       |               0 |           0.155259  |        0.0447415 |       0.8       | family_probability_top2_weighted_candidate |            1 |

## Decision
```text
If many families have diffuse top2/probability mass across malware classes:
  family-aware smoothing is more justified than global L2 smoothing.

If only a few families dominate errors:
  design should focus on those families, not whole L2 classes.

If label_L3 is missing or equals L2:
  this audit cannot answer family-level behavior; fix raw label source first.
```