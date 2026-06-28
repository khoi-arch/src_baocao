# F1e1a_v4 Clean Calibration-Derived Family Smoothing Matrix

## Purpose

```text
No final model training.
No validation-derived hyperparameter selection.
Train temporary L1 on train_inner, infer calibration, derive locked family-aware smoothing matrix.
```

## Leakage status

```text
Validation is not used to derive the matrix.
The locked matrix can be used in F1e1b, then validation can be evaluated once.
Caveat: existing tokenized dataset is reused; fully nested preprocessing would rebuild tokens on train_inner only.
```

## Dataset / split info

```json
{
  "dataset_npz": "/kaggle/working/src_baocao/03_outputs/05_dataset/dataset.npz",
  "train_raw": "/kaggle/working/src_baocao/01_split/train_raw.csv",
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
  "l2_col": "label_L2",
  "l3_col": "label_L3",
  "l3_missing_fallback_to_L2": false,
  "label_L3_unique_count": 16,
  "label_L3_unique_preview": [
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
  "values_candidate": "offset_raw_one",
  "method_limitation": "No validation used. Existing dataset tokens/offsets are reused from prior preprocessing; fully strict nested preprocessing would rebuild tokens using train_inner only."
}
```

## Training config

```json
{
  "experiment": "F1e1a_v4_clean_calibration_family_smoothing",
  "role": "derive locked family smoothing matrix from train_inner/calibration; no validation used",
  "class_names": [
    "Benign",
    "Ransomware",
    "Spyware",
    "Trojan"
  ],
  "malware_classes": [
    "Ransomware",
    "Spyware",
    "Trojan"
  ],
  "model_config": {
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
  "training": {
    "dataset_npz": "03_outputs/05_dataset/dataset.npz",
    "train_raw": "01_split/train_raw.csv",
    "base_config": "03_outputs/06_model/config.json",
    "out_dir": "05_test/outputs/F1e1a_v4_clean_calibration_family_smoothing",
    "combined_zip": "05_test/outputs/F1e1a_v4_clean_calibration_family_smoothing.zip",
    "class_names": "Benign,Ransomware,Spyware,Trojan",
    "malware_classes": "Ransomware,Spyware,Trojan",
    "calib_size": 0.2,
    "eps_cap": 0.2,
    "min_family_support": 30,
    "device": "cuda",
    "epochs": 80,
    "batch_size": 512,
    "lr": 0.001,
    "weight_decay": 0.0001,
    "warmup_epochs": 8,
    "min_lr_ratio": 0.05,
    "patience": 12,
    "min_delta": 0.0001,
    "num_workers": 2,
    "grad_clip_norm": 1.0,
    "seed": 42,
    "amp": true,
    "hidden_dim": null,
    "num_heads": null,
    "classifier_hidden_dim": null,
    "dropout": null,
    "classifier_dropout": null
  },
  "split": {
    "n_train_inner": 37500,
    "n_calibration": 9376,
    "calib_size": 0.2,
    "stratification": "label_L2::label_L3 fallback to L2/y for rare strata"
  },
  "loss": {
    "name": "CrossEntropyLoss",
    "class_weights": [
      0.5,
      1.4961698055267334,
      1.4621022939682007,
      1.543972373008728
    ]
  },
  "methodology": {
    "validation_used_for_matrix": false,
    "fake_data_used": false,
    "known_caveat": "No validation used. Existing dataset tokens/offsets are reused from prior preprocessing; fully strict nested preprocessing would rebuild tokens using train_inner only."
  }
}
```

## Temporary calibration model metrics

| split       |   accuracy |   macro_f1 |   weighted_f1 |     n |
|:------------|-----------:|-----------:|--------------:|------:|
| train_inner |   0.923413 |   0.884379 |      0.92317  | 37500 |
| calibration |   0.849936 |   0.774828 |      0.849861 |  9376 |

Additional scalar metrics:
| metric                               |    value |
|:-------------------------------------|---------:|
| gap_train_inner_minus_calib_macro_f1 | 0.109551 |

## Calibration family summary: hardest families

| true_L2    | true_L3              |    n | audit_reliable_support   |   accuracy |   error_rate |   true_in_top2_rate |   top2_gap_mean |   true_prob_mean |   other_malware_prob_mass_mean |   mean_prob_Ransomware |   mean_prob_Spyware |   mean_prob_Trojan |   pred_rate_Ransomware |   pred_rate_Spyware |   pred_rate_Trojan |   top2_rate_Ransomware |   top2_rate_Spyware |   top2_rate_Trojan |
|:-----------|:---------------------|-----:|:-------------------------|-----------:|-------------:|--------------------:|----------------:|-----------------:|-------------------------------:|-----------------------:|--------------------:|-------------------:|-----------------------:|--------------------:|-------------------:|-----------------------:|--------------------:|-------------------:|
| Trojan     | Trojan-Zeus          |  312 | True                     |   0.307692 |  0.692308    |            0.875    |        0.448341 |         0.409285 |                    0.590518    |            0.472956    |          0.117562   |        0.409285    |              0.554487  |           0.137821  |        0.307692    |               0.317308 |           0.112179  |           0.567308 |
| Spyware    | Spyware-180solutions |  320 | True                     |   0.546875 |  0.453125    |            0.7625   |        0.602032 |         0.493741 |                    0.506168    |            0.205644    |          0.493741   |        0.300524    |              0.1875    |           0.546875  |        0.265625    |               0.421875 |           0.215625  |           0.353125 |
| Trojan     | Trojan-Emotet        |  315 | True                     |   0.628571 |  0.371429    |            0.949206 |        0.607036 |         0.613419 |                    0.386553    |            0.292935    |          0.0936179  |        0.613419    |              0.307937  |           0.0634921 |        0.628571    |               0.406349 |           0.273016  |           0.320635 |
| Ransomware | Ransomware-Ako       |  320 | True                     |   0.65625  |  0.34375     |            0.90625  |        0.649374 |         0.593522 |                    0.406444    |            0.593522    |          0.181441   |        0.225003    |              0.65625   |           0.190625  |        0.153125    |               0.25     |           0.2375    |           0.5125   |
| Spyware    | Spyware-CWS          |  320 | True                     |   0.671875 |  0.328125    |            0.8625   |        0.639438 |         0.580243 |                    0.412761    |            0.228726    |          0.580243   |        0.184035    |              0.18125   |           0.671875  |        0.140625    |               0.509375 |           0.190625  |           0.296875 |
| Trojan     | Trojan-Scar          |  320 | True                     |   0.69375  |  0.30625     |            0.94375  |        0.628374 |         0.660184 |                    0.339793    |            0.187244    |          0.152549   |        0.660184    |              0.175     |           0.13125   |        0.69375     |               0.403125 |           0.346875  |           0.25     |
| Ransomware | Ransomware-Shade     |  340 | True                     |   0.694118 |  0.305882    |            0.920588 |        0.57503  |         0.622988 |                    0.376992    |            0.622988    |          0.153856   |        0.223136    |              0.694118  |           0.161765  |        0.144118    |               0.226471 |           0.147059  |           0.626471 |
| Ransomware | Ransomware-Pysa      |  275 | True                     |   0.72     |  0.28        |            0.934545 |        0.619511 |         0.6603   |                    0.339659    |            0.6603      |          0.11706    |        0.222599    |              0.72      |           0.123636  |        0.156364    |               0.214545 |           0.156364  |           0.625455 |
| Ransomware | Ransomware-Conti     |  318 | True                     |   0.720126 |  0.279874    |            0.902516 |        0.565747 |         0.633888 |                    0.360976    |            0.633888    |          0.138248   |        0.222728    |              0.720126  |           0.147799  |        0.125786    |               0.18239  |           0.176101  |           0.641509 |
| Ransomware | Ransomware-Maze      |  313 | True                     |   0.741214 |  0.258786    |            0.939297 |        0.667524 |         0.696274 |                    0.30232     |            0.696274    |          0.116345   |        0.185975    |              0.741214  |           0.124601  |        0.13099     |               0.198083 |           0.169329  |           0.632588 |
| Spyware    | Spyware-TIBS         |  226 | True                     |   0.769912 |  0.230088    |            0.893805 |        0.714126 |         0.716614 |                    0.280221    |            0.166208    |          0.716614   |        0.114012    |              0.132743  |           0.769912  |        0.0929204   |               0.690265 |           0.123894  |           0.185841 |
| Trojan     | Trojan-Reconyc       |  251 | True                     |   0.780876 |  0.219124    |            0.940239 |        0.741759 |         0.748255 |                    0.251713    |            0.136187    |          0.115526   |        0.748255    |              0.123506  |           0.0956175 |        0.780876    |               0.418327 |           0.418327  |           0.159363 |
| Spyware    | Spyware-Gator        |  352 | True                     |   0.84375  |  0.15625     |            0.928977 |        0.658145 |         0.732204 |                    0.263126    |            0.137277    |          0.732204   |        0.125849    |              0.0738636 |           0.84375   |        0.0767045   |               0.585227 |           0.0852273 |           0.318182 |
| Spyware    | Spyware-Transponder  |  386 | True                     |   0.84715  |  0.15285     |            0.955959 |        0.718859 |         0.752751 |                    0.243108    |            0.12256     |          0.752751   |        0.120547    |              0.0673575 |           0.84715   |        0.0829016   |               0.57772  |           0.108808  |           0.295337 |
| Trojan     | Trojan-Refroso       |  320 | True                     |   0.865625 |  0.134375    |            0.9625   |        0.744169 |         0.790393 |                    0.209581    |            0.11374     |          0.095841   |        0.790393    |              0.0625    |           0.071875  |        0.865625    |               0.48125  |           0.41875   |           0.096875 |
| Benign     | Benign               | 4688 | True                     |   0.999787 |  0.000213311 |            0.999787 |        0.999914 |         0.99977  |                    0.000230522 |            2.49978e-05 |          1.4509e-05 |        0.000191015 |              0         |           0         |        0.000213311 |               0.52965  |           0.21907   |           0.25128  |

## Locked family smoothing matrix

| true_L2    | true_L3              |   n_calibration | reliable_support   |   eps_cap |   eps_family | source                                             |   target_Benign |   target_Ransomware |   target_Spyware |   target_Trojan |   target_sum |
|:-----------|:---------------------|----------------:|:-------------------|----------:|-------------:|:---------------------------------------------------|----------------:|--------------------:|-----------------:|----------------:|-------------:|
| Benign     | Benign               |            4688 | True               |       0.2 |          0   | non_malware_or_benign_one_hot                      |               1 |           0         |        0         |       0         |            1 |
| Ransomware | Ransomware-Ako       |             320 | True               |       0.2 |          0.2 | CALIBRATION_DERIVED_family_prob_top2_pred_weighted |               0 |           0.8       |        0.0798375 |       0.120162  |            1 |
| Ransomware | Ransomware-Conti     |             318 | True               |       0.2 |          0.2 | CALIBRATION_DERIVED_family_prob_top2_pred_weighted |               0 |           0.8       |        0.0623013 |       0.137699  |            1 |
| Ransomware | Ransomware-Maze      |             313 | True               |       0.2 |          0.2 | CALIBRATION_DERIVED_family_prob_top2_pred_weighted |               0 |           0.8       |        0.0599648 |       0.140035  |            1 |
| Ransomware | Ransomware-Pysa      |             275 | True               |       0.2 |          0.2 | CALIBRATION_DERIVED_family_prob_top2_pred_weighted |               0 |           0.8       |        0.056042  |       0.143958  |            1 |
| Ransomware | Ransomware-Shade     |             340 | True               |       0.2 |          0.2 | CALIBRATION_DERIVED_family_prob_top2_pred_weighted |               0 |           0.8       |        0.0630569 |       0.136943  |            1 |
| Spyware    | Spyware-180solutions |             320 | True               |       0.2 |          0.2 | CALIBRATION_DERIVED_family_prob_top2_pred_weighted |               0 |           0.0925236 |        0.8       |       0.107476  |            1 |
| Spyware    | Spyware-CWS          |             320 | True               |       0.2 |          0.2 | CALIBRATION_DERIVED_family_prob_top2_pred_weighted |               0 |           0.118258  |        0.8       |       0.0817424 |            1 |
| Spyware    | Spyware-Gator        |             352 | True               |       0.2 |          0.2 | CALIBRATION_DERIVED_family_prob_top2_pred_weighted |               0 |           0.119614  |        0.8       |       0.0803861 |            1 |
| Spyware    | Spyware-TIBS         |             226 | True               |       0.2 |          0.2 | CALIBRATION_DERIVED_family_prob_top2_pred_weighted |               0 |           0.141262  |        0.8       |       0.0587383 |            1 |
| Spyware    | Spyware-Transponder  |             386 | True               |       0.2 |          0.2 | CALIBRATION_DERIVED_family_prob_top2_pred_weighted |               0 |           0.119966  |        0.8       |       0.0800341 |            1 |
| Trojan     | Trojan-Emotet        |             315 | True               |       0.2 |          0.2 | CALIBRATION_DERIVED_family_prob_top2_pred_weighted |               0 |           0.139458  |        0.0605422 |       0.8       |            1 |
| Trojan     | Trojan-Reconyc       |             251 | True               |       0.2 |          0.2 | CALIBRATION_DERIVED_family_prob_top2_pred_weighted |               0 |           0.103681  |        0.0963186 |       0.8       |            1 |
| Trojan     | Trojan-Refroso       |             320 | True               |       0.2 |          0.2 | CALIBRATION_DERIVED_family_prob_top2_pred_weighted |               0 |           0.106706  |        0.0932939 |       0.8       |            1 |
| Trojan     | Trojan-Scar          |             320 | True               |       0.2 |          0.2 | CALIBRATION_DERIVED_family_prob_top2_pred_weighted |               0 |           0.109287  |        0.0907127 |       0.8       |            1 |
| Trojan     | Trojan-Zeus          |             312 | True               |       0.2 |          0.2 | CALIBRATION_DERIVED_family_prob_top2_pred_weighted |               0 |           0.157313  |        0.0426873 |       0.8       |            1 |

## Next step

```text
F1e1b: train L1 + locked family-aware smoothing matrix on full original train.
Then evaluate once on validation.
Do not tune matrix using validation result unless you create a new calibration protocol.
```