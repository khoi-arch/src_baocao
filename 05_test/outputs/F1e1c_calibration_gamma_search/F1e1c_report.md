# F1e1c Calibration-Only Gamma Search

## Protocol

```text
For each gamma, train temporary L1 from scratch on train_inner.
Evaluate calibration after fixed epochs.
Choose gamma by calibration macro-F1.
Validation is not used.
```

## Gamma results

|   gamma |   train_inner_accuracy |   train_inner_macro_f1 |   train_inner_weighted_f1 |   calibration_accuracy |   calibration_macro_f1 |   calibration_weighted_f1 |   gap_train_inner_minus_calibration_macro_f1 |
|--------:|-----------------------:|-----------------------:|--------------------------:|-----------------------:|-----------------------:|--------------------------:|---------------------------------------------:|
|   0.125 |               0.92632  |               0.888401 |                  0.925861 |               0.850256 |               0.774161 |                  0.849682 |                                     0.114239 |
|   1     |               0.925333 |               0.887037 |                  0.924952 |               0.848763 |               0.772145 |                  0.848267 |                                     0.114892 |
|   0.75  |               0.925333 |               0.887006 |                  0.924931 |               0.848763 |               0.77213  |                  0.848204 |                                     0.114876 |
|   0     |               0.92264  |               0.882888 |                  0.92218  |               0.84759  |               0.770195 |                  0.846774 |                                     0.112693 |
|   0.25  |               0.924293 |               0.885415 |                  0.923867 |               0.84663  |               0.768956 |                  0.846119 |                                     0.116459 |
|   0.5   |               0.925013 |               0.886513 |                  0.924607 |               0.846203 |               0.768371 |                  0.845842 |                                     0.118142 |

## Selected gamma

```json
{
  "selected_gamma": 0.125,
  "selection_metric": "calibration_macro_f1",
  "selected_row": {
    "gamma": 0.125,
    "train_inner_accuracy": 0.92632,
    "train_inner_macro_f1": 0.8884005643533182,
    "train_inner_weighted_f1": 0.9258611447801616,
    "calibration_accuracy": 0.8502559726962458,
    "calibration_macro_f1": 0.7741612023752941,
    "calibration_weighted_f1": 0.8496823001611946,
    "gap_train_inner_minus_calibration_macro_f1": 0.11423936197802409
  },
  "validation_used": false,
  "target_info_by_gamma": {
    "0.0": {
      "gamma": 0.0,
      "matched": 46876,
      "fallback_onehot": 0,
      "missing_top": []
    },
    "0.125": {
      "gamma": 0.125,
      "matched": 46876,
      "fallback_onehot": 0,
      "missing_top": []
    },
    "0.25": {
      "gamma": 0.25,
      "matched": 46876,
      "fallback_onehot": 0,
      "missing_top": []
    },
    "0.5": {
      "gamma": 0.5,
      "matched": 46876,
      "fallback_onehot": 0,
      "missing_top": []
    },
    "0.75": {
      "gamma": 0.75,
      "matched": 46876,
      "fallback_onehot": 0,
      "missing_top": []
    },
    "1.0": {
      "gamma": 1.0,
      "matched": 46876,
      "fallback_onehot": 0,
      "missing_top": []
    }
  },
  "locked_matrix_file": "F1e1c_locked_scaled_matrix_CALIBRATION_SELECTED.csv"
}
```

## Config summary

```json
{
  "experiment": "F1e1c_calibration_gamma_search",
  "methodology": {
    "validation_used": false,
    "gamma_selected_by": "calibration_macro_f1",
    "epochs_fixed": 49,
    "matrix_source": "/kaggle/working/src_baocao/05_test/outputs/F1e1a_v4c_prob_only_calibration_family_matrix/F1e1a_v4c_locked_family_smoothing_matrix_PROB_ONLY_CALIBRATION_DERIVED.csv"
  },
  "dataset_info": {
    "dataset_npz": "/kaggle/working/src_baocao/03_outputs/05_dataset/dataset.npz",
    "train_raw": "/kaggle/working/src_baocao/01_split/train_raw.csv",
    "val_raw_not_used": "/kaggle/working/src_baocao/01_split/val_raw.csv",
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
    "label_L2_col": "label_L2",
    "label_L3_col": "label_L3",
    "L3_unique_count": 16,
    "values_candidate": "offset_raw_one"
  },
  "split_info": {
    "source": "/kaggle/working/src_baocao/05_test/outputs/F1e1a_v4_clean_calibration_family_smoothing/F1e1a_v4_split_indices.npz",
    "train_key": "train_inner_idx",
    "calib_key": "calibration_idx",
    "n_train_inner": 37500,
    "n_calibration": 9376
  },
  "gamma_grid": [
    0.0,
    0.125,
    0.25,
    0.5,
    0.75,
    1.0
  ],
  "args": {
    "dataset_npz": "03_outputs/05_dataset/dataset.npz",
    "train_raw": "01_split/train_raw.csv",
    "val_raw": "01_split/val_raw.csv",
    "base_config": "03_outputs/06_model/config.json",
    "v4_dir": "05_test/outputs/F1e1a_v4_clean_calibration_family_smoothing",
    "matrix_csv": "05_test/outputs/F1e1a_v4c_prob_only_calibration_family_matrix/F1e1a_v4c_locked_family_smoothing_matrix_PROB_ONLY_CALIBRATION_DERIVED.csv",
    "out_dir": "05_test/outputs/F1e1c_calibration_gamma_search",
    "combined_zip": "05_test/outputs/F1e1c_calibration_gamma_search.zip",
    "class_names": "Benign,Ransomware,Spyware,Trojan",
    "gamma_grid": "0.0,0.125,0.25,0.5,0.75,1.0",
    "epochs": 49,
    "calib_size": 0.2,
    "batch_size": 512,
    "lr": 0.001,
    "weight_decay": 0.0001,
    "warmup_epochs": 8,
    "min_lr_ratio": 0.05,
    "grad_clip_norm": 1.0,
    "num_workers": 2,
    "seed": 42,
    "device": "cuda",
    "amp": true,
    "no_class_weights": false,
    "save_models": false,
    "hidden_dim": null,
    "num_heads": null,
    "classifier_hidden_dim": null,
    "dropout": null,
    "classifier_dropout": null
  },
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
  }
}
```

## Next step

```text
If selected gamma is > 0 and calibration improves over gamma=0,
run F1e1d/F1e1b-style full-train experiment using the locked scaled matrix.
If gamma=0 wins, family smoothing should be rejected for now.
```