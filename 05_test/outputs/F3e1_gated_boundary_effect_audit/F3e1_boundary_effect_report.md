# F3e1 gated hard-negative boundary-effect audit

## Scope

```text
No training. No official validation.
Compare F3e0 baseline_ce vs gated_adaptive_hardneg on the same calibration split.
Goal: explain why gated loss fixed Trojan but damaged Ransomware.
```

## Metrics

```json
{
  "baseline": {
    "n": 9376,
    "accuracy": 0.8530290102389079,
    "macro_f1": 0.7786549144264168,
    "weighted_f1": 0.8524508975820402
  },
  "gated": {
    "n": 9376,
    "accuracy": 0.8522824232081911,
    "macro_f1": 0.7775327736739271,
    "weighted_f1": 0.8519090103822188
  },
  "delta": {
    "accuracy": -0.0007465870307167277,
    "macro_f1": -0.0011221407524896199,
    "weighted_f1": -0.0005418871998214003
  }
}
```

## Switch counts

| switch_type        |   count |
|:-------------------|--------:|
| both_correct       |    7621 |
| both_wrong_same    |     871 |
| damaged            |     377 |
| fixed              |     370 |
| both_wrong_changed |     137 |

## Switch by true label

| true_label   | switch_type        |   count |   mean_gated_gate |   mean_delta_true_prob |
|:-------------|:-------------------|--------:|------------------:|-----------------------:|
| Benign       | both_correct       |    4686 |        0.00015308 |           -1.23884e-05 |
| Benign       | both_wrong_changed |       1 |        0.374773   |            3.72738e-05 |
| Benign       | both_wrong_same    |       1 |        0.00304135 |            3.10415e-06 |
| Ransomware   | both_correct       |     966 |        0.175247   |            0.006385    |
| Ransomware   | both_wrong_same    |     285 |        0.293034   |           -0.0173241   |
| Ransomware   | damaged            |     174 |        0.372045   |           -0.418059    |
| Ransomware   | fixed              |      99 |        0.286095   |            0.477778    |
| Ransomware   | both_wrong_changed |      42 |        0.361188   |           -0.00679723  |
| Spyware      | both_correct       |    1120 |        0.169895   |            0.000325961 |
| Spyware      | both_wrong_same    |     203 |        0.257926   |            0.00689623  |
| Spyware      | fixed              |     120 |        0.291906   |            0.512356    |
| Spyware      | damaged            |     107 |        0.316332   |           -0.520934    |
| Spyware      | both_wrong_changed |      54 |        0.347989   |            0.0115314   |
| Trojan       | both_correct       |     849 |        0.140522   |            0.0291388   |
| Trojan       | both_wrong_same    |     382 |        0.314308   |           -0.00577881  |
| Trojan       | fixed              |     151 |        0.36186    |            0.359742    |
| Trojan       | damaged            |      96 |        0.3698     |           -0.439774    |
| Trojan       | both_wrong_changed |      40 |        0.384586   |           -0.00466699  |

## Major prediction transitions

| true_label   | base_pred_label   | gated_pred_label   |   count |   mean_gated_gate |   mean_delta_true_prob |
|:-------------|:------------------|:-------------------|--------:|------------------:|-----------------------:|
| Benign       | Benign            | Benign             |    4686 |        0.00015308 |           -1.23884e-05 |
| Spyware      | Spyware           | Spyware            |    1120 |        0.169895   |            0.000325961 |
| Ransomware   | Ransomware        | Ransomware         |     966 |        0.175247   |            0.006385    |
| Trojan       | Trojan            | Trojan             |     849 |        0.140522   |            0.0291388   |
| Trojan       | Ransomware        | Ransomware         |     240 |        0.327781   |           -0.00736318  |
| Ransomware   | Spyware           | Spyware            |     173 |        0.304118   |            0.000644933 |
| Trojan       | Spyware           | Spyware            |     142 |        0.291536   |           -0.003101    |
| Ransomware   | Ransomware        | Trojan             |     114 |        0.412579   |           -0.355272    |
| Spyware      | Trojan            | Trojan             |     113 |        0.28047    |            0.0098342   |
| Ransomware   | Trojan            | Trojan             |     112 |        0.275912   |           -0.0450798   |
| Trojan       | Ransomware        | Trojan             |     111 |        0.371948   |            0.329693    |
| Spyware      | Ransomware        | Ransomware         |      89 |        0.2322     |            0.00324349  |
| Spyware      | Ransomware        | Spyware            |      73 |        0.289714   |            0.508637    |
| Ransomware   | Ransomware        | Spyware            |      60 |        0.295031   |           -0.537353    |
| Spyware      | Spyware           | Ransomware         |      59 |        0.31649    |           -0.560924    |
| Ransomware   | Trojan            | Ransomware         |      50 |        0.300597   |            0.44744     |
| Trojan       | Trojan            | Spyware            |      48 |        0.393921   |           -0.390837    |
| Trojan       | Trojan            | Ransomware         |      48 |        0.345678   |           -0.488712    |
| Spyware      | Spyware           | Trojan             |      48 |        0.316137   |           -0.471779    |
| Ransomware   | Spyware           | Ransomware         |      47 |        0.277382   |            0.510288    |
| Spyware      | Trojan            | Spyware            |      45 |        0.308125   |            0.498442    |
| Trojan       | Spyware           | Trojan             |      39 |        0.338416   |            0.435238    |
| Spyware      | Ransomware        | Trojan             |      33 |        0.393843   |            0.0165783   |
| Trojan       | Ransomware        | Spyware            |      24 |        0.388364   |            0.0199107   |
| Ransomware   | Spyware           | Trojan             |      21 |        0.353862   |            0.0395994   |
| Ransomware   | Trojan            | Spyware            |      21 |        0.368515   |           -0.0531939   |
| Spyware      | Trojan            | Ransomware         |      19 |        0.298841   |            0.00611434  |
| Trojan       | Spyware           | Ransomware         |      16 |        0.378919   |           -0.0415336   |
| Ransomware   | Benign            | Ransomware         |       2 |        0.128347   |            0.47222     |
| Spyware      | Benign            | Spyware            |       2 |        0.00696648 |            0.961201    |

## Pair margin shifts

| true_label   | confuser   |   support |   base_margin_mean |   gated_margin_mean |   delta_margin_mean |   base_margin_median |   gated_margin_median |   delta_margin_median |   base_correct_rate |   gated_correct_rate |   delta_correct_rate |   fix_count |   damage_count |
|:-------------|:-----------|----------:|-------------------:|--------------------:|--------------------:|---------------------:|----------------------:|----------------------:|--------------------:|---------------------:|---------------------:|------------:|---------------:|
| Ransomware   | Benign     |      1566 |           12.9668  |            11.5016  |         -1.46517    |             13.1236  |              11.6403  |            -1.38842   |            0.727969 |             0.680077 |          -0.0478927  |          99 |            174 |
| Ransomware   | Spyware    |      1566 |            4.39763 |             4.07989 |         -0.317738   |              4.98945 |               4.63182 |            -0.297583  |            0.727969 |             0.680077 |          -0.0478927  |          99 |            174 |
| Ransomware   | Trojan     |      1566 |            2.68899 |             2.69754 |          0.00855346 |              1.65992 |               1.71432 |            -0.0187199 |            0.727969 |             0.680077 |          -0.0478927  |          99 |            174 |
| Benign       | Ransomware |      4688 |           17.8604  |            20.1867  |          2.32628    |             17.9534  |              20.3946  |             2.41907   |            0.999573 |             0.999573 |           0          |           0 |              0 |
| Benign       | Spyware    |      4688 |           18.8613  |            16.8956  |         -1.96574    |             19.4432  |              17.2669  |            -2.04086   |            0.999573 |             0.999573 |           0          |           0 |              0 |
| Benign       | Trojan     |      4688 |           16.206   |            15.5306  |         -0.675388   |             16.4553  |              15.7247  |            -0.725415  |            0.999573 |             0.999573 |           0          |           0 |              0 |
| Spyware      | Benign     |      1604 |           10.7655  |            11.9089  |          1.14341    |             11.0972  |              12.386   |             1.19769   |            0.764963 |             0.773067 |           0.00810474 |         120 |            107 |
| Spyware      | Ransomware |      1604 |            3.04806 |             3.13353 |          0.0854685  |              2.56219 |               2.47916 |             0.142352  |            0.764963 |             0.773067 |           0.00810474 |         120 |            107 |
| Spyware      | Trojan     |      1604 |            4.18415 |             4.26339 |          0.079243   |              2.84979 |               3.05425 |             0.118972  |            0.764963 |             0.773067 |           0.00810474 |         120 |            107 |
| Trojan       | Benign     |      1518 |           12.6468  |            12.3031  |         -0.343712   |             12.5919  |              12.2428  |            -0.320271  |            0.62253  |             0.658762 |           0.0362319  |         151 |             96 |
| Trojan       | Ransomware |      1518 |            3.04031 |             3.17265 |          0.132344   |              1.82046 |               2.33784 |             0.149319  |            0.62253  |             0.658762 |           0.0362319  |         151 |             96 |
| Trojan       | Spyware    |      1518 |            4.00732 |             3.92741 |         -0.0799146  |              4.10782 |               3.88362 |            -0.103583  |            0.62253  |             0.658762 |           0.0362319  |         151 |             96 |

## Gate by switch type

| switch_type        |   count |   gate_mean |   gate_median |   gate_std |   base_true_prob_mean |   gated_true_prob_mean |   delta_true_prob_mean |
|:-------------------|--------:|------------:|--------------:|-----------:|----------------------:|-----------------------:|-----------------------:|
| both_correct       |    7621 |   0.0629302 |   1.15096e-05 |   0.135848 |              0.94677  |               0.950865 |             0.00409576 |
| both_wrong_same    |     871 |   0.293849  |   0.332932    |   0.180967 |              0.180627 |               0.174031 |            -0.00659578 |
| damaged            |     377 |   0.355661  |   0.402189    |   0.162922 |              0.701031 |               0.248245 |            -0.452787   |
| fixed              |     370 |   0.3189    |   0.359855    |   0.179211 |              0.267067 |               0.707888 |             0.440821   |
| both_wrong_changed |     137 |   0.362916  |   0.432386    |   0.169109 |              0.150116 |               0.151215 |             0.00109903 |

## Gate by family

| true_label   | family               |   count |   gate_mean |   gate_median |   base_correct_rate |   gated_correct_rate |   delta_correct_rate |
|:-------------|:---------------------|--------:|------------:|--------------:|--------------------:|---------------------:|---------------------:|
| Benign       | Benign               |    4688 | 0.000233606 |   4.35515e-06 |            0.999573 |             0.999573 |           0          |
| Spyware      | Spyware-Transponder  |     386 | 0.178575    |   0.141188    |            0.857513 |             0.860104 |           0.00259067 |
| Spyware      | Spyware-Gator        |     352 | 0.222229    |   0.203109    |            0.866477 |             0.877841 |           0.0113636  |
| Ransomware   | Ransomware-Shade     |     340 | 0.243735    |   0.248877    |            0.702941 |             0.664706 |          -0.0382353  |
| Spyware      | Spyware-CWS          |     320 | 0.21677     |   0.209348    |            0.7125   |             0.715625 |           0.003125   |
| Trojan       | Trojan-Scar          |     320 | 0.211538    |   0.155938    |            0.60625  |             0.653125 |           0.046875   |
| Spyware      | Spyware-180solutions |     320 | 0.243102    |   0.235934    |            0.553125 |             0.56875  |           0.015625   |
| Ransomware   | Ransomware-Ako       |     320 | 0.216944    |   0.192813    |            0.6875   |             0.703125 |           0.015625   |
| Trojan       | Trojan-Refroso       |     320 | 0.16487     |   0.058585    |            0.821875 |             0.821875 |           0          |
| Ransomware   | Ransomware-Conti     |     318 | 0.250073    |   0.24304     |            0.742138 |             0.694969 |          -0.0471698  |
| Trojan       | Trojan-Emotet        |     315 | 0.232007    |   0.219709    |            0.64127  |             0.679365 |           0.0380952  |
| Ransomware   | Ransomware-Maze      |     313 | 0.202897    |   0.110371    |            0.776358 |             0.683706 |          -0.0926518  |
| Trojan       | Trojan-Zeus          |     312 | 0.341472    |   0.441116    |            0.278846 |             0.355769 |           0.0769231  |
| Ransomware   | Ransomware-Pysa      |     275 | 0.238947    |   0.246495    |            0.734545 |             0.650909 |          -0.0836364  |
| Trojan       | Trojan-Reconyc       |     251 | 0.178573    |   0.123564    |            0.792829 |             0.808765 |           0.0159363  |
| Spyware      | Spyware-TIBS         |     226 | 0.159273    |   0.0394905   |            0.823009 |             0.831858 |           0.00884956 |

## Flags

```json
{
  "fix_count": 370,
  "damage_count": 377,
  "net_fix_minus_damage": -7,
  "delta_macro_f1": -0.0011221407524896199,
  "per_class_delta_acc": [
    {
      "true_label": "Benign",
      "support": 4688,
      "base_acc": 0.9995733788395904,
      "gated_acc": 0.9995733788395904,
      "fix": 0,
      "damage": 0,
      "gate_mean": 0.0002336062170797959,
      "delta_acc": 0.0
    },
    {
      "true_label": "Ransomware",
      "support": 1566,
      "base_acc": 0.7279693486590039,
      "gated_acc": 0.6800766283524904,
      "fix": 99,
      "damage": 174,
      "gate_mean": 0.23054413497447968,
      "delta_acc": -0.04789272030651348
    },
    {
      "true_label": "Spyware",
      "support": 1604,
      "base_acc": 0.7649625935162094,
      "gated_acc": 0.773067331670823,
      "fix": 120,
      "damage": 107,
      "gate_mean": 0.20592831075191498,
      "delta_acc": 0.008104738154613544
    },
    {
      "true_label": "Trojan",
      "support": 1518,
      "base_acc": 0.6225296442687747,
      "gated_acc": 0.6587615283267457,
      "fix": 151,
      "damage": 96,
      "gate_mean": 0.22720274329185486,
      "delta_acc": 0.036231884057971064
    }
  ],
  "gate_mean_by_switch": [
    {
      "switch_type": "both_correct",
      "count": 7621,
      "gate_mean": 0.06293024122714996,
      "gate_median": 1.150955995399272e-05,
      "gate_std": 0.13584759831428528,
      "base_true_prob_mean": 0.9467695951461792,
      "gated_true_prob_mean": 0.9508653879165649,
      "delta_true_prob_mean": 0.004095756448805332
    },
    {
      "switch_type": "both_wrong_same",
      "count": 871,
      "gate_mean": 0.29384860396385193,
      "gate_median": 0.3329315781593323,
      "gate_std": 0.18096676468849182,
      "base_true_prob_mean": 0.18062709271907806,
      "gated_true_prob_mean": 0.1740313172340393,
      "delta_true_prob_mean": -0.006595781072974205
    },
    {
      "switch_type": "damaged",
      "count": 377,
      "gate_mean": 0.35566064715385437,
      "gate_median": 0.4021887183189392,
      "gate_std": 0.16292165219783783,
      "base_true_prob_mean": 0.7010312080383301,
      "gated_true_prob_mean": 0.24824467301368713,
      "delta_true_prob_mean": -0.45278650522232056
    },
    {
      "switch_type": "fixed",
      "count": 370,
      "gate_mean": 0.31889981031417847,
      "gate_median": 0.35985463857650757,
      "gate_std": 0.17921124398708344,
      "base_true_prob_mean": 0.2670670449733734,
      "gated_true_prob_mean": 0.707888126373291,
      "delta_true_prob_mean": 0.4408210515975952
    },
    {
      "switch_type": "both_wrong_changed",
      "count": 137,
      "gate_mean": 0.3629164397716522,
      "gate_median": 0.4323861598968506,
      "gate_std": 0.16910915076732635,
      "base_true_prob_mean": 0.15011556446552277,
      "gated_true_prob_mean": 0.1512145847082138,
      "delta_true_prob_mean": 0.001099028973840177
    }
  ],
  "build_infos": {
    "baseline": {
      "continuous_info": {
        "source": "raw_scaled",
        "train_path": "/kaggle/working/src_baocao/05_test/outputs/F3e0_gated_hardneg_calibration/_split_artifacts/train_inner_raw.csv",
        "val_path": "/kaggle/working/src_baocao/05_test/outputs/F3e0_gated_hardneg_calibration/_split_artifacts/calibration_raw.csv",
        "scale": "train_only_minmax_linear_clip_val",
        "n_constant_features": 3,
        "constant_features": [
          "pslist.nprocs64bit",
          "handles.nport",
          "svcscan.interactive_process_services"
        ],
        "train_min": 0.0,
        "train_max": 1.0,
        "val_min": 0.0,
        "val_max": 1.0
      },
      "missing": [],
      "unexpected": []
    },
    "gated": {
      "continuous_info": {
        "source": "raw_scaled",
        "train_path": "/kaggle/working/src_baocao/05_test/outputs/F3e0_gated_hardneg_calibration/_split_artifacts/train_inner_raw.csv",
        "val_path": "/kaggle/working/src_baocao/05_test/outputs/F3e0_gated_hardneg_calibration/_split_artifacts/calibration_raw.csv",
        "scale": "train_only_minmax_linear_clip_val",
        "n_constant_features": 3,
        "constant_features": [
          "pslist.nprocs64bit",
          "handles.nport",
          "svcscan.interactive_process_services"
        ],
        "train_min": 0.0,
        "train_max": 1.0,
        "val_min": 0.0,
        "val_max": 1.0
      },
      "missing": [],
      "unexpected": []
    }
  },
  "interpretation_hint": "If gated gate is not much higher on fixed/hard samples than damaged/easy samples, the learned gate is not selective enough. If Ransomware margins vs Trojan/Spyware fall while Trojan vs Ransomware rises, boundary moved asymmetrically and overcorrected."
}
```