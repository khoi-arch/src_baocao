# F3a2 Aggregate OOF overlap reproduction audit

## Scope

```text
Train-only OOF audit. Official validation is not used.
Each row is predicted once by a model that did not train on that row.
Outer fold is not used for early stopping in F3a1.
```

## Integrity

```json
{
  "n_rows": 46876,
  "unique_original_row_id": 46876,
  "duplicate_original_row_id_count": 0,
  "expected_n": 46876,
  "missing_original_row_id_count": 0,
  "min_original_row_id": 0,
  "max_original_row_id": 46875,
  "folds": [
    0,
    1,
    2,
    3,
    4
  ],
  "fold_counts": {
    "0": 9376,
    "1": 9375,
    "2": 9375,
    "3": 9375,
    "4": 9375
  },
  "pass": true
}
```

## Overall OOF metrics

```json
{
  "n": 46876,
  "accuracy": 0.8501791961771482,
  "macro_f1": 0.7743309273667882,
  "weighted_f1": 0.849582113438872
}
```

## Fold metrics

|    n |   accuracy |   macro_f1 |   weighted_f1 |   fold |
|-----:|-----------:|-----------:|--------------:|-------:|
| 9376 |   0.855162 |   0.781891 |      0.854616 |      0 |
| 9375 |   0.846933 |   0.769257 |      0.846162 |      1 |
| 9375 |   0.85056  |   0.775103 |      0.849998 |      2 |
| 9375 |   0.853653 |   0.779009 |      0.8528   |      3 |
| 9375 |   0.844587 |   0.766177 |      0.84419  |      4 |

## Hardest families from train-only OOF

| family               | true_label   |   support |   correct |   accuracy |   error_rate | top_pred   |   top_pred_count |   mean_true_prob |   mean_pred_prob |   true_in_top2_rate |
|:---------------------|:-------------|----------:|----------:|-----------:|-------------:|:-----------|-----------------:|-----------------:|-----------------:|--------------------:|
| Trojan-Zeus          | Trojan       |      1560 |       541 |   0.346795 |  0.653205    | Ransomware |              751 |         0.424868 |         0.725367 |            0.84359  |
| Spyware-180solutions | Spyware      |      1600 |       948 |   0.5925   |  0.4075      | Spyware    |              948 |         0.545135 |         0.80395  |            0.806875 |
| Trojan-Emotet        | Trojan       |      1574 |       959 |   0.609276 |  0.390724    | Trojan     |              959 |         0.583261 |         0.785383 |            0.90216  |
| Ransomware-Ako       | Ransomware   |      1600 |      1015 |   0.634375 |  0.365625    | Ransomware |             1015 |         0.592016 |         0.826756 |            0.88875  |
| Trojan-Scar          | Trojan       |      1600 |      1036 |   0.6475   |  0.3525      | Trojan     |             1036 |         0.62368  |         0.810723 |            0.90375  |
| Ransomware-Conti     | Ransomware   |      1590 |      1042 |   0.655346 |  0.344654    | Ransomware |             1042 |         0.612859 |         0.778508 |            0.901258 |
| Ransomware-Pysa      | Ransomware   |      1374 |       919 |   0.66885  |  0.33115     | Ransomware |              919 |         0.618563 |         0.790635 |            0.923581 |
| Ransomware-Shade     | Ransomware   |      1702 |      1141 |   0.670388 |  0.329612    | Ransomware |             1141 |         0.618183 |         0.786195 |            0.883666 |
| Ransomware-Maze      | Ransomware   |      1566 |      1166 |   0.744572 |  0.255428    | Ransomware |             1166 |         0.696165 |         0.821618 |            0.954023 |
| Spyware-CWS          | Spyware      |      1600 |      1203 |   0.751875 |  0.248125    | Spyware    |             1203 |         0.667328 |         0.819304 |            0.911875 |
| Trojan-Reconyc       | Trojan       |      1256 |       966 |   0.769108 |  0.230892    | Trojan     |              966 |         0.72518  |         0.851555 |            0.938694 |
| Spyware-TIBS         | Spyware      |      1128 |       949 |   0.841312 |  0.158688    | Spyware    |              949 |         0.771796 |         0.866462 |            0.932624 |
| Spyware-Transponder  | Spyware      |      1928 |      1651 |   0.856328 |  0.143672    | Spyware    |             1651 |         0.77336  |         0.863741 |            0.952282 |
| Trojan-Refroso       | Trojan       |      1600 |      1376 |   0.86     |  0.14        | Trojan     |             1376 |         0.799739 |         0.870621 |            0.96125  |
| Spyware-Gator        | Spyware      |      1760 |      1516 |   0.861364 |  0.138636    | Spyware    |             1516 |         0.763034 |         0.827472 |            0.955682 |
| Benign               | Benign       |     23438 |     23425 |   0.999445 |  0.000554655 | Benign     |            23425 |         0.999436 |         0.999788 |            0.999787 |

## Malware pair confusion from train-only OOF

| true_label   | pred_label   |   count |   true_support |   rate_within_true |   mean_true_prob |   mean_pred_prob |   true_in_top2_rate | top_family           |   top_family_count |
|:-------------|:-------------|--------:|---------------:|-------------------:|-----------------:|-----------------:|--------------------:|:---------------------|-------------------:|
| Trojan       | Ransomware   |    1709 |           7590 |           0.225165 |         0.245455 |         0.710196 |            0.867759 | Trojan-Zeus          |                751 |
| Ransomware   | Trojan       |    1281 |           7832 |           0.16356  |         0.219208 |         0.703999 |            0.723653 | Ransomware-Shade     |                308 |
| Ransomware   | Spyware      |    1265 |           7832 |           0.161517 |         0.16055  |         0.741462 |            0.720949 | Ransomware-Ako       |                335 |
| Trojan       | Spyware      |     989 |           7590 |           0.130303 |         0.159168 |         0.717532 |            0.537917 | Trojan-Scar          |                286 |
| Spyware      | Trojan       |     909 |           8016 |           0.113398 |         0.163434 |         0.729613 |            0.665567 | Spyware-180solutions |                374 |
| Spyware      | Ransomware   |     817 |           8016 |           0.101921 |         0.124909 |         0.759482 |            0.522644 | Spyware-180solutions |                277 |

## Hard pair-family groups

| true_label   | pred_label   | family               |   count |   mean_true_prob |   mean_pred_prob |   true_in_top2_rate |   folds_present |
|:-------------|:-------------|:---------------------|--------:|-----------------:|-----------------:|--------------------:|----------------:|
| Trojan       | Ransomware   | Trojan-Zeus          |     751 |        0.273572  |         0.701244 |            0.91478  |               5 |
| Trojan       | Ransomware   | Trojan-Emotet        |     434 |        0.222535  |         0.723345 |            0.845622 |               5 |
| Spyware      | Trojan       | Spyware-180solutions |     374 |        0.136195  |         0.753595 |            0.63369  |               5 |
| Ransomware   | Spyware      | Ransomware-Ako       |     335 |        0.11365   |         0.820692 |            0.728358 |               5 |
| Ransomware   | Trojan       | Ransomware-Shade     |     308 |        0.201493  |         0.688878 |            0.633117 |               5 |
| Trojan       | Spyware      | Trojan-Scar          |     286 |        0.162271  |         0.744218 |            0.636364 |               5 |
| Ransomware   | Trojan       | Ransomware-Conti     |     278 |        0.218448  |         0.70551  |            0.744604 |               5 |
| Trojan       | Ransomware   | Trojan-Scar          |     278 |        0.233352  |         0.711164 |            0.820144 |               5 |
| Spyware      | Ransomware   | Spyware-180solutions |     277 |        0.0995499 |         0.757535 |            0.379061 |               5 |
| Ransomware   | Spyware      | Ransomware-Conti     |     269 |        0.199778  |         0.672959 |            0.684015 |               5 |
| Trojan       | Spyware      | Trojan-Zeus          |     266 |        0.139541  |         0.687026 |            0.330827 |               5 |
| Ransomware   | Spyware      | Ransomware-Shade     |     253 |        0.171805  |         0.708728 |            0.664032 |               5 |
| Ransomware   | Trojan       | Ransomware-Ako       |     250 |        0.183629  |         0.738528 |            0.652    |               5 |
| Ransomware   | Trojan       | Ransomware-Pysa      |     248 |        0.239127  |         0.705883 |            0.802419 |               5 |
| Ransomware   | Spyware      | Ransomware-Pysa      |     205 |        0.159226  |         0.741797 |            0.736585 |               5 |
| Ransomware   | Spyware      | Ransomware-Maze      |     203 |        0.173275  |         0.74195  |            0.812808 |               5 |
| Spyware      | Trojan       | Spyware-CWS          |     197 |        0.156367  |         0.749027 |            0.680203 |               5 |
| Ransomware   | Trojan       | Ransomware-Maze      |     197 |        0.268051  |         0.679315 |            0.827411 |               5 |
| Spyware      | Ransomware   | Spyware-CWS          |     192 |        0.133943  |         0.757932 |            0.598958 |               5 |
| Trojan       | Spyware      | Trojan-Emotet        |     176 |        0.160297  |         0.709212 |            0.517045 |               5 |

## Selected hard groups for next clean method design

```json
{
  "selection_source": "train_only_clean_OOF",
  "validation_used": false,
  "criteria": {
    "min_family_support": 500,
    "min_family_error_rate": 0.3,
    "min_pair_count": 700,
    "min_pair_rate": 0.08,
    "min_pair_family_count": 150
  },
  "hard_families": [
    {
      "family": "Trojan-Zeus",
      "true_label": "Trojan",
      "support": 1560,
      "accuracy": 0.3467948717948718,
      "error_rate": 0.6532051282051282,
      "top_pred": "Ransomware",
      "top_pred_count": 751,
      "true_in_top2_rate": 0.8435897435897436
    },
    {
      "family": "Spyware-180solutions",
      "true_label": "Spyware",
      "support": 1600,
      "accuracy": 0.5925,
      "error_rate": 0.4075,
      "top_pred": "Spyware",
      "top_pred_count": 948,
      "true_in_top2_rate": 0.806875
    },
    {
      "family": "Trojan-Emotet",
      "true_label": "Trojan",
      "support": 1574,
      "accuracy": 0.6092757306226175,
      "error_rate": 0.39072426937738247,
      "top_pred": "Trojan",
      "top_pred_count": 959,
      "true_in_top2_rate": 0.9021601016518425
    },
    {
      "family": "Ransomware-Ako",
      "true_label": "Ransomware",
      "support": 1600,
      "accuracy": 0.634375,
      "error_rate": 0.365625,
      "top_pred": "Ransomware",
      "top_pred_count": 1015,
      "true_in_top2_rate": 0.88875
    },
    {
      "family": "Trojan-Scar",
      "true_label": "Trojan",
      "support": 1600,
      "accuracy": 0.6475,
      "error_rate": 0.35250000000000004,
      "top_pred": "Trojan",
      "top_pred_count": 1036,
      "true_in_top2_rate": 0.90375
    },
    {
      "family": "Ransomware-Conti",
      "true_label": "Ransomware",
      "support": 1590,
      "accuracy": 0.6553459119496855,
      "error_rate": 0.3446540880503145,
      "top_pred": "Ransomware",
      "top_pred_count": 1042,
      "true_in_top2_rate": 0.9012578616352201
    },
    {
      "family": "Ransomware-Pysa",
      "true_label": "Ransomware",
      "support": 1374,
      "accuracy": 0.6688500727802038,
      "error_rate": 0.3311499272197962,
      "top_pred": "Ransomware",
      "top_pred_count": 919,
      "true_in_top2_rate": 0.9235807860262009
    },
    {
      "family": "Ransomware-Shade",
      "true_label": "Ransomware",
      "support": 1702,
      "accuracy": 0.6703877790834313,
      "error_rate": 0.3296122209165687,
      "top_pred": "Ransomware",
      "top_pred_count": 1141,
      "true_in_top2_rate": 0.8836662749706228
    }
  ],
  "hard_pairs": [
    {
      "true_label": "Trojan",
      "pred_label": "Ransomware",
      "count": 1709,
      "true_support": 7590,
      "rate_within_true": 0.2251646903820817,
      "mean_true_prob": 0.24545525535006385,
      "mean_pred_prob": 0.7101958061088356,
      "true_in_top2_rate": 0.8677589233469866,
      "top_family": "Trojan-Zeus",
      "top_family_count": 751
    },
    {
      "true_label": "Ransomware",
      "pred_label": "Trojan",
      "count": 1281,
      "true_support": 7832,
      "rate_within_true": 0.16355975485188967,
      "mean_true_prob": 0.21920777987794618,
      "mean_pred_prob": 0.7039985585480094,
      "true_in_top2_rate": 0.7236533957845434,
      "top_family": "Ransomware-Shade",
      "top_family_count": 308
    },
    {
      "true_label": "Ransomware",
      "pred_label": "Spyware",
      "count": 1265,
      "true_support": 7832,
      "rate_within_true": 0.16151685393258428,
      "mean_true_prob": 0.16054994518890872,
      "mean_pred_prob": 0.7414624913201581,
      "true_in_top2_rate": 0.7209486166007905,
      "top_family": "Ransomware-Ako",
      "top_family_count": 335
    },
    {
      "true_label": "Trojan",
      "pred_label": "Spyware",
      "count": 989,
      "true_support": 7590,
      "rate_within_true": 0.1303030303030303,
      "mean_true_prob": 0.15916756233509288,
      "mean_pred_prob": 0.7175320070171891,
      "true_in_top2_rate": 0.537917087967644,
      "top_family": "Trojan-Scar",
      "top_family_count": 286
    },
    {
      "true_label": "Spyware",
      "pred_label": "Trojan",
      "count": 909,
      "true_support": 8016,
      "rate_within_true": 0.11339820359281437,
      "mean_true_prob": 0.16343406935404328,
      "mean_pred_prob": 0.7296133588558856,
      "true_in_top2_rate": 0.6655665566556656,
      "top_family": "Spyware-180solutions",
      "top_family_count": 374
    },
    {
      "true_label": "Spyware",
      "pred_label": "Ransomware",
      "count": 817,
      "true_support": 8016,
      "rate_within_true": 0.10192115768463074,
      "mean_true_prob": 0.12490915138875078,
      "mean_pred_prob": 0.7594824522888617,
      "true_in_top2_rate": 0.5226438188494492,
      "top_family": "Spyware-180solutions",
      "top_family_count": 277
    }
  ],
  "hard_pair_families": [
    {
      "true_label": "Trojan",
      "pred_label": "Ransomware",
      "family": "Trojan-Zeus",
      "count": 751,
      "mean_true_prob": 0.27357162742101065,
      "mean_pred_prob": 0.7012435933155792,
      "true_in_top2_rate": 0.914780292942743,
      "folds_present": 5
    },
    {
      "true_label": "Trojan",
      "pred_label": "Ransomware",
      "family": "Trojan-Emotet",
      "count": 434,
      "mean_true_prob": 0.22253463955450484,
      "mean_pred_prob": 0.7233454213133641,
      "true_in_top2_rate": 0.8456221198156681,
      "folds_present": 5
    },
    {
      "true_label": "Spyware",
      "pred_label": "Trojan",
      "family": "Spyware-180solutions",
      "count": 374,
      "mean_true_prob": 0.13619529546386017,
      "mean_pred_prob": 0.7535950284224598,
      "true_in_top2_rate": 0.6336898395721925,
      "folds_present": 5
    },
    {
      "true_label": "Ransomware",
      "pred_label": "Spyware",
      "family": "Ransomware-Ako",
      "count": 335,
      "mean_true_prob": 0.11364968514899851,
      "mean_pred_prob": 0.8206920642985075,
      "true_in_top2_rate": 0.7283582089552239,
      "folds_present": 5
    },
    {
      "true_label": "Ransomware",
      "pred_label": "Trojan",
      "family": "Ransomware-Shade",
      "count": 308,
      "mean_true_prob": 0.201493467236837,
      "mean_pred_prob": 0.6888784303571429,
      "true_in_top2_rate": 0.6331168831168831,
      "folds_present": 5
    },
    {
      "true_label": "Trojan",
      "pred_label": "Spyware",
      "family": "Trojan-Scar",
      "count": 286,
      "mean_true_prob": 0.16227101485107273,
      "mean_pred_prob": 0.7442181442657343,
      "true_in_top2_rate": 0.6363636363636364,
      "folds_present": 5
    },
    {
      "true_label": "Ransomware",
      "pred_label": "Trojan",
      "family": "Ransomware-Conti",
      "count": 278,
      "mean_true_prob": 0.21844752461205144,
      "mean_pred_prob": 0.7055097707194244,
      "true_in_top2_rate": 0.7446043165467626,
      "folds_present": 5
    },
    {
      "true_label": "Trojan",
      "pred_label": "Ransomware",
      "family": "Trojan-Scar",
      "count": 278,
      "mean_true_prob": 0.23335177610919064,
      "mean_pred_prob": 0.7111639900719425,
      "true_in_top2_rate": 0.8201438848920863,
      "folds_present": 5
    },
    {
      "true_label": "Spyware",
      "pred_label": "Ransomware",
      "family": "Spyware-180solutions",
      "count": 277,
      "mean_true_prob": 0.09954993810134476,
      "mean_pred_prob": 0.7575350391696751,
      "true_in_top2_rate": 0.37906137184115524,
      "folds_present": 5
    },
    {
      "true_label": "Ransomware",
      "pred_label": "Spyware",
      "family": "Ransomware-Conti",
      "count": 269,
      "mean_true_prob": 0.1997775775149591,
      "mean_pred_prob": 0.6729587718959108,
      "true_in_top2_rate": 0.6840148698884758,
      "folds_present": 5
    },
    {
      "true_label": "Trojan",
      "pred_label": "Spyware",
      "family": "Trojan-Zeus",
      "count": 266,
      "mean_true_prob": 0.13954067253861327,
      "mean_pred_prob": 0.6870256778195489,
      "true_in_top2_rate": 0.3308270676691729,
      "folds_present": 5
    },
    {
      "true_label": "Ransomware",
      "pred_label": "Spyware",
      "family": "Ransomware-Shade",
      "count": 253,
      "mean_true_prob": 0.17180493510532016,
      "mean_pred_prob": 0.708727966916996,
      "true_in_top2_rate": 0.6640316205533597,
      "folds_present": 5
    },
    {
      "true_label": "Ransomware",
      "pred_label": "Trojan",
      "family": "Ransomware-Ako",
      "count": 250,
      "mean_true_prob": 0.183629231481224,
      "mean_pred_prob": 0.73852775404,
      "true_in_top2_rate": 0.652,
      "folds_present": 5
    },
    {
      "true_label": "Ransomware",
      "pred_label": "Trojan",
      "family": "Ransomware-Pysa",
      "count": 248,
      "mean_true_prob": 0.2391265990405121,
      "mean_pred_prob": 0.7058827064919354,
      "true_in_top2_rate": 0.8024193548387096,
      "folds_present": 5
    },
    {
      "true_label": "Ransomware",
      "pred_label": "Spyware",
      "family": "Ransomware-Pysa",
      "count": 205,
      "mean_true_prob": 0.15922589784987806,
      "mean_pred_prob": 0.7417966724878049,
      "true_in_top2_rate": 0.7365853658536585,
      "folds_present": 5
    },
    {
      "true_label": "Ransomware",
      "pred_label": "Spyware",
      "family": "Ransomware-Maze",
      "count": 203,
      "mean_true_prob": 0.17327541944167485,
      "mean_pred_prob": 0.7419497874384237,
      "true_in_top2_rate": 0.812807881773399,
      "folds_present": 5
    },
    {
      "true_label": "Spyware",
      "pred_label": "Trojan",
      "family": "Spyware-CWS",
      "count": 197,
      "mean_true_prob": 0.1563672605715921,
      "mean_pred_prob": 0.7490265539593909,
      "true_in_top2_rate": 0.6802030456852792,
      "folds_present": 5
    },
    {
      "true_label": "Ransomware",
      "pred_label": "Trojan",
      "family": "Ransomware-Maze",
      "count": 197,
      "mean_true_prob": 0.26805107533096445,
      "mean_pred_prob": 0.6793148780203045,
      "true_in_top2_rate": 0.8274111675126904,
      "folds_present": 5
    },
    {
      "true_label": "Spyware",
      "pred_label": "Ransomware",
      "family": "Spyware-CWS",
      "count": 192,
      "mean_true_prob": 0.133943371084125,
      "mean_pred_prob": 0.7579315294270833,
      "true_in_top2_rate": 0.5989583333333334,
      "folds_present": 5
    },
    {
      "true_label": "Trojan",
      "pred_label": "Spyware",
      "family": "Trojan-Emotet",
      "count": 176,
      "mean_true_prob": 0.16029660910973068,
      "mean_pred_prob": 0.7092117917613636,
      "true_in_top2_rate": 0.5170454545454546,
      "folds_present": 5
    }
  ]
}
```

## Interpretation

```text
If these OOF hard families/pairs match the earlier validation audit, then the pattern is not a validation-only artifact.
The selected hard groups may be used to design a clean train-only repair method.
Do not use official validation to add/remove hard groups or tune thresholds.
```