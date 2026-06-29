# F3a0 OOF fold split audit

## Purpose

```text
Create deterministic K-fold assignments from original training split.
No training. No validation usage. This is the clean base for OOF mining.
```

## Integrity

```json
{
  "n_rows": 46876,
  "dataset_y_train_len": 46876,
  "n_splits": 5,
  "seed": 42,
  "l2_col": "label_L2",
  "l3_col": "label_L3",
  "stratification_mode": "L2_plus_L3",
  "fold_sizes": {
    "0": 9376,
    "1": 9375,
    "2": 9375,
    "3": 9375,
    "4": 9375
  },
  "min_fold_size": 9375,
  "max_fold_size": 9376,
  "max_abs_l2_pct_diff": 0.00012435190715931466,
  "max_abs_l3_pct_diff": 8.604602212930468e-05,
  "row_id_unique": true,
  "all_rows_assigned_once": true
}
```

## Fold sizes

|   fold |   fold_size |      pct |
|-------:|------------:|---------:|
|      0 |        9376 | 0.200017 |
|      1 |        9375 | 0.199996 |
|      2 |        9375 | 0.199996 |
|      3 |        9375 | 0.199996 |
|      4 |        9375 | 0.199996 |

## L2 fold distribution

|   fold | label_col   | label      |   count |   fold_size |   fold_pct |   global_count |   global_pct |   abs_pct_diff |
|-------:|:------------|:-----------|--------:|------------:|-----------:|---------------:|-------------:|---------------:|
|      0 | label_L2    | Benign     |    4688 |        9376 |   0.5      |          23438 |     0.5      |    0           |
|      1 | label_L2    | Benign     |    4688 |        9375 |   0.500053 |          23438 |     0.5      |    5.33333e-05 |
|      2 | label_L2    | Benign     |    4688 |        9375 |   0.500053 |          23438 |     0.5      |    5.33333e-05 |
|      3 | label_L2    | Benign     |    4687 |        9375 |   0.499947 |          23438 |     0.5      |    5.33333e-05 |
|      4 | label_L2    | Benign     |    4687 |        9375 |   0.499947 |          23438 |     0.5      |    5.33333e-05 |
|      0 | label_L2    | Ransomware |    1567 |        9376 |   0.167129 |           7832 |     0.167079 |    4.97373e-05 |
|      1 | label_L2    | Ransomware |    1566 |        9375 |   0.16704  |           7832 |     0.167079 |    3.91023e-05 |
|      2 | label_L2    | Ransomware |    1567 |        9375 |   0.167147 |           7832 |     0.167079 |    6.75644e-05 |
|      3 | label_L2    | Ransomware |    1566 |        9375 |   0.16704  |           7832 |     0.167079 |    3.91023e-05 |
|      4 | label_L2    | Ransomware |    1566 |        9375 |   0.16704  |           7832 |     0.167079 |    3.91023e-05 |
|      0 | label_L2    | Spyware    |    1603 |        9376 |   0.170968 |           8016 |     0.171004 |    3.59219e-05 |
|      1 | label_L2    | Spyware    |    1602 |        9375 |   0.17088  |           8016 |     0.171004 |    0.000124352 |
|      2 | label_L2    | Spyware    |    1603 |        9375 |   0.170987 |           8016 |     0.171004 |    1.76852e-05 |
|      3 | label_L2    | Spyware    |    1604 |        9375 |   0.171093 |           8016 |     0.171004 |    8.89814e-05 |
|      4 | label_L2    | Spyware    |    1604 |        9375 |   0.171093 |           8016 |     0.171004 |    8.89814e-05 |
|      0 | label_L2    | Trojan     |    1518 |        9376 |   0.161903 |           7590 |     0.161917 |    1.38154e-05 |
|      1 | label_L2    | Trojan     |    1519 |        9375 |   0.162027 |           7590 |     0.161917 |    0.000110121 |
|      2 | label_L2    | Trojan     |    1517 |        9375 |   0.161813 |           7590 |     0.161917 |    0.000103212 |
|      3 | label_L2    | Trojan     |    1518 |        9375 |   0.16192  |           7590 |     0.161917 |    3.45422e-06 |
|      4 | label_L2    | Trojan     |    1518 |        9375 |   0.16192  |           7590 |     0.161917 |    3.45422e-06 |

## Hardest/rarest strata

| stratum                       |   count |
|:------------------------------|--------:|
| Spyware::Spyware-TIBS         |    1128 |
| Trojan::Trojan-Reconyc        |    1256 |
| Ransomware::Ransomware-Pysa   |    1374 |
| Trojan::Trojan-Zeus           |    1560 |
| Ransomware::Ransomware-Maze   |    1566 |
| Trojan::Trojan-Emotet         |    1574 |
| Ransomware::Ransomware-Conti  |    1590 |
| Trojan::Trojan-Scar           |    1600 |
| Spyware::Spyware-180solutions |    1600 |
| Spyware::Spyware-CWS          |    1600 |
| Ransomware::Ransomware-Ako    |    1600 |
| Trojan::Trojan-Refroso        |    1600 |
| Ransomware::Ransomware-Shade  |    1702 |
| Spyware::Spyware-Gator        |    1760 |
| Spyware::Spyware-Transponder  |    1928 |
| Benign::Benign                |   23438 |

## Next step

```text
F3a1 will train one official L1 model per fold:
  train_idx = fold != i
  oof_idx   = fold == i
Then export OOF logits/probs/CLS for train-only hard-pair mining.
```