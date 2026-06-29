# F3a1 fold 3 clean OOF train/export

## Protocol

```text
outer_oof_idx = fold == fold_id
train_pool_idx = fold != fold_id
inner_train/inner_val split is made only from train_pool
official L1 trains on inner_train and early-stops on inner_val
outer_oof_idx is used only for final OOF prediction export
```

## Metrics

```json
{
  "fold_id": 3,
  "oof_n": 9375,
  "inner_train_n": 31875,
  "inner_val_n": 5626,
  "train_pool_n": 37501,
  "oof_accuracy": 0.8536533333333334,
  "oof_macro_f1": 0.7790087358726492,
  "oof_weighted_f1": 0.8527998378976663
}
```

## Hard pair/family summary

| true_label   | pred_label   | family               |   count |   mean_true_prob |   mean_pred_prob |   true_in_top2_rate |
|:-------------|:-------------|:---------------------|--------:|-----------------:|-----------------:|--------------------:|
| Trojan       | Ransomware   | Trojan-Zeus          |     164 |        0.285868  |         0.688254 |            0.920732 |
| Trojan       | Ransomware   | Trojan-Emotet        |      84 |        0.220412  |         0.725871 |            0.77381  |
| Spyware      | Trojan       | Spyware-180solutions |      82 |        0.107248  |         0.788172 |            0.658537 |
| Ransomware   | Spyware      | Ransomware-Ako       |      68 |        0.10542   |         0.819101 |            0.661765 |
| Trojan       | Ransomware   | Trojan-Scar          |      65 |        0.200319  |         0.756801 |            0.769231 |
| Ransomware   | Spyware      | Ransomware-Shade     |      59 |        0.15293   |         0.742483 |            0.644068 |
| Ransomware   | Trojan       | Ransomware-Conti     |      56 |        0.204292  |         0.722359 |            0.714286 |
| Ransomware   | Spyware      | Ransomware-Conti     |      52 |        0.185099  |         0.703106 |            0.673077 |
| Trojan       | Spyware      | Trojan-Scar          |      51 |        0.12852   |         0.786717 |            0.54902  |
| Ransomware   | Trojan       | Ransomware-Shade     |      51 |        0.220501  |         0.709662 |            0.764706 |
| Spyware      | Ransomware   | Spyware-180solutions |      48 |        0.0995234 |         0.739141 |            0.333333 |
| Ransomware   | Trojan       | Ransomware-Ako       |      48 |        0.184772  |         0.774168 |            0.729167 |
| Ransomware   | Spyware      | Ransomware-Pysa      |      48 |        0.175967  |         0.753076 |            0.854167 |
| Trojan       | Spyware      | Trojan-Zeus          |      42 |        0.127819  |         0.731441 |            0.357143 |
| Ransomware   | Trojan       | Ransomware-Maze      |      40 |        0.300087  |         0.660444 |            0.925    |
| Trojan       | Spyware      | Trojan-Emotet        |      40 |        0.133266  |         0.728593 |            0.475    |
| Ransomware   | Spyware      | Ransomware-Maze      |      37 |        0.172373  |         0.73305  |            0.810811 |
| Ransomware   | Trojan       | Ransomware-Pysa      |      37 |        0.225158  |         0.733408 |            0.756757 |
| Trojan       | Ransomware   | Trojan-Reconyc       |      36 |        0.153522  |         0.773614 |            0.777778 |
| Spyware      | Trojan       | Spyware-Gator        |      32 |        0.219581  |         0.675594 |            0.75     |

## Next

```text
If fold 0 output looks sane, run folds 1-4 with the same script.
Then concatenate all fold OOF predictions for F3a2 OOF overlap reproduction audit.
```