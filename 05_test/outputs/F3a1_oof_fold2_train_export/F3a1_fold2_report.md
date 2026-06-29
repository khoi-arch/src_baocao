# F3a1 fold 2 clean OOF train/export

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
  "fold_id": 2,
  "oof_n": 9375,
  "inner_train_n": 31875,
  "inner_val_n": 5626,
  "train_pool_n": 37501,
  "oof_accuracy": 0.85056,
  "oof_macro_f1": 0.7751028880127867,
  "oof_weighted_f1": 0.849998243215775
}
```

## Hard pair/family summary

| true_label   | pred_label   | family               |   count |   mean_true_prob |   mean_pred_prob |   true_in_top2_rate |
|:-------------|:-------------|:---------------------|--------:|-----------------:|-----------------:|--------------------:|
| Trojan       | Ransomware   | Trojan-Zeus          |     126 |         0.243482 |         0.723578 |            0.865079 |
| Trojan       | Ransomware   | Trojan-Emotet        |      80 |         0.216188 |         0.731592 |            0.8125   |
| Spyware      | Trojan       | Spyware-180solutions |      71 |         0.12812  |         0.770235 |            0.647887 |
| Ransomware   | Spyware      | Ransomware-Ako       |      70 |         0.106214 |         0.839098 |            0.8      |
| Ransomware   | Spyware      | Ransomware-Conti     |      66 |         0.233075 |         0.660091 |            0.787879 |
| Ransomware   | Trojan       | Ransomware-Conti     |      63 |         0.233247 |         0.714446 |            0.809524 |
| Spyware      | Ransomware   | Spyware-180solutions |      62 |         0.115554 |         0.75549  |            0.419355 |
| Trojan       | Spyware      | Trojan-Scar          |      61 |         0.1541   |         0.738423 |            0.540984 |
| Trojan       | Spyware      | Trojan-Zeus          |      61 |         0.118462 |         0.711143 |            0.262295 |
| Ransomware   | Trojan       | Ransomware-Shade     |      61 |         0.189617 |         0.736431 |            0.655738 |
| Ransomware   | Trojan       | Ransomware-Ako       |      55 |         0.184231 |         0.773683 |            0.763636 |
| Ransomware   | Trojan       | Ransomware-Pysa      |      53 |         0.256134 |         0.699032 |            0.849057 |
| Trojan       | Ransomware   | Trojan-Scar          |      50 |         0.218137 |         0.72555  |            0.86     |
| Spyware      | Ransomware   | Spyware-Transponder  |      42 |         0.138818 |         0.804588 |            0.619048 |
| Ransomware   | Spyware      | Ransomware-Maze      |      42 |         0.184603 |         0.751513 |            0.904762 |
| Ransomware   | Spyware      | Ransomware-Shade     |      42 |         0.183714 |         0.703082 |            0.714286 |
| Spyware      | Ransomware   | Spyware-CWS          |      42 |         0.156685 |         0.737436 |            0.714286 |
| Trojan       | Spyware      | Trojan-Emotet        |      39 |         0.16915  |         0.729649 |            0.564103 |
| Ransomware   | Spyware      | Ransomware-Pysa      |      39 |         0.119515 |         0.808837 |            0.769231 |
| Ransomware   | Trojan       | Ransomware-Maze      |      38 |         0.218032 |         0.724784 |            0.684211 |

## Next

```text
If fold 0 output looks sane, run folds 1-4 with the same script.
Then concatenate all fold OOF predictions for F3a2 OOF overlap reproduction audit.
```