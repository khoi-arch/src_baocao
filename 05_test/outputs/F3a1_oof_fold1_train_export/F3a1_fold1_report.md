# F3a1 fold 1 clean OOF train/export

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
  "fold_id": 1,
  "oof_n": 9375,
  "inner_train_n": 31875,
  "inner_val_n": 5626,
  "train_pool_n": 37501,
  "oof_accuracy": 0.8469333333333333,
  "oof_macro_f1": 0.7692572593403987,
  "oof_weighted_f1": 0.8461621690393787
}
```

## Hard pair/family summary

| true_label   | pred_label   | family               |   count |   mean_true_prob |   mean_pred_prob |   true_in_top2_rate |
|:-------------|:-------------|:---------------------|--------:|-----------------:|-----------------:|--------------------:|
| Trojan       | Ransomware   | Trojan-Zeus          |     159 |        0.301526  |         0.676811 |            0.955975 |
| Trojan       | Ransomware   | Trojan-Emotet        |      83 |        0.237524  |         0.71918  |            0.891566 |
| Ransomware   | Trojan       | Ransomware-Shade     |      77 |        0.191469  |         0.659125 |            0.519481 |
| Ransomware   | Spyware      | Ransomware-Ako       |      76 |        0.116916  |         0.810825 |            0.75     |
| Spyware      | Trojan       | Spyware-180solutions |      73 |        0.172484  |         0.723992 |            0.657534 |
| Ransomware   | Spyware      | Ransomware-Shade     |      55 |        0.174879  |         0.688778 |            0.618182 |
| Ransomware   | Spyware      | Ransomware-Conti     |      55 |        0.186718  |         0.654287 |            0.6      |
| Ransomware   | Trojan       | Ransomware-Conti     |      53 |        0.24412   |         0.684893 |            0.811321 |
| Trojan       | Ransomware   | Trojan-Scar          |      52 |        0.282973  |         0.665378 |            0.865385 |
| Ransomware   | Spyware      | Ransomware-Maze      |      51 |        0.163539  |         0.744158 |            0.764706 |
| Ransomware   | Spyware      | Ransomware-Pysa      |      51 |        0.137031  |         0.753811 |            0.666667 |
| Trojan       | Spyware      | Trojan-Scar          |      51 |        0.17724   |         0.738038 |            0.607843 |
| Ransomware   | Trojan       | Ransomware-Ako       |      50 |        0.168776  |         0.736626 |            0.64     |
| Spyware      | Ransomware   | Spyware-180solutions |      49 |        0.0978068 |         0.766563 |            0.408163 |
| Ransomware   | Trojan       | Ransomware-Pysa      |      49 |        0.210108  |         0.719041 |            0.714286 |
| Trojan       | Spyware      | Trojan-Zeus          |      48 |        0.123723  |         0.702269 |            0.291667 |
| Spyware      | Trojan       | Spyware-CWS          |      47 |        0.208704  |         0.703916 |            0.787234 |
| Ransomware   | Trojan       | Ransomware-Maze      |      41 |        0.236979  |         0.686725 |            0.707317 |
| Trojan       | Spyware      | Trojan-Emotet        |      35 |        0.181284  |         0.666308 |            0.542857 |
| Trojan       | Spyware      | Trojan-Reconyc       |      34 |        0.192747  |         0.69206  |            0.558824 |

## Next

```text
If fold 0 output looks sane, run folds 1-4 with the same script.
Then concatenate all fold OOF predictions for F3a2 OOF overlap reproduction audit.
```