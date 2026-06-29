# F3a1 fold 4 clean OOF train/export

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
  "fold_id": 4,
  "oof_n": 9375,
  "inner_train_n": 31875,
  "inner_val_n": 5626,
  "train_pool_n": 37501,
  "oof_accuracy": 0.8445866666666667,
  "oof_macro_f1": 0.7661774174187623,
  "oof_weighted_f1": 0.8441896377992314
}
```

## Hard pair/family summary

| true_label   | pred_label   | family               |   count |   mean_true_prob |   mean_pred_prob |   true_in_top2_rate |
|:-------------|:-------------|:---------------------|--------:|-----------------:|-----------------:|--------------------:|
| Trojan       | Ransomware   | Trojan-Zeus          |     154 |        0.250228  |         0.726843 |            0.909091 |
| Trojan       | Ransomware   | Trojan-Emotet        |      98 |        0.207632  |         0.730392 |            0.857143 |
| Spyware      | Trojan       | Spyware-180solutions |      82 |        0.161856  |         0.69516  |            0.573171 |
| Trojan       | Spyware      | Trojan-Scar          |      72 |        0.187666  |         0.709522 |            0.722222 |
| Ransomware   | Trojan       | Ransomware-Shade     |      67 |        0.213224  |         0.627387 |            0.567164 |
| Spyware      | Ransomware   | Spyware-180solutions |      63 |        0.0853714 |         0.768841 |            0.365079 |
| Ransomware   | Trojan       | Ransomware-Pysa      |      62 |        0.286027  |         0.642938 |            0.870968 |
| Ransomware   | Spyware      | Ransomware-Ako       |      59 |        0.153338  |         0.762657 |            0.661017 |
| Trojan       | Ransomware   | Trojan-Scar          |      56 |        0.233444  |         0.691695 |            0.75     |
| Ransomware   | Trojan       | Ransomware-Conti     |      56 |        0.230786  |         0.639773 |            0.696429 |
| Trojan       | Spyware      | Trojan-Zeus          |      54 |        0.18564   |         0.635793 |            0.481481 |
| Ransomware   | Trojan       | Ransomware-Ako       |      52 |        0.215244  |         0.674008 |            0.615385 |
| Ransomware   | Spyware      | Ransomware-Conti     |      46 |        0.194146  |         0.635513 |            0.586957 |
| Spyware      | Ransomware   | Spyware-CWS          |      45 |        0.147137  |         0.733209 |            0.555556 |
| Spyware      | Trojan       | Spyware-CWS          |      43 |        0.185734  |         0.6757   |            0.627907 |
| Trojan       | Ransomware   | Trojan-Reconyc       |      41 |        0.185335  |         0.718491 |            0.707317 |
| Ransomware   | Spyware      | Ransomware-Shade     |      40 |        0.194924  |         0.656813 |            0.675    |
| Ransomware   | Trojan       | Ransomware-Maze      |      37 |        0.2908    |         0.635645 |            0.864865 |
| Ransomware   | Spyware      | Ransomware-Maze      |      36 |        0.167104  |         0.716858 |            0.666667 |
| Trojan       | Spyware      | Trojan-Refroso       |      31 |        0.21799   |         0.657876 |            0.709677 |

## Next

```text
If fold 0 output looks sane, run folds 1-4 with the same script.
Then concatenate all fold OOF predictions for F3a2 OOF overlap reproduction audit.
```