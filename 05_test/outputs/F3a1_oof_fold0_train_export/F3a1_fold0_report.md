# F3a1 fold 0 clean OOF train/export

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
  "fold_id": 0,
  "oof_n": 9376,
  "inner_train_n": 31875,
  "inner_val_n": 5625,
  "train_pool_n": 37500,
  "oof_accuracy": 0.8551621160409556,
  "oof_macro_f1": 0.781890912223131,
  "oof_weighted_f1": 0.8546160258116394
}
```

## Hard pair/family summary

| true_label   | pred_label   | family               |   count |   mean_true_prob |   mean_pred_prob |   true_in_top2_rate |
|:-------------|:-------------|:---------------------|--------:|-----------------:|-----------------:|--------------------:|
| Trojan       | Ransomware   | Trojan-Zeus          |     148 |        0.279821  |         0.696234 |            0.912162 |
| Trojan       | Ransomware   | Trojan-Emotet        |      89 |        0.232674  |         0.709674 |            0.88764  |
| Spyware      | Trojan       | Spyware-180solutions |      66 |        0.108827  |         0.79808  |            0.636364 |
| Ransomware   | Spyware      | Ransomware-Ako       |      62 |        0.089299  |         0.868977 |            0.758065 |
| Trojan       | Spyware      | Trojan-Zeus          |      61 |        0.140328  |         0.665685 |            0.278689 |
| Ransomware   | Spyware      | Ransomware-Shade     |      57 |        0.163377  |         0.733629 |            0.684211 |
| Spyware      | Ransomware   | Spyware-180solutions |      55 |        0.0993259 |         0.754899 |            0.363636 |
| Trojan       | Ransomware   | Trojan-Scar          |      55 |        0.239213  |         0.707264 |            0.872727 |
| Ransomware   | Trojan       | Ransomware-Shade     |      52 |        0.196513  |         0.735998 |            0.730769 |
| Trojan       | Spyware      | Trojan-Scar          |      51 |        0.154976  |         0.763815 |            0.745098 |
| Spyware      | Trojan       | Spyware-CWS          |      50 |        0.122369  |         0.795049 |            0.6      |
| Ransomware   | Trojan       | Ransomware-Conti     |      50 |        0.174621  |         0.770859 |            0.68     |
| Ransomware   | Spyware      | Ransomware-Conti     |      50 |        0.190638  |         0.713581 |            0.74     |
| Ransomware   | Trojan       | Ransomware-Pysa      |      47 |        0.19933   |         0.761253 |            0.787234 |
| Ransomware   | Trojan       | Ransomware-Ako       |      45 |        0.161646  |         0.734213 |            0.488889 |
| Ransomware   | Spyware      | Ransomware-Pysa      |      43 |        0.189864  |         0.675825 |            0.674419 |
| Spyware      | Ransomware   | Spyware-CWS          |      41 |        0.122957  |         0.789217 |            0.658537 |
| Ransomware   | Trojan       | Ransomware-Maze      |      41 |        0.293698  |         0.687582 |            0.95122  |
| Ransomware   | Spyware      | Ransomware-Maze      |      37 |        0.180744  |         0.761364 |            0.918919 |
| Trojan       | Spyware      | Trojan-Emotet        |      33 |        0.14394   |         0.72166  |            0.484848 |

## Next

```text
If fold 0 output looks sane, run folds 1-4 with the same script.
Then concatenate all fold OOF predictions for F3a2 OOF overlap reproduction audit.
```