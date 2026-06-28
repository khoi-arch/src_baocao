# F1e1a_v4c Prob-Only Corrected Calibration Matrix

## Cleanliness decision

```text
The old weighted direction score 0.55/0.30/0.15 did not leak validation,
but it introduced hand-chosen direction hyperparameters.
v4c removes those weights and uses calibration mean probability only.
```

## Eps rule

```text
eps_raw = 0.5 * max(0, cal_error - train_error)
        + 0.5 * max(0, cal_other_mass - train_other_mass)
eps_used = min(max_eps, eps_raw)
```

## Direction rule

```text
For other malware class k:
score_k = calibration_mean_prob_k
target_k = eps_used * score_k / sum(score_other_malware)
```

## Family gap evidence

| true_L2    | true_L3              |   calibration_n |   train_inner_n |   calibration_error_rate |   train_inner_error_rate |   error_gap |   calibration_other_malware_prob_mass_mean |   train_inner_other_malware_prob_mass_mean |   other_malware_mass_gap |   eps_raw_gap_avg |
|:-----------|:---------------------|----------------:|----------------:|-------------------------:|-------------------------:|------------:|-------------------------------------------:|-------------------------------------------:|-------------------------:|------------------:|
| Spyware    | Spyware-CWS          |             320 |            1280 |                 0.328125 |                0.0953125 |   0.232813  |                                   0.412761 |                                   0.209567 |                0.203194  |         0.218003  |
| Spyware    | Spyware-180solutions |             320 |            1280 |                 0.453125 |                0.226562  |   0.226562  |                                   0.506168 |                                   0.309837 |                0.196332  |         0.211447  |
| Ransomware | Ransomware-Ako       |             320 |            1280 |                 0.34375  |                0.160156  |   0.183594  |                                   0.406444 |                                   0.235898 |                0.170546  |         0.17707   |
| Ransomware | Ransomware-Shade     |             340 |            1362 |                 0.305882 |                0.133627  |   0.172255  |                                   0.376992 |                                   0.230304 |                0.146688  |         0.159472  |
| Trojan     | Trojan-Scar          |             320 |            1280 |                 0.30625  |                0.148438  |   0.157813  |                                   0.339793 |                                   0.193502 |                0.146292  |         0.152052  |
| Trojan     | Trojan-Zeus          |             312 |            1248 |                 0.692308 |                0.53125   |   0.161058  |                                   0.590518 |                                   0.452948 |                0.137571  |         0.149314  |
| Spyware    | Spyware-TIBS         |             226 |             902 |                 0.230088 |                0.0676275 |   0.162461  |                                   0.280221 |                                   0.158779 |                0.121442  |         0.141951  |
| Trojan     | Trojan-Emotet        |             315 |            1259 |                 0.371429 |                0.216839  |   0.15459   |                                   0.386553 |                                   0.263458 |                0.123096  |         0.138843  |
| Ransomware | Ransomware-Maze      |             313 |            1253 |                 0.258786 |                0.109338  |   0.149448  |                                   0.30232  |                                   0.194659 |                0.107661  |         0.128555  |
| Trojan     | Trojan-Reconyc       |             251 |            1005 |                 0.219124 |                0.0835821 |   0.135541  |                                   0.251713 |                                   0.133211 |                0.118502  |         0.127022  |
| Ransomware | Ransomware-Pysa      |             275 |            1099 |                 0.28     |                0.149227  |   0.130773  |                                   0.339659 |                                   0.236786 |                0.102873  |         0.116823  |
| Ransomware | Ransomware-Conti     |             318 |            1272 |                 0.279874 |                0.17217   |   0.107704  |                                   0.360976 |                                   0.25979  |                0.101186  |         0.104445  |
| Spyware    | Spyware-Transponder  |             386 |            1542 |                 0.15285  |                0.0674449 |   0.0854049 |                                   0.243108 |                                   0.164527 |                0.0785809 |         0.0819929 |
| Spyware    | Spyware-Gator        |             352 |            1408 |                 0.15625  |                0.0681818 |   0.0880682 |                                   0.263126 |                                   0.190385 |                0.0727417 |         0.0804049 |
| Trojan     | Trojan-Refroso       |             320 |            1280 |                 0.134375 |                0.0632812 |   0.0710938 |                                   0.209581 |                                   0.125827 |                0.0837546 |         0.0774242 |

## Locked matrix

| true_L2    | true_L3              |   n_calibration |     eps_raw |   eps_used |   old_v4_eps_family |   old_v4b_eps_used |   target_Benign |   target_Ransomware |   target_Spyware |   target_Trojan | source                                  |
|:-----------|:---------------------|----------------:|------------:|-----------:|--------------------:|-------------------:|----------------:|--------------------:|-----------------:|----------------:|:----------------------------------------|
| Benign     | Benign               |            4688 | 0.000180078 |  0         |                 0   |                nan |               1 |           0         |        0         |       0         | non_malware_or_benign_one_hot           |
| Ransomware | Ransomware-Ako       |             320 | 0.17707     |  0.17707   |                 0.2 |                nan |               0 |           0.82293   |        0.079046  |       0.0980239 | PROB_ONLY_CORRECTED_CALIBRATION_DERIVED |
| Ransomware | Ransomware-Conti     |             318 | 0.104445    |  0.104445  |                 0.2 |                nan |               0 |           0.895555  |        0.0400008 |       0.0644443 | PROB_ONLY_CORRECTED_CALIBRATION_DERIVED |
| Ransomware | Ransomware-Maze      |             313 | 0.128555    |  0.128555  |                 0.2 |                nan |               0 |           0.871445  |        0.0494732 |       0.0790817 | PROB_ONLY_CORRECTED_CALIBRATION_DERIVED |
| Ransomware | Ransomware-Pysa      |             275 | 0.116823    |  0.116823  |                 0.2 |                nan |               0 |           0.883177  |        0.0402619 |       0.0765613 | PROB_ONLY_CORRECTED_CALIBRATION_DERIVED |
| Ransomware | Ransomware-Shade     |             340 | 0.159472    |  0.159472  |                 0.2 |                nan |               0 |           0.840528  |        0.0650828 |       0.0943888 | PROB_ONLY_CORRECTED_CALIBRATION_DERIVED |
| Spyware    | Spyware-180solutions |             320 | 0.211447    |  0.211447  |                 0.2 |                nan |               0 |           0.0859057 |        0.788553  |       0.125541  | PROB_ONLY_CORRECTED_CALIBRATION_DERIVED |
| Spyware    | Spyware-CWS          |             320 | 0.218003    |  0.218003  |                 0.2 |                nan |               0 |           0.120804  |        0.781997  |       0.0971996 | PROB_ONLY_CORRECTED_CALIBRATION_DERIVED |
| Spyware    | Spyware-Gator        |             352 | 0.0804049   |  0.0804049 |                 0.2 |                nan |               0 |           0.0419486 |        0.919595  |       0.0384564 | PROB_ONLY_CORRECTED_CALIBRATION_DERIVED |
| Spyware    | Spyware-TIBS         |             226 | 0.141951    |  0.141951  |                 0.2 |                nan |               0 |           0.0841963 |        0.858049  |       0.0577551 | PROB_ONLY_CORRECTED_CALIBRATION_DERIVED |
| Spyware    | Spyware-Transponder  |             386 | 0.0819929   |  0.0819929 |                 0.2 |                nan |               0 |           0.0413358 |        0.918007  |       0.040657  | PROB_ONLY_CORRECTED_CALIBRATION_DERIVED |
| Trojan     | Trojan-Emotet        |             315 | 0.138843    |  0.138843  |                 0.2 |                nan |               0 |           0.105217  |        0.0336258 |       0.861157  | PROB_ONLY_CORRECTED_CALIBRATION_DERIVED |
| Trojan     | Trojan-Reconyc       |             251 | 0.127022    |  0.127022  |                 0.2 |                nan |               0 |           0.0687241 |        0.0582978 |       0.872978  | PROB_ONLY_CORRECTED_CALIBRATION_DERIVED |
| Trojan     | Trojan-Refroso       |             320 | 0.0774242   |  0.0774242 |                 0.2 |                nan |               0 |           0.0420183 |        0.0354059 |       0.922576  | PROB_ONLY_CORRECTED_CALIBRATION_DERIVED |
| Trojan     | Trojan-Scar          |             320 | 0.152052    |  0.152052  |                 0.2 |                nan |               0 |           0.0837887 |        0.0682634 |       0.847948  | PROB_ONLY_CORRECTED_CALIBRATION_DERIVED |
| Trojan     | Trojan-Zeus          |             312 | 0.149314    |  0.149314  |                 0.2 |                nan |               0 |           0.119588  |        0.0297259 |       0.850686  | PROB_ONLY_CORRECTED_CALIBRATION_DERIVED |

## Sanity summary

```text
malware_family_count = 15
eps_min = 0.077424
eps_median = 0.138843
eps_mean = 0.137655
eps_max = 0.218003
cap_active_count = 0
```

## Next step

```text
Use this v4c matrix for F1e1b if sanity checks pass.
Do not tune using validation.
```