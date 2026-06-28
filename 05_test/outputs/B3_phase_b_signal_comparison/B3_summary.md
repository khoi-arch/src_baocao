# B3 — Phase B signal comparison

## Purpose

Compare B0/B1/B2 to locate where malware-subtype signal exists before trying any solution.

## Inputs summarized

### B0 — wrong-sample top-2

- `available`: True
- `n_wrong`: 1478
- `wrong_true_in_top2`: 1103
- `wrong_true_in_top2_rate`: 0.7462787550744249
- `top2_accuracy`: None

### B1 gate

- `PASS — CLS has usable pairwise signal`
- Reason: Mean pairwise LogisticRegression macro-F1=0.8359, mean AUC=0.9128, and mean wrong-direction true-in-top2=0.7360. This supports testing reranking or auxiliary pairwise heads.

### B2 gate

- `FAIL — input-space pairwise signal appears weak`
- Reason: Best representation `raw_scaled__bin_norm__offset` has mean macro-F1=0.6695, mean AUC=0.7425. This suggests the subtype overlap is already severe before CLS.

## CLS vs best input representation by pair

| pair                 |   cls_macro_f1 | best_input_representation    |   best_input_macro_f1 |   delta_cls_minus_input_macro_f1 |   cls_auc |   best_input_auc |   delta_cls_minus_input_auc | interpretation                      |
|:---------------------|---------------:|:-----------------------------|----------------------:|---------------------------------:|----------:|-----------------:|----------------------------:|:------------------------------------|
| Ransomware<->Spyware |       0.849668 | d3_scalar_input              |              0.664581 |                         0.185087 |  0.919618 |         0.728156 |                    0.191462 | CLS much stronger than input-linear |
| Ransomware<->Trojan  |       0.79856  | raw_scaled__bin_norm__offset |              0.662262 |                         0.136298 |  0.883855 |         0.732371 |                    0.151485 | CLS much stronger than input-linear |
| Spyware<->Trojan     |       0.85945  | raw_scaled__bin_norm__offset |              0.681834 |                         0.177616 |  0.935063 |         0.766964 |                    0.168099 | CLS much stronger than input-linear |

## Phase B diagnosis

- Result: **PHASE_B_PASS — bottleneck is not primarily preprocessing/input-linear separability**
- Diagnosis: B0 shows high wrong-sample true-in-top2 coverage, B1 shows usable CLS pairwise signal, and B2 shows weak input-space linear signal. CLS is substantially stronger than raw/token/offset linear baselines. Therefore the current evidence points to ambiguous subtype ranking / hard-pair decision in CLS/logit space, not a first-priority preprocessing/tokenization failure.
- Recommended next phase: Proceed to Phase C diagnostic rerank or logit/CLS hard-pair correction tests before Phase D model changes.

## Comparison stats

- `mean_delta_cls_minus_input_macro_f1`: 0.166334
- `min_delta_cls_minus_input_macro_f1`: 0.136298
- `mean_delta_cls_minus_input_auc`: 0.170349
- `min_delta_cls_minus_input_auc`: 0.151485

## Guardrail

Phase B is diagnostic only. It does not prove a final fix. Any rerank/auxiliary-head/margin-loss idea must still be validated as an isolated Phase C/D test.

## Generated files

- `/home/pak/Documents/src_baocao/05_test/outputs/B3_phase_b_signal_comparison/B3_summary.md`
- `/home/pak/Documents/src_baocao/05_test/outputs/B3_phase_b_signal_comparison/B3_cls_vs_input_by_pair.csv`
- `/home/pak/Documents/src_baocao/05_test/outputs/B3_phase_b_signal_comparison/B3_phase_b_decision.json`
