# B1 Export 06 Model CLS

This step exports fresh validation CLS embeddings from the newly trained official C2+D3 model.

## Outputs

- `val_cls_embeddings.npz`: `/home/pak/Documents/src_baocao/05_test/outputs/B1_cls_pairwise_signal/val_cls_embeddings.npz`
- `val_cls_predictions_with_probs.csv`: `/home/pak/Documents/src_baocao/05_test/outputs/B1_cls_pairwise_signal/val_cls_predictions_with_probs.csv`
- `manifest`: `/home/pak/Documents/src_baocao/05_test/outputs/B1_cls_pairwise_signal/B1_export_06_model_cls_manifest.json`

## Shape checks

- n_val: `11720`
- n_features: `55`
- hidden_dim / CLS dim: `128`
- cls_embeddings shape: `(11720, 128)`
- logits shape: `(11720, 4)`
- probs shape: `(11720, 4)`

## Prediction consistency metrics

- accuracy_from_export: `0.8738907850`
- top2_accuracy_from_export: `0.9680034130`
- wrong_total: `1478`
- wrong_true_in_top2: `1103`
- wrong_true_in_top2_rate: `0.7462787551`

## Notes

- This is export-only. It does not modify official baseline files.
- B1 pairwise audit should use `val_cls_embeddings.npz` from this directory.
