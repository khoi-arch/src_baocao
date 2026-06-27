# Official C2+D3 Best Snapshot

This repo snapshot keeps only the official best pipeline and evidence artifacts.

## Paths

```text
00_raw_dataset/Obfuscated-MalMem2022.csv
01_split/train_raw.csv
01_split/val_raw.csv
02_src/
03_outputs/build_mixed_quantile_offset/K512_B512_C2_selective_rank_discrete_compact/
03_outputs/train_runs_fusion_ablation_D0_D7/Keff512/D3_P1_K512_B512_C2_selective_rank_discrete_compact/
03_outputs/audit_c2_best/
```

## Do not mix versions

For future reranker/audit tests, ensure all of these come from the same official run:

- C2 dataset `.npz`
- D3 `best_model.pt`
- prediction CSV/top-2 routing
- CLS embeddings if used

A previous mistake was mixing CLS from the official best run with top-2 routing from a separate Test1 control rerun. Avoid that.
