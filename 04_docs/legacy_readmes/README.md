# Official C2 + D3 Repository

This repository is the clean official reproduction/evidence repo for the best C2 tokenization + D3 model.

## Core pipeline

- Raw dataset: `00_raw_dataset/Obfuscated-MalMem2022.csv`
- Fixed split: `01_split/train_raw.csv`, `01_split/val_raw.csv`
- C2 tokenization artifact:
  `03_outputs/build_mixed_quantile_offset/K512_B512_C2_selective_rank_discrete_compact/`
- D3 best run:
  `03_outputs/train_runs_fusion_ablation_D0_D7/Keff512/D3_P1_K512_B512_C2_selective_rank_discrete_compact/`

## Official C2 tokenization

C2 uses K512/B512 mixed quantile-offset tokenization with:

- selective rank/uniform offset for high-compression/high-unique features
- discrete compact offset=0 for low-unique/discrete features
- keep-current group preserved
- constant features handled separately

## Official D3 model

D3 uses:

- shared bin embedding
- offset interpolation between adjacent bin embeddings
- feature embedding
- raw continuous FiLM/multiply branch
- Transformer encoder with CLS classifier

## Verification

Run:

```bash
python verify_official_c2d3_repo.py
```

Expected official local result should be approximately:

- validation macro-F1 around `0.8171`
- official run dir: `D3_P1_K512_B512_C2_selective_rank_discrete_compact`

## Notes

This clean repo intentionally excludes exploratory ablation clutter and old mixed-version Test A/B outputs. Keep new experiments in separate branches or clearly named folders.
