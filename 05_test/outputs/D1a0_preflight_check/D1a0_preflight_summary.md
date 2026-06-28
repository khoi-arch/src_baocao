# D1a-0 — Preflight check

## Purpose

Inspect official baseline APIs/artifacts before implementing D1a auxiliary pairwise heads.

## Guardrail

- This script does not train.
- This script does not modify `02_src`.
- This script does not modify `03_outputs/06_model`.
- Current baseline under analysis is `03_outputs/06_model`, macro-F1 around `0.8101`, not the separate `0.817` model.

## Paths

- `repo_root`: `/home/pak/Documents/src_baocao`
- `src_dir`: `/home/pak/Documents/src_baocao/02_src`
- `dataset_npz`: `/home/pak/Documents/src_baocao/03_outputs/05_dataset/dataset.npz`
- `metadata_json`: `/home/pak/Documents/src_baocao/03_outputs/05_dataset/metadata.json`
- `model_out_dir`: `/home/pak/Documents/src_baocao/03_outputs/06_model`

## Official baseline snapshot

- `accuracy`: 0.8738907849829352
- `macro_f1`: 0.8100942811251434
- `weighted_f1`: 0.8734741993346654
- `diagnosis phase`: Final pipeline C2 D3
- `best_epoch`: 38
- `val_macro_f1`: None

## Source inventory

| file                | exists   |   size_bytes |   n_lines |
|:--------------------|:---------|-------------:|----------:|
| 06_model.py         | True     |         5686 |       159 |
| 07_train.py         | True     |        61195 |      1494 |
| train_utils.py      | True     |         9403 |       291 |
| config.py           | True     |         9136 |       292 |
| 05_build_dataset.py | True     |        15236 |       331 |
| 04_tokenization.py  | True     |        24309 |       522 |
| 02_embedding.py     | True     |         4637 |       126 |

## Dataset shapes

| key            | shape       | dtype   |   min |   max |       mean |
|:---------------|:------------|:--------|------:|------:|-----------:|
| K              | (1,)        | int64   |   512 |   512 | 512        |
| X_train_bin    | (46876, 55) | int64   |     0 |   511 |  98.0402   |
| X_train_offset | (46876, 55) | float32 |     0 |     1 |   0.241214 |
| X_val_bin      | (11720, 55) | int64   |     0 |   511 |  98.3885   |
| X_val_offset   | (11720, 55) | float32 |     0 |     1 |   0.241586 |
| feature_names  | (55,)       | object  |   nan |   nan | nan        |
| label_names    | (4,)        | object  |   nan |   nan | nan        |
| num_bins       | (1,)        | int64   |   512 |   512 | 512        |
| y_train        | (46876,)    | int64   |     0 |     3 |   0.994837 |
| y_val          | (11720,)    | int64   |     0 |     3 |   0.99471  |

## Candidate model classes/functions from 06_model.py

- class `D3C2D3Transformer` bases=[nn.Module] methods=[__init__, forward, embedding_extra_summary]

Top-level functions:
- `cfg(name, default)`

## Candidate train functions from 07_train.py

- `cfg(name, default)`
- `parse_args()`
- `repo_root()`
- `resolve_path(path_like)`
- `default_model_dir()`
- `default_dataset_path(K, B)`
- `default_metadata_path(K, B)`
- `resolve_repo_path(path_from_meta, fallback_relative)`
- `load_dataset(dataset_path, metadata_path)`
- `load_z_preprocessed(meta)`
- `resolve_raw_paths(args)`
- `load_raw_scaled(meta, args)`
- `load_continuous_for_run(spec, meta, args, train_shape, val_shape)`
- `build_selective_mask(X_train_bin, X_val_bin, X_train_cont, num_bins, tail_frac, wide_quantile)`
- `strip_eval(report)`
- `malware_avg_f1(report)`
- `make_boundary_bin_diagnostics(X_bin, X_offset, num_bins, feature_names, split_name)`
- `make_embedding_runtime_diagnostics(model, X_bin, X_offset, X_cont, X_mask, device, max_rows)`
- `make_diagnosis_summary(args, spec, best_epoch, best_train, best_val, metadata, continuous_info, selective_info, boundary_data_diagnostics, embedding_runtime_diagnostics, model)`
- `main()`
- class `FusionAblationDataset`
- class `BaseValueEmbedding`
- class `D0OffsetInterpolationEmbedding`
- class `D1InterpRawScalarConcatEmbedding`
- class `D2D4D5InterpProjectGateConcatEmbedding`
- class `D3InterpRawFiLMEmbedding`
- class `D6ProjectedOffsetAddEmbedding`
- class `D7ProjectedOffsetConcatEmbedding`
- class `FusionAblationTransformer`

## D1a implementation decision to make next

Based on this preflight, the next step is to choose the safest implementation method:

1. If `06_model.py` exposes a clean model class returning logits/CLS, create a new wrapper in `05_test` that adds auxiliary pairwise heads.
2. If CLS is not exposed, copy the official model class into a new `05_test/D1a...py` file and minimally add `return_cls` / auxiliary heads there.
3. Training output must compare against baseline predictions and log `wrong->correct`, `correct->wrong`, net gain, and pair-level fix/damage.

## Generated files

- `/home/pak/Documents/src_baocao/05_test/outputs/D1a0_preflight_check/D1a0_preflight_summary.md`
- `/home/pak/Documents/src_baocao/05_test/outputs/D1a0_preflight_check/D1a0_source_inventory.csv`
- `/home/pak/Documents/src_baocao/05_test/outputs/D1a0_preflight_check/D1a0_dataset_shapes.csv`
- `/home/pak/Documents/src_baocao/05_test/outputs/D1a0_preflight_check/D1a0_model_ast_summary.json`
- `/home/pak/Documents/src_baocao/05_test/outputs/D1a0_preflight_check/D1a0_train_config_snapshot.json`
