# C2/D3 Repo Readiness Audit

- root: `/home/pak/Documents/src_baocao`
- out_base: `03_outputs`

## Required paths

- `MISSING` `00_token_diag` size=None
- `MISSING` `01_preprocess` size=None
- `OK` `01_split` size=None
- `OK` `02_src` size=None
- `OK` `03_outputs` size=None
- `OK` `01_split/train_raw.csv` size=16255948
- `OK` `01_split/val_raw.csv` size=4069527
- `MISSING` `03_outputs/build_mixed_quantile_offset/K512_B512_C2_selective_rank_discrete_compact/mixed_quantile_offset_dataset.npz` size=None
- `MISSING` `03_outputs/train_runs_fusion_ablation_D0_D7/Keff512/D3_P1_K512_B512_C2_selective_rank_discrete_compact/config.json` size=None
- `MISSING` `03_outputs/train_runs_fusion_ablation_D0_D7/Keff512/D3_P1_K512_B512_C2_selective_rank_discrete_compact/diagnosis_summary.json` size=None
- `MISSING` `03_outputs/train_runs_fusion_ablation_D0_D7/Keff512/D3_P1_K512_B512_C2_selective_rank_discrete_compact/best_model.pt` size=None
- `MISSING` `03_outputs/train_runs_fusion_ablation_D0_D7/Keff512/D3_P1_K512_B512_C2_selective_rank_discrete_compact/val_predictions_best.csv` size=None
- `MISSING` `02_src/10_train_fusion_ablation_D0_D7.py` size=None

## Official baseline


## Path issues

- `{'type': 'missing_required_paths', 'items': ['00_token_diag', '01_preprocess', '03_outputs/build_mixed_quantile_offset/K512_B512_C2_selective_rank_discrete_compact/mixed_quantile_offset_dataset.npz', '03_outputs/train_runs_fusion_ablation_D0_D7/Keff512/D3_P1_K512_B512_C2_selective_rank_discrete_compact/config.json', '03_outputs/train_runs_fusion_ablation_D0_D7/Keff512/D3_P1_K512_B512_C2_selective_rank_discrete_compact/diagnosis_summary.json', '03_outputs/train_runs_fusion_ablation_D0_D7/Keff512/D3_P1_K512_B512_C2_selective_rank_discrete_compact/best_model.pt', '03_outputs/train_runs_fusion_ablation_D0_D7/Keff512/D3_P1_K512_B512_C2_selective_rank_discrete_compact/val_predictions_best.csv', '02_src/10_train_fusion_ablation_D0_D7.py']}`

## Python compile

- returncode: `0`

## Absolute path grep

- returncode: `0`

```text
02_src/07_audit_best.py:801:    if str(root).startswith("/kaggle/working"):
02_src/07_audit_best.py:802:        zip_path = Path("/kaggle/working/c2_best_audit_outputs.zip")
03_outputs/00_dataset/metadata.json:6:    "train_preprocessed": "/home/pak/Documents/dacn/03_outputs/preprocessing/train_preprocessed_K512.csv",
03_outputs/00_dataset/metadata.json:7:    "val_preprocessed": "/home/pak/Documents/dacn/03_outputs/preprocessing/val_preprocessed_K512.csv",
03_outputs/00_dataset/metadata.json:8:    "policy_json": "/home/pak/Documents/dacn/03_outputs/preprocessing/preprocess_policy_K512.json",
03_outputs/00_dataset/metadata.json:9:    "diag_json": "/home/pak/Documents/dacn/03_outputs/bin_diag/quantile_vs_uniform_bin_diag_K512_B512.json"
03_outputs/00_dataset/metadata.json:28226:    "dataset_npz": "/home/pak/Documents/dacn/03_outputs/build_mixed_quantile_offset/K512_B512/mixed_quantile_offset_dataset.npz",
03_outputs/00_dataset/metadata.json:28227:    "metadata_json": "/home/pak/Documents/dacn/03_outputs/build_mixed_quantile_offset/K512_B512/mixed_quantile_offset_metadata.json"
03_outputs/02_audit_best/audit_summary.md:5:- dataset: `/home/pak/Documents/dacn/03_outputs/build_mixed_quantile_offset/K512_B512_C2_selective_rank_discrete_compact/mixed_quantile_offset_dataset.npz`
03_outputs/02_audit_best/audit_summary.md:6:- metadata: `/home/pak/Documents/dacn/03_outputs/build_mixed_quantile_offset/K512_B512_C2_selective_rank_discrete_compact/mixed_quantile_offset_metadata.json`
03_outputs/02_audit_best/audit_summary.md:7:- run_dir: `/home/pak/Documents/dacn/03_outputs/train_runs_fusion_ablation_D0_D7/Keff512/D3_P1_K512_B512_C2_selective_rank_discrete_compact`
03_outputs/02_audit_best/audit_summary.md:8:- checkpoint: `/home/pak/Documents/dacn/03_outputs/train_runs_fusion_ablation_D0_D7/Keff512/D3_P1_K512_B512_C2_selective_rank_discrete_compact/best_model.pt`
03_outputs/02_audit_best/01_result_audit/diagnosis_summary_copy.json:73:    "train_path": "/kaggle/working/dacn/01_split/train_raw.csv",
03_outputs/02_audit_best/01_result_audit/diagnosis_summary_copy.json:74:    "val_path": "/kaggle/working/dacn/01_split/val_raw.csv",
03_outputs/03_audit_rootcause/summary.md:4:- dataset: `/home/pak/Documents/dacn/03_outputs/build_mixed_quantile_offset/K512_B512_C2_selective_rank_discrete_compact/mixed_quantile_offset_dataset.npz`
03_outputs/03_audit_rootcause/summary.md:5:- metadata: `/home/pak/Documents/dacn/03_outputs/build_mixed_quantile_offset/K512_B512_C2_selective_rank_discrete_compact/mixed_quantile_offset_metadata.json`
03_outputs/03_audit_rootcause/summary.md:6:- checkpoint: `/home/pak/Documents/dacn/03_outputs/train_runs_fusion_ablation_D0_D7/Keff512/D3_P1_K512_B512_C2_selective_rank_discrete_compact/best_model.pt`
03_outputs/03_audit_rootcause/summary.md:7:- run_dir: `/home/pak/Documents/dacn/03_outputs/train_runs_fusion_ablation_D0_D7/Keff512/D3_P1_K512_B512_C2_selective_rank_discrete_compact`
03_outputs/01_model/config.json:19:    "train_path": "/kaggle/working/dacn/01_split/train_raw.csv",
03_outputs/01_model/config.json:20:    "val_path": "/kaggle/working/dacn/01_split/val_raw.csv",
03_outputs/01_model/diagnosis_summary.json:73:    "train_path": "/kaggle/working/dacn/01_split/train_raw.csv",
03_outputs/01_model/diagnosis_summary.json:74:    "val_path": "/kaggle/working/dacn/01_split/val_raw.csv",
audit_c2d3_repo_ready.py:139:    "grep -RInE '/home/pak|/mnt/data|/Users/|C:\\\\|/kaggle/working' "

```

## Train script / entrypoint grep


```text
, "D6", "D7"], required=True)
02_src/06_train.py:59:    p.add_argument("--K", type=int, default=int(cfg("TOKEN_K", 1000)))
02_src/06_train.py:60:    p.add_argument("--num-bins", type=int, default=int(cfg("VALUE_NUM_BINS", 128)))
02_src/06_train.py:61:    p.add_argument("--dataset-npz", default="")
02_src/06_train.py:62:    p.add_argument("--metadata-json", default="")
02_src/06_train.py:63:    p.add_argument("--out-root", default=str(cfg("OUTPUT_ROOT", Path("03_outputs")) / "train_runs_fusion_ablation_D0_D7"))
02_src/06_train.py:64:    p.add_argument("--run-name", default="")
02_src/06_train.py:66:    p.add_argument("--train-raw", default="")
02_src/06_train.py:67:    p.add_argument("--val-raw", default="")
02_src/06_train.py:70:    p.add_argument("--tail-frac", type=float, default=0.10)
02_src/06_train.py:71:    p.add_argument("--wide-quantile", type=float, default=0.90)
02_src/06_train.py:73:    p.add_argument("--seed", type=int, default=int(cfg("TRAIN_SEED", 42)))
02_src/06_train.py:74:    p.add_argument("--device", default=str(cfg("TRAIN_DEVICE", "auto")))
02_src/06_train.py:76:    p.add_argument("--epochs", type=int, default=int(cfg("TRAIN_EPOCHS", 80)))
02_src/06_train.py:77:    p.add_argument("--batch-size", type=int, default=int(cfg("TRAIN_BATCH_SIZE", 256)))
02_src/06_train.py:78:    p.add_argument("--lr", type=float, default=float(cfg("TRAIN_LR", 1e-3)))
02_src/06_train.py:79:    p.add_argument("--weight-decay", type=float, default=float(cfg("TRAIN_WEIGHT_DECAY", 1e-4)))
02_src/06_train.py:80:    p.add_argument("--scheduler", choices=["none", "warmup_cosine"], default=str(cfg("TRAIN_SCHEDULER", "warmup_cosine")))
02_src/06_train.py:81:    p.add_argument("--warmup-epochs", type=int, default=int(cfg("TRAIN_WARMUP_EPOCHS", 8)))
02_src/06_train.py:82:    p.add_argument("--min-lr-ratio", type=float, default=float(cfg("TRAIN_MIN_LR_RATIO", 0.05)))
02_src/06_train.py:83:    p.add_argument("--patience", type=int, default=int(cfg("TRAIN_PATIENCE", 12)))
02_src/06_train.py:84:    p.add_argument("--min-delta", type=float, default=float(cfg("TRAIN_MIN_DELTA", 1e-4)))
02_src/06_train.py:85:    p.add_argument("--num-workers", type=int, default=int(cfg("TRAIN_NUM_WORKERS", 0)))
02_src/06_train.py:86:    p.add_argument("--grad-clip-norm", type=float, default=float(cfg("TRAIN_GRAD_CLIP_NORM", 1.0)))
02_src/06_train.py:87:    p.add_argument("--use-class-weights", action=argparse.BooleanOptionalAction, default=bool(cfg("USE_CLASS_WEIGHTS", True)))
02_src/06_train.py:89:    p.add_argument("--value-dim", type=int, default=int(cfg("VALUE_EMBED_DIM", 32)))
02_src/06_train.py:90:    p.add_argument("--feature-dim", type=int, default=int(cfg("FEATURE_EMBED_DIM", 32)))
02_src/06_train.py:91:    p.add_argument("--hidden-dim", type=int, default=int(cfg("MODEL_HIDDEN_DIM", 128)))
02_src/06_train.py:92:    p.add_argument("--num-layers", type=int, default=int(cfg("MODEL_NUM_LAYERS", 3)))
02_src/06_train.py:93:    p.add_argument("--num-heads", type=int, default=int(cfg("MODEL_NUM_HEADS", 4)))
02_src/06_train.py:94:    p.add_argument("--dropout", type=float, default=float(cfg("MODEL_DROPOUT", 0.1)))
02_src/06_train.py:95:    p.add_argument("--classifier-hidden-dim", type=int, default=int(cfg("CLASSIFIER_HIDDEN_DIM", 128)))
02_src/06_train.py:96:    p.add_argument("--classifier-dropout", type=float, default=float(cfg("CLASSIFIER_DROPOUT", 0.1)))
02_src/06_train.py:97:    p.add_argument("--norm-first", action=argparse.BooleanOptionalAction, default=bool(cfg("TRANSFORMER_NORM_FIRST", True)))
02_src/06_train.py:100:    p.add_argument("--gate-init", type=float, default=0.0)
02_src/06_train.py:101:    p.add_argument(
02_src/06_train.py:254:def resolve_raw_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
02_src/06_train.py:266:def load_raw_scaled(meta: Dict[str, object], args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
02_src/06_train.py:322:    args: argparse.Namespace,
02_src/06_train.py:968:    args: argparse.Namespace,
02_src/06_train.py:1077:    dataset_path = Path(args.dataset_npz) if args.dataset_npz else default_dataset_path(K_artifact, B)
02_src/06_train.py:1150:        "dataset_npz": str(dataset_path),
02_src/06_train.py:1310:            "dataset_npz": str(dataset_path),
02_src/06_train.py:1397:                out_dir / "best_model.pt",
02_src/06_train.py:1442:        out_dir / "val_predictions_best.csv",
02_src/06_train.py:1476:if __name__ == "__main__":
02_src/07_audit_best.py:20:import argparse
02_src/07_audit_best.py:82:def resolve_paths(args: argparse.Namespace) -> Dict[str, Path]:
02_src/07_audit_best.py:85:    dataset = Path(args.dataset_npz) if args.dataset_npz else None
02_src/07_audit_best.py:108:            cp = run_dir / "best_model.pt"
02_src/07_audit_best.py:112:            checkpoint = find_first(root, ["best_model.pt"], ["D3_P1_00_dataset"])
02_src/07_audit_best.py:122:        missing.append("C2 D3 best_model.pt checkpoint")
02_src/07_audit_best.py:679:    ap = argparse.ArgumentParser()
02_src/07_audit_best.py:680:    ap.add_argument("--dataset-npz", default="")
02_src/07_audit_best.py:681:    ap.add_argument("--metadata-json", default="")
02_src/07_audit_best.py:682:    ap.add_argument("--run-dir", default="")
02_src/07_audit_best.py:683:    ap.add_argument("--checkpoint", default="")
02_src/07_audit_best.py:684:    ap.add_argument("--out-dir", default=str(CFG.AUDIT_BEST_DIR) if "CFG" in globals() else "03_outputs/02_audit_best")
02_src/07_audit_best.py:685:    ap.add_argument("--device", default="auto")
02_src/07_audit_best.py:686:    ap.add_argument("--batch-size", type=int, default=512)
02_src/07_audit_best.py:687:    ap.add_argument("--rare-threshold", type=int, default=5)
02_src/07_audit_best.py:812:if __name__ == "__main__":
02_src/08_audit_rootcause.py:28:import argparse
02_src/08_audit_rootcause.py:115:def resolve_paths(args: argparse.Namespace) -> Dict[str, Path]:
02_src/08_audit_rootcause.py:124:    dataset = rel(args.dataset_npz)
02_src/08_audit_rootcause.py:138:        if run_dir is not None and (run_dir / "best_model.pt").exists():
02_src/08_audit_rootcause.py:139:            checkpoint = run_dir / "best_model.pt"
02_src/08_audit_rootcause.py:141:            checkpoint = find_first(root, ["best_model.pt"], ["D3_P1_00_dataset"])
02_src/08_audit_rootcause.py:154:        missing.append("C2 best_model.pt")
02_src/08_audit_rootcause.py:842:    ap = argparse.ArgumentParser()
02_src/08_audit_rootcause.py:843:    ap.add_argument("--dataset-npz", default="")
02_src/08_audit_rootcause.py:844:    ap.add_argument("--metadata-json", default="")
02_src/08_audit_rootcause.py:845:    ap.add_argument("--run-dir", default="")
02_src/08_audit_rootcause.py:846:    ap.add_argument("--checkpoint", default="")
02_src/08_audit_rootcause.py:847:    ap.add_argument("--c2-audit-dir", default="")
02_src/08_audit_rootcause.py:848:    ap.add_argument("--out-dir", default=str(CFG.AUDIT_ROOTCAUSE_DIR) if "CFG" in globals() else "03_outputs/03_audit_rootcause")
02_src/08_audit_rootcause.py:849:    ap.add_argument("--device", default="auto")
02_src/08_audit_rootcause.py:850:    ap.add_argument("--batch-size", type=int, default=2048)
02_src/08_audit_rootcause.py:851:    ap.add_argument("--rare-threshold", type=int, default=5)
02_src/08_audit_rootcause.py:852:    ap.add_argument("--knn-k", type=int, default=25)
02_src/08_audit_rootcause.py:853:    ap.add_argument("--max-train-knn", type=int, default=0, help="0=use all train samples; set e.g. 30000 if too slow")
02_src/08_audit_rootcause.py:854:    ap.add_argument("--seed", type=int, default=42)
02_src/08_audit_rootcause.py:855:    ap.add_argument("--skip-group-masking", action="store_true")
02_src/08_audit_rootcause.py:856:    ap.add_argument("--skip-cls", action="store_true")
02_src/08_audit_rootcause.py:1004:if __name__ == "__main__":
02_src/99_config.py:38:BEST_MODEL = MODEL_DIR / "best_model.pt"
02_src/99_config.py:47:VAL_PREDICTIONS = MODEL_PREDICTIONS_DIR / "val_predictions_best.csv"

```
