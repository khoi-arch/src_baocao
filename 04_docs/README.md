# Official C2+D3 Malware Classification Repo

This repository contains the cleaned official implementation and artifacts for the selected C2 tokenization + D3 model.

## Layout

```text
00_raw_dataset/                  Raw CIC-MalMem CSV
01_split/                        Fixed train/validation splits
02_src/                          Clean numbered source files
03_outputs/                      Official dataset/model/audit artifacts
04_docs/                         Documentation and legacy notes
```

## Official artifact mapping

```text
03_outputs/00_dataset/dataset.npz
03_outputs/00_dataset/metadata.json

03_outputs/01_model/best_model.pt
03_outputs/01_model/config.json
03_outputs/01_model/diagnosis_summary.json
03_outputs/01_model/history.csv
03_outputs/01_model/reports/
03_outputs/01_model/predictions/

03_outputs/02_audit_best/
03_outputs/03_audit_rootcause/
```

This cleanup only normalizes names and folders. The next step is to update Python import/path references to the new layout.
