# Setup package for clean official C2+D3 repo

This package contains a shell script that copies the official C2+D3 best source files and artifacts from the old repo into the clean repo.

Default paths:

```bash
OLD_REPO=$HOME/Documents/dacn
NEW_REPO=$HOME/Documents/src_baocao
```

Run:

```bash
cd ~/Documents/src_baocao
bash ~/Downloads/setup_src_baocao_official_c2d3.sh
```

Or with explicit paths:

```bash
bash setup_src_baocao_official_c2d3.sh ~/Documents/dacn ~/Documents/src_baocao
```

The script preserves `00_raw_dataset` and `01_split`, and recreates only `02_src`, `03_outputs`, README files, requirements, and verification scripts.
