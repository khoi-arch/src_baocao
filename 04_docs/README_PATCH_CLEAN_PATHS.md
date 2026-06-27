# Patch clean official paths

Run from repo root after the layout cleanup:

```bash
cd ~/Documents/src_baocao
python patch_clean_official_paths.py
bash check_clean_paths.sh
python -m py_compile 02_src/*.py *.py
```

This writes `02_src/00_config.py`, keeps `02_src/config.py` as a compatibility shim, patches old experiment folder names to the clean official layout, and stores backups under `.archive/`.
