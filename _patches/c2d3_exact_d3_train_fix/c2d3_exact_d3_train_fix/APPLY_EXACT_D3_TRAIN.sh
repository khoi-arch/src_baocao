#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-$PWD}"
cd "$ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP="backup_before_exact_d3_train_${STAMP}"
mkdir -p "$BACKUP/02_src"
cp -a 02_src/07_train.py "$BACKUP/02_src/07_train.py" 2>/dev/null || true
cp -a 02_src/train_utils.py "$BACKUP/02_src/train_utils.py" 2>/dev/null || true
cp "$(dirname "$0")/07_train.py" 02_src/07_train.py
cp "$(dirname "$0")/train_utils.py" 02_src/train_utils.py
python -m py_compile 02_src/07_train.py 02_src/train_utils.py
echo "Applied exact D3 trainer. Backup: $BACKUP"
