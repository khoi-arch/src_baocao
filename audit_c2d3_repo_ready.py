from pathlib import Path
import json
import hashlib
import subprocess
import sys

ROOT = Path.cwd()

OUT_BASE = ROOT / "03_outputs"
if not OUT_BASE.exists() and (ROOT / "03_output").exists():
    OUT_BASE = ROOT / "03_output"

RUN_DIR = OUT_BASE / "train_runs_fusion_ablation_D0_D7/Keff512/D3_P1_K512_B512_C2_selective_rank_discrete_compact"
DATASET_NPZ = OUT_BASE / "build_mixed_quantile_offset/K512_B512_C2_selective_rank_discrete_compact/mixed_quantile_offset_dataset.npz"

CONFIG_JSON = RUN_DIR / "config.json"
DIAG_JSON = RUN_DIR / "diagnosis_summary.json"
BEST_MODEL = RUN_DIR / "best_model.pt"
VAL_PRED = RUN_DIR / "val_predictions_best.csv"
TRAIN_SCRIPT = ROOT / "02_src/10_train_fusion_ablation_D0_D7.py"

REQUIRED_PATHS = [
    ROOT / "00_token_diag",
    ROOT / "01_preprocess",
    ROOT / "01_split",
    ROOT / "02_src",
    OUT_BASE,
    ROOT / "01_split/train_raw.csv",
    ROOT / "01_split/val_raw.csv",
    DATASET_NPZ,
    CONFIG_JSON,
    DIAG_JSON,
    BEST_MODEL,
    VAL_PRED,
    TRAIN_SCRIPT,
]

OUT_DIR = ROOT / "repo_readiness_audit"
OUT_DIR.mkdir(exist_ok=True)

def rel(p):
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)

def sha256_file(p: Path, max_mb=128):
    if not p.exists() or not p.is_file():
        return None
    if p.stat().st_size > max_mb * 1024 * 1024:
        return f"SKIPPED_TOO_LARGE_size_{p.stat().st_size}"
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def run_cmd(cmd, timeout=120):
    try:
        r = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
        return {
            "cmd": " ".join(cmd),
            "returncode": r.returncode,
            "stdout": r.stdout[-8000:],
            "stderr": r.stderr[-8000:],
        }
    except Exception as e:
        return {
            "cmd": " ".join(cmd),
            "error": repr(e),
        }

report = {
    "root": str(ROOT),
    "out_base": rel(OUT_BASE),
    "required_paths": [],
    "missing_required_paths": [],
    "official_baseline": {},
    "path_issues": [],
    "python_compile": None,
    "absolute_path_grep": None,
    "train_script_entrypoints_grep": None,
    "kaggle_readiness": {},
}

for p in REQUIRED_PATHS:
    item = {
        "path": rel(p),
        "exists": p.exists(),
        "is_file": p.is_file(),
        "is_dir": p.is_dir(),
        "size_bytes": p.stat().st_size if p.exists() and p.is_file() else None,
        "sha256": sha256_file(p) if p.exists() and p.is_file() else None,
    }
    report["required_paths"].append(item)
    if not item["exists"]:
        report["missing_required_paths"].append(item["path"])

if report["missing_required_paths"]:
    report["path_issues"].append({
        "type": "missing_required_paths",
        "items": report["missing_required_paths"],
    })

if CONFIG_JSON.exists():
    cfg = json.loads(CONFIG_JSON.read_text(encoding="utf-8"))
    report["official_baseline"]["config_run_id"] = cfg.get("run_id")
    report["official_baseline"]["config_dataset_npz"] = cfg.get("dataset_npz")
    report["official_baseline"]["config_num_bins"] = cfg.get("num_bins")
    report["official_baseline"]["config_effective_token_budget"] = cfg.get("effective_token_budget")
    report["official_baseline"]["config_out_dir"] = cfg.get("out_dir")
    report["official_baseline"]["config_label_names"] = cfg.get("label_names")
    report["official_baseline"]["config_continuous_info"] = cfg.get("continuous_info")

    ds = cfg.get("dataset_npz")
    if ds:
        candidates = [Path(ds), ROOT / ds]
        if not any(c.exists() for c in candidates):
            report["path_issues"].append({
                "type": "config_dataset_npz_not_resolved",
                "dataset_npz": ds,
            })

if DIAG_JSON.exists():
    diag = json.loads(DIAG_JSON.read_text(encoding="utf-8"))
    report["official_baseline"]["diag_run_id"] = diag.get("run_id")
    report["official_baseline"]["best_epoch"] = diag.get("best_epoch")
    report["official_baseline"]["val_macro_f1"] = diag.get("val", {}).get("macro_f1")
    report["official_baseline"]["val_accuracy"] = diag.get("val", {}).get("accuracy")
    report["official_baseline"]["representation"] = diag.get("representation")
    report["official_baseline"]["local"] = diag.get("local")
    report["official_baseline"]["continuous_source"] = diag.get("continuous_source")
    report["official_baseline"]["fusion"] = diag.get("fusion")

report["python_compile"] = run_cmd([sys.executable, "-m", "compileall", "-q", "02_src"])

report["absolute_path_grep"] = run_cmd([
    "bash", "-lc",
    "grep -RInE '/home/pak|/mnt/data|/Users/|C:\\\\|/kaggle/working' "
    "00_token_diag 01_preprocess 01_split 02_src 03_output 03_outputs *.py *.json *.md "
    "2>/dev/null | head -300"
])

report["train_script_entrypoints_grep"] = run_cmd([
    "bash", "-lc",
    "grep -RInE 'argparse|add_argument|if __name__|dataset_npz|mixed_quantile_offset_dataset|best_model|val_predictions_best' "
    "02_src/10_train_fusion_ablation_D0_D7.py 02_src/*.py 2>/dev/null | head -300"
])

tree_lines = []
for base in ["00_token_diag", "01_preprocess", "01_split", "02_src", "03_output", "03_outputs"]:
    p = ROOT / base
    if not p.exists():
        continue
    for f in sorted(p.rglob("*")):
        if f.is_file():
            sf = str(f)
            if "__pycache__" in sf or "/venv/" in sf or "/.venv/" in sf or "/.git/" in sf:
                continue
            tree_lines.append(f"{rel(f)}\t{f.stat().st_size}")

(OUT_DIR / "file_tree_sizes.tsv").write_text("\n".join(tree_lines), encoding="utf-8")

report["kaggle_readiness"] = {
    "minimum_files_for_rerun_from_preprocessed_artifact": [
        "01_split/train_raw.csv",
        "01_split/val_raw.csv",
        "02_src/10_train_fusion_ablation_D0_D7.py",
        "03_outputs/build_mixed_quantile_offset/K512_B512_C2_selective_rank_discrete_compact/mixed_quantile_offset_dataset.npz",
        "03_outputs/build_mixed_quantile_offset/K512_B512_C2_selective_rank_discrete_compact/metadata.json nếu script cần",
        "requirements.txt hoặc notebook install cell",
    ],
    "minimum_files_for_audit_existing_baseline": [
        "best_model.pt",
        "diagnosis_summary.json",
        "config.json",
        "val_predictions_best.csv",
        "mixed_quantile_offset_dataset.npz",
        "train_raw.csv",
        "val_raw.csv",
    ],
    "important_warning": "Nếu repo không có mixed_quantile_offset_dataset.npz hoặc không có script build lại nó từ 00_token_diag/01_preprocess thì Kaggle clone repo sẽ không rerun được baseline D3 từ đầu.",
}

out_json = OUT_DIR / "c2d3_repo_readiness_report.json"
out_md = OUT_DIR / "c2d3_repo_readiness_report.md"

out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

md = []
md.append("# C2/D3 Repo Readiness Audit\n\n")
md.append(f"- root: `{ROOT}`\n")
md.append(f"- out_base: `{rel(OUT_BASE)}`\n\n")

md.append("## Required paths\n\n")
for x in report["required_paths"]:
    mark = "OK" if x["exists"] else "MISSING"
    md.append(f"- `{mark}` `{x['path']}` size={x['size_bytes']}\n")

md.append("\n## Official baseline\n\n")
for k, v in report["official_baseline"].items():
    md.append(f"- `{k}`: `{v}`\n")

md.append("\n## Path issues\n\n")
if report["path_issues"]:
    for issue in report["path_issues"]:
        md.append(f"- `{issue}`\n")
else:
    md.append("- No path issue detected from required checks.\n")

md.append("\n## Python compile\n\n")
md.append(f"- returncode: `{report['python_compile'].get('returncode')}`\n")
if report["python_compile"].get("stderr"):
    md.append("\n```text\n" + report["python_compile"]["stderr"] + "\n```\n")

md.append("\n## Absolute path grep\n\n")
md.append(f"- returncode: `{report['absolute_path_grep'].get('returncode')}`\n")
md.append("\n```text\n" + (report["absolute_path_grep"].get("stdout") or "") + "\n```\n")

md.append("\n## Train script / entrypoint grep\n\n")
md.append("\n```text\n" + (report["train_script_entrypoints_grep"].get("stdout") or "") + "\n```\n")

out_md.write_text("".join(md), encoding="utf-8")

print("saved:", out_json)
print("saved:", out_md)
print("saved:", OUT_DIR / "file_tree_sizes.tsv")

print("\n==== SUMMARY ====")
print(json.dumps({
    "missing_required_paths": report["missing_required_paths"],
    "official_val_macro_f1": report["official_baseline"].get("val_macro_f1"),
    "official_val_accuracy": report["official_baseline"].get("val_accuracy"),
    "python_compile_returncode": report["python_compile"].get("returncode"),
    "absolute_path_hits_preview": (report["absolute_path_grep"].get("stdout") or "")[:1500],
}, indent=2, ensure_ascii=False))
