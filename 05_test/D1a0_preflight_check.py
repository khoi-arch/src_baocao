#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D1a0_preflight_check.py

Purpose
-------
Preflight for D1a before writing the actual training code.

This script does NOT train and does NOT modify the official baseline.
It only inspects:
  - official source files in 02_src
  - dataset artifact shapes
  - official 03_outputs/06_model config/report
  - available model/train class/function names

Why this step exists
--------------------
D1a needs to add auxiliary pairwise heads while preserving the official C2+D3
baseline. Before coding a training script, we need to know the exact model/training
APIs in this repo so we do not guess class names or accidentally modify source files.

Outputs
-------
  05_test/outputs/D1a0_preflight_check/D1a0_preflight_summary.md
  05_test/outputs/D1a0_preflight_check/D1a0_source_inventory.csv
  05_test/outputs/D1a0_preflight_check/D1a0_dataset_shapes.csv
  05_test/outputs/D1a0_preflight_check/D1a0_model_ast_summary.json
  05_test/outputs/D1a0_preflight_check/D1a0_train_config_snapshot.json
  05_test/outputs/D1a0_preflight_check/D1a0_preflight_output.zip
"""

from __future__ import annotations

import argparse
import ast
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


INTERESTING_SRC_FILES = [
    "06_model.py",
    "07_train.py",
    "train_utils.py",
    "config.py",
    "05_build_dataset.py",
    "04_tokenization.py",
    "02_embedding.py",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="D1a preflight source/API/dataset check.")
    p.add_argument("--repo-root", default=".")
    p.add_argument("--src-dir", default="02_src")
    p.add_argument("--dataset-npz", default="03_outputs/05_dataset/dataset.npz")
    p.add_argument("--metadata-json", default="03_outputs/05_dataset/metadata.json")
    p.add_argument("--model-out-dir", default="03_outputs/06_model")
    p.add_argument("--out-dir", default="05_test/outputs/D1a0_preflight_check")
    return p.parse_args()


def repo_path(repo_root: Path, p: str | Path) -> Path:
    q = Path(p)
    return q if q.is_absolute() else repo_root / q


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def ast_arg_names(args: ast.arguments) -> List[str]:
    names = []
    for a in args.posonlyargs + args.args:
        names.append(a.arg)
    if args.vararg:
        names.append("*" + args.vararg.arg)
    for a in args.kwonlyargs:
        names.append(a.arg)
    if args.kwarg:
        names.append("**" + args.kwarg.arg)
    return names


def base_name(base: ast.AST) -> str:
    if isinstance(base, ast.Name):
        return base.id
    if isinstance(base, ast.Attribute):
        parts = []
        cur: ast.AST | None = base
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return ".".join(reversed(parts))
    if isinstance(base, ast.Subscript):
        return base_name(base.value)
    return ast.dump(base)


def parse_python_ast(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(text)

    classes: List[Dict[str, Any]] = []
    functions: List[Dict[str, Any]] = []
    imports: List[str] = []

    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.extend([alias.name for alias in node.names])
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            imports.extend([f"{mod}.{alias.name}" if mod else alias.name for alias in node.names])
        elif isinstance(node, ast.FunctionDef):
            functions.append({
                "name": node.name,
                "args": ast_arg_names(node.args),
                "lineno": node.lineno,
                "end_lineno": getattr(node, "end_lineno", None),
            })
        elif isinstance(node, ast.ClassDef):
            methods = []
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    methods.append({
                        "name": item.name,
                        "args": ast_arg_names(item.args),
                        "lineno": item.lineno,
                        "end_lineno": getattr(item, "end_lineno", None),
                    })
            classes.append({
                "name": node.name,
                "bases": [base_name(b) for b in node.bases],
                "lineno": node.lineno,
                "end_lineno": getattr(node, "end_lineno", None),
                "methods": methods,
            })

    # Lightweight keyword flags useful for D1a.
    lower = text.lower()
    flags = {
        "uses_torch": "torch" in lower,
        "uses_nn_module": "nn.module" in lower or "torch.nn" in lower,
        "contains_transformer": "transformer" in lower,
        "contains_cls": "cls" in lower,
        "contains_forward": "def forward" in lower,
        "contains_focal": "focal" in lower,
        "contains_class_weight": "class_weight" in lower or "class weights" in lower,
        "contains_scheduler": "scheduler" in lower,
        "contains_checkpoint": "best_model" in lower or "checkpoint" in lower,
    }

    return {
        "file": str(path),
        "n_lines": text.count("\n") + 1,
        "imports": imports[:100],
        "classes": classes,
        "functions": functions,
        "flags": flags,
    }


def array_shape_info(data: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for k, v in data.items():
        arr = v
        row = {
            "key": k,
            "shape": str(tuple(arr.shape)),
            "ndim": int(arr.ndim),
            "dtype": str(arr.dtype),
        }
        if arr.size and np.issubdtype(arr.dtype, np.number):
            try:
                row.update({
                    "min": float(np.nanmin(arr)),
                    "max": float(np.nanmax(arr)),
                    "mean": float(np.nanmean(arr)),
                })
            except Exception:
                row.update({"min": None, "max": None, "mean": None})
        else:
            row.update({"min": None, "max": None, "mean": None})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("key").reset_index(drop=True)


def zip_outputs(out_dir: Path, zip_name: str = "D1a0_preflight_output.zip") -> Path:
    out_zip = out_dir / zip_name
    if out_zip.exists():
        out_zip.unlink()

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fp in out_dir.rglob("*"):
            if fp.is_file() and fp != out_zip:
                z.write(fp, fp.relative_to(out_dir))
    return out_zip


def to_md(df: pd.DataFrame, index: bool = False) -> str:
    try:
        return df.to_markdown(index=index)
    except Exception:
        return df.to_string(index=index)


def make_summary(
    *,
    repo_root: Path,
    src_dir: Path,
    dataset_npz: Path,
    metadata_json: Path,
    model_out_dir: Path,
    src_inventory: pd.DataFrame,
    dataset_shapes: pd.DataFrame,
    ast_summary: Dict[str, Any],
    config: Dict[str, Any],
    report: Dict[str, Any],
    diagnosis: Dict[str, Any],
    out_files: List[Path],
) -> str:
    lines = []
    lines.append("# D1a-0 — Preflight check")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("Inspect official baseline APIs/artifacts before implementing D1a auxiliary pairwise heads.")
    lines.append("")
    lines.append("## Guardrail")
    lines.append("")
    lines.append("- This script does not train.")
    lines.append("- This script does not modify `02_src`.")
    lines.append("- This script does not modify `03_outputs/06_model`.")
    lines.append("- Current baseline under analysis is `03_outputs/06_model`, macro-F1 around `0.8101`, not the separate `0.817` model.")
    lines.append("")
    lines.append("## Paths")
    lines.append("")
    lines.append(f"- `repo_root`: `{repo_root}`")
    lines.append(f"- `src_dir`: `{src_dir}`")
    lines.append(f"- `dataset_npz`: `{dataset_npz}`")
    lines.append(f"- `metadata_json`: `{metadata_json}`")
    lines.append(f"- `model_out_dir`: `{model_out_dir}`")
    lines.append("")
    lines.append("## Official baseline snapshot")
    lines.append("")
    if report:
        lines.append(f"- `accuracy`: {report.get('accuracy')}")
        lines.append(f"- `macro_f1`: {report.get('macro_f1')}")
        lines.append(f"- `weighted_f1`: {report.get('weighted_f1')}")
    else:
        lines.append("- classification report not found or empty")
    if diagnosis:
        lines.append(f"- `diagnosis phase`: {diagnosis.get('phase')}")
        lines.append(f"- `best_epoch`: {diagnosis.get('best_epoch')}")
        lines.append(f"- `val_macro_f1`: {diagnosis.get('val_macro_f1')}")
    lines.append("")
    lines.append("## Source inventory")
    lines.append("")
    inv_cols = ["file", "exists", "size_bytes", "n_lines"]
    inv_cols = [c for c in inv_cols if c in src_inventory.columns]
    lines.append(to_md(src_inventory[inv_cols], index=False))
    lines.append("")
    lines.append("## Dataset shapes")
    lines.append("")
    show_dataset_cols = ["key", "shape", "dtype", "min", "max", "mean"]
    show_dataset_cols = [c for c in show_dataset_cols if c in dataset_shapes.columns]
    lines.append(to_md(dataset_shapes[show_dataset_cols], index=False))
    lines.append("")
    lines.append("## Candidate model classes/functions from 06_model.py")
    lines.append("")
    model_ast = ast_summary.get("06_model.py", {})
    classes = model_ast.get("classes", [])
    functions = model_ast.get("functions", [])
    if classes:
        for cls in classes:
            bases = ", ".join(cls.get("bases", []))
            method_names = ", ".join([m["name"] for m in cls.get("methods", [])])
            lines.append(f"- class `{cls['name']}` bases=[{bases}] methods=[{method_names}]")
    else:
        lines.append("- No classes found in 06_model.py")
    if functions:
        lines.append("")
        lines.append("Top-level functions:")
        for fn in functions:
            lines.append(f"- `{fn['name']}({', '.join(fn['args'])})`")
    lines.append("")
    lines.append("## Candidate train functions from 07_train.py")
    lines.append("")
    train_ast = ast_summary.get("07_train.py", {})
    if train_ast.get("functions"):
        for fn in train_ast["functions"]:
            lines.append(f"- `{fn['name']}({', '.join(fn['args'])})`")
    else:
        lines.append("- No top-level functions found in 07_train.py")
    if train_ast.get("classes"):
        for cls in train_ast["classes"]:
            lines.append(f"- class `{cls['name']}`")
    lines.append("")
    lines.append("## D1a implementation decision to make next")
    lines.append("")
    lines.append("Based on this preflight, the next step is to choose the safest implementation method:")
    lines.append("")
    lines.append("1. If `06_model.py` exposes a clean model class returning logits/CLS, create a new wrapper in `05_test` that adds auxiliary pairwise heads.")
    lines.append("2. If CLS is not exposed, copy the official model class into a new `05_test/D1a...py` file and minimally add `return_cls` / auxiliary heads there.")
    lines.append("3. Training output must compare against baseline predictions and log `wrong->correct`, `correct->wrong`, net gain, and pair-level fix/damage.")
    lines.append("")
    lines.append("## Generated files")
    lines.append("")
    for p in out_files:
        lines.append(f"- `{p}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    src_dir = repo_path(repo_root, args.src_dir)
    dataset_npz = repo_path(repo_root, args.dataset_npz)
    metadata_json = repo_path(repo_root, args.metadata_json)
    model_out_dir = repo_path(repo_root, args.model_out_dir)
    out_dir = repo_path(repo_root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not src_dir.exists():
        raise FileNotFoundError(f"Missing src dir: {src_dir}")
    if not dataset_npz.exists():
        raise FileNotFoundError(f"Missing dataset npz: {dataset_npz}")

    # Source inventory + AST.
    inv_rows = []
    ast_summary: Dict[str, Any] = {}
    for fname in INTERESTING_SRC_FILES:
        path = src_dir / fname
        row = {
            "file": fname,
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else None,
            "n_lines": None,
        }
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="replace")
            row["n_lines"] = text.count("\n") + 1
            try:
                ast_summary[fname] = parse_python_ast(path)
            except SyntaxError as e:
                ast_summary[fname] = {
                    "file": str(path),
                    "parse_error": str(e),
                }
        inv_rows.append(row)

    src_inventory = pd.DataFrame(inv_rows)

    # Dataset shape.
    with np.load(dataset_npz, allow_pickle=True) as z:
        data = {k: z[k] for k in z.files}
    dataset_shapes = array_shape_info(data)

    # Metadata/model snapshot.
    metadata = read_json(metadata_json)
    config = read_json(model_out_dir / "config.json")
    report = read_json(model_out_dir / "val_classification_report_best.json")
    diagnosis = read_json(model_out_dir / "diagnosis_summary.json")

    train_config_snapshot = {
        "metadata": metadata,
        "config": config,
        "report": report,
        "diagnosis": diagnosis,
    }

    # Write outputs.
    summary_path = out_dir / "D1a0_preflight_summary.md"
    src_inventory_path = out_dir / "D1a0_source_inventory.csv"
    dataset_shapes_path = out_dir / "D1a0_dataset_shapes.csv"
    ast_summary_path = out_dir / "D1a0_model_ast_summary.json"
    snapshot_path = out_dir / "D1a0_train_config_snapshot.json"

    src_inventory.to_csv(src_inventory_path, index=False)
    dataset_shapes.to_csv(dataset_shapes_path, index=False)
    ast_summary_path.write_text(json.dumps(ast_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    snapshot_path.write_text(json.dumps(train_config_snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    out_files = [
        summary_path,
        src_inventory_path,
        dataset_shapes_path,
        ast_summary_path,
        snapshot_path,
    ]

    summary = make_summary(
        repo_root=repo_root,
        src_dir=src_dir,
        dataset_npz=dataset_npz,
        metadata_json=metadata_json,
        model_out_dir=model_out_dir,
        src_inventory=src_inventory,
        dataset_shapes=dataset_shapes,
        ast_summary=ast_summary,
        config=config,
        report=report,
        diagnosis=diagnosis,
        out_files=out_files,
    )
    summary_path.write_text(summary, encoding="utf-8")

    out_zip = zip_outputs(out_dir)

    print("===== D1a-0 preflight check done =====")
    print("summary:", summary_path)
    print("source_inventory:", src_inventory_path)
    print("dataset_shapes:", dataset_shapes_path)
    print("ast_summary:", ast_summary_path)
    print("snapshot:", snapshot_path)
    print("zip:", out_zip)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
