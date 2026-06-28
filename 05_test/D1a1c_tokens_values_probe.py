#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D1a1c_tokens_values_probe.py

Purpose
-------
D1a-1c final forward probe.

D1a-1b found the real official forward signature:

    forward(self, tokens: torch.Tensor, values: torch.Tensor, *, return_info: bool = False)

So this script tests the correct call:

    tokens: [B, F]      usually X_*_bin
    values: [B, F, 3]   candidate packed scalar channels

It tries a few candidate value channel orders and also tests return_info=True to see
whether CLS/hidden information is exposed.

No training. No modification to 02_src or official outputs.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import traceback
import zipfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="D1a-1c tokens/values forward probe.")
    p.add_argument("--repo-root", default=".")
    p.add_argument("--model-py", default="02_src/06_model.py")
    p.add_argument("--config-json", default="03_outputs/06_model/config.json")
    p.add_argument("--dataset-npz", default="03_outputs/05_dataset/dataset.npz")
    p.add_argument("--out-dir", default="05_test/outputs/D1a1c_tokens_values_probe")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    return p.parse_args()


def repo_path(root: Path, p: str | Path) -> Path:
    q = Path(p)
    return q if q.is_absolute() else root / q


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def cfg(config: Dict[str, Any], key: str, default: Any) -> Any:
    if key in config:
        return config[key]
    mc = config.get("model_config")
    if isinstance(mc, dict) and key in mc:
        return mc[key]
    return default


def import_module(path: Path, name: str = "d1a_official_model_tokens_values_probe"):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def instantiate_model(cls, config: Dict[str, Any], n_features: int, num_classes: int, num_bins: int):
    sig = inspect.signature(cls.__init__)
    candidate = {
        "num_bins": int(cfg(config, "num_bins", num_bins)),
        "n_features": int(cfg(config, "n_features", n_features)),
        "num_classes": int(cfg(config, "num_classes", num_classes)),
        "value_dim": int(cfg(config, "value_dim", 32)),
        "feature_dim": int(cfg(config, "feature_dim", 32)),
        "hidden_dim": int(cfg(config, "hidden_dim", 128)),
        "num_layers": int(cfg(config, "num_layers", 3)),
        "num_heads": int(cfg(config, "num_heads", 4)),
        "dropout": float(cfg(config, "dropout", 0.1)),
        "classifier_hidden_dim": int(cfg(config, "classifier_hidden_dim", 128)),
        "classifier_dropout": float(cfg(config, "classifier_dropout", 0.1)),
        "norm_first": bool(cfg(config, "norm_first", True)),
        "gate_init": float(cfg(config, "gate_init", 0.0)),
        "activation": cfg(config, "activation", None),
    }

    kwargs = {}
    decisions = []
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if name in candidate and candidate[name] is not None:
            kwargs[name] = candidate[name]
            decisions.append({"param": name, "used": True, "value": repr(candidate[name])})
        elif param.default is not inspect.Parameter.empty:
            decisions.append({"param": name, "used": False, "value": f"default={repr(param.default)}"})
        else:
            decisions.append({"param": name, "used": False, "value": "MISSING_REQUIRED"})

    return cls(**kwargs), kwargs, decisions


def load_batch(dataset_npz: Path, batch_size: int):
    with np.load(dataset_npz, allow_pickle=True) as z:
        Xb = z["X_val_bin"][:batch_size].astype(np.int64)
        Xo = z["X_val_offset"][:batch_size].astype(np.float32)
        y = z["y_val"][:batch_size].astype(np.int64)
        label_names = [str(x) for x in z["label_names"].tolist()]
        feature_names = [str(x) for x in z["feature_names"].tolist()]
        num_bins = int(z["num_bins"][0]) if "num_bins" in z.files else int(z["K"][0])

    # Neutral placeholders. The real D1a training script will recompute raw_scaled
    # from train_raw/val_raw, but this probe only needs shape/API.
    Xc = np.full_like(Xo, 0.5, dtype=np.float32)
    Xm = np.ones_like(Xo, dtype=np.float32)

    return {
        "X_bin": Xb,
        "X_offset": Xo,
        "X_cont": Xc,
        "X_mask": Xm,
        "y": y,
        "label_names": label_names,
        "feature_names": feature_names,
        "num_bins": num_bins,
    }


def desc_obj(x):
    import torch

    if torch.is_tensor(x):
        return {
            "type": "tensor",
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "requires_grad": bool(x.requires_grad),
        }
    if isinstance(x, dict):
        return {
            "type": "dict",
            "keys": list(x.keys()),
            "items": {str(k): desc_obj(v) for k, v in x.items()},
        }
    if isinstance(x, (tuple, list)):
        return {
            "type": type(x).__name__,
            "len": len(x),
            "items": [desc_obj(v) for v in x],
        }
    return {"type": type(x).__name__, "repr": repr(x)[:500]}


def has_logits(desc, batch_size, num_classes):
    if desc.get("type") == "tensor":
        return desc.get("shape") == [batch_size, num_classes]
    if desc.get("type") == "dict":
        return any(has_logits(v, batch_size, num_classes) for v in desc.get("items", {}).values())
    if desc.get("type") in ("tuple", "list"):
        return any(has_logits(v, batch_size, num_classes) for v in desc.get("items", []))
    return False


def has_cls_like(desc, batch_size, num_classes):
    if desc.get("type") == "tensor":
        shape = desc.get("shape")
        return isinstance(shape, list) and len(shape) == 2 and shape[0] == batch_size and shape[1] != num_classes
    if desc.get("type") == "dict":
        return any(has_cls_like(v, batch_size, num_classes) for v in desc.get("items", {}).values())
    if desc.get("type") in ("tuple", "list"):
        return any(has_cls_like(v, batch_size, num_classes) for v in desc.get("items", []))
    return False


def run_forward_attempts(model, batch, device):
    import torch

    tokens_long = torch.as_tensor(batch["X_bin"], dtype=torch.long, device=device)
    tokens_float = torch.as_tensor(batch["X_bin"], dtype=torch.float32, device=device)

    offset = torch.as_tensor(batch["X_offset"], dtype=torch.float32, device=device)
    cont = torch.as_tensor(batch["X_cont"], dtype=torch.float32, device=device)
    mask = torch.as_tensor(batch["X_mask"], dtype=torch.float32, device=device)
    bin_float = torch.as_tensor(batch["X_bin"], dtype=torch.float32, device=device)
    bin_norm = bin_float / max(float(batch["num_bins"] - 1), 1.0)

    value_variants = {
        # Most likely for D3: scalar channels after token id is supplied separately.
        "offset_cont_mask": torch.stack([offset, cont, mask], dim=-1),
        "offset_cont_bin_norm": torch.stack([offset, cont, bin_norm], dim=-1),
        "bin_offset_cont": torch.stack([bin_float, offset, cont], dim=-1),
        "bin_norm_offset_cont": torch.stack([bin_norm, offset, cont], dim=-1),
        "cont_offset_mask": torch.stack([cont, offset, mask], dim=-1),
    }

    token_variants = {
        "tokens_long": tokens_long,
        "tokens_float": tokens_float,
    }

    attempts = []
    for token_name, tokens in token_variants.items():
        for val_name, values in value_variants.items():
            for return_info in [False, True]:
                name = f"{token_name}__values_{val_name}__return_info_{return_info}"
                try:
                    with torch.no_grad():
                        out = model(tokens, values, return_info=return_info)
                    d = desc_obj(out)
                    attempts.append({
                        "attempt": name,
                        "ok": True,
                        "has_logits": has_logits(d, len(batch["y"]), len(batch["label_names"])),
                        "has_cls_like": has_cls_like(d, len(batch["y"]), len(batch["label_names"])),
                        "output": d,
                    })
                except Exception as e:
                    attempts.append({
                        "attempt": name,
                        "ok": False,
                        "error_type": type(e).__name__,
                        "error": str(e)[:1000],
                        "traceback": traceback.format_exc()[:2000],
                    })
    return attempts


def zip_outputs(out_dir: Path):
    out_zip = out_dir / "D1a1c_tokens_values_probe_output.zip"
    if out_zip.exists():
        out_zip.unlink()
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fp in out_dir.rglob("*"):
            if fp.is_file() and fp != out_zip:
                z.write(fp, fp.relative_to(out_dir))
    return out_zip


def make_summary(results: Dict[str, Any], out_files: List[Path]) -> str:
    lines = []
    lines.append("# D1a-1c — Tokens/values forward probe")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("Test official model forward with the correct signature `forward(tokens, values, return_info=False)`.")
    lines.append("")
    lines.append("## Import / instantiate")
    lines.append("")
    lines.append(f"- `import_ok`: {results.get('import_ok')}")
    lines.append(f"- `instantiate_ok`: {results.get('instantiate_ok')}")
    lines.append(f"- `init_signature`: `{results.get('init_signature')}`")
    lines.append(f"- `forward_signature`: `{results.get('forward_signature')}`")
    lines.append("")
    lines.append("## Forward attempts")
    lines.append("")
    ok_any = False
    logits_any = False
    cls_any = False
    for a in results.get("forward_attempts", []):
        if a.get("ok"):
            ok_any = True
            logits_any = logits_any or bool(a.get("has_logits"))
            cls_any = cls_any or bool(a.get("has_cls_like"))
            lines.append(f"- `{a['attempt']}`: OK, has_logits={a.get('has_logits')}, has_cls_like={a.get('has_cls_like')}, output={a.get('output')}")
        else:
            lines.append(f"- `{a['attempt']}`: FAIL, {a.get('error_type')}: {a.get('error')}")
    lines.append("")
    lines.append("## D1a implementation decision")
    lines.append("")
    if not ok_any:
        lines.append("**Decision: forward still not solved.** Need inspect model source and train dataloader construction.")
    elif logits_any and cls_any:
        lines.append("**Decision: forward works and return_info likely exposes CLS-like hidden output.** D1a may wrap/subclass official model if the CLS key is identifiable.")
    elif logits_any and not cls_any:
        lines.append("**Decision: forward works but returns logits only.** D1a should copy official model into a new `05_test` file and expose CLS before classifier.")
    else:
        lines.append("**Decision: forward works but logits shape not recognized.** Need inspect output format.")
    lines.append("")
    lines.append("## Generated files")
    lines.append("")
    for p in out_files:
        lines.append(f"- `{p}`")
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    model_py = repo_path(repo_root, args.model_py)
    config_json = repo_path(repo_root, args.config_json)
    dataset_npz = repo_path(repo_root, args.dataset_npz)
    out_dir = repo_path(repo_root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {
        "repo_root": str(repo_root),
        "model_py": str(model_py),
        "config_json": str(config_json),
        "dataset_npz": str(dataset_npz),
    }

    try:
        import torch
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
            results["device_fallback"] = "cuda_not_available_used_cpu"
        results["device_used"] = device

        mod = import_module(model_py)
        cls = getattr(mod, "D3C2D3Transformer")
        results["import_ok"] = True
        results["init_signature"] = str(inspect.signature(cls.__init__))
        results["forward_signature"] = str(inspect.signature(cls.forward))

        config = load_json(config_json)
        batch = load_batch(dataset_npz, args.batch_size)
        results["batch_info"] = {
            "batch_size": len(batch["y"]),
            "n_features": int(batch["X_bin"].shape[1]),
            "num_classes": len(batch["label_names"]),
            "num_bins": int(batch["num_bins"]),
            "label_names": batch["label_names"],
        }

        model, kwargs, decisions = instantiate_model(
            cls,
            config,
            n_features=int(batch["X_bin"].shape[1]),
            num_classes=len(batch["label_names"]),
            num_bins=int(batch["num_bins"]),
        )
        model.to(device)
        model.eval()

        results["instantiate_ok"] = True
        results["constructor_kwargs"] = {k: repr(v) for k, v in kwargs.items()}
        results["constructor_decisions"] = decisions

        try:
            forward_src = inspect.getsource(cls.forward)
        except Exception as e:
            forward_src = f"<inspect failed: {type(e).__name__}: {e}>"

        results["forward_attempts"] = run_forward_attempts(model, batch, device)

    except Exception:
        results["import_ok"] = results.get("import_ok", False)
        results["instantiate_ok"] = False
        results["fatal_error"] = traceback.format_exc()
        results["forward_attempts"] = []
        forward_src = results.get("fatal_error", "")

    summary_path = out_dir / "D1a1c_probe_summary.md"
    results_path = out_dir / "D1a1c_probe_results.json"
    forward_path = out_dir / "D1a1c_forward_source_excerpt.txt"

    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    forward_path.write_text(forward_src, encoding="utf-8")

    out_files = [summary_path, results_path, forward_path]
    summary = make_summary(results, out_files)
    summary_path.write_text(summary, encoding="utf-8")

    out_zip = zip_outputs(out_dir)

    print("===== D1a-1c tokens/values probe done =====")
    print("summary:", summary_path)
    print("results:", results_path)
    print("forward_source:", forward_path)
    print("zip:", out_zip)

    ok_any = any(x.get("ok") for x in results.get("forward_attempts", []))
    logits_any = any(x.get("ok") and x.get("has_logits") for x in results.get("forward_attempts", []))
    cls_any = any(x.get("ok") and x.get("has_cls_like") for x in results.get("forward_attempts", []))
    print("forward_ok_any:", ok_any)
    print("forward_has_logits_any:", logits_any)
    print("forward_has_cls_like_any:", cls_any)


if __name__ == "__main__":
    main()
