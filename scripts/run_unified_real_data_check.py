#!/usr/bin/env python3
"""Fail-closed real-data training admission, separate from synthetic checks.

No dataset download, path substitution, test-set training, source relabelling,
background job, GPU selection by free-memory heuristics, or fake checkpoint.
The supplied config must already declare the intended bounded run. --execute
starts the ordinary trainer synchronously only after all prerequisites pass.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        h = hashlib.file_digest(stream, "sha256")
    return h.hexdigest()


def preflight(config_path: Path, checkpoint: Path, output: Path, device: str) -> dict:
    import torch

    from clearvla.mainline.config import load_config
    from clearvla.mainline.runtime.qualification import declared_runtime

    config = load_config(config_path)
    config.validate()
    required = {
        "raw_hdf5_root": config.data.raw_hdf5_root,
        "split_manifest": config.data.split_manifest,
        "t5_condition": config.data.t5_condition,
        "checkpoint": str(checkpoint),
    }
    if config.data.visual_feature_mode == "dinov3_online_v1":
        required["dinov3_model"] = config.data.dinov3_model
    if config.data.calvin_raw_source:
        required["calvin_raw_source"] = config.data.calvin_raw_source
    if config.data.image_store_mode != "hdf5-direct":
        required["decoded_cache"] = config.data.decoded_cache
    if config.data.visual_feature_mode != "dinov3_online_v1":
        required["dino_cache"] = config.data.dino_cache
    paths = {
        key: {"path": str(value or ""), "exists": bool(value) and Path(value).expanduser().exists()}
        for key, value in required.items()
    }
    runtime = declared_runtime(ROOT, platform.python_version(), torch.__version__)
    errors = ["missing asset: " + key for key, item in paths.items() if not item["exists"]]
    if not runtime["match"]:
        errors.append("runtime does not match pyproject.toml")
    if not device.startswith("cuda") or not torch.cuda.is_available():
        errors.append(
            "requested online training check requires an explicitly available CUDA device"
        )
    elif (
        torch.device(device).index is not None
        and torch.device(device).index >= torch.cuda.device_count()
    ):
        errors.append("requested CUDA device index is unavailable")
    if output.exists():
        errors.append("training output exists; never overwrite a run")
    if config.optimizer.batch_size != 8:
        errors.append("this admission requires the requested real BS8 check")
    if config.runtime.max_train_batches < 2 or config.runtime.max_val_batches < 1:
        errors.append("config must declare at least two training batches and validation")
    for module in ("h5py", "numpy", "PIL", "rich", "transformers"):
        if importlib.util.find_spec(module) is None:
            errors.append("missing training dependency: " + module)
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if dirty:
        errors.append("working tree is not an immutable committed source")
    return {
        "schema": "clearvla-real-data-preflight-v1",
        "passed": not errors,
        "errors": errors,
        "assets": paths,
        "runtime": runtime,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "config_sha256": digest(config_path),
        "batch_size": config.optimizer.batch_size,
        "planned_updates": config.runtime.max_train_batches * config.optimizer.epochs,
        "planned_epochs": config.optimizer.epochs,
        "planned_validation_batches": config.runtime.max_val_batches,
        "checkpoint": str(checkpoint),
        "device": device,
        "scope": "asset/runtime/source admission only; not data-loader, gradient, or learned-behavior acceptance",
    }


def verify_training_receipt(output: Path, admission: dict) -> dict:
    """Check actual trainer records; this is not checkpoint-weight or behavior acceptance."""
    errors: list[str] = []
    artifacts: dict[str, dict] = {}
    for name in ("run_context.json", "metrics.jsonl", "checkpoints/latest.pt"):
        path = output / name
        exists = path.is_file() and path.stat().st_size > 0
        artifacts[name] = {"exists": exists}
        if exists:
            artifacts[name].update(size_bytes=path.stat().st_size, sha256=digest(path))
        else:
            errors.append("missing nonempty trainer artifact: " + name)
    if errors:
        return {"passed": False, "errors": errors, "artifacts": artifacts}
    try:
        context = json.loads((output / "run_context.json").read_text())
        if not isinstance(context, dict):
            raise ValueError("run context must be a JSON object")
        metrics = [
            json.loads(line)
            for line in (output / "metrics.jsonl").read_text().splitlines()
            if line.strip()
        ]
        if not all(isinstance(row, dict) for row in metrics):
            raise ValueError("metrics rows must be JSON objects")
        if context["identity"]["git_commit"] != admission["source_commit"]:
            errors.append("trainer source does not match admitted source")
        if context["execution_mode"] != "training_model_initialized":
            errors.append("receipt is not an initialized training run")
        if context["config"]["optimizer"]["batch_size"] != admission["batch_size"]:
            errors.append("trainer batch size differs from admitted batch size")
        initial = context["initialization_checkpoint"]["initial_global_step"]
        if isinstance(initial, bool) or not isinstance(initial, int) or initial < 0:
            errors.append("invalid initial completed-update count")
            initial = 0
        epochs = [row for row in metrics if row.get("kind") == "epoch"]
        if len(epochs) != admission["planned_epochs"]:
            errors.append("trainer did not complete every declared epoch")
        previous = initial
        for row in epochs:
            step = row["step"]
            if isinstance(step, bool) or not isinstance(step, int) or step <= previous:
                errors.append("non-increasing or invalid completed-update count")
            previous = step
            for group in ("train", "validation"):
                values = row[group]
                if not isinstance(values, dict) or not values:
                    errors.append("missing " + group + " epoch metrics")
                    continue
                for name, value in values.items():
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        errors.append("non-numeric metric: " + group + "." + name)
                    elif not math.isfinite(value):
                        errors.append("non-finite metric: " + group + "." + name)
                required = "loss_total" if group == "train" else "validation_action_rmse_normalized"
                if required not in values:
                    errors.append("missing required metric: " + group + "." + required)
        if previous - initial != admission["planned_updates"]:
            errors.append("actual optimizer-update count differs from declared budget")
        if any(row.get("kind") == "gradient_failure" for row in metrics):
            errors.append("trainer recorded a gradient failure")
    except (KeyError, TypeError, ValueError) as error:
        errors.append("invalid trainer receipt: " + str(error))
    return {
        "passed": not errors,
        "errors": errors,
        "artifacts": artifacts,
        "scope": "source, batch, finite epoch metrics, completed update budget and nonempty artifacts; not weight reload or behavior",
    }


def main() -> int:
    from clearvla.mainline.runtime.causal_identity_migration import CAUSAL_UNIFIED_SOURCE_V1

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("preserve the existing admission record")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    result = preflight(
        args.config.resolve(),
        args.checkpoint.expanduser().resolve(),
        args.output_dir.resolve(),
        args.device,
    )
    result.update(
        training_executed=False,
        training_passed=False,
        state="blocked" if not result["passed"] else "preflight_only",
    )
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    if not result["passed"]:
        print(json.dumps(result, indent=2))
        return 2
    if not args.execute:
        print("PREFLIGHT_ONLY: no optimizer update was requested")
        return 0
    result["checkpoint_sha256"] = digest(args.checkpoint.expanduser().resolve())
    command = [
        sys.executable,
        "-m",
        "clearvla.mainline.train",
        "--config",
        str(args.config.resolve()),
        "--init-checkpoint",
        str(args.checkpoint.expanduser().resolve()),
        "--init-model-contract-migration",
        CAUSAL_UNIFIED_SOURCE_V1,
        "--init-training-clock",
        "checkpoint",
        "--init-optimizer-state",
        "fresh",
        "--output-dir",
        str(args.output_dir.resolve()),
        "--device",
        args.device,
    ]
    result.update(state="running", training_executed=True, command=command)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    log_path = args.report.with_suffix(".training.log")
    with log_path.open("x") as log:
        completed = subprocess.run(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    receipt = (
        verify_training_receipt(args.output_dir.resolve(), result)
        if completed.returncode == 0
        else {"passed": False, "errors": ["trainer returned nonzero status"]}
    )
    training_ok = completed.returncode == 0 and receipt["passed"]
    result.update(
        state="trainer_completed" if training_ok else "failed",
        training_passed=training_ok,
        returncode=completed.returncode,
        receipt=receipt,
        log=str(log_path),
        learned_behavior_passed=False,
    )
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    return 0 if training_ok else (completed.returncode or 3)


if __name__ == "__main__":
    raise SystemExit(main())
