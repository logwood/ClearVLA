"""Validate one formal benchmark batch without instantiating the policy model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from clearvla.mainline.config import load_config
from clearvla.mainline.data.loading import (
    load_mainline_data,
    load_mainline_data_for_smoke,
    to_training_batch,
)
from clearvla.mainline.training.engine import validate_finite_training_batch

from .common import atomic_json, audit_benchmark_dataset

TRAINING_SMOKE_SCHEMA = "clearvla-benchmark-training-data-smoke-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def tensor_metadata(value: Tensor) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
        "device": str(value.device),
        "finite": bool(torch.isfinite(value).all()),
    }


def validate_training_data(
    config_path: Path,
    output: Path,
    *,
    split: str,
    batch_size: int,
    episode_limit: int | None = None,
    audit_dataset: bool = True,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if episode_limit is not None and episode_limit <= 0:
        raise ValueError("episode_limit must be positive when provided")
    if not audit_dataset and episode_limit is None:
        raise ValueError(
            "--skip-dataset-audit is only valid with an explicit bounded episode limit"
        )
    if output.exists():
        raise FileExistsError(f"training-data smoke output already exists: {output}")
    config = load_config(config_path)
    if audit_dataset:
        dataset_audit: dict[str, Any] = audit_benchmark_dataset(config.data.raw_hdf5_root)
    else:
        # Full benchmark inventory auditing opens every source HDF5.  Keep it
        # explicit and opt-in for large remote datasets; the bounded loader
        # below still validates the configured split/cache/language contracts.
        dataset_audit = {
            "schema": "clearvla-benchmark-dataset-audit-skipped-v1",
            "root": str(Path(config.data.raw_hdf5_root).resolve()),
            "reason": "caller requested bounded training-data smoke",
        }
    if episode_limit is None:
        bundle = load_mainline_data(config)
    else:
        bundle = load_mainline_data_for_smoke(
            config,
            split=split,
            episode_limit=episode_limit,
        )
    if split not in bundle.datasets:
        raise ValueError(f"unknown split {split!r}; choices={sorted(bundle.datasets)}")
    loader = bundle.loader(
        split,
        batch_size=batch_size,
        workers=0,
        device=torch.device("cpu"),
        shuffle=False,
    )
    raw = next(iter(loader))
    batch = to_training_batch(
        raw,
        goal=bundle.goal,
        config=config,
        device=torch.device("cpu"),
    )
    batch.validate(config)
    validate_finite_training_batch(batch)
    if hasattr(batch.online, "future"):
        raise ValueError("online policy input unexpectedly exposes future supervision")

    tensors = {
        "online.dino_history": batch.online.observation.dino_history,
        "online.raw_rgb": batch.online.observation.raw_rgb,
        "online.state": batch.online.history.state,
        "online.action_state": batch.online.history.action_state,
        "online.state_history": batch.online.history.state_history,
        "online.executed_action_history": batch.online.history.executed_action_history,
        "online.goal_tokens": batch.online.goal.tokens,
        "online.goal_mask": batch.online.goal.mask,
        "action.normalized": batch.action_target.normalized,
        "action.raw_units": batch.action_target.raw_units,
        "action.current_raw_units": batch.action_target.current_raw_units,
        "future.dino_supports": batch.future.dino_supports,
        "future.action_sequence": batch.future.action_sequence,
        "future.state_sequence": batch.future.state_sequence,
        "future.offsets": batch.future.offsets,
    }
    dino_report = Path(config.data.dino_cache) / "cache_report.json"
    language_bank = Path(config.data.t5_condition)
    report = {
        "schema": TRAINING_SMOKE_SCHEMA,
        "config": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "config_digest_with_paths": config.digest(include_paths=True),
        "dataset": dataset_audit,
        "dataset_audit_full": bool(audit_dataset),
        "materialization": {
            "mode": "full" if episode_limit is None else "bounded",
            "episode_limit": episode_limit,
        },
        "episode_splits": {
            name: len(indices) for name, indices in bundle.splits.items()
        },
        "training_windows": {
            name: len(dataset) for name, dataset in bundle.datasets.items()
        },
        "skipped": [list(row) for row in bundle.skipped],
        "language": bundle.goal.metadata,
        "language_bank_sha256": _sha256(language_bank),
        "dino_cache_report": str(dino_report.resolve()),
        "dino_cache_report_sha256": _sha256(dino_report),
        "normalizers": {
            "action": bundle.action_normalizer.to_dict(),
            "state": bundle.state_normalizer.to_dict(),
        },
        "sampled_split": split,
        "batch_size": batch.online.batch,
        "tensors": {name: tensor_metadata(value) for name, value in tensors.items()},
        "online_future_representable": False,
        "model_forward_executed": False,
        "benchmark_score": None,
    }
    atomic_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--episode-limit",
        type=int,
        default=None,
        help="materialize at most this many episodes in the selected split",
    )
    parser.add_argument(
        "--skip-dataset-audit",
        action="store_true",
        help="skip the full source-HDF5 audit (use only for a bounded smoke)",
    )
    args = parser.parse_args()
    result = validate_training_data(
        args.config,
        args.output,
        split=args.split,
        batch_size=args.batch_size,
        episode_limit=args.episode_limit,
        audit_dataset=not args.skip_dataset_audit,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["TRAINING_SMOKE_SCHEMA", "tensor_metadata", "validate_training_data"]
