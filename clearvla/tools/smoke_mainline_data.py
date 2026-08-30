"""Load and validate one typed batch without constructing a policy model."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from clearvla.mainline.config import ExperimentConfig, load_config
from clearvla.mainline.data.loading import load_mainline_data, to_training_batch
from clearvla.mainline.interfaces import TrainingBatch


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the HDF5-to-typed-batch boundary without a model or optimizer"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("train", "val", "test", "external_test"), default="val"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--decoded-cache", type=Path)
    parser.add_argument("--dino-cache", type=Path)
    parser.add_argument("--t5-condition", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    return parser


def _overrides(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    data = config.data
    for field_name, value in (
        ("raw_hdf5_root", args.data_root),
        ("decoded_cache", args.decoded_cache),
        ("dino_cache", args.dino_cache),
        ("t5_condition", args.t5_condition),
        ("split_manifest", args.split_manifest),
    ):
        if value is not None:
            data = replace(data, **{field_name: str(value)})
    data = replace(data, num_workers=int(args.num_workers))
    result = replace(config, data=data)
    result.validate()
    return result


def _shape(value: torch.Tensor) -> dict[str, object]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
        "device": str(value.device),
    }


def _validate_finite(batch: TrainingBatch) -> None:
    values = {
        "goal": batch.online.goal.tokens,
        "dino_history": batch.online.observation.dino_history,
        "raw_rgb": batch.online.observation.raw_rgb,
        "state": batch.online.history.state,
        "action_state": batch.online.history.action_state,
        "executed_action_history": batch.online.history.executed_action_history,
        "action_normalized": batch.action_target.normalized,
        "action_raw": batch.action_target.raw_units,
        "current_action_state_raw": batch.action_target.current_raw_units,
        "future_dino": batch.future.dino_supports,
        "future_action": batch.future.action_sequence,
        "future_state": batch.future.state_sequence,
    }
    invalid = [name for name, value in values.items() if not bool(torch.isfinite(value).all())]
    if invalid:
        raise ValueError(f"typed data batch contains non-finite values: {invalid}")
    expected = torch.arange(4, 49, 4, dtype=torch.long)[None].expand(
        batch.future.batch, -1
    )
    if not torch.equal(batch.future.offsets.cpu(), expected):
        raise ValueError("typed data batch future offsets must be exactly 4,8,...,48")


def main() -> None:
    args = _parser().parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch size must be positive and workers non-negative")
    config = _overrides(load_config(args.config), args)
    bundle = load_mainline_data(config)
    if args.split not in bundle.datasets:
        raise ValueError(
            f"split {args.split!r} is absent; available={sorted(bundle.datasets)}"
        )
    loader = bundle.loader(
        args.split,
        batch_size=int(args.batch_size),
        workers=int(args.num_workers),
        device=torch.device("cpu"),
        shuffle=False,
    )
    raw = next(iter(loader))
    typed = to_training_batch(
        raw,
        goal=bundle.goal,
        config=config,
        device=torch.device("cpu"),
    )
    _validate_finite(typed)
    report = {
        "schema": "clearvla-mainline-data-smoke-v1",
        "model_constructed": False,
        "optimizer_constructed": False,
        "split": args.split,
        "split_sizes": {name: len(dataset) for name, dataset in bundle.datasets.items()},
        "split_metadata": bundle.split_metadata,
        "data_profile": bundle.data_profile_metadata,
        "ordered_cameras": list(config.data.camera_names),
        "camera_keys": config.data.camera_key_map(),
        "gripper_indices": list(bundle.gripper_indices),
        "sampling_gripper_event_threshold": bundle.gripper_event_threshold,
        "goal": bundle.goal.metadata,
        "episode_count": len(bundle.episodes),
        "skipped": list(bundle.skipped),
        "typed": {
            "goal_tokens": _shape(typed.online.goal.tokens),
            "dino_history": _shape(typed.online.observation.dino_history),
            "raw_rgb": _shape(typed.online.observation.raw_rgb),
            "state": _shape(typed.online.history.state),
            "action_state": _shape(typed.online.history.action_state),
            "executed_action_history": _shape(
                typed.online.history.executed_action_history
            ),
            "action_target_normalized": _shape(typed.action_target.normalized),
            "action_target_raw": _shape(typed.action_target.raw_units),
            "current_action_state_raw": _shape(
                typed.action_target.current_raw_units
            ),
            "future_dino": _shape(typed.future.dino_supports),
            "future_action": _shape(typed.future.action_sequence),
            "future_state": _shape(typed.future.state_sequence),
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
