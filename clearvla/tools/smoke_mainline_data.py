"""Load and validate one typed batch without constructing a policy model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from dataclasses import replace
from pathlib import Path

import torch

from clearvla.mainline.config import ExperimentConfig, load_config
from clearvla.mainline.data.loading import (
    load_mainline_data_for_smoke,
    to_training_batch,
)
from clearvla.mainline.interfaces import TrainingBatch

_FULL_GIT_SHA1 = re.compile(r"[0-9a-f]{40}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the HDF5-to-typed-batch boundary without a model or optimizer"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("train", "val", "test", "external_test"), default="val"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--episode-limit",
        type=int,
        default=1,
        help="Maximum episodes in the selected loader-only dataset.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--decoded-cache", type=Path)
    parser.add_argument("--dino-cache", type=Path)
    parser.add_argument("--t5-condition", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--source-commit")
    parser.add_argument("--output", type=Path)
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


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_identity(path: Path) -> dict[str, object]:
    source = Path(path).expanduser().resolve()
    return {
        "path": str(source),
        "size_bytes": int(source.stat().st_size),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


def _validated_source_commit(requested: str | None, current: str) -> str:
    actual = str(current).strip()
    if _FULL_GIT_SHA1.fullmatch(actual) is None:
        raise ValueError("current source commit must be a full lowercase Git SHA-1")
    if requested is not None:
        expected = str(requested).strip()
        if _FULL_GIT_SHA1.fullmatch(expected) is None:
            raise ValueError("source commit must be a full lowercase Git SHA-1")
        if expected != actual:
            raise ValueError(
                f"source commit {expected} does not match current checkout {actual}"
            )
    return actual


def _current_repository_commit() -> str:
    repository = Path(__file__).resolve().parents[2]
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("smoke report requires a readable Git checkout identity") from exc
    return _validated_source_commit(None, completed.stdout)


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
    if args.batch_size <= 0 or args.episode_limit <= 0 or args.num_workers < 0:
        raise ValueError(
            "batch size/episode limit must be positive and workers non-negative"
        )
    source_commit = _validated_source_commit(
        args.source_commit,
        _current_repository_commit(),
    )
    config = _overrides(load_config(args.config), args)
    bundle = load_mainline_data_for_smoke(
        config,
        split=args.split,
        episode_limit=int(args.episode_limit),
    )
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
    selected_indices = list(bundle.splits[args.split][: int(args.episode_limit)])
    dino_cache = Path(config.data.dino_cache)
    materialized_episodes = []
    for episode_index in selected_indices:
        episode = bundle.episodes[episode_index]
        source_stat = episode.path.stat()
        dino_metadata = dino_cache / episode.cache_key / "meta.json"
        token_array = dino_cache / episode.cache_key / "tokens.float16.npy"
        materialized_episodes.append(
            {
                "episode_index": int(episode_index),
                "episode_id": episode.episode_id,
                "length": int(episode.length),
                "source_size_bytes": int(source_stat.st_size),
                "source_mtime_ns": int(source_stat.st_mtime_ns),
                "dino_metadata": _file_identity(dino_metadata),
                "dino_token_array_path": str(token_array.resolve()),
                "dino_token_array_size_bytes": int(token_array.stat().st_size),
            }
        )
    report = {
        "schema": "clearvla-mainline-data-smoke-v2",
        "source_commit": source_commit,
        "model_constructed": False,
        "optimizer_constructed": False,
        "split": args.split,
        "materialized_episode_limit": int(args.episode_limit),
        "split_sizes": {name: len(dataset) for name, dataset in bundle.datasets.items()},
        "split_metadata": bundle.split_metadata,
        "data_profile": bundle.data_profile_metadata,
        "ordered_cameras": list(config.data.camera_names),
        "camera_keys": config.data.camera_key_map(),
        "gripper_indices": list(bundle.gripper_indices),
        "sampling_gripper_event_threshold": bundle.gripper_event_threshold,
        "goal": bundle.goal.metadata,
        "language_artifact": _file_identity(Path(config.data.t5_condition)),
        "normalizer_identity": {
            "action_sha256": _digest(bundle.action_normalizer.to_dict()),
            "state_sha256": _digest(bundle.state_normalizer.to_dict()),
        },
        "materialized_episodes": materialized_episodes,
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
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is not None:
        destination = args.output.expanduser().resolve()
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite smoke report: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        try:
            temporary.write_text(rendered + "\n", encoding="utf-8")
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
    print(rendered)


if __name__ == "__main__":
    main()
