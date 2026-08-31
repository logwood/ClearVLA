"""Construct one real train-lane typed batch for each selected RDT task."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import default_collate

from clearvla.data.multitask_selection import RDT_MULTITASK_SELECTION_SCHEMA
from clearvla.mainline.config import load_config
from clearvla.mainline.data.dataset import CachedTokenPolicyWindowDataset
from clearvla.mainline.data.loading import load_mainline_data, to_training_batch
from clearvla.mainline.interfaces import TrainingBatch

ACCEPTANCE_SCHEMA = "clearvla-rdt-multitask8-typed-batch-acceptance-v1"
_FULL_GIT_SHA1 = re.compile(r"[0-9a-f]{40}")
_EXPECTED_SHAPES = {
    "goal_tokens": (1, 32, 4096),
    "goal_mask": (1, 32),
    "dino_history": (1, 3, 2, 256, 768),
    "raw_rgb": (1, 3, 2, 3, 336, 336),
    "state": (1, 7),
    "action_state": (1, 7),
    "state_history": (1, 3, 7),
    "executed_action_history": (1, 8, 7),
    "action_target_normalized": (1, 24, 7),
    "action_target_raw": (1, 24, 7),
    "current_action_state_raw": (1, 7),
    "future_dino": (1, 12, 2, 256, 768),
    "future_action": (1, 48, 7),
    "future_state": (1, 48, 7),
    "future_offsets": (1, 12),
}


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, object]:
    source = path.expanduser().resolve()
    return {
        "path": str(source),
        "size_bytes": int(source.stat().st_size),
        "sha256": _file_sha256(source),
    }


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    if tensor.dtype == torch.bfloat16:
        raw = tensor.view(torch.uint16).numpy().tobytes()
    else:
        raw = tensor.numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _shape(value: Tensor) -> dict[str, object]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
        "finite": bool(torch.isfinite(value).all()),
        "storage_sha256": _tensor_sha256(value),
    }


def _typed_tensors(batch: TrainingBatch) -> dict[str, Tensor]:
    return {
        "goal_tokens": batch.online.goal.tokens,
        "goal_mask": batch.online.goal.mask,
        "dino_history": batch.online.observation.dino_history,
        "raw_rgb": batch.online.observation.raw_rgb,
        "state": batch.online.history.state,
        "action_state": batch.online.history.action_state,
        "state_history": batch.online.history.state_history,
        "executed_action_history": batch.online.history.executed_action_history,
        "action_target_normalized": batch.action_target.normalized,
        "action_target_raw": batch.action_target.raw_units,
        "current_action_state_raw": batch.action_target.current_raw_units,
        "future_dino": batch.future.dino_supports,
        "future_action": batch.future.action_sequence,
        "future_state": batch.future.state_sequence,
        "future_offsets": batch.future.offsets,
    }


def _validate_typed(batch: TrainingBatch) -> dict[str, dict[str, object]]:
    tensors = _typed_tensors(batch)
    for name, expected in _EXPECTED_SHAPES.items():
        value = tensors[name]
        if tuple(value.shape) != expected:
            raise ValueError(f"typed field {name} has shape {tuple(value.shape)}, expected {expected}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"typed field {name} contains NaN or infinity")
    expected_offsets = torch.arange(4, 49, 4, dtype=torch.long)[None]
    if not torch.equal(batch.future.offsets.cpu(), expected_offsets):
        raise ValueError("typed future offsets must be exactly 4,8,...,48")
    return {name: _shape(value) for name, value in tensors.items()}


def _current_commit(repository: Path) -> str:
    status = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise ValueError("acceptance requires a clean Git worktree")
    completed = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    if _FULL_GIT_SHA1.fullmatch(value) is None:
        raise ValueError("acceptance requires a full lowercase Git source commit")
    return value


def _verified_selection(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != RDT_MULTITASK_SELECTION_SCHEMA:
        raise ValueError("unsupported RDT multitask selection schema")
    recorded = str(payload.get("selection_sha256", ""))
    digest_payload = dict(payload)
    digest_payload.pop("selection_sha256", None)
    encoded = json.dumps(
        digest_payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != recorded:
        raise ValueError("RDT multitask selection digest is inconsistent")
    return payload


def build_acceptance_report(
    *,
    config_path: Path,
    expected_source_commit: str | None,
) -> dict[str, object]:
    repository = Path(__file__).resolve().parents[2]
    source_commit = _current_commit(repository)
    if expected_source_commit is not None and str(expected_source_commit) != source_commit:
        raise ValueError(
            f"requested source commit {expected_source_commit} != checkout {source_commit}"
        )
    config_source = config_path.expanduser().resolve()
    config = load_config(config_source)
    if (
        config.data.data_profile != "rdt_right_arm_action_chart_v1"
        or tuple(config.data.camera_names) != ("high", "right_wrist")
        or config.dimensions.action_dim != 7
        or config.dimensions.state_dim != 7
        or config.dimensions.action_horizon != 24
        or config.dimensions.num_cameras != 2
    ):
        raise ValueError("acceptance config differs from the first-round RDT ABI")
    selection_path = Path(config.data.task_selection_manifest).expanduser().resolve()
    selection = _verified_selection(selection_path)
    task_order = [str(value) for value in selection.get("task_order", [])]
    task_rows = selection.get("tasks")
    if len(task_order) != 8 or not isinstance(task_rows, list) or len(task_rows) != 8:
        raise ValueError("acceptance requires exactly eight ordered task records")
    task_by_id = {str(value["task_id"]): value for value in task_rows}
    if list(task_by_id) != task_order:
        raise ValueError("selection task records differ from task_order")

    bundle = load_mainline_data(config)
    if set(bundle.datasets) != {"train", "val", "test"}:
        raise ValueError("external_test must not be materialized by the bounded formal loader")
    external_ids = tuple(bundle.splits.get("external_test", ()))
    if not external_ids or any(
        bundle.episodes[index].source_partition != "test" for index in external_ids
    ):
        raise ValueError("external_test no longer identifies only the source /test partition")
    internal_ids = {
        int(index)
        for split in ("train", "val", "test")
        for index in bundle.splits[split]
    }
    if internal_ids.intersection(external_ids) or any(
        bundle.episodes[index].source_partition != "rdt_data" for index in internal_ids
    ):
        raise ValueError("internal and external_test episode identities overlap or changed")
    selection_metadata = bundle.split_metadata.get("task_selection")
    if not isinstance(selection_metadata, dict) or (
        selection_metadata.get("task_order") != task_order
        or selection_metadata.get("model_conditioning") is not False
    ):
        raise ValueError("runtime task selection metadata is stale")
    dataset = bundle.datasets["train"]
    if not isinstance(dataset, CachedTokenPolicyWindowDataset):
        raise TypeError("acceptance requires the cached-token train dataset")

    chosen: dict[str, int] = {}
    for sample_index, ref in enumerate(dataset.base.refs):
        task_id = bundle.episodes[ref.episode_idx].task_id
        if task_id in task_order and task_id not in chosen:
            chosen[task_id] = sample_index
    if list(chosen) != task_order:
        raise ValueError("train dataset does not reproduce all eight tasks in manifest order")

    formal_shuffle_blocker = None
    try:
        bundle.loader(
            "train",
            batch_size=1,
            workers=0,
            device=torch.device("cpu"),
            shuffle=True,
        )
    except ValueError as error:
        formal_shuffle_blocker = str(error)
    if formal_shuffle_blocker is None or "no adopted gripper-event threshold" not in (
        formal_shuffle_blocker
    ):
        raise ValueError("formal shuffled loader did not fail closed on the unresolved threshold")

    per_task: list[dict[str, object]] = []
    dino_root = Path(config.data.dino_cache).expanduser().resolve()
    forbidden_sample_keys = {"task_id", "task_index", "task_condition", "task_embedding"}
    for task_id in task_order:
        sample_index = chosen[task_id]
        ref = dataset.base.refs[sample_index]
        episode = bundle.episodes[ref.episode_idx]
        expected_episode = str(task_by_id[task_id]["splits"]["train"][0])
        if episode.episode_id != expected_episode:
            raise ValueError(f"first deterministic train episode changed for task {task_id}")
        raw_sample = dataset[sample_index]
        leaked = sorted(forbidden_sample_keys.intersection(raw_sample))
        if leaked:
            raise ValueError(f"task identity leaked into the model sample: {leaked}")
        raw = default_collate([raw_sample])
        if int(raw["episode_idx"].item()) != ref.episode_idx:
            raise ValueError("typed sample episode index differs from its CPU reference")
        typed = to_training_batch(
            raw,
            goal=bundle.goal,
            config=config,
            device=torch.device("cpu"),
        )
        tensor_report = _validate_typed(typed)
        condition_index = int(bundle.goal.episode_condition_indices[ref.episode_idx])
        language_identity = task_by_id[task_id]["language_row"]
        expected_language_row = int(language_identity["bank_row"])
        if condition_index != expected_language_row:
            raise ValueError(f"typed language row changed for task {task_id}")
        if int(typed.online.goal.mask[0].sum().item()) != int(
            language_identity["mask_tokens"]
        ):
            raise ValueError(f"typed language mask changed for task {task_id}")
        if _tensor_sha256(typed.online.goal.tokens[0]) != str(
            language_identity["float32_policy_row_sha256"]
        ):
            raise ValueError(f"typed language token row changed for task {task_id}")
        meta_path = dino_root / episode.cache_key / "meta.json"
        token_path = dino_root / episode.cache_key / "tokens.float16.npy"
        cache_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if cache_meta.get("cameras") != ["high", "left_wrist", "right_wrist"]:
            raise ValueError("DINO storage camera order differs from the reusable cache ABI")
        token_array = np.load(token_path, mmap_mode="r")
        if tuple(token_array.shape) != (episode.length, 3, 256, 768):
            raise ValueError(f"DINO storage shape changed for {episode.episode_id}")
        memberships = [
            split for split, indices in bundle.splits.items() if ref.episode_idx in indices
        ]
        if memberships != ["train"]:
            raise ValueError(f"accepted episode has ambiguous split membership: {memberships}")
        per_task.append(
            {
                "task_id": task_id,
                "instruction": episode.instruction,
                "episode_id": episode.episode_id,
                "episode_index": int(ref.episode_idx),
                "sample_index": int(sample_index),
                "center": int(ref.center),
                "split": "train",
                "split_membership": memberships,
                "language_bank_row": condition_index,
                "task_id_model_conditioning": False,
                "model_sample_keys": sorted(raw_sample),
                "dino_cache": {
                    "metadata": _file_identity(meta_path),
                    "token_array_path": str(token_path),
                    "token_array_size_bytes": int(token_path.stat().st_size),
                    "storage_shape": list(token_array.shape),
                    "storage_dtype": str(token_array.dtype),
                    "storage_cameras": cache_meta["cameras"],
                    "model_selected_cameras": list(config.data.camera_names),
                },
                "typed": tensor_report,
                "accepted": True,
            }
        )

    cache_report = dino_root / "cache_report.json"
    report: dict[str, object] = {
        "schema": ACCEPTANCE_SCHEMA,
        "source_commit": source_commit,
        "model_constructed": False,
        "optimizer_constructed": False,
        "backward_executed": False,
        "formal_training_started": False,
        "config": _file_identity(config_source),
        "selection_manifest": _file_identity(selection_path),
        "selection_sha256": selection["selection_sha256"],
        "language_bank": _file_identity(Path(config.data.t5_condition)),
        "normalizer_artifact": _file_identity(Path(config.data.normalizer_artifact)),
        "normalizer_metadata": bundle.normalizer_metadata,
        "dino_cache_report": _file_identity(cache_report),
        "model_abi": {
            "action_dim": 7,
            "state_dim": 7,
            "action_horizon": 24,
            "model_cameras": ["high", "right_wrist"],
            "cache_camera_order": ["high", "left_wrist", "right_wrist"],
            "depth_consumed": False,
            "native_bimanual_consumed": False,
            "three_camera_model_consumed": False,
        },
        "task_order": task_order,
        "accepted_task_count": len(per_task),
        "split_counts": {name: len(indices) for name, indices in bundle.splits.items()},
        "external_test": {
            "episode_count": len(external_ids),
            "materialized": False,
            "used_for_training_or_tuning": False,
            "all_source_partition_test": True,
        },
        "task_id_usage": "cpu_audit_sampling_logging_metadata_only",
        "task_id_model_conditioning": False,
        "gripper_threshold": {
            "adopted": bundle.gripper_event_threshold,
            "formal_shuffled_loader_ready": False,
            "fail_closed_message": formal_shuffle_blocker,
        },
        "tasks": per_task,
    }
    identity = {
        "source_commit": source_commit,
        "task_order": task_order,
        "episodes": [value["episode_id"] for value in per_task],
        "splits": [value["split"] for value in per_task],
        "centers": [value["center"] for value in per_task],
        "selection_sha256": selection["selection_sha256"],
        "language_file_sha256": report["language_bank"]["sha256"],
        "normalizer_file_sha256": report["normalizer_artifact"]["sha256"],
        "cache_report_sha256": report["dino_cache_report"]["sha256"],
        "typed_tensor_sha256": [
            {
                name: details["storage_sha256"]
                for name, details in value["typed"].items()
            }
            for value in per_task
        ],
    }
    report["construction_identity"] = identity
    report["construction_sha256"] = _canonical_digest(identity)
    report["acceptance_sha256"] = _canonical_digest(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Construct and verify one real typed batch for every selected RDT task"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-commit")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite acceptance report: {destination}")
    report = build_acceptance_report(
        config_path=args.config,
        expected_source_commit=args.source_commit,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(
        json.dumps(
            {
                "output": str(destination),
                "accepted_task_count": report["accepted_task_count"],
                "construction_sha256": report["construction_sha256"],
                "acceptance_sha256": report["acceptance_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


__all__ = ["ACCEPTANCE_SCHEMA", "build_acceptance_report"]
