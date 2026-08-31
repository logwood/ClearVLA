"""Build one deterministic, content-verified eight-task RDT selection."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from clearvla.data.action_chart import resolve_action_state_profile
from clearvla.data.hdf5_episode import episode_identity, find_hdf5_files
from clearvla.data.multitask_selection import RDT_MULTITASK_SELECTION_SCHEMA
from clearvla.data.split import RDT_TYPED_WINDOW_MIN_EPISODE_LENGTH
from clearvla.mainline.data.language import load_t5_condition_bank
from clearvla.vision.image_io import decode_image_value

TASK_AUDIT_SCHEMA = "clearvla-rdt-multitask-task-audit-v1"
SELECTION_SPEC_SCHEMA = "clearvla-rdt-multitask-selection-spec-v1"
CAMERA_KEYS = {
    "high": "observations/images/cam_high",
    "right_wrist": "observations/images/cam_right_wrist",
}
CACHE_CAMERAS = ("high", "left_wrist", "right_wrist")
DINO_PATCHES = 256
DINO_WIDTH = 768
DINO_DTYPE = np.dtype(np.float16)


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
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


def _tensor_storage_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    if tensor.dtype == torch.bfloat16:
        raw = tensor.view(torch.uint16).numpy().tobytes()
    else:
        raw = tensor.numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _verified_json(path: Path, *, schema: str, digest_field: str | None = None) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != schema:
        raise ValueError(f"{path}: unsupported schema")
    if digest_field is not None:
        recorded = str(payload.get(digest_field, ""))
        copy = dict(payload)
        copy.pop(digest_field, None)
        if recorded != _canonical_digest(copy):
            raise ValueError(f"{path}: content digest is inconsistent")
    return payload


def _load_contact_sheet_evidence(path: Path) -> dict[str, dict[str, object]]:
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list):
        raise ValueError("contact sheet manifest must be a list")
    rows: dict[str, dict[str, object]] = {}
    for value in values:
        if not isinstance(value, dict) or value.get("camera") != "high":
            continue
        task = str(value.get("task_id", ""))
        if not task or task in rows:
            raise ValueError("contact sheet task identity is missing or duplicated")
        source = Path(str(value.get("path", "")))
        if not source.is_file() or _file_sha256(source) != str(value.get("sha256", "")):
            raise ValueError(f"contact sheet evidence is missing or stale: {task}")
        rows[task] = {
            "camera": "high",
            "episodes": int(value["episodes"]),
            "frames_per_episode": int(value["frames_per_episode"]),
            "file_sha256": str(value["sha256"]),
            "size_bytes": int(value["size_bytes"]),
        }
    return rows


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _npy_size(shape: tuple[int, ...], dtype: np.dtype[Any]) -> int:
    header = {
        "descr": np.lib.format.dtype_to_descr(dtype),
        "fortran_order": False,
        "shape": shape,
    }
    buffer = io.BytesIO()
    np.lib.format.write_array_header_1_0(buffer, header)
    return len(buffer.getvalue()) + int(np.prod(shape, dtype=np.int64)) * int(dtype.itemsize)


def _episode_audit(
    path: Path,
    *,
    root: Path,
    expected_split: str,
    expected_instruction: str,
    verify_all_rgb: bool,
) -> dict[str, Any]:
    identity, partition, task = episode_identity(root, path)
    if partition != "rdt_data":
        raise ValueError(f"selected episode is outside rdt_data: {identity}")
    with h5py.File(path, "r") as handle:
        action = np.asarray(handle["action"], dtype=np.float32)
        qpos = np.asarray(handle["observations/qpos"], dtype=np.float32)
        instruction_value = handle["instruction"][()]
        if isinstance(instruction_value, np.ndarray) and instruction_value.shape == ():
            instruction_value = instruction_value.item()
        instruction = (
            bytes(instruction_value).decode("utf-8")
            if isinstance(instruction_value, (bytes, np.bytes_))
            else str(instruction_value)
        )
        if instruction != expected_instruction:
            raise ValueError(f"selected episode instruction changed: {identity}")
        if action.shape != qpos.shape or action.ndim != 2 or action.shape[1] != 14:
            raise ValueError(f"selected episode is not native aligned 14D: {identity}")
        if not np.isfinite(action).all() or not np.isfinite(qpos).all():
            raise ValueError(f"selected episode has non-finite action/qpos: {identity}")
        length = int(action.shape[0])
        valid_windows = max(length - RDT_TYPED_WINDOW_MIN_EPISODE_LENGTH + 1, 0)
        if valid_windows <= 0:
            raise ValueError(f"selected episode cannot form a typed window: {identity}")

        per_camera: dict[str, object] = {}
        for camera, key in CAMERA_KEYS.items():
            dataset = handle.get(key)
            if (
                not isinstance(dataset, h5py.Dataset)
                or dataset.ndim != 1
                or int(dataset.shape[0]) != length
                or dataset.dtype.kind not in {"S", "O"}
            ):
                raise ValueError(f"selected episode camera is incomplete: {identity}/{camera}")
            digest = hashlib.sha256()
            shape_counts: Counter[tuple[int, int, int]] = Counter()
            decoded_count = 0
            if verify_all_rgb:
                for frame in range(length):
                    encoded = dataset[frame]
                    if isinstance(encoded, np.ndarray) and encoded.shape == ():
                        encoded = encoded.item()
                    if isinstance(encoded, np.bytes_):
                        encoded = bytes(encoded)
                    if not isinstance(encoded, (bytes, bytearray)):
                        encoded_array = np.asarray(encoded, dtype=np.uint8).reshape(-1)
                        encoded = encoded_array.tobytes()
                    digest.update(len(encoded).to_bytes(8, byteorder="little", signed=False))
                    digest.update(encoded)
                    decoded = decode_image_value(encoded)
                    if (
                        decoded.ndim != 3
                        or decoded.shape[-1] != 3
                        or decoded.dtype != np.uint8
                        or not decoded.size
                    ):
                        raise ValueError(
                            f"selected RGB decode is invalid: {identity}/{camera}/{frame}"
                        )
                    shape_counts[tuple(int(value) for value in decoded.shape)] += 1
                    decoded_count += 1
            per_camera[camera] = {
                "key": key,
                "rows": length,
                "all_rows_decoded": bool(verify_all_rgb),
                "decoded_rows": decoded_count,
                "encoded_stream_sha256": digest.hexdigest() if verify_all_rgb else None,
                "decoded_shape_counts": {
                    "x".join(str(value) for value in shape): count
                    for shape, count in sorted(shape_counts.items())
                },
            }

    action_step = np.diff(action, axis=0).astype(np.float64)
    qpos_step = np.diff(qpos, axis=0).astype(np.float64)
    boundary = np.concatenate((qpos[:1], action[:-1]), axis=0)
    gripper_delta = action - boundary
    side_rows: dict[str, object] = {}
    for side, joint_slice, gripper_index in (
        ("left", slice(0, 6), 6),
        ("right", slice(7, 13), 13),
    ):
        action_rms = float(np.sqrt(np.mean(np.square(action_step[:, joint_slice]))))
        qpos_rms = float(np.sqrt(np.mean(np.square(qpos_step[:, joint_slice]))))
        side_rows[side] = {
            "joint_action_step_rms": action_rms,
            "joint_qpos_step_rms": qpos_rms,
            "gripper_action_range": float(np.ptp(action[:, gripper_index])),
            "gripper_boundary_abs_delta_total_variation": float(
                np.abs(gripper_delta[:, gripper_index]).sum(dtype=np.float64)
            ),
        }
    side_rows["left_to_right_joint_step_rms_ratio"] = {
        "action": float(
            side_rows["left"]["joint_action_step_rms"]
            / max(side_rows["right"]["joint_action_step_rms"], np.finfo(float).tiny)
        ),
        "qpos": float(
            side_rows["left"]["joint_qpos_step_rms"]
            / max(side_rows["right"]["joint_qpos_step_rms"], np.finfo(float).tiny)
        ),
    }
    stat = path.stat()
    return {
        "episode_id": identity,
        "task_id": task,
        "split": expected_split,
        "length": length,
        "valid_windows": valid_windows,
        "source_size_bytes": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
        "action_storage_sha256": _array_sha256(action),
        "qpos_storage_sha256": _array_sha256(qpos),
        "activity": side_rows,
        "camera": per_camera,
    }


def build_rdt_multitask_selection(
    *,
    data_root: Path,
    split_manifest: Path,
    task_audit: Path,
    selection_spec: Path,
    contact_sheet_manifest: Path,
    language_bank: Path,
    verify_all_rgb: bool,
) -> dict[str, Any]:
    root = data_root.expanduser().resolve()
    split_path = split_manifest.expanduser().resolve()
    audit_path = task_audit.expanduser().resolve()
    spec_path = selection_spec.expanduser().resolve()
    contact_path = contact_sheet_manifest.expanduser().resolve()
    language_path = language_bank.expanduser().resolve()
    split = _verified_json(
        split_path,
        schema="clearvla-rdt-per-task-split-v2",
        digest_field="manifest_sha256",
    )
    audit = _verified_json(audit_path, schema=TASK_AUDIT_SCHEMA, digest_field="audit_sha256")
    spec = _verified_json(spec_path, schema=SELECTION_SPEC_SCHEMA)
    if int(audit.get("rdt_data_task_count", -1)) != 302:
        raise ValueError("task audit must cover all 302 rdt_data tasks")
    audit_split = audit.get("split_manifest")
    if not isinstance(audit_split, dict) or (
        str(audit_split.get("file_sha256", "")) != _file_sha256(split_path)
        or str(audit_split.get("manifest_sha256", "")) != split["manifest_sha256"]
    ):
        raise ValueError("task audit was not built from the selected split manifest")
    task_order_value = spec.get("task_order")
    task_spec_value = spec.get("tasks")
    if (
        not isinstance(task_order_value, list)
        or len(task_order_value) != 8
        or len(set(task_order_value)) != 8
        or not isinstance(task_spec_value, dict)
        or set(task_order_value) != set(task_spec_value)
    ):
        raise ValueError("selection spec must name exactly eight ordered task records")
    task_order = tuple(str(value) for value in task_order_value)
    role_review = spec.get("role_review")
    if not isinstance(role_review, dict) or role_review.get(
        "required_support_or_collaboration"
    ) is not False:
        raise ValueError("selection spec must record a negative left-role review")
    model_abi = spec.get("model_abi")
    if not isinstance(model_abi, dict) or model_abi != {
        "action_profile": "rdt_right_arm_action_chart_v1",
        "action_dim": 7,
        "action_horizon": 24,
        "model_cameras": ["high", "right_wrist"],
        "cache_camera_order": list(CACHE_CAMERAS),
        "depth_consumed": False,
        "native_bimanual_consumed": False,
        "three_camera_model_consumed": False,
    }:
        raise ValueError("selection model ABI differs from the bounded first-round contract")
    contact = _load_contact_sheet_evidence(contact_path)
    missing_contact = sorted(set(task_order) - set(contact))
    if missing_contact:
        raise ValueError(f"selected tasks lack all-episode contact sheets: {missing_contact}")

    audit_tasks_value = audit.get("tasks")
    if not isinstance(audit_tasks_value, list):
        raise ValueError("task audit records are missing")
    audit_tasks = {
        str(value["task_id"]): value for value in audit_tasks_value if isinstance(value, dict)
    }
    if len(audit_tasks) != 302:
        raise ValueError("task audit contains duplicate or missing task records")

    language = load_t5_condition_bank(
        language_path,
        max_tokens=32,
        expected_width=4096,
    )
    if not language.is_instruction_bank:
        raise ValueError("RDT multitask selection requires a typed instruction bank")
    language_index = {text: index for index, text in enumerate(language.instructions)}
    language_payload = torch.load(language_path, map_location="cpu", weights_only=False)
    embedding_records = language_payload.get("embedding_records")
    if not isinstance(embedding_records, list):
        raise ValueError("language bank lacks precomputed-row provenance")

    path_by_identity = {
        episode_identity(root, path)[0]: path
        for path in find_hdf5_files(root, "**/*.hdf5")
    }
    raw_splits = split.get("splits")
    if not isinstance(raw_splits, dict):
        raise ValueError("base split identities are missing")
    base_membership = {
        str(identity): split_name
        for split_name, identities in raw_splits.items()
        for identity in identities
    }
    selected_splits: dict[str, list[str]] = {name: [] for name in ("train", "val", "test")}
    task_records: list[dict[str, Any]] = []
    selected_episode_rows: list[dict[str, Any]] = []
    for task_id in task_order:
        source_record = audit_tasks.get(task_id)
        if source_record is None:
            raise ValueError(f"selected task is absent from the 302-task audit: {task_id}")
        instruction = str(source_record.get("instruction", ""))
        split_counts = source_record.get("split_episode_counts")
        native = source_record.get("native_profile")
        camera = source_record.get("camera_audit")
        episodes_value = source_record.get("episodes")
        if (
            not isinstance(split_counts, dict)
            or any(int(split_counts.get(name, 0)) <= 0 for name in ("train", "val", "test"))
            or not isinstance(native, dict)
            or native.get("finite_all") is not True
            or native.get("action_widths") != {"14": int(source_record["source_episode_count"])}
            or native.get("qpos_widths") != {"14": int(source_record["source_episode_count"])}
            or not isinstance(camera, dict)
            or camera.get("header_complete_all") is not True
            or camera.get("decode_errors") != []
            or not isinstance(episodes_value, list)
        ):
            raise ValueError(f"selected task fails a structural hard condition: {task_id}")
        if instruction not in language_index:
            raise ValueError(f"selected task has no language row: {task_id}")
        row_index = language_index[instruction]
        provenance = embedding_records[row_index]
        if not isinstance(provenance, dict):
            raise ValueError("selected language provenance row is malformed")
        task_split_rows: dict[str, list[str]] = {name: [] for name in selected_splits}
        task_episode_rows: list[dict[str, Any]] = []
        for summary in episodes_value:
            if not isinstance(summary, dict):
                raise ValueError("task audit episode row is malformed")
            episode_id = str(summary["episode_id"])
            split_name = str(summary["split"])
            if split_name == "excluded_too_short":
                continue
            if split_name not in selected_splits or base_membership.get(episode_id) != split_name:
                raise ValueError(f"selected episode split differs from the base manifest: {episode_id}")
            path = path_by_identity.get(episode_id)
            if path is None:
                raise FileNotFoundError(f"selected episode disappeared: {episode_id}")
            episode_row = _episode_audit(
                path,
                root=root,
                expected_split=split_name,
                expected_instruction=instruction,
                verify_all_rgb=verify_all_rgb,
            )
            task_split_rows[split_name].append(episode_id)
            task_episode_rows.append(episode_row)
        for split_name in selected_splits:
            task_split_rows[split_name].sort()
            if not task_split_rows[split_name]:
                raise ValueError(f"selected task has an empty {split_name} lane: {task_id}")
            selected_splits[split_name].extend(task_split_rows[split_name])
        spec_record = task_spec_value[task_id]
        if not isinstance(spec_record, dict):
            raise ValueError("selection task rationale is malformed")
        task_record = {
            "task_id": task_id,
            "instruction": instruction,
            "instruction_sha256": hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
            "behavior_tags": list(spec_record["behavior_tags"]),
            "selection_reason": str(spec_record["selection_reason"]),
            "left_role_audit": {
                "required_support_or_collaboration": False,
                "reason": str(spec_record["left_role_reason"]),
                "numeric_scope": role_review["numeric_evidence"],
                "visual_scope": role_review["visual_evidence"],
                "contact_sheet": contact[task_id],
                "task_complete_activity": source_record["activity"],
                "per_episode_activity": [
                    {
                        "episode_id": row["episode_id"],
                        "activity": row["activity"],
                    }
                    for row in task_episode_rows
                ],
            },
            "source_episode_count": int(source_record["source_episode_count"]),
            "eligible_episode_count": len(task_episode_rows),
            "splits": task_split_rows,
            "split_episode_counts": {
                name: len(values) for name, values in task_split_rows.items()
            },
            "split_valid_window_counts": {
                name: sum(
                    int(row["valid_windows"])
                    for row in task_episode_rows
                    if row["split"] == name
                )
                for name in selected_splits
            },
            "valid_window_count": sum(int(row["valid_windows"]) for row in task_episode_rows),
            "action_profile": {
                **resolve_action_state_profile("rdt_right_arm_action_chart_v1").as_dict(),
                "sha256": resolve_action_state_profile(
                    "rdt_right_arm_action_chart_v1"
                ).digest(),
            },
            "camera_profile": {
                "source_verified": ["high", "right_wrist"],
                "model_order": ["high", "right_wrist"],
                "cache_order": list(CACHE_CAMERAS),
                "all_selected_rows_decoded": bool(verify_all_rgb),
            },
            "language_row": {
                "bank_row": row_index,
                "mask_tokens": int(language.mask[row_index].sum().item()),
                "float32_policy_row_sha256": _tensor_storage_sha256(
                    language.tokens[row_index]
                ),
                "selected_relative_path": str(provenance["selected_relative_path"]),
                "selected_file_sha256": str(provenance["selected_file_sha256"]),
                "selected_tensor_sha256": str(provenance["selected_tensor_sha256"]),
                "selected_policy_tensor_sha256": str(
                    provenance["selected_policy_tensor_sha256"]
                ),
            },
            "episodes": task_episode_rows,
        }
        task_records.append(task_record)
        selected_episode_rows.extend(task_episode_rows)

    external_names = [str(value) for value in raw_splits["external_test"]]
    exact_npy_bytes = sum(
        _npy_size(
            (int(row["length"]), len(CACHE_CAMERAS), DINO_PATCHES, DINO_WIDTH),
            DINO_DTYPE,
        )
        for row in selected_episode_rows
    )
    raw_token_bytes = sum(
        int(row["length"])
        * len(CACHE_CAMERAS)
        * DINO_PATCHES
        * DINO_WIDTH
        * DINO_DTYPE.itemsize
        for row in selected_episode_rows
    )
    payload: dict[str, Any] = {
        "schema": RDT_MULTITASK_SELECTION_SCHEMA,
        "policy": {
            "task_count": 8,
            "task_id_usage": "cpu_audit_sampling_logging_metadata_only",
            "model_conditioning": False,
            "episode_scope": "all_typed_window_eligible_episodes_per_selected_task",
            "external_test": "preserved_external_only_not_selected_or_tuned",
        },
        "task_order": list(task_order),
        "tasks": task_records,
        "splits": selected_splits,
        "split_counts": {name: len(values) for name, values in selected_splits.items()},
        "split_valid_window_counts": {
            name: sum(
                int(row["valid_windows"])
                for row in selected_episode_rows
                if row["split"] == name
            )
            for name in selected_splits
        },
        "external_test_identity": {
            "episode_count": len(external_names),
            "episode_inventory_sha256": _canonical_digest(external_names),
            "membership": "base_manifest_external_test_only",
            "selected": False,
            "used_for_training_or_tuning": False,
        },
        "base_split_manifest": {
            "file_sha256": _file_sha256(split_path),
            "manifest_sha256": split["manifest_sha256"],
            "source_episode_inventory_sha256": split["source_episode_inventory_sha256"],
            "episode_inventory_sha256": split["episode_inventory_sha256"],
        },
        "task_audit": {
            "file_sha256": _file_sha256(audit_path),
            "audit_sha256": audit["audit_sha256"],
            "rdt_data_task_count": audit["rdt_data_task_count"],
            "eligible_rdt_data_episode_count": audit["eligible_rdt_data_episode_count"],
        },
        "selection_spec": {
            "file_sha256": _file_sha256(spec_path),
            "schema": spec["schema"],
        },
        "language_bank": {
            "path": str(language_path),
            "file_sha256": _file_sha256(language_path),
            "size_bytes": int(language_path.stat().st_size),
            "schema": language.metadata["schema"],
            "encoder_id": language.metadata["encoder_id"],
            "instructions": language.metadata["instructions"],
            "source_episode_count": language.metadata["source_episode_count"],
            "source_instruction_inventory_sha256": language.metadata[
                "source_instruction_inventory_sha256"
            ],
            "embedding_source": language.metadata.get("embedding_source"),
            "embedding_inventory_sha256": language.metadata.get(
                "embedding_inventory_sha256"
            ),
        },
        "model_abi": model_abi,
        "dino_cache_estimate": {
            "selected_episode_count": len(selected_episode_rows),
            "selected_frame_count": sum(int(row["length"]) for row in selected_episode_rows),
            "camera_order": list(CACHE_CAMERAS),
            "shape_per_frame": [len(CACHE_CAMERAS), DINO_PATCHES, DINO_WIDTH],
            "dtype": "float16",
            "raw_token_bytes": raw_token_bytes,
            "exact_npy_file_bytes": exact_npy_bytes,
            "npy_header_bytes": exact_npy_bytes - raw_token_bytes,
            "episode_meta_and_cache_report_excluded": True,
            "filesystem_allocation_rounding_excluded": True,
        },
    }
    payload["selection_sha256"] = _canonical_digest(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deterministic RDT multitask8 selection")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--task-audit", type=Path, required=True)
    parser.add_argument("--selection-spec", type=Path, required=True)
    parser.add_argument("--contact-sheet-manifest", type=Path, required=True)
    parser.add_argument("--language-bank", type=Path, required=True)
    parser.add_argument("--verify-all-rgb", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    destination = args.output.expanduser().resolve()
    if destination.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite multitask selection: {destination}")
    payload = build_rdt_multitask_selection(
        data_root=args.data_root,
        split_manifest=args.split_manifest,
        task_audit=args.task_audit,
        selection_spec=args.selection_spec,
        contact_sheet_manifest=args.contact_sheet_manifest,
        language_bank=args.language_bank,
        verify_all_rgb=bool(args.verify_all_rgb),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
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
                "selection_sha256": payload["selection_sha256"],
                "split_counts": payload["split_counts"],
                "dino_cache_estimate": payload["dino_cache_estimate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


__all__ = ["build_rdt_multitask_selection"]
