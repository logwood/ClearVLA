"""Fit one shared right-arm normalizer from the selected train lane only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np

from clearvla.data.action_chart import resolve_action_state_profile
from clearvla.data.hdf5_episode import LoadedEpisode, episode_identity, find_hdf5_files
from clearvla.data.multitask_selection import RDT_MULTITASK_SELECTION_SCHEMA
from clearvla.mainline.data.normalizer import ArrayNormalizer
from clearvla.mainline.data.normalizer_artifact import (
    SHARED_NORMALIZER_SCHEMA,
    canonical_digest,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selection_digest(value: object) -> str:
    """Match the selection schema's UTF-8 canonicalization exactly."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_shared_normalizer(
    *,
    data_root: Path,
    selection_manifest: Path,
    profile_name: str = "rdt_right_arm_action_chart_v1",
) -> dict[str, object]:
    root = data_root.expanduser().resolve()
    selection_path = selection_manifest.expanduser().resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not isinstance(selection, dict) or selection.get("schema") != RDT_MULTITASK_SELECTION_SCHEMA:
        raise ValueError("unsupported RDT multitask selection schema")
    recorded = str(selection.get("selection_sha256", ""))
    digest_payload = dict(selection)
    digest_payload.pop("selection_sha256", None)
    if recorded != _selection_digest(digest_payload):
        raise ValueError("RDT multitask selection digest is inconsistent")
    splits = selection.get("splits")
    if not isinstance(splits, dict) or not isinstance(splits.get("train"), list):
        raise ValueError("RDT multitask selection train identities are missing")
    train_ids = [str(value) for value in splits["train"]]
    if not train_ids or len(set(train_ids)) != len(train_ids):
        raise ValueError("selected train identities must be non-empty and unique")
    paths = {
        episode_identity(root, path)[0]: path
        for path in find_hdf5_files(root, "**/*.hdf5")
    }
    profile = resolve_action_state_profile(profile_name)
    action_rows: list[np.ndarray] = []
    state_rows: list[np.ndarray] = []
    row_counts: list[dict[str, object]] = []
    for identity in train_ids:
        path = paths.get(identity)
        if path is None:
            raise FileNotFoundError(f"selected train episode disappeared: {identity}")
        with h5py.File(path, "r") as handle:
            action = np.asarray(handle["action"], dtype=np.float32)
            qpos = np.asarray(handle["observations/qpos"], dtype=np.float32)
        native = LoadedEpisode(
            path=path,
            episode_id=identity,
            source_partition="rdt_data",
            task_id=identity.split("/")[1],
            action_key="action",
            camera_keys={},
            actions_raw=action,
            state_key="observations/qpos",
            states_raw=qpos,
            action_states_raw=qpos,
            source_action_dim=int(action.shape[1]),
            source_state_dim=int(qpos.shape[1]),
        )
        projected = profile.project_episode(native)
        if projected.states_raw is None:
            raise AssertionError("profile projection lost qpos state")
        action_rows.append(projected.actions_raw)
        state_rows.append(projected.states_raw)
        row_counts.append({"episode_id": identity, "rows": int(projected.length)})
    action = ArrayNormalizer.fit_zscore(action_rows)
    state = ArrayNormalizer.fit_zscore(state_rows)
    payload: dict[str, object] = {
        "schema": SHARED_NORMALIZER_SCHEMA,
        "fit_scope": "one_shared_normalizer_over_selected_train_split_only",
        "per_task_normalizers": False,
        "selection_manifest": {
            "file_sha256": _file_sha256(selection_path),
            "selection_sha256": recorded,
        },
        "action_profile": {
            "name": profile.name,
            "sha256": profile.digest(),
            "action_chart": profile.action_chart,
            "state_chart": profile.state_chart,
            "action_indices": list(profile.action_indices),
            "state_indices": list(profile.state_indices),
            "state_to_action_scale": list(profile.state_to_action_scale),
        },
        "train_episode_count": len(train_ids),
        "train_row_count": sum(int(value.shape[0]) for value in action_rows),
        "task_order": list(selection.get("task_order", [])),
        "train_episode_inventory_sha256": canonical_digest(train_ids),
        "train_episode_rows_sha256": canonical_digest(row_counts),
        "action": action.to_dict(),
        "state": state.to_dict(),
    }
    payload["normalizer_sha256"] = canonical_digest(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one shared train-only RDT normalizer")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--profile", default="rdt_right_arm_action_chart_v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    destination = args.output.expanduser().resolve()
    if destination.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite shared normalizer: {destination}")
    payload = build_shared_normalizer(
        data_root=args.data_root,
        selection_manifest=args.selection_manifest,
        profile_name=args.profile,
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
                "normalizer_sha256": payload["normalizer_sha256"],
                "train_episode_count": payload["train_episode_count"],
                "train_row_count": payload["train_row_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


__all__ = ["build_shared_normalizer"]
