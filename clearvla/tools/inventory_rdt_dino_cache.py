"""Hash and validate every persisted array in the bounded RDT DINO cache."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np

from clearvla.data.multitask_selection import RDT_MULTITASK_SELECTION_SCHEMA

CACHE_INVENTORY_SCHEMA = "clearvla-rdt-multitask-dino-cache-inventory-v1"
EXPECTED_CAMERAS = ("high", "left_wrist", "right_wrist")


def _canonical_digest(value: object, *, ensure_ascii: bool = True) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=ensure_ascii,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_selection(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != RDT_MULTITASK_SELECTION_SCHEMA:
        raise ValueError("unsupported RDT multitask selection schema")
    recorded = str(payload.get("selection_sha256", ""))
    digest_payload = dict(payload)
    digest_payload.pop("selection_sha256", None)
    if recorded != _canonical_digest(digest_payload, ensure_ascii=False):
        raise ValueError("RDT multitask selection digest is inconsistent")
    return payload


def _package_version(*names: str) -> str | None:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _model_provenance(model_cache_root: Path) -> dict[str, object]:
    root = model_cache_root.expanduser().resolve()
    reference = root / "refs" / "main"
    if reference.is_file():
        revision = reference.read_text(encoding="utf-8").strip()
    else:
        snapshots = sorted(path for path in (root / "snapshots").iterdir() if path.is_dir())
        if len(snapshots) != 1:
            raise ValueError("DINO model cache needs refs/main or exactly one local snapshot")
        revision = snapshots[0].name
    if not revision or "/" in revision or "\\" in revision:
        raise ValueError("DINO model cache revision is invalid")
    snapshot = root / "snapshots" / revision
    if not snapshot.is_dir():
        raise FileNotFoundError(f"DINO model snapshot is missing: {snapshot}")
    files: list[dict[str, object]] = []
    for path in sorted((value for value in snapshot.rglob("*") if value.is_file())):
        files.append(
            {
                "relative_path": path.relative_to(snapshot).as_posix(),
                "size_bytes": int(path.stat().st_size),
                "sha256": _file_sha256(path),
            }
        )
    if not files:
        raise ValueError("DINO model snapshot contains no files")
    return {
        "model_id": "facebook/dinov2-base",
        "cache_root": str(root),
        "revision": revision,
        "snapshot_path": str(snapshot),
        "files": files,
        "snapshot_inventory_sha256": _canonical_digest(files),
    }


def inventory_rdt_dino_cache(
    *,
    cache_dir: Path,
    selection_manifest: Path,
    model_cache_root: Path,
) -> dict[str, object]:
    cache_root = cache_dir.expanduser().resolve()
    selection_path = selection_manifest.expanduser().resolve()
    selection = _verified_selection(selection_path)
    expected_ids = [
        str(episode_id)
        for split in ("train", "val", "test")
        for episode_id in selection["splits"][split]
    ]
    estimate = selection.get("dino_cache_estimate")
    if not isinstance(estimate, dict) or estimate.get("camera_order") != list(
        EXPECTED_CAMERAS
    ):
        raise ValueError("selection DINO estimate differs from the cache inventory ABI")
    report_path = cache_root / "cache_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("schema") != "clearvla-rdt2-dinov2-token-cache-v1"
        or int(report.get("episodes", -1)) != len(expected_ids)
        or report.get("cameras") != list(EXPECTED_CAMERAS)
        or report.get("dtype") != "float16"
        or report.get("image_source_mode") != "hdf5-direct"
        or report.get("dinov2_model") != "facebook/dinov2-base"
    ):
        raise ValueError("DINO cache report differs from the selected HDF5-direct ABI")
    report_selection = report.get("selection")
    if not isinstance(report_selection, dict) or report_selection.get(
        "selected_episode_ids"
    ) != expected_ids:
        raise ValueError("DINO cache report episode order differs from the selection")
    report_task_selection = report_selection.get("task_selection")
    if not isinstance(report_task_selection, dict) or str(
        report_task_selection.get("selection_sha256", "")
    ) != str(selection["selection_sha256"]):
        raise ValueError("DINO cache report task selection identity is stale")
    report_rows = report.get("episode_meta")
    if not isinstance(report_rows, list) or [
        str(value.get("episode_stem", "")) for value in report_rows
    ] != expected_ids:
        raise ValueError("DINO cache report metadata order differs from the selection")

    entries: list[dict[str, object]] = []
    token_rows: list[dict[str, object]] = []
    metadata_rows: list[dict[str, object]] = []
    total_frames = 0
    total_token_bytes = 0
    total_metadata_bytes = 0
    for expected_id, report_row in zip(expected_ids, report_rows, strict=True):
        episode_dir = cache_root / expected_id
        meta_path = episode_dir / "meta.json"
        token_path = episode_dir / "tokens.float16.npy"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta != report_row:
            raise ValueError(f"cache report metadata differs from meta.json: {expected_id}")
        if (
            meta.get("episode_stem") != expected_id
            or meta.get("cameras") != list(EXPECTED_CAMERAS)
            or meta.get("dtype") != "float16"
            or int(meta.get("tokens_per_camera", -1)) != 256
            or int(meta.get("token_dim", -1)) != 768
        ):
            raise ValueError(f"DINO episode metadata differs from the cache ABI: {expected_id}")
        array = np.load(token_path, mmap_mode="r")
        frames = int(meta["num_frames"])
        expected_shape = (frames, 3, 256, 768)
        if tuple(array.shape) != expected_shape or array.dtype != np.float16:
            raise ValueError(
                f"DINO array differs from metadata: {expected_id}/{array.shape}/{array.dtype}"
            )
        meta_size = int(meta_path.stat().st_size)
        token_size = int(token_path.stat().st_size)
        meta_hash = _file_sha256(meta_path)
        token_hash = _file_sha256(token_path)
        metadata_identity = {
            "episode_id": expected_id,
            "size_bytes": meta_size,
            "sha256": meta_hash,
        }
        token_identity = {
            "episode_id": expected_id,
            "size_bytes": token_size,
            "sha256": token_hash,
        }
        metadata_rows.append(metadata_identity)
        token_rows.append(token_identity)
        entries.append(
            {
                "episode_id": expected_id,
                "frames": frames,
                "shape": list(expected_shape),
                "dtype": "float16",
                "metadata": metadata_identity,
                "tokens": token_identity,
            }
        )
        total_frames += frames
        total_metadata_bytes += meta_size
        total_token_bytes += token_size
    if (
        total_frames != int(estimate.get("selected_frame_count", -1))
        or total_token_bytes != int(estimate.get("exact_npy_file_bytes", -1))
    ):
        raise ValueError("actual DINO array inventory differs from the exact selection estimate")

    payload: dict[str, object] = {
        "schema": CACHE_INVENTORY_SCHEMA,
        "cache_dir": str(cache_root),
        "selection_manifest": {
            "path": str(selection_path),
            "file_sha256": _file_sha256(selection_path),
            "selection_sha256": selection["selection_sha256"],
        },
        "cache_report": {
            "path": str(report_path),
            "size_bytes": int(report_path.stat().st_size),
            "sha256": _file_sha256(report_path),
        },
        "image_source_mode": "hdf5-direct",
        "camera_order": list(EXPECTED_CAMERAS),
        "dtype": "float16",
        "encoder_provenance": _model_provenance(model_cache_root),
        "encoder_runtime": {
            "python": platform.python_version(),
            "numpy": _package_version("numpy"),
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "pillow": _package_version("pillow"),
            "opencv": _package_version("opencv-python", "opencv-python-headless"),
            "loader": (
                "DinoV2DenseConditioner: AutoImageProcessor.from_pretrained "
                "without use_fast override, AutoModel.from_pretrained, local_files_only"
            ),
        },
        "episode_count": len(entries),
        "frame_count": total_frames,
        "token_npy_bytes": total_token_bytes,
        "episode_metadata_bytes": total_metadata_bytes,
        "metadata_inventory_sha256": _canonical_digest(metadata_rows),
        "token_inventory_sha256": _canonical_digest(token_rows),
        "entries": entries,
    }
    payload["inventory_sha256"] = _canonical_digest(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and SHA-256 inventory every bounded RDT DINO cache file"
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--model-cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    destination = args.output.expanduser().resolve()
    if destination.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite DINO cache inventory: {destination}")
    payload = inventory_rdt_dino_cache(
        cache_dir=args.cache_dir,
        selection_manifest=args.selection_manifest,
        model_cache_root=args.model_cache_root,
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
                "episode_count": payload["episode_count"],
                "frame_count": payload["frame_count"],
                "token_npy_bytes": payload["token_npy_bytes"],
                "inventory_sha256": payload["inventory_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


__all__ = ["CACHE_INVENTORY_SCHEMA", "inventory_rdt_dino_cache"]
