from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import h5py

ACTION_ALIASES = ("action", "/action", "actions", "/actions")
ACTION_STATE_ALIASES = (
    "action_state",
    "/action_state",
    "observations/action_state",
    "observation/action_state",
    "previous_action",
    "/previous_action",
)
STATE_ALIASES = (
    "qpos",
    "/qpos",
    "observations/qpos",
    "observation/qpos",
    "observation.state",
    "observations/state",
    "state",
    "/state",
)
TOP_ALIASES = (
    "observations/images/cam_high",
    "observations/images/top_camera",
    "observations/images/top",
    "observation/images/cam_high",
    "observation/images/top_camera",
    "observation.images.top_camera",
    "cam_high",
    "top_camera",
)
WRIST_ALIASES = (
    "observations/images/cam_right_wrist",
    "observations/images/cam_left_wrist",
    "observations/images/wrist_camera",
    "observations/images/wrist",
    "observation/images/cam_right_wrist",
    "observation/images/wrist_camera",
    "observation.images.wrist_camera",
    "cam_right_wrist",
    "wrist_camera",
)
LEFT_WRIST_ALIASES = (
    "observations/images/cam_left_wrist",
    "observation/images/cam_left_wrist",
    "cam_left_wrist",
)
RIGHT_WRIST_ALIASES = (
    "observations/images/cam_right_wrist",
    "observation/images/cam_right_wrist",
    "cam_right_wrist",
)

CAMERA_ALIASES: dict[str, tuple[str, ...]] = {
    "top": TOP_ALIASES,
    "wrist": WRIST_ALIASES,
    "high": TOP_ALIASES,
    "left_wrist": LEFT_WRIST_ALIASES,
    "right_wrist": RIGHT_WRIST_ALIASES,
}


def parse_camera_key_overrides(values: Sequence[str] | None) -> dict[str, str]:
    """Parse repeated ``NAME=HDF5/PATH`` CLI assignments without guessing."""

    result: dict[str, str] = {}
    for raw in values or ():
        name, separator, key = str(raw).partition("=")
        name = name.strip()
        key = key.strip().lstrip("/")
        if not separator or not name or not key:
            raise ValueError(f"camera key assignment must be NAME=HDF5/PATH, got {raw!r}")
        if any(character in name for character in "/\\"):
            raise ValueError(f"camera name is not cache safe: {name!r}")
        if name in result:
            raise ValueError(f"duplicate camera key assignment for {name!r}")
        result[name] = key
    return result


def list_hdf5_datasets_from_handle(handle: h5py.File) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}

    def visit(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            out[name] = {
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "chunks": list(obj.chunks) if obj.chunks is not None else None,
                "compression": obj.compression,
            }

    handle.visititems(visit)
    return out


def list_hdf5_datasets(path: str) -> dict[str, dict[str, Any]]:
    with h5py.File(path, "r") as handle:
        return list_hdf5_datasets_from_handle(handle)


def _key_candidates(requested: str | None, aliases: tuple[str, ...]) -> tuple[str, ...]:
    candidates: list[str] = []
    if requested:
        candidates.extend(
            (requested, requested.lstrip("/"), requested.replace(".", "/").lstrip("/"))
        )
    candidates.extend(x.lstrip("/") for x in aliases)
    candidates.extend(x.replace(".", "/").lstrip("/") for x in aliases)

    result: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            result.append(candidate)
    return tuple(result)


def resolve_key(
    datasets: dict[str, dict[str, Any]],
    requested: str | None,
    aliases: tuple[str, ...],
    *,
    required: bool = True,
) -> str | None:
    candidates = _key_candidates(requested, aliases)
    for candidate in candidates:
        if candidate in datasets:
            return candidate

    if required:
        available = "\n".join(sorted(datasets.keys())[:160])
        raise KeyError(
            f"Could not resolve requested dataset {requested!r}. Available datasets:\n{available}"
        )
    return None


def resolve_handle_key(
    handle: h5py.Group,
    requested: str | None,
    aliases: tuple[str, ...],
    *,
    required: bool = True,
) -> str | None:
    """Resolve a known dataset key without traversing unrelated HDF5 objects.

    The normal episode schema has a small, fixed set of candidate paths.  Probe
    those paths directly on the already-open handle; only malformed files that
    miss a required key fall back to a full inventory for diagnostics.
    """

    for candidate in _key_candidates(requested, aliases):
        canonical = candidate.lstrip("/")
        if isinstance(handle.get(canonical), h5py.Dataset):
            return canonical
    if required:
        datasets = list_hdf5_datasets_from_handle(handle)
        return resolve_key(datasets, requested, aliases, required=True)
    return None
