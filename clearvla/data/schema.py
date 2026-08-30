from __future__ import annotations

from typing import Any

import h5py

ACTION_ALIASES = ("action", "/action", "actions", "/actions")
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


def list_hdf5_datasets(path: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with h5py.File(path, "r") as f:

        def visit(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                out[name] = {
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                    "chunks": list(obj.chunks) if obj.chunks is not None else None,
                    "compression": obj.compression,
                }

        f.visititems(visit)
    return out


def resolve_key(
    datasets: dict[str, dict[str, Any]],
    requested: str | None,
    aliases: tuple[str, ...],
    *,
    required: bool = True,
) -> str | None:
    candidates: list[str] = []
    if requested:
        candidates.extend(
            (requested, requested.lstrip("/"), requested.replace(".", "/").lstrip("/"))
        )
    candidates.extend(x.lstrip("/") for x in aliases)
    candidates.extend(x.replace(".", "/").lstrip("/") for x in aliases)

    seen: set[str] = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            if candidate in datasets:
                return candidate

    if required:
        available = "\n".join(sorted(datasets.keys())[:160])
        raise KeyError(
            f"Could not resolve requested dataset {requested!r}. Available datasets:\n{available}"
        )
    return None
