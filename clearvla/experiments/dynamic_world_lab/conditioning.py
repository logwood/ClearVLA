from __future__ import annotations

"""Shared dense-DINO conditioner construction for dynamic-world stages."""

from pathlib import Path
from typing import Any, Sequence

import torch

from clearvla.experiments.classic_policy_lab.rdt2_conditioning import (
    CachedDinoV2DenseConditioner,
    DebugDenseConditioner,
    DinoV2DenseConditioner,
    RDT2Conditioner,
)
from clearvla.experiments.classic_policy_lab.rdt2_dinov2_cache import DinoV2TokenStore


def build_dense_conditioner(
    *,
    mode: str,
    episodes,
    camera_names: Sequence[str],
    preprocessing: Any,
    dinov2_model: str,
    dinov2_local_files_only: bool,
    dinov2_token_cache_dir: Path | None,
    debug_token_dim: int,
    debug_patches_per_camera: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[RDT2Conditioner, int, int | None]:
    if mode == "dinov2-cache":
        if dinov2_token_cache_dir is None:
            raise ValueError("--dinov2-token-cache-dir is required for dinov2-cache")
        store = DinoV2TokenStore(
            dinov2_token_cache_dir,
            episodes=episodes,
            camera_names=tuple(camera_names),
            preprocessing=preprocessing,
            dinov2_model=dinov2_model,
        )
        return CachedDinoV2DenseConditioner(store), store.token_dim, store.tokens_per_camera
    if mode == "debug-dense":
        conditioner = DebugDenseConditioner(
            token_dim=debug_token_dim,
            tokens_per_camera=debug_patches_per_camera,
        )
        return conditioner, debug_token_dim, debug_patches_per_camera
    if mode != "dinov2":
        raise ValueError(f"unsupported dense conditioner mode={mode!r}")
    conditioner = DinoV2DenseConditioner(
        dinov2_model,
        local_files_only=dinov2_local_files_only,
        drop_cls_token=True,
    )
    conditioner.to(device=device, dtype=dtype)
    return conditioner, conditioner.token_dim, None


@torch.no_grad()
def infer_dense_geometry(
    conditioner: RDT2Conditioner,
    sample: dict[str, torch.Tensor],
    *,
    camera_names: Sequence[str],
) -> tuple[int, int]:
    images = sample["history_obs_image"][-1:].to(torch.float32)
    condition = conditioner.encode(images, camera_names=camera_names)
    dense = condition.dense_tokens
    if dense is None:
        raise ValueError("online geometry inference requires dense tokens")
    if dense.shape[1] % len(camera_names):
        raise ValueError("DINO token count is not divisible by camera count")
    return int(dense.shape[-1]), int(dense.shape[1] // len(camera_names))


__all__ = ["build_dense_conditioner", "infer_dense_geometry"]
