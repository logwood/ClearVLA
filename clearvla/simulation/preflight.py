"""Fail-closed offline-cache versus online-deployment equivalence checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as functional

from clearvla.data.hdf5_episode import load_episode
from clearvla.experiments.classic_policy_lab.rdt2_dinov2_cache import (
    DinoV2TokenStore,
)
from clearvla.mainline.config import load_config
from clearvla.vision.decoded_image_store import DecodedImageStore
from clearvla.vision.preprocessing import PreprocessConfig, preprocessing_identity

from .vision import DinoV2OnlineEncoder, preprocess_rgb_history

VISUAL_PARITY_SCHEMA = "clearvla-offline-online-visual-parity-v1"


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for visual parity but is unavailable")
    return result


def _raw_history(
    episode,
    frame_indices: np.ndarray,
    camera_names: tuple[str, ...],
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    with h5py.File(episode.path, "r") as stream:
        for camera in camera_names:
            unique, inverse = np.unique(frame_indices, return_inverse=True)
            value = np.asarray(stream[episode.camera_keys[camera]][unique])[inverse]
            if value.dtype != np.uint8 or value.ndim != 4 or value.shape[-1] != 3:
                raise ValueError(
                    f"native {camera} frames must be uint8 [3,H,W,3], got "
                    f"{value.shape} {value.dtype}"
                )
            result[camera] = np.ascontiguousarray(value)
    return result


def run_visual_parity(
    *,
    config_path: Path,
    episode_path: Path,
    output: Path | None,
    center: int | None,
    device: torch.device,
    dinov2_model: str | Path | None,
    local_files_only: bool,
    max_token_abs: float,
    max_token_mean_abs: float,
) -> dict[str, object]:
    config = load_config(config_path)
    data = config.data
    cameras = tuple(data.camera_names)
    episode = load_episode(
        episode_path,
        cameras=cameras,
        action_key=data.action_key,
        action_state_key=data.action_state_key or None,
        state_key=data.state_key or None,
        camera_key_overrides=data.camera_key_map(),
    )
    selected_center = (
        int(episode.valid_center_start)
        if center is None and episode.valid_center_start is not None
        else (24 if center is None else int(center))
    )
    frame_indices = np.asarray(
        [selected_center - 8, selected_center - 4, selected_center],
        dtype=np.int64,
    )
    if data.data_profile == "maniskill_pd_ee_delta_pose_7d_v2":
        frame_indices = np.maximum(frame_indices, 0)
    if int(frame_indices.min()) < 0 or int(frame_indices.max()) >= episode.length:
        raise ValueError(
            f"visual parity frames {frame_indices.tolist()} are outside T={episode.length}"
        )
    preprocessing = PreprocessConfig(
        resize_hw=(int(data.cache_side), int(data.cache_side)),
        crop_hw=None,
    )
    native = _raw_history(episode, frame_indices, cameras)
    online_pixels = preprocess_rgb_history(
        native,
        preprocessing,
        camera_names=cameras,
    )

    decoded = DecodedImageStore(
        Path(data.decoded_cache),
        camera_names=cameras,
        preprocessing=preprocessing,
    )
    decoded.validate_episode(episode)
    cached_frames = decoded.load_window(episode, frame_indices)
    pixel_differences: dict[str, dict[str, object]] = {}
    pixel_exact = True
    for camera_index, camera in enumerate(cameras):
        offline = (
            cached_frames[camera]
            .permute(0, 2, 3, 1)
            .contiguous()
            .cpu()
            .numpy()
        )
        online = online_pixels[:, camera_index]
        difference = np.abs(
            offline.astype(np.int16) - online.astype(np.int16)
        )
        exact = bool(np.array_equal(offline, online))
        pixel_exact = pixel_exact and exact
        pixel_differences[camera] = {
            "native_shape": list(native[camera].shape),
            "offline_shape": list(offline.shape),
            "online_shape": list(online.shape),
            "exact": exact,
            "max_abs_uint8": int(difference.max()),
            "mean_abs_uint8": float(difference.mean()),
        }

    store = DinoV2TokenStore(
        Path(data.dino_cache),
        episodes=[episode],
        camera_names=cameras,
        preprocessing=preprocessing,
        dinov2_model=data.dinov2_model,
    )
    cached_tokens = store.load_batch(
        [[0, int(index)] for index in frame_indices]
    ).float()
    compute_dtype = {
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[config.runtime.compute_dtype]
    encoder = DinoV2OnlineEncoder(
        str(data.dinov2_model if dinov2_model is None else dinov2_model),
        device=device,
        compute_dtype=compute_dtype,
        expected_patches=config.dimensions.patches_per_camera,
        expected_width=config.dimensions.visual_token_dim,
        reference_batch_size=config.data.dinov2_reference_batch_size,
        local_files_only=local_files_only,
    )
    online_tokens = encoder.encode_preprocessed(online_pixels).float().cpu()
    if tuple(cached_tokens.shape) != tuple(online_tokens.shape):
        raise ValueError(
            f"offline/online DINO shapes differ: "
            f"{tuple(cached_tokens.shape)} != {tuple(online_tokens.shape)}"
        )
    difference = (cached_tokens - online_tokens).abs()
    flat_cached = cached_tokens.reshape(-1, cached_tokens.shape[-1])
    flat_online = online_tokens.reshape(-1, online_tokens.shape[-1])
    cosine = functional.cosine_similarity(flat_cached, flat_online, dim=-1)
    token_metrics = {
        "shape": list(cached_tokens.shape),
        "max_abs": float(difference.max().item()),
        "mean_abs": float(difference.mean().item()),
        "rmse": float(difference.square().mean().sqrt().item()),
        "cosine_min": float(cosine.min().item()),
        "cosine_mean": float(cosine.mean().item()),
        "max_abs_limit": float(max_token_abs),
        "mean_abs_limit": float(max_token_mean_abs),
    }
    token_pass = (
        token_metrics["max_abs"] <= float(max_token_abs)
        and token_metrics["mean_abs"] <= float(max_token_mean_abs)
    )
    passed = bool(pixel_exact and token_pass)
    result: dict[str, object] = {
        "schema": VISUAL_PARITY_SCHEMA,
        "passed": passed,
        "config": str(config_path.resolve()),
        "episode": str(episode_path.resolve()),
        "episode_id": episode.episode_id,
        "frame_indices": frame_indices.tolist(),
        "camera_order": list(cameras),
        "preprocessing": preprocessing_identity(preprocessing),
        "pixels": pixel_differences,
        "dinov2": {
            "cache_model": data.dinov2_model,
            "runtime": encoder.identity(),
            "metrics": token_metrics,
        },
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if not passed:
        raise RuntimeError(
            "offline/online visual parity failed; "
            f"pixel_exact={pixel_exact} token_metrics={token_metrics}"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None,
                        help="optional retained report; default prints diagnostics only")
    parser.add_argument("--center", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dinov2-model", type=Path, default=None)
    parser.add_argument("--dinov2-local-files-only", action="store_true")
    parser.add_argument("--max-token-abs", type=float, default=0.02)
    parser.add_argument("--max-token-mean-abs", type=float, default=0.002)
    args = parser.parse_args()
    result = run_visual_parity(
        config_path=args.config,
        episode_path=args.episode,
        output=args.output,
        center=args.center,
        device=_device(args.device),
        dinov2_model=args.dinov2_model,
        local_files_only=args.dinov2_local_files_only,
        max_token_abs=args.max_token_abs,
        max_token_mean_abs=args.max_token_mean_abs,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["VISUAL_PARITY_SCHEMA", "run_visual_parity"]
