from __future__ import annotations

"""Audit temporal information in paired decoded-image and DINO token caches."""

import argparse
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import Tensor


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-cache", type=Path, required=True)
    parser.add_argument("--dino-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contact-sheet", type=Path)
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_float(value: Tensor | np.ndarray | float | int) -> float:
    if torch.is_tensor(value):
        return float(value.detach().double().cpu())
    if isinstance(value, np.ndarray):
        return float(value.astype(np.float64))
    return float(value)


def _effective_rank(values: Tensor, eps: float = 1e-12) -> dict[str, float]:
    values = values.double()
    values = values - values.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(values)
    energy = singular_values.square()
    total = energy.sum()
    if not bool(total > eps):
        return {
            "entropy_effective_rank": 0.0,
            "participation_rank": 0.0,
            "top1_energy_fraction": 0.0,
            "top4_energy_fraction": 0.0,
            "top8_energy_fraction": 0.0,
            "numerical_rank": 0,
        }
    energy = energy / total
    nonzero = energy[energy > eps]
    entropy_rank = torch.exp(-(nonzero * nonzero.log()).sum())
    participation = energy.sum().square() / energy.square().sum().clamp_min(eps)
    return {
        "entropy_effective_rank": _safe_float(entropy_rank),
        "participation_rank": _safe_float(participation),
        "top1_energy_fraction": _safe_float(energy[0]),
        "top4_energy_fraction": _safe_float(energy[:4].sum()),
        "top8_energy_fraction": _safe_float(energy[:8].sum()),
        "numerical_rank": int((energy > eps).sum()),
    }


def _linear_cka(x: Tensor, y: Tensor, eps: float = 1e-12) -> float:
    x = x.double() - x.double().mean(dim=0, keepdim=True)
    y = y.double() - y.double().mean(dim=0, keepdim=True)
    cross = x.T @ y
    numerator = cross.square().sum()
    denominator = torch.sqrt((x.T @ x).square().sum() * (y.T @ y).square().sum())
    return _safe_float(numerator / denominator.clamp_min(eps))


def _pearson(x: Tensor, y: Tensor, eps: float = 1e-12) -> float:
    x = x.double() - x.double().mean()
    y = y.double() - y.double().mean()
    return _safe_float(
        (x * y).sum() / torch.sqrt(x.square().sum() * y.square().sum()).clamp_min(eps)
    )


def _rankdata(values: Tensor) -> Tensor:
    sorted_values, order = torch.sort(values.double())
    ranks = torch.empty_like(sorted_values)
    start = 0
    while start < sorted_values.numel():
        stop = start + 1
        while stop < sorted_values.numel() and bool(sorted_values[stop] == sorted_values[start]):
            stop += 1
        ranks[order[start:stop]] = 0.5 * float(start + stop - 1)
        start = stop
    return ranks


def _spearman(x: Tensor, y: Tensor) -> float:
    return _pearson(_rankdata(x), _rankdata(y))


def _cosine_rows(x: Tensor, y: Tensor, eps: float = 1e-12) -> Tensor:
    return (x * y).sum(dim=-1) / (x.norm(dim=-1) * y.norm(dim=-1)).clamp_min(eps)


def _summary(values: Tensor) -> dict[str, float]:
    values = values.double().flatten()
    return {
        "mean": _safe_float(values.mean()),
        "std": _safe_float(values.std(unbiased=False)),
        "min": _safe_float(values.min()),
        "p05": _safe_float(torch.quantile(values, 0.05)),
        "median": _safe_float(values.median()),
        "p95": _safe_float(torch.quantile(values, 0.95)),
        "max": _safe_float(values.max()),
    }


def _lag_metrics(features: Tensor, lags: list[int]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for lag in lags:
        if lag >= features.shape[0]:
            continue
        left = features[:-lag]
        right = features[lag:]
        result[str(lag)] = {
            "cosine_mean": _safe_float(_cosine_rows(left, right).mean()),
            "delta_rms": _safe_float((right - left).square().mean().sqrt()),
            "delta_norm_mean": _safe_float((right - left).norm(dim=-1).mean()),
        }
    return result


def _path_metrics(features: Tensor) -> dict[str, float]:
    features = features.double()
    steps = (features[1:] - features[:-1]).norm(dim=-1)
    endpoint = (features[-1] - features[0]).norm()
    return {
        "path_length": _safe_float(steps.sum()),
        "endpoint_distance": _safe_float(endpoint),
        "tortuosity": _safe_float(steps.sum() / endpoint.clamp_min(1e-12)),
        "step_norm_mean": _safe_float(steps.mean()),
        "step_norm_cv": _safe_float(steps.std(unbiased=False) / steps.mean().clamp_min(1e-12)),
    }


def _nearest_temporal_alias(features: Tensor, exclusion: int = 8) -> dict[str, float]:
    normalized = torch.nn.functional.normalize(features.double(), dim=-1)
    similarity = normalized @ normalized.T
    indices = torch.arange(features.shape[0])
    forbidden = (indices[:, None] - indices[None, :]).abs() <= exclusion
    similarity[forbidden] = -torch.inf
    nearest = similarity.argmax(dim=-1)
    distance = (nearest - indices).abs().double()
    score = similarity[indices, nearest]
    return {
        "exclusion_radius": exclusion,
        "nearest_similarity_mean": _safe_float(score.mean()),
        "nearest_similarity_p95": _safe_float(torch.quantile(score, 0.95)),
        "nearest_temporal_distance_median": _safe_float(distance.median()),
        "nearest_temporal_distance_p05": _safe_float(torch.quantile(distance, 0.05)),
    }


def _image_features(images: np.ndarray) -> Tensor:
    # Spatial pooling keeps temporal appearance while avoiding a 115M-value SVD.
    tensor = torch.from_numpy(np.asarray(images, dtype=np.float32)) / 255.0
    tensor = tensor.permute(0, 3, 1, 2)
    tensor = torch.nn.functional.adaptive_avg_pool2d(tensor, (12, 12))
    return tensor.flatten(1)


def _camera_metrics(tokens: Tensor, images: np.ndarray) -> dict[str, Any]:
    tokens = tokens.float()
    pooled = tokens.mean(dim=1)
    flattened = tokens.flatten(1)
    grid = math.isqrt(int(tokens.shape[1]))
    if grid * grid != int(tokens.shape[1]):
        raise ValueError(f"DINO patch count={int(tokens.shape[1])} is not a square spatial grid")
    token_grid = tokens.reshape(tokens.shape[0], grid, grid, tokens.shape[-1]).permute(0, 3, 1, 2)
    spatially_pooled = torch.nn.functional.adaptive_avg_pool2d(token_grid, (4, 4)).flatten(1)
    image_features = _image_features(images)
    image_step = (image_features[1:] - image_features[:-1]).square().mean(dim=-1).sqrt()
    pooled_step = (pooled[1:] - pooled[:-1]).square().mean(dim=-1).sqrt()
    token_step = (flattened[1:] - flattened[:-1]).square().mean(dim=-1).sqrt()
    spatial_std = tokens.std(dim=1, unbiased=False).square().mean(dim=-1).sqrt()

    image_rank = _rankdata(image_step) / max(image_step.numel() - 1, 1)
    token_rank = _rankdata(token_step) / max(token_step.numel() - 1, 1)
    relative_mismatch = image_rank - token_rank
    mismatch_indices = torch.topk(relative_mismatch, k=min(12, relative_mismatch.numel())).indices

    exact_duplicate = torch.from_numpy(np.all(images[1:] == images[:-1], axis=(1, 2, 3)))
    token_duplicate = (flattened[1:] == flattened[:-1]).all(dim=-1)
    duplicate_indices = exact_duplicate.nonzero(as_tuple=False).flatten()

    return {
        "token_scalar": {
            "minimum": _safe_float(tokens.min()),
            "maximum": _safe_float(tokens.max()),
            "mean": _safe_float(tokens.double().mean()),
            "std": _safe_float(tokens.double().std(unbiased=False)),
            "finite_fraction": _safe_float(torch.isfinite(tokens).double().mean()),
        },
        "global_pooled_temporal_rank": _effective_rank(pooled),
        "spatially_pooled_temporal_rank": _effective_rank(spatially_pooled),
        "image_pooled_temporal_rank": _effective_rank(image_features),
        "spatial_patch_diversity_rms": _summary(spatial_std),
        "dino_global_lags": _lag_metrics(pooled, [1, 2, 4, 8, 16, 24, 48]),
        "image_lags": _lag_metrics(image_features, [1, 2, 4, 8, 16, 24, 48]),
        "dino_global_path": _path_metrics(pooled),
        "image_path": _path_metrics(image_features),
        "dino_temporal_alias": _nearest_temporal_alias(pooled),
        "frame_step": {
            "image_rms": _summary(image_step),
            "dino_global_rms": _summary(pooled_step),
            "dino_patch_aligned_rms": _summary(token_step),
            "image_vs_dino_global_pearson": _pearson(image_step, pooled_step),
            "image_vs_dino_global_spearman": _spearman(image_step, pooled_step),
            "image_vs_dino_patch_pearson": _pearson(image_step, token_step),
            "image_vs_dino_patch_spearman": _spearman(image_step, token_step),
        },
        "exact_duplicate_transitions": {
            "count": int(duplicate_indices.numel()),
            "fraction": _safe_float(exact_duplicate.double().mean()),
            "indices": [int(index) for index in duplicate_indices],
            "image_and_token_indices_match": bool(torch.equal(exact_duplicate, token_duplicate)),
        },
        "largest_relative_change_mismatches": [
            {
                "from": int(index),
                "to": int(index + 1),
                "image_change_percentile": _safe_float(image_rank[index]),
                "dino_change_percentile": _safe_float(token_rank[index]),
                "percentile_gap": _safe_float(relative_mismatch[index]),
                "image_rms": _safe_float(image_step[index]),
                "dino_patch_rms": _safe_float(token_step[index]),
            }
            for index in mismatch_indices
        ],
    }


def _contact_sheet(path: Path, camera_images: dict[str, np.ndarray], indices: list[int]) -> None:
    thumb = 224
    label = 24
    cameras = list(camera_images)
    canvas = Image.new("RGB", (thumb * len(indices), (thumb + label) * len(cameras)), "white")
    draw = ImageDraw.Draw(canvas)
    for row, camera in enumerate(cameras):
        for column, index in enumerate(indices):
            image = Image.fromarray(camera_images[camera][index]).resize(
                (thumb, thumb), Image.Resampling.BILINEAR
            )
            x = column * thumb
            y = row * (thumb + label)
            canvas.paste(image, (x, y + label))
            draw.text((x + 4, y + 4), f"{camera} t={index}", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> int:
    args = _parse_args()
    image_meta = _load_json(args.image_cache / "meta.json")
    dino_meta = _load_json(args.dino_cache / "meta.json")
    tokens_np = np.load(args.dino_cache / "tokens.float16.npy", mmap_mode="r")

    cameras = list(dino_meta["cameras"])
    image_arrays = {
        camera: np.load(args.image_cache / f"{camera}.uint8.npy", mmap_mode="r")
        for camera in cameras
    }
    frame_count = int(tokens_np.shape[0])
    if args.max_frames > 0:
        frame_count = min(frame_count, args.max_frames)
    tokens = torch.from_numpy(np.asarray(tokens_np[:frame_count], dtype=np.float32))
    image_arrays = {camera: images[:frame_count] for camera, images in image_arrays.items()}

    fingerprints_match = image_meta.get("source_fingerprint") == dino_meta.get("source_fingerprint")
    episode_stems_match = image_meta.get("episode_stem") == dino_meta.get("episode_stem")
    camera_metadata_match = image_meta.get("cameras") == dino_meta.get("cameras") == cameras
    preprocessing_match = image_meta.get("preprocessing") == dino_meta.get("decoded_preprocessing")
    metadata_frame_counts_match = (
        int(image_meta.get("num_frames", -1))
        == int(dino_meta.get("num_frames", -2))
        == int(tokens_np.shape[0])
    )
    camera_shapes_match = all(len(images) == frame_count for images in image_arrays.values())
    shape_valid = (
        tokens.ndim == 4
        and tokens.shape[1] == len(cameras)
        and int(tokens.shape[2]) == int(dino_meta.get("tokens_per_camera", -1))
        and int(tokens.shape[3]) == int(dino_meta.get("token_dim", -1))
    )
    if not (
        fingerprints_match
        and episode_stems_match
        and camera_metadata_match
        and preprocessing_match
        and metadata_frame_counts_match
        and camera_shapes_match
        and shape_valid
    ):
        raise ValueError("image and DINO caches are not strictly aligned")

    per_camera = {
        camera: _camera_metrics(tokens[:, index], image_arrays[camera])
        for index, camera in enumerate(cameras)
    }
    pooled = tokens.mean(dim=2)
    camera_delta = pooled[1:] - pooled[:-1]
    cross_camera = {
        "global_feature_linear_cka": _linear_cka(pooled[:, 0], pooled[:, 1]),
        "temporal_delta_linear_cka": _linear_cka(camera_delta[:, 0], camera_delta[:, 1]),
        "framewise_delta_cosine": _summary(_cosine_rows(camera_delta[:, 0], camera_delta[:, 1])),
        "time_reversed_delta_cka_control": _linear_cka(
            camera_delta[:, 0], camera_delta.flip(0)[:, 1]
        ),
    }

    result = {
        "schema": "clearvla-dino-trajectory-cache-probe-v1",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "alignment": {
            "frames": frame_count,
            "cameras": cameras,
            "tokens_shape": list(tokens.shape),
            "fingerprints_match": fingerprints_match,
            "episode_stems_match": episode_stems_match,
            "camera_metadata_match": camera_metadata_match,
            "preprocessing_match": preprocessing_match,
            "metadata_frame_counts_match": metadata_frame_counts_match,
            "camera_frame_counts_match": camera_shapes_match,
            "source_fingerprint": image_meta.get("source_fingerprint"),
        },
        "per_camera": per_camera,
        "cross_camera": cross_camera,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")

    if args.contact_sheet is not None:
        indices = sorted(
            {
                0,
                frame_count // 6,
                frame_count // 3,
                frame_count // 2,
                2 * frame_count // 3,
                5 * frame_count // 6,
                frame_count - 1,
            }
        )
        _contact_sheet(args.contact_sheet, image_arrays, indices)

    print(json.dumps(result["alignment"], indent=2))
    for camera, metrics in per_camera.items():
        rank = metrics["global_pooled_temporal_rank"]
        step = metrics["frame_step"]
        print(
            f"[{camera}] temporal_rank={rank['entropy_effective_rank']:.2f} "
            f"top4={rank['top4_energy_fraction']:.3f} "
            f"image/dino_step_r={step['image_vs_dino_patch_pearson']:.3f}"
        )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
