"""Online RGB/DINO boundary shared with the offline cache implementation."""

from __future__ import annotations

import hashlib
import json
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from clearvla.experiments.classic_policy_lab.rdt2_conditioning import (
    DinoV2DenseConditioner,
)
from clearvla.vision.preprocessing import (
    PreprocessConfig,
    apply_preprocess,
    preprocessing_identity,
)

from .contracts import CAMERA_NAMES
from .history import VISUAL_OFFSETS


def preprocess_rgb_history(
    rgb_history: Mapping[str, np.ndarray],
    preprocessing: PreprocessConfig,
    *,
    camera_names: Sequence[str] = CAMERA_NAMES,
) -> np.ndarray:
    """Resize each native camera exactly once and retain ordered uint8 pixels."""

    names = tuple(str(name) for name in camera_names)
    if tuple(rgb_history) != names:
        raise ValueError(
            f"RGB history cameras must be ordered as {names}, got {tuple(rgb_history)}"
        )
    camera_rows: list[np.ndarray] = []
    expected_shape: tuple[int, int, int] | None = None
    for camera in names:
        source = np.asarray(rgb_history[camera])
        if (
            source.ndim != 4
            or source.shape[0] != len(VISUAL_OFFSETS)
            or source.shape[-1] != 3
            or source.dtype != np.uint8
        ):
            raise ValueError(f"{camera} RGB history must be uint8 [3,H,W,3]")
        processed = np.stack(
            [apply_preprocess(frame, preprocessing) for frame in source],
            axis=0,
        )
        shape = tuple(int(value) for value in processed.shape[1:])
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ValueError(
                "checkpoint preprocessing must materialize one shared model-side shape, "
                f"got {expected_shape} and {shape}"
            )
        camera_rows.append(processed)
    # [camera,history,H,W,3] -> [history,camera,H,W,3]
    return np.ascontiguousarray(np.stack(camera_rows, axis=0).transpose(1, 0, 2, 3, 4))


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class DinoV2OnlineEncoder(nn.Module):
    """Frozen encoder that calls the exact class used by the cache builder."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: torch.device,
        compute_dtype: torch.dtype,
        expected_patches: int,
        expected_width: int,
        reference_batch_size: int = 32,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        self.conditioner = DinoV2DenseConditioner(
            model_name_or_path,
            local_files_only=bool(local_files_only),
        )
        encoder_dtype = (
            compute_dtype
            if device.type == "cuda" and compute_dtype in {torch.bfloat16, torch.float16}
            else torch.float32
        )
        self.conditioner.to(device=device, dtype=encoder_dtype).eval()
        self.expected_patches = int(expected_patches)
        self.expected_width = int(expected_width)
        self.reference_batch_size = int(reference_batch_size)
        if self.reference_batch_size <= 0:
            raise ValueError("DINO reference_batch_size must be positive")
        self.model_name_or_path = str(model_name_or_path)
        if int(self.conditioner.token_dim) != self.expected_width:
            raise ValueError(
                f"DINO width {self.conditioner.token_dim} != {self.expected_width}"
            )

    def train(self, mode: bool = True):
        super().train(False)
        self.conditioner.eval()
        return self

    def identity(self) -> dict[str, object]:
        encoder = self.conditioner.encoder
        processor = self.conditioner.processor
        encoder_config = encoder.config.to_dict()
        processor_config = processor.to_dict()
        parameter_dtype = next(encoder.parameters()).dtype
        dtype_name = {
            torch.bfloat16: "bf16",
            torch.float32: "fp32",
            torch.float16: "fp16",
        }.get(parameter_dtype, str(parameter_dtype).replace("torch.", ""))
        return {
            "model": self.model_name_or_path,
            "commit_hash": str(getattr(encoder.config, "_commit_hash", "") or ""),
            "encoder_config_sha256": _digest(encoder_config),
            "processor_config_sha256": _digest(processor_config),
            "hidden_size": int(encoder.config.hidden_size),
            "patches_per_camera": self.expected_patches,
            "storage_dtype": "float16",
            "reference_batch_size": self.reference_batch_size,
            "compute_dtype": dtype_name,
        }

    @torch.no_grad()
    def encode_preprocessed(self, images: np.ndarray) -> Tensor:
        """Encode uint8 [3,2,H,W,3] exactly like the DINO cache builder."""

        value = np.asarray(images)
        if (
            value.ndim != 5
            or value.shape[:2] != (len(VISUAL_OFFSETS), len(CAMERA_NAMES))
            or value.shape[-1] != 3
            or value.dtype != np.uint8
        ):
            raise ValueError("preprocessed online RGB must be uint8 [3,2,H,W,3]")
        original_batch = int(value.shape[0])
        # The offline cache builder uses a fixed batch size (currently 32).
        # On CUDA, bf16 attention kernels can choose a different reduction
        # path for a smaller batch, producing materially different tokens.
        # Repeat the final causal frame only as padding so the first N rows
        # follow the exact cache-builder kernel path.
        target_batch = max(
            self.reference_batch_size,
            int(
                np.ceil(original_batch / self.reference_batch_size)
                * self.reference_batch_size
            ),
        )
        if target_batch != original_batch:
            value = np.concatenate(
                [value, np.repeat(value[-1:], target_batch - original_batch, axis=0)],
                axis=0,
            )
        chw = (
            torch.from_numpy(np.ascontiguousarray(value))
            .permute(0, 1, 4, 2, 3)
            .float()
            .div_(255.0)
        )
        dense = self.conditioner.encode(chw, camera_names=CAMERA_NAMES).dense_tokens
        if dense is None:
            raise AssertionError("DINO conditioner did not return dense tokens")
        expected = (
            target_batch,
            len(CAMERA_NAMES) * self.expected_patches,
            self.expected_width,
        )
        if tuple(dense.shape) != expected:
            raise ValueError(f"online DINO patch chart {tuple(dense.shape)} != {expected}")
        if not bool(torch.isfinite(dense).all()):
            raise ValueError("online DINO produced NaN or infinity")
        return dense[:original_batch].reshape(
            original_batch,
            len(CAMERA_NAMES),
            self.expected_patches,
            self.expected_width,
        ).half()

    @torch.no_grad()
    def encode(
        self,
        rgb_history: Mapping[str, np.ndarray],
        preprocessing: PreprocessConfig,
    ) -> tuple[Tensor, np.ndarray]:
        images = preprocess_rgb_history(rgb_history, preprocessing)
        return self.encode_preprocessed(images), images

    @staticmethod
    def preprocessing_identity(preprocessing: PreprocessConfig) -> dict[str, object]:
        return preprocessing_identity(preprocessing)


__all__ = [
    "DinoV2OnlineEncoder",
    "preprocess_rgb_history",
]
