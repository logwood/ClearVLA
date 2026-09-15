from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch
import torch.nn as nn
import torch.nn.functional as F


@runtime_checkable
class PatchTeacherLike(Protocol):
    """Frozen patch-token provider used to build frame-local latent caches."""

    teacher_name: str
    token_dim: int
    patch_grid: tuple[int, int]

    @torch.inference_mode()
    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``tokens[B,P,C]`` and ``pooled[B,C]`` for RGB images."""
        ...


@dataclass(frozen=True)
class PatchTeacherConfig:
    """Explicit configuration for a frozen patch-token teacher.

    ``tiny_patch`` is deterministic and dependency-free.  It exists only for
    smoke tests and infrastructure validation.  Formal experiments should use
    ``dinov2_vits14`` or another pretrained patch teacher added through this
    protocol.
    """

    backend: str = "dinov2_vits14"
    image_hw: tuple[int, int] = (224, 224)
    patch_size: int | None = None
    token_dim: int | None = None
    tiny_seed: int = 17
    torch_hub_source: str = "github"  # github | local
    torch_hub_repo: str = "facebookresearch/dinov2"
    local_repo_dir: str | None = None
    model_name: str | None = None

    def validate(self) -> None:
        if self.backend not in {"tiny_patch", "dinov2_vits14"}:
            raise ValueError(f"Unsupported patch teacher backend={self.backend!r}")
        h, w = int(self.image_hw[0]), int(self.image_hw[1])
        if h <= 0 or w <= 0:
            raise ValueError(f"image_hw must be positive, got {self.image_hw}")
        if self.torch_hub_source not in {"github", "local"}:
            raise ValueError("torch_hub_source must be 'github' or 'local'")
        if self.backend == "tiny_patch":
            patch = 8 if self.patch_size is None else int(self.patch_size)
            dim = 32 if self.token_dim is None else int(self.token_dim)
            if patch <= 0 or dim <= 0:
                raise ValueError("tiny patch_size and token_dim must be positive")
            if h % patch or w % patch:
                raise ValueError(
                    f"tiny image_hw={self.image_hw} must be divisible by patch_size={patch}"
                )
        elif self.backend == "dinov2_vits14":
            if h % 14 or w % 14:
                raise ValueError(
                    f"DINOv2 image_hw={self.image_hw} must be divisible by patch size 14"
                )
            if self.patch_size not in {None, 14}:
                raise ValueError("dinov2_vits14 patch_size is fixed at 14")
            if self.token_dim not in {None, 384}:
                raise ValueError("dinov2_vits14 token_dim is fixed at 384")
            if self.torch_hub_source == "local" and not self.local_repo_dir:
                raise ValueError("local_repo_dir is required when torch_hub_source='local'")

    def resolved_patch_size(self) -> int:
        self.validate()
        if self.backend == "tiny_patch":
            return 8 if self.patch_size is None else int(self.patch_size)
        return 14

    def resolved_token_dim(self) -> int:
        self.validate()
        if self.backend == "tiny_patch":
            return 32 if self.token_dim is None else int(self.token_dim)
        return 384

    def resolved_model_name(self) -> str:
        self.validate()
        if self.model_name:
            return str(self.model_name)
        return "dinov2_vits14" if self.backend == "dinov2_vits14" else "tiny_patch"

    def patch_grid(self) -> tuple[int, int]:
        patch = self.resolved_patch_size()
        return int(self.image_hw[0]) // patch, int(self.image_hw[1]) // patch

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["image_hw"] = list(self.image_hw)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PatchTeacherConfig":
        image_hw = data.get("image_hw", (224, 224))
        out = cls(
            backend=str(data.get("backend", "dinov2_vits14")),
            image_hw=(int(image_hw[0]), int(image_hw[1])),
            patch_size=None if data.get("patch_size") is None else int(data["patch_size"]),
            token_dim=None if data.get("token_dim") is None else int(data["token_dim"]),
            tiny_seed=int(data.get("tiny_seed", 17)),
            torch_hub_source=str(data.get("torch_hub_source", "github")),
            torch_hub_repo=str(data.get("torch_hub_repo", "facebookresearch/dinov2")),
            local_repo_dir=None
            if data.get("local_repo_dir") is None
            else str(data["local_repo_dir"]),
            model_name=None if data.get("model_name") is None else str(data["model_name"]),
        )
        out.validate()
        return out


class _ImagePreprocessor(nn.Module):
    def __init__(self, image_hw: tuple[int, int], *, imagenet_normalize: bool) -> None:
        super().__init__()
        self.image_hw = (int(image_hw[0]), int(image_hw[1]))
        self.imagenet_normalize = bool(imagenet_normalize)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"images must be [B,3,H,W], got {tuple(images.shape)}")
        value = images.to(dtype=torch.float32)
        if images.dtype == torch.uint8 or float(value.detach().max()) > 1.5:
            value = value / 255.0
        if tuple(value.shape[-2:]) != self.image_hw:
            value = F.interpolate(value, size=self.image_hw, mode="bilinear", align_corners=False)
        if self.imagenet_normalize:
            value = (value - self.mean) / self.std
        if not torch.isfinite(value).all():
            raise ValueError("preprocessed images contain non-finite values")
        return value


class TinyPatchTeacher(nn.Module):
    """Deterministic patch projector for offline tests only.

    It deliberately has no trainable parameters.  The stable random projection
    makes cache round-trips meaningful without pretending to be a useful visual
    backbone.
    """

    teacher_name = "tiny_patch"

    def __init__(self, config: PatchTeacherConfig) -> None:
        super().__init__()
        config.validate()
        if config.backend != "tiny_patch":
            raise ValueError("TinyPatchTeacher requires backend='tiny_patch'")
        self.config = config
        self.patch_size = config.resolved_patch_size()
        self.token_dim = config.resolved_token_dim()
        self.patch_grid = config.patch_grid()
        # Tiny smoke backend intentionally uses patch means rather than a heavy
        # unfold projection. Formal experiments use the DINOv2 teacher.
        patch_elements = 3
        generator = torch.Generator().manual_seed(int(config.tiny_seed))
        projection = torch.randn(patch_elements, self.token_dim, generator=generator) / (
            patch_elements**0.5
        )
        self.register_buffer("projection", projection, persistent=True)
        self.preprocess = _ImagePreprocessor(config.image_hw, imagenet_normalize=False)

    @torch.inference_mode()
    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value = self.preprocess(images)
        patches = F.avg_pool2d(value, kernel_size=self.patch_size, stride=self.patch_size)
        patches = patches.permute(0, 2, 3, 1).reshape(value.shape[0], -1, 3)
        tokens = patches @ self.projection
        pooled = tokens.mean(dim=1)
        if tokens.shape[1] != self.patch_grid[0] * self.patch_grid[1]:
            raise RuntimeError(
                f"unexpected tiny patch count={tokens.shape[1]} grid={self.patch_grid}"
            )
        return tokens, pooled


class DinoV2ViTS14Teacher(nn.Module):
    """Frozen DINOv2 ViT-S/14 patch-token wrapper.

    The official DINOv2 hub model exposes normalized patch features under
    ``x_norm_patchtokens``.  This wrapper keeps that representation frame-local
    and never silently fine-tunes the teacher.
    """

    teacher_name = "dinov2_vits14"

    def __init__(self, config: PatchTeacherConfig) -> None:
        super().__init__()
        config.validate()
        if config.backend != "dinov2_vits14":
            raise ValueError("DinoV2ViTS14Teacher requires backend='dinov2_vits14'")
        self.config = config
        self.patch_size = 14
        self.token_dim = 384
        self.patch_grid = config.patch_grid()
        repo_or_dir = config.torch_hub_repo
        source = config.torch_hub_source
        if source == "local":
            assert config.local_repo_dir is not None
            repo_or_dir = str(Path(config.local_repo_dir).expanduser().resolve())
        try:
            model = torch.hub.load(repo_or_dir, config.resolved_model_name(), source=source)
        except Exception as exc:
            raise RuntimeError(
                "Unable to load the DINOv2 teacher. For an offline environment, clone the official "
                "facebookresearch/dinov2 repository and pass --teacher-source local "
                "--teacher-local-repo /path/to/dinov2."
            ) from exc
        self.model = model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.preprocess = _ImagePreprocessor(config.image_hw, imagenet_normalize=True)

    @torch.inference_mode()
    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value = self.preprocess(images)
        features = self.model.forward_features(value)
        if not isinstance(features, dict) or "x_norm_patchtokens" not in features:
            raise RuntimeError("DINOv2 forward_features() did not return x_norm_patchtokens")
        tokens = features["x_norm_patchtokens"]
        if tokens.ndim != 3 or tokens.shape[-1] != self.token_dim:
            raise RuntimeError(f"unexpected DINOv2 patch token shape={tuple(tokens.shape)}")
        expected = self.patch_grid[0] * self.patch_grid[1]
        if tokens.shape[1] != expected:
            raise RuntimeError(
                f"unexpected DINOv2 patch count={tokens.shape[1]}, expected={expected}"
            )
        pooled = tokens.mean(dim=1)
        return tokens, pooled


def build_patch_teacher(
    config: PatchTeacherConfig, *, device: torch.device | str = "cpu"
) -> PatchTeacherLike:
    config.validate()
    if config.backend == "tiny_patch":
        teacher: nn.Module = TinyPatchTeacher(config)
    elif config.backend == "dinov2_vits14":
        teacher = DinoV2ViTS14Teacher(config)
    else:  # guarded by validate
        raise ValueError(config.backend)
    teacher = teacher.to(device).eval()
    return teacher  # type: ignore[return-value]
