"""Explicit full-FOV, frozen DINOv3 boundary; no dataset token cache.

This source unit deliberately does not change legacy data/model consumers.
Patch tokens remain native patch tokens. Consumers MUST use the coordinate
transform below rather than treating patch endpoints as full-image endpoints.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class DinoV3SpatialSpec:
    input_side: int = 256
    patch_size: int = 16
    width: int = 768
    register_tokens: int = 4

    def validate(self) -> None:
        # This unit admits ViT-B/16 only, not a silent width-changing fallback.
        if (self.input_side, self.patch_size, self.width, self.register_tokens) != (256, 16, 768, 4):
            raise ValueError("this boundary requires DINOv3 ViT-B/16, 256px, 4 registers, width768")

    @property
    def grid_side(self) -> int:
        self.validate()
        return self.input_side // self.patch_size

    def identity(self) -> dict[str, object]:
        self.validate()
        return {
            "schema": "dinov3_full_fov_patch_centers_v1", **asdict(self),
            "resize": "torch_bilinear_align_corners_false_antialias_true_fp32",
            "center_crop": False, "input": "RGB_float32_0_1",
            "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
            "output": "row_major_native_patch_tokens_without_cls_or_registers",
            "coordinate_contract": "full_rgb_pixel_centers_align_corners_true",
            "patch_edge_sampling": "border_replication_outside_patch_centers_only",
        }


def _size(hw: tuple[int, int]) -> tuple[int, int]:
    if len(hw) != 2 or any(type(v) is not int or v <= 1 for v in hw):
        raise ValueError("image/grid dimensions must be integer H,W > 1")
    return hw


def full_image_to_patch_grid(
    coordinates: Tensor, *, image_hw: tuple[int, int], patch_grid_hw: tuple[int, int] = (16, 16),
) -> Tensor:
    """Full RGB pixel-center XY -> native patch-center grid_sample(align_corners=True).

    Full-frame resize uses align_corners=False. A patch center i corresponds
    to x_rgb=(i+0.5)*W/N-0.5. This includes the half-pixel resize convention,
    not merely an N/(N-1) scale. Coordinates near full-frame edges can lie
    beyond the first/last patch center; the sampler uses explicit border mode.
    """
    h, w = _size(image_hw)
    gh, gw = _size(patch_grid_hw)
    if coordinates.shape[-1:] != (2,) or not coordinates.is_floating_point():
        raise ValueError("coordinates must be floating tensors ending in XY")
    if not bool(torch.isfinite(coordinates).all()):
        raise ValueError("coordinates must be finite")
    if bool((coordinates.abs() > 1.0 + 1e-6).any()):
        raise ValueError("full-image coordinates must lie within [-1,1]")
    scale = coordinates.new_tensor((gw * (w - 1) / ((gw - 1) * w), gh * (h - 1) / ((gh - 1) * h)))
    return coordinates * scale


def patch_centers_in_full_image(
    *, image_hw: tuple[int, int], patch_grid_hw: tuple[int, int] = (16, 16), device: torch.device | str = "cpu",
) -> Tensor:
    """Return [Gh,Gw,2] in original RGB pixel-center normalized coordinates."""
    h, w = _size(image_hw)
    gh, gw = _size(patch_grid_hw)
    x = 2 * ((torch.arange(gw, device=device, dtype=torch.float32) + 0.5) * w / gw - 0.5) / (w - 1) - 1
    y = 2 * ((torch.arange(gh, device=device, dtype=torch.float32) + 0.5) * h / gh - 0.5) / (h - 1) - 1
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), -1)


def sample_native_patch_chart(patches: Tensor, coordinates: Tensor, *, image_hw: tuple[int, int]) -> Tensor:
    """Read [B,Gh,Gw,D] at full RGB coordinates [B,...,2]; camera is never pooled.

    Leading camera/time axes can be folded into B by the caller and restored.
    This is a new explicit read, NOT a global replacement of legacy sampling.
    """
    if patches.ndim != 4 or coordinates.ndim < 3 or patches.shape[0] != coordinates.shape[0]:
        raise ValueError("patch chart [B,Gh,Gw,D] and coordinates [B,...,2] required")
    if patches.device != coordinates.device or not patches.is_floating_point():
        raise ValueError("patches and floating coordinates must share a device")
    grid = full_image_to_patch_grid(coordinates.float(), image_hw=image_hw, patch_grid_hw=tuple(patches.shape[1:3]))
    with torch.autocast(device_type=patches.device.type, enabled=False):
        sampled = F.grid_sample(patches.float().permute(0, 3, 1, 2), grid.reshape(patches.shape[0], -1, 1, 2),
                                mode="bilinear", padding_mode="border", align_corners=True)
    return sampled[:, :, :, 0].transpose(1, 2).reshape(*coordinates.shape[:-1], patches.shape[-1]).to(patches.dtype)


def preprocess_full_fov(rgb: Tensor, spec: DinoV3SpatialSpec = DinoV3SpatialSpec()) -> Tensor:
    """[N,3,H,W] float RGB in [0,1] -> normalized 256px; no hidden HF transform."""
    spec.validate()
    if rgb.ndim != 4 or rgb.shape[1] != 3 or min(rgb.shape[-2:]) < 2 or rgb.shape[0] == 0:
        raise ValueError("RGB must be nonempty [N,3,H,W], H/W >=2")
    if not rgb.is_floating_point() or not bool(torch.isfinite(rgb).all()):
        raise ValueError("RGB must contain finite floating values")
    if bool(((rgb < 0) | (rgb > 1)).any()):
        raise ValueError("RGB must be in [0,1]; uint8/normalized inputs are not silently clamped")
    with torch.autocast(device_type=rgb.device.type, enabled=False):
        value = F.interpolate(rgb.float(), size=(spec.input_side, spec.input_side), mode="bilinear", align_corners=False, antialias=True)
        mean = value.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
        std = value.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
        return (value - mean) / std


def extract_native_patches(hidden: Tensor, spec: DinoV3SpatialSpec = DinoV3SpatialSpec()) -> Tensor:
    spec.validate()
    prefix = 1 + spec.register_tokens
    n = spec.grid_side ** 2
    if hidden.ndim != 3 or tuple(hidden.shape[1:]) != (prefix + n, spec.width):
        raise ValueError(f"DINOv3 output must be [N,{prefix+n},{spec.width}], got {tuple(hidden.shape)}")
    result = hidden[:, prefix:]
    if not result.is_floating_point() or not bool(torch.isfinite(result).all()):
        raise ValueError("DINOv3 patch tokens must be finite floating values")
    return result.contiguous()


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _local_weight_identity(root: Path) -> dict[str, object]:
    if not (root / "config.json").is_file():
        raise FileNotFoundError("local DINOv3 directory must include config.json")
    index = root / "model.safetensors.index.json"
    if index.is_file():
        data = json.loads(index.read_text())
        files = sorted(set(data["weight_map"].values()))
        files.append(index.name)
    else:
        files = ["model.safetensors"]
    files.append("config.json")
    manifest: dict[str, str] = {}
    for name in files:
        path = root / name
        # A locally supplied index may not redirect the loader outside the model.
        if Path(name).name != name or not path.is_file():
            raise ValueError(f"missing or unsafe local safetensors member: {name}")
        # Standard HF snapshots use symlinks to content-addressed weight blobs.
        # Hash actual bytes; reject index traversal, not legitimate symlinks.
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                h.update(block)
        manifest[name] = h.hexdigest()
    return {"kind": "local_safetensors_sha256", "files": manifest, "digest": _digest(manifest)}


class FrozenDinoV3Encoder(nn.Module):
    """One resident frozen encoder, bounded online image batches, no disk token store.

    Use from_pretrained in production. Explicit model injection is only useful
    for unit contracts and must NOT be reported as pretrained inference.
    """
    def __init__(self, model: nn.Module, *, spec: DinoV3SpatialSpec = DinoV3SpatialSpec(),
                 microbatch: int = 2, identity: Mapping[str, object] | None = None) -> None:
        super().__init__()
        spec.validate()
        if type(microbatch) is not int or microbatch <= 0:
            raise ValueError("microbatch must be a positive integer")
        config = getattr(model, "config", None)
        if config is None or getattr(config, "model_type", None) != "dinov3_vit":
            raise ValueError("expected a DINOv3 ViT model, not a DINOv2 or ConvNeXt fallback")
        for name, expected in (("patch_size", spec.patch_size), ("hidden_size", spec.width), ("num_register_tokens", spec.register_tokens), ("num_hidden_layers", 12),
                               ("num_attention_heads", 12), ("intermediate_size", 3072)):
            if getattr(config, name, None) != expected:
                raise ValueError(f"DINOv3 config {name} must be {expected}")
        self.encoder = model.requires_grad_(False).eval()
        self.spec = spec
        self.microbatch = microbatch
        self.weight_identity = dict(identity or {"kind": "injected_unverified_model_not_pretrained_acceptance"})
        self.train(False)

    @classmethod
    def from_pretrained(cls, model: str | Path, *, revision: str | None = None,
                        local_files_only: bool = True, device: str | torch.device = "cpu",
                        dtype: torch.dtype = torch.float32, microbatch: int = 2) -> "FrozenDinoV3Encoder":
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError("DINOv3 online requires transformers with dinov3_vit support; no fallback is used") from exc
        source = str(model)
        local = Path(source).is_dir()
        if not local and (revision is None or re.fullmatch(r"[0-9a-f]{40}", revision) is None):
            raise ValueError("remote DINOv3 requires an immutable 40-character revision")
        identity = _local_weight_identity(Path(source)) if local else {
            "kind": "huggingface_immutable_revision", "model": source, "revision": revision,
        }
        loaded, info = AutoModel.from_pretrained(source, revision=None if local else revision,
            local_files_only=local_files_only, trust_remote_code=False, use_safetensors=True,
            output_loading_info=True, torch_dtype=torch.float32)
        failures = {key: info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") if info.get(key)}
        if failures:
            raise ValueError(f"DINOv3 checkpoint must load strictly: {failures}")
        if dtype not in (torch.float32, torch.bfloat16, torch.float16):
            raise ValueError("unsupported encoder compute dtype")
        if torch.device(device).type == "cpu" and dtype == torch.float16:
            raise ValueError("CPU fp16 encoder is not supported; use fp32 or bf16")
        return cls(loaded, microbatch=microbatch, identity=identity).to(device=device, dtype=dtype)

    def train(self, mode: bool = True) -> "FrozenDinoV3Encoder":
        super().train(False)
        self.encoder.eval()
        return self

    def identity(self) -> dict[str, object]:
        param = next(self.encoder.parameters())
        return {"schema": "clearvla_online_dinov3_boundary_v1", "weights": self.weight_identity,
                "spatial": self.spec.identity(), "compute_dtype": str(param.dtype),
                "microbatch": self.microbatch, "padding": "repeat_last_image_to_fixed_microbatch",
                "token_disk_cache": False, "frozen": True}

    @torch.no_grad()
    def forward(self, rgb: Tensor) -> Tensor:
        """Accept [...,3,H,W], keep ALL leading axes (batch, time, camera).

        Output [...,256,768]. Fixed microbatch padding bounds workspace and
        avoids batch-shape changes between online and offline call sites.
        Output tensors are normal detached tensors usable by downstream autograd.
        """
        if rgb.ndim < 4 or rgb.shape[-3] != 3 or any(n == 0 for n in rgb.shape):
            raise ValueError("RGB requires nonempty leading axes and trailing [3,H,W]")
        flat = rgb.reshape(-1, *rgb.shape[-3:])
        parameter = next(self.encoder.parameters())
        chunks = []
        for start in range(0, len(flat), self.microbatch):
            part = flat[start:start+self.microbatch]
            n = len(part)
            pixels = preprocess_full_fov(part.to(device=parameter.device), self.spec)
            if n < self.microbatch:
                pixels = torch.cat((pixels, pixels[-1:].expand(self.microbatch-n, -1, -1, -1)), dim=0)
            with torch.autocast(device_type=parameter.device.type, enabled=False):
                hidden = self.encoder(pixel_values=pixels.to(parameter.dtype)).last_hidden_state
            chunks.append(extract_native_patches(hidden, self.spec)[:n].float())
        return torch.cat(chunks, dim=0).reshape(*rgb.shape[:-3], self.spec.grid_side**2, self.spec.width)
