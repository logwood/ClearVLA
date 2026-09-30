"""One frozen visual producer for mainline training and deployment.

Native patch tokens are rasterized ONCE onto a named full-RGB endpoint grid.
The result is derived evidence, not re-labelled native tokens. Every consumer
(including Teacher/references) receives that same grid. Border values replicate
the nearest patch-center feature; they do not claim extra observed pixels.
"""
from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn

from .dinov3_online import FrozenDinoV3Encoder, sample_native_patch_chart

ONLINE_MODE = "dinov3_online_v1"
FULL_RGB_CHART = "full_rgb_endpoint_v1"


def raster_identity() -> dict[str, object]:
    return {"shape": [16, 16], "image_hw": [336, 336],
            "grid": "linspace_-1_1_rgb_pixel_centers",
            "sample": "bilinear_native_patch_centers_border",
            "derived_not_native": True,
            "tensor_layout": "contiguous_row_major_visual_ingress",
            "raw_pyramid_centers": "padded_odd_kernel_strides_4_8_to_full_endpoints",
            "early_raw_context": "full_endpoint_2G_then_stride2_raster"}


def rasterize_full_rgb(native: Tensor, image_hw: tuple[int, int]) -> Tensor:
    """[...,256,768] -> same shape on a 16x16 full-image endpoint raster."""
    if native.shape[-2:] != (256, 768):
        raise ValueError("full RGB raster requires native ViT-B/16 patches")
    axis = torch.linspace(-1, 1, 16, device=native.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    flat = native.reshape(-1, 16, 16, 768)
    q = torch.stack((xx, yy), -1)[None].expand(flat.shape[0], -1, -1, -1)
    return sample_native_patch_chart(flat, q, image_hw=image_hw).reshape_as(native)


def resize_endpoint_chart(value: Tensor, hw: tuple[int, int], mode: str) -> Tensor:
    """NCHW: new chart samples declared endpoints; old pooling is untouched."""
    if mode == FULL_RGB_CHART:
        return torch.nn.functional.interpolate(value, size=hw, mode="bilinear", align_corners=True)
    if mode != "legacy_v1":
        raise ValueError("unknown visual chart")
    return torch.nn.functional.adaptive_avg_pool2d(value, hw)


class OnlineVisionPipeline(nn.Module):
    """Train-process owned; never put the encoder inside DataLoader workers."""
    def __init__(self, encoder: FrozenDinoV3Encoder):
        super().__init__()
        self.encoder = encoder
        self.last_batch_stats: dict[str, int] = {}
        self.train(False)

    def train(self, mode: bool = True):
        super().train(False)
        self.encoder.eval()
        return self

    @classmethod
    def from_config(cls, config, device: torch.device):
        if config.data.visual_feature_mode != ONLINE_MODE:
            raise ValueError("online visual producer is not selected")
        # Model construction must not consume the policy/training RNG stream.
        with torch.random.fork_rng(devices=[]):
            encoder = FrozenDinoV3Encoder.from_pretrained(
                config.data.dinov3_model, revision=config.data.dinov3_revision or None,
                local_files_only=config.data.dinov3_local_files_only, device=device,
                dtype={"bf16": torch.bfloat16, "fp32": torch.float32}[config.runtime.compute_dtype],
                microbatch=config.data.dinov3_microbatch,
            )
        return cls(encoder)

    def identity(self) -> dict[str, object]:
        return {"schema": "clearvla_online_vision_v1", "feature_mode": ONLINE_MODE,
                "chart": FULL_RGB_CHART, "encoder": self.encoder.identity(),
                "raster": raster_identity(),
                "camera_names": ["top", "wrist"], "disk_feature_cache": False}

    @torch.no_grad()
    def forward(self, rgb: Tensor) -> Tensor:
        if rgb.shape[-3:] != (3, 336, 336):
            raise ValueError("online mainline expects full 336px RGB, not a hidden crop")
        # Worker scatter produces contiguous rows; online inference must return
        # the same layout rather than grid_sample's channel-major view.
        return rasterize_full_rgb(self.encoder(rgb), (336, 336)).contiguous()

    @torch.no_grad()
    def prepare_batch(self, batch: Mapping[str, Tensor], config) -> dict[str, Tensor]:
        """Encode only supported requests, scatter into separate causal/label roles.

        Source keys identify episode, physical frame and camera. Duplicate keys
        must carry identical uint8 pixels; mixed augmentations are rejected rather
        than silently aliasing features. No persistent feature cache is used.
        """
        if "history_dinov2_tokens" in batch or "target_future_dinov2_tokens" in batch:
            raise ValueError("online RGB requests cannot be mixed with cached tokens")
        rgb = batch["visual_request_rgb"]
        keys = batch["visual_request_keys"]
        valid = batch["visual_request_valid"]
        if rgb.ndim != 6 or rgb.dtype != torch.uint8 or rgb.shape[-3:] != (3,336,336):
            raise ValueError("worker RGB requests must be uint8 [B,T,C,3,336,336]")
        if keys.shape != (*rgb.shape[:3], 3) or keys.dtype != torch.int64:
            raise ValueError("visual source keys must be int64 [B,T,C,3]")
        if valid.shape != rgb.shape[:3] or valid.dtype != torch.bool:
            raise ValueError("visual source support must be bool [B,T,C]")
        if keys.device.type != "cpu" or rgb.device.type != "cpu" or valid.device.type != "cpu":
            raise ValueError("visual request metadata is CPU worker transport")
        if rgb.shape[2] != 2 or tuple(config.data.camera_names) != ("top", "wrist"):
            raise ValueError("online visual requests require named top,wrist cameras")
        if not torch.equal(keys[...,2], torch.arange(2).expand(*keys.shape[:-1])):
            raise ValueError("source camera indices disagree with named camera axis")
        if bool((keys[valid] < 0).any()):
            raise ValueError("supported visual sources require nonnegative provenance")
        flat = rgb.flatten(0,2)
        selected = valid.flatten().nonzero().flatten().tolist()
        if not selected:
            raise ValueError("batch has no real visual source")
        unique: list[int] = []
        inverse: list[int] = []
        lookup: dict[tuple[int,...], int] = {}
        for index in selected:
            key = tuple(keys.reshape(-1,3)[index].tolist())
            if key not in lookup:
                lookup[key] = len(unique)
                unique.append(index)
            elif not torch.equal(flat[index], flat[unique[lookup[key]]]):
                raise ValueError("same source key has different RGB pixels/augmentation")
            inverse.append(lookup[key])
        tokens = self(flat[unique].float().div(255))
        output = tokens.new_zeros((len(flat),256,768))
        output.index_copy_(0, torch.tensor(selected, device=tokens.device),
                           tokens[torch.tensor(inverse, device=tokens.device)])
        output = output.reshape(*rgb.shape[:3],256,768)
        result = {k:v for k,v in batch.items() if not k.startswith("visual_request_")}
        h = 3
        f = int(config.dimensions.future_supports)
        # Materialize each role as its own contiguous tensor.  Slicing a
        # larger request canvas otherwise preserves the full-role stride in
        # the leading dimension.  Deployment encodes only the causal window,
        # so the same values arrive with a different layout; downstream
        # kernels can then take different BF16 paths and exact adapter parity
        # is lost.  The role boundary is the right place to make layout part
        # of the producer contract.
        result["history_dinov2_tokens"] = output[:, :h].clone(memory_format=torch.contiguous_format)
        result["target_future_dinov2_tokens"] = output[:, h:h+f].clone(memory_format=torch.contiguous_format)
        # Legacy tensor field names are ABI-internal containers, not encoder IDs.
        cursor = h+f
        if config.top.instruction_reference_mode != "none":
            result["instruction_reference_dino"] = output[:,cursor].clone(memory_format=torch.contiguous_format)
            result["instruction_reference_observed"] = valid[:,cursor].to(tokens.device)[...,None].expand(-1,-1,256)
            cursor += 1
        if config.top.world_feedback_mode != "none":
            result["executed_world_dino"] = output[:,cursor:cursor+3].clone(memory_format=torch.contiguous_format)
            cursor += 3
        if config.top.annotation_goal_mode != "none":
            result["annotation_endpoint_dino"] = output[:,cursor].clone(memory_format=torch.contiguous_format)
            result["annotation_endpoint_visual_observed"] = valid[:,cursor].to(tokens.device)[...,None].expand(-1,-1,256)
            cursor += 1
        if cursor != output.shape[1]:
            raise ValueError("online request role layout differs from graph")
        self.last_batch_stats = {"requested": len(flat), "supported": len(selected), "encoded_unique": len(unique)}
        return result


def rasterize_strided_rgb(value: Tensor, *, image_hw: tuple[int, int], stride: int) -> Tensor:
    """Padded odd-kernel conv centers (0,s,2s,...) -> full-image endpoints.

    Applies after the complete pyramid is built, not between its convolutions.
    This accounts for the last conv center not reaching the last input pixel.
    The boundary extension is explicit; it creates no extra image evidence.
    """
    if value.ndim != 4 or min(value.shape[-2:]) < 2 or stride < 1:
        raise ValueError("strided RGB chart must be NCHW with at least 2x2 centers")
    h, w = image_hw
    gh, gw = value.shape[-2:]
    if (gh,gw) != ((h+stride-1)//stride,(w+stride-1)//stride):
        raise ValueError("RGB feature shape disagrees with declared cumulative stride")
    with torch.autocast(device_type=value.device.type,enabled=False):
        x=torch.linspace(0,w-1,gw,device=value.device)/(stride*(gw-1))*2-1
        y=torch.linspace(0,h-1,gh,device=value.device)/(stride*(gh-1))*2-1
        yy,xx=torch.meshgrid(y,x,indexing='ij')
        grid=torch.stack((xx,yy),-1)[None].expand(value.shape[0],-1,-1,-1)
        result=torch.nn.functional.grid_sample(value.float(),grid,mode='bilinear',padding_mode='border',align_corners=True)
    return result.to(value.dtype)
