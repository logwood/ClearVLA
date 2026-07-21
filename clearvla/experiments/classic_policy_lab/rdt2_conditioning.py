from __future__ import annotations

"""Pluggable condition sources for the RDT2-FM action expert.

Official RDT2-FM receives per-layer KV caches from RDT2-VQ.  Dense visual tokens
are also supported deliberately so that a smaller vision plugin such as DINOv2
can be evaluated without modifying the flow-matching policy core.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn

from .rdt2_dinov2_cache import DinoV2TokenStore


@dataclass
class RDT2ConditionBatch:
    """One of ``kv_cache`` or ``dense_tokens`` must be present."""

    attention_mask: Tensor
    kv_cache: list[tuple[Tensor, Tensor]] | None = None
    dense_tokens: Tensor | None = None

    def validate(self) -> None:
        if (self.kv_cache is None) == (self.dense_tokens is None):
            raise ValueError("exactly one of kv_cache or dense_tokens must be supplied")
        if self.attention_mask.dim() != 2:
            raise ValueError(
                f"attention_mask must be [B,L], got {tuple(self.attention_mask.shape)}"
            )
        token_len = self.attention_mask.shape[1]
        if (
            self.dense_tokens is not None
            and self.dense_tokens.shape[:2] != self.attention_mask.shape
        ):
            raise ValueError("dense token and mask lengths differ")
        if self.kv_cache is not None:
            for layer, (key, value) in enumerate(self.kv_cache):
                if key.shape != value.shape or key.dim() != 4:
                    raise ValueError(
                        f"layer {layer}: KV tensors must match and have shape [B,Hkv,L,Dh]"
                    )
                if key.shape[0] != self.attention_mask.shape[0] or key.shape[2] != token_len:
                    raise ValueError(f"layer {layer}: KV cache is incompatible with attention mask")

    def to(self, *, device: torch.device, dtype: torch.dtype) -> "RDT2ConditionBatch":
        mask = self.attention_mask.to(device=device, dtype=torch.bool)
        dense = (
            None if self.dense_tokens is None else self.dense_tokens.to(device=device, dtype=dtype)
        )
        kv = None
        if self.kv_cache is not None:
            kv = [
                (key.to(device=device, dtype=dtype), value.to(device=device, dtype=dtype))
                for key, value in self.kv_cache
            ]
        out = RDT2ConditionBatch(attention_mask=mask, kv_cache=kv, dense_tokens=dense)
        out.validate()
        return out


class RDT2Conditioner(Protocol):
    def encode(
        self,
        images: Tensor,
        instructions: Sequence[str] | None = None,
        *,
        sample_keys: Tensor | None = None,
        image_ablation: str = "normal",
        camera_names: Sequence[str] | None = None,
    ) -> RDT2ConditionBatch:
        """Encode ``images`` shaped [B,Cam,3,H,W] into expert conditions."""


def _camera_keep_index(camera_names: Sequence[str] | None, target: str, cameras: int) -> int:
    if camera_names is not None and target in camera_names:
        return list(camera_names).index(target)
    fallback = {"top": 0, "wrist": 1}[target]
    if fallback >= cameras:
        raise ValueError(
            f"cannot apply {target}-only ablation with cameras={cameras}, names={camera_names}"
        )
    return fallback


def apply_image_ablation(
    images: Tensor,
    mode: str,
    *,
    camera_names: Sequence[str] | None = None,
) -> Tensor:
    """Apply deterministic visual ablations before an online conditioner.

    ``shuffle-episode`` is handled by the dataset-aware evaluator because it
    needs replacement episode/frame keys.  All other modes operate locally.
    """
    if images.dim() != 5:
        raise ValueError(f"images must be [B,Cam,3,H,W], got {tuple(images.shape)}")
    if mode in {"normal", "shuffle-episode"}:
        return images
    if mode == "zero":
        return torch.zeros_like(images)
    if mode == "mean":
        return images.mean(dim=(-1, -2), keepdim=True).expand_as(images)
    if mode == "shuffle-batch":
        return torch.roll(images, shifts=1, dims=0)
    if mode in {"top-only", "wrist-only"}:
        target = mode.removesuffix("-only")
        keep = _camera_keep_index(camera_names, target, images.shape[1])
        out = torch.zeros_like(images)
        out[:, keep] = images[:, keep]
        return out
    raise ValueError(f"unsupported image ablation mode={mode!r}")


def apply_token_ablation(
    tokens: Tensor,
    mode: str,
    *,
    camera_names: Sequence[str],
) -> Tensor:
    """Token-space counterpart of :func:`apply_image_ablation`.

    Input shape is ``[B,Cam,Patch,Dim]``.  Cached-token experiments therefore
    retain the same visual-ablation semantics without rerunning DINOv2.
    """
    if tokens.dim() != 4:
        raise ValueError(f"tokens must be [B,Cam,Patch,Dim], got {tuple(tokens.shape)}")
    if mode in {"normal", "shuffle-episode"}:
        return tokens
    if mode == "zero":
        return torch.zeros_like(tokens)
    if mode == "mean":
        return tokens.mean(dim=2, keepdim=True).expand_as(tokens)
    if mode == "shuffle-batch":
        return torch.roll(tokens, shifts=1, dims=0)
    if mode in {"top-only", "wrist-only"}:
        target = mode.removesuffix("-only")
        keep = _camera_keep_index(camera_names, target, tokens.shape[1])
        out = torch.zeros_like(tokens)
        out[:, keep] = tokens[:, keep]
        return out
    raise ValueError(f"unsupported image ablation mode={mode!r}")


class NullKVConditioner(nn.Module):
    """Strict qpos-only baseline: constant zero KV condition with no image use."""

    def __init__(self, *, depth: int, num_kv_heads: int, head_dim: int, tokens: int = 1) -> None:
        super().__init__()
        self.depth = int(depth)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.tokens = int(tokens)

    @torch.no_grad()
    def encode(
        self,
        images: Tensor,
        instructions: Sequence[str] | None = None,
        *,
        sample_keys: Tensor | None = None,
        image_ablation: str = "normal",
        camera_names: Sequence[str] | None = None,
    ) -> RDT2ConditionBatch:
        if images.dim() != 5:
            raise ValueError(f"images must be [B,Cam,3,H,W], got {tuple(images.shape)}")
        batch = images.shape[0]
        device = images.device
        dtype = images.dtype if images.is_floating_point() else torch.float32
        base = torch.zeros(
            (batch, self.num_kv_heads, self.tokens, self.head_dim), device=device, dtype=dtype
        )
        kv = [(base.clone(), base.clone()) for _ in range(self.depth)]
        mask = torch.ones((batch, self.tokens), dtype=torch.bool, device=device)
        out = RDT2ConditionBatch(attention_mask=mask, kv_cache=kv)
        out.validate()
        return out


class DebugDenseConditioner(nn.Module):
    """Dependency-free deterministic dense-token plugin for tests only."""

    def __init__(self, token_dim: int = 32, tokens_per_camera: int = 4) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.tokens_per_camera = int(tokens_per_camera)

    @torch.no_grad()
    def encode(
        self,
        images: Tensor,
        instructions: Sequence[str] | None = None,
        *,
        sample_keys: Tensor | None = None,
        image_ablation: str = "normal",
        camera_names: Sequence[str] | None = None,
    ) -> RDT2ConditionBatch:
        images = apply_image_ablation(images, image_ablation, camera_names=camera_names)
        batch, cameras = images.shape[:2]
        pooled = images.float().mean(dim=(-1, -2, -3), keepdim=False)  # [B,Cam]
        tokens = pooled[:, :, None, None].expand(
            batch, cameras, self.tokens_per_camera, self.token_dim
        )
        tokens = tokens.reshape(batch, cameras * self.tokens_per_camera, self.token_dim)
        mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        out = RDT2ConditionBatch(attention_mask=mask, dense_tokens=tokens)
        out.validate()
        return out


class DebugKVConditioner(nn.Module):
    """Dependency-free RDT2-VQ-shaped KV plugin for shape and load tests."""

    def __init__(self, *, depth: int, num_kv_heads: int, head_dim: int, tokens: int = 8) -> None:
        super().__init__()
        self.depth = int(depth)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.tokens = int(tokens)

    @torch.no_grad()
    def encode(
        self,
        images: Tensor,
        instructions: Sequence[str] | None = None,
        *,
        sample_keys: Tensor | None = None,
        image_ablation: str = "normal",
        camera_names: Sequence[str] | None = None,
    ) -> RDT2ConditionBatch:
        images = apply_image_ablation(images, image_ablation, camera_names=camera_names)
        batch = images.shape[0]
        device = images.device
        dtype = images.dtype if images.is_floating_point() else torch.float32
        # Make the token value depend weakly on the observation so the debug path is not silently constant.
        scalar = images.float().mean(dim=(1, 2, 3, 4)).to(device=device, dtype=dtype)
        base = scalar[:, None, None, None].expand(
            batch, self.num_kv_heads, self.tokens, self.head_dim
        )
        kv = [(base.clone(), base.clone()) for _ in range(self.depth)]
        mask = torch.ones((batch, self.tokens), dtype=torch.bool, device=device)
        out = RDT2ConditionBatch(attention_mask=mask, kv_cache=kv)
        out.validate()
        return out


class DinoV2DenseConditioner(nn.Module):
    """Frozen DINOv2 patch-token condition plugin.

    This is an experimental replacement, not an upstream RDT2-VQ equivalent.
    It returns dense visual tokens.  The RDT2 expert should therefore enable a
    ``lang_adaptor`` when the DINO token width is not 1024.
    """

    def __init__(
        self,
        model_name_or_path: str = "facebook/dinov2-large",
        *,
        local_files_only: bool = False,
        drop_cls_token: bool = True,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("DINOv2 plugin requires transformers") from exc
        self.processor = AutoImageProcessor.from_pretrained(
            model_name_or_path, local_files_only=local_files_only
        )
        self.encoder = AutoModel.from_pretrained(
            model_name_or_path, local_files_only=local_files_only
        )
        self.encoder.requires_grad_(False).eval()
        self.drop_cls_token = bool(drop_cls_token)
        self.token_dim = int(self.encoder.config.hidden_size)

    def train(self, mode: bool = True):
        super().train(False)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def encode(
        self,
        images: Tensor,
        instructions: Sequence[str] | None = None,
        *,
        sample_keys: Tensor | None = None,
        image_ablation: str = "normal",
        camera_names: Sequence[str] | None = None,
    ) -> RDT2ConditionBatch:
        images = apply_image_ablation(images, image_ablation, camera_names=camera_names)
        batch, cameras = images.shape[:2]
        flat = images.detach().float().cpu().reshape(-1, *images.shape[2:]).clamp(0, 1)
        # HF processors accept CHW tensors; preprocessing stays on CPU by design.
        encoded = self.processor(images=list(flat), return_tensors="pt", do_rescale=False)
        parameter = next(self.encoder.parameters())
        device, dtype = parameter.device, parameter.dtype
        encoded = {
            key: value.to(device=device, dtype=dtype)
            if value.is_floating_point()
            else value.to(device=device)
            for key, value in encoded.items()
        }
        hidden = self.encoder(**encoded).last_hidden_state
        if self.drop_cls_token:
            hidden = hidden[:, 1:]
        hidden = hidden.reshape(batch, cameras * hidden.shape[1], hidden.shape[2])
        mask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
        out = RDT2ConditionBatch(attention_mask=mask, dense_tokens=hidden)
        out.validate()
        return out


class CachedDinoV2DenseConditioner(nn.Module):
    """Read frozen DINOv2 patch tokens from a strict mmap cache."""

    def __init__(self, store: DinoV2TokenStore) -> None:
        super().__init__()
        self.store = store
        self.token_dim = int(store.token_dim)
        self.camera_names = tuple(store.camera_names)

    @torch.no_grad()
    def encode(
        self,
        images: Tensor,
        instructions: Sequence[str] | None = None,
        *,
        sample_keys: Tensor | None = None,
        image_ablation: str = "normal",
        camera_names: Sequence[str] | None = None,
    ) -> RDT2ConditionBatch:
        if sample_keys is None:
            raise ValueError(
                "cached DINOv2 conditioner requires sample_keys=[episode_idx,image_index]"
            )
        tokens = self.store.load_batch(sample_keys)
        tokens = apply_token_ablation(tokens, image_ablation, camera_names=self.camera_names)
        batch = tokens.shape[0]
        tokens = tokens.reshape(batch, -1, tokens.shape[-1])
        mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        out = RDT2ConditionBatch(attention_mask=mask, dense_tokens=tokens)
        out.validate()
        return out


class RDT2VQKVConditioner(nn.Module):
    """Official-style RDT2-VQ plugin returning selected Qwen2.5-VL KV caches."""

    def __init__(
        self,
        model_name_or_path: str = "robotics-diffusion-transformer/RDT2-VQ",
        *,
        selected_layers: Sequence[int] = tuple(range(14)),
        processor_name_or_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        dtype: torch.dtype = torch.bfloat16,
        local_files_only: bool = False,
        attn_implementation: str | None = "flash_attention_2",
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "RDT2-VQ plugin requires a transformers build with Qwen2.5-VL"
            ) from exc
        kwargs = {"torch_dtype": dtype, "local_files_only": local_files_only}
        if attn_implementation is not None:
            kwargs["attn_implementation"] = attn_implementation
        self.processor = AutoProcessor.from_pretrained(
            processor_name_or_path,
            padding_side="left",
            use_fast=True,
            local_files_only=local_files_only,
        )
        self.encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path, **kwargs
        )
        self.encoder.requires_grad_(False).eval()
        self.selected_layers = tuple(int(x) for x in selected_layers)

    def train(self, mode: bool = True):
        super().train(False)
        self.encoder.eval()
        return self

    @staticmethod
    def _pil_from_chw(image: Tensor) -> Image.Image:
        array = image.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        return Image.fromarray(np.asarray(np.rint(array * 255), dtype=np.uint8), mode="RGB")

    @torch.no_grad()
    def encode(
        self,
        images: Tensor,
        instructions: Sequence[str] | None = None,
        *,
        sample_keys: Tensor | None = None,
        image_ablation: str = "normal",
        camera_names: Sequence[str] | None = None,
    ) -> RDT2ConditionBatch:
        images = apply_image_ablation(images, image_ablation, camera_names=camera_names)
        batch, cameras = images.shape[:2]
        if instructions is None:
            instructions = [""] * batch
        if len(instructions) != batch:
            raise ValueError("instruction count must match batch size")
        texts, pil_images = [], []
        for row in range(batch):
            camera_images = [self._pil_from_chw(images[row, cam]) for cam in range(cameras)]
            arrays = [np.asarray(image) for image in camera_images]
            merged = Image.fromarray(np.concatenate(arrays, axis=1), mode="RGB")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": str(instructions[row])},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(messages, add_generation_prompt=False)
            text += "<|im_start|>assistant\n<|quad_start|>"
            texts.append(text)
            pil_images.append([merged])
        inputs = self.processor(text=texts, images=pil_images, padding=True, return_tensors="pt")
        device = next(self.encoder.parameters()).device
        inputs = {key: value.to(device=device) for key, value in inputs.items()}
        outputs = self.encoder(**inputs, use_cache=True)
        cache = outputs.past_key_values
        kv = [(cache[layer][0], cache[layer][1]) for layer in self.selected_layers]
        mask = inputs["attention_mask"].to(dtype=torch.bool)
        out = RDT2ConditionBatch(attention_mask=mask, kv_cache=kv)
        out.validate()
        return out


__all__ = [
    "RDT2ConditionBatch",
    "RDT2Conditioner",
    "apply_image_ablation",
    "apply_token_ablation",
    "NullKVConditioner",
    "DebugDenseConditioner",
    "DebugKVConditioner",
    "DinoV2DenseConditioner",
    "CachedDinoV2DenseConditioner",
    "RDT2VQKVConditioner",
]
