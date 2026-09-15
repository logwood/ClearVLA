from __future__ import annotations

"""RDT2-FM flow-matching action expert reference implementation.

This module reproduces the released RDT2-FM action-expert architecture rather
than the earlier RDT-170M/RDT-small model.  The official active path consumes
per-layer KV caches produced by the RDT2-VQ Qwen2.5-VL backbone.  The backbone is
kept outside this module intentionally: the action expert is a standalone,
replaceable policy core and the condition source is a plugin.

The state-dict layout mirrors the released ``models/rdt_runner.py`` stack so the
official RDT2-FM checkpoint can be loaded strictly when ``action_dim=20`` and
``state_dim=20``.  A dense-token condition path is retained for experiments such
as DINOv2 replacement; it is not the official pretrained path.
"""

import math
import re
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import LogisticNormal


# -----------------------------------------------------------------------------
# Positional embeddings: equivalent to the released RDT2 helper.
# -----------------------------------------------------------------------------
def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even, got {embed_dim}")
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = np.asarray(pos).reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_nd_sincos_pos_embed_from_grid(embed_dim: int, grid_sizes: tuple[int, ...]) -> np.ndarray:
    emb = np.zeros(grid_sizes + (embed_dim,), dtype=np.float64)
    for size_idx, grid_size in enumerate(grid_sizes):
        if grid_size <= 1:
            continue
        pos = np.arange(grid_size)
        posemb_shape = [1] * len(grid_sizes) + [embed_dim]
        posemb_shape[size_idx] = -1
        emb += get_1d_sincos_pos_embed_from_grid(embed_dim, pos).reshape(posemb_shape)
    return emb


def get_multimodal_pos_embed(
    embed_dim: int, mm_lens: OrderedDict[str, int | tuple[int, ...]]
) -> np.ndarray:
    total_len = 0
    for modality, cond_len in mm_lens.items():
        if modality == "image" and isinstance(cond_len, (tuple, list)):
            total_len += int(np.prod([abs(int(x)) for x in cond_len]))
        else:
            total_len += abs(int(cond_len))
    num_modalities = len(mm_lens)
    modality_pos_embed = None
    if num_modalities > 1:
        modality_pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, np.arange(num_modalities))
    pos_emb = np.zeros((total_len, embed_dim), dtype=np.float64)
    start = 0
    for idx, (modality, cond_len) in enumerate(mm_lens.items()):
        if modality == "image" and isinstance(cond_len, (tuple, list)):
            all_grid_sizes = tuple(abs(int(x)) for x in cond_len)
            embed_grid_sizes = tuple(int(x) if int(x) > 0 else 1 for x in cond_len)
            cond = get_nd_sincos_pos_embed_from_grid(embed_dim, embed_grid_sizes)
            current = np.zeros(all_grid_sizes + (embed_dim,), dtype=np.float64)
            current += cond
            current = current.reshape(-1, embed_dim)
        else:
            length = int(cond_len)
            cond = (
                get_1d_sincos_pos_embed_from_grid(embed_dim, np.arange(length)) if length > 1 else 0
            )
            current = np.zeros((abs(length), embed_dim), dtype=np.float64)
            current += cond
        if modality_pos_embed is not None:
            current += modality_pos_embed[idx]
        pos_emb[start : start + len(current)] = current
        start += len(current)
    return pos_emb


# -----------------------------------------------------------------------------
# Core layers. Names intentionally match the released checkpoint keys.
# -----------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        output = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return output.to(dtype=x.dtype) * self.weight


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """Repeat grouped KV heads. Input shape: [B, L, Hkv, Dh]."""
    batch, tokens, kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(batch, tokens, kv_heads, n_rep, head_dim)
        .reshape(batch, tokens, kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.n_heads = int(config["num_heads"])
        self.n_kv_heads = (
            self.n_heads if config.get("num_kv_heads") is None else int(config["num_kv_heads"])
        )
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.n_rep = self.n_heads // self.n_kv_heads
        self.hidden_size = int(config["hidden_size"])
        if self.hidden_size % self.n_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.head_size = self.hidden_size // self.n_heads
        self.wq = nn.Linear(self.hidden_size, self.n_heads * self.head_size, bias=False)
        self.wkv = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_size * 2, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_size, self.hidden_size, bias=False)
        self.norm_q = RMSNorm(self.head_size, eps=float(config["norm_eps"]))
        self.norm_k = RMSNorm(self.head_size, eps=float(config["norm_eps"]))
        self.use_flash_attn = bool(config.get("use_flash_attn", True))
        self.attn_scale = 1.0 / math.sqrt(self.head_size)

    def forward(
        self,
        x: Tensor,
        *,
        is_causal: bool = False,
        mask: Tensor | None = None,
    ) -> Tensor:
        batch, tokens, _ = x.shape
        q = self.wq(x).view(batch, tokens, self.n_heads, self.head_size)
        kv = self.wkv(x).view(batch, tokens, self.n_kv_heads, self.head_size, 2)
        k, v = kv.unbind(-1)
        q, k = self.norm_q(q), self.norm_k(k)
        k, v = repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        attn_mask = None
        if mask is not None:
            if is_causal:
                raise ValueError(
                    "use either an explicit self-attention mask or is_causal, not both"
                )
            mask = mask.to(device=x.device, dtype=torch.bool)
            if mask.ndim == 2:
                if tuple(mask.shape) != (tokens, tokens):
                    raise ValueError(
                        f"2-D self-attention mask must be [Q,K]={tokens, tokens}, got {tuple(mask.shape)}"
                    )
                attn_mask = mask.reshape(1, 1, tokens, tokens).expand(batch, -1, -1, -1)
            elif mask.ndim == 3:
                if tuple(mask.shape) != (batch, tokens, tokens):
                    raise ValueError(
                        f"3-D self-attention mask must be [B,Q,K]={batch, tokens, tokens}, got {tuple(mask.shape)}"
                    )
                attn_mask = mask.unsqueeze(1)
            else:
                raise ValueError("self-attention mask must be [Q,K] or [B,Q,K]")
        if self.use_flash_attn:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=is_causal and attn_mask is None,
                scale=self.attn_scale,
            )
        else:
            scores = torch.matmul(q, k.transpose(2, 3)) * self.attn_scale
            if is_causal:
                causal = torch.ones(tokens, tokens, dtype=torch.bool, device=x.device).tril()
                scores = scores.masked_fill(
                    causal.logical_not().reshape(1, 1, tokens, tokens), float("-inf")
                )
            if attn_mask is not None:
                scores = scores.masked_fill(attn_mask.logical_not(), float("-inf"))
            probs = F.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
            out = torch.matmul(probs, v)
        out = out.transpose(1, 2).contiguous().view(batch, tokens, -1)
        return self.wo(out)


class CrossAttention(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.n_heads = int(config["num_heads"])
        self.n_kv_heads = (
            self.n_heads if config.get("num_kv_heads") is None else int(config["num_kv_heads"])
        )
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.n_rep = self.n_heads // self.n_kv_heads
        self.hidden_size = int(config["hidden_size"])
        if self.hidden_size % self.n_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.head_size = self.hidden_size // self.n_heads
        self.wq = nn.Linear(self.hidden_size, self.n_heads * self.head_size, bias=False)
        self.wkv = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_size * 2, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_size, self.hidden_size, bias=False)
        self.norm_q = RMSNorm(self.head_size, eps=float(config["norm_eps"]))
        self.norm_k = RMSNorm(self.head_size, eps=float(config["norm_eps"]))
        self.use_flash_attn = bool(config.get("use_flash_attn", True))
        self.attn_scale = 1.0 / math.sqrt(self.head_size)

    def forward(
        self,
        x: Tensor,
        c: Tensor | None = None,
        ck: Tensor | None = None,
        cv: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        batch, tokens, _ = x.shape
        q = self.wq(x).view(batch, tokens, self.n_heads, self.head_size)
        q = self.norm_q(q)
        if c is not None:
            cond_tokens = c.shape[1]
            kv = self.wkv(c).view(batch, cond_tokens, self.n_kv_heads, self.head_size, 2)
            ck, cv = kv.unbind(-1)
            ck = self.norm_k(ck)
        if ck is None or cv is None:
            raise ValueError(
                "cross attention requires dense condition tokens or an external KV cache"
            )
        ck, cv = repeat_kv(ck, self.n_rep), repeat_kv(cv, self.n_rep)
        q, ck, cv = q.transpose(1, 2), ck.transpose(1, 2), cv.transpose(1, 2)
        attn_mask = None
        if mask is not None:
            mask = mask.to(device=x.device, dtype=torch.bool)
            if mask.ndim == 2:
                if tuple(mask.shape) != (batch, ck.shape[2]):
                    raise ValueError(
                        f"2-D cross-attention mask must be [B,K]={batch, ck.shape[2]}, got {tuple(mask.shape)}"
                    )
                attn_mask = mask.reshape(batch, 1, 1, -1).expand(-1, -1, tokens, -1)
            elif mask.ndim == 3:
                if tuple(mask.shape) != (batch, tokens, ck.shape[2]):
                    raise ValueError(
                        f"3-D cross-attention mask must be [B,Q,K]={batch, tokens, ck.shape[2]}, got {tuple(mask.shape)}"
                    )
                attn_mask = mask.unsqueeze(1)
            else:
                raise ValueError("cross-attention mask must be [B,K] or [B,Q,K]")
        if self.use_flash_attn:
            out = F.scaled_dot_product_attention(
                q,
                ck,
                cv,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,
                scale=self.attn_scale,
            )
        else:
            scores = torch.matmul(q, ck.transpose(2, 3)) * self.attn_scale
            if attn_mask is not None:
                scores = scores.masked_fill(attn_mask.logical_not(), float("-inf"))
            probs = F.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
            out = torch.matmul(probs, cv)
        out = out.transpose(1, 2).contiguous().view(batch, tokens, -1)
        return self.wo(out)


class FeedForward(nn.Module):
    def __init__(
        self, dim: int, hidden_dim: int, multiple_of: int, ffn_dim_multiplier: float | None
    ) -> None:
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(float(ffn_dim_multiplier) * hidden_dim)
        hidden_dim = int(multiple_of) * ((hidden_dim + int(multiple_of) - 1) // int(multiple_of))
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Mlp(nn.Module):
    """Minimal timm-compatible MLP with matching state-dict names."""

    def __init__(self, in_features: int, hidden_features: int, out_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.SiLU()
        self.drop1 = nn.Identity()
        self.norm = nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        return self.drop2(x)


class TimestepEmbedder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = int(frequency_embedding_size)
        self.dtype = dtype

    def timestep_embedding(self, t: Tensor, dim: int, max_period: int = 10000) -> Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(self.dtype)

    def forward(self, t: Tensor) -> Tensor:
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class RDTBlock(nn.Module):
    def __init__(self, layer_idx: int, config: dict[str, Any]) -> None:
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.hidden_size = int(config["hidden_size"])
        eps = float(config["norm_eps"])
        self.attn_norm = RMSNorm(self.hidden_size, eps=eps)
        self.attn = Attention(config)
        self.cross_norm = RMSNorm(self.hidden_size, eps=eps)
        self.cond_norm = RMSNorm(self.hidden_size, eps=eps)
        self.cross_attn = CrossAttention(config)
        self.ffn_norm = RMSNorm(self.hidden_size, eps=eps)
        self.ffn = FeedForward(
            dim=self.hidden_size,
            hidden_dim=4 * self.hidden_size,
            multiple_of=int(config["multiple_of"]),
            ffn_dim_multiplier=config.get("ffn_dim_multiplier"),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(2 * self.hidden_size, 9 * self.hidden_size, bias=True)
        )

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        c: Tensor | None = None,
        ck: Tensor | None = None,
        cv: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        if t.shape[1] != 2 * self.hidden_size:
            raise ValueError(
                f"expected modulation input [B,{2 * self.hidden_size}], got {tuple(t.shape)}"
            )
        (
            shift_attn,
            scale_attn,
            gate_attn,
            shift_cross,
            scale_cross,
            gate_cross,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(t).chunk(9, dim=1)
        h = x + gate_attn.unsqueeze(1) * self.attn(
            modulate(self.attn_norm(x), shift_attn, scale_attn)
        )
        if c is not None:
            cross = self.cross_attn(
                modulate(self.cross_norm(h), shift_cross, scale_cross),
                c=self.cond_norm(c),
                mask=mask,
            )
        else:
            cross = self.cross_attn(
                modulate(self.cross_norm(h), shift_cross, scale_cross), ck=ck, cv=cv, mask=mask
            )
        h = h + gate_cross.unsqueeze(1) * cross
        return h + gate_mlp.unsqueeze(1) * self.ffn(
            modulate(self.ffn_norm(h), shift_mlp, scale_mlp)
        )


class FinalLayer(nn.Module):
    def __init__(self, output_size: int, config: dict[str, Any]) -> None:
        super().__init__()
        hidden_size = int(config["hidden_size"])
        self.hidden_size = hidden_size
        self.output_size = int(output_size)
        self.ffn_norm = RMSNorm(hidden_size, eps=float(config["norm_eps"]))
        self.ffn = Mlp(hidden_size, hidden_size * 4, self.output_size)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(2 * hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        return self.ffn(modulate(self.ffn_norm(x), shift, scale))


class RDT(nn.Module):
    def __init__(
        self,
        *,
        horizon: int,
        output_size: int,
        config: dict[str, Any],
        x_pos_emb_config: list[tuple[str, int]],
        lang_pos_emb_config: list[tuple[str, int]] | None = None,
        max_lang_len: int | None = None,
        img_pos_emb_config: list[tuple[str, int | tuple[int, ...]]] | None = None,
        max_img_len: int | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.hidden_size = int(config["hidden_size"])
        self.n_heads = int(config["num_heads"])
        self.dtype = dtype
        self.t_embedder = TimestepEmbedder(self.hidden_size, dtype=dtype)
        self.depth = int(config["depth"])
        self.blocks = nn.ModuleList(
            [RDTBlock(layer_idx, config=config) for layer_idx in range(self.depth)]
        )
        self.final_layer = FinalLayer(output_size, config=config)
        self.num_register_tokens = int(config.get("num_register_tokens", 4))
        self.register_tokens = nn.Parameter(
            torch.randn(1, self.num_register_tokens, self.hidden_size)
        )
        self.x_pos_emb_config = x_pos_emb_config
        self.img_pos_emb_config = img_pos_emb_config
        self.img_pos_emb = (
            nn.Parameter(torch.zeros(1, int(max_img_len), self.hidden_size))
            if img_pos_emb_config is not None
            else None
        )
        self.state_pos_emb_config = [("state", 1)]
        self.x_pos_emb = nn.Parameter(
            torch.zeros(1, self.horizon + self.num_register_tokens, self.hidden_size)
        )
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(basic_init)
        x_pos = get_multimodal_pos_embed(self.hidden_size, OrderedDict(self.x_pos_emb_config))
        self.x_pos_emb.data.copy_(torch.from_numpy(x_pos).float().unsqueeze(0))
        if self.img_pos_emb is not None and self.img_pos_emb_config is not None:
            img_pos = get_multimodal_pos_embed(
                self.hidden_size, OrderedDict(self.img_pos_emb_config)
            )
            self.img_pos_emb.data.copy_(torch.from_numpy(img_pos).float().unsqueeze(0))
        state_pos = get_multimodal_pos_embed(
            self.hidden_size, OrderedDict(self.state_pos_emb_config)
        )
        self.state_pos_emb = nn.Parameter(torch.from_numpy(state_pos).float().unsqueeze(0))
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.ffn.fc2.weight, 0)
        nn.init.constant_(self.final_layer.ffn.fc2.bias, 0)
        self.to(self.dtype)

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        lang_c: Tensor | None = None,
        lang_c_kv: list[tuple[Tensor, Tensor]] | None = None,
        img_c: Tensor | None = None,
        state_c: Tensor | None = None,
        lang_mask: Tensor | None = None,
        img_mask: Tensor | None = None,
    ) -> Tensor:
        t = self.t_embedder(t)
        if t.shape[0] == 1:
            t = t.expand(x.shape[0], -1)
        if state_c is None:
            raise ValueError("state condition must be provided")
        state_c = state_c + self.state_pos_emb
        modulation = torch.cat([t.unsqueeze(1), state_c], dim=1).reshape(
            x.shape[0], self.hidden_size * 2
        )
        registers = self.register_tokens.expand(x.shape[0], -1, -1)
        x = torch.cat([x, registers], dim=1) + self.x_pos_emb
        if img_c is not None and self.img_pos_emb is not None:
            img_c = img_c + self.img_pos_emb
        conds: list[Any] = [lang_c_kv if lang_c_kv is not None else lang_c]
        masks: list[Tensor | None] = [lang_mask]
        if self.img_pos_emb is not None:
            conds.append(img_c)
            masks.append(img_mask)
        if conds[0] is None:
            raise ValueError("RDT2-FM requires either VLM KV cache or dense condition tokens")
        for layer_idx, block in enumerate(self.blocks):
            cond, mask = conds[layer_idx % len(conds)], masks[layer_idx % len(masks)]
            ck = cv = None
            if isinstance(cond, list):
                ck, cv = cond[layer_idx % len(cond)]
                # Hugging Face VLM caches are [B,Hkv,L,Dh]; CrossAttention uses [B,L,Hkv,Dh].
                ck, cv = ck.transpose(1, 2), cv.transpose(1, 2)
                cond = None
            elif cond.dim() == 4:
                cond = cond[:, layer_idx]
            x = block(x, modulation, c=cond, ck=ck, cv=cv, mask=mask)
        x = self.final_layer(x, modulation)
        return x[:, : -self.num_register_tokens]


# -----------------------------------------------------------------------------
# Public reference runner.
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class RDT2FMReferenceConfig:
    action_dim: int = 20
    state_dim: int = 20
    prediction_horizon: int = 24
    hidden_size: int = 1024
    depth: int = 14
    num_heads: int = 8
    num_kv_heads: int = 4
    num_register_tokens: int = 4
    norm_eps: float = 1e-5
    multiple_of: int = 256
    ffn_dim_multiplier: float | None = None
    use_flash_attn: bool = True
    num_inference_timesteps: int = 5
    lang_adaptor: str | None = None
    lang_token_dim: int | None = None
    img_adaptor: str | None = None
    img_token_dim: int | None = None
    img_pos_emb_config: tuple[tuple[str, tuple[int, ...]], ...] | None = None
    max_img_len: int | None = None

    def validate(self) -> None:
        if (
            min(
                self.action_dim,
                self.state_dim,
                self.prediction_horizon,
                self.hidden_size,
                self.depth,
                self.num_heads,
                self.num_kv_heads,
            )
            <= 0
        ):
            raise ValueError("RDT2-FM dimensions must be positive")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if self.lang_adaptor is not None and self.lang_token_dim is None:
            raise ValueError("lang_token_dim is required when lang_adaptor is enabled")
        if self.img_adaptor is not None and self.img_token_dim is None:
            raise ValueError("img_token_dim is required when img_adaptor is enabled")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def upstream_compatible(self) -> bool:
        return (
            self.action_dim == 20
            and self.state_dim == 20
            and self.prediction_horizon == 24
            and self.hidden_size == 1024
            and self.depth == 14
            and self.num_heads == 8
            and self.num_kv_heads == 4
            and self.num_register_tokens == 4
            and self.lang_adaptor is None
            and self.img_adaptor is None
        )


class RDT2FMReference(nn.Module):
    """Standalone RDT2-FM action expert with pluggable condition sources."""

    def __init__(
        self,
        config: RDT2FMReferenceConfig = RDT2FMReferenceConfig(),
        *,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        rdt_cfg: dict[str, Any] = {
            "hidden_size": config.hidden_size,
            "depth": config.depth,
            "num_heads": config.num_heads,
            "num_kv_heads": config.num_kv_heads,
            "num_register_tokens": config.num_register_tokens,
            "norm_eps": config.norm_eps,
            "multiple_of": config.multiple_of,
            "ffn_dim_multiplier": config.ffn_dim_multiplier,
            "use_flash_attn": config.use_flash_attn,
        }
        img_pos_emb_config = (
            None if config.img_pos_emb_config is None else list(config.img_pos_emb_config)
        )
        self.model = RDT(
            horizon=config.prediction_horizon,
            output_size=config.action_dim,
            config=rdt_cfg,
            x_pos_emb_config=[
                ("action", config.prediction_horizon),
                ("register", config.num_register_tokens),
            ],
            img_pos_emb_config=img_pos_emb_config,
            max_img_len=config.max_img_len,
            dtype=dtype,
        )
        self.lang_adaptor = self._build_adapter(
            config.lang_adaptor, config.lang_token_dim, config.hidden_size
        )
        self.img_adaptor = self._build_adapter(
            config.img_adaptor, config.img_token_dim, config.hidden_size
        )
        self.act_adaptor = self._build_adapter("mlp3x_silu", config.action_dim, config.hidden_size)
        self.state_adaptor = self._build_adapter("mlp3x_silu", config.state_dim, config.hidden_size)
        self.num_inference_timesteps = int(config.num_inference_timesteps)
        self.timestep_sampler = LogisticNormal(0, 1)
        self.pred_horizon = int(config.prediction_horizon)
        self.action_dim = int(config.action_dim)
        self.to(dtype=dtype)

    @staticmethod
    def _build_adapter(
        kind: str | None, in_features: int | None, out_features: int
    ) -> nn.Module | None:
        if kind is None:
            return None
        if in_features is None:
            raise ValueError(f"in_features required for adapter {kind}")
        if kind == "linear":
            return nn.Linear(in_features, out_features)
        match = re.match(r"^mlp(\d+)x_silu$", kind)
        if match:
            depth = int(match.group(1))
            modules: list[nn.Module] = [nn.Linear(in_features, out_features)]
            for _ in range(1, depth):
                modules.extend([nn.SiLU(), nn.Linear(out_features, out_features)])
            return nn.Sequential(*modules)
        raise ValueError(f"unknown adapter type: {kind}")

    def sample_timesteps(self, batch_size: int, device: torch.device) -> Tensor:
        # Match upstream: LogisticNormal(0,1) represented as a two-class simplex; use first component.
        distribution = LogisticNormal(
            torch.tensor(0.0, device=device),
            torch.tensor(1.0, device=device),
        )
        return distribution.sample((batch_size,))[:, 0]

    def adapt_conditions(
        self,
        lang_tokens: Tensor | None,
        img_tokens: Tensor | None,
        action_tokens: Tensor | None,
        state_tokens: Tensor,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor]:
        lang = (
            self.lang_adaptor(lang_tokens)
            if self.lang_adaptor is not None and lang_tokens is not None
            else lang_tokens
        )
        img = (
            self.img_adaptor(img_tokens)
            if self.img_adaptor is not None and img_tokens is not None
            else img_tokens
        )
        action = self.act_adaptor(action_tokens) if action_tokens is not None else None
        state = self.state_adaptor(state_tokens)
        return lang, img, action, state

    def compute_loss(
        self,
        *,
        state_tokens: Tensor,
        action_gt: Tensor,
        lang_tokens: Tensor | None = None,
        lang_kv_cache: list[tuple[Tensor, Tensor]] | None = None,
        lang_attn_mask: Tensor | None = None,
        img_tokens: Tensor | None = None,
    ) -> Tensor:
        if state_tokens.dim() == 2:
            state_tokens = state_tokens.unsqueeze(1)
        batch = action_gt.shape[0]
        noise = torch.randn_like(action_gt)
        timesteps = self.sample_timesteps(batch, action_gt.device).to(dtype=action_gt.dtype)
        blend = timesteps.view(-1, 1, 1)
        noisy_action = action_gt * blend + noise * (1 - blend)
        lang_cond, img_cond, action_traj, state_cond = self.adapt_conditions(
            lang_tokens, img_tokens, noisy_action, state_tokens
        )
        if action_traj is None:
            raise AssertionError("action adaptor returned None")
        pred = self.model(
            x=action_traj,
            t=timesteps,
            lang_c=lang_cond,
            lang_c_kv=lang_kv_cache,
            lang_mask=lang_attn_mask,
            img_c=img_cond,
            state_c=state_cond,
        )
        return F.mse_loss(pred, action_gt - noise)

    def predict_velocity(
        self,
        *,
        state_tokens: Tensor,
        noisy_action: Tensor,
        timesteps: Tensor,
        lang_tokens: Tensor | None = None,
        lang_kv_cache: list[tuple[Tensor, Tensor]] | None = None,
        lang_attn_mask: Tensor | None = None,
        img_tokens: Tensor | None = None,
    ) -> Tensor:
        if state_tokens.dim() == 2:
            state_tokens = state_tokens.unsqueeze(1)
        lang_cond, img_cond, action_traj, state_cond = self.adapt_conditions(
            lang_tokens, img_tokens, noisy_action, state_tokens
        )
        if action_traj is None:
            raise AssertionError("action adaptor returned None")
        return self.model(
            x=action_traj,
            t=timesteps,
            lang_c=lang_cond,
            lang_c_kv=lang_kv_cache,
            lang_mask=lang_attn_mask,
            img_c=img_cond,
            state_c=state_cond,
        )

    @torch.no_grad()
    def predict_action(
        self,
        *,
        state_tokens: Tensor,
        lang_tokens: Tensor | None = None,
        lang_kv_cache: list[tuple[Tensor, Tensor]] | None = None,
        lang_attn_mask: Tensor | None = None,
        img_tokens: Tensor | None = None,
        noisy_action: Tensor | None = None,
        generator: torch.Generator | None = None,
        inference_steps: int | None = None,
    ) -> Tensor:
        if state_tokens.dim() == 2:
            state_tokens = state_tokens.unsqueeze(1)
        batch = state_tokens.shape[0]
        device = state_tokens.device
        dtype = state_tokens.dtype
        if noisy_action is None:
            noisy_action = torch.randn(
                (batch, self.pred_horizon, self.action_dim),
                device=device,
                dtype=dtype,
                generator=generator,
            )
        steps = int(inference_steps or self.num_inference_timesteps)
        if steps <= 0:
            raise ValueError("inference_steps must be positive")
        dt = 1.0 / steps
        timestep = torch.tensor([0.0], device=device, dtype=dtype)
        for _ in range(steps):
            velocity = self.predict_velocity(
                state_tokens=state_tokens,
                noisy_action=noisy_action,
                timesteps=timestep,
                lang_tokens=lang_tokens,
                lang_kv_cache=lang_kv_cache,
                lang_attn_mask=lang_attn_mask,
                img_tokens=img_tokens,
            )
            noisy_action = noisy_action + velocity * dt
            timestep = timestep + dt
        return noisy_action

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def config_dict(self) -> dict[str, Any]:
        return asdict(self.config)

    @staticmethod
    def _resolve_state_dict(source: str | Path | dict[str, Tensor]) -> dict[str, Tensor]:
        if isinstance(source, (str, Path)):
            payload = torch.load(source, map_location="cpu", weights_only=False)
        else:
            payload = source
        if (
            isinstance(payload, dict)
            and "module" in payload
            and isinstance(payload["module"], dict)
        ):
            payload = payload["module"]
        if isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
            payload = payload["model"]
        if not isinstance(payload, dict):
            raise TypeError("checkpoint must resolve to a state_dict")
        return {str(key).removeprefix("module."): value for key, value in payload.items()}

    def load_upstream_state_dict(
        self, source: str | Path | dict[str, Tensor], *, strict: bool = True
    ) -> None:
        self.load_state_dict(self._resolve_state_dict(source), strict=strict)

    def load_compatible_upstream_state_dict(
        self, source: str | Path | dict[str, Tensor]
    ) -> dict[str, Any]:
        """Load every released tensor whose key and shape still match.

        This is intended for controlled ablations: local 7-D action heads or a
        DINOv2 condition adaptor can reuse the released Transformer while leaving
        new or shape-incompatible tensors randomly initialized.  It must not be
        described as a strict checkpoint reproduction.
        """
        source_state = self._resolve_state_dict(source)
        target_state = self.state_dict()
        matched: dict[str, Tensor] = {}
        skipped_shape: dict[str, dict[str, list[int]]] = {}
        unexpected: list[str] = []
        for key, value in source_state.items():
            if key not in target_state:
                unexpected.append(key)
                continue
            if tuple(value.shape) != tuple(target_state[key].shape):
                skipped_shape[key] = {
                    "source": list(value.shape),
                    "target": list(target_state[key].shape),
                }
                continue
            matched[key] = value
        missing = sorted(set(target_state) - set(matched))
        self.load_state_dict(matched, strict=False)
        return {
            "matched_tensors": len(matched),
            "source_tensors": len(source_state),
            "target_tensors": len(target_state),
            "missing_target_keys": missing,
            "unexpected_source_keys": sorted(unexpected),
            "shape_mismatches": skipped_shape,
        }


def estimate_rdt2_fm_parameter_count(
    config: RDT2FMReferenceConfig = RDT2FMReferenceConfig(),
) -> int:
    """Analytic parameter count without allocating the released 488M model."""
    config.validate()
    h = config.hidden_size
    heads = config.num_heads
    kv_heads = config.num_kv_heads
    head_dim = h // heads
    ffn_hidden = int(2 * (4 * h) / 3)
    if config.ffn_dim_multiplier is not None:
        ffn_hidden = int(config.ffn_dim_multiplier * ffn_hidden)
    ffn_hidden = config.multiple_of * ((ffn_hidden + config.multiple_of - 1) // config.multiple_of)

    def linear(inp: int, out: int, bias: bool = True) -> int:
        return inp * out + (out if bias else 0)

    # RDT core
    total = 0
    total += linear(256, h) + linear(h, h)  # timestep MLP
    total += config.num_register_tokens * h
    total += (config.prediction_horizon + config.num_register_tokens) * h  # x pos
    total += h  # state pos
    attention = (
        linear(h, h, False)
        + linear(h, 2 * kv_heads * head_dim, False)
        + linear(h, h, False)
        + 2 * head_dim
    )
    block = 0
    block += h + attention  # self-attn norm + attention
    block += h + h + attention  # cross norm + cond norm + cross attention
    block += (
        h
        + linear(h, ffn_hidden, False)
        + linear(ffn_hidden, h, False)
        + linear(h, ffn_hidden, False)
    )
    block += linear(2 * h, 9 * h)
    total += config.depth * block
    total += h  # final RMSNorm
    total += linear(h, 4 * h) + linear(4 * h, config.action_dim)
    total += linear(2 * h, 2 * h)
    # Runner adaptors.
    total += linear(config.action_dim, h) + linear(h, h) + linear(h, h)
    total += linear(config.state_dim, h) + linear(h, h) + linear(h, h)
    if config.lang_adaptor is not None:
        if config.lang_token_dim is None:
            raise ValueError("lang_token_dim required")
        total += _estimate_adapter_count(config.lang_adaptor, config.lang_token_dim, h)
    if config.img_adaptor is not None:
        if config.img_token_dim is None:
            raise ValueError("img_token_dim required")
        total += _estimate_adapter_count(config.img_adaptor, config.img_token_dim, h)
        if config.max_img_len is not None:
            total += config.max_img_len * h
    return int(total)


def _estimate_adapter_count(kind: str, in_features: int, out_features: int) -> int:
    def linear(inp: int, out: int) -> int:
        return inp * out + out

    if kind == "linear":
        return linear(in_features, out_features)
    match = re.match(r"^mlp(\d+)x_silu$", kind)
    if not match:
        raise ValueError(f"unknown adapter type: {kind}")
    depth = int(match.group(1))
    return linear(in_features, out_features) + (depth - 1) * linear(out_features, out_features)


__all__ = [
    "RDT2FMReference",
    "RDT2FMReferenceConfig",
    "RDT",
    "RDTBlock",
    "Attention",
    "CrossAttention",
    "RMSNorm",
    "estimate_rdt2_fm_parameter_count",
]
