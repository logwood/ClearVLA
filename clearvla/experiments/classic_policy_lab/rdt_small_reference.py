from __future__ import annotations

"""Faithful RDT-170M (RDT-small) policy core adapted to ClearVLA HDF5 data.

The transformer and diffusion runner mirror the released RDT implementation:
- unified 128-dimensional state/action space with a validity mask;
- 64-action diffusion horizon;
- timestep and control-frequency tokens;
- 14 transformer blocks with RMSNorm, QK normalization, self-attention,
  alternating language/image cross-attention, and width-preserving GELU FFNs;
- sample-prediction DDPM objective with squared-cosine betas;
- DPM-Solver++ multistep sampling when ``diffusers`` is installed.

The frozen SigLIP encoder and precomputed T5 language embeddings are conditions,
not part of the released 170M policy parameter count.  A deterministic debug
vision encoder is provided only for smoke tests and shape validation.

The implementation is based on the MIT-licensed upstream repository:
https://github.com/thu-ml/RoboticsDiffusionTransformer
"""

import math
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .dp_reference import DDPMScheduler


# -----------------------------------------------------------------------------
# Positional embeddings: equivalent to upstream models/rdt/blocks.py
# -----------------------------------------------------------------------------
def get_1d_sincos_pos_embed_from_grid(
    embed_dim: int, pos: np.ndarray | Tensor | Sequence[int]
) -> np.ndarray:
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even, got {embed_dim}")
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)
    if isinstance(pos, Tensor):
        pos = pos.detach().cpu().numpy()
    if not isinstance(pos, np.ndarray):
        pos = np.asarray(pos, dtype=np.float64)
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_nd_sincos_pos_embed_from_grid(embed_dim: int, grid_sizes: tuple[int, ...]) -> np.ndarray:
    num_valid_sizes = sum(int(value > 1) for value in grid_sizes)
    if num_valid_sizes <= 0:
        return np.zeros(grid_sizes + (embed_dim,), dtype=np.float64)
    emb = np.zeros(grid_sizes + (embed_dim,), dtype=np.float64)
    dim_for_each_grid = embed_dim // num_valid_sizes
    if dim_for_each_grid % 2 != 0:
        dim_for_each_grid -= 1
    valid_size_idx = 0
    for size_idx, grid_size in enumerate(grid_sizes):
        if grid_size <= 1:
            continue
        posemb_shape = [1] * len(grid_sizes) + [dim_for_each_grid]
        posemb_shape[size_idx] = -1
        start = valid_size_idx * dim_for_each_grid
        end = (valid_size_idx + 1) * dim_for_each_grid
        emb[..., start:end] += get_1d_sincos_pos_embed_from_grid(
            dim_for_each_grid, np.arange(grid_size)
        ).reshape(posemb_shape)
        valid_size_idx += 1
    return emb


def get_multimodal_cond_pos_embed(
    embed_dim: int,
    mm_cond_lens: OrderedDict[str, int | tuple[int, ...]],
    *,
    embed_modality: bool = True,
) -> np.ndarray:
    num_modalities = len(mm_cond_lens)
    modality_pos_embed = np.zeros((num_modalities, embed_dim), dtype=np.float64)
    if embed_modality:
        modality_sincos_embed = get_1d_sincos_pos_embed_from_grid(
            embed_dim // 2, np.arange(num_modalities)
        )
        modality_pos_embed[:, : embed_dim // 2] = modality_sincos_embed
        pos_embed_dim = embed_dim // 2
    else:
        pos_embed_dim = embed_dim

    result = np.zeros((0, embed_dim), dtype=np.float64)
    for idx, (modality, cond_len) in enumerate(mm_cond_lens.items()):
        if modality == "image" and isinstance(cond_len, (tuple, list)):
            all_grid_sizes = tuple(abs(int(value)) for value in cond_len)
            embed_grid_sizes = tuple(int(value) if int(value) > 0 else 1 for value in cond_len)
            cond_sincos_embed = get_nd_sincos_pos_embed_from_grid(pos_embed_dim, embed_grid_sizes)
            cond_pos_embed = np.zeros(all_grid_sizes + (embed_dim,), dtype=np.float64)
            cond_pos_embed[..., -pos_embed_dim:] += cond_sincos_embed
            cond_pos_embed = cond_pos_embed.reshape((-1, embed_dim))
        else:
            length = int(cond_len)
            positions = np.arange(length if length > 0 else 1)
            cond_sincos_embed = get_1d_sincos_pos_embed_from_grid(pos_embed_dim, positions)
            cond_pos_embed = np.zeros((abs(length), embed_dim), dtype=np.float64)
            cond_pos_embed[:, -pos_embed_dim:] += cond_sincos_embed
        cond_pos_embed += modality_pos_embed[idx]
        result = np.concatenate([result, cond_pos_embed], axis=0)
    return result


# -----------------------------------------------------------------------------
# Core transformer layers: self-contained equivalents of timm RmsNorm / Mlp /
# Attention used by the upstream repository.
# -----------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: Tensor) -> Tensor:
        # Accumulate in fp32 for stable bf16/fp16 training, then restore dtype.
        dtype = x.dtype
        normed = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed.to(dtype=dtype) * self.weight


class FeedForward(nn.Module):
    """Width-preserving MLP matching timm.models.vision_transformer.Mlp."""

    def __init__(
        self, in_features: int, hidden_features: int | None = None, out_features: int | None = None
    ) -> None:
        super().__init__()
        hidden_features = int(hidden_features or in_features)
        out_features = int(out_features or in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
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
        x = self.drop2(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.attn_drop = nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        batch, tokens, dim = x.shape
        qkv = (
            self.qkv(x)
            .reshape(batch, tokens, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        x = x.transpose(1, 2).reshape(batch, tokens, dim)
        return self.proj_drop(self.proj(x))


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.attn_drop = nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Identity()

    def forward(self, x: Tensor, cond: Tensor, mask: Tensor | None = None) -> Tensor:
        batch, tokens, dim = x.shape
        cond_tokens = cond.shape[1]
        q = self.q(x).reshape(batch, tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = (
            self.kv(cond)
            .reshape(batch, cond_tokens, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        attn_mask = None
        if mask is not None:
            if mask.shape != (batch, cond_tokens):
                raise ValueError(
                    f"condition mask shape {tuple(mask.shape)} != {(batch, cond_tokens)}"
                )
            attn_mask = (
                mask.to(dtype=torch.bool)
                .reshape(batch, 1, 1, cond_tokens)
                .expand(-1, -1, tokens, -1)
            )
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, attn_mask=attn_mask)
        x = x.transpose(1, 2).reshape(batch, tokens, dim)
        return self.proj_drop(self.proj(x))


class RDTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = SelfAttention(hidden_size, num_heads)
        self.cross_attn = CrossAttention(hidden_size, num_heads)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.ffn = FeedForward(hidden_size, hidden_size)
        self.norm3 = RMSNorm(hidden_size, eps=1e-6)

    def forward(self, x: Tensor, cond: Tensor, mask: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), cond, mask)
        x = x + self.ffn(self.norm3(x))
        return x


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = int(frequency_embedding_size)

    def timestep_embedding(self, t: Tensor, dim: int, max_period: int = 10000) -> Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(0, half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t.reshape(-1, 1).float() * freqs.reshape(1, -1)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: Tensor) -> Tensor:
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = RMSNorm(hidden_size, eps=1e-6)
        self.ffn_final = FeedForward(hidden_size, hidden_size, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.ffn_final(self.norm_final(x))


class RDTCore(nn.Module):
    """RDT transformer core with alternating language/image cross-attention."""

    def __init__(
        self,
        *,
        output_dim: int,
        horizon: int,
        hidden_size: int,
        depth: int,
        num_heads: int,
        max_lang_cond_len: int,
        img_cond_len: int,
        lang_pos_embed_config: list[tuple[str, int]] | None,
        img_pos_embed_config: list[tuple[str, tuple[int, ...]]] | None,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.hidden_size = int(hidden_size)
        self.max_lang_cond_len = int(max_lang_cond_len)
        self.img_cond_len = int(img_cond_len)
        self.lang_pos_embed_config = lang_pos_embed_config
        self.img_pos_embed_config = img_pos_embed_config

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.freq_embedder = TimestepEmbedder(hidden_size)
        self.x_pos_embed = nn.Parameter(torch.zeros(1, horizon + 3, hidden_size))
        self.lang_cond_pos_embed = nn.Parameter(torch.zeros(1, max_lang_cond_len, hidden_size))
        self.img_cond_pos_embed = nn.Parameter(torch.zeros(1, img_cond_len, hidden_size))
        self.blocks = nn.ModuleList([RDTBlock(hidden_size, num_heads) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, output_dim)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(basic_init)
        x_pos_embed = get_multimodal_cond_pos_embed(
            self.hidden_size,
            OrderedDict(
                [
                    ("timestep", 1),
                    ("ctrl_freq", 1),
                    ("state", 1),
                    ("action", self.horizon),
                ]
            ),
        )
        self.x_pos_embed.data.copy_(torch.from_numpy(x_pos_embed).float().unsqueeze(0))
        if self.lang_pos_embed_config is None:
            lang = get_1d_sincos_pos_embed_from_grid(
                self.hidden_size, np.arange(self.max_lang_cond_len)
            )
        else:
            lang = get_multimodal_cond_pos_embed(
                self.hidden_size, OrderedDict(self.lang_pos_embed_config), embed_modality=False
            )
        self.lang_cond_pos_embed.data.copy_(torch.from_numpy(lang).float().unsqueeze(0))
        if self.img_pos_embed_config is None:
            img = get_1d_sincos_pos_embed_from_grid(self.hidden_size, np.arange(self.img_cond_len))
        else:
            img = get_multimodal_cond_pos_embed(
                self.hidden_size, OrderedDict(self.img_pos_embed_config), embed_modality=False
            )
        self.img_cond_pos_embed.data.copy_(torch.from_numpy(img).float().unsqueeze(0))
        for embedder in (self.t_embedder, self.freq_embedder):
            nn.init.normal_(embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(embedder.mlp[2].weight, std=0.02)
        nn.init.constant_(self.final_layer.ffn_final.fc2.weight, 0)
        nn.init.constant_(self.final_layer.ffn_final.fc2.bias, 0)

    def forward(
        self,
        x: Tensor,
        freq: Tensor,
        timestep: Tensor,
        lang_cond: Tensor,
        img_cond: Tensor,
        *,
        lang_mask: Tensor | None = None,
        img_mask: Tensor | None = None,
    ) -> Tensor:
        timestep_token = self.t_embedder(timestep.reshape(-1)).unsqueeze(1)
        freq_token = self.freq_embedder(freq.reshape(-1)).unsqueeze(1)
        if timestep_token.shape[0] == 1 and x.shape[0] != 1:
            timestep_token = timestep_token.expand(x.shape[0], -1, -1)
        x = torch.cat([timestep_token, freq_token, x], dim=1)
        if x.shape[1] != self.x_pos_embed.shape[1]:
            raise ValueError(f"x token length {x.shape[1]} != expected {self.x_pos_embed.shape[1]}")
        if img_cond.shape[1] != self.img_cond_pos_embed.shape[1]:
            raise ValueError(
                f"image token length {img_cond.shape[1]} != expected {self.img_cond_pos_embed.shape[1]}"
            )
        if lang_cond.shape[1] > self.lang_cond_pos_embed.shape[1]:
            raise ValueError("language token sequence exceeds max_lang_cond_len")
        x = x + self.x_pos_embed.to(dtype=x.dtype)
        lang_cond = lang_cond + self.lang_cond_pos_embed[:, : lang_cond.shape[1]].to(
            dtype=lang_cond.dtype
        )
        img_cond = img_cond + self.img_cond_pos_embed.to(dtype=img_cond.dtype)
        conditions = (lang_cond, img_cond)
        masks = (lang_mask, img_mask)
        for index, block in enumerate(self.blocks):
            x = block(x, conditions[index % 2], masks[index % 2])
        return self.final_layer(x)[:, -self.horizon :]


# -----------------------------------------------------------------------------
# Released RDT-170M configuration and runner
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class RDTSmallReferenceConfig:
    # Released RDT-170M policy dimensions.
    unified_dim: int = 128
    prediction_horizon: int = 64
    hidden_size: int = 1024
    depth: int = 14
    num_heads: int = 32
    max_lang_cond_len: int = 1024
    lang_token_dim: int = 4096
    img_token_dim: int = 1152
    image_history: int = 2
    max_cameras: int = 3
    patches_per_image: int = 729
    # Released diffusion recipe.
    diffusion_train_steps: int = 1000
    inference_steps: int = 5
    prediction_type: str = "sample"
    # ClearVLA adaptation metadata.
    robot_dim: int = 7
    state_indices: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 10)
    control_frequency: float = 25.0

    @property
    def img_cond_len(self) -> int:
        return self.image_history * self.max_cameras * self.patches_per_image

    def validate(self) -> None:
        if self.prediction_type != "sample":
            raise ValueError("released RDT-170M predicts clean samples, not epsilon")
        if len(self.state_indices) != self.robot_dim:
            raise ValueError("state_indices length must equal robot_dim")
        if len(set(self.state_indices)) != len(self.state_indices):
            raise ValueError("state_indices must be unique")
        if min(self.state_indices) < 0 or max(self.state_indices) >= self.unified_dim:
            raise ValueError("state_indices fall outside unified action space")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

    def to_dict(self) -> dict:
        out = asdict(self)
        out["state_indices"] = list(self.state_indices)
        out["img_cond_len"] = self.img_cond_len
        return out


class UnifiedStateMapper(nn.Module):
    """Embed platform-specific state/action vectors into RDT's 128-D space."""

    def __init__(self, *, robot_dim: int, unified_dim: int, state_indices: Sequence[int]) -> None:
        super().__init__()
        indices = torch.tensor(tuple(int(value) for value in state_indices), dtype=torch.long)
        if len(indices) != robot_dim:
            raise ValueError("state_indices length mismatch")
        if len(set(indices.tolist())) != len(indices):
            raise ValueError("state_indices must be unique")
        if indices.min().item() < 0 or indices.max().item() >= unified_dim:
            raise ValueError("state_indices outside unified vector")
        self.robot_dim = int(robot_dim)
        self.unified_dim = int(unified_dim)
        self.register_buffer("indices", indices, persistent=True)
        mask = torch.zeros(unified_dim, dtype=torch.float32)
        mask[indices] = 1.0
        self.register_buffer("mask", mask, persistent=True)

    def pack(self, values: Tensor) -> Tensor:
        if values.shape[-1] != self.robot_dim:
            raise ValueError(f"robot vector dim {values.shape[-1]} != {self.robot_dim}")
        result = values.new_zeros(*values.shape[:-1], self.unified_dim)
        result[..., self.indices] = values
        return result

    def unpack(self, values: Tensor) -> Tensor:
        if values.shape[-1] != self.unified_dim:
            raise ValueError(f"unified vector dim {values.shape[-1]} != {self.unified_dim}")
        return values[..., self.indices]

    def batch_mask(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        return self.mask.to(device=device, dtype=dtype).reshape(1, 1, -1).expand(batch_size, 1, -1)


class ConditionAdapter(nn.Sequential):
    def __init__(self, in_features: int, out_features: int, depth: int) -> None:
        modules: list[nn.Module] = [nn.Linear(in_features, out_features)]
        for _ in range(1, depth):
            modules.extend([nn.GELU(approximate="tanh"), nn.Linear(out_features, out_features)])
        super().__init__(*modules)


class RDTSmallReference(nn.Module):
    """Released RDT-170M diffusion policy, without external frozen encoders."""

    def __init__(self, config: RDTSmallReferenceConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.mapper = UnifiedStateMapper(
            robot_dim=config.robot_dim,
            unified_dim=config.unified_dim,
            state_indices=config.state_indices,
        )
        self.model = RDTCore(
            output_dim=config.unified_dim,
            horizon=config.prediction_horizon,
            hidden_size=config.hidden_size,
            depth=config.depth,
            num_heads=config.num_heads,
            max_lang_cond_len=config.max_lang_cond_len,
            img_cond_len=config.img_cond_len,
            lang_pos_embed_config=[("lang", -config.max_lang_cond_len)],
            img_pos_embed_config=[
                ("image", (config.image_history, config.max_cameras, -config.patches_per_image))
            ],
        )
        self.lang_adaptor = ConditionAdapter(config.lang_token_dim, config.hidden_size, depth=2)
        self.img_adaptor = ConditionAdapter(config.img_token_dim, config.hidden_size, depth=2)
        self.state_adaptor = ConditionAdapter(config.unified_dim * 2, config.hidden_size, depth=3)
        self.scheduler = DDPMScheduler(
            num_train_timesteps=config.diffusion_train_steps,
            clip_sample=False,
            prediction_type=config.prediction_type,
        )

    def parameter_count(self) -> int:
        """Policy count only: excludes frozen SigLIP/T5 encoders."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def architecture_is_official_170m(self) -> bool:
        cfg = self.config
        return (
            cfg.unified_dim == 128
            and cfg.prediction_horizon == 64
            and cfg.hidden_size == 1024
            and cfg.depth == 14
            and cfg.num_heads == 32
            and cfg.max_lang_cond_len == 1024
            and cfg.lang_token_dim == 4096
            and cfg.img_token_dim == 1152
            and cfg.image_history == 2
            and cfg.max_cameras == 3
            and cfg.patches_per_image == 729
            and cfg.diffusion_train_steps == 1000
            and cfg.prediction_type == "sample"
        )

    def _adapt_conditions(
        self, lang_tokens: Tensor, img_tokens: Tensor, state_tokens: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        return (
            self.lang_adaptor(lang_tokens),
            self.img_adaptor(img_tokens),
            self.state_adaptor(state_tokens),
        )

    def _prepare_inputs(
        self,
        state: Tensor,
        actions: Tensor,
        lang_tokens: Tensor,
        img_tokens: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if state.ndim == 2:
            state = state.unsqueeze(1)
        if state.shape[1] != 1:
            raise ValueError("RDT uses only the latest proprioceptive state token")
        if actions.shape[1:] != (self.config.prediction_horizon, self.config.robot_dim):
            raise ValueError(
                f"actions shape {tuple(actions.shape)} is incompatible with RDT horizon"
            )
        state_unified = self.mapper.pack(state)
        action_unified = self.mapper.pack(actions)
        action_mask = self.mapper.batch_mask(state.shape[0], device=state.device, dtype=state.dtype)
        return (
            state_unified,
            action_unified,
            action_mask,
            self.lang_adaptor(lang_tokens),
            self.img_adaptor(img_tokens),
        )

    def compute_loss(
        self,
        *,
        state: Tensor,
        actions: Tensor,
        lang_tokens: Tensor,
        lang_mask: Tensor,
        img_tokens: Tensor,
        ctrl_freqs: Tensor,
    ) -> Tensor:
        if state.ndim == 2:
            state = state.unsqueeze(1)
        state_unified = self.mapper.pack(state)
        action_unified = self.mapper.pack(actions)
        action_mask = self.mapper.batch_mask(state.shape[0], device=state.device, dtype=state.dtype)
        noise = torch.randn_like(action_unified)
        timesteps = torch.randint(
            0,
            self.config.diffusion_train_steps,
            (state.shape[0],),
            device=state.device,
            dtype=torch.long,
        )
        noisy_action = self.scheduler.add_noise(action_unified, noise, timesteps)
        state_action = torch.cat([state_unified, noisy_action], dim=1)
        expanded_mask = action_mask.expand(-1, state_action.shape[1], -1)
        state_action = torch.cat([state_action, expanded_mask], dim=-1)
        lang_cond, img_cond, state_action = self._adapt_conditions(
            lang_tokens, img_tokens, state_action
        )
        pred = self.model(
            state_action,
            ctrl_freqs,
            timesteps,
            lang_cond,
            img_cond,
            lang_mask=lang_mask,
        )
        return F.mse_loss(pred, action_unified)

    def _sample_ddpm_debug(
        self,
        *,
        lang_cond: Tensor,
        lang_mask: Tensor,
        img_cond: Tensor,
        state_traj: Tensor,
        action_mask: Tensor,
        ctrl_freqs: Tensor,
        inference_steps: int,
        deterministic: bool,
        generator: torch.Generator | None,
    ) -> Tensor:
        noisy_action = torch.randn(
            (state_traj.shape[0], self.config.prediction_horizon, self.config.unified_dim),
            device=state_traj.device,
            dtype=state_traj.dtype,
            generator=generator,
        )
        self.scheduler.set_timesteps(inference_steps)
        expanded_mask = action_mask.expand(-1, self.config.prediction_horizon, -1)
        for timestep in self.scheduler.timesteps.to(state_traj.device):
            action_traj = self.state_adaptor(torch.cat([noisy_action, expanded_mask], dim=-1))
            state_action = torch.cat([state_traj, action_traj], dim=1)
            pred = self.model(
                state_action,
                ctrl_freqs,
                timestep.reshape(1),
                lang_cond,
                img_cond,
                lang_mask=lang_mask,
            )
            noisy_action = self.scheduler.step(
                pred, timestep, noisy_action, deterministic=deterministic, generator=generator
            )
        return noisy_action * expanded_mask

    def _sample_dpm_solver(
        self,
        *,
        lang_cond: Tensor,
        lang_mask: Tensor,
        img_cond: Tensor,
        state_traj: Tensor,
        action_mask: Tensor,
        ctrl_freqs: Tensor,
        inference_steps: int,
        generator: torch.Generator | None,
    ) -> Tensor:
        try:
            from diffusers.schedulers.scheduling_dpmsolver_multistep import (
                DPMSolverMultistepScheduler,
            )
        except ImportError as exc:  # pragma: no cover - optional formal dependency
            raise RuntimeError(
                "formal RDT sampling requires diffusers; install requirements_rdt_small.txt "
                "or use sampler='ddpm_debug' only for smoke tests"
            ) from exc
        scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=self.config.diffusion_train_steps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="sample",
        )
        scheduler.set_timesteps(inference_steps, device=state_traj.device)
        noisy_action = torch.randn(
            (state_traj.shape[0], self.config.prediction_horizon, self.config.unified_dim),
            device=state_traj.device,
            dtype=state_traj.dtype,
            generator=generator,
        )
        expanded_mask = action_mask.expand(-1, self.config.prediction_horizon, -1)
        for timestep in scheduler.timesteps:
            action_traj = self.state_adaptor(torch.cat([noisy_action, expanded_mask], dim=-1))
            state_action = torch.cat([state_traj, action_traj], dim=1)
            pred = self.model(
                state_action,
                ctrl_freqs,
                timestep.reshape(1),
                lang_cond,
                img_cond,
                lang_mask=lang_mask,
            )
            noisy_action = scheduler.step(pred, timestep, noisy_action).prev_sample.to(
                dtype=state_traj.dtype
            )
        return noisy_action * expanded_mask

    @torch.no_grad()
    def predict_action(
        self,
        *,
        state: Tensor,
        lang_tokens: Tensor,
        lang_mask: Tensor,
        img_tokens: Tensor,
        ctrl_freqs: Tensor,
        inference_steps: int | None = None,
        sampler: str = "dpm_solver",
        deterministic: bool = True,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        if state.ndim == 2:
            state = state.unsqueeze(1)
        state_unified = self.mapper.pack(state)
        action_mask = self.mapper.batch_mask(state.shape[0], device=state.device, dtype=state.dtype)
        state_traj = self.state_adaptor(torch.cat([state_unified, action_mask], dim=-1))
        lang_cond = self.lang_adaptor(lang_tokens)
        img_cond = self.img_adaptor(img_tokens)
        steps = int(inference_steps or self.config.inference_steps)
        if sampler == "dpm_solver":
            unified = self._sample_dpm_solver(
                lang_cond=lang_cond,
                lang_mask=lang_mask,
                img_cond=img_cond,
                state_traj=state_traj,
                action_mask=action_mask,
                ctrl_freqs=ctrl_freqs,
                inference_steps=steps,
                generator=generator,
            )
        elif sampler == "ddpm_debug":
            unified = self._sample_ddpm_debug(
                lang_cond=lang_cond,
                lang_mask=lang_mask,
                img_cond=img_cond,
                state_traj=state_traj,
                action_mask=action_mask,
                ctrl_freqs=ctrl_freqs,
                inference_steps=steps,
                deterministic=deterministic,
                generator=generator,
            )
        else:
            raise ValueError(f"unknown RDT sampler: {sampler}")
        return self.mapper.unpack(unified)

    def load_upstream_state_dict(
        self, state_dict: dict[str, Tensor], *, strict: bool = True
    ) -> None:
        """Load released RDTRunner weights, accepting common wrapper prefixes."""
        cleaned: dict[str, Tensor] = {}
        for key, value in state_dict.items():
            for prefix in ("module.", "policy.", "rdt."):
                if key.startswith(prefix):
                    key = key[len(prefix) :]
            cleaned[key] = value
        if strict:
            missing, unexpected = self.load_state_dict(cleaned, strict=False)
            allowed_missing = {"mapper.indices", "mapper.mask"}
            real_missing = [key for key in missing if key not in allowed_missing]
            if real_missing or unexpected:
                raise RuntimeError(
                    f"upstream checkpoint mismatch: missing={real_missing}, unexpected={unexpected}"
                )
        else:
            self.load_state_dict(cleaned, strict=False)


# -----------------------------------------------------------------------------
# Frozen condition encoders
# -----------------------------------------------------------------------------
class EmptyLanguageConditioner:
    """Use one fixed T5-compatible token for a dataset without instructions."""

    def __init__(self, *, token_dim: int = 4096, embedding_path: Path | None = None) -> None:
        if embedding_path is None:
            token = torch.zeros(1, token_dim, dtype=torch.float32)
        else:
            loaded = torch.load(embedding_path, map_location="cpu", weights_only=False)
            if isinstance(loaded, dict):
                loaded = loaded.get("embeddings", loaded.get("embedding", loaded))
            token = torch.as_tensor(loaded, dtype=torch.float32)
            if token.ndim == 1:
                token = token.unsqueeze(0)
            if token.ndim != 2 or token.shape[-1] != token_dim:
                raise ValueError(
                    f"empty language embedding must be [L,{token_dim}], got {tuple(token.shape)}"
                )
        self.token = token.contiguous()

    def batch(
        self, batch_size: int, *, device: torch.device, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor]:
        tokens = self.token.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
        mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=device)
        return tokens, mask


class DebugPatchVisionEncoder(nn.Module):
    """Deterministic shape-only image tokenizer. Not a formal RDT condition encoder."""

    def __init__(
        self, *, token_dim: int, patch_grid: int, image_history: int, max_cameras: int
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.patch_grid = int(patch_grid)
        self.image_history = int(image_history)
        self.max_cameras = int(max_cameras)
        self.patches_per_image = self.patch_grid * self.patch_grid

    def forward(self, images: Tensor) -> Tensor:
        # images [B,History,Cameras,3,H,W], already in [0,1]
        if images.ndim != 6:
            raise ValueError(f"images must be [B,History,Cameras,3,H,W], got {tuple(images.shape)}")
        batch, history, cameras, channels, height, width = images.shape
        if history != self.image_history or cameras > self.max_cameras or channels != 3:
            raise ValueError("debug image condition shape mismatch")
        if cameras < self.max_cameras:
            pad = images.new_full(
                (batch, history, self.max_cameras - cameras, 3, height, width), 0.5
            )
            images = torch.cat([images, pad], dim=2)
        flat = images.reshape(batch * history * self.max_cameras, 3, height, width)
        pooled = F.adaptive_avg_pool2d(flat, (self.patch_grid, self.patch_grid))
        pooled = pooled.flatten(2).transpose(1, 2)
        repeat = math.ceil(self.token_dim / 3)
        tokens = pooled.repeat(1, 1, repeat)[..., : self.token_dim]
        return tokens.reshape(
            batch, history * self.max_cameras * self.patches_per_image, self.token_dim
        )


class SiglipPatchVisionEncoder(nn.Module):
    """Frozen official SigLIP SO400M patch-token encoder used by released RDT."""

    def __init__(
        self,
        *,
        model_name_or_path: str = "google/siglip-so400m-patch14-384",
        image_history: int = 2,
        max_cameras: int = 3,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        try:
            from transformers import SiglipVisionModel
        except ImportError as exc:  # pragma: no cover - optional formal dependency
            raise RuntimeError(
                "formal RDT vision encoding requires transformers; install requirements_rdt_small.txt"
            ) from exc
        self.model = SiglipVisionModel.from_pretrained(
            model_name_or_path, local_files_only=local_files_only
        )
        self.model.eval().requires_grad_(False)
        self.image_history = int(image_history)
        self.max_cameras = int(max_cameras)
        self.image_size = int(getattr(self.model.config, "image_size", 384))
        self.patch_size = int(getattr(self.model.config, "patch_size", 14))
        self.token_dim = int(self.model.config.hidden_size)
        self.patches_per_image = (self.image_size // self.patch_size) ** 2

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    @torch.no_grad()
    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 6:
            raise ValueError(f"images must be [B,History,Cameras,3,H,W], got {tuple(images.shape)}")
        batch, history, cameras, channels, height, width = images.shape
        if history != self.image_history or cameras > self.max_cameras or channels != 3:
            raise ValueError("SigLIP image condition shape mismatch")
        if cameras < self.max_cameras:
            # Upstream pads absent views with a background image at image_mean.
            pad = images.new_full(
                (batch, history, self.max_cameras - cameras, 3, height, width), 0.5
            )
            images = torch.cat([images, pad], dim=2)
        flat = images.reshape(batch * history * self.max_cameras, 3, height, width)
        # Upstream RDT uses SigLIP with image_aspect_ratio="pad". Preserve
        # the same semantics if a caller supplies non-square decoded frames.
        if height != width:
            side = max(height, width)
            square = flat.new_full((flat.shape[0], 3, side, side), 0.5)
            top = (side - height) // 2
            left = (side - width) // 2
            square[:, :, top : top + height, left : left + width] = flat
            flat = square
        # Inputs are cached as [0,1] RGB. SigLIP processor uses resize and mean/std 0.5.
        flat = F.interpolate(
            flat, size=(self.image_size, self.image_size), mode="bicubic", align_corners=False
        )
        flat = (flat - 0.5) / 0.5
        output = self.model(
            pixel_values=flat.to(device=self.device, dtype=self.dtype)
        ).last_hidden_state
        if output.shape[1:] != (self.patches_per_image, self.token_dim):
            raise ValueError(f"unexpected SigLIP output shape {tuple(output.shape)}")
        return output.reshape(
            batch, history * self.max_cameras * self.patches_per_image, self.token_dim
        )


def load_policy_weights(path: Path) -> dict[str, Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        for key in ("model", "module", "state_dict"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
        if all(isinstance(value, Tensor) for value in payload.values()):
            return payload  # upstream pytorch_model.bin
    raise ValueError(f"cannot find policy state dict in {path}")
