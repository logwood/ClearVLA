from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .flow import endpoint_from_velocity


@dataclass(frozen=True)
class VisionUsageLabModelConfig:
    action_dim: int = 7
    chunk_len: int = 25
    past_len: int = 25
    recent_action_len: int = 4
    obs_horizon: int = 2
    camera_names: tuple[str, ...] = ("top", "wrist")
    patch_grid: tuple[int, int] = (16, 16)
    teacher_dim: int = 384
    latent_dim: int = 384
    num_heads: int = 8
    scene_latents: int = 48
    scene_depth: int = 3
    fusion_depth: int = 6
    ffn_hidden: int = 1536
    dense_cross_every: int = 2
    future_visual_horizons: tuple[int, ...] = (1, 4, 8)
    include_visual_delta_tokens: bool = True
    action_history_dropout: float = 0.30
    layerscale_init: float = 1e-3
    dropout: float = 0.0
    local_action_kernel: int = 3
    prefix_len: int = 3
    source_residual_scale: float = 0.50
    history_source_noise_std: float = 0.01

    def validate(self) -> None:
        positive = (
            self.action_dim,
            self.chunk_len,
            self.past_len,
            self.recent_action_len,
            self.obs_horizon,
            self.patch_grid[0],
            self.patch_grid[1],
            self.teacher_dim,
            self.latent_dim,
            self.num_heads,
            self.scene_latents,
            self.scene_depth,
            self.fusion_depth,
            self.ffn_hidden,
            self.dense_cross_every,
            self.local_action_kernel,
            self.prefix_len,
        )
        if any(int(x) <= 0 for x in positive):
            raise ValueError("all dimensions and depths must be positive")
        if self.recent_action_len > self.past_len:
            raise ValueError("recent_action_len cannot exceed past_len")
        if not self.camera_names or len(set(self.camera_names)) != len(self.camera_names):
            raise ValueError(f"invalid camera_names={self.camera_names}")
        if self.latent_dim % self.num_heads:
            raise ValueError("latent_dim must be divisible by num_heads")
        if not 0.0 <= self.action_history_dropout < 1.0:
            raise ValueError("action_history_dropout must be in [0,1)")
        if self.layerscale_init < 0:
            raise ValueError("layerscale_init must be non-negative")
        if self.local_action_kernel % 2 == 0:
            raise ValueError("local_action_kernel must be odd")
        if self.prefix_len > self.chunk_len:
            raise ValueError("prefix_len cannot exceed chunk_len")
        if self.source_residual_scale < 0:
            raise ValueError("source_residual_scale must be non-negative")
        if self.history_source_noise_std < 0:
            raise ValueError("history_source_noise_std must be non-negative")
        if (
            not self.future_visual_horizons
            or tuple(sorted(set(self.future_visual_horizons))) != self.future_visual_horizons
        ):
            raise ValueError("future_visual_horizons must be sorted and unique")
        if any(int(x) <= 0 for x in self.future_visual_horizons):
            raise ValueError("future visual horizons must be positive")

    @property
    def patch_count(self) -> int:
        return int(self.patch_grid[0] * self.patch_grid[1])

    def to_dict(self) -> dict[str, object]:
        out = asdict(self)
        out["camera_names"] = list(self.camera_names)
        out["patch_grid"] = list(self.patch_grid)
        out["future_visual_horizons"] = list(self.future_visual_horizons)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "VisionUsageLabModelConfig":
        payload = dict(data)
        payload["camera_names"] = tuple(
            str(x) for x in payload.get("camera_names", ("top", "wrist"))
        )
        patch = payload.get("patch_grid", (16, 16))
        payload["patch_grid"] = (int(patch[0]), int(patch[1]))  # type: ignore[index]
        payload["future_visual_horizons"] = tuple(
            int(x) for x in payload.get("future_visual_horizons", (1, 4, 8))
        )
        out = cls(**payload)  # type: ignore[arg-type]
        out.validate()
        return out


@dataclass(frozen=True)
class AdaptiveSolverConfig:
    """Deterministic correction-demand router for adaptive flow integration."""

    low_threshold: float = 0.20
    high_threshold: float = 0.50
    low_steps: int = 1
    medium_steps: int = 2
    high_steps: int = 4

    def validate(self) -> None:
        if not 0.0 <= self.low_threshold < self.high_threshold <= 1.0:
            raise ValueError("adaptive solver thresholds must satisfy 0 <= low < high <= 1")
        if min(self.low_steps, self.medium_steps, self.high_steps) <= 0:
            raise ValueError("adaptive solver steps must be positive")
        if not self.low_steps <= self.medium_steps <= self.high_steps:
            raise ValueError("adaptive solver steps must be non-decreasing")

    def select(self, demand_score: torch.Tensor) -> torch.Tensor:
        if demand_score.ndim != 1:
            raise ValueError("demand_score must be [B]")
        self.validate()
        low = torch.full_like(demand_score, int(self.low_steps), dtype=torch.long)
        medium = torch.full_like(demand_score, int(self.medium_steps), dtype=torch.long)
        high = torch.full_like(demand_score, int(self.high_steps), dtype=torch.long)
        return torch.where(
            demand_score < self.low_threshold,
            low,
            torch.where(demand_score < self.high_threshold, medium, high),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        value = x.float()
        value = value * torch.rsqrt(torch.mean(value.square(), dim=-1, keepdim=True) + self.eps)
        return value.to(dtype=dtype) * self.weight.to(dtype=dtype)


@dataclass(frozen=True)
class AttentionKV:
    """Reusable normalized K/V tensors for static cross-attention memory."""

    key: torch.Tensor
    value: torch.Tensor


class QKNormAttention(nn.Module):
    """Multi-head attention with explicit Q/K normalization and reusable K/V."""

    def __init__(self, dim: int, num_heads: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = float(dropout)
        # cosine-attention style learnable scale; clamp bounds numerical range
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def _heads(self, value: torch.Tensor) -> torch.Tensor:
        b, n, _ = value.shape
        return value.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def prepare_key_value(self, key_value: torch.Tensor) -> AttentionKV:
        if key_value.ndim != 3:
            raise ValueError("key_value must be [B,N,D]")
        key = F.normalize(self._heads(self.k_proj(key_value)).float(), dim=-1).to(
            dtype=key_value.dtype
        )
        value = self._heads(self.v_proj(key_value))
        return AttentionKV(key=key, value=value)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor | None = None,
        *,
        prepared_kv: AttentionKV | None = None,
    ) -> torch.Tensor:
        if query.ndim != 3:
            raise ValueError("query must be [B,N,D]")
        if (key_value is None) == (prepared_kv is None):
            raise ValueError("provide exactly one of key_value or prepared_kv")
        kv = self.prepare_key_value(key_value) if key_value is not None else prepared_kv
        assert kv is not None
        if query.shape[0] != kv.key.shape[0]:
            raise ValueError("query and key/value batch sizes differ")
        q = F.normalize(self._heads(self.q_proj(query)).float(), dim=-1).to(dtype=query.dtype)
        scale = self.logit_scale.float().clamp(max=math.log(100.0)).exp().to(dtype=query.dtype)
        out = F.scaled_dot_product_attention(
            q * scale,
            kv.key,
            kv.value,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(query.shape[0], query.shape[1], self.dim)
        return self.out_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.in_proj = nn.Linear(dim, hidden * 2)
        self.out_proj = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        return self.out_proj(self.dropout(value * F.silu(gate)))


class ResidualScale(nn.Module):
    def __init__(self, dim: int, init: float) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.full((dim,), float(init)))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.scale


class AdaLNZero(nn.Module):
    """LayerNorm with zero-initialized shift, scale and residual gate."""

    def __init__(self, dim: int, cond_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.modulation = nn.Linear(cond_dim, dim * 3)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate = self.modulation(cond).chunk(3, dim=-1)
        value = self.norm(x) * (1.0 + scale[:, None]) + shift[:, None]
        return value, gate[:, None]


class TimeCondition(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.mlp = nn.Sequential(nn.Linear(dim + 1, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, time: torch.Tensor, noise_level: torch.Tensor | None = None) -> torch.Tensor:
        if time.ndim != 1:
            raise ValueError("time must be [B]")
        half = self.dim // 2
        frequencies = torch.exp(
            torch.linspace(
                math.log(1.0), math.log(1000.0), half, device=time.device, dtype=time.dtype
            )
        )
        phase = time[:, None] * frequencies[None]
        emb = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        noise = (
            torch.zeros_like(time)
            if noise_level is None
            else noise_level.to(device=time.device, dtype=time.dtype)
        )
        return self.mlp(torch.cat([emb, noise[:, None]], dim=-1))


@dataclass(frozen=True)
class AdaptedVisualTokens:
    """Dense tokens plus explicit per-camera temporal-delta magnitude."""

    dense_tokens: torch.Tensor
    delta_magnitude_by_camera: torch.Tensor


class VisualTokenAdaptor(nn.Module):
    """Adapt frozen teacher patches without erasing temporal-delta magnitude."""

    def __init__(self, config: VisionUsageLabModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        dim = config.latent_dim
        self.observed_norm = nn.LayerNorm(config.teacher_dim)
        self.observed_proj = nn.Sequential(
            nn.Linear(config.teacher_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.delta_direction_proj = nn.Sequential(
            nn.Linear(config.teacher_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.delta_magnitude_proj = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.camera_embed = nn.Parameter(torch.randn(len(config.camera_names), dim) * 0.02)
        self.time_embed = nn.Parameter(torch.randn(config.obs_horizon, dim) * 0.02)
        self.patch_embed = nn.Parameter(torch.randn(config.patch_count, dim) * 0.02)
        self.type_embed = nn.Parameter(
            torch.randn(2, dim) * 0.02
        )  # 0 observed, 1 explicit temporal delta
        self.out_norm = RMSNorm(dim)

    def forward(self, tokens: torch.Tensor) -> AdaptedVisualTokens:
        # [B,H,V,P,C]
        cfg = self.config
        expected = (cfg.obs_horizon, len(cfg.camera_names), cfg.patch_count, cfg.teacher_dim)
        if tokens.ndim != 5 or tuple(tokens.shape[1:]) != expected:
            raise ValueError(f"visual tokens must be [B,{expected}], got {tuple(tokens.shape)}")
        value = self.observed_proj(self.observed_norm(tokens))
        value = value + self.time_embed[None, :, None, None, :]
        value = value + self.camera_embed[None, None, :, None, :]
        value = value + self.patch_embed[None, None, None, :, :]
        value = value + self.type_embed[0][None, None, None, None, :]
        flat = value.reshape(value.shape[0], -1, value.shape[-1])
        delta_camera = torch.zeros(
            (tokens.shape[0], len(cfg.camera_names)),
            device=tokens.device,
            dtype=tokens.dtype,
        )
        if cfg.obs_horizon >= 2:
            delta_raw = tokens[:, -1] - tokens[:, -2]  # [B,V,P,C]
            delta_rms = torch.sqrt(
                torch.mean(delta_raw.float().square(), dim=-1, keepdim=True) + 1e-12
            ).to(dtype=tokens.dtype)
            delta_camera = delta_rms.mean(dim=2).squeeze(-1)
            if cfg.include_visual_delta_tokens:
                delta_direction = delta_raw / delta_rms.clamp_min(1e-6)
                delta = self.delta_direction_proj(delta_direction)
                delta = delta + self.delta_magnitude_proj(torch.log1p(delta_rms))
                delta = delta + self.camera_embed[None, :, None, :]
                delta = delta + self.patch_embed[None, None, :, :]
                delta = delta + self.type_embed[1][None, None, None, :]
                flat = torch.cat([flat, delta.reshape(delta.shape[0], -1, delta.shape[-1])], dim=1)
        return AdaptedVisualTokens(
            dense_tokens=self.out_norm(flat), delta_magnitude_by_camera=delta_camera
        )


class SceneBlock(nn.Module):
    def __init__(
        self, dim: int, heads: int, hidden: int, *, dropout: float, layerscale_init: float
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = QKNormAttention(dim, heads, dropout=dropout)
        self.scale1 = ResidualScale(dim, layerscale_init)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, hidden, dropout=dropout)
        self.scale2 = ResidualScale(dim, layerscale_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.scale1(self.attn(self.norm1(x), self.norm1(x)))
        x = x + self.scale2(self.ffn(self.norm2(x)))
        return x


class SceneEncoder(nn.Module):
    """Perceiver-style scene bottleneck retaining access to all dense teacher patches."""

    def __init__(self, config: VisionUsageLabModelConfig) -> None:
        super().__init__()
        dim = config.latent_dim
        self.latents = nn.Parameter(torch.randn(config.scene_latents, dim) * 0.02)
        self.query_norm = RMSNorm(dim)
        self.visual_norm = RMSNorm(dim)
        self.cross = QKNormAttention(dim, config.num_heads, dropout=config.dropout)
        self.cross_scale = ResidualScale(dim, config.layerscale_init)
        self.blocks = nn.ModuleList(
            [
                SceneBlock(
                    dim,
                    config.num_heads,
                    config.ffn_hidden,
                    dropout=config.dropout,
                    layerscale_init=config.layerscale_init,
                )
                for _ in range(config.scene_depth)
            ]
        )
        self.out_norm = RMSNorm(dim)

    def forward(self, dense_visual: torch.Tensor) -> torch.Tensor:
        batch = dense_visual.shape[0]
        scene = self.latents[None].expand(batch, -1, -1)
        scene = scene + self.cross_scale(
            self.cross(self.query_norm(scene), self.visual_norm(dense_visual))
        )
        for block in self.blocks:
            scene = block(scene)
        return self.out_norm(scene)


class WeakActionAnchorEncoder(nn.Module):
    """Intentionally shallow action-history anchor to prevent shortcut domination."""

    def __init__(self, config: VisionUsageLabModelConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.latent_dim
        self.action_proj = nn.Linear(config.action_dim, dim)
        self.velocity_proj = nn.Linear(config.action_dim, dim)
        self.token_embed = nn.Parameter(torch.randn(config.recent_action_len + 1, dim) * 0.02)
        self.norm = RMSNorm(dim)

    def forward(self, past: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        if past.ndim != 3 or past.shape[1:] != (cfg.past_len, cfg.action_dim):
            raise ValueError(
                f"past must be [B,{cfg.past_len},{cfg.action_dim}], got {tuple(past.shape)}"
            )
        recent = past[:, -cfg.recent_action_len :]
        if self.training and cfg.action_history_dropout > 0:
            keep = (
                torch.rand((past.shape[0], 1, 1), device=past.device) >= cfg.action_history_dropout
            ).to(recent.dtype)
            recent = recent * keep
        if past.shape[1] >= 2:
            velocity = past[:, -1] - past[:, -2]
        else:
            velocity = torch.zeros_like(past[:, -1])
        tokens = torch.cat([self.action_proj(recent), self.velocity_proj(velocity)[:, None]], dim=1)
        return self.norm(tokens + self.token_embed[None])


class TemporalLocalMixer(nn.Module):
    """Cheap local temporal inductive bias for low-dimensional action sequences."""

    def __init__(self, dim: int, kernel: int) -> None:
        super().__init__()
        if kernel <= 0 or kernel % 2 == 0:
            raise ValueError("temporal kernel must be positive and odd")
        self.norm = RMSNorm(dim)
        self.depthwise = nn.Conv1d(dim, dim, kernel_size=kernel, padding=kernel // 2, groups=dim)
        self.pointwise = nn.Conv1d(dim, dim * 2, kernel_size=1)
        self.out = nn.Linear(dim, dim)
        self.scale = ResidualScale(dim, 1e-3)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        value = self.norm(tokens).transpose(1, 2)
        value, gate = self.pointwise(self.depthwise(value)).chunk(2, dim=1)
        update = self.out((value * F.silu(gate)).transpose(1, 2))
        return tokens + self.scale(update)


class LearnedHistorySource(nn.Module):
    """A2A-inspired source: physical prior plus a learned history residual."""

    def __init__(self, config: VisionUsageLabModelConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.latent_dim
        self.action_proj = nn.Linear(config.action_dim, dim)
        self.past_pos = nn.Parameter(torch.randn(config.past_len, dim) * 0.02)
        self.horizon_query = nn.Parameter(torch.randn(config.chunk_len, dim) * 0.02)
        self.local = TemporalLocalMixer(dim, config.local_action_kernel)
        self.query_norm = RMSNorm(dim)
        self.history_norm = RMSNorm(dim)
        self.cross = QKNormAttention(dim, config.num_heads, dropout=config.dropout)
        self.out_norm = RMSNorm(dim)
        self.residual_head = nn.Linear(dim, config.action_dim)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(
        self, past: torch.Tensor, physical_prior: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        if tuple(past.shape[1:]) != (cfg.past_len, cfg.action_dim):
            raise ValueError(f"past must be [B,{cfg.past_len},{cfg.action_dim}]")
        if tuple(physical_prior.shape[1:]) != (cfg.chunk_len, cfg.action_dim):
            raise ValueError(f"physical_prior must be [B,{cfg.chunk_len},{cfg.action_dim}]")
        history_input = past
        if self.training and cfg.history_source_noise_std > 0:
            history_input = (
                history_input + torch.randn_like(history_input) * cfg.history_source_noise_std
            )
        history = self.local(self.action_proj(history_input) + self.past_pos[None])
        query = self.horizon_query[None].expand(past.shape[0], -1, -1)
        query = query + self.cross(self.query_norm(query), self.history_norm(history))
        residual = torch.tanh(self.residual_head(self.out_norm(query))) * cfg.source_residual_scale
        return physical_prior + residual, query


class HistoryBaseField(nn.Module):
    """Visual-free local corrective field around the learned history source."""

    def __init__(self, config: VisionUsageLabModelConfig) -> None:
        super().__init__()
        dim = config.latent_dim
        self.in_proj = nn.Linear(config.action_dim * 3, dim)
        self.cond = AdaLNZero(dim, dim)
        self.local = TemporalLocalMixer(dim, config.local_action_kernel)
        self.out_norm = RMSNorm(dim)
        self.out = nn.Linear(dim, config.action_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self, state: torch.Tensor, source: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        token = self.in_proj(torch.cat([state, source, state - source], dim=-1))
        value, gate = self.cond(token, cond)
        token = token + torch.tanh(gate) * value
        return self.out(self.out_norm(self.local(token)))


class VisualCorrectionGate(nn.Module):
    """Gate visual correction using scene, history, deviation and delta magnitude."""

    def __init__(self, config: VisionUsageLabModelConfig) -> None:
        super().__init__()
        dim = config.latent_dim
        self.scene_norm = RMSNorm(dim)
        self.anchor_norm = RMSNorm(dim)
        self.deviation_proj = nn.Linear(config.action_dim, dim)
        self.delta_proj = nn.Sequential(
            nn.Linear(len(config.camera_names), dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.mlp = nn.Sequential(nn.Linear(dim * 4, dim), nn.SiLU(), nn.Linear(dim, 1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias, -2.0)

    def forward(
        self,
        scene: torch.Tensor,
        anchor: torch.Tensor,
        state: torch.Tensor,
        source: torch.Tensor,
        delta_magnitude_by_camera: torch.Tensor,
    ) -> torch.Tensor:
        if state.shape != source.shape:
            raise ValueError("state and source must share shape")
        if (
            delta_magnitude_by_camera.ndim != 2
            or delta_magnitude_by_camera.shape[0] != state.shape[0]
        ):
            raise ValueError("delta_magnitude_by_camera must be [B,V]")
        horizon = state.shape[1]
        scene_feature = self.scene_norm(scene).mean(dim=1)[:, None].expand(-1, horizon, -1)
        anchor_feature = self.anchor_norm(anchor).mean(dim=1)[:, None].expand(-1, horizon, -1)
        deviation_feature = self.deviation_proj(state - source)
        delta_feature = self.delta_proj(
            torch.log1p(delta_magnitude_by_camera.float()).to(dtype=scene.dtype)
        )[:, None].expand(-1, horizon, -1)
        feature = torch.cat(
            [scene_feature, anchor_feature, deviation_feature, delta_feature], dim=-1
        )
        return torch.sigmoid(self.mlp(feature).squeeze(-1))


class FastPrefixHead(nn.Module):
    """Immediate prefix path; it never depends on deep workspace features."""

    def __init__(self, config: VisionUsageLabModelConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.latent_dim
        self.scene_norm = RMSNorm(dim)
        self.source_norm = RMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.SiLU(),
            nn.Linear(dim, config.prefix_len * config.action_dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self, scene: torch.Tensor, source_tokens: torch.Tensor, source: torch.Tensor
    ) -> torch.Tensor:
        cfg = self.config
        feature = torch.cat(
            [self.scene_norm(scene).mean(dim=1), self.source_norm(source_tokens).mean(dim=1)],
            dim=-1,
        )
        residual = self.mlp(feature).reshape(source.shape[0], cfg.prefix_len, cfg.action_dim)
        return source[:, : cfg.prefix_len] + residual


class StreamingTailHead(nn.Module):
    """SFP-inspired executable tail path conditioned on the previous action."""

    def __init__(self, config: VisionUsageLabModelConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.latent_dim
        self.prev_proj = nn.Linear(config.action_dim, dim)
        self.source_proj = nn.Linear(config.action_dim, dim)
        self.scene_norm = RMSNorm(dim)
        self.step_embed = nn.Parameter(torch.randn(config.chunk_len, dim) * 0.02)
        self.mlp = nn.Sequential(
            nn.Linear(dim * 3, dim), nn.SiLU(), nn.Linear(dim, config.action_dim)
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def teacher_forced(
        self, past: torch.Tensor, future: torch.Tensor, source: torch.Tensor, scene: torch.Tensor
    ) -> torch.Tensor:
        prev = torch.cat([past[:, -1:], future[:, :-1]], dim=1)
        return self._predict(prev, source, scene)

    def _predict(
        self, prev: torch.Tensor, source: torch.Tensor, scene: torch.Tensor
    ) -> torch.Tensor:
        scene_feature = self.scene_norm(scene).mean(dim=1)[:, None].expand(-1, source.shape[1], -1)
        token = torch.cat(
            [self.prev_proj(prev), self.source_proj(source) + self.step_embed[None], scene_feature],
            dim=-1,
        )
        return source + self.mlp(token)

    def rollout(
        self, past: torch.Tensor, source: torch.Tensor, scene: torch.Tensor, prefix: torch.Tensor
    ) -> torch.Tensor:
        cfg = self.config
        actions = [prefix[:, index] for index in range(cfg.prefix_len)]
        for index in range(cfg.prefix_len, cfg.chunk_len):
            prev = actions[-1]
            scene_feature = self.scene_norm(scene).mean(dim=1)
            token = torch.cat(
                [
                    self.prev_proj(prev),
                    self.source_proj(source[:, index]) + self.step_embed[index][None],
                    scene_feature,
                ],
                dim=-1,
            )
            actions.append(source[:, index] + self.mlp(token))
        return torch.stack(actions, dim=1)


class DynamicsPriorEncoder(nn.Module):
    """Encode clean prior and recent velocity for the visual dynamics objective.

    This path is intentionally independent of the bridge state.  The bridge
    state contains a mixture of source and ground-truth future actions during
    flow training and would leak target information into the auxiliary task.
    """

    def __init__(self, config: VisionUsageLabModelConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.latent_dim
        self.action_proj = nn.Linear(config.action_dim, dim)
        self.velocity_proj = nn.Linear(config.action_dim, dim)
        self.token_embed = nn.Parameter(torch.randn(config.chunk_len + 1, dim) * 0.02)
        self.norm = RMSNorm(dim)

    def forward(self, past: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        if past.ndim != 3 or tuple(past.shape[1:]) != (cfg.past_len, cfg.action_dim):
            raise ValueError(
                f"past must be [B,{cfg.past_len},{cfg.action_dim}], got {tuple(past.shape)}"
            )
        if prior.ndim != 3 or tuple(prior.shape[1:]) != (cfg.chunk_len, cfg.action_dim):
            raise ValueError(
                f"prior must be [B,{cfg.chunk_len},{cfg.action_dim}], got {tuple(prior.shape)}"
            )
        velocity = (
            past[:, -1] - past[:, -2] if past.shape[1] >= 2 else torch.zeros_like(past[:, -1])
        )
        tokens = torch.cat([self.action_proj(prior), self.velocity_proj(velocity)[:, None]], dim=1)
        return self.norm(tokens + self.token_embed[None])


@dataclass(frozen=True)
class WorkspaceMemory:
    scene_kv: AttentionKV
    anchor_kv: AttentionKV
    dense_kv: AttentionKV | None


class ActionWorkspaceBlock(nn.Module):
    def __init__(self, config: VisionUsageLabModelConfig, *, use_dense_visual: bool) -> None:
        super().__init__()
        dim = config.latent_dim
        self.use_dense_visual = bool(use_dense_visual)
        self.self_norm = AdaLNZero(dim, dim)
        self.self_attn = QKNormAttention(dim, config.num_heads, dropout=config.dropout)
        self.scene_norm = AdaLNZero(dim, dim)
        self.scene_attn = QKNormAttention(dim, config.num_heads, dropout=config.dropout)
        self.anchor_norm = AdaLNZero(dim, dim)
        self.anchor_attn = QKNormAttention(dim, config.num_heads, dropout=config.dropout)
        if use_dense_visual:
            self.dense_norm = AdaLNZero(dim, dim)
            self.dense_attn = QKNormAttention(dim, config.num_heads, dropout=config.dropout)
        else:
            self.dense_norm = None
            self.dense_attn = None
        self.ffn_norm = AdaLNZero(dim, dim)
        self.ffn = SwiGLU(dim, config.ffn_hidden, dropout=config.dropout)

    @staticmethod
    def _add_gated(x: torch.Tensor, update: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return x + torch.tanh(gate) * update

    def prepare_memory(
        self, *, scene: torch.Tensor, dense_visual: torch.Tensor, anchor: torch.Tensor
    ) -> WorkspaceMemory:
        dense_kv = None
        if self.dense_attn is not None:
            dense_kv = self.dense_attn.prepare_key_value(dense_visual)
        return WorkspaceMemory(
            scene_kv=self.scene_attn.prepare_key_value(scene),
            anchor_kv=self.anchor_attn.prepare_key_value(anchor),
            dense_kv=dense_kv,
        )

    def forward(
        self,
        action: torch.Tensor,
        *,
        cond: torch.Tensor,
        memory: WorkspaceMemory | None = None,
        scene: torch.Tensor | None = None,
        dense_visual: torch.Tensor | None = None,
        anchor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if memory is None:
            if scene is None or dense_visual is None or anchor is None:
                raise ValueError(
                    "scene, dense_visual and anchor are required when memory is not prepared"
                )
            memory = self.prepare_memory(scene=scene, dense_visual=dense_visual, anchor=anchor)
        value, gate = self.self_norm(action, cond)
        action = self._add_gated(action, self.self_attn(value, value), gate)
        value, gate = self.scene_norm(action, cond)
        action = self._add_gated(action, self.scene_attn(value, prepared_kv=memory.scene_kv), gate)
        value, gate = self.anchor_norm(action, cond)
        action = self._add_gated(
            action, self.anchor_attn(value, prepared_kv=memory.anchor_kv), gate
        )
        if self.use_dense_visual:
            assert (
                self.dense_norm is not None
                and self.dense_attn is not None
                and memory.dense_kv is not None
            )
            value, gate = self.dense_norm(action, cond)
            action = self._add_gated(
                action, self.dense_attn(value, prepared_kv=memory.dense_kv), gate
            )
        value, gate = self.ffn_norm(action, cond)
        action = self._add_gated(action, self.ffn(value), gate)
        return action


class FuturePatchDynamicsHead(nn.Module):
    """Predict future teacher-latent deltas for every camera patch and horizon."""

    def __init__(self, config: VisionUsageLabModelConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.latent_dim
        f = len(config.future_visual_horizons)
        v = len(config.camera_names)
        p = config.patch_count
        self.horizon_embed = nn.Parameter(torch.randn(f, dim) * 0.02)
        self.camera_embed = nn.Parameter(torch.randn(v, dim) * 0.02)
        self.patch_embed = nn.Parameter(torch.randn(p, dim) * 0.02)
        self.query_norm = RMSNorm(dim)
        self.memory_norm = RMSNorm(dim)
        self.cross = QKNormAttention(dim, config.num_heads, dropout=config.dropout)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, config.ffn_hidden, dropout=config.dropout)
        self.out_norm = RMSNorm(dim)
        self.out_proj = nn.Linear(dim, config.teacher_dim)

    def forward(self, scene: torch.Tensor, condition_tokens: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        f, v, p = len(cfg.future_visual_horizons), len(cfg.camera_names), cfg.patch_count
        query = (
            self.horizon_embed[:, None, None]
            + self.camera_embed[None, :, None]
            + self.patch_embed[None, None]
        )
        query = query.reshape(1, f * v * p, cfg.latent_dim).expand(scene.shape[0], -1, -1)
        memory = torch.cat([scene, condition_tokens], dim=1)
        query = query + self.cross(self.query_norm(query), self.memory_norm(memory))
        query = query + self.ffn(self.ffn_norm(query))
        return self.out_proj(self.out_norm(query)).reshape(scene.shape[0], f, v, p, cfg.teacher_dim)


class CorrectionDemandHead(nn.Module):
    """Estimate how strongly the history-anchored source needs correction."""

    def __init__(self, config: VisionUsageLabModelConfig) -> None:
        super().__init__()
        dim = config.latent_dim
        self.scene_norm = RMSNorm(dim)
        self.prior_norm = RMSNorm(dim)
        self.delta_proj = nn.Sequential(
            nn.Linear(len(config.camera_names), dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.mlp = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.SiLU(),
            nn.Linear(dim, 1),
        )

    def forward(
        self,
        scene: torch.Tensor,
        prior_tokens: torch.Tensor,
        delta_magnitude_by_camera: torch.Tensor,
    ) -> torch.Tensor:
        if scene.ndim != 3 or prior_tokens.ndim != 3 or scene.shape[0] != prior_tokens.shape[0]:
            raise ValueError("scene and prior_tokens must be [B,N,D] with matching B")
        if (
            delta_magnitude_by_camera.ndim != 2
            or delta_magnitude_by_camera.shape[0] != scene.shape[0]
        ):
            raise ValueError("delta_magnitude_by_camera must be [B,V]")
        delta_feature = self.delta_proj(
            torch.log1p(delta_magnitude_by_camera.float()).to(dtype=scene.dtype)
        )
        feature = torch.cat(
            [
                self.scene_norm(scene).mean(dim=1),
                self.prior_norm(prior_tokens).mean(dim=1),
                delta_feature,
            ],
            dim=-1,
        )
        return self.mlp(feature).squeeze(-1)


@dataclass
class PreparedVisualState:
    """Stable visual memory computed once and reused across action-flow steps."""

    dense_visual_tokens: torch.Tensor
    scene_tokens: torch.Tensor
    event_logit: torch.Tensor
    delta_magnitude_by_camera: torch.Tensor


@dataclass(frozen=True)
class PreparedFlowMemory:
    """Static action-flow cross-attention memory prepared once per observation."""

    anchor_tokens: torch.Tensor
    workspace_memories: tuple[WorkspaceMemory, ...]


@dataclass(frozen=True)
class PreparedFastPath:
    """Low-latency result available before deep workspace K/V preparation."""

    visual: PreparedVisualState
    source_trajectory: torch.Tensor
    source_tokens: torch.Tensor
    fast_prefix: torch.Tensor


@dataclass(frozen=True)
class PreparedFlowCondition:
    """Fast-path outputs plus reusable deep-workspace memory."""

    fast_path: PreparedFastPath
    flow_memory: PreparedFlowMemory


@dataclass
class VisionUsageLabOutput:
    velocity: torch.Tensor | None
    endpoint: torch.Tensor | None
    visual_delta_tokens: torch.Tensor | None
    event_logit: torch.Tensor
    demand_logit: torch.Tensor | None
    demand_score: torch.Tensor | None
    action_tokens: torch.Tensor | None
    learned_source: torch.Tensor
    fast_prefix: torch.Tensor | None
    streaming_actions: torch.Tensor | None
    streaming_teacher_forced_actions: torch.Tensor | None
    base_velocity: torch.Tensor | None
    visual_velocity: torch.Tensor | None
    visual_gate: torch.Tensor | None
    scene_tokens: torch.Tensor
    dense_visual_tokens: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


@dataclass
class AdaptiveIntegrationOutput:
    prediction: torch.Tensor
    demand_score: torch.Tensor
    solver_steps: torch.Tensor


class VisionUsageLabModel(nn.Module):
    """Structured visual-latent policy laboratory.

    Capacity is attached to explicit responsibilities:
      * pretrained patch adaptation and scene-state construction;
      * action-informed flow refinement;
      * future teacher-latent forecasting;
      * event prediction for direct visual supervision.
    """

    def __init__(self, config: VisionUsageLabModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        dim = config.latent_dim
        self.visual_adaptor = VisualTokenAdaptor(config)
        self.scene_encoder = SceneEncoder(config)
        self.action_anchor = WeakActionAnchorEncoder(config)
        self.history_source = LearnedHistorySource(config)
        self.dynamics_prior = DynamicsPriorEncoder(config)
        self.history_field = HistoryBaseField(config)
        self.action_in = nn.Linear(config.action_dim, dim)
        self.action_pos = nn.Parameter(torch.randn(config.chunk_len, dim) * 0.02)
        self.condition = TimeCondition(dim)
        self.fusion_blocks = nn.ModuleList(
            [
                ActionWorkspaceBlock(
                    config, use_dense_visual=((idx + 1) % config.dense_cross_every == 0)
                )
                for idx in range(config.fusion_depth)
            ]
        )
        self.action_out_norm = RMSNorm(dim)
        self.visual_velocity_head = nn.Linear(dim, config.action_dim)
        nn.init.zeros_(self.visual_velocity_head.weight)
        nn.init.zeros_(self.visual_velocity_head.bias)
        self.visual_gate_head = VisualCorrectionGate(config)
        self.fast_prefix_head = FastPrefixHead(config)
        self.streaming_tail_head = StreamingTailHead(config)
        self.dynamics_head = FuturePatchDynamicsHead(config)
        self.event_head = nn.Sequential(
            RMSNorm(dim), nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1)
        )
        self.demand_head = CorrectionDemandHead(config)

    def prepare_visual(self, visual_tokens: torch.Tensor) -> PreparedVisualState:
        """Build stable dense and scene memories once per observation window."""
        adapted = self.visual_adaptor(visual_tokens)
        scene = self.scene_encoder(adapted.dense_tokens)
        event_logit = self.event_head(scene.mean(dim=1)).squeeze(-1)
        return PreparedVisualState(
            dense_visual_tokens=adapted.dense_tokens,
            scene_tokens=scene,
            event_logit=event_logit,
            delta_magnitude_by_camera=adapted.delta_magnitude_by_camera,
        )

    def predict_source(
        self, past: torch.Tensor, physical_prior: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.history_source(past, physical_prior)

    def prepare_flow_memory(
        self, *, past: torch.Tensor, prepared_visual: PreparedVisualState
    ) -> PreparedFlowMemory:
        """Project static cross-attention memories once for repeated integration."""
        anchor = self.action_anchor(past)
        memories = tuple(
            block.prepare_memory(
                scene=prepared_visual.scene_tokens,
                dense_visual=prepared_visual.dense_visual_tokens,
                anchor=anchor,
            )
            for block in self.fusion_blocks
        )
        return PreparedFlowMemory(anchor_tokens=anchor, workspace_memories=memories)

    def prepare_fast_path(
        self, *, past: torch.Tensor, prior: torch.Tensor, visual_tokens: torch.Tensor
    ) -> PreparedFastPath:
        """Return source and executable prefix without constructing workspace K/V."""
        prepared_visual = self.prepare_visual(visual_tokens)
        source, source_tokens = self.predict_source(past, prior)
        prefix = self.fast_prefix_head(prepared_visual.scene_tokens, source_tokens, source)
        return PreparedFastPath(
            visual=prepared_visual,
            source_trajectory=source,
            source_tokens=source_tokens,
            fast_prefix=prefix,
        )

    @torch.no_grad()
    def predict_fast_prefix(
        self, *, past: torch.Tensor, prior: torch.Tensor, visual_tokens: torch.Tensor
    ) -> PreparedFastPath:
        """Low-latency prefix-only API. Deep workspace K/V is intentionally skipped."""
        return self.prepare_fast_path(past=past, prior=prior, visual_tokens=visual_tokens)

    def complete_flow_condition(
        self, *, past: torch.Tensor, fast_path: PreparedFastPath
    ) -> PreparedFlowCondition:
        """Materialize reusable deep-workspace memory after a prefix is available."""
        return PreparedFlowCondition(
            fast_path=fast_path,
            flow_memory=self.prepare_flow_memory(past=past, prepared_visual=fast_path.visual),
        )

    def predict_fast_paths_prepared(
        self,
        *,
        past: torch.Tensor,
        source: torch.Tensor,
        source_tokens: torch.Tensor,
        prepared_visual: PreparedVisualState,
        future: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        prefix = self.fast_prefix_head(prepared_visual.scene_tokens, source_tokens, source)
        streaming_rollout = self.streaming_tail_head.rollout(
            past, source, prepared_visual.scene_tokens, prefix
        )
        streaming_teacher_forced = None
        if future is not None:
            streaming_teacher_forced = self.streaming_tail_head.teacher_forced(
                past,
                future,
                source,
                prepared_visual.scene_tokens,
            )
        return prefix, streaming_rollout, streaming_teacher_forced

    def predict_velocity_prepared(
        self,
        *,
        past: torch.Tensor,
        source: torch.Tensor,
        prepared_visual: PreparedVisualState,
        action_state: torch.Tensor,
        bridge_time: torch.Tensor,
        noise_level: torch.Tensor | None = None,
        prepared_flow: PreparedFlowMemory | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        cfg = self.config
        if tuple(action_state.shape[1:]) != (cfg.chunk_len, cfg.action_dim):
            raise ValueError(f"action_state must be [B,{cfg.chunk_len},{cfg.action_dim}]")
        if source.shape != action_state.shape:
            raise ValueError("source and action_state must share shape")
        scene = prepared_visual.scene_tokens
        flow_memory = (
            self.prepare_flow_memory(past=past, prepared_visual=prepared_visual)
            if prepared_flow is None
            else prepared_flow
        )
        anchor = flow_memory.anchor_tokens
        cond = self.condition(bridge_time, noise_level)
        base_velocity = self.history_field(action_state, source, cond)
        action = self.action_in(action_state - source) + self.action_pos[None]
        for block, memory in zip(self.fusion_blocks, flow_memory.workspace_memories, strict=True):
            action = block(action, cond=cond, memory=memory)
        visual_velocity = self.visual_velocity_head(self.action_out_norm(action))
        visual_gate = self.visual_gate_head(
            scene,
            anchor,
            action_state,
            source,
            prepared_visual.delta_magnitude_by_camera,
        )
        velocity = base_velocity + visual_gate[:, :, None] * visual_velocity
        endpoint = endpoint_from_velocity(action_state, velocity, bridge_time)
        diagnostics = {
            "velocity_norm": torch.linalg.vector_norm(velocity, dim=-1).mean(),
            "base_velocity_norm": torch.linalg.vector_norm(base_velocity, dim=-1).mean(),
            "visual_velocity_norm": torch.linalg.vector_norm(visual_velocity, dim=-1).mean(),
            "visual_gate_mean": visual_gate.mean(),
            "endpoint_update_norm": torch.linalg.vector_norm(
                endpoint - action_state, dim=-1
            ).mean(),
            "action_workspace_norm": torch.linalg.vector_norm(action, dim=-1).mean(),
        }
        return velocity, endpoint, action, base_velocity, visual_velocity, visual_gate, diagnostics

    def predict_demand_prepared(
        self,
        *,
        past: torch.Tensor,
        prior: torch.Tensor,
        prepared_visual: PreparedVisualState,
        prior_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = self.dynamics_prior(past, prior) if prior_tokens is None else prior_tokens
        logit = self.demand_head(
            prepared_visual.scene_tokens,
            tokens,
            prepared_visual.delta_magnitude_by_camera,
        )
        return logit, torch.sigmoid(logit), tokens

    def predict_auxiliary_prepared(
        self,
        *,
        past: torch.Tensor,
        prior: torch.Tensor,
        prepared_visual: PreparedVisualState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        demand_logit, demand_score, condition_tokens = self.predict_demand_prepared(
            past=past,
            prior=prior,
            prepared_visual=prepared_visual,
        )
        visual_delta = self.dynamics_head(prepared_visual.scene_tokens, condition_tokens)
        diagnostics = {
            "scene_norm": torch.linalg.vector_norm(prepared_visual.scene_tokens, dim=-1).mean(),
            "dense_visual_norm": torch.linalg.vector_norm(
                prepared_visual.dense_visual_tokens, dim=-1
            ).mean(),
            "dynamics_condition_norm": torch.linalg.vector_norm(condition_tokens, dim=-1).mean(),
            "demand_score_mean": demand_score.mean(),
            "demand_score_std": demand_score.std(unbiased=False),
            "visual_delta_magnitude_mean": prepared_visual.delta_magnitude_by_camera.mean(),
        }
        return visual_delta, demand_logit, demand_score, diagnostics

    def forward_prepared(
        self,
        *,
        past: torch.Tensor,
        prior: torch.Tensor,
        prepared_visual: PreparedVisualState,
        action_state: torch.Tensor | None = None,
        bridge_time: torch.Tensor | None = None,
        noise_level: torch.Tensor | None = None,
        future_actions: torch.Tensor | None = None,
        source_trajectory: torch.Tensor | None = None,
        source_tokens: torch.Tensor | None = None,
        prepared_flow: PreparedFlowMemory | None = None,
        compute_action: bool = True,
        compute_auxiliary: bool = True,
        compute_fast_paths: bool = True,
    ) -> VisionUsageLabOutput:
        velocity: torch.Tensor | None = None
        endpoint: torch.Tensor | None = None
        action_tokens: torch.Tensor | None = None
        visual_delta: torch.Tensor | None = None
        demand_logit: torch.Tensor | None = None
        demand_score: torch.Tensor | None = None
        diagnostics: dict[str, torch.Tensor] = {}
        if source_trajectory is None:
            learned_source, learned_source_tokens = self.predict_source(past, prior)
        else:
            learned_source = source_trajectory
            learned_source_tokens = (
                source_tokens if source_tokens is not None else self.history_source(past, prior)[1]
            )
        fast_prefix: torch.Tensor | None = None
        streaming_actions: torch.Tensor | None = None
        streaming_teacher_forced_actions: torch.Tensor | None = None
        if compute_fast_paths:
            fast_prefix, streaming_actions, streaming_teacher_forced_actions = (
                self.predict_fast_paths_prepared(
                    past=past,
                    source=learned_source,
                    source_tokens=learned_source_tokens,
                    prepared_visual=prepared_visual,
                    future=future_actions,
                )
            )
        base_velocity: torch.Tensor | None = None
        visual_velocity: torch.Tensor | None = None
        visual_gate: torch.Tensor | None = None
        diagnostics["source_residual_norm"] = torch.linalg.vector_norm(
            learned_source - prior, dim=-1
        ).mean()
        if compute_action:
            if action_state is None or bridge_time is None:
                raise ValueError(
                    "action_state and bridge_time are required when compute_action=True"
                )
            if prior.shape != action_state.shape:
                raise ValueError("prior and action_state must share shape")
            (
                velocity,
                endpoint,
                action_tokens,
                base_velocity,
                visual_velocity,
                visual_gate,
                action_diag,
            ) = self.predict_velocity_prepared(
                past=past,
                source=learned_source,
                prepared_visual=prepared_visual,
                action_state=action_state,
                bridge_time=bridge_time,
                noise_level=noise_level,
                prepared_flow=prepared_flow,
            )
            diagnostics.update(action_diag)
        if compute_auxiliary:
            visual_delta, demand_logit, demand_score, auxiliary_diag = (
                self.predict_auxiliary_prepared(
                    past=past,
                    prior=learned_source,
                    prepared_visual=prepared_visual,
                )
            )
            diagnostics.update(auxiliary_diag)
        return VisionUsageLabOutput(
            velocity=velocity,
            endpoint=endpoint,
            visual_delta_tokens=visual_delta,
            event_logit=prepared_visual.event_logit,
            demand_logit=demand_logit,
            demand_score=demand_score,
            action_tokens=action_tokens,
            learned_source=learned_source,
            fast_prefix=fast_prefix,
            streaming_actions=streaming_actions,
            streaming_teacher_forced_actions=streaming_teacher_forced_actions,
            base_velocity=base_velocity,
            visual_velocity=visual_velocity,
            visual_gate=visual_gate,
            scene_tokens=prepared_visual.scene_tokens,
            dense_visual_tokens=prepared_visual.dense_visual_tokens,
            diagnostics=diagnostics,
        )

    def forward(
        self,
        *,
        past: torch.Tensor,
        prior: torch.Tensor,
        visual_tokens: torch.Tensor,
        action_state: torch.Tensor | None = None,
        bridge_time: torch.Tensor | None = None,
        noise_level: torch.Tensor | None = None,
        future_actions: torch.Tensor | None = None,
        source_trajectory: torch.Tensor | None = None,
        source_tokens: torch.Tensor | None = None,
        prepared_flow: PreparedFlowMemory | None = None,
        compute_action: bool = True,
        compute_auxiliary: bool = True,
        compute_fast_paths: bool = True,
    ) -> VisionUsageLabOutput:
        prepared = self.prepare_visual(visual_tokens)
        return self.forward_prepared(
            past=past,
            prior=prior,
            prepared_visual=prepared,
            action_state=action_state,
            bridge_time=bridge_time,
            noise_level=noise_level,
            future_actions=future_actions,
            source_trajectory=source_trajectory,
            source_tokens=source_tokens,
            prepared_flow=prepared_flow,
            compute_action=compute_action,
            compute_auxiliary=compute_auxiliary,
            compute_fast_paths=compute_fast_paths,
        )

    @staticmethod
    def _slice_prepared(prepared: PreparedVisualState, index: torch.Tensor) -> PreparedVisualState:
        return PreparedVisualState(
            dense_visual_tokens=prepared.dense_visual_tokens[index],
            scene_tokens=prepared.scene_tokens[index],
            event_logit=prepared.event_logit[index],
            delta_magnitude_by_camera=prepared.delta_magnitude_by_camera[index],
        )

    @staticmethod
    def _slice_attention_kv(prepared: AttentionKV, index: torch.Tensor) -> AttentionKV:
        return AttentionKV(key=prepared.key[index], value=prepared.value[index])

    @classmethod
    def _slice_flow_memory(
        cls, prepared: PreparedFlowMemory, index: torch.Tensor
    ) -> PreparedFlowMemory:
        return PreparedFlowMemory(
            anchor_tokens=prepared.anchor_tokens[index],
            workspace_memories=tuple(
                WorkspaceMemory(
                    scene_kv=cls._slice_attention_kv(memory.scene_kv, index),
                    anchor_kv=cls._slice_attention_kv(memory.anchor_kv, index),
                    dense_kv=None
                    if memory.dense_kv is None
                    else cls._slice_attention_kv(memory.dense_kv, index),
                )
                for memory in prepared.workspace_memories
            ),
        )

    @torch.no_grad()
    def integrate_prepared(
        self,
        *,
        past: torch.Tensor,
        prior: torch.Tensor,
        prepared_visual: PreparedVisualState,
        steps: int = 4,
        source_trajectory: torch.Tensor | None = None,
        prepared_flow: PreparedFlowMemory | None = None,
    ) -> torch.Tensor:
        if steps <= 0:
            raise ValueError("steps must be positive")
        source = (
            self.predict_source(past, prior)[0] if source_trajectory is None else source_trajectory
        )
        flow_memory = (
            self.prepare_flow_memory(past=past, prepared_visual=prepared_visual)
            if prepared_flow is None
            else prepared_flow
        )
        state = source
        batch = prior.shape[0]
        noise = torch.zeros((batch,), device=prior.device, dtype=prior.dtype)
        dt = 1.0 / float(steps)
        for index in range(steps):
            time = torch.full(
                (batch,), float(index) / float(steps), device=prior.device, dtype=prior.dtype
            )
            velocity, _, _, _, _, _, _ = self.predict_velocity_prepared(
                past=past,
                source=source,
                prepared_visual=prepared_visual,
                action_state=state,
                bridge_time=time,
                noise_level=noise,
                prepared_flow=flow_memory,
            )
            state = state + dt * velocity
        return state

    @torch.no_grad()
    def integrate_flow_condition_prepared(
        self,
        *,
        past: torch.Tensor,
        prior: torch.Tensor,
        condition: PreparedFlowCondition,
        steps: int = 4,
    ) -> torch.Tensor:
        """Complete a prefix-first request using reusable workspace K/V."""
        return self.integrate_prepared(
            past=past,
            prior=prior,
            prepared_visual=condition.fast_path.visual,
            steps=steps,
            source_trajectory=condition.fast_path.source_trajectory,
            prepared_flow=condition.flow_memory,
        )

    @torch.no_grad()
    def integrate(
        self,
        *,
        past: torch.Tensor,
        prior: torch.Tensor,
        visual_tokens: torch.Tensor,
        steps: int = 4,
    ) -> torch.Tensor:
        prepared = self.prepare_visual(visual_tokens)
        return self.integrate_prepared(
            past=past, prior=prior, prepared_visual=prepared, steps=steps
        )

    @torch.no_grad()
    def stream_prefix_tail_prepared(
        self,
        *,
        past: torch.Tensor,
        prior: torch.Tensor,
        prepared_visual: PreparedVisualState,
    ) -> torch.Tensor:
        source, source_tokens = self.predict_source(past, prior)
        prefix = self.fast_prefix_head(prepared_visual.scene_tokens, source_tokens, source)
        return self.streaming_tail_head.rollout(past, source, prepared_visual.scene_tokens, prefix)

    @torch.no_grad()
    def stream_prefix_tail(
        self, *, past: torch.Tensor, prior: torch.Tensor, visual_tokens: torch.Tensor
    ) -> torch.Tensor:
        prepared = self.prepare_visual(visual_tokens)
        return self.stream_prefix_tail_prepared(past=past, prior=prior, prepared_visual=prepared)

    @torch.no_grad()
    def integrate_adaptive_prepared(
        self,
        *,
        past: torch.Tensor,
        prior: torch.Tensor,
        prepared_visual: PreparedVisualState,
        solver: AdaptiveSolverConfig = AdaptiveSolverConfig(),
        source_trajectory: torch.Tensor | None = None,
        prepared_flow: PreparedFlowMemory | None = None,
    ) -> AdaptiveIntegrationOutput:
        """Route prepared samples to deterministic correction depths.

        Samples are grouped by selected solver depth so low-demand windows skip
        unnecessary workspace evaluations instead of merely masking them.
        """
        solver.validate()
        source = (
            self.predict_source(past, prior)[0] if source_trajectory is None else source_trajectory
        )
        _, demand_score, _ = self.predict_demand_prepared(
            past=past,
            prior=source,
            prepared_visual=prepared_visual,
        )
        steps = solver.select(demand_score)
        flow_memory = (
            self.prepare_flow_memory(past=past, prepared_visual=prepared_visual)
            if prepared_flow is None
            else prepared_flow
        )
        prediction = torch.empty_like(prior)
        for count in torch.unique(steps, sorted=True).tolist():
            mask = steps == int(count)
            index = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            prediction[index] = self.integrate_prepared(
                past=past[index],
                prior=prior[index],
                prepared_visual=self._slice_prepared(prepared_visual, index),
                steps=int(count),
                source_trajectory=source[index],
                prepared_flow=self._slice_flow_memory(flow_memory, index),
            )
        return AdaptiveIntegrationOutput(
            prediction=prediction,
            demand_score=demand_score,
            solver_steps=steps,
        )

    @torch.no_grad()
    def integrate_adaptive(
        self,
        *,
        past: torch.Tensor,
        prior: torch.Tensor,
        visual_tokens: torch.Tensor,
        solver: AdaptiveSolverConfig = AdaptiveSolverConfig(),
    ) -> AdaptiveIntegrationOutput:
        prepared = self.prepare_visual(visual_tokens)
        return self.integrate_adaptive_prepared(
            past=past,
            prior=prior,
            prepared_visual=prepared,
            solver=solver,
        )
