from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from clearvla.experiments.vision_usage_lab.model import AttentionKV, QKNormAttention, RMSNorm, ResidualScale, SwiGLU
from .flow import endpoint_from_velocity


@dataclass(frozen=True)
class ResidualFlowLabModelConfig:
    """Compact history-anchored residual-flow policy.

    The default network intentionally keeps only the mechanisms needed to test
    the core hypothesis: a frozen learned history source provides the default
    chunk and a visual conditional flow predicts the missing residual.
    """

    action_dim: int = 7
    chunk_len: int = 25
    past_len: int = 25
    obs_horizon: int = 2
    camera_names: tuple[str, ...] = ("top", "wrist")
    patch_grid: tuple[int, int] = (16, 16)
    teacher_dim: int = 384
    latent_dim: int = 384
    num_heads: int = 8
    scene_latents: int = 24
    scene_depth: int = 1
    flow_depth: int = 3
    ffn_hidden: int = 1024
    local_action_kernel: int = 3
    include_visual_delta_tokens: bool = True
    independent_camera_dropout: float = 0.15
    layerscale_init: float = 1e-3
    dropout: float = 0.0
    source_residual_scale: float = 0.50
    history_source_noise_std: float = 0.01
    delta_topk: int = 16

    def validate(self) -> None:
        positive = (
            self.action_dim,
            self.chunk_len,
            self.past_len,
            self.obs_horizon,
            self.patch_grid[0],
            self.patch_grid[1],
            self.teacher_dim,
            self.latent_dim,
            self.num_heads,
            self.scene_latents,
            self.scene_depth,
            self.flow_depth,
            self.ffn_hidden,
            self.local_action_kernel,
            self.delta_topk,
        )
        if any(int(value) <= 0 for value in positive):
            raise ValueError("all dimensions and depths must be positive")
        if not self.camera_names or len(set(self.camera_names)) != len(self.camera_names):
            raise ValueError(f"invalid camera_names={self.camera_names}")
        if self.latent_dim % self.num_heads:
            raise ValueError("latent_dim must be divisible by num_heads")
        if self.local_action_kernel % 2 == 0:
            raise ValueError("local_action_kernel must be odd")
        if not 0.0 <= self.independent_camera_dropout < 1.0:
            raise ValueError("independent_camera_dropout must be in [0,1)")
        if self.layerscale_init < 0 or self.source_residual_scale < 0 or self.history_source_noise_std < 0:
            raise ValueError("scale values must be non-negative")
        if self.delta_topk > self.patch_count:
            raise ValueError("delta_topk cannot exceed patch_count")

    @property
    def patch_count(self) -> int:
        return int(self.patch_grid[0] * self.patch_grid[1])

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["camera_names"] = list(self.camera_names)
        payload["patch_grid"] = list(self.patch_grid)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ResidualFlowLabModelConfig":
        payload = dict(data)
        payload["camera_names"] = tuple(str(x) for x in payload.get("camera_names", ("top", "wrist")))
        patch = payload.get("patch_grid", (16, 16))
        payload["patch_grid"] = (int(patch[0]), int(patch[1]))  # type: ignore[index]
        out = cls(**payload)  # type: ignore[arg-type]
        out.validate()
        return out


class TemporalLocalMixer(nn.Module):
    def __init__(self, dim: int, kernel: int, *, scale_init: float = 1e-3) -> None:
        super().__init__()
        if kernel <= 0 or kernel % 2 == 0:
            raise ValueError("kernel must be positive and odd")
        self.norm = RMSNorm(dim)
        self.depthwise = nn.Conv1d(dim, dim, kernel_size=kernel, padding=kernel // 2, groups=dim)
        self.pointwise = nn.Conv1d(dim, dim * 2, kernel_size=1)
        self.out = nn.Linear(dim, dim)
        self.scale = ResidualScale(dim, scale_init)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        value = self.norm(tokens).transpose(1, 2)
        value, gate = self.pointwise(self.depthwise(value)).chunk(2, dim=1)
        update = self.out((value * F.silu(gate)).transpose(1, 2))
        return tokens + self.scale(update)


class LearnedHistorySource(nn.Module):
    """Physical prior plus a learned history-only residual.

    This module is optimized during source pretraining and frozen during the
    residual-flow phase so the corrective policy cannot move its own anchor.
    """

    def __init__(self, config: ResidualFlowLabModelConfig) -> None:
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

    def forward(self, past: torch.Tensor, physical_prior: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        if tuple(past.shape[1:]) != (cfg.past_len, cfg.action_dim):
            raise ValueError(f"past must be [B,{cfg.past_len},{cfg.action_dim}]")
        if tuple(physical_prior.shape[1:]) != (cfg.chunk_len, cfg.action_dim):
            raise ValueError(f"physical_prior must be [B,{cfg.chunk_len},{cfg.action_dim}]")
        history_input = past
        if self.training and cfg.history_source_noise_std > 0:
            history_input = history_input + torch.randn_like(history_input) * cfg.history_source_noise_std
        history = self.local(self.action_proj(history_input) + self.past_pos[None])
        query = self.horizon_query[None].expand(past.shape[0], -1, -1)
        query = query + self.cross(self.query_norm(query), self.history_norm(history))
        residual = torch.tanh(self.residual_head(self.out_norm(query))) * cfg.source_residual_scale
        return physical_prior + residual, query


@dataclass(frozen=True)
class AdaptedVisualTokens:
    dense_tokens: torch.Tensor
    camera_tokens: tuple[torch.Tensor, ...]
    delta_statistics_by_camera: torch.Tensor  # [B,V,4]: mean/max/topk/q90
    camera_keep_mask: torch.Tensor             # [B,V]


class VisualTokenAdaptor(nn.Module):
    """Adapt frozen patch tokens and preserve control-relevant temporal evidence.

    Independent camera masking follows the same shortcut-prevention principle
    used by RDT-style multimodal training: a broad exterior view must not make
    the wrist camera irrelevant.  Delta directions are magnitude-gated so tiny
    latent jitter cannot be promoted to unit-scale evidence.
    """

    def __init__(self, config: ResidualFlowLabModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        dim = config.latent_dim
        self.observed_norm = nn.LayerNorm(config.teacher_dim)
        self.observed_proj = nn.Sequential(nn.Linear(config.teacher_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.delta_direction_proj = nn.Sequential(nn.Linear(config.teacher_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.delta_magnitude_proj = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.delta_direction_gate = nn.Linear(1, 1)
        nn.init.zeros_(self.delta_direction_gate.weight)
        nn.init.constant_(self.delta_direction_gate.bias, -2.0)
        cameras = len(config.camera_names)
        self.camera_embed = nn.Parameter(torch.randn(cameras, dim) * 0.02)
        self.camera_mask_token = nn.Parameter(torch.randn(cameras, dim) * 0.02)
        self.time_embed = nn.Parameter(torch.randn(config.obs_horizon, dim) * 0.02)
        self.patch_embed = nn.Parameter(torch.randn(config.patch_count, dim) * 0.02)
        self.type_embed = nn.Parameter(torch.randn(2, dim) * 0.02)
        self.out_norm = RMSNorm(dim)

    def _sample_camera_keep(self, batch: int, *, device: torch.device) -> torch.Tensor:
        cfg = self.config
        cameras = len(cfg.camera_names)
        keep = torch.ones((batch, cameras), dtype=torch.bool, device=device)
        if self.training and cfg.independent_camera_dropout > 0:
            keep = torch.rand((batch, cameras), device=device) >= cfg.independent_camera_dropout
            missing = ~keep.any(dim=1)
            if bool(missing.any()):
                row = torch.nonzero(missing, as_tuple=False).squeeze(-1)
                chosen = torch.randint(0, cameras, (row.shape[0],), device=device)
                keep[row, chosen] = True
        return keep

    def forward(self, tokens: torch.Tensor) -> AdaptedVisualTokens:
        cfg = self.config
        expected = (cfg.obs_horizon, len(cfg.camera_names), cfg.patch_count, cfg.teacher_dim)
        if tokens.ndim != 5 or tuple(tokens.shape[1:]) != expected:
            raise ValueError(f"visual tokens must be [B,{expected}], got {tuple(tokens.shape)}")
        batch = tokens.shape[0]
        keep = self._sample_camera_keep(batch, device=tokens.device)
        content = self.observed_proj(self.observed_norm(tokens))
        mask_content = self.camera_mask_token[None, None, :, None, :]
        content = torch.where(keep[:, None, :, None, None], content, mask_content)
        observed = content
        observed = observed + self.time_embed[None, :, None, None, :]
        observed = observed + self.camera_embed[None, None, :, None, :]
        observed = observed + self.patch_embed[None, None, None, :, :]
        observed = observed + self.type_embed[0][None, None, None, None, :]

        per_camera: list[torch.Tensor] = []
        for index in range(len(cfg.camera_names)):
            per_camera.append(observed[:, :, index].reshape(batch, -1, cfg.latent_dim))

        statistics = torch.zeros((batch, len(cfg.camera_names), 4), device=tokens.device, dtype=tokens.dtype)
        if cfg.obs_horizon >= 2:
            delta_raw = tokens[:, -1] - tokens[:, -2]  # [B,V,P,C]
            delta_raw = delta_raw * keep[:, :, None, None].to(dtype=delta_raw.dtype)
            delta_rms = torch.sqrt(torch.mean(delta_raw.float().square(), dim=-1, keepdim=True) + 1e-12).to(dtype=tokens.dtype)
            flat_rms = delta_rms.squeeze(-1)
            topk = min(cfg.delta_topk, cfg.patch_count)
            statistics = torch.stack(
                [
                    flat_rms.mean(dim=-1),
                    flat_rms.max(dim=-1).values,
                    flat_rms.topk(topk, dim=-1).values.mean(dim=-1),
                    torch.quantile(flat_rms.float(), 0.90, dim=-1).to(dtype=tokens.dtype),
                ],
                dim=-1,
            )
            if cfg.include_visual_delta_tokens:
                direction = delta_raw / delta_rms.clamp_min(1e-6)
                magnitude = torch.log1p(delta_rms)
                gate = torch.sigmoid(self.delta_direction_gate(magnitude))
                delta = gate * self.delta_direction_proj(direction) + self.delta_magnitude_proj(magnitude)
                delta = delta + self.camera_embed[None, :, None, :]
                delta = delta + self.patch_embed[None, None, :, :]
                delta = delta + self.type_embed[1][None, None, None, :]
                for index in range(len(cfg.camera_names)):
                    per_camera[index] = torch.cat([per_camera[index], delta[:, index]], dim=1)

        camera_tokens = tuple(self.out_norm(value) for value in per_camera)
        dense = self.out_norm(torch.cat(camera_tokens, dim=1))
        return AdaptedVisualTokens(
            dense_tokens=dense,
            camera_tokens=camera_tokens,
            delta_statistics_by_camera=statistics,
            camera_keep_mask=keep,
        )


class SceneBlock(nn.Module):
    def __init__(self, dim: int, heads: int, hidden: int, *, dropout: float, layerscale_init: float) -> None:
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
    def __init__(self, config: ResidualFlowLabModelConfig) -> None:
        super().__init__()
        dim = config.latent_dim
        self.latents = nn.Parameter(torch.randn(config.scene_latents, dim) * 0.02)
        self.query_norm = RMSNorm(dim)
        self.visual_norm = RMSNorm(dim)
        self.cross = QKNormAttention(dim, config.num_heads, dropout=config.dropout)
        self.cross_scale = ResidualScale(dim, config.layerscale_init)
        self.blocks = nn.ModuleList(
            [
                SceneBlock(dim, config.num_heads, config.ffn_hidden, dropout=config.dropout, layerscale_init=config.layerscale_init)
                for _ in range(config.scene_depth)
            ]
        )
        self.out_norm = RMSNorm(dim)

    def forward(self, dense_visual: torch.Tensor) -> torch.Tensor:
        scene = self.latents[None].expand(dense_visual.shape[0], -1, -1)
        scene = scene + self.cross_scale(self.cross(self.query_norm(scene), self.visual_norm(dense_visual)))
        for block in self.blocks:
            scene = block(scene)
        return self.out_norm(scene)


class HistoryTrajectoryEncoder(nn.Module):
    """Small temporal encoder for action, velocity and acceleration history."""

    def __init__(self, config: ResidualFlowLabModelConfig) -> None:
        super().__init__()
        dim = config.latent_dim
        self.config = config
        self.in_proj = nn.Linear(config.action_dim * 3, dim)
        self.pos = nn.Parameter(torch.randn(config.past_len, dim) * 0.02)
        self.local = TemporalLocalMixer(dim, config.local_action_kernel)
        self.norm = RMSNorm(dim)
        self.attn = QKNormAttention(dim, config.num_heads, dropout=config.dropout)
        self.scale = ResidualScale(dim, config.layerscale_init)
        self.out_norm = RMSNorm(dim)

    def forward(self, past: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        if tuple(past.shape[1:]) != (cfg.past_len, cfg.action_dim):
            raise ValueError(f"past must be [B,{cfg.past_len},{cfg.action_dim}]")
        velocity = torch.diff(past, dim=1, prepend=past[:, :1])
        acceleration = torch.diff(velocity, dim=1, prepend=velocity[:, :1])
        token = self.local(self.in_proj(torch.cat([past, velocity, acceleration], dim=-1)) + self.pos[None])
        token = token + self.scale(self.attn(self.norm(token), self.norm(token)))
        token = self.out_norm(token)
        return token, token.mean(dim=1)


class FlowConditionEncoder(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.scalar_proj = nn.Sequential(nn.Linear(3, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.mlp = nn.Sequential(nn.Linear(dim * 3, dim), nn.SiLU(), nn.Linear(dim, dim))

    def _time_embedding(self, time: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        frequencies = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), half, device=time.device, dtype=time.dtype))
        phase = time[:, None] * frequencies[None]
        value = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        if value.shape[-1] < self.dim:
            value = F.pad(value, (0, self.dim - value.shape[-1]))
        return value

    def forward(
        self,
        *,
        time: torch.Tensor,
        step_size: torch.Tensor,
        noise_level: torch.Tensor,
        history_summary: torch.Tensor,
        source_summary: torch.Tensor,
    ) -> torch.Tensor:
        if time.ndim != 1 or step_size.ndim != 1 or noise_level.ndim != 1:
            raise ValueError("time, step_size and noise_level must be [B]")
        scalars = self.scalar_proj(torch.stack([time, step_size, noise_level], dim=-1))
        return self.mlp(torch.cat([self._time_embedding(time) + scalars, history_summary, source_summary], dim=-1))


@dataclass(frozen=True)
class FlowMemory:
    scene_kv: AttentionKV
    camera_kv: AttentionKV | None


class ResidualTrajectoryBlock(nn.Module):
    """Residual trajectory mixer with compact scene and alternating camera reads."""

    def __init__(self, config: ResidualFlowLabModelConfig, *, camera_index: int | None) -> None:
        super().__init__()
        dim = config.latent_dim
        self.camera_index = camera_index
        self.local = TemporalLocalMixer(dim, config.local_action_kernel)
        self.self_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.self_attn = QKNormAttention(dim, config.num_heads, dropout=config.dropout)
        self.scene_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.scene_attn = QKNormAttention(dim, config.num_heads, dropout=config.dropout)
        self.camera_norm = nn.LayerNorm(dim, elementwise_affine=False) if camera_index is not None else None
        self.camera_attn = QKNormAttention(dim, config.num_heads, dropout=config.dropout) if camera_index is not None else None
        self.ffn_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.ffn = SwiGLU(dim, config.ffn_hidden, dropout=config.dropout)
        self.modulation = nn.Linear(dim, dim * 4)
        nn.init.zeros_(self.modulation.weight)
        nn.init.constant_(self.modulation.bias, 0.02)

    def prepare_memory(self, *, scene: torch.Tensor, camera_tokens: tuple[torch.Tensor, ...]) -> FlowMemory:
        camera_kv = None
        if self.camera_attn is not None:
            assert self.camera_index is not None
            camera_kv = self.camera_attn.prepare_key_value(camera_tokens[self.camera_index])
        return FlowMemory(scene_kv=self.scene_attn.prepare_key_value(scene), camera_kv=camera_kv)

    def forward(self, action: torch.Tensor, *, cond: torch.Tensor, memory: FlowMemory) -> torch.Tensor:
        gate_self, gate_scene, gate_camera, gate_ffn = torch.tanh(self.modulation(cond)).chunk(4, dim=-1)
        action = self.local(action)
        action = action + gate_self[:, None] * self.self_attn(self.self_norm(action), self.self_norm(action))
        action = action + gate_scene[:, None] * self.scene_attn(self.scene_norm(action), prepared_kv=memory.scene_kv)
        if self.camera_attn is not None:
            assert self.camera_norm is not None and memory.camera_kv is not None
            action = action + gate_camera[:, None] * self.camera_attn(self.camera_norm(action), prepared_kv=memory.camera_kv)
        action = action + gate_ffn[:, None] * self.ffn(self.ffn_norm(action))
        return action


class NonlinearResidualVelocityDecoder(nn.Module):
    def __init__(self, dim: int, action_dim: int) -> None:
        super().__init__()
        self.norm = RMSNorm(dim)
        self.skip = nn.Linear(dim, action_dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 128), nn.SiLU(), nn.Linear(128, action_dim))
        nn.init.normal_(self.skip.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.skip.bias)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        value = self.norm(token)
        return self.skip(value) + self.mlp(value)


@dataclass(frozen=True)
class PreparedVisualState:
    dense_tokens: torch.Tensor
    camera_tokens: tuple[torch.Tensor, ...]
    scene_tokens: torch.Tensor
    delta_statistics_by_camera: torch.Tensor
    camera_keep_mask: torch.Tensor


@dataclass(frozen=True)
class PreparedFlowMemory:
    block_memories: tuple[FlowMemory, ...]


@dataclass
class ResidualFlowLabOutput:
    residual_velocity: torch.Tensor
    endpoint_residual: torch.Tensor
    endpoint_actions: torch.Tensor
    learned_source: torch.Tensor
    trajectory_tokens: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


class ResidualFlowLabModel(nn.Module):
    """History-anchored residual-flow policy with direct action supervision."""

    def __init__(self, config: ResidualFlowLabModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        dim = config.latent_dim
        self.visual_adaptor = VisualTokenAdaptor(config)
        self.scene_encoder = SceneEncoder(config)
        self.history_source = LearnedHistorySource(config)
        self.history_encoder = HistoryTrajectoryEncoder(config)
        self.source_proj = nn.Linear(config.action_dim, dim)
        self.source_pos = nn.Parameter(torch.randn(config.chunk_len, dim) * 0.02)
        self.source_norm = RMSNorm(dim)
        self.condition = FlowConditionEncoder(dim)
        self.residual_in = nn.Linear(config.action_dim, dim)
        self.residual_pos = nn.Parameter(torch.randn(config.chunk_len, dim) * 0.02)
        cameras = len(config.camera_names)
        schedule: list[int | None] = []
        for index in range(config.flow_depth):
            if index == 0:
                schedule.append(None)
            else:
                # Wrist/local view is read first when cameras are (top, wrist).
                schedule.append((cameras - (index % cameras)) % cameras)
        self.camera_schedule = tuple(schedule)
        self.blocks = nn.ModuleList([ResidualTrajectoryBlock(config, camera_index=index) for index in self.camera_schedule])
        self.decoder = NonlinearResidualVelocityDecoder(dim, config.action_dim)

    def prepare_visual(self, visual_tokens: torch.Tensor) -> PreparedVisualState:
        adapted = self.visual_adaptor(visual_tokens)
        scene = self.scene_encoder(adapted.dense_tokens)
        return PreparedVisualState(
            dense_tokens=adapted.dense_tokens,
            camera_tokens=adapted.camera_tokens,
            scene_tokens=scene,
            delta_statistics_by_camera=adapted.delta_statistics_by_camera,
            camera_keep_mask=adapted.camera_keep_mask,
        )

    def predict_source(self, past: torch.Tensor, prior: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.history_source(past, prior)

    def prepare_flow_memory(self, prepared_visual: PreparedVisualState) -> PreparedFlowMemory:
        return PreparedFlowMemory(
            block_memories=tuple(
                block.prepare_memory(scene=prepared_visual.scene_tokens, camera_tokens=prepared_visual.camera_tokens)
                for block in self.blocks
            )
        )

    def predict_residual_velocity_prepared(
        self,
        *,
        past: torch.Tensor,
        learned_source: torch.Tensor,
        prepared_visual: PreparedVisualState,
        residual_state: torch.Tensor,
        bridge_time: torch.Tensor,
        step_size: torch.Tensor | None = None,
        noise_level: torch.Tensor | None = None,
        prepared_flow: PreparedFlowMemory | None = None,
    ) -> ResidualFlowLabOutput:
        cfg = self.config
        if tuple(residual_state.shape[1:]) != (cfg.chunk_len, cfg.action_dim):
            raise ValueError(f"residual_state must be [B,{cfg.chunk_len},{cfg.action_dim}]")
        if learned_source.shape != residual_state.shape:
            raise ValueError("learned_source and residual_state must share shape")
        batch = residual_state.shape[0]
        if step_size is None:
            step_size = torch.zeros((batch,), device=residual_state.device, dtype=residual_state.dtype)
        if noise_level is None:
            noise_level = torch.zeros((batch,), device=residual_state.device, dtype=residual_state.dtype)
        history_tokens, history_summary = self.history_encoder(past)
        source_tokens = self.source_norm(self.source_proj(learned_source) + self.source_pos[None])
        source_summary = source_tokens.mean(dim=1)
        cond = self.condition(
            time=bridge_time,
            step_size=step_size,
            noise_level=noise_level,
            history_summary=history_summary,
            source_summary=source_summary,
        )
        token = self.residual_in(residual_state) + self.residual_pos[None]
        memory = self.prepare_flow_memory(prepared_visual) if prepared_flow is None else prepared_flow
        for block, block_memory in zip(self.blocks, memory.block_memories, strict=True):
            token = block(token, cond=cond, memory=block_memory)
        velocity = self.decoder(token)
        endpoint_residual = endpoint_from_velocity(residual_state, velocity, bridge_time)
        endpoint_actions = learned_source + endpoint_residual
        diagnostics = {
            "source_norm": torch.linalg.vector_norm(learned_source, dim=-1).mean(),
            "residual_state_norm": torch.linalg.vector_norm(residual_state, dim=-1).mean(),
            "residual_velocity_norm": torch.linalg.vector_norm(velocity, dim=-1).mean(),
            "endpoint_residual_norm": torch.linalg.vector_norm(endpoint_residual, dim=-1).mean(),
            "trajectory_token_norm": torch.linalg.vector_norm(token, dim=-1).mean(),
            "delta_statistics_mean": prepared_visual.delta_statistics_by_camera.mean(),
            "camera_keep_rate": prepared_visual.camera_keep_mask.float().mean(),
        }
        return ResidualFlowLabOutput(
            residual_velocity=velocity,
            endpoint_residual=endpoint_residual,
            endpoint_actions=endpoint_actions,
            learned_source=learned_source,
            trajectory_tokens=token,
            diagnostics=diagnostics,
        )

    def forward(
        self,
        *,
        past: torch.Tensor,
        prior: torch.Tensor,
        visual_tokens: torch.Tensor,
        residual_state: torch.Tensor,
        bridge_time: torch.Tensor,
        step_size: torch.Tensor | None = None,
        noise_level: torch.Tensor | None = None,
        learned_source: torch.Tensor | None = None,
        prepared_visual: PreparedVisualState | None = None,
        prepared_flow: PreparedFlowMemory | None = None,
    ) -> ResidualFlowLabOutput:
        source = self.predict_source(past, prior)[0] if learned_source is None else learned_source
        visual = self.prepare_visual(visual_tokens) if prepared_visual is None else prepared_visual
        return self.predict_residual_velocity_prepared(
            past=past,
            learned_source=source,
            prepared_visual=visual,
            residual_state=residual_state,
            bridge_time=bridge_time,
            step_size=step_size,
            noise_level=noise_level,
            prepared_flow=prepared_flow,
        )

    @torch.no_grad()
    def integrate_prepared(
        self,
        *,
        past: torch.Tensor,
        prior: torch.Tensor,
        prepared_visual: PreparedVisualState,
        steps: int = 4,
        learned_source: torch.Tensor | None = None,
        prepared_flow: PreparedFlowMemory | None = None,
    ) -> torch.Tensor:
        if steps <= 0:
            raise ValueError("steps must be positive")
        source = self.predict_source(past, prior)[0] if learned_source is None else learned_source
        memory = self.prepare_flow_memory(prepared_visual) if prepared_flow is None else prepared_flow
        residual = torch.zeros_like(source)
        batch = source.shape[0]
        dt = 1.0 / float(steps)
        step_size = torch.full((batch,), dt, device=source.device, dtype=source.dtype)
        noise = torch.zeros((batch,), device=source.device, dtype=source.dtype)
        for index in range(steps):
            time = torch.full((batch,), float(index) / float(steps), device=source.device, dtype=source.dtype)
            output = self.predict_residual_velocity_prepared(
                past=past,
                learned_source=source,
                prepared_visual=prepared_visual,
                residual_state=residual,
                bridge_time=time,
                step_size=step_size,
                noise_level=noise,
                prepared_flow=memory,
            )
            residual = residual + dt * output.residual_velocity
        return source + residual

    @torch.no_grad()
    def integrate(self, *, past: torch.Tensor, prior: torch.Tensor, visual_tokens: torch.Tensor, steps: int = 4) -> torch.Tensor:
        prepared = self.prepare_visual(visual_tokens)
        return self.integrate_prepared(past=past, prior=prior, prepared_visual=prepared, steps=steps)
