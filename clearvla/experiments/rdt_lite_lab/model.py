from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .schedule import CosineDiffusionSchedule, DiffusionScheduleConfig

ObjectiveName = Literal["rdt_denoise", "pi_flow"]
ConditioningMode = Literal["concat", "camera_alternate", "alternate"]
TimeEncoding = Literal["rdt_discrete", "pi_continuous"]


def _sincos_1d(length: int, dim: int, *, device: torch.device | None = None) -> torch.Tensor:
    if length <= 0 or dim <= 0:
        raise ValueError("length and dim must be positive")
    half = dim // 2
    pos = torch.arange(length, dtype=torch.float32, device=device)[:, None]
    if half == 0:
        return torch.zeros(length, dim, dtype=torch.float32, device=device)
    omega = torch.arange(half, dtype=torch.float32, device=device)
    omega = torch.exp(-math.log(10000.0) * omega / max(half, 1))
    emb = torch.cat([torch.sin(pos * omega[None]), torch.cos(pos * omega[None])], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb[:, :dim]


def _sincos_2d(height: int, width: int, dim: int) -> torch.Tensor:
    """Separable two-dimensional patch encoding: row + column."""
    row = _sincos_1d(height, dim)[:, None, :]
    col = _sincos_1d(width, dim)[None, :, :]
    return (row + col).reshape(height * width, dim)


def _rdt_timestep_embedding(time: torch.Tensor, dim: int, *, max_period: int = 10000) -> torch.Tensor:
    if time.ndim != 1:
        raise ValueError("time must be [B]")
    half = dim // 2
    freqs = torch.exp(
        -math.log(float(max_period))
        * torch.arange(half, dtype=torch.float32, device=time.device)
        / max(half, 1)
    )
    phase = time.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb[:, :dim].to(dtype=time.dtype)


def _pi_time_embedding(time: torch.Tensor, dim: int, *, min_period: float = 4e-3, max_period: float = 4.0) -> torch.Tensor:
    if time.ndim != 1:
        raise ValueError("time must be [B]")
    half = dim // 2
    fraction = torch.arange(half, device=time.device, dtype=time.dtype) / max(half - 1, 1)
    period = min_period * (max_period / min_period) ** fraction
    phase = time[:, None] / period[None] * (2.0 * math.pi)
    emb = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb[:, :dim]


def _mlp(in_dim: int, out_dim: int, *, depth: int) -> nn.Sequential:
    if depth <= 0:
        raise ValueError("depth must be positive")
    layers: list[nn.Module] = [nn.Linear(in_dim, out_dim)]
    for _ in range(1, depth):
        layers.extend([nn.GELU(approximate="tanh"), nn.Linear(out_dim, out_dim)])
    return nn.Sequential(*layers)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.eps).to(dtype=x.dtype) * self.weight


class HeadRMSNorm(nn.Module):
    """RMSNorm applied independently to each attention head."""
    def __init__(self, head_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(head_dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.eps).to(dtype=x.dtype) * self.weight


@dataclass(frozen=True)
class AttentionKV:
    key: torch.Tensor
    value: torch.Tensor


class RDTQKNormAttention(nn.Module):
    """RDT-style attention: per-head RMSNorm(Q/K) + standard SDPA."""

    def __init__(self, dim: int, heads: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = dim // heads
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.kv_proj = nn.Linear(dim, 2 * dim, bias=True)
        self.q_norm = HeadRMSNorm(self.head_dim)
        self.k_norm = HeadRMSNorm(self.head_dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = float(dropout)

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        return x.reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)

    def prepare_key_value(self, context: torch.Tensor) -> AttentionKV:
        kv = self.kv_proj(context)
        key, value = kv.chunk(2, dim=-1)
        key = self.k_norm(self._reshape(key))
        value = self._reshape(value)
        return AttentionKV(key=key, value=value)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor | None = None,
        *,
        prepared_kv: AttentionKV | None = None,
    ) -> torch.Tensor:
        if (context is None) == (prepared_kv is None):
            raise ValueError("provide exactly one of context or prepared_kv")
        q = self.q_norm(self._reshape(self.q_proj(query)))
        memory = self.prepare_key_value(context) if prepared_kv is None else prepared_kv
        out = F.scaled_dot_product_attention(
            q,
            memory.key,
            memory.value,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(query.shape[0], query.shape[1], self.dim)
        return self.out_proj(out)


@dataclass(frozen=True)
class RDTLiteModelConfig:
    """Clean lightweight RDT reference for a fixed-arm direct action policy."""

    state_dim: int = 7
    action_dim: int = 7
    chunk_len: int = 25
    obs_horizon: int = 2
    state_history_len: int = 1
    camera_names: tuple[str, ...] = ("top", "wrist")
    camera_order: tuple[str, ...] = ("top", "wrist")
    patch_grid: tuple[int, int] = (16, 16)
    teacher_dim: int = 384
    hidden_size: int = 384
    depth: int = 6
    num_heads: int = 8
    ffn_hidden: int = 384
    img_adaptor_depth: int = 2
    state_adaptor_depth: int = 3
    action_adaptor_depth: int = 3
    conditioning_mode: ConditioningMode = "concat"
    independent_camera_dropout: float = 0.0
    include_visual_delta_tokens: bool = False
    delta_gate_bias: float = -2.0
    dropout: float = 0.0
    time_encoding: TimeEncoding = "rdt_discrete"
    control_frequency_hz: float = 30.0
    decoder_output_init_std: float = 1e-3

    def validate(self) -> None:
        positive = (
            self.state_dim, self.action_dim, self.chunk_len, self.obs_horizon,
            self.state_history_len, self.patch_grid[0], self.patch_grid[1],
            self.teacher_dim, self.hidden_size, self.depth, self.num_heads,
            self.ffn_hidden, self.img_adaptor_depth, self.state_adaptor_depth,
            self.action_adaptor_depth,
        )
        if any(int(value) <= 0 for value in positive):
            raise ValueError("all dimensions and depths must be positive")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if not self.camera_names or len(set(self.camera_names)) != len(self.camera_names):
            raise ValueError(f"invalid camera_names={self.camera_names}")
        if not self.camera_order or any(camera not in self.camera_names for camera in self.camera_order):
            raise ValueError("camera_order must be a non-empty subset of camera_names")
        if self.conditioning_mode not in ("concat", "camera_alternate", "alternate"):
            raise ValueError(f"unsupported conditioning_mode={self.conditioning_mode!r}")
        if self.time_encoding not in ("rdt_discrete", "pi_continuous"):
            raise ValueError(f"unsupported time_encoding={self.time_encoding!r}")
        if not 0.0 <= self.independent_camera_dropout < 1.0:
            raise ValueError("independent_camera_dropout must be in [0,1)")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        if self.control_frequency_hz <= 0:
            raise ValueError("control_frequency_hz must be positive")
        if self.decoder_output_init_std < 0:
            raise ValueError("decoder_output_init_std must be non-negative")

    @property
    def patch_count(self) -> int:
        return int(self.patch_grid[0] * self.patch_grid[1])

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["camera_names"] = list(self.camera_names)
        payload["camera_order"] = list(self.camera_order)
        payload["patch_grid"] = list(self.patch_grid)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "RDTLiteModelConfig":
        payload = dict(data)
        payload["camera_names"] = tuple(str(value) for value in payload.get("camera_names", ("top", "wrist")))
        payload["camera_order"] = tuple(str(value) for value in payload.get("camera_order", payload["camera_names"]))
        patch = payload.get("patch_grid", (16, 16))
        payload["patch_grid"] = (int(patch[0]), int(patch[1]))  # type: ignore[index]
        if "state_dim" not in payload:
            payload["state_dim"] = int(payload.get("action_dim", 7))
        out = cls(**payload)  # type: ignore[arg-type]
        out.validate()
        return out


@dataclass(frozen=True)
class PreparedVisualConditions:
    camera_tokens: tuple[torch.Tensor, ...]
    concat_tokens: torch.Tensor
    camera_keep_mask: torch.Tensor
    delta_statistics_by_camera: torch.Tensor


class VisualDeltaAugmentor(nn.Module):
    def __init__(self, config: RDTLiteModelConfig) -> None:
        super().__init__()
        dim = config.hidden_size
        self.config = config
        self.delta_proj = _mlp(config.teacher_dim, dim, depth=config.img_adaptor_depth)
        self.delta_magnitude_proj = _mlp(1, dim, depth=2)
        self.delta_gate = nn.Linear(1, 1)
        nn.init.zeros_(self.delta_gate.weight)
        nn.init.constant_(self.delta_gate.bias, config.delta_gate_bias)
        self.type_embed = nn.Parameter(torch.randn(dim) * 0.02)

    def forward(self, delta_raw: torch.Tensor, *, camera_embed: torch.Tensor, patch_embed: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(delta_raw.float().square().mean(dim=-1, keepdim=True) + 1e-12).to(dtype=delta_raw.dtype)
        direction = delta_raw / rms.clamp_min(1e-6)
        magnitude = torch.log1p(rms)
        delta = torch.sigmoid(self.delta_gate(magnitude)) * self.delta_proj(direction)
        delta = delta + self.delta_magnitude_proj(magnitude)
        delta = delta + camera_embed[None, :, None, :]
        delta = delta + patch_embed[None, None, :, :]
        delta = delta + self.type_embed[None, None, None, :]
        return delta


class VisualConditionAdaptor(nn.Module):
    """Frozen patch-token adaptor with explicit 2D, camera and frame position."""

    def __init__(self, config: RDTLiteModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        dim = config.hidden_size
        cameras = len(config.camera_names)
        self.input_norm = nn.LayerNorm(config.teacher_dim)
        self.image_proj = _mlp(config.teacher_dim, dim, depth=config.img_adaptor_depth)
        self.camera_embed = nn.Parameter(torch.randn(cameras, dim) * 0.02)
        self.frame_embed = nn.Parameter(torch.randn(config.obs_horizon, dim) * 0.02)
        self.mask_token = nn.Parameter(torch.randn(cameras, dim) * 0.02)
        self.observed_type = nn.Parameter(torch.randn(dim) * 0.02)
        self.out_norm = RMSNorm(dim)
        self.delta_augmentor = VisualDeltaAugmentor(config) if config.include_visual_delta_tokens else None
        self.register_buffer("patch_embed", _sincos_2d(config.patch_grid[0], config.patch_grid[1], dim), persistent=True)

    def _sample_keep(self, batch: int, *, device: torch.device) -> torch.Tensor:
        cameras = len(self.config.camera_names)
        keep = torch.ones((batch, cameras), dtype=torch.bool, device=device)
        probability = float(self.config.independent_camera_dropout)
        if self.training and probability > 0:
            keep = torch.rand((batch, cameras), device=device) >= probability
            missing = ~keep.any(dim=1)
            if bool(missing.any()):
                rows = torch.nonzero(missing, as_tuple=False).squeeze(-1)
                selected = torch.randint(0, cameras, (rows.shape[0],), device=device)
                keep[rows, selected] = True
        return keep

    def forward(self, visual_tokens: torch.Tensor) -> PreparedVisualConditions:
        cfg = self.config
        expected = (cfg.obs_horizon, len(cfg.camera_names), cfg.patch_count, cfg.teacher_dim)
        if visual_tokens.ndim != 5 or tuple(visual_tokens.shape[1:]) != expected:
            raise ValueError(f"visual_tokens must be [B,{expected}], got {tuple(visual_tokens.shape)}")
        batch = visual_tokens.shape[0]
        keep = self._sample_keep(batch, device=visual_tokens.device)
        observed = self.image_proj(self.input_norm(visual_tokens))
        observed = torch.where(keep[:, None, :, None, None], observed, self.mask_token[None, None, :, None, :])
        observed = observed + self.frame_embed[None, :, None, None, :]
        observed = observed + self.camera_embed[None, None, :, None, :]
        observed = observed + self.patch_embed[None, None, None, :, :]
        observed = observed + self.observed_type[None, None, None, None, :]
        by_camera = [observed[:, :, camera].reshape(batch, -1, cfg.hidden_size) for camera in range(len(cfg.camera_names))]

        statistics = torch.zeros((batch, len(cfg.camera_names), 4), device=visual_tokens.device, dtype=visual_tokens.dtype)
        if cfg.obs_horizon >= 2:
            delta_raw = visual_tokens[:, -1] - visual_tokens[:, -2]
            delta_raw = delta_raw * keep[:, :, None, None].to(dtype=delta_raw.dtype)
            rms = torch.sqrt(delta_raw.float().square().mean(dim=-1) + 1e-12).to(dtype=visual_tokens.dtype)
            topk = min(16, cfg.patch_count)
            statistics = torch.stack([
                rms.mean(dim=-1),
                rms.max(dim=-1).values,
                rms.topk(topk, dim=-1).values.mean(dim=-1),
                torch.quantile(rms.float(), 0.90, dim=-1).to(dtype=visual_tokens.dtype),
            ], dim=-1)
            if self.delta_augmentor is not None:
                delta = self.delta_augmentor(delta_raw, camera_embed=self.camera_embed, patch_embed=self.patch_embed)
                for camera in range(len(cfg.camera_names)):
                    by_camera[camera] = torch.cat([by_camera[camera], delta[:, camera]], dim=1)

        camera_tokens = tuple(self.out_norm(value) for value in by_camera)
        concat_tokens = self.out_norm(torch.cat(camera_tokens, dim=1))
        return PreparedVisualConditions(camera_tokens, concat_tokens, keep, statistics)


class RDTDenoiseTimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_dim: int = 256) -> None:
        super().__init__()
        self.frequency_dim = int(frequency_dim)
        self.mlp = nn.Sequential(nn.Linear(frequency_dim, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        return self.mlp(_rdt_timestep_embedding(time, self.frequency_dim))


class PiFlowTimeEmbedder(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        return self.mlp(_pi_time_embedding(time, self.hidden_size))


class RDTLiteBlock(nn.Module):
    def __init__(self, config: RDTLiteModelConfig) -> None:
        super().__init__()
        dim = config.hidden_size
        self.norm1 = RMSNorm(dim)
        self.self_attn = RDTQKNormAttention(dim, config.num_heads, dropout=config.dropout)
        self.norm2 = RMSNorm(dim)
        self.cross_attn = RDTQKNormAttention(dim, config.num_heads, dropout=config.dropout)
        self.norm3 = RMSNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, config.ffn_hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_hidden, dim),
        )

    def prepare_memory(self, condition: torch.Tensor) -> AttentionKV:
        return self.cross_attn.prepare_key_value(condition)

    def forward(self, x: torch.Tensor, *, memory: AttentionKV) -> torch.Tensor:
        normed = self.norm1(x)
        x = x + self.self_attn(normed, normed)
        x = x + self.cross_attn(self.norm2(x), prepared_kv=memory)
        x = x + self.ffn(self.norm3(x))
        return x


class SplitActionDecoder(nn.Module):
    """Separate arm and gripper projections so one channel cannot dominate."""
    def __init__(self, hidden_size: int, action_dim: int, *, output_init_std: float) -> None:
        super().__init__()
        if action_dim < 2:
            raise ValueError("SplitActionDecoder requires action_dim >= 2")
        self.arm_dim = action_dim - 1
        self.norm = RMSNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.arm_out = nn.Linear(hidden_size, self.arm_dim)
        self.gripper_out = nn.Linear(hidden_size, 1)
        for layer in (self.arm_out, self.gripper_out):
            if output_init_std == 0:
                nn.init.zeros_(layer.weight)
            else:
                nn.init.normal_(layer.weight, std=float(output_init_std))
            nn.init.zeros_(layer.bias)

    def forward(self, action_tokens: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.fc(self.norm(action_tokens)), approximate="tanh")
        return torch.cat([self.arm_out(hidden), self.gripper_out(hidden)], dim=-1)


@dataclass(frozen=True)
class PreparedRDTLite:
    visual: PreparedVisualConditions
    block_memories: tuple[AttentionKV, ...]


@dataclass(frozen=True)
class RDTLiteOutput:
    prediction: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


class RDTLiteModel(nn.Module):
    """Lightweight, explicit RDT-style direct action generator."""

    def __init__(self, config: RDTLiteModelConfig = RDTLiteModelConfig()) -> None:
        super().__init__()
        config.validate()
        self.config = config
        dim = config.hidden_size
        self.visual_adaptor = VisualConditionAdaptor(config)
        self.state_adaptor = _mlp(config.state_dim, dim, depth=config.state_adaptor_depth)
        self.action_adaptor = _mlp(config.action_dim, dim, depth=config.action_adaptor_depth)
        self.time_embedder = RDTDenoiseTimestepEmbedder(dim) if config.time_encoding == "rdt_discrete" else PiFlowTimeEmbedder(dim)
        self.frequency_embedder = RDTDenoiseTimestepEmbedder(dim)
        self.state_type = nn.Parameter(torch.randn(dim) * 0.02)
        self.action_type = nn.Parameter(torch.randn(dim) * 0.02)
        token_count = 2 + config.state_history_len + config.chunk_len
        self.register_buffer("trajectory_pos", _sincos_1d(token_count, dim)[None], persistent=True)
        self.blocks = nn.ModuleList([RDTLiteBlock(config) for _ in range(config.depth)])
        self.decoder = SplitActionDecoder(dim, config.action_dim, output_init_std=config.decoder_output_init_std)
        self._camera_index = {name: index for index, name in enumerate(config.camera_names)}

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def camera_schedule(self) -> tuple[str, ...]:
        if self.config.conditioning_mode == "concat":
            return tuple("concat" for _ in self.blocks)
        order = self.config.camera_order
        return tuple(order[index % len(order)] for index in range(len(self.blocks)))

    def prepare_visual(self, visual_tokens: torch.Tensor) -> PreparedRDTLite:
        visual = self.visual_adaptor(visual_tokens)
        memories: list[AttentionKV] = []
        for block, source in zip(self.blocks, self.camera_schedule(), strict=True):
            condition = visual.concat_tokens if source == "concat" else visual.camera_tokens[self._camera_index[source]]
            memories.append(block.prepare_memory(condition))
        return PreparedRDTLite(visual=visual, block_memories=tuple(memories))

    def forward_prepared(
        self,
        *,
        state_history: torch.Tensor,
        noisy_actions: torch.Tensor,
        time: torch.Tensor,
        prepared: PreparedRDTLite,
    ) -> RDTLiteOutput:
        cfg = self.config
        if tuple(state_history.shape[1:]) != (cfg.state_history_len, cfg.state_dim):
            raise ValueError(f"state_history must be [B,{cfg.state_history_len},{cfg.state_dim}]")
        if tuple(noisy_actions.shape[1:]) != (cfg.chunk_len, cfg.action_dim):
            raise ValueError(f"noisy_actions must be [B,{cfg.chunk_len},{cfg.action_dim}]")
        if time.ndim != 1 or time.shape[0] != noisy_actions.shape[0]:
            raise ValueError("time must be [B]")
        state_tokens = self.state_adaptor(state_history) + self.state_type[None, None]
        action_tokens = self.action_adaptor(noisy_actions) + self.action_type[None, None]
        timestep = self.time_embedder(time)[:, None]
        frequency = torch.full_like(time, float(cfg.control_frequency_hz))
        frequency_token = self.frequency_embedder(frequency)[:, None]
        tokens = torch.cat([timestep, frequency_token, state_tokens, action_tokens], dim=1)
        tokens = tokens + self.trajectory_pos
        for block, memory in zip(self.blocks, prepared.block_memories, strict=True):
            tokens = block(tokens, memory=memory)
        prediction = self.decoder(tokens[:, -cfg.chunk_len :])
        diagnostics = {
            "trajectory_token_norm": tokens.detach().float().norm(dim=-1).mean(),
            "prediction_norm": prediction.detach().float().norm(dim=-1).mean(),
            "camera_keep_rate": prepared.visual.camera_keep_mask.detach().float().mean(),
            "delta_statistics_mean": prepared.visual.delta_statistics_by_camera.detach().float().mean(),
        }
        return RDTLiteOutput(prediction, diagnostics)

    def forward(self, *, state_history: torch.Tensor, visual_tokens: torch.Tensor, noisy_actions: torch.Tensor, time: torch.Tensor) -> RDTLiteOutput:
        return self.forward_prepared(state_history=state_history, noisy_actions=noisy_actions, time=time, prepared=self.prepare_visual(visual_tokens))

    @torch.no_grad()
    def sample_actions_prepared(
        self,
        *,
        objective: ObjectiveName,
        state_history: torch.Tensor,
        prepared: PreparedRDTLite,
        steps: int,
        diffusion_schedule: CosineDiffusionSchedule | None = None,
        initial_noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        cfg = self.config
        if steps <= 0:
            raise ValueError("steps must be positive")
        batch = state_history.shape[0]
        shape = (batch, cfg.chunk_len, cfg.action_dim)
        state = initial_noise
        if state is None:
            state = torch.randn(shape, device=state_history.device, dtype=state_history.dtype, generator=generator)
        if tuple(state.shape) != shape:
            raise ValueError(f"initial_noise must have shape={shape}")
        if objective == "pi_flow":
            dt = -1.0 / float(steps)
            for index in range(steps):
                time = torch.full((batch,), 1.0 + index * dt, device=state.device, dtype=state.dtype)
                velocity = self.forward_prepared(state_history=state_history, noisy_actions=state, time=time, prepared=prepared).prediction
                state = state + dt * velocity
            return state
        if objective == "rdt_denoise":
            schedule = diffusion_schedule or CosineDiffusionSchedule(DiffusionScheduleConfig())
            timesteps = schedule.inference_timesteps(steps, device=state.device)
            for index, timestep in enumerate(timesteps.tolist()):
                time = torch.full((batch,), float(timestep), device=state.device, dtype=state.dtype)
                pred_clean = self.forward_prepared(state_history=state_history, noisy_actions=state, time=time, prepared=prepared).prediction
                prev = int(timesteps[index + 1]) if index + 1 < len(timesteps) else None
                state = schedule.ddim_step(state, pred_clean, int(timestep), prev)
            return state
        raise ValueError(f"unsupported objective={objective!r}")

    @torch.no_grad()
    def sample_actions(
        self,
        *,
        objective: ObjectiveName,
        state_history: torch.Tensor,
        visual_tokens: torch.Tensor,
        steps: int,
        diffusion_schedule: CosineDiffusionSchedule | None = None,
        initial_noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return self.sample_actions_prepared(
            objective=objective,
            state_history=state_history,
            prepared=self.prepare_visual(visual_tokens),
            steps=steps,
            diffusion_schedule=diffusion_schedule,
            initial_noise=initial_noise,
            generator=generator,
        )
