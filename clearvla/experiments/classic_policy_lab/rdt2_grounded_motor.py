from __future__ import annotations

"""Grounded shallow RDT2-FM experiment.

This branch does not compress the reference Transformer in place.  It changes
where information is processed:

* raw dense visual tokens are grounded once before motor generation;
* task information is represented by a small task-token set rather than being
  re-read by every low-level motor block;
* recent action history and the current state are encoded explicitly into a
  compact motion context;
* a native first-action stage predicts the executable first velocity from one
  action token only;
* a separate shallow tail stage predicts the remaining chunk while consuming a
  detached first-action anchor.

The flow-matching learning problem stays conventional: Gaussian noise is
transported to the recorded action chunk.  The branch intentionally does not
add a learned trajectory prior, residual bridge, or repeated raw-condition
cross attention, so improvements can be attributed to the information layout.
"""

from dataclasses import asdict, dataclass
import math
import re
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import LogisticNormal

from .rdt2_fm_reference import Attention, CrossAttention, FeedForward, RMSNorm, TimestepEmbedder


def _adapter(kind: str, in_features: int, out_features: int) -> nn.Module:
    if kind == "linear":
        return nn.Linear(in_features, out_features)
    match = re.fullmatch(r"mlp(\d+)x_silu", kind)
    if match is None:
        raise ValueError(f"unsupported adaptor kind: {kind!r}")
    depth = int(match.group(1))
    if depth <= 0:
        raise ValueError("MLP adaptor depth must be positive")
    modules: list[nn.Module] = [nn.Linear(in_features, out_features)]
    for _ in range(1, depth):
        modules.extend([nn.SiLU(), nn.Linear(out_features, out_features)])
    return nn.Sequential(*modules)


def _weighted_full_mse(pred: Tensor, target: Tensor, first_weight: float) -> Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}")
    if first_weight <= 0:
        raise ValueError("first_weight must be positive")
    weights = pred.new_ones((pred.shape[1],))
    weights[0] = float(first_weight)
    weights = weights / weights.mean()
    return ((pred - target).square() * weights.view(1, -1, 1)).mean()


@dataclass(frozen=True)
class GroundedMotorRDT2FMConfig:
    action_dim: int = 7
    state_dim: int = 7
    prediction_horizon: int = 24

    # Motor width and depth.  The reference path uses depth=14.  This branch is
    # intentionally shallow because semantic grounding is moved out of motor blocks.
    hidden_size: int = 512
    first_depth: int = 2
    tail_depth: int = 4
    num_heads: int = 8
    num_kv_heads: int = 4
    norm_eps: float = 1e-5
    multiple_of: int = 256
    ffn_dim_multiplier: float | None = None
    use_flash_attn: bool = True
    num_inference_timesteps: int = 5

    # Frozen upstream dense visual-token contract.
    dense_token_dim: int = 768
    visual_adaptor: str = "linear"

    # One-time visual grounding.
    grounding_depth: int = 2
    grounding_queries: int = 8
    default_task_tokens: int = 2

    # Explicit local-motion encoding.
    history_hidden_size: int = 192
    history_layers: int = 1
    motion_tokens: int = 4
    history_noise_std: float = 0.01

    # Losses.  The full target remains the standard FM velocity.  The native
    # first path receives a direct supervision term.
    full_flow_loss_weight: float = 1.0
    first_flow_loss_weight: float = 1.0
    full_first_position_weight: float = 1.0

    # The tail stage consumes the native first prediction as context without
    # sending tail gradients back into the first-action path.
    detach_first_anchor: bool = True

    def validate(self) -> None:
        positive = (
            self.action_dim,
            self.state_dim,
            self.prediction_horizon,
            self.hidden_size,
            self.first_depth,
            self.tail_depth,
            self.num_heads,
            self.num_kv_heads,
            self.dense_token_dim,
            self.grounding_depth,
            self.grounding_queries,
            self.default_task_tokens,
            self.history_hidden_size,
            self.history_layers,
            self.motion_tokens,
            self.num_inference_timesteps,
        )
        if min(positive) <= 0:
            raise ValueError("all grounded-motor dimensions must be positive")
        if self.prediction_horizon < 2:
            raise ValueError("prediction_horizon must be at least 2")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if self.history_hidden_size <= 0:
            raise ValueError("history_hidden_size must be positive")
        if self.history_noise_std < 0:
            raise ValueError("history_noise_std must be non-negative")
        if min(self.full_flow_loss_weight, self.first_flow_loss_weight, self.full_first_position_weight) <= 0:
            raise ValueError("loss weights must be positive")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


class MotionStateEncoder(nn.Module):
    """Encode recent actions and current state into compact motor-context tokens."""

    def __init__(self, config: GroundedMotorRDT2FMConfig) -> None:
        super().__init__()
        self.config = config
        h = config.history_hidden_size
        self.history_in = nn.Linear(2 * config.action_dim, h)
        self.state_in = nn.Linear(config.state_dim, h)
        self.encoder = nn.GRU(h, h, num_layers=config.history_layers, batch_first=True)
        self.motion_queries = nn.Parameter(torch.randn(config.motion_tokens, h) * 0.02)
        heads = 4 if h % 4 == 0 else 1
        self.readout = nn.MultiheadAttention(h, heads, batch_first=True)
        self.norm = nn.LayerNorm(h)
        self.token_out = nn.Linear(h, config.hidden_size)
        self.state_out = nn.Linear(h, config.hidden_size)
        self.summary_out = nn.Linear(2 * h, config.hidden_size)

    def forward(self, *, state: Tensor, past_actions: Tensor) -> tuple[Tensor, Tensor]:
        cfg = self.config
        if state.ndim != 2 or state.shape[-1] != cfg.state_dim:
            raise ValueError(f"state must be [B,{cfg.state_dim}], got {tuple(state.shape)}")
        if past_actions.ndim != 3 or past_actions.shape[-1] != cfg.action_dim:
            raise ValueError(f"past_actions must be [B,T,{cfg.action_dim}], got {tuple(past_actions.shape)}")
        history = past_actions
        if self.training and cfg.history_noise_std:
            history = history + torch.randn_like(history) * cfg.history_noise_std
        delta = torch.zeros_like(history)
        delta[:, 1:] = history[:, 1:] - history[:, :-1]
        encoded, hidden = self.encoder(self.history_in(torch.cat([history, delta], dim=-1)))
        state_hidden = self.state_in(state)
        query = self.motion_queries.unsqueeze(0).expand(state.shape[0], -1, -1) + state_hidden.unsqueeze(1)
        attended, _ = self.readout(query, encoded, encoded, need_weights=False)
        motion = self.token_out(self.norm(query + attended))
        state_token = self.state_out(state_hidden).unsqueeze(1)
        summary = self.summary_out(torch.cat([hidden[-1], state_hidden], dim=-1))
        return torch.cat([state_token, motion], dim=1), summary


class GroundingBlock(nn.Module):
    """Small query-based visual grounding block run before the motor policy."""

    def __init__(self, config: GroundedMotorRDT2FMConfig) -> None:
        super().__init__()
        core = {
            "hidden_size": config.hidden_size,
            "num_heads": config.num_heads,
            "num_kv_heads": config.num_kv_heads,
            "norm_eps": config.norm_eps,
            "multiple_of": config.multiple_of,
            "ffn_dim_multiplier": config.ffn_dim_multiplier,
            "use_flash_attn": config.use_flash_attn,
        }
        self.self_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.self_attn = Attention(core)
        self.cross_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.visual_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.cross_attn = CrossAttention(core)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn = FeedForward(config.hidden_size, 4 * config.hidden_size, config.multiple_of, config.ffn_dim_multiplier)
        self.scale = 1.0 / math.sqrt(max(config.grounding_depth, 1))

    def forward(self, query: Tensor, visual: Tensor, visual_mask: Tensor | None) -> Tensor:
        query = query + self.scale * self.self_attn(self.self_norm(query))
        query = query + self.scale * self.cross_attn(self.cross_norm(query), c=self.visual_norm(visual), mask=visual_mask)
        return query + self.scale * self.ffn(self.ffn_norm(query))


class SemanticVisualGrounder(nn.Module):
    """Ground raw dense visual tokens once into compact task-relevant tokens.

    Optional external ``task_tokens`` are supported for future multi-task use.
    Current single-task experiments use learned default task tokens.  Raw task or
    language tokens never enter the low-level motor blocks directly.
    """

    def __init__(self, config: GroundedMotorRDT2FMConfig) -> None:
        super().__init__()
        self.config = config
        self.visual_in = _adapter(config.visual_adaptor, config.dense_token_dim, config.hidden_size)
        self.default_task = nn.Parameter(torch.randn(1, config.default_task_tokens, config.hidden_size) * 0.02)
        self.query = nn.Parameter(torch.randn(1, config.grounding_queries, config.hidden_size) * 0.02)
        self.query_bias = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.blocks = nn.ModuleList([GroundingBlock(config) for _ in range(config.grounding_depth)])
        self.task_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.out_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(
        self,
        *,
        dense_tokens: Tensor,
        attention_mask: Tensor | None,
        motion_summary: Tensor,
        task_tokens: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        cfg = self.config
        if dense_tokens.ndim != 3 or dense_tokens.shape[-1] != cfg.dense_token_dim:
            raise ValueError(f"dense_tokens must be [B,L,{cfg.dense_token_dim}], got {tuple(dense_tokens.shape)}")
        if attention_mask is not None and attention_mask.shape != dense_tokens.shape[:2]:
            raise ValueError("attention_mask must match dense-token batch and length")
        if motion_summary.ndim != 2 or motion_summary.shape[-1] != cfg.hidden_size:
            raise ValueError(f"motion_summary must be [B,{cfg.hidden_size}], got {tuple(motion_summary.shape)}")
        visual = self.visual_in(dense_tokens)
        if task_tokens is None:
            task = self.default_task.expand(dense_tokens.shape[0], -1, -1)
        else:
            if task_tokens.ndim != 3 or task_tokens.shape[-1] != cfg.hidden_size:
                raise ValueError(f"task_tokens must already be [B,T,{cfg.hidden_size}], got {tuple(task_tokens.shape)}")
            task = task_tokens
        task = self.task_norm(task)
        task_summary = task.mean(dim=1)
        query = self.query.expand(dense_tokens.shape[0], -1, -1)
        query = query + self.query_bias(torch.cat([task_summary, motion_summary], dim=-1)).unsqueeze(1)
        for block in self.blocks:
            query = block(query, visual, attention_mask)
        grounded = self.out_norm(query)
        return torch.cat([task, grounded], dim=1), grounded.mean(dim=1)


class MotorBlock(nn.Module):
    """Shallow pre-norm motor block over already-grounded control context."""

    def __init__(self, config: GroundedMotorRDT2FMConfig, *, stage_depth: int) -> None:
        super().__init__()
        core = {
            "hidden_size": config.hidden_size,
            "num_heads": config.num_heads,
            "num_kv_heads": config.num_kv_heads,
            "norm_eps": config.norm_eps,
            "multiple_of": config.multiple_of,
            "ffn_dim_multiplier": config.ffn_dim_multiplier,
            "use_flash_attn": config.use_flash_attn,
        }
        self.self_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.self_attn = Attention(core)
        self.cross_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.context_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.cross_attn = CrossAttention(core)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn = FeedForward(config.hidden_size, 4 * config.hidden_size, config.multiple_of, config.ffn_dim_multiplier)
        self.scale = 1.0 / math.sqrt(max(stage_depth, 1))

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        x = x + self.scale * self.self_attn(self.self_norm(x))
        x = x + self.scale * self.cross_attn(self.cross_norm(x), c=self.context_norm(context))
        return x + self.scale * self.ffn(self.ffn_norm(x))


class ActionReadout(nn.Module):
    def __init__(self, config: GroundedMotorRDT2FMConfig) -> None:
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.fc1 = nn.Linear(config.hidden_size, 2 * config.hidden_size)
        self.fc2 = nn.Linear(2 * config.hidden_size, config.action_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.silu(self.fc1(self.norm(x))))


@dataclass
class PreparedGroundedContext:
    tokens: Tensor
    grounding_summary: Tensor


@dataclass
class GroundedMotorVelocityOutput:
    first: Tensor
    tail: Tensor
    full: Tensor
    first_anchor_rms: Tensor
    grounding_rms: Tensor


class GroundedMotorCore(nn.Module):
    """Native-first hierarchical motor generator."""

    def __init__(self, config: GroundedMotorRDT2FMConfig, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.config = config
        self.t_embedder = TimestepEmbedder(config.hidden_size, dtype=dtype)
        self.action_in = _adapter("mlp2x_silu", config.action_dim, config.hidden_size)
        self.first_pos = nn.Parameter(torch.randn(1, 1, config.hidden_size) * 0.02)
        self.tail_pos = nn.Parameter(torch.randn(1, config.prediction_horizon - 1, config.hidden_size) * 0.02)
        self.first_blocks = nn.ModuleList([MotorBlock(config, stage_depth=config.first_depth) for _ in range(config.first_depth)])
        self.tail_blocks = nn.ModuleList([MotorBlock(config, stage_depth=config.tail_depth) for _ in range(config.tail_depth)])
        self.readout = ActionReadout(config)
        self.anchor_in = nn.Linear(config.action_dim, config.hidden_size)
        self.anchor_type = nn.Parameter(torch.randn(1, 1, config.hidden_size) * 0.02)
        self._initialize(dtype)

    def _initialize(self, dtype: torch.dtype) -> None:
        def basic(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.apply(basic)
        self.to(dtype=dtype)

    def _time(self, timesteps: Tensor, batch: int) -> Tensor:
        time = self.t_embedder(timesteps)
        if time.shape[0] == 1:
            time = time.expand(batch, -1)
        if time.shape[0] != batch:
            raise ValueError("timestep batch does not match action batch")
        return time.unsqueeze(1)

    def first_velocity(self, *, noisy_first: Tensor, timesteps: Tensor, context: PreparedGroundedContext) -> Tensor:
        if noisy_first.ndim != 3 or tuple(noisy_first.shape[1:]) != (1, self.config.action_dim):
            raise ValueError(f"noisy_first must be [B,1,{self.config.action_dim}], got {tuple(noisy_first.shape)}")
        x = self.action_in(noisy_first) + self.first_pos + self._time(timesteps, noisy_first.shape[0])
        for block in self.first_blocks:
            x = block(x, context.tokens)
        return self.readout(x)

    def full_velocity(self, *, noisy_action: Tensor, timesteps: Tensor, context: PreparedGroundedContext) -> GroundedMotorVelocityOutput:
        cfg = self.config
        if noisy_action.ndim != 3 or tuple(noisy_action.shape[1:]) != (cfg.prediction_horizon, cfg.action_dim):
            raise ValueError(f"noisy_action must be [B,{cfg.prediction_horizon},{cfg.action_dim}], got {tuple(noisy_action.shape)}")
        first = self.first_velocity(noisy_first=noisy_action[:, :1], timesteps=timesteps, context=context)
        anchor_source = first.detach() if cfg.detach_first_anchor else first
        anchor = self.anchor_in(anchor_source) + self.anchor_type
        tail_context = PreparedGroundedContext(
            tokens=torch.cat([context.tokens, anchor], dim=1),
            grounding_summary=context.grounding_summary,
        )
        tail = self.action_in(noisy_action[:, 1:]) + self.tail_pos + self._time(timesteps, noisy_action.shape[0])
        for block in self.tail_blocks:
            tail = block(tail, tail_context.tokens)
        tail_velocity = self.readout(tail)
        full = torch.cat([first, tail_velocity], dim=1)
        return GroundedMotorVelocityOutput(
            first=first,
            tail=tail_velocity,
            full=full,
            first_anchor_rms=anchor_source.square().mean().sqrt(),
            grounding_rms=context.grounding_summary.square().mean().sqrt(),
        )


class GroundedMotorRDT2FM(nn.Module):
    """One-time grounding plus native-first shallow flow-matching policy."""

    def __init__(self, config: GroundedMotorRDT2FMConfig = GroundedMotorRDT2FMConfig(), *, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.motion = MotionStateEncoder(config)
        self.grounder = SemanticVisualGrounder(config)
        self.core = GroundedMotorCore(config, dtype=dtype)
        self.pred_horizon = config.prediction_horizon
        self.action_dim = config.action_dim
        self.num_inference_timesteps = config.num_inference_timesteps
        self.to(dtype=dtype)

    def sample_timesteps(self, batch_size: int, device: torch.device) -> Tensor:
        distribution = LogisticNormal(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))
        return distribution.sample((batch_size,))[:, 0]

    def prepare_context(
        self,
        *,
        state_tokens: Tensor,
        past_actions: Tensor,
        dense_tokens: Tensor,
        attention_mask: Tensor | None,
        task_tokens: Tensor | None = None,
    ) -> PreparedGroundedContext:
        motion_tokens, motion_summary = self.motion(state=state_tokens, past_actions=past_actions)
        grounded_tokens, grounding_summary = self.grounder(
            dense_tokens=dense_tokens,
            attention_mask=attention_mask,
            motion_summary=motion_summary,
            task_tokens=task_tokens,
        )
        return PreparedGroundedContext(tokens=torch.cat([motion_tokens, grounded_tokens], dim=1), grounding_summary=grounding_summary)

    def compute_loss(
        self,
        *,
        state_tokens: Tensor,
        past_actions: Tensor,
        action_gt: Tensor,
        dense_tokens: Tensor,
        attention_mask: Tensor | None = None,
        task_tokens: Tensor | None = None,
    ) -> dict[str, Tensor]:
        batch = action_gt.shape[0]
        noise = torch.randn_like(action_gt)
        timesteps = self.sample_timesteps(batch, action_gt.device).to(dtype=action_gt.dtype)
        blend = timesteps.view(-1, 1, 1)
        noisy = action_gt * blend + noise * (1 - blend)
        target_velocity = action_gt - noise
        context = self.prepare_context(
            state_tokens=state_tokens,
            past_actions=past_actions,
            dense_tokens=dense_tokens,
            attention_mask=attention_mask,
            task_tokens=task_tokens,
        )
        output = self.core.full_velocity(noisy_action=noisy, timesteps=timesteps, context=context)
        full = _weighted_full_mse(output.full, target_velocity, self.config.full_first_position_weight)
        first = F.mse_loss(output.first, target_velocity[:, :1])
        total = self.config.full_flow_loss_weight * full + self.config.first_flow_loss_weight * first
        return {
            "loss": total,
            "full_flow_mse": full.detach(),
            "first_flow_mse": first.detach(),
            "first_anchor_rms": output.first_anchor_rms.detach(),
            "grounding_rms": output.grounding_rms.detach(),
        }

    @torch.no_grad()
    def predict_first_action(
        self,
        *,
        state_tokens: Tensor,
        past_actions: Tensor,
        dense_tokens: Tensor,
        attention_mask: Tensor | None = None,
        task_tokens: Tensor | None = None,
        noisy_first: Tensor | None = None,
        generator: torch.Generator | None = None,
        inference_steps: int | None = None,
    ) -> Tensor:
        context = self.prepare_context(
            state_tokens=state_tokens,
            past_actions=past_actions,
            dense_tokens=dense_tokens,
            attention_mask=attention_mask,
            task_tokens=task_tokens,
        )
        batch = state_tokens.shape[0]
        if noisy_first is None:
            noisy_first = torch.randn((batch, 1, self.action_dim), device=state_tokens.device, dtype=state_tokens.dtype, generator=generator)
        steps = int(inference_steps or self.num_inference_timesteps)
        if steps <= 0:
            raise ValueError("inference_steps must be positive")
        dt = 1.0 / steps
        time = torch.tensor([0.0], device=state_tokens.device, dtype=state_tokens.dtype)
        for _ in range(steps):
            velocity = self.core.first_velocity(noisy_first=noisy_first, timesteps=time, context=context)
            noisy_first = noisy_first + velocity * dt
            time = time + dt
        return noisy_first[:, 0]

    @torch.no_grad()
    def predict_action(
        self,
        *,
        state_tokens: Tensor,
        past_actions: Tensor,
        dense_tokens: Tensor,
        attention_mask: Tensor | None = None,
        task_tokens: Tensor | None = None,
        noisy_action: Tensor | None = None,
        generator: torch.Generator | None = None,
        inference_steps: int | None = None,
    ) -> Tensor:
        context = self.prepare_context(
            state_tokens=state_tokens,
            past_actions=past_actions,
            dense_tokens=dense_tokens,
            attention_mask=attention_mask,
            task_tokens=task_tokens,
        )
        batch = state_tokens.shape[0]
        if noisy_action is None:
            noisy_action = torch.randn((batch, self.pred_horizon, self.action_dim), device=state_tokens.device, dtype=state_tokens.dtype, generator=generator)
        steps = int(inference_steps or self.num_inference_timesteps)
        if steps <= 0:
            raise ValueError("inference_steps must be positive")
        dt = 1.0 / steps
        time = torch.tensor([0.0], device=state_tokens.device, dtype=state_tokens.dtype)
        for _ in range(steps):
            velocity = self.core.full_velocity(noisy_action=noisy_action, timesteps=time, context=context).full
            noisy_action = noisy_action + velocity * dt
            time = time + dt
        return noisy_action

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def config_dict(self) -> dict[str, Any]:
        return asdict(self.config)


__all__ = [
    "GroundedMotorRDT2FM",
    "GroundedMotorRDT2FMConfig",
    "GroundedMotorVelocityOutput",
    "MotionStateEncoder",
    "SemanticVisualGrounder",
]
