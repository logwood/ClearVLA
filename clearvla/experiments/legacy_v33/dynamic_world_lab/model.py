from __future__ import annotations

"""Dynamic-predictive bottleneck for local action-conditioned world modelling.

The model intentionally predicts compact temporal scene and dynamics states
rather than all future DINO patch tokens.  A root observation remains an
immutable condition, while both the compact scene geometry and dynamics state
are rolled forward.  Training and evaluation use the same autoregressive
rollout; a teacher-forced one-step branch exists only as a diagnostic and a
bounded stabilizer.

No policy module is imported or referenced here.  The only closed loop in this
release is internal to the world model:

    predicted (scene, dynamics)_{k+1}
        -> input state for predicting (scene, dynamics)_{k+2}.
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class _FeedForward(nn.Module):
    def __init__(self, hidden: int, expansion: float = 4.0) -> None:
        super().__init__()
        inner = int(round(hidden * expansion))
        self.net = nn.Sequential(
            nn.Linear(hidden, inner),
            nn.GELU(),
            nn.Linear(inner, hidden),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class _SelfAttentionBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden)
        self.ffn = _FeedForward(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, *, causal: bool = False) -> Tensor:
        mask = None
        if causal:
            length = x.shape[1]
            mask = torch.triu(
                torch.ones(length, length, device=x.device, dtype=torch.bool), diagonal=1
            )
        y, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=mask, need_weights=False)
        x = x + self.dropout(y)
        return x + self.dropout(self.ffn(self.norm2(x)))


class _QueryPool(nn.Module):
    def __init__(self, hidden: int, heads: int, query_count: int, dropout: float) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, query_count, hidden) * 0.02)
        self.query_norm = nn.LayerNorm(hidden)
        self.source_norm = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.out_norm = nn.LayerNorm(hidden)

    def forward(self, source: Tensor, *, query_bias: Tensor | None = None) -> Tensor:
        batch = source.shape[0]
        query = self.queries.expand(batch, -1, -1)
        if query_bias is not None:
            if query_bias.shape != query.shape:
                raise ValueError("query_bias must match expanded query shape")
            query = query + query_bias
        pooled, _ = self.attn(
            self.query_norm(query), self.source_norm(source), self.source_norm(source), need_weights=False
        )
        return self.out_norm(pooled)


@dataclass(frozen=True)
class DynamicPredictiveWorldConfig:
    latent_dim: int = 768
    action_dim: int = 7
    state_dim: int = 7
    action_horizon: int = 48
    history_length: int = 3
    num_cameras: int = 2
    patches_per_camera: int = 64
    future_offsets: tuple[int, ...] = (8, 24, 48)
    hidden_size: int = 256
    encoder_depth: int = 3
    predictor_depth: int = 3
    action_depth: int = 3
    num_heads: int = 8
    context_tokens: int = 8
    dynamic_tokens: int = 16
    descriptor_projection_dim: int = 32
    dropout: float = 0.0
    input_mode: Literal["full", "current-only", "action-only"] = "full"
    gripper_dim_index: int = -1
    descriptor_seed: int = 34033

    def validate(self) -> None:
        if min(
            self.latent_dim,
            self.action_dim,
            self.state_dim,
            self.action_horizon,
            self.history_length,
            self.num_cameras,
            self.patches_per_camera,
            self.hidden_size,
            self.num_heads,
            self.context_tokens,
            self.dynamic_tokens,
            self.descriptor_projection_dim,
        ) <= 0:
            raise ValueError("dynamic-world dimensions must be positive")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.action_dim != self.state_dim:
            raise ValueError("action_dim and state_dim must match for state-relative action encoding")
        if len(self.future_offsets) == 0:
            raise ValueError("future_offsets must be non-empty")
        if tuple(sorted(set(self.future_offsets))) != self.future_offsets:
            raise ValueError("future_offsets must be strictly increasing and unique")
        if max(self.future_offsets) > self.action_horizon:
            raise ValueError("future_offsets cannot exceed action_horizon")
        if self.input_mode not in {"full", "current-only", "action-only"}:
            raise ValueError(f"unsupported input_mode={self.input_mode!r}")
        index = self.gripper_dim_index if self.gripper_dim_index >= 0 else self.state_dim + self.gripper_dim_index
        if index < 0 or index >= self.state_dim:
            raise ValueError("gripper_dim_index outside state dimensions")

    @property
    def gripper_index(self) -> int:
        return self.gripper_dim_index if self.gripper_dim_index >= 0 else self.state_dim + self.gripper_dim_index

    @property
    def num_future(self) -> int:
        return len(self.future_offsets)

    @property
    def descriptor_dim(self) -> int:
        # For each adjacent history interval and camera: weighted projected
        # direction, mean projected direction, log-mean energy, log-max energy.
        return (
            (self.history_length - 1)
            * self.num_cameras
            * (2 * self.descriptor_projection_dim + 2)
        )


class TemporalDynamicStateEncoder(nn.Module):
    """Separate low-frequency context from high-value temporal dynamics."""

    def __init__(self, config: DynamicPredictiveWorldConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        h = config.hidden_size
        self.input_proj = nn.Sequential(nn.LayerNorm(config.latent_dim), nn.Linear(config.latent_dim, h))
        self.context_pool = _QueryPool(h, config.num_heads, config.context_tokens, config.dropout)
        self.dynamic_pool = _QueryPool(h, config.num_heads, config.dynamic_tokens, config.dropout)
        self.context_blocks = nn.ModuleList(
            [_SelfAttentionBlock(h, config.num_heads, config.dropout) for _ in range(config.encoder_depth)]
        )
        self.dynamic_blocks = nn.ModuleList(
            [_SelfAttentionBlock(h, config.num_heads, config.dropout) for _ in range(config.encoder_depth)]
        )
        self.camera_embedding = nn.Parameter(torch.randn(1, 1, config.num_cameras, 1, h) * 0.02)
        self.patch_embedding = nn.Parameter(torch.randn(1, 1, 1, config.patches_per_camera, h) * 0.02)
        self.history_embedding = nn.Parameter(torch.randn(1, config.history_length, 1, 1, h) * 0.02)
        self.interval_embedding = nn.Parameter(
            torch.randn(1, config.history_length - 1, 1, 1, h) * 0.02
        )
        # Weak role prior without segmentation or a hand-built object detector.
        role_ids = torch.arange(config.dynamic_tokens) % 3
        self.register_buffer("role_ids", role_ids, persistent=False)
        self.role_embedding = nn.Parameter(torch.randn(3, h) * 0.02)

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        cfg = self.config
        expected = (
            tokens.shape[0],
            cfg.history_length,
            cfg.num_cameras,
            cfg.patches_per_camera,
            cfg.latent_dim,
        )
        if tuple(tokens.shape) != expected:
            raise ValueError(f"history tokens must have shape {expected}, got {tuple(tokens.shape)}")
        x = self.input_proj(tokens)
        positioned = x + self.camera_embedding + self.patch_embedding + self.history_embedding
        current = positioned[:, -1].reshape(tokens.shape[0], -1, cfg.hidden_size)
        context = self.context_pool(current)
        for block in self.context_blocks:
            context = block(context)

        # Differences carry motion; the final frame remains available separately
        # through context and is not itself a reconstruction target.
        diff = x[:, 1:] - x[:, :-1]
        diff = diff + self.camera_embedding[:, :1] + self.patch_embedding[:, :1] + self.interval_embedding
        diff_source = diff.reshape(tokens.shape[0], -1, cfg.hidden_size)
        role_bias = self.role_embedding[self.role_ids].unsqueeze(0).expand(tokens.shape[0], -1, -1)
        dynamic = self.dynamic_pool(diff_source, query_bias=role_bias)
        for block in self.dynamic_blocks:
            dynamic = block(dynamic)
        return context, dynamic


class ActionTrajectoryEncoder(nn.Module):
    """Encode complete action segments aligned to future prediction intervals."""

    def __init__(self, config: DynamicPredictiveWorldConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.input_proj = nn.Sequential(
            nn.Linear(3 * config.action_dim, h), nn.SiLU(), nn.Linear(h, h)
        )
        self.position_embedding = nn.Parameter(torch.randn(1, config.action_horizon, h) * 0.02)
        self.blocks = nn.ModuleList(
            [_SelfAttentionBlock(h, config.num_heads, config.dropout) for _ in range(config.action_depth)]
        )
        self.interval_queries = nn.Parameter(torch.randn(1, config.num_future, 1, h) * 0.02)
        self.interval_attn = nn.MultiheadAttention(h, config.num_heads, batch_first=True)
        self.interval_norm = nn.LayerNorm(h)

    def forward(self, action: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        cfg = self.config
        if tuple(action.shape[1:]) != (cfg.action_horizon, cfg.action_dim):
            raise ValueError("action must be [B,action_horizon,action_dim]")
        if tuple(state.shape[1:]) != (cfg.state_dim,):
            raise ValueError("state must be [B,state_dim]")
        boundary = torch.cat([state[:, None, :], action[:, :-1]], dim=1)
        delta = action - boundary
        relative = action - state[:, None, :]
        x = self.input_proj(torch.cat([action, delta, relative], dim=-1)) + self.position_embedding
        for block in self.blocks:
            x = block(x, causal=True)

        interval_tokens = []
        start = 0
        for interval_idx, end in enumerate(cfg.future_offsets):
            segment = x[:, start:int(end)]
            query = self.interval_queries[:, interval_idx].expand(action.shape[0], -1, -1)
            pooled, _ = self.interval_attn(
                self.interval_norm(query), self.interval_norm(segment), self.interval_norm(segment), need_weights=False
            )
            interval_tokens.append(pooled[:, 0])
            start = int(end)
        return torch.stack(interval_tokens, dim=1), x


class ClosedLoopTransition(nn.Module):
    """One autoregressive update of compact scene and dynamics states.

    The initial visual context remains available as a root condition, but the
    transition also rolls a mutable compact scene state.  This avoids the
    incomplete "dynamic-only loop" in which long-horizon predictions keep
    consulting geometry frozen at time t.
    """

    def __init__(self, config: DynamicPredictiveWorldConfig) -> None:
        super().__init__()
        h = config.hidden_size
        self.dynamic_norm = nn.LayerNorm(h)
        self.scene_norm = nn.LayerNorm(h)
        self.condition_norm = nn.LayerNorm(h)
        self.dynamic_cross = nn.MultiheadAttention(
            h, config.num_heads, dropout=config.dropout, batch_first=True
        )
        self.scene_cross = nn.MultiheadAttention(
            h, config.num_heads, dropout=config.dropout, batch_first=True
        )
        self.dynamic_mask = nn.Sequential(nn.Linear(3 * h, h), nn.SiLU(), nn.Linear(h, 1))
        self.dynamic_value = nn.Sequential(
            nn.Linear(2 * h, 2 * h), nn.SiLU(), nn.Linear(2 * h, h)
        )
        self.scene_mask = nn.Sequential(nn.Linear(3 * h, h), nn.SiLU(), nn.Linear(h, 1))
        self.scene_value = nn.Sequential(
            nn.Linear(2 * h, 2 * h), nn.SiLU(), nn.Linear(2 * h, h)
        )
        self.dynamic_refine = nn.ModuleList(
            [_SelfAttentionBlock(h, config.num_heads, config.dropout) for _ in range(config.predictor_depth)]
        )
        self.scene_refine = nn.ModuleList(
            [_SelfAttentionBlock(h, config.num_heads, config.dropout) for _ in range(config.predictor_depth)]
        )

    def forward(
        self,
        dynamic: Tensor,
        scene: Tensor,
        root_context: Tensor,
        action_token: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        action_seq = action_token[:, None, :]
        dynamic_condition = torch.cat([root_context, scene, action_seq], dim=1)
        dynamic_cross, _ = self.dynamic_cross(
            self.dynamic_norm(dynamic),
            self.condition_norm(dynamic_condition),
            self.condition_norm(dynamic_condition),
            need_weights=False,
        )
        action_dynamic = action_token[:, None, :].expand_as(dynamic)
        scene_mean = scene.mean(dim=1, keepdim=True).expand_as(dynamic)
        dynamic_gate = torch.sigmoid(
            self.dynamic_mask(torch.cat([dynamic, action_dynamic, scene_mean], dim=-1))
        )
        dynamic_effect = self.dynamic_value(torch.cat([dynamic_cross, action_dynamic], dim=-1))
        next_dynamic = dynamic + dynamic_gate * dynamic_effect
        for block in self.dynamic_refine:
            next_dynamic = block(next_dynamic)

        scene_condition = torch.cat([root_context, next_dynamic, action_seq], dim=1)
        scene_cross, _ = self.scene_cross(
            self.scene_norm(scene),
            self.condition_norm(scene_condition),
            self.condition_norm(scene_condition),
            need_weights=False,
        )
        action_scene = action_token[:, None, :].expand_as(scene)
        dynamic_mean = next_dynamic.mean(dim=1, keepdim=True).expand_as(scene)
        scene_gate = torch.sigmoid(
            self.scene_mask(torch.cat([scene, action_scene, dynamic_mean], dim=-1))
        )
        scene_effect = self.scene_value(torch.cat([scene_cross, action_scene], dim=-1))
        next_scene = scene + scene_gate * scene_effect
        for block in self.scene_refine:
            next_scene = block(next_scene)
        return (
            next_dynamic,
            next_scene,
            dynamic_gate,
            dynamic_effect,
            scene_gate,
            scene_effect,
        )


class DynamicPredictiveWorld(nn.Module):
    """Standalone predictor over one frozen temporal visual representation."""

    def __init__(self, config: DynamicPredictiveWorldConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.online_encoder = TemporalDynamicStateEncoder(config)
        self.target_encoder = deepcopy(self.online_encoder)
        self.target_encoder.requires_grad_(False)
        self.target_encoder.eval()
        self.action_encoder = ActionTrajectoryEncoder(config)
        self.transition = ClosedLoopTransition(config)
        h = config.hidden_size
        self.state_token = nn.Sequential(nn.Linear(config.state_dim, h), nn.SiLU(), nn.Linear(h, h))
        self.null_context = nn.Parameter(torch.zeros(1, config.context_tokens, h))
        self.null_dynamic = nn.Parameter(torch.zeros(1, config.dynamic_tokens, h))
        nn.init.normal_(self.null_context, std=0.02)
        nn.init.normal_(self.null_dynamic, std=0.02)

        self.descriptor_head = nn.Sequential(
            nn.LayerNorm(h), nn.Linear(h, 2 * h), nn.SiLU(), nn.Linear(2 * h, config.descriptor_dim)
        )
        self.local_motion_head = nn.Sequential(
            nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, config.state_dim)
        )
        self.context_state_head = nn.Sequential(
            nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, config.state_dim)
        )
        self.state_fusion = nn.Sequential(
            nn.LayerNorm(2 * h), nn.Linear(2 * h, h), nn.SiLU()
        )
        self.state_path_head = nn.Sequential(
            nn.LayerNorm(h), nn.Linear(h, 2 * h), nn.SiLU(), nn.Linear(2 * h, config.state_dim)
        )

        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.descriptor_seed)
        matrix = torch.randn(config.latent_dim, config.descriptor_projection_dim, generator=generator)
        q, _ = torch.linalg.qr(matrix, mode="reduced")
        self.register_buffer("descriptor_projection", q.float(), persistent=True)
        self.representation_frozen = False

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_encoder.eval()
        if self.representation_frozen:
            self.online_encoder.eval()
            self.descriptor_head.eval()
            self.local_motion_head.eval()
            self.context_state_head.eval()
        return self

    def representation_outputs(self, tokens: Tensor) -> dict[str, Tensor]:
        context, dynamic = self.online_encoder(tokens)
        return {
            "context": context,
            "dynamic": dynamic,
            "descriptor": self.descriptor_prediction(dynamic),
            "local_motion": self.local_motion_head(dynamic.mean(dim=1)),
            "context_state": self.context_state_head(context.mean(dim=1)),
        }

    def representation_state_dict(self) -> dict[str, dict[str, Tensor] | Tensor]:
        return {
            "online_encoder": self.online_encoder.state_dict(),
            "descriptor_head": self.descriptor_head.state_dict(),
            "local_motion_head": self.local_motion_head.state_dict(),
            "context_state_head": self.context_state_head.state_dict(),
            "descriptor_projection": self.descriptor_projection.detach().cpu(),
        }

    def load_representation_state_dict(self, state: dict[str, object], *, freeze: bool = True) -> None:
        self.online_encoder.load_state_dict(state["online_encoder"], strict=True)
        self.descriptor_head.load_state_dict(state["descriptor_head"], strict=True)
        self.local_motion_head.load_state_dict(state["local_motion_head"], strict=True)
        self.context_state_head.load_state_dict(state["context_state_head"], strict=True)
        projection = torch.as_tensor(state["descriptor_projection"], dtype=self.descriptor_projection.dtype)
        if projection.shape != self.descriptor_projection.shape:
            raise ValueError("representation descriptor projection shape mismatch")
        self.descriptor_projection.copy_(projection.to(self.descriptor_projection.device))
        self.target_encoder.load_state_dict(self.online_encoder.state_dict(), strict=True)
        if freeze:
            self.freeze_representation()

    def freeze_representation(self) -> None:
        modules = [
            self.online_encoder,
            self.target_encoder,
            self.descriptor_head,
            self.local_motion_head,
            self.context_state_head,
        ]
        for module in modules:
            module.requires_grad_(False)
            module.eval()
        self.representation_frozen = True

    def fixed_dynamic_descriptor(self, tokens: Tensor) -> Tensor:
        cfg = self.config
        if tokens.shape[-1] != cfg.latent_dim or tokens.shape[1] != cfg.history_length:
            raise ValueError("descriptor tokens violate configured history/latent dimensions")
        diff = tokens.float()[:, 1:] - tokens.float()[:, :-1]
        projected = diff @ self.descriptor_projection.float()  # [B,I,C,P,R]
        energy = diff.square().mean(dim=-1).sqrt().clamp_min(1e-8)
        weights = torch.softmax(energy / 0.1, dim=-1)
        weighted = (projected * weights[..., None]).sum(dim=-2)
        mean = projected.mean(dim=-2)
        weighted = F.normalize(weighted, dim=-1)
        mean = F.normalize(mean, dim=-1)
        mean_energy = torch.log1p(energy.mean(dim=-1))[..., None]
        max_energy = torch.log1p(energy.max(dim=-1).values)[..., None]
        descriptor = torch.cat([weighted, mean, mean_energy, max_energy], dim=-1)
        return descriptor.reshape(tokens.shape[0], -1)

    def _append_state_context(self, context: Tensor, state: Tensor) -> Tensor:
        return torch.cat([context, self.state_token(state)[:, None, :]], dim=1)

    def encode_current(
        self, current_tokens: Tensor, state: Tensor, *, mode_override: str | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        cfg = self.config
        mode = cfg.input_mode if mode_override is None else str(mode_override)
        if mode not in {"full", "current-only", "action-only"}:
            raise ValueError(f"unsupported mode_override={mode!r}")
        batch = current_tokens.shape[0]
        if mode == "action-only":
            scene = self.null_context.expand(batch, -1, -1)
            dynamic = self.null_dynamic.expand(batch, -1, -1)
        else:
            scene, dynamic = self.online_encoder(current_tokens)
        root_context = self._append_state_context(scene, state)
        return root_context, scene, dynamic

    @torch.no_grad()
    def encode_future_targets(self, target_tokens: Tensor) -> tuple[Tensor, Tensor]:
        cfg = self.config
        batch = target_tokens.shape[0]
        flat = target_tokens.reshape(
            batch * cfg.num_future,
            cfg.history_length,
            cfg.num_cameras,
            cfg.patches_per_camera,
            cfg.latent_dim,
        )
        target_scene, target_dynamic = self.target_encoder(flat)
        target_scene = target_scene.reshape(
            batch, cfg.num_future, cfg.context_tokens, cfg.hidden_size
        )
        target_dynamic = target_dynamic.reshape(
            batch, cfg.num_future, cfg.dynamic_tokens, cfg.hidden_size
        )
        return target_scene, target_dynamic

    @torch.no_grad()
    def encode_targets(
        self, current_tokens: Tensor, target_tokens: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        target_initial_scene, target_initial_dynamic = self.target_encoder(current_tokens)
        target_scene, target_dynamic = self.encode_future_targets(target_tokens)
        return target_initial_scene, target_initial_dynamic, target_scene, target_dynamic

    def encode_online_future(self, target_tokens: Tensor) -> tuple[Tensor, Tensor]:
        cfg = self.config
        batch = target_tokens.shape[0]
        flat = target_tokens.reshape(
            batch * cfg.num_future,
            cfg.history_length,
            cfg.num_cameras,
            cfg.patches_per_camera,
            cfg.latent_dim,
        )
        scene, dynamic = self.online_encoder(flat)
        return (
            scene.reshape(batch, cfg.num_future, cfg.context_tokens, cfg.hidden_size),
            dynamic.reshape(batch, cfg.num_future, cfg.dynamic_tokens, cfg.hidden_size),
        )

    def rollout_from_encoded(
        self,
        root_context: Tensor,
        initial_scene: Tensor,
        initial_dynamic: Tensor,
        action: Tensor,
        state: Tensor,
        *,
        mode_override: str | None = None,
    ) -> dict[str, Tensor]:
        cfg = self.config
        mode = cfg.input_mode if mode_override is None else str(mode_override)
        interval_action, action_steps = self.action_encoder(action, state)
        if mode == "current-only":
            interval_action = torch.zeros_like(interval_action)
        dynamic = initial_dynamic
        scene = initial_scene
        dynamics, scenes = [], []
        dynamic_gates, dynamic_effects = [], []
        scene_gates, scene_effects = [], []
        for step in range(cfg.num_future):
            (
                dynamic,
                scene,
                dynamic_gate,
                dynamic_effect,
                scene_gate,
                scene_effect,
            ) = self.transition(dynamic, scene, root_context, interval_action[:, step])
            dynamics.append(dynamic)
            scenes.append(scene)
            dynamic_gates.append(dynamic_gate)
            dynamic_effects.append(dynamic_effect)
            scene_gates.append(scene_gate)
            scene_effects.append(scene_effect)
        return {
            "pred_dynamic": torch.stack(dynamics, dim=1),
            "pred_scene": torch.stack(scenes, dim=1),
            "effect_gate": torch.stack(dynamic_gates, dim=1),
            "effect_value": torch.stack(dynamic_effects, dim=1),
            "scene_effect_gate": torch.stack(scene_gates, dim=1),
            "scene_effect_value": torch.stack(scene_effects, dim=1),
            "interval_action": interval_action,
            "action_steps": action_steps,
        }

    def teacher_forced_steps(
        self,
        root_context: Tensor,
        initial_scene: Tensor,
        initial_dynamic: Tensor,
        target_scene: Tensor,
        target_dynamic: Tensor,
        action: Tensor,
        state: Tensor,
        *,
        mode_override: str | None = None,
    ) -> tuple[Tensor, Tensor]:
        mode = self.config.input_mode if mode_override is None else str(mode_override)
        interval_action, _ = self.action_encoder(action, state)
        if mode == "current-only":
            interval_action = torch.zeros_like(interval_action)
        previous_dynamic = initial_dynamic
        previous_scene = initial_scene
        dynamic_rows, scene_rows = [], []
        for step in range(self.config.num_future):
            pred_dynamic, pred_scene, *_ = self.transition(
                previous_dynamic, previous_scene, root_context, interval_action[:, step]
            )
            dynamic_rows.append(pred_dynamic)
            scene_rows.append(pred_scene)
            previous_dynamic = target_dynamic[:, step].detach()
            previous_scene = target_scene[:, step].detach()
        return torch.stack(dynamic_rows, dim=1), torch.stack(scene_rows, dim=1)

    def _interpolated_path_features(
        self,
        initial_scene: Tensor,
        initial_dynamic: Tensor,
        pred_scene: Tensor,
        pred_dynamic: Tensor,
    ) -> Tensor:
        cfg = self.config
        initial_feature = self.state_fusion(
            torch.cat([initial_scene.mean(dim=1), initial_dynamic.mean(dim=1)], dim=-1)
        )
        future_feature = self.state_fusion(
            torch.cat([pred_scene.mean(dim=2), pred_dynamic.mean(dim=2)], dim=-1)
        )
        anchors = torch.cat([initial_feature[:, None], future_feature], dim=1)
        anchor_steps = [0, *cfg.future_offsets]
        rows = []
        for step in range(1, cfg.action_horizon + 1):
            right = next(i for i, value in enumerate(anchor_steps[1:], start=1) if step <= value)
            left = right - 1
            denom = max(anchor_steps[right] - anchor_steps[left], 1)
            alpha = float(step - anchor_steps[left]) / float(denom)
            rows.append((1.0 - alpha) * anchors[:, left] + alpha * anchors[:, right])
        return torch.stack(rows, dim=1)

    def decode_state_path(
        self,
        state: Tensor,
        initial_scene: Tensor,
        initial_dynamic: Tensor,
        pred_scene: Tensor,
        pred_dynamic: Tensor,
    ) -> Tensor:
        features = self._interpolated_path_features(
            initial_scene, initial_dynamic, pred_scene, pred_dynamic
        )
        delta = self.state_path_head(features)
        return state[:, None, :] + delta

    def descriptor_prediction(self, dynamic: Tensor) -> Tensor:
        if dynamic.ndim == 3:
            return self.descriptor_head(dynamic.mean(dim=1))
        if dynamic.ndim == 4:
            shape = dynamic.shape[:2]
            out = self.descriptor_head(dynamic.mean(dim=2))
            return out.reshape(*shape, -1)
        raise ValueError("dynamic must be [B,Q,H] or [B,K,Q,H]")

    def forward(
        self,
        current_tokens: Tensor,
        target_tokens: Tensor,
        state: Tensor,
        action: Tensor,
        *,
        mode_override: str | None = None,
    ) -> dict[str, Tensor]:
        cfg = self.config
        root_context, initial_scene, initial_dynamic = self.encode_current(
            current_tokens, state, mode_override=mode_override
        )
        (
            target_initial_scene,
            target_initial_dynamic,
            target_scene,
            target_dynamic,
        ) = self.encode_targets(current_tokens, target_tokens)
        rollout = self.rollout_from_encoded(
            root_context,
            initial_scene,
            initial_dynamic,
            action,
            state,
            mode_override=mode_override,
        )
        teacher_forced_dynamic, teacher_forced_scene = self.teacher_forced_steps(
            root_context,
            target_initial_scene,
            target_initial_dynamic,
            target_scene,
            target_dynamic,
            action,
            state,
            mode_override=mode_override,
        )
        pred_dynamic = rollout["pred_dynamic"]
        pred_scene = rollout["pred_scene"]
        state_path = self.decode_state_path(
            state, initial_scene, initial_dynamic, pred_scene, pred_dynamic
        )
        return {
            **rollout,
            "context": root_context,
            "initial_scene": initial_scene,
            "initial_dynamic": initial_dynamic,
            "target_initial_scene": target_initial_scene,
            "target_initial_dynamic": target_initial_dynamic,
            "target_scene": target_scene,
            "target_dynamic": target_dynamic,
            "teacher_forced_scene": teacher_forced_scene,
            "teacher_forced_dynamic": teacher_forced_dynamic,
            "pred_descriptor": self.descriptor_prediction(pred_dynamic),
            "initial_descriptor": self.descriptor_prediction(initial_dynamic),
            "target_descriptor": torch.stack(
                [self.fixed_dynamic_descriptor(target_tokens[:, k]) for k in range(cfg.num_future)],
                dim=1,
            ).to(dtype=pred_dynamic.dtype),
            "current_descriptor": self.fixed_dynamic_descriptor(current_tokens).to(
                dtype=pred_dynamic.dtype
            ),
            "pred_state_path": state_path,
        }

    def forward_local_pair(
        self, current_tokens: Tensor, target_tokens: Tensor, state: Tensor, action: Tensor
    ) -> dict[str, Tensor]:
        """Minimal pair path for real cross-episode local-effect supervision."""
        root_context, initial_scene, initial_dynamic = self.encode_current(current_tokens, state)
        target_initial_scene, _, target_scene, target_dynamic = self.encode_targets(
            current_tokens, target_tokens
        )
        rollout = self.rollout_from_encoded(
            root_context, initial_scene, initial_dynamic, action, state
        )
        return {
            "pred_dynamic": rollout["pred_dynamic"],
            "pred_scene": rollout["pred_scene"],
            "target_dynamic": target_dynamic,
            "target_scene": target_scene,
            "initial_scene": initial_scene,
            "target_initial_scene": target_initial_scene,
        }

    def swapped_action_rollout(
        self,
        current_tokens: Tensor,
        state: Tensor,
        swapped_action: Tensor,
    ) -> dict[str, Tensor]:
        """Roll the same observed state with a nearby sample's action trajectory."""
        root_context, initial_scene, initial_dynamic = self.encode_current(current_tokens, state)
        return self.rollout_from_encoded(
            root_context, initial_scene, initial_dynamic, swapped_action, state
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


__all__ = ["DynamicPredictiveWorldConfig", "DynamicPredictiveWorld"]
