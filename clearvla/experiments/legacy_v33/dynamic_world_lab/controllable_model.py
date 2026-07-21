from __future__ import annotations

"""V33.6 joint-AdaLN controllable world model.

This module keeps the full scene/dynamics token capacity of V33.4, but removes
its easiest shortcut: a single transition that can learn the entire future
without action.  V33.6 separates an action-free prior from an action-induced
world residual while preserving full-resolution world tokens throughout.

The action path has no world-only value route.  It receives counterfactual
action features (actual trajectory minus a hold-state trajectory encoded by the
same network), and its cross-attention values come only from those features.
Consequently a hold trajectory produces an exactly zero action effect.  World
state still controls *where* the action acts through the attention queries.

An action-independent, full-width residual adapter aligns the frozen Stage-A
representation with controllability objectives.  It never reduces token count
or hidden width.  A slowly moving EMA copy defines prediction targets during
optional alignment, avoiding a rapidly moving target space.
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .model import (
    DynamicPredictiveWorld,
    DynamicPredictiveWorldConfig,
    _FeedForward,
    _SelfAttentionBlock,
)


@dataclass(frozen=True)
class ControllableWorldConfig(DynamicPredictiveWorldConfig):
    adapter_depth: int = 2
    inverse_depth: int = 2
    prior_depth: int = 3
    effect_depth: int = 3
    adapter_layer_scale: float = 0.02
    prior_layer_scale: float = 0.10
    effect_layer_scale: float = 1.0
    inverse_gripper_classes: int = 3

    def validate(self) -> None:
        super().validate()
        if min(self.adapter_depth, self.inverse_depth, self.prior_depth, self.effect_depth) <= 0:
            raise ValueError("V33.6 depths must be positive")
        for name, value in (
            ("adapter_layer_scale", self.adapter_layer_scale),
            ("prior_layer_scale", self.prior_layer_scale),
            ("effect_layer_scale", self.effect_layer_scale),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.inverse_gripper_classes != 3:
            raise ValueError("inverse_gripper_classes must be 3: hold/open/close")


class FullWorldAdapter(nn.Module):
    """Identity-initialized, full-capacity world representation adapter.

    Scene and dynamics token counts and hidden width remain unchanged.  The
    adapter only adds a small residual after joint self-attention over all world
    tokens.  This creates a controllability-alignment surface without forcing
    the representation through a low-dimensional action bottleneck.
    """

    def __init__(self, config: ControllableWorldConfig) -> None:
        super().__init__()
        h = config.hidden_size
        self.context_tokens = config.context_tokens
        self.dynamic_tokens = config.dynamic_tokens
        self.scene_type = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        self.dynamic_type = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        self.blocks = nn.ModuleList(
            [
                _SelfAttentionBlock(h, config.num_heads, config.dropout)
                for _ in range(config.adapter_depth)
            ]
        )
        self.norm = nn.LayerNorm(h)
        self.out = nn.Linear(h, h, bias=False)
        self.layer_scale = nn.Parameter(torch.full((h,), float(config.adapter_layer_scale)))

    def forward(self, scene: Tensor, dynamic: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if scene.ndim != 3 or dynamic.ndim != 3 or scene.shape[0] != dynamic.shape[0]:
            raise ValueError("scene and dynamic must be [B,Q,H] with the same batch")
        base = torch.cat([scene, dynamic], dim=1)
        typed = torch.cat([scene + self.scene_type, dynamic + self.dynamic_type], dim=1)
        x = typed
        for block in self.blocks:
            x = block(x)
        delta = self.out(self.norm(x)) * self.layer_scale
        adapted = base + delta
        return (
            adapted[:, : self.context_tokens],
            adapted[:, self.context_tokens :],
            delta,
        )


class CounterfactualActionEncoder(nn.Module):
    """Encode all 48 action steps without interval compression.

    The returned effect tokens are the shared encoder response to the actual
    trajectory minus its response to a hold-current-state trajectory.  This
    makes the no-action reference exact and blocks constant/action-independent
    information from entering the action-effect values.
    """

    def __init__(self, config: ControllableWorldConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.input_proj = nn.Sequential(
            nn.Linear(4 * config.action_dim, 2 * h),
            nn.SiLU(),
            nn.Linear(2 * h, h),
        )
        self.position_embedding = nn.Parameter(torch.randn(1, config.action_horizon, h) * 0.02)
        # Dropout is deliberately disabled here.  Exact actual-minus-hold
        # cancellation is a structural anti-shortcut contract.
        self.blocks = nn.ModuleList(
            [_SelfAttentionBlock(h, config.num_heads, 0.0) for _ in range(config.action_depth)]
        )
        self.norm = nn.LayerNorm(h)

    def _features(self, action: Tensor, action_state: Tensor) -> Tensor:
        boundary = torch.cat([action_state[:, None], action[:, :-1]], dim=1)
        velocity = action - boundary
        previous_velocity = torch.cat([torch.zeros_like(velocity[:, :1]), velocity[:, :-1]], dim=1)
        acceleration = velocity - previous_velocity
        relative = action - action_state[:, None]
        return torch.cat([action, velocity, acceleration, relative], dim=-1)

    def _encode(self, action: Tensor, action_state: Tensor) -> Tensor:
        x = self.input_proj(self._features(action, action_state)) + self.position_embedding
        for block in self.blocks:
            x = block(x, causal=True)
        return self.norm(x)

    def forward(self, action: Tensor, action_state: Tensor) -> dict[str, Tensor]:
        cfg = self.config
        if tuple(action.shape[1:]) != (cfg.action_horizon, cfg.action_dim):
            raise ValueError("action must be [B,action_horizon,action_dim]")
        if tuple(action_state.shape[1:]) != (cfg.state_dim,):
            raise ValueError("action_state must be [B,state_dim] in action-normalized coordinates")
        hold = action_state[:, None].expand(-1, cfg.action_horizon, -1)
        stacked_action = torch.cat([action, hold], dim=0)
        stacked_state = torch.cat([action_state, action_state], dim=0)
        encoded = self._encode(stacked_action, stacked_state)
        actual_steps, hold_steps = encoded.chunk(2, dim=0)
        effect_steps = actual_steps - hold_steps
        exact_hold = (action == hold).all(dim=(1, 2), keepdim=True)
        effect_steps = torch.where(exact_hold, torch.zeros_like(effect_steps), effect_steps)
        interval_rows = []
        start = 0
        for end in cfg.future_offsets:
            interval_rows.append(effect_steps[:, start : int(end)].mean(dim=1))
            start = int(end)
        return {
            "effect_steps": effect_steps,
            "actual_steps": actual_steps,
            "hold_steps": hold_steps,
            "interval_action": torch.stack(interval_rows, dim=1),
        }


class PriorWorldTransition(nn.Module):
    """Action-free autoregressive prior over the complete world token set."""

    def __init__(self, config: ControllableWorldConfig) -> None:
        super().__init__()
        h = config.hidden_size
        self.context_tokens = config.context_tokens
        self.scene_type = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        self.dynamic_type = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        self.query_norm = nn.LayerNorm(h)
        self.memory_norm = nn.LayerNorm(h)
        self.cross = nn.MultiheadAttention(
            h, config.num_heads, dropout=config.dropout, batch_first=True
        )
        self.blocks = nn.ModuleList(
            [
                _SelfAttentionBlock(h, config.num_heads, config.dropout)
                for _ in range(config.prior_depth)
            ]
        )
        self.out_norm = nn.LayerNorm(h)
        self.out = nn.Linear(h, h, bias=False)
        self.layer_scale = nn.Parameter(torch.full((h,), float(config.prior_layer_scale)))

    def forward(
        self, scene: Tensor, dynamic: Tensor, root_context: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        base = torch.cat([scene, dynamic], dim=1)
        typed = torch.cat([scene + self.scene_type, dynamic + self.dynamic_type], dim=1)
        memory = torch.cat([root_context, typed], dim=1)
        cross, _ = self.cross(
            self.query_norm(typed),
            self.memory_norm(memory),
            self.memory_norm(memory),
            need_weights=False,
        )
        x = typed + cross
        for block in self.blocks:
            x = block(x)
        delta = self.out(self.out_norm(x)) * self.layer_scale
        next_world = base + delta
        return (
            next_world[:, : self.context_tokens],
            next_world[:, self.context_tokens :],
            delta,
        )


class _JointWorldActionAdaLNZeroBlock(nn.Module):
    """One full-width bidirectional world/action reasoning block.

    The block is deliberately constructed so neither side can create a world
    update on its own:

    * action tokens first read the complete world, but that update is gated by
      a full-width bilinear product of action content and the world readout;
    * world tokens then read the world-conditioned action tokens;
    * AdaLN shift/scale/gates are generated from a bilinear product of the
      action-derived signal and the current world content;
    * the modulation projection is zero-initialized (AdaLN-Zero).

    Consequently an exact hold action (all counterfactual action tokens are
    zero) yields an exact zero residual at every depth, while a zero world
    content tensor also cannot be driven by action alone.
    """

    def __init__(self, hidden: int, heads: int, expansion: float = 4.0) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.world_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.action_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.action_reads_world = nn.MultiheadAttention(
            hidden, heads, dropout=0.0, batch_first=True, bias=False
        )
        self.action_self_factor = nn.Linear(hidden, 2 * hidden, bias=False)
        self.action_world_factor = nn.Linear(hidden, 2 * hidden, bias=False)
        self.action_read_out = nn.Linear(2 * hidden, hidden, bias=False)

        self.world_reads_action = nn.MultiheadAttention(
            hidden, heads, dropout=0.0, batch_first=True, bias=False
        )
        self.world_factor = nn.Linear(hidden, 2 * hidden, bias=False)
        self.action_factor = nn.Linear(hidden, 2 * hidden, bias=False)
        self.modulation = nn.Linear(2 * hidden, 6 * hidden, bias=False)
        nn.init.zeros_(self.modulation.weight)

        self.attn_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(
            hidden, heads, dropout=0.0, batch_first=True, bias=False
        )
        self.ffn_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        inner = int(round(hidden * float(expansion)))
        self.ffn = nn.Sequential(
            nn.Linear(hidden, inner),
            nn.GELU(),
            nn.Linear(inner, hidden),
        )

    @staticmethod
    def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        return x * (1.0 + scale) + shift

    def forward(
        self,
        *,
        world_content: Tensor,
        typed_world: Tensor,
        root_bias: Tensor,
        action_tokens: Tensor,
        effect: Tensor,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        # The action stream reads the full world at every depth.  A zero action
        # token cannot absorb world-only information because the update is a
        # full-width product with the counterfactual action token itself.
        world_hidden = typed_world + effect
        action_norm = self.action_norm(action_tokens)
        world_memory = self.world_norm(world_hidden)
        action_read, _ = self.action_reads_world(
            action_norm, world_memory, world_hidden, need_weights=False
        )
        action_world_joint = self.action_self_factor(action_norm) * self.action_world_factor(
            action_read
        )
        action_update = self.action_read_out(F.silu(action_world_joint))
        action_tokens = action_tokens + action_update

        # The world now reads action tokens that have already been interpreted
        # in the current world.  Values are still exclusively action-derived.
        world_query_content = world_content + effect
        world_query = self.world_norm(world_query_content + root_bias)
        action_memory = self.action_norm(action_tokens)
        action_signal, _ = self.world_reads_action(
            world_query, action_memory, action_tokens, need_weights=False
        )

        # Joint, per-token conditioner.  This is the critical anti-shortcut
        # product: if either current world content or action signal is zero, all
        # AdaLN parameters and gates remain exactly zero.
        joint = self.world_factor(world_query_content) * self.action_factor(action_signal)
        modulation = self.modulation(F.silu(joint))
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = modulation.chunk(
            6, dim=-1
        )

        hidden = typed_world + effect
        attn_input = self._modulate(self.attn_norm(hidden), shift_attn, scale_attn)
        attn_out, _ = self.self_attn(attn_input, attn_input, attn_input, need_weights=False)
        effect = effect + torch.tanh(gate_attn) * attn_out

        hidden = typed_world + effect
        ffn_input = self._modulate(self.ffn_norm(hidden), shift_ffn, scale_ffn)
        effect = effect + torch.tanh(gate_ffn) * self.ffn(ffn_input)

        diagnostics = {
            "gate_abs_mean": 0.5
            * (torch.tanh(gate_attn).abs().mean() + torch.tanh(gate_ffn).abs().mean()),
            "scale_abs_mean": 0.5 * (scale_attn.abs().mean() + scale_ffn.abs().mean()),
            "shift_abs_mean": 0.5 * (shift_attn.abs().mean() + shift_ffn.abs().mean()),
            "action_read_gate_abs_mean": action_world_joint.float().square().mean().sqrt(),
            "joint_rms": joint.float().square().mean().sqrt(),
            "action_signal_rms": action_signal.float().square().mean().sqrt(),
        }
        return effect, action_tokens, diagnostics


class ActionWorldEffect(nn.Module):
    """Multi-layer, full-width joint world/action AdaLN-Zero effect model.

    This restores deep latent conditioning without restoring the V33.4
    shortcut.  The action stream cannot predict a future world directly: it is
    reinterpreted by the current complete world at every block and can only
    generate AdaLN modulation through a bilinear world/action conditioner.
    The prior remains a separate frozen action-free model.
    """

    def __init__(self, config: ControllableWorldConfig) -> None:
        super().__init__()
        h = config.hidden_size
        self.context_tokens = config.context_tokens
        self.scene_type = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        self.dynamic_type = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        self.root_proj = nn.Linear(h, h, bias=False)
        self.blocks = nn.ModuleList(
            [
                _JointWorldActionAdaLNZeroBlock(h, config.num_heads)
                for _ in range(config.effect_depth)
            ]
        )
        self.effect_out = nn.Linear(h, h, bias=False)
        self.scene_scale = nn.Parameter(torch.full((h,), float(config.effect_layer_scale)))
        self.dynamic_scale = nn.Parameter(torch.full((h,), float(config.effect_layer_scale)))

    def forward(
        self,
        scene: Tensor,
        dynamic: Tensor,
        root_context: Tensor,
        action_prefix: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        if action_prefix.ndim != 3 or action_prefix.shape[1] == 0:
            raise ValueError("action_prefix must be non-empty [B,T,H]")
        world_content = torch.cat([scene, dynamic], dim=1)
        typed_world = torch.cat([scene + self.scene_type, dynamic + self.dynamic_type], dim=1)
        root_bias = self.root_proj(root_context.mean(dim=1, keepdim=True))
        effect = torch.zeros_like(world_content)
        action_tokens = action_prefix
        diagnostic_rows: dict[str, list[Tensor]] = {}
        for block in self.blocks:
            effect, action_tokens, diagnostics = block(
                world_content=world_content,
                typed_world=typed_world,
                root_bias=root_bias,
                action_tokens=action_tokens,
                effect=effect,
            )
            for key, value in diagnostics.items():
                diagnostic_rows.setdefault(key, []).append(value)

        effect = self.effect_out(effect)
        scene_effect = effect[:, : self.context_tokens] * self.scene_scale
        dynamic_effect = effect[:, self.context_tokens :] * self.dynamic_scale
        diagnostics = {key: torch.stack(values).mean() for key, values in diagnostic_rows.items()}
        diagnostics["effect_rms"] = effect.float().square().mean().sqrt()
        return scene_effect, dynamic_effect, effect, diagnostics


class WorldActionInverseDecoder(nn.Module):
    """Decode relative action from full world change, not from task phase.

    Learned action-time queries attend to keys containing world-change and time,
    while values contain world-change only.  A full-width bilinear interaction
    with the current world keeps the mapping state-dependent.  Therefore a
    static future cannot be decoded into a non-zero relative action through a
    current-state-only shortcut.
    """

    def __init__(self, config: ControllableWorldConfig) -> None:
        super().__init__()
        h = config.hidden_size
        self.config = config
        self.scene_type = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        self.dynamic_type = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        self.time_embedding = nn.Parameter(torch.randn(1, config.num_future, 1, h) * 0.02)
        self.queries = nn.Parameter(torch.randn(1, config.action_horizon, h) * 0.02)
        self.query_proj = nn.Linear(h, h, bias=False)
        self.key_proj = nn.Linear(h, h, bias=False)
        self.value_proj = nn.Linear(h, h, bias=False)
        self.current_proj = nn.Linear(h, 2 * h, bias=False)
        self.change_proj = nn.Linear(h, 2 * h, bias=False)
        self.fuse_out = nn.Linear(2 * h, h, bias=False)
        self.blocks = nn.ModuleList(
            [_SelfAttentionBlock(h, config.num_heads, 0.0) for _ in range(config.inverse_depth)]
        )
        self.action_head = nn.Sequential(
            nn.LayerNorm(h), nn.Linear(h, config.action_dim, bias=False)
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(h), nn.Linear(h, config.action_dim, bias=False)
        )
        self.gripper_head = nn.Sequential(
            nn.LayerNorm(h), nn.Linear(h, config.inverse_gripper_classes)
        )

    def forward(
        self,
        current_scene: Tensor,
        current_dynamic: Tensor,
        future_scene: Tensor,
        future_dynamic: Tensor,
    ) -> dict[str, Tensor]:
        current_world = torch.cat(
            [current_scene + self.scene_type, current_dynamic + self.dynamic_type], dim=1
        )
        delta_rows = []
        key_rows = []
        for index in range(self.config.num_future):
            future_world = torch.cat(
                [
                    future_scene[:, index] + self.scene_type,
                    future_dynamic[:, index] + self.dynamic_type,
                ],
                dim=1,
            )
            delta = future_world - current_world
            delta_rows.append(delta)
            key_rows.append(delta + self.time_embedding[:, index])
        value_memory = torch.cat(delta_rows, dim=1)
        key_memory = torch.cat(key_rows, dim=1)
        query = self.query_proj(self.queries.expand(current_world.shape[0], -1, -1))
        key = self.key_proj(key_memory)
        value = self.value_proj(value_memory)
        scale = float(query.shape[-1]) ** -0.5
        attention = torch.softmax(torch.matmul(query, key.transpose(-1, -2)) * scale, dim=-1)
        change = torch.matmul(attention, value)
        current_summary = current_world.mean(dim=1, keepdim=True).expand_as(change)
        x = self.fuse_out(F.gelu(self.change_proj(change) * self.current_proj(current_summary)))
        for block in self.blocks:
            x = block(x) - block(torch.zeros_like(x))
        return {
            "inverse_action": self.action_head(x),
            "inverse_delta": self.delta_head(x),
            "inverse_gripper_logits": self.gripper_head(x),
        }


class ControllableDynamicWorld(DynamicPredictiveWorld):
    """Staged prior/residual/alignment world model for V33.6."""

    def __init__(self, config: ControllableWorldConfig) -> None:
        config.validate()
        super().__init__(config)
        self.config = config
        # Remove the V33.4 mixed transition and replace it with explicitly
        # separated responsibilities.
        self.action_encoder = CounterfactualActionEncoder(config)
        del self.transition
        self.online_adapter = FullWorldAdapter(config)
        self.target_adapter = deepcopy(self.online_adapter)
        self.target_adapter.requires_grad_(False)
        self.target_adapter.eval()
        self.prior_transition = PriorWorldTransition(config)
        self.action_effect = ActionWorldEffect(config)
        self.inverse_decoder = WorldActionInverseDecoder(config)
        self.training_phase: Literal["prior", "effect", "align", "eval"] = "eval"
        self._unfreeze_dynamic_blocks = 0

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_adapter.eval()
        if mode and self.training_phase == "align":
            self.online_adapter.train(True)
            count = max(
                0, min(self._unfreeze_dynamic_blocks, len(self.online_encoder.dynamic_blocks))
            )
            if count:
                self.online_encoder.dynamic_pool.train(True)
                for block in self.online_encoder.dynamic_blocks[-count:]:
                    block.train(True)
        return self

    def _adapt(self, scene: Tensor, dynamic: Tensor, *, target: bool = False):
        adapter = self.target_adapter if target else self.online_adapter
        return adapter(scene, dynamic)

    def encode_current_world(
        self, current_tokens: Tensor, state: Tensor, *, mode_override: str | None = None
    ) -> dict[str, Tensor]:
        mode = self.config.input_mode if mode_override is None else str(mode_override)
        batch = current_tokens.shape[0]
        if mode == "action-only":
            base_scene = self.null_context.expand(batch, -1, -1)
            base_dynamic = self.null_dynamic.expand(batch, -1, -1)
        else:
            base_scene, base_dynamic = self.online_encoder(current_tokens)
        scene, dynamic, adapter_delta = self._adapt(base_scene, base_dynamic, target=False)
        root_context = self._append_state_context(scene, state)
        return {
            "root_context": root_context,
            "base_scene": base_scene,
            "base_dynamic": base_dynamic,
            "scene": scene,
            "dynamic": dynamic,
            "adapter_delta": adapter_delta,
        }

    @torch.no_grad()
    def encode_target_world(
        self, current_tokens: Tensor, target_tokens: Tensor
    ) -> dict[str, Tensor]:
        cfg = self.config
        initial_base_scene, initial_base_dynamic = self.target_encoder(current_tokens)
        initial_scene, initial_dynamic, initial_delta = self._adapt(
            initial_base_scene, initial_base_dynamic, target=True
        )
        batch = target_tokens.shape[0]
        flat = target_tokens.reshape(
            batch * cfg.num_future,
            cfg.history_length,
            cfg.num_cameras,
            cfg.patches_per_camera,
            cfg.latent_dim,
        )
        future_base_scene, future_base_dynamic = self.target_encoder(flat)
        future_scene, future_dynamic, future_delta = self._adapt(
            future_base_scene, future_base_dynamic, target=True
        )
        return {
            "target_initial_base_scene": initial_base_scene,
            "target_initial_base_dynamic": initial_base_dynamic,
            "target_initial_scene": initial_scene,
            "target_initial_dynamic": initial_dynamic,
            "target_initial_adapter_delta": initial_delta,
            "target_base_scene": future_base_scene.reshape(
                batch, cfg.num_future, cfg.context_tokens, cfg.hidden_size
            ),
            "target_base_dynamic": future_base_dynamic.reshape(
                batch, cfg.num_future, cfg.dynamic_tokens, cfg.hidden_size
            ),
            "target_scene": future_scene.reshape(
                batch, cfg.num_future, cfg.context_tokens, cfg.hidden_size
            ),
            "target_dynamic": future_dynamic.reshape(
                batch, cfg.num_future, cfg.dynamic_tokens, cfg.hidden_size
            ),
            "target_adapter_delta": future_delta.reshape(
                batch, cfg.num_future, cfg.context_tokens + cfg.dynamic_tokens, cfg.hidden_size
            ),
        }

    def encode_online_future_world(self, target_tokens: Tensor) -> dict[str, Tensor]:
        cfg = self.config
        batch = target_tokens.shape[0]
        flat = target_tokens.reshape(
            batch * cfg.num_future,
            cfg.history_length,
            cfg.num_cameras,
            cfg.patches_per_camera,
            cfg.latent_dim,
        )
        base_scene, base_dynamic = self.online_encoder(flat)
        scene, dynamic, delta = self._adapt(base_scene, base_dynamic, target=False)
        return {
            "online_base_scene": base_scene.reshape(
                batch, cfg.num_future, cfg.context_tokens, cfg.hidden_size
            ),
            "online_base_dynamic": base_dynamic.reshape(
                batch, cfg.num_future, cfg.dynamic_tokens, cfg.hidden_size
            ),
            "online_scene": scene.reshape(
                batch, cfg.num_future, cfg.context_tokens, cfg.hidden_size
            ),
            "online_dynamic": dynamic.reshape(
                batch, cfg.num_future, cfg.dynamic_tokens, cfg.hidden_size
            ),
            "online_adapter_delta": delta.reshape(
                batch, cfg.num_future, cfg.context_tokens + cfg.dynamic_tokens, cfg.hidden_size
            ),
        }

    def _prior_rollout(
        self, root_context: Tensor, initial_scene: Tensor, initial_dynamic: Tensor
    ) -> dict[str, Tensor]:
        scene, dynamic = initial_scene, initial_dynamic
        scenes, dynamics, deltas = [], [], []
        for _ in self.config.future_offsets:
            scene, dynamic, delta = self.prior_transition(scene, dynamic, root_context)
            scenes.append(scene)
            dynamics.append(dynamic)
            deltas.append(delta)
        return {
            "prior_pred_scene": torch.stack(scenes, dim=1),
            "prior_pred_dynamic": torch.stack(dynamics, dim=1),
            "prior_step_delta": torch.stack(deltas, dim=1),
        }

    def rollout_from_encoded(
        self,
        root_context: Tensor,
        initial_scene: Tensor,
        initial_dynamic: Tensor,
        action: Tensor,
        state: Tensor,
        *,
        action_state: Tensor | None = None,
        mode_override: str | None = None,
    ) -> dict[str, Tensor]:
        mode = self.config.input_mode if mode_override is None else str(mode_override)
        action_state = state if action_state is None else action_state
        action_output = self.action_encoder(action, action_state)
        effect_steps = action_output["effect_steps"]
        prior_counterfactual = self._prior_rollout(root_context, initial_scene, initial_dynamic)

        scene, dynamic = initial_scene, initial_dynamic
        scenes, dynamics = [], []
        scene_effects, dynamic_effects = [], []
        adaln_diagnostics: dict[str, list[Tensor]] = {}
        prior_from_full_scenes, prior_from_full_dynamics = [], []
        detach_prior = self.training and self.training_phase == "align"
        for step, end in enumerate(self.config.future_offsets):
            prior_scene, prior_dynamic, _ = self.prior_transition(
                scene.detach() if detach_prior else scene,
                dynamic.detach() if detach_prior else dynamic,
                root_context.detach() if detach_prior else root_context,
            )
            prior_from_full_scenes.append(prior_scene)
            prior_from_full_dynamics.append(prior_dynamic)
            if mode == "current-only":
                scene_effect = torch.zeros_like(prior_scene)
                dynamic_effect = torch.zeros_like(prior_dynamic)
            else:
                scene_effect, dynamic_effect, _, diagnostics = self.action_effect(
                    scene, dynamic, root_context, effect_steps[:, : int(end)]
                )
                for key, value in diagnostics.items():
                    adaln_diagnostics.setdefault(key, []).append(value)
            scene = prior_scene + scene_effect
            dynamic = prior_dynamic + dynamic_effect
            scenes.append(scene)
            dynamics.append(dynamic)
            scene_effects.append(scene_effect)
            dynamic_effects.append(dynamic_effect)
        return {
            **prior_counterfactual,
            "pred_scene": torch.stack(scenes, dim=1),
            "pred_dynamic": torch.stack(dynamics, dim=1),
            "prior_from_full_scene": torch.stack(prior_from_full_scenes, dim=1),
            "prior_from_full_dynamic": torch.stack(prior_from_full_dynamics, dim=1),
            "action_scene_effect": torch.stack(scene_effects, dim=1),
            "action_dynamic_effect": torch.stack(dynamic_effects, dim=1),
            **{
                f"action_effect_{key}": torch.stack(values)
                for key, values in adaln_diagnostics.items()
            },
            **action_output,
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
        action_state: Tensor | None = None,
        mode_override: str | None = None,
    ) -> dict[str, Tensor]:
        mode = self.config.input_mode if mode_override is None else str(mode_override)
        action_state = state if action_state is None else action_state
        effect_steps = self.action_encoder(action, action_state)["effect_steps"]
        previous_scene, previous_dynamic = initial_scene, initial_dynamic
        full_scenes, full_dynamics = [], []
        prior_scenes, prior_dynamics = [], []
        detach_prior = self.training and self.training_phase == "align"
        for step, end in enumerate(self.config.future_offsets):
            prior_scene, prior_dynamic, _ = self.prior_transition(
                previous_scene.detach() if detach_prior else previous_scene,
                previous_dynamic.detach() if detach_prior else previous_dynamic,
                root_context.detach() if detach_prior else root_context,
            )
            if mode == "current-only":
                scene_effect = torch.zeros_like(prior_scene)
                dynamic_effect = torch.zeros_like(prior_dynamic)
            else:
                scene_effect, dynamic_effect, _, _ = self.action_effect(
                    previous_scene,
                    previous_dynamic,
                    root_context,
                    effect_steps[:, : int(end)],
                )
            full_scenes.append(prior_scene + scene_effect)
            full_dynamics.append(prior_dynamic + dynamic_effect)
            prior_scenes.append(prior_scene)
            prior_dynamics.append(prior_dynamic)
            previous_scene = target_scene[:, step].detach()
            previous_dynamic = target_dynamic[:, step].detach()
        return {
            "teacher_forced_scene": torch.stack(full_scenes, dim=1),
            "teacher_forced_dynamic": torch.stack(full_dynamics, dim=1),
            "teacher_forced_prior_scene": torch.stack(prior_scenes, dim=1),
            "teacher_forced_prior_dynamic": torch.stack(prior_dynamics, dim=1),
        }

    def forward(
        self,
        current_tokens: Tensor,
        target_tokens: Tensor,
        state: Tensor,
        action: Tensor,
        *,
        action_state: Tensor | None = None,
        mode_override: str | None = None,
    ) -> dict[str, Tensor]:
        mode = self.config.input_mode if mode_override is None else str(mode_override)
        current = self.encode_current_world(current_tokens, state, mode_override=mode)
        targets = self.encode_target_world(current_tokens, target_tokens)
        rollout = self.rollout_from_encoded(
            current["root_context"],
            current["scene"],
            current["dynamic"],
            action,
            state,
            action_state=action_state,
            mode_override=mode,
        )
        teacher = self.teacher_forced_steps(
            current["root_context"],
            targets["target_initial_scene"],
            targets["target_initial_dynamic"],
            targets["target_scene"],
            targets["target_dynamic"],
            action,
            state,
            action_state=action_state,
            mode_override=mode,
        )
        online_future = self.encode_online_future_world(target_tokens)
        inverse = self.inverse_decoder(
            current["scene"],
            current["dynamic"],
            online_future["online_scene"],
            online_future["online_dynamic"],
        )
        pred_state_path = self.decode_state_path(
            state,
            current["scene"],
            current["dynamic"],
            rollout["pred_scene"],
            rollout["pred_dynamic"],
        )
        prior_state_path = self.decode_state_path(
            state,
            current["scene"],
            current["dynamic"],
            rollout["prior_pred_scene"],
            rollout["prior_pred_dynamic"],
        )
        return {
            **current,
            **targets,
            **online_future,
            **rollout,
            **teacher,
            **inverse,
            "context": current["root_context"],
            "initial_scene": current["scene"],
            "initial_dynamic": current["dynamic"],
            "pred_descriptor": self.descriptor_prediction(rollout["pred_dynamic"]),
            "prior_descriptor": self.descriptor_prediction(rollout["prior_pred_dynamic"]),
            "initial_descriptor": self.descriptor_prediction(current["dynamic"]),
            "target_descriptor": torch.stack(
                [
                    self.fixed_dynamic_descriptor(target_tokens[:, k])
                    for k in range(self.config.num_future)
                ],
                dim=1,
            ).to(dtype=rollout["pred_dynamic"].dtype),
            "current_descriptor": self.fixed_dynamic_descriptor(current_tokens).to(
                dtype=rollout["pred_dynamic"].dtype
            ),
            "pred_state_path": pred_state_path,
            "prior_state_path": prior_state_path,
        }

    def forward_local_pair(
        self,
        current_tokens: Tensor,
        target_tokens: Tensor,
        state: Tensor,
        action: Tensor,
        *,
        action_state: Tensor | None = None,
    ) -> dict[str, Tensor]:
        output = self.forward(
            current_tokens, target_tokens, state, action, action_state=action_state
        )
        keys = (
            "pred_dynamic",
            "pred_scene",
            "prior_pred_dynamic",
            "prior_pred_scene",
            "target_dynamic",
            "target_scene",
            "initial_scene",
            "target_initial_scene",
            "action_dynamic_effect",
            "action_scene_effect",
        )
        return {key: output[key] for key in keys}

    def swapped_action_rollout(
        self,
        current_tokens: Tensor,
        state: Tensor,
        swapped_action: Tensor,
        *,
        action_state: Tensor | None = None,
    ) -> dict[str, Tensor]:
        current = self.encode_current_world(current_tokens, state)
        return self.rollout_from_encoded(
            current["root_context"],
            current["scene"],
            current["dynamic"],
            swapped_action,
            state,
            action_state=action_state,
        )

    @torch.no_grad()
    def update_ema_targets(self, decay: float) -> None:
        decay = float(decay)
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0,1)")
        for target, online in zip(
            self.target_encoder.parameters(), self.online_encoder.parameters(), strict=True
        ):
            target.mul_(decay).add_(online, alpha=1.0 - decay)
        for target, online in zip(
            self.target_adapter.parameters(), self.online_adapter.parameters(), strict=True
        ):
            target.mul_(decay).add_(online, alpha=1.0 - decay)

    def set_training_phase(self, phase: str, *, unfreeze_dynamic_blocks: int = 0) -> None:
        if phase not in {"prior", "effect", "align", "eval"}:
            raise ValueError(f"unsupported training phase={phase!r}")
        self.training_phase = phase  # type: ignore[assignment]
        self._unfreeze_dynamic_blocks = int(unfreeze_dynamic_blocks)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

        def enable(module: nn.Module) -> None:
            module.requires_grad_(True)

        if phase == "prior":
            enable(self.prior_transition)
            enable(self.state_token)
            enable(self.state_fusion)
            enable(self.state_path_head)
        elif phase in {"effect", "align"}:
            enable(self.action_encoder)
            enable(self.action_effect)
            enable(self.inverse_decoder)
            enable(self.state_token)
            enable(self.state_fusion)
            enable(self.state_path_head)
            if self.config.input_mode == "action-only":
                self.null_context.requires_grad_(True)
                self.null_dynamic.requires_grad_(True)
            if phase == "align" and self.config.input_mode == "full":
                enable(self.online_adapter)
                count = max(
                    0, min(int(unfreeze_dynamic_blocks), len(self.online_encoder.dynamic_blocks))
                )
                if count:
                    enable(self.online_encoder.dynamic_pool)
                    for block in self.online_encoder.dynamic_blocks[-count:]:
                        enable(block)
        self.target_encoder.requires_grad_(False)
        self.target_adapter.requires_grad_(False)
        self.target_encoder.eval()
        self.target_adapter.eval()
        if phase == "eval":
            self.eval()

    def trainable_named_parameters(self):
        return [
            (name, parameter)
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        ]


__all__ = ["ControllableWorldConfig", "ControllableDynamicWorld"]
