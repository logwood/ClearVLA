from __future__ import annotations

"""V34.1 uniform latent world model.

The active model is intentionally narrow in *responsibility*, not in
information bandwidth:

* ``WorldPerceiver`` directly reads every visual patch and state-history token
  and produces a structured, action-independent multi-token world state.
* ``LatentDynamicsHead`` advances that state on a dense physical time grid.
  Every action sample is consumed exactly once as a local actuation value;
  cumulative history may condition keys/gates but can never be replayed as a
  value stream.
* task/inverse/view readouts consume latent values only.  Learned time/type
  metadata can select content but cannot create content.

All value paths are zero preserving.  Consequently a zero perceived world
cannot generate a learned template trajectory and arbitrary action cannot drive
an empty world.
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Sequence

import math
import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _sinusoidal_positions(positions: Sequence[int], hidden: int) -> Tensor:
    pos = torch.tensor(tuple(int(x) for x in positions), dtype=torch.float32)[:, None]
    half = hidden // 2
    if half == 0:
        return torch.zeros(len(positions), hidden)
    scale = torch.exp(
        -math.log(10000.0) * torch.arange(half, dtype=torch.float32) / max(half - 1, 1)
    )
    phase = pos * scale[None]
    table = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
    if table.shape[-1] < hidden:
        table = F.pad(table, (0, hidden - table.shape[-1]))
    return table[:, :hidden]


class _BiasFreeFFN(nn.Module):
    def __init__(self, hidden: int, expansion: float = 4.0) -> None:
        super().__init__()
        inner = int(round(hidden * expansion))
        self.net = nn.Sequential(
            nn.Linear(hidden, inner, bias=False),
            nn.GELU(),
            nn.Linear(inner, hidden, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class _ZeroPreservingSelfBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, expansion: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True, bias=False)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.ffn = _BiasFreeFFN(hidden, expansion)

    def forward(self, x: Tensor, *, key_bias: Tensor | None = None, causal: bool = False) -> Tensor:
        value = self.norm1(x)
        query = value if key_bias is None else value + key_bias
        key = query
        mask = None
        if causal:
            length = x.shape[1]
            mask = torch.triu(
                torch.ones(length, length, dtype=torch.bool, device=x.device), diagonal=1
            )
        update, _ = self.attn(query, key, value, attn_mask=mask, need_weights=False)
        x = x + update
        return x + self.ffn(self.norm2(x))


@dataclass(frozen=True)
class LatentWorldConfig:
    latent_dim: int = 768
    action_dim: int = 7
    state_dim: int = 7
    action_horizon: int = 48
    history_length: int = 3
    num_cameras: int = 2
    patches_per_camera: int = 256
    future_offsets: tuple[int, ...] = (8, 24, 48)
    hidden_size: int = 256
    encoder_depth: int = 1  # retained only for serialized-config compatibility
    predictor_depth: int = 1
    action_depth: int = 3
    num_heads: int = 8
    context_tokens: int = 8
    dynamic_tokens: int = 16
    descriptor_projection_dim: int = 32
    dropout: float = 0.0
    input_mode: Literal["full"] = "full"
    gripper_dim_index: int = -1
    descriptor_seed: int = 34033
    perceiver_depth: int = 4
    dynamics_depth: int = 6
    dynamics_unique_blocks: int = 3
    state_decoder_depth: int = 2
    inverse_depth: int = 2
    root_tokens: int = 1
    dynamics_ffn_expansion: float = 4.0
    adaln_zero_init: bool = True
    latent_stride: int = 4
    action_probe_depth: int = 2

    def validate(self) -> None:
        dimensions = (
            self.latent_dim, self.action_dim, self.state_dim, self.action_horizon,
            self.history_length, self.num_cameras, self.patches_per_camera,
            self.hidden_size, self.num_heads, self.context_tokens, self.dynamic_tokens,
            self.descriptor_projection_dim, self.perceiver_depth, self.dynamics_depth,
            self.dynamics_unique_blocks, self.state_decoder_depth, self.inverse_depth,
            self.latent_stride, self.action_probe_depth,
        )
        if min(dimensions) <= 0:
            raise ValueError("all V34.1 dimensions/depths must be positive")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.action_dim != self.state_dim:
            raise ValueError("action_dim and state_dim must match")
        if self.root_tokens != 1:
            raise ValueError("V34.1 requires exactly one root token")
        if tuple(sorted(set(self.future_offsets))) != self.future_offsets or not self.future_offsets:
            raise ValueError("future_offsets must be strictly increasing and non-empty")
        if max(self.future_offsets) > self.action_horizon:
            raise ValueError("future_offsets cannot exceed action_horizon")
        if self.dynamics_unique_blocks > self.dynamics_depth:
            raise ValueError("dynamics_unique_blocks cannot exceed dynamics_depth")
        if self.dynamics_ffn_expansion < 1.0:
            raise ValueError("dynamics_ffn_expansion must be >= 1")
        if self.action_horizon % self.latent_stride != 0:
            raise ValueError("action_horizon must be divisible by latent_stride")
        if any(int(offset) % self.latent_stride != 0 for offset in self.future_offsets):
            raise ValueError("future_offsets must lie on the dense latent grid")
        index = self.gripper_dim_index if self.gripper_dim_index >= 0 else self.state_dim + self.gripper_dim_index
        if not 0 <= index < self.state_dim:
            raise ValueError("gripper_dim_index outside state dimensions")

    @property
    def gripper_index(self) -> int:
        return self.gripper_dim_index if self.gripper_dim_index >= 0 else self.state_dim + self.gripper_dim_index

    @property
    def num_future(self) -> int:
        return len(self.future_offsets)

    @property
    def world_tokens(self) -> int:
        return self.root_tokens + self.context_tokens + self.dynamic_tokens

    @property
    def latent_offsets(self) -> tuple[int, ...]:
        return tuple(range(self.latent_stride, self.action_horizon + 1, self.latent_stride))

    @property
    def num_latent_steps(self) -> int:
        return len(self.latent_offsets)

    @property
    def visual_anchor_indices(self) -> tuple[int, ...]:
        lookup = {offset: index for index, offset in enumerate(self.latent_offsets)}
        return tuple(lookup[int(offset)] for offset in self.future_offsets)

    @property
    def descriptor_dim(self) -> int:
        return (self.history_length - 1) * self.num_cameras * (2 * self.descriptor_projection_dim + 2)


class _PerceiverLayer(nn.Module):
    """Latent refinement whose values always come from real evidence/latent content."""

    def __init__(self, hidden: int, heads: int, expansion: float = 4.0) -> None:
        super().__init__()
        self.q_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.k_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.v_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.cross = nn.MultiheadAttention(hidden, heads, batch_first=True, bias=False)
        self.self_block = _ZeroPreservingSelfBlock(hidden, heads, expansion)

    def read(
        self,
        query_content: Tensor,
        query_bias: Tensor,
        evidence_key: Tensor,
        evidence_value: Tensor,
    ) -> Tensor:
        query = self.q_norm(query_content) + query_bias
        read, _ = self.cross(
            query,
            self.k_norm(evidence_key),
            self.v_norm(evidence_value),
            need_weights=False,
        )
        return read

    def forward(
        self,
        latent: Tensor,
        slot_bias: Tensor,
        evidence_key: Tensor,
        evidence_value: Tensor,
    ) -> Tensor:
        latent = latent + self.read(latent, slot_bias, evidence_key, evidence_value)
        return self.self_block(latent, key_bias=slot_bias)


class WorldPerceiver(nn.Module):
    """Direct full-evidence, action-independent world perceiver.

    Learned history/camera/patch/state/type metadata is used only in attention
    keys/queries.  Attention values are projected observation/state content.
    Hence zero visual and zero state evidence produce exactly zero world latent.
    """

    def __init__(self, config: LatentWorldConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        h = config.hidden_size
        self.visual_norm = nn.LayerNorm(config.latent_dim, elementwise_affine=False)
        self.visual_proj = nn.Linear(config.latent_dim, h, bias=False)
        self.state_norm = nn.LayerNorm(config.state_dim, elementwise_affine=False)
        self.state_proj = nn.Linear(config.state_dim, h, bias=False)

        self.history_key = nn.Parameter(torch.randn(1, config.history_length, 1, 1, h) * 0.02)
        self.camera_key = nn.Parameter(torch.randn(1, 1, config.num_cameras, 1, h) * 0.02)
        self.patch_key = nn.Parameter(torch.randn(1, 1, 1, config.patches_per_camera, h) * 0.02)
        self.visual_type_key = nn.Parameter(torch.randn(1, 1, 1, 1, h) * 0.02)
        self.state_time_key = nn.Parameter(torch.randn(1, config.history_length, h) * 0.02)
        self.state_type_key = nn.Parameter(torch.randn(1, 1, h) * 0.02)

        self.root_query = nn.Parameter(torch.randn(1, config.root_tokens, h) * 0.02)
        self.scene_queries = nn.Parameter(torch.randn(1, config.context_tokens, h) * 0.02)
        self.dynamic_queries = nn.Parameter(torch.randn(1, config.dynamic_tokens, h) * 0.02)
        self.layers = nn.ModuleList(
            [_PerceiverLayer(h, config.num_heads) for _ in range(config.perceiver_depth)]
        )
        self.out_norm = nn.LayerNorm(h, elementwise_affine=False)

    def _slot_bias(self, batch: int) -> Tensor:
        return torch.cat(
            [self.root_query, self.scene_queries, self.dynamic_queries], dim=1
        ).expand(batch, -1, -1)

    def _evidence(self, visual_tokens: Tensor, state_history: Tensor) -> tuple[Tensor, Tensor]:
        cfg = self.config
        expected_visual = (
            cfg.history_length,
            cfg.num_cameras,
            cfg.patches_per_camera,
            cfg.latent_dim,
        )
        if tuple(visual_tokens.shape[1:]) != expected_visual:
            raise ValueError(f"visual_tokens must be [B,{','.join(map(str, expected_visual))}]")
        if tuple(state_history.shape[1:]) != (cfg.history_length, cfg.state_dim):
            raise ValueError("state_history geometry does not match V34.1 config")

        visual_value = self.visual_proj(self.visual_norm(visual_tokens))
        visual_key = (
            visual_value
            + self.history_key
            + self.camera_key
            + self.patch_key
            + self.visual_type_key
        )
        visual_value = visual_value.flatten(1, 3)
        visual_key = visual_key.flatten(1, 3)

        state_value = self.state_proj(self.state_norm(state_history))
        state_key = state_value + self.state_time_key + self.state_type_key
        return (
            torch.cat([visual_key, state_key], dim=1),
            torch.cat([visual_value, state_value], dim=1),
        )

    def forward(self, visual_tokens: Tensor, state_history: Tensor) -> Tensor:
        evidence_key, evidence_value = self._evidence(visual_tokens, state_history)
        batch = visual_tokens.shape[0]
        slots = self._slot_bias(batch)
        # The first read does not retain learned query content as a residual.
        latent = self.layers[0].read(
            torch.zeros_like(slots), slots, evidence_key, evidence_value
        )
        latent = self.layers[0].self_block(latent, key_bias=slots)
        for layer in self.layers[1:]:
            latent = layer(latent, slots, evidence_key, evidence_value)
        return self.out_norm(latent)


class CounterfactualActionTokenizer(nn.Module):
    """Full action tokenizer with exact physical-hold subtraction."""

    def __init__(self, config: LatentWorldConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.input_proj = nn.Sequential(
            nn.Linear(4 * config.action_dim, 2 * h, bias=False),
            nn.SiLU(),
            nn.Linear(2 * h, h, bias=False),
        )
        self.register_buffer(
            "position_key",
            _sinusoidal_positions(range(1, config.action_horizon + 1), h)[None],
            persistent=True,
        )
        self.blocks = nn.ModuleList(
            [_ZeroPreservingSelfBlock(h, config.num_heads) for _ in range(config.action_depth)]
        )
        self.norm = nn.LayerNorm(h, elementwise_affine=False)

    def _features(self, action: Tensor, action_state: Tensor) -> Tensor:
        boundary = torch.cat([action_state[:, None], action[:, :-1]], dim=1)
        velocity = action - boundary
        previous_velocity = torch.cat([torch.zeros_like(velocity[:, :1]), velocity[:, :-1]], dim=1)
        acceleration = velocity - previous_velocity
        relative = action - action_state[:, None]
        return torch.cat([action, velocity, acceleration, relative], dim=-1)

    def _encode(self, action: Tensor, action_state: Tensor) -> Tensor:
        x = self.input_proj(self._features(action, action_state))
        for block in self.blocks:
            x = block(x, key_bias=self.position_key.to(dtype=x.dtype), causal=True)
        return self.norm(x)

    def forward(self, action: Tensor, action_state: Tensor) -> dict[str, Tensor]:
        cfg = self.config
        if tuple(action.shape[1:]) != (cfg.action_horizon, cfg.action_dim):
            raise ValueError("action geometry does not match V34.1 config")
        if tuple(action_state.shape[1:]) != (cfg.state_dim,):
            raise ValueError("action_state must be action-normalized qpos")
        hold = action_state[:, None].expand(-1, cfg.action_horizon, -1)
        encoded = self._encode(
            torch.cat([action, hold], dim=0),
            torch.cat([action_state, action_state], dim=0),
        )
        actual, hold_encoded = encoded.chunk(2, dim=0)
        effect = actual - hold_encoded
        exact_hold = (action == hold).all(dim=(1, 2), keepdim=True)
        effect = torch.where(exact_hold, torch.zeros_like(effect), effect)
        return {
            "effect_steps": effect,
            "actual_steps": actual,
            "hold_steps": hold_encoded,
            "hold_action": hold,
        }


class _UniformJointAdaLNBlock(nn.Module):
    """Zero-preserving base transition plus local-action Joint-AdaLN update."""

    def __init__(self, config: LatentWorldConfig) -> None:
        super().__init__()
        h = config.hidden_size
        heads = config.num_heads
        self.config = config
        self.base_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.base_attn = nn.MultiheadAttention(h, heads, batch_first=True, bias=False)
        self.base_ffn_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.base_ffn = _BiasFreeFFN(h, config.dynamics_ffn_expansion)

        self.world_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.action_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.action_reads_world = nn.MultiheadAttention(h, heads, batch_first=True, bias=False)
        self.action_self_factor = nn.Linear(h, 2 * h, bias=False)
        self.action_world_factor = nn.Linear(h, 2 * h, bias=False)
        self.action_read_out = nn.Linear(2 * h, h, bias=False)
        self.world_reads_action = nn.MultiheadAttention(h, heads, batch_first=True, bias=False)

        self.world_factor = nn.Linear(h, 2 * h, bias=False)
        self.action_factor = nn.Linear(h, 2 * h, bias=False)
        self.normalized_world_factor = nn.Linear(h, 2 * h, bias=False)
        self.normalized_action_factor = nn.Linear(h, 2 * h, bias=False)
        self.joint_mix = nn.Linear(4 * h, 2 * h, bias=False)
        self.joint_strength = nn.Parameter(torch.ones(1, 1, 1))

        self.root_modulation = nn.Linear(2 * h, 6 * h, bias=False)
        self.scene_modulation = nn.Linear(2 * h, 6 * h, bias=False)
        self.dynamic_modulation = nn.Linear(2 * h, 6 * h, bias=False)
        if config.adaln_zero_init:
            nn.init.zeros_(self.root_modulation.weight)
            nn.init.zeros_(self.scene_modulation.weight)
            nn.init.zeros_(self.dynamic_modulation.weight)

        self.action_attn_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.action_self_attn = nn.MultiheadAttention(h, heads, batch_first=True, bias=False)
        self.action_ffn_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.action_ffn = _BiasFreeFFN(h, config.dynamics_ffn_expansion)

    @staticmethod
    def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        return x * (1.0 + scale) + shift

    def _typed_modulation(self, joint: Tensor) -> Tensor:
        cfg = self.config
        scene_start = cfg.root_tokens
        scene_end = scene_start + cfg.context_tokens
        return torch.cat(
            [
                self.root_modulation(joint[:, :scene_start]),
                self.scene_modulation(joint[:, scene_start:scene_end]),
                self.dynamic_modulation(joint[:, scene_end:]),
            ],
            dim=1,
        )

    def forward(
        self,
        world: Tensor,
        local_action: Tensor,
        *,
        world_key_bias: Tensor,
        action_key_bias: Tensor,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        # Base transition: query/key may contain metadata, values contain only
        # current world content.  A zero world therefore remains zero.
        base_value = self.base_norm(world)
        base_qk = base_value + world_key_bias
        base_update, _ = self.base_attn(base_qk, base_qk, base_value, need_weights=False)
        world = world + base_update
        world = world + self.base_ffn(self.base_ffn_norm(world))

        action_value = self.action_norm(local_action)
        action_query = action_value + action_key_bias
        world_value = self.world_norm(world)
        world_key = world_value + world_key_bias
        action_read, _ = self.action_reads_world(
            action_query, world_key, world_value, need_weights=False
        )
        action_joint = self.action_self_factor(action_value) * self.action_world_factor(action_read)
        local_action = local_action + self.action_read_out(F.silu(action_joint))

        action_value = self.action_norm(local_action)
        action_key = action_value + action_key_bias
        world_query = self.world_norm(world) + world_key_bias
        action_signal, _ = self.world_reads_action(
            world_query, action_key, action_value, need_weights=False
        )

        world_content = self.world_norm(world)
        raw_joint = self.world_factor(world_content) * self.action_factor(action_signal)
        normalized_joint = self.normalized_world_factor(world_content) * self.normalized_action_factor(
            self.action_norm(action_signal)
        )
        joint = self.joint_strength * F.silu(
            self.joint_mix(torch.cat([raw_joint, normalized_joint], dim=-1))
        )
        modulation = self._typed_modulation(joint)
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = modulation.chunk(
            6, dim=-1
        )

        value = self.action_attn_norm(world)
        qk = self._modulate(value, shift_attn, scale_attn) + world_key_bias
        attn_out, _ = self.action_self_attn(qk, qk, value, need_weights=False)
        world = world + torch.tanh(gate_attn) * attn_out
        ffn_input = self._modulate(self.action_ffn_norm(world), shift_ffn, scale_ffn)
        world = world + torch.tanh(gate_ffn) * self.action_ffn(ffn_input)

        return world, local_action, {
            "adaln_gate_abs_mean": 0.5
            * (torch.tanh(gate_attn).abs().mean() + torch.tanh(gate_ffn).abs().mean()),
            "adaln_scale_abs_mean": 0.5 * (scale_attn.abs().mean() + scale_ffn.abs().mean()),
            "adaln_shift_abs_mean": 0.5 * (shift_attn.abs().mean() + shift_ffn.abs().mean()),
            "action_read_joint_rms": action_joint.float().square().mean().sqrt(),
            "world_action_joint_rms": joint.float().square().mean().sqrt(),
            "action_signal_rms": action_signal.float().square().mean().sqrt(),
        }


class LatentDynamicsHead(nn.Module):
    """Dense, segment-correct shared dynamics for actual and hold rollouts."""

    def __init__(self, config: LatentWorldConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.root_key = nn.Parameter(torch.randn(1, config.root_tokens, h) * 0.02)
        self.scene_key = nn.Parameter(torch.randn(1, config.context_tokens, h) * 0.02)
        self.dynamic_key = nn.Parameter(torch.randn(1, config.dynamic_tokens, h) * 0.02)
        self.register_buffer(
            "step_key",
            _sinusoidal_positions(config.latent_offsets, h)[None, :, None],
            persistent=True,
        )
        self.register_buffer(
            "action_time_key",
            _sinusoidal_positions(range(1, config.action_horizon + 1), h)[None],
            persistent=True,
        )
        self.history_key_proj = nn.Linear(h, h, bias=False)
        self.blocks = nn.ModuleList(
            [_UniformJointAdaLNBlock(config) for _ in range(config.dynamics_unique_blocks)]
        )
        self.out_norm = nn.LayerNorm(h, elementwise_affine=False)

    def _world_key_bias(self, batch: int, step_index: int, dtype: torch.dtype) -> Tensor:
        slot = torch.cat([self.root_key, self.scene_key, self.dynamic_key], dim=1)
        return (slot + self.step_key[:, step_index]).expand(batch, -1, -1).to(dtype=dtype)

    def _local_action(
        self, effect_steps: Tensor, *, start: int, end: int
    ) -> tuple[Tensor, Tensor]:
        # Local values are consumed exactly once.  Past action can only alter
        # attention keys through a detached-length-preserving context summary.
        local = effect_steps[:, start:end]
        local_key = self.action_time_key[:, start:end].to(dtype=local.dtype).expand(
            local.shape[0], -1, -1
        )
        if start > 0:
            history = effect_steps[:, :start].mean(dim=1, keepdim=True)
            history_key = self.history_key_proj(history).expand(-1, end - start, -1)
            local_key = local_key + history_key
        return local, local_key

    def step(
        self,
        world: Tensor,
        effect_steps: Tensor,
        *,
        step_index: int,
        interval_start: int,
        interval_end: int,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        cfg = self.config
        if tuple(world.shape[1:]) != (cfg.world_tokens, cfg.hidden_size):
            raise ValueError("world geometry does not match V34.1 config")
        local_action, action_key = self._local_action(
            effect_steps, start=int(interval_start), end=int(interval_end)
        )
        world_key = self._world_key_bias(world.shape[0], step_index, world.dtype)
        rows: dict[str, list[Tensor]] = {}
        for depth_index in range(cfg.dynamics_depth):
            block = self.blocks[depth_index % len(self.blocks)]
            world, local_action, diagnostics = block(
                world,
                local_action,
                world_key_bias=world_key,
                action_key_bias=action_key,
            )
            for key, value in diagnostics.items():
                rows.setdefault(key, []).append(value)
        world = self.out_norm(world)
        diagnostics = {key: torch.stack(values).mean() for key, values in rows.items()}
        diagnostics["world_rms"] = world.float().square().mean().sqrt()
        diagnostics["local_action_rms"] = local_action.float().square().mean().sqrt()
        return world, diagnostics

    def rollout_pair(self, initial_world: Tensor, effect_steps: Tensor) -> dict[str, Tensor]:
        cfg = self.config
        batch = initial_world.shape[0]
        world = torch.cat([initial_world, initial_world], dim=0)
        effects = torch.cat([effect_steps, torch.zeros_like(effect_steps)], dim=0)
        dense_worlds: list[Tensor] = []
        diagnostic_rows: dict[str, list[Tensor]] = {}
        start = 0
        for step_index, end in enumerate(cfg.latent_offsets):
            world, diagnostics = self.step(
                world,
                effects,
                step_index=step_index,
                interval_start=start,
                interval_end=int(end),
            )
            dense_worlds.append(world)
            for key, value in diagnostics.items():
                diagnostic_rows.setdefault(key, []).append(value)
            start = int(end)
        dense = torch.stack(dense_worlds, dim=1)
        actual_dense, hold_dense = dense[:batch], dense[batch:]
        indices = torch.tensor(cfg.visual_anchor_indices, device=dense.device)
        actual = actual_dense.index_select(1, indices)
        hold = hold_dense.index_select(1, indices)
        return {
            "dense_pred_world": actual_dense,
            "dense_hold_world": hold_dense,
            "pred_world": actual,
            "hold_world": hold,
            "dense_action_world_effect": actual_dense - hold_dense,
            "action_world_effect": actual - hold,
            **{key: torch.stack(values).mean() for key, values in diagnostic_rows.items()},
        }

    def rollout(self, initial_world: Tensor, effect_steps: Tensor) -> Tensor:
        return self.rollout_pair(initial_world, effect_steps)["pred_world"]


class _LatentReadoutBlock(_ZeroPreservingSelfBlock):
    pass


class LatentStatePathDecoder(nn.Module):
    """Decode 48 states from the dense latent grid, never from learned values."""

    def __init__(self, config: LatentWorldConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.register_buffer(
            "anchor_key",
            _sinusoidal_positions((0,) + config.latent_offsets, h)[None, :, None],
            persistent=True,
        )
        self.register_buffer(
            "path_query",
            _sinusoidal_positions(range(1, config.action_horizon + 1), h)[None],
            persistent=True,
        )
        self.query_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.key_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.value_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.cross = nn.MultiheadAttention(h, config.num_heads, batch_first=True, bias=False)
        self.blocks = nn.ModuleList(
            [_LatentReadoutBlock(h, config.num_heads) for _ in range(config.state_decoder_depth)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(h, elementwise_affine=False),
            nn.Linear(h, config.state_dim, bias=False),
        )

    def forward(self, initial_world: Tensor, dense_future_world: Tensor) -> Tensor:
        if dense_future_world.shape[1] != self.config.num_latent_steps:
            raise ValueError("state decoder requires the dense latent timeline")
        anchors = torch.cat([initial_world[:, None], dense_future_world], dim=1)
        keys = (anchors + self.anchor_key.to(dtype=anchors.dtype)).flatten(1, 2)
        values = anchors.flatten(1, 2)
        queries = self.path_query.to(dtype=anchors.dtype).expand(initial_world.shape[0], -1, -1)
        x, _ = self.cross(
            self.query_norm(queries),
            self.key_norm(keys),
            self.value_norm(values),
            need_weights=False,
        )
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class LatentStateObserver(nn.Module):
    def __init__(self, config: LatentWorldConfig) -> None:
        super().__init__()
        h = config.hidden_size
        self.register_buffer("query", _sinusoidal_positions((0,), h)[None], persistent=True)
        self.q_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.m_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(h, config.num_heads, batch_first=True, bias=False)
        self.head = nn.Sequential(
            nn.LayerNorm(h, elementwise_affine=False),
            nn.Linear(h, config.state_dim, bias=False),
        )

    def forward(self, world: Tensor) -> Tensor:
        query = self.query.to(dtype=world.dtype).expand(world.shape[0], -1, -1)
        memory = self.m_norm(world)
        x, _ = self.attn(self.q_norm(query), memory, memory, need_weights=False)
        return self.head(x[:, 0])


class LatentInverseDecoder(nn.Module):
    """Shared inverse decoder for sparse observed and dense predicted futures."""

    def __init__(self, config: LatentWorldConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.current_proj = nn.Linear(h, h, bias=False)
        self.change_proj = nn.Linear(h, h, bias=False)
        self.key_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.value_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.query_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.cross = nn.MultiheadAttention(h, config.num_heads, batch_first=True, bias=False)
        self.blocks = nn.ModuleList(
            [_LatentReadoutBlock(h, config.num_heads) for _ in range(config.inverse_depth)]
        )
        self.relative_head = nn.Sequential(
            nn.LayerNorm(h, elementwise_affine=False), nn.Linear(h, config.action_dim, bias=False)
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(h, elementwise_affine=False), nn.Linear(h, config.action_dim, bias=False)
        )
        self.gripper_head = nn.Sequential(
            nn.LayerNorm(h, elementwise_affine=False), nn.Linear(h, 3, bias=False)
        )
        self.register_buffer(
            "path_query",
            _sinusoidal_positions(range(1, config.action_horizon + 1), h)[None],
            persistent=True,
        )

    def forward(
        self,
        current_world: Tensor,
        future_world: Tensor,
        offsets: Sequence[int] | Tensor,
    ) -> dict[str, Tensor]:
        if isinstance(offsets, Tensor):
            offset_values = tuple(int(x) for x in offsets.detach().cpu().tolist())
        else:
            offset_values = tuple(int(x) for x in offsets)
        if future_world.shape[1] != len(offset_values):
            raise ValueError("inverse offsets must match future latent count")
        change = future_world - current_world[:, None]
        current = self.current_proj(current_world)[:, None].expand_as(change)
        values = current * self.change_proj(change)
        time = _sinusoidal_positions(offset_values, future_world.shape[-1]).to(
            device=future_world.device, dtype=future_world.dtype
        )[None, :, None]
        keys = (values + time).flatten(1, 2)
        values = values.flatten(1, 2)
        queries = self.path_query.to(dtype=future_world.dtype).expand(current_world.shape[0], -1, -1)
        x, _ = self.cross(
            self.query_norm(queries), self.key_norm(keys), self.value_norm(values), need_weights=False
        )
        for block in self.blocks:
            x = block(x)
        return {
            "action": self.relative_head(x),
            "delta": self.delta_head(x),
            "gripper_logits": self.gripper_head(x),
        }


class LatentViewDescriptorDecoder(nn.Module):
    """Direct visual supervision whose output values must originate in world latent."""

    def __init__(self, config: LatentWorldConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size
        count = config.history_length * config.num_cameras
        self.register_buffer(
            "queries",
            _sinusoidal_positions(range(count), h)[None],
            persistent=True,
        )
        self.q_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.m_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.cross = nn.MultiheadAttention(h, config.num_heads, batch_first=True, bias=False)
        self.head = nn.Linear(h, config.descriptor_projection_dim, bias=False)

    def forward(self, world: Tensor) -> Tensor:
        leading = world.shape[:-2]
        flat = world.reshape(-1, world.shape[-2], world.shape[-1])
        query = self.queries.to(dtype=flat.dtype).expand(flat.shape[0], -1, -1)
        memory = self.m_norm(flat)
        x, _ = self.cross(self.q_norm(query), memory, memory, need_weights=False)
        out = self.head(x).reshape(
            *leading,
            self.config.history_length,
            self.config.num_cameras,
            self.config.descriptor_projection_dim,
        )
        return out


class ActionOnlyDiagnosticProbe(nn.Module):
    """Isolated diagnostic probe; never feeds the world model or task heads."""

    def __init__(self, config: LatentWorldConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.input = nn.Linear(config.action_dim, h)
        self.register_buffer(
            "position",
            _sinusoidal_positions(range(1, config.action_horizon + 1), h)[None],
            persistent=True,
        )
        self.blocks = nn.ModuleList(
            [_ZeroPreservingSelfBlock(h, config.num_heads) for _ in range(config.action_probe_depth)]
        )
        self.queries = nn.Parameter(torch.randn(1, config.num_future * 3, h) * 0.02)
        self.cross = nn.MultiheadAttention(h, config.num_heads, batch_first=True)
        self.head = nn.Linear(h, h)

    def forward(self, action: Tensor) -> Tensor:
        x = self.input(action)
        for block in self.blocks:
            x = block(x, key_bias=self.position.to(dtype=x.dtype), causal=True)
        q = self.queries.expand(action.shape[0], -1, -1)
        out, _ = self.cross(q, x + self.position.to(dtype=x.dtype), x, need_weights=False)
        return self.head(out).reshape(action.shape[0], self.config.num_future, 3, self.config.hidden_size)


class LatentWorldModel(nn.Module):
    """Single-run V34.1 model with no task-side raw-input bypass."""

    def __init__(self, config: LatentWorldConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.online_perceiver = WorldPerceiver(config)
        self.target_perceiver = deepcopy(self.online_perceiver).float()
        self.target_perceiver.requires_grad_(False)
        self.target_perceiver.eval()
        self.action_tokenizer = CounterfactualActionTokenizer(config)
        self.dynamics = LatentDynamicsHead(config)
        self.state_decoder = LatentStatePathDecoder(config)
        self.state_observer = LatentStateObserver(config)
        self.inverse_decoder = LatentInverseDecoder(config)
        self.view_decoder = LatentViewDescriptorDecoder(config)
        self.action_only_probe = ActionOnlyDiagnosticProbe(config)

        h = config.hidden_size
        self.local_motion_head = nn.Sequential(
            nn.LayerNorm(h, elementwise_affine=False),
            nn.Linear(h, h, bias=False),
            nn.SiLU(),
            nn.Linear(h, config.state_dim, bias=False),
        )

        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.descriptor_seed)
        matrix = torch.randn(config.latent_dim, config.descriptor_projection_dim, generator=generator)
        q, _ = torch.linalg.qr(matrix, mode="reduced")
        self.register_buffer("descriptor_projection", q.float(), persistent=True)

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_perceiver.eval()
        return self

    def split_world(self, world: Tensor) -> dict[str, Tensor]:
        cfg = self.config
        scene_start = cfg.root_tokens
        scene_end = scene_start + cfg.context_tokens
        return {
            "root": world[..., :scene_start, :],
            "scene": world[..., scene_start:scene_end, :],
            "dynamic": world[..., scene_end:, :],
        }

    def world_summary(self, world: Tensor) -> Tensor:
        split = self.split_world(world)
        return torch.stack(
            [split["root"].mean(dim=-2), split["scene"].mean(dim=-2), split["dynamic"].mean(dim=-2)],
            dim=-2,
        )

    def fixed_dynamic_descriptor(self, tokens: Tensor) -> Tensor:
        """Compatibility descriptor for pair indexing; it is not a main-path bottleneck."""
        diff = tokens.float()[:, 1:] - tokens.float()[:, :-1]
        projected = diff @ self.descriptor_projection.float()
        energy = diff.square().mean(dim=-1).sqrt().clamp_min(1e-8)
        weights = torch.softmax(energy / 0.1, dim=-1)
        weighted = F.normalize((projected * weights[..., None]).sum(dim=-2), dim=-1)
        mean = F.normalize(projected.mean(dim=-2), dim=-1)
        mean_energy = torch.log1p(energy.mean(dim=-1))[..., None]
        max_energy = torch.log1p(energy.max(dim=-1).values)[..., None]
        return torch.cat([weighted, mean, mean_energy, max_energy], dim=-1).reshape(tokens.shape[0], -1)

    def fixed_view_descriptor(self, tokens: Tensor) -> Tensor:
        # [B,H,C,P,D] or [B,F,H,C,P,D]
        projected = tokens.float() @ self.descriptor_projection.float()
        descriptor = projected.mean(dim=-2)
        return F.normalize(descriptor, dim=-1)

    def encode_online(self, tokens: Tensor, state_history: Tensor) -> Tensor:
        return self.online_perceiver(tokens, state_history)

    @torch.no_grad()
    def encode_targets(
        self,
        current_tokens: Tensor,
        target_tokens: Tensor,
        history_state: Tensor,
        target_history_state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        cfg = self.config
        current = self.target_perceiver(current_tokens.float(), history_state.float())
        flat_tokens = target_tokens.reshape(
            -1, cfg.history_length, cfg.num_cameras, cfg.patches_per_camera, cfg.latent_dim
        )
        flat_states = target_history_state.reshape(-1, cfg.history_length, cfg.state_dim)
        future = self.target_perceiver(flat_tokens.float(), flat_states.float()).reshape(
            target_tokens.shape[0], cfg.num_future, cfg.world_tokens, cfg.hidden_size
        )
        return current, future

    def encode_online_future(self, target_tokens: Tensor, target_history_state: Tensor) -> Tensor:
        cfg = self.config
        flat_tokens = target_tokens.reshape(
            -1, cfg.history_length, cfg.num_cameras, cfg.patches_per_camera, cfg.latent_dim
        )
        flat_states = target_history_state.reshape(-1, cfg.history_length, cfg.state_dim)
        return self.online_perceiver(flat_tokens, flat_states).reshape(
            target_tokens.shape[0], cfg.num_future, cfg.world_tokens, cfg.hidden_size
        )

    def _teacher_forced(
        self, target_initial: Tensor, target_future: Tensor, effect_steps: Tensor
    ) -> Tensor:
        cfg = self.config
        target_by_offset = {int(offset): target_future[:, i] for i, offset in enumerate(cfg.future_offsets)}
        previous = target_initial
        rows: list[Tensor] = []
        start = 0
        for step_index, end in enumerate(cfg.latent_offsets):
            predicted, _ = self.dynamics.step(
                previous,
                effect_steps,
                step_index=step_index,
                interval_start=start,
                interval_end=int(end),
            )
            if int(end) in target_by_offset:
                rows.append(predicted)
                previous = target_by_offset[int(end)].detach()
            else:
                previous = predicted
            start = int(end)
        return torch.stack(rows, dim=1)

    def forward(
        self,
        current_tokens: Tensor,
        target_tokens: Tensor,
        history_state: Tensor,
        target_history_state: Tensor,
        action: Tensor,
        action_state: Tensor,
    ) -> dict[str, Tensor]:
        current_world = self.encode_online(current_tokens, history_state)
        target_initial, target_world = self.encode_targets(
            current_tokens, target_tokens, history_state, target_history_state
        )
        action_output = self.action_tokenizer(action, action_state)
        rollout = self.dynamics.rollout_pair(current_world, action_output["effect_steps"])
        online_future = self.encode_online_future(target_tokens, target_history_state)
        teacher = self._teacher_forced(target_initial, target_world, action_output["effect_steps"])
        inverse_pred = self.inverse_decoder(
            current_world, rollout["dense_pred_world"], self.config.latent_offsets
        )
        inverse_target = self.inverse_decoder(
            current_world, online_future, self.config.future_offsets
        )
        pred_state = self.state_decoder(current_world, rollout["dense_pred_world"])
        hold_state = self.state_decoder(current_world, rollout["dense_hold_world"])
        return {
            "initial_world": current_world,
            "target_initial_world": target_initial,
            "target_world": target_world,
            "online_future_world": online_future,
            "teacher_forced_world": teacher,
            "pred_state_path": pred_state,
            "hold_state_path": hold_state,
            "current_state_prediction": self.state_observer(current_world),
            "local_motion_prediction": self.local_motion_head(
                self.split_world(current_world)["dynamic"].mean(dim=1)
            ),
            "current_view_prediction": self.view_decoder(current_world),
            "pred_view_prediction": self.view_decoder(rollout["pred_world"]),
            "current_view_target": self.fixed_view_descriptor(current_tokens).to(current_world.dtype),
            "future_view_target": self.fixed_view_descriptor(target_tokens).to(current_world.dtype),
            "pred_inverse_action": inverse_pred["action"],
            "pred_inverse_delta": inverse_pred["delta"],
            "pred_inverse_gripper_logits": inverse_pred["gripper_logits"],
            "target_inverse_action": inverse_target["action"],
            "target_inverse_delta": inverse_target["delta"],
            "target_inverse_gripper_logits": inverse_target["gripper_logits"],
            "action_only_probe_prediction": self.action_only_probe(action.detach()),
            "action_only_probe_target": self.world_summary(target_world).detach(),
            **rollout,
            **action_output,
        }

    def forward_local_pair(
        self,
        current_tokens: Tensor,
        target_tokens: Tensor,
        history_state: Tensor,
        target_history_state: Tensor,
        action: Tensor,
        action_state: Tensor,
    ) -> dict[str, Tensor]:
        initial_world = self.encode_online(current_tokens, history_state)
        target_initial, target_world = self.encode_targets(
            current_tokens, target_tokens, history_state, target_history_state
        )
        action_output = self.action_tokenizer(action, action_state)
        rollout = self.dynamics.rollout_pair(initial_world, action_output["effect_steps"])
        return {
            "initial_world": initial_world,
            "target_initial_world": target_initial,
            "target_world": target_world,
            **rollout,
        }

    def swapped_action_rollout(
        self, current_world: Tensor, swapped_action: Tensor, swapped_action_state: Tensor
    ) -> dict[str, Tensor]:
        action = self.action_tokenizer(swapped_action, swapped_action_state)
        return self.dynamics.rollout_pair(current_world, action["effect_steps"])

    @torch.no_grad()
    def update_ema(self, decay: float) -> None:
        if not 0.0 <= float(decay) < 1.0:
            raise ValueError("EMA decay must be in [0,1)")
        for target, online in zip(
            self.target_perceiver.parameters(), self.online_perceiver.parameters(), strict=True
        ):
            target.mul_(float(decay)).add_(online.float(), alpha=1.0 - float(decay))
        for target_buffer, online_buffer in zip(
            self.target_perceiver.buffers(), self.online_perceiver.buffers(), strict=True
        ):
            target_buffer.copy_(online_buffer.float())

    def parameter_report(self) -> dict[str, int]:
        modules: dict[str, nn.Module] = {
            "online_perceiver": self.online_perceiver,
            "target_perceiver": self.target_perceiver,
            "action_tokenizer": self.action_tokenizer,
            "latent_dynamics": self.dynamics,
            "state_decoder": self.state_decoder,
            "state_observer": self.state_observer,
            "inverse_decoder": self.inverse_decoder,
            "view_decoder": self.view_decoder,
            "action_only_probe": self.action_only_probe,
            "representation_heads": self.local_motion_head,
        }
        report = {name: sum(p.numel() for p in module.parameters()) for name, module in modules.items()}
        report["total"] = sum(p.numel() for p in self.parameters())
        report["trainable"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        report["online_inference"] = (
            report["total"]
            - report["target_perceiver"]
            - report["inverse_decoder"]
            - report["action_only_probe"]
        )
        return report


__all__ = [
    "LatentWorldConfig",
    "WorldPerceiver",
    "CounterfactualActionTokenizer",
    "LatentDynamicsHead",
    "LatentWorldModel",
]
