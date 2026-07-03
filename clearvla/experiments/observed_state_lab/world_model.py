from __future__ import annotations

"""V35 observed-state latent dynamics.

The current world state is action independent and is formed only from observed
visual/state/executed-action evidence. Candidate actions enter only the shared
segment-recurrent transition. Every four-step action segment is consumed once.
Metadata is key/query-only; all value paths are zero preserving.
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Sequence
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint


def sinusoidal_positions(positions: Sequence[int], hidden: int) -> Tensor:
    pos = torch.tensor(tuple(int(x) for x in positions), dtype=torch.float32)[:, None]
    half = hidden // 2
    if half == 0:
        return torch.zeros(len(positions), hidden)
    freq = torch.exp(-math.log(10000.0) * torch.arange(half) / max(half - 1, 1))
    out = torch.cat([torch.sin(pos * freq), torch.cos(pos * freq)], dim=-1)
    if out.shape[-1] < hidden:
        out = F.pad(out, (0, hidden - out.shape[-1]))
    return out[:, :hidden]


class BiasFreeFFN(nn.Module):
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


class ZeroPreservingSelfBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, expansion: float = 4.0) -> None:
        super().__init__()
        self.n1 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True, bias=False)
        self.n2 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.ffn = BiasFreeFFN(hidden, expansion)

    def forward(self, x: Tensor, *, key_bias: Tensor | None = None, causal: bool = False) -> Tensor:
        value = self.n1(x)
        qk = value if key_bias is None else value + key_bias
        mask = None
        if causal:
            n = x.shape[1]
            mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=x.device), diagonal=1)
        update, _ = self.attn(qk, qk, value, attn_mask=mask, need_weights=False)
        x = x + update
        return x + self.ffn(self.n2(x))


@dataclass(frozen=True)
class V35WorldConfig:
    latent_dim: int = 768
    action_dim: int = 7
    state_dim: int = 7
    world_horizon: int = 48
    segment_length: int = 4
    history_length: int = 3
    executed_history_length: int = 3
    num_cameras: int = 2
    patches_per_camera: int = 256
    hidden_size: int = 320
    num_heads: int = 8
    world_tokens: int = 32
    global_tokens: int = 8
    interaction_tokens: int = 12
    motion_tokens: int = 12
    evidence_read_depth: int = 2
    latent_mix_depth: int = 2
    action_depth: int = 2
    transition_depth: int = 4
    transition_unique_blocks: int = 2
    inverse_depth: int = 2
    descriptor_projection_dim: int = 32
    descriptor_regions: int = 5
    ffn_expansion: float = 3.0
    gripper_dim_index: int = -1
    descriptor_seed: int = 35035
    adaln_zero_init: bool = True
    rollout_checkpoint: bool = False
    rollout_checkpoint_preserve_rng_state: bool = True
    consequence_slots: int = 3
    consequence_feedback_depth: int = 1
    consequence_temperature: float = 0.7

    def validate(self) -> None:
        ints = (
            self.latent_dim, self.action_dim, self.state_dim, self.world_horizon,
            self.segment_length, self.history_length, self.executed_history_length,
            self.num_cameras, self.patches_per_camera, self.hidden_size, self.num_heads,
            self.world_tokens, self.global_tokens, self.interaction_tokens, self.motion_tokens,
            self.evidence_read_depth, self.latent_mix_depth, self.action_depth,
            self.transition_depth, self.transition_unique_blocks, self.inverse_depth, self.descriptor_projection_dim,
            self.descriptor_regions, self.consequence_slots, self.consequence_feedback_depth,
        )
        if min(ints) <= 0:
            raise ValueError("all V35 dimensions must be positive")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.world_horizon % self.segment_length:
            raise ValueError("world_horizon must be divisible by segment_length")
        if self.global_tokens + self.interaction_tokens + self.motion_tokens != self.world_tokens:
            raise ValueError("world role token counts must sum to world_tokens")
        if self.action_dim != self.state_dim:
            raise ValueError("action and state dimensions must match")
        if self.transition_unique_blocks > self.transition_depth:
            raise ValueError("transition_unique_blocks cannot exceed transition_depth")
        if self.ffn_expansion < 1.0:
            raise ValueError("ffn_expansion must be >= 1")
        if self.consequence_temperature <= 0:
            raise ValueError("consequence_temperature must be positive")
        idx = self.gripper_index
        if not 0 <= idx < self.state_dim:
            raise ValueError("gripper index out of range")

    @property
    def gripper_index(self) -> int:
        return self.gripper_dim_index if self.gripper_dim_index >= 0 else self.state_dim + self.gripper_dim_index

    @property
    def segment_offsets(self) -> tuple[int, ...]:
        return tuple(range(self.segment_length, self.world_horizon + 1, self.segment_length))

    @property
    def num_segments(self) -> int:
        return len(self.segment_offsets)

    @property
    def num_future(self) -> int:
        return self.num_segments

    @property
    def future_offsets(self) -> tuple[int, ...]:
        return self.segment_offsets

    @property
    def action_horizon(self) -> int:
        return self.world_horizon

    @property
    def role_slices(self) -> dict[str, slice]:
        g = self.global_tokens
        i = g + self.interaction_tokens
        return {"global": slice(0, g), "interaction": slice(g, i), "motion": slice(i, self.world_tokens)}


class EvidenceReadLayer(nn.Module):
    def __init__(self, hidden: int, heads: int, expansion: float) -> None:
        super().__init__()
        self.qn = nn.LayerNorm(hidden, elementwise_affine=False)
        self.kn = nn.LayerNorm(hidden, elementwise_affine=False)
        self.vn = nn.LayerNorm(hidden, elementwise_affine=False)
        self.cross = nn.MultiheadAttention(hidden, heads, batch_first=True, bias=False)
        self.mix = ZeroPreservingSelfBlock(hidden, heads, expansion)

    def read(self, latent: Tensor, slot_key: Tensor, evidence_key: Tensor, evidence_value: Tensor) -> Tensor:
        update, _ = self.cross(
            self.qn(latent) + slot_key,
            self.kn(evidence_key),
            self.vn(evidence_value),
            need_weights=False,
        )
        return update

    def forward(self, latent: Tensor, slot_key: Tensor, evidence_key: Tensor, evidence_value: Tensor) -> Tensor:
        latent = latent + self.read(latent, slot_key, evidence_key, evidence_value)
        return self.mix(latent, key_bias=slot_key)


class WorldEvidenceEncoder(nn.Module):
    """Action-independent current world encoder.

    Learned slots are only queries. Visual/state/executed-action values are the
    sole content source, so zero evidence maps to an exactly zero world state.
    """

    def __init__(self, config: V35WorldConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        h = config.hidden_size
        self.visual_norm = nn.LayerNorm(config.latent_dim, elementwise_affine=False)
        self.visual_proj = nn.Linear(config.latent_dim, h, bias=False)
        self.state_norm = nn.LayerNorm(config.state_dim, elementwise_affine=False)
        self.state_proj = nn.Linear(config.state_dim, h, bias=False)
        self.executed_norm = nn.LayerNorm(config.action_dim, elementwise_affine=False)
        self.executed_proj = nn.Linear(config.action_dim, h, bias=False)

        self.visual_history_key = nn.Parameter(torch.randn(1, config.history_length, 1, 1, h) * 0.02)
        self.camera_key = nn.Parameter(torch.randn(1, 1, config.num_cameras, 1, h) * 0.02)
        self.patch_key = nn.Parameter(torch.randn(1, 1, 1, config.patches_per_camera, h) * 0.02)
        self.visual_type_key = nn.Parameter(torch.randn(1, 1, 1, 1, h) * 0.02)
        self.state_time_key = nn.Parameter(torch.randn(1, config.history_length, h) * 0.02)
        self.state_type_key = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        self.executed_time_key = nn.Parameter(torch.randn(1, config.executed_history_length, h) * 0.02)
        self.executed_type_key = nn.Parameter(torch.randn(1, 1, h) * 0.02)

        self.global_slot = nn.Parameter(torch.randn(1, config.global_tokens, h) * 0.02)
        self.interaction_slot = nn.Parameter(torch.randn(1, config.interaction_tokens, h) * 0.02)
        self.motion_slot = nn.Parameter(torch.randn(1, config.motion_tokens, h) * 0.02)
        self.read_layers = nn.ModuleList(
            [EvidenceReadLayer(h, config.num_heads, config.ffn_expansion) for _ in range(config.evidence_read_depth)]
        )
        self.mix_layers = nn.ModuleList(
            [ZeroPreservingSelfBlock(h, config.num_heads, config.ffn_expansion) for _ in range(config.latent_mix_depth)]
        )
        self.out_norm = nn.LayerNorm(h, elementwise_affine=False)

    def slot_key(self, batch: int, dtype: torch.dtype, device: torch.device) -> Tensor:
        return torch.cat([self.global_slot, self.interaction_slot, self.motion_slot], dim=1).to(
            device=device, dtype=dtype
        ).expand(batch, -1, -1)

    def evidence(
        self,
        visual: Tensor,
        state_history: Tensor,
        executed_action_history: Tensor,
    ) -> tuple[Tensor, Tensor]:
        cfg = self.config
        if tuple(visual.shape[1:]) != (
            cfg.history_length, cfg.num_cameras, cfg.patches_per_camera, cfg.latent_dim
        ):
            raise ValueError("visual evidence geometry mismatch")
        if tuple(state_history.shape[1:]) != (cfg.history_length, cfg.state_dim):
            raise ValueError("state history geometry mismatch")
        if tuple(executed_action_history.shape[1:]) != (
            cfg.executed_history_length, cfg.action_dim
        ):
            raise ValueError("executed action history geometry mismatch")

        # The dense visual conditioner may emit bf16 tokens to save memory while
        # the V35 world model intentionally keeps fp32 master weights for stable
        # training/checkpointing.  encode_online() is also used by evaluation
        # ablations and deployment utilities outside an autocast region, so the
        # encoder boundary must normalize evidence tensors to the module dtype
        # before Linear/Attention layers see them.
        evidence_dtype = self.visual_proj.weight.dtype
        visual = visual.to(dtype=evidence_dtype)
        state_history = state_history.to(dtype=evidence_dtype)
        executed_action_history = executed_action_history.to(dtype=evidence_dtype)

        vv = self.visual_proj(self.visual_norm(visual))
        vk = vv + self.visual_history_key + self.camera_key + self.patch_key + self.visual_type_key
        vv, vk = vv.flatten(1, 3), vk.flatten(1, 3)

        sv = self.state_proj(self.state_norm(state_history))
        sk = sv + self.state_time_key + self.state_type_key
        av = self.executed_proj(self.executed_norm(executed_action_history))
        ak = av + self.executed_time_key + self.executed_type_key
        return torch.cat([vk, sk, ak], dim=1), torch.cat([vv, sv, av], dim=1)

    def forward(self, visual: Tensor, state_history: Tensor, executed_action_history: Tensor) -> Tensor:
        key, value = self.evidence(visual, state_history, executed_action_history)
        slots = self.slot_key(visual.shape[0], key.dtype, visual.device)
        latent = self.read_layers[0].read(torch.zeros_like(slots), slots, key, value)
        latent = self.read_layers[0].mix(latent, key_bias=slots)
        for layer in self.read_layers[1:]:
            latent = layer(latent, slots, key, value)
        for layer in self.mix_layers:
            latent = layer(latent, key_bias=slots)
        return self.out_norm(latent)


class FutureActionTokenizer(nn.Module):
    def __init__(self, config: V35WorldConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.input = nn.Sequential(
            nn.Linear(4 * config.action_dim, 2 * h, bias=False),
            nn.SiLU(),
            nn.Linear(2 * h, h, bias=False),
        )
        self.register_buffer(
            "position_key", sinusoidal_positions(range(1, config.world_horizon + 1), h)[None], persistent=True
        )
        self.blocks = nn.ModuleList(
            [ZeroPreservingSelfBlock(h, config.num_heads, config.ffn_expansion) for _ in range(config.action_depth)]
        )
        self.out_norm = nn.LayerNorm(h, elementwise_affine=False)

    def features(self, action: Tensor, action_state: Tensor) -> Tensor:
        boundary = torch.cat([action_state[:, None], action[:, :-1]], dim=1)
        velocity = action - boundary
        previous_velocity = torch.cat([torch.zeros_like(velocity[:, :1]), velocity[:, :-1]], dim=1)
        acceleration = velocity - previous_velocity
        relative = action - action_state[:, None]
        return torch.cat([action, velocity, acceleration, relative], dim=-1)

    def encode(self, action: Tensor, action_state: Tensor) -> Tensor:
        x = self.input(self.features(action, action_state))
        key = self.position_key.to(device=x.device, dtype=x.dtype)
        for block in self.blocks:
            x = block(x, key_bias=key, causal=True)
        return self.out_norm(x)

    def forward(self, action: Tensor, action_state: Tensor) -> dict[str, Tensor]:
        cfg = self.config
        if tuple(action.shape[1:]) != (cfg.world_horizon, cfg.action_dim):
            raise ValueError("future action geometry mismatch")
        hold = action_state[:, None].expand(-1, cfg.world_horizon, -1)
        encoded = self.encode(
            torch.cat([action, hold], dim=0),
            torch.cat([action_state, action_state], dim=0),
        )
        actual, hold_tokens = encoded.chunk(2, dim=0)
        return {"actual_tokens": actual, "hold_tokens": hold_tokens, "hold_action": hold}




class ConsequenceCrossAttention(nn.Module):
    """Small temporal cross-attention primitive used by consequence slots.

    This deliberately exposes the attention distribution because V35.6 treats
    consequence localization as part of the world model, not as an external
    top-k/keyframe heuristic.  It is single-distribution attention over temporal
    segments; value mixing remains learned through linear projections.
    """

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.q = nn.Linear(hidden, hidden, bias=False)
        self.k = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, hidden, bias=False)
        self.out = nn.Linear(hidden, hidden, bias=False)
        self.qn = nn.LayerNorm(hidden, elementwise_affine=False)
        self.mn = nn.LayerNorm(hidden, elementwise_affine=False)

    def forward(self, query: Tensor, memory: Tensor, *, temperature: float = 1.0) -> tuple[Tensor, Tensor]:
        q = self.q(self.qn(query))
        m = self.mn(memory)
        k = self.k(m)
        v = self.v(m)
        scale = math.sqrt(max(q.shape[-1], 1)) * max(float(temperature), 1e-4)
        scores = torch.einsum("bkh,bsh->bks", q, k) / scale
        attention = scores.softmax(dim=-1)
        context = torch.einsum("bks,bsh->bkh", attention, v)
        return self.out(context), attention


class ConsequenceFeedbackAnchorer(nn.Module):
    """Learned consequence-query cross-attention over temporal latent memory.

    V35.5 dense overshooting treated every adjacent segment start as equally
    important.  V35.6 keeps the existing world encoder/action-tokenizer/dynamics
    stack, but adds a small internal attention module that asks which temporal
    regions have high future consequence.  The module does not perform hard
    top-k selection; it produces K sparse-ish consequence distributions that are
    later used to form differentiable anchor states and action contexts.
    """

    def __init__(self, config: V35WorldConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.query = nn.Parameter(torch.randn(1, config.consequence_slots, h) * 0.02)
        self.memory_proj = nn.Linear(5 * h, h, bias=False)
        self.risk_proj = nn.Linear(4, h, bias=False)
        self.first = ConsequenceCrossAttention(h)
        self.feedback = nn.ModuleList(
            [ConsequenceCrossAttention(h) for _ in range(config.consequence_feedback_depth)]
        )
        self.slot_mix = BiasFreeFFN(h, config.ffn_expansion)
        self.slot_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.register_buffer(
            "segment_key",
            sinusoidal_positions(range(config.num_segments), h)[None],
            persistent=True,
        )

    def forward(
        self,
        *,
        target_world: Tensor,
        pred_world: Tensor,
        teacher_world: Tensor,
        action_tokens: Tensor,
        risk_features: Tensor | None = None,
    ) -> dict[str, Tensor]:
        cfg = self.config
        batch = target_world.shape[0]
        action_summary = action_tokens.reshape(
            batch, cfg.num_segments, cfg.segment_length, cfg.hidden_size
        ).mean(dim=2)
        # Stop gradients here: the consequence locator should read the world
        # model's current failure/effect pattern, not create an extra shortcut
        # for the predictive loss to reshape upstream latents.
        target_summary = target_world.detach().mean(dim=-2)
        pred_summary = pred_world.detach().mean(dim=-2)
        teacher_summary = teacher_world.detach().mean(dim=-2)
        delta_summary = (target_world.detach() - pred_world.detach()).mean(dim=-2)
        action_summary = action_summary.detach()
        memory = self.memory_proj(torch.cat(
            [target_summary, pred_summary, teacher_summary, delta_summary, action_summary], dim=-1
        ))
        memory = memory + self.segment_key.to(device=memory.device, dtype=memory.dtype)
        if risk_features is None:
            risk_features = memory.new_zeros(batch, cfg.num_segments, 4)
        risk = risk_features.detach().to(device=memory.device, dtype=memory.dtype)
        feedback_memory = memory + self.risk_proj(risk)
        query = self.query.to(device=memory.device, dtype=memory.dtype).expand(batch, -1, -1)
        context, attention = self.first(query, memory, temperature=cfg.consequence_temperature)
        slots = query + context
        for layer in self.feedback:
            update, attention = layer(slots, feedback_memory, temperature=cfg.consequence_temperature)
            slots = slots + update + self.slot_mix(self.slot_norm(slots))
        entropy = -(attention.clamp_min(1e-8) * attention.clamp_min(1e-8).log()).sum(dim=-1)
        norm_entropy = entropy / math.log(max(cfg.num_segments, 2))
        overlap = attention @ attention.transpose(-1, -2)
        eye = torch.eye(attention.shape[1], device=attention.device, dtype=torch.bool)[None]
        diversity = overlap.masked_select(~eye).mean() if attention.shape[1] > 1 else attention.new_zeros(())
        return {
            "attention": attention,
            "slots": slots,
            "entropy": norm_entropy.mean(),
            "diversity": diversity,
            "peak": attention.max(dim=-1).values.mean(),
            "expected_start": (attention * torch.arange(cfg.num_segments, device=attention.device, dtype=attention.dtype)).sum(dim=-1).mean(),
        }

class SegmentJointBlock(nn.Module):
    def __init__(self, config: V35WorldConfig) -> None:
        super().__init__()
        h = config.hidden_size
        heads = config.num_heads
        self.config = config
        self.base_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.base_attn = nn.MultiheadAttention(h, heads, batch_first=True, bias=False)
        self.base_ffn_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.base_ffn = BiasFreeFFN(h, config.ffn_expansion)

        self.world_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.action_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.action_reads_world = nn.MultiheadAttention(h, heads, batch_first=True, bias=False)
        self.action_self = nn.Linear(h, 2 * h, bias=False)
        self.action_world = nn.Linear(h, 2 * h, bias=False)
        self.action_out = nn.Linear(2 * h, h, bias=False)
        self.world_reads_action = nn.MultiheadAttention(h, heads, batch_first=True, bias=False)

        self.world_factor = nn.Linear(h, 2 * h, bias=False)
        self.action_factor = nn.Linear(h, 2 * h, bias=False)
        self.world_norm_factor = nn.Linear(h, 2 * h, bias=False)
        self.action_norm_factor = nn.Linear(h, 2 * h, bias=False)
        self.joint_mix = nn.Linear(4 * h, 2 * h, bias=False)
        self.joint_strength = nn.Parameter(torch.ones(1, 1, 1))

        self.global_mod = nn.Linear(2 * h, 6 * h, bias=False)
        self.interaction_mod = nn.Linear(2 * h, 6 * h, bias=False)
        self.motion_mod = nn.Linear(2 * h, 6 * h, bias=False)
        if config.adaln_zero_init:
            nn.init.zeros_(self.global_mod.weight)
            nn.init.zeros_(self.interaction_mod.weight)
            nn.init.zeros_(self.motion_mod.weight)

        self.cond_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.cond_attn = nn.MultiheadAttention(h, heads, batch_first=True, bias=False)
        self.cond_ffn_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.cond_ffn = BiasFreeFFN(h, config.ffn_expansion)

    def typed_modulation(self, joint: Tensor) -> Tensor:
        s = self.config.role_slices
        return torch.cat(
            [
                self.global_mod(joint[:, s["global"]]),
                self.interaction_mod(joint[:, s["interaction"]]),
                self.motion_mod(joint[:, s["motion"]]),
            ],
            dim=1,
        )

    def forward(
        self,
        world: Tensor,
        local_action: Tensor,
        *,
        world_key: Tensor,
        action_key: Tensor,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        # Action-free evolution. Metadata selects content but never supplies values.
        base_value = self.base_norm(world)
        base_qk = base_value + world_key
        base, _ = self.base_attn(base_qk, base_qk, base_value, need_weights=False)
        world = world + base
        world = world + self.base_ffn(self.base_ffn_norm(world))

        action_value = self.action_norm(local_action)
        world_value = self.world_norm(world)
        action_read, _ = self.action_reads_world(
            action_value + action_key,
            world_value + world_key,
            world_value,
            need_weights=False,
        )
        action_joint = self.action_self(action_value) * self.action_world(action_read)
        local_action = local_action + self.action_out(F.silu(action_joint))

        action_value = self.action_norm(local_action)
        action_signal, _ = self.world_reads_action(
            self.world_norm(world) + world_key,
            action_value + action_key,
            action_value,
            need_weights=False,
        )
        world_content = self.world_norm(world)
        raw = self.world_factor(world_content) * self.action_factor(action_signal)
        normalized = self.world_norm_factor(world_content) * self.action_norm_factor(
            self.action_norm(action_signal)
        )
        joint = self.joint_strength * F.silu(self.joint_mix(torch.cat([raw, normalized], dim=-1)))
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = self.typed_modulation(joint).chunk(6, dim=-1)

        value = self.cond_norm(world)
        qk = value * (1 + scale_a) + shift_a + world_key
        update, _ = self.cond_attn(qk, qk, value, need_weights=False)
        world = world + torch.tanh(gate_a) * update
        ffn_in = self.cond_ffn_norm(world) * (1 + scale_f) + shift_f
        world = world + torch.tanh(gate_f) * self.cond_ffn(ffn_in)
        return world, local_action, {
            "adaln_gate_abs_mean": 0.5 * (torch.tanh(gate_a).abs().mean() + torch.tanh(gate_f).abs().mean()),
            "world_action_joint_rms": joint.float().square().mean().sqrt(),
            "action_signal_rms": action_signal.float().square().mean().sqrt(),
        }


class SegmentRecurrentLatentDynamics(nn.Module):
    def __init__(self, config: V35WorldConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.role_key = nn.Parameter(torch.randn(1, config.world_tokens, h) * 0.02)
        self.register_buffer(
            "segment_key", sinusoidal_positions(config.segment_offsets, h)[None, :, None], persistent=True
        )
        self.register_buffer(
            "action_key", sinusoidal_positions(range(1, config.world_horizon + 1), h)[None], persistent=True
        )
        self.history_key_proj = nn.Linear(h, h, bias=False)
        self.blocks = nn.ModuleList([SegmentJointBlock(config) for _ in range(config.transition_unique_blocks)])
        self.out_norm = nn.LayerNorm(h, elementwise_affine=False)

    def segment_inputs(self, action_tokens: Tensor, segment_index: int) -> tuple[Tensor, Tensor]:
        cfg = self.config
        start = segment_index * cfg.segment_length
        end = start + cfg.segment_length
        local = action_tokens[:, start:end]
        key = self.action_key[:, start:end].to(device=local.device, dtype=local.dtype).expand(
            local.shape[0], -1, -1
        )
        if start > 0:
            # Historical action is read-only context. It changes keys, never local values.
            history = action_tokens[:, :start].mean(dim=1, keepdim=True)
            key = key + self.history_key_proj(history).expand(-1, cfg.segment_length, -1)
        return local, key

    def _step_tensors(
        self, world: Tensor, action_tokens: Tensor, segment_index: int
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Run one segment transition and return tensor-only diagnostics.

        This helper is intentionally tensor-only so it can be wrapped by
        torch.utils.checkpoint.  The public step() reconstructs the diagnostic
        dictionary outside the checkpoint boundary.
        """
        local, action_key = self.segment_inputs(action_tokens, segment_index)
        world_key = (
            self.role_key + self.segment_key[:, segment_index]
        ).to(device=world.device, dtype=world.dtype).expand(world.shape[0], -1, -1)
        rows: dict[str, list[Tensor]] = {}
        for depth_index in range(self.config.transition_depth):
            block = self.blocks[depth_index % len(self.blocks)]
            world, local, diagnostics = block(
                world, local, world_key=world_key, action_key=action_key
            )
            for name, value in diagnostics.items():
                rows.setdefault(name, []).append(value)
        world = self.out_norm(world)
        adaln_gate = torch.stack(rows["adaln_gate_abs_mean"]).mean()
        joint_rms = torch.stack(rows["world_action_joint_rms"]).mean()
        signal_rms = torch.stack(rows["action_signal_rms"]).mean()
        world_rms = world.float().square().mean().sqrt()
        return world, adaln_gate, joint_rms, signal_rms, world_rms

    def _checkpointed_step_tensors(
        self, world: Tensor, action_tokens: Tensor, segment_index: int
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        return activation_checkpoint(
            lambda world_arg, action_arg: self._step_tensors(world_arg, action_arg, segment_index),
            world,
            action_tokens,
            use_reentrant=False,
            preserve_rng_state=bool(self.config.rollout_checkpoint_preserve_rng_state),
        )

    def step(self, world: Tensor, action_tokens: Tensor, segment_index: int) -> tuple[Tensor, dict[str, Tensor]]:
        use_checkpoint = (
            bool(self.config.rollout_checkpoint)
            and self.training
            and torch.is_grad_enabled()
        )
        if use_checkpoint:
            world, adaln_gate, joint_rms, signal_rms, world_rms = self._checkpointed_step_tensors(
                world, action_tokens, segment_index
            )
        else:
            world, adaln_gate, joint_rms, signal_rms, world_rms = self._step_tensors(
                world, action_tokens, segment_index
            )
        diagnostics = {
            "adaln_gate_abs_mean": adaln_gate.detach(),
            "world_action_joint_rms": joint_rms.detach(),
            "action_signal_rms": signal_rms.detach(),
            "world_rms": world_rms.detach(),
        }
        return world, diagnostics

    def rollout_from_latent(
        self,
        initial_world: Tensor,
        action_tokens: Tensor,
        *,
        start_segment: int = 0,
        num_segments: int | None = None,
    ) -> dict[str, Tensor]:
        """Closed-loop rollout from an arbitrary latent state.

        This is the core V35.5 interface: after the first transition, each
        segment consumes the model's own previous latent prediction.  The start
        latent may be an online current latent, a detached target latent for
        multi-start overshooting, or a policy-produced latent in later rollout
        consequence training.
        """
        cfg = self.config
        start = int(start_segment)
        if start < 0 or start >= cfg.num_segments:
            raise ValueError("start_segment out of range")
        count = cfg.num_segments - start if num_segments is None else int(num_segments)
        count = max(0, min(count, cfg.num_segments - start))
        world = initial_world
        worlds: list[Tensor] = []
        diagnostics: dict[str, list[Tensor]] = {}
        for segment_index in range(start, start + count):
            world, row = self.step(world, action_tokens, segment_index)
            worlds.append(world)
            for name, value in row.items():
                diagnostics.setdefault(name, []).append(value)
        dense = (
            torch.stack(worlds, dim=1)
            if worlds
            else initial_world.new_empty(initial_world.shape[0], 0, cfg.world_tokens, cfg.hidden_size)
        )
        return {
            "world": dense,
            **{
                name: torch.stack(values).mean()
                for name, values in diagnostics.items()
                if values
            },
        }


    def segment_contexts(self, action_tokens: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return per-segment local action values/keys and world keys.

        This preserves the original SegmentJointBlock transition operator while
        allowing V35.6 consequence anchors to run grouped soft-start rollouts
        without Python-looping over every dense start.
        """
        cfg = self.config
        locals_, action_keys, world_keys = [], [], []
        for segment_index in range(cfg.num_segments):
            local, action_key = self.segment_inputs(action_tokens, segment_index)
            world_key = (self.role_key + self.segment_key[:, segment_index]).to(
                device=action_tokens.device, dtype=action_tokens.dtype
            ).expand(action_tokens.shape[0], -1, -1)
            locals_.append(local)
            action_keys.append(action_key)
            world_keys.append(world_key)
        return torch.stack(locals_, dim=1), torch.stack(action_keys, dim=1), torch.stack(world_keys, dim=1)

    def _step_context_tensors(
        self, world: Tensor, local_action: Tensor, action_key: Tensor, world_key: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        local = local_action
        rows: dict[str, list[Tensor]] = {}
        for depth_index in range(self.config.transition_depth):
            block = self.blocks[depth_index % len(self.blocks)]
            world, local, diagnostics = block(
                world, local, world_key=world_key, action_key=action_key
            )
            for name, value in diagnostics.items():
                rows.setdefault(name, []).append(value)
        world = self.out_norm(world)
        adaln_gate = torch.stack(rows["adaln_gate_abs_mean"]).mean()
        joint_rms = torch.stack(rows["world_action_joint_rms"]).mean()
        signal_rms = torch.stack(rows["action_signal_rms"]).mean()
        world_rms = world.float().square().mean().sqrt()
        return world, adaln_gate, joint_rms, signal_rms, world_rms

    def step_with_context(
        self, world: Tensor, local_action: Tensor, action_key: Tensor, world_key: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        use_checkpoint = (
            bool(self.config.rollout_checkpoint)
            and self.training
            and torch.is_grad_enabled()
        )
        if use_checkpoint:
            world, adaln_gate, joint_rms, signal_rms, world_rms = activation_checkpoint(
                lambda w, a, ak, wk: self._step_context_tensors(w, a, ak, wk),
                world,
                local_action,
                action_key,
                world_key,
                use_reentrant=False,
                preserve_rng_state=bool(self.config.rollout_checkpoint_preserve_rng_state),
            )
        else:
            world, adaln_gate, joint_rms, signal_rms, world_rms = self._step_context_tensors(
                world, local_action, action_key, world_key
            )
        return world, {
            "adaln_gate_abs_mean": adaln_gate.detach(),
            "world_action_joint_rms": joint_rms.detach(),
            "action_signal_rms": signal_rms.detach(),
            "world_rms": world_rms.detach(),
        }

    def rollout_pair(self, initial_world: Tensor, actual_tokens: Tensor, hold_tokens: Tensor) -> dict[str, Tensor]:
        batch = initial_world.shape[0]
        world = torch.cat([initial_world, initial_world], dim=0)
        action = torch.cat([actual_tokens, hold_tokens], dim=0)
        rollout = self.rollout_from_latent(world, action, start_segment=0, num_segments=self.config.num_segments)
        dense = rollout["world"]
        actual, hold = dense[:batch], dense[batch:]
        diagnostics = {key: value for key, value in rollout.items() if key != "world"}
        return {
            "pred_world": actual,
            "hold_world": hold,
            "action_world_effect": actual - hold,
            **diagnostics,
        }


class EndpointStateDecoder(nn.Module):
    def __init__(self, config: V35WorldConfig) -> None:
        super().__init__()
        h = config.hidden_size
        self.query = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        self.qn = nn.LayerNorm(h, elementwise_affine=False)
        self.mn = nn.LayerNorm(h, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(h, config.num_heads, batch_first=True, bias=False)
        self.head = nn.Linear(h, config.state_dim, bias=False)

    def forward(self, world: Tensor) -> Tensor:
        shape = world.shape[:-2]
        flat = world.reshape(-1, world.shape[-2], world.shape[-1])
        memory = self.mn(flat)
        query = self.query.to(dtype=flat.dtype).expand(flat.shape[0], -1, -1)
        x, _ = self.attn(self.qn(query), memory, memory, need_weights=False)
        return self.head(x[:, 0]).reshape(*shape, -1)


class SegmentInverseDecoder(nn.Module):
    def __init__(self, config: V35WorldConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.current = nn.Linear(h, h, bias=False)
        self.change = nn.Linear(h, h, bias=False)
        self.query = nn.Parameter(torch.randn(1, config.segment_length, h) * 0.02)
        self.qn = nn.LayerNorm(h, elementwise_affine=False)
        self.mn = nn.LayerNorm(h, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(h, config.num_heads, batch_first=True, bias=False)
        self.blocks = nn.ModuleList(
            [ZeroPreservingSelfBlock(h, config.num_heads, config.ffn_expansion) for _ in range(config.inverse_depth)]
        )
        self.action_head = nn.Linear(h, config.action_dim, bias=False)
        self.gripper_head = nn.Linear(h, 3, bias=False)

    def forward(self, current: Tensor, next_world: Tensor) -> dict[str, Tensor]:
        leading = next_world.shape[:-2]
        current_expanded = current
        while current_expanded.ndim < next_world.ndim:
            current_expanded = current_expanded.unsqueeze(1)
        current_expanded = current_expanded.expand_as(next_world)
        memory = self.current(current_expanded) * self.change(next_world - current_expanded)
        flat = memory.reshape(-1, memory.shape[-2], memory.shape[-1])
        query = self.query.to(dtype=flat.dtype).expand(flat.shape[0], -1, -1)
        x, _ = self.attn(self.qn(query), self.mn(flat), self.mn(flat), need_weights=False)
        for block in self.blocks:
            x = block(x)
        action = self.action_head(x).reshape(*leading, self.config.segment_length, self.config.action_dim)
        gripper = self.gripper_head(x).reshape(*leading, self.config.segment_length, 3)
        return {"action": action, "gripper_logits": gripper}


class RegionDescriptorDecoder(nn.Module):
    def __init__(self, config: V35WorldConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size
        count = config.history_length * config.num_cameras * config.descriptor_regions
        self.register_buffer("query", sinusoidal_positions(range(count), h)[None], persistent=True)
        self.qn = nn.LayerNorm(h, elementwise_affine=False)
        self.mn = nn.LayerNorm(h, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(h, config.num_heads, batch_first=True, bias=False)
        self.head = nn.Linear(h, config.descriptor_projection_dim, bias=False)

    def forward(self, world: Tensor) -> Tensor:
        leading = world.shape[:-2]
        flat = world.reshape(-1, world.shape[-2], world.shape[-1])
        query = self.query.to(dtype=flat.dtype).expand(flat.shape[0], -1, -1)
        memory = self.mn(flat)
        x, _ = self.attn(self.qn(query), memory, memory, need_weights=False)
        return self.head(x).reshape(
            *leading,
            self.config.history_length,
            self.config.num_cameras,
            self.config.descriptor_regions,
            self.config.descriptor_projection_dim,
        )


class ActionOnlyProbe(nn.Module):
    """Isolated diagnostic probe. Its gradients never enter the main world model."""

    def __init__(self, config: V35WorldConfig) -> None:
        super().__init__()
        h = config.hidden_size
        self.input = nn.Linear(config.action_dim, h)
        self.position = nn.Parameter(torch.randn(1, config.world_horizon, h) * 0.02)
        self.block = ZeroPreservingSelfBlock(h, config.num_heads, config.ffn_expansion)
        self.query = nn.Parameter(torch.randn(1, config.num_segments * 3, h) * 0.02)
        self.cross = nn.MultiheadAttention(h, config.num_heads, batch_first=True)
        self.head = nn.Linear(h, h)
        self.config = config

    def forward(self, action: Tensor) -> Tensor:
        x = self.input(action)
        x = self.block(x, key_bias=self.position, causal=True)
        q = self.query.expand(action.shape[0], -1, -1)
        out, _ = self.cross(q, x + self.position, x, need_weights=False)
        return self.head(out).reshape(action.shape[0], self.config.num_segments, 3, self.config.hidden_size)


class V35ObservedStateWorldModel(nn.Module):
    def __init__(self, config: V35WorldConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.online_encoder = WorldEvidenceEncoder(config)
        self.target_encoder = deepcopy(self.online_encoder).float()
        self.target_encoder.requires_grad_(False)
        self.target_encoder.eval()
        self.action_tokenizer = FutureActionTokenizer(config)
        self.dynamics = SegmentRecurrentLatentDynamics(config)
        self.consequence_anchorer = ConsequenceFeedbackAnchorer(config)
        self.state_decoder = EndpointStateDecoder(config)
        self.inverse_decoder = SegmentInverseDecoder(config)
        self.region_decoder = RegionDescriptorDecoder(config)
        self.action_only_probe = ActionOnlyProbe(config)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.descriptor_seed)
        matrix = torch.randn(config.latent_dim, config.descriptor_projection_dim, generator=generator)
        q, _ = torch.linalg.qr(matrix, mode="reduced")
        self.register_buffer("descriptor_projection", q.float(), persistent=True)

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_encoder.eval()
        return self

    def split_world(self, world: Tensor) -> dict[str, Tensor]:
        return {name: world[..., sl, :] for name, sl in self.config.role_slices.items()}

    def world_summary(self, world: Tensor) -> Tensor:
        split = self.split_world(world)
        return torch.stack([split[name].mean(dim=-2) for name in ("global", "interaction", "motion")], dim=-2)

    def fixed_region_descriptor(self, tokens: Tensor) -> Tensor:
        """Global + four quadrant descriptors per frame/camera."""
        cfg = self.config
        projected = tokens.float() @ self.descriptor_projection.float()
        patch_count = projected.shape[-2]
        side = int(round(math.sqrt(patch_count)))
        if side * side != patch_count or cfg.descriptor_regions != 5:
            global_region = projected.mean(dim=-2, keepdim=True)
            regions = global_region.expand(*projected.shape[:-2], cfg.descriptor_regions, projected.shape[-1])
        else:
            grid = projected.reshape(*projected.shape[:-2], side, side, projected.shape[-1])
            half = side // 2
            quads = [
                grid[..., :half, :half, :].mean(dim=(-3, -2)),
                grid[..., :half, half:, :].mean(dim=(-3, -2)),
                grid[..., half:, :half, :].mean(dim=(-3, -2)),
                grid[..., half:, half:, :].mean(dim=(-3, -2)),
            ]
            regions = torch.stack([projected.mean(dim=-2), *quads], dim=-2)
        return F.normalize(regions, dim=-1)

    def encode_online(self, visual: Tensor, state_history: Tensor, executed_history: Tensor) -> Tensor:
        return self.online_encoder(visual, state_history, executed_history)

    @torch.no_grad()
    def encode_targets(
        self,
        current_visual: Tensor,
        target_visual: Tensor,
        state_history: Tensor,
        target_state_history: Tensor,
        executed_history: Tensor,
        target_executed_history: Tensor,
    ) -> tuple[Tensor, Tensor]:
        cfg = self.config
        current = self.target_encoder(
            current_visual.float(), state_history.float(), executed_history.float()
        )
        flat_visual = target_visual.reshape(
            -1, cfg.history_length, cfg.num_cameras, cfg.patches_per_camera, cfg.latent_dim
        )
        flat_state = target_state_history.reshape(-1, cfg.history_length, cfg.state_dim)
        flat_action = target_executed_history.reshape(
            -1, cfg.executed_history_length, cfg.action_dim
        )
        future = self.target_encoder(flat_visual.float(), flat_state.float(), flat_action.float()).reshape(
            target_visual.shape[0], cfg.num_segments, cfg.world_tokens, cfg.hidden_size
        )
        return current, future

    @torch.no_grad()
    def encode_online_future(
        self, target_visual: Tensor, target_state_history: Tensor, target_executed_history: Tensor
    ) -> Tensor:
        cfg = self.config
        return self.online_encoder(
            target_visual.reshape(-1, cfg.history_length, cfg.num_cameras, cfg.patches_per_camera, cfg.latent_dim),
            target_state_history.reshape(-1, cfg.history_length, cfg.state_dim),
            target_executed_history.reshape(-1, cfg.executed_history_length, cfg.action_dim),
        ).reshape(target_visual.shape[0], cfg.num_segments, cfg.world_tokens, cfg.hidden_size)

    def teacher_forced_rollout(self, target_initial: Tensor, target_future: Tensor, action_tokens: Tensor) -> Tensor:
        previous = target_initial
        rows: list[Tensor] = []
        for index in range(self.config.num_segments):
            predicted, _ = self.dynamics.step(previous, action_tokens, index)
            rows.append(predicted)
            previous = target_future[:, index].detach()
        return torch.stack(rows, dim=1)

    def multi_start_overshooting(
        self,
        target_initial: Tensor,
        target_future: Tensor,
        action_tokens: Tensor,
        *,
        depth: int = 2,
        detach_start: bool = True,
        detach_target: bool = True,
    ) -> dict[str, Tensor]:
        """Run detached-target multi-start latent overshooting.

        Each row starts either at z_0 or at an interior target latent z_k, then
        recursively applies the same action-conditioned transition operator.
        Targets are returned alongside predictions with a valid mask so losses
        can be weighted by overshoot depth without padding artifacts.
        """
        cfg = self.config
        depth = max(0, min(int(depth), cfg.num_segments))
        if depth <= 0:
            empty = target_future.new_empty(0, cfg.world_tokens, cfg.hidden_size)
            return {
                "overshoot_world": empty,
                "overshoot_target": empty,
                "overshoot_depth_index": target_future.new_empty(0, dtype=torch.long),
                "overshoot_start_index": target_future.new_empty(0, dtype=torch.long),
            }
        pred_rows: list[Tensor] = []
        target_rows: list[Tensor] = []
        depth_rows: list[Tensor] = []
        start_rows: list[Tensor] = []
        for start in range(cfg.num_segments):
            if start == 0:
                start_world = target_initial
            else:
                start_world = target_future[:, start - 1]
            if detach_start:
                start_world = start_world.detach()
            rollout = self.dynamics.rollout_from_latent(
                start_world, action_tokens, start_segment=start, num_segments=min(depth, cfg.num_segments - start)
            )["world"]
            horizon = rollout.shape[1]
            if horizon == 0:
                continue
            target = target_future[:, start:start + horizon]
            if detach_target:
                target = target.detach()
            pred_rows.append(rollout.reshape(-1, cfg.world_tokens, cfg.hidden_size))
            target_rows.append(target.reshape(-1, cfg.world_tokens, cfg.hidden_size))
            depth_ids = torch.arange(1, horizon + 1, device=target_future.device, dtype=torch.long)
            depth_rows.append(depth_ids[None].expand(target_future.shape[0], -1).reshape(-1))
            start_rows.append(torch.full((target_future.shape[0] * horizon,), start, device=target_future.device, dtype=torch.long))
        if not pred_rows:
            empty = target_future.new_empty(0, cfg.world_tokens, cfg.hidden_size)
            return {
                "overshoot_world": empty,
                "overshoot_target": empty,
                "overshoot_depth_index": target_future.new_empty(0, dtype=torch.long),
                "overshoot_start_index": target_future.new_empty(0, dtype=torch.long),
            }
        return {
            "overshoot_world": torch.cat(pred_rows, dim=0),
            "overshoot_target": torch.cat(target_rows, dim=0),
            "overshoot_depth_index": torch.cat(depth_rows, dim=0),
            "overshoot_start_index": torch.cat(start_rows, dim=0),
        }


    def consequence_overshooting(
        self,
        target_initial: Tensor,
        target_future: Tensor,
        pred_world: Tensor,
        teacher_world: Tensor,
        action_tokens: Tensor,
        *,
        depth: int = 2,
        risk_features: Tensor | None = None,
        detach_start: bool = True,
        detach_target: bool = True,
    ) -> dict[str, Tensor]:
        """Consequence-query soft-anchor overshooting.

        Learned consequence queries cross-attend to the full temporal memory and
        produce K soft anchor distributions.  Each distribution forms a latent
        start state and a soft action/segment context; the original recurrent
        transition blocks are then reused for short closed-loop rollout.
        """
        cfg = self.config
        depth = max(0, min(int(depth), cfg.num_segments))
        if depth <= 0 or cfg.consequence_slots <= 0:
            empty = target_future.new_empty(0, cfg.world_tokens, cfg.hidden_size)
            return {
                "overshoot_world": empty,
                "overshoot_target": empty,
                "overshoot_depth_index": target_future.new_empty(0, dtype=torch.long),
                "overshoot_start_index": target_future.new_empty(0, dtype=torch.long),
                "consequence_attention": target_future.new_empty(target_future.shape[0], 0, cfg.num_segments),
                "consequence_entropy": target_future.new_zeros(()),
                "consequence_diversity": target_future.new_zeros(()),
                "consequence_peak": target_future.new_zeros(()),
                "consequence_expected_start": target_future.new_zeros(()),
            }
        anchor = self.consequence_anchorer(
            target_world=target_future,
            pred_world=pred_world,
            teacher_world=teacher_world,
            action_tokens=action_tokens,
            risk_features=risk_features,
        )
        attention = anchor["attention"]
        if detach_start:
            start_candidates = torch.cat([target_initial[:, None], target_future[:, :-1]], dim=1).detach()
        else:
            start_candidates = torch.cat([target_initial[:, None], target_future[:, :-1]], dim=1)
        if detach_target:
            target_future_for_loss = target_future.detach()
        else:
            target_future_for_loss = target_future
        local_segments, action_keys, world_keys = self.dynamics.segment_contexts(action_tokens)
        start_world = torch.einsum("bks,bswh->bkwh", attention, start_candidates)
        batch, slots = start_world.shape[:2]
        world = start_world.reshape(batch * slots, cfg.world_tokens, cfg.hidden_size)
        pred_rows: list[Tensor] = []
        target_rows: list[Tensor] = []
        depth_rows: list[Tensor] = []
        start_rows: list[Tensor] = []
        starts = torch.arange(cfg.num_segments, device=target_future.device)
        for depth_index in range(depth):
            valid = (starts + depth_index) < cfg.num_segments
            weights = attention * valid.to(dtype=attention.dtype)[None, None, :]
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            def weighted_shift(values: Tensor) -> Tensor:
                # values: [B, S, ...] indexed at segment start + depth_index.
                shifted = values.new_zeros(values.shape)
                if depth_index == 0:
                    shifted = values
                else:
                    shifted[:, : cfg.num_segments - depth_index] = values[:, depth_index:]
                return torch.einsum("bks,bs...->bk...", weights, shifted)
            local = weighted_shift(local_segments).reshape(batch * slots, cfg.segment_length, cfg.hidden_size)
            action_key = weighted_shift(action_keys).reshape(batch * slots, cfg.segment_length, cfg.hidden_size)
            world_key = weighted_shift(world_keys).reshape(batch * slots, cfg.world_tokens, cfg.hidden_size)
            target = weighted_shift(target_future_for_loss).reshape(batch * slots, cfg.world_tokens, cfg.hidden_size)
            world, _ = self.dynamics.step_with_context(world, local, action_key, world_key)
            pred_rows.append(world)
            target_rows.append(target)
            depth_rows.append(torch.full((batch * slots,), depth_index + 1, device=target_future.device, dtype=torch.long))
            start_rows.append((attention.detach() * starts.to(dtype=attention.dtype)[None, None]).sum(dim=-1).reshape(-1).to(torch.long))
        return {
            "overshoot_world": torch.cat(pred_rows, dim=0),
            "overshoot_target": torch.cat(target_rows, dim=0),
            "overshoot_depth_index": torch.cat(depth_rows, dim=0),
            "overshoot_start_index": torch.cat(start_rows, dim=0),
            "consequence_attention": attention,
            "consequence_entropy": anchor["entropy"],
            "consequence_diversity": anchor["diversity"],
            "consequence_peak": anchor["peak"],
            "consequence_expected_start": anchor["expected_start"],
        }

    def forward(
        self,
        current_visual: Tensor,
        target_visual: Tensor,
        state_history: Tensor,
        target_state_history: Tensor,
        executed_history: Tensor,
        target_executed_history: Tensor,
        action: Tensor,
        action_state: Tensor,
    ) -> dict[str, Tensor]:
        current = self.encode_online(current_visual, state_history, executed_history)
        target_initial, target = self.encode_targets(
            current_visual,
            target_visual,
            state_history,
            target_state_history,
            executed_history,
            target_executed_history,
        )
        action_output = self.action_tokenizer(action, action_state)
        rollout = self.dynamics.rollout_pair(
            current, action_output["actual_tokens"], action_output["hold_tokens"]
        )
        online_future = self.encode_online_future(
            target_visual, target_state_history, target_executed_history
        )
        online_future_target = online_future.detach()
        teacher = self.teacher_forced_rollout(
            target_initial, target, action_output["actual_tokens"]
        )
        previous_pred = torch.cat([current[:, None], rollout["pred_world"][:, :-1]], dim=1)
        previous_target = torch.cat([current.detach()[:, None], online_future_target[:, :-1]], dim=1)
        inverse_pred = self.inverse_decoder(previous_pred, rollout["pred_world"])
        inverse_target = self.inverse_decoder(previous_target, online_future_target)
        return {
            "initial_world": current,
            "target_initial_world": target_initial,
            "target_world": target,
            "online_future_world": online_future_target,
            "teacher_forced_world": teacher,
            "pred_segment_state": self.state_decoder(rollout["pred_world"]),
            "hold_segment_state": self.state_decoder(rollout["hold_world"]),
            "target_segment_state_prediction": self.state_decoder(online_future_target),
            "current_region_prediction": self.region_decoder(current),
            "pred_region_prediction": self.region_decoder(rollout["pred_world"]),
            "current_region_target": self.fixed_region_descriptor(current_visual).to(current.dtype),
            "future_region_target": self.fixed_region_descriptor(target_visual).to(current.dtype),
            "pred_inverse_action": inverse_pred["action"],
            "pred_inverse_gripper_logits": inverse_pred["gripper_logits"],
            "target_inverse_action": inverse_target["action"],
            "target_inverse_gripper_logits": inverse_target["gripper_logits"],
            "action_only_probe_prediction": self.action_only_probe(action.detach()),
            "action_only_probe_target": self.world_summary(target).detach(),
            **rollout,
            **action_output,
        }

    def forward_pair(
        self,
        current_visual: Tensor,
        target_visual: Tensor,
        state_history: Tensor,
        target_state_history: Tensor,
        executed_history: Tensor,
        target_executed_history: Tensor,
        action: Tensor,
        action_state: Tensor,
    ) -> dict[str, Tensor]:
        """Minimal pair branch for local-effect objectives.

        Pair losses only need the current/target latent trajectory and the
        action-conditioned rollout.  Avoid the full forward pass here because
        target-side auxiliary decoders, online-future encoders, region heads,
        teacher-forced rollout, and action probes are unused by the pair loss
        but otherwise keep large autograd graphs alive.
        """
        current = self.encode_online(current_visual, state_history, executed_history)
        target_initial, target = self.encode_targets(
            current_visual,
            target_visual,
            state_history,
            target_state_history,
            executed_history,
            target_executed_history,
        )
        action_output = self.action_tokenizer(action, action_state)
        rollout = self.dynamics.rollout_pair(
            current, action_output["actual_tokens"], action_output["hold_tokens"]
        )
        return {
            "initial_world": current,
            "target_initial_world": target_initial,
            "target_world": target,
            "pred_world": rollout["pred_world"],
        }

    @torch.no_grad()
    def update_ema(self, decay: float) -> None:
        if not 0 <= float(decay) < 1:
            raise ValueError("EMA decay must be in [0,1)")
        for target, online in zip(self.target_encoder.parameters(), self.online_encoder.parameters(), strict=True):
            target.mul_(float(decay)).add_(online.float(), alpha=1 - float(decay))
        for target, online in zip(self.target_encoder.buffers(), self.online_encoder.buffers(), strict=True):
            target.copy_(online.float())

    def parameter_report(self) -> dict[str, int]:
        modules = {
            "online_world_encoder": self.online_encoder,
            "target_world_encoder": self.target_encoder,
            "future_action_tokenizer": self.action_tokenizer,
            "segment_recurrent_dynamics": self.dynamics,
            "consequence_anchorer": self.consequence_anchorer,
            "state_decoder": self.state_decoder,
            "inverse_decoder": self.inverse_decoder,
            "region_decoder": self.region_decoder,
            "action_only_probe": self.action_only_probe,
        }
        out = {name: sum(p.numel() for p in module.parameters()) for name, module in modules.items()}
        out["total"] = sum(p.numel() for p in self.parameters())
        out["trainable"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        out["online_inference"] = out["total"] - out["target_world_encoder"] - out["inverse_decoder"] - out["action_only_probe"]
        return out


__all__ = [
    "V35WorldConfig",
    "WorldEvidenceEncoder",
    "FutureActionTokenizer",
    "SegmentRecurrentLatentDynamics",
    "V35ObservedStateWorldModel",
]
