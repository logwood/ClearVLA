from __future__ import annotations

"""V36.3 transition-aware action latent policy.

V36.3 keeps the V36.2 typed physical-action-flow contract but closes the
local gripper-event shortcut at the latent level.  It does not add an external
rule-based gripper controller and it does not feed target event labels into the
forward pass.  Instead, transition information is represented as a residual
subspace inside the action latent tokens.  The same fused action latent is used
for event readout and for the physical velocity head, so event supervision must
shape the action latent that produces the final decoded gripper command.
"""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from .policy import RejectableHistoryProposal, TimeEmbedding
from .policy_v36_2 import (
    ActionExpertBlock,
    DiTPlannerBlock,
    HorizonRoleEmbedding,
    PhysicalActionCodec,
    PhysicalActionTokenLift,
    V362PolicyConfig,
)
from .world_model import V35WorldConfig, WorldEvidenceEncoder, sinusoidal_positions


def _dct_action_basis(horizon: int, controls: int, *, device: torch.device | None = None) -> Tensor:
    if horizon < 1 or controls < 1:
        raise ValueError("DCT action basis requires horizon >= 1 and controls >= 1")
    controls = min(int(controls), int(horizon))
    pos = torch.arange(int(horizon), device=device, dtype=torch.float32)
    cols: list[Tensor] = []
    for k in range(controls):
        scale = math.sqrt(1.0 / float(horizon)) if k == 0 else math.sqrt(2.0 / float(horizon))
        cols.append(scale * torch.cos((math.pi / float(horizon)) * (pos + 0.5) * float(k)))
    return torch.stack(cols, dim=-1)


def _bspline_action_basis(
    horizon: int,
    controls: int,
    *,
    degree: int = 2,
    device: torch.device | None = None,
) -> Tensor:
    """Return an open-uniform clamped B-spline basis [horizon, controls]."""

    if horizon < 1 or controls < 1:
        raise ValueError("B-spline action basis requires horizon >= 1 and controls >= 1")
    controls = min(int(controls), int(horizon))
    degree = min(max(int(degree), 0), controls - 1)
    t = torch.linspace(0.0, 1.0, int(horizon), device=device, dtype=torch.float32)
    # Evaluate the final sample from the left, then set the exact clamped
    # endpoint after recursion.  This avoids zero-length terminal knot spans.
    t_eval = t.clamp(max=1.0 - 1e-6)
    interior = int(controls - degree - 1)
    pieces = [
        torch.zeros(degree + 1, device=device, dtype=torch.float32),
    ]
    if interior > 0:
        pieces.append(torch.linspace(0.0, 1.0, interior + 2, device=device, dtype=torch.float32)[1:-1])
    pieces.append(torch.ones(degree + 1, device=device, dtype=torch.float32))
    knots = torch.cat(pieces)

    basis = []
    for idx in range(int(knots.numel()) - 1):
        left = knots[idx]
        right = knots[idx + 1]
        basis.append(((t_eval >= left) & (t_eval < right)).to(dtype=torch.float32))
    current = torch.stack(basis, dim=-1)

    for order in range(1, degree + 1):
        cols: list[Tensor] = []
        for idx in range(int(current.shape[1]) - 1):
            left_den = knots[idx + order] - knots[idx]
            right_den = knots[idx + order + 1] - knots[idx + 1]
            val = torch.zeros_like(t_eval)
            if float(left_den) > 0.0:
                val = val + ((t_eval - knots[idx]) / left_den) * current[:, idx]
            if float(right_den) > 0.0:
                val = val + ((knots[idx + order + 1] - t_eval) / right_den) * current[:, idx + 1]
            cols.append(val)
        current = torch.stack(cols, dim=-1)

    current = current[:, :controls].clamp_min(0.0)
    if int(horizon) > 0:
        current[-1].zero_()
        current[-1, -1] = 1.0
    return current / current.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def _ridge_basis_analysis(basis: Tensor, ridge: float = 0.0) -> Tensor:
    """Return a stable [controls, horizon] analysis operator for a fixed basis."""

    basis_f = basis.float()
    ridge = max(float(ridge), 0.0)
    if ridge == 0.0:
        return torch.linalg.pinv(basis_f).to(dtype=basis.dtype)
    gram = basis_f.transpose(0, 1) @ basis_f
    eye = torch.eye(int(gram.shape[0]), device=basis.device, dtype=torch.float32)
    return torch.linalg.solve(gram + ridge * eye, basis_f.transpose(0, 1)).to(dtype=basis.dtype)


def _action_coeff_basis_and_analysis(
    horizon: int,
    controls: int,
    *,
    name: str,
    degree: int = 2,
    ridge: float = 0.0,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor]:
    name = str(name).lower()
    if name == "dct":
        basis = _dct_action_basis(horizon, controls, device=device)
        return basis, basis.transpose(0, 1).contiguous()
    if name in {"bspline", "b-spline", "spline"}:
        basis = _bspline_action_basis(horizon, controls, degree=degree, device=device)
        return basis, _ridge_basis_analysis(basis, ridge=ridge).contiguous()
    raise ValueError("latent_cvae_arm_coeff_basis currently supports 'dct' or 'bspline'")


@dataclass(frozen=True)
class V363PolicyConfig(V362PolicyConfig):
    """V36.3 policy config.

    The extra fields control latent coupling only; the output action space and
    deployment action contract remain the V36.2 14-D physical flow decoded to
    native 7-D Alicia-D actions.
    """

    transition_fusion_dropout: float = 0.05
    transition_event_dropout: float = 0.10

    def validate(self) -> None:
        super().validate()
        if not 0 <= self.transition_fusion_dropout < 1:
            raise ValueError("transition_fusion_dropout must be in [0,1)")
        if not 0 <= self.transition_event_dropout < 1:
            raise ValueError("transition_event_dropout must be in [0,1)")


class TransitionLatentFusion(nn.Module):
    """Fuse transition/event evidence back into horizon action tokens.

    This is intentionally a zero-initialized residual.  At initialization the
    module is an identity map, so a V36.2 checkpoint can be warm-started without
    changing behavior.  During training, event supervision and final-action
    losses shape the same fused latent used by the physical velocity emitter.
    """

    def __init__(self, config: V363PolicyConfig) -> None:
        super().__init__()
        h = config.hidden_size
        self.config = config
        self.action_norm = nn.LayerNorm(h)
        self.event_norm = nn.LayerNorm(h)
        self.event_cross = nn.MultiheadAttention(h, config.num_heads, batch_first=True, dropout=config.dropout)
        self.transition_ffn = nn.Sequential(
            nn.LayerNorm(h),
            nn.Linear(h, h * 2),
            nn.SiLU(),
            nn.Dropout(config.transition_fusion_dropout),
            nn.Linear(h * 2, h),
        )
        self.residual = nn.Linear(h, h)
        self.gate = nn.Linear(h, h)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, action_tokens: Tensor, event_tokens: Tensor) -> tuple[Tensor, Tensor]:
        q = self.action_norm(action_tokens)
        event = self.event_norm(event_tokens)
        if self.training and self.config.transition_event_dropout > 0:
            # Drop whole event-context batches, not individual labels.  This
            # prevents the transition subspace from becoming a pure event-token
            # shortcut while preserving deterministic inference.
            keep = (torch.rand(action_tokens.shape[0], 1, 1, device=action_tokens.device) >= self.config.transition_event_dropout)
            event = event * keep.to(dtype=event.dtype)
        event_context, _ = self.event_cross(q, event, event, need_weights=False)
        transition_latent = self.transition_ffn(q + event_context)
        fused = action_tokens + torch.sigmoid(self.gate(q)) * self.residual(transition_latent)
        return fused, transition_latent


class TransitionAwarePhysicalVelocityHead(nn.Module):
    """V36.2 typed velocity head with in-head gripper latent modulation.

    The arm channels are emitted exactly from the normalized action tokens.  The
    gripper channels are emitted from the same tokens after a zero-initialized
    transition-latent residual.  This is not a separate gripper command head:
    the unique action output is still the 14-D physical velocity tensor.
    """

    def __init__(self, config: V363PolicyConfig) -> None:
        super().__init__()
        h = config.hidden_size
        ad = config.arm_dim
        self.config = config
        self.arm_coeff_output = bool(int(getattr(config, "latent_cvae_arm_coeff_output", 0)))
        self.arm_coeff_points = max(1, min(int(getattr(config, "latent_cvae_arm_coeff_points", 8)), int(config.action_horizon)))
        self.arm_coeff_basis_name = str(getattr(config, "latent_cvae_arm_coeff_basis", "dct")).lower()
        self.arm_coeff_degree = max(0, int(getattr(config, "latent_cvae_arm_coeff_degree", 2)))
        self.arm_coeff_ridge = max(0.0, float(getattr(config, "latent_cvae_arm_coeff_ridge", getattr(config, "latent_cvae_trajectory_ridge", 1e-2))))
        if self.arm_coeff_output:
            basis, analysis = _action_coeff_basis_and_analysis(
                int(config.action_horizon),
                self.arm_coeff_points,
                name=self.arm_coeff_basis_name,
                degree=self.arm_coeff_degree,
                ridge=self.arm_coeff_ridge,
            )
            self.register_buffer("arm_coeff_basis", basis, persistent=False)
            self.register_buffer("arm_coeff_analysis", analysis, persistent=False)
        else:
            self.register_buffer("arm_coeff_basis", torch.empty(0), persistent=False)
            self.register_buffer("arm_coeff_analysis", torch.empty(0), persistent=False)
        self.norm = nn.LayerNorm(h)
        self.transition_norm = nn.LayerNorm(h)
        self.gripper_delta = nn.Linear(h, h)
        self.gripper_gate = nn.Linear(h, h)
        nn.init.zeros_(self.gripper_delta.weight)
        nn.init.zeros_(self.gripper_delta.bias)
        nn.init.zeros_(self.gripper_gate.weight)
        nn.init.zeros_(self.gripper_gate.bias)
        self.arm_abs = nn.Linear(h, ad)
        self.arm_delta = nn.Linear(h, ad)
        self.grip_value = nn.Linear(h, 1)
        self.grip_delta = nn.Linear(h, 1)

    def _arm_coeff_basis_analysis_for(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        horizon = int(tokens.shape[1])
        if (
            self.arm_coeff_basis.ndim == 2
            and int(self.arm_coeff_basis.shape[0]) == horizon
            and int(self.arm_coeff_basis.shape[1]) == int(self.arm_coeff_points)
            and self.arm_coeff_analysis.ndim == 2
            and int(self.arm_coeff_analysis.shape[0]) == int(self.arm_coeff_points)
            and int(self.arm_coeff_analysis.shape[1]) == horizon
        ):
            return (
                self.arm_coeff_basis.to(device=tokens.device, dtype=tokens.dtype),
                self.arm_coeff_analysis.to(device=tokens.device, dtype=tokens.dtype),
            )
        controls = min(int(self.arm_coeff_points), horizon)
        basis, analysis = _action_coeff_basis_and_analysis(
            horizon,
            controls,
            name=self.arm_coeff_basis_name,
            degree=self.arm_coeff_degree,
            ridge=self.arm_coeff_ridge,
            device=tokens.device,
        )
        return basis.to(dtype=tokens.dtype), analysis.to(dtype=tokens.dtype)

    def _emit_arm(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if not self.arm_coeff_output:
            return self.arm_abs(x), self.arm_delta(x)
        basis, analysis = self._arm_coeff_basis_analysis_for(x)
        coeff_tokens = torch.einsum("ct,bth->bch", analysis, x)
        arm_abs_coeff = self.arm_abs(coeff_tokens)
        arm_delta_coeff = self.arm_delta(coeff_tokens)
        arm_abs = torch.einsum("tc,bca->bta", basis, arm_abs_coeff)
        arm_delta = torch.einsum("tc,bca->bta", basis, arm_delta_coeff)
        return arm_abs, arm_delta

    def forward(self, tokens: Tensor, transition_latent: Tensor | None = None) -> Tensor:
        x = self.norm(tokens)
        grip_x = x
        if transition_latent is not None:
            z = self.transition_norm(transition_latent)
            grip_x = grip_x + torch.sigmoid(self.gripper_gate(z)) * self.gripper_delta(z)
        arm_abs, arm_delta = self._emit_arm(x)
        return torch.cat([arm_abs, arm_delta, self.grip_value(grip_x), self.grip_delta(grip_x)], dim=-1)


class PolicyLatentDiTPlannerV363(nn.Module):
    def __init__(self, config: V363PolicyConfig, world_hidden: int, world_tokens: int) -> None:
        super().__init__()
        config.validate()
        self.config = config
        h = config.hidden_size
        self.world_proj = nn.Identity() if world_hidden == h else nn.Linear(world_hidden, h)
        self.state_proj = nn.Linear(config.state_dim, h)
        self.executed_proj = nn.Linear(config.action_dim, h)
        self.noisy_physical_lift = PhysicalActionTokenLift(config)
        self.proposal_proj = nn.Identity()
        self.role = HorizonRoleEmbedding(config)
        self.task_token = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        self.world_type = nn.Parameter(torch.randn(1, world_tokens, h) * 0.02)
        self.state_type = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        self.executed_type = nn.Parameter(torch.randn(1, config.executed_history_length, h) * 0.02)
        self.proposal_type = nn.Parameter(torch.randn(1, config.action_horizon, h) * 0.02)
        self.horizon_query = nn.Parameter(torch.randn(1, config.action_horizon, h) * 0.02)
        self.event_query = nn.Parameter(torch.randn(1, config.event_tokens, h) * 0.02)
        self.event_type = nn.Parameter(torch.randn(1, config.event_tokens, h) * 0.02)
        self.register_buffer("horizon_position", sinusoidal_positions(range(1, config.action_horizon + 1), h)[None], persistent=True)
        self.time = TimeEmbedding(h)
        self.blocks = nn.ModuleList([DiTPlannerBlock(config) for _ in range(config.depth)])
        self.transition_fusion = TransitionLatentFusion(config)
        self.event_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 3))
        self.motion_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 1))

    def _tokens(
        self,
        noisy_physical: Tensor,
        world: Tensor,
        state: Tensor,
        executed_history: Tensor,
        proposal_tokens: Tensor,
        proposal_keep: Tensor,
    ) -> tuple[Tensor, slice, slice]:
        batch = noisy_physical.shape[0]
        hpos = self.horizon_position.to(device=noisy_physical.device, dtype=noisy_physical.dtype)
        task = self.task_token.expand(batch, -1, -1)
        world_tokens = self.world_proj(world) + self.world_type
        state_token = self.state_proj(state)[:, None] + self.state_type
        executed = self.executed_proj(executed_history) + self.executed_type
        proposal = self.proposal_proj(proposal_tokens) * proposal_keep[:, None, None] + self.proposal_type
        role = self.role(batch, device=noisy_physical.device, dtype=noisy_physical.dtype)
        horizon = self.horizon_query.expand(batch, -1, -1) + hpos + role + self.noisy_physical_lift(noisy_physical)
        event = self.event_query.expand(batch, -1, -1) + self.event_type
        prefix_len = 1 + world_tokens.shape[1] + 1 + executed.shape[1] + proposal.shape[1]
        action_slice = slice(prefix_len, prefix_len + self.config.action_horizon)
        event_start = prefix_len + self.config.action_horizon
        event_slice = slice(event_start, event_start + self.config.event_tokens)
        tokens = torch.cat([task, world_tokens, state_token, executed, proposal, horizon, event], dim=1)
        return tokens, action_slice, event_slice

    def forward(
        self,
        noisy_physical: Tensor,
        time: Tensor,
        world: Tensor,
        state: Tensor,
        executed_history: Tensor,
        proposal_tokens: Tensor,
        proposal_keep: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if proposal_keep is None:
            proposal_keep = torch.ones(noisy_physical.shape[0], device=noisy_physical.device, dtype=noisy_physical.dtype)
        tokens, action_slice, event_slice = self._tokens(noisy_physical, world, state, executed_history, proposal_tokens, proposal_keep)
        time_emb = self.time(time.to(dtype=tokens.dtype))
        for block in self.blocks:
            tokens = block(tokens, time_emb)
        action_tokens = tokens[:, action_slice, :]
        event_tokens = tokens[:, event_slice, :]
        fused_action_tokens, transition_latent = self.transition_fusion(action_tokens, event_tokens)
        # Make the memory consumed by the decoder share the same fused latent.
        tokens = tokens.clone()
        tokens[:, action_slice, :] = fused_action_tokens
        return {
            "planner_tokens": tokens,
            "planner_action_tokens": fused_action_tokens,
            "planner_event_tokens": event_tokens,
            "transition_latent": transition_latent,
            "event_logits": self.event_head(fused_action_tokens),
            "motion_logits": self.motion_head(fused_action_tokens).squeeze(-1),
        }


class PlannerConditionedPhysicalActionExpertV363(nn.Module):
    """Planner-conditioned decoder that emits typed physical-action velocity."""

    def __init__(self, config: V363PolicyConfig, world_hidden: int, world_tokens: int) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.world_proj = nn.Identity() if world_hidden == h else nn.Linear(world_hidden, h)
        self.state_proj = nn.Linear(config.state_dim, h)
        self.executed_proj = nn.Linear(config.action_dim, h)
        self.noisy_physical_lift = PhysicalActionTokenLift(config)
        self.planner_action_proj = nn.Linear(h, h)
        self.transition_proj = nn.Linear(h, h)
        self.proposal_proj = nn.Identity()
        self.role = HorizonRoleEmbedding(config)
        self.task_token = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        self.world_type = nn.Parameter(torch.randn(1, world_tokens, h) * 0.02)
        self.state_type = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        self.executed_type = nn.Parameter(torch.randn(1, config.executed_history_length, h) * 0.02)
        self.proposal_type = nn.Parameter(torch.randn(1, config.action_horizon, h) * 0.02)
        self.planner_memory_type = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        self.register_buffer("action_position", sinusoidal_positions(range(1, config.action_horizon + 1), h)[None], persistent=True)
        self.time = TimeEmbedding(h)
        self.blocks = nn.ModuleList([ActionExpertBlock(config) for _ in range(config.action_decoder_depth)])
        self.out = TransitionAwarePhysicalVelocityHead(config)

    def memory(
        self,
        world: Tensor,
        state: Tensor,
        executed_history: Tensor,
        proposal_tokens: Tensor,
        proposal_keep: Tensor,
        planner_tokens: Tensor,
    ) -> Tensor:
        world_tokens = self.world_proj(world) + self.world_type
        state_token = self.state_proj(state)[:, None] + self.state_type
        executed = self.executed_proj(executed_history) + self.executed_type
        proposal = self.proposal_proj(proposal_tokens) * proposal_keep[:, None, None] + self.proposal_type
        task = self.task_token.expand(world.shape[0], -1, -1)
        planner = planner_tokens + self.planner_memory_type
        return torch.cat([task, world_tokens, state_token, executed, proposal, planner], dim=1)

    def forward(
        self,
        noisy_physical: Tensor,
        time: Tensor,
        world: Tensor,
        state: Tensor,
        executed_history: Tensor,
        proposal_tokens: Tensor,
        proposal_keep: Tensor,
        planner_tokens: Tensor,
        planner_action_tokens: Tensor,
        transition_latent: Tensor,
    ) -> Tensor:
        batch = noisy_physical.shape[0]
        position = self.action_position.to(device=noisy_physical.device, dtype=noisy_physical.dtype)
        role = self.role(batch, device=noisy_physical.device, dtype=noisy_physical.dtype)
        x = self.noisy_physical_lift(noisy_physical) + self.planner_action_proj(planner_action_tokens) + role
        transition = self.transition_proj(transition_latent.to(dtype=x.dtype))
        memory = self.memory(world, state, executed_history, proposal_tokens, proposal_keep, planner_tokens)
        t = self.time(time.to(dtype=x.dtype))
        for block in self.blocks:
            x = block(x, memory, t, position)
        return self.out(x, transition)


class V363PolicySystem(nn.Module):
    def __init__(
        self,
        world_config: V35WorldConfig,
        policy_config: V363PolicyConfig,
        world_encoder: WorldEvidenceEncoder,
    ) -> None:
        super().__init__()
        self.world_config = world_config
        self.policy_config = policy_config
        self.world_encoder = world_encoder
        self.world_encoder.requires_grad_(False)
        self.world_encoder.eval()
        self.codec = PhysicalActionCodec(policy_config)
        self.proposal = RejectableHistoryProposal(policy_config)
        self.planner = PolicyLatentDiTPlannerV363(policy_config, world_hidden=world_config.hidden_size, world_tokens=world_config.world_tokens)
        self.decoder = PlannerConditionedPhysicalActionExpertV363(policy_config, world_hidden=world_config.hidden_size, world_tokens=world_config.world_tokens)

    def train(self, mode: bool = True):
        super().train(mode)
        self.world_encoder.eval()
        return self

    @torch.no_grad()
    def encode_world(self, visual: Tensor, state_history: Tensor, executed_history: Tensor) -> Tensor:
        return self.world_encoder(visual.float(), state_history.float(), executed_history.float())

    @staticmethod
    def ablate_world_tokens(world: Tensor, mode: str = "normal", *, seed: int | None = None) -> Tensor:
        """Apply evaluation-only world-token interventions.

        These interventions are intentionally placed after the frozen world
        encoder so that direct policy inputs (state/history/proposal/noise) stay
        unchanged.  They measure the marginal reliance of the trained policy on
        the world-token stream rather than changing dataset or sampler inputs.
        """
        normalized = mode.replace("-", "_").lower()
        if normalized in {"", "none", "normal"}:
            return world
        if normalized == "zero":
            return torch.zeros_like(world)
        if normalized == "batch_mean":
            return world.mean(dim=0, keepdim=True).expand_as(world)
        generator = None
        if seed is not None:
            generator = torch.Generator(device=world.device)
            generator.manual_seed(int(seed))
        if normalized == "shuffle":
            if world.shape[0] <= 1:
                return world.clone()
            perm = torch.randperm(world.shape[0], device=world.device, generator=generator)
            return world[perm]
        if normalized == "noise":
            stats = world.detach().float()
            scale = stats.std().clamp_min(1e-6).to(device=world.device, dtype=world.dtype)
            mean = stats.mean().to(device=world.device, dtype=world.dtype)
            return torch.randn(world.shape, device=world.device, dtype=world.dtype, generator=generator) * scale + mean
        raise ValueError(f"unknown world ablation mode: {mode}")

    def _policy_forward(
        self,
        noisy_physical: Tensor,
        time: Tensor,
        world: Tensor,
        state: Tensor,
        executed_history: Tensor,
        proposal_tokens: Tensor,
        proposal_keep: Tensor,
    ) -> dict[str, Tensor]:
        planner = self.planner(noisy_physical, time, world, state, executed_history, proposal_tokens, proposal_keep)
        pred_physical_velocity = self.decoder(
            noisy_physical,
            time,
            world,
            state,
            executed_history,
            proposal_tokens,
            proposal_keep,
            planner["planner_tokens"],
            planner["planner_action_tokens"],
            planner["transition_latent"],
        )
        planner["pred_physical_velocity"] = pred_physical_velocity
        return planner

    def flow_training_forward(
        self,
        visual: Tensor,
        state_history: Tensor,
        executed_history: Tensor,
        state: Tensor,
        target_action: Tensor,
        *,
        proposal_dropout: float | None = None,
    ) -> dict[str, Tensor]:
        world = self.encode_world(visual, state_history, executed_history)
        proposal = self.proposal(executed_history)
        target_physical = self.codec.encode(target_action, state)
        noise = torch.randn_like(target_physical)
        t = torch.rand(target_physical.shape[0], device=target_physical.device, dtype=target_physical.dtype)
        noisy_physical = (1 - t[:, None, None]) * target_physical + t[:, None, None] * noise
        target_physical_velocity = noise - target_physical
        drop = self.policy_config.proposal_dropout if proposal_dropout is None else float(proposal_dropout)
        keep = (torch.rand(target_physical.shape[0], device=target_physical.device) >= drop).to(target_physical.dtype)
        policy = self._policy_forward(noisy_physical, t, world, state, executed_history, proposal["tokens"].detach(), keep)
        clean_physical_estimate = noisy_physical - t[:, None, None] * policy["pred_physical_velocity"]
        decoded_action = self.codec.decode(clean_physical_estimate, state)
        return {
            "pred_physical_velocity": policy["pred_physical_velocity"],
            "target_physical_velocity": target_physical_velocity,
            "target_physical": target_physical,
            "clean_physical_estimate": clean_physical_estimate,
            "proposal_action": proposal["action"],
            "world": world,
            "time": t,
            "noisy_physical_action": noisy_physical,
            "pred_action_estimate": decoded_action,
            "event_logits": policy["event_logits"],
            "motion_logits": policy["motion_logits"],
            "transition_latent": policy["transition_latent"],
        }

    @torch.no_grad()
    def sample(
        self,
        visual: Tensor,
        state_history: Tensor,
        executed_history: Tensor,
        state: Tensor,
        *,
        steps: int | None = None,
        noise: Tensor | None = None,
        use_proposal: bool = True,
        return_event_logits: bool = False,
        world_ablation: str = "normal",
        world_ablation_seed: int | None = None,
    ) -> Tensor | dict[str, Tensor]:
        world = self.encode_world(visual, state_history, executed_history)
        world = self.ablate_world_tokens(world, world_ablation, seed=world_ablation_seed)
        proposal = self.proposal(executed_history)
        steps = int(steps or self.policy_config.inference_steps)
        if noise is None:
            x = torch.randn(
                visual.shape[0],
                self.policy_config.action_horizon,
                self.policy_config.physical_action_dim,
                device=visual.device,
                dtype=visual.dtype,
            )
        else:
            x = noise.clone()
            if x.shape[-1] == self.policy_config.action_dim:
                x = self.codec.encode(x.to(device=visual.device, dtype=visual.dtype), state.to(device=visual.device, dtype=visual.dtype))
            elif x.shape[-1] != self.policy_config.physical_action_dim:
                raise ValueError("noise must have last dim action_dim or physical_action_dim")
        keep = torch.full((visual.shape[0],), 1.0 if use_proposal else 0.0, device=visual.device, dtype=visual.dtype)
        last: dict[str, Tensor] | None = None
        for index in range(steps, 0, -1):
            t = torch.full((visual.shape[0],), float(index) / float(steps), device=visual.device, dtype=visual.dtype)
            last = self._policy_forward(x, t, world, state, executed_history, proposal["tokens"], keep)
            x = x - last["pred_physical_velocity"] / float(steps)
        action = self.codec.decode(x, state)
        if return_event_logits:
            zero_t = torch.zeros((visual.shape[0],), device=visual.device, dtype=visual.dtype)
            event = self._policy_forward(x, zero_t, world, state, executed_history, proposal["tokens"], keep)
            return {"action": action, "physical_action": x, "event_logits": event["event_logits"], "motion_logits": event["motion_logits"]}
        return action

    def parameter_report(self) -> dict[str, int]:
        report = {
            "frozen_world_encoder": sum(p.numel() for p in self.world_encoder.parameters()),
            "history_proposal": sum(p.numel() for p in self.proposal.parameters()),
            "physical_action_codec": sum(p.numel() for p in self.codec.parameters()),
            "latent_dit_planner": sum(p.numel() for p in self.planner.parameters()),
            "physical_action_expert_decoder": sum(p.numel() for p in self.decoder.parameters()),
        }
        report["total"] = sum(p.numel() for p in self.parameters())
        report["trainable"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return report


__all__ = [
    "V363PolicyConfig",
    "TransitionLatentFusion",
    "TransitionAwarePhysicalVelocityHead",
    "PolicyLatentDiTPlannerV363",
    "PlannerConditionedPhysicalActionExpertV363",
    "V363PolicySystem",
]
