from __future__ import annotations

"""V38.6.2 action-centered controlled-residual latent dynamics policy.

V38.6 keeps the V38.5 no-future-input contract but removes the most dangerous
remaining shortcut: a free rollout token can predict an observation-conditioned
average future.  Future DINO residuals are still targets only, but the model now
represents them as:

    pred_effect = weak_visual_base + action_centered_controlled_delta

The base branch is deliberately low-capacity and absorbs observation/phase
average future.  The delta branch is generated from a visual transition basis,
but coefficients are *centered* against a learned no-op/neutral coefficient:

    delta = basis @ (coeff(action) - coeff(neutral_context))

This prevents action-independent coefficient bias from re-entering the delta
path.  Tail action and gripper/event readouts consume the centered delta, not a
free rollout token.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .policy import RejectableHistoryProposal, TimeEmbedding
from .policy_v36_2 import (
    HorizonRoleEmbedding,
    PhysicalActionCodec,
    PhysicalActionTokenLift,
    V362PolicyConfig,
)
from .world_model import BiasFreeFFN, sinusoidal_positions


@dataclass(frozen=True)
class V38PolicyConfig(V362PolicyConfig):
    """Configuration for the latent-dynamics-bound canvas."""

    visual_token_dim: int = 768
    visual_history_length: int = 3
    num_cameras: int = 2
    patches_per_camera: int = 576
    canvas_registers: int = 12
    future_anchors: int = 4
    target_future_count: int = 12
    visual_memory_dropout: float = 0.0
    canvas_dropout: float = 0.0
    role_dropout: float = 0.10
    action_basis_tokens: int = 4
    future_grid_size: int = 4
    # Kept for checkpoint/context compatibility.  V38.5 does not use a
    # future-noisy input branch by default.
    future_flow_loss_weight: float = 0.0
    # Tail action residual binding schedule.  Early actions may be read directly
    # from action tokens; mid/tail actions increasingly must read rollout tokens.
    rollout_tail_start_step: int = 8
    rollout_tail_full_step: int = 13
    # V38.6.2 action-centered controlled residual dynamics.  ``base_effect``
    # has deliberately small capacity; ``controlled_delta`` is produced by
    # action coefficients centered against a neutral/no-op coefficient.
    controlled_delta_rank: int = 8
    base_effect_hidden: int = 128
    latent_action_tokens: int = 8
    controlled_delta_dropout: float = 0.0
    neutral_action_tokens: int = 4

    def validate(self) -> None:
        super().validate()
        if min(
            self.visual_token_dim,
            self.visual_history_length,
            self.num_cameras,
            self.patches_per_camera,
            self.canvas_registers,
            self.future_anchors,
            self.target_future_count,
            self.action_basis_tokens,
            self.future_grid_size,
        ) <= 0:
            raise ValueError("V38 dimensions must be positive")
        if self.future_anchors > self.target_future_count:
            raise ValueError("future_anchors cannot exceed target_future_count")
        if not 0 <= self.visual_memory_dropout < 1:
            raise ValueError("visual_memory_dropout must be in [0,1)")
        if not 0 <= self.canvas_dropout < 1:
            raise ValueError("canvas_dropout must be in [0,1)")
        if not 0 <= self.role_dropout < 1:
            raise ValueError("role_dropout must be in [0,1)")
        if self.rollout_tail_start_step < 1 or self.rollout_tail_full_step < self.rollout_tail_start_step:
            raise ValueError("invalid rollout tail binding schedule")
        if min(self.controlled_delta_rank, self.base_effect_hidden, self.latent_action_tokens, self.neutral_action_tokens) <= 0:
            raise ValueError("controlled residual dynamics dimensions must be positive")
        if not 0 <= self.controlled_delta_dropout < 1:
            raise ValueError("controlled_delta_dropout must be in [0,1)")

    @property
    def future_token_count(self) -> int:
        return int(self.future_anchors) * int(self.num_cameras) * int(self.future_grid_size) * int(self.future_grid_size)

    @property
    def history_length(self) -> int:
        return self.visual_history_length

    @property
    def num_future(self) -> int:
        return self.target_future_count

    @property
    def latent_dim(self) -> int:
        return self.visual_token_dim


class DenseVisualMemory(nn.Module):
    """Per-token DINO projection with factorized identity embeddings."""

    def __init__(self, config: V38PolicyConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size
        d = config.visual_token_dim
        self.proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, h))
        self.history_type = nn.Parameter(torch.randn(1, config.visual_history_length, 1, 1, h) * 0.02)
        self.camera_type = nn.Parameter(torch.randn(1, 1, config.num_cameras, 1, h) * 0.02)
        self.patch_type = nn.Parameter(torch.randn(1, 1, 1, config.patches_per_camera, h) * 0.02)
        self.out_norm = nn.LayerNorm(h)
        self.drop = nn.Dropout(config.visual_memory_dropout)

    def forward(self, visual: Tensor) -> Tensor:
        cfg = self.config
        if visual.ndim != 5:
            raise ValueError(f"visual must be [B,H,C,P,D], got {tuple(visual.shape)}")
        b, hist, cams, patches, dim = visual.shape
        expected = (cfg.visual_history_length, cfg.num_cameras, cfg.patches_per_camera, cfg.visual_token_dim)
        if (hist, cams, patches, dim) != expected:
            raise ValueError(f"V38 visual geometry mismatch: got {(hist, cams, patches, dim)}, expected {expected}")
        x = self.proj(visual)
        x = x + self.history_type.to(device=x.device, dtype=x.dtype)
        x = x + self.camera_type.to(device=x.device, dtype=x.dtype)
        x = x + self.patch_type.to(device=x.device, dtype=x.dtype)
        x = self.out_norm(x)
        return self.drop(x.reshape(b, hist * cams * patches, cfg.hidden_size))


class RolloutTargetCodec(nn.Module):
    """Build rollout init/target tokens from DINO grids.

    The target projection is frozen.  This prevents a learned target projector
    from collapsing the future residual objective into an easy private code.
    """

    def __init__(self, config: V38PolicyConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size
        d = config.visual_token_dim
        self.init_proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, h))
        self.target_proj = nn.Linear(d, h, bias=False)
        nn.init.orthogonal_(self.target_proj.weight)
        self.target_proj.weight.requires_grad_(False)

    def spatial_pool_tokens(self, tokens: Tensor) -> Tensor:
        """Pool [B,F,C,P,D] or [B,C,P,D] to [B,F,C*G*G,D]."""
        cfg = self.config
        original_ndim = tokens.ndim
        if original_ndim == 4:
            tokens = tokens[:, None]
        if tokens.ndim != 5:
            raise ValueError(f"tokens must be [B,F,C,P,D] or [B,C,P,D], got {tuple(tokens.shape)}")
        b, f, c, p, d = tokens.shape
        side = int(round(float(p) ** 0.5))
        if side * side == p:
            x = tokens.reshape(b * f * c, side, side, d).permute(0, 3, 1, 2).float()
            x = F.adaptive_avg_pool2d(x, (cfg.future_grid_size, cfg.future_grid_size))
            x = x.permute(0, 2, 3, 1).reshape(b, f, c * cfg.future_grid_size * cfg.future_grid_size, d)
        else:
            g2 = cfg.future_grid_size * cfg.future_grid_size
            idx = torch.linspace(0, p, steps=g2 + 1, device=tokens.device).long()
            pooled = []
            for i in range(g2):
                lo, hi = int(idx[i]), max(int(idx[i + 1]), int(idx[i]) + 1)
                pooled.append(tokens[..., lo:hi, :].float().mean(dim=-2))
            x = torch.stack(pooled, dim=3).reshape(b, f, c * g2, d)
        if original_ndim == 4:
            return x[:, 0]
        return x

    def current_grid(self, visual: Tensor) -> Tensor:
        if visual.ndim != 5:
            raise ValueError(f"visual must be [B,H,C,P,D], got {tuple(visual.shape)}")
        return self.spatial_pool_tokens(visual[:, -1])  # [B,C*G*G,D]

    def rollout_init(self, visual: Tensor) -> Tensor:
        cfg = self.config
        current = self.current_grid(visual)
        current_h = self.init_proj(current.to(device=next(self.parameters()).device, dtype=next(self.parameters()).dtype))
        return current_h[:, None].expand(-1, cfg.future_anchors, -1, -1).reshape(
            visual.shape[0], cfg.future_token_count, cfg.hidden_size
        )

    @torch.no_grad()
    def target_effect(self, visual: Tensor, target_visual: Tensor) -> Tensor:
        cfg = self.config
        if target_visual.ndim != 6:
            raise ValueError(f"target_visual must be [B,F,H,C,P,D], got {tuple(target_visual.shape)}")
        future = target_visual[:, : cfg.future_anchors, -1]
        current = visual[:, -1][:, None].expand(-1, cfg.future_anchors, -1, -1, -1)
        residual = future.float() - current.float()
        pooled = self.spatial_pool_tokens(residual)  # [B,K,C*G*G,D]
        target = self.target_proj(pooled.to(device=next(self.parameters()).device, dtype=next(self.parameters()).dtype))
        return target.reshape(visual.shape[0], cfg.future_token_count, cfg.hidden_size).detach()


class UnifiedCanvasSeed(nn.Module):
    """Build the initial canvas with rollout tokens initialized from current vision."""

    ROLE_TASK = 0
    ROLE_STATE = 1
    ROLE_STATE_HISTORY = 2
    ROLE_EXECUTED = 3
    ROLE_PROPOSAL = 4
    ROLE_NOISY_ACTION = 5
    ROLE_ROLLOUT = 6
    ROLE_REGISTER = 7

    def __init__(self, config: V38PolicyConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.state_proj = nn.Linear(config.state_dim, h)
        self.state_history_proj = nn.Linear(config.state_dim, h)
        self.executed_proj = nn.Linear(config.action_dim, h)
        self.proposal_proj = nn.Identity()
        self.noisy_physical_lift = PhysicalActionTokenLift(config)
        self.horizon_role = HorizonRoleEmbedding(config)
        self.action_basis_embed = nn.Parameter(torch.randn(1, 1, config.action_basis_tokens, h) * 0.02)
        self.role_embed = nn.Parameter(torch.randn(8, h) * 0.02)
        self.role_drop = nn.Dropout(config.role_dropout)
        self.task_token = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        self.rollout_anchor_type = nn.Parameter(torch.randn(1, config.future_anchors, 1, h) * 0.02)
        self.rollout_grid_type = nn.Parameter(torch.randn(1, 1, config.num_cameras * config.future_grid_size * config.future_grid_size, h) * 0.02)
        self.registers = nn.Parameter(torch.randn(1, config.canvas_registers, h) * 0.02)
        self.proposal_type = nn.Parameter(torch.randn(1, config.action_horizon, h) * 0.02)
        self.executed_type = nn.Parameter(torch.randn(1, config.executed_history_length, h) * 0.02)
        self.state_history_type = nn.Parameter(torch.randn(1, config.visual_history_length, h) * 0.02)
        self.drop = nn.Dropout(config.canvas_dropout)
        self.register_buffer("horizon_position", sinusoidal_positions(range(1, config.action_horizon + 1), h)[None], persistent=True)

    def forward(
        self,
        *,
        noisy_physical: Tensor,
        state: Tensor,
        state_history: Tensor,
        executed_history: Tensor,
        proposal_tokens: Tensor,
        proposal_keep: Tensor,
        rollout_init: Tensor,
    ) -> tuple[Tensor, dict[str, slice]]:
        cfg = self.config
        b = noisy_physical.shape[0]
        device = noisy_physical.device
        dtype = noisy_physical.dtype
        role = self.role_drop(self.role_embed.to(device=device, dtype=dtype))
        task = self.task_token.expand(b, -1, -1).to(device=device, dtype=dtype) + role[self.ROLE_TASK]
        state_tok = self.state_proj(state)[:, None] + role[self.ROLE_STATE]
        hist = self.state_history_proj(state_history) + self.state_history_type.to(device=device, dtype=dtype) + role[self.ROLE_STATE_HISTORY]
        executed = self.executed_proj(executed_history) + self.executed_type.to(device=device, dtype=dtype) + role[self.ROLE_EXECUTED]
        proposal = self.proposal_proj(proposal_tokens) * proposal_keep[:, None, None] + self.proposal_type.to(device=device, dtype=dtype) + role[self.ROLE_PROPOSAL]
        noisy_base = (
            self.noisy_physical_lift(noisy_physical)
            + self.horizon_position.to(device=device, dtype=dtype)
            + self.horizon_role(b, device=device, dtype=dtype)
            + role[self.ROLE_NOISY_ACTION]
        )
        noisy = (noisy_base[:, :, None, :] + self.action_basis_embed.to(device=device, dtype=dtype)).reshape(
            b, cfg.action_horizon * cfg.action_basis_tokens, cfg.hidden_size
        )
        if rollout_init.shape != (b, cfg.future_token_count, cfg.hidden_size):
            raise ValueError(f"rollout_init must be [B,{cfg.future_token_count},{cfg.hidden_size}], got {tuple(rollout_init.shape)}")
        rollout = rollout_init.to(device=device, dtype=dtype).reshape(
            b, cfg.future_anchors, cfg.num_cameras * cfg.future_grid_size * cfg.future_grid_size, cfg.hidden_size
        )
        rollout = rollout + self.rollout_anchor_type.to(device=device, dtype=dtype) + self.rollout_grid_type.to(device=device, dtype=dtype)
        rollout = rollout.reshape(b, cfg.future_token_count, cfg.hidden_size) + role[self.ROLE_ROLLOUT]
        registers = self.registers.expand(b, -1, -1).to(device=device, dtype=dtype) + role[self.ROLE_REGISTER]
        parts = [task, state_tok, hist, executed, proposal, noisy, rollout, registers]
        starts = []
        offset = 0
        for part in parts:
            starts.append(offset)
            offset += part.shape[1]
        slices = {
            "task": slice(starts[0], starts[0] + 1),
            "state": slice(starts[1], starts[1] + 1),
            "state_history": slice(starts[2], starts[2] + cfg.visual_history_length),
            "executed": slice(starts[3], starts[3] + cfg.executed_history_length),
            "proposal": slice(starts[4], starts[4] + cfg.action_horizon),
            "trajectory": slice(starts[5], starts[5] + cfg.action_horizon * cfg.action_basis_tokens),
            "rollout": slice(starts[6], starts[6] + cfg.future_token_count),
            "registers": slice(starts[7], starts[7] + cfg.canvas_registers),
        }
        return self.drop(torch.cat(parts, dim=1)), slices


class TemporalDynamicsBoundDiTBlock(nn.Module):
    """Canvas block with explicit action-to-rollout transition sublayer."""

    def __init__(self, config: V38PolicyConfig) -> None:
        super().__init__()
        h = config.hidden_size
        self.n1 = nn.LayerNorm(h, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(h, config.num_heads, batch_first=True, dropout=config.dropout)
        self.n2 = nn.LayerNorm(h, elementwise_affine=False)
        self.mem_norm = nn.LayerNorm(h)
        self.cross = nn.MultiheadAttention(h, config.num_heads, batch_first=True, dropout=config.dropout)
        self.n_dyn_q = nn.LayerNorm(h, elementwise_affine=False)
        self.n_dyn_kv = nn.LayerNorm(h)
        self.rollout_cross = nn.MultiheadAttention(h, config.num_heads, batch_first=True, dropout=config.dropout)
        self.n3 = nn.LayerNorm(h, elementwise_affine=False)
        self.ffn = BiasFreeFFN(h, config.ffn_expansion)
        self.drop = nn.Dropout(config.dropout)
        self.mod = nn.Linear(h, 12 * h)
        nn.init.normal_(self.mod.weight, mean=0.0, std=3e-3)
        nn.init.zeros_(self.mod.bias)
        with torch.no_grad():
            for idx in (2, 5, 8, 11):
                self.mod.bias[idx * h : (idx + 1) * h].fill_(-2.0)

    @staticmethod
    def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        return x * (1 + scale[:, None]) + shift[:, None]

    def forward(self, canvas: Tensor, visual_memory: Tensor, mod_embed: Tensor, slices: dict[str, slice]) -> tuple[Tensor, dict[str, Tensor]]:
        sa_s, sa_c, sa_g, ca_s, ca_c, ca_g, dy_s, dy_c, dy_g, ff_s, ff_c, ff_g = self.mod(mod_embed).chunk(12, dim=-1)
        value = self.n1(canvas)
        qk = self.modulate(value, sa_s, sa_c)
        update, _ = self.self_attn(qk, qk, value, need_weights=False)
        g_sa = torch.sigmoid(sa_g)
        canvas = canvas + g_sa[:, None] * self.drop(update)

        query = self.modulate(self.n2(canvas), ca_s, ca_c)
        mem = self.mem_norm(visual_memory)
        update, _ = self.cross(query, mem, mem, need_weights=False)
        g_ca = torch.sigmoid(ca_g)
        canvas = canvas + g_ca[:, None] * self.drop(update)

        rollout = canvas[:, slices["rollout"]]
        kv_parts = [canvas[:, slices[name]] for name in ("state", "state_history", "executed", "proposal", "trajectory")]
        kv = self.n_dyn_kv(torch.cat(kv_parts, dim=1))
        q = self.modulate(self.n_dyn_q(rollout), dy_s, dy_c)
        update, _ = self.rollout_cross(q, kv, kv, need_weights=False)
        g_dyn = torch.sigmoid(dy_g)
        canvas = canvas.clone()
        canvas[:, slices["rollout"]] = rollout + g_dyn[:, None] * self.drop(update)

        update = self.ffn(self.modulate(self.n3(canvas), ff_s, ff_c))
        g_ff = torch.sigmoid(ff_g)
        canvas = canvas + g_ff[:, None] * self.drop(update)
        return canvas, {
            "gate_self": g_sa.mean(),
            "gate_visual": g_ca.mean(),
            "gate_rollout": g_dyn.mean(),
            "gate_ffn": g_ff.mean(),
        }


class CanvasPhysicalVelocityHead(nn.Module):
    def __init__(self, config: V38PolicyConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.net = nn.Sequential(
            nn.LayerNorm(config.action_basis_tokens * h),
            nn.Linear(config.action_basis_tokens * h, 2 * h),
            nn.SiLU(),
            nn.Linear(2 * h, config.physical_action_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def pooled(self, trajectory_tokens: Tensor) -> Tensor:
        cfg = self.config
        b = trajectory_tokens.shape[0]
        return trajectory_tokens.reshape(b, cfg.action_horizon, cfg.action_basis_tokens, cfg.hidden_size).mean(dim=2)

    def forward(self, trajectory_tokens: Tensor) -> Tensor:
        cfg = self.config
        b = trajectory_tokens.shape[0]
        grouped = trajectory_tokens.reshape(b, cfg.action_horizon, cfg.action_basis_tokens, cfg.hidden_size)
        return self.net(grouped.reshape(b, cfg.action_horizon, cfg.action_basis_tokens * cfg.hidden_size))


class RolloutActionResidualHead(nn.Module):
    """Tail action residual that must read rollout latent tokens."""

    def __init__(self, config: V38PolicyConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.query_norm = nn.LayerNorm(h)
        self.rollout_norm = nn.LayerNorm(h)
        self.cross = nn.MultiheadAttention(h, config.num_heads, batch_first=True, dropout=config.dropout)
        self.net = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 2 * h), nn.SiLU(), nn.Linear(2 * h, config.physical_action_dim))
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)
        alpha = torch.zeros(config.action_horizon, dtype=torch.float32)
        start = max(int(config.rollout_tail_start_step), 1)
        full = max(int(config.rollout_tail_full_step), start)
        for i in range(config.action_horizon):
            step = i + 1
            if step < start:
                value = 0.0
            elif step >= full:
                value = 1.0
            else:
                value = float(step - start + 1) / float(max(full - start + 1, 1))
            alpha[i] = value
        self.register_buffer("alpha", alpha[None, :, None], persistent=True)

    def forward(self, trajectory_pooled: Tensor, rollout_tokens: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(trajectory_pooled)
        kv = self.rollout_norm(rollout_tokens)
        update, _ = self.cross(q, kv, kv, need_weights=False)
        residual = self.net(update)
        alpha = self.alpha.to(device=residual.device, dtype=residual.dtype)
        return residual * alpha, alpha



class ControlledResidualLatentDynamics(nn.Module):
    """Predict future DINO residual as weak visual base + action-centered delta.

    ``base_effect`` is visual-only and intentionally low-capacity.  ``basis`` is
    a set of local transition directions derived from the current visual rollout
    base.  ``coeff(action)`` is produced by action/state/proposal tokens, but the
    actual delta uses ``coeff(action) - coeff(neutral_context)``.  This centering
    removes action-independent coefficient bias from the controlled path and
    forces the delta to represent an intervention relative to a learned no-op
    context instead of another average-future predictor.
    """

    def __init__(self, config: V38PolicyConfig) -> None:
        super().__init__()
        self.config = config
        h = int(config.hidden_size)
        r = int(config.controlled_delta_rank)
        base_hidden = int(config.base_effect_hidden)
        self.base_head = nn.Sequential(
            nn.LayerNorm(h),
            nn.Linear(h, base_hidden),
            nn.SiLU(),
            nn.Linear(base_hidden, h),
        )
        self.basis_head = nn.Sequential(
            nn.LayerNorm(h),
            nn.Linear(h, 2 * h),
            nn.SiLU(),
            nn.Linear(2 * h, r * h),
        )
        self.action_queries = nn.Parameter(torch.randn(1, int(config.latent_action_tokens), h) * 0.02)
        self.neutral_queries = nn.Parameter(torch.randn(1, int(config.neutral_action_tokens), h) * 0.02)
        self.neutral_bias = nn.Parameter(torch.zeros(1, 1, h))
        self.action_kv_norm = nn.LayerNorm(h)
        self.action_cross = nn.MultiheadAttention(h, config.num_heads, batch_first=True, dropout=config.dropout)
        self.rollout_query_norm = nn.LayerNorm(h)
        self.action_latent_norm = nn.LayerNorm(h)
        self.coeff_cross = nn.MultiheadAttention(h, config.num_heads, batch_first=True, dropout=config.dropout)
        # Direct trajectory summary keeps the coefficient path from being a
        # purely second-order product of tiny random attention features.  It is
        # still a network path, not a rule: coefficients are generated from the
        # actual action/trajectory tokens and are shared across real/hold/shuffle
        # counterfactual forwards.
        self.direct_action_norm = nn.LayerNorm(h)
        self.direct_action_mlp = nn.Sequential(
            nn.LayerNorm(h),
            nn.Linear(h, h),
            nn.SiLU(),
            nn.Linear(h, h),
        )
        self.coeff_head = nn.Sequential(
            nn.LayerNorm(2 * h),
            nn.Linear(2 * h, h),
            nn.SiLU(),
            nn.Linear(h, r),
        )
        self.delta_drop = nn.Dropout(config.controlled_delta_dropout)
        # Do not LayerNorm the intervention delta: amplitude is part of the
        # causal signal.  LayerNorm can amplify tiny centered coefficient noise
        # into a full-magnitude average residual.  Start with a moderate gain
        # and let training adjust it.
        self.delta_gain = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        nn.init.normal_(self.base_head[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.base_head[-1].bias)
        # The controlled path must be alive before long training.  Using 1e-3
        # for both basis and coefficients makes their product almost zero and
        # lets the weak base dominate the preflight task.  Keep the base tiny,
        # but initialize basis/coefficients at normal transformer residual scale.
        nn.init.normal_(self.basis_head[-1].weight, mean=0.0, std=3e-2)
        nn.init.zeros_(self.basis_head[-1].bias)
        nn.init.normal_(self.coeff_head[-1].weight, mean=0.0, std=5e-2)
        nn.init.zeros_(self.coeff_head[-1].bias)
        nn.init.normal_(self.direct_action_mlp[-1].weight, mean=0.0, std=5e-2)
        nn.init.zeros_(self.direct_action_mlp[-1].bias)

    def _coeff(
        self,
        rollout_base: Tensor,
        context_kv: Tensor,
        *,
        action_tokens: Tensor | None,
        neutral: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return coefficients, latent action tokens, and direct summary.

        ``neutral=True`` deliberately excludes trajectory/action tokens and uses
        learned neutral queries.  The output is still state/context dependent,
        so it can absorb no-op phase/context bias, but it cannot carry the
        candidate action chunk.  The centered coefficient subtracts this term.
        """
        b, n, h = rollout_base.shape
        if neutral:
            queries = self.neutral_queries.expand(b, -1, -1).to(device=rollout_base.device, dtype=rollout_base.dtype)
            kv = self.action_kv_norm(context_kv)
            direct = self.neutral_bias.to(device=rollout_base.device, dtype=rollout_base.dtype).expand(b, n, -1)
        else:
            queries = self.action_queries.expand(b, -1, -1).to(device=rollout_base.device, dtype=rollout_base.dtype)
            if action_tokens is None:
                kv_source = context_kv
                action_source = context_kv
            else:
                kv_source = torch.cat([context_kv, action_tokens], dim=1)
                action_source = action_tokens
            kv = self.action_kv_norm(kv_source)
            direct_action = self.direct_action_mlp(self.direct_action_norm(action_source).mean(dim=1))
            direct = direct_action[:, None, :].expand(-1, n, -1)

        latent_action, _ = self.action_cross(queries, kv, kv, need_weights=False)
        rq = self.rollout_query_norm(rollout_base)
        la = self.action_latent_norm(latent_action)
        action_context, _ = self.coeff_cross(rq, la, la, need_weights=False)
        coeff = torch.tanh(self.coeff_head(torch.cat([rq, action_context + direct], dim=-1)))
        return coeff, latent_action, direct

    def forward(self, rollout_base: Tensor, context_kv: Tensor, action_tokens: Tensor | None = None) -> dict[str, Tensor]:
        cfg = self.config
        b, n, h = rollout_base.shape
        if n != cfg.future_token_count or h != cfg.hidden_size:
            raise ValueError(f"rollout_base must be [B,{cfg.future_token_count},{cfg.hidden_size}], got {tuple(rollout_base.shape)}")
        base_effect = self.base_head(rollout_base)
        basis = self.basis_head(rollout_base).reshape(b, n, cfg.controlled_delta_rank, h)
        coeff_action, latent_action, _ = self._coeff(rollout_base, context_kv, action_tokens=action_tokens, neutral=False)
        coeff_neutral, latent_neutral, _ = self._coeff(rollout_base, context_kv, action_tokens=None, neutral=True)
        coeff_delta = coeff_action - coeff_neutral
        controlled_delta = torch.einsum("bnr,bnrh->bnh", coeff_delta, basis) / float(cfg.controlled_delta_rank) ** 0.5
        controlled_delta = self.delta_drop(controlled_delta * self.delta_gain.to(device=controlled_delta.device, dtype=controlled_delta.dtype))
        pred_effect = base_effect + controlled_delta
        return {
            "rollout_base_effect_pred": base_effect,
            "rollout_delta_pred": controlled_delta,
            "rollout_effect_pred": pred_effect,
            "rollout_transition_basis": basis,
            "rollout_action_coeff": coeff_action,
            "rollout_neutral_coeff": coeff_neutral,
            "rollout_centered_coeff": coeff_delta,
            "latent_action_tokens": latent_action,
            "latent_neutral_tokens": latent_neutral,
            "rollout_coeff_abs_mean": coeff_action.detach().float().abs().mean(),
            "rollout_neutral_coeff_abs_mean": coeff_neutral.detach().float().abs().mean(),
            "rollout_centered_coeff_abs_mean": coeff_delta.detach().float().abs().mean(),
            "rollout_basis_norm": basis.detach().float().norm(dim=-1).mean(),
            "rollout_delta_norm": controlled_delta.detach().float().norm(dim=-1).mean(),
            "rollout_base_norm": base_effect.detach().float().norm(dim=-1).mean(),
            "rollout_delta_gain": self.delta_gain.detach().float().abs(),
        }

class TemporalWorldActionDiT(nn.Module):
    def __init__(self, config: V38PolicyConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        h = config.hidden_size
        self.visual_memory = DenseVisualMemory(config)
        self.rollout_codec = RolloutTargetCodec(config)
        self.seed = UnifiedCanvasSeed(config)
        self.time = TimeEmbedding(h)
        self.content_mod = nn.Sequential(nn.LayerNorm(2 * h), nn.Linear(2 * h, h), nn.SiLU(), nn.Linear(h, h))
        nn.init.normal_(self.content_mod[-1].weight, mean=0.0, std=2e-2)
        nn.init.zeros_(self.content_mod[-1].bias)
        self.content_mod_scale = nn.Parameter(torch.tensor(0.10))
        self.blocks = nn.ModuleList([TemporalDynamicsBoundDiTBlock(config) for _ in range(config.depth)])
        self.final_norm = nn.LayerNorm(h)
        self.direct_physical_head = CanvasPhysicalVelocityHead(config)
        self.rollout_residual_head = RolloutActionResidualHead(config)
        self.controlled_dynamics = ControlledResidualLatentDynamics(config)
        self.event_probe = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 3))
        self.motion_probe = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 1))

    def forward(
        self,
        noisy_physical: Tensor,
        time: Tensor,
        visual: Tensor,
        state_history: Tensor,
        state: Tensor,
        executed_history: Tensor,
        proposal_tokens: Tensor,
        proposal_keep: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if proposal_keep is None:
            proposal_keep = torch.ones(noisy_physical.shape[0], device=noisy_physical.device, dtype=noisy_physical.dtype)
        visual_memory = self.visual_memory(visual)
        rollout_init = self.rollout_codec.rollout_init(visual)
        canvas, slices = self.seed(
            noisy_physical=noisy_physical,
            state=state,
            state_history=state_history,
            executed_history=executed_history,
            proposal_tokens=proposal_tokens,
            proposal_keep=proposal_keep,
            rollout_init=rollout_init,
        )
        time_emb = self.time(time.to(dtype=canvas.dtype))
        gate_rows: list[dict[str, Tensor]] = []
        content_norm_rows: list[Tensor] = []
        time_norm_rows: list[Tensor] = []
        for block in self.blocks:
            summary = torch.cat([canvas.mean(dim=1), visual_memory.mean(dim=1)], dim=-1)
            content_delta = self.content_mod(summary) * self.content_mod_scale.to(device=canvas.device, dtype=canvas.dtype)
            mod_emb = time_emb + content_delta
            content_norm_rows.append(content_delta.float().norm(dim=-1).mean())
            time_norm_rows.append(time_emb.float().norm(dim=-1).mean())
            canvas, gates = block(canvas, visual_memory, mod_emb, slices)
            gate_rows.append(gates)
        canvas = self.final_norm(canvas)
        trajectory = canvas[:, slices["trajectory"]]
        rollout = canvas[:, slices["rollout"]]
        registers = canvas[:, slices["registers"]]
        trajectory_pooled = self.direct_physical_head.pooled(trajectory)
        direct_velocity = self.direct_physical_head(trajectory)
        context_kv = torch.cat([
            canvas[:, slices["state"]],
            canvas[:, slices["state_history"]],
            canvas[:, slices["executed"]],
            canvas[:, slices["proposal"]],
        ], dim=1)
        dynamics = self.controlled_dynamics(
            rollout_init.to(device=canvas.device, dtype=canvas.dtype),
            context_kv,
            action_tokens=trajectory,
        )
        controlled_delta = dynamics["rollout_delta_pred"]
        rollout_residual_velocity, rollout_alpha = self.rollout_residual_head(trajectory_pooled, controlled_delta)
        pred_physical_velocity = direct_velocity + rollout_residual_velocity
        rollout_effect_pred = dynamics["rollout_effect_pred"]
        event_context = controlled_delta[:, : self.config.action_horizon] if controlled_delta.shape[1] >= self.config.action_horizon else trajectory_pooled
        return {
            "canvas_tokens": canvas,
            "trajectory_tokens": trajectory,
            "rollout_tokens": rollout,
            "register_tokens": registers,
            "direct_physical_velocity": direct_velocity,
            "rollout_residual_velocity": rollout_residual_velocity,
            "rollout_alpha": rollout_alpha,
            "pred_physical_velocity": pred_physical_velocity,
            "rollout_effect_pred": rollout_effect_pred,
            "rollout_base_effect_pred": dynamics["rollout_base_effect_pred"],
            "rollout_delta_pred": controlled_delta,
            "rollout_coeff_abs_mean": dynamics["rollout_coeff_abs_mean"],
            "rollout_neutral_coeff_abs_mean": dynamics["rollout_neutral_coeff_abs_mean"],
            "rollout_centered_coeff_abs_mean": dynamics["rollout_centered_coeff_abs_mean"],
            "rollout_basis_norm": dynamics["rollout_basis_norm"],
            "rollout_delta_norm": dynamics["rollout_delta_norm"],
            "rollout_base_norm": dynamics["rollout_base_norm"],
            "rollout_delta_gain": dynamics["rollout_delta_gain"],
            # Compatibility aliases; these now refer to controlled residual rollout effect, not future-noisy denoise.
            "future_latent_pred": rollout_effect_pred,
            "action_effect_pred": rollout_effect_pred,
            "event_logits": self.event_probe(event_context),
            "motion_logits": self.motion_probe(trajectory_pooled.detach()).squeeze(-1),
            "transition_latent": controlled_delta.mean(dim=1, keepdim=True).expand(-1, self.config.action_horizon, -1),
            "gate_self": torch.stack([row["gate_self"] for row in gate_rows]).mean() if gate_rows else torch.zeros((), device=canvas.device),
            "gate_visual": torch.stack([row["gate_visual"] for row in gate_rows]).mean() if gate_rows else torch.zeros((), device=canvas.device),
            "gate_rollout": torch.stack([row["gate_rollout"] for row in gate_rows]).mean() if gate_rows else torch.zeros((), device=canvas.device),
            "gate_ffn": torch.stack([row["gate_ffn"] for row in gate_rows]).mean() if gate_rows else torch.zeros((), device=canvas.device),
            "mod_content_norm": torch.stack(content_norm_rows).mean() if content_norm_rows else torch.zeros((), device=canvas.device),
            "mod_time_norm": torch.stack(time_norm_rows).mean() if time_norm_rows else torch.zeros((), device=canvas.device),
            "mod_content_to_time": (torch.stack(content_norm_rows).mean() / torch.stack(time_norm_rows).mean().clamp_min(1e-6)) if content_norm_rows and time_norm_rows else torch.zeros((), device=canvas.device),
        }

    @torch.no_grad()
    def target_rollout_effect(self, visual: Tensor, target_visual: Tensor) -> Tensor:
        return self.rollout_codec.target_effect(visual, target_visual)


class V38PolicySystem(nn.Module):
    def __init__(self, policy_config: V38PolicyConfig) -> None:
        super().__init__()
        self.policy_config = policy_config
        self.codec = PhysicalActionCodec(policy_config)
        self.proposal = RejectableHistoryProposal(policy_config)
        self.planner = TemporalWorldActionDiT(policy_config)

    def _policy_forward(
        self,
        noisy_physical: Tensor,
        time: Tensor,
        visual: Tensor,
        state_history: Tensor,
        state: Tensor,
        executed_history: Tensor,
        proposal_tokens: Tensor,
        proposal_keep: Tensor,
    ) -> dict[str, Tensor]:
        return self.planner(noisy_physical, time, visual, state_history, state, executed_history, proposal_tokens, proposal_keep)

    @torch.no_grad()
    def build_rollout_target_pack(self, visual: Tensor, target_visual: Tensor) -> dict[str, Tensor]:
        target = self.planner.target_rollout_effect(visual, target_visual).detach()
        return {"rollout_effect_target": target, "future_latent_target": target, "action_effect_target": target}

    def flow_training_forward(
        self,
        visual: Tensor,
        state_history: Tensor,
        executed_history: Tensor,
        state: Tensor,
        target_action: Tensor,
        *,
        action_state: Tensor | None = None,
        target_visual: Tensor | None = None,
        rollout_target_pack: dict[str, Tensor] | None = None,
        future_training_pack: dict[str, Tensor] | None = None,
        proposal_dropout: float | None = None,
        make_counterfactuals: bool = True,
    ) -> dict[str, Tensor]:
        del future_training_pack  # Future-noisy denoise is intentionally not a V38.5 input path.
        proposal = self.proposal(executed_history)
        codec_state = state if action_state is None else action_state
        target_physical = self.codec.encode(target_action, codec_state)
        noise = self.codec.sample_noise(target_physical.shape[0], device=target_physical.device, dtype=target_physical.dtype)
        t = torch.rand(target_physical.shape[0], device=target_physical.device, dtype=target_physical.dtype)
        noisy_physical = (1 - t[:, None, None]) * target_physical + t[:, None, None] * noise
        target_physical_velocity = noise - target_physical
        drop = self.policy_config.proposal_dropout if proposal_dropout is None else float(proposal_dropout)
        keep = (torch.rand(target_physical.shape[0], device=target_physical.device) >= drop).to(target_physical.dtype)

        action_policy = self._policy_forward(noisy_physical, t, visual, state_history, state, executed_history, proposal["tokens"].detach(), keep)
        clean_physical_estimate = noisy_physical - t[:, None, None] * action_policy["pred_physical_velocity"]
        decoded_action = self.codec.decode(clean_physical_estimate, codec_state)
        out = {
            "pred_physical_velocity": action_policy["pred_physical_velocity"],
            "target_physical_velocity": target_physical_velocity,
            "target_physical": target_physical,
            "clean_physical_estimate": clean_physical_estimate,
            "proposal_action": proposal["action"],
            "time": t,
            "noisy_physical_action": noisy_physical,
            "pred_action_estimate": decoded_action,
            "event_logits": action_policy["event_logits"],
            "motion_logits": action_policy["motion_logits"],
            "transition_latent": action_policy["transition_latent"],
            "rollout_effect_pred": action_policy["rollout_effect_pred"],
            "rollout_base_effect_pred": action_policy["rollout_base_effect_pred"],
            "rollout_delta_pred": action_policy["rollout_delta_pred"],
            "rollout_coeff_abs_mean": action_policy["rollout_coeff_abs_mean"],
            "rollout_neutral_coeff_abs_mean": action_policy["rollout_neutral_coeff_abs_mean"],
            "rollout_centered_coeff_abs_mean": action_policy["rollout_centered_coeff_abs_mean"],
            "rollout_basis_norm": action_policy["rollout_basis_norm"],
            "rollout_delta_norm": action_policy["rollout_delta_norm"],
            "rollout_base_norm": action_policy["rollout_base_norm"],
            "rollout_delta_gain": action_policy["rollout_delta_gain"],
            "future_latent_pred": action_policy["future_latent_pred"],
            "action_effect_pred": action_policy["action_effect_pred"],
            "direct_physical_velocity": action_policy["direct_physical_velocity"],
            "rollout_residual_velocity": action_policy["rollout_residual_velocity"],
            "rollout_alpha": action_policy["rollout_alpha"],
            "gate_self": action_policy["gate_self"],
            "gate_visual": action_policy["gate_visual"],
            "gate_rollout": action_policy["gate_rollout"],
            "gate_ffn": action_policy["gate_ffn"],
            "mod_content_norm": action_policy["mod_content_norm"],
            "mod_time_norm": action_policy["mod_time_norm"],
            "mod_content_to_time": action_policy["mod_content_to_time"],
            "future_conditioned_action_loss": torch.zeros((), device=target_physical.device, dtype=target_physical.dtype),
        }

        pack = rollout_target_pack
        if pack is None and target_visual is not None:
            pack = self.build_rollout_target_pack(visual, target_visual)
        if pack is not None:
            target = pack["rollout_effect_target"].to(device=target_physical.device, dtype=action_policy["rollout_effect_pred"].dtype)
            out["rollout_effect_target"] = target
            out["future_latent_target"] = target
            out["future_latent_velocity_target"] = target
            out["action_effect_target"] = target
            if make_counterfactuals:
                # Same diffusion time and same noise; only the clean action component changes.
                hold_action = codec_state[:, None].expand_as(target_action)
                hold_physical = self.codec.encode(hold_action, codec_state)
                hold_noisy = (1 - t[:, None, None]) * hold_physical + t[:, None, None] * noise
                hold_policy = self._policy_forward(hold_noisy.detach(), t.detach(), visual, state_history, state, executed_history, proposal["tokens"].detach(), keep)
                out["rollout_effect_pred_hold_action"] = hold_policy["rollout_effect_pred"]
                out["rollout_delta_pred_hold_action"] = hold_policy["rollout_delta_pred"]
                out["rollout_base_effect_pred_hold_action"] = hold_policy["rollout_base_effect_pred"]
                if target_physical.shape[0] > 1:
                    perm = torch.arange(target_physical.shape[0] - 1, -1, -1, device=target_physical.device)
                    shuffle_physical = target_physical[perm]
                else:
                    shuffle_physical = target_physical
                shuffle_noisy = (1 - t[:, None, None]) * shuffle_physical + t[:, None, None] * noise
                shuffle_policy = self._policy_forward(shuffle_noisy.detach(), t.detach(), visual, state_history, state, executed_history, proposal["tokens"].detach(), keep)
                out["rollout_effect_pred_shuffle_action"] = shuffle_policy["rollout_effect_pred"]
                out["rollout_delta_pred_shuffle_action"] = shuffle_policy["rollout_delta_pred"]
                out["rollout_base_effect_pred_shuffle_action"] = shuffle_policy["rollout_base_effect_pred"]
        return out

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
    ) -> Tensor | dict[str, Tensor]:
        proposal = self.proposal(executed_history)
        steps = int(steps or self.policy_config.inference_steps)
        if noise is None:
            x = self.codec.sample_noise(visual.shape[0], device=visual.device, dtype=visual.dtype)
        else:
            x = noise.clone()
            if x.shape[-1] == self.policy_config.action_dim:
                x = self.codec.encode(x.to(device=visual.device, dtype=visual.dtype), state.to(device=visual.device, dtype=visual.dtype))
            elif x.shape[-1] != self.policy_config.physical_action_dim:
                raise ValueError("noise must have last dim action_dim or physical_action_dim")
        keep = torch.full((visual.shape[0],), 1.0 if use_proposal else 0.0, device=visual.device, dtype=visual.dtype)
        last_out: dict[str, Tensor] | None = None
        for index in range(steps, 0, -1):
            t = torch.full((visual.shape[0],), float(index) / float(steps), device=visual.device, dtype=visual.dtype)
            last_out = self._policy_forward(x, t, visual, state_history, state, executed_history, proposal["tokens"], keep)
            x = x - last_out["pred_physical_velocity"] / float(steps)
        action = self.codec.decode(x, state)
        if return_event_logits:
            zero_t = torch.zeros((visual.shape[0],), device=visual.device, dtype=visual.dtype)
            event = self._policy_forward(x, zero_t, visual, state_history, state, executed_history, proposal["tokens"], keep)
            return {"action": action, "physical_action": x, "event_logits": event["event_logits"], "motion_logits": event["motion_logits"]}
        return action

    def parameter_report(self) -> dict[str, int]:
        report = {
            "history_proposal": sum(p.numel() for p in self.proposal.parameters()),
            "physical_action_codec": sum(p.numel() for p in self.codec.parameters()),
            "controlled_residual_dit": sum(p.numel() for p in self.planner.parameters()),
        }
        report["total"] = sum(p.numel() for p in self.parameters())
        report["trainable"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return report


__all__ = [
    "V38PolicyConfig",
    "DenseVisualMemory",
    "RolloutTargetCodec",
    "UnifiedCanvasSeed",
    "TemporalDynamicsBoundDiTBlock",
    "TemporalWorldActionDiT",
    "V38PolicySystem",
]
