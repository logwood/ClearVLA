from __future__ import annotations

"""Shared world/action trunk components used by V38 and the current policy."""

from typing import Protocol

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .codec import PhysicalActionTokenLift
from .primitives import BiasFreeFFN, sinusoidal_positions


class TrunkPrimitiveConfig(Protocol):
    action_basis_tokens: int
    action_dim: int
    action_horizon: int
    arm_dim: int
    base_effect_hidden: int
    canvas_dropout: float
    canvas_registers: int
    controlled_base_mode: str
    controlled_delta_dropout: float
    controlled_delta_rank: int
    dropout: float
    executed_history_length: int
    ffn_expansion: float
    first_execution_steps: int
    future_anchors: int
    future_grid_size: int
    gripper_field_dim: int
    gripper_field_mode: str
    hidden_size: int
    latent_action_tokens: int
    mid_execution_steps: int
    neutral_action_tokens: int
    num_cameras: int
    num_heads: int
    patches_per_camera: int
    physical_action_dim: int
    role_dropout: float
    rollout_tail_full_step: int
    rollout_tail_start_step: int
    state_dim: int
    visual_history_length: int
    visual_memory_dropout: float
    visual_token_dim: int


class HorizonRoleEmbedding(nn.Module):
    """Explicit execution/planning role embedding for horizon tokens."""

    def __init__(self, config: TrunkPrimitiveConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.execution = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        self.mid = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        self.tail = nn.Parameter(torch.randn(1, 1, h) * 0.02)

    def forward(self, batch: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        h = self.execution.shape[-1]
        role = torch.empty(1, self.config.action_horizon, h, device=device, dtype=dtype)
        role[:, : self.config.first_execution_steps] = self.execution.to(device=device, dtype=dtype)
        role[:, self.config.first_execution_steps : self.config.mid_execution_steps] = self.mid.to(
            device=device, dtype=dtype
        )
        role[:, self.config.mid_execution_steps :] = self.tail.to(device=device, dtype=dtype)
        return role.expand(batch, -1, -1)


class DenseVisualMemory(nn.Module):
    """Per-token DINO projection with factorized identity embeddings."""

    def __init__(self, config: TrunkPrimitiveConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size
        d = config.visual_token_dim
        self.proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, h))
        self.history_type = nn.Parameter(
            torch.randn(1, config.visual_history_length, 1, 1, h) * 0.02
        )
        self.camera_type = nn.Parameter(torch.randn(1, 1, config.num_cameras, 1, h) * 0.02)
        self.patch_type = nn.Parameter(torch.randn(1, 1, 1, config.patches_per_camera, h) * 0.02)
        self.out_norm = nn.LayerNorm(h)
        self.drop = nn.Dropout(config.visual_memory_dropout)

    def forward(self, visual: Tensor) -> Tensor:
        cfg = self.config
        if visual.ndim != 5:
            raise ValueError(f"visual must be [B,H,C,P,D], got {tuple(visual.shape)}")
        b, hist, cams, patches, dim = visual.shape
        expected = (
            cfg.visual_history_length,
            cfg.num_cameras,
            cfg.patches_per_camera,
            cfg.visual_token_dim,
        )
        if (hist, cams, patches, dim) != expected:
            raise ValueError(
                f"V38 visual geometry mismatch: got {(hist, cams, patches, dim)}, expected {expected}"
            )
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

    def __init__(self, config: TrunkPrimitiveConfig) -> None:
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
            x = x.permute(0, 2, 3, 1).reshape(
                b, f, c * cfg.future_grid_size * cfg.future_grid_size, d
            )
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
        current_h = self.init_proj(
            current.to(device=next(self.parameters()).device, dtype=next(self.parameters()).dtype)
        )
        return (
            current_h[:, None]
            .expand(-1, cfg.future_anchors, -1, -1)
            .reshape(visual.shape[0], cfg.future_token_count, cfg.hidden_size)
        )

    @torch.no_grad()
    def target_effect(self, visual: Tensor, target_visual: Tensor) -> Tensor:
        cfg = self.config
        if target_visual.ndim != 6:
            raise ValueError(
                f"target_visual must be [B,F,H,C,P,D], got {tuple(target_visual.shape)}"
            )
        future = target_visual[:, : cfg.future_anchors, -1]
        current = visual[:, -1][:, None].expand(-1, cfg.future_anchors, -1, -1, -1)
        residual = future.float() - current.float()
        pooled = self.spatial_pool_tokens(residual)  # [B,K,C*G*G,D]
        target = self.target_proj(
            pooled.to(device=next(self.parameters()).device, dtype=next(self.parameters()).dtype)
        )
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

    def __init__(self, config: TrunkPrimitiveConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.state_proj = nn.Linear(config.state_dim, h)
        self.state_history_proj = nn.Linear(config.state_dim, h)
        self.executed_proj = nn.Linear(config.action_dim, h)
        self.proposal_proj = nn.Identity()
        self.noisy_physical_lift = PhysicalActionTokenLift(config)
        self.horizon_role = HorizonRoleEmbedding(config)
        self.action_basis_embed = nn.Parameter(
            torch.randn(1, 1, config.action_basis_tokens, h) * 0.02
        )
        self.role_embed = nn.Parameter(torch.randn(8, h) * 0.02)
        self.role_drop = nn.Dropout(config.role_dropout)
        self.task_token = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        self.rollout_anchor_type = nn.Parameter(torch.randn(1, config.future_anchors, 1, h) * 0.02)
        self.rollout_grid_type = nn.Parameter(
            torch.randn(
                1, 1, config.num_cameras * config.future_grid_size * config.future_grid_size, h
            )
            * 0.02
        )
        self.registers = nn.Parameter(torch.randn(1, config.canvas_registers, h) * 0.02)
        self.proposal_type = nn.Parameter(torch.randn(1, config.action_horizon, h) * 0.02)
        self.executed_type = nn.Parameter(torch.randn(1, config.executed_history_length, h) * 0.02)
        self.state_history_type = nn.Parameter(
            torch.randn(1, config.visual_history_length, h) * 0.02
        )
        self.drop = nn.Dropout(config.canvas_dropout)
        self.register_buffer(
            "horizon_position",
            sinusoidal_positions(range(1, config.action_horizon + 1), h)[None],
            persistent=True,
        )

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
        task = (
            self.task_token.expand(b, -1, -1).to(device=device, dtype=dtype) + role[self.ROLE_TASK]
        )
        state_tok = self.state_proj(state)[:, None] + role[self.ROLE_STATE]
        hist = (
            self.state_history_proj(state_history)
            + self.state_history_type.to(device=device, dtype=dtype)
            + role[self.ROLE_STATE_HISTORY]
        )
        executed = (
            self.executed_proj(executed_history)
            + self.executed_type.to(device=device, dtype=dtype)
            + role[self.ROLE_EXECUTED]
        )
        proposal = (
            self.proposal_proj(proposal_tokens) * proposal_keep[:, None, None]
            + self.proposal_type.to(device=device, dtype=dtype)
            + role[self.ROLE_PROPOSAL]
        )
        noisy_base = (
            self.noisy_physical_lift(noisy_physical)
            + self.horizon_position.to(device=device, dtype=dtype)
            + self.horizon_role(b, device=device, dtype=dtype)
            + role[self.ROLE_NOISY_ACTION]
        )
        noisy = (
            noisy_base[:, :, None, :] + self.action_basis_embed.to(device=device, dtype=dtype)
        ).reshape(b, cfg.action_horizon * cfg.action_basis_tokens, cfg.hidden_size)
        if rollout_init.shape != (b, cfg.future_token_count, cfg.hidden_size):
            raise ValueError(
                f"rollout_init must be [B,{cfg.future_token_count},{cfg.hidden_size}], got {tuple(rollout_init.shape)}"
            )
        rollout = rollout_init.to(device=device, dtype=dtype).reshape(
            b,
            cfg.future_anchors,
            cfg.num_cameras * cfg.future_grid_size * cfg.future_grid_size,
            cfg.hidden_size,
        )
        rollout = (
            rollout
            + self.rollout_anchor_type.to(device=device, dtype=dtype)
            + self.rollout_grid_type.to(device=device, dtype=dtype)
        )
        rollout = (
            rollout.reshape(b, cfg.future_token_count, cfg.hidden_size) + role[self.ROLE_ROLLOUT]
        )
        registers = (
            self.registers.expand(b, -1, -1).to(device=device, dtype=dtype)
            + role[self.ROLE_REGISTER]
        )
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
            "trajectory": slice(
                starts[5], starts[5] + cfg.action_horizon * cfg.action_basis_tokens
            ),
            "rollout": slice(starts[6], starts[6] + cfg.future_token_count),
            "registers": slice(starts[7], starts[7] + cfg.canvas_registers),
        }
        return self.drop(torch.cat(parts, dim=1)), slices


class TemporalDynamicsBoundDiTBlock(nn.Module):
    """Canvas block with explicit action-to-rollout transition sublayer."""

    def __init__(self, config: TrunkPrimitiveConfig) -> None:
        super().__init__()
        h = config.hidden_size
        self.n1 = nn.LayerNorm(h, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(
            h, config.num_heads, batch_first=True, dropout=config.dropout
        )
        self.n2 = nn.LayerNorm(h, elementwise_affine=False)
        self.mem_norm = nn.LayerNorm(h)
        self.cross = nn.MultiheadAttention(
            h, config.num_heads, batch_first=True, dropout=config.dropout
        )
        self.n_dyn_q = nn.LayerNorm(h, elementwise_affine=False)
        self.n_dyn_kv = nn.LayerNorm(h)
        self.rollout_cross = nn.MultiheadAttention(
            h, config.num_heads, batch_first=True, dropout=config.dropout
        )
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

    def forward(
        self, canvas: Tensor, visual_memory: Tensor, mod_embed: Tensor, slices: dict[str, slice]
    ) -> tuple[Tensor, dict[str, Tensor]]:
        sa_s, sa_c, sa_g, ca_s, ca_c, ca_g, dy_s, dy_c, dy_g, ff_s, ff_c, ff_g = self.mod(
            mod_embed
        ).chunk(12, dim=-1)
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
        kv_parts = [
            canvas[:, slices[name]]
            for name in ("state", "state_history", "executed", "proposal", "trajectory")
        ]
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
    def __init__(self, config: TrunkPrimitiveConfig) -> None:
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
        return trajectory_tokens.reshape(
            b, cfg.action_horizon, cfg.action_basis_tokens, cfg.hidden_size
        ).mean(dim=2)

    def forward(self, trajectory_tokens: Tensor) -> Tensor:
        cfg = self.config
        b = trajectory_tokens.shape[0]
        grouped = trajectory_tokens.reshape(
            b, cfg.action_horizon, cfg.action_basis_tokens, cfg.hidden_size
        )
        return self.net(
            grouped.reshape(b, cfg.action_horizon, cfg.action_basis_tokens * cfg.hidden_size)
        )


class RolloutActionResidualHead(nn.Module):
    """Tail action residual that must read rollout latent tokens."""

    def __init__(self, config: TrunkPrimitiveConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.query_norm = nn.LayerNorm(h)
        self.rollout_norm = nn.LayerNorm(h)
        self.cross = nn.MultiheadAttention(
            h, config.num_heads, batch_first=True, dropout=config.dropout
        )
        self.net = nn.Sequential(
            nn.LayerNorm(h),
            nn.Linear(h, 2 * h),
            nn.SiLU(),
            nn.Linear(2 * h, config.physical_action_dim),
        )
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

    def __init__(self, config: TrunkPrimitiveConfig) -> None:
        super().__init__()
        self.config = config
        self.base_mode = str(getattr(config, "controlled_base_mode", "learned"))
        if self.base_mode not in {"learned", "fixed_zero"}:
            raise ValueError(f"unsupported controlled_base_mode={self.base_mode!r}")
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
        self.action_queries = nn.Parameter(
            torch.randn(1, int(config.latent_action_tokens), h) * 0.02
        )
        self.neutral_queries = nn.Parameter(
            torch.randn(1, int(config.neutral_action_tokens), h) * 0.02
        )
        self.neutral_bias = nn.Parameter(torch.zeros(1, 1, h))
        self.action_kv_norm = nn.LayerNorm(h)
        self.action_cross = nn.MultiheadAttention(
            h, config.num_heads, batch_first=True, dropout=config.dropout
        )
        self.rollout_query_norm = nn.LayerNorm(h)
        self.action_latent_norm = nn.LayerNorm(h)
        self.coeff_cross = nn.MultiheadAttention(
            h, config.num_heads, batch_first=True, dropout=config.dropout
        )
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
        if self.base_mode == "fixed_zero":
            # A learned base has no identifiable target: base + delta can stay
            # correct while both terms grow in opposite directions.  The
            # no-change origin is the only target-free baseline with a unique
            # residual decomposition, so new V39 runs freeze this legacy head.
            self.base_head.requires_grad_(False)

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
            queries = self.neutral_queries.expand(b, -1, -1).to(
                device=rollout_base.device, dtype=rollout_base.dtype
            )
            kv = self.action_kv_norm(context_kv)
            direct = self.neutral_bias.to(
                device=rollout_base.device, dtype=rollout_base.dtype
            ).expand(b, n, -1)
        else:
            queries = self.action_queries.expand(b, -1, -1).to(
                device=rollout_base.device, dtype=rollout_base.dtype
            )
            if action_tokens is None:
                kv_source = context_kv
                action_source = context_kv
            else:
                kv_source = torch.cat([context_kv, action_tokens], dim=1)
                action_source = action_tokens
            kv = self.action_kv_norm(kv_source)
            direct_action = self.direct_action_mlp(
                self.direct_action_norm(action_source).mean(dim=1)
            )
            direct = direct_action[:, None, :].expand(-1, n, -1)

        latent_action, _ = self.action_cross(queries, kv, kv, need_weights=False)
        rq = self.rollout_query_norm(rollout_base)
        la = self.action_latent_norm(latent_action)
        action_context, _ = self.coeff_cross(rq, la, la, need_weights=False)
        coeff = torch.tanh(self.coeff_head(torch.cat([rq, action_context + direct], dim=-1)))
        return coeff, latent_action, direct

    def forward(
        self,
        rollout_base: Tensor,
        context_kv: Tensor,
        action_tokens: Tensor | None = None,
        *,
        transition_tokens: Tensor | None = None,
    ) -> dict[str, Tensor]:
        cfg = self.config
        b, n, h = rollout_base.shape
        if n != cfg.future_token_count or h != cfg.hidden_size:
            raise ValueError(
                f"rollout_base must be [B,{cfg.future_token_count},{cfg.hidden_size}], got {tuple(rollout_base.shape)}"
            )
        transition = rollout_base if transition_tokens is None else transition_tokens
        if transition.shape != rollout_base.shape:
            raise ValueError(
                f"transition_tokens must match rollout_base {tuple(rollout_base.shape)}, got {tuple(transition.shape)}"
            )
        if self.base_mode == "fixed_zero":
            base_effect = torch.zeros_like(rollout_base)
        else:
            base_effect = self.base_head(rollout_base)
        # The baseline and transition representation are deliberately separate.
        # V39 can keep a fixed, identifiable origin while still using the full
        # deep rollout canvas to construct action-conditioned directions.
        basis = self.basis_head(transition).reshape(b, n, cfg.controlled_delta_rank, h)
        coeff_action, latent_action, _ = self._coeff(
            transition, context_kv, action_tokens=action_tokens, neutral=False
        )
        coeff_neutral, latent_neutral, _ = self._coeff(
            transition, context_kv, action_tokens=None, neutral=True
        )
        coeff_delta = coeff_action - coeff_neutral
        controlled_delta = (
            torch.einsum("bnr,bnrh->bnh", coeff_delta, basis)
            / float(cfg.controlled_delta_rank) ** 0.5
        )
        controlled_delta = self.delta_drop(
            controlled_delta
            * self.delta_gain.to(device=controlled_delta.device, dtype=controlled_delta.dtype)
        )
        pred_effect = base_effect + controlled_delta
        base_norm = base_effect.detach().float().norm(dim=-1).mean()
        delta_norm = controlled_delta.detach().float().norm(dim=-1).mean()
        effect_norm = pred_effect.detach().float().norm(dim=-1).mean().clamp_min(1e-6)
        expansion_ratio = (
            effect_norm.new_ones(())
            if self.base_mode == "fixed_zero"
            else (base_norm + delta_norm) / effect_norm
        )
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
            "rollout_delta_norm": delta_norm,
            "rollout_base_norm": base_norm,
            "rollout_decomposition_expansion_ratio": expansion_ratio,
            "rollout_base_is_fixed_zero": base_norm.new_tensor(
                float(self.base_mode == "fixed_zero")
            ),
            "rollout_delta_gain": self.delta_gain.detach().float().abs(),
        }
