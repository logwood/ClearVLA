from __future__ import annotations

"""Current staged world/action trunk and layer contracts."""

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .codec import ParsevalGripperTemporalFrame
from .config import V39PolicyConfig
from .contracts import scaled_contract_view as _scaled_contract_view
from .decoder import HierarchicalMMDiTActionDecoder
from .legacy import (
    AdaptiveRecurrentCVAEActionDecoder,
    HierarchicalLatentMainActionDecoder,
    LatentCVAEActionDecoder,
    LayeredV37StyleResidualActionFlowDenoiser,
    V37StyleResidualActionFlowDenoiser,
)
from .primitives import TimeEmbedding
from .trunk_primitives import (
    CanvasPhysicalVelocityHead,
    ControlledResidualLatentDynamics,
    DenseVisualMemory,
    RolloutActionResidualHead,
    RolloutTargetCodec,
    TemporalDynamicsBoundDiTBlock,
    UnifiedCanvasSeed,
)


def _align_milestone_tokens_to_horizon(tokens: Tensor, horizon: int) -> Tensor:
    """Expand one pooled token per action segment onto the action timeline."""

    if tokens.ndim != 3:
        raise ValueError(f"milestone tokens must be [B,K,H], got {tuple(tokens.shape)}")
    horizon = int(horizon)
    steps = int(tokens.shape[1])
    if horizon < 1 or steps < 1 or steps > horizon:
        raise ValueError(f"expected 1 <= milestone steps <= horizon, got steps={steps} horizon={horizon}")
    rows: list[Tensor] = []
    for step in range(steps):
        lo = int(round(step * horizon / float(steps)))
        hi = int(round((step + 1) * horizon / float(steps)))
        hi = max(hi, lo + 1)
        hi = min(hi, horizon)
        rows.append(tokens[:, step:step + 1].expand(-1, hi - lo, -1))
    aligned = torch.cat(rows, dim=1)
    if aligned.shape[1] != horizon:
        raise RuntimeError(f"milestone alignment produced {aligned.shape[1]} tokens for horizon={horizon}")
    return aligned


def _rollout_tokens_to_action_horizon(tokens: Tensor, config: V39PolicyConfig) -> Tensor:
    """Pool rollout spatial tokens per anchor, then align anchors to action time."""

    if tokens.ndim != 3:
        raise ValueError(f"rollout tokens must be [B,F*G,H], got {tuple(tokens.shape)}")
    grid = int(config.num_cameras) * int(config.future_grid_size) * int(config.future_grid_size)
    expected = int(config.future_anchors) * grid
    if int(tokens.shape[1]) != expected:
        raise ValueError(f"rollout token count must be future_anchors*grid={expected}, got {tokens.shape[1]}")
    milestones = tokens.reshape(
        tokens.shape[0], int(config.future_anchors), grid, tokens.shape[-1]
    ).mean(dim=2)
    return _align_milestone_tokens_to_horizon(milestones, int(config.action_horizon))


class MidcutContractHeads(nn.Module):
    """Intentionally weak readouts from the DiT midpoint.

    The heads are deliberately no stronger than LayerNorm + Linear.  If these
    heads cannot read motion/event/future information, the information is not
    sufficiently explicit at the mid-cut latent.
    """

    def __init__(self, config: V39PolicyConfig) -> None:
        super().__init__()
        self.config = config
        h = int(config.hidden_size)
        self.action_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, config.physical_action_dim))
        self.event_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 3))
        self.motion_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 1))
        self.rollout_effect_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
        self.rollout_delta_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
        self.transition_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
        self.future_gain = nn.Parameter(torch.tensor(float(config.midcut_future_gain_init), dtype=torch.float32))
        # Start action/event readouts small but not exactly zero.  A fully
        # zero final Linear makes the first backward step update only the head
        # itself and gives essentially no gradient to the upstream latent.
        # Small random init keeps the head weak while allowing the contract
        # loss to shape the DiT canvas from the beginning.
        for module in (self.action_head[-1], self.event_head[-1], self.motion_head[-1]):
            nn.init.normal_(module.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(module.bias)
        for module in (self.rollout_effect_head[-1], self.rollout_delta_head[-1], self.transition_head[-1]):
            nn.init.normal_(module.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(module.bias)

    def trajectory_pooled(self, trajectory_tokens: Tensor) -> Tensor:
        cfg = self.config
        b = trajectory_tokens.shape[0]
        grouped = trajectory_tokens.reshape(b, cfg.action_horizon, cfg.action_basis_tokens, cfg.hidden_size)
        return grouped.mean(dim=2)

    def forward(self, canvas: Tensor, slices: dict[str, slice]) -> dict[str, Tensor]:
        cfg = self.config
        trajectory = canvas[:, slices["trajectory"]]
        rollout = canvas[:, slices["rollout"]]
        registers = canvas[:, slices["registers"]]
        trajectory_pooled = self.trajectory_pooled(trajectory)
        gain = self.future_gain.to(device=canvas.device, dtype=canvas.dtype)
        effect = self.rollout_effect_head(rollout) * gain
        delta = self.rollout_delta_head(rollout) * gain
        event_context = _rollout_tokens_to_action_horizon(delta, cfg)
        transition_base = delta.mean(dim=1, keepdim=True)
        transition = self.transition_head(transition_base).expand(-1, cfg.action_horizon, -1)
        return {
            "midcut_canvas_tokens": canvas,
            "midcut_trajectory_tokens": trajectory,
            "midcut_rollout_tokens": rollout,
            "midcut_register_tokens": registers,
            "midcut_state_tokens": canvas[:, slices["state"]],
            "midcut_state_history_tokens": canvas[:, slices["state_history"]],
            "midcut_executed_tokens": canvas[:, slices["executed"]],
            "midcut_proposal_tokens": canvas[:, slices["proposal"]],
            "midcut_trajectory_pooled": trajectory_pooled,
            "midcut_pred_physical_velocity": self.action_head(trajectory_pooled),
            "midcut_direct_physical_velocity": self.action_head(trajectory_pooled),
            "midcut_rollout_residual_velocity": torch.zeros(
                trajectory_pooled.shape[0], cfg.action_horizon, cfg.physical_action_dim,
                device=trajectory_pooled.device, dtype=trajectory_pooled.dtype,
            ),
            "midcut_rollout_alpha": torch.zeros(1, cfg.action_horizon, 1, device=trajectory_pooled.device, dtype=trajectory_pooled.dtype),
            "midcut_rollout_effect_pred": effect,
            "midcut_rollout_delta_pred": delta,
            "midcut_rollout_base_effect_pred": torch.zeros_like(effect),
            "midcut_event_logits": self.event_head(event_context),
            "midcut_motion_logits": self.motion_head(trajectory_pooled).squeeze(-1),
            "midcut_transition_latent": transition,
            "midcut_rollout_delta_norm": delta.detach().float().norm(dim=-1).mean(),
            "midcut_rollout_effect_norm": effect.detach().float().norm(dim=-1).mean(),
            "midcut_future_gain": gain.detach().float().abs(),
        }


class LayerContractAdapterHeads(nn.Module):
    """Tiny per-layer adapter contract for V39.1.

    It first applies a small bottleneck residual adapter, then reuses the same
    deliberately weak readout family as the mid-cut contract.  The adapter keeps
    the probe local and cheap; the heads stay too weak to manufacture motion or
    contact structure after the trunk.
    """

    def __init__(self, config: V39PolicyConfig, *, layer_index: int) -> None:
        super().__init__()
        self.config = config
        self.layer_index = int(layer_index)
        h = int(config.hidden_size)
        b = int(config.layer_contract_adapter_dim)
        self.adapter = nn.Sequential(
            nn.LayerNorm(h),
            nn.Linear(h, b),
            nn.GELU(),
            nn.Linear(b, h),
        )
        nn.init.normal_(self.adapter[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.adapter[-1].bias)
        self.readout = MidcutContractHeads(config)

    def forward(self, canvas: Tensor, slices: dict[str, slice]) -> dict[str, Tensor]:
        scale = torch.as_tensor(
            float(self.config.layer_contract_residual_scale),
            device=canvas.device,
            dtype=canvas.dtype,
        )
        adapted = canvas + scale * self.adapter(canvas)
        mid = self.readout(adapted, slices)
        out: dict[str, Tensor] = {
            key[len("midcut_"):]: value for key, value in mid.items() if key.startswith("midcut_")
        }
        out["layer_index"] = torch.as_tensor(self.layer_index, device=canvas.device, dtype=torch.long)
        return out


class SharedLayerFlowActionProbe(nn.Module):
    """Shared lightweight flow-matching action probe for V39.2.

    Each per-layer adapter first predicts a world/future latent.  This probe then
    reads only the layer-local latent summaries plus the current noisy physical
    action and flow time.  The parameters are shared across layers so lower loss
    identifies a better latent layer rather than a stronger per-layer action
    decoder.
    """

    def __init__(self, config: V39PolicyConfig) -> None:
        super().__init__()
        self.config = config
        h = int(config.hidden_size)
        ph = int(config.physical_action_dim)
        mid = int(config.layer_fm_probe_hidden)
        self.noisy_proj = nn.Linear(ph, h)
        self.latent_proj = nn.Sequential(nn.LayerNorm(2 * h), nn.Linear(2 * h, h), nn.SiLU(), nn.Linear(h, h))
        self.time = TimeEmbedding(h)
        self.net = nn.Sequential(
            nn.LayerNorm(h),
            nn.Linear(h, mid),
            nn.SiLU(),
            nn.Linear(mid, ph),
        )
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        *,
        trajectory_pooled: Tensor,
        rollout_effect_pred: Tensor,
        rollout_delta_pred: Tensor,
        noisy_physical: Tensor,
        time: Tensor,
    ) -> Tensor:
        if noisy_physical.shape[:2] != trajectory_pooled.shape[:2]:
            raise ValueError(
                f"noisy_physical and trajectory_pooled horizon mismatch: "
                f"{tuple(noisy_physical.shape)} vs {tuple(trajectory_pooled.shape)}"
            )
        latent_summary = torch.cat(
            [rollout_effect_pred.mean(dim=1), rollout_delta_pred.mean(dim=1)],
            dim=-1,
        )
        latent_bias = self.latent_proj(latent_summary).to(dtype=trajectory_pooled.dtype)[:, None, :]
        t = self.time(time.to(dtype=trajectory_pooled.dtype)).to(dtype=trajectory_pooled.dtype)[:, None, :]
        x = self.noisy_proj(noisy_physical.to(dtype=trajectory_pooled.dtype)) + trajectory_pooled + latent_bias + t
        return self.net(x)


class LayerRoleScheduler(nn.Module):
    """Deterministic layer-role schedule for V40 latent/causal contracts.

    Lower layers are expected to expose action-sensitive local transition deltas;
    upper layers are expected to expose stable world/future latents.  The schedule
    returns scalar gains used both for prediction mixing and for diagnostics.
    """

    def __init__(self, config: V39PolicyConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, layer_index: int | Tensor, *, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        count = max(int(self.config.depth) - 1, 1)
        if torch.is_tensor(layer_index):
            idx = layer_index.to(device=device, dtype=dtype)
        else:
            idx = torch.as_tensor(float(layer_index), device=device, dtype=dtype)
        progress = (idx / float(count)).clamp(0.0, 1.0)
        c_low = float(self.config.layer_low_causal_weight)
        c_high = float(self.config.layer_high_causal_weight)
        l_low = float(self.config.layer_low_latent_weight)
        l_high = float(self.config.layer_high_latent_weight)
        causal = c_low + (c_high - c_low) * progress
        latent = l_low + (l_high - l_low) * progress
        return causal, latent


class UnifiedInterventionBlock(nn.Module):
    """One light state-action interaction block for V40.1.

    The block is deliberately not a second DiT.  It performs one cross-attention
    step from grid-local intervention state into compact context tokens, followed
    by a small FFN.  Setting ``layer_causal_feedback_depth=0`` bypasses these
    blocks and leaves the FiLM-gated delta path as the main transition operator.
    """

    def __init__(self, hidden: int, heads: int, mid: int) -> None:
        super().__init__()
        self.qn = nn.LayerNorm(hidden)
        self.kn = nn.LayerNorm(hidden)
        self.cross = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.fn = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(nn.Linear(hidden, mid), nn.SiLU(), nn.Linear(mid, hidden))
        nn.init.normal_(self.ffn[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.ffn[-1].bias)

    def forward(self, state: Tensor, context: Tensor) -> Tensor:
        update, _ = self.cross(self.qn(state), self.kn(context), self.kn(context), need_weights=False)
        state = state + update
        state = state + self.ffn(self.fn(state)).to(dtype=state.dtype)
        return state


class RecurrentMilestoneConsequenceCell(nn.Module):
    """V40.1 unified intervention-latent encoder.

    Public name is preserved for checkpoint/CLI compatibility, but the object is
    no longer a separate action-only consequence head.  It is a single
    intervention-latent head that jointly encodes:

    * layer-local rollout/world tokens;
    * current state token and state-history tokens;
    * executed-action history tokens;
    * optional trajectory/proposal canvas tokens;
    * candidate future action segments.

    It emits an action-conditioned residual latent.  The residual is supervised
    by future-latent targets, while action and state counterfactual views test
    whether the same unified head really depends on both the intervention and
    the originating state/frame context.
    """

    def __init__(self, config: V39PolicyConfig) -> None:
        super().__init__()
        self.config = config
        h = int(config.hidden_size)
        ph = int(config.physical_action_dim)
        mid = int(config.layer_consequence_hidden)
        self.gripper_frame = (
            ParsevalGripperTemporalFrame(config.action_horizon, config.gripper_field_dim)
            if str(getattr(config, "gripper_field_mode", "legacy_handcrafted")) == "parseval_temporal"
            else None
        )
        semantic_ph = 2 * int(config.arm_dim) + 1 if self.gripper_frame is not None else ph
        self.action_summary_dim = semantic_ph * 5 + 4
        self.action_encoder = nn.Sequential(
            nn.LayerNorm(self.action_summary_dim),
            nn.Linear(self.action_summary_dim, mid),
            nn.SiLU(),
            nn.Linear(mid, h),
        )
        self.step_embed = nn.Embedding(int(config.layer_consequence_steps), h)
        self.layer_embed = nn.Embedding(int(config.depth), h)
        self.memory_tokens = nn.Parameter(torch.randn(1, int(config.layer_causal_memory_tokens), h) * 0.02)
        self.context_proj = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
        self.action_film = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, mid), nn.SiLU(), nn.Linear(mid, 2 * h))
        self.context_gate = nn.Sequential(nn.LayerNorm(2 * h), nn.Linear(2 * h, mid), nn.SiLU(), nn.Linear(mid, 1))
        self.delta_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, mid), nn.SiLU(), nn.Linear(mid, h))
        self.neutral_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, mid), nn.SiLU(), nn.Linear(mid, h))
        self.policy_effect_proj = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, mid), nn.SiLU(), nn.Linear(mid, h))
        self.interaction_blocks = nn.ModuleList([
            UnifiedInterventionBlock(h, int(config.num_heads), mid)
            for _ in range(int(config.layer_causal_feedback_depth))
        ])
        self.effect_norm = nn.LayerNorm(h)
        self.effect_gain = nn.Parameter(torch.tensor(float(config.layer_consequence_initial_gain), dtype=torch.float32))
        self.delta_scale = nn.Parameter(torch.tensor(float(config.layer_consequence_delta_scale), dtype=torch.float32))
        for module in (
            self.action_encoder[-1], self.context_proj[-1], self.action_film[-1],
            self.context_gate[-1], self.delta_head[-1], self.neutral_head[-1], self.policy_effect_proj[-1],
        ):
            nn.init.normal_(module.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(module.bias)

    def _segment_action(self, action_physical: Tensor) -> Tensor:
        cfg = self.config
        k = int(cfg.layer_consequence_steps)
        if self.gripper_frame is not None:
            ad = int(cfg.arm_dim)
            gripper_field = action_physical[..., 2 * ad :]
            action_physical = torch.cat(
                [action_physical[..., : 2 * ad], self.gripper_frame.synthesis(gripper_field)],
                dim=-1,
            )
        b, horizon, ph = action_physical.shape
        if horizon <= 0:
            raise ValueError("action_physical horizon must be positive")
        rows: list[Tensor] = []
        for step in range(k):
            lo = int(round(step * horizon / float(k)))
            hi = int(round((step + 1) * horizon / float(k)))
            hi = max(hi, lo + 1)
            hi = min(hi, horizon)
            seg = action_physical[:, lo:hi]
            mean = seg.mean(dim=1)
            first = seg[:, 0]
            last = seg[:, -1]
            delta = last - first
            std = seg.float().std(dim=1, unbiased=False).to(dtype=action_physical.dtype)
            ad = int(getattr(cfg, "arm_dim", max((ph - 2) // 2, 0)))
            if ad > 0 and 2 * ad + 2 == ph:
                # action_physical is [arm_abs, arm_delta, gripper_value, gripper_delta].
                grip_value = 2 * ad
                grip_mean = seg[..., grip_value].mean(dim=1, keepdim=True)
                grip_delta = last[:, grip_value:grip_value + 1] - first[:, grip_value:grip_value + 1]
                arm = seg[..., : 2 * ad]
            else:
                g = int(cfg.gripper_dim_index)
                if g < 0:
                    g += ph
                g = min(max(g, 0), ph - 1)
                grip_mean = seg[..., g].mean(dim=1, keepdim=True)
                grip_delta = last[:, g:g + 1] - first[:, g:g + 1]
                arm = torch.cat([seg[..., :g], seg[..., g + 1:]], dim=-1) if ph > 1 else seg[..., :0]
            arm_norm = arm.float().norm(dim=-1).mean(dim=1, keepdim=True).to(dtype=action_physical.dtype) if arm.numel() else torch.zeros(b, 1, device=action_physical.device, dtype=action_physical.dtype)
            action_norm = seg.float().norm(dim=-1).mean(dim=1, keepdim=True).to(dtype=action_physical.dtype)
            rows.append(torch.cat([mean, first, last, delta, std, grip_mean, grip_delta, arm_norm, action_norm], dim=-1))
        return torch.stack(rows, dim=1)

    def _compact_tokens(self, x: Tensor | None, *, max_tokens: int = 8) -> Tensor | None:
        if x is None:
            return None
        if x.ndim != 3:
            raise ValueError(f"context tokens must be [B,N,H], got {tuple(x.shape)}")
        if x.shape[1] <= max_tokens:
            return x
        # Uniform deterministic subsampling keeps the head lightweight while
        # still excluding more than a single frame/state token in counterfactuals.
        idx = torch.linspace(0, x.shape[1] - 1, steps=max_tokens, device=x.device).round().long()
        return x.index_select(1, idx)

    def _context_bank(
        self,
        *,
        base_tokens: Tensor,
        state_tokens: Tensor | None,
        state_history_tokens: Tensor | None,
        executed_tokens: Tensor | None,
        trajectory_tokens: Tensor | None,
        proposal_tokens: Tensor | None,
        action_token: Tensor,
        layer_token: Tensor,
    ) -> tuple[Tensor, Tensor]:
        b = base_tokens.shape[0]
        mem = self.memory_tokens.to(device=base_tokens.device, dtype=base_tokens.dtype).expand(b, -1, -1)
        parts = [
            base_tokens,
            self._compact_tokens(state_tokens, max_tokens=2),
            self._compact_tokens(state_history_tokens, max_tokens=4),
            self._compact_tokens(executed_tokens, max_tokens=4),
            self._compact_tokens(proposal_tokens, max_tokens=4),
            self._compact_tokens(trajectory_tokens, max_tokens=8),
            action_token[:, None, :],
            layer_token[:, None, :],
            mem,
        ]
        kept = [p for p in parts if p is not None]
        bank = self.context_proj(torch.cat(kept, dim=1)).to(dtype=base_tokens.dtype)
        # Pool each semantic group before averaging groups.  This prevents the
        # spatial rollout grid from numerically overwhelming the much shorter
        # state/history groups and keeps explicit context active even when the
        # optional cross-attention feedback depth is zero.
        grouped = torch.stack([part.mean(dim=1) for part in kept], dim=1)
        summary = self.context_proj(grouped).mean(dim=1).to(dtype=base_tokens.dtype)
        return bank, summary

    @staticmethod
    def _align_milestone_tokens_to_horizon(tokens: Tensor, horizon: int) -> Tensor:
        return _align_milestone_tokens_to_horizon(tokens, horizon)

    def forward(
        self,
        *,
        rollout_tokens: Tensor,
        action_physical: Tensor,
        state_tokens: Tensor | None = None,
        state_history_tokens: Tensor | None = None,
        executed_tokens: Tensor | None = None,
        trajectory_tokens: Tensor | None = None,
        proposal_tokens: Tensor | None = None,
        layer_index: int | Tensor | None = None,
    ) -> dict[str, Tensor]:
        cfg = self.config
        b = int(rollout_tokens.shape[0])
        k = int(cfg.layer_consequence_steps)
        grid = int(cfg.num_cameras) * int(cfg.future_grid_size) * int(cfg.future_grid_size)
        h = int(cfg.hidden_size)
        if rollout_tokens.shape[1] != int(cfg.future_token_count):
            raise ValueError(
                f"rollout_tokens must have future_token_count={cfg.future_token_count}, got {rollout_tokens.shape[1]}"
            )
        grouped = rollout_tokens.reshape(b, int(cfg.future_anchors), grid, h)
        action_segments = self._segment_action(action_physical.to(device=rollout_tokens.device, dtype=rollout_tokens.dtype))
        action_embed = self.action_encoder(action_segments).to(dtype=rollout_tokens.dtype)
        step_ids = torch.arange(k, device=rollout_tokens.device)
        step_embed = self.step_embed(step_ids).to(dtype=rollout_tokens.dtype)
        if layer_index is None:
            layer_id = torch.zeros((), device=rollout_tokens.device, dtype=torch.long)
        elif torch.is_tensor(layer_index):
            layer_id = layer_index.to(device=rollout_tokens.device, dtype=torch.long).clamp(0, int(cfg.depth) - 1)
        else:
            layer_id = torch.as_tensor(int(layer_index), device=rollout_tokens.device, dtype=torch.long).clamp(0, int(cfg.depth) - 1)
        layer_token = self.layer_embed(layer_id)[None].expand(b, -1).to(dtype=rollout_tokens.dtype)
        scale = self.delta_scale.to(device=rollout_tokens.device, dtype=rollout_tokens.dtype).abs()
        gain = self.effect_gain.to(device=rollout_tokens.device, dtype=rollout_tokens.dtype).abs()
        effect_state = torch.zeros(b, grid, h, device=rollout_tokens.device, dtype=rollout_tokens.dtype)
        preds: list[Tensor] = []
        deltas: list[Tensor] = []
        gates: list[Tensor] = []
        policy_tokens: list[Tensor] = []
        neutral_tokens: list[Tensor] = []
        intervene_tokens: list[Tensor] = []
        for step in range(k):
            # Validation requires one intervention step per future anchor, so
            # predictions and targets share the same temporal indexing.
            anchor = step
            base = grouped[:, anchor]
            a = action_embed[:, step] + step_embed[step][None] + layer_token
            context, context_summary = self._context_bank(
                base_tokens=base,
                state_tokens=state_tokens,
                state_history_tokens=state_history_tokens,
                executed_tokens=executed_tokens,
                trajectory_tokens=trajectory_tokens,
                proposal_tokens=proposal_tokens,
                action_token=a,
                layer_token=layer_token,
            )
            neutral = base + self.neutral_head(base).to(dtype=rollout_tokens.dtype)
            intervention = neutral + effect_state
            for block in self.interaction_blocks:
                intervention = block(intervention, context)
            joint_condition = a + context_summary
            gamma_beta = self.action_film(joint_condition).to(dtype=rollout_tokens.dtype)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            modulated = intervention * (1.0 + gamma[:, None, :]) + beta[:, None, :]
            gate_in = torch.cat([modulated, joint_condition[:, None, :].expand(-1, grid, -1)], dim=-1)
            gate = torch.sigmoid(self.context_gate(gate_in).to(dtype=rollout_tokens.dtype))
            raw_delta = torch.tanh(self.delta_head(modulated).to(dtype=rollout_tokens.dtype))
            # V40.1 keeps the local/cumulative contract closed, but restores the
            # normalized increment used by the earlier K4/A6 branch.  The
            # unnormalized gated delta is often too small for action-shuffle
            # contrast to see; LayerNorm provides a per-token direction
            # amplifier.  Crucially, the *same* increment is logged/supervised as
            # milestone_step_delta_pred and accumulated into rollout_effect_pred,
            # so delta matching and cumulative rollout remain mathematically
            # consistent.
            local_delta = scale * gate * raw_delta
            step_delta = gain * self.effect_norm(local_delta).to(dtype=rollout_tokens.dtype)
            effect_state = effect_state + step_delta
            z_intervene = neutral + effect_state
            preds.append(effect_state)
            deltas.append(step_delta)
            gates.append(gate)
            policy_tokens.append(self.policy_effect_proj(z_intervene).to(dtype=rollout_tokens.dtype))
            neutral_tokens.append(neutral)
            intervene_tokens.append(z_intervene)
        pred = torch.stack(preds, dim=1)
        delta_stack = torch.stack(deltas, dim=1)
        gate_stack = torch.stack(gates, dim=1)
        policy_stack = torch.stack(policy_tokens, dim=1)
        neutral_stack = torch.stack(neutral_tokens, dim=1)
        intervene_stack = torch.stack(intervene_tokens, dim=1)
        flat_pred = pred.reshape(b, k * grid, h)
        flat_delta = delta_stack.reshape(b, k * grid, h)
        flat_policy = policy_stack.reshape(b, k * grid, h)
        time_policy = self._align_milestone_tokens_to_horizon(
            policy_stack.mean(dim=2), int(cfg.action_horizon)
        )
        return {
            "milestone_rollout_effect_pred": flat_pred,
            "milestone_rollout_delta_pred": flat_pred,
            "milestone_step_delta_pred": flat_delta,
            "milestone_policy_effect_tokens": flat_policy,
            "milestone_policy_time_tokens": time_policy,
            "milestone_neutral_latent_pred": neutral_stack.reshape(b, k * grid, h),
            "milestone_intervention_latent_pred": intervene_stack.reshape(b, k * grid, h),
            "milestone_gate_mean": gate_stack.detach().float().mean(),
            "milestone_step_delta_norm": delta_stack.detach().float().norm(dim=-1).mean(),
            "milestone_effect_norm": pred.detach().float().norm(dim=-1).mean(),
            "milestone_effect_std": pred.detach().float().std(unbiased=False),
            "milestone_effect_gain": gain.detach().float().abs(),
        }


def _zeros_like_scalar(reference: Tensor) -> Tensor:
    return torch.zeros((), device=reference.device, dtype=reference.dtype)

class TemporalMidcutWorldActionDiT(nn.Module):
    """V38 DiT split into a mid-cut contract trunk and a policy tail."""

    def __init__(self, config: V39PolicyConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        h = int(config.hidden_size)
        self.visual_memory = DenseVisualMemory(config)
        self.rollout_codec = RolloutTargetCodec(config)
        self.seed = UnifiedCanvasSeed(config)
        self.time = TimeEmbedding(h)
        self.content_mod = nn.Sequential(nn.LayerNorm(2 * h), nn.Linear(2 * h, h), nn.SiLU(), nn.Linear(h, h))
        nn.init.normal_(self.content_mod[-1].weight, mean=0.0, std=2e-2)
        nn.init.zeros_(self.content_mod[-1].bias)
        self.content_mod_scale = nn.Parameter(torch.tensor(0.10))
        self.blocks = nn.ModuleList([TemporalDynamicsBoundDiTBlock(config) for _ in range(config.depth)])
        self.midcut_norm = nn.LayerNorm(h)
        self.midcut_heads = MidcutContractHeads(config)
        if int(config.layer_contract_adapters):
            self.layer_contract_heads = nn.ModuleList([
                LayerContractAdapterHeads(config, layer_index=i) for i in range(int(config.depth))
            ])
        else:
            self.layer_contract_heads = nn.ModuleList()
        self.layer_fm_probe = SharedLayerFlowActionProbe(config) if int(config.layer_shared_fm_probe) else None
        self.layer_role_scheduler = LayerRoleScheduler(config)
        self.layer_consequence_cell = RecurrentMilestoneConsequenceCell(config) if int(config.layer_recurrent_consequence) else None
        self.final_norm = nn.LayerNorm(h)
        self.direct_physical_head = CanvasPhysicalVelocityHead(config)
        self.rollout_residual_head = RolloutActionResidualHead(config)
        self.controlled_dynamics = ControlledResidualLatentDynamics(config)
        self.event_probe = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 3))
        self.motion_probe = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 1))
        final_decoder = str(getattr(config, "final_action_decoder", "legacy"))
        self.hierarchical_mmdit_action_decoder: HierarchicalMMDiTActionDecoder | None = None
        if final_decoder == "residual_action_flow":
            self.residual_action_flow_denoiser = V37StyleResidualActionFlowDenoiser(config)
            self.latent_main_action_decoder = None
            self.latent_cvae_action_decoder = None
        elif final_decoder == "layered_residual_action_flow":
            self.residual_action_flow_denoiser = LayeredV37StyleResidualActionFlowDenoiser(config)
            self.latent_main_action_decoder = None
            self.latent_cvae_action_decoder = None
        elif final_decoder == "latent_main_action":
            self.residual_action_flow_denoiser = None
            self.latent_main_action_decoder = HierarchicalLatentMainActionDecoder(config)
            self.latent_cvae_action_decoder = None
        elif final_decoder == "latent_cvae_action":
            self.residual_action_flow_denoiser = None
            self.latent_main_action_decoder = None
            self.latent_cvae_action_decoder = LatentCVAEActionDecoder(config)
        elif final_decoder == "adaptive_recurrent_cvae_action":
            self.residual_action_flow_denoiser = None
            self.latent_main_action_decoder = None
            self.latent_cvae_action_decoder = AdaptiveRecurrentCVAEActionDecoder(config)
        elif final_decoder == "hierarchical_mmdit_action":
            self.residual_action_flow_denoiser = None
            self.latent_main_action_decoder = None
            self.latent_cvae_action_decoder = None
            self.hierarchical_mmdit_action_decoder = HierarchicalMMDiTActionDecoder(config)
        else:
            self.residual_action_flow_denoiser = None
            self.latent_main_action_decoder = None
            self.latent_cvae_action_decoder = None
        if (
            self.latent_cvae_action_decoder is not None
            or self.latent_main_action_decoder is not None
            or self.hierarchical_mmdit_action_decoder is not None
        ):
            # These readers belong to the legacy action tower. Keep the modules
            # for checkpoint compatibility and the parameter-free pooled()
            # helper, but do not allocate gradients/optimizer state for outputs
            # that the complete latent decoder never consumes.
            self.direct_physical_head.requires_grad_(False)
            self.rollout_residual_head.requires_grad_(False)
            self.motion_probe.requires_grad_(False)

    def _mod_embed(self, canvas: Tensor, visual_memory: Tensor, time_emb: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        summary = torch.cat([canvas.mean(dim=1), visual_memory.mean(dim=1)], dim=-1)
        content_delta = self.content_mod(summary) * self.content_mod_scale.to(device=canvas.device, dtype=canvas.dtype)
        return time_emb + content_delta, content_delta, time_emb

    def _promote_midcut(self, mid: dict[str, Tensor], *, gates: dict[str, Tensor], content_norm: Tensor, time_norm: Tensor) -> dict[str, Tensor]:
        pred = mid["midcut_pred_physical_velocity"]
        effect = mid["midcut_rollout_effect_pred"]
        delta = mid["midcut_rollout_delta_pred"]
        z = _zeros_like_scalar(pred)
        out = {
            **mid,
            "canvas_tokens": mid["midcut_canvas_tokens"],
            "trajectory_tokens": mid["midcut_trajectory_tokens"],
            "rollout_tokens": mid["midcut_rollout_tokens"],
            "register_tokens": mid["midcut_register_tokens"],
            "direct_physical_velocity": mid["midcut_direct_physical_velocity"],
            "rollout_residual_velocity": mid["midcut_rollout_residual_velocity"],
            "rollout_alpha": mid["midcut_rollout_alpha"],
            "pred_physical_velocity": pred,
            "rollout_effect_pred": effect,
            "rollout_base_effect_pred": mid["midcut_rollout_base_effect_pred"],
            "rollout_delta_pred": delta,
            "future_latent_pred": effect,
            "action_effect_pred": effect,
            "event_logits": mid["midcut_event_logits"],
            "motion_logits": mid["midcut_motion_logits"],
            "transition_latent": mid["midcut_transition_latent"],
            "rollout_coeff_abs_mean": z,
            "rollout_neutral_coeff_abs_mean": z,
            "rollout_centered_coeff_abs_mean": z,
            "rollout_basis_norm": z,
            "rollout_delta_norm": mid["midcut_rollout_delta_norm"],
            "rollout_base_norm": z,
            "rollout_delta_gain": mid["midcut_future_gain"],
            "gate_self": gates.get("gate_self", z),
            "gate_visual": gates.get("gate_visual", z),
            "gate_rollout": gates.get("gate_rollout", z),
            "gate_ffn": gates.get("gate_ffn", z),
            "mod_content_norm": content_norm,
            "mod_time_norm": time_norm,
            "mod_content_to_time": content_norm / time_norm.clamp_min(1e-6),
            "midcut_stop": torch.ones((), device=pred.device, dtype=pred.dtype),
        }
        return out

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
        *,
        stop_at_midcut: bool = False,
        consequence_physical: Tensor | None = None,
        cvae_target_physical: Tensor | None = None,
        enable_layer_contracts: bool = True,
        enable_final_action_decoder: bool = True,
    ) -> dict[str, Tensor]:
        cfg = self.config
        if proposal_keep is None:
            proposal_keep = torch.ones(noisy_physical.shape[0], device=noisy_physical.device, dtype=noisy_physical.dtype)
        if consequence_physical is None:
            consequence_physical = noisy_physical
        else:
            consequence_physical = consequence_physical.to(device=noisy_physical.device, dtype=noisy_physical.dtype)
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
        # Ownership snapshots are taken before any canvas self-attention.  The
        # final state/trajectory slices are contextual mixtures and can carry
        # noisy-action content, so using them as evidence recreates the exact
        # action -> evidence -> action echo this decoder is meant to remove.
        owned_state_memory = [
            canvas[:, slices["state"]],
            canvas[:, slices["state_history"]],
        ]
        owned_trajectory_memory = canvas[:, slices["proposal"]]
        owned_intent_memory = {
            "task": canvas[:, slices["task"]],
            "state": canvas[:, slices["state"]],
            "state_history": canvas[:, slices["state_history"]],
            "executed": canvas[:, slices["executed"]],
            "proposal": canvas[:, slices["proposal"]],
            "visual": canvas[:, slices["rollout"]].mean(dim=1, keepdim=True),
        }
        rollout_seed = canvas[:, slices["rollout"]].detach()
        time_emb = self.time(time.to(dtype=canvas.dtype))
        gate_rows: list[dict[str, Tensor]] = []
        content_norm_rows: list[Tensor] = []
        time_norm_rows: list[Tensor] = []
        midcut: dict[str, Tensor] | None = None
        layer_contracts: list[dict[str, Tensor]] = []
        # The latent-main decoder is the final action path, so inference/eval
        # must still materialize layer contracts even when callers disable
        # auxiliary contract evaluation for speed.  We do not add extra losses;
        # we only expose the latents needed by the action decoder.
        final_decoder = str(getattr(cfg, "final_action_decoder", "legacy"))
        force_layer_contracts = (
            final_decoder == "latent_main_action"
            or (final_decoder in {"latent_cvae_action", "adaptive_recurrent_cvae_action"} and bool(int(getattr(cfg, "latent_cvae_layer_memory", 1))))
            or final_decoder == "hierarchical_mmdit_action"
        )
        effective_layer_contracts = bool(enable_layer_contracts) or force_layer_contracts
        cut = int(cfg.midcut_layer)
        contract_grad_scale = float(getattr(cfg, "layer_contract_grad_scale", 1.0))
        for index, block in enumerate(self.blocks, start=1):
            mod_emb, content_delta, time_row = self._mod_embed(canvas, visual_memory, time_emb)
            content_norm_rows.append(content_delta.float().norm(dim=-1).mean())
            time_norm_rows.append(time_row.float().norm(dim=-1).mean())
            canvas, gates = block(canvas, visual_memory, mod_emb, slices)
            gate_rows.append(gates)
            if effective_layer_contracts and len(self.layer_contract_heads) > 0:
                contract_canvas = _scaled_contract_view(canvas, contract_grad_scale)
                layer_entry = self.layer_contract_heads[index - 1](contract_canvas, slices)
                if self.layer_consequence_cell is not None:
                    # V40: split the layer contract into an explicit world-latent
                    # object and an action-causal object.  Lower layers lean on
                    # the causal branch; upper layers lean on the latent branch.
                    # We keep the old direct outputs for forensics only.
                    latent_effect = layer_entry["rollout_effect_pred"]
                    latent_delta = layer_entry["rollout_delta_pred"]
                    cons = self.layer_consequence_cell(
                        rollout_tokens=layer_entry["rollout_tokens"],
                        action_physical=consequence_physical,
                        state_tokens=layer_entry.get("state_tokens"),
                        state_history_tokens=layer_entry.get("state_history_tokens"),
                        executed_tokens=layer_entry.get("executed_tokens"),
                        trajectory_tokens=layer_entry.get("trajectory_tokens"),
                        proposal_tokens=layer_entry.get("proposal_tokens"),
                        layer_index=index - 1,
                    )
                    causal_gain, latent_gain = self.layer_role_scheduler(
                        index - 1, device=latent_effect.device, dtype=latent_effect.dtype,
                    )
                    causal_effect = cons["milestone_rollout_effect_pred"]
                    causal_delta = cons["milestone_rollout_delta_pred"]
                    if latent_effect.shape[1] != causal_effect.shape[1]:
                        latent_effect_for_mix = latent_effect[:, : causal_effect.shape[1]]
                        latent_delta_for_mix = latent_delta[:, : causal_delta.shape[1]]
                    else:
                        latent_effect_for_mix = latent_effect
                        latent_delta_for_mix = latent_delta
                    layer_entry["latent_rollout_effect_pred"] = latent_effect
                    layer_entry["latent_rollout_delta_pred"] = latent_delta
                    layer_entry["causal_rollout_effect_pred"] = causal_effect
                    layer_entry["causal_rollout_delta_pred"] = causal_delta
                    layer_entry["direct_rollout_effect_pred"] = latent_effect
                    layer_entry["direct_rollout_delta_pred"] = latent_delta
                    # V40.1: one unified intervention-latent head is the
                    # supervised object.  The weak direct latent readout remains
                    # only for forensics; it is no longer mixed into the main
                    # rollout prediction where it can blur causal semantics.
                    layer_entry["rollout_effect_pred"] = causal_effect
                    layer_entry["rollout_delta_pred"] = causal_delta
                    layer_entry["policy_effect_tokens"] = cons["milestone_policy_effect_tokens"]
                    layer_entry["policy_effect_time_tokens"] = cons["milestone_policy_time_tokens"]
                    layer_entry["milestone_step_delta_pred"] = cons["milestone_step_delta_pred"]
                    layer_entry["unified_intervention_latent_pred"] = cons["milestone_intervention_latent_pred"]
                    layer_entry["neutral_latent_pred"] = cons["milestone_neutral_latent_pred"]
                    layer_entry["layer_causal_gain"] = causal_gain.detach().float()
                    layer_entry["layer_latent_gain"] = latent_gain.detach().float()
                    if bool(enable_layer_contracts) and int(getattr(cfg, "layer_zero_base_diagnostic", 0)):
                        # Loss-free shortcut probe.  If zeroing the rollout
                        # tokens barely moves the consequence output, the cell
                        # is probably relying on action features instead of the
                        # state/rollout context.
                        with torch.no_grad():
                            cons_zero = self.layer_consequence_cell(
                                rollout_tokens=torch.zeros_like(layer_entry["rollout_tokens"]),
                                action_physical=consequence_physical,
                                state_tokens=layer_entry.get("state_tokens"),
                                state_history_tokens=layer_entry.get("state_history_tokens"),
                                executed_tokens=layer_entry.get("executed_tokens"),
                                trajectory_tokens=layer_entry.get("trajectory_tokens"),
                                proposal_tokens=layer_entry.get("proposal_tokens"),
                                layer_index=index - 1,
                            )
                            base_eff = cons["milestone_rollout_effect_pred"].detach().float()
                            zero_eff = cons_zero["milestone_rollout_effect_pred"].float()
                            zero_shift = (
                                (base_eff - zero_eff).norm(dim=-1).mean()
                                / base_eff.norm(dim=-1).mean().clamp_min(1e-6)
                            )
                        layer_entry["consequence_zero_base_shift"] = zero_shift
                    if bool(enable_layer_contracts) and int(getattr(cfg, "layer_state_counterfactual", 0)) and int(layer_entry["rollout_tokens"].shape[0]) > 1:
                        flat_state = layer_entry["rollout_tokens"].detach().float().flatten(1)
                        dist_state = torch.cdist(flat_state, flat_state, p=2)
                        eye_state = torch.eye(dist_state.shape[0], device=dist_state.device, dtype=torch.bool)
                        dist_state = dist_state.masked_fill(eye_state, -1.0)
                        state_perm = dist_state.argmax(dim=1)
                        cons_state = self.layer_consequence_cell(
                            rollout_tokens=layer_entry["rollout_tokens"][state_perm],
                            action_physical=consequence_physical,
                            state_tokens=None if layer_entry.get("state_tokens") is None else layer_entry["state_tokens"][state_perm],
                            state_history_tokens=None if layer_entry.get("state_history_tokens") is None else layer_entry["state_history_tokens"][state_perm],
                            executed_tokens=None if layer_entry.get("executed_tokens") is None else layer_entry["executed_tokens"][state_perm],
                            trajectory_tokens=None if layer_entry.get("trajectory_tokens") is None else layer_entry["trajectory_tokens"][state_perm],
                            proposal_tokens=None if layer_entry.get("proposal_tokens") is None else layer_entry["proposal_tokens"][state_perm],
                            layer_index=index - 1,
                        )
                        layer_entry["rollout_effect_pred_shuffle_state"] = cons_state["milestone_rollout_effect_pred"]
                        layer_entry["rollout_delta_pred_shuffle_state"] = cons_state["milestone_rollout_delta_pred"]
                        layer_entry["milestone_step_delta_pred_shuffle_state"] = cons_state["milestone_step_delta_pred"]
                        layer_entry["policy_effect_tokens_shuffle_state"] = cons_state["milestone_policy_effect_tokens"]
                    if int(getattr(cfg, "layer_causal_event_from_effect", 1)):
                        event_src = cons["milestone_policy_time_tokens"]
                        layer_entry["event_logits"] = self.event_probe(event_src)
                    for key in ("milestone_gate_mean", "milestone_step_delta_norm", "milestone_effect_norm", "milestone_effect_std", "milestone_effect_gain"):
                        layer_entry[key] = cons[key]
                if self.layer_fm_probe is not None:
                    probe_velocity = self.layer_fm_probe(
                        trajectory_pooled=layer_entry["trajectory_pooled"],
                        rollout_effect_pred=layer_entry["rollout_effect_pred"],
                        rollout_delta_pred=layer_entry["rollout_delta_pred"],
                        noisy_physical=noisy_physical,
                        time=time,
                    )
                    # In V39.2/V39.3 the action-flow probe is downstream of
                    # the layer latent.  It replaces the per-layer direct
                    # action head for contract losses, while remaining shared
                    # across all layers.
                    layer_entry["pred_physical_velocity"] = probe_velocity
                    layer_entry["direct_physical_velocity"] = probe_velocity
                    layer_entry["layer_fm_probe_velocity"] = probe_velocity
                layer_contracts.append(layer_entry)
            if index == cut:
                mid_canvas = self.midcut_norm(canvas)
                midcut = self.midcut_heads(mid_canvas, slices)
                if stop_at_midcut:
                    content_norm = torch.stack(content_norm_rows).mean() if content_norm_rows else _zeros_like_scalar(canvas)
                    time_norm = torch.stack(time_norm_rows).mean() if time_norm_rows else _zeros_like_scalar(canvas)
                    gate_mean = {
                        key: torch.stack([row[key] for row in gate_rows]).mean()
                        for key in ("gate_self", "gate_visual", "gate_rollout", "gate_ffn")
                    }
                    promoted = self._promote_midcut(midcut, gates=gate_mean, content_norm=content_norm, time_norm=time_norm)
                    if layer_contracts:
                        promoted["layer_contracts"] = layer_contracts
                    return promoted
        if midcut is None:
            # Defensive fallback; validate() should prevent this.
            midcut = self.midcut_heads(self.midcut_norm(canvas), slices)
        canvas = self.final_norm(canvas)
        trajectory = canvas[:, slices["trajectory"]]
        rollout = canvas[:, slices["rollout"]]
        registers = canvas[:, slices["registers"]]
        trajectory_pooled = self.direct_physical_head.pooled(trajectory)
        context_kv = torch.cat([
            canvas[:, slices["state"]],
            canvas[:, slices["state_history"]],
            canvas[:, slices["executed"]],
            canvas[:, slices["proposal"]],
        ], dim=1)
        if str(getattr(cfg, "controlled_base_mode", "learned")) == "fixed_zero":
            dynamics = self.controlled_dynamics(
                rollout_init.to(device=rollout.device, dtype=rollout.dtype),
                context_kv,
                action_tokens=trajectory,
                transition_tokens=rollout,
            )
        else:
            # Preserve the exact learned-base path for historical checkpoints.
            dynamics = self.controlled_dynamics(
                rollout,
                context_kv,
                action_tokens=trajectory,
            )
        controlled_delta = dynamics["rollout_delta_pred"]
        rollout_effect_pred = dynamics["rollout_effect_pred"]
        event_context = _rollout_tokens_to_action_horizon(controlled_delta, cfg)
        decoder_mode = str(getattr(cfg, "final_action_decoder", "legacy"))
        direct_velocity: Tensor | None = None
        rollout_residual_velocity: Tensor | None = None
        rollout_alpha: Tensor | None = None
        legacy_velocity: Tensor | None = None
        pred_physical_velocity: Tensor
        legacy_event_logits: Tensor
        legacy_motion_logits: Tensor
        residual_action_flow: dict[str, Tensor] | None = None
        latent_main_action: dict[str, Tensor] | None = None
        latent_cvae_action: dict[str, Tensor] | None = None
        hierarchical_mmdit_action: dict[str, Tensor] | None = None
        if not enable_final_action_decoder:
            # Counterfactual rollout branches consume only dynamics and layer
            # contracts. Running the final CVAE/MMDiT tower here duplicated a
            # full prior decode whose action output was immediately discarded.
            pred_physical_velocity = torch.zeros_like(noisy_physical)
            legacy_event_logits = event_context.new_zeros(
                int(event_context.shape[0]), int(event_context.shape[1]), 3
            )
            legacy_motion_logits = event_context.new_zeros(
                int(event_context.shape[0]), int(event_context.shape[1])
            )
        elif self.hierarchical_mmdit_action_decoder is not None:
            if str(getattr(cfg, "controlled_base_mode", "learned")) == "fixed_zero":
                transition_memory = [controlled_delta]
            else:
                transition_memory = [controlled_delta, rollout_effect_pred]
            event_evidence = None
            if layer_contracts:
                candidate = layer_contracts[-1].get("event_logits")
                if isinstance(candidate, Tensor) and candidate.ndim == 3 and int(candidate.shape[-1]) == 3:
                    event_evidence = candidate
            if event_evidence is None:
                event_evidence = self.event_probe(event_context)
            hierarchical_mmdit_action = self.hierarchical_mmdit_action_decoder(
                noisy_physical=noisy_physical,
                time=time,
                trajectory_tokens=owned_trajectory_memory,
                trajectory_workspace_tokens=owned_trajectory_memory,
                rollout_tokens=rollout,
                transition_memory=transition_memory,
                event_evidence=event_evidence,
                state_memory=owned_state_memory,
                intent_memory=owned_intent_memory,
                layer_contracts=layer_contracts,
            )
            pred_physical_velocity = hierarchical_mmdit_action["pred_velocity"]
            legacy_event_logits = hierarchical_mmdit_action["event_logits"]
            legacy_motion_logits = hierarchical_mmdit_action["motion_logits"]
        elif self.latent_cvae_action_decoder is not None:
            context_memory = [
                canvas[:, slices["state"]],
                canvas[:, slices["state_history"]],
                canvas[:, slices["executed"]],
                canvas[:, slices["proposal"]],
            ] if int(getattr(cfg, "latent_cvae_context_memory", 0)) else None
            # Rollout has its own full-resolution workspace source. Transition
            # memory therefore carries only explicit consequence semantics and
            # does not duplicate the same rollout grid through a pooled path.
            if int(getattr(cfg, "latent_cvae_transition_memory", 1)):
                if str(getattr(cfg, "controlled_base_mode", "learned")) == "fixed_zero":
                    # effect == delta under a fixed-zero base. Feeding both would
                    # duplicate one condition under two semantic names.
                    transition_memory = [controlled_delta, event_context]
                else:
                    transition_memory = [controlled_delta, rollout_effect_pred, event_context]
            else:
                transition_memory = None
            latent_cvae_action = self.latent_cvae_action_decoder(
                noisy_physical=noisy_physical,
                time=time,
                trajectory_tokens=trajectory_pooled,
                trajectory_workspace_tokens=trajectory,
                rollout_tokens=rollout,
                context_memory=context_memory,
                transition_memory=transition_memory,
                visual_memory=visual_memory if int(getattr(cfg, "latent_cvae_visual_memory", 0)) else None,
                layer_contracts=layer_contracts,
                target_physical=cvae_target_physical,
            )
            pred_physical_velocity = latent_cvae_action["pred_velocity"]
            legacy_event_logits = latent_cvae_action["event_logits"]
            legacy_motion_logits = latent_cvae_action["motion_logits"]
        elif self.latent_main_action_decoder is not None:
            context_memory = context_kv if int(getattr(cfg, "latent_action_context_memory", 0)) else None
            transition_parts = [rollout, controlled_delta, event_context]
            if str(getattr(cfg, "controlled_base_mode", "learned")) != "fixed_zero":
                transition_parts.insert(2, rollout_effect_pred)
            transition_memory = torch.cat(transition_parts, dim=1) if int(getattr(cfg, "latent_action_transition_memory", 1)) else None
            latent_main_action = self.latent_main_action_decoder(
                noisy_physical=noisy_physical,
                time=time,
                trajectory_tokens=trajectory_pooled,
                context_memory=context_memory,
                transition_memory=transition_memory,
                visual_memory=visual_memory if int(getattr(cfg, "latent_action_visual_memory", 0)) else None,
                layer_contracts=layer_contracts,
            )
            pred_physical_velocity = latent_main_action["pred_velocity"]
            legacy_event_logits = latent_main_action["event_logits"]
            legacy_motion_logits = latent_main_action["motion_logits"]
        else:
            # Legacy action readers are needed only by legacy/residual decoder
            # modes. CVAE/MMDiT is a complete final path, so computing a second
            # rollout-to-action tower there wastes memory and creates misleading
            # anchor diagnostics for a path that deployment never uses.
            direct_velocity = self.direct_physical_head(trajectory)
            rollout_residual_velocity, rollout_alpha = self.rollout_residual_head(trajectory_pooled, controlled_delta)
            legacy_velocity = direct_velocity + rollout_residual_velocity
            pred_physical_velocity = legacy_velocity
            legacy_event_logits = self.event_probe(event_context)
            legacy_motion_logits = self.motion_probe(trajectory_pooled.detach()).squeeze(-1)
        if self.latent_cvae_action_decoder is None and self.latent_main_action_decoder is None and self.residual_action_flow_denoiser is not None:
            assert legacy_velocity is not None
            if decoder_mode == "layered_residual_action_flow":
                context_memory = torch.cat([context_kv, registers], dim=1) if int(getattr(cfg, "action_flow_residual_context_memory", 1)) else context_kv
                transition_parts = [rollout, controlled_delta, event_context]
                if str(getattr(cfg, "controlled_base_mode", "learned")) != "fixed_zero":
                    transition_parts.insert(2, rollout_effect_pred)
                transition_memory = torch.cat(transition_parts, dim=1) if int(getattr(cfg, "action_flow_residual_transition_memory", 1)) else None
                residual_action_flow = self.residual_action_flow_denoiser(
                    noisy_physical=noisy_physical,
                    time=time,
                    trajectory_pooled=trajectory_pooled,
                    context_memory=context_memory,
                    transition_memory=transition_memory,
                    visual_memory=visual_memory if int(getattr(cfg, "action_flow_residual_visual_memory", 1)) else None,
                    layer_contracts=layer_contracts,
                )
            else:
                memory_parts: list[Tensor] = []
                if int(getattr(cfg, "action_flow_residual_context_memory", 1)):
                    memory_parts.append(context_kv)
                    memory_parts.append(registers)
                if int(getattr(cfg, "action_flow_residual_transition_memory", 1)):
                    memory_parts.extend([rollout, controlled_delta, rollout_effect_pred, event_context])
                if int(getattr(cfg, "action_flow_residual_visual_memory", 1)):
                    memory_parts.append(visual_memory)
                if int(getattr(cfg, "action_flow_residual_layer_memory", 1)) and layer_contracts:
                    last_layer = layer_contracts[-1]
                    for key in ("policy_effect_time_tokens", "policy_effect_tokens", "rollout_effect_pred", "rollout_delta_pred"):
                        value = last_layer.get(key)
                        if isinstance(value, Tensor) and value.ndim == 3 and value.shape[-1] == cfg.hidden_size:
                            memory_parts.append(value)
                residual_memory = torch.cat(memory_parts, dim=1) if memory_parts else context_kv
                residual_action_flow = self.residual_action_flow_denoiser(
                    noisy_physical=noisy_physical,
                    time=time,
                    trajectory_pooled=trajectory_pooled,
                    memory=residual_memory,
                )
            pred_physical_velocity = legacy_velocity + residual_action_flow["residual_velocity"]
            legacy_event_logits = legacy_event_logits + residual_action_flow["event_delta_logits"]
            legacy_motion_logits = legacy_motion_logits + residual_action_flow["motion_delta_logits"]
        gate_mean = {
            key: torch.stack([row[key] for row in gate_rows]).mean() if gate_rows else _zeros_like_scalar(canvas)
            for key in ("gate_self", "gate_visual", "gate_rollout", "gate_ffn")
        }
        content_norm = torch.stack(content_norm_rows).mean() if content_norm_rows else _zeros_like_scalar(canvas)
        time_norm = torch.stack(time_norm_rows).mean() if time_norm_rows else _zeros_like_scalar(canvas)
        with torch.no_grad():
            rollout_seed_final = self.final_norm(rollout_seed.to(device=rollout.device, dtype=rollout.dtype))
            rollout_deep_update_norm = (rollout.detach() - rollout_seed_final).float().norm(dim=-1).mean()
        out = {
            **midcut,
            "layer_contracts": layer_contracts,
            "canvas_tokens": canvas,
            "trajectory_tokens": trajectory,
            "rollout_tokens": rollout,
            "register_tokens": registers,
            "rollout_deep_update_norm": rollout_deep_update_norm,
            "rollout_deep_token_norm": rollout.detach().float().norm(dim=-1).mean(),
            "pred_physical_velocity": pred_physical_velocity,
            "action_flow_residual_velocity": (
                torch.zeros_like(pred_physical_velocity)
                if residual_action_flow is None else residual_action_flow["residual_velocity"]
            ),
            "action_flow_residual_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if residual_action_flow is None else residual_action_flow["residual_norm"]
            ),
            "action_flow_raw_residual_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if residual_action_flow is None else residual_action_flow["raw_residual_norm"]
            ),
            "action_flow_residual_alpha_mean": (
                _zeros_like_scalar(pred_physical_velocity)
                if residual_action_flow is None else residual_action_flow["alpha_mean"]
            ),
            "action_flow_stage_router_entropy": (
                _zeros_like_scalar(pred_physical_velocity)
                if residual_action_flow is None else residual_action_flow.get("stage_router_entropy", _zeros_like_scalar(pred_physical_velocity))
            ),
            "action_flow_stage_router_max": (
                _zeros_like_scalar(pred_physical_velocity)
                if residual_action_flow is None else residual_action_flow.get("stage_router_max", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_action_stage_router_entropy": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_main_action is None else latent_main_action.get("stage_router_entropy", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_action_stage_router_max": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_main_action is None else latent_main_action.get("stage_router_max", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_action_gripper_gate_mean": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_main_action is None else latent_main_action.get("gripper_gate_mean", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_action_layer_memory_count": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_main_action is None else latent_main_action.get("layer_memory_count", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_action_temporal_update_mean": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_main_action is None else latent_main_action.get("temporal_action_update_mean", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_action_temporal_near_depth": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_main_action is None else latent_main_action.get("temporal_near_depth", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_action_temporal_mid_depth": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_main_action is None else latent_main_action.get("temporal_mid_depth", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_kl": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_kl", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_prior_std": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_prior_std", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_post_std": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_post_std", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_z_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_z_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_condition_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_condition_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_condition_scan_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_condition_scan_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_condition_lateral_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_condition_lateral_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_layer_summary_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_layer_summary_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_transition_condition_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_transition_condition_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_transition_source_raw_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_transition_source_raw_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_rollout_token_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_rollout_token_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_rollout_token_count": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_rollout_token_count", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_consequence_scale_mean": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_consequence_scale_mean", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_consequence_gate_preference": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_consequence_gate_preference", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_consequence_mix_ratio": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_consequence_mix_ratio", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_posterior_used": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_posterior_used", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_gripper_gate_mean": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("gripper_gate_mean", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_layer_memory_count": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("layer_memory_count", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_prior_z_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_prior_z_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_post_z_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_post_z_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_mu_gap": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_mu_gap", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_prior_pred_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_prior_pred_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_post_pred_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_post_pred_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_post_gripper_gate_mean": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_post_gripper_gate_mean", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_refine_update_mean": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_refine_update_mean", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_noisy_gate_mean": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_noisy_gate_mean", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_noisy_branch_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_noisy_branch_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_noisy_branch_ratio": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_noisy_branch_ratio", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_route_entropy": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_route_entropy", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_route_max": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_route_max", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_route_effective_slots": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_route_effective_slots", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_progress_entropy": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_progress_entropy", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_progress_max": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_progress_max", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_progress_effective_slots": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_progress_effective_slots", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_progress_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_progress_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_continue_mean": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_continue_mean", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_prefix_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_prefix_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_progress_seed_entropy": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_progress_seed_entropy", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_progress_seed_max": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_progress_seed_max", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_progress_seed_effective_slots": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_progress_seed_effective_slots", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_progress_seed_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_progress_seed_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_route_temperature_mean": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_route_temperature_mean", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_semantic_bias_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_semantic_bias_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_output_adapter_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_output_adapter_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_function_delta_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_function_delta_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_base_highfreq_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_base_highfreq_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_refine_step_bias_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_refine_step_bias_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_capsule_layer_entropy": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_capsule_layer_entropy", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_capsule_layer_max": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_capsule_layer_max", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_capsule_layer_effective_slots": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_capsule_layer_effective_slots", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_condition_strength_mean": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_condition_strength_mean", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_condition_strength_std": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_condition_strength_std", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_condition_strength_max": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_condition_strength_max", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_condition_strength_min": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_condition_strength_min", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_condition_residual_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_condition_residual_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_context_direction_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_context_direction_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_micro_step_mean": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_micro_step_mean", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_micro_step_std": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_micro_step_std", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_micro_progress_mean": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_micro_progress_mean", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_micro_kp_mean": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_micro_kp_mean", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_micro_kd_mean": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_micro_kd_mean", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_micro_feedforward_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_micro_feedforward_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_micro_feedback_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_micro_feedback_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_micro_damping_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_micro_damping_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_micro_function_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_micro_function_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_micro_control_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_micro_control_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_micro_update_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_micro_update_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_micro_heun_error": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_micro_heun_error", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_micro_refine_block_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_micro_refine_block_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_regularizer": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_regularizer", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_route_entropy_regularizer": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_route_entropy_regularizer", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_mmdit_action_update_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_mmdit_action_update_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_mmdit_cond_update_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_mmdit_cond_update_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_mmdit_action_cond_attention": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_mmdit_action_cond_attention", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_mmdit_action_noisy_attention": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_mmdit_action_noisy_attention", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_mmdit_action_rollout_attention": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_mmdit_action_rollout_attention", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_mmdit_action_rollout_enrichment": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_mmdit_action_rollout_enrichment", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_mmdit_action_token_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_mmdit_action_token_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_mmdit_condition_token_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_mmdit_condition_token_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_mmdit_noisy_token_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_mmdit_noisy_token_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "rollout_effect_pred": rollout_effect_pred,
            "rollout_base_effect_pred": dynamics["rollout_base_effect_pred"],
            "rollout_delta_pred": controlled_delta,
            "rollout_coeff_abs_mean": dynamics["rollout_coeff_abs_mean"],
            "rollout_neutral_coeff_abs_mean": dynamics["rollout_neutral_coeff_abs_mean"],
            "rollout_centered_coeff_abs_mean": dynamics["rollout_centered_coeff_abs_mean"],
            "rollout_basis_norm": dynamics["rollout_basis_norm"],
            "rollout_delta_norm": dynamics["rollout_delta_norm"],
            "rollout_base_norm": dynamics["rollout_base_norm"],
            "rollout_decomposition_expansion_ratio": dynamics["rollout_decomposition_expansion_ratio"],
            "rollout_base_is_fixed_zero": dynamics["rollout_base_is_fixed_zero"],
            "rollout_delta_gain": dynamics["rollout_delta_gain"],
            "future_latent_pred": rollout_effect_pred,
            "action_effect_pred": rollout_effect_pred,
            "event_logits": legacy_event_logits,
            "motion_logits": legacy_motion_logits,
            "transition_latent": (
                event_context
                if hierarchical_mmdit_action is None
                else hierarchical_mmdit_action["transition_latent"]
            ),
            "gate_self": gate_mean["gate_self"],
            "gate_visual": gate_mean["gate_visual"],
            "gate_rollout": gate_mean["gate_rollout"],
            "gate_ffn": gate_mean["gate_ffn"],
            "mod_content_norm": content_norm,
            "mod_time_norm": time_norm,
            "mod_content_to_time": content_norm / time_norm.clamp_min(1e-6),
            "midcut_stop": torch.zeros((), device=canvas.device, dtype=canvas.dtype),
        }
        if legacy_velocity is not None:
            assert direct_velocity is not None
            assert rollout_residual_velocity is not None
            assert rollout_alpha is not None
            out.update({
                "direct_physical_velocity": direct_velocity,
                "rollout_residual_velocity": rollout_residual_velocity,
                "legacy_physical_velocity": legacy_velocity,
                "rollout_alpha": rollout_alpha,
            })
        if latent_cvae_action is not None:
            for key, value in latent_cvae_action.items():
                if key.startswith("cvae_") and isinstance(value, Tensor):
                    out.setdefault(f"latent_{key}", value)
        if latent_cvae_action is not None and "post_pred_velocity" in latent_cvae_action:
            out.update({
                "post_pred_velocity": latent_cvae_action["post_pred_velocity"],
                "post_event_logits": latent_cvae_action.get("post_event_logits", legacy_event_logits),
                "post_motion_logits": latent_cvae_action.get("post_motion_logits", legacy_motion_logits),
            })
        if latent_cvae_action is not None:
            for key in (
                "cvae_adaptive_micro_controller_norm",
                "cvae_adaptive_micro_pred_velocity",
                "cvae_adaptive_micro_event_logits",
                "cvae_adaptive_micro_supervision_logits",
            ):
                if key in latent_cvae_action:
                    out[f"latent_{key}"] = latent_cvae_action[key]
        if hierarchical_mmdit_action is not None:
            for key in tuple(out):
                if key.startswith("latent_cvae_"):
                    out.pop(key)
            for key, value in hierarchical_mmdit_action.items():
                if not isinstance(value, Tensor):
                    continue
                if key.startswith(("intent_", "owned_", "hierarchical_mmdit_")):
                    out[key] = value
        return out

    @torch.no_grad()
    def target_rollout_effect(self, visual: Tensor, target_visual: Tensor) -> Tensor:
        return self.rollout_codec.target_effect(visual, target_visual)
