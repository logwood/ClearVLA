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

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from clearvla.policy.trunk_primitives import (
    CanvasPhysicalVelocityHead,
    ControlledResidualLatentDynamics,
    DenseVisualMemory,
    RolloutActionResidualHead,
    RolloutTargetCodec,
    TemporalDynamicsBoundDiTBlock,
    UnifiedCanvasSeed,
)
from clearvla.policy.config import V38PolicyConfig

from .policy import RejectableHistoryProposal, TimeEmbedding
from .policy_v36_2 import (
    HorizonRoleEmbedding,
    PhysicalActionCodec,
    PhysicalActionTokenLift,
    V362PolicyConfig,
)
from .world_model import BiasFreeFFN, sinusoidal_positions


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
