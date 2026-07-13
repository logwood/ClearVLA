from __future__ import annotations

"""Top-level current policy composition."""

import torch
from torch import Tensor, nn

from .codec import PhysicalActionCodec
from .config import V39PolicyConfig
from .proposal import RejectableHistoryProposal
from .trunk import TemporalMidcutWorldActionDiT


class V39PolicySystem(nn.Module):
    def __init__(self, policy_config: V39PolicyConfig) -> None:
        super().__init__()
        self.policy_config = policy_config
        self.codec = PhysicalActionCodec(policy_config)
        self.proposal = RejectableHistoryProposal(policy_config)
        self.planner = TemporalMidcutWorldActionDiT(policy_config)

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
        *,
        stop_at_midcut: bool = False,
        consequence_physical: Tensor | None = None,
        cvae_target_physical: Tensor | None = None,
        enable_layer_contracts: bool = True,
        enable_final_action_decoder: bool = True,
    ) -> dict[str, Tensor]:
        return self.planner(
            noisy_physical,
            time,
            visual,
            state_history,
            state,
            executed_history,
            proposal_tokens,
            proposal_keep,
            stop_at_midcut=stop_at_midcut,
            consequence_physical=consequence_physical,
            cvae_target_physical=cvae_target_physical,
            enable_layer_contracts=enable_layer_contracts,
            enable_final_action_decoder=enable_final_action_decoder,
        )

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
        stop_at_midcut: bool = False,
    ) -> dict[str, Tensor]:
        del future_training_pack
        proposal = self.proposal(executed_history)
        if self.codec.uses_arm_manifold and action_state is None:
            raise ValueError(
                "manifold_native training requires action_state in action-normalizer coordinates"
            )
        codec_state = state if action_state is None else action_state
        target_physical = self.codec.encode(target_action, codec_state)
        noise = self.codec.sample_noise(
            target_physical.shape[0],
            device=target_physical.device,
            dtype=target_physical.dtype,
            action_state=codec_state,
        )
        t = torch.rand(target_physical.shape[0], device=target_physical.device, dtype=target_physical.dtype)
        noisy_physical = (1 - t[:, None, None]) * target_physical + t[:, None, None] * noise
        target_physical_velocity = noise - target_physical
        drop = self.policy_config.proposal_dropout if proposal_dropout is None else float(proposal_dropout)
        keep = (torch.rand(target_physical.shape[0], device=target_physical.device) >= drop).to(target_physical.dtype)

        consequence_input = noisy_physical
        preview_velocity: Tensor | None = None
        use_self_condition = (
            int(getattr(self.policy_config, "action_consequence_self_condition", 0))
            and int(getattr(self.policy_config, "layer_recurrent_consequence", 0))
            and int(getattr(self.policy_config, "layer_contract_adapters", 0))
        )
        if use_self_condition:
            with torch.no_grad():
                preview = self._policy_forward(
                    noisy_physical.detach(),
                    t.detach(),
                    visual,
                    state_history,
                    state,
                    executed_history,
                    proposal["tokens"].detach(),
                    keep,
                    stop_at_midcut=stop_at_midcut,
                    consequence_physical=noisy_physical.detach(),
                    cvae_target_physical=None,
                    enable_layer_contracts=False,
                )
                preview_velocity = preview["pred_physical_velocity"].detach()
                consequence_input = (noisy_physical - t[:, None, None] * preview_velocity).detach()
                del preview

        action_policy = self._policy_forward(
            noisy_physical,
            t,
            visual,
            state_history,
            state,
            executed_history,
            proposal["tokens"].detach(),
            keep,
            stop_at_midcut=stop_at_midcut,
            consequence_physical=consequence_input,
            cvae_target_physical=target_physical,
        )
        # V70 (H3 fix): every training-time clean estimate is projected onto the
        # physical manifold BEFORE any decode/loss use, matching deployment
        # where each integration step is projected.  This closes the null
        # arbitrage channel (arm decode blend was null-sensitive) and aligns
        # the train/deploy decode geometry.  Null components remain fully
        # visible to the velocity-space flow/null losses.
        clean_physical_estimate = self.codec.project_physical(
            noisy_physical - t[:, None, None] * action_policy["pred_physical_velocity"],
            codec_state,
        )
        decoded_action = self.codec.decode(clean_physical_estimate, codec_state)
        if "post_pred_velocity" in action_policy:
            post_clean = self.codec.project_physical(
                noisy_physical - t[:, None, None] * action_policy["post_pred_velocity"],
                codec_state,
            )
            action_policy["post_clean_physical_estimate"] = post_clean
            action_policy["post_pred_action_estimate"] = self.codec.decode(post_clean, codec_state)
        out = {
            **action_policy,
            "pred_physical_velocity": action_policy["pred_physical_velocity"],
            "target_physical_velocity": target_physical_velocity,
            "target_physical": target_physical,
            "clean_physical_estimate": clean_physical_estimate,
            "proposal_action": proposal["action"],
            "time": t,
            "noisy_physical_action": noisy_physical,
            "source_physical_noise": noise,
            "pred_action_estimate": decoded_action,
            "future_conditioned_action_loss": torch.zeros((), device=target_physical.device, dtype=target_physical.dtype),
        }
        if preview_velocity is not None:
            out["consequence_self_condition"] = torch.ones((), device=target_physical.device, dtype=target_physical.dtype)
            out["consequence_self_condition_target_mse"] = (
                consequence_input.float() - target_physical.detach().float()
            ).square().mean()
            out["consequence_self_condition_noisy_mse"] = (
                consequence_input.float() - noisy_physical.detach().float()
            ).square().mean()
            out["consequence_preview_flow"] = (
                preview_velocity.float() - target_physical_velocity.detach().float()
            ).square().mean()
        if "midcut_pred_physical_velocity" in action_policy:
            mid_clean = self.codec.project_physical(
                noisy_physical - t[:, None, None] * action_policy["midcut_pred_physical_velocity"],
                codec_state,
            )
            out["midcut_clean_physical_estimate"] = mid_clean
            out["midcut_pred_action_estimate"] = self.codec.decode(mid_clean, codec_state)
        if "layer_contracts" in action_policy:
            for entry in action_policy["layer_contracts"]:
                clean = self.codec.project_physical(
                    noisy_physical - t[:, None, None] * entry["pred_physical_velocity"],
                    codec_state,
                )
                entry["clean_physical_estimate"] = clean
                entry["pred_action_estimate"] = self.codec.decode(clean, codec_state)

        pack = rollout_target_pack
        if pack is None and target_visual is not None:
            pack = self.build_rollout_target_pack(visual, target_visual)
        def _merge_layer_contract_counterfactuals(variant: dict[str, Tensor], suffix: str) -> None:
            base_layers = out.get("layer_contracts")
            variant_layers = variant.get("layer_contracts")
            if not isinstance(base_layers, list) or not isinstance(variant_layers, list):
                return
            for base_entry, var_entry in zip(base_layers, variant_layers):
                for key in (
                    "rollout_effect_pred", "rollout_delta_pred", "rollout_base_effect_pred",
                    "milestone_step_delta_pred", "causal_rollout_effect_pred", "causal_rollout_delta_pred",
                    "policy_effect_tokens", "unified_intervention_latent_pred", "neutral_latent_pred",
                    "rollout_effect_pred_shuffle_state", "rollout_delta_pred_shuffle_state",
                    "milestone_step_delta_pred_shuffle_state",
                ):
                    if key in var_entry:
                        base_entry[f"{key}_{suffix}"] = var_entry[key]

        if pack is not None:
            target = pack["rollout_effect_target"].to(device=target_physical.device, dtype=action_policy["rollout_effect_pred"].dtype)
            out["rollout_effect_target"] = target
            out["future_latent_target"] = target
            out["future_latent_velocity_target"] = target
            out["action_effect_target"] = target
            if make_counterfactuals:
                hold_action = codec_state[:, None].expand_as(target_action)
                hold_physical = self.codec.encode(hold_action, codec_state)
                hold_noisy = (1 - t[:, None, None]) * hold_physical + t[:, None, None] * noise
                hold_policy = self._policy_forward(
                    hold_noisy.detach(),
                    t.detach(),
                    visual,
                    state_history,
                    state,
                    executed_history,
                    proposal["tokens"].detach(),
                    keep,
                    stop_at_midcut=stop_at_midcut,
                    consequence_physical=hold_physical,
                    enable_final_action_decoder=False,
                )
                out["rollout_effect_pred_hold_action"] = hold_policy["rollout_effect_pred"]
                out["rollout_delta_pred_hold_action"] = hold_policy["rollout_delta_pred"]
                out["rollout_base_effect_pred_hold_action"] = hold_policy["rollout_base_effect_pred"]
                if "midcut_rollout_effect_pred" in hold_policy:
                    out["midcut_rollout_effect_pred_hold_action"] = hold_policy["midcut_rollout_effect_pred"]
                    out["midcut_rollout_delta_pred_hold_action"] = hold_policy["midcut_rollout_delta_pred"]
                    out["midcut_rollout_base_effect_pred_hold_action"] = hold_policy["midcut_rollout_base_effect_pred"]
                _merge_layer_contract_counterfactuals(hold_policy, "hold_action")
                if target_physical.shape[0] > 1:
                    # V40: within-batch hard action negative.  V39 used a
                    # reverse-batch permutation, which can become an adjacent
                    # near-duplicate on ordered validation batches.  The hard
                    # negative must be encoded in the current sample's action
                    # state; directly permuting target_physical would mix
                    # state-relative coordinate frames and inflate shuffle
                    # diagnostics.
                    batch = int(target_action.shape[0])
                    cand_action = (
                        target_action.detach()[None]
                        .expand(batch, -1, -1, -1)
                        .reshape(batch * batch, int(target_action.shape[1]), int(target_action.shape[2]))
                    )
                    cand_state = (
                        codec_state.detach()[:, None]
                        .expand(-1, batch, -1)
                        .reshape(batch * batch, int(codec_state.shape[-1]))
                    )
                    cand_physical = self.codec.encode(cand_action, cand_state).reshape(
                        batch,
                        batch,
                        int(target_physical.shape[1]),
                        int(target_physical.shape[2]),
                    )
                    dist = (
                        cand_physical.detach().float()
                        - target_physical.detach().float()[:, None]
                    ).flatten(2).norm(dim=-1)
                    eye = torch.eye(batch, device=dist.device, dtype=torch.bool)
                    dist = dist.masked_fill(eye, -1.0)
                    perm = dist.argmax(dim=1)
                    shuffle_physical = cand_physical[torch.arange(batch, device=target_physical.device), perm]
                else:
                    shuffle_physical = target_physical
                shuffle_noisy = (1 - t[:, None, None]) * shuffle_physical + t[:, None, None] * noise
                shuffle_policy = self._policy_forward(
                    shuffle_noisy.detach(),
                    t.detach(),
                    visual,
                    state_history,
                    state,
                    executed_history,
                    proposal["tokens"].detach(),
                    keep,
                    stop_at_midcut=stop_at_midcut,
                    consequence_physical=shuffle_physical,
                    enable_final_action_decoder=False,
                )
                out["rollout_effect_pred_shuffle_action"] = shuffle_policy["rollout_effect_pred"]
                out["rollout_delta_pred_shuffle_action"] = shuffle_policy["rollout_delta_pred"]
                out["rollout_base_effect_pred_shuffle_action"] = shuffle_policy["rollout_base_effect_pred"]
                if "midcut_rollout_effect_pred" in shuffle_policy:
                    out["midcut_rollout_effect_pred_shuffle_action"] = shuffle_policy["midcut_rollout_effect_pred"]
                    out["midcut_rollout_delta_pred_shuffle_action"] = shuffle_policy["midcut_rollout_delta_pred"]
                    out["midcut_rollout_base_effect_pred_shuffle_action"] = shuffle_policy["midcut_rollout_base_effect_pred"]
                _merge_layer_contract_counterfactuals(shuffle_policy, "shuffle_action")
        return out

    @torch.no_grad()
    def sample(
        self,
        visual: Tensor,
        state_history: Tensor,
        executed_history: Tensor,
        state: Tensor,
        *,
        action_state: Tensor | None = None,
        steps: int | None = None,
        noise: Tensor | None = None,
        use_proposal: bool = True,
        return_event_logits: bool = False,
        stop_at_midcut: bool = False,
    ) -> Tensor | dict[str, Tensor]:
        """Sample an action chunk without teacher-forcing target actions.

        ``stop_at_midcut`` is used for contract-stage validation.  It evaluates
        the deployable mid-cut simple heads by iterative denoising from noise;
        unlike ``flow_training_forward`` it never receives ``target_action`` and
        therefore cannot leak validation labels into action metrics.
        """
        proposal = self.proposal(executed_history)
        steps = int(steps or self.policy_config.inference_steps)
        if steps <= 0:
            raise ValueError("steps must be positive")
        if self.codec.uses_arm_manifold and action_state is None:
            raise ValueError("manifold_native sampling requires action_state in action-normalizer coordinates")
        codec_state = (state if action_state is None else action_state).to(
            device=visual.device, dtype=visual.dtype,
        )
        if noise is None:
            x = self.codec.sample_noise(
                visual.shape[0],
                device=visual.device,
                dtype=visual.dtype,
                action_state=codec_state,
            )
        else:
            x = noise.clone()
            if x.shape[-1] == self.policy_config.action_dim:
                x = self.codec.encode(x.to(device=visual.device, dtype=visual.dtype), codec_state)
            elif x.shape[-1] != self.policy_config.physical_action_dim:
                raise ValueError("noise must have last dim action_dim or physical_action_dim")
            else:
                x = x.to(device=visual.device, dtype=visual.dtype)
            x = self.codec.project_physical(x, codec_state)
        keep = torch.full((visual.shape[0],), 1.0 if use_proposal else 0.0, device=visual.device, dtype=visual.dtype)
        use_self_condition = (
            int(getattr(self.policy_config, "action_consequence_self_condition", 0))
            and int(getattr(self.policy_config, "layer_recurrent_consequence", 0))
            and int(getattr(self.policy_config, "layer_contract_adapters", 0))
        )
        sample_diagnostic_sums: dict[str, Tensor] = {}
        sample_diagnostic_count = 0
        for index in range(steps, 0, -1):
            t = torch.full((visual.shape[0],), float(index) / float(steps), device=visual.device, dtype=visual.dtype)
            consequence_input = x
            if use_self_condition:
                preview = self._policy_forward(
                    x,
                    t,
                    visual,
                    state_history,
                    state,
                    executed_history,
                    proposal["tokens"],
                    keep,
                    stop_at_midcut=stop_at_midcut,
                    consequence_physical=x,
                    enable_layer_contracts=False,
                )
                consequence_input = (x - t[:, None, None] * preview["pred_physical_velocity"]).detach()
                del preview
            out = self._policy_forward(
                x, t, visual, state_history, state, executed_history, proposal["tokens"], keep,
                stop_at_midcut=stop_at_midcut,
                consequence_physical=consequence_input,
                enable_layer_contracts=False,
            )
            for key, value in out.items():
                keep_diagnostic = (
                    key.startswith("latent_cvae_workspace_")
                    or key.startswith("latent_cvae_mmdit_")
                    or key.startswith("latent_cvae_hierarchical_")
                    or key.startswith("intent_")
                    or key.startswith("owned_")
                    or key.startswith("hierarchical_mmdit_")
                    or key in (
                        "latent_cvae_primary_condition_norm",
                        "latent_cvae_primary_z_effect_norm",
                    )
                )
                if keep_diagnostic and torch.is_tensor(value) and value.numel() == 1:
                    scalar = value.detach().float()
                    sample_diagnostic_sums[key] = sample_diagnostic_sums.get(key, torch.zeros_like(scalar)) + scalar
            sample_diagnostic_count += 1
            raw_next = x - out["pred_physical_velocity"] / float(steps)
            projected_next = self.codec.project_physical(raw_next, codec_state)
            # Pre-projection null drift: raw per-step magnitude plus the
            # step-size-normalized rate (x steps == / h) so runs with different
            # inference step counts stay comparable.
            null_drift = (raw_next - projected_next).detach().float()
            arm_span = 2 * int(self.codec.arm_dim)
            arm_null_norm = null_drift[..., :arm_span].norm(dim=-1).mean()
            grip_null_norm = null_drift[..., arm_span:].norm(dim=-1).mean()
            for null_key, null_value in (
                ("arm_null_preproject", arm_null_norm),
                ("arm_null_preproject_rate", arm_null_norm * float(steps)),
                ("grip_null_preproject", grip_null_norm),
                ("grip_null_preproject_rate", grip_null_norm * float(steps)),
            ):
                sample_diagnostic_sums[null_key] = sample_diagnostic_sums.get(null_key, torch.zeros_like(null_value)) + null_value
            x = projected_next
        action = self.codec.decode(x, codec_state)
        if return_event_logits:
            zero_t = torch.zeros((visual.shape[0],), device=visual.device, dtype=visual.dtype)
            event = self._policy_forward(
                x, zero_t, visual, state_history, state, executed_history, proposal["tokens"], keep,
                stop_at_midcut=stop_at_midcut,
                enable_layer_contracts=False,
            )
            result = {
                "action": action,
                "physical_action": x,
                "event_logits": event["event_logits"],
                "motion_logits": event["motion_logits"],
            }
            for key, value in sample_diagnostic_sums.items():
                result[f"sample_{key}"] = value / float(max(sample_diagnostic_count, 1))
            return result
        return action

    def parameter_report(self) -> dict[str, int]:
        report = {
            "history_proposal": sum(p.numel() for p in self.proposal.parameters()),
            "physical_action_codec": sum(p.numel() for p in self.codec.parameters()),
            "midcut_contract_heads": sum(p.numel() for p in self.planner.midcut_heads.parameters()),
            "layer_contract_adapters": sum(p.numel() for p in self.planner.layer_contract_heads.parameters()),
            "layer_shared_fm_probe": (0 if self.planner.layer_fm_probe is None else sum(p.numel() for p in self.planner.layer_fm_probe.parameters())),
            "layer_recurrent_consequence": (0 if self.planner.layer_consequence_cell is None else sum(p.numel() for p in self.planner.layer_consequence_cell.parameters())),
            "layer_causal_effect_branch": (0 if self.planner.layer_consequence_cell is None else sum(p.numel() for p in self.planner.layer_consequence_cell.parameters())),
            "layer_role_scheduler": sum(p.numel() for p in self.planner.layer_role_scheduler.parameters()),
            "residual_action_flow_denoiser": (
                0 if self.planner.residual_action_flow_denoiser is None
                else sum(p.numel() for p in self.planner.residual_action_flow_denoiser.parameters())
            ),
            "latent_main_action_decoder": (
                0 if getattr(self.planner, "latent_main_action_decoder", None) is None
                else sum(p.numel() for p in self.planner.latent_main_action_decoder.parameters())
            ),
            "latent_cvae_action_decoder": (
                0 if getattr(self.planner, "latent_cvae_action_decoder", None) is None
                else sum(p.numel() for p in self.planner.latent_cvae_action_decoder.parameters())
            ),
            "hierarchical_mmdit_action_decoder": (
                0 if getattr(self.planner, "hierarchical_mmdit_action_decoder", None) is None
                else sum(p.numel() for p in self.planner.hierarchical_mmdit_action_decoder.parameters())
            ),
            "staged_midcut_dit": sum(p.numel() for p in self.planner.parameters()),
        }
        report["total"] = sum(p.numel() for p in self.parameters())
        report["trainable"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return report
