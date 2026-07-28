"""Top-level current policy composition."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .codec import DCTFlowCodec, PhysicalActionCodec
from .config import V39PolicyConfig
from .proposal import RejectableHistoryProposal
from .trunk import TemporalMidcutWorldActionDiT


@torch.no_grad()
def balanced_future_teacher_mask(
    future_target: Tensor,
    current_target: Tensor,
    observed_mask: Tensor,
    *,
    mask_ratio: float,
    past_fraction: float,
    change_fraction: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Allocate exact per-horizon JEPA target quotas without future leakage.

    ``observed_mask`` is built by the online encoder from past/current
    evidence. ``future_target`` and ``current_target`` are frozen teacher
    charts and are used only here, after the forward pass, to select loss
    positions.  The three lanes are disjoint by construction: observed-motion
    coverage, actual future change, and deterministic spatial coverage.
    """

    if future_target.ndim != 3 or current_target.ndim != 3:
        raise ValueError("future/current teacher targets must be [B,N,H]")
    batch, future_tokens, hidden = future_target.shape
    if tuple(current_target.shape[:1]) != (batch,) or int(current_target.shape[-1]) != hidden:
        raise ValueError("current teacher chart must share batch/hidden dimensions")
    tokens_per_horizon = int(current_target.shape[1])
    if tokens_per_horizon <= 0 or future_tokens % tokens_per_horizon:
        raise ValueError("future teacher tokens must contain complete horizon charts")
    horizons = future_tokens // tokens_per_horizon
    if tuple(observed_mask.shape) != (batch, future_tokens):
        raise ValueError("observed target mask must match future teacher tokens")

    total = int(round(float(mask_ratio) * tokens_per_horizon))
    total = min(max(total, 0), tokens_per_horizon)
    if total == 0:
        empty = torch.zeros_like(observed_mask, dtype=torch.bool)
        zero = future_target.new_zeros((), dtype=torch.float32)
        return empty, {
            "flow_jepa_teacher_mask_past_fraction": zero,
            "flow_jepa_teacher_mask_change_fraction": zero,
            "flow_jepa_teacher_mask_uniform_fraction": zero,
            "flow_jepa_teacher_mask_selected_change_ratio": zero,
        }

    past_count = min(int(round(total * float(past_fraction))), total)
    change_count = min(
        int(round(total * float(change_fraction))), total - past_count
    )
    uniform_count = total - past_count - change_count
    observed = observed_mask.reshape(batch, horizons, tokens_per_horizon).bool()
    future = future_target.reshape(batch, horizons, tokens_per_horizon, hidden).float()
    current = current_target[:, None].float()
    change = (future - current).square().mean(dim=-1).clamp_min(0.0).sqrt()
    lo = change.amin(dim=-1, keepdim=True)
    hi = change.amax(dim=-1, keepdim=True)
    normalized_change = (change - lo) / (hi - lo).clamp_min(1e-6)

    position = torch.arange(
        tokens_per_horizon, device=future_target.device, dtype=torch.float32
    )[None, None]
    horizon = torch.arange(
        horizons, device=future_target.device, dtype=torch.float32
    )[None, :, None]
    # Irrational rotations give deterministic, spatially dispersed tie breaks
    # without consuming the training RNG or privileging the first grid cells.
    coverage = torch.frac((position + 1.0) * 0.754877666 + (horizon + 1.0) * 0.569840291)
    coverage = coverage.expand(batch, -1, -1)
    selected = torch.zeros_like(observed)

    def select(score: Tensor, count: int) -> None:
        nonlocal selected
        if count <= 0:
            return
        available_score = score.masked_fill(selected, torch.finfo(score.dtype).min)
        indices = available_score.topk(k=count, dim=-1).indices
        selected = selected.scatter(-1, indices, True)

    # Observed entries outrank non-observed entries; coverage only breaks ties
    # inside the online evidence mask.  Later lanes cannot select them again.
    select(observed.float() * 2.0 + coverage, past_count)
    select(normalized_change + 1e-3 * coverage, change_count)
    select(coverage, uniform_count)

    selected_float = selected.float()
    selected_change = (normalized_change * selected_float).sum(dim=-1) / selected_float.sum(
        dim=-1
    ).clamp_min(1.0)
    all_change = normalized_change.mean(dim=-1).clamp_min(1e-6)
    metrics = {
        "flow_jepa_teacher_mask_past_fraction": future_target.new_tensor(
            past_count / float(total), dtype=torch.float32
        ),
        "flow_jepa_teacher_mask_change_fraction": future_target.new_tensor(
            change_count / float(total), dtype=torch.float32
        ),
        "flow_jepa_teacher_mask_uniform_fraction": future_target.new_tensor(
            uniform_count / float(total), dtype=torch.float32
        ),
        "flow_jepa_teacher_mask_selected_change_ratio": (
            selected_change / all_change
        ).mean().detach(),
    }
    return selected.flatten(1, 2), metrics

_EVIDENCE_SAMPLING_DIAGNOSTIC_KEYS = frozenset(
    {
        "evidence_condition_norm",
        "evidence_global_condition_norm",
        "evidence_latent_batch_variance",
        "evidence_latent_norm",
        "evidence_mmd_it_action_update_norm",
        "evidence_mmd_it_boundary_identity_error",
        "evidence_mmd_it_capacity_ratio",
        "evidence_mmd_it_committed_operation_count",
        "evidence_mmd_it_contraction_ratio",
        "evidence_mmd_it_controller_slot_common_mode_ratio",
        "evidence_mmd_it_controller_slot_pair_cosine",
        "evidence_mmd_it_controller_slot_private_energy_ratio",
        "evidence_mmd_it_depth_ratio",
        "evidence_mmd_it_dwell_compute_fraction",
        "evidence_mmd_it_dwell_expected",
        "evidence_mmd_it_dynamic_route_next_fraction",
        "evidence_mmd_it_effective_depth",
        "evidence_mmd_it_execution_cost",
        "evidence_mmd_it_execution_progress",
        "evidence_mmd_it_execution_selection_entropy",
        "evidence_mmd_it_execution_selection_max_probability",
        "evidence_mmd_it_hard_dwell_expected",
        "evidence_mmd_it_hard_route_next_fraction",
        "evidence_mmd_it_learned_selection_entropy",
        "evidence_mmd_it_nonexpansive_violation",
        "evidence_mmd_it_removed_channel_fraction",
        "evidence_mmd_it_selected_active_group_fraction",
        "evidence_mmd_it_selected_effective_depth",
        "evidence_top_policy_workspace_horizon_pool",
        "evidence_z_shuffle_condition_delta",
        "evidence_z_zero_condition_delta",
        "flow_jepa_late_detail_attention_entropy",
        "flow_jepa_late_detail_attention_max",
        "flow_jepa_late_detail_fixed_scale",
        "flow_jepa_late_detail_token_count",
        "flow_jepa_late_detail_trajectory_ratio",
        "flow_jepa_late_detail_update_norm",
        "flow_jepa_online_horizon_address",
        "flow_jepa_online_horizon_address_write_rms",
        "flow_jepa_progressive_grounding_address",
        "flow_jepa_coordinate_typed_raw_detail",
        "flow_jepa_literal_rgb_chart_rms",
        "flow_jepa_online_address_boundary_seed_rms",
        "flow_jepa_online_address_boundary_seed_adjacent_cosine",
        "flow_jepa_online_address_boundary_post_g3_rms",
        "flow_jepa_online_address_boundary_post_g3_adjacent_cosine",
        "flow_jepa_online_address_boundary_post_address_rms",
        "flow_jepa_online_address_boundary_post_address_adjacent_cosine",
        "flow_jepa_online_address_boundary_post_w1_rms",
        "flow_jepa_online_address_boundary_post_w1_adjacent_cosine",
        "flow_jepa_online_address_boundary_post_w1_cumulative_address_cosine",
        "flow_jepa_online_address_boundary_post_w1_cumulative_address_projection",
        "flow_jepa_online_address_boundary_post_w2_rms",
        "flow_jepa_online_address_boundary_post_w2_adjacent_cosine",
        "flow_jepa_online_address_boundary_post_w2_cumulative_address_cosine",
        "flow_jepa_online_address_boundary_post_w2_cumulative_address_projection",
        "flow_jepa_online_address_boundary_post_w3_rms",
        "flow_jepa_online_address_boundary_post_w3_adjacent_cosine",
        "flow_jepa_online_address_boundary_post_w3_cumulative_address_cosine",
        "flow_jepa_online_address_boundary_post_w3_cumulative_address_projection",
        "flow_jepa_online_address_boundary_post_interval_rms",
        "flow_jepa_online_address_boundary_post_interval_adjacent_cosine",
        "flow_jepa_online_address_boundary_post_interval_cumulative_address_cosine",
        "flow_jepa_online_address_boundary_post_interval_cumulative_address_projection",
        "flow_jepa_raw_detail_action_independent_compile",
        "flow_jepa_raw_detail_deferred_to_policy",
        "flow_jepa_world_anchor_camera_residual_norm",
        "flow_jepa_world_anchor_write_only",
        "flow_jepa_world_spatial_residual_norm",
    }
)


def _keep_sampling_diagnostic(key: str, *, evidence_active: bool) -> bool:
    if evidence_active:
        return (
            key in _EVIDENCE_SAMPLING_DIAGNOSTIC_KEYS
            or key.startswith("flow_jepa_progressive_")
            or key.startswith("flow_jepa_typed_")
            or (
                key.startswith("evidence_mmd_it_block_")
                and key.endswith("_update_norm")
            )
            or (
                key.startswith("evidence_mmd_it_committed_block_")
                and key.endswith("_fraction")
            )
        )
    return (
        key.startswith("latent_cvae_workspace_")
        or key.startswith("latent_cvae_mmdit_")
        or key.startswith("latent_cvae_hierarchical_")
        or key.startswith("intent_")
        or key.startswith("owned_")
        or key.startswith("hierarchical_mmdit_")
        or key
        in (
            "latent_cvae_primary_condition_norm",
            "latent_cvae_primary_z_effect_norm",
        )
    )


class V39PolicySystem(nn.Module):
    def __init__(self, policy_config: V39PolicyConfig) -> None:
        super().__init__()
        self.policy_config = policy_config
        # Evaluation-only condition intervention state. It is deliberately a
        # plain Python attribute so probes never alter checkpoints or training.
        self._condition_eval_intervention: str | None = None
        self._condition_eval_apply_count = 0
        self._condition_eval_metrics: dict[str, float] = {}
        self.codec = PhysicalActionCodec(policy_config)
        self.flow_codec = (
            DCTFlowCodec(policy_config)
            if int(getattr(policy_config, "hierarchical_mmdit_spectral_state", 0))
            else None
        )
        self.proposal = RejectableHistoryProposal(policy_config)
        self.planner = TemporalMidcutWorldActionDiT(policy_config)
        if int(getattr(policy_config, "goal_conditioning_enabled", 0)):
            default_tokens = torch.zeros(
                1,
                int(policy_config.goal_language_max_tokens),
                int(policy_config.goal_language_dim),
                dtype=torch.float32,
            )
            default_mask = torch.zeros(
                1,
                int(policy_config.goal_language_max_tokens),
                dtype=torch.bool,
            )
            default_mask[:, 0] = True
            self.register_buffer(
                "default_goal_language_tokens",
                default_tokens,
                persistent=True,
            )
            self.register_buffer(
                "default_goal_language_mask",
                default_mask,
                persistent=True,
            )
        else:
            self.register_buffer(
                "default_goal_language_tokens",
                torch.empty(0),
                persistent=False,
            )
            self.register_buffer(
                "default_goal_language_mask",
                torch.empty(0, dtype=torch.bool),
                persistent=False,
            )

    @torch.no_grad()
    def set_default_goal_language(self, tokens: Tensor, mask: Tensor) -> None:
        """Install one precomputed T5 condition for single-task runs."""

        if not int(getattr(self.policy_config, "goal_conditioning_enabled", 0)):
            raise ValueError("goal conditioning is disabled")
        tokens = torch.as_tensor(tokens, dtype=torch.float32, device="cpu")
        mask = torch.as_tensor(mask, dtype=torch.bool, device="cpu")
        if tokens.ndim == 2:
            tokens = tokens[None]
        if mask.ndim == 1:
            mask = mask[None]
        if int(tokens.shape[0]) != 1 or tuple(mask.shape) != tuple(tokens.shape[:2]):
            raise ValueError("default goal language must be tokens=[1,L,D], mask=[1,L]")
        if int(tokens.shape[-1]) != int(self.policy_config.goal_language_dim):
            raise ValueError("default goal language width does not match goal_language_dim")
        if int(tokens.shape[1]) > int(self.policy_config.goal_language_max_tokens):
            raise ValueError("default goal language exceeds goal_language_max_tokens")
        self.default_goal_language_tokens.zero_()
        self.default_goal_language_mask.zero_()
        length = int(tokens.shape[1])
        self.default_goal_language_tokens[:, :length].copy_(tokens)
        self.default_goal_language_mask[:, :length].copy_(mask)

    def _goal_language_batch(
        self,
        batch: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        tokens: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> tuple[Tensor | None, Tensor | None]:
        if not int(getattr(self.policy_config, "goal_conditioning_enabled", 0)):
            if tokens is not None or mask is not None:
                raise ValueError("language tokens were supplied while goal conditioning is disabled")
            return None, None
        source_tokens = self.default_goal_language_tokens if tokens is None else tokens
        source_mask = self.default_goal_language_mask if mask is None else mask
        if source_tokens.ndim != 3 or source_mask.ndim != 2:
            raise ValueError("goal language tokens/mask must be [B,L,D] and [B,L]")
        if int(source_tokens.shape[0]) == 1:
            source_tokens = source_tokens.expand(batch, -1, -1)
            source_mask = source_mask.expand(batch, -1)
        if int(source_tokens.shape[0]) != batch:
            raise ValueError("goal language batch size does not match policy batch")
        return (
            source_tokens.to(device=device, dtype=dtype),
            source_mask.to(device=device, dtype=torch.bool),
        )

    def set_condition_eval_intervention(self, mode: str) -> None:
        """Select a transient deployed-condition intervention."""

        normalized = str(mode).strip().lower().replace("-", "_")
        allowed = {
            "none",
            "goal_zero",
            "goal_batch_shuffle",
            "history_zero",
            "history_condition_zero",
            "history_proposal_zero",
            "history_proposal_batch_shuffle",
            "history_batch_shuffle",
            "history_truncate",
        }
        if normalized not in allowed:
            raise ValueError(
                "condition intervention must be none/goal_zero/"
                "goal_batch_shuffle/history_zero/history_condition_zero/"
                "history_proposal_zero/history_proposal_batch_shuffle/"
                "history_batch_shuffle/history_truncate"
            )
        if self.training:
            raise RuntimeError("condition intervention is evaluation-only")
        if normalized.startswith("goal_") and not int(
            getattr(self.policy_config, "goal_conditioning_enabled", 0)
        ):
            raise RuntimeError("goal intervention requires goal conditioning")
        if normalized == "goal_zero" and not int(
            getattr(self.policy_config, "goal_condition_exact_null", 0)
        ):
            raise RuntimeError("goal_zero requires exact goal-null semantics")
        if normalized.startswith("history_") and not int(
            getattr(self.policy_config, "action_history_enabled", 0)
        ):
            raise RuntimeError("history intervention requires action history")
        if normalized == "history_condition_zero" and not int(
            getattr(
                self.policy_config,
                "action_history_condition_exact_null",
                0,
            )
        ):
            raise RuntimeError(
                "history_condition_zero requires exact history-null semantics"
            )
        self._condition_eval_intervention = normalized
        self._condition_eval_apply_count = 0
        self._condition_eval_metrics = {}

    def clear_condition_eval_intervention(self) -> None:
        self._condition_eval_intervention = None
        self._condition_eval_apply_count = 0
        self._condition_eval_metrics = {}

    def condition_eval_intervention_state(self) -> dict[str, str | int | float]:
        return {
            "mode": (
                "disabled"
                if self._condition_eval_intervention is None
                else self._condition_eval_intervention
            ),
            "apply_count": int(self._condition_eval_apply_count),
            **self._condition_eval_metrics,
        }

    def _intervene_executed_history(self, executed_history: Tensor) -> Tensor:
        mode = self._condition_eval_intervention
        if mode not in {
            "history_zero",
            "history_batch_shuffle",
            "history_truncate",
        }:
            return executed_history
        if mode == "history_zero":
            intervened = torch.zeros_like(executed_history)
        elif mode == "history_batch_shuffle":
            if int(executed_history.shape[0]) > 1:
                intervened = executed_history.roll(shifts=1, dims=0)
            else:
                intervened = executed_history.roll(shifts=1, dims=1)
        else:
            recent = min(
                int(getattr(self.policy_config, "action_history_recent_tokens", 1)),
                int(executed_history.shape[1]),
            )
            if recent >= int(executed_history.shape[1]):
                intervened = executed_history
            else:
                intervened = torch.cat(
                    (
                        torch.zeros_like(executed_history[:, :-recent]),
                        executed_history[:, -recent:],
                    ),
                    dim=1,
                )
        self._condition_eval_metrics["history_input_delta_norm"] = float(
            (intervened - executed_history)
            .detach()
            .float()
            .norm(dim=-1)
            .mean()
            .cpu()
        )
        return intervened

    def _intervene_history_proposal(self, proposal_tokens: Tensor) -> Tensor:
        """Intervene on proposal content without altering direct history memory."""

        mode = self._condition_eval_intervention
        if mode != "history_proposal_batch_shuffle":
            return proposal_tokens
        if int(proposal_tokens.shape[0]) > 1:
            intervened = proposal_tokens.roll(shifts=1, dims=0)
            fallback = 0.0
        else:
            # A one-sample smoke cannot form an episode mismatch. Preserve a
            # non-identity diagnostic by misaligning proposal time, and expose
            # that fallback rather than reporting it as an episode shuffle.
            intervened = proposal_tokens.roll(shifts=1, dims=1)
            fallback = 1.0
        self._condition_eval_metrics[
            "history_proposal_input_delta_norm"
        ] = float(
            (intervened - proposal_tokens)
            .detach()
            .float()
            .norm(dim=-1)
            .mean()
            .cpu()
        )
        self._condition_eval_metrics[
            "history_proposal_shuffle_temporal_fallback"
        ] = fallback
        return intervened

    def _intervene_goal_language(self, tokens: Tensor | None) -> Tensor | None:
        if self._condition_eval_intervention != "goal_batch_shuffle":
            return tokens
        if tokens is None:
            raise RuntimeError("goal shuffle has no active goal-language tensor")
        if int(tokens.shape[0]) > 1:
            intervened = tokens.roll(shifts=1, dims=0)
            fallback = 0.0
        else:
            # A one-sample smoke has no alternate goal. Keep it non-identity,
            # but report that this is a feature mismatch rather than a genuine
            # episode-goal permutation.
            intervened = tokens.roll(
                shifts=max(int(tokens.shape[-1]) // 2, 1),
                dims=-1,
            )
            fallback = 1.0
        self._condition_eval_metrics["goal_input_delta_norm"] = float(
            (intervened - tokens).detach().float().norm(dim=-1).mean().cpu()
        )
        self._condition_eval_metrics["goal_shuffle_feature_fallback"] = fallback
        return intervened

    def _flow_encode(self, physical: Tensor) -> Tensor:
        if self.flow_codec is None:
            return physical
        return self.flow_codec.encode_physical(physical)

    def _flow_decode(self, flow_state: Tensor) -> Tensor:
        if self.flow_codec is None:
            return flow_state
        return self.flow_codec.decode_coefficients(flow_state)

    def _flow_project_state(self, flow_state: Tensor, action_state: Tensor) -> Tensor:
        if self.flow_codec is None:
            return self.codec.project_physical(flow_state, action_state)
        return self.flow_codec.project_state(flow_state, action_state)

    def _flow_velocity_from_output(self, output: dict[str, Tensor]) -> Tensor:
        if self.flow_codec is None:
            return output["pred_physical_velocity"]
        velocity = output.get("pred_velocity_coefficients")
        if not torch.is_tensor(velocity):
            raise RuntimeError(
                "spectral flow is enabled but the action decoder did not return "
                "pred_velocity_coefficients"
            )
        return velocity

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
        executed_memory: Tensor | None = None,
        goal_language_tokens: Tensor | None = None,
        goal_language_mask: Tensor | None = None,
        goal_condition_keep: Tensor | None = None,
        action_history_condition_keep: Tensor | None = None,
        stop_at_midcut: bool = False,
        consequence_physical: Tensor | None = None,
        cvae_target_physical: Tensor | None = None,
        enable_layer_contracts: bool = True,
        enable_final_action_decoder: bool = True,
        collect_diagnostics: bool = True,
        visual_context=None,
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
            executed_memory=executed_memory,
            goal_language_tokens=goal_language_tokens,
            goal_language_mask=goal_language_mask,
            goal_condition_keep=goal_condition_keep,
            action_history_condition_keep=action_history_condition_keep,
            stop_at_midcut=stop_at_midcut,
            consequence_physical=consequence_physical,
            cvae_target_physical=cvae_target_physical,
            enable_layer_contracts=enable_layer_contracts,
            enable_final_action_decoder=enable_final_action_decoder,
            collect_diagnostics=collect_diagnostics,
            visual_context=visual_context,
        )

    def _policy_proposal_tokens(self, tokens: Tensor) -> Tensor:
        """Apply the explicit legacy proposal-gradient compatibility switch."""

        if bool(
            int(getattr(self.policy_config, "action_history_proposal_detach", 1))
        ):
            return tokens.detach()
        return tokens

    @torch.no_grad()
    def build_rollout_target_pack(
        self, visual: Tensor, target_visual: Tensor, *, visual_context=None
    ) -> dict[str, Tensor]:
        pack: dict[str, Tensor] = {}
        if self.planner.flow_dino_evidence is not None:
            if visual_context is None:
                visual_context = self.planner.encode_visual_context(visual)
            if visual_context is None:
                raise RuntimeError("Flow-DINO teacher target is missing its online mask context")
            interval_stage = bool(
                int(
                    getattr(
                        self.policy_config,
                        "flow_jepa_interval_stage_delta",
                        0,
                    )
                )
            )
            if interval_stage:
                interval_targets = (
                    self.planner.flow_jepa_interval_teacher_targets(
                        target_visual,
                        visual,
                    )
                )
                pack.update(
                    {
                        key: value.detach()
                        for key, value in interval_targets.items()
                    }
                )
                window_target = pack["flow_jepa_future_target"]
                stage_target = window_target.new_empty(
                    int(window_target.shape[0]), 0, int(window_target.shape[-1])
                )
            else:
                window_target, stage_target = (
                    self.planner.flow_jepa_teacher_target(
                        target_visual, visual
                    )
                )
                pack["flow_jepa_future_target"] = window_target.detach()
            needs_current_chart = bool(
                int(getattr(self.policy_config, "flow_jepa_late_bottleneck", 0))
                or int(
                    getattr(
                        self.policy_config,
                        "flow_jepa_teacher_balanced_target_mask",
                        0,
                    )
                )
                or int(
                    getattr(
                        self.policy_config,
                        "flow_jepa_predictive_change_contract",
                        0,
                    )
                )
            )
            current_target = None
            if needs_current_chart:
                current_target = (
                    pack["flow_jepa_interval_current_target"]
                    if interval_stage
                    else self.planner.flow_dino_evidence.teacher_current(
                        visual
                    ).detach()
                )
                pack["flow_jepa_current_target"] = current_target
            if int(stage_target.shape[1]) > 0:
                pack["flow_jepa_stage_target"] = stage_target.detach()
            future_target_mask = visual_context.future_target_mask.detach()
            if int(
                getattr(
                    self.policy_config,
                    "flow_jepa_teacher_balanced_target_mask",
                    0,
                )
            ):
                if current_target is None:
                    raise RuntimeError("balanced JEPA target mask requires a current teacher chart")
                future_target_mask, mask_metrics = balanced_future_teacher_mask(
                    window_target.detach(),
                    current_target,
                    future_target_mask,
                    mask_ratio=float(self.policy_config.flow_jepa_mask_ratio),
                    past_fraction=float(
                        self.policy_config.flow_jepa_teacher_mask_past_fraction
                    ),
                    change_fraction=float(
                        self.policy_config.flow_jepa_teacher_mask_change_fraction
                    ),
                )
                pack.update(mask_metrics)
                pack["flow_jepa_teacher_balanced_target_mask"] = window_target.new_ones(
                    (), dtype=torch.float32
                )
            pack["flow_jepa_future_target_mask"] = future_target_mask
        else:
            target = self.planner.target_rollout_effect(visual, target_visual).detach()
            pack.update(
                {
                    "rollout_effect_target": target,
                    "future_latent_target": target,
                    "action_effect_target": target,
                }
            )
        return pack

    def flow_jepa_stage1_forward(
        self,
        visual: Tensor,
        state_history: Tensor,
        executed_history: Tensor,
        state: Tensor,
        target_visual: Tensor,
        *,
        raw_visual: Tensor | None = None,
        goal_language_tokens: Tensor | None = None,
        goal_language_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Run the representation-only V95 Stage1 path.

        Stage1 predicts frozen-DINO window and stage targets from the current
        observation, state, executed-action memory, and optional goal tokens.
        The labelled future action is deliberately absent from this interface:
        neither an action-flow target nor its noised trajectory can leak into
        the representation predictor.  The final action decoder, execution
        controller, and legacy layer-contract tower are also not materialized.
        """

        if self.planner.flow_dino_evidence is None:
            raise RuntimeError("Flow-JEPA Stage1 requested while Flow-DINO JEPA is disabled")
        if target_visual is None:
            raise ValueError("Flow-JEPA Stage1 requires a future visual teacher target")
        batch = int(visual.shape[0])
        neutral_physical = torch.zeros(
            batch,
            int(self.policy_config.action_horizon),
            int(self.policy_config.physical_action_dim),
            device=visual.device,
            dtype=visual.dtype,
        )
        neutral_time = torch.zeros(batch, device=visual.device, dtype=visual.dtype)
        proposal = self.proposal(executed_history)
        goal_language_tokens, goal_language_mask = self._goal_language_batch(
            batch,
            device=executed_history.device,
            dtype=executed_history.dtype,
            tokens=goal_language_tokens,
            mask=goal_language_mask,
        )
        visual_context = self.planner.encode_visual_context(visual, raw_visual=raw_visual)
        output = self._policy_forward(
            neutral_physical,
            neutral_time,
            visual,
            state_history,
            state,
            executed_history,
            self._policy_proposal_tokens(proposal["tokens"]),
            torch.ones(batch, device=visual.device, dtype=visual.dtype),
            executed_memory=proposal["history_tokens"],
            goal_language_tokens=goal_language_tokens,
            goal_language_mask=goal_language_mask,
            consequence_physical=neutral_physical,
            cvae_target_physical=None,
            enable_layer_contracts=False,
            enable_final_action_decoder=False,
            collect_diagnostics=True,
            visual_context=visual_context,
        )
        target_pack = self.build_rollout_target_pack(
            visual,
            target_visual,
            visual_context=visual_context,
        )
        future_prediction = output.get("flow_jepa_future_pred")
        stage_prediction = output.get("flow_jepa_stage_pred")
        if not torch.is_tensor(future_prediction):
            raise RuntimeError("Flow-JEPA Stage1 predictor did not expose future evidence")
        late_bottleneck = bool(
            int(getattr(self.policy_config, "flow_jepa_late_bottleneck", 0))
        )
        if not late_bottleneck and not torch.is_tensor(stage_prediction):
            raise RuntimeError("hierarchical Flow-JEPA Stage1 did not expose its stage prediction")
        teacher_dtype = (
            torch.float32
            if int(
                getattr(
                    self.policy_config,
                    "flow_jepa_interval_stage_delta",
                    0,
                )
            )
            else future_prediction.dtype
        )
        output["flow_jepa_future_target"] = target_pack["flow_jepa_future_target"].to(
            device=future_prediction.device,
            dtype=teacher_dtype,
        )
        output["flow_jepa_future_target_mask"] = target_pack[
            "flow_jepa_future_target_mask"
        ].to(device=future_prediction.device, dtype=torch.bool)
        for key, value in target_pack.items():
            if key.startswith("flow_jepa_teacher_mask_") and torch.is_tensor(value):
                output[key] = value.to(device=future_prediction.device, dtype=torch.float32)
        if "flow_jepa_teacher_balanced_target_mask" in target_pack:
            output["flow_jepa_teacher_balanced_target_mask"] = target_pack[
                "flow_jepa_teacher_balanced_target_mask"
            ].to(device=future_prediction.device, dtype=torch.float32)
        if "flow_jepa_current_target" in target_pack:
            output["flow_jepa_current_target"] = target_pack[
                "flow_jepa_current_target"
            ].to(device=future_prediction.device, dtype=teacher_dtype)
        for key in (
            "flow_jepa_interval_progress_target",
            "flow_jepa_interval_endpoint_target",
        ):
            if key in target_pack:
                output[key] = target_pack[key].to(
                    device=future_prediction.device,
                    dtype=teacher_dtype,
                )
        for key in (
            "flow_jepa_interval_effective_support",
            "flow_jepa_interval_support_count",
        ):
            if key in target_pack:
                output[key] = target_pack[key].to(
                    device=future_prediction.device,
                    dtype=torch.float32,
                )
        if not late_bottleneck:
            assert torch.is_tensor(stage_prediction)
            output["flow_jepa_stage_target"] = target_pack["flow_jepa_stage_target"].to(
                device=stage_prediction.device,
                dtype=torch.float32,
            )
            output["flow_jepa_stage_target_norm"] = (
                output["flow_jepa_stage_target"].norm(dim=-1).mean().detach()
            )
            output["flow_jepa_stage_prediction_norm"] = (
                stage_prediction.detach().float().norm(dim=-1).mean()
            )
        output["flow_jepa_stage1_forward"] = visual.new_ones(())
        output["flow_jepa_stage1_target_action_conditioned"] = visual.new_zeros(())
        return output

    def flow_training_forward(
        self,
        visual: Tensor,
        state_history: Tensor,
        executed_history: Tensor,
        state: Tensor,
        target_action: Tensor,
        *,
        raw_visual: Tensor | None = None,
        action_state: Tensor | None = None,
        target_visual: Tensor | None = None,
        rollout_target_pack: dict[str, Tensor] | None = None,
        future_training_pack: dict[str, Tensor] | None = None,
        proposal_dropout: float | None = None,
        goal_language_tokens: Tensor | None = None,
        goal_language_mask: Tensor | None = None,
        training_noise: Tensor | None = None,
        training_time: Tensor | None = None,
        proposal_keep: Tensor | None = None,
        make_counterfactuals: bool = True,
        stop_at_midcut: bool = False,
    ) -> dict[str, Tensor]:
        del future_training_pack
        proposal = self.proposal(executed_history)
        goal_language_tokens, goal_language_mask = self._goal_language_batch(
            int(executed_history.shape[0]),
            device=executed_history.device,
            dtype=executed_history.dtype,
            tokens=goal_language_tokens,
            mask=goal_language_mask,
        )
        batch_size = int(executed_history.shape[0])
        history_dropout = float(
            getattr(self.policy_config, "action_history_condition_dropout", 0.0)
        )
        if self.training and history_dropout > 0.0:
            history_keep = (
                torch.rand(batch_size, device=executed_history.device) >= history_dropout
            ).to(dtype=executed_history.dtype)
        else:
            history_keep = torch.ones(
                batch_size,
                device=executed_history.device,
                dtype=executed_history.dtype,
            )
        exact_history_null = bool(
            int(
                getattr(
                    self.policy_config,
                    "action_history_condition_exact_null",
                    0,
                )
            )
        )
        conditioned_executed_history = (
            executed_history
            if exact_history_null
            else executed_history * history_keep[:, None, None]
        )
        conditioned_history_memory = (
            proposal["history_tokens"]
            if exact_history_null
            else proposal["history_tokens"] * history_keep[:, None, None]
        )
        # ``proposal.tokens`` are also computed from executed history.  Drop
        # the complete history-derived condition as one branch; otherwise the
        # nominal history intervention could leak through the proposal seed
        # and only test one of two aliases of the same information.
        conditioned_proposal_tokens = proposal["tokens"] * history_keep[:, None, None]
        policy_proposal_tokens = self._policy_proposal_tokens(
            conditioned_proposal_tokens
        )
        goal_dropout = float(getattr(self.policy_config, "goal_condition_dropout", 0.0))
        exact_goal_null = bool(
            int(getattr(self.policy_config, "goal_condition_exact_null", 0))
        )
        if goal_language_tokens is not None and self.training and goal_dropout > 0.0:
            goal_keep = (
                torch.rand(batch_size, device=executed_history.device) >= goal_dropout
            ).to(dtype=goal_language_tokens.dtype)
            conditioned_goal_tokens = (
                goal_language_tokens
                if exact_goal_null
                else goal_language_tokens * goal_keep[:, None, None]
            )
        else:
            goal_keep = torch.ones(
                batch_size,
                device=executed_history.device,
                dtype=executed_history.dtype,
            )
            conditioned_goal_tokens = goal_language_tokens
        # Compile the trainable visual path exactly once.  Self-conditioning
        # and action counterfactuals reuse the same evidence/mask realization,
        # so differences remain attributable to the candidate action.
        visual_context = self.planner.encode_visual_context(visual, raw_visual=raw_visual)
        if self.codec.uses_arm_manifold and action_state is None:
            raise ValueError(
                "manifold_native training requires action_state in action-normalizer coordinates"
            )
        codec_state = state if action_state is None else action_state
        target_physical = self.codec.encode(target_action, codec_state)
        if training_noise is None:
            noise = self.codec.sample_noise(
                target_physical.shape[0],
                device=target_physical.device,
                dtype=target_physical.dtype,
                action_state=codec_state,
            )
        else:
            if tuple(training_noise.shape) != tuple(target_physical.shape):
                raise ValueError("training_noise must match encoded target physical shape")
            noise = training_noise.to(device=target_physical.device, dtype=target_physical.dtype)
        target_flow = self._flow_encode(target_physical)
        noise_flow = self._flow_encode(noise)
        if training_time is None:
            t = torch.rand(
                target_physical.shape[0], device=target_physical.device, dtype=target_physical.dtype
            )
        else:
            if tuple(training_time.shape) != (int(target_physical.shape[0]),):
                raise ValueError("training_time must be [B]")
            t = training_time.to(device=target_physical.device, dtype=target_physical.dtype)
            if bool(((t < 0.0) | (t > 1.0)).any()):
                raise ValueError("training_time must stay in [0,1]")
        noisy_flow = (1 - t[:, None, None]) * target_flow + t[:, None, None] * noise_flow
        noisy_physical = self._flow_decode(noisy_flow)
        target_flow_velocity = noise_flow - target_flow
        target_physical_velocity = self._flow_decode(target_flow_velocity)

        def _bridge_physical(physical: Tensor) -> Tensor:
            flow = self._flow_encode(physical)
            return self._flow_decode((1 - t[:, None, None]) * flow + t[:, None, None] * noise_flow)

        drop = (
            self.policy_config.proposal_dropout
            if proposal_dropout is None
            else float(proposal_dropout)
        )
        if proposal_keep is None:
            keep = (torch.rand(target_physical.shape[0], device=target_physical.device) >= drop).to(
                target_physical.dtype
            )
        else:
            if tuple(proposal_keep.shape) != (int(target_physical.shape[0]),):
                raise ValueError("proposal_keep must be [B]")
            keep = proposal_keep.to(device=target_physical.device, dtype=target_physical.dtype)

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
                    conditioned_executed_history,
                    policy_proposal_tokens,
                    keep,
                    executed_memory=conditioned_history_memory,
                    goal_language_tokens=conditioned_goal_tokens,
                    goal_language_mask=goal_language_mask,
                    goal_condition_keep=goal_keep,
                    action_history_condition_keep=history_keep,
                    stop_at_midcut=stop_at_midcut,
                    consequence_physical=noisy_physical.detach(),
                    cvae_target_physical=None,
                    enable_layer_contracts=False,
                    visual_context=visual_context,
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
            conditioned_executed_history,
            policy_proposal_tokens,
            keep,
            executed_memory=conditioned_history_memory,
            goal_language_tokens=conditioned_goal_tokens,
            goal_language_mask=goal_language_mask,
            goal_condition_keep=goal_keep,
            action_history_condition_keep=history_keep,
            stop_at_midcut=stop_at_midcut,
            consequence_physical=consequence_input,
            cvae_target_physical=target_physical,
            visual_context=visual_context,
        )
        pred_flow_velocity = self._flow_velocity_from_output(action_policy)
        pred_physical_velocity = self._flow_decode(pred_flow_velocity)
        # Project in the active flow chart before exposing a physical view.
        # For spectral flow this keeps training and deployment inside the same
        # coefficient-space affine manifold; IDCT is only a deterministic view
        # used by the existing action/world contracts.
        clean_flow_estimate = self._flow_project_state(
            noisy_flow - t[:, None, None] * pred_flow_velocity,
            codec_state,
        )
        clean_physical_estimate = self._flow_decode(clean_flow_estimate)
        decoded_action = self.codec.decode(clean_physical_estimate, codec_state)
        action_policy["pred_physical_velocity"] = pred_physical_velocity
        if self.flow_codec is not None:
            action_policy["pred_velocity_coefficients"] = pred_flow_velocity
        if "post_pred_velocity" in action_policy:
            post_clean = self.codec.project_physical(
                noisy_physical - t[:, None, None] * action_policy["post_pred_velocity"],
                codec_state,
            )
            action_policy["post_clean_physical_estimate"] = post_clean
            action_policy["post_pred_action_estimate"] = self.codec.decode(post_clean, codec_state)
        out = {
            **action_policy,
            "pred_physical_velocity": pred_physical_velocity,
            "pred_flow_velocity": pred_flow_velocity,
            "target_physical_velocity": target_physical_velocity,
            "target_flow_velocity": target_flow_velocity,
            "target_physical": target_physical,
            "noisy_flow_state": noisy_flow,
            "source_flow_noise": noise_flow,
            "clean_flow_estimate": clean_flow_estimate,
            "clean_physical_estimate": clean_physical_estimate,
            "proposal_action": proposal["action"],
            "time": t,
            "noisy_physical_action": noisy_physical,
            "source_physical_noise": noise,
            "pred_action_estimate": decoded_action,
            "future_conditioned_action_loss": torch.zeros(
                (), device=target_physical.device, dtype=target_physical.dtype
            ),
            "condition_action_history_keep": history_keep.detach().float().mean(),
            "condition_goal_keep": goal_keep.detach().float().mean(),
            "condition_proposal_keep": keep.detach().float().mean(),
        }
        out.update(self.codec.arm_source_diagnostics(noise, codec_state))
        if self.flow_codec is not None:
            with torch.no_grad():
                bridge_roundtrip = self._flow_encode(noisy_physical).float() - noisy_flow.float()
                bridge_null = (
                    self._flow_project_state(noisy_flow, codec_state).float() - noisy_flow.float()
                )
                target_null = (
                    self.flow_codec.project_tangent(target_flow_velocity).float()
                    - target_flow_velocity.float()
                )
                prediction_null = (
                    self.flow_codec.project_tangent(pred_flow_velocity).float()
                    - pred_flow_velocity.float()
                )

                def _energy_fraction(null: Tensor, reference: Tensor) -> Tensor:
                    return null.square().sum() / reference.float().square().sum().clamp_min(1e-8)

                out.update(
                    {
                        "hierarchical_mmdit_spectral_flow_roundtrip_mse": (
                            bridge_roundtrip.square().mean()
                        ),
                        "hierarchical_mmdit_spectral_bridge_null_fraction": (
                            _energy_fraction(bridge_null, noisy_flow)
                        ),
                        "hierarchical_mmdit_spectral_target_tangent_null_fraction": (
                            _energy_fraction(target_null, target_flow_velocity)
                        ),
                        "hierarchical_mmdit_spectral_prediction_tangent_null_fraction": (
                            _energy_fraction(prediction_null, pred_flow_velocity)
                        ),
                    }
                )
        if preview_velocity is not None:
            out["consequence_self_condition"] = torch.ones(
                (), device=target_physical.device, dtype=target_physical.dtype
            )
            out["consequence_self_condition_target_mse"] = (
                (consequence_input.float() - target_physical.detach().float()).square().mean()
            )
            out["consequence_self_condition_noisy_mse"] = (
                (consequence_input.float() - noisy_physical.detach().float()).square().mean()
            )
            out["consequence_preview_flow"] = (
                (preview_velocity.float() - target_physical_velocity.detach().float())
                .square()
                .mean()
            )
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
            pack = self.build_rollout_target_pack(
                visual, target_visual, visual_context=visual_context
            )

        def _merge_layer_contract_counterfactuals(variant: dict[str, Tensor], suffix: str) -> None:
            base_layers = out.get("layer_contracts")
            variant_layers = variant.get("layer_contracts")
            if not isinstance(base_layers, list) or not isinstance(variant_layers, list):
                return
            for base_entry, var_entry in zip(base_layers, variant_layers):
                for key in (
                    "rollout_effect_pred",
                    "rollout_delta_pred",
                    "rollout_base_effect_pred",
                    "milestone_step_delta_pred",
                    "causal_rollout_effect_pred",
                    "causal_rollout_delta_pred",
                    "policy_effect_tokens",
                    "unified_intervention_latent_pred",
                    "neutral_latent_pred",
                    "rollout_effect_pred_shuffle_state",
                    "rollout_delta_pred_shuffle_state",
                    "milestone_step_delta_pred_shuffle_state",
                ):
                    if key in var_entry:
                        base_entry[f"{key}_{suffix}"] = var_entry[key]

        if pack is not None:
            if "rollout_effect_target" in pack:
                target = pack["rollout_effect_target"].to(
                    device=target_physical.device,
                    dtype=action_policy["rollout_effect_pred"].dtype,
                )
                out["rollout_effect_target"] = target
                out["future_latent_target"] = target
                out["future_latent_velocity_target"] = target
                out["action_effect_target"] = target
            if "flow_jepa_future_target" in pack:
                teacher_dtype = (
                    torch.float32
                    if int(
                        getattr(
                            self.policy_config,
                            "flow_jepa_interval_stage_delta",
                            0,
                        )
                    )
                    else action_policy["flow_jepa_future_pred"].dtype
                )
                out["flow_jepa_future_target"] = pack["flow_jepa_future_target"].to(
                    device=target_physical.device,
                    dtype=teacher_dtype,
                )
                out["flow_jepa_future_target_mask"] = pack[
                    "flow_jepa_future_target_mask"
                ].to(device=target_physical.device, dtype=torch.bool)
                for key, value in pack.items():
                    if key.startswith("flow_jepa_teacher_mask_") and torch.is_tensor(value):
                        out[key] = value.to(device=target_physical.device, dtype=torch.float32)
                if "flow_jepa_teacher_balanced_target_mask" in pack:
                    out["flow_jepa_teacher_balanced_target_mask"] = pack[
                        "flow_jepa_teacher_balanced_target_mask"
                    ].to(device=target_physical.device, dtype=torch.float32)
                if "flow_jepa_current_target" in pack:
                    out["flow_jepa_current_target"] = pack[
                        "flow_jepa_current_target"
                    ].to(
                        device=target_physical.device,
                        dtype=teacher_dtype,
                    )
                for key in (
                    "flow_jepa_interval_progress_target",
                    "flow_jepa_interval_endpoint_target",
                ):
                    if key in pack:
                        out[key] = pack[key].to(
                            device=target_physical.device,
                            dtype=teacher_dtype,
                        )
                for key in (
                    "flow_jepa_interval_effective_support",
                    "flow_jepa_interval_support_count",
                ):
                    if key in pack:
                        out[key] = pack[key].to(
                            device=target_physical.device,
                            dtype=torch.float32,
                        )
                if "flow_jepa_stage_target" in pack:
                    out["flow_jepa_stage_target"] = pack["flow_jepa_stage_target"].to(
                        device=target_physical.device,
                        dtype=torch.float32,
                    )
                    out["flow_jepa_stage_target_norm"] = out[
                        "flow_jepa_stage_target"
                    ].norm(dim=-1).mean().detach()
                    out["flow_jepa_stage_prediction_norm"] = action_policy[
                        "flow_jepa_stage_pred"
                    ].detach().float().norm(dim=-1).mean()
            if make_counterfactuals:
                hold_action = codec_state[:, None].expand_as(target_action)
                hold_physical = self.codec.encode(hold_action, codec_state)
                hold_noisy = _bridge_physical(hold_physical)
                hold_policy = self._policy_forward(
                    hold_noisy.detach(),
                    t.detach(),
                    visual,
                    state_history,
                    state,
                    conditioned_executed_history,
                    policy_proposal_tokens,
                    keep,
                    executed_memory=conditioned_history_memory,
                    goal_language_tokens=conditioned_goal_tokens,
                    goal_language_mask=goal_language_mask,
                    goal_condition_keep=goal_keep,
                    action_history_condition_keep=history_keep,
                    stop_at_midcut=stop_at_midcut,
                    consequence_physical=hold_physical,
                    enable_final_action_decoder=False,
                    visual_context=visual_context,
                )
                out["rollout_effect_pred_hold_action"] = hold_policy["rollout_effect_pred"]
                out["rollout_delta_pred_hold_action"] = hold_policy["rollout_delta_pred"]
                out["rollout_base_effect_pred_hold_action"] = hold_policy[
                    "rollout_base_effect_pred"
                ]
                if "midcut_rollout_effect_pred" in hold_policy:
                    out["midcut_rollout_effect_pred_hold_action"] = hold_policy[
                        "midcut_rollout_effect_pred"
                    ]
                    out["midcut_rollout_delta_pred_hold_action"] = hold_policy[
                        "midcut_rollout_delta_pred"
                    ]
                    out["midcut_rollout_base_effect_pred_hold_action"] = hold_policy[
                        "midcut_rollout_base_effect_pred"
                    ]
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
                        .reshape(
                            batch * batch, int(target_action.shape[1]), int(target_action.shape[2])
                        )
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
                        (cand_physical.detach().float() - target_physical.detach().float()[:, None])
                        .flatten(2)
                        .norm(dim=-1)
                    )
                    eye = torch.eye(batch, device=dist.device, dtype=torch.bool)
                    dist = dist.masked_fill(eye, -1.0)
                    perm = dist.argmax(dim=1)
                    shuffle_physical = cand_physical[
                        torch.arange(batch, device=target_physical.device), perm
                    ]
                else:
                    shuffle_physical = target_physical
                shuffle_noisy = _bridge_physical(shuffle_physical)
                shuffle_policy = self._policy_forward(
                    shuffle_noisy.detach(),
                    t.detach(),
                    visual,
                    state_history,
                    state,
                    conditioned_executed_history,
                    policy_proposal_tokens,
                    keep,
                    executed_memory=conditioned_history_memory,
                    goal_language_tokens=conditioned_goal_tokens,
                    goal_language_mask=goal_language_mask,
                    goal_condition_keep=goal_keep,
                    action_history_condition_keep=history_keep,
                    stop_at_midcut=stop_at_midcut,
                    consequence_physical=shuffle_physical,
                    enable_final_action_decoder=False,
                    visual_context=visual_context,
                )
                out["rollout_effect_pred_shuffle_action"] = shuffle_policy["rollout_effect_pred"]
                out["rollout_delta_pred_shuffle_action"] = shuffle_policy["rollout_delta_pred"]
                out["rollout_base_effect_pred_shuffle_action"] = shuffle_policy[
                    "rollout_base_effect_pred"
                ]
                if "midcut_rollout_effect_pred" in shuffle_policy:
                    out["midcut_rollout_effect_pred_shuffle_action"] = shuffle_policy[
                        "midcut_rollout_effect_pred"
                    ]
                    out["midcut_rollout_delta_pred_shuffle_action"] = shuffle_policy[
                        "midcut_rollout_delta_pred"
                    ]
                    out["midcut_rollout_base_effect_pred_shuffle_action"] = shuffle_policy[
                        "midcut_rollout_base_effect_pred"
                    ]
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
        raw_visual: Tensor | None = None,
        action_state: Tensor | None = None,
        steps: int | None = None,
        noise: Tensor | None = None,
        use_proposal: bool = True,
        return_event_logits: bool = False,
        stop_at_midcut: bool = False,
        collect_diagnostics: bool | None = None,
        goal_language_tokens: Tensor | None = None,
        goal_language_mask: Tensor | None = None,
    ) -> Tensor | dict[str, Tensor]:
        """Sample an action chunk without teacher-forcing target actions.

        ``stop_at_midcut`` is used for contract-stage validation.  It evaluates
        the deployable mid-cut simple heads by iterative denoising from noise;
        unlike ``flow_training_forward`` it never receives ``target_action`` and
        therefore cannot leak validation labels into action metrics.
        """
        condition_mode = self._condition_eval_intervention
        if condition_mode is not None and self.training:
            raise RuntimeError("condition intervention is evaluation-only")
        model_executed_history = self._intervene_executed_history(executed_history)
        if condition_mode not in {None, "none"}:
            self._condition_eval_apply_count += 1
        proposal = self.proposal(model_executed_history)
        proposal_tokens = self._intervene_history_proposal(
            proposal["tokens"]
        )
        goal_language_tokens, goal_language_mask = self._goal_language_batch(
            int(model_executed_history.shape[0]),
            device=model_executed_history.device,
            dtype=model_executed_history.dtype,
            tokens=goal_language_tokens,
            mask=goal_language_mask,
        )
        goal_language_tokens = self._intervene_goal_language(
            goal_language_tokens
        )
        goal_condition_keep = torch.ones(
            int(model_executed_history.shape[0]),
            device=model_executed_history.device,
            dtype=model_executed_history.dtype,
        )
        action_history_condition_keep = torch.ones_like(goal_condition_keep)
        if condition_mode == "goal_zero":
            goal_condition_keep = torch.zeros_like(goal_condition_keep)
            self._condition_eval_metrics["goal_condition_keep_delta"] = 1.0
        elif condition_mode in {"history_zero", "history_condition_zero"}:
            action_history_condition_keep = torch.zeros_like(
                action_history_condition_keep
            )
            self._condition_eval_metrics["history_condition_keep_delta"] = 1.0
        visual_context = self.planner.encode_visual_context(visual, raw_visual=raw_visual)
        steps = int(steps or self.policy_config.inference_steps)
        if steps <= 0:
            raise ValueError("steps must be positive")
        if self.codec.uses_arm_manifold and action_state is None:
            raise ValueError(
                "manifold_native sampling requires action_state in action-normalizer coordinates"
            )
        codec_state = (state if action_state is None else action_state).to(
            device=visual.device,
            dtype=visual.dtype,
        )
        if noise is None:
            x_flow = (
                self.flow_codec.sample_noise(
                    visual.shape[0],
                    device=visual.device,
                    dtype=visual.dtype,
                    action_state=codec_state,
                )
                if self.flow_codec is not None
                else self.codec.sample_noise(
                    visual.shape[0],
                    device=visual.device,
                    dtype=visual.dtype,
                    action_state=codec_state,
                )
            )
        else:
            physical_noise = noise.clone()
            if physical_noise.shape[-1] == self.policy_config.action_dim:
                physical_noise = self.codec.encode(
                    physical_noise.to(device=visual.device, dtype=visual.dtype),
                    codec_state,
                )
            elif physical_noise.shape[-1] != self.policy_config.physical_action_dim:
                raise ValueError("noise must have last dim action_dim or physical_action_dim")
            else:
                physical_noise = physical_noise.to(device=visual.device, dtype=visual.dtype)
            x_flow = self._flow_encode(physical_noise)
        x_flow = self._flow_project_state(x_flow, codec_state)
        x_physical = self._flow_decode(x_flow)
        keep = torch.full(
            (visual.shape[0],),
            1.0 if use_proposal else 0.0,
            device=visual.device,
            dtype=visual.dtype,
        )
        if condition_mode == "history_proposal_zero":
            # Null only proposal content. UnifiedCanvasSeed deliberately keeps
            # the proposal slot/type template, just as other exact condition
            # nulls preserve their role identity.
            original_keep = keep
            keep = torch.zeros_like(keep)
            self._condition_eval_metrics[
                "history_proposal_keep_delta"
            ] = float(
                (keep - original_keep).detach().float().abs().mean().cpu()
            )
        use_self_condition = (
            int(getattr(self.policy_config, "action_consequence_self_condition", 0))
            and int(getattr(self.policy_config, "layer_recurrent_consequence", 0))
            and int(getattr(self.policy_config, "layer_contract_adapters", 0))
        )
        if collect_diagnostics is None:
            # Diagnostics are only observable on the dictionary return path.
            # The historical tensor-only path computed and discarded thousands
            # of scalar reductions at every integration step.
            collect_diagnostics = bool(return_event_logits)
        source_diagnostics = (
            self.codec.arm_source_diagnostics(x_physical, codec_state)
            if collect_diagnostics
            else {}
        )
        sample_diagnostic_keys: tuple[str, ...] | None = None
        sample_diagnostic_sum: Tensor | None = None
        sample_diagnostic_count = 0
        evidence_sampling = (
            getattr(self.planner, "evidence_latent_mmdit_action_decoder", None) is not None
        )
        for index in range(steps, 0, -1):
            t = torch.full(
                (visual.shape[0],),
                float(index) / float(steps),
                device=visual.device,
                dtype=visual.dtype,
            )
            x_physical = self._flow_decode(x_flow)
            consequence_input = x_physical
            if use_self_condition:
                preview = self._policy_forward(
                    x_physical,
                    t,
                    visual,
                    state_history,
                    state,
                    model_executed_history,
                    proposal_tokens,
                    keep,
                    executed_memory=proposal["history_tokens"],
                    goal_language_tokens=goal_language_tokens,
                    goal_language_mask=goal_language_mask,
                    goal_condition_keep=goal_condition_keep,
                    action_history_condition_keep=action_history_condition_keep,
                    stop_at_midcut=stop_at_midcut,
                    consequence_physical=x_physical,
                    enable_layer_contracts=False,
                    collect_diagnostics=False,
                    visual_context=visual_context,
                )
                consequence_input = (
                    x_physical - t[:, None, None] * preview["pred_physical_velocity"]
                ).detach()
                del preview
            out = self._policy_forward(
                x_physical,
                t,
                visual,
                state_history,
                state,
                model_executed_history,
                proposal_tokens,
                keep,
                executed_memory=proposal["history_tokens"],
                goal_language_tokens=goal_language_tokens,
                goal_language_mask=goal_language_mask,
                goal_condition_keep=goal_condition_keep,
                action_history_condition_keep=action_history_condition_keep,
                stop_at_midcut=stop_at_midcut,
                consequence_physical=consequence_input,
                enable_layer_contracts=False,
                collect_diagnostics=bool(collect_diagnostics),
                visual_context=visual_context,
            )
            pred_flow_velocity = self._flow_velocity_from_output(out)
            raw_next = x_flow - pred_flow_velocity / float(steps)
            projected_next = self._flow_project_state(raw_next, codec_state)
            if collect_diagnostics:
                diagnostic_items: list[tuple[str, Tensor]] = []
                for key, value in out.items():
                    keep_diagnostic = _keep_sampling_diagnostic(
                        key, evidence_active=evidence_sampling
                    )
                    if keep_diagnostic and torch.is_tensor(value) and value.numel() == 1:
                        diagnostic_items.append((key, value.detach().float().reshape(())))

                # Pre-projection null drift. ``*_rate`` is the step-size-
                # normalized absolute field norm, not a fraction.
                null_drift = (raw_next - projected_next).detach().float()
                raw_update = (raw_next - x_flow).detach().float()
                arm_span = 2 * int(self.codec.arm_dim)
                arm_null_norm = null_drift[..., :arm_span].norm(dim=-1).mean()
                grip_null_norm = null_drift[..., arm_span:].norm(dim=-1).mean()
                arm_update_norm = raw_update[..., :arm_span].norm(dim=-1).mean()
                grip_update_norm = raw_update[..., arm_span:].norm(dim=-1).mean()
                diagnostic_items.extend(
                    (
                        ("arm_null_preproject", arm_null_norm),
                        ("arm_null_preproject_rate", arm_null_norm * float(steps)),
                        (
                            "arm_null_preproject_fraction",
                            arm_null_norm / arm_update_norm.clamp_min(1e-8),
                        ),
                        ("arm_preproject_update_norm", arm_update_norm),
                        ("grip_null_preproject", grip_null_norm),
                        ("grip_null_preproject_rate", grip_null_norm * float(steps)),
                        (
                            "grip_null_preproject_fraction",
                            grip_null_norm / grip_update_norm.clamp_min(1e-8),
                        ),
                        ("grip_preproject_update_norm", grip_update_norm),
                    )
                )
                row_keys = tuple(key for key, _ in diagnostic_items)
                if sample_diagnostic_keys is None:
                    sample_diagnostic_keys = row_keys
                elif row_keys != sample_diagnostic_keys:
                    raise RuntimeError(
                        "sampling diagnostic schema changed between integration steps"
                    )
                row = torch.stack([value for _, value in diagnostic_items])
                sample_diagnostic_sum = (
                    row if sample_diagnostic_sum is None else sample_diagnostic_sum + row
                )
                sample_diagnostic_count += 1
            x_flow = projected_next
        final_physical = self._flow_decode(x_flow)
        action = self.codec.decode(final_physical, codec_state)
        if return_event_logits:
            zero_t = torch.zeros((visual.shape[0],), device=visual.device, dtype=visual.dtype)
            event = self._policy_forward(
                final_physical,
                zero_t,
                visual,
                state_history,
                state,
                model_executed_history,
                proposal_tokens,
                keep,
                executed_memory=proposal["history_tokens"],
                goal_language_tokens=goal_language_tokens,
                goal_language_mask=goal_language_mask,
                goal_condition_keep=goal_condition_keep,
                action_history_condition_keep=action_history_condition_keep,
                stop_at_midcut=stop_at_midcut,
                enable_layer_contracts=False,
                collect_diagnostics=False,
                visual_context=visual_context,
            )
            result = {
                "action": action,
                "physical_action": final_physical,
                "event_logits": event["event_logits"],
                "motion_logits": event["motion_logits"],
            }
            for key, value in source_diagnostics.items():
                result[f"sample_{key}"] = value
            if sample_diagnostic_sum is not None and sample_diagnostic_keys is not None:
                diagnostic_mean = sample_diagnostic_sum / float(max(sample_diagnostic_count, 1))
                for diagnostic_index, key in enumerate(sample_diagnostic_keys):
                    result[f"sample_{key}"] = diagnostic_mean[diagnostic_index]
            return result
        return action

    def parameter_report(self) -> dict[str, int]:
        report = {
            "history_proposal": sum(p.numel() for p in self.proposal.parameters()),
            "physical_action_codec": sum(p.numel() for p in self.codec.parameters()),
            "midcut_contract_heads": sum(p.numel() for p in self.planner.midcut_heads.parameters()),
            "layer_contract_adapters": sum(
                p.numel() for p in self.planner.layer_contract_heads.parameters()
            ),
            "layer_shared_fm_probe": (
                0
                if self.planner.layer_fm_probe is None
                else sum(p.numel() for p in self.planner.layer_fm_probe.parameters())
            ),
            "layer_recurrent_consequence": (
                0
                if self.planner.layer_consequence_cell is None
                else sum(p.numel() for p in self.planner.layer_consequence_cell.parameters())
            ),
            "layer_causal_effect_branch": (
                0
                if self.planner.layer_consequence_cell is None
                else sum(p.numel() for p in self.planner.layer_consequence_cell.parameters())
            ),
            "layer_role_scheduler": sum(
                p.numel() for p in self.planner.layer_role_scheduler.parameters()
            ),
            "residual_action_flow_denoiser": (
                0
                if self.planner.residual_action_flow_denoiser is None
                else sum(p.numel() for p in self.planner.residual_action_flow_denoiser.parameters())
            ),
            "latent_main_action_decoder": (
                0
                if getattr(self.planner, "latent_main_action_decoder", None) is None
                else sum(p.numel() for p in self.planner.latent_main_action_decoder.parameters())
            ),
            "latent_cvae_action_decoder": (
                0
                if getattr(self.planner, "latent_cvae_action_decoder", None) is None
                else sum(p.numel() for p in self.planner.latent_cvae_action_decoder.parameters())
            ),
            "hierarchical_mmdit_action_decoder": (
                0
                if getattr(self.planner, "hierarchical_mmdit_action_decoder", None) is None
                else sum(
                    p.numel() for p in self.planner.hierarchical_mmdit_action_decoder.parameters()
                )
            ),
            "evidence_latent_mmdit_action_decoder": (
                0
                if getattr(self.planner, "evidence_latent_mmdit_action_decoder", None) is None
                else sum(
                    p.numel()
                    for p in self.planner.evidence_latent_mmdit_action_decoder.parameters()
                )
            ),
            "staged_midcut_dit": sum(p.numel() for p in self.planner.parameters()),
        }
        report["total"] = sum(p.numel() for p in self.parameters())
        report["trainable"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return report
