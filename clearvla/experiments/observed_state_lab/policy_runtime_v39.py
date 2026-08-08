"""Training/evaluation runtime for V39 staged mid-cut latent contract policy."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from clearvla.experiments.classic_policy_lab.normalizer import ArrayNormalizer
from clearvla.experiments.classic_policy_lab.rdt2_conditioning import RDT2Conditioner
from clearvla.experiments.dynamic_world_lab.shared_runtime import (
    encode_current_tokens,
    gripper_transition_metrics,
)
from clearvla.policy.differential_intent_effect import (
    DifferentialStatelessIntentController,
)
from clearvla.policy.gauges import masked_candidate_center
from clearvla.policy.grounded_intent_effect import (
    GROUNDING_MANIFEST,
    StatelessIntentOrganizer,
    manifest_from_mapping,
)
from clearvla.policy.object_intent_dynamics_323 import (
    manifest_from_mapping as object_intent_manifest_from_mapping,
)
from clearvla.policy.system import V39PolicySystem

from .policy_runtime_v36_3 import (
    V363PolicyTrainerConfig,
    _normalized_event_emphasis,
    arm_motion_labels,
    balanced_score,
    decode,
    event_head_metrics,
    gripper_event_labels,
    is_deploy_eligible,
    position_weights,
    semantic_physical_velocity_error,
)
from .policy_runtime_v36_3 import flow_losses as v363_flow_losses
from .world_runtime import autocast_context, jsonable, scheduler

POLICY_CHECKPOINT_SCHEMAS = frozenset(
    {
        "clearvla-v39-policy-checkpoint-v1",
        "clearvla-v40-policy-checkpoint-v1",
    }
)


def _operation_candidate_error_field(
    system: V39PolicySystem,
    residual: Tensor,
    sample: dict[str, Tensor],
    trainer: V39PolicyTrainerConfig,
) -> Tensor:
    """Per-horizon arm/gripper value field in the physical training metric."""
    if residual.ndim != 5:
        raise ValueError(
            f"operation candidate residual must be [B,S,C,T,P], got {tuple(residual.shape)}"
        )
    cfg = system.policy_config
    batch, steps, candidates, horizon, physical_dim = residual.shape
    ad = int(cfg.arm_dim)
    gf = int(cfg.gripper_field_dim)
    flat = residual.reshape(-1, horizon, physical_dim)
    if system.codec.uses_arm_manifold:
        arm_native, _, arm_null = system.codec.project_arm_tangent(flat[..., : 2 * ad])
        arm_error = (
            arm_native.square()
            + max(float(trainer.arm_manifold_null_weight), 0.0)
            * 0.5
            * (arm_null[..., :ad].square() + arm_null[..., ad : 2 * ad].square())
        ).sum(dim=-1)
    else:
        arm_error = 0.5 * (flat[..., :ad].square() + flat[..., ad : 2 * ad].square()).sum(dim=-1)
    grip_field = flat[..., 2 * ad : 2 * ad + gf]
    if system.codec.uses_parseval_gripper_field:
        native = system.codec.decode_gripper_field(grip_field)
        null = grip_field - system.codec.project_gripper_field(grip_field)
        grip_error = native[..., 0].square() + null.square().sum(dim=-1)
    else:
        grip_error = grip_field.square().mean(dim=-1)
    grip_error = grip_error.reshape(batch, steps, candidates, horizon)
    labels = gripper_event_labels(
        target_raw=sample["policy_action_raw"].to(device=residual.device),
        current_raw=sample["state_raw"].to(device=residual.device),
        gripper_index=cfg.gripper_index,
        threshold=trainer.gripper_event_threshold,
    )
    event_mask = labels.ne(0).to(dtype=grip_error.dtype)
    event_mask = (
        event_mask[:, None, None].expand(batch, steps, candidates, horizon).reshape(-1, horizon)
    )
    position_weight = position_weights(cfg, trainer, residual.device).to(dtype=grip_error.dtype)
    event_emphasis, _, _, _ = _normalized_event_emphasis(
        event_mask,
        position_weight,
        max(float(trainer.gripper_fm_event_boost), 0.0),
    )
    grip_error = grip_error * event_emphasis.reshape(batch, steps, candidates, horizon)
    arm_error = arm_error.reshape(batch, steps, candidates, horizon) / float(ad)
    return torch.stack([arm_error, grip_error], dim=-1)


@dataclass(frozen=True)
class V39PolicyTrainerConfig(V363PolicyTrainerConfig):
    # V39 staged mid-cut latent-contract objectives. Future
    # target tokens are targets only; no future-noisy latent is fed to the model.
    rollout_dynamics_loss_weight: float = 0.03
    # ``rollout_delta`` duplicates the milestone delta target below.  Keep the
    # old field loadable, but do not double-count one target by default.
    rollout_delta_loss_weight: float = 0.0
    rollout_contrast_loss_weight: float = 0.06
    rollout_contrast_margin: float = 0.02
    # Main controlled rollout must match not only the future direction but also
    # its energy and inter-milestone change. These mirror the proven layer
    # consequence contracts and prevent a low-amplitude conditional mean from
    # satisfying the world objective.
    rollout_variance_loss_weight: float = 0.05
    rollout_norm_loss_weight: float = 0.02
    rollout_milestone_delta_match_weight: float = 0.15
    # Local fuse for the complete CVAE/MMDiT decoder. Architectural pre-norm is
    # the primary protection; this prevents a decoder-only spike from consuming
    # the global clipping budget and starving the world/rollout trunk.
    latent_cvae_grad_clip: float = 0.0
    # Kept for CLI/checkpoint compatibility; defaults disabled to avoid the
    # future self-denoise shortcut.
    future_latent_loss_weight: float = 0.0
    action_effect_loss_weight: float = 0.0
    future_latent_loss_start_epoch: int = 1
    future_latent_max_batches: int = 0
    # V95 future-only JEPA and semantic patch-correspondence objectives.  These
    # names stay separate from action ``physical_flow`` so logs cannot confuse
    # optical patch motion with the policy's flow-matching field.
    flow_jepa_future_loss_weight: float = 0.0
    # Legacy continuous future-change auxiliary for absolute-prediction runs.
    # The predictive-change contract makes delta prediction primary and
    # therefore requires this duplicate weight to be zero.
    flow_jepa_future_change_loss_weight: float = 0.0
    # ``global`` preserves V100: target-change magnitudes across every offset
    # share one denominator. ``per_horizon`` first normalizes each real offset
    # and then averages them, preventing +48 from swallowing +4/+12/+24.
    flow_jepa_horizon_balance_mode: str = "global"
    # V105 retains an unnormalized delta anchor and applies relative
    # normalization only where the frozen teacher change is reliable. This
    # prevents tiny teacher jitter from dominating the normalized objective.
    flow_jepa_future_reliable_normalization: int = 0
    # Teacher change supervises only the observation-derived horizon address
    # posterior. It is never an input to the forward address reader.
    flow_jepa_horizon_address_loss_weight: float = 0.0
    # V106 supervises the bounded W->P interval-stage delta with signed
    # progression and a weak endpoint-consistency term.  The target is
    # teacher-only and spatially aligned; zero preserves the V105 objective.
    flow_jepa_interval_stage_loss_weight: float = 0.0
    flow_jepa_stage_loss_weight: float = 0.0
    flow_jepa_warp_loss_weight: float = 0.0
    flow_jepa_identity_advantage_loss_weight: float = 0.0
    flow_jepa_static_identity_loss_weight: float = 0.0
    flow_jepa_cycle_loss_weight: float = 0.0
    flow_jepa_smoothness_loss_weight: float = 0.0
    flow_jepa_uncertainty_nll_weight: float = 0.0
    flow_jepa_refinement_sequence_loss_weight: float = 0.0
    flow_jepa_lr_scale: float = 1.0

    # Full action/event validation still covers every validation batch. These
    # budgets apply only to the expensive sampling gauge fan-out and the second
    # no-proposal denoising trajectory. Positive budgets are spread uniformly
    # over the loader; zero preserves the historical all-batch diagnostics.
    eval_sampling_diagnostic_batches: int = 16
    eval_proposal_ablation_batches: int = 16
    # Matched-noise structural probes for the Evidence execution plane.
    eval_execution_ablation_batches: int = 8
    # Teacher-forced representation validation is separate from deploy action
    # sampling and is spread across a bounded subset by default.
    eval_representation_batches: int = 16

    # Lightweight CUDA memory accounting. Disabled by default. Set
    # --memory-report-every N to emit a [cuda-mem] line and append JSONL rows
    # every N batches. Set --memory-report-detail 1 to also log stage-level
    # points inside the selected batch. This is monitoring only; it does not
    # change the V38.3 throughput-optimized training/eval semantics.
    memory_report_every: int = 0
    memory_report_detail: int = 0
    memory_report_sync: int = 0

    # Staging.  contract = stop at Z_mid and train only the pre-cut trunk plus
    # simple contract heads.  policy = train the full policy tail, with the
    # pre-cut trunk updated at a lower learning rate and an optional decaying
    # mid-cut preservation loss.
    training_stage: str = "contract"
    upper_lr_scale: float = 0.20
    # Scratch single-stage role hierarchies must not inherit the old Stage2
    # "protect the upper trunk" learning-rate ladder.
    single_stage_role_lr: int = 0
    midcut_head_lr_scale: float = 1.0
    midcut_aux_loss_weight: float = 0.05
    midcut_aux_final_ratio: float = 0.20
    midcut_aux_decay_epochs: int = 4
    midcut_rollout_dynamics_loss_weight: float = 0.03
    midcut_rollout_delta_loss_weight: float = 0.0
    midcut_rollout_contrast_loss_weight: float = 0.03

    # Optional safe residual action-flow denoiser.  This module is random at
    # first load but zero-starts behaviorally, so it gets a distinct LR group
    # in policy stage instead of being hidden inside the large trunk group.
    action_flow_residual_lr_scale: float = 1.5
    latent_action_decoder_lr_scale: float = 1.5
    # V42 compact latent-CVAE head.  Small KL keeps q(z|latent,target) close
    # to the conditional prior used at inference without letting KL dominate
    # the action flow/chunk reconstruction objective.
    latent_cvae_action_decoder_lr_scale: float = 1.0
    # The complete V77 host block uses the ordinary decoder rate. Only
    # orthogonal sidecar bases and depth predictors use this rate.
    hierarchical_mmdit_contraction_lr_scale: float = 2.0
    hierarchical_mmdit_shared_base_lr_scale: float = 1.0
    hierarchical_mmdit_controller_lr_scale: float = 1.0
    # Weak cost on the selected nested depth.  It begins only after the exact
    # identity warm-up and cannot change basis scale or residual amplitude.
    hierarchical_mmdit_depth_usage_loss_weight: float = 2e-4
    # Target-aware supervision is confined to the training loss. Candidate
    # errors are detached, so only the read-only exit controller receives it.
    hierarchical_mmdit_oracle_route_loss_weight: float = 0.0
    hierarchical_mmdit_oracle_route_relative_tolerance: float = 0.0
    hierarchical_mmdit_oracle_route_warmup_steps: int = 200
    # Candidate-operation supervision.  The target is the detached marginal
    # improvement of each legal operation, not an absolute dwell value.
    hierarchical_mmdit_operation_route_loss_weight: float = 0.0
    hierarchical_mmdit_operation_route_temperature: float = 0.5
    hierarchical_mmdit_operation_route_warmup_steps: int = 0
    # V88 candidate-relative physical value regression. The loss is active
    # during the fixed-path cold start; only execution waits for policy-config
    # operation_value_warmup_steps.
    hierarchical_mmdit_operation_value_loss_weight: float = 0.0
    hierarchical_mmdit_operation_value_huber_delta: float = 0.1
    # Zero selects a detached per-batch spread calibration. A positive value is
    # an explicitly calibrated physical-value scale, never a hard threshold.
    hierarchical_mmdit_operation_value_reliability_scale: float = 0.0
    # Legacy names remain loadable for old experiment manifests, but are not
    # consumed by the unified controller path.
    hierarchical_mmdit_dwell_value_loss_weight: float = 0.0
    hierarchical_mmdit_dwell_value_warmup_steps: int = 200
    hierarchical_mmdit_dwell_compute_cost: float = 0.0
    # Native-time evidence decoder execution supervision. Physical candidate
    # errors are detached targets; the attached value context keeps the natural
    # multi-step gradient through controller/evidence/action state.
    latent_cvae_mmdit_execution_value_loss_weight: float = 0.0
    # ``flow_losses`` also needs the policy mode. Keep a trainer-side copy so
    # the CLI cannot silently make a learned decoder look like ``fixed`` when
    # deciding whether candidate-value supervision is active.
    latent_cvae_mmdit_dwell_mode: str = "fixed"
    # Compatibility-only field. The value reader trains from the first batch;
    # execution_warmup_steps controls when selection may leave candidate one.
    latent_cvae_mmdit_execution_value_warmup_steps: int = 200
    # Retained for old checkpoints/configs; native execution cost is audit-only
    # and is intentionally not added to the flow loss.
    latent_cvae_mmdit_execution_compute_cost: float = 0.01
    # V42.1: train the deploy/inference prior path directly.  The posterior path
    # remains a weak auxiliary reconstruction target instead of being allowed to
    # carry the main policy loss by looking at the target chunk.
    latent_cvae_kl_weight: float = 5e-4
    latent_cvae_posterior_recon_weight: float = 0.25
    latent_cvae_legacy_anchor_weight: float = 0.03
    latent_cvae_legacy_anchor_decay_steps: int = 2500
    latent_cvae_legacy_anchor_min_weight: float = 0.0
    latent_cvae_adaptive_regularizer_weight: float = 0.002
    latent_cvae_adaptive_route_entropy_weight: float = 0.001
    latent_cvae_micro_supervision_weight: float = 0.08
    latent_cvae_micro_event_weight: float = 0.02
    latent_cvae_micro_monotonic_weight: float = 0.02
    latent_cvae_micro_weight_kl_weight: float = 0.001
    latent_cvae_micro_coverage_smooth_weight: float = 0.002
    latent_cvae_micro_coverage_floor_weight: float = 0.002
    latent_cvae_micro_coverage_prior_logit_scale: float = 0.25
    latent_cvae_micro_coverage_floor_ratio: float = 0.55
    latent_cvae_micro_learned_weight_max: float = 0.40
    latent_cvae_micro_learned_ramp_steps: int = 2000
    latent_cvae_micro_weight_floor: float = 0.05
    latent_cvae_micro_event_positive_weight: float = 2.0

    # V39.1. contract_mode=layer_adapter keeps the full DiT active in Stage 1
    # and supervises tiny side adapters at every block instead of stopping at a
    # single hard midcut. Final policy loss is optional and weak in Stage 1.
    contract_mode: str = "midcut"
    layer_contract_loss_weight: float = 1.0
    layer_contract_final_action_loss_weight: float = 0.0
    layer_contract_final_action_lr_scale: float = 0.30
    layerwise_lr_min_scale: float = 0.30
    # Policy-stage layer ownership is not a mid-cut preservation objective.
    # Negative values preserve the historical wrapper behavior; V94 sets these
    # explicitly so its attached layer contract has an auditable, independent
    # schedule instead of silently decaying with ``midcut_aux_*``.
    layer_contract_aux_loss_weight: float = -1.0
    layer_contract_aux_final_ratio: float = -1.0
    layer_contract_aux_decay_epochs: int = -1
    # Policy-stage migration knob.  The default 0 preserves the old behavior:
    # inherited layer-contract interfaces stay on upper_lr.  When a pre-fix
    # stage1 checkpoint is loaded with dirty adapter weights skipped, set this
    # to 1.0 so the freshly initialized layer interface learns at base LR.
    layer_contract_adapter_policy_lr_scale: float = 0.0

    # V39.2 layer-latent contract.  The per-layer latent/future head is the
    # primary Stage-1 objective.  A single shared flow-matching action probe is
    # a low-weight downstream readability probe; it should not dominate early
    # world-latent formation.
    layer_latent_loss_weight: float = 1.0
    layer_fm_probe_loss_weight: float = 0.0
    layer_event_loss_weight: float = 0.05
    layer_motion_loss_weight: float = 0.03
    layer_decoded_action_loss_weight: float = 0.0
    layer_contrast_loss_weight: float = 0.03

    # V39.3 recurrent milestone consequence regularizers.  These are applied
    # to the layer-contract rollout latent when the model is configured with
    # --layer-recurrent-consequence 1.  They keep the predicted future trajectory
    # from collapsing to a tiny average vector and make each milestone compare
    # against the corresponding sparse future anchor.
    layer_variance_loss_weight: float = 0.05
    layer_norm_loss_weight: float = 0.02
    layer_delta_match_loss_weight: float = 0.15


def _validate_current_token_tensor(tokens: Tensor, *, system: V39PolicySystem) -> None:
    cfg = system.policy_config
    expected = (
        cfg.visual_history_length,
        cfg.num_cameras,
        cfg.patches_per_camera,
        cfg.visual_token_dim,
    )
    if tokens.ndim != 5 or tuple(tokens.shape[1:]) != expected:
        raise ValueError(f"history_dinov2_tokens must be [B,{expected}], got {tuple(tokens.shape)}")


def _validate_target_anchor_token_tensor(tokens: Tensor, *, system: V39PolicySystem) -> None:
    cfg = system.policy_config
    expected_tail = (cfg.num_cameras, cfg.patches_per_camera, cfg.visual_token_dim)
    if tokens.ndim != 5 or tuple(tokens.shape[2:]) != expected_tail:
        raise ValueError(
            f"target_future_dinov2_tokens must be [B,F,{expected_tail}], got {tuple(tokens.shape)}"
        )
    required = (
        len(cfg.flow_jepa_target_offsets)
        if int(getattr(cfg, "flow_jepa_enabled", 0))
        else int(cfg.future_anchors)
    )
    if int(tokens.shape[1]) < required:
        raise ValueError(
            f"target_future_dinov2_tokens has only {tokens.shape[1]} targets; need {required}"
        )


def _selected_future_indices(sample: dict[str, Tensor], model_config) -> tuple[int, ...]:
    anchors = int(getattr(model_config, "future_anchors", getattr(model_config, "num_future", 1)))
    if not int(getattr(model_config, "flow_jepa_enabled", 0)):
        return tuple(range(anchors))
    expected = tuple(int(value) for value in model_config.flow_jepa_target_offsets)
    available = sample.get("future_offsets")
    if not torch.is_tensor(available):
        raise ValueError("hierarchical Flow-DINO target selection requires future_offsets")
    if available.ndim == 2:
        if not torch.equal(available, available[:1].expand_as(available)):
            raise ValueError("batched future_offsets must be identical across samples")
        available = available[0]
    if available.ndim != 1:
        raise ValueError("future_offsets must be [F] or [B,F]")
    lookup = {int(value): index for index, value in enumerate(available.tolist())}
    missing = [offset for offset in expected if offset not in lookup]
    if missing:
        raise ValueError(f"requested Flow-DINO future offsets are unavailable: {missing}")
    return tuple(lookup[offset] for offset in expected)


def _saved_flow_jepa_hierarchy(
    saved_policy: dict[str, Any],
) -> tuple[tuple[int, ...], int]:
    """Resolve checkpoint hierarchy fields with the policy-config fallbacks."""

    window_offsets = tuple(int(value) for value in saved_policy.get("flow_jepa_window_offsets", ()))
    if not window_offsets:
        anchors = int(saved_policy.get("future_anchors", 0))
        horizon = int(saved_policy.get("action_horizon", 0))
        if anchors <= 0 or horizon <= 0:
            raise ValueError(
                "checkpoint cannot derive Flow-DINO window offsets without positive "
                "future_anchors and action_horizon"
            )
        window_offsets = tuple(
            max(1, int(round((index + 1) * horizon / float(anchors)))) for index in range(anchors)
        )
    stage_offset = int(saved_policy.get("flow_jepa_stage_offset", 0))
    if stage_offset <= 0:
        stage_offset = int(window_offsets[-1]) + 1
    return window_offsets, stage_offset


def _validate_v102_resume_contract(
    saved_policy: dict[str, Any],
    current_policy: Any,
) -> None:
    """Reject shape-compatible resumes that change V102 routing semantics."""

    for field in (
        "flow_jepa_world_anchor_write_only",
        "flow_jepa_late_policy_detail",
        "flow_jepa_soft_address_lattice",
        "flow_jepa_horizon_soft_address",
        "flow_jepa_variance_safe_routing",
        "flow_jepa_complete_numerical_contract",
        "flow_jepa_interval_stage_delta",
        "flow_jepa_policy_multi_glimpse_address",
        "flow_jepa_horizon_cell_fine_address",
        "flow_jepa_online_horizon_address",
        "flow_jepa_progressive_grounding_address",
        "flow_jepa_coordinate_typed_raw_detail",
        "flow_jepa_structured_ownership_bottleneck",
        "flow_jepa_pre_value_owner_routing",
        "flow_jepa_functional_mainline_routing",
        "flow_jepa_utility_precision_mainline",
        "flow_jepa_action_free_world_factual",
        "flow_jepa_shared_factual_glimpse_bank",
        "flow_jepa_g_aligned_future_effect",
        "flow_jepa_stateless_goal_phase_machine",
        "flow_jepa_policy_plan_compiler",
        "flow_jepa_supervised_effect_mainline",
        "flow_jepa_stateless_intent_controller",
        "flow_jepa_window_effect_bank",
        "flow_jepa_effect_read_in_p2",
        "flow_jepa_differential_intent_effect_mainline",
        "flow_jepa_grounded_intent_effect_mainline",
        "flow_jepa_p1_mixed_precision",
        "flow_jepa_interval_stage_typed_value",
        "flow_jepa_policy_workspace_horizon_pool",
        "role_attnres_enabled",
        "role_attnres_ground_to_world",
        "role_attnres_world_to_policy",
        "role_attnres_policy_to_mmdit",
        "role_residual_contract_after_gate",
        "action_history_condition_exact_null",
        "goal_condition_exact_null",
        "stateless_phase_enabled",
    ):
        saved_value = int(saved_policy.get(field, 0))
        current_value = int(getattr(current_policy, field))
        if saved_value != current_value:
            raise ValueError(
                f"resume {field} mismatch: checkpoint={saved_value}, current={current_value}"
            )
    saved_schedule = str(saved_policy.get("flow_jepa_top_role_schedule", "3-3-2"))
    current_schedule = str(current_policy.flow_jepa_top_role_schedule)
    if saved_schedule != current_schedule:
        raise ValueError(
            "resume flow_jepa_top_role_schedule mismatch: "
            f"checkpoint={saved_schedule}, current={current_schedule}"
        )
    if int(getattr(current_policy, "flow_jepa_g_aligned_future_effect", 0)):
        saved_decay = float(saved_policy.get("flow_jepa_teacher_g_ema_decay", float("nan")))
        current_decay = float(current_policy.flow_jepa_teacher_g_ema_decay)
        if saved_decay != current_decay:
            raise ValueError(
                "resume flow_jepa_teacher_g_ema_decay mismatch: "
                f"checkpoint={saved_decay}, current={current_decay}"
            )
    saved_effect_slots = int(
        saved_policy.get("flow_jepa_future_slots", current_policy.future_anchors)
    )
    current_effect_slots = int(
        getattr(current_policy, "flow_jepa_future_slots", current_policy.future_anchors)
    )
    if saved_effect_slots != current_effect_slots:
        raise ValueError(
            "resume flow_jepa_future_slots mismatch: "
            f"checkpoint={saved_effect_slots}, current={current_effect_slots}"
        )
    saved_time_distribution = str(
        saved_policy.get("flow_matching_time_distribution", "uniform")
    )
    current_time_distribution = str(
        getattr(current_policy, "flow_matching_time_distribution", "uniform")
    )
    if saved_time_distribution != current_time_distribution:
        raise ValueError(
            "resume flow_matching_time_distribution mismatch: "
            f"checkpoint={saved_time_distribution}, "
            f"current={current_time_distribution}"
        )
    if int(getattr(current_policy, "flow_jepa_utility_precision_mainline", 0)):
        for field in (
            "flow_jepa_address_query_batch_budget",
            "flow_jepa_microgrid_tile",
            "flow_jepa_checkpoint_min_batch",
        ):
            saved_value = int(saved_policy.get(field, -1))
            current_value = int(getattr(current_policy, field))
            if saved_value != current_value:
                raise ValueError(
                    f"resume {field} mismatch: checkpoint={saved_value}, current={current_value}"
                )
    if int(getattr(current_policy, "flow_jepa_late_policy_detail", 0)):
        saved_detail_scale = float(saved_policy.get("flow_jepa_late_policy_detail_scale", 0.25))
        current_detail_scale = float(current_policy.flow_jepa_late_policy_detail_scale)
        if saved_detail_scale != current_detail_scale:
            raise ValueError(
                "resume flow_jepa_late_policy_detail_scale mismatch: "
                f"checkpoint={saved_detail_scale}, current={current_detail_scale}"
            )
    if int(getattr(current_policy, "flow_jepa_soft_address_lattice", 0)):
        for field in (
            "flow_jepa_address_slots",
            "flow_jepa_address_route_dim",
            "flow_jepa_address_query_chunk",
        ):
            saved_value = int(saved_policy.get(field, 0))
            current_value = int(getattr(current_policy, field))
            if saved_value != current_value:
                raise ValueError(
                    f"resume {field} mismatch: checkpoint={saved_value}, current={current_value}"
                )
    if int(getattr(current_policy, "flow_jepa_coordinate_typed_raw_detail", 0)):
        saved_micro_grid = int(saved_policy.get("flow_jepa_raw_micro_grid", 0))
        current_micro_grid = int(current_policy.flow_jepa_raw_micro_grid)
        if saved_micro_grid != current_micro_grid:
            raise ValueError(
                "resume flow_jepa_raw_micro_grid mismatch: "
                f"checkpoint={saved_micro_grid}, current={current_micro_grid}"
            )
    if int(getattr(current_policy, "flow_jepa_horizon_soft_address", 0)):
        saved_scale = float(saved_policy.get("flow_jepa_horizon_address_update_scale", 0.0))
        current_scale = float(current_policy.flow_jepa_horizon_address_update_scale)
        if saved_scale != current_scale:
            raise ValueError(
                "resume flow_jepa_horizon_address_update_scale mismatch: "
                f"checkpoint={saved_scale}, current={current_scale}"
            )
    if int(getattr(current_policy, "flow_jepa_pre_value_owner_routing", 0)):
        saved_scale = float(saved_policy.get("flow_jepa_pre_value_owner_update_scale", 0.10))
        current_scale = float(current_policy.flow_jepa_pre_value_owner_update_scale)
        if saved_scale != current_scale:
            raise ValueError(
                "resume flow_jepa_pre_value_owner_update_scale mismatch: "
                f"checkpoint={saved_scale}, current={current_scale}"
            )
    if int(getattr(current_policy, "flow_jepa_variance_safe_routing", 0)):
        for field in (
            "flow_jepa_routing_norm_floor",
            "flow_jepa_horizon_value_max_rms",
        ):
            saved_value = float(saved_policy.get(field, float("nan")))
            current_value = float(getattr(current_policy, field))
            if saved_value != current_value:
                raise ValueError(
                    f"resume {field} mismatch: checkpoint={saved_value}, current={current_value}"
                )
    if int(getattr(current_policy, "flow_jepa_complete_numerical_contract", 0)):
        for field in (
            "flow_jepa_correlation_rms_floor",
            "flow_jepa_visibility_transition_fraction",
        ):
            saved_value = float(saved_policy.get(field, float("nan")))
            current_value = float(getattr(current_policy, field))
            if saved_value != current_value:
                raise ValueError(
                    f"resume {field} mismatch: checkpoint={saved_value}, current={current_value}"
                )
    if int(getattr(current_policy, "flow_jepa_interval_stage_delta", 0)):
        for field in ("flow_jepa_interval_stage_update_scale",):
            saved_value = float(saved_policy.get(field, float("nan")))
            current_value = float(getattr(current_policy, field))
            if saved_value != current_value:
                raise ValueError(
                    f"resume {field} mismatch: checkpoint={saved_value}, current={current_value}"
                )
        for field in (
            "flow_jepa_interval_boundaries",
            "flow_jepa_interval_support_offsets",
        ):
            saved_value = tuple(int(value) for value in saved_policy.get(field, ()))
            current_value = tuple(int(value) for value in getattr(current_policy, field))
            if saved_value != current_value:
                raise ValueError(
                    f"resume {field} mismatch: checkpoint={saved_value}, current={current_value}"
                )
    if int(getattr(current_policy, "role_attnres_enabled", 0)):
        saved_key_dim = int(saved_policy.get("role_attnres_key_dim", 0))
        if saved_key_dim != int(current_policy.role_attnres_key_dim):
            raise ValueError(
                "resume role_attnres_key_dim mismatch: "
                f"checkpoint={saved_key_dim}, "
                f"current={int(current_policy.role_attnres_key_dim)}"
            )
        for field in (
            "role_attnres_ground_to_world_scale",
            "role_attnres_world_to_policy_scale",
            "role_attnres_policy_to_mmdit_scale",
        ):
            saved_value = float(saved_policy.get(field, float("nan")))
            current_value = float(getattr(current_policy, field))
            if saved_value != current_value:
                raise ValueError(
                    f"resume {field} mismatch: checkpoint={saved_value}, current={current_value}"
                )
    if int(getattr(current_policy, "stateless_phase_enabled", 0)):
        saved_count = int(saved_policy.get("stateless_phase_count", 0))
        if saved_count != int(current_policy.stateless_phase_count):
            raise ValueError(
                "resume stateless_phase_count mismatch: "
                f"checkpoint={saved_count}, "
                f"current={int(current_policy.stateless_phase_count)}"
            )
        saved_scale = float(saved_policy.get("stateless_phase_query_scale", float("nan")))
        if saved_scale != float(current_policy.stateless_phase_query_scale):
            raise ValueError(
                "resume stateless_phase_query_scale mismatch: "
                f"checkpoint={saved_scale}, "
                f"current={float(current_policy.stateless_phase_query_scale)}"
            )


@torch.no_grad()
def encode_target_anchor_tokens(
    sample: dict[str, Tensor],
    *,
    conditioner: RDT2Conditioner,
    model_config,
    camera_names: Sequence[str],
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Encode the configured future teacher sequence in exact contract order."""
    batch = int(sample["state"].shape[0])
    indices = _selected_future_indices(sample, model_config)
    target_count = len(indices)
    if "target_future_dinov2_tokens" in sample:
        tokens = sample["target_future_dinov2_tokens"][:, :target_count]
        cached_offsets = sample.get("target_future_offsets")
        if int(getattr(model_config, "flow_jepa_enabled", 0)):
            expected_offsets = torch.as_tensor(
                model_config.flow_jepa_target_offsets, dtype=torch.long
            )
            if not torch.is_tensor(cached_offsets):
                raise ValueError("cached hierarchical targets must carry target_future_offsets")
            if cached_offsets.ndim == 2:
                expected_offsets = expected_offsets[None].expand_as(cached_offsets)
            if not torch.equal(cached_offsets.cpu().long(), expected_offsets):
                raise ValueError(
                    "cached target token order does not match the Flow-JEPA horizon contract"
                )
        _validate_target_anchor_token_tensor(tokens, system=model_config_owner(model_config))
        return tokens.to(device=device, dtype=dtype, non_blocking=True)[:, :, None]
    if "target_history_obs_image" in sample:
        images = sample["target_history_obs_image"][:, list(indices), -1]
        flat = images.reshape(batch * target_count, *images.shape[2:])
        condition = conditioner.encode(flat, camera_names=camera_names)
    else:
        keys = sample["target_history_keys"][:, list(indices), -1, :].reshape(
            batch * target_count, 2
        )
        dummy = torch.zeros(
            batch * target_count, model_config.num_cameras, 3, 1, 1, dtype=torch.float32
        )
        condition = conditioner.encode(dummy, sample_keys=keys, camera_names=camera_names)
    if condition.dense_tokens is None:
        raise ValueError("V38 future target requires dense DINO tokens")
    dense = condition.dense_tokens
    expected_tokens = model_config.num_cameras * model_config.patches_per_camera
    if (
        dense.ndim != 3
        or dense.shape[0] != batch * target_count
        or dense.shape[1] != expected_tokens
        or dense.shape[2] != model_config.latent_dim
    ):
        raise ValueError(
            "DINO target-anchor geometry mismatch: "
            f"got {tuple(dense.shape)}, expected ({batch * target_count},{expected_tokens},{model_config.latent_dim})"
        )
    return dense.reshape(
        batch,
        target_count,
        model_config.num_cameras,
        model_config.patches_per_camera,
        model_config.latent_dim,
    ).to(device=device, dtype=dtype)[:, :, None]


def model_config_owner(model_config):
    # Small adapter so token-prefetch validation can reuse the full system-style
    # shape checker.  It intentionally exposes only policy_config.
    class _Owner:
        policy_config = model_config

    return _Owner()


@torch.no_grad()
def prepare_v39_policy_sample(
    sample: dict[str, Tensor],
    *,
    conditioner: RDT2Conditioner,
    system: V39PolicySystem,
    camera_names: Sequence[str],
    device: torch.device,
    dtype: torch.dtype,
    include_target_visual: bool = False,
) -> dict[str, Tensor]:
    if "history_dinov2_tokens" in sample:
        visual = sample["history_dinov2_tokens"]
        _validate_current_token_tensor(visual, system=system)
        visual = visual.to(device=device, dtype=dtype, non_blocking=True)
    else:
        conditioning_sample = sample
        if (
            "history_obs_image" in sample
            and "history_keys" in sample
            and hasattr(conditioner, "store")
        ):
            # Raw images are still required by V98 Flow-JEPA, but the cached
            # DINO conditioner must be addressed by episode/image keys.
            conditioning_sample = dict(sample)
            conditioning_sample.pop("history_obs_image")
        visual = encode_current_tokens(
            conditioning_sample,
            conditioner=conditioner,
            model_config=system.policy_config,
            camera_names=camera_names,
            device=device,
            dtype=dtype,
        )
    keys = (
        "state",
        "state_raw",
        "action_state",
        "history_state",
        "executed_action_history",
        "executed_action_history_raw",
        "policy_action",
        "policy_action_raw",
    )
    out = {key: sample[key].to(device=device, non_blocking=True) for key in keys}
    for key in (
        "state",
        "action_state",
        "history_state",
        "executed_action_history",
        "policy_action",
    ):
        out[key] = out[key].float()
    if int(
        getattr(
            system.policy_config,
            "flow_jepa_object_intent_dynamics_mainline",
            0,
        )
    ):
        for key in ("action", "future_state", "future_offsets"):
            value = sample.get(key)
            if not torch.is_tensor(value):
                raise RuntimeError(
                    f"object-intent training requires dataset field {key!r}"
                )
            out[key] = value.to(device=device, non_blocking=True)
        out["action"] = out["action"].float()
        out["future_state"] = out["future_state"].float()
    compute_dtype = dtype if device.type == "cuda" else torch.float32
    out["visual"] = visual.to(dtype=compute_dtype)
    if int(getattr(system.policy_config, "flow_jepa_raw_image_enabled", 0)):
        raw_visual = sample.get("history_obs_image")
        if not torch.is_tensor(raw_visual):
            raise RuntimeError(
                "raw-image Flow-JEPA requires history_obs_image beside cached DINO tokens"
            )
        expected_prefix = (
            int(visual.shape[0]),
            int(system.policy_config.visual_history_length),
            int(system.policy_config.num_cameras),
            3,
        )
        if raw_visual.ndim != 6 or tuple(raw_visual.shape[:4]) != expected_prefix:
            raise ValueError(
                "history_obs_image must be [B,H,C,3,R,R] for raw-image Flow-JEPA; "
                f"got {tuple(raw_visual.shape)}"
            )
        if int(raw_visual.shape[-2]) != int(raw_visual.shape[-1]):
            raise ValueError("raw-image Flow-JEPA currently requires square RGB frames")
        if int(raw_visual.shape[-1]) < 32 or int(raw_visual.shape[-1]) % 16:
            raise ValueError("raw RGB side must be >=32 and divisible by 16")
        out["raw_visual"] = raw_visual.to(device=device, dtype=compute_dtype, non_blocking=True)
    frame_progress = sample.get("frame_progress")
    if torch.is_tensor(frame_progress):
        # Audit-only metadata: deliberately carried beside, rather than into,
        # every model-forward argument list.
        out["frame_progress"] = frame_progress.to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
    if include_target_visual:
        target_visual = encode_target_anchor_tokens(
            sample,
            conditioner=conditioner,
            model_config=system.policy_config,
            camera_names=camera_names,
            device=device,
            dtype=dtype,
        )
        out["target_visual"] = target_visual.to(dtype=compute_dtype)
    return out


def _object_intent_future_training_pack(
    sample: dict[str, Tensor],
    *,
    system: V39PolicySystem,
    require_teacher: bool,
) -> dict[str, Tensor] | None:
    """Build the sole training-only object teacher input at its runtime boundary.

    Training, preflight, and representation validation previously assembled
    this mapping independently.  That allowed validation to request the object
    future loss while silently omitting its teacher.  Keep the decision and
    required fields in one place; deployment sampling never calls this helper.
    """

    if not bool(
        int(
            getattr(
                system.policy_config,
                "flow_jepa_object_intent_dynamics_mainline",
                0,
            )
        )
    ):
        return None
    target_visual = sample.get("target_visual")
    if not require_teacher and not torch.is_tensor(target_visual):
        return None
    required = ("action", "future_state", "future_offsets", "target_visual")
    missing = [key for key in required if not torch.is_tensor(sample.get(key))]
    if missing:
        raise RuntimeError(
            "object-intent teacher input is incomplete; missing tensor fields: "
            + ", ".join(missing)
        )
    return {
        "future_action": sample["action"],
        "future_state": sample["future_state"],
        "future_offsets": sample["future_offsets"],
        "target_visual": sample["target_visual"],
    }


def _effect_distance(pred: Tensor, target: Tensor) -> Tensor:
    diff = pred.float() - target.float().detach()
    mse = diff.square().mean(dim=(1, 2))
    pred_n = F.normalize(pred.float(), dim=-1)
    target_n = F.normalize(target.float().detach(), dim=-1)
    cosine = 1.0 - (pred_n * target_n).sum(dim=-1).mean(dim=1)
    return mse + 0.10 * cosine


def _future_grid_count(system_or_output: Any | None, output: dict[str, Tensor]) -> int:
    # Prefer explicit policy config when present; fall back to shape inference.
    cfg = None
    if system_or_output is not None:
        cfg = getattr(system_or_output, "policy_config", None)
    if cfg is not None:
        return int(cfg.num_cameras) * int(cfg.future_grid_size) * int(cfg.future_grid_size)
    target = output.get("rollout_effect_target")
    pred = output.get("rollout_effect_pred")
    ref = target if torch.is_tensor(target) else pred
    if not torch.is_tensor(ref):
        return 1
    # Default geometry in the V39 scripts is two cameras and a 4x4 future grid.
    return 32 if ref.shape[1] % 32 == 0 else max(int(ref.shape[1]), 1)


def _reshape_milestones(x: Tensor, *, grid: int) -> Tensor:
    if x.ndim != 3:
        raise ValueError(f"milestone tensor must be [B,K*G,H], got {tuple(x.shape)}")
    grid = max(int(grid), 1)
    usable = (int(x.shape[1]) // grid) * grid
    if usable <= 0:
        raise ValueError(f"cannot infer milestones from shape {tuple(x.shape)} and grid={grid}")
    if usable != int(x.shape[1]):
        x = x[:, :usable]
    return x.reshape(x.shape[0], usable // grid, grid, x.shape[-1])


def latent_variance_loss(output: dict[str, Tensor], *, grid: int) -> Tensor:
    if "rollout_effect_target" not in output:
        return torch.zeros(
            (),
            device=output["pred_physical_velocity"].device,
            dtype=output["pred_physical_velocity"].dtype,
        )
    pred = output["rollout_effect_pred"].float()
    target = output["rollout_effect_target"].float().detach()
    steps = min(pred.shape[1], target.shape[1])
    pred = pred[:, :steps]
    target = target[:, :steps]
    pred_std = pred.std(dim=(0, 1), unbiased=False).mean().clamp_min(1e-6)
    target_std = target.std(dim=(0, 1), unbiased=False).mean().clamp_min(1e-6)
    return (torch.log(pred_std) - torch.log(target_std)).square()


def latent_norm_loss(output: dict[str, Tensor]) -> Tensor:
    if "rollout_effect_target" not in output:
        return torch.zeros(
            (),
            device=output["pred_physical_velocity"].device,
            dtype=output["pred_physical_velocity"].dtype,
        )
    pred = output["rollout_effect_pred"].float()
    target = output["rollout_effect_target"].float().detach()
    steps = min(pred.shape[1], target.shape[1])
    pred_norm = pred[:, :steps].norm(dim=-1).mean()
    target_norm = target[:, :steps].norm(dim=-1).mean().detach()
    return F.smooth_l1_loss(pred_norm, target_norm)


def milestone_delta_match_loss(output: dict[str, Tensor], *, grid: int) -> Tensor:
    if "rollout_effect_target" not in output:
        return torch.zeros(
            (),
            device=output["pred_physical_velocity"].device,
            dtype=output["pred_physical_velocity"].dtype,
        )
    if "milestone_step_delta_pred" in output:
        pred_delta_flat = output["milestone_step_delta_pred"].float()
        target_delta_flat = _milestone_step_delta_target(output, grid=grid)
        steps = min(pred_delta_flat.shape[1], target_delta_flat.shape[1])
        pred_delta = pred_delta_flat[:, :steps]
        target_delta = target_delta_flat[:, :steps]
    else:
        pred = _reshape_milestones(output["rollout_effect_pred"].float(), grid=grid)
        target = _reshape_milestones(output["rollout_effect_target"].float().detach(), grid=grid)
        k = min(pred.shape[1], target.shape[1])
        pred = pred[:, :k]
        target = target[:, :k]
        z_pred = torch.zeros_like(pred[:, :1])
        z_target = torch.zeros_like(target[:, :1])
        pred_delta = pred - torch.cat([z_pred, pred[:, :-1]], dim=1)
        target_delta = target - torch.cat([z_target, target[:, :-1]], dim=1)
    smooth = F.smooth_l1_loss(pred_delta, target_delta)
    pred_n = F.normalize(pred_delta, dim=-1)
    target_n = F.normalize(target_delta, dim=-1)
    cosine = (1.0 - (pred_n * target_n).sum(dim=-1)).mean()
    return smooth + 0.10 * cosine


def _rollout_residual_target(output: dict[str, Tensor]) -> Tensor:
    target = output["rollout_effect_target"].float().detach()
    if "rollout_base_effect_pred" not in output:
        return target
    base = output["rollout_base_effect_pred"].float().detach()
    return target - base


def _milestone_step_delta_target(output: dict[str, Tensor], *, grid: int | None = None) -> Tensor:
    """Return per-milestone target deltas aligned to V40 step-delta predictions."""
    target = _rollout_residual_target(output)
    if grid is None:
        grid = _future_grid_count(None, output)
    target_m = _reshape_milestones(target, grid=int(grid))
    z_target = torch.zeros_like(target_m[:, :1])
    target_delta = target_m - torch.cat([z_target, target_m[:, :-1]], dim=1)
    return target_delta.reshape(
        target_delta.shape[0], target_delta.shape[1] * target_delta.shape[2], target_delta.shape[-1]
    )


def rollout_delta_loss(output: dict[str, Tensor]) -> Tensor:
    """Supervise the action-centered local delta.

    V40 prefers ``milestone_step_delta_pred`` when present.  That makes the
    causal branch learn per-segment action effects instead of using a cumulative
    future residual under a misleading ``delta`` name.  Older checkpoints still
    fall back to ``rollout_delta_pred`` for compatibility.
    """
    if "rollout_effect_target" not in output:
        device = output["pred_physical_velocity"].device
        return torch.zeros((), device=device, dtype=output["pred_physical_velocity"].dtype)
    if "milestone_step_delta_pred" in output:
        pred = output["milestone_step_delta_pred"].float()
        target = _milestone_step_delta_target(output)
        steps = min(pred.shape[1], target.shape[1])
        pred = pred[:, :steps]
        target = target[:, :steps]
    elif "rollout_delta_pred" in output:
        pred = output["rollout_delta_pred"].float()
        target = _rollout_residual_target(output)
    else:
        device = output["pred_physical_velocity"].device
        return torch.zeros((), device=device, dtype=output["pred_physical_velocity"].dtype)
    smooth = F.smooth_l1_loss(pred, target)
    pred_n = F.normalize(pred, dim=-1)
    target_n = F.normalize(target, dim=-1)
    cosine = (1.0 - (pred_n * target_n).sum(dim=-1)).mean()
    return smooth + 0.10 * cosine


def _future_horizon_charts(
    output: dict[str, Tensor],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Align concatenated future tokens as [B,K,N,H] without copying."""

    pred = output["flow_jepa_future_pred"]
    target = output["flow_jepa_future_target"]
    current = output.get("flow_jepa_current_target")
    mask = output["flow_jepa_future_target_mask"]
    if pred.ndim != 3 or tuple(pred.shape) != tuple(target.shape):
        raise ValueError("future JEPA prediction and target must align as [B,K*N,H]")
    if tuple(mask.shape) != tuple(pred.shape[:2]):
        raise ValueError("future JEPA target mask must be [B,K*N]")
    batch, tokens, hidden = pred.shape
    if torch.is_tensor(current):
        if (
            current.ndim != 3
            or int(current.shape[0]) != int(pred.shape[0])
            or int(current.shape[2]) != int(pred.shape[2])
        ):
            raise ValueError("current JEPA chart must align with future batch/hidden axes")
        if int(pred.shape[1]) % int(current.shape[1]) != 0:
            raise ValueError("future JEPA token count must be a multiple of the current chart")
        per_horizon = int(current.shape[1])
        horizons = tokens // per_horizon
        current_chart = current
    else:
        offsets = output.get("flow_jepa_future_offsets")
        if not isinstance(offsets, (tuple, list)) or not offsets:
            raise ValueError(
                "per-horizon JEPA reduction requires current chart or real future offsets"
            )
        horizons = len(offsets)
        if tokens % horizons:
            raise ValueError("future JEPA tokens must divide across real horizon offsets")
        per_horizon = tokens // horizons
        # Absolute prediction does not consume this placeholder. Change losses
        # validate and require a real current teacher chart before calling here.
        current_chart = pred.new_zeros(batch, per_horizon, hidden)
    return (
        pred.reshape(batch, horizons, per_horizon, hidden),
        target.reshape(batch, horizons, per_horizon, hidden),
        current_chart[:, None].expand(-1, horizons, -1, -1),
        mask.reshape(batch, horizons, per_horizon),
    )


def _mean_valid_horizon_rows(values: Tensor, counts: Tensor) -> Tensor:
    """Average horizon rows without device-synchronizing Python branches."""

    valid = (counts > 0).to(dtype=values.dtype)
    return (values * valid).sum() / valid.sum().clamp_min(1.0)


def _future_delta_charts(
    output: dict[str, Tensor],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return predicted/teacher changes and the selected-token chart."""

    pred_h, target_h, current_h, mask_h = _future_horizon_charts(output)
    explicit_delta = output.get("flow_jepa_future_delta_pred")
    if torch.is_tensor(explicit_delta):
        if tuple(explicit_delta.shape) != tuple(output["flow_jepa_future_pred"].shape):
            raise ValueError("explicit future delta must align with flow_jepa_future_pred")
        pred_delta_h = explicit_delta.float().reshape_as(pred_h)
    else:
        pred_delta_h = pred_h.float() - current_h.detach().float()
    target_delta_h = target_h.detach().float() - current_h.detach().float()
    return pred_delta_h, target_delta_h, current_h.detach().float(), mask_h


_FUTURE_CHANGE_REFERENCE_FRACTION = 0.05


def _future_change_weighting(
    target_delta_h: Tensor,
    current_h: Tensor,
    selected_h: Tensor,
    *,
    reliable_normalization: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Build a continuous, absolute-scale-aware future-change weight.

    Legacy V104 normalizes by the future delta's own RMS.  That ratio is
    invariant when an entire weak/noisy teacher delta shrinks, so it cannot
    decide whether normalization is trustworthy.  V105 instead adds a smooth
    floor equal to five percent of the current teacher-chart RMS.  This keeps
    fine changes continuous while preventing arbitrarily weak change from
    receiving scale-invariant normalized gradients.
    """

    selected_h = selected_h.to(
        device=target_delta_h.device,
        dtype=torch.float32,
    )
    counts_h = selected_h.sum(dim=(0, 2))
    target_energy_h = target_delta_h.square().mean(dim=-1)
    delta_scale_h = (
        torch.sqrt((target_energy_h * selected_h).sum(dim=(0, 2)) / counts_h.clamp_min(1.0))
        .detach()
        .clamp_min(1e-3)
    )
    current_energy_h = current_h.square().mean(dim=-1)
    current_scale_h = torch.sqrt(
        (current_energy_h * selected_h).sum(dim=(0, 2)) / counts_h.clamp_min(1.0)
    ).detach()
    if reliable_normalization:
        normalization_scale_h = torch.sqrt(
            delta_scale_h.square() + (_FUTURE_CHANGE_REFERENCE_FRACTION * current_scale_h).square()
        ).clamp_min(1e-3)
    else:
        # Preserve the serialized V104 objective exactly when the V105
        # reliability contract is disabled.
        normalization_scale_h = delta_scale_h
    hidden = int(target_delta_h.shape[-1])
    strength_h = target_delta_h.norm(dim=-1).detach() / math.sqrt(float(hidden))
    reliability_h = strength_h / (strength_h + normalization_scale_h[None, :, None].clamp_min(1e-3))
    return (
        counts_h,
        delta_scale_h,
        current_scale_h,
        normalization_scale_h,
        reliability_h,
    )


def _scale_floored_direction_rows(
    prediction_h: Tensor,
    target_h: Tensor,
    normalization_scale_h: Tensor,
) -> tuple[Tensor, Tensor]:
    """Signed alignment with a teacher-scale, zero-safe denominator.

    Near-zero delta predictors are an intentional initialization state.  A
    conventional cosine normalization has an inverse-norm derivative there.
    This form retains signed alignment while bounding both prediction and
    target normalization by a detached scale already owned by the reliable
    future-target contract.
    """

    if tuple(prediction_h.shape) != tuple(target_h.shape):
        raise ValueError("direction prediction and target charts must align")
    if prediction_h.ndim != 4:
        raise ValueError("direction charts must be [B,A,N,H]")
    if tuple(normalization_scale_h.shape) != (int(prediction_h.shape[1]),):
        raise ValueError("direction scale must contain one value per horizon")
    floor_h = (0.25 * normalization_scale_h.detach().float()).clamp_min(1e-3)
    floor = floor_h[None, :, None, None]
    prediction_direction = prediction_h.float() / torch.sqrt(
        prediction_h.float().square().mean(dim=-1, keepdim=True) + floor.square()
    )
    target_direction = target_h.detach().float() / torch.sqrt(
        target_h.detach().float().square().mean(dim=-1, keepdim=True) + floor.square()
    )
    return (
        1.0 - (prediction_direction * target_direction).mean(dim=-1),
        floor_h,
    )


def _flow_jepa_predictive_change_contract_terms(
    output: dict[str, Tensor],
    *,
    reliable_normalization: bool = False,
) -> dict[str, Tensor]:
    """Build the exact per-horizon terms used by predictive JEPA backward.

    Smooth-L1 on every selected token keeps genuinely static cells stable.
    Direction is weighted continuously by teacher change strength, so visible
    changes matter more without introducing a hard motion threshold. Keeping
    this calculation in one helper prevents the diagnostic view from silently
    drifting away from the scalar sent to backward.
    """

    pred_delta_h, target_delta_h, current_h, mask_h = _future_delta_charts(output)
    selected_h = mask_h.to(device=pred_delta_h.device, dtype=torch.float32)
    (
        counts_h,
        delta_scale_h,
        current_scale_h,
        normalization_scale_h,
        reliability_h,
    ) = _future_change_weighting(
        target_delta_h,
        current_h,
        selected_h,
        reliable_normalization=reliable_normalization,
    )
    raw_error_h = F.smooth_l1_loss(
        pred_delta_h,
        target_delta_h,
        reduction="none",
    ).mean(dim=-1)
    raw_h = (raw_error_h * selected_h).sum(dim=(0, 2)) / counts_h.clamp_min(1.0)
    normalized_error_h = F.smooth_l1_loss(
        pred_delta_h / normalization_scale_h[None, :, None, None],
        target_delta_h / normalization_scale_h[None, :, None, None],
        reduction="none",
    ).mean(dim=-1)

    if reliable_normalization:
        normalized_weight_h = selected_h * reliability_h
        smooth_h = (normalized_error_h * normalized_weight_h).sum(dim=(0, 2)) / counts_h.clamp_min(
            1.0
        )
        magnitude_h = raw_h + smooth_h
    else:
        normalized_weight_h = selected_h
        smooth_h = (normalized_error_h * selected_h).sum(dim=(0, 2)) / counts_h.clamp_min(1.0)
        magnitude_h = smooth_h
    direction_weight_h = selected_h * reliability_h
    direction_denominator_h = direction_weight_h.sum(dim=(0, 2))
    if torch.is_tensor(output.get("flow_jepa_variance_safe_routing")):
        direction_rows_h, direction_floor_h = _scale_floored_direction_rows(
            pred_delta_h,
            target_delta_h,
            normalization_scale_h,
        )
    else:
        # Exact V105 compatibility when the V106 variance-safe contract is
        # absent from the real forward output.
        direction_rows_h = 1.0 - (
            F.normalize(pred_delta_h, dim=-1) * F.normalize(target_delta_h, dim=-1)
        ).sum(dim=-1)
        direction_floor_h = normalization_scale_h.new_zeros(normalization_scale_h.shape)
    direction_h = (direction_rows_h * direction_weight_h).sum(dim=(0, 2)) / (
        counts_h.clamp_min(1.0)
        if reliable_normalization
        else direction_denominator_h.clamp_min(1e-12)
    )
    reliability_mean_h = direction_weight_h.sum(dim=(0, 2)) / counts_h.clamp_min(1.0)
    return {
        "counts_h": counts_h,
        "direction_denominator_h": direction_denominator_h,
        "delta_scale_h": delta_scale_h,
        "current_scale_h": current_scale_h,
        "normalization_scale_h": normalization_scale_h,
        "reliability_mean_h": reliability_mean_h,
        "raw_h": raw_h,
        "normalized_h": smooth_h,
        "magnitude_h": magnitude_h,
        "direction_h": direction_h,
        "direction_floor_h": direction_floor_h,
        "active_h": magnitude_h + 0.10 * direction_h,
    }


def _flow_jepa_predictive_change_contract_loss(
    output: dict[str, Tensor],
    *,
    balance_horizons: bool,
    reliable_normalization: bool = False,
) -> Tensor:
    """Predict a teacher-chart delta without rewarding an absolute copy."""

    terms = _flow_jepa_predictive_change_contract_terms(
        output,
        reliable_normalization=reliable_normalization,
    )
    counts_h = terms["counts_h"]
    if balance_horizons:
        return _mean_valid_horizon_rows(terms["active_h"], counts_h)

    total_count = counts_h.sum().clamp_min(1.0)
    if reliable_normalization:
        raw = (terms["raw_h"] * counts_h).sum() / total_count
        smooth = raw + (terms["normalized_h"] * counts_h).sum() / total_count
    else:
        smooth = (terms["normalized_h"] * counts_h).sum() / total_count
    if reliable_normalization:
        direction = (terms["direction_h"] * counts_h).sum() / total_count
    else:
        direction_denominator_h = terms["direction_denominator_h"]
        direction_denominator = direction_denominator_h.sum().clamp_min(1e-12)
        direction = (terms["direction_h"] * direction_denominator_h).sum() / direction_denominator
    return smooth + 0.10 * direction


def flow_jepa_future_prediction_loss(
    output: dict[str, Tensor],
    *,
    balance_horizons: bool = False,
    reliable_normalization: bool = False,
) -> Tensor:
    """Future-only masked latent prediction on frozen DINO teacher tokens."""

    pred = output.get("flow_jepa_future_pred")
    target = output.get("flow_jepa_future_target")
    mask = output.get("flow_jepa_future_target_mask")
    if not all(torch.is_tensor(value) for value in (pred, target, mask)):
        reference = output["pred_physical_velocity"]
        return torch.zeros((), device=reference.device, dtype=reference.dtype)
    if pred.ndim != 3 or tuple(pred.shape) != tuple(target.shape):
        raise ValueError("Flow-DINO JEPA prediction and teacher must align as [B,N,H]")
    if tuple(mask.shape) != tuple(pred.shape[:2]):
        raise ValueError("Flow-DINO JEPA target mask must be [B,N]")
    selected = mask.to(device=pred.device, dtype=torch.bool)
    if not bool(selected.any()):
        return pred.sum() * 0.0
    if torch.is_tensor(output.get("flow_jepa_future_delta_pred")):
        if not torch.is_tensor(output.get("flow_jepa_current_target")):
            raise ValueError("predictive-change JEPA requires a frozen current teacher chart")
        return _flow_jepa_predictive_change_contract_loss(
            output,
            balance_horizons=balance_horizons,
            reliable_normalization=reliable_normalization,
        )
    if balance_horizons:
        pred_h, target_h, _, mask_h = _future_horizon_charts(output)
        weight_h = mask_h.to(device=pred.device, dtype=torch.float32)
        count_h = weight_h.sum(dim=(0, 2))
        smooth_rows = F.smooth_l1_loss(
            pred_h.float(), target_h.detach().float(), reduction="none"
        ).mean(dim=-1)
        cosine_rows = 1.0 - (
            F.normalize(pred_h.float(), dim=-1) * F.normalize(target_h.detach().float(), dim=-1)
        ).sum(dim=-1)
        loss_h = ((smooth_rows + 0.10 * cosine_rows) * weight_h).sum(dim=(0, 2))
        loss_h = loss_h / count_h.clamp_min(1.0)
        base = _mean_valid_horizon_rows(loss_h, count_h)
    else:
        pred_f = pred.float()[selected]
        target_f = target.detach().float()[selected]
        smooth = F.smooth_l1_loss(pred_f, target_f)
        cosine = (
            1.0 - (F.normalize(pred_f, dim=-1) * F.normalize(target_f, dim=-1)).sum(dim=-1).mean()
        )
        base = smooth + 0.10 * cosine
    change_direction = flow_jepa_future_change_direction_loss(
        output,
        balance_horizons=balance_horizons,
    )
    return base + 0.10 * change_direction


def flow_jepa_future_reliable_diagnostics(
    output: dict[str, Tensor],
    *,
    reliable_normalization: bool = True,
    balance_horizons: bool = True,
) -> dict[str, Tensor]:
    """Expose the exact predictive-JEPA components used by backward."""

    if not torch.is_tensor(output.get("flow_jepa_future_delta_pred")):
        return {}
    terms = _flow_jepa_predictive_change_contract_terms(
        output,
        reliable_normalization=reliable_normalization,
    )
    delta_scale_h = terms["delta_scale_h"]
    current_scale_h = terms["current_scale_h"]
    normalization_scale_h = terms["normalization_scale_h"]
    raw_h = terms["raw_h"]
    normalized_h = terms["normalized_h"]
    direction_h = terms["direction_h"]
    active_h = terms["active_h"]
    reliability_mean_h = terms["reliability_mean_h"]
    counts_h = terms["counts_h"]
    if balance_horizons:
        active_direction = _mean_valid_horizon_rows(
            direction_h,
            counts_h,
        )
        active_composite = _mean_valid_horizon_rows(
            active_h,
            counts_h,
        )
    elif reliable_normalization:
        total_count = counts_h.sum().clamp_min(1.0)
        active_direction = (direction_h * counts_h).sum() / total_count
        active_magnitude = ((raw_h + normalized_h) * counts_h).sum() / total_count
        active_composite = active_magnitude + 0.10 * active_direction
    else:
        total_count = counts_h.sum().clamp_min(1.0)
        direction_denominator_h = terms["direction_denominator_h"]
        active_direction = (
            direction_h * direction_denominator_h
        ).sum() / direction_denominator_h.sum().clamp_min(1e-12)
        active_magnitude = (normalized_h * counts_h).sum() / total_count
        active_composite = active_magnitude + 0.10 * active_direction
    offsets = tuple(int(value) for value in output.get("flow_jepa_future_offsets", ()))
    diagnostics: dict[str, Tensor] = {
        "flow_jepa_future_target_delta_scale": delta_scale_h.mean().detach(),
        "flow_jepa_future_current_reference_scale": (current_scale_h.mean().detach()),
        "flow_jepa_future_normalization_scale": (normalization_scale_h.mean().detach()),
        "flow_jepa_future_raw_delta_loss": raw_h.mean().detach(),
        "flow_jepa_future_reliable_normalized_loss": (normalized_h.mean().detach()),
        "flow_jepa_future_change_reliability": reliability_mean_h.mean().detach(),
        "flow_jepa_future_active_direction_loss": active_direction.detach(),
        "flow_jepa_future_active_composite_loss": active_composite.detach(),
    }
    if torch.is_tensor(output.get("flow_jepa_variance_safe_routing")):
        diagnostics["flow_jepa_future_direction_floor_min"] = (
            terms["direction_floor_h"].amin().detach()
        )
    if len(offsets) == int(delta_scale_h.shape[0]):
        for index, offset in enumerate(offsets):
            diagnostics[f"flow_jepa_future_horizon_{offset}_target_scale"] = delta_scale_h[
                index
            ].detach()
            diagnostics[f"flow_jepa_future_horizon_{offset}_normalization_scale"] = (
                normalization_scale_h[index].detach()
            )
            diagnostics[f"flow_jepa_future_horizon_{offset}_raw_delta"] = raw_h[index].detach()
            diagnostics[f"flow_jepa_future_horizon_{offset}_reliable_normalized"] = normalized_h[
                index
            ].detach()
            diagnostics[f"flow_jepa_future_horizon_{offset}_reliability"] = reliability_mean_h[
                index
            ].detach()
            diagnostics[f"flow_jepa_future_horizon_{offset}_active_direction"] = direction_h[
                index
            ].detach()
            diagnostics[f"flow_jepa_future_horizon_{offset}_active_loss"] = active_h[index].detach()
            if torch.is_tensor(output.get("flow_jepa_variance_safe_routing")):
                diagnostics[f"flow_jepa_future_horizon_{offset}_direction_floor"] = terms[
                    "direction_floor_h"
                ][index].detach()
    return diagnostics


def _flow_jepa_horizon_address_terms(
    output: dict[str, Tensor],
) -> dict[str, Tensor]:
    """Teacher-only supervision for the horizon-specific soft address."""

    logits = output.get("flow_jepa_horizon_address_logits")
    future = output.get("flow_jepa_future_target")
    current = output.get("flow_jepa_current_target")
    if not all(torch.is_tensor(value) for value in (logits, future, current)):
        # Stage-1 JEPA training deliberately has no action-policy prediction.
        # Keep the disabled address term connected to whichever prediction
        # actually owns this objective instead of depending on a downstream
        # policy-only key.
        reference = next(
            (
                value
                for key in (
                    "flow_jepa_future_pred",
                    "flow_jepa_stage_pred",
                    "pred_physical_velocity",
                )
                if torch.is_tensor(value := output.get(key))
            ),
            None,
        )
        if reference is None:
            raise KeyError("horizon address loss needs a JEPA or policy prediction reference")
        return {
            "flow_jepa_horizon_address": reference.float().sum() * 0.0,
        }
    if logits.ndim != 5:
        raise ValueError("horizon address logits must be [B,A,C,G,G]")
    batch, anchors, cameras, grid, grid_b = logits.shape
    if grid != grid_b:
        raise ValueError("horizon address logits require a square query chart")
    positions = cameras * grid * grid
    if tuple(future.shape[:2]) != (batch, anchors * positions):
        raise ValueError("future teacher does not align with horizon address logits")
    if tuple(current.shape[:2]) != (batch, positions):
        raise ValueError("current teacher does not align with horizon address logits")
    future_h = future.detach().float().reshape(batch, anchors, positions, int(future.shape[-1]))
    current_h = current.detach().float()[:, None].expand(-1, anchors, -1, -1)
    strength = (future_h - current_h).square().mean(dim=-1).clamp_min(0.0).sqrt()
    interval_progress = output.get("flow_jepa_interval_progress_target")
    if torch.is_tensor(interval_progress):
        if tuple(interval_progress.shape) != tuple(future.shape):
            raise ValueError("interval progression teacher must align with future address targets")
        progression_strength = (
            interval_progress.detach()
            .float()
            .reshape(
                batch,
                anchors,
                positions,
                int(interval_progress.shape[-1]),
            )
            .square()
            .mean(dim=-1)
            .clamp_min(0.0)
            .sqrt()
        )
        # Stage progression may change spatial relevance, but it does not
        # replace the content-change teacher or enter the online reader.
        strength = torch.sqrt(strength.square() + 0.25 * progression_strength.square())
    mean_strength = strength.mean(dim=-1)
    spatial_std = strength.std(dim=-1, unbiased=False)
    spatial_contrast = spatial_std / (mean_strength + spatial_std + 1e-8)
    current_scale = current.detach().float().square().mean(dim=-1).sqrt().mean(dim=-1)
    temporal_reliability = mean_strength / (mean_strength + current_scale[:, None].clamp_min(1e-6))
    reliability = (spatial_contrast * temporal_reliability).detach()
    teacher_scale = strength.mean(dim=-1, keepdim=True).detach().clamp_min(1e-6)
    teacher_probability = torch.softmax(
        (strength / teacher_scale).detach(),
        dim=-1,
    )
    log_predicted = F.log_softmax(
        logits.float().reshape(batch, anchors, positions),
        dim=-1,
    )
    teacher_log = teacher_probability.clamp_min(1e-8).log()
    kl_h = (teacher_probability * (teacher_log - log_predicted)).sum(dim=-1)
    loss = (kl_h * reliability).mean()
    predicted_probability = log_predicted.exp()
    entropy_denominator = math.log(float(max(positions, 2)))
    teacher_entropy = (
        -(teacher_probability.clamp_min(1e-8) * teacher_log).sum(dim=-1) / entropy_denominator
    )
    predicted_entropy = (
        -(predicted_probability.clamp_min(1e-8) * log_predicted).sum(dim=-1) / entropy_denominator
    )
    result = {
        "flow_jepa_horizon_address": loss,
        "flow_jepa_horizon_address_kl": kl_h.mean().detach(),
        "flow_jepa_horizon_address_teacher_entropy": teacher_entropy.mean().detach(),
        "flow_jepa_horizon_address_predicted_entropy": (predicted_entropy.mean().detach()),
        "flow_jepa_horizon_address_teacher_max": teacher_probability.max(dim=-1)
        .values.mean()
        .detach(),
        "flow_jepa_horizon_address_predicted_max": predicted_probability.max(dim=-1)
        .values.mean()
        .detach(),
        "flow_jepa_horizon_address_teacher_contrast": spatial_contrast.mean().detach(),
        "flow_jepa_horizon_address_teacher_reliability": reliability.mean().detach(),
        "flow_jepa_horizon_address_teacher_change": mean_strength.mean().detach(),
    }
    offsets = tuple(int(value) for value in output.get("flow_jepa_future_offsets", ()))
    if len(offsets) == anchors:
        for index, offset in enumerate(offsets):
            result[f"flow_jepa_horizon_address_{offset}_kl"] = kl_h[:, index].mean().detach()
            result[f"flow_jepa_horizon_address_{offset}_reliability"] = (
                reliability[:, index].mean().detach()
            )
    return result


def flow_jepa_horizon_address_loss(output: dict[str, Tensor]) -> Tensor:
    return _flow_jepa_horizon_address_terms(output)["flow_jepa_horizon_address"]


def flow_jepa_future_change_direction_loss(
    output: dict[str, Tensor],
    *,
    balance_horizons: bool = False,
) -> Tensor:
    """Continuously weight future direction by observed teacher change."""

    pred = output.get("flow_jepa_future_pred")
    target = output.get("flow_jepa_future_target")
    current = output.get("flow_jepa_current_target")
    mask = output.get("flow_jepa_future_target_mask")
    if not all(torch.is_tensor(value) for value in (pred, target, current, mask)):
        reference = pred if torch.is_tensor(pred) else output.get("pred_physical_velocity")
        if not torch.is_tensor(reference):
            raise ValueError("future change direction loss requires a tensor reference")
        return reference.sum() * 0.0
    if pred.ndim != 3 or tuple(pred.shape) != tuple(target.shape):
        raise ValueError("future change direction requires aligned [B,N,H] prediction/target")
    if tuple(mask.shape) != tuple(pred.shape[:2]):
        raise ValueError("future change direction mask must be [B,N]")
    if (
        current.ndim != 3
        or int(current.shape[0]) != int(pred.shape[0])
        or int(current.shape[2]) != int(pred.shape[2])
    ):
        raise ValueError("current JEPA teacher chart must align with future batch/hidden axes")
    if int(pred.shape[1]) % int(current.shape[1]) != 0:
        raise ValueError("future JEPA token count must be a multiple of the current chart")
    horizons = int(pred.shape[1]) // int(current.shape[1])
    current_expanded = current[:, None].expand(-1, horizons, -1, -1).reshape_as(pred)
    selected = mask.to(device=pred.device, dtype=torch.bool)
    explicit_delta = output.get("flow_jepa_future_delta_pred")
    pred_delta = (
        explicit_delta.float()
        if torch.is_tensor(explicit_delta)
        else pred.float() - current_expanded.detach().float()
    )
    target_delta = target.detach().float() - current_expanded.detach().float()
    target_strength = target_delta.norm(dim=-1).detach()
    if balance_horizons:
        pred_h, target_h, current_h, mask_h = _future_horizon_charts(output)
        pred_delta_h = (
            explicit_delta.float().reshape_as(pred_h)
            if torch.is_tensor(explicit_delta)
            else pred_h.float() - current_h.detach().float()
        )
        target_delta_h = target_h.detach().float() - current_h.detach().float()
        strength_h = target_delta_h.norm(dim=-1).detach()
        weights_h = mask_h.to(dtype=strength_h.dtype) * strength_h
        denominator_h = weights_h.sum(dim=(0, 2))
        direction_h = 1.0 - (
            F.normalize(pred_delta_h, dim=-1) * F.normalize(target_delta_h, dim=-1)
        ).sum(dim=-1)
        loss_h = (direction_h * weights_h).sum(dim=(0, 2)) / denominator_h.clamp_min(1e-12)
        return _mean_valid_horizon_rows(loss_h, denominator_h)
    weights = selected.float() * target_strength
    denominator = weights.sum().clamp_min(1e-12)
    direction = 1.0 - (F.normalize(pred_delta, dim=-1) * F.normalize(target_delta, dim=-1)).sum(
        dim=-1
    )
    return (direction * weights).sum() / denominator


def flow_jepa_future_change_loss(
    output: dict[str, Tensor],
    *,
    balance_horizons: bool = False,
) -> Tensor:
    """Supervise change direction and scale without a hard motion threshold.

    Absolute JEPA prediction is dominated by static semantic content.  This
    objective reweights every selected token continuously by the teacher's
    observed change magnitude, then matches both direction and delta scale.
    Target deltas and weights are detached; gradients flow only into the
    prediction route.
    """

    pred = output.get("flow_jepa_future_pred")
    target = output.get("flow_jepa_future_target")
    current = output.get("flow_jepa_current_target")
    mask = output.get("flow_jepa_future_target_mask")
    if not all(torch.is_tensor(value) for value in (pred, target, current, mask)):
        reference = pred if torch.is_tensor(pred) else output.get("pred_physical_velocity")
        if not torch.is_tensor(reference):
            raise ValueError("future change loss requires a tensor reference")
        return reference.sum() * 0.0
    if pred.ndim != 3 or tuple(pred.shape) != tuple(target.shape):
        raise ValueError("future change loss requires aligned [B,N,H] prediction/target")
    if tuple(mask.shape) != tuple(pred.shape[:2]):
        raise ValueError("future change loss mask must be [B,N]")
    if (
        current.ndim != 3
        or int(current.shape[0]) != int(pred.shape[0])
        or int(current.shape[2]) != int(pred.shape[2])
        or int(pred.shape[1]) % int(current.shape[1]) != 0
    ):
        raise ValueError("current JEPA chart does not align with future prediction")
    horizons = int(pred.shape[1]) // int(current.shape[1])
    current_expanded = current[:, None].expand(-1, horizons, -1, -1).reshape_as(pred)
    explicit_delta = output.get("flow_jepa_future_delta_pred")
    pred_delta = (
        explicit_delta.float()
        if torch.is_tensor(explicit_delta)
        else pred.float() - current_expanded.detach().float()
    )
    target_delta = target.detach().float() - current_expanded.detach().float()
    selected = mask.to(device=pred.device, dtype=torch.bool)
    target_strength = target_delta.norm(dim=-1).detach()
    if balance_horizons:
        pred_h, target_h, current_h, mask_h = _future_horizon_charts(output)
        pred_delta_h = (
            explicit_delta.float().reshape_as(pred_h)
            if torch.is_tensor(explicit_delta)
            else pred_h.float() - current_h.detach().float()
        )
        target_delta_h = target_h.detach().float() - current_h.detach().float()
        selected_h = mask_h.to(device=pred.device, dtype=torch.float32)
        strength_h = target_delta_h.norm(dim=-1).detach()
        weights_h = selected_h * strength_h
        denominator_h = weights_h.sum(dim=(0, 2))
        selected_count_h = selected_h.sum(dim=(0, 2)).clamp_min(1.0)
        delta_scale_h = (denominator_h / selected_count_h).detach().clamp_min(1e-3)
        direction_h = 1.0 - (
            F.normalize(pred_delta_h, dim=-1) * F.normalize(target_delta_h, dim=-1)
        ).sum(dim=-1)
        scale = delta_scale_h[None, :, None, None]
        delta_match_h = F.smooth_l1_loss(
            pred_delta_h / scale,
            target_delta_h / scale,
            reduction="none",
        ).mean(dim=-1)
        loss_h = ((direction_h + 0.25 * delta_match_h) * weights_h).sum(
            dim=(0, 2)
        ) / denominator_h.clamp_min(1e-12)
        return _mean_valid_horizon_rows(loss_h, denominator_h)
    weights = selected.float() * target_strength
    denominator = weights.sum().clamp_min(1e-12)
    direction = 1.0 - (F.normalize(pred_delta, dim=-1) * F.normalize(target_delta, dim=-1)).sum(
        dim=-1
    )
    selected_count = selected.float().sum().clamp_min(1.0)
    delta_scale = (weights.sum() / selected_count).detach().clamp_min(1e-3)
    delta_match = F.smooth_l1_loss(
        pred_delta / delta_scale,
        target_delta / delta_scale,
        reduction="none",
    ).mean(dim=-1)
    return ((direction + 0.25 * delta_match) * weights).sum() / denominator


def flow_jepa_future_horizon_diagnostics(
    output: dict[str, Tensor],
    *,
    reliable_normalization: bool = False,
) -> dict[int, Tensor]:
    """Expose the active masked-JEPA loss at each real frame offset.

    Predictive-change runs reuse the exact raw/normalized/direction composite
    that enters backward. Legacy absolute-prediction runs retain their
    historical smooth-L1 plus cosine diagnostic.
    """

    pred = output.get("flow_jepa_future_pred")
    target = output.get("flow_jepa_future_target")
    mask = output.get("flow_jepa_future_target_mask")
    offsets = output.get("flow_jepa_future_offsets")
    if not all(torch.is_tensor(value) for value in (pred, target, mask)):
        return {}
    if not isinstance(offsets, (tuple, list)) or not offsets:
        return {}
    if pred.ndim != 3 or tuple(pred.shape) != tuple(target.shape):
        raise ValueError("per-horizon JEPA diagnostics require aligned [B,N,H] tensors")
    horizon_count = len(offsets)
    if int(pred.shape[1]) % horizon_count:
        raise ValueError("future token count must divide evenly across real horizon offsets")
    if torch.is_tensor(output.get("flow_jepa_future_delta_pred")):
        active_h = _flow_jepa_predictive_change_contract_terms(
            output,
            reliable_normalization=reliable_normalization,
        )["active_h"]
        if int(active_h.shape[0]) != horizon_count:
            raise ValueError("active predictive JEPA rows do not align with offsets")
        return {int(offset): active_h[index] for index, offset in enumerate(offsets)}
    tokens = int(pred.shape[1]) // horizon_count
    pred_h = pred.float().reshape(int(pred.shape[0]), horizon_count, tokens, int(pred.shape[2]))
    target_h = target.detach().float().reshape_as(pred_h)
    mask_h = mask.to(device=pred.device, dtype=torch.bool).reshape(
        int(pred.shape[0]), horizon_count, tokens
    )
    rows: dict[int, Tensor] = {}
    for index, offset in enumerate(offsets):
        weight = mask_h[:, index].float()
        denominator = weight.sum().clamp_min(1.0)
        smooth_rows = F.smooth_l1_loss(pred_h[:, index], target_h[:, index], reduction="none").mean(
            dim=-1
        )
        cosine_rows = 1.0 - (
            F.normalize(pred_h[:, index], dim=-1) * F.normalize(target_h[:, index], dim=-1)
        ).sum(dim=-1)
        smooth = (smooth_rows * weight).sum() / denominator
        cosine = (cosine_rows * weight).sum() / denominator
        rows[int(offset)] = smooth + 0.10 * cosine
    return rows


def _flow_jepa_balance_horizons(trainer: V39PolicyTrainerConfig) -> bool:
    mode = str(getattr(trainer, "flow_jepa_horizon_balance_mode", "global"))
    mode = mode.strip().lower().replace("-", "_")
    if mode not in {"global", "per_horizon"}:
        raise ValueError(
            f"flow_jepa_horizon_balance_mode must be 'global' or 'per_horizon', got {mode!r}"
        )
    return mode == "per_horizon"


def _flow_jepa_uses_reliable_normalization(
    trainer: V39PolicyTrainerConfig,
) -> bool:
    value = int(getattr(trainer, "flow_jepa_future_reliable_normalization", 0))
    if value not in (0, 1):
        raise ValueError("flow_jepa_future_reliable_normalization must be 0 or 1")
    return bool(value)


def flow_jepa_stage_prediction_loss(output: dict[str, Tensor]) -> Tensor:
    """Coarse far-horizon DINO-delta prediction without static-delta amplification."""

    pred = output.get("flow_jepa_stage_pred")
    target = output.get("flow_jepa_stage_target")
    if not all(torch.is_tensor(value) for value in (pred, target)):
        reference = output["pred_physical_velocity"]
        return torch.zeros((), device=reference.device, dtype=reference.dtype)
    if pred.ndim != 3 or tuple(pred.shape) != tuple(target.shape) or int(pred.shape[1]) != 1:
        raise ValueError("Flow-DINO stage prediction and target must align as [B,1,H]")
    pred_f = pred.float()
    target_f = target.detach().float()
    smooth = F.smooth_l1_loss(pred_f, target_f)
    target_norm = target_f.norm(dim=-1)
    informative = target_norm > 1e-3
    if bool(informative.any()):
        cosine_rows = 1.0 - (F.normalize(pred_f, dim=-1) * F.normalize(target_f, dim=-1)).sum(
            dim=-1
        )
        cosine = cosine_rows[informative].mean()
    else:
        cosine = smooth * 0.0
    return smooth + 0.10 * cosine


def flow_jepa_interval_stage_terms(
    output: dict[str, Tensor],
) -> dict[str, Tensor]:
    """Supervise one signed, spatial interval increment per W horizon.

    The primary target is a least-squares temporal progression.  A weak
    endpoint term prevents a transient sequence from matching the slope while
    missing the actual interval displacement.  Reliability uses the same
    current-chart-relative floor as V105, so weak teacher jitter cannot regain
    a scale-invariant gradient through normalization.
    """

    pred = output.get("flow_jepa_interval_progress_pred")
    target = output.get("flow_jepa_interval_progress_target")
    endpoint = output.get("flow_jepa_interval_endpoint_target")
    current = output.get("flow_jepa_current_target")
    mask = output.get("flow_jepa_future_target_mask")
    if not all(torch.is_tensor(value) for value in (pred, target, endpoint, current, mask)):
        reference = next(
            (
                value
                for key in (
                    "flow_jepa_future_pred",
                    "flow_jepa_stage_pred",
                    "pred_physical_velocity",
                )
                if torch.is_tensor(value := output.get(key))
            ),
            None,
        )
        if reference is None:
            raise KeyError("interval stage loss needs a JEPA or policy prediction reference")
        zero = reference.float().sum() * 0.0
        return {"flow_jepa_interval_stage": zero}
    if (
        pred.ndim != 3
        or tuple(pred.shape) != tuple(target.shape)
        or tuple(pred.shape) != tuple(endpoint.shape)
        or tuple(mask.shape) != tuple(pred.shape[:2])
    ):
        raise ValueError(
            "interval progression prediction/targets/mask must align as [B,A*N,H]/[B,A*N]"
        )
    batch, tokens, hidden = pred.shape
    if current.ndim != 3 or int(current.shape[0]) != batch or int(current.shape[-1]) != hidden:
        raise ValueError("interval current teacher chart has invalid geometry")
    per_horizon = int(current.shape[1])
    if tokens % per_horizon:
        raise ValueError("interval token count must divide by the current chart")
    anchors = tokens // per_horizon
    pred_h = pred.float().reshape(batch, anchors, per_horizon, hidden)
    target_h = target.detach().float().reshape_as(pred_h)
    endpoint_h = endpoint.detach().float().reshape_as(pred_h)
    current_h = current.detach().float()[:, None].expand(-1, anchors, -1, -1)
    selected_h = mask.reshape(batch, anchors, per_horizon).to(
        device=pred.device,
        dtype=torch.float32,
    )
    (
        counts_h,
        target_scale_h,
        _,
        normalization_scale_h,
        reliability_h,
    ) = _future_change_weighting(
        target_h,
        current_h,
        selected_h,
        reliable_normalization=True,
    )
    raw_rows = F.smooth_l1_loss(
        pred_h,
        target_h,
        reduction="none",
    ).mean(dim=-1)
    normalized_rows = F.smooth_l1_loss(
        pred_h / normalization_scale_h[None, :, None, None],
        target_h / normalization_scale_h[None, :, None, None],
        reduction="none",
    ).mean(dim=-1)
    endpoint_rows = F.smooth_l1_loss(
        pred_h,
        endpoint_h,
        reduction="none",
    ).mean(dim=-1)
    reliability_weight = selected_h * reliability_h
    raw_h = (raw_rows * selected_h).sum(dim=(0, 2)) / counts_h.clamp_min(1.0)
    normalized_h = (normalized_rows * reliability_weight).sum(dim=(0, 2)) / counts_h.clamp_min(1.0)
    endpoint_h_loss = (endpoint_rows * selected_h).sum(dim=(0, 2)) / counts_h.clamp_min(1.0)
    # The interval organizer is deliberately initialized near zero.  A plain
    # cosine/F.normalize objective would therefore have a 1/||prediction||
    # backward singularity and could recreate the V105 address-gradient
    # explosion in a new loss term.  Use the frozen teacher normalization
    # scale as a smooth denominator floor.  The loss still rewards signed
    # alignment, while its Jacobian stays bounded at a zero prediction.
    direction_rows, direction_floor_h = _scale_floored_direction_rows(
        pred_h,
        target_h,
        normalization_scale_h,
    )
    direction_h = (direction_rows * reliability_weight).sum(dim=(0, 2)) / counts_h.clamp_min(1.0)
    loss_h = raw_h + normalized_h + 0.10 * direction_h + 0.25 * endpoint_h_loss
    loss = _mean_valid_horizon_rows(loss_h, counts_h)
    # Keep the historical slot-reduced interval objective available for
    # diagnostics, but do not let it own the grounded mainline. Reducing the
    # object/camera/space axes before supervision recreates the averaging
    # shortcut that Grounded Intent-Effect is designed to remove.
    legacy_interval_loss = loss
    grounded_core_loss: Tensor | None = None
    effect_components: dict[str, Tensor] = {}
    legacy_effect_names = (
        (
            "semantic",
            "flow_jepa_future_effect_semantic_pred_slots",
            "flow_jepa_future_effect_semantic_target_slots",
            0.50,
        ),
        (
            "transport",
            "flow_jepa_future_effect_transport_pred_slots",
            "flow_jepa_future_effect_transport_target_slots",
            0.25,
        ),
        (
            "transport_covariance",
            "flow_jepa_future_effect_transport_covariance_pred_slots",
            "flow_jepa_future_effect_transport_covariance_target_slots",
            0.05,
        ),
        (
            "persistence",
            "flow_jepa_future_effect_persistence_pred_slots",
            "flow_jepa_future_effect_persistence_target_slots",
            0.10,
        ),
        (
            "visibility",
            "flow_jepa_future_effect_visibility_pred_slots",
            "flow_jepa_future_effect_visibility_target_slots",
            0.10,
        ),
        (
            "uncertainty",
            "flow_jepa_future_effect_uncertainty_pred_slots",
            "flow_jepa_future_effect_uncertainty_target_slots",
            0.05,
        ),
    )
    v116_effect_names = (
        (
            "current",
            "flow_jepa_future_effect_current_pred_slots",
            "flow_jepa_future_effect_current_target_slots",
            0.10,
            False,
        ),
        (
            "successor",
            "flow_jepa_future_effect_successor_pred_slots",
            "flow_jepa_future_effect_successor_target_slots",
            0.25,
            True,
        ),
        (
            "semantic",
            "flow_jepa_future_effect_semantic_pred_slots",
            "flow_jepa_future_effect_semantic_target_slots",
            0.25,
            True,
        ),
        (
            "transport",
            "flow_jepa_future_effect_transport_pred_slots",
            "flow_jepa_future_effect_transport_target_slots",
            0.15,
            True,
        ),
        (
            "transport_covariance",
            "flow_jepa_future_effect_transport_covariance_pred_slots",
            "flow_jepa_future_effect_transport_covariance_target_slots",
            0.05,
            True,
        ),
        (
            "persistence",
            "flow_jepa_future_effect_persistence_pred_slots",
            "flow_jepa_future_effect_persistence_target_slots",
            0.05,
            True,
        ),
        (
            "visibility",
            "flow_jepa_future_effect_visibility_pred_slots",
            "flow_jepa_future_effect_visibility_target_slots",
            0.10,
            False,
        ),
        (
            "uncertainty",
            "flow_jepa_future_effect_uncertainty_pred_slots",
            "flow_jepa_future_effect_uncertainty_target_slots",
            0.10,
            False,
        ),
    )
    v116_effect = all(
        torch.is_tensor(output.get(prediction_key))
        and torch.is_tensor(output.get(target_key))
        for _, prediction_key, target_key, _, _ in v116_effect_names
    ) and torch.is_tensor(
        output.get("flow_jepa_future_effect_w1_semantic_pred_slots")
    )
    differential_effect = all(
        torch.is_tensor(output.get(key))
        for key in (
            "flow_jepa_future_effect_current_reference",
            "flow_jepa_future_effect_successor_pred_slots",
            "flow_jepa_future_effect_successor_target_slots",
            "flow_jepa_future_effect_semantic_pred_slots",
            "flow_jepa_future_effect_semantic_target_slots",
            "flow_jepa_future_effect_transport_pred_slots",
            "flow_jepa_future_effect_transport_target_slots",
            "flow_jepa_future_effect_transport_covariance_pred_slots",
            "flow_jepa_future_effect_transport_covariance_target_slots",
            "flow_jepa_future_effect_persistence_pred_slots",
            "flow_jepa_future_effect_persistence_target_slots",
            "flow_jepa_future_effect_visibility_pred_slots",
            "flow_jepa_future_effect_visibility_target_slots",
            "flow_jepa_future_effect_uncertainty_pred_slots",
            "flow_jepa_future_effect_uncertainty_target_slots",
            "flow_jepa_future_effect_reliability_target_slots",
            "flow_jepa_intent_predictive_effect",
            "flow_jepa_future_effect_intent_summary_target_slots",
        )
    )
    grounded_effect = bool(
        torch.is_tensor(output.get("grounded_intent_effect_active"))
        and all(
            torch.is_tensor(output.get(key))
            for key in (
                "flow_jepa_future_effect_current_reference",
                "flow_jepa_future_effect_current_reference_target",
                "flow_jepa_future_effect_successor_pred_slots",
                "flow_jepa_future_effect_successor_target_slots",
                "flow_jepa_future_effect_semantic_pred_slots",
                "flow_jepa_future_effect_semantic_target_slots",
                "flow_jepa_future_effect_transport_pred_slots",
                "flow_jepa_future_effect_transport_target_slots",
                "flow_jepa_future_effect_transport_covariance_pred_slots",
                "flow_jepa_future_effect_transport_covariance_target_slots",
                "flow_jepa_future_effect_persistence_pred_slots",
                "flow_jepa_future_effect_persistence_target_slots",
                "flow_jepa_future_effect_visibility_pred_slots",
                "flow_jepa_future_effect_visibility_target_slots",
                "flow_jepa_future_effect_uncertainty_pred_slots",
                "flow_jepa_future_effect_uncertainty_target_slots",
                "flow_jepa_future_effect_reliability_pred_slots",
                "flow_jepa_future_effect_reliability_target_slots",
                "flow_jepa_future_effect_slot_valid",
            )
        )
    )
    effect_names = legacy_effect_names
    effect_available = all(
        torch.is_tensor(output.get(prediction_key)) and torch.is_tensor(output.get(target_key))
        for _, prediction_key, target_key, _ in effect_names
    )
    effect_diagnostics: dict[str, Tensor] = {}
    if grounded_effect:
        interval_names = ("h4_8", "h8_16", "h16_32", "h32_48")
        semantic_prediction = output[
            "flow_jepa_future_effect_semantic_pred_slots"
        ].float()
        semantic_teacher = output[
            "flow_jepa_future_effect_semantic_target_slots"
        ].detach().float()
        current_reference = output[
            "flow_jepa_future_effect_current_reference"
        ].detach().float()
        current_teacher = output[
            "flow_jepa_future_effect_current_reference_target"
        ].detach().float()
        successor_prediction = output[
            "flow_jepa_future_effect_successor_pred_slots"
        ].float()
        successor_teacher = output[
            "flow_jepa_future_effect_successor_target_slots"
        ].detach().float()
        teacher_reliability = output[
            "flow_jepa_future_effect_reliability_target_slots"
        ].detach().float().clamp(0.0, 1.0)
        slot_valid = output[
            "flow_jepa_future_effect_slot_valid"
        ].detach().float().clamp(0.0, 1.0)
        expected_effect_shape = tuple(semantic_prediction.shape)
        if (
            semantic_prediction.ndim != 7
            or int(semantic_prediction.shape[1]) != 4
            or tuple(semantic_teacher.shape) != expected_effect_shape
            or tuple(successor_prediction.shape) != expected_effect_shape
            or tuple(successor_teacher.shape) != expected_effect_shape
            or tuple(current_reference.shape)
            != tuple(semantic_prediction.shape[:1] + semantic_prediction.shape[2:])
            or tuple(current_teacher.shape) != tuple(current_reference.shape)
            or tuple(teacher_reliability.shape)
            != tuple(semantic_prediction.shape[:-1] + (1,))
            or tuple(slot_valid.shape) != tuple(teacher_reliability.shape)
        ):
            raise ValueError(
                "grounded FutureEffect must preserve "
                "[B,4,C,Y,X,M,D] and its object validity axis"
            )
        # The grounded preflight performs the full finite/value-domain audit.
        # Avoid eight device reductions and Python-bool synchronizations in
        # every training batch; the ordinary non-finite total-loss guard still
        # protects optimization.

        def grounded_scale_floored_rows(
            prediction: Tensor,
            teacher: Tensor,
        ) -> Tensor:
            raw = F.smooth_l1_loss(
                prediction,
                teacher,
                reduction="none",
            ).mean(dim=-1, keepdim=True)
            teacher_rms = teacher.square().mean(dim=-1, keepdim=True).sqrt()
            scale_floor = (
                0.25
                * teacher_rms.mean(
                    dim=(0, 2, 3, 4, 5),
                    keepdim=True,
                )
            ).clamp_min(1e-3)
            scale = torch.sqrt(teacher_rms.square() + scale_floor.square())
            normalized = F.smooth_l1_loss(
                prediction / scale,
                teacher / scale,
                reduction="none",
            ).mean(dim=-1, keepdim=True)
            prediction_direction = prediction / torch.sqrt(
                prediction.square().mean(dim=-1, keepdim=True)
                + scale_floor.square()
            )
            teacher_direction = teacher / torch.sqrt(
                teacher.square().mean(dim=-1, keepdim=True)
                + scale_floor.square()
            )
            direction = 1.0 - (
                prediction_direction * teacher_direction
            ).mean(dim=-1, keepdim=True)
            return raw + normalized + 0.10 * direction

        component_specs = (
            (
                "successor",
                successor_prediction,
                successor_teacher,
                0.25,
                False,
                False,
            ),
            (
                "semantic",
                semantic_prediction,
                semantic_teacher,
                0.20,
                True,
                True,
            ),
            (
                "transport",
                output["flow_jepa_future_effect_transport_pred_slots"].float(),
                output[
                    "flow_jepa_future_effect_transport_target_slots"
                ].detach().float(),
                0.10,
                True,
                False,
            ),
            (
                "transport_covariance",
                output[
                    "flow_jepa_future_effect_transport_covariance_pred_slots"
                ].float(),
                output[
                    "flow_jepa_future_effect_transport_covariance_target_slots"
                ].detach().float(),
                0.05,
                True,
                False,
            ),
            (
                "persistence_change",
                output[
                    "flow_jepa_future_effect_persistence_pred_slots"
                ].float(),
                output[
                    "flow_jepa_future_effect_persistence_target_slots"
                ].detach().float(),
                0.05,
                False,
                False,
            ),
            (
                "visibility_change",
                output[
                    "flow_jepa_future_effect_visibility_pred_slots"
                ].float(),
                output[
                    "flow_jepa_future_effect_visibility_target_slots"
                ].detach().float(),
                0.05,
                False,
                False,
            ),
            (
                "uncertainty_calibration",
                output[
                    "flow_jepa_future_effect_uncertainty_pred_slots"
                ].float(),
                output[
                    "flow_jepa_future_effect_uncertainty_target_slots"
                ].detach().float(),
                0.05,
                False,
                False,
            ),
            (
                "reliability_calibration",
                output[
                    "flow_jepa_future_effect_reliability_pred_slots"
                ].float(),
                teacher_reliability,
                0.05,
                False,
                False,
            ),
        )
        valid_denominator = slot_valid.sum().clamp_min(1.0)
        grounded_total = loss.new_zeros(())
        for (
            name,
            prediction,
            teacher,
            internal_weight,
            reliability_calibrated,
            scale_floored,
        ) in component_specs:
            if tuple(prediction.shape) != tuple(teacher.shape):
                raise ValueError(
                    f"grounded FutureEffect {name} does not align"
                )
            rows = (
                grounded_scale_floored_rows(prediction, teacher)
                if scale_floored
                else F.smooth_l1_loss(
                    prediction,
                    teacher,
                    reduction="none",
                ).mean(dim=-1, keepdim=True)
            )
            row_weight = slot_valid
            if reliability_calibrated:
                row_weight = row_weight * (
                    0.25 + 0.75 * teacher_reliability
                )
            component = (rows * row_weight).sum() / valid_denominator
            effect_components[name] = component
            grounded_total = (
                grounded_total + float(internal_weight) * component
            )
            squared_error = (
                (prediction - teacher).square().mean(dim=-1, keepdim=True)
            )
            target_power = teacher.square().mean(dim=-1, keepdim=True)
            for interval_index, interval_name in enumerate(interval_names):
                interval_valid = slot_valid[:, interval_index]
                interval_denominator = interval_valid.sum().clamp_min(1.0)
                interval_rows = rows[:, interval_index]
                interval_weight = row_weight[:, interval_index]
                interval_component = (
                    interval_rows * interval_weight
                ).sum() / interval_denominator
                effect_components[
                    f"{name}_{interval_name}"
                ] = interval_component
                error_rms = torch.sqrt(
                    (
                        squared_error[:, interval_index]
                        * interval_valid
                    ).sum()
                    / interval_denominator
                )
                target_rms = torch.sqrt(
                    (
                        target_power[:, interval_index]
                        * interval_valid
                    ).sum()
                    / interval_denominator
                ).clamp_min(1e-3)
                effect_diagnostics[
                    "grounded_future_effect_"
                    f"{name}_{interval_name}_target_normalized_error"
                ] = (error_rms / target_rms).detach()

        # The externally weighted future objective owns this complete
        # object-level field. The separately weighted interval objective owns
        # only adjacent-interval differentiation.
        grounded_core_loss = grounded_total
        prediction_transition = (
            semantic_prediction[:, 1:] - semantic_prediction[:, :-1]
        )
        teacher_transition = semantic_teacher[:, 1:] - semantic_teacher[:, :-1]
        transition_rows = grounded_scale_floored_rows(
            prediction_transition,
            teacher_transition,
        )
        transition_valid = torch.minimum(
            slot_valid[:, 1:],
            slot_valid[:, :-1],
        )
        transition_reliability = 0.25 + 0.75 * torch.minimum(
            teacher_reliability[:, 1:],
            teacher_reliability[:, :-1],
        )
        transition_denominator = transition_valid.sum().clamp_min(1.0)
        transition_component = (
            transition_rows
            * transition_valid
            * transition_reliability
        ).sum() / transition_denominator
        effect_components["relative_transition"] = transition_component
        for edge_index, (left, right) in enumerate(
            zip(interval_names[:-1], interval_names[1:])
        ):
            edge_valid = transition_valid[:, edge_index]
            edge_denominator = edge_valid.sum().clamp_min(1.0)
            effect_components[
                f"relative_transition_{left}_{right}"
            ] = (
                transition_rows[:, edge_index]
                * edge_valid
                * transition_reliability[:, edge_index]
            ).sum() / edge_denominator

        current_alignment = (
            current_reference - current_teacher
        ).square().mean().sqrt()
        effect_diagnostics[
            "grounded_future_effect_current_reference_alignment_rms"
        ] = current_alignment.detach()
        pooled_prediction = semantic_prediction.mean(dim=(2, 3, 4, 5))
        pooled_teacher = semantic_teacher.mean(dim=(2, 3, 4, 5))
        transport_prediction = output[
            "flow_jepa_future_effect_transport_pred_slots"
        ].float().mean(dim=(2, 3, 4, 5))
        transport_teacher = output[
            "flow_jepa_future_effect_transport_target_slots"
        ].detach().float().mean(dim=(2, 3, 4, 5))
        effect_diagnostics.update(
            {
                "grounded_future_effect_prediction_adjacent_cosine": (
                    F.cosine_similarity(
                        pooled_prediction[:, 1:],
                        pooled_prediction[:, :-1],
                        dim=-1,
                        eps=1e-6,
                    ).mean().detach()
                ),
                "grounded_future_effect_target_adjacent_cosine": (
                    F.cosine_similarity(
                        pooled_teacher[:, 1:],
                        pooled_teacher[:, :-1],
                        dim=-1,
                        eps=1e-6,
                    ).mean().detach()
                ),
                "grounded_future_effect_prediction_interval_variation": (
                    pooled_prediction.std(dim=1, unbiased=False).mean().detach()
                ),
                "grounded_future_effect_target_interval_variation": (
                    pooled_teacher.std(dim=1, unbiased=False).mean().detach()
                ),
                "grounded_future_effect_prediction_transport_variation": (
                    transport_prediction.std(
                        dim=1,
                        unbiased=False,
                    ).mean().detach()
                ),
                "grounded_future_effect_target_transport_variation": (
                    transport_teacher.std(
                        dim=1,
                        unbiased=False,
                    ).mean().detach()
                ),
            }
        )
        loss = transition_component
        effect_available = True
    elif differential_effect:
        reliability = output[
            "flow_jepa_future_effect_reliability_target_slots"
        ].detach().float().clamp(0.0, 1.0)
        semantic_prediction = output[
            "flow_jepa_future_effect_semantic_pred_slots"
        ].float()
        semantic_teacher = output[
            "flow_jepa_future_effect_semantic_target_slots"
        ].detach().float()
        current_reference = output[
            "flow_jepa_future_effect_current_reference"
        ].detach().float()
        successor_prediction = current_reference[:, None] + semantic_prediction
        successor_teacher = output[
            "flow_jepa_future_effect_successor_target_slots"
        ].detach().float()
        if (
            tuple(semantic_prediction.shape) != tuple(semantic_teacher.shape)
            or tuple(successor_prediction.shape) != tuple(successor_teacher.shape)
            or int(semantic_prediction.shape[1]) != 3
        ):
            raise ValueError(
                "differential FutureEffect prediction/teacher shapes do not align"
            )

        def scale_floored_rows(
            prediction: Tensor,
            teacher: Tensor,
        ) -> Tensor:
            raw = F.smooth_l1_loss(
                prediction,
                teacher,
                reduction="none",
            ).mean(dim=-1, keepdim=True)
            teacher_rms = teacher.square().mean(dim=-1, keepdim=True).sqrt()
            reduce_dims = tuple(
                index
                for index in range(teacher_rms.ndim)
                if index not in {1, teacher_rms.ndim - 1}
            )
            scale_floor = (
                0.25
                * teacher_rms.mean(dim=reduce_dims, keepdim=True)
            ).clamp_min(1e-3)
            scale = torch.sqrt(teacher_rms.square() + scale_floor.square())
            normalized = F.smooth_l1_loss(
                prediction / scale,
                teacher / scale,
                reduction="none",
            ).mean(dim=-1, keepdim=True)
            prediction_direction = prediction / torch.sqrt(
                prediction.square().mean(dim=-1, keepdim=True)
                + scale_floor.square()
            )
            teacher_direction = teacher / torch.sqrt(
                teacher.square().mean(dim=-1, keepdim=True)
                + scale_floor.square()
            )
            direction = 1.0 - (
                prediction_direction * teacher_direction
            ).mean(dim=-1, keepdim=True)
            return raw + normalized + 0.10 * direction

        component_specs = (
            (
                "successor",
                successor_prediction,
                successor_teacher,
                0.30,
                False,
                False,
            ),
            (
                "semantic",
                semantic_prediction,
                semantic_teacher,
                0.20,
                True,
                True,
            ),
            (
                "transport",
                output["flow_jepa_future_effect_transport_pred_slots"].float(),
                output[
                    "flow_jepa_future_effect_transport_target_slots"
                ].detach().float(),
                0.10,
                True,
                False,
            ),
            (
                "transport_covariance",
                output[
                    "flow_jepa_future_effect_transport_covariance_pred_slots"
                ].float(),
                output[
                    "flow_jepa_future_effect_transport_covariance_target_slots"
                ].detach().float(),
                0.05,
                True,
                False,
            ),
            (
                "persistence",
                output[
                    "flow_jepa_future_effect_persistence_pred_slots"
                ].float(),
                output[
                    "flow_jepa_future_effect_persistence_target_slots"
                ].detach().float(),
                0.05,
                False,
                False,
            ),
            (
                "visibility",
                output[
                    "flow_jepa_future_effect_visibility_pred_slots"
                ].float(),
                output[
                    "flow_jepa_future_effect_visibility_target_slots"
                ].detach().float(),
                0.05,
                False,
                False,
            ),
            (
                "uncertainty",
                output[
                    "flow_jepa_future_effect_uncertainty_pred_slots"
                ].float(),
                output[
                    "flow_jepa_future_effect_uncertainty_target_slots"
                ].detach().float(),
                0.05,
                False,
                False,
            ),
        )
        differential_total = loss.new_zeros(())
        slot_effective = {
            name: loss.new_zeros(())
            for name in ("near", "mid", "late")
        }
        for (
            name,
            prediction,
            teacher,
            internal_weight,
            reliability_calibrated,
            scale_floored,
        ) in component_specs:
            if tuple(prediction.shape) != tuple(teacher.shape):
                raise ValueError(
                    f"differential FutureEffect {name} does not align"
                )
            rows = (
                scale_floored_rows(prediction, teacher)
                if scale_floored
                else F.smooth_l1_loss(
                    prediction,
                    teacher,
                    reduction="none",
                ).mean(dim=-1, keepdim=True)
            )
            row_weight = (
                0.25 + 0.75 * reliability
                if reliability_calibrated
                else torch.ones_like(reliability)
            )
            component = (rows * row_weight).sum() / float(
                max(rows.numel(), 1)
            )
            effect_components[name] = component
            for slot_index, slot_name in enumerate(("near", "mid", "late")):
                slot_rows = rows[:, slot_index]
                slot_weight = row_weight[:, slot_index]
                slot_component = (
                    slot_rows * slot_weight
                ).sum() / float(max(slot_rows.numel(), 1))
                effect_components[f"{name}_{slot_name}"] = slot_component
                slot_effective[slot_name] = (
                    slot_effective[slot_name]
                    + float(internal_weight) * slot_component
                )
            differential_total = (
                differential_total + float(internal_weight) * component
            )

        prediction_transition = (
            semantic_prediction[:, 1:] - semantic_prediction[:, :-1]
        )
        teacher_transition = semantic_teacher[:, 1:] - semantic_teacher[:, :-1]
        transition_rows = scale_floored_rows(
            prediction_transition,
            teacher_transition,
        )
        transition_reliability = 0.25 + 0.75 * torch.minimum(
            reliability[:, 1:],
            reliability[:, :-1],
        )
        transition_component = (
            transition_rows * transition_reliability
        ).sum() / float(max(transition_rows.numel(), 1))
        effect_components["relative_transition"] = transition_component
        differential_total = differential_total + 0.10 * transition_component
        for edge_index, (left, right) in enumerate(
            (("near", "mid"), ("mid", "late"))
        ):
            edge_rows = transition_rows[:, edge_index]
            edge_weight = transition_reliability[:, edge_index]
            edge_component = (edge_rows * edge_weight).sum() / float(
                max(edge_rows.numel(), 1)
            )
            effect_components[
                f"relative_transition_{left}_{right}"
            ] = edge_component
            slot_effective[left] = (
                slot_effective[left] + 0.05 * edge_component
            )
            slot_effective[right] = (
                slot_effective[right] + 0.05 * edge_component
            )

        intent_prediction = output[
            "flow_jepa_intent_predictive_effect"
        ].float()
        intent_teacher = output[
            "flow_jepa_future_effect_intent_summary_target_slots"
        ].detach().float()
        if tuple(intent_prediction.shape) != tuple(intent_teacher.shape):
            raise ValueError(
                "intent window prediction and future-effect summary do not align"
            )
        intent_rows = scale_floored_rows(intent_prediction, intent_teacher)
        intent_component = intent_rows.mean()
        effect_components["intent_summary"] = intent_component
        differential_total = differential_total + 0.10 * intent_component
        for slot_index, slot_name in enumerate(("near", "mid", "late")):
            slot_intent = intent_rows[:, slot_index].mean()
            effect_components[
                f"intent_summary_{slot_name}"
            ] = slot_intent
            slot_effective[slot_name] = (
                slot_effective[slot_name] + 0.10 * slot_intent
            )
            effect_components[
                f"effective_{slot_name}"
            ] = slot_effective[slot_name]
        loss = loss + differential_total
        effect_available = True
    elif v116_effect:
        teacher_reliability = output.get(
            "flow_jepa_future_effect_reliability_target_slots"
        )
        if not torch.is_tensor(teacher_reliability):
            raise RuntimeError("V116 FutureEffect requires teacher reliability")
        teacher_reliability = teacher_reliability.detach().float().clamp(0.0, 1.0)
        valid_denominator = teacher_reliability.numel()
        window_effect = bool(
            torch.is_tensor(output.get("flow_jepa_future_effect_slot_valid"))
            and torch.is_tensor(
                output.get("flow_jepa_future_effect_w1_slot_valid")
            )
            and int(teacher_reliability.shape[1]) == 3
        )

        def supervise_effect_stage(
            *,
            prediction_prefix: str,
            stage_name: str,
            stage_weight: float,
            slot_mask: Tensor | None = None,
        ) -> Tensor:
            stage_total = loss.new_zeros(())
            if slot_mask is not None:
                if tuple(slot_mask.shape) != (3,):
                    raise ValueError("V117 effect slot mask must be [3]")
                broadcast_slot_mask = slot_mask.detach().float().reshape(
                    1, 3, 1, 1, 1, 1, 1
                )
                stage_denominator = (
                    float(valid_denominator)
                    * float(slot_mask.detach().float().mean().item())
                )
            else:
                broadcast_slot_mask = None
                stage_denominator = float(valid_denominator)
            for (
                name,
                final_prediction_key,
                target_key,
                internal_weight,
                reliability_weighted,
            ) in v116_effect_names:
                prediction_key = (
                    final_prediction_key
                    if not prediction_prefix
                    else final_prediction_key.replace(
                        "flow_jepa_future_effect_",
                        prediction_prefix,
                    )
                )
                prediction = output.get(prediction_key)
                teacher = output.get(target_key)
                if not torch.is_tensor(prediction) or not torch.is_tensor(teacher):
                    raise RuntimeError(
                        f"V116 FutureEffect {stage_name}/{name} is incomplete"
                    )
                prediction = prediction.float()
                teacher = teacher.detach().float()
                if tuple(prediction.shape) != tuple(teacher.shape):
                    raise ValueError(
                        f"V116 FutureEffect {stage_name}/{name} does not align"
                    )
                rows = F.smooth_l1_loss(
                    prediction,
                    teacher,
                    reduction="none",
                ).mean(dim=-1, keepdim=True)
                if name == "semantic":
                    teacher_rms = teacher.square().mean(
                        dim=-1, keepdim=True
                    ).sqrt()
                    anchor_floor = (
                        0.25
                        * teacher_rms.mean(
                            dim=(0, 2, 3, 4, 5),
                            keepdim=True,
                        )
                    ).clamp_min(1e-3)
                    normalization = torch.sqrt(
                        teacher_rms.square() + anchor_floor.square()
                    )
                    normalized_rows = F.smooth_l1_loss(
                        prediction / normalization,
                        teacher / normalization,
                        reduction="none",
                    ).mean(dim=-1, keepdim=True)
                    prediction_direction = prediction / torch.sqrt(
                        prediction.square().mean(dim=-1, keepdim=True)
                        + anchor_floor.square()
                    )
                    teacher_direction = teacher / torch.sqrt(
                        teacher.square().mean(dim=-1, keepdim=True)
                        + anchor_floor.square()
                    )
                    direction_rows = 1.0 - (
                        prediction_direction * teacher_direction
                    ).mean(dim=-1, keepdim=True)
                    rows = rows + normalized_rows + 0.10 * direction_rows
                row_weight = (
                    teacher_reliability
                    if reliability_weighted
                    else torch.ones_like(teacher_reliability)
                )
                if broadcast_slot_mask is not None:
                    row_weight = row_weight * broadcast_slot_mask
                # Divide by the valid element count, never reliability mass:
                # weak matching reduces unreliable delta pressure without
                # magnifying the few surviving cells.
                component = (rows * row_weight).sum() / max(
                    stage_denominator, 1.0
                )
                effect_components[f"{stage_name}_{name}"] = component
                if stage_name == "w2" and not window_effect:
                    effect_components[name] = component
                stage_total = stage_total + float(internal_weight) * component
            return float(stage_weight) * stage_total

        if window_effect:
            w1_slot_mask = output[
                "flow_jepa_future_effect_w1_slot_valid"
            ].detach().float()
            w2_slot_mask = w1_slot_mask.new_tensor((0.0, 0.0, 1.0))
            loss = loss + supervise_effect_stage(
                prediction_prefix="flow_jepa_future_effect_w1_",
                stage_name="w1",
                stage_weight=2.0 / 3.0,
                slot_mask=w1_slot_mask,
            )
            loss = loss + supervise_effect_stage(
                prediction_prefix="",
                stage_name="w2",
                stage_weight=1.0 / 3.0,
                slot_mask=w2_slot_mask,
            )
            # The generic loss name means the complete three-slot interface,
            # not merely W2's late slot.  Keep W1/W2 component rows as well so
            # a weak near/mid stage cannot be hidden by a healthy late stage
            # (or vice versa).
            for name, *_ in v116_effect_names:
                effect_components[name] = (
                    (2.0 / 3.0) * effect_components[f"w1_{name}"]
                    + (1.0 / 3.0) * effect_components[f"w2_{name}"]
                )
            final_semantic = output[
                "flow_jepa_future_effect_semantic_pred_slots"
            ].float()
            teacher_semantic = output[
                "flow_jepa_future_effect_semantic_target_slots"
            ].detach().float()
            prediction_transition = (
                final_semantic[:, 1:] - final_semantic[:, :-1]
            )
            teacher_transition = (
                teacher_semantic[:, 1:] - teacher_semantic[:, :-1]
            )
            transition_rows = F.smooth_l1_loss(
                prediction_transition,
                teacher_transition,
                reduction="none",
            ).mean(dim=-1, keepdim=True)
            transition_reliability = torch.minimum(
                teacher_reliability[:, 1:],
                teacher_reliability[:, :-1],
            )
            transition_component = (
                transition_rows * transition_reliability
            ).sum() / float(max(transition_reliability.numel(), 1))
            effect_components["relative_transition"] = transition_component
            loss = loss + 0.10 * transition_component
        else:
            loss = loss + supervise_effect_stage(
                prediction_prefix="flow_jepa_future_effect_w1_",
                stage_name="w1",
                stage_weight=0.25,
            )
            loss = loss + supervise_effect_stage(
                prediction_prefix="",
                stage_name="w2",
                stage_weight=0.75,
            )
        effect_available = True
    elif effect_available:
        teacher_reliability = output.get(
            "flow_jepa_future_effect_reliability_target_slots"
        )
        if not torch.is_tensor(teacher_reliability):
            raise RuntimeError("FutureEffect supervision requires teacher reliability")
        teacher_reliability = teacher_reliability.detach().float()
        # The fallback target (current fact / zero delta / identity transport)
        # remains meaningful when matching is weak. Reliability modulates the
        # precision of the supervision but cannot erase the whole objective.
        effect_weight = 0.25 + 0.75 * teacher_reliability.clamp(0.0, 1.0)
        weight_denominator = effect_weight.sum().clamp_min(1.0)
        for (
            name,
            prediction_key,
            target_key,
            internal_weight,
        ) in effect_names:
            prediction = output[prediction_key].float()
            teacher = output[target_key].detach().float()
            if tuple(prediction.shape) != tuple(teacher.shape):
                raise ValueError(f"FutureEffect {name} prediction and teacher do not align")
            raw_rows = F.smooth_l1_loss(
                prediction,
                teacher,
                reduction="none",
            ).mean(dim=-1, keepdim=True)
            if name == "semantic":
                # A semantic-delta field is intentionally near zero at
                # initialization and for reliable static facts.  Normalize
                # with a detached teacher-scale floor, not F.normalize, so a
                # zero prediction cannot acquire an inverse-norm gradient.
                teacher_rms = teacher.square().mean(dim=-1, keepdim=True).sqrt()
                anchor_floor = (
                    0.25
                    * teacher_rms.mean(
                        dim=(0, 2, 3, 4, 5),
                        keepdim=True,
                    )
                ).clamp_min(1e-3)
                normalization = torch.sqrt(teacher_rms.square() + anchor_floor.square())
                normalized_rows = F.smooth_l1_loss(
                    prediction / normalization,
                    teacher / normalization,
                    reduction="none",
                ).mean(dim=-1, keepdim=True)
                prediction_direction = prediction / torch.sqrt(
                    prediction.square().mean(dim=-1, keepdim=True) + anchor_floor.square()
                )
                teacher_direction = teacher / torch.sqrt(
                    teacher.square().mean(dim=-1, keepdim=True) + anchor_floor.square()
                )
                direction_rows = 1.0 - (prediction_direction * teacher_direction).mean(
                    dim=-1, keepdim=True
                )
                rows = raw_rows + normalized_rows + 0.10 * direction_rows
            else:
                rows = raw_rows
            component = (rows * effect_weight).sum() / weight_denominator
            effect_components[name] = component
            loss = loss + float(internal_weight) * component
    reliability_mean_h = reliability_weight.sum(dim=(0, 2)) / counts_h.clamp_min(1.0)
    result = {
        "flow_jepa_interval_stage": loss,
        "flow_jepa_interval_stage_raw": raw_h.mean().detach(),
        "flow_jepa_interval_stage_normalized": normalized_h.mean().detach(),
        "flow_jepa_interval_stage_direction": direction_h.mean().detach(),
        "flow_jepa_interval_stage_endpoint": endpoint_h_loss.mean().detach(),
        "flow_jepa_interval_stage_target_scale": target_scale_h.mean().detach(),
        "flow_jepa_interval_stage_reliability": reliability_mean_h.mean().detach(),
        "flow_jepa_interval_stage_direction_floor_min": (direction_floor_h.amin().detach()),
        "flow_jepa_future_effect_supervision_active": loss.new_tensor(float(effect_available)),
    }
    if grounded_core_loss is not None:
        # This tensor intentionally remains differentiable: the existing
        # future-loss weight is routed to this exact supervised field.
        result["grounded_future_effect_core"] = grounded_core_loss
        result["grounded_slot_reduced_interval_audit"] = (
            legacy_interval_loss.detach()
        )
    for name, component in effect_components.items():
        result[f"flow_jepa_future_effect_{name}_loss"] = component.detach()
    result.update(effect_diagnostics)
    offsets = tuple(int(value) for value in output.get("flow_jepa_future_offsets", ()))
    if len(offsets) == anchors:
        for index, offset in enumerate(offsets):
            result[f"flow_jepa_interval_stage_horizon_{offset}_loss"] = loss_h[index].detach()
            result[f"flow_jepa_interval_stage_horizon_{offset}_target_scale"] = target_scale_h[
                index
            ].detach()
            result[f"flow_jepa_interval_stage_horizon_{offset}_reliability"] = reliability_mean_h[
                index
            ].detach()
    return result


def flow_jepa_stage1_losses(
    output: dict[str, Tensor],
    trainer: V39PolicyTrainerConfig,
    *,
    enable_future_loss: bool = True,
) -> dict[str, Tensor]:
    """V95 Stage1 objective for the new top representation.

    Stage1 must train the Flow-DINO/JEPA representation itself rather than the
    historical layer-contract rollout surrogate.  The frozen future DINO
    targets supervise the far stage and masked window predictions directly;
    flow correspondence terms remain auxiliary geometry constraints.  No
    deploy action or execution-controller loss is part of this objective.
    """

    if not enable_future_loss:
        raise RuntimeError("V95 Stage1 requires frozen future targets on every training batch")

    pred = output.get("flow_jepa_future_pred")
    if not torch.is_tensor(pred):
        raise RuntimeError("V95 Stage1 requires flow_jepa_future_pred")
    total = pred.float().sum() * 0.0
    losses: dict[str, Tensor] = {}
    balance_horizons = _flow_jepa_balance_horizons(trainer)
    losses["flow_jepa_horizon_balance_active"] = torch.as_tensor(
        float(balance_horizons),
        device=pred.device,
    )
    reliable_normalization = _flow_jepa_uses_reliable_normalization(trainer)
    losses["flow_jepa_future_reliable_normalization"] = torch.as_tensor(
        float(reliable_normalization),
        device=pred.device,
    )
    for key, value in output.items():
        if key.startswith("flow_jepa_") and torch.is_tensor(value) and value.numel() == 1:
            losses[key] = value.detach().float().reshape(())

    def add(name: str, value: Tensor, weight: float) -> None:
        nonlocal total
        effective_weight = max(float(weight), 0.0)
        losses[name] = value.detach().float().reshape(())
        contribution = effective_weight * value
        losses[f"loss_contrib_{name}"] = contribution.detach().float().reshape(())
        if enable_future_loss and effective_weight > 0.0:
            total = total + contribution

    grounded_effect_active = torch.is_tensor(
        output.get("grounded_intent_effect_active")
    )
    legacy_future_prediction = flow_jepa_future_prediction_loss(
        output,
        balance_horizons=balance_horizons,
        reliable_normalization=reliable_normalization,
    )
    interval_terms: dict[str, Tensor] | None = None
    future_prediction = legacy_future_prediction
    if grounded_effect_active:
        interval_terms = flow_jepa_interval_stage_terms(output)
        grounded_core = interval_terms.get("grounded_future_effect_core")
        if not torch.is_tensor(grounded_core):
            raise RuntimeError(
                "grounded future objective lost its object-level "
                "FutureEffect core"
            )
        future_prediction = grounded_core
        losses["grounded_slot_reduced_future_audit"] = (
            legacy_future_prediction.detach().float().reshape(())
        )
    add(
        "flow_jepa_future_prediction",
        future_prediction,
        float(getattr(trainer, "flow_jepa_future_loss_weight", 0.0)),
    )
    for key, value in flow_jepa_future_reliable_diagnostics(
        output,
        reliable_normalization=reliable_normalization,
        balance_horizons=balance_horizons,
    ).items():
        losses[key] = value.detach().float().reshape(())
    address_weight = float(getattr(trainer, "flow_jepa_horizon_address_loss_weight", 0.0))
    losses["flow_jepa_horizon_address_supervision_active"] = torch.as_tensor(
        float(address_weight > 0.0),
        device=pred.device,
    )
    if address_weight > 0.0 and not all(
        torch.is_tensor(output.get(name))
        for name in (
            "flow_jepa_horizon_address_logits",
            "flow_jepa_future_target",
            "flow_jepa_current_target",
        )
    ):
        raise RuntimeError(
            "active horizon-address supervision requires forward logits and "
            "current/future frozen teacher charts"
        )
    address_terms = _flow_jepa_horizon_address_terms(output)
    for key, value in address_terms.items():
        if key != "flow_jepa_horizon_address":
            losses[key] = value.detach().float().reshape(())
    add(
        "flow_jepa_horizon_address",
        address_terms["flow_jepa_horizon_address"],
        address_weight,
    )
    if interval_terms is None:
        interval_terms = flow_jepa_interval_stage_terms(output)
    for key, value in interval_terms.items():
        if key != "flow_jepa_interval_stage":
            losses[key] = value.detach().float().reshape(())
    add(
        "flow_jepa_interval_stage",
        interval_terms["flow_jepa_interval_stage"],
        float(
            getattr(
                trainer,
                "flow_jepa_interval_stage_loss_weight",
                0.0,
            )
        ),
    )
    future_change_weight = float(getattr(trainer, "flow_jepa_future_change_loss_weight", 0.0))
    if future_change_weight > 0.0 and not all(
        torch.is_tensor(output.get(name))
        for name in (
            "flow_jepa_future_pred",
            "flow_jepa_future_target",
            "flow_jepa_current_target",
            "flow_jepa_future_target_mask",
        )
    ):
        raise RuntimeError("active future-change supervision requires current/future JEPA charts")
    future_change = flow_jepa_future_change_loss(
        output,
        balance_horizons=balance_horizons,
    )
    if grounded_effect_active:
        losses["flow_jepa_future_change"] = (
            future_change.detach().float().reshape(())
        )
        losses["grounded_slot_reduced_future_change_audit"] = (
            future_change.detach().float().reshape(())
        )
        losses["grounded_slot_reduced_future_change_audit_only"] = (
            future_change.new_ones((), dtype=torch.float32)
        )
        losses["loss_contrib_flow_jepa_future_change"] = (
            future_change.detach().float().reshape(()) * 0.0
        )
    else:
        add(
            "flow_jepa_future_change",
            future_change,
            future_change_weight,
        )
    for offset, value in flow_jepa_future_horizon_diagnostics(
        output,
        reliable_normalization=reliable_normalization,
    ).items():
        losses[f"flow_jepa_future_horizon_{offset}"] = value.detach().float().reshape(())
    losses["flow_jepa_future_change_direction"] = (
        flow_jepa_future_change_direction_loss(
            output,
            balance_horizons=balance_horizons,
        )
        .detach()
        .float()
        .reshape(())
    )
    add(
        "flow_jepa_stage_prediction",
        flow_jepa_stage_prediction_loss(output),
        float(getattr(trainer, "flow_jepa_stage_loss_weight", 0.0)),
    )
    for loss_name, weight_name in (
        ("flow_jepa_warp_loss", "flow_jepa_warp_loss_weight"),
        ("flow_jepa_cycle_loss", "flow_jepa_cycle_loss_weight"),
        ("flow_jepa_smoothness_loss", "flow_jepa_smoothness_loss_weight"),
        ("flow_jepa_uncertainty_nll", "flow_jepa_uncertainty_nll_weight"),
        (
            "flow_jepa_refinement_sequence_loss",
            "flow_jepa_refinement_sequence_loss_weight",
        ),
    ):
        value = output.get(loss_name)
        if not torch.is_tensor(value) or value.numel() != 1:
            raise RuntimeError(f"V95 Stage1 did not expose {loss_name}")
        add(loss_name, value, float(getattr(trainer, weight_name, 0.0)))
    identity_weight = float(getattr(trainer, "flow_jepa_identity_advantage_loss_weight", 0.0))
    identity_value = output.get("flow_jepa_identity_advantage_loss")
    if torch.is_tensor(identity_value) and identity_value.numel() == 1:
        add("flow_jepa_identity_advantage_loss", identity_value, identity_weight)
    elif identity_weight > 0.0:
        raise RuntimeError(
            "identity-advantage supervision requires raw-image zero-flow guard output"
        )
    static_identity_weight = float(getattr(trainer, "flow_jepa_static_identity_loss_weight", 0.0))
    static_identity_value = output.get("flow_jepa_static_identity_loss")
    if torch.is_tensor(static_identity_value) and static_identity_value.numel() == 1:
        add(
            "flow_jepa_static_identity_loss",
            static_identity_value,
            static_identity_weight,
        )
    elif static_identity_weight > 0.0:
        raise RuntimeError("static-identity supervision requires raw-image zero-flow guard output")

    if torch.is_grad_enabled() and not total.requires_grad:
        raise RuntimeError("V95 Stage1 objective has no active differentiable term")
    losses["loss"] = total
    losses["loss_group_representation"] = total.detach().float().reshape(())
    losses["loss_ledger_sum"] = total.detach().float().reshape(())
    losses["loss_ledger_residual"] = total.detach().float().reshape(()) - losses["loss_ledger_sum"]
    losses["flow_jepa_stage1_objective"] = torch.ones_like(losses["loss_group_representation"])
    return losses


def rollout_dynamics_loss(output: dict[str, Tensor]) -> Tensor:
    """Supervise action-conditioned rollout latent against future residual.

    The target is stop-gradient DINO future-current residual. It is never fed
    as an input to the model, so this cannot become future self-denoising.
    """
    if "rollout_effect_target" not in output:
        device = output["pred_physical_velocity"].device
        return torch.zeros((), device=device, dtype=output["pred_physical_velocity"].dtype)
    pred = output["rollout_effect_pred"].float()
    target = output["rollout_effect_target"].float().detach()
    smooth = F.smooth_l1_loss(pred, target)
    pred_n = F.normalize(pred, dim=-1)
    target_n = F.normalize(target, dim=-1)
    cosine = (1.0 - (pred_n * target_n).sum(dim=-1)).mean()
    return smooth + 0.10 * cosine


def rollout_contrast_loss(output: dict[str, Tensor], *, margin: float = 0.02) -> Tensor:
    """Force real action-controlled local effects to beat counterfactual actions."""
    if "rollout_effect_target" not in output:
        device = output["pred_physical_velocity"].device
        return torch.zeros((), device=device, dtype=output["pred_physical_velocity"].dtype)
    state = None
    if (
        "milestone_step_delta_pred_hold_action" in output
        and "milestone_step_delta_pred_shuffle_action" in output
    ):
        target = _milestone_step_delta_target(output)
        real_pred = output["milestone_step_delta_pred"]
        hold_pred = output["milestone_step_delta_pred_hold_action"]
        shuf_pred = output["milestone_step_delta_pred_shuffle_action"]
        shapes = [real_pred.shape[1], hold_pred.shape[1], shuf_pred.shape[1], target.shape[1]]
        if "milestone_step_delta_pred_shuffle_state" in output:
            shapes.append(output["milestone_step_delta_pred_shuffle_state"].shape[1])
        steps = min(shapes)
        real = _effect_distance(real_pred[:, :steps], target[:, :steps])
        hold = _effect_distance(hold_pred[:, :steps], target[:, :steps])
        shuf = _effect_distance(shuf_pred[:, :steps], target[:, :steps])
        if "milestone_step_delta_pred_shuffle_state" in output:
            state = _effect_distance(
                output["milestone_step_delta_pred_shuffle_state"][:, :steps], target[:, :steps]
            )
    elif "rollout_delta_pred_hold_action" in output:
        target = _rollout_residual_target(output)
        real = _effect_distance(output["rollout_delta_pred"], target)
        hold = _effect_distance(output["rollout_delta_pred_hold_action"], target)
        shuf = _effect_distance(output["rollout_delta_pred_shuffle_action"], target)
        if "rollout_delta_pred_shuffle_state" in output:
            state = _effect_distance(output["rollout_delta_pred_shuffle_state"], target)
    else:
        device = output["pred_physical_velocity"].device
        return torch.zeros((), device=device, dtype=output["pred_physical_velocity"].dtype)
    m = torch.as_tensor(float(margin), device=real.device, dtype=real.dtype)
    loss = F.relu(m + real - hold) + F.relu(m + real - shuf)
    if state is not None:
        loss = loss + F.relu(m + real - state)
    return loss.mean()


def rollout_diagnostics(output: dict[str, Tensor]) -> dict[str, Tensor]:
    rows: dict[str, Tensor] = {}
    if "rollout_effect_target" not in output:
        device = output["pred_physical_velocity"].device
        z = torch.zeros((), device=device, dtype=output["pred_physical_velocity"].dtype)
        for key in (
            "rollout_distance_real",
            "rollout_distance_hold",
            "rollout_distance_shuffle",
            "rollout_delta_hold",
            "rollout_delta_shuffle",
            "rollout_full_distance_real",
        ):
            rows[key] = z
        return rows
    target_full = output["rollout_effect_target"].float().detach()
    full_real = _effect_distance(output["rollout_effect_pred"], target_full).mean()
    rows["rollout_full_distance_real"] = full_real.detach()
    rows["rollout_distance_real"] = full_real.detach()
    rows["rollout_base_mse"] = (
        (
            output.get("rollout_base_effect_pred", output["rollout_effect_pred"]).float()
            - target_full
        )
        .square()
        .mean()
        .detach()
    )
    rows["rollout_delta_target_norm"] = (
        _rollout_residual_target(output).detach().float().norm(dim=-1).mean()
    )
    if "rollout_delta_pred" in output:
        target_delta = _rollout_residual_target(output)
        delta_real = _effect_distance(output["rollout_delta_pred"], target_delta).mean()
        rows["rollout_delta_distance_real"] = delta_real.detach()
        # In V38.6 the legacy rollout_distance/delta names refer to the
        # controlled-delta contrast path because that is the causal path being
        # tested by train logs and offline diagnostics.
        rows["rollout_distance_real"] = delta_real.detach()
    pred = output["rollout_effect_pred"].float()
    target = target_full
    steps = min(pred.shape[1], target.shape[1])
    pred_s = pred[:, :steps]
    target_s = target[:, :steps]
    pred_std = pred_s.std(dim=(0, 1), unbiased=False).mean().detach()
    target_std = target_s.std(dim=(0, 1), unbiased=False).mean().detach().clamp_min(1e-6)
    pred_norm = pred_s.norm(dim=-1).mean().detach()
    target_norm = target_s.norm(dim=-1).mean().detach().clamp_min(1e-6)
    rows["rollout_pred_std_norm"] = pred_std
    rows["rollout_target_std_norm"] = target_std
    rows["rollout_pred_std_ratio"] = pred_std / target_std
    rows["rollout_pred_norm_ratio"] = pred_norm / target_norm
    grid = _future_grid_count(None, output)
    try:
        pred_m = _reshape_milestones(pred_s, grid=grid)
        target_m = _reshape_milestones(target_s, grid=grid)
        k = min(pred_m.shape[1], target_m.shape[1])
        pred_m = pred_m[:, :k]
        target_m = target_m[:, :k]
        z_pred = torch.zeros_like(pred_m[:, :1])
        z_target = torch.zeros_like(target_m[:, :1])
        pred_step = pred_m - torch.cat([z_pred, pred_m[:, :-1]], dim=1)
        target_step = target_m - torch.cat([z_target, target_m[:, :-1]], dim=1)
        pred_step_norm = pred_step.norm(dim=-1).mean().detach()
        target_step_norm = target_step.norm(dim=-1).mean().detach().clamp_min(1e-6)
        rows["rollout_milestone_delta_pred_norm"] = pred_step_norm
        rows["rollout_milestone_delta_target_norm"] = target_step_norm
        rows["rollout_milestone_delta_norm_ratio"] = pred_step_norm / target_step_norm
    except Exception:
        pass
    if "milestone_step_delta_pred_hold_action" in output:
        target_step = _milestone_step_delta_target(output)
        steps = min(output["milestone_step_delta_pred"].shape[1], target_step.shape[1])
        real_step = _effect_distance(
            output["milestone_step_delta_pred"][:, :steps], target_step[:, :steps]
        ).mean()
        hold_step = _effect_distance(
            output["milestone_step_delta_pred_hold_action"][:, :steps], target_step[:, :steps]
        ).mean()
        shuf_step = _effect_distance(
            output["milestone_step_delta_pred_shuffle_action"][:, :steps], target_step[:, :steps]
        ).mean()
        rows["step_delta_distance_real"] = real_step.detach()
        rows["step_delta_distance_hold"] = hold_step.detach()
        rows["step_delta_distance_shuffle"] = shuf_step.detach()
        rows["step_delta_hold"] = (hold_step - real_step).detach()
        rows["step_delta_shuffle"] = (shuf_step - real_step).detach()
        rows["step_delta_change_hold"] = (
            (
                output["milestone_step_delta_pred"].float()
                - output["milestone_step_delta_pred_hold_action"].float()
            )
            .square()
            .mean()
            .detach()
        )
        rows["step_delta_change_shuffle"] = (
            (
                output["milestone_step_delta_pred"].float()
                - output["milestone_step_delta_pred_shuffle_action"].float()
            )
            .square()
            .mean()
            .detach()
        )
        if "milestone_step_delta_pred_shuffle_state" in output:
            state_steps = min(
                output["milestone_step_delta_pred_shuffle_state"].shape[1], target_step.shape[1]
            )
            state_step = _effect_distance(
                output["milestone_step_delta_pred_shuffle_state"][:, :state_steps],
                target_step[:, :state_steps],
            ).mean()
            rows["step_delta_state_shuffle"] = (state_step - real_step).detach()
            rows["step_delta_change_state_shuffle"] = (
                (
                    output["milestone_step_delta_pred"][:, :state_steps].float()
                    - output["milestone_step_delta_pred_shuffle_state"][:, :state_steps].float()
                )
                .square()
                .mean()
                .detach()
            )
    if "rollout_delta_pred_hold_action" in output:
        target_delta = _rollout_residual_target(output)
        hold = _effect_distance(output["rollout_delta_pred_hold_action"], target_delta).mean()
        shuf = _effect_distance(output["rollout_delta_pred_shuffle_action"], target_delta).mean()
        real = rows["rollout_distance_real"]
        rows["rollout_distance_hold"] = hold.detach()
        rows["rollout_distance_shuffle"] = shuf.detach()
        rows["rollout_delta_hold"] = (hold - real).detach()
        rows["rollout_delta_shuffle"] = (shuf - real).detach()
        delta_change_hold = (
            (
                output["rollout_delta_pred"].float()
                - output["rollout_delta_pred_hold_action"].float()
            )
            .square()
            .mean()
        )
        delta_change_shuffle = (
            (
                output["rollout_delta_pred"].float()
                - output["rollout_delta_pred_shuffle_action"].float()
            )
            .square()
            .mean()
        )
        full_change_hold = (
            (
                output["rollout_effect_pred"].float()
                - output["rollout_effect_pred_hold_action"].float()
            )
            .square()
            .mean()
        )
        full_change_shuffle = (
            (
                output["rollout_effect_pred"].float()
                - output["rollout_effect_pred_shuffle_action"].float()
            )
            .square()
            .mean()
        )
        rows["rollout_effect_change_hold"] = delta_change_hold.detach()
        rows["rollout_effect_change_shuffle"] = delta_change_shuffle.detach()
        rows["rollout_full_effect_change_hold"] = full_change_hold.detach()
        rows["rollout_full_effect_change_shuffle"] = full_change_shuffle.detach()
        rows["rollout_hold_cancellation_fraction"] = torch.where(
            delta_change_hold > 1e-8,
            1.0 - full_change_hold / delta_change_hold.clamp_min(1e-8),
            torch.zeros_like(delta_change_hold),
        ).detach()
        rows["rollout_shuffle_cancellation_fraction"] = torch.where(
            delta_change_shuffle > 1e-8,
            1.0 - full_change_shuffle / delta_change_shuffle.clamp_min(1e-8),
            torch.zeros_like(delta_change_shuffle),
        ).detach()
        if "rollout_base_effect_pred" in output:
            base = output["rollout_base_effect_pred"].float()
            rows["rollout_effect_delta_gap"] = (
                (output["rollout_effect_pred"].float() - output["rollout_delta_pred"].float())
                .square()
                .mean()
                .detach()
            )
            if "rollout_base_effect_pred_hold_action" in output:
                rows["rollout_base_change_hold"] = (
                    (base - output["rollout_base_effect_pred_hold_action"].float())
                    .square()
                    .mean()
                    .detach()
                )
            if "rollout_base_effect_pred_shuffle_action" in output:
                rows["rollout_base_change_shuffle"] = (
                    (base - output["rollout_base_effect_pred_shuffle_action"].float())
                    .square()
                    .mean()
                    .detach()
                )
        if "rollout_delta_pred_shuffle_state" in output:
            state = _effect_distance(
                output["rollout_delta_pred_shuffle_state"], target_delta
            ).mean()
            rows["rollout_delta_state_shuffle"] = (state - real).detach()
            rows["rollout_effect_change_state_shuffle"] = (
                (
                    output["rollout_delta_pred"].float()
                    - output["rollout_delta_pred_shuffle_state"].float()
                )
                .square()
                .mean()
                .detach()
            )
        if "rollout_effect_pred_shuffle_state" in output:
            rows["rollout_full_effect_change_state_shuffle"] = (
                (
                    output["rollout_effect_pred"].float()
                    - output["rollout_effect_pred_shuffle_state"].float()
                )
                .square()
                .mean()
                .detach()
            )
    for key in (
        "rollout_coeff_abs_mean",
        "rollout_neutral_coeff_abs_mean",
        "rollout_centered_coeff_abs_mean",
        "rollout_basis_norm",
        "rollout_delta_norm",
        "rollout_base_norm",
        "rollout_decomposition_expansion_ratio",
        "rollout_base_is_fixed_zero",
        "rollout_delta_gain",
        "rollout_deep_update_norm",
        "rollout_deep_token_norm",
        "milestone_gate_mean",
        "milestone_step_delta_norm",
        "milestone_effect_norm",
        "milestone_effect_std",
        "milestone_effect_gain",
    ):
        if key in output:
            rows[key] = output[key].detach().float()
    return rows


def future_latent_loss(output: dict[str, Tensor]) -> Tensor:
    # Compatibility alias: this is the full base+delta rollout loss, not a
    # future-noisy denoising loss.
    return rollout_dynamics_loss(output)


def action_effect_loss(output: dict[str, Tensor]) -> Tensor:
    # Compatibility alias: this now tracks the controlled action delta path.
    return rollout_delta_loss(output)


def _normalize_horizon_weight(weight: Tensor) -> Tensor:
    return weight / weight.mean(dim=-1, keepdim=True).clamp_min(1e-6)


def _micro_fixed_unfolded_weights(
    *,
    system: V39PolicySystem,
    trainer: V39PolicyTrainerConfig,
    steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor]:
    """Fixed coarse-to-fine supervision map for micro refine states.

    The final micro state is anchored to the normal policy horizon weighting.
    Earlier states emphasize near-horizon trajectory quality without assigning
    any micro step to a hard physical stage.
    """
    horizon = int(system.policy_config.action_horizon)
    pos = position_weights(system.policy_config, trainer, device).to(dtype=torch.float32)
    idx = torch.arange(horizon, device=device, dtype=torch.float32)
    near = pos * (0.25 + 2.0 * (idx < 4).float() + 0.6 * (idx < 8).float())
    mid = pos * (0.35 + 1.2 * (idx < 8).float() + 0.5 * torch.exp(-((idx - 7.0).square()) / 24.0))
    tail = pos * (0.35 + 1.4 * (idx >= 8).float())
    full = pos
    basis = torch.stack(
        [
            _normalize_horizon_weight(near),
            _normalize_horizon_weight(mid),
            _normalize_horizon_weight(tail),
            _normalize_horizon_weight(full),
        ],
        dim=0,
    )

    full_mix = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=torch.float32)
    if steps <= 1:
        mix = full_mix[None]
    else:
        s = torch.linspace(0.0, 1.0, steps, device=device, dtype=torch.float32)
        near_c = (1.0 - s).square()
        mid_c = torch.exp(-((s - 0.42).square()) / 0.08)
        tail_c = torch.exp(-((s - 0.72).square()) / 0.06) * (1.0 - 0.7 * s)
        full_c = s.square()
        mix = torch.stack([near_c, mid_c, tail_c, full_c], dim=-1).clamp_min(1e-4)
        mix = mix / mix.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        terminal = ((s - 0.55) / 0.45).clamp(0.0, 1.0)
        terminal = terminal.square() * (3.0 - 2.0 * terminal)
        mix = (1.0 - terminal[:, None]) * mix + terminal[:, None] * full_mix[None]
        mix[-1] = full_mix
    fixed = torch.einsum("sk,kt->st", mix, basis)
    return (
        _normalize_horizon_weight(fixed).to(dtype=dtype),
        mix.to(dtype=dtype),
        basis.to(dtype=dtype),
    )


def micro_refine_supervision_losses(
    system: V39PolicySystem,
    sample: dict[str, Tensor],
    output: dict[str, Tensor],
    trainer: V39PolicyTrainerConfig,
    *,
    global_step: int | None = None,
) -> dict[str, Tensor]:
    pred = output.get("latent_cvae_adaptive_micro_pred_velocity")
    if not isinstance(pred, Tensor) or pred.ndim != 4:
        ref = output["pred_physical_velocity"]
        z = torch.zeros((), device=ref.device, dtype=ref.dtype)
        return {
            "latent_cvae_micro_supervision": z,
            "latent_cvae_micro_event": z,
            "latent_cvae_micro_monotonic": z,
            "latent_cvae_micro_weight_kl": z,
            "latent_cvae_micro_coverage_smooth": z,
            "latent_cvae_micro_coverage_floor": z,
            "latent_cvae_micro_weight_alpha": z,
            "latent_cvae_micro_step_alpha_mean": z,
            "latent_cvae_micro_weight_final_diff": z,
            "latent_cvae_micro_coverage_tail_mass": z,
        }
    device = pred.device
    dtype = pred.dtype
    batch, steps, horizon, _ = pred.shape
    target = output["target_physical_velocity"].to(device=device, dtype=dtype)
    fixed, fixed_mix, basis = _micro_fixed_unfolded_weights(
        system=system,
        trainer=trainer,
        steps=steps,
        device=device,
        dtype=dtype,
    )
    fixed_prob = fixed.float().clamp_min(1e-8)
    fixed_prob = fixed_prob / fixed_prob.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    normal = basis[-1].to(device=device, dtype=dtype)
    normal_prob = normal.float().clamp_min(1e-8)
    normal_prob = normal_prob / normal_prob.sum().clamp_min(1e-8)
    prior_scale = max(
        float(getattr(trainer, "latent_cvae_micro_coverage_prior_logit_scale", 0.25)), 0.0
    )

    logits = output.get("latent_cvae_adaptive_micro_supervision_logits")
    if (
        isinstance(logits, Tensor)
        and logits.ndim == 3
        and int(logits.shape[1]) == steps
        and int(logits.shape[2]) == horizon
    ):
        logits_f = logits.to(device=device, dtype=torch.float32)
        prior_logits = prior_scale * fixed_prob.clamp_min(1e-8).log()[None]
        learned_prob = torch.softmax(logits_f + prior_logits, dim=-1)
    elif (
        isinstance(logits, Tensor)
        and logits.ndim == 3
        and int(logits.shape[1]) == steps
        and int(logits.shape[2]) == 4
    ):
        logits_f = logits.to(device=device, dtype=torch.float32)
        mix_prior = prior_scale * fixed_mix.float().clamp_min(1e-8).log()[None]
        learned_mix = torch.softmax(logits_f + mix_prior, dim=-1).to(dtype=dtype)
        learned = torch.einsum("bsk,kt->bst", learned_mix, basis)
        learned_prob = learned.float().clamp_min(1e-8)
        learned_prob = learned_prob / learned_prob.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    else:
        learned_prob = fixed_prob[None].expand(batch, -1, -1)
    learned = (learned_prob * float(horizon)).to(dtype=dtype)

    ramp_steps = max(int(getattr(trainer, "latent_cvae_micro_learned_ramp_steps", 2000)), 1)
    max_alpha = max(float(getattr(trainer, "latent_cvae_micro_learned_weight_max", 0.25)), 0.0)
    step_value = 0 if global_step is None else max(int(global_step), 0)
    alpha_value = min(max_alpha, max_alpha * float(step_value) / float(ramp_steps))
    alpha = torch.as_tensor(alpha_value, device=device, dtype=dtype)
    step_alpha_scale = (1.0 - fixed_mix[:, 3]).clamp(0.0, 1.0).to(device=device, dtype=dtype)
    step_alpha = alpha * step_alpha_scale[None, :, None]
    weights = (1.0 - step_alpha) * fixed[None] + step_alpha * learned
    floor = max(float(getattr(trainer, "latent_cvae_micro_weight_floor", 0.05)), 0.0)
    weights = _normalize_horizon_weight(weights.clamp_min(floor))
    final_normal = normal[None, None].expand(batch, 1, horizon)
    weights = torch.cat([weights[:, :-1], final_normal], dim=1) if steps > 1 else final_normal

    physical_error = semantic_physical_velocity_error(
        system,
        pred - target[:, None],
        arm_null_weight=trainer.arm_manifold_null_weight,
    )
    micro_flow = (physical_error * weights).mean()
    mono_weight = normal.to(device=device, dtype=dtype)[None, None]
    mono_error = (physical_error * mono_weight).mean(dim=-1)
    monotonic = (
        F.relu(mono_error[:, 1:] - mono_error[:, :-1]).mean()
        if steps > 1
        else torch.zeros((), device=device, dtype=dtype)
    )

    event_logits = output.get("latent_cvae_adaptive_micro_event_logits")
    micro_event = torch.zeros((), device=device, dtype=dtype)
    if isinstance(event_logits, Tensor) and event_logits.ndim == 4:
        labels = gripper_event_labels(
            target_raw=sample["policy_action_raw"].to(device=device),
            current_raw=sample["state_raw"].to(device=device),
            gripper_index=system.policy_config.gripper_index,
            threshold=trainer.gripper_event_threshold,
        )
        labels = labels[:, None].expand(-1, steps, -1)
        ce = F.cross_entropy(
            event_logits.reshape(-1, 3).float(), labels.reshape(-1), reduction="none"
        ).reshape(batch, steps, horizon)
        pos = (labels != 0).to(dtype=ce.dtype)
        event_weight = 1.0 + pos * max(
            float(getattr(trainer, "latent_cvae_micro_event_positive_weight", 2.0)) - 1.0, 0.0
        )
        micro_event = (ce.to(dtype=dtype) * event_weight.to(dtype=dtype) * weights).sum() / (
            event_weight.to(dtype=dtype) * weights
        ).sum().clamp_min(1.0)

    fixed_prob_b = fixed_prob[None].expand(batch, -1, -1)
    weight_kl = (
        (
            learned_prob.clamp_min(1e-8)
            * (learned_prob.clamp_min(1e-8).log() - fixed_prob_b.clamp_min(1e-8).log())
        )
        .sum(dim=-1)
        .mean()
        .to(dtype=dtype)
    )
    smooth = (
        (learned_prob[:, 1:] - learned_prob[:, :-1]).square().sum(dim=-1).mean() * float(horizon)
        if steps > 1
        else torch.zeros((), device=device, dtype=dtype)
    )
    avg_prob = learned_prob.mean(dim=1)
    tail_start = min(
        max(int(getattr(system.policy_config, "rollout_tail_start_step", 8)), 0),
        max(horizon - 1, 0),
    )
    tail_mask = torch.arange(horizon, device=device) >= tail_start
    tail_mass = avg_prob[:, tail_mask].sum(dim=-1)
    tail_target = normal_prob[tail_mask].sum() * max(
        float(getattr(trainer, "latent_cvae_micro_coverage_floor_ratio", 0.55)), 0.0
    )
    coverage_floor = (
        F.relu(tail_target.to(device=device) - tail_mass).square().mean().to(dtype=dtype)
    )
    final_weight_diff = (weights[:, -1] - normal[None]).abs().mean()
    return {
        "latent_cvae_micro_supervision": micro_flow,
        "latent_cvae_micro_event": micro_event,
        "latent_cvae_micro_monotonic": monotonic,
        "latent_cvae_micro_weight_kl": weight_kl,
        "latent_cvae_micro_coverage_smooth": smooth,
        "latent_cvae_micro_coverage_floor": coverage_floor,
        "latent_cvae_micro_weight_alpha": alpha,
        "latent_cvae_micro_step_alpha_mean": step_alpha.mean().detach(),
        "latent_cvae_micro_weight_final_diff": final_weight_diff.detach(),
        "latent_cvae_micro_coverage_tail_mass": tail_mass.detach().mean(),
        "latent_cvae_micro_weight_first": weights[:, 0, :4].mean().detach(),
        "latent_cvae_micro_weight_last": weights[:, -1].mean().detach(),
    }


def _shadow_refinement_probe_metrics(
    system: V39PolicySystem,
    output: dict[str, Tensor],
    *,
    arm_null_weight: float,
) -> dict[str, Tensor]:
    """Measure shadow early-exit quality without feeding target error to routing."""
    shadow_predictions = output.get("refinement_shadow_probe_pred_velocity")
    shadow_active = output.get("refinement_shadow_probe_active")
    fixed_predictions = output.get("refinement_probe_pred_velocity")
    fixed_active = output.get("refinement_probe_active")
    target_velocity = output.get("target_physical_velocity")
    if not all(
        torch.is_tensor(value)
        for value in (
            shadow_predictions,
            shadow_active,
            fixed_predictions,
            fixed_active,
            target_velocity,
        )
    ):
        return {}

    def final_error(predictions: Tensor, active: Tensor) -> Tensor:
        residual = predictions.float() - target_velocity.detach().float()[:, None]
        step_error = semantic_physical_velocity_error(
            system,
            residual,
            arm_null_weight=arm_null_weight,
        ).mean(dim=-1)
        final_index = active.float().sum(dim=1).long().clamp_min(1)
        return step_error.gather(1, final_index[:, None]).squeeze(1)

    with torch.no_grad():
        shadow_error = final_error(shadow_predictions, shadow_active)
        fixed_error = final_error(fixed_predictions, fixed_active)
        shadow_steps = shadow_active.float().sum(dim=1)
        fixed_steps = fixed_active.float().sum(dim=1)
        return {
            "hierarchical_mmdit_shadow_refine_error_final": shadow_error.mean(),
            "hierarchical_mmdit_shadow_refine_error_gap": (shadow_error - fixed_error).mean(),
            "hierarchical_mmdit_shadow_refine_error_ratio": (
                shadow_error / fixed_error.clamp_min(1e-8)
            ).mean(),
            "hierarchical_mmdit_shadow_step_saving": (fixed_steps - shadow_steps).mean(),
        }


def _oracle_exit_supervision(
    *,
    exit_logits: Tensor,
    candidate_error: Tensor,
    initial_error: Tensor,
    candidate_mask: Tensor,
    relative_tolerance: float,
) -> dict[str, Tensor]:
    """Train an online stop/continue head from detached full-prefix errors."""
    if exit_logits.ndim != 2:
        raise ValueError("exit_logits must be [B,S]")
    expected = tuple(exit_logits.shape)
    if tuple(candidate_error.shape) != expected or tuple(candidate_mask.shape) != expected:
        raise ValueError("oracle route tensors must share [B,S] geometry")
    if tuple(initial_error.shape) != (expected[0],):
        raise ValueError("initial_error must be [B]")
    if float(relative_tolerance) < 0.0:
        raise ValueError("relative_tolerance must be non-negative")

    device = exit_logits.device
    steps = int(exit_logits.shape[1])
    indices = torch.arange(steps, device=device, dtype=torch.long)[None]
    mask = candidate_mask.detach().bool()
    errors = candidate_error.detach().float()
    initial = initial_error.detach().float().abs().clamp_min(1e-6)
    valid = mask.any(dim=1)
    valid_float = valid.float()
    valid_denominator = valid_float.sum().clamp_min(1.0)

    masked_error = torch.where(mask, errors, torch.full_like(errors, float("inf")))
    best_error = masked_error.min(dim=1).values
    safe_best_error = torch.where(valid, best_error, torch.zeros_like(best_error))
    tolerance = float(relative_tolerance) * initial
    near_best = mask & (errors <= best_error[:, None] + tolerance[:, None])
    oracle_index = near_best.float().argmax(dim=1)

    decision_mask = mask & (indices <= oracle_index[:, None])
    stop_target = (indices == oracle_index[:, None]).to(dtype=exit_logits.dtype)
    decision_float = decision_mask.to(dtype=exit_logits.dtype)
    route_loss = (
        F.binary_cross_entropy_with_logits(
            exit_logits.float(), stop_target.float(), reduction="none"
        )
        * decision_float.float()
    ).sum() / decision_float.float().sum().clamp_min(1.0)

    with torch.no_grad():
        probabilities = torch.sigmoid(exit_logits.detach().float())
        candidate_depth = mask.long().cumsum(dim=1)
        predicted_stop = mask & (probabilities > 0.5)
        sentinel = torch.full_like(indices, steps)
        first_predicted = torch.where(predicted_stop, indices, sentinel).min(dim=1).values
        last_candidate = torch.where(mask, indices, torch.full_like(indices, -1)).max(dim=1).values
        predicted_index = torch.where(
            first_predicted < steps,
            first_predicted,
            last_candidate.clamp_min(0),
        )
        predicted_error = errors.gather(1, predicted_index[:, None]).squeeze(1)
        oracle_error = errors.gather(1, oracle_index[:, None]).squeeze(1)
        oracle_depth = candidate_depth.gather(1, oracle_index[:, None]).squeeze(1).float()
        predicted_depth = candidate_depth.gather(1, predicted_index[:, None]).squeeze(1).float()
        mean_probability = (probabilities * mask.float()).sum() / mask.float().sum().clamp_min(1.0)

    return {
        "loss": route_loss,
        "target_depth": (oracle_depth * valid_float).sum() / valid_denominator,
        "predicted_depth": (predicted_depth * valid_float).sum() / valid_denominator,
        "depth_accuracy": ((predicted_index == oracle_index).float() * valid_float).sum()
        / valid_denominator,
        "depth_mae": ((predicted_depth - oracle_depth).abs() * valid_float).sum()
        / valid_denominator,
        "best_error": (safe_best_error * valid_float).sum() / valid_denominator,
        "target_error": (oracle_error * valid_float).sum() / valid_denominator,
        "predicted_error": (predicted_error * valid_float).sum() / valid_denominator,
        "predicted_regret": ((predicted_error - safe_best_error).clamp_min(0.0) * valid_float).sum()
        / valid_denominator,
        "stop_probability": mean_probability,
        "valid_fraction": valid_float.mean(),
    }


def object_intent_dynamics_terms(
    output: dict[str, Tensor],
    *,
    require_teacher: bool,
) -> dict[str, Tensor]:
    """Losses for the sole object-level G->S->W interface.

    The existing future weight owns W's complete four-interval dynamics.  The
    existing interval weight owns chronological differentiation plus the
    small G/S recognizer scaffolding.  No reliability denominator can cancel
    attenuation, and visibility/persistence are supervised for existing
    objects even when a future match is occluded.
    """

    reference = output.get("pred_physical_velocity")
    if not torch.is_tensor(reference):
        raise KeyError("object-intent loss requires the policy velocity reference")
    zero = reference.float().sum() * 0.0
    prediction_target_pairs = (
        (
            "successor",
            "object_future_successor_prediction",
            "object_future_successor_target",
            0.30,
            False,
        ),
        (
            "semantic",
            "object_future_semantic_prediction",
            "object_future_semantic_target",
            0.25,
            True,
        ),
        (
            "transport",
            "object_future_transport_prediction",
            "object_future_transport_target",
            0.15,
            False,
        ),
        (
            "covariance",
            "object_future_covariance_prediction",
            "object_future_covariance_target",
            0.05,
            False,
        ),
        (
            "visibility",
            "object_future_visibility_prediction",
            "object_future_visibility_target",
            0.08,
            False,
        ),
        (
            "persistence",
            "object_future_persistence_prediction",
            "object_future_persistence_target",
            0.07,
            False,
        ),
        (
            "uncertainty",
            "object_future_uncertainty_prediction",
            "object_future_uncertainty_target",
            0.10,
            False,
        ),
    )
    teacher_available = all(
        torch.is_tensor(output.get(prediction_key))
        and torch.is_tensor(output.get(target_key))
        for _, prediction_key, target_key, _, _ in prediction_target_pairs
    ) and torch.is_tensor(output.get("object_future_validity_target"))
    if not teacher_available:
        if require_teacher:
            raise RuntimeError(
                "object-intent dynamics requires the four-interval object teacher"
            )
        return {
            "object_future_dynamics": zero,
            "object_future_transition": zero,
            "object_intent_structure": zero,
        }

    teacher_validity = output["object_future_validity_target"].detach().float()
    current_validity = output.get("object_fact_validity")
    if not torch.is_tensor(current_validity):
        raise RuntimeError("object-intent loss lost physical current object validity")
    validity_weight = current_validity.detach().float()[:, None].expand_as(
        teacher_validity
    )
    if teacher_validity.ndim != 4 or int(teacher_validity.shape[1]) != 4:
        raise ValueError(
            "object future validity must preserve [B,4,K,1]"
        )

    def row_loss(
        prediction: Tensor,
        teacher: Tensor,
        *,
        scale_floored: bool,
    ) -> Tensor:
        raw = F.smooth_l1_loss(
            prediction.float(), teacher.detach().float(), reduction="none"
        ).mean(dim=-1, keepdim=True)
        if not scale_floored:
            return raw
        teacher_f = teacher.detach().float()
        prediction_f = prediction.float()
        teacher_rms = teacher_f.square().mean(dim=-1, keepdim=True).sqrt()
        scale_floor = (
            0.25 * teacher_rms.mean(dim=(0, 2), keepdim=True)
        ).clamp_min(1e-3)
        scale = torch.sqrt(teacher_rms.square() + scale_floor.square())
        normalized = F.smooth_l1_loss(
            prediction_f / scale,
            teacher_f / scale,
            reduction="none",
        ).mean(dim=-1, keepdim=True)
        prediction_direction = prediction_f / torch.sqrt(
            prediction_f.square().mean(dim=-1, keepdim=True)
            + scale_floor.square()
        )
        teacher_direction = teacher_f / torch.sqrt(
            teacher_f.square().mean(dim=-1, keepdim=True)
            + scale_floor.square()
        )
        direction = 1.0 - (
            prediction_direction * teacher_direction
        ).mean(dim=-1, keepdim=True)
        # A genuinely zero semantic change has no defined direction.  Without
        # this smooth target-strength factor, an exact zero prediction/target
        # pair contributes a constant 0.1 loss (and near-zero teacher noise can
        # dominate the useful magnitude objective).  Non-trivial targets keep
        # the directional term; static targets reduce exactly to magnitude
        # matching and therefore do not create a forced-nonzero shortcut.
        direction_strength = teacher_rms.square() / (
            teacher_rms.square() + scale_floor.square()
        )
        return raw + normalized + 0.10 * direction_strength * direction

    result: dict[str, Tensor] = {}
    future_total = zero
    semantic_rows: Tensor | None = None
    component_losses: dict[str, Tensor] = {}
    for (
        name,
        prediction_key,
        target_key,
        internal_weight,
        scale_floored,
    ) in prediction_target_pairs:
        prediction = output[prediction_key]
        teacher = output[target_key]
        if tuple(prediction.shape) != tuple(teacher.shape):
            raise ValueError(
                f"object future {name} prediction/target shapes do not align"
            )
        rows = row_loss(
            prediction,
            teacher,
            scale_floored=scale_floored,
        )
        # Association reliability is already represented in the teacher
        # value: null mass blends successor back to the current fact, motion
        # to zero, visibility to zero and uncertainty upward.  Multiplying by
        # reliability again would erase precisely those conservative fallback
        # targets and recreate an unsupervised W direction.  The physical
        # current-object validity derived from allocated chart support is
        # therefore the sole loss mask for every field.
        weight = validity_weight
        denominator = weight.sum().clamp_min(1.0)
        component = (rows * weight).sum() / denominator
        component_losses[name] = component
        if name not in {"successor", "semantic"}:
            result[f"object_future_{name}"] = component
            future_total = future_total + float(internal_weight) * component
        if name == "semantic":
            semantic_rows = rows
        squared_error = (
            prediction.float() - teacher.detach().float()
        ).square().mean(dim=-1, keepdim=True)
        target_power = teacher.detach().float().square().mean(
            dim=-1, keepdim=True
        )
        for interval_index, interval_name in enumerate(
            ("h4_8", "h8_16", "h16_32", "h32_48")
        ):
            interval_weight = weight[:, interval_index]
            interval_denominator = interval_weight.sum().clamp_min(1.0)
            error_rms = torch.sqrt(
                (
                    squared_error[:, interval_index] * interval_weight
                ).sum()
                / interval_denominator
            )
            target_rms = torch.sqrt(
                (
                    target_power[:, interval_index] * interval_weight
                ).sum()
                / interval_denominator
            ).clamp_min(1e-3)
            result[
                f"object_future_{name}_{interval_name}_normalized_error"
            ] = (error_rms / target_rms).detach()
    if semantic_rows is None:
        raise RuntimeError("object future dynamics lost semantic rows")
    # Stable interval content and ordered end-state change are two views of
    # one future-content objective.  They retain separate diagnostics above,
    # but no longer own duplicate top-level losses.
    content_weight = 0.30 + 0.25
    content_objective = (
        0.30 * component_losses["successor"]
        + 0.25 * component_losses["semantic"]
    ) / content_weight
    result["object_future_content"] = content_objective
    future_total = future_total + content_weight * content_objective
    semantic_prediction = output["object_future_semantic_prediction"].float()
    semantic_target = output["object_future_semantic_target"].detach().float()
    transition_rows = row_loss(
        semantic_prediction[:, 1:] - semantic_prediction[:, :-1],
        semantic_target[:, 1:] - semantic_target[:, :-1],
        scale_floored=True,
    )
    transition_weight = torch.minimum(
        validity_weight[:, 1:], validity_weight[:, :-1]
    )
    transition = (
        transition_rows * transition_weight
    ).sum() / transition_weight.sum().clamp_min(1.0)

    structure_specs = (
        ("object_reconstruction_loss_raw", 0.25),
        ("object_intent_online_match_loss_raw", 0.35),
        ("object_plan_recognition_loss_raw", 0.20),
        ("object_coarse_action_loss_raw", 0.20),
    )
    structure = zero
    for key, weight in structure_specs:
        value = output.get(key)
        if not torch.is_tensor(value) or value.numel() != 1:
            if require_teacher:
                raise RuntimeError(f"object-intent training lost {key}")
            value = zero
        value = value.float().reshape(())
        result[key.removesuffix("_raw")] = value
        structure = structure + float(weight) * value
    # Chronological differentiation and the G/S recognizer scaffold share the
    # pre-existing interval budget.  Neither creates a new external weight.
    interval_objective = 0.50 * transition + 0.50 * structure
    result.update(
        {
            "object_future_dynamics": future_total,
            "object_future_transition": transition,
            "object_intent_structure_core": structure,
            "object_intent_structure": interval_objective,
            "object_future_target_validity": teacher_validity.mean().detach(),
            "object_future_prediction_adjacent_cosine": F.cosine_similarity(
                semantic_prediction[:, 1:].flatten(2),
                semantic_prediction[:, :-1].flatten(2),
                dim=-1,
                eps=1e-4,
            ).mean().detach(),
            "object_future_target_adjacent_cosine": F.cosine_similarity(
                semantic_target[:, 1:].flatten(2),
                semantic_target[:, :-1].flatten(2),
                dim=-1,
                eps=1e-4,
            ).mean().detach(),
            "object_future_prediction_interval_variation": (
                semantic_prediction.std(dim=1, unbiased=False).mean().detach()
            ),
            "object_future_target_interval_variation": (
                semantic_target.std(dim=1, unbiased=False).mean().detach()
            ),
        }
    )
    return result


def flow_losses(
    system: V39PolicySystem,
    sample: dict[str, Tensor],
    output: dict[str, Tensor],
    trainer: V39PolicyTrainerConfig,
    *,
    enable_future_loss: bool = True,
    global_step: int | None = None,
) -> dict[str, Tensor]:
    losses = v363_flow_losses(system, sample, output, trainer)  # type: ignore[arg-type]
    # V116 names the actual action-flow ledger explicitly. These are aliases
    # over the tensors already used by the objective, not new loss terms.
    losses["action_flow_objective"] = losses["physical_flow"]
    losses["native_velocity_mse"] = losses[
        "physical_flow_native_uniform"
    ]
    losses["arm_tangent_mse"] = losses["arm_fm_native"]
    losses["arm_null_mse"] = losses["arm_fm_null"]
    losses["gripper_tangent_mse"] = losses["gripper_fm_native"]
    losses["gripper_null_mse"] = losses["gripper_fm_null"]
    losses["event_reweight_delta"] = (
        losses["physical_flow"]
        - losses["physical_flow_no_information_balance"]
    ).detach()
    balance_horizons = _flow_jepa_balance_horizons(trainer)
    temporal_balance_active = bool(
        balance_horizons
        or str(getattr(trainer, "horizon_weight_mode", "legacy")).strip().lower().replace("-", "_")
        != "legacy"
        or float(getattr(trainer, "trajectory_information_weight", 0.0)) > 0.0
    )
    reference = losses["loss"]
    losses["flow_jepa_horizon_balance_active"] = torch.as_tensor(
        float(balance_horizons),
        device=reference.device,
    )
    reliable_normalization = _flow_jepa_uses_reliable_normalization(trainer)
    losses["flow_jepa_future_reliable_normalization"] = torch.as_tensor(
        float(reliable_normalization),
        device=reference.device,
    )
    losses["temporal_balance_active"] = torch.as_tensor(
        float(temporal_balance_active),
        device=reference.device,
    )
    for key, value in output.items():
        if (
            (
                key.startswith("arm_source_")
                or key.startswith("condition_")
                or key.startswith("evidence_")
                or key.startswith("flow_jepa_")
                or key.startswith("grounded_")
                or key.startswith("object_")
                or key.startswith("role_")
                or key.startswith("attnres_")
            )
            and torch.is_tensor(value)
            and value.numel() == 1
        ):
            losses[key] = value.detach().float().reshape(())
    object_dynamics_active = torch.is_tensor(
        output.get("object_intent_dynamics_active")
    )
    if object_dynamics_active:
        object_terms = object_intent_dynamics_terms(
            output,
            require_teacher=enable_future_loss,
        )
        for key, value in object_terms.items():
            losses[key] = value.detach().float().reshape(())
        future_jepa = object_terms["object_future_dynamics"]
        interval_loss = object_terms["object_intent_structure"]
        losses["flow_jepa_future_prediction"] = future_jepa.detach().float()
        losses["flow_jepa_interval_stage"] = interval_loss.detach().float()
        future_weight = max(
            float(getattr(trainer, "flow_jepa_future_loss_weight", 0.0)),
            0.0,
        )
        interval_weight = max(
            float(
                getattr(
                    trainer,
                    "flow_jepa_interval_stage_loss_weight",
                    0.0,
                )
            ),
            0.0,
        )
        future_contribution = future_weight * future_jepa
        interval_contribution = interval_weight * interval_loss
        losses["loss_contrib_flow_jepa_future"] = (
            future_contribution.detach().float()
        )
        losses["loss_contrib_flow_jepa_interval_stage"] = (
            interval_contribution.detach().float()
        )
        if enable_future_loss:
            losses["loss"] = (
                losses["loss"]
                + future_contribution
                + interval_contribution
            )
        # Historical slot-reduced future/change/address/stage objectives are
        # not owners in this capability.  Keep their canonical ledger rows at
        # exact zero so an inherited nonzero trainer knob cannot silently add
        # a second, averaging-based objective.
        for name in (
            "flow_jepa_future_change_direction",
            "flow_jepa_future_change",
            "flow_jepa_horizon_address",
            "flow_jepa_stage_prediction",
        ):
            losses[name] = future_jepa.detach().new_zeros(())
        losses["flow_jepa_horizon_address_supervision_active"] = (
            future_jepa.detach().new_zeros(())
        )
        losses["loss_contrib_flow_jepa_future_change"] = (
            future_jepa.detach().new_zeros(())
        )
        losses["loss_contrib_flow_jepa_horizon_address"] = (
            future_jepa.detach().new_zeros(())
        )
        losses["loss_contrib_flow_jepa_stage"] = (
            future_jepa.detach().new_zeros(())
        )
    if "flow_jepa_future_pred" in output and not object_dynamics_active:
        grounded_effect_active = torch.is_tensor(
            output.get("grounded_intent_effect_active")
        )
        legacy_future_jepa = flow_jepa_future_prediction_loss(
            output,
            balance_horizons=balance_horizons,
            reliable_normalization=reliable_normalization,
        )
        interval_terms: dict[str, Tensor] | None = None
        future_jepa = legacy_future_jepa
        if grounded_effect_active:
            interval_terms = flow_jepa_interval_stage_terms(output)
            grounded_core = interval_terms.get(
                "grounded_future_effect_core"
            )
            if not torch.is_tensor(grounded_core):
                raise RuntimeError(
                    "grounded future objective lost its object-level "
                    "FutureEffect core"
                )
            future_jepa = grounded_core
            losses["grounded_slot_reduced_future_audit"] = (
                legacy_future_jepa.detach().float().reshape(())
            )
        losses["flow_jepa_future_prediction"] = future_jepa.detach().float()
        losses["flow_jepa_future_change_direction"] = (
            flow_jepa_future_change_direction_loss(
                output,
                balance_horizons=balance_horizons,
            )
            .detach()
            .float()
        )
        future_change = flow_jepa_future_change_loss(
            output,
            balance_horizons=balance_horizons,
        )
        losses["flow_jepa_future_change"] = future_change.detach().float()
        for offset, value in flow_jepa_future_horizon_diagnostics(
            output,
            reliable_normalization=reliable_normalization,
        ).items():
            losses[f"flow_jepa_future_horizon_{offset}"] = value.detach().float().reshape(())
        for key, value in flow_jepa_future_reliable_diagnostics(
            output,
            reliable_normalization=reliable_normalization,
            balance_horizons=balance_horizons,
        ).items():
            losses[key] = value.detach().float().reshape(())
        future_weight = max(float(getattr(trainer, "flow_jepa_future_loss_weight", 0.0)), 0.0)
        future_contribution = future_weight * future_jepa
        losses["loss_contrib_flow_jepa_future"] = future_contribution.detach().float()
        if enable_future_loss and future_weight > 0.0:
            losses["loss"] = losses["loss"] + future_contribution
        future_change_weight = max(
            float(getattr(trainer, "flow_jepa_future_change_loss_weight", 0.0)),
            0.0,
        )
        if future_change_weight > 0.0 and not torch.is_tensor(
            output.get("flow_jepa_current_target")
        ):
            raise RuntimeError(
                "active future-change supervision requires the current JEPA teacher chart"
            )
        active_future_change_weight = (
            0.0 if grounded_effect_active else future_change_weight
        )
        future_change_contribution = (
            active_future_change_weight * future_change
        )
        losses["loss_contrib_flow_jepa_future_change"] = future_change_contribution.detach().float()
        if grounded_effect_active:
            losses["grounded_slot_reduced_future_change_audit"] = (
                future_change.detach().float().reshape(())
            )
            losses[
                "grounded_slot_reduced_future_change_audit_only"
            ] = future_change.new_ones((), dtype=torch.float32)
        if enable_future_loss and active_future_change_weight > 0.0:
            losses["loss"] = losses["loss"] + future_change_contribution
        address_weight = max(
            float(
                getattr(
                    trainer,
                    "flow_jepa_horizon_address_loss_weight",
                    0.0,
                )
            ),
            0.0,
        )
        losses["flow_jepa_horizon_address_supervision_active"] = torch.as_tensor(
            float(address_weight > 0.0),
            device=reference.device,
        )
        if address_weight > 0.0 and not all(
            torch.is_tensor(output.get(name))
            for name in (
                "flow_jepa_horizon_address_logits",
                "flow_jepa_future_target",
                "flow_jepa_current_target",
            )
        ):
            raise RuntimeError(
                "active horizon-address supervision requires forward logits "
                "and current/future frozen teacher charts"
            )
        address_terms = _flow_jepa_horizon_address_terms(output)
        address_loss = address_terms["flow_jepa_horizon_address"]
        losses["flow_jepa_horizon_address"] = address_loss.detach().float()
        for key, value in address_terms.items():
            if key != "flow_jepa_horizon_address":
                losses[key] = value.detach().float().reshape(())
        address_contribution = address_weight * address_loss
        losses["loss_contrib_flow_jepa_horizon_address"] = address_contribution.detach().float()
        if enable_future_loss and address_weight > 0.0:
            losses["loss"] = losses["loss"] + address_contribution
        if interval_terms is None:
            interval_terms = flow_jepa_interval_stage_terms(output)
        interval_loss = interval_terms["flow_jepa_interval_stage"]
        for key, value in interval_terms.items():
            losses[key] = value.detach().float().reshape(())
        interval_weight = max(
            float(
                getattr(
                    trainer,
                    "flow_jepa_interval_stage_loss_weight",
                    0.0,
                )
            ),
            0.0,
        )
        interval_contribution = interval_weight * interval_loss
        losses["loss_contrib_flow_jepa_interval_stage"] = interval_contribution.detach().float()
        if enable_future_loss and interval_weight > 0.0:
            losses["loss"] = losses["loss"] + interval_contribution
        stage_jepa = flow_jepa_stage_prediction_loss(output)
        losses["flow_jepa_stage_prediction"] = stage_jepa.detach().float()
        stage_weight = max(float(getattr(trainer, "flow_jepa_stage_loss_weight", 0.0)), 0.0)
        stage_contribution = stage_weight * stage_jepa
        losses["loss_contrib_flow_jepa_stage"] = stage_contribution.detach().float()
        if enable_future_loss and stage_weight > 0.0:
            losses["loss"] = losses["loss"] + stage_contribution
        auxiliary_terms = (
            ("flow_jepa_warp_loss", "flow_jepa_warp_loss_weight"),
            (
                "flow_jepa_identity_advantage_loss",
                "flow_jepa_identity_advantage_loss_weight",
            ),
            (
                "flow_jepa_static_identity_loss",
                "flow_jepa_static_identity_loss_weight",
            ),
            ("flow_jepa_cycle_loss", "flow_jepa_cycle_loss_weight"),
            ("flow_jepa_smoothness_loss", "flow_jepa_smoothness_loss_weight"),
            ("flow_jepa_uncertainty_nll", "flow_jepa_uncertainty_nll_weight"),
            (
                "flow_jepa_refinement_sequence_loss",
                "flow_jepa_refinement_sequence_loss_weight",
            ),
        )
        for loss_name, weight_name in auxiliary_terms:
            term = output.get(loss_name)
            weight = max(float(getattr(trainer, weight_name, 0.0)), 0.0)
            if not torch.is_tensor(term) or term.numel() != 1:
                if (
                    loss_name
                    in {
                        "flow_jepa_identity_advantage_loss",
                        "flow_jepa_static_identity_loss",
                    }
                    and weight <= 0.0
                ):
                    continue
                raise RuntimeError(f"enabled Flow-DINO path did not expose {loss_name}")
            contribution = weight * term
            losses[f"loss_contrib_{loss_name}"] = contribution.detach().float().reshape(())
            if weight > 0.0:
                losses["loss"] = losses["loss"] + contribution
    elif object_dynamics_active:
        # Flow geometry remains a real observation-side objective.  Only the
        # obsolete slot-reduced future/address owners above are disabled.
        auxiliary_terms = (
            ("flow_jepa_warp_loss", "flow_jepa_warp_loss_weight"),
            (
                "flow_jepa_identity_advantage_loss",
                "flow_jepa_identity_advantage_loss_weight",
            ),
            (
                "flow_jepa_static_identity_loss",
                "flow_jepa_static_identity_loss_weight",
            ),
            ("flow_jepa_cycle_loss", "flow_jepa_cycle_loss_weight"),
            ("flow_jepa_smoothness_loss", "flow_jepa_smoothness_loss_weight"),
            ("flow_jepa_uncertainty_nll", "flow_jepa_uncertainty_nll_weight"),
            (
                "flow_jepa_refinement_sequence_loss",
                "flow_jepa_refinement_sequence_loss_weight",
            ),
        )
        for loss_name, weight_name in auxiliary_terms:
            term = output.get(loss_name)
            weight = max(float(getattr(trainer, weight_name, 0.0)), 0.0)
            if not torch.is_tensor(term) or term.numel() != 1:
                if (
                    loss_name
                    in {
                        "flow_jepa_identity_advantage_loss",
                        "flow_jepa_static_identity_loss",
                    }
                    and weight <= 0.0
                ):
                    continue
                raise RuntimeError(
                    f"object-intent Flow-DINO path did not expose {loss_name}"
                )
            contribution = weight * term
            losses[f"loss_contrib_{loss_name}"] = (
                contribution.detach().float().reshape(())
            )
            if weight > 0.0:
                losses["loss"] = losses["loss"] + contribution
    execution_cost = output.get("evidence_mmd_it_execution_cost")
    if torch.is_tensor(execution_cost) and execution_cost.numel() == 1:
        # The native controller exposes this as an audit-only statistic.  It
        # is intentionally detached and must not inherit the hierarchical
        # decoder's depth-usage penalty: adding a detached value would change
        # the reported loss without giving any parameter a useful gradient.
        execution_cost_weight = 0.0
        losses["evidence_mmd_it_execution_cost"] = execution_cost.detach().float().reshape(())
        losses["evidence_mmd_it_execution_cost_weight"] = torch.as_tensor(
            execution_cost_weight,
            device=execution_cost.device,
            dtype=torch.float32,
        )
        if enable_future_loss and execution_cost_weight > 0.0:
            losses["loss"] = losses["loss"] + execution_cost_weight * execution_cost
    execution_value_field = output.get("evidence_mmd_it_execution_candidate_value_field")
    execution_candidates = output.get("evidence_mmd_it_dwell_candidate_pred_velocity")
    execution_candidate_mask = output.get("evidence_mmd_it_execution_candidate_value_mask")
    execution_baseline = output.get("evidence_mmd_it_execution_baseline_pred_velocity")
    execution_target = output.get("target_physical_velocity")
    execution_value_weight = max(
        float(getattr(trainer, "latent_cvae_mmdit_execution_value_loss_weight", 0.0)),
        0.0,
    )
    execution_dwell_mode = str(getattr(trainer, "latent_cvae_mmdit_dwell_mode", "fixed"))
    if (
        execution_value_weight > 0.0
        and execution_dwell_mode in {"learned", "learned_shadow"}
        and all(
            torch.is_tensor(value)
            for value in (
                execution_value_field,
                execution_candidates,
                execution_candidate_mask,
                execution_baseline,
                execution_target,
            )
        )
    ):
        # Candidate probes contain the global operation chart plus one terminal
        # identity.  The latter is the actionable baseline, so centered values
        # now learn whether doing more work is physically better than stopping.
        # The final value-field axis is typed (arm, gripper), not categorical.
        if (
            execution_value_field.ndim != 5
            or execution_candidates.ndim != 5
            or execution_candidate_mask.ndim != 3
            or execution_baseline.ndim != 4
            or execution_target.ndim != 3
            or int(execution_value_field.shape[-1]) != 2
        ):
            raise ValueError("native execution candidate value probes have invalid shapes")
        if tuple(execution_value_field.shape[:4]) != tuple(execution_candidates.shape[:4]):
            raise ValueError("native execution value field and candidates are misaligned")
        if tuple(execution_candidate_mask.shape) != tuple(execution_candidates.shape[:3]):
            raise ValueError("native execution dwell candidate mask is misaligned")
        expected_target_shape = (
            int(execution_candidates.shape[0]),
            int(execution_candidates.shape[3]),
            int(execution_candidates.shape[4]),
        )
        if tuple(execution_target.shape) != expected_target_shape:
            raise ValueError("native execution target has the wrong physical shape")
        expected_baseline_shape = (
            int(execution_candidates.shape[0]),
            int(execution_candidates.shape[1]),
            int(execution_candidates.shape[3]),
            int(execution_candidates.shape[4]),
        )
        if tuple(execution_baseline.shape) != expected_baseline_shape:
            raise ValueError("native execution baseline has the wrong physical shape")
        with torch.no_grad():
            terminal_identity_error = (
                (execution_candidates[:, :, -1].float() - execution_baseline.detach().float())
                .square()
                .mean()
                .sqrt()
            )
            candidate_residual = (
                execution_candidates.float() - execution_target.detach().float()[:, None, None]
            )
            candidate_value = _operation_candidate_error_field(
                system,
                candidate_residual,
                sample,
                trainer,
            ).detach()
            target_value = candidate_value
            valid = execution_candidate_mask.detach().bool()
            target_centered, _ = masked_candidate_center(target_value, valid, candidate_dim=2)
        predicted_value = execution_value_field.float()
        predicted_centered, predicted_mean = masked_candidate_center(
            predicted_value, valid, candidate_dim=2
        )
        valid_field = valid[..., None, None].expand_as(predicted_value)
        valid_field_float = valid_field.float()
        arm_dim = int(system.policy_config.arm_dim)
        component_weight = torch.tensor(
            [float(arm_dim), 1.0],
            device=predicted_value.device,
            dtype=predicted_value.dtype,
        ) / float(arm_dim + 1)
        physical_field_weight = valid_field_float * component_weight[None, None, None, None]
        candidate_count = valid.float().sum(dim=2)
        active_candidate = candidate_count > 1.0
        row_denominator = (
            valid[..., None]
            .expand(-1, -1, -1, int(predicted_value.shape[-2]))
            .float()
            .sum(dim=(2, 3))
            .clamp_min(1.0)
        )
        target_spread = torch.sqrt(
            (target_centered.square() * physical_field_weight).sum(dim=(2, 3, 4)) / row_denominator
        )
        configured_scale = max(
            float(getattr(trainer, "hierarchical_mmdit_operation_value_reliability_scale", 0.0)),
            0.0,
        )
        active_float = active_candidate.float()
        active_denominator = active_float.sum().clamp_min(1.0)
        if configured_scale > 0.0:
            reliability_scale = torch.as_tensor(
                configured_scale,
                device=target_spread.device,
                dtype=target_spread.dtype,
            )
        else:
            reliability_scale = (
                (target_spread.detach() * active_float).sum() / active_denominator
            ).clamp_min(1e-6)
        reliability = target_spread / (target_spread + reliability_scale)
        reliability = reliability * active_float
        reliability_denominator = reliability.sum().clamp_min(1e-6)
        # The reader emits a dimensionless candidate advantage.  Normalize the
        # physical target per decision so temperature=1 has a stable meaning
        # across batches and throughout training. Unreliable near-ties remain
        # down-weighted by the physical spread above.
        normalization_scale = torch.maximum(target_spread.detach(), reliability_scale.detach())
        normalized_target = target_centered / normalization_scale[..., None, None, None]
        huber_delta = max(
            float(getattr(trainer, "hierarchical_mmdit_operation_value_huber_delta", 0.1)),
            1e-6,
        )
        value_loss_field = (
            F.smooth_l1_loss(
                predicted_centered,
                normalized_target,
                reduction="none",
                beta=huber_delta,
            )
            * physical_field_weight
        )
        value_loss_rows = value_loss_field.sum(dim=(2, 3, 4)) / row_denominator
        execution_value_loss = (value_loss_rows * reliability).sum() / reliability_denominator
        predicted_scalar = (
            (predicted_value * component_weight[None, None, None, None]).sum(dim=-1).mean(dim=-1)
        )
        target_scalar = (
            (normalized_target * component_weight[None, None, None, None]).sum(dim=-1).mean(dim=-1)
        )
        invalid_max = torch.finfo(predicted_scalar.dtype).max
        predicted_best = predicted_scalar.masked_fill(~valid, invalid_max).argmin(dim=-1)
        target_best = target_scalar.masked_fill(~valid, invalid_max).argmin(dim=-1)
        decision_accuracy = (
            (predicted_best == target_best).float() * active_float
        ).sum() / active_denominator
        target_difference = target_scalar[..., :, None] - target_scalar[..., None, :]
        predicted_difference = predicted_scalar[..., :, None] - predicted_scalar[..., None, :]
        pair_mask = valid[..., :, None] & valid[..., None, :]
        pair_mask = pair_mask & torch.triu(
            torch.ones(
                int(valid.shape[-1]),
                int(valid.shape[-1]),
                device=valid.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        informative_pair = pair_mask & target_difference.ne(0.0)
        pair_denominator = informative_pair.float().sum().clamp_min(1.0)
        pairwise_accuracy = (
            (predicted_difference * target_difference > 0.0).float() * informative_pair.float()
        ).sum() / pair_denominator
        correlation = (predicted_centered * normalized_target * physical_field_weight).sum() / (
            (predicted_centered.square() * physical_field_weight).sum().sqrt()
            * (normalized_target.square() * physical_field_weight).sum().sqrt()
        ).clamp_min(1e-8)
        predicted_rms = (
            (predicted_value.square() * physical_field_weight).sum()
            / row_denominator.sum().clamp_min(1.0)
        ).sqrt()
        active_common = active_float[..., None, None, None]
        predicted_common_rms = (
            (
                predicted_mean.square() * active_common * component_weight[None, None, None, None]
            ).sum()
            / (active_common.sum() * int(predicted_mean.shape[-2])).clamp_min(1.0)
        ).sqrt()
        common_mode_ratio = predicted_common_rms / predicted_rms.clamp_min(1e-8)
        selected_spread = target_spread[active_candidate]
        if int(selected_spread.numel()) > 0:
            spread_p25, spread_p50, spread_p75 = (
                torch.quantile(selected_spread, q) for q in (0.25, 0.50, 0.75)
            )
        else:
            spread_p25 = spread_p50 = spread_p75 = torch.zeros(
                (), device=target_spread.device, dtype=torch.float32
            )
        losses["evidence_mmd_it_execution_value_loss"] = execution_value_loss.detach().float()
        losses["evidence_mmd_it_execution_value_weight"] = torch.as_tensor(
            execution_value_weight,
            device=execution_value_loss.device,
            dtype=torch.float32,
        )
        losses["evidence_mmd_it_execution_value_reliability_scale"] = (
            reliability_scale.detach().float()
        )
        losses["evidence_mmd_it_execution_value_reliability"] = (
            (reliability.sum() / active_denominator).detach().float()
        )
        losses["evidence_mmd_it_execution_value_target_spread"] = (
            ((target_spread * active_float).sum() / active_denominator).detach().float()
        )
        predicted_standardized_spread = torch.sqrt(
            (predicted_centered.square() * physical_field_weight).sum(dim=(2, 3, 4))
            / row_denominator
        )
        predicted_spread = predicted_standardized_spread * normalization_scale
        losses["evidence_mmd_it_execution_value_predicted_spread"] = (
            ((predicted_spread * active_float).sum() / active_denominator).detach().float()
        )
        losses["evidence_mmd_it_execution_value_predicted_standardized_spread"] = (
            ((predicted_standardized_spread * active_float).sum() / active_denominator)
            .detach()
            .float()
        )
        losses["evidence_mmd_it_execution_value_target_spread_p25"] = spread_p25.detach().float()
        losses["evidence_mmd_it_execution_value_target_spread_p50"] = spread_p50.detach().float()
        losses["evidence_mmd_it_execution_value_target_spread_p75"] = spread_p75.detach().float()
        losses["evidence_mmd_it_execution_value_correlation"] = correlation.detach().float()
        losses["evidence_mmd_it_execution_value_pairwise_accuracy"] = (
            pairwise_accuracy.detach().float()
        )
        losses["evidence_mmd_it_execution_value_decision_accuracy"] = (
            decision_accuracy.detach().float()
        )
        losses["evidence_mmd_it_execution_value_common_mode_ratio"] = (
            common_mode_ratio.detach().float()
        )
        losses["evidence_mmd_it_execution_candidate_coverage"] = valid.float().mean()
        losses["evidence_mmd_it_terminal_identity_velocity_error"] = (
            terminal_identity_error.detach().float()
        )
        terminal_valid = valid[..., -1] & active_candidate
        operation_scalar = target_scalar[..., :-1].masked_fill(~valid[..., :-1], invalid_max)
        best_operation = operation_scalar.amin(dim=-1)
        terminal_target_margin = target_scalar[..., -1] - best_operation
        predicted_operation = predicted_scalar[..., :-1].masked_fill(~valid[..., :-1], invalid_max)
        predicted_terminal_margin = predicted_scalar[..., -1] - predicted_operation.amin(dim=-1)
        terminal_denominator = terminal_valid.float().sum().clamp_min(1.0)
        losses["evidence_mmd_it_terminal_target_cost_margin"] = (
            ((terminal_target_margin * terminal_valid.float()).sum() / terminal_denominator)
            .detach()
            .float()
        )
        losses["evidence_mmd_it_terminal_predicted_cost_margin"] = (
            ((predicted_terminal_margin * terminal_valid.float()).sum() / terminal_denominator)
            .detach()
            .float()
        )
        losses["evidence_mmd_it_terminal_target_preferred_fraction"] = (
            (((terminal_target_margin < 0.0) & terminal_valid).float().sum() / terminal_denominator)
            .detach()
            .float()
        )
        if enable_future_loss:
            losses["loss"] = losses["loss"] + execution_value_weight * execution_value_loss
    dyn = rollout_dynamics_loss(output)
    delta = rollout_delta_loss(output)
    con = rollout_contrast_loss(output, margin=float(trainer.rollout_contrast_margin))
    grid = _future_grid_count(system, output)
    rollout_var = latent_variance_loss(output, grid=grid)
    rollout_norm = latent_norm_loss(output)
    rollout_milestone = milestone_delta_match_loss(output, grid=grid)
    losses["rollout_dynamics"] = dyn
    losses["rollout_delta"] = delta
    losses["rollout_contrast"] = con
    losses["rollout_variance"] = rollout_var
    losses["rollout_norm"] = rollout_norm
    losses["rollout_milestone_delta_match"] = rollout_milestone
    spectral_competition = output.get("hierarchical_mmdit_spectral_competition_loss")
    if torch.is_tensor(spectral_competition) and spectral_competition.numel() == 1:
        spectral_weight = max(
            float(
                getattr(
                    system.policy_config,
                    "hierarchical_mmdit_spectral_competition_loss_weight",
                    0.0,
                )
            ),
            0.0,
        )
        warmup_steps = max(
            int(
                getattr(
                    system.policy_config,
                    "hierarchical_mmdit_spectral_competition_warmup_steps",
                    0,
                )
            ),
            0,
        )
        step_value = 0 if global_step is None else max(int(global_step), 0)
        losses["hierarchical_mmdit_spectral_competition_loss"] = (
            spectral_competition.detach().float().reshape(())
        )
        losses["hierarchical_mmdit_spectral_competition_weight"] = torch.as_tensor(
            spectral_weight if step_value >= warmup_steps else 0.0,
            device=spectral_competition.device,
            dtype=spectral_competition.dtype,
        )
        if enable_future_loss and spectral_weight > 0.0 and step_value >= warmup_steps:
            losses["loss"] = losses["loss"] + spectral_weight * spectral_competition
    predicted_coefficients = output.get("pred_velocity_coefficients")
    target_flow_velocity = output.get("target_flow_velocity")
    if (
        torch.is_tensor(predicted_coefficients)
        and predicted_coefficients.ndim == 3
        and torch.is_tensor(target_flow_velocity)
    ):
        losses["hierarchical_mmdit_spectral_coefficient_flow_mse"] = (
            predicted_coefficients.detach()
            .float()
            .sub(target_flow_velocity.detach().float())
            .square()
            .mean()
        )
        losses["hierarchical_mmdit_spectral_coefficient_flow_rms"] = (
            predicted_coefficients.detach()
            .float()
            .sub(target_flow_velocity.detach().float())
            .square()
            .mean()
            .sqrt()
        )
        losses["hierarchical_mmdit_spectral_coefficient_flow_energy"] = (
            target_flow_velocity.detach().float().square().mean()
        )
        losses["hierarchical_mmdit_spectral_coefficient_prediction_energy"] = (
            predicted_coefficients.detach().float().square().mean()
        )
    # Compatibility log names: these no longer correspond to self-denoise.
    losses["future_latent"] = dyn.detach()
    losses["action_effect"] = delta.detach()
    if enable_future_loss and float(trainer.rollout_dynamics_loss_weight) > 0:
        losses["loss"] = losses["loss"] + float(trainer.rollout_dynamics_loss_weight) * dyn
    if enable_future_loss and float(trainer.rollout_delta_loss_weight) > 0:
        losses["loss"] = losses["loss"] + float(trainer.rollout_delta_loss_weight) * delta
    if enable_future_loss and float(trainer.rollout_contrast_loss_weight) > 0:
        losses["loss"] = losses["loss"] + float(trainer.rollout_contrast_loss_weight) * con
    if enable_future_loss and float(trainer.rollout_variance_loss_weight) > 0:
        losses["loss"] = losses["loss"] + float(trainer.rollout_variance_loss_weight) * rollout_var
    if enable_future_loss and float(trainer.rollout_norm_loss_weight) > 0:
        losses["loss"] = losses["loss"] + float(trainer.rollout_norm_loss_weight) * rollout_norm
    if enable_future_loss and float(trainer.rollout_milestone_delta_match_weight) > 0:
        losses["loss"] = (
            losses["loss"] + float(trainer.rollout_milestone_delta_match_weight) * rollout_milestone
        )
    # Disabled by default. Kept only as compatibility knobs; they map to the
    # same dynamics-bound target rather than future-noisy denoise.
    if enable_future_loss and float(trainer.future_latent_loss_weight) > 0:
        losses["loss"] = losses["loss"] + float(trainer.future_latent_loss_weight) * dyn
    if enable_future_loss and float(trainer.action_effect_loss_weight) > 0:
        losses["loss"] = losses["loss"] + float(trainer.action_effect_loss_weight) * dyn
    if "latent_cvae_kl" in output:
        kl = output["latent_cvae_kl"]
        weight = float(getattr(trainer, "latent_cvae_kl_weight", 0.0))
        losses["latent_cvae_kl"] = kl.detach().float()
        losses["latent_cvae_kl_weight"] = torch.as_tensor(weight, device=kl.device, dtype=kl.dtype)
        if weight > 0:
            losses["loss"] = losses["loss"] + weight * kl
    if "latent_cvae_kl" in output and "legacy_physical_velocity" in output:
        pred = output["pred_physical_velocity"]
        legacy = (
            output["legacy_physical_velocity"].detach().to(device=pred.device, dtype=pred.dtype)
        )
        anchor_error = semantic_physical_velocity_error(
            system,
            pred - legacy,
            arm_null_weight=trainer.arm_manifold_null_weight,
        )
        pos_w = position_weights(system.policy_config, trainer, pred.device).to(dtype=pred.dtype)
        anchor = (anchor_error * pos_w[None]).mean()
        base_weight = max(float(getattr(trainer, "latent_cvae_legacy_anchor_weight", 0.0)), 0.0)
        min_weight = min(
            max(float(getattr(trainer, "latent_cvae_legacy_anchor_min_weight", 0.0)), 0.0),
            base_weight,
        )
        decay_steps = int(getattr(trainer, "latent_cvae_legacy_anchor_decay_steps", 0))
        step_value = 0 if global_step is None else max(int(global_step), 0)
        if decay_steps > 0:
            remain = max(1.0 - float(step_value) / float(decay_steps), 0.0)
            weight_value = min_weight + (base_weight - min_weight) * remain * remain
        else:
            weight_value = base_weight
        anchor_weight = torch.as_tensor(weight_value, device=pred.device, dtype=pred.dtype)
        losses["latent_cvae_legacy_anchor"] = anchor.detach().float()
        losses["latent_cvae_legacy_anchor_weight"] = anchor_weight.detach().float()
        flat_pred = pred.detach().float().flatten(1)
        flat_legacy = legacy.detach().float().flatten(1)
        losses["latent_cvae_legacy_cosine"] = F.cosine_similarity(
            flat_pred, flat_legacy, dim=-1
        ).mean()
        losses["latent_cvae_legacy_norm_ratio"] = (
            flat_pred.norm(dim=-1) / flat_legacy.norm(dim=-1).clamp_min(1e-6)
        ).mean()
        if weight_value > 0:
            losses["loss"] = losses["loss"] + anchor_weight * anchor

    # V42.1: keep the prior/deploy path as the main output.  The posterior
    # path is a weak auxiliary reconstruction target only; this prevents the
    # target-conditioned z from becoming the only path that learns action.
    if "post_pred_velocity" in output:
        post_pred = output["post_pred_velocity"]
        post_error = semantic_physical_velocity_error(
            system,
            post_pred - output["target_physical_velocity"],
            arm_null_weight=trainer.arm_manifold_null_weight,
        )
        pos_w = position_weights(system.policy_config, trainer, post_pred.device).to(
            dtype=post_pred.dtype
        )
        post_flow = (post_error * pos_w[None]).mean()
        losses["latent_cvae_post_flow"] = post_flow.detach().float()
        post_recon = post_flow
        if "post_pred_action_estimate" in output:
            post_decoded = F.smooth_l1_loss(
                output["post_pred_action_estimate"],
                sample["policy_action"].to(device=post_pred.device),
            )
            losses["latent_cvae_post_decoded_action"] = post_decoded.detach().float()
            post_recon = post_recon + float(trainer.decoded_action_loss_weight) * post_decoded
        post_weight = float(getattr(trainer, "latent_cvae_posterior_recon_weight", 0.0))
        losses["latent_cvae_posterior_recon"] = post_recon.detach().float()
        losses["latent_cvae_posterior_recon_weight"] = torch.as_tensor(
            post_weight, device=post_pred.device, dtype=post_pred.dtype
        )
        if post_weight > 0:
            losses["loss"] = losses["loss"] + post_weight * post_recon
    if "latent_cvae_adaptive_regularizer" in output:
        reg = output["latent_cvae_adaptive_regularizer"]
        weight = float(getattr(trainer, "latent_cvae_adaptive_regularizer_weight", 0.0))
        losses["latent_cvae_adaptive_regularizer"] = reg.detach().float()
        losses["latent_cvae_adaptive_regularizer_weight"] = torch.as_tensor(
            weight, device=reg.device, dtype=reg.dtype
        )
        if weight > 0:
            losses["loss"] = losses["loss"] + weight * reg
    if "latent_cvae_adaptive_route_entropy_regularizer" in output:
        route_reg = output["latent_cvae_adaptive_route_entropy_regularizer"]
        route_weight = float(getattr(trainer, "latent_cvae_adaptive_route_entropy_weight", 0.0))
        losses["latent_cvae_adaptive_route_entropy_regularizer"] = route_reg.detach().float()
        losses["latent_cvae_adaptive_route_entropy_weight"] = torch.as_tensor(
            route_weight, device=route_reg.device, dtype=route_reg.dtype
        )
        if route_weight > 0:
            losses["loss"] = losses["loss"] + route_weight * route_reg
    if "hierarchical_mmdit_depth_usage_regularizer" in output:
        depth_reg = output["hierarchical_mmdit_depth_usage_regularizer"]
        depth_weight = float(getattr(trainer, "hierarchical_mmdit_depth_usage_loss_weight", 0.0))
        losses["hierarchical_mmdit_depth_usage_regularizer"] = depth_reg.detach().float()
        losses["hierarchical_mmdit_depth_usage_loss_weight"] = torch.as_tensor(
            depth_weight, device=depth_reg.device, dtype=depth_reg.dtype
        )
        if depth_weight > 0.0:
            losses["loss"] = losses["loss"] + depth_weight * depth_reg
    exit_logits = output.get("hierarchical_mmdit_exit_logits")
    probe_predictions = output.get("refinement_probe_pred_velocity")
    probe_active = output.get("refinement_probe_active")
    exit_candidates = output.get("refinement_probe_exit_candidates")
    target_velocity = output.get("target_physical_velocity")
    base_route_weight = max(float(trainer.hierarchical_mmdit_oracle_route_loss_weight), 0.0)
    if base_route_weight > 0.0 and int(
        getattr(system.policy_config, "hierarchical_mmdit_unified_controller", 0)
    ):
        raise ValueError(
            "legacy oracle exit supervision is incompatible with the unified controller; "
            "use hierarchical_mmdit_operation_value_loss_weight"
        )
    if (
        base_route_weight > 0.0
        and str(system.policy_config.hierarchical_mmdit_schedule_mode) != "fixed"
    ):
        raise ValueError("oracle route supervision requires hierarchical_mmdit_schedule_mode=fixed")
    route_inputs = {
        "exit_logits": exit_logits,
        "probe_predictions": probe_predictions,
        "probe_active": probe_active,
        "exit_candidates": exit_candidates,
        "target_velocity": target_velocity,
    }
    if base_route_weight > 0.0:
        missing_route_inputs = [
            name for name, value in route_inputs.items() if not torch.is_tensor(value)
        ]
        if missing_route_inputs:
            raise RuntimeError(
                "oracle route supervision is enabled but decoder probes are missing: "
                + ", ".join(missing_route_inputs)
            )
        with torch.no_grad():
            route_residual = probe_predictions.float() - target_velocity.detach().float()[:, None]
            route_horizon_error = semantic_physical_velocity_error(
                system,
                route_residual,
                arm_null_weight=trainer.arm_manifold_null_weight,
            )
            route_position_weight = position_weights(
                system.policy_config, trainer, route_horizon_error.device
            ).to(dtype=route_horizon_error.dtype)
            route_step_error = (route_horizon_error * route_position_weight[None, None]).mean(
                dim=-1
            )
        route = _oracle_exit_supervision(
            exit_logits=exit_logits,
            candidate_error=route_step_error[:, 1:],
            initial_error=route_step_error[:, 0],
            candidate_mask=probe_active.bool() & exit_candidates.bool(),
            relative_tolerance=float(trainer.hierarchical_mmdit_oracle_route_relative_tolerance),
        )
        route_warmup = max(int(trainer.hierarchical_mmdit_oracle_route_warmup_steps), 0)
        step_value = 0 if global_step is None else max(int(global_step), 0)
        route_weight = base_route_weight if system.training and step_value >= route_warmup else 0.0
        losses["hierarchical_mmdit_oracle_route_loss"] = route["loss"].detach().float()
        losses["hierarchical_mmdit_oracle_route_weight"] = torch.as_tensor(
            route_weight, device=exit_logits.device, dtype=torch.float32
        )
        losses["hierarchical_mmdit_oracle_position_weighted"] = torch.ones(
            (), device=exit_logits.device, dtype=torch.float32
        )
        for name, value in route.items():
            if name != "loss":
                losses[f"hierarchical_mmdit_oracle_{name}"] = value.detach().float()
        if route_weight > 0.0:
            losses["loss"] = losses["loss"] + route_weight * route["loss"]

    refinement_step_error_cache: Tensor | None = None
    if torch.is_tensor(probe_predictions) and torch.is_tensor(target_velocity):
        with torch.no_grad():
            prefix_residual = probe_predictions.float() - target_velocity.detach().float()[:, None]
            prefix_horizon_error = semantic_physical_velocity_error(
                system,
                prefix_residual,
                arm_null_weight=trainer.arm_manifold_null_weight,
            )
            prefix_position_weight = position_weights(
                system.policy_config, trainer, prefix_horizon_error.device
            ).to(dtype=prefix_horizon_error.dtype)
            prefix_error = (prefix_horizon_error * prefix_position_weight[None, None]).mean(dim=-1)
            refinement_step_error_cache = prefix_error.detach()
            prefix_gain = prefix_error[:, :-1] - prefix_error[:, 1:]
            prefix_active = (
                probe_active.float()
                if torch.is_tensor(probe_active)
                else torch.ones_like(prefix_gain)
            )
            prefix_denominator = prefix_active.sum().clamp_min(1.0)
            losses["hierarchical_mmdit_prefix_error_initial"] = prefix_error[:, 0].mean()
            losses["hierarchical_mmdit_prefix_error_final"] = prefix_error[:, -1].mean()
            losses["hierarchical_mmdit_prefix_gain_mean"] = (
                prefix_gain * prefix_active
            ).sum() / prefix_denominator
            losses["hierarchical_mmdit_prefix_gain_positive_fraction"] = (
                (prefix_gain > 0.0).float() * prefix_active
            ).sum() / prefix_denominator

    operation_value_field = output.get("hierarchical_mmdit_operation_value_field")
    operation_candidates = output.get("hierarchical_mmdit_operation_candidate_predictions")
    operation_mask = output.get("hierarchical_mmdit_operation_candidate_mask")
    retired_route_weight = max(
        float(getattr(trainer, "hierarchical_mmdit_operation_route_loss_weight", 0.0)),
        0.0,
    )
    if retired_route_weight > 0.0:
        raise ValueError(
            "categorical operation-route supervision was retired in V88; "
            "use hierarchical_mmdit_operation_value_loss_weight"
        )
    operation_weight_base = max(
        float(
            getattr(
                trainer,
                "hierarchical_mmdit_operation_value_loss_weight",
                0.0,
            )
        ),
        0.0,
    )
    operation_inputs = {
        "operation_value_field": operation_value_field,
        "operation_candidates": operation_candidates,
        "operation_mask": operation_mask,
        "target_velocity": target_velocity,
    }
    if operation_weight_base > 0.0:
        if not int(
            getattr(
                system.policy_config,
                "hierarchical_mmdit_operation_candidate_probes",
                0,
            )
        ):
            raise ValueError(
                "operation value supervision requires "
                "hierarchical_mmdit_operation_candidate_probes=1"
            )
        missing = [name for name, value in operation_inputs.items() if not torch.is_tensor(value)]
        if missing:
            raise RuntimeError(
                "operation value supervision is missing probes: " + ", ".join(missing)
            )
        if operation_candidates.ndim != 5 or operation_value_field.ndim != 5:
            raise ValueError("operation value tensors must be [B,S,C,T,P] and [B,S,C,T,2]")
        expected_value_shape = (
            int(operation_candidates.shape[0]),
            int(operation_candidates.shape[1]),
            int(operation_candidates.shape[2]) - 1,
            int(operation_candidates.shape[3]),
            2,
        )
        if tuple(operation_value_field.shape) != expected_value_shape:
            raise ValueError(
                "operation value field shape does not match candidate probes: "
                f"expected {expected_value_shape}, got {tuple(operation_value_field.shape)}"
            )
        if tuple(operation_mask.shape) != tuple(operation_candidates.shape[:3]):
            raise ValueError("operation candidate mask has the wrong shape")
        with torch.no_grad():
            candidate_residual = (
                operation_candidates.float() - target_velocity.detach().float()[:, None, None]
            )
            candidate_error_field = _operation_candidate_error_field(
                system, candidate_residual, sample, trainer
            )
            candidate_position_weight = position_weights(
                system.policy_config, trainer, candidate_error_field.device
            ).to(dtype=candidate_error_field.dtype)
            candidate_error_field = (
                candidate_error_field * candidate_position_weight[None, None, None, :, None]
            )
            baseline_error = candidate_error_field[..., :1, :, :]
            target_value = (candidate_error_field[..., 1:, :, :] - baseline_error).detach()
            valid = operation_mask[..., 1:].detach().bool()
            target_centered, target_mean = masked_candidate_center(
                target_value, valid, candidate_dim=2
            )

        predicted_value = operation_value_field.float()
        predicted_centered, predicted_mean = masked_candidate_center(
            predicted_value, valid, candidate_dim=2
        )
        valid_field = valid[..., None, None].expand_as(predicted_value)
        valid_field_float = valid_field.float()
        arm_dim = int(system.policy_config.arm_dim)
        component_weight = torch.tensor(
            [float(arm_dim), 1.0],
            device=predicted_value.device,
            dtype=predicted_value.dtype,
        ) / float(arm_dim + 1)
        physical_field_weight = valid_field_float * component_weight[None, None, None, None]
        candidate_count = valid.float().sum(dim=2)
        active_candidate = candidate_count > 1.0
        row_denominator = (
            valid[..., None]
            .expand(-1, -1, -1, int(predicted_value.shape[-2]))
            .float()
            .sum(dim=(2, 3))
            .clamp_min(1.0)
        )
        target_spread = torch.sqrt(
            (target_centered.square() * physical_field_weight).sum(dim=(2, 3, 4)) / row_denominator
        )
        configured_scale = max(
            float(
                getattr(
                    trainer,
                    "hierarchical_mmdit_operation_value_reliability_scale",
                    0.0,
                )
            ),
            0.0,
        )
        active_float = active_candidate.float()
        active_denominator = active_float.sum().clamp_min(1.0)
        if configured_scale > 0.0:
            reliability_scale = torch.as_tensor(
                configured_scale,
                device=target_spread.device,
                dtype=target_spread.dtype,
            )
        else:
            reliability_scale = (
                (target_spread.detach() * active_float).sum() / active_denominator
            ).clamp_min(1e-6)
        reliability = target_spread / (target_spread + reliability_scale)
        reliability = reliability * active_float
        reliability_denominator = reliability.sum().clamp_min(1e-6)
        huber_delta = max(
            float(
                getattr(
                    trainer,
                    "hierarchical_mmdit_operation_value_huber_delta",
                    0.1,
                )
            ),
            1e-6,
        )
        value_loss_field = (
            F.smooth_l1_loss(
                predicted_centered,
                target_centered,
                reduction="none",
                beta=huber_delta,
            )
            * physical_field_weight
        )
        value_loss_rows = value_loss_field.sum(dim=(2, 3, 4)) / row_denominator
        value_loss = (value_loss_rows * reliability).sum() / reliability_denominator
        operation_weight = operation_weight_base if system.training else 0.0
        predicted_scalar = (
            (predicted_value * component_weight[None, None, None, None]).sum(dim=-1).mean(dim=-1)
        )
        target_scalar = (
            (target_value * component_weight[None, None, None, None]).sum(dim=-1).mean(dim=-1)
        )
        invalid_max = torch.finfo(predicted_scalar.dtype).max
        predicted_best = predicted_scalar.masked_fill(~valid, invalid_max).argmin(dim=-1)
        target_best = target_scalar.masked_fill(~valid, invalid_max).argmin(dim=-1)
        decision_accuracy = (
            (predicted_best == target_best).float() * active_float
        ).sum() / active_denominator

        pair_mask = valid[..., :, None] & valid[..., None, :]
        upper = torch.triu(
            torch.ones(
                int(valid.shape[-1]),
                int(valid.shape[-1]),
                device=valid.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        pair_mask = pair_mask & upper
        target_difference = target_scalar[..., :, None] - target_scalar[..., None, :]
        predicted_difference = predicted_scalar[..., :, None] - predicted_scalar[..., None, :]
        informative_pair = pair_mask & target_difference.ne(0.0)
        pair_denominator = informative_pair.float().sum().clamp_min(1.0)
        pairwise_accuracy = (
            (predicted_difference * target_difference > 0.0).float() * informative_pair.float()
        ).sum() / pair_denominator

        predicted_flat = predicted_centered
        target_flat = target_centered
        correlation = (predicted_flat * target_flat * physical_field_weight).sum() / (
            (predicted_flat.square() * physical_field_weight).sum().sqrt()
            * (target_flat.square() * physical_field_weight).sum().sqrt()
        ).clamp_min(1e-8)
        predicted_rms = (
            (predicted_value.square() * physical_field_weight).sum()
            / row_denominator.sum().clamp_min(1.0)
        ).sqrt()
        active_common = active_float[..., None, None, None]
        predicted_common_rms = (
            (
                predicted_mean.square() * active_common * component_weight[None, None, None, None]
            ).sum()
            / (active_common.sum() * int(predicted_mean.shape[-2])).clamp_min(1.0)
        ).sqrt()
        common_mode_ratio = predicted_common_rms / predicted_rms.clamp_min(1e-8)
        selected_spread = target_spread[active_candidate]
        if int(selected_spread.numel()) > 0:
            spread_p25, spread_p50, spread_p75 = (
                torch.quantile(selected_spread, q) for q in (0.25, 0.50, 0.75)
            )
        else:
            spread_p25 = spread_p50 = spread_p75 = torch.zeros(
                (), device=target_spread.device, dtype=torch.float32
            )
        losses["hierarchical_mmdit_operation_value_loss"] = value_loss.detach().float()
        losses["hierarchical_mmdit_operation_value_weight"] = torch.as_tensor(
            operation_weight, device=predicted_value.device, dtype=torch.float32
        )
        losses["hierarchical_mmdit_operation_value_reliability_scale"] = (
            reliability_scale.detach().float()
        )
        losses["hierarchical_mmdit_operation_value_reliability"] = (
            (reliability.sum() / active_denominator).detach().float()
        )
        losses["hierarchical_mmdit_operation_value_target_spread"] = (
            ((target_spread * active_float).sum() / active_denominator).detach().float()
        )
        losses["hierarchical_mmdit_operation_value_target_spread_p25"] = spread_p25.detach().float()
        losses["hierarchical_mmdit_operation_value_target_spread_p50"] = spread_p50.detach().float()
        losses["hierarchical_mmdit_operation_value_target_spread_p75"] = spread_p75.detach().float()
        losses["hierarchical_mmdit_operation_value_correlation"] = correlation.detach().float()
        losses["hierarchical_mmdit_operation_value_pairwise_accuracy"] = (
            pairwise_accuracy.detach().float()
        )
        losses["hierarchical_mmdit_operation_value_decision_accuracy"] = (
            decision_accuracy.detach().float()
        )
        losses["hierarchical_mmdit_operation_value_common_mode_ratio"] = (
            common_mode_ratio.detach().float()
        )
        losses["hierarchical_mmdit_operation_candidate_coverage"] = valid.float().mean()
        if operation_weight > 0.0:
            losses["loss"] = losses["loss"] + operation_weight * value_loss
    # V70: stable-denominator replacements for the retired xratio gauge.
    # volume parity = noisy token norm vs workspace token norm (1.0 = parity;
    # after the noisy LayerNorm lands this pins to 1 by construction).
    # influence ratio = (attention share x value volume) of noisy vs workspace
    # evidence -- the honest "how much is the action listening to x_t" gauge.
    noisy_vol = output.get("latent_cvae_mmdit_noisy_token_norm")
    ws_vol = output.get("latent_cvae_workspace_token_norm")
    if torch.is_tensor(noisy_vol) and torch.is_tensor(ws_vol):
        losses["mmdit_noisy_volume_parity"] = (
            noisy_vol.detach().float() / ws_vol.detach().float().clamp_min(1e-6)
        )
        noisy_attn = output.get("latent_cvae_mmdit_action_noisy_attention")
        ws_attn = output.get("latent_cvae_mmdit_action_workspace_attention")
        if torch.is_tensor(noisy_attn) and torch.is_tensor(ws_attn):
            losses["mmdit_noisy_influence_ratio"] = (
                noisy_attn.detach().float() * noisy_vol.detach().float()
            ) / (ws_attn.detach().float() * ws_vol.detach().float()).clamp_min(1e-8)
    deterministic_intent_decoder = "intent_contract_deterministic" in output
    micro_losses = (
        {}
        if deterministic_intent_decoder
        else micro_refine_supervision_losses(
            system, sample, output, trainer, global_step=global_step
        )
    )
    for key, value in micro_losses.items():
        losses[key] = value.detach().float()
    micro_weight = float(getattr(trainer, "latent_cvae_micro_supervision_weight", 0.0))
    micro_event_weight = float(getattr(trainer, "latent_cvae_micro_event_weight", 0.0))
    micro_mono_weight = float(getattr(trainer, "latent_cvae_micro_monotonic_weight", 0.0))
    micro_kl_weight = float(getattr(trainer, "latent_cvae_micro_weight_kl_weight", 0.0))
    micro_smooth_weight = float(getattr(trainer, "latent_cvae_micro_coverage_smooth_weight", 0.0))
    micro_floor_weight = float(getattr(trainer, "latent_cvae_micro_coverage_floor_weight", 0.0))
    if micro_weight > 0 and "latent_cvae_micro_supervision" in micro_losses:
        losses["loss"] = (
            losses["loss"] + micro_weight * micro_losses["latent_cvae_micro_supervision"]
        )
    if micro_event_weight > 0 and "latent_cvae_micro_event" in micro_losses:
        losses["loss"] = (
            losses["loss"] + micro_event_weight * micro_losses["latent_cvae_micro_event"]
        )
    if micro_mono_weight > 0 and "latent_cvae_micro_monotonic" in micro_losses:
        losses["loss"] = (
            losses["loss"] + micro_mono_weight * micro_losses["latent_cvae_micro_monotonic"]
        )
    if micro_kl_weight > 0 and "latent_cvae_micro_weight_kl" in micro_losses:
        losses["loss"] = (
            losses["loss"] + micro_kl_weight * micro_losses["latent_cvae_micro_weight_kl"]
        )
    if micro_smooth_weight > 0 and "latent_cvae_micro_coverage_smooth" in micro_losses:
        losses["loss"] = (
            losses["loss"] + micro_smooth_weight * micro_losses["latent_cvae_micro_coverage_smooth"]
        )
    if micro_floor_weight > 0 and "latent_cvae_micro_coverage_floor" in micro_losses:
        losses["loss"] = (
            losses["loss"] + micro_floor_weight * micro_losses["latent_cvae_micro_coverage_floor"]
        )
    losses.update(rollout_diagnostics(output))
    if "gate_self" in output:
        losses["gate_self"] = output["gate_self"].detach()
        losses["gate_visual"] = output["gate_visual"].detach()
        losses["gate_rollout"] = output.get(
            "gate_rollout", torch.zeros_like(output["gate_self"])
        ).detach()
        losses["gate_ffn"] = output["gate_ffn"].detach()
    for key in (
        "mod_content_norm",
        "mod_time_norm",
        "mod_content_to_time",
        "future_conditioned_action_loss",
    ):
        if key in output:
            losses[key] = output[key].detach()
    if "rollout_alpha" in output:
        losses["rollout_alpha_mean"] = output["rollout_alpha"].detach().float().mean()
    for key in (
        "action_flow_residual_norm",
        "action_flow_raw_residual_norm",
        "action_flow_residual_alpha_mean",
        "action_flow_stage_router_entropy",
        "action_flow_stage_router_max",
        "latent_action_stage_router_entropy",
        "latent_action_stage_router_max",
        "latent_action_gripper_gate_mean",
        "latent_action_layer_memory_count",
        "latent_action_temporal_update_mean",
        "latent_action_temporal_near_depth",
        "latent_action_temporal_mid_depth",
        "latent_cvae_kl",
        "latent_cvae_kl_weight",
        "latent_cvae_prior_std",
        "latent_cvae_post_std",
        "latent_cvae_z_norm",
        "latent_cvae_prior_z_norm",
        "latent_cvae_post_z_norm",
        "latent_cvae_mu_gap",
        "latent_cvae_prior_pred_norm",
        "latent_cvae_post_pred_norm",
        "latent_cvae_post_gripper_gate_mean",
        "latent_cvae_condition_norm",
        "latent_cvae_condition_raw_norm",
        "latent_cvae_condition_scan_norm",
        "latent_cvae_condition_lateral_norm",
        "latent_cvae_layer_summary_norm",
        "latent_cvae_transition_source_raw_norm",
        "latent_cvae_transition_condition_norm",
        "latent_cvae_rollout_token_norm",
        "latent_cvae_rollout_token_count",
        "latent_cvae_consequence_scale_mean",
        "latent_cvae_consequence_gate_preference",
        "latent_cvae_consequence_mix_ratio",
        "latent_cvae_posterior_used",
        "latent_cvae_gripper_gate_mean",
        "latent_cvae_layer_memory_count",
        "latent_cvae_adaptive_refine_update_mean",
        "latent_cvae_adaptive_noisy_gate_mean",
        "latent_cvae_adaptive_noisy_branch_norm",
        "latent_cvae_adaptive_noisy_branch_ratio",
        "latent_cvae_adaptive_route_entropy",
        "latent_cvae_adaptive_route_max",
        "latent_cvae_adaptive_route_effective_slots",
        "latent_cvae_adaptive_progress_entropy",
        "latent_cvae_adaptive_progress_max",
        "latent_cvae_adaptive_progress_effective_slots",
        "latent_cvae_adaptive_progress_norm",
        "latent_cvae_adaptive_continue_mean",
        "latent_cvae_adaptive_prefix_norm",
        "latent_cvae_adaptive_progress_seed_entropy",
        "latent_cvae_adaptive_progress_seed_max",
        "latent_cvae_adaptive_progress_seed_effective_slots",
        "latent_cvae_adaptive_progress_seed_norm",
        "latent_cvae_adaptive_route_temperature_mean",
        "latent_cvae_adaptive_route_time_query_norm",
        "latent_cvae_adaptive_semantic_bias_norm",
        "latent_cvae_adaptive_function_delta_norm",
        "latent_cvae_adaptive_base_highfreq_norm",
        "latent_cvae_adaptive_refine_step_bias_norm",
        "latent_cvae_adaptive_capsule_layer_entropy",
        "latent_cvae_adaptive_capsule_layer_max",
        "latent_cvae_adaptive_capsule_layer_effective_slots",
        "latent_cvae_adaptive_condition_strength_mean",
        "latent_cvae_adaptive_condition_strength_std",
        "latent_cvae_adaptive_condition_strength_max",
        "latent_cvae_adaptive_condition_strength_min",
        "latent_cvae_adaptive_condition_residual_norm",
        "latent_cvae_adaptive_context_direction_norm",
        "latent_cvae_adaptive_micro_step_mean",
        "latent_cvae_adaptive_micro_step_std",
        "latent_cvae_adaptive_micro_progress_mean",
        "latent_cvae_adaptive_micro_kp_mean",
        "latent_cvae_adaptive_micro_kd_mean",
        "latent_cvae_adaptive_micro_feedforward_norm",
        "latent_cvae_adaptive_micro_feedback_norm",
        "latent_cvae_adaptive_micro_damping_norm",
        "latent_cvae_adaptive_micro_function_norm",
        "latent_cvae_adaptive_micro_control_norm",
        "latent_cvae_adaptive_micro_update_norm",
        "latent_cvae_adaptive_micro_heun_error",
        "latent_cvae_adaptive_micro_refine_block_norm",
        "latent_cvae_adaptive_micro_controller_norm",
        "latent_cvae_adaptive_regularizer",
        "latent_cvae_adaptive_regularizer_weight",
        "latent_cvae_adaptive_route_entropy_regularizer",
        "latent_cvae_adaptive_route_entropy_weight",
        "latent_cvae_mmdit_action_update_norm",
        "latent_cvae_mmdit_cond_update_norm",
        "latent_cvae_mmdit_action_cond_attention",
        "latent_cvae_mmdit_action_noisy_attention",
        "latent_cvae_mmdit_action_workspace_attention",
        "latent_cvae_mmdit_action_workspace_enrichment",
        "latent_cvae_mmdit_action_low_attention",
        "latent_cvae_mmdit_action_stage_attention",
        "latent_cvae_mmdit_action_low_enrichment",
        "latent_cvae_mmdit_action_stage_enrichment",
        "latent_cvae_mmdit_action_rollout_attention",
        "latent_cvae_mmdit_action_rollout_enrichment",
        "latent_cvae_mmdit_action_token_norm",
        "latent_cvae_mmdit_condition_token_norm",
        "latent_cvae_mmdit_noisy_token_norm",
        # V72: time-stratified x_t/workspace attention (sum+count pairs; the
        # epoch mean of sums divided by the epoch mean of counts recovers the
        # exact stratified mean, robust to empty buckets in small batches).
        "latent_cvae_mmdit_noisy_attn_t0_sum",
        "latent_cvae_mmdit_noisy_attn_t1_sum",
        "latent_cvae_mmdit_noisy_attn_t2_sum",
        "latent_cvae_mmdit_workspace_attn_t0_sum",
        "latent_cvae_mmdit_workspace_attn_t1_sum",
        "latent_cvae_mmdit_workspace_attn_t2_sum",
        "latent_cvae_mmdit_low_attn_t0_sum",
        "latent_cvae_mmdit_low_attn_t1_sum",
        "latent_cvae_mmdit_low_attn_t2_sum",
        "latent_cvae_mmdit_stage_attn_t0_sum",
        "latent_cvae_mmdit_stage_attn_t1_sum",
        "latent_cvae_mmdit_stage_attn_t2_sum",
        "latent_cvae_mmdit_attn_t0_count",
        "latent_cvae_mmdit_attn_t1_count",
        "latent_cvae_mmdit_attn_t2_count",
        "latent_cvae_primary_condition_norm",
        "latent_cvae_primary_z_effect_norm",
        "latent_cvae_workspace_progress_update_norm",
        # V72: echo probe -- fraction of the progress update attributable to
        # the raw action summary input (0 by construction under isolation).
        "latent_cvae_workspace_progress_action_dependence",
        # CR0 probes (do_before_v76): legacy stem activity + z interventions.
        "latent_cvae_legacy_stem_effect_ratio",
        "latent_cvae_z_zero_delta",
        "latent_cvae_z_shuffle_delta",
        "latent_cvae_workspace_token_count",
        "latent_cvae_workspace_token_norm",
        "latent_cvae_workspace_update_norm",
        "latent_cvae_workspace_global_state_norm",
        "latent_cvae_workspace_global_slot_delta_norm",
        "latent_cvae_workspace_global_slot_diversity",
        "latent_cvae_workspace_source_count",
        "latent_cvae_workspace_cached_token_fraction",
        "latent_cvae_workspace_attention_entropy",
        "latent_cvae_workspace_attention_max",
        "latent_cvae_workspace_group_attention_entropy",
        "latent_cvae_workspace_group_effective_sources",
        "latent_cvae_workspace_attention_mass_error",
        "latent_cvae_workspace_action_update_ratio",
        "latent_cvae_workspace_noisy_query_scale",
        "latent_cvae_workspace_progress_query_norm",
        "latent_cvae_workspace_role_geom_attention",
        "latent_cvae_workspace_role_transition_attention",
        "latent_cvae_workspace_role_event_attention",
        "latent_cvae_workspace_role_state_attention",
        "latent_cvae_workspace_role_layer_attention",
        "latent_cvae_workspace_role_global_attention",
        "latent_cvae_workspace_role_geom_token_count",
        "latent_cvae_workspace_role_transition_token_count",
        "latent_cvae_workspace_role_event_token_count",
        "latent_cvae_workspace_role_state_token_count",
        "latent_cvae_workspace_role_layer_token_count",
        "latent_cvae_workspace_role_global_token_count",
        "latent_cvae_workspace_controller_capacity",
        "latent_cvae_workspace_controller_delay",
        "latent_cvae_workspace_controller_temperature",
        "latent_cvae_workspace_controller_role_entropy",
        "latent_cvae_workspace_controller_role_max",
        "latent_cvae_workspace_controller_query_delta_norm",
        "latent_cvae_workspace_controller_workspace_delta_norm",
        "latent_cvae_workspace_controller_role_geom_prob",
        "latent_cvae_workspace_controller_role_transition_prob",
        "latent_cvae_workspace_controller_role_event_prob",
        "latent_cvae_workspace_controller_role_state_prob",
        "latent_cvae_workspace_controller_role_layer_prob",
        "latent_cvae_workspace_controller_role_global_prob",
        "latent_cvae_workspace_controller_role_geom_logit",
        "latent_cvae_workspace_controller_role_transition_logit",
        "latent_cvae_workspace_controller_role_event_logit",
        "latent_cvae_workspace_controller_role_state_logit",
        "latent_cvae_workspace_controller_role_layer_logit",
        "latent_cvae_workspace_controller_role_global_logit",
        "latent_cvae_hierarchical_low_token_count",
        "latent_cvae_hierarchical_low_token_norm",
        "latent_cvae_hierarchical_low_selector_stage_entropy",
        "latent_cvae_hierarchical_low_selector_stage_max",
        "latent_cvae_hierarchical_low_selector_stage_effective_slots",
        "latent_cvae_hierarchical_low_selector_role_norm",
        "latent_cvae_hierarchical_low_selector_content_norm",
        "latent_cvae_hierarchical_stage_token_count",
        "latent_cvae_hierarchical_stage_role_norm",
        "latent_cvae_hierarchical_stage_role_diversity",
        "latent_cvae_hierarchical_stage_content_norm",
        "latent_cvae_hierarchical_stage_content_diversity",
        "latent_cvae_hierarchical_stage_role_content_cosine",
        "latent_cvae_hierarchical_stage_role_output_norm",
        "latent_cvae_hierarchical_stage_content_output_norm",
        "latent_cvae_hierarchical_stage_role_output_fraction",
        "latent_cvae_hierarchical_stage_update_norm",
        "latent_cvae_hierarchical_stage_retain_mean",
        "latent_cvae_hierarchical_stage_promote_attention_entropy",
        "latent_cvae_hierarchical_stage_promote_attention_max",
        "latent_cvae_hierarchical_stage_promoted_norm",
        "latent_cvae_hierarchical_stage_promoted_projected_rms",
        "latent_cvae_hierarchical_stage_promoted_normalized_rms",
        "latent_cvae_hierarchical_stage_promoted_realized_scale",
        "latent_cvae_hierarchical_stage_promote_gate_scale_error",
        "latent_cvae_hierarchical_stage_promote_scale",
        "latent_cvae_hierarchical_manager_stage_attention_entropy",
        "latent_cvae_hierarchical_manager_stage_attention_max",
        "latent_cvae_hierarchical_manager_role_entropy",
        "latent_cvae_hierarchical_manager_role_max",
        "latent_cvae_hierarchical_manager_query_shift_norm",
        "latent_cvae_hierarchical_manager_promote_gate",
        "latent_cvae_hierarchical_manager_low_output_strength",
        "latent_cvae_hierarchical_manager_stage_output_strength",
        "latent_cvae_hierarchical_manager_role_geom_prob",
        "latent_cvae_hierarchical_manager_role_transition_prob",
        "latent_cvae_hierarchical_manager_role_event_prob",
        "latent_cvae_hierarchical_manager_role_state_prob",
        "latent_cvae_hierarchical_manager_role_layer_prob",
        "latent_cvae_hierarchical_manager_role_global_prob",
        "latent_cvae_workspace_layer_attention",
        "latent_cvae_workspace_scan_attention",
        "latent_cvae_workspace_lateral_attention",
        "latent_cvae_workspace_transition_attention",
        "latent_cvae_workspace_transition_delta_attention",
        "latent_cvae_workspace_transition_effect_attention",
        "latent_cvae_workspace_transition_timeline_attention",
        "latent_cvae_workspace_transition_total_attention",
        "latent_cvae_workspace_context_attention",
        "latent_cvae_workspace_visual_attention",
        "latent_cvae_workspace_trajectory_attention",
        "latent_cvae_workspace_rollout_attention",
        "latent_cvae_workspace_capsule_attention",
        "latent_cvae_workspace_progress_attention",
        "latent_cvae_workspace_routed_layer_attention",
        "consequence_self_condition",
        "consequence_self_condition_target_mse",
        "consequence_self_condition_noisy_mse",
        "consequence_preview_flow",
    ):
        if key in output:
            losses[key] = output[key].detach().float()
    probe_predictions = output.get("refinement_probe_pred_velocity")
    probe_active = output.get("refinement_probe_active")
    probe_response = output.get("refinement_probe_action_response_rel")
    probe_pressure = output.get("refinement_probe_stage_pressure_rel")
    probe_stage_ids = output.get("refinement_probe_stage_ids")
    probe_block_ids = output.get("refinement_probe_block_ids")
    target_velocity = output.get("target_physical_velocity")
    if all(
        torch.is_tensor(value)
        for value in (
            probe_predictions,
            probe_active,
            probe_response,
            probe_pressure,
            probe_stage_ids,
            probe_block_ids,
            target_velocity,
        )
    ):
        with torch.no_grad():
            if torch.is_tensor(refinement_step_error_cache) and tuple(
                refinement_step_error_cache.shape[:2]
            ) == tuple(probe_predictions.shape[:2]):
                step_error = refinement_step_error_cache
            else:
                residual = probe_predictions.float() - target_velocity.detach().float()[:, None]
                step_error = semantic_physical_velocity_error(
                    system,
                    residual,
                    arm_null_weight=trainer.arm_manifold_null_weight,
                ).mean(dim=-1)
            marginal_gain = step_error[:, :-1] - step_error[:, 1:]
            active = probe_active.float()
            denominator = active.sum().clamp_min(1.0)
            losses["hierarchical_mmdit_refine_error_initial"] = step_error[:, 0].mean()
            final_index = active.sum(dim=1).long().clamp_min(1)
            final_error = step_error.gather(1, final_index[:, None]).squeeze(1)
            losses["hierarchical_mmdit_refine_error_final"] = final_error.mean()
            losses["hierarchical_mmdit_refine_gain"] = (marginal_gain * active).sum() / denominator
            losses["hierarchical_mmdit_refine_positive_gain_fraction"] = (
                (marginal_gain > 0).float() * active
            ).sum() / denominator

            def masked_correlation(
                x: Tensor,
                y: Tensor,
                selected: Tensor | None = None,
            ) -> Tensor:
                if selected is None:
                    selected = active.bool()
                x_rows = x[selected].float()
                y_rows = y[selected].float()
                if int(x_rows.numel()) < 2:
                    return torch.zeros((), device=x.device, dtype=torch.float32)
                x_rows = x_rows - x_rows.mean()
                y_rows = y_rows - y_rows.mean()
                scale = x_rows.square().sum().sqrt() * y_rows.square().sum().sqrt()
                return (x_rows * y_rows).sum() / scale.clamp_min(1e-8)

            losses["hierarchical_mmdit_response_gain_corr"] = masked_correlation(
                probe_response,
                marginal_gain,
            )
            losses["hierarchical_mmdit_pressure_gain_corr"] = masked_correlation(
                probe_pressure,
                marginal_gain,
            )
            probe_time = output.get("time")
            if torch.is_tensor(probe_time) and tuple(probe_time.shape) == (int(active.shape[0]),):
                time_bins = torch.clamp(
                    (probe_time.detach().float().clamp(0.0, 1.0) * 3.0).long(),
                    max=2,
                )
                for time_bin in range(3):
                    selected = active.bool() & (time_bins[:, None] == time_bin)
                    selected_float = selected.float()
                    selected_denominator = selected_float.sum().clamp_min(1.0)
                    losses[f"hierarchical_mmdit_response_gain_corr_t{time_bin}"] = (
                        masked_correlation(probe_response, marginal_gain, selected)
                    )
                    losses[f"hierarchical_mmdit_pressure_gain_corr_t{time_bin}"] = (
                        masked_correlation(probe_pressure, marginal_gain, selected)
                    )
                    losses[f"hierarchical_mmdit_refine_gain_t{time_bin}"] = (
                        marginal_gain * selected_float
                    ).sum() / selected_denominator
                    losses[f"hierarchical_mmdit_refine_positive_gain_fraction_t{time_bin}"] = (
                        (marginal_gain > 0).float() * selected_float
                    ).sum() / selected_denominator
            for step_index in range(int(active.shape[1])):
                step_mask = active[:, step_index]
                step_denominator = step_mask.sum().clamp_min(1.0)
                losses[f"hierarchical_mmdit_step_{step_index}_gain"] = (
                    marginal_gain[:, step_index] * step_mask
                ).sum() / step_denominator
                losses[f"hierarchical_mmdit_step_{step_index}_positive_gain_fraction"] = (
                    (marginal_gain[:, step_index] > 0).float() * step_mask
                ).sum() / step_denominator
            for stage_index in range(int(system.policy_config.hierarchical_mmdit_operator_stages)):
                stage_mask = active * (probe_stage_ids == stage_index).float()
                stage_denominator = stage_mask.sum().clamp_min(1.0)
                losses[f"hierarchical_mmdit_stage_{stage_index}_gain"] = (
                    marginal_gain * stage_mask
                ).sum() / stage_denominator
            for block_index in range(int(system.policy_config.hierarchical_mmdit_depth)):
                block_mask = active * (probe_block_ids == block_index).float()
                block_denominator = block_mask.sum().clamp_min(1.0)
                losses[f"hierarchical_mmdit_block_{block_index}_gain"] = (
                    marginal_gain * block_mask
                ).sum() / block_denominator
    losses.update(
        _shadow_refinement_probe_metrics(
            system,
            output,
            arm_null_weight=trainer.arm_manifold_null_weight,
        )
    )

    # The deterministic decoder intentionally does not alias its diagnostics
    # into latent_cvae_* names.  Prefix pass-through keeps the new contract
    # auditable without reviving retired CVAE losses or zero-filled log fields.
    for key, value in output.items():
        if (
            key.startswith(("intent_", "owned_", "hierarchical_mmdit_"))
            and torch.is_tensor(value)
            and value.numel() == 1
        ):
            losses[key] = value.detach().float().reshape(())
    clean_noisy_vol = output.get("hierarchical_mmdit_noisy_token_norm")
    clean_cond_vol = output.get("hierarchical_mmdit_condition_token_norm")
    if torch.is_tensor(clean_noisy_vol) and torch.is_tensor(clean_cond_vol):
        losses["hierarchical_mmdit_noisy_volume_parity"] = (
            clean_noisy_vol.detach().float() / clean_cond_vol.detach().float().clamp_min(1e-6)
        )
    clean_noisy_fraction = output.get("hierarchical_mmdit_action_noisy_update_fraction")
    clean_stage_fraction = output.get("hierarchical_mmdit_action_stage_update_fraction")
    clean_low_fraction = output.get("hierarchical_mmdit_action_low_update_fraction")
    if all(
        torch.is_tensor(value)
        for value in (
            clean_noisy_fraction,
            clean_stage_fraction,
            clean_low_fraction,
        )
    ):
        workspace_fraction = (
            clean_stage_fraction.detach().float() + clean_low_fraction.detach().float()
        )
        losses["hierarchical_mmdit_noisy_to_workspace_update_ratio"] = (
            clean_noisy_fraction.detach().float() / workspace_fraction.clamp_min(1e-8)
        )
    return losses


def _midcut_aux_scale(trainer: V39PolicyTrainerConfig, epoch: int) -> float:
    base = float(getattr(trainer, "midcut_aux_loss_weight", 0.0))
    if base <= 0:
        return 0.0
    decay_epochs = max(int(getattr(trainer, "midcut_aux_decay_epochs", 0)), 0)
    final_ratio = float(getattr(trainer, "midcut_aux_final_ratio", 1.0))
    final_ratio = min(max(final_ratio, 0.0), 1.0)
    if decay_epochs <= 1:
        return base * final_ratio
    progress = min(max((int(epoch) - 1) / float(decay_epochs - 1), 0.0), 1.0)
    ratio = 1.0 + (final_ratio - 1.0) * progress
    return base * ratio


def _layer_contract_aux_scale(trainer: V39PolicyTrainerConfig, epoch: int) -> float:
    """Return the policy-stage layer-contract scale.

    Older entry points inherit the mid-cut schedule.  Newer experiments can
    opt into a distinct schedule without changing checkpoint compatibility.
    """

    configured = float(getattr(trainer, "layer_contract_aux_loss_weight", -1.0))
    if configured < 0.0:
        return _midcut_aux_scale(trainer, epoch)
    if configured == 0.0:
        return 0.0
    decay_epochs = int(getattr(trainer, "layer_contract_aux_decay_epochs", -1))
    final_ratio = float(getattr(trainer, "layer_contract_aux_final_ratio", -1.0))
    if decay_epochs < 0:
        decay_epochs = int(getattr(trainer, "midcut_aux_decay_epochs", 0))
    if final_ratio < 0.0:
        final_ratio = float(getattr(trainer, "midcut_aux_final_ratio", 1.0))
    final_ratio = min(max(final_ratio, 0.0), 1.0)
    if decay_epochs <= 1:
        return configured * final_ratio
    progress = min(max((int(epoch) - 1) / float(decay_epochs - 1), 0.0), 1.0)
    return configured * (1.0 + (final_ratio - 1.0) * progress)


def _midcut_as_primary(output: dict[str, Tensor]) -> dict[str, Tensor]:
    """Build a primary-output view using only Z_mid simple-head predictions."""
    required = ("midcut_pred_physical_velocity", "midcut_event_logits", "midcut_motion_logits")
    if not all(key in output for key in required):
        return output
    fake = dict(output)
    replacements = {
        "pred_physical_velocity": "midcut_pred_physical_velocity",
        "direct_physical_velocity": "midcut_direct_physical_velocity",
        "rollout_residual_velocity": "midcut_rollout_residual_velocity",
        "rollout_alpha": "midcut_rollout_alpha",
        "clean_physical_estimate": "midcut_clean_physical_estimate",
        "pred_action_estimate": "midcut_pred_action_estimate",
        "event_logits": "midcut_event_logits",
        "motion_logits": "midcut_motion_logits",
        "transition_latent": "midcut_transition_latent",
        "rollout_effect_pred": "midcut_rollout_effect_pred",
        "rollout_delta_pred": "midcut_rollout_delta_pred",
        "rollout_base_effect_pred": "midcut_rollout_base_effect_pred",
        "rollout_effect_pred_hold_action": "midcut_rollout_effect_pred_hold_action",
        "rollout_delta_pred_hold_action": "midcut_rollout_delta_pred_hold_action",
        "rollout_base_effect_pred_hold_action": "midcut_rollout_base_effect_pred_hold_action",
        "rollout_effect_pred_shuffle_action": "midcut_rollout_effect_pred_shuffle_action",
        "rollout_delta_pred_shuffle_action": "midcut_rollout_delta_pred_shuffle_action",
        "rollout_base_effect_pred_shuffle_action": "midcut_rollout_base_effect_pred_shuffle_action",
    }
    for dst, src in replacements.items():
        if src in output:
            fake[dst] = output[src]
    return fake


def midcut_contract_losses(
    system: V39PolicySystem,
    sample: dict[str, Tensor],
    output: dict[str, Tensor],
    trainer: V39PolicyTrainerConfig,
    *,
    enable_future_loss: bool = True,
) -> dict[str, Tensor]:
    """Losses for the mid-cut simple-head contract.

    These are deliberately reported separately from the final policy losses so
    Stage 2 can preserve Z_mid without letting the auxiliary objective dominate
    the deployable action decoder.
    """
    fake = _midcut_as_primary(output)
    base = v363_flow_losses(system, sample, fake, trainer)  # type: ignore[arg-type]
    dyn = rollout_dynamics_loss(fake)
    delta = rollout_delta_loss(fake)
    con = rollout_contrast_loss(fake, margin=float(trainer.rollout_contrast_margin))
    total = base["loss"]
    if enable_future_loss:
        total = total + float(getattr(trainer, "midcut_rollout_dynamics_loss_weight", 0.0)) * dyn
        total = total + float(getattr(trainer, "midcut_rollout_delta_loss_weight", 0.0)) * delta
        total = total + float(getattr(trainer, "midcut_rollout_contrast_loss_weight", 0.0)) * con
    out = {f"midcut_{key}": value for key, value in base.items() if torch.is_tensor(value)}
    out["midcut_rollout_dynamics"] = dyn
    out["midcut_rollout_delta"] = delta
    out["midcut_rollout_contrast"] = con
    out["midcut_contract"] = total
    out.update({f"midcut_{key}": value for key, value in rollout_diagnostics(fake).items()})
    return out


def _uses_layer_adapter_contract(trainer: V39PolicyTrainerConfig) -> bool:
    mode = str(getattr(trainer, "contract_mode", "midcut")).lower().replace("-", "_")
    if mode in {
        "layer",
        "layers",
        "adapter",
        "layer_adapter",
        "multilayer",
        "multi_layer",
        "multi_layer_adapter",
    }:
        return True
    if mode in {"midcut", "mid_cut"}:
        return False
    raise ValueError(f"unknown contract_mode: {trainer.contract_mode!r}")


def _needs_future_targets(trainer: V39PolicyTrainerConfig, epoch: int) -> bool:
    """Whether this epoch has an active objective that reads future observations."""

    if epoch < int(trainer.future_latent_loss_start_epoch):
        return False
    direct_weights = (
        "flow_jepa_future_loss_weight",
        "flow_jepa_future_change_loss_weight",
        "flow_jepa_horizon_address_loss_weight",
        "flow_jepa_interval_stage_loss_weight",
        "flow_jepa_stage_loss_weight",
        "rollout_dynamics_loss_weight",
        "rollout_delta_loss_weight",
        "rollout_contrast_loss_weight",
        "rollout_variance_loss_weight",
        "rollout_norm_loss_weight",
        "rollout_milestone_delta_match_weight",
        "future_latent_loss_weight",
        "action_effect_loss_weight",
    )
    if any(float(getattr(trainer, name, 0.0)) > 0.0 for name in direct_weights):
        return True
    layer_mode = _uses_layer_adapter_contract(trainer)
    contract_stage = _is_contract_stage(trainer)
    if contract_stage and layer_mode:
        return any(
            float(getattr(trainer, name, 0.0)) > 0.0
            for name in (
                "layer_latent_loss_weight",
                "layer_contrast_loss_weight",
                "layer_variance_loss_weight",
                "layer_norm_loss_weight",
                "layer_delta_match_loss_weight",
            )
        )
    if contract_stage:
        return False
    aux_scale = (
        _layer_contract_aux_scale(trainer, epoch)
        if layer_mode
        else _midcut_aux_scale(trainer, epoch)
    )
    if aux_scale <= 0.0:
        return False
    names = (
        (
            "layer_latent_loss_weight",
            "layer_contrast_loss_weight",
            "layer_variance_loss_weight",
            "layer_norm_loss_weight",
            "layer_delta_match_loss_weight",
        )
        if layer_mode
        else (
            "midcut_rollout_dynamics_loss_weight",
            "midcut_rollout_delta_loss_weight",
            "midcut_rollout_contrast_loss_weight",
        )
    )
    return any(float(getattr(trainer, name, 0.0)) > 0.0 for name in names)


def _needs_action_counterfactuals(trainer: V39PolicyTrainerConfig, epoch: int) -> bool:
    """Whether hold/shuffle predictions contribute to this epoch's backward scalar."""

    if epoch < int(trainer.future_latent_loss_start_epoch):
        return False
    if float(getattr(trainer, "rollout_contrast_loss_weight", 0.0)) > 0.0:
        return True
    layer_mode = _uses_layer_adapter_contract(trainer)
    contract_stage = _is_contract_stage(trainer)
    if contract_stage:
        return bool(layer_mode and float(getattr(trainer, "layer_contrast_loss_weight", 0.0)) > 0.0)
    if layer_mode:
        return bool(
            _layer_contract_aux_scale(trainer, epoch) > 0.0
            and float(getattr(trainer, "layer_contrast_loss_weight", 0.0)) > 0.0
        )
    return bool(
        _midcut_aux_scale(trainer, epoch) > 0.0
        and float(getattr(trainer, "midcut_rollout_contrast_loss_weight", 0.0)) > 0.0
    )


def _layer_contract_weight(index: int, count: int) -> float:
    if count <= 1:
        return 1.0
    if count == 6:
        return [0.10, 0.30, 1.00, 1.00, 0.30, 0.10][int(index)]
    if count == 4:
        return [0.30, 1.00, 1.00, 0.30][int(index)]
    if count == 3:
        return [0.30, 1.00, 0.30][int(index)]
    center = 0.5 * float(count - 1)
    distance = abs(float(index) - center) / max(center, 1.0)
    return max(0.10, 1.0 - 0.90 * distance)


def _layer_contract_as_primary(
    system: V39PolicySystem,
    sample: dict[str, Tensor],
    output: dict[str, Tensor],
    entry: dict[str, Tensor],
) -> dict[str, Tensor]:
    fake = dict(output)
    # These describe the top-level controlled-dynamics decomposition. A layer
    # consequence entry has a different contract and must not inherit them.
    fake.pop("rollout_decomposition_expansion_ratio", None)
    fake.pop("rollout_base_is_fixed_zero", None)
    replacements = {
        "pred_physical_velocity": "pred_physical_velocity",
        "direct_physical_velocity": "direct_physical_velocity",
        "rollout_residual_velocity": "rollout_residual_velocity",
        "rollout_alpha": "rollout_alpha",
        "clean_physical_estimate": "clean_physical_estimate",
        "pred_action_estimate": "pred_action_estimate",
        "event_logits": "event_logits",
        "motion_logits": "motion_logits",
        "transition_latent": "transition_latent",
        "rollout_effect_pred": "rollout_effect_pred",
        "rollout_delta_pred": "rollout_delta_pred",
        "rollout_base_effect_pred": "rollout_base_effect_pred",
        "rollout_effect_pred_hold_action": "rollout_effect_pred_hold_action",
        "rollout_delta_pred_hold_action": "rollout_delta_pred_hold_action",
        "rollout_base_effect_pred_hold_action": "rollout_base_effect_pred_hold_action",
        "rollout_effect_pred_shuffle_action": "rollout_effect_pred_shuffle_action",
        "rollout_delta_pred_shuffle_action": "rollout_delta_pred_shuffle_action",
        "rollout_base_effect_pred_shuffle_action": "rollout_base_effect_pred_shuffle_action",
        "milestone_step_delta_pred": "milestone_step_delta_pred",
        "milestone_step_delta_pred_hold_action": "milestone_step_delta_pred_hold_action",
        "milestone_step_delta_pred_shuffle_action": "milestone_step_delta_pred_shuffle_action",
        "milestone_step_delta_pred_shuffle_state": "milestone_step_delta_pred_shuffle_state",
        "rollout_effect_pred_shuffle_state": "rollout_effect_pred_shuffle_state",
        "rollout_delta_pred_shuffle_state": "rollout_delta_pred_shuffle_state",
        "causal_rollout_effect_pred": "causal_rollout_effect_pred",
        "latent_rollout_effect_pred": "latent_rollout_effect_pred",
        "policy_effect_tokens": "policy_effect_tokens",
        "unified_intervention_latent_pred": "unified_intervention_latent_pred",
        "neutral_latent_pred": "neutral_latent_pred",
        "layer_causal_gain": "layer_causal_gain",
        "layer_latent_gain": "layer_latent_gain",
    }
    for dst, src in replacements.items():
        if src in entry:
            fake[dst] = entry[src]
    if (
        "clean_physical_estimate" not in fake
        and "time" in output
        and "noisy_physical_action" in output
    ):
        t = output["time"].to(
            device=fake["pred_physical_velocity"].device, dtype=fake["pred_physical_velocity"].dtype
        )
        boundary = sample["action_state"].to(
            device=fake["pred_physical_velocity"].device, dtype=fake["pred_physical_velocity"].dtype
        )
        # V70 (H3 fix): project before decode, matching the training forward
        # and deployment geometry.
        clean = system.codec.project_physical(
            output["noisy_physical_action"] - t[:, None, None] * fake["pred_physical_velocity"],
            boundary,
        )
        fake["clean_physical_estimate"] = clean
        fake["pred_action_estimate"] = system.codec.decode(clean, boundary)
    return fake


def layer_contract_losses(
    system: V39PolicySystem,
    sample: dict[str, Tensor],
    output: dict[str, Tensor],
    trainer: V39PolicyTrainerConfig,
    *,
    enable_future_loss: bool = True,
) -> dict[str, Tensor]:
    """Weighted multi-layer latent/action-probe contract losses.

    V39.2 makes the world/future latent head the primary per-layer objective.
    The shared flow-matching action probe is deliberately low weight and reads
    only the layer latent produced by the tiny adapter.  This avoids six full
    action heads while still testing whether each layer latent is action-usable.
    """
    layers = output.get("layer_contracts")
    if not isinstance(layers, list) or not layers:
        z = torch.zeros(
            (),
            device=output["pred_physical_velocity"].device,
            dtype=output["pred_physical_velocity"].dtype,
        )
        return {"loss": z, "layer_contract": z, "layer_contract_weight_sum": z}

    total: Tensor | None = None
    weight_sum = 0.0
    metric_acc: dict[str, Tensor] = {}
    log_rows: dict[str, Tensor] = {}
    count = len(layers)

    w_latent = float(getattr(trainer, "layer_latent_loss_weight", 1.0))
    w_fm = float(getattr(trainer, "layer_fm_probe_loss_weight", 0.02))
    w_event = float(getattr(trainer, "layer_event_loss_weight", 0.05))
    w_motion = float(getattr(trainer, "layer_motion_loss_weight", 0.03))
    w_decoded = float(getattr(trainer, "layer_decoded_action_loss_weight", 0.0))
    w_contrast = float(getattr(trainer, "layer_contrast_loss_weight", 0.0))
    w_var = float(getattr(trainer, "layer_variance_loss_weight", 0.0))
    w_norm = float(getattr(trainer, "layer_norm_loss_weight", 0.0))
    w_delta_match = float(getattr(trainer, "layer_delta_match_loss_weight", 0.0))
    grid = (
        int(system.policy_config.num_cameras)
        * int(system.policy_config.future_grid_size)
        * int(system.policy_config.future_grid_size)
    )

    for i, entry in enumerate(layers):
        weight = float(_layer_contract_weight(i, count))
        fake = _layer_contract_as_primary(system, sample, output, entry)
        base = v363_flow_losses(system, sample, fake, trainer)  # type: ignore[arg-type]
        dyn = rollout_dynamics_loss(fake)
        delta = rollout_delta_loss(fake)
        con = rollout_contrast_loss(fake, margin=float(trainer.rollout_contrast_margin))
        var_loss = latent_variance_loss(fake, grid=grid)
        norm_loss = latent_norm_loss(fake)
        delta_match = milestone_delta_match_loss(fake, grid=grid)

        # The latent/future objective is the primary readout contract.  It is
        # only enabled when future target tokens are present.  Action-flow is a
        # shared probe downstream of the latent and should remain lightweight.
        zero = torch.zeros_like(base["physical_flow"])
        layer_total = zero
        if enable_future_loss:
            layer_total = layer_total + w_latent * dyn
            # Delta keeps a separate action-conditioned latent residual alive;
            # default follows the existing midcut delta weight.
            layer_total = (
                layer_total
                + float(getattr(trainer, "midcut_rollout_delta_loss_weight", 0.0)) * delta
            )
            layer_total = layer_total + w_delta_match * delta_match
            layer_total = layer_total + w_var * var_loss
            layer_total = layer_total + w_norm * norm_loss
            layer_total = layer_total + w_contrast * con
        layer_total = layer_total + w_fm * base["physical_flow"]
        layer_total = layer_total + w_event * base["event"]
        layer_total = layer_total + w_motion * base["motion"]
        if w_decoded > 0:
            layer_total = layer_total + w_decoded * base["decoded_action"]

        total = layer_total * weight if total is None else total + layer_total * weight
        weight_sum += weight
        diag = rollout_diagnostics(fake)
        merged = {
            **{k: v for k, v in base.items() if torch.is_tensor(v)},
            "latent": dyn,
            "rollout_dynamics": dyn,
            "rollout_delta": delta,
            "rollout_contrast": con,
            "latent_variance": var_loss,
            "latent_norm": norm_loss,
            "milestone_delta_match": delta_match,
            **diag,
        }
        for key, value in merged.items():
            if not torch.is_tensor(value):
                continue
            log_key = "layer_base_loss" if key == "loss" else key
            metric_acc[log_key] = (
                metric_acc.get(log_key, torch.zeros_like(value.detach().float()))
                + value.detach().float() * weight
            )
        log_rows[f"layer{i}_contract"] = layer_total.detach()
        for key in (
            "latent",
            "rollout_dynamics",
            "physical_flow",
            "decoded_action",
            "event",
            "motion",
            "rollout_delta",
            "rollout_delta_shuffle",
            "rollout_delta_hold",
            "rollout_contrast",
            "rollout_effect_change_shuffle",
            "rollout_effect_change_hold",
            "latent_variance",
            "latent_norm",
            "milestone_delta_match",
            "rollout_pred_std_ratio",
            "rollout_pred_norm_ratio",
            "rollout_milestone_delta_norm_ratio",
            "milestone_gate_mean",
            "milestone_step_delta_norm",
            "milestone_effect_norm",
            "milestone_effect_std",
            "layer_causal_gain",
            "layer_latent_gain",
            "step_delta_shuffle",
            "step_delta_hold",
            "step_delta_change_shuffle",
            "step_delta_change_hold",
            "step_delta_state_shuffle",
            "step_delta_change_state_shuffle",
            "rollout_delta_state_shuffle",
            "rollout_effect_change_state_shuffle",
            "rollout_full_effect_change_state_shuffle",
        ):
            if key in merged:
                log_rows[f"layer{i}_{key}"] = merged[key].detach()
        if "consequence_zero_base_shift" in entry:
            log_rows[f"layer{i}_czbase"] = entry["consequence_zero_base_shift"].detach()
    assert total is not None
    denom = max(weight_sum, 1e-6)
    contract = total / denom
    out: dict[str, Tensor] = {
        "loss": contract * float(getattr(trainer, "layer_contract_loss_weight", 1.0)),
        "layer_contract": contract,
        "layer_contract_weight_sum": torch.as_tensor(
            weight_sum, device=contract.device, dtype=contract.dtype
        ),
        "layer_latent_weight": torch.as_tensor(
            w_latent, device=contract.device, dtype=contract.dtype
        ),
        "layer_fm_probe_weight": torch.as_tensor(
            w_fm, device=contract.device, dtype=contract.dtype
        ),
        "layer_contrast_weight": torch.as_tensor(
            w_contrast, device=contract.device, dtype=contract.dtype
        ),
        "layer_variance_weight": torch.as_tensor(
            w_var, device=contract.device, dtype=contract.dtype
        ),
        "layer_norm_weight": torch.as_tensor(w_norm, device=contract.device, dtype=contract.dtype),
        "layer_delta_match_weight": torch.as_tensor(
            w_delta_match, device=contract.device, dtype=contract.dtype
        ),
    }
    zero_base = [
        entry["consequence_zero_base_shift"]
        for entry in layers
        if torch.is_tensor(entry.get("consequence_zero_base_shift"))
    ]
    if zero_base:
        out["consequence_zero_base_shift"] = torch.stack(zero_base).mean()
    for key, value in metric_acc.items():
        out[key] = value / denom
    out.update(log_rows)
    return out


def motion_head_metrics(
    logits_rows: list[np.ndarray], target_rows: list[np.ndarray]
) -> dict[str, float]:
    """Binary validation metrics for the arm-motion auxiliary head."""

    if not logits_rows:
        return {}
    logits = np.concatenate(logits_rows, axis=0)
    target = np.concatenate(target_rows, axis=0) >= 0.5
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    pred = probability >= 0.5
    tp = float(np.logical_and(pred, target).sum())
    fp = float(np.logical_and(pred, ~target).sum())
    fn = float(np.logical_and(~pred, target).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    return {
        "motion_head_accuracy": float((pred == target).mean()),
        "motion_head_precision": float(precision),
        "motion_head_recall": float(recall),
        "motion_head_f1": float(f1),
        "motion_head_pred_moving": float(pred.sum()),
        "motion_head_target_moving": float(target.sum()),
        "motion_head_predicted_rate": float(pred.mean()),
        "motion_head_target_rate": float(target.mean()),
        "motion_head_mean_probability": float(probability.mean()),
    }


@torch.no_grad()
def evaluate_flow_jepa_stage1(
    *,
    system: V39PolicySystem,
    loader: DataLoader,
    conditioner: RDT2Conditioner,
    device: torch.device,
    dtype: torch.dtype,
    camera_names: Sequence[str],
    trainer: V39PolicyTrainerConfig,
    max_batches: int = 0,
    memory_reporter: CudaMemoryReporter | None = None,
    epoch: int | None = None,
    global_step: int | None = None,
) -> dict[str, float]:
    """Teacher-forced validation for the new V95 representation Stage1.

    A Stage1 checkpoint is selected only by its frozen-DINO representation
    objective.  Deploy action sampling belongs to Stage2 and would evaluate a
    deliberately untrained final decoder here.
    """

    system.eval()
    sums: dict[str, float] = {}
    count = 0
    for batch_index, batch in enumerate(loader, start=1):
        if max_batches and batch_index > int(max_batches):
            break
        report_mem = memory_reporter is not None and memory_reporter.should_report(batch_index)
        if report_mem:
            memory_reporter.reset_peak()
        sample = prepare_v39_policy_sample(
            batch,
            conditioner=conditioner,
            system=system,
            camera_names=camera_names,
            device=device,
            dtype=dtype,
            include_target_visual=True,
        )
        with autocast_context(device, dtype):
            output = system.flow_jepa_stage1_forward(
                sample["visual"],
                sample["history_state"],
                sample["executed_action_history"],
                sample["state"],
                sample["target_visual"],
                raw_visual=sample.get("raw_visual"),
            )
            losses = flow_jepa_stage1_losses(output, trainer, enable_future_loss=True)
        for key, value in losses.items():
            if torch.is_tensor(value) and value.numel() == 1:
                sums[key] = sums.get(key, 0.0) + float(value.detach().float().cpu())
        count += 1
        if report_mem:
            memory_reporter.snapshot(
                tag="stage1_eval_batch_end",
                phase="eval",
                epoch=epoch,
                batch=batch_index,
                global_step=global_step,
                print_line=True,
            )
    if count <= 0:
        raise RuntimeError("V95 Stage1 validation loader produced no batches")
    metrics = {key: value / float(count) for key, value in sums.items()}
    metrics["eval_batches"] = float(count)
    metrics["eval_representation_batches"] = float(count)
    metrics["eval_representation_coverage"] = 1.0
    metrics["eval_teacher_forced"] = 1.0
    metrics["eval_uses_target_action"] = 0.0
    metrics["flow_jepa_stage1_validation"] = 1.0
    return metrics


@torch.no_grad()
def evaluate_v39_policy(
    *,
    system: V39PolicySystem,
    loader: DataLoader,
    conditioner: RDT2Conditioner,
    device: torch.device,
    dtype: torch.dtype,
    camera_names: Sequence[str],
    action_normalizer: ArrayNormalizer,
    trainer: V39PolicyTrainerConfig,
    max_batches: int = 0,
    memory_reporter: CudaMemoryReporter | None = None,
    epoch: int | None = None,
    global_step: int | None = None,
) -> dict[str, float]:
    system.eval()
    pred_rows, target_rows, current_rows = [], [], []
    no_proposal_rows = []
    no_proposal_target_rows = []
    proposal_ablation_pred_rows = []
    execution_ablation_pred_rows: dict[str, list[np.ndarray]] = {
        name: []
        for name in ("primary", "hard", "neutral", "full_capacity", "three_basis_reduction")
    }
    execution_ablation_target_rows: list[np.ndarray] = []
    event_logits_rows: list[np.ndarray] = []
    event_target_rows: list[np.ndarray] = []
    motion_logits_rows: list[np.ndarray] = []
    motion_target_rows: list[np.ndarray] = []
    eval_field_null_sse = 0.0
    eval_field_energy = 0.0
    eval_field_null_count = 0
    eval_noise_projection_sse = 0.0
    eval_noise_projection_count = 0
    eval_arm_field_null_sse = 0.0
    eval_arm_field_energy = 0.0
    eval_arm_field_null_count = 0
    eval_arm_noise_projection_sse = 0.0
    eval_arm_noise_projection_count = 0
    contract_eval = _is_contract_stage(trainer) and _uses_layer_adapter_contract(trainer)
    representation_eval = bool(int(getattr(system.policy_config, "flow_jepa_enabled", 0))) and (
        float(getattr(trainer, "flow_jepa_future_loss_weight", 0.0)) > 0.0
        or float(getattr(trainer, "flow_jepa_horizon_address_loss_weight", 0.0)) > 0.0
        or float(getattr(trainer, "flow_jepa_interval_stage_loss_weight", 0.0)) > 0.0
        or float(getattr(trainer, "flow_jepa_stage_loss_weight", 0.0)) > 0.0
    )
    contract_metric_sums: dict[str, float] = {}
    contract_metric_count = 0
    sampling_diagnostic_sums: dict[str, float] = {}
    sampling_diagnostic_counts: dict[str, int] = {}
    completed_batches = 0
    sampling_diagnostic_batches = 0
    proposal_ablation_batches = 0
    proposal_ablation_samples = 0
    execution_ablation_batches = 0
    representation_eval_batches = 0
    eval_started_at = time.perf_counter()
    primary_sample_seconds = 0.0
    proposal_ablation_seconds = 0.0
    planned_batches = len(loader)
    if max_batches:
        planned_batches = min(planned_batches, int(max_batches))

    def diagnostic_batch_indices(budget: int) -> set[int]:
        if budget < 0:
            raise ValueError("eval diagnostic batch budgets must be non-negative")
        if budget == 0 or budget >= planned_batches:
            return set(range(1, planned_batches + 1))
        if budget == 1:
            return {1 + (planned_batches - 1) // 2}
        return {
            1 + round(index * (planned_batches - 1) / float(budget - 1)) for index in range(budget)
        }

    sampling_diagnostic_indices = diagnostic_batch_indices(
        int(trainer.eval_sampling_diagnostic_batches)
    )
    proposal_ablation_indices = diagnostic_batch_indices(
        int(trainer.eval_proposal_ablation_batches)
    )
    execution_ablation_indices = diagnostic_batch_indices(
        int(trainer.eval_execution_ablation_batches)
    )
    representation_eval_indices = (
        diagnostic_batch_indices(int(trainer.eval_representation_batches))
        if representation_eval
        else set()
    )
    evidence_execution_decoder = getattr(
        system.planner, "evidence_latent_mmdit_action_decoder", None
    )
    shadow_probe_eval = (
        str(getattr(system.policy_config, "final_action_decoder", "legacy"))
        == "hierarchical_mmdit_action"
        and str(getattr(system.policy_config, "hierarchical_mmdit_exhaustion_mode", "off"))
        == "shadow"
    )
    for batch_index, batch in enumerate(loader, start=1):
        if max_batches and batch_index > max_batches:
            break
        completed_batches += 1
        collect_sampling_diagnostics = batch_index in sampling_diagnostic_indices
        run_proposal_ablation = batch_index in proposal_ablation_indices
        run_execution_ablation = (
            evidence_execution_decoder is not None and batch_index in execution_ablation_indices
        )
        run_representation_eval = batch_index in representation_eval_indices
        report_mem = memory_reporter is not None and memory_reporter.should_report(batch_index)
        if report_mem:
            memory_reporter.reset_peak()
            if memory_reporter.detail:
                memory_reporter.snapshot(
                    tag="eval_batch_start",
                    phase="eval",
                    epoch=epoch,
                    batch=batch_index,
                    global_step=global_step,
                )
        sample = prepare_v39_policy_sample(
            batch,
            conditioner=conditioner,
            system=system,
            camera_names=camera_names,
            device=device,
            dtype=dtype,
            include_target_visual=contract_eval or run_representation_eval,
        )
        if report_mem and memory_reporter.detail:
            memory_reporter.snapshot(
                tag="eval_after_prepare",
                phase="eval",
                epoch=epoch,
                batch=batch_index,
                global_step=global_step,
            )
        generator = torch.Generator(device=device)
        generator.manual_seed(37237 + batch_index)
        noise = system.codec.sample_noise(
            sample["policy_action"].shape[0],
            generator=generator,
            device=device,
            dtype=sample["visual"].dtype,
            action_state=sample["action_state"],
        )
        contract_losses: dict[str, Tensor] | None = None
        primary_start_event: torch.cuda.Event | None = None
        primary_end_event: torch.cuda.Event | None = None
        ablation_start_event: torch.cuda.Event | None = None
        ablation_end_event: torch.cuda.Event | None = None
        primary_started_at = time.perf_counter()
        with autocast_context(device, dtype):
            # Action metrics always use deploy-style sampling.  Layer-contract
            # stages additionally run a separately labelled teacher-forced
            # contract evaluation below; its values never enter action metrics.
            stop_midcut_eval = _is_contract_stage(trainer) and not _uses_layer_adapter_contract(
                trainer
            )
            if device.type == "cuda":
                primary_start_event = torch.cuda.Event(enable_timing=True)
                primary_end_event = torch.cuda.Event(enable_timing=True)
                primary_start_event.record()
            pred_pack = system.sample(
                sample["visual"],
                sample["history_state"],
                sample["executed_action_history"],
                sample["state"],
                raw_visual=sample.get("raw_visual"),
                action_state=sample["action_state"],
                steps=trainer.eval_inference_steps,
                noise=noise,
                use_proposal=True,
                return_event_logits=True,
                stop_at_midcut=stop_midcut_eval,
                collect_diagnostics=collect_sampling_diagnostics,
            )
            if primary_end_event is not None:
                primary_end_event.record()
            else:
                primary_sample_seconds += time.perf_counter() - primary_started_at
            assert isinstance(pred_pack, dict)
            diagnostic_weight = int(pred_pack["action"].shape[0])
            if collect_sampling_diagnostics:
                sampling_diagnostic_batches += 1
            sampling_diagnostic_items: list[tuple[str, Tensor]] = []
            for key, value in pred_pack.items():
                keep_sampling_diagnostic = (
                    key.startswith("sample_latent_cvae_")
                    or key.startswith("sample_intent_")
                    or key.startswith("sample_owned_")
                    or key.startswith("sample_hierarchical_mmdit_")
                    or key.startswith("sample_evidence_")
                    or key.startswith("sample_arm_null_")
                    or key.startswith("sample_grip_null_")
                )
                if keep_sampling_diagnostic and torch.is_tensor(value) and value.numel() == 1:
                    sampling_diagnostic_items.append((key, value.detach().float().reshape(())))
            if sampling_diagnostic_items:
                # One device-to-host transfer per validation batch, rather
                # than one synchronization for every scalar gauge.
                diagnostic_values = (
                    torch.stack([value for _, value in sampling_diagnostic_items]).cpu().tolist()
                )
                for (key, _), value in zip(
                    sampling_diagnostic_items, diagnostic_values, strict=True
                ):
                    sampling_diagnostic_sums[key] = (
                        sampling_diagnostic_sums.get(key, 0.0) + float(value) * diagnostic_weight
                    )
                    sampling_diagnostic_counts[key] = (
                        sampling_diagnostic_counts.get(key, 0) + diagnostic_weight
                    )
            no_proposal: Tensor | dict[str, Tensor] | None = None
            execution_ablation_packs: dict[str, Tensor | dict[str, Tensor]] = {}
            if run_proposal_ablation:
                ablation_started_at = time.perf_counter()
                if device.type == "cuda":
                    ablation_start_event = torch.cuda.Event(enable_timing=True)
                    ablation_end_event = torch.cuda.Event(enable_timing=True)
                    ablation_start_event.record()
                no_proposal = system.sample(
                    sample["visual"],
                    sample["history_state"],
                    sample["executed_action_history"],
                    sample["state"],
                    raw_visual=sample.get("raw_visual"),
                    action_state=sample["action_state"],
                    steps=trainer.eval_inference_steps,
                    noise=noise,
                    use_proposal=False,
                    stop_at_midcut=stop_midcut_eval,
                    collect_diagnostics=False,
                )
                if ablation_end_event is not None:
                    ablation_end_event.record()
                else:
                    proposal_ablation_seconds += time.perf_counter() - ablation_started_at
                proposal_ablation_batches += 1
                proposal_ablation_samples += diagnostic_weight
            if run_execution_ablation:
                execution_ablation_batches += 1
                rank = max(
                    int(getattr(system.policy_config, "latent_cvae_mmdit_operator_rank", 32)),
                    1,
                )
                ablation_specs = {
                    "hard": ("hard", None),
                    "neutral": ("neutral", 1.0),
                    "full_capacity": ("soft", 1.0),
                    "three_basis_reduction": (
                        "soft",
                        max(float(rank - 3), 1.0) / float(rank),
                    ),
                }
                for name, (policy, capacity_gate) in ablation_specs.items():
                    evidence_execution_decoder.set_execution_eval_ablation(
                        policy=policy,
                        capacity_gate=capacity_gate,
                    )
                    try:
                        execution_ablation_packs[name] = system.sample(
                            sample["visual"],
                            sample["history_state"],
                            sample["executed_action_history"],
                            sample["state"],
                            raw_visual=sample.get("raw_visual"),
                            action_state=sample["action_state"],
                            steps=trainer.eval_inference_steps,
                            noise=noise,
                            use_proposal=True,
                            stop_at_midcut=stop_midcut_eval,
                            collect_diagnostics=False,
                        )
                    finally:
                        evidence_execution_decoder.clear_execution_eval_ablation()
            if shadow_probe_eval and collect_sampling_diagnostics:
                fork_devices = (
                    [device.index if device.index is not None else torch.cuda.current_device()]
                    if device.type == "cuda"
                    else []
                )
                with torch.random.fork_rng(devices=fork_devices):
                    probe_seed = 91073 + batch_index
                    torch.manual_seed(probe_seed)
                    if device.type == "cuda":
                        torch.cuda.manual_seed_all(probe_seed)
                    shadow_output = system.flow_training_forward(
                        sample["visual"],
                        sample["history_state"],
                        sample["executed_action_history"],
                        sample["state"],
                        sample["policy_action"],
                        raw_visual=sample.get("raw_visual"),
                        action_state=sample["action_state"],
                        proposal_dropout=0.0,
                        make_counterfactuals=False,
                        stop_at_midcut=False,
                    )
                shadow_metrics = _shadow_refinement_probe_metrics(
                    system,
                    shadow_output,
                    arm_null_weight=trainer.arm_manifold_null_weight,
                )
                shadow_items = [
                    (key, value.detach().float().reshape(()))
                    for key, value in shadow_metrics.items()
                    if torch.is_tensor(value) and value.numel() == 1
                ]
                shadow_values = (
                    torch.stack([value for _, value in shadow_items]).cpu().tolist()
                    if shadow_items
                    else []
                )
                for (key, _), value in zip(shadow_items, shadow_values, strict=True):
                    metric_key = f"sample_{key}"
                    sampling_diagnostic_sums[metric_key] = (
                        sampling_diagnostic_sums.get(metric_key, 0.0)
                        + float(value) * diagnostic_weight
                    )
                    sampling_diagnostic_counts[metric_key] = (
                        sampling_diagnostic_counts.get(metric_key, 0) + diagnostic_weight
                    )
            if system.codec.uses_parseval_gripper_field:
                ad = int(system.policy_config.arm_dim)
                gf = int(system.policy_config.gripper_field_dim)
                noise_field = noise[..., 2 * ad : 2 * ad + gf]
                noise_null = noise_field - system.codec.project_gripper_field(noise_field)
                eval_noise_projection_sse += float(noise_null.float().square().sum().cpu())
                eval_noise_projection_count += int(noise_null.numel())
                pred_field = pred_pack["physical_action"][..., 2 * ad : 2 * ad + gf]
                pred_null = pred_field - system.codec.project_gripper_field(pred_field)
                eval_field_null_sse += float(pred_null.float().square().sum().cpu())
                eval_field_energy += float(pred_field.float().square().sum().cpu())
                eval_field_null_count += int(pred_null.numel())
            if system.codec.uses_arm_manifold:
                ad = int(system.policy_config.arm_dim)
                action_state = sample["action_state"]
                noise_arm = noise[..., : 2 * ad]
                projected_noise_arm = system.codec.project_arm_field(noise_arm, action_state)
                noise_arm_null = noise_arm - projected_noise_arm
                eval_arm_noise_projection_sse += float(noise_arm_null.float().square().sum().cpu())
                eval_arm_noise_projection_count += int(noise_arm_null.numel())
                pred_arm = pred_pack["physical_action"][..., : 2 * ad]
                projected_pred_arm = system.codec.project_arm_field(pred_arm, action_state)
                pred_arm_null = pred_arm - projected_pred_arm
                eval_arm_field_null_sse += float(pred_arm_null.float().square().sum().cpu())
                eval_arm_field_energy += float(pred_arm.float().square().sum().cpu())
                eval_arm_field_null_count += int(pred_arm_null.numel())
            if contract_eval or run_representation_eval:
                representation_eval_batches += int(representation_eval)
                representation_time = torch.rand(
                    int(sample["policy_action"].shape[0]),
                    generator=generator,
                    device=device,
                    dtype=sample["policy_action"].dtype,
                )
                contract_output = system.flow_training_forward(
                    sample["visual"],
                    sample["history_state"],
                    sample["executed_action_history"],
                    sample["state"],
                    sample["policy_action"],
                    raw_visual=sample.get("raw_visual"),
                    action_state=sample["action_state"],
                    target_visual=sample["target_visual"],
                    future_training_pack=_object_intent_future_training_pack(
                        sample,
                        system=system,
                        require_teacher=True,
                    ),
                    training_noise=noise,
                    training_time=representation_time,
                    proposal_keep=torch.ones_like(representation_time),
                    make_counterfactuals=contract_eval,
                    stop_at_midcut=False,
                )
                if contract_eval:
                    contract_losses = layer_contract_losses(
                        system,
                        sample,
                        contract_output,
                        trainer,
                        enable_future_loss=True,
                    )
                else:
                    contract_losses = flow_losses(
                        system,
                        sample,
                        contract_output,
                        trainer,
                        enable_future_loss=True,
                    )
        if report_mem and memory_reporter.detail:
            memory_reporter.snapshot(
                tag="eval_after_sample",
                phase="eval",
                epoch=epoch,
                batch=batch_index,
                global_step=global_step,
            )
        decoded_pred = decode(action_normalizer, pred_pack["action"])
        pred_rows.append(decoded_pred)
        target_raw = sample["policy_action_raw"].cpu().numpy()
        target_rows.append(target_raw)
        if run_execution_ablation:
            execution_ablation_pred_rows["primary"].append(decoded_pred)
            execution_ablation_target_rows.append(target_raw)
            for name, pack in execution_ablation_packs.items():
                action_value = pack["action"] if isinstance(pack, dict) else pack
                execution_ablation_pred_rows[name].append(decode(action_normalizer, action_value))
        if torch.is_tensor(no_proposal):
            no_proposal_rows.append(decode(action_normalizer, no_proposal))
            no_proposal_target_rows.append(target_raw)
            proposal_ablation_pred_rows.append(decoded_pred)
        current_rows.append(sample["state_raw"].cpu().numpy())
        labels = gripper_event_labels(
            target_raw=sample["policy_action_raw"],
            current_raw=sample["state_raw"],
            gripper_index=system.policy_config.gripper_index,
            threshold=trainer.gripper_event_threshold,
        )
        event_logits_rows.append(pred_pack["event_logits"].detach().float().cpu().numpy())
        event_target_rows.append(labels.cpu().numpy())
        motion_target = arm_motion_labels(
            system,
            sample["policy_action"],
            sample["action_state"],
            trainer.arm_motion_threshold,
        )
        motion_logits_rows.append(pred_pack["motion_logits"].detach().float().cpu().numpy())
        motion_target_rows.append(motion_target.detach().float().cpu().numpy())
        if primary_start_event is not None and primary_end_event is not None:
            primary_sample_seconds += primary_start_event.elapsed_time(primary_end_event) / 1000.0
        if ablation_start_event is not None and ablation_end_event is not None:
            proposal_ablation_seconds += (
                ablation_start_event.elapsed_time(ablation_end_event) / 1000.0
            )
        if contract_losses is not None:
            fixed_contract_metric_keys = {
                "loss",
                "layer_contract",
                "latent",
                "rollout_dynamics",
                "rollout_delta",
                "rollout_contrast",
                "milestone_delta_match",
                "flow_jepa_future_prediction",
                "flow_jepa_stage_prediction",
                "flow_jepa_stage_target_norm",
                "flow_jepa_stage_prediction_norm",
                "flow_jepa_stage_to_window_gate",
                "flow_jepa_stage_to_window_update_norm",
                "flow_jepa_future_raw_delta_loss",
                "flow_jepa_future_reliable_normalized_loss",
                "flow_jepa_future_change_reliability",
                "flow_jepa_future_active_direction_loss",
                "flow_jepa_future_active_composite_loss",
                "flow_jepa_future_direction_floor_min",
            }
            for key, value in contract_losses.items():
                if (
                    key in fixed_contract_metric_keys or key.startswith("flow_jepa_future_horizon_")
                ) and torch.is_tensor(value):
                    contract_metric_sums[key] = contract_metric_sums.get(key, 0.0) + float(
                        value.detach().float().cpu()
                    )
            contract_metric_count += 1
        if report_mem:
            memory_reporter.snapshot(
                tag="eval_batch_end",
                phase="eval",
                epoch=epoch,
                batch=batch_index,
                global_step=global_step,
                print_line=True,
            )
    pred = np.concatenate(pred_rows)
    target = np.concatenate(target_rows)
    current = np.concatenate(current_rows)
    squared = (pred - target) ** 2
    arm_squared = squared[..., :-1]
    gripper_squared = squared[..., -1]
    metrics = {
        "full_mse": float(squared.mean()),
        "full_rmse": float(np.sqrt(squared.mean())),
        "first_rmse": float(np.sqrt(squared[:, 0].mean())),
        "first4_rmse": float(np.sqrt(squared[:, :4].mean())),
        "first8_rmse": float(np.sqrt(squared[:, :8].mean())),
        "tail_rmse": float(np.sqrt(squared[:, 8:].mean()))
        if squared.shape[1] > 8
        else float("nan"),
        "arm_full_rmse": float(np.sqrt(arm_squared.mean())),
        "arm_first_rmse": float(np.sqrt(arm_squared[:, 0].mean())),
        "arm_first4_rmse": float(np.sqrt(arm_squared[:, :4].mean())),
        "arm_first8_rmse": float(np.sqrt(arm_squared[:, :8].mean())),
        "arm_tail_rmse": float(np.sqrt(arm_squared[:, 8:].mean()))
        if arm_squared.shape[1] > 8
        else float("nan"),
        "gripper_full_rmse": float(np.sqrt(gripper_squared.mean())),
        "gripper_first_rmse": float(np.sqrt(gripper_squared[:, 0].mean())),
        "gripper_first4_rmse": float(np.sqrt(gripper_squared[:, :4].mean())),
        "gripper_first8_rmse": float(np.sqrt(gripper_squared[:, :8].mean())),
        "gripper_tail_rmse": float(np.sqrt(gripper_squared[:, 8:].mean()))
        if gripper_squared.shape[1] > 8
        else float("nan"),
    }
    if int(getattr(system.policy_config, "flow_jepa_enabled", 0)):
        band_start = 0
        for band_end in system.policy_config.flow_jepa_action_offsets:
            band_end = int(band_end)
            if band_end <= band_start or band_end > int(squared.shape[1]):
                continue
            label = f"{band_start + 1}_{band_end}"
            metrics[f"action_band_{label}_rmse"] = float(
                np.sqrt(squared[:, band_start:band_end].mean())
            )
            band_start = band_end
    if no_proposal_rows:
        no_proposal = np.concatenate(no_proposal_rows)
        no_proposal_target = np.concatenate(no_proposal_target_rows)
        selected_pred = np.concatenate(proposal_ablation_pred_rows)
        metrics["proposal_utility_mse_gain"] = float(
            ((no_proposal - no_proposal_target) ** 2).mean()
            - ((selected_pred - no_proposal_target) ** 2).mean()
        )
    else:
        metrics["proposal_utility_mse_gain"] = float("nan")
    metrics["eval_batches"] = float(completed_batches)
    metrics["eval_sampling_diagnostic_batches"] = float(sampling_diagnostic_batches)
    metrics["eval_sampling_diagnostic_coverage"] = float(
        sampling_diagnostic_batches / max(completed_batches, 1)
    )
    metrics["eval_proposal_ablation_batches"] = float(proposal_ablation_batches)
    metrics["eval_proposal_ablation_samples"] = float(proposal_ablation_samples)
    metrics["eval_proposal_ablation_coverage"] = float(
        proposal_ablation_batches / max(completed_batches, 1)
    )
    metrics["eval_execution_ablation_batches"] = float(execution_ablation_batches)
    metrics["eval_execution_ablation_coverage"] = float(
        execution_ablation_batches / max(completed_batches, 1)
    )
    metrics["eval_representation_batches"] = float(representation_eval_batches)
    metrics["eval_representation_coverage"] = float(
        representation_eval_batches / max(completed_batches, 1)
    )
    if execution_ablation_target_rows:
        ablation_target = np.concatenate(execution_ablation_target_rows)
        primary_ablation_mse = float(
            (
                (np.concatenate(execution_ablation_pred_rows["primary"]) - ablation_target) ** 2
            ).mean()
        )
        metrics["execution_ablation_primary_full_rmse"] = float(math.sqrt(primary_ablation_mse))
        for name in ("hard", "neutral", "full_capacity", "three_basis_reduction"):
            mode_pred = np.concatenate(execution_ablation_pred_rows[name])
            mode_squared = (mode_pred - ablation_target) ** 2
            metrics[f"execution_ablation_{name}_full_rmse"] = float(np.sqrt(mode_squared.mean()))
            metrics[f"execution_ablation_{name}_tail_rmse"] = (
                float(np.sqrt(mode_squared[:, 8:].mean()))
                if mode_squared.shape[1] > 8
                else float("nan")
            )
            metrics[f"execution_ablation_{name}_mse_gain_vs_primary"] = float(
                primary_ablation_mse - mode_squared.mean()
            )
    eval_wall_seconds = time.perf_counter() - eval_started_at
    metrics["eval_wall_seconds"] = float(eval_wall_seconds)
    metrics["eval_seconds_per_batch"] = float(eval_wall_seconds / max(completed_batches, 1))
    metrics["eval_primary_sample_seconds"] = float(primary_sample_seconds)
    metrics["eval_primary_sample_seconds_per_batch"] = float(
        primary_sample_seconds / max(completed_batches, 1)
    )
    metrics["eval_proposal_ablation_seconds"] = float(proposal_ablation_seconds)
    metrics["eval_proposal_ablation_seconds_per_probe_batch"] = float(
        proposal_ablation_seconds / max(proposal_ablation_batches, 1)
    )
    non_sampling_seconds = max(
        eval_wall_seconds - primary_sample_seconds - proposal_ablation_seconds,
        0.0,
    )
    metrics["eval_non_sampling_seconds"] = float(non_sampling_seconds)
    metrics["eval_non_sampling_seconds_per_batch"] = float(
        non_sampling_seconds / max(completed_batches, 1)
    )
    metrics.update(
        gripper_transition_metrics(
            pred,
            target,
            current,
            gripper_index=system.policy_config.gripper_index,
            threshold=trainer.gripper_event_threshold,
            tolerance=2,
        )
    )
    metrics.update(event_head_metrics(event_logits_rows, event_target_rows))
    metrics.update(motion_head_metrics(motion_logits_rows, motion_target_rows))
    metrics["event_head_minus_decoded_gripper_f1"] = float(
        metrics.get("event_head_f1", 0.0) - metrics.get("gripper_f1", 0.0)
    )
    metrics["event_head_to_decoded_event_count_ratio"] = float(
        metrics.get("event_head_pred_events", 0.0)
        / max(metrics.get("gripper_pred_events", 0.0), 1.0)
    )
    metrics["tail_first_ratio"] = float(metrics["tail_rmse"] / max(metrics["first_rmse"], 1e-8))
    metrics["gripper_event_ratio"] = float(
        metrics.get("gripper_pred_events", 0.0)
        / max(metrics.get("gripper_target_events", 0.0), 1.0)
    )
    metrics["eval_uses_target_action"] = 0.0
    metrics["eval_teacher_forced"] = 0.0
    metrics["eval_stop_at_midcut"] = float(
        _is_contract_stage(trainer) and not _uses_layer_adapter_contract(trainer)
    )
    metrics["eval_layer_adapter_contract"] = float(_uses_layer_adapter_contract(trainer))
    metrics["contract_eval_teacher_forced_action"] = float(contract_eval)
    if system.codec.uses_parseval_gripper_field:
        metrics["eval_gripper_field_projection_mse"] = eval_field_null_sse / max(
            eval_field_null_count, 1
        )
        metrics["eval_gripper_field_null_ratio"] = eval_field_null_sse / max(
            eval_field_energy, 1e-12
        )
        metrics["eval_gripper_noise_projection_mse"] = eval_noise_projection_sse / max(
            eval_noise_projection_count, 1
        )
    if system.codec.uses_arm_manifold:
        metrics["eval_arm_field_projection_mse"] = eval_arm_field_null_sse / max(
            eval_arm_field_null_count, 1
        )
        metrics["eval_arm_field_null_ratio"] = eval_arm_field_null_sse / max(
            eval_arm_field_energy, 1e-12
        )
        metrics["eval_arm_noise_projection_mse"] = eval_arm_noise_projection_sse / max(
            eval_arm_noise_projection_count, 1
        )
    if contract_metric_count:
        for key, value in contract_metric_sums.items():
            metrics[f"contract_{key}"] = value / float(contract_metric_count)
            if key.startswith("flow_jepa_"):
                metrics[key] = value / float(contract_metric_count)
    for key, value in sampling_diagnostic_sums.items():
        metrics[key] = value / float(max(sampling_diagnostic_counts.get(key, 0), 1))
    metrics["eval_sampling_diagnostic_count"] = float(len(sampling_diagnostic_sums))
    return metrics


def _flow_address_action_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    current: np.ndarray,
    *,
    gripper_index: int,
    gripper_event_threshold: float,
    action_offsets: Sequence[int] | None = None,
) -> dict[str, float]:
    squared = (pred - target) ** 2
    arm_squared = squared[..., :-1]
    gripper_squared = squared[..., -1]
    first_rmse = float(np.sqrt(squared[:, 0].mean()))
    tail_rmse = float(np.sqrt(squared[:, 8:].mean())) if squared.shape[1] > 8 else float("nan")
    metrics = {
        "full_rmse": float(np.sqrt(squared.mean())),
        "first_rmse": first_rmse,
        "first8_rmse": float(np.sqrt(squared[:, :8].mean())),
        "tail_rmse": tail_rmse,
        "tail_first_ratio": float(tail_rmse / max(first_rmse, 1e-8)),
        "arm_full_rmse": float(np.sqrt(arm_squared.mean())),
        "gripper_full_rmse": float(np.sqrt(gripper_squared.mean())),
    }
    if action_offsets is not None:
        band_start = 0
        for band_end in action_offsets:
            band_end = int(band_end)
            if band_end <= band_start or band_end > int(squared.shape[1]):
                raise ValueError(
                    "action intervention offsets must be increasing and within the horizon"
                )
            label = f"{band_start + 1}_{band_end}"
            metrics[f"action_band_{label}_rmse"] = float(
                np.sqrt(squared[:, band_start:band_end].mean())
            )
            band_start = band_end
        if band_start != int(squared.shape[1]):
            raise ValueError("action intervention offsets must end at action_horizon")
    gripper = gripper_transition_metrics(
        pred,
        target,
        current,
        gripper_index=gripper_index,
        threshold=gripper_event_threshold,
        tolerance=2,
    )
    for key in (
        "gripper_precision",
        "gripper_recall",
        "gripper_f1",
        "gripper_pred_events",
        "gripper_target_events",
    ):
        metrics[key] = float(gripper[key])
    return metrics


def _paired_bootstrap_interval(
    values: np.ndarray,
    *,
    seed: int,
    reps: int,
) -> dict[str, float | int]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        raise ValueError("paired bootstrap requires at least one sample")
    if reps < 1000:
        raise ValueError("paired bootstrap requires at least 1000 resamples")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, flat.size, size=(reps, flat.size))
    means = flat[indices].mean(axis=1)
    lo, hi = np.quantile(means, (0.025, 0.975))
    return {
        "mean": float(flat.mean()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "samples": int(flat.size),
        "bootstrap_reps": int(reps),
        "seed": int(seed),
    }


def _paired_cluster_bootstrap_interval(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    reps: int,
) -> dict[str, float | int | str]:
    """Paired interval with trajectory/episode, not window, as the IID unit."""

    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    group_rows = np.asarray(groups).reshape(-1)
    if flat.size == 0 or group_rows.shape != flat.shape:
        raise ValueError("cluster bootstrap requires aligned non-empty values/groups")
    if reps < 1000:
        raise ValueError("cluster bootstrap requires at least 1000 resamples")
    unique, inverse = np.unique(group_rows, return_inverse=True)
    cluster_sums = np.bincount(inverse, weights=flat, minlength=len(unique))
    cluster_counts = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(unique), size=(reps, len(unique)))
    means = cluster_sums[indices].sum(axis=1) / cluster_counts[indices].sum(axis=1)
    lo, hi = np.quantile(means, (0.025, 0.975))
    return {
        "mean": float(flat.mean()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "samples": int(flat.size),
        "clusters": int(len(unique)),
        "bootstrap_reps": int(reps),
        "seed": int(seed),
        "bootstrap_unit": "episode",
    }


@torch.no_grad()
def evaluate_flow_address_intervention(
    *,
    system: V39PolicySystem,
    loader: DataLoader,
    conditioner: RDT2Conditioner,
    device: torch.device,
    dtype: torch.dtype,
    camera_names: Sequence[str],
    action_normalizer: ArrayNormalizer,
    trainer: V39PolicyTrainerConfig,
    intervention_batches: int,
    max_batches: int = 0,
    bootstrap_reps: int = 2000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Paired action probe that changes raw-reader coordinates only.

    This applies to V98 and all later anti-collapse contracts.  The shuffled
    control is deliberately spatial rather than a one-sample batch roll, so
    stride-1 neighbouring windows cannot provide an almost identical donor.
    """

    if intervention_batches <= 0:
        raise ValueError("intervention_batches must be positive")
    if not int(getattr(system.policy_config, "flow_jepa_raw_image_enabled", 0)):
        raise ValueError("V98 flow-address probe requires raw-image Flow-JEPA")
    encoder = getattr(system.planner, "flow_dino_evidence", None)
    if encoder is None or not hasattr(encoder, "set_raw_address_eval_intervention"):
        raise RuntimeError("Flow-DINO encoder lacks the transient address intervention")

    planned_batches = len(loader)
    if max_batches:
        planned_batches = min(planned_batches, int(max_batches))
    if planned_batches <= 0:
        raise ValueError("validation loader is empty")
    budget = min(int(intervention_batches), planned_batches)
    if budget == 1:
        selected_indices = {1 + (planned_batches - 1) // 2}
    else:
        selected_indices = {
            1 + round(index * (planned_batches - 1) / float(budget - 1)) for index in range(budget)
        }

    system.eval()
    predictions: dict[str, list[np.ndarray]] = {
        mode: [] for mode in ("baseline", "zero", "shuffle")
    }
    target_rows: list[np.ndarray] = []
    current_rows: list[np.ndarray] = []
    episode_rows: list[np.ndarray] = []
    representation_sums: dict[str, float] = {}
    representation_weight = 0
    finished_batches = 0
    intervention_samples = 0
    baseline_identity_max_abs_delta = 0.0
    verified_ordinary_baseline = False
    spatial_shuffle_fallback_batches = 0

    for batch_index, batch in enumerate(loader, start=1):
        if batch_index > planned_batches:
            break
        if batch_index not in selected_indices:
            continue
        sample = prepare_v39_policy_sample(
            batch,
            conditioner=conditioner,
            system=system,
            camera_names=camera_names,
            device=device,
            dtype=dtype,
            include_target_visual=False,
        )
        sample_count = int(sample["policy_action"].shape[0])
        generator = torch.Generator(device=device)
        generator.manual_seed(37237 + batch_index)
        noise = system.codec.sample_noise(
            sample_count,
            generator=generator,
            device=device,
            dtype=sample["visual"].dtype,
            action_state=sample["action_state"],
        )
        stop_midcut_eval = _is_contract_stage(trainer) and not _uses_layer_adapter_contract(trainer)

        if not verified_ordinary_baseline:
            encoder.clear_raw_address_eval_intervention()
            with autocast_context(device, dtype):
                ordinary = system.sample(
                    sample["visual"],
                    sample["history_state"],
                    sample["executed_action_history"],
                    sample["state"],
                    raw_visual=sample.get("raw_visual"),
                    action_state=sample["action_state"],
                    steps=trainer.eval_inference_steps,
                    noise=noise,
                    use_proposal=True,
                    stop_at_midcut=stop_midcut_eval,
                    collect_diagnostics=False,
                )
            if not torch.is_tensor(ordinary):
                raise TypeError("ordinary V98 baseline did not return an action tensor")
        else:
            ordinary = None

        for mode, output_name in (
            ("none", "baseline"),
            ("zero", "zero"),
            ("spatial_shuffle", "shuffle"),
        ):
            encoder.set_raw_address_eval_intervention(mode)
            try:
                with autocast_context(device, dtype):
                    action = system.sample(
                        sample["visual"],
                        sample["history_state"],
                        sample["executed_action_history"],
                        sample["state"],
                        raw_visual=sample.get("raw_visual"),
                        action_state=sample["action_state"],
                        steps=trainer.eval_inference_steps,
                        noise=noise,
                        use_proposal=True,
                        stop_at_midcut=stop_midcut_eval,
                        collect_diagnostics=False,
                    )
                if not torch.is_tensor(action):
                    raise TypeError("V98 address intervention did not return an action tensor")
                reader_metrics = encoder.raw_address_eval_metrics()
            finally:
                encoder.clear_raw_address_eval_intervention()
            if output_name == "baseline":
                for key, value in reader_metrics.items():
                    if key == "flow_jepa_raw_address_intervention_code":
                        continue
                    representation_sums[key] = (
                        representation_sums.get(key, 0.0) + float(value) * sample_count
                    )
                representation_weight += sample_count
            if (
                mode == "spatial_shuffle"
                and reader_metrics.get("flow_jepa_raw_address_shuffle_spatial_fallback", 0.0) > 0.5
            ):
                spatial_shuffle_fallback_batches += 1
            if ordinary is not None and output_name == "baseline":
                baseline_identity_max_abs_delta = max(
                    baseline_identity_max_abs_delta,
                    float((ordinary - action).detach().float().abs().max().cpu()),
                )
                verified_ordinary_baseline = True
            predictions[output_name].append(decode(action_normalizer, action))

        target_rows.append(sample["policy_action_raw"].cpu().numpy())
        current_rows.append(sample["state_raw"].cpu().numpy())
        episode = batch.get("episode_idx")
        if not torch.is_tensor(episode) or int(episode.numel()) != sample_count:
            raise ValueError("flow-address probe requires one episode_idx per sample")
        episode_rows.append(episode.detach().cpu().numpy().reshape(-1))
        finished_batches += 1
        intervention_samples += sample_count

    if finished_batches != len(selected_indices):
        raise RuntimeError(
            "flow-address probe finished "
            f"{finished_batches}/{len(selected_indices)} selected batches"
        )
    target = np.concatenate(target_rows)
    current = np.concatenate(current_rows)
    episode_ids = np.concatenate(episode_rows)
    joined = {mode: np.concatenate(rows) for mode, rows in predictions.items()}
    mode_metrics = {
        mode: _flow_address_action_metrics(
            pred,
            target,
            current,
            gripper_index=system.policy_config.gripper_index,
            gripper_event_threshold=trainer.gripper_event_threshold,
            action_offsets=system.policy_config.flow_jepa_action_offsets,
        )
        for mode, pred in joined.items()
    }
    baseline_sample_mse = ((joined["baseline"] - target) ** 2).mean(axis=(1, 2))
    zero_sample_mse = ((joined["zero"] - target) ** 2).mean(axis=(1, 2))
    shuffle_sample_mse = ((joined["shuffle"] - target) ** 2).mean(axis=(1, 2))
    zero_delta = zero_sample_mse - baseline_sample_mse
    shuffle_delta = shuffle_sample_mse - baseline_sample_mse
    paired = {
        "zero_action_delta_rmse": float(
            np.sqrt(((joined["zero"] - joined["baseline"]) ** 2).mean())
        ),
        "shuffle_action_delta_rmse": float(
            np.sqrt(((joined["shuffle"] - joined["baseline"]) ** 2).mean())
        ),
        "zero_mse_delta_vs_baseline": float(zero_delta.mean()),
        "shuffle_mse_delta_vs_baseline": float(shuffle_delta.mean()),
        "zero_relative_mse_delta": float(
            zero_delta.mean() / max(float(baseline_sample_mse.mean()), 1e-12)
        ),
        "shuffle_relative_mse_delta": float(
            shuffle_delta.mean() / max(float(baseline_sample_mse.mean()), 1e-12)
        ),
        "zero_mse_delta_ci": _paired_cluster_bootstrap_interval(
            zero_delta,
            episode_ids,
            seed=bootstrap_seed,
            reps=bootstrap_reps,
        ),
        "shuffle_mse_delta_ci": _paired_cluster_bootstrap_interval(
            shuffle_delta,
            episode_ids,
            seed=bootstrap_seed + 1,
            reps=bootstrap_reps,
        ),
        "per_sample_zero_mse_delta": zero_delta.astype(float).tolist(),
        "per_sample_shuffle_mse_delta": shuffle_delta.astype(float).tolist(),
    }
    representation = {
        key: value / float(max(representation_weight, 1))
        for key, value in representation_sums.items()
    }
    if "flow_jepa_raw_seed_reliability" in representation:
        representation["flow_jepa_seed_reliability"] = representation[
            "flow_jepa_raw_seed_reliability"
        ]
    return {
        "schema": "clearvla-flow-address-intervention-v2",
        "planned_batches": int(planned_batches),
        "selected_batch_indices": sorted(selected_indices),
        "finished_intervention_batches": int(finished_batches),
        "intervention_samples": int(intervention_samples),
        "intervention_coverage": float(finished_batches / planned_batches),
        "shuffle_spatial_fallback_batches": int(spatial_shuffle_fallback_batches),
        "patched_baseline_max_abs_delta": float(baseline_identity_max_abs_delta),
        "representation": representation,
        "modes": mode_metrics,
        "paired": paired,
    }


# Compatibility name used by the existing evaluation CLI and V98 research
# scripts.  The implementation itself is version-agnostic from V98 onward.
evaluate_v98_flow_address_intervention = evaluate_flow_address_intervention


def _action_path_paired_metrics(
    *,
    joined: dict[str, np.ndarray],
    target: np.ndarray,
    episode_ids: np.ndarray,
    action_offsets: Sequence[int],
    bootstrap_reps: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    baseline = joined["baseline"]
    baseline_sample_mse = ((baseline - target) ** 2).mean(axis=(1, 2))
    rows: dict[str, Any] = {}
    for mode_index, (mode, prediction) in enumerate(
        (item for item in joined.items() if item[0] != "baseline")
    ):
        sample_mse = ((prediction - target) ** 2).mean(axis=(1, 2))
        mse_delta = sample_mse - baseline_sample_mse
        band_rows: dict[str, Any] = {}
        band_start = 0
        for band_index, band_end_value in enumerate(action_offsets):
            band_end = int(band_end_value)
            label = f"{band_start + 1}_{band_end}"
            baseline_band_mse = (
                (baseline[:, band_start:band_end] - target[:, band_start:band_end]) ** 2
            ).mean(axis=(1, 2))
            mode_band_mse = (
                (prediction[:, band_start:band_end] - target[:, band_start:band_end]) ** 2
            ).mean(axis=(1, 2))
            band_delta = mode_band_mse - baseline_band_mse
            band_rows[label] = {
                "action_delta_rmse": float(
                    np.sqrt(
                        (
                            (prediction[:, band_start:band_end] - baseline[:, band_start:band_end])
                            ** 2
                        ).mean()
                    )
                ),
                "mse_delta_vs_baseline": float(band_delta.mean()),
                "relative_mse_delta": float(
                    band_delta.mean() / max(float(baseline_band_mse.mean()), 1e-12)
                ),
                "mse_delta_ci": _paired_cluster_bootstrap_interval(
                    band_delta,
                    episode_ids,
                    seed=bootstrap_seed + 100 * mode_index + band_index + 1,
                    reps=bootstrap_reps,
                ),
            }
            band_start = band_end
        rows[mode] = {
            "action_delta_rmse": float(np.sqrt(((prediction - baseline) ** 2).mean())),
            "mse_delta_vs_baseline": float(mse_delta.mean()),
            "relative_mse_delta": float(
                mse_delta.mean() / max(float(baseline_sample_mse.mean()), 1e-12)
            ),
            "mse_delta_ci": _paired_cluster_bootstrap_interval(
                mse_delta,
                episode_ids,
                seed=bootstrap_seed + 100 * mode_index,
                reps=bootstrap_reps,
            ),
            "bands": band_rows,
            "per_sample_mse_delta": mse_delta.astype(float).tolist(),
        }
    return rows


def _model_path_boundary_metric_names(mode: str) -> tuple[str, ...]:
    """Return the first-boundary metrics owned by one intervention.

    This is intentionally an allow-list. Natural model quantities such as an
    observed G1 residual contain the word delta but are not differences caused
    by the requested intervention. Treating every such scalar as causal made
    the old acceptance matrix report a pass for unchanged fields.
    """

    normalized = str(mode).strip().lower()
    if normalized.startswith("flow_"):
        return ("flow_jepa_raw_flow_intervention_delta_norm",)
    if normalized.startswith("raw_value_"):
        return (
            "flow_jepa_raw_value_intervention_delta_norm",
            "flow_jepa_dense_raw_value_intervention_delta_norm",
        )
    if normalized.startswith("dino_key_"):
        return ("flow_jepa_dino_key_intervention_delta_norm",)
    if normalized.startswith("source_raw_match_"):
        return ("flow_jepa_source_raw_key_intervention_delta_norm",)
    if normalized.startswith("joint_address_key_"):
        return (
            "flow_jepa_raw_flow_intervention_delta_norm",
            "flow_jepa_dino_key_intervention_delta_norm",
            "flow_jepa_source_raw_key_intervention_delta_norm",
        )
    if normalized.startswith("literal_current_rgb_"):
        return ("flow_jepa_literal_rgb_intervention_delta_norm",)
    if normalized == "current_context_masked":
        return ("flow_jepa_current_context_mask_fraction",)
    if normalized == "goal_zero":
        return ("goal_condition_keep_delta",)
    if normalized == "goal_episode_shuffle":
        return ("goal_input_delta_norm",)
    if normalized == "action_history_zero":
        return ("history_input_delta_norm", "history_condition_keep_delta")
    if normalized == "action_history_condition_zero":
        return ("history_condition_keep_delta",)
    if normalized == "action_history_proposal_zero":
        return ("history_proposal_keep_delta",)
    if normalized == "action_history_proposal_episode_shuffle":
        return ("history_proposal_input_delta_norm",)
    if normalized in {
        "action_history_episode_shuffle",
        "action_history_truncate",
    }:
        return ("history_input_delta_norm",)
    if normalized.startswith("future_effect_"):
        return (
            "future_effect_boundary_delta_norm",
            # Historical V115 probes used the older explicit name.
            "future_effect_intervention_delta_norm",
        )
    if normalized.startswith("intent_"):
        return (
            "grounded_intent_boundary_delta_norm",
            "intent_window_state_delta_norm",
            "intent_window_selector_delta_norm",
            "intent_temporal_delta_norm",
        )
    if normalized in {
        "address_g3_slot_permute",
        "address_g3_slot_mean",
    }:
        return ("grounded_g3_slot_intervention_delta_norm",)
    if normalized.startswith("address_g"):
        return (
            "address_posterior_signature_l2_delta",
            "fine_posterior_signature_l2_delta",
        )
    if normalized.startswith("g") and "_delta_" in normalized:
        stage = normalized.split("_", 1)[0]
        return (f"{stage}_delta_norm", f"{stage}_delta_delta_norm")
    if normalized.startswith("grounding_entry_"):
        return ("grounding_entry_delta_norm",)
    if normalized.startswith("p3_") and "_delta_" in normalized:
        lane = normalized[len("p3_") :].split("_delta_", 1)[0]
        return (f"p3_{lane}_delta_norm",)
    if normalized.startswith("protected_detail_"):
        return ("protected_detail_delta_norm",)
    if normalized in {"policy_zero", "policy_temporal_shuffle"}:
        return ("policy_bank_delta_norm", "policy_workspace_delta_norm")
    if normalized.startswith("world_residual_"):
        return ("world_residual_delta_norm",)
    if normalized.startswith("interval_stage_"):
        return ("interval_stage_intervention_delta_norm",)
    if normalized.startswith("horizon_address_"):
        return ("horizon_address_intervention_delta_norm",)
    if normalized.startswith("w2p_far_context_"):
        return ("w2p_far_context_delta_norm",)
    if normalized.startswith("bottom_far_rollout_"):
        return ("bottom_far_rollout_delta_norm",)
    if normalized.startswith("all_far_context_"):
        return (
            "w2p_far_context_delta_norm",
            "bottom_far_rollout_delta_norm",
        )
    if normalized.startswith("phase_"):
        return ("phase_context_delta_norm",)
    if normalized == "condition_query_zero":
        return ("condition_query_context_delta_norm",)
    if normalized in {
        "address_posterior_uniform",
        "camera_posterior_uniform",
    }:
        return ("address_posterior_l1_delta",)
    if normalized == "fine_offset_zero":
        return ("fine_posterior_l1_delta",)
    if normalized == "camera_swap":
        return ("camera_bank_value_delta_norm",)
    if normalized.startswith("world_address_query_"):
        return ("world_query_input_delta_norm",)
    if normalized.startswith("future_transport_"):
        return ("future_transport_input_delta_norm",)
    if normalized.startswith(
        ("semantic_owner_", "appearance_owner_", "geometry_owner_")
    ):
        return ("address_posterior_signature_l2_delta",)
    if normalized.startswith("p1_appearance_gateway_"):
        return (
            "flow_jepa_typed_p1_appearance_gateway_intervention_delta_norm",
        )
    if normalized.startswith(("p2_rgb_precision_", "p2_detail_precision_")):
        return ("detail_update_signature_l2_delta",)
    if normalized == "p1_zero":
        return ("p1_delta_norm",)
    return ()


def _model_path_acceptance_matrix(
    *,
    joined: dict[str, np.ndarray],
    paired: dict[str, Any],
    verification_counts: dict[str, int],
    boundary_diagnostics: dict[str, dict[str, float]],
    baseline_identity_max_abs_delta: float,
    representation: dict[str, float] | None = None,
    numerical_tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Compile factual model-path gates without inventing a utility threshold."""

    rows: dict[str, Any] = {}
    for mode, paired_row in paired.items():
        diagnostics = boundary_diagnostics.get(mode, {})
        metric_names = _model_path_boundary_metric_names(mode)
        delta_components = {
            key: float(value)
            for key in metric_names
            if (value := diagnostics.get(key)) is not None
        }
        boundary_delta_l2 = math.sqrt(sum(value * value for value in delta_components.values()))
        interval = paired_row["mse_delta_ci"]
        ci_low = float(interval["ci95_low"])
        ci_high = float(interval["ci95_high"])
        utility_direction = (
            "ablation_harmful_path_helpful"
            if ci_low > 0.0
            else "ablation_helpful_path_harmful"
            if ci_high < 0.0
            else "inconclusive"
        )
        action_delta_rmse = float(paired_row["action_delta_rmse"])
        boundary_verified_batches = int(verification_counts.get(mode, 0))
        boundary_changed = bool(
            boundary_verified_batches > 0
            and delta_components
            and boundary_delta_l2 > numerical_tolerance
        )
        rows[mode] = {
            "boundary_verified_batches": boundary_verified_batches,
            "boundary_metric_contract": list(metric_names),
            "boundary_metric_values": delta_components,
            "boundary_contract_observed": bool(delta_components),
            "boundary_delta_l2": float(boundary_delta_l2),
            "boundary_changed": boundary_changed,
            "action_delta_rmse": action_delta_rmse,
            "action_changed": bool(action_delta_rmse > numerical_tolerance),
            "mse_delta_ci95_low": ci_low,
            "mse_delta_ci95_high": ci_high,
            "utility_direction": utility_direction,
        }

    def action_observed(modes: Sequence[str]) -> bool | None:
        available = [rows[mode] for mode in modes if mode in rows]
        if not available:
            return None
        return bool(
            any(
                row["boundary_changed"] and row["action_changed"]
                for row in available
            )
        )

    def boundary_observed(modes: Sequence[str]) -> bool | None:
        available = [rows[mode] for mode in modes if mode in rows]
        if not available:
            return None
        return bool(any(row["boundary_changed"] for row in available))

    far_pairwise: dict[str, float | bool] = {}
    if {
        "w2p_far_context_zero",
        "bottom_far_rollout_zero",
        "all_far_context_zero",
    }.issubset(joined):
        joint = joined["all_far_context_zero"]
        typed_only = joined["w2p_far_context_zero"]
        bottom_only = joined["bottom_far_rollout_zero"]
        joint_vs_typed = float(np.sqrt(((joint - typed_only) ** 2).mean()))
        joint_vs_bottom = float(np.sqrt(((joint - bottom_only) ** 2).mean()))
        far_pairwise = {
            "joint_vs_typed_only_action_delta_rmse": joint_vs_typed,
            "joint_vs_bottom_only_action_delta_rmse": joint_vs_bottom,
            "joint_distinguishable_from_each_single_path": bool(
                joint_vs_typed > numerical_tolerance and joint_vs_bottom > numerical_tolerance
            ),
        }

    spatial_modes = (
        "flow_zero",
        "flow_episode_shuffle",
        "flow_spatial_shuffle",
        "dino_key_spatial_shuffle",
        "source_raw_match_zero",
        "source_raw_match_spatial_shuffle",
        "joint_address_key_spatial_shuffle",
        "address_posterior_uniform",
        "world_address_query_zero",
        "world_address_query_spatial_shuffle",
        "future_transport_neutral",
        "future_transport_spatial_shuffle",
        "address_g1_zero",
        "address_g1_episode_shuffle",
        "address_g2_zero",
        "address_g2_episode_shuffle",
        "address_g3_zero",
        "address_g3_episode_shuffle",
        "address_g3_slot_permute",
        "address_g3_slot_mean",
    )
    detail_modes = (
        "raw_value_zero",
        "raw_value_spatial_shuffle",
        "literal_current_rgb_zero",
        "literal_current_rgb_spatial_shuffle",
        "protected_detail_zero",
        "protected_detail_episode_shuffle",
    )
    representation = {} if representation is None else representation
    slot_count = representation.get("flow_jepa_address_slot_count")
    slot_center_distance = representation.get("flow_jepa_address_slot_pair_distance_normalized")
    slot_posterior_distance = representation.get("flow_jepa_address_slot_posterior_hellinger")
    slot_effective_count = representation.get("flow_jepa_address_policy_slot_effective_count")
    slot_query_variation = representation.get("flow_jepa_address_policy_slot_query_variation")
    slot_observed = all(
        value is not None
        for value in (
            slot_count,
            slot_center_distance,
            slot_posterior_distance,
            slot_effective_count,
            slot_query_variation,
        )
    )
    address_slot_structure = {
        "observed": bool(slot_observed),
        "configured_slot_count": (None if slot_count is None else float(slot_count)),
        "coarse_center_pair_distance_normalized": (
            None if slot_center_distance is None else float(slot_center_distance)
        ),
        "coarse_posterior_pair_hellinger": (
            None if slot_posterior_distance is None else float(slot_posterior_distance)
        ),
        "policy_slot_effective_count": (
            None if slot_effective_count is None else float(slot_effective_count)
        ),
        "policy_slot_query_variation": (
            None if slot_query_variation is None else float(slot_query_variation)
        ),
        # These are numerical-identity checks, not practical utility
        # thresholds. They prevent a nominal M-slot tensor whose hypotheses
        # are exact copies from being reported as a working multi-slot path.
        "coarse_centers_numerically_distinct": (
            None
            if slot_center_distance is None
            else bool(float(slot_center_distance) > numerical_tolerance)
        ),
        "coarse_posteriors_numerically_distinct": (
            None
            if slot_posterior_distance is None
            else bool(float(slot_posterior_distance) > numerical_tolerance)
        ),
        "policy_uses_multiple_slots_numerically": (
            None
            if slot_effective_count is None
            else bool(float(slot_effective_count) > 1.0 + numerical_tolerance)
        ),
        "policy_slot_route_varies_by_query": (
            None
            if slot_query_variation is None
            else bool(float(slot_query_variation) > numerical_tolerance)
        ),
    }
    typed_route_specs = {
        "ground_to_world": {
            "source_effective_count": ("attnres_ground_to_world_source_effective_count"),
            "anchor_route_std": ("attnres_ground_to_world_anchor_route_std"),
            "camera_route_std": ("attnres_ground_to_world_camera_route_std"),
        },
        "world_to_policy": {
            "source_effective_count": ("attnres_world_to_policy_source_effective_count"),
            "horizon_route_std": ("attnres_world_to_policy_horizon_route_std"),
            "basis_route_std": ("attnres_world_to_policy_basis_route_std"),
        },
        "policy_to_mmdit": {
            "source_effective_count": ("evidence_policy_delta_attnres_source_effective_count"),
            "horizon_route_std": ("evidence_policy_delta_attnres_horizon_route_std"),
        },
        "protected_detail_basis": {
            "source_effective_count": ("evidence_protected_detail_basis_source_effective_count"),
            "horizon_route_std": ("evidence_protected_detail_basis_horizon_route_std"),
        },
    }
    typed_route_structure: dict[str, Any] = {}
    for route_name, fields in typed_route_specs.items():
        observed_values = {
            name: representation.get(metric_name) for name, metric_name in fields.items()
        }
        observed = all(value is not None for value in observed_values.values())
        axis_values = [
            float(value)
            for name, value in observed_values.items()
            if name.endswith("_route_std") and value is not None
        ]
        effective_count = observed_values["source_effective_count"]
        typed_route_structure[route_name] = {
            "observed": bool(observed),
            **{
                name: None if value is None else float(value)
                for name, value in observed_values.items()
            },
            "uses_multiple_sources_numerically": (
                None
                if effective_count is None
                else bool(float(effective_count) > 1.0 + numerical_tolerance)
            ),
            "query_axes_vary_numerically": (
                None
                if not axis_values
                else bool(all(value > numerical_tolerance for value in axis_values))
            ),
        }
    typed_policy_plan_lanes = {
        lane: {
            "boundary_changed": boundary_observed(
                (
                    f"p3_{lane}_delta_zero",
                    f"p3_{lane}_delta_episode_shuffle",
                )
            ),
            "reaches_action": action_observed(
                (
                    f"p3_{lane}_delta_zero",
                    f"p3_{lane}_delta_episode_shuffle",
                )
            ),
        }
        for lane in ("precision", "effect", "temporal", "terminal")
    }
    return {
        "numerical_tolerance": float(numerical_tolerance),
        "replay": {
            "baseline_max_abs_delta": float(baseline_identity_max_abs_delta),
            "numerically_identical": bool(baseline_identity_max_abs_delta <= numerical_tolerance),
        },
        "aggregate": {
            "spatial_boundary_changed": boundary_observed(spatial_modes),
            "spatial_path_reaches_action": action_observed(spatial_modes),
            "detail_boundary_changed": boundary_observed(detail_modes),
            "detail_path_reaches_action": action_observed(detail_modes),
            "goal_path_reaches_action": action_observed(("goal_zero", "goal_episode_shuffle")),
            "history_path_reaches_action": action_observed(
                (
                    "action_history_zero",
                    "action_history_condition_zero",
                    "action_history_proposal_zero",
                    "action_history_proposal_episode_shuffle",
                    "action_history_episode_shuffle",
                    "action_history_truncate",
                )
            ),
            "history_condition_boundary_changed": boundary_observed(
                ("action_history_condition_zero",)
            ),
            "history_condition_path_reaches_action": action_observed(
                ("action_history_condition_zero",)
            ),
            "history_proposal_boundary_changed": boundary_observed(
                (
                    "action_history_proposal_zero",
                    "action_history_proposal_episode_shuffle",
                )
            ),
            "history_proposal_path_reaches_action": action_observed(
                (
                    "action_history_proposal_zero",
                    "action_history_proposal_episode_shuffle",
                )
            ),
            "phase_path_reaches_action": action_observed(
                (
                    "phase_belief_zero",
                    "phase_belief_episode_shuffle",
                    "condition_query_zero",
                )
            ),
            "online_horizon_address_boundary_changed": boundary_observed(
                (
                    "horizon_address_zero",
                    "horizon_address_episode_shuffle",
                )
            ),
            "online_horizon_address_reaches_action": action_observed(
                (
                    "horizon_address_zero",
                    "horizon_address_episode_shuffle",
                )
            ),
            "progressive_grounding_boundary_changed": boundary_observed(
                (
                    "address_g1_zero",
                    "address_g1_episode_shuffle",
                    "address_g2_zero",
                    "address_g2_episode_shuffle",
                    "address_g3_zero",
                    "address_g3_episode_shuffle",
                    "address_g3_slot_permute",
                    "address_g3_slot_mean",
                )
            ),
            "progressive_grounding_reaches_action": action_observed(
                (
                    "address_g1_zero",
                    "address_g1_episode_shuffle",
                    "address_g2_zero",
                    "address_g2_episode_shuffle",
                    "address_g3_zero",
                    "address_g3_episode_shuffle",
                    "address_g3_slot_permute",
                    "address_g3_slot_mean",
                )
            ),
            "interval_stage_boundary_changed": boundary_observed(
                (
                    "interval_stage_zero",
                    "interval_stage_episode_shuffle",
                )
            ),
            "interval_stage_reaches_action": action_observed(
                (
                    "interval_stage_zero",
                    "interval_stage_episode_shuffle",
                )
            ),
            "future_effect_boundary_changed": boundary_observed(
                (
                    "future_effect_zero",
                    "future_effect_spatial_shuffle",
                )
            ),
            "future_effect_reaches_action": action_observed(
                (
                    "future_effect_zero",
                    "future_effect_spatial_shuffle",
                )
            ),
            "typed_policy_plan_boundary_changed": boundary_observed(
                tuple(mode for mode in rows if mode.startswith("p3_") and "_delta_" in mode)
            ),
            "typed_policy_plan_reaches_action": action_observed(
                tuple(mode for mode in rows if mode.startswith("p3_") and "_delta_" in mode)
            ),
            "functional_world_route_boundary_changed": boundary_observed(
                tuple(
                    mode for mode in rows if mode.startswith("functional_w") and "_route_" in mode
                )
            ),
            "functional_world_route_reaches_action": action_observed(
                tuple(
                    mode for mode in rows if mode.startswith("functional_w") and "_route_" in mode
                )
            ),
            "p1_appearance_gateway_boundary_changed": boundary_observed(
                (
                    "p1_appearance_gateway_zero",
                    "p1_appearance_gateway_spatial_shuffle",
                )
            ),
            "p1_appearance_gateway_reaches_action": action_observed(
                (
                    "p1_appearance_gateway_zero",
                    "p1_appearance_gateway_spatial_shuffle",
                )
            ),
            "current_context_mask_boundary_changed": boundary_observed(("current_context_masked",)),
            "current_context_mask_reaches_action": action_observed(("current_context_masked",)),
        },
        "address_slot_structure": address_slot_structure,
        "typed_route_structure": typed_route_structure,
        "typed_policy_plan_lanes": typed_policy_plan_lanes,
        "long_horizon_pairwise": far_pairwise,
        "modes": rows,
        "interpretation": (
            "boundary/action booleans test causal accessibility only; utility "
            "requires the paired error interval and is not inferred from a "
            "nonzero action change"
        ),
    }


def _evenly_spaced_probe_indices(
    candidates: Sequence[int],
    count: int,
) -> list[int]:
    """Choose deterministic, approximately evenly spaced values."""

    values = [int(value) for value in candidates]
    if count <= 0 or not values:
        return []
    if count >= len(values):
        return values
    if count == 1:
        return [values[(len(values) - 1) // 2]]
    selected: list[int] = []
    for index in range(count):
        position = round(index * (len(values) - 1) / float(count - 1))
        value = values[position]
        if value not in selected:
            selected.append(value)
    if len(selected) < count:
        selected.extend(value for value in values if value not in selected)
    return selected[:count]


def _action_path_probe_batch_selection(
    *,
    loader: DataLoader,
    planned_batches: int,
    budget: int,
    gripper_index: int,
    event_threshold: float,
) -> tuple[set[int], dict[str, Any]]:
    """Mix validation-wide coverage with action-only gripper-event coverage."""

    all_indices = list(range(1, int(planned_batches) + 1))
    if budget <= 0 or budget > len(all_indices):
        raise ValueError("action-path probe budget must be within planned batches")

    dataset: Any = loader.dataset
    seen: set[int] = set()
    while (
        not hasattr(dataset, "training_information_signals")
        and hasattr(dataset, "base")
        and id(dataset) not in seen
    ):
        seen.add(id(dataset))
        dataset = dataset.base
    batch_size = getattr(loader, "batch_size", None)
    if batch_size is None:
        batch_size = getattr(getattr(loader, "batch_sampler", None), "batch_size", None)

    event_batches: set[int] = set()
    event_samples_array: np.ndarray | None = None
    signal_available = bool(
        hasattr(dataset, "training_information_signals")
        and isinstance(batch_size, int)
        and batch_size > 0
    )
    if signal_available:
        _, event_samples = dataset.training_information_signals(
            gripper_index=int(gripper_index),
            event_threshold=float(event_threshold),
        )
        limit = min(len(event_samples), int(planned_batches) * int(batch_size))
        event_samples_array = np.asarray(event_samples[:limit], dtype=bool)
        event_batches = {
            1 + int(sample_index) // int(batch_size)
            for sample_index in np.flatnonzero(event_samples_array)
        }
        event_batches = {index for index in event_batches if 1 <= index <= int(planned_batches)}

    episode_batches: dict[int, set[int]] = {}
    event_episode_batches: dict[int, set[int]] = {}
    refs = getattr(dataset, "refs", None)
    if isinstance(refs, Sequence) and isinstance(batch_size, int) and batch_size > 0:
        sample_limit = min(len(refs), int(planned_batches) * int(batch_size))
        for sample_index in range(sample_limit):
            episode_id = getattr(refs[sample_index], "episode_idx", None)
            if episode_id is None:
                episode_batches = {}
                event_episode_batches = {}
                break
            episode_id = int(episode_id)
            batch_index = 1 + sample_index // int(batch_size)
            episode_batches.setdefault(episode_id, set()).add(batch_index)
            if (
                event_samples_array is not None
                and sample_index < len(event_samples_array)
                and bool(event_samples_array[sample_index])
            ):
                event_episode_batches.setdefault(episode_id, set()).add(batch_index)

    event_quota = min((int(budget) + 1) // 2, len(event_batches))
    selected: set[int] = set()
    episode_aware = bool(episode_batches)
    if episode_aware:
        # First spread event batches across trajectories.  Adjacent windows from
        # one episode are not independent evidence, so global spacing alone is
        # insufficient for the episode-cluster bootstrap used below.
        event_episode_ids = [
            episode_id
            for episode_id in sorted(episode_batches)
            if event_episode_batches.get(episode_id)
        ]
        for episode_id in _evenly_spaced_probe_indices(
            event_episode_ids,
            min(event_quota, len(event_episode_ids)),
        ):
            selected.update(
                _evenly_spaced_probe_indices(
                    sorted(event_episode_batches[episode_id]),
                    1,
                )
            )
        if len(selected) < event_quota:
            selected.update(
                _evenly_spaced_probe_indices(
                    sorted(event_batches.difference(selected)),
                    event_quota - len(selected),
                )
            )

        # Guarantee as much episode coverage as the budget permits, then add a
        # second within-episode view in round-robin order.  With the recommended
        # ten batches and five validation trajectories this yields one event and
        # one general batch per trajectory whenever those candidates exist.
        episode_ids = sorted(episode_batches)
        for episode_id in _evenly_spaced_probe_indices(
            episode_ids,
            min(int(budget), len(episode_ids)),
        ):
            if len(selected) >= int(budget):
                break
            if selected.intersection(episode_batches[episode_id]):
                continue
            candidates = sorted(episode_batches[episode_id].difference(selected))
            selected.update(_evenly_spaced_probe_indices(candidates, 1))
        while len(selected) < int(budget):
            changed = False
            for episode_id in episode_ids:
                if len(selected) >= int(budget):
                    break
                candidates = sorted(episode_batches[episode_id].difference(selected))
                if not candidates:
                    continue
                selected.update(_evenly_spaced_probe_indices(candidates, 1))
                changed = True
            if not changed:
                break
    else:
        event_selected = _evenly_spaced_probe_indices(
            sorted(event_batches),
            event_quota,
        )
        selected.update(event_selected)
    if len(selected) < int(budget):
        uniform_pool = [index for index in all_indices if index not in selected]
        selected.update(
            _evenly_spaced_probe_indices(
                uniform_pool,
                int(budget) - len(selected),
            )
        )
    if len(selected) != int(budget):
        raise RuntimeError(
            f"action-path batch selection produced {len(selected)}/{int(budget)} batches"
        )
    selected_event = sorted(selected.intersection(event_batches))
    selected_episode_ids = sorted(
        episode_id
        for episode_id, batches in episode_batches.items()
        if selected.intersection(batches)
    )
    selected_event_episode_ids = sorted(
        episode_id
        for episode_id, batches in event_episode_batches.items()
        if selected.intersection(batches)
    )
    return selected, {
        "selection_strategy": (
            "episode_stratified_uniform_plus_gripper_event"
            if episode_aware and signal_available and event_batches
            else "episode_stratified_uniform_no_event_candidates"
            if episode_aware and signal_available
            else "episode_stratified_uniform_event_signal_unavailable"
            if episode_aware
            else "uniform_plus_gripper_event"
            if signal_available and event_batches
            else "uniform_no_event_candidates"
            if signal_available
            else "uniform_event_signal_unavailable"
        ),
        "event_signal_available": bool(signal_available),
        "event_candidate_batches": int(len(event_batches)),
        "selected_event_batches": int(len(selected_event)),
        "selected_event_batch_indices": selected_event,
        "episode_signal_available": bool(episode_aware),
        "candidate_episode_ids": sorted(episode_batches),
        "selected_episode_ids": selected_episode_ids,
        "selected_episode_count": int(len(selected_episode_ids)),
        "selected_event_episode_ids": selected_event_episode_ids,
    }


def _validate_complete_v103_model_probe_contract(
    cfg: Any,
    trainer: V39PolicyTrainerConfig,
) -> None:
    """Reject checkpoints that only resemble, but are not, the V103 graph."""

    required_flags = {
        "flow_jepa_enabled": 1,
        "flow_jepa_late_bottleneck": 1,
        "flow_jepa_raw_image_enabled": 1,
        "flow_jepa_role_hierarchy": 1,
        "flow_jepa_zero_flow_guard": 1,
        "flow_jepa_complementary_raw_detail": 1,
        "flow_jepa_strict_role_visual_path": 1,
        "flow_jepa_source_aligned_raw_fusion": 1,
        "flow_jepa_policy_workspace_fixed_fusion": 0,
        "flow_jepa_world_anchor_write_only": 0,
        "flow_jepa_late_policy_detail": 1,
        "flow_jepa_policy_workspace_horizon_pool": 1,
        "flow_jepa_soft_address_lattice": 1,
        "flow_jepa_directed_canvas_attention": 1,
        "role_attnres_enabled": 1,
        "role_attnres_ground_to_world": 1,
        "role_attnres_world_to_policy": 1,
        "role_attnres_policy_to_mmdit": 1,
        "layer_contract_adapters": 1,
        "layer_shared_fm_probe": 0,
        "layer_recurrent_consequence": 0,
        "action_history_enabled": 1,
        "action_history_condition_exact_null": 1,
        "action_history_proposal_detach": 0,
        "goal_conditioning_enabled": 1,
        "goal_condition_exact_null": 1,
        "stateless_phase_enabled": 1,
        "flow_jepa_teacher_balanced_target_mask": 0,
        "flow_jepa_predictive_change_contract": 1,
        "latent_cvae_layer_memory": 1,
        "latent_cvae_transition_memory": 1,
        "latent_cvae_layer_detach": 0,
        "latent_cvae_transition_detach": 0,
        "latent_cvae_mmdit_operator_capacity": 1,
        "latent_cvae_mmdit_execution_controller": 1,
        "latent_cvae_mmdit_dynamic_block_route": 1,
        "latent_cvae_mmdit_identity_candidate": 1,
        "latent_cvae_workspace_trajectory_source": 1,
        "latent_cvae_workspace_global_sources": 1,
        "latent_cvae_workspace_layer_source": 1,
        "latent_cvae_workspace_progress_value": 1,
        "latent_cvae_workspace_slot_time_state": 1,
        "latent_cvae_workspace_time_state": 0,
        "latent_cvae_workspace_controller": 0,
        "latent_cvae_hierarchical_workspace": 0,
        "action_consequence_self_condition": 0,
    }
    violations = [
        f"{name}={int(getattr(cfg, name, -1))}, expected {expected}"
        for name, expected in required_flags.items()
        if int(getattr(cfg, name, -1)) != int(expected)
    ]
    v115_schedule = str(getattr(cfg, "flow_jepa_top_role_schedule", "3-3-2")) == "3-2-3"
    expected_values = {
        "depth": 8,
        "flow_jepa_grounding_blocks": 3,
        "flow_jepa_world_blocks": 2 if v115_schedule else 3,
        "flow_jepa_policy_blocks": 3 if v115_schedule else 2,
        "future_anchors": 4,
        "flow_jepa_stage_tokens": 0,
        "latent_cvae_mmdit_depth": 3,
    }
    violations.extend(
        f"{name}={int(getattr(cfg, name, -1))}, expected {expected}"
        for name, expected in expected_values.items()
        if int(getattr(cfg, name, -1)) != int(expected)
    )
    if tuple(int(value) for value in cfg.flow_jepa_window_offsets) != (
        4,
        12,
        24,
        48,
    ):
        violations.append("flow_jepa_window_offsets must be exactly (4,12,24,48)")
    if tuple(int(value) for value in cfg.flow_jepa_action_offsets) != (
        4,
        12,
        24,
    ):
        violations.append("flow_jepa_action_offsets must be exactly (4,12,24)")
    if float(getattr(cfg, "flow_jepa_address_flow_prior_floor", 0.0)) <= 0.0:
        violations.append("flow_jepa_address_flow_prior_floor must be positive")
    if int(getattr(cfg, "flow_jepa_address_slots", 0)) < 2:
        violations.append("flow_jepa_address_slots must preserve at least two hypotheses")
    for name in (
        "flow_jepa_address_route_dim",
        "flow_jepa_address_query_chunk",
        "goal_token_count",
        "action_history_token_count",
        "stateless_phase_count",
    ):
        if int(getattr(cfg, name, 0)) < 1:
            violations.append(f"{name} must be positive")
    for name in (
        "flow_jepa_late_policy_detail_scale",
        "role_attnres_ground_to_world_scale",
        "role_attnres_world_to_policy_scale",
        "role_attnres_policy_to_mmdit_scale",
        "stateless_phase_query_scale",
        "layer_contract_grad_scale",
        "latent_cvae_layer_grad_scale",
    ):
        if float(getattr(cfg, name, 0.0)) <= 0.0:
            violations.append(f"{name} must be positive")
    if str(getattr(cfg, "final_action_decoder", "")) != ("evidence_latent_mmdit_action"):
        violations.append("final_action_decoder must be evidence_latent_mmdit_action")
    if str(getattr(cfg, "latent_cvae_mmdit_dwell_mode", "")) != "learned":
        violations.append("latent_cvae_mmdit_dwell_mode must be learned")
    if str(getattr(cfg, "latent_cvae_mmdit_execution_eval_policy", "")) != "soft":
        violations.append("latent_cvae_mmdit_execution_eval_policy must be soft")
    operator_rank = int(getattr(cfg, "latent_cvae_mmdit_operator_rank", 0))
    operator_groups = int(getattr(cfg, "latent_cvae_mmdit_operator_groups", 0))
    if operator_rank < 1 or operator_groups < 1 or operator_rank % operator_groups:
        violations.append("latent_cvae_mmdit_operator_rank/groups must be positive and divisible")

    if str(trainer.training_stage).strip().lower().replace("-", "_") not in {
        "policy",
        "stage2",
    }:
        violations.append("training_stage must be policy/stage2")
    if not _uses_layer_adapter_contract(trainer):
        violations.append("contract_mode must be layer_adapter")
    if not int(getattr(trainer, "single_stage_role_lr", 0)):
        violations.append("single_stage_role_lr must be enabled")
    if float(getattr(trainer, "flow_jepa_future_loss_weight", 0.0)) <= 0.0:
        violations.append("flow_jepa_future_loss_weight must be positive")
    if str(getattr(trainer, "latent_cvae_mmdit_dwell_mode", "")) != "learned":
        violations.append("trainer latent_cvae_mmdit_dwell_mode must be learned")
    if (
        float(
            getattr(
                trainer,
                "latent_cvae_mmdit_execution_value_loss_weight",
                0.0,
            )
        )
        <= 0.0
    ):
        violations.append("latent_cvae_mmdit_execution_value_loss_weight must be positive")
    if (
        str(getattr(trainer, "flow_jepa_horizon_balance_mode", ""))
        .strip()
        .lower()
        .replace("-", "_")
        != "per_horizon"
    ):
        violations.append("flow_jepa_horizon_balance_mode must be per_horizon")
    for name in (
        "flow_jepa_warp_loss_weight",
        "flow_jepa_identity_advantage_loss_weight",
        "flow_jepa_static_identity_loss_weight",
        "flow_jepa_cycle_loss_weight",
        "flow_jepa_smoothness_loss_weight",
        "flow_jepa_uncertainty_nll_weight",
        "flow_jepa_refinement_sequence_loss_weight",
    ):
        if float(getattr(trainer, name, 0.0)) <= 0.0:
            violations.append(f"{name} must be positive")
    disabled_objectives = (
        "flow_jepa_future_change_loss_weight",
        "flow_jepa_stage_loss_weight",
        "rollout_dynamics_loss_weight",
        "rollout_delta_loss_weight",
        "rollout_contrast_loss_weight",
        "rollout_variance_loss_weight",
        "rollout_norm_loss_weight",
        "rollout_milestone_delta_match_weight",
        "future_latent_loss_weight",
        "action_effect_loss_weight",
        "layer_contract_aux_loss_weight",
    )
    violations.extend(
        f"{name}={float(getattr(trainer, name, 0.0))}, expected 0"
        for name in disabled_objectives
        if abs(float(getattr(trainer, name, 0.0))) > 1e-12
    )
    if violations:
        raise ValueError(
            "V103 model-path probe requires the complete serialized model "
            "contract; " + "; ".join(violations)
        )


def _validate_complete_v104_model_contract(
    cfg: Any,
    trainer: V39PolicyTrainerConfig,
) -> None:
    """Require V103 plus the geometry, residual, and sequential-memory repair."""

    _validate_complete_v103_model_probe_contract(cfg, trainer)
    required_flags = {
        "flow_jepa_bounded_flow_coordinates": 1,
        "flow_jepa_sequential_horizon_memory": 1,
        "role_residual_amplitude_contract": 1,
    }
    violations = [
        f"{name}={int(getattr(cfg, name, -1))}, expected {expected}"
        for name, expected in required_flags.items()
        if int(getattr(cfg, name, -1)) != int(expected)
    ]
    for name in (
        "role_residual_max_update_rms",
        "role_attnres_max_value_rms",
    ):
        if float(getattr(cfg, name, 0.0)) <= 0.0:
            violations.append(f"{name} must be positive")
    if violations:
        raise ValueError(
            "V104 model contract requires the complete V103 graph plus "
            "bounded flow coordinates, sequential horizon memory, and the "
            "role residual amplitude contract; " + "; ".join(violations)
        )


def _validate_complete_v105_model_contract(
    cfg: Any,
    trainer: V39PolicyTrainerConfig,
) -> None:
    """Require V104 plus horizon-address ownership and reliable delta scaling."""

    _validate_complete_v104_model_contract(cfg, trainer)
    violations: list[str] = []
    if int(getattr(cfg, "flow_jepa_horizon_soft_address", -1)) != 1:
        violations.append("flow_jepa_horizon_soft_address must be enabled")
    if not 0.0 < float(getattr(cfg, "flow_jepa_horizon_address_update_scale", 0.0)) <= 1.0:
        violations.append("flow_jepa_horizon_address_update_scale must be in (0,1]")
    if int(getattr(trainer, "flow_jepa_future_reliable_normalization", -1)) != 1:
        violations.append("flow_jepa_future_reliable_normalization must be enabled")
    if float(getattr(trainer, "flow_jepa_horizon_address_loss_weight", 0.0)) <= 0.0:
        violations.append("flow_jepa_horizon_address_loss_weight must be positive")
    if violations:
        raise ValueError(
            "V105 model contract requires the complete V104 graph plus a "
            "horizon-specific observation-only soft address, reliable future "
            "delta normalization, and teacher-only address supervision; " + "; ".join(violations)
        )


def _validate_complete_v106_model_contract(
    cfg: Any,
    trainer: V39PolicyTrainerConfig,
) -> None:
    """Require V105 plus interval ownership and the complete numerical graph."""

    _validate_complete_v105_model_contract(cfg, trainer)
    required_flags = {
        "flow_jepa_interval_stage_delta": 1,
        "flow_jepa_variance_safe_routing": 1,
        "flow_jepa_complete_numerical_contract": 1,
    }
    violations = [
        f"{name}={int(getattr(cfg, name, -1))}, expected {expected}"
        for name, expected in required_flags.items()
        if int(getattr(cfg, name, -1)) != int(expected)
    ]
    expected_boundaries = (4, 8, 16, 32, 48)
    boundaries = tuple(
        int(value)
        for value in getattr(
            cfg,
            "flow_jepa_effective_interval_boundaries",
            getattr(cfg, "flow_jepa_interval_boundaries", ()),
        )
    )
    if boundaries != expected_boundaries:
        violations.append("flow_jepa_interval_boundaries must resolve to exactly (4,8,16,32,48)")
    supports = tuple(
        int(value)
        for value in getattr(
            cfg,
            "flow_jepa_effective_interval_support_offsets",
            getattr(cfg, "flow_jepa_interval_support_offsets", ()),
        )
    )
    expected_supports = (
        4,
        8,
        12,
        16,
        20,
        24,
        28,
        32,
        36,
        40,
        44,
        48,
    )
    if supports != expected_supports:
        violations.append(
            "flow_jepa_interval_support_offsets must resolve to exactly "
            "(4,8,12,16,20,24,28,32,36,40,44,48)"
        )
    minimum_numerical_floors = {
        "flow_jepa_routing_norm_floor": 0.25,
        "flow_jepa_correlation_rms_floor": 0.10,
        "flow_jepa_visibility_transition_fraction": 0.10,
    }
    for name, minimum in minimum_numerical_floors.items():
        value = float(getattr(cfg, name, 0.0))
        if value < minimum or value > 1.0:
            violations.append(f"{name}={value:.6g}, requires [{minimum:.6g},1]")
    for start, end in zip(expected_boundaries[:-1], expected_boundaries[1:]):
        interval_supports = tuple(value for value in supports if start <= value <= end)
        if (
            len(interval_supports) < 2
            or interval_supports[0] != start
            or interval_supports[-1] != end
        ):
            violations.append(
                f"interval [{start},{end}] must include both boundaries and "
                "at least two teacher support observations"
            )
    for name in (
        "flow_jepa_routing_norm_floor",
        "flow_jepa_horizon_value_max_rms",
        "flow_jepa_interval_stage_update_scale",
    ):
        if float(getattr(cfg, name, 0.0)) <= 0.0:
            violations.append(f"{name} must be positive")
    if float(getattr(trainer, "flow_jepa_interval_stage_loss_weight", 0.0)) <= 0.0:
        violations.append("flow_jepa_interval_stage_loss_weight must be positive")
    if violations:
        raise ValueError(
            "V106 model contract requires the complete V105 graph plus a "
            "spatially aligned interval-stage delta at W->P, bounded "
            "variance-safe routing, bounded learned-correlation/visibility/"
            "role normalization, and explicit interval progression supervision; "
            + "; ".join(violations)
        )


def _validate_complete_v107_model_contract(
    cfg: Any,
    trainer: V39PolicyTrainerConfig,
) -> None:
    """Require V106 plus the complete top-to-bottom address/write repair."""

    _validate_complete_v106_model_contract(cfg, trainer)
    required_flags = {
        "flow_jepa_policy_multi_glimpse_address": 1,
        "flow_jepa_horizon_cell_fine_address": 1,
        "flow_jepa_interval_stage_typed_value": 1,
        "role_residual_contract_after_gate": 1,
    }
    violations = [
        f"{name}={int(getattr(cfg, name, -1))}, expected {expected}"
        for name, expected in required_flags.items()
        if int(getattr(cfg, name, -1)) != int(expected)
    ]
    if int(getattr(cfg, "flow_jepa_raw_reader_heads", 0)) < 2:
        violations.append("flow_jepa_raw_reader_heads must be at least two for V107 glimpses")
    if int(getattr(cfg, "flow_jepa_address_query_chunk", 0)) < 1:
        violations.append("flow_jepa_address_query_chunk must be positive")
    if violations:
        raise ValueError(
            "V107 model contract requires the complete V106 graph plus factual "
            "multi-glimpse policy addressing, target-cell-specific horizon "
            "fine addressing, a typed interval-stage W->P value, and the "
            "post-gate role residual write contract; " + "; ".join(violations)
        )


def _validate_complete_v108_model_contract(
    cfg: Any,
    trainer: V39PolicyTrainerConfig,
) -> None:
    """Require V107 plus the single online G3 -> W1 address topology."""

    _validate_complete_v107_model_contract(cfg, trainer)
    if int(getattr(cfg, "flow_jepa_online_horizon_address", -1)) != 1:
        raise ValueError(
            "V108 model contract requires the complete V107 graph plus "
            "flow_jepa_online_horizon_address=1 so the owned address write "
            "precedes W/P/action decoding"
        )


def _validate_complete_v109_model_contract(
    cfg: Any,
    trainer: V39PolicyTrainerConfig,
) -> None:
    """Require V108 ancestry plus typed progressive G1/G2/G3 addressing."""

    _validate_complete_v108_model_contract(cfg, trainer)
    if int(getattr(cfg, "flow_jepa_progressive_grounding_address", -1)) != 1:
        raise ValueError(
            "V109 model contract requires the complete V108 graph plus "
            "flow_jepa_progressive_grounding_address=1 so G1/G2/G3 own "
            "hypothesis alignment, geometric rectification, and canonical "
            "handoff while P retains the first high-resolution value read"
        )


def _validate_complete_v110_model_contract(
    cfg: Any,
    trainer: V39PolicyTrainerConfig,
) -> None:
    """Require V109 plus coordinate-typed current/future raw ownership."""

    _validate_complete_v109_model_contract(cfg, trainer)
    if int(getattr(cfg, "flow_jepa_coordinate_typed_raw_detail", -1)) != 1:
        raise ValueError(
            "V110 model contract requires the complete V109 graph plus "
            "flow_jepa_coordinate_typed_raw_detail=1 so exact current RGB, "
            "typed address evidence, future transport and P1/P2 local "
            "refinement form one deployed path"
        )
    if int(getattr(cfg, "flow_jepa_raw_micro_grid", -1)) != 3:
        raise ValueError("V110 model contract requires flow_jepa_raw_micro_grid=3")


def _validate_complete_v111_model_contract(
    cfg: Any,
    trainer: V39PolicyTrainerConfig,
) -> None:
    """Require V110 plus functional G/W/P evidence ownership."""

    _validate_complete_v110_model_contract(cfg, trainer)
    if int(getattr(cfg, "flow_jepa_structured_ownership_bottleneck", -1)) != 1:
        raise ValueError(
            "V111 model contract requires the complete V110 graph plus "
            "flow_jepa_structured_ownership_bottleneck=1 so public scene state, "
            "typed evidence sidecars, interval innovations and factorized P reads "
            "remain distinct until action-ready local fusion"
        )


def _validate_complete_v112_model_contract(
    cfg: Any,
    trainer: V39PolicyTrainerConfig,
) -> None:
    """Require V111 plus pre-value public/private owner routing."""

    _validate_complete_v111_model_contract(cfg, trainer)
    if int(getattr(cfg, "flow_jepa_pre_value_owner_routing", -1)) != 1:
        raise ValueError(
            "V112 model contract requires the complete V111 graph plus "
            "flow_jepa_pre_value_owner_routing=1 so the explicit public chart, "
            "W1-W3 private owner states, and P1 appearance fine factor form "
            "one deployed pre-value route"
        )
    if float(getattr(cfg, "flow_jepa_pre_value_owner_update_scale", -1.0)) != 0.10:
        raise ValueError("V112 model contract requires flow_jepa_pre_value_owner_update_scale=0.10")


def _validate_complete_v113_model_contract(
    cfg: Any,
    trainer: V39PolicyTrainerConfig,
) -> None:
    """Require functional typed routing across W, P1, P2 and horizons."""

    _validate_complete_v112_model_contract(cfg, trainer)
    if int(getattr(cfg, "flow_jepa_functional_mainline_routing", -1)) != 1:
        raise ValueError(
            "V113 model contract requires "
            "flow_jepa_functional_mainline_routing=1 so typed W owners are "
            "selected before one hidden reconstruction, W appearance is a "
            "mandatory P1 verifier, P2 keeps a protected policy carrier, and "
            "phase/goal/history remain distinct per horizon"
        )


def _validate_complete_v114_model_contract(
    cfg: Any,
    trainer: V39PolicyTrainerConfig,
) -> None:
    """Require shared factual P1 and protected utility/precision P2."""

    _validate_complete_v113_model_contract(cfg, trainer)
    if int(getattr(cfg, "flow_jepa_utility_precision_mainline", -1)) != 1:
        raise ValueError(
            "V114 model contract requires "
            "flow_jepa_utility_precision_mainline=1 so P1 performs one "
            "action-invariant factual read per horizon and the four action "
            "basis tokens consume protected base/precision facts in P2"
        )
    if int(getattr(cfg, "flow_jepa_action_free_world_factual", -1)) != 1:
        raise ValueError(
            "V114 model contract requires "
            "flow_jepa_action_free_world_factual=1 so noisy x_t cannot "
            "re-enter P1 indirectly through W self-attention or dynamics"
        )
    if int(getattr(cfg, "flow_jepa_address_query_batch_budget", -1)) != 32:
        raise ValueError("V114 model contract requires flow_jepa_address_query_batch_budget=32")
    if int(getattr(cfg, "flow_jepa_microgrid_tile", -1)) != 3:
        raise ValueError("V114 model contract requires flow_jepa_microgrid_tile=3")
    if int(getattr(cfg, "flow_jepa_p1_mixed_precision", -1)) != 1:
        raise ValueError(
            "V114 model contract requires FP32 posterior/geometry with "
            "BF16 factual value contraction"
        )
    if int(getattr(cfg, "flow_jepa_checkpoint_min_batch", -1)) != 4:
        raise ValueError("V114 model contract requires flow_jepa_checkpoint_min_batch=4")


def _validate_complete_v115_model_contract(
    cfg: Any,
    trainer: V39PolicyTrainerConfig,
) -> None:
    """Require G-aligned consequences, stateless goal phases and 3-2-3 P3."""

    # V115 retains the V114 model ancestry but intentionally retires V105's
    # fixed-chart horizon-address objective.  Validate every other inherited
    # contract against an audit copy, then require the real trainer to keep
    # that dead auxiliary loss disabled.
    v114_ancestry_trainer = replace(
        trainer,
        flow_jepa_horizon_address_loss_weight=max(
            float(
                getattr(
                    trainer,
                    "flow_jepa_horizon_address_loss_weight",
                    0.0,
                )
            ),
            1e-6,
        ),
    )
    _validate_complete_v114_model_contract(cfg, v114_ancestry_trainer)
    required_flags = {
        "flow_jepa_shared_factual_glimpse_bank": 1,
        "flow_jepa_g_aligned_future_effect": 1,
        "flow_jepa_stateless_goal_phase_machine": 1,
        "flow_jepa_policy_plan_compiler": 1,
    }
    violations = [
        f"{name}={int(getattr(cfg, name, -1))}, expected {expected}"
        for name, expected in required_flags.items()
        if int(getattr(cfg, name, -1)) != int(expected)
    ]
    expected_values = {
        "depth": 8,
        "flow_jepa_grounding_blocks": 3,
        "flow_jepa_world_blocks": 2,
        "flow_jepa_policy_blocks": 3,
        "flow_jepa_raw_reader_heads": 4,
        "future_anchors": 4,
        "stateless_phase_count": 4,
    }
    violations.extend(
        f"{name}={int(getattr(cfg, name, -1))}, expected {expected}"
        for name, expected in expected_values.items()
        if int(getattr(cfg, name, -1)) != int(expected)
    )
    if str(getattr(cfg, "flow_jepa_top_role_schedule", "")) != "3-2-3":
        violations.append("flow_jepa_top_role_schedule must be 3-2-3")
    if (
        abs(
            float(
                getattr(
                    trainer,
                    "flow_jepa_horizon_address_loss_weight",
                    -1.0,
                )
            )
        )
        > 1e-12
    ):
        violations.append(
            "flow_jepa_horizon_address_loss_weight must be 0 because the "
            "legacy fixed-chart W posterior is not part of V115"
        )
    decay = float(getattr(cfg, "flow_jepa_teacher_g_ema_decay", -1.0))
    if not 0.0 <= decay < 1.0:
        violations.append("flow_jepa_teacher_g_ema_decay must be in [0,1)")
    if violations:
        raise ValueError(
            "V115 model contract requires the complete V114 ancestry plus "
            "one G-aligned FutureEffectField, the observable stateless "
            "Goal-Phase program, and a non-generic P3 plan compiler on the "
            "3-2-3 top schedule; " + "; ".join(violations)
        )


def _validate_complete_v116_model_contract(
    cfg: Any,
    trainer: V39PolicyTrainerConfig,
) -> None:
    """Require V115 plus unique supervised W effect and formal flow time."""

    _validate_complete_v115_model_contract(cfg, trainer)
    violations: list[str] = []
    if int(
        getattr(cfg, "flow_jepa_supervised_effect_mainline", -1)
    ) != 1:
        violations.append("flow_jepa_supervised_effect_mainline=1 is required")
    if str(
        getattr(cfg, "flow_matching_time_distribution", "")
    ) != "beta_1_5_1":
        violations.append(
            "flow_matching_time_distribution must be beta_1_5_1"
        )
    if violations:
        raise ValueError(
            "V116 model contract requires the complete V115 graph plus a "
            "fully supervised FutureEffect W->P boundary, separate terminal "
            "execution evidence, four-state phase belief and formal Beta "
            "flow-time sampling; "
            + "; ".join(violations)
        )


def _validate_complete_v117_model_contract(
    cfg: Any,
    trainer: V39PolicyTrainerConfig,
) -> None:
    """Require V116 plus observable intent, three effects, and a real P2 read."""

    _validate_complete_v116_model_contract(cfg, trainer)
    required = {
        "flow_jepa_stateless_intent_controller": 1,
        "flow_jepa_window_effect_bank": 1,
        "flow_jepa_effect_read_in_p2": 1,
        "flow_jepa_future_slots": 3,
    }
    violations = [
        f"{name}={int(getattr(cfg, name, -1))}, expected {expected}"
        for name, expected in required.items()
        if int(getattr(cfg, name, -1)) != int(expected)
    ]
    if int(getattr(cfg, "future_anchors", -1)) != 4:
        violations.append("future_anchors must remain 4 for inherited online JEPA")
    if violations:
        raise ValueError(
            "V117 model contract requires the complete V116 graph plus the "
            "three-block stateless intent controller, near/mid/late window "
            "effect ownership, and a structured read in the actual P2 block; "
            + "; ".join(violations)
        )


def _validate_differential_intent_effect_323_model_contract(
    cfg: Any,
    trainer: V39PolicyTrainerConfig,
) -> None:
    """Validate the capability graph directly, without replaying vXXX ancestry."""

    required_flags = {
        "flow_jepa_enabled": 1,
        "flow_jepa_progressive_grounding_address": 1,
        "flow_jepa_pre_value_owner_routing": 1,
        "flow_jepa_functional_mainline_routing": 1,
        "flow_jepa_shared_factual_glimpse_bank": 1,
        "flow_jepa_g_aligned_future_effect": 1,
        "flow_jepa_stateless_goal_phase_machine": 1,
        "flow_jepa_policy_plan_compiler": 1,
        "flow_jepa_supervised_effect_mainline": 1,
        "flow_jepa_stateless_intent_controller": 1,
        "flow_jepa_window_effect_bank": 1,
        "flow_jepa_effect_read_in_p2": 1,
        "flow_jepa_differential_intent_effect_mainline": 1,
        "flow_jepa_action_free_world_factual": 1,
        "flow_jepa_p1_mixed_precision": 1,
    }
    required_values = {
        "depth": 8,
        "flow_jepa_grounding_blocks": 3,
        "flow_jepa_world_blocks": 2,
        "flow_jepa_policy_blocks": 3,
        "flow_jepa_future_slots": 3,
        "future_anchors": 4,
        "stateless_phase_count": 4,
        "flow_jepa_raw_reader_heads": 4,
        "flow_jepa_address_query_batch_budget": 32,
        "flow_jepa_microgrid_tile": 3,
        "flow_jepa_checkpoint_min_batch": 4,
    }
    violations = [
        f"{name}={int(getattr(cfg, name, -1))}, expected {expected}"
        for name, expected in {**required_flags, **required_values}.items()
        if int(getattr(cfg, name, -1)) != int(expected)
    ]
    if str(getattr(cfg, "flow_jepa_top_role_schedule", "")) != "3-2-3":
        violations.append("flow_jepa_top_role_schedule must be 3-2-3")
    if str(
        getattr(cfg, "flow_matching_time_distribution", "")
    ) != "beta_1_5_1":
        violations.append(
            "flow_matching_time_distribution must be beta_1_5_1"
        )
    if str(
        getattr(cfg, "final_action_decoder", "")
    ) != "evidence_latent_mmdit_action":
        violations.append(
            "final_action_decoder must be evidence_latent_mmdit_action"
        )
    if str(
        getattr(trainer, "training_stage", "")
    ).lower().replace("-", "_") not in {"policy", "stage2"}:
        violations.append(
            "training_stage must be policy/stage2 (single-stage end-to-end)"
        )
    if int(getattr(trainer, "single_stage_role_lr", 0)) != 1:
        violations.append(
            "single_stage_role_lr must be enabled so S/W/P are not inherited "
            "as low-LR probes"
        )
    if abs(
        float(
            getattr(
                trainer,
                "flow_jepa_horizon_address_loss_weight",
                -1.0,
            )
        )
    ) > 1e-12:
        violations.append(
            "legacy fixed-chart horizon-address loss must remain disabled"
        )
    if float(
        getattr(trainer, "flow_jepa_future_loss_weight", 0.0)
    ) <= 0.0:
        violations.append("flow_jepa_future_loss_weight must be positive")
    if float(
        getattr(trainer, "flow_jepa_interval_stage_loss_weight", 0.0)
    ) <= 0.0:
        violations.append(
            "flow_jepa_interval_stage_loss_weight must be positive"
        )
    if violations:
        raise ValueError(
            "differential_intent_effect_323 requires one coherent observable "
            "S / differentiated W / consequence-aware P graph; "
            + "; ".join(violations)
        )


def _validate_grounded_intent_effect_323_model_contract(
    cfg: Any,
    trainer: V39PolicyTrainerConfig,
) -> None:
    """Validate the compact capability manifest and its live graph."""

    GROUNDING_MANIFEST.validate()
    required_flags = {
        "flow_jepa_enabled": 1,
        "flow_jepa_raw_image_enabled": 1,
        "flow_jepa_late_policy_detail": 1,
        "flow_jepa_soft_address_lattice": 1,
        "flow_jepa_progressive_grounding_address": 1,
        "flow_jepa_coordinate_typed_raw_detail": 1,
        "flow_jepa_pre_value_owner_routing": 1,
        "flow_jepa_functional_mainline_routing": 1,
        "flow_jepa_utility_precision_mainline": 1,
        "flow_jepa_shared_factual_glimpse_bank": 1,
        "flow_jepa_g_aligned_future_effect": 1,
        "flow_jepa_stateless_goal_phase_machine": 1,
        "flow_jepa_policy_plan_compiler": 1,
        "flow_jepa_supervised_effect_mainline": 1,
        "flow_jepa_action_free_world_factual": 1,
        "flow_jepa_p1_mixed_precision": 1,
        "flow_jepa_grounded_intent_effect_mainline": 1,
        "goal_conditioning_enabled": 1,
        "action_history_enabled": 1,
    }
    required_disabled = {
        "flow_jepa_stateless_intent_controller": 0,
        "flow_jepa_window_effect_bank": 0,
        "flow_jepa_effect_read_in_p2": 0,
        "flow_jepa_differential_intent_effect_mainline": 0,
    }
    required_values = {
        "depth": 8,
        "flow_jepa_grounding_blocks": 3,
        "flow_jepa_world_blocks": 2,
        "flow_jepa_policy_blocks": 3,
        "flow_jepa_future_slots": 4,
        "future_anchors": 4,
        "flow_jepa_raw_reader_heads": 4,
        "flow_jepa_raw_micro_grid": 3,
    }
    violations = [
        f"{name}={int(getattr(cfg, name, -1))}, expected {expected}"
        for name, expected in {
            **required_flags,
            **required_disabled,
            **required_values,
        }.items()
        if int(getattr(cfg, name, -1)) != int(expected)
    ]
    if str(getattr(cfg, "flow_jepa_top_role_schedule", "")) != "3-2-3":
        violations.append("flow_jepa_top_role_schedule must be 3-2-3")
    intervals = tuple(
        tuple(int(value) for value in interval)
        for interval in getattr(cfg, "flow_jepa_interval_windows", ())
    )
    if intervals != tuple(GROUNDING_MANIFEST.intervals):
        violations.append(
            "flow_jepa interval windows must be "
            "((4,8),(8,16),(16,32),(32,48))"
        )
    if str(
        getattr(cfg, "flow_matching_time_distribution", "")
    ) != "beta_1_5_1":
        violations.append(
            "flow_matching_time_distribution must be beta_1_5_1"
        )
    if str(
        getattr(cfg, "final_action_decoder", "")
    ) != "evidence_latent_mmdit_action":
        violations.append(
            "final_action_decoder must be evidence_latent_mmdit_action"
        )
    if str(
        getattr(trainer, "training_stage", "")
    ).lower().replace("-", "_") not in {"policy", "stage2"}:
        violations.append(
            "training_stage must be policy/stage2 (single-stage end-to-end)"
        )
    if int(getattr(trainer, "single_stage_role_lr", 0)) != 1:
        violations.append(
            "single_stage_role_lr must own the new S/W/P parameters"
        )
    if float(
        getattr(trainer, "flow_jepa_future_loss_weight", 0.0)
    ) <= 0.0:
        violations.append("flow_jepa_future_loss_weight must be positive")
    if float(
        getattr(trainer, "flow_jepa_interval_stage_loss_weight", 0.0)
    ) <= 0.0:
        violations.append(
            "flow_jepa_interval_stage_loss_weight must be positive"
        )
    if abs(
        float(
            getattr(
                trainer,
                "flow_jepa_horizon_address_loss_weight",
                -1.0,
            )
        )
    ) > 1e-12:
        violations.append(
            "legacy fixed-chart horizon-address loss must remain disabled"
        )
    if violations:
        raise ValueError(
            "grounded_intent_effect_323 manifest mismatch; "
            + "; ".join(violations)
        )


def _summarize_current_context_mask_comparison(
    *,
    enabled: bool,
    finished_batches: int,
    intervention_samples: int,
    comparison_batches: int,
    comparison_weight: int,
    metric_sums: dict[str, dict[str, float]],
    boundary_sums: dict[str, float],
) -> dict[str, Any] | None:
    """Finalize the V113-only matched mask audit when it was collected."""

    if not enabled:
        return None
    if (
        comparison_batches != finished_batches
        or comparison_weight != intervention_samples
    ):
        raise RuntimeError(
            "current-context mask comparison did not cover every selected "
            "V113 probe batch"
        )
    denominator = float(max(comparison_weight, 1))
    averaged_modes = {
        mode: {
            key: value / denominator
            for key, value in sorted(values.items())
        }
        for mode, values in metric_sums.items()
    }
    shared_metric_keys = set(averaged_modes["unmasked"]).intersection(
        averaged_modes["masked"]
    )
    return {
        "schema": "clearvla-v113-current-context-mask-comparison-v1",
        "matched_eval_mode": True,
        "matched_checkpoint": True,
        "matched_action_noise": True,
        "matched_training_time": 0.5,
        "comparison_batches": int(comparison_batches),
        "comparison_samples": int(comparison_weight),
        "modes": averaged_modes,
        "masked_minus_unmasked": {
            key: averaged_modes["masked"][key]
            - averaged_modes["unmasked"][key]
            for key in sorted(shared_metric_keys)
        },
        "masked_boundary": {
            key: value / denominator
            for key, value in sorted(boundary_sums.items())
        },
    }


@torch.no_grad()
def evaluate_v101_action_path_intervention(
    *,
    system: V39PolicySystem,
    loader: DataLoader,
    conditioner: RDT2Conditioner,
    device: torch.device,
    dtype: torch.dtype,
    camera_names: Sequence[str],
    action_normalizer: ArrayNormalizer,
    trainer: V39PolicyTrainerConfig,
    intervention_batches: int,
    max_batches: int = 0,
    bootstrap_reps: int = 2000,
    bootstrap_seed: int = 0,
    intervention_modes: Sequence[str] | None = None,
    require_complete_v103_contract: bool = False,
    require_complete_v104_contract: bool = False,
    require_complete_v105_contract: bool = False,
    require_complete_v106_contract: bool = False,
    require_complete_v107_contract: bool = False,
    require_complete_v108_contract: bool = False,
    require_complete_v109_contract: bool = False,
    require_complete_v110_contract: bool = False,
    require_complete_v111_contract: bool = False,
    require_complete_v112_contract: bool = False,
    require_complete_v113_contract: bool = False,
    require_complete_v114_contract: bool = False,
    require_complete_v115_contract: bool = False,
    require_complete_v116_contract: bool = False,
    require_complete_v117_contract: bool = False,
    require_differential_intent_effect_contract: bool = False,
    require_grounded_intent_effect_contract: bool = False,
) -> dict[str, Any]:
    """Paired deployed-action probe for V101 and typed V103 boundaries.

    Every mode uses the same frozen checkpoint, validation batch, initial
    action noise, inference-step count and execution policy. Non-targeted
    inputs and the proposal are fixed; full-history zero/shuffle/truncate modes
    intentionally alter the history-derived proposal as part of that path,
    while ``action_history_condition_zero`` isolates the attached condition
    lane with the proposal retained. World and policy interventions are
    transient planner state; world modes retain the grounding/position seed
    and alter only the world-block residual. Raw modes separately test
    coordinate use, source/target fine-key matching and the exact
    high-frequency residual after address-bank compilation.
    """

    if intervention_batches <= 0:
        raise ValueError("intervention_batches must be positive")
    cfg = system.policy_config
    formal_contract_count = sum(
        int(value)
        for value in (
            require_complete_v103_contract,
            require_complete_v104_contract,
            require_complete_v105_contract,
            require_complete_v106_contract,
            require_complete_v107_contract,
            require_complete_v108_contract,
            require_complete_v109_contract,
            require_complete_v110_contract,
            require_complete_v111_contract,
            require_complete_v112_contract,
            require_complete_v113_contract,
            require_complete_v114_contract,
            require_complete_v115_contract,
            require_complete_v116_contract,
            require_complete_v117_contract,
            require_differential_intent_effect_contract,
            require_grounded_intent_effect_contract,
        )
    )
    if formal_contract_count > 1:
        raise ValueError("choose exactly one formal model-path contract")
    complete_v111_or_later = bool(
        require_complete_v111_contract
        or require_complete_v112_contract
        or require_complete_v113_contract
        or require_complete_v114_contract
        or require_complete_v115_contract
        or require_complete_v116_contract
        or require_complete_v117_contract
        or require_differential_intent_effect_contract
        or require_grounded_intent_effect_contract
    )
    complete_v113_or_later = bool(
        require_complete_v113_contract
        or require_complete_v114_contract
        or require_complete_v115_contract
        or require_complete_v116_contract
        or require_complete_v117_contract
        or require_differential_intent_effect_contract
        or require_grounded_intent_effect_contract
    )
    matched_current_context_probe = bool(
        complete_v113_or_later
        and not require_grounded_intent_effect_contract
    )
    if require_grounded_intent_effect_contract:
        _validate_grounded_intent_effect_323_model_contract(cfg, trainer)
    elif require_differential_intent_effect_contract:
        _validate_differential_intent_effect_323_model_contract(cfg, trainer)
    elif require_complete_v117_contract:
        _validate_complete_v117_model_contract(cfg, trainer)
    elif require_complete_v116_contract:
        _validate_complete_v116_model_contract(cfg, trainer)
    elif require_complete_v115_contract:
        _validate_complete_v115_model_contract(cfg, trainer)
    elif require_complete_v114_contract:
        _validate_complete_v114_model_contract(cfg, trainer)
    elif require_complete_v113_contract:
        _validate_complete_v113_model_contract(cfg, trainer)
    elif require_complete_v112_contract:
        _validate_complete_v112_model_contract(cfg, trainer)
    elif require_complete_v111_contract:
        _validate_complete_v111_model_contract(cfg, trainer)
    elif require_complete_v110_contract:
        _validate_complete_v110_model_contract(cfg, trainer)
    elif require_complete_v109_contract:
        _validate_complete_v109_model_contract(cfg, trainer)
    elif require_complete_v108_contract:
        _validate_complete_v108_model_contract(cfg, trainer)
    elif require_complete_v107_contract:
        _validate_complete_v107_model_contract(cfg, trainer)
    elif require_complete_v106_contract:
        _validate_complete_v106_model_contract(cfg, trainer)
    elif require_complete_v105_contract:
        _validate_complete_v105_model_contract(cfg, trainer)
    elif require_complete_v104_contract:
        _validate_complete_v104_model_contract(cfg, trainer)
    elif require_complete_v103_contract:
        _validate_complete_v103_model_probe_contract(cfg, trainer)
    strict_role = bool(
        int(getattr(cfg, "flow_jepa_role_hierarchy", 0))
        and int(getattr(cfg, "flow_jepa_strict_role_visual_path", 0))
    )
    fixed_policy_fusion = bool(int(getattr(cfg, "flow_jepa_policy_workspace_fixed_fusion", 0)))
    typed_policy_fusion = bool(
        int(getattr(cfg, "role_attnres_enabled", 0))
        and int(getattr(cfg, "role_attnres_ground_to_world", 0))
        and int(getattr(cfg, "role_attnres_world_to_policy", 0))
        and int(getattr(cfg, "role_attnres_policy_to_mmdit", 0))
        and not fixed_policy_fusion
    )
    if not strict_role or not (fixed_policy_fusion or typed_policy_fusion):
        raise ValueError(
            "model-path probe requires strict role ownership and either the "
            "V101 fixed policy fusion or the typed V103 G->W->P->MMDiT bridges"
        )
    if not int(getattr(cfg, "flow_jepa_raw_image_enabled", 0)):
        raise ValueError("V101 action-path probe requires the raw-image evidence path")
    if not int(getattr(cfg, "flow_jepa_complementary_raw_detail", 0)):
        raise ValueError(
            "V101 action-path probe requires complementary raw detail "
            "for the post-reader intervention"
        )
    planner = system.planner
    encoder = getattr(planner, "flow_dino_evidence", None)
    late_reader = getattr(planner, "late_raw_detail_reader", None)
    soft_address = bool(int(getattr(cfg, "flow_jepa_soft_address_lattice", 0)))
    if not hasattr(planner, "set_action_path_eval_intervention"):
        raise RuntimeError("planner lacks the transient action-path intervention")
    if encoder is None or not hasattr(encoder, "set_raw_address_eval_intervention"):
        raise RuntimeError("Flow-DINO encoder lacks the transient address intervention")
    if soft_address and (
        late_reader is None or not hasattr(late_reader, "set_address_eval_intervention")
    ):
        raise RuntimeError("late raw reader lacks the transient address-posterior intervention")
    if not hasattr(system, "set_condition_eval_intervention"):
        raise RuntimeError("policy system lacks transient condition interventions")

    planned_batches = len(loader)
    if max_batches:
        planned_batches = min(planned_batches, int(max_batches))
    if planned_batches <= 0:
        raise ValueError("validation loader is empty")
    budget = min(int(intervention_batches), planned_batches)
    selected_indices, selection_metadata = _action_path_probe_batch_selection(
        loader=loader,
        planned_batches=planned_batches,
        budget=budget,
        gripper_index=cfg.gripper_index,
        event_threshold=trainer.gripper_event_threshold,
    )
    probe_prefix = (
        "[grounded-intent-effect-323-model-path-probe]"
        if require_grounded_intent_effect_contract
        else "[v118-model-path-probe]"
        if require_differential_intent_effect_contract
        else "[v117-model-path-probe]"
        if require_complete_v117_contract
        else "[v116-model-path-probe]"
        if require_complete_v116_contract
        else "[v115-model-path-probe]"
        if require_complete_v115_contract
        else "[v114-model-path-probe]"
        if require_complete_v114_contract
        else "[v113-model-path-probe]"
        if require_complete_v113_contract
        else "[v112-model-path-probe]"
        if require_complete_v112_contract
        else "[v111-model-path-probe]"
        if require_complete_v111_contract
        else "[v110-model-path-probe]"
        if require_complete_v110_contract
        else "[v109-model-path-probe]"
        if require_complete_v109_contract
        else "[v108-model-path-probe]"
        if require_complete_v108_contract
        else "[v107-model-path-probe]"
        if require_complete_v107_contract
        else "[v106-model-path-probe]"
        if require_complete_v106_contract
        else "[v105-model-path-probe]"
        if require_complete_v105_contract
        else "[v104-model-path-probe]"
        if require_complete_v104_contract
        else "[v103-model-path-probe]"
        if typed_policy_fusion
        else "[v101-action-path-probe]"
    )
    print(
        f"{probe_prefix} "
        f"selection={selection_metadata['selection_strategy']} "
        f"selected={','.join(str(index) for index in sorted(selected_indices))} "
        f"event_batches={selection_metadata['selected_event_batches']}/"
        f"{selection_metadata['event_candidate_batches']} "
        f"episodes={selection_metadata['selected_episode_count']}/"
        f"{len(selection_metadata['candidate_episode_ids'])}",
        flush=True,
    )

    # output name, planner mode, encoder mode, condition mode, posterior mode
    mode_contract: list[tuple[str, str | None, str | None, str | None, str | None]] = [
        (
            "baseline",
            "none",
            "none",
            None,
            "none" if soft_address else None,
        ),
        ("policy_zero", "policy_zero", None, None, None),
        (
            "policy_temporal_shuffle",
            "policy_temporal_shuffle",
            None,
            None,
            None,
        ),
        ("world_residual_zero", "world_residual_zero", None, None, None),
        (
            "world_residual_anchor_shuffle",
            "world_residual_anchor_shuffle",
            None,
            None,
            None,
        ),
        (
            "world_residual_spatial_shuffle",
            "world_residual_spatial_shuffle",
            None,
            None,
            None,
        ),
        (
            "world_residual_spatiotemporal_shuffle",
            "world_residual_spatiotemporal_shuffle",
            None,
            None,
            None,
        ),
        ("flow_zero", None, "zero", None, None),
        ("flow_episode_shuffle", None, "shuffle", None, None),
        ("flow_spatial_shuffle", None, "spatial_shuffle", None, None),
        ("raw_value_zero", None, "detail_zero", None, None),
        (
            "raw_value_spatial_shuffle",
            None,
            "detail_spatial_shuffle",
            None,
            None,
        ),
    ]
    if soft_address:
        mode_contract.extend(
            (
                (
                    "source_raw_match_zero",
                    None,
                    "source_raw_key_zero",
                    None,
                    None,
                ),
                (
                    "source_raw_match_spatial_shuffle",
                    None,
                    "source_raw_key_spatial_shuffle",
                    None,
                    None,
                ),
                (
                    "dino_key_spatial_shuffle",
                    None,
                    "dino_key_spatial_shuffle",
                    None,
                    None,
                ),
                (
                    "joint_address_key_spatial_shuffle",
                    None,
                    "joint_address_key_spatial_shuffle",
                    None,
                    None,
                ),
                (
                    "address_posterior_uniform",
                    None,
                    None,
                    None,
                    "address_posterior_uniform",
                ),
                (
                    "fine_offset_zero",
                    None,
                    None,
                    None,
                    "fine_offset_zero",
                ),
                (
                    "camera_posterior_uniform",
                    None,
                    None,
                    None,
                    "camera_posterior_uniform",
                ),
                ("camera_swap", None, None, None, "camera_swap"),
                (
                    "world_address_query_zero",
                    None,
                    None,
                    None,
                    "world_query_zero",
                ),
                (
                    "world_address_query_spatial_shuffle",
                    None,
                    None,
                    None,
                    "world_query_spatial_shuffle",
                ),
            )
        )
    if require_complete_v110_contract or complete_v111_or_later:
        mode_contract.extend(
            (
                (
                    "literal_current_rgb_zero",
                    None,
                    "literal_rgb_zero",
                    None,
                    None,
                ),
                (
                    "literal_current_rgb_spatial_shuffle",
                    None,
                    "literal_rgb_spatial_shuffle",
                    None,
                    None,
                ),
                (
                    "future_transport_neutral",
                    None,
                    None,
                    None,
                    "future_transport_neutral",
                ),
                (
                    "future_transport_spatial_shuffle",
                    None,
                    None,
                    None,
                    "future_transport_spatial_shuffle",
                ),
            )
        )
    if complete_v111_or_later:
        mode_contract.extend(
            (
                (
                    "semantic_owner_zero",
                    None,
                    None,
                    None,
                    "semantic_owner_zero",
                ),
                (
                    "semantic_owner_shuffle",
                    None,
                    None,
                    None,
                    "semantic_owner_shuffle",
                ),
                (
                    "appearance_owner_zero",
                    None,
                    None,
                    None,
                    "appearance_owner_zero",
                ),
                (
                    "appearance_owner_shuffle",
                    None,
                    None,
                    None,
                    "appearance_owner_shuffle",
                ),
                (
                    "geometry_owner_zero",
                    None,
                    None,
                    None,
                    "geometry_owner_zero",
                ),
                (
                    "geometry_owner_shuffle",
                    None,
                    None,
                    None,
                    "geometry_owner_shuffle",
                ),
            )
        )
    if bool(int(getattr(cfg, "goal_conditioning_enabled", 0))):
        if bool(int(getattr(cfg, "goal_condition_exact_null", 0))):
            mode_contract.append(("goal_zero", None, None, "goal_zero", None))
        mode_contract.append(
            (
                "goal_episode_shuffle",
                None,
                None,
                "goal_batch_shuffle",
                None,
            )
        )
    if bool(int(getattr(cfg, "action_history_enabled", 0))):
        mode_contract.extend(
            (
                ("action_history_zero", None, None, "history_zero", None),
                (
                    "action_history_episode_shuffle",
                    None,
                    None,
                    "history_batch_shuffle",
                    None,
                ),
                (
                    "action_history_truncate",
                    None,
                    None,
                    "history_truncate",
                    None,
                ),
            )
        )
        if bool(int(getattr(cfg, "action_history_condition_exact_null", 0))):
            mode_contract.append(
                (
                    "action_history_condition_zero",
                    None,
                    None,
                    "history_condition_zero",
                    None,
                )
            )
    if (
        require_complete_v115_contract
        or require_complete_v116_contract
        or require_complete_v117_contract
        or require_differential_intent_effect_contract
        or require_grounded_intent_effect_contract
    ):
        # These modes intervene on the single online effect boundary after W
        # has produced it and before P2/P3 consume it.  They do not touch
        # the frozen FutureTeacherTrackPack or an auxiliary prediction head.
        mode_contract.extend(
            (
                (
                    "future_effect_zero",
                    "future_effect_zero",
                    None,
                    None,
                    None,
                ),
                (
                    "future_effect_spatial_shuffle",
                    "future_effect_spatial_shuffle",
                    None,
                    None,
                    None,
                ),
            )
        )
        if (
            require_complete_v116_contract
            or require_complete_v117_contract
            or require_differential_intent_effect_contract
            or require_grounded_intent_effect_contract
        ):
            for component in (
                "current",
                "semantic",
                "transport",
                "reliability",
            ):
                mode_contract.extend(
                    (
                        (
                            f"future_effect_{component}_zero",
                            f"future_effect_{component}_zero",
                            None,
                            None,
                            None,
                        ),
                        (
                            f"future_effect_{component}_spatial_shuffle",
                            f"future_effect_{component}_spatial_shuffle",
                            None,
                            None,
                            None,
                        ),
                    )
                )
            if require_grounded_intent_effect_contract:
                mode_contract.append(
                    (
                        "future_effect_reliability_one",
                        "future_effect_reliability_one",
                        None,
                        None,
                        None,
                    )
                )
        if require_differential_intent_effect_contract:
            for slot_name in ("near", "mid", "late"):
                mode_contract.extend(
                    (
                        (
                            f"future_effect_{slot_name}_zero",
                            f"future_effect_{slot_name}_zero",
                            None,
                            None,
                            None,
                        ),
                        (
                            f"future_effect_{slot_name}_shuffle",
                            f"future_effect_{slot_name}_shuffle",
                            None,
                            None,
                            None,
                        ),
                    )
                )
        if require_grounded_intent_effect_contract:
            for interval_name in (
                "h4_8",
                "h8_16",
                "h16_32",
                "h32_48",
            ):
                mode_contract.extend(
                    (
                        (
                            f"future_effect_{interval_name}_zero",
                            f"future_effect_{interval_name}_zero",
                            None,
                            None,
                            None,
                        ),
                        (
                            f"future_effect_{interval_name}_shuffle",
                            f"future_effect_{interval_name}_shuffle",
                            None,
                            None,
                            None,
                        ),
                    )
                )
    if typed_policy_fusion:
        # V103 attaches the causal proposal to action loss. Probe that lane
        # independently: direct compressed history remains fixed.
        mode_contract.extend(
            (
                (
                    "action_history_proposal_zero",
                    None,
                    None,
                    "history_proposal_zero",
                    None,
                ),
                (
                    "action_history_proposal_episode_shuffle",
                    None,
                    None,
                    "history_proposal_batch_shuffle",
                    None,
                ),
            )
        )
    if bool(int(getattr(cfg, "stateless_phase_enabled", 0))) or (
        require_grounded_intent_effect_contract
    ):
        if not (
            require_differential_intent_effect_contract
            or require_grounded_intent_effect_contract
        ):
            mode_contract.extend(
                (
                    ("phase_belief_zero", "phase_zero", None, None, None),
                    (
                        "phase_belief_episode_shuffle",
                        "phase_batch_shuffle",
                        None,
                        None,
                        None,
                    ),
                    (
                        "condition_query_zero",
                        "condition_query_zero",
                        None,
                        None,
                        None,
                    ),
                )
            )
        if (
            complete_v113_or_later
            and not require_differential_intent_effect_contract
            and not require_grounded_intent_effect_contract
        ):
            mode_contract.extend(
                (
                    (
                        "goal_horizon_context_zero",
                        "goal_context_zero",
                        None,
                        None,
                        None,
                    ),
                    (
                        "history_horizon_context_zero",
                        "history_context_zero",
                        None,
                        None,
                        None,
                    ),
                )
            )
        if require_complete_v117_contract:
            mode_contract.extend(
                (
                    (
                        "intent_window_selector_uniform",
                        "intent_window_selector_uniform",
                        None,
                        None,
                        None,
                    ),
                    (
                        "intent_window_selector_episode_shuffle",
                        "intent_window_selector_shuffle",
                        None,
                        None,
                        None,
                    ),
                    (
                        "intent_temporal_zero",
                        "intent_temporal_zero",
                        None,
                        None,
                        None,
                    ),
                    (
                        "intent_temporal_episode_shuffle",
                        "intent_temporal_shuffle",
                        None,
                        None,
                        None,
                    ),
                )
            )
        if require_differential_intent_effect_contract:
            mode_contract.extend(
                (
                    (
                        "intent_state_zero",
                        "intent_state_zero",
                        None,
                        None,
                        None,
                    ),
                    (
                        "intent_state_episode_shuffle",
                        "intent_state_shuffle",
                        None,
                        None,
                        None,
                    ),
                    (
                        "intent_temporal_zero",
                        "intent_temporal_zero",
                        None,
                        None,
                        None,
                    ),
                    (
                        "intent_temporal_episode_shuffle",
                        "intent_temporal_shuffle",
                        None,
                        None,
                        None,
                    ),
                )
            )
            for window_name in ("near", "mid", "late"):
                mode_contract.extend(
                    (
                        (
                            f"intent_window_{window_name}_zero",
                            f"intent_window_{window_name}_zero",
                            None,
                            None,
                            None,
                        ),
                        (
                            f"intent_window_{window_name}_shuffle",
                            f"intent_window_{window_name}_shuffle",
                            None,
                            None,
                            None,
                        ),
                    )
                )
        if require_grounded_intent_effect_contract:
            mode_contract.extend(
                (
                    (
                        "intent_state_zero",
                        "intent_state_zero",
                        None,
                        None,
                        None,
                    ),
                    (
                        "intent_state_episode_shuffle",
                        "intent_state_shuffle",
                        None,
                        None,
                        None,
                    ),
                    (
                        "intent_goal_set_zero",
                        "intent_goal_set_zero",
                        None,
                        None,
                        None,
                    ),
                    (
                        "intent_goal_set_episode_shuffle",
                        "intent_goal_set_shuffle",
                        None,
                        None,
                        None,
                    ),
                    (
                        "intent_achieved_zero",
                        "intent_achieved_zero",
                        None,
                        None,
                        None,
                    ),
                    (
                        "intent_achieved_episode_shuffle",
                        "intent_achieved_shuffle",
                        None,
                        None,
                        None,
                    ),
                    (
                        "intent_remaining_zero",
                        "intent_remaining_zero",
                        None,
                        None,
                        None,
                    ),
                    (
                        "intent_remaining_episode_shuffle",
                        "intent_remaining_shuffle",
                        None,
                        None,
                        None,
                    ),
                    (
                        "intent_temporal_zero",
                        "intent_temporal_zero",
                        None,
                        None,
                        None,
                    ),
                    (
                        "intent_temporal_episode_shuffle",
                        "intent_temporal_shuffle",
                        None,
                        None,
                        None,
                    ),
                )
            )
            for interval_name in (
                "h4_8",
                "h8_16",
                "h16_32",
                "h32_48",
            ):
                mode_contract.extend(
                    (
                        (
                            f"intent_interval_{interval_name}_zero",
                            f"intent_interval_{interval_name}_zero",
                            None,
                            None,
                            None,
                        ),
                        (
                            f"intent_interval_{interval_name}_shuffle",
                            f"intent_interval_{interval_name}_shuffle",
                            None,
                            None,
                            None,
                        ),
                    )
                )
    progressive_address = bool(int(getattr(cfg, "flow_jepa_progressive_grounding_address", 0)))
    if bool(int(getattr(cfg, "flow_jepa_online_horizon_address", 0))) and not progressive_address:
        mode_contract.extend(
            (
                (
                    "horizon_address_zero",
                    "horizon_address_zero",
                    None,
                    None,
                    None,
                ),
                (
                    "horizon_address_episode_shuffle",
                    "horizon_address_shuffle",
                    None,
                    None,
                    None,
                ),
            )
        )
    if progressive_address:
        for stage in (1, 2, 3):
            mode_contract.extend(
                (
                    (
                        f"address_g{stage}_zero",
                        f"address_g{stage}_zero",
                        None,
                        None,
                        None,
                    ),
                    (
                        f"address_g{stage}_episode_shuffle",
                        f"address_g{stage}_shuffle",
                        None,
                        None,
                        None,
                    ),
                )
            )
        if require_grounded_intent_effect_contract:
            mode_contract.extend(
                (
                    (
                        "address_g3_slot_permute",
                        "address_g3_slot_permute",
                        None,
                        None,
                        None,
                    ),
                    (
                        "address_g3_slot_mean",
                        "address_g3_slot_mean",
                        None,
                        None,
                        None,
                    ),
                )
            )
    if bool(int(getattr(cfg, "flow_jepa_interval_stage_delta", 0))):
        mode_contract.extend(
            (
                (
                    "interval_stage_zero",
                    "interval_stage_zero",
                    None,
                    None,
                    None,
                ),
                (
                    "interval_stage_episode_shuffle",
                    "interval_stage_shuffle",
                    None,
                    None,
                    None,
                ),
            )
        )
    if complete_v113_or_later:
        mode_contract.append(
            (
                "current_context_masked",
                None,
                "current_context_masked",
                None,
                None,
            )
        )
        for depth in range(int(cfg.flow_jepa_world_blocks) + 1):
            mode_contract.extend(
                (
                    (
                        f"functional_w{depth}_route_zero",
                        f"functional_w{depth}_route_zero",
                        None,
                        None,
                        None,
                    ),
                    (
                        f"functional_w{depth}_route_spatial_shuffle",
                        f"functional_w{depth}_route_shuffle",
                        None,
                        None,
                        None,
                    ),
                )
            )
        mode_contract.extend(
            (
                (
                    "p1_appearance_gateway_zero",
                    None,
                    None,
                    None,
                    "p1_appearance_gateway_zero",
                ),
                (
                    "p1_appearance_gateway_spatial_shuffle",
                    None,
                    None,
                    None,
                    "p1_appearance_gateway_spatial_shuffle",
                ),
            )
        )
        for owner in ("semantic", "appearance", "geometry", "horizon"):
            mode_contract.extend(
                (
                    (
                        f"p2_{owner}_zero",
                        None,
                        None,
                        None,
                        f"p2_{owner}_zero",
                    ),
                    (
                        f"p2_{owner}_shuffle",
                        None,
                        None,
                        None,
                        f"p2_{owner}_shuffle",
                    ),
                )
            )
        if int(getattr(cfg, "flow_jepa_utility_precision_mainline", 0)):
            for value_lane in ("rgb_precision", "detail_precision"):
                mode_contract.extend(
                    (
                        (
                            f"p2_{value_lane}_zero",
                            None,
                            None,
                            None,
                            f"p2_{value_lane}_zero",
                        ),
                        (
                            f"p2_{value_lane}_spatial_shuffle",
                            None,
                            None,
                            None,
                            f"p2_{value_lane}_spatial_shuffle",
                        ),
                    )
                )
            for basis_index in range(int(cfg.action_basis_tokens)):
                mode_contract.extend(
                    (
                        (
                            f"p2_basis{basis_index}_zero",
                            None,
                            None,
                            None,
                            f"p2_basis{basis_index}_zero",
                        ),
                        (
                            f"p2_basis{basis_index}_horizon_shuffle",
                            None,
                            None,
                            None,
                            f"p2_basis{basis_index}_horizon_shuffle",
                        ),
                    )
                )
    if typed_policy_fusion:
        grounding_sources = tuple(
            f"g{index + 1}" for index in range(int(cfg.flow_jepa_grounding_blocks))
        )
        if (
            require_complete_v115_contract
            or require_complete_v116_contract
            or require_complete_v117_contract
            or require_differential_intent_effect_contract
            or require_grounded_intent_effect_contract
        ):
            # Generic W/P hidden deltas no longer cross an ownership boundary
            # in V115. Probe only the deployed final-W innovation and P3
            # typed lanes instead of manufacturing interventions on old names.
            if (
                require_differential_intent_effect_contract
                or require_grounded_intent_effect_contract
            ):
                world_sources = ("grounding_entry",)
                policy_sources = [
                    "p3_precision",
                    "p3_temporal",
                ]
            else:
                world_sources = (
                    ("grounding_entry",)
                    if require_complete_v117_contract
                    else (
                        "grounding_entry",
                        "functional_owner_boundary",
                    )
                )
                policy_sources = [
                    "p3_precision",
                    "p3_effect",
                    "p3_temporal",
                ]
                if not (
                    require_complete_v116_contract
                    or require_complete_v117_contract
                ):
                    policy_sources.append("p3_terminal")
            policy_sources = tuple(policy_sources)
        else:
            world_sources = (
                "grounding_entry",
                *(f"w{index + 1}" for index in range(int(cfg.flow_jepa_world_blocks))),
            )
            policy_sources = (
                "world_to_policy",
                *(f"p{index + 1}" for index in range(int(cfg.flow_jepa_policy_blocks))),
            )
        for source in (*grounding_sources, *world_sources, *policy_sources):
            mode_contract.append((f"{source}_delta_zero", f"{source}_zero", None, None, None))
            mode_contract.append(
                (
                    f"{source}_delta_episode_shuffle",
                    f"{source}_shuffle",
                    None,
                    None,
                    None,
                )
            )
        mode_contract.extend(
            (
                (
                    "w2p_far_context_zero",
                    "w2p_far_context_zero",
                    None,
                    None,
                    None,
                ),
                (
                    "w2p_far_context_episode_shuffle",
                    "w2p_far_context_shuffle",
                    None,
                    None,
                    None,
                ),
                (
                    "bottom_far_rollout_zero",
                    "bottom_far_rollout_zero",
                    None,
                    None,
                    None,
                ),
                (
                    "bottom_far_rollout_episode_shuffle",
                    "bottom_far_rollout_shuffle",
                    None,
                    None,
                    None,
                ),
                (
                    "all_far_context_zero",
                    "all_far_context_zero",
                    None,
                    None,
                    None,
                ),
                (
                    "all_far_context_episode_shuffle",
                    "all_far_context_shuffle",
                    None,
                    None,
                    None,
                ),
                (
                    "protected_detail_zero",
                    "protected_detail_zero",
                    None,
                    None,
                    None,
                ),
                (
                    "protected_detail_episode_shuffle",
                    "protected_detail_shuffle",
                    None,
                    None,
                    None,
                ),
            )
        )
    if require_grounded_intent_effect_contract:
        # The grounded capability is a sibling graph, not "V118 plus every
        # ancestral probe."  Keeping inactive W residual/router names here
        # would waste frozen-checkpoint runtime and recreate misleading zero
        # diagnostics.  This allow-list contains only live grounded boundaries
        # plus the unchanged observation/P1/bottom controls.
        grounded_active_modes = {
            "baseline",
            "policy_zero",
            "policy_temporal_shuffle",
            "flow_zero",
            "flow_episode_shuffle",
            "flow_spatial_shuffle",
            "raw_value_zero",
            "raw_value_spatial_shuffle",
            "source_raw_match_zero",
            "source_raw_match_spatial_shuffle",
            "dino_key_spatial_shuffle",
            "joint_address_key_spatial_shuffle",
            "address_posterior_uniform",
            "fine_offset_zero",
            "camera_posterior_uniform",
            "camera_swap",
            "literal_current_rgb_zero",
            "literal_current_rgb_spatial_shuffle",
            "semantic_owner_zero",
            "semantic_owner_shuffle",
            "appearance_owner_zero",
            "appearance_owner_shuffle",
            "geometry_owner_zero",
            "geometry_owner_shuffle",
            "goal_zero",
            "goal_episode_shuffle",
            "action_history_zero",
            "action_history_episode_shuffle",
            "action_history_truncate",
            "action_history_condition_zero",
            "action_history_proposal_zero",
            "action_history_proposal_episode_shuffle",
            "future_effect_zero",
            "future_effect_spatial_shuffle",
            "future_effect_current_zero",
            "future_effect_current_spatial_shuffle",
            "future_effect_semantic_zero",
            "future_effect_semantic_spatial_shuffle",
            "future_effect_transport_zero",
            "future_effect_transport_spatial_shuffle",
            "future_effect_reliability_zero",
            "future_effect_reliability_spatial_shuffle",
            "future_effect_reliability_one",
            "intent_state_zero",
            "intent_state_episode_shuffle",
            "intent_goal_set_zero",
            "intent_goal_set_episode_shuffle",
            "intent_achieved_zero",
            "intent_achieved_episode_shuffle",
            "intent_remaining_zero",
            "intent_remaining_episode_shuffle",
            "intent_temporal_zero",
            "intent_temporal_episode_shuffle",
            "current_context_masked",
            "protected_detail_zero",
            "protected_detail_episode_shuffle",
            "p2_rgb_precision_zero",
            "p2_rgb_precision_spatial_shuffle",
            "p2_detail_precision_zero",
            "p2_detail_precision_spatial_shuffle",
            "p3_precision_delta_zero",
            "p3_precision_delta_episode_shuffle",
            "p3_temporal_delta_zero",
            "p3_temporal_delta_episode_shuffle",
            "grounding_entry_delta_zero",
            "grounding_entry_delta_episode_shuffle",
            "address_g3_slot_permute",
            "address_g3_slot_mean",
        }
        for stage in range(1, int(cfg.flow_jepa_grounding_blocks) + 1):
            grounded_active_modes.update(
                {
                    f"address_g{stage}_zero",
                    f"address_g{stage}_episode_shuffle",
                    f"g{stage}_delta_zero",
                    f"g{stage}_delta_episode_shuffle",
                }
            )
        for interval_name in (
            "h4_8",
            "h8_16",
            "h16_32",
            "h32_48",
        ):
            grounded_active_modes.update(
                {
                    f"future_effect_{interval_name}_zero",
                    f"future_effect_{interval_name}_shuffle",
                    f"intent_interval_{interval_name}_zero",
                    f"intent_interval_{interval_name}_shuffle",
                }
            )
        mode_contract = [
            row for row in mode_contract if row[0] in grounded_active_modes
        ]
        if not mode_contract or mode_contract[0][0] != "baseline":
            raise RuntimeError("grounded probe lost its baseline mode")
    requested_intervention_modes: tuple[str, ...] | None = None
    if intervention_modes is not None:
        requested_intervention_modes = tuple(
            dict.fromkeys(str(mode).strip() for mode in intervention_modes if str(mode).strip())
        )
        available_modes = {output_name for output_name, _, _, _, _ in mode_contract}
        unknown_modes = sorted(set(requested_intervention_modes).difference(available_modes))
        if unknown_modes:
            raise ValueError(
                "unknown model-path intervention modes: "
                + ", ".join(unknown_modes)
                + "; available modes: "
                + ", ".join(sorted(available_modes))
            )
        selected_modes = {"baseline", *requested_intervention_modes}
        mode_contract = [row for row in mode_contract if row[0] in selected_modes]
    system.eval()
    predictions: dict[str, list[np.ndarray]] = {
        output_name: [] for output_name, _, _, _, _ in mode_contract
    }
    target_rows: list[np.ndarray] = []
    current_rows: list[np.ndarray] = []
    episode_rows: list[np.ndarray] = []
    representation_sums: dict[str, float] = {}
    representation_weight = 0
    reader_diagnostic_keys = (
        "flow_jepa_raw_post_reader_detail_selector_residual_norm",
        "flow_jepa_raw_post_reader_detail_value_residual_norm",
        "flow_jepa_raw_post_reader_detail_selector_intervention_delta",
        "flow_jepa_raw_post_reader_detail_value_intervention_delta",
    )
    mode_reader_sums: dict[str, dict[str, float]] = {
        output_name: {} for output_name, _, _, _, _ in mode_contract
    }
    mode_reader_weights = {output_name: 0 for output_name, _, _, _, _ in mode_contract}
    verification_counts = {output_name: 0 for output_name, _, _, _, _ in mode_contract}
    boundary_diagnostics: dict[str, dict[str, float]] = {
        output_name: {} for output_name, _, _, _, _ in mode_contract
    }
    boundary_diagnostic_weights: dict[str, dict[str, int]] = {
        output_name: {} for output_name, _, _, _, _ in mode_contract
    }
    current_mask_metric_sums: dict[str, dict[str, float]] = {
        "unmasked": {},
        "masked": {},
    }
    current_mask_boundary_sums: dict[str, float] = {}
    current_mask_comparison_weight = 0
    current_mask_comparison_batches = 0
    finished_batches = 0
    intervention_samples = 0
    baseline_identity_max_abs_delta = 0.0
    baseline_identity_checked_batches = 0
    verified_ordinary_baseline = False
    replay_tolerance = 1e-8

    for batch_index, batch in enumerate(loader, start=1):
        if batch_index > planned_batches:
            break
        if batch_index not in selected_indices:
            continue
        sample = prepare_v39_policy_sample(
            batch,
            conditioner=conditioner,
            system=system,
            camera_names=camera_names,
            device=device,
            dtype=dtype,
            include_target_visual=matched_current_context_probe,
        )
        sample_count = int(sample["policy_action"].shape[0])
        generator = torch.Generator(device=device)
        generator.manual_seed(37237 + batch_index)
        noise = system.codec.sample_noise(
            sample_count,
            generator=generator,
            device=device,
            dtype=sample["visual"].dtype,
            action_state=sample["action_state"],
        )
        stop_midcut_eval = _is_contract_stage(trainer) and not _uses_layer_adapter_contract(trainer)

        check_ordinary_baseline = bool(
            require_grounded_intent_effect_contract
            or not verified_ordinary_baseline
        )
        if check_ordinary_baseline:
            planner.clear_action_path_eval_intervention()
            encoder.clear_raw_address_eval_intervention()
            system.clear_condition_eval_intervention()
            if late_reader is not None:
                late_reader.clear_address_eval_intervention()
            with autocast_context(device, dtype):
                ordinary = system.sample(
                    sample["visual"],
                    sample["history_state"],
                    sample["executed_action_history"],
                    sample["state"],
                    raw_visual=sample.get("raw_visual"),
                    action_state=sample["action_state"],
                    steps=trainer.eval_inference_steps,
                    noise=noise,
                    use_proposal=True,
                    stop_at_midcut=stop_midcut_eval,
                    collect_diagnostics=False,
                )
            if not torch.is_tensor(ordinary):
                raise TypeError("ordinary V101 baseline did not return an action tensor")
        else:
            ordinary = None

        if matched_current_context_probe:
            target_visual = sample.get("target_visual")
            if not torch.is_tensor(target_visual):
                raise RuntimeError("V113 current-context mask comparison requires future teachers")
            representation_time = torch.full(
                (sample_count,),
                0.5,
                device=device,
                dtype=sample["policy_action"].dtype,
            )
            matched_mask_rows: dict[str, dict[str, Tensor]] = {}
            matched_mask_boundaries: dict[str, dict[str, float]] = {}
            for comparison_name, address_mode in (
                ("unmasked", "none"),
                ("masked", "current_context_masked"),
            ):
                encoder.set_raw_address_eval_intervention(address_mode)
                try:
                    with autocast_context(device, dtype):
                        representation_output = system.flow_training_forward(
                            sample["visual"],
                            sample["history_state"],
                            sample["executed_action_history"],
                            sample["state"],
                            sample["policy_action"],
                            raw_visual=sample.get("raw_visual"),
                            action_state=sample["action_state"],
                            target_visual=target_visual,
                            training_noise=noise,
                            training_time=representation_time,
                            proposal_keep=torch.ones_like(representation_time),
                            make_counterfactuals=False,
                            stop_at_midcut=False,
                        )
                        representation_losses = flow_losses(
                            system,
                            sample,
                            representation_output,
                            trainer,
                            enable_future_loss=True,
                        )
                    matched_mask_rows[comparison_name] = {
                        key: value.detach().float().reshape(())
                        for key, value in representation_losses.items()
                        if (
                            key
                            in {
                                "flow_jepa_future_prediction",
                                "flow_jepa_future_raw_delta_loss",
                                "flow_jepa_future_reliable_normalized_loss",
                                "flow_jepa_future_change_reliability",
                                "flow_jepa_future_active_direction_loss",
                                "flow_jepa_future_active_composite_loss",
                                "flow_jepa_future_direction_floor_min",
                                "flow_jepa_horizon_address",
                            }
                            or key.startswith("flow_jepa_future_horizon_")
                        )
                        and torch.is_tensor(value)
                        and value.numel() == 1
                    }
                    matched_mask_boundaries[comparison_name] = encoder.raw_address_eval_metrics()
                finally:
                    encoder.clear_raw_address_eval_intervention()
            expected_mask_codes = {"unmasked": 0.0, "masked": 12.0}
            for comparison_name, expected_code in expected_mask_codes.items():
                observed_code = matched_mask_boundaries[comparison_name].get(
                    "flow_jepa_raw_address_intervention_code"
                )
                if observed_code is None or not math.isclose(
                    float(observed_code),
                    expected_code,
                    abs_tol=1e-6,
                ):
                    raise RuntimeError(
                        "current-context comparison did not reach the raw "
                        f"reader for {comparison_name!r}"
                    )
            masked_fraction = matched_mask_boundaries["masked"].get(
                "flow_jepa_current_context_mask_fraction",
                0.0,
            )
            unmasked_fraction = matched_mask_boundaries["unmasked"].get(
                "flow_jepa_current_context_mask_fraction",
                0.0,
            )
            if float(masked_fraction) <= float(unmasked_fraction):
                raise RuntimeError(
                    "matched current-context intervention did not increase latest-context masking"
                )
            shared_keys = set(matched_mask_rows["unmasked"]).intersection(
                matched_mask_rows["masked"]
            )
            if not shared_keys:
                raise RuntimeError("current-context comparison collected no shared JEPA metrics")
            for comparison_name in ("unmasked", "masked"):
                for key in shared_keys:
                    current_mask_metric_sums[comparison_name][key] = (
                        current_mask_metric_sums[comparison_name].get(key, 0.0)
                        + float(matched_mask_rows[comparison_name][key].cpu()) * sample_count
                    )
            for key, value in matched_mask_boundaries["masked"].items():
                if key.startswith("flow_jepa_current_context_mask_") and isinstance(
                    value, (int, float)
                ):
                    current_mask_boundary_sums[key] = (
                        current_mask_boundary_sums.get(key, 0.0) + float(value) * sample_count
                    )
            current_mask_comparison_weight += sample_count
            current_mask_comparison_batches += 1

        baseline_address_signature: dict[str, float] | None = None
        baseline_fine_signature: dict[str, float] | None = None
        baseline_detail_signature: dict[str, float] | None = None
        for (
            output_name,
            planner_mode,
            address_mode,
            condition_mode,
            posterior_mode,
        ) in mode_contract:
            if planner_mode is not None:
                planner.set_action_path_eval_intervention(planner_mode)
            else:
                planner.clear_action_path_eval_intervention()
            if address_mode is not None:
                encoder.set_raw_address_eval_intervention(address_mode)
            else:
                encoder.clear_raw_address_eval_intervention()
            if condition_mode is not None:
                system.set_condition_eval_intervention(condition_mode)
            else:
                system.clear_condition_eval_intervention()
            if posterior_mode is not None:
                assert late_reader is not None
                late_reader.set_address_eval_intervention(posterior_mode)
            elif soft_address:
                assert late_reader is not None
                # ``none`` changes no model value; it only captures a compact
                # posterior/detail signature so every upstream intervention
                # can be compared at the same late-read boundary.
                late_reader.set_address_eval_intervention("none")
            elif late_reader is not None:
                late_reader.clear_address_eval_intervention()
            try:
                try:
                    with autocast_context(device, dtype):
                        action = system.sample(
                            sample["visual"],
                            sample["history_state"],
                            sample["executed_action_history"],
                            sample["state"],
                            raw_visual=sample.get("raw_visual"),
                            action_state=sample["action_state"],
                            steps=trainer.eval_inference_steps,
                            noise=noise,
                            use_proposal=True,
                            stop_at_midcut=stop_midcut_eval,
                            collect_diagnostics=False,
                        )
                except Exception as error:
                    raise RuntimeError(
                        "model-path probe failed while evaluating "
                        f"mode={output_name!r} batch_index={batch_index}"
                    ) from error
                if not torch.is_tensor(action):
                    raise TypeError(
                        f"model-path mode {output_name!r} did not return an action tensor"
                    )
                planner_state = planner.action_path_eval_intervention_state()
                reader_metrics = encoder.raw_address_eval_metrics()
                condition_state = system.condition_eval_intervention_state()
                posterior_state = (
                    late_reader.address_eval_intervention_state()
                    if late_reader is not None
                    else {"mode": "disabled", "apply_count": 0}
                )
            finally:
                planner.clear_action_path_eval_intervention()
                encoder.clear_raw_address_eval_intervention()
                system.clear_condition_eval_intervention()
                if late_reader is not None:
                    late_reader.clear_address_eval_intervention()

            if planner_mode not in {None, "none"}:
                apply_count = int(planner_state["apply_count"])
                if apply_count <= 0:
                    raise RuntimeError(
                        f"model-path mode {output_name!r} never reached its boundary"
                    )
                verification_counts[output_name] += 1
            if output_name in {
                "address_g3_slot_permute",
                "address_g3_slot_mean",
            }:
                public_delta = planner_state.get(
                    "grounded_g3_slot_intervention_public_base_delta_norm"
                )
                if public_delta is None or abs(float(public_delta)) > 1e-8:
                    raise RuntimeError(
                        f"model-path mode {output_name!r} changed the protected "
                        "G3 public/P1 control base"
                    )
            if condition_mode not in {None, "none"}:
                apply_count = int(condition_state["apply_count"])
                if apply_count <= 0:
                    raise RuntimeError(f"condition mode {output_name!r} never reached its boundary")
                verification_counts[output_name] += 1
            if posterior_mode not in {None, "none"}:
                apply_count = int(posterior_state["apply_count"])
                if apply_count <= 0:
                    raise RuntimeError(f"posterior mode {output_name!r} never reached its boundary")
                verification_counts[output_name] += 1
            if address_mode is not None:
                code = reader_metrics.get("flow_jepa_raw_address_intervention_code")
                if code is None:
                    raise RuntimeError(f"raw mode {output_name!r} never reached the reader")
                expected_code = {
                    "none": 0.0,
                    "zero": 1.0,
                    "shuffle": 2.0,
                    "spatial_shuffle": 3.0,
                    "detail_zero": 4.0,
                    "detail_spatial_shuffle": 5.0,
                    "dino_key_spatial_shuffle": 6.0,
                    "source_raw_key_zero": 7.0,
                    "source_raw_key_spatial_shuffle": 8.0,
                    "joint_address_key_spatial_shuffle": 9.0,
                    "literal_rgb_zero": 10.0,
                    "literal_rgb_spatial_shuffle": 11.0,
                    "current_context_masked": 12.0,
                }[address_mode]
                if not math.isclose(float(code), expected_code, abs_tol=1e-6):
                    raise RuntimeError(
                        f"raw mode {output_name!r} reported code {code}, expected {expected_code}"
                    )
                verification_counts[output_name] += 1
            for state in (planner_state, condition_state, posterior_state):
                for key, value in state.items():
                    if key in {"mode", "apply_count"}:
                        continue
                    if "_signature_" in key:
                        continue
                    if isinstance(value, (int, float)):
                        boundary_diagnostics[output_name][key] = (
                            boundary_diagnostics[output_name].get(key, 0.0)
                            + float(value) * sample_count
                        )
                        boundary_diagnostic_weights[output_name][key] = (
                            boundary_diagnostic_weights[output_name].get(key, 0) + sample_count
                        )
            if soft_address:
                address_signature = {
                    key: float(value)
                    for key, value in posterior_state.items()
                    if key.startswith("address_posterior_signature_")
                }
                detail_signature = {
                    key: float(value)
                    for key, value in posterior_state.items()
                    if key.startswith("detail_update_signature_")
                }
                fine_signature = {
                    key: float(value)
                    for key, value in posterior_state.items()
                    if key.startswith("fine_posterior_signature_")
                }
            else:
                address_signature = {}
                detail_signature = {}
                fine_signature = {}
            if soft_address and output_name == "baseline":
                baseline_address_signature = address_signature
                baseline_fine_signature = fine_signature
                baseline_detail_signature = detail_signature
            elif soft_address:
                if (
                    not baseline_address_signature
                    or not baseline_fine_signature
                    or not baseline_detail_signature
                ):
                    raise RuntimeError("model-path mode ran before its paired baseline signatures")
                address_signature_delta = math.sqrt(
                    sum(
                        (address_signature[key] - baseline_address_signature[key]) ** 2
                        for key in baseline_address_signature
                    )
                )
                detail_signature_delta = math.sqrt(
                    sum(
                        (detail_signature[key] - baseline_detail_signature[key]) ** 2
                        for key in baseline_detail_signature
                    )
                )
                fine_signature_delta = math.sqrt(
                    sum(
                        (fine_signature[key] - baseline_fine_signature[key]) ** 2
                        for key in baseline_fine_signature
                    )
                )
                for key, value in (
                    (
                        "address_posterior_signature_l2_delta",
                        address_signature_delta,
                    ),
                    (
                        "fine_posterior_signature_l2_delta",
                        fine_signature_delta,
                    ),
                    (
                        "detail_update_signature_l2_delta",
                        detail_signature_delta,
                    ),
                ):
                    boundary_diagnostics[output_name][key] = (
                        boundary_diagnostics[output_name].get(key, 0.0)
                        + float(value) * sample_count
                    )
                    boundary_diagnostic_weights[output_name][key] = (
                        boundary_diagnostic_weights[output_name].get(key, 0) + sample_count
                    )
            for key, value in reader_metrics.items():
                if key.endswith("_intervention_delta_norm") or key.endswith("_intervention_delta"):
                    boundary_diagnostics[output_name][key] = (
                        boundary_diagnostics[output_name].get(key, 0.0)
                        + float(value) * sample_count
                    )
                    boundary_diagnostic_weights[output_name][key] = (
                        boundary_diagnostic_weights[output_name].get(key, 0) + sample_count
                    )
            for key in reader_diagnostic_keys:
                if key not in reader_metrics:
                    continue
                mode_reader_sums[output_name][key] = (
                    mode_reader_sums[output_name].get(key, 0.0)
                    + float(reader_metrics[key]) * sample_count
                )
            if any(key in reader_metrics for key in reader_diagnostic_keys):
                mode_reader_weights[output_name] += sample_count
            if output_name == "baseline":
                baseline_representation = {
                    key: value
                    for key, value in reader_metrics.items()
                    if key != "flow_jepa_raw_address_intervention_code"
                }
                baseline_representation.update(
                    {
                        key: value
                        for key, value in planner_state.items()
                        if (key not in {"mode", "apply_count"} and isinstance(value, (int, float)))
                    }
                )
                baseline_representation.update(
                    {
                        key: value
                        for key, value in posterior_state.items()
                        if (
                            key not in {"mode", "apply_count"}
                            and "_signature_" not in key
                            and isinstance(value, (int, float))
                        )
                    }
                )
                for key, value in baseline_representation.items():
                    if key == "flow_jepa_raw_address_intervention_code":
                        continue
                    representation_sums[key] = (
                        representation_sums.get(key, 0.0) + float(value) * sample_count
                    )
                representation_weight += sample_count
                if ordinary is not None:
                    replay_delta = float(
                        (ordinary - action)
                        .detach()
                        .float()
                        .abs()
                        .max()
                        .cpu()
                    )
                    baseline_identity_max_abs_delta = max(
                        baseline_identity_max_abs_delta,
                        replay_delta,
                    )
                    baseline_identity_checked_batches += 1
                    if (
                        require_grounded_intent_effect_contract
                        and replay_delta > replay_tolerance
                    ):
                        raise RuntimeError(
                            "grounded model-path probe baseline replay changed "
                            "the deployed action on "
                            f"batch_index={batch_index}: "
                            f"max_abs_delta={replay_delta:.3e}, "
                            f"tolerance={replay_tolerance:.1e}. "
                            "The explicit none instrumentation is not "
                            "deployment-equivalent; causal intervention "
                            "results would be invalid."
                        )
                    verified_ordinary_baseline = True
            predictions[output_name].append(decode(action_normalizer, action))

        target_rows.append(sample["policy_action_raw"].cpu().numpy())
        current_rows.append(sample["state_raw"].cpu().numpy())
        episode = batch.get("episode_idx")
        if not torch.is_tensor(episode) or int(episode.numel()) != sample_count:
            raise ValueError("action-path probe requires one episode_idx per sample")
        episode_rows.append(episode.detach().cpu().numpy().reshape(-1))
        finished_batches += 1
        intervention_samples += sample_count
        print(
            f"{probe_prefix} "
            f"batch={batch_index}/{planned_batches} "
            f"selected={finished_batches}/{len(selected_indices)} "
            f"samples={intervention_samples}",
            flush=True,
        )

    if finished_batches != len(selected_indices):
        raise RuntimeError(
            "action-path probe finished "
            f"{finished_batches}/{len(selected_indices)} selected batches"
        )
    if (
        require_grounded_intent_effect_contract
        and baseline_identity_checked_batches != finished_batches
    ):
        raise RuntimeError(
            "grounded model-path replay comparison covered "
            f"{baseline_identity_checked_batches}/{finished_batches} "
            "selected batches"
        )
    target = np.concatenate(target_rows)
    current = np.concatenate(current_rows)
    episode_ids = np.concatenate(episode_rows)
    joined = {mode: np.concatenate(rows) for mode, rows in predictions.items()}
    action_offsets = tuple(int(value) for value in cfg.flow_jepa_action_offsets)
    mode_metrics = {
        mode: _flow_address_action_metrics(
            pred,
            target,
            current,
            gripper_index=cfg.gripper_index,
            gripper_event_threshold=trainer.gripper_event_threshold,
            action_offsets=action_offsets,
        )
        for mode, pred in joined.items()
    }
    paired = _action_path_paired_metrics(
        joined=joined,
        target=target,
        episode_ids=episode_ids,
        action_offsets=action_offsets,
        bootstrap_reps=bootstrap_reps,
        bootstrap_seed=bootstrap_seed,
    )
    representation = {
        key: value / float(max(representation_weight, 1))
        for key, value in representation_sums.items()
    }
    if "flow_jepa_raw_seed_reliability" in representation:
        representation["flow_jepa_seed_reliability"] = representation[
            "flow_jepa_raw_seed_reliability"
        ]
    reader_intervention_diagnostics = {
        mode: {
            key: value / float(max(mode_reader_weights[mode], 1)) for key, value in values.items()
        }
        for mode, values in mode_reader_sums.items()
        if values
    }
    averaged_boundary_diagnostics = {
        mode: {
            key: value / float(max(boundary_diagnostic_weights[mode].get(key, 0), 1))
            for key, value in values.items()
        }
        for mode, values in boundary_diagnostics.items()
        if values
    }
    acceptance_matrix = _model_path_acceptance_matrix(
        joined=joined,
        paired=paired,
        verification_counts=verification_counts,
        boundary_diagnostics=averaged_boundary_diagnostics,
        baseline_identity_max_abs_delta=baseline_identity_max_abs_delta,
        representation=representation,
    )
    mode_semantics = {
        "policy_zero": (
            "remove the complete typed policy bank (or the legacy fixed "
            "policy workspace) entering the final decoder"
        ),
        "policy_temporal_shuffle": ("misalign that policy input across the action horizon"),
        "world_residual_zero": (
            "keep the grounding output at every slot and remove only the "
            "residual written by the world blocks"
        ),
        "world_residual_anchor_shuffle": (
            "keep grounding/position slots fixed and roll only the world-block "
            "residual across future anchors"
        ),
        "world_residual_spatial_shuffle": (
            "keep grounding/position slots and camera identity fixed and roll "
            "only the world-block residual across xy coordinates"
        ),
        "world_residual_spatiotemporal_shuffle": (
            "keep grounding/position slots fixed and misalign only the "
            "world-block residual across anchors and xy"
        ),
        "flow_zero": "zero only learned flow entering the soft address compiler",
        "flow_episode_shuffle": (
            "attach each sample's learned flow to another sample while keeping "
            "DINO, RGB, camera identity, and action noise fixed"
        ),
        "flow_spatial_shuffle": (
            "spatially misalign only learned flow entering the address compiler"
        ),
        "dino_key_spatial_shuffle": (
            "misalign target DINO address keys while raw values and flow remain fixed"
        ),
        "source_raw_match_zero": (
            "remove only the source-side raw appearance used by the learned "
            "source/target fine-address pair key; target keys, target "
            "high-frequency values, DINO, flow, and camera identity remain fixed"
        ),
        "source_raw_match_spatial_shuffle": (
            "spatially misalign only source-side raw appearance in the learned "
            "fine-address pair key while target keys/values, DINO, flow, and "
            "camera identity remain fixed"
        ),
        "joint_address_key_spatial_shuffle": (
            "jointly spatially misalign learned flow, target-DINO keys, and "
            "source-side raw pair evidence while target raw keys/values and "
            "camera identity remain fixed"
        ),
        "raw_value_zero": ("remove only high-frequency raw values after address-bank compilation"),
        "raw_value_spatial_shuffle": (
            "misalign only high-frequency raw values while address keys stay fixed"
        ),
        "literal_current_rgb_zero": (
            "remove only exact current coordinate-sampled RGB values while learned "
            "detail, DINO, geometry, routing, and future transport stay fixed"
        ),
        "literal_current_rgb_spatial_shuffle": (
            "spatially misalign only exact current RGB values while every selector "
            "key and learned-detail value stays fixed"
        ),
        "future_transport_neutral": (
            "replace only P's learned future transport by the current anchor, "
            "unit scale and neutral visibility while retaining W source priors"
        ),
        "future_transport_spatial_shuffle": (
            "spatially misalign only the future transport consumed by P while "
            "current RGB/detail values and W source priors remain fixed"
        ),
        "current_context_masked": (
            "reuse the deterministic observation-derived JEPA target mask on "
            "the latest online RGB/DINO context while checkpoint, eval mode, "
            "future targets, action noise and every other condition stay fixed"
        ),
        "semantic_owner_zero": (
            "remove P's semantic fine/coarse keys and semantic W sidecar while "
            "retaining appearance, geometry, public W state and precision values"
        ),
        "semantic_owner_shuffle": ("misalign only P's semantic keys and semantic W sidecar"),
        "appearance_owner_zero": (
            "remove P's appearance verifier keys and appearance W sidecar while "
            "retaining semantic source ownership, geometry and precision values"
        ),
        "appearance_owner_shuffle": ("misalign only P's appearance keys and appearance W sidecar"),
        "geometry_owner_zero": (
            "remove P's geometry keys and geometry W sidecar while retaining "
            "semantic relevance, appearance and precision values"
        ),
        "geometry_owner_shuffle": ("misalign only P's geometry keys and geometry W sidecar"),
        "address_posterior_uniform": (
            "replace the learned joint camera/slot/xy posterior by a valid-state uniform posterior"
        ),
        "fine_offset_zero": (
            "force every coarse hypothesis to read its zero-offset fine candidate"
        ),
        "camera_posterior_uniform": (
            "retain within-camera routing but force equal posterior mass across valid cameras"
        ),
        "camera_swap": (
            "swap complete camera-owned key/value banks while world camera queries remain fixed"
        ),
        "world_address_query_zero": (
            "zero only the W-organized per-horizon/per-camera/per-xy chart "
            "entering late address compatibility; action queries and the "
            "observation-owned key/value bank remain fixed"
        ),
        "world_address_query_spatial_shuffle": (
            "spatially misalign only the W-organized chart entering late "
            "address compatibility while action queries, camera identity, "
            "and the observation-owned key/value bank remain fixed"
        ),
        "goal_zero": (
            "zero the real T5 condition before the grounded S organizer while "
            "leaving state/history/vision fixed"
        ),
        "goal_episode_shuffle": (
            "permute per-sample T5 tensors before the grounded S organizer"
        ),
        "action_history_zero": (
            "zero executed-action history before both proposal and condition memory"
        ),
        "action_history_condition_zero": (
            "null only the explicit history-condition lane while retaining the proposal"
        ),
        "action_history_proposal_zero": (
            "null only history-derived proposal content at the seed keep "
            "boundary while retaining direct executed-history condition memory "
            "and the proposal slot/type identity"
        ),
        "action_history_proposal_episode_shuffle": (
            "episode-shuffle only history-derived proposal tokens while "
            "retaining direct executed-history condition memory"
        ),
        "action_history_episode_shuffle": (
            "permute executed-action history before proposal and condition encoding"
        ),
        "action_history_truncate": (
            "retain only the configured recent history and remove the older prefix"
        ),
        "phase_belief_zero": ("zero only the stateless phase selector context"),
        "phase_belief_episode_shuffle": ("permute only the stateless phase selector context"),
        "condition_query_zero": (
            "zero the separate goal/history selector context used by W blocks, W->P, and detail reads"
        ),
        "horizon_address_zero": ("remove only the bounded online G3-to-W1 horizon-address write"),
        "horizon_address_episode_shuffle": (
            "episode-shuffle only the online G3-to-W1 address write; for a "
            "one-sample smoke, rotate its horizon axis deterministically"
        ),
        "address_g1_zero": (
            "remove the learned G1 candidate-chart logit update while retaining "
            "the observation prior and all later soft selection"
        ),
        "address_g1_episode_shuffle": (
            "misalign only the G1 candidate-hypothesis update across episodes"
        ),
        "address_g2_zero": (
            "replace the learned G2 fine posterior by the valid uniform support "
            "without changing raw values"
        ),
        "address_g2_episode_shuffle": (
            "misalign only the G2 rectified fine posterior across episodes"
        ),
        "address_g3_zero": (
            "remove only the G3 canonical selector priors and low-rank handoff; "
            "the observation bank and final P read remain intact"
        ),
        "address_g3_episode_shuffle": (
            "misalign only the G3 canonical priors and selector summary"
        ),
        "address_g3_slot_permute": (
            "consistently cycle only the within-sample G3 object-fact/owner "
            "slot axis; preserve the public scene base and P1 address/value lattice"
        ),
        "address_g3_slot_mean": (
            "replace each G3 object-fact/owner slot by its within-cell slot "
            "mean; preserve the public scene base and P1 address/value lattice"
        ),
        "interval_stage_zero": (
            "remove only the bounded per-camera/per-xy interval-stage delta "
            "at W->P while retaining the W carrier and precision bank"
        ),
        "interval_stage_episode_shuffle": (
            "episode-shuffle only the bounded interval-stage delta at W->P; "
            "for a one-sample smoke, rotate the horizon axis deterministically"
        ),
    }
    if typed_policy_fusion:
        for source in (*grounding_sources, *world_sources, *policy_sources):
            mode_semantics[f"{source}_delta_zero"] = (
                f"zero only the typed {source} residual candidate at its next role bridge"
            )
            mode_semantics[f"{source}_delta_episode_shuffle"] = (
                f"episode-shuffle only the typed {source} residual candidate at its next role bridge"
            )
        mode_semantics["protected_detail_zero"] = (
            "zero only the protected precision-detail candidate entering bottom MMDiT"
        )
        mode_semantics["protected_detail_episode_shuffle"] = (
            "episode-shuffle only the protected precision-detail candidate entering bottom MMDiT"
        )
        mode_semantics["w2p_far_context_zero"] = (
            "zero only the non-action (+48) typed context candidates at W->P; "
            "the 4/12/24 action alignment and full bottom rollout evidence remain fixed"
        )
        mode_semantics["w2p_far_context_episode_shuffle"] = (
            "episode-shuffle only the non-action (+48) typed context candidates at W->P; "
            "the 4/12/24 action alignment and full bottom rollout evidence remain fixed"
        )
        mode_semantics["bottom_far_rollout_zero"] = (
            "zero only the +48 chart entering bottom Evidence MMDiT; the typed "
            "W->P far candidate and 4/12/24 rollout charts remain fixed"
        )
        mode_semantics["bottom_far_rollout_episode_shuffle"] = (
            "episode-shuffle only the +48 chart entering bottom Evidence MMDiT; "
            "the typed W->P far candidate and 4/12/24 rollout charts remain fixed"
        )
        mode_semantics["all_far_context_zero"] = (
            "jointly zero the typed W->P +48 candidates and the +48 bottom "
            "rollout chart while retaining every 4/12/24 path"
        )
        mode_semantics["all_far_context_episode_shuffle"] = (
            "jointly episode-shuffle the typed W->P +48 candidates and the +48 "
            "bottom rollout chart while retaining every 4/12/24 path"
        )
    if matched_current_context_probe:
        mode_semantics["interval_stage_zero"] = (
            "zero the interval owner state at every configured online W "
            "boundary; the old post-W organizer is frozen and is not the "
            "intervention target"
        )
        mode_semantics["interval_stage_episode_shuffle"] = (
            "spatially misalign the interval owner state at every configured "
            "online W boundary while retaining the other three typed owners"
        )
        mode_semantics["goal_horizon_context_zero"] = (
            "zero only the ordered goal selector bank after goal resampling"
        )
        mode_semantics["history_horizon_context_zero"] = (
            "zero only the ordered executed-history selector bank"
        )
        if require_complete_v117_contract:
            mode_semantics.update(
                {
                    "intent_window_selector_uniform": (
                        "replace only S3's three-window P2 logit prior with a "
                        "uniform distribution; effect values and temporal control stay fixed"
                    ),
                    "intent_window_selector_episode_shuffle": (
                        "misalign only S3's three-window P2 logit prior"
                    ),
                    "intent_temporal_zero": (
                        "zero only S3's horizon-resolved temporal control before P3"
                    ),
                    "intent_temporal_episode_shuffle": (
                        "misalign only S3's horizon-resolved temporal control before P3"
                    ),
                }
            )
        if require_differential_intent_effect_contract:
            mode_semantics.update(
                {
                    "intent_state_zero": (
                        "zero the three typed reads from the one canonical "
                        "four-token IntentStateBank"
                    ),
                    "intent_state_episode_shuffle": (
                        "misalign the three typed reads from the canonical "
                        "IntentStateBank without changing Goal/history/G3 inputs"
                    ),
                    "intent_temporal_zero": (
                        "zero only the horizon-resolved read from the same "
                        "IntentStateBank before P3"
                    ),
                    "intent_temporal_episode_shuffle": (
                        "misalign only the horizon-resolved IntentStateBank read"
                    ),
                }
            )
            for window_name in ("near", "mid", "late"):
                mode_semantics[
                    f"intent_window_{window_name}_zero"
                ] = (
                    f"zero only the {window_name} typed IntentStateBank read "
                    "before W/P2"
                )
                mode_semantics[
                    f"intent_window_{window_name}_shuffle"
                ] = (
                    f"misalign only the {window_name} typed IntentStateBank "
                    "read before W/P2"
                )
        if require_grounded_intent_effect_contract:
            mode_semantics.update(
                {
                    "intent_state_zero": (
                        "zero every observable output of the one grounded "
                        "StatelessIntentState after S and before G/W/P consumers"
                    ),
                    "intent_state_episode_shuffle": (
                        "episode-shuffle the complete grounded intent state "
                        "without changing T5/history/G inputs"
                    ),
                    "intent_goal_set_zero": (
                        "zero only S's already-compiled protected goal output; "
                        "this audits an independent second landing and is not "
                        "a T5-input intervention"
                    ),
                    "intent_goal_set_episode_shuffle": (
                        "episode-shuffle only S's already-compiled protected "
                        "goal output; this does not recompute other S fields"
                    ),
                    "intent_achieved_zero": (
                        "zero only S's achieved-evidence output"
                    ),
                    "intent_achieved_episode_shuffle": (
                        "episode-shuffle only S's achieved-evidence output"
                    ),
                    "intent_remaining_zero": (
                        "zero only S's remaining-goal output"
                    ),
                    "intent_remaining_episode_shuffle": (
                        "episode-shuffle only S's remaining-goal output"
                    ),
                    "intent_temporal_zero": (
                        "zero only the 24-query temporal control consumed by P3"
                    ),
                    "intent_temporal_episode_shuffle": (
                        "episode-shuffle only the 24-query temporal control"
                    ),
                }
            )
            for interval_name in (
                "h4_8",
                "h8_16",
                "h16_32",
                "h32_48",
            ):
                mode_semantics[
                    f"intent_interval_{interval_name}_zero"
                ] = (
                    f"zero only S's {interval_name} interval intent before W/P2"
                )
                mode_semantics[
                    f"intent_interval_{interval_name}_shuffle"
                ] = (
                    f"misalign only S's {interval_name} interval intent before W/P2"
                )
        mode_semantics["p1_appearance_gateway_zero"] = (
            "zero only the mandatory W-conditioned appearance query after "
            "its P1 gateway projection; upstream W appearance state, policy "
            "query, candidate keys and precision values remain fixed"
        )
        mode_semantics["p1_appearance_gateway_spatial_shuffle"] = (
            "spatially misalign only the mandatory W-conditioned appearance "
            "query after its P1 gateway projection"
        )
        for depth in range(int(cfg.flow_jepa_world_blocks) + 1):
            mode_semantics[f"functional_w{depth}_route_zero"] = (
                f"zero only the selected typed owner write at W boundary {depth}"
            )
            mode_semantics[f"functional_w{depth}_route_spatial_shuffle"] = (
                f"spatially misalign only the selected typed owner write at W boundary {depth}"
            )
        for owner in ("semantic", "appearance", "geometry", "horizon"):
            mode_semantics[f"p2_{owner}_zero"] = (
                f"zero only P2's {owner} innovation before its null-capable router"
            )
            mode_semantics[f"p2_{owner}_shuffle"] = (
                f"misalign only P2's {owner} innovation before routing"
            )
    if (
        require_complete_v115_contract
        or require_complete_v116_contract
        or require_complete_v117_contract
        or require_differential_intent_effect_contract
        or require_grounded_intent_effect_contract
    ):
        mode_semantics["future_effect_zero"] = (
            (
                "zero every slotwise effect component in the one "
                "DifferentialWindowEffectBank while retaining the protected "
                "current G3 reference"
            )
            if (
                require_differential_intent_effect_contract
                or require_grounded_intent_effect_contract
            )
            else (
                "replace the single W2-produced FutureEffectField at the W->P "
                "boundary with an exact-zero field, including uncertainty; "
                "the frozen future teacher is not touched"
            )
        )
        mode_semantics["future_effect_spatial_shuffle"] = (
            (
                "spatially misalign every slotwise effect component while "
                "retaining the protected current G3 reference"
            )
            if (
                require_differential_intent_effect_contract
                or require_grounded_intent_effect_contract
            )
            else (
                "spatially misalign every component of that same online "
                "FutureEffectField before P2/P3 consume it"
            )
        )
        if (
            require_complete_v116_contract
            or require_complete_v117_contract
            or require_differential_intent_effect_contract
            or require_grounded_intent_effect_contract
        ):
            mode_semantics.update(
                {
                    "future_effect_current_zero": (
                        "zero only protected current content and reconstruct "
                        "successor=current+semantic_delta"
                    ),
                    "future_effect_current_spatial_shuffle": (
                        "spatially shuffle only current content and reconstruct "
                        "successor=current+semantic_delta"
                    ),
                    "future_effect_semantic_zero": (
                        "zero only semantic delta and set successor=current"
                    ),
                    "future_effect_semantic_spatial_shuffle": (
                        "spatially shuffle semantic delta and reconstruct successor"
                    ),
                    "future_effect_transport_zero": (
                        "zero only transport mean/covariance"
                    ),
                    "future_effect_transport_spatial_shuffle": (
                        "spatially shuffle only transport mean/covariance"
                    ),
                    "future_effect_reliability_zero": (
                        (
                            "zero only reliability and uncertainty; semantic, "
                            "transport and zero-centred visibility/persistence "
                            "changes remain fixed"
                        )
                        if require_grounded_intent_effect_contract
                        else (
                            "zero only persistence, visibility and uncertainty channels"
                        )
                    ),
                    "future_effect_reliability_spatial_shuffle": (
                        (
                            "spatially shuffle only reliability and uncertainty"
                        )
                        if require_grounded_intent_effect_contract
                        else (
                            "spatially shuffle only persistence, visibility and uncertainty"
                        )
                    ),
                    "future_effect_reliability_one": (
                        "set only the grounded online effect reliability to one "
                        "before P2; keep effect content, geometry, validity and "
                        "uncertainty fixed (audit-only bypass)"
                    ),
                }
            )
        if require_differential_intent_effect_contract:
            for slot_name in ("near", "mid", "late"):
                mode_semantics[f"future_effect_{slot_name}_zero"] = (
                    f"zero only the {slot_name} DifferentialWindowEffectBank "
                    "slot before P2"
                )
                mode_semantics[f"future_effect_{slot_name}_shuffle"] = (
                    f"misalign only the {slot_name} "
                    "DifferentialWindowEffectBank slot before P2"
                )
        if require_grounded_intent_effect_contract:
            for interval_name in (
                "h4_8",
                "h8_16",
                "h16_32",
                "h32_48",
            ):
                mode_semantics[f"future_effect_{interval_name}_zero"] = (
                    f"zero only the {interval_name} object-level effect fields "
                    "before the one P2 read"
                )
                mode_semantics[f"future_effect_{interval_name}_shuffle"] = (
                    f"spatially misalign only the {interval_name} object-level "
                    "effect fields before P2"
                )
        lanes = (
            ["precision", "temporal"]
            if (
                require_differential_intent_effect_contract
                or require_grounded_intent_effect_contract
            )
            else ["precision", "effect", "temporal"]
        )
        if not (
            require_complete_v116_contract
            or require_complete_v117_contract
            or require_differential_intent_effect_contract
            or require_grounded_intent_effect_contract
        ):
            lanes.append("terminal")
        for lane in lanes:
            mode_semantics[f"p3_{lane}_delta_zero"] = (
                f"zero only P3's typed {lane} lane entering bottom MMDiT"
            )
            mode_semantics[f"p3_{lane}_delta_episode_shuffle"] = (
                f"episode-shuffle only P3's typed {lane} lane entering bottom MMDiT"
            )
    schema = "clearvla-v101-action-path-intervention-v3"
    if typed_policy_fusion:
        schema = "clearvla-v103-model-path-intervention-v3"
        for enabled, candidate in (
            (
                require_complete_v104_contract,
                "clearvla-v104-model-path-intervention-v3",
            ),
            (
                require_complete_v105_contract,
                "clearvla-v105-model-path-intervention-v4",
            ),
            (
                require_complete_v106_contract,
                "clearvla-v106-model-path-intervention-v5",
            ),
            (
                require_complete_v107_contract,
                "clearvla-v107-model-path-intervention-v6",
            ),
            (
                require_complete_v108_contract,
                "clearvla-v108-model-path-intervention-v7",
            ),
            (
                require_complete_v109_contract,
                "clearvla-v109-model-path-intervention-v8",
            ),
            (
                require_complete_v110_contract,
                "clearvla-v110-model-path-intervention-v9",
            ),
            (
                require_complete_v111_contract,
                "clearvla-v111-model-path-intervention-v10",
            ),
            (
                require_complete_v112_contract,
                "clearvla-v112-model-path-intervention-v11",
            ),
            (
                require_complete_v113_contract,
                "clearvla-v113-model-path-intervention-v13",
            ),
            (
                require_complete_v114_contract,
                "clearvla-v114-model-path-intervention-v14",
            ),
            (
                require_complete_v115_contract,
                "clearvla-v115-model-path-intervention-v15",
            ),
            (
                require_complete_v117_contract,
                "clearvla-v117-model-path-intervention-v17",
            ),
            (
                require_differential_intent_effect_contract,
                "clearvla-differential-intent-effect-model-path-v18",
            ),
            (
                require_grounded_intent_effect_contract,
                "clearvla-grounded-intent-effect-323-model-path-v1",
            ),
            (
                require_complete_v116_contract,
                "clearvla-v116-model-path-intervention-v16",
            ),
        ):
            if enabled:
                schema = candidate
    current_context_mask_comparison = (
        _summarize_current_context_mask_comparison(
            enabled=matched_current_context_probe,
            finished_batches=finished_batches,
            intervention_samples=intervention_samples,
            comparison_batches=current_mask_comparison_batches,
            comparison_weight=current_mask_comparison_weight,
            metric_sums=current_mask_metric_sums,
            boundary_sums=current_mask_boundary_sums,
        )
    )
    return {
        "schema": schema,
        "complete_v103_contract_verified": bool(
            require_complete_v103_contract
            or require_complete_v104_contract
            or require_complete_v105_contract
            or require_complete_v106_contract
            or require_complete_v107_contract
            or require_complete_v108_contract
            or require_complete_v109_contract
            or require_complete_v110_contract
            or complete_v111_or_later
        ),
        "complete_v104_contract_verified": bool(
            require_complete_v104_contract
            or require_complete_v105_contract
            or require_complete_v106_contract
            or require_complete_v107_contract
            or require_complete_v108_contract
            or require_complete_v109_contract
            or require_complete_v110_contract
            or complete_v111_or_later
        ),
        "complete_v105_contract_verified": bool(
            require_complete_v105_contract
            or require_complete_v106_contract
            or require_complete_v107_contract
            or require_complete_v108_contract
            or require_complete_v109_contract
            or require_complete_v110_contract
            or complete_v111_or_later
        ),
        "complete_v106_contract_verified": bool(
            require_complete_v106_contract
            or require_complete_v107_contract
            or require_complete_v108_contract
            or require_complete_v109_contract
            or require_complete_v110_contract
            or complete_v111_or_later
        ),
        "complete_v107_contract_verified": bool(
            require_complete_v107_contract
            or require_complete_v108_contract
            or require_complete_v109_contract
            or require_complete_v110_contract
            or complete_v111_or_later
        ),
        "complete_v108_contract_verified": bool(
            require_complete_v108_contract
            or require_complete_v109_contract
            or require_complete_v110_contract
            or complete_v111_or_later
        ),
        "complete_v109_contract_verified": bool(
            require_complete_v109_contract
            or require_complete_v110_contract
            or complete_v111_or_later
        ),
        "complete_v110_contract_verified": bool(
            require_complete_v110_contract or complete_v111_or_later
        ),
        "complete_v111_contract_verified": bool(complete_v111_or_later),
        "complete_v112_contract_verified": bool(
            require_complete_v112_contract or complete_v113_or_later
        ),
        "complete_v113_contract_verified": bool(complete_v113_or_later),
        "complete_v114_contract_verified": bool(
            require_complete_v114_contract
            or require_complete_v115_contract
            or require_complete_v116_contract
            or require_complete_v117_contract
            or require_differential_intent_effect_contract
            or require_grounded_intent_effect_contract
        ),
        "complete_v115_contract_verified": bool(
            require_complete_v115_contract
            or require_complete_v116_contract
            or require_complete_v117_contract
            or require_differential_intent_effect_contract
            or require_grounded_intent_effect_contract
        ),
        "complete_v116_contract_verified": bool(
            require_complete_v116_contract
            or require_complete_v117_contract
            or require_differential_intent_effect_contract
            or require_grounded_intent_effect_contract
        ),
        "complete_v117_contract_verified": bool(
            require_complete_v117_contract
            or require_differential_intent_effect_contract
        ),
        "differential_intent_effect_contract_verified": bool(
            require_differential_intent_effect_contract
        ),
        "grounded_intent_effect_contract_verified": bool(
            require_grounded_intent_effect_contract
        ),
        "architecture_manifest": (
            GROUNDING_MANIFEST.as_dict()
            if require_grounded_intent_effect_contract
            else None
        ),
        "planned_batches": int(planned_batches),
        "selected_batch_indices": sorted(selected_indices),
        **selection_metadata,
        "finished_intervention_batches": int(finished_batches),
        "intervention_samples": int(intervention_samples),
        "intervention_coverage": float(finished_batches / planned_batches),
        "patched_baseline_max_abs_delta": float(baseline_identity_max_abs_delta),
        "baseline_identity_checked_batches": int(
            baseline_identity_checked_batches
        ),
        "baseline_identity_tolerance": float(replay_tolerance),
        "inference_steps": int(trainer.eval_inference_steps),
        "requested_intervention_modes": (
            None if requested_intervention_modes is None else list(requested_intervention_modes)
        ),
        "action_offsets": list(action_offsets),
        "mode_semantics": mode_semantics,
        "scope_limits": {
            "policy": (
                "policy modes do not remove decoder evidence, noisy action state, "
                "state/history, language, proposal, or terminal policy-layer contracts"
            ),
            "world": (
                "world modes preserve the rollout chart entering the world blocks and "
                "intervene only on the world-block residual; all other conditions remain"
            ),
            "raw_address": (
                "raw-address modes alter reader coordinates only; RGB/DINO value content "
                "and camera identity remain intact"
            ),
            "raw_detail": (
                "raw-detail modes intervene after reader output projection; DINO, "
                "base-only reader output, raw-detail amplitude, and camera identity "
                "remain fixed except that zero removes detail amplitude by design"
            ),
            "history_proposal": (
                "proposal-only modes alter only proposal content/identity pairing; "
                "the direct compressed history condition, vision, goal, phase, "
                "state, and action noise remain fixed"
            ),
        },
        "pairing_contract": (
            "same checkpoint, validation samples, initial action noise, inference "
            "steps, execution policy, and all non-targeted inputs; the proposal "
            "is also fixed except when a full-history or proposal-only "
            "intervention intentionally tests the history-derived proposal path; "
            "proposal-only modes keep the direct history-condition memory fixed"
            "; grounded runs additionally require the explicit none replay to "
            "match ordinary deployment on every selected batch before any "
            "causal result is returned"
        ),
        "intervention_verified_batches": verification_counts,
        "episode_ids": episode_ids.astype(int).tolist(),
        "observed_episode_ids": sorted(int(value) for value in np.unique(episode_ids)),
        "episode_clusters": int(len(np.unique(episode_ids))),
        "representation": representation,
        "boundary_diagnostics": averaged_boundary_diagnostics,
        "current_context_mask_comparison": (current_context_mask_comparison),
        "acceptance_matrix": acceptance_matrix,
        "reader_intervention_diagnostics": reader_intervention_diagnostics,
        "modes": mode_metrics,
        "paired": paired,
    }


# Version-neutral name for the V103 launcher; retain the historical function
# for existing V101 scripts and serialized evaluation workflows.
evaluate_model_path_intervention = evaluate_v101_action_path_intervention


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


@torch.no_grad()
def _validate_v106_preflight_target_pack(
    pack: dict[str, Tensor],
    *,
    config: Any,
    batch_size: int,
) -> None:
    """Validate the formal interval teacher before the first training batch."""

    positions = (
        int(config.num_cameras) * int(config.flow_jepa_grid_size) * int(config.flow_jepa_grid_size)
    )
    anchors = int(config.future_anchors)
    hidden = int(config.hidden_size)
    grounded_mainline = bool(
        int(
            getattr(
                config,
                "flow_jepa_grounded_intent_effect_mainline",
                0,
            )
        )
    )
    teacher_content_width = (
        int(config.visual_token_dim) if grounded_mainline else hidden
    )
    future_shape = (
        int(batch_size),
        anchors * positions,
        teacher_content_width,
    )
    current_shape = (
        int(batch_size),
        positions,
        teacher_content_width,
    )
    for key in (
        "flow_jepa_future_target",
        "flow_jepa_interval_progress_target",
        "flow_jepa_interval_endpoint_target",
    ):
        value = pack.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != future_shape:
            raise ValueError(
                f"V106 preflight {key} must be {future_shape}, "
                f"got {None if value is None else tuple(value.shape)}"
            )
        if value.dtype != torch.float32:
            raise TypeError(f"V106 preflight {key} must remain float32, got {value.dtype}")
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"V106 preflight {key} is non-finite")
    current = pack.get("flow_jepa_current_target")
    if not torch.is_tensor(current) or tuple(current.shape) != current_shape:
        raise ValueError(
            "V106 preflight flow_jepa_current_target must be "
            f"{current_shape}, got "
            f"{None if current is None else tuple(current.shape)}"
        )
    if current.dtype != torch.float32:
        raise TypeError(
            f"V106 preflight flow_jepa_current_target must remain float32, got {current.dtype}"
        )
    if not bool(torch.isfinite(current).all()):
        raise FloatingPointError("V106 preflight flow_jepa_current_target is non-finite")
    mask = pack.get("flow_jepa_future_target_mask")
    expected_mask_shape = future_shape[:2]
    if not torch.is_tensor(mask) or tuple(mask.shape) != expected_mask_shape:
        raise ValueError(
            "V106 preflight flow_jepa_future_target_mask must be "
            f"{expected_mask_shape}, got "
            f"{None if mask is None else tuple(mask.shape)}"
        )
    if mask.dtype != torch.bool:
        raise TypeError(
            f"V106 preflight flow_jepa_future_target_mask must be bool, got {mask.dtype}"
        )
    expected_support_count = len(tuple(config.flow_jepa_effective_interval_support_offsets))
    support_count = pack.get("flow_jepa_interval_support_count")
    if (
        not torch.is_tensor(support_count)
        or int(support_count.numel()) != 1
        or not bool(torch.isfinite(support_count).all())
        or float(support_count.detach().cpu()) != float(expected_support_count)
    ):
        raise ValueError(
            "V106 preflight interval support count does not match the "
            f"serialized contract ({expected_support_count})"
        )
    effective_support = pack.get("flow_jepa_interval_effective_support")
    max_interval_supports = max(
        sum(
            int(start) <= int(offset) <= int(end)
            for offset in config.flow_jepa_effective_interval_support_offsets
        )
        for start, end in config.flow_jepa_interval_windows
    )
    support_tolerance = 1e-4
    if (
        not torch.is_tensor(effective_support)
        or int(effective_support.numel()) != 1
        or not bool(torch.isfinite(effective_support).all())
        or not (
            1.0 - support_tolerance
            <= float(effective_support.detach().cpu())
            <= float(max_interval_supports) + support_tolerance
        )
    ):
        raise ValueError(
            "V106 preflight effective interval support is outside the "
            f"valid [1,{max_interval_supports}] range"
        )
    if grounded_mainline:
        slots = 4
        content_width = int(config.visual_token_dim)
        object_slots = int(config.flow_jepa_address_slots)
        prefix = (
            int(batch_size),
            slots,
            int(config.num_cameras),
            int(config.flow_jepa_grid_size),
            int(config.flow_jepa_grid_size),
            object_slots,
        )
        typed_widths = {
            "successor": content_width,
            "semantic": content_width,
            "transport": 2,
            "transport_covariance": 3,
            "persistence": 1,
            "visibility": 1,
            "uncertainty": 1,
            "reliability": 1,
        }
        for name, width in typed_widths.items():
            key = f"flow_jepa_future_effect_{name}_target_slots"
            value = pack.get(key)
            expected = (*prefix, int(width))
            if (
                not torch.is_tensor(value)
                or tuple(value.shape) != expected
                or value.dtype != torch.float32
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(
                    f"grounded preflight {key} must be finite float32 "
                    f"{expected}, got "
                    f"{None if value is None else tuple(value.shape)}"
                )
        current_reference = pack.get(
            "flow_jepa_future_effect_current_reference_target"
        )
        expected_current = (
            int(batch_size),
            int(config.num_cameras),
            int(config.flow_jepa_grid_size),
            int(config.flow_jepa_grid_size),
            object_slots,
            content_width,
        )
        if (
            not torch.is_tensor(current_reference)
            or tuple(current_reference.shape) != expected_current
            or current_reference.dtype != torch.float32
            or not bool(torch.isfinite(current_reference).all())
        ):
            raise ValueError(
                "grounded preflight current reference must be finite "
                f"float32 {expected_current}, got "
                f"{None if current_reference is None else tuple(current_reference.shape)}"
            )
        for key in (
            "flow_jepa_future_effect_persistence_target_slots",
            "flow_jepa_future_effect_visibility_target_slots",
        ):
            change = pack[key]
            if bool((change > 1e-6).any()) or bool((change < -1.0001).any()):
                raise ValueError(
                    f"grounded preflight {key} is not a zero-centred "
                    "change in [-1,0]"
                )
        active = pack.get("grounded_intent_effect_active")
        if (
            not torch.is_tensor(active)
            or int(active.numel()) != 1
            or float(active.detach().cpu()) != 1.0
        ):
            raise ValueError(
                "grounded preflight capability marker is missing"
            )
    if int(getattr(config, "flow_jepa_window_effect_bank", 0)):
        slots = int(getattr(config, "flow_jepa_future_slots", 0))
        canonical_slots = int(config.flow_jepa_address_slots)
        prefix = (
            int(batch_size),
            slots,
            int(config.num_cameras),
            int(config.flow_jepa_grid_size),
            int(config.flow_jepa_grid_size),
            canonical_slots,
        )
        differential = bool(
            int(
                getattr(
                    config,
                    "flow_jepa_differential_intent_effect_mainline",
                    0,
                )
            )
        )
        typed_widths = {
            "successor": hidden,
            "semantic": hidden,
            "transport": 2,
            "transport_covariance": 3,
            "persistence": 1,
            "visibility": 1,
            "uncertainty": 1,
            "reliability": 1,
        }
        if not differential:
            typed_widths = {"current": hidden, **typed_widths}
        for name, width in typed_widths.items():
            key = f"flow_jepa_future_effect_{name}_target_slots"
            value = pack.get(key)
            expected = (*prefix, int(width))
            if not torch.is_tensor(value) or tuple(value.shape) != expected:
                raise ValueError(
                    f"V117 preflight {key} must be {expected}, "
                    f"got {None if value is None else tuple(value.shape)}"
                )
            if value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
                raise ValueError(
                    f"V117 preflight {key} must be finite float32"
                )
        if differential:
            current_reference = pack.get(
                "flow_jepa_future_effect_current_reference_target"
            )
            expected_current = (
                int(batch_size),
                int(config.num_cameras),
                int(config.flow_jepa_grid_size),
                int(config.flow_jepa_grid_size),
                canonical_slots,
                hidden,
            )
            if (
                not torch.is_tensor(current_reference)
                or tuple(current_reference.shape) != expected_current
                or current_reference.dtype != torch.float32
                or not bool(torch.isfinite(current_reference).all())
            ):
                raise ValueError(
                    "differential preflight current reference must be finite "
                    f"float32 {expected_current}, got "
                    f"{None if current_reference is None else tuple(current_reference.shape)}"
                )
            intent_summary = pack.get(
                "flow_jepa_future_effect_intent_summary_target_slots"
            )
            expected_summary = (int(batch_size), slots, hidden)
            if (
                not torch.is_tensor(intent_summary)
                or tuple(intent_summary.shape) != expected_summary
                or intent_summary.dtype != torch.float32
                or not bool(torch.isfinite(intent_summary).all())
            ):
                raise ValueError(
                    "differential preflight intent summary must be finite "
                    f"float32 {expected_summary}, got "
                    f"{None if intent_summary is None else tuple(intent_summary.shape)}"
                )


@torch.no_grad()
def _validate_object_intent_preflight_output(
    output: dict[str, Tensor],
    *,
    config: Any,
    batch_size: int,
) -> None:
    """Validate the capability boundary, not the retired V106 target pack."""

    objects = 4
    intervals = 4
    content = int(config.visual_token_dim)
    expected = {
        "object_future_successor_target": (batch_size, intervals, objects, content),
        "object_future_semantic_target": (batch_size, intervals, objects, content),
        "object_future_transport_target": (batch_size, intervals, objects, 2),
        "object_future_covariance_target": (batch_size, intervals, objects, 3),
        "object_future_visibility_target": (batch_size, intervals, objects, 1),
        "object_future_persistence_target": (batch_size, intervals, objects, 1),
        "object_future_uncertainty_target": (batch_size, intervals, objects, 1),
        "object_future_validity_target": (batch_size, intervals, objects, 1),
    }
    for key, shape in expected.items():
        value = output.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != shape:
            raise ValueError(
                f"object-intent preflight {key} must be {shape}, got "
                f"{None if value is None else tuple(value.shape)}"
            )
        if value.requires_grad:
            raise ValueError(f"object-intent teacher target {key} retained autograd")
        finite = bool(torch.isfinite(value).all())
        if value.dtype != torch.float32 or not finite:
            raise ValueError(
                f"object-intent teacher target {key} must be finite float32; "
                f"got dtype={value.dtype} finite={finite}"
            )
    active = output.get("object_intent_dynamics_active")
    if (
        not torch.is_tensor(active)
        or int(active.numel()) != 1
        or float(active.detach().cpu()) != 1.0
    ):
        raise ValueError("object-intent preflight capability marker is missing")
    for key in (
        "object_fact_content",
        "object_intent_interval_queries",
        "object_intent_state_change_evidence",
        "object_future_semantic_prediction",
        "object_consequence_protected",
        "flow_jepa_policy_plan_precision",
        "flow_jepa_policy_plan_temporal",
        "object_policy_plan_state_change",
    ):
        value = output.get(key)
        if not torch.is_tensor(value) or not bool(torch.isfinite(value).all()):
            raise ValueError(f"object-intent preflight online boundary {key} is invalid")
    if "flow_jepa_execution_terminal_evidence" in output:
        raise ValueError(
            "object-intent state-change evidence cannot become terminal control"
        )
    external_terminal_bias = output.get(
        "evidence_execution_terminal_external_bias"
    )
    if torch.is_tensor(external_terminal_bias) and not bool(
        (external_terminal_bias.detach().float() == 0.0).all()
    ):
        raise ValueError(
            "object-intent path injected an external execution-terminal bias"
        )


@torch.no_grad()
def _preflight_evidence_dynamic_sampling(
    *,
    system: V39PolicySystem,
    loader: DataLoader,
    conditioner: RDT2Conditioner,
    device: torch.device,
    dtype: torch.dtype,
    camera_names: Sequence[str],
    trainer: V39PolicyTrainerConfig,
) -> None:
    """Exercise the real AMP deploy path before paying for a training epoch.

    Dynamic Evidence execution combines an autocast action stream with an FP32
    controller/policy plane.  Unit tests cover that contract, but only a real
    prepared validation batch reproduces the production input dtypes and
    iterative sampler.  Preserve RNG and model mode so this fail-fast check has
    no effect on the experiment trajectory.
    """

    decoder = getattr(system.planner, "evidence_latent_mmdit_action_decoder", None)
    interval_stage = bool(int(getattr(system.policy_config, "flow_jepa_interval_stage_delta", 0)))
    object_dynamics_mainline = bool(
        int(
            getattr(
                system.policy_config,
                "flow_jepa_object_intent_dynamics_mainline",
                0,
            )
        )
    )
    run_deploy_preflight = bool(
        decoder is not None
        and bool(getattr(decoder, "dynamic_block_route_enabled", False))
        and device.type == "cuda"
        and dtype in (torch.float16, torch.bfloat16)
    )
    if not run_deploy_preflight and not interval_stage:
        return

    saved_rng = rng_state()
    was_training = bool(system.training)
    system.eval()
    try:
        try:
            batch = next(iter(loader))
        except StopIteration as exc:
            raise ValueError("validation loader is empty during deploy preflight") from exc
        sample = prepare_v39_policy_sample(
            batch,
            conditioner=conditioner,
            system=system,
            camera_names=camera_names,
            device=device,
            dtype=dtype,
            include_target_visual=interval_stage,
        )
        deploy_sampling_passed = False
        if run_deploy_preflight:
            generator = torch.Generator(device=device)
            generator.manual_seed(37238)
            noise = system.codec.sample_noise(
                sample["policy_action"].shape[0],
                generator=generator,
                device=device,
                dtype=sample["visual"].dtype,
                action_state=sample["action_state"],
            )
            stop_midcut_eval = _is_contract_stage(trainer) and not _uses_layer_adapter_contract(
                trainer
            )
            with autocast_context(device, dtype):
                prediction_pack = system.sample(
                    sample["visual"],
                    sample["history_state"],
                    sample["executed_action_history"],
                    sample["state"],
                    raw_visual=sample.get("raw_visual"),
                    action_state=sample["action_state"],
                    steps=trainer.eval_inference_steps,
                    noise=noise,
                    use_proposal=True,
                    return_event_logits=True,
                    stop_at_midcut=stop_midcut_eval,
                    collect_diagnostics=False,
                )
            if not isinstance(prediction_pack, dict) or not isinstance(
                prediction_pack.get("action"), Tensor
            ):
                raise TypeError("deploy preflight expected a sampled action pack")
            prediction = prediction_pack["action"]
            if not bool(torch.isfinite(prediction).all()):
                raise FloatingPointError("non-finite action in deploy sampling preflight")
            deploy_sampling_passed = True

        interval_teacher_passed = False
        if interval_stage:
            if "target_visual" not in sample:
                raise RuntimeError("V106 preflight did not prepare interval teacher observations")
            if object_dynamics_mainline:
                with autocast_context(device, dtype):
                    object_output = system.flow_training_forward(
                        sample["visual"],
                        sample["history_state"],
                        sample["executed_action_history"],
                        sample["state"],
                        sample["policy_action"],
                        raw_visual=sample.get("raw_visual"),
                        action_state=sample["action_state"],
                        target_visual=sample["target_visual"],
                        future_training_pack=_object_intent_future_training_pack(
                            sample,
                            system=system,
                            require_teacher=True,
                        ),
                        make_counterfactuals=False,
                        collect_audit_metrics=True,
                    )
                _validate_object_intent_preflight_output(
                    object_output,
                    config=system.policy_config,
                    batch_size=int(sample["visual"].shape[0]),
                )
                del object_output
            else:
                with autocast_context(device, dtype):
                    visual_context = system.planner.encode_visual_context(
                        sample["visual"],
                        raw_visual=sample.get("raw_visual"),
                    )
                    if visual_context is None:
                        raise RuntimeError("V106 preflight could not compile online visual context")
                    target_pack = system.build_rollout_target_pack(
                        sample["visual"],
                        sample["target_visual"],
                        visual_context=visual_context,
                    )
                _validate_v106_preflight_target_pack(
                    target_pack,
                    config=system.policy_config,
                    batch_size=int(sample["visual"].shape[0]),
                )
            interval_teacher_passed = True
        preflight_version = (
            "object_intent_dynamics_323"
            if object_dynamics_mainline
            else "grounded_intent_effect_323"
            if int(
                getattr(
                    system.policy_config,
                    "flow_jepa_grounded_intent_effect_mainline",
                    0,
                )
            )
            else "v118"
            if int(
                getattr(
                    system.policy_config,
                    "flow_jepa_differential_intent_effect_mainline",
                    0,
                )
            )
            else "v117"
            if int(
                getattr(
                    system.policy_config,
                    "flow_jepa_stateless_intent_controller",
                    0,
                )
            )
            else "v116"
            if int(
                getattr(
                    system.policy_config,
                    "flow_jepa_supervised_effect_mainline",
                    0,
                )
            )
            else "v115"
            if int(
                getattr(
                    system.policy_config,
                    "flow_jepa_policy_plan_compiler",
                    0,
                )
            )
            else "v114"
            if int(
                getattr(
                    system.policy_config,
                    "flow_jepa_utility_precision_mainline",
                    0,
                )
            )
            else "v113"
            if int(
                getattr(
                    system.policy_config,
                    "flow_jepa_functional_mainline_routing",
                    0,
                )
            )
            else "v112"
            if int(
                getattr(
                    system.policy_config,
                    "flow_jepa_pre_value_owner_routing",
                    0,
                )
            )
            else "v111"
            if int(
                getattr(
                    system.policy_config,
                    "flow_jepa_structured_ownership_bottleneck",
                    0,
                )
            )
            else "v110"
            if int(
                getattr(
                    system.policy_config,
                    "flow_jepa_coordinate_typed_raw_detail",
                    0,
                )
            )
            else "v109"
            if int(
                getattr(
                    system.policy_config,
                    "flow_jepa_progressive_grounding_address",
                    0,
                )
            )
            else "v108"
            if int(
                getattr(
                    system.policy_config,
                    "flow_jepa_online_horizon_address",
                    0,
                )
            )
            else "v107"
            if all(
                int(getattr(system.policy_config, name, 0)) == 1
                for name in (
                    "flow_jepa_policy_multi_glimpse_address",
                    "flow_jepa_horizon_cell_fine_address",
                    "flow_jepa_interval_stage_typed_value",
                    "role_residual_contract_after_gate",
                )
            )
            else "v106"
            if (
                interval_stage
                and int(
                    getattr(
                        system.policy_config,
                        "flow_jepa_variance_safe_routing",
                        0,
                    )
                )
                and int(
                    getattr(
                        system.policy_config,
                        "flow_jepa_complete_numerical_contract",
                        0,
                    )
                )
                and float(
                    getattr(
                        trainer,
                        "flow_jepa_interval_stage_loss_weight",
                        0.0,
                    )
                )
                > 0.0
            )
            else "v105"
            if (
                int(
                    getattr(
                        system.policy_config,
                        "flow_jepa_horizon_soft_address",
                        0,
                    )
                )
                and int(
                    getattr(
                        trainer,
                        "flow_jepa_future_reliable_normalization",
                        0,
                    )
                )
                and float(
                    getattr(
                        trainer,
                        "flow_jepa_horizon_address_loss_weight",
                        0.0,
                    )
                )
                > 0.0
            )
            else "v104"
            if int(
                getattr(
                    system.policy_config,
                    "flow_jepa_sequential_horizon_memory",
                    0,
                )
            )
            else "v103"
            if int(
                getattr(
                    system.policy_config,
                    "flow_jepa_predictive_change_contract",
                    0,
                )
            )
            else "v99"
            if int(getattr(system.policy_config, "flow_jepa_zero_flow_guard", 0))
            else "v98"
            if int(getattr(system.policy_config, "flow_jepa_raw_image_enabled", 0))
            else "v96"
            if int(getattr(system.policy_config, "flow_jepa_late_bottleneck", 0))
            else "v95"
            if int(getattr(system.policy_config, "flow_jepa_enabled", 0))
            else "v94"
        )
        status_parts: list[str] = []
        if deploy_sampling_passed:
            status_parts.append(
                "deploy_sampling=pass "
                f"dtype={str(dtype).removeprefix('torch.')} "
                f"steps={int(trainer.eval_inference_steps)}"
            )
        if interval_teacher_passed:
            status_parts.append(
                "interval_teacher=pass "
                "supports="
                f"{len(system.policy_config.flow_jepa_effective_interval_support_offsets)}"
            )
        if int(
            getattr(
                system.policy_config,
                "flow_jepa_complete_numerical_contract",
                0,
            )
        ):
            status_parts.append(
                "numerical_contract=pass "
                f"role_floor={float(system.policy_config.flow_jepa_routing_norm_floor):.3f} "
                f"corr_floor={float(system.policy_config.flow_jepa_correlation_rms_floor):.3f} "
                "visibility_width="
                f"{float(system.policy_config.flow_jepa_visibility_transition_fraction):.3f}"
            )
        print(
            f"[{preflight_version}-preflight] " + " ".join(status_parts),
            flush=True,
        )
    finally:
        restore_rng(saved_rng)
        system.train(was_training)


@torch.no_grad()
def _preflight_flow_jepa_stage1(
    *,
    system: V39PolicySystem,
    loader: DataLoader,
    conditioner: RDT2Conditioner,
    device: torch.device,
    dtype: torch.dtype,
    camera_names: Sequence[str],
    trainer: V39PolicyTrainerConfig,
) -> None:
    """Fail fast on the actual V95 Stage1 teacher path without deploy sampling."""

    saved_rng = rng_state()
    was_training = system.training
    try:
        system.eval()
        batch = next(iter(loader))
        sample = prepare_v39_policy_sample(
            batch,
            conditioner=conditioner,
            system=system,
            camera_names=camera_names,
            device=device,
            dtype=dtype,
            include_target_visual=True,
        )
        with autocast_context(device, dtype):
            output = system.flow_jepa_stage1_forward(
                sample["visual"],
                sample["history_state"],
                sample["executed_action_history"],
                sample["state"],
                sample["target_visual"],
                raw_visual=sample.get("raw_visual"),
            )
            losses = flow_jepa_stage1_losses(output, trainer, enable_future_loss=True)
        if not bool(torch.isfinite(losses["loss"]).all()):
            raise FloatingPointError("non-finite V95 Stage1 representation objective")
        print(
            f"[v95-stage1-preflight] representation_forward=pass "
            f"dtype={str(dtype).removeprefix('torch.')} teacher=frozen_dino_no_grad",
            flush=True,
        )
    finally:
        restore_rng(saved_rng)
        system.train(was_training)


def _bytes_to_gib(value: int | float) -> float:
    return float(value) / float(1024**3)


class CudaMemoryReporter:
    """Small built-in CUDA memory profiler for V38.5 training/eval.

    It records PyTorch allocator state, not an operator-level trace. It is
    cheap enough to keep disabled by default and turn on for short smoke or
    formal runs when diagnosing memory pressure.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        out_dir: Path,
        every: int = 0,
        detail: int = 0,
        sync: int = 0,
    ) -> None:
        self.device = device
        self.enabled = bool(device.type == "cuda" and int(every) > 0 and torch.cuda.is_available())
        self.every = max(int(every), 0)
        self.detail = bool(int(detail))
        self.sync = bool(int(sync))
        self.trace_path = out_dir / "v38_cuda_memory.jsonl"
        if self.enabled:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self.trace_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "schema": "clearvla-v38-cuda-memory-trace-v1",
                            "event": "start",
                            "variant": "v39_staged_midcut_contract",
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )

    def should_report(self, batch_index: int) -> bool:
        return self.enabled and self.every > 0 and int(batch_index) % self.every == 0

    def reset_peak(self) -> None:
        if self.enabled:
            torch.cuda.reset_peak_memory_stats(self.device)

    def snapshot(
        self,
        *,
        tag: str,
        epoch: int | None = None,
        batch: int | None = None,
        global_step: int | None = None,
        phase: str = "train",
        print_line: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {}
        if self.sync:
            torch.cuda.synchronize(self.device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
        row: dict[str, Any] = {
            "schema": "clearvla-v38-cuda-memory-trace-v1",
            "variant": "v39_staged_midcut_contract",
            "phase": phase,
            "tag": tag,
            "epoch": epoch,
            "batch": batch,
            "global_step": global_step,
            "allocated_gib": _bytes_to_gib(torch.cuda.memory_allocated(self.device)),
            "reserved_gib": _bytes_to_gib(torch.cuda.memory_reserved(self.device)),
            "max_allocated_gib": _bytes_to_gib(torch.cuda.max_memory_allocated(self.device)),
            "max_reserved_gib": _bytes_to_gib(torch.cuda.max_memory_reserved(self.device)),
            "free_gib": _bytes_to_gib(free_bytes),
            "total_gib": _bytes_to_gib(total_bytes),
        }
        row["used_by_context_gib"] = row["total_gib"] - row["free_gib"]
        if extra:
            row.update(extra)
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(jsonable(row), separators=(",", ":")) + "\n")
        if print_line:
            print(
                f"[cuda-mem] phase={phase} tag={tag} epoch={epoch} batch={batch} step={global_step} "
                f"alloc={row['allocated_gib']:.3f}GiB reserved={row['reserved_gib']:.3f}GiB "
                f"peak_alloc={row['max_allocated_gib']:.3f}GiB peak_reserved={row['max_reserved_gib']:.3f}GiB "
                f"ctx_used={row['used_by_context_gib']:.3f}/{row['total_gib']:.3f}GiB",
                flush=True,
            )
        return row


def _detached_scalar_metric(key: str, value: Tensor) -> Tensor:
    detached = value.detach().float()
    if detached.numel() != 1:
        raise ValueError(
            f"metric {key!r} must contain exactly one element; got shape={tuple(detached.shape)}"
        )
    return detached.reshape(())


def _accumulate_metric_tensors(
    acc: dict[str, Tensor],
    losses: dict[str, Tensor],
    *,
    counts: dict[str, int] | None = None,
    grad: Tensor | float | None = None,
) -> None:
    for key, value in losses.items():
        if not torch.is_tensor(value):
            continue
        detached = _detached_scalar_metric(key, value)
        acc[key] = (
            acc.get(key, torch.zeros((), device=detached.device, dtype=torch.float32)) + detached
        )
        if counts is not None:
            counts[key] = counts.get(key, 0) + 1
    if grad is not None:
        g = (
            _detached_scalar_metric("grad", grad)
            if torch.is_tensor(grad)
            else torch.tensor(float(grad))
        )
        acc["grad"] = acc.get("grad", torch.zeros((), device=g.device, dtype=torch.float32)) + g
        if counts is not None:
            counts["grad"] = counts.get("grad", 0) + 1


def _finalize_metric_tensors(
    acc: dict[str, Tensor],
    count: int,
    *,
    counts: dict[str, int] | None = None,
) -> dict[str, float]:
    if count <= 0:
        return {}
    return {
        key: float(
            (
                _detached_scalar_metric(key, value)
                / float(
                    max(
                        counts.get(key, count) if counts is not None else count,
                        1,
                    )
                )
            ).cpu()
        )
        for key, value in acc.items()
    }


def _sync_loss_row(
    losses: dict[str, Tensor], *, grad: Tensor | float | None = None
) -> dict[str, float]:
    row = {
        key: float(_detached_scalar_metric(key, value).cpu())
        for key, value in losses.items()
        if torch.is_tensor(value)
    }
    if grad is not None:
        row["grad"] = (
            float(_detached_scalar_metric("grad", grad).cpu())
            if torch.is_tensor(grad)
            else float(grad)
        )
    return row


@torch.no_grad()
def _attach_intent_frame_progress_audit(
    losses: dict[str, Tensor],
    sample: dict[str, Tensor],
    output: dict[str, Tensor],
) -> None:
    """Compare S progress with factual episode position without training on it."""

    frame_progress = sample.get("frame_progress")
    if not torch.is_tensor(frame_progress):
        return
    frame = frame_progress.detach().float().reshape(-1)
    intent_progress = output.get(
        "flow_jepa_intent_progress_coordinate_per_sample",
        output.get("flow_jepa_intent_progress_coordinate"),
    )
    if not torch.is_tensor(intent_progress):
        return
    intent = intent_progress.detach().float()
    if int(intent.shape[0]) != int(frame.shape[0]):
        return
    intent = intent.reshape(int(frame.shape[0]), -1).mean(dim=-1)
    gap = intent - frame
    # All three values therefore use the exact same diagnostic samples.
    losses["flow_jepa_frame_progress"] = frame.mean()
    losses["flow_jepa_intent_frame_progress_gap"] = gap.mean()
    losses["flow_jepa_intent_frame_progress_mae"] = gap.abs().mean()


def _attach_v94_loss_ledger(
    losses: dict[str, Tensor],
    trainer: V39PolicyTrainerConfig,
    *,
    enable_future_loss: bool,
    layer_aux_contribution: Tensor | None = None,
) -> None:
    """Attach an exact, detached ledger for the active Evidence objective.

    Raw losses alone are not comparable because every term has a different
    effective weight.  The ledger records weighted contributions and a signed
    residual against the actual scalar sent to backward.  A non-trivial
    residual is therefore an immediate signal that a new objective was added
    without being made observable.
    """

    reference = losses["loss"].detach().float().reshape(())
    grouped: dict[str, list[Tensor]] = {
        "action": [],
        "rollout": [],
        "execution": [],
        "latent": [],
        "layer": [],
        "representation": [],
    }

    def add(name: str, metric: str, weight: float, group: str) -> None:
        value = losses.get(metric)
        effective_weight = float(weight)
        if effective_weight <= 0.0 or not torch.is_tensor(value) or value.numel() != 1:
            return
        contribution = value.detach().float().reshape(()) * effective_weight
        losses[f"loss_contrib_{name}"] = contribution
        grouped[group].append(contribution)

    add("flow", "physical_flow", 1.0, "action")
    for name, metric, weight_name in (
        ("flow_jepa_warp_loss", "flow_jepa_warp_loss", "flow_jepa_warp_loss_weight"),
        (
            "flow_jepa_identity_advantage_loss",
            "flow_jepa_identity_advantage_loss",
            "flow_jepa_identity_advantage_loss_weight",
        ),
        (
            "flow_jepa_static_identity_loss",
            "flow_jepa_static_identity_loss",
            "flow_jepa_static_identity_loss_weight",
        ),
        ("flow_jepa_cycle_loss", "flow_jepa_cycle_loss", "flow_jepa_cycle_loss_weight"),
        (
            "flow_jepa_smoothness_loss",
            "flow_jepa_smoothness_loss",
            "flow_jepa_smoothness_loss_weight",
        ),
        (
            "flow_jepa_uncertainty_nll",
            "flow_jepa_uncertainty_nll",
            "flow_jepa_uncertainty_nll_weight",
        ),
        (
            "flow_jepa_refinement_sequence_loss",
            "flow_jepa_refinement_sequence_loss",
            "flow_jepa_refinement_sequence_loss_weight",
        ),
    ):
        add(name, metric, float(getattr(trainer, weight_name, 0.0)), "representation")
    for name, metric, weight_name in (
        ("proposal", "proposal", "proposal_loss_weight"),
        ("event", "event", "event_loss_weight"),
        ("motion", "motion", "arm_motion_loss_weight"),
        ("gripper_transition", "transition_l1", "gripper_transition_l1_weight"),
        ("smooth_delta", "smooth_delta", "smooth_delta_weight"),
        ("decoded_action", "decoded_action", "decoded_action_loss_weight"),
        (
            "physical_delta_consistency",
            "physical_delta_consistency",
            "physical_delta_consistency_weight",
        ),
        (
            "transition_gripper_flow",
            "transition_gripper_flow",
            "transition_gripper_flow_weight",
        ),
        (
            "event_delta_consistency",
            "event_delta_consistency",
            "event_delta_consistency_weight",
        ),
        ("event_magnitude", "event_magnitude", "event_magnitude_weight"),
        ("event_off_delta", "event_off_delta", "event_off_delta_weight"),
    ):
        add(name, metric, float(getattr(trainer, weight_name, 0.0)), "action")

    if enable_future_loss:
        add(
            "flow_jepa_future",
            "flow_jepa_future_prediction",
            float(getattr(trainer, "flow_jepa_future_loss_weight", 0.0)),
            "representation",
        )
        add(
            "flow_jepa_future_change",
            "flow_jepa_future_change",
            float(getattr(trainer, "flow_jepa_future_change_loss_weight", 0.0)),
            "representation",
        )
        add(
            "flow_jepa_horizon_address",
            "flow_jepa_horizon_address",
            float(
                getattr(
                    trainer,
                    "flow_jepa_horizon_address_loss_weight",
                    0.0,
                )
            ),
            "representation",
        )
        add(
            "flow_jepa_interval_stage",
            "flow_jepa_interval_stage",
            float(
                getattr(
                    trainer,
                    "flow_jepa_interval_stage_loss_weight",
                    0.0,
                )
            ),
            "representation",
        )
        add(
            "flow_jepa_stage",
            "flow_jepa_stage_prediction",
            float(getattr(trainer, "flow_jepa_stage_loss_weight", 0.0)),
            "representation",
        )
        for name, metric, weight_name in (
            ("rollout_dynamics", "rollout_dynamics", "rollout_dynamics_loss_weight"),
            ("rollout_delta", "rollout_delta", "rollout_delta_loss_weight"),
            ("rollout_contrast", "rollout_contrast", "rollout_contrast_loss_weight"),
            ("rollout_variance", "rollout_variance", "rollout_variance_loss_weight"),
            ("rollout_norm", "rollout_norm", "rollout_norm_loss_weight"),
            (
                "rollout_milestone",
                "rollout_milestone_delta_match",
                "rollout_milestone_delta_match_weight",
            ),
        ):
            add(name, metric, float(getattr(trainer, weight_name, 0.0)), "rollout")
        # Compatibility knobs intentionally reuse the dynamics target.  Keep
        # their contributions separately named if an old experiment enables them.
        add(
            "future_latent_compat",
            "rollout_dynamics",
            float(getattr(trainer, "future_latent_loss_weight", 0.0)),
            "rollout",
        )
        add(
            "action_effect_compat",
            "rollout_dynamics",
            float(getattr(trainer, "action_effect_loss_weight", 0.0)),
            "rollout",
        )
        add(
            "execution_value",
            "evidence_mmd_it_execution_value_loss",
            float(getattr(trainer, "latent_cvae_mmdit_execution_value_loss_weight", 0.0)),
            "execution",
        )

    add(
        "latent_kl",
        "latent_cvae_kl",
        float(getattr(trainer, "latent_cvae_kl_weight", 0.0)),
        "latent",
    )
    add(
        "latent_posterior_recon",
        "latent_cvae_posterior_recon",
        float(getattr(trainer, "latent_cvae_posterior_recon_weight", 0.0)),
        "latent",
    )
    add(
        "latent_adaptive_regularizer",
        "latent_cvae_adaptive_regularizer",
        float(getattr(trainer, "latent_cvae_adaptive_regularizer_weight", 0.0)),
        "latent",
    )
    add(
        "latent_route_entropy",
        "latent_cvae_adaptive_route_entropy_regularizer",
        float(getattr(trainer, "latent_cvae_adaptive_route_entropy_weight", 0.0)),
        "latent",
    )
    legacy_anchor = losses.get("latent_cvae_legacy_anchor")
    legacy_anchor_weight = losses.get("latent_cvae_legacy_anchor_weight")
    if (
        torch.is_tensor(legacy_anchor)
        and legacy_anchor.numel() == 1
        and torch.is_tensor(legacy_anchor_weight)
        and legacy_anchor_weight.numel() == 1
        and float(legacy_anchor_weight.detach().float()) > 0.0
    ):
        contribution = legacy_anchor.detach().float().reshape(
            ()
        ) * legacy_anchor_weight.detach().float().reshape(())
        losses["loss_contrib_latent_legacy_anchor"] = contribution
        grouped["latent"].append(contribution)

    if torch.is_tensor(layer_aux_contribution):
        contribution = layer_aux_contribution.detach().float().reshape(())
        losses["loss_contrib_layer_contract"] = contribution
        grouped["layer"].append(contribution)

    group_values: list[Tensor] = []
    zero = torch.zeros_like(reference)
    for group, contributions in grouped.items():
        if not contributions:
            continue
        group_value = torch.stack(contributions).sum()
        losses[f"loss_group_{group}"] = group_value
        group_values.append(group_value)
    ledger_sum = torch.stack(group_values).sum() if group_values else zero
    losses["loss_ledger_sum"] = ledger_sum
    losses["loss_ledger_residual"] = reference - ledger_sum


_FLOW_JEPA_LOG_VERSIONS = frozenset(
    {
        "v95",
        "v96",
        "v97",
        "v98",
        "v99",
        "v100",
        "v101",
        "v102",
        "v103",
        "v104",
        "v105",
        "v106",
        "v107",
        "v108",
        "v109",
        "v110",
        "v111",
        "v112",
        "v113",
        "v114",
        "v115",
        "v116",
        "v117",
        "v118",
        "v119",
        "v120",
        "v121",
    }
)
_RAW_FLOW_JEPA_LOG_VERSIONS = frozenset(
    {
        "v97",
        "v98",
        "v99",
        "v100",
        "v101",
        "v102",
        "v103",
        "v104",
        "v105",
        "v106",
        "v107",
        "v108",
        "v109",
        "v110",
        "v111",
        "v112",
        "v113",
        "v114",
        "v115",
        "v116",
        "v117",
        "v118",
        "v119",
        "v120",
        "v121",
    }
)
_COMPLEMENTARY_FLOW_JEPA_LOG_VERSIONS = frozenset(
    {
        "v100",
        "v101",
        "v102",
        "v103",
        "v104",
        "v105",
        "v106",
        "v107",
        "v108",
        "v109",
        "v110",
        "v111",
        "v112",
        "v113",
        "v114",
        "v115",
        "v116",
        "v117",
        "v118",
        "v119",
        "v120",
        "v121",
    }
)
_BALANCED_FLOW_JEPA_LOG_VERSIONS = frozenset(
    {
        "v101",
        "v102",
        "v103",
        "v104",
        "v105",
        "v106",
        "v107",
        "v108",
        "v109",
        "v110",
        "v111",
        "v112",
        "v113",
        "v114",
        "v115",
        "v116",
        "v117",
        "v118",
        "v119",
        "v120",
        "v121",
    }
)


def _evidence_log_version(*rows: dict[str, float]) -> str:
    def maximum(*keys: str) -> float:
        return max(
            (float(row.get(key, 0.0)) for row in rows for key in keys),
            default=0.0,
        )

    if maximum("object_intent_dynamics_active") > 0.5:
        return "v121"
    if maximum("grounded_intent_effect_active") > 0.5:
        return "v119"
    if maximum("flow_jepa_differential_effect_bank_active") > 0.5:
        return "v118"
    if maximum("flow_jepa_stateless_intent_controller_active") > 0.5:
        return "v117"
    if maximum("flow_jepa_supervised_effect_mainline_active") > 0.5:
        return "v116"
    if (
        maximum(
            "flow_jepa_policy_plan_compiler_active",
            "flow_jepa_goal_phase_machine_active",
            "flow_jepa_future_effect_field_active",
        )
        > 0.5
    ):
        return "v115"
    if (
        maximum(
            "flow_jepa_p1_shared_factual",
            "flow_jepa_typed_p2_utility_precision",
        )
        > 0.5
    ):
        return "v114"
    if maximum("flow_jepa_functional_mainline_routing") > 0.5:
        return "v113"
    if maximum("flow_jepa_pre_value_owner_routing") > 0.5:
        return "v112"
    if maximum("flow_jepa_structured_ownership_bottleneck") > 0.5:
        return "v111"
    if maximum("flow_jepa_coordinate_typed_raw_detail") > 0.5:
        return "v110"
    if maximum("flow_jepa_progressive_grounding_address") > 0.5:
        return "v109"
    if maximum("flow_jepa_online_horizon_address") > 0.5:
        return "v108"
    if all(
        maximum(key) > 0.5
        for key in (
            "flow_jepa_policy_multi_glimpse_address",
            "flow_jepa_horizon_cell_fine_address",
            "flow_jepa_interval_stage_typed_value",
            "role_residual_contract_after_gate",
        )
    ):
        return "v107"
    if all(
        maximum(key) > 0.5
        for key in (
            "flow_jepa_interval_stage_enabled",
            "flow_jepa_variance_safe_routing",
            "flow_jepa_complete_numerical_contract",
        )
    ):
        return "v106"
    if all(
        maximum(key) > 0.5
        for key in (
            "flow_jepa_bounded_flow_coordinates",
            "flow_jepa_sequential_horizon_memory",
            "role_residual_contract_enabled",
            "flow_jepa_horizon_soft_address",
            "flow_jepa_future_reliable_normalization",
            "flow_jepa_horizon_address_supervision_active",
        )
    ):
        return "v105"
    if all(
        maximum(key) > 0.5
        for key in (
            "flow_jepa_bounded_flow_coordinates",
            "flow_jepa_sequential_horizon_memory",
            "role_residual_contract_enabled",
        )
    ):
        return "v104"
    if (
        maximum(
            "flow_jepa_predictive_change_contract",
            "flow_jepa_soft_address_lattice",
            "evidence_policy_delta_bridge_enabled",
        )
        > 0.5
    ):
        return "v103"
    if (
        maximum(
            "flow_jepa_raw_detail_deferred_to_policy",
            "flow_jepa_world_anchor_write_only",
            "evidence_top_policy_workspace_horizon_pool",
        )
        > 0.5
    ):
        return "v102"
    if (
        maximum(
            "temporal_balance_active",
            "flow_jepa_teacher_balanced_target_mask",
            "evidence_top_policy_workspace_fixed_fusion",
        )
        > 0.5
    ):
        return "v101"
    if (
        maximum(
            "flow_jepa_strict_role_visual_path",
            "flow_jepa_raw_additive_detail_path",
        )
        > 0.5
    ):
        return "v100"
    if maximum("flow_jepa_zero_flow_guard") > 0.5:
        return "v99"
    if maximum("flow_jepa_raw_image_enabled") > 0.5:
        return "v98"
    if maximum("flow_jepa_late_bottleneck") > 0.5:
        return "v96"
    if any(key.startswith("flow_jepa_") for row in rows for key in row):
        return "v95"
    return "v94"


def _evidence_serial_log_line(
    row: dict[str, float],
    *,
    epoch: int,
    batch_index: int,
    learning_rate: float,
    seconds_per_batch: float,
) -> str:
    """Compact, active-branch batch log for the V94 Evidence decoder.

    Optional values are emitted only when the forward/loss path actually
    produced them.  A present zero gradient or invariant is retained because
    it is evidence; fields from inactive decoder families are never fabricated.
    """

    if "role_residual_written_rms" not in row and "role_residual_bounded_rms" in row:
        # V104-V106 reported the pre-gate bounded tensor under the historical
        # write label.  Preserve old-log readability while V107 prefers the
        # factual post-gate write metric whenever it exists.
        row = {
            **row,
            "role_residual_written_rms": row["role_residual_bounded_rms"],
        }

    def append(
        parts: list[str],
        label: str,
        key: str,
        spec: str = ".4f",
        *,
        keep_zero: bool = False,
    ) -> None:
        if key in row and (keep_zero or abs(float(row[key])) > 1e-12):
            parts.append(f"{label}={format(row[key], spec)}")

    log_version = _evidence_log_version(row)
    loss_parts = [
        f"[{log_version}-train] epoch={epoch:03d}",
        f"batch={batch_index:04d}",
        f"loss_total={row['loss']:.6f}",
        f"flow_loss={row['physical_flow']:.6f}",
    ]
    if log_version in {"v116", "v117", "v118"}:
        # These are semantic aliases over the actual objective tensors.  Keep
        # the ancestral fields below for cross-version parsers, while making
        # the V116 action ledger readable without knowing the old shorthand.
        for label, key, spec in (
            ("native_velocity_mse", "native_velocity_mse", ".6f"),
            ("arm_tangent_mse", "arm_tangent_mse", ".5f"),
            ("arm_null_mse", "arm_null_mse", ".5f"),
            ("gripper_tangent_mse", "gripper_tangent_mse", ".5f"),
            ("gripper_null_mse", "gripper_null_mse", ".5f"),
            ("event_reweight_delta", "event_reweight_delta", "+.3e"),
        ):
            append(loss_parts, label, key, spec, keep_zero=True)
    for label, key, spec in (
        ("native_flow", "physical_flow_native", ".6f"),
        ("arm_flow", "arm_fm_per_dim", ".5f"),
        ("grip_flow", "gripper_fm_field", ".5f"),
        ("decode_loss", "decoded_action", ".6f"),
        ("flow_first8", "first8_physical_flow", ".6f"),
        ("flow_tail", "tail_physical_flow", ".6f"),
        ("event_loss", "event", ".5f"),
        ("motion_loss", "motion", ".5f"),
        ("proposal_loss", "proposal", ".5f"),
        ("rollout_loss", "rollout_dynamics", ".5f"),
        ("rollout_step", "rollout_milestone_delta_match", ".5f"),
        ("rollout_contrast", "rollout_contrast", ".5f"),
        ("rollout_std_ratio", "rollout_pred_std_ratio", ".3f"),
        ("step_norm_ratio", "rollout_milestone_delta_norm_ratio", ".3f"),
    ):
        append(loss_parts, label, key, spec)
    group_keys = ("action", "representation", "rollout", "execution", "latent", "layer")
    present_groups = [
        name
        for name in group_keys
        if f"loss_group_{name}" in row and abs(float(row[f"loss_group_{name}"])) > 1e-12
    ]
    if present_groups:
        loss_parts.append(
            "loss_groups="
            + "/".join(f"{name}:{row[f'loss_group_{name}']:.5f}" for name in present_groups)
        )
    contribution_keys = [
        key for key in row if key.startswith("loss_contrib_") and abs(float(row[key])) > 1e-12
    ]
    if contribution_keys:
        ranked_contributions = sorted(
            contribution_keys,
            key=lambda key: abs(float(row[key])),
            reverse=True,
        )[:6]
        loss_parts.append(
            "top_contrib="
            + "/".join(
                f"{key.removeprefix('loss_contrib_')}:{row[key]:.5f}"
                for key in ranked_contributions
            )
        )
    append(loss_parts, "ledger_gap", "loss_ledger_residual", "+.2e")

    representation_parts: list[str] = []
    if log_version in _FLOW_JEPA_LOG_VERSIONS:
        representation_parts.append(f"[{log_version}-repr]")
        future_prediction_label = "future_pred" if log_version != "v95" else "window_pred"
        for label, key, spec in (
            (future_prediction_label, "flow_jepa_future_prediction", ".5f"),
            ("change_dir", "flow_jepa_future_change_direction", ".5f"),
            ("change_obj", "flow_jepa_future_change", ".5f"),
            ("stage_pred", "flow_jepa_stage_prediction", ".5f"),
            ("warp", "flow_jepa_warp_loss", ".5f"),
            ("identity_adv", "flow_jepa_identity_advantage_loss", ".5f"),
            ("static_identity", "flow_jepa_static_identity_loss", ".5f"),
            ("cycle", "flow_jepa_cycle_loss", ".5f"),
            ("smooth", "flow_jepa_smoothness_loss", ".5f"),
            ("uncert_nll", "flow_jepa_uncertainty_nll", ".5f"),
            ("refine_seq", "flow_jepa_refinement_sequence_loss", ".5f"),
            ("flow_mag", "flow_jepa_patch_flow_magnitude", ".3f"),
            ("confidence", "flow_jepa_confidence_mean", ".3f"),
            ("occlusion", "flow_jepa_occlusion_fraction", ".3f"),
            ("corr_entropy", "flow_jepa_correlation_entropy", ".3f"),
            ("corr_margin", "flow_jepa_correlation_margin", ".3f"),
            ("context_drop", "flow_jepa_context_dropout_fraction", ".3f"),
            ("target_mask", "flow_jepa_future_target_fraction", ".3f"),
            ("window_hmax", "flow_jepa_window_horizon_max", ".0f"),
            ("stage_h", "flow_jepa_stage_horizon", ".0f"),
            ("stage_norm", "flow_jepa_stage_token_norm", ".3f"),
            ("stage_target_norm", "flow_jepa_stage_target_norm", ".3f"),
            ("stage_prediction_norm", "flow_jepa_stage_prediction_norm", ".3f"),
            ("stage_window_cos", "flow_jepa_stage_window_cosine", ".3f"),
            ("stage_window_gate", "flow_jepa_stage_to_window_gate", ".3f"),
            ("stage_window_update", "flow_jepa_stage_to_window_update_norm", ".3f"),
            ("goal_norm", "flow_jepa_goal_condition_norm", ".3f"),
            ("goal_pair_cos", "flow_jepa_goal_pair_cosine", ".3f"),
            ("action_mem_norm", "flow_jepa_action_condition_norm", ".3f"),
            ("goal_action_cos", "flow_jepa_goal_action_cosine", ".3f"),
            (
                "future_raw_delta",
                "flow_jepa_future_raw_delta_loss",
                ".5f",
            ),
            (
                "future_reliable_norm",
                "flow_jepa_future_reliable_normalized_loss",
                ".5f",
            ),
            (
                "future_reliability",
                "flow_jepa_future_change_reliability",
                ".3f",
            ),
            (
                "future_reference_scale",
                "flow_jepa_future_current_reference_scale",
                ".3f",
            ),
            (
                "future_normalization_scale",
                "flow_jepa_future_normalization_scale",
                ".3f",
            ),
            (
                "future_direction_floor",
                "flow_jepa_future_direction_floor_min",
                ".3e",
            ),
            (
                "horizon_address_loss",
                "flow_jepa_horizon_address",
                ".5f",
            ),
            (
                "horizon_address_teacher_rel",
                "flow_jepa_horizon_address_teacher_reliability",
                ".3f",
            ),
            (
                "horizon_address_teacher_entropy",
                "flow_jepa_horizon_address_teacher_entropy",
                ".3f",
            ),
            (
                "horizon_address_pred_entropy",
                "flow_jepa_horizon_address_predicted_entropy",
                ".3f",
            ),
            (
                "horizon_address_update",
                "flow_jepa_horizon_address_update_rms",
                ".3f",
            ),
            (
                "horizon_address_ratio",
                "flow_jepa_horizon_address_update_ratio",
                ".3f",
            ),
            (
                "horizon_address_route_entropy",
                "flow_jepa_horizon_address_route_entropy",
                ".3f",
            ),
            (
                "horizon_address_route_max",
                "flow_jepa_horizon_address_route_max",
                ".3f",
            ),
            (
                "horizon_address_fine_entropy",
                "flow_jepa_horizon_address_fine_entropy",
                ".3f",
            ),
            (
                "horizon_address_variation",
                "flow_jepa_horizon_address_variation",
                ".3f",
            ),
            (
                "horizon_address_cross_cell",
                "flow_jepa_horizon_address_cross_cell_distance",
                ".3f",
            ),
            (
                "horizon_fine_cell_specific",
                "flow_jepa_horizon_cell_fine_address",
                ".0f",
            ),
            (
                "online_horizon_address",
                "flow_jepa_online_horizon_address",
                ".0f",
            ),
            (
                "online_address_write",
                "flow_jepa_online_horizon_address_write_rms",
                ".3f",
            ),
            (
                "progressive_address",
                "flow_jepa_progressive_grounding_address",
                ".0f",
            ),
            (
                "g1_coarse_entropy",
                "flow_jepa_progressive_g1_coarse_entropy",
                ".3f",
            ),
            (
                "g1_coarse_max",
                "flow_jepa_progressive_g1_coarse_max",
                ".3f",
            ),
            (
                "g2_fine_entropy",
                "flow_jepa_progressive_g2_fine_entropy",
                ".3f",
            ),
            (
                "g2_center_shift",
                "flow_jepa_progressive_g2_center_shift",
                ".3f",
            ),
            (
                "g3_coarse_prior",
                "flow_jepa_progressive_g3_coarse_bias_rms",
                ".3f",
            ),
            (
                "g3_summary",
                "flow_jepa_progressive_g3_summary_rms",
                ".3f",
            ),
            (
                "g3_sem_summary",
                "flow_jepa_progressive_g3_semantic_summary_rms",
                ".3f",
            ),
            (
                "g3_app_summary",
                "flow_jepa_progressive_g3_appearance_summary_rms",
                ".3f",
            ),
            (
                "g3_geo_summary",
                "flow_jepa_progressive_g3_geometry_summary_rms",
                ".3f",
            ),
            (
                "world_address_entropy",
                "flow_jepa_progressive_world_posterior_entropy",
                ".3f",
            ),
            (
                "world_horizon_variation",
                "flow_jepa_progressive_world_horizon_variation",
                ".3f",
            ),
            (
                "world_source_max",
                "flow_jepa_progressive_world_source_prior_max",
                ".3f",
            ),
            (
                "world_source_variation",
                "flow_jepa_progressive_world_source_horizon_variation",
                ".3f",
            ),
            (
                "policy_address_prior",
                "flow_jepa_progressive_policy_prior_active",
                ".0f",
            ),
            (
                "policy_world_prior",
                "flow_jepa_progressive_policy_world_prior_rms",
                ".3f",
            ),
            ("typed_raw", "flow_jepa_coordinate_typed_raw_detail", ".0f"),
            (
                "structured_ownership",
                "flow_jepa_structured_ownership_bottleneck",
                ".0f",
            ),
            (
                "pre_value_owner",
                "flow_jepa_pre_value_owner_routing",
                ".0f",
            ),
            (
                "functional_mainline",
                "flow_jepa_functional_mainline_routing",
                ".0f",
            ),
            (
                "g3_query_private_cos",
                "flow_jepa_progressive_g3_query_private_cosine",
                ".3f",
            ),
            (
                "g2_owner_sem_app_l1",
                "flow_jepa_progressive_g2_semantic_appearance_posterior_l1",
                ".3f",
            ),
            (
                "g2_owner_app_geo_l1",
                "flow_jepa_progressive_g2_appearance_geometry_posterior_l1",
                ".3f",
            ),
            (
                "g3_owner_sem_app_l1",
                "flow_jepa_progressive_g3_semantic_appearance_slot_l1",
                ".3f",
            ),
            (
                "g3_sem_owner_rms",
                "flow_jepa_progressive_g3_semantic_owner_sidecar_rms",
                ".3f",
            ),
            (
                "g3_app_owner_rms",
                "flow_jepa_progressive_g3_appearance_owner_sidecar_rms",
                ".3f",
            ),
            (
                "g3_geo_owner_rms",
                "flow_jepa_progressive_g3_geometry_owner_sidecar_rms",
                ".3f",
            ),
            (
                "g3_public_tokens",
                "flow_jepa_progressive_g3_summary_token_count",
                ".0f",
            ),
            (
                "g3_owner_tokens",
                "flow_jepa_progressive_g3_owner_sidecar_token_count",
                ".0f",
            ),
            (
                "world_public_ratio",
                "flow_jepa_progressive_world_public_ratio",
                ".3f",
            ),
            (
                "world_public_private_ratio",
                "flow_jepa_progressive_world_public_private_ratio",
                ".3f",
            ),
            (
                "world_private_rms",
                "flow_jepa_progressive_world_private_state_rms",
                ".3f",
            ),
            (
                "w0_app_state",
                "flow_jepa_pre_value_w0_appearance_state_rms",
                ".3f",
            ),
            (
                "w1_app_state",
                "flow_jepa_pre_value_w1_appearance_state_rms",
                ".3f",
            ),
            (
                "w2_app_state",
                "flow_jepa_pre_value_w2_appearance_state_rms",
                ".3f",
            ),
            (
                "w3_app_state",
                "flow_jepa_pre_value_w3_appearance_state_rms",
                ".3f",
            ),
            (
                "w3_sem_state",
                "flow_jepa_pre_value_w3_semantic_state_rms",
                ".3f",
            ),
            (
                "w3_geo_state",
                "flow_jepa_pre_value_w3_geometry_state_rms",
                ".3f",
            ),
            (
                "w3_interval_state",
                "flow_jepa_pre_value_w3_interval_state_rms",
                ".3f",
            ),
            (
                "w0_carrier_ratio",
                "flow_jepa_pre_value_w0_carrier_ratio",
                ".3f",
            ),
            (
                "w1_carrier_ratio",
                "flow_jepa_pre_value_w1_carrier_ratio",
                ".3f",
            ),
            (
                "w2_carrier_ratio",
                "flow_jepa_pre_value_w2_carrier_ratio",
                ".3f",
            ),
            (
                "w3_carrier_ratio",
                "flow_jepa_pre_value_w3_carrier_ratio",
                ".3f",
            ),
            (
                "p1_app_prior",
                "flow_jepa_typed_p1_appearance_pre_value_prior_rms",
                ".3f",
            ),
            (
                "p1_w_app_candidate",
                "flow_jepa_typed_p1_world_appearance_candidate_logit_rms",
                ".3f",
            ),
            (
                "p1_app_gateway_query",
                "flow_jepa_typed_p1_appearance_gateway_query_rms",
                ".3f",
            ),
            ("p1_query_rows", "flow_jepa_p1_query_rows", ".0f"),
            ("p2_query_rows", "flow_jepa_p2_query_rows", ".0f"),
            (
                "p1_query_chunk",
                "flow_jepa_address_query_chunk_actual",
                ".0f",
            ),
            (
                "p1_checkpoint_configured",
                "flow_jepa_typed_p1_activation_checkpoint",
                ".0f",
            ),
            (
                "p1_checkpoint_active",
                "flow_jepa_typed_p1_activation_checkpoint_active",
                ".0f",
            ),
            (
                "w0_owner_route_entropy",
                "flow_jepa_functional_w0_route_entropy",
                ".3f",
            ),
            (
                "w0_owner_route_null",
                "flow_jepa_functional_w0_route_null_mass",
                ".3f",
            ),
            (
                "w3_owner_route_entropy",
                "flow_jepa_functional_w3_route_entropy",
                ".3f",
            ),
            (
                "w3_owner_route_null",
                "flow_jepa_functional_w3_route_null_mass",
                ".3f",
            ),
            (
                "w3_owner_route_sem",
                "flow_jepa_functional_w3_semantic_route_mass",
                ".3f",
            ),
            (
                "w3_owner_route_app",
                "flow_jepa_functional_w3_appearance_route_mass",
                ".3f",
            ),
            (
                "w3_owner_route_geo",
                "flow_jepa_functional_w3_geometry_route_mass",
                ".3f",
            ),
            (
                "w3_owner_route_interval",
                "flow_jepa_functional_w3_interval_route_mass",
                ".3f",
            ),
            (
                "world_innovation",
                "flow_jepa_progressive_world_horizon_innovation_rms",
                ".3f",
            ),
            (
                "world_owner_sem_geo_l1",
                "flow_jepa_progressive_world_semantic_geometry_source_l1",
                ".3f",
            ),
            (
                "world_owner_slot_contract_min",
                "flow_jepa_progressive_world_owner_slot_contract_min",
                ".3f",
            ),
            (
                "world_owner_source_contract_min",
                "flow_jepa_progressive_world_owner_source_contract_min",
                ".3f",
            ),
            (
                "p1_owner_fine_l1",
                "flow_jepa_typed_p1_appearance_geometry_fine_l1",
                ".3f",
            ),
            (
                "p1_owner_route_l1",
                "flow_jepa_typed_p1_semantic_appearance_route_l1",
                ".3f",
            ),
            ("literal_rgb", "flow_jepa_literal_rgb_chart_rms", ".3f"),
            (
                "future_transport_offset",
                "flow_jepa_progressive_future_transport_offset_rms",
                ".3f",
            ),
            (
                "future_visibility",
                "flow_jepa_progressive_future_transport_visibility_mean",
                ".3f",
            ),
            (
                "future_transport_variation",
                "flow_jepa_progressive_future_transport_horizon_variation",
                ".3f",
            ),
            (
                "effect_pred_cos",
                "flow_jepa_future_effect_pred_adjacent_cosine",
                ".3f",
            ),
            (
                "effect_target_cos",
                "flow_jepa_future_effect_target_adjacent_cosine",
                ".3f",
            ),
            (
                "effect_pred_var",
                "flow_jepa_future_effect_pred_interval_variation",
                ".3f",
            ),
            (
                "effect_target_var",
                "flow_jepa_future_effect_target_interval_variation",
                ".3f",
            ),
            (
                "effect_transport_pred_var",
                "flow_jepa_future_effect_pred_transport_variation",
                ".3f",
            ),
            (
                "effect_transport_target_var",
                "flow_jepa_future_effect_target_transport_variation",
                ".3f",
            ),
            (
                "effect_teacher_rel",
                "flow_jepa_future_effect_teacher_reliability_mean",
                ".3f",
            ),
            (
                "effect_teacher_entropy",
                "flow_jepa_future_effect_teacher_association_entropy",
                ".3f",
            ),
            (
                "effect_teacher_semantic_advantage",
                "flow_jepa_future_effect_teacher_semantic_advantage",
                ".3f",
            ),
            (
                "effect_semantic_loss",
                "flow_jepa_future_effect_semantic_loss",
                ".4f",
            ),
            (
                "effect_successor_loss",
                "flow_jepa_future_effect_successor_loss",
                ".4f",
            ),
            (
                "effect_semantic_near_loss",
                "flow_jepa_future_effect_semantic_near_loss",
                ".4f",
            ),
            (
                "effect_semantic_mid_loss",
                "flow_jepa_future_effect_semantic_mid_loss",
                ".4f",
            ),
            (
                "effect_semantic_late_loss",
                "flow_jepa_future_effect_semantic_late_loss",
                ".4f",
            ),
            (
                "effect_intent_summary_loss",
                "flow_jepa_future_effect_intent_summary_loss",
                ".4f",
            ),
            (
                "effect_near_contrib",
                "flow_jepa_future_effect_effective_near_loss",
                ".4f",
            ),
            (
                "effect_mid_contrib",
                "flow_jepa_future_effect_effective_mid_loss",
                ".4f",
            ),
            (
                "effect_late_contrib",
                "flow_jepa_future_effect_effective_late_loss",
                ".4f",
            ),
            (
                "effect_w1_current_loss",
                "flow_jepa_future_effect_w1_current_loss",
                ".4f",
            ),
            (
                "effect_w1_successor_loss",
                "flow_jepa_future_effect_w1_successor_loss",
                ".4f",
            ),
            (
                "effect_w1_semantic_loss",
                "flow_jepa_future_effect_w1_semantic_loss",
                ".4f",
            ),
            (
                "effect_w2_current_loss",
                "flow_jepa_future_effect_w2_current_loss",
                ".4f",
            ),
            (
                "effect_w2_successor_loss",
                "flow_jepa_future_effect_w2_successor_loss",
                ".4f",
            ),
            (
                "effect_w2_semantic_loss",
                "flow_jepa_future_effect_w2_semantic_loss",
                ".4f",
            ),
            (
                "effect_transport_loss",
                "flow_jepa_future_effect_transport_loss",
                ".4f",
            ),
            (
                "effect_cov_loss",
                "flow_jepa_future_effect_transport_covariance_loss",
                ".4f",
            ),
            (
                "effect_persist_loss",
                "flow_jepa_future_effect_persistence_loss",
                ".4f",
            ),
            (
                "effect_visible_loss",
                "flow_jepa_future_effect_visibility_loss",
                ".4f",
            ),
            (
                "effect_uncert_loss",
                "flow_jepa_future_effect_uncertainty_loss",
                ".4f",
            ),
            (
                "effect_relative_transition_loss",
                "flow_jepa_future_effect_relative_transition_loss",
                ".4f",
            ),
            (
                "w1_effect_cos",
                "flow_jepa_differential_w1_adjacent_cosine",
                ".3f",
            ),
            (
                "w1_effect_var",
                "flow_jepa_differential_w1_slot_variation",
                ".3f",
            ),
            (
                "w2_effect_cos",
                "flow_jepa_differential_w2_adjacent_cosine",
                ".3f",
            ),
            (
                "w2_effect_var",
                "flow_jepa_differential_w2_slot_variation",
                ".3f",
            ),
            (
                "effect_pred_near",
                "flow_jepa_differential_w2_near_effect_rms",
                ".3f",
            ),
            (
                "effect_pred_mid",
                "flow_jepa_differential_w2_mid_effect_rms",
                ".3f",
            ),
            (
                "effect_pred_late",
                "flow_jepa_differential_w2_late_effect_rms",
                ".3f",
            ),
            (
                "effect_target_near",
                "flow_jepa_future_effect_target_near_rms",
                ".3f",
            ),
            (
                "effect_target_mid",
                "flow_jepa_future_effect_target_mid_rms",
                ".3f",
            ),
            (
                "effect_target_late",
                "flow_jepa_future_effect_target_late_rms",
                ".3f",
            ),
            (
                "effect_rel_near",
                "flow_jepa_future_effect_teacher_reliability_near",
                ".3f",
            ),
            (
                "effect_rel_mid",
                "flow_jepa_future_effect_teacher_reliability_mid",
                ".3f",
            ),
            (
                "effect_rel_late",
                "flow_jepa_future_effect_teacher_reliability_late",
                ".3f",
            ),
            (
                "p2_diff_read",
                "flow_jepa_p2_effect_read_rms",
                ".3f",
            ),
            (
                "p2_diff_content_score",
                "flow_jepa_p2_effect_content_score_rms",
                ".3f",
            ),
            (
                "p2_diff_intent_score",
                "flow_jepa_p2_effect_intent_score_rms",
                ".3f",
            ),
            (
                "p2_diff_coordinate_score",
                "flow_jepa_p2_effect_coordinate_score_rms",
                ".3f",
            ),
            (
                "p2_diff_entropy",
                "flow_jepa_p2_effect_entropy",
                ".3f",
            ),
            (
                "p2_effect_read",
                "flow_jepa_p2_structured_effect_read_rms",
                ".3f",
            ),
            (
                "p2_effect_entropy",
                "flow_jepa_p2_structured_effect_entropy",
                ".3f",
            ),
            (
                "p2_effect_interval_var",
                "flow_jepa_p2_structured_effect_interval_rms_variation",
                ".3f",
            ),
            (
                "p2_effect_slot_var",
                "flow_jepa_p2_structured_effect_slot_variation",
                ".3f",
            ),
            ("p2_effect_near", "flow_jepa_p2_effect_near_mass", ".3f"),
            ("p2_effect_mid", "flow_jepa_p2_effect_mid_mass", ".3f"),
            ("p2_effect_late", "flow_jepa_p2_effect_late_mass", ".3f"),
            (
                "consequence_effect",
                "flow_jepa_consequence_effect_base_rms",
                ".3f",
            ),
            (
                "consequence_organized",
                "flow_jepa_consequence_organized_delta_rms",
                ".3f",
            ),
            (
                "plan_protected_base",
                "flow_jepa_policy_plan_protected_base_rms",
                ".3f",
            ),
            (
                "plan_precision",
                "flow_jepa_policy_plan_precision_rms",
                ".3f",
            ),
            (
                "plan_temporal",
                "flow_jepa_policy_plan_temporal_rms",
                ".3f",
            ),
            (
                "intent_progress",
                "flow_jepa_intent_progress_coordinate",
                ".3f",
            ),
            ("frame_progress", "flow_jepa_frame_progress", ".3f"),
            (
                "progress_gap",
                "flow_jepa_intent_frame_progress_gap",
                "+.3f",
            ),
            (
                "progress_mae",
                "flow_jepa_intent_frame_progress_mae",
                ".3f",
            ),
            (
                "intent_selector_max",
                "flow_jepa_intent_window_selector_max",
                ".3f",
            ),
            (
                "intent_selector_entropy",
                "flow_jepa_intent_window_selector_entropy",
                ".3f",
            ),
            (
                "intent_window_cos",
                "flow_jepa_intent_window_adjacent_cosine",
                ".3f",
            ),
            (
                "intent_program_cos",
                "flow_jepa_intent_program_adjacent_cosine",
                ".3f",
            ),
            (
                "intent_attention_entropy",
                "flow_jepa_intent_program_attention_entropy",
                ".3f",
            ),
            (
                "intent_predictive_effect",
                "flow_jepa_intent_predictive_effect_rms",
                ".3f",
            ),
            (
                "intent_language_innovation",
                "flow_jepa_intent_language_innovation_rms",
                ".3f",
            ),
            (
                "intent_history_innovation",
                "flow_jepa_intent_history_innovation_rms",
                ".3f",
            ),
            (
                "intent_grounding_innovation",
                "flow_jepa_intent_grounding_innovation_rms",
                ".3f",
            ),
            (
                "intent_ordered_innovation",
                "flow_jepa_intent_ordered_innovation_rms",
                ".3f",
            ),
            (
                "intent_near_program",
                "flow_jepa_intent_near_program_argmax",
                ".2f",
            ),
            (
                "intent_mid_program",
                "flow_jepa_intent_mid_program_argmax",
                ".2f",
            ),
            (
                "intent_late_program",
                "flow_jepa_intent_late_program_argmax",
                ".2f",
            ),
            (
                "intent_observation_steps",
                "flow_jepa_intent_observation_steps",
                ".0f",
            ),
            (
                "w0_proposal_mass",
                "flow_jepa_w0_typed_condition_proposal_mass",
                ".3f",
            ),
            (
                "w1_proposal_mass",
                "flow_jepa_w1_typed_condition_proposal_mass",
                ".3f",
            ),
            (
                "w2_proposal_mass",
                "flow_jepa_w2_typed_condition_proposal_mass",
                ".3f",
            ),
            (
                "w0_clean_proposal",
                "flow_jepa_w0_clean_proposal_context_rms",
                ".3f",
            ),
            (
                "w1_clean_proposal",
                "flow_jepa_w1_clean_proposal_context_rms",
                ".3f",
            ),
            (
                "w2_clean_proposal",
                "flow_jepa_w2_clean_proposal_context_rms",
                ".3f",
            ),
            (
                "w0_direct_intent_bypass",
                "flow_jepa_w0_direct_intent_bypass",
                ".1e",
            ),
            (
                "w1_direct_intent_bypass",
                "flow_jepa_w1_direct_intent_bypass",
                ".1e",
            ),
            (
                "w2_direct_intent_bypass",
                "flow_jepa_w2_direct_intent_bypass",
                ".1e",
            ),
            (
                "p1_intent_query",
                "flow_jepa_phase_detail_query_norm",
                ".3f",
            ),
            (
                "p1_direct_condition_bypass",
                "flow_jepa_differential_p1_direct_condition_bypass",
                ".1e",
            ),
            (
                "g_to_p_intent_query",
                "attnres_world_to_policy_phase_query_norm",
                ".3f",
            ),
            (
                "g_to_p_goal_bypass",
                "attnres_world_to_policy_condition_query_norm",
                ".1e",
            ),
            (
                "g_to_p_history_bypass",
                "attnres_world_to_policy_history_query_norm",
                ".1e",
            ),
            ("phase_entropy", "flow_jepa_phase_entropy", ".3f"),
            ("phase_max", "flow_jepa_phase_max", ".3f"),
            (
                "phase_terminal",
                "flow_jepa_phase_terminal_mass",
                ".3f",
            ),
            (
                "execution_terminal",
                "flow_jepa_execution_terminal_probability",
                ".3f",
            ),
            (
                "execution_terminal_uncert",
                "flow_jepa_execution_terminal_uncertainty",
                ".3f",
            ),
            (
                "execution_terminal_bias",
                "evidence_execution_terminal_external_bias",
                "+.3f",
            ),
            (
                "phase_index",
                "flow_jepa_phase_expected_index",
                ".3f",
            ),
            (
                "phase_index_std",
                "flow_jepa_phase_expected_index_std",
                ".3f",
            ),
            (
                "phase_replay_steps",
                "flow_jepa_phase_replay_steps",
                ".0f",
            ),
            (
                "phase_interval_cos",
                "flow_jepa_phase_horizon_adjacent_cosine",
                ".3f",
            ),
            (
                "phase_interval_var",
                "flow_jepa_phase_horizon_variation",
                ".3f",
            ),
            (
                "phase_grounding_program_var",
                "flow_jepa_phase_program_grounding_variation",
                ".3f",
            ),
            (
                "p1_g3_only_address",
                "flow_jepa_p1_g3_only_factual_address",
                ".0f",
            ),
            (
                "legacy_w_posterior_skipped",
                "flow_jepa_v115_legacy_w_posterior_skipped",
                ".0f",
            ),
            (
                "future_transport_spatial_logit",
                "flow_jepa_progressive_future_transport_spatial_logit_rms",
                ".3f",
            ),
            (
                "p1_future_transport_logit",
                "flow_jepa_typed_p1_future_transport_logit_rms",
                ".3f",
            ),
            ("p1_micro_value", "flow_jepa_typed_p1_micro_value_rms", ".3f"),
            (
                "p1_spatial_variation",
                "flow_jepa_typed_p1_spatial_variation",
                ".3f",
            ),
            ("p2_detail_output", "flow_jepa_typed_p2_output_rms", ".3f"),
            (
                "p2_policy_carrier",
                "flow_jepa_typed_p2_policy_carrier_rms",
                ".3f",
            ),
            (
                "p2_owner_delta",
                "flow_jepa_typed_p2_routed_delta_rms",
                ".3f",
            ),
            (
                "p2_route_null",
                "flow_jepa_typed_p2_route_null_mass",
                ".3f",
            ),
            (
                "phase_horizon_var",
                "flow_jepa_phase_horizon_variation",
                ".3f",
            ),
            ("phase_entropy", "flow_jepa_phase_entropy", ".3f"),
            ("phase_max", "flow_jepa_phase_max", ".3f"),
            (
                "phase_terminal",
                "flow_jepa_phase_terminal_mass",
                ".3f",
            ),
            (
                "goal_horizon_var",
                "flow_jepa_goal_horizon_variation",
                ".3f",
            ),
            (
                "history_horizon_var",
                "flow_jepa_history_horizon_variation",
                ".3f",
            ),
            (
                "phase_horizon_cos",
                "flow_jepa_phase_horizon_adjacent_cosine",
                ".3f",
            ),
            (
                "goal_horizon_cos",
                "flow_jepa_goal_horizon_adjacent_cosine",
                ".3f",
            ),
            (
                "history_horizon_cos",
                "flow_jepa_history_horizon_adjacent_cosine",
                ".3f",
            ),
            (
                "p3_protected_base",
                "flow_jepa_policy_plan_protected_base_rms",
                ".3f",
            ),
            (
                "p3_precision",
                "flow_jepa_policy_plan_precision_rms",
                ".3f",
            ),
            (
                "p3_effect",
                "flow_jepa_policy_plan_effect_rms",
                ".3f",
            ),
            (
                "p3_temporal",
                "flow_jepa_policy_plan_temporal_rms",
                ".3f",
            ),
            (
                "p3_terminal",
                "flow_jepa_policy_plan_terminal_rms",
                ".3f",
            ),
            (
                "p3_lane_cos",
                "flow_jepa_policy_plan_lane_cosine",
                ".3f",
            ),
            (
                "p3_lane_var",
                "flow_jepa_policy_plan_lane_variation",
                ".3f",
            ),
            (
                "horizon_cos_seed",
                "flow_jepa_online_address_boundary_seed_adjacent_cosine",
                ".3f",
            ),
            (
                "horizon_cos_g3",
                "flow_jepa_online_address_boundary_post_g3_adjacent_cosine",
                ".3f",
            ),
            (
                "horizon_cos_address",
                "flow_jepa_online_address_boundary_post_address_adjacent_cosine",
                ".3f",
            ),
            (
                "horizon_cos_w1",
                "flow_jepa_online_address_boundary_post_w1_adjacent_cosine",
                ".3f",
            ),
            (
                "horizon_cos_w2",
                "flow_jepa_online_address_boundary_post_w2_adjacent_cosine",
                ".3f",
            ),
            (
                "horizon_cos_w3",
                "flow_jepa_online_address_boundary_post_w3_adjacent_cosine",
                ".3f",
            ),
            (
                "horizon_cos_interval",
                "flow_jepa_online_address_boundary_post_interval_adjacent_cosine",
                ".3f",
            ),
            (
                "address_projection_w3",
                "flow_jepa_online_address_boundary_post_w3_cumulative_address_projection",
                ".3f",
            ),
            (
                "address_projection_interval",
                "flow_jepa_online_address_boundary_post_interval_cumulative_address_projection",
                ".3f",
            ),
            (
                "interval_stage_loss",
                "flow_jepa_interval_stage",
                ".5f",
            ),
            (
                "interval_stage_raw",
                "flow_jepa_interval_stage_raw",
                ".5f",
            ),
            (
                "interval_stage_reliable",
                "flow_jepa_interval_stage_normalized",
                ".5f",
            ),
            (
                "interval_stage_direction",
                "flow_jepa_interval_stage_direction",
                ".5f",
            ),
            (
                "interval_stage_direction_floor",
                "flow_jepa_interval_stage_direction_floor_min",
                ".3e",
            ),
            (
                "interval_stage_endpoint",
                "flow_jepa_interval_stage_endpoint",
                ".5f",
            ),
            (
                "interval_stage_target_scale",
                "flow_jepa_interval_stage_target_scale",
                ".3f",
            ),
            (
                "interval_stage_reliability",
                "flow_jepa_interval_stage_reliability",
                ".3f",
            ),
            (
                "interval_stage_write",
                "flow_jepa_interval_stage_written_delta_rms",
                ".3f",
            ),
            (
                "interval_stage_online_w",
                "flow_jepa_interval_stage_online_w_candidate",
                ".0f",
            ),
            (
                "interval_stage_carrier_ratio",
                "flow_jepa_interval_stage_carrier_ratio",
                ".3f",
            ),
            (
                "interval_stage_typed_mass",
                "attnres_world_to_policy_interval_stage_source_mass",
                ".3f",
            ),
            (
                "interval_stage_norm_floor_seen",
                "flow_jepa_interval_stage_norm_denominator_min",
                ".3f",
            ),
            (
                "horizon_address_value_rms",
                "flow_jepa_horizon_address_value_precontract_rms",
                ".3f",
            ),
            (
                "horizon_address_value_contract",
                "flow_jepa_horizon_address_value_contraction",
                ".3f",
            ),
            (
                "horizon_address_value_channel_std",
                "flow_jepa_horizon_address_value_channel_std",
                ".3f",
            ),
        ):
            append(representation_parts, label, key, spec)
        horizon_keys = sorted(
            (
                key
                for key in row
                if key.startswith("flow_jepa_future_horizon_")
                and key.removeprefix("flow_jepa_future_horizon_").isdigit()
            ),
            key=lambda key: int(key.rsplit("_", 1)[-1]),
        )
        for key in horizon_keys:
            append(
                representation_parts,
                f"future_h{key.rsplit('_', 1)[-1]}",
                key,
                ".5f",
            )
        reliable_offsets = sorted(
            {
                int(match.group(1))
                for key in row
                for match in (
                    re.fullmatch(
                        r"flow_jepa_future_horizon_(\d+)_target_scale",
                        key,
                    ),
                )
                if match is not None
            }
        )
        for label, suffix, spec in (
            ("future_scale", "target_scale", ".3f"),
            ("future_norm_scale", "normalization_scale", ".3f"),
            ("future_rel", "reliability", ".3f"),
            ("future_direction", "active_direction", ".4f"),
            ("future_active", "active_loss", ".4f"),
            ("address_kl", "address_kl", ".3f"),
            ("address_rel", "address_reliability", ".3f"),
        ):
            entries: list[str] = []
            for offset in reliable_offsets:
                if suffix == "address_kl":
                    key = f"flow_jepa_horizon_address_{offset}_kl"
                elif suffix == "address_reliability":
                    key = f"flow_jepa_horizon_address_{offset}_reliability"
                else:
                    key = f"flow_jepa_future_horizon_{offset}_{suffix}"
                if key in row:
                    entries.append(f"{offset}:{format(row[key], spec)}")
            if entries:
                representation_parts.append(f"{label}=" + "/".join(entries))
        interval_offsets = sorted(
            {
                int(match.group(1))
                for key in row
                for match in (
                    re.fullmatch(
                        r"flow_jepa_interval_stage_horizon_(\d+)_loss",
                        key,
                    ),
                )
                if match is not None
            }
        )
        for offset in interval_offsets:
            parts: list[str] = []
            for label, suffix, spec in (
                ("l", "loss", ".4f"),
                ("r", "reliability", ".3f"),
                ("w", "write_rms", ".3f"),
            ):
                key = f"flow_jepa_interval_stage_horizon_{offset}_{suffix}"
                if key in row:
                    parts.append(f"{label}:{format(row[key], spec)}")
            if parts:
                representation_parts.append(f"interval_h{offset}=" + "/".join(parts))
        if log_version == "v96":
            for label, key, spec in (
                ("horizon_count", "flow_jepa_horizon_count", ".0f"),
                ("horizon_max", "flow_jepa_horizon_max", ".0f"),
                ("native_grid", "flow_jepa_native_grid_size", ".0f"),
                ("coarse_grid", "flow_jepa_coarse_grid_size", ".0f"),
                ("native_flow", "flow_jepa_native_flow_magnitude", ".3f"),
                ("detail_gate_mean", "flow_jepa_detail_gate_mean", ".3f"),
                ("detail_weighted_cmp", "flow_jepa_detail_effective_comparisons", ".0f"),
                ("detail_candidate_cmp", "flow_jepa_detail_candidate_comparisons", ".0f"),
                ("address_flow_mass", "flow_jepa_address_flow_mass", ".3f"),
                ("address_fallback_mass", "flow_jepa_address_fallback_mass", ".3f"),
                ("address_entropy", "flow_jepa_address_entropy", ".3f"),
                ("horizon_adj_cos", "flow_jepa_horizon_adjacent_cosine", ".3f"),
                ("far_horizon_norm", "flow_jepa_far_horizon_norm", ".3f"),
            ):
                append(representation_parts, label, key, spec)
        if log_version in _RAW_FLOW_JEPA_LOG_VERSIONS:
            for label, key, spec in (
                ("horizon_count", "flow_jepa_horizon_count", ".0f"),
                ("horizon_max", "flow_jepa_horizon_max", ".0f"),
                ("dino_grid", "flow_jepa_native_grid_size", ".0f"),
                ("reader_grid", "flow_jepa_coarse_grid_size", ".0f"),
                ("raw_high_grid", "flow_jepa_raw_high_grid_size", ".0f"),
                ("raw_mid_grid", "flow_jepa_raw_mid_grid_size", ".0f"),
                ("raw_coarse_grid", "flow_jepa_raw_coarse_grid_size", ".0f"),
                ("raw_flow", "flow_jepa_raw_flow_magnitude", ".3f"),
                ("raw_flow_grid", "flow_jepa_raw_flow_grid_magnitude", ".3f"),
                ("seed_reliability", "flow_jepa_raw_seed_reliability", ".3f"),
                ("mid_residual", "flow_jepa_raw_mid_residual_magnitude", ".3f"),
                ("high_residual", "flow_jepa_raw_high_residual_magnitude", ".3f"),
                (
                    "mid_bound_compress",
                    "flow_jepa_raw_mid_boundary_compression",
                    ".3f",
                ),
                (
                    "high_bound_compress",
                    "flow_jepa_raw_high_boundary_compression",
                    ".3f",
                ),
                (
                    "motion_flow_norm",
                    "flow_jepa_motion_evidence_flow_magnitude",
                    ".3f",
                ),
                ("raw_cycle_core", "flow_jepa_raw_cycle_core", ".4f"),
                ("raw_boundary", "flow_jepa_raw_boundary_penalty", ".4f"),
                ("raw_valid", "flow_jepa_raw_valid_fraction", ".3f"),
                ("raw_conf", "flow_jepa_raw_confidence_mean", ".3f"),
                ("raw_occ", "flow_jepa_raw_occlusion_fraction", ".3f"),
                (
                    "hard_occ_audit",
                    "flow_jepa_raw_hard_occlusion_fraction",
                    ".3f",
                ),
                (
                    "visibility_width_min",
                    "flow_jepa_visibility_transition_width_min",
                    ".3f",
                ),
                (
                    "visibility_gain_bound",
                    "flow_jepa_visibility_gain_bound_max",
                    ".2f",
                ),
                (
                    "corr_feature_rms_min",
                    "flow_jepa_correlation_feature_rms_min",
                    ".3e",
                ),
                (
                    "corr_norm_denom_min",
                    "flow_jepa_correlation_norm_denominator_min",
                    ".3f",
                ),
                (
                    "corr_norm_gain_max",
                    "flow_jepa_correlation_norm_gain_max",
                    ".2f",
                ),
                ("zero_warp", "flow_jepa_raw_identity_warp_error", ".4f"),
                ("warp_gain", "flow_jepa_raw_warp_gain_over_zero", "+.4f"),
                ("moving_gain", "flow_jepa_raw_moving_warp_gain", "+.4f"),
                ("static_gain", "flow_jepa_raw_static_warp_gain", "+.4f"),
                (
                    "moving_corr_entropy",
                    "flow_jepa_raw_moving_correlation_entropy",
                    ".3f",
                ),
                (
                    "moving_corr_margin",
                    "flow_jepa_raw_moving_correlation_margin",
                    ".3f",
                ),
                ("motion_visible", "flow_jepa_raw_observable_motion_fraction", ".3f"),
                ("raw_precision", "flow_jepa_raw_detail_precision_mean", ".3f"),
                (
                    "raw_detail_share"
                    if log_version in _COMPLEMENTARY_FLOW_JEPA_LOG_VERSIONS
                    else "raw_address_flow",
                    "flow_jepa_raw_address_flow_mass",
                    ".3f",
                ),
                (
                    "raw_base_share"
                    if log_version in _COMPLEMENTARY_FLOW_JEPA_LOG_VERSIONS
                    else "raw_address_fallback",
                    "flow_jepa_raw_address_fallback_mass",
                    ".3f",
                ),
                (
                    "detail_address_entropy"
                    if log_version in _COMPLEMENTARY_FLOW_JEPA_LOG_VERSIONS
                    else "raw_address_entropy",
                    "flow_jepa_raw_address_entropy",
                    ".3f",
                ),
                (
                    "address_separation",
                    "flow_jepa_raw_address_center_separation",
                    ".3f",
                ),
                (
                    "address_value_delta",
                    "flow_jepa_raw_address_lane_value_difference",
                    ".3f",
                ),
                (
                    "detail_address_concentration"
                    if log_version in _COMPLEMENTARY_FLOW_JEPA_LOG_VERSIONS
                    else "address_logit_gain",
                    "flow_jepa_raw_address_logit_advantage",
                    "+.3f",
                ),
                (
                    "address_zero_delta",
                    "flow_jepa_raw_address_zero_flow_value_delta",
                    ".3f",
                ),
                (
                    "address_shuffle_delta",
                    "flow_jepa_raw_address_shuffled_flow_value_delta",
                    ".3f",
                ),
                ("raw_candidates", "flow_jepa_raw_candidates_per_cell", ".0f"),
                ("raw_detail_tokens", "flow_jepa_raw_detail_token_count", ".0f"),
                (
                    "raw_dino_fused",
                    "flow_jepa_raw_detail_fused_with_latest_dino",
                    ".0f",
                ),
                (
                    "raw_source_dino_fused",
                    "flow_jepa_raw_detail_fused_with_source_dino",
                    ".0f",
                ),
                (
                    "refined_visual_tokens",
                    "flow_jepa_refined_evidence_token_count",
                    ".0f",
                ),
                ("grounding_blocks", "flow_jepa_grounding_block_count", ".0f"),
                ("world_blocks", "flow_jepa_world_block_count", ".0f"),
                ("policy_blocks", "flow_jepa_policy_block_count", ".0f"),
                ("horizon_adj_cos", "flow_jepa_horizon_adjacent_cosine", ".3f"),
                (
                    "query_adj_cos",
                    "flow_jepa_future_query_adjacent_cosine",
                    ".3f",
                ),
                (
                    "history_entropy",
                    "flow_jepa_perceptual_history_entropy",
                    ".3f",
                ),
                (
                    "history_latest",
                    "flow_jepa_perceptual_history_latest_mass",
                    ".3f",
                ),
                (
                    "horizon_step_update",
                    "flow_jepa_horizon_transition_update_rms",
                    ".3f",
                ),
                (
                    "horizon_state_delta",
                    "flow_jepa_horizon_transition_state_delta",
                    ".3f",
                ),
                ("far_horizon_norm", "flow_jepa_far_horizon_norm", ".3f"),
                ("world_xy_residual", "flow_jepa_world_spatial_residual_norm", ".3e"),
                (
                    "world_anchor_residual",
                    "flow_jepa_world_anchor_camera_residual_norm",
                    ".3f",
                ),
                (
                    "late_detail_entropy",
                    "flow_jepa_late_detail_attention_entropy",
                    ".3f",
                ),
                (
                    "late_detail_max",
                    "flow_jepa_late_detail_attention_max",
                    ".3f",
                ),
                (
                    "late_detail_update",
                    "flow_jepa_late_detail_update_norm",
                    ".3f",
                ),
                (
                    "late_detail_ratio",
                    "flow_jepa_late_detail_trajectory_ratio",
                    ".3f",
                ),
                (
                    "late_detail_scale",
                    "flow_jepa_late_detail_fixed_scale",
                    ".3f",
                ),
                (
                    "late_detail_tokens",
                    "flow_jepa_late_detail_token_count",
                    ".0f",
                ),
                (
                    "late_detail_glimpses",
                    "flow_jepa_address_policy_glimpse_count",
                    ".0f",
                ),
                (
                    "late_detail_glimpse_var",
                    "flow_jepa_address_policy_glimpse_route_variation",
                    ".3f",
                ),
            ):
                append(representation_parts, label, key, spec)

    execution_parts = [f"[{log_version}-exec]"]
    for label, key, spec in (
        ("exec_progress", "evidence_mmd_it_execution_progress", ".2f"),
        ("capacity_gate_mass", "evidence_mmd_it_capacity_gate_mass", ".5f"),
        ("effective_basis_mass", "evidence_mmd_it_effective_basis_mass", ".3f"),
        ("operation_probability", "evidence_mmd_it_operation_probability", ".3f"),
        ("workload_audit", "evidence_mmd_it_execution_cost", ".3f"),
        ("nonexp_violation", "evidence_mmd_it_nonexpansive_violation", ".1e"),
        ("selection_entropy", "evidence_mmd_it_execution_selection_entropy", ".3f"),
        ("selection_max", "evidence_mmd_it_execution_selection_max_probability", ".3f"),
        ("terminal_prior", "evidence_mmd_it_terminal_prior_weight", ".3f"),
        ("terminal_probability", "evidence_mmd_it_terminal_probability", ".3f"),
        ("hard_terminal_fraction", "evidence_mmd_it_hard_terminal_fraction", ".3f"),
        ("top_policy_scale", "evidence_top_policy_workspace_scale", ".3f"),
        ("top_policy_update", "evidence_top_policy_workspace_update_norm", ".3f"),
        ("top_policy_fixed_fusion", "evidence_top_policy_workspace_fixed_fusion", ".0f"),
        ("role_raw_rms", "role_residual_raw_rms", ".3f"),
        ("role_proposal_rms", "role_residual_proposed_rms", ".3f"),
        ("role_write_rms", "role_residual_written_rms", ".3f"),
        ("role_compress", "role_residual_compression", ".3f"),
        (
            "role_write_max_g",
            "role_residual_grounding_written_rms_max",
            ".3f",
        ),
        (
            "role_write_max_w",
            "role_residual_world_written_rms_max",
            ".3f",
        ),
        (
            "role_write_max_p",
            "role_residual_policy_written_rms_max",
            ".3f",
        ),
        (
            "role_norm_denom_min",
            "role_normalization_denominator_min",
            ".3f",
        ),
        (
            "role_norm_gain_max",
            "role_normalization_gain_max",
            ".2f",
        ),
        (
            "w2p_raw_value_rms",
            "attnres_world_to_policy_raw_value_rms",
            ".3f",
        ),
        (
            "w2p_value_rms",
            "attnres_world_to_policy_value_rms",
            ".3f",
        ),
        (
            "w2p_value_compress",
            "attnres_world_to_policy_value_compression",
            ".3f",
        ),
        (
            "g2w_query_norm_denom",
            "attnres_ground_to_world_query_norm_denominator_min",
            ".3f",
        ),
        (
            "w2p_query_norm_denom",
            "attnres_world_to_policy_query_norm_denominator_min",
            ".3f",
        ),
        (
            "bottom_raw_value_rms",
            "evidence_policy_delta_attnres_raw_value_rms",
            ".3f",
        ),
        (
            "bottom_value_rms",
            "evidence_policy_delta_attnres_value_rms",
            ".3f",
        ),
        (
            "bottom_value_compress",
            "evidence_policy_delta_attnres_value_compression",
            ".3f",
        ),
        (
            "bottom_query_norm_denom",
            "evidence_policy_delta_attnres_query_norm_denominator_min",
            ".3f",
        ),
        (
            "detail_query_norm_denom",
            "evidence_protected_detail_basis_query_norm_denominator_min",
            ".3f",
        ),
    ):
        append(execution_parts, label, key, spec)
    for label, soft_key, hard_key in (
        (
            "route",
            "evidence_mmd_it_dynamic_route_next_fraction",
            "evidence_mmd_it_hard_route_next_fraction",
        ),
        (
            "dwell",
            "evidence_mmd_it_dwell_expected",
            "evidence_mmd_it_hard_dwell_expected",
        ),
    ):
        if soft_key in row and hard_key in row:
            soft, hard = row[soft_key], row[hard_key]
            if abs(float(soft)) > 1e-12 or abs(float(hard)) > 1e-12:
                execution_parts.append(
                    f"{label}=soft:{soft:.3f}/hard:{hard:.3f}/gap:{soft - hard:+.3f}"
                )
    for label, key, spec in (
        ("value_loss", "evidence_mmd_it_execution_value_loss", ".4f"),
        ("value_target_spread", "evidence_mmd_it_execution_value_target_spread", ".4f"),
        ("value_pred_spread", "evidence_mmd_it_execution_value_predicted_spread", ".4f"),
        ("value_corr", "evidence_mmd_it_execution_value_correlation", "+.2f"),
        ("value_pair_acc", "evidence_mmd_it_execution_value_pairwise_accuracy", ".2f"),
        ("value_top1_acc", "evidence_mmd_it_execution_value_decision_accuracy", ".2f"),
        ("candidate_coverage", "evidence_mmd_it_execution_candidate_coverage", ".2f"),
        ("value_common_ratio", "evidence_mmd_it_execution_value_common_mode_ratio", ".2f"),
        ("terminal_target_margin", "evidence_mmd_it_terminal_target_cost_margin", "+.4f"),
        ("terminal_pred_margin", "evidence_mmd_it_terminal_predicted_cost_margin", "+.4f"),
        (
            "terminal_target_preferred",
            "evidence_mmd_it_terminal_target_preferred_fraction",
            ".2f",
        ),
        (
            "terminal_identity_error",
            "evidence_mmd_it_terminal_identity_velocity_error",
            ".2e",
        ),
        ("layer_loss_raw", "layer_contract", ".4f"),
        ("layer_scale", "layer_contract_aux_scale", ".4f"),
        ("layer_contrib", "loss_contrib_layer_contract", ".5f"),
    ):
        append(execution_parts, label, key, spec)
    block_update_keys = sorted(
        key
        for key in row
        if key.startswith("evidence_mmd_it_block_") and key.endswith("_update_norm")
    )
    if block_update_keys:
        block_update_keys = [key for key in block_update_keys if abs(float(row[key])) > 1e-12]
    if block_update_keys:
        execution_parts.append(
            "block_updates=" + "/".join(f"{row[key]:.3f}" for key in block_update_keys)
        )
    layer_contract_keys = sorted(
        key
        for key in row
        if key.startswith("layer") and key.endswith("_contract") and key[5:-9].isdigit()
    )
    if layer_contract_keys:
        layer_contract_keys = [key for key in layer_contract_keys if abs(float(row[key])) > 1e-12]
    if layer_contract_keys:
        execution_parts.append(
            "layer_losses=" + "/".join(f"{row[key]:.4f}" for key in layer_contract_keys)
        )

    grad_parts = [f"[{log_version}-grad]"]
    for label, key in (
        ("view_adapter", "grad_evidence_view_adapter"),
        ("organizer", "grad_evidence_condition_organizer"),
        ("evidence_reader", "grad_evidence_mmdit_evidence_reader"),
        ("action_state", "grad_evidence_mmdit_action_state"),
        ("top_policy_lift", "grad_evidence_top_policy_workspace_lift"),
        ("mmdit_blocks", "grad_evidence_mmdit_blocks"),
        ("exec_controller", "grad_evidence_mmdit_execution_controller"),
        ("capacity_control", "grad_evidence_mmdit_capacity_control"),
        ("operator_capacity", "grad_evidence_mmdit_operator_capacity"),
        ("operator_basis", "grad_evidence_mmdit_operator_basis"),
        ("value_reader", "grad_evidence_mmdit_execution_value_reader"),
        ("layer_adapter", "grad_layer_contract_adapters"),
        ("consequence", "grad_layer_consequence_cell"),
        ("dynamics", "grad_controlled_dynamics"),
        ("flow_dino", "grad_flow_dino_evidence"),
        ("coarse_flow", "grad_flow_dino_coarse_flow"),
        ("fine_flow", "grad_flow_dino_sparse_fine"),
        ("detail_router", "grad_flow_dino_detail_router"),
        ("address_reader", "grad_flow_dino_address_reader"),
        ("future_predictor", "grad_flow_dino_future_predictor"),
        ("raw_pyramid", "grad_flow_dino_raw_pyramid"),
        ("early_raw_context", "grad_flow_dino_early_masked_raw_context"),
        ("soft_address", "grad_flow_dino_soft_address_compiler"),
        ("semantic_coarse_flow", "grad_flow_dino_semantic_coarse_flow"),
        ("raw_mid_flow", "grad_flow_dino_raw_mid_flow"),
        ("raw_high_flow", "grad_flow_dino_raw_high_flow"),
        ("raw_detail_router", "grad_flow_dino_raw_detail_router"),
        ("raw_address_reader", "grad_flow_dino_raw_address_reader"),
        ("horizon_address", "grad_flow_dino_horizon_address"),
        ("address_g1", "grad_flow_dino_progressive_g1"),
        ("address_g2", "grad_flow_dino_progressive_g2"),
        ("address_g3", "grad_flow_dino_progressive_g3"),
        ("address_world_query", "grad_flow_dino_progressive_world_query"),
        ("functional_world_route", "grad_flow_dino_functional_world_router"),
        ("future_effect_sem", "grad_flow_dino_future_effect_semantic"),
        ("future_effect_geo", "grad_flow_dino_future_effect_geometry"),
        ("future_transport", "grad_flow_dino_progressive_future_transport"),
        ("interval_stage", "grad_flow_dino_interval_stage"),
        ("attnres_g2w", "grad_attnres_ground_to_world"),
        ("attnres_w2p", "grad_attnres_world_to_policy"),
        ("attnres_p2bottom", "grad_attnres_policy_to_mmdit"),
        ("protected_detail_route", "grad_protected_detail_basis_reader"),
        ("late_detail_reader", "grad_late_raw_detail_reader"),
        ("typed_p1_selector", "grad_late_raw_detail_typed_p1_selector"),
        ("p1_app_gateway", "grad_late_raw_detail_p1_appearance_gateway"),
        ("literal_rgb_value", "grad_late_raw_detail_literal_rgb_value"),
        ("learned_detail_value", "grad_late_raw_detail_learned_detail_value"),
        ("typed_p2_condition", "grad_late_raw_detail_typed_p2_condition"),
        ("p2_owner_router", "grad_late_raw_detail_typed_p2_router"),
        ("typed_p2_refiner", "grad_late_raw_detail_typed_p2"),
        ("goal_tokens", "grad_goal_resampler"),
        ("horizon_condition", "grad_stateless_horizon_adapter"),
        ("horizon_phase", "grad_stateless_horizon_phase_path"),
        ("horizon_goal", "grad_stateless_horizon_goal_path"),
        ("horizon_history", "grad_stateless_horizon_history_path"),
        ("horizon_world_queries", "grad_stateless_horizon_world_queries"),
        ("goal_phase", "grad_stateless_goal_phase_machine"),
        ("grounded_s_goal", "grad_grounded_intent_goal"),
        ("grounded_s_observable", "grad_grounded_intent_observable"),
        ("grounded_s_intervals", "grad_grounded_intent_intervals"),
        ("grounded_s_temporal", "grad_grounded_intent_temporal"),
        ("grounded_s_completion", "grad_grounded_intent_completion"),
        ("grounded_w_proposal", "grad_grounded_clean_proposal"),
        ("grounded_w_inputs", "grad_grounded_world_shared_inputs"),
        ("grounded_w1_blocks", "grad_grounded_world_w1_blocks"),
        ("grounded_w2_blocks", "grad_grounded_world_w2_blocks"),
        ("grounded_w_shared_heads", "grad_grounded_world_shared_heads"),
        ("object_grounder", "grad_object_grounder"),
        ("object_s_goal", "grad_object_s_goal"),
        ("object_s_history", "grad_object_s_history"),
        ("object_s_typed", "grad_object_s_typed_intervals"),
        ("object_s_temporal", "grad_object_s_temporal"),
        ("object_s_state_change", "grad_object_s_state_change"),
        ("object_recognizer", "grad_object_plan_recognizer"),
        ("object_coarse_action", "grad_object_coarse_action"),
        ("object_w_inputs", "grad_object_w_inputs"),
        ("object_w1", "grad_object_w1"),
        ("object_w2", "grad_object_w2"),
        ("object_w_heads", "grad_object_w_heads"),
        ("object_p2", "grad_object_p2_effect_reader"),
        ("object_consequence", "grad_object_consequence"),
        ("object_p3_precision", "grad_object_p3_precision"),
        ("object_p3_temporal", "grad_object_p3_temporal"),
        ("object_p3_state_change", "grad_object_p3_state_change"),
        ("intent_goal", "grad_intent_goal_program"),
        ("intent_history", "grad_intent_history_encoder"),
        ("intent_history_write", "grad_intent_history_write"),
        ("intent_grounding", "grad_intent_grounding_write"),
        ("intent_ordered", "grad_intent_ordered_refinement"),
        ("intent_window", "grad_intent_window_read"),
        ("intent_predictive", "grad_intent_predictive_effect"),
        ("intent_terminal", "grad_intent_terminal"),
        (
            "w_clean_proposal",
            "grad_differential_clean_proposal_world_condition",
        ),
        ("intent_g_to_p_query", "grad_intent_canonical_g_to_p_query"),
        ("intent_p1_query", "grad_intent_canonical_p1_query"),
        ("intent_s1", "grad_stateless_intent_s1"),
        ("intent_s2", "grad_stateless_intent_s2"),
        ("intent_s3", "grad_stateless_intent_s3"),
        ("intent_mlp", "grad_stateless_intent_mlp"),
        ("goal_program", "grad_goal_phase_program"),
        ("phase_transition", "grad_goal_phase_transition"),
        ("phase_observation", "grad_goal_phase_observation"),
        ("goal_world_typed", "grad_goal_phase_typed_world_context"),
        ("p3_compiler", "grad_policy_plan_compiler"),
        ("p2_effect_reader", "grad_p2_structured_effect_reader"),
        ("consequence_organizer", "grad_consequence_plan_organizer"),
        ("window_effect_near_mid", "grad_flow_dino_window_effect_near_mid"),
        ("window_effect_late", "grad_flow_dino_window_effect_late"),
        (
            "differential_w1",
            "grad_differential_w1_near_mid_transition",
        ),
        (
            "differential_w2",
            "grad_differential_w2_late_transition",
        ),
        ("effect_decoder", "grad_differential_effect_decoder"),
        (
            "current_reference",
            "grad_differential_current_reference_bridge",
        ),
        ("p3_precision", "grad_policy_plan_precision"),
        ("p3_effect", "grad_policy_plan_effect"),
        ("p3_temporal", "grad_policy_plan_temporal"),
        ("p3_terminal", "grad_policy_plan_terminal"),
        ("action_history", "grad_action_history_encoder"),
        ("dit_blocks", "grad_dit_blocks"),
        ("grounding_blocks", "grad_dit_grounding_blocks"),
        ("world_blocks", "grad_dit_world_blocks"),
        ("policy_blocks", "grad_dit_policy_blocks"),
        ("policy_heads", "grad_final_policy_heads"),
        ("global_preclip", "grad"),
    ):
        append(
            grad_parts,
            label,
            key,
            ".2e",
            keep_zero=(log_version not in {"v119", "v120"}),
        )
    grad_parts.extend((f"lr={learning_rate:.3e}", f"sec_per_batch={seconds_per_batch:.3f}"))
    if log_version in {"v119", "v120"}:
        # V119 is a capability-selected sibling, not another layer of V115-
        # V118 semantics.  Rebuild its representation line from the live
        # current-observation path so retired phase/world/carrier aliases
        # cannot reappear merely because an ancestry tensor is still exposed
        # for checkpoint compatibility or audit.
        representation_parts = [f"[{log_version}-repr]"]
        for label, key, spec in (
            ("warp", "flow_jepa_warp_loss", ".5f"),
            ("identity_adv", "flow_jepa_identity_advantage_loss", ".5f"),
            ("static_identity", "flow_jepa_static_identity_loss", ".5f"),
            ("cycle", "flow_jepa_cycle_loss", ".5f"),
            ("smooth", "flow_jepa_smoothness_loss", ".5f"),
            ("refine_seq", "flow_jepa_refinement_sequence_loss", ".5f"),
            ("flow_mag", "flow_jepa_patch_flow_magnitude", ".3f"),
            ("confidence", "flow_jepa_confidence_mean", ".3f"),
            ("occlusion", "flow_jepa_occlusion_fraction", ".3f"),
            ("corr_entropy", "flow_jepa_correlation_entropy", ".3f"),
            ("corr_margin", "flow_jepa_correlation_margin", ".3f"),
            ("context_drop", "flow_jepa_context_dropout_fraction", ".3f"),
            ("target_mask", "flow_jepa_future_target_fraction", ".3f"),
            (
                "p1_spatial_var",
                "flow_jepa_typed_p1_spatial_variation",
                ".3f",
            ),
            ("literal_rgb", "flow_jepa_literal_rgb_chart_rms", ".3f"),
            (
                "p1_query_chunk",
                "flow_jepa_address_query_chunk_actual",
                ".0f",
            ),
            (
                "p1_checkpoint_active",
                "flow_jepa_typed_p1_activation_checkpoint_active",
                ".0f",
            ),
        ):
            append(representation_parts, label, key, spec)
    lines = [" ".join(loss_parts)]
    if representation_parts:
        lines.append(" ".join(representation_parts))
    if log_version == "v119":
        grounded_parts = ["[v119-ground]"]
        for label, key, spec in (
            ("active", "grounded_intent_effect_active", ".0f"),
            ("g2_fine_H", "flow_jepa_progressive_g2_fine_entropy", ".3f"),
            (
                "g2_sem_app_l1",
                "flow_jepa_progressive_g2_semantic_appearance_posterior_l1",
                ".3f",
            ),
            (
                "g2_app_geo_l1",
                "flow_jepa_progressive_g2_appearance_geometry_posterior_l1",
                ".3f",
            ),
            (
                "g3_sem_H",
                "flow_jepa_progressive_g3_semantic_slot_entropy",
                ".3f",
            ),
            (
                "g3_app_H",
                "flow_jepa_progressive_g3_appearance_slot_entropy",
                ".3f",
            ),
            (
                "g3_geo_H",
                "flow_jepa_progressive_g3_geometry_slot_entropy",
                ".3f",
            ),
            ("g2g3_sem", "grounded_g2_g3_semantic_owner_l1", ".3f"),
            ("g2g3_app", "grounded_g2_g3_appearance_owner_l1", ".3f"),
            ("g2g3_geo", "grounded_g2_g3_geometry_owner_l1", ".3f"),
            (
                "p1_app_geo_l1",
                "flow_jepa_typed_p1_appearance_geometry_fine_l1",
                ".3f",
            ),
            (
                "p1_sem_app_l1",
                "flow_jepa_typed_p1_semantic_appearance_route_l1",
                ".3f",
            ),
            (
                "current_ref_align",
                "grounded_future_effect_current_reference_alignment_rms",
                ".3e",
            ),
        ):
            append(grounded_parts, label, key, spec)
        intent_parts = ["[v119-intent]"]
        for label, key, spec in (
            ("goal_attention_H", "grounded_s_goal_attention_entropy", ".3f"),
            (
                "interval_goal_H",
                "grounded_s_interval_goal_attention_entropy",
                ".3f",
            ),
            ("source_attention_H", "grounded_s_interval_source_entropy", ".3f"),
            ("interval_cos", "grounded_s_interval_adjacent_cosine", ".3f"),
            ("interval_var", "grounded_s_interval_variation", ".3f"),
            ("achieved", "grounded_s_achieved_rms", ".3f"),
            ("remaining", "grounded_s_remaining_rms", ".3f"),
            ("completion", "grounded_s_completion_probability", ".3f"),
            ("source_null", "grounded_s_interval_null_mass", ".3f"),
            ("source_observable", "grounded_s_interval_observable_mass", ".3f"),
            ("source_history", "grounded_s_interval_history_mass", ".3f"),
            ("source_semantic", "grounded_s_interval_semantic_mass", ".3f"),
            ("source_appearance", "grounded_s_interval_appearance_mass", ".3f"),
            ("source_geometry", "grounded_s_interval_geometry_mass", ".3f"),
        ):
            append(intent_parts, label, key, spec)
        for interval in ("h4_8", "h8_16", "h16_32", "h32_48"):
            append(
                intent_parts,
                f"{interval}_goal_H",
                f"grounded_s_{interval}_goal_attention_entropy",
                ".3f",
            )
            append(
                intent_parts,
                f"{interval}_source_H",
                f"grounded_s_{interval}_source_attention_entropy",
                ".3f",
            )
        effect_parts = ["[v119-effect]"]
        for label, key, spec in (
            ("w1_sem", "grounded_w1_semantic_rms", ".3f"),
            ("w1_transport", "grounded_w1_transport_rms", ".3f"),
            ("w1_interval_var", "grounded_w1_interval_variation", ".3f"),
            ("w1_object_var", "grounded_w1_object_variation", ".3f"),
            ("w1_cos", "grounded_w1_adjacent_cosine", ".3f"),
            ("w2_sem", "grounded_w2_semantic_rms", ".3f"),
            ("w2_transport", "grounded_w2_transport_rms", ".3f"),
            ("w2_interval_var", "grounded_w2_interval_variation", ".3f"),
            ("w2_object_var", "grounded_w2_object_variation", ".3f"),
            ("w2_cos", "grounded_w2_adjacent_cosine", ".3f"),
            (
                "pred_cos",
                "grounded_future_effect_prediction_adjacent_cosine",
                ".3f",
            ),
            (
                "target_cos",
                "grounded_future_effect_target_adjacent_cosine",
                ".3f",
            ),
            (
                "pred_var",
                "grounded_future_effect_prediction_interval_variation",
                ".3f",
            ),
            (
                "target_var",
                "grounded_future_effect_target_interval_variation",
                ".3f",
            ),
            (
                "transport_pred_var",
                "grounded_future_effect_prediction_transport_variation",
                ".3f",
            ),
            (
                "transport_target_var",
                "grounded_future_effect_target_transport_variation",
                ".3f",
            ),
            ("loss_successor", "flow_jepa_future_effect_successor_loss", ".4f"),
            ("loss_semantic", "flow_jepa_future_effect_semantic_loss", ".4f"),
            ("loss_transport", "flow_jepa_future_effect_transport_loss", ".4f"),
            (
                "loss_covariance",
                "flow_jepa_future_effect_transport_covariance_loss",
                ".4f",
            ),
            (
                "loss_persistence_change",
                "flow_jepa_future_effect_persistence_change_loss",
                ".4f",
            ),
            (
                "loss_visibility_change",
                "flow_jepa_future_effect_visibility_change_loss",
                ".4f",
            ),
            (
                "loss_uncertainty_calibration",
                "flow_jepa_future_effect_uncertainty_calibration_loss",
                ".4f",
            ),
            (
                "loss_reliability_calibration",
                "flow_jepa_future_effect_reliability_calibration_loss",
                ".4f",
            ),
            (
                "loss_interval_transition",
                "flow_jepa_future_effect_relative_transition_loss",
                ".4f",
            ),
        ):
            append(effect_parts, label, key, spec)
        policy_parts = ["[v119-policy]"]
        for label, key, spec in (
            ("effect_read", "grounded_p2_effect_read_rms", ".3f"),
            ("content_score_max", "grounded_p2_content_score_abs_max", ".3f"),
            ("intent_score_max", "grounded_p2_intent_score_abs_max", ".3f"),
            ("coordinate_score_max", "grounded_p2_coordinate_score_abs_max", ".3f"),
            (
                "query_coordinate_std",
                "grounded_p2_query_coordinate_std",
                ".3f",
            ),
            ("posterior_max", "grounded_p2_posterior_max", ".3f"),
            ("posterior_H", "grounded_p2_posterior_entropy", ".3f"),
            ("tau_content", "grounded_p2_content_temperature", ".3f"),
            ("tau_intent", "grounded_p2_intent_temperature", ".3f"),
            ("tau_coordinate", "grounded_p2_coordinate_temperature", ".3f"),
            ("mass_h4_8", "grounded_p2_h4_8_mass", ".3f"),
            ("mass_h8_16", "grounded_p2_h8_16_mass", ".3f"),
            ("mass_h16_32", "grounded_p2_h16_32_mass", ".3f"),
            ("mass_h32_48", "grounded_p2_h32_48_mass", ".3f"),
            ("consequence_effect", "grounded_consequence_effect_rms", ".3f"),
            (
                "consequence_interaction",
                "grounded_consequence_interaction_rms",
                ".3f",
            ),
            ("p3_precision", "grounded_p3_precision_rms", ".3f"),
            ("p3_temporal", "grounded_p3_temporal_rms", ".3f"),
        ):
            append(policy_parts, label, key, spec)
        for parts in (grounded_parts, intent_parts, effect_parts, policy_parts):
            if len(parts) > 1:
                lines.append(" ".join(parts))
        normalized_fields = (
            ("successor", "successor"),
            ("semantic", "semantic"),
            ("transport", "transport"),
            ("covariance", "transport_covariance"),
            ("persistence", "persistence_change"),
            ("visibility", "visibility_change"),
            ("uncertainty", "uncertainty_calibration"),
            ("reliability", "reliability_calibration"),
        )
        for interval_name in ("h4_8", "h8_16", "h16_32", "h32_48"):
            interval_parts = [
                "[v119-effect-error]",
                f"interval={interval_name}",
            ]
            append(
                interval_parts,
                "teacher_reliability",
                "grounded_future_effect_teacher_reliability_"
                f"{interval_name}",
                ".3f",
            )
            for display_name, field_name in normalized_fields:
                append(
                    interval_parts,
                    display_name,
                    "grounded_future_effect_"
                    f"{field_name}_{interval_name}_target_normalized_error",
                    ".3f",
                )
            if len(interval_parts) > 2:
                lines.append(" ".join(interval_parts))
    if log_version in {"v120", "v121"}:
        ground_parts = [f"[{log_version}-ground]"]
        for label, key, spec in (
            ("reconstruction", "object_grounding_reconstruction_mse", ".5f"),
            ("prototype_mse", "object_grounding_prototype_mse", ".5f"),
            (
                "spatial_refine_mse",
                "object_grounding_spatial_refinement_mse",
                ".5f",
            ),
            ("existence", "object_grounding_existence_mean", ".3f"),
            ("validity", "object_grounding_validity_mean", ".3f"),
            ("allocation", "object_grounding_allocation_share_mean", ".3f"),
            ("null", "object_grounding_null_mass", ".3f"),
            ("mass_error", "object_grounding_mass_conservation_error", ".1e"),
            ("owner_H", "object_grounding_candidate_owner_entropy", ".3f"),
            ("local_prior_H", "object_grounding_local_prior_entropy", ".3f"),
            ("chart_H", "object_grounding_chart_entropy", ".3f"),
            ("g3_parent_l1", "object_grounding_g3_parent_l1", ".3e"),
            ("object_pair_cos", "object_grounding_object_content_pair_cosine", ".3f"),
            ("chart_pair_overlap", "object_grounding_object_chart_pair_overlap", ".3f"),
            (
                "sem_app_post_l1",
                "object_grounding_semantic_appearance_posterior_l1",
                ".3f",
            ),
            (
                "sem_geo_post_l1",
                "object_grounding_semantic_geometry_posterior_l1",
                ".3f",
            ),
            (
                "app_geo_post_l1",
                "object_grounding_appearance_geometry_posterior_l1",
                ".3f",
            ),
            ("flow_prior", "object_grounding_transport_prior_rms", ".3f"),
        ):
            append(ground_parts, label, key, spec, keep_zero=True)
        intent_parts = [f"[{log_version}-intent]"]
        for label, key, spec in (
            ("goal_H", "object_intent_goal_attention_entropy", ".3f"),
            ("interval_goal_H", "object_intent_interval_goal_entropy", ".3f"),
            ("history_H", "object_intent_interval_history_entropy", ".3f"),
            ("object_H", "object_intent_interval_object_entropy", ".3f"),
            ("semantic_H", "object_intent_interval_semantic_entropy", ".3f"),
            ("appearance_H", "object_intent_interval_appearance_entropy", ".3f"),
            ("geometry_H", "object_intent_interval_geometry_entropy", ".3f"),
            ("interval_var", "object_intent_interval_variation", ".3f"),
            ("state_interval_var", "object_intent_interval_state_variation", ".3f"),
            ("object_key_var", "object_intent_interval_object_key_variation", ".3f"),
            ("object_value_var", "object_intent_interval_object_value_variation", ".3f"),
            ("temporal_var", "object_intent_temporal_variation", ".3f"),
            ("goal_innov", "object_intent_goal_innovation_rms", ".3f"),
            ("history_innov", "object_intent_history_innovation_rms", ".3f"),
            ("object_innov", "object_intent_object_innovation_rms", ".3f"),
            ("typed_innov", "object_intent_typed_innovation_rms", ".3f"),
            ("state_delta", "object_intent_observed_state_delta_rms", ".3f"),
            ("transport", "object_intent_observed_transport_rms", ".3f"),
            ("change_history", "object_intent_state_change_history_rms", ".3f"),
            ("change_transport", "object_intent_state_change_transport_rms", ".3f"),
            ("state_change", "object_intent_state_change_evidence_rms", ".3f"),
            ("state_change_H", "object_intent_state_change_attention_entropy", ".3f"),
            ("online_match", "object_intent_online_match_loss", ".5f"),
            ("action_match", "object_intent_action_match_loss", ".5f"),
            ("state_match", "object_intent_state_match_loss", ".5f"),
            ("object_key_match", "object_intent_object_key_match_loss", ".5f"),
            ("object_value_match", "object_intent_object_value_match_loss", ".5f"),
            ("recognizer", "object_plan_recognition_loss", ".5f"),
            ("coarse_action", "object_coarse_action_loss", ".5f"),
        ):
            append(intent_parts, label, key, spec, keep_zero=True)
        dynamics_parts = [f"[{log_version}-dynamics]"]
        for label, key, spec in (
            ("intent_innov", "object_w_interval_innovation_rms", ".3f"),
            ("action_innov", "object_w_action_innovation_rms", ".3f"),
            ("state_innov", "object_w_state_innovation_rms", ".3f"),
            ("object_key_innov", "object_w_object_key_innovation_rms", ".3f"),
            ("object_value_innov", "object_w_object_value_innovation_rms", ".3f"),
            ("typed_innov", "object_w_typed_innovation_rms", ".3f"),
            ("w1_delta", "object_w1_semantic_delta_rms", ".3f"),
            ("w1_transport", "object_w1_transport_rms", ".3f"),
            ("w1_interval_cos", "object_w1_interval_adjacent_cosine", ".3f"),
            ("w1_object_cos", "object_w1_object_pair_cosine", ".3f"),
            ("w2_delta", "object_w2_semantic_delta_rms", ".3f"),
            ("w2_transport", "object_w2_transport_rms", ".3f"),
            ("w2_interval_cos", "object_w2_interval_adjacent_cosine", ".3f"),
            ("w2_object_cos", "object_w2_object_pair_cosine", ".3f"),
            ("teacher_visibility", "object_teacher_visibility", ".3f"),
            ("teacher_visibility_change", "object_teacher_visibility_change", ".3f"),
            ("teacher_persistence_change", "object_teacher_persistence_change", ".3f"),
            ("teacher_null", "object_teacher_null_probability", ".3f"),
            ("teacher_sem_max", "object_teacher_semantic_max", ".3f"),
            ("teacher_sem_margin", "object_teacher_semantic_margin", ".3f"),
            ("teacher_app_max", "object_teacher_appearance_max", ".3f"),
            ("teacher_app_margin", "object_teacher_appearance_margin", ".3f"),
            ("teacher_geom_margin", "object_teacher_geometry_margin", ".3f"),
            ("teacher_uncert", "object_teacher_uncertainty", ".3f"),
            ("teacher_delta", "object_teacher_semantic_delta_rms", ".3f"),
            ("teacher_transport", "object_teacher_transport_rms", ".3f"),
            ("teacher_supports", "object_teacher_supports_per_interval", ".2f"),
            ("content_loss", "object_future_content", ".5f"),
            ("transport_loss", "object_future_transport", ".5f"),
            ("covariance_loss", "object_future_covariance", ".5f"),
            ("visibility_loss", "object_future_visibility", ".5f"),
            ("persistence_loss", "object_future_persistence", ".5f"),
            ("uncertainty_loss", "object_future_uncertainty", ".5f"),
            ("transition_loss", "object_future_transition", ".5f"),
            ("pred_cos", "object_future_prediction_adjacent_cosine", ".3f"),
            ("target_cos", "object_future_target_adjacent_cosine", ".3f"),
            ("pred_var", "object_future_prediction_interval_variation", ".3f"),
            ("target_var", "object_future_target_interval_variation", ".3f"),
        ):
            append(dynamics_parts, label, key, spec, keep_zero=True)
        policy_parts = [f"[{log_version}-policy]"]
        for label, key, spec in (
            ("semantic_score", "object_p2_semantic_score_abs", ".3f"),
            ("semantic_score_max", "object_p2_semantic_score_max_abs", ".3f"),
            ("geometry_score", "object_p2_geometry_score_abs", ".3f"),
            ("geometry_score_max", "object_p2_geometry_score_max_abs", ".3f"),
            ("intent_score", "object_p2_intent_score_abs", ".3f"),
            ("intent_score_max", "object_p2_intent_score_max_abs", ".3f"),
            ("coordinate_score", "object_p2_coordinate_score_abs", ".3f"),
            ("coordinate_score_max", "object_p2_coordinate_score_max_abs", ".3f"),
            ("address_score", "object_p2_address_score_abs", ".3f"),
            ("transport_score", "object_p2_transport_score_abs", ".3f"),
            ("semantic_logit_max", "object_p2_semantic_logit_max_abs", ".3f"),
            ("geometry_logit_max", "object_p2_geometry_logit_max_abs", ".3f"),
            ("tau_content", "object_p2_temperature_content", ".3f"),
            ("tau_intent", "object_p2_temperature_intent", ".3f"),
            ("tau_coordinate", "object_p2_temperature_coordinate", ".3f"),
            ("semantic_H", "object_p2_semantic_posterior_entropy", ".3f"),
            ("geometry_H", "object_p2_geometry_posterior_entropy", ".3f"),
            ("semantic_max", "object_p2_semantic_posterior_max", ".3f"),
            ("geometry_max", "object_p2_geometry_posterior_max", ".3f"),
            ("semantic_null", "object_p2_semantic_null_mass", ".3f"),
            ("geometry_null", "object_p2_geometry_null_mass", ".3f"),
            ("calibration", "object_p2_selector_calibration", ".3f"),
            ("semantic_mass", "object_p2_semantic_value_mass", ".3f"),
            ("geometry_mass", "object_p2_geometry_value_mass", ".3f"),
            ("effect_precontract", "object_p2_effect_precontract_rms", ".3f"),
            ("effect", "object_p2_effect_rms", ".3f"),
            ("contract_min", "object_p2_contract_min", ".3f"),
            ("consequence_effect", "object_consequence_effect_rms", ".3f"),
            ("interaction", "object_consequence_interaction_rms", ".3f"),
            ("consequence_ratio", "object_consequence_ratio", ".3f"),
            ("p3_precision", "object_p3_precision_rms", ".3f"),
            ("p3_temporal", "object_p3_temporal_rms", ".3f"),
            ("p3_state_change", "object_p3_state_change_rms", ".3f"),
        ):
            append(policy_parts, label, key, spec, keep_zero=True)
        for interval_index, interval_name in enumerate(
            ("h4_8", "h8_16", "h16_32", "h32_48")
        ):
            for owner in ("semantic", "geometry"):
                append(
                    policy_parts,
                    f"{owner}_{interval_name}_mass",
                    f"object_p2_{owner}_interval_{interval_index}_mass",
                    ".3f",
                    keep_zero=True,
                )
        for parts in (ground_parts, intent_parts, dynamics_parts, policy_parts):
            if len(parts) > 1:
                lines.append(" ".join(parts))
        for interval_name in ("h4_8", "h8_16", "h16_32", "h32_48"):
            interval_parts = [
                f"[{log_version}-dynamics-error]",
                f"interval={interval_name}",
            ]
            for field in (
                "successor",
                "semantic",
                "transport",
                "covariance",
                "visibility",
                "persistence",
                "uncertainty",
            ):
                append(
                    interval_parts,
                    field,
                    f"object_future_{field}_{interval_name}_normalized_error",
                    ".3f",
                    keep_zero=True,
                )
            if len(interval_parts) > 2:
                lines.append(" ".join(interval_parts))
    if (
        log_version in _BALANCED_FLOW_JEPA_LOG_VERSIONS
        and log_version != "v119"
    ):
        balance_parts = [f"[{log_version}-balance]"]
        for label, key, spec in (
            ("flow_without_info_balance", "physical_flow_no_information_balance", ".6f"),
            ("trajectory_info", "trajectory_information_score", ".4f"),
            ("info_weight_min", "trajectory_information_weight_min", ".3f"),
            ("info_weight_max", "trajectory_information_weight_max", ".3f"),
            ("info_effective_fraction", "trajectory_information_effective_fraction", ".3f"),
            ("horizon_weight_first", "action_horizon_weight_first", ".3f"),
            ("horizon_weight_tail", "action_horizon_weight_tail", ".3f"),
            ("history_keep", "condition_action_history_keep", ".3f"),
            ("goal_keep", "condition_goal_keep", ".3f"),
            ("proposal_keep", "condition_proposal_keep", ".3f"),
            ("teacher_past_quota", "flow_jepa_teacher_mask_past_fraction", ".3f"),
            ("teacher_change_quota", "flow_jepa_teacher_mask_change_fraction", ".3f"),
            ("teacher_uniform_quota", "flow_jepa_teacher_mask_uniform_fraction", ".3f"),
            ("selected_change_ratio", "flow_jepa_teacher_mask_selected_change_ratio", ".3f"),
        ):
            append(balance_parts, label, key, spec, keep_zero=True)
        for key in sorted(
            key for key in row if key.startswith("action_band_") and key.endswith("_physical_flow")
        ):
            label = key.removeprefix("action_band_").removesuffix("_physical_flow")
            append(balance_parts, f"action_h{label}", key, ".6f", keep_zero=True)
        lines.append(" ".join(balance_parts))
    lines.extend((" ".join(execution_parts), " ".join(grad_parts)))
    if log_version in {"v111", "v112", "v113"}:
        owner_grad_parts = [f"[{log_version}-owner-grad]"]
        for label, key in (
            ("g2_sem", "grad_flow_dino_progressive_g2_semantic_owner"),
            ("g2_app", "grad_flow_dino_progressive_g2_appearance_owner"),
            ("g2_geo", "grad_flow_dino_progressive_g2_geometry_owner"),
            ("g3_public", "grad_flow_dino_progressive_g3_public"),
            ("g3_sem", "grad_flow_dino_progressive_g3_semantic_owner"),
            ("g3_app", "grad_flow_dino_progressive_g3_appearance_owner"),
            ("g3_geo", "grad_flow_dino_progressive_g3_geometry_owner"),
            ("w_sem", "grad_flow_dino_progressive_world_semantic_owner"),
            ("w_app", "grad_flow_dino_progressive_world_appearance_owner"),
            ("w_geo", "grad_flow_dino_progressive_world_geometry_owner"),
            ("w_interval", "grad_flow_dino_progressive_world_interval_owner"),
            ("p2_policy", "grad_late_raw_detail_typed_p2_policy_owner"),
            ("p2_sem", "grad_late_raw_detail_typed_p2_semantic_owner"),
            ("p2_app", "grad_late_raw_detail_typed_p2_appearance_owner"),
            ("p2_geo", "grad_late_raw_detail_typed_p2_geometry_owner"),
            ("p2_horizon", "grad_late_raw_detail_typed_p2_horizon_owner"),
        ):
            append(owner_grad_parts, label, key, ".2e", keep_zero=True)
        lines.append(" ".join(owner_grad_parts))
    return "\n".join(lines)


def _flow_jepa_stage1_serial_log_line(
    row: dict[str, float],
    *,
    epoch: int,
    batch_index: int,
    learning_rate: float,
    seconds_per_batch: float,
) -> str:
    """Compact log for the representation-only V95 Stage1 experiment."""

    def append(parts: list[str], label: str, key: str, spec: str = ".4f") -> None:
        if key in row:
            parts.append(f"{label}={format(row[key], spec)}")

    train = [
        f"[v95-stage1-train] epoch={epoch:03d}",
        f"batch={batch_index:04d}",
        f"loss_representation={row['loss']:.6f}",
    ]
    contributions = sorted(
        (key for key in row if key.startswith("loss_contrib_") and abs(float(row[key])) > 1e-12),
        key=lambda key: abs(float(row[key])),
        reverse=True,
    )
    if contributions:
        train.append(
            "contrib="
            + "/".join(
                f"{key.removeprefix('loss_contrib_')}:{row[key]:.5f}" for key in contributions
            )
        )
    append(train, "ledger_gap", "loss_ledger_residual", "+.2e")

    representation = ["[v95-stage1-repr]"]
    for label, key, spec in (
        ("window_pred", "flow_jepa_future_prediction", ".5f"),
        ("change_dir", "flow_jepa_future_change_direction", ".5f"),
        ("stage_pred", "flow_jepa_stage_prediction", ".5f"),
        ("warp", "flow_jepa_warp_loss", ".5f"),
        ("cycle", "flow_jepa_cycle_loss", ".5f"),
        ("smooth", "flow_jepa_smoothness_loss", ".5f"),
        ("uncert_nll", "flow_jepa_uncertainty_nll", ".5f"),
        ("refine_seq", "flow_jepa_refinement_sequence_loss", ".5f"),
        ("flow_mag", "flow_jepa_patch_flow_magnitude", ".3f"),
        ("confidence", "flow_jepa_confidence_mean", ".3f"),
        ("occlusion", "flow_jepa_occlusion_fraction", ".3f"),
        ("corr_entropy", "flow_jepa_correlation_entropy", ".3f"),
        ("corr_margin", "flow_jepa_correlation_margin", ".3f"),
        ("target_mask", "flow_jepa_future_target_fraction", ".3f"),
        ("stage_target_norm", "flow_jepa_stage_target_norm", ".3f"),
        ("stage_prediction_norm", "flow_jepa_stage_prediction_norm", ".3f"),
        ("stage_window_cos", "flow_jepa_stage_window_cosine", ".3f"),
        ("stage_window_gate", "flow_jepa_stage_to_window_gate", ".3f"),
        ("stage_window_update", "flow_jepa_stage_to_window_update_norm", ".3f"),
        ("goal_norm", "flow_jepa_goal_condition_norm", ".3f"),
        ("goal_pair_cos", "flow_jepa_goal_pair_cosine", ".3f"),
        ("action_mem_norm", "flow_jepa_action_condition_norm", ".3f"),
        ("goal_action_cos", "flow_jepa_goal_action_cosine", ".3f"),
    ):
        append(representation, label, key, spec)

    gradients = ["[v95-stage1-grad]"]
    for label, key in (
        ("view_adapter", "grad_evidence_view_adapter"),
        ("organizer", "grad_evidence_condition_organizer"),
        ("flow_dino", "grad_flow_dino_evidence"),
        ("goal_tokens", "grad_goal_resampler"),
        ("action_history", "grad_action_history_encoder"),
        ("dit_blocks", "grad_dit_blocks"),
        ("global_preclip", "grad"),
    ):
        append(gradients, label, key, ".2e")
    gradients.extend((f"lr={learning_rate:.3e}", f"sec_per_batch={seconds_per_batch:.3f}"))
    return "\n".join((" ".join(train), " ".join(representation), " ".join(gradients)))


def _flow_jepa_stage1_epoch_log_line(
    *,
    epoch: int,
    global_step: int,
    train: dict[str, float],
    val: dict[str, float],
) -> str:
    return (
        f"[v95-stage1-epoch] epoch={epoch:03d} step={global_step} "
        f"train_representation={train.get('loss', float('nan')):.6f} "
        f"val_representation={val.get('loss', float('nan')):.6f} "
        f"window_pred={val.get('flow_jepa_future_prediction', float('nan')):.5f} "
        f"stage_pred={val.get('flow_jepa_stage_prediction', float('nan')):.5f} "
        f"stage_window_cos={val.get('flow_jepa_stage_window_cosine', float('nan')):.3f} "
        f"goal_pair_cos={val.get('flow_jepa_goal_pair_cosine', float('nan')):.3f} "
        f"repr_batch_cov={val.get('eval_representation_coverage', 0.0):.2f}"
    )


def _filter_inactive_evidence_epoch_metrics(
    metrics: dict[str, float],
) -> dict[str, float]:
    """Drop zero placeholders from decoder families that V94 does not instantiate."""

    inactive_zero_prefixes = (
        "adaptive_cvae_",
        "grad_hierarchical_mmdit_",
        "grad_intent_",
        "grad_latent_cvae_",
        "grad_owned_",
        "grad_residual_action_flow",
        "hierarchical_mmdit_",
        "intent_",
        "latent_cvae_",
        "owned_",
    )
    inactive_zero_keys = {
        "arm_fm_null",
        "arm_fm_null_output_fraction",
        "arm_fm_null_ratio",
        "arm_fm_null_rms",
        "arm_fm_noise_projection_error",
        "arm_fm_target_projection_error",
        "arm_noise_abs_std",
        "arm_noise_delta_std",
        "arm_target_abs_std",
        "arm_target_delta_std",
        "gripper_fm_null",
        "gripper_fm_null_event_hold_ratio",
        "gripper_fm_null_event_rms",
        "gripper_fm_null_hold_rms",
        "gripper_fm_null_output_fraction",
        "gripper_fm_null_ratio",
        "gripper_fm_null_rms",
        "gripper_fm_target_energy_ratio",
        "gripper_fm_target_projection_error",
    }
    return {
        key: value
        for key, value in metrics.items()
        if not (
            abs(float(value)) <= 1e-12
            and (key in inactive_zero_keys or key.startswith(inactive_zero_prefixes))
        )
    }


def _evidence_epoch_log_line(
    *,
    epoch: int,
    global_step: int,
    train: dict[str, float],
    val: dict[str, float],
) -> str:
    """Human-readable epoch summary; the JSONL remains the full record."""

    log_version = _evidence_log_version(train, val)

    train_parts = [
        f"[{log_version}-epoch] epoch={epoch:03d}",
        f"step={global_step}",
        f"loss_total={train.get('loss', float('nan')):.6f}",
        f"flow_loss={train.get('physical_flow', float('nan')):.6f}",
    ]
    groups = [
        name
        for name in ("action", "representation", "rollout", "execution", "latent", "layer")
        if f"loss_group_{name}" in train and abs(float(train[f"loss_group_{name}"])) > 1e-12
    ]
    if groups:
        train_parts.append(
            "loss_groups="
            + "/".join(f"{name}:{train[f'loss_group_{name}']:.5f}" for name in groups)
        )
    if "loss_ledger_residual" in train:
        train_parts.append(f"ledger_gap={train['loss_ledger_residual']:+.2e}")

    val_parts = [
        f"[{log_version}-val]",
        f"action_rmse={val.get('full_rmse', float('nan')):.5f}",
        f"first_rmse={val.get('first_rmse', float('nan')):.5f}",
        f"first8_rmse={val.get('first8_rmse', float('nan')):.5f}",
        f"tail_rmse={val.get('tail_rmse', float('nan')):.5f}",
        f"tail_first_ratio={val.get('tail_first_ratio', float('nan')):.3f}",
        f"arm_rmse={val.get('arm_full_rmse', float('nan')):.5f}",
        f"grip_rmse={val.get('gripper_full_rmse', float('nan')):.5f}",
        f"grip_event_ratio={val.get('gripper_event_ratio', float('nan')):.3f}",
        f"grip_events_pred={val.get('gripper_pred_events', float('nan')):.0f}",
        f"grip_events_target={val.get('gripper_target_events', float('nan')):.0f}",
        "grip_event="
        f"p:{val.get('gripper_precision', float('nan')):.3f}/"
        f"r:{val.get('gripper_recall', float('nan')):.3f}/"
        f"f1:{val.get('gripper_f1', float('nan')):.3f}",
        "event_head="
        f"p:{val.get('event_head_precision', float('nan')):.3f}/"
        f"r:{val.get('event_head_recall', float('nan')):.3f}/"
        f"f1:{val.get('event_head_f1', float('nan')):.3f}",
        f"event_head_events_pred={val.get('event_head_pred_events', float('nan')):.0f}",
        f"event_head_events_target={val.get('event_head_target_events', float('nan')):.0f}",
        f"event_head_minus_decoded_f1={val.get('event_head_minus_decoded_gripper_f1', float('nan')):+.3f}",
        "motion_head="
        f"p:{val.get('motion_head_precision', float('nan')):.3f}/"
        f"r:{val.get('motion_head_recall', float('nan')):.3f}/"
        f"f1:{val.get('motion_head_f1', float('nan')):.3f}",
        f"proposal_mse_gain={val.get('proposal_utility_mse_gain', float('nan')):+.3e}",
        f"proposal_batch_cov={val.get('eval_proposal_ablation_coverage', 0.0):.2f}",
        f"balanced_score={val.get('balanced_score', float('nan')):.5f}",
        f"deploy_gate={val.get('deploy_eligible', 0.0):.0f}",
    ]
    if log_version in _BALANCED_FLOW_JEPA_LOG_VERSIONS:
        action_band_keys = sorted(
            (key for key in val if key.startswith("action_band_") and key.endswith("_rmse")),
            key=lambda key: int(key.removeprefix("action_band_").split("_", 1)[0]),
        )
        if action_band_keys:
            val_parts.append(
                "action_band_rmse="
                + "/".join(
                    f"{key.removeprefix('action_band_').removesuffix('_rmse')}:{val[key]:.5f}"
                    for key in action_band_keys
                )
            )
    execution_ablation_names = (
        "hard",
        "neutral",
        "full_capacity",
        "three_basis_reduction",
    )
    available_ablations = [
        name for name in execution_ablation_names if f"execution_ablation_{name}_full_rmse" in val
    ]
    if available_ablations:
        val_parts.append(
            "execution_ablation_rmse="
            + "/".join(
                f"{name}:{val[f'execution_ablation_{name}_full_rmse']:.5f}"
                for name in available_ablations
            )
        )
        val_parts.append(
            f"execution_ablation_cov={val.get('eval_execution_ablation_coverage', 0.0):.2f}"
        )

    if log_version in _FLOW_JEPA_LOG_VERSIONS:
        future_validation_label = "jepa_future" if log_version != "v95" else "jepa_window"
        for label, key, spec in (
            (future_validation_label, "flow_jepa_future_prediction", ".5f"),
            ("jepa_change", "flow_jepa_future_change_direction", ".5f"),
            ("change_obj", "flow_jepa_future_change", ".5f"),
            ("jepa_stage", "flow_jepa_stage_prediction", ".5f"),
            ("patch_warp", "flow_jepa_warp_loss", ".5f"),
            ("identity_adv", "flow_jepa_identity_advantage_loss", ".5f"),
            ("static_identity", "flow_jepa_static_identity_loss", ".5f"),
            ("patch_cycle", "flow_jepa_cycle_loss", ".5f"),
            ("patch_flow", "flow_jepa_patch_flow_magnitude", ".3f"),
            ("patch_conf", "flow_jepa_confidence_mean", ".3f"),
            ("patch_occ", "flow_jepa_occlusion_fraction", ".3f"),
            ("future_raw_delta", "flow_jepa_future_raw_delta_loss", ".5f"),
            (
                "future_reliable_norm",
                "flow_jepa_future_reliable_normalized_loss",
                ".5f",
            ),
            (
                "future_reliability",
                "flow_jepa_future_change_reliability",
                ".3f",
            ),
            (
                "future_reference_scale",
                "flow_jepa_future_current_reference_scale",
                ".3f",
            ),
            (
                "future_normalization_scale",
                "flow_jepa_future_normalization_scale",
                ".3f",
            ),
            (
                "future_direction_floor",
                "flow_jepa_future_direction_floor_min",
                ".3e",
            ),
            (
                "horizon_address",
                "flow_jepa_horizon_address",
                ".5f",
            ),
            (
                "horizon_address_teacher_rel",
                "flow_jepa_horizon_address_teacher_reliability",
                ".3f",
            ),
            (
                "horizon_address_teacher_entropy",
                "flow_jepa_horizon_address_teacher_entropy",
                ".3f",
            ),
            (
                "horizon_address_pred_entropy",
                "flow_jepa_horizon_address_predicted_entropy",
                ".3f",
            ),
            (
                "horizon_address_update",
                "flow_jepa_horizon_address_update_rms",
                ".3f",
            ),
            (
                "horizon_address_ratio",
                "flow_jepa_horizon_address_update_ratio",
                ".3f",
            ),
            (
                "horizon_address_route_entropy",
                "flow_jepa_horizon_address_route_entropy",
                ".3f",
            ),
            (
                "horizon_address_variation",
                "flow_jepa_horizon_address_variation",
                ".3f",
            ),
            (
                "online_address_write",
                "flow_jepa_online_horizon_address_write_rms",
                ".3f",
            ),
            (
                "progressive_address",
                "flow_jepa_progressive_grounding_address",
                ".0f",
            ),
            (
                "g1_coarse_entropy",
                "flow_jepa_progressive_g1_coarse_entropy",
                ".3f",
            ),
            (
                "g2_fine_entropy",
                "flow_jepa_progressive_g2_fine_entropy",
                ".3f",
            ),
            (
                "g3_summary",
                "flow_jepa_progressive_g3_summary_rms",
                ".3f",
            ),
            (
                "g3_sem_summary",
                "flow_jepa_progressive_g3_semantic_summary_rms",
                ".3f",
            ),
            (
                "g3_app_summary",
                "flow_jepa_progressive_g3_appearance_summary_rms",
                ".3f",
            ),
            (
                "g3_geo_summary",
                "flow_jepa_progressive_g3_geometry_summary_rms",
                ".3f",
            ),
            (
                "world_address_entropy",
                "flow_jepa_progressive_world_posterior_entropy",
                ".3f",
            ),
            (
                "world_horizon_variation",
                "flow_jepa_progressive_world_horizon_variation",
                ".3f",
            ),
            (
                "world_source_max",
                "flow_jepa_progressive_world_source_prior_max",
                ".3f",
            ),
            (
                "world_source_variation",
                "flow_jepa_progressive_world_source_horizon_variation",
                ".3f",
            ),
            (
                "policy_world_prior",
                "flow_jepa_progressive_policy_world_prior_rms",
                ".3f",
            ),
            ("typed_raw", "flow_jepa_coordinate_typed_raw_detail", ".0f"),
            (
                "structured_ownership",
                "flow_jepa_structured_ownership_bottleneck",
                ".0f",
            ),
            (
                "g2_owner_sem_app_l1",
                "flow_jepa_progressive_g2_semantic_appearance_posterior_l1",
                ".3f",
            ),
            (
                "g2_owner_app_geo_l1",
                "flow_jepa_progressive_g2_appearance_geometry_posterior_l1",
                ".3f",
            ),
            (
                "g3_owner_sem_app_l1",
                "flow_jepa_progressive_g3_semantic_appearance_slot_l1",
                ".3f",
            ),
            (
                "g3_sem_owner_rms",
                "flow_jepa_progressive_g3_semantic_owner_sidecar_rms",
                ".3f",
            ),
            (
                "g3_app_owner_rms",
                "flow_jepa_progressive_g3_appearance_owner_sidecar_rms",
                ".3f",
            ),
            (
                "g3_geo_owner_rms",
                "flow_jepa_progressive_g3_geometry_owner_sidecar_rms",
                ".3f",
            ),
            (
                "world_public_ratio",
                "flow_jepa_progressive_world_public_ratio",
                ".3f",
            ),
            (
                "world_innovation",
                "flow_jepa_progressive_world_horizon_innovation_rms",
                ".3f",
            ),
            (
                "world_owner_slot_contract_min",
                "flow_jepa_progressive_world_owner_slot_contract_min",
                ".3f",
            ),
            (
                "world_owner_source_contract_min",
                "flow_jepa_progressive_world_owner_source_contract_min",
                ".3f",
            ),
            (
                "p1_owner_fine_l1",
                "flow_jepa_typed_p1_appearance_geometry_fine_l1",
                ".3f",
            ),
            (
                "p1_owner_route_l1",
                "flow_jepa_typed_p1_semantic_appearance_route_l1",
                ".3f",
            ),
            ("literal_rgb", "flow_jepa_literal_rgb_chart_rms", ".3f"),
            (
                "future_transport_offset",
                "flow_jepa_progressive_future_transport_offset_rms",
                ".3f",
            ),
            (
                "future_visibility",
                "flow_jepa_progressive_future_transport_visibility_mean",
                ".3f",
            ),
            (
                "future_transport_variation",
                "flow_jepa_progressive_future_transport_horizon_variation",
                ".3f",
            ),
            (
                "future_transport_spatial_logit",
                "flow_jepa_progressive_future_transport_spatial_logit_rms",
                ".3f",
            ),
            (
                "p1_future_transport_logit",
                "flow_jepa_typed_p1_future_transport_logit_rms",
                ".3f",
            ),
            ("p1_micro_value", "flow_jepa_typed_p1_micro_value_rms", ".3f"),
            (
                "p1_spatial_variation",
                "flow_jepa_typed_p1_spatial_variation",
                ".3f",
            ),
            ("p2_detail_output", "flow_jepa_typed_p2_output_rms", ".3f"),
            ("p1_query_rows", "flow_jepa_p1_query_rows", ".0f"),
            ("p2_query_rows", "flow_jepa_p2_query_rows", ".0f"),
            (
                "p1_query_chunk",
                "flow_jepa_address_query_chunk_actual",
                ".0f",
            ),
            (
                "p1_checkpoint_configured",
                "flow_jepa_typed_p1_activation_checkpoint",
                ".0f",
            ),
            (
                "p1_checkpoint_active",
                "flow_jepa_typed_p1_activation_checkpoint_active",
                ".0f",
            ),
            (
                "horizon_cos_seed",
                "flow_jepa_online_address_boundary_seed_adjacent_cosine",
                ".3f",
            ),
            (
                "horizon_cos_g3",
                "flow_jepa_online_address_boundary_post_g3_adjacent_cosine",
                ".3f",
            ),
            (
                "horizon_cos_address",
                "flow_jepa_online_address_boundary_post_address_adjacent_cosine",
                ".3f",
            ),
            (
                "horizon_cos_w3",
                "flow_jepa_online_address_boundary_post_w3_adjacent_cosine",
                ".3f",
            ),
            (
                "horizon_cos_interval",
                "flow_jepa_online_address_boundary_post_interval_adjacent_cosine",
                ".3f",
            ),
            (
                "interval_stage",
                "flow_jepa_interval_stage",
                ".5f",
            ),
            (
                "interval_stage_raw",
                "flow_jepa_interval_stage_raw",
                ".5f",
            ),
            (
                "interval_stage_reliable",
                "flow_jepa_interval_stage_normalized",
                ".5f",
            ),
            (
                "interval_stage_direction",
                "flow_jepa_interval_stage_direction",
                ".5f",
            ),
            (
                "interval_stage_direction_floor",
                "flow_jepa_interval_stage_direction_floor_min",
                ".3e",
            ),
            (
                "interval_stage_endpoint",
                "flow_jepa_interval_stage_endpoint",
                ".5f",
            ),
            (
                "interval_stage_target_scale",
                "flow_jepa_interval_stage_target_scale",
                ".3f",
            ),
            (
                "interval_stage_reliability",
                "flow_jepa_interval_stage_reliability",
                ".3f",
            ),
        ):
            source = val if key in val else train
            if key in source:
                val_parts.append(f"{label}={format(source[key], spec)}")
        horizon_keys = sorted(
            {
                key
                for source in (val, train)
                for key in source
                if key.startswith("flow_jepa_future_horizon_")
                and key.removeprefix("flow_jepa_future_horizon_").isdigit()
            },
            key=lambda key: int(key.rsplit("_", 1)[-1]),
        )
        for key in horizon_keys:
            source = val if key in val else train
            val_parts.append(f"future_h{key.rsplit('_', 1)[-1]}={source[key]:.5f}")
        reliable_offsets = sorted(
            {
                int(match.group(1))
                for source in (val, train)
                for key in source
                for match in (
                    re.fullmatch(
                        r"flow_jepa_future_horizon_(\d+)_active_loss",
                        key,
                    ),
                )
                if match is not None
            }
        )
        for label, suffix in (
            ("future_direction", "active_direction"),
            ("future_active", "active_loss"),
        ):
            entries: list[str] = []
            for offset in reliable_offsets:
                key = f"flow_jepa_future_horizon_{offset}_{suffix}"
                source = val if key in val else train
                if key in source:
                    entries.append(f"{offset}:{source[key]:.4f}")
            if entries:
                val_parts.append(f"{label}=" + "/".join(entries))
        interval_offsets = sorted(
            {
                int(match.group(1))
                for source in (val, train)
                for key in source
                for match in (
                    re.fullmatch(
                        r"flow_jepa_interval_stage_horizon_(\d+)_loss",
                        key,
                    ),
                )
                if match is not None
            }
        )
        for offset in interval_offsets:
            parts: list[str] = []
            for label, suffix, spec in (
                ("l", "loss", ".4f"),
                ("r", "reliability", ".3f"),
                ("w", "write_rms", ".3f"),
            ):
                key = f"flow_jepa_interval_stage_horizon_{offset}_{suffix}"
                source = val if key in val else train
                if key in source:
                    parts.append(f"{label}:{format(source[key], spec)}")
            if parts:
                val_parts.append(f"interval_h{offset}=" + "/".join(parts))
        if log_version == "v96":
            for label, key, spec in (
                ("detail_gate", "flow_jepa_detail_gate_mean", ".3f"),
                ("address_flow", "flow_jepa_address_flow_mass", ".3f"),
                ("address_fallback", "flow_jepa_address_fallback_mass", ".3f"),
                ("horizon_cos", "flow_jepa_horizon_adjacent_cosine", ".3f"),
            ):
                source = val if key in val else train
                if key in source:
                    val_parts.append(f"{label}={format(source[key], spec)}")
        if log_version in _RAW_FLOW_JEPA_LOG_VERSIONS:
            for label, key, spec in (
                ("raw_high_grid", "flow_jepa_raw_high_grid_size", ".0f"),
                ("raw_flow", "flow_jepa_raw_flow_magnitude", ".3f"),
                ("raw_flow_grid", "flow_jepa_raw_flow_grid_magnitude", ".3f"),
                ("seed_reliability", "flow_jepa_raw_seed_reliability", ".3f"),
                (
                    "mid_bound_compress",
                    "flow_jepa_raw_mid_boundary_compression",
                    ".3f",
                ),
                (
                    "high_bound_compress",
                    "flow_jepa_raw_high_boundary_compression",
                    ".3f",
                ),
                ("raw_boundary", "flow_jepa_raw_boundary_penalty", ".4f"),
                ("raw_valid", "flow_jepa_raw_valid_fraction", ".3f"),
                ("zero_warp", "flow_jepa_raw_identity_warp_error", ".4f"),
                ("warp_gain", "flow_jepa_raw_warp_gain_over_zero", "+.4f"),
                ("moving_gain", "flow_jepa_raw_moving_warp_gain", "+.4f"),
                ("static_gain", "flow_jepa_raw_static_warp_gain", "+.4f"),
                (
                    "moving_corr_entropy",
                    "flow_jepa_raw_moving_correlation_entropy",
                    ".3f",
                ),
                (
                    "moving_corr_margin",
                    "flow_jepa_raw_moving_correlation_margin",
                    ".3f",
                ),
                ("motion_visible", "flow_jepa_raw_observable_motion_fraction", ".3f"),
                ("raw_precision", "flow_jepa_raw_detail_precision_mean", ".3f"),
                (
                    "raw_detail_share"
                    if log_version in _COMPLEMENTARY_FLOW_JEPA_LOG_VERSIONS
                    else "raw_address_flow",
                    "flow_jepa_raw_address_flow_mass",
                    ".3f",
                ),
                (
                    "raw_base_share"
                    if log_version in _COMPLEMENTARY_FLOW_JEPA_LOG_VERSIONS
                    else "raw_address_fallback",
                    "flow_jepa_raw_address_fallback_mass",
                    ".3f",
                ),
                (
                    "address_separation",
                    "flow_jepa_raw_address_center_separation",
                    ".3f",
                ),
                (
                    "detail_address_concentration"
                    if log_version in _COMPLEMENTARY_FLOW_JEPA_LOG_VERSIONS
                    else "address_logit_gain",
                    "flow_jepa_raw_address_logit_advantage",
                    "+.3f",
                ),
                (
                    "address_zero_delta",
                    "flow_jepa_raw_address_zero_flow_value_delta",
                    ".3f",
                ),
                (
                    "address_shuffle_delta",
                    "flow_jepa_raw_address_shuffled_flow_value_delta",
                    ".3f",
                ),
                ("horizon_cos", "flow_jepa_horizon_adjacent_cosine", ".3f"),
                (
                    "query_horizon_cos",
                    "flow_jepa_future_query_adjacent_cosine",
                    ".3f",
                ),
                (
                    "history_entropy",
                    "flow_jepa_perceptual_history_entropy",
                    ".3f",
                ),
                (
                    "horizon_step_update",
                    "flow_jepa_horizon_transition_update_rms",
                    ".3f",
                ),
                (
                    "horizon_state_delta",
                    "flow_jepa_horizon_transition_state_delta",
                    ".3f",
                ),
                ("world_xy_residual", "flow_jepa_world_spatial_residual_norm", ".3e"),
                (
                    "world_anchor_residual",
                    "flow_jepa_world_anchor_camera_residual_norm",
                    ".3f",
                ),
                ("late_detail_update", "flow_jepa_late_detail_update_norm", ".3f"),
                (
                    "late_detail_ratio",
                    "flow_jepa_late_detail_trajectory_ratio",
                    ".3f",
                ),
                (
                    "late_detail_scale",
                    "flow_jepa_late_detail_fixed_scale",
                    ".3f",
                ),
                (
                    "late_detail_tokens",
                    "flow_jepa_late_detail_token_count",
                    ".0f",
                ),
            ):
                source = val if key in val else train
                if key in source:
                    val_parts.append(f"{label}={format(source[key], spec)}")

    probe_parts = [f"[{log_version}-probe]"]
    for label, key, spec in (
        ("z_zero_cond_delta", "sample_evidence_z_zero_condition_delta", ".4e"),
        ("z_shuffle_cond_delta", "sample_evidence_z_shuffle_condition_delta", ".4e"),
        ("capacity_gate_mass", "sample_evidence_mmd_it_capacity_gate_mass", ".5f"),
        ("effective_basis_mass", "sample_evidence_mmd_it_effective_basis_mass", ".3f"),
        ("route_soft", "sample_evidence_mmd_it_dynamic_route_next_fraction", ".3f"),
        ("route_hard", "sample_evidence_mmd_it_hard_route_next_fraction", ".3f"),
        ("dwell_soft", "sample_evidence_mmd_it_dwell_expected", ".3f"),
        ("dwell_hard", "sample_evidence_mmd_it_hard_dwell_expected", ".3f"),
        ("terminal_prior", "sample_evidence_mmd_it_terminal_prior_weight", ".3f"),
        ("terminal_probability", "sample_evidence_mmd_it_terminal_probability", ".3f"),
        ("hard_terminal_fraction", "sample_evidence_mmd_it_hard_terminal_fraction", ".3f"),
        ("nonexp_violation", "sample_evidence_mmd_it_nonexpansive_violation", ".1e"),
        (
            "late_detail_update",
            "sample_flow_jepa_late_detail_update_norm",
            ".3f",
        ),
        (
            "late_detail_ratio",
            "sample_flow_jepa_late_detail_trajectory_ratio",
            ".3f",
        ),
        (
            "late_detail_entropy",
            "sample_flow_jepa_late_detail_attention_entropy",
            ".3f",
        ),
        (
            "late_detail_max",
            "sample_flow_jepa_late_detail_attention_max",
            ".3f",
        ),
        (
            "late_detail_scale",
            "sample_flow_jepa_late_detail_fixed_scale",
            ".3f",
        ),
        (
            "late_detail_tokens",
            "sample_flow_jepa_late_detail_token_count",
            ".0f",
        ),
        (
            "world_xy_residual",
            "sample_flow_jepa_world_spatial_residual_norm",
            ".2e",
        ),
        (
            "world_anchor_residual",
            "sample_flow_jepa_world_anchor_camera_residual_norm",
            ".3f",
        ),
        ("probe_batch_cov", "eval_sampling_diagnostic_coverage", ".2f"),
        ("repr_batch_cov", "eval_representation_coverage", ".2f"),
    ):
        if key in val and abs(float(val[key])) > 1e-12:
            probe_parts.append(f"{label}={format(val[key], spec)}")
    return "\n".join((" ".join(train_parts), " ".join(val_parts), " ".join(probe_parts)))


def _format_hierarchical_stage_usage(row: dict[str, float]) -> str:
    if row.get("hierarchical_mmdit_memory_stage_execution_decoupled", 0.0) > 0.5:
        return "memory-only"
    count = int(round(row.get("hierarchical_mmdit_operator_stage_count", 0.0)))
    if count <= 0:
        prefix = "hierarchical_mmdit_stage_"
        suffix = "_usage"
        indices = []
        for key in row:
            if key.startswith(prefix) and key.endswith(suffix):
                middle = key[len(prefix) : -len(suffix)]
                if middle.isdigit():
                    indices.append(int(middle))
        count = max(indices, default=-1) + 1
    return (
        "/".join(
            f"{row.get(f'hierarchical_mmdit_stage_{index}_usage', 0.0):.2f}"
            for index in range(count)
        )
        or "-"
    )


def _format_hierarchical_block_usage(row: dict[str, float]) -> str:
    count = int(round(row.get("hierarchical_mmdit_refine_block_count", 0.0)))
    return (
        "/".join(
            f"{row.get(f'hierarchical_mmdit_block_{index}_usage', 0.0):.2f}"
            for index in range(count)
        )
        or "-"
    )


def _owned_serial_log_line(
    row: dict[str, float],
    *,
    epoch: int,
    batch_index: int,
    learning_rate: float,
    seconds_per_batch: float,
) -> str:
    """High-signal batch line for the clean owned-evidence decoder.

    Full scalar diagnostics remain in the epoch JSON.  This line deliberately
    omits retired latent-CVAE fields instead of rendering them as misleading
    zeroes beside the active serial decoder gauges.
    """
    return (
        f"[v39-layer] epoch={epoch:03d} batch={batch_index:04d} loss={row['loss']:.6f} "
        f"pflow={row['physical_flow']:.6f} "
        f"pfn={row.get('physical_flow_native', 0.0):.6f} "
        f"afmd={row.get('arm_fm_per_dim', 0.0):.5f} "
        f"gfmf={row.get('gripper_fm_field', 0.0):.5f} "
        f"gfar={row.get('gripper_arm_fm_ratio', 0.0):.3f} "
        f"anull={row.get('arm_fm_null_output_fraction', 0.0):.4f} "
        f"gnull={row.get('gripper_fm_null_output_fraction', 0.0):.4f} "
        f"asrc={row.get('arm_source_residual_rms', 0.0):.3f}/"
        f"{row.get('arm_source_delta_rms', 0.0):.3f}/"
        f"{row.get('arm_source_acceleration_rms', 0.0):.3f} "
        f"asexp={row.get('arm_source_expected_rms', 0.0):.3f}/"
        f"{row.get('arm_source_expected_delta_rms', 0.0):.3f}/"
        f"{row.get('arm_source_expected_acceleration_rms', 0.0):.3f} "
        f"asgeo={row.get('arm_source_covariance_effective_dimension', 0.0):.2f}/"
        f"{row.get('arm_source_covariance_condition', 0.0):.1f} "
        f"asfirst={row.get('arm_source_first_step_rms', 0.0):.3f}/"
        f"{row.get('arm_source_expected_first_step_std', 0.0):.3f} "
        f"astail={row.get('arm_source_expected_terminal_std', 0.0):.3f} "
        f"hmchart={row.get('hierarchical_mmdit_native_time_chart_active', 0.0):.0f}/"
        f"{row.get('hierarchical_mmdit_native_time_chart_complete', 0.0):.0f}/"
        f"{row.get('hierarchical_mmdit_native_time_position_alignment', 0.0):.0f} "
        f"hmspec={row.get('hierarchical_mmdit_spectral_state', 0.0):.0f}/"
        f"{row.get('hierarchical_mmdit_spectral_final_progress', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_spectral_final_arm_mask', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_spectral_final_gripper_mask', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_spectral_competition_loss', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_spectral_coefficient_flow_mse', 0.0):.4f} "
        f"hmswarp={row.get('hierarchical_mmdit_spectral_frequency_warp_rms', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_spectral_frequency_spacing_min', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_spectral_frequency_spacing_max', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_spectral_final_controller_global_shift_rms', 0.0):.3f} "
        f"hmsgeo={row.get('hierarchical_mmdit_spectral_flow_roundtrip_mse', 0.0):.1e}/"
        f"{row.get('hierarchical_mmdit_spectral_bridge_null_fraction', 0.0):.1e}/"
        f"{row.get('hierarchical_mmdit_spectral_target_tangent_null_fraction', 0.0):.1e}/"
        f"{row.get('hierarchical_mmdit_spectral_prediction_tangent_null_fraction', 0.0):.1e} "
        f"hmtan={row.get('hierarchical_mmdit_velocity_arm_tangent_null_ratio', 0.0):.1e}/"
        f"{row.get('hierarchical_mmdit_velocity_gripper_tangent_null_ratio', 0.0):.1e}/"
        f"{row.get('hierarchical_mmdit_noisy_gripper_chart_null_ratio', 0.0):.1e} "
        f"decode={row.get('decoded_action', 0.0):.6f} "
        f"rollout={row.get('rollout_dynamics', 0.0):.6f} "
        f"rvar={row.get('rollout_variance', 0.0):.4f} "
        f"rnorm={row.get('rollout_norm', 0.0):.4f} "
        f"rstep={row.get('rollout_milestone_delta_match', 0.0):.4f} "
        f"first8={row.get('first8_physical_flow', 0.0):.6f} "
        f"tail={row.get('tail_physical_flow', 0.0):.6f} "
        f"delta={row.get('rollout_delta', 0.0):.6f} "
        f"contrast={row.get('rollout_contrast', 0.0):.6f} "
        f"d_shuffle={row.get('rollout_delta_shuffle', 0.0):.6f} "
        f"dshuf_src={row.get('rollout_effect_change_shuffle', 0.0):.3e}/"
        f"{row.get('rollout_delta_state_shuffle', 0.0):.3e}/"
        f"{row.get('rollout_effect_change_state_shuffle', 0.0):.3e} "
        f"stdr={row.get('rollout_pred_std_ratio', 0.0):.4f} "
        f"dnratio={row.get('rollout_milestone_delta_norm_ratio', 0.0):.4f} "
        f"event={row.get('event', 0.0):.6f} "
        f"icgs={row.get('intent_global_stage_cosine', 0.0):.3f} "
        f"icgr={row.get('intent_global_read_cosine', 0.0):.3f} "
        f"icsr={row.get('intent_stage_read_cosine', 0.0):.3f} "
        f"icdiv={row.get('intent_global_batch_diversity', 0.0):.2f}/"
        f"{row.get('intent_stage_batch_diversity', 0.0):.2f}/"
        f"{row.get('intent_read_batch_diversity', 0.0):.2f} "
        f"hmdu={row.get('hierarchical_mmdit_action_update_norm', 0.0):.3f} "
        f"hmur={row.get('hierarchical_mmdit_action_update_ratio', 0.0):.3f} "
        f"hmcan={row.get('hierarchical_mmdit_action_serial_cancellation_fraction', 0.0):.3f} "
        f"hmorth={row.get('hierarchical_mmdit_action_serial_cancellation_orthogonal_baseline', 0.0):.3f} "
        f"hmxcan={row.get('hierarchical_mmdit_action_serial_cancellation_excess', 0.0):+.3f} "
        f"hmbdot={row.get('hierarchical_mmdit_action_branch_weighted_cosine', 0.0):+.3f} "
        f"hmcos={row.get('hierarchical_mmdit_action_state_cosine', 0.0):.3f} "
        f"hmbcos={row.get('hierarchical_mmdit_action_noisy_stage_cosine', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_stage_low_cosine', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_noisy_low_cosine', 0.0):.3f} "
        f"hmnu={row.get('hierarchical_mmdit_action_noisy_update_norm', 0.0):.3f} "
        f"hmsu={row.get('hierarchical_mmdit_action_stage_update_norm', 0.0):.3f} "
        f"hmlu={row.get('hierarchical_mmdit_action_low_update_norm', 0.0):.3f} "
        f"hmnf={row.get('hierarchical_mmdit_action_noisy_update_fraction', 0.0):.3f} "
        f"hmsf={row.get('hierarchical_mmdit_action_stage_update_fraction', 0.0):.3f} "
        f"hmlf={row.get('hierarchical_mmdit_action_low_update_fraction', 0.0):.3f} "
        f"hmnw={row.get('hierarchical_mmdit_noisy_to_workspace_update_ratio', 0.0):.3f} "
        f"hmdepth={row.get('hierarchical_mmdit_action_noisy_depth_ratio', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_action_stage_depth_ratio', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_action_low_depth_ratio', 0.0):.2f} "
        f"hmraw={row.get('hierarchical_mmdit_action_noisy_raw_depth_ratio', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_action_stage_raw_depth_ratio', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_action_low_raw_depth_ratio', 0.0):.2f} "
        f"hmkeep={row.get('hierarchical_mmdit_action_self_update_keep', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_action_noisy_update_keep', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_action_stage_update_keep', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_action_low_update_keep', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_action_ffn_update_keep', 0.0):.2f} "
        f"hmedepth={row.get('hierarchical_mmdit_action_noisy_effective_depth', 0.0):.1f}/"
        f"{row.get('hierarchical_mmdit_action_stage_effective_depth', 0.0):.1f}/"
        f"{row.get('hierarchical_mmdit_action_low_effective_depth', 0.0):.1f} "
        f"hmhost={row.get('hierarchical_mmdit_action_noisy_host_update_rms', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_stage_host_update_rms', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_low_host_update_rms', 0.0):.3f} "
        f"hmcover={row.get('hierarchical_mmdit_action_noisy_subspace_energy_fraction', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_stage_subspace_energy_fraction', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_low_subspace_energy_fraction', 0.0):.3f} "
        f"hmremove={row.get('hierarchical_mmdit_action_noisy_removed_fraction', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_stage_removed_fraction', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_low_removed_fraction', 0.0):.3f} "
        f"hmdepthreg={row.get('hierarchical_mmdit_operator_contraction_progress', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_depth_usage_regularizer', 0.0):.4f} "
        f"hmdwell={row.get('hierarchical_mmdit_learned_execution_active', 0.0):.0f}/"
        f"{row.get('hierarchical_mmdit_value_dwell_warmup_active', 0.0):.0f}/"
        f"{row.get('hierarchical_mmdit_value_dwell_shadow_active', 0.0):.0f}/"
        f"{row.get('hierarchical_mmdit_operation_decision_shadow_active', 0.0):.0f} "
        f"hmpfx={row.get('hierarchical_mmdit_prefix_error_initial', 0.0):.4f}/"
        f"{row.get('hierarchical_mmdit_prefix_error_final', 0.0):.4f}/"
        f"{row.get('hierarchical_mmdit_prefix_gain_mean', 0.0):+.4f}/"
        f"{row.get('hierarchical_mmdit_prefix_gain_positive_fraction', 0.0):.2f} "
        f"hmval={row.get('hierarchical_mmdit_operation_value_loss', 0.0):.4f}/"
        f"{row.get('hierarchical_mmdit_operation_value_weight', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_operation_value_target_spread', 0.0):.4f}/"
        f"{row.get('hierarchical_mmdit_operation_value_predicted_spread', 0.0):.4f}/"
        f"{row.get('hierarchical_mmdit_operation_value_correlation', 0.0):+.2f}/"
        f"{row.get('hierarchical_mmdit_operation_value_decision_accuracy', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_operation_candidate_coverage', 0.0):.2f} "
        f"hmvalq={row.get('hierarchical_mmdit_operation_value_target_spread_p25', 0.0):.4f}/"
        f"{row.get('hierarchical_mmdit_operation_value_target_spread_p50', 0.0):.4f}/"
        f"{row.get('hierarchical_mmdit_operation_value_target_spread_p75', 0.0):.4f}/"
        f"{row.get('hierarchical_mmdit_operation_value_reliability', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_operation_value_common_mode_ratio', 0.0):.2f} "
        f"hmctrl={row.get('hierarchical_mmdit_controller_state_direction_participation', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_controller_state_pair_cosine', 0.0):+.2f}/"
        f"{row.get('hierarchical_mmdit_controller_recurrent_change', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_controller_operation_value_rms', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_controller_operation_value_block_spread', row.get('hierarchical_mmdit_controller_operation_value_stage_spread', 0.0)):.3f} "
        f"hmvctx={row.get('hierarchical_mmdit_controller_operation_value_memory_context_rms', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_controller_operation_value_action_context_rms', 0.0):.3f} "
        f"hmpriv={row.get('hierarchical_mmdit_controller_private_pair_cosine', 0.0):+.2f}/"
        f"{row.get('hierarchical_mmdit_controller_private_centered_energy_ratio', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_controller_private_global_energy_ratio', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_controller_private_residual_value_rms', 0.0):.3f} "
        f"hmread={row.get('hierarchical_mmdit_controller_reader_operator_memory_attention', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_controller_reader_spectral_memory_attention', 0.0):.2f} "
        f"hmmem={row.get('hierarchical_mmdit_controller_reader_operator_global_memory_attention', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_controller_reader_operator_private_memory_attention', 0.0):.2f} "
        f"hmrdiv={row.get('hierarchical_mmdit_controller_reader_operator_attention_diversity', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_controller_reader_spectral_attention_local_change', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_controller_reader_family_attention_diversity', 0.0):.3f} "
        f"hmfunc={row.get('hierarchical_mmdit_controller_operator_representation_diversity', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_controller_spectral_representation_local_change', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_controller_state_centered_energy_ratio', 0.0):.3f} "
        f"hmfcand={row.get('hierarchical_mmdit_function_candidate_cosine', 0.0):+.2f}/"
        f"{row.get('hierarchical_mmdit_function_candidate_diversity', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_function_candidate_update_rms', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_function_candidate_update_spread', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_function_candidate_valid_count', 0.0):.2f} "
        f"hmcomp={row.get('hierarchical_mmdit_controller_competition_source_effective_slots', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_controller_competition_source_owner_max', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_controller_competition_slot_load_effective', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_controller_competition_slot_load_max', 0.0):.2f} "
        f"hmcap={row.get('hierarchical_mmdit_controller_operator_raw_depth_mean', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_controller_operator_depth_stage_std', 0.0):.3f} "
        f"hmwi={row.get('owned_workspace_interface_state_norm', 0.0):.2f}/"
        f"{row.get('owned_workspace_interface_state_slot_diversity', 0.0):.2f}/"
        f"{row.get('owned_workspace_interface_low_query_delta_ratio', 0.0):.3f}/"
        f"{row.get('owned_workspace_interface_stage_query_delta_ratio', 0.0):.3f}/"
        f"{row.get('owned_workspace_interface_promote_mean', 0.0):.3f}/"
        f"{row.get('owned_workspace_interface_promote_std', 0.0):.3f} "
        f"hmwic={row.get('owned_workspace_interface_low_control_effective_control_tokens', 0.0):.2f}/"
        f"{row.get('owned_workspace_interface_low_control_load_effective_tokens', 0.0):.2f}/"
        f"{row.get('owned_workspace_interface_low_control_slot_diversity', 0.0):.3f}/"
        f"{row.get('owned_workspace_interface_stage_control_effective_control_tokens', 0.0):.2f}/"
        f"{row.get('owned_workspace_interface_stage_control_load_effective_tokens', 0.0):.2f}/"
        f"{row.get('owned_workspace_interface_stage_control_slot_diversity', 0.0):.3f} "
        f"hmca={row.get('hierarchical_mmdit_controller_source_intent_attention', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_controller_source_flow_time_attention', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_controller_source_refine_time_attention', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_controller_source_action_attention', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_controller_source_evidence_attention', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_controller_source_stage_role_attention', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_controller_source_stage_content_attention', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_controller_source_feedback_attention', 0.0):.2f} "
        f"hmce={row.get('hierarchical_mmdit_controller_evidence_role_geom_attention', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_controller_evidence_role_transition_attention', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_controller_evidence_role_event_attention', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_controller_evidence_role_state_attention', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_controller_evidence_role_layer_attention', 0.0):.2f} "
        f"hmstage={_format_hierarchical_stage_usage(row)} "
        f"hmblock={_format_hierarchical_block_usage(row)} "
        f"hmopgain={row.get('hierarchical_mmdit_action_noisy_operator_gain', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_stage_operator_gain', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_low_operator_gain', 0.0):.3f} "
        f"hmdir={row.get('hierarchical_mmdit_action_noisy_direction_change', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_stage_direction_change', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_low_direction_change', 0.0):.3f} "
        f"hmbcos2={row.get('hierarchical_mmdit_action_noisy_direction_cosine', 0.0):+.2f}/"
        f"{row.get('hierarchical_mmdit_action_stage_direction_cosine', 0.0):+.2f}/"
        f"{row.get('hierarchical_mmdit_action_low_direction_cosine', 0.0):+.2f} "
        f"hmbgain={row.get('hierarchical_mmdit_action_noisy_base_data_gain', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_action_stage_base_data_gain', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_action_low_base_data_gain', 0.0):.2f} "
        f"hmgate={row.get('hierarchical_mmdit_action_self_base_gate', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_noisy_base_gate', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_stage_base_gate', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_low_base_gate', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_ffn_base_gate', 0.0):.3f} "
        f"hmegate={row.get('hierarchical_mmdit_action_self_effective_gate', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_noisy_effective_gate', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_stage_effective_gate', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_low_effective_gate', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_ffn_effective_gate', 0.0):.3f} "
        f"hmkerr={row.get('hierarchical_mmdit_action_noisy_keep_scale_error', 0.0):.1e}/"
        f"{row.get('hierarchical_mmdit_action_stage_keep_scale_error', 0.0):.1e}/"
        f"{row.get('hierarchical_mmdit_action_low_keep_scale_error', 0.0):.1e} "
        f"hmgerr={max(row.get(f'hierarchical_mmdit_action_{name}_gate_scale_error', 0.0) for name in ('self', 'noisy', 'stage', 'low', 'ffn')):.1e} "
        f"hmnrms={row.get('hierarchical_mmdit_action_pre_norm_rms', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_post_norm_rms', 0.0):.3f} "
        f"hexh={row.get('hierarchical_mmdit_executed_steps', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_action_response_rel', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_stage_pressure_rel', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_refine_gain', 0.0):+.4f}/"
        f"{row.get('hierarchical_mmdit_response_gain_corr', 0.0):+.2f}/"
        f"{row.get('hierarchical_mmdit_unresolved_rate', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_budget_exhausted_rate', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_final_block', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_final_stage', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_block_advance_rate', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_stage_advance_rate', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_operation_fixed_path_agreement', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_operation_monotonic_violation', 0.0):.0f} "
        f"hmshadow={row.get('hierarchical_mmdit_shadow_executed_steps', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_shadow_step_saving', 0.0):+.2f}/"
        f"{row.get('hierarchical_mmdit_shadow_refine_error_ratio', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_shadow_operation_stay_rate', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_shadow_operation_advance_rate', 0.0):.2f} "
        f"hmuresp={row.get('hierarchical_mmdit_action_response_arm', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_response_gripper', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_response_arm_null', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_response_gripper_null', 0.0):.3f} "
        f"hmuq={row.get('hierarchical_mmdit_action_response_p25', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_response_p50', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_response_p75', 0.0):.3f} "
        f"hmpq={row.get('hierarchical_mmdit_stage_pressure_p25', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_stage_pressure_p50', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_stage_pressure_p75', 0.0):.3f} "
        f"hmuT50={row.get('hierarchical_mmdit_action_response_t0_p50', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_response_t1_p50', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_response_t2_p50', 0.0):.3f} "
        f"hmpT50={row.get('hierarchical_mmdit_stage_pressure_t0_p50', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_stage_pressure_t1_p50', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_stage_pressure_t2_p50', 0.0):.3f} "
        f"hmucT={row.get('hierarchical_mmdit_response_gain_corr_t0', 0.0):+.2f}/"
        f"{row.get('hierarchical_mmdit_response_gain_corr_t1', 0.0):+.2f}/"
        f"{row.get('hierarchical_mmdit_response_gain_corr_t2', 0.0):+.2f} "
        f"hmpcT={row.get('hierarchical_mmdit_pressure_gain_corr_t0', 0.0):+.2f}/"
        f"{row.get('hierarchical_mmdit_pressure_gain_corr_t1', 0.0):+.2f}/"
        f"{row.get('hierarchical_mmdit_pressure_gain_corr_t2', 0.0):+.2f} "
        f"hmnmax={row.get('hierarchical_mmdit_action_noisy_attention_max', 0.0):.3f} "
        f"hmsmax={row.get('hierarchical_mmdit_action_stage_attention_max', 0.0):.3f} "
        f"hmlmax={row.get('hierarchical_mmdit_action_low_attention_max', 0.0):.3f} "
        f"hmnfT={row.get('hierarchical_mmdit_noisy_update_fraction_t0_sum', 0.0) / max(row.get('hierarchical_mmdit_update_fraction_t0_count', 0.0), 1e-6):.3f}/"
        f"{row.get('hierarchical_mmdit_noisy_update_fraction_t1_sum', 0.0) / max(row.get('hierarchical_mmdit_update_fraction_t1_count', 0.0), 1e-6):.3f}/"
        f"{row.get('hierarchical_mmdit_noisy_update_fraction_t2_sum', 0.0) / max(row.get('hierarchical_mmdit_update_fraction_t2_count', 0.0), 1e-6):.3f} "
        f"hmwfT={row.get('hierarchical_mmdit_workspace_update_fraction_t0_sum', 0.0) / max(row.get('hierarchical_mmdit_update_fraction_t0_count', 0.0), 1e-6):.3f}/"
        f"{row.get('hierarchical_mmdit_workspace_update_fraction_t1_sum', 0.0) / max(row.get('hierarchical_mmdit_update_fraction_t1_count', 0.0), 1e-6):.3f}/"
        f"{row.get('hierarchical_mmdit_workspace_update_fraction_t2_sum', 0.0) / max(row.get('hierarchical_mmdit_update_fraction_t2_count', 0.0), 1e-6):.3f} "
        f"halreff={row.get('hierarchical_mmdit_action_low_role_effective_count', 0.0):.2f} "
        f"hal={row.get('hierarchical_mmdit_action_low_role_geom_attention', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_low_role_transition_attention', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_low_role_event_attention', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_low_role_state_attention', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_low_role_layer_attention', 0.0):.3f} "
        f"olupd={row.get('owned_hierarchical_low_role_geom_update_norm', 0.0):.3f}/"
        f"{row.get('owned_hierarchical_low_role_transition_update_norm', 0.0):.3f}/"
        f"{row.get('owned_hierarchical_low_role_event_update_norm', 0.0):.3f}/"
        f"{row.get('owned_hierarchical_low_role_state_update_norm', 0.0):.3f}/"
        f"{row.get('owned_hierarchical_low_role_layer_update_norm', 0.0):.3f} "
        f"ohsupd={row.get('owned_hierarchical_stage_update_norm', 0.0):.3f} "
        f"ohsret={row.get('owned_hierarchical_stage_retain_mean', 0.0):.3f} "
        f"ohprom={row.get('owned_hierarchical_manager_promote_gate', 0.0):.3f} "
        f"ohprms={row.get('owned_hierarchical_stage_promoted_projected_rms', 0.0):.3f}/"
        f"{row.get('owned_hierarchical_stage_promoted_normalized_rms', 0.0):.3f}/"
        f"{row.get('owned_hierarchical_stage_promoted_realized_scale', 0.0):.3f} "
        f"ohgerr={row.get('owned_hierarchical_stage_promote_gate_scale_error', 0.0):.1e} "
        f"hmwireff={row.get('owned_workspace_interface_role_geom_effective_control_tokens', 0.0):.2f}/"
        f"{row.get('owned_workspace_interface_role_transition_effective_control_tokens', 0.0):.2f}/"
        f"{row.get('owned_workspace_interface_role_event_effective_control_tokens', 0.0):.2f}/"
        f"{row.get('owned_workspace_interface_role_state_effective_control_tokens', 0.0):.2f}/"
        f"{row.get('owned_workspace_interface_role_layer_effective_control_tokens', 0.0):.2f} "
        f"hmdgrad={row.get('grad_hierarchical_mmdit_action', 0.0):.3e} "
        f"hmvgrad={row.get('grad_hierarchical_mmdit_velocity_head', 0.0):.3e} "
        f"icgrad={row.get('grad_intent_contract_compiler', 0.0):.3e} "
        f"owgrad={row.get('grad_owned_workspace', 0.0):.3e} "
        f"hmbgrad={row.get('grad_hierarchical_mmdit_blocks', 0.0):.3e} "
        f"hmbasegrad={row.get('grad_hierarchical_mmdit_shared_base', 0.0):.3e} "
        f"hmwgrad={row.get('grad_hierarchical_mmdit_base_projection', 0.0):.3e} "
        f"hmcopgrad={row.get('grad_hierarchical_mmdit_contractions', 0.0):.3e} "
        f"hmcgrad={row.get('grad_hierarchical_mmdit_contraction_basis', 0.0):.3e} "
        f"hmctrlgrad={row.get('grad_hierarchical_mmdit_unified_controller', 0.0):.3e} "
        f"hmcg={row.get('grad_hierarchical_mmdit_controller_backbone', 0.0):.2e}/"
        f"{row.get('grad_hierarchical_mmdit_controller_operator_controls', 0.0):.2e}/"
        f"{row.get('grad_hierarchical_mmdit_controller_value_reader', 0.0):.2e}/"
        f"{row.get('grad_hierarchical_mmdit_controller_workspace_interface', 0.0):.2e} "
        f"hmcmodgrad={row.get('grad_hierarchical_mmdit_content_modulation', 0.0):.3e} "
        f"hmgategrad={row.get('grad_hierarchical_mmdit_host_gates', 0.0):.3e} "
        f"hmdclip={row.get('grad_hierarchical_mmdit_action_post_clip', 0.0):.3e} "
        f"grad={row['grad']:.3e} lr={learning_rate:.3e} spb={seconds_per_batch:.3f}"
    )


def _module_grad_norm(module: torch.nn.Module, *, reference: Tensor) -> Tensor:
    """Return a detached scalar grad norm for diagnostics only."""
    total = torch.zeros((), device=reference.device, dtype=torch.float32)
    for param in module.parameters():
        if param.grad is None:
            continue
        total = total + param.grad.detach().float().pow(2).sum()
    return total.sqrt()


def _parameter_grad_norm(parameters: Iterable[torch.nn.Parameter], *, reference: Tensor) -> Tensor:
    total = torch.zeros((), device=reference.device, dtype=torch.float32)
    for parameter in parameters:
        if parameter.grad is not None:
            total = total + parameter.grad.detach().float().pow(2).sum()
    return total.sqrt()


def _linear_row_grad_norm(
    modules: Iterable[torch.nn.Linear],
    *,
    start: int,
    stop: int | None,
    reference: Tensor,
) -> Tensor:
    """Gradient norm for an output-row contract inside shared Linear owners."""
    total = torch.zeros((), device=reference.device, dtype=torch.float32)
    for module in modules:
        weight_grad = module.weight.grad
        if weight_grad is not None:
            total = total + weight_grad.detach().float()[start:stop].pow(2).sum()
        if module.bias is not None and module.bias.grad is not None:
            total = total + module.bias.grad.detach().float()[start:stop].pow(2).sum()
    return total.sqrt()


@torch.no_grad()
def _nonfinite_gradient_report(module: torch.nn.Module, *, limit: int = 12) -> str:
    """Identify corrupt parameter gradients only after a clip failure."""

    rows: list[str] = []
    omitted = 0
    for name, parameter in module.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        values = gradient.detach()
        if values.is_sparse:
            values = values.coalesce().values()
        values = values.float()
        finite = torch.isfinite(values)
        if bool(finite.all()):
            continue
        if len(rows) >= max(int(limit), 1):
            omitted += 1
            continue
        finite_values = values[finite]
        finite_max = float(finite_values.abs().max()) if int(finite_values.numel()) else 0.0
        rows.append(
            f"{name}[shape={tuple(parameter.shape)},nan={int(torch.isnan(values).sum())},"
            f"+inf={int(torch.isposinf(values).sum())},-inf={int(torch.isneginf(values).sum())},"
            f"finite_max={finite_max:.3e}]"
        )
    if not rows:
        return "no individual non-finite parameter gradient found"
    suffix = f"; ... {omitted} more" if omitted else ""
    return "; ".join(rows) + suffix


def _clip_grad_norm_or_report(
    parameters: Iterable[torch.nn.Parameter],
    max_norm: float,
    *,
    system: V39PolicySystem,
    epoch: int,
    batch: int,
    label: str,
) -> Tensor:
    """Clip normally, but turn the generic PyTorch error into code evidence."""

    rows = list(parameters)
    try:
        return torch.nn.utils.clip_grad_norm_(
            rows,
            max_norm,
            error_if_nonfinite=True,
        )
    except RuntimeError as error:
        if "non-finite" not in str(error).lower():
            raise
        report = _nonfinite_gradient_report(system)
        raise FloatingPointError(
            "non-finite gradients after backward "
            f"at epoch={epoch} batch={batch} clip={label}: {report}"
        ) from error


def _attach_grad_diagnostics(losses: dict[str, Tensor], system: V39PolicySystem) -> None:
    """Log whether the contract objective reaches the intended modules.

    These values are diagnostics; they are added after backward and before
    optimizer.step, and never participate in the loss.
    """
    reference = losses["loss"]
    planner = system.planner
    losses["grad_dit_blocks"] = _module_grad_norm(planner.blocks, reference=reference)
    if bool(int(getattr(system.policy_config, "flow_jepa_role_hierarchy", 0))):
        grounding_stop = int(system.policy_config.flow_jepa_grounding_blocks)
        world_stop = grounding_stop + int(system.policy_config.flow_jepa_world_blocks)
        losses["grad_dit_grounding_blocks"] = _parameter_grad_norm(
            (
                parameter
                for block in planner.blocks[:grounding_stop]
                for parameter in block.parameters()
            ),
            reference=reference,
        )
        losses["grad_dit_world_blocks"] = _parameter_grad_norm(
            (
                parameter
                for block in planner.blocks[grounding_stop:world_stop]
                for parameter in block.parameters()
            ),
            reference=reference,
        )
        losses["grad_dit_policy_blocks"] = _parameter_grad_norm(
            (
                parameter
                for block in planner.blocks[world_stop:]
                for parameter in block.parameters()
            ),
            reference=reference,
        )
    losses["grad_layer_contract_adapters"] = _module_grad_norm(
        planner.layer_contract_heads, reference=reference
    )
    if getattr(planner, "layer_fm_probe", None) is not None:
        losses["grad_layer_fm_probe"] = _module_grad_norm(
            planner.layer_fm_probe, reference=reference
        )
    if getattr(planner, "layer_consequence_cell", None) is not None:
        losses["grad_layer_consequence_cell"] = _module_grad_norm(
            planner.layer_consequence_cell, reference=reference
        )
    losses["grad_midcut_heads"] = _module_grad_norm(planner.midcut_heads, reference=reference)
    losses["grad_controlled_dynamics"] = _module_grad_norm(
        planner.controlled_dynamics, reference=reference
    )
    if getattr(planner, "flow_dino_evidence", None) is not None:
        flow_dino = planner.flow_dino_evidence
        losses["grad_flow_dino_evidence"] = _module_grad_norm(flow_dino, reference=reference)
        if bool(getattr(flow_dino, "raw_enabled", False)):
            raw_flow = flow_dino.raw_flow
            if raw_flow is None:
                raise RuntimeError("raw Flow-DINO diagnostics have no raw flow module")
            losses["grad_flow_dino_raw_pyramid"] = _module_grad_norm(
                raw_flow.pyramid, reference=reference
            )
            if flow_dino.early_masked_raw_context is not None:
                losses["grad_flow_dino_early_masked_raw_context"] = _module_grad_norm(
                    flow_dino.early_masked_raw_context, reference=reference
                )
            if flow_dino.soft_address_compiler is not None:
                losses["grad_flow_dino_soft_address_compiler"] = _module_grad_norm(
                    flow_dino.soft_address_compiler, reference=reference
                )
            losses["grad_flow_dino_semantic_coarse_flow"] = _module_grad_norm(
                flow_dino.flow, reference=reference
            )
            losses["grad_flow_dino_raw_mid_flow"] = _module_grad_norm(
                raw_flow.mid, reference=reference
            )
            losses["grad_flow_dino_raw_high_flow"] = _module_grad_norm(
                raw_flow.high, reference=reference
            )
            losses["grad_flow_dino_raw_detail_router"] = _parameter_grad_norm(
                (
                    parameter
                    for module in (flow_dino.raw_detail_query, flow_dino.raw_detail_motion)
                    if module is not None
                    for parameter in module.parameters()
                ),
                reference=reference,
            )
            if flow_dino.raw_address_reader is None:
                raise RuntimeError("raw Flow-DINO diagnostics have no address reader")
            losses["grad_flow_dino_raw_address_reader"] = _module_grad_norm(
                flow_dino.raw_address_reader, reference=reference
            )
            losses["grad_flow_dino_future_predictor"] = _module_grad_norm(
                flow_dino.future_prediction, reference=reference
            )
            if flow_dino.horizon_address_jepa is not None:
                losses["grad_flow_dino_horizon_address"] = _module_grad_norm(
                    flow_dino.horizon_address_jepa,
                    reference=reference,
                )
            progressive = flow_dino.progressive_grounding_address
            if progressive is not None:
                losses["grad_flow_dino_progressive_g1"] = _parameter_grad_norm(
                    (
                        *progressive.query_norms[0].parameters(),
                        *progressive.query_projections[0].parameters(),
                        progressive.slot_identity,
                    ),
                    reference=reference,
                )
                losses["grad_flow_dino_progressive_g2"] = _parameter_grad_norm(
                    (
                        *progressive.query_norms[1].parameters(),
                        *progressive.query_projections[1].parameters(),
                        *progressive.g2_rectifier.parameters(),
                        *(
                            progressive.g2_typed_rectifier.parameters()
                            if progressive.g2_typed_rectifier is not None
                            else ()
                        ),
                        *(
                            progressive.g2_typed_query.parameters()
                            if progressive.g2_typed_query is not None
                            else ()
                        ),
                    ),
                    reference=reference,
                )
                if progressive.g2_typed_query is not None:
                    for owner_name in ("semantic", "appearance", "geometry"):
                        losses[f"grad_flow_dino_progressive_g2_{owner_name}_owner"] = (
                            _module_grad_norm(
                                progressive.g2_typed_query[owner_name],
                                reference=reference,
                            )
                        )
                losses["grad_flow_dino_progressive_g3"] = _parameter_grad_norm(
                    (
                        *progressive.query_norms[2].parameters(),
                        *progressive.query_projections[2].parameters(),
                        *progressive.g3_slot_score.parameters(),
                        *progressive.g3_summary_out.parameters(),
                        *(
                            progressive.g3_typed_slot_score.parameters()
                            if progressive.g3_typed_slot_score is not None
                            else ()
                        ),
                        *(
                            progressive.g3_typed_summary_out.parameters()
                            if progressive.g3_typed_summary_out is not None
                            else ()
                        ),
                        *(
                            progressive.g3_public_summary_out.parameters()
                            if progressive.g3_public_summary_out is not None
                            else ()
                        ),
                        *(
                            progressive.g3_owner_residual.parameters()
                            if progressive.g3_owner_residual is not None
                            else ()
                        ),
                    ),
                    reference=reference,
                )
                if progressive.g3_public_summary_out is not None:
                    losses["grad_flow_dino_progressive_g3_public"] = _module_grad_norm(
                        progressive.g3_public_summary_out,
                        reference=reference,
                    )
                if progressive.g3_owner_residual is not None:
                    for owner_name in ("semantic", "appearance", "geometry"):
                        losses[f"grad_flow_dino_progressive_g3_{owner_name}_owner"] = (
                            _module_grad_norm(
                                progressive.g3_owner_residual[owner_name],
                                reference=reference,
                            )
                        )
                elif progressive.g3_typed_slot_score is not None:
                    for owner_name in ("semantic", "appearance", "geometry"):
                        losses[f"grad_flow_dino_progressive_g3_{owner_name}_owner"] = (
                            _module_grad_norm(
                                progressive.g3_typed_slot_score[owner_name],
                                reference=reference,
                            )
                        )
                losses["grad_flow_dino_progressive_world_query"] = _parameter_grad_norm(
                    (
                        *progressive.horizon_query_norm.parameters(),
                        *progressive.horizon_query_proj.parameters(),
                        *(
                            progressive.world_typed_query.parameters()
                            if progressive.world_typed_query is not None
                            else ()
                        ),
                        *(
                            progressive.world_owner_transitions.parameters()
                            if progressive.world_owner_transitions is not None
                            else ()
                        ),
                        *(
                            progressive.world_owner_writes.parameters()
                            if progressive.world_owner_writes is not None
                            else ()
                        ),
                        *(
                            progressive.world_owner_route_attnres.parameters()
                            if progressive.world_owner_route_attnres is not None
                            else ()
                        ),
                        *(
                            progressive.world_owner_fused_writes.parameters()
                            if progressive.world_owner_fused_writes is not None
                            else ()
                        ),
                        *(
                            progressive.world_horizon_condition.parameters()
                            if progressive.world_horizon_condition is not None
                            else ()
                        ),
                    ),
                    reference=reference,
                )
                if progressive.world_typed_query is not None:
                    for owner_name in ("semantic", "appearance", "geometry"):
                        losses[f"grad_flow_dino_progressive_world_{owner_name}_owner"] = (
                            _parameter_grad_norm(
                                (
                                    *progressive.world_typed_query[owner_name].parameters(),
                                    *(
                                        parameter
                                        for transition_bank in (
                                            progressive.world_owner_transitions
                                            if progressive.world_owner_transitions is not None
                                            else ()
                                        )
                                        for parameter in transition_bank[owner_name].parameters()
                                    ),
                                    *(
                                        progressive.world_owner_writes[owner_name].parameters()
                                        if progressive.world_owner_writes is not None
                                        else ()
                                    ),
                                ),
                                reference=reference,
                            )
                        )
                    if progressive.world_owner_transitions is not None:
                        losses["grad_flow_dino_progressive_world_interval_owner"] = (
                            _parameter_grad_norm(
                                (
                                    *(
                                        parameter
                                        for transition_bank in progressive.world_owner_transitions
                                        for parameter in transition_bank["interval"].parameters()
                                    ),
                                    *(
                                        progressive.world_owner_writes["interval"].parameters()
                                        if progressive.world_owner_writes is not None
                                        else ()
                                    ),
                                ),
                                reference=reference,
                            )
                        )
                if progressive.future_transport is not None:
                    losses["grad_flow_dino_progressive_future_transport"] = _module_grad_norm(
                        progressive.future_transport,
                        reference=reference,
                    )
                if progressive.future_effect_semantic is not None:
                    losses["grad_flow_dino_future_effect_semantic"] = _module_grad_norm(
                        progressive.future_effect_semantic,
                        reference=reference,
                    )
                if progressive.future_effect_geometry is not None:
                    losses["grad_flow_dino_future_effect_geometry"] = _module_grad_norm(
                        progressive.future_effect_geometry,
                        reference=reference,
                    )
                if progressive.window_successor_cell is not None:
                    losses["grad_flow_dino_window_effect_near_mid"] = _module_grad_norm(
                        progressive.window_successor_cell,
                        reference=reference,
                    )
                if progressive.window_late_cell is not None:
                    losses["grad_flow_dino_window_effect_late"] = _module_grad_norm(
                        progressive.window_late_cell,
                        reference=reference,
                    )
                differential_window = getattr(
                    progressive,
                    "differential_window_compiler",
                    None,
                )
                if differential_window is not None:
                    losses[
                        "grad_differential_w1_near_mid_transition"
                    ] = _parameter_grad_norm(
                        (
                            *differential_window.intent_to_route.parameters(),
                            *differential_window.w1_transition.parameters(),
                        ),
                        reference=reference,
                    )
                    losses[
                        "grad_differential_w2_late_transition"
                    ] = _parameter_grad_norm(
                        (
                            differential_window.late_query,
                            differential_window.late_source_type,
                            *differential_window.late_query_norm.parameters(),
                            *differential_window.late_memory_norm.parameters(),
                            *differential_window.late_attention.parameters(),
                            *differential_window.late_ffn.parameters(),
                        ),
                        reference=reference,
                    )
                    losses[
                        "grad_differential_effect_decoder"
                    ] = _parameter_grad_norm(
                        (
                            *differential_window.effect_semantic.parameters(),
                            *differential_window.effect_geometry.parameters(),
                        ),
                        reference=reference,
                    )
                    losses[
                        "grad_differential_current_reference_bridge"
                    ] = _module_grad_norm(
                        differential_window.current_reference,
                        reference=reference,
                    )
                    losses["grad_flow_dino_window_effect_near_mid"] = losses[
                        "grad_differential_w1_near_mid_transition"
                    ]
                    losses["grad_flow_dino_window_effect_late"] = losses[
                        "grad_differential_w2_late_transition"
                    ]
                grounded_world = getattr(
                    progressive,
                    "grounded_world_compiler",
                    None,
                )
                if grounded_world is not None:
                    losses["grad_grounded_world_shared_inputs"] = (
                        _parameter_grad_norm(
                            (
                                *grounded_world.world_input.parameters(),
                                *grounded_world.intent_input.parameters(),
                                *grounded_world.proposal_input.parameters(),
                                *grounded_world.owner_input.parameters(),
                            ),
                            reference=reference,
                        )
                    )
                    losses["grad_grounded_world_w1_blocks"] = _module_grad_norm(
                        grounded_world.w1_blocks,
                        reference=reference,
                    )
                    losses["grad_grounded_world_w2_blocks"] = _module_grad_norm(
                        grounded_world.w2_blocks,
                        reference=reference,
                    )
                    losses["grad_grounded_world_shared_heads"] = (
                        _parameter_grad_norm(
                            (
                                *grounded_world.semantic_head.parameters(),
                                *grounded_world.geometry_head.parameters(),
                                *grounded_world.appearance_head.parameters(),
                                *grounded_world.reliability_head.parameters(),
                                *grounded_world.uncertainty_head.parameters(),
                            ),
                            reference=reference,
                        )
                    )
                    losses["grad_flow_dino_window_effect_near_mid"] = (
                        losses["grad_grounded_world_w1_blocks"]
                    )
                    losses["grad_flow_dino_window_effect_late"] = losses[
                        "grad_grounded_world_w2_blocks"
                    ]
                if progressive.world_owner_route_attnres is not None:
                    losses["grad_flow_dino_functional_world_router"] = _parameter_grad_norm(
                        (
                            *progressive.world_owner_route_attnres.parameters(),
                            *progressive.world_owner_fused_writes.parameters(),
                            *progressive.world_horizon_condition.parameters(),
                        ),
                        reference=reference,
                    )
            if (
                bool(
                    getattr(
                        flow_dino,
                        "functional_mainline_routing_enabled",
                        False,
                    )
                )
                and progressive is not None
                and progressive.world_owner_transitions is not None
                and progressive.world_owner_fused_writes is not None
            ):
                losses["grad_flow_dino_interval_stage"] = _parameter_grad_norm(
                    (
                        *(
                            parameter
                            for transition_bank in (progressive.world_owner_transitions)
                            for parameter in transition_bank["interval"].parameters()
                        ),
                        *progressive.world_owner_fused_writes.parameters(),
                    ),
                    reference=reference,
                )
            elif flow_dino.interval_stage_organizer is not None:
                losses["grad_flow_dino_interval_stage"] = _module_grad_norm(
                    flow_dino.interval_stage_organizer,
                    reference=reference,
                )
        elif bool(getattr(flow_dino, "late_bottleneck", False)):
            losses["grad_flow_dino_coarse_flow"] = _module_grad_norm(
                flow_dino.flow, reference=reference
            )
            losses["grad_flow_dino_sparse_fine"] = _module_grad_norm(
                flow_dino.sparse_fine_flow, reference=reference
            )
            losses["grad_flow_dino_detail_router"] = _module_grad_norm(
                flow_dino.detail_router, reference=reference
            )
            losses["grad_flow_dino_address_reader"] = _module_grad_norm(
                flow_dino.address_reader, reference=reference
            )
            losses["grad_flow_dino_future_predictor"] = _module_grad_norm(
                flow_dino.future_prediction, reference=reference
            )
    if bool(getattr(planner, "object_intent_dynamics_mainline", False)):
        grounder = planner.object_grounder
        intent = planner.object_intent_organizer
        recognizer = planner.object_plan_recognizer
        coarse_action = planner.object_coarse_action
        world = planner.object_future_compiler
        p2 = planner.p2_effect_reader
        consequence = planner.consequence_plan_organizer
        p3 = planner.policy_plan_compiler
        if any(
            module is None
            for module in (
                grounder,
                intent,
                recognizer,
                coarse_action,
                world,
                p2,
                consequence,
                p3,
            )
        ):
            raise RuntimeError("object-intent gradient audit lost a capability owner")
        losses["grad_object_grounder"] = _module_grad_norm(
            grounder, reference=reference
        )
        losses["grad_object_s_goal"] = _parameter_grad_norm(
            (
                intent.goal_queries,
                *intent.goal_input.parameters(),
                *intent.goal_read.parameters(),
                *intent.goal_self.parameters(),
            ),
            reference=reference,
        )
        losses["grad_object_s_history"] = _parameter_grad_norm(
            (
                *intent.history_input.parameters(),
                *intent.history_blocks.parameters(),
            ),
            reference=reference,
        )
        losses["grad_object_s_typed_intervals"] = _parameter_grad_norm(
            (
                intent.interval_identity,
                *intent.object_content.parameters(),
                *intent.object_semantic.parameters(),
                *intent.object_appearance.parameters(),
                *intent.object_geometry.parameters(),
                *intent.interval_goal.parameters(),
                *intent.interval_history.parameters(),
                *intent.interval_typed_router.parameters(),
                *intent.interval_self.parameters(),
                *intent.interval_state_self.parameters(),
                *intent.interval_object_key.parameters(),
                *intent.interval_object_value.parameters(),
            ),
            reference=reference,
        )
        losses["grad_object_s_temporal"] = _parameter_grad_norm(
            (
                intent.temporal_identity,
                *intent.temporal_read.parameters(),
            ),
            reference=reference,
        )
        losses["grad_object_s_state_change"] = _parameter_grad_norm(
            (
                intent.state_change_query,
                *intent.state_change_read.parameters(),
                *intent.state_change_input.parameters(),
                *intent.state_change_transport.parameters(),
                *intent.state_change_fuse.parameters(),
            ),
            reference=reference,
        )
        losses["grad_object_plan_recognizer"] = _module_grad_norm(
            recognizer, reference=reference
        )
        losses["grad_object_coarse_action"] = _module_grad_norm(
            coarse_action, reference=reference
        )
        losses["grad_object_w_inputs"] = _parameter_grad_norm(
            (
                world.interval_identity,
                world.decoder_identity,
                *world.object_content.parameters(),
                *world.intent_action.parameters(),
                *world.intent_state.parameters(),
                *world.intent_object_key.parameters(),
                *world.intent_object_value.parameters(),
                *world.coarse_action.parameters(),
                *world.typed_router.parameters(),
            ),
            reference=reference,
        )
        losses["grad_object_w1"] = _module_grad_norm(world.w1, reference=reference)
        losses["grad_object_w2"] = _parameter_grad_norm(
            (*world.w1_to_w2.parameters(), *world.w2.parameters()),
            reference=reference,
        )
        losses["grad_object_w_heads"] = _parameter_grad_norm(
            (
                *world.near_heads.parameters(),
                *world.far_heads.parameters(),
            ),
            reference=reference,
        )
        losses["grad_object_p2_effect_reader"] = _module_grad_norm(
            p2, reference=reference
        )
        losses["grad_object_consequence"] = _module_grad_norm(
            consequence, reference=reference
        )
        losses["grad_object_p3_precision"] = _parameter_grad_norm(
            (
                *p3.precision_action.parameters(),
                *p3.precision_fact.parameters(),
                *p3.precision_consequence.parameters(),
                *p3.precision_lane.parameters(),
            ),
            reference=reference,
        )
        losses["grad_object_p3_temporal"] = _parameter_grad_norm(
            (
                *p3.temporal_action.parameters(),
                *p3.temporal_consequence.parameters(),
                *p3.temporal_lane.parameters(),
            ),
            reference=reference,
        )
        losses["grad_object_p3_state_change"] = _parameter_grad_norm(
            (
                *p3.state_change_action.parameters(),
                *p3.state_change_temporal.parameters(),
                *p3.state_change_lane.parameters(),
            ),
            reference=reference,
        )
    if getattr(planner, "goal_resampler", None) is not None:
        losses["grad_goal_resampler"] = _module_grad_norm(
            planner.goal_resampler, reference=reference
        )
    if getattr(planner, "stateless_phase_adapter", None) is not None:
        losses["grad_stateless_phase_adapter"] = _module_grad_norm(
            planner.stateless_phase_adapter, reference=reference
        )
        if getattr(planner, "phase_world_query_proj", None) is not None:
            losses["grad_stateless_phase_world_query"] = _module_grad_norm(
                planner.phase_world_query_proj, reference=reference
            )
            losses["grad_condition_world_query"] = _module_grad_norm(
                planner.condition_world_query_proj, reference=reference
            )
        phase_world_blocks = getattr(planner, "phase_world_block_query_proj", None)
        condition_world_blocks = getattr(planner, "condition_world_block_query_proj", None)
        if phase_world_blocks is not None and condition_world_blocks is not None:
            losses["grad_stateless_phase_world_blocks"] = _parameter_grad_norm(
                (
                    parameter
                    for module in (
                        *list(phase_world_blocks),
                        *list(condition_world_blocks),
                    )
                    for parameter in module.parameters()
                ),
                reference=reference,
            )
    if getattr(planner, "stateless_horizon_adapter", None) is not None:
        horizon_adapter = planner.stateless_horizon_adapter
        losses["grad_stateless_horizon_adapter"] = _module_grad_norm(
            horizon_adapter, reference=reference
        )
        losses["grad_stateless_horizon_phase_path"] = _parameter_grad_norm(
            (
                parameter
                for module in (
                    horizon_adapter.anchor_query,
                    horizon_adapter.state_context,
                    horizon_adapter.visual_context,
                    horizon_adapter.phase_query,
                    horizon_adapter.phase_key,
                    horizon_adapter.phase_output,
                )
                for parameter in module.parameters()
            ),
            reference=reference,
        )
        losses["grad_stateless_horizon_goal_path"] = _parameter_grad_norm(
            (
                parameter
                for module in (
                    horizon_adapter.goal_cross,
                    horizon_adapter.goal_output,
                )
                for parameter in module.parameters()
            ),
            reference=reference,
        )
        losses["grad_stateless_horizon_history_path"] = _parameter_grad_norm(
            (
                parameter
                for module in (
                    horizon_adapter.history_cross,
                    horizon_adapter.history_output,
                )
                for parameter in module.parameters()
            ),
            reference=reference,
        )
        horizon_projection_banks = tuple(
            bank
            for bank in (
                getattr(planner, "horizon_phase_world_block_query_proj", None),
                getattr(planner, "horizon_goal_world_block_query_proj", None),
                getattr(planner, "horizon_history_world_block_query_proj", None),
                getattr(planner, "horizon_proposal_world_block_query_proj", None),
            )
            if bank is not None
        )
        if horizon_projection_banks:
            losses["grad_stateless_horizon_world_queries"] = _parameter_grad_norm(
                (parameter for bank in horizon_projection_banks for parameter in bank.parameters()),
                reference=reference,
            )
    if getattr(planner, "stateless_goal_phase_machine", None) is not None:
        goal_phase = planner.stateless_goal_phase_machine
        losses["grad_stateless_goal_phase_machine"] = _module_grad_norm(
            goal_phase, reference=reference
        )
        if isinstance(goal_phase, StatelessIntentOrganizer):
            losses["grad_grounded_intent_goal"] = _parameter_grad_norm(
                (
                    goal_phase.goal_queries,
                    *goal_phase.goal_input.parameters(),
                    *goal_phase.goal_block.parameters(),
                ),
                reference=reference,
            )
            losses["grad_grounded_intent_observable"] = (
                _parameter_grad_norm(
                    (
                        goal_phase.history_type,
                        goal_phase.observable_queries,
                        *goal_phase.state_input.parameters(),
                        *goal_phase.action_input.parameters(),
                        *goal_phase.history_blocks.parameters(),
                        *goal_phase.observable_goal.parameters(),
                        *goal_phase.observable_history.parameters(),
                        *goal_phase.fact_inputs.parameters(),
                        *goal_phase.observable_fact.parameters(),
                        *goal_phase.observable_router.parameters(),
                    ),
                    reference=reference,
                )
            )
            losses["grad_grounded_intent_intervals"] = (
                _parameter_grad_norm(
                    (
                        goal_phase.interval_queries,
                        *goal_phase.interval_goal.parameters(),
                        *goal_phase.interval_observable.parameters(),
                        *goal_phase.interval_history.parameters(),
                        *goal_phase.interval_fact.parameters(),
                        *goal_phase.interval_router.parameters(),
                    ),
                    reference=reference,
                )
            )
            losses["grad_grounded_intent_temporal"] = (
                _parameter_grad_norm(
                    (
                        *goal_phase.temporal_query.parameters(),
                        *goal_phase.temporal_read.parameters(),
                    ),
                    reference=reference,
                )
            )
            losses["grad_grounded_intent_completion"] = (
                _parameter_grad_norm(
                    (
                        *goal_phase.completion.parameters(),
                        *goal_phase.completion_head.parameters(),
                    ),
                    reference=reference,
                )
            )
            proposal_entry = getattr(
                planner,
                "grounded_clean_proposal_proj",
                None,
            )
            if proposal_entry is not None:
                losses["grad_grounded_clean_proposal"] = _module_grad_norm(
                    proposal_entry,
                    reference=reference,
                )
        elif isinstance(goal_phase, DifferentialStatelessIntentController):
            losses["grad_intent_goal_program"] = _parameter_grad_norm(
                (
                    parameter
                    for module in (
                        goal_phase.program_seed,
                        goal_phase.goal_input,
                        goal_phase.goal_block,
                    )
                    for parameter in module.parameters()
                ),
                reference=reference,
            )
            losses["grad_intent_history_encoder"] = _parameter_grad_norm(
                (
                    parameter
                    for module in (
                        goal_phase.state_input,
                        goal_phase.action_input,
                        goal_phase.history_fuse,
                        goal_phase.history_time,
                        goal_phase.history_blocks,
                    )
                    for parameter in module.parameters()
                ),
                reference=reference,
            )
            losses["grad_intent_history_write"] = _module_grad_norm(
                goal_phase.history_to_program,
                reference=reference,
            )
            losses["grad_intent_grounding_write"] = _parameter_grad_norm(
                (
                    parameter
                    for module in (
                        goal_phase.grounding_input,
                        goal_phase.grounding_to_program,
                    )
                    for parameter in module.parameters()
                ),
                reference=reference,
            )
            losses["grad_intent_ordered_refinement"] = _module_grad_norm(
                goal_phase.ordered_refinement,
                reference=reference,
            )
            losses["grad_intent_window_read"] = _parameter_grad_norm(
                (
                    goal_phase.window_query,
                    *goal_phase.window_coordinate_key.parameters(),
                    *goal_phase.window_read.parameters(),
                    *goal_phase.window_refinement.parameters(),
                ),
                reference=reference,
            )
            losses["grad_intent_predictive_effect"] = _module_grad_norm(
                goal_phase.predictive_effect,
                reference=reference,
            )
            losses["grad_intent_terminal"] = _module_grad_norm(
                goal_phase.terminal_head,
                reference=reference,
            )
            proposal_world_queries = getattr(
                planner,
                "horizon_proposal_world_block_query_proj",
                None,
            )
            if proposal_world_queries is not None:
                losses[
                    "grad_differential_clean_proposal_world_condition"
                ] = _module_grad_norm(
                    proposal_world_queries,
                    reference=reference,
                )
            canonical_g_to_p = getattr(
                planner,
                "phase_world_query_proj",
                None,
            )
            if canonical_g_to_p is not None:
                losses["grad_intent_canonical_g_to_p_query"] = (
                    _module_grad_norm(
                        canonical_g_to_p,
                        reference=reference,
                    )
                )
            late_reader = getattr(planner, "late_raw_detail_reader", None)
            canonical_p1 = (
                getattr(late_reader, "phase_query_proj", None)
                if late_reader is not None
                else None
            )
            if canonical_p1 is not None:
                losses["grad_intent_canonical_p1_query"] = _module_grad_norm(
                    canonical_p1,
                    reference=reference,
                )
            # Compatibility aliases preserve historical parsers while the
            # explicit names above expose the actual five-block ownership.
            losses["grad_stateless_intent_s1"] = losses[
                "grad_intent_goal_program"
            ]
            losses["grad_stateless_intent_s2"] = _parameter_grad_norm(
                (
                    *goal_phase.history_blocks.parameters(),
                    *goal_phase.history_to_program.parameters(),
                ),
                reference=reference,
            )
            losses["grad_stateless_intent_s3"] = _parameter_grad_norm(
                (
                    *goal_phase.grounding_to_program.parameters(),
                    *goal_phase.ordered_refinement.parameters(),
                    goal_phase.window_query,
                    *goal_phase.window_read.parameters(),
                ),
                reference=reference,
            )
            losses["grad_stateless_intent_mlp"] = _parameter_grad_norm(
                (
                    *goal_phase.window_refinement.parameters(),
                    *goal_phase.predictive_effect.parameters(),
                    *goal_phase.terminal_head.parameters(),
                ),
                reference=reference,
            )
            losses["grad_goal_phase_program"] = losses[
                "grad_intent_goal_program"
            ]
            losses["grad_goal_phase_transition"] = losses[
                "grad_intent_ordered_refinement"
            ]
            losses["grad_goal_phase_observation"] = losses[
                "grad_intent_grounding_write"
            ]
        elif hasattr(goal_phase, "history_block"):
            losses["grad_stateless_intent_s1"] = _parameter_grad_norm(
                (
                    parameter
                    for module in (
                        goal_phase.goal_input,
                        goal_phase.program_query,
                        goal_phase.goal_cross,
                        goal_phase.program_ffn,
                    )
                    for parameter in module.parameters()
                ),
                reference=reference,
            )
            losses["grad_stateless_intent_s2"] = _parameter_grad_norm(
                (
                    parameter
                    for module in (
                        goal_phase.state_input,
                        goal_phase.action_input,
                        goal_phase.grounding_input,
                        goal_phase.history_fuse,
                        goal_phase.grounding_cross,
                        goal_phase.history_block,
                    )
                    for parameter in module.parameters()
                ),
                reference=reference,
            )
            losses["grad_stateless_intent_s3"] = _parameter_grad_norm(
                (
                    parameter
                    for module in (
                        goal_phase.control_cross,
                        goal_phase.control_query,
                        goal_phase.program_key,
                        goal_phase.observable_role,
                    )
                    for parameter in module.parameters()
                ),
                reference=reference,
            )
            losses["grad_stateless_intent_mlp"] = _parameter_grad_norm(
                (
                    parameter
                    for module in (
                        goal_phase.progress_head,
                        goal_phase.window_score,
                        goal_phase.completion_head,
                        goal_phase.intent_output,
                        goal_phase.phase_output,
                        goal_phase.goal_output,
                        goal_phase.history_output,
                    )
                    for parameter in module.parameters()
                ),
                reference=reference,
            )
            # Compatibility aliases retain parsers without pretending that S
            # still owns a recurrent transition matrix.
            losses["grad_goal_phase_program"] = losses[
                "grad_stateless_intent_s1"
            ]
            losses["grad_goal_phase_transition"] = losses[
                "grad_stateless_intent_s3"
            ]
            losses["grad_goal_phase_observation"] = losses[
                "grad_stateless_intent_s2"
            ]
        else:
            losses["grad_goal_phase_program"] = _parameter_grad_norm(
                (
                    parameter
                    for module in (
                        goal_phase.goal_cross,
                        goal_phase.program_ffn,
                        goal_phase.program_query,
                    )
                    for parameter in module.parameters()
                ),
                reference=reference,
            )
            losses["grad_goal_phase_transition"] = _parameter_grad_norm(
                goal_phase.transition.parameters(),
                reference=reference,
            )
            losses["grad_goal_phase_observation"] = _parameter_grad_norm(
                (
                    parameter
                    for module in (
                        goal_phase.observation_encoder,
                        goal_phase.grounding_cross,
                        goal_phase.state_input,
                        goal_phase.action_input,
                        goal_phase.grounding_input,
                    )
                    for parameter in module.parameters()
                ),
                reference=reference,
            )
        typed_world_context_parameters: list[torch.nn.Parameter] = []
        for bank in (
            getattr(
                planner,
                "horizon_phase_world_block_query_proj",
                None,
            ),
            getattr(
                planner,
                "horizon_goal_world_block_query_proj",
                None,
            ),
            getattr(
                planner,
                "horizon_history_world_block_query_proj",
                None,
            ),
            getattr(planner, "horizon_typed_context_router", None),
        ):
            if bank is not None:
                typed_world_context_parameters.extend(bank.parameters())
        typed_query = getattr(planner, "horizon_typed_context_query", None)
        if isinstance(typed_query, torch.nn.Parameter):
            typed_world_context_parameters.append(typed_query)
        if typed_world_context_parameters:
            losses["grad_goal_phase_typed_world_context"] = _parameter_grad_norm(
                typed_world_context_parameters,
                reference=reference,
            )
    if getattr(planner, "ground_to_world_attnres", None) is not None:
        losses["grad_attnres_ground_to_world"] = _module_grad_norm(
            planner.ground_to_world_attnres, reference=reference
        )
    if getattr(planner, "world_to_policy_attnres", None) is not None:
        losses["grad_attnres_world_to_policy"] = _module_grad_norm(
            planner.world_to_policy_attnres, reference=reference
        )
    if bool(getattr(system.proposal, "history_enabled", False)):
        losses["grad_action_history_encoder"] = _parameter_grad_norm(
            (
                parameter
                for name, parameter in system.proposal.named_parameters()
                if name.startswith("history_")
            ),
            reference=reference,
        )
    final_modules = [
        planner.final_norm,
        planner.direct_physical_head,
        planner.rollout_residual_head,
        planner.controlled_dynamics,
        planner.event_probe,
        planner.motion_probe,
    ]
    if getattr(planner, "late_raw_detail_reader", None) is not None:
        losses["grad_late_raw_detail_reader"] = _module_grad_norm(
            planner.late_raw_detail_reader, reference=reference
        )
        typed_refiners = getattr(planner.late_raw_detail_reader, "typed_local_refiners", None)
        if typed_refiners is not None:
            reader = planner.late_raw_detail_reader
            p1_modules = tuple(
                module
                for module in (
                    getattr(reader, "lattice_query_proj", None),
                    getattr(reader, "lattice_world_key_proj", None),
                    getattr(reader, "typed_fine_query", None),
                    getattr(reader, "typed_coarse_query", None),
                    getattr(reader, "appearance_world_owner_query", None),
                )
                if module is not None
            )
            losses["grad_late_raw_detail_typed_p1_selector"] = _parameter_grad_norm(
                (parameter for module in p1_modules for parameter in module.parameters()),
                reference=reference,
            )
            appearance_gateway = getattr(reader, "appearance_world_owner_query", None)
            if appearance_gateway is not None:
                losses["grad_late_raw_detail_p1_appearance_gateway"] = _module_grad_norm(
                    appearance_gateway,
                    reference=reference,
                )
            losses["grad_late_raw_detail_literal_rgb_value"] = _parameter_grad_norm(
                (
                    parameter
                    for refiner in typed_refiners
                    for parameter in refiner.rgb_value.parameters()
                ),
                reference=reference,
            )
            losses["grad_late_raw_detail_learned_detail_value"] = _parameter_grad_norm(
                (
                    parameter
                    for refiner in typed_refiners
                    for parameter in refiner.detail_value.parameters()
                ),
                reference=reference,
            )
            losses["grad_late_raw_detail_typed_p2_condition"] = _parameter_grad_norm(
                (
                    parameter
                    for refiner in typed_refiners
                    for module in (
                        (
                            refiner.coordinate_key,
                            refiner.geometry_key,
                            refiner.owner_conditions,
                        )
                        if hasattr(refiner, "owner_conditions")
                        else (
                            refiner.coordinate_key,
                            refiner.query_condition,
                            refiner.semantic_condition,
                            refiner.appearance_condition,
                            refiner.geometry_condition,
                            refiner.future_condition,
                        )
                    )
                    for parameter in module.parameters()
                ),
                reference=reference,
            )
            if all(hasattr(refiner, "owner_conditions") for refiner in typed_refiners):
                for owner_name in (
                    "policy",
                    "semantic",
                    "appearance",
                    "geometry",
                    "horizon",
                ):
                    losses[f"grad_late_raw_detail_typed_p2_{owner_name}_owner"] = (
                        _parameter_grad_norm(
                            (
                                parameter
                                for refiner in typed_refiners
                                for module in (
                                    refiner.owner_conditions[owner_name],
                                    refiner.owner_outputs[owner_name],
                                )
                                for parameter in module.parameters()
                            ),
                            reference=reference,
                        )
                    )
                losses["grad_late_raw_detail_typed_p2_router"] = _parameter_grad_norm(
                    (
                        parameter
                        for refiner in typed_refiners
                        for parameter in refiner.delta_router.parameters()
                    ),
                    reference=reference,
                )
            losses["grad_late_raw_detail_typed_p2"] = _module_grad_norm(
                typed_refiners, reference=reference
            )
        final_modules.append(planner.late_raw_detail_reader)
    if getattr(planner, "policy_plan_compiler", None) is not None:
        compiler = planner.policy_plan_compiler
        losses["grad_policy_plan_compiler"] = _module_grad_norm(compiler, reference=reference)
        for lane in ("precision", "effect", "temporal", "terminal"):
            module = getattr(compiler, f"{lane}_lane", None)
            if module is not None:
                losses[f"grad_policy_plan_{lane}"] = _module_grad_norm(
                    module,
                    reference=reference,
                )
        final_modules.append(compiler)
    if getattr(planner, "p2_effect_reader", None) is not None:
        losses["grad_p2_structured_effect_reader"] = _module_grad_norm(
            planner.p2_effect_reader,
            reference=reference,
        )
        final_modules.append(planner.p2_effect_reader)
    if getattr(planner, "consequence_plan_organizer", None) is not None:
        losses["grad_consequence_plan_organizer"] = _module_grad_norm(
            planner.consequence_plan_organizer,
            reference=reference,
        )
        final_modules.append(planner.consequence_plan_organizer)
    if getattr(planner, "residual_action_flow_denoiser", None) is not None:
        losses["grad_residual_action_flow"] = _module_grad_norm(
            planner.residual_action_flow_denoiser, reference=reference
        )
    if getattr(planner, "latent_main_action_decoder", None) is not None:
        losses["grad_latent_main_action"] = _module_grad_norm(
            planner.latent_main_action_decoder, reference=reference
        )
    if getattr(planner, "hierarchical_mmdit_action_decoder", None) is not None:
        decoder = planner.hierarchical_mmdit_action_decoder
        losses["grad_hierarchical_mmdit_action"] = _module_grad_norm(decoder, reference=reference)
        losses["grad_hierarchical_mmdit_velocity_head"] = _module_grad_norm(
            decoder.velocity_head,
            reference=reference,
        )
        losses["grad_intent_contract_compiler"] = _module_grad_norm(
            decoder.intent_compiler,
            reference=reference,
        )
        losses["grad_condition_organizer"] = _module_grad_norm(
            decoder.organizer,
            reference=reference,
        )
        losses["grad_owned_workspace"] = _module_grad_norm(
            decoder.workspace,
            reference=reference,
        )
        blocks = decoder.blocks
        losses["grad_hierarchical_mmdit_blocks"] = _module_grad_norm(
            blocks,
            reference=reference,
        )
        shared_base_modules = torch.nn.ModuleList(
            [
                module
                for block in blocks
                for module in (
                    block.self_qkv,
                    block.self_out,
                    block.cross_q,
                    block.noisy_kv,
                    block.stage_kv,
                    block.low_kv,
                    block.noisy_out,
                    block.stage_out,
                    block.low_out,
                    block.ffn,
                )
            ]
        )
        losses["grad_hierarchical_mmdit_shared_base"] = _module_grad_norm(
            shared_base_modules,
            reference=reference,
        )
        losses["grad_hierarchical_mmdit_distinct_base"] = losses[
            "grad_hierarchical_mmdit_shared_base"
        ]
        base_projection_modules = torch.nn.ModuleList(
            [
                module
                for block in blocks
                for module in (
                    block.self_out,
                    block.noisy_out,
                    block.stage_out,
                    block.low_out,
                    block.ffn.net[2],
                )
            ]
        )
        losses["grad_hierarchical_mmdit_base_projection"] = _module_grad_norm(
            base_projection_modules,
            reference=reference,
        )
        losses["grad_hierarchical_mmdit_contractions"] = _module_grad_norm(
            decoder.operator_contractions,
            reference=reference,
        )
        losses["grad_hierarchical_mmdit_contraction_basis"] = _parameter_grad_norm(
            decoder.factor_parameters(),
            reference=reference,
        )
        losses["grad_hierarchical_mmdit_contraction_depth"] = _parameter_grad_norm(
            (
                parameter
                for bank in decoder.operator_contractions
                for contraction in bank.values()
                for parameter in (contraction.depth_weight, contraction.depth_bias)
            ),
            reference=reference,
        )
        losses["grad_hierarchical_mmdit_stage_selector"] = _parameter_grad_norm(
            decoder.stage_selector_parameters(),
            reference=reference,
        )
        losses["grad_hierarchical_mmdit_exit_controller"] = _parameter_grad_norm(
            decoder.exit_controller_parameters(),
            reference=reference,
        )
        losses["grad_hierarchical_mmdit_unified_controller"] = _parameter_grad_norm(
            decoder.unified_controller_parameters(),
            reference=reference,
        )
        for group_name, parameters in decoder.unified_controller_parameter_groups().items():
            losses[f"grad_hierarchical_mmdit_controller_{group_name}"] = _parameter_grad_norm(
                parameters, reference=reference
            )
        losses["grad_hierarchical_mmdit_content_modulation"] = _linear_row_grad_norm(
            (block.mod for block in blocks),
            start=0,
            stop=2 * int(decoder.hidden_size),
            reference=reference,
        )
        losses["grad_hierarchical_mmdit_host_gates"] = _linear_row_grad_norm(
            (block.mod for block in blocks),
            start=2 * int(decoder.hidden_size),
            stop=None,
            reference=reference,
        )
    if getattr(planner, "latent_cvae_action_decoder", None) is not None:
        decoder = planner.latent_cvae_action_decoder
        losses["grad_latent_cvae_action"] = _module_grad_norm(decoder, reference=reference)
        hierarchical_workspace = getattr(decoder, "hierarchical_workspace", None)
        legacy_workspace = getattr(decoder, "evidence_workspace", None)
        workspace = (
            hierarchical_workspace if hierarchical_workspace is not None else legacy_workspace
        )
        if workspace is not None:
            losses["grad_latent_cvae_workspace"] = _module_grad_norm(workspace, reference=reference)
            if hierarchical_workspace is not None:
                losses["grad_latent_cvae_hierarchical_workspace"] = _module_grad_norm(
                    hierarchical_workspace,
                    reference=reference,
                )
                losses["grad_latent_cvae_hierarchical_manager"] = _module_grad_norm(
                    hierarchical_workspace.manager,
                    reference=reference,
                )
                low_modules = torch.nn.ModuleList(
                    [
                        hierarchical_workspace.condition_query,
                        hierarchical_workspace.low_stage_query,
                        hierarchical_workspace.low_stage_role_key,
                        hierarchical_workspace.low_stage_content_key,
                        hierarchical_workspace.low_stage_role_value,
                        hierarchical_workspace.low_stage_content_value,
                        hierarchical_workspace.low_stage_out,
                        hierarchical_workspace.low_blocks,
                        hierarchical_workspace.low_final_norm,
                    ]
                )
                losses["grad_latent_cvae_hierarchical_low"] = _module_grad_norm(
                    low_modules,
                    reference=reference,
                )
                stage_modules = torch.nn.ModuleList(
                    [
                        hierarchical_workspace.stage_init,
                        hierarchical_workspace.stage_role_query,
                        hierarchical_workspace.stage_content_query,
                        hierarchical_workspace.stage_condition_query,
                        hierarchical_workspace.stage_low_key,
                        hierarchical_workspace.stage_low_value,
                        hierarchical_workspace.stage_promote_out,
                        hierarchical_workspace.stage_gru,
                        hierarchical_workspace.stage_content_out,
                        hierarchical_workspace.stage_role_out,
                    ]
                )
                losses["grad_latent_cvae_hierarchical_stage"] = _module_grad_norm(
                    stage_modules,
                    reference=reference,
                )
            primary_modules: list[torch.nn.Module] = []
            primary_modules.extend(
                block.mod for block in getattr(decoder, "blocks", []) if hasattr(block, "mod")
            )
            primary_modules.extend(
                block.action_mod
                for block in getattr(decoder, "mmdit_blocks", [])
                if hasattr(block, "action_mod")
            )
            primary_modules.extend(
                block.mod
                for block in (
                    getattr(workspace, "low_blocks", [])
                    if hierarchical_workspace is not None
                    else getattr(workspace, "blocks", [])
                )
                if hasattr(block, "mod")
            )
            if primary_modules:
                losses["grad_latent_cvae_primary_modulation"] = _module_grad_norm(
                    torch.nn.ModuleList(primary_modules),
                    reference=reference,
                )
        else:
            rollout_projection = getattr(decoder, "mmdit_rollout_cond_proj", None)
            if rollout_projection is not None:
                losses["grad_latent_cvae_rollout_condition"] = _module_grad_norm(
                    rollout_projection, reference=reference
                )
    if getattr(planner, "evidence_latent_mmdit_action_decoder", None) is not None:
        decoder = planner.evidence_latent_mmdit_action_decoder
        losses["grad_evidence_latent_mmdit_action"] = _module_grad_norm(
            decoder, reference=reference
        )
        losses["grad_evidence_view_adapter"] = _module_grad_norm(
            decoder.evidence_adapter,
            reference=reference,
        )
        losses["grad_evidence_condition_organizer"] = _module_grad_norm(
            decoder.organizer,
            reference=reference,
        )
        losses["grad_evidence_mmdit_blocks"] = _module_grad_norm(
            decoder.blocks,
            reference=reference,
        )
        evidence_reader_parameters = [
            parameter
            for block in decoder.blocks
            for parameter in block.evidence_reader_parameters()
        ]
        losses["grad_evidence_mmdit_evidence_reader"] = _parameter_grad_norm(
            evidence_reader_parameters,
            reference=reference,
        )
        losses["grad_evidence_mmdit_action_state"] = _module_grad_norm(
            decoder.noisy_lift,
            reference=reference,
        )
        if decoder.top_policy_workspace_lift is not None:
            losses["grad_evidence_top_policy_workspace_lift"] = _module_grad_norm(
                decoder.top_policy_workspace_lift,
                reference=reference,
            )
        if decoder.policy_delta_attnres is not None:
            losses["grad_attnres_policy_to_mmdit"] = _module_grad_norm(
                decoder.policy_delta_attnres,
                reference=reference,
            )
        if decoder.protected_detail_basis_attnres is not None:
            losses["grad_protected_detail_basis_reader"] = _module_grad_norm(
                decoder.protected_detail_basis_attnres,
                reference=reference,
            )
        if decoder.execution_controller is not None:
            controller = decoder.execution_controller
            losses["grad_evidence_mmdit_execution_controller"] = _module_grad_norm(
                controller,
                reference=reference,
            )
            losses["grad_evidence_mmdit_capacity_control"] = _module_grad_norm(
                controller.capacity_head,
                reference=reference,
            )
            losses["grad_evidence_mmdit_execution_value_reader"] = _module_grad_norm(
                controller.value_reader,
                reference=reference,
            )
        if decoder.operator_contractions:
            losses["grad_evidence_mmdit_operator_capacity"] = _module_grad_norm(
                decoder.operator_contractions,
                reference=reference,
            )
            factor_parameters = [
                parameter
                for contraction in decoder.operator_contractions
                for parameter in contraction.factor_parameters()
            ]
            losses["grad_evidence_mmdit_operator_basis"] = _parameter_grad_norm(
                factor_parameters,
                reference=reference,
            )
        for index, block in enumerate(decoder.blocks):
            losses[f"grad_evidence_mmdit_block_{index}"] = _module_grad_norm(
                block,
                reference=reference,
            )
    final = torch.nn.ModuleList(final_modules)
    losses["grad_final_policy_heads"] = _module_grad_norm(final, reference=reference)


def _unique_params(
    sources: Sequence[torch.nn.Module | torch.nn.Parameter],
) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for source in sources:
        candidates = (source,) if isinstance(source, torch.nn.Parameter) else source.parameters()
        for param in candidates:
            ident = id(param)
            if param.requires_grad and ident not in seen:
                seen.add(ident)
                params.append(param)
    return params


def _optimizer_groups(
    system: V39PolicySystem, trainer: V39PolicyTrainerConfig
) -> list[dict[str, Any]]:
    stage = str(getattr(trainer, "training_stage", "contract")).lower().replace("-", "_")
    if stage not in {"contract", "stage1", "policy", "stage2"}:
        raise ValueError("training_stage must be contract/stage1 or policy/stage2")
    if bool(int(getattr(trainer, "single_stage_role_lr", 0))) and (
        stage not in {"policy", "stage2"}
        or not int(getattr(system.policy_config, "flow_jepa_role_hierarchy", 0))
    ):
        raise ValueError(
            "single_stage_role_lr requires policy/stage2 and the Flow-JEPA role hierarchy"
        )
    cut = int(system.policy_config.midcut_layer)
    planner = system.planner
    phase_world_block_modules = [
        *list(getattr(planner, "phase_world_block_query_proj", None) or ()),
        *list(getattr(planner, "condition_world_block_query_proj", None) or ()),
    ]
    horizon_world_block_modules = [
        *list(getattr(planner, "horizon_phase_world_block_query_proj", None) or ()),
        *list(getattr(planner, "horizon_goal_world_block_query_proj", None) or ()),
        *list(getattr(planner, "horizon_history_world_block_query_proj", None) or ()),
        *list(getattr(planner, "horizon_proposal_world_block_query_proj", None) or ()),
    ]
    typed_horizon_router = getattr(planner, "horizon_typed_context_router", None)
    if typed_horizon_router is not None:
        horizon_world_block_modules.extend(list(typed_horizon_router))
    typed_horizon_query = getattr(planner, "horizon_typed_context_query", None)
    if isinstance(typed_horizon_query, torch.nn.Parameter):
        horizon_world_block_modules.append(typed_horizon_query)
    groups: list[dict[str, Any]] = []
    object_top_modules = [
        module
        for module in (
            getattr(planner, "object_grounder", None),
            getattr(planner, "object_intent_organizer", None),
            getattr(planner, "object_plan_recognizer", None),
            getattr(planner, "object_coarse_action", None),
            getattr(planner, "object_future_compiler", None),
        )
        if module is not None
    ]

    def add_object_top_group() -> None:
        params = _unique_params(object_top_modules)
        if params:
            groups.append(
                {
                    "params": params,
                    # G/S/W are the new mainline, not inherited low-LR probes.
                    # P2/P3 retain their existing final-policy ownership.
                    "lr": trainer.lr,
                    "name": "object_intent_dynamics_323_top",
                }
            )
    complete_latent_decoder = (
        getattr(planner, "latent_cvae_action_decoder", None) is not None
        or getattr(planner, "latent_main_action_decoder", None) is not None
        or getattr(planner, "hierarchical_mmdit_action_decoder", None) is not None
        or getattr(planner, "evidence_latent_mmdit_action_decoder", None) is not None
    )
    legacy_action_readers = (
        []
        if complete_latent_decoder
        else [planner.direct_physical_head, planner.rollout_residual_head]
    )
    legacy_motion_readers = [] if complete_latent_decoder else [planner.motion_probe]

    def add_hierarchical_decoder_groups(*, lr: float, name: str) -> None:
        decoder = getattr(planner, "hierarchical_mmdit_action_decoder", None)
        if decoder is None:
            return
        factor_params = list(decoder.factor_parameters())
        contraction_control_params = list(decoder.contraction_control_parameters())
        controller_params = list(decoder.unified_controller_parameters())
        base_scale_params = list(decoder.scale_invariant_base_parameters())
        factor_ids = {id(parameter) for parameter in factor_params}
        contraction_control_ids = {id(parameter) for parameter in contraction_control_params}
        controller_ids = {id(parameter) for parameter in controller_params}
        base_scale_ids = {id(parameter) for parameter in base_scale_params}
        owner_sets = (
            factor_ids,
            contraction_control_ids,
            controller_ids,
            base_scale_ids,
        )
        if any(
            left & right
            for index, left in enumerate(owner_sets)
            for right in owner_sets[index + 1 :]
        ):
            raise RuntimeError("hierarchical MMDiT optimizer parameter owners overlap")
        special_ids = set().union(*owner_sets)
        regular_params = [
            parameter
            for parameter in decoder.parameters()
            if parameter.requires_grad and id(parameter) not in special_ids
        ]
        if regular_params:
            groups.append({"params": regular_params, "lr": lr, "name": name})
        if factor_params:
            groups.append(
                {
                    "params": factor_params,
                    "lr": lr * float(trainer.hierarchical_mmdit_contraction_lr_scale),
                    "weight_decay": 0.0,
                    "name": f"{name}_contraction_basis_no_decay",
                }
            )
        if contraction_control_params:
            groups.append(
                {
                    "params": contraction_control_params,
                    "lr": lr * float(trainer.hierarchical_mmdit_contraction_lr_scale),
                    "weight_decay": 0.0,
                    "name": f"{name}_contraction_depth_no_decay",
                }
            )
        if controller_params:
            groups.append(
                {
                    "params": controller_params,
                    "lr": lr * float(trainer.hierarchical_mmdit_controller_lr_scale),
                    "name": f"{name}_unified_controller",
                }
            )
        if base_scale_params:
            groups.append(
                {
                    "params": base_scale_params,
                    "lr": lr * float(trainer.hierarchical_mmdit_shared_base_lr_scale),
                    "weight_decay": 0.0,
                    "name": f"{name}_scale_invariant_base_no_decay",
                }
            )

    def add_evidence_decoder_groups(*, lr: float, name: str) -> None:
        decoder = getattr(planner, "evidence_latent_mmdit_action_decoder", None)
        if decoder is None:
            return
        factor_params = [
            parameter
            for contraction in decoder.operator_contractions
            for parameter in contraction.factor_parameters()
            if parameter.requires_grad
        ]
        depth_params = [
            parameter
            for contraction in decoder.operator_contractions
            for parameter in contraction.control_parameters()
            if parameter.requires_grad
        ]
        controller_params = (
            []
            if decoder.execution_controller is None
            else [
                parameter
                for parameter in decoder.execution_controller.parameters()
                if parameter.requires_grad
            ]
        )
        factor_ids = {id(parameter) for parameter in factor_params}
        depth_ids = {id(parameter) for parameter in depth_params}
        controller_ids = {id(parameter) for parameter in controller_params}
        if factor_ids & depth_ids or factor_ids & controller_ids or depth_ids & controller_ids:
            raise RuntimeError("native evidence MMDiT optimizer parameter owners overlap")
        special_ids = factor_ids | depth_ids | controller_ids
        regular_params = [
            parameter
            for parameter in decoder.parameters()
            if parameter.requires_grad and id(parameter) not in special_ids
        ]
        if regular_params:
            groups.append({"params": regular_params, "lr": lr, "name": name})
        if factor_params:
            groups.append(
                {
                    "params": factor_params,
                    "lr": lr * float(trainer.hierarchical_mmdit_contraction_lr_scale),
                    "weight_decay": 0.0,
                    "name": f"{name}_operator_basis_no_decay",
                }
            )
        if depth_params:
            groups.append(
                {
                    "params": depth_params,
                    "lr": lr * float(trainer.hierarchical_mmdit_contraction_lr_scale),
                    "weight_decay": 0.0,
                    "name": f"{name}_operator_depth_no_decay",
                }
            )
        if controller_params:
            groups.append(
                {
                    "params": controller_params,
                    "lr": lr * float(trainer.hierarchical_mmdit_controller_lr_scale),
                    "name": f"{name}_execution_controller",
                }
            )

    if stage in {"contract", "stage1"} and int(
        getattr(system.policy_config, "flow_jepa_enabled", 0)
    ):
        # V95 Stage1 has its own representation graph.  Its optimizer must not
        # inherit the legacy layer-contract ownership map: those heads are not
        # executed, while final_norm is on the actual JEPA prediction path.
        shared_modules = [
            planner.seed,
            planner.time,
            planner.content_mod,
            planner.content_mod_scale,
            planner.final_norm,
        ]
        if getattr(planner, "goal_resampler", None) is not None:
            shared_modules.append(planner.goal_resampler)
        if getattr(planner, "stateless_phase_adapter", None) is not None:
            shared_modules.append(planner.stateless_phase_adapter)
            if getattr(planner, "phase_world_query_proj", None) is not None:
                shared_modules.extend(
                    [
                        planner.phase_world_query_proj,
                        planner.condition_world_query_proj,
                    ]
                )
            shared_modules.extend(phase_world_block_modules)
        if getattr(planner, "stateless_horizon_adapter", None) is not None:
            shared_modules.extend(
                module
                for module in [
                    planner.stateless_horizon_adapter,
                    planner.phase_world_query_proj,
                    planner.condition_world_query_proj,
                    planner.history_world_query_proj,
                    *horizon_world_block_modules,
                ]
                if module is not None
            )
        if getattr(planner, "stateless_goal_phase_machine", None) is not None:
            shared_modules.extend(
                module
                for module in [
                    planner.stateless_goal_phase_machine,
                    planner.phase_world_query_proj,
                    planner.condition_world_query_proj,
                    planner.history_world_query_proj,
                    *horizon_world_block_modules,
                ]
                if module is not None
            )
        if getattr(planner, "grounded_clean_proposal_proj", None) is not None:
            shared_modules.append(planner.grounded_clean_proposal_proj)
        for bridge_name in (
            "ground_to_world_attnres",
            "world_to_policy_attnres",
        ):
            bridge = getattr(planner, bridge_name, None)
            if bridge is not None:
                shared_modules.append(bridge)
        groups.append(
            {
                "params": _unique_params(shared_modules),
                "lr": trainer.lr * 0.5,
                "name": "flow_jepa_stage1_inputs_and_final_norm",
            }
        )
        depth = len(planner.blocks)
        min_scale = float(getattr(trainer, "layerwise_lr_min_scale", 0.30))
        min_scale = min(max(min_scale, 0.01), 1.0)
        for index, block in enumerate(planner.blocks):
            fraction = 0.0 if depth <= 1 else float(index) / float(depth - 1)
            scale = min_scale + (1.0 - min_scale) * fraction
            groups.append(
                {
                    "params": _unique_params([block]),
                    "lr": trainer.lr * scale,
                    "name": f"flow_jepa_stage1_dit_block_{index}_lr{scale:.2f}",
                }
            )
        history_parameters = [
            parameter
            for name, parameter in system.proposal.named_parameters()
            if name.startswith("history_") and parameter.requires_grad
        ]
        if history_parameters:
            groups.append(
                {
                    "params": history_parameters,
                    "lr": trainer.proposal_lr,
                    "name": "flow_jepa_stage1_action_history",
                }
            )
        flow_dino = getattr(planner, "flow_dino_evidence", None)
        if flow_dino is None:
            raise RuntimeError("V95 Stage1 optimizer has no Flow-DINO evidence owner")
        groups.append(
            {
                "params": [
                    parameter for parameter in flow_dino.parameters() if parameter.requires_grad
                ],
                "lr": trainer.lr * float(getattr(trainer, "flow_jepa_lr_scale", 1.0)),
                "name": "flow_jepa_stage1_evidence",
            }
        )
        add_object_top_group()
        return [group for group in groups if len(group["params"]) > 0]

    if (
        _uses_layer_adapter_contract(trainer)
        and len(getattr(planner, "layer_contract_heads", [])) > 0
    ):
        shared_modules = [
            planner.visual_memory,
            planner.rollout_codec,
            planner.seed,
            planner.time,
            planner.content_mod,
            planner.content_mod_scale,
        ]
        if getattr(planner, "goal_resampler", None) is not None:
            shared_modules.append(planner.goal_resampler)
        if getattr(planner, "stateless_phase_adapter", None) is not None:
            shared_modules.append(planner.stateless_phase_adapter)
            if getattr(planner, "phase_world_query_proj", None) is not None:
                shared_modules.extend(
                    [
                        planner.phase_world_query_proj,
                        planner.condition_world_query_proj,
                    ]
                )
            shared_modules.extend(phase_world_block_modules)
        if getattr(planner, "stateless_horizon_adapter", None) is not None:
            shared_modules.extend(
                module
                for module in [
                    planner.stateless_horizon_adapter,
                    planner.phase_world_query_proj,
                    planner.condition_world_query_proj,
                    planner.history_world_query_proj,
                    *horizon_world_block_modules,
                ]
                if module is not None
            )
        if getattr(planner, "stateless_goal_phase_machine", None) is not None:
            shared_modules.extend(
                module
                for module in [
                    planner.stateless_goal_phase_machine,
                    planner.phase_world_query_proj,
                    planner.condition_world_query_proj,
                    planner.history_world_query_proj,
                    *horizon_world_block_modules,
                ]
                if module is not None
            )
        if getattr(planner, "grounded_clean_proposal_proj", None) is not None:
            shared_modules.append(planner.grounded_clean_proposal_proj)
        for bridge_name in (
            "ground_to_world_attnres",
            "world_to_policy_attnres",
        ):
            bridge = getattr(planner, bridge_name, None)
            if bridge is not None:
                shared_modules.append(bridge)
        depth = len(planner.blocks)
        min_scale = float(getattr(trainer, "layerwise_lr_min_scale", 0.30))
        min_scale = min(max(min_scale, 0.01), 1.0)
        if stage in {"contract", "stage1"}:
            groups.append(
                {
                    "params": _unique_params(shared_modules),
                    "lr": trainer.lr * 0.5,
                    "name": "shared_input_low_lr",
                }
            )
            for i, block in enumerate(planner.blocks):
                frac = 0.0 if depth <= 1 else float(i) / float(depth - 1)
                scale = min_scale + (1.0 - min_scale) * frac
                groups.append(
                    {
                        "params": _unique_params([block]),
                        "lr": trainer.lr * scale,
                        "name": f"dit_block_{i}_lr{scale:.2f}",
                    }
                )
            contract_modules = [
                planner.midcut_norm,
                planner.midcut_heads,
                planner.layer_contract_heads,
            ]
            if planner.layer_fm_probe is not None:
                contract_modules.append(planner.layer_fm_probe)
            if getattr(planner, "layer_consequence_cell", None) is not None:
                contract_modules.append(planner.layer_consequence_cell)
            event_probe_in_contract = bool(
                int(getattr(system.policy_config, "layer_causal_event_from_effect", 0))
                and float(getattr(trainer, "layer_event_loss_weight", 0.0)) > 0
            )
            if event_probe_in_contract:
                contract_modules.append(planner.event_probe)
            groups.append(
                {
                    "params": _unique_params(contract_modules),
                    "lr": trainer.lr * float(getattr(trainer, "midcut_head_lr_scale", 1.0)),
                    "name": "contract_adapters_heads",
                }
            )
            final_modules = [
                planner.final_norm,
                *legacy_action_readers,
                planner.controlled_dynamics,
                *legacy_motion_readers,
            ]
            if getattr(planner, "late_raw_detail_reader", None) is not None:
                final_modules.append(planner.late_raw_detail_reader)
            if getattr(planner, "policy_plan_compiler", None) is not None:
                final_modules.append(planner.policy_plan_compiler)
            if getattr(planner, "p2_effect_reader", None) is not None:
                final_modules.append(planner.p2_effect_reader)
            if getattr(planner, "consequence_plan_organizer", None) is not None:
                final_modules.append(planner.consequence_plan_organizer)
            if not event_probe_in_contract:
                final_modules.append(planner.event_probe)
            if getattr(planner, "residual_action_flow_denoiser", None) is not None:
                final_modules.append(planner.residual_action_flow_denoiser)
            if getattr(planner, "latent_main_action_decoder", None) is not None:
                final_modules.append(planner.latent_main_action_decoder)
            if getattr(planner, "latent_cvae_action_decoder", None) is not None:
                final_modules.append(planner.latent_cvae_action_decoder)
            if float(getattr(trainer, "layer_contract_final_action_loss_weight", 0.0)) > 0:
                weak_final_lr = trainer.lr * float(
                    getattr(trainer, "layer_contract_final_action_lr_scale", 0.30)
                )
                groups.append(
                    {
                        "params": _unique_params(final_modules),
                        "lr": weak_final_lr,
                        "name": "weak_final_policy_probe",
                    }
                )
                add_evidence_decoder_groups(
                    lr=weak_final_lr,
                    name="weak_evidence_latent_mmdit_action_decoder",
                )
                add_hierarchical_decoder_groups(
                    lr=weak_final_lr,
                    name="weak_hierarchical_mmdit_action_decoder",
                )
            groups.append(
                {
                    "params": [
                        parameter
                        for parameter in system.proposal.parameters()
                        if parameter.requires_grad
                    ],
                    "lr": trainer.proposal_lr,
                    "name": "proposal",
                }
            )
        else:
            upper_lr = trainer.lr * float(getattr(trainer, "upper_lr_scale", 0.20))
            single_stage_role_lr = bool(int(getattr(trainer, "single_stage_role_lr", 0)))
            shared_lr = trainer.lr if single_stage_role_lr else upper_lr * 0.5
            groups.append(
                {
                    "params": _unique_params(shared_modules),
                    "lr": shared_lr,
                    "name": (
                        "single_stage_shared_input"
                        if single_stage_role_lr
                        else "shared_input_low_lr"
                    ),
                }
            )
            for i, block in enumerate(planner.blocks):
                frac = 0.0 if depth <= 1 else float(i) / float(depth - 1)
                lr = (
                    trainer.lr
                    if single_stage_role_lr
                    else max(
                        upper_lr + (trainer.lr - upper_lr) * frac,
                        trainer.lr * min_scale * 0.25,
                    )
                )
                groups.append(
                    {
                        "params": _unique_params([block]),
                        "lr": lr,
                        "name": (
                            f"dit_block_{i}_single_stage"
                            if single_stage_role_lr
                            else f"dit_block_{i}_policy_layerwise"
                        ),
                    }
                )
            inherited_contract_lr = (
                trainer.lr
                if single_stage_role_lr
                else upper_lr * float(getattr(trainer, "midcut_head_lr_scale", 1.0))
            )
            groups.append(
                {
                    "params": _unique_params([planner.midcut_norm, planner.midcut_heads]),
                    "lr": inherited_contract_lr,
                    "name": (
                        "midcut_contract_heads_single_stage"
                        if single_stage_role_lr
                        else "midcut_contract_heads_low_lr"
                    ),
                }
            )
            adapter_modules = [planner.layer_contract_heads]
            if planner.layer_fm_probe is not None:
                adapter_modules.append(planner.layer_fm_probe)
            if getattr(planner, "layer_consequence_cell", None) is not None:
                adapter_modules.append(planner.layer_consequence_cell)
            adapter_lr_scale = float(
                getattr(trainer, "layer_contract_adapter_policy_lr_scale", 0.0)
            )
            adapter_lr = (
                trainer.lr
                if single_stage_role_lr
                else trainer.lr * adapter_lr_scale
                if adapter_lr_scale > 0
                else inherited_contract_lr
            )
            adapter_name = (
                "layer_contract_adapters_single_stage"
                if single_stage_role_lr
                else "layer_contract_adapters_reset_lr"
                if adapter_lr_scale > 0
                else "layer_contract_adapters_low_lr"
            )
            groups.append(
                {"params": _unique_params(adapter_modules), "lr": adapter_lr, "name": adapter_name}
            )
            final_modules = [
                planner.final_norm,
                *legacy_action_readers,
                planner.controlled_dynamics,
                planner.event_probe,
                *legacy_motion_readers,
            ]
            if getattr(planner, "late_raw_detail_reader", None) is not None:
                final_modules.append(planner.late_raw_detail_reader)
            if getattr(planner, "policy_plan_compiler", None) is not None:
                final_modules.append(planner.policy_plan_compiler)
            if getattr(planner, "p2_effect_reader", None) is not None:
                final_modules.append(planner.p2_effect_reader)
            if getattr(planner, "consequence_plan_organizer", None) is not None:
                final_modules.append(planner.consequence_plan_organizer)
            groups.append(
                {
                    "params": _unique_params(final_modules),
                    "lr": trainer.lr,
                    "name": "final_policy_heads",
                }
            )
            if getattr(planner, "residual_action_flow_denoiser", None) is not None:
                groups.append(
                    {
                        "params": list(planner.residual_action_flow_denoiser.parameters()),
                        "lr": trainer.lr
                        * float(getattr(trainer, "action_flow_residual_lr_scale", 1.5)),
                        "name": "residual_action_flow_denoiser",
                    }
                )
            if getattr(planner, "latent_main_action_decoder", None) is not None:
                groups.append(
                    {
                        "params": list(planner.latent_main_action_decoder.parameters()),
                        "lr": trainer.lr
                        * float(getattr(trainer, "latent_action_decoder_lr_scale", 1.5)),
                        "name": "latent_main_action_decoder",
                    }
                )
            if getattr(planner, "latent_cvae_action_decoder", None) is not None:
                groups.append(
                    {
                        "params": list(planner.latent_cvae_action_decoder.parameters()),
                        "lr": trainer.lr
                        * float(getattr(trainer, "latent_cvae_action_decoder_lr_scale", 1.0)),
                        "name": "latent_cvae_action_decoder",
                    }
                )
            if getattr(planner, "evidence_latent_mmdit_action_decoder", None) is not None:
                add_evidence_decoder_groups(
                    lr=trainer.lr
                    * float(getattr(trainer, "latent_cvae_action_decoder_lr_scale", 1.0)),
                    name="evidence_latent_mmdit_action_decoder",
                )
            if getattr(planner, "hierarchical_mmdit_action_decoder", None) is not None:
                add_hierarchical_decoder_groups(
                    lr=trainer.lr
                    * float(getattr(trainer, "latent_cvae_action_decoder_lr_scale", 1.0)),
                    name="hierarchical_mmdit_action_decoder",
                )
            groups.append(
                {
                    "params": [
                        parameter
                        for parameter in system.proposal.parameters()
                        if parameter.requires_grad
                    ],
                    "lr": trainer.proposal_lr,
                    "name": "proposal",
                }
            )
        flow_dino = getattr(planner, "flow_dino_evidence", None)
        if flow_dino is not None:
            groups.append(
                {
                    "params": [
                        parameter for parameter in flow_dino.parameters() if parameter.requires_grad
                    ],
                    "lr": trainer.lr * float(getattr(trainer, "flow_jepa_lr_scale", 1.0)),
                    "name": "flow_dino_evidence",
                }
            )
        add_object_top_group()
        return [group for group in groups if len(group["params"]) > 0]

    pre_modules = [
        planner.visual_memory,
        planner.rollout_codec,
        planner.seed,
        planner.time,
        planner.content_mod,
        planner.content_mod_scale,
        *list(planner.blocks[:cut]),
        planner.midcut_norm,
    ]
    if getattr(planner, "goal_resampler", None) is not None:
        pre_modules.append(planner.goal_resampler)
    if getattr(planner, "stateless_phase_adapter", None) is not None:
        pre_modules.append(planner.stateless_phase_adapter)
        pre_modules.extend(phase_world_block_modules)
    if getattr(planner, "stateless_horizon_adapter", None) is not None:
        pre_modules.extend(
            [
                planner.stateless_horizon_adapter,
                *horizon_world_block_modules,
            ]
        )
    if getattr(planner, "stateless_goal_phase_machine", None) is not None:
        pre_modules.extend(
            [
                planner.stateless_goal_phase_machine,
                *horizon_world_block_modules,
            ]
        )
    if getattr(planner, "grounded_clean_proposal_proj", None) is not None:
        pre_modules.append(planner.grounded_clean_proposal_proj)
    if getattr(planner, "ground_to_world_attnres", None) is not None:
        pre_modules.append(planner.ground_to_world_attnres)
    mid_modules = [planner.midcut_heads]
    post_modules = [
        *list(planner.blocks[cut:]),
        planner.final_norm,
        *legacy_action_readers,
        planner.controlled_dynamics,
        planner.event_probe,
        *legacy_motion_readers,
    ]
    if getattr(planner, "late_raw_detail_reader", None) is not None:
        post_modules.append(planner.late_raw_detail_reader)
    if getattr(planner, "policy_plan_compiler", None) is not None:
        post_modules.append(planner.policy_plan_compiler)
    if getattr(planner, "p2_effect_reader", None) is not None:
        post_modules.append(planner.p2_effect_reader)
    if getattr(planner, "consequence_plan_organizer", None) is not None:
        post_modules.append(planner.consequence_plan_organizer)
    if getattr(planner, "world_to_policy_attnres", None) is not None:
        post_modules.append(planner.world_to_policy_attnres)
    if getattr(planner, "phase_world_query_proj", None) is not None:
        post_modules.append(planner.phase_world_query_proj)
    if getattr(planner, "condition_world_query_proj", None) is not None:
        post_modules.append(planner.condition_world_query_proj)
    if getattr(planner, "history_world_query_proj", None) is not None:
        post_modules.append(planner.history_world_query_proj)
    if stage in {"contract", "stage1"}:
        groups.append(
            {"params": _unique_params(pre_modules), "lr": trainer.lr, "name": "pre_midcut_trunk"}
        )
        groups.append(
            {
                "params": _unique_params(mid_modules),
                "lr": trainer.lr * float(getattr(trainer, "midcut_head_lr_scale", 1.0)),
                "name": "midcut_contract_heads",
            }
        )
        groups.append(
            {
                "params": [
                    parameter
                    for parameter in system.proposal.parameters()
                    if parameter.requires_grad
                ],
                "lr": trainer.proposal_lr,
                "name": "proposal",
            }
        )
    else:
        upper_lr = trainer.lr * float(getattr(trainer, "upper_lr_scale", 0.20))
        groups.append(
            {
                "params": _unique_params(pre_modules),
                "lr": upper_lr,
                "name": "pre_midcut_trunk_low_lr",
            }
        )
        groups.append(
            {
                "params": _unique_params(mid_modules),
                "lr": upper_lr * float(getattr(trainer, "midcut_head_lr_scale", 1.0)),
                "name": "midcut_contract_heads_low_lr",
            }
        )
        groups.append(
            {"params": _unique_params(post_modules), "lr": trainer.lr, "name": "post_midcut_policy"}
        )
        if getattr(planner, "residual_action_flow_denoiser", None) is not None:
            groups.append(
                {
                    "params": list(planner.residual_action_flow_denoiser.parameters()),
                    "lr": trainer.lr
                    * float(getattr(trainer, "action_flow_residual_lr_scale", 1.5)),
                    "name": "residual_action_flow_denoiser",
                }
            )
        if getattr(planner, "latent_main_action_decoder", None) is not None:
            groups.append(
                {
                    "params": list(planner.latent_main_action_decoder.parameters()),
                    "lr": trainer.lr
                    * float(getattr(trainer, "latent_action_decoder_lr_scale", 1.5)),
                    "name": "latent_main_action_decoder",
                }
            )
        if getattr(planner, "latent_cvae_action_decoder", None) is not None:
            groups.append(
                {
                    "params": list(planner.latent_cvae_action_decoder.parameters()),
                    "lr": trainer.lr
                    * float(getattr(trainer, "latent_cvae_action_decoder_lr_scale", 1.0)),
                    "name": "latent_cvae_action_decoder",
                }
            )
        if getattr(planner, "evidence_latent_mmdit_action_decoder", None) is not None:
            add_evidence_decoder_groups(
                lr=trainer.lr * float(getattr(trainer, "latent_cvae_action_decoder_lr_scale", 1.0)),
                name="evidence_latent_mmdit_action_decoder",
            )
        if getattr(planner, "hierarchical_mmdit_action_decoder", None) is not None:
            add_hierarchical_decoder_groups(
                lr=trainer.lr * float(getattr(trainer, "latent_cvae_action_decoder_lr_scale", 1.0)),
                name="hierarchical_mmdit_action_decoder",
            )
        groups.append(
            {
                "params": [
                    parameter
                    for parameter in system.proposal.parameters()
                    if parameter.requires_grad
                ],
                "lr": trainer.proposal_lr,
                "name": "proposal",
            }
        )
    flow_dino = getattr(planner, "flow_dino_evidence", None)
    if flow_dino is not None:
        groups.append(
            {
                "params": [
                    parameter for parameter in flow_dino.parameters() if parameter.requires_grad
                ],
                "lr": trainer.lr * float(getattr(trainer, "flow_jepa_lr_scale", 1.0)),
                "name": "flow_dino_evidence",
            }
        )
    add_object_top_group()
    return [group for group in groups if len(group["params"]) > 0]


def _validate_object_optimizer_ownership(
    system: V39PolicySystem,
    groups: Sequence[dict[str, Any]],
) -> None:
    """Every trainable system parameter must have exactly one optimizer owner.

    The capability modules are the most likely omission point, but validating
    only that subset would still permit an ancestral trainable parameter to be
    silently dropped or duplicated while the new top appears healthy.  The
    object graph is fresh-only, so its preflight can enforce the stronger
    whole-system invariant without affecting historical launchers.
    """

    if not int(
        getattr(
            system.policy_config,
            "flow_jepa_object_intent_dynamics_mainline",
            0,
        )
    ):
        return
    owned_count: dict[int, int] = {}
    for group in groups:
        for parameter in group["params"]:
            owned_count[id(parameter)] = owned_count.get(id(parameter), 0) + 1
    trainable = {
        id(parameter): name
        for name, parameter in system.named_parameters()
        if parameter.requires_grad
    }
    missing = [
        name
        for parameter_id, name in trainable.items()
        if owned_count.get(parameter_id, 0) == 0
    ]
    duplicate = [
        f"{trainable.get(parameter_id, '<unregistered>')}:{count}"
        for parameter_id, count in owned_count.items()
        if count != 1
    ]
    extra = [
        f"<unregistered>:{parameter_id}"
        for parameter_id in owned_count
        if parameter_id not in trainable
    ]
    if missing or duplicate or extra:
        raise RuntimeError(
            "object-intent optimizer ownership is not a one-to-one partition "
            "of all trainable system parameters: "
            f"missing={missing[:12]} duplicate={duplicate[:12]} extra={extra[:12]}"
        )


def _is_contract_stage(trainer: V39PolicyTrainerConfig) -> bool:
    return str(getattr(trainer, "training_stage", "contract")).lower().replace("-", "_") in {
        "contract",
        "stage1",
    }


def _is_flow_jepa_stage1(system: V39PolicySystem, trainer: V39PolicyTrainerConfig) -> bool:
    return bool(
        _is_contract_stage(trainer) and int(getattr(system.policy_config, "flow_jepa_enabled", 0))
    )


def _stable_json_fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_contract_manifest(
    *,
    system: V39PolicySystem,
    trainer: V39PolicyTrainerConfig,
    context: dict[str, Any],
) -> dict[str, Any]:
    trainer_contract = asdict(trainer)
    # These fields control how long/where a compatible run proceeds, not the
    # mathematical experiment.  They may change for an explicit resume.
    for name in ("epochs", "max_train_batches", "max_val_batches"):
        trainer_contract.pop(name, None)
    contract = {
        "policy_config": asdict(system.policy_config),
        "trainer_contract": trainer_contract,
        "context_schema": context.get("schema"),
        "dataset": context.get("dataset"),
        "splits": context.get("splits"),
        "visual_geometry": context.get("visual_geometry"),
        "goal_language": context.get("goal_language"),
        "performance_contract": context.get("performance_contract"),
        "source_fingerprint": context.get("source_fingerprint"),
    }
    return {
        "schema": "clearvla-run-contract-manifest-v1",
        "fingerprint": _stable_json_fingerprint(contract),
        "contract": contract,
    }


def _prepare_run_directory(
    *,
    out_dir: Path,
    manifest: dict[str, Any],
    resume: Path | None,
) -> None:
    """Prevent accidental JSONL append/checkpoint overwrite across contracts."""

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "run_manifest.json"
    artifacts = (
        out_dir / "v39_policy_epochs.jsonl",
        out_dir / "v40_policy_summary.json",
        out_dir / "checkpoints" / "latest.pt",
    )
    has_artifacts = any(path.exists() for path in artifacts)
    existing: dict[str, Any] | None = None
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    if has_artifacts and resume is None:
        raise FileExistsError(
            f"output directory already contains a run: {out_dir}; choose a new OUT_DIR "
            "or pass an explicit compatible --resume checkpoint"
        )
    if has_artifacts and existing is None:
        raise ValueError(
            f"cannot resume legacy output directory without run_manifest.json: {out_dir}; "
            "resume into a new OUT_DIR instead"
        )
    if existing is not None and existing.get("fingerprint") != manifest.get("fingerprint"):
        raise ValueError(
            "run contract mismatch for output directory: "
            f"saved={existing.get('fingerprint')}, current={manifest.get('fingerprint')}"
        )
    if existing is None:
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )


def train_v39_policy(
    *,
    system: V39PolicySystem,
    train_loader: DataLoader,
    val_loader: DataLoader,
    conditioner: RDT2Conditioner,
    device: torch.device,
    dtype: torch.dtype,
    camera_names: Sequence[str],
    action_normalizer: ArrayNormalizer,
    state_normalizer: ArrayNormalizer,
    trainer: V39PolicyTrainerConfig,
    out_dir: Path,
    context: dict[str, Any],
    resume: Path | None = None,
) -> dict[str, Any]:
    run_manifest = _run_contract_manifest(
        system=system,
        trainer=trainer,
        context=context,
    )
    _prepare_run_directory(
        out_dir=out_dir,
        manifest=run_manifest,
        resume=resume,
    )
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    memory_reporter = CudaMemoryReporter(
        device=device,
        out_dir=out_dir,
        every=int(getattr(trainer, "memory_report_every", 0)),
        detail=int(getattr(trainer, "memory_report_detail", 0)),
        sync=int(getattr(trainer, "memory_report_sync", 0)),
    )
    system.to(device=device, dtype=torch.float32)
    if memory_reporter.enabled:
        memory_reporter.snapshot(tag="after_model_to_device", phase="setup", print_line=True)
    optimizer_groups = _optimizer_groups(system, trainer)
    _validate_object_optimizer_ownership(system, optimizer_groups)
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        weight_decay=trainer.weight_decay,
        betas=(trainer.beta1, trainer.beta2),
        eps=trainer.eps,
    )
    steps_per_epoch = trainer.max_train_batches or len(train_loader)
    schedule = scheduler(
        optimizer, steps_per_epoch * trainer.epochs, trainer.warmup_steps, trainer.min_lr_ratio
    )
    start_epoch, global_step = 1, 0
    # do_before_v78 M1 quarantine proof: CALLGRAPH_AUDIT=1 turns this run into
    # a one-batch diagnostic -- hooks attach here, the report is written right
    # after the first backward (plus one sampled eval batch), then the run
    # exits.  Zero effect when the env var is absent.
    callgraph_auditor = None
    import os as _os

    if _os.environ.get("CALLGRAPH_AUDIT", "0") == "1":
        from clearvla.tools.callgraph_audit import CallGraphAuditor

        callgraph_auditor = CallGraphAuditor(system)
        callgraph_auditor.attach(first_phase="train")
        print(
            "[callgraph-audit] hooks attached; diagnostic run, will exit after first batch",
            flush=True,
        )
    history: list[dict[str, Any]] = []
    best = {
        "full_mse": float("inf"),
        "gripper_f1": -float("inf"),
        "gripper_recall": -float("inf"),
        "balanced": float("inf"),
        "deploy_full_rmse": float("inf"),
        "layer_contract": float("inf"),
        "stage1_representation": float("inf"),
    }
    if resume is not None:
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        if payload.get("schema") not in POLICY_CHECKPOINT_SCHEMAS:
            raise ValueError("resume checkpoint is not V39/V40 policy")
        saved_policy = payload.get("policy_config", {})
        saved_final_decoder = str(saved_policy.get("final_action_decoder", "legacy"))
        current_final_decoder = str(getattr(system.policy_config, "final_action_decoder", "legacy"))
        if saved_final_decoder != current_final_decoder:
            raise ValueError(
                "resume final-action-decoder mismatch: "
                f"checkpoint={saved_final_decoder!r}, current={current_final_decoder!r}"
            )
        saved_flow_jepa = int(saved_policy.get("flow_jepa_enabled", 0))
        current_flow_jepa = int(getattr(system.policy_config, "flow_jepa_enabled", 0))
        if saved_flow_jepa != current_flow_jepa:
            raise ValueError(
                "resume Flow-DINO JEPA architecture mismatch: "
                f"checkpoint={saved_flow_jepa}, current={current_flow_jepa}; "
                "an old V94 checkpoint may be used only as --stage1-checkpoint"
            )
        if current_flow_jepa:
            for field in (
                "future_grid_size",
                "flow_jepa_grid_size",
                "flow_jepa_feature_dim",
                "flow_jepa_flow_iters",
                "flow_jepa_corr_levels",
                "flow_jepa_corr_radius",
                "flow_jepa_directed_canvas_attention",
                "flow_jepa_late_bottleneck",
                "flow_jepa_dense_depth",
                "flow_jepa_fine_radius",
                "flow_jepa_reader_radius",
                "flow_jepa_reader_heads",
                "flow_jepa_raw_image_enabled",
                "flow_jepa_role_hierarchy",
                "flow_jepa_zero_flow_guard",
                "future_anchors",
                "flow_jepa_stage_tokens",
            ):
                default_value = (
                    0
                    if field
                    in {
                        "flow_jepa_raw_image_enabled",
                        "flow_jepa_role_hierarchy",
                        "flow_jepa_zero_flow_guard",
                    }
                    else -1
                )
                saved_value = int(saved_policy.get(field, default_value))
                current_value = int(getattr(system.policy_config, field))
                if saved_value != current_value:
                    raise ValueError(
                        f"resume {field} mismatch: checkpoint={saved_value}, current={current_value}"
                    )
            if int(getattr(system.policy_config, "flow_jepa_raw_image_enabled", 0)):
                for field in (
                    "depth",
                    "midcut_layer",
                    "flow_jepa_raw_base_channels",
                    "flow_jepa_raw_mid_radius",
                    "flow_jepa_raw_high_radius",
                    "flow_jepa_raw_reader_radius",
                    "flow_jepa_raw_reader_heads",
                    "flow_jepa_raw_activation_checkpoint",
                    "flow_jepa_strict_role_visual_path",
                    "flow_jepa_complementary_raw_detail",
                    "flow_jepa_source_aligned_raw_fusion",
                    "flow_jepa_grounding_blocks",
                    "flow_jepa_world_blocks",
                    "flow_jepa_policy_blocks",
                    "flow_jepa_policy_workspace_fixed_fusion",
                ):
                    saved_value = int(saved_policy.get(field, -1))
                    current_value = int(getattr(system.policy_config, field))
                    if saved_value != current_value:
                        raise ValueError(
                            f"resume {field} mismatch: checkpoint={saved_value}, "
                            f"current={current_value}"
                        )
                saved_workspace_scale = float(
                    saved_policy.get("flow_jepa_policy_workspace_scale", float("nan"))
                )
                current_workspace_scale = float(
                    system.policy_config.flow_jepa_policy_workspace_scale
                )
                if saved_workspace_scale != current_workspace_scale:
                    raise ValueError(
                        "resume flow_jepa_policy_workspace_scale mismatch: "
                        f"checkpoint={saved_workspace_scale}, current={current_workspace_scale}"
                    )
                # V102 routing switches can change experiment semantics without
                # changing tensor shapes, so strict state-dict loading cannot
                # detect a mismatched resume. Missing fields mean "off" for
                # historical V98-V101 checkpoints.
                _validate_v102_resume_contract(
                    saved_policy,
                    system.policy_config,
                )
                grounded_manifest_active = bool(
                    int(
                        getattr(
                            system.policy_config,
                            "flow_jepa_grounded_intent_effect_mainline",
                            0,
                        )
                    )
                )
                object_manifest_active = bool(
                    int(
                        getattr(
                            system.policy_config,
                            "flow_jepa_object_intent_dynamics_mainline",
                            0,
                        )
                    )
                )
                if grounded_manifest_active or object_manifest_active:
                    saved_manifest = (
                        payload.get("context", {}).get(
                            "architecture_manifest"
                        )
                    )
                    current_manifest = context.get(
                        "architecture_manifest"
                    )
                    if not isinstance(saved_manifest, dict):
                        raise ValueError(
                            "capability resume checkpoint has no architecture "
                            "manifest; start a fresh run"
                        )
                    if not isinstance(current_manifest, dict):
                        raise ValueError(
                            "capability current run has no architecture manifest"
                        )
                    manifest_parser = (
                        object_intent_manifest_from_mapping
                        if object_manifest_active
                        else manifest_from_mapping
                    )
                    saved_identity = manifest_parser(saved_manifest)
                    current_identity = manifest_parser(current_manifest)
                    if (
                        saved_identity.as_dict()
                        != current_identity.as_dict()
                    ):
                        raise ValueError(
                            "capability architecture manifest mismatch; "
                            "start a fresh top run"
                        )
            saved_offsets = tuple(
                int(value) for value in saved_policy.get("flow_jepa_history_offsets", ())
            )
            current_offsets = tuple(
                int(value) for value in system.policy_config.flow_jepa_history_offsets
            )
            if saved_offsets != current_offsets:
                raise ValueError(
                    "resume flow_jepa_history_offsets mismatch: "
                    f"checkpoint={saved_offsets}, current={current_offsets}"
                )
            saved_window_offsets, saved_stage_offset = _saved_flow_jepa_hierarchy(saved_policy)
            current_window_offsets = tuple(
                int(value) for value in system.policy_config.flow_jepa_effective_window_offsets
            )
            if saved_window_offsets != current_window_offsets:
                raise ValueError(
                    "resume flow_jepa_window_offsets mismatch: "
                    f"checkpoint={saved_window_offsets}, current={current_window_offsets}; "
                    "pre-hierarchy V95 checkpoints may be used only as --stage1-checkpoint"
                )
            current_stage_offset = (
                0
                if int(system.policy_config.flow_jepa_late_bottleneck)
                else int(system.policy_config.flow_jepa_effective_stage_offset)
            )
            if int(system.policy_config.flow_jepa_late_bottleneck):
                saved_stage_offset = int(saved_policy.get("flow_jepa_stage_offset", 0))
            if saved_stage_offset != current_stage_offset:
                raise ValueError(
                    "resume flow_jepa_stage_offset mismatch: "
                    f"checkpoint={saved_stage_offset}, current={current_stage_offset}; "
                    "pre-hierarchy V95 checkpoints may be used only as --stage1-checkpoint"
                )
        saved_action_history = int(saved_policy.get("action_history_enabled", 0))
        current_action_history = int(getattr(system.policy_config, "action_history_enabled", 0))
        if saved_action_history != current_action_history:
            raise ValueError(
                "resume action-history architecture mismatch: "
                f"checkpoint={saved_action_history}, current={current_action_history}; "
                "use the checkpoint only as --stage1-checkpoint"
            )
        if current_action_history:
            for field in (
                "executed_history_length",
                "action_history_recent_tokens",
                "action_history_summary_tokens",
            ):
                saved_value = int(saved_policy.get(field, -1))
                current_value = int(getattr(system.policy_config, field))
                if saved_value != current_value:
                    raise ValueError(
                        f"resume {field} mismatch: checkpoint={saved_value}, "
                        f"current={current_value}"
                    )
            saved_action_offsets = tuple(
                int(value) for value in saved_policy.get("executed_action_offsets", ())
            )
            current_action_offsets = tuple(
                int(value) for value in system.policy_config.executed_action_offsets
            )
            if saved_action_offsets != current_action_offsets:
                raise ValueError(
                    "resume executed_action_offsets mismatch: "
                    f"checkpoint={saved_action_offsets}, current={current_action_offsets}"
                )
        saved_goal = int(saved_policy.get("goal_conditioning_enabled", 0))
        current_goal = int(getattr(system.policy_config, "goal_conditioning_enabled", 0))
        if saved_goal != current_goal:
            raise ValueError(
                "resume goal-conditioning architecture mismatch: "
                f"checkpoint={saved_goal}, current={current_goal}; "
                "use the checkpoint only as --stage1-checkpoint"
            )
        if current_goal:
            for field in (
                "goal_token_count",
                "goal_language_dim",
                "goal_language_max_tokens",
                "goal_resampler_depth",
            ):
                saved_value = int(saved_policy.get(field, -1))
                current_value = int(getattr(system.policy_config, field))
                if saved_value != current_value:
                    raise ValueError(
                        f"resume {field} mismatch: checkpoint={saved_value}, "
                        f"current={current_value}"
                    )
            saved_goal_meta = payload.get("context", {}).get("goal_language") or {}
            current_goal_meta = context.get("goal_language") or {}
            for field in ("embedding_sha256",):
                if saved_goal_meta.get(field) != current_goal_meta.get(field):
                    raise ValueError(
                        f"resume goal-language {field} mismatch; use the same T5 condition "
                        "or start a new run"
                    )
        if current_final_decoder == "hierarchical_mmdit_action":
            saved_architecture = str(
                saved_policy.get("hierarchical_mmdit_architecture_version", "competitive_v1")
            )
            current_architecture = str(
                getattr(
                    system.policy_config,
                    "hierarchical_mmdit_architecture_version",
                    "post_gate_contraction_sidecar_v12_value_dwell",
                )
            )
            if saved_architecture != current_architecture:
                raise ValueError(
                    "resume hierarchical-MMDiT architecture mismatch: "
                    f"checkpoint={saved_architecture!r}, current={current_architecture!r}; "
                    "use the checkpoint as --stage1-checkpoint to retain the trunk while "
                    "reinitializing the final decoder"
                )
            for field in (
                "hierarchical_mmdit_depth",
                "hierarchical_mmdit_refine_steps",
                "hierarchical_mmdit_low_slots",
                "hierarchical_mmdit_stage_slots",
                "hierarchical_mmdit_noisy_causal",
                "hierarchical_mmdit_noisy_gate_mode",
                "hierarchical_mmdit_output_contract",
                "hierarchical_mmdit_operator_stages",
                "hierarchical_mmdit_operator_rank",
                "hierarchical_mmdit_operator_groups",
                "hierarchical_mmdit_operator_contraction_warmup_steps",
                "hierarchical_mmdit_operator_contraction_transition_steps",
                "hierarchical_mmdit_unified_controller",
                "hierarchical_mmdit_control_tokens",
                "hierarchical_mmdit_controller_depth",
                "hierarchical_mmdit_controller_heads",
                "hierarchical_mmdit_spectral_state",
                "hierarchical_mmdit_operation_candidate_probes",
            ):
                saved_value = int(saved_policy.get(field, getattr(system.policy_config, field)))
                current_value = int(getattr(system.policy_config, field))
                if saved_value != current_value:
                    raise ValueError(
                        f"resume {field} mismatch: checkpoint={saved_value}, current={current_value}"
                    )
            for field in (
                "hierarchical_mmdit_ffn_expansion",
                "hierarchical_mmdit_noisy_gate_min",
                "hierarchical_mmdit_noisy_gate_power",
                "hierarchical_mmdit_operator_depth_logit_init",
                "hierarchical_mmdit_residual_scale_init",
                "hierarchical_mmdit_residual_scale_max",
                "hierarchical_mmdit_random_prefix_probability",
                "hierarchical_mmdit_controller_ffn_expansion",
                "hierarchical_mmdit_spectral_arm_start_fraction",
                "hierarchical_mmdit_spectral_gripper_start_fraction",
                "hierarchical_mmdit_spectral_temperature",
                "hierarchical_mmdit_spectral_schedule_power",
                "hierarchical_mmdit_spectral_controller_shift_limit",
            ):
                saved_value = float(saved_policy.get(field, getattr(system.policy_config, field)))
                current_value = float(getattr(system.policy_config, field))
                if not math.isclose(saved_value, current_value, rel_tol=0.0, abs_tol=1e-12):
                    raise ValueError(
                        f"resume {field} mismatch: checkpoint={saved_value}, current={current_value}"
                    )
            for field in ("hierarchical_mmdit_schedule_mode",):
                saved_value = str(saved_policy.get(field, getattr(system.policy_config, field)))
                current_value = str(getattr(system.policy_config, field))
                if saved_value != current_value:
                    raise ValueError(
                        f"resume {field} mismatch: checkpoint={saved_value!r}, current={current_value!r}"
                    )
            runtime_override_fields = (
                "hierarchical_mmdit_exhaustion_mode",
                "hierarchical_mmdit_action_response_thresholds",
                "hierarchical_mmdit_stage_pressure_thresholds",
                "hierarchical_mmdit_action_response_floor",
                "hierarchical_mmdit_exhaustion_confirm_steps",
                "hierarchical_mmdit_dwell_mode",
                "hierarchical_mmdit_operation_value_warmup_steps",
            )
            for field in runtime_override_fields:
                saved_value = saved_policy.get(field, getattr(system.policy_config, field))
                current_value = getattr(system.policy_config, field)
                if field.endswith("_thresholds"):
                    saved_value = tuple(float(value) for value in saved_value)
                    current_value = tuple(float(value) for value in current_value)
                if saved_value != current_value:
                    print(
                        f"[v39-resume] runtime refinement override {field}: "
                        f"checkpoint={saved_value!r} current={current_value!r}",
                        flush=True,
                    )
        saved_workspace_tokens = int(saved_policy.get("latent_cvae_horizon_tokens", 24))
        current_workspace_tokens = int(
            getattr(system.policy_config, "latent_cvae_horizon_tokens", 24)
        )
        if saved_workspace_tokens != current_workspace_tokens:
            raise ValueError(
                "resume workspace-token mismatch: "
                f"checkpoint={saved_workspace_tokens}, current={current_workspace_tokens}"
            )
        saved_hierarchical = int(saved_policy.get("latent_cvae_hierarchical_workspace", 0))
        current_hierarchical = int(
            getattr(system.policy_config, "latent_cvae_hierarchical_workspace", 0)
        )
        if saved_hierarchical != current_hierarchical:
            raise ValueError(
                "resume hierarchical-workspace mismatch: "
                f"checkpoint={saved_hierarchical}, current={current_hierarchical}"
            )
        if current_hierarchical:
            saved_stage_slots = int(saved_policy.get("latent_cvae_stage_slots", 6))
            current_stage_slots = int(getattr(system.policy_config, "latent_cvae_stage_slots", 6))
            if saved_stage_slots != current_stage_slots:
                raise ValueError(
                    "resume stage-slot mismatch: "
                    f"checkpoint={saved_stage_slots}, current={current_stage_slots}"
                )
        system.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        schedule.load_state_dict(payload["scheduler"])
        start_epoch = int(payload["epoch"]) + 1
        global_step = int(payload["global_step"])
        history = list(payload.get("history", []))
        best.update(payload.get("best", {}))
        restore_rng(payload.get("rng"))

    flow_jepa_stage1 = _is_flow_jepa_stage1(system, trainer)
    if flow_jepa_stage1:
        _preflight_flow_jepa_stage1(
            system=system,
            loader=val_loader,
            conditioner=conditioner,
            device=device,
            dtype=dtype,
            camera_names=camera_names,
            trainer=trainer,
        )
    else:
        _preflight_evidence_dynamic_sampling(
            system=system,
            loader=val_loader,
            conditioner=conditioner,
            device=device,
            dtype=dtype,
            camera_names=camera_names,
            trainer=trainer,
        )

    for epoch in range(start_epoch, trainer.epochs + 1):
        system.train()
        train_batch_sampler = getattr(train_loader, "batch_sampler", None)
        if hasattr(train_batch_sampler, "set_epoch"):
            train_batch_sampler.set_epoch(epoch)
        metric_sums: dict[str, Tensor] = {}
        metric_counts: dict[str, int] = {}
        metric_count = 0
        throughput_start = time.perf_counter()
        throughput_batch = 0
        include_future = _needs_future_targets(trainer, epoch)
        include_counterfactuals = (
            False if flow_jepa_stage1 else _needs_action_counterfactuals(trainer, epoch)
        )
        for batch_index, batch in enumerate(train_loader, start=1):
            if trainer.max_train_batches and batch_index > trainer.max_train_batches:
                break
            use_future = include_future and (
                not trainer.future_latent_max_batches
                or batch_index <= trainer.future_latent_max_batches
            )
            report_mem = memory_reporter.should_report(batch_index)
            if report_mem:
                memory_reporter.reset_peak()
                if memory_reporter.detail:
                    memory_reporter.snapshot(
                        tag="train_batch_start",
                        epoch=epoch,
                        batch=batch_index,
                        global_step=global_step,
                        extra={"use_future": bool(use_future)},
                    )
            sample = prepare_v39_policy_sample(
                batch,
                conditioner=conditioner,
                system=system,
                camera_names=camera_names,
                device=device,
                dtype=dtype,
                include_target_visual=use_future,
            )
            if report_mem and memory_reporter.detail:
                memory_reporter.snapshot(
                    tag="train_after_prepare",
                    epoch=epoch,
                    batch=batch_index,
                    global_step=global_step,
                    extra={"use_future": bool(use_future)},
                )
            # Clear every model gradient, including parameters intentionally
            # frozen out of the current optimizer stage.  This prevents stale
            # gradients from accumulating and polluting global grad clipping.
            system.zero_grad(set_to_none=True)
            hierarchical_decoder = getattr(
                system.planner, "hierarchical_mmdit_action_decoder", None
            )
            if hierarchical_decoder is not None and not flow_jepa_stage1:
                hierarchical_decoder.set_operator_contraction_training_step(global_step)
            evidence_decoder = getattr(system.planner, "evidence_latent_mmdit_action_decoder", None)
            if evidence_decoder is not None and not flow_jepa_stage1:
                evidence_decoder.set_execution_training_step(global_step)
            if report_mem and memory_reporter.detail:
                memory_reporter.snapshot(
                    tag="train_after_zero_grad",
                    epoch=epoch,
                    batch=batch_index,
                    global_step=global_step,
                    extra={"use_future": bool(use_future)},
                )
            layer_mode = _uses_layer_adapter_contract(trainer)
            contract_stage = _is_contract_stage(trainer)
            stop_midcut = contract_stage and not layer_mode
            layer_aux_contribution: Tensor | None = None
            collect_step_diagnostics = bool(
                batch_index == 1
                or (trainer.log_every and batch_index % int(trainer.log_every) == 0)
                or callgraph_auditor is not None
            )
            with autocast_context(device, dtype):
                if flow_jepa_stage1:
                    target_visual = sample.get("target_visual")
                    if target_visual is None:
                        raise RuntimeError("V95 Stage1 batch is missing its future visual target")
                    output = system.flow_jepa_stage1_forward(
                        sample["visual"],
                        sample["history_state"],
                        sample["executed_action_history"],
                        sample["state"],
                        target_visual,
                        raw_visual=sample.get("raw_visual"),
                    )
                    losses = flow_jepa_stage1_losses(
                        output,
                        trainer,
                        enable_future_loss=use_future,
                    )
                    losses["stage_contract"] = losses["loss"].detach()
                else:
                    output = system.flow_training_forward(
                        sample["visual"],
                        sample["history_state"],
                        sample["executed_action_history"],
                        sample["state"],
                        sample["policy_action"],
                        raw_visual=sample.get("raw_visual"),
                        action_state=sample["action_state"],
                        target_visual=sample.get("target_visual"),
                        future_training_pack=_object_intent_future_training_pack(
                            sample,
                            system=system,
                            require_teacher=bool(use_future),
                        ),
                        # Future teacher targets and action counterfactuals are
                        # different requirements.  V95 needs frozen future DINO
                        # targets, but its zero-weight legacy contrast objectives
                        # must not retain two extra full policy graphs.
                        make_counterfactuals=bool(use_future and include_counterfactuals),
                        stop_at_midcut=stop_midcut,
                        collect_audit_metrics=collect_step_diagnostics,
                    )
                if flow_jepa_stage1:
                    # The dedicated branch above already owns the complete
                    # representation objective.  Do not fall through into an
                    # old contract or policy action loss.
                    pass
                elif contract_stage and layer_mode:
                    losses = layer_contract_losses(
                        system, sample, output, trainer, enable_future_loss=use_future
                    )
                    final_weight = float(
                        getattr(trainer, "layer_contract_final_action_loss_weight", 0.0)
                    )
                    if final_weight > 0:
                        final_losses = flow_losses(
                            system,
                            sample,
                            output,
                            trainer,
                            enable_future_loss=False,
                            global_step=global_step,
                        )
                        losses["loss"] = losses["loss"] + final_weight * final_losses["loss"]
                        losses["final_action_probe"] = final_losses["loss"].detach()
                    losses["stage_contract"] = losses["loss"].detach()
                    losses["layer_adapter_contract"] = torch.as_tensor(
                        1.0, device=losses["loss"].device, dtype=losses["loss"].dtype
                    )
                elif stop_midcut:
                    losses = flow_losses(
                        system,
                        sample,
                        output,
                        trainer,
                        enable_future_loss=use_future,
                        global_step=global_step,
                    )
                    losses["stage_contract"] = losses["loss"].detach()
                else:
                    losses = flow_losses(
                        system,
                        sample,
                        output,
                        trainer,
                        enable_future_loss=use_future,
                        global_step=global_step,
                    )
                    total_loss = losses["loss"]
                    aux_key = "layer_contract" if layer_mode else "midcut_contract"
                    aux_scale = (
                        _layer_contract_aux_scale(trainer, epoch)
                        if layer_mode
                        else _midcut_aux_scale(trainer, epoch)
                    )
                    if aux_scale > 0:
                        if layer_mode:
                            aux_losses = layer_contract_losses(
                                system,
                                sample,
                                output,
                                trainer,
                                enable_future_loss=use_future,
                            )
                            # ``loss`` is the weighted layer-contract objective.
                            # Using the raw ``layer_contract`` here silently made
                            # layer_contract_loss_weight ineffective in Stage 2.
                            aux_objective = aux_losses["loss"]
                        else:
                            aux_losses = midcut_contract_losses(
                                system,
                                sample,
                                output,
                                trainer,
                                enable_future_loss=use_future,
                            )
                            aux_objective = aux_losses[aux_key]
                        scaled_aux_objective = aux_scale * aux_objective
                        total_loss = total_loss + scaled_aux_objective
                        if layer_mode:
                            layer_aux_contribution = scaled_aux_objective.detach()
                        # Merge auxiliary logs without overwriting the deployable
                        # policy loss.  Zero-scale objectives are not evaluated:
                        # computing them only for logging used memory and exposed
                        # semantics that did not participate in optimization.
                        for key, value in aux_losses.items():
                            if key == "loss":
                                losses[f"aux_{aux_key}_loss"] = (
                                    value.detach() if torch.is_tensor(value) else value
                                )
                            elif key in losses:
                                losses[f"aux_{aux_key}_{key}"] = (
                                    value.detach() if torch.is_tensor(value) else value
                                )
                            else:
                                losses[key] = value
                    losses["loss"] = total_loss
                    aux_scale_key = "layer_contract_aux_scale" if layer_mode else "midcut_aux_scale"
                    losses[aux_scale_key] = torch.as_tensor(
                        aux_scale, device=losses["loss"].device, dtype=losses["loss"].dtype
                    )
            if not flow_jepa_stage1:
                _attach_intent_frame_progress_audit(losses, sample, output)
            if evidence_decoder is not None and not contract_stage:
                _attach_v94_loss_ledger(
                    losses,
                    trainer,
                    enable_future_loss=use_future,
                    layer_aux_contribution=layer_aux_contribution,
                )
            if report_mem and memory_reporter.detail:
                memory_reporter.snapshot(
                    tag="train_after_forward_loss",
                    epoch=epoch,
                    batch=batch_index,
                    global_step=global_step,
                    extra={"use_future": bool(use_future)},
                )
            if not torch.isfinite(losses["loss"].detach()).all():
                raise FloatingPointError(
                    f"non-finite training loss before backward at epoch={epoch} batch={batch_index}"
                )
            losses["loss"].float().backward()
            collect_grad_diagnostics = collect_step_diagnostics
            if collect_grad_diagnostics:
                _attach_grad_diagnostics(losses, system)
            losses["grad_diagnostics_coverage"] = losses["loss"].new_tensor(
                float(collect_grad_diagnostics), dtype=torch.float32
            )
            if callgraph_auditor is not None:
                callgraph_auditor.capture_gradients()
                callgraph_auditor.begin_phase("sample")
                with torch.no_grad():
                    evaluate_v39_policy(
                        system=system,
                        loader=val_loader,
                        conditioner=conditioner,
                        device=device,
                        dtype=dtype,
                        camera_names=camera_names,
                        action_normalizer=action_normalizer,
                        trainer=trainer,
                        max_batches=1,
                    )
                callgraph_auditor.detach()
                report_path = callgraph_auditor.write_report(
                    out_dir / "callgraph_audit",
                    context_note=(
                        f"epoch={epoch} batch={batch_index} decoder="
                        + (
                            "hierarchical"
                            if getattr(system.planner, "hierarchical_mmdit_action_decoder", None)
                            is not None
                            else "legacy"
                        )
                    ),
                )
                print(
                    f"[callgraph-audit] report written to {report_path}; exiting diagnostic run",
                    flush=True,
                )
                return {"callgraph_audit": str(report_path)}
            if report_mem and memory_reporter.detail:
                memory_reporter.snapshot(
                    tag="train_after_backward",
                    epoch=epoch,
                    batch=batch_index,
                    global_step=global_step,
                    extra={"use_future": bool(use_future)},
                )
            latent_decoder = getattr(system.planner, "latent_cvae_action_decoder", None)
            clean_decoder = getattr(system.planner, "hierarchical_mmdit_action_decoder", None)
            evidence_decoder = getattr(system.planner, "evidence_latent_mmdit_action_decoder", None)
            decoder_for_local_clip = (
                clean_decoder
                if clean_decoder is not None
                else evidence_decoder
                if evidence_decoder is not None
                else latent_decoder
            )
            exit_controller_params = (
                list(clean_decoder.exit_controller_parameters())
                if clean_decoder is not None
                else []
            )
            exit_controller_ids = {id(parameter) for parameter in exit_controller_params}
            latent_clip = float(getattr(trainer, "latent_cvae_grad_clip", 0.0))
            if decoder_for_local_clip is not None and latent_clip > 0:
                local_clip_params = [
                    parameter
                    for parameter in decoder_for_local_clip.parameters()
                    if id(parameter) not in exit_controller_ids
                ]
                _clip_grad_norm_or_report(
                    local_clip_params,
                    latent_clip,
                    system=system,
                    epoch=epoch,
                    batch=batch_index,
                    label="decoder-local",
                )
                clip_key = (
                    "grad_hierarchical_mmdit_action_post_clip"
                    if clean_decoder is not None
                    else "grad_evidence_latent_mmdit_action_post_clip"
                    if evidence_decoder is not None
                    else "grad_latent_cvae_action_post_clip"
                )
                if collect_grad_diagnostics:
                    losses[clip_key] = _parameter_grad_norm(
                        local_clip_params,
                        reference=losses["loss"],
                    )
            main_clip_params = [
                parameter
                for parameter in system.parameters()
                if id(parameter) not in exit_controller_ids
            ]
            grad = _clip_grad_norm_or_report(
                main_clip_params,
                trainer.grad_clip,
                system=system,
                epoch=epoch,
                batch=batch_index,
                label="main",
            )
            if exit_controller_params:
                _clip_grad_norm_or_report(
                    exit_controller_params,
                    trainer.grad_clip,
                    system=system,
                    epoch=epoch,
                    batch=batch_index,
                    label="exit-controller",
                )
                if collect_grad_diagnostics:
                    losses["grad_hierarchical_mmdit_exit_controller_post_clip"] = (
                        _parameter_grad_norm(
                            exit_controller_params,
                            reference=losses["loss"],
                        )
                    )
            if report_mem and memory_reporter.detail:
                memory_reporter.snapshot(
                    tag="train_after_clip",
                    epoch=epoch,
                    batch=batch_index,
                    global_step=global_step,
                    extra={"use_future": bool(use_future)},
                )
            optimizer.step()
            schedule.step()
            global_step += 1
            if report_mem:
                memory_reporter.snapshot(
                    tag="train_after_step",
                    epoch=epoch,
                    batch=batch_index,
                    global_step=global_step,
                    print_line=True,
                    extra={"use_future": bool(use_future)},
                )
            _accumulate_metric_tensors(
                metric_sums,
                losses,
                counts=metric_counts,
                grad=grad,
            )
            metric_count += 1
            if trainer.log_every and batch_index % trainer.log_every == 0:
                row = _sync_loss_row(losses, grad=grad)
                throughput_now = time.perf_counter()
                throughput_count = max(batch_index - throughput_batch, 1)
                seconds_per_batch = (throughput_now - throughput_start) / float(throughput_count)
                throughput_start = throughput_now
                throughput_batch = batch_index
                print(
                    _flow_jepa_stage1_serial_log_line(
                        row,
                        epoch=epoch,
                        batch_index=batch_index,
                        learning_rate=float(optimizer.param_groups[0]["lr"]),
                        seconds_per_batch=seconds_per_batch,
                    )
                    if flow_jepa_stage1
                    else _owned_serial_log_line(
                        row,
                        epoch=epoch,
                        batch_index=batch_index,
                        learning_rate=float(optimizer.param_groups[0]["lr"]),
                        seconds_per_batch=seconds_per_batch,
                    )
                    if clean_decoder is not None
                    else _evidence_serial_log_line(
                        row,
                        epoch=epoch,
                        batch_index=batch_index,
                        learning_rate=float(optimizer.param_groups[0]["lr"]),
                        seconds_per_batch=seconds_per_batch,
                    )
                    if evidence_decoder is not None
                    else (
                        f"[v39-layer] epoch={epoch:03d} batch={batch_index:04d} loss={row['loss']:.6f} "
                        f"pflow={row['physical_flow']:.6f} pflowu={row.get('physical_flow_uniform', row['physical_flow']):.6f} "
                        f"pfn={row.get('physical_flow_native', 0.0):.6f} pfnu={row.get('physical_flow_native_uniform', 0.0):.6f} "
                        f"afmd={row.get('arm_fm_per_dim', 0.0):.5f} gfmf={row.get('gripper_fm_field', 0.0):.5f} "
                        f"afmn={row.get('arm_fm_native', 0.0):.5f} afmnull={row.get('arm_fm_null', 0.0):.5f} "
                        f"afmnrms={row.get('arm_fm_null_rms', 0.0):.4f} afmnf={row.get('arm_fm_null_output_fraction', 0.0):.4f} "
                        f"afmproj={row.get('arm_fm_target_projection_error', 0.0):.2e} "
                        f"afmnoise={row.get('arm_fm_noise_projection_error', 0.0):.2e} "
                        f"anstd={row.get('arm_noise_abs_std', 0.0):.3f}/{row.get('arm_noise_delta_std', 0.0):.3f} "
                        f"atstd={row.get('arm_target_abs_std', 0.0):.3f}/{row.get('arm_target_delta_std', 0.0):.3f} "
                        f"asrc={row.get('arm_source_residual_rms', 0.0):.3f}/"
                        f"{row.get('arm_source_delta_rms', 0.0):.3f}/"
                        f"{row.get('arm_source_acceleration_rms', 0.0):.3f} "
                        f"asexp={row.get('arm_source_expected_rms', 0.0):.3f}/"
                        f"{row.get('arm_source_expected_delta_rms', 0.0):.3f}/"
                        f"{row.get('arm_source_expected_acceleration_rms', 0.0):.3f} "
                        f"asgeo={row.get('arm_source_covariance_effective_dimension', 0.0):.2f}/"
                        f"{row.get('arm_source_covariance_condition', 0.0):.1f} "
                        f"asfirst={row.get('arm_source_first_step_rms', 0.0):.3f}/"
                        f"{row.get('arm_source_expected_first_step_std', 0.0):.3f} "
                        f"astail={row.get('arm_source_expected_terminal_std', 0.0):.3f} "
                        f"gfar={row.get('gripper_arm_fm_ratio', 0.0):.3f} gfmv={row.get('gripper_fm_value', 0.0):.5f} gfmd={row.get('gripper_fm_delta', 0.0):.5f} "
                        f"gfme={row.get('gripper_fm_event', 0.0):.5f} gfmh={row.get('gripper_fm_hold', 0.0):.5f} "
                        f"gfmem={row.get('gripper_fm_event_loss_mass', 0.0):.3f} "
                        f"gfmew={row.get('gripper_fm_event_emphasis_mean', 0.0):.2f}/"
                        f"{row.get('gripper_fm_hold_emphasis_mean', 0.0):.2f} "
                        f"gfmn={row.get('gripper_fm_native', 0.0):.5f} gfmnull={row.get('gripper_fm_null', 0.0):.5f} "
                        f"gfmnrms={row.get('gripper_fm_null_rms', 0.0):.4f} gfmnf={row.get('gripper_fm_null_output_fraction', 0.0):.4f} "
                        f"hmchart={row.get('hierarchical_mmdit_native_time_chart_active', 0.0):.0f}/"
                        f"{row.get('hierarchical_mmdit_native_time_chart_complete', 0.0):.0f}/"
                        f"{row.get('hierarchical_mmdit_native_time_position_alignment', 0.0):.0f} "
                        f"hmtan={row.get('hierarchical_mmdit_velocity_arm_tangent_null_ratio', 0.0):.1e}/"
                        f"{row.get('hierarchical_mmdit_velocity_gripper_tangent_null_ratio', 0.0):.1e}/"
                        f"{row.get('hierarchical_mmdit_noisy_gripper_chart_null_ratio', 0.0):.1e} "
                        f"gfnehr={row.get('gripper_fm_null_event_hold_ratio', 0.0):.2f} "
                        f"gfmproj={row.get('gripper_fm_target_projection_error', 0.0):.2e} "
                        f"gfmer={row.get('gripper_fm_target_energy_ratio', 0.0):.3f} "
                        f"decode={row['decoded_action']:.6f} rollout={row.get('rollout_dynamics', 0.0):.6f} "
                        f"rvar={row.get('rollout_variance', 0.0):.4f} rnorm={row.get('rollout_norm', 0.0):.4f} "
                        f"rstep={row.get('rollout_milestone_delta_match', 0.0):.4f} "
                        f"first8={row.get('first8_physical_flow', 0.0):.6f} tail={row.get('tail_physical_flow', 0.0):.6f} "
                        f"delta={row.get('rollout_delta', 0.0):.6f} contrast={row.get('rollout_contrast', 0.0):.6f} "
                        f"d_shuffle={row.get('rollout_delta_shuffle', 0.0):.6f} "
                        f"dshuf_src={row.get('rollout_effect_change_shuffle', 0.0):.3e}/"
                        f"{row.get('rollout_delta_state_shuffle', 0.0):.3e}/"
                        f"{row.get('rollout_effect_change_state_shuffle', 0.0):.3e} "
                        f"rbase={row.get('rollout_base_norm', 0.0):.3f} "
                        f"rexp={row.get('rollout_decomposition_expansion_ratio', 0.0):.3f} "
                        f"rcancel={row.get('rollout_shuffle_cancellation_fraction', 0.0):.3f} "
                        f"rbleak={row.get('rollout_base_change_shuffle', 0.0):.2e} "
                        f"stdr={row.get('rollout_pred_std_ratio', 0.0):.4f} dnratio={row.get('rollout_milestone_delta_norm_ratio', 0.0):.4f} "
                        f"rdeep={row.get('rollout_deep_update_norm', 0.0):.2f} "
                        f"rdnorm={row.get('rollout_deep_token_norm', 0.0):.2f} "
                        f"event={row['event']:.6f} "
                        f"ev={row.get('evidence_condition_scan_norm', 0.0):.2f}/"
                        f"{row.get('evidence_condition_lateral_norm', 0.0):.2f}/"
                        f"{row.get('evidence_condition_norm', 0.0):.2f}/"
                        f"{row.get('evidence_latent_norm', 0.0):.2f} "
                        f"evwrite={row.get('evidence_mmd_it_self_update_norm', 0.0):.3f}/"
                        f"{row.get('evidence_mmd_it_evidence_update_norm', 0.0):.3f}/"
                        f"{row.get('evidence_mmd_it_ffn_update_norm', 0.0):.3f} "
                        f"evstate={row.get('evidence_mmd_it_action_state_scale', 1.0):.2f}/"
                        f"{row.get('evidence_mmd_it_action_state_token_norm', 0.0):.2f} "
                        f"evscale={row.get('evidence_mmd_it_evidence_scale', 1.0):.2f}/"
                        f"{row.get('evidence_mmd_it_action_state_scale', 1.0):.2f} "
                        f"evgate={row.get('evidence_mmd_it_residual_gate_mean', 0.0):.3f}/"
                        f"{row.get('evidence_mmd_it_ffn_gate_mean', 0.0):.3f} "
                        f"evupd={row.get('evidence_mmd_it_block_0_update_norm', 0.0):.3f}/"
                        f"{row.get('evidence_mmd_it_block_1_update_norm', 0.0):.3f}/"
                        f"{row.get('evidence_mmd_it_block_2_update_norm', 0.0):.3f} "
                        f"evexec={row.get('evidence_mmd_it_execution_progress', 0.0):.2f}/"
                        f"{row.get('evidence_mmd_it_capacity_ratio', 1.0):.5f}/"
                        f"{row.get('evidence_mmd_it_dwell_expected', 1.0):.2f}/"
                        f"{row.get('evidence_mmd_it_execution_cost', 0.0):.3f} "
                        f"evroute={row.get('evidence_mmd_it_dynamic_route_next_fraction', 0.0):.3f}/"
                        f"{row.get('evidence_mmd_it_hard_route_next_fraction', 0.0):.3f} "
                        f"evdwell={row.get('evidence_mmd_it_dwell_expected', 1.0):.3f}/"
                        f"{row.get('evidence_mmd_it_hard_dwell_expected', 1.0):.3f} "
                        f"evctrl={row.get('evidence_mmd_it_controller_slot_pair_cosine', 0.0):+.2f}/"
                        f"{row.get('evidence_mmd_it_controller_slot_common_mode_ratio', 0.0):.2f}/"
                        f"{row.get('evidence_mmd_it_controller_slot_private_energy_ratio', 0.0):.2f} "
                        f"evval={row.get('evidence_mmd_it_execution_value_loss', 0.0):.4f}/"
                        f"{row.get('evidence_mmd_it_execution_value_target_spread', 0.0):.4f}/"
                        f"{row.get('evidence_mmd_it_execution_value_predicted_spread', 0.0):.4f}/"
                        f"{row.get('evidence_mmd_it_execution_value_decision_accuracy', 0.0):.2f}/"
                        f"{row.get('evidence_mmd_it_execution_value_common_mode_ratio', 0.0):.2f} "
                        f"evcap={row.get('evidence_mmd_it_effective_depth', 0.0):.3f}/"
                        f"{row.get('evidence_mmd_it_removed_channel_fraction', 0.0):.5f}/"
                        f"{row.get('evidence_mmd_it_nonexpansive_violation', 0.0):.1e} "
                        f"evsel={row.get('evidence_mmd_it_execution_selection_entropy', 0.0):.3f}/"
                        f"{row.get('evidence_mmd_it_learned_selection_entropy', 0.0):.3f}/"
                        f"{row.get('evidence_mmd_it_execution_selection_max_probability', 1.0):.3f} "
                        f"evgrad={row.get('grad_evidence_view_adapter', 0.0):.2e}/"
                        f"{row.get('grad_evidence_condition_organizer', 0.0):.2e}/"
                        f"{row.get('grad_evidence_mmdit_evidence_reader', 0.0):.2e}/"
                        f"{row.get('grad_evidence_mmdit_action_state', 0.0):.2e}/"
                        f"{row.get('grad_evidence_mmdit_blocks', 0.0):.2e} "
                        f"evcgrad={row.get('grad_evidence_mmdit_execution_controller', 0.0):.2e}/"
                        f"{row.get('grad_evidence_mmdit_operator_basis', 0.0):.2e}/"
                        f"{row.get('grad_evidence_mmdit_execution_value_reader', 0.0):.2e} "
                        f"cz={row.get('latent_cvae_prior_z_norm', row.get('latent_cvae_z_norm', 0.0)):.2f} "
                        f"cpz={row.get('latent_cvae_post_z_norm', 0.0):.2f} "
                        f"cmug={row.get('latent_cvae_mu_gap', 0.0):.2f} "
                        f"ckl={row.get('latent_cvae_kl', 0.0):.4f} "
                        f"cpflow={row.get('latent_cvae_post_flow', 0.0):.4f} "
                        f"cstd={row.get('latent_cvae_prior_std', 0.0):.3f} "
                        f"cgate={row.get('latent_cvae_gripper_gate_mean', 0.0):.3f} "
                        f"clmem={row.get('latent_cvae_layer_memory_count', 0.0):.1f} "
                        f"cscan={row.get('latent_cvae_condition_scan_norm', 0.0):.2f} "
                        f"clat={row.get('latent_cvae_condition_lateral_norm', 0.0):.2f} "
                        f"craw={row.get('latent_cvae_condition_raw_norm', 0.0):.2f} "
                        f"zcond={row.get('latent_cvae_primary_condition_norm', 0.0):.2f} "
                        f"zfx={row.get('latent_cvae_primary_z_effect_norm', 0.0):.2f} "
                        f"clsum={row.get('latent_cvae_layer_summary_norm', 0.0):.2f} "
                        f"ctraw={row.get('latent_cvae_transition_source_raw_norm', 0.0):.2f} "
                        f"ctmem={row.get('latent_cvae_transition_condition_norm', 0.0):.2f} "
                        f"ccscale={row.get('latent_cvae_consequence_scale_mean', 0.0):.3f} "
                        f"ccpref={row.get('latent_cvae_consequence_gate_preference', 0.0):.3f} "
                        f"ccmix={row.get('latent_cvae_consequence_mix_ratio', 0.0):.3f} "
                        f"cadu={row.get('latent_cvae_adaptive_refine_update_mean', 0.0):.3f} "
                        f"cxgate={row.get('latent_cvae_adaptive_noisy_gate_mean', 0.0):.3f} "
                        f"xnorm={row.get('latent_cvae_adaptive_noisy_branch_norm', 0.0):.3f} "
                        f"volpar={row.get('mmdit_noisy_volume_parity', 0.0):.3f} "
                        f"xinfl={row.get('mmdit_noisy_influence_ratio', 0.0):.2f} "
                        f"crmax={row.get('latent_cvae_adaptive_route_max', 0.0):.3f} "
                        f"crent={row.get('latent_cvae_adaptive_route_entropy', 0.0):.3f} "
                        f"creff={row.get('latent_cvae_adaptive_route_effective_slots', 0.0):.2f} "
                        f"cprmax={row.get('latent_cvae_adaptive_progress_max', 0.0):.3f} "
                        f"cprent={row.get('latent_cvae_adaptive_progress_entropy', 0.0):.3f} "
                        f"cpeff={row.get('latent_cvae_adaptive_progress_effective_slots', 0.0):.2f} "
                        f"cprog={row.get('latent_cvae_adaptive_progress_norm', 0.0):.2f} "
                        f"ccont={row.get('latent_cvae_adaptive_continue_mean', 0.0):.3f} "
                        f"cprefix={row.get('latent_cvae_adaptive_prefix_norm', 0.0):.2f} "
                        f"czseed={row.get('latent_cvae_adaptive_progress_seed_norm', 0.0):.3f} "
                        f"czseff={row.get('latent_cvae_adaptive_progress_seed_effective_slots', 0.0):.2f} "
                        f"ctemp={row.get('latent_cvae_adaptive_route_temperature_mean', 0.0):.2f} "
                        f"rtime={row.get('latent_cvae_adaptive_route_time_query_norm', 0.0):.3f} "
                        f"cfunc={row.get('latent_cvae_adaptive_function_delta_norm', 0.0):.3f} "
                        f"cbasehf={row.get('latent_cvae_adaptive_base_highfreq_norm', 0.0):.3f} "
                        f"cstep={row.get('latent_cvae_adaptive_refine_step_bias_norm', 0.0):.3f} "
                        f"ccmax={row.get('latent_cvae_adaptive_capsule_layer_max', 0.0):.3f} "
                        f"ccleff={row.get('latent_cvae_adaptive_capsule_layer_effective_slots', 0.0):.2f} "
                        f"cstr={row.get('latent_cvae_adaptive_condition_strength_mean', 0.0):.3f} "
                        f"ccond={row.get('latent_cvae_adaptive_condition_residual_norm', 0.0):.3f} "
                        f"mdu={row.get('latent_cvae_mmdit_action_update_norm', 0.0):.3f} "
                        f"mdcu={row.get('latent_cvae_mmdit_cond_update_norm', 0.0):.3f} "
                        f"mdca={row.get('latent_cvae_mmdit_action_cond_attention', 0.0):.3f} "
                        f"mdna={row.get('latent_cvae_mmdit_action_noisy_attention', 0.0):.3f} "
                        f"mdat={row.get('latent_cvae_mmdit_action_token_norm', 0.0):.2f} "
                        f"mdct={row.get('latent_cvae_mmdit_condition_token_norm', 0.0):.2f} "
                        f"mdnt={row.get('latent_cvae_mmdit_noisy_token_norm', 0.0):.2f} "
                        f"mdwa={row.get('latent_cvae_mmdit_action_workspace_attention', 0.0):.3f} "
                        f"mdwe={row.get('latent_cvae_mmdit_action_workspace_enrichment', 0.0):.3f} "
                        f"mdla={row.get('latent_cvae_mmdit_action_low_attention', 0.0):.3f} "
                        f"mdsa={row.get('latent_cvae_mmdit_action_stage_attention', 0.0):.3f} "
                        f"mdle={row.get('latent_cvae_mmdit_action_low_enrichment', 0.0):.3f} "
                        f"mdse={row.get('latent_cvae_mmdit_action_stage_enrichment', 0.0):.3f} "
                        f"mdnaT={row.get('latent_cvae_mmdit_noisy_attn_t0_sum', 0.0) / max(row.get('latent_cvae_mmdit_attn_t0_count', 0.0), 1e-6):.3f}"
                        f"/{row.get('latent_cvae_mmdit_noisy_attn_t1_sum', 0.0) / max(row.get('latent_cvae_mmdit_attn_t1_count', 0.0), 1e-6):.3f}"
                        f"/{row.get('latent_cvae_mmdit_noisy_attn_t2_sum', 0.0) / max(row.get('latent_cvae_mmdit_attn_t2_count', 0.0), 1e-6):.3f} "
                        f"mdwaT={row.get('latent_cvae_mmdit_workspace_attn_t0_sum', 0.0) / max(row.get('latent_cvae_mmdit_attn_t0_count', 0.0), 1e-6):.3f}"
                        f"/{row.get('latent_cvae_mmdit_workspace_attn_t1_sum', 0.0) / max(row.get('latent_cvae_mmdit_attn_t1_count', 0.0), 1e-6):.3f}"
                        f"/{row.get('latent_cvae_mmdit_workspace_attn_t2_sum', 0.0) / max(row.get('latent_cvae_mmdit_attn_t2_count', 0.0), 1e-6):.3f} "
                        f"mdlaT={row.get('latent_cvae_mmdit_low_attn_t0_sum', 0.0) / max(row.get('latent_cvae_mmdit_attn_t0_count', 0.0), 1e-6):.3f}"
                        f"/{row.get('latent_cvae_mmdit_low_attn_t1_sum', 0.0) / max(row.get('latent_cvae_mmdit_attn_t1_count', 0.0), 1e-6):.3f}"
                        f"/{row.get('latent_cvae_mmdit_low_attn_t2_sum', 0.0) / max(row.get('latent_cvae_mmdit_attn_t2_count', 0.0), 1e-6):.3f} "
                        f"mdsaT={row.get('latent_cvae_mmdit_stage_attn_t0_sum', 0.0) / max(row.get('latent_cvae_mmdit_attn_t0_count', 0.0), 1e-6):.3f}"
                        f"/{row.get('latent_cvae_mmdit_stage_attn_t1_sum', 0.0) / max(row.get('latent_cvae_mmdit_attn_t1_count', 0.0), 1e-6):.3f}"
                        f"/{row.get('latent_cvae_mmdit_stage_attn_t2_sum', 0.0) / max(row.get('latent_cvae_mmdit_attn_t2_count', 0.0), 1e-6):.3f} "
                        f"mdra={row.get('latent_cvae_mmdit_action_rollout_attention', 0.0):.3f} "
                        f"mdre={row.get('latent_cvae_mmdit_action_rollout_enrichment', 0.0):.3f} "
                        f"mdrn={row.get('latent_cvae_rollout_token_norm', 0.0):.2f} "
                        f"mdrc={row.get('latent_cvae_rollout_token_count', 0.0):.0f} "
                        f"wk={row.get('latent_cvae_workspace_token_count', 0.0):.0f} "
                        f"wtok={row.get('latent_cvae_workspace_token_norm', 0.0):.2f} "
                        f"wdelta={row.get('latent_cvae_workspace_update_norm', 0.0):.2f} "
                        f"wgstate={row.get('latent_cvae_workspace_global_state_norm', 0.0):.2f} "
                        f"wgslot={row.get('latent_cvae_workspace_global_slot_delta_norm', 0.0):.3f} "
                        f"wsdiv={row.get('latent_cvae_workspace_global_slot_diversity', 0.0):.3f} "
                        f"wsrc={row.get('latent_cvae_workspace_source_count', 0.0):.0f} "
                        f"wcache={row.get('latent_cvae_workspace_cached_token_fraction', 0.0):.3f} "
                        f"wqscale={row.get('latent_cvae_workspace_noisy_query_scale', 0.0):.3f} "
                        f"went={row.get('latent_cvae_workspace_attention_entropy', 0.0):.3f} "
                        f"wmax={row.get('latent_cvae_workspace_attention_max', 0.0):.3f} "
                        f"wgent={row.get('latent_cvae_workspace_group_attention_entropy', 0.0):.3f} "
                        f"wgeff={row.get('latent_cvae_workspace_group_effective_sources', 0.0):.2f} "
                        f"wmass={row.get('latent_cvae_workspace_attention_mass_error', 0.0):.1e} "
                        f"wupd={row.get('latent_cvae_workspace_action_update_ratio', 0.0):.3f} "
                        f"wprog={row.get('latent_cvae_workspace_progress_attention', 0.0):.3f} "
                        f"wpq={row.get('latent_cvae_workspace_progress_query_norm', 0.0):.3f} "
                        f"wpupd={row.get('latent_cvae_workspace_progress_update_norm', 0.0):.3f} "
                        f"wpact={row.get('latent_cvae_workspace_progress_action_dependence', 0.0):.3f} "
                        f"rstem={row.get('latent_cvae_legacy_stem_effect_ratio', 0.0):.3f} "
                        f"zzero={row.get('latent_cvae_z_zero_delta', 0.0):.3f} "
                        f"zshuf={row.get('latent_cvae_z_shuffle_delta', 0.0):.3f} "
                        f"wscan={row.get('latent_cvae_workspace_scan_attention', 0.0):.3f} "
                        f"wlat={row.get('latent_cvae_workspace_lateral_attention', 0.0):.3f} "
                        f"wtrans={row.get('latent_cvae_workspace_transition_total_attention', row.get('latent_cvae_workspace_transition_attention', 0.0)):.3f} "
                        f"wtraj={row.get('latent_cvae_workspace_trajectory_attention', 0.0):.3f} "
                        f"wroll={row.get('latent_cvae_workspace_rollout_attention', 0.0):.3f} "
                        f"wcaps={row.get('latent_cvae_workspace_capsule_attention', 0.0):.3f} "
                        f"wroute={row.get('latent_cvae_workspace_routed_layer_attention', 0.0):.3f} "
                        f"wgeom={row.get('latent_cvae_workspace_role_geom_attention', 0.0):.3f} "
                        f"wtrn={row.get('latent_cvae_workspace_role_transition_attention', 0.0):.3f} "
                        f"wevt={row.get('latent_cvae_workspace_role_event_attention', 0.0):.3f} "
                        f"wstate={row.get('latent_cvae_workspace_role_state_attention', 0.0):.3f} "
                        f"wrlay={row.get('latent_cvae_workspace_role_layer_attention', 0.0):.3f} "
                        f"wglob={row.get('latent_cvae_workspace_role_global_attention', 0.0):.3f} "
                        f"ctrlcap={row.get('latent_cvae_workspace_controller_capacity', 0.0):.3f} "
                        f"ctrldly={row.get('latent_cvae_workspace_controller_delay', 0.0):.3f} "
                        f"ctrlent={row.get('latent_cvae_workspace_controller_role_entropy', 0.0):.3f} "
                        f"ctrlq={row.get('latent_cvae_workspace_controller_query_delta_norm', 0.0):.3f} "
                        f"hlow={row.get('latent_cvae_hierarchical_low_token_count', 0.0):.0f}/{row.get('latent_cvae_hierarchical_low_token_norm', 0.0):.2f} "
                        f"hlsel={row.get('latent_cvae_hierarchical_low_selector_stage_effective_slots', 0.0):.2f} "
                        f"hsrole={row.get('latent_cvae_hierarchical_stage_token_count', 0.0):.0f}/{row.get('latent_cvae_hierarchical_stage_role_norm', 0.0):.2f} "
                        f"hscont={row.get('latent_cvae_hierarchical_stage_content_norm', 0.0):.2f} "
                        f"hsrdiv={row.get('latent_cvae_hierarchical_stage_role_diversity', 0.0):.2f} "
                        f"hscdiv={row.get('latent_cvae_hierarchical_stage_content_diversity', 0.0):.2f} "
                        f"hsrcos={row.get('latent_cvae_hierarchical_stage_role_content_cosine', 0.0):.3f} "
                        f"hsrfrac={row.get('latent_cvae_hierarchical_stage_role_output_fraction', 0.0):.3f} "
                        f"hsupd={row.get('latent_cvae_hierarchical_stage_update_norm', 0.0):.3f} "
                        f"hsret={row.get('latent_cvae_hierarchical_stage_retain_mean', 0.0):.3f} "
                        f"hsprom={row.get('latent_cvae_hierarchical_stage_promote_scale', 0.0):.3f} "
                        f"hmrole={row.get('latent_cvae_hierarchical_manager_role_entropy', 0.0):.3f} "
                        f"hmprom={row.get('latent_cvae_hierarchical_manager_promote_gate', 0.0):.3f} "
                        f"hmlow={row.get('latent_cvae_hierarchical_manager_low_output_strength', 0.0):.3f} "
                        f"hmstage={row.get('latent_cvae_hierarchical_manager_stage_output_strength', 0.0):.3f} "
                        # V72 (S5 cleanup): dead cm* micro-controller console keys
                        # removed -- micro is config-off AND structurally excluded
                        # under mmdit_refine; the loss-dict keys remain intact for
                        # legacy non-MMDiT configs.
                        f"careg={row.get('latent_cvae_adaptive_regularizer', 0.0):.4f} "
                        f"carent={row.get('latent_cvae_adaptive_route_entropy_regularizer', 0.0):.4f} "
                        f"czbase={row.get('consequence_zero_base_shift', 0.0):.3f} "
                        f"csc={row.get('consequence_self_condition', 0.0):.0f} "
                        f"cscmse={row.get('consequence_self_condition_target_mse', 0.0):.4f} "
                        f"cscnmse={row.get('consequence_self_condition_noisy_mse', 0.0):.4f} "
                        f"cspflow={row.get('consequence_preview_flow', 0.0):.4f} "
                        f"iglob={row.get('intent_global_norm', 0.0):.2f} "
                        f"istage={row.get('intent_stage_contract_norm', 0.0):.2f} "
                        f"iread={row.get('intent_read_contract_norm', 0.0):.2f} "
                        f"icgs={row.get('intent_global_stage_cosine', 0.0):.3f} "
                        f"icgr={row.get('intent_global_read_cosine', 0.0):.3f} "
                        f"icsr={row.get('intent_stage_read_cosine', 0.0):.3f} "
                        f"hmdu={row.get('hierarchical_mmdit_action_update_norm', 0.0):.3f} "
                        f"hmur={row.get('hierarchical_mmdit_action_update_ratio', 0.0):.3f} "
                        f"hmcan={row.get('hierarchical_mmdit_action_serial_cancellation_fraction', 0.0):.3f} "
                        f"hmorth={row.get('hierarchical_mmdit_action_serial_cancellation_orthogonal_baseline', 0.0):.3f} "
                        f"hmxcan={row.get('hierarchical_mmdit_action_serial_cancellation_excess', 0.0):+.3f} "
                        f"hmbdot={row.get('hierarchical_mmdit_action_branch_weighted_cosine', 0.0):+.3f} "
                        f"hmcos={row.get('hierarchical_mmdit_action_state_cosine', 0.0):.3f} "
                        f"hmnu={row.get('hierarchical_mmdit_action_noisy_update_norm', 0.0):.3f} "
                        f"hmsu={row.get('hierarchical_mmdit_action_stage_update_norm', 0.0):.3f} "
                        f"hmlu={row.get('hierarchical_mmdit_action_low_update_norm', 0.0):.3f} "
                        f"hmnf={row.get('hierarchical_mmdit_action_noisy_update_fraction', 0.0):.3f} "
                        f"hmsf={row.get('hierarchical_mmdit_action_stage_update_fraction', 0.0):.3f} "
                        f"hmlf={row.get('hierarchical_mmdit_action_low_update_fraction', 0.0):.3f} "
                        f"hmdepth={row.get('hierarchical_mmdit_action_noisy_depth_ratio', 0.0):.2f}/"
                        f"{row.get('hierarchical_mmdit_action_stage_depth_ratio', 0.0):.2f}/"
                        f"{row.get('hierarchical_mmdit_action_low_depth_ratio', 0.0):.2f} "
                        f"hmraw={row.get('hierarchical_mmdit_action_noisy_raw_depth_ratio', 0.0):.2f}/"
                        f"{row.get('hierarchical_mmdit_action_stage_raw_depth_ratio', 0.0):.2f}/"
                        f"{row.get('hierarchical_mmdit_action_low_raw_depth_ratio', 0.0):.2f} "
                        f"hmkeep={row.get('hierarchical_mmdit_action_self_update_keep', 0.0):.2f}/"
                        f"{row.get('hierarchical_mmdit_action_noisy_update_keep', 0.0):.2f}/"
                        f"{row.get('hierarchical_mmdit_action_stage_update_keep', 0.0):.2f}/"
                        f"{row.get('hierarchical_mmdit_action_low_update_keep', 0.0):.2f}/"
                        f"{row.get('hierarchical_mmdit_action_ffn_update_keep', 0.0):.2f} "
                        f"hmedepth={row.get('hierarchical_mmdit_action_noisy_effective_depth', 0.0):.1f}/"
                        f"{row.get('hierarchical_mmdit_action_stage_effective_depth', 0.0):.1f}/"
                        f"{row.get('hierarchical_mmdit_action_low_effective_depth', 0.0):.1f} "
                        f"hmcontract={row.get('hierarchical_mmdit_action_noisy_contraction_ratio', 0.0):.3f}/"
                        f"{row.get('hierarchical_mmdit_action_stage_contraction_ratio', 0.0):.3f}/"
                        f"{row.get('hierarchical_mmdit_action_low_contraction_ratio', 0.0):.3f} "
                        f"hmhost={row.get('hierarchical_mmdit_action_noisy_host_update_rms', 0.0):.3f}/"
                        f"{row.get('hierarchical_mmdit_action_stage_host_update_rms', 0.0):.3f}/"
                        f"{row.get('hierarchical_mmdit_action_low_host_update_rms', 0.0):.3f} "
                        f"hmcover={row.get('hierarchical_mmdit_action_noisy_subspace_energy_fraction', 0.0):.3f}/"
                        f"{row.get('hierarchical_mmdit_action_stage_subspace_energy_fraction', 0.0):.3f}/"
                        f"{row.get('hierarchical_mmdit_action_low_subspace_energy_fraction', 0.0):.3f} "
                        f"hmremove={row.get('hierarchical_mmdit_action_noisy_removed_fraction', 0.0):.3f}/"
                        f"{row.get('hierarchical_mmdit_action_stage_removed_fraction', 0.0):.3f}/"
                        f"{row.get('hierarchical_mmdit_action_low_removed_fraction', 0.0):.3f} "
                        f"hmdepthreg={row.get('hierarchical_mmdit_operator_contraction_progress', 0.0):.2f}/"
                        f"{row.get('hierarchical_mmdit_depth_usage_regularizer', 0.0):.4f} "
                        f"hmdwell={row.get('hierarchical_mmdit_learned_execution_active', 0.0):.0f}/"
                        f"{row.get('hierarchical_mmdit_value_dwell_warmup_active', 0.0):.0f}/"
                        f"{row.get('hierarchical_mmdit_value_dwell_shadow_active', 0.0):.0f}/"
                        f"{row.get('hierarchical_mmdit_operation_decision_shadow_active', 0.0):.0f} "
                        f"hmval={row.get('hierarchical_mmdit_operation_value_loss', 0.0):.4f}/"
                        f"{row.get('hierarchical_mmdit_operation_value_target_spread', 0.0):.4f}/"
                        f"{row.get('hierarchical_mmdit_operation_value_predicted_spread', 0.0):.4f}/"
                        f"{row.get('hierarchical_mmdit_operation_value_correlation', 0.0):+.2f}/"
                        f"{row.get('hierarchical_mmdit_operation_value_decision_accuracy', 0.0):.2f} "
                        f"hmread={row.get('hierarchical_mmdit_controller_reader_operator_memory_attention', 0.0):.2f}/"
                        f"{row.get('hierarchical_mmdit_controller_reader_spectral_memory_attention', 0.0):.2f} "
                        f"hmmem={row.get('hierarchical_mmdit_controller_reader_operator_global_memory_attention', 0.0):.2f}/"
                        f"{row.get('hierarchical_mmdit_controller_reader_operator_private_memory_attention', 0.0):.2f} "
                        f"hmrdiv={row.get('hierarchical_mmdit_controller_reader_operator_attention_diversity', 0.0):.3f}/"
                        f"{row.get('hierarchical_mmdit_controller_reader_spectral_attention_local_change', 0.0):.3f}/"
                        f"{row.get('hierarchical_mmdit_controller_reader_family_attention_diversity', 0.0):.3f} "
                        f"hmfunc={row.get('hierarchical_mmdit_controller_operator_representation_diversity', 0.0):.3f}/"
                        f"{row.get('hierarchical_mmdit_controller_spectral_representation_local_change', 0.0):.3f}/"
                        f"{row.get('hierarchical_mmdit_controller_state_centered_energy_ratio', 0.0):.3f} "
                        f"hmfcand={row.get('hierarchical_mmdit_function_candidate_cosine', 0.0):+.2f}/"
                        f"{row.get('hierarchical_mmdit_function_candidate_diversity', 0.0):.2f}/"
                        f"{row.get('hierarchical_mmdit_function_candidate_update_rms', 0.0):.3f}/"
                        f"{row.get('hierarchical_mmdit_function_candidate_update_spread', 0.0):.3f}/"
                        f"{row.get('hierarchical_mmdit_function_candidate_valid_count', 0.0):.2f} "
                        f"hmpriv={row.get('hierarchical_mmdit_controller_private_pair_cosine', 0.0):+.2f}/"
                        f"{row.get('hierarchical_mmdit_controller_private_centered_energy_ratio', 0.0):.3f}/"
                        f"{row.get('hierarchical_mmdit_controller_private_global_energy_ratio', 0.0):.3f}/"
                        f"{row.get('hierarchical_mmdit_controller_private_residual_value_rms', 0.0):.3f} "
                        f"hmcomp={row.get('hierarchical_mmdit_controller_competition_source_effective_slots', 0.0):.2f}/"
                        f"{row.get('hierarchical_mmdit_controller_competition_source_owner_max', 0.0):.2f}/"
                        f"{row.get('hierarchical_mmdit_controller_competition_slot_load_effective', 0.0):.2f}/"
                        f"{row.get('hierarchical_mmdit_controller_competition_slot_load_max', 0.0):.2f} "
                        f"hmcap={row.get('hierarchical_mmdit_controller_operator_raw_depth_mean', 0.0):.3f}/"
                        f"{row.get('hierarchical_mmdit_controller_operator_depth_stage_std', 0.0):.3f} "
                        f"hmoracle={row.get('hierarchical_mmdit_oracle_route_loss', 0.0):.3f}/"
                        f"{row.get('hierarchical_mmdit_oracle_route_weight', 0.0):.3f} "
                        f"hmodepth={row.get('hierarchical_mmdit_oracle_target_depth', 0.0):.2f}/"
                        f"{row.get('hierarchical_mmdit_oracle_predicted_depth', 0.0):.2f}/"
                        f"{row.get('hierarchical_mmdit_oracle_depth_accuracy', 0.0):.2f}/"
                        f"{row.get('hierarchical_mmdit_oracle_predicted_regret', 0.0):.4f} "
                        f"hmstage={_format_hierarchical_stage_usage(row)} "
                        f"hmblock={_format_hierarchical_block_usage(row)} "
                        f"hmopgain={row.get('hierarchical_mmdit_action_noisy_operator_gain', 0.0):.3f}/"
                        f"{row.get('hierarchical_mmdit_action_stage_operator_gain', 0.0):.3f}/"
                        f"{row.get('hierarchical_mmdit_action_low_operator_gain', 0.0):.3f} "
                        f"hmdir={row.get('hierarchical_mmdit_action_noisy_direction_change', 0.0):.3f}/"
                        f"{row.get('hierarchical_mmdit_action_stage_direction_change', 0.0):.3f}/"
                        f"{row.get('hierarchical_mmdit_action_low_direction_change', 0.0):.3f} "
                        f"hmbcos2={row.get('hierarchical_mmdit_action_noisy_direction_cosine', 0.0):+.2f}/"
                        f"{row.get('hierarchical_mmdit_action_stage_direction_cosine', 0.0):+.2f}/"
                        f"{row.get('hierarchical_mmdit_action_low_direction_cosine', 0.0):+.2f} "
                        f"hmbgain={row.get('hierarchical_mmdit_action_noisy_base_data_gain', 0.0):.2f}/"
                        f"{row.get('hierarchical_mmdit_action_stage_base_data_gain', 0.0):.2f}/"
                        f"{row.get('hierarchical_mmdit_action_low_base_data_gain', 0.0):.2f} "
                        f"hmbprm={row.get('hierarchical_mmdit_action_noisy_base_parameter_rms', 0.0):.2f}/"
                        f"{row.get('hierarchical_mmdit_action_stage_base_parameter_rms', 0.0):.2f}/"
                        f"{row.get('hierarchical_mmdit_action_low_base_parameter_rms', 0.0):.2f} "
                        f"hmgate={row.get('hierarchical_mmdit_action_self_base_gate', 0.0):.3f}/"
                        f"{row.get('hierarchical_mmdit_action_noisy_base_gate', 0.0):.3f}/"
                        f"{row.get('hierarchical_mmdit_action_stage_base_gate', 0.0):.3f}/"
                        f"{row.get('hierarchical_mmdit_action_low_base_gate', 0.0):.3f}/"
                        f"{row.get('hierarchical_mmdit_action_ffn_base_gate', 0.0):.3f} "
                        f"hmegate={row.get('hierarchical_mmdit_action_self_effective_gate', 0.0):.3f}/"
                        f"{row.get('hierarchical_mmdit_action_noisy_effective_gate', 0.0):.3f}/"
                        f"{row.get('hierarchical_mmdit_action_stage_effective_gate', 0.0):.3f}/"
                        f"{row.get('hierarchical_mmdit_action_low_effective_gate', 0.0):.3f}/"
                        f"{row.get('hierarchical_mmdit_action_ffn_effective_gate', 0.0):.3f} "
                        f"hmkerr={row.get('hierarchical_mmdit_action_noisy_keep_scale_error', 0.0):.1e}/"
                        f"{row.get('hierarchical_mmdit_action_stage_keep_scale_error', 0.0):.1e}/"
                        f"{row.get('hierarchical_mmdit_action_low_keep_scale_error', 0.0):.1e} "
                        f"hmgerr={max(row.get(f'hierarchical_mmdit_action_{name}_gate_scale_error', 0.0) for name in ('self', 'noisy', 'stage', 'low', 'ffn')):.1e} "
                        f"hmnrms={row.get('hierarchical_mmdit_action_pre_norm_rms', 0.0):.3f}/"
                        f"{row.get('hierarchical_mmdit_action_post_norm_rms', 0.0):.3f} "
                        f"hmbound={max(row.get(f'hierarchical_mmdit_action_{name}_boundary_identity_error', 0.0) for name in ('self', 'noisy', 'stage', 'low', 'ffn')):.1e} "
                        f"hmnexp={max(row.get(f'hierarchical_mmdit_action_{name}_nonexpansive_violation', 0.0) for name in ('self', 'noisy', 'stage', 'low', 'ffn')):.1e} "
                        f"hmnest={max(row.get(f'hierarchical_mmdit_action_{name}_nested_order_violation', 0.0) for name in ('self', 'noisy', 'stage', 'low', 'ffn')):.1e} "
                        f"hmbasis={max(row.get(f'hierarchical_mmdit_action_{name}_basis_norm_error', 0.0) for name in ('self', 'noisy', 'stage', 'low', 'ffn')):.1e}/"
                        f"{max(row.get(f'hierarchical_mmdit_action_{name}_basis_orthogonality_error', 0.0) for name in ('self', 'noisy', 'stage', 'low', 'ffn')):.1e} "
                        f"hexh={row.get('hierarchical_mmdit_executed_steps', 0.0):.2f}/"
                        f"{row.get('hierarchical_mmdit_action_response_rel', 0.0):.3f}/"
                        f"{row.get('hierarchical_mmdit_stage_pressure_rel', 0.0):.3f}/"
                        f"{row.get('hierarchical_mmdit_refine_gain', 0.0):+.4f}/"
                        f"{row.get('hierarchical_mmdit_response_gain_corr', 0.0):+.2f}/"
                        f"{row.get('hierarchical_mmdit_unresolved_rate', 0.0):.2f}/"
                        f"{row.get('hierarchical_mmdit_budget_exhausted_rate', 0.0):.2f}/"
                        f"{row.get('hierarchical_mmdit_final_block', 0.0):.2f}/"
                        f"{row.get('hierarchical_mmdit_final_stage', 0.0):.2f}/"
                        f"{row.get('hierarchical_mmdit_early_exit_rate', 0.0):.2f}/"
                        f"{row.get('hierarchical_mmdit_block_advance_rate', 0.0):.2f}/"
                        f"{row.get('hierarchical_mmdit_stage_advance_rate', 0.0):.2f} "
                        f"hmuresp={row.get('hierarchical_mmdit_action_response_arm', 0.0):.3f}/"
                        f"{row.get('hierarchical_mmdit_action_response_gripper', 0.0):.3f}/"
                        f"{row.get('hierarchical_mmdit_action_response_arm_null', 0.0):.3f}/"
                        f"{row.get('hierarchical_mmdit_action_response_gripper_null', 0.0):.3f} "
                        f"hmuq={row.get('hierarchical_mmdit_action_response_p25', 0.0):.3f}/"
                        f"{row.get('hierarchical_mmdit_action_response_p50', 0.0):.3f}/"
                        f"{row.get('hierarchical_mmdit_action_response_p75', 0.0):.3f} "
                        f"hmpq={row.get('hierarchical_mmdit_stage_pressure_p25', 0.0):.3f}/"
                        f"{row.get('hierarchical_mmdit_stage_pressure_p50', 0.0):.3f}/"
                        f"{row.get('hierarchical_mmdit_stage_pressure_p75', 0.0):.3f} "
                        f"hmuT50={row.get('hierarchical_mmdit_action_response_t0_p50', 0.0):.3f}/"
                        f"{row.get('hierarchical_mmdit_action_response_t1_p50', 0.0):.3f}/"
                        f"{row.get('hierarchical_mmdit_action_response_t2_p50', 0.0):.3f} "
                        f"hmpT50={row.get('hierarchical_mmdit_stage_pressure_t0_p50', 0.0):.3f}/"
                        f"{row.get('hierarchical_mmdit_stage_pressure_t1_p50', 0.0):.3f}/"
                        f"{row.get('hierarchical_mmdit_stage_pressure_t2_p50', 0.0):.3f} "
                        f"hmucT={row.get('hierarchical_mmdit_response_gain_corr_t0', 0.0):+.2f}/"
                        f"{row.get('hierarchical_mmdit_response_gain_corr_t1', 0.0):+.2f}/"
                        f"{row.get('hierarchical_mmdit_response_gain_corr_t2', 0.0):+.2f} "
                        f"hmpcT={row.get('hierarchical_mmdit_pressure_gain_corr_t0', 0.0):+.2f}/"
                        f"{row.get('hierarchical_mmdit_pressure_gain_corr_t1', 0.0):+.2f}/"
                        f"{row.get('hierarchical_mmdit_pressure_gain_corr_t2', 0.0):+.2f} "
                        f"hmnmax={row.get('hierarchical_mmdit_action_noisy_attention_max', 0.0):.3f} "
                        f"hmsmax={row.get('hierarchical_mmdit_action_stage_attention_max', 0.0):.3f} "
                        f"hmlmax={row.get('hierarchical_mmdit_action_low_attention_max', 0.0):.3f} "
                        f"halreff={row.get('hierarchical_mmdit_action_low_role_effective_count', 0.0):.2f} "
                        f"halgeo={row.get('hierarchical_mmdit_action_low_role_geom_attention', 0.0):.3f} "
                        f"haltrn={row.get('hierarchical_mmdit_action_low_role_transition_attention', 0.0):.3f} "
                        f"halevt={row.get('hierarchical_mmdit_action_low_role_event_attention', 0.0):.3f} "
                        f"halsta={row.get('hierarchical_mmdit_action_low_role_state_attention', 0.0):.3f} "
                        f"hallay={row.get('hierarchical_mmdit_action_low_role_layer_attention', 0.0):.3f} "
                        f"ogeo={row.get('owned_workspace_role_geom_attention', 0.0):.3f} "
                        f"otrn={row.get('owned_workspace_role_transition_attention', 0.0):.3f} "
                        f"oevt={row.get('owned_workspace_role_event_attention', 0.0):.3f} "
                        f"osta={row.get('owned_workspace_role_state_attention', 0.0):.3f} "
                        f"olay={row.get('owned_workspace_role_layer_attention', 0.0):.3f} "
                        f"ofixed={row.get('owned_hierarchical_manager_fixed_output_prior', 0.0):.0f} "
                        f"ofrole={row.get('owned_hierarchical_manager_fixed_role_prior', 0.0):.0f} "
                        f"ostrat={row.get('owned_hierarchical_low_role_stratified', 0.0):.0f} "
                        f"hmwieff={row.get('owned_workspace_interface_low_control_effective_control_tokens', 0.0):.2f}/"
                        f"{row.get('owned_workspace_interface_stage_control_effective_control_tokens', 0.0):.2f} "
                        f"ohsupd={row.get('owned_hierarchical_stage_update_norm', 0.0):.3f} "
                        f"ohprom={row.get('owned_hierarchical_manager_promote_gate', 0.0):.3f} "
                        f"ohprms={row.get('owned_hierarchical_stage_promoted_projected_rms', 0.0):.3f}/"
                        f"{row.get('owned_hierarchical_stage_promoted_normalized_rms', 0.0):.3f}/"
                        f"{row.get('owned_hierarchical_stage_promoted_realized_scale', 0.0):.3f} "
                        f"ohgerr={row.get('owned_hierarchical_stage_promote_gate_scale_error', 0.0):.1e} "
                        f"cgrad={row.get('grad_latent_cvae_action', 0.0):.3e} "
                        f"hwgrad={row.get('grad_latent_cvae_hierarchical_workspace', 0.0):.3e} "
                        f"hlgrad={row.get('grad_latent_cvae_hierarchical_low', 0.0):.3e} "
                        f"hsgrad={row.get('grad_latent_cvae_hierarchical_stage', 0.0):.3e} "
                        f"hmgrad={row.get('grad_latent_cvae_hierarchical_manager', 0.0):.3e} "
                        f"cgclip={row.get('grad_latent_cvae_action_post_clip', 0.0):.3e} "
                        f"agrad={row.get('grad_residual_action_flow', 0.0):.3e} "
                        f"rdgrad={row.get('grad_controlled_dynamics', 0.0):.3e} "
                        f"rcgrad={row.get('grad_latent_cvae_rollout_condition', 0.0):.3e} "
                        f"wgrad={row.get('grad_latent_cvae_workspace', 0.0):.3e} "
                        f"zpgrad={row.get('grad_latent_cvae_primary_modulation', 0.0):.3e} "
                        f"hmdgrad={row.get('grad_hierarchical_mmdit_action', 0.0):.3e} "
                        f"hmvgrad={row.get('grad_hierarchical_mmdit_velocity_head', 0.0):.3e} "
                        f"icgrad={row.get('grad_intent_contract_compiler', 0.0):.3e} "
                        f"owgrad={row.get('grad_owned_workspace', 0.0):.3e} "
                        f"hmbgrad={row.get('grad_hierarchical_mmdit_blocks', 0.0):.3e} "
                        f"hmbasegrad={row.get('grad_hierarchical_mmdit_shared_base', 0.0):.3e} "
                        f"hmwgrad={row.get('grad_hierarchical_mmdit_base_projection', 0.0):.3e} "
                        f"hmcopgrad={row.get('grad_hierarchical_mmdit_contractions', 0.0):.3e} "
                        f"hmcgrad={row.get('grad_hierarchical_mmdit_contraction_basis', 0.0):.3e}/"
                        f"{row.get('grad_hierarchical_mmdit_contraction_depth', 0.0):.3e} "
                        f"hmcmodgrad={row.get('grad_hierarchical_mmdit_content_modulation', 0.0):.3e} "
                        f"hmgategrad={row.get('grad_hierarchical_mmdit_host_gates', 0.0):.3e} "
                        f"hmdclip={row.get('grad_hierarchical_mmdit_action_post_clip', 0.0):.3e} "
                        f"grad={row['grad']:.3e} lr={optimizer.param_groups[0]['lr']:.3e} "
                        f"spb={seconds_per_batch:.3f}"
                    ),
                    flush=True,
                )
        train_metrics = _finalize_metric_tensors(
            metric_sums,
            metric_count,
            counts=metric_counts,
        )
        evidence_epoch = (
            getattr(system.planner, "evidence_latent_mmdit_action_decoder", None) is not None
        )
        if evidence_epoch:
            train_metrics = _filter_inactive_evidence_epoch_metrics(train_metrics)
        if flow_jepa_stage1:
            val_metrics = evaluate_flow_jepa_stage1(
                system=system,
                loader=val_loader,
                conditioner=conditioner,
                device=device,
                dtype=dtype,
                camera_names=camera_names,
                trainer=trainer,
                max_batches=trainer.max_val_batches,
                memory_reporter=memory_reporter,
                epoch=epoch,
                global_step=global_step,
            )
            score = float(val_metrics["loss"])
            deploy_eligible = False
        else:
            val_metrics = evaluate_v39_policy(
                system=system,
                loader=val_loader,
                conditioner=conditioner,
                device=device,
                dtype=dtype,
                camera_names=camera_names,
                action_normalizer=action_normalizer,
                trainer=trainer,
                max_batches=trainer.max_val_batches,
                memory_reporter=memory_reporter,
                epoch=epoch,
                global_step=global_step,
            )
            score = balanced_score(val_metrics, trainer)  # type: ignore[arg-type]
            deploy_eligible = is_deploy_eligible(val_metrics, trainer)  # type: ignore[arg-type]
        val_metrics["balanced_score"] = score
        val_metrics["deploy_eligible"] = float(deploy_eligible)
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        with (out_dir / "v39_policy_epochs.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(jsonable(record), separators=(",", ":")) + "\n")
        full = float(val_metrics.get("full_mse", float("inf")))
        f1 = float(val_metrics.get("gripper_f1", 0.0))
        recall = float(val_metrics.get("gripper_recall", 0.0))
        save = []
        select_contract = _is_contract_stage(trainer) and _uses_layer_adapter_contract(trainer)
        if flow_jepa_stage1:
            representation_value = float(val_metrics["loss"])
            if representation_value < best["stage1_representation"]:
                best["stage1_representation"] = representation_value
                save.append("best_stage1_representation.pt")
        elif select_contract:
            contract_value = float(val_metrics.get("contract_layer_contract", float("inf")))
            if contract_value < best["layer_contract"]:
                best["layer_contract"] = contract_value
                save.append("best_contract.pt")
        else:
            if full < best["full_mse"]:
                best["full_mse"] = full
                save.append("best_full.pt")
            if f1 > best["gripper_f1"]:
                best["gripper_f1"] = f1
                save.append("best_gripper_f1.pt")
            if recall > best["gripper_recall"]:
                best["gripper_recall"] = recall
                save.append("best_gripper_recall.pt")
            if score < best["balanced"]:
                best["balanced"] = score
                save.append("best_balanced.pt")
            if deploy_eligible and float(val_metrics["full_rmse"]) < best["deploy_full_rmse"]:
                best["deploy_full_rmse"] = float(val_metrics["full_rmse"])
                save.append("best_deploy.pt")
        payload = {
            "schema": "clearvla-v40-policy-checkpoint-v1",
            "stage1_contract": (
                {
                    "kind": "flow_dino_jepa_representation_v1",
                    "teacher": "frozen_dino_no_grad",
                    "target_action_conditioned": False,
                    "final_action_decoder_executed": False,
                    "layer_contracts_executed": False,
                    "window_offsets": list(system.policy_config.flow_jepa_effective_window_offsets),
                    "stage_offset": int(system.policy_config.flow_jepa_effective_stage_offset),
                }
                if flow_jepa_stage1
                else None
            ),
            "epoch": epoch,
            "global_step": global_step,
            "model": system.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": schedule.state_dict(),
            "policy_config": asdict(system.policy_config),
            "trainer_config": asdict(trainer),
            "action_normalizer": action_normalizer.to_dict(),
            "state_normalizer": state_normalizer.to_dict(),
            "context": context,
            "history": history,
            "best": best,
            "rng": rng_state(),
        }
        for name in save:
            torch.save(payload, ckpt_dir / name)
        torch.save(payload, ckpt_dir / "latest.pt")
        (out_dir / "v40_policy_summary.json").write_text(
            json.dumps(
                jsonable(
                    {"schema": "clearvla-v40-policy-summary-v1", "best": best, "latest": record}
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        if flow_jepa_stage1:
            print(
                _flow_jepa_stage1_epoch_log_line(
                    epoch=epoch,
                    global_step=global_step,
                    train=train_metrics,
                    val=val_metrics,
                ),
                flush=True,
            )
        elif evidence_epoch:
            print(
                _evidence_epoch_log_line(
                    epoch=epoch,
                    global_step=global_step,
                    train=train_metrics,
                    val=val_metrics,
                ),
                flush=True,
            )
        else:
            print(json.dumps(jsonable(record), separators=(",", ":")), flush=True)
    return {"history": history, "best": best}


__all__ = [
    "POLICY_CHECKPOINT_SCHEMAS",
    "V39PolicyTrainerConfig",
    "prepare_v39_policy_sample",
    "encode_target_anchor_tokens",
    "future_latent_loss",
    "action_effect_loss",
    "rollout_dynamics_loss",
    "rollout_delta_loss",
    "rollout_contrast_loss",
    "flow_jepa_stage1_losses",
    "flow_jepa_interval_stage_terms",
    "_validate_complete_v115_model_contract",
    "_validate_complete_v116_model_contract",
    "_validate_complete_v117_model_contract",
    "_validate_differential_intent_effect_323_model_contract",
    "_validate_grounded_intent_effect_323_model_contract",
    "flow_losses",
    "layer_contract_losses",
    "evaluate_flow_jepa_stage1",
    "evaluate_v39_policy",
    "CudaMemoryReporter",
    "train_v39_policy",
    "_validate_complete_v103_model_probe_contract",
    "_validate_complete_v104_model_contract",
    "_validate_complete_v105_model_contract",
    "_validate_complete_v106_model_contract",
    "_validate_complete_v107_model_contract",
    "_validate_complete_v108_model_contract",
    "_validate_complete_v109_model_contract",
    "_validate_complete_v110_model_contract",
    "_validate_complete_v111_model_contract",
    "_validate_complete_v112_model_contract",
    "_validate_complete_v113_model_contract",
    "evaluate_model_path_intervention",
]
