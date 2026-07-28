import inspect
import math
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset

from clearvla.cli.train_v40_policy import (
    _source_fingerprint,
    _validate_flow_jepa_stage1_checkpoint,
    _validate_required_model_contract,
)
from clearvla.experiments.observed_state_lab.dataset import CachedTokenPolicyWindowDataset
from clearvla.experiments.classic_policy_lab.cli_common import make_loader
from clearvla.experiments.observed_state_lab.policy_runtime_v39 import (
    V39PolicyTrainerConfig,
    _action_path_paired_metrics,
    _action_path_probe_batch_selection,
    _model_path_acceptance_matrix,
    _optimizer_groups,
    _validate_complete_v103_model_probe_contract,
    _validate_complete_v104_model_contract,
    _validate_complete_v105_model_contract,
    _validate_complete_v106_model_contract,
    _validate_complete_v107_model_contract,
    _validate_complete_v108_model_contract,
    _validate_complete_v109_model_contract,
    _validate_complete_v110_model_contract,
    _validate_complete_v111_model_contract,
    _validate_v106_preflight_target_pack,
    _validate_v102_resume_contract,
    flow_losses,
    flow_jepa_future_change_loss,
    flow_jepa_future_change_direction_loss,
    flow_jepa_future_horizon_diagnostics,
    flow_jepa_future_reliable_diagnostics,
    flow_jepa_horizon_address_loss,
    flow_jepa_interval_stage_terms,
    flow_jepa_future_prediction_loss,
    flow_jepa_stage1_losses,
)
from clearvla.experiments.observed_state_lab.policy_runtime_v36_3 import (
    position_weights,
    trajectory_information_weights,
)
from clearvla.data.samplers import (
    InformationBalancedBatchSampler,
    InformationBalancedSamplerConfig,
)
from clearvla.policy.config import V39PolicyConfig
from clearvla.policy.flow_dino_evidence import (
    FlowDINOEvidenceEncoder,
    LateRawDetailEvidence,
    LatentSeaRaft,
    SoftAddressLatticeBank,
    _CorrelationPyramid,
    _DenseRawFlowRefiner,
    _EarlyMaskedRawContextEncoder,
    _RawDeformableAddressReader,
    _SoftMultiResolutionAddressCompiler,
    _HorizonSoftAddressJEPA,
    _IntervalStageDeltaOrganizer,
    _SoftFlowAddressReader,
    _SparseFineFlowRefiner,
    _fixed_observable_motion,
    _fixed_raw_motion_descriptor,
    _continuous_cycle_visibility,
    _smooth_bound_flow_to_image,
    _stable_sqrt,
    _stable_vector_norm,
    warp_patch_grid,
)
from clearvla.policy.goal_conditioning import (
    StatelessPhaseAdapter,
    load_precomputed_t5_condition,
)
from clearvla.policy.proposal import RejectableHistoryProposal
from clearvla.policy.role_delta_attnres import (
    RoleDeltaAttnRes,
    rms_floored_l2_normalize,
    smooth_rms_contract,
    variance_floored_centered_norm,
)
from clearvla.policy.system import V39PolicySystem, balanced_future_teacher_mask
from clearvla.policy.time_domain_mmdit import EvidenceLatentMMDiTActionDecoder
from clearvla.policy.trunk import (
    LateRawDetailPolicyReader,
    _CoordinateTypedLocalRefiner,
    _StructuredOwnershipLocalRefiner,
    _align_milestone_tokens_to_horizon,
)
from clearvla.policy.trunk_primitives import (
    TemporalDynamicsBoundDiTBlock,
    UnifiedCanvasSeed,
)


def _flow_config(**overrides: object) -> V39PolicyConfig:
    values: dict[str, object] = {
        "action_dim": 3,
        "state_dim": 3,
        "action_horizon": 4,
        "executed_history_length": 2,
        "hidden_size": 32,
        "num_heads": 4,
        "depth": 2,
        "action_decoder_depth": 2,
        "proposal_depth": 1,
        "first_execution_steps": 1,
        "mid_execution_steps": 2,
        "visual_token_dim": 16,
        "visual_history_length": 3,
        "num_cameras": 2,
        "patches_per_camera": 16,
        "canvas_registers": 2,
        "future_anchors": 2,
        "target_future_count": 3,
        "future_grid_size": 4,
        "rollout_tail_start_step": 2,
        "rollout_tail_full_step": 3,
        "midcut_layer": 1,
        "latent_action_near_steps": 1,
        "latent_action_mid_steps": 2,
        "flow_jepa_enabled": 1,
        "flow_jepa_grid_size": 4,
        "flow_jepa_feature_dim": 32,
        "flow_jepa_flow_iters": 1,
        "flow_jepa_corr_levels": 2,
        "flow_jepa_corr_radius": 1,
        "flow_jepa_mask_ratio": 0.375,
        "flow_jepa_mask_block_size": 2,
        "flow_jepa_motion_mask_fraction": 0.6,
        "flow_jepa_directed_canvas_attention": 1,
        "dropout": 0.0,
        "canvas_dropout": 0.0,
        "role_dropout": 0.0,
    }
    values.update(overrides)
    config = V39PolicyConfig(**values)
    config.validate()
    return config


def _visual(config: V39PolicyConfig, batch: int = 2) -> torch.Tensor:
    return torch.randn(
        batch,
        config.visual_history_length,
        config.num_cameras,
        config.patches_per_camera,
        config.visual_token_dim,
    )


def _late_bottleneck_config(**overrides: object) -> V39PolicyConfig:
    values: dict[str, object] = {
        "patches_per_camera": 64,
        "future_anchors": 3,
        "target_future_count": 3,
        "flow_jepa_late_bottleneck": 1,
        "flow_jepa_dense_depth": 1,
        "flow_jepa_fine_radius": 1,
        "flow_jepa_reader_radius": 1,
        "flow_jepa_reader_heads": 2,
        "flow_jepa_window_offsets": (2, 4, 8),
        "flow_jepa_stage_offset": 0,
        "flow_jepa_stage_tokens": 0,
        "layer_recurrent_consequence": 0,
    }
    values.update(overrides)
    return _flow_config(**values)


def _raw_role_config(**overrides: object) -> V39PolicyConfig:
    values: dict[str, object] = {
        "depth": 8,
        "midcut_layer": 6,
        "flow_jepa_raw_image_enabled": 1,
        "flow_jepa_role_hierarchy": 1,
        "flow_jepa_raw_base_channels": 8,
        "flow_jepa_raw_mid_radius": 1,
        "flow_jepa_raw_high_radius": 1,
        "flow_jepa_raw_reader_radius": 1,
        "flow_jepa_raw_reader_heads": 4,
        "flow_jepa_grounding_blocks": 3,
        "flow_jepa_world_blocks": 3,
        "flow_jepa_policy_blocks": 2,
    }
    values.update(overrides)
    return _late_bottleneck_config(**values)


def _v102_config(**overrides: object) -> V39PolicyConfig:
    values: dict[str, object] = {
        "flow_jepa_zero_flow_guard": 1,
        "flow_jepa_complementary_raw_detail": 1,
        "flow_jepa_strict_role_visual_path": 1,
        "flow_jepa_source_aligned_raw_fusion": 1,
        "flow_jepa_policy_workspace_fixed_fusion": 1,
        "flow_jepa_world_anchor_write_only": 1,
        "flow_jepa_late_policy_detail": 1,
        "flow_jepa_late_policy_detail_scale": 0.25,
        "flow_jepa_policy_workspace_horizon_pool": 1,
    }
    values.update(overrides)
    return _raw_role_config(**values)


def _v103_config(**overrides: object) -> V39PolicyConfig:
    return _v102_config(
        flow_jepa_world_anchor_write_only=0,
        flow_jepa_soft_address_lattice=1,
        flow_jepa_address_slots=3,
        flow_jepa_address_route_dim=16,
        flow_jepa_address_query_chunk=2,
        flow_jepa_address_flow_prior_floor=0.25,
        **overrides,
    )


def _typed_332_config(**overrides: object) -> V39PolicyConfig:
    values: dict[str, object] = {
        "final_action_decoder": "evidence_latent_mmdit_action",
        "layer_contract_adapters": 1,
        "layer_contract_adapter_dim": 32,
        "latent_cvae_mmdit_depth": 1,
        "flow_jepa_zero_flow_guard": 1,
        "flow_jepa_complementary_raw_detail": 1,
        "flow_jepa_strict_role_visual_path": 1,
        "flow_jepa_source_aligned_raw_fusion": 1,
        "flow_jepa_policy_workspace_fixed_fusion": 0,
        "flow_jepa_world_anchor_write_only": 0,
        "flow_jepa_late_policy_detail": 1,
        "flow_jepa_late_policy_detail_scale": 0.25,
        "flow_jepa_policy_workspace_horizon_pool": 1,
        "flow_jepa_soft_address_lattice": 1,
        "flow_jepa_address_slots": 3,
        "flow_jepa_address_route_dim": 16,
        "flow_jepa_address_query_chunk": 2,
        "flow_jepa_address_flow_prior_floor": 0.25,
        "role_attnres_enabled": 1,
        "role_attnres_key_dim": 16,
        "role_attnres_ground_to_world": 1,
        "role_attnres_world_to_policy": 1,
        "role_attnres_policy_to_mmdit": 1,
    }
    values.update(overrides)
    return _raw_role_config(**values)


def _complete_v103_config(**overrides: object) -> V39PolicyConfig:
    values: dict[str, object] = {
        "action_horizon": 24,
        "future_anchors": 4,
        "target_future_count": 4,
        "flow_jepa_window_offsets": (4, 12, 24, 48),
        "flow_jepa_teacher_balanced_target_mask": 0,
        "flow_jepa_predictive_change_contract": 1,
        "action_history_enabled": 1,
        "action_history_recent_tokens": 1,
        "action_history_summary_tokens": 1,
        "action_history_condition_exact_null": 1,
        "action_history_proposal_detach": 0,
        "goal_conditioning_enabled": 1,
        "goal_token_count": 2,
        "goal_language_dim": 16,
        "goal_language_max_tokens": 5,
        "goal_resampler_depth": 1,
        "goal_condition_exact_null": 1,
        "stateless_phase_enabled": 1,
        "stateless_phase_count": 4,
        "latent_cvae_mmdit_depth": 3,
        "latent_cvae_mmdit_operator_capacity": 1,
        "latent_cvae_mmdit_operator_rank": 8,
        "latent_cvae_mmdit_operator_groups": 8,
        "latent_cvae_mmdit_execution_controller": 1,
        "latent_cvae_mmdit_dynamic_block_route": 1,
        "latent_cvae_mmdit_control_tokens": 2,
        "latent_cvae_mmdit_controller_heads": 4,
        "latent_cvae_mmdit_max_dwell": 2,
        "latent_cvae_mmdit_dwell_mode": "learned",
        "latent_cvae_mmdit_identity_candidate": 1,
        "latent_cvae_mmdit_execution_eval_policy": "soft",
    }
    values.update(overrides)
    return _typed_332_config(**values)


def _complete_v104_config(**overrides: object) -> V39PolicyConfig:
    values: dict[str, object] = {
        "flow_jepa_bounded_flow_coordinates": 1,
        "flow_jepa_sequential_horizon_memory": 1,
        "role_residual_amplitude_contract": 1,
        "role_residual_max_update_rms": 0.50,
        "role_attnres_max_value_rms": 1.00,
    }
    values.update(overrides)
    return _complete_v103_config(**values)


def _complete_v105_config(**overrides: object) -> V39PolicyConfig:
    values: dict[str, object] = {
        "flow_jepa_horizon_soft_address": 1,
        "flow_jepa_horizon_address_update_scale": 0.10,
    }
    values.update(overrides)
    return _complete_v104_config(**values)


def _complete_v106_config(**overrides: object) -> V39PolicyConfig:
    values: dict[str, object] = {
        "flow_jepa_variance_safe_routing": 1,
        "flow_jepa_complete_numerical_contract": 1,
        "flow_jepa_routing_norm_floor": 0.25,
        "flow_jepa_correlation_rms_floor": 0.10,
        "flow_jepa_visibility_transition_fraction": 0.10,
        "flow_jepa_horizon_value_max_rms": 0.50,
        "flow_jepa_interval_stage_delta": 1,
        "flow_jepa_interval_boundaries": (4, 8, 16, 32, 48),
        "flow_jepa_interval_support_offsets": tuple(range(4, 49, 4)),
        "flow_jepa_interval_stage_update_scale": 0.10,
    }
    values.update(overrides)
    return _complete_v105_config(**values)


def _complete_v107_config(**overrides: object) -> V39PolicyConfig:
    values: dict[str, object] = {
        "flow_jepa_policy_multi_glimpse_address": 1,
        "flow_jepa_horizon_cell_fine_address": 1,
        "flow_jepa_interval_stage_typed_value": 1,
        "role_residual_contract_after_gate": 1,
    }
    values.update(overrides)
    return _complete_v106_config(**values)


def _complete_v108_config(**overrides: object) -> V39PolicyConfig:
    values: dict[str, object] = {
        "flow_jepa_online_horizon_address": 1,
    }
    values.update(overrides)
    return _complete_v107_config(**values)


def _complete_v109_config(**overrides: object) -> V39PolicyConfig:
    values: dict[str, object] = {
        "flow_jepa_progressive_grounding_address": 1,
    }
    values.update(overrides)
    return _complete_v108_config(**values)


def _complete_v110_config(**overrides: object) -> V39PolicyConfig:
    values: dict[str, object] = {
        "flow_jepa_coordinate_typed_raw_detail": 1,
        "flow_jepa_raw_micro_grid": 3,
    }
    values.update(overrides)
    return _complete_v109_config(**values)


def _complete_v111_config(**overrides: object) -> V39PolicyConfig:
    values: dict[str, object] = {
        "flow_jepa_structured_ownership_bottleneck": 1,
    }
    values.update(overrides)
    return _complete_v110_config(**values)


def _complete_v103_trainer(**overrides: object) -> V39PolicyTrainerConfig:
    values: dict[str, object] = {
        "training_stage": "policy",
        "contract_mode": "layer_adapter",
        "single_stage_role_lr": 1,
        "flow_jepa_future_loss_weight": 0.10,
        "flow_jepa_future_change_loss_weight": 0.0,
        "flow_jepa_horizon_balance_mode": "per_horizon",
        "flow_jepa_stage_loss_weight": 0.0,
        "flow_jepa_warp_loss_weight": 0.03,
        "flow_jepa_identity_advantage_loss_weight": 0.02,
        "flow_jepa_static_identity_loss_weight": 0.01,
        "flow_jepa_cycle_loss_weight": 0.01,
        "flow_jepa_smoothness_loss_weight": 0.002,
        "flow_jepa_uncertainty_nll_weight": 0.005,
        "flow_jepa_refinement_sequence_loss_weight": 0.02,
        "latent_cvae_mmdit_dwell_mode": "learned",
        "latent_cvae_mmdit_execution_value_loss_weight": 0.05,
        "rollout_dynamics_loss_weight": 0.0,
        "rollout_delta_loss_weight": 0.0,
        "rollout_contrast_loss_weight": 0.0,
        "rollout_variance_loss_weight": 0.0,
        "rollout_norm_loss_weight": 0.0,
        "rollout_milestone_delta_match_weight": 0.0,
        "future_latent_loss_weight": 0.0,
        "action_effect_loss_weight": 0.0,
        "layer_contract_aux_loss_weight": 0.0,
    }
    values.update(overrides)
    return V39PolicyTrainerConfig(**values)


def _complete_v105_trainer(**overrides: object) -> V39PolicyTrainerConfig:
    values: dict[str, object] = {
        "flow_jepa_future_reliable_normalization": 1,
        "flow_jepa_horizon_address_loss_weight": 0.02,
    }
    values.update(overrides)
    return replace(_complete_v103_trainer(), **values)


def _complete_v106_trainer(**overrides: object) -> V39PolicyTrainerConfig:
    values: dict[str, object] = {
        "flow_jepa_interval_stage_loss_weight": 0.02,
    }
    values.update(overrides)
    return replace(_complete_v105_trainer(), **values)


def _complete_v107_trainer(**overrides: object) -> V39PolicyTrainerConfig:
    return replace(_complete_v106_trainer(), **overrides)


def _complete_v108_trainer(**overrides: object) -> V39PolicyTrainerConfig:
    return replace(_complete_v107_trainer(), **overrides)


def _complete_v109_trainer(**overrides: object) -> V39PolicyTrainerConfig:
    return replace(_complete_v108_trainer(), **overrides)


def _complete_v110_trainer(**overrides: object) -> V39PolicyTrainerConfig:
    return replace(_complete_v109_trainer(), **overrides)


def _complete_v111_trainer(**overrides: object) -> V39PolicyTrainerConfig:
    return replace(_complete_v110_trainer(), **overrides)


def _raw_visual(
    config: V39PolicyConfig, batch: int = 1, side: int = 64
) -> torch.Tensor:
    return torch.rand(
        batch,
        config.visual_history_length,
        config.num_cameras,
        3,
        side,
        side,
    )


def test_early_raw_mask_hides_pixels_before_any_learned_spatial_mixing() -> None:
    torch.manual_seed(109)
    grid = 4
    encoder = _EarlyMaskedRawContextEncoder(
        16, 32, grid, activation_checkpoint=False
    ).eval()
    raw = torch.rand(1, 2, 1, 3, 64, 64)
    mask = torch.zeros(1, 2, 1, grid, grid, dtype=torch.bool)
    mask[..., 1:3, 1:3] = True
    pixel_mask = F.interpolate(
        mask.reshape(2, 1, grid, grid).float(),
        size=(64, 64),
        mode="nearest",
    ).bool().reshape(1, 2, 1, 1, 64, 64)
    changed = torch.where(pixel_mask, 10.0 * torch.randn_like(raw), raw)
    with torch.no_grad():
        first = encoder(raw, mask)
        second = encoder(changed, mask)
    # This is an exact information-boundary test, not a tolerance-based
    # similarity check: hidden pixels never enter a trainable mixing layer.
    assert torch.equal(first, second)


def test_predictive_change_loss_uses_delta_not_absolute_scene_copy() -> None:
    torch.manual_seed(110)
    batch, horizons, cells, hidden = 2, 2, 3, 8
    current = torch.randn(batch, cells, hidden)
    target_delta = 0.2 * torch.randn(batch, horizons, cells, hidden)
    target = current[:, None] + target_delta
    perfect = target_delta.reshape(batch, horizons * cells, hidden).clone()
    common = {
        "flow_jepa_future_target": target.reshape(
            batch, horizons * cells, hidden
        ),
        "flow_jepa_current_target": current,
        "flow_jepa_future_target_mask": torch.ones(
            batch, horizons * cells, dtype=torch.bool
        ),
    }
    perfect_output = {
        **common,
        "flow_jepa_future_pred": perfect,
        "flow_jepa_future_delta_pred": perfect,
    }
    zero = torch.zeros_like(perfect, requires_grad=True)
    zero_output = {
        **common,
        "flow_jepa_future_pred": zero,
        "flow_jepa_future_delta_pred": zero,
    }
    perfect_loss = flow_jepa_future_prediction_loss(
        perfect_output, balance_horizons=True
    )
    zero_loss = flow_jepa_future_prediction_loss(
        zero_output, balance_horizons=True
    )
    assert float(perfect_loss) < 1e-6
    assert float(zero_loss) > float(perfect_loss) + 1e-3
    zero_loss.backward()
    assert zero.grad is not None and float(zero.grad.abs().sum()) > 0.0


def test_predictive_change_reuses_one_online_mask_for_context_and_targets() -> None:
    torch.manual_seed(116)
    config = _typed_332_config(
        flow_jepa_raw_activation_checkpoint=0,
        flow_jepa_teacher_balanced_target_mask=0,
        flow_jepa_predictive_change_contract=1,
    )
    encoder = FlowDINOEvidenceEncoder(config).train()
    pack = encoder(
        _visual(config, batch=2),
        raw_visual=_raw_visual(config, batch=2),
    )
    future_mask = pack.future_target_mask.reshape(
        2,
        config.future_anchors,
        config.num_cameras,
        config.flow_jepa_grid_size,
        config.flow_jepa_grid_size,
    )
    latest_context_mask = pack.context_dropout_mask[:, -1][:, None].expand_as(
        future_mask
    )
    assert torch.equal(future_mask, latest_context_mask)
    assert torch.equal(
        future_mask[:, 1:],
        future_mask[:, :1].expand_as(future_mask[:, 1:]),
    )
    assert float(pack.metrics["flow_jepa_context_target_mask_aligned"]) == 1.0
    assert float(pack.metrics["flow_jepa_future_shared_spatial_mask"]) == 1.0

    encoder.eval()
    with torch.no_grad():
        eval_pack = encoder(
            _visual(config, batch=2),
            raw_visual=_raw_visual(config, batch=2),
        )
    assert not bool(eval_pack.context_dropout_mask.any())
    assert float(eval_pack.metrics["flow_jepa_context_target_mask_aligned"]) == 0.0
    assert float(eval_pack.metrics["flow_jepa_deploy_context_unmasked"]) == 1.0

    try:
        _typed_332_config(
            flow_jepa_teacher_balanced_target_mask=1,
            flow_jepa_predictive_change_contract=1,
        )
    except ValueError as error:
        assert "teacher-balanced target selection disabled" in str(error)
    else:
        raise AssertionError(
            "predictive-change config accepted a future-teacher-selected mask"
        )


def test_v101_action_anchor_weights_are_mild_normalized_and_tail_aware() -> None:
    config = SimpleNamespace(
        action_horizon=24,
        flow_jepa_action_offsets=(4, 12, 24),
    )
    trainer = V39PolicyTrainerConfig(
        horizon_weight_mode="anchor_bands",
        horizon_tail_emphasis=0.20,
        horizon_first_step_protection=0.05,
    )
    weights = position_weights(config, trainer, torch.device("cpu"))
    torch.testing.assert_close(weights.mean(), torch.ones(()))
    assert float(weights[0]) > float(weights[1])
    assert float(weights[12]) > float(weights[4]) > float(weights[1])
    assert float(weights.max() / weights.min()) < 1.25


def test_v101_information_sampler_keeps_uniform_lane_and_is_epoch_deterministic() -> None:
    motion = np.linspace(0.0, 1.0, 32, dtype=np.float64)
    events = np.zeros(32, dtype=bool)
    events[[7, 23]] = True
    config = InformationBalancedSamplerConfig(
        batch_size=8,
        uniform_fraction=0.50,
        event_fraction=0.125,
        motion_quantile=0.75,
        seed=19,
    )
    first = InformationBalancedBatchSampler(motion, events, config)
    second = InformationBalancedBatchSampler(motion, events, config)
    assert list(first) == list(second)
    first.set_epoch(1)
    assert list(first) != list(second)
    for batch in InformationBalancedBatchSampler(motion, events, config):
        assert len(batch) == 8
        assert len(set(batch)) == len(batch)


def test_v101_teacher_mask_has_exact_disjoint_per_horizon_quota() -> None:
    current = torch.zeros(1, 8, 2)
    future = torch.zeros(1, 16, 2, requires_grad=True)
    with torch.no_grad():
        future[:, 4, 0] = 2.0
        future[:, 5, 0] = 1.5
        future[:, 14, 1] = 2.5
        future[:, 15, 1] = 2.0
    observed = torch.zeros(1, 16, dtype=torch.bool)
    observed[:, [0, 1, 8, 9]] = True
    mask, metrics = balanced_future_teacher_mask(
        future,
        current,
        observed,
        mask_ratio=0.50,
        past_fraction=0.25,
        change_fraction=0.50,
    )
    assert mask.reshape(1, 2, 8).sum(dim=-1).tolist() == [[4, 4]]
    assert not mask.requires_grad
    assert float(metrics["flow_jepa_teacher_mask_past_fraction"]) == 0.25
    assert float(metrics["flow_jepa_teacher_mask_change_fraction"]) == 0.50
    assert float(metrics["flow_jepa_teacher_mask_uniform_fraction"]) == 0.25


def test_v101_per_horizon_change_gradient_is_invariant_to_far_scale() -> None:
    def gradient(far_scale: float, *, balanced: bool) -> torch.Tensor:
        pred = torch.full((1, 4, 2), 0.2, requires_grad=True)
        target = torch.tensor(
            [[[1.0, 0.0], [0.8, 0.2], [far_scale, 0.0], [0.0, far_scale]]]
        )
        output = {
            "flow_jepa_future_pred": pred,
            "flow_jepa_future_target": target,
            "flow_jepa_current_target": torch.zeros(1, 2, 2),
            "flow_jepa_future_target_mask": torch.ones(1, 4, dtype=torch.bool),
        }
        flow_jepa_future_change_loss(
            output, balance_horizons=balanced
        ).backward()
        assert pred.grad is not None
        return pred.grad[:, :2].detach()

    torch.testing.assert_close(
        gradient(10.0, balanced=True),
        gradient(100.0, balanced=True),
        rtol=1e-5,
        atol=1e-6,
    )
    assert not torch.allclose(
        gradient(10.0, balanced=False),
        gradient(100.0, balanced=False),
    )


def test_per_horizon_absolute_jepa_can_use_offsets_without_change_chart() -> None:
    pred = torch.zeros(1, 6, 3, requires_grad=True)
    loss = flow_jepa_future_prediction_loss(
        {
            "flow_jepa_future_pred": pred,
            "flow_jepa_future_target": torch.ones_like(pred),
            "flow_jepa_future_target_mask": torch.ones(1, 6, dtype=torch.bool),
            "flow_jepa_future_offsets": (4, 12, 24),
        },
        balance_horizons=True,
    )
    loss.backward()
    assert pred.grad is not None
    assert bool(torch.isfinite(pred.grad).all())


def test_v101_zero_information_weight_preserves_exact_legacy_sample_scale() -> None:
    config = _late_bottleneck_config()
    trainer = V39PolicyTrainerConfig(trajectory_information_weight=0.0)
    sample = {
        "policy_action": torch.randn(3, config.action_horizon, config.action_dim),
        "action_state": torch.randn(3, config.action_dim),
    }
    weight, score = trajectory_information_weights(
        sample, trainer, device=torch.device("cpu")
    )
    torch.testing.assert_close(weight, torch.ones_like(weight))
    assert bool(torch.isfinite(score).all())


def test_v101_history_dropout_cuts_direct_memory_and_proposal_alias_together() -> None:
    torch.manual_seed(101)
    config = _flow_config(action_history_condition_dropout=0.50)
    system = V39PolicySystem(config).train()
    captured: dict[str, torch.Tensor] = {}

    def capture_seed(_module, _args, kwargs) -> None:
        for key in ("executed_history", "executed_memory", "proposal_tokens"):
            captured[key] = kwargs[key].detach().clone()

    handle = system.planner.seed.register_forward_pre_hook(capture_seed, with_kwargs=True)
    batch = 1
    target_action = torch.randn(batch, config.action_horizon, config.action_dim)
    encoded = system.codec.encode(target_action, torch.zeros(batch, config.action_dim))
    with patch("torch.rand", return_value=torch.zeros(batch)):
        output = system.flow_training_forward(
            _visual(config, batch=batch),
            torch.randn(batch, config.visual_history_length, config.state_dim),
            torch.randn(batch, config.executed_history_length, config.action_dim),
            torch.zeros(batch, config.state_dim),
            target_action,
            target_visual=torch.randn(
                batch,
                config.target_future_count,
                config.visual_history_length,
                config.num_cameras,
                config.patches_per_camera,
                config.visual_token_dim,
            ),
            training_noise=torch.zeros_like(encoded),
            training_time=torch.full((batch,), 0.5),
            proposal_keep=torch.ones(batch),
            make_counterfactuals=False,
        )
    handle.remove()
    assert float(output["condition_action_history_keep"]) == 0.0
    for value in captured.values():
        torch.testing.assert_close(value, torch.zeros_like(value))


def test_raw_grounding_preserves_high_resolution_until_grounded_reader() -> None:
    torch.manual_seed(97)
    config = _raw_role_config()
    encoder = FlowDINOEvidenceEncoder(config).train()
    pack = encoder(_visual(config, batch=1), raw_visual=_raw_visual(config))
    assert pack.raw_context is not None
    assert pack.raw_context.high_features.shape[-2:] == (16, 16)
    assert pack.patch_flow_forward.shape[-2:] == (
        config.flow_jepa_grid_size,
        config.flow_jepa_grid_size,
    )
    rollout_count = config.future_anchors * config.num_cameras * config.flow_jepa_grid_size**2
    grounding_canvas = torch.randn(1, rollout_count + 2, config.hidden_size)
    selector, values, metrics = encoder.refine_raw_evidence(
        pack,
        grounding_canvas,
        {"rollout": slice(2, 2 + rollout_count)},
    )
    raw_count = config.num_cameras * config.flow_jepa_grid_size**2
    assert selector.shape[1] == pack.selector_tokens.shape[1] + raw_count
    assert values.shape == selector.shape
    assert int(metrics["flow_jepa_raw_detail_token_count"]) == raw_count
    torch.testing.assert_close(
        metrics["flow_jepa_raw_address_flow_mass"]
        + metrics["flow_jepa_raw_address_fallback_mass"],
        torch.ones(()),
    )
    objective = values.float().square().mean() + sum(pack.losses.values())
    objective.backward()
    assert encoder.raw_flow is not None
    assert encoder.raw_address_reader is not None
    assert encoder.raw_flow.pyramid.stem[0].weight.grad is not None
    assert encoder.flow.delta_head[-1].weight.grad is not None
    assert encoder.raw_flow.high.update[-1].weight.grad is not None
    assert encoder.raw_address_reader.query.grad is not None
    assert pack.raw_context is None


def test_identity_centered_dino_seed_has_no_uniform_center_drift() -> None:
    config = _raw_role_config(flow_jepa_raw_activation_checkpoint=0)
    flow = LatentSeaRaft(
        config, identity_centered_initialization=True
    ).eval()
    with torch.no_grad():
        for parameter in flow.parameters():
            parameter.zero_()
    visual = torch.randn(
        2,
        config.flow_jepa_grid_size,
        config.flow_jepa_grid_size,
        config.visual_token_dim,
    )
    estimate = flow(visual, visual.roll(1, dims=0))
    torch.testing.assert_close(estimate.flow, torch.zeros_like(estimate.flow))
    torch.testing.assert_close(
        estimate.correlation_entropy, torch.ones_like(estimate.correlation_entropy)
    )


def test_smooth_flow_chart_cannot_escape_supervision_and_keeps_gradients() -> None:
    flow = torch.tensor(
        [
            [
                [[-50.0, -2.0, 2.0, 50.0]] * 4,
                [[-50.0] * 4, [-2.0] * 4, [2.0] * 4, [50.0] * 4],
            ]
        ],
        requires_grad=True,
    )
    bounded, compression = _smooth_bound_flow_to_image(flow)
    base_y, base_x = torch.meshgrid(
        torch.arange(4, dtype=torch.float32),
        torch.arange(4, dtype=torch.float32),
        indexing="ij",
    )
    coordinates = torch.stack((base_x, base_y), dim=0)[None] + bounded
    assert bool((coordinates >= 0.0).all())
    assert bool((coordinates <= 3.0).all())
    assert float(compression.detach().mean()) > 0.0
    bounded.square().mean().backward()
    assert flow.grad is not None and torch.isfinite(flow.grad).all()

    identity, identity_compression = _smooth_bound_flow_to_image(
        torch.zeros(2, 2, 4, 4)
    )
    assert torch.equal(identity, torch.zeros_like(identity))
    assert torch.equal(
        identity_compression, torch.zeros_like(identity_compression)
    )


def test_bounded_raw_refiner_preserves_valid_coordinates_for_large_seed() -> None:
    refiner = _DenseRawFlowRefiner(
        8,
        32,
        radius=1,
        uncertainty_floor=0.03,
        activation_checkpoint=False,
        preserve_uncertain_seed=True,
        bounded_coordinates=True,
    ).eval()
    with torch.no_grad():
        for parameter in refiner.parameters():
            parameter.zero_()
    first = torch.randn(1, 8, 12, 12)
    estimate = refiner(
        first,
        first.roll(1, dims=0),
        torch.full((1, 2, 4, 4), 100.0),
        torch.zeros(1, 1, 4, 4),
    )
    base = torch.stack(
        torch.meshgrid(
            torch.arange(12, dtype=torch.float32),
            torch.arange(12, dtype=torch.float32),
            indexing="ij",
        )[::-1],
        dim=0,
    )[None]
    coordinates = base + estimate.flow
    assert bool((coordinates >= 0.0).all())
    assert bool((coordinates <= 11.0).all())
    assert estimate.boundary_compression is not None
    assert float(estimate.boundary_compression.mean()) > 0.0


def test_zero_motion_magnitudes_have_finite_gradients_and_zero_vector_subgradient() -> None:
    vector = torch.zeros(2, 2, 4, 4, requires_grad=True)
    scalar = torch.zeros(2, 1, 4, 4, requires_grad=True)
    vector_norm = _stable_vector_norm(vector, dim=1, keepdim=True)
    scalar_root = _stable_sqrt(scalar)
    torch.testing.assert_close(vector_norm, torch.zeros_like(vector_norm))
    torch.testing.assert_close(scalar_root, torch.zeros_like(scalar_root))
    (vector_norm.sum() + scalar_root.sum()).backward()
    assert vector.grad is not None and torch.isfinite(vector.grad).all()
    assert scalar.grad is not None and torch.isfinite(scalar.grad).all()
    torch.testing.assert_close(vector.grad, torch.zeros_like(vector.grad))


def test_semantic_only_encoder_uses_identity_centered_flow_seed() -> None:
    encoder = FlowDINOEvidenceEncoder(_late_bottleneck_config())
    assert encoder.flow.identity_centered_initialization


def test_uniform_local_raw_correlation_has_zero_boundary_residual() -> None:
    refiner = _DenseRawFlowRefiner(
        8,
        32,
        radius=1,
        uncertainty_floor=0.03,
        activation_checkpoint=False,
    ).eval()
    with torch.no_grad():
        for parameter in refiner.parameters():
            parameter.zero_()
    first = torch.randn(2, 8, 12, 12)
    estimate = refiner(first, first.roll(1, dims=0), torch.zeros(2, 2, 4, 4))
    torch.testing.assert_close(estimate.flow, torch.zeros_like(estimate.flow))


def test_uncertain_raw_seed_preserves_coordinate_and_widens_search() -> None:
    refiner = _DenseRawFlowRefiner(
        8,
        32,
        radius=1,
        uncertainty_floor=0.03,
        activation_checkpoint=False,
        preserve_uncertain_seed=True,
    ).eval()
    with torch.no_grad():
        for parameter in refiner.parameters():
            parameter.zero_()
    first = torch.randn(1, 8, 12, 12)
    coarse = torch.ones(1, 2, 4, 4)
    estimate = refiner(
        first,
        first.roll(1, dims=0),
        coarse,
        torch.zeros(1, 1, 4, 4),
    )
    expected = torch.full_like(estimate.flow, 11.0 / 3.0)
    torch.testing.assert_close(estimate.iterations[0], expected)
    torch.testing.assert_close(estimate.flow, expected)


def test_fixed_motion_evidence_rewards_correct_translation_but_not_static_motion() -> None:
    image = torch.zeros(1, 3, 32, 32)
    image[:, 0, 8:18, 7:15] = 1.0
    image[:, 1, 12:23, 17:25] = 0.7
    shifted = torch.zeros_like(image)
    shifted[..., 4:] = image[..., :-4]
    first = _fixed_raw_motion_descriptor(image, 16)
    second = _fixed_raw_motion_descriptor(shifted, 16)
    identity_error, observable = _fixed_observable_motion(first, second)
    correct_flow = torch.zeros(1, 2, 16, 16)
    correct_flow[:, 0] = 2.0
    aligned, valid = warp_patch_grid(second, correct_flow)
    aligned_error = _stable_sqrt(
        (first.float() - aligned.float()).square().mean(dim=1, keepdim=True),
        epsilon=1e-8,
    )
    weights = observable * valid.float()
    assert float(observable.mean()) > 0.0
    assert float((aligned_error * weights).sum() / weights.sum()) < float(
        (identity_error * weights).sum() / weights.sum()
    )

    static_identity, static_observable = _fixed_observable_motion(first, first)
    torch.testing.assert_close(static_identity, torch.zeros_like(static_identity))
    torch.testing.assert_close(static_observable, torch.zeros_like(static_observable))

    noisy_identity, noisy_observable = _fixed_observable_motion(
        first, first + 1e-4 * torch.randn_like(first)
    )
    assert float(noisy_identity.mean()) > 0.0
    assert float(noisy_observable.mean()) < 0.01


def test_v99_identity_advantage_is_zero_for_static_and_backpropagates_on_motion() -> None:
    torch.manual_seed(973)
    config = _raw_role_config(
        flow_jepa_zero_flow_guard=1, flow_jepa_raw_activation_checkpoint=0
    )
    encoder = FlowDINOEvidenceEncoder(config).train()
    visual_frame = torch.randn(
        1, 1, config.num_cameras, config.patches_per_camera, config.visual_token_dim
    )
    visual = visual_frame.expand(-1, config.visual_history_length, -1, -1, -1).clone()
    raw_frame = torch.rand(1, 1, config.num_cameras, 3, 64, 64)
    raw_static = raw_frame.expand(-1, config.visual_history_length, -1, -1, -1, -1).clone()

    static_pack = encoder(visual, raw_visual=raw_static)
    static_loss = static_pack.losses["flow_jepa_identity_advantage_loss"]
    torch.testing.assert_close(static_loss, torch.zeros_like(static_loss))
    static_loss.backward(retain_graph=True)
    assert encoder.raw_flow is not None
    static_grad = encoder.raw_flow.high.update[-1].weight.grad
    assert static_grad is not None
    torch.testing.assert_close(static_grad, torch.zeros_like(static_grad))

    encoder.zero_grad(set_to_none=True)
    static_identity_loss = static_pack.losses["flow_jepa_static_identity_loss"]
    assert float(static_identity_loss.detach()) >= 0.0
    static_identity_loss.backward()
    static_identity_grad = encoder.raw_flow.high.update[-1].weight.grad
    assert static_identity_grad is not None
    assert torch.isfinite(static_identity_grad).all()

    encoder.zero_grad(set_to_none=True)
    raw_moving = raw_static.clone()
    raw_moving[:, 1:] = raw_moving[:, 1:].roll(4, dims=-1)
    moving_pack = encoder(visual, raw_visual=raw_moving)
    moving_loss = moving_pack.losses["flow_jepa_identity_advantage_loss"]
    assert float(moving_loss.detach()) > 0.0
    moving_loss.backward()
    moving_grad = encoder.raw_flow.high.update[-1].weight.grad
    assert moving_grad is not None and torch.isfinite(moving_grad).all()
    assert float(moving_grad.abs().sum()) > 0.0


def test_guarded_raw_reader_fallback_is_not_a_duplicate_local_candidate_bank() -> None:
    reader = _RawDeformableAddressReader(
        8, 16, 4, radius=1, heads=4, nonduplicate_fallback=True
    ).eval()
    source = torch.randn(1, 8, 16, 16)
    target = torch.randn_like(source)
    selector, value, metrics = reader(
        source,
        target,
        torch.zeros(1, 2, 16, 16),
        torch.ones(1, 1, 16, 16),
        torch.zeros(1, 4, 4, 16),
        torch.ones(1, 1, 4, 4),
    )
    assert selector.shape == value.shape == (1, 4, 4, 16)
    assert int(metrics["candidate_count"]) == 10
    assert float(metrics["lane_value_difference"]) > 0.0
    torch.testing.assert_close(
        metrics["flow_mass"] + metrics["fallback_mass"], torch.ones(())
    )


def test_complementary_raw_reader_adds_base_and_flow_addressed_detail() -> None:
    torch.manual_seed(976)
    reader = _RawDeformableAddressReader(
        8,
        16,
        4,
        radius=1,
        heads=4,
        nonduplicate_fallback=True,
        complementary_detail=True,
    ).train()
    source = torch.randn(1, 8, 16, 16)
    target = torch.randn_like(source)
    flow = torch.zeros(1, 2, 16, 16, requires_grad=True)
    selector, value, metrics = reader(
        source,
        target,
        flow,
        torch.ones(1, 1, 16, 16),
        torch.randn(1, 4, 4, 16),
        torch.ones(1, 1, 4, 4),
    )
    assert selector.shape == value.shape == (1, 4, 4, 16)
    assert float(metrics["additive_detail_path"]) == 1.0
    assert int(metrics["candidate_count"]) == 10
    torch.testing.assert_close(
        metrics["flow_mass"] + metrics["fallback_mass"], torch.ones(())
    )
    assert 0.0 < float(metrics["flow_mass"]) < 1.0
    value.float().square().mean().backward()
    assert flow.grad is not None and torch.isfinite(flow.grad).all()
    assert float(flow.grad.abs().sum()) > 0.0


def test_complementary_raw_reader_post_output_detail_intervention_is_exact() -> None:
    torch.manual_seed(977)
    reader = _RawDeformableAddressReader(
        8,
        16,
        4,
        radius=1,
        heads=4,
        nonduplicate_fallback=True,
        complementary_detail=True,
    ).eval()
    source = torch.randn(2, 8, 16, 16)
    target = torch.randn_like(source)
    flow = torch.randn(2, 2, 16, 16) * 0.25
    confidence = torch.ones(2, 1, 16, 16)
    grounding = torch.randn(2, 4, 4, 16)
    detail_gate = torch.ones(2, 1, 4, 4)
    with torch.no_grad():
        full_selector, full_value, full_metrics = reader(
            source,
            target,
            flow,
            confidence,
            grounding,
            detail_gate,
            post_reader_detail_intervention="measure",
        )
        base_selector, base_value, zero_metrics = reader(
            source,
            target,
            flow,
            confidence,
            grounding,
            detail_gate,
            post_reader_detail_intervention="zero",
        )
        shuffled_selector, shuffled_value, shuffle_metrics = reader(
            source,
            target,
            flow,
            confidence,
            grounding,
            detail_gate,
            post_reader_detail_intervention="spatial_shuffle",
        )
    selector_residual = full_selector - base_selector
    value_residual = full_value - base_value
    torch.testing.assert_close(
        shuffled_selector,
        base_selector + selector_residual.roll(shifts=(2, 1), dims=(1, 2)),
    )
    torch.testing.assert_close(
        shuffled_value,
        base_value + value_residual.roll(shifts=(2, 1), dims=(1, 2)),
    )
    assert float(full_metrics["post_reader_detail_value_intervention_delta"]) == 0.0
    torch.testing.assert_close(
        zero_metrics["post_reader_detail_selector_intervention_delta"],
        zero_metrics["post_reader_detail_selector_residual_norm"],
    )
    torch.testing.assert_close(
        zero_metrics["post_reader_detail_value_intervention_delta"],
        zero_metrics["post_reader_detail_value_residual_norm"],
    )
    assert float(shuffle_metrics["post_reader_detail_value_intervention_delta"]) > 0.0


def test_complementary_raw_reader_can_return_exact_policy_detail_residual() -> None:
    torch.manual_seed(978)
    reader = _RawDeformableAddressReader(
        8,
        16,
        4,
        radius=1,
        heads=4,
        nonduplicate_fallback=True,
        complementary_detail=True,
    ).eval()
    source = torch.randn(2, 8, 16, 16)
    target = torch.randn_like(source)
    arguments = (
        source,
        target,
        torch.randn(2, 2, 16, 16) * 0.2,
        torch.ones(2, 1, 16, 16),
        torch.randn(2, 4, 4, 16),
        torch.ones(2, 1, 4, 4),
    )
    with torch.no_grad():
        full_selector, full_value, _, selector_detail, value_detail = reader(
            *arguments,
            return_detail_residual=True,
        )
        base_selector, base_value, _ = reader(
            *arguments,
            post_reader_detail_intervention="zero",
        )
    assert selector_detail is not None and value_detail is not None
    torch.testing.assert_close(selector_detail, full_selector - base_selector)
    torch.testing.assert_close(value_detail, full_value - base_value)


def test_v102_raw_detail_bank_is_observation_only_and_reusable() -> None:
    torch.manual_seed(102)
    config = _v102_config()
    encoder = FlowDINOEvidenceEncoder(config).eval()
    with torch.no_grad():
        pack = encoder(
            _visual(config, batch=1),
            raw_visual=_raw_visual(config, batch=1),
        )
        rollout_count = (
            config.future_anchors
            * config.num_cameras
            * config.flow_jepa_grid_size**2
        )
        slices = {"rollout": slice(1, 1 + rollout_count)}
        first = encoder.refine_raw_evidence(
            pack,
            torch.randn(1, rollout_count + 1, config.hidden_size),
            slices,
            return_late_detail=True,
        )
        second = encoder.refine_raw_evidence(
            pack,
            100.0 * torch.randn(1, rollout_count + 1, config.hidden_size),
            slices,
            return_late_detail=True,
        )
    first_selector, first_value, first_metrics, first_detail = first
    second_selector, second_value, _, second_detail = second
    assert pack.raw_context is None
    assert pack.late_raw_detail is first_detail
    assert pack.late_raw_detail_metrics is first_metrics
    assert first_detail is not None and second_detail is not None
    torch.testing.assert_close(first_selector, pack.selector_tokens)
    torch.testing.assert_close(first_value, pack.value_tokens)
    torch.testing.assert_close(second_selector, pack.selector_tokens)
    torch.testing.assert_close(second_value, pack.value_tokens)
    torch.testing.assert_close(
        first_detail.selector_tokens, second_detail.selector_tokens
    )
    torch.testing.assert_close(
        first_detail.value_tokens, second_detail.value_tokens
    )
    assert float(first_metrics["flow_jepa_raw_detail_deferred_to_policy"]) == 1.0
    assert (
        float(first_metrics["flow_jepa_raw_detail_action_independent_compile"])
        == 1.0
    )


def test_v102_training_preview_does_not_consume_trainable_detail_cache() -> None:
    torch.manual_seed(106)
    config = _v102_config()
    encoder = FlowDINOEvidenceEncoder(config).train()
    pack = encoder(
        _visual(config, batch=1),
        raw_visual=_raw_visual(config, batch=1),
    )
    rollout_count = (
        config.future_anchors
        * config.num_cameras
        * config.flow_jepa_grid_size**2
    )
    slices = {"rollout": slice(1, 1 + rollout_count)}
    canvas = torch.randn(1, rollout_count + 1, config.hidden_size)
    with torch.no_grad():
        preview = encoder.refine_raw_evidence(
            pack,
            canvas,
            slices,
            return_late_detail=True,
        )
    assert preview[-1] is not None
    assert pack.raw_context is not None
    assert pack.late_raw_detail is None
    main = encoder.refine_raw_evidence(
        pack,
        canvas,
        slices,
        return_late_detail=True,
    )
    main_detail = main[-1]
    assert main_detail is not None
    assert pack.raw_context is None
    assert pack.late_raw_detail is main_detail
    cached_a = encoder.refine_raw_evidence(
        pack,
        10.0 * canvas,
        slices,
        return_late_detail=True,
    )
    cached_b = encoder.refine_raw_evidence(
        pack,
        -10.0 * canvas,
        slices,
        return_late_detail=True,
    )
    assert cached_a[-1] is main_detail
    assert cached_b[-1] is main_detail
    main_detail.value_tokens.float().square().mean().backward()
    assert encoder.raw_flow is not None
    gradient = encoder.raw_flow.high.update[-1].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert bool(torch.count_nonzero(gradient.detach()))


def test_v102_five_step_sampling_compiles_raw_detail_once() -> None:
    torch.manual_seed(107)
    config = _v102_config(
        final_action_decoder="evidence_latent_mmdit_action",
        layer_contract_adapters=1,
        layer_contract_adapter_dim=32,
        latent_cvae_mmdit_depth=1,
        inference_steps=5,
    )
    system = V39PolicySystem(config).eval()
    encoder = system.planner.flow_dino_evidence
    assert encoder is not None and encoder.raw_address_reader is not None
    reader_calls = 0

    def count_reader(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        _output: object,
    ) -> None:
        nonlocal reader_calls
        reader_calls += 1

    handle = encoder.raw_address_reader.register_forward_hook(count_reader)
    try:
        with torch.no_grad():
            sampled = system.sample(
                _visual(config, batch=1),
                torch.randn(1, config.visual_history_length, config.state_dim),
                torch.randn(1, config.executed_history_length, config.action_dim),
                torch.randn(1, config.state_dim),
                raw_visual=_raw_visual(config, batch=1),
                steps=5,
                return_event_logits=True,
                collect_diagnostics=True,
            )
    finally:
        handle.remove()
    assert isinstance(sampled, dict)
    # One main compile plus its zero-flow and shuffled-flow diagnostics.
    assert reader_calls == 3
    for key in (
        "sample_flow_jepa_late_detail_update_norm",
        "sample_flow_jepa_late_detail_trajectory_ratio",
        "sample_flow_jepa_world_spatial_residual_norm",
    ):
        assert key in sampled
        assert torch.isfinite(sampled[key])


def test_v102_late_detail_reader_has_exact_zero_and_direct_gradient() -> None:
    torch.manual_seed(103)
    config = _v102_config()
    reader = LateRawDetailPolicyReader(config)
    batch = 2
    trajectory = torch.randn(
        batch,
        config.action_horizon * config.action_basis_tokens,
        config.hidden_size,
    )
    rollout = torch.randn(
        batch,
        config.future_anchors
        * config.num_cameras
        * config.future_grid_size**2,
        config.hidden_size,
    )
    detail_tokens = config.num_cameras * config.future_grid_size**2
    selector = torch.randn(batch, detail_tokens, config.hidden_size)
    zero_detail = LateRawDetailEvidence(
        selector_tokens=selector,
        value_tokens=torch.zeros_like(selector),
    )
    zero_output, zero_metrics = reader(trajectory, rollout, zero_detail)
    torch.testing.assert_close(zero_output, trajectory)
    assert float(zero_metrics["flow_jepa_late_detail_update_norm"]) == 0.0

    values = torch.randn_like(selector, requires_grad=True)
    output, metrics = reader(
        trajectory,
        rollout,
        LateRawDetailEvidence(selector_tokens=selector, value_tokens=values),
    )
    assert float(metrics["flow_jepa_late_detail_update_norm"]) > 0.0
    output.float().square().mean().backward()
    assert values.grad is not None and torch.isfinite(values.grad).all()
    assert float(values.grad.abs().sum()) > 0.0
    assert reader.query_proj.weight.grad is not None
    assert reader.key_proj.weight.grad is not None


def test_soft_address_lattice_preserves_slots_and_candidates_until_policy_query() -> None:
    torch.manual_seed(103)
    config = _v103_config(flow_jepa_raw_activation_checkpoint=0)
    encoder = FlowDINOEvidenceEncoder(config).train()
    pack = encoder(
        _visual(config, batch=1),
        raw_visual=_raw_visual(config, batch=1),
    )
    rollout_count = (
        config.future_anchors
        * config.num_cameras
        * config.flow_jepa_grid_size**2
    )
    grounding_canvas = torch.randn(
        1, rollout_count + 2, config.hidden_size
    )
    selector, values, metrics, detail = encoder.refine_raw_evidence(
        pack,
        grounding_canvas,
        {"rollout": slice(2, 2 + rollout_count)},
        return_late_detail=True,
    )
    assert detail is not None and detail.address_bank is not None
    bank = detail.address_bank
    candidates = (2 * config.flow_jepa_raw_reader_radius + 1) ** 2
    assert bank.coarse_keys.shape == (
        1,
        config.num_cameras,
        config.flow_jepa_grid_size,
        config.flow_jepa_grid_size,
        config.flow_jepa_address_slots,
        config.flow_jepa_address_route_dim,
    )
    assert bank.fine_keys.shape[-2:] == (
        candidates,
        config.flow_jepa_address_route_dim,
    )
    assert bank.fine_values.shape[-2:] == (
        candidates,
        config.flow_jepa_raw_base_channels
        + config.flow_jepa_raw_base_channels // 2,
    )
    assert selector.shape == pack.selector_tokens.shape
    assert values.shape == pack.value_tokens.shape
    assert float(metrics["flow_jepa_address_lattice_enabled"]) == 1.0
    assert float(metrics["flow_jepa_address_source_raw_match_active"]) == 1.0
    assert int(metrics["flow_jepa_address_slot_count"]) == 3
    assert (
        float(metrics["flow_jepa_address_slot_pair_distance_normalized"])
        > 0.0
    )
    assert float(metrics["flow_jepa_address_slot_posterior_hellinger"]) > 0.0
    assert float(metrics["flow_jepa_address_flow_prior_floor_sigma"]) == 2.0
    assert (
        float(metrics["flow_jepa_address_flow_prior_floor_logit_span"]) > 0.0
    )

    reader = LateRawDetailPolicyReader(config).train()
    trajectory = torch.randn(
        1,
        config.action_horizon * config.action_basis_tokens,
        config.hidden_size,
    )
    rollout = torch.randn(
        1, rollout_count, config.hidden_size
    )
    updated, reader_metrics = reader(trajectory, rollout, detail)
    assert updated.shape == trajectory.shape
    assert bool(torch.isfinite(updated).all())
    assert 0.0 <= float(reader_metrics["flow_jepa_address_camera_entropy"]) <= 1.0
    assert (
        1.0
        <= float(
            reader_metrics[
                "flow_jepa_address_policy_slot_effective_count"
            ]
        )
        <= float(config.flow_jepa_address_slots)
    )
    (updated - trajectory).float().square().mean().backward()
    assert encoder.soft_address_compiler is not None
    assert encoder.raw_flow is not None
    assert (
        encoder.soft_address_compiler.source_dino[1].weight.grad is not None
    )
    assert encoder.soft_address_compiler.raw_key[1].weight.grad is not None
    assert encoder.soft_address_compiler.source_raw_key[1].weight.grad is not None
    assert encoder.soft_address_compiler.raw_pair_key[-1].weight.grad is not None
    assert encoder.raw_flow.high.update[-1].weight.grad is not None
    assert reader.lattice_query_proj is not None
    assert reader.lattice_query_proj.weight.grad is not None
    assert reader.lattice_world_key_proj is not None
    assert reader.lattice_world_key_proj.weight.grad is not None


def test_soft_address_source_raw_can_correct_fine_keys_without_rewriting_values() -> None:
    torch.manual_seed(117)
    config = _v103_config(flow_jepa_raw_activation_checkpoint=0)
    raw_dim = (
        config.flow_jepa_raw_base_channels
        + config.flow_jepa_raw_base_channels // 2
    )
    compiler = _SoftMultiResolutionAddressCompiler(
        config,
        raw_dim=raw_dim,
    ).train()
    batch = 1
    dino_side = int(round(float(config.patches_per_camera) ** 0.5))
    raw_side = 16
    source_dino = torch.randn(
        batch,
        config.num_cameras,
        dino_side,
        dino_side,
        config.visual_token_dim,
    )
    target_dino = torch.randn_like(source_dino)
    source_raw = torch.randn(
        batch,
        config.num_cameras,
        raw_dim,
        raw_side,
        raw_side,
        requires_grad=True,
    )
    target_raw = torch.randn_like(source_raw, requires_grad=True)
    flow = torch.zeros(
        batch, config.num_cameras, 2, raw_side, raw_side
    )
    confidence = torch.full(
        (batch, config.num_cameras, 1, raw_side, raw_side), 0.5
    )
    uncertainty = torch.full_like(confidence, 0.1)
    occlusion = torch.zeros_like(confidence)
    first, metrics = compiler(
        source_dino=source_dino,
        target_dino=target_dino,
        source_raw=source_raw,
        target_raw=target_raw,
        flow=flow,
        confidence=confidence,
        uncertainty=uncertainty,
        occlusion=occlusion,
    )
    shifted, _ = compiler(
        source_dino=source_dino,
        target_dino=target_dino,
        source_raw=source_raw.detach().roll(shifts=3, dims=-1),
        target_raw=target_raw.detach(),
        flow=flow,
        confidence=confidence,
        uncertainty=uncertainty,
        occlusion=occlusion,
    )
    assert not torch.allclose(first.fine_keys, shifted.fine_keys)
    # Raw values remain target-owned; source appearance changes only the
    # selector evidence used to choose them.
    torch.testing.assert_close(first.fine_values, shifted.fine_values)
    assert float(metrics["flow_jepa_address_source_raw_match_active"]) == 1.0
    first.fine_keys.float().square().mean().backward()
    assert source_raw.grad is not None
    assert float(source_raw.grad.abs().sum()) > 0.0
    assert compiler.source_raw_key[1].weight.grad is not None
    assert compiler.raw_pair_key[-1].weight.grad is not None


def test_soft_address_reader_uses_world_xy_state_before_precision_read() -> None:
    """A mean-preserving W-chart change must still alter the raw address read."""

    torch.manual_seed(114)
    config = _v103_config(flow_jepa_raw_activation_checkpoint=0)
    encoder = FlowDINOEvidenceEncoder(config).eval()
    with torch.no_grad():
        pack = encoder(
            _visual(config, batch=1),
            raw_visual=_raw_visual(config, batch=1),
        )
        rollout_count = (
            config.future_anchors
            * config.num_cameras
            * config.flow_jepa_grid_size**2
        )
        _, _, _, detail = encoder.refine_raw_evidence(
            pack,
            torch.randn(1, rollout_count + 2, config.hidden_size),
            {"rollout": slice(2, 2 + rollout_count)},
            return_late_detail=True,
        )
    assert detail is not None and detail.address_bank is not None
    reader = LateRawDetailPolicyReader(config).train()
    trajectory = torch.randn(
        1,
        config.action_horizon * config.action_basis_tokens,
        config.hidden_size,
    )
    rollout = torch.randn(
        1,
        config.future_anchors,
        config.num_cameras,
        config.future_grid_size,
        config.future_grid_size,
        config.hidden_size,
        requires_grad=True,
    )
    shifted_rollout = rollout.detach().roll(shifts=1, dims=3)
    torch.testing.assert_close(
        rollout.detach().mean(dim=(3, 4)),
        shifted_rollout.mean(dim=(3, 4)),
    )
    output, metrics = reader(
        trajectory,
        rollout.reshape(1, rollout_count, config.hidden_size),
        detail,
    )
    shifted_output, _ = reader(
        trajectory,
        shifted_rollout.reshape(1, rollout_count, config.hidden_size),
        detail,
    )
    assert not torch.allclose(output, shifted_output)
    assert float(metrics["flow_jepa_address_world_spatial_logit_std"]) > 0.0

    (output - trajectory).float().square().mean().backward()
    assert rollout.grad is not None
    rollout_grad = rollout.grad.float()
    spatial_grad = rollout_grad - rollout_grad.mean(dim=(3, 4), keepdim=True)
    assert float(spatial_grad.abs().sum()) > 0.0
    assert reader.lattice_world_key_proj is not None
    assert reader.lattice_world_key_proj.weight.grad is not None
    assert float(reader.lattice_world_key_proj.weight.grad.abs().sum()) > 0.0


def test_soft_address_lattice_zero_detail_is_an_exact_zero_update() -> None:
    torch.manual_seed(104)
    config = _v103_config(flow_jepa_raw_activation_checkpoint=0)
    encoder = FlowDINOEvidenceEncoder(config).eval()
    with torch.no_grad():
        pack = encoder(
            _visual(config, batch=1),
            raw_visual=_raw_visual(config, batch=1),
        )
        rollout_count = (
            config.future_anchors
            * config.num_cameras
            * config.flow_jepa_grid_size**2
        )
        _, _, _, detail = encoder.refine_raw_evidence(
            pack,
            torch.randn(1, rollout_count + 2, config.hidden_size),
            {"rollout": slice(2, 2 + rollout_count)},
            return_late_detail=True,
        )
        assert detail is not None and detail.address_bank is not None
        zero_bank = replace(
            detail.address_bank,
            fine_values=torch.zeros_like(detail.address_bank.fine_values),
        )
        zero_detail = replace(detail, address_bank=zero_bank)
        trajectory = torch.randn(
            1,
            config.action_horizon * config.action_basis_tokens,
            config.hidden_size,
        )
        rollout = torch.randn(1, rollout_count, config.hidden_size)
        updated, _ = LateRawDetailPolicyReader(config).eval()(
            trajectory, rollout, zero_detail
        )
    torch.testing.assert_close(updated, trajectory)


def test_soft_address_posterior_interventions_are_transient_and_reach_reader() -> None:
    torch.manual_seed(113)
    config = _v103_config(flow_jepa_raw_activation_checkpoint=0)
    encoder = FlowDINOEvidenceEncoder(config).eval()
    with torch.no_grad():
        pack = encoder(
            _visual(config, batch=1),
            raw_visual=_raw_visual(config, batch=1),
        )
        rollout_count = (
            config.future_anchors
            * config.num_cameras
            * config.flow_jepa_grid_size**2
        )
        _, _, _, detail = encoder.refine_raw_evidence(
            pack,
            torch.randn(1, rollout_count + 2, config.hidden_size),
            {"rollout": slice(2, 2 + rollout_count)},
            return_late_detail=True,
        )
        assert detail is not None and detail.address_bank is not None
        trajectory = torch.randn(
            1,
            config.action_horizon * config.action_basis_tokens,
            config.hidden_size,
        )
        rollout = torch.randn(1, rollout_count, config.hidden_size)
        reader = LateRawDetailPolicyReader(config).eval()
        baseline, _ = reader(trajectory, rollout, detail)
        original_state = {
            key: value.detach().clone() for key, value in reader.state_dict().items()
        }
        expected_metric = {
            "address_posterior_uniform": "address_posterior_l1_delta",
            "fine_offset_zero": "fine_posterior_l1_delta",
            "camera_posterior_uniform": "address_posterior_l1_delta",
            "camera_swap": "camera_bank_value_delta_norm",
        }
        changed = {}
        for mode, metric in expected_metric.items():
            reader.set_address_eval_intervention(mode)
            output, _ = reader(trajectory, rollout, detail)
            state = reader.address_eval_intervention_state()
            assert int(state["apply_count"]) == 1
            assert float(state[metric]) > 0.0
            changed[mode] = float((output - baseline).abs().max())
            reader.clear_address_eval_intervention()
        assert all(value > 0.0 for value in changed.values())
        ordinary, _ = reader(trajectory, rollout, detail)
        torch.testing.assert_close(ordinary, baseline, rtol=0.0, atol=0.0)
        for key, value in reader.state_dict().items():
            torch.testing.assert_close(
                value, original_state[key], rtol=0.0, atol=0.0
            )


def test_v103_condition_phase_and_typed_delta_interventions_hit_real_sample_path() -> None:
    torch.manual_seed(114)
    config = _typed_332_config(
        flow_jepa_raw_activation_checkpoint=0,
        action_history_enabled=1,
        action_history_recent_tokens=1,
        action_history_summary_tokens=1,
        action_history_condition_exact_null=1,
        action_history_proposal_detach=0,
        goal_conditioning_enabled=1,
        goal_token_count=2,
        goal_language_dim=16,
        goal_language_max_tokens=5,
        goal_resampler_depth=1,
        goal_condition_exact_null=1,
        stateless_phase_enabled=1,
        stateless_phase_count=4,
        inference_steps=1,
    )
    system = V39PolicySystem(config).eval()
    batch = 2
    visual = _visual(config, batch=batch)
    raw_visual = _raw_visual(config, batch=batch)
    history_state = torch.randn(
        batch, config.visual_history_length, config.state_dim
    )
    executed = torch.randn(
        batch, config.executed_history_length, config.action_dim
    )
    state = torch.randn(batch, config.state_dim)
    goal = torch.randn(batch, 3, config.goal_language_dim)
    goal_mask = torch.ones(batch, 3, dtype=torch.bool)
    noise = torch.randn(batch, config.action_horizon, config.action_dim)
    frozen_state = {
        key: value.detach().clone() for key, value in system.state_dict().items()
    }

    def run() -> torch.Tensor:
        output = system.sample(
            visual,
            history_state,
            executed,
            state,
            raw_visual=raw_visual,
            noise=noise,
            steps=1,
            goal_language_tokens=goal,
            goal_language_mask=goal_mask,
        )
        assert torch.is_tensor(output)
        assert torch.isfinite(output).all()
        return output

    baseline = run()
    system.planner.set_action_path_eval_intervention("none")
    matched_baseline = run()
    route_state = system.planner.action_path_eval_intervention_state()
    torch.testing.assert_close(
        matched_baseline, baseline, rtol=0.0, atol=0.0
    )
    for key in (
        "attnres_ground_to_world_source_effective_count",
        "attnres_ground_to_world_anchor_route_std",
        "attnres_ground_to_world_camera_route_std",
        "attnres_world_to_policy_source_effective_count",
        "attnres_world_to_policy_horizon_route_std",
        "attnres_world_to_policy_basis_route_std",
        "evidence_policy_delta_attnres_source_effective_count",
        "evidence_policy_delta_attnres_horizon_route_std",
        "evidence_protected_detail_basis_source_effective_count",
        "evidence_protected_detail_basis_horizon_route_std",
    ):
        assert key in route_state
        assert math.isfinite(float(route_state[key]))
    system.planner.clear_action_path_eval_intervention()
    for mode in (
        "goal_zero",
        "goal_batch_shuffle",
        "history_zero",
        "history_condition_zero",
        "history_proposal_zero",
        "history_proposal_batch_shuffle",
        "history_batch_shuffle",
        "history_truncate",
    ):
        system.set_condition_eval_intervention(mode)
        changed = run()
        intervention_state = system.condition_eval_intervention_state()
        assert int(intervention_state["apply_count"]) == 1
        assert float((changed - baseline).abs().max()) > 0.0
        system.clear_condition_eval_intervention()

    for mode in (
        "phase_zero",
        "phase_batch_shuffle",
        "condition_query_zero",
        "g1_zero",
        "g2_shuffle",
        "grounding_entry_zero",
        "w1_shuffle",
        "world_to_policy_zero",
        "w2p_far_context_zero",
        "w2p_far_context_shuffle",
        "bottom_far_rollout_zero",
        "bottom_far_rollout_shuffle",
        "all_far_context_zero",
        "all_far_context_shuffle",
        "p1_zero",
        "protected_detail_zero",
    ):
        system.planner.set_action_path_eval_intervention(mode)
        changed = run()
        intervention_state = (
            system.planner.action_path_eval_intervention_state()
        )
        assert int(intervention_state["apply_count"]) > 0
        assert float((changed - baseline).abs().max()) > 0.0
        system.planner.clear_action_path_eval_intervention()

    encoder = system.planner.flow_dino_evidence
    assert encoder is not None
    for mode, expected_code, delta_keys in (
        ("zero", 1.0, ("flow_jepa_raw_flow_intervention_delta_norm",)),
        ("shuffle", 2.0, ("flow_jepa_raw_flow_intervention_delta_norm",)),
        (
            "spatial_shuffle",
            3.0,
            ("flow_jepa_raw_flow_intervention_delta_norm",),
        ),
        (
            "detail_zero",
            4.0,
            ("flow_jepa_raw_value_intervention_delta_norm",),
        ),
        (
            "detail_spatial_shuffle",
            5.0,
            ("flow_jepa_raw_value_intervention_delta_norm",),
        ),
        (
            "dino_key_spatial_shuffle",
            6.0,
            ("flow_jepa_dino_key_intervention_delta_norm",),
        ),
        (
            "source_raw_key_zero",
            7.0,
            ("flow_jepa_source_raw_key_intervention_delta_norm",),
        ),
        (
            "source_raw_key_spatial_shuffle",
            8.0,
            ("flow_jepa_source_raw_key_intervention_delta_norm",),
        ),
        (
            "joint_address_key_spatial_shuffle",
            9.0,
            (
                "flow_jepa_raw_flow_intervention_delta_norm",
                "flow_jepa_dino_key_intervention_delta_norm",
                "flow_jepa_source_raw_key_intervention_delta_norm",
            ),
        ),
    ):
        encoder.set_raw_address_eval_intervention(mode)
        changed = run()
        intervention_metrics = encoder.raw_address_eval_metrics()
        assert intervention_metrics[
            "flow_jepa_raw_address_intervention_code"
        ] == expected_code
        for delta_key in delta_keys:
            assert float(intervention_metrics[delta_key]) > 0.0
        assert float((changed - baseline).abs().max()) > 0.0
        encoder.clear_raw_address_eval_intervention()

    late_reader = system.planner.late_raw_detail_reader
    assert late_reader is not None
    for mode in (
        "address_posterior_uniform",
        "fine_offset_zero",
        "camera_posterior_uniform",
        "camera_swap",
        "world_query_zero",
        "world_query_spatial_shuffle",
    ):
        late_reader.set_address_eval_intervention(mode)
        changed = run()
        intervention_state = late_reader.address_eval_intervention_state()
        assert int(intervention_state["apply_count"]) > 0
        assert float((changed - baseline).abs().max()) > 0.0
        if mode.startswith("world_query_"):
            assert float(intervention_state["world_query_input_delta_norm"]) > 0.0
        late_reader.clear_address_eval_intervention()

    for key, value in system.state_dict().items():
        torch.testing.assert_close(value, frozen_state[key], rtol=0.0, atol=0.0)


def test_v103_proposal_interventions_isolate_proposal_from_direct_history() -> None:
    torch.manual_seed(130)
    config = _typed_332_config(
        flow_jepa_raw_activation_checkpoint=0,
        action_history_enabled=1,
        action_history_recent_tokens=1,
        action_history_summary_tokens=1,
        action_history_condition_exact_null=1,
        action_history_proposal_detach=0,
        inference_steps=1,
    )
    system = V39PolicySystem(config).eval()
    batch = 2
    visual = _visual(config, batch=batch)
    raw_visual = _raw_visual(config, batch=batch)
    history_state = torch.randn(
        batch, config.visual_history_length, config.state_dim
    )
    executed = torch.randn(
        batch, config.executed_history_length, config.action_dim
    )
    state = torch.randn(batch, config.state_dim)
    noise = torch.randn(batch, config.action_horizon, config.action_dim)
    frozen_state = {
        key: value.detach().clone() for key, value in system.state_dict().items()
    }

    def run(
        mode: str | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, str | int | float]]:
        captured: dict[str, torch.Tensor] = {}

        def capture_seed(_module, _args, kwargs) -> None:
            for key in (
                "executed_history",
                "executed_memory",
                "proposal_tokens",
                "proposal_keep",
            ):
                captured[key] = kwargs[key].detach().clone()

        handle = system.planner.seed.register_forward_pre_hook(
            capture_seed, with_kwargs=True
        )
        if mode is not None:
            system.set_condition_eval_intervention(mode)
        try:
            output = system.sample(
                visual,
                history_state,
                executed,
                state,
                raw_visual=raw_visual,
                noise=noise,
                steps=1,
            )
            assert torch.is_tensor(output)
            intervention_state = system.condition_eval_intervention_state()
        finally:
            handle.remove()
            system.clear_condition_eval_intervention()
        return output, captured, intervention_state

    baseline, baseline_seed, _ = run()
    zero, zero_seed, zero_state = run("history_proposal_zero")
    assert int(zero_state["apply_count"]) == 1
    assert float(zero_state["history_proposal_keep_delta"]) == 1.0
    for key in ("executed_history", "executed_memory", "proposal_tokens"):
        torch.testing.assert_close(
            zero_seed[key], baseline_seed[key], rtol=0.0, atol=0.0
        )
    torch.testing.assert_close(
        zero_seed["proposal_keep"],
        torch.zeros_like(zero_seed["proposal_keep"]),
        rtol=0.0,
        atol=0.0,
    )
    assert float((zero - baseline).abs().max()) > 0.0

    shuffled, shuffled_seed, shuffled_state = run(
        "history_proposal_batch_shuffle"
    )
    assert int(shuffled_state["apply_count"]) == 1
    assert float(shuffled_state["history_proposal_input_delta_norm"]) > 0.0
    assert (
        float(shuffled_state["history_proposal_shuffle_temporal_fallback"])
        == 0.0
    )
    for key in ("executed_history", "executed_memory", "proposal_keep"):
        torch.testing.assert_close(
            shuffled_seed[key], baseline_seed[key], rtol=0.0, atol=0.0
        )
    torch.testing.assert_close(
        shuffled_seed["proposal_tokens"],
        baseline_seed["proposal_tokens"].roll(shifts=1, dims=0),
        rtol=0.0,
        atol=0.0,
    )
    assert float((shuffled - baseline).abs().max()) > 0.0

    restored, restored_seed, _ = run()
    torch.testing.assert_close(restored, baseline, rtol=0.0, atol=0.0)
    for key, value in baseline_seed.items():
        torch.testing.assert_close(
            restored_seed[key], value, rtol=0.0, atol=0.0
        )
    for key, value in system.state_dict().items():
        torch.testing.assert_close(value, frozen_state[key], rtol=0.0, atol=0.0)


def test_source_raw_probe_changes_only_fine_keys_before_policy_read() -> None:
    torch.manual_seed(119)
    config = _typed_332_config(flow_jepa_raw_activation_checkpoint=0)
    encoder = FlowDINOEvidenceEncoder(config).eval()
    visual = _visual(config, batch=2)
    raw_visual = _raw_visual(config, batch=2)
    rollout_count = (
        config.future_anchors
        * config.num_cameras
        * config.flow_jepa_grid_size**2
    )
    canvas = torch.randn(2, rollout_count + 2, config.hidden_size)
    slices = {"rollout": slice(2, 2 + rollout_count)}
    frozen_state = {
        key: value.detach().clone() for key, value in encoder.state_dict().items()
    }

    def compile_bank(mode: str) -> tuple[SoftAddressLatticeBank, dict[str, float]]:
        encoder.set_raw_address_eval_intervention(mode)
        try:
            with torch.no_grad():
                pack = encoder(visual, raw_visual=raw_visual)
                detail = encoder.refine_raw_evidence(
                    pack,
                    canvas,
                    slices,
                    return_late_detail=True,
                )[-1]
            assert detail is not None and detail.address_bank is not None
            return detail.address_bank, encoder.raw_address_eval_metrics()
        finally:
            encoder.clear_raw_address_eval_intervention()

    baseline, _ = compile_bank("none")
    changed, metrics = compile_bank("source_raw_key_spatial_shuffle")
    torch.testing.assert_close(
        changed.coarse_keys, baseline.coarse_keys, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        changed.fine_values, baseline.fine_values, rtol=0.0, atol=0.0
    )
    assert float((changed.fine_keys - baseline.fine_keys).abs().max()) > 0.0
    assert (
        metrics["flow_jepa_source_raw_key_intervention_delta_norm"] > 0.0
    )
    assert encoder.raw_address_eval_metrics() == {}
    for key, value in encoder.state_dict().items():
        torch.testing.assert_close(value, frozen_state[key], rtol=0.0, atol=0.0)


def test_model_path_acceptance_matrix_separates_access_from_utility() -> None:
    baseline = np.zeros((2, 2, 1), dtype=np.float32)
    typed_only = baseline + 1.0
    bottom_only = baseline + 2.0
    joint = baseline + 4.0
    paired = {
        "source_raw_match_zero": {
            "action_delta_rmse": 0.25,
            "mse_delta_ci": {"ci95_low": -0.1, "ci95_high": 0.2},
        },
        "w2p_far_context_zero": {
            "action_delta_rmse": 1.0,
            "mse_delta_ci": {"ci95_low": 0.1, "ci95_high": 0.3},
        },
        "bottom_far_rollout_zero": {
            "action_delta_rmse": 2.0,
            "mse_delta_ci": {"ci95_low": -0.3, "ci95_high": -0.1},
        },
        "all_far_context_zero": {
            "action_delta_rmse": 4.0,
            "mse_delta_ci": {"ci95_low": -0.1, "ci95_high": 0.1},
        },
        "action_history_proposal_zero": {
            "action_delta_rmse": 0.5,
            "mse_delta_ci": {"ci95_low": -0.2, "ci95_high": 0.1},
        },
    }
    matrix = _model_path_acceptance_matrix(
        joined={
            "baseline": baseline,
            "source_raw_match_zero": baseline + 0.25,
            "w2p_far_context_zero": typed_only,
            "bottom_far_rollout_zero": bottom_only,
            "all_far_context_zero": joint,
            "action_history_proposal_zero": baseline + 0.5,
        },
        paired=paired,
        verification_counts={mode: 1 for mode in paired},
        boundary_diagnostics={
            "source_raw_match_zero": {
                "flow_jepa_source_raw_key_intervention_delta_norm": 2.0
            },
            "w2p_far_context_zero": {"w2p_far_context_delta_norm": 1.0},
            "bottom_far_rollout_zero": {
                "bottom_far_rollout_delta_norm": 2.0
            },
            "all_far_context_zero": {
                "w2p_far_context_delta_norm": 1.0,
                "bottom_far_rollout_delta_norm": 2.0,
            },
            "action_history_proposal_zero": {
                "history_proposal_keep_delta": 1.0
            },
        },
        baseline_identity_max_abs_delta=0.0,
        representation={
            "flow_jepa_address_slot_count": 4.0,
            "flow_jepa_address_slot_pair_distance_normalized": 0.1,
            "flow_jepa_address_slot_posterior_hellinger": 0.2,
            "flow_jepa_address_policy_slot_effective_count": 2.5,
            "flow_jepa_address_policy_slot_query_variation": 0.03,
            "attnres_ground_to_world_source_effective_count": 2.1,
            "attnres_ground_to_world_anchor_route_std": 0.02,
            "attnres_ground_to_world_camera_route_std": 0.03,
            "attnres_world_to_policy_source_effective_count": 3.1,
            "attnres_world_to_policy_horizon_route_std": 0.04,
            "attnres_world_to_policy_basis_route_std": 0.05,
            "evidence_policy_delta_attnres_source_effective_count": 2.4,
            "evidence_policy_delta_attnres_horizon_route_std": 0.06,
            "evidence_protected_detail_basis_source_effective_count": 2.2,
            "evidence_protected_detail_basis_horizon_route_std": 0.07,
        },
    )
    assert matrix["replay"]["numerically_identical"]
    assert matrix["aggregate"]["spatial_boundary_changed"]
    assert matrix["aggregate"]["spatial_path_reaches_action"]
    assert matrix["aggregate"]["history_proposal_boundary_changed"]
    assert matrix["aggregate"]["history_proposal_path_reaches_action"]
    assert matrix["aggregate"]["history_condition_path_reaches_action"] is None
    assert matrix["address_slot_structure"]["observed"]
    assert matrix["address_slot_structure"][
        "coarse_posteriors_numerically_distinct"
    ]
    assert matrix["address_slot_structure"][
        "policy_uses_multiple_slots_numerically"
    ]
    assert matrix["typed_route_structure"]["ground_to_world"]["observed"]
    assert matrix["typed_route_structure"]["world_to_policy"][
        "query_axes_vary_numerically"
    ]
    assert matrix["typed_route_structure"]["policy_to_mmdit"][
        "uses_multiple_sources_numerically"
    ]
    assert matrix["typed_route_structure"]["protected_detail_basis"][
        "query_axes_vary_numerically"
    ]
    assert matrix["long_horizon_pairwise"][
        "joint_distinguishable_from_each_single_path"
    ]
    rows = matrix["modes"]
    assert rows["source_raw_match_zero"]["utility_direction"] == "inconclusive"
    assert (
        rows["w2p_far_context_zero"]["utility_direction"]
        == "ablation_harmful_path_helpful"
    )
    assert (
        rows["bottom_far_rollout_zero"]["utility_direction"]
        == "ablation_helpful_path_harmful"
    )


def test_role_delta_attnres_zero_values_are_exact_and_gradients_are_natural() -> None:
    torch.manual_seed(105)
    route = RoleDeltaAttnRes(32, 16, max_sources=6)
    query = torch.randn(2, 4, 32, requires_grad=True)
    values = torch.randn(2, 4, 5, 32, requires_grad=True)
    routed, metrics = route(query, values)
    assert routed.shape == query.shape
    assert 0.0 <= float(metrics["null_mass"]) <= 1.0
    assert 1.0 <= float(metrics["source_effective_count"]) <= 6.0
    assert 1.0 <= float(metrics["candidate_effective_count"]) <= 6.0
    assert "query_axis_1_route_std" in metrics
    routed.float().square().mean().backward()
    assert query.grad is not None and float(query.grad.abs().sum()) > 0.0
    assert values.grad is not None and float(values.grad.abs().sum()) > 0.0
    assert route.query_proj.weight.grad is not None
    assert route.key_proj.weight.grad is not None

    zero, _ = route(query.detach(), torch.zeros_like(values.detach()))
    assert torch.equal(zero, torch.zeros_like(zero))

    protected_route = RoleDeltaAttnRes(
        32,
        16,
        max_sources=3,
        include_null=False,
    )
    protected_values = torch.randn(2, 4, 3, 32, requires_grad=True)
    protected, protected_metrics = protected_route(
        query.detach(),
        protected_values,
    )
    assert protected_route.null_key is None
    assert float(protected_metrics["null_mass"]) == 0.0
    torch.testing.assert_close(
        protected_metrics["source_mass"].sum(),
        torch.ones(()),
    )
    protected.float().square().mean().backward()
    assert protected_values.grad is not None
    assert float(protected_values.grad.abs().sum()) > 0.0


def test_role_value_contract_bounds_amplitude_without_detach_or_hard_clip() -> None:
    torch.manual_seed(130)
    raw = (1000.0 * torch.randn(2, 4, 5, 32)).requires_grad_()
    bounded, scale = smooth_rms_contract(raw, 1.0)
    bounded_rms = bounded.float().square().mean(dim=-1).sqrt()
    assert float(bounded_rms.max()) <= 1.0001
    assert 0.0 < float(scale.min()) < 1.0
    bounded.square().mean().backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()
    assert float(raw.grad.abs().sum()) > 0.0

    route = RoleDeltaAttnRes(
        32,
        16,
        max_sources=5,
        max_value_rms=1.0,
    )
    query = torch.randn(2, 4, 32, requires_grad=True)
    values = (1000.0 * torch.randn(2, 4, 5, 32)).requires_grad_()
    routed, metrics = route(query, values)
    assert float(metrics["value_contract_enabled"]) == 1.0
    assert float(metrics["raw_value_rms"]) > 100.0
    assert float(metrics["value_rms"]) <= 1.0001
    assert float(metrics["value_compression"]) > 0.9
    routed.square().mean().backward()
    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert values.grad is not None and torch.isfinite(values.grad).all()
    assert float(values.grad.abs().sum()) > 0.0


def test_typed_332_deltas_reach_action_without_fixed_policy_superhighway() -> None:
    torch.manual_seed(106)
    config = _typed_332_config(flow_jepa_raw_activation_checkpoint=0)
    system = V39PolicySystem(config).train()
    decoder = system.planner.evidence_latent_mmdit_action_decoder
    assert decoder is not None
    assert decoder.top_policy_workspace_lift is None
    assert decoder.policy_delta_attnres is not None
    assert decoder.protected_detail_basis_attnres is not None
    assert system.planner.world_to_policy_far_anchor_count == 1
    batch = 1
    output = system.flow_training_forward(
        _visual(config, batch=batch),
        torch.randn(batch, config.visual_history_length, config.state_dim),
        torch.randn(batch, config.executed_history_length, config.action_dim),
        torch.randn(batch, config.state_dim),
        torch.randn(batch, config.action_horizon, config.action_dim),
        target_visual=torch.randn(
            batch,
            config.future_anchors,
            config.visual_history_length,
            config.num_cameras,
            config.patches_per_camera,
            config.visual_token_dim,
        ),
        raw_visual=_raw_visual(config, batch=batch),
        make_counterfactuals=False,
    )
    assert float(output["evidence_policy_delta_bridge_enabled"]) == 1.0
    assert float(output["evidence_top_policy_workspace_fixed_fusion"]) == 0.0
    assert float(output["evidence_top_policy_protected_detail_update_norm"]) > 0.0
    for prefix in ("ground_to_world", "world_to_policy"):
        approved = output[f"attnres_{prefix}_approved_value_norm"].float()
        structured = output[f"attnres_{prefix}_structured_update_norm"].float()
        fixed_scale = output[f"attnres_{prefix}_fixed_scale"].float()
        torch.testing.assert_close(structured, approved * fixed_scale)
        assert float(approved) > float(structured)
    protected_basis_mass = sum(
        output[f"evidence_protected_detail_basis_mass_{index}"]
        for index in range(config.action_basis_tokens)
    )
    torch.testing.assert_close(
        protected_basis_mass.float(),
        torch.ones_like(protected_basis_mass.float()),
    )
    world_xy_updates = [
        output[f"attnres_observed_world_xy_update_norm_w{depth}"]
        for depth in range(1, 4)
    ]
    assert all(torch.isfinite(value) for value in world_xy_updates)
    assert float(sum(world_xy_updates)) > 0.0
    far_masses = [
        value
        for key, value in output.items()
        if key.startswith("attnres_world_to_policy_source_mass_")
        and "_far1_camera" in key
    ]
    assert len(far_masses) == (
        (config.flow_jepa_world_blocks + 1) * config.num_cameras
    )
    assert float(sum(far_masses)) > 0.0
    output["pred_physical_velocity"].float().square().mean().backward()
    active_routes = (
        system.planner.ground_to_world_attnres,
        system.planner.world_to_policy_attnres,
        decoder.policy_delta_attnres,
        decoder.protected_detail_basis_attnres,
    )
    assert all(route is not None for route in active_routes)
    for route in active_routes:
        assert route is not None
        assert route.query_proj.weight.grad is not None
        assert float(route.query_proj.weight.grad.abs().sum()) > 0.0
        assert route.key_proj.weight.grad is not None
        assert float(route.key_proj.weight.grad.abs().sum()) > 0.0
    for block_group in (
        system.planner.blocks[:3],
        system.planner.blocks[3:6],
        system.planner.blocks[6:],
    ):
        assert any(
            parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
            for block in block_group
            for parameter in block.parameters()
        )

    trainer = V39PolicyTrainerConfig(
        training_stage="policy",
        contract_mode="layer_adapter",
        lr=8e-5,
        single_stage_role_lr=1,
    )
    groups = _optimizer_groups(system, trainer)
    owned = [
        id(parameter)
        for group in groups
        for parameter in group["params"]
        if parameter.requires_grad
    ]
    trainable = {
        id(parameter) for parameter in system.parameters() if parameter.requires_grad
    }
    assert len(owned) == len(set(owned))
    assert set(owned) == trainable


def test_far_world_anchor_stays_separate_from_action_time_and_keeps_gradient() -> None:
    torch.manual_seed(114)
    config = _typed_332_config(flow_jepa_raw_activation_checkpoint=0)
    planner = V39PolicySystem(config).planner
    hidden = config.hidden_size
    cameras = config.num_cameras
    value = torch.zeros(
        1,
        config.future_anchors,
        cameras,
        hidden,
        requires_grad=True,
    )
    with torch.no_grad():
        value[:, -1, 0].fill_(2.0)
        value[:, -1, 1].fill_(3.0)
    far = planner._far_anchor_camera_context(value)
    candidates, names = planner._world_to_policy_source_candidates(
        value,
        far,
        "w1",
    )
    assert tuple(candidates.shape) == (
        1,
        config.action_horizon,
        cameras * 2,
        hidden,
    )
    assert names == (
        "w1_camera0",
        "w1_camera1",
        "w1_far1_camera0",
        "w1_far1_camera1",
    )
    assert torch.equal(
        candidates[:, :, :cameras],
        torch.zeros_like(candidates[:, :, :cameras]),
    )
    torch.testing.assert_close(
        candidates[:, :, cameras],
        value[:, -1, 0][:, None].expand(-1, config.action_horizon, -1),
    )
    torch.testing.assert_close(
        candidates[:, :, cameras + 1],
        value[:, -1, 1][:, None].expand(-1, config.action_horizon, -1),
    )
    candidates.sum().backward()
    assert value.grad is not None
    assert float(value.grad[:, :-1].abs().sum()) > 0.0
    assert float(value.grad[:, -1].abs().sum()) > 0.0


def test_predictive_change_contract_reaches_masked_raw_context() -> None:
    torch.manual_seed(111)
    config = _typed_332_config(
        flow_jepa_raw_activation_checkpoint=0,
        flow_jepa_teacher_balanced_target_mask=0,
        flow_jepa_predictive_change_contract=1,
    )
    system = V39PolicySystem(config).train()
    batch = 1
    output = system.flow_training_forward(
        _visual(config, batch=batch),
        torch.randn(batch, config.visual_history_length, config.state_dim),
        torch.randn(batch, config.executed_history_length, config.action_dim),
        torch.randn(batch, config.state_dim),
        torch.randn(batch, config.action_horizon, config.action_dim),
        target_visual=torch.randn(
            batch,
            config.future_anchors,
            config.visual_history_length,
            config.num_cameras,
            config.patches_per_camera,
            config.visual_token_dim,
        ),
        raw_visual=_raw_visual(config, batch=batch),
        make_counterfactuals=False,
    )
    assert float(output["flow_jepa_early_raw_mask_before_mixing"]) == 1.0
    assert float(output["flow_jepa_future_absolute_dino_seed"]) == 0.0
    assert float(output["flow_jepa_context_target_mask_aligned"]) == 1.0
    assert float(output["flow_jepa_future_shared_spatial_mask"]) == 1.0
    assert float(output["flow_jepa_address_flow_prior_scale"]) >= 0.25
    assert float(output["flow_jepa_address_flow_prior_floor"]) == 0.25
    assert torch.equal(
        output["flow_jepa_future_pred"],
        output["flow_jepa_future_delta_pred"],
    )
    loss = flow_jepa_future_prediction_loss(output, balance_horizons=True)
    assert torch.isfinite(loss)
    loss.backward(retain_graph=True)
    encoder = system.planner.flow_dino_evidence
    assert encoder is not None and encoder.early_masked_raw_context is not None
    assert any(
        parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
        for parameter in encoder.early_masked_raw_context.parameters()
    )
    for block in system.planner.blocks[3:6]:
        assert any(
            parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
            for parameter in block.parameters()
        )
    system.zero_grad(set_to_none=True)
    output["pred_physical_velocity"].float().square().mean().backward()
    assert encoder.soft_address_compiler is not None
    assert encoder.soft_address_compiler.flow_prior_log_scale.grad is not None
    assert (
        float(
            encoder.soft_address_compiler.flow_prior_log_scale.grad.abs().sum()
        )
        > 0.0
    )


def test_predictive_mask_also_owns_late_soft_address_appearance() -> None:
    """Hidden latest-frame appearance must not re-enter at the G boundary."""

    torch.manual_seed(118)
    config = _typed_332_config(
        flow_jepa_raw_activation_checkpoint=0,
        flow_jepa_teacher_balanced_target_mask=0,
        flow_jepa_predictive_change_contract=1,
    )
    encoder = FlowDINOEvidenceEncoder(config).train()
    pack = encoder(
        _visual(config, batch=1),
        raw_visual=_raw_visual(config, batch=1),
    )
    context = pack.raw_context
    assert context is not None
    high = context.high_features
    high_side = int(high.shape[-1])
    latest_mask = F.interpolate(
        pack.context_dropout_mask[:, -1].reshape(
            config.num_cameras,
            1,
            config.flow_jepa_grid_size,
            config.flow_jepa_grid_size,
        ).float(),
        size=(high_side, high_side),
        mode="nearest",
    ).bool().reshape(1, config.num_cameras, 1, high_side, high_side)
    assert bool(latest_mask.any())
    changed_high = high.clone()
    changed_high[:, -1] = torch.where(
        latest_mask,
        changed_high[:, -1] + 1000.0,
        changed_high[:, -1],
    )
    baseline_pack = replace(pack, raw_context=replace(context))
    changed_pack = replace(
        pack,
        raw_context=replace(context, high_features=changed_high),
    )
    rollout_count = (
        config.future_anchors
        * config.num_cameras
        * config.flow_jepa_grid_size**2
    )
    canvas = torch.randn(1, rollout_count + 2, config.hidden_size)
    slices = {"rollout": slice(2, 2 + rollout_count)}
    with torch.no_grad():
        baseline_detail = encoder.refine_raw_evidence(
            baseline_pack,
            canvas,
            slices,
            return_late_detail=True,
        )[-1]
        changed_detail = encoder.refine_raw_evidence(
            changed_pack,
            canvas,
            slices,
            return_late_detail=True,
        )[-1]
    assert baseline_detail is not None and baseline_detail.address_bank is not None
    assert changed_detail is not None and changed_detail.address_bank is not None
    for field in ("coarse_keys", "fine_keys", "fine_values"):
        torch.testing.assert_close(
            getattr(baseline_detail.address_bank, field),
            getattr(changed_detail.address_bank, field),
            rtol=0.0,
            atol=0.0,
        )


def test_full_repaired_contract_samples_with_one_observation_compile() -> None:
    torch.manual_seed(112)
    config = _typed_332_config(
        flow_jepa_raw_activation_checkpoint=0,
        flow_jepa_teacher_balanced_target_mask=0,
        flow_jepa_predictive_change_contract=1,
        action_history_enabled=1,
        action_history_recent_tokens=1,
        action_history_summary_tokens=1,
        action_history_condition_exact_null=1,
        goal_conditioning_enabled=1,
        goal_token_count=2,
        goal_language_dim=16,
        goal_resampler_depth=1,
        goal_condition_exact_null=1,
        stateless_phase_enabled=1,
        stateless_phase_count=4,
        inference_steps=3,
    )
    system = V39PolicySystem(config).eval()
    encoder = system.planner.flow_dino_evidence
    decoder = system.planner.evidence_latent_mmdit_action_decoder
    assert encoder is not None
    assert encoder.soft_address_compiler is not None
    assert encoder.early_masked_raw_context is not None
    assert system.planner.ground_to_world_attnres is not None
    assert system.planner.world_to_policy_attnres is not None
    assert decoder is not None and decoder.policy_delta_attnres is not None
    calls = {
        "compiler": 0,
        "early": 0,
        "g2w": 0,
        "w2p": 0,
        "p2m": 0,
    }

    def count(name: str):
        def hook(_module: torch.nn.Module, _inputs: tuple[object, ...], _output: object) -> None:
            calls[name] += 1

        return hook

    handles = (
        encoder.soft_address_compiler.register_forward_hook(count("compiler")),
        encoder.early_masked_raw_context.register_forward_hook(count("early")),
        system.planner.ground_to_world_attnres.register_forward_hook(count("g2w")),
        system.planner.world_to_policy_attnres.register_forward_hook(count("w2p")),
        decoder.policy_delta_attnres.register_forward_hook(count("p2m")),
    )
    try:
        with torch.no_grad():
            sampled = system.sample(
                _visual(config, batch=1),
                torch.randn(1, config.visual_history_length, config.state_dim),
                torch.randn(1, config.executed_history_length, config.action_dim),
                torch.randn(1, config.state_dim),
                raw_visual=_raw_visual(config, batch=1),
                steps=3,
                goal_language_tokens=torch.randn(1, 3, config.goal_language_dim),
                goal_language_mask=torch.ones(1, 3, dtype=torch.bool),
            )
    finally:
        for handle in handles:
            handle.remove()
    assert torch.is_tensor(sampled)
    assert tuple(sampled.shape) == (1, config.action_horizon, config.action_dim)
    assert torch.isfinite(sampled).all()
    assert calls["compiler"] == 1
    assert calls["early"] == 1
    assert calls["g2w"] == calls["w2p"] == calls["p2m"] == 3


def test_v103_action_field_has_no_target_leak_and_matches_one_step_deploy() -> None:
    """At fixed x_t,t, labels cannot alter the deployable velocity field."""

    torch.manual_seed(119)
    config = _complete_v103_config(
        flow_jepa_raw_activation_checkpoint=0,
        inference_steps=1,
    )
    system = V39PolicySystem(config).eval()
    batch = 1
    visual = _visual(config, batch=batch)
    raw_visual = _raw_visual(config, batch=batch)
    state_history = torch.randn(
        batch, config.visual_history_length, config.state_dim
    )
    executed_history = torch.randn(
        batch, config.executed_history_length, config.action_dim
    )
    state = torch.randn(batch, config.state_dim)
    goal_tokens = torch.randn(batch, 3, config.goal_language_dim)
    goal_mask = torch.ones(batch, 3, dtype=torch.bool)
    target_a = torch.randn(batch, config.action_horizon, config.action_dim)
    target_b = target_a + 3.0 * torch.randn_like(target_a)
    future_a = torch.randn(
        batch,
        config.future_anchors,
        config.visual_history_length,
        config.num_cameras,
        config.patches_per_camera,
        config.visual_token_dim,
    )
    future_b = future_a + 3.0 * torch.randn_like(future_a)
    noise_action = torch.randn(
        batch, config.action_horizon, config.action_dim
    )
    noise_physical = system.codec.encode(noise_action, state)
    common = {
        "raw_visual": raw_visual,
        "goal_language_tokens": goal_tokens,
        "goal_language_mask": goal_mask,
        "training_noise": noise_physical,
        "training_time": torch.ones(batch),
        "proposal_keep": torch.ones(batch),
        "make_counterfactuals": False,
    }
    with torch.no_grad():
        first = system.flow_training_forward(
            visual,
            state_history,
            executed_history,
            state,
            target_a,
            target_visual=future_a,
            **common,
        )
        second = system.flow_training_forward(
            visual,
            state_history,
            executed_history,
            state,
            target_b,
            target_visual=future_a,
            **common,
        )
        changed_teacher = system.flow_training_forward(
            visual,
            state_history,
            executed_history,
            state,
            target_a,
            target_visual=future_b,
            **common,
        )
        deployed = system.sample(
            visual,
            state_history,
            executed_history,
            state,
            raw_visual=raw_visual,
            steps=1,
            noise=noise_action,
            use_proposal=True,
            goal_language_tokens=goal_tokens,
            goal_language_mask=goal_mask,
        )

    # At t=1 the noisy state is exactly the supplied noise, independent of
    # target_action. Any difference here would be a teacher/posterior leak.
    torch.testing.assert_close(
        first["pred_physical_velocity"],
        second["pred_physical_velocity"],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        first["pred_action_estimate"],
        second["pred_action_estimate"],
        rtol=0.0,
        atol=0.0,
    )
    # Future DINO is a frozen target assembled after the policy forward. It
    # must change the target tensor without becoming an action condition.
    assert not torch.equal(
        first["flow_jepa_future_target"],
        changed_teacher["flow_jepa_future_target"],
    )
    torch.testing.assert_close(
        first["pred_physical_velocity"],
        changed_teacher["pred_physical_velocity"],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        first["pred_action_estimate"],
        changed_teacher["pred_action_estimate"],
        rtol=0.0,
        atol=0.0,
    )
    # A one-step deployment uses the same x_1 - v(x_1,1) update as the
    # training-side clean estimate. Auxiliary layer diagnostics are disabled
    # during sampling, so equality also proves that flag does not alter the
    # V103 action graph.
    assert torch.is_tensor(deployed)
    torch.testing.assert_close(
        first["pred_action_estimate"],
        deployed,
        rtol=1e-6,
        atol=1e-6,
    )


def test_v103_action_loss_trains_the_history_proposal_without_changing_forward() -> None:
    """The V103 switch changes gradient ownership, never deployed values."""

    def action_only_backward(detach: int) -> tuple[torch.Tensor, dict[str, float | None]]:
        torch.manual_seed(124)
        config = _complete_v103_config(
            flow_jepa_raw_activation_checkpoint=0,
            action_history_proposal_detach=detach,
        )
        system = V39PolicySystem(config).train()
        batch = 1
        target = torch.randn(batch, config.action_horizon, config.action_dim)
        state = torch.randn(batch, config.state_dim)
        output = system.flow_training_forward(
            _visual(config, batch=batch),
            torch.randn(batch, config.visual_history_length, config.state_dim),
            torch.randn(batch, config.executed_history_length, config.action_dim),
            state,
            target,
            raw_visual=_raw_visual(config, batch=batch),
            goal_language_tokens=torch.randn(batch, 3, config.goal_language_dim),
            goal_language_mask=torch.ones(batch, 3, dtype=torch.bool),
            training_noise=system.codec.encode(torch.randn_like(target), state),
            training_time=torch.full((batch,), 0.5),
            proposal_keep=torch.ones(batch),
            make_counterfactuals=False,
        )
        action_loss = output["pred_physical_velocity"].float().square().mean()
        action_loss.backward()
        proposal_grads = {
            name: (
                None
                if parameter.grad is None
                else float(parameter.grad.detach().abs().sum())
            )
            for name, parameter in system.proposal.named_parameters()
            if name == "future_query" or name.startswith("blocks.")
        }
        return output["pred_physical_velocity"].detach(), proposal_grads

    attached_output, attached_grads = action_only_backward(0)
    legacy_output, legacy_grads = action_only_backward(1)
    torch.testing.assert_close(attached_output, legacy_output, rtol=0.0, atol=0.0)
    assert attached_grads
    assert all(value is not None and value > 0.0 for value in attached_grads.values())
    assert all(value is None for value in legacy_grads.values())


def test_full_v103_training_graph_has_one_attached_model_path() -> None:
    """All active V103 repairs must coexist in one attached training graph."""

    torch.manual_seed(120)
    config = _complete_v103_config(
        flow_jepa_raw_activation_checkpoint=0,
    )
    system = V39PolicySystem(config).train()
    encoder = system.planner.flow_dino_evidence
    decoder = system.planner.evidence_latent_mmdit_action_decoder
    phase = system.planner.stateless_phase_adapter
    assert encoder is not None and encoder.soft_address_compiler is not None
    assert encoder.early_masked_raw_context is not None
    assert system.planner.goal_resampler is not None
    assert phase is not None
    assert decoder is not None and decoder.policy_delta_attnres is not None
    compile_calls = 0

    def count_compile(
        _module: torch.nn.Module,
        _inputs: tuple[object, ...],
        _output: object,
    ) -> None:
        nonlocal compile_calls
        compile_calls += 1

    handle = encoder.soft_address_compiler.register_forward_hook(count_compile)
    batch = 2
    try:
        output = system.flow_training_forward(
            _visual(config, batch=batch),
            torch.randn(batch, config.visual_history_length, config.state_dim),
            torch.randn(batch, config.executed_history_length, config.action_dim),
            torch.randn(batch, config.state_dim),
            torch.randn(batch, config.action_horizon, config.action_dim),
            target_visual=torch.randn(
                batch,
                config.future_anchors,
                config.visual_history_length,
                config.num_cameras,
                config.patches_per_camera,
                config.visual_token_dim,
            ),
            raw_visual=_raw_visual(config, batch=batch),
            goal_language_tokens=torch.randn(
                batch, 3, config.goal_language_dim
            ),
            goal_language_mask=torch.ones(batch, 3, dtype=torch.bool),
            make_counterfactuals=False,
        )
    finally:
        handle.remove()

    assert compile_calls == 1
    assert not output["flow_jepa_future_target"].requires_grad
    assert float(output["flow_jepa_context_target_mask_aligned"]) == 1.0
    assert float(output["flow_jepa_future_shared_spatial_mask"]) == 1.0
    assert float(output["flow_jepa_address_source_raw_match_active"]) == 1.0
    assert float(output["evidence_policy_delta_bridge_enabled"]) == 1.0
    assert float(output["evidence_top_policy_workspace_fixed_fusion"]) == 0.0
    total = (
        output["pred_physical_velocity"].float().square().mean()
        + 0.10
        * flow_jepa_future_prediction_loss(
            output,
            balance_horizons=True,
        )
    )
    total.backward()

    def attached(module: torch.nn.Module | None) -> bool:
        return module is not None and any(
            parameter.grad is not None
            and float(parameter.grad.detach().float().abs().sum()) > 0.0
            for parameter in module.parameters()
            if parameter.requires_grad
        )

    assert attached(encoder.early_masked_raw_context)
    assert attached(encoder.soft_address_compiler.source_raw_key)
    assert attached(encoder.soft_address_compiler.raw_pair_key)
    assert attached(encoder.future_prediction)
    assert attached(system.planner.goal_resampler)
    assert attached(system.proposal.history_proj)
    assert attached(phase)
    assert attached(system.planner.ground_to_world_attnres)
    assert attached(system.planner.world_to_policy_attnres)
    assert attached(decoder.policy_delta_attnres)
    assert attached(decoder.protected_detail_basis_attnres)
    for block_group in (
        system.planner.blocks[:3],
        system.planner.blocks[3:6],
        system.planner.blocks[6:],
    ):
        assert any(attached(block) for block in block_group)


def test_v104_future_memory_is_sequential_history_driven_and_attached() -> None:
    torch.manual_seed(131)
    config = _complete_v104_config(flow_jepa_raw_activation_checkpoint=0)
    encoder = FlowDINOEvidenceEncoder(config).train()
    batch = 2
    identity = torch.randn(
        batch,
        config.future_anchors,
        config.num_cameras,
        config.flow_jepa_grid_size,
        config.flow_jepa_grid_size,
        config.hidden_size,
    )
    motion = torch.randn(
        batch,
        config.visual_history_length - 1,
        config.num_cameras,
        config.flow_jepa_grid_size,
        config.flow_jepa_grid_size,
        encoder.MOTION_DIM,
        requires_grad=True,
    )
    context = torch.randn(
        batch,
        1,
        config.num_cameras,
        config.flow_jepa_grid_size,
        config.flow_jepa_grid_size,
        config.hidden_size,
        requires_grad=True,
    )
    queries, metrics = encoder._compose_future_queries(
        identity,
        motion,
        context,
    )
    assert queries.shape == identity.shape
    assert float(metrics["flow_jepa_sequential_horizon_memory"]) == 1.0
    assert 0.0 <= float(metrics["flow_jepa_perceptual_history_entropy"]) <= 1.0
    assert float(metrics["flow_jepa_horizon_transition_state_delta"]) > 0.0

    changed_motion = motion.detach().clone()
    changed_motion[:, 0] += 2.0
    changed, _ = encoder._compose_future_queries(
        identity,
        changed_motion,
        context.detach(),
    )
    assert float(
        (changed[:, -1] - queries[:, -1].detach()).detach().abs().sum()
    ) > 0.0

    queries[:, -1].float().square().mean().backward()
    assert motion.grad is not None and float(motion.grad.abs().sum()) > 0.0
    assert context.grad is not None and float(context.grad.abs().sum()) > 0.0
    assert encoder.future_history_score is not None
    assert encoder.future_transition is not None
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in encoder.future_history_score.parameters()
    )
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in encoder.future_transition.parameters()
    )


def test_v104_complete_forward_reports_effective_structural_contracts() -> None:
    torch.manual_seed(132)
    config = _complete_v104_config(flow_jepa_raw_activation_checkpoint=0)
    system = V39PolicySystem(config).train()
    batch = 1
    output = system.flow_training_forward(
        _visual(config, batch=batch),
        torch.randn(batch, config.visual_history_length, config.state_dim),
        torch.randn(batch, config.executed_history_length, config.action_dim),
        torch.randn(batch, config.state_dim),
        torch.randn(batch, config.action_horizon, config.action_dim),
        raw_visual=_raw_visual(config, batch=batch),
        target_visual=torch.randn(
            batch,
            config.future_anchors,
            config.visual_history_length,
            config.num_cameras,
            config.patches_per_camera,
            config.visual_token_dim,
        ),
        goal_language_tokens=torch.randn(
            batch, 3, config.goal_language_dim
        ),
        goal_language_mask=torch.ones(batch, 3, dtype=torch.bool),
        make_counterfactuals=False,
    )
    assert float(output["flow_jepa_bounded_flow_coordinates"]) == 1.0
    assert float(output["flow_jepa_sequential_horizon_memory"]) == 1.0
    assert float(output["role_residual_contract_enabled"]) == 1.0
    assert float(output["flow_jepa_raw_valid_fraction"]) > 0.999
    assert float(output["role_residual_bounded_rms"]) <= float(
        output["role_residual_raw_rms"]
    ) + 1e-6
    assert float(output["attnres_world_to_policy_value_rms"]) <= (
        config.role_attnres_max_value_rms + 1e-4
    )
    assert float(
        output["evidence_policy_delta_attnres_value_rms"]
    ) <= config.role_attnres_max_value_rms + 1e-4
    total = (
        output["pred_physical_velocity"].float().square().mean()
        + 0.10 * flow_jepa_future_prediction_loss(output, balance_horizons=True)
        + sum(output[key] for key in output if key.startswith("flow_jepa_") and key.endswith("_loss"))
    )
    total.backward()
    encoder = system.planner.flow_dino_evidence
    assert encoder is not None and encoder.future_transition is not None
    assert any(
        parameter.grad is not None
        and float(parameter.grad.detach().abs().sum()) > 0.0
        for parameter in encoder.future_transition.parameters()
    )


@pytest.mark.parametrize("contract", ("v104", "v105"))
def test_v104_v105_bfloat16_structural_paths_are_finite(
    contract: str,
) -> None:
    torch.manual_seed(133)
    config = (
        _complete_v105_config
        if contract == "v105"
        else _complete_v104_config
    )(flow_jepa_raw_activation_checkpoint=0)
    system = V39PolicySystem(config).train()
    batch = 1
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = system.flow_training_forward(
            _visual(config, batch=batch),
            torch.randn(batch, config.visual_history_length, config.state_dim),
            torch.randn(
                batch, config.executed_history_length, config.action_dim
            ),
            torch.randn(batch, config.state_dim),
            torch.randn(batch, config.action_horizon, config.action_dim),
            raw_visual=_raw_visual(config, batch=batch),
            target_visual=torch.randn(
                batch,
                config.future_anchors,
                config.visual_history_length,
                config.num_cameras,
                config.patches_per_camera,
                config.visual_token_dim,
            ),
            goal_language_tokens=torch.randn(
                batch, 3, config.goal_language_dim
            ),
            goal_language_mask=torch.ones(batch, 3, dtype=torch.bool),
            make_counterfactuals=False,
        )
        total = (
            output["pred_physical_velocity"].float().square().mean()
            + flow_jepa_future_prediction_loss(
                output,
                balance_horizons=True,
                reliable_normalization=(contract == "v105"),
            )
            + (
                0.02 * flow_jepa_horizon_address_loss(output)
                if contract == "v105"
                else 0.0
            )
            + output["flow_jepa_warp_loss"]
            + output["flow_jepa_cycle_loss"]
        )
    assert output["pred_physical_velocity"].dtype == torch.bfloat16
    assert output["flow_jepa_future_pred"].dtype == torch.bfloat16
    assert torch.isfinite(total)
    total.backward()
    structural_modules = [
        system.planner.flow_dino_evidence.future_transition,
        system.planner.ground_to_world_attnres,
        system.planner.world_to_policy_attnres,
    ]
    if contract == "v105":
        structural_modules.append(
            system.planner.flow_dino_evidence.horizon_address_jepa
        )
    for module in structural_modules:
        assert module is not None
        for parameter in module.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("contract", ("v103", "v104", "v105"))
def test_complete_model_optimizer_owns_every_trainable_parameter_once(
    contract: str,
) -> None:
    config_factory = {
        "v103": _complete_v103_config,
        "v104": _complete_v104_config,
        "v105": _complete_v105_config,
    }[contract]
    config = config_factory(flow_jepa_raw_activation_checkpoint=0)
    system = V39PolicySystem(config)
    trainer = (
        _complete_v105_trainer()
        if contract == "v105"
        else _complete_v103_trainer()
    )
    groups = _optimizer_groups(system, trainer)
    owners = [
        (id(parameter), str(group["name"]), float(group["lr"]))
        for group in groups
        for parameter in group["params"]
    ]
    owner_ids = [identifier for identifier, _, _ in owners]
    trainable_ids = {
        id(parameter)
        for parameter in system.parameters()
        if parameter.requires_grad
    }
    assert len(owner_ids) == len(set(owner_ids))
    assert set(owner_ids) == trainable_ids

    owner = {
        identifier: (name, learning_rate)
        for identifier, name, learning_rate in owners
    }
    planner = system.planner
    decoder = planner.evidence_latent_mmdit_action_decoder
    assert decoder is not None
    assert decoder.top_policy_workspace_lift is None
    assert planner.terminal_policy_layer_contracts_only
    assert not any(parameter.requires_grad for parameter in planner.midcut_norm.parameters())
    assert not any(parameter.requires_grad for parameter in planner.midcut_heads.parameters())
    policy_start = config.depth - config.flow_jepa_policy_blocks
    for head in planner.layer_contract_heads[:policy_start]:
        assert not any(parameter.requires_grad for parameter in head.parameters())
    for block in planner.blocks[-config.flow_jepa_policy_blocks :]:
        assert not any(parameter.requires_grad for parameter in block.n_dyn_kv.parameters())
        assert not any(parameter.requires_grad for parameter in block.rollout_cross.parameters())
    assert not any(
        parameter.requires_grad
        for parameter in decoder.evidence_adapter.intent_proj["visual"].parameters()
    )
    for module in (
        planner.ground_to_world_attnres,
        planner.world_to_policy_attnres,
        planner.late_raw_detail_reader,
        planner.stateless_phase_adapter,
        decoder.policy_delta_attnres,
        decoder.protected_detail_basis_attnres,
    ):
        assert module is not None
        assert all(
            id(parameter) in owner
            for parameter in module.parameters()
            if parameter.requires_grad
        )
    if contract in {"v104", "v105"}:
        encoder = planner.flow_dino_evidence
        assert encoder is not None
        for module in (
            encoder.future_history_score,
            encoder.future_transition,
        ):
            assert module is not None
            assert {
                owner[id(parameter)]
                for parameter in module.parameters()
                if parameter.requires_grad
            } == {("flow_dino_evidence", trainer.lr)}
        for module in (
            planner.ground_to_world_attnres,
            planner.world_to_policy_attnres,
        ):
            assert module is not None
            assert {
                owner[id(parameter)]
                for parameter in module.parameters()
                if parameter.requires_grad
            } == {("single_stage_shared_input", trainer.lr)}
        if contract == "v105":
            assert encoder.horizon_address_jepa is not None
            assert {
                owner[id(parameter)]
                for parameter in encoder.horizon_address_jepa.parameters()
                if parameter.requires_grad
            } == {("flow_dino_evidence", trainer.lr)}
    for index, block in enumerate(planner.blocks):
        block_owners = {
            owner[id(parameter)]
            for parameter in block.parameters()
            if parameter.requires_grad
        }
        assert block_owners == {
            (f"dit_block_{index}_single_stage", trainer.lr)
        }


@pytest.mark.parametrize("contract", ("v103", "v104", "v105"))
def test_complete_model_total_loss_reaches_every_trainable_parameter(
    contract: str,
) -> None:
    """No formal V103/V104/V105 parameter may be an optimizer-owned relic."""

    torch.manual_seed(121)
    config_factory = {
        "v103": _complete_v103_config,
        "v104": _complete_v104_config,
        "v105": _complete_v105_config,
    }[contract]
    config = config_factory(flow_jepa_raw_activation_checkpoint=0)
    trainer = (
        _complete_v105_trainer()
        if contract == "v105"
        else _complete_v103_trainer()
    )
    system = V39PolicySystem(config).train()
    decoder = system.planner.evidence_latent_mmdit_action_decoder
    assert decoder is not None
    decoder.set_execution_training_step(2000)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in system.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    batch = 2
    visual = _visual(config, batch=batch)
    raw_visual = _raw_visual(config, batch=batch)
    state_history = torch.randn(
        batch, config.visual_history_length, config.state_dim
    )
    executed_history = torch.randn(
        batch, config.executed_history_length, config.action_dim
    )
    state = torch.randn(batch, config.state_dim)
    target_action = torch.randn(
        batch, config.action_horizon, config.action_dim
    )
    target_visual = torch.randn(
        batch,
        config.future_anchors,
        config.visual_history_length,
        config.num_cameras,
        config.patches_per_camera,
        config.visual_token_dim,
    )
    goal_tokens = torch.randn(batch, 3, config.goal_language_dim)
    goal_mask = torch.ones(batch, 3, dtype=torch.bool)
    sample = {
        "policy_action": target_action,
        "policy_action_raw": target_action,
        "state_raw": state,
        "action_state": state,
    }

    def backward_once() -> tuple[dict[str, torch.Tensor], list[str], list[str]]:
        system.zero_grad(set_to_none=True)
        decoder.set_execution_training_step(2000)
        output = system.flow_training_forward(
            visual,
            state_history,
            executed_history,
            state,
            target_action,
            raw_visual=raw_visual,
            target_visual=target_visual,
            goal_language_tokens=goal_tokens,
            goal_language_mask=goal_mask,
            make_counterfactuals=False,
        )
        losses = flow_losses(
            system,
            sample,
            output,
            trainer,
            global_step=2000,
        )
        losses["loss"].backward()
        missing = [
            name
            for name, parameter in system.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        exact_zero = [
            name
            for name, parameter in system.named_parameters()
            if (
                parameter.requires_grad
                and parameter.grad is not None
                and float(parameter.grad.detach().float().abs().sum()) == 0.0
            )
        ]
        return output, missing, exact_zero

    first_output, first_missing, first_zero = backward_once()
    assert len(first_output["layer_contracts"]) == config.flow_jepa_policy_blocks
    assert first_missing == []
    # Zero-initialized controller/readout interiors may require the first
    # update to break symmetry, but they must become live immediately after.
    assert first_zero
    optimizer.step()
    _, second_missing, second_zero = backward_once()
    assert second_missing == []
    assert second_zero == []


def test_v103_primary_action_graph_excludes_only_declared_auxiliary_readouts() -> None:
    """No hidden trainable branch may live only outside the deployed action field."""

    torch.manual_seed(128)
    config = _complete_v103_config(flow_jepa_raw_activation_checkpoint=0)
    system = V39PolicySystem(config).train()
    decoder = system.planner.evidence_latent_mmdit_action_decoder
    assert decoder is not None
    optimizer = torch.optim.AdamW(
        [parameter for parameter in system.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    batch = 2
    visual = _visual(config, batch=batch)
    raw_visual = _raw_visual(config, batch=batch)
    state_history = torch.randn(
        batch, config.visual_history_length, config.state_dim
    )
    executed_history = torch.randn(
        batch, config.executed_history_length, config.action_dim
    )
    state = torch.randn(batch, config.state_dim)
    target_action = torch.randn(batch, config.action_horizon, config.action_dim)
    goal_tokens = torch.randn(batch, 3, config.goal_language_dim)
    goal_mask = torch.ones(batch, 3, dtype=torch.bool)
    noise = system.codec.encode(torch.randn_like(target_action), state)
    time = torch.full((batch,), 0.5)

    def action_backward() -> tuple[list[str], list[str]]:
        optimizer.zero_grad(set_to_none=True)
        decoder.set_execution_training_step(2000)
        output = system.flow_training_forward(
            visual,
            state_history,
            executed_history,
            state,
            target_action,
            raw_visual=raw_visual,
            goal_language_tokens=goal_tokens,
            goal_language_mask=goal_mask,
            training_noise=noise,
            training_time=time,
            proposal_keep=torch.ones(batch),
            make_counterfactuals=False,
        )
        output["pred_physical_velocity"].float().square().mean().backward()
        missing = [
            name
            for name, parameter in system.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        zero = [
            name
            for name, parameter in system.named_parameters()
            if parameter.requires_grad
            and parameter.grad is not None
            and float(parameter.grad.detach().abs().sum()) == 0.0
        ]
        return missing, zero

    action_backward()
    optimizer.step()
    missing, zero = action_backward()
    declared_auxiliary = {
        "proposal.action_head.weight",
        "proposal.action_head.bias",
        "planner.flow_dino_evidence.future_prediction.0.weight",
        "planner.flow_dino_evidence.future_prediction.0.bias",
        "planner.flow_dino_evidence.future_prediction.1.weight",
        "planner.flow_dino_evidence.future_prediction.1.bias",
        "planner.evidence_latent_mmdit_action_decoder.event_head.0.weight",
        "planner.evidence_latent_mmdit_action_decoder.event_head.0.bias",
        "planner.evidence_latent_mmdit_action_decoder.event_head.1.weight",
        "planner.evidence_latent_mmdit_action_decoder.event_head.1.bias",
        "planner.evidence_latent_mmdit_action_decoder.motion_head.0.weight",
        "planner.evidence_latent_mmdit_action_decoder.motion_head.0.bias",
        "planner.evidence_latent_mmdit_action_decoder.motion_head.1.weight",
        "planner.evidence_latent_mmdit_action_decoder.motion_head.1.bias",
    }
    assert set(missing) == declared_auxiliary
    assert zero == []


def test_complete_v103_bfloat16_forward_and_total_loss_backward_are_finite() -> None:
    """Exercise the complete formal graph, not isolated BF16-compatible parts."""

    torch.manual_seed(129)
    config = _complete_v103_config(flow_jepa_raw_activation_checkpoint=0)
    trainer = _complete_v103_trainer()
    system = V39PolicySystem(config).train()
    decoder = system.planner.evidence_latent_mmdit_action_decoder
    encoder = system.planner.flow_dino_evidence
    assert decoder is not None and encoder is not None
    decoder.set_execution_training_step(2000)
    batch = 1
    state = torch.randn(batch, config.state_dim)
    target_action = torch.randn(batch, config.action_horizon, config.action_dim)
    sample = {
        "policy_action": target_action,
        "policy_action_raw": target_action,
        "state_raw": state,
        "action_state": state,
    }
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = system.flow_training_forward(
            _visual(config, batch=batch),
            torch.randn(batch, config.visual_history_length, config.state_dim),
            torch.randn(batch, config.executed_history_length, config.action_dim),
            state,
            target_action,
            raw_visual=_raw_visual(config, batch=batch),
            target_visual=torch.randn(
                batch,
                config.future_anchors,
                config.visual_history_length,
                config.num_cameras,
                config.patches_per_camera,
                config.visual_token_dim,
            ),
            goal_language_tokens=torch.randn(
                batch, 3, config.goal_language_dim
            ),
            goal_language_mask=torch.ones(batch, 3, dtype=torch.bool),
            make_counterfactuals=False,
        )
        losses = flow_losses(
            system,
            sample,
            output,
            trainer,
            global_step=2000,
        )
        total = losses["loss"]

    assert output["pred_physical_velocity"].dtype == torch.bfloat16
    assert output["flow_jepa_future_pred"].dtype == torch.bfloat16
    assert total.dtype == torch.float32
    assert torch.isfinite(total)
    total.backward()
    for parameter in system.parameters():
        if not parameter.requires_grad:
            continue
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert system.proposal.future_query.grad is not None
    assert float(system.proposal.future_query.grad.detach().abs().sum()) > 0.0
    flow_head = encoder.flow.delta_head[-1].weight
    assert flow_head.grad is not None
    assert float(flow_head.grad.detach().abs().sum()) > 0.0


def test_v103_launcher_activates_the_repaired_model_contract() -> None:
    root = Path(__file__).parents[1]
    v103 = (
        root / "scripts" / "current_v103_typed_predictive_flow_jepa.sh"
    ).read_text(encoding="utf-8")
    v104 = (
        root / "scripts" / "current_v104_sequential_bounded_flow_jepa.sh"
    ).read_text(encoding="utf-8")
    v105 = (
        root / "scripts" / "current_v105_horizon_addressed_flow_jepa.sh"
    ).read_text(encoding="utf-8")
    v106 = (
        root / "scripts" / "current_v106_interval_stage_flow_jepa.sh"
    ).read_text(encoding="utf-8")
    v107 = (
        root / "scripts" / "current_v107_complete_top_path_flow_jepa.sh"
    ).read_text(encoding="utf-8")
    v108 = (
        root / "scripts" / "current_v108_online_horizon_address_flow_jepa.sh"
    ).read_text(encoding="utf-8")
    v109 = (
        root
        / "scripts"
        / "current_v109_progressive_grounding_address_flow_jepa.sh"
    ).read_text(encoding="utf-8")
    v104_probe = (
        root / "scripts" / "run_v104_model_path_probe.sh"
    ).read_text(encoding="utf-8")
    v105_probe = (
        root / "scripts" / "run_v105_model_path_probe.sh"
    ).read_text(encoding="utf-8")
    v106_probe = (
        root / "scripts" / "run_v106_model_path_probe.sh"
    ).read_text(encoding="utf-8")
    v107_probe = (
        root / "scripts" / "run_v107_model_path_probe.sh"
    ).read_text(encoding="utf-8")
    v108_probe = (
        root / "scripts" / "run_v108_model_path_probe.sh"
    ).read_text(encoding="utf-8")
    v109_probe = (
        root / "scripts" / "run_v109_model_path_probe.sh"
    ).read_text(encoding="utf-8")
    v100 = (
        root / "scripts" / "current_v100_strict_complementary_flow_jepa.sh"
    ).read_text(encoding="utf-8")
    v99 = (
        root / "scripts" / "current_v99_observable_raw_flow_332_jepa.sh"
    ).read_text(encoding="utf-8")
    required_v103 = (
        "export FLOW_JEPA_FUTURE_CHANGE_WEIGHT=0",
        'export CLEARVLA_REQUIRED_MODEL_CONTRACT="${CLEARVLA_REQUIRED_MODEL_CONTRACT:-v103}"',
        "--flow-jepa-teacher-balanced-target-mask 0",
        "--flow-jepa-policy-workspace-fixed-fusion 0",
        "--flow-jepa-world-anchor-write-only 0",
        "--flow-jepa-late-policy-detail 1",
        "--flow-jepa-soft-address-lattice 1",
        "--flow-jepa-address-flow-prior-floor",
        "--role-attnres-ground-to-world 1",
        "--role-attnres-world-to-policy 1",
        "--role-attnres-policy-to-mmdit 1",
        "--layer-shared-fm-probe 0",
        "--layer-recurrent-consequence 0",
        "--action-history-condition-exact-null 1",
        "--action-history-proposal-detach 0",
        "--goal-condition-exact-null 1",
        "--stateless-phase-enabled 1",
        "--flow-jepa-predictive-change-contract 1",
    )
    for fragment in required_v103:
        assert fragment in v103
    for fragment in (
        'export CLEARVLA_REQUIRED_MODEL_CONTRACT="${CLEARVLA_REQUIRED_MODEL_CONTRACT:-v104}"',
        "--flow-jepa-bounded-flow-coordinates 1",
        "--flow-jepa-sequential-horizon-memory 1",
        "--role-residual-amplitude-contract 1",
        "--role-residual-max-update-rms",
        "--role-attnres-max-value-rms",
    ):
        assert fragment in v104
    for fragment in (
        'export CLEARVLA_REQUIRED_MODEL_CONTRACT="${CLEARVLA_REQUIRED_MODEL_CONTRACT:-v105}"',
        "--flow-jepa-horizon-soft-address 1",
        "--flow-jepa-horizon-address-update-scale",
        "--flow-jepa-future-reliable-normalization 1",
        "--flow-jepa-horizon-address-loss-weight",
    ):
        assert fragment in v105
    for fragment in (
        'export CLEARVLA_REQUIRED_MODEL_CONTRACT="${CLEARVLA_REQUIRED_MODEL_CONTRACT:-v106}"',
        "--flow-jepa-variance-safe-routing 1",
        "--flow-jepa-complete-numerical-contract 1",
        "--flow-jepa-routing-norm-floor",
        "--flow-jepa-correlation-rms-floor",
        "--flow-jepa-visibility-transition-fraction",
        "--flow-jepa-horizon-value-max-rms",
        "--flow-jepa-interval-stage-delta 1",
        '--flow-jepa-interval-boundaries "4,8,16,32,48"',
        "--flow-jepa-interval-support-offsets",
        "--flow-jepa-interval-stage-update-scale",
        "--flow-jepa-interval-stage-loss-weight",
    ):
        assert fragment in v106
    for fragment in (
        'export CLEARVLA_REQUIRED_MODEL_CONTRACT="${CLEARVLA_REQUIRED_MODEL_CONTRACT:-v107}"',
        "--flow-jepa-policy-multi-glimpse-address 1",
        "--flow-jepa-horizon-cell-fine-address 1",
        "--flow-jepa-interval-stage-typed-value 1",
        "--role-residual-contract-after-gate 1",
    ):
        assert fragment in v107
    for fragment in (
        'export CLEARVLA_REQUIRED_MODEL_CONTRACT="${CLEARVLA_REQUIRED_MODEL_CONTRACT:-v108}"',
        'export FLOW_JEPA_PARENT_VERSION="${FLOW_JEPA_PARENT_VERSION:-v108}"',
        "--flow-jepa-online-horizon-address 1",
        "current_v107_complete_top_path_flow_jepa.sh",
    ):
        assert fragment in v108
    for fragment in (
        'export CLEARVLA_REQUIRED_MODEL_CONTRACT="${CLEARVLA_REQUIRED_MODEL_CONTRACT:-v109}"',
        'export FLOW_JEPA_PARENT_VERSION="${FLOW_JEPA_PARENT_VERSION:-v109}"',
        "--flow-jepa-progressive-grounding-address 1",
        "current_v108_online_horizon_address_flow_jepa.sh",
    ):
        assert fragment in v109
    assert "runs/v104_sequential_bounded_flow_jepa/checkpoints/latest.pt" in v104_probe
    assert "MODEL_PATH_PROBE_LABEL=v104" in v104_probe
    assert "MODEL_PATH_REQUIRED_CONTRACT=v104" in v104_probe
    assert "runs/v105_horizon_addressed_flow_jepa/checkpoints/latest.pt" in v105_probe
    assert "MODEL_PATH_PROBE_LABEL=v105" in v105_probe
    assert "MODEL_PATH_REQUIRED_CONTRACT=v105" in v105_probe
    assert "runs/v106_interval_stage_flow_jepa/checkpoints/latest.pt" in v106_probe
    assert "MODEL_PATH_PROBE_LABEL=v106" in v106_probe
    assert "MODEL_PATH_REQUIRED_CONTRACT=v106" in v106_probe
    assert "runs/v107_complete_top_path_flow_jepa/checkpoints/latest.pt" in v107_probe
    assert "MODEL_PATH_PROBE_LABEL=v107" in v107_probe
    assert "MODEL_PATH_REQUIRED_CONTRACT=v107" in v107_probe
    assert "runs/v108_online_horizon_address_flow_jepa/checkpoints/latest.pt" in v108_probe
    assert "MODEL_PATH_PROBE_LABEL=v108" in v108_probe
    assert "MODEL_PATH_REQUIRED_CONTRACT=v108" in v108_probe
    assert "runs/v109_progressive_grounding_address_flow_jepa/checkpoints/latest.pt" in v109_probe
    assert "MODEL_PATH_PROBE_LABEL=v109" in v109_probe
    assert "MODEL_PATH_REQUIRED_CONTRACT=v109" in v109_probe
    assert (
        '--flow-jepa-future-change-loss-weight "${FLOW_JEPA_FUTURE_CHANGE_WEIGHT}"'
        in v100
    )
    assert "--stage1-initialization-enabled 0" in v99
    assert "--require-flow-jepa-stage1-checkpoint 0" in v99
    v48 = (
        root / "scripts" / "current_v48_justok.sh"
    ).read_text(encoding="utf-8")
    assert "--layer-recurrent-consequence 1" in v48
    assert v103.rfind("--layer-recurrent-consequence 0") >= 0
    # V48 is the historical source of the contradictory default. No
    # intermediate wrapper may reintroduce it after V103 has selected the
    # terminal-P/typed-bottom graph.
    for script_name in (
        "current_v100_strict_complementary_flow_jepa.sh",
        "current_v99_observable_raw_flow_332_jepa.sh",
        "current_v96_late_bottleneck_jepa.sh",
        "current_v94_latent_ownership_execution.sh",
        "current_v91_time_domain_evidence_mmdit.sh",
        "current_v65_z_workspace_full_diag.sh",
    ):
        wrapper = (root / "scripts" / script_name).read_text(
            encoding="utf-8"
        )
        assert "--layer-recurrent-consequence 1" not in wrapper


def test_v103_model_probe_rejects_partial_or_duplicate_contracts() -> None:
    config = _complete_v103_config()
    trainer = _complete_v103_trainer()
    _validate_complete_v103_model_probe_contract(config, trainer)
    assert _validate_required_model_contract("v103", config, trainer) == "v103"

    partial = _complete_v103_config(
        flow_jepa_predictive_change_contract=0,
    )
    try:
        _validate_complete_v103_model_probe_contract(partial, trainer)
    except ValueError as error:
        assert "flow_jepa_predictive_change_contract=0" in str(error)
    else:
        raise AssertionError("V103 probe accepted a partial predictive contract")

    detached_proposal = replace(config, action_history_proposal_detach=1)
    try:
        _validate_complete_v103_model_probe_contract(detached_proposal, trainer)
    except ValueError as error:
        assert "action_history_proposal_detach=1" in str(error)
    else:
        raise AssertionError("V103 probe accepted an action-detached proposal condition")

    duplicate = replace(
        trainer,
        flow_jepa_future_change_loss_weight=0.02,
    )
    try:
        _validate_complete_v103_model_probe_contract(config, duplicate)
    except ValueError as error:
        assert "flow_jepa_future_change_loss_weight=0.02" in str(error)
    else:
        raise AssertionError("V103 probe accepted duplicate future supervision")

    zero_bridge = replace(config, role_attnres_ground_to_world_scale=0.0)
    try:
        _validate_complete_v103_model_probe_contract(zero_bridge, trainer)
    except ValueError as error:
        assert "role_attnres_ground_to_world_scale must be positive" in str(
            error
        )
    else:
        raise AssertionError("V103 probe accepted a zero-scale role bridge")


def test_v104_contract_requires_all_three_structural_repairs() -> None:
    config = _complete_v104_config()
    trainer = _complete_v103_trainer()
    _validate_complete_v104_model_contract(config, trainer)
    assert _validate_required_model_contract("v104", config, trainer) == "v104"
    for field in (
        "flow_jepa_bounded_flow_coordinates",
        "flow_jepa_sequential_horizon_memory",
        "role_residual_amplitude_contract",
    ):
        partial = replace(config, **{field: 0})
        with pytest.raises(ValueError, match=field):
            _validate_complete_v104_model_contract(partial, trainer)


def test_v105_horizon_address_reads_continuous_bank_without_hard_selection() -> None:
    torch.manual_seed(144)
    config = _complete_v105_config()
    raw_dim = int(config.flow_jepa_raw_base_channels) + int(
        config.flow_jepa_raw_base_channels
    ) // 2
    reader = _HorizonSoftAddressJEPA(config, raw_dim=raw_dim)
    assert tuple(inspect.signature(reader.forward).parameters) == (
        "future_tokens",
        "bank",
    )
    batch = 2
    grid = int(config.flow_jepa_grid_size)
    cameras = int(config.num_cameras)
    slots = int(config.flow_jepa_address_slots)
    route = int(config.flow_jepa_address_route_dim)
    candidates = 9
    coarse = torch.randn(
        batch, cameras, grid, grid, slots, route, requires_grad=True
    )
    fine_keys = torch.randn(
        batch,
        cameras,
        grid,
        grid,
        slots,
        candidates,
        route,
        requires_grad=True,
    )
    fine_values = torch.randn(
        batch,
        cameras,
        grid,
        grid,
        slots,
        candidates,
        raw_dim,
        requires_grad=True,
    )
    bank = SoftAddressLatticeBank(
        coarse_keys=coarse,
        fine_keys=fine_keys,
        fine_values=fine_values,
        fine_valid=torch.ones(
            batch,
            cameras,
            grid,
            grid,
            slots,
            candidates,
            dtype=torch.bool,
        ),
        coarse_centers=torch.zeros(batch, cameras, grid, grid, slots, 2),
        coarse_variance=torch.zeros(batch, cameras, grid, grid, slots, 2),
        fine_radius=torch.ones(batch, cameras, grid, grid, slots),
    )
    future = torch.randn(
        batch,
        config.future_anchors * cameras * grid * grid,
        config.hidden_size,
        requires_grad=True,
    )
    refined, logits, metrics = reader(future, bank)
    assert tuple(refined.shape) == tuple(future.shape)
    assert tuple(logits.shape) == (
        batch,
        config.future_anchors,
        cameras,
        grid,
        grid,
    )
    probability = logits.float().flatten(2).softmax(dim=-1)
    torch.testing.assert_close(
        probability.sum(dim=-1),
        torch.ones_like(probability.sum(dim=-1)),
    )
    with torch.no_grad():
        zero_refined, _, _ = reader(
            future,
            replace(bank, fine_values=torch.zeros_like(fine_values)),
        )
    torch.testing.assert_close(zero_refined, future, rtol=0.0, atol=0.0)
    assert 0.0 < float(metrics["flow_jepa_horizon_address_route_entropy"]) <= 1.0
    assert 0.0 < float(metrics["flow_jepa_horizon_address_fine_entropy"]) <= 1.0
    (refined.float().square().mean() + logits.float().square().mean()).backward()
    for tensor in (future, coarse, fine_keys, fine_values):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert float(tensor.grad.abs().sum()) > 0.0


def test_v105_flag_off_is_exact_v104_future_prediction() -> None:
    torch.manual_seed(146)
    encoder = FlowDINOEvidenceEncoder(_complete_v104_config()).eval()
    tokens = torch.randn(
        2,
        encoder.config.future_anchors
        * encoder.config.num_cameras
        * encoder.config.flow_jepa_grid_size
        * encoder.config.flow_jepa_grid_size,
        encoder.config.hidden_size,
    )
    direct = encoder.predict_future(tokens)
    addressed, metrics = encoder.predict_future_with_address(tokens, None)
    torch.testing.assert_close(addressed, direct, rtol=0.0, atol=0.0)
    assert metrics == {}
    assert encoder.horizon_address_jepa is None


def test_v105_deploy_diagnostics_off_skips_auxiliary_address_exactly() -> None:
    torch.manual_seed(147)
    encoder = FlowDINOEvidenceEncoder(_complete_v105_config()).eval()
    tokens = torch.randn(
        1,
        encoder.config.future_anchors
        * encoder.config.num_cameras
        * encoder.config.flow_jepa_grid_size
        * encoder.config.flow_jepa_grid_size,
        encoder.config.hidden_size,
    )
    direct = encoder.predict_future(tokens)
    skipped, metrics = encoder.predict_future_with_address(
        tokens,
        None,
        enable_address=False,
    )
    torch.testing.assert_close(skipped, direct, rtol=0.0, atol=0.0)
    assert metrics == {}


def test_v105_complete_forward_attaches_address_to_jepa_not_a_second_action_lane() -> None:
    torch.manual_seed(145)
    config = _complete_v105_config(flow_jepa_raw_activation_checkpoint=0)
    system = V39PolicySystem(config).train()
    batch = 1
    target_action = torch.randn(
        batch, config.action_horizon, config.action_dim
    )
    output = system.flow_training_forward(
        _visual(config, batch=batch),
        torch.randn(batch, config.visual_history_length, config.state_dim),
        torch.randn(
            batch, config.executed_history_length, config.action_dim
        ),
        torch.randn(batch, config.state_dim),
        target_action,
        target_visual=torch.randn(
            batch,
            config.future_anchors,
            config.visual_history_length,
            config.num_cameras,
            config.patches_per_camera,
            config.visual_token_dim,
        ),
        raw_visual=_raw_visual(config, batch=batch),
        goal_language_tokens=torch.randn(
            batch, 3, config.goal_language_dim
        ),
        goal_language_mask=torch.ones(batch, 3, dtype=torch.bool),
        make_counterfactuals=False,
    )
    assert "flow_jepa_horizon_address_logits" in output
    assert float(output["flow_jepa_horizon_soft_address"]) == 1.0
    assert not any(
        "address" in key
        for key in output
        if key.startswith("pred_") or key.startswith("post_")
    )
    objective = flow_jepa_future_prediction_loss(
        output,
        balance_horizons=True,
        reliable_normalization=True,
    ) + 0.02 * flow_jepa_horizon_address_loss(output)
    objective.backward()
    encoder = system.planner.flow_dino_evidence
    assert encoder is not None and encoder.horizon_address_jepa is not None
    assert any(
        parameter.grad is not None
        and float(parameter.grad.detach().abs().sum()) > 0.0
        for parameter in encoder.horizon_address_jepa.parameters()
        if parameter.requires_grad
    )
    assert encoder.soft_address_compiler is not None
    assert (
        encoder.soft_address_compiler.raw_pair_key[-1].weight.grad
        is not None
    )
    trainer = _complete_v105_trainer()
    covered = {
        id(parameter)
        for group in _optimizer_groups(system, trainer)
        for parameter in group["params"]
    }
    assert all(
        id(parameter) in covered
        for parameter in encoder.horizon_address_jepa.parameters()
        if parameter.requires_grad
    )


def test_v105_teacher_address_is_loss_only_and_cannot_collapse_mass_to_zero() -> None:
    batch, anchors, cameras, grid, hidden = 1, 4, 2, 2, 8
    positions = cameras * grid * grid
    current = torch.zeros(batch, positions, hidden)
    future = current[:, None].expand(-1, anchors, -1, -1).clone()
    changed_positions = (0, 3, 5, 7)
    for horizon, position in enumerate(changed_positions):
        future[:, horizon, position, 0] = 1.0 + float(horizon)
    future = future.reshape(batch, anchors * positions, hidden)
    uniform_logits = torch.zeros(
        batch, anchors, cameras, grid, grid, requires_grad=True
    )
    matched_logits = torch.zeros_like(uniform_logits).detach()
    matched_flat = matched_logits.reshape(batch, anchors, positions)
    for horizon, position in enumerate(changed_positions):
        matched_flat[:, horizon, position] = 6.0
    matched_logits = matched_logits.requires_grad_()

    def output(logits: torch.Tensor, mask_value: bool) -> dict[str, torch.Tensor | tuple[int, ...]]:
        return {
            "pred_physical_velocity": torch.zeros(1, 1, 1),
            "flow_jepa_horizon_address_logits": logits,
            "flow_jepa_future_target": future,
            "flow_jepa_current_target": current,
            # The address teacher uses every spatial cell and cannot be
            # steered by the online context-mask allocation.
            "flow_jepa_future_target_mask": torch.full(
                (batch, anchors * positions), mask_value, dtype=torch.bool
            ),
            "flow_jepa_future_offsets": (4, 12, 24, 48),
        }

    uniform_loss = flow_jepa_horizon_address_loss(output(uniform_logits, False))
    matched_loss = flow_jepa_horizon_address_loss(output(matched_logits, True))
    assert float(matched_loss) < float(uniform_loss)
    uniform_loss.backward()
    assert uniform_logits.grad is not None
    assert torch.isfinite(uniform_logits.grad).all()
    assert float(uniform_logits.grad.abs().sum()) > 0.0
    probability = uniform_logits.detach().flatten(2).softmax(dim=-1)
    torch.testing.assert_close(
        probability.sum(dim=-1),
        torch.ones(batch, anchors),
    )


def test_v105_reliable_future_loss_keeps_raw_static_anchor_without_jitter_amplification() -> None:
    pred = torch.full((1, 8, 4), 0.1, requires_grad=True)
    output = {
        "pred_physical_velocity": torch.zeros(1, 1, 1),
        "flow_jepa_future_pred": pred,
        "flow_jepa_future_delta_pred": pred,
        "flow_jepa_future_target": torch.zeros_like(pred),
        "flow_jepa_current_target": torch.zeros(1, 4, 4),
        "flow_jepa_future_target_mask": torch.ones(1, 8, dtype=torch.bool),
        "flow_jepa_future_offsets": (4, 48),
    }
    legacy = flow_jepa_future_prediction_loss(
        output,
        balance_horizons=True,
        reliable_normalization=False,
    )
    reliable = flow_jepa_future_prediction_loss(
        output,
        balance_horizons=True,
        reliable_normalization=True,
    )
    assert 0.0 < float(reliable) < float(legacy)
    reliable.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    assert float(pred.grad.abs().sum()) > 0.0


def test_v105_reliable_future_loss_is_not_invariant_to_weak_teacher_scale() -> None:
    current = torch.ones(1, 4, 4)

    def example(delta: float) -> dict[str, Tensor | tuple[int, ...]]:
        current_future = current[:, None].expand(-1, 2, -1, -1).reshape(
            1, 8, 4
        )
        return {
            "pred_physical_velocity": torch.zeros(1, 1, 1),
            "flow_jepa_future_pred": torch.zeros(1, 8, 4),
            "flow_jepa_future_delta_pred": torch.zeros(1, 8, 4),
            "flow_jepa_future_target": current_future + delta,
            "flow_jepa_current_target": current,
            "flow_jepa_future_target_mask": torch.ones(
                1, 8, dtype=torch.bool
            ),
            "flow_jepa_future_offsets": (4, 48),
        }

    def objective(delta: float, *, reliable: bool) -> Tensor:
        return flow_jepa_future_prediction_loss(
            example(delta),
            balance_horizons=True,
            reliable_normalization=reliable,
        )

    legacy_weak = objective(0.01, reliable=False)
    legacy_strong = objective(0.10, reliable=False)
    reliable_weak = objective(0.01, reliable=True)
    reliable_strong = objective(0.10, reliable=True)
    torch.testing.assert_close(legacy_weak, legacy_strong, rtol=1e-5, atol=1e-6)
    assert float(reliable_weak) < 0.25 * float(reliable_strong)
    weak_diagnostics = flow_jepa_future_reliable_diagnostics(example(0.01))
    assert float(weak_diagnostics["flow_jepa_future_normalization_scale"]) > float(
        weak_diagnostics["flow_jepa_future_target_delta_scale"]
    )
    assert float(
        weak_diagnostics["flow_jepa_future_current_reference_scale"]
    ) == pytest.approx(1.0)


def test_v105_contract_requires_address_and_reliable_normalization() -> None:
    config = _complete_v105_config()
    trainer = _complete_v105_trainer()
    _validate_complete_v105_model_contract(config, trainer)
    assert _validate_required_model_contract("v105", config, trainer) == "v105"
    with pytest.raises(ValueError, match="flow_jepa_horizon_soft_address"):
        _validate_complete_v105_model_contract(
            replace(config, flow_jepa_horizon_soft_address=0),
            trainer,
        )
    with pytest.raises(
        ValueError, match="flow_jepa_future_reliable_normalization"
    ):
        _validate_complete_v105_model_contract(
            config,
            replace(trainer, flow_jepa_future_reliable_normalization=0),
        )
    with pytest.raises(ValueError, match="flow_jepa_horizon_address_loss_weight"):
        _validate_complete_v105_model_contract(
            config,
            replace(trainer, flow_jepa_horizon_address_loss_weight=0.0),
        )


def test_v106_variance_floor_preserves_zero_and_bounds_small_signal_gain() -> None:
    constant = torch.full((2, 8), 1e-7)
    normalized, denominator = variance_floored_centered_norm(constant, 0.25)
    torch.testing.assert_close(
        normalized,
        torch.zeros_like(normalized),
        rtol=0.0,
        atol=0.0,
    )
    assert float(denominator.min()) == pytest.approx(0.25)

    vector = torch.linspace(-1e-5, 1e-5, 8, dtype=torch.float64)

    def normalize(row: torch.Tensor) -> torch.Tensor:
        return variance_floored_centered_norm(row, 0.25)[0]

    jacobian = torch.autograd.functional.jacobian(normalize, vector)
    largest_gain = torch.linalg.svdvals(jacobian).max()
    assert float(largest_gain) <= 4.0 + 1e-6


def test_v106_correlation_floor_preserves_cosine_scale_and_bounds_cancellation() -> None:
    ordinary = torch.randn(2, 16, 3, 3)
    legacy = F.normalize(ordinary, dim=1)
    safe, denominator = rms_floored_l2_normalize(
        ordinary, 0.10, dim=1
    )
    cosine = F.cosine_similarity(
        legacy.flatten(2).transpose(1, 2),
        safe.flatten(2).transpose(1, 2),
        dim=-1,
    )
    torch.testing.assert_close(cosine, torch.ones_like(cosine), atol=1e-6, rtol=1e-6)
    assert float(denominator.min()) >= 0.10
    legacy_pyramid = _CorrelationPyramid(
        ordinary,
        ordinary.roll(shifts=1, dims=-1),
        levels=1,
        radius=0,
    )
    expected_legacy = torch.einsum(
        "bchw,bcij->bhwij",
        F.normalize(ordinary.float(), dim=1),
        F.normalize(ordinary.roll(shifts=1, dims=-1).float(), dim=1),
    ).reshape(2, 9, 9)
    torch.testing.assert_close(
        legacy_pyramid.matrix,
        expected_legacy,
        rtol=0.0,
        atol=0.0,
    )

    cancelled = (1e-8 * torch.randn(1, 4, 2, 2)).requires_grad_(True)
    other = (1e-8 * torch.randn(1, 4, 2, 2)).requires_grad_(True)
    correlation = _CorrelationPyramid(
        cancelled,
        other,
        levels=1,
        radius=0,
        normalization_floor=0.10,
    )
    correlation.matrix.square().mean().backward()
    assert cancelled.grad is not None and torch.isfinite(cancelled.grad).all()
    assert other.grad is not None and torch.isfinite(other.grad).all()
    assert float(correlation.normalization_denominator_min) >= 0.10
    assert float(correlation.normalization_gain_max) <= 10.0 + 1e-6


def test_v106_cycle_visibility_is_continuous_and_has_an_explicit_gain_bound() -> None:
    squared_error = torch.tensor(
        [[[[0.45, 0.50, 0.55]]]], requires_grad=True
    )
    threshold = torch.full_like(squared_error, 0.50)
    valid = torch.ones_like(squared_error, dtype=torch.bool)
    visible, hard, width_min, gain_max = _continuous_cycle_visibility(
        valid,
        squared_error,
        threshold,
        transition_fraction=0.10,
    )
    assert 0.0 < float(visible.min()) < float(visible.max()) < 1.0
    assert torch.equal(hard, torch.tensor([[[[1.0, 0.0, 0.0]]]]))
    visible.sum().backward()
    assert squared_error.grad is not None
    assert torch.isfinite(squared_error.grad).all()
    assert float(squared_error.grad.abs().min()) > 0.0
    assert float(width_min) == pytest.approx(0.05)
    assert float(gain_max) == pytest.approx(5.0)


def test_v106_role_block_normalization_bounds_near_constant_backward_gain() -> None:
    config = _complete_v106_config()
    block = TemporalDynamicsBoundDiTBlock(config, role="world").train()
    lengths = {
        "task": 1,
        "state": 1,
        "state_history": 1,
        "executed": 1,
        "proposal": 1,
        "trajectory": config.action_horizon * config.action_basis_tokens,
        "stage": 0,
        "rollout": (
            config.future_anchors
            * config.num_cameras
            * config.future_grid_size**2
        ),
        "registers": config.canvas_registers,
    }
    slices: dict[str, slice] = {}
    cursor = 0
    for name, length in lengths.items():
        slices[name] = slice(cursor, cursor + int(length))
        cursor += int(length)
    canvas = (
        torch.full((1, cursor, config.hidden_size), 1e-6)
        + 1e-8 * torch.randn(1, cursor, config.hidden_size)
    ).requires_grad_(True)
    visual = (
        torch.full((1, 7, config.hidden_size), 1e-6)
        + 1e-8 * torch.randn(1, 7, config.hidden_size)
    ).requires_grad_(True)
    output, metrics = block(
        canvas,
        visual,
        torch.randn(1, config.hidden_size),
        slices,
    )
    output.square().mean().backward()
    assert canvas.grad is not None and torch.isfinite(canvas.grad).all()
    assert visual.grad is not None and torch.isfinite(visual.grad).all()
    assert float(metrics["normalization_contract_enabled"]) == 1.0
    assert float(metrics["normalization_denominator_min"]) >= 0.25
    assert float(metrics["normalization_gain_max"]) <= 16.0 + 1e-5


def test_v106_interval_teacher_is_signed_spatial_increment_not_global_mean() -> None:
    config = _complete_v106_config()
    encoder = FlowDINOEvidenceEncoder(config).eval()
    batch = 1
    supports = config.flow_jepa_effective_interval_support_offsets
    target = torch.zeros(
        batch,
        len(supports),
        config.visual_history_length,
        config.num_cameras,
        config.patches_per_camera,
        config.visual_token_dim,
    )
    current = torch.zeros(
        batch,
        config.visual_history_length,
        config.num_cameras,
        config.patches_per_camera,
        config.visual_token_dim,
    )
    # Two cells carry different signed velocities.  If the teacher pooled
    # spatially or reduced the interval to a plain frame average, this ratio
    # and the signed progression target would disappear.
    for support_index, offset in enumerate(supports):
        target[:, support_index, -1, 0, 0, 0] = float(offset)
        target[:, support_index, -1, 0, 1, 0] = -3.0 * float(offset)

    def project(tokens: torch.Tensor) -> torch.Tensor:
        leading = tuple(tokens.shape[:-2])
        scalar = tokens[..., 0].reshape(
            *leading,
            config.flow_jepa_grid_size,
            config.flow_jepa_grid_size,
        )
        return scalar[..., None].expand(
            *scalar.shape,
            config.hidden_size,
        )

    with patch.object(encoder, "_teacher_project_grid", side_effect=project):
        targets = encoder.teacher_interval_targets(target, current)
    positions = (
        config.num_cameras
        * config.flow_jepa_grid_size
        * config.flow_jepa_grid_size
    )
    progression = targets["flow_jepa_interval_progress_target"].reshape(
        batch,
        config.future_anchors,
        positions,
        config.hidden_size,
    )
    endpoint = targets["flow_jepa_interval_endpoint_target"].reshape_as(
        progression
    )
    for horizon, (start, end) in enumerate(config.flow_jepa_interval_windows):
        expected = float(end - start)
        assert float(progression[0, horizon, 0, 0]) == pytest.approx(expected)
        assert float(progression[0, horizon, 1, 0]) == pytest.approx(
            -3.0 * expected
        )
        torch.testing.assert_close(
            progression[:, horizon],
            endpoint[:, horizon],
        )
    assert tuple(targets["flow_jepa_future_target"].shape) == (
        batch,
        config.future_anchors * positions,
        config.hidden_size,
    )

    outlier_target = target.clone()
    outlier_target[:, supports.index(12), -1, 0, 0, 0] += 100.0
    with patch.object(encoder, "_teacher_project_grid", side_effect=project):
        outlier_targets = encoder.teacher_interval_targets(
            outlier_target,
            current,
        )
    outlier_content = outlier_targets["flow_jepa_future_target"].reshape(
        batch,
        config.future_anchors,
        positions,
        config.hidden_size,
    )
    plain_frame_mean = (8.0 + 112.0 + 16.0) / 3.0
    assert float(outlier_content[0, 1, 0, 0]) != pytest.approx(
        plain_frame_mean
    )
    teacher_source = inspect.getsource(
        FlowDINOEvidenceEncoder.teacher_interval_targets
    )
    assert "self.anchors" not in teacher_source


def test_v106_preflight_validates_the_real_interval_teacher_pack() -> None:
    config = _complete_v106_config()
    batch = 2
    positions = (
        config.num_cameras
        * config.flow_jepa_grid_size
        * config.flow_jepa_grid_size
    )
    future_shape = (
        batch,
        config.future_anchors * positions,
        config.hidden_size,
    )
    pack = {
        "flow_jepa_future_target": torch.randn(*future_shape),
        "flow_jepa_interval_progress_target": torch.randn(*future_shape),
        "flow_jepa_interval_endpoint_target": torch.randn(*future_shape),
        "flow_jepa_current_target": torch.randn(
            batch,
            positions,
            config.hidden_size,
        ),
        "flow_jepa_future_target_mask": torch.ones(
            batch,
            config.future_anchors * positions,
            dtype=torch.bool,
        ),
        "flow_jepa_interval_support_count": torch.tensor(12.0),
        "flow_jepa_interval_effective_support": torch.tensor(3.0),
    }
    _validate_v106_preflight_target_pack(
        pack,
        config=config,
        batch_size=batch,
    )
    with pytest.raises(ValueError, match="progress_target"):
        _validate_v106_preflight_target_pack(
            {
                **pack,
                "flow_jepa_interval_progress_target": torch.randn(
                    batch,
                    positions,
                    config.hidden_size,
                ),
            },
            config=config,
            batch_size=batch,
        )
    with pytest.raises(ValueError, match="effective interval support"):
        _validate_v106_preflight_target_pack(
            {
                **pack,
                "flow_jepa_interval_effective_support": torch.tensor(0.0),
            },
            config=config,
            batch_size=batch,
        )


def test_v106_interval_organizer_is_cell_local_horizon_causal_and_attached() -> None:
    torch.manual_seed(151)
    config = _complete_v106_config()
    organizer = _IntervalStageDeltaOrganizer(config).eval()
    token_count = (
        config.future_anchors
        * config.num_cameras
        * config.flow_jepa_grid_size
        * config.flow_jepa_grid_size
    )
    baseline = torch.zeros(1, token_count, config.hidden_size)
    late = baseline.clone()
    grouped_late = late.reshape(
        1,
        config.future_anchors,
        config.num_cameras,
        config.flow_jepa_grid_size,
        config.flow_jepa_grid_size,
        config.hidden_size,
    )
    grouped_late[:, -1, 0, 0, 0, 0] = 1.0
    base_refined, base_progress, _ = organizer(baseline)
    late_refined, late_progress, metrics = organizer(late)
    # Horizon encodings are selector information, not a constant value source.
    # With no observable W evidence the stage write must remain exactly zero.
    torch.testing.assert_close(base_refined, baseline)
    torch.testing.assert_close(base_progress, baseline)
    torch.testing.assert_close(
        late_refined - late,
        config.flow_jepa_interval_stage_update_scale * late_progress,
    )
    assert not hasattr(organizer, "progress_out")
    difference = (
        (late_refined - base_refined).abs().sum(dim=-1)
        + (late_progress - base_progress).abs().sum(dim=-1)
    ).reshape(
        1,
        config.future_anchors,
        config.num_cameras,
        config.flow_jepa_grid_size,
        config.flow_jepa_grid_size,
    )
    # A last-interval perturbation cannot leak backward in horizon time or
    # sideways into another camera/xy precision address.
    assert float(difference[:, :-1].max()) == 0.0
    spatial_elsewhere = difference[:, -1].clone()
    spatial_elsewhere[:, 0, 0, 0] = 0.0
    assert float(spatial_elsewhere.max()) == 0.0
    assert float(difference[:, -1, 0, 0, 0]) > 0.0
    assert float(metrics["flow_jepa_interval_stage_active"]) == 1.0

    attached = torch.randn_like(baseline, requires_grad=True)
    refined, progress, _ = organizer(attached)
    (refined.square().mean() + progress.square().mean()).backward()
    assert attached.grad is not None and torch.isfinite(attached.grad).all()
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and float(parameter.grad.abs().sum()) > 0.0
        for parameter in organizer.parameters()
    )


def test_v106_interval_loss_and_complete_contract_are_explicit() -> None:
    config = _complete_v106_config()
    trainer = _complete_v106_trainer()
    _validate_complete_v106_model_contract(config, trainer)
    assert _validate_required_model_contract("v106", config, trainer) == "v106"
    for field in (
        "flow_jepa_interval_stage_delta",
        "flow_jepa_variance_safe_routing",
        "flow_jepa_complete_numerical_contract",
    ):
        with pytest.raises(ValueError, match=field):
            _validate_complete_v106_model_contract(
                replace(config, **{field: 0}),
                trainer,
            )
    for field, unsafe in (
        ("flow_jepa_routing_norm_floor", 0.249),
        ("flow_jepa_correlation_rms_floor", 0.099),
        ("flow_jepa_visibility_transition_fraction", 0.099),
    ):
        with pytest.raises(ValueError, match=field):
            _validate_complete_v106_model_contract(
                replace(config, **{field: unsafe}),
                trainer,
            )
    with pytest.raises(ValueError, match="flow_jepa_interval_stage_loss_weight"):
        _validate_complete_v106_model_contract(
            config,
            replace(trainer, flow_jepa_interval_stage_loss_weight=0.0),
        )
    with pytest.raises(ValueError, match="interval_support_offsets"):
        _validate_complete_v106_model_contract(
            replace(
                config,
                flow_jepa_interval_support_offsets=(
                    4,
                    8,
                    16,
                    20,
                    24,
                    28,
                    32,
                    36,
                    40,
                    44,
                    48,
                ),
            ),
            trainer,
        )

    batch, anchors, cells, hidden = 2, 4, 6, 8
    current = torch.randn(batch, cells, hidden)
    target = torch.randn(batch, anchors * cells, hidden)
    endpoint = torch.randn_like(target)
    prediction = target.detach().clone().requires_grad_(True)
    terms = flow_jepa_interval_stage_terms(
        {
            "pred_physical_velocity": torch.zeros(batch, 1, 1),
            "flow_jepa_interval_progress_pred": prediction,
            "flow_jepa_interval_progress_target": target,
            "flow_jepa_interval_endpoint_target": endpoint,
            "flow_jepa_current_target": current,
            "flow_jepa_future_target_mask": torch.ones(
                batch, anchors * cells, dtype=torch.bool
            ),
        }
    )
    assert torch.isfinite(terms["flow_jepa_interval_stage"])
    terms["flow_jepa_interval_stage"].backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()

    # Near-zero initialization is part of the deployed contract. Direction
    # supervision must have a finite, bounded derivative at zero.
    zero_prediction = torch.zeros_like(target, requires_grad=True)
    zero_terms = flow_jepa_interval_stage_terms(
        {
            "pred_physical_velocity": torch.zeros(batch, 1, 1),
            "flow_jepa_interval_progress_pred": zero_prediction,
            "flow_jepa_interval_progress_target": target,
            "flow_jepa_interval_endpoint_target": endpoint,
            "flow_jepa_current_target": current,
            "flow_jepa_future_target_mask": torch.ones(
                batch, anchors * cells, dtype=torch.bool
            ),
        }
    )
    zero_terms["flow_jepa_interval_stage"].backward()
    assert zero_prediction.grad is not None
    assert torch.isfinite(zero_prediction.grad).all()
    assert float(zero_prediction.grad.norm()) < 100.0
    assert (
        float(zero_terms["flow_jepa_interval_stage_direction_floor_min"])
        >= 1e-3
    )

    zero_future_delta = torch.zeros_like(target, requires_grad=True)
    safe_future_loss = flow_jepa_future_prediction_loss(
        {
            "pred_physical_velocity": torch.zeros(batch, 1, 1),
            "flow_jepa_future_pred": torch.zeros_like(target),
            "flow_jepa_future_delta_pred": zero_future_delta,
            "flow_jepa_future_target": target,
            "flow_jepa_current_target": current,
            "flow_jepa_future_target_mask": torch.ones(
                batch, anchors * cells, dtype=torch.bool
            ),
            "flow_jepa_variance_safe_routing": torch.tensor(1.0),
        },
        balance_horizons=True,
        reliable_normalization=True,
    )
    safe_future_loss.backward()
    assert zero_future_delta.grad is not None
    assert torch.isfinite(zero_future_delta.grad).all()
    assert float(zero_future_delta.grad.norm()) < 100.0


def test_exact_goal_and_history_nulls_remove_all_content_templates() -> None:
    torch.manual_seed(107)
    config = _flow_config(
        action_history_enabled=1,
        action_history_recent_tokens=1,
        action_history_summary_tokens=1,
        action_history_condition_exact_null=1,
        goal_conditioning_enabled=1,
        goal_token_count=2,
        goal_language_dim=16,
        goal_resampler_depth=1,
        goal_condition_exact_null=1,
    )
    seed = UnifiedCanvasSeed(config).eval()
    batch = 2
    common = {
        "noisy_physical": torch.randn(
            batch, config.action_horizon, config.physical_action_dim
        ),
        "state": torch.randn(batch, config.state_dim),
        "state_history": torch.randn(
            batch, config.visual_history_length, config.state_dim
        ),
        "executed_history": torch.randn(
            batch, config.executed_history_length, config.action_dim
        ),
        "proposal_tokens": torch.zeros(
            batch, config.action_horizon, config.hidden_size
        ),
        "proposal_keep": torch.ones(batch),
        "rollout_init": torch.randn(
            batch, config.future_token_count, config.hidden_size
        ),
        "stage_init": torch.randn(batch, 1, config.hidden_size),
        "goal_condition_keep": torch.zeros(batch),
        "action_history_condition_keep": torch.zeros(batch),
    }
    first, first_slices = seed(
        **common,
        executed_memory=torch.randn(
            batch, config.action_history_token_count, config.hidden_size
        ),
        goal_tokens=torch.randn(
            batch, config.goal_token_count, config.hidden_size
        ),
    )
    second, second_slices = seed(
        **common,
        executed_memory=10.0
        * torch.randn(
            batch, config.action_history_token_count, config.hidden_size
        ),
        goal_tokens=10.0
        * torch.randn(
            batch, config.goal_token_count, config.hidden_size
        ),
    )
    for name in ("task", "executed"):
        assert torch.equal(
            first[:, first_slices[name]], second[:, second_slices[name]]
        )

    phase = StatelessPhaseAdapter(config.hidden_size, 4).eval()
    visual = torch.randn(batch, 8, config.hidden_size)
    first_phase, first_condition, _ = phase(
        goal_tokens=first[:, first_slices["task"]],
        history_tokens=first[:, first_slices["executed"]],
        state_tokens=first[:, first_slices["state"]],
        visual_tokens=visual,
    )
    second_phase, second_condition, _ = phase(
        goal_tokens=second[:, second_slices["task"]],
        history_tokens=second[:, second_slices["executed"]],
        state_tokens=second[:, second_slices["state"]],
        visual_tokens=visual,
    )
    assert torch.equal(first_phase, second_phase)
    assert torch.equal(first_condition, second_condition)


def test_stateless_phase_only_conditions_world_and_detail_queries() -> None:
    torch.manual_seed(108)
    config = _typed_332_config(
        flow_jepa_raw_activation_checkpoint=0,
        action_history_enabled=1,
        action_history_recent_tokens=1,
        action_history_summary_tokens=1,
        action_history_condition_exact_null=1,
        goal_conditioning_enabled=1,
        goal_token_count=2,
        goal_language_dim=16,
        goal_language_max_tokens=5,
        goal_resampler_depth=1,
        goal_condition_exact_null=1,
        stateless_phase_enabled=1,
        stateless_phase_count=4,
        stateless_phase_query_scale=0.10,
    )
    system = V39PolicySystem(config).train()
    batch = 1
    output = system.flow_training_forward(
        _visual(config, batch=batch),
        torch.randn(batch, config.visual_history_length, config.state_dim),
        torch.randn(batch, config.executed_history_length, config.action_dim),
        torch.randn(batch, config.state_dim),
        torch.randn(batch, config.action_horizon, config.action_dim),
        target_visual=torch.randn(
            batch,
            config.future_anchors,
            config.visual_history_length,
            config.num_cameras,
            config.patches_per_camera,
            config.visual_token_dim,
        ),
        raw_visual=_raw_visual(config, batch=batch),
        goal_language_tokens=torch.randn(
            batch, 3, config.goal_language_dim
        ),
        goal_language_mask=torch.ones(batch, 3, dtype=torch.bool),
        make_counterfactuals=False,
    )
    assert 0.0 <= float(output["flow_jepa_phase_entropy"]) <= 1.0
    assert float(output["flow_jepa_phase_detail_query_norm"]) > 0.0
    assert float(output["flow_jepa_condition_detail_query_norm"]) > 0.0
    assert float(output["attnres_world_to_policy_phase_query_norm"]) > 0.0
    assert float(output["attnres_world_to_policy_condition_query_norm"]) > 0.0
    for depth in range(1, config.flow_jepa_world_blocks + 1):
        assert (
            float(output[f"flow_jepa_world_block_query_delta_norm_w{depth}"])
            > 0.0
        )
    output["pred_physical_velocity"].float().square().mean().backward()
    phase_modules = (
        system.planner.stateless_phase_adapter,
        system.planner.phase_world_query_proj,
        system.planner.condition_world_query_proj,
        system.planner.late_raw_detail_reader.phase_query_proj,
        system.planner.late_raw_detail_reader.condition_query_proj,
        *tuple(system.planner.phase_world_block_query_proj or ()),
        *tuple(system.planner.condition_world_block_query_proj or ()),
    )
    assert all(module is not None for module in phase_modules)
    for module in phase_modules:
        assert module is not None
        assert any(
            parameter.grad is not None
            and float(parameter.grad.abs().sum()) > 0.0
            for parameter in module.parameters()
        )
    trainer = V39PolicyTrainerConfig(
        training_stage="policy",
        contract_mode="layer_adapter",
        lr=8e-5,
        single_stage_role_lr=1,
    )
    groups = _optimizer_groups(system, trainer)
    owned = [
        id(parameter)
        for group in groups
        for parameter in group["params"]
        if parameter.requires_grad
    ]
    trainable = {
        id(parameter) for parameter in system.parameters() if parameter.requires_grad
    }
    assert len(owned) == len(set(owned))
    assert set(owned) == trainable


def test_v102_late_detail_attention_does_not_mix_camera_charts() -> None:
    torch.manual_seed(109)
    config = _v102_config()
    reader = LateRawDetailPolicyReader(config).eval()
    batch = 1
    trajectory = torch.randn(
        batch,
        config.action_horizon * config.action_basis_tokens,
        config.hidden_size,
    )
    rollout = torch.randn(
        batch,
        config.future_anchors
        * config.num_cameras
        * config.future_grid_size**2,
        config.hidden_size,
    )
    cells = config.future_grid_size**2
    selector = torch.randn(
        batch,
        config.num_cameras,
        cells,
        config.hidden_size,
    )
    values = torch.zeros_like(selector)
    values[:, 0] = torch.randn_like(values[:, 0])
    detail = LateRawDetailEvidence(
        selector_tokens=selector.flatten(1, 2),
        value_tokens=values.flatten(1, 2),
    )
    changed_other_camera = selector.clone()
    changed_other_camera[:, 1] = 100.0 * torch.randn_like(
        changed_other_camera[:, 1]
    )
    changed_detail = LateRawDetailEvidence(
        selector_tokens=changed_other_camera.flatten(1, 2),
        value_tokens=values.flatten(1, 2),
    )
    with torch.no_grad():
        baseline, _ = reader(trajectory, rollout, detail)
        changed, _ = reader(trajectory, rollout, changed_detail)
    # Camera 1 has zero values. Its keys must not enter camera 0's softmax
    # denominator, otherwise changing them would alter camera 0's contribution.
    torch.testing.assert_close(baseline, changed)


def test_v102_world_write_is_anchor_camera_only_after_dropout() -> None:
    torch.manual_seed(104)
    config = _v102_config()
    block = TemporalDynamicsBoundDiTBlock(config, role="world")
    tokens = (
        config.future_anchors
        * config.num_cameras
        * config.future_grid_size**2
    )
    update = torch.randn(2, tokens, config.hidden_size)
    structured = block._structure_world_rollout_update(update)
    grouped = structured.reshape(
        2,
        config.future_anchors,
        config.num_cameras,
        config.future_grid_size,
        config.future_grid_size,
        config.hidden_size,
    )
    spatial_mean = grouped.mean(dim=(3, 4), keepdim=True)
    torch.testing.assert_close(grouped, spatial_mean.expand_as(grouped))
    original_mean = update.reshape_as(grouped).mean(dim=(3, 4), keepdim=True)
    torch.testing.assert_close(spatial_mean, original_mean)


def test_v102_full_world_block_writes_no_xy_specific_residual() -> None:
    torch.manual_seed(108)
    config = _v102_config(dropout=0.45)
    block = TemporalDynamicsBoundDiTBlock(config, role="world").train()
    lengths = {
        "task": 1,
        "state": 1,
        "state_history": 1,
        "executed": 1,
        "proposal": 1,
        "registers": config.canvas_registers,
        "trajectory": config.action_horizon * config.action_basis_tokens,
        "stage": 0,
        "rollout": (
            config.future_anchors
            * config.num_cameras
            * config.future_grid_size**2
        ),
    }
    slices: dict[str, slice] = {}
    cursor = 0
    for name, length in lengths.items():
        slices[name] = slice(cursor, cursor + int(length))
        cursor += int(length)
    canvas = torch.randn(2, cursor, config.hidden_size)
    output, _ = block(
        canvas,
        torch.randn(2, 17, config.hidden_size),
        torch.randn(2, config.hidden_size),
        slices,
    )
    rollout_delta = (
        output[:, slices["rollout"]] - canvas[:, slices["rollout"]]
    ).reshape(
        2,
        config.future_anchors,
        config.num_cameras,
        config.future_grid_size,
        config.future_grid_size,
        config.hidden_size,
    )
    spatial_mean = rollout_delta.mean(dim=(3, 4), keepdim=True)
    torch.testing.assert_close(
        rollout_delta,
        spatial_mean.expand_as(rollout_delta),
        rtol=1e-5,
        atol=1e-6,
    )


def test_v102_policy_workspace_pools_basis_inside_each_horizon() -> None:
    config = _v102_config(
        final_action_decoder="evidence_latent_mmdit_action",
        latent_cvae_mmdit_depth=1,
        layer_contract_adapters=1,
    )
    decoder = EvidenceLatentMMDiTActionDecoder(config)
    decoder.top_policy_workspace_lift = torch.nn.Identity()
    basis = config.action_basis_tokens
    tokens = torch.arange(
        config.action_horizon, dtype=torch.float32
    )[None, :, None, None].expand(
        1, config.action_horizon, basis, config.hidden_size
    )
    tokens = tokens + torch.arange(basis, dtype=torch.float32)[
        None, None, :, None
    ]
    aligned = decoder._lift_policy_workspace(
        tokens.reshape(
            1,
            config.action_horizon * basis,
            config.hidden_size,
        )
    )
    expected = tokens.mean(dim=2)
    torch.testing.assert_close(aligned, expected)


def test_v102_event_gradient_reaches_late_raw_action_path() -> None:
    torch.manual_seed(105)
    config = _v102_config(
        final_action_decoder="evidence_latent_mmdit_action",
        layer_contract_adapters=1,
        layer_contract_adapter_dim=32,
        latent_cvae_mmdit_depth=1,
    )
    system = V39PolicySystem(config).train()
    decoder = system.planner.evidence_latent_mmdit_action_decoder
    assert decoder is not None
    with torch.no_grad():
        torch.nn.init.normal_(decoder.event_head[-1].weight, mean=0.0, std=0.02)
    batch = 1
    output = system.flow_training_forward(
        _visual(config, batch=batch),
        torch.randn(batch, config.visual_history_length, config.state_dim),
        torch.randn(batch, config.executed_history_length, config.action_dim),
        torch.randn(batch, config.state_dim),
        torch.randn(batch, config.action_horizon, config.action_dim),
        target_visual=torch.randn(
            batch,
            config.target_future_count,
            config.visual_history_length,
            config.num_cameras,
            config.patches_per_camera,
            config.visual_token_dim,
        ),
        raw_visual=_raw_visual(config, batch=batch),
        make_counterfactuals=False,
    )
    event_target = torch.ones(
        batch * config.action_horizon, dtype=torch.long
    )
    event_loss = F.cross_entropy(
        output["event_logits"].float().reshape(-1, 3),
        event_target,
    )
    # Event timing is decoded from the same action tokens.  Test it in
    # isolation so action-flow supervision cannot mask a disconnected event
    # route.
    event_loss.backward()
    late_reader = system.planner.late_raw_detail_reader
    assert late_reader is not None
    assert late_reader.query_proj.weight.grad is not None
    assert late_reader.key_proj.weight.grad is not None
    encoder = system.planner.flow_dino_evidence
    assert encoder is not None and encoder.raw_address_reader is not None
    assert encoder.raw_address_reader.query.grad is not None
    assert encoder.raw_flow is not None
    assert encoder.raw_flow.high.update[-1].weight.grad is not None
    for gradient in (
        late_reader.query_proj.weight.grad,
        late_reader.key_proj.weight.grad,
        encoder.raw_address_reader.query.grad,
        encoder.raw_flow.high.update[-1].weight.grad,
    ):
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert bool(torch.count_nonzero(gradient.detach()))
    assert float(output["flow_jepa_world_spatial_residual_norm"]) < 1e-5
    assert float(output["flow_jepa_late_detail_update_norm"]) > 0.0
    assert float(output["evidence_top_policy_workspace_horizon_pool"]) == 1.0


def test_raw_reader_router_never_scales_detail_value_amplitude() -> None:
    reader = _RawDeformableAddressReader(
        8, 16, 4, radius=1, heads=4
    ).eval()
    with torch.no_grad():
        reader.query.zero_()
        reader.flow_prior_strength.zero_()
        for module in (reader.source_proj, reader.key_proj):
            for parameter in module.parameters():
                parameter.zero_()
    source = torch.randn(1, 8, 16, 16)
    target = torch.randn_like(source)
    flow = torch.zeros(1, 2, 16, 16)
    confidence = torch.ones(1, 1, 16, 16)
    grounding = torch.zeros(1, 4, 4, 16)
    _, low_value, _ = reader(
        source, target, flow, confidence, grounding, torch.zeros(1, 1, 4, 4)
    )
    _, high_value, _ = reader(
        source, target, flow, confidence, grounding, torch.ones(1, 1, 4, 4)
    )
    torch.testing.assert_close(low_value, high_value)


def test_raw_grounding_cpu_bf16_forward_backward_is_finite() -> None:
    torch.manual_seed(971)
    config = _raw_role_config(flow_jepa_zero_flow_guard=1)
    encoder = FlowDINOEvidenceEncoder(config).train()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        pack = encoder(_visual(config, batch=1), raw_visual=_raw_visual(config))
        rollout_count = (
            config.future_anchors
            * config.num_cameras
            * config.flow_jepa_grid_size**2
        )
        grounding_canvas = torch.randn(
            1, rollout_count + 1, config.hidden_size
        )
        _, values, _ = encoder.refine_raw_evidence(
            pack,
            grounding_canvas,
            {"rollout": slice(1, 1 + rollout_count)},
        )
        objective = values.float().square().mean() + sum(pack.losses.values())
        assert "flow_jepa_identity_advantage_loss" in pack.losses
        assert float(pack.metrics["flow_jepa_zero_flow_guard"]) == 1.0
    assert bool(torch.isfinite(objective))
    objective.backward()
    assert encoder.flow.delta_head[-1].weight.grad is not None
    assert encoder.raw_flow is not None
    assert encoder.raw_flow.high.update[-1].weight.grad is not None


def test_v99_eval_reports_zero_and_shuffled_flow_reader_interventions() -> None:
    torch.manual_seed(972)
    config = _raw_role_config(flow_jepa_zero_flow_guard=1)
    encoder = FlowDINOEvidenceEncoder(config).eval()
    with torch.no_grad():
        pack = encoder(_visual(config, batch=1), raw_visual=_raw_visual(config))
        rollout_count = (
            config.future_anchors
            * config.num_cameras
            * config.flow_jepa_grid_size**2
        )
        grounding_canvas = torch.randn(
            1, rollout_count + 1, config.hidden_size
        )
        _, _, metrics = encoder.refine_raw_evidence(
            pack,
            grounding_canvas,
            {"rollout": slice(1, 1 + rollout_count)},
        )
    for name in (
        "flow_jepa_raw_address_zero_flow_value_delta",
        "flow_jepa_raw_address_shuffled_flow_value_delta",
    ):
        assert name in metrics
        assert bool(torch.isfinite(metrics[name]))


def test_v98_transient_address_intervention_preserves_checkpoint_and_camera_identity() -> None:
    torch.manual_seed(974)
    config = _raw_role_config(
        flow_jepa_zero_flow_guard=1,
        flow_jepa_complementary_raw_detail=1,
    )
    encoder = FlowDINOEvidenceEncoder(config).eval()
    state_keys = tuple(encoder.state_dict())
    flow = torch.arange(3 * config.num_cameras * 2 * 4 * 4).reshape(
        3 * config.num_cameras, 2, 4, 4
    )
    shuffled, fallback = encoder._intervened_raw_address_flow(
        flow.float(), batch=3, mode="shuffle"
    )
    expected = flow.reshape(3, config.num_cameras, 2, 4, 4).roll(1, dims=0)
    torch.testing.assert_close(
        shuffled.reshape_as(expected), expected.float()
    )
    assert not fallback

    encoder.set_raw_address_eval_intervention("none")
    with torch.no_grad():
        pack = encoder(_visual(config, batch=2), raw_visual=_raw_visual(config, batch=2))
        rollout_count = (
            config.future_anchors
            * config.num_cameras
            * config.flow_jepa_grid_size**2
        )
        grounding_canvas = torch.randn(2, rollout_count + 1, config.hidden_size)
        encoder.refine_raw_evidence(
            pack,
            grounding_canvas,
            {"rollout": slice(1, 1 + rollout_count)},
        )
    captured = encoder.raw_address_eval_metrics()
    for name in (
        "flow_jepa_raw_flow_grid_magnitude",
        "flow_jepa_raw_seed_reliability",
        "flow_jepa_raw_address_logit_advantage",
        "flow_jepa_raw_address_zero_flow_value_delta",
        "flow_jepa_raw_address_shuffled_flow_value_delta",
    ):
        assert name in captured
        assert math.isfinite(captured[name])
    encoder.clear_raw_address_eval_intervention()
    assert encoder.raw_address_eval_metrics() == {}
    encoder.set_raw_address_eval_intervention("detail_zero")
    with torch.no_grad():
        pack = encoder(_visual(config, batch=2), raw_visual=_raw_visual(config, batch=2))
        encoder.refine_raw_evidence(
            pack,
            grounding_canvas,
            {"rollout": slice(1, 1 + rollout_count)},
        )
    captured = encoder.raw_address_eval_metrics()
    assert captured["flow_jepa_raw_address_intervention_code"] == pytest.approx(4.0)
    assert (
        captured["flow_jepa_raw_post_reader_detail_value_intervention_delta"] > 0.0
    )
    encoder.clear_raw_address_eval_intervention()
    assert tuple(encoder.state_dict()) == state_keys


def test_future_jepa_diagnostics_preserve_real_horizon_offsets() -> None:
    pred = torch.zeros(1, 6, 4)
    target = torch.zeros_like(pred)
    target[:, 2:4] = 1.0
    target[:, 4:] = 2.0
    rows = flow_jepa_future_horizon_diagnostics(
        {
            "flow_jepa_future_pred": pred,
            "flow_jepa_future_target": target,
            "flow_jepa_future_target_mask": torch.ones(1, 6, dtype=torch.bool),
            "flow_jepa_future_offsets": (4, 12, 48),
        }
    )
    assert set(rows) == {4, 12, 48}
    assert float(rows[4]) < float(rows[12]) < float(rows[48])


def test_loader_shuffle_is_independent_of_architecture_rng_consumption() -> None:
    dataset = TensorDataset(torch.arange(24))

    def order(draws: int) -> list[int]:
        generator = torch.Generator().manual_seed(1234)
        loader = make_loader(
            dataset,
            batch_size=4,
            workers=0,
            shuffle=True,
            device=torch.device("cpu"),
            generator=generator,
        )
        _ = torch.randn(draws)
        return [int(value) for batch in loader for value in batch[0]]

    assert order(1) == order(10_000)


def test_v101_source_aligned_raw_detail_updates_source_not_latest_dino_chart() -> None:
    torch.manual_seed(1001)
    config = _raw_role_config(
        flow_jepa_zero_flow_guard=1,
        flow_jepa_complementary_raw_detail=1,
        flow_jepa_strict_role_visual_path=1,
        flow_jepa_source_aligned_raw_fusion=1,
    )
    encoder = FlowDINOEvidenceEncoder(config).eval()
    with torch.no_grad():
        pack = encoder(_visual(config, batch=1), raw_visual=_raw_visual(config))
        original = pack.selector_tokens.clone()
        rollout_count = (
            config.future_anchors
            * config.num_cameras
            * config.flow_jepa_grid_size**2
        )
        selector, _, metrics = encoder.refine_raw_evidence(
            pack,
            torch.randn(1, rollout_count, config.hidden_size),
            {"rollout": slice(0, rollout_count)},
        )
    spatial = config.num_cameras * config.flow_jepa_grid_size**2
    source_start = (config.visual_history_length - 2) * spatial
    latest_start = (config.visual_history_length - 1) * spatial
    assert not torch.allclose(
        selector[:, source_start : source_start + spatial],
        original[:, source_start : source_start + spatial],
    )
    torch.testing.assert_close(
        selector[:, latest_start : latest_start + spatial],
        original[:, latest_start : latest_start + spatial],
    )
    assert float(metrics["flow_jepa_raw_detail_fused_with_source_dino"]) == 1.0
    assert float(metrics["flow_jepa_raw_detail_fused_with_latest_dino"]) == 0.0


def test_v101_action_path_interventions_are_transient_and_preserve_camera_identity() -> None:
    config = _raw_role_config(
        flow_jepa_strict_role_visual_path=1,
        flow_jepa_policy_workspace_fixed_fusion=1,
    )
    planner = V39PolicySystem(config).planner.eval()
    state_keys = tuple(planner.state_dict())

    batch = 2
    anchors = config.future_anchors
    cameras = config.num_cameras
    grid = config.future_grid_size
    world_entry = torch.zeros(
        batch,
        anchors,
        cameras,
        grid,
        grid,
        config.hidden_size,
    )
    for camera in range(cameras):
        world_entry[:, :, camera] = float(100 * (camera + 1))
    world_residual = torch.arange(
        world_entry.numel(), dtype=torch.float32
    ).reshape_as(world_entry)
    world_output = world_entry + world_residual
    entry_flat = world_entry.reshape(batch, -1, config.hidden_size)
    output_flat = world_output.reshape_as(entry_flat)

    planner.set_action_path_eval_intervention(
        "world_residual_spatiotemporal_shuffle"
    )
    world_shuffled = planner._intervene_world_rollout(
        output_flat,
        world_entry_rollout=entry_flat,
    ).reshape_as(world_entry)
    expected_residual = world_residual.roll(shifts=1, dims=1)
    expected_residual = expected_residual.roll(shifts=max(grid // 2, 1), dims=3)
    expected_residual = expected_residual.roll(shifts=max(grid // 3, 1), dims=4)
    torch.testing.assert_close(world_shuffled, world_entry + expected_residual)
    assert planner.action_path_eval_intervention_state()["apply_count"] == 1

    planner.set_action_path_eval_intervention("world_residual_anchor_shuffle")
    anchor_shuffled = planner._intervene_world_rollout(
        output_flat,
        world_entry_rollout=entry_flat,
    ).reshape_as(world_entry)
    torch.testing.assert_close(
        anchor_shuffled,
        world_entry + world_residual.roll(shifts=1, dims=1),
    )

    planner.set_action_path_eval_intervention("world_residual_spatial_shuffle")
    spatial_shuffled = planner._intervene_world_rollout(
        output_flat,
        world_entry_rollout=entry_flat,
    ).reshape_as(world_entry)
    expected_spatial = world_residual.roll(shifts=max(grid // 2, 1), dims=3)
    expected_spatial = expected_spatial.roll(shifts=max(grid // 3, 1), dims=4)
    torch.testing.assert_close(spatial_shuffled, world_entry + expected_spatial)

    planner.set_action_path_eval_intervention("world_residual_zero")
    torch.testing.assert_close(
        planner._intervene_world_rollout(
            output_flat,
            world_entry_rollout=entry_flat,
        ),
        entry_flat,
    )

    workspace = torch.arange(
        batch
        * config.action_horizon
        * config.action_basis_tokens
        * config.hidden_size,
        dtype=torch.float32,
    ).reshape(
        batch,
        config.action_horizon * config.action_basis_tokens,
        config.hidden_size,
    )
    planner.set_action_path_eval_intervention("policy_temporal_shuffle")
    policy_shuffled = planner._intervene_policy_workspace(workspace).reshape(
        batch,
        config.action_horizon,
        config.action_basis_tokens,
        config.hidden_size,
    )
    expected = workspace.reshape_as(policy_shuffled).roll(
        max(config.action_horizon // 2, 1), dims=1
    )
    torch.testing.assert_close(policy_shuffled, expected)

    planner.set_action_path_eval_intervention("policy_zero")
    torch.testing.assert_close(
        planner._intervene_policy_workspace(workspace), torch.zeros_like(workspace)
    )
    planner.clear_action_path_eval_intervention()
    assert planner.action_path_eval_intervention_state() == {
        "mode": "disabled",
        "apply_count": 0,
    }
    assert tuple(planner.state_dict()) == state_keys

    planner.train()
    with pytest.raises(RuntimeError, match="evaluation-only"):
        planner.set_action_path_eval_intervention("policy_zero")


def test_v101_action_path_probe_selection_includes_event_and_uniform_batches() -> None:
    class _SignalDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return 20

        def __getitem__(self, index: int) -> torch.Tensor:
            return torch.tensor(index)

        def training_information_signals(
            self,
            *,
            gripper_index: int,
            event_threshold: float,
        ) -> tuple[np.ndarray, np.ndarray]:
            assert gripper_index == 6
            assert event_threshold == pytest.approx(0.1)
            event = np.zeros(20, dtype=bool)
            event[[4, 5, 14, 15]] = True
            return np.zeros(20, dtype=np.float32), event

    loader = torch.utils.data.DataLoader(_SignalDataset(), batch_size=2, shuffle=False)
    selected, metadata = _action_path_probe_batch_selection(
        loader=loader,
        planned_batches=len(loader),
        budget=4,
        gripper_index=6,
        event_threshold=0.1,
    )
    assert len(selected) == 4
    assert metadata["selection_strategy"] == "uniform_plus_gripper_event"
    assert metadata["event_candidate_batches"] == 2
    assert metadata["selected_event_batches"] == 2
    assert set(metadata["selected_event_batch_indices"]) <= selected
    assert 1 in selected and len(loader) in selected


def test_v101_action_path_probe_selection_stratifies_episode_clusters() -> None:
    class _EpisodeSignalDataset(torch.utils.data.Dataset):
        def __init__(self) -> None:
            self.refs = [
                SimpleNamespace(episode_idx=episode_id)
                for episode_id in range(4)
                for _ in range(10)
            ]

        def __len__(self) -> int:
            return len(self.refs)

        def __getitem__(self, index: int) -> torch.Tensor:
            return torch.tensor(index)

        def training_information_signals(
            self,
            *,
            gripper_index: int,
            event_threshold: float,
        ) -> tuple[np.ndarray, np.ndarray]:
            assert gripper_index == 6
            assert event_threshold == pytest.approx(0.1)
            event = np.zeros(len(self.refs), dtype=bool)
            event[[2, 12, 22, 32]] = True
            return np.zeros(len(self.refs), dtype=np.float32), event

    loader = torch.utils.data.DataLoader(
        _EpisodeSignalDataset(), batch_size=2, shuffle=False
    )
    selected, metadata = _action_path_probe_batch_selection(
        loader=loader,
        planned_batches=len(loader),
        budget=8,
        gripper_index=6,
        event_threshold=0.1,
    )
    assert len(selected) == 8
    assert (
        metadata["selection_strategy"]
        == "episode_stratified_uniform_plus_gripper_event"
    )
    assert metadata["candidate_episode_ids"] == [0, 1, 2, 3]
    assert metadata["selected_episode_ids"] == [0, 1, 2, 3]
    assert metadata["selected_episode_count"] == 4
    assert metadata["selected_event_episode_ids"] == [0, 1, 2, 3]
    assert metadata["selected_event_batches"] == 4


def test_v101_action_path_paired_metrics_are_horizon_specific() -> None:
    baseline = np.zeros((4, 24, 2), dtype=np.float64)
    tail_only = baseline.copy()
    tail_only[:, 12:] = 0.5
    rows = _action_path_paired_metrics(
        joined={"baseline": baseline, "tail_only": tail_only},
        target=baseline,
        episode_ids=np.asarray([0, 0, 1, 1]),
        action_offsets=(4, 12, 24),
        bootstrap_reps=1000,
        bootstrap_seed=101,
    )
    result = rows["tail_only"]
    assert result["bands"]["1_4"]["mse_delta_vs_baseline"] == 0.0
    assert result["bands"]["5_12"]["mse_delta_vs_baseline"] == 0.0
    assert result["bands"]["13_24"]["mse_delta_vs_baseline"] > 0.0
    assert result["bands"]["13_24"]["mse_delta_ci"]["ci95_low"] > 0.0


def test_role_hierarchy_has_332_ownership_and_action_loss_gradient() -> None:
    torch.manual_seed(98)
    config = _raw_role_config(
        final_action_decoder="evidence_latent_mmdit_action",
        layer_contract_adapters=1,
        layer_contract_adapter_dim=32,
        latent_cvae_mmdit_depth=1,
        flow_jepa_zero_flow_guard=1,
        flow_jepa_complementary_raw_detail=1,
        flow_jepa_strict_role_visual_path=1,
        flow_jepa_source_aligned_raw_fusion=1,
        flow_jepa_policy_workspace_fixed_fusion=1,
        flow_jepa_teacher_balanced_target_mask=1,
        flow_jepa_teacher_mask_past_fraction=0.25,
        flow_jepa_teacher_mask_change_fraction=0.50,
    )
    system = V39PolicySystem(config).train()
    assert system.planner.block_roles == (
        "grounding",
        "grounding",
        "grounding",
        "world",
        "world",
        "world",
        "policy",
        "policy",
    )
    batch = 1
    policy_visual_calls = 0
    world_visual_calls = 0
    adapter_inputs: dict[str, object] = {}

    def _policy_visual_hook(_module, _inputs, _output) -> None:
        nonlocal policy_visual_calls
        policy_visual_calls += 1

    def _world_visual_hook(_module, _inputs, _output) -> None:
        nonlocal world_visual_calls
        world_visual_calls += 1

    def _adapter_hook(_module, _args, kwargs) -> None:
        adapter_inputs.update(kwargs)

    handles = [
        block.cross.register_forward_hook(_policy_visual_hook)
        for block in system.planner.blocks[-2:]
    ]
    handles.extend(
        block.cross.register_forward_hook(_world_visual_hook)
        for block in system.planner.blocks[3:6]
    )
    assert system.planner.evidence_latent_mmdit_action_decoder is not None
    handles.append(
        system.planner.evidence_latent_mmdit_action_decoder.evidence_adapter.register_forward_pre_hook(
            _adapter_hook, with_kwargs=True
        )
    )
    output = system.flow_training_forward(
        _visual(config, batch=batch),
        torch.randn(batch, config.visual_history_length, config.state_dim),
        torch.randn(batch, config.executed_history_length, config.action_dim),
        torch.randn(batch, config.state_dim),
        torch.randn(batch, config.action_horizon, config.action_dim),
        target_visual=torch.randn(
            batch,
            config.future_anchors,
            config.visual_history_length,
            config.num_cameras,
            config.patches_per_camera,
            config.visual_token_dim,
        ),
        raw_visual=_raw_visual(config, batch=batch),
        make_counterfactuals=False,
    )
    for handle in handles:
        handle.remove()
    assert policy_visual_calls == 0
    assert world_visual_calls == 3
    assert adapter_inputs["visual_selector_tokens"] is None
    assert adapter_inputs["visual_value_tokens"] is None
    assert "visual" not in adapter_inputs["intent_memory"]
    assert len(adapter_inputs["layer_contracts"]) == config.flow_jepa_policy_blocks
    assert float(output["flow_jepa_raw_additive_detail_path"]) == 1.0
    assert float(output["flow_jepa_raw_detail_fused_with_source_dino"]) == 1.0
    assert float(output["flow_jepa_raw_detail_fused_with_latest_dino"]) == 0.0
    assert float(output["flow_jepa_refined_evidence_token_count"]) == float(
        output["flow_jepa_evidence_token_count"]
    )
    assert int(output["flow_jepa_raw_detail_token_count"]) == (
        config.num_cameras * config.flow_jepa_grid_size**2
    )
    assert float(output["evidence_top_policy_workspace_fixed_fusion"]) == 1.0
    assert float(output["flow_jepa_teacher_balanced_target_mask"]) == 1.0
    assert float(output["flow_jepa_teacher_mask_change_fraction"]) == 0.5
    action_objective = output["pred_physical_velocity"].float().square().mean()
    action_objective.backward()
    for block in system.planner.blocks:
        gradient = block.self_attn.in_proj_weight.grad
        assert gradient is not None
        assert float(gradient.float().abs().sum()) > 0.0
    assert system.planner.flow_dino_evidence is not None
    assert system.planner.flow_dino_evidence.raw_address_reader is not None
    assert system.planner.flow_dino_evidence.raw_address_reader.query.grad is not None
    decoder = system.planner.evidence_latent_mmdit_action_decoder
    assert decoder is not None and decoder.top_policy_workspace_lift is not None
    workspace_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in decoder.top_policy_workspace_lift.parameters()
        if parameter.grad is not None
    )
    assert workspace_grad > 0.0


def test_role_blocks_only_write_their_owned_canvas_regions() -> None:
    torch.manual_seed(99)
    config = _raw_role_config()
    slices = {
        "task": slice(0, 1),
        "state": slice(1, 2),
        "state_history": slice(2, 3),
        "executed": slice(3, 4),
        "proposal": slice(4, 5),
        "trajectory": slice(5, 7),
        "stage": slice(7, 7),
        "rollout": slice(7, 10),
        "registers": slice(10, 11),
    }
    canvas = torch.randn(2, 11, config.hidden_size)
    visual = torch.randn(2, 9, config.hidden_size)
    condition = torch.randn(2, config.hidden_size)
    owned = {
        "grounding": ("task", "state", "state_history", "executed", "proposal", "rollout", "registers"),
        "world": ("rollout",),
        "policy": ("trajectory",),
    }
    for role, names in owned.items():
        block = TemporalDynamicsBoundDiTBlock(config, role=role).eval()
        output, _ = block(canvas, visual, condition, slices)
        allowed = torch.zeros(11, dtype=torch.bool)
        for name in names:
            allowed[slices[name]] = True
        torch.testing.assert_close(output[:, ~allowed], canvas[:, ~allowed])
        assert not torch.allclose(output[:, allowed], canvas[:, allowed])


def test_grounding_rollout_is_clean_of_noisy_and_proposal_action() -> None:
    """G owns observation alignment; candidate-action consequence starts at W."""

    torch.manual_seed(115)
    config = _raw_role_config()
    slices = {
        "task": slice(0, 1),
        "state": slice(1, 2),
        "state_history": slice(2, 3),
        "executed": slice(3, 4),
        "proposal": slice(4, 5),
        "trajectory": slice(5, 7),
        "stage": slice(7, 7),
        "rollout": slice(7, 10),
        "registers": slice(10, 11),
    }
    canvas = torch.randn(2, 11, config.hidden_size)
    changed = canvas.clone()
    changed[:, slices["proposal"]] += 20.0 * torch.randn_like(
        changed[:, slices["proposal"]]
    )
    changed[:, slices["trajectory"]] += 20.0 * torch.randn_like(
        changed[:, slices["trajectory"]]
    )
    visual = torch.randn(2, 9, config.hidden_size)
    condition = torch.randn(2, config.hidden_size)

    grounding = TemporalDynamicsBoundDiTBlock(
        config, role="grounding"
    ).eval()
    world = TemporalDynamicsBoundDiTBlock(config, role="world").eval()
    grounding_base, _ = grounding(canvas, visual, condition, slices)
    grounding_changed, _ = grounding(changed, visual, condition, slices)
    torch.testing.assert_close(
        grounding_base[:, slices["rollout"]],
        grounding_changed[:, slices["rollout"]],
        rtol=0.0,
        atol=0.0,
    )

    world_base, _ = world(canvas, visual, condition, slices)
    world_changed, _ = world(changed, visual, condition, slices)
    assert not torch.allclose(
        world_base[:, slices["rollout"]],
        world_changed[:, slices["rollout"]],
    )


def test_late_bottleneck_keeps_far_horizon_spatial_and_refines_only_reader_chart() -> None:
    torch.manual_seed(101)
    config = _late_bottleneck_config()
    assert config.flow_jepa_target_offsets == (2, 4, 8)
    assert config.flow_jepa_action_offsets == (2, 4)
    encoder = FlowDINOEvidenceEncoder(config).train()
    pack = encoder(_visual(config, batch=1))
    grid = config.flow_jepa_grid_size
    future_count = config.future_anchors * config.num_cameras * grid**2
    assert pack.future_queries.shape == (1, future_count, config.hidden_size)
    assert pack.stage_query.shape == (1, 0, config.hidden_size)
    assert pack.patch_flow_forward.shape == (
        1,
        config.visual_history_length - 1,
        config.num_cameras,
        2,
        grid,
        grid,
    )
    assert int(pack.metrics["flow_jepa_native_grid_size"]) == 8
    assert int(pack.metrics["flow_jepa_coarse_grid_size"]) == grid
    torch.testing.assert_close(
        pack.metrics["flow_jepa_address_flow_mass"]
        + pack.metrics["flow_jepa_address_fallback_mass"],
        torch.ones(()),
    )
    target = torch.randn(
        1,
        config.future_anchors,
        config.visual_history_length,
        config.num_cameras,
        config.patches_per_camera,
        config.visual_token_dim,
        requires_grad=True,
    )
    future_target, stage_target = encoder.teacher_target(target, _visual(config, batch=1))
    assert future_target.shape == pack.future_queries.shape
    assert stage_target.shape == (1, 0, config.hidden_size)
    assert not future_target.requires_grad

    objective = (
        pack.selector_tokens.float().square().mean()
        + pack.future_queries.float().square().mean()
        + sum(pack.losses.values())
    )
    objective.backward()
    assert encoder.sparse_fine_flow is not None
    assert encoder.address_reader is not None
    assert encoder.detail_router is not None
    assert encoder.sparse_fine_flow.update[-1].weight.grad is not None
    assert encoder.address_reader.query.grad is not None
    assert encoder.detail_router[-1].weight.grad is not None


def test_late_bottleneck_full_policy_has_one_jepa_action_path_without_stage() -> None:
    base = _late_bottleneck_config()
    config = V39PolicyConfig(
        **{
            **base.__dict__,
            "layer_recurrent_consequence": 1,
            "layer_consequence_steps": 2,
            "final_action_decoder": "evidence_latent_mmdit_action",
            "layer_contract_adapters": 1,
            "layer_contract_adapter_dim": 32,
            "latent_cvae_mmdit_depth": 2,
        }
    )
    config.validate()
    system = V39PolicySystem(config).train()
    batch = 1
    output = system.flow_training_forward(
        _visual(config, batch=batch),
        torch.randn(batch, config.visual_history_length, config.state_dim),
        torch.randn(batch, config.executed_history_length, config.action_dim),
        torch.randn(batch, config.state_dim),
        torch.randn(batch, config.action_horizon, config.action_dim),
        target_visual=torch.randn(
            batch,
            config.future_anchors,
            config.visual_history_length,
            config.num_cameras,
            config.patches_per_camera,
            config.visual_token_dim,
        ),
        make_counterfactuals=False,
    )
    assert output["flow_jepa_future_pred"].shape == output["flow_jepa_future_target"].shape
    assert output["flow_jepa_current_target"].shape[1] * config.future_anchors == output[
        "flow_jepa_future_target"
    ].shape[1]
    assert "flow_jepa_stage_pred" not in output
    assert "flow_jepa_stage_target" not in output
    change_direction = flow_jepa_future_change_direction_loss(output)
    assert torch.isfinite(change_direction)
    objective = (
        output["pred_physical_velocity"].float().square().mean()
        + output["flow_jepa_future_pred"].float().square().mean()
        + output["flow_jepa_warp_loss"]
        + 0.10 * change_direction
    )
    objective.backward()
    encoder = system.planner.flow_dino_evidence
    assert encoder is not None and encoder.address_reader is not None
    assert encoder.address_reader.query.grad is not None
    assert system.planner.blocks[0].self_attn.in_proj_weight.grad is not None


def test_future_change_objective_has_no_hard_small_change_cutoff() -> None:
    torch.manual_seed(977)
    current = torch.zeros(1, 2, 4)
    target = current.clone()
    target[:, 0, 0] = 1e-5
    pred = (1e-5 * torch.randn_like(target)).requires_grad_()
    output = {
        "flow_jepa_future_pred": pred,
        "flow_jepa_future_target": target,
        "flow_jepa_current_target": current,
        "flow_jepa_future_target_mask": torch.ones(1, 2, dtype=torch.bool),
    }
    objective = flow_jepa_future_change_loss(output)
    assert torch.isfinite(objective)
    assert float(objective.detach()) > 0.0
    objective.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    assert float(pred.grad.abs().sum()) > 0.0


def test_patch_warp_uses_source_to_target_patch_coordinates() -> None:
    value = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    identity, valid = warp_patch_grid(value, torch.zeros(1, 2, 4, 4))
    torch.testing.assert_close(identity, value)
    assert valid.all()

    flow = torch.zeros(1, 2, 4, 4)
    flow[:, 0] = 1.0
    shifted, valid = warp_patch_grid(value, flow)
    torch.testing.assert_close(shifted[..., :-1], value[..., 1:])
    assert valid[..., :-1].all()
    assert not valid[..., -1].any()


def test_late_local_read_is_source_queried_and_offscreen_safe() -> None:
    torch.manual_seed(303)
    source = torch.randn(1, 8, 4, 4, requires_grad=True)
    target_key = torch.randn(1, 8, 4, 4, requires_grad=True)
    target_value = torch.randn(1, 8, 4, 4, requires_grad=True)
    confidence = torch.ones(1, 1, 2, 2)
    reader = _SoftFlowAddressReader(8, 2, radius=1, heads=2)
    read_key, read_value, _ = reader(
        source, target_key, target_value, torch.zeros(1, 2, 2, 2), confidence
    )
    (read_key.float().square().mean() + read_value.float().square().mean()).backward()
    assert source.grad is not None and float(source.grad.abs().sum()) > 0.0
    assert target_key.grad is not None and float(target_key.grad.abs().sum()) > 0.0
    assert target_value.grad is not None and float(target_value.grad.abs().sum()) > 0.0

    huge_flow = torch.full((1, 2, 2, 2), 100.0)
    read_key, read_value, metrics = reader(
        source.detach(), target_key.detach(), target_value.detach(), huge_flow, confidence
    )
    assert torch.isfinite(read_key).all() and torch.isfinite(read_value).all()
    torch.testing.assert_close(metrics["flow_mass"], torch.zeros(()))
    torch.testing.assert_close(metrics["fallback_mass"], torch.ones(()))

    refiner = _SparseFineFlowRefiner(
        6, 8, radius=1, grid=2, uncertainty_floor=0.03
    )
    estimate = refiner(
        torch.randn(1, 4, 4, 6),
        torch.randn(1, 4, 4, 6),
        huge_flow,
        torch.ones(1, 1, 2, 2),
    )
    assert torch.isfinite(estimate.flow).all()
    assert torch.isfinite(estimate.uncertainty).all()


def test_flow_dino_pack_has_exact_masks_typed_evidence_and_trainable_flow() -> None:
    torch.manual_seed(7)
    config = _flow_config()
    encoder = FlowDINOEvidenceEncoder(config)
    visual = _visual(config)
    pack = encoder(visual)

    grid_tokens = config.flow_jepa_grid_size**2
    content_count = config.visual_history_length * config.num_cameras * grid_tokens
    pair_count = (config.visual_history_length - 1) * config.num_cameras * grid_tokens
    evidence_count = content_count + 2 * pair_count
    future_count = config.future_anchors * config.num_cameras * grid_tokens
    assert pack.selector_tokens.shape == (2, evidence_count, config.hidden_size)
    assert pack.value_tokens.shape == pack.selector_tokens.shape
    assert pack.stage_query.shape == (2, 1, config.hidden_size)
    assert pack.future_queries.shape == (2, future_count, config.hidden_size)
    assert pack.future_target_mask.shape == (2, future_count)
    assert pack.patch_flow_forward.shape == (
        2,
        config.visual_history_length - 1,
        config.num_cameras,
        2,
        config.flow_jepa_grid_size,
        config.flow_jepa_grid_size,
    )
    assert torch.equal(
        pack.context_dropout_mask.sum(dim=(-1, -2)),
        torch.full(
            (2, config.visual_history_length, config.num_cameras),
            6,
            dtype=torch.long,
        ),
    )
    assert torch.equal(
        pack.future_target_mask.reshape(
            2,
            config.future_anchors,
            config.num_cameras,
            config.flow_jepa_grid_size,
            config.flow_jepa_grid_size,
        ).sum(dim=(-1, -2)),
        torch.full((2, config.future_anchors, config.num_cameras), 6, dtype=torch.long),
    )
    assert all(torch.isfinite(value) for value in pack.losses.values())
    assert 0.0 <= float(pack.metrics["flow_jepa_confidence_mean"]) <= 1.0
    assert 0.0 <= float(pack.metrics["flow_jepa_occlusion_fraction"]) <= 1.0

    objective = (
        pack.selector_tokens.float().square().mean()
        + pack.value_tokens.float().square().mean()
        + pack.stage_query.float().square().mean()
        + pack.future_queries.float().square().mean()
        + sum(pack.losses.values())
    )
    objective.backward()
    assert encoder.flow.delta_head[-1].weight.grad is not None
    assert encoder.content_key[1].weight.grad is not None
    assert encoder.future_query.grad is not None
    assert torch.isfinite(encoder.flow.delta_head[-1].weight.grad).all()


def test_future_dino_is_a_separate_no_grad_teacher_boundary() -> None:
    torch.manual_seed(11)
    config = _flow_config()
    encoder = FlowDINOEvidenceEncoder(config)
    assert tuple(inspect.signature(encoder.forward).parameters) == ("visual", "raw_visual")

    target = torch.randn(
        2,
        config.future_anchors + 1,
        config.visual_history_length,
        config.num_cameras,
        config.patches_per_camera,
        config.visual_token_dim,
        requires_grad=True,
    )
    current = _visual(config)
    first_window, first_stage = encoder.teacher_target(target, current)
    changed = target.detach().clone()
    changed[:, 0] += 2.0
    second_window, second_stage = encoder.teacher_target(changed, current)
    assert first_window.shape == (
        2,
        config.future_anchors * config.num_cameras * config.flow_jepa_grid_size**2,
        config.hidden_size,
    )
    assert first_stage.shape == (2, 1, config.hidden_size)
    assert first_window.dtype == torch.float32
    assert first_stage.dtype == torch.float32
    assert not first_window.requires_grad
    assert not first_stage.requires_grad
    assert not torch.allclose(first_window, second_window)
    torch.testing.assert_close(first_stage, second_stage)
    changed_stage = target.detach().clone()
    changed_stage[:, -1] += torch.randn_like(changed_stage[:, -1])
    _, third_stage = encoder.teacher_target(changed_stage, current)
    assert not torch.allclose(first_stage, third_stage)
    assert encoder.teacher_projection.weight.requires_grad is False


def test_stage_teacher_preserves_zero_delta_and_coarse_spatial_change() -> None:
    torch.manual_seed(13)
    config = _flow_config()
    encoder = FlowDINOEvidenceEncoder(config)
    current = _visual(config, batch=1)
    target = torch.zeros(
        1,
        config.future_anchors + 1,
        config.visual_history_length,
        config.num_cameras,
        config.patches_per_camera,
        config.visual_token_dim,
    )
    target[:, -1, -1] = current[:, -1]
    _, zero_stage = encoder.teacher_target(target, current)
    assert float(zero_stage.abs().max()) < 1e-6

    moved = target.clone()
    first = moved[:, -1, -1, :, 0].clone()
    moved[:, -1, -1, :, 0] = moved[:, -1, -1, :, -1]
    moved[:, -1, -1, :, -1] = first
    _, moved_stage = encoder.teacher_target(moved, current)
    assert float(moved_stage.norm()) > 1e-4


def test_flow_dino_cpu_bf16_autocast_is_finite() -> None:
    torch.manual_seed(19)
    config = _flow_config()
    encoder = FlowDINOEvidenceEncoder(config).eval()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        pack = encoder(_visual(config, batch=1))
    assert pack.selector_tokens.dtype == torch.bfloat16
    assert pack.value_tokens.dtype == torch.bfloat16
    assert pack.future_queries.dtype == torch.bfloat16
    assert torch.isfinite(pack.selector_tokens.float()).all()
    assert all(torch.isfinite(value.float()) for value in pack.losses.values())


def test_flow_dino_eval_disables_context_dropout_and_is_deterministic() -> None:
    torch.manual_seed(29)
    config = _flow_config()
    encoder = FlowDINOEvidenceEncoder(config).eval()
    visual = _visual(config, batch=1)
    with torch.no_grad():
        first = encoder(visual)
        second = encoder(visual)
    assert not first.context_dropout_mask.any()
    assert not second.context_dropout_mask.any()
    assert torch.equal(first.future_target_mask, second.future_target_mask)
    torch.testing.assert_close(first.selector_tokens, second.selector_tokens)
    torch.testing.assert_close(first.value_tokens, second.value_tokens)


def test_full_evidence_policy_backpropagates_through_one_flow_dino_path() -> None:
    base = _flow_config()
    config = V39PolicyConfig(
        **{
            **base.__dict__,
            "final_action_decoder": "evidence_latent_mmdit_action",
            "layer_contract_adapters": 1,
            "layer_contract_adapter_dim": 32,
            "latent_cvae_mmdit_depth": 2,
        }
    )
    config.validate()
    system = V39PolicySystem(config).train()
    batch = 1
    target_visual = torch.randn(
        batch,
        config.target_future_count,
        config.visual_history_length,
        config.num_cameras,
        config.patches_per_camera,
        config.visual_token_dim,
    )
    output = system.flow_training_forward(
        _visual(config, batch=batch),
        torch.randn(batch, config.visual_history_length, config.state_dim),
        torch.randn(batch, config.executed_history_length, config.action_dim),
        torch.randn(batch, config.state_dim),
        torch.randn(batch, config.action_horizon, config.action_dim),
        target_visual=target_visual,
        make_counterfactuals=False,
    )
    future_count = config.future_anchors * config.num_cameras * config.flow_jepa_grid_size**2
    assert output["pred_physical_velocity"].shape == (
        batch,
        config.action_horizon,
        config.physical_action_dim,
    )
    assert output["flow_jepa_future_pred"].shape == (
        batch,
        future_count,
        config.hidden_size,
    )
    assert output["flow_jepa_future_target"].shape == output["flow_jepa_future_pred"].shape
    assert output["flow_jepa_stage_pred"].shape == (batch, 1, config.hidden_size)
    assert output["flow_jepa_stage_target"].shape == output["flow_jepa_stage_pred"].shape
    assert output["flow_jepa_stage_target"].dtype == torch.float32
    assert not output["flow_jepa_future_target"].requires_grad
    assert "rollout_effect_target" not in output
    assert "future_latent_target" not in output

    objective = (
        output["pred_physical_velocity"].float().square().mean()
        + output["flow_jepa_future_pred"].float().square().mean()
        + output["flow_jepa_stage_pred"].float().square().mean()
        + sum(
            output[key]
            for key in (
                "flow_jepa_warp_loss",
                "flow_jepa_cycle_loss",
                "flow_jepa_smoothness_loss",
                "flow_jepa_uncertainty_nll",
                "flow_jepa_refinement_sequence_loss",
            )
        )
    )
    objective.backward()
    flow_module = system.planner.flow_dino_evidence
    assert flow_module is not None
    assert flow_module.flow.delta_head[-1].weight.grad is not None
    assert flow_module.content_value[1].weight.grad is not None
    assert flow_module.stage_query_token.grad is not None
    assert flow_module.stage_prediction[-1].weight.grad is not None
    assert flow_module.teacher_projection.weight.grad is None
    assert system.planner.blocks[0].stage_to_window_gate_logit.grad is not None
    assert system.planner.blocks[0].self_attn.in_proj_weight.grad is not None
    assert system.planner.visual_memory.proj[1].weight.grad is None
    assert system.planner.rollout_codec.init_proj[1].weight.grad is None


def test_full_flow_dino_evidence_policy_is_finite_under_cpu_bf16_autocast() -> None:
    base = _flow_config()
    config = V39PolicyConfig(
        **{
            **base.__dict__,
            "final_action_decoder": "evidence_latent_mmdit_action",
            "layer_contract_adapters": 1,
            "layer_contract_adapter_dim": 32,
            "latent_cvae_mmdit_depth": 2,
        }
    )
    system = V39PolicySystem(config).train()
    batch = 1
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = system.flow_training_forward(
            _visual(config, batch=batch),
            torch.randn(batch, config.visual_history_length, config.state_dim),
            torch.randn(batch, config.executed_history_length, config.action_dim),
            torch.randn(batch, config.state_dim),
            torch.randn(batch, config.action_horizon, config.action_dim),
            target_visual=torch.randn(
                batch,
                config.target_future_count,
                config.visual_history_length,
                config.num_cameras,
                config.patches_per_camera,
                config.visual_token_dim,
            ),
            make_counterfactuals=False,
        )
        objective = (
            output["pred_physical_velocity"].float().square().mean()
            + output["flow_jepa_future_pred"].float().square().mean()
            + output["flow_jepa_stage_pred"].float().square().mean()
            + output["flow_jepa_warp_loss"]
        )
    assert output["pred_physical_velocity"].dtype == torch.bfloat16
    assert output["flow_jepa_future_pred"].dtype == torch.bfloat16
    assert output["flow_jepa_stage_pred"].dtype == torch.bfloat16
    assert output["flow_jepa_stage_target"].dtype == torch.float32
    assert torch.isfinite(output["pred_physical_velocity"].float()).all()
    assert torch.isfinite(output["flow_jepa_future_pred"].float()).all()
    objective.backward()
    assert system.planner.flow_dino_evidence is not None
    assert system.planner.flow_dino_evidence.flow.delta_head[-1].weight.grad is not None


def test_directed_canvas_attention_preserves_context_action_future_order() -> None:
    config = _flow_config()
    block = TemporalDynamicsBoundDiTBlock(config).eval()
    slices = {
        "task": slice(0, 1),
        "state": slice(1, 2),
        "state_history": slice(2, 3),
        "executed": slice(3, 4),
        "proposal": slice(4, 5),
        "trajectory": slice(5, 7),
        "stage": slice(7, 8),
        "rollout": slice(8, 10),
        "registers": slice(10, 11),
    }
    mask = block._directed_attention_mask(11, slices, device=torch.device("cpu"))
    context_rows = torch.tensor([0, 1, 2, 3, 4, 10])
    assert mask[context_rows, 5:10].all()
    assert mask[5:7, 7:10].all()
    assert mask[7:8, 8:10].all()
    assert not mask[8:10].any()

    torch.manual_seed(23)
    canvas = torch.randn(2, 11, config.hidden_size)
    changed = canvas.clone()
    changed[:, 8:10] += 10.0
    visual = torch.randn(2, 5, config.hidden_size)
    condition = torch.randn(2, config.hidden_size)
    with torch.no_grad():
        output, _ = block(canvas, visual, condition, slices)
        changed_output, _ = block(changed, visual, condition, slices)
    torch.testing.assert_close(output[:, context_rows], changed_output[:, context_rows])
    torch.testing.assert_close(output[:, slices["trajectory"]], changed_output[:, slices["trajectory"]])
    torch.testing.assert_close(output[:, slices["stage"]], changed_output[:, slices["stage"]])

    changed_stage = canvas.clone()
    changed_stage[:, slices["stage"]] += 10.0 * torch.randn_like(
        changed_stage[:, slices["stage"]]
    )
    with torch.no_grad():
        stage_output, _ = block(changed_stage, visual, condition, slices)
    assert not torch.allclose(output[:, slices["rollout"]], stage_output[:, slices["rollout"]])

    legacy = TemporalDynamicsBoundDiTBlock(
        _flow_config(flow_jepa_enabled=0, future_grid_size=1)
    )
    assert legacy.directed_canvas_attention is False


def test_nonuniform_window_offsets_control_action_timeline_alignment() -> None:
    tokens = torch.tensor([[[1.0], [2.0], [3.0]]])
    aligned = _align_milestone_tokens_to_horizon(
        tokens, 24, boundaries=(4, 12, 24)
    )
    assert torch.equal(aligned[:, :4], torch.ones(1, 4, 1))
    assert torch.equal(aligned[:, 4:12], torch.full((1, 8, 1), 2.0))
    assert torch.equal(aligned[:, 12:], torch.full((1, 12, 1), 3.0))


def test_flow_dino_modulation_excludes_stage_and_window_feedback() -> None:
    config = _flow_config()
    system = V39PolicySystem(config)
    planner = system.planner
    visual_context = planner.encode_visual_context(_visual(config, batch=1))
    assert visual_context is not None
    noisy = torch.randn(1, config.action_horizon, config.physical_action_dim)
    canvas, slices = planner.seed(
        noisy_physical=noisy,
        state=torch.randn(1, config.state_dim),
        state_history=torch.randn(1, config.visual_history_length, config.state_dim),
        executed_history=torch.randn(1, config.executed_history_length, config.action_dim),
        proposal_tokens=torch.randn(1, config.action_horizon, config.hidden_size),
        proposal_keep=torch.ones(1),
        rollout_init=visual_context.future_queries,
        stage_init=visual_context.stage_query,
    )
    time_emb = planner.time(torch.rand(1))
    base, _, _ = planner._mod_embed(
        canvas, visual_context.selector_tokens, time_emb, slices
    )
    changed = canvas.clone()
    changed[:, slices["stage"]] += 50.0
    changed[:, slices["rollout"]] -= 50.0
    changed_mod, _, _ = planner._mod_embed(
        changed, visual_context.selector_tokens, time_emb, slices
    )
    torch.testing.assert_close(base, changed_mod)


def test_cached_targets_gather_sparse_window_and_stage_offsets() -> None:
    class _Base:
        camera_names = ("cam0",)

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            del index
            history_keys = torch.tensor([[100, 0], [101, 0], [102, 0]])
            target_keys = torch.zeros(12, 3, 2, dtype=torch.long)
            for future in range(12):
                target_keys[future, :, 0] = 200 + future
            return {
                "state": torch.zeros(3),
                "history_keys": history_keys,
                "history_obs_image": torch.rand(3, 1, 3, 32, 32),
                "target_history_keys": target_keys,
                "future_offsets": torch.arange(4, 49, 4),
            }

    class _Store:
        tokens_per_camera = 1
        token_dim = 1

        def load_batch(self, keys: torch.Tensor) -> torch.Tensor:
            return keys[:, 0].float().reshape(-1, 1, 1, 1)

    dataset = CachedTokenPolicyWindowDataset(
        _Base(),
        token_store=_Store(),
        future_anchors=3,
        future_indices=(0, 2, 5, 11),
    )
    sample = dataset[0]
    assert sample["history_obs_image"].shape == (3, 1, 3, 32, 32)
    assert sample["history_dinov2_tokens"].shape == (3, 1, 1, 1)
    assert torch.equal(sample["target_future_offsets"], torch.tensor([4, 12, 24, 48]))
    assert torch.equal(
        sample["target_future_dinov2_tokens"].flatten(),
        torch.tensor([200.0, 202.0, 205.0, 211.0]),
    )


def test_action_history_preserves_recent_tokens_and_compresses_old_prefix() -> None:
    config = _flow_config(
        action_history_enabled=1,
        executed_action_offsets=(-8, -1),
        action_history_recent_tokens=1,
        action_history_summary_tokens=1,
    )
    proposal = RejectableHistoryProposal(config)
    history = torch.randn(2, config.executed_history_length, config.action_dim)
    memory = proposal.encode_history(history)
    assert memory.shape == (2, 2, config.hidden_size)
    assert proposal.history_token_count == 2
    memory.square().mean().backward()
    assert proposal.history_summary_query is not None
    assert proposal.history_summary_query.grad is not None
    assert proposal.history_proj.weight.grad is not None


def test_goal_and_action_history_condition_the_single_policy_path() -> None:
    base = _flow_config(
        action_history_enabled=1,
        executed_action_offsets=(-8, -1),
        action_history_recent_tokens=1,
        action_history_summary_tokens=1,
        goal_conditioning_enabled=1,
        goal_token_count=2,
        goal_language_dim=16,
        goal_language_max_tokens=5,
        goal_resampler_depth=1,
    )
    config = V39PolicyConfig(
        **{
            **base.__dict__,
            "final_action_decoder": "evidence_latent_mmdit_action",
            "layer_contract_adapters": 1,
            "layer_contract_adapter_dim": 32,
            "latent_cvae_mmdit_depth": 1,
        }
    )
    system = V39PolicySystem(config).eval()
    first_language = torch.randn(1, 4, config.goal_language_dim)
    language_mask = torch.tensor([[True, True, True, False]])
    system.set_default_goal_language(first_language, language_mask)
    batch = 1
    visual = _visual(config, batch=batch)
    state_history = torch.randn(batch, config.visual_history_length, config.state_dim)
    executed = torch.randn(batch, config.executed_history_length, config.action_dim)
    state = torch.randn(batch, config.state_dim)
    action = torch.randn(batch, config.action_horizon, config.action_dim)

    torch.manual_seed(101)
    first = system.flow_training_forward(
        visual,
        state_history,
        executed,
        state,
        action,
        make_counterfactuals=False,
    )
    system.set_default_goal_language(first_language.roll(1, dims=-1), language_mask)
    torch.manual_seed(101)
    second = system.flow_training_forward(
        visual,
        state_history,
        executed,
        state,
        action,
        make_counterfactuals=False,
    )
    assert not torch.allclose(
        first["pred_physical_velocity"], second["pred_physical_velocity"]
    )
    assert float(first["flow_jepa_goal_token_count"]) == config.goal_token_count
    assert (
        float(first["flow_jepa_action_memory_token_count"])
        == config.action_history_token_count
    )

    system.train()
    objective = second["pred_physical_velocity"].float().square().mean()
    objective.backward()
    assert system.planner.goal_resampler is not None
    assert system.planner.goal_resampler.query.grad is not None
    assert system.proposal.history_proj.weight.grad is not None
    assert not system.default_goal_language_tokens.requires_grad


def test_precomputed_t5_condition_avoids_a_resident_text_encoder(tmp_path) -> None:
    cached = tmp_path / "goal_tokens.pt"
    torch.save(torch.randn(6, 12), cached)
    tokens, mask, metadata = load_precomputed_t5_condition(
        condition_path=cached,
        max_tokens=4,
    )
    assert tokens.shape == (1, 4, 12)
    assert mask.shape == (1, 4)
    assert mask.all()
    assert metadata["source"] == "precomputed_t5_condition"


def test_repository_rdt_t5_tensor_format_is_accepted() -> None:
    condition = Path(__file__).parents[1] / "clearvla" / "assets" / "rdt_empty_lang_embed.pt"
    tokens, mask, metadata = load_precomputed_t5_condition(
        condition_path=condition,
        max_tokens=32,
    )
    assert tokens.shape == (1, 1, 4096)
    assert mask.shape == (1, 1)
    assert mask.all()
    assert metadata["original_shape"] == [1, 4096]


def test_goal_and_action_history_parameters_have_optimizer_owners() -> None:
    config = _flow_config(
        action_history_enabled=1,
        executed_action_offsets=(-8, -1),
        action_history_recent_tokens=1,
        action_history_summary_tokens=1,
        goal_conditioning_enabled=1,
        goal_token_count=2,
        goal_language_dim=16,
        goal_language_max_tokens=5,
        goal_resampler_depth=1,
        layer_contract_adapters=1,
        layer_contract_adapter_dim=32,
        final_action_decoder="evidence_latent_mmdit_action",
        latent_cvae_mmdit_depth=1,
    )
    system = V39PolicySystem(config)
    groups = _optimizer_groups(
        system,
        V39PolicyTrainerConfig(training_stage="policy", contract_mode="layer_adapter"),
    )
    parameter_ids = [id(parameter) for group in groups for parameter in group["params"]]
    assert len(parameter_ids) == len(set(parameter_ids))
    covered = set(parameter_ids)
    assert system.planner.goal_resampler is not None
    assert all(
        id(parameter) in covered
        for parameter in system.planner.goal_resampler.parameters()
        if parameter.requires_grad
    )
    assert all(
        id(parameter) in covered
        for name, parameter in system.proposal.named_parameters()
        if parameter.requires_grad and name.startswith("history_")
    )


def test_single_stage_role_optimizer_uses_one_base_lr_for_all_top_blocks() -> None:
    config = _raw_role_config(
        final_action_decoder="evidence_latent_mmdit_action",
        layer_contract_adapters=1,
        layer_contract_adapter_dim=32,
        latent_cvae_mmdit_depth=1,
        flow_jepa_zero_flow_guard=1,
        flow_jepa_complementary_raw_detail=1,
        flow_jepa_strict_role_visual_path=1,
        flow_jepa_source_aligned_raw_fusion=1,
        flow_jepa_policy_workspace_fixed_fusion=1,
    )
    system = V39PolicySystem(config)
    trainer = V39PolicyTrainerConfig(
        training_stage="policy",
        contract_mode="layer_adapter",
        lr=8e-5,
        single_stage_role_lr=1,
    )
    groups = _optimizer_groups(system, trainer)
    parameter_ids = [id(parameter) for group in groups for parameter in group["params"]]
    assert len(parameter_ids) == len(set(parameter_ids))
    covered = set(parameter_ids)
    trainable = {
        id(parameter)
        for parameter in system.parameters()
        if parameter.requires_grad
    }
    # V101 adds source-aligned raw fusion and a complete top-policy workspace
    # path.  A positive backward gradient is insufficient if either branch is
    # absent from AdamW, so require exact one-owner coverage of the full
    # single-stage graph.
    assert covered == trainable
    top_groups = [
        group for group in groups if str(group["name"]).startswith("dit_block_")
    ]
    assert len(top_groups) == config.depth
    assert all(float(group["lr"]) == trainer.lr for group in top_groups)
    by_name = {str(group["name"]): float(group["lr"]) for group in groups}
    assert by_name["single_stage_shared_input"] == trainer.lr
    assert "midcut_contract_heads_single_stage" not in by_name
    assert by_name["layer_contract_adapters_single_stage"] == trainer.lr


def test_v102_late_detail_reader_has_one_base_lr_optimizer_owner() -> None:
    config = _v102_config(
        final_action_decoder="evidence_latent_mmdit_action",
        layer_contract_adapters=1,
        layer_contract_adapter_dim=32,
        latent_cvae_mmdit_depth=1,
    )
    system = V39PolicySystem(config)
    trainer = V39PolicyTrainerConfig(
        training_stage="policy",
        contract_mode="layer_adapter",
        lr=8e-5,
        single_stage_role_lr=1,
    )
    groups = _optimizer_groups(system, trainer)
    assert system.planner.late_raw_detail_reader is not None
    owner = {
        id(parameter): (str(group["name"]), float(group["lr"]))
        for group in groups
        for parameter in group["params"]
    }
    late_parameters = [
        parameter
        for parameter in system.planner.late_raw_detail_reader.parameters()
        if parameter.requires_grad
    ]
    assert late_parameters
    assert all(id(parameter) in owner for parameter in late_parameters)
    assert all(owner[id(parameter)] == ("final_policy_heads", trainer.lr) for parameter in late_parameters)


def test_v102_freezes_dormant_legacy_routes_and_optimizer_covers_exact_graph() -> None:
    config = _v102_config(
        final_action_decoder="evidence_latent_mmdit_action",
        layer_contract_adapters=1,
        layer_contract_adapter_dim=32,
        latent_cvae_mmdit_depth=1,
    )
    system = V39PolicySystem(config)
    encoder = system.planner.flow_dino_evidence
    assert encoder is not None and encoder.raw_address_reader is not None
    dormant = [
        *encoder.raw_address_reader.key_proj.parameters(),
        *encoder.raw_address_reader.value_proj.parameters(),
        encoder.raw_address_reader.flow_prior_strength,
        *encoder.content_key.parameters(),
        *encoder.content_value.parameters(),
        *encoder.warp_key.parameters(),
        *encoder.warp_value.parameters(),
    ]
    for role, block in zip(system.planner.block_roles, system.planner.blocks):
        if role == "policy":
            dormant.extend(block.mem_norm.parameters())
            dormant.extend(block.cross.parameters())
    assert dormant and all(not parameter.requires_grad for parameter in dormant)
    groups = _optimizer_groups(
        system,
        V39PolicyTrainerConfig(
            training_stage="policy",
            contract_mode="layer_adapter",
            single_stage_role_lr=1,
        ),
    )
    covered = [id(parameter) for group in groups for parameter in group["params"]]
    trainable = [
        id(parameter) for parameter in system.parameters() if parameter.requires_grad
    ]
    assert len(covered) == len(set(covered))
    assert set(covered) == set(trainable)


def test_v102_fallback_optimizer_owns_late_reader() -> None:
    config = _v102_config(
        final_action_decoder="legacy",
        layer_contract_adapters=0,
    )
    system = V39PolicySystem(config)
    groups = _optimizer_groups(
        system,
        V39PolicyTrainerConfig(
            training_stage="policy",
            contract_mode="midcut",
        ),
    )
    covered = {
        id(parameter) for group in groups for parameter in group["params"]
    }
    reader = system.planner.late_raw_detail_reader
    assert reader is not None
    assert all(
        id(parameter) in covered
        for parameter in reader.parameters()
        if parameter.requires_grad
    )


def test_raw_and_late_detail_flags_require_an_enabled_flow_graph() -> None:
    valid = _v102_config()
    invalid_values = asdict(valid)
    invalid_values["flow_jepa_enabled"] = 0
    invalid = V39PolicyConfig(**invalid_values)
    try:
        invalid.validate()
    except ValueError as error:
        assert "raw-image grounding requires Flow-DINO JEPA" in str(error)
    else:
        raise AssertionError("late raw detail accepted a disabled Flow-DINO graph")


def test_v102_resume_contract_rejects_shape_compatible_semantic_changes() -> None:
    current = _v102_config()
    saved = asdict(current)
    _validate_v102_resume_contract(saved, current)
    for field in (
        "flow_jepa_world_anchor_write_only",
        "flow_jepa_late_policy_detail",
        "flow_jepa_policy_workspace_horizon_pool",
    ):
        mismatched = dict(saved)
        mismatched[field] = 0
        try:
            _validate_v102_resume_contract(mismatched, current)
        except ValueError as error:
            assert field in str(error)
        else:
            raise AssertionError(f"resume accepted changed {field}")
    mismatched_scale = dict(saved)
    mismatched_scale["flow_jepa_late_policy_detail_scale"] = 0.5
    try:
        _validate_v102_resume_contract(mismatched_scale, current)
    except ValueError as error:
        assert "late_policy_detail_scale" in str(error)
    else:
        raise AssertionError("resume accepted a changed late-detail scale")

    legacy_current = _raw_role_config()
    legacy_saved = asdict(legacy_current)
    for field in (
        "flow_jepa_world_anchor_write_only",
        "flow_jepa_late_policy_detail",
        "flow_jepa_late_policy_detail_scale",
        "flow_jepa_policy_workspace_horizon_pool",
    ):
        legacy_saved.pop(field, None)
    _validate_v102_resume_contract(legacy_saved, legacy_current)


def test_new_stage1_owns_representation_but_not_final_action_decoder() -> None:
    config = _flow_config(
        action_history_enabled=1,
        executed_action_offsets=(-8, -1),
        action_history_recent_tokens=1,
        action_history_summary_tokens=1,
        goal_conditioning_enabled=1,
        goal_token_count=2,
        goal_language_dim=16,
        goal_language_max_tokens=5,
        goal_resampler_depth=1,
        layer_contract_adapters=1,
        layer_contract_adapter_dim=32,
        final_action_decoder="evidence_latent_mmdit_action",
        latent_cvae_mmdit_depth=1,
    )
    system = V39PolicySystem(config)
    groups = _optimizer_groups(
        system,
        V39PolicyTrainerConfig(training_stage="stage1", contract_mode="layer_adapter"),
    )
    covered = {id(parameter) for group in groups for parameter in group["params"]}
    group_names = {str(group["name"]) for group in groups}
    assert system.planner.flow_dino_evidence is not None
    assert system.planner.goal_resampler is not None
    assert system.planner.evidence_latent_mmdit_action_decoder is not None
    assert all(
        id(parameter) in covered
        for parameter in system.planner.flow_dino_evidence.parameters()
        if parameter.requires_grad
    )
    assert all(
        id(parameter) in covered
        for parameter in system.planner.goal_resampler.parameters()
        if parameter.requires_grad
    )
    assert all(
        id(parameter) not in covered
        for parameter in system.planner.evidence_latent_mmdit_action_decoder.parameters()
        if parameter.requires_grad
    )
    assert all(
        id(parameter) in covered
        for parameter in system.planner.final_norm.parameters()
        if parameter.requires_grad
    )
    assert all(
        id(parameter) not in covered
        for parameter in system.planner.layer_contract_heads.parameters()
        if parameter.requires_grad
    )
    assert all("contract" not in name for name in group_names)


def test_new_stage1_loss_is_only_the_direct_flow_jepa_objective() -> None:
    future_pred = torch.randn(2, 4, 8, requires_grad=True)
    stage_pred = torch.randn(2, 1, 8, requires_grad=True)
    auxiliary = {
        "flow_jepa_warp_loss": torch.tensor(0.4, requires_grad=True),
        "flow_jepa_cycle_loss": torch.tensor(0.3, requires_grad=True),
        "flow_jepa_smoothness_loss": torch.tensor(0.2, requires_grad=True),
        "flow_jepa_uncertainty_nll": torch.tensor(-0.1, requires_grad=True),
        "flow_jepa_refinement_sequence_loss": torch.tensor(0.5, requires_grad=True),
    }
    output = {
        "flow_jepa_future_pred": future_pred,
        "flow_jepa_future_target": torch.randn_like(future_pred),
        "flow_jepa_current_target": torch.zeros_like(future_pred),
        "flow_jepa_future_target_mask": torch.ones(2, 4, dtype=torch.bool),
        "flow_jepa_stage_pred": stage_pred,
        "flow_jepa_stage_target": torch.randn_like(stage_pred),
        **auxiliary,
    }
    trainer = V39PolicyTrainerConfig(
        flow_jepa_future_loss_weight=0.10,
        flow_jepa_future_change_loss_weight=0.02,
        flow_jepa_stage_loss_weight=0.02,
        flow_jepa_warp_loss_weight=0.03,
        flow_jepa_cycle_loss_weight=0.01,
        flow_jepa_smoothness_loss_weight=0.002,
        flow_jepa_uncertainty_nll_weight=0.005,
        flow_jepa_refinement_sequence_loss_weight=0.02,
    )
    losses = flow_jepa_stage1_losses(output, trainer)
    contributions = [
        value
        for key, value in losses.items()
        if key.startswith("loss_contrib_")
    ]
    assert torch.allclose(losses["loss"].detach(), torch.stack(contributions).sum())
    assert torch.equal(losses["loss_group_representation"], losses["loss"].detach())
    assert "physical_flow" not in losses
    losses["loss"].backward()
    assert future_pred.grad is not None
    assert stage_pred.grad is not None
    assert all(value.grad is not None for value in auxiliary.values())
    with torch.no_grad():
        validation_losses = flow_jepa_stage1_losses(output, trainer)
    assert torch.isfinite(validation_losses["loss"])
    assert not validation_losses["loss"].requires_grad


def test_new_stage1_backpropagates_through_goal_action_history_and_flow_dino() -> None:
    config = _flow_config(
        action_history_enabled=1,
        executed_action_offsets=(-8, -1),
        action_history_recent_tokens=1,
        action_history_summary_tokens=1,
        goal_conditioning_enabled=1,
        goal_token_count=2,
        goal_language_dim=16,
        goal_language_max_tokens=5,
        goal_resampler_depth=1,
        layer_contract_adapters=1,
        layer_contract_adapter_dim=32,
        final_action_decoder="evidence_latent_mmdit_action",
        latent_cvae_mmdit_depth=1,
    )
    system = V39PolicySystem(config).train()
    system.set_default_goal_language(
        torch.randn(1, 3, config.goal_language_dim),
        torch.ones(1, 3, dtype=torch.bool),
    )
    batch = 2
    decoder_calls = 0

    def _count_decoder_calls(_module, _inputs, _output) -> None:
        nonlocal decoder_calls
        decoder_calls += 1

    assert system.planner.evidence_latent_mmdit_action_decoder is not None
    handle = system.planner.evidence_latent_mmdit_action_decoder.register_forward_hook(
        _count_decoder_calls
    )
    output = system.flow_jepa_stage1_forward(
        _visual(config, batch=batch),
        torch.randn(batch, config.visual_history_length, config.state_dim),
        torch.randn(batch, config.executed_history_length, config.action_dim),
        torch.randn(batch, config.state_dim),
        torch.randn(
            batch,
            config.target_future_count,
            config.visual_history_length,
            config.num_cameras,
            config.patches_per_camera,
            config.visual_token_dim,
        ),
    )
    handle.remove()
    assert decoder_calls == 0
    assert output["layer_contracts"] == []
    assert float(output["flow_jepa_stage1_target_action_conditioned"]) == 0.0
    trainer = V39PolicyTrainerConfig(
        training_stage="stage1",
        flow_jepa_future_loss_weight=0.10,
        flow_jepa_stage_loss_weight=0.02,
        flow_jepa_warp_loss_weight=0.03,
        flow_jepa_cycle_loss_weight=0.01,
        flow_jepa_smoothness_loss_weight=0.002,
        flow_jepa_uncertainty_nll_weight=0.005,
        flow_jepa_refinement_sequence_loss_weight=0.02,
    )
    losses = flow_jepa_stage1_losses(
        output,
        trainer,
    )
    losses["loss"].backward()
    assert system.planner.flow_dino_evidence is not None
    assert system.planner.goal_resampler is not None
    assert any(
        parameter.grad is not None
        for parameter in system.planner.flow_dino_evidence.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in system.planner.goal_resampler.parameters()
    )
    assert any(
        parameter.grad is not None
        for name, parameter in system.proposal.named_parameters()
        if name.startswith("history_")
    )
    assert all(
        parameter.grad is None
        for parameter in system.planner.evidence_latent_mmdit_action_decoder.parameters()
    )
    covered = {
        id(parameter)
        for group in _optimizer_groups(system, trainer)
        for parameter in group["params"]
    }
    active_gradient_parameters = {
        id(parameter)
        for parameter in system.parameters()
        if parameter.grad is not None and bool(torch.count_nonzero(parameter.grad.detach()))
    }
    assert active_gradient_parameters
    assert active_gradient_parameters <= covered


def test_v95_policy_rejects_an_old_or_mismatched_stage1_checkpoint() -> None:
    config = _flow_config(
        action_history_enabled=1,
        executed_action_offsets=(-8, -1),
        action_history_recent_tokens=1,
        action_history_summary_tokens=1,
        goal_conditioning_enabled=1,
        goal_token_count=2,
        goal_language_dim=16,
    )
    goal = {"embedding_sha256": "goal-hash"}
    payload = {
        "stage1_contract": {
            "kind": "flow_dino_jepa_representation_v1",
            "target_action_conditioned": False,
            "final_action_decoder_executed": False,
            "layer_contracts_executed": False,
        },
        "trainer_config": {"training_stage": "stage1"},
        "policy_config": asdict(config),
        "context": {
            "goal_language": goal,
            "source_fingerprint": _source_fingerprint(),
        },
    }
    _validate_flow_jepa_stage1_checkpoint(
        payload,
        policy_config=config,
        goal_language_metadata=goal,
    )
    legacy_compatible_payload = {
        **payload,
        "policy_config": dict(payload["policy_config"]),
    }
    legacy_compatible_payload["policy_config"].pop(
        "flow_jepa_late_policy_detail_scale"
    )
    _validate_flow_jepa_stage1_checkpoint(
        legacy_compatible_payload,
        policy_config=config,
        goal_language_metadata=goal,
    )
    old_payload = dict(payload)
    old_payload.pop("stage1_contract")
    try:
        _validate_flow_jepa_stage1_checkpoint(
            old_payload,
            policy_config=config,
            goal_language_metadata=goal,
        )
    except ValueError as error:
        assert "old best_contract.pt" in str(error)
    else:
        raise AssertionError("old Stage1 checkpoint was accepted by V95 policy")


def test_future_teacher_can_run_without_extra_counterfactual_policy_graphs() -> None:
    config = _flow_config()
    system = V39PolicySystem(config).train()
    batch = 2
    inputs = (
        _visual(config, batch=batch),
        torch.randn(batch, config.visual_history_length, config.state_dim),
        torch.randn(batch, config.executed_history_length, config.action_dim),
        torch.randn(batch, config.state_dim),
        torch.randn(batch, config.action_horizon, config.action_dim),
    )
    target_visual = torch.randn(
        batch,
        config.target_future_count,
        config.visual_history_length,
        config.num_cameras,
        config.patches_per_camera,
        config.visual_token_dim,
    )
    with patch.object(system, "_policy_forward", wraps=system._policy_forward) as policy_forward:
        output = system.flow_training_forward(
            *inputs,
            target_visual=target_visual,
            make_counterfactuals=False,
        )
    assert policy_forward.call_count == 1
    assert "flow_jepa_future_target" in output
    assert "rollout_effect_pred_hold_action" not in output

    with patch.object(system, "_policy_forward", wraps=system._policy_forward) as policy_forward:
        counterfactual = system.flow_training_forward(
            *inputs,
            target_visual=target_visual,
            make_counterfactuals=True,
        )
    assert policy_forward.call_count == 3
    assert "rollout_effect_pred_hold_action" in counterfactual


def test_v107_contract_requires_every_top_path_repair() -> None:
    config = _complete_v107_config()
    trainer = _complete_v107_trainer()
    _validate_complete_v107_model_contract(config, trainer)
    assert _validate_required_model_contract("v107", config, trainer) == "v107"
    for field in (
        "flow_jepa_policy_multi_glimpse_address",
        "flow_jepa_horizon_cell_fine_address",
        "flow_jepa_interval_stage_typed_value",
        "role_residual_contract_after_gate",
    ):
        with pytest.raises(ValueError, match=field):
            _validate_complete_v107_model_contract(
                replace(config, **{field: 0}), trainer
            )


def test_v107_policy_soft_lattice_uses_real_independent_glimpses() -> None:
    torch.manual_seed(177)
    config = _complete_v107_config()
    reader = LateRawDetailPolicyReader(config).train()
    assert isinstance(reader.lattice_value_out, torch.nn.ModuleList)
    assert reader.lattice_query_proj is not None
    assert int(reader.lattice_query_proj.out_features) == (
        int(config.flow_jepa_raw_reader_heads)
        * int(config.flow_jepa_address_route_dim)
    )
    batch = 2
    cameras = int(config.num_cameras)
    grid = int(config.future_grid_size)
    slots = int(config.flow_jepa_address_slots)
    route = int(config.flow_jepa_address_route_dim)
    raw_dim = int(config.flow_jepa_raw_base_channels) + int(
        config.flow_jepa_raw_base_channels
    ) // 2
    candidates = 5
    fine_values = torch.randn(
        batch, cameras, grid, grid, slots, candidates, raw_dim,
        requires_grad=True,
    )
    bank = SoftAddressLatticeBank(
        coarse_keys=torch.randn(batch, cameras, grid, grid, slots, route),
        fine_keys=torch.randn(
            batch, cameras, grid, grid, slots, candidates, route
        ),
        fine_values=fine_values,
        fine_valid=torch.ones(
            batch, cameras, grid, grid, slots, candidates, dtype=torch.bool
        ),
        coarse_centers=torch.zeros(batch, cameras, grid, grid, slots, 2),
        coarse_variance=torch.ones(batch, cameras, grid, grid, slots, 2),
        fine_radius=torch.ones(batch, cameras, grid, grid, slots),
    )
    detail_tokens = cameras * grid * grid
    dummy = torch.zeros(batch, detail_tokens, config.hidden_size)
    detail = LateRawDetailEvidence(
        selector_tokens=dummy, value_tokens=dummy, address_bank=bank
    )
    trajectory = torch.randn(
        batch,
        config.action_horizon * config.action_basis_tokens,
        config.hidden_size,
    )
    rollout = torch.randn(
        batch,
        config.future_anchors * cameras * grid * grid,
        config.hidden_size,
    )
    phase_context = torch.randn(batch, config.hidden_size)
    condition_context = torch.randn(batch, config.hidden_size)
    updated, metrics = reader(
        trajectory,
        rollout,
        detail,
        phase_context=phase_context,
        condition_query_context=condition_context,
    )
    assert float(metrics["flow_jepa_policy_multi_glimpse_address"]) == 1.0
    assert int(metrics["flow_jepa_address_policy_glimpse_count"]) == int(
        config.flow_jepa_raw_reader_heads
    )
    assert float(
        metrics["flow_jepa_address_policy_glimpse_route_variation"]
    ) > 0.0
    (updated - trajectory).float().square().mean().backward()
    assert fine_values.grad is not None
    assert float(fine_values.grad.abs().sum()) > 0.0
    assert reader.lattice_query_proj.weight.grad is not None
    gradient = reader.lattice_query_proj.weight.grad.reshape(
        config.flow_jepa_raw_reader_heads,
        config.flow_jepa_address_route_dim,
        -1,
    )
    assert bool((gradient.abs().sum(dim=(1, 2)) > 0.0).all())
    zero_detail = replace(
        detail,
        address_bank=replace(bank, fine_values=torch.zeros_like(fine_values)),
    )
    zero_updated, _ = reader(
        trajectory,
        rollout,
        zero_detail,
        phase_context=phase_context,
        condition_query_context=condition_context,
    )
    torch.testing.assert_close(zero_updated, trajectory, rtol=0.0, atol=0.0)


def test_v107_horizon_fine_address_retains_target_cell_identity() -> None:
    torch.manual_seed(178)
    config = _complete_v107_config(
        flow_jepa_grid_size=2,
        future_grid_size=2,
        flow_jepa_address_slots=1,
        flow_jepa_address_route_dim=4,
    )
    raw_dim = 3
    reader = _HorizonSoftAddressJEPA(config, raw_dim=raw_dim)
    batch = 1
    cells = 4
    candidates = 2
    query = torch.zeros(
        batch,
        config.future_anchors,
        config.num_cameras,
        cells,
        config.flow_jepa_address_route_dim,
    )
    query[..., 0, 0] = 4.0
    query[..., 1, 0] = -4.0
    fine_key = torch.zeros(
        batch,
        config.num_cameras,
        cells,
        1,
        candidates,
        config.flow_jepa_address_route_dim,
    )
    fine_key[..., 0, 0] = 1.0
    fine_key[..., 1, 0] = -1.0
    fine_value = torch.zeros(
        batch,
        config.num_cameras,
        cells,
        1,
        candidates,
        raw_dim,
    )
    fine_value[..., 0, 0] = 1.0
    fine_value[..., 1, 0] = -1.0
    address, _, _, _, _, _, _ = reader._read_cell_specific_fine_address(
        query,
        torch.zeros(
            batch,
            config.num_cameras,
            cells,
            1,
            config.flow_jepa_address_route_dim,
        ),
        fine_key,
        fine_value,
        torch.ones(
            batch,
            config.num_cameras,
            cells,
            1,
            candidates,
            dtype=torch.bool,
        ),
    )
    assert bool((address[..., 0, 0] > 0.0).all())
    assert bool((address[..., 1, 0] < 0.0).all())
    bank = SoftAddressLatticeBank(
        coarse_keys=torch.zeros(
            batch,
            config.num_cameras,
            2,
            2,
            1,
            config.flow_jepa_address_route_dim,
        ),
        fine_keys=fine_key.reshape(
            batch,
            config.num_cameras,
            2,
            2,
            1,
            candidates,
            config.flow_jepa_address_route_dim,
        ),
        fine_values=torch.zeros_like(fine_value).reshape(
            batch,
            config.num_cameras,
            2,
            2,
            1,
            candidates,
            raw_dim,
        ),
        fine_valid=torch.ones(
            batch,
            config.num_cameras,
            2,
            2,
            1,
            candidates,
            dtype=torch.bool,
        ),
        coarse_centers=torch.zeros(
            batch, config.num_cameras, 2, 2, 1, 2
        ),
        coarse_variance=torch.ones(
            batch, config.num_cameras, 2, 2, 1, 2
        ),
        fine_radius=torch.ones(batch, config.num_cameras, 2, 2, 1),
    )
    future = torch.randn(
        batch,
        config.future_anchors * config.num_cameras * cells,
        config.hidden_size,
    )
    refined, _, metrics = reader(future, bank)
    torch.testing.assert_close(refined, future, rtol=0.0, atol=0.0)
    assert float(metrics["flow_jepa_horizon_cell_fine_address"]) == 1.0


def test_v107_post_gate_contract_bounds_actual_writes_and_types_interval() -> None:
    torch.manual_seed(179)
    config = _complete_v107_config(dropout=0.0)
    system = V39PolicySystem(config)
    planner = system.planner
    expected_sources = (
        config.flow_jepa_world_blocks + 2
    ) * config.num_cameras * (1 + planner.world_to_policy_far_anchor_count)
    assert planner.world_to_policy_attnres is not None
    assert int(planner.world_to_policy_attnres.max_sources) == expected_sources

    block = TemporalDynamicsBoundDiTBlock(config, role="world").eval()
    lengths = {
        "task": config.goal_token_count,
        "state": 1,
        "state_history": config.visual_history_length,
        "executed": config.action_history_token_count,
        "proposal": config.action_horizon,
        "trajectory": config.action_horizon * config.action_basis_tokens,
        "stage": 0,
        "rollout": config.future_token_count,
        "registers": config.canvas_registers,
    }
    slices: dict[str, slice] = {}
    cursor = 0
    for name, length in lengths.items():
        slices[name] = slice(cursor, cursor + int(length))
        cursor += int(length)
    canvas = torch.randn(2, cursor, config.hidden_size)
    visual = torch.randn(2, 7, config.hidden_size)
    _, metrics = block(
        canvas,
        visual,
        torch.randn(2, config.hidden_size),
        slices,
        visual_value_memory=visual,
    )
    assert float(metrics["residual_contract_after_gate"]) == 1.0
    for key, value in metrics.items():
        if key.endswith("_written_rms"):
            assert float(value) <= config.role_residual_max_update_rms + 1e-5


def test_v107_action_loss_reaches_multi_glimpse_and_typed_interval_path() -> None:
    torch.manual_seed(181)
    config = _complete_v107_config(flow_jepa_raw_activation_checkpoint=0)
    system = V39PolicySystem(config).train()
    batch = 1
    output = system.flow_training_forward(
        _visual(config, batch=batch),
        torch.randn(batch, config.visual_history_length, config.state_dim),
        torch.randn(batch, config.executed_history_length, config.action_dim),
        torch.randn(batch, config.state_dim),
        torch.randn(batch, config.action_horizon, config.action_dim),
        target_visual=torch.randn(
            batch,
            len(config.flow_jepa_effective_interval_support_offsets),
            config.visual_history_length,
            config.num_cameras,
            config.patches_per_camera,
            config.visual_token_dim,
        ),
        raw_visual=_raw_visual(config, batch=batch),
        goal_language_tokens=torch.randn(
            batch, 3, config.goal_language_dim
        ),
        goal_language_mask=torch.ones(batch, 3, dtype=torch.bool),
        make_counterfactuals=False,
    )
    for key in (
        "flow_jepa_policy_multi_glimpse_address",
        "flow_jepa_horizon_cell_fine_address",
        "flow_jepa_interval_stage_typed_value",
        "role_residual_contract_after_gate",
        "attnres_world_to_policy_source_mass_interval_stage_camera0",
    ):
        assert key in output
        assert torch.isfinite(output[key]).all()
    output["pred_physical_velocity"].float().square().mean().backward()
    interval = system.planner.flow_dino_evidence
    assert interval is not None and interval.interval_stage_organizer is not None
    late_reader = system.planner.late_raw_detail_reader
    assert late_reader is not None and late_reader.lattice_query_proj is not None
    bridge = system.planner.world_to_policy_attnres
    assert bridge is not None
    for parameter in (
        late_reader.lattice_query_proj.weight,
        interval.interval_stage_organizer.delta_out.weight,
        bridge.query_proj.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert float(parameter.grad.abs().sum()) > 0.0


def test_v108_contract_requires_online_horizon_address() -> None:
    config = _complete_v108_config()
    trainer = _complete_v108_trainer()
    _validate_complete_v108_model_contract(config, trainer)
    assert _validate_required_model_contract("v108", config, trainer) == "v108"
    with pytest.raises(ValueError, match="flow_jepa_online_horizon_address"):
        _validate_complete_v108_model_contract(
            replace(config, flow_jepa_online_horizon_address=0),
            trainer,
        )


def test_v108_online_address_is_exact_identity_for_zero_observation_values() -> None:
    torch.manual_seed(181)
    config = _complete_v108_config(
        flow_jepa_grid_size=2,
        future_grid_size=2,
        flow_jepa_address_slots=1,
        flow_jepa_address_route_dim=4,
    )
    encoder = FlowDINOEvidenceEncoder(config).eval()
    reader = encoder.horizon_address_jepa
    assert reader is not None
    batch = 1
    candidates = 2
    bank = SoftAddressLatticeBank(
        coarse_keys=torch.randn(
            batch,
            config.num_cameras,
            2,
            2,
            1,
            config.flow_jepa_address_route_dim,
        ),
        fine_keys=torch.randn(
            batch,
            config.num_cameras,
            2,
            2,
            1,
            candidates,
            config.flow_jepa_address_route_dim,
        ),
        fine_values=torch.zeros(
            batch,
            config.num_cameras,
            2,
            2,
            1,
            candidates,
            reader.raw_dim,
        ),
        fine_valid=torch.ones(
            batch,
            config.num_cameras,
            2,
            2,
            1,
            candidates,
            dtype=torch.bool,
        ),
        coarse_centers=torch.zeros(
            batch, config.num_cameras, 2, 2, 1, 2
        ),
        coarse_variance=torch.ones(
            batch, config.num_cameras, 2, 2, 1, 2
        ),
        fine_radius=torch.ones(batch, config.num_cameras, 2, 2, 1),
    )
    rollout = torch.randn(
        batch,
        config.future_anchors * config.num_cameras * 2 * 2,
        config.hidden_size,
    )
    refined, metrics = encoder.organize_horizon_address(rollout, bank)
    torch.testing.assert_close(refined, rollout, rtol=0.0, atol=0.0)
    assert float(metrics["flow_jepa_online_horizon_address"]) == 1.0


def test_v108_action_only_loss_reaches_online_address_owners() -> None:
    torch.manual_seed(182)
    config = _complete_v108_config(
        dropout=0.0,
        flow_jepa_raw_activation_checkpoint=0,
    )
    system = V39PolicySystem(config).train()
    batch = 1
    output = system.flow_training_forward(
        _visual(config, batch=batch),
        torch.randn(batch, config.visual_history_length, config.state_dim),
        torch.randn(batch, config.executed_history_length, config.action_dim),
        torch.randn(batch, config.state_dim),
        torch.randn(batch, config.action_horizon, config.action_dim),
        target_visual=torch.randn(
            batch,
            len(config.flow_jepa_effective_interval_support_offsets),
            config.visual_history_length,
            config.num_cameras,
            config.patches_per_camera,
            config.visual_token_dim,
        ),
        raw_visual=_raw_visual(config, batch=batch),
        goal_language_tokens=torch.randn(batch, 3, config.goal_language_dim),
        goal_language_mask=torch.ones(batch, 3, dtype=torch.bool),
        make_counterfactuals=False,
    )
    for key in (
        "flow_jepa_online_horizon_address",
        "flow_jepa_online_horizon_address_write_rms",
        "flow_jepa_online_address_boundary_post_g3_adjacent_cosine",
        "flow_jepa_online_address_boundary_post_address_adjacent_cosine",
        "flow_jepa_online_address_boundary_post_w1_adjacent_cosine",
        "flow_jepa_online_address_boundary_post_w2_adjacent_cosine",
        "flow_jepa_online_address_boundary_post_w3_adjacent_cosine",
        "flow_jepa_online_address_boundary_post_interval_adjacent_cosine",
    ):
        assert key in output
        assert torch.isfinite(output[key]).all()
    output["pred_physical_velocity"].float().square().mean().backward()
    encoder = system.planner.flow_dino_evidence
    assert encoder is not None and encoder.horizon_address_jepa is not None
    assert encoder.soft_address_compiler is not None
    for parameter in (
        encoder.horizon_address_jepa.query_proj.weight,
        encoder.horizon_address_jepa.value_out.weight,
        encoder.soft_address_compiler.source_dino[1].weight,
        encoder.soft_address_compiler.raw_pair_key[-1].weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert float(parameter.grad.abs().sum()) > 0.0

    groups = _optimizer_groups(system, _complete_v108_trainer())
    owners = [id(parameter) for group in groups for parameter in group["params"]]
    trainable = [
        id(parameter) for parameter in system.parameters() if parameter.requires_grad
    ]
    assert len(owners) == len(set(owners))
    assert set(owners) == set(trainable)


def test_v108_future_teacher_cannot_condition_online_address_action() -> None:
    torch.manual_seed(184)
    config = _complete_v108_config(
        dropout=0.0,
        flow_jepa_raw_activation_checkpoint=0,
    )
    system = V39PolicySystem(config).eval()
    batch = 1
    visual = _visual(config, batch=batch)
    raw_visual = _raw_visual(config, batch=batch)
    state_history = torch.randn(
        batch, config.visual_history_length, config.state_dim
    )
    executed = torch.randn(
        batch, config.executed_history_length, config.action_dim
    )
    state = torch.randn(batch, config.state_dim)
    target_action = torch.randn(batch, config.action_horizon, config.action_dim)
    target_a = torch.randn(
        batch,
        len(config.flow_jepa_effective_interval_support_offsets),
        config.visual_history_length,
        config.num_cameras,
        config.patches_per_camera,
        config.visual_token_dim,
    )
    target_b = target_a + 3.0 * torch.randn_like(target_a)
    noise = torch.randn(batch, config.action_horizon, config.action_dim)
    common = {
        "raw_visual": raw_visual,
        "goal_language_tokens": torch.randn(batch, 3, config.goal_language_dim),
        "goal_language_mask": torch.ones(batch, 3, dtype=torch.bool),
        "training_noise": system.codec.encode(noise, state),
        "training_time": torch.ones(batch),
        "proposal_keep": torch.ones(batch),
        "make_counterfactuals": False,
    }
    with torch.no_grad():
        first = system.flow_training_forward(
            visual,
            state_history,
            executed,
            state,
            target_action,
            target_visual=target_a,
            **common,
        )
        changed = system.flow_training_forward(
            visual,
            state_history,
            executed,
            state,
            target_action,
            target_visual=target_b,
            **common,
        )
    assert not torch.equal(
        first["flow_jepa_future_target"],
        changed["flow_jepa_future_target"],
    )
    torch.testing.assert_close(
        first["pred_physical_velocity"],
        changed["pred_physical_velocity"],
        rtol=0.0,
        atol=0.0,
    )


def test_v108_deploy_address_is_diagnostic_independent_and_single_read() -> None:
    torch.manual_seed(183)
    config = _complete_v108_config(
        dropout=0.0,
        inference_steps=1,
        flow_jepa_raw_activation_checkpoint=0,
    )
    system = V39PolicySystem(config).eval()
    encoder = system.planner.flow_dino_evidence
    assert encoder is not None and encoder.horizon_address_jepa is not None
    batch = 1
    visual = _visual(config, batch=batch)
    raw_visual = _raw_visual(config, batch=batch)
    state_history = torch.randn(
        batch, config.visual_history_length, config.state_dim
    )
    executed = torch.randn(
        batch, config.executed_history_length, config.action_dim
    )
    state = torch.randn(batch, config.state_dim)
    noise = torch.randn(batch, config.action_horizon, config.action_dim)
    goal = torch.randn(batch, 3, config.goal_language_dim)
    goal_mask = torch.ones(batch, 3, dtype=torch.bool)
    address_calls = 0
    planner_calls = 0

    def count_address(
        _module: torch.nn.Module,
        _inputs: tuple[object, ...],
        _output: object,
    ) -> None:
        nonlocal address_calls
        address_calls += 1

    def count_planner(
        _module: torch.nn.Module,
        _inputs: tuple[object, ...],
        _output: object,
    ) -> None:
        nonlocal planner_calls
        planner_calls += 1

    address_handle = encoder.horizon_address_jepa.register_forward_hook(
        count_address
    )
    planner_handle = system.planner.register_forward_hook(count_planner)
    try:
        with patch.object(
            encoder,
            "predict_future_with_address",
            wraps=encoder.predict_future_with_address,
        ) as late_address:
            diagnostic = system.sample(
                visual,
                state_history,
                executed,
                state,
                raw_visual=raw_visual,
                noise=noise,
                steps=3,
                return_event_logits=True,
                collect_diagnostics=True,
                goal_language_tokens=goal,
                goal_language_mask=goal_mask,
            )
            minimal = system.sample(
                visual,
                state_history,
                executed,
                state,
                raw_visual=raw_visual,
                noise=noise,
                steps=3,
                return_event_logits=True,
                collect_diagnostics=False,
                goal_language_tokens=goal,
                goal_language_mask=goal_mask,
            )
            assert late_address.call_count == 0
    finally:
        address_handle.remove()
        planner_handle.remove()
    assert isinstance(diagnostic, dict) and isinstance(minimal, dict)
    torch.testing.assert_close(
        diagnostic["action"],
        minimal["action"],
        rtol=0.0,
        atol=0.0,
    )
    assert planner_calls > 2
    assert address_calls == planner_calls


def test_v108_flags_off_and_owned_address_intervention_contract() -> None:
    config = _complete_v107_config(dropout=0.0)
    assert int(config.flow_jepa_online_horizon_address) == 0
    planner = V39PolicySystem(config).planner.eval()
    planner.set_action_path_eval_intervention("horizon_address_zero")
    base = torch.randn(2, config.future_token_count, config.hidden_size)
    refined = base + torch.randn_like(base) * 0.1
    zeroed = planner._intervene_online_horizon_address(base, refined)
    torch.testing.assert_close(zeroed, base, rtol=0.0, atol=0.0)
    state = planner.action_path_eval_intervention_state()
    assert state["apply_count"] == 1
    assert float(state["horizon_address_intervention_delta_norm"]) > 0.0


def test_v108_flag_off_uses_the_v107_late_auxiliary_topology() -> None:
    torch.manual_seed(185)
    config = _complete_v107_config(
        dropout=0.0,
        inference_steps=1,
        flow_jepa_raw_activation_checkpoint=0,
    )
    system = V39PolicySystem(config).eval()
    encoder = system.planner.flow_dino_evidence
    assert encoder is not None
    with patch.object(
        encoder,
        "organize_horizon_address",
        wraps=encoder.organize_horizon_address,
    ) as online_address, patch.object(
        encoder,
        "predict_future_with_address",
        wraps=encoder.predict_future_with_address,
    ) as late_address:
        sampled = system.sample(
            _visual(config, batch=1),
            torch.randn(1, config.visual_history_length, config.state_dim),
            torch.randn(1, config.executed_history_length, config.action_dim),
            torch.randn(1, config.state_dim),
            raw_visual=_raw_visual(config, batch=1),
            noise=torch.randn(1, config.action_horizon, config.action_dim),
            steps=1,
            return_event_logits=True,
            collect_diagnostics=True,
            goal_language_tokens=torch.randn(1, 3, config.goal_language_dim),
            goal_language_mask=torch.ones(1, 3, dtype=torch.bool),
        )
    assert isinstance(sampled, dict)
    assert online_address.call_count == 0
    assert late_address.call_count == 1


def _synthetic_v109_address_bank(
    config: V39PolicyConfig,
    *,
    batch: int = 2,
    candidates: int = 4,
    zero_values: bool = False,
) -> SoftAddressLatticeBank:
    cameras = int(config.num_cameras)
    grid = int(config.future_grid_size)
    slots = int(config.flow_jepa_address_slots)
    route_dim = int(config.flow_jepa_address_route_dim)
    chart = grid * grid
    raw_dim = int(config.flow_jepa_raw_base_channels)
    raw_dim += raw_dim // 2
    axis = torch.linspace(-1.0, 1.0, grid)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    chart_coordinates = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)
    coarse_coordinates = chart_coordinates[None, None].expand(
        batch, cameras, -1, -1
    ).clone()
    flow_centers = chart_coordinates.reshape(1, 1, grid, grid, 2).expand(
        batch, cameras, -1, -1, -1
    ).clone()
    fine_coordinates = flow_centers[..., None, None, :].expand(
        batch, cameras, grid, grid, slots, candidates, 2
    ).clone()
    fine_coordinates = fine_coordinates + 0.05 * torch.randn_like(
        fine_coordinates
    )
    values = torch.randn(
        batch, cameras, grid, grid, slots, candidates, raw_dim
    )
    if zero_values:
        values.zero_()
    return SoftAddressLatticeBank(
        coarse_keys=torch.randn(
            batch, cameras, grid, grid, slots, route_dim
        ),
        fine_keys=torch.randn(
            batch, cameras, grid, grid, slots, candidates, route_dim
        ),
        fine_values=values,
        fine_valid=torch.ones(
            batch,
            cameras,
            grid,
            grid,
            slots,
            candidates,
            dtype=torch.bool,
        ),
        coarse_centers=flow_centers[..., None, :].expand(
            batch, cameras, grid, grid, slots, 2
        ).clone(),
        coarse_variance=torch.full(
            (batch, cameras, grid, grid, slots, 2), 0.05
        ),
        fine_radius=torch.full(
            (batch, cameras, grid, grid, slots), 0.10
        ),
        coarse_base_logits=torch.randn(
            batch, cameras, grid, grid, slots, chart
        ),
        coarse_candidate_keys=torch.randn(
            batch, cameras, chart, route_dim
        ),
        coarse_candidate_coordinates=coarse_coordinates,
        coarse_flow_centers=flow_centers,
        coarse_confidence=torch.rand(batch, cameras, grid, grid),
        coarse_uncertainty=torch.rand(batch, cameras, grid, grid),
        coarse_occlusion=torch.rand(batch, cameras, grid, grid),
        coarse_cycle_error=torch.rand(batch, cameras, grid, grid),
        fine_coordinates=fine_coordinates,
        coarse_source_centers=flow_centers,
        dense_source_raw_keys=torch.randn(
            batch, cameras, route_dim, 8, 8
        ),
        dense_target_raw_keys=torch.randn(
            batch, cameras, route_dim, 8, 8
        ),
        dense_target_dino_keys=torch.randn(
            batch, cameras, route_dim, grid, grid
        ),
        dense_target_detail=(
            torch.zeros(batch, cameras, raw_dim, 8, 8)
            if zero_values
            else torch.randn(batch, cameras, raw_dim, 8, 8)
        ),
        dense_confidence=torch.rand(batch, cameras, 1, 8, 8),
        dense_uncertainty=torch.rand(batch, cameras, 1, 8, 8),
        dense_occlusion=torch.rand(batch, cameras, 1, 8, 8),
        dense_current_rgb=(
            torch.zeros(batch, cameras, 3, 16, 16)
            if zero_values
            else torch.rand(batch, cameras, 3, 16, 16).mul(2.0).sub(1.0)
        )
        if int(getattr(config, "flow_jepa_coordinate_typed_raw_detail", 0))
        else None,
    )


def test_v109_contract_and_flag_off_ancestry() -> None:
    config = _complete_v109_config()
    trainer = _complete_v109_trainer()
    _validate_complete_v109_model_contract(config, trainer)
    assert _validate_required_model_contract("v109", config, trainer) == "v109"
    with pytest.raises(
        ValueError, match="flow_jepa_progressive_grounding_address"
    ):
        _validate_complete_v109_model_contract(
            replace(config, flow_jepa_progressive_grounding_address=0),
            trainer,
        )
    v108 = _complete_v108_config()
    encoder = FlowDINOEvidenceEncoder(v108)
    assert encoder.progressive_grounding_address is None
    assert encoder.horizon_address_jepa is not None
    v109_encoder = FlowDINOEvidenceEncoder(config)
    assert v109_encoder.horizon_address_jepa is None
    assert v109_encoder.progressive_grounding_address is not None
    assert v109_encoder.soft_address_compiler is not None
    assert v109_encoder.soft_address_compiler.target_dino_value is None
    assert v109_encoder.soft_address_compiler.coarse_geometry is None


def test_v109_g1_g2_g3_priors_reach_the_only_policy_value_read() -> None:
    torch.manual_seed(186)
    config = _complete_v109_config(
        flow_jepa_grid_size=2,
        future_grid_size=2,
        patches_per_camera=4,
        flow_jepa_address_slots=2,
        flow_jepa_address_route_dim=8,
        flow_jepa_raw_reader_heads=2,
    )
    encoder = FlowDINOEvidenceEncoder(config)
    organizer = encoder.progressive_grounding_address
    assert organizer is not None
    bank = _synthetic_v109_address_bank(config)
    state = encoder.begin_progressive_grounding_address(bank)
    rollout = torch.randn(
        2,
        config.future_anchors
        * config.num_cameras
        * config.future_grid_size
        * config.future_grid_size,
        config.hidden_size,
    )
    for stage in (1, 2, 3):
        state = encoder.update_progressive_grounding_address(
            state, rollout + 0.1 * stage, stage=stage
        )
    encoder.score_progressive_horizon_posterior(rollout, state)
    detail = LateRawDetailEvidence(
        selector_tokens=rollout.new_empty(2, 0, config.hidden_size),
        value_tokens=rollout.new_empty(2, 0, config.hidden_size),
        address_bank=bank,
        progressive_address=state,
    )
    reader = LateRawDetailPolicyReader(config)
    trajectory = torch.randn(
        2,
        config.action_horizon * config.action_basis_tokens,
        config.hidden_size,
    )
    updated, metrics = reader(
        trajectory,
        rollout,
        detail,
        phase_context=torch.randn(2, config.hidden_size),
        condition_query_context=torch.randn(2, config.hidden_size),
    )
    assert float(metrics["flow_jepa_progressive_policy_prior_active"]) == 1.0
    updated.float().square().mean().backward()
    for parameter in (
        organizer.query_projections[0].weight,
        organizer.g2_rectifier[-1].weight,
        organizer.g3_slot_score[-1].weight,
        organizer.horizon_query_proj.weight,
        reader.lattice_query_proj.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert float(parameter.grad.abs().sum()) > 0.0


def test_v109_selector_priors_cannot_manufacture_raw_detail_values() -> None:
    torch.manual_seed(187)
    config = _complete_v109_config(
        flow_jepa_grid_size=2,
        future_grid_size=2,
        patches_per_camera=4,
        flow_jepa_address_slots=2,
        flow_jepa_address_route_dim=8,
        flow_jepa_raw_reader_heads=2,
    )
    encoder = FlowDINOEvidenceEncoder(config).eval()
    bank = _synthetic_v109_address_bank(
        config, batch=1, zero_values=True
    )
    state = encoder.begin_progressive_grounding_address(bank)
    rollout = torch.randn(
        1,
        config.future_token_count,
        config.hidden_size,
    )
    for stage in (1, 2, 3):
        state = encoder.update_progressive_grounding_address(
            state, rollout, stage=stage
        )
    encoder.score_progressive_horizon_posterior(rollout, state)
    reader = LateRawDetailPolicyReader(config).eval()
    trajectory = torch.randn(
        1,
        config.action_horizon * config.action_basis_tokens,
        config.hidden_size,
    )
    updated, _ = reader(
        trajectory,
        rollout,
        LateRawDetailEvidence(
            selector_tokens=rollout.new_empty(1, 0, config.hidden_size),
            value_tokens=rollout.new_empty(1, 0, config.hidden_size),
            address_bank=bank,
            progressive_address=state,
        ),
        phase_context=torch.randn(1, config.hidden_size),
        condition_query_context=torch.randn(1, config.hidden_size),
    )
    torch.testing.assert_close(updated, trajectory, rtol=0.0, atol=0.0)


def test_v109_full_action_path_uses_dynamic_g2_then_one_p_value_read() -> None:
    torch.manual_seed(188)
    config = _complete_v109_config(
        dropout=0.0,
        flow_jepa_raw_activation_checkpoint=0,
        flow_jepa_grid_size=2,
        future_grid_size=2,
        patches_per_camera=4,
        flow_jepa_address_slots=2,
        flow_jepa_address_route_dim=8,
        flow_jepa_raw_reader_heads=2,
    )
    system = V39PolicySystem(config).train()
    encoder = system.planner.flow_dino_evidence
    late_reader = system.planner.late_raw_detail_reader
    assert encoder is not None and encoder.soft_address_compiler is not None
    assert encoder.progressive_grounding_address is not None
    assert late_reader is not None
    batch = 1
    with patch.object(
        encoder,
        "organize_horizon_address",
        wraps=encoder.organize_horizon_address,
    ) as old_online, patch.object(
        encoder,
        "predict_future_with_address",
        wraps=encoder.predict_future_with_address,
    ) as old_late, patch.object(
        encoder.soft_address_compiler,
        "progressive_fine_candidates",
        wraps=encoder.soft_address_compiler.progressive_fine_candidates,
    ) as dynamic_candidates, patch.object(
        late_reader,
        "_read_soft_address_lattice",
        wraps=late_reader._read_soft_address_lattice,
    ) as policy_value_read:
        output = system.flow_training_forward(
            _visual(config, batch=batch),
            torch.randn(
                batch, config.visual_history_length, config.state_dim
            ),
            torch.randn(
                batch,
                config.executed_history_length,
                config.action_dim,
            ),
            torch.randn(batch, config.state_dim),
            torch.randn(batch, config.action_horizon, config.action_dim),
            target_visual=torch.randn(
                batch,
                len(config.flow_jepa_effective_interval_support_offsets),
                config.visual_history_length,
                config.num_cameras,
                config.patches_per_camera,
                config.visual_token_dim,
            ),
            raw_visual=_raw_visual(config, batch=batch, side=32),
            goal_language_tokens=torch.randn(
                batch, 3, config.goal_language_dim
            ),
            goal_language_mask=torch.ones(batch, 3, dtype=torch.bool),
            make_counterfactuals=False,
        )
    assert old_online.call_count == 0
    assert old_late.call_count == 0
    assert dynamic_candidates.call_count == 1
    assert policy_value_read.call_count == 1
    for key in (
        "flow_jepa_progressive_grounding_address",
        "flow_jepa_progressive_g1_coarse_entropy",
        "flow_jepa_progressive_g2_dynamic_center_distance",
        "flow_jepa_progressive_g3_summary_rms",
        "flow_jepa_progressive_world_posterior_entropy",
        "flow_jepa_progressive_world_source_horizon_variation",
        "flow_jepa_progressive_policy_prior_active",
        "flow_jepa_progressive_policy_world_prior_rms",
    ):
        assert key in output
        assert torch.isfinite(output[key]).all()
    assert "flow_jepa_online_horizon_address_write_rms" not in output
    output["pred_physical_velocity"].float().square().mean().backward()
    organizer = encoder.progressive_grounding_address
    for parameter in (
        organizer.query_projections[0].weight,
        organizer.g2_rectifier[-1].weight,
        organizer.g3_slot_score[-1].weight,
        organizer.g3_summary_out[-1].weight,
        organizer.horizon_query_proj.weight,
        encoder.soft_address_compiler.raw_pair_key[-1].weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert float(parameter.grad.abs().sum()) > 0.0


def test_v110_contract_launcher_and_flag_off_ancestry() -> None:
    config = _complete_v110_config()
    trainer = _complete_v110_trainer()
    _validate_complete_v110_model_contract(config, trainer)
    assert _validate_required_model_contract("v110", config, trainer) == "v110"
    # The micro-grid is topology-owned: with the V110 flag off it must not
    # narrow the V109 configuration domain.
    replace(
        config,
        flow_jepa_coordinate_typed_raw_detail=0,
        flow_jepa_raw_micro_grid=2,
    ).validate()
    with pytest.raises(
        ValueError, match="flow_jepa_coordinate_typed_raw_detail"
    ):
        _validate_complete_v110_model_contract(
            replace(config, flow_jepa_coordinate_typed_raw_detail=0), trainer
        )
    with pytest.raises(ValueError, match="flow_jepa_raw_micro_grid=3"):
        _validate_complete_v110_model_contract(
            replace(config, flow_jepa_raw_micro_grid=5), trainer
        )
    launcher = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "current_v110_coordinate_typed_raw_jepa.sh"
    ).read_text(encoding="utf-8")
    assert "CLEARVLA_REQUIRED_MODEL_CONTRACT=v110" in launcher
    assert "--flow-jepa-coordinate-typed-raw-detail 1" in launcher
    assert "--flow-jepa-raw-micro-grid 3" in launcher
    smoke = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "current_v110_coordinate_typed_raw_jepa_smoke.sh"
    ).read_text(encoding="utf-8")
    assert "--max-train-batches" in smoke
    assert "--eval-sampling-diagnostic-batches 1" in smoke


def test_v110_p2_conditioning_cannot_manufacture_precision_values() -> None:
    torch.manual_seed(190)
    refiner = _CoordinateTypedLocalRefiner(
        width=16,
        raw_dim=12,
        route_dim=8,
        depth=2,
    ).eval()
    zero_rgb = torch.zeros(3, 1, 9, 3)
    zero_detail = torch.zeros(3, 1, 9, 12)
    output, _ = refiner(
        rgb=zero_rgb,
        learned_detail=zero_detail,
        coordinates=torch.randn(3, 1, 9, 2),
        query=torch.randn(3, 1, 8),
        semantic=torch.randn(3, 1, 8),
        appearance=torch.randn(3, 1, 8),
        geometry=torch.randn(3, 1, 8),
        future_transport=torch.randn(3, 1, 5),
    )
    torch.testing.assert_close(output, torch.zeros_like(output), rtol=0.0, atol=0.0)


def test_v110_streamed_microgrid_matches_materialized_value_and_gradients() -> None:
    torch.manual_seed(193)
    batch, queries, glimpses, cameras = 1, 2, 2, 2
    side, slots, candidates, micro = 2, 2, 5, 3
    detail_dim = 4
    route_logits = torch.randn(
        batch,
        queries,
        glimpses,
        cameras,
        side,
        side,
        slots,
        requires_grad=True,
    )
    fine_logits = torch.randn(
        *route_logits.shape,
        candidates,
        requires_grad=True,
    )
    route = torch.softmax(route_logits.flatten(3), dim=-1).reshape_as(route_logits)
    fine = torch.softmax(fine_logits, dim=-1)
    basis = torch.softmax(torch.randn(candidates, micro), dim=-1)
    value_shape = (batch, cameras, side, side, slots, candidates)
    rgb = torch.randn(*value_shape, 3, requires_grad=True)
    detail = torch.randn(*value_shape, detail_dim, requires_grad=True)
    coordinates = torch.randn(*value_shape, 2, requires_grad=True)

    materialized_weight = fine[..., None] * basis
    materialized_weight = materialized_weight / materialized_weight.sum(
        dim=-2, keepdim=True
    ).clamp_min(1e-8)

    def reference(value: torch.Tensor) -> torch.Tensor:
        state = torch.einsum(
            "bqgcijmkl,bcijmkv->bqgcijmlv", materialized_weight, value
        )
        return torch.einsum("bqgcijm,bqgcijmlv->bqglv", route, state)

    expected = (reference(rgb), reference(detail), reference(coordinates))
    actual = LateRawDetailPolicyReader._typed_microgrid_expectation(
        route,
        fine,
        basis,
        rgb,
        detail,
        coordinates,
    )
    for actual_row, expected_row in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_row, expected_row, rtol=2e-5, atol=2e-6)

    probes = tuple(torch.randn_like(row) for row in expected)
    parameters = (route_logits, fine_logits, rgb, detail, coordinates)
    expected_loss = sum(
        (value * probe).sum()
        for value, probe in zip(expected, probes, strict=True)
    )
    expected_gradients = torch.autograd.grad(
        expected_loss,
        parameters,
        retain_graph=True,
    )
    actual_loss = sum(
        (value * probe).sum()
        for value, probe in zip(actual, probes, strict=True)
    )
    actual_gradients = torch.autograd.grad(actual_loss, parameters)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            rtol=5e-5,
            atol=5e-6,
        )


def test_v110_literal_rgb_chart_keeps_native_resolution() -> None:
    torch.manual_seed(192)
    config = _complete_v110_config(
        flow_jepa_grid_size=2,
        future_grid_size=2,
        patches_per_camera=4,
        flow_jepa_address_slots=2,
        flow_jepa_address_route_dim=8,
    )
    raw_dim = (
        config.flow_jepa_raw_base_channels
        + config.flow_jepa_raw_base_channels // 2
    )
    compiler = _SoftMultiResolutionAddressCompiler(config, raw_dim=raw_dim).eval()
    batch = 1
    cameras = config.num_cameras
    dino_side = 2
    raw_side = 16
    rgb_side = 64
    source_dino = torch.randn(
        batch, cameras, dino_side, dino_side, config.visual_token_dim
    )
    target_dino = torch.randn_like(source_dino)
    source_raw = torch.randn(batch, cameras, raw_dim, raw_side, raw_side)
    target_raw = torch.randn_like(source_raw)
    current_rgb = torch.rand(batch, cameras, 3, rgb_side, rgb_side)
    flow = torch.zeros(batch, cameras, 2, raw_side, raw_side)
    confidence = torch.full((batch, cameras, 1, raw_side, raw_side), 0.5)
    uncertainty = torch.full_like(confidence, 0.1)
    occlusion = torch.zeros_like(confidence)
    bank, _ = compiler(
        source_dino=source_dino,
        target_dino=target_dino,
        source_raw=source_raw,
        target_raw=target_raw,
        current_rgb=current_rgb,
        flow=flow,
        confidence=confidence,
        uncertainty=uncertainty,
        occlusion=occlusion,
    )
    assert bank.dense_current_rgb is not None
    assert tuple(bank.dense_current_rgb.shape) == (
        batch,
        cameras,
        3,
        rgb_side,
        rgb_side,
    )
    torch.testing.assert_close(
        bank.dense_current_rgb.float(),
        2.0 * current_rgb.float() - 1.0,
    )


def test_v110_typed_gwp_path_is_attached_and_zero_value_exact() -> None:
    torch.manual_seed(191)
    config = _complete_v110_config(
        flow_jepa_grid_size=2,
        future_grid_size=2,
        patches_per_camera=4,
        flow_jepa_address_slots=2,
        flow_jepa_address_route_dim=8,
        flow_jepa_raw_reader_heads=2,
    )
    encoder = FlowDINOEvidenceEncoder(config).eval()
    organizer = encoder.progressive_grounding_address
    assert organizer is not None
    assert organizer.g3_typed_slot_score is not None
    assert organizer.world_typed_query is not None

    def run_bank(bank: SoftAddressLatticeBank) -> tuple[torch.Tensor, torch.Tensor]:
        # Match the BF16 autocast contract used by training/preflight.  This
        # traverses G2, W and both P query projections, so a learned module
        # accidentally called from an enabled=False FP32 island fails here.
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            state = encoder.begin_progressive_grounding_address(bank)
            rollout = torch.randn(1, config.future_token_count, config.hidden_size)
            for stage in (1, 2, 3):
                state = encoder.update_progressive_grounding_address(
                    state, rollout, stage=stage
                )
            encoder.score_progressive_horizon_posterior(rollout, state)
            assert state.dynamic_semantic_keys is not None
            assert state.dynamic_appearance_keys is not None
            assert state.dynamic_geometry_keys is not None
            assert state.dynamic_literal_rgb is not None
            assert state.world_future_offset is not None
            assert state.canonical_summary_tokens is not None
            assert int(state.canonical_summary_tokens.shape[1]) == (
                3
                * config.num_cameras
                * config.future_grid_size
                * config.future_grid_size
            )
            trajectory = torch.randn(
                1,
                config.action_horizon * config.action_basis_tokens,
                config.hidden_size,
            )
            reader = LateRawDetailPolicyReader(config).train()
            updated, metrics = reader(
                trajectory,
                rollout,
                LateRawDetailEvidence(
                    selector_tokens=rollout.new_empty(1, 0, config.hidden_size),
                    value_tokens=rollout.new_empty(1, 0, config.hidden_size),
                    address_bank=bank,
                    progressive_address=state,
                ),
                phase_context=torch.randn(1, config.hidden_size),
                condition_query_context=torch.randn(1, config.hidden_size),
            )
        assert float(metrics["flow_jepa_coordinate_typed_raw_detail"]) == 1.0
        assert float(metrics["flow_jepa_typed_p1_activation_checkpoint"]) == 1.0
        return updated, trajectory

    zero_updated, zero_trajectory = run_bank(
        _synthetic_v109_address_bank(config, batch=1, zero_values=True)
    )
    torch.testing.assert_close(
        zero_updated, zero_trajectory, rtol=0.0, atol=0.0
    )

    updated, trajectory = run_bank(
        _synthetic_v109_address_bank(config, batch=1, zero_values=False)
    )
    (updated - trajectory).float().square().mean().backward()
    parameters = (
        organizer.g2_typed_rectifier[-1].weight,
        organizer.g3_typed_slot_score["semantic"][-1].weight,
        organizer.world_typed_query["semantic"].weight,
        organizer.future_transport[-1].weight,
    )
    for parameter in parameters:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert float(parameter.grad.abs().sum()) > 0.0
    rectifier_grad = organizer.g2_typed_rectifier[-1].weight.grad
    assert rectifier_grad is not None
    # dx, dy, support, and prior-strength must all remain attached.  This
    # guards against silently shrinking the typed rectifier to three outputs.
    assert tuple(rectifier_grad.shape[:1]) == (4,)
    assert torch.all(rectifier_grad.abs().sum(dim=-1) > 0.0)


def test_v111_contract_launcher_and_v110_flag_off_ancestry() -> None:
    config = _complete_v111_config()
    trainer = _complete_v111_trainer()
    _validate_complete_v111_model_contract(config, trainer)
    assert _validate_required_model_contract("v111", config, trainer) == "v111"
    v110 = replace(config, flow_jepa_structured_ownership_bottleneck=0)
    _validate_complete_v110_model_contract(v110, trainer)
    with pytest.raises(
        ValueError, match="flow_jepa_structured_ownership_bottleneck"
    ):
        _validate_complete_v111_model_contract(v110, trainer)
    launcher = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "current_v111_structured_ownership_bottleneck.sh"
    ).read_text(encoding="utf-8")
    assert "CLEARVLA_REQUIRED_MODEL_CONTRACT=v111" in launcher
    assert "--flow-jepa-structured-ownership-bottleneck 1" in launcher
    smoke = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "current_v111_structured_ownership_bottleneck_smoke.sh"
    ).read_text(encoding="utf-8")
    assert "--max-train-batches" in smoke
    assert "--eval-sampling-diagnostic-batches 1" in smoke


def test_v111_p2_keeps_value_lanes_zero_exact_and_owner_gradients_natural() -> None:
    torch.manual_seed(211)
    refiner = _StructuredOwnershipLocalRefiner(
        width=16,
        raw_dim=12,
        route_dim=8,
        depth=2,
    ).eval()
    contexts = {
        "coordinates": torch.randn(3, 1, 9, 2),
        "query": torch.randn(3, 1, 8),
        "semantic": torch.randn(3, 1, 8),
        "appearance": torch.randn(3, 1, 8),
        "geometry": torch.randn(3, 1, 8),
        "future_transport": torch.randn(3, 1, 5),
    }
    zero_output, _ = refiner(
        rgb=torch.zeros(3, 1, 9, 3),
        learned_detail=torch.zeros(3, 1, 9, 12),
        **contexts,
    )
    torch.testing.assert_close(
        zero_output, torch.zeros_like(zero_output), rtol=0.0, atol=0.0
    )

    output, metrics = refiner(
        rgb=torch.randn(3, 1, 9, 3),
        learned_detail=torch.randn(3, 1, 9, 12),
        **contexts,
    )
    output.square().mean().backward()
    for name in refiner.OWNER_NAMES:
        condition_grad = refiner.owner_conditions[name].weight.grad
        output_grad = refiner.owner_outputs[name].weight.grad
        assert condition_grad is not None and torch.isfinite(condition_grad).all()
        assert output_grad is not None and torch.isfinite(output_grad).all()
        assert float(condition_grad.abs().sum()) > 0.0
        assert float(output_grad.abs().sum()) > 0.0
        assert f"flow_jepa_typed_p2_{name}_contribution_rms" in metrics
    for parameter_name, parameter in refiner.named_parameters():
        if not parameter.requires_grad:
            continue
        assert parameter.grad is not None, parameter_name
        assert torch.isfinite(parameter.grad).all(), parameter_name
        assert float(parameter.grad.abs().sum()) > 0.0, parameter_name


def test_v111_structured_gwp_ownership_is_attached_without_capacity_loss() -> None:
    torch.manual_seed(212)
    config = _complete_v111_config(
        flow_jepa_grid_size=2,
        future_grid_size=2,
        patches_per_camera=4,
        flow_jepa_address_slots=2,
        flow_jepa_address_route_dim=8,
        flow_jepa_raw_reader_heads=2,
    )
    encoder = FlowDINOEvidenceEncoder(config).eval()
    organizer = encoder.progressive_grounding_address
    assert organizer is not None and organizer.structured_ownership
    bank = _synthetic_v109_address_bank(config, batch=1, zero_values=False)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        state = encoder.begin_progressive_grounding_address(bank)
        rollout = torch.randn(1, config.future_token_count, config.hidden_size)
        for stage in (1, 2, 3):
            state = encoder.update_progressive_grounding_address(
                state, rollout, stage=stage
            )
        assert state.g2_semantic_probability is not None
        assert state.g2_appearance_probability is not None
        assert state.g2_geometry_probability is not None
        assert state.canonical_summary_tokens is not None
        public_tokens = (
            config.num_cameras
            * config.future_grid_size
            * config.future_grid_size
        )
        assert int(state.canonical_summary_tokens.shape[1]) == public_tokens
        owner_keys = (
            state.canonical_semantic_keys,
            state.canonical_appearance_keys,
            state.canonical_geometry_keys,
        )
        assert all(value is not None for value in owner_keys)
        owner_tokens = sum(
            int(value.reshape(1, -1, value.shape[-1]).shape[1])
            for value in owner_keys
            if value is not None
        )
        assert owner_tokens == 3 * public_tokens * config.flow_jepa_address_slots

        encoder.score_progressive_horizon_posterior(rollout, state)
        assert state.world_public_query is not None
        assert state.world_horizon_innovation is not None
        torch.testing.assert_close(
            state.world_horizon_innovation.float().mean(dim=1),
            torch.zeros_like(state.world_horizon_innovation[:, 0]).float(),
            rtol=2e-3,
            atol=2e-3,
        )
        assert state.world_interval_offset_delta is not None
        assert state.world_interval_log_scale_delta is not None
        assert state.world_future_uncertainty is not None
        uncertainty_delta = (
            state.world_future_uncertainty[:, 1:].float()
            - state.world_future_uncertainty[:, :-1].float()
        )
        assert torch.all(uncertainty_delta >= -1e-5)

        trajectory = torch.randn(
            1,
            config.action_horizon * config.action_basis_tokens,
            config.hidden_size,
        )
        reader = LateRawDetailPolicyReader(config).train()
        assert isinstance(
            reader.typed_local_refiners[0], _StructuredOwnershipLocalRefiner
        )
        updated, metrics = reader(
            trajectory,
            rollout,
            LateRawDetailEvidence(
                selector_tokens=rollout.new_empty(1, 0, config.hidden_size),
                value_tokens=rollout.new_empty(1, 0, config.hidden_size),
                address_bank=bank,
                progressive_address=state,
            ),
            phase_context=torch.randn(1, config.hidden_size),
            condition_query_context=torch.randn(1, config.hidden_size),
        )
    assert float(metrics["flow_jepa_structured_ownership_bottleneck"]) == 1.0
    for name in (
        "semantic",
        "appearance",
        "geometry",
    ):
        metric = f"flow_jepa_progressive_g3_{name}_owner_sidecar_rms"
        assert metric in state.metrics and torch.isfinite(state.metrics[metric])
    for name in (
        "flow_jepa_typed_p1_semantic_appearance_fine_l1",
        "flow_jepa_typed_p1_appearance_geometry_fine_l1",
        "flow_jepa_typed_p1_semantic_appearance_route_l1",
        "flow_jepa_typed_p1_appearance_geometry_route_l1",
    ):
        assert name in metrics and torch.isfinite(metrics[name])
    (updated - trajectory).float().square().mean().backward()
    for parameter in (
        organizer.g2_typed_query["semantic"].weight,
        organizer.g2_typed_query["appearance"].weight,
        organizer.g2_typed_query["geometry"].weight,
        organizer.g3_typed_slot_score["semantic"][-1].weight,
        organizer.g3_typed_slot_score["appearance"][-1].weight,
        organizer.g3_typed_slot_score["geometry"][-1].weight,
        organizer.world_typed_query["semantic"].weight,
        organizer.world_typed_query["appearance"].weight,
        organizer.world_typed_query["geometry"].weight,
        organizer.future_transport[-1].weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert float(parameter.grad.abs().sum()) > 0.0
