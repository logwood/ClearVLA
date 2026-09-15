from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from clearvla.data.samplers import (
    InformationBalancedBatchSampler,
    InformationBalancedSamplerConfig,
)
from clearvla.experiments.classic_policy_lab.cli_common import (
    add_data_args,
    load_data,
    make_loader,
    preprocessing_from_args,
    print_context,
    resolve_device,
)
from clearvla.experiments.classic_policy_lab.normalizer import ArrayNormalizer
from clearvla.experiments.dynamic_world_lab.conditioning import (
    build_dense_conditioner,
    infer_dense_geometry,
)
from clearvla.experiments.observed_state_lab.dataset import (
    CachedTokenPolicyWindowDataset,
    ObservedStateDatasetConfig,
    ObservedStateWindowDataset,
    PolicyWindowDataset,
)
from clearvla.experiments.observed_state_lab.policy_runtime_v39 import (
    POLICY_CHECKPOINT_SCHEMAS,
    V39PolicyTrainerConfig,
    _validate_complete_v103_model_probe_contract,
    _validate_complete_v104_model_contract,
    _validate_complete_v105_model_contract,
    _validate_complete_v106_model_contract,
    _validate_complete_v107_model_contract,
    _validate_complete_v108_model_contract,
    _validate_complete_v109_model_contract,
    _validate_complete_v110_model_contract,
    _validate_complete_v111_model_contract,
    _validate_complete_v112_model_contract,
    _validate_complete_v113_model_contract,
    _validate_complete_v114_model_contract,
    _validate_complete_v115_model_contract,
    _validate_complete_v116_model_contract,
    _validate_complete_v117_model_contract,
    _validate_differential_intent_effect_323_model_contract,
    _validate_grounded_intent_effect_323_model_contract,
    train_v39_policy,
)
from clearvla.policy.config import V39PolicyConfig
from clearvla.policy.goal_conditioning import load_precomputed_t5_condition
from clearvla.policy.grounded_intent_effect import GROUNDING_MANIFEST
from clearvla.policy.object_intent_dynamics_323 import (
    ARCHITECTURE_MANIFEST as OBJECT_INTENT_DYNAMICS_MANIFEST,
)
from clearvla.policy.system import V39PolicySystem


def _parse_offsets(text: str) -> tuple[int, ...]:
    values = tuple(int(x) for x in str(text).replace(",", " ").split())
    if not values:
        raise argparse.ArgumentTypeError("offset list must be non-empty")
    return values


def _normalizer_fingerprint(normalizer: ArrayNormalizer) -> str:
    """Short stable hash of the normalizer statistics.

    physical_flow_native_uniform is a cross-version anchor metric ONLY between
    runs that share this fingerprint: the native metric is priced in
    normalizer coordinates, so refitting the normalizer silently re-scales it.
    """
    def _round(value: Any) -> Any:
        array = np.asarray(value)
        if array.dtype.kind in "fc":
            return np.round(array.astype(np.float64), 6).tolist()
        return array.tolist() if array.ndim else value

    payload = json.dumps(
        {key: _round(value) for key, value in sorted(normalizer.to_dict().items())},
        sort_keys=True,
        default=str,
    )
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]


def _validate_required_model_contract(
    required_contract: str | None,
    policy_config: V39PolicyConfig,
    trainer: V39PolicyTrainerConfig,
) -> str | None:
    """Fail before training when a formal launcher resolves to another graph."""

    normalized = (
        ""
        if required_contract is None
        else str(required_contract).strip().lower().replace("-", "_")
    )
    if not normalized:
        return None
    if normalized not in {
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
        "differential_intent_effect_323",
        "grounded_intent_effect_323",
        "object_intent_dynamics_323",
    }:
        raise ValueError(f"unknown required model contract: {required_contract!r}")
    if normalized == "object_intent_dynamics_323":
        OBJECT_INTENT_DYNAMICS_MANIFEST.validate()
        if not int(policy_config.flow_jepa_object_intent_dynamics_mainline):
            raise ValueError(
                "object_intent_dynamics_323 launcher resolved to another top graph"
            )
        if int(policy_config.flow_jepa_grounded_intent_effect_mainline) or int(
            policy_config.flow_jepa_differential_intent_effect_mainline
        ):
            raise ValueError(
                "object_intent_dynamics_323 cannot share a historical top graph"
            )
        if not int(policy_config.goal_conditioning_enabled):
            raise ValueError(
                "object_intent_dynamics_323 requires the complete T5 condition"
            )
        if int(trainer.future_latent_loss_start_epoch) != 1 or int(
            trainer.future_latent_max_batches
        ) != 0:
            raise ValueError(
                "object-intent training requires its future teacher on every batch"
            )
        if float(trainer.flow_jepa_future_loss_weight) <= 0.0 or float(
            trainer.flow_jepa_interval_stage_loss_weight
        ) <= 0.0:
            raise ValueError(
                "object-intent W and G/S supervision require the existing "
                "future and interval objective budgets"
            )
        if str(policy_config.flow_matching_time_distribution) != "beta_1_5_1":
            raise ValueError(
                "object-intent formal training requires beta_1_5_1 flow time"
            )
        if str(trainer.training_stage).lower().replace("-", "_") not in {
            "policy",
            "stage2",
        }:
            raise ValueError(
                "object-intent dynamics is a single-stage end-to-end policy "
                "graph; the historical representation-only Stage1 is invalid"
            )
    elif normalized == "grounded_intent_effect_323":
        _validate_grounded_intent_effect_323_model_contract(
            policy_config,
            trainer,
        )
    elif normalized == "differential_intent_effect_323":
        _validate_differential_intent_effect_323_model_contract(
            policy_config,
            trainer,
        )
    elif normalized == "v117":
        _validate_complete_v117_model_contract(policy_config, trainer)
    elif normalized == "v116":
        _validate_complete_v116_model_contract(policy_config, trainer)
    elif normalized == "v115":
        _validate_complete_v115_model_contract(policy_config, trainer)
    elif normalized == "v114":
        _validate_complete_v114_model_contract(policy_config, trainer)
    elif normalized == "v113":
        _validate_complete_v113_model_contract(policy_config, trainer)
    elif normalized == "v112":
        _validate_complete_v112_model_contract(policy_config, trainer)
    elif normalized == "v111":
        _validate_complete_v111_model_contract(policy_config, trainer)
    elif normalized == "v110":
        _validate_complete_v110_model_contract(policy_config, trainer)
    elif normalized == "v109":
        _validate_complete_v109_model_contract(policy_config, trainer)
    elif normalized == "v108":
        _validate_complete_v108_model_contract(policy_config, trainer)
    elif normalized == "v107":
        _validate_complete_v107_model_contract(policy_config, trainer)
    elif normalized == "v106":
        _validate_complete_v106_model_contract(policy_config, trainer)
    elif normalized == "v105":
        _validate_complete_v105_model_contract(policy_config, trainer)
    elif normalized == "v104":
        _validate_complete_v104_model_contract(policy_config, trainer)
    else:
        _validate_complete_v103_model_probe_contract(policy_config, trainer)
    return normalized


def _source_fingerprint() -> dict[str, str]:
    """Hash the source surface that defines the current experiment contract."""

    root = Path(__file__).resolve().parents[2]
    relative_paths = (
        "clearvla/cli/train_v40_policy.py",
        "clearvla/data/samplers.py",
        "clearvla/experiments/classic_policy_lab/cli_common.py",
        "clearvla/experiments/observed_state_lab/dataset.py",
        "clearvla/experiments/observed_state_lab/policy_runtime_v36_3.py",
        "clearvla/experiments/observed_state_lab/policy_runtime_v39.py",
        "clearvla/policy/codec.py",
        "clearvla/policy/config.py",
        "clearvla/policy/controller.py",
        "clearvla/policy/differential_intent_effect.py",
        "clearvla/policy/grounded_intent_effect.py",
        "clearvla/policy/object_intent_dynamics_323/__init__.py",
        "clearvla/policy/object_intent_dynamics_323/types.py",
        "clearvla/policy/object_intent_dynamics_323/grounding.py",
        "clearvla/policy/object_intent_dynamics_323/intent.py",
        "clearvla/policy/object_intent_dynamics_323/teacher.py",
        "clearvla/policy/object_intent_dynamics_323/dynamics.py",
        "clearvla/policy/object_intent_dynamics_323/compiler.py",
        "clearvla/policy/flow_dino_evidence.py",
        "clearvla/policy/goal_conditioning.py",
        "clearvla/policy/proposal.py",
        "clearvla/policy/role_delta_attnres.py",
        "clearvla/policy/system.py",
        "clearvla/policy/time_domain_mmdit.py",
        "clearvla/policy/trunk.py",
        "clearvla/policy/trunk_primitives.py",
        "scripts/current_v94_latent_ownership_execution.sh",
        "scripts/current_v95_flow_dino_jepa.sh",
        "scripts/current_v95_flow_dino_jepa_stage1.sh",
        "scripts/current_v95_flow_dino_jepa_policy.sh",
        "scripts/current_v96_late_bottleneck_jepa.sh",
        "scripts/current_v97_raw_flow_332_jepa.sh",
        "scripts/current_v98_dino_seeded_raw_flow_332_jepa.sh",
        "scripts/current_v99_observable_raw_flow_332_jepa.sh",
        "scripts/current_v100_strict_complementary_flow_jepa.sh",
        "scripts/current_v101_information_balanced_long_horizon.sh",
        "scripts/current_v102_anchor_world_late_raw_detail.sh",
        "scripts/current_v103_typed_predictive_flow_jepa.sh",
        "scripts/current_v104_sequential_bounded_flow_jepa.sh",
        "scripts/current_v105_horizon_addressed_flow_jepa.sh",
        "scripts/current_v106_interval_stage_flow_jepa.sh",
        "scripts/current_v107_complete_top_path_flow_jepa.sh",
        "scripts/run_v107_model_path_probe.sh",
        "scripts/current_v108_online_horizon_address_flow_jepa.sh",
        "scripts/run_v108_model_path_probe.sh",
        "scripts/current_v109_progressive_grounding_address_flow_jepa.sh",
        "scripts/run_v109_model_path_probe.sh",
        "scripts/current_v110_coordinate_typed_raw_jepa.sh",
        "scripts/current_v110_coordinate_typed_raw_jepa_smoke.sh",
        "scripts/run_v110_model_path_probe.sh",
        "scripts/current_v111_structured_ownership_bottleneck.sh",
        "scripts/current_v111_structured_ownership_bottleneck_smoke.sh",
        "scripts/run_v111_model_path_probe.sh",
        "scripts/current_v112_pre_value_owner_routing.sh",
        "scripts/current_v112_pre_value_owner_routing_smoke.sh",
        "scripts/run_v112_model_path_probe.sh",
        "scripts/current_v113_functional_mainline_routing.sh",
        "scripts/current_v113_functional_mainline_routing_smoke.sh",
        "scripts/run_v113_model_path_probe.sh",
        "scripts/current_v114_shared_factual_utility_precision.sh",
        "scripts/current_v114_shared_factual_utility_precision_smoke.sh",
        "scripts/current_v115_g_aligned_goal_phase_323.sh",
        "scripts/current_v115_g_aligned_goal_phase_323_smoke.sh",
        "scripts/current_v116_supervised_effect_mainline.sh",
        "scripts/current_v116_supervised_effect_mainline_smoke.sh",
        "scripts/run_v116_model_path_probe.sh",
        "scripts/current_v117_window_effect_intent_p2.sh",
        "scripts/current_v117_window_effect_intent_p2_smoke.sh",
        "scripts/run_v117_model_path_probe.sh",
        "scripts/current_v118_differential_intent_effect_323.sh",
        "scripts/current_v118_differential_intent_effect_323_smoke.sh",
        "scripts/run_v118_model_path_probe.sh",
        "scripts/current_grounded_intent_effect_323.sh",
        "scripts/current_grounded_intent_effect_323_smoke.sh",
        "scripts/current_object_intent_dynamics_323.sh",
        "scripts/current_object_intent_dynamics_323_smoke.sh",
        "scripts/run_grounded_intent_effect_323_model_path_probe.sh",
    )
    result: dict[str, str] = {}
    for relative in relative_paths:
        path = root / relative
        result[relative] = (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "missing"
        )
    return result


def _legacy_payload(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "clearvla-v35-world-checkpoint-v1":
        raise ValueError("--legacy-context-checkpoint must be a V35 world checkpoint")
    return payload


_STAGE1_DIRTY_ADAPTER_PREFIXES = (
    # V48 migration: keep the DiT trunk/readout warm start, but do not inherit
    # layer-contract interfaces trained under older consequence/action semantics.
    "planner.layer_contract_heads.",
    "planner.layer_consequence_cell.",
    "planner.layer_fm_probe.",
    "planner.event_probe.",
    "planner.motion_probe.",
)

_PARSEVAL_REPLACED_STAGE1_PREFIXES = (
    "planner.seed.noisy_physical_lift.grip_value.",
    "planner.seed.noisy_physical_lift.grip_delta.",
    "planner.seed.noisy_physical_lift.grip_extra.",
    "planner.latent_cvae_action_decoder.noisy_action_lift.",
    "planner.latent_cvae_action_decoder.posterior_action.",
    "planner.latent_cvae_action_decoder.velocity_head.",
)

_V74_TIME_CONTROLLER_STAGE1_PREFIXES = (
    "planner.latent_cvae_action_decoder.evidence_workspace.global_state_proj.",
    "planner.latent_cvae_action_decoder.evidence_workspace.controller.",
    "planner.latent_cvae_action_decoder.route_time_query.",
)


def _filter_stage1_state_dict(
    state: dict[str, torch.Tensor],
    *,
    reset_dirty_adapters: bool,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    if not reset_dirty_adapters:
        return state, []
    skipped = [key for key in state if key.startswith(_STAGE1_DIRTY_ADAPTER_PREFIXES)]
    if not skipped:
        return state, []
    skipped_set = set(skipped)
    return {key: value for key, value in state.items() if key not in skipped_set}, skipped


def _filter_shape_mismatched_state_dict(
    state: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[str]]:
    skipped = [
        key
        for key, value in state.items()
        if key in target and tuple(value.shape) != tuple(target[key].shape)
    ]
    if not skipped:
        return state, []
    skipped_set = set(skipped)
    return {key: value for key, value in state.items() if key not in skipped_set}, skipped


def _filter_parseval_replaced_state_dict(
    state: dict[str, torch.Tensor],
    *,
    enabled: bool,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    if not enabled:
        return state, []
    skipped = [key for key in state if key.startswith(_PARSEVAL_REPLACED_STAGE1_PREFIXES)]
    if not skipped:
        return state, []
    skipped_set = set(skipped)
    return {key: value for key, value in state.items() if key not in skipped_set}, skipped


def _filter_v74_time_controller_state_dict(
    state: dict[str, torch.Tensor],
    *,
    enabled: bool,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    if not enabled:
        return state, []
    skipped = [key for key in state if key.startswith(_V74_TIME_CONTROLLER_STAGE1_PREFIXES)]
    if not skipped:
        return state, []
    skipped_set = set(skipped)
    return {key: value for key, value in state.items() if key not in skipped_set}, skipped


def _validate_flow_jepa_stage1_checkpoint(
    payload: dict[str, Any],
    *,
    policy_config: V39PolicyConfig,
    goal_language_metadata: dict[str, Any] | None,
) -> None:
    """Reject an old/foreign Stage1 before a V95 policy run can consume it."""

    contract = payload.get("stage1_contract")
    if not isinstance(contract, dict) or contract.get("kind") != (
        "flow_dino_jepa_representation_v1"
    ):
        raise ValueError(
            "V95 policy requires a checkpoint produced by the new Flow-DINO/JEPA "
            "Stage1 experiment; an old best_contract.pt is not valid"
        )
    saved_trainer = payload.get("trainer_config") or {}
    saved_stage = str(saved_trainer.get("training_stage", "")).lower().replace("-", "_")
    if saved_stage not in {"contract", "stage1"}:
        raise ValueError(
            f"V95 Stage1 checkpoint has wrong training_stage={saved_stage!r}"
        )
    required_contract_flags = {
        "target_action_conditioned": False,
        "final_action_decoder_executed": False,
        "layer_contracts_executed": False,
    }
    for key, expected in required_contract_flags.items():
        if contract.get(key) is not expected:
            raise ValueError(
                f"V95 Stage1 checkpoint has an unsafe {key} contract: "
                f"checkpoint={contract.get(key)!r}, required={expected!r}"
            )
    saved_policy = payload.get("policy_config") or {}
    expected_fields = {
        "flow_jepa_enabled": int(policy_config.flow_jepa_enabled),
        "flow_jepa_grid_size": int(policy_config.flow_jepa_grid_size),
        "flow_jepa_feature_dim": int(policy_config.flow_jepa_feature_dim),
        "flow_jepa_flow_iters": int(policy_config.flow_jepa_flow_iters),
        "flow_jepa_corr_levels": int(policy_config.flow_jepa_corr_levels),
        "flow_jepa_corr_radius": int(policy_config.flow_jepa_corr_radius),
        "flow_jepa_late_bottleneck": int(policy_config.flow_jepa_late_bottleneck),
        "flow_jepa_dense_depth": int(policy_config.flow_jepa_dense_depth),
        "flow_jepa_fine_radius": int(policy_config.flow_jepa_fine_radius),
        "flow_jepa_reader_radius": int(policy_config.flow_jepa_reader_radius),
        "flow_jepa_reader_heads": int(policy_config.flow_jepa_reader_heads),
        "flow_jepa_raw_image_enabled": int(policy_config.flow_jepa_raw_image_enabled),
        "flow_jepa_role_hierarchy": int(policy_config.flow_jepa_role_hierarchy),
        "flow_jepa_raw_base_channels": int(policy_config.flow_jepa_raw_base_channels),
        "flow_jepa_raw_mid_radius": int(policy_config.flow_jepa_raw_mid_radius),
        "flow_jepa_raw_high_radius": int(policy_config.flow_jepa_raw_high_radius),
        "flow_jepa_raw_reader_radius": int(policy_config.flow_jepa_raw_reader_radius),
        "flow_jepa_raw_reader_heads": int(policy_config.flow_jepa_raw_reader_heads),
        "flow_jepa_raw_activation_checkpoint": int(
            policy_config.flow_jepa_raw_activation_checkpoint
        ),
        "flow_jepa_zero_flow_guard": int(policy_config.flow_jepa_zero_flow_guard),
        "flow_jepa_strict_role_visual_path": int(
            policy_config.flow_jepa_strict_role_visual_path
        ),
        "flow_jepa_complementary_raw_detail": int(
            policy_config.flow_jepa_complementary_raw_detail
        ),
        "flow_jepa_source_aligned_raw_fusion": int(
            policy_config.flow_jepa_source_aligned_raw_fusion
        ),
        "flow_jepa_teacher_balanced_target_mask": int(
            policy_config.flow_jepa_teacher_balanced_target_mask
        ),
        "flow_jepa_predictive_change_contract": int(
            policy_config.flow_jepa_predictive_change_contract
        ),
        "flow_jepa_grounding_blocks": int(policy_config.flow_jepa_grounding_blocks),
        "flow_jepa_world_blocks": int(policy_config.flow_jepa_world_blocks),
        "flow_jepa_policy_blocks": int(policy_config.flow_jepa_policy_blocks),
        "flow_jepa_policy_workspace_scale": float(
            policy_config.flow_jepa_policy_workspace_scale
        ),
        "flow_jepa_policy_workspace_fixed_fusion": int(
            policy_config.flow_jepa_policy_workspace_fixed_fusion
        ),
        "flow_jepa_world_anchor_write_only": int(
            policy_config.flow_jepa_world_anchor_write_only
        ),
        "flow_jepa_late_policy_detail": int(
            policy_config.flow_jepa_late_policy_detail
        ),
        "flow_jepa_late_policy_detail_scale": float(
            policy_config.flow_jepa_late_policy_detail_scale
        ),
        "flow_jepa_soft_address_lattice": int(
            policy_config.flow_jepa_soft_address_lattice
        ),
        "flow_jepa_horizon_soft_address": int(
            policy_config.flow_jepa_horizon_soft_address
        ),
        "flow_jepa_horizon_address_update_scale": float(
            policy_config.flow_jepa_horizon_address_update_scale
        ),
        "flow_jepa_variance_safe_routing": int(
            policy_config.flow_jepa_variance_safe_routing
        ),
        "flow_jepa_complete_numerical_contract": int(
            policy_config.flow_jepa_complete_numerical_contract
        ),
        "flow_jepa_routing_norm_floor": float(
            policy_config.flow_jepa_routing_norm_floor
        ),
        "flow_jepa_correlation_rms_floor": float(
            policy_config.flow_jepa_correlation_rms_floor
        ),
        "flow_jepa_visibility_transition_fraction": float(
            policy_config.flow_jepa_visibility_transition_fraction
        ),
        "flow_jepa_address_slots": int(policy_config.flow_jepa_address_slots),
        "flow_jepa_address_route_dim": int(
            policy_config.flow_jepa_address_route_dim
        ),
        "flow_jepa_address_query_chunk": int(
            policy_config.flow_jepa_address_query_chunk
        ),
        "flow_jepa_address_flow_prior_floor": float(
            policy_config.flow_jepa_address_flow_prior_floor
        ),
        "role_attnres_enabled": int(policy_config.role_attnres_enabled),
        "role_attnres_key_dim": int(policy_config.role_attnres_key_dim),
        "role_attnres_ground_to_world": int(
            policy_config.role_attnres_ground_to_world
        ),
        "role_attnres_world_to_policy": int(
            policy_config.role_attnres_world_to_policy
        ),
        "role_attnres_policy_to_mmdit": int(
            policy_config.role_attnres_policy_to_mmdit
        ),
        "role_attnres_ground_to_world_scale": float(
            policy_config.role_attnres_ground_to_world_scale
        ),
        "role_attnres_world_to_policy_scale": float(
            policy_config.role_attnres_world_to_policy_scale
        ),
        "role_attnres_policy_to_mmdit_scale": float(
            policy_config.role_attnres_policy_to_mmdit_scale
        ),
        "flow_jepa_policy_workspace_horizon_pool": int(
            policy_config.flow_jepa_policy_workspace_horizon_pool
        ),
        "flow_jepa_window_offsets": tuple(policy_config.flow_jepa_window_offsets),
        "flow_jepa_stage_offset": int(policy_config.flow_jepa_stage_offset),
        "action_history_enabled": int(policy_config.action_history_enabled),
        "action_history_condition_dropout": float(
            policy_config.action_history_condition_dropout
        ),
        "action_history_condition_exact_null": int(
            policy_config.action_history_condition_exact_null
        ),
        "action_history_proposal_detach": int(
            policy_config.action_history_proposal_detach
        ),
        "goal_conditioning_enabled": int(policy_config.goal_conditioning_enabled),
        "goal_token_count": int(policy_config.goal_token_count),
        "goal_language_dim": int(policy_config.goal_language_dim),
        "goal_condition_dropout": float(policy_config.goal_condition_dropout),
        "goal_condition_exact_null": int(policy_config.goal_condition_exact_null),
        "stateless_phase_enabled": int(policy_config.stateless_phase_enabled),
        "stateless_phase_count": int(policy_config.stateless_phase_count),
        "stateless_phase_query_scale": float(
            policy_config.stateless_phase_query_scale
        ),
    }
    # The scale is semantically inactive when the late-detail path is off.
    # Historical Stage1 checkpoints predate this field, so requiring the new
    # nonzero default (0.25) would reject otherwise compatible V95-V101
    # checkpoints even though their graph is unchanged.
    if not int(policy_config.flow_jepa_late_policy_detail):
        expected_fields.pop("flow_jepa_late_policy_detail_scale", None)
    if not int(policy_config.flow_jepa_soft_address_lattice):
        expected_fields.pop("flow_jepa_soft_address_lattice", None)
        for field in (
            "flow_jepa_address_slots",
            "flow_jepa_address_route_dim",
            "flow_jepa_address_query_chunk",
            "flow_jepa_address_flow_prior_floor",
        ):
            expected_fields.pop(field, None)
    if not int(policy_config.flow_jepa_horizon_soft_address):
        expected_fields.pop("flow_jepa_horizon_soft_address", None)
        expected_fields.pop(
            "flow_jepa_horizon_address_update_scale",
            None,
        )
    if not int(policy_config.flow_jepa_variance_safe_routing):
        expected_fields.pop("flow_jepa_variance_safe_routing", None)
        expected_fields.pop("flow_jepa_routing_norm_floor", None)
    if not int(policy_config.flow_jepa_complete_numerical_contract):
        expected_fields.pop("flow_jepa_complete_numerical_contract", None)
        expected_fields.pop("flow_jepa_correlation_rms_floor", None)
        expected_fields.pop(
            "flow_jepa_visibility_transition_fraction", None
        )
    if not int(policy_config.role_attnres_enabled):
        for field in (
            "role_attnres_enabled",
            "role_attnres_key_dim",
            "role_attnres_ground_to_world",
            "role_attnres_world_to_policy",
            "role_attnres_policy_to_mmdit",
            "role_attnres_ground_to_world_scale",
            "role_attnres_world_to_policy_scale",
            "role_attnres_policy_to_mmdit_scale",
        ):
            expected_fields.pop(field, None)
    if not int(policy_config.stateless_phase_enabled):
        expected_fields.pop("stateless_phase_count", None)
        expected_fields.pop("stateless_phase_query_scale", None)
    if not int(policy_config.flow_jepa_predictive_change_contract):
        expected_fields.pop("flow_jepa_predictive_change_contract", None)
    for field, current in expected_fields.items():
        saved: Any = saved_policy.get(field)
        if field == "flow_jepa_window_offsets":
            saved = tuple(saved or ())
        elif field == "action_history_proposal_detach":
            # Checkpoints written before this explicit switch used the
            # historical detached proposal path.
            saved = int(saved_policy.get(field, 1))
        elif isinstance(current, int):
            saved = int(saved or 0)
        elif isinstance(current, float):
            saved = float(saved or 0.0)
        if saved != current:
            raise ValueError(
                f"V95 Stage1 checkpoint mismatch for {field}: "
                f"checkpoint={saved!r}, current={current!r}"
            )
    saved_goal = (payload.get("context") or {}).get("goal_language") or {}
    current_goal = goal_language_metadata or {}
    if saved_goal.get("embedding_sha256") != current_goal.get("embedding_sha256"):
        raise ValueError(
            "V95 Stage1 language embedding hash does not match the policy run"
        )
    saved_source = (payload.get("context") or {}).get("source_fingerprint")
    current_source = _source_fingerprint()
    if saved_source != current_source:
        raise ValueError(
            "V95 Stage1 source fingerprint differs from the current policy source; "
            "rerun Stage1 with this implementation"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train V40 layer-role causal/latent temporal policy."
    )
    add_data_args(parser, default_resize=(336, 336))
    parser.add_argument(
        "--legacy-context-checkpoint",
        type=Path,
        default=None,
        help="Optional migration source for splits/normalizers only; not a model dependency.",
    )
    parser.add_argument("--normalizer", choices=["zscore", "limits", "identity"], default="zscore")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--stage1-checkpoint",
        type=Path,
        default=None,
        help="Load a V39 contract-stage checkpoint as model initialization before policy-stage finetuning.",
    )
    parser.add_argument(
        "--stage1-initialization-enabled",
        type=int,
        choices=[0, 1],
        default=1,
        help=(
            "Honor --stage1-checkpoint. Set to 0 for a true single-stage run even when "
            "an inherited launcher supplies a historical checkpoint path."
        ),
    )
    parser.add_argument(
        "--stage1-reset-dirty-adapters",
        type=int,
        default=0,
        help="Skip old layer adapter/consequence interface weights when migrating from a pre-fix stage1 checkpoint.",
    )
    parser.add_argument(
        "--require-flow-jepa-stage1-checkpoint",
        type=int,
        choices=[0, 1],
        default=0,
        help="Require --stage1-checkpoint to be produced by the new V95 representation Stage1.",
    )
    parser.add_argument(
        "--condition-mode",
        choices=["dinov2", "dinov2-cache", "debug-dense"],
        default="dinov2-cache",
    )
    parser.add_argument("--dinov2-model", default="facebook/dinov2-base")
    parser.add_argument("--dinov2-local-files-only", action="store_true")
    parser.add_argument("--dinov2-token-cache-dir", type=Path, default=None)
    parser.add_argument(
        "--prefetch-dinov2-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For dinov2-cache mode, load current and compact future DINO tokens in DataLoader workers instead of the main training loop.",
    )
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")

    parser.add_argument("--world-horizon", type=int, default=48)
    parser.add_argument("--policy-horizon", type=int, default=24)
    parser.add_argument("--segment-length", type=int, default=4)
    parser.add_argument("--history-offsets", type=_parse_offsets, default=(-8, -4, 0))
    parser.add_argument("--executed-action-offsets", type=_parse_offsets, default=(-8, -4, -1))
    parser.add_argument("--action-history-enabled", type=int, choices=[0, 1], default=0)
    parser.add_argument("--action-history-recent-tokens", type=int, default=4)
    parser.add_argument("--action-history-summary-tokens", type=int, default=3)
    parser.add_argument("--action-history-condition-dropout", type=float, default=0.0)
    parser.add_argument(
        "--action-history-condition-exact-null", type=int, choices=[0, 1], default=0
    )
    parser.add_argument(
        "--action-history-proposal-detach",
        type=int,
        choices=[0, 1],
        default=1,
        help=(
            "Detach history-derived proposal tokens before the policy. "
            "Use 0 for end-to-end action-gradient conditioning; 1 reproduces "
            "the historical auxiliary-only proposal path."
        ),
    )
    parser.add_argument("--goal-conditioning-enabled", type=int, choices=[0, 1], default=0)
    parser.add_argument("--goal-token-count", type=int, default=4)
    parser.add_argument("--goal-resampler-depth", type=int, default=2)
    parser.add_argument("--goal-language-max-tokens", type=int, default=32)
    parser.add_argument("--goal-condition-dropout", type=float, default=0.0)
    parser.add_argument(
        "--goal-condition-exact-null", type=int, choices=[0, 1], default=0
    )
    parser.add_argument(
        "--stateless-phase-enabled", type=int, choices=[0, 1], default=0
    )
    parser.add_argument("--stateless-phase-count", type=int, default=4)
    parser.add_argument("--stateless-phase-query-scale", type=float, default=0.10)
    parser.add_argument(
        "--t5-condition-path",
        "--goal-language-condition-path",
        dest="t5_condition_path",
        type=Path,
        default=None,
        help=(
            "Precomputed T5 condition .pt/.pth. Accepts [L,D], [1,L,D], or "
            "a dict containing embeddings/tokens and an optional attention mask."
        ),
    )
    parser.add_argument("--target-history-offsets", type=_parse_offsets, default=(-8, -4, 0))
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--information-balanced-sampling", type=int, choices=[0, 1], default=0
    )
    parser.add_argument("--information-uniform-fraction", type=float, default=0.50)
    parser.add_argument("--information-event-fraction", type=float, default=0.125)
    parser.add_argument("--information-motion-quantile", type=float, default=0.70)

    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--depth", type=int, default=8, help="shared temporal canvas depth")
    parser.add_argument(
        "--midcut-layer",
        type=int,
        default=3,
        help="Expose Z_mid after this many DiT blocks for the simple latent-contract heads.",
    )
    parser.add_argument("--midcut-future-gain-init", type=float, default=0.10)
    parser.add_argument(
        "--layer-contract-adapters",
        type=int,
        default=1,
        help="V40: enable layer-role contract adapters after every DiT block.",
    )
    parser.add_argument("--layer-contract-adapter-dim", type=int, default=128)
    parser.add_argument("--layer-contract-grad-scale", type=float, default=1.0)
    parser.add_argument("--layer-contract-residual-scale", type=float, default=0.50)
    parser.add_argument(
        "--layer-shared-fm-probe",
        type=int,
        default=0,
        help="Deprecated in V40 by default; keep disabled so action_pred does not become a side branch.",
    )
    parser.add_argument("--layer-fm-probe-hidden", type=int, default=256)
    parser.add_argument(
        "--layer-recurrent-consequence",
        type=int,
        default=1,
        help="V40: enable the layer-role causal/effect branch.",
    )
    parser.add_argument("--layer-consequence-steps", type=int, default=6)
    parser.add_argument("--layer-consequence-hidden", type=int, default=256)
    parser.add_argument("--layer-consequence-delta-scale", type=float, default=1.0)
    parser.add_argument("--layer-consequence-initial-gain", type=float, default=0.10)
    parser.add_argument(
        "--layer-causal-feedback-depth",
        type=int,
        default=0,
        help="V40.1: optional inner interaction blocks; default 0 because the unified intervention head already encodes state-action jointly.",
    )
    parser.add_argument(
        "--layer-causal-memory-tokens",
        type=int,
        default=4,
        help="V40.1: compact memory/context tokens for unified intervention latent.",
    )
    parser.add_argument(
        "--layer-low-causal-weight",
        type=float,
        default=1.0,
        help="V40: causal gain at the shallowest layer.",
    )
    parser.add_argument(
        "--layer-high-causal-weight",
        type=float,
        default=1.0,
        help="V40.1: retained as diagnostic; unified intervention head is not mixed with a separate latent head.",
    )
    parser.add_argument(
        "--layer-low-latent-weight",
        type=float,
        default=1.0,
        help="V40.1: retained as diagnostic; unified intervention head is not mixed with a separate latent head.",
    )
    parser.add_argument(
        "--layer-high-latent-weight",
        type=float,
        default=1.0,
        help="V40: latent gain at the deepest layer.",
    )
    parser.add_argument(
        "--layer-causal-event-from-effect",
        type=int,
        default=1,
        help="V40: read event logits from causal effect tokens in layer contracts.",
    )
    parser.add_argument(
        "--layer-state-counterfactual",
        type=int,
        default=0,
        help="Experimental non-strict state/frame shuffle diagnostic; disabled by default and excluded from the recommended training path.",
    )
    parser.add_argument(
        "--action-consequence-self-condition",
        type=int,
        default=0,
        help="Use a no-grad deployable clean-action preview as the layer consequence action input.",
    )
    parser.add_argument(
        "--layer-zero-base-diagnostic",
        type=int,
        default=1,
        help="Log consequence-output shift when layer rollout tokens are zeroed; diagnostic only, no loss.",
    )
    parser.add_argument("--proposal-depth", type=int, default=2)
    parser.add_argument("--proposal-dropout", type=float, default=0.25)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--event-tokens", type=int, default=3)
    parser.add_argument("--canvas-registers", type=int, default=12)
    parser.add_argument("--future-anchors", type=int, default=6)
    parser.add_argument("--future-grid-size", type=int, default=4)
    parser.add_argument("--action-basis-tokens", type=int, default=4)
    parser.add_argument("--rollout-tail-start-step", type=int, default=8)
    parser.add_argument("--rollout-tail-full-step", type=int, default=13)
    parser.add_argument("--controlled-delta-rank", type=int, default=8)
    parser.add_argument("--base-effect-hidden", type=int, default=128)
    parser.add_argument(
        "--controlled-base-mode",
        choices=("learned", "fixed_zero"),
        default="fixed_zero",
        help="Use an identifiable no-change rollout base; learned is retained only for historical checkpoint evaluation.",
    )
    parser.add_argument("--latent-action-tokens", type=int, default=8)
    parser.add_argument("--neutral-action-tokens", type=int, default=4)
    parser.add_argument("--controlled-delta-dropout", type=float, default=0.0)
    parser.add_argument("--role-dropout", type=float, default=0.10)
    parser.add_argument("--visual-memory-dropout", type=float, default=0.0)
    parser.add_argument("--canvas-dropout", type=float, default=0.0)
    parser.add_argument("--flow-jepa-enabled", type=int, choices=[0, 1], default=0)
    parser.add_argument("--flow-jepa-grid-size", type=int, default=8)
    parser.add_argument("--flow-jepa-feature-dim", type=int, default=96)
    parser.add_argument("--flow-jepa-flow-iters", type=int, default=3)
    parser.add_argument("--flow-jepa-corr-levels", type=int, default=3)
    parser.add_argument("--flow-jepa-corr-radius", type=int, default=2)
    parser.add_argument("--flow-jepa-mask-ratio", type=float, default=0.375)
    parser.add_argument("--flow-jepa-mask-block-size", type=int, default=2)
    parser.add_argument("--flow-jepa-motion-mask-fraction", type=float, default=0.60)
    parser.add_argument(
        "--flow-jepa-teacher-balanced-target-mask", type=int, choices=[0, 1], default=0
    )
    parser.add_argument("--flow-jepa-teacher-mask-past-fraction", type=float, default=0.25)
    parser.add_argument("--flow-jepa-teacher-mask-change-fraction", type=float, default=0.50)
    parser.add_argument(
        "--flow-jepa-predictive-change-contract",
        type=int,
        choices=[0, 1],
        default=0,
    )
    parser.add_argument("--flow-jepa-uncertainty-floor", type=float, default=0.03)
    parser.add_argument(
        "--flow-jepa-late-bottleneck", type=int, choices=[0, 1], default=0
    )
    parser.add_argument("--flow-jepa-dense-depth", type=int, default=2)
    parser.add_argument("--flow-jepa-fine-radius", type=int, default=2)
    parser.add_argument("--flow-jepa-reader-radius", type=int, default=1)
    parser.add_argument("--flow-jepa-reader-heads", type=int, default=2)
    parser.add_argument("--flow-jepa-raw-image-enabled", type=int, choices=[0, 1], default=0)
    parser.add_argument("--flow-jepa-role-hierarchy", type=int, choices=[0, 1], default=0)
    parser.add_argument("--flow-jepa-raw-base-channels", type=int, default=32)
    parser.add_argument("--flow-jepa-raw-mid-radius", type=int, default=2)
    parser.add_argument("--flow-jepa-raw-high-radius", type=int, default=1)
    parser.add_argument("--flow-jepa-raw-reader-radius", type=int, default=3)
    parser.add_argument("--flow-jepa-raw-reader-heads", type=int, default=4)
    parser.add_argument(
        "--flow-jepa-raw-activation-checkpoint", type=int, choices=[0, 1], default=1
    )
    parser.add_argument("--flow-jepa-zero-flow-guard", type=int, choices=[0, 1], default=0)
    parser.add_argument(
        "--flow-jepa-strict-role-visual-path", type=int, choices=[0, 1], default=0
    )
    parser.add_argument(
        "--flow-jepa-complementary-raw-detail", type=int, choices=[0, 1], default=0
    )
    parser.add_argument(
        "--flow-jepa-source-aligned-raw-fusion", type=int, choices=[0, 1], default=0
    )
    parser.add_argument("--flow-jepa-grounding-blocks", type=int, default=3)
    parser.add_argument("--flow-jepa-world-blocks", type=int, default=3)
    parser.add_argument("--flow-jepa-policy-blocks", type=int, default=2)
    parser.add_argument("--flow-jepa-policy-workspace-scale", type=float, default=0.10)
    parser.add_argument(
        "--flow-jepa-policy-workspace-fixed-fusion", type=int, choices=[0, 1], default=0
    )
    parser.add_argument(
        "--flow-jepa-world-anchor-write-only", type=int, choices=[0, 1], default=0
    )
    parser.add_argument(
        "--flow-jepa-late-policy-detail", type=int, choices=[0, 1], default=0
    )
    parser.add_argument("--flow-jepa-late-policy-detail-scale", type=float, default=0.25)
    parser.add_argument(
        "--flow-jepa-soft-address-lattice", type=int, choices=[0, 1], default=0
    )
    parser.add_argument("--flow-jepa-address-slots", type=int, default=4)
    parser.add_argument("--flow-jepa-address-route-dim", type=int, default=32)
    parser.add_argument("--flow-jepa-address-query-chunk", type=int, default=4)
    parser.add_argument(
        "--flow-jepa-policy-multi-glimpse-address",
        type=int,
        choices=[0, 1],
        default=0,
    )
    parser.add_argument("--flow-jepa-address-flow-prior-floor", type=float, default=0.0)
    parser.add_argument(
        "--flow-jepa-bounded-flow-coordinates",
        type=int,
        choices=[0, 1],
        default=0,
    )
    parser.add_argument(
        "--flow-jepa-sequential-horizon-memory",
        type=int,
        choices=[0, 1],
        default=0,
    )
    parser.add_argument(
        "--flow-jepa-horizon-soft-address",
        type=int,
        choices=[0, 1],
        default=0,
    )
    parser.add_argument(
        "--flow-jepa-horizon-address-update-scale",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--flow-jepa-horizon-cell-fine-address",
        type=int,
        choices=[0, 1],
        default=0,
    )
    parser.add_argument(
        "--flow-jepa-online-horizon-address",
        type=int,
        choices=[0, 1],
        default=0,
    )
    parser.add_argument(
        "--flow-jepa-progressive-grounding-address",
        type=int,
        choices=[0, 1],
        default=0,
    )
    parser.add_argument(
        "--flow-jepa-coordinate-typed-raw-detail",
        type=int,
        choices=[0, 1],
        default=0,
    )
    parser.add_argument(
        "--flow-jepa-structured-ownership-bottleneck",
        type=int,
        choices=[0, 1],
        default=0,
    )
    parser.add_argument(
        "--flow-jepa-pre-value-owner-routing",
        type=int,
        choices=[0, 1],
        default=0,
    )
    parser.add_argument(
        "--flow-jepa-pre-value-owner-update-scale",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--flow-jepa-functional-mainline-routing",
        type=int,
        choices=[0, 1],
        default=0,
    )
    parser.add_argument(
        "--flow-jepa-utility-precision-mainline",
        type=int,
        choices=[0, 1],
        default=0,
    )
    parser.add_argument(
        "--flow-jepa-action-free-world-factual",
        type=int,
        choices=[0, 1],
        default=0,
    )
    parser.add_argument(
        "--flow-jepa-shared-factual-glimpse-bank",
        type=int,
        choices=[0, 1],
        default=0,
    )
    parser.add_argument(
        "--flow-jepa-g-aligned-future-effect",
        type=int,
        choices=[0, 1],
        default=0,
    )
    parser.add_argument(
        "--flow-jepa-teacher-g-ema-decay",
        type=float,
        default=0.995,
    )
    parser.add_argument(
        "--flow-jepa-stateless-goal-phase-machine",
        type=int,
        choices=[0, 1],
        default=0,
    )
    parser.add_argument(
        "--flow-jepa-top-role-schedule",
        choices=["3-3-2", "3-2-3"],
        default="3-3-2",
    )
    parser.add_argument(
        "--flow-jepa-policy-plan-compiler",
        type=int,
        choices=[0, 1],
        default=0,
    )
    parser.add_argument(
        "--flow-jepa-supervised-effect-mainline",
        type=int,
        choices=[0, 1],
        default=0,
    )
    parser.add_argument(
        "--flow-jepa-stateless-intent-controller",
        type=int,
        choices=[0, 1],
        default=0,
    )
    parser.add_argument(
        "--flow-jepa-window-effect-bank",
        type=int,
        choices=[0, 1],
        default=0,
    )
    parser.add_argument("--flow-jepa-future-slots", type=int, default=4)
    parser.add_argument(
        "--flow-jepa-effect-read-in-p2",
        type=int,
        choices=[0, 1],
        default=0,
    )
    parser.add_argument(
        "--flow-jepa-differential-intent-effect-mainline",
        type=int,
        choices=[0, 1],
        default=0,
    )
    parser.add_argument(
        "--flow-jepa-grounded-intent-effect-mainline",
        type=int,
        choices=[0, 1],
        default=0,
    )
    parser.add_argument(
        "--flow-jepa-object-intent-dynamics-mainline",
        type=int,
        choices=[0, 1],
        default=0,
    )
    parser.add_argument(
        "--flow-matching-time-distribution",
        choices=["uniform", "beta_1_5_1"],
        default="uniform",
    )
    parser.add_argument(
        "--flow-jepa-address-query-batch-budget",
        type=int,
        default=32,
    )
    parser.add_argument("--flow-jepa-microgrid-tile", type=int, default=3)
    parser.add_argument(
        "--flow-jepa-p1-mixed-precision",
        type=int,
        choices=[0, 1],
        default=0,
    )
    parser.add_argument(
        "--flow-jepa-checkpoint-min-batch",
        type=int,
        default=4,
    )
    parser.add_argument("--flow-jepa-raw-micro-grid", type=int, default=3)
    parser.add_argument(
        "--flow-jepa-variance-safe-routing",
        type=int,
        choices=[0, 1],
        default=0,
    )
    parser.add_argument(
        "--flow-jepa-complete-numerical-contract",
        type=int,
        choices=[0, 1],
        default=0,
    )
    parser.add_argument("--flow-jepa-routing-norm-floor", type=float, default=0.25)
    parser.add_argument(
        "--flow-jepa-correlation-rms-floor", type=float, default=0.10
    )
    parser.add_argument(
        "--flow-jepa-visibility-transition-fraction",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--flow-jepa-horizon-value-max-rms", type=float, default=0.50
    )
    parser.add_argument(
        "--flow-jepa-interval-stage-delta",
        type=int,
        choices=[0, 1],
        default=0,
    )
    parser.add_argument(
        "--flow-jepa-interval-boundaries",
        type=_parse_offsets,
        default=(4, 8, 16, 32, 48),
    )
    parser.add_argument(
        "--flow-jepa-interval-support-offsets",
        type=_parse_offsets,
        default=tuple(range(4, 49, 4)),
    )
    parser.add_argument(
        "--flow-jepa-interval-stage-update-scale",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--flow-jepa-interval-stage-typed-value",
        type=int,
        choices=[0, 1],
        default=0,
    )
    parser.add_argument("--role-attnres-enabled", type=int, choices=[0, 1], default=0)
    parser.add_argument("--role-attnres-key-dim", type=int, default=32)
    parser.add_argument(
        "--role-attnres-ground-to-world", type=int, choices=[0, 1], default=0
    )
    parser.add_argument(
        "--role-attnres-world-to-policy", type=int, choices=[0, 1], default=0
    )
    parser.add_argument(
        "--role-attnres-policy-to-mmdit", type=int, choices=[0, 1], default=0
    )
    parser.add_argument("--role-attnres-ground-to-world-scale", type=float, default=0.10)
    parser.add_argument("--role-attnres-world-to-policy-scale", type=float, default=0.10)
    parser.add_argument("--role-attnres-policy-to-mmdit-scale", type=float, default=0.25)
    parser.add_argument(
        "--role-residual-amplitude-contract",
        type=int,
        choices=[0, 1],
        default=0,
    )
    parser.add_argument("--role-residual-max-update-rms", type=float, default=0.50)
    parser.add_argument("--role-attnres-max-value-rms", type=float, default=1.00)
    parser.add_argument(
        "--role-residual-contract-after-gate",
        type=int,
        choices=[0, 1],
        default=0,
    )
    parser.add_argument(
        "--flow-jepa-policy-workspace-horizon-pool",
        type=int,
        choices=[0, 1],
        default=0,
    )
    parser.add_argument(
        "--flow-jepa-window-offsets",
        type=_parse_offsets,
        default=(4, 12, 24),
        help="Sparse patch-level JEPA horizons, all inside policy_horizon.",
    )
    parser.add_argument(
        "--flow-jepa-stage-offset",
        type=int,
        default=48,
        help="Far global DINO-delta horizon used only for stage supervision.",
    )
    parser.add_argument(
        "--flow-jepa-directed-canvas-attention", type=int, choices=[0, 1], default=1
    )
    parser.add_argument("--inference-steps", type=int, default=5)
    parser.add_argument("--gripper-dim-index", type=int, default=-1)
    parser.add_argument("--first-execution-steps", type=int, default=4)
    parser.add_argument("--mid-execution-steps", type=int, default=8)
    parser.add_argument("--physical-decode-delta-blend", type=float, default=0.25)
    parser.add_argument(
        "--arm-flow-mode",
        choices=("legacy_independent", "manifold_native"),
        default="legacy_independent",
    )
    parser.add_argument("--arm-noise-temporal-rho", type=float, default=0.0)
    parser.add_argument(
        "--arm-source-mode",
        choices=("ar1", "boundary_multiscale"),
        default="ar1",
        help=(
            "Native arm bridge source. boundary_multiscale is conditioned only "
            "on action_state and mixes trace-normalized position/velocity/acceleration operators."
        ),
    )
    parser.add_argument("--arm-source-scale", type=float, default=1.0)
    parser.add_argument("--arm-source-innovation-weight", type=float, default=0.50)
    parser.add_argument("--arm-source-velocity-weight", type=float, default=0.35)
    parser.add_argument("--arm-source-acceleration-weight", type=float, default=0.15)
    parser.add_argument("--gripper-field-dim", type=int, default=12)
    parser.add_argument(
        "--gripper-field-mode",
        choices=("legacy_handcrafted", "parseval_temporal"),
        default="legacy_handcrafted",
    )
    parser.add_argument(
        "--final-action-decoder",
        choices=[
            "legacy",
            "residual_action_flow",
            "layered_residual_action_flow",
            "latent_main_action",
            "latent_cvae_action",
            "adaptive_recurrent_cvae_action",
            "hierarchical_mmdit_action",
            "evidence_latent_mmdit_action",
        ],
        default="legacy",
    )
    parser.add_argument("--action-flow-residual-depth", type=int, default=2)
    parser.add_argument("--action-flow-residual-high-slots", type=int, default=4)
    parser.add_argument("--action-flow-residual-max-scale", type=float, default=0.20)
    parser.add_argument("--action-flow-residual-visual-memory", type=int, default=1)
    parser.add_argument("--action-flow-residual-context-memory", type=int, default=1)
    parser.add_argument("--action-flow-residual-transition-memory", type=int, default=1)
    parser.add_argument("--action-flow-residual-layer-memory", type=int, default=1)
    parser.add_argument(
        "--action-flow-residual-layer-pair-schedule", type=str, default="0:1,1:3,3:5,5:7"
    )
    parser.add_argument("--action-flow-residual-layer-detach", type=int, default=1)
    parser.add_argument("--action-flow-residual-stage-router", type=int, default=1)
    parser.add_argument("--action-flow-residual-anchor-memory", type=int, default=1)
    parser.add_argument("--latent-action-decoder-depth", type=int, default=8)
    parser.add_argument("--latent-action-high-slots", type=int, default=4)
    parser.add_argument(
        "--latent-action-layer-schedule", type=str, default="0:1,1:2,2:3,3:4,4:5,5:6,6:7,7:7"
    )
    parser.add_argument("--latent-action-visual-memory", type=int, default=0)
    parser.add_argument("--latent-action-context-memory", type=int, default=0)
    parser.add_argument("--latent-action-transition-memory", type=int, default=1)
    parser.add_argument("--latent-action-layer-memory", type=int, default=1)
    parser.add_argument("--latent-action-anchor-memory", type=int, default=1)
    parser.add_argument("--latent-action-stage-router", type=int, default=0)
    parser.add_argument("--latent-action-layer-detach", type=int, default=0)
    parser.add_argument("--latent-action-event-gripper-gate", type=int, default=1)
    parser.add_argument("--latent-action-temporal-depth", type=int, default=0)
    parser.add_argument("--latent-action-near-steps", type=int, default=4)
    parser.add_argument("--latent-action-mid-steps", type=int, default=8)
    parser.add_argument("--latent-action-near-depth", type=int, default=2)
    parser.add_argument("--latent-action-mid-depth", type=int, default=4)
    parser.add_argument("--latent-cvae-z-dim", type=int, default=64)
    parser.add_argument("--latent-cvae-decoder-depth", type=int, default=3)
    parser.add_argument("--latent-cvae-ffn-expansion", type=float, default=2.0)
    parser.add_argument("--latent-cvae-layer-memory", type=int, default=1)
    parser.add_argument("--latent-cvae-transition-memory", type=int, default=1)
    parser.add_argument("--latent-cvae-transition-detach", type=int, default=0)
    parser.add_argument("--latent-cvae-context-memory", type=int, default=0)
    parser.add_argument("--latent-cvae-visual-memory", type=int, default=0)
    parser.add_argument("--latent-cvae-layer-detach", type=int, default=0)
    parser.add_argument("--latent-cvae-layer-grad-scale", type=float, default=1.0)
    parser.add_argument(
        "--latent-cvae-condition-source-norm",
        type=int,
        default=1,
        help="Normalize each CVAE condition source before fusion.",
    )
    parser.add_argument(
        "--latent-cvae-bounded-consequence-fusion",
        type=int,
        default=1,
        help="Fuse world and action-conditioned summaries separately so routing cannot bypass consequence scaling.",
    )
    parser.add_argument(
        "--latent-cvae-consequence-scale-init",
        type=float,
        default=0.10,
        help="Initial strength of action-conditioned layer-contract summaries.",
    )
    parser.add_argument(
        "--latent-cvae-consequence-scale-max",
        type=float,
        default=0.50,
        help="Upper bound for action-conditioned layer-contract summary strength.",
    )
    parser.add_argument("--latent-cvae-event-gripper-gate", type=int, default=1)
    parser.add_argument("--latent-cvae-inference-sample", type=int, default=0)
    parser.add_argument(
        "--latent-cvae-variational",
        type=int,
        default=1,
        help="CR1/B1: 0 bypasses posterior/KL/aux-decode training scaffold while keeping the deterministic prior-mean deploy mapping bit-identical.",
    )
    parser.add_argument("--latent-cvae-output-init-std", type=float, default=1e-3)
    parser.add_argument("--latent-cvae-mu-bound", type=float, default=1.5)
    parser.add_argument("--latent-cvae-min-std", type=float, default=0.5)
    parser.add_argument("--latent-cvae-causal-attention", type=int, default=1)
    parser.add_argument(
        "--latent-cvae-noisy-gate",
        type=int,
        default=0,
        help="t-gate the direct noisy-action branch of the CVAE decoder.",
    )
    parser.add_argument("--latent-cvae-noisy-gate-min", type=float, default=0.05)
    parser.add_argument("--latent-cvae-noisy-gate-power", type=float, default=1.5)
    parser.add_argument(
        "--latent-cvae-layer-scan",
        type=int,
        default=0,
        help="Use a recurrent scan over ordered layer summaries as the CVAE condition.",
    )
    parser.add_argument(
        "--latent-cvae-layer-scan-alpha",
        type=float,
        default=0.2,
        help="Residual weight for the flat lateral layer condition when layer scan is enabled.",
    )
    parser.add_argument(
        "--latent-cvae-mmdit-decoder",
        type=int,
        default=0,
        help="Use the MMDiT-lite CVAE action-condition token decoder.",
    )
    parser.add_argument("--latent-cvae-mmdit-depth", type=int, default=3)
    parser.add_argument(
        "--latent-cvae-mmdit-cond-update",
        type=int,
        default=0,
        help="Allow condition tokens to update inside the MMDiT-lite decoder.",
    )
    parser.add_argument(
        "--latent-cvae-mmdit-noisy-causal",
        type=int,
        default=1,
        help="Mask future noisy-action condition tokens for action queries.",
    )
    parser.add_argument(
        "--latent-cvae-mmdit-noisy-logit-gate",
        type=int,
        default=0,
        help="V70: LayerNorm noisy condition tokens and move the t-gate to an additive log g(t) attention-logit bias (closes the value-volume degree of freedom).",
    )
    parser.add_argument(
        "--latent-cvae-mmdit-residual-scale-max",
        type=float,
        default=0.25,
        help="V91.1: shared residual budget for split evidence/noisy readers.",
    )
    parser.add_argument(
        "--latent-cvae-mmdit-source-route-delta-max",
        type=float,
        default=1.0,
        help="Deprecated V91 two-source route bound; retained for launch-script compatibility.",
    )
    parser.add_argument(
        "--latent-cvae-mmdit-noisy-correction-min",
        type=float,
        default=0.05,
        help="Minimum smooth noisy residual-correction budget.",
    )
    parser.add_argument(
        "--latent-cvae-mmdit-noisy-correction-max",
        type=float,
        default=0.75,
        help="Maximum smooth noisy residual-correction budget.",
    )
    parser.add_argument(
        "--latent-cvae-mmdit-noisy-correction-power",
        type=float,
        default=1.5,
        help="Flow-time exponent for the noisy residual-correction prior.",
    )
    parser.add_argument(
        "--latent-cvae-mmdit-noisy-correction-logit-delta",
        type=float,
        default=1.0,
        help="Maximum controller displacement of correction budget in logit space.",
    )
    parser.add_argument(
        "--latent-cvae-mmdit-controller-modulation-scale",
        type=float,
        default=0.25,
        help="Bound on controller modulation of noisy-reader query selection.",
    )
    parser.add_argument(
        "--latent-cvae-mmdit-evidence-scale",
        type=float,
        default=1.0,
        help="Diagnostic source ablation scale for evidence reader; keep at 1 for training.",
    )
    parser.add_argument(
        "--latent-cvae-mmdit-noisy-scale",
        type=float,
        default=1.0,
        help="Diagnostic source ablation scale for noisy reader; keep at 1 for training.",
    )
    parser.add_argument("--latent-cvae-mmdit-operator-capacity", type=int, default=0)
    parser.add_argument("--latent-cvae-mmdit-operator-rank", type=int, default=32)
    parser.add_argument("--latent-cvae-mmdit-operator-groups", type=int, default=32)
    parser.add_argument(
        "--latent-cvae-mmdit-operator-depth-logit-init", type=float, default=4.0
    )
    parser.add_argument("--latent-cvae-mmdit-execution-controller", type=int, default=0)
    parser.add_argument(
        "--latent-cvae-mmdit-dynamic-block-route",
        type=int,
        default=0,
        help="Allow the native controller to choose any remaining host block or terminal identity.",
    )
    parser.add_argument("--latent-cvae-mmdit-control-tokens", type=int, default=8)
    parser.add_argument("--latent-cvae-mmdit-controller-depth", type=int, default=2)
    parser.add_argument("--latent-cvae-mmdit-controller-heads", type=int, default=8)
    parser.add_argument(
        "--latent-cvae-mmdit-controller-ffn-expansion", type=float, default=2.0
    )
    parser.add_argument("--latent-cvae-mmdit-max-dwell", type=int, default=2)
    parser.add_argument(
        "--latent-cvae-mmdit-dwell-mode",
        choices=("fixed", "random", "learned_shadow", "learned"),
        default="fixed",
    )
    parser.add_argument(
        "--latent-cvae-mmdit-execution-soft-temperature",
        type=float,
        default=1.0,
        help="Temperature for the attached learned-training execution mixture.",
    )
    parser.add_argument(
        "--latent-cvae-mmdit-identity-candidate",
        type=int,
        default=1,
        help="Expose terminal identity as a real execution candidate.",
    )
    parser.add_argument(
        "--latent-cvae-mmdit-terminal-prior-weight",
        type=float,
        default=0.25,
        help="Relative softmax prior for the high-consequence terminal candidate.",
    )
    parser.add_argument(
        "--latent-cvae-mmdit-execution-eval-policy",
        choices=("soft", "hard", "neutral"),
        default="soft",
        help="Evaluation execution contract; hard/neutral are explicit ablations.",
    )
    parser.add_argument("--latent-cvae-mmdit-execution-warmup-steps", type=int, default=200)
    parser.add_argument(
        "--latent-cvae-mmdit-execution-transition-steps", type=int, default=1000
    )
    parser.add_argument(
        "--latent-cvae-progress-action-isolation",
        type=int,
        default=0,
        help="V72: cut the action->progress->workspace echo; the per-step progress update no longer receives the raw action summary (zeros fed, parameter shapes unchanged).",
    )
    parser.add_argument(
        "--latent-cvae-horizon-tokens",
        type=int,
        default=24,
        help="Number of z-conditioned evidence workspace tokens supplied to MMDiT.",
    )
    parser.add_argument(
        "--latent-cvae-workspace-noisy-query",
        type=int,
        default=0,
        help="Condition workspace evidence queries on the current noisy flow state.",
    )
    parser.add_argument(
        "--latent-cvae-workspace-trajectory-source",
        type=int,
        default=1,
        help="Expose full-resolution trajectory canvas tokens as workspace values; set 0 for the x_t-echo ablation.",
    )
    parser.add_argument(
        "--latent-cvae-workspace-global-sources",
        type=int,
        default=1,
        help="Expose scan/lateral global condition summaries as workspace values; set 0 to keep them only in cond/z.",
    )
    parser.add_argument(
        "--latent-cvae-workspace-layer-source",
        type=int,
        default=1,
        help="Expose full layer_stack as a static workspace value; set 0 to force layer information through routed_layer/capsules.",
    )
    parser.add_argument(
        "--latent-cvae-workspace-progress-value",
        type=int,
        default=1,
        help="Expose progress as a workspace value; set 0 to use progress only as workspace step/query state.",
    )
    parser.add_argument(
        "--latent-cvae-workspace-time-state",
        type=int,
        default=0,
        help="Inject the existing z+time primary condition into workspace slots as explicit state.",
    )
    parser.add_argument(
        "--latent-cvae-workspace-slot-time-state",
        type=int,
        default=1,
        help="Make workspace time-state slot-aware instead of broadcasting one identical z+time vector to every workspace token.",
    )
    parser.add_argument(
        "--latent-cvae-workspace-slot-time-scale",
        type=float,
        default=0.10,
        help="Scale of the slot-specific component used by --latent-cvae-workspace-slot-time-state.",
    )
    parser.add_argument(
        "--latent-cvae-workspace-controller",
        type=int,
        default=0,
        help="V74B: use the central workspace controller for role bias, capacity, and query modulation.",
    )
    parser.add_argument(
        "--latent-cvae-hierarchical-workspace",
        type=int,
        default=0,
        help="V75: use temporary low evidence reads plus persistent role-separated stage memory.",
    )
    parser.add_argument(
        "--latent-cvae-stage-slots",
        type=int,
        default=6,
        help="Number of persistent stage-memory slots in the hierarchical workspace.",
    )
    parser.add_argument(
        "--latent-cvae-stage-promote-scale-init",
        type=float,
        default=0.05,
        help="Initial bounded residual scale for low-to-stage promotion.",
    )
    parser.add_argument(
        "--hierarchical-mmdit-depth",
        type=int,
        default=3,
        help="Number of distinct full-rank action MMDiT refinement blocks.",
    )
    parser.add_argument(
        "--hierarchical-mmdit-refine-steps",
        type=int,
        default=3,
        help="Maximum recurrent refinement budget; fixed, randomized-dwell, and adaptive execution share this cap.",
    )
    parser.add_argument(
        "--hierarchical-mmdit-low-slots",
        type=int,
        default=25,
        help="Role-stratified low evidence slots; 25 gives five slots to each owned role.",
    )
    parser.add_argument(
        "--hierarchical-mmdit-stage-slots",
        type=int,
        default=6,
        help="Persistent stage-content slots; role identity remains a separate tensor.",
    )
    parser.add_argument("--hierarchical-mmdit-ffn-expansion", type=float, default=2.0)
    parser.add_argument("--hierarchical-mmdit-layer-grad-scale", type=float, default=0.0)
    parser.add_argument("--hierarchical-mmdit-source-grad-scale", type=float, default=0.0)
    parser.add_argument("--hierarchical-mmdit-consequence-scale-init", type=float, default=0.10)
    parser.add_argument("--hierarchical-mmdit-consequence-scale-max", type=float, default=0.50)
    parser.add_argument("--hierarchical-mmdit-noisy-causal", type=int, default=1)
    parser.add_argument("--hierarchical-mmdit-noisy-gate-min", type=float, default=0.05)
    parser.add_argument("--hierarchical-mmdit-noisy-gate-power", type=float, default=1.5)
    parser.add_argument("--hierarchical-mmdit-stage-promote-scale-init", type=float, default=0.05)
    parser.add_argument("--hierarchical-mmdit-output-init-std", type=float, default=1e-3)
    parser.add_argument(
        "--hierarchical-mmdit-operator-stages",
        type=int,
        default=6,
        help="Number of semantic stage-owned contraction paths, independent of refinement block count.",
    )
    parser.add_argument(
        "--hierarchical-mmdit-operator-rank",
        type=int,
        default=32,
        help="Maximum number of ordered contraction directions per semantic stage.",
    )
    parser.add_argument(
        "--hierarchical-mmdit-operator-groups",
        type=int,
        default=32,
        help="Number of ordered transparency groups; set equal to rank for channel-continuous depth.",
    )
    parser.add_argument(
        "--hierarchical-mmdit-operator-depth-logit-init",
        type=float,
        default=2.0,
        help="Initial retained-depth logit after the exact identity warm-up.",
    )
    parser.add_argument(
        "--hierarchical-mmdit-exit-logit-init",
        type=float,
        default=-4.0,
        help="Initial exit logit; negative keeps the controller on the continue path before evidence supports early exit.",
    )
    parser.add_argument(
        "--hierarchical-mmdit-operator-contraction-warmup-steps",
        type=int,
        default=200,
        help="Steps for which contraction is pinned exactly to the original operation.",
    )
    parser.add_argument(
        "--hierarchical-mmdit-operator-contraction-transition-steps",
        type=int,
        default=1500,
        help="Steps over which the learned nested contraction is introduced continuously.",
    )
    parser.add_argument(
        "--hierarchical-mmdit-unified-controller",
        type=int,
        default=0,
        help="Use the recurrent multi-token controller for retrieval, promotion, operator selection, and compute value.",
    )
    parser.add_argument(
        "--hierarchical-mmdit-control-tokens",
        type=int,
        default=8,
        help="Number of exchangeable recurrent controller state tokens.",
    )
    parser.add_argument("--hierarchical-mmdit-controller-depth", type=int, default=2)
    parser.add_argument("--hierarchical-mmdit-controller-heads", type=int, default=8)
    parser.add_argument("--hierarchical-mmdit-controller-ffn-expansion", type=float, default=2.0)
    parser.add_argument("--hierarchical-mmdit-spectral-state", type=int, choices=[0, 1], default=0)
    parser.add_argument(
        "--hierarchical-mmdit-spectral-arm-start-fraction", type=float, default=0.16
    )
    parser.add_argument(
        "--hierarchical-mmdit-spectral-gripper-start-fraction", type=float, default=0.33
    )
    parser.add_argument("--hierarchical-mmdit-spectral-temperature", type=float, default=1.5)
    parser.add_argument("--hierarchical-mmdit-spectral-schedule-power", type=float, default=1.0)
    parser.add_argument(
        "--hierarchical-mmdit-spectral-controller-shift-limit", type=float, default=2.0
    )
    parser.add_argument(
        "--hierarchical-mmdit-spectral-competition-loss-weight", type=float, default=0.0
    )
    parser.add_argument(
        "--hierarchical-mmdit-spectral-competition-warmup-steps", type=int, default=200
    )
    parser.add_argument(
        "--hierarchical-mmdit-operation-candidate-probes", type=int, choices=[0, 1], default=0
    )
    parser.add_argument("--hierarchical-mmdit-operation-route-loss-weight", type=float, default=0.0)
    parser.add_argument("--hierarchical-mmdit-operation-route-temperature", type=float, default=0.5)
    parser.add_argument("--hierarchical-mmdit-operation-route-warmup-steps", type=int, default=0)
    parser.add_argument("--hierarchical-mmdit-operation-value-loss-weight", type=float, default=0.0)
    parser.add_argument("--hierarchical-mmdit-operation-value-huber-delta", type=float, default=0.1)
    parser.add_argument(
        "--hierarchical-mmdit-operation-value-reliability-scale", type=float, default=0.0
    )
    parser.add_argument("--hierarchical-mmdit-operation-value-warmup-steps", type=int, default=200)
    parser.add_argument(
        "--hierarchical-mmdit-dwell-mode",
        choices=["fixed", "shadow", "learned"],
        default="fixed",
    )
    parser.add_argument(
        "--hierarchical-mmdit-execution-contract",
        choices=["legacy_stage_keep", "typed_block_budget"],
        default="legacy_stage_keep",
        help=(
            "Execution ownership contract. typed_block_budget routes dwell over "
            "real MMDiT blocks, keeps host LayerScale as the only residual "
            "amplitude owner, and uses the controller only for compute depth."
        ),
    )
    parser.add_argument(
        "--hierarchical-mmdit-schedule-mode", choices=["fixed", "random_dwell"], default="fixed"
    )
    parser.add_argument("--hierarchical-mmdit-random-prefix-probability", type=float, default=0.0)
    parser.add_argument(
        "--hierarchical-mmdit-exhaustion-mode",
        choices=["off", "shadow", "adaptive", "learned_shadow", "learned"],
        default="off",
    )
    parser.add_argument(
        "--hierarchical-mmdit-action-response-thresholds",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("T0", "T1", "T2"),
    )
    parser.add_argument(
        "--hierarchical-mmdit-stage-pressure-thresholds",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("T0", "T1", "T2"),
    )
    parser.add_argument("--hierarchical-mmdit-action-response-floor", type=float, default=0.05)
    parser.add_argument("--hierarchical-mmdit-exhaustion-confirm-steps", type=int, default=2)
    parser.add_argument(
        "--hierarchical-mmdit-residual-scale-init",
        type=float,
        default=0.05,
        help="Base value used to initialize the original V77 host-gate profile.",
    )
    parser.add_argument(
        "--hierarchical-mmdit-residual-scale-max",
        type=float,
        default=0.20,
        help="Bound for the original V77 host LayerScale gates; the sidecar has no amplitude control.",
    )
    parser.add_argument(
        "--hierarchical-mmdit-output-contract",
        type=int,
        default=0,
        help="CR7 fallback: dedicated restricted contract for event/motion subheads only (zero-init injection; velocity head untouched).",
    )
    parser.add_argument(
        "--hierarchical-mmdit-noisy-market-bias",
        type=int,
        default=0,
        help="Compatibility flag for v76a checkpoints; the serial clean decoder has no condition-group market.",
    )
    parser.add_argument(
        "--hierarchical-mmdit-noisy-gate-mode",
        type=int,
        default=0,
        help="0 = no t-dependent noisy modulation; 1 = apply the t schedule inside noisy-branch low-rank channel amplitudes.",
    )
    parser.add_argument(
        "--latent-cvae-z-probe",
        type=int,
        default=0,
        help="CR0: eval-time z zero/shuffle intervention probes on the legacy decoder (two extra decodes per eval batch; diagnostic runs only).",
    )
    parser.add_argument("--adaptive-cvae-refine-steps", type=int, default=3)
    parser.add_argument("--adaptive-cvae-progress-memory", type=int, default=1)
    parser.add_argument("--adaptive-cvae-progress-steps", type=int, default=6)
    parser.add_argument("--adaptive-cvae-prefix-memory", type=int, default=0)
    parser.add_argument("--adaptive-cvae-layer-routing", type=int, default=1)
    parser.add_argument("--adaptive-cvae-route-cosine", type=int, default=1)
    parser.add_argument("--adaptive-cvae-route-temperature", type=float, default=1.0)
    parser.add_argument("--adaptive-cvae-prefix-detach", type=int, default=1)
    parser.add_argument("--adaptive-cvae-progress-z-injection", type=int, default=1)
    parser.add_argument("--adaptive-cvae-route-query-bias", type=int, default=1)
    parser.add_argument(
        "--adaptive-cvae-route-time-query",
        type=int,
        default=0,
        help="V74B: add the existing z+time primary condition to route queries only.",
    )
    parser.add_argument("--adaptive-cvae-token-semantic-adapter", type=int, default=1)
    parser.add_argument("--adaptive-cvae-output-adapter", type=int, default=0)
    parser.add_argument("--adaptive-cvae-context-dropout", type=float, default=0.05)
    parser.add_argument("--adaptive-cvae-route-entropy-floor-ratio", type=float, default=0.35)
    parser.add_argument("--adaptive-cvae-function-adapters", type=int, default=1)
    parser.add_argument("--adaptive-cvae-function-rank", type=int, default=64)
    parser.add_argument("--adaptive-cvae-progress-role-dim", type=int, default=16)
    parser.add_argument("--adaptive-cvae-route-topk", type=int, default=0)
    parser.add_argument("--adaptive-cvae-route-sparsemax", type=int, default=1)
    parser.add_argument("--adaptive-cvae-route-adaptive-temperature", type=int, default=1)
    parser.add_argument("--adaptive-cvae-route-min-temperature", type=float, default=0.35)
    parser.add_argument("--adaptive-cvae-route-max-temperature", type=float, default=1.25)
    parser.add_argument("--adaptive-cvae-role-query", type=int, default=1)
    parser.add_argument("--adaptive-cvae-step-roles", type=int, default=1)
    parser.add_argument("--adaptive-cvae-coarse-stride", type=int, default=4)
    parser.add_argument("--adaptive-cvae-coarse-strength", type=float, default=0.35)
    parser.add_argument("--adaptive-cvae-seed-scale", type=float, default=0.35)
    parser.add_argument("--adaptive-cvae-output-scale", type=float, default=0.05)
    parser.add_argument("--adaptive-cvae-context-capsules", type=int, default=1)
    parser.add_argument("--adaptive-cvae-context-capsule-count", type=int, default=6)
    parser.add_argument("--adaptive-cvae-direct-condition-residual", type=int, default=0)
    parser.add_argument("--adaptive-cvae-condition-strength", type=int, default=0)
    parser.add_argument("--adaptive-cvae-condition-strength-min", type=float, default=0.03)
    parser.add_argument("--adaptive-cvae-condition-strength-max", type=float, default=1.5)
    parser.add_argument("--adaptive-cvae-condition-strength-init", type=float, default=0.35)
    parser.add_argument("--adaptive-cvae-micro-control", type=int, default=1)
    parser.add_argument("--adaptive-cvae-micro-refine-block", type=int, default=1)
    parser.add_argument("--adaptive-cvae-micro-supervision", type=int, default=1)
    parser.add_argument("--adaptive-cvae-micro-heun", type=int, default=1)
    parser.add_argument("--adaptive-cvae-micro-monotonic-progress", type=int, default=1)
    parser.add_argument("--adaptive-cvae-micro-min-step", type=float, default=0.03)
    parser.add_argument("--adaptive-cvae-micro-max-step", type=float, default=0.35)
    parser.add_argument("--adaptive-cvae-micro-step-init", type=float, default=0.12)
    parser.add_argument("--adaptive-cvae-micro-kp-max", type=float, default=0.6)
    parser.add_argument("--adaptive-cvae-micro-kp-init", type=float, default=0.18)
    parser.add_argument("--adaptive-cvae-micro-kd-max", type=float, default=0.45)
    parser.add_argument("--adaptive-cvae-micro-kd-init", type=float, default=0.08)
    parser.add_argument("--adaptive-cvae-micro-update-scale", type=float, default=1.0)
    parser.add_argument("--adaptive-cvae-micro-refine-block-scale", type=float, default=0.3)
    parser.add_argument("--adaptive-cvae-micro-progress-distance-scale", type=float, default=4.0)

    defaults = V39PolicyTrainerConfig()
    # These fields are also policy-config arguments above because the same
    # command-line value controls both decoder construction and the training
    # objective. Do not register them a second time in the generic trainer
    # field loop.
    explicitly_registered_trainer_fields = {
        "hierarchical_mmdit_operation_candidate_probes",
        "hierarchical_mmdit_operation_route_loss_weight",
        "hierarchical_mmdit_operation_route_temperature",
        "hierarchical_mmdit_operation_route_warmup_steps",
        "hierarchical_mmdit_operation_value_loss_weight",
        "hierarchical_mmdit_operation_value_huber_delta",
        "hierarchical_mmdit_operation_value_reliability_scale",
        "latent_cvae_mmdit_dwell_mode",
    }
    for field in V39PolicyTrainerConfig.__dataclass_fields__:
        if field in explicitly_registered_trainer_fields:
            continue
        value = getattr(defaults, field)
        parser.add_argument(
            "--" + field.replace("_", "-"), dest=field, type=type(value), default=value
        )
    # The V40 entry point is specifically the multi-layer intervention-latent
    # experiment; inheriting V39's midcut default silently skips that branch.
    parser.set_defaults(contract_mode="layer_adapter")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    identity_advantage_weight = float(args.flow_jepa_identity_advantage_loss_weight)
    static_identity_weight = float(args.flow_jepa_static_identity_loss_weight)
    if int(args.flow_jepa_zero_flow_guard) and identity_advantage_weight <= 0.0:
        raise ValueError(
            "zero-flow guard requires an active --flow-jepa-identity-advantage-loss-weight"
        )
    if identity_advantage_weight > 0.0 and not int(args.flow_jepa_zero_flow_guard):
        raise ValueError(
            "identity-advantage supervision requires --flow-jepa-zero-flow-guard 1"
        )
    if static_identity_weight > 0.0 and not int(args.flow_jepa_zero_flow_guard):
        raise ValueError(
            "static-identity supervision requires --flow-jepa-zero-flow-guard 1"
        )
    if int(args.flow_jepa_complementary_raw_detail) and static_identity_weight <= 0.0:
        raise ValueError(
            "complementary raw detail requires an active "
            "--flow-jepa-static-identity-loss-weight"
        )
    if int(args.flow_jepa_predictive_change_contract):
        if float(args.flow_jepa_future_loss_weight) <= 0.0:
            raise ValueError(
                "predictive-change contract requires an active future JEPA loss"
            )
        if int(args.flow_jepa_teacher_balanced_target_mask):
            raise ValueError(
                "predictive-change contract requires the same observation-only "
                "mask for online context and future targets; disable the "
                "teacher-balanced target mask"
            )
        if float(args.flow_jepa_future_change_loss_weight) > 0.0:
            raise ValueError(
                "predictive-change contract already makes change the primary future "
                "objective; disable the duplicate future-change auxiliary weight"
            )
    interval_stage_weight = float(args.flow_jepa_interval_stage_loss_weight)
    if int(args.flow_jepa_interval_stage_delta) and interval_stage_weight <= 0.0:
        raise ValueError(
            "interval-stage delta requires an active "
            "--flow-jepa-interval-stage-loss-weight"
        )
    if interval_stage_weight > 0.0 and not int(
        args.flow_jepa_interval_stage_delta
    ):
        raise ValueError(
            "interval-stage supervision requires "
            "--flow-jepa-interval-stage-delta 1"
        )
    if int(args.flow_jepa_raw_image_enabled) and str(args.final_action_decoder) != (
        "evidence_latent_mmdit_action"
    ):
        raise ValueError(
            "raw 3+3+2 Flow-JEPA requires the evidence_latent_mmdit_action final decoder"
        )
    requested_stage = str(args.training_stage).lower().replace("-", "_")
    if int(args.single_stage_role_lr):
        if requested_stage not in {"policy", "stage2"}:
            raise ValueError("single-stage role LR requires policy/stage2 training")
        if not int(args.flow_jepa_role_hierarchy):
            raise ValueError("single-stage role LR requires the Flow-JEPA role hierarchy")
        if int(args.stage1_initialization_enabled):
            raise ValueError("single-stage role LR requires Stage1 initialization to be off")
    if int(args.flow_jepa_raw_image_enabled):
        if requested_stage not in {"policy", "stage2"}:
            raise ValueError("raw 3+3+2 Flow-JEPA is a single-stage policy experiment")
        if int(args.stage1_initialization_enabled) and args.stage1_checkpoint is not None:
            raise ValueError(
                "raw 3+3+2 Flow-JEPA must not initialize from a historical Stage1 checkpoint"
            )
    flow_jepa_stage1 = bool(
        int(args.flow_jepa_enabled) and requested_stage in {"contract", "stage1"}
    )
    if flow_jepa_stage1 and int(args.flow_jepa_late_policy_detail):
        raise ValueError(
            "late policy detail is a single-stage action path and cannot run "
            "inside the representation-only Flow-JEPA Stage1 objective"
        )
    if flow_jepa_stage1:
        if int(args.future_latent_loss_start_epoch) != 1:
            raise ValueError("V95 Stage1 requires --future-latent-loss-start-epoch 1")
        if int(args.future_latent_max_batches) != 0:
            raise ValueError("V95 Stage1 requires future teacher targets on every batch")
        if max(
            float(args.flow_jepa_future_loss_weight),
            float(args.flow_jepa_future_change_loss_weight),
            float(args.flow_jepa_interval_stage_loss_weight),
            float(args.flow_jepa_stage_loss_weight),
        ) <= 0.0:
            raise ValueError("V95 Stage1 requires an active future or stage JEPA target")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
    device = resolve_device(args.device)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]
    cameras = tuple(str(x) for x in args.cameras)

    legacy = _legacy_payload(args.legacy_context_checkpoint)
    legacy_context = None if legacy is None else dict(legacy["context"])
    legacy_splits = None if legacy_context is None else legacy_context.get("splits")
    action_norm = None if legacy is None else ArrayNormalizer.from_dict(legacy["action_normalizer"])
    state_norm = None if legacy is None else ArrayNormalizer.from_dict(legacy["state_normalizer"])

    dataset_config = ObservedStateDatasetConfig(
        world_horizon=args.world_horizon,
        policy_horizon=args.policy_horizon,
        segment_length=args.segment_length,
        history_offsets=tuple(args.history_offsets),
        executed_action_offsets=tuple(args.executed_action_offsets),
        target_history_offsets=tuple(args.target_history_offsets),
        stride=args.stride,
        return_images=(
            args.condition_mode != "dinov2-cache"
            or bool(int(args.flow_jepa_raw_image_enabled))
        ),
        return_target_images=args.condition_mode != "dinov2-cache",
    )
    dataset_config.validate()
    if int(args.flow_jepa_enabled):
        window_offsets = tuple(int(value) for value in args.flow_jepa_window_offsets)
        if len(window_offsets) != int(args.future_anchors):
            raise ValueError(
                "--future-anchors must equal the number of --flow-jepa-window-offsets"
            )
        if int(args.flow_jepa_interval_stage_delta):
            if not int(args.flow_jepa_late_bottleneck):
                raise ValueError(
                    "interval-stage targets require the late-bottleneck path"
                )
            requested_future_offsets = tuple(
                int(value)
                for value in args.flow_jepa_interval_support_offsets
            )
        else:
            requested_future_offsets = (
                window_offsets
                if int(args.flow_jepa_late_bottleneck)
                else (*window_offsets, int(args.flow_jepa_stage_offset))
            )
        available_lookup = {
            int(offset): index for index, offset in enumerate(dataset_config.future_offsets)
        }
        missing_offsets = [
            offset for offset in requested_future_offsets if offset not in available_lookup
        ]
        if missing_offsets:
            raise ValueError(
                "Flow-DINO requested offsets are not on the dataset future grid: "
                f"{missing_offsets}; available={dataset_config.future_offsets}"
            )
        target_future_indices = tuple(
            available_lookup[offset] for offset in requested_future_offsets
        )
    else:
        target_future_indices = tuple(range(int(args.future_anchors)))
    min_length = (
        dataset_config.world_horizon
        + abs(min(dataset_config.history_offsets + dataset_config.executed_action_offsets))
        + 2
    )
    episodes, train_ids, val_ids, test_ids, action_norm, state_norm, image_store, skipped = (
        load_data(
            args,
            min_length=min_length,
            normalizer_mode=(action_norm.mode if action_norm is not None else args.normalizer),
            action_normalizer=action_norm,
            state_normalizer=state_norm,
            splits=legacy_splits,
        )
    )
    bases = {
        name: ObservedStateWindowDataset(
            episodes,
            ids,
            image_store=image_store,
            camera_names=cameras,
            state_normalizer=state_norm,
            action_normalizer=action_norm,
            config=dataset_config,
        )
        for name, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids))
    }
    conditioner, latent_dim, patches = build_dense_conditioner(
        mode=args.condition_mode,
        episodes=episodes,
        camera_names=cameras,
        preprocessing=preprocessing_from_args(args),
        dinov2_model=args.dinov2_model,
        dinov2_local_files_only=args.dinov2_local_files_only,
        dinov2_token_cache_dir=args.dinov2_token_cache_dir,
        debug_token_dim=768,
        debug_patches_per_camera=256,
        device=device,
        dtype=dtype,
    )
    use_token_prefetch = bool(
        args.prefetch_dinov2_tokens
        and args.condition_mode == "dinov2-cache"
        and hasattr(conditioner, "store")
    )
    if use_token_prefetch:
        token_store = conditioner.store  # type: ignore[attr-defined]
        train_dataset = CachedTokenPolicyWindowDataset(
            bases["train"],
            token_store=token_store,
            future_anchors=int(args.future_anchors),
            future_indices=target_future_indices,
        )
        val_dataset = CachedTokenPolicyWindowDataset(
            bases["val"],
            token_store=token_store,
            future_anchors=int(args.future_anchors),
            future_indices=target_future_indices,
        )
    else:
        train_dataset = PolicyWindowDataset(bases["train"])
        val_dataset = PolicyWindowDataset(bases["val"])
    train_batch_sampler = None
    information_sampling_summary: dict[str, float | int] | None = None
    if int(args.information_balanced_sampling):
        motion_score, is_event = bases["train"].training_information_signals(
            gripper_index=int(args.gripper_dim_index),
            event_threshold=float(args.gripper_event_threshold),
        )
        train_batch_sampler = InformationBalancedBatchSampler(
            motion_score,
            is_event,
            InformationBalancedSamplerConfig(
                batch_size=int(args.batch_size),
                uniform_fraction=float(args.information_uniform_fraction),
                event_fraction=float(args.information_event_fraction),
                motion_quantile=float(args.information_motion_quantile),
                seed=int(args.seed),
            ),
        )
        information_sampling_summary = train_batch_sampler.summary
    train_loader_generator = torch.Generator()
    train_loader_generator.manual_seed(int(args.seed))
    val_loader_generator = torch.Generator()
    val_loader_generator.manual_seed(int(args.seed) + 1)
    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        workers=args.num_workers,
        shuffle=train_batch_sampler is None,
        device=device,
        generator=train_loader_generator,
        batch_sampler=train_batch_sampler,
    )
    val_loader = make_loader(
        val_dataset,
        batch_size=args.batch_size,
        workers=args.num_workers,
        shuffle=False,
        device=device,
        generator=val_loader_generator,
    )
    if patches is None:
        probe_sample = bases["train"][0]
        latent_dim, patches = infer_dense_geometry(conditioner, probe_sample, camera_names=cameras)
    visual_geometry = {
        "source": args.condition_mode,
        "latent_dim": int(latent_dim),
        "patches_per_camera": int(patches),
        "num_cameras": len(cameras),
        "history_length": len(dataset_config.history_offsets),
        "future_count": len(dataset_config.future_offsets),
    }

    goal_language_tokens: torch.Tensor | None = None
    goal_language_mask: torch.Tensor | None = None
    goal_language_metadata: dict[str, Any] | None = None
    goal_language_dim = 768
    if int(args.goal_conditioning_enabled):
        if args.t5_condition_path is None:
            raise ValueError(
                "--t5-condition-path is required when goal conditioning is enabled"
            )
        goal_language_tokens, goal_language_mask, goal_language_metadata = (
            load_precomputed_t5_condition(
                condition_path=args.t5_condition_path,
                max_tokens=args.goal_language_max_tokens,
            )
        )
        goal_language_dim = int(goal_language_tokens.shape[-1])
        goal_language_metadata = {
            **goal_language_metadata,
            "embedding_dim": goal_language_dim,
            "valid_tokens": int(goal_language_mask.sum().item()),
            "embedding_sha256": hashlib.sha256(
                goal_language_tokens.contiguous().numpy().tobytes()
            ).hexdigest(),
        }

    policy_config = V39PolicyConfig(
        action_dim=int(action_norm.scale.shape[-1]),
        state_dim=int(state_norm.scale.shape[-1]),
        action_horizon=dataset_config.policy_horizon,
        executed_history_length=len(dataset_config.executed_action_offsets),
        action_history_enabled=args.action_history_enabled,
        executed_action_offsets=tuple(dataset_config.executed_action_offsets),
        action_history_recent_tokens=args.action_history_recent_tokens,
        action_history_summary_tokens=args.action_history_summary_tokens,
        action_history_condition_dropout=args.action_history_condition_dropout,
        action_history_condition_exact_null=(
            args.action_history_condition_exact_null
        ),
        action_history_proposal_detach=args.action_history_proposal_detach,
        goal_conditioning_enabled=args.goal_conditioning_enabled,
        goal_token_count=args.goal_token_count,
        goal_language_dim=goal_language_dim,
        goal_language_max_tokens=args.goal_language_max_tokens,
        goal_resampler_depth=args.goal_resampler_depth,
        goal_condition_dropout=args.goal_condition_dropout,
        goal_condition_exact_null=args.goal_condition_exact_null,
        stateless_phase_enabled=args.stateless_phase_enabled,
        stateless_phase_count=args.stateless_phase_count,
        stateless_phase_query_scale=args.stateless_phase_query_scale,
        hidden_size=args.hidden_size,
        num_heads=args.heads,
        depth=args.depth,
        action_decoder_depth=1,
        proposal_depth=args.proposal_depth,
        proposal_dropout=args.proposal_dropout,
        dropout=args.dropout,
        event_tokens=args.event_tokens,
        gripper_dim_index=args.gripper_dim_index,
        inference_steps=args.inference_steps,
        first_execution_steps=args.first_execution_steps,
        mid_execution_steps=args.mid_execution_steps,
        physical_decode_delta_blend=args.physical_decode_delta_blend,
        arm_flow_mode=args.arm_flow_mode,
        arm_noise_temporal_rho=args.arm_noise_temporal_rho,
        arm_source_mode=args.arm_source_mode,
        arm_source_scale=args.arm_source_scale,
        arm_source_innovation_weight=args.arm_source_innovation_weight,
        arm_source_velocity_weight=args.arm_source_velocity_weight,
        arm_source_acceleration_weight=args.arm_source_acceleration_weight,
        gripper_field_dim=args.gripper_field_dim,
        gripper_field_mode=args.gripper_field_mode,
        visual_token_dim=int(latent_dim),
        visual_history_length=len(dataset_config.history_offsets),
        num_cameras=len(cameras),
        patches_per_camera=int(patches),
        canvas_registers=args.canvas_registers,
        future_anchors=min(int(args.future_anchors), len(dataset_config.future_offsets)),
        target_future_count=len(dataset_config.future_offsets),
        visual_memory_dropout=args.visual_memory_dropout,
        canvas_dropout=args.canvas_dropout,
        role_dropout=args.role_dropout,
        action_basis_tokens=args.action_basis_tokens,
        future_grid_size=args.future_grid_size,
        flow_jepa_enabled=args.flow_jepa_enabled,
        flow_jepa_grid_size=args.flow_jepa_grid_size,
        flow_jepa_feature_dim=args.flow_jepa_feature_dim,
        flow_jepa_flow_iters=args.flow_jepa_flow_iters,
        flow_jepa_corr_levels=args.flow_jepa_corr_levels,
        flow_jepa_corr_radius=args.flow_jepa_corr_radius,
        flow_jepa_mask_ratio=args.flow_jepa_mask_ratio,
        flow_jepa_mask_block_size=args.flow_jepa_mask_block_size,
        flow_jepa_motion_mask_fraction=args.flow_jepa_motion_mask_fraction,
        flow_jepa_teacher_balanced_target_mask=(
            args.flow_jepa_teacher_balanced_target_mask
        ),
        flow_jepa_teacher_mask_past_fraction=(
            args.flow_jepa_teacher_mask_past_fraction
        ),
        flow_jepa_teacher_mask_change_fraction=(
            args.flow_jepa_teacher_mask_change_fraction
        ),
        flow_jepa_predictive_change_contract=(
            args.flow_jepa_predictive_change_contract
        ),
        flow_jepa_uncertainty_floor=args.flow_jepa_uncertainty_floor,
        flow_jepa_late_bottleneck=args.flow_jepa_late_bottleneck,
        flow_jepa_dense_depth=args.flow_jepa_dense_depth,
        flow_jepa_fine_radius=args.flow_jepa_fine_radius,
        flow_jepa_reader_radius=args.flow_jepa_reader_radius,
        flow_jepa_reader_heads=args.flow_jepa_reader_heads,
        flow_jepa_raw_image_enabled=args.flow_jepa_raw_image_enabled,
        flow_jepa_role_hierarchy=args.flow_jepa_role_hierarchy,
        flow_jepa_raw_base_channels=args.flow_jepa_raw_base_channels,
        flow_jepa_raw_mid_radius=args.flow_jepa_raw_mid_radius,
        flow_jepa_raw_high_radius=args.flow_jepa_raw_high_radius,
        flow_jepa_raw_reader_radius=args.flow_jepa_raw_reader_radius,
        flow_jepa_raw_reader_heads=args.flow_jepa_raw_reader_heads,
        flow_jepa_raw_activation_checkpoint=args.flow_jepa_raw_activation_checkpoint,
        flow_jepa_zero_flow_guard=args.flow_jepa_zero_flow_guard,
        flow_jepa_strict_role_visual_path=args.flow_jepa_strict_role_visual_path,
        flow_jepa_complementary_raw_detail=args.flow_jepa_complementary_raw_detail,
        flow_jepa_source_aligned_raw_fusion=(
            args.flow_jepa_source_aligned_raw_fusion
        ),
        flow_jepa_grounding_blocks=args.flow_jepa_grounding_blocks,
        flow_jepa_world_blocks=args.flow_jepa_world_blocks,
        flow_jepa_policy_blocks=args.flow_jepa_policy_blocks,
        flow_jepa_policy_workspace_scale=args.flow_jepa_policy_workspace_scale,
        flow_jepa_policy_workspace_fixed_fusion=(
            args.flow_jepa_policy_workspace_fixed_fusion
        ),
        flow_jepa_world_anchor_write_only=args.flow_jepa_world_anchor_write_only,
        flow_jepa_late_policy_detail=args.flow_jepa_late_policy_detail,
        flow_jepa_late_policy_detail_scale=args.flow_jepa_late_policy_detail_scale,
        flow_jepa_soft_address_lattice=args.flow_jepa_soft_address_lattice,
        flow_jepa_address_slots=args.flow_jepa_address_slots,
        flow_jepa_address_route_dim=args.flow_jepa_address_route_dim,
        flow_jepa_address_query_chunk=args.flow_jepa_address_query_chunk,
        flow_jepa_policy_multi_glimpse_address=(
            args.flow_jepa_policy_multi_glimpse_address
        ),
        flow_jepa_address_flow_prior_floor=(
            args.flow_jepa_address_flow_prior_floor
        ),
        flow_jepa_bounded_flow_coordinates=(
            args.flow_jepa_bounded_flow_coordinates
        ),
        flow_jepa_sequential_horizon_memory=(
            args.flow_jepa_sequential_horizon_memory
        ),
        flow_jepa_horizon_soft_address=(
            args.flow_jepa_horizon_soft_address
        ),
        flow_jepa_horizon_address_update_scale=(
            args.flow_jepa_horizon_address_update_scale
        ),
        flow_jepa_horizon_cell_fine_address=(
            args.flow_jepa_horizon_cell_fine_address
        ),
        flow_jepa_online_horizon_address=(
            args.flow_jepa_online_horizon_address
        ),
        flow_jepa_progressive_grounding_address=(
            args.flow_jepa_progressive_grounding_address
        ),
        flow_jepa_coordinate_typed_raw_detail=(
            args.flow_jepa_coordinate_typed_raw_detail
        ),
        flow_jepa_structured_ownership_bottleneck=(
            args.flow_jepa_structured_ownership_bottleneck
        ),
        flow_jepa_pre_value_owner_routing=(
            args.flow_jepa_pre_value_owner_routing
        ),
        flow_jepa_pre_value_owner_update_scale=(
            args.flow_jepa_pre_value_owner_update_scale
        ),
        flow_jepa_functional_mainline_routing=(
            args.flow_jepa_functional_mainline_routing
        ),
        flow_jepa_utility_precision_mainline=(
            args.flow_jepa_utility_precision_mainline
        ),
        flow_jepa_action_free_world_factual=(
            args.flow_jepa_action_free_world_factual
        ),
        flow_jepa_shared_factual_glimpse_bank=(
            args.flow_jepa_shared_factual_glimpse_bank
        ),
        flow_jepa_g_aligned_future_effect=(
            args.flow_jepa_g_aligned_future_effect
        ),
        flow_jepa_teacher_g_ema_decay=(
            args.flow_jepa_teacher_g_ema_decay
        ),
        flow_jepa_stateless_goal_phase_machine=(
            args.flow_jepa_stateless_goal_phase_machine
        ),
        flow_jepa_top_role_schedule=(
            args.flow_jepa_top_role_schedule
        ),
        flow_jepa_policy_plan_compiler=(
            args.flow_jepa_policy_plan_compiler
        ),
        flow_jepa_supervised_effect_mainline=(
            args.flow_jepa_supervised_effect_mainline
        ),
        flow_jepa_stateless_intent_controller=(
            args.flow_jepa_stateless_intent_controller
        ),
        flow_jepa_window_effect_bank=args.flow_jepa_window_effect_bank,
        flow_jepa_future_slots=args.flow_jepa_future_slots,
        flow_jepa_effect_read_in_p2=args.flow_jepa_effect_read_in_p2,
        flow_jepa_differential_intent_effect_mainline=(
            args.flow_jepa_differential_intent_effect_mainline
        ),
        flow_jepa_grounded_intent_effect_mainline=(
            args.flow_jepa_grounded_intent_effect_mainline
        ),
        flow_jepa_object_intent_dynamics_mainline=(
            args.flow_jepa_object_intent_dynamics_mainline
        ),
        flow_matching_time_distribution=(
            args.flow_matching_time_distribution
        ),
        flow_jepa_address_query_batch_budget=(
            args.flow_jepa_address_query_batch_budget
        ),
        flow_jepa_microgrid_tile=args.flow_jepa_microgrid_tile,
        flow_jepa_p1_mixed_precision=args.flow_jepa_p1_mixed_precision,
        flow_jepa_checkpoint_min_batch=(
            args.flow_jepa_checkpoint_min_batch
        ),
        flow_jepa_raw_micro_grid=args.flow_jepa_raw_micro_grid,
        flow_jepa_variance_safe_routing=(
            args.flow_jepa_variance_safe_routing
        ),
        flow_jepa_complete_numerical_contract=(
            args.flow_jepa_complete_numerical_contract
        ),
        flow_jepa_routing_norm_floor=args.flow_jepa_routing_norm_floor,
        flow_jepa_correlation_rms_floor=(
            args.flow_jepa_correlation_rms_floor
        ),
        flow_jepa_visibility_transition_fraction=(
            args.flow_jepa_visibility_transition_fraction
        ),
        flow_jepa_horizon_value_max_rms=(
            args.flow_jepa_horizon_value_max_rms
        ),
        flow_jepa_interval_stage_delta=(
            args.flow_jepa_interval_stage_delta
        ),
        flow_jepa_interval_boundaries=tuple(
            args.flow_jepa_interval_boundaries
        ),
        flow_jepa_interval_support_offsets=tuple(
            args.flow_jepa_interval_support_offsets
        ),
        flow_jepa_interval_stage_update_scale=(
            args.flow_jepa_interval_stage_update_scale
        ),
        flow_jepa_interval_stage_typed_value=(
            args.flow_jepa_interval_stage_typed_value
        ),
        role_attnres_enabled=args.role_attnres_enabled,
        role_attnres_key_dim=args.role_attnres_key_dim,
        role_attnres_ground_to_world=args.role_attnres_ground_to_world,
        role_attnres_world_to_policy=args.role_attnres_world_to_policy,
        role_attnres_policy_to_mmdit=args.role_attnres_policy_to_mmdit,
        role_attnres_ground_to_world_scale=(
            args.role_attnres_ground_to_world_scale
        ),
        role_attnres_world_to_policy_scale=(
            args.role_attnres_world_to_policy_scale
        ),
        role_attnres_policy_to_mmdit_scale=(
            args.role_attnres_policy_to_mmdit_scale
        ),
        role_residual_amplitude_contract=(
            args.role_residual_amplitude_contract
        ),
        role_residual_max_update_rms=args.role_residual_max_update_rms,
        role_attnres_max_value_rms=args.role_attnres_max_value_rms,
        role_residual_contract_after_gate=(
            args.role_residual_contract_after_gate
        ),
        flow_jepa_policy_workspace_horizon_pool=(
            args.flow_jepa_policy_workspace_horizon_pool
        ),
        flow_jepa_directed_canvas_attention=args.flow_jepa_directed_canvas_attention,
        flow_jepa_history_offsets=tuple(dataset_config.history_offsets),
        flow_jepa_window_offsets=tuple(args.flow_jepa_window_offsets),
        flow_jepa_stage_offset=args.flow_jepa_stage_offset,
        flow_jepa_stage_tokens=0 if int(args.flow_jepa_late_bottleneck) else 1,
        rollout_tail_start_step=args.rollout_tail_start_step,
        rollout_tail_full_step=args.rollout_tail_full_step,
        controlled_delta_rank=args.controlled_delta_rank,
        base_effect_hidden=args.base_effect_hidden,
        controlled_base_mode=args.controlled_base_mode,
        latent_action_tokens=args.latent_action_tokens,
        neutral_action_tokens=args.neutral_action_tokens,
        controlled_delta_dropout=args.controlled_delta_dropout,
        midcut_layer=args.midcut_layer,
        midcut_future_gain_init=args.midcut_future_gain_init,
        layer_contract_adapters=args.layer_contract_adapters,
        layer_contract_adapter_dim=args.layer_contract_adapter_dim,
        layer_contract_grad_scale=args.layer_contract_grad_scale,
        layer_contract_residual_scale=args.layer_contract_residual_scale,
        layer_shared_fm_probe=args.layer_shared_fm_probe,
        layer_fm_probe_hidden=args.layer_fm_probe_hidden,
        layer_recurrent_consequence=args.layer_recurrent_consequence,
        layer_consequence_steps=args.layer_consequence_steps,
        layer_consequence_hidden=args.layer_consequence_hidden,
        layer_consequence_delta_scale=args.layer_consequence_delta_scale,
        layer_consequence_initial_gain=args.layer_consequence_initial_gain,
        layer_causal_feedback_depth=args.layer_causal_feedback_depth,
        layer_causal_memory_tokens=args.layer_causal_memory_tokens,
        layer_low_causal_weight=args.layer_low_causal_weight,
        layer_high_causal_weight=args.layer_high_causal_weight,
        layer_low_latent_weight=args.layer_low_latent_weight,
        layer_high_latent_weight=args.layer_high_latent_weight,
        layer_causal_event_from_effect=args.layer_causal_event_from_effect,
        layer_state_counterfactual=args.layer_state_counterfactual,
        action_consequence_self_condition=args.action_consequence_self_condition,
        layer_zero_base_diagnostic=args.layer_zero_base_diagnostic,
        final_action_decoder=args.final_action_decoder,
        action_flow_residual_depth=args.action_flow_residual_depth,
        action_flow_residual_high_slots=args.action_flow_residual_high_slots,
        action_flow_residual_max_scale=args.action_flow_residual_max_scale,
        action_flow_residual_visual_memory=args.action_flow_residual_visual_memory,
        action_flow_residual_context_memory=args.action_flow_residual_context_memory,
        action_flow_residual_transition_memory=args.action_flow_residual_transition_memory,
        action_flow_residual_layer_memory=args.action_flow_residual_layer_memory,
        action_flow_residual_layer_pair_schedule=args.action_flow_residual_layer_pair_schedule,
        action_flow_residual_layer_detach=args.action_flow_residual_layer_detach,
        action_flow_residual_stage_router=args.action_flow_residual_stage_router,
        action_flow_residual_anchor_memory=args.action_flow_residual_anchor_memory,
        latent_action_decoder_depth=args.latent_action_decoder_depth,
        latent_action_high_slots=args.latent_action_high_slots,
        latent_action_layer_schedule=args.latent_action_layer_schedule,
        latent_action_visual_memory=args.latent_action_visual_memory,
        latent_action_context_memory=args.latent_action_context_memory,
        latent_action_transition_memory=args.latent_action_transition_memory,
        latent_action_layer_memory=args.latent_action_layer_memory,
        latent_action_anchor_memory=args.latent_action_anchor_memory,
        latent_action_stage_router=args.latent_action_stage_router,
        latent_action_layer_detach=args.latent_action_layer_detach,
        latent_action_event_gripper_gate=args.latent_action_event_gripper_gate,
        latent_action_temporal_depth=args.latent_action_temporal_depth,
        latent_action_near_steps=args.latent_action_near_steps,
        latent_action_mid_steps=args.latent_action_mid_steps,
        latent_action_near_depth=args.latent_action_near_depth,
        latent_action_mid_depth=args.latent_action_mid_depth,
        latent_cvae_z_dim=args.latent_cvae_z_dim,
        latent_cvae_decoder_depth=args.latent_cvae_decoder_depth,
        latent_cvae_ffn_expansion=args.latent_cvae_ffn_expansion,
        latent_cvae_layer_memory=args.latent_cvae_layer_memory,
        latent_cvae_transition_memory=args.latent_cvae_transition_memory,
        latent_cvae_transition_detach=args.latent_cvae_transition_detach,
        latent_cvae_context_memory=args.latent_cvae_context_memory,
        latent_cvae_visual_memory=args.latent_cvae_visual_memory,
        latent_cvae_layer_detach=args.latent_cvae_layer_detach,
        latent_cvae_layer_grad_scale=args.latent_cvae_layer_grad_scale,
        latent_cvae_condition_source_norm=args.latent_cvae_condition_source_norm,
        latent_cvae_bounded_consequence_fusion=args.latent_cvae_bounded_consequence_fusion,
        latent_cvae_consequence_scale_init=args.latent_cvae_consequence_scale_init,
        latent_cvae_consequence_scale_max=args.latent_cvae_consequence_scale_max,
        latent_cvae_event_gripper_gate=args.latent_cvae_event_gripper_gate,
        latent_cvae_inference_sample=args.latent_cvae_inference_sample,
        latent_cvae_variational=args.latent_cvae_variational,
        latent_cvae_output_init_std=args.latent_cvae_output_init_std,
        latent_cvae_mu_bound=args.latent_cvae_mu_bound,
        latent_cvae_min_std=args.latent_cvae_min_std,
        latent_cvae_causal_attention=args.latent_cvae_causal_attention,
        latent_cvae_noisy_gate=args.latent_cvae_noisy_gate,
        latent_cvae_noisy_gate_min=args.latent_cvae_noisy_gate_min,
        latent_cvae_noisy_gate_power=args.latent_cvae_noisy_gate_power,
        latent_cvae_layer_scan=args.latent_cvae_layer_scan,
        latent_cvae_layer_scan_alpha=args.latent_cvae_layer_scan_alpha,
        latent_cvae_mmdit_decoder=args.latent_cvae_mmdit_decoder,
        latent_cvae_mmdit_depth=args.latent_cvae_mmdit_depth,
        latent_cvae_mmdit_cond_update=args.latent_cvae_mmdit_cond_update,
        latent_cvae_mmdit_noisy_causal=args.latent_cvae_mmdit_noisy_causal,
        latent_cvae_mmdit_noisy_logit_gate=args.latent_cvae_mmdit_noisy_logit_gate,
        latent_cvae_mmdit_residual_scale_max=args.latent_cvae_mmdit_residual_scale_max,
        latent_cvae_mmdit_source_route_delta_max=args.latent_cvae_mmdit_source_route_delta_max,
        latent_cvae_mmdit_noisy_correction_min=args.latent_cvae_mmdit_noisy_correction_min,
        latent_cvae_mmdit_noisy_correction_max=args.latent_cvae_mmdit_noisy_correction_max,
        latent_cvae_mmdit_noisy_correction_power=args.latent_cvae_mmdit_noisy_correction_power,
        latent_cvae_mmdit_noisy_correction_logit_delta=(
            args.latent_cvae_mmdit_noisy_correction_logit_delta
        ),
        latent_cvae_mmdit_controller_modulation_scale=(
            args.latent_cvae_mmdit_controller_modulation_scale
        ),
        latent_cvae_mmdit_evidence_scale=args.latent_cvae_mmdit_evidence_scale,
        latent_cvae_mmdit_noisy_scale=args.latent_cvae_mmdit_noisy_scale,
        latent_cvae_mmdit_operator_capacity=args.latent_cvae_mmdit_operator_capacity,
        latent_cvae_mmdit_operator_rank=args.latent_cvae_mmdit_operator_rank,
        latent_cvae_mmdit_operator_groups=args.latent_cvae_mmdit_operator_groups,
        latent_cvae_mmdit_operator_depth_logit_init=(
            args.latent_cvae_mmdit_operator_depth_logit_init
        ),
        latent_cvae_mmdit_execution_controller=args.latent_cvae_mmdit_execution_controller,
        latent_cvae_mmdit_dynamic_block_route=args.latent_cvae_mmdit_dynamic_block_route,
        latent_cvae_mmdit_control_tokens=args.latent_cvae_mmdit_control_tokens,
        latent_cvae_mmdit_controller_depth=args.latent_cvae_mmdit_controller_depth,
        latent_cvae_mmdit_controller_heads=args.latent_cvae_mmdit_controller_heads,
        latent_cvae_mmdit_controller_ffn_expansion=(
            args.latent_cvae_mmdit_controller_ffn_expansion
        ),
        latent_cvae_mmdit_max_dwell=args.latent_cvae_mmdit_max_dwell,
        latent_cvae_mmdit_dwell_mode=args.latent_cvae_mmdit_dwell_mode,
        latent_cvae_mmdit_execution_soft_temperature=(
            args.latent_cvae_mmdit_execution_soft_temperature
        ),
        latent_cvae_mmdit_identity_candidate=args.latent_cvae_mmdit_identity_candidate,
        latent_cvae_mmdit_terminal_prior_weight=(
            args.latent_cvae_mmdit_terminal_prior_weight
        ),
        latent_cvae_mmdit_execution_eval_policy=(
            args.latent_cvae_mmdit_execution_eval_policy
        ),
        latent_cvae_mmdit_execution_warmup_steps=(
            args.latent_cvae_mmdit_execution_warmup_steps
        ),
        latent_cvae_mmdit_execution_transition_steps=(
            args.latent_cvae_mmdit_execution_transition_steps
        ),
        latent_cvae_progress_action_isolation=args.latent_cvae_progress_action_isolation,
        latent_cvae_horizon_tokens=args.latent_cvae_horizon_tokens,
        latent_cvae_workspace_noisy_query=args.latent_cvae_workspace_noisy_query,
        latent_cvae_workspace_trajectory_source=args.latent_cvae_workspace_trajectory_source,
        latent_cvae_workspace_global_sources=args.latent_cvae_workspace_global_sources,
        latent_cvae_workspace_layer_source=args.latent_cvae_workspace_layer_source,
        latent_cvae_workspace_progress_value=args.latent_cvae_workspace_progress_value,
        latent_cvae_workspace_time_state=args.latent_cvae_workspace_time_state,
        latent_cvae_workspace_slot_time_state=args.latent_cvae_workspace_slot_time_state,
        latent_cvae_workspace_slot_time_scale=args.latent_cvae_workspace_slot_time_scale,
        latent_cvae_workspace_controller=args.latent_cvae_workspace_controller,
        latent_cvae_hierarchical_workspace=args.latent_cvae_hierarchical_workspace,
        latent_cvae_stage_slots=args.latent_cvae_stage_slots,
        latent_cvae_stage_promote_scale_init=args.latent_cvae_stage_promote_scale_init,
        hierarchical_mmdit_depth=args.hierarchical_mmdit_depth,
        hierarchical_mmdit_refine_steps=args.hierarchical_mmdit_refine_steps,
        hierarchical_mmdit_low_slots=args.hierarchical_mmdit_low_slots,
        hierarchical_mmdit_stage_slots=args.hierarchical_mmdit_stage_slots,
        hierarchical_mmdit_ffn_expansion=args.hierarchical_mmdit_ffn_expansion,
        hierarchical_mmdit_layer_grad_scale=args.hierarchical_mmdit_layer_grad_scale,
        hierarchical_mmdit_source_grad_scale=args.hierarchical_mmdit_source_grad_scale,
        hierarchical_mmdit_consequence_scale_init=args.hierarchical_mmdit_consequence_scale_init,
        hierarchical_mmdit_consequence_scale_max=args.hierarchical_mmdit_consequence_scale_max,
        hierarchical_mmdit_noisy_causal=args.hierarchical_mmdit_noisy_causal,
        hierarchical_mmdit_noisy_gate_min=args.hierarchical_mmdit_noisy_gate_min,
        hierarchical_mmdit_noisy_gate_power=args.hierarchical_mmdit_noisy_gate_power,
        hierarchical_mmdit_stage_promote_scale_init=args.hierarchical_mmdit_stage_promote_scale_init,
        hierarchical_mmdit_output_init_std=args.hierarchical_mmdit_output_init_std,
        hierarchical_mmdit_operator_stages=args.hierarchical_mmdit_operator_stages,
        hierarchical_mmdit_operator_rank=args.hierarchical_mmdit_operator_rank,
        hierarchical_mmdit_operator_groups=args.hierarchical_mmdit_operator_groups,
        hierarchical_mmdit_operator_depth_logit_init=args.hierarchical_mmdit_operator_depth_logit_init,
        hierarchical_mmdit_exit_logit_init=args.hierarchical_mmdit_exit_logit_init,
        hierarchical_mmdit_operator_contraction_warmup_steps=args.hierarchical_mmdit_operator_contraction_warmup_steps,
        hierarchical_mmdit_operator_contraction_transition_steps=args.hierarchical_mmdit_operator_contraction_transition_steps,
        hierarchical_mmdit_unified_controller=args.hierarchical_mmdit_unified_controller,
        hierarchical_mmdit_control_tokens=args.hierarchical_mmdit_control_tokens,
        hierarchical_mmdit_controller_depth=args.hierarchical_mmdit_controller_depth,
        hierarchical_mmdit_controller_heads=args.hierarchical_mmdit_controller_heads,
        hierarchical_mmdit_controller_ffn_expansion=args.hierarchical_mmdit_controller_ffn_expansion,
        hierarchical_mmdit_spectral_state=args.hierarchical_mmdit_spectral_state,
        hierarchical_mmdit_spectral_arm_start_fraction=args.hierarchical_mmdit_spectral_arm_start_fraction,
        hierarchical_mmdit_spectral_gripper_start_fraction=args.hierarchical_mmdit_spectral_gripper_start_fraction,
        hierarchical_mmdit_spectral_temperature=args.hierarchical_mmdit_spectral_temperature,
        hierarchical_mmdit_spectral_schedule_power=args.hierarchical_mmdit_spectral_schedule_power,
        hierarchical_mmdit_spectral_controller_shift_limit=args.hierarchical_mmdit_spectral_controller_shift_limit,
        hierarchical_mmdit_spectral_competition_loss_weight=args.hierarchical_mmdit_spectral_competition_loss_weight,
        hierarchical_mmdit_spectral_competition_warmup_steps=args.hierarchical_mmdit_spectral_competition_warmup_steps,
        hierarchical_mmdit_operation_candidate_probes=args.hierarchical_mmdit_operation_candidate_probes,
        hierarchical_mmdit_operation_value_warmup_steps=args.hierarchical_mmdit_operation_value_warmup_steps,
        hierarchical_mmdit_dwell_mode=args.hierarchical_mmdit_dwell_mode,
        hierarchical_mmdit_execution_contract=args.hierarchical_mmdit_execution_contract,
        hierarchical_mmdit_schedule_mode=args.hierarchical_mmdit_schedule_mode,
        hierarchical_mmdit_random_prefix_probability=args.hierarchical_mmdit_random_prefix_probability,
        hierarchical_mmdit_exhaustion_mode=args.hierarchical_mmdit_exhaustion_mode,
        hierarchical_mmdit_action_response_thresholds=tuple(
            args.hierarchical_mmdit_action_response_thresholds
        ),
        hierarchical_mmdit_stage_pressure_thresholds=tuple(
            args.hierarchical_mmdit_stage_pressure_thresholds
        ),
        hierarchical_mmdit_action_response_floor=args.hierarchical_mmdit_action_response_floor,
        hierarchical_mmdit_exhaustion_confirm_steps=args.hierarchical_mmdit_exhaustion_confirm_steps,
        hierarchical_mmdit_residual_scale_init=args.hierarchical_mmdit_residual_scale_init,
        hierarchical_mmdit_residual_scale_max=args.hierarchical_mmdit_residual_scale_max,
        hierarchical_mmdit_output_contract=args.hierarchical_mmdit_output_contract,
        hierarchical_mmdit_noisy_market_bias=args.hierarchical_mmdit_noisy_market_bias,
        hierarchical_mmdit_noisy_gate_mode=args.hierarchical_mmdit_noisy_gate_mode,
        latent_cvae_z_probe=args.latent_cvae_z_probe,
        adaptive_cvae_refine_steps=args.adaptive_cvae_refine_steps,
        adaptive_cvae_progress_memory=args.adaptive_cvae_progress_memory,
        adaptive_cvae_progress_steps=args.adaptive_cvae_progress_steps,
        adaptive_cvae_prefix_memory=args.adaptive_cvae_prefix_memory,
        adaptive_cvae_layer_routing=args.adaptive_cvae_layer_routing,
        adaptive_cvae_route_cosine=args.adaptive_cvae_route_cosine,
        adaptive_cvae_route_temperature=args.adaptive_cvae_route_temperature,
        adaptive_cvae_prefix_detach=args.adaptive_cvae_prefix_detach,
        adaptive_cvae_progress_z_injection=args.adaptive_cvae_progress_z_injection,
        adaptive_cvae_route_query_bias=args.adaptive_cvae_route_query_bias,
        adaptive_cvae_route_time_query=args.adaptive_cvae_route_time_query,
        adaptive_cvae_token_semantic_adapter=args.adaptive_cvae_token_semantic_adapter,
        adaptive_cvae_output_adapter=args.adaptive_cvae_output_adapter,
        adaptive_cvae_context_dropout=args.adaptive_cvae_context_dropout,
        adaptive_cvae_route_entropy_floor_ratio=args.adaptive_cvae_route_entropy_floor_ratio,
        adaptive_cvae_function_adapters=args.adaptive_cvae_function_adapters,
        adaptive_cvae_function_rank=args.adaptive_cvae_function_rank,
        adaptive_cvae_progress_role_dim=args.adaptive_cvae_progress_role_dim,
        adaptive_cvae_route_topk=args.adaptive_cvae_route_topk,
        adaptive_cvae_route_sparsemax=args.adaptive_cvae_route_sparsemax,
        adaptive_cvae_route_adaptive_temperature=args.adaptive_cvae_route_adaptive_temperature,
        adaptive_cvae_route_min_temperature=args.adaptive_cvae_route_min_temperature,
        adaptive_cvae_route_max_temperature=args.adaptive_cvae_route_max_temperature,
        adaptive_cvae_role_query=args.adaptive_cvae_role_query,
        adaptive_cvae_step_roles=args.adaptive_cvae_step_roles,
        adaptive_cvae_coarse_stride=args.adaptive_cvae_coarse_stride,
        adaptive_cvae_coarse_strength=args.adaptive_cvae_coarse_strength,
        adaptive_cvae_seed_scale=args.adaptive_cvae_seed_scale,
        adaptive_cvae_output_scale=args.adaptive_cvae_output_scale,
        adaptive_cvae_context_capsules=args.adaptive_cvae_context_capsules,
        adaptive_cvae_context_capsule_count=args.adaptive_cvae_context_capsule_count,
        adaptive_cvae_direct_condition_residual=args.adaptive_cvae_direct_condition_residual,
        adaptive_cvae_condition_strength=args.adaptive_cvae_condition_strength,
        adaptive_cvae_condition_strength_min=args.adaptive_cvae_condition_strength_min,
        adaptive_cvae_condition_strength_max=args.adaptive_cvae_condition_strength_max,
        adaptive_cvae_condition_strength_init=args.adaptive_cvae_condition_strength_init,
        adaptive_cvae_micro_control=args.adaptive_cvae_micro_control,
        adaptive_cvae_micro_refine_block=args.adaptive_cvae_micro_refine_block,
        adaptive_cvae_micro_supervision=args.adaptive_cvae_micro_supervision,
        adaptive_cvae_micro_heun=args.adaptive_cvae_micro_heun,
        adaptive_cvae_micro_monotonic_progress=args.adaptive_cvae_micro_monotonic_progress,
        adaptive_cvae_micro_min_step=args.adaptive_cvae_micro_min_step,
        adaptive_cvae_micro_max_step=args.adaptive_cvae_micro_max_step,
        adaptive_cvae_micro_step_init=args.adaptive_cvae_micro_step_init,
        adaptive_cvae_micro_kp_max=args.adaptive_cvae_micro_kp_max,
        adaptive_cvae_micro_kp_init=args.adaptive_cvae_micro_kp_init,
        adaptive_cvae_micro_kd_max=args.adaptive_cvae_micro_kd_max,
        adaptive_cvae_micro_kd_init=args.adaptive_cvae_micro_kd_init,
        adaptive_cvae_micro_update_scale=args.adaptive_cvae_micro_update_scale,
        adaptive_cvae_micro_refine_block_scale=args.adaptive_cvae_micro_refine_block_scale,
        adaptive_cvae_micro_progress_distance_scale=args.adaptive_cvae_micro_progress_distance_scale,
    )
    system = V39PolicySystem(policy_config)
    if goal_language_tokens is not None and goal_language_mask is not None:
        system.set_default_goal_language(goal_language_tokens, goal_language_mask)
    stage1_checkpoint = (
        args.stage1_checkpoint if int(args.stage1_initialization_enabled) else None
    )
    if int(args.require_flow_jepa_stage1_checkpoint) and stage1_checkpoint is None:
        raise ValueError(
            "--require-flow-jepa-stage1-checkpoint=1 requires --stage1-checkpoint"
        )
    if stage1_checkpoint is not None:
        stage_payload = torch.load(stage1_checkpoint, map_location="cpu", weights_only=False)
        if stage_payload.get("schema") not in POLICY_CHECKPOINT_SCHEMAS:
            raise ValueError("--stage1-checkpoint must be a V39/V40 checkpoint")
        if int(args.require_flow_jepa_stage1_checkpoint):
            _validate_flow_jepa_stage1_checkpoint(
                stage_payload,
                policy_config=policy_config,
                goal_language_metadata=goal_language_metadata,
            )
        stage_state, skipped_stage_keys = _filter_stage1_state_dict(
            stage_payload["model"],
            reset_dirty_adapters=bool(args.stage1_reset_dirty_adapters),
        )
        if skipped_stage_keys:
            print(
                f"[v39-init] skipped dirty stage1 adapter keys: "
                f"{skipped_stage_keys[:8]} count={len(skipped_stage_keys)}",
                flush=True,
            )
        stage_state, skipped_parseval_keys = _filter_parseval_replaced_state_dict(
            stage_state,
            enabled=str(args.gripper_field_mode) == "parseval_temporal",
        )
        if skipped_parseval_keys:
            print(
                f"[v39-init] skipped replaced Parseval-interface keys: "
                f"{skipped_parseval_keys[:8]} count={len(skipped_parseval_keys)}",
                flush=True,
            )
        stage_state, skipped_v74_time_keys = _filter_v74_time_controller_state_dict(
            stage_state,
            enabled=bool(
                int(args.latent_cvae_workspace_time_state)
                or int(args.latent_cvae_workspace_controller)
                or int(args.adaptive_cvae_route_time_query)
            ),
        )
        if skipped_v74_time_keys:
            print(
                f"[v39-init] skipped V74 time/controller stage1 keys: "
                f"{skipped_v74_time_keys[:8]} count={len(skipped_v74_time_keys)}",
                flush=True,
            )
        if str(args.final_action_decoder) in {
            "hierarchical_mmdit_action",
            "evidence_latent_mmdit_action",
        }:
            # A stage1 checkpoint supplies the trunk/contracts only.  Both the
            # historical CVAE tower and the short-lived competitive clean
            # decoder have incompatible ownership semantics and must start
            # fresh; use --resume, not --stage1-checkpoint, to continue a
            # checkpoint produced by this exact serial architecture.
            obsolete_prefixes = (
                "planner.latent_cvae_action_decoder.",
                "planner.hierarchical_mmdit_action_decoder.",
                "planner.evidence_latent_mmdit_action_decoder.",
            )
            skipped_obsolete_decoder = [
                key for key in stage_state if key.startswith(obsolete_prefixes)
            ]
            if skipped_obsolete_decoder:
                stage_state = {
                    key: value
                    for key, value in stage_state.items()
                    if not key.startswith(obsolete_prefixes)
                }
                print(
                    f"[v39-init] skipped stage1 final-decoder keys: "
                    f"{skipped_obsolete_decoder[:8]} count={len(skipped_obsolete_decoder)}",
                    flush=True,
                )
        stage_state, skipped_shape_keys = _filter_shape_mismatched_state_dict(
            stage_state, system.state_dict()
        )
        if skipped_shape_keys:
            print(
                f"[v39-init] skipped shape-mismatched stage1 keys: "
                f"{skipped_shape_keys[:8]} count={len(skipped_shape_keys)}",
                flush=True,
            )
        missing, unexpected = system.load_state_dict(stage_state, strict=False)
        if unexpected:
            raise ValueError(f"unexpected keys while loading --stage1-checkpoint: {unexpected[:8]}")
        if missing:
            print(
                f"[v39-init] missing keys from stage1 checkpoint: {missing[:8]} count={len(missing)}",
                flush=True,
            )
    trainer = V39PolicyTrainerConfig(
        **{name: getattr(args, name) for name in V39PolicyTrainerConfig.__dataclass_fields__}
    )
    required_model_contract = _validate_required_model_contract(
        os.environ.get("CLEARVLA_REQUIRED_MODEL_CONTRACT"),
        policy_config,
        trainer,
    )
    context = {
        "schema": "clearvla-v40-1-unified-intervention-latent-context-v1",
        "source_fingerprint": _source_fingerprint(),
        "args": vars(args),
        "legacy_context_checkpoint": None
        if args.legacy_context_checkpoint is None
        else str(args.legacy_context_checkpoint),
        "stage1_checkpoint": None if stage1_checkpoint is None else str(stage1_checkpoint),
        "stage1_initialization_enabled": bool(int(args.stage1_initialization_enabled)),
        "required_model_contract": required_model_contract,
        "architecture_manifest": (
            OBJECT_INTENT_DYNAMICS_MANIFEST.as_dict()
            if int(policy_config.flow_jepa_object_intent_dynamics_mainline)
            else (
                GROUNDING_MANIFEST.as_dict()
                if int(
                    policy_config.flow_jepa_grounded_intent_effect_mainline
                )
                else None
            )
        ),
        "splits": {"train": train_ids, "val": val_ids, "test": test_ids},
        "dataset": asdict(dataset_config),
        "visual_geometry": visual_geometry,
        "goal_language": goal_language_metadata,
        "policy_model": asdict(policy_config),
        "trainer": asdict(trainer),
        "parameter_report": system.parameter_report(),
        "performance_contract": {
            "v95_experiment_stage": str(args.training_stage),
            "flow_jepa_stage1_objective": flow_jepa_stage1,
            "forward_contract": (
                "representation_only" if flow_jepa_stage1 else "action_policy"
            ),
            "target_action_conditioned": not flow_jepa_stage1,
            "final_action_decoder_executed": not flow_jepa_stage1,
            "layer_contracts_executed": bool(
                not flow_jepa_stage1
                and not int(
                    policy_config.flow_jepa_object_intent_dynamics_mainline
                )
            ),
            "prefetch_dinov2_tokens": bool(use_token_prefetch),
            "information_balanced_sampling": information_sampling_summary,
            "target_future_encoding": (
                "multihorizon_spatial_evidence_last_history_only"
                if int(args.flow_jepa_enabled) and int(args.flow_jepa_late_bottleneck)
                else "sparse_window_offsets_then_stage_last_history_only"
                if int(args.flow_jepa_enabled)
                else "future_anchors_last_history_only"
            ),
            "future_target_is_input": False,
            "flow_jepa_enabled": bool(int(args.flow_jepa_enabled)),
            "action_history_enabled": bool(int(args.action_history_enabled)),
            "action_history_offsets": list(dataset_config.executed_action_offsets),
            "action_history_tokens": int(policy_config.action_history_token_count),
            "action_history_condition_dropout": float(
                policy_config.action_history_condition_dropout
            ),
            "action_history_condition_exact_null": bool(
                int(policy_config.action_history_condition_exact_null)
            ),
            "action_history_proposal_detach": bool(
                int(policy_config.action_history_proposal_detach)
            ),
            "goal_conditioning_enabled": bool(int(args.goal_conditioning_enabled)),
            "goal_token_count": int(args.goal_token_count),
            "goal_condition_dropout": float(args.goal_condition_dropout),
            "goal_condition_exact_null": bool(
                int(policy_config.goal_condition_exact_null)
            ),
            "stateless_phase_enabled": bool(
                int(policy_config.stateless_phase_enabled)
            ),
            "stateless_phase_count": int(policy_config.stateless_phase_count),
            "stateless_phase_query_scale": float(
                policy_config.stateless_phase_query_scale
            ),
            "goal_language_source": "precomputed_t5_condition",
            "goal_encoder_resident_during_training": False,
            "flow_jepa_future_only_teacher": bool(int(args.flow_jepa_enabled)),
            "flow_jepa_directed_canvas_attention": bool(
                int(args.flow_jepa_directed_canvas_attention)
            ),
            "flow_jepa_selector_value_separated": bool(int(args.flow_jepa_enabled)),
            "flow_jepa_late_bottleneck": bool(int(args.flow_jepa_late_bottleneck)),
            "flow_jepa_coarse_to_fine": bool(int(args.flow_jepa_late_bottleneck)),
            "flow_jepa_fine_radius": int(args.flow_jepa_fine_radius),
            "flow_jepa_reader_radius": int(args.flow_jepa_reader_radius),
            "flow_jepa_reader_heads": int(args.flow_jepa_reader_heads),
            "flow_jepa_rgb_backbone": bool(int(args.flow_jepa_raw_image_enabled)),
            "flow_jepa_raw_image_enabled": bool(int(args.flow_jepa_raw_image_enabled)),
            "flow_jepa_raw_pyramid_channels": int(args.flow_jepa_raw_base_channels),
            "flow_jepa_raw_flow_dino_to_local": bool(
                int(args.flow_jepa_raw_image_enabled)
            ),
            "flow_jepa_raw_detail_read_after_grounding": bool(
                int(args.flow_jepa_raw_image_enabled)
            ),
            "flow_jepa_raw_reader_radius": int(args.flow_jepa_raw_reader_radius),
            "flow_jepa_raw_reader_heads": int(args.flow_jepa_raw_reader_heads),
            "flow_jepa_raw_activation_checkpoint": bool(
                int(args.flow_jepa_raw_activation_checkpoint)
            ),
            "flow_jepa_zero_flow_guard": bool(int(args.flow_jepa_zero_flow_guard)),
            "flow_jepa_strict_role_visual_path": bool(
                int(args.flow_jepa_strict_role_visual_path)
            ),
            "flow_jepa_complementary_raw_detail": bool(
                int(args.flow_jepa_complementary_raw_detail)
            ),
            "flow_jepa_source_aligned_raw_fusion": bool(
                int(args.flow_jepa_source_aligned_raw_fusion)
            ),
            "flow_jepa_teacher_balanced_target_mask": bool(
                int(args.flow_jepa_teacher_balanced_target_mask)
            ),
            "flow_jepa_predictive_change_contract": bool(
                int(args.flow_jepa_predictive_change_contract)
            ),
            "flow_jepa_horizon_soft_address": bool(
                int(args.flow_jepa_horizon_soft_address)
            ),
            "flow_jepa_horizon_address_update_scale": float(
                args.flow_jepa_horizon_address_update_scale
            ),
            "flow_jepa_future_reliable_normalization": bool(
                int(args.flow_jepa_future_reliable_normalization)
            ),
            "flow_jepa_horizon_address_loss_weight": float(
                args.flow_jepa_horizon_address_loss_weight
            ),
            "flow_jepa_teacher_target_mask_quotas": {
                "past": float(args.flow_jepa_teacher_mask_past_fraction),
                "future_change": float(args.flow_jepa_teacher_mask_change_fraction),
                "uniform": float(
                    1.0
                    - args.flow_jepa_teacher_mask_past_fraction
                    - args.flow_jepa_teacher_mask_change_fraction
                ),
            },
            "flow_jepa_horizon_balance_mode": str(
                args.flow_jepa_horizon_balance_mode
            ),
            "action_horizon_weight_mode": str(args.horizon_weight_mode),
            "flow_jepa_single_stage_uniform_role_lr": bool(
                int(args.single_stage_role_lr)
            ),
            "flow_jepa_raw_motion_evidence": (
                "fixed_rgb_census"
                if int(args.flow_jepa_zero_flow_guard)
                else "predicted_flow_reliability"
            ),
            "flow_jepa_raw_fallback": (
                "additive_low_frequency_content"
                if int(args.flow_jepa_complementary_raw_detail)
                else "pooled_content"
                if int(args.flow_jepa_zero_flow_guard)
                else "identity_local_candidates"
            ),
            "flow_jepa_raw_seed_source": (
                "dino_identity_centered"
                if int(args.flow_jepa_raw_image_enabled)
                else None
            ),
            "flow_jepa_raw_value_amplitude_gate": False,
            "flow_jepa_single_stage_end_to_end": bool(
                int(args.flow_jepa_raw_image_enabled)
                and not int(args.stage1_initialization_enabled)
            ),
            "flow_jepa_role_groups": (
                [
                    int(args.flow_jepa_grounding_blocks),
                    int(args.flow_jepa_world_blocks),
                    int(args.flow_jepa_policy_blocks),
                ]
                if int(args.flow_jepa_role_hierarchy)
                else None
            ),
            "flow_jepa_policy_workspace_scale": float(
                args.flow_jepa_policy_workspace_scale
            ),
            "flow_jepa_policy_workspace_fixed_fusion": bool(
                int(args.flow_jepa_policy_workspace_fixed_fusion)
            ),
            "flow_jepa_world_anchor_write_only": bool(
                int(args.flow_jepa_world_anchor_write_only)
            ),
            "flow_jepa_late_policy_detail": bool(
                int(args.flow_jepa_late_policy_detail)
            ),
            "flow_jepa_late_policy_detail_scale": float(
                args.flow_jepa_late_policy_detail_scale
            ),
            "flow_jepa_soft_address_lattice": bool(
                int(args.flow_jepa_soft_address_lattice)
            ),
            "flow_jepa_address_slots": int(args.flow_jepa_address_slots),
            "flow_jepa_address_route_dim": int(
                args.flow_jepa_address_route_dim
            ),
            "flow_jepa_address_query_chunk": int(
                args.flow_jepa_address_query_chunk
            ),
            "flow_jepa_policy_multi_glimpse_address": bool(
                int(args.flow_jepa_policy_multi_glimpse_address)
            ),
            "flow_jepa_address_flow_prior_floor": float(
                args.flow_jepa_address_flow_prior_floor
            ),
            "flow_jepa_bounded_flow_coordinates": bool(
                int(args.flow_jepa_bounded_flow_coordinates)
            ),
            "flow_jepa_sequential_horizon_memory": bool(
                int(args.flow_jepa_sequential_horizon_memory)
            ),
            "flow_jepa_horizon_cell_fine_address": bool(
                int(args.flow_jepa_horizon_cell_fine_address)
            ),
            "flow_jepa_online_horizon_address": bool(
                int(args.flow_jepa_online_horizon_address)
            ),
            "flow_jepa_progressive_grounding_address": bool(
                int(args.flow_jepa_progressive_grounding_address)
            ),
            "flow_jepa_coordinate_typed_raw_detail": bool(
                int(args.flow_jepa_coordinate_typed_raw_detail)
            ),
            "flow_jepa_structured_ownership_bottleneck": bool(
                int(args.flow_jepa_structured_ownership_bottleneck)
            ),
            "flow_jepa_pre_value_owner_routing": bool(
                int(args.flow_jepa_pre_value_owner_routing)
            ),
            "flow_jepa_pre_value_owner_update_scale": float(
                args.flow_jepa_pre_value_owner_update_scale
            ),
            "flow_jepa_functional_mainline_routing": bool(
                int(args.flow_jepa_functional_mainline_routing)
            ),
            "flow_jepa_utility_precision_mainline": bool(
                int(args.flow_jepa_utility_precision_mainline)
            ),
            "flow_jepa_action_free_world_factual": bool(
                int(args.flow_jepa_action_free_world_factual)
            ),
            "flow_jepa_shared_factual_glimpse_bank": bool(
                int(args.flow_jepa_shared_factual_glimpse_bank)
            ),
            "flow_jepa_g_aligned_future_effect": bool(
                int(args.flow_jepa_g_aligned_future_effect)
            ),
            "flow_jepa_stateless_goal_phase_machine": bool(
                int(args.flow_jepa_stateless_goal_phase_machine)
            ),
            "flow_jepa_top_role_schedule": str(
                args.flow_jepa_top_role_schedule
            ),
            "flow_jepa_policy_plan_compiler": bool(
                int(args.flow_jepa_policy_plan_compiler)
            ),
            "flow_jepa_supervised_effect_mainline": bool(
                int(args.flow_jepa_supervised_effect_mainline)
            ),
            "flow_jepa_stateless_intent_controller": bool(
                int(args.flow_jepa_stateless_intent_controller)
            ),
            "flow_jepa_window_effect_bank": bool(
                int(args.flow_jepa_window_effect_bank)
            ),
            "flow_jepa_future_slots": int(args.flow_jepa_future_slots),
            "flow_jepa_effect_read_in_p2": bool(
                int(args.flow_jepa_effect_read_in_p2)
            ),
            "flow_jepa_differential_intent_effect_mainline": bool(
                int(args.flow_jepa_differential_intent_effect_mainline)
            ),
            "flow_jepa_grounded_intent_effect_mainline": bool(
                int(args.flow_jepa_grounded_intent_effect_mainline)
            ),
            "flow_jepa_object_intent_dynamics_mainline": bool(
                int(args.flow_jepa_object_intent_dynamics_mainline)
            ),
            "flow_matching_time_distribution": str(
                args.flow_matching_time_distribution
            ),
            "flow_jepa_teacher_g_ema_decay": float(
                args.flow_jepa_teacher_g_ema_decay
            ),
            "flow_jepa_address_query_batch_budget": int(
                args.flow_jepa_address_query_batch_budget
            ),
            "flow_jepa_microgrid_tile": int(args.flow_jepa_microgrid_tile),
            "flow_jepa_p1_mixed_precision": bool(
                int(args.flow_jepa_p1_mixed_precision)
            ),
            "flow_jepa_checkpoint_min_batch": int(
                args.flow_jepa_checkpoint_min_batch
            ),
            "flow_jepa_raw_micro_grid": int(args.flow_jepa_raw_micro_grid),
            "flow_jepa_variance_safe_routing": bool(
                int(args.flow_jepa_variance_safe_routing)
            ),
            "flow_jepa_complete_numerical_contract": bool(
                int(args.flow_jepa_complete_numerical_contract)
            ),
            "flow_jepa_routing_norm_floor": float(
                args.flow_jepa_routing_norm_floor
            ),
            "flow_jepa_correlation_rms_floor": float(
                args.flow_jepa_correlation_rms_floor
            ),
            "flow_jepa_visibility_transition_fraction": float(
                args.flow_jepa_visibility_transition_fraction
            ),
            "flow_jepa_interval_stage_typed_value": bool(
                int(args.flow_jepa_interval_stage_typed_value)
            ),
            "role_attnres_enabled": bool(int(args.role_attnres_enabled)),
            "role_attnres_key_dim": int(args.role_attnres_key_dim),
            "role_attnres_ground_to_world": bool(
                int(args.role_attnres_ground_to_world)
            ),
            "role_attnres_world_to_policy": bool(
                int(args.role_attnres_world_to_policy)
            ),
            "role_attnres_policy_to_mmdit": bool(
                int(args.role_attnres_policy_to_mmdit)
            ),
            "role_attnres_ground_to_world_scale": float(
                args.role_attnres_ground_to_world_scale
            ),
            "role_attnres_world_to_policy_scale": float(
                args.role_attnres_world_to_policy_scale
            ),
            "role_attnres_policy_to_mmdit_scale": float(
                args.role_attnres_policy_to_mmdit_scale
            ),
            "role_residual_amplitude_contract": bool(
                int(args.role_residual_amplitude_contract)
            ),
            "role_residual_max_update_rms": float(
                args.role_residual_max_update_rms
            ),
            "role_attnres_max_value_rms": float(
                args.role_attnres_max_value_rms
            ),
            "role_residual_contract_after_gate": bool(
                int(args.role_residual_contract_after_gate)
            ),
            "flow_jepa_policy_workspace_horizon_pool": bool(
                int(args.flow_jepa_policy_workspace_horizon_pool)
            ),
            "flow_jepa_policy_workspace_is_action_stream": bool(
                int(args.flow_jepa_role_hierarchy)
            ),
            "flow_jepa_history_offsets": list(dataset_config.history_offsets),
            "flow_jepa_window_offsets": list(policy_config.flow_jepa_effective_window_offsets),
            "flow_jepa_stage_offset": (
                0
                if int(args.flow_jepa_late_bottleneck)
                else int(policy_config.flow_jepa_effective_stage_offset)
            ),
            "flow_jepa_stage_target": (
                "none_far_horizon_is_spatial_evidence"
                if int(args.flow_jepa_late_bottleneck)
                else "camera_2x2_frozen_dino_delta_no_grad"
            ),
            "flow_jepa_stage_conditions_window": not bool(
                int(args.flow_jepa_late_bottleneck)
            ),
            "rollout_dynamics_bound": True,
            "controlled_residual_dynamics": True,
            "weak_visual_base_plus_action_delta": True,
            "counterfactual_delta_contrast": not flow_jepa_stage1,
            "tail_action_reads_controlled_delta": not flow_jepa_stage1,
            "final_action_decoder": str(args.final_action_decoder),
            "gripper_field_mode": str(args.gripper_field_mode),
            "gripper_field_dim": int(args.gripper_field_dim),
            "gripper_parseval_shared_native_noise": str(args.gripper_field_mode)
            == "parseval_temporal",
            "gripper_parseval_all_channels_decode": str(args.gripper_field_mode)
            == "parseval_temporal",
            "arm_flow_mode": str(args.arm_flow_mode),
            "arm_noise_temporal_rho": float(args.arm_noise_temporal_rho),
            "arm_source_mode": str(args.arm_source_mode),
            "arm_source_scale": float(args.arm_source_scale),
            "arm_source_scale_active": str(args.arm_source_mode) == "boundary_multiscale",
            "arm_source_component_weights": {
                "innovation": float(args.arm_source_innovation_weight),
                "velocity": float(args.arm_source_velocity_weight),
                "acceleration": float(args.arm_source_acceleration_weight),
            },
            "arm_source_condition_inputs": ["action_state"],
            "arm_source_target_conditioned": False,
            "arm_source_controller_conditioned": False,
            "arm_source_single_rng_draw": True,
            "action_normalizer_fingerprint": _normalizer_fingerprint(action_norm),
            "arm_manifold_native_noise": str(args.arm_flow_mode) == "manifold_native",
            "arm_manifold_projected_sampling": str(args.arm_flow_mode) == "manifold_native",
            "residual_action_flow_safe_start": str(args.final_action_decoder)
            in {"residual_action_flow", "layered_residual_action_flow"},
            "residual_action_flow_uses_v37_high_action_event_tokens": str(args.final_action_decoder)
            in {"residual_action_flow", "layered_residual_action_flow"},
            "layered_residual_action_flow_layer_pair_injection": str(args.final_action_decoder)
            == "layered_residual_action_flow",
            "latent_main_action_decoder": str(args.final_action_decoder) == "latent_main_action",
            "latent_main_action_single_final_path": str(args.final_action_decoder)
            == "latent_main_action",
            "latent_main_action_every_layer_summary_injected": str(args.final_action_decoder)
            == "latent_main_action",
            "latent_main_action_no_legacy_velocity_base": str(args.final_action_decoder)
            == "latent_main_action",
            "latent_main_action_horizon_dependent_depth": bool(
                int(args.latent_action_temporal_depth)
            )
            and str(args.final_action_decoder) == "latent_main_action",
            "latent_cvae_action_decoder": str(args.final_action_decoder)
            in {"latent_cvae_action", "adaptive_recurrent_cvae_action"},
            "latent_cvae_single_final_path": str(args.final_action_decoder)
            in {"latent_cvae_action", "adaptive_recurrent_cvae_action"},
            "latent_cvae_no_legacy_velocity_base": str(args.final_action_decoder)
            in {"latent_cvae_action", "adaptive_recurrent_cvae_action"},
            "adaptive_recurrent_cvae_action_decoder": str(args.final_action_decoder)
            == "adaptive_recurrent_cvae_action",
            "hierarchical_mmdit_action_decoder": str(args.final_action_decoder)
            == "hierarchical_mmdit_action",
            "evidence_latent_mmdit_action_decoder": str(args.final_action_decoder)
            == "evidence_latent_mmdit_action",
            "evidence_latent_mmdit_native_time": str(args.final_action_decoder)
            == "evidence_latent_mmdit_action",
            "evidence_latent_mmdit_no_posterior": str(args.final_action_decoder)
            == "evidence_latent_mmdit_action",
            "evidence_latent_mmdit_no_dct": str(args.final_action_decoder)
            == "evidence_latent_mmdit_action",
            "evidence_latent_mmdit_read_only_evidence": str(args.final_action_decoder)
            == "evidence_latent_mmdit_action",
            "evidence_latent_mmdit_distinct_blocks": str(args.final_action_decoder)
            == "evidence_latent_mmdit_action",
            "hierarchical_mmdit_native_time_input_chart": (
                str(args.final_action_decoder) == "hierarchical_mmdit_action"
                and not bool(int(args.hierarchical_mmdit_spectral_state))
                and (
                    str(args.arm_flow_mode) == "manifold_native"
                    or str(args.gripper_field_mode) == "parseval_temporal"
                )
            ),
            "hierarchical_mmdit_native_time_tangent_output": (
                str(args.final_action_decoder) == "hierarchical_mmdit_action"
                and not bool(int(args.hierarchical_mmdit_spectral_state))
                and (
                    str(args.arm_flow_mode) == "manifold_native"
                    or str(args.gripper_field_mode) == "parseval_temporal"
                )
            ),
            "hierarchical_mmdit_native_time_position_alignment": (
                str(args.final_action_decoder) == "hierarchical_mmdit_action"
                and not bool(int(args.hierarchical_mmdit_spectral_state))
                and (
                    str(args.arm_flow_mode) == "manifold_native"
                    or str(args.gripper_field_mode) == "parseval_temporal"
                )
            ),
            "hierarchical_mmdit_field_coordinates_outside_token_network": (
                str(args.final_action_decoder) == "hierarchical_mmdit_action"
                and not bool(int(args.hierarchical_mmdit_spectral_state))
                and str(args.arm_flow_mode) == "manifold_native"
                and str(args.gripper_field_mode) == "parseval_temporal"
            ),
            "hierarchical_mmdit_deterministic_intent_contracts": str(args.final_action_decoder)
            == "hierarchical_mmdit_action",
            "hierarchical_mmdit_no_target_or_posterior_path": str(args.final_action_decoder)
            == "hierarchical_mmdit_action",
            "hierarchical_mmdit_owned_five_role_evidence": str(args.final_action_decoder)
            == "hierarchical_mmdit_action",
            "hierarchical_mmdit_fixed_role_prior": (
                str(args.final_action_decoder) == "hierarchical_mmdit_action"
                and not bool(int(args.hierarchical_mmdit_unified_controller))
            ),
            "hierarchical_mmdit_unified_role_selector": (
                str(args.final_action_decoder) == "hierarchical_mmdit_action"
                and bool(int(args.hierarchical_mmdit_unified_controller))
            ),
            "hierarchical_mmdit_workspace_controller_token_interface": (
                str(args.final_action_decoder) == "hierarchical_mmdit_action"
                and bool(int(args.hierarchical_mmdit_unified_controller))
            ),
            "hierarchical_mmdit_workspace_controller_value_firewall": (
                str(args.final_action_decoder) == "hierarchical_mmdit_action"
                and bool(int(args.hierarchical_mmdit_unified_controller))
            ),
            "hierarchical_mmdit_manager_selector_only": str(args.final_action_decoder)
            == "hierarchical_mmdit_action",
            "hierarchical_mmdit_mmdit_owns_final_consumption": str(args.final_action_decoder)
            == "hierarchical_mmdit_action",
            "hierarchical_mmdit_serial_condition_composition": str(args.final_action_decoder)
            == "hierarchical_mmdit_action",
            "hierarchical_mmdit_competitive_condition_market": False,
            "hierarchical_mmdit_identifiable_residual_gates": str(args.final_action_decoder)
            == "hierarchical_mmdit_action",
            "hierarchical_mmdit_conditional_low_rank_operators": False,
            "hierarchical_mmdit_stage_low_rank_adapters": False,
            "hierarchical_mmdit_stage_nested_contraction": str(args.final_action_decoder)
            == "hierarchical_mmdit_action",
            "hierarchical_mmdit_contraction_sidecar": str(args.final_action_decoder)
            == "hierarchical_mmdit_action",
            "hierarchical_mmdit_post_gate_sidecar": str(args.final_action_decoder)
            == "hierarchical_mmdit_action",
            "hierarchical_mmdit_shared_amplitude_owner": str(args.final_action_decoder)
            == "hierarchical_mmdit_action",
            "hierarchical_mmdit_duplicate_amplitude_owner": False,
            "hierarchical_mmdit_host_update_amplitude_owner": str(args.final_action_decoder)
            == "hierarchical_mmdit_action",
            "hierarchical_mmdit_unified_update_amplitude_owner": False,
            "hierarchical_mmdit_unified_relative_update_keep_owner": False,
            "hierarchical_mmdit_shared_full_rank_path": False,
            "hierarchical_mmdit_distinct_full_rank_path": str(args.final_action_decoder)
            == "hierarchical_mmdit_action",
            "hierarchical_mmdit_step_conditioned_full_rank": str(args.final_action_decoder)
            == "hierarchical_mmdit_action",
            "hierarchical_mmdit_shared_base_scale_identifiable": False,
            "hierarchical_mmdit_shared_base_bias_free": False,
            "hierarchical_mmdit_scale_invariant_base_no_decay": False,
            "hierarchical_mmdit_mandatory_operator_writeback": False,
            "hierarchical_mmdit_block_state_normalized": str(args.final_action_decoder)
            == "hierarchical_mmdit_action",
            "hierarchical_mmdit_factor_cache_per_forward": str(args.final_action_decoder)
            == "hierarchical_mmdit_action",
            "hierarchical_mmdit_dynamic_orthogonal_baseline": str(args.final_action_decoder)
            == "hierarchical_mmdit_action",
            "hierarchical_mmdit_architecture_version": policy_config.hierarchical_mmdit_architecture_version,
            "hierarchical_mmdit_operator_stages": int(args.hierarchical_mmdit_operator_stages),
            "hierarchical_mmdit_operator_rank": int(args.hierarchical_mmdit_operator_rank),
            "hierarchical_mmdit_operator_groups": int(args.hierarchical_mmdit_operator_groups),
            "hierarchical_mmdit_operator_boundary_identity": str(args.final_action_decoder)
            == "hierarchical_mmdit_action",
            "hierarchical_mmdit_operator_nested_path": str(args.final_action_decoder)
            == "hierarchical_mmdit_action",
            "hierarchical_mmdit_operator_continuous_depth": str(args.final_action_decoder)
            == "hierarchical_mmdit_action",
            "hierarchical_mmdit_operator_nonexpansive": str(args.final_action_decoder)
            == "hierarchical_mmdit_action",
            "hierarchical_mmdit_operator_post_contraction_renorm": False,
            "hierarchical_mmdit_unified_controller": bool(
                int(args.hierarchical_mmdit_unified_controller)
            ),
            "hierarchical_mmdit_unified_operator_depth_owner": bool(
                int(args.hierarchical_mmdit_unified_controller)
            ),
            "hierarchical_mmdit_control_tokens": int(args.hierarchical_mmdit_control_tokens),
            "hierarchical_mmdit_operation_candidate_probes": bool(
                int(args.hierarchical_mmdit_operation_candidate_probes)
            ),
            "hierarchical_mmdit_operation_value_warmup_steps": int(
                args.hierarchical_mmdit_operation_value_warmup_steps
            ),
            "hierarchical_mmdit_dwell_mode": str(args.hierarchical_mmdit_dwell_mode),
            "hierarchical_mmdit_execution_contract": str(
                args.hierarchical_mmdit_execution_contract
            ),
            "hierarchical_mmdit_controller_owns_residual_amplitude": (
                str(args.hierarchical_mmdit_execution_contract) == "legacy_stage_keep"
            ),
            "hierarchical_mmdit_value_candidates_are_blocks": (
                str(args.hierarchical_mmdit_execution_contract) == "typed_block_budget"
            ),
            "hierarchical_mmdit_controller_depth": int(args.hierarchical_mmdit_controller_depth),
            "hierarchical_mmdit_controller_heads": int(args.hierarchical_mmdit_controller_heads),
            "hierarchical_mmdit_controller_ffn_expansion": float(
                args.hierarchical_mmdit_controller_ffn_expansion
            ),
            "hierarchical_mmdit_spectral_state": bool(int(args.hierarchical_mmdit_spectral_state)),
            "hierarchical_mmdit_spectral_bridge": bool(int(args.hierarchical_mmdit_spectral_state)),
            "hierarchical_mmdit_spectral_velocity_target": bool(
                int(args.hierarchical_mmdit_spectral_state)
            ),
            "hierarchical_mmdit_spectral_integrator": bool(
                int(args.hierarchical_mmdit_spectral_state)
            ),
            "hierarchical_mmdit_spectral_tangent_head": bool(
                int(args.hierarchical_mmdit_spectral_state)
            ),
            "hierarchical_mmdit_spectral_physical_view_only": bool(
                int(args.hierarchical_mmdit_spectral_state)
            ),
            "hierarchical_mmdit_spectral_arm_start_fraction": float(
                args.hierarchical_mmdit_spectral_arm_start_fraction
            ),
            "hierarchical_mmdit_spectral_gripper_start_fraction": float(
                args.hierarchical_mmdit_spectral_gripper_start_fraction
            ),
            "hierarchical_mmdit_spectral_temperature": float(
                args.hierarchical_mmdit_spectral_temperature
            ),
            "hierarchical_mmdit_spectral_schedule_power": float(
                args.hierarchical_mmdit_spectral_schedule_power
            ),
            "hierarchical_mmdit_spectral_controller_shift_limit": float(
                args.hierarchical_mmdit_spectral_controller_shift_limit
            ),
            "hierarchical_mmdit_spectral_competition_loss_weight": float(
                args.hierarchical_mmdit_spectral_competition_loss_weight
            ),
            "hierarchical_mmdit_spectral_competition_warmup_steps": int(
                args.hierarchical_mmdit_spectral_competition_warmup_steps
            ),
            "hierarchical_mmdit_schedule_mode": str(args.hierarchical_mmdit_schedule_mode),
            "hierarchical_mmdit_exhaustion_mode": str(args.hierarchical_mmdit_exhaustion_mode),
            "hierarchical_mmdit_distinct_blocks": int(args.hierarchical_mmdit_depth),
            "hierarchical_mmdit_full_rank_block_count": int(args.hierarchical_mmdit_depth),
            "hierarchical_mmdit_shared_core_count": 0,
            "hierarchical_mmdit_operator_stage_count": int(args.hierarchical_mmdit_operator_stages),
            "hierarchical_mmdit_refine_block_count": int(args.hierarchical_mmdit_depth),
            "hierarchical_mmdit_max_refine_steps": int(args.hierarchical_mmdit_refine_steps),
            "latent_cvae_mmdit_decoder": bool(int(args.latent_cvae_mmdit_decoder)),
            "latent_cvae_mmdit_no_direct_noisy_residual": bool(int(args.latent_cvae_mmdit_decoder)),
            "latent_cvae_mmdit_condition_update": bool(int(args.latent_cvae_mmdit_cond_update)),
            "latent_cvae_mmdit_noisy_causal": bool(int(args.latent_cvae_mmdit_noisy_causal)),
            "latent_cvae_horizon_tokens": int(args.latent_cvae_horizon_tokens),
            "latent_cvae_workspace_noisy_query": bool(int(args.latent_cvae_workspace_noisy_query)),
            "latent_cvae_workspace_trajectory_source": bool(
                int(args.latent_cvae_workspace_trajectory_source)
            ),
            "latent_cvae_workspace_global_sources": bool(
                int(args.latent_cvae_workspace_global_sources)
            ),
            "latent_cvae_workspace_layer_source": bool(
                int(args.latent_cvae_workspace_layer_source)
            ),
            "latent_cvae_workspace_progress_value": bool(
                int(args.latent_cvae_workspace_progress_value)
            ),
            "latent_cvae_workspace_time_state": bool(int(args.latent_cvae_workspace_time_state)),
            "latent_cvae_workspace_slot_time_state": bool(
                int(args.latent_cvae_workspace_slot_time_state)
            ),
            "latent_cvae_workspace_slot_time_scale": float(
                args.latent_cvae_workspace_slot_time_scale
            ),
            "latent_cvae_workspace_controller": bool(int(args.latent_cvae_workspace_controller)),
            "latent_cvae_hierarchical_workspace": bool(
                int(args.latent_cvae_hierarchical_workspace)
            ),
            "latent_cvae_stage_slots": int(args.latent_cvae_stage_slots),
            "latent_cvae_stage_promote_scale_init": float(
                args.latent_cvae_stage_promote_scale_init
            ),
            "latent_cvae_z_primary_denoising": bool(int(args.latent_cvae_mmdit_decoder)),
            "latent_cvae_typed_evidence_workspace": bool(int(args.latent_cvae_mmdit_decoder)),
            "latent_cvae_single_workspace_action_write": bool(int(args.latent_cvae_mmdit_decoder))
            and not bool(int(args.latent_cvae_hierarchical_workspace)),
            "latent_cvae_workspace_no_action_feedback": bool(
                int(args.latent_cvae_hierarchical_workspace)
            ),
            "latent_cvae_stage_to_low_selector_only": bool(
                int(args.latent_cvae_hierarchical_workspace)
            ),
            "latent_cvae_stage_role_content_separated": bool(
                int(args.latent_cvae_hierarchical_workspace)
            ),
            "adaptive_cvae_refine_steps": int(args.adaptive_cvae_refine_steps),
            "adaptive_cvae_progress_memory": int(args.adaptive_cvae_progress_memory),
            "adaptive_cvae_progress_steps": int(args.adaptive_cvae_progress_steps),
            "adaptive_cvae_prefix_memory": int(args.adaptive_cvae_prefix_memory),
            "adaptive_cvae_layer_routing": int(args.adaptive_cvae_layer_routing),
            "adaptive_cvae_route_cosine": int(args.adaptive_cvae_route_cosine),
            "adaptive_cvae_route_temperature": float(args.adaptive_cvae_route_temperature),
            "adaptive_cvae_prefix_detach": int(args.adaptive_cvae_prefix_detach),
            "adaptive_cvae_progress_z_injection": int(args.adaptive_cvae_progress_z_injection),
            "adaptive_cvae_route_query_bias": int(args.adaptive_cvae_route_query_bias),
            "adaptive_cvae_route_time_query": int(args.adaptive_cvae_route_time_query),
            "adaptive_cvae_token_semantic_adapter": int(args.adaptive_cvae_token_semantic_adapter),
            "adaptive_cvae_output_adapter": int(args.adaptive_cvae_output_adapter),
            "adaptive_cvae_context_dropout": float(args.adaptive_cvae_context_dropout),
            "adaptive_cvae_route_entropy_floor_ratio": float(
                args.adaptive_cvae_route_entropy_floor_ratio
            ),
            "adaptive_cvae_function_adapters": int(args.adaptive_cvae_function_adapters),
            "adaptive_cvae_function_rank": int(args.adaptive_cvae_function_rank),
            "adaptive_cvae_progress_role_dim": int(args.adaptive_cvae_progress_role_dim),
            "adaptive_cvae_route_topk": int(args.adaptive_cvae_route_topk),
            "adaptive_cvae_route_sparsemax": int(args.adaptive_cvae_route_sparsemax),
            "adaptive_cvae_route_adaptive_temperature": int(
                args.adaptive_cvae_route_adaptive_temperature
            ),
            "adaptive_cvae_route_min_temperature": float(args.adaptive_cvae_route_min_temperature),
            "adaptive_cvae_route_max_temperature": float(args.adaptive_cvae_route_max_temperature),
            "adaptive_cvae_role_query": int(args.adaptive_cvae_role_query),
            "adaptive_cvae_step_roles": int(args.adaptive_cvae_step_roles),
            "adaptive_cvae_coarse_stride": int(args.adaptive_cvae_coarse_stride),
            "adaptive_cvae_coarse_strength": float(args.adaptive_cvae_coarse_strength),
            "adaptive_cvae_seed_scale": float(args.adaptive_cvae_seed_scale),
            "adaptive_cvae_output_scale": float(args.adaptive_cvae_output_scale),
            "adaptive_cvae_context_capsules": int(args.adaptive_cvae_context_capsules),
            "adaptive_cvae_context_capsule_count": int(args.adaptive_cvae_context_capsule_count),
            "adaptive_cvae_direct_condition_residual": int(
                args.adaptive_cvae_direct_condition_residual
            ),
            "adaptive_cvae_condition_strength": int(args.adaptive_cvae_condition_strength),
            "adaptive_cvae_condition_strength_min": float(
                args.adaptive_cvae_condition_strength_min
            ),
            "adaptive_cvae_condition_strength_max": float(
                args.adaptive_cvae_condition_strength_max
            ),
            "adaptive_cvae_condition_strength_init": float(
                args.adaptive_cvae_condition_strength_init
            ),
            "adaptive_cvae_micro_control": int(args.adaptive_cvae_micro_control),
            "adaptive_cvae_micro_refine_block": int(args.adaptive_cvae_micro_refine_block),
            "adaptive_cvae_micro_supervision": int(args.adaptive_cvae_micro_supervision),
            "adaptive_cvae_micro_heun": int(args.adaptive_cvae_micro_heun),
            "adaptive_cvae_micro_monotonic_progress": int(
                args.adaptive_cvae_micro_monotonic_progress
            ),
            "adaptive_cvae_micro_min_step": float(args.adaptive_cvae_micro_min_step),
            "adaptive_cvae_micro_max_step": float(args.adaptive_cvae_micro_max_step),
            "adaptive_cvae_micro_step_init": float(args.adaptive_cvae_micro_step_init),
            "adaptive_cvae_micro_kp_max": float(args.adaptive_cvae_micro_kp_max),
            "adaptive_cvae_micro_kp_init": float(args.adaptive_cvae_micro_kp_init),
            "adaptive_cvae_micro_kd_max": float(args.adaptive_cvae_micro_kd_max),
            "adaptive_cvae_micro_kd_init": float(args.adaptive_cvae_micro_kd_init),
            "adaptive_cvae_micro_update_scale": float(args.adaptive_cvae_micro_update_scale),
            "adaptive_cvae_micro_refine_block_scale": float(
                args.adaptive_cvae_micro_refine_block_scale
            ),
            "adaptive_cvae_micro_progress_distance_scale": float(
                args.adaptive_cvae_micro_progress_distance_scale
            ),
            "latent_cvae_z_dim": int(args.latent_cvae_z_dim),
            "latent_cvae_decoder_depth": int(args.latent_cvae_decoder_depth),
            "latent_cvae_layer_memory": int(args.latent_cvae_layer_memory),
            "latent_cvae_transition_memory": int(args.latent_cvae_transition_memory),
            "latent_cvae_transition_detach": int(args.latent_cvae_transition_detach),
            "latent_cvae_layer_grad_scale": float(args.latent_cvae_layer_grad_scale),
            "latent_cvae_visual_memory": int(args.latent_cvae_visual_memory),
            "latent_cvae_context_memory": int(args.latent_cvae_context_memory),
            "latent_cvae_mu_bound": float(args.latent_cvae_mu_bound),
            "latent_cvae_min_std": float(args.latent_cvae_min_std),
            "latent_cvae_causal_attention": int(args.latent_cvae_causal_attention),
            "latent_cvae_noisy_gate": int(args.latent_cvae_noisy_gate),
            "latent_cvae_noisy_gate_min": float(args.latent_cvae_noisy_gate_min),
            "latent_cvae_noisy_gate_power": float(args.latent_cvae_noisy_gate_power),
            "latent_cvae_layer_scan": int(args.latent_cvae_layer_scan),
            "latent_cvae_layer_scan_alpha": float(args.latent_cvae_layer_scan_alpha),
            "latent_cvae_mmdit_depth": int(args.latent_cvae_mmdit_depth),
            "latent_cvae_mmdit_cond_update": int(args.latent_cvae_mmdit_cond_update),
            "latent_cvae_mmdit_residual_scale_max": float(
                args.latent_cvae_mmdit_residual_scale_max
            ),
            "latent_cvae_mmdit_noisy_correction_min": float(
                args.latent_cvae_mmdit_noisy_correction_min
            ),
            "latent_cvae_mmdit_noisy_correction_max": float(
                args.latent_cvae_mmdit_noisy_correction_max
            ),
            "latent_cvae_mmdit_noisy_correction_power": float(
                args.latent_cvae_mmdit_noisy_correction_power
            ),
            "latent_cvae_mmdit_noisy_correction_logit_delta": float(
                args.latent_cvae_mmdit_noisy_correction_logit_delta
            ),
            "latent_cvae_mmdit_controller_modulation_scale": float(
                args.latent_cvae_mmdit_controller_modulation_scale
            ),
            "latent_cvae_mmdit_operator_capacity": int(
                args.latent_cvae_mmdit_operator_capacity
            ),
            "latent_cvae_mmdit_operator_rank": int(args.latent_cvae_mmdit_operator_rank),
            "latent_cvae_mmdit_operator_groups": int(args.latent_cvae_mmdit_operator_groups),
            "latent_cvae_mmdit_operator_depth_logit_init": float(
                args.latent_cvae_mmdit_operator_depth_logit_init
            ),
            "latent_cvae_mmdit_execution_controller": int(
                args.latent_cvae_mmdit_execution_controller
            ),
            "latent_cvae_mmdit_dynamic_block_route": int(
                args.latent_cvae_mmdit_dynamic_block_route
            ),
            "latent_cvae_mmdit_control_tokens": int(args.latent_cvae_mmdit_control_tokens),
            "latent_cvae_mmdit_controller_depth": int(
                args.latent_cvae_mmdit_controller_depth
            ),
            "latent_cvae_mmdit_controller_heads": int(args.latent_cvae_mmdit_controller_heads),
            "latent_cvae_mmdit_controller_ffn_expansion": float(
                args.latent_cvae_mmdit_controller_ffn_expansion
            ),
            "latent_cvae_mmdit_max_dwell": int(args.latent_cvae_mmdit_max_dwell),
            "latent_cvae_mmdit_dwell_mode": str(args.latent_cvae_mmdit_dwell_mode),
            "latent_cvae_mmdit_identity_candidate": int(
                args.latent_cvae_mmdit_identity_candidate
            ),
            "latent_cvae_mmdit_terminal_prior_weight": float(
                args.latent_cvae_mmdit_terminal_prior_weight
            ),
            "latent_cvae_mmdit_execution_eval_policy": str(
                args.latent_cvae_mmdit_execution_eval_policy
            ),
            "latent_cvae_mmdit_execution_warmup_steps": int(
                args.latent_cvae_mmdit_execution_warmup_steps
            ),
            "latent_cvae_mmdit_execution_transition_steps": int(
                args.latent_cvae_mmdit_execution_transition_steps
            ),
            "latent_action_decoder_depth": int(args.latent_action_decoder_depth),
            "latent_action_layer_schedule": str(args.latent_action_layer_schedule),
            "latent_action_visual_memory": int(args.latent_action_visual_memory),
            "latent_action_context_memory": int(args.latent_action_context_memory),
            "latent_action_transition_memory": int(args.latent_action_transition_memory),
            "latent_action_event_gripper_gate": int(args.latent_action_event_gripper_gate),
            "latent_action_temporal_depth": int(args.latent_action_temporal_depth),
            "latent_action_near_steps": int(args.latent_action_near_steps),
            "latent_action_mid_steps": int(args.latent_action_mid_steps),
            "latent_action_near_depth": int(args.latent_action_near_depth),
            "latent_action_mid_depth": int(args.latent_action_mid_depth),
            "action_flow_residual_depth": int(args.action_flow_residual_depth),
            "action_flow_residual_max_scale": float(args.action_flow_residual_max_scale),
            "action_flow_residual_layer_pair_schedule": str(
                args.action_flow_residual_layer_pair_schedule
            ),
            "action_flow_residual_layer_detach": bool(args.action_flow_residual_layer_detach),
            "action_flow_residual_stage_router": bool(args.action_flow_residual_stage_router),
            "midcut_layer": int(args.midcut_layer),
            "layer_contract_adapters": int(args.layer_contract_adapters),
            "staged_midcut_contract": True,
            "v39_1_multi_layer_adapter_contract": bool(args.layer_contract_adapters),
            "v39_2_multi_layer_latent_head_shared_fm_probe": bool(args.layer_shared_fm_probe),
            "v39_3_recurrent_milestone_consequence": bool(args.layer_recurrent_consequence),
            "v40_layer_causal_latent_split": False,
            "v40_1_unified_intervention_latent": True,
            "v40_context_fusion_at_zero_feedback_depth": True,
            "v40_milestone_anchor_alignment": "required_equal",
            "v40_state_counterfactual_enabled": bool(args.layer_state_counterfactual),
            "action_consequence_self_condition": bool(args.action_consequence_self_condition),
            "layer_zero_base_diagnostic": bool(args.layer_zero_base_diagnostic),
            "v40_hard_action_negative": "within_batch_farthest_action",
            "v39_3_consequence_steps": int(args.layer_consequence_steps),
            "v40_causal_feedback_depth": int(args.layer_causal_feedback_depth),
            "metric_sync": "log_every_and_epoch_end",
            "dino_cache_reads": "episode_grouped_mmap",
            "data_loader_seed": int(args.seed),
            "training_rng_reset_after_initialization": not bool(args.resume),
        },
        "skipped": skipped,
        "policy_contract": (
            "V40.1 is independent of old world checkpoints by default. It keeps the V39 staged training contract but uses one unified intervention-latent head at every supervised layer. "
            "Explicit state/history/action context is fused into the FiLM/gate path even when feedback depth is zero. Counterfactual contrast uses hold-action and within-batch "
            "farthest-action negatives; the non-strict state/frame shuffle remains disabled by default."
        ),
    }
    print_context(context)
    if not args.resume:
        # Model construction and architecture-specific preflight setup consume
        # different numbers of random draws.  Resetting here makes fresh V96/
        # V98 controls share the same dropout, flow-time and noise stream; the
        # DataLoader has its own generator and therefore cannot be perturbed by
        # either architecture.
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
    train_v39_policy(
        system=system,
        train_loader=train_loader,
        val_loader=val_loader,
        conditioner=conditioner,
        device=device,
        dtype=dtype,
        camera_names=cameras,
        action_normalizer=action_norm,
        state_normalizer=state_norm,
        trainer=trainer,
        out_dir=args.out_dir,
        context=context,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
