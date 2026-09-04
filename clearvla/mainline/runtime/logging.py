"""Active-semantic logging without ancestry aliases or per-batch CUDA sync."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor

ACTIVE_PREFIXES = (
    "loss_",
    "gradient_",
    "condition_",
    "observation_",
    "grounding_",
    "object_grounding_",
    "object_intent_",
    "object_plan_",
    "object_coarse_",
    "object_action_",
    "object_teacher_",
    "object_w_",
    "object_w1_",
    "object_w2_",
    "object_future_",
    "p1_",
    "object_p2_",
    "object_consequence_",
    "object_p3_",
    "bottom_",
    # Native V120 Evidence-MMDiT/controller metrics retain their established
    # names so old and recovered runs remain directly comparable.
    "evidence_",
    "validation_",
    "runtime_",
    "training_",
    "action_",
    "event_",
    "motion_",
    "gripper_private_",
    "flow_",
    "sampling_",
    "controlled_transition_",
    "history_action_",
    "learning_rate_",
)


def archival_metrics(values: Mapping[str, float]) -> dict[str, float]:
    """Keep every active scalar, including exact zeros, for the JSONL record.

    Absence and exact zero are different experimental facts.  In particular,
    a collapsed W/P2/flow path must remain visible in the lossless archive even
    though the compact nohup projection may omit an ordinary zero.
    """

    return {
        name: float(value)
        for name, value in values.items()
        if name.startswith(ACTIVE_PREFIXES)
        or name in {"learning_rate", "gradient_global_preclip_l2"}
    }


def active_metrics(values: Mapping[str, float], *, zero_tolerance: float = 0.0) -> dict[str, float]:
    """Build the compact console projection of active metrics.

    Historical ancestry counters should disappear when inactive.  In contrast,
    a zero owner gradient, mass error or non-expansive violation is itself a
    decision-critical observation and must not be made indistinguishable from
    a metric that was never wired.
    """

    result: dict[str, float] = {}
    for name, value in archival_metrics(values).items():
        scalar = float(value)
        visible_zero = (
            name == "loss_ledger_gap"
            or name == "loss_contribution_gap"
            or name.startswith("gradient_")
            or name.endswith("_nonexpansive_violation")
            or name.endswith("_mass_conservation_error")
            or name.endswith("_identity_error")
            or name.endswith("_has_null")
            or name.endswith("_null_mass")
            or name
            in {
                "object_grounding_mass_conservation_error",
                "grounding_clean_endpoint_t_v120",
            }
        )
        if abs(scalar) <= float(zero_tolerance) and not visible_zero:
            continue
        result[name] = scalar
    return result


@dataclass
class MetricAccumulator:
    sums: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)

    def update(self, values: Mapping[str, float], *, weight: float = 1.0) -> None:
        for name, value in values.items():
            self.sums[name] = self.sums.get(name, 0.0) + float(value) * float(weight)
            self.weights[name] = self.weights.get(name, 0.0) + float(weight)

    def means(self) -> dict[str, float]:
        return {
            name: value / max(self.weights.get(name, 0.0), 1e-12)
            for name, value in self.sums.items()
        }


@dataclass
class DeviceMetricAccumulator:
    """Accumulate detached scalars with one vector update per key signature.

    A Python loop that updates one CUDA tensor per metric avoids host
    synchronization but still launches tens of tiny kernels twice per train
    batch (window and epoch ledgers).  Active loss keys are stable on ordinary
    batches and diagnostics add only a second signature, so vector ownership
    removes that hidden logging tax without moving values to the CPU.
    """

    sums: dict[tuple[str, ...], Tensor] = field(default_factory=dict)
    weights: dict[tuple[str, ...], float] = field(default_factory=dict)

    def update(self, values: Mapping[str, Tensor], *, weight: float = 1.0) -> None:
        scalar_weight = float(weight)
        scalar_rows = sorted(
            (name, value.detach().float()) for name, value in values.items() if value.ndim == 0
        )
        if not scalar_rows:
            return
        names = tuple(name for name, _ in scalar_rows)
        vector = torch.stack([value for _, value in scalar_rows])
        if names in self.sums:
            self.sums[names] = self.sums[names] + vector * scalar_weight
        else:
            self.sums[names] = vector * scalar_weight
        self.weights[names] = self.weights.get(names, 0.0) + scalar_weight

    def materialize(self) -> dict[str, float]:
        scalar_sums: dict[str, Tensor] = {}
        scalar_weights: dict[str, float] = {}
        for names, vector in self.sums.items():
            weight = self.weights[names]
            for index, name in enumerate(names):
                value = vector[index]
                scalar_sums[name] = scalar_sums.get(name, value.new_zeros(())) + value
                scalar_weights[name] = scalar_weights.get(name, 0.0) + weight
        return tensor_scalars(
            {name: value / max(scalar_weights[name], 1e-12) for name, value in scalar_sums.items()}
        )


class JsonlRunLogger:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / "metrics.jsonl"

    def write(self, kind: str, **payload: object) -> None:
        row = {"kind": str(kind), **payload}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")

    @staticmethod
    def compact_line(
        kind: str,
        *,
        epoch: int,
        batch: int | None,
        step: int,
        metrics: Mapping[str, float],
    ) -> str:
        lead = f"[mainline-{kind}] epoch={epoch:03d}"
        if batch is not None:
            lead += f" batch={batch:04d}"
        lead += f" step={step}"
        if kind == "train":
            # This is the frequent health row.  Keep it task-facing and short;
            # exact objective ownership and source-path localization live in
            # the less-frequent decision rows below and every scalar remains in
            # metrics.jsonl.
            priority = (
                "loss_total",
                "loss_action_flow",
                "loss_action_flow_native",
                "loss_action_flow_first",
                "loss_action_flow_first8",
                "loss_action_flow_tail",
                "loss_action_flow_band_1_4",
                "loss_action_flow_band_5_12",
                "loss_action_flow_band_13_24",
                "loss_decoded_action",
                "gradient_window_preclip_l2_mean",
                "gradient_window_preclip_l2_max",
                "runtime_window_seconds_per_batch",
                "learning_rate",
            )
        elif kind == "val":
            # Validation is emitted only once per epoch.  Put the two truthful
            # chart spaces first: normalized values are directly comparable
            # across outlets, while source-native values preserve the producer
            # chart without claiming SI units.  The historical ``*_physical``
            # aliases remain at the end solely for dashboard compatibility.
            priority = (
                "validation_action_rmse_normalized",
                "validation_first_rmse_normalized",
                "validation_first8_rmse_normalized",
                "validation_tail_rmse_normalized",
                "validation_tail_first_ratio_normalized",
                "validation_band_1_4_rmse_normalized",
                "validation_band_5_12_rmse_normalized",
                "validation_band_13_24_rmse_normalized",
                "validation_arm_rmse_normalized",
                "validation_gripper_rmse_normalized",
                "validation_action_rmse_source_native",
                "validation_first_rmse_source_native",
                "validation_first8_rmse_source_native",
                "validation_tail_rmse_source_native",
                "validation_tail_first_ratio_source_native",
                "validation_band_1_4_rmse_source_native",
                "validation_band_5_12_rmse_source_native",
                "validation_band_13_24_rmse_source_native",
                "validation_arm_rmse_source_native",
                "validation_gripper_rmse_source_native",
                "validation_codec_gripper_boundary_qpos_gap_rms_normalized",
                "validation_codec_gripper_boundary_qpos_gap_rms_source_native",
                "validation_action_state_gripper_abs_gt3_rate_normalized",
                "validation_action_state_gripper_abs_gt5_rate_normalized",
                "validation_gripper_command_accuracy",
                "validation_gripper_command_precision",
                "validation_gripper_command_recall",
                "validation_gripper_command_f1",
                "validation_gripper_command_rmse_physical",
                # Compatibility aliases; keep them visible only after the
                # normalized/source-native decision metrics above.
                "validation_action_rmse_physical",
                "validation_first_rmse_physical",
                "validation_first8_rmse_physical",
                "validation_tail_rmse_physical",
                "validation_tail_first_ratio_physical",
                "validation_band_1_4_rmse_physical",
                "validation_band_5_12_rmse_physical",
                "validation_band_13_24_rmse_physical",
                "validation_arm_rmse_physical",
                "validation_gripper_rmse_physical",
            )
        else:
            priority = ()
        fields: list[str] = []
        for name in priority:
            if name in metrics:
                fields.append(f"{name}={metrics[name]:.6g}")
        return " ".join((lead, *fields))

    @staticmethod
    def diagnostic_lines(
        kind: str,
        *,
        epoch: int,
        batch: int | None,
        step: int,
        metrics: Mapping[str, float],
    ) -> tuple[str, ...]:
        """Return a bounded decision panel for captured console logs.

        ``metrics.jsonl`` remains the complete archival record.  Console rows
        are deliberately limited to quantities that can change a continue,
        stop, or localization decision; source-level state dumps and repeated
        aliases do not belong here.
        """

        if kind == "train":
            groups: tuple[tuple[str, tuple[str, ...]], ...] = (
                (
                    "objective",
                    (
                        "loss_group_action",
                        "loss_group_representation",
                        "loss_group_execution",
                        "loss_contrib_action_flow",
                        "loss_contrib_decoded_action",
                        "loss_contrib_gripper_command",
                        "loss_contrib_gripper_trajectory",
                        "loss_contrib_motion",
                        "loss_contrib_future_dynamics",
                        "loss_contrib_future_transition",
                        "loss_contrib_object_reconstruction",
                        "loss_contrib_flow_warp",
                        "loss_contrib_execution_value",
                        "loss_gripper_trajectory_transition",
                        "loss_gripper_trajectory_persistence",
                        "loss_gripper_trajectory_transition_mask_fraction",
                        "loss_gripper_trajectory_persistence_mask_fraction",
                        "loss_ledger_gap",
                        "loss_contribution_gap",
                    ),
                ),
                (
                    "path",
                    (
                        "object_grounding_object_content_pair_cosine",
                        "object_intent_public_interval_variation",
                        "object_intent_semantic_interval_variation",
                        "object_intent_geometry_interval_variation",
                        "object_w_physical_action_condition_rms",
                        "object_w_physical_action_delta_rms",
                        "object_w_physical_action_interval_variation",
                        "object_w_goal_direct_ingress",
                        "object_w_coarse_hidden_direct_ingress",
                        "object_action_world_initial_materialization_count",
                        "object_action_world_initial_tag_identity_error",
                        "object_action_world_refinement_count",
                        "object_action_world_refinement_pre_action_interval_rms",
                        "object_action_world_refinement_post_action_interval_rms",
                        "object_action_world_refinement_action_interval_delta_rms",
                        "object_action_world_refinement_pre_semantic_delta_rms",
                        "object_action_world_refinement_post_semantic_delta_rms",
                        "object_action_world_refinement_semantic_delta_change_rms",
                        "object_action_world_refinement_pre_transport_rms",
                        "object_action_world_refinement_post_transport_rms",
                        "object_action_world_refinement_transport_change_rms",
                        "object_action_world_refinement_tag_identity_error",
                        "object_w2_interval_adjacent_cosine",
                        "loss_future_semantic_common",
                        "loss_future_semantic_innovation",
                        "loss_future_transition",
                        "p1_spatial_variation",
                        "p1_policy_query_residual_rms",
                        "object_p2_semantic_effect_rms",
                        "object_p2_geometry_effect_rms",
                        "object_p2_geometry_address_correction_rms",
                        "object_p2_effect_precontract_rms",
                        "object_p2_effect_postcontract_rms",
                        "object_p3_protected_policy_precision_rms",
                        "object_p3_temporal_rms",
                        "object_p3_state_change_rms",
                        "bottom_action_rms",
                    ),
                ),
                (
                    "numerics",
                    (
                        "flow_jepa_address_coarse_variance_min",
                        "flow_jepa_address_coarse_std_gain_max",
                        "flow_jepa_progressive_g2_input_variance_min",
                        "flow_jepa_progressive_g2_input_std_gain_max",
                        "flow_jepa_progressive_g2_aligned_variance_min",
                        "flow_jepa_progressive_g2_correction_std_gain_max",
                        "object_grounding_mass_conservation_error",
                        "object_p2_spatial_selector_has_null",
                        "object_p2_terminal_has_null",
                        "object_p2_spatial_common_residual_identity_error",
                        "object_p2_terminal_common_residual_identity_error",
                        "controlled_transition_trajectory_norm_denominator_min",
                        "controlled_transition_trajectory_norm_gain",
                        "object_w_typed_norm_denominator_min",
                        "object_w_typed_norm_gain_max",
                        "object_w_typed_norm_output_input_rms_ratio_max",
                        "bottom_capacity_mean",
                        "evidence_mmd_it_removed_channel_fraction",
                        "evidence_mmd_it_contraction_ratio",
                        "sampling_outer_world_refinement",
                        "sampling_outer_proposal_action_rms",
                        "sampling_outer_refined_action_rms",
                        "sampling_outer_refined_action_delta_rms",
                        "sampling_outer_final_world_action_interval_rms",
                        "sampling_outer_final_world_action_delta_rms",
                        "sampling_outer_final_world_action_interval_mismatch_rms",
                        "sampling_outer_final_world_action_delta_mismatch_rms",
                    ),
                ),
                (
                    "b-spine",
                    (
                        "bottom_spine_coarse_field_rms",
                        "bottom_spine_detail_field_rms",
                        "bottom_spine_coarse_token_rms",
                        "bottom_spine_detail_token_rms",
                        "bottom_spine_update_rms",
                        "bottom_spine_raw_token_rms",
                        "bottom_spine_to_raw_token_rms_ratio",
                        "bottom_spine_decomposition_max_abs",
                        "gradient_raw_bottom_spine_coarse_l2",
                        "gradient_raw_bottom_spine_detail_l2",
                    ),
                ),
                (
                    "grad-owner",
                    (
                        "gradient_raw_observation_l2",
                        "gradient_raw_grounding_l2",
                        "gradient_raw_grounder_l2",
                        "gradient_raw_intent_l2",
                        "gradient_raw_dynamics_l2",
                        "gradient_raw_p1_factual_l2",
                        "gradient_raw_p2_effect_reader_l2",
                        "gradient_raw_consequence_l2",
                        "gradient_raw_p3_compiler_l2",
                        "gradient_raw_controlled_transition_l2",
                        "gradient_raw_bottom_mmdit_l2",
                        "gradient_raw_bottom_execution_l2",
                        "gradient_raw_bottom_heads_l2",
                        "gradient_raw_bottom_spine_l2",
                        "gradient_raw_global_l2",
                        "gradient_postlocal_global_l2",
                        "gradient_postglobal_global_l2",
                    ),
                ),
                (
                    "grad-boundary",
                    (
                        "gradient_tensor_s_public_interval_carrier_rms",
                        "gradient_tensor_s_typed_common_rms",
                        "gradient_tensor_s_typed_interval_residual_rms",
                        "gradient_tensor_p1_static_fact_rms",
                        "gradient_tensor_p1_dynamic_query_residual_rms",
                        "gradient_tensor_w2_semantic_common_rms",
                        "gradient_tensor_w2_geometry_interval_rms",
                        "gradient_tensor_w_semantic_fact_ingress_rms",
                        "gradient_tensor_w_geometry_fact_ingress_rms",
                        "gradient_tensor_w_physical_action_condition_rms",
                        "gradient_tensor_w_physical_action_carrier_rms",
                        "gradient_tensor_p2_semantic_effect_rms",
                        "gradient_tensor_p2_geometry_effect_rms",
                        "gradient_tensor_p2_geometry_address_correction_rms",
                        "gradient_tensor_p1_protected_policy_precision_rms",
                        "gradient_tensor_p3_temporal_rms",
                        "gradient_tensor_p3_state_change_rms",
                    ),
                ),
            )
        elif kind == "val":
            groups = (
                (
                    "command",
                    (
                        "validation_gripper_command_rmse_physical",
                        "validation_gripper_command_accuracy",
                        "validation_gripper_command_precision",
                        "validation_gripper_command_recall",
                        "validation_gripper_command_f1",
                        "validation_gripper_command_predicted_positive_rate",
                        "validation_gripper_command_target_positive_rate",
                    ),
                ),
                (
                    "boundary",
                    (
                        "validation_codec_gripper_boundary_qpos_gap_rms_normalized",
                        "validation_codec_gripper_boundary_qpos_gap_rms_source_native",
                        "validation_action_state_gripper_abs_gt3_rate_normalized",
                        "validation_action_state_gripper_abs_gt5_rate_normalized",
                    ),
                ),
                (
                    "gripper",
                    (
                        "validation_gripper_rmse_normalized",
                        "validation_gripper_rmse_source_native",
                        "validation_gripper_band_1_4_rmse_physical",
                        "validation_gripper_band_5_12_rmse_physical",
                        "validation_gripper_band_13_24_rmse_physical",
                        "validation_gripper_post_event_1_2_rmse_physical",
                        "validation_gripper_post_event_3_6_rmse_physical",
                        "validation_gripper_post_event_7_plus_rmse_physical",
                        "validation_gripper_post_event_rows_1_2",
                        "validation_gripper_post_event_rows_3_6",
                        "validation_gripper_post_event_rows_7_plus",
                    ),
                ),
                (
                    "gripper-branch",
                    (
                        "validation_gripper_absolute_branch_band_1_4_rmse_physical",
                        "validation_gripper_absolute_branch_band_5_12_rmse_physical",
                        "validation_gripper_absolute_branch_band_13_24_rmse_physical",
                        "validation_gripper_delta_branch_band_1_4_rmse_physical",
                        "validation_gripper_delta_branch_band_5_12_rmse_physical",
                        "validation_gripper_delta_branch_band_13_24_rmse_physical",
                        "validation_gripper_branch_disagreement_band_1_4_rms_physical",
                        "validation_gripper_branch_disagreement_band_5_12_rms_physical",
                        "validation_gripper_branch_disagreement_band_13_24_rms_physical",
                        "validation_gripper_branch_decode_identity_max_abs",
                    ),
                ),
                (
                    "events",
                    (
                        "validation_decoded_gripper_event_precision",
                        "validation_decoded_gripper_event_recall",
                        "validation_decoded_gripper_event_f1",
                        "validation_decoded_gripper_events_predicted",
                        "validation_decoded_gripper_events_target",
                        "validation_decoded_gripper_event_ratio",
                        "validation_decoded_gripper_timing_mae_steps",
                        "validation_motion_head_precision",
                        "validation_motion_head_recall",
                        "validation_motion_head_f1",
                        "validation_motion_head_events_predicted",
                        "validation_motion_head_events_target",
                        "validation_decoded_motion_precision",
                        "validation_decoded_motion_recall",
                        "validation_decoded_motion_f1",
                    ),
                ),
                (
                    "path",
                    (
                        "loss_group_action",
                        "loss_group_representation",
                        "loss_group_execution",
                        "object_grounding_object_content_pair_cosine",
                        "object_intent_public_interval_variation",
                        "object_w2_interval_adjacent_cosine",
                        "loss_future_semantic_common",
                        "loss_future_semantic_innovation",
                        "loss_future_transition",
                        "p1_spatial_variation",
                        "object_p2_semantic_effect_rms",
                        "object_p2_geometry_effect_rms",
                        "object_p2_effect_postcontract_rms",
                        "object_p3_protected_policy_precision_rms",
                        "object_p3_temporal_rms",
                        "object_p3_state_change_rms",
                        "bottom_action_rms",
                    ),
                ),
                (
                    "closure",
                    (
                        "validation_deploy_sampling_outer_world_refinement",
                        "validation_deploy_sampling_outer_proposal_action_rms",
                        "validation_deploy_sampling_outer_refined_action_rms",
                        "validation_deploy_sampling_outer_refined_action_delta_rms",
                        "validation_deploy_object_action_world_refinement_count",
                        "validation_deploy_object_action_world_refinement_action_interval_delta_rms",
                        "validation_deploy_object_action_world_refinement_semantic_delta_change_rms",
                        "validation_deploy_object_action_world_refinement_transport_change_rms",
                        "validation_deploy_sampling_outer_final_world_action_interval_mismatch_rms",
                        "validation_deploy_sampling_outer_final_world_action_delta_mismatch_rms",
                    ),
                ),
                (
                    "action-estimator-match",
                    (
                        "validation_action_estimator_match_coverage",
                        "validation_action_estimator_to_full_interval_action_rms",
                        "validation_action_estimator_to_full_interval_action_ratio_vs_coarse",
                        "validation_action_estimator_to_full_interval_delta_rms",
                        "validation_action_estimator_to_full_interval_delta_ratio_vs_coarse",
                        "validation_action_estimator_full_update_direction_cosine",
                        "validation_action_estimator_full_update_direction_valid_fraction",
                        "validation_action_estimator_to_full_semantic_rms",
                        "validation_action_estimator_to_full_semantic_ratio_vs_coarse",
                        "validation_action_estimator_to_full_transport_rms",
                        "validation_action_estimator_to_full_transport_ratio_vs_coarse",
                        "validation_action_estimator_extra_path_runtime_seconds",
                        "validation_action_estimator_extra_path_live_allocation_gib",
                    ),
                ),
                (
                    "causal",
                    (
                        "validation_sampling_diagnostic_coverage",
                        "validation_p2_intervention_coverage",
                        "validation_proposal_ablation_coverage",
                        "validation_proposal_primary_rmse_physical",
                        "validation_proposal_zero_rmse_physical",
                        "validation_proposal_zero_mse_gain_vs_primary_physical",
                        "validation_proposal_zero_action_delta_rmse_physical",
                        "validation_execution_ablation_coverage",
                        "validation_execution_primary_rmse_physical",
                        "validation_execution_hard_mse_gain_vs_primary_physical",
                        "validation_execution_hard_action_delta_rmse_physical",
                        "validation_execution_neutral_mse_gain_vs_primary_physical",
                        "validation_execution_neutral_action_delta_rmse_physical",
                        "validation_execution_full_capacity_mse_gain_vs_primary_physical",
                        "validation_execution_full_capacity_action_delta_rmse_physical",
                        "validation_execution_three_basis_reduction_mse_gain_vs_primary_physical",
                        "validation_execution_three_basis_reduction_action_delta_rmse_physical",
                        "validation_execution_spine_zero_mse_gain_vs_primary_physical",
                        "validation_execution_spine_zero_action_delta_rmse_physical",
                    ),
                ),
                (
                    "b-spine-ablation",
                    tuple(
                        name
                        for band in ("1_4", "5_12", "13_24")
                        for owner in ("arm", "gripper")
                        for name in (
                            f"validation_execution_spine_zero_{owner}_band_{band}_mse_gain_vs_primary_physical",
                            f"validation_execution_spine_zero_{owner}_band_{band}_action_delta_rmse_physical",
                        )
                    ),
                ),
                (
                    "p2-intervention",
                    tuple(
                        name
                        for mode in (
                            "semantic_far_zero",
                            "geometry_value_all_zero",
                            "geometry_address_neutral",
                            "geometry_value_and_address_zero",
                        )
                        for name in (
                            f"validation_p2_intervention_{mode}_gripper_band_13_24_mse_gain_vs_primary_physical",
                            f"validation_p2_intervention_{mode}_gripper_band_13_24_action_delta_rmse_physical",
                            f"validation_p2_intervention_{mode}_post_event_1_2_mse_gain_vs_primary_physical",
                        )
                    ),
                ),
                (
                    "core-attribution-id",
                    (
                        "validation_core_attribution_coverage",
                        "validation_core_attribution_primary_vs_explicit_none_normalized_action_max_abs",
                        "validation_core_attribution_primary_vs_explicit_none_normalized_bit_exact",
                        "validation_core_attribution_world_vs_consequence_neutral_normalized_action_max_abs",
                        "validation_core_attribution_world_vs_consequence_neutral_normalized_bit_exact",
                        "validation_core_attribution_wrong_action_world_donor_valid_fraction",
                        "validation_core_attribution_wrong_action_world_donor_valid_rows",
                        "validation_core_attribution_wrong_action_world_donor_total_rows",
                    ),
                ),
                (
                    "core-attribution-effect",
                    tuple(
                        name
                        for mode in (
                            "world_dynamic_neutral",
                            "consequence_effect_neutral",
                            "controlled_transition_delta_neutral",
                            "world_and_controlled_transition_neutral",
                            "wrong_action_world",
                        )
                        for name in (
                            f"validation_core_attribution_{mode}_band_13_24_mse_gain_vs_primary_physical",
                            f"validation_core_attribution_{mode}_band_13_24_action_delta_rmse_physical",
                            f"validation_core_attribution_{mode}_gripper_band_13_24_mse_gain_vs_primary_physical",
                            f"validation_core_attribution_{mode}_gripper_band_13_24_action_delta_rmse_physical",
                        )
                    ),
                ),
            )
        else:
            groups = ()
        lead_suffix = f"epoch={epoch:03d}"
        if batch is not None:
            lead_suffix += f" batch={batch:04d}"
        lead_suffix += f" step={step}"
        rows: list[str] = []
        for label, names in groups:
            fields = [f"{name}={metrics[name]:.6g}" for name in names if name in metrics]
            if fields:
                rows.append(
                    " ".join((f"[mainline-{kind}-{label}]", lead_suffix, *fields))
                )
        return tuple(rows)


def validate_resume_metric_boundary(
    output_dir: str | Path,
    *,
    checkpoint_epoch: int,
    checkpoint_step: int,
) -> None:
    """Require an existing metric stream to end at the resumed checkpoint.

    Checkpoints are committed only at epoch boundaries. A crash after an
    epoch metric row was appended but before the checkpoint was atomically
    replaced leaves the stream ahead of ``latest.pt``; a crash in the next
    epoch leaves trailing train rows. Appending after either state would
    silently duplicate or reorder measurements. Resuming into a new output
    directory remains legal because there is no pre-existing stream to own.
    """

    path = Path(output_dir) / "metrics.jsonl"
    if not path.exists() or path.stat().st_size == 0:
        return
    last_row: object | None = None
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    last_row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"exact resume metrics stream has invalid JSON at line {line_number}"
                    ) from error
    except OSError as error:
        raise ValueError("exact resume metrics stream is unreadable") from error
    if last_row is None:
        return
    if not isinstance(last_row, dict):
        raise ValueError("exact resume metrics stream ends with a non-object row")
    kind = last_row.get("kind")
    epoch = last_row.get("epoch")
    step = last_row.get("step")
    if (
        kind != "epoch"
        or isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or isinstance(step, bool)
        or not isinstance(step, int)
    ):
        raise ValueError(
            "exact resume metrics stream must end with a committed epoch row"
        )
    if epoch != int(checkpoint_epoch) or step != int(checkpoint_step):
        raise ValueError(
            "exact resume metrics/checkpoint boundary differs: "
            f"metrics=epoch{epoch}/step{step}, "
            f"checkpoint=epoch{checkpoint_epoch}/step{checkpoint_step}"
        )


def tensor_scalars(values: Mapping[str, Tensor]) -> dict[str, float]:
    """Materialize detached scalars with one transfer per source device.

    Calling ``bool(isfinite(cuda_scalar))`` and then ``scalar.cpu()`` for
    every metric creates two synchronization points per key.  Logging owns a
    synchronization boundary, but it should be one vector boundary rather
    than dozens of serial device round trips.
    """

    grouped: dict[torch.device, list[tuple[str, Tensor]]] = {}
    for name, value in values.items():
        if value.ndim == 0:
            grouped.setdefault(value.device, []).append((name, value.detach().float()))
    result: dict[str, float] = {}
    for rows in grouped.values():
        names = [name for name, _ in rows]
        vector = torch.stack([value for _, value in rows]).cpu()
        finite = torch.isfinite(vector)
        for index, name in enumerate(names):
            if bool(finite[index]):
                result[name] = float(vector[index])
    return result


__all__ = [
    "ACTIVE_PREFIXES",
    "DeviceMetricAccumulator",
    "JsonlRunLogger",
    "MetricAccumulator",
    "active_metrics",
    "archival_metrics",
    "tensor_scalars",
]
