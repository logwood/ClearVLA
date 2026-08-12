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
    "action_",
    "event_",
    "motion_",
    "flow_",
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
            or name.startswith("gradient_postclip_")
            or name.endswith("_nonexpansive_violation")
            or name.endswith("_mass_conservation_error")
            or name
            in {
                "object_grounding_mass_conservation_error",
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
        priority = (
            "loss_total",
            "loss_action_flow",
            "loss_action_flow_v120_comparable",
            "loss_action_flow_native",
            "loss_decoded_action",
            "loss_decoded_action_v120_comparable",
            "loss_future_dynamics",
            "loss_future_successor",
            "loss_future_semantic_delta",
            "loss_future_transition",
            "loss_execution_value",
            "loss_object_reconstruction",
            "object_grounding_object_content_pair_cosine",
            "object_intent_interval_variation",
            "object_w_intent_object_interaction_rms",
            "object_w_action_object_interaction_rms",
            "object_w2_interval_adjacent_cosine",
            "object_w2_object_pair_cosine",
            "loss_action_flow_band_1_4",
            "loss_action_flow_band_5_12",
            "loss_action_flow_band_13_24",
            "validation_action_rmse_normalized",
            "validation_action_rmse_physical",
            "validation_first_rmse_normalized",
            "validation_tail_rmse_normalized",
            "validation_band_1_4_rmse_normalized",
            "validation_band_5_12_rmse_normalized",
            "validation_band_13_24_rmse_normalized",
            "gradient_global_preclip_l2",
            "learning_rate",
        )
        fields = []
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
        """Return V120-equivalent semantic rows for captured nohup logs.

        The JSONL remains the lossless record.  These rows expose the same
        decision-critical groups in stdout so an attached training log is not
        reduced to one RMSE and a handful of top diagnostics.
        """

        groups: tuple[tuple[str, tuple[str, ...]], ...] = (
            (
                "optimization",
                (
                    "learning_rate",
                    "learning_rate_history_proposal",
                    "learning_rate_bottom_decoder",
                    "learning_rate_bottom_capacity",
                ),
            ),
            (
                "performance",
                (
                    "runtime_window_seconds_per_batch",
                    "runtime_window_samples_per_second",
                ),
            ),
            (
                "action",
                (
                    "loss_action_arm_flow",
                    "loss_action_gripper_flow",
                    "loss_action_gripper_event_flow",
                    "loss_action_gripper_hold_flow",
                    "loss_action_decoded_gripper_event",
                    "loss_action_decoded_gripper_hold",
                    "loss_event_precision",
                    "loss_event_recall",
                    "loss_event_f1",
                    "loss_motion_precision",
                    "loss_motion_recall",
                    "loss_action_horizon_mass_band_1_4",
                    "loss_action_horizon_mass_band_5_12",
                    "loss_action_horizon_mass_band_13_24",
                ),
            ),
            (
                "flow",
                (
                    "loss_flow_warp",
                    "loss_flow_photometric_warp",
                    "loss_flow_photometric_zero_warp",
                    "loss_flow_identity_advantage",
                    "loss_flow_static_identity",
                    "loss_flow_cycle",
                    "observation_flow_grid_cell_magnitude",
                    "observation_flow_native_patch_magnitude",
                    "observation_flow_confidence",
                    "observation_flow_occlusion",
                    "observation_flow_correlation_entropy",
                    "observation_flow_correlation_margin",
                    "loss_flow_recent_warp",
                    "loss_flow_earlier_warp",
                    "observation_flow_rms",
                    "observation_earlier_flow_rms",
                    "observation_flow_acceleration_rms",
                    "observation_visual_history_innovation_rms",
                    "observation_recent_motion_rms",
                    "observation_earlier_motion_aligned_rms",
                    "observation_address_flow_mass",
                    "observation_address_fallback_mass",
                    "observation_address_entropy",
                    "observation_raw_detail_emphasis",
                    "observation_raw_detail_precision",
                    "observation_raw_address_flow_mass",
                    "observation_raw_address_fallback_mass",
                    "observation_raw_address_entropy",
                ),
            ),
            (
                "top",
                (
                    "grounding_g1_innovation_rms",
                    "grounding_g2_innovation_rms",
                    "grounding_g3_innovation_rms",
                    "grounding_host_g1_visual_rms",
                    "grounding_host_g2_visual_rms",
                    "grounding_host_g3_visual_rms",
                    "grounding_host_g1_transition_rms",
                    "grounding_host_g2_transition_rms",
                    "grounding_host_g3_transition_rms",
                    "object_grounding_candidate_key_rms",
                    "object_grounding_candidate_value_rms",
                    "object_grounding_g3_parent_l1",
                    "object_grounding_object_content_pair_cosine",
                    "object_grounding_object_chart_pair_overlap",
                    "object_grounding_semantic_appearance_posterior_l1",
                    "object_grounding_semantic_geometry_posterior_l1",
                    "object_grounding_appearance_geometry_posterior_l1",
                    "object_grounding_camera_coordinate_variation",
                    "object_grounding_prototype_mse",
                    "object_grounding_spatial_refinement_mse",
                    "object_grounding_typed_consistency",
                    "object_intent_goal_innovation_rms",
                    "object_intent_history_innovation_rms",
                    "object_intent_object_innovation_rms",
                    "object_intent_state_innovation_rms",
                    "object_intent_action_innovation_rms",
                    "object_intent_typed_innovation_rms",
                    "object_intent_interval_variation",
                    "object_intent_temporal_variation",
                    "object_w_state_innovation_rms",
                    "object_w_object_key_innovation_rms",
                    "object_w_object_value_innovation_rms",
                    "object_w_typed_innovation_rms",
                    "object_w_intent_object_interaction_rms",
                    "object_w_action_object_interaction_rms",
                    "object_w_prediction_interval_variation",
                    "object_teacher_interval_variation",
                    "object_teacher_association_confidence",
                    "object_teacher_reliability",
                    "object_teacher_null_probability",
                    "object_teacher_semantic_delta_rms",
                    "object_teacher_transport_rms",
                    "object_w2_interval_adjacent_cosine",
                    "object_w2_object_pair_cosine",
                    "loss_future_transition",
                    "p1_query_chart_variation",
                    "p1_query_coordinate_variation",
                    "p1_object_posterior_entropy",
                    "p1_object_posterior_max",
                    "p1_null_mass",
                    "p1_progressive_candidate_valid_fraction",
                    "p1_progressive_candidate_entropy",
                    "p1_progressive_candidate_max",
                    "p1_progressive_candidate_count",
                    "p1_local_content_rms",
                    "p1_detail_rms",
                    "p1_fact_by_object_rms",
                    "p1_fact_rms",
                    "p1_existence_is_diagnostic_only",
                ),
            ),
            (
                "policy",
                (
                    "p1_protected_detail_rms",
                    "p1_dynamic_delta_rms",
                    "p1_completed_fact_rms",
                    "p1_policy_content_mod_rms",
                    "p1_policy_self_written_rms",
                    "p1_policy_ffn_written_rms",
                    "object_p2_content_score_abs",
                    "object_p2_content_score_max_abs",
                    "object_p2_intent_score_abs",
                    "object_p2_intent_score_max_abs",
                    "object_p2_coordinate_score_abs",
                    "object_p2_coordinate_score_max_abs",
                    "object_p2_combined_logit_max_abs",
                    "object_p2_temperature_content",
                    "object_p2_temperature_intent",
                    "object_p2_temperature_coordinate",
                    "object_p2_posterior_entropy",
                    "object_p2_posterior_max",
                    "object_p2_null_mass",
                    "object_p2_effect_precontract_rms",
                    "object_p2_semantic_value_mass",
                    "object_p2_geometry_value_mass",
                    "object_p2_status_value_mass",
                    "object_p2_interval_0_mass",
                    "object_p2_interval_1_mass",
                    "object_p2_interval_2_mass",
                    "object_p2_interval_3_mass",
                    "object_consequence_effect_rms",
                    "object_consequence_interaction_rms",
                    "object_consequence_ratio",
                    "object_p3_factual_rms",
                    "object_p3_precision_rms",
                    "object_p3_effect_rms",
                    "object_p3_temporal_rms",
                    "object_p3_state_change_rms",
                    "controlled_transition_value_rms",
                    "controlled_transition_spatial_value_variation",
                    "controlled_transition_pool_entropy",
                    "controlled_transition_dense_rows",
                    "controlled_transition_pooled_rows",
                ),
            ),
            (
                "bottom",
                (
                    "bottom_capacity_mean",
                    "bottom_capacity_block_std",
                    "bottom_expected_depth",
                    "bottom_continue_mean",
                    "bottom_continue_block_std",
                    "bottom_execution_cost_audit",
                    "loss_execution_value",
                    "loss_execution_value_target_spread",
                    "loss_execution_value_predicted_spread",
                    "loss_execution_value_correlation",
                    "loss_execution_value_pairwise_accuracy",
                    "loss_execution_value_decision_accuracy",
                    "loss_execution_value_common_mode_ratio",
                    "loss_execution_candidate_coverage",
                    "loss_execution_terminal_target_cost_margin",
                    "loss_execution_terminal_predicted_cost_margin",
                    "loss_execution_terminal_target_preferred_fraction",
                    "loss_execution_terminal_identity_error",
                    "bottom_controller_common_ratio",
                    "bottom_controller_private_ratio",
                    "bottom_controller_ownership_max",
                    "bottom_protected_update_rms",
                    "bottom_evidence_value_rms",
                    "bottom_action_rms",
                    "bottom_latent_rms",
                    "bottom_latent_batch_variance",
                ),
            ),
            (
                "val-detail",
                (
                    "validation_arm_rmse_normalized",
                    "validation_gripper_rmse_normalized",
                    "validation_decoded_gripper_event_precision",
                    "validation_decoded_gripper_event_recall",
                    "validation_decoded_gripper_event_f1",
                    "validation_decoded_gripper_event_ratio",
                    "validation_decoded_gripper_timing_mae_steps",
                    "validation_event_head_precision",
                    "validation_event_head_recall",
                    "validation_event_head_f1",
                    "validation_event_head_events_predicted",
                    "validation_event_head_events_target",
                    "validation_event_head_minus_decoded_f1",
                    "validation_motion_head_precision",
                    "validation_motion_head_recall",
                    "validation_motion_head_f1",
                    "validation_decoded_motion_precision",
                    "validation_decoded_motion_recall",
                    "validation_decoded_motion_f1",
                    "validation_arm_rmse_physical",
                    "validation_gripper_rmse_physical",
                    "validation_ablation_batches",
                    "validation_ablation_coverage",
                    "validation_diagnostic_primary_rmse_physical",
                    "validation_proposal_zero_rmse_physical",
                    "validation_proposal_zero_mse_gain_vs_primary_physical",
                    "validation_proposal_zero_action_delta_rmse_physical",
                    "validation_execution_no_updates_rmse_physical",
                    "validation_execution_no_updates_mse_gain_vs_primary_physical",
                    "validation_execution_no_updates_action_delta_rmse_physical",
                    "validation_execution_full_updates_rmse_physical",
                    "validation_execution_full_updates_mse_gain_vs_primary_physical",
                    "validation_execution_full_updates_action_delta_rmse_physical",
                ),
            ),
        )
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
        gradient_fields = [
            f"{name}={value:.3e}"
            for name, value in sorted(metrics.items())
            if name.startswith("gradient_postclip_")
        ]
        if gradient_fields:
            rows.append(
                " ".join(
                    (f"[mainline-{kind}-grad]", lead_suffix, *gradient_fields)
                )
            )
        block_fields = [
            f"{name}={value:.6g}"
            for name, value in sorted(metrics.items())
            if name.startswith("bottom_block_")
            and (
                name.endswith("_executed_update_rms")
                or name.endswith("_capacity_ratio")
                or name.endswith("_effective_basis_mass")
                or name.endswith("_contraction_ratio")
                or name.endswith("_capacity_nonexpansive_violation")
                or name.endswith("_nonexpansive_violation")
            )
        ]
        if block_fields:
            rows.append(
                " ".join(
                    (f"[mainline-{kind}-bottom-block]", lead_suffix, *block_fields)
                )
            )
        for index in range(4):
            interval_names = (
                f"object_w2_interval_{index}_successor_innovation_rms",
                f"object_w2_interval_{index}_semantic_delta_rms",
                f"object_w2_interval_{index}_transport_rms",
                f"object_w2_interval_{index}_reliability",
                f"object_teacher_interval_{index}_successor_innovation_rms",
                f"object_teacher_interval_{index}_semantic_delta_rms",
                f"object_teacher_interval_{index}_transport_rms",
                f"object_teacher_interval_{index}_reliability",
                f"loss_future_interval_{index}_successor",
                f"loss_future_interval_{index}_semantic_delta",
                f"loss_future_interval_{index}_transport",
            )
            interval_fields = [
                f"{name}={metrics[name]:.6g}" for name in interval_names if name in metrics
            ]
            if interval_fields:
                rows.append(
                    " ".join(
                        (
                            f"[mainline-{kind}-future-{index}]",
                            lead_suffix,
                            *interval_fields,
                        )
                    )
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
