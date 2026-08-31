from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from typing import Any

from clearvla.tools.audit_policy_logs import (
    BatchPoint,
    ParsedRun,
    _recovery_assessment,
    build_summary,
    parse_log,
    parse_run_input,
)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _complete_recovery_summary(label: str) -> dict[str, Any]:
    validation = {
        "full_rmse": 0.08,
        "first_rmse": 0.02,
        "first8_rmse": 0.04,
        "tail_rmse": 0.09,
        "arm_full_rmse": 0.06,
        "gripper_full_rmse": 0.14,
        "action_band_1_4_rmse": 0.03,
        "action_band_5_12_rmse": 0.06,
        "action_band_13_24_rmse": 0.1,
        "validation_band_1_4_rmse_physical": 0.03,
        "validation_band_5_12_rmse_physical": 0.06,
        "validation_band_13_24_rmse_physical": 0.1,
        "gripper_f1": 0.35,
        "gripper_event_f1": 0.35,
        "event_head_f1": 0.14,
        "motion_head_f1": 0.83,
        "gripper_event_ratio": 0.4,
        "validation_ablation_coverage": 1.0,
        "validation_diagnostic_primary_rmse_physical": 0.08,
        "validation_proposal_zero_mse_gain_vs_primary_physical": 0.0,
        "validation_execution_no_updates_mse_gain_vs_primary_physical": 0.0,
        "validation_execution_full_updates_mse_gain_vs_primary_physical": 0.0,
    }
    structure_names = (
        "object_grounding_object_content_pair_cosine",
        "object_intent_interval_variation",
        "object_w_prediction_interval_variation",
        "object_w1_object_pair_cosine",
        "object_w2_object_pair_cosine",
        "object_teacher_reliability",
        "p1_query_chart_variation",
        "object_p2_successor_innovation_rms",
        "object_p3_precision_base_rms",
        "bottom_capacity_mean",
    )
    gradient_names = (
        "gradient_postclip_grounder_l2",
        "gradient_postclip_intent_l2",
        "gradient_postclip_dynamics_l2",
        "gradient_postclip_p1_factual_l2",
        "gradient_postclip_p2_effect_reader_l2",
        "gradient_postclip_p3_compiler_l2",
        "gradient_postclip_v120_canvas_seed_l2",
        "gradient_postclip_v120_layer_contracts_l2",
        "gradient_postclip_bottom_evidence_adapter_l2",
        "gradient_postclip_bottom_policy_bridge_l2",
        "gradient_postclip_bottom_capacity_l2",
        "gradient_postclip_bottom_execution_l2",
    )
    trajectories = {
        name: {"tail_median": value}
        for name, value in {
            "physical_flow": 0.011,
            "physical_flow_native": 0.016,
            "arm_fm_per_dim": 0.01,
            "gripper_fm_field": 0.02,
            "decoded_action": 0.002,
        }.items()
    }
    aligned_batch_2200 = {
        name: {"tail_median": value}
        for name, value in {
            "object_grounding_g3_parent_l1": 0.02,
            "object_grounding_object_content_pair_cosine": 0.5,
            "object_p1_spatial_variation": 0.01,
            "p1_query_chart_variation": 0.01,
            "object_p2_null_mass": 0.2,
            "object_p2_effect_precontract_rms": 0.03,
            "object_consequence_effect_rms": 0.03,
        }.items()
    }
    return {
        "label": label,
        "manifest": {
            "seed": 0,
            "batch_size": 8,
            "data_root": "/dataset",
            "train_episode_count": 63,
            "val_episode_count": 5,
            "action_normalizer_fingerprint": "a" * 12,
        },
        "coverage": {
            "epoch_records": 8,
            "batch_rows": 100,
            "fatal_errors": [],
            "traceback_count": 0,
        },
        "observability": {"batch_metric_count": 100},
        "epochs": [
            {"epoch": epoch, "global_step": epoch * 100, "val": dict(validation)}
            for epoch in range(1, 9)
        ],
        "trajectories": trajectories,
        "aligned_batch_2200": aligned_batch_2200,
        "structure": {
            name: {
                "tail_median": (
                    0.5 if name.endswith("object_pair_cosine") else 0.1
                )
            }
            for name in structure_names
        },
        "gradients": {name: {"tail_median": 0.01} for name in gradient_names},
        "performance": {
            "seconds_per_batch": {
                "count": 100,
                "source": "legacy-window",
                "median": 2.0,
                "p90": 2.4,
                "minimum": 1.8,
                "maximum": 2.8,
            },
            "cuda_peak_process_estimate_gib": 10.0,
        },
    }


class AuditPolicyLogsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_compact_v94_rows_merge_loss_execution_and_gradient_lines(self) -> None:
        epoch_record = {
            "epoch": 1,
            "global_step": 1200,
            "train": {
                "loss": 0.4,
                "physical_flow": 0.3,
                "loss_group_action": 0.3,
                "loss_group_execution": 0.05,
                "loss_group_layer": 0.05,
                "loss_ledger_residual": 0.0,
            },
            "val": {
                "full_rmse": 0.1,
                "tail_first_ratio": 1.5,
                "gripper_event_ratio": 1.1,
                "sample_evidence_z_zero_condition_delta": 0.02,
                "sample_evidence_z_shuffle_condition_delta": 0.03,
            },
        }
        log = _write(
            self.tmp_path / "nohup.log",
            "\n".join(
                (
                    "[v94] decoder=evidence_latent_mmdit_action rank=32 groups=32 "
                    "depth_logit_init=2.268683541 warmup=200 transition=1000 z_probe=1",
                    "[v94-train] epoch=001 batch=1200 loss=0.400000 pflow=0.300000 "
                    "lossgrp=action:0.30000/execution:0.05000/layer:0.05000 "
                    "losscontrib=flow:0.30000/execution_value:0.05000/layer_contract:0.05000 "
                    "ledger_residual=+0.00e+00",
                    "[v94-exec] progress=1.00 capacity=0.90625 depth=29.000 "
                    "route=soft:0.400/hard:0.250/gap:+0.150 "
                    "dwell=soft:1.600/hard:2.000/gap:-0.400 value_common=0.30",
                    "[v94-grad] cap_control=2.00e-03 layer_adapter=3.00e-03 global=1.00e+00",
                    json.dumps(epoch_record),
                )
            ),
        )

        parsed = parse_log(log)
        self.assertEqual(len(parsed.batch_points), 1)
        row = parsed.batch_points[0].metrics
        self.assertEqual(row["loss_group_action"], 0.3)
        self.assertEqual(row["loss_contrib_execution_value"], 0.05)
        self.assertEqual(row["evidence_mmd_it_dynamic_route_next_fraction"], 0.4)
        self.assertEqual(row["evidence_mmd_it_hard_route_next_fraction"], 0.25)
        self.assertEqual(row["evidence_mmd_it_hard_dwell_expected"], 2.0)
        self.assertEqual(row["grad_evidence_mmdit_capacity_control"], 0.002)

        summary = build_summary(parsed)
        self.assertEqual(summary["loss_budget"]["mode"], "exact-ledger")
        self.assertEqual(summary["loss_budget"]["residual"], 0.0)
        self.assertEqual(summary["loss_budget"]["components"]["flow"], 0.3)
        codes = {item["code"] for item in summary["findings"]}
        self.assertNotIn("capacity-saturated", codes)
        self.assertNotIn("z-probe-missing", codes)

    def test_v119_rows_preserve_grounded_boundaries_and_interval_errors(self) -> None:
        path = _write(
            self.tmp_path / "v119.log",
            "\n".join(
                (
                    "[v119-train] epoch=001 batch=0020 loss_total=0.500000 "
                    "flow_loss=0.400000",
                    "[v119-repr] warp=0.01000 p1_spatial_var=0.120",
                    "[v119-ground] active=1 g2_fine_H=0.700 "
                    "g2g3_sem=0.050 current_ref_align=1.0e-04",
                    "[v119-intent] goal_attention_H=0.800 interval_var=0.040 "
                    "h4_8_goal_H=0.750 h32_48_source_H=0.650",
                    "[v119-effect] w1_sem=0.110 w2_sem=0.220 "
                    "loss_semantic=0.0300 loss_interval_transition=0.0400",
                    "[v119-effect-error] interval=h4_8 teacher_reliability=0.300 "
                    "successor=0.400 semantic=0.500 transport=0.600",
                    "[v119-effect-error] interval=h32_48 teacher_reliability=0.700 "
                    "successor=0.800 semantic=0.900 transport=1.000",
                    "[v119-policy] effect_read=0.130 posterior_H=0.810 "
                    "mass_h4_8=0.300 p3_precision=0.140",
                    "[v119-exec] capacity_gate_mass=1.00000 "
                    "effective_basis_mass=32.000",
                    "[v119-grad] grounded_w_inputs=1.00e-03 "
                    "grounded_w1_blocks=2.00e-03 grounded_w2_blocks=3.00e-03 "
                    "grounded_w_shared_heads=4.00e-03 global_preclip=1.00e+00",
                    "[v119-epoch] epoch=001 step=20 loss_total=0.500000 "
                    "flow_loss=0.400000",
                    "[v119-val] action_rmse=0.10000",
                )
            ),
        )

        run = parse_log(path)
        self.assertEqual(len(run.batch_points), 1)
        self.assertEqual(run.batch_points[0].source, "v119")
        row = run.batch_points[0].metrics
        self.assertEqual(row["grounded_intent_effect_active"], 1.0)
        self.assertEqual(row["grounded_g2_g3_semantic_owner_l1"], 0.05)
        self.assertEqual(row["grounded_s_h4_8_goal_attention_entropy"], 0.75)
        self.assertEqual(row["grounded_w2_semantic_rms"], 0.22)
        self.assertEqual(
            row["grounded_future_effect_semantic_h4_8_target_normalized_error"],
            0.5,
        )
        self.assertEqual(
            row[
                "grounded_future_effect_teacher_reliability_h32_48"
            ],
            0.7,
        )
        self.assertEqual(row["grounded_p2_effect_read_rms"], 0.13)
        self.assertEqual(row["grad_grounded_world_shared_inputs"], 0.001)
        self.assertEqual(row["grad_grounded_world_w2_blocks"], 0.003)
        self.assertEqual(len(run.epoch_records), 1)
        self.assertEqual(run.epoch_records[0]["val"]["full_rmse"], 0.1)

    def test_v120_rows_preserve_object_intent_dynamics_boundaries(self) -> None:
        path = _write(
            self.tmp_path / "v120.log",
            "\n".join(
                (
                    "[v120] stage1_checkpoint=/tmp/old.pt "
                    "stage1_initialization_enabled=0 fresh=1",
                    "[v120-train] epoch=001 batch=0020 loss_total=0.500000 "
                    "flow_loss=0.400000",
                    "[v120-ground] reconstruction=0.03000 existence=0.700 "
                    "validity=1.000 allocation=0.200 null=0.200 mass_error=0.0e+00 "
                    "object_pair_cos=0.400 g3_parent_l1=0.000e+00",
                    "[v120-intent] goal_H=0.800 interval_var=0.040 "
                    "state_delta=0.120 transport=0.030 state_change=0.050 "
                    "online_match=0.02000 recognizer=0.03000 coarse_action=0.04000",
                    "[v120-dynamics] w1_delta=0.110 w2_delta=0.220 "
                    "teacher_visibility_change=-0.300 teacher_supports=3.00 "
                    "semantic_loss=0.0300 transition_loss=0.0400",
                    "[v120-dynamics-error] interval=h4_8 successor=0.400 "
                    "semantic=0.500 transport=0.600 visibility=0.700",
                    "[v120-dynamics-error] interval=h32_48 successor=0.800 "
                    "semantic=0.900 transport=1.000 uncertainty=1.100",
                    "[v120-policy] content_score_max=0.900 intent_score_max=0.800 "
                    "coordinate_score_max=0.700 combined_logit_max=4.200 "
                    "semantic_mass=0.400 h4_8_mass=0.300 p3_effect=0.140 "
                    "p3_state_change=0.004",
                    "[v120-grad] object_w_inputs=1.00e-03 global_preclip=1.00e+00",
                    "[v120-epoch] epoch=001 step=20 loss_total=0.500000 "
                    "flow_loss=0.400000",
                    "[v120-val] action_rmse=0.10000",
                )
            ),
        )

        run = parse_log(path)
        self.assertEqual(len(run.batch_points), 1)
        self.assertEqual(run.batch_points[0].source, "v120")
        row = run.batch_points[0].metrics
        self.assertEqual(row["object_grounding_reconstruction_mse"], 0.03)
        self.assertEqual(row["object_grounding_validity_mean"], 1.0)
        self.assertEqual(row["object_grounding_allocation_share_mean"], 0.2)
        self.assertEqual(row["object_grounding_mass_conservation_error"], 0.0)
        self.assertEqual(row["object_intent_interval_variation"], 0.04)
        self.assertEqual(row["object_intent_observed_state_delta_rms"], 0.12)
        self.assertEqual(row["object_intent_observed_transport_rms"], 0.03)
        self.assertEqual(row["object_intent_state_change_evidence_rms"], 0.05)
        self.assertEqual(row["object_w2_semantic_delta_rms"], 0.22)
        self.assertEqual(row["object_teacher_visibility_change"], -0.3)
        self.assertEqual(
            row["object_future_semantic_h4_8_normalized_error"], 0.5
        )
        self.assertEqual(
            row["object_future_uncertainty_h32_48_normalized_error"], 1.1
        )
        self.assertEqual(row["object_p2_combined_logit_max_abs"], 4.2)
        self.assertEqual(row["object_p3_effect_rms"], 0.14)
        self.assertEqual(row["object_p3_state_change_rms"], 0.004)
        self.assertEqual(len(run.epoch_records), 1)
        self.assertEqual(run.epoch_records[0]["val"]["full_rmse"], 0.1)
        manifest = build_summary(run)["manifest"]
        self.assertEqual(manifest["stage1_checkpoint"], "/tmp/old.pt")
        self.assertEqual(manifest["stage1_initialization_enabled"], 0)
        self.assertEqual(manifest["fresh_run"], 1)

    def test_v121_rows_parse_typed_grounding_and_policy_routes(self) -> None:
        path = _write(
            self.tmp_path / "v121.log",
            "\n".join(
                (
                    "[v121-train] epoch=001 batch=0020 loss_total=0.450000 "
                    "flow_loss=0.350000",
                    "[v121-ground] reconstruction=0.02000 prototype_mse=0.02400 "
                    "spatial_refine_mse=0.00800 chart_pair_overlap=0.310 "
                    "sem_app_post_l1=0.120 sem_geo_post_l1=0.230 "
                    "app_geo_post_l1=0.190",
                    "[v121-dynamics] w1_delta=0.090 w2_delta=0.180 "
                    "semantic_loss=0.0200 transition_loss=0.0300",
                    "[v121-policy] semantic_score_max=0.910 "
                    "geometry_score_max=0.720 semantic_H=0.810 geometry_H=0.640 "
                    "semantic_mass=0.570 geometry_mass=0.430 "
                    "p3_precision=0.120 p3_temporal=0.080 p3_state_change=0.003",
                    "[v121-epoch] epoch=001 step=20 loss_total=0.450000 "
                    "flow_loss=0.350000",
                    "[v121-val] action_rmse=0.09000",
                )
            ),
        )
        run = parse_log(path)
        self.assertEqual(run.batch_points[0].source, "v121")
        row = run.batch_points[0].metrics
        self.assertEqual(row["object_grounding_prototype_mse"], 0.024)
        self.assertEqual(
            row["object_grounding_spatial_refinement_mse"], 0.008
        )
        self.assertEqual(row["object_grounding_object_chart_pair_overlap"], 0.31)
        self.assertEqual(
            row["object_grounding_semantic_appearance_posterior_l1"], 0.12
        )
        self.assertEqual(
            row["object_grounding_semantic_geometry_posterior_l1"], 0.23
        )
        self.assertEqual(
            row["object_grounding_appearance_geometry_posterior_l1"], 0.19
        )
        self.assertEqual(row["object_p2_semantic_score_max_abs"], 0.91)
        self.assertEqual(row["object_p2_geometry_score_max_abs"], 0.72)
        self.assertEqual(row["object_p2_semantic_value_mass"], 0.57)
        self.assertEqual(row["object_p2_geometry_value_mass"], 0.43)
        self.assertNotIn("object_p3_effect_rms", row)
        self.assertEqual(row["object_p3_state_change_rms"], 0.003)

    def test_v122_rows_parse_identity_innovation_and_camera_metrics(self) -> None:
        path = _write(
            self.tmp_path / "v122.log",
            "\n".join(
                (
                    "[v122-train] epoch=001 batch=0020 loss_total=0.420000 "
                    "flow_loss=0.330000",
                    "[v122-ground] typed_consistency=0.210 camera_coord_var=0.080",
                    "[v122-intent] object_sim_H=0.760 action_innov=0.110 "
                    "temporal_innov=0.070 coarse_innov=0.090",
                    "[v122-dynamics] condition_interaction=0.060 w2_delta=0.170",
                    "[v122-policy] relative_status_abs=0.140 "
                    "p3_centered_detail=0.190 p3_consequence_innov=0.050",
                    "[v122-epoch] epoch=001 step=20 loss_total=0.420000 "
                    "flow_loss=0.330000",
                    "[v122-val] action_rmse=0.08800",
                )
            ),
        )
        run = parse_log(path)
        self.assertEqual(run.batch_points[0].source, "v122")
        row = run.batch_points[0].metrics
        self.assertEqual(row["object_grounding_typed_consistency"], 0.21)
        self.assertEqual(
            row["object_grounding_camera_coordinate_variation"], 0.08
        )
        self.assertEqual(
            row["object_intent_interval_object_audit_similarity_entropy"],
            0.76,
        )
        self.assertEqual(row["object_intent_action_innovation_rms"], 0.11)
        self.assertEqual(row["object_w_condition_interaction_rms"], 0.06)
        self.assertEqual(row["object_p2_relative_status_abs"], 0.14)
        self.assertEqual(row["object_p3_centered_detail_rms"], 0.19)

    def test_unhandled_nonfinite_backward_is_a_critical_finding(self) -> None:
        log = _write(
            self.tmp_path / "nonfinite.log",
            "\n".join(
                (
                    "[v96-train] epoch=001 batch=2500 loss_total=0.138861 flow_loss=0.081567",
                    "[v96-grad] global_preclip=1.14e+00",
                    "Traceback (most recent call last):",
                    '  File "train.py", line 7, in <module>',
                    "    clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)",
                    "RuntimeError: The total norm of order 2.0 for gradients "
                    "from parameters is non-finite, so it cannot be clipped.",
                )
            ),
        )
        summary = build_summary(parse_log(log))
        findings = {item["code"]: item for item in summary["findings"]}
        self.assertIn("non-finite-backward", findings)
        self.assertEqual(findings["non-finite-backward"]["severity"], "critical")
        self.assertEqual(summary["coverage"]["traceback_count"], 1)
        self.assertIn("non-finite", summary["coverage"]["fatal_errors"][0])

    def test_legacy_log_reports_duplicate_objective_and_placeholder_noise(self) -> None:
        context = {
            "schema": "test",
            "trainer": {
                "rollout_dynamics_loss_weight": 0.03,
                "rollout_delta_loss_weight": 0.01,
            },
        }
        rows = [json.dumps(context, indent=2)]
        for batch in range(20, 140, 20):
            rows.append(
                "[v39-layer] epoch=001 batch="
                f"{batch:04d} loss=0.5 pflow=0.3 rollout=0.2 delta=0.2 "
                "evexec=1.0/1.0/1.5/1.5 evcap=32/0/0 "
                "evval=0.1/0.2/0.2/0.5/1.0 evcgrad=1e-3/0/1e-3 "
                "unused_a=0 unused_b=0 grad=1.0"
            )
        log = _write(self.tmp_path / "legacy.log", "\n".join(rows))
        summary = build_summary(parse_log(log))

        findings = {item["code"]: item for item in summary["findings"]}
        self.assertEqual(findings["duplicate-series"]["severity"], "warning")
        self.assertEqual(
            summary["structure"]["evidence_mmd_it_effective_depth"]["tail_median"],
            32.0,
        )
        self.assertGreaterEqual(summary["observability"]["always_zero_count"], 2)

    def test_medium_length_v94_labels_map_to_canonical_metrics(self) -> None:
        log = _write(
            self.tmp_path / "medium_labels.log",
            "\n".join(
                (
                    "[v94-train] epoch=001 batch=0020 loss_total=0.400000 "
                    "flow_loss=0.300000 native_flow=0.250000 arm_flow=0.20000 "
                    "grip_flow=0.90000 rollout_step=0.12000 "
                    "loss_groups=action:0.30000/rollout:0.05000 "
                    "top_contrib=flow:0.30000/rollout_milestone:0.05000 "
                    "ledger_gap=+0.00e+00",
                    "[v94-exec] exec_progress=1.00 soft_capacity=0.90625 "
                    "effective_rank=29.000 selected_rank=29.000 "
                    "capacity_gate_mass=0.90625 effective_basis_mass=29.000 "
                    "terminal_prior=0.250 terminal_probability=0.100 "
                    "hard_terminal_fraction=0.050 terminal_target_margin=+0.0200 "
                    "workload_audit=1.450 value_top1_acc=0.60 candidate_coverage=0.75",
                    "[v94-grad] capacity_control=2.00e-03 "
                    "global_preclip=1.00e+00 sec_per_batch=0.500",
                    "[v94-epoch] epoch=001 step=20 loss_total=0.400000 "
                    "flow_loss=0.300000 loss_groups=action:0.30000/rollout:0.05000 "
                    "ledger_gap=+0.00e+00",
                    "[v94-val] action_rmse=0.10000 first_rmse=0.05000 "
                    "first8_rmse=0.07000 tail_rmse=0.12000 tail_first_ratio=2.400 "
                    "arm_rmse=0.08000 grip_rmse=0.20000 grip_event_ratio=4.000 "
                    "grip_events_pred=400 grip_events_target=100 "
                    "grip_event=p:0.200/r:0.700/f1:0.310 proposal_mse_gain=+1.000e-02 "
                    "proposal_batch_cov=0.25 balanced_score=0.12000 deploy_gate=0",
                    "[v94-probe] z_zero_cond_delta=2.0000e-02 "
                    "z_shuffle_cond_delta=3.0000e-02 soft_capacity=0.90625 "
                    "effective_rank=29.000 terminal_probability=0.100 probe_batch_cov=0.25",
                )
            ),
        )

        parsed = parse_log(log)
        row = parsed.batch_points[0].metrics
        self.assertEqual(row["physical_flow"], 0.3)
        self.assertEqual(row["physical_flow_native"], 0.25)
        self.assertEqual(row["loss_contrib_rollout_milestone"], 0.05)
        self.assertEqual(row["evidence_mmd_it_effective_depth"], 29.0)
        self.assertEqual(row["evidence_mmd_it_capacity_gate_mass"], 0.90625)
        self.assertEqual(row["evidence_mmd_it_effective_basis_mass"], 29.0)
        self.assertEqual(row["evidence_mmd_it_terminal_prior_weight"], 0.25)
        self.assertEqual(row["evidence_mmd_it_terminal_probability"], 0.1)
        self.assertEqual(row["evidence_mmd_it_terminal_target_cost_margin"], 0.02)
        self.assertEqual(row["evidence_mmd_it_execution_value_decision_accuracy"], 0.6)
        self.assertEqual(row["grad_evidence_mmdit_capacity_control"], 0.002)
        self.assertEqual(row["seconds_per_batch"], 0.5)

        val = parsed.epoch_records[0]["val"]
        self.assertEqual(val["full_rmse"], 0.1)
        self.assertEqual(val["gripper_event_ratio"], 4.0)
        self.assertEqual(val["gripper_precision"], 0.2)
        self.assertEqual(val["eval_proposal_ablation_coverage"], 0.25)
        self.assertEqual(val["sample_evidence_z_zero_condition_delta"], 0.02)
        self.assertEqual(val["sample_evidence_mmd_it_terminal_probability"], 0.1)

    def test_v95_stage1_rows_are_representation_not_action_training(self) -> None:
        log = _write(
            self.tmp_path / "v95_stage1.log",
            "\n".join(
                (
                    "[v95] experiment_stage=stage1 decoder=evidence_latent_mmdit_action",
                    "[v95-stage1-train] epoch=001 batch=0020 "
                    "loss_representation=0.125000 "
                    "contrib=flow_jepa_future_prediction:0.08000/"
                    "flow_jepa_stage_prediction:0.04500 ledger_gap=+0.00e+00",
                    "[v95-stage1-repr] window_pred=0.80000 stage_pred=2.25000 "
                    "warp=0.01000 cycle=0.40000 stage_window_cos=0.350 "
                    "goal_pair_cos=0.120 action_mem_norm=4.500",
                    "[v95-stage1-grad] flow_dino=1.20e-01 goal_tokens=2.00e-02 "
                    "action_history=3.00e-02 dit_blocks=4.00e-02 "
                    "global_preclip=8.00e-01 lr=8.000e-07 sec_per_batch=1.200",
                    "[v95-stage1-epoch] epoch=001 step=20 "
                    "train_representation=0.125000 val_representation=0.110000 "
                    "window_pred=0.70000 stage_pred=2.00000 stage_window_cos=0.400 "
                    "goal_pair_cos=0.150 repr_batch_cov=1.00",
                )
            ),
        )

        parsed = parse_log(log)
        self.assertEqual(len(parsed.batch_points), 1)
        row = parsed.batch_points[0]
        self.assertEqual(row.source, "v95-stage1")
        self.assertEqual(row.metrics["loss"], 0.125)
        self.assertEqual(row.metrics["loss_group_representation"], 0.125)
        self.assertEqual(row.metrics["flow_jepa_future_prediction"], 0.8)
        self.assertEqual(row.metrics["grad_flow_dino_evidence"], 0.12)
        self.assertNotIn("physical_flow", row.metrics)

        summary = build_summary(parsed)
        self.assertEqual(summary["manifest"]["training_stage"], "stage1")
        self.assertEqual(summary["loss_budget"]["mode"], "exact-ledger")
        self.assertEqual(summary["loss_budget"]["groups"]["representation"], 0.125)
        self.assertEqual(summary["observability"]["missing_core"], [])
        val = parsed.epoch_records[0]["val"]
        self.assertEqual(val["loss"], 0.11)
        self.assertEqual(val["flow_jepa_stage_prediction"], 2.0)
        self.assertEqual(val["eval_representation_coverage"], 1.0)

    def test_v95_policy_rows_share_the_compact_v94_parser(self) -> None:
        log = _write(
            self.tmp_path / "v95_policy.log",
            "\n".join(
                (
                    "[v95-train] epoch=001 batch=0020 loss_total=1.200000 "
                    "flow_loss=1.000000 loss_groups=action:1.00000/representation:0.20000 "
                    "top_contrib=flow:1.00000/flow_jepa_future:0.20000",
                    "[v95-repr] window_pred=0.50000 stage_pred=1.50000 "
                    "goal_norm=2.000 action_mem_norm=3.000",
                    "[v95-grad] flow_dino=1.00e-01 dit_blocks=2.00e-02 "
                    "global_preclip=9.00e-01 sec_per_batch=1.000",
                )
            ),
        )

        parsed = parse_log(log)
        self.assertEqual(len(parsed.batch_points), 1)
        row = parsed.batch_points[0]
        self.assertEqual(row.source, "v95")
        self.assertEqual(row.metrics["physical_flow"], 1.0)
        self.assertEqual(row.metrics["flow_jepa_future_prediction"], 0.5)
        self.assertEqual(row.metrics["grad_flow_dino_evidence"], 0.1)

    def test_v118_differential_boundary_labels_map_to_canonical_metrics(
        self,
    ) -> None:
        log = _write(
            self.tmp_path / "v118_differential.log",
            "\n".join(
                (
                    "[v118-train] epoch=001 batch=0020 loss_total=1.200000 "
                    "flow_loss=1.000000 loss_groups=action:1.00000/"
                    "representation:0.20000",
                    "[v118-repr] w0_clean_proposal=0.1200 "
                    "w1_clean_proposal=0.1300 w2_clean_proposal=0.1400 "
                    "w0_direct_intent_bypass=0.0000 "
                    "w1_direct_intent_bypass=0.0000 "
                    "w2_direct_intent_bypass=0.0000 "
                    "p1_intent_query=0.2100 "
                    "p1_direct_condition_bypass=0.0000 "
                    "g_to_p_intent_query=0.2200 "
                    "g_to_p_goal_bypass=0.0000 "
                    "g_to_p_history_bypass=0.0000",
                    "[v118-grad] w_clean_proposal=1.00e-02 "
                    "intent_g_to_p_query=2.00e-02 "
                    "intent_p1_query=3.00e-02 "
                    "global_preclip=9.00e-01 sec_per_batch=1.000",
                )
            ),
        )

        parsed = parse_log(log)
        self.assertEqual(len(parsed.batch_points), 1)
        row = parsed.batch_points[0].metrics
        self.assertEqual(
            row["flow_jepa_w0_clean_proposal_context_rms"],
            0.12,
        )
        self.assertEqual(row["flow_jepa_w2_direct_intent_bypass"], 0.0)
        self.assertEqual(row["flow_jepa_phase_detail_query_norm"], 0.21)
        self.assertEqual(
            row["flow_jepa_differential_p1_direct_condition_bypass"],
            0.0,
        )
        self.assertEqual(
            row["attnres_world_to_policy_phase_query_norm"],
            0.22,
        )
        self.assertEqual(
            row["attnres_world_to_policy_condition_query_norm"],
            0.0,
        )
        self.assertEqual(
            row["attnres_world_to_policy_history_query_norm"],
            0.0,
        )
        self.assertEqual(
            row["grad_differential_clean_proposal_world_condition"],
            0.01,
        )
        self.assertEqual(row["grad_intent_canonical_g_to_p_query"], 0.02)
        self.assertEqual(row["grad_intent_canonical_p1_query"], 0.03)

    def test_preledger_loss_budget_is_explicitly_estimated(self) -> None:
        context = {
            "schema": "test",
            "trainer": {
                "proposal_loss_weight": 0.05,
                "event_loss_weight": 0.08,
                "rollout_dynamics_loss_weight": 0.03,
            },
        }
        record = {
            "epoch": 1,
            "global_step": 10,
            "train": {
                "loss": 1.13,
                "physical_flow": 1.0,
                "proposal": 1.0,
                "event": 1.0,
                "rollout_dynamics": 0.0,
            },
            "val": {"full_rmse": 0.2},
        }
        log = _write(
            self.tmp_path / "jsonl.log",
            json.dumps(context, indent=2) + "\n" + json.dumps(record),
        )
        summary = build_summary(parse_log(log))
        budget = summary["loss_budget"]
        self.assertEqual(budget["mode"], "estimated-known-terms")
        self.assertAlmostEqual(budget["total"], 1.13)
        self.assertAlmostEqual(budget["residual"], 0.0)

    def test_run_directory_merges_compact_nohup_with_full_epoch_jsonl(self) -> None:
        run_dir = self.tmp_path / "run"
        run_dir.mkdir()
        _write(
            run_dir / "nohup.log",
            "\n".join(
                (
                    "[v94] decoder=evidence_latent_mmdit_action rank=32 groups=32 z_probe=1",
                    "[v94-epoch] epoch=001 step=1200 loss=0.400000 pflow=0.300000 "
                    "lossgrp=action:0.30000/layer:0.10000 ledger_residual=+0.00e+00",
                    "[v94-val] rmse=0.10000 first=0.05000 first8=0.07000 tail=0.12000 "
                    "tail_first=2.400 arm=0.08000 grip=0.20000 event_ratio=4.000 "
                    "gripper_pred_events=400 gripper_target_events=100 "
                    "gripper=p:0.200/r:0.700/f1:0.310 "
                    "event_head=p:0.300/r:0.400/f1:0.340 "
                    "event_head_pred_events=150 event_head_target_events=100",
                    "[v94-probe] z_zero=2.0000e-02 z_shuffle=3.0000e-02 sample_coverage=0.25",
                )
            ),
        )
        full = {
            "epoch": 1,
            "global_step": 1200,
            "train": {"loss": 0.4, "physical_flow": 0.3},
            "val": {
                "full_rmse": 0.1,
                "event_head_accuracy": 0.9,
                "event_head_f1": 0.34,
            },
        }
        _write(run_dir / "v39_policy_epochs.jsonl", json.dumps(full))

        parsed = parse_run_input(run_dir)
        self.assertEqual(len(parsed.epoch_records), 1)
        val = parsed.epoch_records[0]["val"]
        self.assertEqual(val["gripper_event_ratio"], 4.0)
        self.assertEqual(val["sample_evidence_z_shuffle_condition_delta"], 0.03)
        self.assertEqual(val["event_head_accuracy"], 0.9)

    def test_v96_rows_preserve_late_reader_metrics(self) -> None:
        path = _write(
            self.tmp_path / "v96.log",
            "\n".join(
                (
                    "[v96-train] epoch=001 batch=0020 loss_total=1.000000 "
                    "flow_loss=0.800000 loss_groups=action:0.8/representation:0.2",
                    "[v96-repr] future_pred=0.40000 native_grid=24 coarse_grid=8 "
                    "detail_gate_mean=0.375 detail_weighted_cmp=600 "
                    "detail_candidate_cmp=1600 address_flow_mass=0.620 "
                    "address_fallback_mass=0.380",
                    "[v96-grad] flow_dino=1.0e-01 fine_flow=2.0e-02 "
                    "detail_router=3.0e-03 address_reader=4.0e-02",
                    "[v96-epoch] epoch=001 step=20 loss_total=1.000000 flow_loss=0.800000",
                    "[v96-val] action_rmse=0.20000 jepa_future=0.30000",
                    "[v96-probe] z_zero_cond_delta=1.0e-02",
                )
            ),
        )
        run = parse_log(path)
        self.assertEqual(len(run.batch_points), 1)
        row = run.batch_points[0]
        self.assertEqual(row.source, "v96")
        self.assertEqual(row.metrics["flow_jepa_native_grid_size"], 24.0)
        self.assertEqual(row.metrics["flow_jepa_address_fallback_mass"], 0.38)
        self.assertEqual(row.metrics["flow_jepa_detail_effective_comparisons"], 600.0)
        self.assertEqual(row.metrics["grad_flow_dino_detail_router"], 0.003)
        self.assertEqual(len(run.epoch_records), 1)

    def test_v100_rows_preserve_complementary_detail_semantics(self) -> None:
        path = _write(
            self.tmp_path / "v100.log",
            "\n".join(
                (
                    "[v100-train] epoch=001 batch=0020 loss_total=1.000000 flow_loss=0.800000",
                    "[v100-repr] future_pred=0.40000 change_dir=0.30000 "
                    "change_obj=0.50000 static_identity=0.02000 "
                    "raw_detail_share=0.370 raw_base_share=0.630 "
                    "detail_address_entropy=0.810 "
                    "detail_address_concentration=+0.220 "
                    "raw_dino_fused=1 refined_visual_tokens=320",
                    "[v100-grad] grounding_blocks=1.0e-02 world_blocks=2.0e-02 "
                    "policy_blocks=3.0e-02",
                    "[v100-epoch] epoch=001 step=20 loss_total=1.000000 flow_loss=0.800000",
                    "[v100-val] action_rmse=0.20000 change_obj=0.48000",
                    "[v100-probe] z_zero_cond_delta=1.0e-02",
                )
            ),
        )
        run = parse_log(path)
        self.assertEqual(len(run.batch_points), 1)
        row = run.batch_points[0]
        self.assertEqual(row.source, "v100")
        self.assertEqual(row.metrics["flow_jepa_future_change"], 0.5)
        self.assertEqual(row.metrics["flow_jepa_static_identity_loss"], 0.02)
        self.assertEqual(row.metrics["flow_jepa_raw_address_flow_mass"], 0.37)
        self.assertEqual(row.metrics["flow_jepa_raw_address_fallback_mass"], 0.63)
        self.assertEqual(row.metrics["flow_jepa_raw_address_entropy"], 0.81)
        self.assertEqual(row.metrics["flow_jepa_raw_address_logit_advantage"], 0.22)
        self.assertEqual(row.metrics["flow_jepa_raw_detail_fused_with_latest_dino"], 1.0)
        self.assertEqual(row.metrics["flow_jepa_refined_evidence_token_count"], 320.0)
        self.assertEqual(row.metrics["grad_dit_grounding_blocks"], 0.01)
        self.assertEqual(row.metrics["grad_dit_world_blocks"], 0.02)
        self.assertEqual(row.metrics["grad_dit_policy_blocks"], 0.03)
        self.assertEqual(len(run.epoch_records), 1)

    def test_v98_rows_preserve_raw_reader_and_role_gradients(self) -> None:
        path = _write(
            self.tmp_path / "v98.log",
            "\n".join(
                (
                    "[v98-train] epoch=001 batch=0020 loss_total=1.000000 flow_loss=0.800000",
                    "[v98-repr] future_pred=0.40000 raw_high_grid=84 "
                    "future_h4=0.31000 future_h48=0.52000 "
                    "raw_mid_grid=42 raw_coarse_grid=8 raw_flow=3.200 "
                    "raw_flow_grid=0.270 seed_reliability=0.180 "
                    "raw_boundary=0.0120 raw_valid=0.990 raw_precision=0.930 "
                    "raw_address_flow=0.610 "
                    "raw_address_fallback=0.390 grounding_blocks=3 "
                    "world_blocks=3 policy_blocks=2",
                    "[v98-grad] semantic_coarse_flow=1.0e-02 "
                    "raw_high_flow=2.0e-02 raw_address_reader=3.0e-02 "
                    "grounding_blocks=4.0e-02 world_blocks=5.0e-02 "
                    "policy_blocks=6.0e-02",
                    "[v98-epoch] epoch=001 step=20 loss_total=1.000000 flow_loss=0.800000",
                    "[v98-val] action_rmse=0.20000 jepa_future=0.30000 "
                    "raw_high_grid=84 raw_flow=3.100",
                    "[v98-probe] z_zero_cond_delta=1.0e-02",
                )
            ),
        )
        run = parse_log(path)
        self.assertEqual(len(run.batch_points), 1)
        row = run.batch_points[0]
        self.assertEqual(row.source, "v98")
        self.assertEqual(row.metrics["flow_jepa_raw_high_grid_size"], 84.0)
        self.assertEqual(row.metrics["flow_jepa_raw_coarse_grid_size"], 8.0)
        self.assertEqual(row.metrics["flow_jepa_raw_flow_grid_magnitude"], 0.27)
        self.assertEqual(row.metrics["flow_jepa_raw_seed_reliability"], 0.18)
        self.assertEqual(row.metrics["flow_jepa_future_horizon_4"], 0.31)
        self.assertEqual(row.metrics["flow_jepa_future_horizon_48"], 0.52)
        self.assertEqual(row.metrics["flow_jepa_raw_address_flow_mass"], 0.61)
        self.assertEqual(row.metrics["flow_jepa_grounding_block_count"], 3.0)
        self.assertEqual(row.metrics["grad_flow_dino_semantic_coarse_flow"], 0.01)
        self.assertEqual(row.metrics["grad_dit_policy_blocks"], 0.06)
        self.assertEqual(len(run.epoch_records), 1)

    def test_v99_rows_preserve_zero_flow_baseline_and_address_evidence(self) -> None:
        path = _write(
            self.tmp_path / "v99.log",
            "\n".join(
                (
                    "[v99-train] epoch=001 batch=0020 loss_total=1.000000 flow_loss=0.800000",
                    "[v99-repr] future_pred=0.40000 identity_adv=0.02000 "
                    "raw_flow_grid=0.270 zero_warp=0.1200 warp_gain=+0.0300 "
                    "moving_gain=+0.0800 static_gain=+0.0020 "
                    "moving_corr_entropy=0.620 moving_corr_margin=0.140 "
                    "motion_visible=0.180 "
                    "address_separation=0.420 address_value_delta=0.310 "
                    "address_logit_gain=+0.220 address_zero_delta=0.140 "
                    "address_shuffle_delta=0.190",
                    "[v99-grad] semantic_coarse_flow=1.0e-02 raw_high_flow=2.0e-02",
                    "[v99-epoch] epoch=001 step=20 loss_total=1.000000 flow_loss=0.800000",
                    "[v99-val] action_rmse=0.20000 identity_adv=0.01900 "
                    "warp_gain=+0.0280 moving_gain=+0.0750",
                    "[v99-probe] z_zero_cond_delta=1.0e-02",
                )
            ),
        )
        run = parse_log(path)
        self.assertEqual(len(run.batch_points), 1)
        row = run.batch_points[0]
        self.assertEqual(row.source, "v99")
        self.assertEqual(row.metrics["flow_jepa_identity_advantage_loss"], 0.02)
        self.assertEqual(row.metrics["flow_jepa_raw_identity_warp_error"], 0.12)
        self.assertEqual(row.metrics["flow_jepa_raw_warp_gain_over_zero"], 0.03)
        self.assertEqual(row.metrics["flow_jepa_raw_moving_warp_gain"], 0.08)
        self.assertEqual(row.metrics["flow_jepa_raw_static_warp_gain"], 0.002)
        self.assertEqual(row.metrics["flow_jepa_raw_moving_correlation_entropy"], 0.62)
        self.assertEqual(row.metrics["flow_jepa_raw_moving_correlation_margin"], 0.14)
        self.assertEqual(row.metrics["flow_jepa_raw_observable_motion_fraction"], 0.18)
        self.assertEqual(row.metrics["flow_jepa_raw_address_center_separation"], 0.42)
        self.assertEqual(row.metrics["flow_jepa_raw_address_lane_value_difference"], 0.31)
        self.assertEqual(row.metrics["flow_jepa_raw_address_logit_advantage"], 0.22)
        self.assertEqual(row.metrics["flow_jepa_raw_address_zero_flow_value_delta"], 0.14)
        self.assertEqual(row.metrics["flow_jepa_raw_address_shuffled_flow_value_delta"], 0.19)
        self.assertEqual(len(run.epoch_records), 1)

    def test_v113_rows_preserve_active_per_horizon_jepa_components(self) -> None:
        path = _write(
            self.tmp_path / "v113.log",
            "\n".join(
                (
                    "[v113-train] epoch=001 batch=0020 loss_total=1.000000 flow_loss=0.800000",
                    "[v113-repr] future_h4=0.21000 future_h12=0.32000 "
                    "future_direction=4:0.4000/12:0.5000 "
                    "future_active=4:0.2100/12:0.3200 "
                    "future_scale=4:0.020/12:0.040 "
                    "future_norm_scale=4:0.060/12:0.080 "
                    "future_rel=4:0.250/12:0.500",
                    "[v113-grad] global_preclip=1.00e+00",
                )
            ),
        )
        run = parse_log(path)
        self.assertEqual(len(run.batch_points), 1)
        row = run.batch_points[0].metrics
        self.assertEqual(row["flow_jepa_future_horizon_4"], 0.21)
        self.assertEqual(
            row["flow_jepa_future_horizon_4_active_direction"],
            0.4,
        )
        self.assertEqual(
            row["flow_jepa_future_horizon_12_active_loss"],
            0.32,
        )
        self.assertEqual(
            row["flow_jepa_future_horizon_4_target_scale"],
            0.02,
        )
        self.assertEqual(
            row["flow_jepa_future_horizon_12_normalization_scale"],
            0.08,
        )
        self.assertEqual(
            row["flow_jepa_future_horizon_12_reliability"],
            0.5,
        )

    def test_v114_compact_rows_are_not_silently_ignored(self) -> None:
        path = _write(
            self.tmp_path / "v114.log",
            "\n".join(
                (
                    "[v114-train] epoch=001 batch=0020 loss_total=1.000000 flow_loss=0.800000",
                    "[v114-repr] future_pred=0.31000",
                    "[v114-balance] p1_query_rows=24 p2_basis_rows=96",
                    "[v114-grad] global_preclip=1.00e+00",
                    "[v114-epoch] epoch=001 step=20 loss_total=1.000000 flow_loss=0.800000",
                    "[v114-val] action_rmse=0.20000 jepa_future=0.31000",
                )
            ),
        )
        run = parse_log(path)
        self.assertEqual(len(run.batch_points), 1)
        self.assertEqual(run.batch_points[0].source, "v114")
        self.assertEqual(
            run.batch_points[0].metrics["flow_jepa_future_prediction"],
            0.31,
        )
        self.assertEqual(len(run.epoch_records), 1)
        self.assertEqual(run.epoch_records[0]["val"]["full_rmse"], 0.2)

    def test_v116_compact_rows_keep_supervised_effect_semantics(self) -> None:
        path = _write(
            self.tmp_path / "v116.log",
            "\n".join(
                (
                    "[v116-train] epoch=001 batch=0020 loss_total=1.000000 "
                    "flow_loss=0.800000 native_velocity_mse=0.700000",
                    "[v116-repr] effect_w1_current_loss=0.0400 "
                    "effect_w2_successor_loss=0.0500 p2_effect_read=0.120 "
                    "w1_proposal_mass=0.240 phase_terminal=0.080 "
                    "execution_terminal=0.080 execution_terminal_bias=-0.010",
                    "[v116-grad] global_preclip=1.00e+00",
                    "[v116-epoch] epoch=001 step=20 loss_total=1.000000 "
                    "flow_loss=0.800000",
                    "[v116-val] action_rmse=0.19000",
                )
            ),
        )
        run = parse_log(path)
        self.assertEqual(len(run.batch_points), 1)
        self.assertEqual(run.batch_points[0].source, "v116")
        row = run.batch_points[0].metrics
        self.assertEqual(row["native_velocity_mse"], 0.7)
        self.assertEqual(
            row["flow_jepa_future_effect_w1_current_loss"],
            0.04,
        )
        self.assertEqual(
            row["flow_jepa_p2_structured_effect_read_rms"],
            0.12,
        )
        self.assertEqual(
            row["flow_jepa_execution_terminal_probability"],
            0.08,
        )
        self.assertEqual(len(run.epoch_records), 1)
        self.assertEqual(run.epoch_records[0]["val"]["full_rmse"], 0.19)

    def test_v101_rows_preserve_temporal_balance_contract(self) -> None:
        path = _write(
            self.tmp_path / "v101.log",
            "\n".join(
                (
                    "[v101-train] epoch=001 batch=0020 loss_total=1.000000 flow_loss=0.800000",
                    "[v101-repr] future_pred=0.40000 future_h4=0.31000 "
                    "future_h12=0.36000 future_h24=0.42000",
                    "[v101-balance] flow_without_info_balance=0.790000 "
                    "trajectory_info=0.1200 info_effective_fraction=1.000 "
                    "horizon_weight_first=0.955 horizon_weight_tail=1.091 "
                    "history_keep=0.875 goal_keep=1.000 proposal_keep=0.750 "
                    "teacher_past_quota=0.250 teacher_change_quota=0.500 "
                    "teacher_uniform_quota=0.250 selected_change_ratio=1.240 "
                    "action_h1_4=0.700000 action_h5_12=0.790000 "
                    "action_h13_24=0.880000",
                    "[v101-exec] top_policy_fixed_fusion=1",
                    "[v101-grad] top_policy_lift=2.0e-02",
                    "[v101-epoch] epoch=001 step=20 loss_total=1.000000 flow_loss=0.800000",
                    "[v101-val] action_rmse=0.20000 "
                    "action_band_rmse=1_4:0.12000/5_12:0.18000/13_24:0.26000",
                    "[v101-probe] z_zero_cond_delta=1.0e-02",
                )
            ),
        )
        run = parse_log(path)
        self.assertEqual(len(run.batch_points), 1)
        row = run.batch_points[0]
        self.assertEqual(row.source, "v101")
        self.assertEqual(row.metrics["physical_flow_no_information_balance"], 0.79)
        self.assertEqual(row.metrics["action_band_13_24_physical_flow"], 0.88)
        self.assertEqual(row.metrics["flow_jepa_teacher_mask_change_fraction"], 0.5)
        self.assertEqual(row.metrics["condition_action_history_keep"], 0.875)
        self.assertEqual(row.metrics["evidence_top_policy_workspace_fixed_fusion"], 1.0)
        self.assertEqual(row.metrics["grad_evidence_top_policy_workspace_lift"], 0.02)
        self.assertEqual(len(run.epoch_records), 1)
        self.assertEqual(run.epoch_records[0]["val"]["action_band_13_24_rmse"], 0.26)

    def test_v102_rows_preserve_late_detail_and_world_contract(self) -> None:
        path = _write(
            self.tmp_path / "v102.log",
            "\n".join(
                (
                    "[v102-train] epoch=001 batch=0020 loss_total=1.000000 flow_loss=0.800000",
                    "[v102-repr] world_xy_residual=0.000e+00 "
                    "world_anchor_residual=0.420 late_detail_entropy=0.610 "
                    "late_detail_max=0.120 late_detail_update=0.330 "
                    "late_detail_ratio=0.070 late_detail_scale=0.250 "
                    "late_detail_tokens=128",
                    "[v102-balance] flow_without_info_balance=0.790000",
                    "[v102-exec] top_policy_fixed_fusion=1",
                    "[v102-grad] late_detail_reader=2.0e-02",
                    "[v102-epoch] epoch=001 step=20 loss_total=1.000000 flow_loss=0.800000",
                    "[v102-val] action_rmse=0.20000 world_xy_residual=0.000e+00 "
                    "late_detail_update=0.330",
                    "[v102-probe] late_detail_update=0.330 "
                    "late_detail_entropy=0.600 late_detail_max=0.125 "
                    "late_detail_scale=0.250 late_detail_tokens=128 "
                    "world_xy_residual=0.00e+00 world_anchor_residual=0.410",
                )
            ),
        )
        run = parse_log(path)
        self.assertEqual(len(run.batch_points), 1)
        row = run.batch_points[0]
        self.assertEqual(row.source, "v102")
        self.assertEqual(row.metrics["flow_jepa_world_spatial_residual_norm"], 0.0)
        self.assertEqual(row.metrics["flow_jepa_late_detail_update_norm"], 0.33)
        self.assertEqual(row.metrics["flow_jepa_late_detail_fixed_scale"], 0.25)
        self.assertEqual(row.metrics["flow_jepa_late_detail_token_count"], 128.0)
        self.assertEqual(row.metrics["grad_late_raw_detail_reader"], 0.02)
        self.assertEqual(len(run.epoch_records), 1)
        val = run.epoch_records[0]["val"]
        self.assertEqual(val["sample_flow_jepa_late_detail_attention_entropy"], 0.6)
        self.assertEqual(val["sample_flow_jepa_world_anchor_camera_residual_norm"], 0.41)

    def test_mainline_compact_rows_preserve_batch_and_validation_semantics(self) -> None:
        path = _write(
            self.tmp_path / "mainline.log",
            "\n".join(
                (
                    "[mainline] capability=object_intent_dynamics_323 schema=19",
                    "[mainline-train] epoch=001 batch=0020 step=20 "
                    "loss_total=1.2 loss_action_flow=0.8 loss_action_flow_native=0.7",
                    "[mainline-train-top] epoch=001 batch=0020 step=20 "
                    "object_grounding_object_content_pair_cosine=0.4",
                    "[mainline-train-bottom] epoch=001 batch=0020 step=20 "
                    "bottom_capacity_mean=0.9 bottom_expected_depth=2.5",
                    "[mainline-val] epoch=001 step=100 "
                    "validation_action_rmse_normalized=0.2 "
                    "validation_action_rmse_physical=0.08 "
                    "validation_first_rmse_physical=0.03 "
                    "validation_tail_rmse_physical=0.1",
                    "[mainline-val-detail] epoch=001 step=100 "
                    "validation_gripper_event_f1_normalized=0.35 "
                    "validation_motion_f1_normalized=0.82",
                )
            ),
        )
        run = parse_log(path)
        self.assertEqual(run.header_config["capability"], "object_intent_dynamics_323")
        self.assertEqual(len(run.batch_points), 1)
        point = run.batch_points[0]
        self.assertEqual(point.source, "mainline")
        self.assertEqual(point.metrics["physical_flow"], 0.8)
        self.assertEqual(point.metrics["physical_flow_native"], 0.7)
        self.assertEqual(point.metrics["evidence_mmd_it_capacity_ratio"], 0.9)
        self.assertEqual(
            point.metrics["object_grounding_object_content_pair_cosine"], 0.4
        )
        self.assertEqual(len(run.epoch_records), 1)
        validation = run.epoch_records[0]["validation"]
        self.assertEqual(validation["full_rmse"], 0.08)
        self.assertEqual(validation["first_rmse"], 0.03)
        self.assertEqual(validation["tail_rmse"], 0.1)
        self.assertEqual(validation["gripper_event_f1"], 0.35)
        self.assertEqual(validation["motion_head_f1"], 0.82)

    def test_mainline_metrics_jsonl_is_a_first_class_run_input(self) -> None:
        path = _write(
            self.tmp_path / "metrics.jsonl",
            "\n".join(
                (
                    json.dumps(
                        {
                            "kind": "train",
                            "epoch": 1,
                            "batch": 20,
                            "step": 20,
                            "metrics": {
                                "loss_total": 1.0,
                                "loss_action_flow": 0.75,
                                "loss_action_flow_v120_comparable": 0.50,
                                "loss_action_gripper_flow": 0.42,
                                "loss_action_gripper_flow_unweighted": 0.31,
                                "loss_decoded_action": 0.20,
                                "loss_decoded_action_v120_comparable": 0.15,
                                "loss_execution_value": 0.07,
                                "loss_execution_value_target_spread": 0.11,
                                "loss_execution_value_predicted_spread": 0.09,
                                "loss_execution_terminal_target_cost_margin": -0.03,
                                "object_grounding_reconstruction_mse": 0.12,
                                "object_intent_interval_variation": 0.08,
                                "object_w2_object_pair_cosine": 0.6,
                                "object_p3_precision_base_rms": 0.2,
                                "gradient_postclip_grounder_l2": 0.03,
                                "gradient_postclip_p3_compiler_l2": 0.04,
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "kind": "epoch",
                            "epoch": 1,
                            "step": 100,
                            "train": {"loss_total": 0.9},
                            "validation": {
                                "validation_action_rmse_physical": 0.081,
                                "validation_action_rmse_normalized": 0.19,
                                "validation_band_13_24_rmse_physical": 0.1,
                                "validation_gripper_event_f1_normalized": 0.31,
                                "validation_ablation_coverage": 0.06,
                                "validation_diagnostic_primary_rmse_physical": 0.08,
                                "validation_proposal_zero_mse_gain_vs_primary_physical": 0.0001,
                                "validation_proposal_zero_action_delta_rmse_physical": 0.01,
                            },
                        }
                    ),
                )
            ),
        )
        run = parse_log(path)
        self.assertEqual(len(run.batch_points), 1)
        self.assertEqual(run.batch_points[0].metrics["physical_flow"], 0.50)
        self.assertEqual(
            run.batch_points[0].metrics["physical_flow_event_balanced"], 0.75
        )
        self.assertEqual(run.batch_points[0].metrics["gripper_fm_field"], 0.31)
        self.assertEqual(
            run.batch_points[0].metrics["gripper_fm_field_event_balanced"], 0.42
        )
        self.assertEqual(run.batch_points[0].metrics["decoded_action"], 0.15)
        self.assertEqual(
            run.batch_points[0].metrics["decoded_action_event_balanced"], 0.20
        )
        self.assertEqual(
            run.batch_points[0].metrics[
                "evidence_mmd_it_execution_value_loss"
            ],
            0.07,
        )
        self.assertEqual(
            run.batch_points[0].metrics[
                "evidence_mmd_it_execution_value_predicted_spread"
            ],
            0.09,
        )
        self.assertEqual(
            run.batch_points[0].metrics[
                "evidence_mmd_it_terminal_target_cost_margin"
            ],
            -0.03,
        )
        self.assertEqual(len(run.epoch_records), 1)
        self.assertEqual(run.epoch_records[0]["validation"]["full_rmse"], 0.081)
        summary = build_summary(run)
        self.assertEqual(
            summary["structure"]["object_grounding_reconstruction_mse"][
                "tail_median"
            ],
            0.12,
        )
        self.assertEqual(
            summary["structure"]["object_intent_interval_variation"][
                "tail_median"
            ],
            0.08,
        )
        self.assertEqual(
            summary["structure"]["object_p3_precision_base_rms"]["tail_median"],
            0.2,
        )
        self.assertEqual(
            summary["gradients"]["gradient_postclip_grounder_l2"]["tail_median"],
            0.03,
        )
        self.assertEqual(
            summary["gradients"]["gradient_postclip_p3_compiler_l2"][
                "tail_median"
            ],
            0.04,
        )
        self.assertIn(
            "proposal-zero-better",
            {finding["code"] for finding in summary["findings"]},
        )
        self.assertEqual(
            summary["epochs"][0]["val"]["validation_ablation_coverage"],
            0.06,
        )
        self.assertEqual(
            summary["epochs"][0]["val"][
                "validation_band_13_24_rmse_physical"
            ],
            0.1,
        )

    def test_mainline_run_directory_loads_metrics_and_serialized_identity(self) -> None:
        run_dir = self.tmp_path / "schema19_run"
        run_dir.mkdir()
        _write(
            run_dir / "metrics.jsonl",
            json.dumps(
                {
                    "kind": "epoch",
                    "epoch": 1,
                    "step": 100,
                    "train": {
                        "loss_total": 0.9,
                        "runtime_seconds_per_batch": 1.25,
                        "runtime_samples_per_second": 6.4,
                        "runtime_cuda_peak_allocated_gib": 8.0,
                        "runtime_cuda_peak_reserved_gib": 9.0,
                        "runtime_cuda_peak_process_estimate_gib": 9.8,
                    },
                    "validation": {"validation_action_rmse_physical": 0.08},
                }
            ),
        )
        _write(
            run_dir / "run_context.json",
            json.dumps(
                {
                    "config": {
                        "data": {
                            "seed": 0,
                            "raw_hdf5_root": "/dataset",
                            "train_episodes": 63,
                            "val_episodes": 5,
                        },
                        "optimizer": {"batch_size": 8, "warmup_steps": 500},
                        "bottom": {
                            "operator_rank": 32,
                            "operator_groups": 32,
                            "operator_depth_logit_init": 2.25,
                        },
                    },
                    "identity": {
                        "manifest": {
                            "capability": "object_intent_dynamics_323",
                            "schema": 19,
                            "layout": "clearvla_mainline",
                            "layout_schema": 1,
                        },
                        "dataset": {"action_normalizer_sha256": "a" * 64},
                    },
                    "normalizer_fingerprints": {"action_v120": "b" * 12},
                }
            ),
        )
        summary = build_summary(parse_run_input(run_dir))
        self.assertEqual(summary["coverage"]["epoch_records"], 1)
        self.assertEqual(summary["coverage"]["context_schema"], 19)
        manifest = summary["manifest"]
        self.assertEqual(manifest["capability"], "object_intent_dynamics_323")
        self.assertEqual(manifest["architecture_schema"], 19)
        self.assertEqual(manifest["layout"], "clearvla_mainline")
        self.assertEqual(manifest["seed"], 0)
        self.assertEqual(manifest["batch_size"], 8)
        self.assertEqual(manifest["data_root"], "/dataset")
        self.assertEqual(manifest["train_episode_count"], 63)
        self.assertEqual(manifest["val_episode_count"], 5)
        self.assertEqual(manifest["action_normalizer_fingerprint"], "b" * 12)
        self.assertEqual(manifest["action_normalizer_sha256"], "a" * 64)
        self.assertEqual(manifest["rank"], 32)
        self.assertEqual(manifest["groups"], 32)
        self.assertEqual(manifest["warmup"], 500)
        self.assertEqual(summary["performance"]["seconds_per_batch"]["median"], 1.25)
        self.assertEqual(summary["performance"]["cuda_peak_reserved_gib"], 9.0)
        self.assertEqual(
            summary["performance"]["cuda_peak_process_estimate_gib"],
            9.8,
        )

    def test_core_attribution_is_lossless_validation_only_and_coverage_checked(
        self,
    ) -> None:
        run_dir = self.tmp_path / "schema28_core_attribution"
        run_dir.mkdir()
        validation = {
            "validation_action_rmse_physical": 0.08,
            "validation_core_attribution_coverage": 0.2,
            "validation_core_attribution_primary_vs_explicit_none_normalized_action_max_abs": 0.0,
            "validation_core_attribution_primary_vs_explicit_none_normalized_bit_exact": 1.0,
            "validation_core_attribution_world_vs_consequence_neutral_normalized_action_max_abs": 0.0,
            "validation_core_attribution_world_vs_consequence_neutral_normalized_bit_exact": 1.0,
            "validation_core_attribution_wrong_action_world_donor_valid_fraction": 1.0,
            "validation_core_attribution_consequence_effect_neutral_first_boundary_delta_rms": 0.03,
        }
        _write(
            run_dir / "metrics.jsonl",
            json.dumps(
                {
                    "kind": "epoch",
                    "epoch": 1,
                    "step": 100,
                    "train": {"loss_total": 0.9},
                    "validation": validation,
                }
            ),
        )
        _write(
            run_dir / "run_context.json",
            json.dumps(
                {
                    "config": {
                        "data": {"seed": 0},
                        "optimizer": {"batch_size": 8},
                    },
                    "identity": {
                        "manifest": {
                            "capability": "object_intent_dynamics_323",
                            "schema": 28,
                            "layout": "clearvla_mainline",
                            "layout_schema": 1,
                        }
                    },
                }
            ),
        )
        summary = build_summary(parse_run_input(run_dir))
        recorded = summary["epochs"][0]["val"]
        for name, value in validation.items():
            self.assertEqual(recorded[name], value)
        self.assertEqual(recorded["eval_core_attribution_coverage"], 0.2)
        self.assertNotIn(
            "validation_core_attribution_consequence_effect_neutral_first_boundary_delta_rms",
            summary["trajectories"],
        )

        baseline = _complete_recovery_summary("v120")
        candidate = deepcopy(baseline)
        candidate["label"] = "schema28-attribution"
        candidate["manifest"]["architecture_schema"] = 28
        candidate["trajectories"]["gripper_trajectory"] = {
            "count": 100,
            "tail_median": 0.12,
        }
        latest = candidate["epochs"][-1]["val"]
        latest.update(
            {
                "validation_sampling_diagnostic_coverage": 0.2,
                "validation_p2_intervention_coverage": 0.2,
                "validation_proposal_ablation_coverage": 0.2,
                "validation_execution_ablation_coverage": 0.2,
                **validation,
            }
        )
        assessment = _recovery_assessment(baseline, candidate)
        checks = {item["name"]: item["status"] for item in assessment["checks"]}
        self.assertEqual(checks["causal_ablation/core_attribution_coverage"], "pass")
        self.assertEqual(
            checks["core_attribution/primary_explicit_none_identity"],
            "pass",
        )
        self.assertEqual(
            checks["core_attribution/world_consequence_sole_consumer_identity"],
            "pass",
        )
        self.assertEqual(
            checks["core_attribution/wrong_world_donor_identifiability"],
            "pass",
        )

        latest[
            "validation_core_attribution_primary_vs_explicit_none_normalized_action_max_abs"
        ] = 1e-7
        latest[
            "validation_core_attribution_wrong_action_world_donor_valid_fraction"
        ] = 0.0
        assessment = _recovery_assessment(baseline, candidate)
        checks = {item["name"]: item["status"] for item in assessment["checks"]}
        self.assertEqual(
            checks["core_attribution/primary_explicit_none_identity"],
            "fail",
        )
        self.assertEqual(
            checks["core_attribution/wrong_world_donor_identifiability"],
            "incomplete",
        )

    def test_mainline_console_runtime_rows_remain_auditable(self) -> None:
        log = _write(
            self.tmp_path / "mainline_runtime.log",
            "\n".join(
                (
                    "[mainline] capability=object_intent_dynamics_323 batch=8",
                    "[mainline-train] epoch=001 batch=0020 step=20 loss_total=1",
                    "[mainline-train-performance] epoch=001 batch=0020 step=20 "
                    "runtime_window_seconds_per_batch=1.2 "
                    "runtime_window_samples_per_second=6.66667",
                    "[mainline-runtime] epoch=001 step=100 "
                    "runtime_seconds_per_batch=1.3 "
                    "runtime_cuda_peak_reserved_gib=9.1 "
                    "runtime_cuda_peak_process_estimate_gib=9.8",
                    "[mainline-val] epoch=001 step=100 validation_action_rmse_physical=0.1",
                )
            ),
        )
        summary = build_summary(parse_log(log))
        self.assertEqual(summary["performance"]["seconds_per_batch"]["source"], "window")
        self.assertEqual(summary["performance"]["seconds_per_batch"]["median"], 1.2)
        self.assertEqual(summary["performance"]["cuda_peak_reserved_gib"], 9.1)
        self.assertEqual(
            summary["performance"]["cuda_peak_process_estimate_gib"],
            9.8,
        )

    def test_v120_recovery_assessment_requires_complete_behavior_not_one_rmse(self) -> None:
        baseline = _complete_recovery_summary("v120")
        candidate = deepcopy(baseline)
        candidate["label"] = "candidate"
        assessment = _recovery_assessment(baseline, candidate)
        self.assertEqual(assessment["status"], "pass")
        self.assertEqual(assessment["failed"], 0)
        self.assertEqual(assessment["incomplete"], 0)

        candidate["manifest"]["action_normalizer_fingerprint"] = "a" * 64
        assessment = _recovery_assessment(baseline, candidate)
        identity_check = next(
            item
            for item in assessment["checks"]
            if item["name"] == "identity/action_normalizer"
        )
        self.assertEqual(identity_check["status"], "incomplete")

        candidate["manifest"]["action_normalizer_fingerprint"] = "a" * 12
        candidate["epochs"][-1]["val"]["full_rmse"] = 0.12
        candidate["structure"]["object_w2_object_pair_cosine"][
            "tail_median"
        ] = 1.0
        candidate["gradients"]["gradient_postclip_bottom_capacity_l2"][
            "tail_median"
        ] = 0.0
        assessment = _recovery_assessment(baseline, candidate)
        self.assertEqual(assessment["status"], "fail")
        failed = {
            item["name"] for item in assessment["checks"] if item["status"] == "fail"
        }
        self.assertIn("validation/action_rmse", failed)
        self.assertIn("structure/object_w2_object_pair_cosine", failed)
        self.assertIn(
            "gradient/gradient_postclip_bottom_capacity_l2",
            failed,
        )

    def test_v120_recovery_assessment_enforces_runtime_and_memory_envelope(self) -> None:
        baseline = _complete_recovery_summary("v120")
        candidate = deepcopy(baseline)
        candidate["label"] = "candidate"
        candidate["performance"]["seconds_per_batch"]["median"] = 3.01
        candidate["performance"]["seconds_per_batch"]["p90"] = 4.81
        candidate["performance"]["cuda_peak_process_estimate_gib"] = 22.01
        assessment = _recovery_assessment(baseline, candidate)
        failed = {
            item["name"] for item in assessment["checks"] if item["status"] == "fail"
        }
        self.assertIn("performance/seconds_per_batch_median", failed)
        self.assertIn("performance/seconds_per_batch_p90", failed)
        self.assertIn("performance/cuda_peak_process_gib", failed)

        candidate = deepcopy(baseline)
        candidate["performance"]["cuda_peak_process_estimate_gib"] = None
        assessment = _recovery_assessment(baseline, candidate)
        incomplete = {
            item["name"]
            for item in assessment["checks"]
            if item["status"] == "incomplete"
        }
        self.assertIn("performance/cuda_peak_process_gib", incomplete)

    def test_schema26_recovery_uses_continuous_gripper_and_p2_intervention_surface(
        self,
    ) -> None:
        baseline = _complete_recovery_summary("v120")
        candidate = deepcopy(baseline)
        candidate["label"] = "schema26"
        candidate["manifest"]["architecture_schema"] = 26
        candidate["trajectories"]["gripper_trajectory"] = {
            "count": 100,
            "tail_median": 0.12,
        }
        for record in candidate["epochs"]:
            record["val"].pop("event_head_f1")
        latest = candidate["epochs"][-1]["val"]
        latest.update(
            {
                "validation_sampling_diagnostic_coverage": 0.1,
                "validation_p2_intervention_coverage": 0.1,
                "validation_proposal_ablation_coverage": 0.1,
                "validation_execution_ablation_coverage": 0.1,
                "validation_proposal_primary_rmse_physical": 0.08,
                "validation_proposal_zero_mse_gain_vs_primary_physical": 0.0,
                "validation_execution_primary_rmse_physical": 0.08,
                "validation_execution_hard_mse_gain_vs_primary_physical": 0.0,
                "validation_execution_neutral_mse_gain_vs_primary_physical": 0.0,
                "validation_execution_full_capacity_mse_gain_vs_primary_physical": 0.0,
                "validation_execution_three_basis_reduction_mse_gain_vs_primary_physical": 0.0,
            }
        )

        assessment = _recovery_assessment(baseline, candidate)
        checks = {item["name"]: item["status"] for item in assessment["checks"]}

        self.assertNotIn("validation/event_head_f1", checks)
        self.assertEqual(checks["objective/gripper_trajectory_observed"], "pass")
        self.assertEqual(
            checks["causal_ablation/p2_intervention_coverage"],
            "pass",
        )

    def test_v120_recovery_assessment_reports_missing_epochs_without_index_error(self) -> None:
        baseline = _complete_recovery_summary("v120")
        candidate = deepcopy(baseline)
        candidate["label"] = "unfinished"
        candidate["coverage"]["epoch_records"] = 0
        candidate["epochs"] = []
        assessment = _recovery_assessment(baseline, candidate)
        checks = {item["name"]: item["status"] for item in assessment["checks"]}
        self.assertEqual(checks["coverage/all_epochs"], "fail")
        self.assertEqual(checks["validation/action_rmse"], "incomplete")
        self.assertEqual(
            checks["validation/gripper_event_rate_calibration"],
            "incomplete",
        )

    def test_aligned_batch_2200_does_not_use_later_epoch_tail(self) -> None:
        run = ParsedRun(path=self.tmp_path / "aligned.log", label="aligned")
        run.batch_points = [
            BatchPoint(1, 20, {"physical_flow": 1.0}, "mainline"),
            BatchPoint(1, 2200, {"physical_flow": 0.4}, "mainline"),
            BatchPoint(2, 20, {"physical_flow": 0.01}, "mainline"),
        ]
        summary = build_summary(run, tail=2)
        self.assertEqual(
            summary["aligned_batch_2200"]["physical_flow"]["tail_median"],
            0.7,
        )
        self.assertAlmostEqual(
            summary["trajectories"]["physical_flow"]["tail_median"],
            0.205,
        )

    def test_three_way_early_gate_requires_half_schema23_to_v120_gap_closure(self) -> None:
        baseline = _complete_recovery_summary("v120")
        parent = deepcopy(baseline)
        parent["label"] = "schema23"
        candidate = deepcopy(baseline)
        candidate["label"] = "schema24"
        parent["aligned_batch_2200"]["object_grounding_g3_parent_l1"][
            "tail_median"
        ] = 0.10
        candidate["aligned_batch_2200"]["object_grounding_g3_parent_l1"][
            "tail_median"
        ] = 0.055
        assessment = _recovery_assessment(baseline, candidate, parent)
        check = next(
            item
            for item in assessment["checks"]
            if item["name"] == "early_batch_2200/g3_parent_l1"
        )
        self.assertEqual(check["status"], "pass")
        self.assertAlmostEqual(check["candidate"]["gap_closure"], 0.5625)

        candidate["aligned_batch_2200"]["object_grounding_g3_parent_l1"][
            "tail_median"
        ] = 0.07
        assessment = _recovery_assessment(baseline, candidate, parent)
        check = next(
            item
            for item in assessment["checks"]
            if item["name"] == "early_batch_2200/g3_parent_l1"
        )
        self.assertEqual(check["status"], "fail")


if __name__ == "__main__":
    unittest.main()
