from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from clearvla.tools.audit_policy_logs import build_summary, parse_log, parse_run_input


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


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

    def test_unhandled_nonfinite_backward_is_a_critical_finding(self) -> None:
        log = _write(
            self.tmp_path / "nonfinite.log",
            "\n".join(
                (
                    "[v96-train] epoch=001 batch=2500 loss_total=0.138861 "
                    "flow_loss=0.081567",
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
                    "[v100-train] epoch=001 batch=0020 loss_total=1.000000 "
                    "flow_loss=0.800000",
                    "[v100-repr] future_pred=0.40000 change_dir=0.30000 "
                    "change_obj=0.50000 static_identity=0.02000 "
                    "raw_detail_share=0.370 raw_base_share=0.630 "
                    "detail_address_entropy=0.810 "
                    "detail_address_concentration=+0.220 "
                    "raw_dino_fused=1 refined_visual_tokens=320",
                    "[v100-grad] grounding_blocks=1.0e-02 world_blocks=2.0e-02 "
                    "policy_blocks=3.0e-02",
                    "[v100-epoch] epoch=001 step=20 loss_total=1.000000 "
                    "flow_loss=0.800000",
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
        self.assertEqual(
            row.metrics["flow_jepa_raw_detail_fused_with_latest_dino"], 1.0
        )
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
                    "[v98-train] epoch=001 batch=0020 loss_total=1.000000 "
                    "flow_loss=0.800000",
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
                    "[v98-epoch] epoch=001 step=20 loss_total=1.000000 "
                    "flow_loss=0.800000",
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
                    "[v99-train] epoch=001 batch=0020 loss_total=1.000000 "
                    "flow_loss=0.800000",
                    "[v99-repr] future_pred=0.40000 identity_adv=0.02000 "
                    "raw_flow_grid=0.270 zero_warp=0.1200 warp_gain=+0.0300 "
                    "moving_gain=+0.0800 static_gain=+0.0020 "
                    "moving_corr_entropy=0.620 moving_corr_margin=0.140 "
                    "motion_visible=0.180 "
                    "address_separation=0.420 address_value_delta=0.310 "
                    "address_logit_gain=+0.220 address_zero_delta=0.140 "
                    "address_shuffle_delta=0.190",
                    "[v99-grad] semantic_coarse_flow=1.0e-02 raw_high_flow=2.0e-02",
                    "[v99-epoch] epoch=001 step=20 loss_total=1.000000 "
                    "flow_loss=0.800000",
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
        self.assertEqual(
            row.metrics["flow_jepa_raw_address_lane_value_difference"], 0.31
        )
        self.assertEqual(row.metrics["flow_jepa_raw_address_logit_advantage"], 0.22)
        self.assertEqual(row.metrics["flow_jepa_raw_address_zero_flow_value_delta"], 0.14)
        self.assertEqual(
            row.metrics["flow_jepa_raw_address_shuffled_flow_value_delta"], 0.19
        )
        self.assertEqual(len(run.epoch_records), 1)

    def test_v101_rows_preserve_temporal_balance_contract(self) -> None:
        path = _write(
            self.tmp_path / "v101.log",
            "\n".join(
                (
                    "[v101-train] epoch=001 batch=0020 loss_total=1.000000 "
                    "flow_loss=0.800000",
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
                    "[v101-epoch] epoch=001 step=20 loss_total=1.000000 "
                    "flow_loss=0.800000",
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
        self.assertEqual(
            run.epoch_records[0]["val"]["action_band_13_24_rmse"], 0.26
        )

    def test_v102_rows_preserve_late_detail_and_world_contract(self) -> None:
        path = _write(
            self.tmp_path / "v102.log",
            "\n".join(
                (
                    "[v102-train] epoch=001 batch=0020 loss_total=1.000000 "
                    "flow_loss=0.800000",
                    "[v102-repr] world_xy_residual=0.000e+00 "
                    "world_anchor_residual=0.420 late_detail_entropy=0.610 "
                    "late_detail_max=0.120 late_detail_update=0.330 "
                    "late_detail_ratio=0.070 late_detail_scale=0.250 "
                    "late_detail_tokens=128",
                    "[v102-balance] flow_without_info_balance=0.790000",
                    "[v102-exec] top_policy_fixed_fusion=1",
                    "[v102-grad] late_detail_reader=2.0e-02",
                    "[v102-epoch] epoch=001 step=20 loss_total=1.000000 "
                    "flow_loss=0.800000",
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
        self.assertEqual(
            val["sample_flow_jepa_late_detail_attention_entropy"], 0.6
        )
        self.assertEqual(
            val["sample_flow_jepa_world_anchor_camera_residual_norm"], 0.41
        )


if __name__ == "__main__":
    unittest.main()
