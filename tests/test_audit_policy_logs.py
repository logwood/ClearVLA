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


if __name__ == "__main__":
    unittest.main()
