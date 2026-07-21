import numpy as np
import torch
from pathlib import Path
from tempfile import TemporaryDirectory

from clearvla.experiments.observed_state_lab.policy_runtime_v39 import (
    V39PolicyTrainerConfig,
    _attach_v94_loss_ledger,
    _evidence_epoch_log_line,
    _evidence_serial_log_line,
    _layer_contract_aux_scale,
    _prepare_run_directory,
    motion_head_metrics,
)
from clearvla.policy.system import _keep_sampling_diagnostic


def test_v94_layer_contract_aux_schedule_is_independent_and_constant():
    trainer = V39PolicyTrainerConfig(
        midcut_aux_loss_weight=0.03,
        midcut_aux_final_ratio=0.15,
        midcut_aux_decay_epochs=4,
        layer_contract_aux_loss_weight=0.03,
        layer_contract_aux_final_ratio=1.0,
        layer_contract_aux_decay_epochs=0,
    )
    assert _layer_contract_aux_scale(trainer, 1) == 0.03
    assert _layer_contract_aux_scale(trainer, 4) == 0.03
    assert _layer_contract_aux_scale(trainer, 40) == 0.03


def test_v94_loss_ledger_reconstructs_the_backward_scalar():
    trainer = V39PolicyTrainerConfig(
        latent_cvae_mmdit_execution_value_loss_weight=0.5,
    )
    losses = {
        "loss": torch.tensor(6.87),
        "physical_flow": torch.tensor(2.0),
        "proposal": torch.tensor(1.0),
        "event": torch.tensor(2.0),
        "motion": torch.tensor(3.0),
        "rollout_dynamics": torch.tensor(4.0),
        "rollout_milestone_delta_match": torch.tensor(5.0),
        "evidence_mmd_it_execution_value_loss": torch.tensor(6.0),
    }
    _attach_v94_loss_ledger(
        losses,
        trainer,
        enable_future_loss=True,
        layer_aux_contribution=torch.tensor(0.7),
    )
    assert torch.isclose(losses["loss_group_action"], torch.tensor(2.30))
    assert torch.isclose(losses["loss_group_rollout"], torch.tensor(0.87))
    assert torch.isclose(losses["loss_group_execution"], torch.tensor(3.0))
    assert torch.isclose(losses["loss_group_layer"], torch.tensor(0.7))
    assert torch.isclose(losses["loss_ledger_sum"], losses["loss"])
    assert abs(float(losses["loss_ledger_residual"])) < 1e-6


def test_v94_evidence_console_log_uses_only_present_active_fields():
    line = _evidence_serial_log_line(
        {
            "loss": 0.4,
            "physical_flow": 0.3,
            "motion": 0.2,
            "loss_group_action": 0.32,
            "loss_group_rollout": 0.08,
            "loss_contrib_flow": 0.30,
            "loss_contrib_event": 0.02,
            "loss_ledger_residual": 0.0,
            "evidence_mmd_it_capacity_gate_mass": 0.91,
            "evidence_mmd_it_effective_basis_mass": 29.0,
            "evidence_mmd_it_terminal_prior_weight": 0.25,
            "evidence_mmd_it_dynamic_route_next_fraction": 0.4,
            "evidence_mmd_it_hard_route_next_fraction": 0.25,
            "grad_evidence_mmdit_execution_controller": 0.0,
            "grad": 1.0,
        },
        epoch=1,
        batch_index=20,
        learning_rate=1e-4,
        seconds_per_batch=0.5,
    )
    assert line.count("\n") == 2
    assert "[v94-train]" in line
    assert "[v94-exec]" in line
    assert "[v94-grad]" in line
    assert "route=soft:0.400/hard:0.250/gap:+0.150" in line
    assert "top_contrib=flow:0.30000/event:0.02000" in line
    assert "exec_controller=0.00e+00" in line
    assert "loss_total=0.400000" in line
    assert "flow_loss=0.300000" in line
    assert "capacity_gate_mass=0.91000" in line
    assert "effective_basis_mass=29.000" in line
    assert "terminal_prior=0.250" in line
    assert "global_preclip=1.00e+00" in line
    assert "latent_cvae" not in line
    assert "hierarchical_mmdit" not in line


def test_v94_sampling_diagnostics_keep_z_probe_and_drop_inactive_families():
    assert _keep_sampling_diagnostic("evidence_z_zero_condition_delta", evidence_active=True)
    assert _keep_sampling_diagnostic("evidence_z_shuffle_condition_delta", evidence_active=True)
    assert _keep_sampling_diagnostic("evidence_mmd_it_block_2_update_norm", evidence_active=True)
    assert not _keep_sampling_diagnostic("latent_cvae_primary_z_effect_norm", evidence_active=True)


def test_v94_epoch_console_keeps_event_rate_and_count_evidence():
    line = _evidence_epoch_log_line(
        epoch=1,
        global_step=100,
        train={"loss": 0.4, "physical_flow": 0.3},
        val={
            "full_rmse": 0.1,
            "first_rmse": 0.05,
            "first8_rmse": 0.07,
            "tail_rmse": 0.12,
            "tail_first_ratio": 2.4,
            "arm_full_rmse": 0.08,
            "gripper_full_rmse": 0.2,
            "gripper_event_ratio": 4.0,
            "gripper_pred_events": 400.0,
            "gripper_target_events": 100.0,
            "event_head_pred_events": 150.0,
            "event_head_target_events": 100.0,
        },
    )
    assert "tail_first_ratio=2.400" in line
    assert "grip_event_ratio=4.000" in line
    assert "grip_events_pred=400" in line
    assert "event_head_events_pred=150" in line


def test_motion_head_validation_metrics_report_the_active_auxiliary_head():
    metrics = motion_head_metrics(
        [np.array([[10.0, -10.0], [-10.0, 10.0]], dtype=np.float32)],
        [np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)],
    )
    assert metrics["motion_head_accuracy"] == 1.0
    assert metrics["motion_head_precision"] == 1.0
    assert metrics["motion_head_recall"] == 1.0
    assert metrics["motion_head_f1"] == 1.0


def test_v94_run_directory_rejects_accidental_append_without_resume():
    manifest = {
        "schema": "clearvla-run-contract-manifest-v1",
        "fingerprint": "contract-a",
        "contract": {},
    }
    with TemporaryDirectory() as directory:
        out_dir = Path(directory)
        _prepare_run_directory(out_dir=out_dir, manifest=manifest, resume=None)
        assert (out_dir / "run_manifest.json").exists()
        (out_dir / "v39_policy_epochs.jsonl").write_text("{}\n", encoding="utf-8")
        try:
            _prepare_run_directory(out_dir=out_dir, manifest=manifest, resume=None)
        except FileExistsError as error:
            assert "choose a new OUT_DIR" in str(error)
        else:
            raise AssertionError("an existing run must not be appended without --resume")
