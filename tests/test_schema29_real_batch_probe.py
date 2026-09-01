from __future__ import annotations

import torch
from test_mainline_policy import _batch, _config

from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
from clearvla.mainline.training.losses import sample_flow_matching
from clearvla.tools.probe_schema29_real_batch import run_schema29_real_batch_probe


def test_schema29_ab_probe_separates_velocity_motion_and_gripper_vjps() -> None:
    torch.manual_seed(2910)
    config = _config()
    model = ClearVLAMainlinePolicy(config)
    batch = _batch(config, batch=2)
    report = run_schema29_real_batch_probe(
        model=model,
        config=config,
        batch=batch,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        flow_generator=torch.Generator().manual_seed(29102),
        condition_generator=torch.Generator().manual_seed(29101),
    )

    assert report["schema"] == "clearvla-schema29-real-batch-gradient-ab-v1"
    assert report["optimizer_constructed"] is False
    assert report["optimizer_step_taken"] is False
    assert all(parameter.grad is None for parameter in model.parameters())

    modes = report["modes"]
    baseline = modes["cache0_single"]["losses_and_vjps"]
    formal = modes["cache1_self_conditioned"]["losses_and_vjps"]
    for rows in (baseline, formal):
        assert rows["contrib_action_flow"][
            "velocity_output_parameter_gradient_l2"
        ] > 0.0
        assert rows["contrib_action_flow"]["physical_velocity_gradient_rms"] > 0.0
        assert rows["contrib_action_flow"][
            "gripper_gate_parameter_gradient_l2"
        ] > 0.0
        assert rows["contrib_action_flow"]["motion_head_parameter_gradient_l2"] == 0.0
        assert rows["contrib_motion"]["motion_head_parameter_gradient_l2"] > 0.0
        assert rows["contrib_motion"][
            "velocity_output_parameter_gradient_l2"
        ] == 0.0
        assert rows["contrib_motion"]["gripper_gate_parameter_gradient_l2"] == 0.0
        assert rows["group_representation"][
            "velocity_output_parameter_gradient_l2"
        ] == 0.0

    paired = report["paired_boundaries"]
    assert paired["velocity_head_input"]["sample_flat_cosine"] > 0.99
    assert paired["physical_velocity"]["sample_flat_cosine"] > 0.99
    assert report["relative_decision"]["action_flow_to_velocity_output_layers"][
        "classification"
    ] == "no_large_cache0_cache1_difference"


def test_probe_flow_generator_contract_matches_direct_sampler() -> None:
    """Keep the fake-batch probe on the same owned flow-state entry contract."""

    config = _config()
    model = ClearVLAMainlinePolicy(config)
    batch = _batch(config, batch=1)
    generator = torch.Generator().manual_seed(29102)
    state = sample_flow_matching(
        batch.action_target.normalized,
        action_state=batch.online.history.action_state,
        codec=model.action_codec,
        distribution=config.bottom.flow_time_distribution,
        generator=generator,
    )
    assert tuple(state.noisy_physical.shape) == (
        1,
        config.dimensions.action_horizon,
        model.action_codec.physical_dim,
    )
    assert torch.isfinite(state.noisy_physical).all()
