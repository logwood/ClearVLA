from __future__ import annotations

import pytest
import torch
from test_mainline_policy import _batch, _config

from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
from clearvla.mainline.training.engine import _autocast
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

    assert report["schema"] == "clearvla-schema29-real-batch-gradient-ab-v2"
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

    for mode in modes.values():
        total_owners = mode["total_owner_vjps"]
        assert total_owners["all_trainable"]["gradient_l2"] > 0.0
        assert total_owners["parameter_roles"]["bottom_heads"][
            "gradient_l2"
        ] > 0.0
        for block_index in range(3):
            assert total_owners["bottom_mmdit_blocks"][f"block_{block_index}"][
                "gradient_l2"
            ] > 0.0

    paired = report["paired_boundaries"]
    assert paired["velocity_head_input"]["sample_flat_cosine"] > 0.99
    assert paired["physical_velocity"]["sample_flat_cosine"] > 0.99
    assert report["relative_decision"]["action_flow_to_velocity_output_layers"][
        "classification"
    ] == "no_large_cache0_cache1_difference"
    assert report["autocast"] == {
        "cache0_formal_cache_enabled": True,
        "cache1_pass0_cache_enabled": False,
        "cache1_formal_cache_enabled": True,
    }
    assert report["self_conditioning"]["pass0_velocity_requires_grad"] is False
    assert report["self_conditioning"]["pass0_action_requires_grad"] is False
    assert report["self_conditioning"][
        "pass0_condition_interval_action_requires_grad"
    ] is False
    assert report["self_conditioning"][
        "pass0_condition_interval_delta_requires_grad"
    ] is False
    for block_index in range(3):
        decision = report["relative_decision"]["total_loss_owners"][
            "bottom_mmdit_blocks"
        ][f"block_{block_index}"]
        assert decision["classification"] != "cache1_specific_strong_attenuation"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA BF16 autocast cache regression requires a CUDA device",
)
def test_pass0_cache_isolation_preserves_cuda_bf16_parameter_vjp() -> None:
    device = torch.device("cuda")
    layer = torch.nn.Linear(16, 16, bias=False).to(device)
    value = torch.randn(4, 16, device=device)

    with _autocast(device, torch.bfloat16):
        with torch.no_grad():
            with _autocast(device, torch.bfloat16, cache_enabled=False):
                assert torch.is_autocast_cache_enabled() is False
                detached = layer(value)
                assert detached.requires_grad is False
        assert torch.is_autocast_cache_enabled() is True
        formal = layer(value)

    gradient = torch.autograd.grad(formal.square().mean(), layer.weight)[0]
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_probe_flow_generator_contract_matches_direct_sampler() -> None:
    """Keep the fake-batch probe on the same owned flow-state entry contract."""

    config = _config()
    model = ClearVLAMainlinePolicy(config)
    batch = _batch(config, batch=1)
    generator = torch.Generator().manual_seed(29102)
    state = sample_flow_matching(
        batch.action_target.normalized,
        action_state=batch.online.history.action_state,
        codec=model.outlet_adapter.codec,
        distribution=config.bottom.flow_time_distribution,
        generator=generator,
    )
    assert tuple(state.noisy_physical.shape) == (
        1,
        config.dimensions.action_horizon,
        model.outlet_adapter.physical_dim,
    )
    assert torch.isfinite(state.noisy_physical).all()
