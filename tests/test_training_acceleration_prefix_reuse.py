from __future__ import annotations

import copy

import torch
from test_mainline_policy import _batch, _config

from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
from clearvla.mainline.training.engine import MainlineTrainingEngine
from clearvla.mainline.training.optimizer import WarmupCosineSchedule, build_optimizer


def _engine(model: ClearVLAMainlinePolicy, config):
    optimizer, _ = build_optimizer(model, config)
    schedule = WarmupCosineSchedule(
        optimizer,
        warmup_steps=2,
        total_steps=4,
        minimum_ratio=0.1,
    )
    return MainlineTrainingEngine(
        model=model,
        config=config,
        optimizer=optimizer,
        schedule=schedule,
        device=torch.device("cpu"),
        dtype=torch.float32,
        train_flow_generator=torch.Generator().manual_seed(9102),
        train_condition_generator=torch.Generator().manual_seed(9103),
        skip_postglobal_audit=True,
        gradient_spike_audit_threshold=None,
    )


def test_training_candidate_prefix_reuse_preserves_update_within_tolerance() -> None:
    config = _config()
    torch.manual_seed(9100)
    reference = ClearVLAMainlinePolicy(config).train()
    initial = copy.deepcopy(reference.state_dict())
    optimized = ClearVLAMainlinePolicy(config).train()
    optimized.load_state_dict(initial)
    optimized.execution_bottom.decoder._training_candidate_prefix_reuse = True

    reference_engine = _engine(reference, config)
    optimized_engine = _engine(optimized, config)
    batch = _batch(config, batch=2)

    torch.manual_seed(9104)
    reference_result = reference_engine.train_step(batch, collect_diagnostics=False)
    torch.manual_seed(9104)
    optimized_result = optimized_engine.train_step(batch, collect_diagnostics=False)

    torch.testing.assert_close(
        optimized_result.loss,
        reference_result.loss,
        rtol=2e-4,
        atol=2e-5,
    )
    for (reference_name, reference_parameter), (
        optimized_name,
        optimized_parameter,
    ) in zip(
        reference.named_parameters(), optimized.named_parameters(), strict=True
    ):
        assert optimized_name == reference_name
        torch.testing.assert_close(
            optimized_parameter,
            reference_parameter,
            rtol=5e-4,
            atol=5e-5,
        )
        if reference_parameter.grad is None or optimized_parameter.grad is None:
            assert reference_parameter.grad is optimized_parameter.grad
        else:
            torch.testing.assert_close(
                optimized_parameter.grad,
                reference_parameter.grad,
                rtol=5e-4,
                atol=5e-5,
            )


def test_prepared_block_contexts_preserve_training_update_exactly() -> None:
    """Common-subexpression reuse must not change the formal training update."""

    config = _config()
    torch.manual_seed(9110)
    reference = ClearVLAMainlinePolicy(config).train()
    initial = copy.deepcopy(reference.state_dict())
    optimized = ClearVLAMainlinePolicy(config).train()
    optimized.load_state_dict(initial)
    optimized.execution_bottom.decoder._reuse_prepared_block_contexts = True

    reference_engine = _engine(reference, config)
    optimized_engine = _engine(optimized, config)
    batch = _batch(config, batch=2)

    torch.manual_seed(9114)
    reference_result = reference_engine.train_step(batch, collect_diagnostics=False)
    torch.manual_seed(9114)
    optimized_result = optimized_engine.train_step(batch, collect_diagnostics=False)

    torch.testing.assert_close(
        optimized_result.loss,
        reference_result.loss,
        rtol=0.0,
        atol=0.0,
    )
    for (reference_name, reference_parameter), (
        optimized_name,
        optimized_parameter,
    ) in zip(
        reference.named_parameters(), optimized.named_parameters(), strict=True
    ):
        assert optimized_name == reference_name
        torch.testing.assert_close(
            optimized_parameter,
            reference_parameter,
            rtol=0.0,
            atol=0.0,
            msg=reference_name,
        )
        if reference_parameter.grad is None or optimized_parameter.grad is None:
            assert reference_parameter.grad is optimized_parameter.grad
        else:
            torch.testing.assert_close(
                optimized_parameter.grad,
                reference_parameter.grad,
                rtol=0.0,
                atol=0.0,
                msg=f"gradient: {reference_name}",
            )
