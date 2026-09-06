from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from test_mainline_policy import _batch, _config

from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
from clearvla.mainline.training.engine import MainlineTrainingEngine
from clearvla.mainline.training.losses import LossLedger
from clearvla.mainline.training.optimizer import WarmupCosineSchedule, build_optimizer


def _engine(model: ClearVLAMainlinePolicy) -> MainlineTrainingEngine:
    config = model.config
    optimizer, _ = build_optimizer(model, config)
    schedule = WarmupCosineSchedule(optimizer, warmup_steps=2, total_steps=2400, minimum_ratio=0.1)
    return MainlineTrainingEngine(
        model=model,
        config=config,
        optimizer=optimizer,
        schedule=schedule,
        device=torch.device("cpu"),
    )


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_loss_with_finite_derivative_cannot_step_optimizer(bad: float) -> None:
    config = _config()
    model = ClearVLAMainlinePolicy(config)
    engine = _engine(model)
    parameter = next(p for p in model.parameters() if p.requires_grad)
    before = parameter.detach().clone()
    value = parameter.sum() + parameter.new_tensor(bad)
    # Explicitly exhibit the failure of a gradient-only check.
    derivative = torch.autograd.grad(value, parameter, retain_graph=True)[0]
    assert torch.isfinite(derivative).all()
    zero = parameter.new_zeros(())
    ledger = LossLedger(
        value, {"action": value, "representation": zero, "execution": zero}, {"test": value}, {}
    )
    with (
        patch.object(engine, "_forward", return_value=(ledger, {})),
        patch.object(engine.optimizer, "step") as step,
    ):
        with pytest.raises(FloatingPointError, match="non-finite loss before backward"):
            engine.train_step(_batch(config))
    step.assert_not_called()
    assert engine.global_step == 0 and engine.schedule.step_index == 0
    assert torch.equal(parameter, before)
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("training", [False, True])
def test_full_formal_forward_and_backward_are_diagnostic_invariant(
    dtype: torch.dtype, training: bool
) -> None:
    torch.manual_seed(8301)
    config = _config()
    model = ClearVLAMainlinePolicy(config).train(training)
    model.set_training_step(1800)
    # Open the zero-initialized W outputs so the test exercises the repaired
    # selectors all the way through P2, not only an algebraically neutral W.
    with torch.no_grad():
        model.world.dynamics.delta_head.weight.normal_(std=0.01)
        model.world.dynamics.transport_head.weight.normal_(std=0.01)
    engine = _engine(model)
    batch = _batch(config)
    names, parameters = zip(*[(n, p) for n, p in model.named_parameters() if p.requires_grad])
    records = []
    rng = torch.get_rng_state().clone()
    for diagnostic in (False, True):
        torch.set_rng_state(rng)
        with torch.autocast("cpu", dtype=dtype, enabled=dtype == torch.bfloat16):
            ledger, _ = engine._forward(
                batch,
                training=training,
                collect_diagnostics=diagnostic,
                generator=torch.Generator().manual_seed(8302),
            )
        assert torch.isfinite(ledger.total)
        torch.testing.assert_close(
            ledger.total, sum(ledger.contributions.values()), rtol=1e-6, atol=1e-7
        )
        gradients = torch.autograd.grad(ledger.total, parameters, allow_unused=True)
        records.append((ledger.total.detach(), gradients, torch.get_rng_state().clone()))
    assert torch.equal(records[0][0], records[1][0])
    assert torch.equal(records[0][2], records[1][2])
    for name, left, right in zip(names, records[0][1], records[1][1]):
        assert (left is None) == (right is None), name
        if left is not None:
            assert torch.isfinite(left).all() and torch.isfinite(right).all(), name
            assert torch.equal(left, right), name
    for prefix in (
        "observation.",
        "grounding.",
        "intent.",
        "world.",
        "p1.",
        "policy_compiler.",
        "transition.",
        "execution_bottom.",
        "conditioning.",
        "bridge.",
        "training_targets.recognizer.",
    ):
        assert any(
            name.startswith(prefix) and gradient is not None and gradient.abs().sum() > 0
            for name, gradient in zip(names, records[0][1])
        ), prefix
