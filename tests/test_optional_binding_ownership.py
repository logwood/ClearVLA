"""Optional CALVIN ownership must not appear as a zero-filled Pen owner."""
from dataclasses import replace

import pytest
import torch

from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
from clearvla.mainline.training.engine import MainlineTrainingEngine
from clearvla.mainline.training.optimizer import (
    WarmupCosineSchedule,
    build_optimizer,
    gradient_diagnostics,
)
from tests.test_mainline_policy import _batch, _calvin_binary_config, _config


@pytest.mark.parametrize("binding", [None, "disabled", "calvin_primary_v1", "calvin_primary_v2"])
def test_optional_binding_owner_matches_constructed_parameters(binding) -> None:
    config = _config() if binding is None else _calvin_binary_config()
    if binding is not None:
        config = replace(config, top=replace(config.top, calvin_object_binding=binding))
    model = ClearVLAMainlinePolicy(config)
    optimizer, ownership = build_optimizer(model, config)
    enabled = binding in {"calvin_primary_v1", "calvin_primary_v2"}
    assert ("calvin_object_binding" in ownership.role_counts) == enabled
    diagnostics = gradient_diagnostics(model, stage="raw")
    assert ("gradient_raw_calvin_object_binding_l2" in diagnostics) == enabled
    optimized = [id(p) for group in optimizer.param_groups for p in group["params"]]
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    assert len(optimized) == len(set(optimized)) and set(optimized) == expected


def test_v2_formal_training_loss_reaches_null_scorer() -> None:
    torch.manual_seed(628)
    base = _calvin_binary_config()
    config = replace(base, top=replace(base.top, calvin_object_binding="calvin_primary_v2"))
    model = ClearVLAMainlinePolicy(config).train()
    optimizer, _ = build_optimizer(model, config)
    schedule = WarmupCosineSchedule(optimizer, warmup_steps=2, total_steps=4, minimum_ratio=0.1)
    engine = MainlineTrainingEngine(
        model=model, config=config, optimizer=optimizer, schedule=schedule,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    ledger, _ = engine._forward(
        _batch(config), training=True, collect_diagnostics=False,
        generator=torch.Generator().manual_seed(629),
    )
    assert torch.isfinite(ledger.total)
    ledger.total.backward()
    gradient = model.intent.calvin_object_binding.null_scorer.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert gradient.abs().max() > 1e-7
