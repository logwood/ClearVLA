from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from test_mainline_end_to_end_audit import _engine
from test_mainline_policy import _batch, _config
from torch.utils.checkpoint import checkpoint

from clearvla.mainline.config import load_config
from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
from clearvla.mainline.runtime.sampling import deployment_cache, sample_refined_cached_action


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_activation_recomputation_preserves_candidate_forward_and_parameter_vjp(dtype) -> None:
    torch.manual_seed(8401)
    config = _config()
    model = ClearVLAMainlinePolicy(config).train()
    model.set_training_step(1800)
    engine = _engine(model)
    batch = _batch(config)
    names, parameters = zip(*[(name, p) for name, p in model.named_parameters() if p.requires_grad])
    records = []
    rng = torch.get_rng_state().clone()
    reader = model.p1.factual_reader
    for enabled in (False, True):
        # Enter the production recomputation branch without scaling the test
        # model to 512-wide/B8. The function and autograd path are unchanged.
        for module in model.modules():
            if hasattr(module, "activation_checkpoint"):
                module.activation_checkpoint = enabled
        reader.raw_activation_checkpoint = enabled
        reader.checkpoint_min_batch = 1
        torch.set_rng_state(rng)
        with patch("clearvla.mainline.model.v120_p1.checkpoint", wraps=checkpoint) as p1_checkpoint:
            with torch.autocast("cpu", dtype=dtype, enabled=dtype == torch.bfloat16):
                ledger, metrics = engine._forward(
                    batch,
                    training=True,
                    collect_diagnostics=True,
                    generator=torch.Generator().manual_seed(8402),
                )
            gradients = torch.autograd.grad(ledger.total, parameters, allow_unused=True)
        assert (p1_checkpoint.call_count > 0) == enabled
        records.append((ledger.total.detach(), gradients, torch.get_rng_state().clone()))
    assert torch.equal(records[0][0], records[1][0])
    assert torch.equal(records[0][2], records[1][2])
    for name, left, right in zip(names, records[0][1], records[1][1]):
        assert (left is None) == (right is None), name
        if left is not None:
            assert torch.isfinite(left).all() and torch.isfinite(right).all(), name
            torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5, msg=name)


@pytest.mark.parametrize("dtype_name", ["fp32", "bf16"])
def test_q5_complete_lifecycle_keeps_fastpath_and_diagnostics_equivalent(dtype_name: str) -> None:
    torch.manual_seed(8403)
    preset = load_config("configs/mainline/object_intent_dynamics_323_pen_w_interval_q5.json")
    base = _config()
    config = replace(
        base,
        runtime=replace(
            base.runtime,
            compute_dtype=dtype_name,
            deployment_flow_schedule=preset.runtime.deployment_flow_schedule,
        ),
    )
    model = ClearVLAMainlinePolicy(config).eval()
    model.set_training_step(1800)
    with torch.no_grad():
        model.world.dynamics.delta_head.weight.normal_(std=0.01)
        model.world.dynamics.transport_head.weight.normal_(std=0.01)
    batch = _batch(config)
    cache, _ = deployment_cache(model, batch.online, config)
    noise = model.outlet_adapter.sample_noise(
        1,
        device=torch.device("cpu"),
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(8404),
    )
    reference = None
    for fastpath, diagnostic in ((False, False), (False, True), (True, True)):
        with (
            patch.object(model, "velocity", wraps=model.velocity) as velocity,
            patch.object(model.world, "materialize", wraps=model.world.materialize) as world,
        ):
            result = sample_refined_cached_action(
                model,
                cache,
                config,
                initial_physical_noise=noise,
                deployment_fastpath=fastpath,
                collect_diagnostics=diagnostic,
            )
        assert velocity.call_count == 12 and world.call_count == 1
        assert torch.equal(result.initial_physical_noise, noise)
        assert torch.isfinite(result.action).all()
        assert result.flow_schedule_identity["candidate_id"] == "Q5/Q5"
        if reference is None:
            reference = result
        else:
            for name in ("action", "physical_field", "motion_logits"):
                assert torch.equal(getattr(reference, name), getattr(result, name)), name
