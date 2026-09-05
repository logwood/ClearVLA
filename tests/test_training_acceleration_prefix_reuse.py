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
    optimized.execution_bottom.decoder._batched_candidate_prefix = True

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


def test_prepared_controller_context_stays_within_fp_tolerance() -> None:
    config = _config()
    torch.manual_seed(9118)
    reference = ClearVLAMainlinePolicy(config).train()
    initial = copy.deepcopy(reference.state_dict())
    optimized = ClearVLAMainlinePolicy(config).train()
    optimized.load_state_dict(initial)
    optimized.execution_bottom.decoder._reuse_prepared_block_contexts = True
    optimized.execution_bottom.decoder._reuse_prepared_controller_context = True
    optimized.execution_bottom.decoder._batched_candidate_prefix = True

    reference_engine = _engine(reference, config)
    optimized_engine = _engine(optimized, config)
    batch = _batch(config, batch=2)
    torch.manual_seed(9122)
    reference_result = reference_engine.train_step(batch, collect_diagnostics=False)
    torch.manual_seed(9122)
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
            rtol=2e-5,
            atol=2e-7,
            msg=reference_name,
        )
        if reference_parameter.grad is None or optimized_parameter.grad is None:
            assert reference_parameter.grad is optimized_parameter.grad
        else:
            torch.testing.assert_close(
                optimized_parameter.grad,
                reference_parameter.grad,
                rtol=2e-5,
                atol=2e-7,
                msg=f"gradient: {reference_name}",
            )


def test_batched_candidate_prefix_preserves_nonidentity_training_update() -> None:
    """Owner batching must remain equivalent after contraction warm-up."""

    config = _config()
    torch.manual_seed(9124)
    reference = ClearVLAMainlinePolicy(config).train()
    initial = copy.deepcopy(reference.state_dict())
    optimized = ClearVLAMainlinePolicy(config).train()
    optimized.load_state_dict(initial)
    for model in (reference, optimized):
        model.execution_bottom.decoder._training_candidate_prefix_reuse = True
        model.execution_bottom.decoder._reuse_prepared_block_contexts = True
    optimized.execution_bottom.decoder._batched_candidate_prefix = True

    reference_engine = _engine(reference, config)
    optimized_engine = _engine(optimized, config)
    # The recovered execution schedule opens after step 200.  Step 700 probes
    # the real contraction bank at progress=0.5 rather than its identity
    # warm-up shortcut.
    reference_engine.global_step = 700
    optimized_engine.global_step = 700
    batch = _batch(config, batch=2)
    torch.manual_seed(9128)
    reference_result = reference_engine.train_step(batch, collect_diagnostics=False)
    torch.manual_seed(9128)
    optimized_result = optimized_engine.train_step(batch, collect_diagnostics=False)

    torch.testing.assert_close(
        optimized_result.loss,
        reference_result.loss,
        rtol=2e-5,
        atol=2e-7,
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
            rtol=2e-5,
            atol=2e-7,
            msg=reference_name,
        )
        if reference_parameter.grad is None or optimized_parameter.grad is None:
            assert reference_parameter.grad is optimized_parameter.grad
        else:
            torch.testing.assert_close(
                optimized_parameter.grad,
                reference_parameter.grad,
                rtol=2e-5,
                atol=2e-7,
                msg=f"gradient: {reference_name}",
            )


def test_prepared_controller_source_lanes_preserve_forward_values() -> None:
    config = _config()
    torch.manual_seed(9120)
    controller = ClearVLAMainlinePolicy(config).execution_bottom.decoder.execution_controller
    assert controller is not None
    batch = 2
    evidence = torch.randn(batch, 7, config.dimensions.hidden_size)
    evidence_value = torch.randn_like(evidence)
    global_condition = torch.randn(batch, config.dimensions.hidden_size)
    time_context = torch.randn_like(global_condition)
    action = torch.randn(batch, config.dimensions.action_horizon, config.dimensions.hidden_size)
    feedback = torch.randn_like(action)
    reference = controller._source_lanes(
        global_condition=global_condition,
        time_context=time_context,
        evidence_tokens=evidence,
        evidence_value_tokens=evidence_value,
        action_tokens=action,
        feedback=feedback,
    )
    prepared = controller.prepare_static_source_context(
        global_condition=global_condition,
        time_context=time_context,
        evidence_tokens=evidence,
        evidence_value_tokens=evidence_value,
    )
    optimized = controller._source_lanes(
        global_condition=global_condition,
        time_context=time_context,
        evidence_tokens=evidence,
        evidence_value_tokens=evidence_value,
        action_tokens=action,
        feedback=feedback,
        prepared_static=prepared,
    )
    for left, right in zip(reference, optimized, strict=True):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)


def test_experimental_vmap_block_bank_matches_individual_blocks() -> None:
    """Probe whether an owner-batched block bank can preserve block values."""

    from torch.func import functional_call, vmap

    from clearvla.mainline.v120_core.time_domain_mmdit import (
        _PreparedMMDiTBlockContext,
    )

    config = _config()
    torch.manual_seed(9130)
    model = ClearVLAMainlinePolicy(config).eval()
    blocks = tuple(model.execution_bottom.decoder.blocks)
    parameter_names = tuple(name for name, _ in blocks[0].named_parameters())
    buffer_names = tuple(name for name, _ in blocks[0].named_buffers())
    parameter_maps = tuple(dict(block.named_parameters()) for block in blocks)
    buffer_maps = tuple(dict(block.named_buffers()) for block in blocks)
    params = {
        name: torch.stack(tuple(values[name] for values in parameter_maps), dim=0)
        for name in parameter_names
    }
    buffers = {
        name: torch.stack(tuple(values[name] for values in buffer_maps), dim=0)
        for name in buffer_names
    }
    template = blocks[0]
    batch = 2
    action = torch.randn(len(blocks), batch, config.dimensions.action_horizon, config.dimensions.hidden_size)
    evidence = torch.randn(len(blocks), batch, 7, config.dimensions.hidden_size)
    evidence_value = torch.randn_like(evidence)
    condition = torch.randn(len(blocks), batch, config.dimensions.hidden_size)
    scale = torch.ones(len(blocks), batch)
    contexts = tuple(
        block.prepare_static_context(
            condition[index], evidence[index], evidence_value[index], action_length=action.shape[2]
        )
        for index, block in enumerate(blocks)
    )
    context_global = torch.stack([item.global_modulation for item in contexts])
    context_key = torch.stack([item.evidence_key for item in contexts])
    context_value = torch.stack([item.evidence_value for item in contexts])

    def apply_one(
        parameter,
        buffer,
        action_row,
        evidence_row,
        condition_row,
        value_row,
        scale_row,
        context_global_row,
        context_key_row,
        context_value_row,
    ):
        context_row = _PreparedMMDiTBlockContext(
            global_modulation=context_global_row,
            evidence_key=context_key_row,
            evidence_value=context_value_row,
            self_mask=contexts[0].self_mask,
        )
        return functional_call(
            template,
            (parameter, buffer),
            (action_row, evidence_row, condition_row),
            {
                "evidence_value_tokens": value_row,
                "evidence_scale": scale_row,
                "collect_diagnostics": False,
                "prepared_static": context_row,
            },
        )[0]

    batched = vmap(apply_one)(
        params,
        buffers,
        action,
        evidence,
        condition,
        evidence_value,
        scale,
        context_global,
        context_key,
        context_value,
    )
    batched.square().sum().backward()
    assert all(parameter.grad is not None for block in blocks for parameter in block.parameters())
    for index, block in enumerate(blocks):
        expected = block(
            action[index],
            evidence[index],
            condition[index],
            evidence_value_tokens=evidence_value[index],
            evidence_scale=scale[index],
            collect_diagnostics=False,
            prepared_static=contexts[index],
        )[0]
        torch.testing.assert_close(batched[index], expected, rtol=0.0, atol=0.0)


def test_experimental_vmap_contraction_bank_preserves_nonidentity_values() -> None:
    from torch.func import functional_call, vmap

    from clearvla.mainline.v120_core.refinement import NestedLowRankContractionBank

    torch.manual_seed(9134)
    banks = tuple(
        NestedLowRankContractionBank(
            hidden_size=8,
            condition_size=8,
            stage_count=1,
            rank=4,
            group_count=2,
            depth_logit_init=2.0,
        )
        for _ in range(2)
    )
    template = banks[0]
    parameter_maps = tuple(dict(bank.named_parameters()) for bank in banks)
    parameters = {
        name: torch.stack(tuple(values[name] for values in parameter_maps), dim=0)
        for name, _ in template.named_parameters()
    }
    factors = torch.stack(tuple(bank.prepare_factors() for bank in banks), dim=0)
    base_update = torch.randn(2, 3, 8)
    condition = torch.randn(2, 8)
    capacity = torch.full((2,), 0.5)

    def apply_one(parameter, base, cond, ratio, factor):
        return functional_call(
            template,
            (parameter, {}),
            (base, cond, torch.zeros(base.shape[0], dtype=torch.long)),
            {
                "contraction_progress": torch.tensor(0.5),
                "prepared_factors": factor,
                "depth_ratio_override": ratio,
                "identity_bypass": False,
                "collect_diagnostics": False,
            },
        )[0]

    batched = vmap(apply_one)(
        parameters,
        base_update.unsqueeze(0).expand(2, -1, -1, -1),
        condition.unsqueeze(0).expand(2, -1, -1),
        capacity.unsqueeze(0).expand(2, -1),
        factors,
    )
    for index, bank in enumerate(banks):
        expected = bank(
            base_update,
            condition,
            torch.zeros(2, dtype=torch.long),
            contraction_progress=torch.tensor(0.5),
            prepared_factors=bank.prepare_factors(),
            depth_ratio_override=capacity,
            identity_bypass=False,
            collect_diagnostics=False,
        )[0]
        torch.testing.assert_close(batched[index], expected, rtol=0.0, atol=0.0)
