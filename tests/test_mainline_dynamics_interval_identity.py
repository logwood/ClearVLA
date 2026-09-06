from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from test_mainline_checkpoint import _dataset

from clearvla.mainline.checkpoint import (
    ArtifactIdentity,
    build_checkpoint_identity,
    compare_checkpoint_identity,
)
from clearvla.mainline.config import ExperimentConfig
from clearvla.mainline.manifest import ARCHITECTURE_MANIFEST
from clearvla.mainline.model.component_contracts import ComponentSelection
from clearvla.mainline.model.dynamics import ObjectFutureDynamicsCompiler, _ObjectIntervalBlock
from clearvla.mainline.model.types import ObjectWorldBelief, PhysicalActionCondition


def test_typed_interval_position_is_qk_only_and_zero_preserving() -> None:
    torch.manual_seed(7201)
    block = _ObjectIntervalBlock(
        hidden=16,
        heads=4,
        typed_normalization_floor=0.25,
    )
    value = torch.zeros(2, 3, 2, 16, requires_grad=True)
    position = torch.randn(1, 3, 1, 16)
    output, _ = block.forward_typed(
        value,
        causal_interval=True,
        interval_position=position,
        collect_diagnostics=True,
    )
    assert torch.count_nonzero(output) == 0
    output.square().sum().backward()
    assert value.grad is not None
    assert torch.count_nonzero(value.grad) == 0


def test_typed_interval_position_changes_selector_for_nonzero_values() -> None:
    torch.manual_seed(7202)
    block = _ObjectIntervalBlock(
        hidden=16,
        heads=4,
        typed_normalization_floor=0.25,
    )
    value = torch.randn(2, 3, 2, 16)
    position = torch.randn(1, 3, 1, 16)
    without, _ = block.forward_typed(
        value,
        causal_interval=True,
        collect_diagnostics=False,
    )
    with_position, _ = block.forward_typed(
        value,
        causal_interval=True,
        interval_position=position,
        collect_diagnostics=False,
    )
    assert bool(torch.isfinite(with_position).all())
    assert float((with_position - without).detach().abs().sum()) > 0.0


def test_compiler_retains_one_existing_interval_identity_parameter() -> None:
    compiler = ObjectFutureDynamicsCompiler(
        hidden=16,
        content_dim=16,
        route_dim=8,
        action_dim=7,
        heads=4,
    )
    assert tuple(compiler.interval_identity.shape) == (1, 4, 1, 16)
    assert (
        sum(
            parameter.numel()
            for name, parameter in compiler.named_parameters()
            if name == "interval_identity"
        )
        == 64
    )


def _belief() -> ObjectWorldBelief:
    return ObjectWorldBelief(
        content=torch.randn(2, 3, 16),
        semantic=torch.randn(2, 3, 8),
        appearance=torch.randn(2, 3, 8),
        geometry=torch.randn(2, 3, 8),
        camera_coordinates=torch.randn(2, 3, 2, 2).tanh(),
        camera_transport_prior=torch.randn(2, 3, 2, 2) * 0.1,
        camera_support=torch.ones(2, 3, 2, 1),
        camera_validity=torch.ones(2, 3, 2, 1),
        log_camera_validity=torch.zeros(2, 3, 2, 1),
        validity=torch.ones(2, 3, 1),
        log_validity=torch.zeros(2, 3, 1),
    )


def _compiler() -> ObjectFutureDynamicsCompiler:
    result = ObjectFutureDynamicsCompiler(
        hidden=16,
        content_dim=16,
        route_dim=8,
        action_dim=7,
        heads=4,
    )
    # The formal output heads start at zero. Open them for a non-vacuous
    # reverse-path test, without introducing a different test-only graph.
    with torch.no_grad():
        result.delta_head.weight.normal_(std=0.1)
        result.transport_head.weight.normal_(std=0.1)
    return result


def test_self_attention_position_changes_qk_but_preserves_v_and_causality() -> None:
    torch.manual_seed(7203)
    block = _ObjectIntervalBlock(hidden=16, heads=4, typed_normalization_floor=0.25)
    values = torch.randn(2, 4, 3, 16)
    position = torch.randn(1, 4, 1, 16) * 0.02
    records = []
    handle = block.interval_attention.register_forward_pre_hook(
        lambda _module, args: records.append(tuple(value.detach().clone() for value in args[:3]))
    )
    try:
        without, _ = block(values, typed=True, causal_interval=True)
        positioned, _ = block(values, typed=True, causal_interval=True, interval_position=position)
        zeros, _ = block(
            values, typed=True, causal_interval=True, interval_position=torch.zeros_like(position)
        )
    finally:
        handle.remove()
    torch.testing.assert_close(records[1][0] - records[0][0], position[:, :, 0].expand(6, -1, -1))
    torch.testing.assert_close(records[1][1] - records[0][1], position[:, :, 0].expand(6, -1, -1))
    assert torch.equal(records[1][2], records[0][2])
    assert torch.equal(without, zeros)
    changed = values.clone()
    changed[:, -1] += 5.0 * torch.randn_like(changed[:, -1])
    perturbed, _ = block(changed, typed=True, causal_interval=True, interval_position=position)
    torch.testing.assert_close(positioned[:, :-1], perturbed[:, :-1], atol=0.0, rtol=0.0)


def test_compiler_positions_match_common_near_far_owners_and_leave_values_alone() -> None:
    torch.manual_seed(7204)
    compiler = _compiler()
    facts = _belief()
    action = PhysicalActionCondition.from_interval_action(torch.randn(2, 4, 7), torch.zeros(2, 7))
    block_records, cross_records = [], []

    def remember_block(_module, _args, kwargs):
        if kwargs.get("typed"):
            block_records.append(kwargs.get("interval_position"))

    handles = [
        compiler.w1.register_forward_pre_hook(remember_block, with_kwargs=True),
        compiler.w2.register_forward_pre_hook(remember_block, with_kwargs=True),
        compiler.w1_to_w2.register_forward_pre_hook(
            lambda _m, args: cross_records.append(tuple(v.detach().clone() for v in args[:3]))
        ),
    ]
    try:
        _, working, _ = compiler.forward_w1(facts=facts, action=action)
        compiler.forward_w2(facts=facts, w1_state=working)
    finally:
        for handle in handles:
            handle.remove()
    assert len(block_records) == 3
    assert block_records[0] is None
    assert torch.equal(block_records[1], compiler.interval_identity[:, :2])
    assert torch.equal(block_records[2], compiler.interval_identity[:, 2:4])
    assert len(cross_records) == 2  # existing generic and typed cross-attention only
    query, key, value = cross_records[-1]
    typed_query = working.far_interval_innovation.permute(0, 2, 3, 1, 4).reshape(18, 2, 16)
    memory = torch.cat((working.common_typed[:, None], working.near_interval_innovation), dim=1)
    memory = memory.permute(0, 2, 3, 1, 4).reshape(18, 3, 16)
    normalized_memory = compiler.typed_w1_memory_norm(memory)
    expected_position = torch.cat(
        (torch.zeros(1, 1, 16), compiler.interval_identity[:, :2, 0]), dim=1
    )
    torch.testing.assert_close(
        query, compiler.typed_w2_query_norm(typed_query) + compiler.interval_identity[:, 2:, 0]
    )
    torch.testing.assert_close(key, normalized_memory + expected_position)
    assert torch.equal(value, normalized_memory)
    assert torch.equal(key[:, 0], value[:, 0])  # common is not falsely tagged interval 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_complete_w_zero_values_stay_zero_and_position_owner_has_finite_vjp(dtype) -> None:
    torch.manual_seed(7205)
    compiler = _compiler()
    facts = _belief()
    action = PhysicalActionCondition.from_interval_action(torch.randn(2, 4, 7), torch.zeros(2, 7))
    zero = replace(
        facts,
        **{k: torch.zeros_like(getattr(facts, k)) for k in ("semantic", "appearance", "geometry")},
    )
    with torch.autocast("cpu", dtype=dtype, enabled=dtype == torch.bfloat16):
        _, working, _ = compiler.forward_w1(facts=zero, action=action)
        field, _ = compiler.forward_w2(facts=zero, w1_state=working)
        assert torch.count_nonzero(field.semantic_delta) == 0
        assert torch.count_nonzero(field.transport_mean) == 0
        _, working, _ = compiler.forward_w1(facts=facts, action=action)
        # Detach the generic W context to prove the new selector edge itself
        # reaches interval_identity, rather than borrowing its old base path.
        detached = replace(
            working,
            **{name: getattr(working, name).detach() for name in working.__dataclass_fields__},
        )
        field, _ = compiler.forward_w2(facts=facts, w1_state=detached)
        loss = field.semantic_delta.square().mean() + field.transport_mean.square().mean()
    (grad,) = torch.autograd.grad(loss, compiler.interval_identity)
    assert bool(torch.isfinite(grad).all())
    assert grad.abs().sum() > 0
    assert field.semantic_delta.dtype == dtype


def test_position_shape_is_exact_and_generic_branch_rejects_typed_metadata() -> None:
    block = _ObjectIntervalBlock(hidden=16, heads=4, typed_normalization_floor=0.25)
    with pytest.raises(ValueError, match="typed W interval position"):
        block(
            torch.ones(2, 3, 1, 16),
            typed=True,
            causal_interval=True,
            interval_position=torch.ones(2, 3, 1, 16),
        )
    with pytest.raises(ValueError, match="typed W path"):
        block(
            torch.ones(2, 3, 1, 16), causal_interval=True, interval_position=torch.ones(1, 3, 1, 16)
        )


def test_temporal_behavior_change_cannot_alias_recovery_checkpoint_identity() -> None:
    config = ExperimentConfig()
    current = build_checkpoint_identity(
        config,
        repo_root=Path(__file__).resolve().parents[1],
        dataset=_dataset(),
        language=ArtifactIdentity("test", "test", 0, "0" * 64),
        commit="1" * 40,
    )
    prior_manifest = replace(
        ARCHITECTURE_MANIFEST,
        components=replace(
            ARCHITECTURE_MANIFEST.components,
            top=ARCHITECTURE_MANIFEST.components.top.removesuffix("_typed_interval_qk_v1"),
        ),
    )
    prior = replace(
        current, manifest=prior_manifest.as_dict(), manifest_digest=prior_manifest.digest()
    )
    compatibility = compare_checkpoint_identity(prior, current)
    assert not compatibility.exact_resume
    assert "manifest identity differs" in compatibility.reasons
    assert prior_manifest.components.bottom == ARCHITECTURE_MANIFEST.components.bottom
    assert (
        ComponentSelection.from_config(config).world == "object_candidate_w12_typed_interval_qk_v1"
    )
