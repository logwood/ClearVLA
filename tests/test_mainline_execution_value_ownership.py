"""M9: computation choices must use the value coordinates supervised by the outlet.

These are controlled synthetic value charts, not measured robot task performance.
"""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from test_mainline_policy import _config
from torch import Tensor

from clearvla.mainline.model.restored_bottom import _build_decoder_config
from clearvla.mainline.v120_core.time_domain_mmdit import EvidenceLatentMMDiTActionDecoder

BINARY_MODES = ("calvin_binary_command", "maniskill_binary_command")


def _decoder(mode: str) -> EvidenceLatentMMDiTActionDecoder:
    torch.manual_seed(9101)
    core = replace(_build_decoder_config(_config()), gripper_output_mode=mode)
    decoder = EvidenceLatentMMDiTActionDecoder(core).eval()
    decoder.set_execution_training_step(10000)
    return decoder


def _chart(decoder: EvidenceLatentMMDiTActionDecoder, dtype: torch.dtype):
    depth, dwell = len(decoder.blocks), decoder.max_dwell
    count = depth * dwell + 1
    field = torch.zeros(1, count, 24, 2, dtype=dtype)
    field[..., 0] = torch.arange(count)[None, :, None] * 0.05
    changed = field.clone()
    changed[:, 0, :, 1] = 4.0
    changed[:, -1, :, 1] = -4.0
    pointer = torch.zeros(1, depth + 1)
    pointer[:, 0] = 1.0
    blocks, _ = decoder._global_execution_candidate_chart(batch=1, device=field.device)
    return field, changed, pointer, blocks


def _policy(decoder, field: Tensor, pointer: Tensor, blocks: Tensor, kind: str):
    if kind == "mean_field":
        return decoder._mean_field_execution_policy(field, pointer)
    valid = torch.ones_like(blocks, dtype=torch.bool)
    if kind == "soft":
        return (decoder._soft_execution_probabilities(field, valid, blocks),)
    return decoder._scheduled_hard_policy(
        field, valid, torch.zeros(field.shape[0], dtype=torch.long), blocks
    )


@pytest.mark.parametrize("mode", BINARY_MODES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("kind", ["mean_field", "soft", "scheduled_hard"])
def test_binary_unused_value_cannot_change_policy(mode, dtype, kind):
    decoder = _decoder(mode)
    field, changed, pointer, blocks = _chart(decoder, dtype)
    before = _policy(decoder, field, pointer, blocks, kind)
    after = _policy(decoder, changed, pointer, blocks, kind)
    for left, right in zip(before, after, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize("mode", BINARY_MODES)
@pytest.mark.parametrize("kind", ["mean_field", "soft"])
def test_binary_policy_has_no_unused_value_vjp(mode, kind):
    decoder = _decoder(mode)
    field, _, pointer, blocks = _chart(decoder, torch.float32)
    field.requires_grad_()
    probability = _policy(decoder, field, pointer, blocks, kind)[0]
    weights = torch.arange(probability.shape[1])[None].float()
    gradient = torch.autograd.grad((probability * weights).sum(), field)[0]
    assert gradient[..., 0].abs().sum() > 0
    assert torch.count_nonzero(gradient[..., 1]) == 0


@pytest.mark.parametrize("kind", ["mean_field", "soft", "scheduled_hard"])
def test_continuous_gripper_value_is_still_a_live_choice(kind):
    decoder = _decoder("continuous")
    field, changed, pointer, blocks = _chart(decoder, torch.float32)
    before = _policy(decoder, field, pointer, blocks, kind)[0]
    after = _policy(decoder, changed, pointer, blocks, kind)[0]
    assert not torch.equal(before, after)


@pytest.mark.parametrize("mode", ["continuous", *BINARY_MODES])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_score_matches_supervised_component_metric(mode, dtype):
    from clearvla.mainline.execution_values import execution_value_component_weights

    decoder = _decoder(mode)
    field = torch.randn(2, 7, 24, 2, dtype=dtype, requires_grad=True)
    weight = execution_value_component_weights(field, arm_dim=6, gripper_output_mode=mode)
    actual = decoder._execution_value_score(field, arm_dim=6, gripper_output_mode=mode)
    expected = ((field * weight).sum(-1).mean(-1) if mode == "continuous"
                else field[..., 0].mean(-1))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if mode == "continuous":
        # Reference keeps the original weights-then-multiply expression order.
        inherited = (field * (field.new_tensor([6.0, 1.0]) / 7.0)).sum(-1).mean(-1)
        torch.testing.assert_close(actual, inherited, rtol=0, atol=0)


@pytest.mark.parametrize("mode", BINARY_MODES)
def test_compatibility_hard_selector_uses_the_binary_contract(mode):
    decoder = _decoder(mode)
    field, changed, _, _ = _chart(decoder, torch.float32)
    valid = torch.ones(field.shape[:2], dtype=torch.bool)
    before = decoder._select_execution_candidate(
        field, valid, arm_dim=6, gripper_output_mode=mode
    )
    after = decoder._select_execution_candidate(
        changed, valid, arm_dim=6, gripper_output_mode=mode
    )
    torch.testing.assert_close(before, after, rtol=0, atol=0)


@pytest.mark.parametrize("mode", BINARY_MODES)
def test_binary_head_row_has_no_task_policy_gradient(mode):
    decoder = _decoder(mode)
    assert decoder.execution_controller is not None
    reader = decoder.execution_controller.value_reader
    assert reader.value_head.out_features == 2
    hidden = torch.randn(1, 7, 24, reader.hidden_size)
    field = reader.value_head(hidden)
    pointer = torch.zeros(1, len(decoder.blocks) + 1)
    pointer[:, 0] = 1
    probability = decoder._mean_field_execution_policy(field, pointer)[0]
    gradient = torch.autograd.grad(
        (probability * torch.arange(7)[None]).sum(), reader.value_head.weight
    )[0]
    assert gradient[0].abs().sum() > 0
    assert torch.count_nonzero(gradient[1]) == 0


def test_source_identity_includes_the_execution_metric():
    from pathlib import Path

    from clearvla.mainline.checkpoint import active_source_snapshot

    names = dict(active_source_snapshot(Path(__file__).resolve().parents[1]).files)
    assert "clearvla/mainline/execution_values.py" in names


def test_metric_rejects_unknown_mode():
    from clearvla.mainline.execution_values import execution_value_score
    with pytest.raises(ValueError, match="gripper output mode"):
        execution_value_score(torch.zeros(1, 2, 24, 2), arm_dim=6, gripper_output_mode="guess")
