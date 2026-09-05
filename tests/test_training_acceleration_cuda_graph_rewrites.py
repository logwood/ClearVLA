from __future__ import annotations

import dataclasses
import math

import pytest
import torch
import torch.nn.functional as F
from test_mainline_policy import _batch as _training_batch
from test_mainline_policy import _config as _mainline_config
from test_time_domain_mmdit import _config as _decoder_config

from clearvla.mainline.training.cuda_graph import (
    StaticTrainingInputMismatch,
    _clone_static_tree,
    _copy_static_tree,
    _static_tree_signature,
)
from clearvla.mainline.v120_core.flow_dino_evidence import (
    _normalize_flow_evidence,
    _smooth_bound_flow_to_image,
)
from clearvla.mainline.v120_core.time_domain_mmdit import (
    EvidenceLatentMMDiTActionDecoder,
)


def _legacy_execution_value_score(value_field: torch.Tensor, arm_dim: int) -> torch.Tensor:
    component_weight = value_field.new_tensor([float(arm_dim), 1.0]) / float(
        arm_dim + 1
    )
    return (value_field * component_weight).sum(dim=-1).mean(dim=-1)


def _legacy_mean_field_policy(
    decoder: EvidenceLatentMMDiTActionDecoder,
    value_field: torch.Tensor,
    pointer_mass: torch.Tensor,
    terminal_logit_bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = int(value_field.shape[0])
    depth = len(decoder.blocks)
    dwell = decoder.max_dwell
    operation_candidate_count = depth * dwell
    candidate_count = operation_candidate_count + 1
    scores = _legacy_execution_value_score(value_field.float(), decoder.arm_dim)
    temperature = max(
        float(decoder.config.latent_cvae_mmdit_execution_soft_temperature), 1e-3
    )
    global_probabilities = scores.new_zeros(batch, candidate_count)
    conditional_entropy = scores.new_zeros(batch)
    skip_probability = scores.new_zeros(batch)
    terminal_probability = scores.new_zeros(batch)
    prior_weight = float(decoder.config.latent_cvae_mmdit_terminal_prior_weight)
    for pointer_index in range(depth):
        local_indices = list(
            range(pointer_index * dwell, operation_candidate_count)
        )
        local_indices.append(operation_candidate_count)
        index = torch.as_tensor(local_indices, device=scores.device, dtype=torch.long)
        local_logits = -scores.index_select(1, index) / temperature
        local_logits[:, -1] = local_logits[:, -1] + math.log(prior_weight)
        if terminal_logit_bias is not None:
            local_logits[:, -1] = (
                local_logits[:, -1] + terminal_logit_bias.float()
            )
        local_probabilities = torch.softmax(local_logits, dim=-1)
        source_mass = pointer_mass.float()[:, pointer_index]
        contribution = source_mass[:, None] * local_probabilities
        global_probabilities = global_probabilities.index_add(
            1, index, contribution
        )
        conditional_entropy = conditional_entropy + source_mass * (
            -(
                local_probabilities
                * local_probabilities.clamp_min(1e-8).log()
            ).sum(dim=-1)
        )
        local_blocks = torch.div(index[:-1], dwell, rounding_mode="floor")
        skip_mask = local_blocks > pointer_index
        if bool(skip_mask.any()):
            skip_probability = skip_probability + contribution[:, :-1][
                :, skip_mask
            ].sum(dim=-1)
        terminal_probability = terminal_probability + contribution[:, -1]

    terminal_mass = pointer_mass.float()[:, depth]
    next_pointer_mass = scores.new_zeros(batch, depth + 1)
    terminal_index = torch.as_tensor([depth], device=scores.device)
    next_pointer_mass = next_pointer_mass.index_add(
        1, terminal_index, (terminal_mass + terminal_probability)[:, None]
    )
    for block_index in range(depth):
        block_mass = global_probabilities[
            :, block_index * dwell : (block_index + 1) * dwell
        ].sum(dim=-1)
        destination = torch.as_tensor([block_index + 1], device=scores.device)
        next_pointer_mass = next_pointer_mass.index_add(
            1, destination, block_mass[:, None]
        )
    return (
        global_probabilities,
        next_pointer_mass,
        conditional_entropy,
        skip_probability,
        terminal_probability,
    )


def test_cuda_graph_static_execution_policy_rewrite_matches_legacy_values_and_vjp() -> None:
    torch.manual_seed(9300)
    decoder = EvidenceLatentMMDiTActionDecoder(_decoder_config()).train()
    depth = len(decoder.blocks)
    candidate_count = depth * decoder.max_dwell + 1
    reference_value = torch.randn(
        3,
        candidate_count,
        decoder.horizon,
        2,
        requires_grad=True,
    )
    optimized_value = reference_value.detach().clone().requires_grad_(True)
    pointer = torch.rand(3, depth + 1)
    pointer = pointer / pointer.sum(dim=-1, keepdim=True)
    terminal_bias = torch.randn(3)

    reference = _legacy_mean_field_policy(
        decoder, reference_value, pointer, terminal_bias
    )
    optimized = decoder._mean_field_execution_policy(
        optimized_value, pointer, terminal_bias
    )
    for actual, expected in zip(optimized, reference, strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)

    probes = [torch.randn_like(value) for value in reference]
    reference_loss = sum(
        (value * probe).sum()
        for value, probe in zip(reference, probes, strict=True)
    )
    optimized_loss = sum(
        (value * probe).sum()
        for value, probe in zip(optimized, probes, strict=True)
    )
    reference_gradient = torch.autograd.grad(reference_loss, reference_value)[0]
    optimized_gradient = torch.autograd.grad(optimized_loss, optimized_value)[0]
    torch.testing.assert_close(
        optimized_gradient,
        reference_gradient,
        rtol=2e-6,
        atol=2e-7,
    )


def test_cuda_graph_hard_policy_tensorization_matches_legacy_inactive_rows() -> None:
    torch.manual_seed(9301)
    decoder = EvidenceLatentMMDiTActionDecoder(_decoder_config()).train()
    blocks, _repeats = decoder._global_execution_candidate_chart(
        batch=3, device=torch.device("cpu")
    )
    candidate_count = int(blocks.shape[1])
    value = torch.randn(3, candidate_count, decoder.horizon, 2)
    candidate_mask = torch.zeros(3, candidate_count, dtype=torch.bool)
    candidate_mask[0, :2] = True
    candidate_mask[1, 2:] = True
    neutral_index = torch.tensor([0, 2, candidate_count - 1])

    safe_mask = candidate_mask.clone()
    inactive = ~safe_mask.any(dim=-1)
    if bool(inactive.any()):
        safe_mask[inactive, 0] = True
        legacy_neutral = torch.where(
            inactive, torch.zeros_like(neutral_index), neutral_index
        )
    else:
        legacy_neutral = neutral_index
    learned = decoder._soft_execution_probabilities(value, safe_mask, blocks)
    neutral = F.one_hot(legacy_neutral, num_classes=candidate_count).to(
        dtype=learned.dtype
    )
    progress = decoder.execution_progress.to(dtype=learned.dtype)
    expected_scheduled = (1.0 - progress) * neutral + progress * learned
    expected_selected = torch.where(
        inactive,
        torch.zeros_like(neutral_index),
        expected_scheduled.argmax(dim=-1),
    )

    scheduled, selected = decoder._scheduled_hard_policy(
        value, candidate_mask, neutral_index, blocks
    )
    torch.testing.assert_close(scheduled, expected_scheduled, rtol=0.0, atol=0.0)
    assert torch.equal(selected, expected_selected)


def test_cuda_graph_candidate_chart_buffer_matches_legacy_construction() -> None:
    decoder = EvidenceLatentMMDiTActionDecoder(_decoder_config())
    blocks, repeats = decoder._global_execution_candidate_chart(
        batch=2, device=torch.device("cpu")
    )
    expected_blocks = torch.arange(len(decoder.blocks)).repeat_interleave(
        decoder.max_dwell
    )
    expected_repeats = torch.arange(decoder.max_dwell).repeat(len(decoder.blocks))
    expected_blocks = torch.cat((expected_blocks, torch.tensor([len(decoder.blocks)])))
    expected_repeats = torch.cat((expected_repeats, torch.zeros(1, dtype=torch.long)))
    assert torch.equal(blocks, expected_blocks[None].expand(2, -1))
    assert torch.equal(repeats, expected_repeats[None].expand(2, -1))


def test_cuda_graph_flow_constant_rewrites_match_legacy_rectangular_chart() -> None:
    torch.manual_seed(9302)
    reference_flow = torch.randn(2, 2, 4, 7, requires_grad=True)
    optimized_flow = reference_flow.detach().clone().requires_grad_(True)

    batch, _, height, width = reference_flow.shape
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    base = torch.stack((x, y), dim=0)[None].expand(batch, -1, -1, -1)
    maximum = reference_flow.new_tensor(
        (float(width - 1), float(height - 1)), dtype=torch.float32
    )[None, :, None, None]
    positive_limit = (maximum - base).clamp_min(0.0)
    negative_limit = base.clamp_min(0.0)

    def signed_chart(value: torch.Tensor, limit: torch.Tensor) -> torch.Tensor:
        safe_limit = limit.clamp_min(1e-6)
        return limit * torch.tanh(value / safe_limit)

    bounded_reference = torch.where(
        reference_flow >= 0.0,
        signed_chart(reference_flow.float(), positive_limit),
        signed_chart(reference_flow.float(), negative_limit),
    ).to(dtype=reference_flow.dtype)
    bounded_optimized, _compression = _smooth_bound_flow_to_image(optimized_flow)
    torch.testing.assert_close(
        bounded_optimized, bounded_reference, rtol=0.0, atol=0.0
    )
    probe = torch.randn_like(bounded_reference)
    reference_gradient = torch.autograd.grad(
        (bounded_reference * probe).sum(), reference_flow
    )[0]
    optimized_gradient = torch.autograd.grad(
        (bounded_optimized * probe).sum(), optimized_flow
    )[0]
    torch.testing.assert_close(
        optimized_gradient, reference_gradient, rtol=0.0, atol=0.0
    )

    normalized = _normalize_flow_evidence(reference_flow.detach())
    scale = reference_flow.new_tensor(
        (float(width - 1), float(height - 1)), dtype=torch.float32
    )[None, :, None, None]
    torch.testing.assert_close(
        normalized,
        reference_flow.detach().float() / scale,
        rtol=0.0,
        atol=0.0,
    )


def _tensor_rows(value: object, path: str = "batch") -> dict[str, torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return {path: value}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        result: dict[str, torch.Tensor] = {}
        for field in dataclasses.fields(value):
            result.update(
                _tensor_rows(getattr(value, field.name), f"{path}.{field.name}")
            )
        return result
    return {}


def test_cuda_graph_static_batch_copy_preserves_slots_and_updates_every_value() -> None:
    config = _mainline_config()
    torch.manual_seed(9303)
    initial = _training_batch(config, batch=2)
    static = _clone_static_tree(initial)
    initial_signature = _static_tree_signature(initial)
    assert _static_tree_signature(static) == initial_signature
    static_rows = _tensor_rows(static)
    static_ids = {name: id(value) for name, value in static_rows.items()}

    torch.manual_seed(9304)
    incoming = _training_batch(config, batch=2)
    assert _static_tree_signature(incoming) == initial_signature
    _copy_static_tree(static, incoming)
    incoming_rows = _tensor_rows(incoming)
    assert tuple(static_rows) == tuple(incoming_rows)
    for name in static_rows:
        assert id(static_rows[name]) == static_ids[name]
        assert torch.equal(static_rows[name], incoming_rows[name]), name

    shortened_goal = dataclasses.replace(
        incoming,
        online=dataclasses.replace(
            incoming.online,
            goal=dataclasses.replace(
                incoming.online.goal,
                tokens=incoming.online.goal.tokens[:, :-1],
                mask=incoming.online.goal.mask[:, :-1],
            ),
        ),
    )
    with pytest.raises(StaticTrainingInputMismatch, match="goal.tokens"):
        _copy_static_tree(static, shortened_goal)
