from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from test_mainline_policy import _config

from clearvla.mainline.model.restored_bottom import _build_decoder_config
from clearvla.mainline.v120_core.role_delta_attnres import PolicyRoleDeltaBank
from clearvla.mainline.v120_core.time_domain_mmdit import (
    EvidenceLatentMMDiTActionDecoder,
)


def _decoder(*, dynamic_block_route: bool) -> EvidenceLatentMMDiTActionDecoder:
    config = replace(
        _build_decoder_config(_config()),
        latent_cvae_mmdit_dynamic_block_route=int(dynamic_block_route),
    )
    config.validate()
    decoder = EvidenceLatentMMDiTActionDecoder(
        config,
        hidden_event_head=False,
    ).train()
    decoder.set_execution_training_step(1200)
    return decoder


def _formal_block_vjp_after_candidate_probe(
    decoder: EvidenceLatentMMDiTActionDecoder,
    *,
    device: torch.device,
) -> tuple[list[tuple[bool, bool]], list[tuple[bool, bool]]]:
    config = decoder.config
    batch = 1
    hidden = int(config.hidden_size)
    horizon = int(config.action_horizon)
    evidence_rows = 5
    action = torch.randn(batch, horizon, hidden, device=device)
    selector = torch.randn(batch, evidence_rows, hidden, device=device)
    values = torch.randn_like(selector)
    condition = torch.randn(batch, hidden, device=device)
    key_bias = torch.zeros(evidence_rows, device=device)
    prediction = torch.zeros(
        batch,
        horizon,
        int(config.physical_action_dim),
        device=device,
        dtype=torch.bfloat16,
    )
    block_modes: list[tuple[bool, bool]] = []
    head_modes: list[tuple[bool, bool]] = []

    def capture_block(_module, _args, _kwargs):
        block_modes.append(
            (torch.is_grad_enabled(), torch.is_autocast_cache_enabled())
        )

    def capture_head(_module, _args):
        head_modes.append(
            (torch.is_grad_enabled(), torch.is_autocast_cache_enabled())
        )

    block_hook = decoder.blocks[0].register_forward_pre_hook(
        capture_block,
        with_kwargs=True,
    )
    head_hook = decoder.terminal_controller.velocity_head.register_forward_pre_hook(capture_head)
    try:
        with torch.autocast(device.type, dtype=torch.bfloat16):
            decoder._probe_native_candidates(
                action,
                prediction_reference=prediction,
                candidate_blocks=torch.zeros(
                    batch, 1, device=device, dtype=torch.long
                ),
                candidate_repeats=torch.zeros(
                    batch, 1, device=device, dtype=torch.long
                ),
                candidate_mask=torch.ones(
                    batch, 1, device=device, dtype=torch.bool
                ),
                evidence_tokens=selector,
                evidence_value_tokens=values,
                global_condition=condition,
                evidence_key_bias=key_bias,
                evidence_scale=1.0,
                capacity_ratios=None,
                identity_boundary=True,
            )
            formal_action, _ = decoder.blocks[0](
                action,
                selector,
                condition,
                evidence_value_tokens=values,
                evidence_key_bias=key_bias,
            )
            formal_velocity = decoder.terminal_controller.predict_candidate_velocity(
                decoder.terminal_controller.normalize(formal_action)
            )
        formal_velocity.float().square().mean().backward()
    finally:
        block_hook.remove()
        head_hook.remove()

    block_gradient = decoder.blocks[0].action_mod.weight.grad
    assert block_gradient is not None
    assert torch.isfinite(block_gradient).all()
    assert torch.count_nonzero(block_gradient) > 0
    head_gradients = tuple(
        parameter.grad
        for parameter in decoder.terminal_controller.velocity_head.parameters()
        if parameter.requires_grad
    )
    assert any(
        gradient is not None and bool(torch.count_nonzero(gradient))
        for gradient in head_gradients
    )
    return block_modes, head_modes


def test_candidate_probe_keeps_detached_weight_casts_out_of_amp_cache() -> None:
    decoder = _decoder(dynamic_block_route=True)
    block_modes, head_modes = _formal_block_vjp_after_candidate_probe(
        decoder,
        device=torch.device("cpu"),
    )
    assert block_modes[0] == (False, False)
    assert block_modes[-1] == (True, True)
    assert head_modes[0] == (False, False)
    assert head_modes[-1] == (True, True)


def _sequential_decoder_inputs(
    decoder: EvidenceLatentMMDiTActionDecoder,
    *,
    device: torch.device,
) -> dict[str, object]:
    config = decoder.config
    batch = 1
    hidden = int(config.hidden_size)
    horizon = int(config.action_horizon)
    basis = int(config.action_basis_tokens)
    state = torch.randn(batch, 1, hidden, device=device)
    state_history = torch.randn(batch, 2, hidden, device=device)
    executed = torch.randn(batch, 1, hidden, device=device)
    rollout = torch.randn(batch, 3, hidden, device=device)
    protected_shape = (batch, horizon, basis, hidden)
    role_bank = PolicyRoleDeltaBank(
        values=torch.randn(batch, 2, *protected_shape[1:], device=device),
        source_names=("temporal", "state_change"),
        source_depths=(7, 7),
        protected_detail=torch.randn(*protected_shape, device=device),
        protected_policy_precision=torch.randn(*protected_shape, device=device),
    )
    layer_contracts = [
        {
            "rollout_tokens": rollout,
            "state_tokens": state,
            "state_history_tokens": state_history,
        }
        for _ in range(2)
    ]
    trajectory = torch.zeros(batch, horizon, hidden, device=device)
    return {
        "noisy_physical": torch.randn(
            batch,
            horizon,
            int(config.physical_action_dim),
            device=device,
        ),
        "time": torch.full((batch,), 0.5, device=device),
        "trajectory_tokens": trajectory,
        "trajectory_workspace_tokens": trajectory,
        "policy_action_tokens": None,
        "policy_role_delta_bank": role_bank,
        "rollout_tokens": rollout,
        "transition_memory": [
            torch.randn(batch, horizon, hidden, device=device)
        ],
        "event_evidence": torch.randn(batch, horizon, 3, device=device),
        "state_memory": [state, state_history],
        "layer_contracts": layer_contracts,
        "intent_memory": {"state": state, "executed": executed},
        "collect_diagnostics": True,
        "collect_gripper_diagnostics": False,
    }


def _learned_execution_vjp(
    *,
    device: torch.device,
) -> list[list[tuple[bool, bool]]]:
    decoder = _decoder(dynamic_block_route=False).to(device)
    block_modes: list[list[tuple[bool, bool]]] = [
        [] for _ in decoder.blocks
    ]
    hooks = []
    for index, block in enumerate(decoder.blocks):

        def capture(_module, _args, _kwargs, *, block_index=index):
            block_modes[block_index].append(
                (torch.is_grad_enabled(), torch.is_autocast_cache_enabled())
            )

        hooks.append(block.register_forward_pre_hook(capture, with_kwargs=True))
    try:
        with torch.autocast(device.type, dtype=torch.bfloat16):
            output = decoder(**_sequential_decoder_inputs(decoder, device=device))
        output["pred_velocity"].float().square().mean().backward()
    finally:
        for hook in hooks:
            hook.remove()

    for block in decoder.blocks:
        gradient = block.action_mod.weight.grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0
    return block_modes


def test_learned_execution_hard_audit_uses_cache_off_before_soft_vjp() -> None:
    block_modes = _learned_execution_vjp(device=torch.device("cpu"))
    for modes in block_modes:
        assert (False, False) in modes
        assert (True, True) in modes
        assert modes.index((False, False)) < modes.index((True, True))


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA BF16 autocast cache regression requires a CUDA device",
)
def test_internal_parameter_probes_preserve_cuda_bf16_vjps() -> None:
    device = torch.device("cuda")
    decoder = _decoder(dynamic_block_route=True).to(device)
    block_modes, head_modes = _formal_block_vjp_after_candidate_probe(
        decoder,
        device=device,
    )
    assert block_modes[0] == (False, False)
    assert block_modes[-1] == (True, True)
    assert head_modes[0] == (False, False)
    assert head_modes[-1] == (True, True)

    sequential_modes = _learned_execution_vjp(device=device)
    for modes in sequential_modes:
        assert (False, False) in modes
        assert (True, True) in modes
