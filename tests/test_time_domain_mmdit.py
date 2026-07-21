import inspect
import types

import torch

from clearvla.policy.config import V39PolicyConfig
from clearvla.policy.controller import EvidenceExecutionController, UnifiedHierarchicalController
from clearvla.policy.time_domain_mmdit import (
    EvidenceLatentMMDiTActionDecoder,
    TimeDomainMMDiTBlock,
)


def _config() -> V39PolicyConfig:
    config = V39PolicyConfig(
        final_action_decoder="evidence_latent_mmdit_action",
        layer_contract_adapters=1,
        hidden_size=64,
        num_heads=8,
        depth=3,
        action_horizon=4,
        first_execution_steps=1,
        mid_execution_steps=2,
        latent_action_near_steps=1,
        latent_action_mid_steps=2,
        future_anchors=2,
        future_grid_size=1,
        num_cameras=1,
        gripper_field_dim=4,
        latent_cvae_noisy_gate=1,
        latent_cvae_noisy_gate_min=0.08,
        latent_cvae_noisy_gate_power=1.0,
        latent_cvae_mmdit_noisy_logit_gate=1,
        latent_cvae_mmdit_residual_scale_max=0.25,
        latent_cvae_mmdit_source_route_delta_max=1.0,
    )
    config.validate()
    return config


def _inputs(config: V39PolicyConfig) -> dict[str, object]:
    batch = 2
    hidden = int(config.hidden_size)
    layers = [
        {
            "rollout_tokens": torch.randn(batch, config.future_token_count, hidden),
            "state_tokens": torch.randn(batch, 1, hidden),
            "state_history_tokens": torch.randn(batch, 2, hidden),
        }
        for _ in range(config.depth)
    ]
    return {
        "noisy_physical": torch.randn(batch, config.action_horizon, config.physical_action_dim),
        "time": torch.tensor([0.10, 0.90]),
        "trajectory_tokens": torch.randn(batch, config.action_horizon, hidden),
        "rollout_tokens": torch.randn(batch, config.future_token_count, hidden),
        "transition_memory": [torch.randn(batch, config.action_horizon, hidden)],
        "event_evidence": torch.randn(batch, config.action_horizon, 3),
        "state_memory": [torch.randn(batch, 1, hidden), torch.randn(batch, 2, hidden)],
        "layer_contracts": layers,
    }


def test_time_domain_decoder_has_one_read_only_evidence_path_and_three_blocks():
    config = _config()
    decoder = EvidenceLatentMMDiTActionDecoder(config)
    with torch.no_grad():
        for block in decoder.blocks:
            block.action_mod.bias[2 * config.hidden_size : 3 * config.hidden_size].fill_(1.0)
    assert len(decoder.blocks) == 3
    signature = inspect.signature(decoder.forward)
    assert "target_physical" not in signature.parameters
    output = decoder(**_inputs(config))
    assert output["pred_velocity"].shape == (
        2,
        config.action_horizon,
        config.physical_action_dim,
    )
    assert output["event_logits"].shape == (2, config.action_horizon, 3)
    assert "cvae_kl" not in output
    assert output["evidence_semantic_seed_norm"] > 0
    assert output["evidence_action_state_token_norm"] > 0
    assert output["evidence_mmd_it_action_state_scale"] == 1

    loss = output["pred_velocity"].square().mean() + output["event_logits"].square().mean()
    loss.backward()
    for block in decoder.blocks:
        assert block.action_mod.weight.grad is not None
    assert decoder.noisy_lift.mix[1].weight.grad is not None


def test_time_domain_mmdit_does_not_write_condition_tokens():
    config = _config()
    block = TimeDomainMMDiTBlock(config)
    assert not hasattr(block, "source_router")
    action = torch.randn(2, 4, config.hidden_size)
    evidence = torch.randn(2, 7, config.hidden_size)
    evidence_before = evidence.clone()
    block(
        action,
        evidence,
        torch.randn(2, config.hidden_size),
        evidence_key_bias=torch.zeros(7),
    )
    assert torch.equal(evidence, evidence_before)


def test_flow_state_is_ingested_by_the_action_stream():
    config = _config()
    decoder = EvidenceLatentMMDiTActionDecoder(config).eval()
    inputs = _inputs(config)
    with torch.no_grad():
        output = decoder(**inputs)
        changed = dict(inputs)
        changed["noisy_physical"] = inputs["noisy_physical"] + 0.5
        changed_output = decoder(**changed)
    assert not torch.allclose(output["pred_velocity"], changed_output["pred_velocity"])


def test_source_ablation_removes_only_the_selected_write_direction():
    config = _config()
    block = TimeDomainMMDiTBlock(config).eval()
    h = int(config.hidden_size)
    with torch.no_grad():
        block.action_mod.bias[2 * h : 3 * h].fill_(1.0)
    action = torch.randn(2, 4, h)
    evidence = torch.randn(2, 7, h)
    kwargs = {
        "evidence_key_bias": torch.zeros(7),
    }
    global_condition = torch.randn(2, h)
    full, full_metrics = block(action, evidence, global_condition, **kwargs)
    without_evidence, ablated_metrics = block(
        action,
        evidence,
        global_condition,
        evidence_scale=0.0,
        **kwargs,
    )
    assert not torch.allclose(full, without_evidence)
    assert full_metrics["evidence_update_norm"] > 0
    assert ablated_metrics["evidence_update_norm"] == 0


def test_native_execution_controls_have_an_exact_host_warmup_boundary():
    base_config = _config()
    active_config = V39PolicyConfig(
        **{
            **base_config.__dict__,
            "latent_cvae_mmdit_operator_capacity": 1,
            "latent_cvae_mmdit_operator_rank": 32,
            "latent_cvae_mmdit_operator_groups": 4,
            "latent_cvae_mmdit_execution_controller": 1,
            "latent_cvae_mmdit_dynamic_block_route": 1,
            "latent_cvae_mmdit_dwell_mode": "learned",
            "latent_cvae_mmdit_max_dwell": 2,
        }
    )
    torch.manual_seed(113)
    base = EvidenceLatentMMDiTActionDecoder(base_config)
    torch.manual_seed(113)
    active = EvidenceLatentMMDiTActionDecoder(active_config)
    inputs = _inputs(base_config)
    active.set_execution_training_step(0)
    with torch.no_grad():
        base_output = base(**inputs)
        active_output = active(**inputs)
    assert torch.equal(base_output["pred_velocity"], active_output["pred_velocity"])
    assert torch.equal(base_output["event_logits"], active_output["event_logits"])
    assert active_output["evidence_mmd_it_capacity_ratio"] == 1
    assert active_output["evidence_mmd_it_dwell_expected"] == 1
    assert active_output["evidence_mmd_it_controller_execution_candidate_count"] == 2
    warmup_mask = active_output["evidence_mmd_it_execution_candidate_value_mask"]
    assert warmup_mask[:, 0].all()
    assert warmup_mask[:, :, -1].all()
    assert warmup_mask[:, 1:, :2].logical_not().all()


def test_native_controller_value_reader_receives_task_gradient_through_soft_execution():
    config = V39PolicyConfig(
        **{
            **_config().__dict__,
            "latent_cvae_mmdit_operator_capacity": 1,
            "latent_cvae_mmdit_operator_rank": 32,
            "latent_cvae_mmdit_operator_groups": 4,
            "latent_cvae_mmdit_execution_controller": 1,
            "latent_cvae_mmdit_dwell_mode": "learned",
            "latent_cvae_mmdit_max_dwell": 2,
        }
    )
    decoder = EvidenceLatentMMDiTActionDecoder(config)
    decoder.set_execution_training_step(300)
    # Make the two legal dwell candidates produce distinct native actions so
    # the test exercises the route-to-task derivative rather than a degenerate
    # equal-candidate tie at the zero-initialized host gates.
    with torch.no_grad():
        hidden = int(config.hidden_size)
        decoder.blocks[0].action_mod.bias[2 * hidden : 3 * hidden].fill_(0.25)
        decoder.execution_controller.value_reader.value_head.weight.normal_(0.0, 1e-3)
    output = decoder(**_inputs(config))
    # Compute usage is an audit metric in the native path.  It must not become
    # a detached constant added to the main loss or a second amplitude owner.
    assert not output["evidence_mmd_it_execution_cost"].requires_grad
    loss = output["pred_velocity"].square().mean()
    loss.backward(retain_graph=True)
    assert decoder.execution_controller is not None
    assert decoder.execution_controller.capacity_head.weight.grad is not None
    # Learned training uses a soft candidate action chart, so the task loss can
    # reach the value reader through the actual execution decision.
    value_reader = decoder.execution_controller.value_reader
    assert any(parameter.grad is not None for parameter in value_reader.parameters())
    decoder.zero_grad(set_to_none=True)
    value_field = output["evidence_mmd_it_execution_candidate_value_field"]
    target = torch.randn_like(value_field)
    target = target - target.mean(dim=2, keepdim=True)
    value_loss = (value_field - target).square().mean()
    value_loss.backward()
    assert value_reader.value_head.weight.grad is not None
    # The value reader is a selector-plane readout. Its supervision must reach
    # the recurrent controller state, otherwise dwell learns as an isolated
    # head and operation selection is not actually unified with the controller.
    assert decoder.execution_controller.control_tokens.grad is not None
    # Inputs stay attached by design: later value decisions may send a natural
    # multi-step gradient through earlier capacity-controlled action states.
    assert decoder.execution_controller.capacity_head.weight.grad is not None
    torch.testing.assert_close(
        value_field.mean(dim=2),
        torch.zeros_like(value_field.mean(dim=2)),
        atol=1e-6,
        rtol=0.0,
    )
    assert output["evidence_mmd_it_prefix_pred_velocity"].requires_grad
    # Candidate action predictions remain detached teacher metrics; the attached
    # candidate action chart used for the task loss is internal to the decoder.
    assert not output["evidence_mmd_it_dwell_candidate_pred_velocity"].requires_grad
    assert output["evidence_mmd_it_execution_candidate_value_field"].shape == (
        2,
        config.depth,
        config.latent_cvae_mmdit_max_dwell,
        config.action_horizon,
        2,
    )
    assert output["evidence_mmd_it_controller_slot_pair_cosine"] < 0.999
    assert output["evidence_mmd_it_controller_slot_ownership_profile_diversity"] > 1e-4
    assert output["evidence_mmd_it_nonexpansive_violation"] == 0


def test_native_controller_value_reader_mask_follows_query_dtype():
    config = V39PolicyConfig(
        **{
            **_config().__dict__,
            "latent_cvae_mmdit_operator_capacity": 1,
            "latent_cvae_mmdit_operator_rank": 32,
            "latent_cvae_mmdit_operator_groups": 4,
            "latent_cvae_mmdit_execution_controller": 1,
            "latent_cvae_mmdit_dwell_mode": "learned",
            "latent_cvae_mmdit_max_dwell": 2,
        }
    )
    decoder = EvidenceLatentMMDiTActionDecoder(config).to(dtype=torch.bfloat16).eval()
    decoder.set_execution_training_step(300)
    inputs = _inputs(config)
    inputs = {
        key: value.to(dtype=torch.bfloat16)
        if torch.is_tensor(value) and value.is_floating_point()
        else value
        for key, value in inputs.items()
    }
    with torch.no_grad():
        output = decoder(**inputs)
    assert torch.isfinite(output["pred_velocity"]).all()


def test_native_candidate_probe_reuses_autocast_velocity_dtype():
    config = V39PolicyConfig(
        **{
            **_config().__dict__,
            "latent_cvae_mmdit_operator_capacity": 1,
            "latent_cvae_mmdit_operator_rank": 32,
            "latent_cvae_mmdit_operator_groups": 4,
            "latent_cvae_mmdit_execution_controller": 1,
            "latent_cvae_mmdit_dwell_mode": "learned",
            "latent_cvae_mmdit_max_dwell": 2,
        }
    )
    decoder = EvidenceLatentMMDiTActionDecoder(config).train()
    decoder.set_execution_training_step(300)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = decoder(**_inputs(config))
    candidates = output["evidence_mmd_it_dwell_candidate_pred_velocity"]
    assert candidates.dtype == output["pred_velocity"].dtype
    assert torch.isfinite(candidates).all()


def test_native_execution_progress_survives_checkpoint_reload():
    config = V39PolicyConfig(
        **{
            **_config().__dict__,
            "latent_cvae_mmdit_operator_capacity": 1,
            "latent_cvae_mmdit_operator_rank": 32,
            "latent_cvae_mmdit_operator_groups": 4,
            "latent_cvae_mmdit_execution_controller": 1,
            "latent_cvae_mmdit_dwell_mode": "learned",
            "latent_cvae_mmdit_max_dwell": 2,
        }
    )
    decoder = EvidenceLatentMMDiTActionDecoder(config)
    decoder.set_execution_training_step(300)
    state = decoder.state_dict()
    restored = EvidenceLatentMMDiTActionDecoder(config)
    restored.load_state_dict(state)
    assert restored.execution_progress == decoder.execution_progress
    assert restored._execution_progress_value == decoder._execution_progress_value


def test_native_value_reader_consumes_only_the_legacy_absolute_bias_key():
    config = V39PolicyConfig(
        **{
            **_config().__dict__,
            "latent_cvae_mmdit_operator_capacity": 1,
            "latent_cvae_mmdit_operator_rank": 32,
            "latent_cvae_mmdit_operator_groups": 4,
            "latent_cvae_mmdit_execution_controller": 1,
            "latent_cvae_mmdit_dwell_mode": "learned",
            "latent_cvae_mmdit_max_dwell": 2,
        }
    )
    decoder = EvidenceLatentMMDiTActionDecoder(config)
    state = decoder.state_dict()
    legacy_key = "execution_controller.value_reader.value_head.bias"
    terminal_key = "execution_controller.value_reader.terminal_identity"
    state.pop(terminal_key)
    state[legacy_key] = torch.randn(2)
    restored = EvidenceLatentMMDiTActionDecoder(config)
    restored.load_state_dict(state, strict=True)
    assert legacy_key not in restored.state_dict()
    assert terminal_key in restored.state_dict()


def test_native_execution_metric_snapshots_do_not_alias_progress_buffer():
    config = _config()
    decoder = EvidenceLatentMMDiTActionDecoder(config).eval()
    inputs = _inputs(config)
    decoder.set_execution_training_step(0)
    with torch.no_grad():
        warm = decoder(**inputs)
    decoder.set_execution_training_step(300)
    assert warm["evidence_mmd_it_execution_progress"] == 0
    with torch.no_grad():
        hot = decoder(**inputs)
    assert hot["evidence_mmd_it_execution_progress"] > 0


def test_unified_controller_keeps_action_stage_values_out_of_value_lane():
    config = _config()
    controller = UnifiedHierarchicalController(config, operator_branch_count=3)
    action = torch.randn(2, config.action_horizon, config.hidden_size)
    for source_index, source in (
        (3, action),
        (5, torch.randn(2, 2, config.hidden_size)),
        (6, torch.randn(2, 2, config.hidden_size)),
        (7, torch.randn(2, config.action_horizon, config.hidden_size)),
    ):
        value, key = controller._typed_source(source, source_index)
        torch.testing.assert_close(value, torch.zeros_like(value))
        assert key.abs().sum() > 0


def test_native_controller_keeps_action_feedback_out_of_value_lane():
    config = _config()
    controller = EvidenceExecutionController(config, block_count=3)
    batch = 2
    global_condition = torch.randn(batch, config.hidden_size)
    time_context = torch.randn(batch, config.hidden_size)
    evidence = torch.randn(batch, 3, config.hidden_size)
    action = torch.randn(batch, config.action_horizon, config.hidden_size)
    feedback = torch.randn_like(action)
    keys, values = controller._source_lanes(
        global_condition=global_condition,
        time_context=time_context,
        evidence_tokens=evidence,
        action_tokens=action,
        feedback=feedback,
    )
    torch.testing.assert_close(
        values[:, 2 + evidence.shape[1] :],
        torch.zeros_like(values[:, 2 + evidence.shape[1] :]),
    )
    assert keys[:, 2 + evidence.shape[1] :].abs().sum() > 0


def test_native_controller_selector_lane_keeps_upstream_sources_attached():
    config = _config()
    controller = EvidenceExecutionController(config, block_count=3)
    batch = 2
    global_condition = torch.randn(batch, config.hidden_size, requires_grad=True)
    time_context = torch.randn(batch, config.hidden_size, requires_grad=True)
    evidence = torch.randn(batch, 3, config.hidden_size, requires_grad=True)
    action = torch.randn(batch, config.action_horizon, config.hidden_size, requires_grad=True)
    feedback = torch.randn_like(action, requires_grad=True)
    keys, values = controller._source_lanes(
        global_condition=global_condition,
        time_context=time_context,
        evidence_tokens=evidence,
        action_tokens=action,
        feedback=feedback,
    )
    upstream = (keys.sum() + values.sum())
    gradients = torch.autograd.grad(
        upstream,
        (global_condition, time_context, evidence, action, feedback),
        allow_unused=True,
    )
    assert all(gradient is not None for gradient in gradients)


def test_native_controller_uses_selector_keys_and_value_evidence_separately():
    config = _config()
    controller = EvidenceExecutionController(config, block_count=3)
    batch = 2
    global_condition = torch.randn(batch, config.hidden_size)
    time_context = torch.randn(batch, config.hidden_size)
    selector_evidence = torch.randn(batch, 3, config.hidden_size)
    value_evidence = torch.randn_like(selector_evidence)
    action = torch.randn(batch, config.action_horizon, config.hidden_size)
    feedback = torch.randn_like(action)
    keys, values = controller._source_lanes(
        global_condition=global_condition,
        time_context=time_context,
        evidence_tokens=selector_evidence,
        evidence_value_tokens=value_evidence,
        action_tokens=action,
        feedback=feedback,
    )
    expected_keys = controller.key_proj(
        controller.source_norm(selector_evidence)
    )
    expected_values = controller.value_proj(
        controller.source_norm(value_evidence)
    )
    torch.testing.assert_close(
        keys[:, 2 : 2 + selector_evidence.shape[1]], expected_keys
    )
    torch.testing.assert_close(
        values[:, 2 : 2 + value_evidence.shape[1]], expected_values
    )


def test_native_layer_values_are_invariant_to_mixed_selector_content():
    config = _config()
    decoder = EvidenceLatentMMDiTActionDecoder(config).eval()
    inputs = _inputs(config)
    intent_memory = {
        "state": inputs["state_memory"][0],
        "proposal": inputs["trajectory_tokens"],
    }
    view_a = decoder.evidence_adapter(
        trajectory_tokens=inputs["trajectory_tokens"],
        rollout_tokens=inputs["rollout_tokens"],
        transition_memory=inputs["transition_memory"],
        event_evidence=inputs["event_evidence"],
        state_memory=inputs["state_memory"],
        layer_contracts=inputs["layer_contracts"],
        intent_memory=intent_memory,
    )
    changed_layers = []
    for layer in inputs["layer_contracts"]:
        changed_layers.append(
            {
                key: (value + 50.0 if isinstance(value, torch.Tensor) else value)
                for key, value in layer.items()
            }
        )
    view_b = decoder.evidence_adapter(
        trajectory_tokens=inputs["trajectory_tokens"],
        rollout_tokens=inputs["rollout_tokens"],
        transition_memory=inputs["transition_memory"],
        event_evidence=inputs["event_evidence"],
        state_memory=inputs["state_memory"],
        layer_contracts=changed_layers,
        intent_memory=intent_memory,
    )
    torch.testing.assert_close(view_a.value_tokens, view_b.value_tokens)
    organized_a = decoder.organizer(view_a, inputs["time"])
    organized_b = decoder.organizer(view_b, inputs["time"])
    torch.testing.assert_close(organized_a["latent"], organized_b["latent"])
    assert not torch.allclose(view_a.tokens, view_b.tokens)


def test_native_dynamic_route_keeps_block_and_dwell_axes_typed():
    base_config = _config()
    config = V39PolicyConfig(
        **{
            **base_config.__dict__,
            "latent_cvae_mmdit_operator_capacity": 1,
            "latent_cvae_mmdit_operator_rank": 32,
            "latent_cvae_mmdit_operator_groups": 4,
            "latent_cvae_mmdit_execution_controller": 1,
            "latent_cvae_mmdit_dynamic_block_route": 1,
            "latent_cvae_mmdit_dwell_mode": "learned",
            "latent_cvae_mmdit_max_dwell": 2,
        }
    )
    decoder = EvidenceLatentMMDiTActionDecoder(config)
    decoder.set_execution_training_step(300)
    output = decoder(**_inputs(config))
    candidate_count = config.depth * config.latent_cvae_mmdit_max_dwell + 1
    assert output["evidence_mmd_it_dwell_candidate_pred_velocity"].shape[2] == candidate_count
    assert output["evidence_mmd_it_execution_candidate_value_field"].shape[2] == candidate_count
    assert output["evidence_mmd_it_execution_candidate_value_mask"].shape[2] == candidate_count
    torch.testing.assert_close(
        output["evidence_mmd_it_execution_candidate_value_field"].mean(dim=2),
        torch.zeros_like(
            output["evidence_mmd_it_execution_candidate_value_field"].mean(dim=2)
        ),
        atol=1e-6,
        rtol=0.0,
    )
    assert output["evidence_mmd_it_dynamic_route_next_fraction"] >= 0
    assert output["evidence_mmd_it_dynamic_route_next_fraction"] <= 1
    assert not output["evidence_mmd_it_execution_cost"].requires_grad
    assert not output["evidence_mmd_it_capacity_ratio"].requires_grad
    assert output["evidence_mmd_it_execution_candidate_value_field"].requires_grad


def test_native_dynamic_eval_executes_only_the_committed_operations():
    base_config = _config()
    config = V39PolicyConfig(
        **{
            **base_config.__dict__,
            "latent_cvae_mmdit_operator_capacity": 1,
            "latent_cvae_mmdit_operator_rank": 32,
            "latent_cvae_mmdit_operator_groups": 4,
            "latent_cvae_mmdit_execution_controller": 1,
            "latent_cvae_mmdit_dynamic_block_route": 1,
            "latent_cvae_mmdit_dwell_mode": "learned",
            "latent_cvae_mmdit_max_dwell": 2,
            "latent_cvae_mmdit_execution_eval_policy": "hard",
        }
    )
    decoder = EvidenceLatentMMDiTActionDecoder(config).eval()
    decoder.set_execution_training_step(300)
    calls = [0 for _ in decoder.blocks]
    handles = []
    for block_index, block in enumerate(decoder.blocks):
        def count_call(_module, _args, *, index=block_index):
            calls[index] += 1

        handles.append(block.register_forward_pre_hook(count_call))
    try:
        with torch.no_grad():
            output = decoder(**_inputs(config))
    finally:
        for handle in handles:
            handle.remove()
    # The scheduled neutral policy follows the host sequence once: 0 -> 1 -> 2.
    assert sum(calls) == len(decoder.blocks)
    assert calls == [1 for _ in calls]
    assert output["evidence_mmd_it_committed_operation_count"] == 1
    assert output["evidence_mmd_it_candidate_probe_operation_count"] == 0
    assert output["evidence_mmd_it_candidate_probe_enabled"] == 0
    assert not output["evidence_mmd_it_execution_candidate_value_mask"].any()


def test_native_dynamic_eval_keeps_fixed_output_shapes_after_early_terminal():
    base_config = _config()
    config = V39PolicyConfig(
        **{
            **base_config.__dict__,
            "latent_cvae_mmdit_operator_capacity": 1,
            "latent_cvae_mmdit_operator_rank": 32,
            "latent_cvae_mmdit_operator_groups": 4,
            "latent_cvae_mmdit_execution_controller": 1,
            "latent_cvae_mmdit_dynamic_block_route": 1,
            "latent_cvae_mmdit_dwell_mode": "learned",
            "latent_cvae_mmdit_max_dwell": 2,
            "latent_cvae_mmdit_execution_eval_policy": "hard",
        }
    )
    decoder = EvidenceLatentMMDiTActionDecoder(config).eval()
    decoder.set_execution_training_step(1200)

    def favor_next(self, **kwargs):
        del self
        blocks = kwargs["candidate_block_index"]
        current = kwargs["block_index"]
        batch, candidates = blocks.shape
        score = torch.where(
            blocks == current[:, None] + 1,
            -torch.ones_like(blocks, dtype=torch.float32),
            torch.zeros_like(blocks, dtype=torch.float32),
        )
        field = score[:, :, None, None].expand(
            batch, candidates, config.action_horizon, 2
        ).clone()
        return field - field.mean(dim=1, keepdim=True)

    assert decoder.execution_controller is not None
    decoder.execution_controller.predict_execution_value = types.MethodType(
        favor_next, decoder.execution_controller
    )
    with torch.no_grad():
        output = decoder(**_inputs(config))
    candidate_count = config.depth * config.latent_cvae_mmdit_max_dwell + 1
    assert output["evidence_mmd_it_execution_candidate_value_field"].shape[:3] == (
        2,
        config.depth,
        candidate_count,
    )
    assert output["evidence_mmd_it_prefix_pred_velocity"].shape[1] == config.depth + 1
    assert output["evidence_mmd_it_hard_route_next_fraction"] > 0
    assert output["evidence_mmd_it_hard_dwell_expected"] < 1
    # The padded terminal step contributes zero compute, but it is not a
    # fictitious rank-zero operation and therefore cannot dilute capacity.
    assert output["evidence_mmd_it_capacity_ratio"] > 0.9


def test_native_dynamic_warmup_opens_continuously_without_changing_the_chart():
    base_config = _config()
    config = V39PolicyConfig(
        **{
            **base_config.__dict__,
            "latent_cvae_mmdit_operator_capacity": 1,
            "latent_cvae_mmdit_operator_rank": 32,
            "latent_cvae_mmdit_operator_groups": 4,
            "latent_cvae_mmdit_execution_controller": 1,
            "latent_cvae_mmdit_dynamic_block_route": 1,
            "latent_cvae_mmdit_dwell_mode": "learned",
            "latent_cvae_mmdit_max_dwell": 2,
        }
    )
    torch.manual_seed(613)
    decoder = EvidenceLatentMMDiTActionDecoder(config).train()
    with torch.no_grad():
        hidden = int(config.hidden_size)
        for block in decoder.blocks:
            block.action_mod.bias[2 * hidden : 3 * hidden].fill_(0.25)
            block.action_mod.bias[5 * hidden : 6 * hidden].fill_(0.20)
    inputs = _inputs(config)

    def run(step: int) -> dict[str, torch.Tensor]:
        # Hold host dropout fixed so the comparison isolates execution progress.
        torch.manual_seed(997)
        decoder.set_execution_training_step(step)
        with torch.no_grad():
            return decoder(**inputs)

    at_boundary = run(200)
    first_open_step = run(201)
    fully_open = run(1200)
    first_delta = (
        first_open_step["pred_velocity"] - at_boundary["pred_velocity"]
    ).float().square().mean().sqrt()
    full_delta = (
        fully_open["pred_velocity"] - at_boundary["pred_velocity"]
    ).float().square().mean().sqrt().clamp_min(1e-8)
    assert first_delta / full_delta < 0.02
    expected_candidates = config.depth * config.latent_cvae_mmdit_max_dwell + 1
    assert at_boundary["evidence_mmd_it_execution_candidate_value_field"].shape[2] == expected_candidates
    assert first_open_step["evidence_mmd_it_execution_candidate_value_field"].shape[2] == expected_candidates
    assert torch.equal(
        at_boundary["evidence_mmd_it_execution_candidate_value_mask"],
        first_open_step["evidence_mmd_it_execution_candidate_value_mask"],
    )
    assert at_boundary["evidence_mmd_it_execution_candidate_value_mask"][:, :, -1].all()
    assert at_boundary["evidence_mmd_it_execution_selection_entropy"] == 0
    assert first_open_step["evidence_mmd_it_execution_selection_entropy"] > 0


def test_native_terminal_candidate_is_identity_with_a_smaller_differentiable_prior():
    base_config = _config()
    config = V39PolicyConfig(
        **{
            **base_config.__dict__,
            "latent_cvae_mmdit_operator_capacity": 1,
            "latent_cvae_mmdit_operator_rank": 32,
            "latent_cvae_mmdit_operator_groups": 4,
            "latent_cvae_mmdit_execution_controller": 1,
            "latent_cvae_mmdit_dynamic_block_route": 1,
            "latent_cvae_mmdit_dwell_mode": "learned",
            "latent_cvae_mmdit_max_dwell": 2,
            "latent_cvae_mmdit_terminal_prior_weight": 0.25,
        }
    )
    decoder = EvidenceLatentMMDiTActionDecoder(config).train()
    blocks, repeats = decoder._global_execution_candidate_chart(batch=2, device=torch.device("cpu"))
    assert blocks.shape[1] == config.depth * config.latent_cvae_mmdit_max_dwell + 1
    assert torch.equal(blocks[:, -1], torch.full((2,), config.depth))
    assert torch.equal(repeats[:, -1], torch.zeros(2, dtype=torch.long))

    value = torch.zeros(
        2,
        blocks.shape[1],
        config.action_horizon,
        2,
        requires_grad=True,
    )
    pointer = torch.zeros(2, config.depth + 1)
    pointer[:, 0] = 1.0
    probabilities, _, _, _, terminal = decoder._mean_field_execution_policy(value, pointer)
    expected_terminal = 0.25 / (
        config.depth * config.latent_cvae_mmdit_max_dwell + 0.25
    )
    torch.testing.assert_close(
        terminal,
        torch.full_like(terminal, expected_terminal),
        atol=1e-7,
        rtol=0.0,
    )
    torch.testing.assert_close(probabilities[:, -1], terminal)
    terminal.sum().backward()
    assert value.grad is not None
    assert value.grad[:, -1].abs().sum() > 0

    decoder.set_execution_training_step(1200)
    output = decoder(**_inputs(config))
    torch.testing.assert_close(
        output["evidence_mmd_it_dwell_candidate_pred_velocity"][:, :, -1],
        output["evidence_mmd_it_execution_baseline_pred_velocity"],
    )
    assert output["evidence_mmd_it_terminal_prior_weight"] == 0.25
    assert output["evidence_mmd_it_terminal_probability"] > 0


def test_native_eval_ablation_reports_continuous_32_to_29_basis_reduction_honestly():
    config = V39PolicyConfig(
        **{
            **_config().__dict__,
            "latent_cvae_mmdit_operator_capacity": 1,
            "latent_cvae_mmdit_operator_rank": 32,
            "latent_cvae_mmdit_operator_groups": 32,
            "latent_cvae_mmdit_execution_controller": 1,
            "latent_cvae_mmdit_dynamic_block_route": 1,
            "latent_cvae_mmdit_dwell_mode": "learned",
            "latent_cvae_mmdit_max_dwell": 2,
            "latent_cvae_mmdit_execution_eval_policy": "soft",
        }
    )
    decoder = EvidenceLatentMMDiTActionDecoder(config).eval()
    decoder.set_execution_training_step(1200)
    decoder.set_execution_eval_ablation(policy="soft", capacity_gate=29.0 / 32.0)
    try:
        with torch.no_grad():
            reduced = decoder(**_inputs(config))
    finally:
        decoder.clear_execution_eval_ablation()
    torch.testing.assert_close(
        reduced["evidence_mmd_it_capacity_gate_mass"],
        torch.tensor(29.0 / 32.0),
    )
    torch.testing.assert_close(
        reduced["evidence_mmd_it_effective_basis_mass"],
        torch.tensor(29.0),
    )
    assert reduced["evidence_mmd_it_execution_eval_policy_code"] == 0
    with torch.no_grad():
        restored = decoder(**_inputs(config))
    assert restored["evidence_mmd_it_execution_eval_policy_code"] == 0


def test_native_capacity_never_owns_host_residual_amplitude():
    base_config = _config()
    config = V39PolicyConfig(
        **{
            **base_config.__dict__,
            "latent_cvae_mmdit_operator_capacity": 1,
            "latent_cvae_mmdit_operator_rank": 32,
            "latent_cvae_mmdit_operator_groups": 4,
            "latent_cvae_mmdit_execution_controller": 1,
            "latent_cvae_mmdit_dynamic_block_route": 1,
            "latent_cvae_mmdit_dwell_mode": "learned",
            "latent_cvae_mmdit_max_dwell": 2,
        }
    )
    decoder = EvidenceLatentMMDiTActionDecoder(config).train()
    decoder.set_execution_training_step(300)
    observed_execution_gates: list[object] = []
    handles = []
    for block in decoder.blocks:
        def capture_gate(_module, _args, kwargs):
            observed_execution_gates.append(kwargs.get("execution_gate"))

        handles.append(block.register_forward_pre_hook(capture_gate, with_kwargs=True))
    try:
        decoder(**_inputs(config))
    finally:
        for handle in handles:
            handle.remove()
    assert observed_execution_gates
    assert all(value is None for value in observed_execution_gates)


def test_native_execution_value_reader_uses_repeat_as_neutral_tie_break():
    config = _config()
    value_field = torch.zeros(2, 3, config.action_horizon, 2)
    non_final_mask = torch.tensor([[True, True, True], [True, True, True]])
    selected = EvidenceLatentMMDiTActionDecoder._select_execution_candidate(
        value_field,
        non_final_mask,
    )
    assert torch.equal(selected, torch.zeros(2, dtype=torch.long))
    value_field[:, 1] = -1.0
    selected = EvidenceLatentMMDiTActionDecoder._select_execution_candidate(
        value_field,
        non_final_mask,
    )
    assert torch.equal(selected, torch.ones(2, dtype=torch.long))
    final_mask = torch.tensor([[True, True, False], [True, True, False]])
    selected = EvidenceLatentMMDiTActionDecoder._select_execution_candidate(
        value_field,
        final_mask,
    )
    assert torch.equal(selected, torch.ones(2, dtype=torch.long))
