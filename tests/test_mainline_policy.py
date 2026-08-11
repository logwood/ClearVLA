from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import torch

from clearvla.mainline.config import ExperimentConfig
from clearvla.mainline.data.dataset import ObservedStateDatasetConfig
from clearvla.mainline.interfaces import (
    ActionSupervision,
    AuditMetadata,
    CurrentObservation,
    FutureSupervision,
    GoalCondition,
    ObservableHistory,
    OnlinePolicyInput,
    TrainingBatch,
)
from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
from clearvla.mainline.model.restored_observation import (
    _align_chart_to_later_frame,
    _v120_flow_field,
)
from clearvla.mainline.runtime.logging import archival_metrics
from clearvla.mainline.runtime.sampling import sample_action, sample_cached_action
from clearvla.mainline.train import _optimizer_group_context
from clearvla.mainline.training.engine import (
    EncodedTrainingBatch,
    MainlineTrainingEngine,
    validate_finite_training_batch,
)
from clearvla.mainline.training.optimizer import (
    WarmupCosineSchedule,
    build_optimizer,
    role_lr_scale,
)


def _config() -> ExperimentConfig:
    base = ExperimentConfig()
    config = replace(
        base,
        dimensions=replace(
            base.dimensions,
            action_basis_tokens=2,
            hidden_size=32,
            num_heads=4,
            visual_token_dim=16,
            goal_token_dim=16,
            patches_per_camera=64,
        ),
        observation=replace(
            base.observation,
            feature_dim=16,
            address_route_dim=8,
            flow_iterations=2,
            correlation_radius=1,
            raw_base_channels=8,
        ),
        top=replace(
            base.top,
            grounder_iterations=2,
            teacher_key_dim=8,
            goal_condition_dropout=0.0,
            action_history_condition_dropout=0.0,
            proposal_condition_dropout=0.0,
        ),
        bottom=replace(
            base.bottom,
            operator_rank=8,
            operator_groups=8,
            controller_tokens=4,
            controller_depth=1,
            controller_heads=4,
        ),
        optimizer=replace(base.optimizer, warmup_steps=2),
        runtime=replace(base.runtime, compute_dtype="fp32"),
    )
    config.validate()
    return config


def test_v120_exported_flow_is_reindexed_and_scaled_by_chart_side() -> None:
    source_forward = torch.full((1, 2, 1, 2, 8, 8), 2.0)
    source_backward = torch.full((1, 2, 1, 2, 8, 8), -2.0)
    scalar = torch.ones(1, 2, 1, 1, 8, 8)
    pack = SimpleNamespace(
        patch_flow_forward=source_forward,
        patch_flow_backward=source_backward,
        flow_confidence=scalar,
        flow_occlusion=torch.zeros_like(scalar),
    )

    field = _v120_flow_field(pack, -1)

    expected = 4.0 / 7.0
    assert torch.allclose(field.forward, torch.full_like(field.forward, expected))
    assert field.backward is not None
    assert torch.allclose(field.backward, torch.full_like(field.backward, -expected))
    # The exported pack has already converted the raw high-resolution flow to
    # 8x8 chart cells. Applying a second 24x24/native-patch conversion would
    # incorrectly shrink this two-cell displacement to 4/23.
    assert not torch.allclose(field.forward, torch.full_like(field.forward, 4.0 / 23.0))


def test_destination_indexed_forward_flow_aligns_source_chart() -> None:
    x = torch.arange(8, dtype=torch.float32)[None, None, None, :, None]
    value = x.expand(1, 1, 8, 8, 1).clone()
    flow = torch.zeros(1, 1, 2, 8, 8)
    flow[:, :, 0] = 2.0 / 7.0  # one 8x8 chart cell to the right

    aligned = _align_chart_to_later_frame(value, flow)

    # Destination x=1 samples source x=0; border padding defines x=0.
    assert torch.allclose(aligned[0, 0, :, 1:, 0], value[0, 0, :, :-1, 0])
    assert torch.allclose(aligned[0, 0, :, :1, 0], value[0, 0, :, :1, 0])


def _batch(config: ExperimentConfig, batch: int = 1) -> TrainingBatch:
    dims = config.dimensions
    device = torch.device("cpu")
    online = OnlinePolicyInput(
        observation=CurrentObservation(
            dino_history=torch.randn(
                batch,
                dims.visual_history_length,
                dims.num_cameras,
                dims.patches_per_camera,
                dims.visual_token_dim,
                device=device,
            ),
            raw_rgb=torch.rand(
                batch,
                dims.visual_history_length,
                dims.num_cameras,
                3,
                48,
                48,
                device=device,
            ),
        ),
        history=ObservableHistory(
            state=torch.randn(batch, dims.state_dim),
            action_state=torch.randn(batch, dims.action_dim),
            state_history=torch.randn(batch, dims.state_history_length, dims.state_dim),
            executed_action_history=torch.randn(
                batch,
                dims.executed_history_length,
                dims.action_dim,
            ),
        ),
        goal=GoalCondition(
            tokens=torch.randn(batch, 6, dims.goal_token_dim),
            mask=torch.ones(batch, 6, dtype=torch.bool),
        ),
    )
    offsets = torch.tensor(
        [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48],
        dtype=torch.long,
    )[None].expand(batch, -1)
    return TrainingBatch(
        online=online,
        action_target=ActionSupervision(
            normalized=torch.randn(batch, dims.action_horizon, dims.action_dim),
            raw_units=torch.randn(batch, dims.action_horizon, dims.action_dim),
            current_raw_units=torch.randn(batch, dims.action_dim),
        ),
        future=FutureSupervision(
            dino_supports=torch.randn(
                batch,
                dims.future_supports,
                dims.num_cameras,
                dims.patches_per_camera,
                dims.visual_token_dim,
                dtype=torch.float32,
            ),
            action_sequence=torch.randn(batch, 48, dims.action_dim, dtype=torch.float32),
            state_sequence=torch.randn(batch, 48, dims.state_dim, dtype=torch.float32),
            offsets=offsets,
        ),
    )


def test_full_mainline_has_complete_gradient_ownership() -> None:
    torch.manual_seed(4)
    config = _config()
    model = ClearVLAMainlinePolicy(config)
    optimizer, ownership = build_optimizer(model, config)
    assert set(ownership.role_counts) == {
        "observation",
        "grounding_host",
        "grounder",
        "intent",
        "coarse_action",
        "plan_recognizer",
        "history_proposal",
        "dynamics",
        "controlled_transition",
        "p1_factual",
        "p2_effect_reader",
        "consequence",
        "p3_compiler",
        "bottom_query",
        "bottom_evidence_adapter",
        "bottom_policy_bridge",
        "bottom_organizer",
        "bottom_mmdit",
        "bottom_capacity",
        "bottom_execution",
        "bottom_heads",
    }
    assert all(count > 0 for count in ownership.role_counts.values())
    schedule = WarmupCosineSchedule(
        optimizer,
        warmup_steps=2,
        total_steps=4,
        minimum_ratio=0.1,
    )
    engine = MainlineTrainingEngine(
        model=model,
        config=config,
        optimizer=optimizer,
        schedule=schedule,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    result = engine.train_step(_batch(config), collect_diagnostics=True)
    assert torch.isfinite(result.loss)
    assert result.learning_rate == config.optimizer.learning_rate / 2.0
    group_lrs = {str(group["name"]): float(group["lr"]) for group in optimizer.param_groups}
    assert group_lrs["grounder/decay"] == config.optimizer.learning_rate
    assert group_lrs["history_proposal/decay"] == config.optimizer.learning_rate * 0.625
    assert group_lrs["bottom_mmdit/decay"] == config.optimizer.learning_rate * 0.7
    assert group_lrs["bottom_capacity/nodecay"] == config.optimizer.learning_rate * 1.4
    assert "gradient_postclip_observation_l2" in result.metrics
    assert "gradient_observation_l2" not in result.metrics
    assert "gradient_global_preclip_l2" in result.materialize()
    archived = archival_metrics(result.materialize())
    assert "loss_action_flow_v120_comparable" in archived
    assert "loss_action_flow_event_balance_delta" in archived
    assert archived["loss_action_gripper_flow"] == archived[
        "loss_action_gripper_flow_unweighted"
    ]
    assert "loss_action_gripper_flow_event_balanced_audit" in archived
    assert "loss_action_flow_event_balanced_audit_first" in archived
    assert "loss_action_flow_event_balanced_audit_band_1_4" in archived
    assert "loss_action_flow_balanced_first" not in archived
    assert "loss_decoded_action_v120_comparable" in archived
    assert "loss_decoded_action_event_balance_delta" in archived
    assert "loss_contrib_action_flow" in archived
    assert "loss_contrib_future_dynamics" in archived
    assert "loss_contrib_future_transition" in archived
    assert "loss_contrib_object_reconstruction" in archived
    assert "loss_contrib_execution_value" in archived
    for name in (
        "loss_execution_value_target_spread",
        "loss_execution_value_predicted_spread",
        "loss_execution_value_correlation",
        "loss_execution_value_pairwise_accuracy",
        "loss_execution_value_decision_accuracy",
        "loss_execution_value_common_mode_ratio",
        "loss_execution_terminal_target_cost_margin",
        "loss_execution_terminal_predicted_cost_margin",
        "loss_execution_terminal_target_preferred_fraction",
    ):
        assert name in archived
    assert abs(archived["loss_contribution_gap"]) < 1e-5
    contribution_sum = sum(
        value for name, value in archived.items() if name.startswith("loss_contrib_")
    )
    assert abs(contribution_sum - archived["loss_total"]) < 1e-5
    # V120 serialized 287 active batch metrics.  The recovery mainline must
    # preserve at least that observability floor while exposing every new
    # owner boundary; a green forward/backward with a sparse log is not a
    # behavioral recovery result.
    assert len(archived) >= 287
    minimum_prefix_counts = {
        "loss_": 80,
        "observation_": 5,
        "object_grounding_": 20,
        "object_intent_": 40,
        "object_teacher_": 30,
        "object_w": 50,
        "p1_": 10,
        "object_p2_": 20,
        "object_p3_": 8,
        "controlled_transition_": 5,
        "bottom_": 4,
        "evidence_": 40,
        "gradient_postclip_": 20,
    }
    for prefix, minimum in minimum_prefix_counts.items():
        assert sum(name.startswith(prefix) for name in archived) >= minimum
    assert all(torch.isfinite(torch.tensor(float(value))) for value in archived.values())

    # Several action-facing projections deliberately start at exact zero so
    # the first optimizer step cannot disturb the protected factual base.  A
    # disconnected owner can hide behind that same first-step pattern, though.
    # Once the zero-initialized boundary has taken one update, every trainable
    # tensor must receive a nonzero ordinary-autograd signal on the next step.
    second_result = engine.train_step(_batch(config), collect_diagnostics=True)
    assert torch.isfinite(second_result.loss)
    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    # V120 deliberately keeps only the capacity/operation selector out of the
    # task graph during its first 200 steps.  The candidate value reader is
    # already supervised during this interval.
    assert missing
    assert all(
        name.startswith("bottom.decoder.operator_contractions.")
        or name.startswith("bottom.decoder.execution_controller.operation_")
        or name == "bottom.decoder.execution_controller.block_queries"
        or name.startswith("bottom.decoder.execution_controller.capacity_head.")
        for name in missing
    )
    dormant = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and parameter.grad is not None
        and not bool(parameter.grad.detach().abs().sum() > 0)
    ]
    assert dormant == []

    # Cross the serialized V120 warm-up boundary and verify that the same
    # ordinary task graph opens every mature execution owner.
    engine.global_step = 201
    engine.train_step(_batch(config), collect_diagnostics=True)
    engine.train_step(_batch(config), collect_diagnostics=True)
    capacity_gradient = sum(
        parameter.grad.detach().abs().sum()
        for operator in model.bottom.capacity
        for parameter in operator.parameters()
        if parameter.requires_grad and parameter.grad is not None
    )
    execution_capacity_gradient = sum(
        parameter.grad.detach().abs().sum()
        for parameter in model.bottom.execution.capacity_head.parameters()
        if parameter.requires_grad and parameter.grad is not None
    )
    execution_value_gradient = sum(
        parameter.grad.detach().abs().sum()
        for parameter in model.bottom.execution.value_reader.parameters()
        if parameter.requires_grad and parameter.grad is not None
    )
    assert capacity_gradient > 0
    assert execution_capacity_gradient > 0
    assert execution_value_gradient > 0
    assert len(ownership.trainable_names) == len(
        [parameter for parameter in model.parameters() if parameter.requires_grad]
    )


def test_optimizer_restores_v120_role_scales_and_capacity_no_decay() -> None:
    config = _config()
    model = ClearVLAMainlinePolicy(config)
    optimizer, _ = build_optimizer(model, config)
    groups = {str(group["name"]): group for group in optimizer.param_groups}
    base = config.optimizer.learning_rate
    assert role_lr_scale("grounder", config) == 1.0
    assert role_lr_scale("history_proposal", config) == 0.625
    assert role_lr_scale("bottom_mmdit", config) == 0.7
    assert role_lr_scale("bottom_capacity", config) == 1.4
    assert groups["history_proposal/decay"]["lr"] == base * 0.625
    assert groups["bottom_mmdit/decay"]["lr"] == base * 0.7
    assert groups["bottom_capacity/nodecay"]["lr"] == base * 1.4
    assert groups["bottom_capacity/nodecay"]["weight_decay"] == 0.0
    assert not any(name.startswith("bottom_capacity/decay") for name in groups)
    context = _optimizer_group_context(optimizer, config)
    history = context["history_proposal/decay"]
    assert history["base_learning_rate"] == base
    assert history["initial_learning_rate"] == base * 0.625
    assert history["role_learning_rate_scale"] == 0.625
    assert history["parameter_tensor_count"] > 0
    assert history["parameter_count"] > history["parameter_tensor_count"]


def test_full_mainline_cpu_bf16_forward_backward_is_finite() -> None:
    """Exercise the same autocast/FP32-teacher boundary used by CUDA smoke."""

    torch.manual_seed(44)
    base = _config()
    config = replace(base, runtime=replace(base.runtime, compute_dtype="bf16"))
    model = ClearVLAMainlinePolicy(config)
    optimizer, _ = build_optimizer(model, config)
    schedule = WarmupCosineSchedule(
        optimizer,
        warmup_steps=2,
        total_steps=4,
        minimum_ratio=0.1,
    )
    engine = MainlineTrainingEngine(
        model=model,
        config=config,
        optimizer=optimizer,
        schedule=schedule,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    result = engine.train_step(_batch(config), collect_diagnostics=True)
    assert torch.isfinite(result.loss)
    assert torch.isfinite(result.gradient_norm)
    assert result.gradient_norm > 0


def test_formal_eight_row_history_proposal_is_preserved_and_supervised() -> None:
    torch.manual_seed(41)
    config = _config()
    assert config.dimensions.executed_history_length == 8
    assert ObservedStateDatasetConfig().executed_action_offsets == (
        -24,
        -16,
        -12,
        -8,
        -6,
        -4,
        -2,
        -1,
    )
    model = ClearVLAMainlinePolicy(config)
    batch = _batch(config)
    proposal = model.history_proposal(
        batch.online.history.executed_action_history
    )
    assert proposal.tokens.shape == (
        1,
        config.dimensions.action_horizon,
        config.dimensions.hidden_size,
    )
    assert proposal.history_tokens.shape[1] == 7
    _, training_state, _ = model.encode_online(batch.online)
    targets, _ = model.build_training_targets(training_state, batch.future)
    assert targets.history_proposal_loss.isfinite()
    assert targets.history_proposal_loss > 0


def test_formal_condition_dropout_is_exact_null_only_on_the_policy_path() -> None:
    torch.manual_seed(43)
    base = _config()
    config = replace(
        base,
        top=replace(
            base.top,
            goal_condition_dropout=0.5,
            action_history_condition_dropout=0.5,
            proposal_condition_dropout=0.5,
        ),
    )
    model = ClearVLAMainlinePolicy(config).train()
    batch = _batch(config)
    complete_proposal = model.history_proposal(
        batch.online.history.executed_action_history
    )
    captured: dict[str, dict[str, object]] = {}

    def capture(name: str):
        def hook(_module, _args, kwargs):
            captured[name] = dict(kwargs)

        return hook

    handles = [
        model.top.intent.register_forward_pre_hook(capture("intent"), with_kwargs=True),
        model.factual_reader.register_forward_pre_hook(
            capture("factual"), with_kwargs=True
        ),
        model.transition.register_forward_pre_hook(
            capture("transition"), with_kwargs=True
        ),
    ]

    def zero_random(*size, **kwargs):
        kwargs.pop("generator", None)
        return torch.zeros(size, **kwargs)

    with mock.patch(
        "clearvla.mainline.model.policy.torch.rand",
        side_effect=zero_random,
    ):
        cache, training_state, metrics = model.encode_online(
            batch.online,
            training_mask=True,
            collect_diagnostics=True,
        )
    for handle in handles:
        handle.remove()

    assert torch.count_nonzero(captured["intent"]["goal_tokens"]) == 0
    assert torch.count_nonzero(captured["intent"]["executed_history"]) == 0
    factual_proposal = captured["factual"]["history_proposal"]
    transition_proposal = captured["transition"]["proposal"]
    assert torch.count_nonzero(factual_proposal.tokens) == 0
    assert torch.count_nonzero(transition_proposal.tokens) == 0
    assert torch.count_nonzero(cache.history.executed_action_history) == 0
    assert torch.equal(
        training_state.history_proposal.action_prediction,
        complete_proposal.action_prediction,
    )
    assert metrics["condition_goal_keep"] == 0
    assert metrics["condition_action_history_keep"] == 0
    assert metrics["condition_proposal_keep"] == 0

    deployment_generator = torch.Generator().manual_seed(91)
    generator_state = deployment_generator.get_state().clone()
    model.eval()
    model.encode_online(
        batch.online,
        training_mask=False,
        condition_generator=deployment_generator,
    )
    assert torch.equal(generator_state, deployment_generator.get_state())


def test_p1_protected_fact_cannot_be_attenuated_by_grounding_existence() -> None:
    torch.manual_seed(47)
    config = _config()
    model = ClearVLAMainlinePolicy(config).eval()
    batch = _batch(config)
    _, training_state, _ = model.encode_online(batch.online)
    facts = training_state.top.facts

    def read(existence: torch.Tensor):
        return model.factual_reader(
            evidence=training_state.observation,
            facts=replace(facts, existence=existence),
            intent=training_state.top.intent,
            coarse_action=training_state.top.coarse_action,
            history_proposal=training_state.history_proposal,
        )[0]

    low = read(torch.full_like(facts.existence, 1.0e-4))
    high = read(torch.ones_like(facts.existence))
    for name in (
        "fact_by_object",
        "object_posterior",
        "null_posterior",
        "chart_posterior",
        "camera_coordinates",
        "aggregate_fact",
    ):
        torch.testing.assert_close(getattr(low, name), getattr(high, name), rtol=0, atol=0)


def test_controlled_transition_is_a_real_zero_preserving_bottom_lane() -> None:
    torch.manual_seed(42)
    config = _config()
    model = ClearVLAMainlinePolicy(config)
    batch = _batch(config)
    cache, training_state, _ = model.encode_online(batch.online)
    time = torch.full((1,), 0.5)
    physical = model.action_codec.encode(
        batch.action_target.normalized,
        batch.online.history.action_state,
    )
    query = model.bottom.action_query(physical, time)
    compiled, _ = model.top.compile_policy(
        cache.top,
        factual_dock=cache.factual_dock,
        action_query=query,
    )
    evidence = model.bottom.compile_evidence_view(
        plan=compiled.plan,
        intent=cache.top.intent,
        history=cache.history,
        transition=cache.transition,
    )
    trajectory_start, trajectory_stop = evidence.ranges["trajectory"]
    assert torch.count_nonzero(
        evidence.value_tokens[:, trajectory_start:trajectory_stop]
    ) == 0
    assert all(
        not parameter.requires_grad
        for parameter in model.bottom.decoder.evidence_adapter.source_proj[
            "trajectory"
        ].parameters()
    )
    rollout_start, rollout_stop = evidence.ranges["rollout"]
    assert torch.count_nonzero(
        evidence.tokens[:, rollout_start:rollout_stop]
    ) > 0
    assert torch.count_nonzero(
        evidence.value_tokens[:, rollout_start:rollout_stop]
    ) == 0
    role_bank = model.bottom._role_bank(compiled.plan)
    assert torch.equal(role_bank.protected_detail, compiled.plan.protected_base)
    state_tokens, executed_tokens = model.bottom._state_memory(cache.history)
    intent_memory = model.bottom._intent_memory(
        cache.top.intent,
        state_tokens,
        executed_tokens,
    )
    assert set(intent_memory) == {"state", "executed"}
    start, stop = evidence.ranges["transition"]
    assert torch.count_nonzero(evidence.value_tokens[:, start:stop]) > 0
    neutral_transition = replace(
        cache.transition,
        value=torch.zeros_like(cache.transition.value),
    )
    neutral_evidence = model.bottom.compile_evidence_view(
        plan=compiled.plan,
        intent=cache.top.intent,
        history=cache.history,
        transition=neutral_transition,
    )
    assert torch.count_nonzero(neutral_evidence.value_tokens[:, start:stop]) == 0

    anchors = int(model.bottom.core_config.future_anchors)
    spatial = (
        int(model.bottom.core_config.num_cameras)
        * int(model.bottom.core_config.future_grid_size) ** 2
    )
    marker = torch.arange(
        1,
        anchors + 1,
        dtype=cache.transition.value.dtype,
    )[None, :, None, None].expand(1, anchors, spatial, config.dimensions.hidden_size)
    marked_transition = replace(
        cache.transition,
        value=marker.reshape(1, anchors * spatial, config.dimensions.hidden_size),
    )
    event_context = model.bottom._transition_event_context(marked_transition)
    lower = 0
    for index, upper in enumerate(model.bottom.core_config.flow_jepa_action_offsets):
        assert torch.equal(
            event_context[:, lower:upper],
            torch.full_like(event_context[:, lower:upper], float(index + 1)),
        )
        lower = int(upper)
    assert lower == config.dimensions.action_horizon
    zero_proposal = replace(
        training_state.history_proposal,
        tokens=torch.zeros_like(training_state.history_proposal.tokens),
    )
    zero_transition, _ = model.transition(
        dynamics=cache.top.predicted_dynamics,
        facts=training_state.top.facts,
        proposal=zero_proposal,
        history=cache.history,
    )
    assert torch.equal(
        zero_transition.action_coefficients,
        zero_transition.neutral_coefficients,
    )
    assert torch.count_nonzero(zero_transition.value) == 0
    assert len(model.top.grounding_host.blocks) == 3
    assert model.top.dynamics.w1 is not model.top.dynamics.w2
    assert model.top.dynamics.near_heads is not model.top.dynamics.far_heads
    assert model.history_proposal.OFFSETS == (-24, -16, -12, -8, -6, -4, -2, -1)
    assert len(model.history_proposal.blocks) == 2
    assert model.history_proposal.recent_tokens == 4
    assert model.history_proposal.summary_tokens == 3
    assert model.factual_reader.role_host is not None
    assert model.transition.rank == 8
    assert model.transition.action_queries.shape[1] == 8
    assert len(model.bottom.blocks) == 3
    assert len(model.bottom.capacity) == 3
    assert model.bottom.execution is not None


def test_five_step_deployment_builds_static_evidence_once_and_no_teacher() -> None:
    torch.manual_seed(5)
    config = _config()
    model = ClearVLAMainlinePolicy(config)
    batch = _batch(config)
    calls = [0 for _ in model.bottom.blocks]
    observation_calls = 0
    factual_calls = 0
    teacher_calls = 0
    grounding_host_calls = 0
    history_proposal_calls = 0
    p1_host_calls = 0
    transition_calls = 0
    handles = []

    def count_observation(_module, _args, _output):
        nonlocal observation_calls
        observation_calls += 1

    def count_factual(_module, _args, _output):
        nonlocal factual_calls
        factual_calls += 1

    def count_teacher(_module, _args, _output):
        nonlocal teacher_calls
        teacher_calls += 1

    def count_grounding_host(_module, _args, _output):
        nonlocal grounding_host_calls
        grounding_host_calls += 1

    def count_history_proposal(_module, _args, _output):
        nonlocal history_proposal_calls
        history_proposal_calls += 1

    def count_p1_host(_module, _args, _output):
        nonlocal p1_host_calls
        p1_host_calls += 1

    def count_transition(_module, _args, _output):
        nonlocal transition_calls
        transition_calls += 1

    handles.append(model.observation.register_forward_hook(count_observation))
    handles.append(model.factual_reader.register_forward_hook(count_factual))
    handles.append(model.top.teacher.register_forward_hook(count_teacher))
    handles.append(model.top.grounding_host.register_forward_hook(count_grounding_host))
    handles.append(model.history_proposal.register_forward_hook(count_history_proposal))
    handles.append(model.factual_reader.role_host.register_forward_hook(count_p1_host))
    handles.append(model.transition.register_forward_hook(count_transition))
    for index, block in enumerate(model.bottom.blocks):

        def count_call(_module, _args, _output, *, index=index):
            calls[index] += 1

        handles.append(block.register_forward_hook(count_call))
    with mock.patch.object(
        model.observation.encoder.flow,
        "forward",
        wraps=model.observation.encoder.flow.forward,
    ) as semantic_flow, mock.patch.object(
        model.observation.encoder.raw_flow,
        "forward",
        wraps=model.observation.encoder.raw_flow.forward,
    ) as raw_flow:
        result = sample_action(
            model,
            batch.online,
            config,
            dtype=torch.float32,
        )
    assert observation_calls == 1
    assert factual_calls == 1
    assert teacher_calls == 0
    assert grounding_host_calls == 1
    assert history_proposal_calls == 1
    assert p1_host_calls == 1
    assert transition_calls == 1
    # Both V120 correspondence scales batch all adjacent pairs/directions in
    # one invocation and are built once outside the five ODE steps.
    assert semantic_flow.call_count == 1
    assert raw_flow.call_count == 1
    assert tuple(result.step_times.shape) == (5,)
    assert tuple(result.action.shape) == tuple(batch.action_target.normalized.shape)
    assert torch.isfinite(result.action).all()
    # The restored learned V120 execution chart may evaluate several
    # block/dwell candidates inside one ODE step.  Those are dynamic bottom
    # operations; the expensive observation/G/S/W/P1 sources above must still
    # be built exactly once.
    assert all(value >= config.runtime.inference_steps for value in calls)
    for handle in handles:
        handle.remove()


def test_p1_refines_the_local_chart_per_query_and_returns_action_pressure_to_g() -> None:
    torch.manual_seed(51)
    config = _config()
    model = ClearVLAMainlinePolicy(config).eval()
    batch = _batch(config)
    cache, training_state, metrics = model.encode_online(
        batch.online,
        geometry_supervision=False,
        collect_diagnostics=True,
    )
    dock = cache.factual_dock
    inherited = training_state.top.facts.object_to_chart[:, None, None].expand_as(
        dock.chart_posterior
    )
    assert not torch.equal(dock.chart_posterior, inherited)
    assert metrics["p1_query_chart_variation"] > 0
    assert metrics["p1_query_coordinate_variation"] > 0
    assignment_gradient = torch.autograd.grad(
        dock.chart_posterior.square().sum(),
        training_state.top.facts.candidate_assignment,
        retain_graph=True,
    )[0]
    assert torch.count_nonzero(assignment_gradient) > 0
    assert tuple(cache.transition.value.shape[1:]) == (
        4 * config.dimensions.num_cameras * 8 * 8,
        config.dimensions.hidden_size,
    )
    assert metrics["controlled_transition_dense_rows"] == (
        4 * config.dimensions.num_cameras * 8 * 8
    )
    assert metrics["controlled_transition_retained_rows"] == (
        4 * config.dimensions.num_cameras * 8 * 8
    )
    assert metrics["controlled_transition_pool_removed"] == 0


def test_sampling_rejects_dtype_that_differs_from_serialized_runtime() -> None:
    config = _config()
    model = ClearVLAMainlinePolicy(config)
    try:
        sample_action(
            model,
            _batch(config).online,
            config,
            dtype=torch.bfloat16,
        )
    except ValueError as error:
        assert "compute dtype differs" in str(error)
    else:
        raise AssertionError("sampling must not silently override checkpoint dtype identity")


def test_cached_deployment_forces_eval_mode_and_is_repeatable() -> None:
    torch.manual_seed(23)
    config = _config()
    model = ClearVLAMainlinePolicy(config).eval()
    batch = _batch(config)
    with torch.no_grad():
        cache, _, _ = model.encode_online(
            batch.online,
            training_mask=False,
            geometry_supervision=False,
        )
    noise = torch.randn(
        batch.action_target.batch,
        config.dimensions.action_horizon,
        model.action_codec.physical_dim,
    )
    model.train()
    first = sample_cached_action(
        model,
        cache,
        config,
        initial_physical_noise=noise,
        dtype=torch.float32,
    )
    assert not model.training
    model.train()
    second = sample_cached_action(
        model,
        cache,
        config,
        initial_physical_noise=noise,
        dtype=torch.float32,
    )
    assert not model.training
    assert torch.equal(first.action, second.action)


def test_validation_execution_interventions_reach_the_native_v120_controller() -> None:
    torch.manual_seed(231)
    config = _config()
    model = ClearVLAMainlinePolicy(config).eval()
    model.set_training_step(1200)
    batch = _batch(config)
    with torch.no_grad():
        cache, _, _ = model.encode_online(batch.online)
        physical = model.action_codec.sample_noise(
            batch.online.batch,
            device=batch.online.device,
            dtype=torch.float32,
        )
        time = torch.full((batch.online.batch,), 0.5)
        no_updates = model.velocity(
            cache,
            noisy_action_field=physical,
            time=time,
            execution_mode="no_updates",
            collect_diagnostics=True,
        )
        full_updates = model.velocity(
            cache,
            noisy_action_field=physical,
            time=time,
            execution_mode="full_updates",
            collect_diagnostics=True,
        )
    # V120 capacity is rank retention, not block amplitude.  The no-update
    # intervention therefore selects prefix row zero instead of pretending
    # that capacity=0 disables the host operation.
    assert no_updates.metrics["evidence_mmd_it_capacity_ratio"] == 1
    assert full_updates.metrics["evidence_mmd_it_capacity_ratio"] == 1
    assert no_updates.metrics["evidence_mmd_it_execution_eval_policy_code"] == 2
    assert full_updates.metrics["evidence_mmd_it_execution_eval_policy_code"] == 1
    assert not torch.equal(
        no_updates.bottom.physical_velocity,
        full_updates.bottom.physical_velocity,
    )
    torch.testing.assert_close(
        no_updates.bottom.physical_velocity,
        no_updates.bottom.decoder_tensors[
            "evidence_mmd_it_prefix_pred_velocity"
        ][:, 0],
    )
    assert no_updates.metrics["bottom_execution_output_block_count"] == 0
    assert full_updates.metrics["bottom_execution_output_block_count"] == 3


def test_proposal_ablation_rebuilds_only_the_proposal_owned_cache_boundary() -> None:
    torch.manual_seed(232)
    config = _config()
    model = ClearVLAMainlinePolicy(config).eval()
    batch = _batch(config)
    with torch.no_grad():
        cache, training_state, _ = model.encode_online(batch.online)
        original_tokens = training_state.history_proposal.tokens.clone()
        ablated = model.proposal_ablation_cache(cache, training_state)
    assert torch.equal(training_state.history_proposal.tokens, original_tokens)
    assert torch.equal(ablated.top.intent.interval_queries, cache.top.intent.interval_queries)
    assert torch.equal(
        ablated.top.predicted_dynamics.semantic_delta,
        cache.top.predicted_dynamics.semantic_delta,
    )
    assert not torch.equal(ablated.factual_dock.aggregate_fact, cache.factual_dock.aggregate_fact)
    assert not torch.equal(ablated.transition.value, cache.transition.value)


def test_frame_progress_audit_is_detached_from_forward_and_reports_s_w_correlations() -> None:
    torch.manual_seed(31)
    config = _config()
    model = ClearVLAMainlinePolicy(config).eval()
    batch = _batch(config, batch=4)
    batch = replace(
        batch,
        audit=AuditMetadata(frame_progress=torch.linspace(0.0, 1.0, 4)),
    )
    with torch.no_grad():
        cache, training_state, metrics = model.encode_online(
            batch.online,
            training_mask=False,
            geometry_supervision=False,
            collect_diagnostics=False,
        )
    audit_metrics = MainlineTrainingEngine._audit_progress_metrics(
        batch,
        EncodedTrainingBatch(cache=cache, training_state=training_state, metrics=metrics),
    )
    expected = {
        "object_intent_audit_frame_progress_centroid_correlation",
        "object_intent_audit_frame_progress_state_change_correlation",
        "object_w_audit_frame_progress_successor_correlation",
        "object_w_audit_frame_progress_interval_variation_correlation",
    }
    assert expected <= set(audit_metrics)
    assert all(torch.isfinite(value) for value in audit_metrics.values())


def test_native_execution_probabilities_and_dwell_are_bounded() -> None:
    torch.manual_seed(24)
    config = _config()
    model = ClearVLAMainlinePolicy(config)
    model.set_training_step(1200)
    batch = _batch(config)
    cache, _, _ = model.encode_online(batch.online)
    time = torch.full((1,), 0.5)
    physical = model.action_codec.encode(
        batch.action_target.normalized,
        batch.online.history.action_state,
    )
    output = model.velocity(
        cache,
        noisy_action_field=physical,
        time=time,
        collect_diagnostics=True,
    )
    operation = output.metrics["evidence_mmd_it_operation_probability"]
    terminal = output.metrics["evidence_mmd_it_terminal_probability"]
    capacity = output.metrics["evidence_mmd_it_capacity_ratio"]
    dwell = output.metrics["evidence_mmd_it_dwell_expected"]
    assert torch.allclose(operation + terminal, operation.new_ones(()), atol=1e-6)
    assert 0 <= capacity <= 1
    assert 0 <= dwell <= model.bottom.decoder.max_dwell


def test_scheduler_applies_warmup_before_first_optimizer_update() -> None:
    config = _config()
    model = ClearVLAMainlinePolicy(config)
    optimizer, _ = build_optimizer(model, config)
    schedule = WarmupCosineSchedule(
        optimizer,
        warmup_steps=4,
        total_steps=8,
        minimum_ratio=0.1,
    )
    for base, group in zip(schedule.base_lrs, optimizer.param_groups, strict=True):
        assert float(group["lr"]) == base / 4.0
    schedule.step()
    for base, group in zip(schedule.base_lrs, optimizer.param_groups, strict=True):
        assert float(group["lr"]) == base / 2.0


def test_preflight_rejects_a_same_shape_but_wrong_future_offset_schedule() -> None:
    config = _config()
    batch = _batch(config)
    wrong = batch.future.offsets.clone()
    wrong[:, 5] = 25
    broken = replace(batch, future=replace(batch.future, offsets=wrong))
    try:
        validate_finite_training_batch(broken)
    except ValueError as error:
        assert "exactly 4,8,...,48" in str(error)
    else:
        raise AssertionError("future teacher offsets are part of the semantic ABI")
