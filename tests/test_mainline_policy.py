from __future__ import annotations

import copy
import math
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
    _v120_flow_field,
)
from clearvla.mainline.runtime.logging import archival_metrics
from clearvla.mainline.runtime.sampling import sample_action, sample_cached_action
from clearvla.mainline.train import _optimizer_group_context
from clearvla.mainline.training.engine import (
    EncodedTrainingBatch,
    MainlineTrainingEngine,
    NonFiniteGradientError,
    validate_finite_training_batch,
)
from clearvla.mainline.training.losses import LossLedger
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


def test_restored_observation_keeps_consumed_v120_address_modules_trainable() -> None:
    model = ClearVLAMainlinePolicy(_config())
    parameters = dict(model.observation.encoder.named_parameters())
    for name in (
        "history_type",
        "camera_type",
        "spatial_type",
        "evidence_type",
        "future_query",
        "future_anchor_type",
    ):
        assert parameters[name].requires_grad, name
    for prefix in (
        "motion_key.",
        "organized_key.",
        "early_masked_raw_context.",
        "future_motion.",
        "future_history_score.",
        "future_transition.",
    ):
        rows = [
            parameter
            for name, parameter in parameters.items()
            if name.startswith(prefix)
        ]
        assert rows, prefix
        assert all(parameter.requires_grad for parameter in rows), prefix
    # The G3 block remains active, while the parallel generic route query is
    # absent from the exported object-intent GroundedFactSet.
    assert not parameters[
        "progressive_grounding_address.query_projections.2.weight"
    ].requires_grad


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
        "grounding",
        "grounder",
        "intent",
        "coarse_action",
        "intent_supervisor",
        "history_proposal",
        "dynamics",
        "controlled_transition",
        "p1_factual",
        "p2_effect_reader",
        "consequence",
        "p3_compiler",
        "v120_canvas_seed",
        "v120_layer_contracts",
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
    assert group_lrs["v120_canvas_seed/decay"] == config.optimizer.learning_rate
    assert group_lrs["v120_layer_contracts/decay"] == config.optimizer.learning_rate
    assert group_lrs["history_proposal/decay"] == config.optimizer.learning_rate * 0.625
    assert group_lrs["bottom_mmdit/decay"] == config.optimizer.learning_rate * 0.7
    assert group_lrs["bottom_capacity/nodecay"] == config.optimizer.learning_rate * 1.4
    assert "gradient_raw_observation_l2" in result.metrics
    assert "gradient_postlocal_observation_l2" in result.metrics
    assert "gradient_postglobal_observation_l2" in result.metrics
    assert result.metrics["gradient_postlocal_bottom_decoder_l2"] <= 1.0001
    torch.testing.assert_close(
        result.metrics["gradient_raw_bottom_execution_l2"],
        result.metrics["gradient_postlocal_bottom_execution_l2"],
        rtol=1e-6,
        atol=1e-8,
    )
    expected_controller = torch.minimum(
        result.metrics["gradient_raw_bottom_execution_l2"],
        torch.ones_like(result.metrics["gradient_raw_bottom_execution_l2"]),
    )
    torch.testing.assert_close(
        result.metrics["gradient_postglobal_bottom_execution_l2"],
        expected_controller,
        rtol=1e-5,
        atol=1e-7,
    )
    assert result.metrics["gradient_postglobal_main_l2"] <= 1.0001
    assert result.metrics["gradient_postglobal_execution_controller_l2"] <= 1.0001
    assert result.metrics["gradient_postglobal_global_l2"] <= math.sqrt(2.0) + 1e-4
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
        "object_w": 30,
        "p1_": 10,
        "object_p2_": 20,
            "object_p3_": 4,
        "controlled_transition_": 5,
        "bottom_": 4,
        "evidence_": 40,
        "gradient_raw_": 20,
        "gradient_postlocal_": 20,
        "gradient_postglobal_": 20,
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
    ), missing
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


def test_nonfinite_gradient_reports_first_owner_before_any_update() -> None:
    class TinyOwnerModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.observation = torch.nn.Linear(2, 2, bias=False)
            self.seen_step: int | None = None

        def set_training_step(self, step: int) -> None:
            self.seen_step = int(step)

    config = _config()
    model = TinyOwnerModel()
    parameter = model.observation.weight
    optimizer = torch.optim.AdamW(
        [
            {
                "params": (parameter,),
                "lr": config.optimizer.learning_rate,
                "name": "observation/decay",
                "parameter_names": ("observation.weight",),
            }
        ],
        lr=config.optimizer.learning_rate,
    )
    schedule = WarmupCosineSchedule(
        optimizer,
        warmup_steps=2,
        total_steps=8,
        minimum_ratio=0.1,
    )
    engine = MainlineTrainingEngine(
        model=model,  # type: ignore[arg-type]
        config=config,
        optimizer=optimizer,
        schedule=schedule,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    engine.global_step = 7
    loss = parameter.square().sum()
    ledger = LossLedger(
        total=loss,
        groups={
            "action": loss,
            "representation": loss.new_zeros(()),
            "execution": loss.new_zeros(()),
        },
        contributions={"action_flow": loss},
        terms={"action_flow": loss},
    )

    def corrupt(gradient: torch.Tensor) -> torch.Tensor:
        result = gradient.clone()
        result.flatten()[0] = torch.nan
        result.flatten()[1] = torch.inf
        result.flatten()[2] = -torch.inf
        return result

    handle = parameter.register_hook(corrupt)
    parameter_before = parameter.detach().clone()
    learning_rate_before = float(optimizer.param_groups[0]["lr"])
    schedule_before = schedule.step_index
    try:
        with mock.patch.object(engine, "_forward", return_value=(ledger, {})):
            engine.train_step(object())  # type: ignore[arg-type]
    except NonFiniteGradientError as error:
        report = error.report
    else:
        raise AssertionError("non-finite gradients must fail before clipping")
    finally:
        handle.remove()

    assert report.parameter_name == "observation.weight"
    assert report.parameter_role == "observation"
    assert report.optimizer_group == "observation/decay"
    assert report.shape == (2, 2)
    assert report.dtype == "float32"
    assert report.nan_count == 1
    assert report.positive_inf_count == 1
    assert report.negative_inf_count == 1
    assert report.finite_fraction == 0.25
    assert report.finite_max_abs > 0.0
    assert report.global_norm == "nan"
    torch.testing.assert_close(parameter, parameter_before)
    assert not optimizer.state
    assert float(optimizer.param_groups[0]["lr"]) == learning_rate_before
    assert schedule.step_index == schedule_before
    assert engine.global_step == 7

    parameter.grad = torch.full_like(parameter, torch.finfo(parameter.dtype).max)
    try:
        engine._clip_gradients_with_first_offender()
    except NonFiniteGradientError as error:
        overflow_report = error.report
    else:
        raise AssertionError("a finite-element norm overflow must name its owner")
    assert overflow_report.parameter_name == "observation.weight"
    assert overflow_report.finite_fraction == 1.0
    assert overflow_report.nan_count == 0
    assert overflow_report.positive_inf_count == 0
    assert overflow_report.negative_inf_count == 0
    assert overflow_report.global_norm == "+inf"


def test_optimizer_restores_v120_role_scales_and_capacity_no_decay() -> None:
    config = _config()
    model = ClearVLAMainlinePolicy(config)
    optimizer, _ = build_optimizer(model, config)
    groups = {str(group["name"]): group for group in optimizer.param_groups}
    base = config.optimizer.learning_rate
    assert role_lr_scale("grounder", config) == 1.0
    assert role_lr_scale("history_proposal", config) == 0.625
    assert role_lr_scale("v120_canvas_seed", config) == 1.0
    assert role_lr_scale("v120_layer_contracts", config) == 1.0
    assert role_lr_scale("bottom_mmdit", config) == 0.7
    assert role_lr_scale("bottom_capacity", config) == 1.4
    assert groups["history_proposal/decay"]["lr"] == base * 0.625
    assert groups["v120_canvas_seed/decay"]["lr"] == base
    assert groups["v120_layer_contracts/decay"]["lr"] == base
    assert groups["bottom_mmdit/decay"]["lr"] == base * 0.7
    assert groups["bottom_capacity/nodecay"]["lr"] == base * 1.4
    assert groups["bottom_capacity/nodecay"]["weight_decay"] == 0.0
    assert not any(name.startswith("bottom_capacity/decay") for name in groups)
    grounder_names = set(groups["grounder/decay"]["parameter_names"])
    dynamics_names = set(groups["dynamics/decay"]["parameter_names"])
    assert "top.grounder.decode_content_residual.weight" in grounder_names
    assert "top.grounder.decode_public_position.weight" in grounder_names
    assert "top.dynamics.typed_base_interaction.weight" in dynamics_names
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
    teacher_calls = 0

    def _count_teacher_call(_module, _inputs, _output) -> None:
        nonlocal teacher_calls
        teacher_calls += 1

    hook = model.top.teacher.register_forward_hook(_count_teacher_call)
    try:
        result = engine.train_step(_batch(config), collect_diagnostics=True)
    finally:
        hook.remove()
    assert teacher_calls == 1
    assert torch.isfinite(result.loss)
    assert torch.isfinite(result.gradient_norm)
    assert result.gradient_norm > 0


def test_eval_step_retains_v120_execution_candidate_supervision() -> None:
    """Regression for schema-20 validation dropping non-scalar decoder rows."""

    torch.manual_seed(45)
    config = _config()
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
        dtype=torch.float32,
    )

    # The historical failure appeared immediately after the configured four
    # diagnostic validation batches.  Exercise the non-diagnostic path.
    result = engine.eval_step(_batch(config), collect_diagnostics=False)

    assert torch.isfinite(result.loss)
    for name in (
        "loss_execution_value",
        "loss_execution_value_target_spread",
        "loss_execution_value_predicted_spread",
        "loss_execution_value_pairwise_accuracy",
    ):
        assert name in result.metrics
        assert torch.isfinite(torch.as_tensor(result.metrics[name]))


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
    assert "history_proposal" not in captured["factual"]
    assert "proposal" not in captured["factual"]
    assert torch.count_nonzero(captured["factual"]["clean_basis_tokens"]) > 0
    assert torch.count_nonzero(cache.history.executed_action_history) == 0
    assert torch.equal(
        training_state.history_proposal.action_prediction,
        complete_proposal.action_prediction,
    )
    assert metrics["condition_goal_keep"] == 0
    assert metrics["condition_action_history_keep"] == 0
    assert "condition_proposal_keep" not in metrics

    deployment_generator = torch.Generator().manual_seed(91)
    generator_state = deployment_generator.get_state().clone()
    model.eval()
    model.encode_online(
        batch.online,
        training_mask=False,
        condition_generator=deployment_generator,
    )
    assert torch.equal(generator_state, deployment_generator.get_state())


def test_p1_dock_is_exact_v120_precision_without_global_k_axis() -> None:
    torch.manual_seed(47)
    config = _config()
    model = ClearVLAMainlinePolicy(config).eval()
    batch = _batch(config)
    cache, training_state, metrics = model.encode_online(
        batch.online,
        collect_diagnostics=True,
    )
    assert tuple(cache.factual_dock.protected_detail.shape) == (
        1,
        24,
        config.dimensions.action_basis_tokens,
        config.dimensions.hidden_size,
    )
    assert set(cache.factual_dock.__dataclass_fields__) == {"protected_detail"}
    assert metrics["flow_jepa_p1_query_rows"] == 24
    assert metrics["flow_jepa_typed_p1_micro_grid"] == 3
    assert metrics["flow_jepa_typed_p1_micro_token_count"] == 9
    fine = training_state.observation.progressive_state.dynamic_fine_values
    assert fine is not None and int(fine.shape[-2]) == 49


def test_progressive_grounding_executes_g1_g2_g3_and_rematerializes_n49_once() -> None:
    torch.manual_seed(470)
    config = _config()
    model = ClearVLAMainlinePolicy(config).eval()
    batch = _batch(config)
    compiler = model.observation.encoder.soft_address_compiler
    with mock.patch.object(
        compiler,
        "progressive_fine_candidates",
        wraps=compiler.progressive_fine_candidates,
    ) as rematerialize, mock.patch.object(
        model.observation,
        "advance_progressive_grounding",
        wraps=model.observation.advance_progressive_grounding,
    ) as advance:
        _, training_state, metrics = model.encode_online(
            batch.online,
            collect_diagnostics=True,
        )
    assert [call.kwargs["stage"] for call in advance.call_args_list] == [1, 2, 3]
    assert rematerialize.call_count == 1
    progressive = training_state.observation.progressive_state
    assert progressive.stage == 3
    assert progressive.grounded_fact_set is not None
    assert progressive.dynamic_fine_values is not None
    assert int(progressive.dynamic_fine_values.shape[-2]) == 49
    assert metrics["observation_g1_g2_g3_completed"] == 1
    # The G3 correction is zero-initialized, so the fresh model must inherit
    # its G2 parent posterior exactly instead of silently replacing it.
    assert metrics["observation_g3_parent_semantic_l1"] <= 1e-7


def test_grounding_canvas_structurally_excludes_forbidden_conditions() -> None:
    torch.manual_seed(4701)
    config = _config()
    model = ClearVLAMainlinePolicy(config).eval()
    batch = _batch(config)
    prepared = model.observation.prepare(batch.online.observation)
    role = model.bottom.sample_role_table(prepared.pack.value_tokens)
    canvas, slices = model.bottom.grounding_canvas(
        state=batch.online.history.state,
        rollout_init=prepared.pack.future_queries,
        role=role,
    )
    for name in ("task", "state_history", "executed", "proposal", "trajectory", "stage"):
        assert slices[name].start == slices[name].stop
    assert slices["state"].stop - slices["state"].start == 1
    assert slices["rollout"].stop > slices["rollout"].start
    assert slices["registers"].stop > slices["registers"].start
    assert canvas.shape[1] == sum(
        current.stop - current.start for current in slices.values()
    )


def test_v120_p1_query_chunking_preserves_output_and_parameter_gradients() -> None:
    torch.manual_seed(4702)
    config = _config()
    model = ClearVLAMainlinePolicy(config).eval()
    batch = _batch(config)
    captured: dict[str, object] = {}

    def capture(_module, args, kwargs):
        captured["args"] = args
        captured["kwargs"] = {**kwargs, "collect_diagnostics": False}

    hook = model.factual_reader.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        with torch.no_grad():
            model.encode_online(batch.online)
    finally:
        hook.remove()
    assert "args" in captured and "kwargs" in captured
    chunked = copy.deepcopy(model.factual_reader).eval()
    unchunked = copy.deepcopy(model.factual_reader).eval()
    chunked.address_query_batch_budget = 1
    unchunked.address_query_batch_budget = 1_000_000
    # Checkpointing is an independent production memory contract. Disable it
    # here so this test isolates the query-factorization equivalence itself.
    chunked.raw_activation_checkpoint = False
    unchunked.raw_activation_checkpoint = False
    args = captured["args"]
    kwargs = captured["kwargs"]
    assert isinstance(args, tuple) and isinstance(kwargs, dict)
    output_chunked, _ = chunked(*args, **kwargs)
    output_full, _ = unchunked(*args, **kwargs)
    torch.testing.assert_close(output_chunked, output_full, atol=3e-6, rtol=3e-6)

    named_chunked = tuple(
        (name, parameter)
        for name, parameter in chunked.named_parameters()
        if parameter.requires_grad
    )
    named_full = tuple(
        (name, parameter)
        for name, parameter in unchunked.named_parameters()
        if parameter.requires_grad
    )
    assert tuple(name for name, _ in named_chunked) == tuple(name for name, _ in named_full)
    gradients_chunked = torch.autograd.grad(
        output_chunked.float().square().mean(),
        tuple(parameter for _, parameter in named_chunked),
        allow_unused=True,
    )
    gradients_full = torch.autograd.grad(
        output_full.float().square().mean(),
        tuple(parameter for _, parameter in named_full),
        allow_unused=True,
    )
    for (name, _), left, right in zip(
        named_chunked,
        gradients_chunked,
        gradients_full,
        strict=True,
    ):
        assert (left is None) == (right is None), name
        if left is not None and right is not None:
            torch.testing.assert_close(left, right, atol=1e-5, rtol=1e-5, msg=name)


def test_p1_has_no_global_object_value_or_learned_null_shortcut() -> None:
    torch.manual_seed(471)
    config = _config()
    model = ClearVLAMainlinePolicy(config).eval()
    batch = _batch(config)
    cache, _, _ = model.encode_online(batch.online)
    parameter_names = {name for name, _ in model.factual_reader.named_parameters()}
    assert not any("object_value" in name for name in parameter_names)
    assert not any("learned_null" in name for name in parameter_names)
    assert torch.count_nonzero(cache.factual_dock.protected_detail) > 0


def test_dynamic_p1_completes_cached_detail_at_each_ode_time() -> None:
    torch.manual_seed(472)
    config = _config()
    model = ClearVLAMainlinePolicy(config).eval()
    batch = _batch(config)
    cache, _, _ = model.encode_online(batch.online)
    physical = model.action_codec.encode(
        batch.action_target.normalized,
        batch.online.history.action_state,
    )
    query = model.bottom.action_query(physical, torch.full((1,), 0.25))
    first, first_metrics = model.bottom.complete_p1_fact(
        action_query=query,
        protected_detail=cache.factual_dock.protected_detail,
        time=torch.full((1,), 0.25),
        collect_diagnostics=True,
    )
    second, _ = model.bottom.complete_p1_fact(
        action_query=query,
        protected_detail=cache.factual_dock.protected_detail,
        time=torch.full((1,), 0.75),
    )
    assert not torch.equal(first, second)
    assert first_metrics["p1_protected_detail_rms"] > 0
    assert first_metrics["p1_dynamic_delta_rms"] > 0
    assert first_metrics["p1_completed_fact_rms"] > 0


def test_controlled_transition_restores_v120_dynamic_action_and_bottom_lane() -> None:
    torch.manual_seed(42)
    config = _config()
    model = ClearVLAMainlinePolicy(config)
    model.eval()
    batch = _batch(config)
    cache, training_state, _ = model.encode_online(batch.online)
    time = torch.full((1,), 0.5)
    physical = model.action_codec.encode(
        batch.action_target.normalized,
        batch.online.history.action_state,
    )
    query, seed_context = model.bottom.action_and_context(
        physical,
        time,
        cache.history,
        executed_memory=cache.executed_memory,
        action_history_keep=cache.action_history_keep,
    )
    compiled, _ = model.top.compile_policy(
        cache.top,
        p1_fact=model.bottom.complete_p1_fact(
            action_query=query,
            protected_detail=cache.factual_dock.protected_detail,
            time=time,
        )[0],
        p1_precision_innovation=cache.factual_dock.protected_detail,
        action_query=query,
    )
    captured_transition: dict[str, torch.Tensor] = {}

    def capture_transition(_module, _args, kwargs):
        captured_transition["action_tokens"] = kwargs["action_tokens"].detach().clone()

    transition_hook = model.transition.v120_transition.register_forward_pre_hook(
        capture_transition,
        with_kwargs=True,
    )
    try:
        transition, transition_metrics = model.transition(
            source=cache.transition_source,
            action_query=query,
            plan=compiled.plan,
            seed=seed_context,
            collect_diagnostics=True,
        )
    finally:
        transition_hook.remove()
    expected_transition_action = model.transition.trajectory_norm(
        query + compiled.plan.protected_base
    ).flatten(1, 2)
    torch.testing.assert_close(
        captured_transition["action_tokens"],
        expected_transition_action,
        atol=0.0,
        rtol=0.0,
    )
    assert transition_metrics["controlled_transition_action_token_rows"] == (
        config.dimensions.action_horizon * config.dimensions.action_basis_tokens
    )
    contract_inputs: list[tuple[torch.Tensor, torch.Tensor]] = []
    contract_outputs: list[dict[str, torch.Tensor]] = []
    captured_event: dict[str, torch.Tensor] = {}

    def capture_contract_input(_module, args):
        canvas, slices = args
        contract_inputs.append(
            (
                canvas[:, slices["trajectory"]].detach().clone(),
                canvas[:, slices["rollout"]].detach().clone(),
            )
        )

    def capture_contract_output(_module, _args, output):
        contract_outputs.append(output)

    def capture_evidence_input(_module, _args, kwargs):
        captured_event["value"] = kwargs["event_evidence"].detach().clone()

    contract_hooks = []
    for head in model.bottom.layer_contract_heads:
        contract_hooks.append(head.register_forward_pre_hook(capture_contract_input))
        contract_hooks.append(head.register_forward_hook(capture_contract_output))
    evidence_hook = model.bottom.decoder.evidence_adapter.register_forward_pre_hook(
        capture_evidence_input,
        with_kwargs=True,
    )
    try:
        evidence = model.bottom.compile_evidence_view(
            action_query=query,
            p1_fact=compiled.consequence.factual_base,
            plan=compiled.plan,
            intent=cache.top.intent,
            seed=seed_context,
            transition=transition,
        )
    finally:
        evidence_hook.remove()
        for hook in contract_hooks:
            hook.remove()
    assert [head.layer_index for head in model.bottom.layer_contract_heads] == [5, 6]
    assert len(contract_inputs) == len(contract_outputs) == 2
    torch.testing.assert_close(
        contract_inputs[0][0],
        (query + compiled.consequence.factual_base).flatten(1, 2),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        contract_inputs[1][0],
        (query + compiled.plan.protected_base).flatten(1, 2),
        atol=0.0,
        rtol=0.0,
    )
    for _, rollout in contract_inputs:
        torch.testing.assert_close(
            rollout,
            transition.selector,
            atol=0.0,
            rtol=0.0,
        )
    torch.testing.assert_close(
        captured_event["value"],
        contract_outputs[-1]["event_logits"],
        atol=0.0,
        rtol=0.0,
    )
    assert all(
        parameter.requires_grad
        for head in model.bottom.layer_contract_heads
        for parameter in head.adapter.parameters()
    )
    assert not any(
        parameter.requires_grad
        for head in model.bottom.layer_contract_heads
        for parameter in head.readout.parameters()
    )
    trajectory_start, trajectory_stop = evidence.ranges["trajectory"]
    trajectory_projection = model.bottom.decoder.evidence_adapter.source_proj[
        "trajectory"
    ]
    assert not trajectory_projection[0].weight.requires_grad
    assert any(parameter.requires_grad for parameter in trajectory_projection.parameters())
    rollout_start, rollout_stop = evidence.ranges["rollout"]
    assert torch.count_nonzero(
        evidence.tokens[:, rollout_start:rollout_stop]
    ) > 0
    assert torch.count_nonzero(evidence.value_tokens[:, rollout_start:rollout_stop]) > 0
    role_bank = model.bottom._role_bank(compiled.plan)
    assert role_bank.source_names == compiled.plan.source_names
    # The protected factual base is algebraically outside optional routing.
    # Only the four non-zero P3 innovations may compete with the explicit null.
    assert len(role_bank.source_names) == 4
    assert torch.equal(role_bank.protected_detail, compiled.plan.protected_base)
    state_tokens, state_history_tokens, executed_tokens = model.bottom._state_memory(
        seed_context
    )
    intent_memory = model.bottom._intent_memory(
        cache.top.intent,
        state_tokens,
        executed_tokens,
    )
    assert set(intent_memory) == {"state", "executed"}
    start, stop = evidence.ranges["transition"]
    assert torch.count_nonzero(evidence.value_tokens[:, start:stop]) > 0
    alternate_query, alternate_seed = model.bottom.action_and_context(
        torch.zeros_like(physical),
        time,
        cache.history,
        executed_memory=cache.executed_memory,
        action_history_keep=cache.action_history_keep,
    )
    alternate_transition, _ = model.transition(
        source=cache.transition_source,
        action_query=alternate_query,
        plan=compiled.plan,
        seed=alternate_seed,
    )
    assert not torch.equal(transition.action_coefficients, alternate_transition.action_coefficients)
    assert not torch.equal(transition.value, alternate_transition.value)

    anchors = int(model.bottom.core_config.future_anchors)
    spatial = (
        int(model.bottom.core_config.num_cameras)
        * int(model.bottom.core_config.future_grid_size) ** 2
    )
    marker = torch.arange(
        1,
        anchors + 1,
        dtype=transition.value.dtype,
    )[None, :, None, None].expand(1, anchors, spatial, config.dimensions.hidden_size)
    marked_transition = replace(
        transition,
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
    assert len(model.top.grounding_blocks) == 3
    assert model.top.dynamics.w1 is not model.top.dynamics.w2
    assert model.history_proposal.OFFSETS == (-24, -16, -12, -8, -6, -4, -2, -1)
    assert len(model.history_proposal.blocks) == 2
    assert model.history_proposal.recent_tokens == 4
    assert model.history_proposal.summary_tokens == 3
    assert model.bottom.p1_policy_block is not None
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
    factual_calls = 0
    teacher_calls = 0
    grounding_block_calls = [0, 0, 0]
    history_proposal_calls = 0
    p1_host_calls = 0
    transition_calls = 0
    handles = []

    def count_factual(_module, _args, _output):
        nonlocal factual_calls
        factual_calls += 1

    def count_teacher(_module, _args, _output):
        nonlocal teacher_calls
        teacher_calls += 1

    def count_history_proposal(_module, _args, _output):
        nonlocal history_proposal_calls
        history_proposal_calls += 1

    def count_p1_host(_module, _args, _output):
        nonlocal p1_host_calls
        p1_host_calls += 1

    def count_transition(_module, _args, _output):
        nonlocal transition_calls
        transition_calls += 1

    handles.append(model.factual_reader.register_forward_hook(count_factual))
    handles.append(model.top.teacher.register_forward_hook(count_teacher))
    for index, block in enumerate(model.top.grounding_blocks):

        def count_grounding_block(_module, _args, _output, *, index=index):
            grounding_block_calls[index] += 1

        handles.append(block.register_forward_hook(count_grounding_block))
    handles.append(model.history_proposal.register_forward_hook(count_history_proposal))
    handles.append(model.bottom.p1_policy_block.register_forward_hook(count_p1_host))
    handles.append(model.transition.register_forward_hook(count_transition))
    for index, block in enumerate(model.bottom.blocks):

        def count_call(_module, _args, _output, *, index=index):
            calls[index] += 1

        handles.append(block.register_forward_hook(count_call))
    with mock.patch.object(
        model.observation,
        "prepare",
        wraps=model.observation.prepare,
    ) as observation_prepare, mock.patch.object(
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
    assert observation_prepare.call_count == 1
    assert factual_calls == 1
    assert teacher_calls == 0
    assert grounding_block_calls == [1, 1, 1]
    assert history_proposal_calls == 1
    # Five action updates are followed by one complete endpoint forward for
    # event/motion heads.  The endpoint pass must not rebuild static evidence.
    assert p1_host_calls == config.runtime.inference_steps + 1
    assert transition_calls == config.runtime.inference_steps + 1
    # Both V120 correspondence scales batch all adjacent pairs/directions in
    # one invocation and are built once outside the five ODE steps.
    assert semantic_flow.call_count == 1
    assert raw_flow.call_count == 1
    assert tuple(result.step_times.shape) == (5,)
    torch.testing.assert_close(
        result.step_times,
        torch.tensor((0.0, 0.2, 0.4, 0.6, 0.8)),
    )
    torch.testing.assert_close(result.metrics["sampling_endpoint_head_time"], torch.tensor(1.0))
    torch.testing.assert_close(
        result.metrics["sampling_velocity_update_calls"], torch.tensor(5.0)
    )
    torch.testing.assert_close(
        result.metrics["sampling_endpoint_head_calls"], torch.tensor(1.0)
    )
    assert tuple(result.action.shape) == tuple(batch.action_target.normalized.shape)
    assert torch.isfinite(result.action).all()
    # The restored learned V120 execution chart may evaluate several
    # block/dwell candidates inside one ODE step.  Those are dynamic bottom
    # operations; the expensive observation/G/S/W/P1-detail sources above
    # must still be built exactly once.  The compact P1 policy write is a live
    # noisy-action block and therefore executes once per ODE step.
    assert all(value >= config.runtime.inference_steps + 1 for value in calls)
    for handle in handles:
        handle.remove()


def test_clean_endpoint_head_forward_cannot_change_integrated_action() -> None:
    torch.manual_seed(52)
    config = _config()
    model = ClearVLAMainlinePolicy(config).eval()
    batch = _batch(config)
    initial_noise = model.action_codec.sample_noise(
        batch.online.batch,
        device=batch.online.device,
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(71),
    )
    baseline = sample_action(
        model,
        batch.online,
        config,
        initial_physical_noise=initial_noise,
        dtype=torch.float32,
    )
    original_velocity = model.velocity

    def poisoned_endpoint(*args, **kwargs):
        output = original_velocity(*args, **kwargs)
        time = kwargs["time"]
        if bool((time == 1.0).all()):
            output = replace(
                output,
                bottom=replace(
                    output.bottom,
                    physical_velocity=torch.full_like(
                        output.bottom.physical_velocity, 1.0e6
                    ),
                ),
            )
        return output

    with mock.patch.object(model, "velocity", side_effect=poisoned_endpoint):
        endpoint_poisoned = sample_action(
            model,
            batch.online,
            config,
            initial_physical_noise=initial_noise,
            dtype=torch.float32,
        )
    torch.testing.assert_close(endpoint_poisoned.physical_field, baseline.physical_field)
    torch.testing.assert_close(endpoint_poisoned.action, baseline.action)


def test_p1_refines_the_local_chart_per_query_and_returns_action_pressure_to_g() -> None:
    torch.manual_seed(51)
    config = _config()
    model = ClearVLAMainlinePolicy(config).eval()
    batch = _batch(config)
    captured: dict[str, torch.Tensor] = {}

    def capture_g3(_module, args):
        captured["g3_rollout"] = args[1]

    handle = model.factual_reader.register_forward_pre_hook(capture_g3)
    cache, training_state, metrics = model.encode_online(
        batch.online,
        geometry_supervision=False,
        collect_diagnostics=True,
    )
    handle.remove()
    assert tuple(cache.factual_dock.protected_detail.shape[1:3]) == (
        24,
        config.dimensions.action_basis_tokens,
    )
    assert metrics["flow_jepa_p1_query_rows"] == 24
    assert metrics["flow_jepa_p1_shared_factual"] == 1
    assert metrics["flow_jepa_typed_p1_spatial_variation"] > 0
    progressive = training_state.observation.progressive_state
    assert progressive.dynamic_fine_values is not None
    assert progressive.dynamic_fine_coordinates is not None
    assert int(progressive.dynamic_fine_values.shape[-2]) == 49
    assert int(progressive.dynamic_fine_coordinates.shape[-2]) == 49
    g_gradient = torch.autograd.grad(
        cache.factual_dock.protected_detail.square().sum(),
        progressive.dynamic_fine_values,
        retain_graph=True,
        allow_unused=True,
    )[0]
    assert g_gradient is not None and torch.count_nonzero(g_gradient) > 0
    assert tuple(cache.transition_source.selector.shape[1:]) == (
        4 * config.dimensions.num_cameras * 8 * 8,
        config.dimensions.hidden_size,
    )
    assert metrics["controlled_transition_source_spatial_variation"] >= 0
    assert metrics["controlled_transition_source_anchor_variation"] >= 0
    assert torch.equal(
        cache.transition_source.selector,
        captured["g3_rollout"],
    )
    assert not hasattr(model.transition, "interval_identity")
    assert cache.transition_source.selector.shape[1] == (
        4 * config.dimensions.num_cameras * 8 * 8
    )


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


def test_validation_execution_interventions_match_the_native_v120_modes() -> None:
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
        outputs = {
            mode: model.velocity(
                cache,
                noisy_action_field=physical,
                time=time,
                execution_mode=mode,
                collect_diagnostics=True,
            )
            for mode in (
                "no_updates",
                "hard",
                "neutral",
                "full_capacity",
                "three_basis_reduction",
            )
        }
    no_updates = outputs["no_updates"]
    # V120 capacity is rank retention, not block amplitude.  The no-update
    # intervention therefore selects prefix row zero instead of pretending
    # that capacity=0 disables the host operation.
    assert no_updates.metrics["evidence_mmd_it_capacity_ratio"] == 1
    assert no_updates.metrics["evidence_mmd_it_execution_eval_policy_code"] == 2
    assert outputs["hard"].metrics["evidence_mmd_it_execution_eval_policy_code"] == 1
    assert outputs["neutral"].metrics["evidence_mmd_it_execution_eval_policy_code"] == 2
    assert outputs["full_capacity"].metrics[
        "evidence_mmd_it_execution_eval_policy_code"
    ] == 0
    assert outputs["three_basis_reduction"].metrics[
        "evidence_mmd_it_execution_eval_policy_code"
    ] == 0
    assert outputs["neutral"].metrics["evidence_mmd_it_capacity_ratio"] == 1
    assert outputs["full_capacity"].metrics["evidence_mmd_it_capacity_ratio"] == 1
    expected_reduction = max(config.bottom.operator_rank - 3, 1) / float(
        config.bottom.operator_rank
    )
    torch.testing.assert_close(
        outputs["three_basis_reduction"].metrics[
            "evidence_mmd_it_capacity_ratio"
        ],
        torch.tensor(expected_reduction),
    )
    assert not torch.equal(
        outputs["hard"].bottom.physical_velocity,
        outputs["full_capacity"].bottom.physical_velocity,
    )
    torch.testing.assert_close(
        no_updates.bottom.physical_velocity,
        no_updates.bottom.decoder_tensors[
            "evidence_mmd_it_prefix_pred_velocity"
        ][:, 0],
    )
    assert no_updates.metrics["bottom_execution_output_block_count"] == 0
    for mode in ("hard", "neutral", "full_capacity", "three_basis_reduction"):
        assert outputs[mode].metrics["bottom_execution_output_block_count"] == 3


def test_inactive_proposal_path_has_no_deployment_ablation_api() -> None:
    model = ClearVLAMainlinePolicy(_config()).eval()
    assert not hasattr(model, "proposal_ablation_cache")


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

    # A learned interval-address template is not observable progress.  If the
    # exact supervised condition innovation is zero, its audit centroid must
    # be the legal zero result even while the public carrier remains nonzero.
    zero_intent = replace(
        training_state.top.intent,
        interval_condition_innovation=torch.zeros_like(
            training_state.top.intent.interval_condition_innovation
        ),
    )
    zero_state = replace(
        training_state,
        top=replace(training_state.top, intent=zero_intent),
    )
    zero_metrics = MainlineTrainingEngine._audit_progress_metrics(
        batch,
        EncodedTrainingBatch(cache=cache, training_state=zero_state, metrics=metrics),
    )
    assert float(zero_metrics["object_intent_audit_interval_energy_centroid"]) == 0.0


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
