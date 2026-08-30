from __future__ import annotations

import copy
import inspect
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as F

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
from clearvla.mainline.runtime.sampling import (
    sample_action,
    sample_cached_action,
    sample_refined_cached_action,
)
from clearvla.mainline.train import _optimizer_group_context
from clearvla.mainline.training.engine import (
    EncodedTrainingBatch,
    MainlineTrainingEngine,
    NonFiniteGradientError,
    validate_finite_training_batch,
)
from clearvla.mainline.training.losses import LossLedger, anchored_gripper_persistence
from clearvla.mainline.training.optimizer import (
    WarmupCosineSchedule,
    build_optimizer,
    role_lr_scale,
)
from clearvla.mainline.v120_core.layer_contracts import LayerContractAdapterHeads


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
        "plan_recognizer",
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
    assert result.metrics["gradient_postglobal_global_l2"] <= 1.0001
    assert "gradient_observation_l2" not in result.metrics
    assert "gradient_global_preclip_l2" in result.materialize()
    for name in (
        "gradient_tensor_s_public_interval_carrier_rms",
        "gradient_tensor_s_typed_common_rms",
        "gradient_tensor_s_typed_interval_residual_rms",
        "gradient_tensor_p1_static_fact_rms",
        "gradient_tensor_p1_dynamic_query_residual_rms",
        "gradient_tensor_w2_semantic_common_rms",
        "gradient_tensor_w2_geometry_interval_rms",
            "gradient_tensor_w_semantic_fact_ingress_rms",
            "gradient_tensor_w_appearance_fact_ingress_rms",
            "gradient_tensor_w_geometry_fact_ingress_rms",
            "gradient_tensor_w_physical_action_condition_rms",
        "gradient_tensor_p2_semantic_effect_rms",
        "gradient_tensor_p2_geometry_effect_rms",
        "gradient_tensor_p2_geometry_address_correction_rms",
        "gradient_tensor_p1_protected_policy_precision_rms",
        "gradient_tensor_p3_temporal_rms",
        "gradient_tensor_p3_state_change_rms",
        "gradient_parameter_w_transport_head_weight_rms",
        "gradient_parameter_p2_semantic_spatial_query_weight_rms",
        "gradient_parameter_p2_geometry_spatial_query_weight_rms",
        "gradient_parameter_p2_semantic_terminal_query_weight_rms",
        "gradient_parameter_p2_geometry_terminal_query_weight_rms",
        "gradient_parameter_p2_semantic_value_weight_rms",
        "gradient_parameter_p2_geometry_value_weight_rms",
        "gradient_parameter_consequence_semantic_interaction_weight_rms",
        "gradient_parameter_consequence_geometry_interaction_weight_rms",
        "gradient_parameter_gripper_private_gate_weight_rms",
    ):
        assert name in result.metrics
        assert torch.isfinite(result.metrics[name])
    assert result.metrics["object_w_typed_norm_denominator_min"] >= 0.25
    assert result.metrics["object_w_typed_norm_gain_max"] <= 4.000001
    assert (
        result.metrics["object_w_typed_norm_output_input_rms_ratio_max"]
        <= 4.000001
    )
    assert result.metrics["gradient_tensor_p2_semantic_effect_rms"] > 0
    assert result.metrics["gradient_tensor_p2_geometry_effect_rms"] > 0
    # Parameter hooks survive their forward graph. R2 parameter-gradient
    # diagnostics must therefore read .grad in the engine rather than stacking
    # a new persistent hook on every diagnostic batch.
    for parameter_name in (
        "top.dynamics.transport_head.weight",
        "top.effect_reader.source_query.0.weight",
        "top.effect_reader.source_query.1.weight",
        "top.effect_reader.terminal_query.0.weight",
        "top.effect_reader.terminal_query.1.weight",
        "top.effect_reader.semantic_value.weight",
        "top.effect_reader.transport_value.weight",
        "top.consequence.semantic_interaction.weight",
        "top.consequence.geometry_interaction.weight",
        "bottom.decoder.velocity_head.gripper_gate.weight",
    ):
        parameter = dict(model.named_parameters())[parameter_name]
        assert parameter._backward_hooks is None or not parameter._backward_hooks
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
        "object_p3_": 5,
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


def test_finite_spike_audit_is_preclip_read_only_and_skips_ordinary_scan() -> None:
    class TinyOwnerModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.observation = torch.nn.Linear(2, 2, bias=False)

        def set_training_step(self, step: int) -> None:
            del step

    def build_engine(*, threshold: float | None):
        config = _config()
        model = TinyOwnerModel()
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": (model.observation.weight,),
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
            gradient_spike_audit_threshold=threshold,
        )
        return model, engine

    torch.manual_seed(410)
    observed_model, observed_engine = build_engine(threshold=0.01)
    reference_model, reference_engine = build_engine(threshold=None)
    reference_model.load_state_dict(observed_model.state_dict())

    def ledger(model: TinyOwnerModel) -> LossLedger:
        loss = model.observation.weight.square().sum()
        return LossLedger(
            total=loss,
            groups={
                "action": loss,
                "representation": loss.new_zeros(()),
                "execution": loss.new_zeros(()),
            },
            contributions={"action_flow": loss},
            terms={"action_flow": loss},
        )

    reports = []
    with mock.patch.object(
        observed_engine,
        "_forward",
        side_effect=lambda *args, **kwargs: (ledger(observed_model), {}),
    ):
        observed_result = observed_engine.train_step(
            object(),  # type: ignore[arg-type]
            gradient_spike_handler=reports.append,
        )
    with mock.patch.object(
        reference_engine,
        "_forward",
        side_effect=lambda *args, **kwargs: (ledger(reference_model), {}),
    ):
        reference_result = reference_engine.train_step(object())  # type: ignore[arg-type]
    assert len(reports) == 1
    assert reports[0].max_l2.parameter_name == "observation.weight"
    assert reports[0].gradient_global_preclip_l2 == observed_result.gradient_norm_scalar
    assert observed_result.gradient_norm_scalar == reference_result.gradient_norm_scalar
    torch.testing.assert_close(
        observed_model.observation.weight,
        reference_model.observation.weight,
        rtol=0.0,
        atol=0.0,
    )

    ordinary_model, ordinary_engine = build_engine(threshold=1.0e9)
    with mock.patch.object(
        ordinary_engine,
        "_forward",
        side_effect=lambda *args, **kwargs: (ledger(ordinary_model), {}),
    ), mock.patch(
        "clearvla.mainline.training.engine.build_finite_gradient_spike_report"
    ) as scanner:
        ordinary_engine.train_step(
            object(),  # type: ignore[arg-type]
            gradient_spike_handler=lambda report: None,
        )
    scanner.assert_not_called()


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
    context = _optimizer_group_context(optimizer, config)
    history = context["history_proposal/decay"]
    assert history["base_learning_rate"] == base
    assert history["initial_learning_rate"] == base * 0.625
    assert history["role_learning_rate_scale"] == 0.625
    assert history["parameter_tensor_count"] > 0
    assert history["parameter_count"] > history["parameter_tensor_count"]


def test_gripper_private_state_is_exact_zero_and_local_to_deployed_heads() -> None:
    torch.manual_seed(43)
    decoder = ClearVLAMainlinePolicy(_config()).bottom.decoder
    head = decoder.velocity_head
    assert head.arm_abs is not None and head.arm_delta is not None
    assert head.grip_value is not None and head.grip_delta is not None
    assert head.grip_extra is not None and head.grip_native is None
    tokens = torch.randn(2, 24, 32)
    base_read = head.norm(tokens)
    expected = torch.cat(
        (
            head.arm_abs(base_read),
            head.arm_delta(base_read),
            head.grip_value(base_read),
            head.grip_delta(base_read),
            head.grip_extra(base_read),
        ),
        dim=-1,
    )
    field, gripper_state, gate = head.forward_with_gripper_state(tokens)
    assert torch.equal(field, expected)
    assert torch.equal(gripper_state, tokens)
    assert torch.count_nonzero(gate) == 0

    with torch.no_grad():
        head.gripper_gate.weight.copy_(0.25 * torch.eye(32))
    changed, changed_state, changed_gate = head.forward_with_gripper_state(tokens)
    assert torch.equal(changed[..., :12], expected[..., :12])
    assert torch.equal(changed[..., 14:], expected[..., 14:])
    assert not torch.equal(changed[..., 12:14], expected[..., 12:14])
    assert not torch.equal(changed_state, tokens)
    assert float(changed_gate.detach().abs().amax()) < 1.0

    parseval_head = type(head)(
        replace(decoder.config, gripper_field_mode="parseval_temporal")
    )
    assert parseval_head.arm_abs is not None and parseval_head.arm_delta is not None
    assert parseval_head.grip_native is not None
    parseval_base = parseval_head.norm(tokens)
    parseval_expected = torch.cat(
        (
            parseval_head.arm_abs(parseval_base),
            parseval_head.arm_delta(parseval_base),
            parseval_head.codec.encode_gripper_tangent(
                parseval_head.grip_native(parseval_base)
            ),
        ),
        dim=-1,
    )
    parseval_field, parseval_state, parseval_gate = (
        parseval_head.forward_with_gripper_state(tokens)
    )
    assert torch.equal(parseval_field, parseval_expected)
    assert torch.equal(parseval_state, tokens)
    assert torch.count_nonzero(parseval_gate) == 0
    with torch.no_grad():
        parseval_head.gripper_gate.weight.copy_(0.25 * torch.eye(32))
    parseval_changed, _, _ = parseval_head.forward_with_gripper_state(tokens)
    assert torch.equal(parseval_changed[..., :12], parseval_expected[..., :12])
    assert not torch.equal(parseval_changed[..., 12:], parseval_expected[..., 12:])


def test_continuous_gripper_trajectory_reads_only_value_and_delta_channels() -> None:
    torch.manual_seed(44)
    model = ClearVLAMainlinePolicy(_config()).train()
    decoder = model.bottom.decoder
    assert decoder.event_head is None
    assert not hasattr(model, "decoded_gripper_event_head")
    batch = 2
    physical = torch.randn(
        batch,
        model.config.dimensions.action_horizon,
        model.action_codec.physical_dim,
        requires_grad=True,
    )
    parts = model.action_codec.split(physical)
    absolute = parts.gripper_field[..., :1]
    event = torch.zeros(batch, model.config.dimensions.action_horizon)
    event[:, 0] = 1.0
    cumulative = anchored_gripper_persistence(
        absolute,
        parts.gripper_field[..., 1:2],
        event,
    )
    target = torch.randn_like(absolute)
    loss = 0.5 * (
        torch.nn.functional.smooth_l1_loss(absolute, target)
        + torch.nn.functional.smooth_l1_loss(cumulative, target)
    )

    # Arm plus the four auxiliary gripper coordinates are outside the two
    # continuous deployed gripper trajectories.
    loss.backward()
    assert physical.grad is not None
    assert torch.count_nonzero(physical.grad[..., :12]) == 0
    assert torch.count_nonzero(physical.grad[..., 14:]) == 0
    assert torch.count_nonzero(physical.grad[..., 12:14]) > 0



def test_gripper_head_diagnostics_are_separate_from_execution_diagnostics() -> None:
    torch.manual_seed(45)
    decoder = ClearVLAMainlinePolicy(_config()).bottom.decoder.train()
    action = torch.randn(2, 24, 32)
    _, event_logits, _, quiet_metrics = decoder._read_output_heads(
        action,
        collect_diagnostics=True,
        collect_gripper_diagnostics=False,
    )
    assert event_logits is None
    assert quiet_metrics == {}
    _, event_logits, _, captured_metrics = decoder._read_output_heads(
        action,
        collect_diagnostics=True,
        collect_gripper_diagnostics=True,
    )
    assert event_logits is None
    assert "gripper_private_gate_tensor" in captured_metrics
    assert "gripper_private_state_tensor" in captured_metrics
    assert "gripper_private_state_delta_tensor" in captured_metrics
    assert "gradient_tensor_gripper_private_state_rms" in captured_metrics


def test_continuous_gripper_trajectory_loss_reaches_the_private_gate() -> None:
    torch.manual_seed(46)
    model = ClearVLAMainlinePolicy(_config()).train()
    head = model.bottom.decoder.velocity_head
    tokens = torch.randn(2, 24, 32, requires_grad=True)
    physical, _, _ = head.forward_with_gripper_state(tokens)
    parts = model.action_codec.split(physical)
    absolute = parts.gripper_field[..., :1]
    event = torch.zeros(2, model.config.dimensions.action_horizon)
    event[:, 0] = 1.0
    cumulative = anchored_gripper_persistence(
        absolute,
        parts.gripper_field[..., 1:2],
        event,
    )
    target = torch.randn_like(absolute)
    (
        0.5
        * (
            torch.nn.functional.smooth_l1_loss(absolute, target)
            + torch.nn.functional.smooth_l1_loss(cumulative, target)
        )
    ).backward()
    gate_gradient = head.gripper_gate.weight.grad
    assert gate_gradient is not None
    assert torch.isfinite(gate_gradient).all()
    assert torch.count_nonzero(gate_gradient) > 0


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


def test_capacity_stays_fp32_below_one_and_reaches_its_ordered_bank() -> None:
    torch.manual_seed(461)
    model = ClearVLAMainlinePolicy(_config()).train()
    decoder = model.bottom.decoder
    controller = decoder.execution_controller
    assert controller is not None
    with torch.no_grad():
        controller.capacity_head.weight.normal_(mean=0.0, std=0.01)
        controller.capacity_head.bias.fill_(6.5)
        decoder.execution_progress.fill_(1.0)
    context = torch.randn(
        2,
        len(decoder.blocks),
        controller.capacity_head.in_features,
        requires_grad=True,
    )
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        learned = controller._capacity_ratios(context)
    reference = torch.sigmoid(
        F.linear(
            context.float(),
            controller.capacity_head.weight.float(),
            controller.capacity_head.bias.float(),
        )
    ).squeeze(-1)
    assert learned.dtype == torch.float32
    torch.testing.assert_close(learned, reference, atol=0.0, rtol=0.0)
    assert bool((learned < 1.0).all())
    assert bool((learned > 0.99).all())

    effective = decoder._execution_capacity(learned)
    assert effective.dtype == torch.float32
    torch.testing.assert_close(effective, learned, atol=0.0, rtol=0.0)
    source = inspect.getsource(type(decoder)._apply_native_operation)
    assert ").to(dtype=action.dtype)" not in source

    bank = decoder.operator_contractions[0]
    base_update = torch.randn(2, 5, model.config.dimensions.hidden_size)
    condition = torch.randn(2, model.config.dimensions.hidden_size)
    contracted, _ = bank(
        base_update,
        condition,
        torch.zeros(2, dtype=torch.long),
        depth_ratio_override=effective[:, 0],
        identity_bypass=False,
        collect_diagnostics=False,
    )
    gradients = torch.autograd.grad(
        contracted.float().square().mean(),
        (
            controller.capacity_head.weight,
            controller.capacity_head.bias,
            bank.basis_raw,
        ),
    )
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0


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
    grounded = progressive.grounded_fact_set
    for probability, log_probability in (
        (grounded.semantic_owner_probs, grounded.semantic_owner_log_probs),
        (grounded.appearance_owner_probs, grounded.appearance_owner_log_probs),
        (grounded.geometry_owner_probs, grounded.geometry_owner_log_probs),
    ):
        assert probability.dtype == torch.float32
        assert log_probability is not None
        assert log_probability.dtype == torch.float32
        assert torch.isfinite(log_probability).all()
        torch.testing.assert_close(probability, log_probability.exp())
    assert progressive.dynamic_fine_values is not None
    assert int(progressive.dynamic_fine_values.shape[-2]) == 49
    assert metrics["observation_g1_g2_g3_completed"] == 1
    for name in (
        "flow_jepa_address_coarse_variance_min",
        "flow_jepa_address_coarse_std_dino_rms",
        "flow_jepa_address_coarse_std_gain_max",
        "flow_jepa_progressive_g2_input_variance_min",
        "flow_jepa_progressive_g2_input_std_rms",
        "flow_jepa_progressive_g2_input_std_gain_max",
        "flow_jepa_progressive_g2_aligned_variance_min",
        "flow_jepa_progressive_g2_correction_scale_min",
        "flow_jepa_progressive_g2_correction_std_gain_max",
    ):
        assert name in metrics
        assert torch.isfinite(metrics[name])
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
    assert set(first.__dataclass_fields__) == {
        "factual_base",
        "policy_query_residual",
    }
    assert first.factual_base is cache.factual_dock.protected_detail
    assert second.factual_base is cache.factual_dock.protected_detail
    torch.testing.assert_close(
        first.factual_base,
        cache.factual_dock.protected_detail,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        second.factual_base,
        cache.factual_dock.protected_detail,
        atol=0.0,
        rtol=0.0,
    )
    assert not torch.equal(
        first.policy_query_residual,
        second.policy_query_residual,
    )
    torch.testing.assert_close(
        first.p2_dock(query).combined(),
        query + first.factual_base + first.policy_query_residual,
        atol=0.0,
        rtol=0.0,
    )
    assert first_metrics["p1_protected_detail_rms"] > 0
    assert first_metrics["p1_dynamic_delta_rms"] > 0
    assert first_metrics["p1_completed_fact_rms"] > 0
    assert first_metrics["p1_factual_base_rms"] > 0
    assert first_metrics["p1_policy_query_residual_rms"] > 0


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
    p1_state, _ = model.bottom.complete_p1_fact(
        action_query=query,
        protected_detail=cache.factual_dock.protected_detail,
        time=time,
    )
    compiled, _ = model.top.compile_policy(
        cache.top,
        p1_state=p1_state,
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
        query
        + compiled.plan.protected_base
        + compiled.plan.protected_policy_precision
    ).flatten(1, 2)
    torch.testing.assert_close(
        captured_transition["action_tokens"],
        expected_transition_action,
        atol=0.0,
        rtol=0.0,
    )
    transition_policy_gradient = torch.autograd.grad(
        transition.value.square().mean(),
        p1_state.policy_query_residual,
        retain_graph=True,
    )[0]
    assert torch.isfinite(transition_policy_gradient).all()
    assert torch.count_nonzero(transition_policy_gradient) > 0
    assert transition_metrics["controlled_transition_action_token_rows"] == (
        config.dimensions.action_horizon * config.dimensions.action_basis_tokens
    )
    assert "p1_fact" not in inspect.signature(model.bottom.forward).parameters
    assert "action_query" not in inspect.signature(
        model.bottom._layer_contracts
    ).parameters
    assert "p1_fact" not in inspect.signature(model.bottom._layer_contracts).parameters
    assert "plan" not in inspect.signature(model.bottom._layer_contracts).parameters
    contract_inputs: list[tuple[torch.Tensor, dict[str, slice]]] = []
    contract_outputs: list[dict[str, torch.Tensor]] = []
    captured_event: dict[str, torch.Tensor] = {}

    def capture_contract_input(_module, args):
        canvas, slices = args
        contract_inputs.append((canvas.detach().clone(), dict(slices)))

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
            plan=compiled.plan,
            intent=cache.top.intent,
            seed=seed_context,
            transition=transition,
        )
    finally:
        evidence_hook.remove()
        for hook in contract_hooks:
            hook.remove()
    contract_heads: list[LayerContractAdapterHeads] = []
    for module in model.bottom.layer_contract_heads:
        assert isinstance(module, LayerContractAdapterHeads)
        contract_heads.append(module)
    assert [head.layer_index for head in contract_heads] == [5, 6]
    assert len(contract_inputs) == len(contract_outputs) == 2
    retained_contract_keys = (
        "rollout_tokens",
        "state_tokens",
        "state_history_tokens",
        "event_logits",
    )
    removed_trajectory_keys = (
        "trajectory_tokens",
        "trajectory_pooled",
        "pred_physical_velocity",
        "direct_physical_velocity",
        "rollout_residual_velocity",
        "rollout_alpha",
        "motion_logits",
    )
    for canvas, slices in contract_inputs:
        assert slices["trajectory"].start == slices["trajectory"].stop
        torch.testing.assert_close(
            canvas[:, slices["rollout"]],
            transition.selector,
            atol=0.0,
            rtol=0.0,
        )
    for output in contract_outputs:
        assert not set(removed_trajectory_keys).intersection(output)
    torch.testing.assert_close(
        captured_event["value"],
        contract_outputs[-1]["event_logits"],
        atol=0.0,
        rtol=0.0,
    )
    assert all(
        parameter.requires_grad
        for head in contract_heads
        for parameter in head.adapter.parameters()
    )
    assert not any(
        parameter.requires_grad
        for head in contract_heads
        for parameter in head.readout.parameters()
    )
    for head in contract_heads:
        assert not hasattr(head.readout, "action_head")
        assert not hasattr(head.readout, "motion_head")
    contract_terms = tuple(
        output[name].square().mean()
        for output in contract_outputs
        for name in retained_contract_keys
    )
    contract_loss = torch.stack(contract_terms).sum()
    adapter_parameters = tuple(
        parameter
        for head in contract_heads
        for parameter in head.adapter.parameters()
    )
    adapter_gradients = torch.autograd.grad(
        contract_loss,
        adapter_parameters,
        retain_graph=True,
    )
    assert len(adapter_gradients) == 12
    for gradient in adapter_gradients:
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0

    for head, (canvas, slices), reference in zip(
        contract_heads,
        contract_inputs,
        contract_outputs,
        strict=True,
    ):
        insertion = int(slices["trajectory"].start)
        fake_trajectory = torch.randn(
            int(canvas.shape[0]),
            config.dimensions.action_horizon
            * config.dimensions.action_basis_tokens,
            int(canvas.shape[-1]),
            device=canvas.device,
            dtype=canvas.dtype,
        )
        expanded_canvas = torch.cat(
            (canvas[:, :insertion], fake_trajectory, canvas[:, insertion:]),
            dim=1,
        )
        shift = int(fake_trajectory.shape[1])
        expanded_slices = dict(slices)
        expanded_slices["trajectory"] = slice(insertion, insertion + shift)
        for name in ("rollout", "registers"):
            current = slices[name]
            expanded_slices[name] = slice(
                int(current.start) + shift,
                int(current.stop) + shift,
            )
            intervened = head(expanded_canvas, expanded_slices)
            for name in retained_contract_keys:
                torch.testing.assert_close(
                    reference[name],
                    intervened[name],
                    atol=2e-8,
                    rtol=2e-7,
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
    assert role_bank.source_names == ("p3_temporal", "p3_state_change")
    protected_detail = role_bank.protected_detail
    protected_policy_precision = role_bank.protected_policy_precision
    assert protected_detail is not None
    assert protected_policy_precision is not None
    assert protected_policy_precision is compiled.plan.protected_policy_precision
    assert torch.equal(protected_detail, compiled.plan.protected_base)
    assert torch.equal(
        protected_policy_precision,
        p1_state.policy_query_residual,
    )
    assert model.bottom.decoder.protected_detail_basis_attnres is not None
    assert not model.bottom.decoder.protected_detail_basis_attnres.include_null
    optional_reader = model.bottom.decoder.policy_delta_attnres
    assert optional_reader is not None
    basis = config.dimensions.action_basis_tokens
    assert optional_reader.max_sources == basis
    assert tuple(optional_reader.source_key.shape) == (
        basis,
        optional_reader.route_dim,
    )
    bottom_query = torch.randn(
        1,
        config.dimensions.action_horizon,
        config.dimensions.hidden_size,
    )
    isolated_dynamic_bank = replace(
        role_bank,
        values=torch.zeros_like(role_bank.values),
        protected_detail=torch.zeros_like(protected_detail),
    )
    dynamic_basis_read, _ = (
        model.bottom.decoder.protected_detail_basis_attnres(
            bottom_query,
            protected_policy_precision,
            collect_diagnostics=False,
        )
    )
    optional_update, consequence_update, _ = (
        model.bottom.decoder._read_policy_delta_bank(
            bottom_query,
            isolated_dynamic_bank,
            collect_diagnostics=False,
        )
    )
    expected_optional = (
        float(model.bottom.core_config.role_attnres_policy_to_mmdit_scale)
        * dynamic_basis_read
    )
    torch.testing.assert_close(
        optional_update,
        expected_optional,
        atol=1e-7,
        rtol=1e-7,
    )
    bottom_policy_gradient = torch.autograd.grad(
        optional_update.square().mean(),
        p1_state.policy_query_residual,
        retain_graph=True,
    )[0]
    assert torch.isfinite(bottom_policy_gradient).all()
    assert torch.count_nonzero(bottom_policy_gradient) > 0
    assert torch.count_nonzero(consequence_update) == 0
    zero_dynamic_bank = replace(
        isolated_dynamic_bank,
        protected_policy_precision=torch.zeros_like(protected_policy_precision),
    )
    zero_optional, zero_consequence, _ = (
        model.bottom.decoder._read_policy_delta_bank(
            bottom_query,
            zero_dynamic_bank,
            collect_diagnostics=False,
        )
    )
    assert torch.count_nonzero(zero_optional) == 0
    assert torch.count_nonzero(zero_consequence) == 0

    lane_bank = replace(
        role_bank,
        protected_detail=torch.zeros_like(protected_detail),
        protected_policy_precision=torch.zeros_like(protected_policy_precision),
    )
    lane_update, lane_consequence, lane_metrics = (
        model.bottom.decoder._read_policy_delta_bank(
            bottom_query,
            lane_bank,
            collect_diagnostics=True,
        )
    )
    temporal_read, _ = optional_reader(
        bottom_query,
        lane_bank.values[:, 0],
        collect_diagnostics=False,
    )
    state_change_read, _ = optional_reader(
        bottom_query,
        lane_bank.values[:, 1],
        collect_diagnostics=False,
    )
    expected_lane_update = (
        float(model.bottom.core_config.role_attnres_policy_to_mmdit_scale)
        * (temporal_read + state_change_read)
    )
    torch.testing.assert_close(lane_update, expected_lane_update)
    assert torch.count_nonzero(lane_consequence) == 0
    lane_gradient = torch.autograd.grad(
        lane_update.square().mean(),
        lane_bank.values,
        retain_graph=True,
    )[0]
    assert torch.isfinite(lane_gradient).all()
    for lane_index in range(len(lane_bank.source_names)):
        assert torch.count_nonzero(lane_gradient[:, lane_index]) > 0
    changed_values = lane_bank.values.clone()
    changed_values[:, 0] = 1000.0 * torch.randn_like(changed_values[:, 0])
    _, _, changed_metrics = model.bottom.decoder._read_policy_delta_bank(
        bottom_query,
        replace(lane_bank, values=changed_values),
        collect_diagnostics=True,
    )
    torch.testing.assert_close(
        lane_metrics[
            "evidence_policy_delta_attnres_null_mass_p3_state_change"
        ],
        changed_metrics[
            "evidence_policy_delta_attnres_null_mass_p3_state_change"
        ],
        atol=0.0,
        rtol=0.0,
    )
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
            collect_diagnostics=True,
            dtype=torch.float32,
        )
    assert observation_prepare.call_count == 1
    assert factual_calls == 1
    assert teacher_calls == 0
    assert grounding_block_calls == [1, 1, 1]
    assert history_proposal_calls == 1
    # Schema28 performs one proposal pass, rematerializes W once, then runs
    # the same five-update + endpoint chart for the refined candidate.  Static
    # observation/G/S/P1-detail evidence is still built exactly once.
    expected_dynamic_calls = 2 * (config.runtime.inference_steps + 1)
    assert p1_host_calls == expected_dynamic_calls
    assert transition_calls == expected_dynamic_calls
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
    assert not hasattr(result, "event_logits")
    assert tuple(result.motion_logits.shape) == (1, 24)
    # The restored learned V120 execution chart may evaluate several
    # block/dwell candidates inside one ODE step.  Those are dynamic bottom
    # operations; the expensive observation/G/S/W/P1-detail sources above
    # must still be built exactly once.  The compact P1 policy write is a live
    # noisy-action block and therefore executes once per ODE step/pass.
    assert all(value >= expected_dynamic_calls for value in calls)
    assert result.metrics["sampling_outer_world_refinement"] == 1
    closure_metrics = (
        "object_action_world_refinement_count",
        "object_action_world_refinement_pre_action_interval_rms",
        "object_action_world_refinement_post_action_interval_rms",
        "object_action_world_refinement_action_interval_delta_rms",
        "object_action_world_refinement_pre_semantic_delta_rms",
        "object_action_world_refinement_post_semantic_delta_rms",
        "object_action_world_refinement_semantic_delta_change_rms",
        "object_action_world_refinement_pre_transport_rms",
        "object_action_world_refinement_post_transport_rms",
        "object_action_world_refinement_transport_change_rms",
        "object_action_world_refinement_tag_identity_error",
        "sampling_outer_final_world_action_interval_mismatch_rms",
        "sampling_outer_final_world_action_delta_mismatch_rms",
    )
    assert all(name in result.metrics for name in closure_metrics)
    assert all(torch.isfinite(result.metrics[name]) for name in closure_metrics)
    # One bounded outer correction intentionally does not claim a fixed point;
    # the final residual is observed, not asserted to be zero.
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

    def capture_p1_g3_rollout(_module, args):
        captured["g3_rollout"] = args[1]

    handle = model.factual_reader.register_forward_pre_hook(
        capture_p1_g3_rollout
    )
    try:
        cache, training_state, metrics = model.encode_online(
            batch.online,
            geometry_supervision=False,
            collect_diagnostics=True,
        )
    finally:
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
    g3_rollout = captured["g3_rollout"]
    assert cache.transition_source.selector is g3_rollout
    cotangent = torch.zeros_like(cache.transition_source.selector)
    sentinel_row = 3 * config.dimensions.num_cameras * 8 * 8 + 67
    cotangent[:, sentinel_row, 5] = 1.0
    direct_gradient = torch.autograd.grad(
        cache.transition_source.selector,
        g3_rollout,
        grad_outputs=cotangent,
        retain_graph=True,
    )[0]
    torch.testing.assert_close(
        direct_gradient,
        cotangent,
        atol=0.0,
        rtol=0.0,
    )
    noisy_action = model.action_codec.encode(
        batch.action_target.normalized,
        batch.online.history.action_state,
    )
    dynamic_output = model.velocity(
        cache,
        noisy_action_field=noisy_action,
        time=torch.full((batch.action_target.batch,), 0.5),
    )
    transition_gradient = torch.autograd.grad(
        dynamic_output.bottom.evidence_tokens.square().sum(),
        g3_rollout,
        retain_graph=True,
        allow_unused=True,
    )[0]
    assert transition_gradient is not None
    assert torch.isfinite(transition_gradient).all()
    assert torch.count_nonzero(transition_gradient) > 0
    assert metrics["controlled_transition_source_spatial_variation"] >= 0
    assert metrics["controlled_transition_source_anchor_variation"] >= 0
    assert not hasattr(model.transition, "interval_identity")
    assert cache.transition_source.selector.shape[1] == (
        4 * config.dimensions.num_cameras * 8 * 8
    )


def test_controlled_transition_source_preserves_g3_rows_and_exact_zero() -> None:
    torch.manual_seed(52)
    config = _config()
    transition = ClearVLAMainlinePolicy(config).transition
    rows = 4 * config.dimensions.num_cameras * 8 * 8
    hidden = config.dimensions.hidden_size
    sentinel_row = 2 * config.dimensions.num_cameras * 8 * 8 + 19
    g3_rollout = torch.zeros(1, rows, hidden)
    g3_rollout[0, sentinel_row] = torch.arange(hidden, dtype=g3_rollout.dtype)
    g3_rollout.requires_grad_(True)

    source, metrics = transition.build_source(
        g3_rollout=g3_rollout,
        collect_diagnostics=True,
    )

    assert source.selector is g3_rollout
    torch.testing.assert_close(
        source.selector[0, sentinel_row],
        torch.arange(hidden, dtype=g3_rollout.dtype),
        atol=0.0,
        rtol=0.0,
    )
    assert torch.count_nonzero(source.selector) == hidden - 1
    assert metrics["controlled_transition_source_anchor_variation"] >= 0
    parameters = inspect.signature(transition.build_source).parameters
    assert "g3_rollout" in parameters
    assert "facts" not in parameters

    zero = torch.zeros_like(g3_rollout, requires_grad=True)
    zero_source, _ = transition.build_source(g3_rollout=zero)
    assert zero_source.selector is zero
    assert torch.count_nonzero(zero_source.selector) == 0

    invalid = torch.zeros(1, rows - 1, hidden)
    try:
        transition.build_source(g3_rollout=invalid)
    except ValueError as error:
        assert "exact [B,4*C*8*8,H] G3 rollout" in str(error)
    else:
        raise AssertionError("transition source must reject a partial G3 chart")


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


def test_proposal_ablation_does_not_alias_p1_or_controlled_transition() -> None:
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
    assert torch.equal(
        ablated.factual_dock.protected_detail,
        cache.factual_dock.protected_detail,
    )
    assert torch.equal(
        ablated.transition_source.selector,
        cache.transition_source.selector,
    )
    initial_noise = torch.randn(
        batch.online.batch,
        config.dimensions.action_horizon,
        model.action_codec.physical_dim,
    )
    primary = sample_refined_cached_action(
        model,
        cache,
        config,
        initial_physical_noise=initial_noise,
        dtype=torch.float32,
    )
    matched_ablation = sample_refined_cached_action(
        model,
        ablated,
        config,
        initial_physical_noise=initial_noise,
        dtype=torch.float32,
    )
    torch.testing.assert_close(
        matched_ablation.action,
        primary.action,
        atol=0.0,
        rtol=0.0,
    )


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
