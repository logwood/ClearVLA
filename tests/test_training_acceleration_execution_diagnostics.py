from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence

import torch
from test_mainline_policy import _batch, _config

from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
from clearvla.mainline.training.engine import MainlineTrainingEngine
from clearvla.mainline.training.optimizer import WarmupCosineSchedule, build_optimizer


def _engine(model: ClearVLAMainlinePolicy, config) -> MainlineTrainingEngine:
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
        train_flow_generator=torch.Generator().manual_seed(9202),
        train_condition_generator=torch.Generator().manual_seed(9203),
        skip_postglobal_audit=True,
        gradient_spike_audit_threshold=None,
    )


def _capture_raw_gradients(model: ClearVLAMainlinePolicy) -> dict[str, torch.Tensor]:
    gradients: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            parameter.register_hook(
                lambda gradient, name=name: gradients.__setitem__(
                    name, gradient.detach().clone()
                )
            )
    return gradients


def _assert_state_close(left, right, *, path: str = "state") -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor), path
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0, msg=path)
        return
    if isinstance(left, Mapping):
        assert isinstance(right, Mapping), path
        assert tuple(left) == tuple(right), path
        for name in left:
            _assert_state_close(left[name], right[name], path=f"{path}.{name}")
        return
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes)):
        assert isinstance(right, Sequence), path
        assert len(left) == len(right), path
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            _assert_state_close(
                left_item,
                right_item,
                path=f"{path}[{index}]",
            )
        return
    assert left == right, path


def test_execution_only_training_surface_preserves_complete_update_and_rng() -> None:
    config = _config()
    torch.manual_seed(9200)
    diagnostic_model = ClearVLAMainlinePolicy(config).train()
    initial = copy.deepcopy(diagnostic_model.state_dict())
    execution_only_model = ClearVLAMainlinePolicy(config).train()
    execution_only_model.load_state_dict(initial)
    diagnostic_model.execution_bottom._retain_training_decoder_diagnostics = True

    diagnostic_engine = _engine(diagnostic_model, config)
    execution_only_engine = _engine(execution_only_model, config)
    diagnostic_engine._retain_training_execution_diagnostics = True
    diagnostic_raw_gradients = _capture_raw_gradients(diagnostic_model)
    execution_only_raw_gradients = _capture_raw_gradients(execution_only_model)
    batch = _batch(config, batch=2)

    torch.manual_seed(9204)
    diagnostic_result = diagnostic_engine.train_step(
        batch,
        collect_diagnostics=False,
    )
    diagnostic_rng = torch.get_rng_state().clone()
    diagnostic_flow_rng = diagnostic_engine.train_flow_generator.get_state().clone()
    diagnostic_condition_rng = (
        diagnostic_engine.train_condition_generator.get_state().clone()
    )

    torch.manual_seed(9204)
    execution_only_result = execution_only_engine.train_step(
        batch,
        collect_diagnostics=False,
    )
    execution_only_rng = torch.get_rng_state().clone()
    execution_only_flow_rng = (
        execution_only_engine.train_flow_generator.get_state().clone()
    )
    execution_only_condition_rng = (
        execution_only_engine.train_condition_generator.get_state().clone()
    )

    torch.testing.assert_close(
        execution_only_result.loss,
        diagnostic_result.loss,
        rtol=0.0,
        atol=0.0,
    )
    for prefix in ("loss_group_", "loss_contrib_"):
        expected_names = {
            name for name in diagnostic_result.metrics if name.startswith(prefix)
        }
        assert expected_names
        assert expected_names <= set(execution_only_result.metrics)
        for name in expected_names:
            torch.testing.assert_close(
                execution_only_result.metrics[name],
                diagnostic_result.metrics[name],
                rtol=0.0,
                atol=0.0,
                msg=name,
            )
    torch.testing.assert_close(
        execution_only_result.metrics["loss_execution_value"],
        diagnostic_result.metrics["loss_execution_value"],
        rtol=0.0,
        atol=0.0,
    )

    assert tuple(diagnostic_raw_gradients) == tuple(execution_only_raw_gradients)
    for name in diagnostic_raw_gradients:
        torch.testing.assert_close(
            execution_only_raw_gradients[name],
            diagnostic_raw_gradients[name],
            rtol=0.0,
            atol=0.0,
            msg=f"raw gradient: {name}",
        )
    for (diagnostic_name, diagnostic_parameter), (
        execution_name,
        execution_parameter,
    ) in zip(
        diagnostic_model.named_parameters(),
        execution_only_model.named_parameters(),
        strict=True,
    ):
        assert execution_name == diagnostic_name
        torch.testing.assert_close(
            execution_parameter,
            diagnostic_parameter,
            rtol=0.0,
            atol=0.0,
            msg=f"parameter update: {diagnostic_name}",
        )
        if diagnostic_parameter.grad is None or execution_parameter.grad is None:
            assert diagnostic_parameter.grad is execution_parameter.grad
        else:
            torch.testing.assert_close(
                execution_parameter.grad,
                diagnostic_parameter.grad,
                rtol=0.0,
                atol=0.0,
                msg=f"clipped gradient: {diagnostic_name}",
            )

    _assert_state_close(
        execution_only_engine.optimizer.state_dict(),
        diagnostic_engine.optimizer.state_dict(),
        path="optimizer",
    )
    assert execution_only_engine.schedule.step_index == diagnostic_engine.schedule.step_index
    assert torch.equal(execution_only_rng, diagnostic_rng)
    assert torch.equal(execution_only_flow_rng, diagnostic_flow_rng)
    assert torch.equal(execution_only_condition_rng, diagnostic_condition_rng)

    # An ordinary batch retains only the scalar needed by the objective.
    for name in (
        "loss_execution_value_correlation",
        "loss_execution_value_pairwise_accuracy",
        "loss_execution_value_target_spread_p50",
        "evidence_mmd_it_execution_cost",
        "evidence_mmd_it_execution_selection_entropy",
    ):
        assert name not in execution_only_result.metrics
    assert "loss_execution_value_correlation" in diagnostic_result.metrics

    # The bounded logging call still exposes the complete audit surface.
    logging_result = diagnostic_engine.train_step(
        batch,
        collect_diagnostics=True,
    )
    for name in (
        "loss_execution_value_correlation",
        "loss_execution_value_pairwise_accuracy",
        "loss_execution_value_target_spread_p50",
        "evidence_mmd_it_execution_cost",
        "evidence_mmd_it_execution_selection_entropy",
    ):
        assert name in logging_result.metrics
