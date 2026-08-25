from __future__ import annotations

import json
import math
from dataclasses import replace
from unittest import mock

import torch
from torch import nn

from clearvla.mainline.config import ExperimentConfig
from clearvla.mainline.runtime.logging import DeviceMetricAccumulator, JsonlRunLogger
from clearvla.mainline.train import _emit_training_window, _write_gradient_spike
from clearvla.mainline.training.engine import MainlineTrainingEngine
from clearvla.mainline.training.gradient_audit import (
    GradientPreclipWindowAccumulator,
)
from clearvla.mainline.training.losses import LossLedger
from clearvla.mainline.training.optimizer import WarmupCosineSchedule


class _TinyGradientOwners(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.observation = nn.Module()
        self.observation.register_parameter("l2_owner", nn.Parameter(torch.zeros(4)))
        self.observation.register_parameter("abs_owner", nn.Parameter(torch.zeros(1)))
        self.seen_step: int | None = None

    def set_training_step(self, step: int) -> None:
        self.seen_step = int(step)


def _tiny_engine(
    *,
    threshold: float | None,
) -> tuple[
    MainlineTrainingEngine,
    _TinyGradientOwners,
    torch.optim.Optimizer,
    WarmupCosineSchedule,
    LossLedger,
]:
    base = ExperimentConfig()
    config = replace(base, runtime=replace(base.runtime, compute_dtype="fp32"))
    model = _TinyGradientOwners()
    named = tuple(model.named_parameters())
    optimizer = torch.optim.AdamW(
        [
            {
                "params": tuple(parameter for _name, parameter in named),
                "parameter_names": tuple(name for name, _parameter in named),
                "name": "observation/decay",
                "lr": config.optimizer.learning_rate,
                "weight_decay": config.optimizer.weight_decay,
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
    loss = (
        4.0 * model.observation.l2_owner.sum()
        + 7.0 * model.observation.abs_owner.sum()
    )
    zero = loss.new_zeros(())
    ledger = LossLedger(
        total=loss,
        groups={"action": loss, "representation": zero, "execution": zero},
        contributions={"action_flow": loss},
        terms={"action_flow": loss},
    )
    return engine, model, optimizer, schedule, ledger


def _assert_optimizer_states_equal(
    left: torch.optim.Optimizer,
    right: torch.optim.Optimizer,
) -> None:
    left_state = left.state_dict()
    right_state = right.state_dict()
    assert left_state["param_groups"] == right_state["param_groups"]
    assert set(left_state["state"]) == set(right_state["state"])
    for parameter_index in left_state["state"]:
        left_row = left_state["state"][parameter_index]
        right_row = right_state["state"][parameter_index]
        assert set(left_row) == set(right_row)
        for name, left_value in left_row.items():
            right_value = right_row[name]
            if isinstance(left_value, torch.Tensor):
                assert torch.equal(left_value, right_value), name
            else:
                assert left_value == right_value, name


def test_finite_spike_is_named_and_written_before_clipping(tmp_path) -> None:
    engine, model, _optimizer, _schedule, ledger = _tiny_engine(threshold=5.0)
    logger = JsonlRunLogger(tmp_path)
    callback_saw_raw_gradients = False

    def handler(report) -> None:
        nonlocal callback_saw_raw_gradients
        torch.testing.assert_close(
            model.observation.l2_owner.grad,
            torch.full((4,), 4.0),
        )
        torch.testing.assert_close(
            model.observation.abs_owner.grad,
            torch.full((1,), 7.0),
        )
        callback_saw_raw_gradients = True
        _write_gradient_spike(
            logger,
            report,
            epoch=3,
            batch=19,
            step=41,
        )

    with mock.patch.object(engine, "_forward", return_value=(ledger, {})):
        result = engine.train_step(
            object(),  # type: ignore[arg-type]
            gradient_spike_handler=handler,
        )

    assert callback_saw_raw_gradients
    assert result.gradient_norm_scalar is not None
    assert math.isclose(result.gradient_norm_scalar, math.sqrt(113.0), rel_tol=1e-6)
    rows = [json.loads(line) for line in logger.path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "gradient_spike"
    assert (row["epoch"], row["batch"], row["step"]) == (3, 19, 41)
    assert row["max_l2_parameter_name"] == "observation.l2_owner"
    assert row["max_l2_parameter_role"] == "observation"
    assert row["max_l2_optimizer_group"] == "observation/decay"
    assert row["max_l2_shape"] == [4]
    assert row["max_l2_dtype"] == "float32"
    assert math.isclose(row["max_l2_l2"], 8.0, rel_tol=1e-6)
    assert row["max_abs_parameter_name"] == "observation.abs_owner"
    assert row["max_abs_parameter_role"] == "observation"
    assert row["max_abs_optimizer_group"] == "observation/decay"
    assert row["max_abs_shape"] == [1]
    assert row["max_abs_dtype"] == "float32"
    assert math.isclose(row["max_abs_max_abs"], 7.0, rel_tol=1e-6)


def test_ordinary_finite_batch_does_not_scan_parameters() -> None:
    engine, _model, _optimizer, _schedule, ledger = _tiny_engine(threshold=100.0)
    events: list[object] = []
    with (
        mock.patch.object(engine, "_forward", return_value=(ledger, {})),
        mock.patch(
            "clearvla.mainline.training.engine.build_finite_gradient_spike_report"
        ) as scanner,
    ):
        result = engine.train_step(
            object(),  # type: ignore[arg-type]
            gradient_spike_handler=events.append,
        )
    scanner.assert_not_called()
    assert not events
    assert result.gradient_norm_scalar is not None


def test_gradient_audit_switch_preserves_exact_update() -> None:
    enabled = _tiny_engine(threshold=5.0)
    disabled = _tiny_engine(threshold=None)
    enabled_engine, enabled_model, enabled_optimizer, enabled_schedule, enabled_ledger = enabled
    disabled_engine, disabled_model, disabled_optimizer, disabled_schedule, disabled_ledger = (
        disabled
    )
    enabled_events: list[object] = []
    disabled_events: list[object] = []

    with mock.patch.object(enabled_engine, "_forward", return_value=(enabled_ledger, {})):
        enabled_engine.train_step(
            object(),  # type: ignore[arg-type]
            gradient_spike_handler=enabled_events.append,
        )
    with mock.patch.object(disabled_engine, "_forward", return_value=(disabled_ledger, {})):
        disabled_engine.train_step(
            object(),  # type: ignore[arg-type]
            gradient_spike_handler=disabled_events.append,
        )

    assert len(enabled_events) == 1
    assert not disabled_events
    for enabled_parameter, disabled_parameter in zip(
        enabled_model.parameters(),
        disabled_model.parameters(),
        strict=True,
    ):
        assert torch.equal(enabled_parameter, disabled_parameter)
    _assert_optimizer_states_equal(enabled_optimizer, disabled_optimizer)
    assert enabled_schedule.state_dict() == disabled_schedule.state_dict()
    assert enabled_engine.global_step == disabled_engine.global_step == 1


def test_preclip_window_tracks_weighted_mean_max_current_and_owner_indices() -> None:
    window = GradientPreclipWindowAccumulator()
    window.update(1.0, weight=8, batch_offset=1, global_step=101)
    window.update(4.0, weight=8, batch_offset=2, global_step=102)
    window.update(2.0, weight=4, batch_offset=3, global_step=103)

    values = window.materialize()
    assert math.isclose(values["gradient_window_preclip_l2_mean"], 2.4)
    assert values["gradient_window_preclip_l2_max"] == 4.0
    assert values["gradient_window_preclip_l2_current"] == 2.0
    assert values["gradient_window_preclip_l2_max_batch_offset"] == 2.0
    assert values["gradient_window_preclip_l2_max_global_step"] == 102.0


def test_epoch_tail_training_window_is_persisted_with_gradient_owner(tmp_path) -> None:
    metrics = DeviceMetricAccumulator()
    metrics.update(
        {
            "loss_total": torch.tensor(0.25),
            "loss_action_flow": torch.tensor(0.20),
        },
        weight=8,
    )
    gradients = GradientPreclipWindowAccumulator()
    gradients.update(
        3.5,
        weight=8,
        batch_offset=1,
        global_step=101,
    )
    logger = JsonlRunLogger(tmp_path)

    values = _emit_training_window(
        logger=logger,
        config=ExperimentConfig(),
        window_metrics=metrics,
        gradient_window=gradients,
        epoch=2,
        batch=1,
        step=101,
        window_seconds=2.0,
        window_samples=8,
        window_batches=1,
        learning_rate=1.0e-4,
        boundary="epoch_tail",
    )

    rows = [json.loads(line) for line in logger.path.read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "train"
    assert row["window_boundary"] == "epoch_tail"
    assert row["window_batches"] == 1
    assert row["window_samples"] == 8
    assert row["metrics"]["gradient_window_preclip_l2_max"] == 3.5
    assert row["metrics"]["gradient_window_preclip_l2_max_global_step"] == 101.0
    assert values["runtime_window_seconds_per_batch"] == 2.0
    assert values["runtime_window_samples_per_second"] == 4.0
