"""Deployment-path validation metrics for the clean mainline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch
from torch import Tensor

from ..config import ExperimentConfig
from ..data.loading import GoalTemplate, to_training_batch
from ..data.normalizer import ArrayNormalizer
from ..interfaces import TrainingBatch
from ..model.policy import ClearVLAMainlinePolicy
from .logging import tensor_scalars
from .sampling import sample_action


def _rmse(value: Tensor) -> Tensor:
    return value.float().square().mean().sqrt()


def _gripper_event_class(delta: Tensor, *, threshold: float = 0.05) -> Tensor:
    """Map gripper deltas to hold/open/close without losing direction."""

    result = torch.zeros_like(delta, dtype=torch.int8)
    result = torch.where(delta >= threshold, torch.ones_like(result), result)
    return torch.where(delta <= -threshold, -torch.ones_like(result), result)


def validation_metrics(prediction: Tensor, batch: TrainingBatch) -> dict[str, Tensor]:
    target = batch.action_target.normalized.float()
    error = prediction.float() - target
    first = error[:, :1]
    first8 = error[:, :8]
    tail = error[:, 8:]
    arm = error[..., :-1]
    grip = error[..., -1:]
    current = batch.online.history.action_state.float()
    boundary = torch.cat((current[:, None], target[:, :-1]), dim=1)
    target_delta = target[..., -1] - boundary[..., -1]
    pred_boundary = torch.cat((current[:, None], prediction[:, :-1].float()), dim=1)
    pred_delta = prediction[..., -1].float() - pred_boundary[..., -1]
    target_class = _gripper_event_class(target_delta)
    pred_class = _gripper_event_class(pred_delta)
    target_event = target_class != 0
    pred_event = pred_class != 0
    true_positive = (target_event & (pred_class == target_class)).float().sum()
    precision = true_positive / pred_event.float().sum().clamp_min(1.0)
    recall = true_positive / target_event.float().sum().clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)
    return {
        "validation_action_rmse_normalized": _rmse(error),
        "validation_first_rmse_normalized": _rmse(first),
        "validation_first8_rmse_normalized": _rmse(first8),
        "validation_tail_rmse_normalized": _rmse(tail),
        "validation_tail_first_ratio_normalized": _rmse(tail) / _rmse(first).clamp_min(1e-8),
        "validation_arm_rmse_normalized": _rmse(arm),
        "validation_gripper_rmse_normalized": _rmse(grip),
        "validation_gripper_event_precision_normalized": precision,
        "validation_gripper_event_recall_normalized": recall,
        "validation_gripper_event_f1_normalized": f1,
        "validation_gripper_events_predicted_normalized": pred_event.float().sum(),
        "validation_gripper_events_target_normalized": target_event.float().sum(),
    }


@dataclass
class ValidationAccumulator:
    action_scale: Tensor | None = None
    square_error: dict[str, Tensor] = field(default_factory=dict)
    element_count: dict[str, int] = field(default_factory=dict)
    samples: int = 0
    event_true_positive: Tensor | None = None
    event_predicted: Tensor | None = None
    event_target: Tensor | None = None

    @classmethod
    def from_action_normalizer(
        cls,
        normalizer: ArrayNormalizer,
        *,
        device: torch.device,
    ) -> "ValidationAccumulator":
        """Build physical-unit accounting from the exact training chart.

        For an affine chart ``normalized = physical * scale + offset``, the
        physical prediction error is simply ``normalized_error / scale``.
        The offset cancels, so no decoded prediction tensor needs to be kept.
        """

        scale = torch.as_tensor(
            normalizer.scale,
            device=device,
            dtype=torch.float32,
        ).reshape(1, 1, -1)
        if bool((scale <= 0.0).any()):
            raise ValueError("action normalizer scale must be strictly positive")
        return cls(action_scale=scale)

    def update(self, prediction: Tensor, batch: TrainingBatch) -> None:
        target = batch.action_target.normalized.float()
        error = prediction.float() - target
        rows = {
            "normalized_action": error,
            "normalized_first": error[:, :1],
            "normalized_first8": error[:, :8],
            "normalized_tail": error[:, 8:],
            "normalized_arm": error[..., :-1],
            "normalized_gripper": error[..., -1:],
        }
        if self.action_scale is not None:
            if self.action_scale.device != error.device:
                raise ValueError("validation action scale is on the wrong device")
            if int(self.action_scale.shape[-1]) != int(error.shape[-1]):
                raise ValueError("validation action scale width does not match actions")
            physical_error = error / self.action_scale
            rows.update(
                {
                    "physical_action": physical_error,
                    "physical_first": physical_error[:, :1],
                    "physical_first8": physical_error[:, :8],
                    "physical_tail": physical_error[:, 8:],
                    "physical_arm": physical_error[..., :-1],
                    "physical_gripper": physical_error[..., -1:],
                }
            )
        self.samples += int(prediction.shape[0])
        for name, value in rows.items():
            update = value.detach().float().square().sum()
            self.square_error[name] = self.square_error.get(name, update.new_zeros(())) + update
            self.element_count[name] = self.element_count.get(name, 0) + int(value.numel())
        current = batch.online.history.action_state.float()
        target_boundary = torch.cat((current[:, None], target[:, :-1]), dim=1)
        pred_boundary = torch.cat((current[:, None], prediction[:, :-1].float()), dim=1)
        target_class = _gripper_event_class(target[..., -1] - target_boundary[..., -1])
        pred_class = _gripper_event_class(prediction[..., -1].float() - pred_boundary[..., -1])
        target_event = target_class != 0
        pred_event = pred_class != 0
        true_positive = (target_event & (pred_class == target_class)).float().sum()
        predicted = pred_event.float().sum()
        target_count = target_event.float().sum()
        self.event_true_positive = (
            true_positive
            if self.event_true_positive is None
            else self.event_true_positive + true_positive
        )
        self.event_predicted = (
            predicted if self.event_predicted is None else self.event_predicted + predicted
        )
        self.event_target = (
            target_count if self.event_target is None else self.event_target + target_count
        )

    def means(self) -> dict[str, float]:
        if self.samples <= 0:
            raise ValueError("validation did not consume any samples")
        if (
            self.event_true_positive is None
            or self.event_predicted is None
            or self.event_target is None
        ):
            raise ValueError("validation event accounting is incomplete")
        rmse = {
            name: (value / max(self.element_count[name], 1)).sqrt()
            for name, value in self.square_error.items()
        }
        precision = self.event_true_positive / self.event_predicted.clamp_min(1.0)
        recall = self.event_true_positive / self.event_target.clamp_min(1.0)
        f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)
        tensors = {
            "validation_action_rmse_normalized": rmse["normalized_action"],
            "validation_first_rmse_normalized": rmse["normalized_first"],
            "validation_first8_rmse_normalized": rmse["normalized_first8"],
            "validation_tail_rmse_normalized": rmse["normalized_tail"],
            "validation_tail_first_ratio_normalized": rmse["normalized_tail"]
            / rmse["normalized_first"].clamp_min(1e-8),
            "validation_arm_rmse_normalized": rmse["normalized_arm"],
            "validation_gripper_rmse_normalized": rmse["normalized_gripper"],
            "validation_gripper_event_precision_normalized": precision,
            "validation_gripper_event_recall_normalized": recall,
            "validation_gripper_event_f1_normalized": f1,
            "validation_gripper_events_predicted_normalized": self.event_predicted,
            "validation_gripper_events_target_normalized": self.event_target,
        }
        if "physical_action" in rmse:
            tensors.update(
                {
                    "validation_action_rmse_physical": rmse["physical_action"],
                    "validation_first_rmse_physical": rmse["physical_first"],
                    "validation_first8_rmse_physical": rmse["physical_first8"],
                    "validation_tail_rmse_physical": rmse["physical_tail"],
                    "validation_tail_first_ratio_physical": rmse["physical_tail"]
                    / rmse["physical_first"].clamp_min(1e-8),
                    "validation_arm_rmse_physical": rmse["physical_arm"],
                    "validation_gripper_rmse_physical": rmse["physical_gripper"],
                }
            )
        return tensor_scalars(tensors)


@torch.no_grad()
def evaluate_loader(
    model: ClearVLAMainlinePolicy,
    loader,
    config: ExperimentConfig,
    *,
    generator: torch.Generator | None = None,
    max_batches: int = 0,
    goal: GoalTemplate | None = None,
    action_normalizer: ArrayNormalizer | None = None,
    device: torch.device | None = None,
) -> dict[str, float]:
    model_device = next(model.parameters()).device if device is None else device
    accumulator = (
        ValidationAccumulator()
        if action_normalizer is None
        else ValidationAccumulator.from_action_normalizer(
            action_normalizer,
            device=model_device,
        )
    )
    for batch_index, raw_batch in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        if isinstance(raw_batch, TrainingBatch):
            batch = raw_batch
        elif isinstance(raw_batch, Mapping):
            if goal is None:
                raise ValueError("raw validation batches require the explicit T5 goal template")
            batch = to_training_batch(
                raw_batch,
                goal=goal,
                config=config,
                device=model_device,
            )
        else:
            raise TypeError("mainline validation loader yielded an unsupported batch type")
        result = sample_action(model, batch.online, config, generator=generator)
        accumulator.update(result.action, batch)
    return accumulator.means()


__all__ = ["ValidationAccumulator", "evaluate_loader", "validation_metrics"]
