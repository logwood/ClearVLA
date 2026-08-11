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


def _gripper_event_class(delta: Tensor, *, threshold: float = 0.05) -> Tensor:
    """Map gripper deltas to hold/open/close without losing direction."""

    result = torch.zeros_like(delta, dtype=torch.int8)
    result = torch.where(delta >= threshold, torch.ones_like(result), result)
    return torch.where(delta <= -threshold, -torch.ones_like(result), result)


def _tolerant_event_match(
    predicted: Tensor,
    target: Tensor,
    *,
    direction: int,
    tolerance: int,
) -> tuple[int, int, int, float, int]:
    """Match same-direction event indices with the V120 temporal tolerance.

    Decoded action events are sequence events, so moving an otherwise correct
    open/close by one or two action rows should be reported as timing error,
    not as an unrelated false-positive/false-negative pair.  The dedicated
    event head is still evaluated row by row below.
    """

    predicted_rows = predicted.detach().to(device="cpu").tolist()
    target_rows = target.detach().to(device="cpu").tolist()
    true_positive = 0
    predicted_count = 0
    target_count = 0
    timing_sum = 0.0
    timing_count = 0
    for predicted_row, target_row in zip(predicted_rows, target_rows, strict=True):
        predicted_indices = [
            index for index, value in enumerate(predicted_row) if int(value) == direction
        ]
        target_indices = [
            index for index, value in enumerate(target_row) if int(value) == direction
        ]
        predicted_count += len(predicted_indices)
        target_count += len(target_indices)
        # Each entry is (matches, total timing error, matched distances).  This
        # is tiny (24x24) and runs only during validation.
        table: list[list[tuple[int, float, tuple[int, ...]]]] = [
            [(0, 0.0, ()) for _ in range(len(target_indices) + 1)]
            for _ in range(len(predicted_indices) + 1)
        ]

        def better(
            left: tuple[int, float, tuple[int, ...]],
            right: tuple[int, float, tuple[int, ...]],
        ) -> tuple[int, float, tuple[int, ...]]:
            if left[0] != right[0]:
                return left if left[0] > right[0] else right
            return left if left[1] <= right[1] else right

        for pred_position in range(1, len(predicted_indices) + 1):
            for target_position in range(1, len(target_indices) + 1):
                best = better(
                    table[pred_position - 1][target_position],
                    table[pred_position][target_position - 1],
                )
                distance = abs(
                    predicted_indices[pred_position - 1]
                    - target_indices[target_position - 1]
                )
                if distance <= int(tolerance):
                    previous = table[pred_position - 1][target_position - 1]
                    best = better(
                        best,
                        (
                            previous[0] + 1,
                            previous[1] + float(distance),
                            (*previous[2], distance),
                        ),
                    )
                table[pred_position][target_position] = best
        matched, _, distances = table[-1][-1]
        true_positive += matched
        timing_sum += float(sum(distances))
        timing_count += len(distances)
    return true_positive, predicted_count, target_count, timing_sum, timing_count


@dataclass
class ValidationAccumulator:
    action_scale: Tensor | None = None
    action_offset: Tensor | None = None
    gripper_event_threshold: float = 0.10
    arm_motion_threshold: float = 0.02
    gripper_event_tolerance: int = 2
    square_error: dict[str, Tensor] = field(default_factory=dict)
    element_count: dict[str, int] = field(default_factory=dict)
    samples: int = 0
    classification_counts: dict[str, Tensor] = field(default_factory=dict)
    scalar_totals: dict[str, Tensor] = field(default_factory=dict)

    @classmethod
    def from_action_normalizer(
        cls,
        normalizer: ArrayNormalizer,
        *,
        device: torch.device,
        gripper_event_threshold: float = 0.10,
        arm_motion_threshold: float = 0.02,
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
        offset = torch.as_tensor(
            normalizer.offset,
            device=device,
            dtype=torch.float32,
        ).reshape(1, 1, -1)
        return cls(
            action_scale=scale,
            action_offset=offset,
            gripper_event_threshold=float(gripper_event_threshold),
            arm_motion_threshold=float(arm_motion_threshold),
        )

    def _add_classification(
        self,
        name: str,
        *,
        true_positive: Tensor,
        predicted: Tensor,
        target: Tensor,
    ) -> None:
        for suffix, value in (
            ("true_positive", true_positive),
            ("predicted", predicted),
            ("target", target),
        ):
            key = f"{name}_{suffix}"
            self.classification_counts[key] = self.classification_counts.get(
                key, value.new_zeros(())
            ) + value

    def _add_scalar(self, name: str, value: Tensor) -> None:
        self.scalar_totals[name] = self.scalar_totals.get(
            name, value.new_zeros(())
        ) + value

    def update(
        self,
        prediction: Tensor,
        batch: TrainingBatch,
        *,
        event_logits: Tensor | None = None,
        motion_logits: Tensor | None = None,
        motion_target: Tensor | None = None,
    ) -> None:
        target = batch.action_target.normalized.float()
        error = prediction.float() - target
        rows = {
            "normalized_action": error,
            "normalized_first": error[:, :1],
            "normalized_first8": error[:, :8],
            "normalized_tail": error[:, 8:],
            "normalized_arm": error[..., :-1],
            "normalized_gripper": error[..., -1:],
            "normalized_band_1_4": error[:, :4],
            "normalized_band_5_12": error[:, 4:12],
            "normalized_band_13_24": error[:, 12:24],
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
                    "physical_band_1_4": physical_error[:, :4],
                    "physical_band_5_12": physical_error[:, 4:12],
                    "physical_band_13_24": physical_error[:, 12:24],
                }
            )
        self.samples += int(prediction.shape[0])
        for name, value in rows.items():
            update = value.detach().float().square().sum()
            self.square_error[name] = self.square_error.get(name, update.new_zeros(())) + update
            self.element_count[name] = self.element_count.get(name, 0) + int(value.numel())
        normalized_current = batch.online.history.action_state.float()
        if self.action_scale is not None:
            if self.action_offset is None:
                raise ValueError("validation action offset is missing")
            raw_prediction = (prediction.float() - self.action_offset) / self.action_scale
            raw_target = batch.action_target.raw_units.float()
            raw_current = batch.action_target.current_raw_units.float()
        else:
            raw_prediction = prediction.float()
            raw_target = target
            raw_current = normalized_current
        target_boundary = torch.cat((raw_current[:, None], raw_target[:, :-1]), dim=1)
        pred_boundary = torch.cat((raw_current[:, None], raw_prediction[:, :-1]), dim=1)
        target_delta = raw_target - target_boundary
        pred_delta = raw_prediction - pred_boundary
        target_class = _gripper_event_class(
            target_delta[..., -1], threshold=self.gripper_event_threshold
        )
        pred_class = _gripper_event_class(
            pred_delta[..., -1], threshold=self.gripper_event_threshold
        )
        target_event = target_class != 0
        pred_event = pred_class != 0
        decoded_counts: dict[str, tuple[int, int, int, float, int]] = {}
        for direction, name in ((-1, "open"), (1, "close")):
            decoded_counts[name] = _tolerant_event_match(
                pred_class,
                target_class,
                direction=direction,
                tolerance=self.gripper_event_tolerance,
            )
            matched, predicted_count, target_count, timing_sum, timing_count = (
                decoded_counts[name]
            )
            self._add_classification(
                f"decoded_gripper_{name}",
                true_positive=error.new_tensor(float(matched)),
                predicted=error.new_tensor(float(predicted_count)),
                target=error.new_tensor(float(target_count)),
            )
            self._add_scalar(
                f"decoded_gripper_{name}_timing_sum",
                error.new_tensor(timing_sum),
            )
            self._add_scalar(
                f"decoded_gripper_{name}_timing_count",
                error.new_tensor(float(timing_count)),
            )
        self._add_classification(
            "decoded_gripper",
            true_positive=error.new_tensor(
                float(sum(row[0] for row in decoded_counts.values()))
            ),
            predicted=pred_event.float().sum(),
            target=target_event.float().sum(),
        )
        if event_logits is None:
            event_head_class = pred_class
        else:
            if tuple(event_logits.shape) != (*target_class.shape, 3):
                raise ValueError("validation event logits must be [B,T,3]")
            event_index = event_logits.detach().float().argmax(dim=-1)
            event_head_class = torch.where(
                event_index == 1,
                -torch.ones_like(event_index),
                torch.where(event_index == 2, torch.ones_like(event_index), event_index),
            ).to(dtype=target_class.dtype)
        event_head_predicted = event_head_class != 0
        self._add_classification(
            "event_head",
            true_positive=(target_event & event_head_predicted).float().sum(),
            predicted=event_head_predicted.float().sum(),
            target=target_event.float().sum(),
        )
        self._add_scalar(
            "event_head_correct",
            (event_head_class == target_class).float().sum(),
        )
        self._add_scalar("event_head_rows", error.new_tensor(float(target_class.numel())))
        for direction, name in ((-1, "open"), (1, "close")):
            predicted_direction = event_head_class == direction
            target_direction = target_class == direction
            self._add_classification(
                f"event_head_{name}",
                true_positive=(predicted_direction & target_direction).float().sum(),
                predicted=predicted_direction.float().sum(),
                target=target_direction.float().sum(),
            )

        decoded_motion_target = target_delta[..., :-1].norm(dim=-1) >= float(
            self.arm_motion_threshold
        )
        decoded_motion = pred_delta[..., :-1].norm(dim=-1) >= float(
            self.arm_motion_threshold
        )
        self._add_classification(
            "decoded_motion",
            true_positive=(decoded_motion_target & decoded_motion).float().sum(),
            predicted=decoded_motion.float().sum(),
            target=decoded_motion_target.float().sum(),
        )
        head_target = decoded_motion_target if motion_target is None else motion_target.bool()
        if tuple(head_target.shape) != tuple(decoded_motion_target.shape):
            raise ValueError("validation motion target must be [B,T]")
        head_prediction = (
            decoded_motion
            if motion_logits is None
            else torch.sigmoid(motion_logits.detach().float()) >= 0.5
        )
        if tuple(head_prediction.shape) != tuple(head_target.shape):
            raise ValueError("validation motion logits must be [B,T]")
        self._add_classification(
            "motion_head",
            true_positive=(head_target & head_prediction).float().sum(),
            predicted=head_prediction.float().sum(),
            target=head_target.float().sum(),
        )
        self._add_scalar(
            "motion_head_correct",
            (head_target == head_prediction).float().sum(),
        )
        self._add_scalar("motion_head_rows", error.new_tensor(float(head_target.numel())))
        if motion_logits is None:
            motion_probability = decoded_motion.float()
        else:
            motion_probability = torch.sigmoid(motion_logits.detach().float())
        self._add_scalar("motion_head_probability_sum", motion_probability.sum())

    def means(self) -> dict[str, float]:
        if self.samples <= 0:
            raise ValueError("validation did not consume any samples")
        required_counts = {
            f"{name}_{suffix}"
            for name in (
                "decoded_gripper",
                "decoded_gripper_open",
                "decoded_gripper_close",
                "event_head",
                "event_head_open",
                "event_head_close",
                "decoded_motion",
                "motion_head",
            )
            for suffix in ("true_positive", "predicted", "target")
        }
        if not required_counts.issubset(self.classification_counts):
            raise ValueError("validation classification accounting is incomplete")
        rmse = {
            name: (value / max(self.element_count[name], 1)).sqrt()
            for name, value in self.square_error.items()
        }
        def classification(name: str) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
            true_positive = self.classification_counts[f"{name}_true_positive"]
            predicted = self.classification_counts[f"{name}_predicted"]
            target_count = self.classification_counts[f"{name}_target"]
            precision = true_positive / predicted.clamp_min(1.0)
            recall = true_positive / target_count.clamp_min(1.0)
            f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)
            return precision, recall, f1, predicted, target_count

        precision, recall, f1, event_predicted, event_target = classification(
            "decoded_gripper"
        )
        head_precision, head_recall, head_f1, head_predicted, head_target = classification(
            "event_head"
        )
        motion_precision, motion_recall, motion_f1, motion_predicted, motion_target_count = (
            classification("motion_head")
        )
        decoded_motion_precision, decoded_motion_recall, decoded_motion_f1, _, _ = (
            classification("decoded_motion")
        )
        open_precision, open_recall, open_f1, _, _ = classification(
            "decoded_gripper_open"
        )
        close_precision, close_recall, close_f1, _, _ = classification(
            "decoded_gripper_close"
        )
        head_open_precision, head_open_recall, head_open_f1, _, _ = classification(
            "event_head_open"
        )
        head_close_precision, head_close_recall, head_close_f1, _, _ = classification(
            "event_head_close"
        )
        timing_sum = (
            self.scalar_totals["decoded_gripper_open_timing_sum"]
            + self.scalar_totals["decoded_gripper_close_timing_sum"]
        )
        timing_count = (
            self.scalar_totals["decoded_gripper_open_timing_count"]
            + self.scalar_totals["decoded_gripper_close_timing_count"]
        )
        event_rows = self.scalar_totals["event_head_rows"].clamp_min(1.0)
        motion_rows = self.scalar_totals["motion_head_rows"].clamp_min(1.0)
        tensors = {
            "validation_action_rmse_normalized": rmse["normalized_action"],
            "validation_first_rmse_normalized": rmse["normalized_first"],
            "validation_first8_rmse_normalized": rmse["normalized_first8"],
            "validation_tail_rmse_normalized": rmse["normalized_tail"],
            "validation_tail_first_ratio_normalized": rmse["normalized_tail"]
            / rmse["normalized_first"].clamp_min(1e-8),
            "validation_arm_rmse_normalized": rmse["normalized_arm"],
            "validation_gripper_rmse_normalized": rmse["normalized_gripper"],
            "validation_decoded_gripper_event_precision": precision,
            "validation_decoded_gripper_event_recall": recall,
            "validation_decoded_gripper_event_f1": f1,
            "validation_decoded_gripper_events_predicted": event_predicted,
            "validation_decoded_gripper_events_target": event_target,
            "validation_decoded_gripper_event_ratio": event_predicted
            / event_target.clamp_min(1.0),
            "validation_decoded_gripper_open_precision": open_precision,
            "validation_decoded_gripper_open_recall": open_recall,
            "validation_decoded_gripper_open_f1": open_f1,
            "validation_decoded_gripper_close_precision": close_precision,
            "validation_decoded_gripper_close_recall": close_recall,
            "validation_decoded_gripper_close_f1": close_f1,
            "validation_decoded_gripper_timing_mae_steps": timing_sum
            / timing_count.clamp_min(1.0),
            "validation_event_head_precision": head_precision,
            "validation_event_head_recall": head_recall,
            "validation_event_head_f1": head_f1,
            "validation_event_head_events_predicted": head_predicted,
            "validation_event_head_events_target": head_target,
            "validation_event_head_accuracy": self.scalar_totals["event_head_correct"]
            / event_rows,
            "validation_event_head_open_precision": head_open_precision,
            "validation_event_head_open_recall": head_open_recall,
            "validation_event_head_open_f1": head_open_f1,
            "validation_event_head_close_precision": head_close_precision,
            "validation_event_head_close_recall": head_close_recall,
            "validation_event_head_close_f1": head_close_f1,
            "validation_event_head_minus_decoded_f1": head_f1 - f1,
            "validation_motion_head_precision": motion_precision,
            "validation_motion_head_recall": motion_recall,
            "validation_motion_head_f1": motion_f1,
            "validation_motion_head_events_predicted": motion_predicted,
            "validation_motion_head_events_target": motion_target_count,
            "validation_motion_head_accuracy": self.scalar_totals["motion_head_correct"]
            / motion_rows,
            "validation_motion_head_predicted_rate": motion_predicted / motion_rows,
            "validation_motion_head_target_rate": motion_target_count / motion_rows,
            "validation_motion_head_mean_probability": self.scalar_totals[
                "motion_head_probability_sum"
            ]
            / motion_rows,
            "validation_decoded_motion_precision": decoded_motion_precision,
            "validation_decoded_motion_recall": decoded_motion_recall,
            "validation_decoded_motion_f1": decoded_motion_f1,
            "validation_band_1_4_rmse_normalized": rmse["normalized_band_1_4"],
            "validation_band_5_12_rmse_normalized": rmse["normalized_band_5_12"],
            "validation_band_13_24_rmse_normalized": rmse["normalized_band_13_24"],
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
                    "validation_band_1_4_rmse_physical": rmse["physical_band_1_4"],
                    "validation_band_5_12_rmse_physical": rmse["physical_band_5_12"],
                    "validation_band_13_24_rmse_physical": rmse["physical_band_13_24"],
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
            gripper_event_threshold=config.objectives.gripper_event_threshold,
            arm_motion_threshold=config.objectives.arm_motion_threshold,
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
        target_physical = model.action_codec.encode(
            batch.action_target.normalized,
            batch.online.history.action_state,
        )
        motion_target = (
            model.action_codec.split(target_physical).arm_delta.float().norm(dim=-1)
            >= float(config.objectives.arm_motion_threshold)
        )
        accumulator.update(
            result.action,
            batch,
            event_logits=result.event_logits,
            motion_logits=result.motion_logits,
            motion_target=motion_target,
        )
    return accumulator.means()


__all__ = ["ValidationAccumulator", "evaluate_loader"]
