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
from ..model.action_codec import ACTION_BAND_ENDS
from ..model.policy import ClearVLAMainlinePolicy
from .logging import tensor_scalars
from .sampling import sample_action


def _gripper_event_class(delta: Tensor, *, threshold: float = 0.05) -> Tensor:
    """Map gripper deltas to hold/open/close without losing direction."""

    result = torch.zeros_like(delta, dtype=torch.int8)
    result = torch.where(delta >= threshold, torch.ones_like(result), result)
    return torch.where(delta <= -threshold, -torch.ones_like(result), result)


def _action_band_slices(horizon: int) -> tuple[tuple[str, slice], ...]:
    if int(horizon) != ACTION_BAND_ENDS[-1]:
        raise ValueError("validation action bands require the 24-row action horizon")
    rows: list[tuple[str, slice]] = []
    start = 0
    for end in ACTION_BAND_ENDS:
        rows.append((f"{start + 1}_{end}", slice(start, end)))
        start = end
    return tuple(rows)


def _post_event_distance(target_event: Tensor) -> Tensor:
    """Return rows since the latest target event, excluding the event row.

    Values are ``-1`` before the first event, ``0`` on an event row, and
    positive afterwards.  A later event resets the distance, so a persistence
    bin cannot cross an intervening target transition.
    """

    if target_event.ndim != 2 or target_event.dtype != torch.bool:
        raise ValueError("target event mask must be boolean [B,T]")
    horizon = int(target_event.shape[1])
    rows = torch.arange(
        horizon,
        device=target_event.device,
        dtype=torch.long,
    )[None].expand_as(target_event)
    event_rows = torch.where(target_event, rows, torch.full_like(rows, -1))
    latest_event = event_rows.cummax(dim=1).values
    distance = rows - latest_event
    return torch.where(latest_event >= 0, distance, torch.full_like(distance, -1))


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
    scalar_maxima: dict[str, Tensor] = field(default_factory=dict)

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

    def _add_maximum(self, name: str, value: Tensor) -> None:
        previous = self.scalar_maxima.get(name)
        self.scalar_maxima[name] = value if previous is None else torch.maximum(
            previous, value
        )

    def update(
        self,
        prediction: Tensor,
        batch: TrainingBatch,
        *,
        motion_logits: Tensor | None = None,
        motion_target: Tensor | None = None,
        physical_field: Tensor | None = None,
        gripper_decode_delta_blend: float | None = None,
    ) -> None:
        target = batch.action_target.normalized.float()
        error = prediction.float() - target
        action_bands = _action_band_slices(int(error.shape[1]))
        rows = {
            "normalized_action": error,
            "normalized_first": error[:, :1],
            "normalized_first8": error[:, :8],
            "normalized_tail": error[:, 8:],
            "normalized_arm": error[..., :-1],
            "normalized_gripper": error[..., -1:],
        }
        for band_name, band_slice in action_bands:
            rows[f"normalized_band_{band_name}"] = error[:, band_slice]
            rows[f"normalized_gripper_band_{band_name}"] = error[
                :, band_slice, -1:
            ]
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
            for band_name, band_slice in action_bands:
                rows[f"physical_band_{band_name}"] = physical_error[:, band_slice]
                rows[f"physical_gripper_band_{band_name}"] = physical_error[
                    :, band_slice, -1:
                ]
        if physical_field is not None:
            if tuple(physical_field.shape[:2]) != tuple(prediction.shape[:2]) or int(
                physical_field.shape[-1]
            ) != 18:
                raise ValueError("validation physical field must be [B,T,18]")
            if gripper_decode_delta_blend is None:
                raise ValueError("gripper branch diagnostics require the decode blend")
            blend = float(gripper_decode_delta_blend)
            if not 0.0 <= blend <= 1.0:
                raise ValueError("gripper decode blend must lie in [0,1]")
            gripper_field = physical_field.detach().float()[..., -6:]
            absolute_branch = gripper_field[..., :1]
            cumulative_branch = (
                batch.online.history.action_state.detach().float()[:, None, -1:]
                + torch.cumsum(gripper_field[..., 1:2], dim=1)
            )
            reconstructed = (1.0 - blend) * absolute_branch + blend * cumulative_branch
            identity_error = (
                reconstructed - prediction.detach().float()[..., -1:]
            ).abs().amax()
            self._add_maximum("gripper_branch_decode_identity_max_abs", identity_error)
            for band_name, band_slice in action_bands:
                absolute_error = absolute_branch[:, band_slice] - target[
                    :, band_slice, -1:
                ]
                cumulative_error = cumulative_branch[:, band_slice] - target[
                    :, band_slice, -1:
                ]
                disagreement = absolute_branch[:, band_slice] - cumulative_branch[
                    :, band_slice
                ]
                rows[f"normalized_gripper_absolute_branch_band_{band_name}"] = (
                    absolute_error
                )
                rows[f"normalized_gripper_delta_branch_band_{band_name}"] = (
                    cumulative_error
                )
                rows[f"normalized_gripper_branch_disagreement_band_{band_name}"] = (
                    disagreement
                )
                if self.action_scale is not None:
                    gripper_scale = self.action_scale[..., -1:]
                    rows[f"physical_gripper_absolute_branch_band_{band_name}"] = (
                        absolute_error / gripper_scale
                    )
                    rows[f"physical_gripper_delta_branch_band_{band_name}"] = (
                        cumulative_error / gripper_scale
                    )
                    rows[
                        f"physical_gripper_branch_disagreement_band_{band_name}"
                    ] = disagreement / gripper_scale
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
        if self.action_scale is not None:
            post_event_distance = _post_event_distance(target_event)
            physical_gripper_error = raw_prediction[..., -1] - raw_target[..., -1]
            for bin_name, lower, upper in (
                ("1_2", 1, 2),
                ("3_6", 3, 6),
                ("7_plus", 7, None),
            ):
                mask = post_event_distance >= lower
                if upper is not None:
                    mask = mask & (post_event_distance <= upper)
                self._add_scalar(
                    f"gripper_post_event_{bin_name}_square_error",
                    physical_gripper_error.square().masked_select(mask).sum(),
                )
                self._add_scalar(
                    f"gripper_post_event_{bin_name}_rows",
                    mask.detach().float().sum(),
                )
            context_masks = {
                "before_any_event": post_event_distance < 0,
                "event": post_event_distance == 0,
                "post_1_2": (post_event_distance >= 1) & (post_event_distance <= 2),
                "post_3_6": (post_event_distance >= 3) & (post_event_distance <= 6),
                "post_7_plus": post_event_distance >= 7,
            }
            for band_name, band_slice in action_bands:
                for context_name, context_mask in context_masks.items():
                    mask = context_mask[:, band_slice]
                    stem = f"gripper_band_{band_name}_{context_name}"
                    self._add_scalar(
                        f"{stem}_square_error",
                        physical_gripper_error[:, band_slice]
                        .square()
                        .masked_select(mask)
                        .sum(),
                    )
                    self._add_scalar(f"{stem}_rows", mask.detach().float().sum())
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
        timing_sum = (
            self.scalar_totals["decoded_gripper_open_timing_sum"]
            + self.scalar_totals["decoded_gripper_close_timing_sum"]
        )
        timing_count = (
            self.scalar_totals["decoded_gripper_open_timing_count"]
            + self.scalar_totals["decoded_gripper_close_timing_count"]
        )
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
        for band_name, _ in _action_band_slices(ACTION_BAND_ENDS[-1]):
            tensors[f"validation_gripper_band_{band_name}_rmse_normalized"] = rmse[
                f"normalized_gripper_band_{band_name}"
            ]
            for branch_name in (
                "absolute_branch",
                "delta_branch",
                "branch_disagreement",
            ):
                key = f"normalized_gripper_{branch_name}_band_{band_name}"
                if key in rmse:
                    statistic = "rms" if branch_name == "branch_disagreement" else "rmse"
                    tensors[
                        f"validation_gripper_{branch_name}_band_{band_name}_{statistic}_normalized"
                    ] = rmse[key]
        if "gripper_branch_decode_identity_max_abs" in self.scalar_maxima:
            tensors["validation_gripper_branch_decode_identity_max_abs"] = (
                self.scalar_maxima["gripper_branch_decode_identity_max_abs"]
            )
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
            for band_name, _ in _action_band_slices(ACTION_BAND_ENDS[-1]):
                tensors[f"validation_gripper_band_{band_name}_rmse_physical"] = rmse[
                    f"physical_gripper_band_{band_name}"
                ]
                for branch_name in (
                    "absolute_branch",
                    "delta_branch",
                    "branch_disagreement",
                ):
                    key = f"physical_gripper_{branch_name}_band_{band_name}"
                    if key in rmse:
                        statistic = (
                            "rms" if branch_name == "branch_disagreement" else "rmse"
                        )
                        tensors[
                            f"validation_gripper_{branch_name}_band_{band_name}_{statistic}_physical"
                        ] = rmse[key]
                for context_name in (
                    "before_any_event",
                    "event",
                    "post_1_2",
                    "post_3_6",
                    "post_7_plus",
                ):
                    stem = f"gripper_band_{band_name}_{context_name}"
                    row_count = self.scalar_totals[f"{stem}_rows"]
                    square_error = self.scalar_totals[f"{stem}_square_error"]
                    tensors[f"validation_{stem}_rmse_physical"] = (
                        square_error / row_count.clamp_min(1.0)
                    ).sqrt()
                    tensors[f"validation_{stem}_rows"] = row_count
            for bin_name in ("1_2", "3_6", "7_plus"):
                row_count = self.scalar_totals[
                    f"gripper_post_event_{bin_name}_rows"
                ]
                square_error = self.scalar_totals[
                    f"gripper_post_event_{bin_name}_square_error"
                ]
                tensors[
                    f"validation_gripper_post_event_{bin_name}_rmse_physical"
                ] = (square_error / row_count.clamp_min(1.0)).sqrt()
                tensors[f"validation_gripper_post_event_rows_{bin_name}"] = row_count
        return tensor_scalars(tensors)


@dataclass
class MatchedP2InterventionAccumulator:
    """Paired action/error accounting for P2 value/address counterfactuals."""

    action_scale: Tensor
    action_offset: Tensor
    gripper_event_threshold: float
    arm_motion_threshold: float
    primary: dict[str, ValidationAccumulator] = field(default_factory=dict)
    counterfactual: dict[str, ValidationAccumulator] = field(default_factory=dict)
    delta_square_error: dict[str, Tensor] = field(default_factory=dict)
    delta_element_count: dict[str, int] = field(default_factory=dict)
    batches: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_action_normalizer(
        cls,
        normalizer: ArrayNormalizer,
        *,
        device: torch.device,
        gripper_event_threshold: float,
        arm_motion_threshold: float,
    ) -> "MatchedP2InterventionAccumulator":
        base = ValidationAccumulator.from_action_normalizer(
            normalizer,
            device=device,
            gripper_event_threshold=gripper_event_threshold,
            arm_motion_threshold=arm_motion_threshold,
        )
        if base.action_scale is None or base.action_offset is None:
            raise RuntimeError("matched P2 accounting requires the physical action chart")
        return cls(
            action_scale=base.action_scale,
            action_offset=base.action_offset,
            gripper_event_threshold=float(gripper_event_threshold),
            arm_motion_threshold=float(arm_motion_threshold),
        )

    def _new_validation_accumulator(self) -> ValidationAccumulator:
        return ValidationAccumulator(
            action_scale=self.action_scale,
            action_offset=self.action_offset,
            gripper_event_threshold=self.gripper_event_threshold,
            arm_motion_threshold=self.arm_motion_threshold,
        )

    def update(
        self,
        mode: str,
        *,
        primary_action: Tensor,
        counterfactual_action: Tensor,
        batch: TrainingBatch,
    ) -> None:
        if mode not in self.primary:
            self.primary[mode] = self._new_validation_accumulator()
            self.counterfactual[mode] = self._new_validation_accumulator()
        self.primary[mode].update(
            primary_action,
            batch,
        )
        self.counterfactual[mode].update(
            counterfactual_action,
            batch,
        )
        action_delta = (
            counterfactual_action.detach().float() - primary_action.detach().float()
        ) / self.action_scale
        for band_name, band_slice in _action_band_slices(int(action_delta.shape[1])):
            for owner_name, value in (
                ("arm", action_delta[:, band_slice, :-1]),
                ("gripper", action_delta[:, band_slice, -1:]),
            ):
                key = f"{mode}_{owner_name}_band_{band_name}"
                update = value.square().sum()
                self.delta_square_error[key] = self.delta_square_error.get(
                    key, update.new_zeros(())
                ) + update
                self.delta_element_count[key] = self.delta_element_count.get(
                    key, 0
                ) + int(value.numel())
        self.batches[mode] = self.batches.get(mode, 0) + 1

    def means(self) -> dict[str, float]:
        if not self.batches:
            raise ValueError("matched P2 accounting did not consume any batches")
        result: dict[str, float] = {}
        reference_primary: dict[str, float] | None = None
        for mode in sorted(self.batches):
            primary = self.primary[mode].means()
            counterfactual = self.counterfactual[mode].means()
            if reference_primary is None:
                reference_primary = primary
            stem = f"validation_p2_intervention_{mode}"
            result[f"{stem}_batches"] = float(self.batches[mode])
            for band_name, _ in _action_band_slices(ACTION_BAND_ENDS[-1]):
                name = f"validation_gripper_band_{band_name}_rmse_physical"
                primary_rmse = primary[name]
                counterfactual_rmse = counterfactual[name]
                result[f"{stem}_primary_gripper_band_{band_name}_rmse_physical"] = (
                    primary_rmse
                )
                result[f"{stem}_gripper_band_{band_name}_rmse_physical"] = (
                    counterfactual_rmse
                )
                result[
                    f"{stem}_gripper_band_{band_name}_mse_gain_vs_primary_physical"
                ] = primary_rmse**2 - counterfactual_rmse**2
                for owner_name in ("arm", "gripper"):
                    delta_name = f"{mode}_{owner_name}_band_{band_name}"
                    result[
                        f"{stem}_{owner_name}_band_{band_name}_action_delta_rmse_physical"
                    ] = float(
                        (
                            self.delta_square_error[delta_name]
                            / max(self.delta_element_count[delta_name], 1)
                        )
                        .sqrt()
                        .item()
                    )
            for bin_name in ("1_2", "3_6", "7_plus"):
                name = f"validation_gripper_post_event_{bin_name}_rmse_physical"
                rows_name = f"validation_gripper_post_event_rows_{bin_name}"
                primary_rmse = primary[name]
                counterfactual_rmse = counterfactual[name]
                result[f"{stem}_primary_post_event_{bin_name}_rmse_physical"] = (
                    primary_rmse
                )
                result[f"{stem}_post_event_{bin_name}_rmse_physical"] = (
                    counterfactual_rmse
                )
                result[
                    f"{stem}_post_event_{bin_name}_mse_gain_vs_primary_physical"
                ] = primary_rmse**2 - counterfactual_rmse**2
                result[f"{stem}_post_event_rows_{bin_name}"] = counterfactual[
                    rows_name
                ]
            for suffix in (
                "event_precision",
                "event_recall",
                "event_f1",
                "event_ratio",
                "timing_mae_steps",
            ):
                source = f"validation_decoded_gripper_{suffix}"
                result[f"{stem}_decoded_gripper_{suffix}"] = counterfactual[source]
        if reference_primary is None:
            raise RuntimeError("matched P2 primary accounting is missing")
        for band_name, _ in _action_band_slices(ACTION_BAND_ENDS[-1]):
            result[
                f"validation_p2_intervention_primary_gripper_band_{band_name}_rmse_physical"
            ] = reference_primary[
                f"validation_gripper_band_{band_name}_rmse_physical"
            ]
        for bin_name in ("1_2", "3_6", "7_plus"):
            result[
                f"validation_p2_intervention_primary_post_event_{bin_name}_rmse_physical"
            ] = reference_primary[
                f"validation_gripper_post_event_{bin_name}_rmse_physical"
            ]
        for suffix in (
            "event_precision",
            "event_recall",
            "event_f1",
            "event_ratio",
            "timing_mae_steps",
        ):
            result[
                f"validation_p2_intervention_primary_decoded_gripper_{suffix}"
            ] = reference_primary[f"validation_decoded_gripper_{suffix}"]
        return result


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
            motion_logits=result.motion_logits,
            motion_target=motion_target,
            physical_field=result.physical_field,
            gripper_decode_delta_blend=model.action_codec.decode_delta_blend,
        )
    return accumulator.means()


__all__ = [
    "MatchedP2InterventionAccumulator",
    "ValidationAccumulator",
    "evaluate_loader",
]
