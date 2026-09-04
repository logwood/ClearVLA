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
    # The mode is part of the validation ABI, not a property inferred from a
    # particular batch.  Keeping it on the accumulator lets matched
    # counterfactual reports use the same CALVIN command semantics even when
    # they intentionally do not carry the primary head logits.
    gripper_output_mode: str = "continuous"
    arm_flow_mode: str = "legacy_independent"
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
        gripper_output_mode: str = "continuous",
        arm_flow_mode: str = "legacy_independent",
    ) -> "ValidationAccumulator":
        """Build source-native accounting from the exact training chart.

        For an affine chart ``normalized = source_native * scale + offset``,
        the source-native prediction error is ``normalized_error / scale``.
        Legacy ``*_physical`` metric names remain compatibility aliases only.
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
        mode = str(gripper_output_mode)
        if mode not in {"continuous", "calvin_binary_command"}:
            raise ValueError(f"unknown validation gripper output mode {mode!r}")
        arm_mode = str(arm_flow_mode)
        if arm_mode not in {"legacy_independent", "relative_command_direct"}:
            raise ValueError(f"unknown validation arm flow mode {arm_mode!r}")
        return cls(
            action_scale=scale,
            action_offset=offset,
            gripper_event_threshold=float(gripper_event_threshold),
            arm_motion_threshold=float(arm_motion_threshold),
            gripper_output_mode=mode,
            arm_flow_mode=arm_mode,
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
        gripper_command_logits: Tensor | None = None,
        gripper_command: Tensor | None = None,
        gripper_output_mode: str | None = None,
        row_indices: Tensor | None = None,
    ) -> None:
        target = batch.action_target.normalized.float()
        normalized_current = batch.online.history.action_state.float()
        normalized_codec_gripper_boundary = (
            batch.online.history.codec_gripper_boundary.float()
        )
        raw_target = batch.action_target.raw_units.float()
        raw_current = batch.action_target.current_raw_units.float()
        raw_codec_gripper_boundary = (
            batch.action_target.gripper_transition_boundary_raw_units[..., -1:].float()
        )
        expected_boundary_shape = (int(prediction.shape[0]), 1)
        if tuple(normalized_codec_gripper_boundary.shape) != expected_boundary_shape:
            raise ValueError("normalized codec gripper boundary must be [B,1]")
        if tuple(raw_codec_gripper_boundary.shape) != expected_boundary_shape:
            raise ValueError("source-native codec gripper boundary must be [B,1]")
        if row_indices is not None:
            rows = row_indices.detach().to(device="cpu", dtype=torch.long)
            if rows.ndim != 1 or not rows.numel():
                raise ValueError("validation row selection must be a non-empty flat vector")
            if int(rows.min()) < 0 or int(rows.max()) >= int(prediction.shape[0]):
                raise IndexError("validation row selection is outside the batch")
            if int(torch.unique(rows).numel()) != int(rows.numel()):
                raise ValueError("validation row selection cannot contain duplicates")
            device_rows = rows.to(device=prediction.device)
            prediction = prediction.index_select(0, device_rows)
            target = target.index_select(0, device_rows)
            normalized_current = normalized_current.index_select(0, device_rows)
            normalized_codec_gripper_boundary = (
                normalized_codec_gripper_boundary.index_select(0, device_rows)
            )
            raw_target = raw_target.index_select(0, device_rows)
            raw_current = raw_current.index_select(0, device_rows)
            raw_codec_gripper_boundary = raw_codec_gripper_boundary.index_select(
                0, device_rows
            )
            if motion_logits is not None:
                motion_logits = motion_logits.index_select(0, device_rows)
            if motion_target is not None:
                motion_target = motion_target.index_select(0, device_rows)
            if physical_field is not None:
                physical_field = physical_field.index_select(0, device_rows)
            if gripper_command_logits is not None:
                gripper_command_logits = gripper_command_logits.index_select(
                    0, device_rows
                )
            if gripper_command is not None:
                gripper_command = gripper_command.index_select(0, device_rows)
        output_mode = (
            self.gripper_output_mode
            if gripper_output_mode is None
            else str(gripper_output_mode)
        )
        if output_mode not in {"continuous", "calvin_binary_command"}:
            raise ValueError(f"unknown validation gripper output mode {output_mode!r}")
        if output_mode == "calvin_binary_command":
            command_target = (raw_target[..., -1] >= 0.0).to(dtype=torch.long)
            if gripper_command_logits is not None:
                expected_command_shape = (
                    int(prediction.shape[0]),
                    int(prediction.shape[1]),
                    2,
                )
                if tuple(gripper_command_logits.shape) != expected_command_shape:
                    raise ValueError(
                        "validation gripper command logits must be [B,T,2], got "
                        f"{tuple(gripper_command_logits.shape)}"
                    )
                command_prediction = gripper_command_logits.detach().float().argmax(dim=-1)
            else:
                # Matched intervention paths intentionally retain only their
                # decoded action.  Their binary command is still unambiguous,
                # so infer the class from the strict +/-1 action alphabet.
                command_prediction = (prediction.detach().float()[..., -1] >= 0.0).to(
                    dtype=torch.long
                )
            if gripper_command is not None:
                if tuple(gripper_command.shape) != tuple(command_prediction.shape):
                    raise ValueError("validation gripper command has the wrong shape")
                command_float = gripper_command.float()
                if not torch.isfinite(command_float).all() or not (
                    (command_float == -1.0) | (command_float == 1.0)
                ).all():
                    raise ValueError("validation gripper command must contain only {-1,+1}")
                expected_command = command_prediction.float().mul(2.0).sub(1.0)
                if not torch.equal(
                    gripper_command.detach().float().cpu(), expected_command.cpu()
                ):
                    raise ValueError("validation command tensor disagrees with logits")
            command_value = command_prediction.float().mul(2.0).sub(1.0)
            command_raw_error = command_value - raw_target[..., -1]
            command_correct = (command_prediction == command_target).float()
            command_tp = (
                (command_prediction == 1) & (command_target == 1)
            ).float().sum()
            command_fp = (
                (command_prediction == 1) & (command_target == 0)
            ).float().sum()
            command_fn = (
                (command_prediction == 0) & (command_target == 1)
            ).float().sum()
            self._add_scalar("gripper_command_correct", command_correct.sum())
            self._add_scalar(
                "gripper_command_rows",
                prediction.new_tensor(float(command_target.numel())),
            )
            self._add_scalar(
                "gripper_command_predicted_positive",
                (command_prediction == 1).float().sum(),
            )
            self._add_scalar(
                "gripper_command_target_positive",
                (command_target == 1).float().sum(),
            )
            self._add_scalar("gripper_command_true_positive", command_tp)
            self._add_scalar("gripper_command_false_positive", command_fp)
            self._add_scalar("gripper_command_false_negative", command_fn)
            self._add_scalar(
                "gripper_command_square_error",
                command_raw_error.square().sum(),
            )
        elif gripper_command_logits is not None or gripper_command is not None:
            raise ValueError(
                "continuous validation cannot consume a gripper command-state tensor"
            )
        metric_prediction = prediction.float()
        if output_mode == "calvin_binary_command":
            # ``SamplingResult.action`` keeps the strict command alphabet in
            # its final slot for the bridge, whereas the rest of this
            # accumulator expects normalized action coordinates.  Re-encode
            # the command for RMSE surfaces, and decode it back to raw units
            # below for event timing and command-specific accounting.
            metric_prediction = metric_prediction.clone()
            if self.action_scale is not None and self.action_offset is not None:
                metric_prediction[..., -1] = (
                    command_value * self.action_scale[..., -1]
                    + self.action_offset[..., -1]
                )
            else:
                metric_prediction[..., -1] = command_value
        error = metric_prediction - target
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
        normalized_boundary_gap = (
            normalized_codec_gripper_boundary - normalized_current[..., -1:]
        )
        source_native_boundary_gap = (
            raw_codec_gripper_boundary - raw_current[..., -1:]
        )
        boundary_rows = error.new_tensor(float(normalized_boundary_gap.numel()))
        self._add_scalar(
            "codec_gripper_boundary_qpos_gap_square_normalized",
            normalized_boundary_gap.square().sum(),
        )
        self._add_scalar(
            "codec_gripper_boundary_qpos_gap_square_source_native",
            source_native_boundary_gap.square().sum(),
        )
        self._add_scalar("codec_gripper_boundary_rows", boundary_rows)
        action_state_gripper_abs = normalized_current[..., -1].abs()
        self._add_scalar(
            "action_state_gripper_abs_gt3",
            (action_state_gripper_abs > 3.0).float().sum(),
        )
        self._add_scalar(
            "action_state_gripper_abs_gt5",
            (action_state_gripper_abs > 5.0).float().sum(),
        )
        self._add_scalar(
            "action_state_gripper_rows",
            error.new_tensor(float(action_state_gripper_abs.numel())),
        )
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
                normalized_codec_gripper_boundary.detach()[:, None]
                + torch.cumsum(gripper_field[..., 1:2], dim=1)
            )
            reconstructed = (1.0 - blend) * absolute_branch + blend * cumulative_branch
            if output_mode == "continuous":
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
        if self.action_scale is not None:
            if self.action_offset is None:
                raise ValueError("validation action offset is missing")
            raw_prediction = (metric_prediction - self.action_offset) / self.action_scale
        else:
            raw_prediction = metric_prediction
            raw_target = target
            raw_current = normalized_current
            raw_codec_gripper_boundary = normalized_codec_gripper_boundary
        if output_mode == "calvin_binary_command":
            raw_prediction = raw_prediction.clone()
            raw_prediction[..., -1] = command_value
        first_boundary = torch.cat(
            (raw_current[..., :-1], raw_codec_gripper_boundary), dim=-1
        )
        target_boundary = torch.cat((first_boundary[:, None], raw_target[:, :-1]), dim=1)
        pred_boundary = torch.cat(
            (first_boundary[:, None], raw_prediction[:, :-1]), dim=1
        )
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
        if self.arm_flow_mode == "relative_command_direct":
            # CALVIN rows are already relative TCP commands.  Their magnitude
            # is the motion signal; differencing adjacent commands would turn
            # this audit back into the retired acceleration chart.
            decoded_motion_target = target[..., :-1].norm(dim=-1) >= float(
                self.arm_motion_threshold
            )
            decoded_motion = metric_prediction[..., :-1].norm(dim=-1) >= float(
                self.arm_motion_threshold
            )
        else:
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
        boundary_rows = self.scalar_totals["codec_gripper_boundary_rows"].clamp_min(
            1.0
        )
        action_state_gripper_rows = self.scalar_totals[
            "action_state_gripper_rows"
        ].clamp_min(1.0)
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
            "validation_codec_gripper_boundary_qpos_gap_rms_normalized": (
                self.scalar_totals[
                    "codec_gripper_boundary_qpos_gap_square_normalized"
                ]
                / boundary_rows
            ).sqrt(),
            "validation_codec_gripper_boundary_qpos_gap_rms_source_native": (
                self.scalar_totals[
                    "codec_gripper_boundary_qpos_gap_square_source_native"
                ]
                / boundary_rows
            ).sqrt(),
            "validation_action_state_gripper_abs_gt3_rate_normalized": (
                self.scalar_totals["action_state_gripper_abs_gt3"]
                / action_state_gripper_rows
            ),
            "validation_action_state_gripper_abs_gt5_rate_normalized": (
                self.scalar_totals["action_state_gripper_abs_gt5"]
                / action_state_gripper_rows
            ),
        }
        if "gripper_command_rows" in self.scalar_totals:
            command_rows = self.scalar_totals["gripper_command_rows"].clamp_min(1.0)
            command_correct = self.scalar_totals["gripper_command_correct"]
            command_predicted_positive = self.scalar_totals[
                "gripper_command_predicted_positive"
            ]
            command_target_positive = self.scalar_totals[
                "gripper_command_target_positive"
            ]
            command_tp = self.scalar_totals["gripper_command_true_positive"]
            command_fp = self.scalar_totals["gripper_command_false_positive"]
            command_fn = self.scalar_totals["gripper_command_false_negative"]
            command_precision = command_tp / (command_tp + command_fp).clamp_min(1.0)
            command_recall = command_tp / (command_tp + command_fn).clamp_min(1.0)
            command_f1 = (
                2.0 * command_precision * command_recall
                / (command_precision + command_recall).clamp_min(1e-8)
            )
            tensors.update(
                {
                    "validation_gripper_command_accuracy": command_correct
                    / command_rows,
                    "validation_gripper_command_predicted_positive_rate": (
                        command_predicted_positive / command_rows
                    ),
                    "validation_gripper_command_positive_rate": (
                        command_predicted_positive / command_rows
                    ),
                    "validation_gripper_command_target_positive_rate": (
                        command_target_positive / command_rows
                    ),
                    "validation_gripper_command_precision": command_precision,
                    "validation_gripper_command_recall": command_recall,
                    "validation_gripper_command_f1": command_f1,
                    "validation_gripper_command_rows": command_rows,
                }
            )
            if "gripper_command_square_error" in self.scalar_totals:
                tensors["validation_gripper_command_rmse_physical"] = (
                    self.scalar_totals["gripper_command_square_error"]
                    / command_rows
                ).sqrt()
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
        # Historical logs called the de-normalized HDF5/source chart
        # "physical" even when the producer's SI units were unknown.  Emit the
        # truthful name alongside the old key so dashboards remain readable.
        tensors.update(
            {
                f"{name[:-len('_physical')]}_source_native": value
                for name, value in tuple(tensors.items())
                if name.endswith("_physical")
            }
        )
        return tensor_scalars(tensors)


@dataclass
class MatchedP2InterventionAccumulator:
    """Paired action/error accounting for P2 value/address counterfactuals."""

    action_scale: Tensor
    action_offset: Tensor
    gripper_event_threshold: float
    arm_motion_threshold: float
    gripper_output_mode: str = "continuous"
    arm_flow_mode: str = "legacy_independent"
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
        gripper_output_mode: str = "continuous",
        arm_flow_mode: str = "legacy_independent",
    ) -> "MatchedP2InterventionAccumulator":
        base = ValidationAccumulator.from_action_normalizer(
            normalizer,
            device=device,
            gripper_event_threshold=gripper_event_threshold,
            arm_motion_threshold=arm_motion_threshold,
            gripper_output_mode=gripper_output_mode,
            arm_flow_mode=arm_flow_mode,
        )
        if base.action_scale is None or base.action_offset is None:
            raise RuntimeError("matched P2 accounting requires the physical action chart")
        return cls(
            action_scale=base.action_scale,
            action_offset=base.action_offset,
            gripper_event_threshold=float(gripper_event_threshold),
            arm_motion_threshold=float(arm_motion_threshold),
            gripper_output_mode=str(gripper_output_mode),
            arm_flow_mode=str(arm_flow_mode),
        )

    def _new_validation_accumulator(self) -> ValidationAccumulator:
        return ValidationAccumulator(
            action_scale=self.action_scale,
            action_offset=self.action_offset,
            gripper_event_threshold=self.gripper_event_threshold,
            arm_motion_threshold=self.arm_motion_threshold,
            gripper_output_mode=self.gripper_output_mode,
            arm_flow_mode=self.arm_flow_mode,
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


@dataclass
class MatchedCoreAttributionAccumulator:
    """Matched Schema28 W/consequence/CT responsibility accounting.

    The primary refined action is accumulated once per selected validation
    batch.  Every counterfactual then reuses that same primary action, refined
    cache and initial noise while this object records only decision-making
    scalars; no action/tensor dump is retained.
    """

    action_scale: Tensor
    action_offset: Tensor
    gripper_event_threshold: float
    arm_motion_threshold: float
    gripper_output_mode: str = "continuous"
    arm_flow_mode: str = "legacy_independent"
    primary: ValidationAccumulator | None = None
    counterfactual: dict[str, ValidationAccumulator] = field(default_factory=dict)
    delta_square_error: dict[str, Tensor] = field(default_factory=dict)
    delta_element_count: dict[str, int] = field(default_factory=dict)
    boundary_totals: dict[str, Tensor] = field(default_factory=dict)
    boundary_weights: dict[str, int] = field(default_factory=dict)
    boundary_maxima: dict[str, Tensor] = field(default_factory=dict)
    identity_square_error: dict[str, Tensor] = field(default_factory=dict)
    identity_element_count: dict[str, int] = field(default_factory=dict)
    identity_maxima: dict[str, Tensor] = field(default_factory=dict)
    primary_batches: int = 0
    batches: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_action_normalizer(
        cls,
        normalizer: ArrayNormalizer,
        *,
        device: torch.device,
        gripper_event_threshold: float,
        arm_motion_threshold: float,
        gripper_output_mode: str = "continuous",
        arm_flow_mode: str = "legacy_independent",
    ) -> "MatchedCoreAttributionAccumulator":
        base = ValidationAccumulator.from_action_normalizer(
            normalizer,
            device=device,
            gripper_event_threshold=gripper_event_threshold,
            arm_motion_threshold=arm_motion_threshold,
            gripper_output_mode=gripper_output_mode,
            arm_flow_mode=arm_flow_mode,
        )
        if base.action_scale is None or base.action_offset is None:
            raise RuntimeError("core attribution requires the physical action chart")
        result = cls(
            action_scale=base.action_scale,
            action_offset=base.action_offset,
            gripper_event_threshold=float(gripper_event_threshold),
            arm_motion_threshold=float(arm_motion_threshold),
            gripper_output_mode=str(gripper_output_mode),
            arm_flow_mode=str(arm_flow_mode),
        )
        result.primary = result._new_validation_accumulator()
        return result

    def _new_validation_accumulator(self) -> ValidationAccumulator:
        return ValidationAccumulator(
            action_scale=self.action_scale,
            action_offset=self.action_offset,
            gripper_event_threshold=self.gripper_event_threshold,
            arm_motion_threshold=self.arm_motion_threshold,
            gripper_output_mode=self.gripper_output_mode,
            arm_flow_mode=self.arm_flow_mode,
        )

    def update_primary(self, action: Tensor, batch: TrainingBatch) -> None:
        if self.primary is None:
            raise RuntimeError("core-attribution primary accumulator is missing")
        self.primary.update(action, batch)
        self.primary_batches += 1

    def _add_delta(self, key: str, value: Tensor) -> None:
        update = value.detach().float().square().sum()
        self.delta_square_error[key] = self.delta_square_error.get(
            key, update.new_zeros(())
        ) + update
        self.delta_element_count[key] = self.delta_element_count.get(key, 0) + int(
            value.numel()
        )

    def update(
        self,
        mode: str,
        *,
        primary_action: Tensor,
        counterfactual_action: Tensor,
        batch: TrainingBatch,
        boundary_metrics: Mapping[str, Tensor] | None = None,
    ) -> None:
        if self.primary_batches <= self.batches.get(mode, 0):
            raise ValueError(
                "core attribution requires one primary update before every mode update"
            )
        accumulator = self.counterfactual.get(mode)
        if accumulator is None:
            accumulator = self._new_validation_accumulator()
            self.counterfactual[mode] = accumulator
        accumulator.update(counterfactual_action, batch)
        delta = (
            counterfactual_action.detach().float() - primary_action.detach().float()
        ) / self.action_scale
        self._add_delta(f"{mode}_action", delta)
        self._add_delta(f"{mode}_arm", delta[..., :-1])
        self._add_delta(f"{mode}_gripper", delta[..., -1:])
        for band_name, band_slice in _action_band_slices(int(delta.shape[1])):
            self._add_delta(f"{mode}_band_{band_name}", delta[:, band_slice])
            self._add_delta(
                f"{mode}_gripper_band_{band_name}",
                delta[:, band_slice, -1:],
            )
        batch_weight = int(primary_action.shape[0])
        for name, value in (boundary_metrics or {}).items():
            scalar = value.detach().float()
            if scalar.ndim != 0:
                raise ValueError("core-attribution boundary metrics must be scalar")
            key = f"{mode}_{name}"
            if name.endswith("_max_abs"):
                previous = self.boundary_maxima.get(key)
                self.boundary_maxima[key] = (
                    scalar if previous is None else torch.maximum(previous, scalar)
                )
            elif name.endswith(("_rows", "_batches")):
                self.boundary_totals[key] = self.boundary_totals.get(
                    key, scalar.new_zeros(())
                ) + scalar
                self.boundary_weights[key] = 1
            else:
                self.boundary_totals[key] = self.boundary_totals.get(
                    key, scalar.new_zeros(())
                ) + scalar * batch_weight
                self.boundary_weights[key] = self.boundary_weights.get(key, 0) + batch_weight
        self.batches[mode] = self.batches.get(mode, 0) + 1

    def update_identity(self, name: str, left: Tensor, right: Tensor) -> None:
        if tuple(left.shape) != tuple(right.shape):
            raise ValueError("core-attribution identity tensors must align")
        normalized = left.detach().float() - right.detach().float()
        physical = normalized / self.action_scale
        for chart, value in (("normalized", normalized), ("physical", physical)):
            key = f"{name}_{chart}"
            update = value.square().sum()
            self.identity_square_error[key] = self.identity_square_error.get(
                key, update.new_zeros(())
            ) + update
            self.identity_element_count[key] = self.identity_element_count.get(
                key, 0
            ) + int(value.numel())
            maximum = value.abs().amax()
            previous = self.identity_maxima.get(key)
            self.identity_maxima[key] = (
                maximum if previous is None else torch.maximum(previous, maximum)
            )

    @staticmethod
    def _rmse_surfaces(values: Mapping[str, float]) -> dict[str, float]:
        result = {
            "action": values["validation_action_rmse_physical"],
            "arm": values["validation_arm_rmse_physical"],
            "gripper": values["validation_gripper_rmse_physical"],
        }
        for band_name, _ in _action_band_slices(ACTION_BAND_ENDS[-1]):
            result[f"band_{band_name}"] = values[
                f"validation_band_{band_name}_rmse_physical"
            ]
            result[f"gripper_band_{band_name}"] = values[
                f"validation_gripper_band_{band_name}_rmse_physical"
            ]
        return result

    def means(self) -> dict[str, float]:
        if self.primary is None or self.primary_batches <= 0:
            raise ValueError("core attribution did not consume a primary batch")
        if not self.batches:
            raise ValueError("core attribution did not consume any counterfactual")
        incomplete = {
            mode: count
            for mode, count in self.batches.items()
            if count != self.primary_batches
        }
        if incomplete:
            raise ValueError(
                "core-attribution mode coverage is incomplete: " + repr(incomplete)
            )
        primary = self.primary.means()
        primary_surfaces = self._rmse_surfaces(primary)
        result: dict[str, float] = {
            "validation_core_attribution_primary_batches": float(self.primary_batches)
        }
        for surface, rmse in primary_surfaces.items():
            result[
                f"validation_core_attribution_primary_{surface}_rmse_physical"
            ] = rmse
        for suffix in (
            "event_precision",
            "event_recall",
            "event_f1",
            "events_predicted",
            "events_target",
            "event_ratio",
            "timing_mae_steps",
        ):
            result[
                f"validation_core_attribution_primary_decoded_gripper_{suffix}"
            ] = primary[f"validation_decoded_gripper_{suffix}"]
        for mode in sorted(self.batches):
            counterfactual = self.counterfactual[mode].means()
            surfaces = self._rmse_surfaces(counterfactual)
            stem = f"validation_core_attribution_{mode}"
            result[f"{stem}_batches"] = float(self.batches[mode])
            for surface, rmse in surfaces.items():
                primary_rmse = primary_surfaces[surface]
                result[f"{stem}_{surface}_rmse_physical"] = rmse
                result[f"{stem}_{surface}_mse_gain_vs_primary_physical"] = (
                    primary_rmse**2 - rmse**2
                )
                delta_key = f"{mode}_{surface}"
                delta_name = (
                    "action_delta_rmse_physical"
                    if surface == "action"
                    else f"{surface}_action_delta_rmse_physical"
                )
                result[f"{stem}_{delta_name}"] = float(
                    (
                        self.delta_square_error[delta_key]
                        / max(self.delta_element_count[delta_key], 1)
                    )
                    .sqrt()
                    .item()
                )
            for suffix in (
                "event_precision",
                "event_recall",
                "event_f1",
                "events_predicted",
                "events_target",
                "event_ratio",
                "timing_mae_steps",
            ):
                result[f"{stem}_decoded_gripper_{suffix}"] = counterfactual[
                    f"validation_decoded_gripper_{suffix}"
                ]
        for key, total in self.boundary_totals.items():
            result[f"validation_core_attribution_{key}"] = float(
                (total / max(self.boundary_weights[key], 1)).item()
            )
        for key, maximum in self.boundary_maxima.items():
            result[f"validation_core_attribution_{key}"] = float(maximum.item())
        for key, square_error in self.identity_square_error.items():
            stem = f"validation_core_attribution_{key}"
            result[f"{stem}_action_delta_rms"] = float(
                (
                    square_error / max(self.identity_element_count[key], 1)
                )
                .sqrt()
                .item()
            )
            maximum = float(self.identity_maxima[key].item())
            result[f"{stem}_action_max_abs"] = maximum
            result[f"{stem}_bit_exact"] = float(maximum == 0.0)
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
        ValidationAccumulator(
            gripper_output_mode=config.bottom.gripper_output_mode,
            arm_flow_mode=config.bottom.arm_flow_mode,
        )
        if action_normalizer is None
        else ValidationAccumulator.from_action_normalizer(
            action_normalizer,
            device=model_device,
            gripper_event_threshold=config.objectives.gripper_event_threshold,
            arm_motion_threshold=config.objectives.arm_motion_threshold,
            gripper_output_mode=config.bottom.gripper_output_mode,
            arm_flow_mode=config.bottom.arm_flow_mode,
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
        motion_target = (
            model.outlet_adapter.arm_motion_magnitude(
                batch.action_target.normalized,
                batch.online.history.action_state,
            )
            >= float(config.objectives.arm_motion_threshold)
        )
        accumulator.update(
            result.action,
            batch,
            motion_logits=result.motion_logits,
            motion_target=motion_target,
            physical_field=result.physical_field,
            gripper_decode_delta_blend=model.outlet_adapter.decode_delta_blend,
            gripper_command_logits=result.gripper_command_logits,
            gripper_command=result.gripper_command,
            gripper_output_mode=config.bottom.gripper_output_mode,
        )
    return accumulator.means()


__all__ = [
    "MatchedCoreAttributionAccumulator",
    "MatchedP2InterventionAccumulator",
    "ValidationAccumulator",
    "evaluate_loader",
]
