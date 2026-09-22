"""Observable task expectations and source-owned direct outcome supervision.

S predicts what demonstrations tend to change for the instruction. W predicts
what a particular candidate action does. Neither prediction is measured progress.
No learned target probability, future reliability or predicted value masks a loss.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..future_time import CONTROL_ALIGNED_FUTURE_TIME, resolve_future_time
from ..operation_expectation import OperationExpectation, operation_expectation_metadata
from ..supervision import FutureLabelSupport, quarantine, supported_mean
from .target_binding import TargetBinding
from .types import FutureObjectDynamics, ObjectFactSet


class ObjectOperationPredictor(nn.Module):
    """All K outcomes keep identity; binding applies only at subsequent value reads."""

    role_basis: Tensor

    def __init__(
        self, *, hidden: int, content_dim: int, state_dim: int, camera_names: tuple[str, ...]
    ) -> None:
        super().__init__()
        operation_expectation_metadata(camera_names)
        self.camera_names = camera_names
        self.content = nn.Linear(content_dim, hidden, bias=False)
        self.state = nn.Linear(state_dim, hidden, bias=False)
        self.task = nn.Linear(hidden, hidden, bias=False)
        self.object_model = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.SiLU()
        )
        self.semantic = nn.Linear(hidden, content_dim, bias=False)
        self.coordinates = nn.Linear(2, hidden, bias=False)
        self.view_role = nn.Linear(len(camera_names), hidden, bias=False)
        self.view_model = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.SiLU())
        self.image = nn.Linear(hidden, 2, bias=False)
        self.robot_model = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, state_dim, bias=False),
        )
        names = sorted(camera_names)
        self.register_buffer(
            "role_basis",
            torch.eye(len(names))[[names.index(n) for n in camera_names]],
            persistent=False,
        )

    def forward(
        self, *, task_intervals: Tensor, facts: ObjectFactSet, state: Tensor, binding: TargetBinding
    ) -> OperationExpectation:
        b, k, _ = facts.content.shape
        if task_intervals.shape != (
            b,
            4,
            self.task.in_features,
        ) or facts.camera_coordinates.shape != (b, k, len(self.camera_names), 2):
            raise ValueError("operation expectation input axes do not match S/G")
        valid = (facts.validity[..., 0] > 0)[..., None] & (facts.camera_validity[..., 0] > 0)
        valid = valid.detach()
        dtype = self.task.weight.dtype
        task = self.task(task_intervals.to(dtype)) + self.state(state.to(dtype))[:, None]
        content = self.content(torch.where(valid.any(-1)[..., None], facts.content, 0.0).to(dtype))
        objects = self.object_model(task[:, :, None] + content[:, None])
        semantic = torch.where(valid.any(-1)[:, None, :, None], self.semantic(objects).float(), 0.0)
        coordinates = self.coordinates(
            torch.where(valid[..., None], facts.camera_coordinates, 0.0).to(dtype)
        )
        role = self.view_role(self.role_basis.to(dtype))[None, None]
        view = self.view_model(objects[:, :, :, None] + (coordinates + role)[:, None])
        image = torch.where(valid[:, None, :, :, None], self.image(view).float(), 0.0)
        result = OperationExpectation(
            semantic,
            image,
            self.robot_model(task).float(),
            valid,
            binding,
            state,
            facts.content,
            self.camera_names,
        )
        result.validate()
        return result


class OperationExpectationRead(nn.Module):
    """Zero-preserving typed reads. Camera axes have meaning before reduction."""

    role_basis: Tensor

    def __init__(
        self, *, hidden: int, content_dim: int, state_dim: int, camera_names: tuple[str, ...]
    ) -> None:
        super().__init__()
        operation_expectation_metadata(camera_names)
        self.camera_names = camera_names
        self.semantic = nn.Linear(content_dim, hidden, bias=False)
        self.image = nn.Linear(2, hidden, bias=False)
        self.robot = nn.Linear(state_dim, hidden, bias=False)
        self.view_role = nn.Linear(len(camera_names), hidden, bias=False)
        names = sorted(camera_names)
        self.register_buffer(
            "role_basis",
            torch.eye(len(names))[[names.index(n) for n in camera_names]],
            persistent=False,
        )

    def forward(self, value: OperationExpectation) -> Tensor:
        value.validate()
        if value.camera_names != self.camera_names:
            raise ValueError("operation expectation read camera chart mismatch")
        dtype = self.semantic.weight.dtype
        valid = value.view_observed
        sem = self.semantic(
            torch.where(valid.any(-1)[:, None, :, None], value.semantic_delta, 0.0).to(dtype)
        )
        image = self.image(
            torch.where(valid[:, None, :, :, None], value.image_delta, 0.0).to(dtype)
        )
        # No learned confidence or cross-object softmax; null remains in the law.
        weight = valid.float() / valid.sum(-1, keepdim=True).clamp_min(1)
        role = 1.0 + torch.tanh(self.view_role(self.role_basis.to(dtype))).float()
        image_value = (image.float() * role[None, None, None] * weight[:, None, :, :, None]).sum(3)
        mass = value.binding.mass[:, None, :, None]
        target = ((sem.float() + image_value) * mass).sum(2)
        # The robot value is its own chart; for a null operated-object read,
        # no task expectation is asserted from this branch either.
        robot = self.robot(value.robot_delta.to(dtype)).float() * mass.sum(2)
        return (target + robot).to(dtype)


class OperationExpectationPlanRead(nn.Module):
    row_weights: Tensor

    def __init__(
        self, *, hidden: int, content_dim: int, state_dim: int, camera_names: tuple[str, ...]
    ) -> None:
        super().__init__()
        self.values = OperationExpectationRead(
            hidden=hidden, content_dim=content_dim, state_dim=state_dim, camera_names=camera_names
        )
        self.context = nn.Linear(hidden, hidden, bias=False)
        self.output = nn.Linear(hidden, hidden, bias=False)
        self.register_buffer(
            "row_weights",
            resolve_future_time(CONTROL_ALIGNED_FUTURE_TIME).row_interpolation(),
            persistent=True,
        )

    def forward(self, value: OperationExpectation, context: Tensor) -> Tensor:
        if (
            context.ndim != 4
            or context.shape[0] != value.current_state.shape[0]
            or context.shape[1] != 24
            or context.shape[2] < 1
            or context.shape[-1] != self.context.in_features
            or context.device != value.current_state.device
        ):
            raise ValueError("expected operation plan needs matching [B,24,Q,H] context")
        interval = self.values(value)
        rows = torch.einsum("ti,bih->bth", self.row_weights.to(interval), interval)[:, :, None]
        # These are intent features interpolated for plan use, NOT interpolated
        # physical endpoints or an error against a newly observed image.
        return self.output(rows * (1.0 + torch.tanh(self.context(context))))


class OperationExpectationSupervisor(nn.Module):
    """No learned target encoder/decoder: targets remain declared source measurements."""

    def __init__(self) -> None:
        super().__init__()
        self.time_grid = resolve_future_time(CONTROL_ALIGNED_FUTURE_TIME)

    def forward(
        self,
        *,
        expectation: OperationExpectation,
        teacher: FutureObjectDynamics,
        current_loss_support: Tensor,
        future_state: Tensor,
        label_support: FutureLabelSupport | None = None,
        future_interval_valid: Tensor | None = None,
    ) -> dict[str, Tensor]:
        expectation.validate()
        teacher.validate()
        b, _, k, _ = expectation.semantic_delta.shape
        c = len(expectation.camera_names)
        if (
            teacher.time_grid_mode != self.time_grid.mode
            or teacher.camera_names != expectation.camera_names
        ):
            raise ValueError("operation expectation/Teacher time or camera chart mismatch")
        if (
            teacher.semantic_delta.shape != expectation.semantic_delta.shape
            or teacher.transport_mean.shape != expectation.image_delta.shape
        ):
            raise ValueError("operation supervision lost typed object/view axes")
        if current_loss_support.shape != (b, k, c, 1):
            raise ValueError("operation supervision needs current source-owned [B,K,C,1] support")
        # Source availability is boolean, not probability-weighted by binding,
        # visibility estimates, error magnitude, or Teacher match confidence.
        current = current_loss_support.detach()[..., 0] > 0
        if not torch.equal(current, expectation.view_observed):
            raise ValueError("operation supervision current observation support differs")
        iv = torch.ones((b, 4), device=future_state.device, dtype=torch.bool)
        if label_support is not None:
            # The optional convenience API must not assume complete visual
            # labels merely because the caller omitted its precomputed mask.
            offsets = torch.tensor(
                self.time_grid.support_offsets, device=future_state.device, dtype=torch.long
            )[None].expand(b, -1)
            iv = self.time_grid.interval_observed(offsets, label_support.visual)
        if future_interval_valid is not None:
            if future_interval_valid.shape != (b, 4) or future_interval_valid.dtype != torch.bool:
                raise ValueError("operation future observation support must be bool [B,4]")
            iv = iv & future_interval_valid
        visual_mask = current[:, None] & iv[:, :, None, None]
        semantic_mask = visual_mask.any(-1)
        if (
            future_state.ndim != 3
            or future_state.shape[0] != b
            or future_state.shape[-1] != expectation.current_state.shape[-1]
        ):
            raise ValueError("operation robot target must be a declared state-feature sequence")
        state = future_state.detach().float()
        slices = self.time_grid.action_slices(state.shape[1])
        robot_valid = torch.ones((b, 4), device=state.device, dtype=torch.bool)
        if label_support is not None:
            state = quarantine(state, label_support.state)
            robot_valid = torch.stack([label_support.state[:, row].all(1) for row in slices], 1)
        robot_target = (
            torch.stack([state[:, row].mean(1) for row in slices], 1)
            - expectation.current_state.detach().float()[:, None]
        )

        def loss(pred: Tensor, target: Tensor, valid: Tensor) -> Tensor:
            pred, target = (
                quarantine(pred.float(), valid),
                quarantine(target.detach().float(), valid),
            )
            if not bool(torch.isfinite(target).all()) or not bool(torch.isfinite(pred).all()):
                raise ValueError("nonfinite operation value on supervised source support")
            return supported_mean(F.smooth_l1_loss(pred, target, reduction="none"), valid)

        semantic = loss(expectation.semantic_delta, teacher.semantic_delta, semantic_mask)
        image = loss(expectation.image_delta, teacher.transport_mean, visual_mask)
        robot = loss(expectation.robot_delta, robot_target, robot_valid)
        return {
            "operation_semantic": semantic,
            "operation_image": image,
            "operation_robot": robot,
            "operation_total": (semantic + image + robot) / 3.0,
            "operation_semantic_count": semantic_mask.float().sum().detach(),
            "operation_image_count": visual_mask.float().sum().detach(),
            "operation_robot_count": robot_valid.float().sum().detach(),
        }
