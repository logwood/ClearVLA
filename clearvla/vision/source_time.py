"""Source-time semantics for cached causal images and observed optical motion.

A repeated reset image is the same measurement, not an extra four-step motion
example. This record is built from source offsets, never from image content,
model confidence, future labels or an ODE iteration counter.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

FIXED_VISUAL_TIME = "fixed_history_steps_v1"
SOURCE_VISUAL_TIME = "source_history_steps_v1"


def pair_mask(value: Tensor, support: Tensor | None) -> Tensor:
    """Mask the leading pair axis before arithmetic; None preserves legacy."""
    if support is None:
        return value
    if support.dtype != torch.bool or tuple(support.shape) != (value.shape[0],):
        raise ValueError("observed pair support must be Boolean [N]")
    shape = (support.shape[0],) + (1,) * (value.ndim - 1)
    return torch.where(support.reshape(shape), value, torch.zeros_like(value))


def pair_mean(value: Tensor, support: Tensor | None) -> Tensor:
    """Mean over observed pairs and all their cells; empty support is zero."""
    if support is None:
        return value.mean()
    safe = pair_mask(value, support)
    per_pair = value[0].numel()
    denominator = (support.sum().to(dtype=torch.float32) * per_pair).clamp_min(1.0)
    return safe.float().sum() / denominator


def displacement_rate(displacement: Tensor, steps: Tensor) -> Tensor:
    """Convert observed displacement to control-step rate, not metres/second."""
    validate_reference_steps(steps, batch=int(displacement.shape[0]), device=displacement.device)
    shape = (steps.shape[0],) + (1,) * (displacement.ndim - 1)
    known = steps.reshape(shape) > 0
    safe = torch.where(known, displacement, torch.zeros_like(displacement))
    return safe / steps.clamp_min(1).reshape(shape).to(dtype=displacement.dtype)


def validate_reference_steps(steps: Tensor, *, batch: int, device: torch.device) -> None:
    if steps.dtype != torch.long or tuple(steps.shape) != (batch,) or steps.device != device:
        raise ValueError("flow reference duration must be integer source steps [B]")


@dataclass(frozen=True)
class VisualSourceTime:
    """Actual frame offsets. Duplicate source rows share the latest copy.

    With reset padding, [-1,-1,0] means two copies of the known reset image
    followed by the current image. Temporal evidence counts the reset image
    once, and the last flow pair spans one physical control step, not four.
    """
    frame_offsets: Tensor

    def validate(self, *, batch: int, frames: int, device: torch.device, strict: bool = False) -> None:
        if self.frame_offsets.dtype != torch.long or tuple(self.frame_offsets.shape) != (batch, frames):
            raise ValueError("visual source offsets must be integer [B,T]")
        if frames < 2 or batch < 1 or self.frame_offsets.device != device:
            raise ValueError("visual source offsets have an invalid batch/frame/device boundary")
        if strict:
            if bool((self.frame_offsets[:, -1] != 0).any()) or bool((self.pair_steps < 0).any()):
                raise ValueError("visual history must be causal, ordered and end at the current source")

    @property
    def pair_steps(self) -> Tensor:
        return self.frame_offsets[:, 1:] - self.frame_offsets[:, :-1]

    @property
    def pair_observed(self) -> Tensor:
        return self.pair_steps > 0

    @property
    def frame_observed(self) -> Tensor:
        return torch.cat((self.pair_observed, torch.ones_like(self.frame_offsets[:, -1:], dtype=torch.bool)), dim=1)

    def canonicalize(self, value: Tensor) -> Tensor:
        """Gather only the latest physical copy, quarantining padded payloads.

        This also aligns context-dropout masks across copies of one frame. A
        masked current cell cannot reappear through its reset-padding alias.
        """
        if tuple(value.shape[:2]) != tuple(self.frame_offsets.shape) or value.device != self.frame_offsets.device:
            raise ValueError("visual payload and source clock must share [B,T] and device")
        rows = torch.arange(value.shape[1], device=value.device)
        same = self.frame_offsets[:, :, None] == self.frame_offsets[:, None, :]
        latest = torch.where(same, rows[None, None], -1).amax(dim=-1)
        index = latest.reshape(*latest.shape, *((1,) * (value.ndim - 2))).expand_as(value)
        return value.gather(1, index)

    def memory_observed(self, *, cameras: int, grid: int) -> Tensor:
        frames = self.frame_observed[:, :, None].expand(-1, -1, cameras * grid * grid).flatten(1)
        pairs = self.pair_observed[:, :, None].expand(-1, -1, cameras * grid * grid).flatten(1)
        return torch.cat((frames, pairs), dim=1)


def visual_time_metadata(mode: str) -> dict[str, object]:
    if mode not in (FIXED_VISUAL_TIME, SOURCE_VISUAL_TIME):
        raise ValueError("unknown visual source-time mode")
    return {
        "contract": mode,
        "unit": "physical_control_step",
        "seconds_per_step": None,
        "duplicate_source": "latest_copy_once",
        "pair_support": "strictly_positive_source_gap",
        "motion_rate": "image_chart_displacement_per_control_step",
        "teacher_transport": "future_source_steps_over_actual_observed_pair_steps",
        "state_update": "new_physical_observation_only_not_solver_node",
    }
