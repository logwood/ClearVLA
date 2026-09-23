"""Already executed controls and observed-world error; never future labels.

Observation o[t] precedes a[t]. Four is the first existing aligned W endpoint,
not a task timer. Unknown pre-reset transitions are explicitly unsupported.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from .config import ExperimentConfig
    from .model.target_binding import TargetBinding
    from .model.types import ExecutedActionSequenceCondition, ObjectFactSet

EXECUTED_WORLD_FEEDBACK = "executed_four_step_world_v1"


def executed_world_metadata() -> dict[str, object]:
    return {
        "schema": "causal-executed-world-comparison-v1",
        "mode": EXECUTED_WORLD_FEEDBACK,
        "anchor": -4,
        "commands": [-4, -3, -2, -1],
        "source_visual_offsets": [-12, -8, -4],
        "history_retention": 13,
        "prediction": "shared-G-and-W1-replayed-from-past-and-confirmed-controls-no-current-image",
        "measurement": "same-frozen-soft-DINO-observation-law-as-W-labels",
        "measurement_grid": "existing-pooled-normalized-W-grid-not-native-token-resolution",
        "values": "matched-observation-minus-first-W-endpoint-with-explicit-null-status",
        "association": "current-G3-image-support-overlap-plus-soft-K-plus-null-no-slot-identity",
        "gradients": "detached-replay-measurement-and-image-laws-P3-reader-plus-existing-target-selector",
        "clock": "physical-control-steps-not-ODE-or-instruction-age",
        "cost": "one-extra-past-G-and-W1-once-per-online-encoding-no-extra-W2-or-ODE-call",
        "claim": "retrodictive-source-aligned-feature-error-not-contact-or-goal-progress",
    }


@dataclass(frozen=True)
class ExecutedWorldWindow:
    dino_history: Tensor  # B,3,C,N,D
    raw_rgb: Tensor  # B,3,C,3,R,R
    state: Tensor  # B,S at t-4
    action_state: Tensor  # B,A at t-4
    commands: Tensor  # B,4,A acknowledged canonical input
    observed: Tensor  # bool B, entire executed prefix known
    visual_offsets: Tensor  # int64 B,3, relative to t-4

    def validate(
        self, config: ExperimentConfig, *, batch: int, device: torch.device, strict: bool = False
    ) -> None:
        d = config.dimensions
        if self.dino_history.shape != (
            batch,
            3,
            d.num_cameras,
            d.patches_per_camera,
            d.visual_token_dim,
        ):
            raise ValueError("executed W source must retain the complete past DINO chart")
        if self.raw_rgb.ndim != 6 or self.raw_rgb.shape[:4] != (batch, 3, d.num_cameras, 3):
            raise ValueError("executed W RGB needs the complete past three-frame chart")
        if (
            self.raw_rgb.shape[-1] != self.raw_rgb.shape[-2]
            or self.raw_rgb.shape[-1] < 32
            or self.raw_rgb.shape[-1] % 16
        ):
            raise ValueError("executed W RGB preprocessing chart differs")
        if self.state.shape != (batch, d.state_dim) or self.action_state.shape != (
            batch,
            d.action_dim,
        ):
            raise ValueError("executed W past robot state/command chart differs")
        if self.commands.shape != (batch, 4, d.action_dim):
            raise ValueError("executed W requires every command of the actual four-step prefix")
        if self.observed.shape != (batch,) or self.observed.dtype != torch.bool:
            raise ValueError("executed W support must be bool [B]")
        if self.visual_offsets.shape != (batch, 3) or self.visual_offsets.dtype != torch.long:
            raise ValueError("executed W source time must be int64 [B,3]")
        tensors = (self.dino_history, self.raw_rgb, self.state, self.action_state, self.commands)
        if any(not x.is_floating_point() for x in tensors):
            raise TypeError("executed W source payload must be floating point")
        if any(x.device != device for x in (*tensors, self.observed, self.visual_offsets)):
            raise ValueError("executed W source tensors must share online device")
        if strict:
            off = self.visual_offsets
            if bool((off[:, -1] != 0).any()) or bool((off[:, 1:] < off[:, :-1]).any()):
                raise ValueError("executed W visual times must be causal and ordered")
            if (
                bool((off[:, 0] < -8).any())
                or bool((off[:, 1] < -4).any())
                or bool((off > 0).any())
            ):
                raise ValueError("executed W visual times escape the past source window")
            if not torch.equal(off[:, 1], off[:, 0].clamp_min(-4)):
                raise ValueError(
                    "executed W visual padding does not describe its declared sparse source"
                )
            for x in tensors:
                mask = self.observed.reshape(batch, *([1] * (x.ndim - 1)))
                if not bool(torch.isfinite(torch.where(mask, x, 0.0)).all()):
                    raise ValueError("nonfinite observed executed W source")

    def without_actions(self, keep: Tensor) -> ExecutedWorldWindow:
        if keep.shape != self.observed.shape or keep.device != self.observed.device:
            raise ValueError("executed W dropout must follow the same batch")
        return replace(self, observed=self.observed & keep.bool())


@dataclass(frozen=True)
class ExecutedWorldPrediction:
    """Only first existing endpoint; unknown later controls are not predictions."""

    semantic: Tensor
    image: Tensor
    covariance: Tensor
    source: ExecutedActionSequenceCondition
    observed: Tensor


@dataclass(frozen=True)
class ExecutedWorldFeedback:
    semantic: Tensor  # detached FP32 B,Kold,D
    image: Tensor  # B,Kold,C,2
    covariance: Tensor  # B,Kold,C,3; not a calibrated bound
    null: Tensor  # B,Kold,1
    posterior: Tensor  # B,Kold,C,Q; actual W measurement grid
    past_content: Tensor  # B,Kold,D
    view_observed: Tensor  # bool B,Kold,C
    window: ExecutedWorldWindow
    current_dino: Tensor
    camera_names: tuple[str, ...]
    measurement_shape: tuple[int, int]

    def validate(self, *, strict: bool = False) -> None:
        if self.semantic.ndim != 3 or self.current_dino.ndim != 5:
            raise ValueError("executed W comparison lost content/source chart")
        if len(self.measurement_shape) != 2 or any(
            type(n) is not int or n <= 0 for n in self.measurement_shape
        ):
            raise ValueError("executed W measurement shape must be two positive integers")
        b, k, d = self.semantic.shape
        c = len(self.camera_names)
        if self.image.shape != (b, k, c, 2) or self.covariance.shape != (b, k, c, 3):
            raise ValueError("executed W errors lost object/view charts")
        if self.null.shape != (b, k, 1) or self.past_content.shape != (b, k, d):
            raise ValueError("executed W status/content lost past-object identity")
        if self.posterior.shape != (b, k, c, self.measurement_shape[0] * self.measurement_shape[1]):
            raise ValueError("executed W observation lost declared measurement support")
        if self.view_observed.shape != (b, k, c) or self.view_observed.dtype != torch.bool:
            raise ValueError("executed W source support must remain boolean")
        for t in (
            self.semantic,
            self.image,
            self.covariance,
            self.null,
            self.posterior,
            self.past_content,
        ):
            if t.requires_grad or t.device != self.current_dino.device or t.dtype != torch.float32:
                raise ValueError("executed W measured/replayed values must be detached FP32")
        if strict:
            if (
                self.view_observed.device != self.current_dino.device
                or self.window.observed.shape != (b,)
            ):
                raise ValueError("executed W support has another source")
            if bool((self.view_observed & ~self.window.observed[:, None, None]).any()):
                raise ValueError("executed W invented a pre-reset observation")
            supported = self.view_observed.any(-1)
            for value, mask in (
                (self.semantic, supported),
                (self.past_content, supported),
                (self.image, self.view_observed),
                (self.covariance, self.view_observed),
                (self.posterior, self.view_observed),
                (self.null, supported),
            ):
                m = mask.reshape(*mask.shape, *([1] * (value.ndim - mask.ndim)))
                if not bool(torch.isfinite(torch.where(m, value, 0.0)).all()):
                    raise ValueError("nonfinite supported executed W comparison")
            p = torch.where(self.view_observed[..., None], self.posterior, 0.0)
            null = torch.where(supported[..., None], self.null, 1.0)
            if bool((p < 0).any()) or bool(((null < 0) | (null > 1)).any()):
                raise ValueError("executed W soft association has invalid mass")
            mass = p.sum((-2, -1)) + null[..., 0]
            if not torch.allclose(mass, torch.ones_like(mass), atol=2e-5, rtol=0):
                raise ValueError("executed W real/null measure is not normalized")


@dataclass(frozen=True)
class ExecutedWorldPlanValues:
    value: Tensor
    feedback: ExecutedWorldFeedback
    current_facts: ObjectFactSet
    reader_identity: int
    binding: TargetBinding
