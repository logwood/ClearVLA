"""Training-only, source-owned correspondence pairs without camera-name guesses.

The generic representation has one explicit source/target view per group. It
never invents pairs, negatives, calibration, or physical instance identities.
The CALVIN producer continues to emit its versioned legacy pair format.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class IdentityPairGroup:
    kind: str
    source_time: int  # 0: previous observed frame; 1: current
    source_camera: int
    target_camera: int  # target time is current; never future
    source: Tensor  # [B,N,2] full-image endpoint coordinates
    target: Tensor
    valid: Tensor  # [B,N], independent label support
    provenance: str  # producer/version, not a physical-instance oracle

    def validate(self, *, batch: int, cameras: int, device: torch.device) -> None:
        if self.kind not in {"cross", "temporal"} or not self.provenance.strip():
            raise ValueError("correspondence needs an admitted kind and provenance")
        if type(self.source_time) is not int or self.source_time not in (0, 1):
            raise ValueError("correspondence cannot use a future source frame")
        if any(
            type(c) is not int or not 0 <= c < cameras
            for c in (self.source_camera, self.target_camera)
        ):
            raise ValueError("correspondence camera index is outside the declared source set")
        if self.kind == "cross" and (
            self.source_time != 1 or self.source_camera == self.target_camera
        ):
            raise ValueError("cross-view labels must link distinct current views")
        if self.kind == "temporal" and (
            self.source_time != 0 or self.source_camera != self.target_camera
        ):
            raise ValueError("temporal labels must link the declared past/current view")
        if (
            self.source.ndim != 3
            or self.source.shape[0] != batch
            or self.source.shape[-1] != 2
            or self.target.shape != self.source.shape
            or self.valid.shape != self.source.shape[:-1]
        ):
            raise ValueError("correspondence pair axes do not agree")
        if (
            self.source.dtype != torch.float32
            or self.target.dtype != torch.float32
            or self.valid.dtype != torch.bool
        ):
            raise ValueError("correspondence requires FP32 coordinates and Boolean support")
        for value in (self.source, self.target, self.valid):
            if value.requires_grad or value.device != device:
                raise ValueError("correspondence labels must be detached on the batch device")
        for value in (self.source, self.target):
            safe = torch.where(self.valid[..., None], value, 0.0)
            if not bool(torch.isfinite(safe).all()) or bool((safe.abs() > 1 + 1e-6).any()):
                raise ValueError("supported label escaped the full-image endpoint chart")


@dataclass(frozen=True)
class IdentityPairBatch:
    camera_names: tuple[str, ...]
    source_frames: Tensor
    groups: tuple[IdentityPairGroup, ...]

    def validate(self, *, batch: int, device: torch.device) -> None:
        if (
            not self.camera_names
            or len(set(self.camera_names)) != len(self.camera_names)
            or any(not n.strip() for n in self.camera_names)
        ):
            raise ValueError("correspondence requires distinct declared camera sources")
        if (
            self.source_frames.shape != (batch, 2)
            or self.source_frames.dtype != torch.long
            or self.source_frames.device != device
        ):
            raise ValueError("correspondence source clock must be int64 [B,2]")
        if bool((self.source_frames < 0).any()) or bool(
            (self.source_frames[:, 1] < self.source_frames[:, 0]).any()
        ):
            raise ValueError("correspondence frames must be nonnegative and causal")
        for group in self.groups:
            group.validate(batch=batch, cameras=len(self.camera_names), device=device)
            if group.kind == "temporal" and bool(
                (group.valid.any(1) & (self.source_frames[:, 0] == self.source_frames[:, 1])).any()
            ):
                raise ValueError("a duplicated reset frame cannot create temporal labels")

    def pair_groups(self) -> tuple[IdentityPairGroup, ...]:
        return self.groups


def validate_identity_clock(labels, online) -> None:
    """Identity labels must describe the same two images actually encoded.

    Absolute episode origins are loader-owned. Here we check the relative
    physical span; tensor length alone never establishes source time.
    """
    timing = online.history.timing
    if timing is None or timing.state_offsets.shape[1] < 2:
        raise ValueError("identity loss requires the online source clock")
    expected = timing.state_offsets[:, -2:]
    actual = labels.source_frames - labels.source_frames[:, -1:]
    if not torch.equal(actual, expected):
        raise ValueError("identity label source span differs from encoded image history")
