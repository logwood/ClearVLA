"""Training-only correspondence labels with explicit source clocks/charts."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class IdentityCorrespondence:
    cross_source: Tensor  # B,C,N,2 endpoint coordinates, source camera c
    cross_target: Tensor  # target camera is 1-c, no fabricated common XY chart
    cross_valid: Tensor
    temporal_source: Tensor  # past-to-current in camera c
    temporal_target: Tensor
    temporal_valid: Tensor
    source_frames: Tensor  # B,2 absolute source rows: past/current
    camera_names: tuple[str, ...] = ("top", "wrist")

    def validate(self, *, batch, device):
        if self.camera_names != ("top", "wrist") or self.source_frames.shape != (batch, 2):
            raise ValueError("identity labels lost their calibrated camera/source clock")
        if self.source_frames.dtype != torch.long or self.source_frames.device != device:
            raise ValueError("identity source clock must be int64 on the label device")
        if bool((self.source_frames[:, 1] < self.source_frames[:, 0]).any()):
            raise ValueError("identity temporal labels cannot reverse the observed source clock")
        for prefix in ("cross", "temporal"):
            source = getattr(self, prefix + "_source")
            target = getattr(self, prefix + "_target")
            valid = getattr(self, prefix + "_valid")
            if (
                source.ndim != 4
                or source.shape[:2] != (batch, 2)
                or source.shape[-1] != 2
                or target.shape != source.shape
                or valid.shape != source.shape[:-1]
            ):
                raise ValueError("identity labels require full pair/camera/coordinate axes")
            if (
                source.dtype != torch.float32
                or target.dtype != torch.float32
                or valid.dtype != torch.bool
            ):
                raise ValueError(
                    "identity coordinates require FP32 and independent Boolean support"
                )
            for value in (source, target, valid):
                if value.requires_grad or value.device != device:
                    raise ValueError("identity labels must be detached on the label device")
            for value in (source, target):
                safe = torch.where(valid[..., None], value, 0.0)
                if not bool(torch.isfinite(safe).all()) or bool((safe.abs() > 1 + 1e-6).any()):
                    raise ValueError("supported correspondence escaped its endpoint image chart")
        if bool(
            (
                self.temporal_valid.any((1, 2))
                & (self.source_frames[:, 1] == self.source_frames[:, 0])
            ).any()
        ):
            raise ValueError("padded duplicate frame cannot manufacture temporal evidence")

    def pair_groups(self):
        """Loss adapter only; retain the legacy two-camera label contract."""
        from .identity_pairs import IdentityPairGroup

        return tuple(
            IdentityPairGroup(
                kind,
                time,
                camera,
                1 - camera if kind == "cross" else camera,
                getattr(self, kind + "_source")[:, camera],
                getattr(self, kind + "_target")[:, camera],
                getattr(self, kind + "_valid")[:, camera],
                "legacy-calvin-rgbd-temporal-v1",
            )
            for kind, time in (("cross", 1), ("temporal", 0))
            for camera in range(2)
        )
