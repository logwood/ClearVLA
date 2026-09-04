"""Bottom-internal fixed B-spline view of the noisy physical action field.

The module deliberately keeps the original raw action lift outside this file.
It only constructs a parallel, time-aligned coarse/detail view that can be
added to the raw tokens before the existing Evidence-MMDiT blocks.
"""

from __future__ import annotations

import hashlib
import struct
from contextlib import nullcontext
from typing import Final

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...action_representations.bspline import BSplineSpec, build_basis_bundle

BSPINE_DISABLED_IMPLEMENTATION: Final[str] = "disabled"
BSPINE0_IMPLEMENTATION: Final[str] = "fixed_bspline_coarse_detail_v1"
BSPINE0_HORIZON: Final[int] = 24
BSPINE0_PHYSICAL_DIM: Final[int] = 18
BSPINE0_DEGREE: Final[int] = 3
BSPINE0_CONTROL_POINTS: Final[int] = 12
BSPINE0_BASIS_DIGEST: Final[str] = (
    "f4d169cdeab9606dfacb92abbbc71bc3dbb7a4abefb8ef5244bc411670caab34"
)
BSPINE0_SPEC_FINGERPRINT: Final[str] = (
    "a2234eb6c9f553c47e793e11c8734d8cfadfbaaf86c5b950dab8f672965a8c10"
)
_RUNTIME_BASIS_DIGEST_SCHEMA: Final[bytes] = b"clearvla-bspine0-runtime-basis-v1"


def _runtime_basis_digest(*, analysis: Tensor, synthesis: Tensor) -> str:
    """Hash exactly the FP32 operators used by the production B-spine.

    The standalone lossless representation also constructs an arbitrary
    orthogonal detail coordinate chart.  That chart is useful for serialized
    representation payloads, but its QR completion is not consumed by
    ``BSpine0`` and may legitimately differ across LAPACK backends.  The
    production identity therefore covers the unique coarse pseudoinverse and
    collocation matrices after their actual FP32 runtime cast.
    """

    digest = hashlib.sha256()
    digest.update(_RUNTIME_BASIS_DIGEST_SCHEMA)
    for name, source in (("analysis", analysis), ("synthesis", synthesis)):
        value = source.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if value.ndim != 2 or not bool(torch.isfinite(value).all()):
            raise ValueError(f"B-spine runtime {name} must be a finite matrix")
        digest.update(name.encode("ascii"))
        digest.update(struct.pack("<I", value.ndim))
        for dimension in value.shape:
            digest.update(struct.pack("<Q", int(dimension)))
        for scalar in value.reshape(-1).tolist():
            digest.update(struct.pack("<f", float(scalar)))
    return digest.hexdigest()


class _ZeroInitializedBiasFreeLinear(nn.Module):
    """A linear map whose construction consumes no random-number state."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = nn.Parameter(torch.zeros(self.out_features, self.in_features))
        self.register_parameter("bias", None)

    def forward(self, value: Tensor) -> Tensor:
        return F.linear(value.float(), self.weight.float())


class BSpine0(nn.Module):
    """Fixed FP32 analysis/synthesis with zero-initialized learned lifts.

    ``x_t`` remains in the canonical 18-D physical flow chart.  Every field
    role owns independent coarse and detail projections so no learned map can
    silently mix arm, gripper, or auxiliary meanings before the existing
    action stream.
    """

    def __init__(
        self,
        *,
        horizon: int,
        hidden_size: int,
        arm_dim: int,
        gripper_field_dim: int,
        degree: int,
        control_points: int,
        expected_basis_digest: str,
        expected_spec_fingerprint: str,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.hidden_size = int(hidden_size)
        self.arm_dim = int(arm_dim)
        self.gripper_field_dim = int(gripper_field_dim)
        self.degree = int(degree)
        self.control_points = int(control_points)
        self.physical_dim = 2 * self.arm_dim + self.gripper_field_dim
        if self.arm_dim <= 0 or self.gripper_field_dim < 2:
            raise ValueError("B-spine field roles require arm and gripper coordinates")
        if self.hidden_size <= 0:
            raise ValueError("B-spine hidden size must be positive")
        if (
            self.horizon != BSPINE0_HORIZON
            or self.physical_dim != BSPINE0_PHYSICAL_DIM
            or self.degree != BSPINE0_DEGREE
            or self.control_points != BSPINE0_CONTROL_POINTS
        ):
            raise ValueError(
                "B-spine-0 is fixed at horizon=24, physical_dim=18, "
                "degree=3 and control_points=12"
            )

        spec = BSplineSpec.uniform(
            horizon=self.horizon,
            arm_dim=self.physical_dim,
            num_control_points=self.control_points,
            degree=self.degree,
            mode="hierarchical_exact",
        )
        bundle = build_basis_bundle(spec)
        synthesis64 = bundle.coarse_collocation
        analysis64 = torch.linalg.solve_triangular(
            bundle.coarse_r,
            bundle.coarse_q.T,
            upper=True,
        )
        identity64 = torch.eye(self.control_points, dtype=torch.float64)
        closure_error = (analysis64 @ synthesis64 - identity64).abs().amax()
        if not bool(torch.isfinite(closure_error)) or float(closure_error) > 1.0e-10:
            raise ValueError("B-spine analysis/synthesis matrices do not close")
        analysis32 = analysis64.to(dtype=torch.float32)
        synthesis32 = synthesis64.to(dtype=torch.float32)
        runtime_basis_digest = _runtime_basis_digest(
            analysis=analysis32,
            synthesis=synthesis32,
        )
        if not expected_basis_digest:
            raise ValueError("enabled B-spine requires a serialized basis digest")
        if str(expected_basis_digest) != runtime_basis_digest:
            raise ValueError(
                "configured B-spine basis digest does not match the runtime operators"
            )
        if not expected_spec_fingerprint:
            raise ValueError("enabled B-spine requires a serialized spec fingerprint")
        if str(expected_spec_fingerprint) != spec.fingerprint:
            raise ValueError(
                "configured B-spine spec fingerprint does not match the constructed chart"
            )
        self.basis_digest = runtime_basis_digest
        self.spec_fingerprint = spec.fingerprint
        self.register_buffer(
            "analysis",
            analysis32,
            persistent=True,
        )
        self.register_buffer(
            "synthesis",
            synthesis32,
            persistent=True,
        )

        auxiliary_width = self.gripper_field_dim - 2
        widths = {
            "arm_absolute": self.arm_dim,
            "arm_delta": self.arm_dim,
            "gripper_value": 1,
            "gripper_delta": 1,
            "gripper_auxiliary": auxiliary_width,
        }
        self.coarse_lifts = nn.ModuleDict(
            {
                name: _ZeroInitializedBiasFreeLinear(width, self.hidden_size)
                for name, width in widths.items()
            }
        )
        self.detail_lifts = nn.ModuleDict(
            {
                name: _ZeroInitializedBiasFreeLinear(width, self.hidden_size)
                for name, width in widths.items()
            }
        )

        arm_stop = self.arm_dim
        delta_stop = 2 * self.arm_dim
        gripper_start = delta_stop
        self._role_slices = {
            "arm_absolute": slice(0, arm_stop),
            "arm_delta": slice(arm_stop, delta_stop),
            "gripper_value": slice(gripper_start, gripper_start + 1),
            "gripper_delta": slice(gripper_start + 1, gripper_start + 2),
            "gripper_auxiliary": slice(gripper_start + 2, self.physical_dim),
        }

    def _fp32_scope(self, reference: Tensor):
        if reference.device.type in {"cpu", "cuda"}:
            return torch.autocast(device_type=reference.device.type, enabled=False)
        return nullcontext()

    def decompose(self, physical: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return FP32 controls, coarse rows and the exact residual."""

        if tuple(physical.shape[1:]) != (self.horizon, self.physical_dim):
            raise ValueError(
                "B-spine expected physical field "
                f"[B,{self.horizon},{self.physical_dim}], got {tuple(physical.shape)}"
            )
        with self._fp32_scope(physical):
            value = physical.float()
            analysis = self.analysis.to(device=value.device)
            synthesis = self.synthesis.to(device=value.device)
            controls = torch.einsum("kt,btp->bkp", analysis, value)
            coarse = torch.einsum("tk,bkp->btp", synthesis, controls)
            detail = value - coarse
        return controls, coarse, detail

    def forward(
        self,
        physical: Tensor,
        *,
        zero_output: bool = False,
        collect_diagnostics: bool = False,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        controls, coarse, detail = self.decompose(physical)
        with self._fp32_scope(physical):
            synthesis = self.synthesis.to(device=physical.device)
            coarse_hidden_controls: Tensor | None = None
            detail_hidden: Tensor | None = None
            for name, field_slice in self._role_slices.items():
                coarse_role = self.coarse_lifts[name](controls[..., field_slice])
                detail_role = self.detail_lifts[name](detail[..., field_slice])
                coarse_hidden_controls = (
                    coarse_role
                    if coarse_hidden_controls is None
                    else coarse_hidden_controls + coarse_role
                )
                detail_hidden = (
                    detail_role if detail_hidden is None else detail_hidden + detail_role
                )
            if coarse_hidden_controls is None or detail_hidden is None:
                raise RuntimeError("B-spine field-role chart is empty")
            coarse_tokens = torch.einsum(
                "tk,bkh->bth",
                synthesis,
                coarse_hidden_controls,
            )
            unablated = coarse_tokens + detail_hidden
            tokens = torch.zeros_like(unablated) if zero_output else unablated
        tokens = tokens.to(dtype=physical.dtype)
        if not collect_diagnostics:
            return tokens, {}
        reconstruction = coarse + detail
        metrics = {
            "bottom_spine_coarse_field_rms": coarse.detach().square().mean().sqrt(),
            "bottom_spine_detail_field_rms": detail.detach().square().mean().sqrt(),
            "bottom_spine_coarse_token_rms": (coarse_tokens.detach().square().mean().sqrt()),
            "bottom_spine_detail_token_rms": (detail_hidden.detach().square().mean().sqrt()),
            "bottom_spine_update_rms": unablated.detach().square().mean().sqrt(),
            "bottom_spine_decomposition_max_abs": (
                reconstruction.detach() - physical.detach().float()
            )
            .abs()
            .amax(),
            "bottom_spine_zero_intervention_active": physical.new_tensor(
                float(zero_output), dtype=torch.float32
            ),
        }
        return tokens, metrics


__all__ = [
    "BSPINE_DISABLED_IMPLEMENTATION",
    "BSPINE0_BASIS_DIGEST",
    "BSPINE0_CONTROL_POINTS",
    "BSPINE0_DEGREE",
    "BSPINE0_HORIZON",
    "BSPINE0_IMPLEMENTATION",
    "BSPINE0_PHYSICAL_DIM",
    "BSPINE0_SPEC_FINGERPRINT",
    "BSpine0",
]
