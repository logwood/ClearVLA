"""Read-only finite-gradient and window diagnostics for the mainline trainer.

These helpers own no optimizer state and never modify gradients.  The
parameter-level scan runs only after an already-computed finite global norm
crosses the configured spike boundary.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable

import torch
from torch import Tensor, nn

from .optimizer import parameter_role

DEFAULT_GRADIENT_SPIKE_AUDIT_THRESHOLD = 5.0


@dataclass(frozen=True)
class GradientParameterMaximum:
    """Identity and magnitudes for one raw-gradient tensor."""

    parameter_name: str
    parameter_role: str
    optimizer_group: str
    shape: tuple[int, ...]
    dtype: str
    l2: float
    max_abs: float


@dataclass(frozen=True)
class FiniteGradientSpikeReport:
    """JSON-safe attribution of one finite global pre-clip spike."""

    gradient_global_preclip_l2: float
    gradient_spike_audit_threshold: float
    max_l2: GradientParameterMaximum
    max_abs: GradientParameterMaximum
    flow_delta_head_channel_l2: tuple[float, ...] | None = None

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "gradient_global_preclip_l2": float(
                self.gradient_global_preclip_l2
            ),
            "gradient_spike_audit_threshold": float(
                self.gradient_spike_audit_threshold
            ),
        }
        for prefix, row in (("max_l2", self.max_l2), ("max_abs", self.max_abs)):
            for name, value in asdict(row).items():
                result[f"{prefix}_{name}"] = value
        if self.flow_delta_head_channel_l2 is not None:
            for index, value in enumerate(self.flow_delta_head_channel_l2):
                result[f"flow_delta_head_channel_{index}_l2"] = float(value)
            result["flow_delta_head_flow_channels_l2"] = math.sqrt(
                sum(value * value for value in self.flow_delta_head_channel_l2[:2])
            )
            result["flow_delta_head_uncertainty_channels_l2"] = math.sqrt(
                sum(value * value for value in self.flow_delta_head_channel_l2[2:])
            )
        return result


def build_finite_gradient_spike_report(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
    *,
    global_norm: float,
    audit_threshold: float,
    optimizer_group_name: Callable[[str], str],
) -> FiniteGradientSpikeReport:
    """Attribute a finite spike without changing any parameter gradient."""

    rows: list[tuple[str, nn.Parameter, Tensor]] = []
    scores: list[Tensor] = []
    for name, parameter in named_parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        detached = gradient.detach()
        rows.append((name, parameter, detached))
        float_gradient = detached.float()
        scores.append(
            torch.stack(
                (
                    torch.linalg.vector_norm(float_gradient.reshape(-1), ord=2),
                    float_gradient.abs().amax(),
                )
            )
        )
    if not rows:
        raise RuntimeError("finite gradient spike audit found no parameter gradients")

    host_scores = torch.stack(scores).detach().cpu()
    if not bool(torch.isfinite(host_scores).all()):
        raise RuntimeError(
            "finite global gradient spike produced non-finite owner scores"
        )
    max_l2_index = int(host_scores[:, 0].argmax().item())
    max_abs_index = int(host_scores[:, 1].argmax().item())

    def describe(index: int) -> GradientParameterMaximum:
        name, _parameter, gradient = rows[index]
        try:
            role = parameter_role(name)
        except ValueError:
            # The report is also used by focused diagnostic tests and future
            # modules before optimizer role tables are extended.  Attribution
            # must still name the exact tensor instead of masking the spike.
            role = "unowned"
        return GradientParameterMaximum(
            parameter_name=name,
            parameter_role=role,
            optimizer_group=optimizer_group_name(name),
            shape=tuple(int(value) for value in gradient.shape),
            dtype=str(gradient.dtype).removeprefix("torch."),
            l2=float(host_scores[index, 0].item()),
            max_abs=float(host_scores[index, 1].item()),
        )

    owner_name, _owner_parameter, owner_gradient = rows[max_l2_index]
    observation_owner = owner_name.startswith("observation.") or (
        ".observation." in owner_name
    )
    owns_flow_channels = (
        observation_owner
        and "delta_head" in owner_name
        and owner_gradient.ndim >= 1
        and int(owner_gradient.shape[0]) == 6
    )
    if owns_flow_channels:
        channel_l2 = (
            owner_gradient.float()
            .reshape(6, -1)
            .square()
            .sum(dim=1)
            .sqrt()
            .cpu()
        )
        flow_delta_head_channel_l2 = tuple(
            float(value) for value in channel_l2.tolist()
        )
    else:
        flow_delta_head_channel_l2 = None

    return FiniteGradientSpikeReport(
        gradient_global_preclip_l2=float(global_norm),
        gradient_spike_audit_threshold=float(audit_threshold),
        max_l2=describe(max_l2_index),
        max_abs=describe(max_abs_index),
        flow_delta_head_channel_l2=flow_delta_head_channel_l2,
    )


@dataclass
class GradientPreclipWindowAccumulator:
    """Host-side mean/max/current ownership for one logging window."""

    weighted_sum: float = 0.0
    weight: float = 0.0
    maximum: float = -math.inf
    current: float = math.nan
    maximum_batch_offset: int = 0
    maximum_global_step: int = 0

    def update(
        self,
        value: float,
        *,
        weight: float,
        batch_offset: int,
        global_step: int,
    ) -> None:
        scalar = float(value)
        scalar_weight = float(weight)
        if not math.isfinite(scalar):
            raise ValueError("pre-clip gradient window value must be finite")
        if not math.isfinite(scalar_weight) or scalar_weight <= 0.0:
            raise ValueError(
                "pre-clip gradient window weight must be finite and positive"
            )
        if int(batch_offset) <= 0 or int(global_step) < 0:
            raise ValueError("pre-clip gradient window indices are invalid")
        self.weighted_sum += scalar * scalar_weight
        self.weight += scalar_weight
        self.current = scalar
        if scalar > self.maximum:
            self.maximum = scalar
            self.maximum_batch_offset = int(batch_offset)
            self.maximum_global_step = int(global_step)

    def materialize(self) -> dict[str, float]:
        if self.weight <= 0.0 or not math.isfinite(self.current):
            raise RuntimeError("cannot materialize an empty pre-clip gradient window")
        return {
            "gradient_window_preclip_l2_mean": self.weighted_sum / self.weight,
            "gradient_window_preclip_l2_max": self.maximum,
            "gradient_window_preclip_l2_current": self.current,
            "gradient_window_preclip_l2_max_batch_offset": float(
                self.maximum_batch_offset
            ),
            "gradient_window_preclip_l2_max_global_step": float(
                self.maximum_global_step
            ),
        }


__all__ = [
    "DEFAULT_GRADIENT_SPIKE_AUDIT_THRESHOLD",
    "FiniteGradientSpikeReport",
    "GradientParameterMaximum",
    "GradientPreclipWindowAccumulator",
    "build_finite_gradient_spike_report",
]
