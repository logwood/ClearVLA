"""Active-semantic logging without ancestry aliases or per-batch CUDA sync."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor

ACTIVE_PREFIXES = (
    "loss_",
    "gradient_",
    "condition_",
    "observation_",
    "grounding_",
    "object_grounding_",
    "object_intent_",
    "object_plan_",
    "object_coarse_",
    "object_teacher_",
    "object_w_",
    "object_w1_",
    "object_w2_",
    "object_future_",
    "p1_",
    "object_p2_",
    "object_consequence_",
    "object_p3_",
    "bottom_",
    "validation_",
    "runtime_",
)


def active_metrics(values: Mapping[str, float], *, zero_tolerance: float = 0.0) -> dict[str, float]:
    """Keep interpretable active metrics and suppress inactive exact zeros."""

    result: dict[str, float] = {}
    for name, value in values.items():
        if not name.startswith(ACTIVE_PREFIXES) and name not in {
            "learning_rate",
            "gradient_global_preclip_l2",
        }:
            continue
        scalar = float(value)
        if abs(scalar) <= float(zero_tolerance) and name not in {
            "loss_ledger_gap",
        }:
            continue
        result[name] = scalar
    return result


@dataclass
class MetricAccumulator:
    sums: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)

    def update(self, values: Mapping[str, float], *, weight: float = 1.0) -> None:
        for name, value in values.items():
            self.sums[name] = self.sums.get(name, 0.0) + float(value) * float(weight)
            self.weights[name] = self.weights.get(name, 0.0) + float(weight)

    def means(self) -> dict[str, float]:
        return {
            name: value / max(self.weights.get(name, 0.0), 1e-12)
            for name, value in self.sums.items()
        }


@dataclass
class DeviceMetricAccumulator:
    """Accumulate detached scalars with one vector update per key signature.

    A Python loop that updates one CUDA tensor per metric avoids host
    synchronization but still launches tens of tiny kernels twice per train
    batch (window and epoch ledgers).  Active loss keys are stable on ordinary
    batches and diagnostics add only a second signature, so vector ownership
    removes that hidden logging tax without moving values to the CPU.
    """

    sums: dict[tuple[str, ...], Tensor] = field(default_factory=dict)
    weights: dict[tuple[str, ...], float] = field(default_factory=dict)

    def update(self, values: Mapping[str, Tensor], *, weight: float = 1.0) -> None:
        scalar_weight = float(weight)
        scalar_rows = sorted(
            (name, value.detach().float()) for name, value in values.items() if value.ndim == 0
        )
        if not scalar_rows:
            return
        names = tuple(name for name, _ in scalar_rows)
        vector = torch.stack([value for _, value in scalar_rows])
        if names in self.sums:
            self.sums[names] = self.sums[names] + vector * scalar_weight
        else:
            self.sums[names] = vector * scalar_weight
        self.weights[names] = self.weights.get(names, 0.0) + scalar_weight

    def materialize(self) -> dict[str, float]:
        scalar_sums: dict[str, Tensor] = {}
        scalar_weights: dict[str, float] = {}
        for names, vector in self.sums.items():
            weight = self.weights[names]
            for index, name in enumerate(names):
                value = vector[index]
                scalar_sums[name] = scalar_sums.get(name, value.new_zeros(())) + value
                scalar_weights[name] = scalar_weights.get(name, 0.0) + weight
        return tensor_scalars(
            {name: value / max(scalar_weights[name], 1e-12) for name, value in scalar_sums.items()}
        )


class JsonlRunLogger:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / "metrics.jsonl"

    def write(self, kind: str, **payload: object) -> None:
        row = {"kind": str(kind), **payload}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")

    @staticmethod
    def compact_line(
        kind: str,
        *,
        epoch: int,
        batch: int | None,
        step: int,
        metrics: Mapping[str, float],
    ) -> str:
        lead = f"[mainline-{kind}] epoch={epoch:03d}"
        if batch is not None:
            lead += f" batch={batch:04d}"
        lead += f" step={step}"
        priority = (
            "loss_total",
            "loss_action_flow",
            "loss_future_dynamics",
            "loss_future_successor",
            "loss_future_semantic_delta",
            "loss_future_address",
            "loss_object_reconstruction",
            "object_grounding_object_content_pair_cosine",
            "object_intent_interval_variation",
            "object_w_intent_object_interaction_rms",
            "object_w_action_object_interaction_rms",
            "object_w2_interval_adjacent_cosine",
            "object_w2_object_pair_cosine",
            "action_flow_balanced_band_1_4",
            "action_flow_balanced_band_5_12",
            "action_flow_balanced_band_13_24",
            "action_gripper_event_flow",
            "action_gripper_hold_flow",
            "validation_action_rmse_normalized",
            "validation_action_rmse_physical",
            "validation_first_rmse_normalized",
            "validation_tail_rmse_normalized",
            "gradient_global_preclip_l2",
            "learning_rate",
        )
        fields = []
        for name in priority:
            if name in metrics:
                fields.append(f"{name}={metrics[name]:.6g}")
        return " ".join((lead, *fields))


def tensor_scalars(values: Mapping[str, Tensor]) -> dict[str, float]:
    """Materialize detached scalars with one transfer per source device.

    Calling ``bool(isfinite(cuda_scalar))`` and then ``scalar.cpu()`` for
    every metric creates two synchronization points per key.  Logging owns a
    synchronization boundary, but it should be one vector boundary rather
    than dozens of serial device round trips.
    """

    grouped: dict[torch.device, list[tuple[str, Tensor]]] = {}
    for name, value in values.items():
        if value.ndim == 0:
            grouped.setdefault(value.device, []).append((name, value.detach().float()))
    result: dict[str, float] = {}
    for rows in grouped.values():
        names = [name for name, _ in rows]
        vector = torch.stack([value for _, value in rows]).cpu()
        finite = torch.isfinite(vector)
        for index, name in enumerate(names):
            if bool(finite[index]):
                result[name] = float(vector[index])
    return result


__all__ = [
    "ACTIVE_PREFIXES",
    "DeviceMetricAccumulator",
    "JsonlRunLogger",
    "MetricAccumulator",
    "active_metrics",
    "tensor_scalars",
]
