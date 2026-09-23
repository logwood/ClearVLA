"""No-grad object-aligned multi-frame future teacher for the mainline."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ..future_time import LEGACY_FUTURE_TIME, resolve_future_time
from ..supervision import supported_mean
from .observation_association import ObjectObservationAssociation
from .types import (
    FutureObjectDynamics,
    ObjectFactSet,
)


class ObjectFutureTeacher(ObjectObservationAssociation):
    """Training-only interval aggregation over the frozen observation law.

    Future labels enter only forward. The inherited observation measurement
    primitive also supports separately admitted already-observed comparisons.
    """

    def __init__(
        self,
        *,
        content_dim: int,
        future_time_grid_mode: str = LEGACY_FUTURE_TIME,
        key_dim: int = 64,
        flow_reference_frames: int = 4,
        camera_names: tuple[str, ...] = (),
    ) -> None:
        grid = resolve_future_time(future_time_grid_mode)
        super().__init__(
            content_dim=content_dim,
            key_dim=key_dim,
            flow_reference_frames=flow_reference_frames,
            camera_names=camera_names,
        )
        self.time_grid = grid

    @torch.no_grad()
    def forward(
        self,
        *,
        facts: ObjectFactSet,
        future_supports: Tensor,
        future_offsets: Tensor,
        future_observed: Tensor | None = None,
        collect_diagnostics: bool = True,
    ) -> tuple[FutureObjectDynamics, dict[str, Tensor]]:
        # Teacher association is a target-construction plane, not an online
        # BF16 value path.  An outer training autocast would otherwise cast
        # Linear/einsum outputs back to BF16 even though their operands were
        # explicitly converted with ``.float()``.  Keep semantic similarity,
        # the 129-way softmax, moments, and exported targets in FP32.
        if future_supports.device.type in {"cpu", "cuda"}:
            with torch.autocast(
                device_type=future_supports.device.type,
                enabled=False,
            ):
                return self._forward_fp32(
                    facts=facts,
                    future_supports=future_supports,
                    future_offsets=future_offsets,
                    future_observed=future_observed,
                    collect_diagnostics=collect_diagnostics,
                )
        return self._forward_fp32(
            facts=facts,
            future_supports=future_supports,
            future_offsets=future_offsets,
            future_observed=future_observed,
            collect_diagnostics=collect_diagnostics,
        )

    def _forward_fp32(
        self,
        *,
        facts: ObjectFactSet,
        future_supports: Tensor,
        future_offsets: Tensor,
        future_observed: Tensor | None = None,
        collect_diagnostics: bool,
    ) -> tuple[FutureObjectDynamics, dict[str, Tensor]]:
        if self.time_grid.aligned:
            self.time_grid.validate_offsets(
                self._offsets(
                    future_offsets.to(future_supports.device),
                    batch=int(future_supports.shape[0]),
                    supports=int(future_supports.shape[1]),
                )
            )
        measurement = self.measure_observations(
            facts=facts,
            observations=future_supports,
            relative_offsets=future_offsets,
            observed=future_observed,
        )
        batch, supports, cameras, rows, columns, width = future_supports.shape
        successor_per_support = measurement.successor_per_support
        transport_per_support = measurement.transport_per_support
        covariance_per_support = measurement.covariance_per_support
        null_probability = measurement.null_probability
        current_reference = measurement.current_reference
        object_validity = measurement.object_validity
        camera_validity = measurement.camera_validity
        geometry_coordinate = measurement.geometry_coordinate
        camera_legal = measurement.camera_legal
        offsets = measurement.offsets
        association_real_mass_per_support = measurement.association_real_mass_per_support
        uncertainty_per_support = measurement.uncertainty_per_support
        reliability_per_support = measurement.reliability_per_support
        association_confidence = measurement.association_confidence
        semantic = measurement.semantic
        appearance = measurement.appearance
        geometry = measurement.geometry
        fraction = measurement.fraction
        successor_rows: list[Tensor] = []
        transport_rows: list[Tensor] = []
        covariance_rows: list[Tensor] = []
        support_counts: list[Tensor] = []
        interval_observed_rows: list[Tensor] = []
        for interval_index, (lower, upper) in enumerate(self.time_grid.bounds):
            selected = self.time_grid.selection(offsets)[:, interval_index].float()
            # Only the legacy layout admits an irregular support fallback.
            # Aligned supports are validated against the exact declared lattice.
            if not self.time_grid.aligned:
                midpoint = 0.5 * float(lower + upper)
                nearest = (offsets.float() - midpoint).abs().argmin(dim=1)
                fallback = F.one_hot(nearest, num_classes=supports).float()
                selected = torch.where(
                    (selected.sum(dim=1, keepdim=True) > 0.0),
                    selected,
                    fallback,
                )
            interval_observed = torch.ones(batch, device=offsets.device, dtype=torch.bool)
            if future_observed is not None:
                interval_observed = ((selected == 0) | future_observed).all(dim=1)
            interval_observed_rows.append(interval_observed)
            weight = selected / selected.sum(dim=1, keepdim=True).clamp_min(1.0)
            # An incomplete fixed interval is unavailable, not a shorter target.
            if future_observed is not None:
                weight = weight * interval_observed[:, None]
            interval_successor = torch.einsum("bf,bfkd->bkd", weight, successor_per_support)
            if future_observed is not None:
                interval_successor = torch.where(
                    interval_observed[:, None, None], interval_successor, current_reference[:, 0]
                )
            successor_rows.append(interval_successor)
            interval_transport = torch.einsum("bf,bfkcd->bkcd", weight, transport_per_support)
            transport_rows.append(interval_transport)
            covariance_rows.append(
                torch.einsum(
                    "bf,bfkcd->bkcd",
                    weight,
                    covariance_per_support,
                )
            )
            support_counts.append((selected.sum(dim=1).float() * interval_observed).mean())
        successor = torch.stack(successor_rows, dim=1)
        transport = torch.stack(transport_rows, dim=1)
        covariance = torch.stack(covariance_rows, dim=1)
        if self.camera_names and len(self.camera_names) != cameras:
            raise ValueError("Teacher camera names lost current source axes")
        target = FutureObjectDynamics(
            camera_names=self.camera_names,
            time_grid_mode=self.time_grid.mode,
            current_reference=current_reference[:, 0],
            successor_content=successor,
            semantic_delta=(successor - current_reference),
            transport_mean=transport,
            transport_covariance=covariance,
            chart_availability=object_validity,
            log_chart_availability=self._supported_finite(
                facts.log_validity,
                object_validity,
                name="object log validity",
            ),
            camera_coordinates=geometry_coordinate,
            camera_chart_availability=camera_validity,
            log_camera_chart_availability=self._supported_finite(
                facts.log_camera_validity,
                camera_legal,
                name="camera log validity",
            ),
        )
        target.validate()
        if not collect_diagnostics:
            return target, {}
        metrics = {
            "object_teacher_association_real_mass": (
                supported_mean(association_real_mass_per_support, future_observed)
            ),
            "object_teacher_association_uncertainty": supported_mean(
                uncertainty_per_support, future_observed
            ),
            "object_teacher_reliability": supported_mean(reliability_per_support, future_observed),
            "object_teacher_association_confidence": supported_mean(
                association_confidence, future_observed
            ),
            "object_teacher_interval_variation": target.semantic_delta.float()
            .std(dim=1, unbiased=False)
            .mean(),
            "object_teacher_semantic_delta_rms": target.semantic_delta.square().mean().sqrt(),
            "object_teacher_transport_rms": transport.square().mean().sqrt(),
            "object_teacher_covariance_rms": covariance.square().mean().sqrt(),
            "object_teacher_current_loss_support": camera_validity.mean(),
            "object_teacher_successor_delta_identity_max_abs": (
                target.semantic_delta
                - (target.successor_content - target.current_reference[:, None])
            )
            .abs()
            .amax(),
            "object_teacher_null_probability": supported_mean(null_probability, future_observed),
            "object_teacher_semantic_max": semantic.detach().float().amax(dim=(-3, -2, -1)).mean(),
            "object_teacher_semantic_margin": (
                semantic.detach().float().amax(dim=(-3, -2, -1))
                - semantic.detach().float().mean(dim=(-3, -2, -1))
            ).mean(),
            "object_teacher_appearance_max": appearance.detach()
            .float()
            .amax(dim=(-3, -2, -1))
            .mean(),
            "object_teacher_appearance_margin": (
                appearance.detach().float().amax(dim=(-3, -2, -1))
                - appearance.detach().float().mean(dim=(-3, -2, -1))
            ).mean(),
            "object_teacher_geometry_margin": (
                geometry.detach().float().amax(dim=(-3, -2, -1))
                - geometry.detach().float().mean(dim=(-3, -2, -1))
            ).mean(),
            "object_teacher_supports_per_interval": torch.stack(support_counts).mean(),
            "object_teacher_flow_horizon_scale_mean": fraction.detach().float().mean(),
            "object_teacher_flow_horizon_scale_max": fraction.detach().float().amax(),
            "object_teacher_adjacent_cosine": F.cosine_similarity(
                successor[:, 1:].flatten(2), successor[:, :-1].flatten(2), dim=-1
            ).mean(),
        }
        if future_observed is not None:
            observed_intervals = torch.stack(interval_observed_rows, dim=1)
            metrics["object_teacher_observed_support_count"] = future_observed.float().sum()
            metrics["object_teacher_observed_interval_count"] = observed_intervals.float().sum()
            metrics["object_teacher_semantic_delta_rms"] = supported_mean(
                target.semantic_delta.square(), observed_intervals
            ).sqrt()
            metrics["object_teacher_transport_rms"] = supported_mean(
                transport.square(), observed_intervals
            ).sqrt()
            metrics["object_teacher_covariance_rms"] = supported_mean(
                covariance.square(), observed_intervals
            ).sqrt()
            for name, value in (
                ("semantic", semantic),
                ("appearance", appearance),
                ("geometry", geometry),
            ):
                maximum = value.detach().float().amax(dim=(-3, -2, -1))
                margin = maximum - value.detach().float().mean(dim=(-3, -2, -1))
                if name != "geometry":
                    metrics[f"object_teacher_{name}_max"] = supported_mean(maximum, future_observed)
                metrics[f"object_teacher_{name}_margin"] = supported_mean(margin, future_observed)
            pair_valid = observed_intervals[:, 1:] & observed_intervals[:, :-1]
            metrics["object_teacher_adjacent_cosine"] = supported_mean(
                F.cosine_similarity(
                    successor[:, 1:].flatten(2), successor[:, :-1].flatten(2), dim=-1
                ),
                pair_valid,
            )
            weights = observed_intervals[:, :, None, None]
            count = weights.float().sum(dim=1, keepdim=True).clamp_min(1.0)
            common = (target.semantic_delta * weights).sum(dim=1, keepdim=True) / count
            variance = ((target.semantic_delta - common).square() * weights).sum(dim=1) / count[
                :, 0
            ]
            metrics["object_teacher_interval_variation"] = supported_mean(
                variance.sqrt(), observed_intervals.any(dim=1)
            )
        successor_innovation = successor - target.current_reference[:, None]
        for index in range(len(self.time_grid.bounds)):
            row = f"object_teacher_interval_{index}"
            metrics[f"{row}_successor_innovation_rms"] = (
                successor_innovation[:, index].square().mean().sqrt()
            )
            metrics[f"{row}_semantic_delta_rms"] = (
                target.semantic_delta[:, index].square().mean().sqrt()
            )
            metrics[f"{row}_transport_rms"] = transport[:, index].square().mean().sqrt()
            if future_observed is not None:
                interval_mask = interval_observed_rows[index]
                metrics[f"{row}_observed_samples"] = interval_mask.float().sum()
                for name, value in (
                    ("successor_innovation", successor_innovation),
                    ("semantic_delta", target.semantic_delta),
                    ("transport", transport),
                ):
                    metrics[f"{row}_{name}_rms"] = supported_mean(
                        value[:, index].square(), interval_mask
                    ).sqrt()
        return target, metrics
