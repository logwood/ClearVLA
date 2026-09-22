"""No-grad object-aligned multi-frame future teacher for the mainline."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..future_time import LEGACY_FUTURE_TIME, resolve_future_time
from ..supervision import quarantine, supported_mean
from .types import (
    FutureObjectDynamics,
    ObjectFactSet,
    intersect_object_camera_validity,
)


class ObjectFutureTeacher(nn.Module):
    """Associate every current object with future DINO supports.

    The content projection is fixed and used only as a low-rank association
    key.  Full-width DINO values are retained in the target.  The association
    contains a legal null candidate, supports non-equal pixel positions and
    never enters the deployment forward path.
    """

    def __init__(
        self,
        *,
        content_dim: int,
        future_time_grid_mode: str = LEGACY_FUTURE_TIME,
        key_dim: int = 64,
        flow_reference_frames: int = 4,
    ) -> None:
        super().__init__()
        self.time_grid = resolve_future_time(future_time_grid_mode)
        self.content_dim = int(content_dim)
        self.key_dim = int(key_dim)
        self.flow_reference_frames = int(flow_reference_frames)
        if self.flow_reference_frames <= 0:
            raise ValueError("teacher flow reference frames must be positive")
        self.semantic_content_key = nn.Linear(content_dim, key_dim, bias=False)
        self.appearance_content_key = nn.Linear(content_dim, key_dim, bias=False)
        with torch.no_grad():
            for module in (
                self.semantic_content_key,
                self.appearance_content_key,
            ):
                nn.init.orthogonal_(module.weight)
        for module in (
            self.semantic_content_key,
            self.appearance_content_key,
        ):
            module.requires_grad_(False)

    @staticmethod
    def _offsets(offsets: Tensor, *, batch: int, supports: int) -> Tensor:
        if offsets.ndim == 1:
            if int(offsets.shape[0]) != supports:
                raise ValueError("future offsets do not match teacher supports")
            return offsets[None].expand(batch, -1)
        if tuple(offsets.shape) != (batch, supports):
            raise ValueError("future offsets must be [F] or [B,F]")
        # Per-example offsets are legal and handled by the vectorized interval
        # masks below.  Avoid a tensor-to-Python equality check here: Teacher
        # runs once per training batch and such a check would force a device
        # synchronization without protecting an actual semantic invariant.
        return offsets

    def _flow_horizon_scale(self, offsets: Tensor, source_steps: Tensor | None = None) -> Tensor:
        """Convert a raw-pair displacement into each future support horizon."""

        if source_steps is None:
            return offsets.float() / float(self.flow_reference_frames)
        # No motion observation is not a zero-duration division or a future label.
        return torch.where(source_steps[:, None] > 0, offsets.float() / source_steps[:, None].clamp_min(1), torch.zeros_like(offsets, dtype=torch.float32))

    @staticmethod
    def _supported_finite(
        value: Tensor,
        support: Tensor,
        *,
        name: str,
    ) -> Tensor:
        """Quarantine unsupported rows and fail closed on supported NaNs.

        The Teacher is detached, but its output is consumed by trainable
        recognizer/loss code.  A producer-invalid value must therefore be
        removed *before* any weighted product (``NaN * 0`` is still NaN).
        Conversely, a value marked physically supported but non-finite is a
        producer contract violation and should stop target construction rather
        than silently changing supervision.
        """

        value_f = value.detach().float()
        support_f = torch.nan_to_num(
            support.detach().float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)
        legal = support_f > 0.0
        if bool((legal & ~torch.isfinite(value_f)).any()):
            raise ValueError(f"Teacher {name} contains non-finite supported values")
        value_f = torch.nan_to_num(
            value_f,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        return torch.where(legal, value_f, torch.zeros_like(value_f))

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
        facts.validate()
        if future_supports.ndim != 6:
            raise ValueError("future supports must be [B,F,C,Y,X,D]")
        batch, supports, cameras, rows, columns, width = future_supports.shape
        if batch != facts.batch or width != self.content_dim:
            raise ValueError("future supports do not match current object content")
        if future_observed is not None:
            if (
                tuple(future_observed.shape) != (batch, supports)
                or future_observed.dtype != torch.bool
            ):
                raise ValueError("Teacher source support must be bool [B,F]")
            future_supports = quarantine(future_supports, future_observed)
        if not bool(torch.isfinite(future_supports).all()):
            raise ValueError("Teacher future supports contain non-finite values")
        offsets = self._offsets(
            future_offsets.to(device=future_supports.device),
            batch=batch,
            supports=supports,
        )
        if not bool(torch.isfinite(offsets.float()).all()):
            raise ValueError("Teacher future offsets contain non-finite values")
        if self.time_grid.aligned:
            self.time_grid.validate_offsets(offsets)
        objects = facts.objects
        # ``DenseObjectGrounder`` quarantines invalid candidate rows before its
        # learned projections, but the complete dense chart intentionally keeps
        # the producer values for reconstruction/audit.  A malformed invalid
        # row can therefore still be NaN here.  The Teacher consumes detached
        # G assignments, so applying the producer-owned validity boundary once
        # before the assignment/value product is the only safe way to prevent
        # ``NaN * 0`` from poisoning the target plane (and the downstream loss
        # graph when the target is later read by the recognizer).
        candidate_validity = torch.nan_to_num(
            facts.dense_chart.candidate_validity.detach().float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        candidate_content = self._supported_finite(
            facts.dense_chart.candidate_content,
            candidate_validity,
            name="dense candidate content",
        )

        def typed_current(assignment: Tensor, value: Tensor) -> Tensor:
            weight = torch.nan_to_num(
                assignment.detach().float(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).clamp_min(0.0).flatten(2)
            weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            flat_value = value.detach().float().flatten(1, -2)
            return torch.einsum("bkn,bnd->bkd", weight, flat_value)

        # The typed G posterior selects full-DINO current content before the
        # fixed association projection.  This keeps current/future keys in one
        # measurable space; comparing a learned route vector with an unrelated
        # random DINO projection would only manufacture an appearance score.
        semantic_current_content = typed_current(
            facts.semantic_candidate_assignment, candidate_content
        )
        appearance_current_content = typed_current(
            facts.appearance_candidate_assignment, candidate_content
        )
        current_semantic_key = F.normalize(
            self.semantic_content_key(semantic_current_content),
            dim=-1,
            eps=1e-4,
        )
        support_semantic_key = F.normalize(
            self.semantic_content_key(future_supports.detach().float()),
            dim=-1,
            eps=1e-4,
        )
        semantic = torch.einsum(
            "bkr,bfcyxr->bfkcyx",
            current_semantic_key,
            support_semantic_key,
        )
        current_appearance_key = F.normalize(
            self.appearance_content_key(appearance_current_content),
            dim=-1,
            eps=1e-4,
        )
        support_appearance_key = F.normalize(
            self.appearance_content_key(future_supports.detach().float()),
            dim=-1,
            eps=1e-4,
        )
        appearance = torch.einsum(
            "bkr,bfcyxr->bfkcyx",
            current_appearance_key,
            support_appearance_key,
        )
        axis_y = torch.linspace(-1.0, 1.0, rows, device=future_supports.device)
        axis_x = torch.linspace(-1.0, 1.0, columns, device=future_supports.device)
        coordinate_y, coordinate_x = torch.meshgrid(axis_y, axis_x, indexing="ij")
        coordinate = torch.stack((coordinate_x, coordinate_y), dim=-1)
        coordinate = coordinate.reshape(1, 1, 1, 1, rows, columns, 2)
        # Learned flow spans ``flow_reference_frames`` in the observable raw
        # pair.  Future supports are absolute frame offsets from the current
        # image, so constant-velocity extrapolation uses offset/reference,
        # not offset/max_future.  The latter silently shrank the H4 prior by
        # 12x and still supplied only one four-frame displacement at H48.
        fraction = self._flow_horizon_scale(offsets, facts.latest_flow_steps)
        # Spatial search uncertainty grows with future time even at reset.
        # It is not the displacement-extrapolation denominator.
        search_fraction = self._flow_horizon_scale(offsets)
        # This is the explicit previous->current displacement exported by G3,
        # indexed on the current fact chart and already expressed in true
        # normalized-coordinate units.  It is only a prior: semantic matching
        # remains global within the camera, so zero flow and non-equal-pixel
        # motion are legal.
        object_validity = torch.nan_to_num(
            facts.validity.detach().float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        camera_validity = intersect_object_camera_validity(
            object_validity,
            facts.camera_validity,
        )
        camera_support = torch.nan_to_num(
            facts.camera_support.detach().float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)
        camera_legal = (
            (object_validity[:, :, None, :] > 0.0)
            & (camera_validity > 0.0)
            & (camera_support > 0.0)
        )
        geometry_coordinate = self._supported_finite(
            facts.camera_coordinates,
            camera_legal,
            name="camera coordinates",
        )
        geometry_support = torch.where(
            camera_legal,
            camera_support,
            torch.zeros_like(camera_support),
        )
        flow_hint = self._supported_finite(
            facts.camera_transport_prior,
            camera_legal,
            name="camera transport prior",
        )
        prior = geometry_coordinate[:, None] + (
            fraction[:, :, None, None, None] * flow_hint[:, None]
        )
        prior = prior.clamp(-1.0, 1.0)
        delta = coordinate - prior[:, :, :, :, None, None]
        support_width = geometry_support.detach().float().clamp(0.03, 1.0)
        geometry = (
            -0.5
            * delta.square().sum(dim=-1)
            / (
                support_width[..., 0][:, None, :, :, None, None].square()
                + 0.08
                + 0.20 * search_fraction[:, :, None, None, None, None]
            )
        )
        geometry = geometry.clamp(-8.0, 0.0)
        camera_prior = self._supported_finite(
            facts.object_to_chart,
            camera_legal[..., 0, None, None],
            name="object-to-chart posterior",
        ).sum(dim=(-2, -1))
        camera_prior = camera_prior / camera_prior.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        camera_log = camera_prior.clamp_min(1e-5).log()[:, None, :, :, None, None]
        # Camera prior already integrates to one across cameras.  Normalize
        # the spatial candidate partition so one null hypothesis does not lose
        # merely because it competes with Y*X cells.  A fixed contrastive
        # temperature then rewards genuinely matching DINO evidence.
        candidate_logit = (
            4.5 * semantic
            + 1.5 * appearance
            + geometry
            + camera_log
            - math.log(float(max(rows * columns, 1)))
        )
        candidate_flat = candidate_logit.flatten(3)
        candidate_support = camera_legal.squeeze(-1)[:, None, :, :, None, None].expand(
            batch,
            supports,
            objects,
            cameras,
            rows,
            columns,
        )
        if future_observed is not None:
            candidate_support = candidate_support & future_observed[:, :, None, None, None, None]
        # Keep the null hypothesis as the finite all-invalid fallback while
        # removing unsupported camera cells from the real candidate partition.
        candidate_flat = torch.where(
            candidate_support.flatten(3),
            candidate_flat,
            torch.full_like(candidate_flat, -torch.inf),
        )
        null_logit = torch.zeros(
            batch,
            supports,
            objects,
            1,
            device=future_supports.device,
            dtype=candidate_flat.dtype,
        )
        posterior = torch.softmax(torch.cat((candidate_flat, null_logit), dim=-1), dim=-1)
        candidate_posterior = posterior[..., :-1].reshape(
            batch, supports, objects, cameras, rows, columns
        )
        null_probability = posterior[..., -1:]
        support_content = (
            future_supports.detach()
            .float()
            .reshape(batch, supports, cameras * rows * columns, width)
        )
        candidate_flat_probability = candidate_posterior.flatten(3)
        matched = torch.einsum("bfkn,bfnd->bfkd", candidate_flat_probability, support_content)
        candidate_coordinate = coordinate[0, 0, 0, 0].unsqueeze(0).expand(cameras, -1, -1, -1)
        current_camera_measure = camera_validity
        current_camera_measure = torch.where(
            current_camera_measure.sum(dim=2, keepdim=True) > 1.0e-8,
            current_camera_measure
            / current_camera_measure.sum(dim=2, keepdim=True).clamp_min(1.0e-8),
            torch.zeros_like(current_camera_measure),
        )
        transport_per_support, covariance_per_support = self._relative_geometry_moments(
            candidate_posterior=candidate_posterior,
            candidate_coordinate=candidate_coordinate,
            current_camera_coordinate=geometry_coordinate,
            null_probability=null_probability,
            null_camera_measure=current_camera_measure,
        )
        entropy = -(posterior.clamp_min(1e-8) * posterior.clamp_min(1e-8).log()).sum(
            dim=-1, keepdim=True
        ) / math.log(float(cameras * rows * columns + 1))
        association_real_mass_per_support = 1.0 - null_probability
        # A confident null match is still epistemically uncertain about the
        # future object state.  Plain posterior entropy would incorrectly call
        # it certain, so null mass supplies the fallback uncertainty floor.
        uncertainty_per_support = null_probability + association_real_mass_per_support * entropy
        association_confidence = (1.0 - entropy).clamp(0.0, 1.0)
        reliability_per_support = association_real_mass_per_support * association_confidence
        current_reference = self._supported_finite(
            facts.content,
            object_validity,
            name="current object content",
        )[:, None]
        # This is the exact V120 physical target algebra.  Null mass already
        # supplies the only identity fallback.  Association confidence remains
        # a calibration diagnostic and must not contract a diffuse-but-visible
        # target toward the current fact a second time.
        successor_per_support = matched + null_probability * current_reference

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
        target = FutureObjectDynamics(
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

    @staticmethod
    def _relative_geometry_moments(
        *,
        candidate_posterior: Tensor,
        candidate_coordinate: Tensor,
        current_camera_coordinate: Tensor,
        null_probability: Tensor,
        null_camera_measure: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Form camera-specific moments with the row-softmax null identity.

        Each camera has its own normalized image chart.  Subtracting one
        separately reduced global current coordinate from a posterior whose
        camera mass can change creates motion for a static object. Null is the
        legitimate zero-displacement hypothesis: allocate it over current
        observable cameras and include it in each camera denominator.
        """

        if candidate_posterior.ndim != 6:
            raise ValueError("candidate posterior must be [B,F,K,C,Y,X]")
        batch, _, objects, cameras, rows, columns = candidate_posterior.shape
        if tuple(candidate_coordinate.shape) != (cameras, rows, columns, 2):
            raise ValueError("candidate coordinate chart does not align with cameras")
        if tuple(current_camera_coordinate.shape) != (batch, objects, cameras, 2):
            raise ValueError("current camera coordinates must be [B,K,C,2]")
        if tuple(null_probability.shape) != tuple(candidate_posterior.shape[:3]) + (1,):
            raise ValueError("null probability must be [B,F,K,1]")
        if tuple(null_camera_measure.shape) != (batch, objects, cameras, 1):
            raise ValueError("null camera measure must be [B,K,C,1]")
        real_mass = candidate_posterior.float().sum(
            dim=(-2, -1),
            keepdim=True,
        )
        identity_mass = (
            null_probability.float()[:, :, :, None, None]
            * null_camera_measure.float()[:, None, :, :, None]
        )
        total_mass = real_mass + identity_mass
        normalized_real = torch.where(
            total_mass > 1.0e-8,
            candidate_posterior.float() / total_mass.clamp_min(1.0e-8),
            torch.zeros_like(candidate_posterior.float()),
        )
        displacement = (
            candidate_coordinate[None, None, None].float()
            - current_camera_coordinate[:, None, :, :, None, None].float()
        )
        transport = torch.einsum(
            "bfkcyx,bfkcyxd->bfkcd",
            normalized_real,
            displacement,
        )
        second_xx = torch.einsum(
            "bfkcyx,bfkcyx->bfkc",
            normalized_real,
            displacement[..., 0].square(),
        )
        second_xy = torch.einsum(
            "bfkcyx,bfkcyx->bfkc",
            normalized_real,
            displacement[..., 0] * displacement[..., 1],
        )
        second_yy = torch.einsum(
            "bfkcyx,bfkcyx->bfkc",
            normalized_real,
            displacement[..., 1].square(),
        )
        covariance_xx = (second_xx - transport[..., 0].square()).clamp_min(0.0)
        covariance_xy = second_xy - transport[..., 0] * transport[..., 1]
        covariance_yy = (second_yy - transport[..., 1].square()).clamp_min(0.0)
        covariance_xy_limit = torch.sqrt(covariance_xx * covariance_yy)
        covariance_xy = torch.maximum(
            torch.minimum(covariance_xy, covariance_xy_limit),
            -covariance_xy_limit,
        )
        covariance = torch.stack(
            (covariance_xx, covariance_xy, covariance_yy),
            dim=-1,
        )
        return transport, covariance
