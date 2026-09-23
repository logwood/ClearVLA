"""Frozen object-to-observation association shared by labels and causal feedback.

Measures an explicitly supplied observation relative to a source object chart.
The caller owns observation availability and physical chronology. This module
chooses no task, control, terminal state, or persistent physical identity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..supervision import quarantine
from .types import ObjectFactSet, intersect_object_camera_validity


@dataclass(frozen=True)
class ObjectObservationMeasurement:
    """Full soft measurement and null mass before interval aggregation."""

    successor_per_support: Tensor
    transport_per_support: Tensor
    covariance_per_support: Tensor
    candidate_posterior: Tensor
    null_probability: Tensor
    current_reference: Tensor
    object_validity: Tensor
    camera_validity: Tensor
    geometry_coordinate: Tensor
    camera_legal: Tensor
    offsets: Tensor
    association_real_mass_per_support: Tensor
    uncertainty_per_support: Tensor
    reliability_per_support: Tensor
    association_confidence: Tensor
    semantic: Tensor
    appearance: Tensor
    geometry: Tensor
    fraction: Tensor


class ObjectObservationAssociation(nn.Module):
    """Established frozen association law, not learned persistent identity."""

    def __init__(
        self,
        *,
        content_dim: int,
        key_dim: int = 64,
        flow_reference_frames: int = 4,
        camera_names: tuple[str, ...] = (),
    ) -> None:
        super().__init__()
        self.camera_names = tuple(camera_names)
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
        return torch.where(
            source_steps[:, None] > 0,
            offsets.float() / source_steps[:, None].clamp_min(1),
            torch.zeros_like(offsets, dtype=torch.float32),
        )

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
    def measure_observations(
        self,
        *,
        facts: ObjectFactSet,
        observations: Tensor,
        relative_offsets: Tensor,
        observed: Tensor | None = None,
    ) -> ObjectObservationMeasurement:
        with torch.autocast(device_type=observations.device.type, enabled=False):
            return self._measure_fp32(
                facts=facts,
                observations=observations,
                relative_offsets=relative_offsets,
                observed=observed,
            )

    def _measure_fp32(
        self,
        *,
        facts: ObjectFactSet,
        observations: Tensor,
        relative_offsets: Tensor,
        observed: Tensor | None,
    ) -> ObjectObservationMeasurement:
        facts.validate()
        if observations.ndim != 6:
            raise ValueError("future supports must be [B,F,C,Y,X,D]")
        batch, supports, cameras, rows, columns, width = observations.shape
        if batch != facts.batch or width != self.content_dim:
            raise ValueError("future supports do not match current object content")
        if observed is not None:
            if tuple(observed.shape) != (batch, supports) or observed.dtype != torch.bool:
                raise ValueError("Teacher source support must be bool [B,F]")
            observations = quarantine(observations, observed)
        if not bool(torch.isfinite(observations).all()):
            raise ValueError("Teacher future supports contain non-finite values")
        offsets = self._offsets(
            relative_offsets.to(device=observations.device),
            batch=batch,
            supports=supports,
        )
        if not bool(torch.isfinite(offsets.float()).all()):
            raise ValueError("Teacher future offsets contain non-finite values")
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
            weight = (
                torch.nan_to_num(
                    assignment.detach().float(),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                .clamp_min(0.0)
                .flatten(2)
            )
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
            self.semantic_content_key(observations.detach().float()),
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
            self.appearance_content_key(observations.detach().float()),
            dim=-1,
            eps=1e-4,
        )
        appearance = torch.einsum(
            "bkr,bfcyxr->bfkcyx",
            current_appearance_key,
            support_appearance_key,
        )
        axis_y = torch.linspace(-1.0, 1.0, rows, device=observations.device)
        axis_x = torch.linspace(-1.0, 1.0, columns, device=observations.device)
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
        if observed is not None:
            candidate_support = candidate_support & observed[:, :, None, None, None, None]
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
            device=observations.device,
            dtype=candidate_flat.dtype,
        )
        posterior = torch.softmax(torch.cat((candidate_flat, null_logit), dim=-1), dim=-1)
        candidate_posterior = posterior[..., :-1].reshape(
            batch, supports, objects, cameras, rows, columns
        )
        null_probability = posterior[..., -1:]
        support_content = (
            observations.detach().float().reshape(batch, supports, cameras * rows * columns, width)
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

        return ObjectObservationMeasurement(
            successor_per_support=successor_per_support,
            transport_per_support=transport_per_support,
            covariance_per_support=covariance_per_support,
            candidate_posterior=candidate_posterior,
            null_probability=null_probability,
            current_reference=current_reference,
            object_validity=object_validity,
            camera_validity=camera_validity,
            geometry_coordinate=geometry_coordinate,
            camera_legal=camera_legal,
            offsets=offsets,
            association_real_mass_per_support=association_real_mass_per_support,
            uncertainty_per_support=uncertainty_per_support,
            reliability_per_support=reliability_per_support,
            association_confidence=association_confidence,
            semantic=semantic,
            appearance=appearance,
            geometry=geometry,
            fraction=fraction,
        )

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
