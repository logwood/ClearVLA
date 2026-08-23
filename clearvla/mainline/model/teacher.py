"""No-grad object-aligned multi-frame future teacher for the mainline."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .types import INTERVAL_BOUNDS, FutureObjectDynamics, ObjectFactSet


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
        key_dim: int = 64,
        flow_reference_frames: int = 4,
    ) -> None:
        super().__init__()
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

    def _flow_horizon_scale(self, offsets: Tensor) -> Tensor:
        """Convert a raw-pair displacement into each future support horizon."""

        return offsets.float() / float(self.flow_reference_frames)

    @staticmethod
    def _partial_assignment(
        score: Tensor,
        *,
        dustbin_score: float = 0.0,
        iterations: int = 20,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return row-normalized real and dustbin mass via partial OT.

        ``score`` is ``[..., K, N]``.  The augmented transport has one
        dustbin row and column.  Real rows each carry unit mass, real columns
        each accept unit mass, and the two dustbins absorb unmatched objects
        or future cells.  The fixed dustbin has no learned parameter and the
        whole routine runs on the no-grad Teacher plane.
        """

        if score.ndim < 3:
            raise ValueError("partial assignment score must be [...,K,N]")
        objects, candidates = int(score.shape[-2]), int(score.shape[-1])
        if objects < 1 or candidates < 1:
            raise ValueError("partial assignment requires real rows and columns")
        alpha = score.new_tensor(float(dustbin_score))
        bins_object = alpha.expand(*score.shape[:-2], objects, 1)
        bins_candidate = alpha.expand(*score.shape[:-2], 1, candidates)
        bin_corner = alpha.expand(*score.shape[:-2], 1, 1)
        coupling = torch.cat(
            (
                torch.cat((score, bins_object), dim=-1),
                torch.cat((bins_candidate, bin_corner), dim=-1),
            ),
            dim=-2,
        )
        normalizer = -math.log(float(objects + candidates))
        log_mu = torch.cat(
            (
                score.new_full((*score.shape[:-2], objects), normalizer),
                score.new_full(
                    (*score.shape[:-2], 1),
                    math.log(float(candidates)) + normalizer,
                ),
            ),
            dim=-1,
        )
        log_nu = torch.cat(
            (
                score.new_full((*score.shape[:-2], candidates), normalizer),
                score.new_full(
                    (*score.shape[:-2], 1),
                    math.log(float(objects)) + normalizer,
                ),
            ),
            dim=-1,
        )
        u = torch.zeros_like(log_mu)
        v = torch.zeros_like(log_nu)
        for _ in range(int(iterations)):
            u = log_mu - torch.logsumexp(coupling + v.unsqueeze(-2), dim=-1)
            v = log_nu - torch.logsumexp(
                coupling + u.unsqueeze(-1), dim=-2
            )
        log_transport = coupling + u.unsqueeze(-1) + v.unsqueeze(-2)
        # Undo the common marginal normalizer.  Every real object row then
        # sums to one across real candidates plus its dustbin probability.
        transport = torch.exp(log_transport - normalizer)
        real = transport[..., :objects, :candidates]
        null = transport[..., :objects, candidates:]
        row_error = (real.sum(dim=-1, keepdim=True) + null - 1.0).abs()
        return real, null, row_error

    @torch.no_grad()
    def forward(
        self,
        *,
        facts: ObjectFactSet,
        future_supports: Tensor,
        future_offsets: Tensor,
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
                    collect_diagnostics=collect_diagnostics,
                )
        return self._forward_fp32(
            facts=facts,
            future_supports=future_supports,
            future_offsets=future_offsets,
            collect_diagnostics=collect_diagnostics,
        )

    def _forward_fp32(
        self,
        *,
        facts: ObjectFactSet,
        future_supports: Tensor,
        future_offsets: Tensor,
        collect_diagnostics: bool,
    ) -> tuple[FutureObjectDynamics, dict[str, Tensor]]:
        facts.validate()
        if future_supports.ndim != 6:
            raise ValueError("future supports must be [B,F,C,Y,X,D]")
        batch, supports, cameras, rows, columns, width = future_supports.shape
        if batch != facts.batch or width != self.content_dim:
            raise ValueError("future supports do not match current object content")
        offsets = self._offsets(
            future_offsets.to(device=future_supports.device),
            batch=batch,
            supports=supports,
        )
        objects = facts.objects
        candidate_content = facts.dense_chart.candidate_content.detach().float()

        def typed_current(assignment: Tensor, value: Tensor) -> Tensor:
            weight = assignment.detach().float().flatten(2)
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
        fraction = self._flow_horizon_scale(offsets)
        # This is the explicit previous->current displacement exported by G3,
        # indexed on the current fact chart and already expressed in true
        # normalized-coordinate units.  It is only a prior: semantic matching
        # remains global within the camera, so zero flow and non-equal-pixel
        # motion are legal.
        geometry_coordinate = facts.camera_coordinates.detach().float()
        geometry_support = facts.camera_support.detach().float()
        flow_hint = facts.camera_transport_prior.detach().float()
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
                + 0.20 * fraction[:, :, None, None, None, None]
            )
        )
        geometry = geometry.clamp(-8.0, 0.0)
        camera_prior = facts.object_to_chart.detach().float().sum(dim=(-2, -1))
        camera_prior = camera_prior / camera_prior.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        camera_log = camera_prior.clamp_min(1e-5).log()[:, None, :, :, None, None]
        # Remove the broad spatial background of each fixed DINO key before
        # matching.  Diffuse positive cosine is not evidence that one specific
        # future cell belongs to this object and must not suppress dustbin by
        # candidate count alone.
        semantic_background = semantic.mean(
            dim=(-3, -2, -1), keepdim=True
        )
        appearance_background = appearance.mean(
            dim=(-3, -2, -1), keepdim=True
        )
        semantic_contrast = semantic - semantic_background
        appearance_contrast = appearance - appearance_background
        # Camera priors already sum to one across cameras.  Dividing the
        # spatial evidence by Y*X gives a diffuse, background-level field one
        # unit of aggregate evidence rather than Y*X accidental votes against
        # the dustbin.  A localized positive contrast can still overcome this
        # fixed calibration; no learned threshold or matching quota is added.
        candidate_logit = (
            4.5 * semantic_contrast
            + 1.5 * appearance_contrast
            + geometry
            + camera_log
            - math.log(float(rows * columns))
        )
        candidate_flat = candidate_logit.flatten(3)
        candidate_flat_probability, null_probability, assignment_row_error = (
            self._partial_assignment(
                candidate_flat,
                dustbin_score=0.0,
                iterations=20,
            )
        )
        candidate_posterior = candidate_flat_probability.reshape(
            batch, supports, objects, cameras, rows, columns
        )
        posterior = torch.cat(
            (candidate_flat_probability, null_probability), dim=-1
        )
        support_content = (
            future_supports.detach()
            .float()
            .reshape(batch, supports, cameras * rows * columns, width)
        )
        matched = torch.einsum("bfkn,bfnd->bfkd", candidate_flat_probability, support_content)
        candidate_coordinate = coordinate[0, 0, 0, 0].unsqueeze(0).expand(
            cameras, -1, -1, -1
        )
        current_camera_measure = (
            facts.camera_evidence_mass.detach().float()
            * facts.camera_chart_availability.detach().float()
        )
        available_camera_measure = facts.camera_chart_availability.detach().float()
        normalized_available_camera = available_camera_measure / available_camera_measure.sum(
            dim=2, keepdim=True
        ).clamp_min(1.0)
        current_camera_measure = torch.where(
            current_camera_measure.sum(dim=2, keepdim=True) > 1.0e-8,
            current_camera_measure
            / current_camera_measure.sum(dim=2, keepdim=True).clamp_min(1.0e-8),
            normalized_available_camera,
        )
        transport_per_support, covariance_per_support = self._relative_geometry_moments(
            candidate_posterior=candidate_posterior,
            candidate_coordinate=candidate_coordinate,
            current_camera_coordinate=facts.camera_coordinates.detach().float(),
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
        uncertainty_per_support = (
            null_probability + association_real_mass_per_support * entropy
        )
        association_confidence = (1.0 - entropy).clamp(0.0, 1.0)
        reliability_per_support = (
            association_real_mass_per_support * association_confidence
        )
        conditional_candidate = (
            candidate_flat_probability
            / association_real_mass_per_support.clamp_min(1.0e-8)
        )
        conditional_entropy = -(
            conditional_candidate.clamp_min(1.0e-8)
            * conditional_candidate.clamp_min(1.0e-8).log()
        ).sum(dim=-1, keepdim=True)
        effective_support = torch.exp(conditional_entropy) * (
            association_real_mass_per_support > 1.0e-8
        )
        column_ownership = candidate_flat_probability / candidate_flat_probability.sum(
            dim=2, keepdim=True
        ).clamp_min(1.0e-8)
        mutual_assignment_mass = (
            candidate_flat_probability * column_ownership
        ).sum(dim=-1, keepdim=True)
        current_reference = facts.content.detach().float()[:, None]
        # This is the exact V120 physical target algebra.  Null mass already
        # supplies the only identity fallback.  Association confidence remains
        # a calibration diagnostic and must not contract a diffuse-but-visible
        # target toward the current fact a second time.
        successor_per_support = matched + null_probability * current_reference

        successor_rows: list[Tensor] = []
        transport_rows: list[Tensor] = []
        covariance_rows: list[Tensor] = []
        association_real_mass_rows: list[Tensor] = []
        uncertainty_rows: list[Tensor] = []
        reliability_rows: list[Tensor] = []
        support_counts: list[Tensor] = []
        for lower, upper in INTERVAL_BOUNDS:
            selected = ((offsets >= lower) & (offsets <= upper)).float()
            # Configured support sets normally cover every interval.  The
            # fallback selects the nearest support deterministically without
            # creating a zero/random target.
            midpoint = 0.5 * float(lower + upper)
            nearest = (offsets.float() - midpoint).abs().argmin(dim=1)
            fallback = F.one_hot(nearest, num_classes=supports).float()
            selected = torch.where(
                (selected.sum(dim=1, keepdim=True) > 0.0),
                selected,
                fallback,
            )
            weight = selected / selected.sum(dim=1, keepdim=True).clamp_min(1.0)
            interval_successor = torch.einsum(
                "bf,bfkd->bkd", weight, successor_per_support
            )
            successor_rows.append(interval_successor)
            interval_transport = torch.einsum(
                "bf,bfkcd->bkcd", weight, transport_per_support
            )
            transport_rows.append(interval_transport)
            covariance_rows.append(
                torch.einsum(
                    "bf,bfkcd->bkcd",
                    weight,
                    covariance_per_support,
                )
            )
            interval_association_real_mass = torch.einsum(
                "bf,bfkd->bkd",
                weight,
                association_real_mass_per_support,
            )
            association_real_mass_rows.append(interval_association_real_mass)
            uncertainty_rows.append(
                torch.einsum("bf,bfkd->bkd", weight, uncertainty_per_support)
                + (
                    torch.einsum(
                        "bf,bfkd->bkd",
                        weight,
                        (successor_per_support - interval_successor[:, None])
                        .square()
                        .mean(dim=-1, keepdim=True),
                    )
                ).sqrt()
            )
            reliability_rows.append(torch.einsum("bf,bfkd->bkd", weight, reliability_per_support))
            support_counts.append(selected.sum(dim=1).float().mean())
        successor = torch.stack(successor_rows, dim=1)
        transport = torch.stack(transport_rows, dim=1)
        covariance = torch.stack(covariance_rows, dim=1)
        association_real_mass = torch.stack(association_real_mass_rows, dim=1)
        uncertainty = torch.stack(uncertainty_rows, dim=1)
        reliability = torch.stack(reliability_rows, dim=1)
        current_chart_availability = facts.chart_availability.detach().float()
        current_validity = current_chart_availability[:, None]
        # Dustbin is an association-null probability, not an observable
        # disappearance label. It remains the sole successor identity fallback
        # and contributes to uncertainty/reliability above, but status targets
        # stay at the exact neutral value until physical future visibility is
        # independently observable. Future selector support therefore carries
        # only detached current chart availability, never dustbin or existence.
        visibility_change = torch.zeros_like(association_real_mass)
        persistence_change = torch.zeros_like(association_real_mass)
        future_selector_validity = current_validity.expand_as(
            association_real_mass
        )
        target = FutureObjectDynamics(
            current_reference=facts.content.detach().float(),
            successor_content=successor,
            semantic_delta=(successor - facts.content.detach().float()[:, None]),
            transport_mean=transport,
            transport_covariance=covariance,
            visibility=visibility_change,
            persistence=persistence_change,
            chart_availability=current_chart_availability,
            future_selector_validity=future_selector_validity,
            camera_coordinates=facts.camera_coordinates.detach().float(),
            camera_chart_availability=(
                facts.camera_chart_availability.detach().float()
            ),
            camera_weights=(
                facts.camera_evidence_mass.detach().float()
                * facts.camera_chart_availability.detach().float()
            ),
        )
        target.validate()
        if not collect_diagnostics:
            return target, {}
        metrics = {
            "object_teacher_association_real_mass": association_real_mass.mean(),
            "object_teacher_visibility_change": visibility_change.mean(),
            "object_teacher_persistence_change": persistence_change.mean(),
            "object_teacher_uncertainty": uncertainty.mean(),
            "object_teacher_reliability": reliability.mean(),
            "object_teacher_association_confidence": association_confidence.mean(),
            "object_teacher_interval_variation": target.semantic_delta.float()
            .std(dim=1, unbiased=False)
            .mean(),
            "object_teacher_semantic_delta_rms": target.semantic_delta.square().mean().sqrt(),
            "object_teacher_transport_rms": transport.square().mean().sqrt(),
            "object_teacher_covariance_rms": covariance.square().mean().sqrt(),
            "object_teacher_current_loss_support": facts.camera_chart_availability.detach()
            .float()
            .mean(),
            "object_teacher_future_selector_validity": future_selector_validity.mean(),
            "object_teacher_successor_delta_identity_max_abs": (
                target.semantic_delta
                - (target.successor_content - target.current_reference[:, None])
            )
            .abs()
            .amax(),
            "object_teacher_null_probability": null_probability.mean(),
            "object_teacher_dustbin_probability": null_probability.mean(),
            "object_teacher_effective_support": effective_support.mean(),
            "object_teacher_mutual_assignment_mass": mutual_assignment_mass.mean(),
            "object_teacher_partial_assignment_row_error": assignment_row_error.amax(),
            "object_teacher_best_minus_background": candidate_logit.detach()
            .float()
            .amax(dim=(-3, -2, -1))
            .mean(),
            "object_teacher_interval_residual_rms": target.semantic_interval_residual
            .square()
            .mean()
            .sqrt(),
            "object_teacher_common_effect_rms": target.semantic_common.square()
            .mean()
            .sqrt(),
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
        for index in range(len(INTERVAL_BOUNDS)):
            row = f"object_teacher_interval_{index}"
            metrics[f"{row}_semantic_delta_rms"] = (
                target.semantic_delta[:, index].square().mean().sqrt()
            )
            metrics[f"{row}_transport_rms"] = transport[:, index].square().mean().sqrt()
            metrics[f"{row}_visibility_change"] = visibility_change[:, index].mean()
            metrics[f"{row}_persistence_change"] = persistence_change[:, index].mean()
            metrics[f"{row}_reliability"] = reliability[:, index].mean()
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
        """Form camera-specific moments including identity-null transport.

        Each camera has its own normalized image chart.  Subtracting one
        separately reduced global current coordinate from a posterior whose
        camera mass can change creates motion for a static object.  Compute
        ``future-current`` and both moments independently inside every camera.
        Dustbin is not disappearance, but it is the legitimate identity-motion
        hypothesis.  Allocate its zero displacement over the currently
        observable camera measure and include it in the same first/second
        moment denominator.  This prevents a tiny ambiguous real match from
        being renormalized into a certain large motion.
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
            dim=(-2, -1), keepdim=True
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
        # Exact moments are PSD; enforce only the floating-point boundary so a
        # tiny cancellation error cannot serialize an invalid 2x2 target.
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
