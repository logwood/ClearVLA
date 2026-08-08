"""Dense-chart to global-object binding for G3."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..grounded_intent_effect import GroundedFactSet
from ..role_delta_attnres import smooth_rms_contract
from .types import DenseFactChart, ObjectFactSet, normalized_entropy


def _coordinate_basis(coordinates: Tensor, width: int) -> Tensor:
    frequencies = torch.arange(
        1,
        max(int(width) // 4, 1) + 1,
        device=coordinates.device,
        dtype=torch.float32,
    )
    angle = math.pi * coordinates.float()[..., None] * frequencies
    value = torch.cat((angle.sin(), angle.cos()), dim=-1).flatten(-2)
    if int(value.shape[-1]) < int(width):
        value = F.pad(value, (0, int(width) - int(value.shape[-1])))
    return value[..., : int(width)].to(dtype=coordinates.dtype)


def dense_chart_from_local_facts(local: GroundedFactSet) -> DenseFactChart:
    """Preserve every local hypothesis while exposing a dense DINO target."""

    local.validate()
    validity = local.slot_validity.to(dtype=local.content_slots.dtype)
    semantic_mass = local.semantic_owner_probs[..., None] * validity
    dense_content = (
        local.content_slots * semantic_mass
    ).sum(dim=-2) / semantic_mass.sum(dim=-2).clamp_min(1e-6)
    # This is a conditional mixture over the local M hypotheses.  Candidate
    # validity is a separate Bernoulli support variable and must not be folded
    # into this distribution: doing so turns the complement of a perfectly
    # legal (for example uniform 1/M) prior into false null evidence.
    owner_prior = torch.sqrt(
        local.semantic_owner_probs.float().clamp_min(0.0)
        * local.geometry_owner_probs.float().clamp_min(0.0)
    )
    prior_mass = owner_prior.sum(dim=-1, keepdim=True)
    uniform_prior = torch.full_like(
        owner_prior, 1.0 / float(max(int(owner_prior.shape[-1]), 1))
    )
    owner_prior = torch.where(
        prior_mass > 1e-6,
        owner_prior / prior_mass.clamp_min(1e-6),
        uniform_prior,
    )
    chart = DenseFactChart(
        public_scene_base=local.public_scene_base,
        dino_content=dense_content,
        candidate_content=local.content_slots,
        candidate_semantic=local.semantic_slots,
        candidate_appearance=local.appearance_slots,
        candidate_geometry=local.geometry_slots,
        candidate_coordinates=local.slot_coordinates,
        candidate_support=local.slot_support,
        candidate_validity=local.slot_validity,
        candidate_owner_prior=owner_prior.to(dtype=local.content_slots.dtype),
        candidate_transport_prior=(
            local.slot_transport_prior
            if local.slot_transport_prior is not None
            else torch.zeros_like(local.slot_coordinates)
        ),
    )
    chart.validate()
    return chart


class DenseObjectGrounder(nn.Module):
    """Competition-normalized Slot Attention over the complete current chart.

    Local ``M`` hypotheses are candidates; global ``K`` objects are the first
    identity-bearing representation.  Candidate competition is normalized
    across objects plus a null owner before each object's read posterior is
    normalized across candidates.  Consequently the same dense fact cannot be
    copied independently into all objects.
    """

    def __init__(
        self,
        *,
        hidden: int,
        content_dim: int,
        route_dim: int,
        objects: int = 4,
        iterations: int = 3,
        maximum_update_rms: float = 0.35,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.content_dim = int(content_dim)
        self.route_dim = int(route_dim)
        self.objects = int(objects)
        self.iterations = int(iterations)
        if min(self.hidden, self.content_dim, self.route_dim, self.objects, self.iterations) < 1:
            raise ValueError("object grounder dimensions must be positive")
        self.content_key = nn.Sequential(
            nn.LayerNorm(content_dim, elementwise_affine=False),
            nn.Linear(content_dim, hidden, bias=False),
        )
        self.semantic_key = nn.Linear(route_dim, hidden, bias=False)
        self.appearance_key = nn.Linear(route_dim, hidden, bias=False)
        self.geometry_key = nn.Linear(route_dim, hidden, bias=False)
        self.coordinate_key = nn.Linear(16, hidden, bias=False)
        self.candidate_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.slot_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        # Slot identities only break symmetry.  All transition/read modules
        # are shared across K and therefore cannot assign role-specific heads.
        self.slot_seed = nn.Parameter(torch.randn(1, objects, hidden) * 0.02)
        self.null_key = nn.Parameter(torch.zeros(1, 1, hidden))
        self.gru = nn.GRUCell(hidden, hidden)
        self.update_ffn = nn.Sequential(
            nn.LayerNorm(hidden, elementwise_affine=False),
            nn.Linear(hidden, 2 * hidden, bias=False),
            nn.SiLU(),
            nn.Linear(2 * hidden, hidden, bias=False),
        )
        self.g3_residual = nn.Sequential(
            nn.LayerNorm(2 * hidden, elementwise_affine=False),
            nn.Linear(2 * hidden, hidden, bias=False),
            nn.SiLU(),
            nn.Linear(hidden, 1, bias=False),
        )
        g3_output = self.g3_residual[-1]
        if not isinstance(g3_output, nn.Linear):
            raise TypeError("G3 residual output must remain a linear layer")
        nn.init.zeros_(g3_output.weight)
        # Semantic, appearance and geometry verify evidence *inside* the one
        # physical K+null assignment.  Zero initialization makes all three
        # reads exactly inherit the physical posterior at startup; ordinary
        # gradients may then refine them without creating three drifting
        # object identities.
        self.typed_verifier = nn.ModuleDict(
            {
                name: nn.Linear(2 * route_dim, 1, bias=False)
                for name in ("semantic", "appearance", "geometry")
            }
        )
        for verifier in self.typed_verifier.values():
            if not isinstance(verifier, nn.Linear):
                raise TypeError("typed verifier must remain a linear layer")
            nn.init.zeros_(verifier.weight)
        self.decode_content_residual = nn.Linear(hidden, content_dim, bias=False)
        nn.init.zeros_(self.decode_content_residual.weight)
        self.decode_position = nn.Linear(16, content_dim, bias=False)
        self.maximum_update_rms = float(maximum_update_rms)

    def _candidate_tokens(self, chart: DenseFactChart) -> Tensor:
        content = self.content_key(chart.candidate_content)
        typed = (
            self.semantic_key(chart.candidate_semantic)
            + self.appearance_key(chart.candidate_appearance)
            + self.geometry_key(chart.candidate_geometry)
        ) / math.sqrt(3.0)
        coordinate = self.coordinate_key(
            _coordinate_basis(chart.candidate_coordinates, 16)
        )
        return self.candidate_norm(content + typed + coordinate)

    def _competition(
        self,
        slots: Tensor,
        candidates: Tensor,
        validity: Tensor,
        candidate_prior: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch, count, _ = candidates.shape
        slot_key = self.slot_norm(slots)
        logits = torch.einsum("bkh,bnh->bnk", slot_key.float(), candidates.float())
        logits = logits / math.sqrt(float(self.hidden))
        null = torch.einsum(
            "bnh,bqh->bnq",
            candidates.float(),
            self.null_key.to(device=candidates.device, dtype=candidates.dtype).expand(batch, -1, -1).float(),
        )
        # ``owner`` is conditional on one local candidate.  The local prior
        # is applied afterwards as joint mixture mass.  Only true invalidity
        # may move probability to null; ``1 - candidate_prior`` denotes other
        # hypotheses at the same cell, not absence.
        owner = torch.softmax(torch.cat((logits, null), dim=-1), dim=-1)
        valid = validity.float().reshape(batch, count, 1).clamp(0.0, 1.0)
        prior = candidate_prior.float().reshape(batch, count, 1).clamp_min(0.0)
        object_mass = owner[..., : self.objects] * valid * prior
        null_mass = (
            owner[..., self.objects] * valid[..., 0]
            + (1.0 - valid[..., 0])
        ) * prior[..., 0]
        read = object_mass.transpose(1, 2)
        read = read / read.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return owner, object_mass, null_mass, read

    def forward(self, local_facts: GroundedFactSet) -> tuple[ObjectFactSet, dict[str, Tensor]]:
        chart = dense_chart_from_local_facts(local_facts)
        candidates_structured = self._candidate_tokens(chart)
        batch = int(candidates_structured.shape[0])
        candidate_shape = candidates_structured.shape[1:-1]
        count = math.prod(int(value) for value in candidate_shape)
        candidates = candidates_structured.reshape(batch, count, self.hidden)
        validity = chart.candidate_validity.reshape(batch, count, 1)
        candidate_prior = chart.candidate_owner_prior.reshape(batch, count, 1)
        slots = self.slot_seed.to(device=candidates.device, dtype=candidates.dtype).expand(batch, -1, -1)
        parent_owner: Tensor | None = None
        parent_null: Tensor | None = None
        read: Tensor | None = None
        for _ in range(self.iterations):
            parent_owner, _, parent_null, read = self._competition(
                slots, candidates, validity, candidate_prior
            )
            update = torch.einsum("bkn,bnh->bkh", read.to(dtype=candidates.dtype), candidates)
            update, _ = smooth_rms_contract(update, self.maximum_update_rms)
            next_slots = self.gru(
                update.reshape(batch * self.objects, self.hidden).float(),
                slots.reshape(batch * self.objects, self.hidden).float(),
            ).reshape(batch, self.objects, self.hidden).to(dtype=slots.dtype)
            ffn, _ = smooth_rms_contract(self.update_ffn(next_slots), self.maximum_update_rms)
            slots = next_slots + ffn
        if (
            parent_owner is None
            or parent_null is None
            or read is None
        ):
            raise RuntimeError("object binding did not execute")
        # G3 is a bounded correction over the actual G2/binder posterior.  Its
        # zero initialization makes the initial graph an exact identity.
        pair = torch.cat(
            (
                slots[:, None].expand(-1, count, -1, -1),
                candidates[:, :, None].expand(-1, -1, self.objects, -1),
            ),
            dim=-1,
        )
        residual = 0.50 * torch.tanh(self.g3_residual(pair).squeeze(-1).float())
        # Correct the conditional K+null owner posterior.  Local-hypothesis
        # prior and physical validity remain outside this softmax, so a zero
        # G3 residual preserves both the parent posterior and its exact mass
        # semantics.
        corrected = torch.softmax(
            parent_owner.clamp_min(1e-8).log()
            + torch.cat((residual, torch.zeros_like(parent_null[..., None])), dim=-1),
            dim=-1,
        )
        valid = validity.float().clamp(0.0, 1.0)
        prior = candidate_prior.float().clamp_min(0.0)
        assignment = corrected[..., : self.objects] * valid * prior
        null_assignment = (
            corrected[..., self.objects] * valid[..., 0]
            + (1.0 - valid[..., 0])
        ) * prior[..., 0]
        read = assignment.transpose(1, 2)
        read = read / read.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        def aggregate(value: Tensor, weight: Tensor = read) -> Tensor:
            flat = value.reshape(batch, count, int(value.shape[-1]))
            return torch.einsum("bkn,bnd->bkd", weight.to(dtype=flat.dtype), flat)

        def typed_verify(
            name: str,
            value: Tensor,
        ) -> tuple[Tensor, Tensor, Tensor]:
            flat = value.reshape(batch, count, int(value.shape[-1]))
            parent_value = aggregate(value)
            pair = torch.cat(
                (
                    parent_value[:, :, None].expand(-1, -1, count, -1),
                    flat[:, None].expand(-1, self.objects, -1, -1),
                ),
                dim=-1,
            )
            residual = 0.50 * torch.tanh(
                self.typed_verifier[name](pair).squeeze(-1).float()
            )
            # Multiplication by ``read`` makes the typed posterior absolutely
            # continuous with respect to the physical object support: a typed
            # verifier cannot resurrect an invalid/null candidate.
            typed_read = read.float() * residual.exp()
            typed_read = typed_read / typed_read.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-6)
            # ``assignment.sum(dim=1)`` is [B,K]; keep exactly the physical
            # allocation owned by each K object while redistributing evidence
            # only inside that support.
            object_mass = assignment.sum(dim=1)
            typed_joint = typed_read * object_mass[..., None]
            typed_value = torch.einsum(
                "bkn,bnd->bkd", typed_read.to(dtype=flat.dtype), flat
            )
            return typed_value, typed_read, typed_joint

        content = aggregate(chart.candidate_content)
        semantic, semantic_read, semantic_assignment = typed_verify(
            "semantic", chart.candidate_semantic
        )
        appearance, appearance_read, appearance_assignment = typed_verify(
            "appearance", chart.candidate_appearance
        )
        geometry, geometry_read, geometry_assignment = typed_verify(
            "geometry", chart.candidate_geometry
        )
        support = aggregate(chart.candidate_support[..., None])
        # Presence is not an object's fraction of the complete chart.  That
        # quantity systematically penalizes small objects and, when reused as
        # P2 validity, suppresses every one of K mutually exclusive slots.
        # Instead measure object-vs-null confidence only where that object's
        # own read posterior places mass.  This remains soft and naturally
        # becomes zero when an object receives no valid candidate support.
        object_vs_null = corrected[..., : self.objects] / (
            corrected[..., : self.objects]
            + corrected[..., self.objects, None]
        ).clamp_min(1e-6)
        existence = (
            read * object_vs_null.transpose(1, 2)
        ).sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
        object_validity = (
            read * valid[..., 0][:, None]
        ).sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
        valid_prior_mass = (valid * prior).sum(dim=1).clamp_min(1e-6)
        allocation_share = assignment.sum(dim=1)[..., None] / valid_prior_mass[:, None]
        structured_assignment = assignment.transpose(1, 2).reshape(
            batch, self.objects, *candidate_shape
        )
        structured_semantic_assignment = semantic_assignment.reshape(
            batch, self.objects, *candidate_shape
        )
        structured_appearance_assignment = appearance_assignment.reshape(
            batch, self.objects, *candidate_shape
        )
        structured_geometry_assignment = geometry_assignment.reshape(
            batch, self.objects, *candidate_shape
        )
        structured_null = null_assignment.reshape(batch, *candidate_shape)

        def camera_aggregate(value: Tensor, weight: Tensor) -> Tensor:
            """Aggregate inside each real camera without recreating C later."""

            cameras = int(value.shape[1])
            flat_value = value.reshape(
                batch, cameras, -1, int(value.shape[-1])
            )
            flat_weight = weight.reshape(
                batch, self.objects, cameras, -1
            ).float()
            numerator = torch.einsum(
                "bkcn,bcnd->bkcd",
                flat_weight.to(dtype=flat_value.dtype),
                flat_value,
            )
            return numerator / flat_weight.sum(
                dim=-1, keepdim=True
            ).to(dtype=numerator.dtype).clamp_min(1e-6)

        camera_coordinates = camera_aggregate(
            chart.candidate_coordinates,
            structured_geometry_assignment,
        )
        camera_transport_prior = camera_aggregate(
            chart.candidate_transport_prior,
            structured_geometry_assignment,
        )
        camera_support = camera_aggregate(
            chart.candidate_support[..., None],
            structured_geometry_assignment,
        )
        camera_validity = camera_aggregate(
            chart.candidate_validity,
            structured_assignment,
        ).clamp(0.0, 1.0)
        chart_assignment = structured_assignment.sum(dim=-1)
        # ``chart_read`` is the reverse lookup used by Teacher/P2 and is
        # normalized over space for each object.  It is *not* a per-cell owner
        # posterior.  The old draft used it for reconstruction, which divided
        # every object's value by the complete chart area and made the object
        # reconstruction pressure almost vanish.
        chart_read = chart_assignment / chart_assignment.flatten(2).sum(
            dim=-1
        )[..., None, None, None].clamp_min(1e-6)
        owner_prior_per_cell = chart.candidate_owner_prior.float().sum(
            dim=-1
        ).clamp_min(1e-6)
        chart_owner = chart_assignment.float() / owner_prior_per_cell[:, None]
        coordinate_weight = (
            chart.candidate_validity.float()
            * chart.candidate_owner_prior[..., None].float()
        )
        chart_coordinate = (
            chart.candidate_coordinates.float() * coordinate_weight
        ).sum(dim=-2) / coordinate_weight.sum(dim=-2).clamp_min(1e-6)
        position = _coordinate_basis(
            chart_coordinate.to(dtype=chart.candidate_coordinates.dtype), 16
        )
        # The exported full-DINO object content is itself the reconstruction
        # base.  A zero-initialized slot residual may add object-specific
        # detail, while the shared coordinate decoder accounts for smooth
        # within-object variation.  This prevents a private hidden slot from
        # satisfying the loss while the W-visible object content remains weak.
        decoded_slot = content + self.decode_content_residual(slots)
        decoded_position = self.decode_position(position)
        prototype_value = decoded_slot[:, :, None, None, None, :]
        prototype_reconstruction = torch.einsum(
            "bkcyx,bkcyxd->bcyxd",
            chart_owner.to(dtype=prototype_value.dtype),
            prototype_value.expand(-1, -1, *chart_owner.shape[2:], -1),
        )
        reconstruction_value = prototype_value + decoded_position[:, None]
        reconstructed = torch.einsum(
            "bkcyx,bkcyxd->bcyxd",
            chart_owner.to(dtype=reconstruction_value.dtype),
            reconstruction_value,
        )
        target_content = chart.dino_content.detach().float()
        prototype_error = (
            prototype_reconstruction.float() - target_content
        ).square().mean()
        spatial_error = (
            reconstructed.float() - target_content
        ).square().mean()
        # A shared coordinate decoder can cheaply explain spatially smooth
        # DINO structure while every K slot remains identical.  Keep that
        # within-object refinement, but make the object-prototype clustering
        # term own most of the unchanged reconstruction budget so coordinate
        # features cannot satisfy the G objective by themselves.
        def typed_consistency(
            value: Tensor,
            typed_joint: Tensor,
            prototype: Tensor,
        ) -> Tensor:
            """Reconstruct each live candidate from the typed G read.

            This is a proximal field-specific objective on one physical K
            assignment.  It does not ask the three evidence reads to differ;
            when the same support explains two fields, equal posteriors remain
            legal.  Multiplying the dimensionless error by the detached DINO
            power keeps it inside the existing reconstruction unit/budget.
            """

            flat_value = value.reshape(batch, count, int(value.shape[-1])).float()
            joint = typed_joint.float()
            conditional_owner = joint / joint.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-6)
            reconstruction = torch.einsum(
                "bkn,bkd->bnd", conditional_owner, prototype.float()
            )
            target_power = flat_value.detach().square().mean(
                dim=-1, keepdim=True
            )
            population_floor = (
                0.10 * target_power.mean().detach()
            ).clamp_min(1e-4)
            normalized = (
                reconstruction - flat_value.detach()
            ).square().mean(dim=-1, keepdim=True) / (
                target_power + population_floor
            )
            live = (
                validity.float().reshape(batch, count, 1)
                * candidate_prior.float().reshape(batch, count, 1)
            )
            return (normalized * live).sum() / live.sum().clamp_min(1.0)

        semantic_consistency = typed_consistency(
            chart.candidate_semantic, semantic_assignment, semantic
        )
        appearance_consistency = typed_consistency(
            chart.candidate_appearance, appearance_assignment, appearance
        )
        geometry_consistency = typed_consistency(
            chart.candidate_geometry, geometry_assignment, geometry
        )
        typed_consistency_error = (
            semantic_consistency
            + appearance_consistency
            + geometry_consistency
        ) / 3.0
        dino_unit = target_content.detach().square().mean().clamp_min(1e-3)
        typed_consistency_scaled = typed_consistency_error * dino_unit
        reconstruction_error = (
            0.65 * prototype_error
            + 0.20 * spatial_error
            + 0.15 * typed_consistency_scaled
        )
        facts = ObjectFactSet(
            dense_chart=chart,
            content=content,
            semantic=semantic,
            appearance=appearance,
            geometry=geometry,
            camera_coordinates=camera_coordinates,
            camera_transport_prior=camera_transport_prior,
            camera_support=camera_support,
            camera_validity=camera_validity,
            support=support,
            existence=existence,
            validity=object_validity,
            object_to_chart=chart_read,
            candidate_assignment=structured_assignment,
            semantic_candidate_assignment=structured_semantic_assignment,
            appearance_candidate_assignment=structured_appearance_assignment,
            geometry_candidate_assignment=structured_geometry_assignment,
            null_assignment=structured_null,
            reconstructed_dino=reconstructed,
            reconstruction_error=reconstruction_error,
            typed_consistency_error=typed_consistency_error,
        )
        facts.validate()
        metrics = {
            "object_grounding_reconstruction_mse": reconstruction_error.detach(),
            "object_grounding_prototype_mse": prototype_error.detach(),
            "object_grounding_spatial_refinement_mse": spatial_error.detach(),
            "object_grounding_typed_consistency": typed_consistency_error.detach(),
            "object_grounding_typed_consistency_scaled": typed_consistency_scaled.detach(),
            "object_grounding_semantic_consistency": semantic_consistency.detach(),
            "object_grounding_appearance_consistency": appearance_consistency.detach(),
            "object_grounding_geometry_consistency": geometry_consistency.detach(),
            "object_grounding_existence_mean": existence.detach().float().mean(),
            "object_grounding_validity_mean": object_validity.detach().float().mean(),
            "object_grounding_allocation_share_mean": allocation_share.detach().float().mean(),
            # Null is a per-cell probability after summing the mutually
            # exclusive local-M hypotheses, not a mean over those hypotheses.
            "object_grounding_null_mass": structured_null.detach().float().sum(dim=-1).mean(),
            "object_grounding_mass_conservation_error": (
                (
                    structured_assignment.detach().float().sum(dim=1)
                    + structured_null.detach().float()
                ).sum(dim=-1)
                - chart.candidate_owner_prior.detach().float().sum(dim=-1)
            ).abs().mean(),
            "object_grounding_candidate_owner_entropy": normalized_entropy(
                corrected, dim=-1
            ).detach().mean(),
            "object_grounding_local_prior_entropy": normalized_entropy(
                chart.candidate_owner_prior, dim=-1
            ).detach().mean(),
            "object_grounding_chart_entropy": normalized_entropy(
                chart_read.flatten(2), dim=-1
            ).detach().mean(),
            "object_grounding_g3_parent_l1": (
                corrected.detach().float()
                - parent_owner.detach().float()
            ).abs().mean(),
            "object_grounding_object_content_pair_cosine": self._pair_cosine(content),
            "object_grounding_object_chart_pair_overlap": self._pair_overlap(
                chart_read.flatten(2)
            ),
            "object_grounding_semantic_appearance_posterior_l1": (
                0.5
                * (semantic_read.detach().float() - appearance_read.detach().float())
                .abs()
                .sum(dim=-1)
                .mean()
            ),
            "object_grounding_semantic_geometry_posterior_l1": (
                0.5
                * (semantic_read.detach().float() - geometry_read.detach().float())
                .abs()
                .sum(dim=-1)
                .mean()
            ),
            "object_grounding_appearance_geometry_posterior_l1": (
                0.5
                * (appearance_read.detach().float() - geometry_read.detach().float())
                .abs()
                .sum(dim=-1)
                .mean()
            ),
            "object_grounding_transport_prior_rms": camera_transport_prior.detach().float().square().mean().sqrt(),
            "object_grounding_camera_coordinate_variation": camera_coordinates.detach().float().std(
                dim=2, unbiased=False
            ).mean(),
        }
        return facts, metrics

    @staticmethod
    def _pair_cosine(value: Tensor) -> Tensor:
        normalized = F.normalize(value.detach().float(), dim=-1, eps=1e-4)
        similarity = torch.einsum("bkd,bjd->bkj", normalized, normalized)
        objects = int(value.shape[1])
        mask = ~torch.eye(objects, device=value.device, dtype=torch.bool)
        return similarity[:, mask].mean() if objects > 1 else similarity.new_zeros(())

    @staticmethod
    def _pair_overlap(probability: Tensor) -> Tensor:
        probability = probability.detach().float()
        overlap = torch.einsum(
            "bkn,bjn->bkj",
            probability.clamp_min(0.0).sqrt(),
            probability.clamp_min(0.0).sqrt(),
        )
        objects = int(probability.shape[1])
        mask = ~torch.eye(objects, device=probability.device, dtype=torch.bool)
        return overlap[:, mask].mean() if objects > 1 else overlap.new_zeros(())
