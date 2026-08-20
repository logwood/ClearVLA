"""Dense-chart to global-object binding for G3."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .routing import smooth_rms_contract
from .types import DenseFactChart, LocalFactSet, ObjectFactSet, normalized_entropy


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


def dense_chart_from_local_facts(local: LocalFactSet) -> DenseFactChart:
    """Preserve every local hypothesis while exposing a dense DINO target."""

    local.validate()
    # The reconstruction target must be independent of the online hypotheses
    # that are being evaluated.  Rebuilding it from ``content_slots`` makes a
    # self-consistent collapsed chart a valid target.  The current DINO chart
    # is observable online, detached by the observation compiler and carries
    # no future information.
    dense_content = local.target_dino_content
    # This is a conditional mixture over the local M hypotheses.  Candidate
    # validity is a separate Bernoulli support variable and must not be folded
    # into this distribution: doing so turns the complement of a perfectly
    # legal (for example uniform 1/M) prior into false null evidence.
    owner_prior = torch.sqrt(
        local.semantic_owner_probs.float().clamp_min(0.0)
        * local.geometry_owner_probs.float().clamp_min(0.0)
    )
    prior_mass = owner_prior.sum(dim=-1, keepdim=True)
    uniform_prior = torch.full_like(owner_prior, 1.0 / float(max(int(owner_prior.shape[-1]), 1)))
    owner_prior = torch.where(
        prior_mass > 1e-6,
        owner_prior / prior_mass.clamp_min(1e-6),
        uniform_prior,
    )
    chart = DenseFactChart(
        public_scene_base=local.public_scene_base,
        dino_content=dense_content,
        cell_observed=local.cell_observed,
        candidate_content=local.content_slots,
        candidate_semantic=local.semantic_slots,
        candidate_appearance=local.appearance_slots,
        candidate_geometry=local.geometry_slots,
        candidate_coordinates=local.slot_coordinates,
        candidate_support=local.slot_support,
        candidate_validity=local.slot_validity,
        candidate_owner_prior=owner_prior.to(dtype=local.content_slots.dtype),
        candidate_semantic_prior=local.semantic_owner_probs,
        candidate_appearance_prior=local.appearance_owner_probs,
        candidate_geometry_prior=local.geometry_owner_probs,
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
        self.slot_typed_keys = nn.ModuleList(
            nn.Linear(hidden, hidden, bias=False) for _ in range(3)
        )
        for projection in self.slot_typed_keys:
            nn.init.eye_(projection.weight)
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
        self.decode_content_residual = nn.Linear(hidden, content_dim, bias=False)
        nn.init.zeros_(self.decode_content_residual.weight)
        self.decode_position = nn.Linear(16, content_dim, bias=False)
        self.maximum_update_rms = float(maximum_update_rms)

    def _candidate_tokens(self, chart: DenseFactChart) -> Tensor:
        """Return the public audit key without re-injecting the scene chart."""

        content = self.content_key(chart.candidate_content)
        typed = (
            self.semantic_key(chart.candidate_semantic)
            + self.appearance_key(chart.candidate_appearance)
            + self.geometry_key(chart.candidate_geometry)
        ) / math.sqrt(3.0)
        coordinate = self.coordinate_key(_coordinate_basis(chart.candidate_coordinates, 16))
        return self.candidate_norm(content + typed + coordinate)

    def _candidate_key_views(self, chart: DenseFactChart) -> Tensor:
        """Return separate semantic/appearance/geometry pre-binding keys.

        The three posteriors vote on one physical K+null assignment.  They are
        not three object identities and cannot resurrect mass outside the
        resulting physical support.
        """

        content = self.content_key(chart.candidate_content)
        coordinate = self.coordinate_key(_coordinate_basis(chart.candidate_coordinates, 16))
        semantic = self.candidate_norm(content + self.semantic_key(chart.candidate_semantic))
        appearance = self.candidate_norm(
            content + self.appearance_key(chart.candidate_appearance)
        )
        geometry = self.candidate_norm(
            coordinate + self.geometry_key(chart.candidate_geometry)
        )
        return torch.stack((semantic, appearance, geometry), dim=-2)

    def _competition(
        self,
        slots: Tensor,
        candidate_views: Tensor,
        validity: Tensor,
        candidate_prior: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if candidate_views.ndim != 4 or int(candidate_views.shape[-2]) != 3:
            raise ValueError("typed candidate views must be [B,N,3,H]")
        batch, count, _, _ = candidate_views.shape
        slot_key = self.slot_norm(slots)
        typed_slot_key = torch.stack(
            tuple(projection(slot_key) for projection in self.slot_typed_keys),
            dim=2,
        )
        logits = torch.einsum(
            "bkvh,bnvh->bnkv", typed_slot_key.float(), candidate_views.float()
        ) / math.sqrt(float(self.hidden))
        null = torch.einsum(
            "bnvh,bqh->bnvq",
            candidate_views.float(),
            self.null_key.to(
                device=candidate_views.device, dtype=candidate_views.dtype
            )
            .expand(batch, -1, -1)
            .float(),
        ).permute(0, 1, 3, 2)
        # ``owner`` is conditional on one local candidate.  The local prior
        # is applied afterwards as joint mixture mass.  Only true invalidity
        # may move probability to null; ``1 - candidate_prior`` denotes other
        # hypotheses at the same cell, not absence.
        typed_logits = torch.cat((logits, null), dim=2)
        typed_owner = torch.softmax(typed_logits, dim=2)
        # One physical object identity is selected from the consensus of the
        # three typed compatibility views.  Averaging three already-normalized
        # posteriors would retain three competing object identities and merely
        # blur them after the fact; averaging bounded logits before the single
        # softmax makes semantic/appearance/geometry genuine pre-binding
        # evidence for the same K+null assignment.
        owner = torch.softmax(typed_logits.mean(dim=-1), dim=2)
        valid = validity.float().reshape(batch, count, 1).clamp(0.0, 1.0)
        prior = candidate_prior.float().reshape(batch, count, 1).clamp_min(0.0)
        object_mass = owner[..., : self.objects] * valid * prior
        null_mass = (owner[..., self.objects] * valid[..., 0] + (1.0 - valid[..., 0])) * prior[
            ..., 0
        ]
        read = object_mass.transpose(1, 2)
        read = read / read.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return owner, typed_owner.permute(0, 1, 3, 2), object_mass, null_mass, read

    def forward(
        self,
        local_facts: LocalFactSet,
        *,
        collect_diagnostics: bool = True,
    ) -> tuple[ObjectFactSet, dict[str, Tensor]]:
        chart = dense_chart_from_local_facts(local_facts)
        candidates_structured = self._candidate_tokens(chart)
        candidate_views_structured = self._candidate_key_views(chart)
        batch = int(candidates_structured.shape[0])
        candidate_shape = candidates_structured.shape[1:-1]
        count = math.prod(int(value) for value in candidate_shape)
        candidates = candidates_structured.reshape(batch, count, self.hidden)
        candidate_views = candidate_views_structured.reshape(
            batch, count, 3, self.hidden
        )
        validity = chart.candidate_validity.reshape(batch, count, 1)
        candidate_prior = chart.candidate_owner_prior.reshape(batch, count, 1)
        slots = self.slot_seed.to(
            device=candidates.device,
            dtype=candidates.dtype,
        ).expand(
            batch, -1, -1
        )
        parent_owner: Tensor | None = None
        parent_null: Tensor | None = None
        typed_parent: Tensor | None = None
        read: Tensor | None = None
        for _ in range(self.iterations):
            parent_owner, typed_parent, _, parent_null, read = self._competition(
                slots, candidate_views, validity, candidate_prior
            )
            update = torch.einsum(
                "bkn,bnh->bkh",
                read.to(dtype=candidates.dtype),
                candidates,
            )
            update, _ = smooth_rms_contract(update, self.maximum_update_rms)
            next_slots = (
                self.gru(
                    update.reshape(batch * self.objects, self.hidden).float(),
                    slots.reshape(batch * self.objects, self.hidden).float(),
                )
                .reshape(batch, self.objects, self.hidden)
                .to(dtype=slots.dtype)
            )
            ffn, _ = smooth_rms_contract(self.update_ffn(next_slots), self.maximum_update_rms)
            slots = next_slots + ffn
        # The final GRU/FFN update changes the slot queries.  Reusing the
        # pre-update posterior here would combine a stale G2 assignment with
        # a new G3 slot state.  Recompute the parent posterior once so the
        # bounded G3 correction is genuinely relative to the final binder.
        parent_owner, typed_parent, _, parent_null, read = self._competition(
            slots, candidate_views, validity, candidate_prior
        )
        if (
            parent_owner is None
            or typed_parent is None
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
            corrected[..., self.objects] * valid[..., 0] + (1.0 - valid[..., 0])
        ) * prior[..., 0]
        read = assignment.transpose(1, 2)
        read = read / read.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        def aggregate(value: Tensor, weight: Tensor = read) -> Tensor:
            flat = value.reshape(batch, count, int(value.shape[-1]))
            return torch.einsum("bkn,bnd->bkd", weight.to(dtype=flat.dtype), flat)

        def typed_reweight(
            name: str,
            value: Tensor,
        ) -> tuple[Tensor, Tensor, Tensor]:
            flat = value.reshape(batch, count, int(value.shape[-1]))
            # Parameter-free conditional reweighting inside physical support.
            # Typed compatibility already participated before K binding.  Its
            # posterior and the local-M prior may refine a read, but neither
            # can resurrect a candidate with zero corrected physical mass.
            type_index = {"semantic": 0, "appearance": 1, "geometry": 2}[name]
            typed_prior = getattr(chart, f"candidate_{name}_prior").reshape(batch, count).float()
            physical_prior = candidate_prior.reshape(batch, count).float()
            prior_ratio = typed_prior / physical_prior.clamp_min(1e-6)
            typed_object_probability = typed_parent[..., type_index, : self.objects]
            physical_object_probability = parent_owner[..., : self.objects]
            compatibility_ratio = typed_object_probability / physical_object_probability.clamp_min(
                1e-6
            )
            typed_read = (
                read.float()
                * compatibility_ratio.transpose(1, 2)
                * prior_ratio[:, None]
            )
            typed_read = typed_read / typed_read.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            # ``assignment.sum(dim=1)`` is [B,K]; keep exactly the physical
            # allocation owned by each K object while redistributing evidence
            # only inside that support.
            object_mass = assignment.sum(dim=1)
            typed_joint = typed_read * object_mass[..., None]
            typed_value = torch.einsum("bkn,bnd->bkd", typed_read.to(dtype=flat.dtype), flat)
            return typed_value, typed_read, typed_joint

        target_candidates = chart.dino_content[..., None, :].expand(
            *chart.candidate_content.shape[:-1], chart.dino_content.shape[-1]
        )
        content = aggregate(target_candidates)
        semantic, semantic_read, semantic_assignment = typed_reweight(
            "semantic", chart.candidate_semantic
        )
        appearance, appearance_read, appearance_assignment = typed_reweight(
            "appearance", chart.candidate_appearance
        )
        geometry, geometry_read, geometry_assignment = typed_reweight(
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
            corrected[..., : self.objects] + corrected[..., self.objects, None]
        ).clamp_min(1e-6)
        existence = (
            (read * object_vs_null.transpose(1, 2)).sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
        )
        object_validity = (read * valid[..., 0][:, None]).sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
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
            flat_value = value.reshape(batch, cameras, -1, int(value.shape[-1]))
            flat_weight = weight.reshape(batch, self.objects, cameras, -1).float()
            numerator = torch.einsum(
                "bkcn,bcnd->bkcd",
                flat_weight.to(dtype=flat_value.dtype),
                flat_value,
            )
            return numerator / flat_weight.sum(dim=-1, keepdim=True).to(
                dtype=numerator.dtype
            ).clamp_min(1e-6)

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
        chart_read = chart_assignment / chart_assignment.flatten(2).sum(dim=-1)[
            ..., None, None, None
        ].clamp_min(1e-6)
        owner_prior_per_cell = chart.candidate_owner_prior.float().sum(dim=-1).clamp_min(1e-6)
        chart_owner = chart_assignment.float() / owner_prior_per_cell[:, None]
        coordinate_weight = (
            chart.candidate_validity.float() * chart.candidate_owner_prior[..., None].float()
        )
        chart_coordinate = (chart.candidate_coordinates.float() * coordinate_weight).sum(
            dim=-2
        ) / coordinate_weight.sum(dim=-2).clamp_min(1e-6)
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
        reconstruction_value = prototype_value + decoded_position[:, None]
        reconstructed = torch.einsum(
            "bkcyx,bkcyxd->bcyxd",
            chart_owner.to(dtype=reconstruction_value.dtype),
            reconstruction_value,
        )
        target_content = chart.dino_content.detach().float()
        observed = chart.cell_observed.detach().float()
        reconstruction_per_cell = (
            reconstructed.float() - target_content
        ).square().mean(dim=-1, keepdim=True)
        reconstruction_error = (
            reconstruction_per_cell * observed
        ).sum() / observed.sum().clamp_min(1.0)
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
        )
        facts.validate()
        # The single dense reconstruction scalar above is the only G-side
        # objective. Everything below is a detached audit reduction.
        if not collect_diagnostics:
            return facts, {}
        metrics = {
            "object_grounding_reconstruction_mse": reconstruction_error.detach(),
            "object_grounding_dense_objective_count": reconstruction_error.new_ones(
                (), dtype=torch.float32
            ),
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
            )
            .abs()
            .mean(),
            "object_grounding_candidate_owner_entropy": normalized_entropy(corrected, dim=-1)
            .detach()
            .mean(),
            "object_grounding_local_prior_entropy": normalized_entropy(
                chart.candidate_owner_prior, dim=-1
            )
            .detach()
            .mean(),
            "object_grounding_chart_entropy": normalized_entropy(chart_read.flatten(2), dim=-1)
            .detach()
            .mean(),
            "object_grounding_global_k_binder_correction_l1": (
                corrected.detach().float() - parent_owner.detach().float()
            )
            .abs()
            .mean(),
            "object_grounding_global_k_binder_residual_rms": residual.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_grounding_prebind_typed_consensus_l1": (
                0.5
                * (
                    typed_parent.detach().float()
                    - parent_owner.detach().float()[:, :, None]
                )
                .abs()
                .sum(dim=-1)
                .mean()
            ),
            "object_grounding_prebind_semantic_appearance_l1": (
                0.5
                * (
                    typed_parent.detach().float()[:, :, 0]
                    - typed_parent.detach().float()[:, :, 1]
                )
                .abs()
                .sum(dim=-1)
                .mean()
            ),
            "object_grounding_prebind_semantic_geometry_l1": (
                0.5
                * (
                    typed_parent.detach().float()[:, :, 0]
                    - typed_parent.detach().float()[:, :, 2]
                )
                .abs()
                .sum(dim=-1)
                .mean()
            ),
            "object_grounding_object_content_pair_cosine": self._pair_cosine(content),
            "object_grounding_object_chart_pair_overlap": self._pair_overlap(chart_read.flatten(2)),
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
            "object_grounding_transport_prior_rms": camera_transport_prior.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_grounding_camera_coordinate_variation": camera_coordinates.detach()
            .float()
            .std(dim=2, unbiased=False)
            .mean(),
            "object_grounding_candidate_key_rms": candidate_views.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_grounding_full_dino_value_rms": content.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
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
