"""Dense-chart to global-object binding for G3."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .routing import smooth_rms_contract
from .types import DenseFactChart, LocalFactSet, ObjectFactSet, normalized_entropy


def _finite_log_measure(measure: Tensor) -> Tensor:
    """Return finite FP32 logs; zero support is represented by finite zero."""

    measure_f = measure.float().clamp_min(0.0)
    support = measure_f > 0.0
    safe = torch.where(support, measure_f, torch.ones_like(measure_f))
    return torch.where(support, safe.log(), torch.zeros_like(measure_f))


def _masked_log_softmax(
    log_measure: Tensor,
    support: Tensor,
    *,
    dim: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Normalize a finite FP32 log measure with an exact empty-row zero.

    Returns ``(probability, log_probability, log_mass, has_support)``.  The
    unsupported entries of ``log_probability`` deliberately contain finite
    zero; the boolean support is the only authority for interpreting them.
    This avoids both an all-``-inf`` reduction and a tiny numerical prior.
    """

    log_measure_f = log_measure.float()
    if tuple(log_measure_f.shape) != tuple(support.shape):
        raise ValueError("log measure and support must align")
    if not torch.isfinite(log_measure_f).all():
        raise ValueError("masked log measure must be finite before support masking")
    support_b = support.bool()
    has_support = support_b.any(dim=dim, keepdim=True)
    masked = log_measure_f.masked_fill(~support_b, -torch.inf)
    safe_masked = torch.where(has_support, masked, torch.zeros_like(masked))
    log_mass = torch.logsumexp(safe_masked, dim=dim, keepdim=True)
    log_probability = torch.where(
        support_b & has_support,
        safe_masked - log_mass,
        torch.zeros_like(safe_masked),
    )
    probability = torch.where(
        support_b & has_support,
        log_probability.exp(),
        torch.zeros_like(log_probability),
    )
    log_mass = torch.where(has_support, log_mass, torch.zeros_like(log_mass))
    return probability, log_probability, log_mass, has_support


def _observable_log_read(
    base_log_measure: Tensor,
    observable_validity: Tensor,
    *,
    dim: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return a conditional read and separate observable availability.

    ``base_log_measure`` owns the allocation prior.  ``observable_validity``
    is a physical support probability in ``[0,1]`` and may remove mass, but it
    may not be replaced by an epsilon.  The returned tuple is
    ``(conditional, conditional_log, read, availability, log_availability,
    observed_log_measure, observed_support)``.
    """

    base_log = base_log_measure.float()
    validity = observable_validity.float().clamp(0.0, 1.0)
    if tuple(base_log.shape) != tuple(validity.shape):
        raise ValueError("observable log read inputs must align")
    base_support = torch.isfinite(base_log)
    _, _, base_log_mass, base_has_support = _masked_log_softmax(
        torch.where(base_support, base_log, torch.zeros_like(base_log)),
        base_support,
        dim=dim,
    )
    observed_support = base_support & (validity > 0.0)
    validity_log = _finite_log_measure(validity)
    observed_log = base_log + validity_log
    conditional, conditional_log, observed_log_mass, observed_has_support = (
        _masked_log_softmax(
            torch.where(
                observed_support,
                observed_log,
                torch.zeros_like(observed_log),
            ),
            observed_support,
            dim=dim,
        )
    )
    has_ratio = base_has_support & observed_has_support
    log_availability = torch.where(
        has_ratio,
        observed_log_mass - base_log_mass,
        torch.zeros_like(observed_log_mass),
    )
    availability = torch.where(
        has_ratio,
        log_availability.exp().clamp(0.0, 1.0),
        torch.zeros_like(log_availability),
    )
    read = conditional * availability
    return (
        conditional,
        conditional_log,
        read,
        availability,
        log_availability,
        observed_log,
        observed_support,
    )


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


def _conditional_k_reconstruction_assignment(
    conditional_owner: Tensor,
    candidate_prior: Tensor,
    candidate_validity: Tensor,
) -> Tensor:
    """Build a null-independent reconstruction assignment over local M and K.

    The online K+null posterior and its absolute object mass remain untouched.
    Reconstruction instead consumes the already-computed conditional-K
    posterior, normalized local-hypothesis prior and observable validity.  The
    input is already a softmax-normalized conditional distribution; this
    helper deliberately performs no further normalization.  Learned null mass
    is therefore absent from both the value and the Jacobian, while true
    candidate invalidity may still return a cell to the protected public base.
    """

    if conditional_owner.ndim != 3:
        raise ValueError("conditional K owner must be [B,N,K]")
    if int(conditional_owner.shape[-1]) < 1:
        raise ValueError("conditional K owner requires at least one slot")
    if tuple(candidate_prior.shape) != tuple(conditional_owner.shape[:-1]) + (1,):
        raise ValueError("candidate prior must align as [B,N,1]")
    if tuple(candidate_validity.shape) != tuple(candidate_prior.shape):
        raise ValueError("candidate validity must align as [B,N,1]")
    return (
        conditional_owner.float().clamp_min(0.0)
        * candidate_prior.float().clamp_min(0.0)
        * candidate_validity.float().clamp(0.0, 1.0)
    )


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
    if (
        local.semantic_owner_log_probs is not None
        and local.geometry_owner_log_probs is not None
    ):
        semantic_log = local.semantic_owner_log_probs.float()
        geometry_log = local.geometry_owner_log_probs.float()
        owner_log_prior = torch.log_softmax(
            0.5 * (semantic_log + geometry_log),
            dim=-1,
        )
        owner_prior = owner_log_prior.exp()
        semantic_probability = semantic_log.exp()
        geometry_probability = geometry_log.exp()
    else:
        # Compact legacy fixtures do not carry the Schema39 log boundary.
        # Their values are already FP32 in tests; the active restored path is
        # rejected above unless it supplies producer-owned logs.
        semantic_probability = local.semantic_owner_probs.float().clamp_min(0.0)
        geometry_probability = local.geometry_owner_probs.float().clamp_min(0.0)
        owner_prior = torch.sqrt(semantic_probability * geometry_probability)
        prior_mass = owner_prior.sum(dim=-1, keepdim=True)
        uniform_prior = torch.full_like(
            owner_prior,
            1.0 / float(max(int(owner_prior.shape[-1]), 1)),
        )
        owner_prior = torch.where(
            prior_mass > 0.0,
            owner_prior
            / torch.where(
                prior_mass > 0.0,
                prior_mass,
                torch.ones_like(prior_mass),
            ),
            uniform_prior,
        )
        owner_log_prior = _finite_log_measure(owner_prior)
    appearance_probability = (
        local.appearance_owner_log_probs.float().exp()
        if local.appearance_owner_log_probs is not None
        else local.appearance_owner_probs.float().clamp_min(0.0)
    )
    semantic_log_prior = (
        local.semantic_owner_log_probs.float()
        if local.semantic_owner_log_probs is not None
        else _finite_log_measure(semantic_probability)
    )
    appearance_log_prior = (
        local.appearance_owner_log_probs.float()
        if local.appearance_owner_log_probs is not None
        else _finite_log_measure(appearance_probability)
    )
    geometry_log_prior = (
        local.geometry_owner_log_probs.float()
        if local.geometry_owner_log_probs is not None
        else _finite_log_measure(geometry_probability)
    )
    chart = DenseFactChart(
        g3_public_scene_audit=local.public_scene_base,
        dino_content=dense_content,
        cell_observed=local.cell_observed,
        candidate_content=local.content_slots,
        candidate_semantic=local.semantic_slots,
        candidate_appearance=local.appearance_slots,
        candidate_geometry=local.geometry_slots,
        candidate_coordinates=local.slot_coordinates,
        candidate_support=local.slot_support,
        candidate_validity=local.slot_validity,
        candidate_owner_prior=owner_prior,
        candidate_owner_log_prior=owner_log_prior,
        candidate_semantic_prior=semantic_probability,
        candidate_appearance_prior=appearance_probability,
        candidate_geometry_prior=geometry_probability,
        candidate_semantic_log_prior=semantic_log_prior,
        candidate_appearance_log_prior=appearance_log_prior,
        candidate_geometry_log_prior=geometry_log_prior,
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
        # Restore V120's useful reconstruction bandwidth, but make the
        # decoded slot residual part of the one exported object value.  The
        # historical implementation decoded a private slot only for the
        # reconstruction loss, so S/Teacher/W never received what that loss
        # learned.  Zero initialization preserves the former online value at
        # step zero while ordinary reconstruction gradients can enrich it.
        # New zero-residual capacity must not silently reshuffle every module
        # constructed after the grounder. Linear construction initializes
        # eagerly, so preserve the global CPU RNG around these new owners.
        construction_rng = torch.get_rng_state()
        self.decode_content_residual = nn.Linear(hidden, content_dim, bias=False)
        self.decode_public_position = nn.Linear(16, content_dim, bias=False)
        torch.set_rng_state(construction_rng)
        nn.init.zeros_(self.decode_content_residual.weight)
        # This shared coordinate term is a protected public spatial basis. It
        # cannot encode K identity and is centred below, so it cannot replace
        # either the public scene value or an object-owned slot residual. Zero
        # initialization keeps the former reconstruction and online values
        # exact at step zero while the dense objective supplies ordinary
        # gradients to both new decoders.
        nn.init.zeros_(self.decode_public_position.weight)
        self.maximum_update_rms = float(maximum_update_rms)

    def _candidate_tokens(self, chart: DenseFactChart) -> Tensor:
        """Return the sole content value used by global-K slot updates."""

        content = self.content_key(chart.candidate_content)
        return self.candidate_norm(content)

    def _candidate_key_views(self, chart: DenseFactChart) -> Tensor:
        """Return typed keys without copying full content into either sidecar.

        Content alone owns the K+null base. Semantic and appearance may only
        provide bounded conditional-K corrections, while geometry is retained
        solely for a typed read inside the resulting physical support.
        """

        coordinate = self.coordinate_key(_coordinate_basis(chart.candidate_coordinates, 16))
        semantic = self.candidate_norm(self.semantic_key(chart.candidate_semantic))
        appearance = self.candidate_norm(self.appearance_key(chart.candidate_appearance))
        geometry = self.candidate_norm(
            coordinate + self.geometry_key(chart.candidate_geometry)
        )
        return torch.stack((semantic, appearance, geometry), dim=-2)

    def _competition(
        self,
        slots: Tensor,
        candidate_content: Tensor,
        candidate_views: Tensor,
        validity: Tensor,
        candidate_log_prior: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        if candidate_content.ndim != 3:
            raise ValueError("content candidate view must be [B,N,H]")
        if candidate_views.ndim != 4 or int(candidate_views.shape[-2]) != 3:
            raise ValueError("typed candidate views must be [B,N,3,H]")
        batch, count, _, _ = candidate_views.shape
        if tuple(candidate_content.shape) != (batch, count, self.hidden):
            raise ValueError("content and typed candidate views must align")
        slot_key = self.slot_norm(slots)
        base_k_logits = torch.einsum(
            "bkh,bnh->bnk", slot_key.float(), candidate_content.float()
        ) / math.sqrt(float(self.hidden))
        base_null_logit = torch.einsum(
            "bnh,bqh->bnq",
            candidate_content.float(),
            self.null_key.to(
                device=candidate_content.device,
                dtype=candidate_content.dtype,
            )
            .expand(batch, -1, -1)
            .float(),
        )
        base_log_owner = torch.log_softmax(
            torch.cat((base_k_logits, base_null_logit), dim=-1),
            dim=-1,
        )
        base_k_log_conditional = torch.log_softmax(base_k_logits, dim=-1)
        typed_slot_key = torch.stack(
            tuple(projection(slot_key) for projection in self.slot_typed_keys),
            dim=2,
        )
        logits = torch.einsum(
            "bkvh,bnvh->bnkv", typed_slot_key.float(), candidate_views.float()
        ) / math.sqrt(float(self.hidden))
        # Every typed correction is bounded independently. Semantic and
        # appearance together can move a conditional-K logit by at most 0.5;
        # neither can alter content's exact object-vs-null mass. Geometry gets
        # the same bounded typed posterior for support-internal reads but is
        # deliberately excluded from the physical owner below.
        typed_correction = 0.25 * torch.tanh(logits)

        base_k_log_mass = torch.logsumexp(
            base_log_owner[..., : self.objects],
            dim=-1,
            keepdim=True,
        )

        def conditional_log_owner(correction: Tensor) -> Tensor:
            conditional_log = torch.log_softmax(
                base_k_log_conditional + correction,
                dim=-1,
            )
            return torch.cat(
                (
                    base_k_log_mass + conditional_log,
                    base_log_owner[..., self.objects :],
                ),
                dim=-1,
            )

        typed_log_owner = torch.stack(
            tuple(
                conditional_log_owner(typed_correction[..., index])
                for index in range(3)
            ),
            dim=2,
        )
        typed_owner = typed_log_owner.exp()
        # ``owner`` is conditional on one local candidate.  The local prior
        # is applied afterwards as joint mixture mass.  Only true invalidity
        # may move probability to null; ``1 - candidate_prior`` denotes other
        # hypotheses at the same cell, not absence.
        log_owner = conditional_log_owner(
            typed_correction[..., 0] + typed_correction[..., 1]
        )
        owner = log_owner.exp()
        valid = validity.float().reshape(batch, count, 1).clamp(0.0, 1.0)
        log_prior = candidate_log_prior.float().reshape(batch, count, 1)
        if not torch.isfinite(log_prior).all():
            raise ValueError("candidate log prior must remain finite FP32")
        prior = log_prior.exp()
        base_log_assignment = log_owner[..., : self.objects] + log_prior
        (
            _conditional_read,
            _conditional_log_read,
            read,
            _availability,
            _log_availability,
            observed_log_assignment,
            observed_support,
        ) = _observable_log_read(
            base_log_assignment.transpose(1, 2),
            valid[..., 0][:, None].expand(-1, self.objects, -1),
            dim=-1,
        )
        object_mass = torch.where(
            observed_support.transpose(1, 2),
            observed_log_assignment.transpose(1, 2).exp(),
            torch.zeros_like(observed_log_assignment.transpose(1, 2)),
        )
        null_base_mass = (log_owner[..., self.objects] + log_prior[..., 0]).exp()
        null_mass = (
            null_base_mass * valid[..., 0]
            + prior[..., 0] * (1.0 - valid[..., 0])
        )
        return (
            owner,
            typed_owner,
            object_mass,
            null_mass,
            read,
            log_owner,
            typed_log_owner,
        )

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
        candidate_log_prior = chart.candidate_owner_log_prior.reshape(
            batch, count, 1
        )
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
            (
                parent_owner,
                typed_parent,
                _,
                parent_null,
                read,
                _parent_log_owner,
                _typed_parent_log,
            ) = self._competition(
                slots,
                candidates,
                candidate_views,
                validity,
                candidate_log_prior,
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
        (
            parent_owner,
            typed_parent,
            _,
            parent_null,
            read,
            parent_log_owner,
            typed_parent_log,
        ) = self._competition(
            slots,
            candidates,
            candidate_views,
            validity,
            candidate_log_prior,
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
        raw_residual = 0.50 * torch.tanh(
            self.g3_residual(pair).squeeze(-1).float()
        )
        # Correct the conditional K+null owner posterior.  Local-hypothesis
        # prior and physical validity remain outside this softmax, so a zero
        # G3 residual preserves both the parent posterior and its exact mass
        # semantics.
        # G3 may refine *which* K object owns a candidate, but it must not
        # change the G2 object-vs-null decision.  Correcting K+null in one
        # softmax made any common-mode K residual an existence logit and was
        # the structural source of the observed global-K collapse.
        parent_k_log_mass = torch.logsumexp(
            parent_log_owner[..., : self.objects],
            dim=-1,
            keepdim=True,
        )
        parent_k_mass = parent_k_log_mass.exp()
        parent_k_log_conditional = (
            parent_log_owner[..., : self.objects] - parent_k_log_mass
        )
        parent_k_conditional = parent_k_log_conditional.exp()
        # A candidate-local scalar subtracted from every K logit is exactly a
        # softmax gauge: it changes neither the corrected posterior nor its
        # gradient.  The former post-tanh, parent-weighted subtraction only
        # made the reported residual exceed the declared [-0.5, 0.5] bound.
        # Keep the actual bounded logits as the sole G3 correction.
        corrected_k_log_conditional = torch.log_softmax(
            parent_k_log_conditional + raw_residual,
            dim=-1,
        )
        corrected_k_conditional = corrected_k_log_conditional.exp()
        corrected_log_owner = torch.cat(
            (
                parent_k_log_mass + corrected_k_log_conditional,
                parent_log_owner[..., self.objects :],
            ),
            dim=-1,
        )
        corrected = corrected_log_owner.exp()
        valid = validity.float().clamp(0.0, 1.0)
        prior = candidate_prior.float().clamp_min(0.0)
        reconstruction_assignment = _conditional_k_reconstruction_assignment(
            corrected_k_conditional,
            prior,
            valid,
        )
        base_log_assignment = (
            corrected_log_owner[..., : self.objects] + candidate_log_prior
        )
        (
            conditional_read,
            conditional_log_read,
            read,
            object_availability,
            log_chart_availability,
            observed_log_assignment,
            observed_support,
        ) = _observable_log_read(
            base_log_assignment.transpose(1, 2),
            valid[..., 0][:, None].expand(-1, self.objects, -1),
            dim=-1,
        )
        assignment = torch.where(
            observed_support.transpose(1, 2),
            observed_log_assignment.transpose(1, 2).exp(),
            torch.zeros_like(observed_log_assignment.transpose(1, 2)),
        )
        null_assignment = (
            corrected[..., self.objects] * valid[..., 0] + (1.0 - valid[..., 0])
        ) * prior[..., 0]

        def aggregate(value: Tensor, weight: Tensor = read) -> Tensor:
            flat = value.reshape(batch, count, int(value.shape[-1]))
            return torch.einsum("bkn,bnd->bkd", weight.to(dtype=flat.dtype), flat)

        def typed_reweight(
            name: str,
            value: Tensor,
        ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
            flat = value.reshape(batch, count, int(value.shape[-1]))
            # Parameter-free conditional reweighting inside physical support.
            # Typed compatibility already participated before K binding.  Its
            # posterior and the local-M prior may refine a read, but neither
            # can resurrect a candidate with zero corrected physical mass.
            type_index = {"semantic": 0, "appearance": 1, "geometry": 2}[name]
            typed_prior = getattr(chart, f"candidate_{name}_prior").reshape(
                batch, count
            ).float()
            typed_log_prior = getattr(
                chart,
                f"candidate_{name}_log_prior",
            ).reshape(batch, count).float()
            typed_object_log_probability = typed_parent_log[
                ..., type_index, : self.objects
            ].transpose(1, 2)
            # The physical read is the immutable support.  Typed evidence may
            # only reweight inside it; dividing by the physical posterior or
            # physical prior algebraically cancelled exactly the constraint
            # this boundary was supposed to preserve.
            typed_log_measure = (
                conditional_log_read
                + typed_object_log_probability
                + typed_log_prior[:, None]
            )
            typed_support = observed_support & (typed_prior[:, None] > 0.0)
            (
                normalized_typed_read,
                normalized_typed_log_read,
                _,
                typed_has_support,
            ) = _masked_log_softmax(
                torch.where(
                    typed_support,
                    typed_log_measure,
                    torch.zeros_like(typed_log_measure),
                ),
                typed_support,
                dim=-1,
            )
            typed_read = torch.where(
                typed_has_support,
                normalized_typed_read,
                conditional_read.float(),
            )
            typed_log_read = torch.where(
                typed_has_support,
                normalized_typed_log_read,
                conditional_log_read,
            )
            # ``assignment.sum(dim=1)`` is [B,K]; keep exactly the physical
            # allocation owned by each K object while redistributing evidence
            # only inside that support.
            object_mass = assignment.sum(dim=1)
            typed_joint = typed_read * object_mass[..., None]
            typed_value = torch.einsum(
                "bkn,bnd->bkd",
                typed_read.to(dtype=flat.dtype),
                flat,
            ) * object_availability.to(dtype=flat.dtype)
            return typed_value, typed_read, typed_log_read, typed_joint

        target_candidates = chart.dino_content[..., None, :].expand(
            *chart.candidate_content.shape[:-1], chart.dino_content.shape[-1]
        )
        aggregated_content = aggregate(target_candidates)
        canonical_slot_residual = self.decode_content_residual(slots) * (
            object_availability.to(dtype=slots.dtype)
        )
        content = aggregated_content + canonical_slot_residual
        semantic, semantic_read, _semantic_log_read, semantic_assignment = typed_reweight(
            "semantic", chart.candidate_semantic
        )
        appearance, appearance_read, _appearance_log_read, appearance_assignment = typed_reweight(
            "appearance", chart.candidate_appearance
        )
        geometry, geometry_read, geometry_log_read, geometry_assignment = typed_reweight(
            "geometry", chart.candidate_geometry
        )
        support = aggregate(chart.candidate_support[..., None])
        # Presence is not an object's fraction of the complete chart.  That
        # quantity systematically penalizes small objects and, when reused as
        # P2 validity, suppresses every one of K mutually exclusive slots.
        # Instead measure object-vs-null confidence only where that object's
        # own read posterior places mass.  This remains soft and naturally
        # becomes zero when an object receives no valid candidate support.
        object_null_mass = (
            corrected[..., : self.objects] + corrected[..., self.objects, None]
        )
        object_vs_null = torch.where(
            object_null_mass > 0.0,
            corrected[..., : self.objects]
            / torch.where(
                object_null_mass > 0.0,
                object_null_mass,
                torch.ones_like(object_null_mass),
            ),
            torch.zeros_like(corrected[..., : self.objects]),
        )
        existence = (
            (
                conditional_read * object_vs_null.transpose(1, 2)
            ).sum(dim=-1, keepdim=True)
            * object_availability
        ).clamp(0.0, 1.0)
        object_validity = object_availability
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
        structured_geometry_log_read = geometry_log_read.reshape(
            batch, self.objects, *candidate_shape
        )
        structured_observed_support = observed_support.reshape(
            batch, self.objects, *candidate_shape
        )
        structured_base_log_assignment = base_log_assignment.transpose(
            1, 2
        ).reshape(batch, self.objects, *candidate_shape)
        structured_observed_log_assignment = observed_log_assignment.reshape(
            batch, self.objects, *candidate_shape
        )
        structured_null = null_assignment.reshape(batch, *candidate_shape)
        structured_reconstruction_assignment = reconstruction_assignment.transpose(
            1, 2
        ).reshape(batch, self.objects, *candidate_shape)

        def camera_aggregate(
            value: Tensor,
            log_weight: Tensor,
            weight_support: Tensor,
        ) -> Tensor:
            """Aggregate inside each real camera without recreating C later."""

            cameras = int(value.shape[1])
            flat_value = value.reshape(batch, cameras, -1, int(value.shape[-1]))
            flat_log_weight = log_weight.reshape(
                batch, self.objects, cameras, -1
            )
            flat_support = weight_support.reshape(
                batch, self.objects, cameras, -1
            )
            camera_probability, _, _, _ = _masked_log_softmax(
                torch.where(
                    flat_support,
                    flat_log_weight,
                    torch.zeros_like(flat_log_weight),
                ),
                flat_support,
                dim=-1,
            )
            return torch.einsum(
                "bkcn,bcnd->bkcd",
                camera_probability.to(dtype=flat_value.dtype),
                flat_value,
            )

        camera_coordinates = camera_aggregate(
            chart.candidate_coordinates,
            structured_geometry_log_read,
            structured_observed_support,
        )
        camera_transport_prior = camera_aggregate(
            chart.candidate_transport_prior,
            structured_geometry_log_read,
            structured_observed_support,
        )
        camera_support = camera_aggregate(
            chart.candidate_support[..., None],
            structured_geometry_log_read,
            structured_observed_support,
        )
        cameras = int(chart.candidate_content.shape[1])
        flat_base_log_assignment = structured_base_log_assignment.reshape(
            batch, self.objects, cameras, -1
        )
        flat_observed_log_assignment = structured_observed_log_assignment.reshape(
            batch, self.objects, cameras, -1
        )
        flat_observed_support = structured_observed_support.reshape(
            batch, self.objects, cameras, -1
        )
        base_support = torch.isfinite(flat_base_log_assignment)
        _, _, camera_base_log_mass, camera_base_support = _masked_log_softmax(
            torch.where(
                base_support,
                flat_base_log_assignment,
                torch.zeros_like(flat_base_log_assignment),
            ),
            base_support,
            dim=-1,
        )
        _, _, camera_joint_log_mass, camera_joint_support = _masked_log_softmax(
            torch.where(
                flat_observed_support,
                flat_observed_log_assignment,
                torch.zeros_like(flat_observed_log_assignment),
            ),
            flat_observed_support,
            dim=-1,
        )
        camera_chart_availability = torch.where(
            camera_base_support & camera_joint_support,
            (camera_joint_log_mass - camera_base_log_mass).exp().clamp(0.0, 1.0),
            torch.zeros_like(camera_joint_log_mass),
        ).clamp(0.0, 1.0)
        _, camera_evidence_log_mass, total_base_log_mass, total_base_support = (
            _masked_log_softmax(
                camera_base_log_mass[..., 0],
                camera_base_support[..., 0],
                dim=2,
            )
        )
        camera_evidence_mass = torch.where(
            camera_base_support,
            camera_evidence_log_mass.exp()[..., None],
            torch.zeros_like(camera_base_log_mass),
        )
        camera_weight_support = camera_joint_support & total_base_support[..., None]
        log_camera_weight = torch.where(
            camera_weight_support,
            camera_joint_log_mass - total_base_log_mass[..., None],
            torch.zeros_like(camera_joint_log_mass),
        )
        camera_weight = torch.where(
            camera_weight_support,
            log_camera_weight.exp(),
            torch.zeros_like(log_camera_weight),
        )
        # ``chart_read`` is the reverse lookup used by Teacher/P2 and is
        # normalized over space for each object.  It is *not* a per-cell owner
        # posterior.  The old draft used it for reconstruction, which divided
        # every object's value by the complete chart area and made the object
        # reconstruction pressure almost vanish.
        chart_read = conditional_read.reshape(
            batch, self.objects, *candidate_shape
        ).sum(dim=-1)
        reconstruction_owner = structured_reconstruction_assignment.sum(dim=-1)
        reconstruction_object_mass = reconstruction_owner.sum(dim=1, keepdim=True)
        target_content = chart.dino_content.detach().float()
        observed = chart.cell_observed.detach().float()
        # The reconstruction may only use the exported object content.  A
        # protected public mean explains camera-wide common content; K owns
        # only object-specific residuals.  Reconstruction uses the conditional
        # K owner rather than absolute object-vs-null mass: a learned null is a
        # legal routing hypothesis, but it cannot switch off the only pressure
        # that makes the exported K content identifiable.  The slot decoder
        # has already been folded into ``content`` above, so no private value
        # exists between this loss and S/Teacher/W/P2.
        public_content = (
            target_content * observed
        ).sum(dim=(1, 2, 3), keepdim=True) / observed.sum(
            dim=(1, 2, 3), keepdim=True
        ).clamp_min(1.0)
        coordinate_weight = (
            chart.candidate_validity.float()
            * chart.candidate_owner_prior[..., None].float()
        )
        chart_coordinate = (
            chart.candidate_coordinates.float() * coordinate_weight
        ).sum(dim=-2) / coordinate_weight.sum(dim=-2).clamp_min(1e-6)
        public_position = self.decode_public_position(
            _coordinate_basis(
                chart_coordinate.to(dtype=chart.candidate_coordinates.dtype),
                16,
            )
        ).float()
        # Keep the spatial term strictly zero-mean over observed cells.  The
        # scene-wide mean remains owned by public_content; position explains
        # only shared within-scene variation.
        public_position = public_position - (
            public_position * observed
        ).sum(dim=(1, 2, 3), keepdim=True) / observed.sum(
            dim=(1, 2, 3), keepdim=True
        ).clamp_min(1.0)
        object_residual = content.float() - public_content[:, 0, 0, 0, None, :]
        reconstructed = public_content + public_position + torch.einsum(
            "bkcyx,bkd->bcyxd",
            reconstruction_owner,
            object_residual,
        )
        reconstructed = reconstructed.to(dtype=chart.dino_content.dtype)
        reconstruction_per_cell = (
            reconstructed.float() - target_content
        ).square().mean(dim=-1, keepdim=True)
        reconstruction_error = (
            reconstruction_per_cell * observed
        ).sum() / observed.sum().clamp_min(1.0)
        facts = ObjectFactSet(
            dense_chart=chart,
            public_content=public_content[:, 0, 0, 0].to(dtype=content.dtype),
            content=content,
            semantic=semantic,
            appearance=appearance,
            geometry=geometry,
            camera_coordinates=camera_coordinates,
            camera_transport_prior=camera_transport_prior,
            camera_support=camera_support,
            camera_chart_availability=camera_chart_availability,
            camera_evidence_mass=camera_evidence_mass,
            log_camera_weight=log_camera_weight,
            support=support,
            existence=existence,
            chart_availability=object_validity,
            log_chart_availability=log_chart_availability,
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
        support_audit = camera_support.detach().float()[..., 0].flatten()
        evidence_audit = camera_evidence_mass.detach().float()[..., 0].flatten()
        support_centered = support_audit - support_audit.mean()
        evidence_centered = evidence_audit - evidence_audit.mean()
        support_evidence_correlation = (
            (support_centered * evidence_centered).sum()
            / torch.sqrt(
                support_centered.square().sum()
                * evidence_centered.square().sum()
                + 1e-12
            )
        )
        content_innovation = facts.content_innovation.detach().float()
        reconstruction_owner_active = (
            reconstruction_object_mass.detach().float() > 1.0e-6
        )
        reconstruction_conditional_owner = torch.where(
            reconstruction_owner_active,
            reconstruction_owner.detach().float()
            / reconstruction_object_mass.detach().float().clamp_min(1.0e-6),
            torch.zeros_like(reconstruction_owner.detach().float()),
        )
        reconstruction_owner_entropy = normalized_entropy(
            reconstruction_conditional_owner, dim=1
        )
        reconstruction_owner_entropy = (
            reconstruction_owner_entropy
            * reconstruction_owner_active[:, 0].to(
                dtype=reconstruction_owner_entropy.dtype
            )
        ).sum() / reconstruction_owner_active[:, 0].float().sum().clamp_min(1.0)
        # A small correction norm is not enough to decide whether G3 is
        # functionally idle: the same residual can either be irrelevant on a
        # well-separated parent or flip an ambiguous assignment. Record the
        # parent margin and the realized discrete change on the exact physical
        # support used by the binder. These remain audit-only and introduce
        # no assignment pressure.
        if self.objects > 1:
            parent_top2 = parent_k_conditional.detach().float().topk(
                k=2, dim=-1
            ).values
            corrected_top2 = corrected_k_conditional.detach().float().topk(
                k=2, dim=-1
            ).values
            parent_margin = parent_top2[..., 0] - parent_top2[..., 1]
            corrected_margin = corrected_top2[..., 0] - corrected_top2[..., 1]
            assignment_changed = (
                parent_k_conditional.detach().argmax(dim=-1)
                != corrected_k_conditional.detach().argmax(dim=-1)
            ).float()
        else:
            parent_margin = parent_k_conditional.detach().float()[..., 0]
            corrected_margin = corrected_k_conditional.detach().float()[..., 0]
            assignment_changed = torch.zeros_like(parent_margin)
        binder_support = (
            valid.detach().float()[..., 0]
            * prior.detach().float()[..., 0]
            * (parent_k_mass.detach().float()[..., 0] > 1.0e-6).float()
        )
        binder_support_sum = binder_support.sum().clamp_min(1.0)
        parent_margin_mean = (parent_margin * binder_support).sum() / binder_support_sum
        corrected_margin_mean = (
            corrected_margin * binder_support
        ).sum() / binder_support_sum
        assignment_change_fraction = (
            assignment_changed * binder_support
        ).sum() / binder_support_sum
        bounded_logit = raw_residual.detach().float()
        conditional_logit = bounded_logit - bounded_logit.mean(
            dim=-1,
            keepdim=True,
        )
        conditional_logit_spread = conditional_logit.square().mean(
            dim=-1,
        ).sqrt()
        conditional_logit_span = (
            bounded_logit.amax(dim=-1) - bounded_logit.amin(dim=-1)
        )
        metrics = {
            "object_grounding_reconstruction_mse": reconstruction_error.detach(),
            "object_grounding_aggregated_content_rms": (
                aggregated_content.detach().float().square().mean().sqrt()
            ),
            "object_grounding_canonical_slot_residual_rms": (
                canonical_slot_residual.detach().float().square().mean().sqrt()
            ),
            "object_grounding_canonical_content_rms": (
                content.detach().float().square().mean().sqrt()
            ),
            "object_grounding_public_position_rms": (
                public_position.detach().float().square().mean().sqrt()
            ),
            "object_grounding_dense_objective_count": reconstruction_error.new_ones(
                (), dtype=torch.float32
            ),
            "object_grounding_existence_mean": existence.detach().float().mean(),
            "object_grounding_validity_mean": object_validity.detach().float().mean(),
            "object_grounding_chart_availability_min_positive": torch.where(
                (object_validity.detach().float() > 0.0).any(),
                torch.where(
                    object_validity.detach().float() > 0.0,
                    object_validity.detach().float(),
                    torch.full_like(object_validity.detach().float(), torch.inf),
                ).amin(),
                object_validity.new_zeros((), dtype=torch.float32),
            ),
            "object_grounding_camera_chart_availability_mean": (
                camera_chart_availability.detach().float().mean()
            ),
            "object_grounding_camera_weight_min_positive": torch.where(
                (camera_weight.detach().float() > 0.0).any(),
                torch.where(
                    camera_weight.detach().float() > 0.0,
                    camera_weight.detach().float(),
                    torch.full_like(camera_weight.detach().float(), torch.inf),
                ).amin(),
                camera_weight.new_zeros((), dtype=torch.float32),
            ),
            "object_grounding_reconstruction_object_mass_mean": (
                reconstruction_object_mass.detach().float().mean()
            ),
            "object_grounding_reconstruction_active_fraction": (
                reconstruction_owner_active.detach().float().mean()
            ),
            "object_grounding_reconstruction_conditional_owner_entropy": (
                reconstruction_owner_entropy
            ),
            "object_grounding_camera_evidence_mass": camera_evidence_mass.detach()
            .float()
            .mean(),
            "object_grounding_camera_evidence_mass_std": camera_evidence_mass
            .detach()
            .float()
            .std(unbiased=False),
            "object_grounding_camera_evidence_mass_min": camera_evidence_mass
            .detach()
            .float()
            .amin(),
            "object_grounding_camera_evidence_mass_max": camera_evidence_mass
            .detach()
            .float()
            .amax(),
            "object_grounding_camera_support_width_mean": camera_support.detach()
            .float()
            .mean(),
            "object_grounding_camera_support_width_std": camera_support.detach()
            .float()
            .std(unbiased=False),
            "object_grounding_camera_support_evidence_correlation": support_evidence_correlation,
            "object_grounding_g3_null_identity_error": (
                corrected[..., self.objects]
                - parent_owner[..., self.objects]
            )
            .detach()
            .float()
            .abs()
            .amax(),
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
            "object_grounding_global_k_binder_bounded_logit_rms": bounded_logit
            .square()
            .mean()
            .sqrt(),
            "object_grounding_global_k_binder_bounded_logit_max_abs": (
                bounded_logit.abs().amax()
            ),
            "object_grounding_global_k_binder_conditional_logit_spread_rms": (
                conditional_logit_spread * binder_support
            ).sum()
            / binder_support_sum,
            "object_grounding_global_k_binder_conditional_logit_span_mean": (
                conditional_logit_span * binder_support
            ).sum()
            / binder_support_sum,
            "object_grounding_parent_k_conditional_entropy": normalized_entropy(
                parent_k_conditional,
                dim=-1,
            )
            .detach()
            .mean(),
            "object_grounding_corrected_k_conditional_entropy": normalized_entropy(
                corrected_k_conditional,
                dim=-1,
            )
            .detach()
            .mean(),
            "object_grounding_g3_parent_top2_margin": parent_margin_mean,
            "object_grounding_g3_corrected_top2_margin": corrected_margin_mean,
            "object_grounding_g3_assignment_change_fraction": assignment_change_fraction,
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
            "object_grounding_public_content_rms": facts.public_content.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_grounding_object_content_innovation_rms": content_innovation
            .square()
            .mean()
            .sqrt(),
            "object_grounding_object_content_innovation_variation": content_innovation
            .std(dim=1, unbiased=False)
            .mean(),
            "object_grounding_object_innovation_pair_cosine": self._pair_cosine(
                facts.content_innovation
            ),
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
            "object_grounding_candidate_key_rms": torch.cat(
                (candidates[:, :, None], candidate_views), dim=2
            ).detach()
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
