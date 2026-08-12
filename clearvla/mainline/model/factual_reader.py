"""P1: one V120-compatible query-specific high-resolution factual read."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .observation_contract import ObservationEvidence
from .routing import smooth_rms_contract, variance_floored_centered_norm
from .types import ObjectFactSet, ObjectFactualDock, ObjectIntentState, normalized_entropy


class ObjectFactualReader(nn.Module):
    """Select the full N=49 progressive bank before contracting its values.

    The former independent reader first averaged those candidates to one
    coordinate and then sampled a new 3x3 grid.  That made the action query
    unable to choose among the real high-resolution alternatives.  This
    implementation keeps C/Y/X/M/N until each clean action-basis query has
    produced its posterior, and checkpoints chunks to keep batch-eight memory
    bounded.
    """

    def __init__(
        self,
        *,
        hidden: int,
        content_dim: int,
        raw_dim: int,
        route_dim: int,
        horizon: int,
        basis: int,
        cameras: int,
        microgrid_side: int = 3,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.route_dim = int(route_dim)
        self.horizon = int(horizon)
        self.basis = int(basis)
        self.cameras = int(cameras)
        if int(microgrid_side) != 3:
            raise ValueError("the active P1 contract requires a 3x3 value basis")
        self.temporal_query = nn.Linear(hidden, hidden, bias=False)
        self.goal_query = nn.Linear(hidden, hidden, bias=False)
        self.history_query = nn.Linear(hidden, hidden, bias=False)
        self.object_query = nn.Linear(hidden, hidden, bias=False)
        self.object_key = nn.Linear(hidden, hidden, bias=False)
        self.content_key = nn.Linear(content_dim, hidden, bias=False)
        self.semantic_key = nn.Linear(route_dim, hidden, bias=False)
        self.appearance_key = nn.Linear(route_dim, hidden, bias=False)
        self.semantic_fine_query = nn.Linear(hidden, route_dim, bias=False)
        self.appearance_fine_query = nn.Linear(hidden, route_dim, bias=False)
        self.geometry_fine_query = nn.Linear(hidden, route_dim, bias=False)
        self.detail_value = nn.Linear(raw_dim, hidden, bias=False)
        self.rgb_value = nn.Linear(3, hidden, bias=False)
        self.object_value = nn.Linear(content_dim, hidden, bias=False)
        self.fact_key = nn.Linear(hidden, hidden, bias=False)

    def _queries(
        self,
        intent: ObjectIntentState,
        clean_action_basis: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        batch = int(intent.temporal_queries.shape[0])
        expected = (batch, self.horizon, self.basis, self.hidden)
        if tuple(clean_action_basis.shape) != expected:
            raise ValueError("P1 requires V120 clean action-basis tokens")
        temporal = self.temporal_query(intent.temporal_queries)[:, :, None]
        goal = self.goal_query(intent.protected_goal_set.mean(dim=1))[:, None, None]
        history = self.history_query(intent.history_tokens[:, -1])[:, None, None]
        query = (
            clean_action_basis + temporal + goal + history
        ) / math.sqrt(4.0)
        # This is the ODE-invariant factual selection query.  V120's active
        # P1 policy block is an ODE-dependent trajectory write and therefore
        # does not belong here.  It is executed after this protected detail
        # has been cached, immediately before P2.
        return query, {}

    def _candidate_chunk(
        self,
        local_query: Tensor,
        semantic_candidates: Tensor,
        appearance_candidates: Tensor,
        geometry_candidates: Tensor,
        detail_candidates: Tensor,
        rgb_candidates: Tensor,
        candidate_coordinates: Tensor,
        candidate_valid: Tensor,
        candidate_prior: Tensor,
        local_content: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Contract one T*Q chunk without materializing hidden candidate values."""

        semantic_query, _ = variance_floored_centered_norm(
            self.semantic_fine_query(local_query), 0.25
        )
        appearance_query, _ = variance_floored_centered_norm(
            self.appearance_fine_query(local_query), 0.25
        )
        geometry_query, _ = variance_floored_centered_norm(
            self.geometry_fine_query(local_query), 0.25
        )
        semantic_key, _ = variance_floored_centered_norm(semantic_candidates, 0.25)
        appearance_key, _ = variance_floored_centered_norm(appearance_candidates, 0.25)
        geometry_key, _ = variance_floored_centered_norm(geometry_candidates, 0.25)
        semantic_score = torch.einsum(
            "brkh,bcyxmnh->brkcyxmn", semantic_query.float(), semantic_key.float()
        )
        appearance_score = torch.einsum(
            "brkh,bcyxmnh->brkcyxmn", appearance_query.float(), appearance_key.float()
        )
        geometry_score = torch.einsum(
            "brkh,bcyxmnh->brkcyxmn", geometry_query.float(), geometry_key.float()
        )
        score = (semantic_score + appearance_score + geometry_score) / math.sqrt(
            3.0 * float(self.route_dim)
        )
        prior = candidate_prior.float().clamp_min(0.0)
        valid = candidate_valid[:, None, None] & (prior[:, None, :, ..., None] > 0)
        score = score + prior[:, None, :, ..., None].clamp_min(1e-30).log()
        score = score.masked_fill(~valid, -1.0e4)
        flat_score = score.flatten(3)
        flat_valid = valid.flatten(3)
        probability = torch.softmax(flat_score, dim=-1) * flat_valid.float()
        probability = probability / probability.sum(dim=-1, keepdim=True).clamp_min(1.0)
        probability = probability.reshape_as(score)
        chart_posterior = probability.sum(dim=(-1, -2))
        camera_mass = probability.sum(dim=(-1, -2, -3, -4))
        camera_coordinates = torch.einsum(
            "brkcyxmn,bcyxmnd->brkcd",
            probability,
            candidate_coordinates.float(),
        ) / camera_mass[..., None].clamp_min(1e-6)
        selected_detail = torch.einsum(
            "brkcyxmn,bcyxmnf->brkf",
            probability.to(dtype=detail_candidates.dtype),
            detail_candidates,
        )
        selected_rgb = torch.einsum(
            "brkcyxmn,bcyxmnd->brkd",
            probability.to(dtype=rgb_candidates.dtype),
            rgb_candidates,
        )
        coarse_probability = probability.sum(dim=-1)
        selected_content = torch.einsum(
            "brkcyxm,bcyxmd->brkd",
            coarse_probability.to(dtype=local_content.dtype),
            local_content,
        )
        entropy = normalized_entropy(probability.flatten(3), dim=-1)
        maximum = probability.flatten(3).amax(dim=-1)
        return (
            selected_content,
            selected_detail,
            selected_rgb,
            chart_posterior,
            camera_coordinates,
            entropy,
            maximum,
        )

    def forward(
        self,
        *,
        evidence: ObservationEvidence,
        facts: ObjectFactSet,
        intent: ObjectIntentState,
        clean_action_basis: Tensor,
        collect_diagnostics: bool = False,
    ) -> tuple[ObjectFactualDock, dict[str, Tensor]]:
        evidence.validate()
        facts.validate()
        queries, host_metrics = self._queries(intent, clean_action_basis)
        batch, horizon, basis = queries.shape[:3]
        objects = facts.objects
        local = evidence.local_facts
        fine = evidence.progressive_candidates
        assert fine.semantic_keys is not None
        assert fine.appearance_keys is not None
        assert fine.geometry_keys is not None
        assert fine.literal_rgb is not None
        object_context = (
            self.content_key(facts.content)
            + self.semantic_key(facts.semantic)
            + self.appearance_key(facts.appearance)
        ) / math.sqrt(3.0)
        local_query, _ = variance_floored_centered_norm(
            self.object_query(queries)[:, :, :, None]
            + self.object_key(object_context)[:, None, None],
            0.25,
        )
        query_rows = horizon * basis
        flattened_query = local_query.reshape(batch, query_rows, objects, self.hidden)
        # At batch eight this keeps each FP32 posterior below roughly 110 MB;
        # checkpointing prevents all 96 query rows from being retained for
        # backward at once.
        chunk = max(1, min(query_rows, 64 // max(batch, 1)))
        content_rows: list[Tensor] = []
        detail_rows: list[Tensor] = []
        rgb_rows: list[Tensor] = []
        chart_rows: list[Tensor] = []
        coordinate_rows: list[Tensor] = []
        entropy_rows: list[Tensor] = []
        maximum_rows: list[Tensor] = []
        candidate_valid = fine.valid.to(dtype=torch.bool)
        candidate_prior = facts.candidate_assignment
        use_checkpoint = bool(self.training and torch.is_grad_enabled())
        for start in range(0, query_rows, chunk):
            stop = min(start + chunk, query_rows)
            arguments = (
                flattened_query[:, start:stop],
                fine.semantic_keys,
                fine.appearance_keys,
                fine.geometry_keys,
                fine.learned_detail,
                fine.literal_rgb,
                fine.current_coordinates,
                candidate_valid,
                candidate_prior,
                local.content_slots,
            )
            if use_checkpoint:
                result = checkpoint(
                    self._candidate_chunk,
                    *arguments,
                    use_reentrant=False,
                )
            else:
                result = self._candidate_chunk(*arguments)
            typed_result = cast(
                tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
                result,
            )
            content_rows.append(typed_result[0])
            detail_rows.append(typed_result[1])
            rgb_rows.append(typed_result[2])
            chart_rows.append(typed_result[3])
            coordinate_rows.append(typed_result[4])
            entropy_rows.append(typed_result[5])
            maximum_rows.append(typed_result[6])
        selected_content = torch.cat(content_rows, dim=1).reshape(
            batch, horizon, basis, objects, -1
        )
        selected_detail_raw = torch.cat(detail_rows, dim=1).reshape(
            batch, horizon, basis, objects, -1
        )
        selected_rgb_raw = torch.cat(rgb_rows, dim=1).reshape(
            batch, horizon, basis, objects, 3
        )
        chart_posterior = torch.cat(chart_rows, dim=1).reshape(
            batch,
            horizon,
            basis,
            objects,
            self.cameras,
            8,
            8,
        )
        camera_coordinates = torch.cat(coordinate_rows, dim=1).reshape(
            batch, horizon, basis, objects, self.cameras, 2
        )
        camera_mass = chart_posterior.sum(dim=(-2, -1))
        camera_coordinates = torch.where(
            camera_mass[..., None] > 1e-6,
            camera_coordinates,
            facts.camera_coordinates[:, None, None].float(),
        )
        selected_detail = self.detail_value(selected_detail_raw) + self.rgb_value(
            selected_rgb_raw
        )
        object_base = self.object_value(selected_content)
        # The protected value is literal/learned current detail.  Object
        # content is legal selector context, but adding a second signed value
        # branch here lets ``object_base == -selected_detail`` erase P1 while
        # every routing tensor remains non-empty.  The historical V120 reader
        # likewise used semantic/appearance/geometry to select/refine detail;
        # it did not add an independently cancellable public-object value to
        # the protected write.
        fact_by_object, _ = smooth_rms_contract(selected_detail, 1.0)

        object_query, _ = variance_floored_centered_norm(queries, 0.25)
        object_key, _ = variance_floored_centered_norm(
            self.fact_key((fact_by_object + object_base) / math.sqrt(2.0)),
            0.25,
        )
        object_score = torch.einsum(
            "btqh,btqkh->btqk", object_query, object_key
        ) / math.sqrt(float(self.hidden))
        physical = facts.validity[..., 0].float().clamp(0.0, 1.0)
        fine_available = candidate_valid.any(dim=-1)
        local_available = (
            (facts.candidate_assignment > 0.0) & fine_available[:, None]
        ).flatten(2).any(dim=-1).float()
        object_prior = physical * local_available
        object_logit = object_score.float() + torch.where(
            object_prior[:, None, None] > 0,
            object_prior[:, None, None].clamp_min(1e-30).log(),
            torch.full_like(object_score.float(), -1.0e4),
        )
        # P1 is the protected current-fact path, not an optional effect lane.
        # It must select among physically available K objects without a
        # learned null shortcut.  Null is reserved for the real no-evidence
        # case; P2 retains its own legal optional/null future-effect route.
        available = object_prior[:, None, None] > 0
        object_weight = torch.softmax(object_logit, dim=-1) * available.float()
        available_mass = object_weight.sum(dim=-1, keepdim=True)
        object_posterior = object_weight / available_mass.clamp_min(1.0e-6)
        null_posterior = (available_mass <= 0).to(dtype=object_posterior.dtype)
        aggregate = torch.einsum(
            "btqk,btqkh->btqh",
            object_posterior.to(dtype=fact_by_object.dtype),
            fact_by_object,
        )
        dock = ObjectFactualDock(
            fact_by_object=fact_by_object,
            object_posterior=object_posterior.to(dtype=queries.dtype),
            null_posterior=null_posterior.to(dtype=queries.dtype),
            chart_posterior=chart_posterior.to(dtype=queries.dtype),
            camera_coordinates=camera_coordinates.to(dtype=queries.dtype),
            aggregate_fact=aggregate,
        )
        dock.validate()
        if not collect_diagnostics:
            return dock, {}
        posterior = torch.cat((object_posterior, null_posterior), dim=-1)
        candidate_entropy = torch.cat(entropy_rows, dim=1)
        candidate_max = torch.cat(maximum_rows, dim=1)
        return dock, {
            **host_metrics,
            "p1_object_posterior_entropy": normalized_entropy(
                posterior, dim=-1
            ).detach().mean(),
            "p1_object_posterior_max": posterior.detach().amax(dim=-1).mean(),
            "p1_null_mass": null_posterior.detach().float().mean(),
            "p1_progressive_candidate_valid_fraction": candidate_valid.detach()
            .float()
            .mean(),
            "p1_progressive_candidate_entropy": candidate_entropy.detach().mean(),
            "p1_progressive_candidate_max": candidate_max.detach().mean(),
            "p1_progressive_candidate_count": aggregate.new_tensor(
                float(fine.valid.shape[-1]), dtype=torch.float32
            ),
            "p1_query_chart_variation": chart_posterior.detach()
            .float()
            .std(dim=(1, 2), unbiased=False)
            .mean(),
            "p1_query_coordinate_variation": camera_coordinates.detach()
            .float()
            .std(dim=(1, 2), unbiased=False)
            .mean(),
            "p1_local_content_rms": object_base.detach().float().square().mean().sqrt(),
            "p1_detail_rms": selected_detail.detach().float().square().mean().sqrt(),
            "p1_fact_by_object_rms": fact_by_object.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "p1_fact_rms": aggregate.detach().float().square().mean().sqrt(),
            "p1_existence_is_diagnostic_only": aggregate.new_ones(
                (), dtype=torch.float32
            ),
        }


__all__ = ["ObjectFactualReader"]
