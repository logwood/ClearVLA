"""P1: one current-only high-resolution factual read.

P1 is built once per observation and cached across the five ODE steps.  Its
query is formed from S temporal intent, the clean coarse proposal and the
formal eight-row history proposal; a noisy action is not accepted.  The
reader preserves global object, camera and 3x3
microgrid axes until the last conditional read and returns the sole
``ObjectFactualDock`` consumed by P2/P3.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .observation import ObservationEvidence, _sample_feature_chart
from .role_hosts import StaticP1RoleHost
from .routing import smooth_rms_contract, variance_floored_centered_norm
from .types import (
    CoarseActionIntentState,
    HistoryActionProposalState,
    ObjectFactSet,
    ObjectFactualDock,
    ObjectIntentState,
    normalized_entropy,
)


class ObjectFactualReader(nn.Module):
    """Read four typed factual glimpses under the global-K support."""

    microgrid_offset: Tensor

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
        heads: int,
        host_expansion: float = 4.0,
        host_dropout: float = 0.05,
        microgrid_side: int = 3,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.horizon = int(horizon)
        self.basis = int(basis)
        self.cameras = int(cameras)
        self.microgrid_side = int(microgrid_side)
        if self.microgrid_side != 3:
            raise ValueError("the active P1 reader requires a 3x3 microgrid")
        self.temporal_query = nn.Linear(hidden, hidden, bias=False)
        self.coarse_query = nn.Linear(hidden, hidden, bias=False)
        self.history_proposal_query = nn.Linear(hidden, hidden, bias=False)
        self.object_query = nn.Linear(hidden, hidden, bias=False)
        self.object_key = nn.Linear(hidden, hidden, bias=False)
        self.content_key = nn.Linear(content_dim, hidden, bias=False)
        self.semantic_key = nn.Linear(route_dim, hidden, bias=False)
        self.appearance_key = nn.Linear(route_dim, hidden, bias=False)
        self.detail_key = nn.Linear(raw_dim, hidden, bias=False)
        self.detail_value = nn.Linear(raw_dim, hidden, bias=False)
        self.rgb_value = nn.Linear(3, hidden, bias=False)
        self.object_value = nn.Linear(content_dim, hidden, bias=False)
        self.position_key = nn.Linear(2, hidden, bias=False)
        self.basis_identity = nn.Parameter(torch.randn(1, 1, basis, hidden) * 0.02)
        self.camera_identity = nn.Parameter(torch.randn(1, 1, cameras, 1, hidden) * 0.02)
        self.null_key = nn.Parameter(torch.zeros(1, 1, 1, hidden))
        self.role_host = StaticP1RoleHost(
            hidden=hidden,
            heads=heads,
            expansion=host_expansion,
            dropout=host_dropout,
        )
        offsets = torch.linspace(-1.0, 1.0, self.microgrid_side)
        yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")
        self.register_buffer(
            "microgrid_offset",
            torch.stack((xx, yy), dim=-1).reshape(-1, 2),
            persistent=True,
        )

    def _queries(
        self,
        intent: ObjectIntentState,
        coarse: CoarseActionIntentState,
        history_proposal: HistoryActionProposalState,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        batch = int(intent.temporal_queries.shape[0])
        if tuple(intent.temporal_queries.shape[1:]) != (self.horizon, self.hidden):
            raise ValueError("P1 intent temporal queries do not match the action horizon")
        if tuple(coarse.innovations.shape) != (batch, 4, self.hidden):
            raise ValueError("P1 coarse proposal must retain four intervals")
        if tuple(history_proposal.tokens.shape) != (
            batch,
            self.horizon,
            self.hidden,
        ):
            raise ValueError("P1 history proposal must retain the 24-step clean path")
        interval_index = torch.div(
            torch.arange(self.horizon, device=intent.temporal_queries.device) * 4,
            self.horizon,
            rounding_mode="floor",
        ).clamp_max(3)
        coarse_time = coarse.innovations[:, interval_index]
        query = (
            self.temporal_query(intent.temporal_queries)
            + self.coarse_query(coarse_time)
            + self.history_proposal_query(history_proposal.tokens)
        ) / math.sqrt(3.0)
        query = query[:, :, None] + self.basis_identity.to(
            device=query.device,
            dtype=query.dtype,
        )
        return self.role_host(query)

    def _microgrid(
        self,
        evidence: ObservationEvidence,
        facts: ObjectFactSet,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch, objects, cameras = facts.camera_coordinates.shape[:3]
        if cameras != self.cameras:
            raise ValueError("P1 object facts use an unexpected camera count")
        support = facts.camera_support.float().clamp(0.01, 0.35)
        offsets = self.microgrid_offset.to(device=support.device)[None, None, None]
        coordinates = facts.camera_coordinates.float()[..., None, :]
        coordinates = coordinates + (1.0 - coordinates.square()).clamp_min(0.0) * (
            offsets * support[..., None, :]
        )
        # The generic sampler owns B,C,Y,X,M.  Reinterpret K as Y and a
        # singleton X, then restore the object/camera layout.
        sampler_coordinates = coordinates.permute(0, 2, 1, 3, 4)[:, :, :, None]
        detail, detail_valid = _sample_feature_chart(
            evidence.detail_features,
            sampler_coordinates,
        )
        rgb, rgb_valid = _sample_feature_chart(
            evidence.literal_rgb,
            sampler_coordinates,
        )
        # [B,C,K,1,N,D] -> [B,K,C,N,D]
        detail = detail[:, :, :, 0].permute(0, 2, 1, 3, 4).contiguous()
        rgb = rgb[:, :, :, 0].permute(0, 2, 1, 3, 4).contiguous()
        valid = (detail_valid & rgb_valid)[:, :, :, 0].permute(0, 2, 1, 3, 4)
        valid = valid & (facts.camera_validity[..., None, :] > 0)
        return detail, rgb, coordinates, valid

    def forward(
        self,
        *,
        evidence: ObservationEvidence,
        facts: ObjectFactSet,
        intent: ObjectIntentState,
        coarse_action: CoarseActionIntentState,
        history_proposal: HistoryActionProposalState,
        collect_diagnostics: bool = False,
    ) -> tuple[ObjectFactualDock, dict[str, Tensor]]:
        evidence.validate()
        facts.validate()
        queries, host_metrics = self._queries(
            intent,
            coarse_action,
            history_proposal,
        )
        batch, horizon, basis = queries.shape[:3]
        objects = facts.objects
        detail, rgb, coordinates, valid = self._microgrid(evidence, facts)
        samples = int(detail.shape[3])

        detail_key = self.detail_key(detail)
        coordinate_key = self.position_key(coordinates)
        key = (
            detail_key
            + coordinate_key
            + self.camera_identity.to(device=detail.device, dtype=detail.dtype)
        )
        object_context = (
            self.content_key(facts.content)
            + self.semantic_key(facts.semantic)
            + self.appearance_key(facts.appearance)
        ) / math.sqrt(3.0)
        fine_query, _ = variance_floored_centered_norm(
            self.object_query(queries)[:, :, :, None, None, None]
            + object_context[:, None, None, :, None, None],
            0.25,
        )
        fine_key, _ = variance_floored_centered_norm(
            key[:, None, None],
            0.25,
        )
        fine_score = torch.einsum(
            "btqkcnh,btqkcnh->btqkcn",
            fine_query.expand(-1, -1, -1, -1, self.cameras, samples, -1),
            fine_key.expand(-1, horizon, basis, -1, -1, -1, -1),
        ) / math.sqrt(float(self.hidden))
        fine_valid = valid[..., 0][:, None, None].expand(-1, horizon, basis, -1, -1, -1)
        fine_score = fine_score.masked_fill(~fine_valid, -1.0e4)
        flat_score = fine_score.flatten(-2)
        flat_valid = fine_valid.flatten(-2)
        fine_probability = torch.softmax(flat_score, dim=-1) * flat_valid.float()
        fine_probability = fine_probability / fine_probability.sum(dim=-1, keepdim=True).clamp_min(
            1.0
        )
        detail_value = self.detail_value(detail) + self.rgb_value(rgb)
        detail_value = detail_value.reshape(batch, objects, self.cameras * samples, self.hidden)
        selected_detail = torch.einsum(
            "btqkn,bknh->btqkh",
            fine_probability.to(dtype=detail_value.dtype),
            detail_value,
        )
        object_base = self.object_value(facts.content)[:, None, None]
        fact_by_object, _ = smooth_rms_contract(object_base + selected_detail, 1.0)

        object_query, _ = variance_floored_centered_norm(queries, 0.25)
        # Centering prevents a common K carrier from pretending to be object
        # ownership.  Global support/existence stays as the factual prior.
        centered_object = object_context - object_context.mean(dim=1, keepdim=True)
        object_key, _ = variance_floored_centered_norm(self.object_key(centered_object), 0.25)
        object_score = torch.einsum("btqh,bkh->btqk", object_query, object_key)
        physical = facts.validity[..., 0].float().clamp(0.0, 1.0)
        existence = facts.existence[..., 0].float().clamp(1e-6, 1.0)
        object_prior = physical * existence
        object_logit = object_score.float() + torch.where(
            object_prior[:, None, None] > 0,
            object_prior[:, None, None].clamp_min(1e-30).log(),
            torch.full_like(object_score.float(), -1.0e4),
        )
        null_logit = (
            object_query.float()
            * self.null_key.to(device=queries.device, dtype=queries.dtype).float()
        ).sum(dim=-1, keepdim=True)
        posterior = torch.softmax(torch.cat((object_logit, null_logit), dim=-1), dim=-1)
        object_posterior = posterior[..., :objects]
        null_posterior = posterior[..., objects:]
        aggregate = torch.einsum(
            "btqk,btqkh->btqh",
            object_posterior.to(dtype=fact_by_object.dtype),
            fact_by_object,
        )
        chart_posterior = facts.object_to_chart[:, None, None].expand(
            -1, horizon, basis, -1, -1, -1, -1
        )
        camera_coordinates = facts.camera_coordinates[:, None, None].expand(
            -1, horizon, basis, -1, -1, -1
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
        return dock, {
            **host_metrics,
            "p1_object_posterior_entropy": normalized_entropy(posterior, dim=-1).detach().mean(),
            "p1_object_posterior_max": posterior.detach().amax(dim=-1).mean(),
            "p1_null_mass": null_posterior.detach().float().mean(),
            "p1_microgrid_valid_fraction": fine_valid.detach().float().mean(),
            "p1_detail_rms": selected_detail.detach().float().square().mean().sqrt(),
            "p1_fact_rms": aggregate.detach().float().square().mean().sqrt(),
        }


__all__ = ["ObjectFactualReader"]
