"""Recovered V120 W1/W2 four-interval object dynamics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .routing import RoleDeltaAttnRes, smooth_rms_contract
from .types import (
    CoarseActionIntentState,
    FutureObjectDynamics,
    ObjectFactSet,
    ObjectIntentState,
)


@dataclass(frozen=True)
class ObjectW1WorkingState:
    """Private W carrier; only decoded FutureObjectDynamics reaches P."""

    near: Tensor
    far_base: Tensor
    near_field: FutureObjectDynamics


class _ObjectIntervalBlock(nn.Module):
    def __init__(self, hidden: int, heads: int) -> None:
        super().__init__()
        self.object_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.object_attention = nn.MultiheadAttention(
            hidden, heads, bias=False, dropout=0.0, batch_first=True
        )
        self.interval_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.interval_attention = nn.MultiheadAttention(
            hidden, heads, bias=False, dropout=0.0, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden, elementwise_affine=False),
            nn.Linear(hidden, 2 * hidden, bias=False),
            nn.SiLU(),
            nn.Linear(2 * hidden, hidden, bias=False),
        )

    def forward(self, value: Tensor, *, causal_interval: bool) -> Tensor:
        batch, intervals, objects, hidden = value.shape
        object_view = value.reshape(batch * intervals, objects, hidden)
        normalized = self.object_norm(object_view)
        update, _ = self.object_attention(
            normalized, normalized, normalized, need_weights=False
        )
        update, _ = smooth_rms_contract(update, 0.35)
        value = value + update.reshape_as(value)
        interval_view = value.transpose(1, 2).reshape(
            batch * objects, intervals, hidden
        )
        normalized = self.interval_norm(interval_view)
        mask = (
            torch.triu(
                torch.ones(
                    intervals, intervals, device=value.device, dtype=torch.bool
                ),
                diagonal=1,
            )
            if causal_interval
            else None
        )
        update, _ = self.interval_attention(
            normalized,
            normalized,
            normalized,
            attn_mask=mask,
            need_weights=False,
        )
        update, _ = smooth_rms_contract(update, 0.35)
        value = value + update.reshape(
            batch, objects, intervals, hidden
        ).transpose(1, 2)
        ffn, _ = smooth_rms_contract(self.ffn(value), 0.35)
        return value + ffn


class ObjectFutureDynamicsCompiler(nn.Module):
    """W1 predicts 4-8/8-16; W2 predicts 16-32/32-48."""

    def __init__(
        self,
        *,
        hidden: int,
        content_dim: int,
        route_dim: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.content_dim = int(content_dim)
        self.object_content = nn.Linear(content_dim, hidden, bias=False)
        self.object_semantic = nn.Linear(route_dim, hidden, bias=False)
        self.object_appearance = nn.Linear(route_dim, hidden, bias=False)
        self.object_geometry = nn.Linear(route_dim, hidden, bias=False)
        self.object_transport_prior = nn.Linear(2, hidden, bias=False)
        self.typed_router = RoleDeltaAttnRes(
            hidden,
            max(hidden // 8, 32),
            max_sources=3,
            include_null=True,
            max_value_rms=0.35,
            normalization_floor=0.25,
        )
        self.goal_query_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.goal_memory_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.goal_read = nn.MultiheadAttention(
            hidden, heads, bias=False, dropout=0.0, batch_first=True
        )
        self.interval_identity = nn.Parameter(torch.randn(1, 4, 1, hidden) * 0.02)
        self.w1 = _ObjectIntervalBlock(hidden, heads)
        self.w2 = _ObjectIntervalBlock(hidden, heads)
        self.w2_query_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.w1_memory_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.w1_to_w2 = nn.MultiheadAttention(
            hidden, heads, bias=False, dropout=0.0, batch_first=True
        )
        self.delta_head = nn.Linear(hidden, content_dim, bias=False)
        self.transport_head = nn.Linear(hidden, 2, bias=False)
        self.covariance_head = nn.Linear(hidden, 3)
        self.visibility_head = nn.Linear(hidden, 1)
        self.persistence_head = nn.Linear(hidden, 1)
        self.uncertainty_head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.transport_head.weight)
        nn.init.zeros_(self.covariance_head.weight)
        nn.init.constant_(self.covariance_head.bias, -3.0)
        nn.init.zeros_(self.visibility_head.weight)
        nn.init.zeros_(self.visibility_head.bias)
        nn.init.zeros_(self.persistence_head.weight)
        nn.init.zeros_(self.persistence_head.bias)
        nn.init.zeros_(self.uncertainty_head.weight)
        nn.init.constant_(self.uncertainty_head.bias, -2.0)

    def _base(
        self,
        facts: ObjectFactSet,
        intent: ObjectIntentState,
        action: CoarseActionIntentState,
        *,
        collect_diagnostics: bool,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        objects = self.object_content(facts.content)
        semantic = self.object_semantic(facts.semantic)
        appearance = self.object_appearance(facts.appearance)
        geometry = self.object_geometry(facts.geometry) + self.object_transport_prior(
            facts.transport_prior.to(dtype=facts.geometry.dtype)
        )
        interval = (
            intent.interval_queries
            + action.tokens
            + self.interval_identity.to(
                device=objects.device, dtype=objects.dtype
            )[:, :, 0]
        )
        normalized_goal = self.goal_memory_norm(intent.protected_goal_set)
        goal_update, goal_attention = self.goal_read(
            self.goal_query_norm(interval),
            normalized_goal,
            normalized_goal,
            need_weights=collect_diagnostics,
            average_attn_weights=True,
        )
        base = objects[:, None] + interval[:, :, None] + goal_update[:, :, None]
        typed_values = torch.stack((semantic, appearance, geometry), dim=-2)
        typed_values = typed_values[:, None].expand(-1, 4, -1, -1, -1)
        typed_update, typed_metrics = self.typed_router(
            base,
            typed_values,
            collect_diagnostics=collect_diagnostics,
        )
        if not collect_diagnostics:
            return base + typed_update, {}
        if goal_attention is None:
            goal_attention = interval.new_zeros(
                interval.shape[0], interval.shape[1], intent.protected_goal_set.shape[1]
            )
        metrics = {
            "object_w_goal_attention_entropy": (
                -(
                    goal_attention.detach().float().clamp_min(1e-8)
                    * goal_attention.detach().float().clamp_min(1e-8).log()
                ).sum(dim=-1)
                / math.log(float(max(int(goal_attention.shape[-1]), 2)))
            ).mean(),
            "object_w_goal_innovation_rms": goal_update.detach().float().square().mean().sqrt(),
            "object_w_typed_innovation_rms": typed_update.detach().float().square().mean().sqrt(),
        }
        for key, value in typed_metrics.items():
            metrics[f"object_w_typed_{key}"] = value
        return base + typed_update, metrics

    def _field(
        self,
        *,
        facts: ObjectFactSet,
        hidden: Tensor,
    ) -> FutureObjectDynamics:
        semantic_delta = self.delta_head(hidden)
        object_transport = 0.50 * torch.tanh(
            self.transport_head(hidden).float()
        ).to(dtype=hidden.dtype)
        object_covariance = F.softplus(
            self.covariance_head(hidden).float()
        ).to(dtype=hidden.dtype)
        visibility = (
            1.0 - 2.0 * torch.sigmoid(self.visibility_head(hidden).float())
        ).to(dtype=hidden.dtype)
        persistence = (
            1.0 - 2.0 * torch.sigmoid(self.persistence_head(hidden).float())
        ).to(dtype=hidden.dtype)
        uncertainty = F.softplus(
            self.uncertainty_head(hidden).float()
        ).to(dtype=hidden.dtype)
        reliability = torch.zeros_like(uncertainty)
        visibility_probability = (1.0 + visibility.float()).clamp(0.0, 1.0)
        future_selector_validity = facts.validity[:, None].float() * visibility_probability
        address = self._transport_address(facts.object_to_chart, object_transport)
        current_reference = facts.content.detach()
        return FutureObjectDynamics(
            current_reference=current_reference,
            successor_content=current_reference[:, None] + semantic_delta,
            semantic_delta=semantic_delta,
            transport_mean=object_transport,
            transport_covariance=object_covariance,
            visibility=visibility,
            persistence=persistence,
            uncertainty=uncertainty,
            reliability=reliability,
            future_selector_validity=future_selector_validity,
            future_address=address,
            object_coordinates=facts.coordinates.to(dtype=hidden.dtype),
        )

    @staticmethod
    def _transport_address(current: Tensor, transport: Tensor) -> Tensor:
        """Differentiably translate the complete current object chart."""

        batch, objects, cameras, rows, columns = current.shape
        intervals = int(transport.shape[1])
        axis_y = torch.linspace(
            -1.0, 1.0, rows, device=current.device, dtype=torch.float32
        )
        axis_x = torch.linspace(
            -1.0, 1.0, columns, device=current.device, dtype=torch.float32
        )
        coordinate_y, coordinate_x = torch.meshgrid(axis_y, axis_x, indexing="ij")
        base_grid = torch.stack((coordinate_x, coordinate_y), dim=-1)
        # This diagnostic applies the one physical object displacement to
        # each *existing* camera chart.  Let the camera axis come from the
        # current object address itself; do not manufacture a camera-valued W
        # prediction by expanding transport.
        camera_grid = (
            base_grid.reshape(1, 1, 1, 1, rows, columns, 2)
            + torch.zeros_like(current.float()[:, None, :, :, :, :, None])
        )
        sampling_grid = (
            camera_grid
            - transport.float()[:, :, :, None, None, None, :]
        )
        source = current.float()[:, None].expand(
            -1, intervals, -1, -1, -1, -1
        )
        sampled = F.grid_sample(
            source.reshape(batch * intervals * objects * cameras, 1, rows, columns),
            sampling_grid.reshape(
                batch * intervals * objects * cameras, rows, columns, 2
            ),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).reshape(batch, intervals, objects, cameras, rows, columns)
        source_mass = source.sum(dim=(-2, -1), keepdim=True)
        sampled_mass = sampled.sum(dim=(-2, -1), keepdim=True)
        sampled = sampled * source_mass / sampled_mass.clamp_min(1e-8)
        sampled = torch.where(sampled_mass > 1e-8, sampled, source)
        return sampled.to(dtype=current.dtype)

    def forward_w1(
        self,
        *,
        facts: ObjectFactSet,
        intent: ObjectIntentState,
        action: CoarseActionIntentState,
        collect_diagnostics: bool = False,
    ) -> tuple[FutureObjectDynamics, ObjectW1WorkingState, dict[str, Tensor]]:
        hidden, base_metrics = self._base(
            facts, intent, action, collect_diagnostics=collect_diagnostics
        )
        near = self.w1(hidden[:, :2], causal_interval=True)
        field = self._field(facts=facts, hidden=near)
        field.validate(expected_intervals=2)
        metrics = self._metrics(field, prefix="object_w1") if collect_diagnostics else {}
        metrics.update(base_metrics)
        return field, ObjectW1WorkingState(
            near=near,
            far_base=hidden[:, 2:],
            near_field=field,
        ), metrics

    def forward_w2(
        self,
        *,
        facts: ObjectFactSet,
        intent: ObjectIntentState,
        action: CoarseActionIntentState,
        w1_state: ObjectW1WorkingState,
        collect_diagnostics: bool = False,
    ) -> tuple[FutureObjectDynamics, dict[str, Tensor]]:
        del intent, action
        if tuple(w1_state.near.shape[1:3]) != (2, facts.objects) or tuple(
            w1_state.far_base.shape[1:3]
        ) != (2, facts.objects):
            raise ValueError("W2 requires both completed W1 intervals")
        batch, _, objects, hidden = w1_state.near.shape
        far_query = w1_state.far_base.transpose(1, 2).reshape(
            batch * objects, 2, hidden
        )
        near_memory = w1_state.near.transpose(1, 2).reshape(
            batch * objects, 2, hidden
        )
        normalized_near = self.w1_memory_norm(near_memory)
        near_update, _ = self.w1_to_w2(
            self.w2_query_norm(far_query),
            normalized_near,
            normalized_near,
            need_weights=False,
        )
        near_update, _ = smooth_rms_contract(near_update, 0.35)
        far = (far_query + near_update).reshape(
            batch, objects, 2, hidden
        ).transpose(1, 2)
        far = self.w2(far, causal_interval=True)
        far_field = self._field(facts=facts, hidden=far)
        near_field = w1_state.near_field
        near_field.validate(expected_intervals=2)
        field = FutureObjectDynamics(
            current_reference=near_field.current_reference,
            successor_content=torch.cat(
                (near_field.successor_content, far_field.successor_content), dim=1
            ),
            semantic_delta=torch.cat(
                (near_field.semantic_delta, far_field.semantic_delta), dim=1
            ),
            transport_mean=torch.cat(
                (near_field.transport_mean, far_field.transport_mean), dim=1
            ),
            transport_covariance=torch.cat(
                (near_field.transport_covariance, far_field.transport_covariance), dim=1
            ),
            visibility=torch.cat((near_field.visibility, far_field.visibility), dim=1),
            persistence=torch.cat((near_field.persistence, far_field.persistence), dim=1),
            uncertainty=torch.cat((near_field.uncertainty, far_field.uncertainty), dim=1),
            reliability=torch.cat((near_field.reliability, far_field.reliability), dim=1),
            future_selector_validity=torch.cat(
                (
                    near_field.future_selector_validity,
                    far_field.future_selector_validity,
                ),
                dim=1,
            ),
            future_address=torch.cat(
                (near_field.future_address, far_field.future_address), dim=1
            ),
            object_coordinates=near_field.object_coordinates,
        )
        field.validate()
        metrics = self._metrics(field, prefix="object_w2") if collect_diagnostics else {}
        return field, metrics

    @staticmethod
    def _metrics(field: FutureObjectDynamics, *, prefix: str) -> dict[str, Tensor]:
        delta = field.semantic_delta.detach().float()
        adjacent = (
            F.cosine_similarity(
                delta[:, 1:].flatten(2),
                delta[:, :-1].flatten(2),
                dim=-1,
                eps=1e-4,
            ).mean()
            if field.intervals > 1
            else delta.new_zeros(())
        )
        object_similarity = F.normalize(delta, dim=-1, eps=1e-4)
        object_similarity = torch.einsum(
            "bikd,bijd->bikj", object_similarity, object_similarity
        )
        objects = int(delta.shape[2])
        mask = ~torch.eye(objects, device=delta.device, dtype=torch.bool)
        pair = (
            object_similarity.masked_select(mask[None, None]).mean()
            if objects > 1
            else delta.new_zeros(())
        )
        return {
            f"{prefix}_semantic_delta_rms": delta.square().mean().sqrt(),
            f"{prefix}_interval_adjacent_cosine": adjacent,
            f"{prefix}_object_pair_cosine": pair,
            f"{prefix}_transport_rms": field.transport_mean.detach().float().square().mean().sqrt(),
            f"{prefix}_future_selector_validity": field.future_selector_validity
            .detach()
            .float()
            .mean(),
        }


__all__ = ["ObjectFutureDynamicsCompiler", "ObjectW1WorkingState"]
