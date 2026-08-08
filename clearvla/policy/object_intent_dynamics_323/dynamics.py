"""W1/W2 object-level future dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..role_delta_attnres import RoleDeltaAttnRes, smooth_rms_contract
from .types import (
    CoarseActionIntentState,
    FutureObjectDynamics,
    ObjectFactSet,
    ObjectIntentState,
)


@dataclass(frozen=True)
class ObjectW1WorkingState:
    """Private W carrier; only the decoded FutureObjectDynamics may reach P."""

    near: Tensor  # [B,2,K,H]
    far_base: Tensor  # [B,2,K,H]
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
        interval_view = value.transpose(1, 2).reshape(batch * objects, intervals, hidden)
        normalized = self.interval_norm(interval_view)
        mask = (
            torch.triu(
                torch.ones(intervals, intervals, device=value.device, dtype=torch.bool),
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
        value = value + update.reshape(batch, objects, intervals, hidden).transpose(1, 2)
        ffn, _ = smooth_rms_contract(self.ffn(value), 0.35)
        return value + ffn


class _ObjectDynamicsHeads(nn.Module):
    """Small owner-specific decoder for one W depth range."""

    def __init__(self, hidden: int, content_dim: int) -> None:
        super().__init__()
        self.successor_residual = nn.Linear(hidden, content_dim, bias=False)
        self.semantic_delta = nn.Linear(hidden, content_dim, bias=False)
        self.transport = nn.Linear(hidden, 2, bias=False)
        self.covariance = nn.Linear(hidden, 3)
        self.visibility = nn.Linear(hidden, 1)
        self.persistence = nn.Linear(hidden, 1)
        self.uncertainty = nn.Linear(hidden, 1)
        nn.init.zeros_(self.successor_residual.weight)
        nn.init.zeros_(self.semantic_delta.weight)
        nn.init.zeros_(self.transport.weight)
        nn.init.zeros_(self.covariance.weight)
        nn.init.constant_(self.covariance.bias, -3.0)
        nn.init.zeros_(self.visibility.weight)
        nn.init.zeros_(self.visibility.bias)
        nn.init.zeros_(self.persistence.weight)
        nn.init.zeros_(self.persistence.bias)
        nn.init.zeros_(self.uncertainty.weight)
        nn.init.constant_(self.uncertainty.bias, -2.0)


class ObjectFutureDynamicsCompiler(nn.Module):
    """W1 predicts near intervals; W2 predicts far intervals causally."""

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
        self.intent_action = nn.Linear(hidden, hidden, bias=False)
        self.intent_state = nn.Linear(hidden, hidden, bias=False)
        self.intent_object_key = nn.Linear(hidden, hidden, bias=False)
        self.intent_object_value = nn.Linear(hidden, hidden, bias=False)
        self.coarse_action = nn.Linear(hidden, hidden, bias=False)
        self.typed_router = RoleDeltaAttnRes(
            hidden,
            max(hidden // 8, 32),
            max_sources=3,
            include_null=True,
            max_value_rms=0.35,
            normalization_floor=0.25,
        )
        self.interval_identity = nn.Parameter(torch.randn(1, 4, 1, hidden) * 0.02)
        self.decoder_identity = nn.Parameter(torch.randn(1, 4, 1, hidden) * 0.02)
        self.w1 = _ObjectIntervalBlock(hidden, heads)
        self.w2 = _ObjectIntervalBlock(hidden, heads)
        self.w2_query_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.w1_memory_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.w1_to_w2 = nn.MultiheadAttention(
            hidden,
            heads,
            bias=False,
            dropout=0.0,
            batch_first=True,
        )
        # W1 and W2 share their backbone but not their output owner.  This is
        # enough capacity to represent near/far semantics without adding more
        # world blocks or a high-gain public residual.
        self.near_heads = _ObjectDynamicsHeads(hidden, content_dim)
        self.far_heads = _ObjectDynamicsHeads(hidden, content_dim)

    def _base(
        self,
        facts: ObjectFactSet,
        intent: ObjectIntentState,
        action: CoarseActionIntentState,
        *,
        collect_diagnostics: bool,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        objects = self.object_content(facts.content)
        identity = self.interval_identity.to(
            device=objects.device, dtype=objects.dtype
        )
        current_base = objects[:, None] + identity
        interval_innovation = self.intent_action(
            intent.interval_action_queries
        )[:, :, None]
        action_innovation = self.coarse_action(action.tokens)[:, :, None]
        state_innovation = self.intent_state(
            intent.interval_state_queries
        )[:, :, None].expand_as(current_base)
        object_key_innovation = self.intent_object_key(
            intent.interval_object_keys
        )
        object_value_innovation = self.intent_object_value(
            intent.interval_object_values
        )
        base = current_base + interval_innovation + action_innovation
        typed_values = torch.stack(
            (
                state_innovation,
                object_key_innovation,
                object_value_innovation,
            ),
            dim=-2,
        )
        typed_update, typed_metrics = self.typed_router(
            base,
            typed_values,
            collect_diagnostics=collect_diagnostics,
        )
        if not collect_diagnostics:
            return base + typed_update, {}
        metrics = {
            "object_w_interval_innovation_rms": interval_innovation.detach().float().square().mean().sqrt(),
            "object_w_action_innovation_rms": action_innovation.detach().float().square().mean().sqrt(),
            "object_w_state_innovation_rms": state_innovation.detach().float().square().mean().sqrt(),
            "object_w_object_key_innovation_rms": object_key_innovation.detach().float().square().mean().sqrt(),
            "object_w_object_value_innovation_rms": object_value_innovation.detach().float().square().mean().sqrt(),
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
        global_start: int,
        heads: _ObjectDynamicsHeads,
    ) -> FutureObjectDynamics:
        intervals = int(hidden.shape[1])
        value = hidden + self.decoder_identity.to(
            device=hidden.device, dtype=hidden.dtype
        )[:, global_start : global_start + intervals]
        successor_residual = heads.successor_residual(value)
        semantic_delta = heads.semantic_delta(value)
        transport = 0.50 * torch.tanh(heads.transport(value).float()).to(dtype=value.dtype)
        covariance = F.softplus(heads.covariance(value).float()).to(dtype=value.dtype)
        # Zero-centred changes relative to the currently observed object.  The
        # map is exactly zero at initialization but keeps ordinary gradients
        # in both directions.
        visibility = (
            1.0 - 2.0 * torch.sigmoid(heads.visibility(value).float())
        ).to(dtype=value.dtype)
        persistence = (
            1.0 - 2.0 * torch.sigmoid(heads.persistence(value).float())
        ).to(dtype=value.dtype)
        uncertainty = F.softplus(heads.uncertainty(value).float()).to(dtype=value.dtype)
        # Physical support is immutable across W.  Predicted future visibility
        # is a supervised/calibration field and cannot erase either its own
        # training target or the semantic effect path.
        validity = facts.validity[:, None].float().expand(
            -1, intervals, -1, -1
        )
        address = self._transport_address(
            facts.object_to_chart,
            facts.coordinates,
            transport,
        )
        return FutureObjectDynamics(
            current_reference=facts.content,
            # The current fact is a protected reference, not a trainable
            # additive escape hatch for the future error.  W/G still receive
            # ordinary gradients through the delta-producing input path.
            successor_content=(
                facts.content.detach()[:, None] + successor_residual
            ),
            semantic_delta=semantic_delta,
            transport_mean=transport,
            transport_covariance=covariance,
            visibility=visibility,
            persistence=persistence,
            uncertainty=uncertainty,
            validity=validity,
            future_address=address,
            object_coordinates=facts.coordinates,
        )

    @staticmethod
    def _transport_address(
        current: Tensor,
        coordinates: Tensor,
        transport: Tensor,
    ) -> Tensor:
        batch, objects, cameras, rows, columns = current.shape
        intervals = int(transport.shape[1])
        axis_y = torch.linspace(-1.0, 1.0, rows, device=current.device)
        axis_x = torch.linspace(-1.0, 1.0, columns, device=current.device)
        coordinate_y, coordinate_x = torch.meshgrid(axis_y, axis_x, indexing="ij")
        grid = torch.stack((coordinate_x, coordinate_y), dim=-1)
        center = (coordinates[:, None] + transport.float()).clamp(-1.0, 1.0)
        distance = (
            grid.reshape(1, 1, 1, 1, rows, columns, 2)
            - center[:, :, :, None, None, None]
        ).square().sum(dim=-1)
        logits = (
            current.float().clamp_min(1e-6).log()[:, None]
            - 4.0 * distance
        )
        return torch.softmax(logits.flatten(3), dim=-1).reshape(
            batch, intervals, objects, cameras, rows, columns
        ).to(dtype=current.dtype)

    def forward_w1(
        self,
        *,
        facts: ObjectFactSet,
        intent: ObjectIntentState,
        action: CoarseActionIntentState,
        collect_diagnostics: bool = True,
    ) -> tuple[FutureObjectDynamics, ObjectW1WorkingState, dict[str, Tensor]]:
        hidden, base_metrics = self._base(
            facts,
            intent,
            action,
            collect_diagnostics=collect_diagnostics,
        )
        near = self.w1(hidden[:, :2], causal_interval=True)
        field = self._field(
            facts=facts,
            hidden=near,
            global_start=0,
            heads=self.near_heads,
        )
        field.validate(expected_intervals=2)
        metrics = (
            self._metrics(field, prefix="object_w1")
            if collect_diagnostics
            else {}
        )
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
        collect_diagnostics: bool = True,
    ) -> tuple[FutureObjectDynamics, dict[str, Tensor]]:
        if tuple(w1_state.near.shape[1:3]) != (2, facts.objects) or tuple(
            w1_state.far_base.shape[1:3]
        ) != (2, facts.objects):
            raise ValueError("W2 requires the two completed W1 intervals")
        # Both near intervals precede both far intervals, so W2 may read the
        # complete W1 sequence without a future leak.  Do not mean-pool W1:
        # that would erase the 4-8 versus 8-16 distinction immediately before
        # the long-horizon compiler is expected to use it.
        batch, _, objects, hidden = w1_state.near.shape
        far_query = w1_state.far_base.transpose(1, 2).reshape(
            batch * objects, 2, hidden
        )
        near_memory = w1_state.near.transpose(1, 2).reshape(
            batch * objects, 2, hidden
        )
        near_update, _ = self.w1_to_w2(
            self.w2_query_norm(far_query),
            self.w1_memory_norm(near_memory),
            self.w1_memory_norm(near_memory),
            need_weights=False,
        )
        near_update, _ = smooth_rms_contract(near_update, 0.35)
        far = (
            far_query + near_update
        ).reshape(batch, objects, 2, hidden).transpose(1, 2)
        far = self.w2(far, causal_interval=True)
        far_field = self._field(
            facts=facts,
            hidden=far,
            global_start=2,
            heads=self.far_heads,
        )
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
                (
                    near_field.transport_covariance,
                    far_field.transport_covariance,
                ),
                dim=1,
            ),
            visibility=torch.cat(
                (near_field.visibility, far_field.visibility), dim=1
            ),
            persistence=torch.cat(
                (near_field.persistence, far_field.persistence), dim=1
            ),
            uncertainty=torch.cat(
                (near_field.uncertainty, far_field.uncertainty), dim=1
            ),
            validity=torch.cat((near_field.validity, far_field.validity), dim=1),
            future_address=torch.cat(
                (near_field.future_address, far_field.future_address), dim=1
            ),
            object_coordinates=near_field.object_coordinates,
        )
        field.validate()
        metrics = (
            self._metrics(field, prefix="object_w2")
            if collect_diagnostics
            else {}
        )
        return field, metrics

    @staticmethod
    def _metrics(field: FutureObjectDynamics, *, prefix: str) -> dict[str, Tensor]:
        delta = field.semantic_delta.detach().float()
        adjacent = (
            F.cosine_similarity(delta[:, 1:].flatten(2), delta[:, :-1].flatten(2), dim=-1).mean()
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
            f"{prefix}_validity": field.validity.detach().float().mean(),
        }
