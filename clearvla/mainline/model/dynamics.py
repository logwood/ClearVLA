"""Recovered V120 W1/W2 four-interval object dynamics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .routing import smooth_rms_contract
from .types import (
    CoarseActionIntentState,
    FutureObjectDynamics,
    ObjectFactSet,
    WorldIntentDock,
)


@dataclass(frozen=True)
class ObjectW1WorkingState:
    """Private W carrier; only decoded FutureObjectDynamics reaches P."""

    near: Tensor
    far_base: Tensor
    near_typed: Tensor
    far_typed: Tensor
    near_field: FutureObjectDynamics
    intent_boundary_field: FutureObjectDynamics


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
        self.goal_query_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.goal_memory_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.goal_read = nn.MultiheadAttention(
            hidden, heads, bias=False, dropout=0.0, batch_first=True
        )
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
        # Status changes are optional typed values.  A free bias can learn the
        # dataset-wide mean disappearance/persistence and bypass the
        # appearance sidecar entirely, so both heads are bias-free.
        self.visibility_head = nn.Linear(hidden, 1, bias=False)
        self.persistence_head = nn.Linear(hidden, 1, bias=False)
        self.uncertainty_head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.transport_head.weight)
        nn.init.zeros_(self.covariance_head.weight)
        nn.init.constant_(self.covariance_head.bias, -3.0)
        nn.init.zeros_(self.visibility_head.weight)
        nn.init.zeros_(self.persistence_head.weight)
        nn.init.zeros_(self.uncertainty_head.weight)
        nn.init.constant_(self.uncertainty_head.bias, -2.0)

    def _base(
        self,
        facts: ObjectFactSet,
        intent: WorldIntentDock,
        action: CoarseActionIntentState,
        *,
        collect_diagnostics: bool,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        intent.validate(hidden=self.hidden)
        public_object = self.object_content(facts.public_content)[:, None]
        private_objects = self.object_content(facts.content_innovation)
        # Preserve the old absolute object coordinate exactly while exposing
        # its provenance: one public base plus K object-owned innovations.
        # No new projection, capacity, or gain is introduced here.
        objects = public_object + private_objects
        transport_prior = self.object_transport_prior(
            facts.transport_prior.to(dtype=facts.content.dtype)
        )
        interval = (
            intent.interval_condition_innovation
            + action.tokens
        )
        normalized_goal = self.goal_memory_norm(intent.protected_goal_memory)
        goal_update, goal_attention = self.goal_read(
            self.goal_query_norm(interval),
            normalized_goal,
            normalized_goal,
            need_weights=collect_diagnostics,
            average_attn_weights=True,
        )
        base = (
            objects[:, None]
            + transport_prior[:, None]
            + interval[:, :, None]
            + goal_update[:, :, None]
        )
        typed_components = []
        for type_index, projection in enumerate(
            (self.object_semantic, self.object_appearance, self.object_geometry)
        ):
            component, _ = smooth_rms_contract(
                projection(intent.typed_relevance_value[..., type_index, :]),
                0.35,
            )
            typed_components.append(component)
        typed_components_value = torch.stack(typed_components, dim=3)
        # Typed values are a zero-preserving W working state, not a decoder
        # sidecar.  Public context may modulate their local update but cannot
        # suppress them to zero (the coefficient is bounded in [0.5, 1.5]).
        typed_components_value = typed_components_value * (
            1.0 + 0.5 * torch.tanh(base.float())[:, :, :, None]
        ).to(dtype=typed_components_value.dtype)
        if not collect_diagnostics:
            return base, typed_components_value, {}
        if goal_attention is None:
            goal_attention = interval.new_zeros(
                interval.shape[0],
                interval.shape[1],
                intent.protected_goal_memory.shape[1],
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
            "object_w_typed_sidecar_rms": typed_components_value.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_w_public_content_rms": public_object.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_w_object_innovation_rms": private_objects.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_w_object_innovation_variation": private_objects.detach()
            .float()
            .std(dim=1, unbiased=False)
            .mean(),
        }
        for type_index, name in enumerate(("semantic", "appearance", "geometry")):
            component = typed_components_value[..., type_index, :].detach().float()
            input_mass = intent.typed_relevance_mass[
                ..., type_index, 0
            ].detach().float()
            input_value = intent.typed_relevance_value[
                ..., type_index, :
            ].detach().float()
            metrics[f"object_w_{name}_contribution_rms"] = (
                component.square().mean().sqrt()
            )
            metrics[f"object_w_{name}_contribution_interval_variation"] = (
                component.std(dim=1, unbiased=False).mean()
            )
            metrics[f"object_w_{name}_contribution_object_variation"] = (
                component.std(dim=2, unbiased=False).mean()
            )
            metrics[f"object_w_{name}_input_relevance_mass"] = input_mass.mean()
            metrics[f"object_w_{name}_input_value_rms"] = (
                input_value.square().mean().sqrt()
            )
            metrics[f"object_w_{name}_input_interval_variation"] = (
                input_value.std(dim=1, unbiased=False).mean()
            )
            metrics[f"object_w_{name}_input_object_variation"] = (
                input_value.std(dim=2, unbiased=False).mean()
            )
        return base, typed_components_value, metrics

    def _field(
        self,
        *,
        facts: ObjectFactSet,
        hidden: Tensor,
        typed_sidecars: Tensor,
    ) -> FutureObjectDynamics:
        if tuple(typed_sidecars.shape) != (*hidden.shape[:-1], 3, hidden.shape[-1]):
            raise ValueError("W typed sidecars must align as [B,I,K,3,H]")
        # The typed states have already crossed the W blocks.  Decode those
        # exact states directly; multiplying them by the final public carrier
        # here previously re-publicized and often suppressed the very
        # interval/object variation W was meant to preserve.
        semantic_hidden = typed_sidecars[..., 0, :]
        appearance_hidden = typed_sidecars[..., 1, :]
        geometry_hidden = typed_sidecars[..., 2, :]
        semantic_delta = self.delta_head(semantic_hidden)
        object_transport = 0.50 * torch.tanh(
            self.transport_head(geometry_hidden).float()
        ).to(dtype=hidden.dtype)
        object_covariance = F.softplus(
            self.covariance_head(geometry_hidden).float()
        ).to(dtype=hidden.dtype)
        visibility = (
            1.0 - 2.0 * torch.sigmoid(self.visibility_head(appearance_hidden).float())
        ).to(dtype=hidden.dtype)
        persistence = (
            1.0 - 2.0 * torch.sigmoid(self.persistence_head(appearance_hidden).float())
        ).to(dtype=hidden.dtype)
        uncertainty = F.softplus(
            self.uncertainty_head(hidden).float()
        ).to(dtype=hidden.dtype)
        reliability = torch.zeros_like(uncertainty)
        visibility_probability = (1.0 + visibility.float()).clamp(0.0, 1.0)
        future_selector_validity = (
            facts.validity[:, None].float()
            * facts.existence.detach()[:, None].float().clamp(0.0, 1.0)
            * visibility_probability
        )
        current_selector_validity = (
            facts.validity.float()
            * facts.existence.detach().float().clamp(0.0, 1.0)
        )
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
            current_selector_validity=current_selector_validity,
            future_selector_validity=future_selector_validity,
            object_coordinates=facts.coordinates.to(dtype=hidden.dtype),
        )

    @staticmethod
    def _run_typed_block(
        block: _ObjectIntervalBlock,
        value: Tensor,
        *,
        causal_interval: bool,
    ) -> Tensor:
        if value.ndim != 5 or int(value.shape[3]) != 3:
            raise ValueError("typed W state must be [B,I,K,3,H]")
        batch, intervals, objects, types, hidden = value.shape
        typed_batch = value.permute(0, 3, 1, 2, 4).reshape(
            batch * types, intervals, objects, hidden
        )
        typed_batch = block(typed_batch, causal_interval=causal_interval)
        return typed_batch.reshape(
            batch, types, intervals, objects, hidden
        ).permute(0, 2, 3, 1, 4)

    @staticmethod
    def _typed_state_metrics(value: Tensor, *, prefix: str) -> dict[str, Tensor]:
        if value.ndim != 5 or int(value.shape[3]) != 3:
            raise ValueError("typed W diagnostics require [B,I,K,3,H]")
        metrics: dict[str, Tensor] = {}
        for type_index, name in enumerate(("semantic", "appearance", "geometry")):
            typed = value[..., type_index, :].detach().float()
            metrics[f"{prefix}_{name}_state_rms"] = typed.square().mean().sqrt()
            metrics[f"{prefix}_{name}_state_interval_variation"] = typed.std(
                dim=1,
                unbiased=False,
            ).mean()
            metrics[f"{prefix}_{name}_state_object_variation"] = typed.std(
                dim=2,
                unbiased=False,
            ).mean()
        return metrics

    def forward_w1(
        self,
        *,
        facts: ObjectFactSet,
        intent: WorldIntentDock,
        action: CoarseActionIntentState,
        collect_diagnostics: bool = False,
    ) -> tuple[FutureObjectDynamics, ObjectW1WorkingState, dict[str, Tensor]]:
        hidden, typed_sidecars, base_metrics = self._base(
            facts, intent, action, collect_diagnostics=collect_diagnostics
        )
        intent_boundary_field = self._field(
            facts=facts,
            hidden=hidden,
            typed_sidecars=typed_sidecars,
        )
        intent_boundary_field.validate()
        near = self.w1(hidden[:, :2], causal_interval=True)
        near_typed = self._run_typed_block(
            self.w1,
            typed_sidecars[:, :2],
            causal_interval=True,
        )
        field = self._field(
            facts=facts,
            hidden=near,
            typed_sidecars=near_typed,
        )
        field.validate(expected_intervals=2)
        metrics = self._metrics(field, prefix="object_w1") if collect_diagnostics else {}
        metrics.update(base_metrics)
        if collect_diagnostics:
            metrics.update(
                self._typed_state_metrics(near_typed, prefix="object_w1")
            )
        return field, ObjectW1WorkingState(
            near=near,
            far_base=hidden[:, 2:],
            near_typed=near_typed,
            far_typed=typed_sidecars[:, 2:],
            near_field=field,
            intent_boundary_field=intent_boundary_field,
        ), metrics

    def forward_w2(
        self,
        *,
        facts: ObjectFactSet,
        intent: WorldIntentDock,
        action: CoarseActionIntentState,
        w1_state: ObjectW1WorkingState,
        collect_diagnostics: bool = False,
    ) -> tuple[FutureObjectDynamics, dict[str, Tensor]]:
        del intent, action
        if tuple(w1_state.near.shape[1:3]) != (2, facts.objects) or tuple(
            w1_state.far_base.shape[1:3]
        ) != (2, facts.objects):
            raise ValueError("W2 requires both completed W1 intervals")
        if tuple(w1_state.near_typed.shape[1:4]) != (2, facts.objects, 3) or tuple(
            w1_state.far_typed.shape[1:4]
        ) != (2, facts.objects, 3):
            raise ValueError("W2 typed sidecars lost interval/object/type identity")
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
        typed_far_query = w1_state.far_typed.permute(0, 2, 3, 1, 4).reshape(
            batch * objects * 3, 2, hidden
        )
        typed_near_memory = w1_state.near_typed.permute(
            0, 2, 3, 1, 4
        ).reshape(batch * objects * 3, 2, hidden)
        typed_near_normalized = self.w1_memory_norm(typed_near_memory)
        typed_near_update, _ = self.w1_to_w2(
            self.w2_query_norm(typed_far_query),
            typed_near_normalized,
            typed_near_normalized,
            need_weights=False,
        )
        typed_near_update, _ = smooth_rms_contract(typed_near_update, 0.35)
        far_typed = (typed_far_query + typed_near_update).reshape(
            batch, objects, 3, 2, hidden
        ).permute(0, 3, 1, 2, 4)
        far_typed = self._run_typed_block(
            self.w2,
            far_typed,
            causal_interval=True,
        )
        far_field = self._field(
            facts=facts,
            hidden=far,
            typed_sidecars=far_typed,
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
                (near_field.transport_covariance, far_field.transport_covariance), dim=1
            ),
            visibility=torch.cat((near_field.visibility, far_field.visibility), dim=1),
            persistence=torch.cat((near_field.persistence, far_field.persistence), dim=1),
            uncertainty=torch.cat((near_field.uncertainty, far_field.uncertainty), dim=1),
            reliability=torch.cat((near_field.reliability, far_field.reliability), dim=1),
            current_selector_validity=near_field.current_selector_validity,
            future_selector_validity=torch.cat(
                (
                    near_field.future_selector_validity,
                    far_field.future_selector_validity,
                ),
                dim=1,
            ),
            object_coordinates=near_field.object_coordinates,
        )
        field.validate()
        metrics = self._metrics(field, prefix="object_w2") if collect_diagnostics else {}
        if collect_diagnostics:
            metrics.update(
                self._typed_state_metrics(
                    torch.cat((w1_state.near_typed, far_typed), dim=1),
                    prefix="object_w2",
                )
            )
        return field, metrics

    @staticmethod
    def _metrics(field: FutureObjectDynamics, *, prefix: str) -> dict[str, Tensor]:
        delta = field.semantic_delta.detach().float()
        condition_centered = delta - delta.mean(dim=0, keepdim=True)
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
            f"{prefix}_condition_centered_interval_variation": (
                condition_centered.std(dim=1, unbiased=False).mean()
            ),
            f"{prefix}_interval_adjacent_cosine": adjacent,
            f"{prefix}_object_pair_cosine": pair,
            f"{prefix}_transport_rms": field.transport_mean.detach().float().square().mean().sqrt(),
            f"{prefix}_future_selector_validity": field.future_selector_validity
            .detach()
            .float()
            .mean(),
        }


__all__ = ["ObjectFutureDynamicsCompiler", "ObjectW1WorkingState"]
