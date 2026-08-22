"""Recovered V120 W1/W2 four-interval object dynamics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .routing import smooth_rms_contract, variance_floored_centered_norm
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
    common_typed: Tensor
    near_residual_typed: Tensor
    far_residual_typed: Tensor
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
        self.base_interaction_norm = nn.LayerNorm(
            hidden, elementwise_affine=False
        )
        # One shared full-rank interaction is used by every type and by both
        # common/residual owners.  It is bias-free and reads a product with
        # the typed value, so a missing typed owner remains exact zero while
        # the complete object/action/goal base can change a present effect.
        construction_rng = torch.get_rng_state()
        self.typed_base_interaction = nn.Linear(hidden, hidden, bias=False)
        torch.set_rng_state(construction_rng)
        nn.init.zeros_(self.typed_base_interaction.weight)
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

    def _interact_with_base(
        self, typed: Tensor, base: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Inject the complete W condition without creating a free carrier."""

        if tuple(typed.shape[:-2]) != tuple(base.shape[:-1]) or int(
            typed.shape[-1]
        ) != self.hidden:
            raise ValueError("typed/base W interaction axes do not align")
        base_value = torch.tanh(self.base_interaction_norm(base).float())
        # Never unit-normalize a weak typed owner. A fixed variance floor
        # preserves exact zero and keeps the small-signal Jacobian bounded,
        # so the full base cannot turn numerical dust into a strong effect.
        typed_value, denominator = variance_floored_centered_norm(typed, 0.25)
        typed_value = typed_value.float()
        product = typed_value * base_value[..., None, :]
        interaction, _ = smooth_rms_contract(
            self.typed_base_interaction(
                product.to(dtype=typed.dtype)
            ),
            0.35,
        )
        return typed + interaction, interaction, denominator

    def _base(
        self,
        facts: ObjectFactSet,
        intent: WorldIntentDock,
        action: CoarseActionIntentState,
        *,
        collect_diagnostics: bool,
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
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
        common_components = []
        residual_components = []
        for type_index, projection in enumerate(
            (self.object_semantic, self.object_appearance, self.object_geometry)
        ):
            common_component, _ = smooth_rms_contract(
                projection(intent.typed_common_value[..., type_index, :]),
                0.35,
            )
            common_components.append(common_component)
            residual_component, _ = smooth_rms_contract(
                projection(
                    intent.typed_interval_residual_value[..., type_index, :]
                ),
                0.35,
            )
            residual_components.append(residual_component)
        typed_common = torch.stack(common_components, dim=2)
        typed_residual = torch.stack(residual_components, dim=3)
        # The protected typed values remain outside the interaction.  The
        # complete W base enters only through a bias-free typed-by-base product:
        # it can reorganize a present owner but cannot synthesize one from
        # zero. Common remains interval-free; residual retains [I,K,type].
        common_base = base.mean(dim=1)
        typed_common, common_interaction, common_interaction_denominator = self._interact_with_base(
            typed_common, common_base
        )
        typed_residual, residual_interaction, residual_interaction_denominator = self._interact_with_base(
            typed_residual, base
        )
        if not collect_diagnostics:
            return base, typed_common, typed_residual, {}
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
            "object_w_typed_common_state_rms": typed_common.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_w_typed_interval_residual_state_rms": typed_residual.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_w_common_base_interaction_rms": common_interaction.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_w_residual_base_interaction_rms": residual_interaction.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_w_common_base_interaction_denominator_min": (
                common_interaction_denominator.detach().float().amin()
            ),
            "object_w_residual_base_interaction_denominator_min": (
                residual_interaction_denominator.detach().float().amin()
            ),
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
            common_component = typed_common[..., type_index, :].detach().float()
            residual_component = typed_residual[..., type_index, :].detach().float()
            common_mass = intent.typed_common_mass[..., type_index, 0].detach().float()
            residual_mass = intent.typed_interval_residual_mass[
                ..., type_index, 0
            ].detach().float()
            common_value = intent.typed_common_value[..., type_index, :].detach().float()
            residual_value = intent.typed_interval_residual_value[
                ..., type_index, :
            ].detach().float()
            metrics[f"object_w_{name}_common_contribution_rms"] = (
                common_component.square().mean().sqrt()
            )
            metrics[f"object_w_{name}_common_contribution_object_variation"] = (
                common_component.std(dim=1, unbiased=False).mean()
            )
            metrics[f"object_w_{name}_residual_contribution_rms"] = (
                residual_component.square().mean().sqrt()
            )
            metrics[f"object_w_{name}_residual_contribution_interval_variation"] = (
                residual_component.std(dim=1, unbiased=False).mean()
            )
            metrics[f"object_w_{name}_residual_contribution_object_variation"] = (
                residual_component.std(dim=2, unbiased=False).mean()
            )
            metrics[f"object_w_{name}_common_input_mass"] = common_mass.mean()
            metrics[f"object_w_{name}_residual_input_mass"] = residual_mass.mean()
            metrics[f"object_w_{name}_common_input_value_rms"] = (
                common_value.square().mean().sqrt()
            )
            metrics[f"object_w_{name}_residual_input_value_rms"] = (
                residual_value.square().mean().sqrt()
            )
        return base, typed_common, typed_residual, metrics

    def _field(
        self,
        *,
        facts: ObjectFactSet,
        hidden: Tensor,
        typed_common: Tensor,
        typed_interval_residual: Tensor,
    ) -> FutureObjectDynamics:
        if tuple(typed_common.shape) != (
            hidden.shape[0], hidden.shape[2], 3, hidden.shape[-1]
        ):
            raise ValueError("W typed common state must align as [B,K,3,H]")
        if tuple(typed_interval_residual.shape) != (
            *hidden.shape[:-1], 3, hidden.shape[-1]
        ):
            raise ValueError("W typed residual must align as [B,I,K,3,H]")

        semantic_common = self.delta_head(typed_common[..., 0, :]).float()
        semantic_residual = self.delta_head(
            typed_interval_residual[..., 0, :]
        ).float()
        semantic_residual = semantic_residual - semantic_residual.mean(
            dim=1, keepdim=True
        )
        semantic_delta = (semantic_common[:, None] + semantic_residual).to(
            dtype=hidden.dtype
        )

        transport_common = 0.50 * torch.tanh(
            self.transport_head(typed_common[..., 2, :]).float()
        )
        transport_residual = 0.50 * torch.tanh(
            self.transport_head(typed_interval_residual[..., 2, :]).float()
        )
        transport_residual = transport_residual - transport_residual.mean(
            dim=1, keepdim=True
        )
        object_transport = (transport_common[:, None] + transport_residual).to(
            dtype=hidden.dtype
        )
        object_covariance = F.softplus(
            self.covariance_head(typed_interval_residual[..., 2, :]).float()
        ).to(dtype=hidden.dtype)
        visibility_common = -torch.tanh(
            0.5 * self.visibility_head(typed_common[..., 1, :]).float()
        )
        visibility_residual = -torch.tanh(
            0.5 * self.visibility_head(
                typed_interval_residual[..., 1, :]
            ).float()
        )
        visibility_residual = visibility_residual - visibility_residual.mean(
            dim=1, keepdim=True
        )
        visibility = (visibility_common[:, None] + visibility_residual).to(
            dtype=hidden.dtype
        )
        persistence_common = -torch.tanh(
            0.5 * self.persistence_head(typed_common[..., 1, :]).float()
        )
        persistence_residual = -torch.tanh(
            0.5 * self.persistence_head(
                typed_interval_residual[..., 1, :]
            ).float()
        )
        persistence_residual = persistence_residual - persistence_residual.mean(
            dim=1, keepdim=True
        )
        persistence = (persistence_common[:, None] + persistence_residual).to(
            dtype=hidden.dtype
        )
        uncertainty = F.softplus(
            self.uncertainty_head(hidden).float()
        ).to(dtype=hidden.dtype)
        reliability = torch.zeros_like(uncertainty)
        current_selector_validity = (
            facts.validity.float()
            * facts.existence.detach().float().clamp(0.0, 1.0)
        )
        # Predicted visibility is a status value, not authority to erase the
        # semantic/geometry candidates supervised in the same field.
        future_selector_validity = current_selector_validity[:, None].expand(
            -1, hidden.shape[1], -1, -1
        )
        current_reference = facts.content.detach().to(dtype=hidden.dtype)
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
            camera_coordinates=facts.camera_coordinates.to(dtype=hidden.dtype),
            camera_weights=(
                facts.camera_evidence_mass.float()
                * facts.camera_validity.float()
            ).to(dtype=hidden.dtype),
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

    @classmethod
    def _run_owned_typed_block(
        cls,
        block: _ObjectIntervalBlock,
        common: Tensor,
        residual: Tensor,
        *,
        causal_interval: bool,
    ) -> tuple[Tensor, Tensor]:
        """Run one W block over its common owner and interval innovations.

        Common is the first causal token for every object/type.  It therefore
        receives real W capacity while remaining a distinct owner; interval
        rows may read it, but it is never recovered by averaging residuals.
        """

        if common.ndim != 4 or residual.ndim != 5:
            raise ValueError("owned typed W state lost common/residual axes")
        if tuple(residual.shape[:1] + residual.shape[2:]) != tuple(common.shape):
            raise ValueError("owned typed W common/residual axes do not align")
        combined = torch.cat((common[:, None], residual), dim=1)
        completed = cls._run_typed_block(
            block,
            combined,
            causal_interval=causal_interval,
        )
        return completed[:, 0], completed[:, 1:]

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
        hidden, typed_common, typed_residual, base_metrics = self._base(
            facts, intent, action, collect_diagnostics=collect_diagnostics
        )
        near = self.w1(hidden[:, :2], causal_interval=True)
        w1_common, near_typed = self._run_owned_typed_block(
            self.w1,
            typed_common,
            typed_residual[:, :2],
            causal_interval=True,
        )
        field = self._field(
            facts=facts,
            hidden=near,
            typed_common=w1_common,
            typed_interval_residual=near_typed,
        )
        field.validate(expected_intervals=2)
        metrics = self._metrics(field, prefix="object_w1") if collect_diagnostics else {}
        metrics.update(base_metrics)
        if collect_diagnostics:
            metrics.update(
                self._typed_state_metrics(near_typed, prefix="object_w1")
            )
            metrics["object_w1_common_processing_delta_rms"] = (
                w1_common.detach().float() - typed_common.detach().float()
            ).square().mean().sqrt()
        return field, ObjectW1WorkingState(
            near=near,
            far_base=hidden[:, 2:],
            common_typed=w1_common,
            near_residual_typed=near_typed,
            far_residual_typed=typed_residual[:, 2:],
            near_field=field,
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
        if tuple(w1_state.near_residual_typed.shape[1:4]) != (2, facts.objects, 3) or tuple(
            w1_state.far_residual_typed.shape[1:4]
        ) != (2, facts.objects, 3):
            raise ValueError("W2 typed residual lost interval/object/type identity")
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
        # Common is a real W owner, not a pre-W bypass.  Carry it as the first
        # causal token in both the near memory and far query.  W2 can refine
        # that owner from W1 evidence while far residuals retain their own two
        # interval identities.
        typed_far_owned = torch.cat(
            (w1_state.common_typed[:, None], w1_state.far_residual_typed),
            dim=1,
        )
        typed_near_owned = torch.cat(
            (w1_state.common_typed[:, None], w1_state.near_residual_typed),
            dim=1,
        )
        typed_far_query = typed_far_owned.permute(0, 2, 3, 1, 4).reshape(
            batch * objects * 3, 3, hidden
        )
        typed_near_memory = typed_near_owned.permute(
            0, 2, 3, 1, 4
        ).reshape(batch * objects * 3, 3, hidden)
        typed_near_normalized = self.w1_memory_norm(typed_near_memory)
        typed_near_update, _ = self.w1_to_w2(
            self.w2_query_norm(typed_far_query),
            typed_near_normalized,
            typed_near_normalized,
            need_weights=False,
        )
        typed_near_update, _ = smooth_rms_contract(typed_near_update, 0.35)
        completed_far_owned = (typed_far_query + typed_near_update).reshape(
            batch, objects, 3, 3, hidden
        ).permute(0, 3, 1, 2, 4)
        completed_far_owned = self._run_typed_block(
            self.w2,
            completed_far_owned,
            causal_interval=True,
        )
        w2_common = completed_far_owned[:, 0]
        far_typed = completed_far_owned[:, 1:]
        completed_residual = torch.cat(
            (w1_state.near_residual_typed, far_typed), dim=1
        )
        # W blocks may introduce a common component into their private
        # interval carrier.  Remove it once at the public field boundary so
        # only the protected common owner can represent shared future effect.
        completed_residual = completed_residual - completed_residual.mean(
            dim=1, keepdim=True
        )
        field = self._field(
            facts=facts,
            hidden=torch.cat((w1_state.near, far), dim=1),
            typed_common=w2_common,
            typed_interval_residual=completed_residual,
        )
        field.validate()
        metrics = self._metrics(field, prefix="object_w2") if collect_diagnostics else {}
        if collect_diagnostics:
            metrics.update(
                self._typed_state_metrics(
                    completed_residual,
                    prefix="object_w2",
                )
            )
            metrics["object_w2_common_processing_delta_rms"] = (
                w2_common.detach().float()
                - w1_state.common_typed.detach().float()
            ).square().mean().sqrt()
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
