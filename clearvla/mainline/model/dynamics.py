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
    common_base_interaction: Tensor
    common_base_interaction_denominator: Tensor
    near_residual_base_interaction: Tensor
    near_residual_base_interaction_denominator: Tensor


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
        # One shared full-rank interaction is used by every type and owner.
        # It is applied only after that owner crosses its generic W block:
        # W1 common reads the protected base once, W1 near reads completed
        # generic near once, and W2 far reads completed generic far once.
        # The bias-free typed product keeps a missing owner exact zero.
        construction_rng = torch.get_rng_state()
        self.typed_base_interaction = nn.Linear(hidden, hidden, bias=False)
        torch.set_rng_state(construction_rng)
        # A near-zero identity keeps the initial public heads neutral while
        # giving the protected base an ordinary gradient as soon as those
        # zero-initialized heads take their first optimizer update.  Exact
        # typed zero remains exact zero because the interaction is a product.
        with torch.no_grad():
            nn.init.eye_(self.typed_base_interaction.weight)
            self.typed_base_interaction.weight.mul_(1.0e-3)
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
        # Camera geometry is decoded on the real C axis. One shared
        # zero-preserving conditioner sees the typed geometry carrier together
        # with observable per-camera facts; the output heads are shared over C
        # and therefore cannot recreate C by broadcasting one object value.
        camera_construction_rng = torch.get_rng_state()
        self.camera_geometry_condition = nn.Linear(hidden + 6, hidden, bias=False)
        torch.set_rng_state(camera_construction_rng)
        self.transport_head = nn.Linear(hidden, 2, bias=False)
        self.covariance_head = nn.Linear(hidden, 3)
        # Status changes are optional typed values.  A free bias can learn the
        # dataset-wide mean disappearance/persistence and bypass the
        # appearance sidecar entirely, so both heads are bias-free.
        self.visibility_head = nn.Linear(hidden, 1, bias=False)
        self.persistence_head = nn.Linear(hidden, 1, bias=False)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.transport_head.weight)
        nn.init.zeros_(self.covariance_head.weight)
        with torch.no_grad():
            # Raw order is xx-logit, yy-logit, correlation-logit.
            self.covariance_head.bias.copy_(
                torch.tensor(
                    (-3.0, -3.0, 0.0),
                    dtype=self.covariance_head.bias.dtype,
                )
            )
        nn.init.zeros_(self.visibility_head.weight)
        nn.init.zeros_(self.persistence_head.weight)

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
        # These are protected owner inputs, not completed W states.  Each
        # typed-by-base interaction is deliberately delayed until the matching
        # common/near/far owner has crossed its generic W block.  This prevents
        # one pre-W interaction from being processed again by both W1 and W2.
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
        far_owned_gauge_start: int | None = None,
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
        semantic_residual = self._maybe_apply_far_owned_gauge(
            semantic_residual,
            far_owned_gauge_start,
        )
        semantic_delta = (semantic_common[:, None] + semantic_residual).to(
            dtype=hidden.dtype
        )

        transport_common = torch.tanh(
            self.transport_head(
                self._camera_geometry_carrier(
                    facts,
                    typed_common[..., 2, :],
                )
            ).float()
        )
        transport_residual = torch.tanh(
            self.transport_head(
                self._camera_geometry_carrier(
                    facts,
                    typed_interval_residual[..., 2, :],
                )
            ).float()
        )
        transport_residual = self._maybe_apply_far_owned_gauge(
            transport_residual,
            far_owned_gauge_start,
        )
        object_transport = (transport_common[:, None] + transport_residual).to(
            dtype=hidden.dtype
        )
        covariance_raw = self.covariance_head(
            self._camera_geometry_carrier(
                facts,
                typed_common[:, None, ..., 2, :]
                + typed_interval_residual[..., 2, :],
            )
        ).float()
        # The normalized 8x8 chart spacing is 2/7. Bound each diagonal from
        # one-cell variance through the maximum coordinate variance on
        # [-1, 1], then parameterize xy through a bounded correlation.
        variance_floor = (2.0 / 7.0) ** 2
        variance_ceiling = 1.0
        variance_xx = variance_floor + (
            variance_ceiling - variance_floor
        ) * torch.sigmoid(covariance_raw[..., 0])
        variance_yy = variance_floor + (
            variance_ceiling - variance_floor
        ) * torch.sigmoid(covariance_raw[..., 1])
        correlation = torch.tanh(covariance_raw[..., 2])
        covariance_xy = correlation * torch.sqrt(variance_xx * variance_yy)
        object_covariance = torch.stack(
            (variance_xx, covariance_xy, variance_yy),
            dim=-1,
        ).to(dtype=hidden.dtype)
        visibility_common = -torch.tanh(
            0.5 * self.visibility_head(typed_common[..., 1, :]).float()
        )
        visibility_residual = -torch.tanh(
            0.5 * self.visibility_head(
                typed_interval_residual[..., 1, :]
            ).float()
        )
        visibility_residual = self._maybe_apply_far_owned_gauge(
            visibility_residual,
            far_owned_gauge_start,
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
        persistence_residual = self._maybe_apply_far_owned_gauge(
            persistence_residual,
            far_owned_gauge_start,
        )
        persistence = (persistence_common[:, None] + persistence_residual).to(
            dtype=hidden.dtype
        )
        chart_availability = facts.chart_availability.float()
        # Predicted visibility is a status value, not authority to erase the
        # semantic/geometry candidates supervised in the same field.
        future_selector_validity = chart_availability[:, None].expand(
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
            chart_availability=chart_availability,
            future_selector_validity=future_selector_validity,
            camera_coordinates=facts.camera_coordinates.to(dtype=hidden.dtype),
            camera_chart_availability=facts.camera_chart_availability.to(
                dtype=hidden.dtype
            ),
            camera_weights=(
                facts.camera_evidence_mass.float()
                * facts.camera_chart_availability.float()
            ).to(dtype=hidden.dtype),
        )

    def _camera_geometry_carrier(
        self,
        facts: ObjectFactSet,
        typed_geometry: Tensor,
    ) -> Tensor:
        """Condition a present geometry owner independently in each camera."""

        if typed_geometry.ndim not in (3, 4) or int(typed_geometry.shape[-1]) != self.hidden:
            raise ValueError("typed camera geometry must be [B,K,H] or [B,I,K,H]")
        camera_context = torch.cat(
            (
                facts.camera_coordinates,
                facts.camera_transport_prior,
                facts.camera_support,
                facts.camera_chart_availability,
            ),
            dim=-1,
        ).to(dtype=typed_geometry.dtype)
        if typed_geometry.ndim == 3:
            if tuple(typed_geometry.shape[:2]) != tuple(camera_context.shape[:2]):
                raise ValueError("typed common geometry lost object identity")
            carrier = typed_geometry[:, :, None, :].expand(
                -1,
                -1,
                int(camera_context.shape[2]),
                -1,
            )
            context = camera_context
        else:
            if tuple(typed_geometry.shape[:1] + typed_geometry.shape[2:3]) != tuple(
                camera_context.shape[:2]
            ):
                raise ValueError("typed interval geometry lost object identity")
            carrier = typed_geometry[:, :, :, None, :].expand(
                -1,
                -1,
                -1,
                int(camera_context.shape[2]),
                -1,
            )
            context = camera_context[:, None].expand(
                -1,
                int(typed_geometry.shape[1]),
                -1,
                -1,
                -1,
            )
        condition = torch.tanh(
            self.camera_geometry_condition(
                torch.cat((carrier, context), dim=-1)
            ).float()
        ).to(dtype=carrier.dtype)
        # Camera facts may condition a nonzero typed owner, never synthesize
        # an interval effect from an absent one.
        return carrier * (1.0 + condition)

    @staticmethod
    def _complete_with_far_owned_zero_mean(near: Tensor, far: Tensor) -> Tensor:
        """Close the interval gauge without allowing far to rewrite near.

        The W1 rows are already public owner values.  Any common component
        left by W processing is therefore removed only from the W2 rows.  The
        final rounding closure is also charged to the last far row; gradients
        remain the ordinary derivatives of this algebra.
        """

        if near.ndim != far.ndim or tuple(near.shape[:1] + near.shape[2:]) != tuple(
            far.shape[:1] + far.shape[2:]
        ):
            raise ValueError("near/far causal gauge axes do not align")
        far_intervals = int(far.shape[1])
        if int(near.shape[1]) < 1 or far_intervals < 1:
            raise ValueError("causal gauge requires non-empty near and far owners")
        correction = (
            near.sum(dim=1, keepdim=True) + far.sum(dim=1, keepdim=True)
        ) / float(far_intervals)
        corrected_far = far - correction
        rounding_residual = near.sum(dim=1, keepdim=True) + corrected_far.sum(
            dim=1, keepdim=True
        )
        corrected_far = torch.cat(
            (
                corrected_far[:, :-1],
                corrected_far[:, -1:] - rounding_residual,
            ),
            dim=1,
        )
        return torch.cat((near, corrected_far), dim=1)

    @classmethod
    def _maybe_apply_far_owned_gauge(
        cls,
        value: Tensor,
        far_owned_gauge_start: int | None,
    ) -> Tensor:
        if far_owned_gauge_start is None:
            return value
        start = int(far_owned_gauge_start)
        if not 0 < start < int(value.shape[1]):
            raise ValueError("far-owned gauge must split non-empty near/far rows")
        return cls._complete_with_far_owned_zero_mean(
            value[:, :start],
            value[:, start:],
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
    def _run_separated_owned_typed_block(
        cls,
        block: _ObjectIntervalBlock,
        common: Tensor,
        residual: Tensor,
        *,
        causal_interval: bool,
    ) -> tuple[Tensor, Tensor]:
        """Run common and interval owners through one shared-parameter block.

        The two owners need the same W capacity, but they are not sequence
        positions of one carrier.  Concatenating common as causal row zero lets
        every residual repeatedly read it; later interval centring cannot undo
        the interval-dependent nonlinear write.  Separate calls preserve the
        owner boundary while reusing the exact same parameters.  Observable
        facts, action and goal have already entered each present owner through
        the zero-preserving typed-by-base interaction in :meth:`_base`.
        """

        if common.ndim != 4 or residual.ndim != 5:
            raise ValueError("owned typed W state lost common/residual axes")
        if tuple(residual.shape[:1] + residual.shape[2:]) != tuple(common.shape):
            raise ValueError("owned typed W common/residual axes do not align")
        completed_common = cls._run_typed_block(
            block,
            common[:, None],
            causal_interval=False,
        )
        completed_residual = cls._run_typed_block(
            block,
            residual,
            causal_interval=causal_interval,
        )
        return completed_common[:, 0], completed_residual

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
        w1_common_before_interaction, near_before_interaction = (
            self._run_separated_owned_typed_block(
                self.w1,
                typed_common,
                typed_residual[:, :2],
                causal_interval=True,
            )
        )
        # Common has one protected base condition and one W1 owner.  W2 never
        # processes or interacts with it again.
        w1_common, common_interaction, common_denominator = self._interact_with_base(
            w1_common_before_interaction,
            hidden.mean(dim=1),
        )
        # Near residuals read only the completed generic W1 near rows, once.
        near_typed, near_interaction, near_denominator = self._interact_with_base(
            near_before_interaction,
            near,
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
            metrics.update(
                {
                    "object_w_typed_common_state_rms": w1_common.detach()
                    .float()
                    .square()
                    .mean()
                    .sqrt(),
                    "object_w_typed_interval_residual_state_rms": near_typed.detach()
                    .float()
                    .square()
                    .mean()
                    .sqrt(),
                    "object_w_common_base_interaction_rms": common_interaction.detach()
                    .float()
                    .square()
                    .mean()
                    .sqrt(),
                    "object_w_residual_base_interaction_rms": near_interaction.detach()
                    .float()
                    .square()
                    .mean()
                    .sqrt(),
                    "object_w_common_base_interaction_denominator_min": common_denominator.detach()
                    .float()
                    .amin(),
                    "object_w_residual_base_interaction_denominator_min": near_denominator.detach()
                    .float()
                    .amin(),
                }
            )
            metrics["object_w1_common_processing_delta_rms"] = (
                w1_common_before_interaction.detach().float()
                - typed_common.detach().float()
            ).square().mean().sqrt()
        return field, ObjectW1WorkingState(
            near=near,
            far_base=hidden[:, 2:],
            common_typed=w1_common,
            near_residual_typed=near_typed,
            far_residual_typed=typed_residual[:, 2:],
            near_field=field,
            common_base_interaction=common_interaction,
            common_base_interaction_denominator=common_denominator,
            near_residual_base_interaction=near_interaction,
            near_residual_base_interaction_denominator=near_denominator,
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
        # Near interval innovations may inform the two far interval
        # innovations.  The bridge is residual-only: W2 neither processes nor
        # interacts with the protected common owner, and it never rewrites a
        # completed W1 near row.
        typed_far_query = w1_state.far_residual_typed.permute(
            0, 2, 3, 1, 4
        ).reshape(batch * objects * 3, 2, hidden)
        typed_near_memory = w1_state.near_residual_typed.permute(
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
        far_with_near = (typed_far_query + typed_near_update).reshape(
            batch, objects, 3, 2, hidden
        ).permute(0, 3, 1, 2, 4)
        far_before_interaction = self._run_typed_block(
            self.w2,
            far_with_near,
            causal_interval=True,
        )
        # Far residuals read only the completed generic W2 far rows, once.
        far_typed, far_interaction, far_denominator = self._interact_with_base(
            far_before_interaction,
            far,
        )
        completed_residual = self._complete_with_far_owned_zero_mean(
            w1_state.near_residual_typed,
            far_typed,
        )
        field = self._field(
            facts=facts,
            hidden=torch.cat((w1_state.near, far), dim=1),
            typed_common=w1_state.common_typed,
            typed_interval_residual=completed_residual,
            far_owned_gauge_start=2,
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
            residual_interaction = torch.cat(
                (w1_state.near_residual_base_interaction, far_interaction),
                dim=1,
            )
            residual_denominator = torch.cat(
                (
                    w1_state.near_residual_base_interaction_denominator,
                    far_denominator,
                ),
                dim=1,
            )
            metrics.update(
                {
                    "object_w_typed_common_state_rms": w1_state.common_typed.detach()
                    .float()
                    .square()
                    .mean()
                    .sqrt(),
                    "object_w_typed_interval_residual_state_rms": completed_residual.detach()
                    .float()
                    .square()
                    .mean()
                    .sqrt(),
                    "object_w_common_base_interaction_rms": w1_state.common_base_interaction.detach()
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
                        w1_state.common_base_interaction_denominator.detach()
                        .float()
                        .amin()
                    ),
                    "object_w_residual_base_interaction_denominator_min": residual_denominator.detach()
                    .float()
                    .amin(),
                }
            )
            metrics["object_w2_common_processing_delta_rms"] = (
                w1_state.common_typed.detach().new_zeros(())
            )
            metrics["object_w2_near_to_far_residual_update_rms"] = (
                typed_near_update.detach().float().square().mean().sqrt()
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
