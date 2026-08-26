"""Recovered V120 P2 consequence read and five-lane P3 compiler."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .routing import (
    PolicyRoleDeltaBank,
    register_gradient_axis_rms_metrics,
    register_gradient_rms_metric,
    smooth_rms_contract,
)
from .types import FutureObjectDynamics, PolicyIntentDock, normalized_entropy


@dataclass(frozen=True)
class ObjectConsequenceState:
    factual_base: Tensor
    effect: Tensor
    interaction: Tensor
    protected_consequence: Tensor


@dataclass(frozen=True)
class ObjectPolicyPlanDeltaBank:
    """Two optional innovations around two disjoint protected carriers."""

    protected_base: Tensor
    protected_policy_precision: Tensor
    temporal: Tensor
    state_change: Tensor

    @property
    def source_names(self) -> tuple[str, ...]:
        return (
            "p3_temporal",
            "p3_state_change",
        )

    def validate(self) -> None:
        expected = tuple(self.protected_base.shape)
        if len(expected) != 4:
            raise ValueError("object policy plan must be [B,T,Q,H]")
        if tuple(self.protected_policy_precision.shape) != expected:
            raise ValueError("protected policy precision lost [B,T,Q,H]")
        for name in ("temporal", "state_change"):
            if tuple(getattr(self, name).shape) != expected:
                raise ValueError(f"object policy {name} lost [B,T,Q,H]")

    def as_policy_role_bank(self, *, source_depth: int) -> PolicyRoleDeltaBank:
        self.validate()
        return PolicyRoleDeltaBank(
            values=torch.stack(
                (
                    self.temporal,
                    self.state_change,
                ),
                dim=1,
            ),
            source_names=self.source_names,
            source_depths=(int(source_depth),) * 5,
            protected_detail=self.protected_base,
            protected_policy_precision=self.protected_policy_precision,
        )


@dataclass(frozen=True)
class SelectedIntervalEvidence:
    """W evidence after type-local spatial selection and before I removal."""

    key: Tensor  # [B,T,Q,I,Z,H]
    value: Tensor  # [B,T,Q,I,Z,H]
    common_value: Tensor  # [B,T,Q,I,Z,H]
    residual_value: Tensor  # [B,T,Q,I,Z,H]
    selected_s_context: Tensor  # [B,T,Q,I,Z,H]
    support: Tensor  # bool [B,I,Z]

    def validate(self) -> None:
        if self.key.ndim != 6 or int(self.key.shape[-2]) != 2:
            raise ValueError("selected P2 evidence must be [B,T,Q,I,2,H]")
        expected = tuple(self.key.shape)
        for name in (
            "value",
            "common_value",
            "residual_value",
            "selected_s_context",
        ):
            if tuple(getattr(self, name).shape) != expected:
                raise ValueError(f"selected P2 {name} lost an identity axis")
        if tuple(self.support.shape) != (expected[0], expected[3], expected[4]):
            raise ValueError("selected P2 support must be [B,I,2]")
        if self.support.dtype != torch.bool:
            raise TypeError("selected P2 support must be boolean")
        if not torch.equal(self.value, self.common_value + self.residual_value):
            raise ValueError("selected P2 common/residual identity failed")


def _safe_masked_log_softmax(
    logit: Tensor,
    support: Tensor,
    *,
    dim: int,
) -> tuple[Tensor, Tensor]:
    """Return FP32 probability/log pairs with exact-zero invalid rows."""

    if tuple(logit.shape) != tuple(support.shape):
        raise ValueError("masked softmax logit and support must align")
    has_support = support.any(dim=dim, keepdim=True)
    finite_logit = torch.where(support, logit.float(), torch.zeros_like(logit.float()))
    finite_logit = torch.where(has_support, finite_logit, torch.zeros_like(finite_logit))
    masked = finite_logit.masked_fill(~support & has_support, -torch.inf)
    log_probability = torch.log_softmax(masked, dim=dim)
    active = support & has_support
    log_probability = torch.where(
        active,
        log_probability,
        torch.zeros_like(log_probability),
    )
    probability = torch.where(
        active,
        log_probability.exp(),
        torch.zeros_like(log_probability),
    )
    return probability, log_probability


def _safe_masked_softmax(logit: Tensor, support: Tensor, *, dim: int) -> Tensor:
    """Return finite probabilities and exact zero on all-invalid rows."""

    probability, _ = _safe_masked_log_softmax(logit, support, dim=dim)
    return probability


class ObjectFutureEffectReader(nn.Module):
    """Select W spatial evidence before the no-null physical-I terminal."""

    TYPE_NAMES = ("semantic", "geometry")
    S_TYPE_INDEX_BY_P2 = (0, 2)

    def __init__(self, *, hidden: int, content_dim: int, route_dim: int) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.source_query = nn.ModuleList(
            nn.Linear(hidden, hidden, bias=False) for _ in self.TYPE_NAMES
        )
        self.source_key = nn.ModuleList(
            (
                nn.Linear(content_dim, hidden, bias=False),
                nn.Linear(2, hidden, bias=False),
            )
        )
        self.public_interval_key = nn.Linear(hidden, hidden, bias=False)
        self.typed_intent_key = nn.ModuleList(
            nn.Linear(route_dim, hidden, bias=False) for _ in self.TYPE_NAMES
        )
        self.coordinate_query = nn.Linear(hidden, 2, bias=False)
        self.semantic_value = nn.Linear(content_dim, hidden, bias=False)
        self.transport_value = nn.Linear(2, hidden, bias=False)
        self.temperature_logit = nn.Parameter(torch.zeros(3))

    def _temperatures(self) -> Tensor:
        return 0.25 + 3.75 * torch.sigmoid(self.temperature_logit.float())

    @staticmethod
    def _bounded_unit(value: Tensor, *, norm_floor: float = 0.25) -> Tensor:
        value_f = value.float()
        return value_f / (
            value_f.square().sum(dim=-1, keepdim=True) + float(norm_floor) ** 2
        ).sqrt()

    @staticmethod
    def _covariance_aware_distance(delta: Tensor, covariance: Tensor) -> Tensor:
        """Retain the active I+Sigma positive-definite coordinate metric."""

        covariance_f = covariance.float()
        metric_xx = 1.0 + covariance_f[..., 0]
        metric_xy = covariance_f[..., 1]
        metric_yy = 1.0 + covariance_f[..., 2]
        determinant = metric_xx * metric_yy - metric_xy.square()
        dx, dy = delta[..., 0], delta[..., 1]
        return (
            metric_yy * dx.square()
            - 2.0 * metric_xy * dx * dy
            + metric_xx * dy.square()
        ) / determinant

    def spatial_select(
        self,
        action_query: Tensor,
        dynamics: FutureObjectDynamics,
        intent: PolicyIntentDock,
        *,
        collect_diagnostics: bool,
    ) -> tuple[SelectedIntervalEvidence, dict[str, Tensor]]:
        dynamics.validate()
        if action_query.ndim != 4:
            raise ValueError("P2 action query must be [B,T,Q,H]")
        batch, horizon, basis, hidden = action_query.shape
        if hidden != self.hidden:
            raise ValueError("P2 action query hidden width is invalid")
        intent.validate(horizon=horizon, hidden=self.hidden)
        intervals, objects = dynamics.semantic_delta.shape[1:3]
        if tuple(intent.typed_common_value.shape[:3]) != (batch, objects, 3):
            raise ValueError("P2 typed common S metadata lost K/type")
        if tuple(intent.typed_interval_residual_value.shape[:4]) != (
            batch,
            intervals,
            objects,
            3,
        ):
            raise ValueError("P2 typed residual S metadata lost I/K/type")

        temperature = self._temperatures().to(device=action_query.device)
        object_measure = dynamics.chart_availability[..., 0].float().clamp(0.0, 1.0)
        semantic_support = object_measure > 0.0
        semantic_log_measure = torch.where(
            semantic_support,
            dynamics.log_chart_availability[..., 0].float(),
            torch.zeros_like(object_measure),
        )
        camera_measure = dynamics.camera_chart_availability[..., 0].float().clamp(
            0.0, 1.0
        )
        camera_support = semantic_support[..., None] & (camera_measure > 0.0)
        camera_log_measure = torch.where(
            camera_support,
            dynamics.log_camera_chart_availability[..., 0].float(),
            torch.zeros_like(camera_measure),
        )
        conditional_camera, conditional_camera_log = _safe_masked_log_softmax(
            camera_log_measure,
            camera_support,
            dim=-1,
        )
        geometry_log_measure = torch.where(
            camera_support,
            semantic_log_measure[..., None] + conditional_camera_log,
            torch.zeros_like(conditional_camera),
        )

        coordinate_query = torch.tanh(self.coordinate_query(action_query).float())
        future_coordinate = (
            dynamics.camera_coordinates[:, None].float()
            + dynamics.transport_mean.float()
        ).clamp(-1.0, 1.0)
        coordinate_delta = (
            coordinate_query[:, :, :, None, None, None]
            - future_coordinate[:, None, None]
        )
        coordinate_distance = self._covariance_aware_distance(
            coordinate_delta,
            dynamics.transport_covariance[:, None, None],
        )
        coordinate_score = (-0.25 * coordinate_distance).clamp(-1.0, 0.0)

        common_fields = (
            dynamics.semantic_common,
            dynamics.transport_common,
        )
        residual_fields = (
            dynamics.semantic_interval_innovation,
            dynamics.transport_interval_innovation,
        )
        full_fields = (dynamics.semantic_delta, dynamics.transport_mean)
        common_values = (
            self.semantic_value(common_fields[0]),
            self.transport_value(common_fields[1]),
        )
        residual_values = (
            self.semantic_value(residual_fields[0]),
            self.transport_value(residual_fields[1]),
        )
        public_interval = self.public_interval_key(intent.interval_key).float()

        selected_keys: list[Tensor] = []
        selected_common_values: list[Tensor] = []
        selected_residual_values: list[Tensor] = []
        selected_s_contexts: list[Tensor] = []
        interval_supports: list[Tensor] = []
        spatial_posteriors: list[Tensor] = []
        source_scores: list[Tensor] = []

        for type_index in range(len(self.TYPE_NAMES)):
            query = self._bounded_unit(self.source_query[type_index](action_query))
            source_key = self._bounded_unit(
                self.source_key[type_index](full_fields[type_index])
            )
            typed_route = (
                intent.typed_common_value[..., self.S_TYPE_INDEX_BY_P2[type_index], :][
                    :, None
                ]
                + intent.typed_interval_residual_value[
                    ..., self.S_TYPE_INDEX_BY_P2[type_index], :
                ]
            )
            typed_candidate = self.typed_intent_key[type_index](typed_route).float()

            if type_index == 0:
                source_score = torch.einsum(
                    "btqh,bikh->btqik",
                    query,
                    source_key,
                )
                support = semantic_support[:, None].expand(-1, intervals, -1)
                log_measure = semantic_log_measure[:, None].expand_as(support)
                candidate_logit = temperature[0] * source_score + log_measure[
                    :, None, None
                ]
                candidate_support = support[:, None, None].expand_as(candidate_logit)
                candidate_typed = typed_candidate
            else:
                source_score = torch.einsum(
                    "btqh,bikch->btqikc",
                    query,
                    source_key,
                )
                support = camera_support[:, None].expand(
                    -1, intervals, -1, -1
                )
                log_measure = geometry_log_measure[:, None].expand_as(support)
                candidate_logit = (
                    temperature[0] * source_score
                    + temperature[2] * coordinate_score
                    + log_measure[:, None, None]
                )
                candidate_support = support[:, None, None].expand_as(candidate_logit)
                candidate_typed = typed_candidate[..., None, :].expand_as(source_key)

            flat_logit = candidate_logit.reshape(batch, horizon, basis, intervals, -1)
            flat_support = candidate_support.reshape_as(flat_logit)
            posterior = _safe_masked_softmax(flat_logit, flat_support, dim=-1)
            interval_support = support.flatten(start_dim=2).any(dim=-1)
            flat_key = source_key.reshape(batch, intervals, -1, self.hidden)
            flat_typed = candidate_typed.reshape(batch, intervals, -1, self.hidden)
            flat_common = common_values[type_index].reshape(
                batch, -1, self.hidden
            )[:, None].expand(-1, intervals, -1, -1)
            flat_residual = residual_values[type_index].reshape(
                batch,
                intervals,
                -1,
                self.hidden,
            )
            posterior_for_key = posterior.to(dtype=flat_key.dtype)
            selected_keys.append(
                torch.einsum("btqin,binh->btqih", posterior_for_key, flat_key)
            )
            selected_common_values.append(
                torch.einsum(
                    "btqin,binh->btqih",
                    posterior.to(dtype=flat_common.dtype),
                    flat_common,
                )
            )
            selected_residual_values.append(
                torch.einsum(
                    "btqin,binh->btqih",
                    posterior.to(dtype=flat_residual.dtype),
                    flat_residual,
                )
            )
            selected_typed = torch.einsum(
                "btqin,binh->btqih",
                posterior.to(dtype=flat_typed.dtype),
                flat_typed,
            )
            selected_s_contexts.append(
                public_interval[:, None, None] + selected_typed.float()
            )
            interval_supports.append(interval_support)
            spatial_posteriors.append(posterior)
            source_scores.append(source_score)

        common_value = torch.stack(selected_common_values, dim=4)
        residual_value = torch.stack(selected_residual_values, dim=4)
        selected = SelectedIntervalEvidence(
            key=torch.stack(selected_keys, dim=4),
            value=common_value + residual_value,
            common_value=common_value,
            residual_value=residual_value,
            selected_s_context=torch.stack(selected_s_contexts, dim=4),
            support=torch.stack(interval_supports, dim=-1),
        )
        selected.validate()
        if not collect_diagnostics:
            return selected, {}
        return selected, {
            "object_p2_content_score_abs": torch.stack(
                [score.detach().float().abs().mean() for score in source_scores]
            ).mean(),
            "object_p2_content_score_max_abs": torch.stack(
                [score.detach().float().abs().amax() for score in source_scores]
            ).amax(),
            "object_p2_coordinate_score_abs": coordinate_score.detach().abs().mean(),
            "object_p2_coordinate_score_max_abs": coordinate_score.detach().abs().amax(),
            "object_p2_spatial_posterior_entropy": torch.stack(
                [normalized_entropy(value, dim=-1).detach().mean() for value in spatial_posteriors]
            ).mean(),
            "object_p2_spatial_posterior_max": torch.stack(
                [value.detach().amax(dim=-1).mean() for value in spatial_posteriors]
            ).mean(),
            "object_p2_spatial_selector_has_null": action_query.new_zeros(
                (), dtype=torch.float32
            ),
            "object_p2_selected_w_key_rms": selected.key.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p2_selected_s_context_rms": selected.selected_s_context.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p2_spatial_common_residual_identity_error": (
                selected.value.detach().float()
                - (selected.common_value.detach() + selected.residual_value.detach()).float()
            )
            .abs()
            .amax(),
        }

    def temporal_terminal(
        self,
        action_query: Tensor,
        selected: SelectedIntervalEvidence,
        *,
        collect_diagnostics: bool,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        selected.validate()
        batch, horizon, basis, intervals, types, hidden = selected.key.shape
        if tuple(action_query.shape) != (batch, horizon, basis, hidden):
            raise ValueError("P2 terminal action query lost [B,T,Q,H]")
        action_by_type = torch.stack(
            [
                self._bounded_unit(self.source_query[index](action_query))
                for index in range(types)
            ],
            dim=3,
        )
        selected_key = self._bounded_unit(selected.key)
        s_context = torch.tanh(self._bounded_unit(selected.selected_s_context))
        conditioned_key = selected_key + selected_key * s_context
        interval_score = torch.einsum(
            "btqzh,btqizh->btqiz",
            action_by_type,
            conditioned_key,
        )
        temperature = self._temperatures().to(device=action_query.device)
        support = selected.support[:, None, None].expand_as(interval_score)
        posterior = _safe_masked_softmax(
            temperature[1] * interval_score,
            support,
            dim=3,
        )
        common_by_type = torch.einsum(
            "btqiz,btqizh->btqzh",
            posterior.to(dtype=selected.common_value.dtype),
            selected.common_value,
        )
        residual_by_type = torch.einsum(
            "btqiz,btqizh->btqzh",
            posterior.to(dtype=selected.residual_value.dtype),
            selected.residual_value,
        )
        value_by_type = common_by_type + residual_by_type
        gradient_metrics: dict[str, Tensor] = {}
        if collect_diagnostics and self.training:
            register_gradient_axis_rms_metrics(
                value_by_type,
                gradient_metrics,
                (
                    "gradient_tensor_p2_semantic_effect_rms",
                    "gradient_tensor_p2_geometry_effect_rms",
                ),
                dim=3,
            )
        raw_effect = value_by_type.sum(dim=3)
        if not collect_diagnostics:
            return raw_effect, {}
        neutral_key_score = torch.einsum(
            "btqzh,btqizh->btqiz",
            action_by_type,
            selected_key,
        )
        neutral_posterior = _safe_masked_softmax(
            temperature[1] * neutral_key_score,
            support,
            dim=3,
        )
        metrics: dict[str, Tensor] = {
            "object_p2_intent_score_abs": (
                interval_score.detach() - neutral_key_score.detach()
            )
            .abs()
            .mean(),
            "object_p2_intent_score_max_abs": (
                interval_score.detach() - neutral_key_score.detach()
            )
            .abs()
            .amax(),
            "object_p2_combined_logit_max_abs": interval_score.detach().abs().amax(),
            "object_p2_temperature_content": temperature[0].detach(),
            "object_p2_temperature_intent": temperature[1].detach(),
            "object_p2_temperature_coordinate": temperature[2].detach(),
            "object_p2_posterior_entropy": normalized_entropy(
                posterior.movedim(3, -1), dim=-1
            )
            .detach()
            .mean(),
            "object_p2_posterior_max": posterior.detach().amax(dim=3).mean(),
            "object_p2_null_mass": action_query.new_zeros((), dtype=torch.float32),
            "object_p2_terminal_has_null": action_query.new_zeros(
                (), dtype=torch.float32
            ),
            "object_p2_s_condition_posterior_l1": (
                posterior.detach() - neutral_posterior.detach()
            )
            .abs()
            .mean(),
            "object_p2_effect_precontract_rms": raw_effect.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p2_semantic_effect_rms": value_by_type[..., 0, :]
            .detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p2_geometry_effect_rms": value_by_type[..., 1, :]
            .detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p2_terminal_common_residual_identity_error": (
                value_by_type.detach().float()
                - (common_by_type.detach() + residual_by_type.detach()).float()
            )
            .abs()
            .amax(),
        }
        metrics.update(gradient_metrics)
        interval_mass = posterior.detach().float().mean(dim=4)
        for index in range(intervals):
            metrics[f"object_p2_interval_{index}_mass"] = interval_mass[
                ..., index
            ].mean()
        return raw_effect, metrics

    def forward(
        self,
        action_query: Tensor,
        dynamics: FutureObjectDynamics,
        intent: PolicyIntentDock,
        *,
        collect_diagnostics: bool,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        selected, spatial_metrics = self.spatial_select(
            action_query,
            dynamics,
            intent,
            collect_diagnostics=collect_diagnostics,
        )
        value, terminal_metrics = self.temporal_terminal(
            action_query,
            selected,
            collect_diagnostics=collect_diagnostics,
        )
        if not collect_diagnostics:
            return value, {}
        return value, {**spatial_metrics, **terminal_metrics}


class ZeroPreservingObjectConsequence(nn.Module):
    """Exact identity when the P2 effect is zero."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.fact = nn.Linear(hidden, hidden, bias=False)
        self.interaction = nn.Linear(hidden, hidden, bias=False)

    def forward(
        self,
        *,
        factual_base: Tensor,
        effect: Tensor,
        collect_diagnostics: bool = True,
    ) -> tuple[ObjectConsequenceState, dict[str, Tensor]]:
        if tuple(factual_base.shape) != tuple(effect.shape):
            raise ValueError("factual base and effect must align")
        interaction = self.interaction(torch.tanh(self.fact(factual_base)) * effect)
        protected = factual_base + effect + interaction
        state = ObjectConsequenceState(
            factual_base=factual_base,
            effect=effect,
            interaction=interaction,
            protected_consequence=protected,
        )
        if not collect_diagnostics:
            return state, {}
        return state, {
            "object_consequence_effect_rms": effect.detach().float().square().mean().sqrt(),
            "object_consequence_interaction_rms": interaction.detach().float().square().mean().sqrt(),
            "object_consequence_ratio": (
                (effect + interaction).detach().float().square().mean().sqrt()
                / factual_base.detach().float().square().mean().sqrt().clamp_min(1e-6)
            ),
        }


class ObjectPolicyPlanCompiler(nn.Module):
    """Compile the two P3 innovations without duplicating protected owners."""

    def __init__(self, *, hidden: int, horizon: int, basis: int) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.horizon = int(horizon)
        self.basis = int(basis)
        # Consume the six removed alias projections' historical draws so the
        # retained P3 weights and every subsequently constructed module keep
        # their R1f fresh-run initialization stream. These temporary tensors
        # are never registered, serialized, moved or executed.
        removed_aliases = tuple(
            nn.Linear(hidden, hidden, bias=False) for _ in range(6)
        )
        self.temporal_action = nn.Linear(hidden, hidden, bias=False)
        self.temporal_effect = nn.Linear(hidden, hidden, bias=False)
        self.temporal_lane = nn.Linear(hidden, hidden, bias=False)
        self.state_change_action = nn.Linear(hidden, hidden, bias=False)
        self.state_change_temporal = nn.Linear(hidden, hidden, bias=False)
        self.state_change_lane = nn.Linear(hidden, hidden, bias=False)
        del removed_aliases

    def forward(
        self,
        *,
        p1_policy_residual: Tensor,
        consequence: ObjectConsequenceState,
        intent: PolicyIntentDock,
        action_query: Tensor,
        collect_diagnostics: bool = True,
    ) -> tuple[ObjectPolicyPlanDeltaBank, dict[str, Tensor]]:
        expected = (int(action_query.shape[0]), self.horizon, self.basis, self.hidden)
        if tuple(action_query.shape) != expected or tuple(
            p1_policy_residual.shape
        ) != expected:
            raise ValueError("P3 inputs must align as [B,T,Q,H]")
        for name in (
            "factual_base",
            "effect",
            "interaction",
            "protected_consequence",
        ):
            if tuple(getattr(consequence, name).shape) != expected:
                raise ValueError(f"P3 consequence {name} lost [B,T,Q,H]")
        intent.validate(horizon=self.horizon, hidden=self.hidden)
        temporal_context = intent.temporal_control[:, :, None].expand(
            -1, -1, self.basis, -1
        )
        consequence_innovation = consequence.effect + consequence.interaction
        temporal_private = temporal_context + self.temporal_effect(
            consequence_innovation
        )
        temporal_raw = self.temporal_lane(
            temporal_private * torch.tanh(self.temporal_action(action_query))
        )
        state_change_source = intent.state_change_evidence[:, None, None].expand(
            -1, self.horizon, self.basis, -1
        )
        state_change_modulation = torch.tanh(
            self.state_change_action(action_query)
            + self.state_change_temporal(temporal_context)
        )
        state_change_raw = self.state_change_lane(
            state_change_source * state_change_modulation
        )
        temporal, temporal_scale = smooth_rms_contract(temporal_raw, 0.35)
        state_change, state_change_scale = smooth_rms_contract(
            state_change_raw,
            0.35,
        )
        bank = ObjectPolicyPlanDeltaBank(
            protected_base=consequence.protected_consequence,
            protected_policy_precision=p1_policy_residual,
            temporal=temporal,
            state_change=state_change,
        )
        bank.validate()
        if not collect_diagnostics:
            return bank, {}
        metrics = {
            "object_p3_protected_policy_precision_rms": p1_policy_residual.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p3_consequence_innovation_rms": consequence_innovation.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p3_temporal_private_rms": temporal_private.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p3_temporal_rms": temporal.detach().float().square().mean().sqrt(),
            "object_p3_temporal_contract_min": temporal_scale.detach().float().amin(),
            "object_p3_state_change_rms": state_change.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p3_state_change_contract_min": state_change_scale.detach()
            .float()
            .amin(),
        }
        if self.training:
            register_gradient_rms_metric(
                p1_policy_residual,
                metrics,
                "gradient_tensor_p1_protected_policy_precision_rms",
            )
            register_gradient_rms_metric(
                temporal,
                metrics,
                "gradient_tensor_p3_temporal_rms",
            )
            register_gradient_rms_metric(
                state_change,
                metrics,
                "gradient_tensor_p3_state_change_rms",
            )
        return bank, metrics


__all__ = [
    "ObjectConsequenceState",
    "ObjectFutureEffectReader",
    "ObjectPolicyPlanCompiler",
    "ObjectPolicyPlanDeltaBank",
    "SelectedIntervalEvidence",
    "ZeroPreservingObjectConsequence",
]
