"""Schema39 spatial future selection and physical interval terminal.

P2 owns only object/camera selection.  It preserves interval and type axes in
``SelectedIntervalEvidence``.  The terminal then chooses among the four
physical intervals without a learned null; S may condition the W-owned key,
but it cannot create a value, support, object posterior, or independent time
vote.  This keeps one common W field and its interval innovation together until
the action-query consumer that is legally allowed to remove the interval axis.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .compiler import TypedP2EffectRead
from .routing import (
    register_gradient_axis_rms_metrics,
    smooth_rms_contract,
)
from .types import FutureObjectDynamics, PolicyIntentDock, normalized_entropy


@dataclass(frozen=True)
class SelectedIntervalEvidence:
    """W-owned evidence after spatial selection and before time selection."""

    key: Tensor  # [B,T,Q,I,Z,H]
    value: Tensor  # [B,T,Q,I,Z,H]
    common_value: Tensor  # [B,T,Q,I,Z,H]
    residual_value: Tensor  # [B,T,Q,I,Z,H]
    selected_s_context: Tensor  # [B,T,Q,I,Z,H]
    support: Tensor  # bool [B,I,Z]

    def validate(self) -> None:
        if self.key.ndim != 6:
            raise ValueError("selected interval key must be [B,T,Q,I,Z,H]")
        expected = tuple(self.key.shape)
        for name in (
            "value",
            "common_value",
            "residual_value",
            "selected_s_context",
        ):
            if tuple(getattr(self, name).shape) != expected:
                raise ValueError(f"selected interval {name} lost an identity axis")
        if int(self.key.shape[-2]) != 2:
            raise ValueError("selected interval evidence requires semantic/geometry")
        if tuple(self.support.shape) != (
            int(self.key.shape[0]),
            int(self.key.shape[3]),
            int(self.key.shape[4]),
        ):
            raise ValueError("selected interval support must be [B,I,Z]")
        if self.support.dtype != torch.bool:
            raise TypeError("selected interval support must be boolean")
        if not torch.allclose(
            self.value.float(),
            (self.common_value + self.residual_value).float(),
            atol=0.0,
            rtol=0.0,
        ):
            raise ValueError("selected interval common/residual identity failed")


def _safe_masked_softmax(logit: Tensor, support: Tensor, *, dim: int) -> Tensor:
    """Softmax with exact-zero all-invalid rows and finite backward."""

    if tuple(logit.shape) != tuple(support.shape):
        raise ValueError("masked softmax logit and support must align")
    has_support = support.any(dim=dim, keepdim=True)
    finite_logit = torch.where(support, logit.float(), torch.zeros_like(logit.float()))
    finite_logit = torch.where(has_support, finite_logit, torch.zeros_like(finite_logit))
    probability = torch.softmax(
        finite_logit.masked_fill(support.logical_not() & has_support, -torch.inf),
        dim=dim,
    )
    return probability * has_support.to(dtype=probability.dtype)


class ObjectFutureEffectTerminal(nn.Module):
    """P2 spatial selector followed by the P3 physical interval terminal."""

    TYPE_NAMES = TypedP2EffectRead.TYPE_NAMES
    S_TYPE_INDEX_BY_P2 = (0, 2)
    COMPLEMENTARY_VALUE_MAX_RMS = 0.35

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
        self.common_intent_key = nn.ModuleList(
            nn.Linear(hidden, hidden, bias=False) for _ in self.TYPE_NAMES
        )
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
    def _conditional_camera_probability(
        log_weight: Tensor,
        support: Tensor,
    ) -> Tensor:
        if tuple(log_weight.shape) != tuple(support.shape):
            raise ValueError("camera log measure and support must align")
        return _safe_masked_softmax(log_weight, support, dim=-1)

    @staticmethod
    def _covariance_aware_distance(delta: Tensor, covariance: Tensor) -> Tensor:
        variance_floor = 1.0 / float(7 * 7)
        xx = covariance[..., 0].float().clamp_min(0.0) + variance_floor
        xy = covariance[..., 1].float()
        yy = covariance[..., 2].float().clamp_min(0.0) + variance_floor
        determinant = (xx * yy - xy.square()).clamp_min(variance_floor**2)
        dx, dy = delta[..., 0], delta[..., 1]
        return variance_floor * (
            yy * dx.square() - 2.0 * xy * dx * dy + xx * dy.square()
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
        typed_common_intent = intent.typed_common_object_value
        typed_residual_intent = intent.typed_interval_residual_value
        if tuple(typed_common_intent.shape[:3]) != (batch, objects, 3):
            raise ValueError("P2 typed common intent lost object/type identity")
        if tuple(typed_residual_intent.shape[:4]) != (
            batch,
            intervals,
            objects,
            3,
        ):
            raise ValueError("P2 typed residual intent lost interval/object/type identity")

        temperature = self._temperatures().to(device=action_query.device)
        semantic_support = dynamics.chart_availability.float()[..., 0] > 0.0
        semantic_log_measure = dynamics.log_chart_availability[..., 0].float()
        camera_support = (
            (dynamics.camera_chart_availability.float()[..., 0] > 0.0)
            & (dynamics.camera_weights.float()[..., 0] > 0.0)
        )
        camera_log_measure = dynamics.log_camera_weight[..., 0].float()
        camera_probability = self._conditional_camera_probability(
            camera_log_measure,
            camera_support,
        )
        camera_has_support = camera_support.any(dim=-1)
        camera_coordinate = dynamics.camera_coordinates.float()
        camera_coordinate_mean = (
            camera_probability[..., None] * camera_coordinate
        ).sum(dim=2)
        camera_coordinate_variation = (
            camera_probability[..., None]
            * (camera_coordinate - camera_coordinate_mean[:, :, None]).square()
        ).sum(dim=2).mean(dim=-1).sqrt()
        camera_coordinate_variation = (
            camera_coordinate_variation * camera_has_support.float()
        ).mean()

        coordinate_query = torch.tanh(self.coordinate_query(action_query).float())
        future_coordinate = (
            camera_coordinate[:, None] + dynamics.transport_mean.float()
        ).clamp(-1.0, 1.0)
        coordinate_delta = (
            coordinate_query[:, :, :, None, None, None]
            - future_coordinate[:, None, None]
        )
        coordinate_distance = self._covariance_aware_distance(
            coordinate_delta,
            dynamics.transport_covariance.float()[:, None, None],
        )
        coordinate_score = (1.0 - 0.5 * coordinate_distance).clamp(-1.0, 1.0)
        coordinate_score = torch.where(
            camera_support[:, None, None, None],
            coordinate_score,
            torch.zeros_like(coordinate_score),
        )

        # Geometry contributes a conditional-K correction to the semantic
        # posterior without voting for object availability a second time.
        safe_camera_logit = torch.where(
            camera_has_support[..., None],
            camera_log_measure.masked_fill(~camera_support, -torch.inf),
            torch.zeros_like(camera_log_measure),
        )
        camera_log_normalizer = torch.logsumexp(
            safe_camera_logit,
            dim=-1,
            keepdim=True,
        )
        conditional_camera_log = torch.where(
            camera_support,
            safe_camera_logit - camera_log_normalizer,
            torch.zeros_like(camera_log_measure),
        )
        geometry_coordinate_logit = (
            temperature[2] * coordinate_score
            + conditional_camera_log[:, None, None, None]
        )
        geometry_coordinate_logit = torch.where(
            camera_support[:, None, None, None],
            geometry_coordinate_logit,
            torch.zeros_like(geometry_coordinate_logit),
        )
        safe_geometry_logit = torch.where(
            camera_has_support[:, None, None, None, :, None],
            geometry_coordinate_logit.masked_fill(
                ~camera_support[:, None, None, None], -torch.inf
            ),
            torch.zeros_like(geometry_coordinate_logit),
        )
        geometry_k_evidence = torch.logsumexp(
            safe_geometry_logit,
            dim=-1,
        )
        geometry_k_evidence = torch.where(
            camera_has_support[:, None, None, None],
            geometry_k_evidence,
            torch.zeros_like(geometry_k_evidence),
        )
        geometry_k_weight = camera_has_support[:, None, None, None].float()
        geometry_k_count = geometry_k_weight.sum(dim=-1, keepdim=True)
        geometry_k_common = torch.where(
            geometry_k_count > 0.0,
            (geometry_k_evidence * geometry_k_weight).sum(
                dim=-1, keepdim=True
            )
            / torch.where(
                geometry_k_count > 0.0,
                geometry_k_count,
                torch.ones_like(geometry_k_count),
            ),
            torch.zeros_like(geometry_k_count),
        )
        semantic_geometry_correction = torch.tanh(
            geometry_k_evidence - geometry_k_common
        ) * geometry_k_weight

        common_fields = (
            dynamics.semantic_common,
            dynamics.transport_common,
        )
        residual_fields = (
            dynamics.semantic_interval_residual,
            dynamics.transport_interval_residual,
        )
        full_fields = (
            dynamics.semantic_delta,
            dynamics.transport_mean,
        )
        common_values = (
            self.semantic_value(common_fields[0]),
            self.transport_value(common_fields[1]),
        )
        residual_values = (
            self.semantic_value(residual_fields[0]),
            self.transport_value(residual_fields[1]),
        )
        public_interval = self._bounded_unit(
            self.public_interval_key(intent.interval_residual_key)
        )

        selected_keys: list[Tensor] = []
        selected_values: list[Tensor] = []
        selected_common_values: list[Tensor] = []
        selected_residual_values: list[Tensor] = []
        selected_s_contexts: list[Tensor] = []
        interval_supports: list[Tensor] = []
        spatial_entropies: list[Tensor] = []
        spatial_maxima: list[Tensor] = []
        source_score_abs: list[Tensor] = []
        source_score_max: list[Tensor] = []

        for type_index in range(len(self.TYPE_NAMES)):
            intent_type_index = self.S_TYPE_INDEX_BY_P2[type_index]
            query = self._bounded_unit(self.source_query[type_index](action_query))
            source_key = self._bounded_unit(
                self.source_key[type_index](full_fields[type_index])
            )
            typed_route = (
                typed_common_intent[..., intent_type_index, :][:, None]
                + typed_residual_intent[..., intent_type_index, :]
            )
            typed_key = self._bounded_unit(
                self.typed_intent_key[type_index](typed_route)
            )
            public_common = self.common_intent_key[type_index](intent.common_key)
            public_context = (
                public_common[:, None, None, None]
                + public_interval[:, None, None]
            )

            if type_index == 0:
                source_score = torch.einsum(
                    "btqh,bikh->btqik", query, source_key
                )
                bounded_logit = (
                    temperature[0] * source_score
                    + semantic_geometry_correction
                )
                support = semantic_support[:, None].expand(-1, intervals, -1)
                log_measure = semantic_log_measure[:, None].expand(
                    -1, intervals, -1
                )
                candidate_typed_key = typed_key
            else:
                source_score = torch.einsum(
                    "btqh,bikch->btqikc", query, source_key
                )
                bounded_logit = (
                    temperature[0] * source_score
                    + temperature[2] * coordinate_score
                )
                support = camera_support[:, None].expand(
                    -1, intervals, -1, -1
                )
                log_measure = camera_log_measure[:, None].expand(
                    -1, intervals, -1, -1
                )
                candidate_typed_key = typed_key[..., None, :].expand_as(source_key)

            candidate_logit = bounded_logit + log_measure[:, None, None]
            candidate_support = support[:, None, None].expand_as(candidate_logit)
            flat_logit = candidate_logit.reshape(
                batch, horizon, basis, intervals, -1
            )
            flat_support = candidate_support.reshape_as(flat_logit)
            posterior = _safe_masked_softmax(flat_logit, flat_support, dim=-1)
            interval_support = support.flatten(start_dim=2).any(dim=-1)

            flat_key = source_key.reshape(batch, intervals, -1, self.hidden)
            flat_typed_key = candidate_typed_key.reshape(
                batch, intervals, -1, self.hidden
            )
            flat_common = common_values[type_index].reshape(
                batch, -1, self.hidden
            )[:, None].expand(-1, intervals, -1, -1)
            flat_residual = residual_values[type_index].reshape(
                batch, intervals, -1, self.hidden
            )
            selected_key = torch.einsum(
                "btqin,binh->btqih", posterior.to(dtype=flat_key.dtype), flat_key
            )
            selected_typed = torch.einsum(
                "btqin,binh->btqih",
                posterior.to(dtype=flat_typed_key.dtype),
                flat_typed_key,
            )
            selected_common = torch.einsum(
                "btqin,binh->btqih",
                posterior.to(dtype=flat_common.dtype),
                flat_common,
            )
            selected_residual = torch.einsum(
                "btqin,binh->btqih",
                posterior.to(dtype=flat_residual.dtype),
                flat_residual,
            )
            selected_keys.append(selected_key)
            selected_common_values.append(selected_common)
            selected_residual_values.append(selected_residual)
            selected_values.append(selected_common + selected_residual)
            selected_s_contexts.append(public_context + selected_typed.float())
            interval_supports.append(interval_support)
            source_score_abs.append(source_score.detach().float().abs().mean())
            source_score_max.append(source_score.detach().float().abs().amax())
            if collect_diagnostics:
                spatial_entropies.append(
                    normalized_entropy(posterior, dim=-1).detach().mean()
                )
                spatial_maxima.append(posterior.detach().amax(dim=-1).mean())

        selected = SelectedIntervalEvidence(
            key=torch.stack(selected_keys, dim=4),
            value=torch.stack(selected_values, dim=4),
            common_value=torch.stack(selected_common_values, dim=4),
            residual_value=torch.stack(selected_residual_values, dim=4),
            selected_s_context=torch.stack(selected_s_contexts, dim=4),
            support=torch.stack(interval_supports, dim=-1),
        )
        selected.validate()
        if not collect_diagnostics:
            return selected, {}
        metrics: dict[str, Tensor] = {
            "object_p2_temperature_content": temperature[0].detach(),
            "object_p2_temperature_intent": temperature[1].detach(),
            "object_p2_temperature_coordinate": temperature[2].detach(),
            "object_p2_content_score_abs": torch.stack(source_score_abs).mean(),
            "object_p2_content_score_max_abs": torch.stack(source_score_max).amax(),
            "object_p2_coordinate_score_abs": coordinate_score.detach().abs().mean(),
            "object_p2_coordinate_score_max_abs": coordinate_score.detach().abs().amax(),
            "object_p2_camera_support_fraction": camera_has_support.detach().float().mean(),
            "object_p2_camera_mixture_effective_count": torch.exp(
                -(
                    camera_probability.detach()
                    * torch.where(
                        camera_probability.detach() > 0.0,
                        camera_probability.detach().clamp_min(1.0e-30).log(),
                        torch.zeros_like(camera_probability.detach()),
                    )
                ).sum(dim=-1)
            ).mul(camera_has_support.detach().float()).mean(),
            "object_p2_camera_coordinate_variation": camera_coordinate_variation.detach(),
            "object_p2_geometry_to_semantic_k_correction_rms": (
                semantic_geometry_correction.detach().square().mean().sqrt()
            ),
            "object_p2_spatial_posterior_entropy": torch.stack(spatial_entropies).mean(),
            "object_p2_spatial_posterior_max": torch.stack(spatial_maxima).mean(),
            "object_p2_selected_w_key_rms": selected.key.detach().float().square().mean().sqrt(),
            "object_p2_w_key_interval_centered_variation_rms": (
                selected.key.detach().float()
                - selected.key.detach().float().mean(dim=3, keepdim=True)
            ).square().mean().sqrt(),
            "object_p2_selected_s_context_rms": (
                selected.selected_s_context.detach().float().square().mean().sqrt()
            ),
            "object_p2_spatial_selected_common_rms": (
                selected.common_value.detach().float().square().mean().sqrt()
            ),
            "object_p2_spatial_selected_interval_innovation_rms": (
                selected.residual_value.detach().float().square().mean().sqrt()
            ),
            "object_p2_spatial_common_innovation_identity_error": (
                selected.value.detach().float()
                - (
                    selected.common_value.detach()
                    + selected.residual_value.detach()
                ).float()
            ).abs().amax(),
            "object_p2_independent_s_interval_vote": action_query.new_zeros(
                (), dtype=torch.float32
            ),
            "object_p2_spatial_selector_has_null": action_query.new_zeros(
                (), dtype=torch.float32
            ),
        }
        for type_index, name in enumerate(self.TYPE_NAMES):
            metrics[f"object_p2_{name}_score_max_abs"] = source_score_max[type_index]
            metrics[f"object_p2_{name}_spatial_posterior_entropy"] = spatial_entropies[type_index]
            metrics[f"object_p2_{name}_spatial_posterior_max"] = spatial_maxima[type_index]
        return selected, metrics

    def temporal_terminal(
        self,
        action_query: Tensor,
        selected: SelectedIntervalEvidence,
        *,
        collect_diagnostics: bool,
    ) -> tuple[TypedP2EffectRead, dict[str, Tensor]]:
        selected.validate()
        batch, horizon, basis, intervals, types, hidden = selected.key.shape
        if tuple(action_query.shape) != (batch, horizon, basis, hidden):
            raise ValueError("temporal terminal action query is misaligned")
        temperature = self._temperatures().to(device=action_query.device)
        action_by_type = torch.stack(
            tuple(
                self._bounded_unit(self.source_query[index](action_query))
                for index in range(types)
            ),
            dim=3,
        )
        key = self._bounded_unit(selected.key)
        s_context = torch.tanh(self._bounded_unit(selected.selected_s_context))
        action_score = torch.einsum(
            "btqzh,btqizh->btqiz", action_by_type, key
        )
        intent_score = torch.einsum(
            "btqizh,btqizh->btqiz",
            action_by_type[:, :, :, None] * s_context,
            key,
        )
        logit = temperature[0] * action_score + temperature[1] * intent_score
        support = selected.support[:, None, None].expand_as(logit)
        posterior = _safe_masked_softmax(logit, support, dim=3)
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
        # Common is selected once and interval innovation is added once.  Do
        # not independently read a pre-added complete field: in BF16 that
        # creates a different rounding order and a false identity residual.
        value_by_type = common_by_type + residual_by_type
        raw_physical = value_by_type.sum(dim=3)
        _, shared_scale = smooth_rms_contract(
            raw_physical,
            self.COMPLEMENTARY_VALUE_MAX_RMS,
        )
        contracted = value_by_type * shared_scale.to(
            dtype=value_by_type.dtype
        )[..., None, :]
        effect = TypedP2EffectRead(effect_by_type=contracted)
        effect.validate()
        if not collect_diagnostics:
            return effect, {}

        neutral_posterior = _safe_masked_softmax(
            temperature[0] * action_score,
            support,
            dim=3,
        )
        interval_type_rms = selected.residual_value.detach().float().square().mean(
            dim=-1
        ).sqrt()
        weighted_residual_rms = (
            posterior.detach().float() * interval_type_rms
        ).sum(dim=3)
        selected_residual_rms = residual_by_type.detach().float().square().mean(
            dim=-1
        ).sqrt()
        residual_support = weighted_residual_rms > 1.0e-8
        retained = torch.where(
            residual_support,
            (selected_residual_rms / weighted_residual_rms.clamp_min(1.0e-8)).clamp(
                0.0, 1.0
            ),
            torch.zeros_like(selected_residual_rms),
        )
        support_count = residual_support.float().sum().clamp_min(1.0)
        retained_ratio = (retained * residual_support.float()).sum() / support_count
        has_residual_support = residual_support.any()
        cancelled_fraction = torch.where(
            has_residual_support,
            1.0 - retained_ratio,
            torch.zeros_like(retained_ratio),
        )
        common_value = common_by_type.sum(dim=3)
        residual_value = residual_by_type.sum(dim=3)
        metrics: dict[str, Tensor] = {
            "object_p3_interval_action_score_abs": action_score.detach().abs().mean(),
            "object_p3_interval_intent_score_abs": intent_score.detach().abs().mean(),
            "object_p3_interval_posterior_entropy": normalized_entropy(
                posterior.movedim(3, -1), dim=-1
            ).detach().mean(),
            "object_p3_interval_posterior_max": posterior.detach().amax(dim=3).mean(),
            "object_p3_interval_terminal_has_null": action_query.new_zeros(
                (), dtype=torch.float32
            ),
            "object_p3_s_condition_neutral_posterior_l1": (
                posterior.detach().float() - neutral_posterior.detach().float()
            ).abs().mean(),
            "object_p3_interval_innovation_retained_rms_ratio": retained_ratio,
            "object_p3_interval_innovation_cancelled_rms_fraction": (
                cancelled_fraction
            ),
            "object_p3_interval_innovation_cancellation_support_fraction": (
                residual_support.float().mean()
            ),
            "object_p3_selected_common_rms": common_value.square().mean().sqrt(),
            "object_p3_selected_interval_innovation_rms": residual_value.square().mean().sqrt(),
            "object_p3_interval_innovation_to_common_rms_ratio": (
                residual_value.square().mean().sqrt()
                / common_value.square().mean().sqrt().clamp_min(1.0e-8)
            ),
            "object_p3_terminal_common_innovation_identity_error": (
                value_by_type.detach().float()
                - (common_by_type.detach() + residual_by_type.detach()).float()
            ).abs().amax(),
            "object_p3_effect_precontract_rms": raw_physical.detach().square().mean().sqrt(),
            "object_p3_effect_postcontract_rms": effect.physical_sum.detach().float().square().mean().sqrt(),
            "object_p3_shared_effect_contract_scale_mean": shared_scale.detach().float().mean(),
            "object_p3_shared_effect_contract_compression": (
                1.0 - shared_scale.detach().float()
            ).mean(),
            "object_p3_semantic_effect_rms": effect.semantic.detach().float().square().mean().sqrt(),
            "object_p3_geometry_effect_rms": effect.geometry.detach().float().square().mean().sqrt(),
        }
        interval_mass = posterior.detach().float().mean(dim=4)
        for interval_index in range(intervals):
            metrics[f"object_p3_interval_{interval_index}_mass"] = (
                interval_mass[..., interval_index].mean()
            )
            for type_index, name in enumerate(self.TYPE_NAMES):
                metrics[f"object_p3_{name}_interval_{interval_index}_mass"] = (
                    posterior[..., interval_index, type_index].detach().float().mean()
                )
        for type_index, name in enumerate(self.TYPE_NAMES):
            metrics[f"object_p3_{name}_interval_posterior_entropy"] = normalized_entropy(
                posterior[..., type_index].movedim(3, -1), dim=-1
            ).detach().mean()
            metrics[f"object_p3_{name}_interval_posterior_max"] = (
                posterior[..., type_index].detach().amax(dim=3).mean()
            )
        if self.training:
            register_gradient_axis_rms_metrics(
                effect.effect_by_type,
                metrics,
                (
                    "gradient_tensor_p2_semantic_effect_rms",
                    "gradient_tensor_p2_geometry_effect_rms",
                ),
                dim=-2,
            )
        return effect, metrics

    def forward(
        self,
        action_query: Tensor,
        dynamics: FutureObjectDynamics,
        intent: PolicyIntentDock,
        *,
        collect_diagnostics: bool,
    ) -> tuple[TypedP2EffectRead, dict[str, Tensor]]:
        selected, spatial_metrics = self.spatial_select(
            action_query,
            dynamics,
            intent,
            collect_diagnostics=collect_diagnostics,
        )
        effect, terminal_metrics = self.temporal_terminal(
            action_query,
            selected,
            collect_diagnostics=collect_diagnostics,
        )
        if not collect_diagnostics:
            return effect, {}
        return effect, {**spatial_metrics, **terminal_metrics}


__all__ = ["ObjectFutureEffectTerminal", "SelectedIntervalEvidence"]
