"""Stateless intent, direct future supervision and coarse action."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .routing import smooth_rms_contract, variance_floored_centered_norm
from .types import (
    INTERVAL_BOUNDS,
    ActionIntentDock,
    CoarseActionIntentState,
    FutureObjectDynamics,
    IntentFutureSupervision,
    ObjectFactSet,
    ObjectIntentState,
    normalized_entropy,
)

TYPED_INTENT_NAMES = ("semantic", "appearance", "geometry")


def _causal_mask(length: int, device: torch.device) -> Tensor:
    return torch.triu(
        torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1
    )


def _interval_slices(length: int) -> tuple[slice, ...]:
    if length < 1:
        raise ValueError("future sequence cannot be empty")
    rows: list[slice] = []
    for lower, upper in INTERVAL_BOUNDS:
        start = min(max(int(lower) - 1, 0), length - 1)
        stop = min(max(int(upper), start + 1), length)
        rows.append(slice(start, stop))
    return tuple(rows)


class _CrossRead(nn.Module):
    def __init__(self, hidden: int, heads: int, maximum_rms: float = 0.35) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.memory_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(
            hidden, heads, bias=False, dropout=0.0, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden, elementwise_affine=False),
            nn.Linear(hidden, 2 * hidden, bias=False),
            nn.SiLU(),
            nn.Linear(2 * hidden, hidden, bias=False),
        )
        self.maximum_rms = float(maximum_rms)

    def forward(
        self,
        query: Tensor,
        memory: Tensor,
        *,
        padding_mask: Tensor | None = None,
        diagnostics: bool = False,
        zero_preserving_innovation: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        normalized_memory = self.memory_norm(memory)
        update, weights = self.attention(
            self.query_norm(query),
            normalized_memory,
            normalized_memory,
            key_padding_mask=padding_mask,
            need_weights=diagnostics,
            average_attn_weights=True,
        )
        update, _ = smooth_rms_contract(update, self.maximum_rms)
        value = query + update
        ffn_input = update if zero_preserving_innovation else value
        ffn, _ = smooth_rms_contract(self.ffn(ffn_input), self.maximum_rms)
        value = value + ffn
        if weights is None:
            weights = query.new_zeros(query.shape[0], query.shape[1], memory.shape[1])
        return value, update + ffn, weights


class _SelfBlock(nn.Module):
    def __init__(self, hidden: int, heads: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(
            hidden, heads, bias=False, dropout=0.0, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden, elementwise_affine=False),
            nn.Linear(hidden, 2 * hidden, bias=False),
            nn.GELU(),
            nn.Linear(2 * hidden, hidden, bias=False),
        )

    def forward(self, value: Tensor, *, causal: bool = False) -> Tensor:
        normalized = self.norm(value)
        update, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=_causal_mask(int(value.shape[1]), value.device) if causal else None,
            need_weights=False,
        )
        update, _ = smooth_rms_contract(update, 0.35)
        value = value + update
        ffn, _ = smooth_rms_contract(self.ffn(value), 0.35)
        return value + ffn


class StatelessObjectIntentOrganizer(nn.Module):
    """V120 observable intent without scalar progress or synthetic phases."""

    def __init__(
        self,
        *,
        hidden: int,
        goal_dim: int,
        state_dim: int,
        action_dim: int,
        content_dim: int,
        route_dim: int,
        horizon: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.horizon = int(horizon)
        self.goal_input = nn.Linear(goal_dim, hidden, bias=False)
        self.goal_queries = nn.Parameter(torch.randn(1, 4, hidden) * 0.02)
        self.goal_read = _CrossRead(hidden, heads)
        self.goal_self = _SelfBlock(hidden, heads)
        # State observations (-8/-4/0) and executed actions
        # (-24/-16/-12/-8/-6/-4/-2/-1) do not describe paired rows.  Keep a
        # typed time-union sequence: absent modalities are algebraic zero and
        # the final scalar records state(+1) versus action(-1) ownership.
        history_width = state_dim + action_dim + state_dim + 2
        self.history_input = nn.Sequential(
            nn.LayerNorm(history_width, elementwise_affine=False),
            nn.Linear(history_width, hidden, bias=False),
        )
        self.history_blocks = nn.ModuleList(_SelfBlock(hidden, heads) for _ in range(2))
        self.object_content = nn.Linear(content_dim, hidden, bias=False)
        self.object_semantic = nn.Linear(route_dim, hidden, bias=False)
        self.object_appearance = nn.Linear(route_dim, hidden, bias=False)
        self.object_geometry = nn.Linear(route_dim, hidden, bias=False)
        self.interval_identity = nn.Parameter(torch.randn(1, 4, hidden) * 0.02)
        self.interval_goal = _CrossRead(hidden, heads)
        self.interval_history = _CrossRead(hidden, heads)
        self.interval_object = _CrossRead(hidden, heads)
        self.typed_relevance_queries = nn.ModuleList(
            nn.Linear(hidden, route_dim, bias=False) for _ in TYPED_INTENT_NAMES
        )
        # Initial temperature is exactly one.  It remains bounded in [0.25, 4]
        # and therefore cannot turn the fixed-zero null comparison into an
        # unbounded selector gain.
        self.typed_temperature_logit = nn.Parameter(
            torch.full((len(TYPED_INTENT_NAMES),), -1.3862943611198906)
        )
        self.interval_self = _SelfBlock(hidden, heads)
        self.temporal_identity = nn.Parameter(torch.randn(1, horizon, hidden) * 0.02)
        self.temporal_read = _CrossRead(hidden, heads)
        self.state_change_query = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        self.state_change_query_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.state_change_key_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.state_change_read = nn.MultiheadAttention(
            hidden, heads, bias=False, dropout=0.0, batch_first=True
        )
        self.state_change_input = nn.Linear(state_dim, hidden, bias=False)
        self.state_change_transport = nn.Linear(2, hidden, bias=False)
        self.state_change_fuse = nn.Linear(2 * hidden, hidden, bias=False)

    @staticmethod
    def _paired_history(
        state_history: Tensor,
        state: Tensor,
        executed_history: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if state_history.ndim != 3 or state.ndim != 2 or executed_history.ndim != 3:
            raise ValueError("intent history requires state/action sequences")
        if int(state_history.shape[1]) != 3 or int(executed_history.shape[1]) != 8:
            raise ValueError(
                "intent history requires state offsets -8/-4/0 and the eight "
                "executed-action offsets"
            )
        batch, _, state_dim = state_history.shape
        action_dim = int(executed_history.shape[-1])
        # The last state-history row already denotes offset zero.  Use the
        # explicitly supplied current state at that row instead of duplicating
        # it as a fourth observation.
        states = torch.cat((state_history[:, :2], state[:, None]), dim=1)
        previous = torch.cat((states[:, :1], states[:, :-1]), dim=1)
        state_delta = states - previous
        state_delta[:, 0] = 0

        state_offsets = (-8, -4, 0)
        action_offsets = (-24, -16, -12, -8, -6, -4, -2, -1)
        rows: list[tuple[int, int, Tensor, Tensor, Tensor]] = []
        zero_state = states.new_zeros(batch, state_dim)
        zero_action = executed_history.new_zeros(batch, action_dim)
        for index, offset in enumerate(state_offsets):
            rows.append(
                (offset, 0, states[:, index], zero_action, state_delta[:, index])
            )
        for index, offset in enumerate(action_offsets):
            rows.append(
                (offset, 1, zero_state, executed_history[:, index], zero_state)
            )
        rows.sort(key=lambda item: (item[0], item[1]))
        packed: list[Tensor] = []
        aligned_delta: list[Tensor] = []
        for offset_value, source_order, state_value, action_value, delta_value in rows:
            offset = state_value.new_full(
                (batch, 1), float(offset_value) / 24.0
            )
            source = state_value.new_full(
                (batch, 1), 1.0 if source_order == 0 else -1.0
            )
            packed.append(
                torch.cat(
                    (state_value, action_value, delta_value, offset, source), dim=-1
                )
            )
            aligned_delta.append(delta_value)
        return torch.stack(packed, dim=1), torch.stack(aligned_delta, dim=1)

    def _object_tokens(self, facts: ObjectFactSet) -> tuple[Tensor, Tensor]:
        facts.validate()
        # One bias-free projection owns both coordinates.  The scene-wide
        # value is represented once while K carries only object innovations;
        # public + innovation is algebraically the former absolute K value.
        public_scene = self.object_content(facts.public_content)[:, None]
        object_innovations = self.object_content(facts.content_innovation)
        return public_scene, object_innovations

    @staticmethod
    def _bounded_unit(value: Tensor, *, floor: float = 0.25) -> Tensor:
        value_f = value.float()
        return value_f / (
            value_f.square().sum(dim=-1, keepdim=True) + float(floor) ** 2
        ).sqrt()

    def _typed_relevance(
        self,
        *,
        interval_condition_innovation: Tensor,
        facts: ObjectFactSet,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        """Build protected common and signed interval-residual typed values.

        Fixed interval identities never enter either selector.  The common
        carrier and interval-centred carrier are normalized and scored
        independently, then exported as different value objects.  They are
        never averaged into one scalar selector, so common evidence cannot
        consume the range or ownership of a real interval residual.
        """

        common_carrier = interval_condition_innovation.mean(dim=1, keepdim=True)
        differential_carrier = interval_condition_innovation - common_carrier
        # Neither common nor differential evidence may turn a numerically
        # tiny condition into a confident selector merely by passing through
        # LayerNorm.  The shared floor preserves zero and bounds both
        # Jacobians by the same normalized-chart contract.
        common_source, common_denominator = variance_floored_centered_norm(
            common_carrier,
            0.25,
        )
        differential_source, differential_denominator = variance_floored_centered_norm(
            differential_carrier,
            0.25,
        )
        common_query = torch.stack(
            tuple(projection(common_source) for projection in self.typed_relevance_queries),
            dim=2,
        )  # [B,1,type,R]
        differential_query = torch.stack(
            tuple(
                projection(differential_source)
                for projection in self.typed_relevance_queries
            ),
            dim=2,
        )  # [B,I,type,R]
        typed_route = torch.stack(
            (facts.semantic, facts.appearance, facts.geometry), dim=2
        )  # [B,K,type,R]
        common_score = torch.einsum(
            "bitr,bktr->bikt",
            self._bounded_unit(common_query),
            self._bounded_unit(typed_route),
        ).clamp(-1.0, 1.0)
        differential_score = torch.einsum(
            "bitr,bktr->bikt",
            self._bounded_unit(differential_query),
            self._bounded_unit(typed_route),
        ).clamp(-1.0, 1.0)
        temperature = 0.25 + 3.75 * torch.sigmoid(
            self.typed_temperature_logit.float()
        )
        common_score = common_score[:, 0]
        common_signal = torch.tanh(
            common_score
            * temperature.to(device=common_score.device)[None, None]
        )
        residual_signal = torch.tanh(
            differential_score
            * temperature.to(device=differential_score.device)[None, None, None]
        )
        residual_signal = residual_signal - residual_signal.mean(
            dim=1, keepdim=True
        )
        # One shared scale per object/type preserves the zero-mean identity
        # while keeping the signed selector in [-1, 1].
        residual_signal = residual_signal / residual_signal.abs().amax(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        common_validity = facts.validity.float()[:, :, None, :].clamp(0.0, 1.0)
        interval_validity = common_validity[:, None]
        common_mass = (
            common_signal.abs()[..., None] * common_validity
        ).to(dtype=typed_route.dtype)
        common_value = (
            common_signal[..., None].to(dtype=typed_route.dtype)
            * common_validity.to(dtype=typed_route.dtype)
            * typed_route
        )
        interval_residual_mass = (
            residual_signal.abs()[..., None] * interval_validity
        ).to(dtype=typed_route.dtype)
        interval_residual_value = (
            residual_signal[..., None].to(dtype=typed_route.dtype)
            * interval_validity.to(dtype=typed_route.dtype)
            * typed_route[:, None]
        )

        common_components: list[Tensor] = []
        residual_components: list[Tensor] = []
        for type_index, projection in enumerate(
            (self.object_semantic, self.object_appearance, self.object_geometry)
        ):
            common_route = common_value[..., type_index, :].mean(dim=1)
            common_component, _ = smooth_rms_contract(
                projection(common_route), 0.35
            )
            common_components.append(common_component)
            residual_route = interval_residual_value[
                ..., type_index, :
            ].mean(dim=2)
            residual_component, _ = smooth_rms_contract(
                projection(residual_route), 0.35
            )
            residual_components.append(residual_component)
        typed_common_components = torch.stack(common_components, dim=1)
        typed_interval_residual_components = torch.stack(
            residual_components, dim=2
        )
        common_raw_context = typed_common_components.sum(dim=1) / (3.0**0.5)
        _, common_context_scale = smooth_rms_contract(common_raw_context, 0.35)
        typed_common_components = typed_common_components * common_context_scale[
            :, None
        ].to(dtype=typed_common_components.dtype)
        residual_raw_context = (
            typed_interval_residual_components.sum(dim=2) / (3.0**0.5)
        )
        _, residual_context_scale = smooth_rms_contract(
            residual_raw_context, 0.35
        )
        typed_interval_residual_components = (
            typed_interval_residual_components
            * residual_context_scale[:, :, None].to(
                dtype=typed_interval_residual_components.dtype
            )
        )
        return (
            common_mass,
            common_value,
            interval_residual_mass,
            interval_residual_value,
            typed_common_components,
            typed_interval_residual_components,
            common_score,
            differential_score,
            common_signal.abs(),
            residual_signal.abs(),
            temperature,
            common_denominator,
            differential_denominator,
        )

    def forward(
        self,
        *,
        goal_tokens: Tensor,
        goal_mask: Tensor,
        state_history: Tensor,
        state: Tensor,
        executed_history: Tensor,
        facts: ObjectFactSet,
        collect_diagnostics: bool,
    ) -> tuple[ObjectIntentState, dict[str, Tensor]]:
        if goal_tokens.ndim != 3 or goal_mask.ndim != 2:
            raise ValueError("intent organizer requires full T5 tokens and mask")
        batch = int(goal_tokens.shape[0])
        goal_memory = self.goal_input(goal_tokens)
        goal_query = self.goal_queries.to(
            device=goal_memory.device, dtype=goal_memory.dtype
        ).expand(batch, -1, -1)
        _, protected_goal, goal_attention = self.goal_read(
            goal_query,
            goal_memory,
            padding_mask=~goal_mask.to(device=goal_memory.device, dtype=torch.bool),
            diagnostics=collect_diagnostics,
            zero_preserving_innovation=True,
        )
        # Learned goal queries are addresses only.  Retaining their residual
        # as a value made goal-dropout samples a trainable task prior instead
        # of an exact language null.  The innovation path preserves query-
        # specific reads while zero T5 content now remains algebraic zero.
        protected_goal = self.goal_self(protected_goal)
        paired_history, observed_state_delta = self._paired_history(
            state_history, state, executed_history
        )
        history = self.history_input(paired_history)
        for block in self.history_blocks:
            history = block(history, causal=True)
        public_scene, objects = self._object_tokens(facts)
        object_memory = torch.cat((public_scene, objects), dim=1)
        interval_base = self.interval_identity.to(
            device=objects.device, dtype=objects.dtype
        ).expand(batch, -1, -1)
        _, goal_innovation, interval_goal_attention = self.interval_goal(
            interval_base,
            protected_goal,
            diagnostics=collect_diagnostics,
            zero_preserving_innovation=True,
        )
        _, history_innovation, interval_history_attention = self.interval_history(
            interval_base,
            history,
            diagnostics=collect_diagnostics,
            zero_preserving_innovation=True,
        )
        _, object_innovation, interval_object_attention = self.interval_object(
            interval_base,
            object_memory,
            diagnostics=collect_diagnostics,
            zero_preserving_innovation=True,
        )
        interval_template = self.interval_self(interval_base)
        public_intervals = self.interval_self(
            interval_base
            + goal_innovation
            + history_innovation
            + object_innovation
        )
        interval_condition_innovation = public_intervals - interval_template
        (
            typed_common_mass,
            typed_common_value,
            typed_interval_residual_mass,
            typed_interval_residual_value,
            typed_common_policy_components,
            typed_interval_residual_policy_components,
            typed_common_score,
            typed_differential_score,
            typed_common_signal_strength,
            typed_residual_signal_strength,
            typed_temperature,
            typed_common_denominator,
            typed_differential_denominator,
        ) = self._typed_relevance(
            interval_condition_innovation=interval_condition_innovation,
            facts=facts,
        )
        typed_common_policy_context = (
            typed_common_policy_components.sum(dim=1) / (3.0**0.5)
        )
        typed_interval_residual_policy_context = (
            typed_interval_residual_policy_components.sum(dim=2) / (3.0**0.5)
        )
        typed_policy_context = (
            typed_common_policy_context[:, None]
            + typed_interval_residual_policy_context
        )
        policy_intervals = public_intervals + typed_policy_context
        policy_interval_innovation = (
            interval_condition_innovation + typed_policy_context
        )
        temporal_base = self.temporal_identity.to(
            device=public_intervals.device, dtype=public_intervals.dtype
        ).expand(batch, -1, -1)
        temporal, temporal_innovation, temporal_attention = self.temporal_read(
            temporal_base,
            policy_interval_innovation,
            diagnostics=collect_diagnostics,
            zero_preserving_innovation=True,
        )
        state_change_values = self.state_change_input(observed_state_delta)
        state_change_history, state_change_attention = self.state_change_read(
            self.state_change_query_norm(
                self.state_change_query.to(
                    device=history.device, dtype=history.dtype
                ).expand(batch, -1, -1)
            ),
            self.state_change_key_norm(history),
            state_change_values,
            need_weights=collect_diagnostics,
            average_attn_weights=True,
        )
        if state_change_attention is None:
            state_change_attention = history.new_zeros(batch, 1, history.shape[1])
        state_change_history = state_change_history[:, 0]
        transport_tokens = self.state_change_transport(facts.transport_prior)
        transport_validity = facts.validity.to(
            device=transport_tokens.device, dtype=transport_tokens.dtype
        )
        state_change_transport = (
            transport_tokens * transport_validity
        ).sum(dim=1) / transport_validity.sum(dim=1).clamp_min(1.0)
        state_change_evidence, _ = smooth_rms_contract(
            self.state_change_fuse(
                torch.cat((state_change_history, state_change_transport), dim=-1)
            ),
            0.20,
        )
        state_out = ObjectIntentState(
            protected_goal_set=protected_goal,
            history_tokens=history,
            public_scene_token=public_scene,
            object_tokens=objects,
            public_interval_carrier=public_intervals,
            interval_condition_innovation=interval_condition_innovation,
            policy_interval_context=policy_intervals,
            policy_interval_innovation=policy_interval_innovation,
            temporal_queries=temporal_innovation,
            state_change_evidence=state_change_evidence,
            typed_common_mass=typed_common_mass,
            typed_common_value=typed_common_value,
            typed_interval_residual_mass=typed_interval_residual_mass,
            typed_interval_residual_value=typed_interval_residual_value,
            typed_common_policy_components=typed_common_policy_components,
            typed_interval_residual_policy_components=(
                typed_interval_residual_policy_components
            ),
            goal_attention=goal_attention,
            interval_goal_attention=interval_goal_attention,
            interval_history_attention=interval_history_attention,
            interval_object_attention=interval_object_attention,
        )
        state_out.validate(horizon=self.horizon, hidden=self.hidden)
        if not collect_diagnostics:
            return state_out, {}
        public_condition_centered = public_intervals.detach().float()
        public_condition_centered = (
            public_condition_centered
            - public_condition_centered.mean(dim=0, keepdim=True)
        )
        metrics: dict[str, Tensor] = {
            "object_intent_goal_attention_entropy": normalized_entropy(
                goal_attention, dim=-1
            ).detach().mean(),
            "object_intent_interval_goal_entropy": normalized_entropy(
                interval_goal_attention, dim=-1
            ).detach().mean(),
            "object_intent_interval_history_entropy": normalized_entropy(
                interval_history_attention, dim=-1
            ).detach().mean(),
            "object_intent_interval_object_entropy": normalized_entropy(
                interval_object_attention, dim=-1
            ).detach().mean(),
            "object_intent_public_interval_variation": public_intervals.detach().float().std(
                dim=1, unbiased=False
            ).mean(),
            "object_intent_condition_innovation_rms": interval_condition_innovation
            .detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_intent_condition_interval_variation": interval_condition_innovation
            .detach()
            .float()
            .std(dim=1, unbiased=False)
            .mean(),
            "object_intent_policy_innovation_rms": policy_interval_innovation
            .detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_intent_policy_innovation_interval_variation": policy_interval_innovation
            .detach()
            .float()
            .std(dim=1, unbiased=False)
            .mean(),
            "object_intent_public_condition_centered_interval_variation": (
                public_condition_centered.std(dim=1, unbiased=False).mean()
            ),
            "object_intent_policy_interval_variation": policy_intervals.detach().float().std(
                dim=1, unbiased=False
            ).mean(),
            "object_intent_temporal_variation": temporal.detach().float().std(
                dim=1, unbiased=False
            ).mean(),
            "object_intent_temporal_read_innovation_rms": temporal_innovation
            .detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_intent_temporal_read_interval_variation": temporal_innovation
            .detach()
            .float()
            .std(dim=1, unbiased=False)
            .mean(),
            "object_intent_temporal_attention_entropy": normalized_entropy(
                temporal_attention,
                dim=-1,
            )
            .detach()
            .mean(),
            "object_intent_goal_innovation_rms": goal_innovation.detach().float().square().mean().sqrt(),
            "object_intent_history_innovation_rms": history_innovation.detach().float().square().mean().sqrt(),
            "object_intent_object_innovation_rms": object_innovation.detach().float().square().mean().sqrt(),
            "object_intent_public_scene_content_rms": public_scene.detach().float().square().mean().sqrt(),
            "object_intent_object_content_innovation_rms": objects.detach().float().square().mean().sqrt(),
            "object_intent_object_content_innovation_variation": objects.detach().float().std(dim=1, unbiased=False).mean(),
            "object_intent_typed_policy_context_rms": typed_policy_context.detach().float().square().mean().sqrt(),
            "object_intent_typed_common_policy_context_rms": typed_common_policy_context.detach().float().square().mean().sqrt(),
            "object_intent_typed_interval_residual_policy_context_rms": typed_interval_residual_policy_context.detach().float().square().mean().sqrt(),
            "object_intent_typed_common_norm_denominator_min": typed_common_denominator.detach().float().amin(),
            "object_intent_typed_differential_norm_denominator_min": typed_differential_denominator.detach().float().amin(),
            "object_intent_typed_fact_unsupported_fraction": (
                1.0 - facts.validity.detach().float().clamp(0.0, 1.0)
            ).mean(),
            "object_intent_observed_state_delta_rms": observed_state_delta.detach().float().square().mean().sqrt(),
            "object_intent_observed_transport_rms": facts.transport_prior.detach().float().square().mean().sqrt(),
            "object_intent_state_change_history_rms": state_change_history.detach().float().square().mean().sqrt(),
            "object_intent_state_change_transport_rms": state_change_transport.detach().float().square().mean().sqrt(),
            "object_intent_state_change_evidence_rms": state_change_evidence.detach().float().square().mean().sqrt(),
            "object_intent_state_change_attention_entropy": normalized_entropy(
                state_change_attention, dim=-1
            ).detach().mean(),
        }
        raw_routes = (facts.semantic, facts.appearance, facts.geometry)
        for type_index, name in enumerate(TYPED_INTENT_NAMES):
            common_mass = typed_common_mass[..., type_index, 0].detach().float()
            common_selected = typed_common_value[..., type_index, :].detach().float()
            residual_mass = typed_interval_residual_mass[
                ..., type_index, 0
            ].detach().float()
            residual_selected = typed_interval_residual_value[
                ..., type_index, :
            ].detach().float()
            common_component = typed_common_policy_components[
                ..., type_index, :
            ].detach().float()
            residual_component = typed_interval_residual_policy_components[
                ..., type_index, :
            ].detach().float()
            metrics.update(
                {
                    f"object_intent_{name}_route_raw_rms": raw_routes[type_index]
                    .detach()
                    .float()
                    .square()
                    .mean()
                    .sqrt(),
                    f"object_intent_{name}_common_relevance_mass": common_mass.mean(),
                    f"object_intent_{name}_common_signal_absence": (
                        1.0
                        - typed_common_signal_strength[..., type_index]
                        .detach()
                        .float()
                    ).mean(),
                    f"object_intent_{name}_common_value_rms": common_selected.square()
                    .mean()
                    .sqrt(),
                    f"object_intent_{name}_common_object_variation": common_selected.std(
                        dim=1, unbiased=False
                    ).mean(),
                    f"object_intent_{name}_interval_residual_mass": residual_mass.mean(),
                    f"object_intent_{name}_interval_residual_signal_absence": (
                        1.0
                        - typed_residual_signal_strength[..., type_index]
                        .detach()
                        .float()
                    ).mean(),
                    f"object_intent_{name}_interval_residual_value_rms": residual_selected.square()
                    .mean()
                    .sqrt(),
                    f"object_intent_{name}_interval_residual_object_variation": residual_selected.std(
                        dim=2, unbiased=False
                    ).mean(),
                    f"object_intent_{name}_interval_residual_variation": residual_selected.std(
                        dim=1, unbiased=False
                    ).mean(),
                    f"object_intent_{name}_common_policy_context_rms": common_component.square()
                    .mean()
                    .sqrt(),
                    f"object_intent_{name}_interval_residual_policy_context_rms": residual_component.square()
                    .mean()
                    .sqrt(),
                    f"object_intent_{name}_common_score_abs": typed_common_score[
                        ..., type_index
                    ]
                    .detach()
                    .abs()
                    .mean(),
                    f"object_intent_{name}_differential_score_abs": typed_differential_score[
                        ..., type_index
                    ]
                    .detach()
                    .abs()
                    .mean(),
                    f"object_intent_{name}_temperature": typed_temperature[
                        type_index
                    ].detach(),
                }
            )
        return state_out, metrics


class DirectIntentFutureSupervisor(nn.Module):
    """Decode stable physical future quantities from the online S boundary.

    No hidden recognizer coordinate is learned.  The condition-generated S
    innovation owns a direct future-state prediction, while the exact typed
    W boundary owns matching semantic, status and transport predictions.
    """

    def __init__(
        self,
        *,
        hidden: int,
        state_dim: int,
        content_dim: int,
        route_dim: int,
    ) -> None:
        super().__init__()
        del content_dim, route_dim
        self.state_head = nn.Linear(hidden, state_dim, bias=False)

    @staticmethod
    def _supported_field_loss(
        prediction: Tensor,
        target: Tensor,
        object_support: Tensor,
        *,
        scale: Tensor | None = None,
    ) -> Tensor:
        if tuple(prediction.shape) != tuple(target.shape):
            raise ValueError("typed intent prediction and target must align")
        if object_support.ndim != 3 or int(object_support.shape[-1]) != 1:
            raise ValueError("typed intent support must be [B,K,1]")
        if prediction.ndim == 3:
            mask = object_support
        elif prediction.ndim == 4:
            mask = object_support[:, None].expand(
                prediction.shape[0], prediction.shape[1], prediction.shape[2], 1
            )
        else:
            raise ValueError("typed intent field must be [B,K,D] or [B,I,K,D]")
        prediction_value = prediction.float()
        target_value = target.detach().float()
        if scale is not None:
            scale_value = scale.detach().float().clamp_min(1.0e-4)
            while scale_value.ndim < prediction_value.ndim:
                scale_value = scale_value.unsqueeze(1)
            prediction_value = prediction_value / scale_value
            target_value = target_value / scale_value
        error = F.smooth_l1_loss(
            prediction_value, target_value, reduction="none"
        ).mean(dim=-1, keepdim=True)
        return (error * mask).sum() / mask.sum().clamp_min(1.0)

    def forward(
        self,
        *,
        intent: ObjectIntentState,
        intent_boundary: FutureObjectDynamics,
        future_state: Tensor,
        teacher: FutureObjectDynamics,
        current_loss_support: Tensor,
    ) -> IntentFutureSupervision:
        if future_state.ndim != 3:
            raise ValueError("intent supervision requires a future state sequence")
        teacher.validate()
        intent_boundary.validate()
        intent.validate(
            horizon=int(intent.temporal_queries.shape[1]),
            hidden=int(intent.public_interval_carrier.shape[-1]),
        )
        length = int(future_state.shape[1])
        slices = _interval_slices(length)
        state_summary = torch.stack(
            [future_state[:, row].mean(dim=1) for row in slices], dim=1
        )
        if current_loss_support.ndim != 4 or int(current_loss_support.shape[-1]) != 1:
            raise ValueError("intent current loss support must be [B,K,C,1]")
        if int(current_loss_support.shape[0]) != int(future_state.shape[0]):
            raise ValueError("intent current loss support batch does not align")
        object_support = current_loss_support.detach().float().amax(dim=2)
        expected_support = (
            teacher.semantic_delta.shape[0],
            teacher.semantic_delta.shape[2],
            1,
        )
        if tuple(object_support.shape) != expected_support:
            raise ValueError("intent object support does not align with teacher")

        state_prediction = self.state_head(intent.interval_condition_innovation)
        # These are decoded with the exact W field projections and heads that
        # consume S online.  Independent S-only heads previously let the
        # auxiliary loss improve in a coordinate W/P never observed.
        semantic_prediction = intent_boundary.semantic_delta
        status_prediction = torch.cat(
            (intent_boundary.visibility, intent_boundary.persistence), dim=-1
        )
        transport_prediction = intent_boundary.transport_mean
        semantic_target = teacher.semantic_delta.detach()
        status_target = torch.cat(
            (teacher.visibility.detach(), teacher.persistence.detach()), dim=-1
        )
        transport_target = teacher.transport_mean.detach()
        public_loss = F.smooth_l1_loss(
            state_prediction.float(), state_summary.detach().float()
        )
        semantic_scale = teacher.current_reference.detach().float().square().mean(
            dim=-1, keepdim=True
        ).sqrt().clamp_min(0.25)
        semantic_common_loss = self._supported_field_loss(
            intent_boundary.semantic_common,
            teacher.semantic_common.detach(),
            object_support,
            scale=semantic_scale,
        )
        semantic_residual_loss = self._supported_field_loss(
            intent_boundary.semantic_interval_residual,
            teacher.semantic_interval_residual.detach(),
            object_support,
            scale=semantic_scale,
        )
        semantic_loss = 0.5 * (semantic_common_loss + semantic_residual_loss)
        status_common_loss = self._supported_field_loss(
            torch.cat(
                (intent_boundary.visibility_common, intent_boundary.persistence_common),
                dim=-1,
            ),
            torch.cat(
                (teacher.visibility_common, teacher.persistence_common), dim=-1
            ).detach(),
            object_support,
        )
        status_residual_loss = self._supported_field_loss(
            torch.cat(
                (
                    intent_boundary.visibility_interval_residual,
                    intent_boundary.persistence_interval_residual,
                ),
                dim=-1,
            ),
            torch.cat(
                (
                    teacher.visibility_interval_residual,
                    teacher.persistence_interval_residual,
                ),
                dim=-1,
            ).detach(),
            object_support,
        )
        status_loss = 0.5 * (status_common_loss + status_residual_loss)
        transport_common_loss = self._supported_field_loss(
            intent_boundary.transport_common,
            teacher.transport_common.detach(),
            object_support,
        )
        transport_residual_loss = self._supported_field_loss(
            intent_boundary.transport_interval_residual,
            teacher.transport_interval_residual.detach(),
            object_support,
        )
        transport_loss = 0.5 * (
            transport_common_loss + transport_residual_loss
        )
        typed_loss = (semantic_loss + status_loss + transport_loss) / 3.0
        result = IntentFutureSupervision(
            state_prediction=state_prediction,
            state_target=state_summary.detach(),
            semantic_prediction=semantic_prediction,
            semantic_target=semantic_target,
            status_prediction=status_prediction,
            status_target=status_target,
            transport_prediction=transport_prediction,
            transport_target=transport_target,
            public_loss=public_loss,
            semantic_loss=semantic_loss,
            semantic_common_loss=semantic_common_loss,
            semantic_residual_loss=semantic_residual_loss,
            status_loss=status_loss,
            status_common_loss=status_common_loss,
            status_residual_loss=status_residual_loss,
            transport_loss=transport_loss,
            transport_common_loss=transport_common_loss,
            transport_residual_loss=transport_residual_loss,
            typed_loss=typed_loss,
        )
        result.validate()
        return result


class CoarseActionIntent(nn.Module):
    """V120 online clean action intent used exactly once by W."""

    def __init__(
        self, *, hidden: int, action_dim: int, route_dim: int, heads: int
    ) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 4, hidden) * 0.02)
        self.intent_read = _CrossRead(hidden, heads)
        self.object_read = _CrossRead(hidden, heads)
        self.history_read = _CrossRead(hidden, heads)
        # Schema31 removes the duplicate typed CoarseAction path, but the
        # retained block and every module constructed after it must keep the
        # Schema30 initialization under the same seed.  Consume exactly the
        # random draws of the removed Linear/MHA weights as short-lived CPU
        # objects.  They are never registered, serialized, moved to CUDA or
        # executed, so this preserves controlled initialization without
        # retaining dead parameters or runtime memory.
        removed_typed_inputs = tuple(
            nn.Linear(route_dim, hidden, bias=False)
            for _ in TYPED_INTENT_NAMES
        )
        removed_typed_reads = tuple(
            nn.MultiheadAttention(
                hidden,
                heads,
                bias=False,
                dropout=0.0,
                batch_first=True,
            )
            for _ in TYPED_INTENT_NAMES
        )
        removed_typed_router = nn.Linear(
            hidden,
            len(TYPED_INTENT_NAMES),
            bias=False,
        )
        del removed_typed_inputs, removed_typed_reads, removed_typed_router
        self.block = _SelfBlock(hidden, heads)
        self.action_head = nn.Linear(hidden, action_dim, bias=False)

    def forward(
        self,
        intent: ActionIntentDock,
        *,
        future_action: Tensor | None = None,
    ) -> CoarseActionIntentState:
        intent.validate(hidden=int(self.query.shape[-1]))
        batch = int(intent.interval_condition_innovation.shape[0])
        query = self.query.to(
            device=intent.interval_condition_innovation.device,
            dtype=intent.interval_condition_innovation.dtype,
        ).expand(batch, -1, -1)
        _, intent_delta, _ = self.intent_read(
            query,
            intent.interval_condition_innovation,
            zero_preserving_innovation=True,
        )
        _, object_delta, _ = self.object_read(
            query,
            torch.cat(
                (intent.public_scene_memory, intent.object_innovation_memory),
                dim=1,
            ),
            zero_preserving_innovation=True,
        )
        _, history_delta, _ = self.history_read(
            query,
            intent.history_memory,
            zero_preserving_innovation=True,
        )
        # The learned query is an address, not a value.  The block receives
        # only public observable innovations.  Typed K/type values have one
        # owner and enter W through WorldIntentDock; rereading them here used
        # to create a second, publicized S->coarse-action->W ingress.
        token = self.block(
            intent_delta
            + object_delta
            + history_delta
        )
        action_prediction = self.action_head(token)
        if future_action is None:
            target = None
            loss = action_prediction.new_zeros(())
        else:
            slices = _interval_slices(int(future_action.shape[1]))
            target = torch.stack(
                [future_action[:, row].mean(dim=1) for row in slices], dim=1
            ).detach()
            loss = (action_prediction.float() - target.float()).square().mean()
        return CoarseActionIntentState(
            tokens=token,
            action_prediction=action_prediction,
            target=target,
            loss=loss,
        )

    @staticmethod
    def attach_training_target(
        online: CoarseActionIntentState,
        future_action: Tensor,
    ) -> CoarseActionIntentState:
        """Supervise the exact online tensor already consumed by W.

        Re-running this module solely to obtain an auxiliary loss creates a
        second autograd graph and weakens the ownership statement even when
        dropout is disabled.  Target attachment is pure bookkeeping: it does
        not rebuild or modify the online action-intent value.
        """

        if future_action.ndim != 3:
            raise ValueError("coarse action target must be [B,T,A]")
        slices = _interval_slices(int(future_action.shape[1]))
        target = torch.stack(
            [future_action[:, row].mean(dim=1) for row in slices], dim=1
        ).detach()
        if tuple(target.shape) != tuple(online.action_prediction.shape):
            raise ValueError("coarse action prediction and target do not align")
        loss = (
            online.action_prediction.float() - target.float()
        ).square().mean()
        return CoarseActionIntentState(
            tokens=online.tokens,
            action_prediction=online.action_prediction,
            target=target,
            loss=loss,
        )


__all__ = [
    "CoarseActionIntent",
    "DirectIntentFutureSupervisor",
    "StatelessObjectIntentOrganizer",
]
