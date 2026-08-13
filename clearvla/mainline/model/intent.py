"""Recovered V120 stateless intent, plan recognition and coarse action."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .routing import RoleDeltaAttnRes, smooth_rms_contract
from .types import (
    INTERVAL_BOUNDS,
    CoarseActionIntentState,
    FutureObjectDynamics,
    FuturePlanRecognition,
    ObjectFactSet,
    ObjectIntentState,
    normalized_entropy,
)


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
        ffn, _ = smooth_rms_contract(self.ffn(value), self.maximum_rms)
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
        history_width = state_dim + action_dim + state_dim + 1
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
        self.interval_semantic = _CrossRead(hidden, heads)
        self.interval_appearance = _CrossRead(hidden, heads)
        self.interval_geometry = _CrossRead(hidden, heads)
        self.interval_typed_router = RoleDeltaAttnRes(
            hidden,
            max(hidden // 8, 32),
            max_sources=3,
            include_null=True,
            max_value_rms=0.35,
            normalization_floor=0.25,
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
        state_sequence = torch.cat((state_history, state[:, None]), dim=1)
        length = max(int(state_sequence.shape[1]), int(executed_history.shape[1]))

        def left_pad(value: Tensor, target: int) -> Tensor:
            missing = target - int(value.shape[1])
            if missing <= 0:
                return value[:, -target:]
            return torch.cat((value[:, :1].expand(-1, missing, -1), value), dim=1)

        states = left_pad(state_sequence, length)
        actions = left_pad(executed_history, length)
        previous = torch.cat((states[:, :1], states[:, :-1]), dim=1)
        delta = states - previous
        offset = torch.linspace(
            -1.0, 0.0, length, device=states.device, dtype=states.dtype
        )[None, :, None].expand(states.shape[0], -1, -1)
        return torch.cat((states, actions, delta, offset), dim=-1), delta

    def _object_tokens(
        self, facts: ObjectFactSet
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        facts.validate()
        return (
            self.object_content(facts.content),
            self.object_semantic(facts.semantic),
            self.object_appearance(facts.appearance),
            self.object_geometry(facts.geometry),
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
        protected_goal, _, goal_attention = self.goal_read(
            goal_query,
            goal_memory,
            padding_mask=~goal_mask.to(device=goal_memory.device, dtype=torch.bool),
            diagnostics=collect_diagnostics,
        )
        protected_goal = self.goal_self(protected_goal)
        paired_history, observed_state_delta = self._paired_history(
            state_history, state, executed_history
        )
        history = self.history_input(paired_history)
        for block in self.history_blocks:
            history = block(history, causal=True)
        objects, semantic_objects, appearance_objects, geometry_objects = (
            self._object_tokens(facts)
        )
        interval_base = self.interval_identity.to(
            device=objects.device, dtype=objects.dtype
        ).expand(batch, -1, -1)
        _, goal_innovation, interval_goal_attention = self.interval_goal(
            interval_base, protected_goal, diagnostics=collect_diagnostics
        )
        _, history_innovation, interval_history_attention = self.interval_history(
            interval_base, history, diagnostics=collect_diagnostics
        )
        _, object_innovation, interval_object_attention = self.interval_object(
            interval_base, objects, diagnostics=collect_diagnostics
        )
        _, semantic_innovation, interval_semantic_attention = self.interval_semantic(
            interval_base, semantic_objects, diagnostics=collect_diagnostics
        )
        _, appearance_innovation, interval_appearance_attention = self.interval_appearance(
            interval_base, appearance_objects, diagnostics=collect_diagnostics
        )
        _, geometry_innovation, interval_geometry_attention = self.interval_geometry(
            interval_base, geometry_objects, diagnostics=collect_diagnostics
        )
        typed_innovation, typed_route_metrics = self.interval_typed_router(
            interval_base + goal_innovation + history_innovation,
            torch.stack(
                (semantic_innovation, appearance_innovation, geometry_innovation),
                dim=-2,
            ),
            collect_diagnostics=collect_diagnostics,
        )
        intervals = self.interval_self(
            interval_base
            + goal_innovation
            + history_innovation
            + object_innovation
            + typed_innovation
        )
        temporal_base = self.temporal_identity.to(
            device=intervals.device, dtype=intervals.dtype
        ).expand(batch, -1, -1)
        temporal, _, _ = self.temporal_read(
            temporal_base, intervals, diagnostics=False
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
            object_tokens=objects,
            semantic_object_tokens=semantic_objects,
            appearance_object_tokens=appearance_objects,
            geometry_object_tokens=geometry_objects,
            interval_queries=intervals,
            temporal_queries=temporal,
            state_change_evidence=state_change_evidence,
            goal_attention=goal_attention,
            interval_goal_attention=interval_goal_attention,
            interval_history_attention=interval_history_attention,
            interval_object_attention=interval_object_attention,
            interval_semantic_attention=interval_semantic_attention,
            interval_appearance_attention=interval_appearance_attention,
            interval_geometry_attention=interval_geometry_attention,
        )
        state_out.validate(horizon=self.horizon, hidden=self.hidden)
        if not collect_diagnostics:
            return state_out, {}
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
            "object_intent_interval_semantic_entropy": normalized_entropy(
                interval_semantic_attention, dim=-1
            ).detach().mean(),
            "object_intent_interval_appearance_entropy": normalized_entropy(
                interval_appearance_attention, dim=-1
            ).detach().mean(),
            "object_intent_interval_geometry_entropy": normalized_entropy(
                interval_geometry_attention, dim=-1
            ).detach().mean(),
            "object_intent_interval_variation": intervals.detach().float().std(
                dim=1, unbiased=False
            ).mean(),
            "object_intent_temporal_variation": temporal.detach().float().std(
                dim=1, unbiased=False
            ).mean(),
            "object_intent_goal_innovation_rms": goal_innovation.detach().float().square().mean().sqrt(),
            "object_intent_history_innovation_rms": history_innovation.detach().float().square().mean().sqrt(),
            "object_intent_object_innovation_rms": object_innovation.detach().float().square().mean().sqrt(),
            "object_intent_typed_innovation_rms": typed_innovation.detach().float().square().mean().sqrt(),
            "object_intent_observed_state_delta_rms": observed_state_delta.detach().float().square().mean().sqrt(),
            "object_intent_observed_transport_rms": facts.transport_prior.detach().float().square().mean().sqrt(),
            "object_intent_state_change_history_rms": state_change_history.detach().float().square().mean().sqrt(),
            "object_intent_state_change_transport_rms": state_change_transport.detach().float().square().mean().sqrt(),
            "object_intent_state_change_evidence_rms": state_change_evidence.detach().float().square().mean().sqrt(),
            "object_intent_state_change_attention_entropy": normalized_entropy(
                state_change_attention, dim=-1
            ).detach().mean(),
        }
        for key, value in typed_route_metrics.items():
            metrics[f"object_intent_typed_{key}"] = value
        return state_out, metrics


class FuturePlanRecognizer(nn.Module):
    """Training-only V120 whole-segment posterior."""

    def __init__(
        self,
        *,
        hidden: int,
        action_dim: int,
        state_dim: int,
        content_dim: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.content_dim = int(content_dim)
        self.action_input = nn.Linear(action_dim, hidden, bias=False)
        self.state_input = nn.Linear(state_dim, hidden, bias=False)
        self.effect_input = nn.Linear(content_dim, hidden, bias=False)
        self.interval_identity = nn.Parameter(torch.randn(1, 4, hidden) * 0.02)
        self.block = _SelfBlock(hidden, heads)
        self.action_reconstruction = nn.Linear(hidden, action_dim, bias=False)
        self.state_reconstruction = nn.Linear(hidden, state_dim, bias=False)
        self.effect_reconstruction = nn.Linear(hidden, content_dim, bias=False)

    def forward(
        self,
        *,
        future_action: Tensor,
        future_state: Tensor,
        teacher: FutureObjectDynamics | None,
        current_loss_support: Tensor,
    ) -> FuturePlanRecognition:
        if future_action.ndim != 3 or future_state.ndim != 3:
            raise ValueError("plan recognizer requires full future action/state sequences")
        length = min(int(future_action.shape[1]), int(future_state.shape[1]))
        slices = _interval_slices(length)
        action_summary = torch.stack(
            [future_action[:, row].mean(dim=1) for row in slices], dim=1
        )
        state_summary = torch.stack(
            [future_state[:, row].mean(dim=1) for row in slices], dim=1
        )
        if current_loss_support.ndim != 4 or int(current_loss_support.shape[-1]) != 1:
            raise ValueError("recognizer current loss support must be [B,K,C,1]")
        if int(current_loss_support.shape[0]) != int(future_action.shape[0]):
            raise ValueError("recognizer current loss support batch does not align")
        # This is the same detached current-fact support used by the object
        # losses.  Future reliability and selector validity are deliberately
        # absent: neither may shrink a supervised target or create a routing
        # shortcut.  A camera reduction is performed exactly once because W
        # exports object-level future geometry/content.
        object_support = current_loss_support.detach().float().amax(dim=2)
        if teacher is None:
            effect_summary = future_action.new_zeros(
                future_action.shape[0], 4, self.content_dim
            )
            teacher_valid = future_action.new_zeros(future_action.shape[0], 4, 1)
        else:
            teacher.validate()
            expected_support = (
                teacher.semantic_delta.shape[0],
                teacher.semantic_delta.shape[2],
                1,
            )
            if tuple(object_support.shape) != expected_support:
                raise ValueError("recognizer object support does not align with teacher")
            support = object_support[:, None]
            denominator = support.sum(dim=2).clamp_min(1.0)
            effect_summary = (
                teacher.semantic_delta.detach().float() * support
            ).sum(dim=2) / denominator
            effect_summary = effect_summary.to(dtype=teacher.semantic_delta.dtype)
            teacher_valid = (support.sum(dim=2) > 0).to(
                dtype=teacher.semantic_delta.dtype
            ).expand(-1, teacher.semantic_delta.shape[1], -1)
        token = (
            self.action_input(action_summary)
            + self.state_input(state_summary)
            + self.effect_input(effect_summary)
            + self.interval_identity.to(
                device=future_action.device, dtype=future_action.dtype
            )
        )
        token = self.block(token)
        action_pred = self.action_reconstruction(token)
        state_pred = self.state_reconstruction(token)
        effect_pred = self.effect_reconstruction(token)
        effect_error = (
            (effect_pred.float() - effect_summary.detach().float()).square()
            * teacher_valid.detach().float()
        ).sum() / teacher_valid.detach().float().sum().clamp_min(1.0) / float(
            self.content_dim
        )
        reconstruction = (
            (action_pred.float() - action_summary.detach().float()).square().mean()
            + (state_pred.float() - state_summary.detach().float()).square().mean()
            + 0.25 * effect_error
        )
        result = FuturePlanRecognition(
            interval_targets=token.detach(),
            action_summary=action_summary.detach(),
            state_summary=state_summary.detach(),
            effect_summary=effect_summary.detach(),
            reconstruction_loss=reconstruction,
        )
        result.validate(hidden=self.hidden)
        return result


class CoarseActionIntent(nn.Module):
    """V120 online clean action intent used exactly once by W."""

    def __init__(self, *, hidden: int, action_dim: int, heads: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 4, hidden) * 0.02)
        self.intent_read = _CrossRead(hidden, heads)
        self.object_read = _CrossRead(hidden, heads)
        self.semantic_read = _CrossRead(hidden, heads)
        self.appearance_read = _CrossRead(hidden, heads)
        self.geometry_read = _CrossRead(hidden, heads)
        self.typed_router = RoleDeltaAttnRes(
            hidden,
            max(hidden // 8, 32),
            max_sources=3,
            include_null=True,
            max_value_rms=0.35,
            normalization_floor=0.25,
        )
        self.history_read = _CrossRead(hidden, heads)
        self.block = _SelfBlock(hidden, heads)
        self.action_head = nn.Linear(hidden, action_dim, bias=False)

    def forward(
        self,
        intent: ObjectIntentState,
        *,
        future_action: Tensor | None = None,
    ) -> CoarseActionIntentState:
        batch = int(intent.interval_queries.shape[0])
        query = self.query.to(
            device=intent.interval_queries.device,
            dtype=intent.interval_queries.dtype,
        ).expand(batch, -1, -1)
        _, intent_delta, _ = self.intent_read(query, intent.interval_queries)
        _, object_delta, _ = self.object_read(query, intent.object_tokens)
        _, semantic_delta, _ = self.semantic_read(query, intent.semantic_object_tokens)
        _, appearance_delta, _ = self.appearance_read(
            query, intent.appearance_object_tokens
        )
        _, geometry_delta, _ = self.geometry_read(query, intent.geometry_object_tokens)
        typed_delta, _ = self.typed_router(
            query + intent_delta + object_delta,
            torch.stack((semantic_delta, appearance_delta, geometry_delta), dim=-2),
            collect_diagnostics=False,
        )
        _, history_delta, _ = self.history_read(query, intent.history_tokens)
        token = self.block(
            query + intent_delta + object_delta + typed_delta + history_delta
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


__all__ = [
    "CoarseActionIntent",
    "FuturePlanRecognizer",
    "StatelessObjectIntentOrganizer",
]
