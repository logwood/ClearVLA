"""Online stateless intent, training-only plan recognition and coarse action."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .routing import (
    RoleDeltaAttnRes,
    VarianceFlooredCenteredNorm,
    rms_floored_l2_normalize,
    smooth_rms_contract,
)
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
    return torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)


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
        self.query_norm = VarianceFlooredCenteredNorm(0.25)
        self.memory_norm = VarianceFlooredCenteredNorm(0.25)
        self.attention = nn.MultiheadAttention(
            hidden, heads, bias=False, dropout=0.0, batch_first=True
        )
        self.ffn = nn.Sequential(
            VarianceFlooredCenteredNorm(0.25),
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
        memory_value: Tensor | None = None,
        padding_mask: Tensor | None = None,
        diagnostics: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        value = memory if memory_value is None else memory_value
        if tuple(value.shape) != tuple(memory.shape):
            raise ValueError("cross-read key and value memories must align")
        update, weights = self.attention(
            self.query_norm(query),
            self.memory_norm(memory),
            self.memory_norm(value),
            key_padding_mask=padding_mask,
            need_weights=diagnostics,
            average_attn_weights=True,
        )
        update, _ = smooth_rms_contract(update, self.maximum_rms)
        # The learned query is an address, not a value.  Feed only the actual
        # attention update through the bias-free FFN so an empty/zero memory
        # cannot manufacture an innovation from query identity alone.
        ffn, _ = smooth_rms_contract(self.ffn(update), self.maximum_rms)
        value = query + update
        value = value + ffn
        if weights is None:
            weights = query.new_zeros(query.shape[0], query.shape[1], memory.shape[1])
        return value, update + ffn, weights


class _SelfBlock(nn.Module):
    def __init__(self, hidden: int, heads: int) -> None:
        super().__init__()
        self.norm = VarianceFlooredCenteredNorm(0.25)
        self.attention = nn.MultiheadAttention(
            hidden, heads, bias=False, dropout=0.0, batch_first=True
        )
        self.ffn = nn.Sequential(
            VarianceFlooredCenteredNorm(0.25),
            nn.Linear(hidden, 2 * hidden, bias=False),
            nn.GELU(),
            nn.Linear(2 * hidden, hidden, bias=False),
        )

    def forward(self, value: Tensor, *, causal: bool = False) -> Tensor:
        full, _ = self.forward_with_innovation(value, causal=causal)
        return full

    def forward_with_innovation(
        self,
        value: Tensor,
        *,
        causal: bool = False,
        value_innovation: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Use cumulative state for Q/K and innovation only for V/FFN.

        ``value_innovation=None`` retains the ordinary transformer behavior
        for data-bearing sequences such as observed history.  Supplying an
        explicit innovation gives query identities strict zero-value
        semantics while retaining their indexing role in Q/K.
        """

        source = value if value_innovation is None else value_innovation
        normalized = self.norm(value)
        update, _ = self.attention(
            normalized,
            normalized,
            self.norm(source),
            attn_mask=_causal_mask(int(value.shape[1]), value.device) if causal else None,
            need_weights=False,
        )
        update, _ = smooth_rms_contract(update, 0.35)
        value = value + update
        ffn, _ = smooth_rms_contract(self.ffn(source + update), 0.35)
        value = value + ffn
        return value, update + ffn


class StatelessObjectIntentOrganizer(nn.Module):
    """Observable intent without scalar progress or synthetic phase labels."""

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
        history_width = state_dim + action_dim + state_dim
        self.history_input = nn.Sequential(
            nn.LayerNorm(history_width, elementwise_affine=False),
            nn.Linear(history_width, hidden, bias=False),
        )
        # Relative history position is an address, never an observable value.
        # Keeping it outside ``history_input`` prevents a fixed [-1,0] ramp
        # from synthesizing an apparent progress signal when state/action
        # evidence is constant.
        self.history_position = nn.Linear(1, hidden, bias=False)
        self.history_blocks = nn.ModuleList(_SelfBlock(hidden, heads) for _ in range(2))
        self.object_content = nn.Linear(content_dim, hidden, bias=False)
        self.object_semantic = nn.Linear(route_dim, hidden, bias=False)
        self.object_appearance = nn.Linear(route_dim, hidden, bias=False)
        self.object_geometry = nn.Linear(route_dim, hidden, bias=False)
        self.interval_identity = nn.Parameter(torch.randn(1, 4, hidden) * 0.02)
        self.interval_goal = _CrossRead(hidden, heads)
        self.interval_history = _CrossRead(hidden, heads)
        self.interval_typed_router = RoleDeltaAttnRes(
            hidden,
            max(hidden // 8, 32),
            max_sources=3,
            include_null=True,
            max_value_rms=0.35,
            normalization_floor=0.25,
        )
        self.interval_self = _SelfBlock(hidden, heads)
        self.interval_state_self = _SelfBlock(hidden, heads)
        self.interval_object_key = nn.Sequential(
            nn.LayerNorm(hidden, elementwise_affine=False),
            nn.Linear(hidden, 2 * hidden, bias=False),
            nn.SiLU(),
            nn.Linear(2 * hidden, hidden, bias=False),
        )
        self.interval_object_value = nn.Sequential(
            nn.LayerNorm(hidden, elementwise_affine=False),
            nn.Linear(hidden, 2 * hidden, bias=False),
            nn.SiLU(),
            nn.Linear(2 * hidden, hidden, bias=False),
        )
        self.temporal_identity = nn.Parameter(torch.randn(1, horizon, hidden) * 0.02)
        self.temporal_read = _CrossRead(hidden, heads)
        # This branch is deliberately value-zero-preserving.  The learned
        # query and absolute history may choose which observed delta matters,
        # but they cannot manufacture a value when every delta is zero.
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
    ) -> tuple[Tensor, Tensor, Tensor]:
        if state_history.ndim != 3 or state.ndim != 2 or executed_history.ndim != 3:
            raise ValueError("intent history requires state/action sequences")
        if int(state_history.shape[1]) < 1:
            raise ValueError("state history must contain one current causal row")
        # The configured offsets end in zero, so the final cached row already
        # denotes the present.  Replace it with the canonical online state;
        # appending would count the current observation twice.
        state_sequence = torch.cat((state_history[:, :-1], state[:, None]), dim=1)
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
        offset = torch.linspace(-1.0, 0.0, length, device=states.device, dtype=states.dtype)[
            None, :, None
        ].expand(states.shape[0], -1, -1)
        return torch.cat((states, actions, delta), dim=-1), delta, offset

    def _object_tokens(self, facts: ObjectFactSet) -> tuple[Tensor, Tensor, Tensor, Tensor]:
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
        goal_read_value, goal_read_innovation, goal_attention = self.goal_read(
            goal_query,
            goal_memory,
            padding_mask=~goal_mask.to(device=goal_memory.device, dtype=torch.bool),
            diagnostics=collect_diagnostics,
        )
        protected_goal, goal_block_innovation = self.goal_self.forward_with_innovation(
            goal_read_value,
            value_innovation=goal_read_innovation,
        )
        protected_goal_innovation = goal_read_innovation + goal_block_innovation
        paired_history, observed_state_delta, history_offset = self._paired_history(
            state_history, state, executed_history
        )
        history_innovation = self.history_input(paired_history)
        history = history_innovation + self.history_position(history_offset)
        for raw_block in self.history_blocks:
            block = cast(_SelfBlock, raw_block)
            history, block_innovation = block.forward_with_innovation(
                history,
                causal=True,
                value_innovation=history_innovation,
            )
            history_innovation = history_innovation + block_innovation
        (
            objects,
            semantic_objects,
            appearance_objects,
            geometry_objects,
        ) = self._object_tokens(facts)
        interval_base = self.interval_identity.to(
            device=objects.device, dtype=objects.dtype
        ).expand(batch, -1, -1)
        goal_value, goal_innovation, interval_goal_attention = self.interval_goal(
            interval_base,
            protected_goal_innovation,
            diagnostics=collect_diagnostics,
        )
        history_value, history_innovation, interval_history_attention = self.interval_history(
            interval_base,
            history,
            memory_value=history_innovation,
            diagnostics=collect_diagnostics,
        )
        object_base = objects[:, None] + interval_base[:, :, None]
        typed_query = object_base
        typed_innovation, typed_route_metrics = self.interval_typed_router(
            typed_query,
            torch.stack(
                (
                    semantic_objects[:, None].expand(-1, 4, -1, -1),
                    appearance_objects[:, None].expand(-1, 4, -1, -1),
                    geometry_objects[:, None].expand(-1, 4, -1, -1),
                ),
                dim=-2,
            ),
            collect_diagnostics=collect_diagnostics,
        )
        # The canonical P1 query owns goal/history exactly once.  Global K
        # object evidence is not pooled into this query and then reintroduced
        # in W under another name.
        action_input = interval_base + goal_innovation + history_innovation
        (
            interval_action_queries,
            action_block_innovation,
        ) = self.interval_self.forward_with_innovation(
            action_input,
            causal=True,
            value_innovation=goal_innovation + history_innovation,
        )
        interval_action_innovations = goal_innovation + history_innovation + action_block_innovation
        state_input = interval_base + history_innovation
        (
            interval_state_queries,
            state_block_innovation,
        ) = self.interval_state_self.forward_with_innovation(
            state_input,
            causal=True,
            value_innovation=history_innovation,
        )
        interval_state_innovations = history_innovation + state_block_innovation
        # Object keys answer *which fact is relevant to the intended action*;
        # object values answer *what observable state that fact currently
        # carries*.  Summing action and state before tanh duplicated history,
        # coupled their gradients and encouraged one common interval context.
        # Keep the two zero-centred operands factorized and equally bounded.
        object_action_context, object_action_context_scale = smooth_rms_contract(
            interval_action_innovations,
            1.0,
        )
        object_state_context, object_state_context_scale = smooth_rms_contract(
            interval_state_innovations,
            1.0,
        )
        object_key_seed = semantic_objects[:, None] * torch.tanh(
            object_action_context[:, :, None]
        )
        object_value_seed = typed_innovation * torch.tanh(object_state_context[:, :, None])
        interval_object_keys = self.interval_object_key(object_key_seed)
        interval_object_values = self.interval_object_value(object_value_seed)
        intervals = interval_action_queries
        temporal_base = self.temporal_identity.to(
            device=intervals.device, dtype=intervals.dtype
        ).expand(batch, -1, -1)
        temporal, temporal_innovation, _ = self.temporal_read(
            temporal_base, interval_action_innovations, diagnostics=False
        )
        state_change_values = self.state_change_input(observed_state_delta)
        state_change_history, state_change_attention = self.state_change_read(
            self.state_change_query_norm(
                self.state_change_query.to(device=history.device, dtype=history.dtype).expand(
                    batch, -1, -1
                )
            ),
            self.state_change_key_norm(history),
            state_change_values,
            need_weights=collect_diagnostics,
            average_attn_weights=True,
        )
        if state_change_attention is None:
            state_change_attention = history.new_zeros(batch, 1, history.shape[1])
        state_change_history = state_change_history[:, 0]
        transport_tokens = self.state_change_transport(facts.camera_transport_prior)
        transport_validity = facts.camera_validity.to(
            device=transport_tokens.device, dtype=transport_tokens.dtype
        )
        state_change_transport = (transport_tokens * transport_validity).sum(
            dim=(1, 2)
        ) / transport_validity.sum(dim=(1, 2)).clamp_min(1.0)
        state_change_evidence, _ = smooth_rms_contract(
            self.state_change_fuse(
                torch.cat((state_change_history, state_change_transport), dim=-1)
            ),
            0.20,
        )
        state_out = ObjectIntentState(
            interval_queries=intervals,
            interval_action_innovations=interval_action_innovations,
            interval_state_innovations=interval_state_innovations,
            interval_object_keys=interval_object_keys,
            interval_object_values=interval_object_values,
            temporal_queries=temporal,
            temporal_innovations=temporal_innovation,
            state_change_evidence=state_change_evidence,
        )
        state_out.validate(horizon=self.horizon, hidden=self.hidden)
        if not collect_diagnostics:
            return state_out, {}
        # Read-only diagnostics for object/type selectivity.  They are built
        # only when emitted and never retained in the deployment cache.
        normalized_interval = F.normalize(intervals.detach().float(), dim=-1, eps=1e-4)
        normalized_object = F.normalize(interval_object_keys.detach().float(), dim=-1, eps=1e-4)
        interval_object_attention = torch.softmax(
            torch.einsum("bih,bikh->bik", normalized_interval, normalized_object),
            dim=-1,
        )

        def typed_attention(value: Tensor) -> Tensor:
            normalized = F.normalize(value.detach().float(), dim=-1, eps=1e-4)
            return torch.softmax(
                torch.einsum("bih,bkh->bik", normalized_interval, normalized),
                dim=-1,
            )

        interval_semantic_attention = typed_attention(semantic_objects)
        interval_appearance_attention = typed_attention(appearance_objects)
        interval_geometry_attention = typed_attention(geometry_objects)
        metrics: dict[str, Tensor] = {
            "object_intent_goal_attention_entropy": normalized_entropy(goal_attention, dim=-1)
            .detach()
            .mean(),
            "object_intent_interval_goal_entropy": normalized_entropy(
                interval_goal_attention, dim=-1
            )
            .detach()
            .mean(),
            "object_intent_interval_history_entropy": normalized_entropy(
                interval_history_attention, dim=-1
            )
            .detach()
            .mean(),
            "object_intent_interval_object_audit_similarity_entropy": normalized_entropy(
                interval_object_attention, dim=-1
            )
            .detach()
            .mean(),
            "object_intent_interval_semantic_audit_similarity_entropy": normalized_entropy(
                interval_semantic_attention, dim=-1
            )
            .detach()
            .mean(),
            "object_intent_interval_appearance_audit_similarity_entropy": normalized_entropy(
                interval_appearance_attention, dim=-1
            )
            .detach()
            .mean(),
            "object_intent_interval_geometry_audit_similarity_entropy": normalized_entropy(
                interval_geometry_attention, dim=-1
            )
            .detach()
            .mean(),
            "object_intent_interval_variation": intervals.detach()
            .float()
            .std(dim=1, unbiased=False)
            .mean(),
            "object_intent_interval_state_variation": (
                interval_state_innovations.detach().float().std(dim=1, unbiased=False).mean()
            ),
            "object_intent_interval_object_key_variation": (
                interval_object_keys.detach().float().std(dim=1, unbiased=False).mean()
            ),
            "object_intent_interval_object_value_variation": (
                interval_object_values.detach().float().std(dim=1, unbiased=False).mean()
            ),
            "object_intent_temporal_variation": temporal_innovation.detach()
            .float()
            .std(dim=1, unbiased=False)
            .mean(),
            "object_intent_action_innovation_rms": interval_action_innovations.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_intent_state_innovation_rms": interval_state_innovations.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_intent_object_action_context_contract_scale": (
                object_action_context_scale.detach().float().mean()
            ),
            "object_intent_object_state_context_contract_scale": (
                object_state_context_scale.detach().float().mean()
            ),
            "object_intent_temporal_innovation_rms": temporal_innovation.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_intent_goal_innovation_rms": goal_innovation.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_intent_history_innovation_rms": history_innovation.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_intent_object_innovation_rms": (
                interval_object_values.detach().float().square().mean().sqrt()
            ),
            "object_intent_typed_innovation_rms": typed_innovation.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_intent_observed_state_delta_rms": observed_state_delta.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_intent_observed_transport_rms": facts.camera_transport_prior.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_intent_state_change_history_rms": state_change_history.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_intent_state_change_transport_rms": state_change_transport.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_intent_state_change_evidence_rms": state_change_evidence.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_intent_state_change_attention_entropy": normalized_entropy(
                state_change_attention, dim=-1
            )
            .detach()
            .mean(),
        }
        for index in range(4):
            row = f"object_intent_interval_{index}"
            metrics[f"{row}_action_innovation_rms"] = (
                interval_action_innovations[:, index].detach().float().square().mean().sqrt()
            )
            metrics[f"{row}_state_innovation_rms"] = (
                interval_state_innovations[:, index].detach().float().square().mean().sqrt()
            )
            metrics[f"{row}_object_key_rms"] = (
                interval_object_keys[:, index].detach().float().square().mean().sqrt()
            )
            metrics[f"{row}_object_value_rms"] = (
                interval_object_values[:, index].detach().float().square().mean().sqrt()
            )
        typed_source_keys = {f"source_{index}_mass" for index in range(3)}
        for key, value in typed_route_metrics.items():
            if key in typed_source_keys:
                continue
            metrics[f"object_intent_typed_{key}"] = value
        for index, name in enumerate(("semantic", "appearance", "geometry")):
            metrics[f"object_intent_typed_{name}_source_mass"] = typed_route_metrics[
                f"source_{index}_mass"
            ]
        # Keep lints honest: the value paths are intentionally not combined.
        del goal_value, history_value
        return state_out, metrics


class FuturePlanRecognizer(nn.Module):
    """Training-only factorized posterior, never an online action value.

    Future action order, state endpoints and global-K object effects remain
    separate.  The online organizer is supervised at those four boundaries;
    no temporal/object mean is added into a single hidden label.
    """

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
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.content_dim = int(content_dim)
        self.action_input = nn.Linear(action_dim, hidden, bias=False)
        self.action_key = nn.Linear(hidden, hidden, bias=False)
        self.action_value = nn.Linear(hidden, hidden, bias=False)
        self.action_queries = nn.Parameter(torch.randn(1, 4, hidden) * 0.02)
        self.action_time = nn.Linear(2, hidden, bias=False)
        self.action_block = _SelfBlock(hidden, heads)
        self.state_input = nn.Linear(3 * state_dim, hidden, bias=False)
        self.state_block = _SelfBlock(hidden, heads)
        self.effect_key_input = nn.Linear(2 * content_dim + 5, hidden, bias=False)
        self.effect_value_input = nn.Linear(2 * content_dim + 2, hidden, bias=False)
        self.effect_key_block = _SelfBlock(hidden, heads)
        self.effect_value_block = _SelfBlock(hidden, heads)
        self.interval_identity = nn.Parameter(torch.randn(1, 4, hidden) * 0.02)
        self.action_reconstruction = nn.Linear(hidden, 3 * action_dim, bias=False)
        self.state_reconstruction = nn.Linear(hidden, 3 * state_dim, bias=False)
        self.effect_key_reconstruction = nn.Linear(hidden, 2 * content_dim + 5, bias=False)
        self.effect_value_reconstruction = nn.Linear(hidden, 2 * content_dim + 2, bias=False)

    @staticmethod
    def _endpoint_summary(value: Tensor, slices: tuple[slice, ...]) -> Tensor:
        rows: list[Tensor] = []
        for row in slices:
            segment = value[:, row]
            start = segment[:, 0]
            end = segment[:, -1]
            rows.append(torch.cat((start, end, end - start), dim=-1))
        return torch.stack(rows, dim=1)

    def _ordered_action_tokens(
        self,
        action: Tensor,
        slices: tuple[slice, ...],
    ) -> tuple[Tensor, Tensor]:
        batch, length = action.shape[:2]
        position = torch.linspace(
            0.0,
            1.0,
            length,
            device=action.device,
            dtype=action.dtype,
        )
        time_feature = torch.stack((2.0 * position - 1.0, position.square()), dim=-1)
        sequence = self.action_input(action) + self.action_time(time_feature)[None]
        query = self.action_queries.to(device=action.device, dtype=sequence.dtype).expand(
            batch, -1, -1
        )
        key, _ = rms_floored_l2_normalize(self.action_key(sequence), 0.25, dim=-1)
        normalized_query, _ = rms_floored_l2_normalize(query, 0.25, dim=-1)
        score = torch.einsum("bih,blh->bil", normalized_query, key)
        legal = torch.zeros(4, length, device=action.device, dtype=torch.bool)
        for index, row in enumerate(slices):
            legal[index, row] = True
        score = score.masked_fill(~legal[None], -1.0e4)
        weight = torch.softmax(score, dim=-1)
        update = torch.einsum(
            "bil,blh->bih", weight.to(dtype=sequence.dtype), self.action_value(sequence)
        )
        full, block_innovation = self.action_block.forward_with_innovation(
            query + update,
            causal=True,
            value_innovation=update,
        )
        return full, update + block_innovation

    @staticmethod
    def _masked_mse(
        prediction: Tensor,
        target: Tensor,
        weight: Tensor,
    ) -> Tensor:
        error = (prediction.float() - target.detach().float()).square()
        expanded_weight = weight.detach().float()
        while expanded_weight.ndim < error.ndim:
            expanded_weight = expanded_weight.unsqueeze(-1)
        expanded_weight = expanded_weight.expand_as(error)
        return (error * expanded_weight).sum() / expanded_weight.sum().clamp_min(1.0)

    def forward(
        self,
        *,
        future_action: Tensor,
        future_state: Tensor,
        teacher: FutureObjectDynamics | None,
    ) -> FuturePlanRecognition:
        if future_action.ndim != 3 or future_state.ndim != 3:
            raise ValueError("plan recognizer requires full future action/state sequences")
        length = min(int(future_action.shape[1]), int(future_state.shape[1]))
        future_action = future_action[:, :length]
        future_state = future_state[:, :length]
        slices = _interval_slices(length)
        action_summary = self._endpoint_summary(future_action, slices)
        state_summary = self._endpoint_summary(future_state, slices)
        action_token, action_innovation = self._ordered_action_tokens(future_action, slices)
        interval_identity = self.interval_identity.to(
            device=future_action.device, dtype=action_token.dtype
        )
        state_input = self.state_input(state_summary)
        state_token, state_block_innovation = self.state_block.forward_with_innovation(
            state_input + interval_identity,
            causal=True,
            value_innovation=state_input,
        )
        state_innovation = state_input + state_block_innovation
        if teacher is None:
            effect_summary = future_action.new_zeros(
                future_action.shape[0], 4, 4, 2 * self.content_dim + 2
            )
            effect_key_summary = future_action.new_zeros(
                future_action.shape[0], 4, 4, 2 * self.content_dim + 5
            )
            teacher_valid = future_action.new_zeros(future_action.shape[0], 4, 4, 1)
        else:
            teacher.validate()
            camera_weight = teacher.validity.detach().float()
            camera_denominator = camera_weight.sum(dim=3).clamp_min(1e-6)
            transport_summary = (teacher.transport_mean.detach().float() * camera_weight).sum(
                dim=3
            ) / camera_denominator
            covariance_summary = (
                teacher.transport_covariance.detach().float() * camera_weight
            ).sum(dim=3) / camera_denominator
            successor_innovation = teacher.successor_content - teacher.current_reference[:, None]
            effect_summary = torch.cat(
                (
                    successor_innovation,
                    teacher.semantic_delta,
                    transport_summary,
                ),
                dim=-1,
            ).detach()
            effect_key_summary = torch.cat(
                (
                    successor_innovation,
                    teacher.semantic_delta,
                    transport_summary,
                    covariance_summary,
                ),
                dim=-1,
            ).detach()
            # Teacher-G already confidence-blends ambiguous associations to
            # current/zero-effect targets.  Multiplying the recognizer target
            # by reliability again made the S object lane weakest exactly on
            # the common neutral rows that it must learn to represent.  Keep
            # physical object validity here; unreliable geometry has already
            # been zeroed in the target fields themselves.
            teacher_valid = teacher.validity.detach().float().amax(dim=3)
        batch, intervals, objects = effect_summary.shape[:3]
        object_identity = interval_identity[:, :, None].expand(batch, -1, objects, -1)
        object_key_input = self.effect_key_input(effect_key_summary)
        object_value_input = self.effect_value_input(effect_summary)
        object_key, object_key_block_innovation = self.effect_key_block.forward_with_innovation(
            (object_key_input + object_identity)
            .transpose(1, 2)
            .reshape(batch * objects, intervals, self.hidden),
            causal=True,
            value_innovation=object_key_input.transpose(1, 2).reshape(
                batch * objects, intervals, self.hidden
            ),
        )
        object_value, object_value_block_innovation = (
            self.effect_value_block.forward_with_innovation(
                (object_value_input + object_identity)
                .transpose(1, 2)
                .reshape(batch * objects, intervals, self.hidden),
                causal=True,
                value_innovation=object_value_input.transpose(1, 2).reshape(
                    batch * objects, intervals, self.hidden
                ),
            )
        )
        object_key_innovation = (
            (
                object_key_input.transpose(1, 2).reshape(batch * objects, intervals, self.hidden)
                + object_key_block_innovation
            )
            .reshape(batch, objects, intervals, self.hidden)
            .transpose(1, 2)
        )
        object_value_innovation = (
            (
                object_value_input.transpose(1, 2).reshape(batch * objects, intervals, self.hidden)
                + object_value_block_innovation
            )
            .reshape(batch, objects, intervals, self.hidden)
            .transpose(1, 2)
        )
        # Full recognizer queries are private diagnostic carriers.  Online S
        # is matched only to data-dependent innovations, so four learned
        # interval identities cannot satisfy the objective by themselves.
        action_pred = self.action_reconstruction(action_innovation)
        state_pred = self.state_reconstruction(state_innovation)
        effect_key_pred = self.effect_key_reconstruction(object_key_innovation)
        effect_value_pred = self.effect_value_reconstruction(object_value_innovation)
        effect_error = 0.5 * (
            self._masked_mse(effect_key_pred, effect_key_summary, teacher_valid)
            + self._masked_mse(effect_value_pred, effect_summary, teacher_valid)
        )
        reconstruction = (
            (action_pred.float() - action_summary.detach().float()).square().mean()
            + (state_pred.float() - state_summary.detach().float()).square().mean()
            + 0.25 * effect_error
        )
        result = FuturePlanRecognition(
            action_targets=action_innovation.detach(),
            state_targets=state_innovation.detach(),
            object_key_targets=object_key_innovation.detach(),
            object_value_targets=object_value_innovation.detach(),
            action_summary=action_summary.detach(),
            state_summary=state_summary.detach(),
            effect_summary=effect_summary.detach(),
            object_validity=teacher_valid.detach(),
            reconstruction_loss=reconstruction,
        )
        result.validate(hidden=self.hidden)
        return result


class CoarseActionIntent(nn.Module):
    """The sole online action-conditioned input to W.

    It is predicted from the factorized online S state; it never re-reads raw
    goal/history/G aliases and never reads the denoising proposal or noisy
    action.
    """

    def __init__(
        self,
        *,
        hidden: int,
        action_dim: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.query = nn.Parameter(torch.randn(1, 4, hidden) * 0.02)
        self.intent_read = _CrossRead(hidden, heads)
        self.state_read = _CrossRead(hidden, heads)
        self.object_read = _CrossRead(hidden, heads)
        self.object_key_read = _CrossRead(hidden, heads)
        self.typed_router = RoleDeltaAttnRes(
            hidden,
            max(hidden // 8, 32),
            max_sources=3,
            include_null=True,
            max_value_rms=0.35,
            normalization_floor=0.25,
        )
        self.block = _SelfBlock(hidden, heads)
        self.action_head = nn.Linear(hidden, 3 * action_dim, bias=False)

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
        _, intent_delta, _ = self.intent_read(query, intent.interval_action_innovations)
        _, state_delta, _ = self.state_read(query, intent.interval_state_innovations)
        interval_query = query.reshape(batch * 4, 1, -1)
        object_memory = intent.interval_object_values.reshape(
            batch * 4, intent.interval_object_values.shape[2], -1
        )
        object_key_memory = intent.interval_object_keys.reshape(
            batch * 4, intent.interval_object_keys.shape[2], -1
        )
        _, object_delta, _ = self.object_read(interval_query, object_memory)
        _, object_key_delta, _ = self.object_key_read(interval_query, object_key_memory)
        object_delta = object_delta.reshape(batch, 4, -1)
        object_key_delta = object_key_delta.reshape(batch, 4, -1)
        typed_delta, _ = self.typed_router(
            query + intent_delta,
            torch.stack((state_delta, object_key_delta, object_delta), dim=-2),
            collect_diagnostics=False,
        )
        token, block_delta = self.block.forward_with_innovation(
            query + intent_delta + typed_delta,
            causal=True,
            value_innovation=intent_delta + typed_delta,
        )
        innovation = intent_delta + typed_delta + block_delta
        action_prediction = self.action_head(innovation)
        if future_action is None:
            target = None
            loss = action_prediction.new_zeros(())
        else:
            slices = _interval_slices(int(future_action.shape[1]))
            target = FuturePlanRecognizer._endpoint_summary(future_action, slices).detach()
            loss = (action_prediction.float() - target.float()).square().mean()
        return CoarseActionIntentState(
            tokens=token,
            innovations=innovation,
            action_prediction=action_prediction,
            target=target,
            loss=loss,
        )
