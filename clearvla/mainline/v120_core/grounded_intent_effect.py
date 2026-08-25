"""Grounded intent/effect ownership for the post-V118 3-2-3 mainline.

This module is deliberately independent from the historical ``vXXX`` graph.
It owns only the typed boundaries introduced by the
``grounded_intent_effect_323`` capability:

* G exposes a lossless, object-slotted current fact set;
* S organizes observable intent without a phase label or scalar progress;
* W predicts four object-level, window-local effect fields;
* P2 performs one bounded read over that supervised field;
* P3 compiles consequence-conditioned precision and temporal lanes.

The code is intentionally algebraic at the important boundaries.  In
particular, a neutral future effect produces an exact-zero P2 value and an
exact identity consequence update.  Reliability and uncertainty are metadata;
they never become non-zero fallback values on the action path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .role_delta_attnres import PolicyRoleDeltaBank, smooth_rms_contract

CAPABILITY_NAME = "grounded_intent_effect_323"
CAPABILITY_SCHEMA = 1
INTERVAL_NAMES = ("h4_8", "h8_16", "h16_32", "h32_48")
INTERVAL_BOUNDS = ((4, 8), (8, 16), (16, 32), (32, 48))


@dataclass(frozen=True)
class ArchitectureManifest:
    """Small serialized identity for the new top graph."""

    capability: str = CAPABILITY_NAME
    schema: int = CAPABILITY_SCHEMA
    topology: tuple[int, int, int] = (3, 2, 3)
    intervals: tuple[tuple[int, int], ...] = INTERVAL_BOUNDS
    language_required: bool = True
    bottom_compatibility: str = "evidence_mmdit_cvae_workspace_v1"

    def as_dict(self) -> dict[str, object]:
        return {
            "capability": self.capability,
            "schema": int(self.schema),
            "topology": tuple(int(value) for value in self.topology),
            "intervals": tuple(
                tuple(int(value) for value in interval)
                for interval in self.intervals
            ),
            "language_required": bool(self.language_required),
            "bottom_compatibility": self.bottom_compatibility,
        }

    def validate(self) -> None:
        if self.capability != CAPABILITY_NAME:
            raise ValueError("grounded architecture capability identity is invalid")
        if int(self.schema) != CAPABILITY_SCHEMA:
            raise ValueError("grounded architecture schema is unsupported")
        if tuple(self.topology) != (3, 2, 3):
            raise ValueError("grounded architecture requires the 3-2-3 topology")
        if tuple(self.intervals) != INTERVAL_BOUNDS:
            raise ValueError("grounded architecture requires four canonical intervals")
        if not self.language_required:
            raise ValueError("formal grounded training requires language")


GROUNDING_MANIFEST = ArchitectureManifest()


def _ordered_basis(length: int, width: int) -> Tensor:
    if min(int(length), int(width)) < 1:
        raise ValueError("ordered basis dimensions must be positive")
    position = torch.linspace(0.0, 1.0, int(length), dtype=torch.float32)[:, None]
    frequency = torch.arange(1, int(width) // 2 + 1, dtype=torch.float32)[None]
    angle = math.pi * position * frequency
    basis = torch.cat((torch.sin(angle), torch.cos(angle)), dim=-1)
    if int(basis.shape[-1]) < int(width):
        basis = F.pad(basis, (0, int(width) - int(basis.shape[-1])))
    return basis[:, : int(width)]


def _causal_mask(length: int, *, device: torch.device) -> Tensor:
    return torch.triu(
        torch.ones(length, length, device=device, dtype=torch.bool),
        diagonal=1,
    )


def _normalized_entropy(probability: Tensor, *, dim: int = -1) -> Tensor:
    support = int(probability.shape[dim])
    if support < 2:
        return probability.new_zeros(probability.shape[:-1], dtype=torch.float32)
    value = probability.float().clamp_min(1e-8)
    return -(value * value.log()).sum(dim=dim) / math.log(float(support))


def bounded_owner_update(
    parent_probability: Tensor,
    residual_logit: Tensor,
    *,
    maximum_residual: float = 0.50,
) -> Tensor:
    """Refine a G2 owner posterior while preserving an exact zero identity."""

    if tuple(parent_probability.shape) != tuple(residual_logit.shape):
        raise ValueError("owner parent probability and residual must align")
    parent = parent_probability.float().clamp_min(1e-8)
    parent = parent / parent.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    residual = float(maximum_residual) * torch.tanh(residual_logit.float())
    return torch.softmax(parent.log() + residual, dim=-1).to(
        dtype=parent_probability.dtype
    )


def sample_spatial_slots(chart: Tensor, coordinates: Tensor) -> Tensor:
    """Bilinearly sample a camera chart at every G object-slot coordinate.

    ``chart`` is ``[B,C,Y,X,H]`` and normalized ``coordinates`` are
    ``[B,C,Y,X,M,2]`` in PyTorch's ``(x,y)`` order.
    """

    if chart.ndim != 5 or coordinates.ndim != 6:
        raise ValueError("slot sampling requires [B,C,Y,X,H] and [B,C,Y,X,M,2]")
    batch, cameras, rows, columns, hidden = chart.shape
    if tuple(coordinates.shape[:4]) != (batch, cameras, rows, columns):
        raise ValueError("slot coordinates do not align with the current chart")
    if int(coordinates.shape[-1]) != 2:
        raise ValueError("slot coordinates must contain normalized x/y")
    slots = int(coordinates.shape[-2])
    source = chart.permute(0, 1, 4, 2, 3).reshape(
        batch * cameras,
        hidden,
        rows,
        columns,
    )
    grid = coordinates.reshape(
        batch * cameras,
        rows,
        columns * slots,
        2,
    ).to(dtype=source.dtype)
    sampled = F.grid_sample(
        source,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.reshape(
        batch,
        cameras,
        hidden,
        rows,
        columns,
        slots,
    ).permute(0, 1, 3, 4, 5, 2)


@dataclass(frozen=True)
class GroundedFactSet:
    """Completed G3 facts with camera/cell/object/type identity intact."""

    public_scene_base: Tensor
    content_slots: Tensor
    semantic_slots: Tensor
    appearance_slots: Tensor
    geometry_slots: Tensor
    semantic_owner_probs: Tensor
    appearance_owner_probs: Tensor
    geometry_owner_probs: Tensor
    slot_coordinates: Tensor
    slot_support: Tensor
    slot_validity: Tensor
    # Explicit normalized source->learned-flow displacement.  Older callers
    # may omit it; new object dynamics never infer motion from arbitrary
    # geometry feature channels.
    slot_transport_prior: Tensor | None = None
    # Producer-owned FP32 log posteriors.  These are optional only for
    # historical fixtures outside the active mainline.  Schema39 keeps them
    # beside the model-dtype probabilities so no downstream consumer takes a
    # logarithm after BF16 rounding or underflow.
    semantic_owner_log_probs: Tensor | None = None
    appearance_owner_log_probs: Tensor | None = None
    geometry_owner_log_probs: Tensor | None = None

    @property
    def batch(self) -> int:
        return int(self.semantic_slots.shape[0])

    @property
    def slots(self) -> int:
        return int(self.semantic_slots.shape[-2])

    @property
    def route_dim(self) -> int:
        return int(self.semantic_slots.shape[-1])

    @property
    def hidden(self) -> int:
        return int(self.content_slots.shape[-1])

    def validate(self) -> None:
        if self.public_scene_base.ndim != 5:
            raise ValueError("public scene base must be [B,C,Y,X,H]")
        if self.content_slots.ndim != 6:
            raise ValueError("content slots must be [B,C,Y,X,M,H]")
        if self.semantic_slots.ndim != 6:
            raise ValueError("typed G slots must be [B,C,Y,X,M,R]")
        prefix = tuple(self.semantic_slots.shape[:-1])
        if tuple(self.content_slots.shape[:-1]) != prefix:
            raise ValueError("G content and typed slot axes do not align")
        for name in ("appearance_slots", "geometry_slots"):
            value = getattr(self, name)
            if tuple(value.shape) != tuple(self.semantic_slots.shape):
                raise ValueError(f"G {name} is not aligned to semantic slots")
        owner_shape = prefix
        for name in (
            "semantic_owner_probs",
            "appearance_owner_probs",
            "geometry_owner_probs",
        ):
            value = getattr(self, name)
            if tuple(value.shape) != owner_shape:
                raise ValueError(f"G {name} must retain the object-slot axis")
        if tuple(self.slot_coordinates.shape) != (*prefix, 2):
            raise ValueError("G slot coordinates are misaligned")
        if tuple(self.slot_support.shape) != prefix:
            raise ValueError("G slot support is misaligned")
        if tuple(self.slot_validity.shape) != (*prefix, 1):
            raise ValueError("G slot validity is misaligned")
        if self.slot_transport_prior is not None and tuple(
            self.slot_transport_prior.shape
        ) != (*prefix, 2):
            raise ValueError("G slot transport prior is misaligned")
        for name in (
            "semantic_owner_log_probs",
            "appearance_owner_log_probs",
            "geometry_owner_log_probs",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            if tuple(value.shape) != owner_shape:
                raise ValueError(f"G {name} must retain the object-slot axis")
            if value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise TypeError(f"G {name} must be finite FP32")
        if tuple(self.public_scene_base.shape[:4]) != tuple(prefix[:4]):
            raise ValueError("G public scene base lost camera/spatial identity")
        # Value-domain checks live in the one-shot preflight.  This method is
        # intentionally shape-only because G facts are validated repeatedly by
        # S, W and Teacher-G; reducing every large CUDA tensor to a Python bool
        # here would serialize the training hot path.


class _BoundedCrossBlock(nn.Module):
    """Typed cross-attention returning the real bounded innovation."""

    def __init__(self, hidden: int, heads: int, *, maximum_rms: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.memory_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(
            hidden,
            heads,
            bias=False,
            dropout=0.0,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.ffn = nn.Sequential(
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
        need_weights: bool = False,
        memory_key_padding_mask: Tensor | None = None,
        average_attn_weights: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        update, attention = self.attention(
            self.query_norm(query),
            self.memory_norm(memory),
            self.memory_norm(memory),
            key_padding_mask=memory_key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=average_attn_weights,
        )
        update, _ = smooth_rms_contract(update, self.maximum_rms)
        intermediate = query + update
        ffn_update, _ = smooth_rms_contract(
            self.ffn(self.ffn_norm(intermediate)),
            self.maximum_rms,
        )
        innovation = update + ffn_update
        return intermediate + ffn_update, innovation, attention

    @torch.no_grad()
    def diagnostic_attention_weights(
        self,
        query: Tensor,
        memory: Tensor,
        *,
        memory_key_padding_mask: Tensor | None = None,
        average_attn_weights: bool = True,
    ) -> Tensor:
        """Read attention probabilities without changing the value kernel.

        ``nn.MultiheadAttention`` selects a different implementation when
        ``need_weights=True``.  That difference is normally harmless, but a
        BF16 diagnostic replay is expected to be bit-identical to deployment
        and repeated action-flow sampling can amplify the small kernel drift.
        The deployed value path therefore always keeps its original
        ``need_weights=False`` call; this detached FP32 side calculation owns
        diagnostics only.
        """

        projection = self.attention.in_proj_weight
        if projection is None:
            raise RuntimeError("grounded cross-attention lost its QKV projection")
        hidden = int(query.shape[-1])
        heads = int(self.attention.num_heads)
        if hidden % heads:
            raise RuntimeError("grounded attention width is not head-aligned")
        head_width = hidden // heads
        normalized_query = self.query_norm(query).detach().float()
        normalized_memory = self.memory_norm(memory).detach().float()
        projection_f = projection.detach().float()
        query_projection = F.linear(
            normalized_query,
            projection_f[:hidden],
        )
        key_projection = F.linear(
            normalized_memory,
            projection_f[hidden : 2 * hidden],
        )
        batch, query_count = query_projection.shape[:2]
        memory_count = int(key_projection.shape[1])
        query_heads = query_projection.reshape(
            batch,
            query_count,
            heads,
            head_width,
        ).transpose(1, 2)
        key_heads = key_projection.reshape(
            batch,
            memory_count,
            heads,
            head_width,
        ).transpose(1, 2)
        score = torch.einsum(
            "bhqd,bhkd->bhqk",
            query_heads,
            key_heads,
        ) / math.sqrt(float(head_width))
        if memory_key_padding_mask is not None:
            if tuple(memory_key_padding_mask.shape) != (batch, memory_count):
                raise ValueError(
                    "grounded diagnostic attention mask is misaligned"
                )
            score = score.masked_fill(
                memory_key_padding_mask[:, None, None].to(
                    device=score.device,
                    dtype=torch.bool,
                ),
                torch.finfo(score.dtype).min,
            )
        probability = torch.softmax(score, dim=-1)
        return probability.mean(dim=1) if average_attn_weights else probability


class _BoundedSelfBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, *, maximum_rms: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(
            hidden,
            heads,
            bias=False,
            dropout=0.0,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, 2 * hidden, bias=False),
            nn.GELU(),
            nn.Linear(2 * hidden, hidden, bias=False),
        )
        self.maximum_rms = float(maximum_rms)

    def forward(
        self,
        value: Tensor,
        *,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        normalized = self.norm(value)
        update, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=attention_mask,
            need_weights=False,
        )
        update, _ = smooth_rms_contract(update, self.maximum_rms)
        intermediate = value + update
        ffn_update, _ = smooth_rms_contract(
            self.ffn(self.ffn_norm(intermediate)),
            self.maximum_rms,
        )
        return intermediate + ffn_update, update + ffn_update


class _TypedInnovationRouter(nn.Module):
    """AttnRes-style routing after source types have produced innovations."""

    def __init__(self, hidden: int, maximum_sources: int) -> None:
        super().__init__()
        self.maximum_sources = int(maximum_sources)
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.keys = nn.ModuleList(
            nn.Linear(hidden, hidden, bias=False)
            for _ in range(self.maximum_sources)
        )

    def forward(
        self,
        protected_base: Tensor,
        innovations: tuple[Tensor, ...],
    ) -> tuple[Tensor, Tensor]:
        if not 1 <= len(innovations) <= self.maximum_sources:
            raise ValueError("typed innovation source count is invalid")
        if any(tuple(value.shape) != tuple(protected_base.shape) for value in innovations):
            raise ValueError("typed innovations must align with their protected base")
        query = F.normalize(self.query(protected_base).float(), dim=-1, eps=1e-4)
        logits = [
            (query * F.normalize(self.keys[index](value).float(), dim=-1, eps=1e-4))
            .sum(dim=-1)
            for index, value in enumerate(innovations)
        ]
        # The null route is a legal no-innovation choice.  The protected base
        # never enters this softmax.
        probability = torch.softmax(
            torch.stack(
                (
                    protected_base.new_zeros(
                        protected_base.shape[:-1],
                        dtype=torch.float32,
                    ),
                    *logits,
                ),
                dim=-1,
            ),
            dim=-1,
        )
        values = torch.stack(innovations, dim=-2)
        selected = (
            probability[..., 1:, None].to(dtype=values.dtype) * values
        ).sum(dim=-2)
        return protected_base + selected, probability


@dataclass(frozen=True)
class StatelessIntentState:
    """Observable, set-valued intent state with no phase/progress variable."""

    protected_goal_tokens: Tensor
    achieved_evidence: Tensor
    remaining_goal: Tensor
    interval_intents: Tensor
    temporal_control: Tensor
    completion_evidence: Tensor
    completion_probability: Tensor
    completion_uncertainty: Tensor
    goal_attention: Tensor
    interval_source_attention: Tensor

    @property
    def interval_selector(self) -> Tensor:
        return self.interval_intents

    @property
    def goal_context(self) -> Tensor:
        return self.protected_goal_tokens

    @property
    def history_context(self) -> Tensor:
        return torch.stack(
            (self.achieved_evidence, self.remaining_goal),
            dim=1,
        )

    @property
    def terminal_probability(self) -> Tensor:
        return self.completion_probability

    def validate(self, *, batch: int, hidden: int, horizon: int) -> None:
        expected: dict[str, tuple[int, ...]] = {
            "protected_goal_tokens": (batch, 4, hidden),
            "achieved_evidence": (batch, hidden),
            "remaining_goal": (batch, hidden),
            "interval_intents": (batch, 4, hidden),
            "temporal_control": (batch, horizon, hidden),
            "completion_evidence": (batch, hidden),
            "completion_probability": (batch, 1),
            "completion_uncertainty": (batch, 1),
            "goal_attention": (batch, 4, int(self.goal_attention.shape[-1])),
            "interval_source_attention": (batch, 4, 6),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"grounded intent {name} must be {shape}, got {tuple(value.shape)}"
                )


class StatelessIntentOrganizer(nn.Module):
    """Three functional S blocks over goal, observable history and G facts."""

    interval_basis: Tensor
    temporal_basis: Tensor

    def __init__(
        self,
        *,
        hidden: int,
        state_dim: int,
        action_dim: int,
        fact_dim: int,
        action_horizon: int,
        goal_dim: int | None = None,
        heads: int = 4,
    ) -> None:
        super().__init__()
        if min(hidden, state_dim, action_dim, fact_dim, action_horizon, heads) < 1:
            raise ValueError("grounded intent dimensions must be positive")
        if hidden % heads:
            raise ValueError("grounded intent hidden width must divide heads")
        self.hidden = int(hidden)
        self.action_horizon = int(action_horizon)
        self.goal_queries = nn.Parameter(torch.randn(1, 4, hidden) * 0.02)
        self.goal_dim = int(hidden if goal_dim is None else goal_dim)
        self.goal_input = nn.Linear(self.goal_dim, hidden, bias=False)
        self.goal_block = _BoundedCrossBlock(hidden, heads, maximum_rms=0.35)

        self.state_input = nn.Linear(state_dim, hidden, bias=False)
        self.action_input = nn.Linear(action_dim, hidden, bias=False)
        self.history_type = nn.Parameter(torch.randn(1, 2, hidden) * 0.02)
        self.history_blocks = nn.ModuleList(
            _BoundedSelfBlock(hidden, heads, maximum_rms=0.35)
            for _ in range(2)
        )
        self.observable_queries = nn.Parameter(torch.randn(1, 2, hidden) * 0.02)
        self.observable_goal = _BoundedCrossBlock(hidden, heads, maximum_rms=0.35)
        self.observable_history = _BoundedCrossBlock(
            hidden, heads, maximum_rms=0.35
        )
        self.fact_inputs = nn.ModuleDict(
            {
                name: nn.Linear(fact_dim, hidden, bias=False)
                for name in ("semantic", "appearance", "geometry")
            }
        )
        self.observable_fact = nn.ModuleDict(
            {
                name: _BoundedCrossBlock(hidden, heads, maximum_rms=0.35)
                for name in ("semantic", "appearance", "geometry")
            }
        )
        self.observable_router = _TypedInnovationRouter(hidden, 4)

        self.register_buffer(
            "interval_basis",
            _ordered_basis(4, hidden),
            persistent=False,
        )
        self.interval_queries = nn.Parameter(torch.randn(1, 4, hidden) * 0.02)
        self.interval_goal = _BoundedCrossBlock(hidden, heads, maximum_rms=0.35)
        self.interval_observable = _BoundedCrossBlock(
            hidden, heads, maximum_rms=0.35
        )
        self.interval_history = _BoundedCrossBlock(
            hidden, heads, maximum_rms=0.35
        )
        self.interval_fact = nn.ModuleDict(
            {
                name: _BoundedCrossBlock(hidden, heads, maximum_rms=0.35)
                for name in ("semantic", "appearance", "geometry")
            }
        )
        self.interval_router = _TypedInnovationRouter(hidden, 5)

        self.register_buffer(
            "temporal_basis",
            _ordered_basis(action_horizon, hidden),
            persistent=False,
        )
        self.temporal_query = nn.Linear(hidden, hidden, bias=False)
        self.temporal_read = _BoundedCrossBlock(hidden, heads, maximum_rms=0.35)
        self.completion = nn.Sequential(
            nn.LayerNorm(2 * hidden, elementwise_affine=False),
            nn.Linear(2 * hidden, hidden, bias=False),
            nn.SiLU(),
            nn.Linear(hidden, hidden, bias=False),
        )
        self.completion_head = nn.Linear(hidden, 2, bias=True)
        nn.init.normal_(self.completion_head.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.completion_head.bias, -2.5)

    @staticmethod
    def _validate_tokens(name: str, value: Tensor, batch: int, width: int) -> None:
        if (
            value.ndim != 3
            or int(value.shape[0]) != int(batch)
            or int(value.shape[1]) < 1
            or int(value.shape[2]) != int(width)
        ):
            raise ValueError(f"grounded intent {name} must be non-empty [B,N,{width}]")

    def _history(
        self,
        state_history_tokens: Tensor,
        action_history_tokens: Tensor,
    ) -> Tensor:
        state = self.state_input(state_history_tokens)
        action = self.action_input(action_history_tokens)
        state = state + self.history_type[:, :1].to(dtype=state.dtype)
        action = action + self.history_type[:, 1:].to(dtype=action.dtype)
        # State and executed-action histories are two ordered modalities, not
        # two halves of one artificial timeline.  Concatenating them before a
        # causal mask makes every state token precede every action token and
        # silently prevents the state stream from seeing action history.  Give
        # each stream its real within-modality order, share the temporal
        # encoder weights, and join them only when the observable/interval
        # queries perform typed cross-attention.
        streams = tuple(
            stream
            + _ordered_basis(int(stream.shape[1]), self.hidden)
            .to(device=stream.device, dtype=stream.dtype)[None]
            for stream in (state, action)
        )
        for block in self.history_blocks:
            next_streams: list[Tensor] = []
            for stream in streams:
                encoded, _ = block(
                    stream,
                    attention_mask=_causal_mask(
                        int(stream.shape[1]),
                        device=stream.device,
                    ),
                )
                next_streams.append(encoded)
            streams = tuple(next_streams)
        return torch.cat(streams, dim=1)

    def _fact_tokens(self, facts: GroundedFactSet) -> dict[str, Tensor]:
        facts.validate()
        result: dict[str, Tensor] = {}
        for name, slots, owner in (
            ("semantic", facts.semantic_slots, facts.semantic_owner_probs),
            ("appearance", facts.appearance_slots, facts.appearance_owner_probs),
            ("geometry", facts.geometry_slots, facts.geometry_owner_probs),
        ):
            projected = self.fact_inputs[name](slots)
            # The slot axis stays explicit.  Owner probability scales evidence
            # strength but never sums or recreates slots.
            result[name] = (
                projected * owner[..., None].to(dtype=projected.dtype)
            ).reshape(facts.batch, -1, self.hidden)
        return result

    def forward(
        self,
        *,
        goal_tokens: Tensor,
        goal_mask: Tensor | None = None,
        state_history_tokens: Tensor,
        action_history_tokens: Tensor,
        facts: GroundedFactSet,
        collect_diagnostics: bool = True,
    ) -> tuple[StatelessIntentState, dict[str, Tensor]]:
        facts.validate()
        batch = facts.batch
        self._validate_tokens("goal", goal_tokens, batch, self.goal_dim)
        if goal_mask is not None:
            if tuple(goal_mask.shape) != tuple(goal_tokens.shape[:2]):
                raise ValueError("grounded intent goal mask must align with T5 tokens")
            goal_mask = goal_mask.to(
                device=goal_tokens.device,
                dtype=torch.bool,
            )
            # Formal loading/preflight rejects an empty language condition.
            # Keep attention numerically defined without a device->host sync
            # if a malformed per-sample row nevertheless reaches this layer.
            row_has_token = goal_mask.any(dim=-1, keepdim=True)
            first_token = torch.zeros_like(goal_mask)
            first_token[:, 0] = True
            goal_mask = torch.where(row_has_token, goal_mask, first_token)
        self._validate_tokens(
            "state history",
            state_history_tokens,
            batch,
            self.state_input.in_features,
        )
        self._validate_tokens(
            "action history",
            action_history_tokens,
            batch,
            self.action_input.in_features,
        )
        goal_seed = self.goal_queries.to(
            device=goal_tokens.device,
            dtype=goal_tokens.dtype,
        ).expand(batch, -1, -1)
        protected_goal, _, goal_attention = self.goal_block(
            goal_seed,
            self.goal_input(goal_tokens),
            need_weights=True,
            memory_key_padding_mask=(
                None
                if goal_mask is None
                else ~goal_mask
            ),
        )
        assert goal_attention is not None
        history = self._history(state_history_tokens, action_history_tokens)
        fact_tokens = self._fact_tokens(facts)

        observable_seed = self.observable_queries.to(
            device=goal_tokens.device,
            dtype=goal_tokens.dtype,
        ).expand(batch, -1, -1)
        observable_base, _, _ = self.observable_goal(
            observable_seed,
            protected_goal,
        )
        _, history_innovation, _ = self.observable_history(
            observable_base,
            history,
        )
        typed_observable: list[Tensor] = [history_innovation]
        for name in ("semantic", "appearance", "geometry"):
            _, innovation, _ = self.observable_fact[name](
                observable_base,
                fact_tokens[name],
            )
            typed_observable.append(innovation)
        observable_state, observable_attention = self.observable_router(
            observable_base,
            tuple(typed_observable),
        )
        achieved = observable_state[:, 0]
        remaining = observable_state[:, 1]

        interval_seed = (
            self.interval_queries.to(
                device=goal_tokens.device,
                dtype=goal_tokens.dtype,
            )
            + self.interval_basis.to(
                device=goal_tokens.device,
                dtype=goal_tokens.dtype,
            )[None]
        ).expand(batch, -1, -1)
        interval_base, _, _ = self.interval_goal(
            interval_seed,
            protected_goal,
            # Keep deployment and diagnostic replays on the same value
            # kernel.  Per-head probabilities are reconstructed below on a
            # detached audit-only side path.
            need_weights=False,
        )
        interval_goal_attention = (
            self.interval_goal.diagnostic_attention_weights(
                interval_seed,
                protected_goal,
                average_attn_weights=False,
            )
            if collect_diagnostics
            else None
        )
        _, observable_innovation, _ = self.interval_observable(
            interval_base,
            observable_state,
        )
        _, interval_history_innovation, _ = self.interval_history(
            interval_base,
            history,
        )
        interval_innovations: list[Tensor] = [
            observable_innovation,
            interval_history_innovation,
        ]
        for name in ("semantic", "appearance", "geometry"):
            _, innovation, _ = self.interval_fact[name](
                interval_base,
                fact_tokens[name],
            )
            interval_innovations.append(innovation)
        interval_intents, interval_source_attention = self.interval_router(
            interval_base,
            tuple(interval_innovations),
        )

        temporal_seed = self.temporal_query(
            self.temporal_basis.to(
                device=goal_tokens.device,
                dtype=goal_tokens.dtype,
            )
        )[None].expand(batch, -1, -1)
        temporal_control, _, _ = self.temporal_read(
            temporal_seed,
            interval_intents,
        )
        completion_evidence, _ = smooth_rms_contract(
            self.completion(torch.cat((achieved, remaining), dim=-1)),
            0.35,
        )
        completion_raw = self.completion_head(completion_evidence).float()
        completion_probability = torch.sigmoid(completion_raw[:, :1]).to(
            dtype=goal_tokens.dtype
        )
        completion_uncertainty = torch.sigmoid(completion_raw[:, 1:2]).to(
            dtype=goal_tokens.dtype
        )
        state = StatelessIntentState(
            protected_goal_tokens=protected_goal,
            achieved_evidence=achieved,
            remaining_goal=remaining,
            interval_intents=interval_intents,
            temporal_control=temporal_control,
            completion_evidence=completion_evidence,
            completion_probability=completion_probability,
            completion_uncertainty=completion_uncertainty,
            goal_attention=goal_attention.to(dtype=goal_tokens.dtype),
            interval_source_attention=interval_source_attention.to(
                dtype=goal_tokens.dtype
            ),
        )
        state.validate(batch=batch, hidden=self.hidden, horizon=self.action_horizon)

        metrics: dict[str, Tensor] = {}
        if collect_diagnostics:
            if interval_goal_attention is None:
                raise RuntimeError(
                    "grounded S diagnostics lost interval-to-goal attention"
                )
            interval_f = interval_intents.detach().float()
            interval_goal_attention_f = (
                interval_goal_attention.detach().float()
            )
            metrics = {
                "grounded_s_goal_attention_entropy": _normalized_entropy(
                    goal_attention.detach().float()
                ).mean(),
                "grounded_s_interval_goal_attention_entropy": (
                    _normalized_entropy(
                        interval_goal_attention_f,
                        dim=-1,
                    ).mean()
                ),
                "grounded_s_interval_source_entropy": _normalized_entropy(
                    interval_source_attention.detach().float()
                ).mean(),
                "grounded_s_interval_adjacent_cosine": F.cosine_similarity(
                    interval_f[:, 1:],
                    interval_f[:, :-1],
                    dim=-1,
                    eps=1e-6,
                ).mean(),
                "grounded_s_interval_variation": interval_f.std(
                    dim=1,
                    unbiased=False,
                ).mean(),
                "grounded_s_achieved_rms": achieved.detach()
                .float()
                .square()
                .mean()
                .sqrt(),
                "grounded_s_remaining_rms": remaining.detach()
                .float()
                .square()
                .mean()
                .sqrt(),
                "grounded_s_completion_probability": completion_probability.detach()
                .float()
                .mean(),
            }
            source_names = (
                "null",
                "observable",
                "history",
                "semantic",
                "appearance",
                "geometry",
            )
            source_mass = interval_source_attention.detach().float().mean(
                dim=(0, 1)
            )
            for index, name in enumerate(source_names):
                metrics[f"grounded_s_interval_{name}_mass"] = source_mass[index]
            for index, name in enumerate(INTERVAL_NAMES):
                metrics[f"grounded_s_{name}_goal_attention_entropy"] = (
                    _normalized_entropy(
                        interval_goal_attention_f[:, :, index],
                        dim=-1,
                    ).mean()
                )
                metrics[f"grounded_s_{name}_source_attention_entropy"] = (
                    _normalized_entropy(
                        interval_source_attention.detach().float()[:, index],
                        dim=-1,
                    ).mean()
                )
                for source_index, source_name in enumerate(source_names):
                    metrics[
                        f"grounded_s_{name}_{source_name}_mass"
                    ] = (
                        interval_source_attention.detach().float()[
                            :, index, source_index
                        ].mean()
                    )
                for head_index in range(
                    int(interval_goal_attention_f.shape[1])
                ):
                    metrics[
                        f"grounded_s_{name}_goal_head_{head_index}_entropy"
                    ] = (
                        _normalized_entropy(
                            interval_goal_attention_f[
                                :, head_index, index
                            ],
                            dim=-1,
                        ).mean()
                    )
        return state, metrics


@dataclass(frozen=True)
class FutureEffectField:
    """The only supervised W object that can cross into P2."""

    current_reference: Tensor
    semantic_delta: Tensor
    transport_delta: Tensor
    covariance_delta: Tensor
    visibility_change: Tensor
    persistence_change: Tensor
    reliability: Tensor
    validity: Tensor
    uncertainty: Tensor
    source_coordinates: Tensor
    interval_names: tuple[str, ...] = INTERVAL_NAMES

    @property
    def successor_content(self) -> Tensor:
        return self.current_reference[:, None] + self.semantic_delta

    @property
    def intervals(self) -> int:
        return int(self.semantic_delta.shape[1])

    def validate(self, *, expected_intervals: int = 4) -> None:
        if self.current_reference.ndim != 6:
            raise ValueError("effect current reference must be [B,C,Y,X,M,H]")
        if self.semantic_delta.ndim != 7:
            raise ValueError("effect semantic delta must be [B,A,C,Y,X,M,H]")
        if tuple(self.semantic_delta.shape[:1] + self.semantic_delta.shape[2:]) != tuple(
            self.current_reference.shape
        ):
            raise ValueError("effect current and interval axes do not align")
        if self.intervals != int(expected_intervals):
            raise ValueError(
                f"effect requires {expected_intervals} intervals, got {self.intervals}"
            )
        prefix = tuple(self.semantic_delta.shape[:-1])
        expected = {
            "transport_delta": (*prefix, 2),
            "covariance_delta": (*prefix, 3),
            "visibility_change": (*prefix, 1),
            "persistence_change": (*prefix, 1),
            "reliability": (*prefix, 1),
            "validity": (*prefix, 1),
            "uncertainty": (*prefix, 1),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape:
                raise ValueError(f"effect {name} must be {shape}")
        coordinate_shape = (
            int(prefix[0]),
            int(prefix[2]),
            int(prefix[3]),
            int(prefix[4]),
            int(prefix[5]),
            2,
        )
        if tuple(self.source_coordinates.shape) != coordinate_shape:
            raise ValueError("effect source coordinates are misaligned")
        if len(self.interval_names) != self.intervals:
            raise ValueError("effect interval names do not match its interval axis")

    @classmethod
    def neutral_from(cls, field: "FutureEffectField") -> "FutureEffectField":
        field.validate(expected_intervals=field.intervals)
        return cls(
            current_reference=field.current_reference,
            semantic_delta=torch.zeros_like(field.semantic_delta),
            transport_delta=torch.zeros_like(field.transport_delta),
            covariance_delta=torch.zeros_like(field.covariance_delta),
            visibility_change=torch.zeros_like(field.visibility_change),
            persistence_change=torch.zeros_like(field.persistence_change),
            reliability=torch.zeros_like(field.reliability),
            validity=field.validity,
            uncertainty=torch.zeros_like(field.uncertainty),
            source_coordinates=field.source_coordinates,
            interval_names=field.interval_names,
        )


@dataclass(frozen=True)
class GroundedWorldWorkingState:
    """Private W state; only ``effect`` is allowed to leave W."""

    clean_proposal: Tensor
    semantic_w1: Tensor | None = None
    appearance_w1: Tensor | None = None
    geometry_w1: Tensor | None = None
    effect_w1: FutureEffectField | None = None
    effect: FutureEffectField | None = None


class GroundedWorldEffectCompiler(nn.Module):
    """W1/W2 four-interval object effects without a shared owner soup."""

    def __init__(
        self,
        *,
        hidden: int,
        fact_dim: int,
        route_dim: int,
        content_dim: int | None = None,
        heads: int = 4,
    ) -> None:
        super().__init__()
        if min(hidden, fact_dim, route_dim, heads) < 1 or route_dim % heads:
            raise ValueError("grounded W dimensions are invalid")
        self.hidden = int(hidden)
        self.fact_dim = int(fact_dim)
        self.route_dim = int(route_dim)
        self.content_dim = int(
            hidden if content_dim is None else content_dim
        )
        if self.content_dim < 1:
            raise ValueError("grounded W content width must be positive")
        self.world_input = nn.Linear(hidden, route_dim, bias=False)
        self.intent_input = nn.Linear(hidden, route_dim, bias=False)
        self.proposal_input = nn.Linear(hidden, route_dim, bias=False)
        self.owner_input = nn.ModuleDict(
            {
                name: nn.Linear(fact_dim, route_dim, bias=False)
                for name in ("semantic", "appearance", "geometry")
            }
        )
        self.w1_blocks = nn.ModuleDict(
            {
                name: _BoundedSelfBlock(route_dim, heads, maximum_rms=0.50)
                for name in ("semantic", "appearance", "geometry")
            }
        )
        self.w2_blocks = nn.ModuleDict(
            {
                name: _BoundedSelfBlock(route_dim, heads, maximum_rms=0.50)
                for name in ("semantic", "appearance", "geometry")
            }
        )
        self.semantic_head = nn.Sequential(
            nn.LayerNorm(route_dim, elementwise_affine=False),
            nn.Linear(route_dim, 2 * route_dim, bias=False),
            nn.SiLU(),
            nn.Linear(2 * route_dim, self.content_dim, bias=False),
        )
        self.geometry_head = nn.Sequential(
            nn.LayerNorm(route_dim, elementwise_affine=False),
            nn.Linear(route_dim, 2 * route_dim, bias=False),
            nn.SiLU(),
            nn.Linear(2 * route_dim, 5, bias=False),
        )
        self.appearance_head = nn.Sequential(
            nn.LayerNorm(route_dim, elementwise_affine=False),
            nn.Linear(route_dim, 2 * route_dim, bias=False),
            nn.SiLU(),
            nn.Linear(2 * route_dim, 2, bias=False),
        )
        self.reliability_head = nn.Linear(route_dim, 1, bias=True)
        self.uncertainty_head = nn.Linear(route_dim, 1, bias=True)
        for module in (
            self.semantic_head[-1],
            self.geometry_head[-1],
            self.appearance_head[-1],
        ):
            output = cast(nn.Linear, module)
            nn.init.normal_(output.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.reliability_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.reliability_head.bias)
        nn.init.normal_(self.uncertainty_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.uncertainty_head.bias)

    def initialize(self, clean_proposal: Tensor) -> GroundedWorldWorkingState:
        if (
            clean_proposal.ndim != 3
            or int(clean_proposal.shape[1]) != 4
            or int(clean_proposal.shape[-1]) != self.hidden
        ):
            raise ValueError("grounded W requires one [B,4,H] clean proposal")
        return GroundedWorldWorkingState(clean_proposal=clean_proposal)

    def _owner_seed(
        self,
        world_tokens: Tensor,
        facts: GroundedFactSet,
        intent: StatelessIntentState,
        *,
        interval: slice,
        include_proposal: bool,
    ) -> dict[str, Tensor]:
        facts.validate()
        batch, anchors, cameras, rows, columns, hidden = world_tokens.shape
        if anchors != 4 or hidden != self.hidden:
            raise ValueError("grounded W tokens must preserve four spatial anchors")
        if tuple(facts.semantic_slots.shape[:4]) != (
            batch,
            cameras,
            rows,
            columns,
        ):
            raise ValueError("grounded W facts do not align to the rollout chart")
        route = self.world_input(world_tokens[:, interval])
        route = route + self.intent_input(intent.interval_intents[:, interval])[
            :,
            :,
            None,
            None,
            None,
        ]
        if include_proposal:
            raise RuntimeError("proposal must be supplied through the private W state")
        owner_rows: dict[str, Tensor] = {}
        for name, value in (
            ("semantic", facts.semantic_slots),
            ("appearance", facts.appearance_slots),
            ("geometry", facts.geometry_slots),
        ):
            owner_rows[name] = (
                route[..., None, :]
                + self.owner_input[name](value)[:, None]
            )
        return owner_rows

    def _advance_owner(
        self,
        value: Tensor,
        *,
        block: _BoundedSelfBlock,
        prefix: Tensor | None,
    ) -> Tensor:
        if value.ndim != 7:
            raise ValueError("grounded W owner state must be [B,A,C,Y,X,M,R]")
        complete = value if prefix is None else torch.cat((prefix, value), dim=1)
        batch, intervals, cameras, rows, columns, slots, route_dim = complete.shape
        flattened = complete.permute(0, 2, 3, 4, 5, 1, 6).reshape(
            batch * cameras * rows * columns * slots,
            intervals,
            route_dim,
        )
        flattened, _ = block(
            flattened,
            attention_mask=_causal_mask(intervals, device=flattened.device),
        )
        complete = flattened.reshape(
            batch,
            cameras,
            rows,
            columns,
            slots,
            intervals,
            route_dim,
        ).permute(0, 5, 1, 2, 3, 4, 6)
        return complete if prefix is None else complete[:, -int(value.shape[1]) :]

    def _decode(
        self,
        *,
        semantic: Tensor,
        appearance: Tensor,
        geometry: Tensor,
        facts: GroundedFactSet,
        interval_names: tuple[str, ...],
        output_dtype: torch.dtype,
    ) -> FutureEffectField:
        if int(facts.content_slots.shape[-1]) != self.content_dim:
            raise ValueError(
                "grounded W current content width does not match the "
                "uncompressed DINO target space"
            )
        semantic_delta, _ = smooth_rms_contract(
            self.semantic_head(semantic),
            0.50,
        )
        raw_geometry = self.geometry_head(geometry).float()
        transport = 0.50 * torch.tanh(raw_geometry[..., :2])
        covariance = 0.50 * torch.tanh(raw_geometry[..., 2:5])
        raw_appearance = self.appearance_head(appearance).float()
        persistence = torch.tanh(raw_appearance[..., :1])
        visibility = torch.tanh(raw_appearance[..., 1:2])
        reliability = torch.sigmoid(self.reliability_head(semantic).float())
        uncertainty = F.softplus(self.uncertainty_head(semantic).float())
        validity = facts.slot_validity[:, None].expand(
            -1,
            int(semantic.shape[1]),
            -1,
            -1,
            -1,
            -1,
            -1,
        )
        field = FutureEffectField(
            current_reference=facts.content_slots.to(dtype=output_dtype),
            semantic_delta=semantic_delta.to(dtype=output_dtype),
            transport_delta=transport.to(dtype=output_dtype),
            covariance_delta=covariance.to(dtype=output_dtype),
            visibility_change=visibility.to(dtype=output_dtype),
            persistence_change=persistence.to(dtype=output_dtype),
            reliability=reliability.to(dtype=output_dtype),
            validity=validity.to(dtype=output_dtype),
            uncertainty=uncertainty.to(dtype=output_dtype),
            source_coordinates=facts.slot_coordinates.to(dtype=output_dtype),
            interval_names=interval_names,
        )
        field.validate(expected_intervals=len(interval_names))
        return field

    @staticmethod
    def _concat(first: FutureEffectField, second: FutureEffectField) -> FutureEffectField:
        first.validate(expected_intervals=2)
        second.validate(expected_intervals=2)
        field = FutureEffectField(
            current_reference=first.current_reference,
            semantic_delta=torch.cat((first.semantic_delta, second.semantic_delta), dim=1),
            transport_delta=torch.cat(
                (first.transport_delta, second.transport_delta), dim=1
            ),
            covariance_delta=torch.cat(
                (first.covariance_delta, second.covariance_delta), dim=1
            ),
            visibility_change=torch.cat(
                (first.visibility_change, second.visibility_change), dim=1
            ),
            persistence_change=torch.cat(
                (first.persistence_change, second.persistence_change), dim=1
            ),
            reliability=torch.cat((first.reliability, second.reliability), dim=1),
            validity=torch.cat((first.validity, second.validity), dim=1),
            uncertainty=torch.cat((first.uncertainty, second.uncertainty), dim=1),
            source_coordinates=first.source_coordinates,
            interval_names=first.interval_names + second.interval_names,
        )
        field.validate()
        return field

    @staticmethod
    def _metrics(prefix: str, field: FutureEffectField) -> dict[str, Tensor]:
        pooled = field.semantic_delta.detach().float().mean(dim=(2, 3, 4, 5))
        metrics = {
            f"grounded_{prefix}_semantic_rms": field.semantic_delta.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            f"grounded_{prefix}_transport_rms": field.transport_delta.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            f"grounded_{prefix}_interval_variation": pooled.std(
                dim=1,
                unbiased=False,
            ).mean(),
            f"grounded_{prefix}_object_variation": field.semantic_delta.detach()
            .float()
            .mean(dim=(2, 3, 4))
            .std(dim=2, unbiased=False)
            .mean(),
        }
        if int(pooled.shape[1]) > 1:
            metrics[f"grounded_{prefix}_adjacent_cosine"] = F.cosine_similarity(
                pooled[:, 1:],
                pooled[:, :-1],
                dim=-1,
                eps=1e-6,
            ).mean()
        for index, name in enumerate(field.interval_names):
            metrics[f"grounded_{prefix}_{name}_semantic_rms"] = (
                field.semantic_delta[:, index]
                .detach()
                .float()
                .square()
                .mean()
                .sqrt()
            )
        return metrics

    def forward_w1(
        self,
        *,
        world_tokens: Tensor,
        facts: GroundedFactSet,
        intent: StatelessIntentState,
        working: GroundedWorldWorkingState,
        output_dtype: torch.dtype,
        collect_diagnostics: bool = True,
    ) -> tuple[GroundedWorldWorkingState, dict[str, Tensor]]:
        seed = self._owner_seed(
            world_tokens,
            facts,
            intent,
            interval=slice(0, 2),
            include_proposal=False,
        )
        proposal = self.proposal_input(working.clean_proposal[:, :2])[
            :,
            :,
            None,
            None,
            None,
            None,
        ]
        for name in seed:
            seed[name] = seed[name] + proposal
        semantic = self._advance_owner(
            seed["semantic"],
            block=cast(_BoundedSelfBlock, self.w1_blocks["semantic"]),
            prefix=None,
        )
        appearance = self._advance_owner(
            seed["appearance"],
            block=cast(_BoundedSelfBlock, self.w1_blocks["appearance"]),
            prefix=None,
        )
        geometry = self._advance_owner(
            seed["geometry"],
            block=cast(_BoundedSelfBlock, self.w1_blocks["geometry"]),
            prefix=None,
        )
        effect = self._decode(
            semantic=semantic,
            appearance=appearance,
            geometry=geometry,
            facts=facts,
            interval_names=INTERVAL_NAMES[:2],
            output_dtype=output_dtype,
        )
        state = GroundedWorldWorkingState(
            clean_proposal=working.clean_proposal,
            semantic_w1=semantic,
            appearance_w1=appearance,
            geometry_w1=geometry,
            effect_w1=effect,
        )
        return state, self._metrics("w1", effect) if collect_diagnostics else {}

    def forward_w2(
        self,
        *,
        world_tokens: Tensor,
        facts: GroundedFactSet,
        intent: StatelessIntentState,
        working: GroundedWorldWorkingState,
        output_dtype: torch.dtype,
        collect_diagnostics: bool = True,
    ) -> tuple[GroundedWorldWorkingState, dict[str, Tensor]]:
        if any(
            value is None
            for value in (
                working.semantic_w1,
                working.appearance_w1,
                working.geometry_w1,
                working.effect_w1,
            )
        ):
            raise RuntimeError("grounded W2 requires the completed W1 state")
        seed = self._owner_seed(
            world_tokens,
            facts,
            intent,
            interval=slice(2, 4),
            include_proposal=False,
        )
        assert working.semantic_w1 is not None
        assert working.appearance_w1 is not None
        assert working.geometry_w1 is not None
        assert working.effect_w1 is not None
        semantic = self._advance_owner(
            seed["semantic"],
            block=cast(_BoundedSelfBlock, self.w2_blocks["semantic"]),
            prefix=working.semantic_w1,
        )
        appearance = self._advance_owner(
            seed["appearance"],
            block=cast(_BoundedSelfBlock, self.w2_blocks["appearance"]),
            prefix=working.appearance_w1,
        )
        geometry = self._advance_owner(
            seed["geometry"],
            block=cast(_BoundedSelfBlock, self.w2_blocks["geometry"]),
            prefix=working.geometry_w1,
        )
        late = self._decode(
            semantic=semantic,
            appearance=appearance,
            geometry=geometry,
            facts=facts,
            interval_names=INTERVAL_NAMES[2:],
            output_dtype=output_dtype,
        )
        complete = self._concat(working.effect_w1, late)
        state = GroundedWorldWorkingState(
            clean_proposal=working.clean_proposal,
            semantic_w1=working.semantic_w1,
            appearance_w1=working.appearance_w1,
            geometry_w1=working.geometry_w1,
            effect_w1=working.effect_w1,
            effect=complete,
        )
        # W2 owns only the late two intervals.  The complete four-interval
        # field is the public W -> P object, but reporting it as ``w2`` makes
        # W1 behavior silently contaminate every W2 statistic.
        metrics = self._metrics("w2", late) if collect_diagnostics else {}
        return state, metrics


class BoundedFutureEffectReader(nn.Module):
    """P2 read with bounded content/intent/coordinate score components."""

    def __init__(
        self,
        *,
        hidden: int,
        horizon: int,
        basis: int,
        effect_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.effect_dim = int(
            hidden if effect_dim is None else effect_dim
        )
        self.horizon = int(horizon)
        self.basis = int(basis)
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.current_key = nn.Linear(
            self.effect_dim, hidden, bias=False
        )
        self.effect_key = nn.Linear(
            self.effect_dim, hidden, bias=False
        )
        self.geometry_key = nn.Linear(7, hidden, bias=False)
        self.intent_key = nn.Linear(hidden, hidden, bias=False)
        # P1 has already grounded every action query in the current visual
        # chart.  Predict its soft image-space address from that query instead
        # of pretending that the action time/basis indices are image x/y.
        self.action_coordinate = nn.Linear(hidden, 2, bias=False)
        self.effect_value = nn.Linear(
            self.effect_dim + 7,
            hidden,
            bias=False,
        )
        nn.init.normal_(self.effect_value.weight, mean=0.0, std=3e-3)
        initial_temperature = math.log(0.20 / 0.80)
        self.temperature_logit = nn.Parameter(
            torch.full((3,), initial_temperature, dtype=torch.float32)
        )

    @property
    def temperatures(self) -> Tensor:
        return 0.25 + 3.75 * torch.sigmoid(self.temperature_logit.float())

    @staticmethod
    def _zero_centered_geometry(field: FutureEffectField) -> Tensor:
        return torch.cat(
            (
                field.transport_delta,
                field.covariance_delta,
                field.visibility_change,
                field.persistence_change,
            ),
            dim=-1,
        )

    def forward(
        self,
        query_tokens: Tensor,
        field: FutureEffectField,
        intent: StatelessIntentState,
        *,
        collect_diagnostics: bool = True,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        field.validate()
        batch = int(query_tokens.shape[0])
        expected = (batch, self.horizon, self.basis, self.hidden)
        if tuple(query_tokens.shape) != expected:
            raise ValueError(f"grounded P2 query must be {expected}")
        intent.validate(batch=batch, hidden=self.hidden, horizon=self.horizon)
        intervals = field.intervals
        if (
            int(field.semantic_delta.shape[-1]) != self.effect_dim
            or int(field.current_reference.shape[-1]) != self.effect_dim
        ):
            raise ValueError(
                "grounded P2 effect content is not in the configured "
                "DINO target space"
            )
        spatial = int(
            field.semantic_delta[0, 0].numel() // self.effect_dim
        )
        query = F.normalize(self.query(query_tokens).float(), dim=-1, eps=1e-4)
        geometry = self._zero_centered_geometry(field)
        key = (
            self.current_key(field.current_reference)[:, None]
            + self.effect_key(field.semantic_delta)
            + self.geometry_key(geometry)
        ).reshape(batch, intervals, spatial, self.hidden)
        key = F.normalize(key.float(), dim=-1, eps=1e-4)
        raw_value = self.effect_value(
            torch.cat((field.semantic_delta, geometry), dim=-1)
        ).reshape(batch, intervals, spatial, self.hidden)
        validity = field.validity.float().reshape(
            batch,
            intervals,
            spatial,
            1,
        )
        reliability = field.reliability.float().reshape(
            batch,
            intervals,
            spatial,
            1,
        )
        validity_masked_value = raw_value * validity.to(dtype=raw_value.dtype)
        value = validity_masked_value * reliability.to(
            dtype=raw_value.dtype
        )
        intent_key = F.normalize(
            self.intent_key(intent.interval_intents).float(),
            dim=-1,
            eps=1e-4,
        )
        content_score = torch.einsum(
            "btkh,bsnh->btksn",
            query,
            key,
        ).clamp(-1.0, 1.0)
        intent_score = torch.einsum(
            "btkh,bsh->btks",
            query,
            intent_key,
        ).clamp(-1.0, 1.0)

        action_coordinate = torch.tanh(
            self.action_coordinate(query_tokens).float()
        )
        transported_coordinate = (
            field.source_coordinates[:, None] + field.transport_delta
        ).clamp(-1.0, 1.0)
        support_coordinate = transported_coordinate.float().reshape(
            batch,
            intervals,
            spatial,
            2,
        )
        # Both coordinates lie in [-1,1].  Their squared 2-D distance is at
        # most eight, so 1 - distance/4 is construction-bounded to [-1,1].
        coordinate_distance = (
            action_coordinate[:, :, :, None, None]
            - support_coordinate[:, None, None]
        ).square().sum(dim=-1)
        coordinate_score = (1.0 - 0.25 * coordinate_distance).clamp(
            -1.0,
            1.0,
        )

        temperature = self.temperatures.to(device=query_tokens.device)
        logits = (
            temperature[0] * content_score
            + temperature[1] * intent_score[..., None]
            + temperature[2] * coordinate_score
        )
        valid = field.validity.reshape(batch, intervals, spatial).float()
        valid_flat = valid.reshape(batch, -1)
        valid_any = valid_flat.gt(0.0).any(dim=-1, keepdim=True)
        safe_valid_flat = torch.where(
            valid_any,
            valid_flat.gt(0.0),
            F.one_hot(
                torch.zeros(batch, device=valid.device, dtype=torch.long),
                num_classes=int(valid_flat.shape[-1]),
            ).to(dtype=torch.bool),
        )
        safe_valid = safe_valid_flat.reshape(batch, intervals, spatial)
        logits = logits.masked_fill(
            ~safe_valid[:, None, None],
            torch.finfo(logits.dtype).min,
        )
        posterior = torch.softmax(
            logits.reshape(batch, self.horizon, self.basis, -1),
            dim=-1,
        ).reshape(batch, self.horizon, self.basis, intervals, spatial)
        read = torch.einsum(
            "btksn,bsnh->btkh",
            posterior.to(dtype=value.dtype),
            value,
        )
        metrics: dict[str, Tensor] = {}
        if collect_diagnostics:
            posterior_f = posterior.detach().float()
            flattened = posterior_f.reshape(batch, self.horizon, self.basis, -1)
            metrics = {
                "grounded_p2_effect_read_rms": read.detach()
                .float()
                .square()
                .mean()
                .sqrt(),
                "grounded_p2_effect_value_pre_mask_rms": raw_value.detach()
                .float()
                .square()
                .mean()
                .sqrt(),
                "grounded_p2_effect_value_post_validity_rms": (
                    validity_masked_value.detach()
                    .float()
                    .square()
                    .mean()
                    .sqrt()
                ),
                "grounded_p2_effect_value_post_reliability_rms": (
                    value.detach().float().square().mean().sqrt()
                ),
                "grounded_p2_effect_reliability_valid_mean": (
                    (reliability.detach() * validity.detach()).sum()
                    / validity.detach().sum().clamp_min(1.0)
                ),
                "grounded_p2_effect_reliability_attenuation_ratio": (
                    value.detach().float().square().mean().sqrt()
                    / validity_masked_value.detach()
                    .float()
                    .square()
                    .mean()
                    .sqrt()
                    .clamp_min(1e-8)
                ),
                "grounded_p2_content_score_abs_max": content_score.detach()
                .abs()
                .amax(),
                "grounded_p2_intent_score_abs_max": intent_score.detach()
                .abs()
                .amax(),
                "grounded_p2_coordinate_score_abs_max": coordinate_score.detach()
                .abs()
                .amax(),
                "grounded_p2_query_coordinate_std": action_coordinate.detach()
                .std(dim=(1, 2), unbiased=False)
                .mean(),
                "grounded_p2_posterior_max": flattened.amax(dim=-1).mean(),
                "grounded_p2_posterior_entropy": _normalized_entropy(
                    flattened,
                    dim=-1,
                ).mean(),
                "grounded_p2_content_temperature": temperature[0].detach(),
                "grounded_p2_intent_temperature": temperature[1].detach(),
                "grounded_p2_coordinate_temperature": temperature[2].detach(),
            }
            interval_mass = posterior_f.sum(dim=-1).mean(dim=(0, 1, 2))
            for index, name in enumerate(INTERVAL_NAMES):
                metrics[f"grounded_p2_{name}_mass"] = interval_mass[index]
        return read, metrics


@dataclass(frozen=True)
class ConsequencePlanState:
    factual_base: Tensor
    effect: Tensor
    interaction: Tensor
    protected_consequence: Tensor

    @property
    def effect_base(self) -> Tensor:
        return self.effect

    @property
    def organized_delta(self) -> Tensor:
        return self.interaction

    @property
    def protected_base(self) -> Tensor:
        return self.protected_consequence

    def validate(self) -> None:
        shape = tuple(self.protected_consequence.shape)
        if len(shape) != 4:
            raise ValueError("consequence state must be [B,T,K,H]")
        for name in ("factual_base", "effect", "interaction"):
            value = getattr(self, name)
            if tuple(value.shape) != shape:
                raise ValueError(f"consequence {name} is misaligned")


class ZeroPreservingConsequenceOrganizer(nn.Module):
    """Algebraic identity when the supervised P2 effect is neutral."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.effect_projection = nn.Linear(hidden, hidden, bias=False)
        self.factual_projection = nn.Linear(hidden, hidden, bias=False)
        self.interaction_projection = nn.Linear(hidden, hidden, bias=False)
        nn.init.normal_(self.effect_projection.weight, mean=0.0, std=3e-3)
        nn.init.normal_(self.interaction_projection.weight, mean=0.0, std=3e-3)

    def forward(
        self,
        *,
        factual_base: Tensor,
        effect_read: Tensor,
    ) -> tuple[ConsequencePlanState, dict[str, Tensor]]:
        if tuple(factual_base.shape) != tuple(effect_read.shape):
            raise ValueError("consequence factual and effect tensors must align")
        effect, effect_scale = smooth_rms_contract(
            self.effect_projection(effect_read),
            0.35,
        )
        interaction, interaction_scale = smooth_rms_contract(
            self.interaction_projection(
                torch.tanh(self.factual_projection(factual_base)) * effect
            ),
            0.25,
        )
        state = ConsequencePlanState(
            factual_base=factual_base,
            effect=effect,
            interaction=interaction,
            protected_consequence=factual_base + effect + interaction,
        )
        state.validate()
        return state, {
            "grounded_consequence_effect_rms": effect.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "grounded_consequence_interaction_rms": interaction.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "grounded_consequence_effect_contract_min": effect_scale.detach()
            .float()
            .amin(),
            "grounded_consequence_interaction_contract_min": interaction_scale.detach()
            .float()
            .amin(),
        }


@dataclass(frozen=True)
class ExecutionTerminalEvidence:
    probability: Tensor
    uncertainty: Tensor

    def validate(self, batch: int) -> None:
        if tuple(self.probability.shape) != (batch, 1):
            raise ValueError("terminal probability must be [B,1]")
        if tuple(self.uncertainty.shape) != (batch, 1):
            raise ValueError("terminal uncertainty must be [B,1]")


@dataclass(frozen=True)
class PolicyPlanDeltaBank:
    protected_base: Tensor
    precision: Tensor
    temporal: Tensor
    execution_terminal: ExecutionTerminalEvidence

    @property
    def source_names(self) -> tuple[str, str]:
        return ("p3_precision", "p3_temporal")

    def validate(self) -> None:
        shape = tuple(self.protected_base.shape)
        if len(shape) != 4:
            raise ValueError("grounded policy plan must be [B,T,K,H]")
        for name in ("precision", "temporal"):
            value = getattr(self, name)
            if tuple(value.shape) != shape:
                raise ValueError(f"grounded policy plan {name} is misaligned")
        self.execution_terminal.validate(int(shape[0]))

    def as_policy_role_bank(self, *, source_depth: int) -> PolicyRoleDeltaBank:
        self.validate()
        return PolicyRoleDeltaBank(
            values=torch.stack((self.precision, self.temporal), dim=1),
            source_names=self.source_names,
            source_depths=(int(source_depth), int(source_depth)),
            protected_detail=self.protected_base,
        )


class ConsequenceConditionedPolicyPlanCompiler(nn.Module):
    """P3 lanes that both explicitly consume consequence and action query."""

    def __init__(self, *, hidden: int, horizon: int, basis: int) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.horizon = int(horizon)
        self.basis = int(basis)
        self.precision_lane = nn.Sequential(
            nn.LayerNorm(4 * hidden, elementwise_affine=False),
            nn.Linear(4 * hidden, 2 * hidden, bias=False),
            nn.SiLU(),
            nn.Linear(2 * hidden, hidden, bias=False),
        )
        self.temporal_lane = nn.Sequential(
            nn.LayerNorm(3 * hidden, elementwise_affine=False),
            nn.Linear(3 * hidden, 2 * hidden, bias=False),
            nn.SiLU(),
            nn.Linear(2 * hidden, hidden, bias=False),
        )
        for module in (self.precision_lane[-1], self.temporal_lane[-1]):
            output = cast(nn.Linear, module)
            nn.init.normal_(output.weight, mean=0.0, std=3e-3)

    def forward(
        self,
        *,
        p1_delta: Tensor,
        protected_detail: Tensor,
        consequence: ConsequencePlanState,
        intent: StatelessIntentState,
        action_query: Tensor,
    ) -> tuple[PolicyPlanDeltaBank, dict[str, Tensor]]:
        consequence.validate()
        expected = (
            int(p1_delta.shape[0]),
            self.horizon,
            self.basis,
            self.hidden,
        )
        for name, value in (
            ("P1 delta", p1_delta),
            ("protected detail", protected_detail),
            ("consequence", consequence.protected_consequence),
            ("action query", action_query),
        ):
            if tuple(value.shape) != expected:
                raise ValueError(f"grounded P3 {name} must be {expected}")
        intent.validate(batch=expected[0], hidden=self.hidden, horizon=self.horizon)
        temporal_control = intent.temporal_control[:, :, None].expand(
            -1,
            -1,
            self.basis,
            -1,
        )
        precision, precision_scale = smooth_rms_contract(
            self.precision_lane(
                torch.cat(
                    (
                        protected_detail,
                        p1_delta,
                        consequence.protected_consequence,
                        action_query,
                    ),
                    dim=-1,
                )
            ),
            0.35,
        )
        temporal, temporal_scale = smooth_rms_contract(
            self.temporal_lane(
                torch.cat(
                    (
                        consequence.protected_consequence,
                        temporal_control,
                        action_query,
                    ),
                    dim=-1,
                )
            ),
            0.35,
        )
        bank = PolicyPlanDeltaBank(
            protected_base=consequence.protected_consequence,
            precision=precision,
            temporal=temporal,
            execution_terminal=ExecutionTerminalEvidence(
                probability=intent.completion_probability,
                uncertainty=intent.completion_uncertainty,
            ),
        )
        bank.validate()
        return bank, {
            "grounded_p3_protected_consequence_rms": consequence.protected_consequence.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "grounded_p3_precision_rms": precision.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "grounded_p3_temporal_rms": temporal.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "grounded_p3_precision_contract_min": precision_scale.detach()
            .float()
            .amin(),
            "grounded_p3_temporal_contract_min": temporal_scale.detach()
            .float()
            .amin(),
        }


def manifest_from_mapping(value: Mapping[str, object]) -> ArchitectureManifest:
    """Load only the small architecture identity, never a historical contract."""

    def integer(raw: object, name: str) -> int:
        if not isinstance(raw, (int, str)):
            raise ValueError(f"grounded manifest {name} must be an integer")
        return int(raw)

    topology_value = value.get("topology", ())
    if not isinstance(topology_value, (tuple, list)):
        raise ValueError("grounded manifest topology must be a sequence")
    topology_items = cast(tuple[object, ...] | list[object], topology_value)
    interval_value = value.get("intervals", ())
    if not isinstance(interval_value, (tuple, list)):
        raise ValueError("grounded manifest intervals must be a sequence")
    interval_items = cast(tuple[object, ...] | list[object], interval_value)
    intervals: list[tuple[int, ...]] = []
    for index, interval in enumerate(interval_items):
        if not isinstance(interval, (tuple, list)):
            raise ValueError(
                f"grounded manifest interval {index} must be a sequence"
            )
        interval_sequence = cast(tuple[object, ...] | list[object], interval)
        intervals.append(
            tuple(
                integer(item, f"intervals[{index}]")
                for item in interval_sequence
            )
        )
    topology = tuple(integer(item, "topology") for item in topology_items)
    if len(topology) != 3:
        raise ValueError("grounded manifest topology must have three entries")
    if any(len(interval) != 2 for interval in intervals):
        raise ValueError("grounded manifest intervals must contain pairs")
    manifest = ArchitectureManifest(
        capability=str(value.get("capability", "")),
        schema=integer(value.get("schema", -1), "schema"),
        topology=cast(tuple[int, int, int], topology),
        intervals=cast(tuple[tuple[int, int], ...], tuple(intervals)),
        language_required=bool(value.get("language_required", False)),
        bottom_compatibility=str(value.get("bottom_compatibility", "")),
    )
    manifest.validate()
    return manifest
