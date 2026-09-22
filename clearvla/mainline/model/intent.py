"""Recovered V120 stateless intent, plan recognition and coarse action."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..supervision import FutureLabelSupport, quarantine, supported_mean
from ..temporal import LEGACY_HISTORY_ENCODING, TIMED_HISTORY_ENCODING, HistoryTiming
from .routing import register_gradient_rms_metric, smooth_rms_contract
from .target_binding import (
    LOCAL_TARGET_READERS,
    SHARED_TARGET_BINDING,
    BoundTargetRead,
    SharedTargetBinder,
    TargetBinding,
    TargetEvidence,
    masked_probability,
)
from .timed_history import TimedHistoryEncoder
from .types import (
    INTERVAL_BOUNDS,
    ActionIntentDock,
    CoarseActionIntentState,
    FutureObjectDynamics,
    FuturePlanRecognition,
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


def _interval_common_residual(value: Tensor) -> tuple[Tensor, Tensor]:
    """Express four interval rows as one common value plus exact residuals."""

    if value.ndim < 2 or int(value.shape[1]) != 4:
        raise ValueError("S decomposition requires four interval rows")
    common = value.mean(dim=1)
    residual = value - common[:, None]
    return common, residual


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
        # ``MultiheadAttention`` returns NaNs when every key in one batch row
        # is masked.  More importantly, normalizing a padded/invalid key before
        # masking still lets a malformed cache leak through the reduction.  A
        # dummy zero key keeps the kernel finite for the all-invalid row; its
        # update is discarded below so that the row is an exact identity read.
        all_invalid: Tensor | None = None
        effective_padding_mask = padding_mask
        if padding_mask is not None:
            if padding_mask.ndim != 2 or tuple(padding_mask.shape) != tuple(
                memory.shape[:2]
            ):
                raise ValueError("cross-read padding mask must align with memory")
            if int(memory.shape[1]) < 1:
                raise ValueError("cross-read memory cannot be empty")
            effective_padding_mask = padding_mask.to(
                device=memory.device,
                dtype=torch.bool,
            )
            valid = ~effective_padding_mask
            all_invalid = ~valid.any(dim=-1)
            safe_memory = torch.where(
                valid[..., None],
                memory,
                torch.zeros_like(memory),
            )
            normalized_memory = self.memory_norm(safe_memory)
            # Unmask one zero key only for rows with no legal key.  The
            # resulting attention weight is removed after the kernel call.
            effective_padding_mask = effective_padding_mask.clone()
            effective_padding_mask[all_invalid, 0] = False
        else:
            normalized_memory = self.memory_norm(memory)
        update, weights = self.attention(
            self.query_norm(query),
            normalized_memory,
            normalized_memory,
            key_padding_mask=effective_padding_mask,
            need_weights=diagnostics,
            average_attn_weights=True,
        )
        update, _ = smooth_rms_contract(update, self.maximum_rms)
        value = query + update
        ffn, _ = smooth_rms_contract(self.ffn(value), self.maximum_rms)
        innovation = update + ffn
        value = value + ffn
        if all_invalid is not None:
            invalid = all_invalid[:, None, None]
            # No valid object/evidence is an exact no-op, including its FFN
            # residual.  This is intentionally a value-level mask rather than
            # a detached loss-side convention.
            value = torch.where(invalid, query, value)
            innovation = torch.where(invalid, torch.zeros_like(innovation), innovation)
            if weights is not None:
                weights = torch.where(
                    all_invalid[:, None, None],
                    torch.zeros_like(weights),
                    weights,
                )
        if weights is None:
            weights = query.new_zeros(query.shape[0], query.shape[1], memory.shape[1])
        return value, innovation, weights


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

    def forward(
        self,
        value: Tensor,
        *,
        causal: bool = False,
        valid: Tensor | None = None,
    ) -> Tensor:
        mask = _causal_mask(int(value.shape[1]), value.device) if causal else None
        if valid is not None:
            batch, length = valid.shape
            if tuple(value.shape[:2]) != (batch, length) or valid.dtype != torch.bool:
                raise ValueError("self-read support must be bool [B,L]")
            value = quarantine(value, valid)
            allowed = valid[:, None, :].expand(-1, length, -1)
            if mask is not None:
                allowed = allowed & ~mask[None]
            own_zero = torch.eye(length, device=value.device, dtype=torch.bool)[None]
            allowed = torch.where(valid[:, :, None], allowed, own_zero)
            mask = (~allowed)[:, None].expand(-1, self.attention.num_heads, -1, -1)
            mask = mask.reshape(batch * self.attention.num_heads, length, length)
        normalized = self.norm(value)
        update, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=mask,
            need_weights=False,
        )
        update, _ = smooth_rms_contract(update, 0.35)
        value = value + update
        if valid is not None:
            value = quarantine(value, valid)
        ffn, _ = smooth_rms_contract(self.ffn(value), 0.35)
        return value + ffn if valid is None else quarantine(value + ffn, valid)


class _ZeroStartTargetAddress(nn.Module):
    """Bias-free shared type-to-K address head without constructor RNG use."""

    def __init__(self, width: int) -> None:
        super().__init__()
        if int(width) <= 0:
            raise ValueError("target-address width must be positive")
        self.weight = nn.Parameter(torch.zeros(1, int(width), dtype=torch.float32))

    def forward(self, value: Tensor) -> Tensor:
        if int(value.shape[-1]) != int(self.weight.shape[-1]):
            raise ValueError("target-address input lost typed evidence width")
        # Elementwise FP32 arithmetic is deliberately used instead of an
        # autocast-eligible GEMM.  Equal typed evidence must remain an exact
        # K-common value before the centering contract below.
        return (value.float() * self.weight.float()).sum(dim=-1, keepdim=True)


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
        target_object_address_mode: str = "post_pool_only",
        history_encoding_mode: str = LEGACY_HISTORY_ENCODING,
        target_binding_mode: str = LOCAL_TARGET_READERS,
        camera_names: tuple[str, ...] = ("top", "wrist"),
    ) -> None:
        super().__init__()
        if target_object_address_mode not in {
            "post_pool_only",
            "shared_target_prior_v1",
        }:
            raise ValueError("unknown target object address mode")
        if target_binding_mode not in {LOCAL_TARGET_READERS, SHARED_TARGET_BINDING}:
            raise ValueError("unknown target binding mode")
        if target_binding_mode == SHARED_TARGET_BINDING and target_object_address_mode != "post_pool_only":
            raise ValueError("shared binding cannot duplicate the additive target address")
        self.target_binding_mode = target_binding_mode
        self.camera_names = tuple(camera_names)
        self.target_object_address_mode = target_object_address_mode
        self.hidden = int(hidden)
        self.horizon = int(horizon)
        self.goal_input = nn.Linear(goal_dim, hidden, bias=False)
        self.goal_queries = nn.Parameter(torch.randn(1, 4, hidden) * 0.02)
        self.goal_read = _CrossRead(hidden, heads)
        self.goal_self = _SelfBlock(hidden, heads)
        self.history_encoding_mode = history_encoding_mode
        self.timed_history: TimedHistoryEncoder | None = None
        self.history_input: nn.Sequential | None = None
        self.history_blocks: nn.ModuleList
        if history_encoding_mode == LEGACY_HISTORY_ENCODING:
            history_width = state_dim + action_dim + state_dim + 1
            self.history_input = nn.Sequential(
                nn.LayerNorm(history_width, elementwise_affine=False),
                nn.Linear(history_width, hidden, bias=False),
            )
            self.history_blocks = nn.ModuleList(_SelfBlock(hidden, heads) for _ in range(2))
        elif history_encoding_mode == TIMED_HISTORY_ENCODING:
            self.history_blocks = nn.ModuleList()
            self.timed_history = TimedHistoryEncoder(
                state_dim=state_dim, action_dim=action_dim, hidden=hidden, heads=heads
            )
        else:
            raise ValueError("unknown history_encoding_mode")
        self.object_content = nn.Linear(content_dim, hidden, bias=False)
        self.object_semantic = nn.Linear(route_dim, hidden, bias=False)
        self.object_appearance = nn.Linear(route_dim, hidden, bias=False)
        self.object_geometry = nn.Linear(route_dim, hidden, bias=False)
        self.interval_identity = nn.Parameter(torch.randn(1, 4, hidden) * 0.02)
        self.interval_goal = _CrossRead(hidden, heads)
        self.interval_history = _CrossRead(hidden, heads)
        self.interval_object = (
            BoundTargetRead(hidden, heads) if target_binding_mode == SHARED_TARGET_BINDING
            else _CrossRead(hidden, heads)
        )
        self.typed_query_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.typed_relevance_queries = nn.ModuleList(
            nn.Linear(hidden, route_dim, bias=False) for _ in TYPED_INTENT_NAMES
        )
        # One shared object identity must bind semantic and geometry reads.
        # The head learns which typed evidence identifies the operated object
        # (including appearance for color instructions) without creating one
        # target per downstream reader.  Exact zero initialization preserves
        # the legacy spatial reader at initialization and lets ordinary action
        # loss open the route without a hand-set gain.
        self.target_object_address: _ZeroStartTargetAddress | None
        if target_object_address_mode == "shared_target_prior_v1":
            self.target_object_address = _ZeroStartTargetAddress(
                len(TYPED_INTENT_NAMES)
            )
        else:
            # The accepted reader must retain the exact old parameter set and
            # constructor RNG stream.  Runtime still exposes a typed zero field
            # so the downstream dock has one stable schema.
            self.target_object_address = None
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
        self.shared_binder: SharedTargetBinder | None = None
        self.target_coordinate: nn.Linear | None = None
        self.target_state: nn.Linear | None = None
        self.target_view: nn.Embedding | None = None
        if target_binding_mode == SHARED_TARGET_BINDING:
            if not camera_names or len(set(camera_names)) != len(camera_names):
                raise ValueError("target evidence requires unique declared cameras")
            self.shared_binder = SharedTargetBinder(hidden)
            self.target_coordinate = nn.Linear(2, hidden, bias=False)
            self.target_state = nn.Linear(state_dim, hidden, bias=False)
            self.target_view = nn.Embedding(len(camera_names), hidden)

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

    def _object_tokens(self, facts: ObjectFactSet) -> Tensor:
        facts.validate()
        validity = torch.nan_to_num(
            facts.validity.to(device=facts.content.device, dtype=torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        support = validity[..., 0] > 0.0
        content = torch.where(
            support[..., None],
            facts.content,
            torch.zeros_like(facts.content),
        )
        return self.object_content(content)

    @staticmethod
    def _bounded_unit(value: Tensor, *, floor: float = 0.25) -> Tensor:
        value_f = value.float()
        return value_f / (
            value_f.square().sum(dim=-1, keepdim=True) + float(floor) ** 2
        ).sqrt()

    def _typed_relevance(
        self,
        *,
        public_interval_carrier: Tensor,
        facts: ObjectFactSet,
        binding: TargetBinding | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Build fixed-zero, per-type relevance without collapsing K."""

        query_source = self.typed_query_norm(public_interval_carrier)
        typed_query = torch.stack(
            tuple(projection(query_source) for projection in self.typed_relevance_queries),
            dim=2,
        )  # [B,I,type,R]
        validity_values = torch.nan_to_num(
            facts.validity.to(device=public_interval_carrier.device, dtype=torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        object_support = validity_values[..., 0] > 0.0
        typed_route = torch.stack(
            (facts.semantic, facts.appearance, facts.geometry), dim=2
        )  # [B,K,type,R]
        # Quarantine invalid producer rows before the bounded cosine and the
        # subsequent type/object reductions.  Multiplying after a projection
        # would still allow NaN * 0 to poison the upstream VJP.
        typed_route = torch.where(
            object_support[..., None, None],
            typed_route,
            torch.zeros_like(typed_route),
        )
        score = torch.einsum(
            "bitr,bktr->bikt",
            self._bounded_unit(typed_query),
            self._bounded_unit(typed_route),
        ).clamp(-1.0, 1.0)
        temperature = 0.25 + 3.75 * torch.sigmoid(
            self.typed_temperature_logit.float()
        )
        signal_probability = torch.sigmoid(
            score * temperature.to(device=score.device)[None, None, None]
        )
        # Target identity is one shared K-indexed fact, not one independently
        # chosen object per downstream P2 type.  The shared zero-start head can
        # learn whether semantic, appearance or geometry identifies the target.
        # Removing the supported K-common component makes uniform evidence an
        # exact neutral prior and leaves producer support authority unchanged.
        typed_address_score = score * temperature.to(device=score.device)[
            None, None, None
        ]
        if self.target_object_address is None:
            target_address_raw = torch.zeros_like(typed_address_score[..., 0])
        else:
            target_address_raw = self.target_object_address(typed_address_score)[..., 0]
        target_support = object_support[:, None].expand_as(target_address_raw)
        safe_target_address = torch.where(
            target_support,
            target_address_raw.float(),
            torch.zeros_like(target_address_raw, dtype=torch.float32),
        )
        first_legal_k = target_support.to(dtype=torch.int64).argmax(
            dim=-1, keepdim=True
        )
        target_address_reference = safe_target_address.gather(-1, first_legal_k)
        target_address_relative = safe_target_address - target_address_reference
        target_count = target_support.float().sum(dim=-1, keepdim=True)
        target_address_mean = torch.where(
            target_support,
            target_address_relative,
            torch.zeros_like(target_address_relative),
        ).sum(dim=-1, keepdim=True) / target_count.clamp_min(1.0)
        target_object_address_logit = torch.where(
            target_support,
            target_address_relative - target_address_mean,
            torch.zeros_like(target_address_relative),
        )
        if binding is not None:
            # Operation/type gates can vary by future interval, but not select
            # a different K for each type. The single target law owns K.
            operation_score = (typed_address_score * binding.mass[:, None, :, None]).sum(dim=2)
            signal_probability = binding.mass[:, None, :, None] * torch.sigmoid(operation_score)[:, :, None]
        validity = (validity_values if binding is None else object_support[..., None].float())[:, None, :, None, :]
        relevance_mass = signal_probability[..., None] * validity
        relevance_mass = relevance_mass.to(dtype=typed_route.dtype)
        relevance_value = (
            relevance_mass * typed_route[:, None].to(dtype=relevance_mass.dtype)
        )

        components: list[Tensor] = []
        for type_index, projection in enumerate(
            (self.object_semantic, self.object_appearance, self.object_geometry)
        ):
            # K is a fixed identity axis.  A fixed mean preserves zero and
            # cannot cancel the optionality by renormalizing selected mass.
            selected_route = (
                relevance_value[..., type_index, :].mean(dim=2) if binding is None
                else relevance_value[..., type_index, :].sum(dim=2)
            )
            component, _ = smooth_rms_contract(projection(selected_route), 0.35)
            components.append(component)
        typed_components = torch.stack(components, dim=2)
        raw_context = typed_components.sum(dim=2) / (3.0**0.5)
        _, context_scale = smooth_rms_contract(raw_context, 0.35)
        typed_components = typed_components * context_scale[:, :, None].to(
            dtype=typed_components.dtype
        )
        return (
            relevance_mass,
            relevance_value,
            typed_components,
            target_object_address_logit,
            score,
            temperature,
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
        history_timing: HistoryTiming | None = None,
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
        history_validity = None
        rate_validity = None
        if self.timed_history is None:
            if self.history_input is None:
                raise RuntimeError("legacy history encoder is absent")
            paired_history, observed_state_delta = self._paired_history(
                state_history, state, executed_history
            )
            history = self.history_input(paired_history)
            for block in self.history_blocks:
                history = block(history, causal=True)
        else:
            if history_timing is None:
                raise ValueError("timestamped history requires explicit producer timing")
            encoded = self.timed_history(state_history, state, executed_history, history_timing)
            history = encoded.tokens
            history_validity = encoded.valid
            observed_state_delta = encoded.state_rates
            rate_validity = encoded.rate_valid
        # Materialize the producer-owned object support before any learned
        # projection.  This keeps invalid/unknown rows out of both the S
        # language read and the backward graph.
        validity_values = torch.nan_to_num(
            facts.validity.to(device=facts.content.device, dtype=torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        object_validity = validity_values[..., 0] > 0.0
        object_mask = object_validity[..., None]
        content_input = torch.where(
            object_mask,
            facts.content,
            torch.zeros_like(facts.content),
        )
        semantic_input = torch.where(
            object_mask,
            facts.semantic,
            torch.zeros_like(facts.semantic),
        )
        appearance_input = torch.where(
            object_mask,
            facts.appearance,
            torch.zeros_like(facts.appearance),
        )
        geometry_input = torch.where(
            object_mask,
            facts.geometry,
            torch.zeros_like(facts.geometry),
        )
        content_objects = self.object_content(content_input)
        # Fuse the three observable object routes before the language read.
        # The sum is shared over K and symmetric over semantic/appearance/
        # geometry, so it supplies affordance/appearance evidence without
        # creating a color-specific slot or a learned role sidecar.  Keep the
        # update bounded and quarantine producer-invalid rows before any
        # attention reduction.
        typed_object_context = (
            self.object_semantic(semantic_input)
            + self.object_appearance(appearance_input)
            + self.object_geometry(geometry_input)
        ) / (3.0**0.5)
        typed_object_context, _ = smooth_rms_contract(typed_object_context, 0.35)
        # Keep the original content-only memory as the interval evidence
        # source so typed S relevance remains owned by its named routes.  The
        # symmetric route-enriched view is exported as the public K memory for
        # the language-conditioned coarse compiler below; this prevents an
        # appearance/geometry helper from becoming a second public interval
        # owner while still making those facts available to action selection.
        objects = content_objects + typed_object_context
        interval_objects = content_objects
        # Keep the public K memory physically safe before it reaches either
        # S's language-conditioned read or the coarse-action compiler.  The
        # producer-owned validity mask is not a learned confidence and is not
        # renormalized into a task-dependent selector.
        objects = torch.where(
            object_validity[..., None],
            objects,
            torch.zeros_like(objects),
        )
        interval_objects = torch.where(
            object_validity[..., None],
            interval_objects,
            torch.zeros_like(interval_objects),
        )
        target_binding = None
        target_evidence = None
        if self.shared_binder is not None:
            if self.target_coordinate is None or self.target_view is None or self.target_state is None:
                raise RuntimeError("shared target evidence encoder is incomplete")
            if facts.camera_coordinates.shape[2] != len(self.camera_names):
                raise ValueError("target facts lost declared camera axis")
            history_support = torch.ones(history.shape[:2], dtype=torch.bool, device=history.device) if history_validity is None else history_validity
            safe_history = torch.where(history_support[..., None], history, torch.zeros_like(history))
            history_context = safe_history.sum(1) / history_support.sum(1, keepdim=True).clamp_min(1)
            target_binding = self.shared_binder(protected_goal.mean(1) + history_context, objects, object_validity)
            camera_valid = (facts.camera_validity[..., 0] > 0) & object_validity[..., None]
            camera_positions = torch.where(camera_valid[..., None], facts.camera_coordinates, torch.zeros_like(facts.camera_coordinates))
            view_ids = torch.arange(len(self.camera_names), device=objects.device)
            # Native image coordinates are not subtracted from world TCP.
            # Declared view identity + robot state condition a learnable relation.
            camera_tokens = (
                self.target_coordinate(camera_positions.to(objects.dtype))
                + self.target_view(view_ids)[None, None].to(objects.dtype)
                + self.target_state(state).to(objects.dtype)[:, None, None]
            )
            attribute_tokens = torch.stack((content_objects, self.object_semantic(semantic_input),
                self.object_appearance(appearance_input), self.object_geometry(geometry_input)), dim=2)
            target_valid = torch.cat((object_validity[..., None].expand(-1, -1, 4), camera_valid), dim=-1)
            target_tokens = torch.cat((attribute_tokens, camera_tokens), dim=2)
            target_evidence = TargetEvidence(
                torch.where(target_valid[..., None], target_tokens, torch.zeros_like(target_tokens)), target_valid
            )
        interval_base = self.interval_identity.to(
            device=objects.device, dtype=objects.dtype
        ).expand(batch, -1, -1)
        _, goal_innovation, interval_goal_attention = self.interval_goal(
            interval_base, protected_goal, diagnostics=collect_diagnostics
        )
        _, history_innovation, interval_history_attention = self.interval_history(
            interval_base,
            history,
            diagnostics=collect_diagnostics,
            padding_mask=None if history_validity is None else ~history_validity,
        )
        # This is the repaired language/object seam.  The old query was only
        # ``interval_base``; consequently its K attention and object update
        # were invariant to an instruction swap.  Goal/history innovations are
        # already S-owned, bounded deltas, so using them as the object-read
        # query preserves the K axis while making object identity conditional
        # on the instruction and the current causal context.
        language_object_query = interval_base + goal_innovation + history_innovation
        if isinstance(self.interval_object, BoundTargetRead):
            if target_binding is None or target_evidence is None:
                raise RuntimeError("shared S read has no operated-object binding")
            object_innovation = self.interval_object(language_object_query, target_evidence, target_binding)
            interval_object_attention = target_binding.mass[:, None].expand(-1, 4, -1)
        else:
            _, object_innovation, interval_object_attention = self.interval_object(
                language_object_query, interval_objects, padding_mask=~object_validity,
                diagnostics=collect_diagnostics,
            )
        public_intervals = self.interval_self(
            interval_base
            + goal_innovation
            + history_innovation
            + object_innovation
        )
        (
            typed_relevance_mass,
            typed_relevance_value,
            typed_policy_components,
            target_object_address_logit,
            typed_relevance_score,
            typed_temperature,
        ) = self._typed_relevance(
            public_interval_carrier=public_intervals,
            facts=facts,
            binding=target_binding,
        )
        typed_common_mass, typed_interval_residual_mass = (
            _interval_common_residual(typed_relevance_mass)
        )
        typed_common_value, typed_interval_residual_value = (
            _interval_common_residual(typed_relevance_value)
        )
        typed_policy_context = typed_policy_components.sum(dim=2) / (3.0**0.5)
        policy_intervals = public_intervals + typed_policy_context
        temporal_base = self.temporal_identity.to(
            device=public_intervals.device, dtype=public_intervals.dtype
        ).expand(batch, -1, -1)
        temporal, _, _ = self.temporal_read(
            temporal_base, public_intervals, diagnostics=False
        )
        state_change_values = self.state_change_input(observed_state_delta)
        rate_padding = None
        no_rate = None
        if rate_validity is not None:
            no_rate = ~rate_validity.any(dim=1)
            rate_padding = ~rate_validity.clone()
            # Zero value sentinel for a reset with no observed state interval.
            rate_padding[no_rate, -1] = False
        state_change_history, state_change_attention = self.state_change_read(
            self.state_change_query_norm(
                self.state_change_query.to(device=history.device, dtype=history.dtype).expand(
                    batch, -1, -1
                )
            ),
            self.state_change_key_norm(history),
            state_change_values,
            key_padding_mask=rate_padding,
            need_weights=collect_diagnostics,
            average_attn_weights=True,
        )
        if state_change_attention is None:
            state_change_attention = history.new_zeros(batch, 1, history.shape[1])
        state_change_history = state_change_history[:, 0]
        if no_rate is not None:
            state_change_history = torch.where(
                no_rate[:, None], torch.zeros_like(state_change_history), state_change_history
            )
        if target_binding is None:
            observed_motion = facts.transport_prior if facts.latest_flow_steps is None else facts.transport_rate
            transport_prior = observed_motion.to(
                device=objects.device,
                dtype=objects.dtype,
            )
            transport_prior = torch.where(
                object_mask.to(device=transport_prior.device),
                transport_prior,
                torch.zeros_like(transport_prior),
            )
            transport_tokens = self.state_change_transport(transport_prior)
            transport_validity = validity_values.to(
                device=transport_tokens.device, dtype=transport_tokens.dtype
            )
            state_change_transport = (
                transport_tokens * transport_validity
            ).sum(dim=1) / transport_validity.sum(dim=1).clamp_min(1.0)
        else:
            if self.target_view is None:
                raise RuntimeError("target motion has no declared view identity")
            view_valid = (facts.camera_validity[..., 0] > 0) & object_validity[..., None]
            motion = facts.camera_transport_prior if facts.latest_flow_steps is None else facts.camera_transport_rate
            motion = torch.where(view_valid[..., None], motion, torch.zeros_like(motion))
            view_ids = torch.arange(len(self.camera_names), device=motion.device)
            # Map each view into a role-aware feature BEFORE view reduction;
            # a view embedding alone cannot manufacture motion when rate=0.
            view_motion = self.state_change_transport(motion.to(objects.dtype))
            view_motion = view_motion * (1 + torch.tanh(self.target_view(view_ids)))[None, None]
            view_weights = masked_probability(facts.log_camera_validity[..., 0], view_valid)
            per_object_motion = (view_motion.float() * view_weights[..., None]).sum(2)
            state_change_transport = (per_object_motion * target_binding.mass[..., None]).sum(1).to(objects.dtype)
        state_change_evidence, _ = smooth_rms_contract(
            self.state_change_fuse(
                torch.cat((state_change_history, state_change_transport), dim=-1)
            ),
            0.20,
        )
        state_out = ObjectIntentState(
            protected_goal_set=protected_goal,
            history_tokens=history,
            history_validity=history_validity,
            object_tokens=objects,
            target_binding=target_binding,
            target_evidence=target_evidence,
            public_interval_carrier=public_intervals,
            policy_interval_context=policy_intervals,
            temporal_queries=temporal,
            state_change_evidence=state_change_evidence,
            target_object_address_logit=target_object_address_logit,
            typed_common_mass=typed_common_mass,
            typed_common_value=typed_common_value,
            typed_interval_residual_mass=typed_interval_residual_mass,
            typed_interval_residual_value=typed_interval_residual_value,
            typed_policy_components=typed_policy_components,
            goal_attention=goal_attention,
            interval_goal_attention=interval_goal_attention,
            interval_history_attention=interval_history_attention,
            interval_object_attention=interval_object_attention,
            object_validity=validity_values.detach().to(
                device=objects.device,
                dtype=torch.float32,
            ),
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
            "object_intent_language_object_attention_entropy": normalized_entropy(
                interval_object_attention, dim=-1
            ).detach().mean(),
            "object_intent_language_object_attention_valid_fraction": (
                object_validity.detach().float().mean()
            ),
            "object_intent_public_interval_variation": public_intervals.detach().float().std(
                dim=1, unbiased=False
            ).mean(),
            "object_intent_policy_interval_variation": policy_intervals.detach().float().std(
                dim=1, unbiased=False
            ).mean(),
            "object_intent_temporal_variation": temporal.detach().float().std(
                dim=1, unbiased=False
            ).mean(),
            "object_intent_goal_innovation_rms": goal_innovation.detach().float().square().mean().sqrt(),
            "object_intent_history_innovation_rms": history_innovation.detach().float().square().mean().sqrt(),
            "object_intent_object_innovation_rms": object_innovation.detach().float().square().mean().sqrt(),
            "object_intent_typed_action_context_rms": typed_policy_context.detach().float().square().mean().sqrt(),
            "object_intent_target_address_logit_rms": target_object_address_logit.detach().square().mean().sqrt(),
            "object_intent_target_address_logit_k_variation": target_object_address_logit.detach().std(
                dim=2, unbiased=False
            ).mean(),
            "object_intent_target_address_logit_k_center_error": (
                (
                    target_object_address_logit.detach()
                    * object_validity[:, None].detach().float()
                ).sum(dim=2)
                / object_validity.detach().float().sum(dim=1)[:, None].clamp_min(1.0)
            ).abs().amax(),
            "object_intent_observed_state_delta_rms": observed_state_delta.detach().float().square().mean().sqrt(),
            "object_intent_observed_transport_rms": transport_prior.detach().float().square().mean().sqrt(),
            "object_intent_state_change_history_rms": state_change_history.detach().float().square().mean().sqrt(),
            "object_intent_state_change_transport_rms": state_change_transport.detach().float().square().mean().sqrt(),
            "object_intent_state_change_evidence_rms": state_change_evidence.detach().float().square().mean().sqrt(),
            "object_intent_state_change_attention_entropy": normalized_entropy(
                state_change_attention, dim=-1
            ).detach().mean(),
        }
        if self.timed_history is not None:
            # A physical-step rate is not the legacy paired-row displacement.
            metrics["object_intent_observed_state_rate_per_step_rms"] = metrics.pop(
                "object_intent_observed_state_delta_rms"
            )
            assert history_validity is not None
            metrics["object_intent_timed_history_real_rows"] = (
                history_validity.float().sum(dim=1).mean()
            )
            metrics["object_intent_timed_history_active"] = history.new_ones(
                (), dtype=torch.float32
            )
        # The returned attention probability is an auxiliary observation; it
        # is not consumed by the action path and therefore has no ordinary
        # owner VJP.  Observe the two value tensors that actually carry the
        # repaired language/object edge instead: the conditioned K query and
        # its bounded object innovation.  Keep the historical attention name
        # as a compatibility alias, but bind it to the consumed innovation
        # rather than reporting a guaranteed zero from a sibling output.
        register_gradient_rms_metric(
            language_object_query,
            metrics,
            "gradient_tensor_s_language_object_query_rms",
        )
        register_gradient_rms_metric(
            object_innovation,
            metrics,
            "gradient_tensor_s_language_object_update_rms",
        )
        register_gradient_rms_metric(
            object_innovation,
            metrics,
            "gradient_tensor_s_language_object_attention_rms",
        )
        if self.target_object_address is not None:
            register_gradient_rms_metric(
                target_object_address_logit,
                metrics,
                "gradient_tensor_s_target_object_address_logit_rms",
            )
        raw_routes = (semantic_input, appearance_input, geometry_input)
        for type_index, name in enumerate(TYPED_INTENT_NAMES):
            mass = typed_relevance_mass[..., type_index, 0].detach().float()
            selected = typed_relevance_value[..., type_index, :].detach().float()
            component = typed_policy_components[..., type_index, :].detach().float()
            metrics.update(
                {
                    f"object_intent_{name}_route_raw_rms": raw_routes[type_index]
                    .detach()
                    .float()
                    .square()
                    .mean()
                    .sqrt(),
                    f"object_intent_{name}_relevance_mass": mass.mean(),
                    f"object_intent_{name}_null_mass": (1.0 - mass).mean(),
                    f"object_intent_{name}_selected_value_rms": selected.square()
                    .mean()
                    .sqrt(),
                    f"object_intent_{name}_object_variation": selected.std(
                        dim=2, unbiased=False
                    ).mean(),
                    f"object_intent_{name}_interval_variation": selected.std(
                        dim=1, unbiased=False
                    ).mean(),
                    f"object_intent_{name}_action_context_rms": component.square()
                    .mean()
                    .sqrt(),
                    f"object_intent_{name}_score_abs": typed_relevance_score[
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
        label_support: FutureLabelSupport | None = None,
        future_interval_valid: Tensor | None = None,
    ) -> FuturePlanRecognition:
        if future_action.ndim != 3 or future_state.ndim != 3:
            raise ValueError("plan recognizer requires full future action/state sequences")
        length = min(int(future_action.shape[1]), int(future_state.shape[1]))
        slices = _interval_slices(length)
        interval_valid = None
        if label_support is not None:
            future_action = quarantine(future_action, label_support.action)
            future_state = quarantine(future_state, label_support.state)
            interval_valid = torch.stack(
                [
                    label_support.action[:, row].all(dim=1) & label_support.state[:, row].all(dim=1)
                    for row in slices
                ],
                dim=1,
            )
            if future_interval_valid is not None:
                interval_valid = interval_valid & future_interval_valid
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
        if interval_valid is not None:
            action_summary = quarantine(action_summary, interval_valid)
            state_summary = quarantine(state_summary, interval_valid)
            effect_summary = quarantine(effect_summary, interval_valid)
            teacher_valid = quarantine(teacher_valid, interval_valid)
        token = (
            self.action_input(action_summary)
            + self.state_input(state_summary)
            + self.effect_input(effect_summary)
            + self.interval_identity.to(
                device=future_action.device, dtype=future_action.dtype
            )
        )
        token = self.block(token, valid=interval_valid)
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
            supported_mean(
                (action_pred.float() - action_summary.detach().float()).square(), interval_valid
            )
            + supported_mean(
                (state_pred.float() - state_summary.detach().float()).square(), interval_valid
            )
            + 0.25 * effect_error
        )
        result = FuturePlanRecognition(
            interval_targets=token.detach(),
            action_summary=action_summary.detach(),
            state_summary=state_summary.detach(),
            effect_summary=effect_summary.detach(),
            reconstruction_loss=reconstruction,
            interval_valid=interval_valid,
        )
        result.validate(hidden=self.hidden)
        return result


class CoarseActionIntent(nn.Module):
    """V120 online clean action intent used exactly once by W."""

    def __init__(
        self,
        *,
        hidden: int,
        action_dim: int,
        heads: int,
        horizon: int = 24,
        action_condition_mode: str = "interval_mean_v1",
        target_binding_mode: str = LOCAL_TARGET_READERS,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        if self.horizon != 24:
            raise ValueError("the active coarse action contract requires 24 rows")
        self.action_condition_mode = str(action_condition_mode)
        if self.action_condition_mode not in {
            "interval_mean_v1",
            "sequence_prefix_v1",
        }:
            raise ValueError("unknown coarse action condition mode")
        self.query = nn.Parameter(torch.randn(1, 4, hidden) * 0.02)
        # Keep the trained four-query basis as the shared temporal scaffold.
        # The new zero-initialized row offsets add row-specific capacity
        # without replacing or resetting the existing action head/query.
        self.sequence_row_offset: nn.Parameter | None = None
        if self.action_condition_mode == "sequence_prefix_v1":
            self.sequence_row_offset = nn.Parameter(
                torch.zeros(1, self.horizon, hidden)
            )
        self.intent_read = _CrossRead(hidden, heads)
        self.object_read = (
            BoundTargetRead(hidden, heads) if target_binding_mode == SHARED_TARGET_BINDING
            else _CrossRead(hidden, heads)
        )
        self.history_read = _CrossRead(hidden, heads)
        self.block = _SelfBlock(hidden, heads)
        self.action_head = nn.Linear(hidden, action_dim, bias=False)

    def forward(
        self,
        intent: ActionIntentDock,
        *,
        future_action: Tensor | None = None,
        future_action_valid: Tensor | None = None,
        collect_diagnostics: bool = False,
    ) -> CoarseActionIntentState:
        intent.validate(hidden=int(self.query.shape[-1]))
        batch = int(intent.public_interval_carrier.shape[0])
        query_seed = self.query
        if self.action_condition_mode == "sequence_prefix_v1":
            if self.sequence_row_offset is None:
                raise RuntimeError("sequence coarse action has no row offsets")
            query_seed = F.interpolate(
                self.query.transpose(1, 2),
                size=self.horizon,
                mode="linear",
                align_corners=True,
            ).transpose(1, 2) + self.sequence_row_offset
        query = query_seed.to(
            device=intent.public_interval_carrier.device,
            dtype=intent.public_interval_carrier.dtype,
        ).expand(batch, -1, -1)
        _, intent_delta, _ = self.intent_read(
            query,
            intent.public_interval_carrier,
            diagnostics=collect_diagnostics,
        )
        # Carry the language-conditioned interval read into both auxiliary
        # memories.  This keeps the object and history reads in the same S
        # query frame instead of letting a learned query silently wash out the
        # instruction before the physical proposal is formed.
        conditioned_query = query + intent_delta
        object_padding_mask = None
        if intent.public_object_validity is not None:
            validity = intent.public_object_validity.to(
                device=intent.public_object_memory.device,
                dtype=torch.float32,
            ).squeeze(-1).clamp(0.0, 1.0)
            object_padding_mask = ~(validity > 0.0)
        if isinstance(self.object_read, BoundTargetRead):
            if intent.target_binding is None or intent.target_evidence is None:
                raise ValueError("shared coarse read requires target binding and current facts")
            object_delta = self.object_read(conditioned_query, intent.target_evidence, intent.target_binding)
        else:
            if intent.target_binding is not None or intent.target_evidence is not None:
                raise ValueError("legacy coarse read cannot silently ignore target binding")
            _, object_delta, _ = self.object_read(
                conditioned_query, intent.public_object_memory,
                padding_mask=object_padding_mask, diagnostics=collect_diagnostics,
            )
        _, history_delta, _ = self.history_read(
            conditioned_query,
            intent.history_memory,
            padding_mask=None if intent.history_validity is None else ~intent.history_validity,
            diagnostics=collect_diagnostics,
        )
        token = self.block(
            conditioned_query
            + object_delta
            + history_delta
        )
        action_prediction = self.action_head(token)
        if future_action is None:
            target = None
            loss = action_prediction.new_zeros(())
        else:
            supervised_rows = future_action_valid
            if future_action_valid is not None:
                future_action = quarantine(future_action, future_action_valid)
            if self.action_condition_mode == "sequence_prefix_v1":
                if int(future_action.shape[1]) < self.horizon:
                    raise ValueError(
                        "sequence coarse supervision requires 24 future rows"
                    )
                target = future_action[:, : self.horizon].detach()
                if supervised_rows is not None:
                    supervised_rows = supervised_rows[:, : self.horizon]
            else:
                slices = _interval_slices(int(future_action.shape[1]))
                target = torch.stack(
                    [future_action[:, row].mean(dim=1) for row in slices], dim=1
                ).detach()
                if future_action_valid is not None:
                    supervised_rows = torch.stack(
                        [future_action_valid[:, row].all(dim=1) for row in slices], dim=1
                    )
            loss = supported_mean(
                (action_prediction.float() - target.float()).square(), supervised_rows
            )
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
