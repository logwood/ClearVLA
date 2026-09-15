"""Differential intent/effect ownership for the post-V117 3-2-3 mainline.

This module is intentionally isolated from the historical ``vXXX`` graph.
It contains the small set-valued intent state, the differentiated future
effect interface, and the consequence-aware policy boundary shared by the
new architecture.  Historical launchers keep their exact implementations in
``goal_conditioning.py``, ``flow_dino_evidence.py`` and ``trunk.py``.

The design follows four numerical and ownership rules:

* scalar progress is diagnostic only and never participates in a forward
  decision;
* current facts are protected references while W owns only changes;
* the exact intent/effect tensors decoded by representation losses are the
  tensors consumed by the deployed action path;
* optional innovations are smooth RMS-bounded and values that are required to
  preserve zero use bias-free projections.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .role_delta_attnres import PolicyRoleDeltaBank, smooth_rms_contract


def _ordered_basis(length: int, width: int) -> Tensor:
    """Return a deterministic ordered Fourier basis without trainable state."""

    if length < 1 or width < 1:
        raise ValueError("ordered basis dimensions must be positive")
    position = torch.linspace(0.0, 1.0, int(length), dtype=torch.float32)[:, None]
    frequency = torch.arange(1, int(width) // 2 + 1, dtype=torch.float32)[None]
    angle = math.pi * position * frequency
    basis = torch.cat((torch.sin(angle), torch.cos(angle)), dim=-1)
    if int(basis.shape[-1]) < int(width):
        basis = F.pad(basis, (0, int(width) - int(basis.shape[-1])))
    return basis[:, : int(width)]


def _resize_tokens(value: Tensor, steps: int) -> Tensor:
    if value.ndim != 3 or int(value.shape[1]) < 1:
        raise ValueError("token resize requires non-empty [B,N,H]")
    if int(value.shape[1]) == int(steps):
        return value
    return F.interpolate(
        value.float().transpose(1, 2),
        size=int(steps),
        mode="linear",
        align_corners=True,
    ).transpose(1, 2).to(dtype=value.dtype)


def _causal_mask(length: int, *, device: torch.device) -> Tensor:
    return torch.triu(
        torch.ones(length, length, device=device, dtype=torch.bool),
        diagonal=1,
    )


def _normalized_entropy(probability: Tensor, *, dim: int) -> Tensor:
    support = int(probability.shape[dim])
    if support < 2:
        return probability.new_zeros(
            tuple(
                size
                for index, size in enumerate(probability.shape)
                if index != (dim % probability.ndim)
            ),
            dtype=torch.float32,
        )
    value = probability.float().clamp_min(1e-8)
    return -(value * value.log()).sum(dim=dim) / math.log(float(support))


class _BoundedCrossResidual(nn.Module):
    """Pre-normalized typed cross-attention that returns its true innovation."""

    def __init__(
        self,
        hidden: int,
        heads: int,
        *,
        max_update_rms: float,
    ) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.memory_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(
            hidden,
            heads,
            dropout=0.0,
            bias=False,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, 2 * hidden, bias=False),
            nn.SiLU(),
            nn.Linear(2 * hidden, hidden, bias=False),
        )
        self.max_update_rms = float(max_update_rms)

    def forward(
        self,
        state: Tensor,
        memory: Tensor,
        *,
        need_weights: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        attention_update, attention = self.attention(
            self.query_norm(state),
            self.memory_norm(memory),
            self.memory_norm(memory),
            need_weights=need_weights,
            average_attn_weights=True,
        )
        attention_update, attention_scale = smooth_rms_contract(
            attention_update,
            self.max_update_rms,
        )
        intermediate = state + attention_update
        ffn_update, ffn_scale = smooth_rms_contract(
            self.ffn(self.ffn_norm(intermediate)),
            self.max_update_rms,
        )
        innovation = attention_update + ffn_update
        return (
            intermediate + ffn_update,
            innovation,
            attention,
        )


class _BoundedSelfResidual(nn.Module):
    """Pre-normalized self-attention/FFN block with a finite update contract."""

    def __init__(
        self,
        hidden: int,
        heads: int,
        *,
        max_update_rms: float,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(
            hidden,
            heads,
            dropout=0.0,
            bias=False,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, 2 * hidden, bias=False),
            nn.GELU(),
            nn.Linear(2 * hidden, hidden, bias=False),
        )
        self.max_update_rms = float(max_update_rms)

    def forward(
        self,
        state: Tensor,
        *,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        normalized = self.norm(state)
        attention_update, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=attention_mask,
            need_weights=False,
        )
        attention_update, _ = smooth_rms_contract(
            attention_update,
            self.max_update_rms,
        )
        intermediate = state + attention_update
        ffn_update, _ = smooth_rms_contract(
            self.ffn(self.ffn_norm(intermediate)),
            self.max_update_rms,
        )
        return intermediate + ffn_update, attention_update + ffn_update


@dataclass(frozen=True)
class IntentWindowView:
    """Three typed reads over one canonical four-token intent state."""

    tokens: Tensor
    program_attention: Tensor
    support_coordinates: Tensor
    predictive_effect: Tensor

    def validate(self, *, batch: int, program_states: int, hidden: int) -> None:
        expected = {
            "tokens": (batch, 3, hidden),
            "program_attention": (batch, 3, program_states),
            "support_coordinates": (batch, 3, 2),
            "predictive_effect": (batch, 3, hidden),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"intent window {name} must be {shape}, got {tuple(value.shape)}"
                )
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"intent window {name} contains NaN or infinity")

    def as_anchor_context(self, anchors: int) -> Tensor:
        """Compatibility view for four online W anchors, never a mass prior."""

        return _resize_tokens(self.tokens, int(anchors))


@dataclass(frozen=True)
class IntentStateBank:
    """One stateless set-valued program state shared by S, W and P2."""

    protected_goal_program: Tensor
    intent_state: Tensor
    language_innovation: Tensor
    history_innovation: Tensor
    grounding_innovation: Tensor
    ordered_innovation: Tensor
    window_view: IntentWindowView
    temporal_control: Tensor
    phase_uncertainty: Tensor
    terminal_probability: Tensor
    diagnostic_progress: Tensor

    @property
    def active_intent(self) -> Tensor:
        return self.window_view.tokens[:, 0]

    @property
    def next_intent(self) -> Tensor:
        return self.window_view.tokens[:, 1]

    @property
    def remaining_intent(self) -> Tensor:
        return self.window_view.tokens[:, 2]

    @property
    def active_goal(self) -> Tensor:
        return self.active_intent

    @property
    def next_goal(self) -> Tensor:
        return self.next_intent

    @property
    def remaining_goal(self) -> Tensor:
        return self.remaining_intent

    @property
    def goal_program(self) -> Tensor:
        return self.intent_state

    @property
    def phase_belief(self) -> Tensor:
        return self.window_view.program_attention.mean(dim=1)

    @property
    def progress_coordinate(self) -> Tensor:
        return self.diagnostic_progress

    @property
    def window_context(self) -> Tensor:
        return self.window_view.tokens

    @property
    def window_selector(self) -> Tensor:
        """Audit-only attention concentration; never consumed as slot mass."""

        return self.window_view.program_attention.amax(dim=-1)

    @property
    def interval_selector(self) -> Tensor:
        return self.window_view.as_anchor_context(4)

    @property
    def world_context(self) -> Tensor:
        return self.interval_selector

    @property
    def goal_context(self) -> Tensor:
        return _resize_tokens(self.language_innovation, 4)

    @property
    def history_context(self) -> Tensor:
        return _resize_tokens(self.history_innovation, 4)

    def validate(
        self,
        *,
        batch: int,
        program_states: int,
        hidden: int,
        action_horizon: int | None = None,
        intervals: int | None = None,
    ) -> None:
        program_shape = (batch, program_states, hidden)
        for name in (
            "protected_goal_program",
            "intent_state",
            "language_innovation",
            "history_innovation",
            "grounding_innovation",
            "ordered_innovation",
        ):
            value = getattr(self, name)
            if tuple(value.shape) != program_shape:
                raise ValueError(
                    f"intent state {name} must be {program_shape}, "
                    f"got {tuple(value.shape)}"
                )
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"intent state {name} contains NaN or infinity")
        expected_horizon = (
            int(action_horizon)
            if action_horizon is not None
            else int(self.temporal_control.shape[1])
        )
        if tuple(self.temporal_control.shape) != (batch, expected_horizon, hidden):
            raise ValueError("intent temporal_control must be [B,T,H]")
        if intervals is not None:
            for name, value in (
                ("interval_selector", self.interval_selector),
                ("goal_context", self.goal_context),
                ("history_context", self.history_context),
            ):
                if tuple(value.shape) != (batch, int(intervals), hidden):
                    raise ValueError(f"intent state {name} must be [B,A,H]")
        for name in (
            "phase_uncertainty",
            "terminal_probability",
            "diagnostic_progress",
        ):
            value = getattr(self, name)
            if tuple(value.shape) != (batch, 1):
                raise ValueError(f"intent state {name} must be [B,1]")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"intent state {name} contains NaN or infinity")
        self.window_view.validate(
            batch=batch,
            program_states=program_states,
            hidden=hidden,
        )


class DifferentialStatelessIntentController(nn.Module):
    """Six bounded residual blocks plus one typed read over one state bank.

    The four ordered program tokens are the only learned intent carrier.
    Language, causal observed history and current G3 facts update those tokens
    through separate typed residuals.  Near/mid/late are reads over the same
    final bank, not independent states or scalar-positioned heads.
    """

    ordered_program_basis: Tensor
    history_type_basis: Tensor
    window_coordinates: Tensor

    def __init__(
        self,
        hidden: int,
        program_states: int,
        world_intervals: int,
        action_horizon: int,
        *,
        state_dim: int,
        action_dim: int,
        heads: int = 4,
        max_update_rms: float = 0.35,
    ) -> None:
        super().__init__()
        if min(
            hidden,
            program_states,
            world_intervals,
            action_horizon,
            state_dim,
            action_dim,
            heads,
        ) < 1:
            raise ValueError("differential intent dimensions must be positive")
        if hidden % heads:
            raise ValueError("intent hidden width must divide attention heads")
        if int(program_states) != 4:
            raise ValueError("differential intent requires four program states")
        self.hidden = int(hidden)
        self.program_states = int(program_states)
        self.world_intervals = int(world_intervals)
        self.action_horizon = int(action_horizon)
        self.max_update_rms = float(max_update_rms)

        self.register_buffer(
            "ordered_program_basis",
            _ordered_basis(self.program_states, self.hidden),
        )
        self.register_buffer(
            "history_type_basis",
            _ordered_basis(2, self.hidden),
        )
        self.register_buffer(
            "window_coordinates",
            torch.tensor(
                (
                    (0.125, 0.250),
                    (0.500, 0.625),
                    (0.875, 1.000),
                ),
                dtype=torch.float32,
            ),
        )
        self.program_seed = nn.Linear(self.hidden, self.hidden, bias=False)
        self.goal_input = nn.Linear(self.hidden, self.hidden, bias=False)
        self.goal_block = _BoundedCrossResidual(
            self.hidden,
            heads,
            max_update_rms=self.max_update_rms,
        )

        self.state_input = nn.Linear(state_dim, self.hidden, bias=False)
        self.action_input = nn.Linear(action_dim, self.hidden, bias=False)
        self.history_fuse = nn.Sequential(
            nn.LayerNorm(2 * self.hidden, elementwise_affine=False),
            nn.Linear(2 * self.hidden, 2 * self.hidden, bias=False),
            nn.SiLU(),
            nn.Linear(2 * self.hidden, self.hidden, bias=False),
        )
        self.history_time = nn.Linear(self.hidden, self.hidden, bias=False)
        self.history_blocks = nn.ModuleList(
            (
                _BoundedSelfResidual(
                    self.hidden,
                    heads,
                    max_update_rms=self.max_update_rms,
                ),
                _BoundedSelfResidual(
                    self.hidden,
                    heads,
                    max_update_rms=self.max_update_rms,
                ),
            )
        )
        self.history_to_program = _BoundedCrossResidual(
            self.hidden,
            heads,
            max_update_rms=self.max_update_rms,
        )

        self.grounding_input = nn.Linear(self.hidden, self.hidden, bias=False)
        self.grounding_to_program = _BoundedCrossResidual(
            self.hidden,
            heads,
            max_update_rms=self.max_update_rms,
        )
        self.ordered_refinement = _BoundedSelfResidual(
            self.hidden,
            heads,
            max_update_rms=self.max_update_rms,
        )

        self.window_query = nn.Parameter(
            torch.randn(1, 3, self.hidden) * 0.02
        )
        self.window_coordinate_key = nn.Linear(2, self.hidden, bias=False)
        self.window_read = nn.MultiheadAttention(
            self.hidden,
            heads,
            dropout=0.0,
            bias=False,
            batch_first=True,
        )
        self.window_query_norm = nn.LayerNorm(
            self.hidden,
            elementwise_affine=False,
        )
        self.program_read_norm = nn.LayerNorm(
            self.hidden,
            elementwise_affine=False,
        )
        self.window_refinement = nn.Sequential(
            nn.LayerNorm(self.hidden, elementwise_affine=False),
            nn.Linear(self.hidden, 2 * self.hidden, bias=False),
            nn.SiLU(),
            nn.Linear(2 * self.hidden, self.hidden, bias=False),
        )
        self.predictive_effect = nn.Linear(
            self.hidden,
            self.hidden,
            bias=False,
        )
        self.terminal_head = nn.Sequential(
            nn.LayerNorm(2 * self.hidden, elementwise_affine=False),
            nn.Linear(2 * self.hidden, self.hidden, bias=False),
            nn.SiLU(),
            nn.Linear(self.hidden, 1, bias=True),
        )
        terminal_output = cast(nn.Linear, self.terminal_head[-1])
        nn.init.normal_(terminal_output.weight, mean=0.0, std=1e-3)
        nn.init.constant_(terminal_output.bias, -2.5)

    @staticmethod
    def _validate_input(
        name: str,
        value: Tensor,
        *,
        batch: int,
        width: int,
    ) -> None:
        if (
            value.ndim != 3
            or int(value.shape[0]) != int(batch)
            or int(value.shape[1]) < 1
            or int(value.shape[2]) != int(width)
        ):
            raise ValueError(
                f"differential intent {name} must be non-empty [B,N,{width}]"
            )

    def forward(
        self,
        *,
        goal_tokens: Tensor,
        state_history_tokens: Tensor,
        history_tokens: Tensor,
        grounding_tokens: Tensor,
        collect_diagnostics: bool = True,
    ) -> tuple[IntentStateBank, dict[str, Tensor]]:
        batch = int(goal_tokens.shape[0])
        self._validate_input(
            "goal",
            goal_tokens,
            batch=batch,
            width=self.hidden,
        )
        self._validate_input(
            "state history",
            state_history_tokens,
            batch=batch,
            width=self.state_input.in_features,
        )
        self._validate_input(
            "action history",
            history_tokens,
            batch=batch,
            width=self.action_input.in_features,
        )
        self._validate_input(
            "grounding",
            grounding_tokens,
            batch=batch,
            width=self.hidden,
        )

        basis = self.ordered_program_basis.to(
            device=goal_tokens.device,
            dtype=goal_tokens.dtype,
        )
        seed = self.program_seed(basis)[None].expand(batch, -1, -1)
        protected_goal_program, language_innovation, _ = self.goal_block(
            seed,
            self.goal_input(goal_tokens),
        )

        state = self.state_input(state_history_tokens)
        action = self.action_input(history_tokens)
        replay_steps = max(int(state.shape[1]), int(action.shape[1]))
        state = _resize_tokens(state, replay_steps)
        action = _resize_tokens(action, replay_steps)
        type_basis = self.history_type_basis.to(
            device=state.device,
            dtype=state.dtype,
        )
        state = state + type_basis[0][None, None]
        action = action + type_basis[1][None, None]
        history_state = self.history_fuse(torch.cat((state, action), dim=-1))
        time_basis = _ordered_basis(replay_steps, self.hidden).to(
            device=history_state.device,
            dtype=history_state.dtype,
        )
        history_state = history_state + self.history_time(time_basis)[None]
        history_mask = _causal_mask(
            replay_steps,
            device=history_state.device,
        )
        for block in self.history_blocks:
            history_state, _ = block(
                history_state,
                attention_mask=history_mask,
            )

        intent_state, history_innovation, _ = self.history_to_program(
            protected_goal_program,
            history_state,
        )
        intent_state, grounding_innovation, _ = self.grounding_to_program(
            intent_state,
            self.grounding_input(grounding_tokens),
        )
        intent_state, ordered_innovation = self.ordered_refinement(intent_state)

        coordinates = self.window_coordinates.to(
            device=intent_state.device,
            dtype=intent_state.dtype,
        )[None].expand(batch, -1, -1)
        query = self.window_query.to(
            device=intent_state.device,
            dtype=intent_state.dtype,
        ).expand(batch, -1, -1)
        query = query + self.window_coordinate_key(coordinates)
        window_tokens, program_attention = self.window_read(
            self.window_query_norm(query),
            self.program_read_norm(intent_state),
            self.program_read_norm(intent_state),
            need_weights=True,
            average_attn_weights=True,
        )
        window_update, _ = smooth_rms_contract(
            self.window_refinement(window_tokens),
            self.max_update_rms,
        )
        window_tokens = window_tokens + window_update
        predictive_effect = self.predictive_effect(window_tokens)
        temporal_control = _resize_tokens(
            window_tokens,
            self.action_horizon,
        )

        phase_belief = program_attention.mean(dim=1)
        phase_uncertainty = _normalized_entropy(
            phase_belief,
            dim=-1,
        )[:, None].to(dtype=goal_tokens.dtype)
        program_position = torch.linspace(
            0.0,
            1.0,
            self.program_states,
            device=program_attention.device,
            dtype=torch.float32,
        )
        # Diagnostic only: no consumer receives this scalar.
        diagnostic_progress = torch.einsum(
            "bs,s->b",
            phase_belief.float(),
            program_position,
        )[:, None].to(dtype=goal_tokens.dtype)
        terminal_probability = torch.sigmoid(
            self.terminal_head(
                torch.cat(
                    (
                        history_state[:, -1],
                        window_tokens[:, -1],
                    ),
                    dim=-1,
                )
            ).float()
        ).to(dtype=goal_tokens.dtype)

        window_view = IntentWindowView(
            tokens=window_tokens,
            program_attention=program_attention.to(dtype=goal_tokens.dtype),
            support_coordinates=coordinates,
            predictive_effect=predictive_effect,
        )
        bank = IntentStateBank(
            protected_goal_program=protected_goal_program,
            intent_state=intent_state,
            language_innovation=language_innovation,
            history_innovation=history_innovation,
            grounding_innovation=grounding_innovation,
            ordered_innovation=ordered_innovation,
            window_view=window_view,
            temporal_control=temporal_control,
            phase_uncertainty=phase_uncertainty,
            terminal_probability=terminal_probability,
            diagnostic_progress=diagnostic_progress,
        )
        bank.validate(
            batch=batch,
            program_states=self.program_states,
            action_horizon=self.action_horizon,
            hidden=self.hidden,
        )

        metrics: dict[str, Tensor] = {}
        if collect_diagnostics:
            program_norm = intent_state.detach().float()
            window_norm = window_tokens.detach().float()
            attention_f = program_attention.detach().float()
            metrics = {
                "flow_jepa_intent_progress_coordinate": (
                    diagnostic_progress.detach().float().mean()
                ),
                "flow_jepa_intent_phase_uncertainty": (
                    phase_uncertainty.detach().float().mean()
                ),
                "flow_jepa_intent_program_adjacent_cosine": F.cosine_similarity(
                    program_norm[:, 1:],
                    program_norm[:, :-1],
                    dim=-1,
                    eps=1e-6,
                ).mean(),
                "flow_jepa_intent_window_adjacent_cosine": F.cosine_similarity(
                    window_norm[:, 1:],
                    window_norm[:, :-1],
                    dim=-1,
                    eps=1e-6,
                ).mean(),
                "flow_jepa_intent_program_attention_entropy": (
                    _normalized_entropy(attention_f, dim=-1).mean()
                ),
                "flow_jepa_intent_predictive_effect_rms": (
                    predictive_effect.detach().float().square().mean().sqrt()
                ),
                "flow_jepa_intent_language_innovation_rms": (
                    language_innovation.detach().float().square().mean().sqrt()
                ),
                "flow_jepa_intent_history_innovation_rms": (
                    history_innovation.detach().float().square().mean().sqrt()
                ),
                "flow_jepa_intent_grounding_innovation_rms": (
                    grounding_innovation.detach().float().square().mean().sqrt()
                ),
                "flow_jepa_intent_ordered_innovation_rms": (
                    ordered_innovation.detach().float().square().mean().sqrt()
                ),
            }
            for window_index, name in enumerate(("near", "mid", "late")):
                metrics[f"flow_jepa_intent_{name}_program_argmax"] = (
                    attention_f[:, window_index].argmax(dim=-1).float().mean()
                )
                metrics[f"flow_jepa_intent_{name}_program_max_mass"] = (
                    attention_f[:, window_index].amax(dim=-1).mean()
                )
        return bank, metrics


@dataclass(frozen=True)
class DifferentialWindowEffectBank:
    """One current reference and three differentiated online effect slots."""

    current_reference: Tensor
    semantic_delta: Tensor
    transport_mean: Tensor
    transport_covariance: Tensor
    persistence: Tensor
    visibility: Tensor
    uncertainty: Tensor
    slot_valid: Tensor
    slot_names: tuple[str, ...]

    @property
    def successor_content(self) -> Tensor:
        return self.current_reference[:, None] + self.semantic_delta

    @property
    def slots(self) -> int:
        return int(self.semantic_delta.shape[1])

    def validate(self, *, expected_slots: int | None = None) -> None:
        if self.current_reference.ndim != 6:
            raise ValueError(
                "differential current reference must preserve [B,C,G,G,M,H]"
            )
        if self.semantic_delta.ndim != 7:
            raise ValueError(
                "differential semantic delta must preserve [B,S,C,G,G,M,H]"
            )
        batch = int(self.current_reference.shape[0])
        slots = int(self.semantic_delta.shape[1])
        if tuple(self.semantic_delta.shape[:1] + self.semantic_delta.shape[2:]) != tuple(
            self.current_reference.shape
        ):
            raise ValueError("differential current/effect spatial charts do not align")
        if expected_slots is not None and slots != int(expected_slots):
            raise ValueError(
                f"differential effect requires {expected_slots} slots, got {slots}"
            )
        prefix = tuple(self.semantic_delta.shape[:-1])
        expected = {
            "transport_mean": (*prefix, 2),
            "transport_covariance": (*prefix, 3),
            "persistence": (*prefix, 1),
            "visibility": (*prefix, 1),
            "uncertainty": (*prefix, 1),
            "slot_valid": (*prefix, 1),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"differential effect {name} must be {shape}, "
                    f"got {tuple(value.shape)}"
                )
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"differential effect {name} is non-finite")
        if len(self.slot_names) != slots:
            raise ValueError("differential effect slot names do not match slots")
        if batch < 1 or not bool(torch.isfinite(self.current_reference).all()):
            raise ValueError("differential current reference is invalid")
        if not bool(torch.isfinite(self.semantic_delta).all()):
            raise ValueError("differential semantic delta is non-finite")


class DifferentialWindowRouteCompiler(nn.Module):
    """W1 near/mid transitions and W2 typed late read without identity means."""

    def __init__(
        self,
        *,
        route_dim: int,
        hidden: int,
        heads: int = 4,
        slots_per_cell: int = 4,
    ) -> None:
        super().__init__()
        if min(route_dim, hidden, heads, slots_per_cell) < 1:
            raise ValueError("differential W dimensions must be positive")
        if route_dim % heads:
            raise ValueError("differential W route width must divide heads")
        self.route_dim = int(route_dim)
        self.hidden = int(hidden)
        self.slots_per_cell = int(slots_per_cell)
        self.intent_to_route = nn.Linear(hidden, route_dim, bias=False)
        self.w1_transition = _BoundedSelfResidual(
            route_dim,
            heads,
            max_update_rms=0.50,
        )
        self.late_query = nn.Parameter(
            torch.randn(1, 1, route_dim) * 0.02
        )
        self.late_source_type = nn.Parameter(
            torch.randn(1, 4, route_dim) * 0.02
        )
        self.late_query_norm = nn.LayerNorm(
            route_dim,
            elementwise_affine=False,
        )
        self.late_memory_norm = nn.LayerNorm(
            route_dim,
            elementwise_affine=False,
        )
        self.late_attention = nn.MultiheadAttention(
            route_dim,
            heads,
            dropout=0.0,
            bias=False,
            batch_first=True,
        )
        self.late_ffn = nn.Sequential(
            nn.LayerNorm(route_dim, elementwise_affine=False),
            nn.Linear(route_dim, 2 * route_dim, bias=False),
            nn.SiLU(),
            nn.Linear(2 * route_dim, route_dim, bias=False),
        )
        self.current_reference = nn.Sequential(
            nn.LayerNorm(route_dim, elementwise_affine=False),
            nn.Linear(route_dim, hidden, bias=False),
        )
        self.effect_semantic = nn.Sequential(
            nn.LayerNorm(route_dim, elementwise_affine=False),
            nn.Linear(route_dim, 2 * route_dim, bias=False),
            nn.SiLU(),
            nn.Linear(2 * route_dim, hidden, bias=False),
        )
        self.effect_geometry = nn.Sequential(
            nn.LayerNorm(route_dim, elementwise_affine=False),
            nn.Linear(route_dim, 2 * route_dim, bias=False),
            nn.SiLU(),
            nn.Linear(2 * route_dim, 8, bias=False),
        )
        current_output = cast(nn.Linear, self.current_reference[-1])
        semantic_output = cast(nn.Linear, self.effect_semantic[-1])
        geometry_output = cast(nn.Linear, self.effect_geometry[-1])
        nn.init.normal_(current_output.weight, mean=0.0, std=1e-3)
        nn.init.normal_(semantic_output.weight, mean=0.0, std=1e-3)
        nn.init.normal_(geometry_output.weight, mean=0.0, std=1e-3)

    def _validate_inputs(
        self,
        selected_route: Tensor,
        current_keys: Tensor,
        intent: IntentWindowView,
    ) -> tuple[int, int, int, int]:
        if selected_route.ndim != 6 or int(selected_route.shape[-1]) != self.route_dim:
            raise ValueError("differential W route must be [B,A,C,G,G,R]")
        if current_keys.ndim != 6 or int(current_keys.shape[-1]) != self.route_dim:
            raise ValueError("differential W current keys must be [B,C,G,G,M,R]")
        batch, anchors, cameras, rows, columns, _ = selected_route.shape
        if rows != columns or anchors != 4:
            raise ValueError("differential W requires four anchors on a square chart")
        if tuple(current_keys.shape[:4]) != (batch, cameras, rows, columns):
            raise ValueError("differential W current keys do not align to routes")
        if int(current_keys.shape[4]) != self.slots_per_cell:
            raise ValueError("differential W current keys have the wrong slot count")
        intent.validate(
            batch=int(batch),
            program_states=int(intent.program_attention.shape[-1]),
            hidden=self.hidden,
        )
        return int(batch), int(cameras), int(rows), int(columns)

    def _decode(
        self,
        route_state: Tensor,
        current_keys: Tensor,
        *,
        slot_names: tuple[str, ...],
        output_dtype: torch.dtype,
    ) -> DifferentialWindowEffectBank:
        if route_state.ndim != 6:
            raise ValueError("differential W state must be [B,S,C,G,G,R]")
        routes = route_state[..., None, :].expand(
            -1,
            -1,
            -1,
            -1,
            -1,
            self.slots_per_cell,
            -1,
        )
        semantic_delta, _ = smooth_rms_contract(
            self.effect_semantic(routes),
            0.50,
        )
        raw_geometry = self.effect_geometry(routes).float()
        transport_mean = 0.50 * torch.tanh(raw_geometry[..., :2])
        variance_diag = 0.01 + 0.99 * torch.sigmoid(
            raw_geometry[..., 2:4]
        )
        covariance_cross = (
            0.50
            * torch.tanh(raw_geometry[..., 4:5])
            * variance_diag.prod(dim=-1, keepdim=True).sqrt()
        )
        covariance = torch.cat((variance_diag, covariance_cross), dim=-1)
        persistence = torch.sigmoid(raw_geometry[..., 5:6])
        visibility = torch.sigmoid(raw_geometry[..., 6:7])
        uncertainty = 0.05 + 3.95 * torch.sigmoid(
            raw_geometry[..., 7:8] - 1.5
        )
        current_reference, _ = smooth_rms_contract(
            self.current_reference(current_keys),
            0.75,
        )
        slot_valid = torch.ones_like(persistence, dtype=output_dtype)
        bank = DifferentialWindowEffectBank(
            current_reference=current_reference.to(dtype=output_dtype),
            semantic_delta=semantic_delta.to(dtype=output_dtype),
            transport_mean=transport_mean.to(dtype=output_dtype),
            transport_covariance=covariance.to(dtype=output_dtype),
            persistence=persistence.to(dtype=output_dtype),
            visibility=visibility.to(dtype=output_dtype),
            uncertainty=uncertainty.to(dtype=output_dtype),
            slot_valid=slot_valid,
            slot_names=slot_names,
        )
        bank.validate(expected_slots=len(slot_names))
        return bank

    @staticmethod
    def _route_metrics(
        bank: DifferentialWindowEffectBank,
        route_state: Tensor,
        prefix: str,
    ) -> dict[str, Tensor]:
        metrics: dict[str, Tensor] = {
            f"flow_jepa_{prefix}_route_rms": (
                route_state.detach().float().square().mean().sqrt()
            ),
            f"flow_jepa_{prefix}_effect_rms": (
                bank.semantic_delta.detach().float().square().mean().sqrt()
            ),
        }
        if bank.slots > 1:
            pooled = bank.semantic_delta.detach().float().mean(
                dim=(2, 3, 4, 5)
            )
            metrics[f"flow_jepa_{prefix}_adjacent_cosine"] = (
                F.cosine_similarity(
                    pooled[:, 1:],
                    pooled[:, :-1],
                    dim=-1,
                    eps=1e-6,
                ).mean()
            )
            metrics[f"flow_jepa_{prefix}_slot_variation"] = pooled.std(
                dim=1,
                unbiased=False,
            ).mean()
        for index, name in enumerate(bank.slot_names):
            metrics[f"flow_jepa_{prefix}_{name}_effect_rms"] = (
                bank.semantic_delta[:, index]
                .detach()
                .float()
                .square()
                .mean()
                .sqrt()
            )
        return metrics

    def forward_w1(
        self,
        selected_route: Tensor,
        current_keys: Tensor,
        intent: IntentWindowView,
        *,
        output_dtype: torch.dtype,
        collect_diagnostics: bool = True,
    ) -> tuple[DifferentialWindowEffectBank, Tensor, dict[str, Tensor]]:
        batch, cameras, rows, columns = self._validate_inputs(
            selected_route,
            current_keys,
            intent,
        )
        condition = self.intent_to_route(intent.tokens[:, :2]).to(
            dtype=selected_route.dtype
        )
        state = selected_route[:, :2] + condition[
            :,
            :,
            None,
            None,
            None,
        ]
        flattened = state.permute(0, 2, 3, 4, 1, 5).reshape(
            batch * cameras * rows * columns,
            2,
            self.route_dim,
        )
        flattened, _ = self.w1_transition(
            flattened,
            attention_mask=_causal_mask(2, device=flattened.device),
        )
        route_state = flattened.reshape(
            batch,
            cameras,
            rows,
            columns,
            2,
            self.route_dim,
        ).permute(0, 4, 1, 2, 3, 5)
        bank = self._decode(
            route_state,
            current_keys,
            slot_names=("near", "mid"),
            output_dtype=output_dtype,
        )
        metrics = (
            self._route_metrics(bank, route_state, "differential_w1")
            if collect_diagnostics
            else {}
        )
        return bank, route_state, metrics

    def forward_w2(
        self,
        selected_route: Tensor,
        current_keys: Tensor,
        intent: IntentWindowView,
        *,
        w1_bank: DifferentialWindowEffectBank,
        w1_route_state: Tensor,
        output_dtype: torch.dtype,
        collect_diagnostics: bool = True,
    ) -> tuple[DifferentialWindowEffectBank, Tensor, dict[str, Tensor]]:
        batch, cameras, rows, columns = self._validate_inputs(
            selected_route,
            current_keys,
            intent,
        )
        w1_bank.validate(expected_slots=2)
        expected_w1 = (batch, 2, cameras, rows, columns, self.route_dim)
        if tuple(w1_route_state.shape) != expected_w1:
            raise ValueError(
                f"differential W2 needs W1 routes {expected_w1}, "
                f"got {tuple(w1_route_state.shape)}"
            )
        far = selected_route[:, 2:4]
        memory = torch.cat((w1_route_state, far), dim=1)
        source_type = self.late_source_type.to(
            device=memory.device,
            dtype=memory.dtype,
        )
        memory = memory + source_type[
            :,
            :,
            None,
            None,
            None,
        ]
        memory = memory.permute(0, 2, 3, 4, 1, 5).reshape(
            batch * cameras * rows * columns,
            4,
            self.route_dim,
        )
        late_intent = self.intent_to_route(intent.tokens[:, 2]).to(
            dtype=selected_route.dtype
        )
        query = self.late_query.to(
            device=selected_route.device,
            dtype=selected_route.dtype,
        ).expand(batch, -1, -1)
        query = query + late_intent[:, None]
        query = query[:, None, None, None].expand(
            batch,
            cameras,
            rows,
            columns,
            1,
            self.route_dim,
        ).reshape(
            batch * cameras * rows * columns,
            1,
            self.route_dim,
        )
        late, _ = self.late_attention(
            self.late_query_norm(query),
            self.late_memory_norm(memory),
            self.late_memory_norm(memory),
            need_weights=False,
        )
        late_update, _ = smooth_rms_contract(self.late_ffn(late), 0.50)
        late = late + late_update
        late_route = late.reshape(
            batch,
            cameras,
            rows,
            columns,
            self.route_dim,
        )[:, None]
        route_state = torch.cat((w1_route_state, late_route), dim=1)
        bank = self._decode(
            route_state,
            current_keys,
            slot_names=("near", "mid", "late"),
            output_dtype=output_dtype,
        )
        # One protected current object is reused; W2 does not reconstruct W1's
        # near/mid current reference or effects through another decoder.
        bank = DifferentialWindowEffectBank(
            current_reference=w1_bank.current_reference,
            semantic_delta=torch.cat(
                (
                    w1_bank.semantic_delta,
                    bank.semantic_delta[:, 2:3],
                ),
                dim=1,
            ),
            transport_mean=torch.cat(
                (
                    w1_bank.transport_mean,
                    bank.transport_mean[:, 2:3],
                ),
                dim=1,
            ),
            transport_covariance=torch.cat(
                (
                    w1_bank.transport_covariance,
                    bank.transport_covariance[:, 2:3],
                ),
                dim=1,
            ),
            persistence=torch.cat(
                (
                    w1_bank.persistence,
                    bank.persistence[:, 2:3],
                ),
                dim=1,
            ),
            visibility=torch.cat(
                (
                    w1_bank.visibility,
                    bank.visibility[:, 2:3],
                ),
                dim=1,
            ),
            uncertainty=torch.cat(
                (
                    w1_bank.uncertainty,
                    bank.uncertainty[:, 2:3],
                ),
                dim=1,
            ),
            slot_valid=torch.cat(
                (
                    w1_bank.slot_valid,
                    bank.slot_valid[:, 2:3],
                ),
                dim=1,
            ),
            slot_names=("near", "mid", "late"),
        )
        bank.validate(expected_slots=3)
        metrics = (
            self._route_metrics(bank, route_state, "differential_w2")
            if collect_diagnostics
            else {}
        )
        return bank, route_state, metrics


class DifferentialFutureEffectReader(nn.Module):
    """P2 effect-specific spatial read with learned time coordinates."""

    def __init__(
        self,
        *,
        hidden: int,
        horizon: int,
        basis: int,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.horizon = int(horizon)
        self.basis = int(basis)
        self.query = nn.Sequential(
            nn.LayerNorm(hidden, elementwise_affine=False),
            nn.Linear(hidden, hidden, bias=False),
        )
        self.current_context = nn.Sequential(
            nn.LayerNorm(hidden, elementwise_affine=False),
            nn.Linear(hidden, hidden, bias=False),
        )
        self.effect_key = nn.Sequential(
            nn.LayerNorm(hidden, elementwise_affine=False),
            nn.Linear(hidden, hidden, bias=False),
        )
        self.geometry_key = nn.Sequential(
            nn.LayerNorm(8, elementwise_affine=False),
            nn.Linear(8, hidden, bias=False),
        )
        self.intent_key = nn.Sequential(
            nn.LayerNorm(hidden, elementwise_affine=False),
            nn.Linear(hidden, hidden, bias=False),
        )
        self.action_coordinate_query = nn.Linear(2, hidden, bias=False)
        self.support_coordinate_key = nn.Linear(2, hidden, bias=False)
        self.effect_value = nn.Sequential(
            nn.LayerNorm(hidden + 8, elementwise_affine=False),
            nn.Linear(hidden + 8, hidden, bias=False),
        )
        effect_value_output = cast(nn.Linear, self.effect_value[-1])
        nn.init.normal_(effect_value_output.weight, mean=0.0, std=3e-3)

    @staticmethod
    def _geometry(bank: DifferentialWindowEffectBank) -> Tensor:
        return torch.cat(
            (
                bank.transport_mean,
                bank.transport_covariance,
                bank.persistence,
                bank.visibility,
                bank.uncertainty,
            ),
            dim=-1,
        )

    def forward(
        self,
        query_tokens: Tensor,
        bank: DifferentialWindowEffectBank,
        intent: IntentWindowView,
        *,
        collect_diagnostics: bool = True,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        bank.validate(expected_slots=3)
        batch = int(query_tokens.shape[0])
        expected = (batch, self.horizon, self.basis, self.hidden)
        if tuple(query_tokens.shape) != expected:
            raise ValueError(
                f"differential P2 query must be {expected}, "
                f"got {tuple(query_tokens.shape)}"
            )
        intent.validate(
            batch=batch,
            program_states=int(intent.program_attention.shape[-1]),
            hidden=self.hidden,
        )
        slots = bank.slots
        spatial = int(bank.semantic_delta[0, 0].numel() // self.hidden)
        current = bank.current_reference.reshape(batch, spatial, self.hidden)
        current_summary = current.mean(dim=1)
        query = self.query(query_tokens) + self.current_context(
            current_summary
        )[:, None, None]

        geometry = self._geometry(bank)
        key = self.effect_key(bank.semantic_delta) + self.geometry_key(geometry)
        key = key.reshape(batch, slots, spatial, self.hidden)
        value = self.effect_value(
            torch.cat((bank.semantic_delta, geometry), dim=-1)
        ).reshape(batch, slots, spatial, self.hidden)
        intent_key = self.intent_key(intent.tokens)

        content_score = torch.einsum(
            "btkh,bsnh->btksn",
            query.float(),
            key.float(),
        ) / math.sqrt(float(self.hidden))
        intent_score = torch.einsum(
            "btkh,bsh->btks",
            query.float(),
            intent_key.float(),
        ) / math.sqrt(float(self.hidden))
        action_time = torch.linspace(
            0.0,
            1.0,
            self.horizon,
            device=query_tokens.device,
            dtype=torch.float32,
        )
        basis_position = torch.linspace(
            0.0,
            1.0,
            self.basis,
            device=query_tokens.device,
            dtype=torch.float32,
        )
        action_coordinates = torch.stack(
            torch.meshgrid(action_time, basis_position, indexing="ij"),
            dim=-1,
        )
        coordinate_query = self.action_coordinate_query(
            action_coordinates.to(dtype=query_tokens.dtype)
        )
        support_key = self.support_coordinate_key(
            intent.support_coordinates.to(dtype=query_tokens.dtype)
        )
        coordinate_score = torch.einsum(
            "tkh,bsh->btks",
            coordinate_query.float(),
            support_key.float(),
        ) / math.sqrt(float(self.hidden))

        valid = bank.slot_valid.reshape(
            batch,
            slots,
            spatial,
            1,
        )[..., 0].float()
        logits = (
            content_score
            + intent_score[..., None]
            + coordinate_score[..., None]
        )
        logits = logits.masked_fill(
            valid[:, None, None] <= 0.0,
            torch.finfo(logits.dtype).min,
        )
        posterior = torch.softmax(
            logits.reshape(batch, self.horizon, self.basis, -1),
            dim=-1,
        ).reshape(
            batch,
            self.horizon,
            self.basis,
            slots,
            spatial,
        )
        read = torch.einsum(
            "btksn,bsnh->btkh",
            posterior.to(dtype=value.dtype),
            value,
        )
        metrics: dict[str, Tensor] = {}
        if collect_diagnostics:
            posterior_f = posterior.detach().float()
            slot_mass = posterior_f.sum(dim=-1).mean(dim=(0, 1, 2))
            metrics = {
                "flow_jepa_p2_effect_read_rms": (
                    read.detach().float().square().mean().sqrt()
                ),
                "flow_jepa_p2_effect_content_score_rms": (
                    content_score.detach().square().mean().sqrt()
                ),
                "flow_jepa_p2_effect_intent_score_rms": (
                    intent_score.detach().square().mean().sqrt()
                ),
                "flow_jepa_p2_effect_coordinate_score_rms": (
                    coordinate_score.detach().square().mean().sqrt()
                ),
                "flow_jepa_p2_effect_entropy": (
                    -(
                        posterior_f.reshape(
                            batch,
                            self.horizon,
                            self.basis,
                            -1,
                        ).clamp_min(1e-8)
                        * posterior_f.reshape(
                            batch,
                            self.horizon,
                            self.basis,
                            -1,
                        ).clamp_min(1e-8).log()
                    ).sum(dim=-1).mean()
                    / math.log(float(max(slots * spatial, 2)))
                ),
            }
            for index, name in enumerate(bank.slot_names):
                metrics[f"flow_jepa_p2_effect_{name}_mass"] = slot_mass[index]
        return read, metrics


@dataclass(frozen=True)
class ConsequenceAwarePlanState:
    """P2 factual/effect organization that is mandatory before P3 routing."""

    factual_base: Tensor
    effect_base: Tensor
    organized_delta: Tensor
    protected_base: Tensor

    def validate(self) -> None:
        shape = tuple(self.protected_base.shape)
        if len(shape) != 4:
            raise ValueError("consequence plan state must be [B,T,K,H]")
        for name in (
            "factual_base",
            "effect_base",
            "organized_delta",
            "protected_base",
        ):
            value = getattr(self, name)
            if tuple(value.shape) != shape:
                raise ValueError(f"consequence plan {name} is misaligned")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"consequence plan {name} is non-finite")


class ConsequencePlanOrganizer(nn.Module):
    """Make the effect a mandatory bounded part of the protected plan base."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.organizer = nn.Sequential(
            nn.LayerNorm(3 * hidden, elementwise_affine=False),
            nn.Linear(3 * hidden, 2 * hidden, bias=False),
            nn.SiLU(),
            nn.Linear(2 * hidden, hidden, bias=False),
        )
        organizer_output = cast(nn.Linear, self.organizer[-1])
        nn.init.normal_(organizer_output.weight, mean=0.0, std=3e-3)

    def forward(
        self,
        *,
        factual_base: Tensor,
        effect_read: Tensor,
        p2_delta: Tensor,
    ) -> tuple[ConsequenceAwarePlanState, dict[str, Tensor]]:
        if not (
            tuple(factual_base.shape)
            == tuple(effect_read.shape)
            == tuple(p2_delta.shape)
        ):
            raise ValueError("consequence plan operands must align")
        effect_base, effect_scale = smooth_rms_contract(effect_read, 0.35)
        organized_delta, organizer_scale = smooth_rms_contract(
            self.organizer(
                torch.cat((factual_base, effect_base, p2_delta), dim=-1)
            ),
            0.25,
        )
        state = ConsequenceAwarePlanState(
            factual_base=factual_base,
            effect_base=effect_base,
            organized_delta=organized_delta,
            protected_base=factual_base + effect_base + organized_delta,
        )
        state.validate()
        metrics = {
            "flow_jepa_consequence_effect_base_rms": (
                effect_base.detach().float().square().mean().sqrt()
            ),
            "flow_jepa_consequence_organized_delta_rms": (
                organized_delta.detach().float().square().mean().sqrt()
            ),
            "flow_jepa_consequence_effect_contract_min": (
                effect_scale.detach().float().amin()
            ),
            "flow_jepa_consequence_organizer_contract_min": (
                organizer_scale.detach().float().amin()
            ),
        }
        return state, metrics


@dataclass(frozen=True)
class DifferentialExecutionTerminalEvidence:
    probability: Tensor
    uncertainty: Tensor

    def validate(self, *, batch: int) -> None:
        for name in ("probability", "uncertainty"):
            value = getattr(self, name)
            if tuple(value.shape) != (batch, 1):
                raise ValueError(f"differential terminal {name} must be [B,1]")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"differential terminal {name} is non-finite")


@dataclass(frozen=True)
class DifferentialPolicyPlanBank:
    """P3 output without a second optional effect lane."""

    protected_base: Tensor
    precision: Tensor
    temporal: Tensor
    execution_terminal: DifferentialExecutionTerminalEvidence

    @property
    def source_names(self) -> tuple[str, str]:
        return ("p3_precision", "p3_temporal")

    def validate(self) -> None:
        expected = tuple(self.protected_base.shape)
        if len(expected) != 4:
            raise ValueError("differential policy plan must be [B,T,K,H]")
        for name in ("precision", "temporal"):
            value = getattr(self, name)
            if tuple(value.shape) != expected:
                raise ValueError(f"differential policy plan {name} is misaligned")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"differential policy plan {name} is non-finite")
        self.execution_terminal.validate(batch=int(expected[0]))

    def as_policy_role_bank(self, *, source_depth: int) -> PolicyRoleDeltaBank:
        self.validate()
        return PolicyRoleDeltaBank(
            values=torch.stack((self.precision, self.temporal), dim=1),
            source_names=self.source_names,
            source_depths=(int(source_depth), int(source_depth)),
            protected_detail=self.protected_base,
        )


class DifferentialPolicyPlanCompiler(nn.Module):
    """P3 precision/temporal compiler over an already organized consequence."""

    def __init__(self, *, hidden: int, horizon: int, basis: int) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.horizon = int(horizon)
        self.basis = int(basis)
        self.basis_identity = nn.Parameter(
            torch.randn(1, 1, basis, hidden) * 0.02
        )
        self.precision_lane = nn.Sequential(
            nn.LayerNorm(3 * hidden, elementwise_affine=False),
            nn.Linear(3 * hidden, 2 * hidden, bias=False),
            nn.SiLU(),
            nn.Linear(2 * hidden, hidden, bias=False),
        )
        self.temporal_lane = nn.Sequential(
            nn.LayerNorm(2 * hidden, elementwise_affine=False),
            nn.Linear(2 * hidden, 2 * hidden, bias=False),
            nn.SiLU(),
            nn.Linear(2 * hidden, hidden, bias=False),
        )
        precision_output = cast(nn.Linear, self.precision_lane[-1])
        temporal_output = cast(nn.Linear, self.temporal_lane[-1])
        nn.init.normal_(precision_output.weight, mean=0.0, std=3e-3)
        nn.init.normal_(temporal_output.weight, mean=0.0, std=3e-3)

    def forward(
        self,
        *,
        p1_delta: Tensor,
        protected_detail: Tensor,
        consequence: ConsequenceAwarePlanState,
        intent: IntentStateBank,
    ) -> tuple[DifferentialPolicyPlanBank, dict[str, Tensor]]:
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
            ("consequence base", consequence.protected_base),
        ):
            if tuple(value.shape) != expected:
                raise ValueError(f"differential P3 {name} must be {expected}")
        intent.validate(
            batch=expected[0],
            program_states=int(intent.intent_state.shape[1]),
            action_horizon=self.horizon,
            hidden=self.hidden,
        )
        basis = self.basis_identity.to(
            device=p1_delta.device,
            dtype=p1_delta.dtype,
        ).expand(expected[0], self.horizon, -1, -1)
        temporal_control = intent.temporal_control[:, :, None].expand(
            -1,
            -1,
            self.basis,
            -1,
        )
        precision, precision_scale = smooth_rms_contract(
            self.precision_lane(
                torch.cat((protected_detail, p1_delta, basis), dim=-1)
            ),
            0.35,
        )
        temporal, temporal_scale = smooth_rms_contract(
            self.temporal_lane(
                torch.cat((temporal_control, basis), dim=-1)
            ),
            0.35,
        )
        terminal = DifferentialExecutionTerminalEvidence(
            probability=intent.terminal_probability,
            uncertainty=intent.phase_uncertainty,
        )
        bank = DifferentialPolicyPlanBank(
            protected_base=consequence.protected_base,
            precision=precision,
            temporal=temporal,
            execution_terminal=terminal,
        )
        bank.validate()
        metrics = {
            "flow_jepa_policy_plan_protected_base_rms": (
                consequence.protected_base.detach().float().square().mean().sqrt()
            ),
            "flow_jepa_policy_plan_precision_rms": (
                precision.detach().float().square().mean().sqrt()
            ),
            "flow_jepa_policy_plan_temporal_rms": (
                temporal.detach().float().square().mean().sqrt()
            ),
            "flow_jepa_policy_plan_precision_contract_min": (
                precision_scale.detach().float().amin()
            ),
            "flow_jepa_policy_plan_temporal_contract_min": (
                temporal_scale.detach().float().amin()
            ),
        }
        return bank, metrics
