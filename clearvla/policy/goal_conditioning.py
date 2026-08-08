"""Compact precomputed-T5 goal conditioning for the policy canvas."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class GoalPhaseState:
    """Typed online result of the stateless goal/phase program.

    Goal, phase and history remain distinct operands.  The four interval
    contexts are selector values only: none of them is a future-teacher value
    or a direct action write.
    """

    active_goal: Tensor
    next_goal: Tensor
    remaining_goal: Tensor
    phase_belief: Tensor
    phase_uncertainty: Tensor
    interval_selector: Tensor
    goal_context: Tensor
    history_context: Tensor
    goal_program: Tensor
    terminal_probability: Tensor | None = None

    def validate(
        self,
        *,
        batch: int,
        program_states: int,
        intervals: int,
        hidden: int,
    ) -> None:
        vectors = {
            "active_goal": self.active_goal,
            "next_goal": self.next_goal,
            "remaining_goal": self.remaining_goal,
        }
        for name, value in vectors.items():
            if tuple(value.shape) != (batch, hidden):
                raise ValueError(f"{name} must be [B,H]")
        legacy_belief = tuple(self.phase_belief.shape) == (
            batch,
            program_states + 1,
        )
        v116_belief = tuple(self.phase_belief.shape) == (
            batch,
            program_states,
        )
        if not (legacy_belief or v116_belief):
            raise ValueError(
                "phase_belief must be [B,state] or legacy [B,state+terminal]"
            )
        if v116_belief:
            if self.terminal_probability is None or tuple(
                self.terminal_probability.shape
            ) != (batch, 1):
                raise ValueError(
                    "four-state phase belief requires separate [B,1] terminal probability"
                )
        elif self.terminal_probability is not None and tuple(
            self.terminal_probability.shape
        ) != (batch, 1):
            raise ValueError("terminal_probability must be [B,1]")
        if tuple(self.phase_uncertainty.shape) != (batch, 1):
            raise ValueError("phase_uncertainty must be [B,1]")
        for name, value in (
            ("interval_selector", self.interval_selector),
            ("goal_context", self.goal_context),
            ("history_context", self.history_context),
        ):
            if tuple(value.shape) != (batch, intervals, hidden):
                raise ValueError(f"{name} must be [B,interval,H]")
        if tuple(self.goal_program.shape) != (
            batch,
            program_states,
            hidden,
        ):
            raise ValueError("goal_program must be [B,state,H]")
        finite_values = [
            *vectors.items(),
            ("phase_belief", self.phase_belief),
            ("phase_uncertainty", self.phase_uncertainty),
            ("interval_selector", self.interval_selector),
            ("goal_context", self.goal_context),
            ("history_context", self.history_context),
            ("goal_program", self.goal_program),
        ]
        if self.terminal_probability is not None:
            finite_values.append(
                ("terminal_probability", self.terminal_probability)
            )
        for name, value in finite_values:
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} contains NaN or infinity")


class StatelessGoalPhaseMachine(nn.Module):
    """Observation-conditioned monotone goal program without recurrent state.

    Four ordered program queries read the full resampled T5 goal bank.  A
    causal observation encoder then replays state history, executed-action
    history and the completed current G3 facts on every forward pass.  The
    explicit probability recurrence permits only stay, advance-one and an
    absorbing terminal state, while the initial observation may enter at any
    program state.
    """

    def __init__(
        self,
        hidden: int,
        program_states: int,
        intervals: int,
        heads: int,
        *,
        state_dim: int | None = None,
        action_dim: int | None = None,
        separate_terminal: bool = False,
    ) -> None:
        super().__init__()
        if min(hidden, program_states, intervals, heads) < 1:
            raise ValueError("goal-phase dimensions must be positive")
        if program_states < 2:
            raise ValueError("goal-phase machine needs at least two program states")
        if hidden % heads:
            raise ValueError("goal-phase hidden size must divide attention heads")
        self.hidden = int(hidden)
        self.program_states = int(program_states)
        self.intervals = int(intervals)
        self.separate_terminal = bool(separate_terminal)
        self.state_dim = (
            self.hidden if state_dim is None else int(state_dim)
        )
        self.action_dim = (
            self.hidden if action_dim is None else int(action_dim)
        )
        if min(self.state_dim, self.action_dim) < 1:
            raise ValueError("goal-phase observable dimensions must be positive")

        program_basis = self._ordered_basis(self.program_states, self.hidden)
        interval_basis = self._ordered_basis(self.intervals, self.hidden)
        self.register_buffer("ordered_program_basis", program_basis)
        self.register_buffer("ordered_interval_basis", interval_basis)

        self.program_query = nn.Linear(hidden, hidden, bias=False)
        self.goal_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.goal_cross = nn.MultiheadAttention(
            hidden,
            heads,
            dropout=0.0,
            bias=False,
            batch_first=True,
        )
        self.program_ffn = nn.Sequential(
            nn.LayerNorm(hidden, elementwise_affine=False),
            nn.Linear(hidden, 2 * hidden, bias=False),
            nn.SiLU(),
            nn.Linear(2 * hidden, hidden, bias=False),
        )

        self.state_type = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        self.action_type = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        self.grounding_type = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        self.state_input = nn.Linear(self.state_dim, hidden, bias=False)
        self.action_input = nn.Linear(self.action_dim, hidden, bias=False)
        self.grounding_input = nn.Linear(hidden, hidden, bias=False)
        self.observation_fuse = nn.Sequential(
            nn.LayerNorm(2 * hidden, elementwise_affine=False),
            nn.Linear(2 * hidden, 2 * hidden, bias=False),
            nn.SiLU(),
            nn.Linear(2 * hidden, hidden, bias=False),
        )
        self.observation_time = nn.Linear(hidden, hidden, bias=False)
        self.grounding_observation_query = nn.Linear(
            hidden, hidden, bias=False
        )
        self.grounding_summary_query = nn.Parameter(
            torch.randn(1, 1, hidden) * 0.02
        )
        self.grounding_norm = nn.LayerNorm(
            hidden, elementwise_affine=False
        )
        self.grounding_cross = nn.MultiheadAttention(
            hidden,
            heads,
            dropout=0.0,
            bias=False,
            batch_first=True,
        )
        observation_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=2 * hidden,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            bias=False,
        )
        self.observation_encoder = nn.TransformerEncoder(
            observation_layer,
            num_layers=2,
            enable_nested_tensor=False,
        )
        self.observation_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.observation_query = nn.Linear(hidden, hidden, bias=False)
        self.program_key = nn.Linear(hidden, hidden, bias=False)
        if self.separate_terminal:
            self.register_parameter("terminal_key", None)
            self.initial_bias = nn.Parameter(torch.zeros(self.program_states))
            self.completion_head = nn.Sequential(
                nn.LayerNorm(2 * hidden, elementwise_affine=False),
                nn.Linear(2 * hidden, hidden, bias=False),
                nn.SiLU(),
                nn.Linear(hidden, 1, bias=True),
            )
            nn.init.normal_(self.completion_head[-1].weight, mean=0.0, std=1e-3)
            nn.init.constant_(self.completion_head[-1].bias, -2.5)
        else:
            self.terminal_key = nn.Parameter(torch.randn(1, hidden) * 0.02)
            self.initial_bias = nn.Parameter(
                torch.zeros(self.program_states + 1)
            )
            self.completion_head = None
        self.transition = nn.Sequential(
            nn.LayerNorm(2 * hidden, elementwise_affine=False),
            nn.Linear(2 * hidden, hidden, bias=False),
            nn.SiLU(),
            nn.Linear(hidden, 2, bias=True),
        )
        # A mild stay prior makes the untrained recurrence stable without
        # imposing a phase distribution or a phase-usage target.
        with torch.no_grad():
            self.transition[-1].bias.copy_(torch.tensor([0.75, -0.75]))

        self.active_output = nn.Linear(hidden, hidden, bias=False)
        self.next_output = nn.Linear(hidden, hidden, bias=False)
        self.remaining_output = nn.Linear(hidden, hidden, bias=False)
        self.interval_query = nn.Linear(hidden, hidden, bias=False)
        self.phase_condition = nn.Sequential(
            nn.LayerNorm(3 * hidden + 1, elementwise_affine=False),
            nn.Linear(3 * hidden + 1, hidden, bias=False),
        )
        self.interval_phase_output = nn.Linear(hidden, hidden, bias=False)
        self.interval_goal_cross = nn.MultiheadAttention(
            hidden,
            heads,
            dropout=0.0,
            bias=False,
            batch_first=True,
        )
        self.interval_history_cross = nn.MultiheadAttention(
            hidden,
            heads,
            dropout=0.0,
            bias=False,
            batch_first=True,
        )
        self.goal_output = nn.Linear(hidden, hidden, bias=False)
        self.history_output = nn.Linear(hidden, hidden, bias=False)

    @staticmethod
    def _ordered_basis(count: int, hidden: int) -> Tensor:
        position = torch.linspace(0.0, 1.0, int(count))
        half = max(int(hidden) // 2, 1)
        exponent = torch.arange(half, dtype=torch.float32) / float(
            max(half - 1, 1)
        )
        frequency = torch.exp(
            -math.log(10_000.0) * exponent
        )
        angle = position[:, None] * frequency[None] * (2.0 * torch.pi)
        basis = torch.cat((angle.sin(), angle.cos()), dim=-1)
        if int(basis.shape[-1]) < int(hidden):
            basis = torch.nn.functional.pad(
                basis, (0, int(hidden) - int(basis.shape[-1]))
            )
        return basis[:, : int(hidden)]

    @staticmethod
    def _check_tokens(
        value: Tensor,
        *,
        batch: int,
        width: int,
        name: str,
    ) -> None:
        if (
            value.ndim != 3
            or int(value.shape[0]) != int(batch)
            or int(value.shape[1]) < 1
            or int(value.shape[2]) != int(width)
        ):
            raise ValueError(f"{name} must be non-empty [B,N,{width}]")

    @staticmethod
    def _resample_observable(value: Tensor, steps: int) -> Tensor:
        """Align observable histories without inventing a persistent cache."""

        if int(value.shape[1]) == int(steps):
            return value
        return torch.nn.functional.interpolate(
            value.float().transpose(1, 2),
            size=int(steps),
            mode="linear",
            align_corners=True,
        ).transpose(1, 2).to(dtype=value.dtype)

    def _replay_monotone_program(
        self,
        observations: Tensor,
        program: Tensor,
        *,
        final_program_grounding: Tensor | None = None,
    ) -> Tensor:
        batch, steps, _ = observations.shape
        obs_query = self.observation_query(
            self.observation_norm(observations)
        ).float()
        program_key = self.program_key(program).float()
        program_score = torch.einsum(
            "bth,bsh->bts", obs_query, program_key
        ) / math.sqrt(float(self.hidden))
        if self.separate_terminal:
            evidence_logits = program_score
        else:
            if self.terminal_key is None:
                raise RuntimeError("legacy phase replay has no terminal key")
            terminal_key = self.terminal_key.to(
                device=program.device,
                dtype=torch.float32,
            ).expand(batch, -1)
            terminal_score = torch.einsum(
                "bth,bh->bt", obs_query, terminal_key
            )[:, :, None] / math.sqrt(float(self.hidden))
            evidence_logits = torch.cat(
                (program_score, terminal_score), dim=-1
            )
        if final_program_grounding is not None:
            expected = (batch, self.program_states, self.hidden)
            if tuple(final_program_grounding.shape) != expected:
                raise ValueError(
                    "program-specific G3 grounding must be [B,state,H]"
                )
            grounding_query = self.observation_query(
                self.observation_norm(final_program_grounding)
            ).float()
            grounding_score = (
                grounding_query * program_key
            ).sum(dim=-1) / math.sqrt(float(self.hidden))
            final_evidence = (
                grounding_score
                if self.separate_terminal
                else torch.cat(
                    (
                        grounding_score,
                        grounding_score.new_zeros(batch, 1),
                    ),
                    dim=-1,
                )
            )
            final_step = torch.nn.functional.one_hot(
                torch.as_tensor(
                    steps - 1,
                    device=observations.device,
                    dtype=torch.long,
                ),
                num_classes=steps,
            ).to(dtype=evidence_logits.dtype)
            evidence_logits = evidence_logits + (
                final_step[None, :, None]
                * final_evidence[:, None, :]
            )
        belief = torch.softmax(
            evidence_logits[:, 0]
            + self.initial_bias.to(
                device=observations.device, dtype=torch.float32
            ),
            dim=-1,
        )
        for step in range(1, int(steps)):
            current = observations[:, step : step + 1].expand(
                -1, self.program_states, -1
            )
            transition = torch.softmax(
                self.transition(
                    torch.cat((current, program), dim=-1)
                ).float(),
                dim=-1,
            )
            stay = transition[..., 0]
            advance = transition[..., 1]
            propagated = torch.zeros_like(belief)
            active_belief = (
                belief
                if self.separate_terminal
                else belief[:, : self.program_states]
            )
            propagated[:, : self.program_states] = active_belief * stay
            propagated[:, 1 : self.program_states] = (
                propagated[:, 1 : self.program_states]
                + active_belief[:, : self.program_states - 1]
                * advance[:, : self.program_states - 1]
            )
            if self.separate_terminal:
                # Advancing from the last program state stays in that state;
                # completion is separate evidence and never absorbs belief.
                propagated[:, -1] = (
                    propagated[:, -1]
                    + active_belief[:, -1] * advance[:, -1]
                )
            else:
                propagated[:, -1] = (
                    belief[:, -1]
                    + active_belief[:, -1] * advance[:, -1]
                )
            evidence = torch.softmax(evidence_logits[:, step], dim=-1)
            belief = propagated * evidence
            belief = belief / belief.sum(dim=-1, keepdim=True).clamp_min(
                1e-8
            )
        return belief

    @staticmethod
    def _adjacent_cosine(value: Tensor) -> Tensor:
        if int(value.shape[1]) < 2:
            return value.new_ones((), dtype=torch.float32)
        return torch.nn.functional.cosine_similarity(
            value[:, 1:].float(),
            value[:, :-1].float(),
            dim=-1,
            eps=1e-6,
        ).mean()

    def forward(
        self,
        *,
        goal_tokens: Tensor,
        state_history_tokens: Tensor,
        history_tokens: Tensor,
        grounding_tokens: Tensor,
        collect_diagnostics: bool = True,
    ) -> tuple[GoalPhaseState, dict[str, Tensor]]:
        batch = int(goal_tokens.shape[0])
        for name, value in (
            ("goal program", goal_tokens),
            ("state history", state_history_tokens),
            ("executed action history", history_tokens),
            ("G3 grounding", grounding_tokens),
        ):
            self._check_tokens(
                value,
                batch=batch,
                width=(
                    self.state_dim
                    if name == "state history"
                    else self.action_dim
                    if name == "executed action history"
                    else self.hidden
                ),
                name=name,
            )

        program_basis = self.ordered_program_basis.to(
            device=goal_tokens.device, dtype=goal_tokens.dtype
        )
        program_query = self.program_query(program_basis)[None].expand(
            batch, -1, -1
        )
        goal_memory = self.goal_norm(goal_tokens)
        goal_read, _ = self.goal_cross(
            program_query,
            goal_memory,
            goal_memory,
            need_weights=False,
        )
        program = program_query + goal_read
        program = program + self.program_ffn(program)

        typed_state = (
            self.state_input(state_history_tokens)
            + self.state_type.to(
                device=state_history_tokens.device,
                dtype=state_history_tokens.dtype,
            )
        )
        typed_history = (
            self.action_input(history_tokens)
            + self.action_type.to(
                device=history_tokens.device,
                dtype=history_tokens.dtype,
            )
        )
        grounding_memory = (
            self.grounding_input(grounding_tokens)
            + self.grounding_type.to(
                device=grounding_tokens.device,
                dtype=grounding_tokens.dtype,
            )
        )
        # Read one goal-independent factual summary for the replay timeline
        # and one G3 view per ordered program state for phase evidence.  A
        # mean over the four program queries would prematurely publicize the
        # very stage-specific fact compatibility this machine must retain.
        grounding_query = torch.cat(
            (
                self.grounding_summary_query.to(
                    device=program.device, dtype=program.dtype
                ).expand(batch, -1, -1),
                self.grounding_observation_query(program),
            ),
            dim=1,
        )
        grounding_reads, _ = self.grounding_cross(
            grounding_query,
            self.grounding_norm(grounding_memory),
            self.grounding_norm(grounding_memory),
            need_weights=False,
        )
        typed_grounding = grounding_reads[:, :1]
        program_grounding = grounding_reads[:, 1:]
        # State and executed-action histories are different modalities over
        # the same elapsed past.  Align their temporal axes, retain their
        # types through concatenation, and fuse once per time step.  Appending
        # all state tokens before all action tokens would falsely replay a
        # modality ordering as if it were time.
        replay_steps = max(
            int(typed_state.shape[1]),
            int(typed_history.shape[1]),
        )
        typed_state = self._resample_observable(
            typed_state, replay_steps
        )
        typed_history = self._resample_observable(
            typed_history, replay_steps
        )
        observations = self.observation_fuse(
            torch.cat((typed_state, typed_history), dim=-1)
        )
        time_basis = self._ordered_basis(
            replay_steps + 1, self.hidden
        ).to(device=observations.device, dtype=observations.dtype)
        observations = observations + self.observation_time(
            time_basis[:-1]
        )[None]
        current_grounding = typed_grounding + self.observation_time(
            time_basis[-1:]
        )[None]
        observations = torch.cat(
            (observations, current_grounding), dim=1
        )
        length = int(observations.shape[1])
        causal_mask = torch.triu(
            torch.ones(
                length,
                length,
                device=observations.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        observations = self.observation_encoder(
            observations,
            mask=causal_mask,
        )
        belief = self._replay_monotone_program(
            observations,
            program,
            final_program_grounding=program_grounding,
        )
        active_weights = (
            belief
            if self.separate_terminal
            else belief[:, : self.program_states]
        )
        active_goal = self.active_output(
            torch.einsum("bs,bsh->bh", active_weights, program)
        )
        next_program = torch.cat(
            (program[:, 1:], program[:, -1:]), dim=1
        )
        next_goal = self.next_output(
            torch.einsum("bs,bsh->bh", active_weights, next_program)
        )
        suffix = torch.stack(
            [
                program[:, index:].mean(dim=1)
                for index in range(self.program_states)
            ],
            dim=1,
        )
        remaining_goal = self.remaining_output(
            torch.einsum("bs,bsh->bh", active_weights, suffix)
        )
        entropy_states = (
            self.program_states
            if self.separate_terminal
            else self.program_states + 1
        )
        entropy = -(
            belief * belief.clamp_min(1e-8).log()
        ).sum(dim=-1, keepdim=True) / math.log(
            float(entropy_states)
        )
        if self.separate_terminal:
            if self.completion_head is None:
                raise RuntimeError("separate terminal completion head is missing")
            completion_evidence = torch.sigmoid(
                self.completion_head(
                    torch.cat(
                        (observations[:, -1], program[:, -1]),
                        dim=-1,
                    )
                ).float()
            ).to(dtype=goal_tokens.dtype)
            terminal_probability = (
                belief[:, -1:].to(dtype=goal_tokens.dtype)
                * completion_evidence
            )
        else:
            terminal_probability = belief[:, -1:].to(dtype=goal_tokens.dtype)

        interval_basis = self.ordered_interval_basis.to(
            device=goal_tokens.device, dtype=goal_tokens.dtype
        )
        interval_query = self.interval_query(interval_basis)[None].expand(
            batch, -1, -1
        )
        typed_phase = self.phase_condition(
            torch.cat(
                (
                    active_goal,
                    next_goal,
                    remaining_goal,
                    entropy.to(dtype=active_goal.dtype),
                ),
                dim=-1,
            )
        )
        interval_selector = self.interval_phase_output(
            interval_query + typed_phase[:, None]
        )
        goal_context, _ = self.interval_goal_cross(
            interval_query + active_goal[:, None],
            program,
            program,
            need_weights=False,
        )
        history_context, _ = self.interval_history_cross(
            interval_query + remaining_goal[:, None],
            observations,
            observations,
            need_weights=False,
        )
        goal_context = self.goal_output(goal_context)
        history_context = self.history_output(history_context)

        state = GoalPhaseState(
            active_goal=active_goal,
            next_goal=next_goal,
            remaining_goal=remaining_goal,
            phase_belief=belief.to(dtype=goal_tokens.dtype),
            phase_uncertainty=entropy.to(dtype=goal_tokens.dtype),
            interval_selector=interval_selector,
            goal_context=goal_context,
            history_context=history_context,
            goal_program=program,
            terminal_probability=terminal_probability,
        )
        state.validate(
            batch=batch,
            program_states=self.program_states,
            intervals=self.intervals,
            hidden=self.hidden,
        )
        if not collect_diagnostics:
            return state, {}
        detached_belief = belief.detach()
        expected = (
            detached_belief[:, : self.program_states]
            * torch.arange(
                self.program_states,
                device=belief.device,
                dtype=belief.dtype,
            )[None]
        ).sum(dim=-1)
        terminal_mass = (
            terminal_probability.detach().float().mean()
            if terminal_probability is not None
            else detached_belief[:, -1].float().mean()
        )
        metrics = {
            "flow_jepa_goal_phase_machine_active": belief.new_ones(()),
            "flow_jepa_phase_entropy": entropy.detach().mean(),
            "flow_jepa_phase_max": detached_belief.amax(dim=-1).mean(),
            # V116 separates completion from the fourth program state.  Keep
            # the historical metric name for log compatibility, but report
            # the actual terminal head rather than mislabelling state four as
            # terminal.  V115 and earlier retain their exact old semantics.
            "flow_jepa_phase_terminal_mass": terminal_mass,
            "flow_jepa_phase_expected_index": expected.mean(),
            "flow_jepa_phase_expected_index_std": expected.std(
                unbiased=False
            ),
            "flow_jepa_phase_replay_steps": belief.new_tensor(
                float(replay_steps + 1)
            ),
            "flow_jepa_phase_context_norm": (
                interval_selector.detach().float().norm(dim=-1).mean()
            ),
            "flow_jepa_phase_program_grounding_variation": (
                program_grounding.detach().float().std(
                    dim=1, unbiased=False
                ).mean()
            ),
            "flow_jepa_goal_selector_context_norm": (
                goal_context.detach().float().norm(dim=-1).mean()
            ),
            "flow_jepa_history_selector_context_norm": (
                history_context.detach().float().norm(dim=-1).mean()
            ),
            "flow_jepa_phase_horizon_adjacent_cosine": (
                self._adjacent_cosine(interval_selector.detach())
            ),
            "flow_jepa_goal_horizon_adjacent_cosine": (
                self._adjacent_cosine(goal_context.detach())
            ),
            "flow_jepa_history_horizon_adjacent_cosine": (
                self._adjacent_cosine(history_context.detach())
            ),
            "flow_jepa_phase_horizon_variation": (
                interval_selector.detach().float().std(
                    dim=1, unbiased=False
                ).mean()
            ),
            "flow_jepa_goal_horizon_variation": (
                goal_context.detach().float().std(
                    dim=1, unbiased=False
                ).mean()
            ),
            "flow_jepa_history_horizon_variation": (
                history_context.detach().float().std(
                    dim=1, unbiased=False
                ).mean()
            ),
        }
        for index in range(self.program_states):
            metrics[f"flow_jepa_phase_mass_{index}"] = (
                detached_belief[:, index].mean()
            )
        return state, metrics


@dataclass(frozen=True)
class StatelessIntentState:
    """V117 one-shot goal/progress control with typed downstream operands.

    The four-state belief is an audit view over the active-program attention,
    not a recurrent state and not the sole carrier used by W or P.  Window,
    goal, history and temporal controls remain separately addressable.
    """

    active_intent: Tensor
    next_intent: Tensor
    remaining_intent: Tensor
    phase_belief: Tensor
    phase_uncertainty: Tensor
    window_selector: Tensor
    window_context: Tensor
    temporal_control: Tensor
    world_context: Tensor
    goal_context: Tensor
    history_context: Tensor
    goal_program: Tensor
    progress_coordinate: Tensor
    terminal_probability: Tensor

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
    def interval_selector(self) -> Tensor:
        """Compatibility name for the four online W-anchor phase contexts."""

        return self.world_context

    def validate(
        self,
        *,
        batch: int,
        program_states: int,
        intervals: int,
        hidden: int,
    ) -> None:
        vector_shape = (batch, hidden)
        for name in ("active_intent", "next_intent", "remaining_intent"):
            if tuple(getattr(self, name).shape) != vector_shape:
                raise ValueError(f"{name} must be [B,H]")
        if tuple(self.phase_belief.shape) != (batch, program_states):
            raise ValueError("phase_belief must be [B,program]")
        for name in (
            "phase_uncertainty",
            "progress_coordinate",
            "terminal_probability",
        ):
            if tuple(getattr(self, name).shape) != (batch, 1):
                raise ValueError(f"{name} must be [B,1]")
        windows = int(self.window_selector.shape[1])
        if tuple(self.window_selector.shape) != (batch, windows) or windows < 1:
            raise ValueError("window_selector must be non-empty [B,W]")
        if tuple(self.window_context.shape) != (batch, windows, hidden):
            raise ValueError("window_context must be [B,W,H]")
        if (
            self.temporal_control.ndim != 3
            or tuple(self.temporal_control.shape[::2]) != (batch, hidden)
        ):
            raise ValueError("temporal_control must be [B,T,H]")
        for name in ("world_context", "goal_context", "history_context"):
            if tuple(getattr(self, name).shape) != (batch, intervals, hidden):
                raise ValueError(f"{name} must be [B,A,H]")
        if tuple(self.goal_program.shape) != (batch, program_states, hidden):
            raise ValueError("goal_program must be [B,program,H]")
        for name, value in self.__dict__.items():
            if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} contains NaN or infinity")


class StatelessIntentController(nn.Module):
    """Three compact blocks and a small MLP for observable intent control.

    S1 compiles ordered program tokens from the complete T5 bank. S2 encodes
    aligned state/executed-action history with a causal mask. S3 performs a
    one-shot typed control read.  There is no repeated probability product,
    persistent episode cache, future teacher, or noisy-action input.
    """

    def __init__(
        self,
        hidden: int,
        program_states: int,
        world_intervals: int,
        action_horizon: int,
        *,
        state_dim: int,
        action_dim: int,
        control_hidden: int = 256,
        heads: int = 4,
        windows: int = 3,
    ) -> None:
        super().__init__()
        if min(
            hidden,
            program_states,
            world_intervals,
            action_horizon,
            state_dim,
            action_dim,
            control_hidden,
            heads,
            windows,
        ) < 1:
            raise ValueError("stateless intent dimensions must be positive")
        if control_hidden % heads:
            raise ValueError("intent control width must divide attention heads")
        if program_states < 2 or windows != 3:
            raise ValueError("V117 requires at least two program states and three windows")
        self.hidden = int(hidden)
        self.control_hidden = int(control_hidden)
        self.program_states = int(program_states)
        self.world_intervals = int(world_intervals)
        self.action_horizon = int(action_horizon)
        self.windows = int(windows)

        program_basis = StatelessGoalPhaseMachine._ordered_basis(
            self.program_states, self.control_hidden
        )
        window_basis = StatelessGoalPhaseMachine._ordered_basis(
            self.windows, self.control_hidden
        )
        self.register_buffer("ordered_program_basis", program_basis)
        self.register_buffer("ordered_window_basis", window_basis)
        self.register_buffer(
            "program_position", torch.linspace(0.0, 1.0, self.program_states)
        )

        # S1: protected ordered basis plus a language innovation.
        self.goal_input = nn.Linear(hidden, self.control_hidden, bias=False)
        self.program_query = nn.Linear(
            self.control_hidden, self.control_hidden, bias=False
        )
        self.goal_norm = nn.LayerNorm(
            self.control_hidden, elementwise_affine=False
        )
        self.goal_cross = nn.MultiheadAttention(
            self.control_hidden,
            heads,
            dropout=0.0,
            bias=False,
            batch_first=True,
        )
        self.program_ffn = nn.Sequential(
            nn.LayerNorm(self.control_hidden, elementwise_affine=False),
            nn.Linear(self.control_hidden, 2 * self.control_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(2 * self.control_hidden, self.control_hidden, bias=False),
        )

        # S2: one causal observable-history block. State and executed action
        # are aligned in time before fusion; G3 is appended as the current fact.
        self.state_input = nn.Linear(state_dim, self.control_hidden, bias=False)
        self.action_input = nn.Linear(action_dim, self.control_hidden, bias=False)
        self.grounding_input = nn.Linear(hidden, self.control_hidden, bias=False)
        self.state_type = nn.Parameter(
            torch.randn(1, 1, self.control_hidden) * 0.02
        )
        self.action_type = nn.Parameter(
            torch.randn(1, 1, self.control_hidden) * 0.02
        )
        self.grounding_type = nn.Parameter(
            torch.randn(1, 1, self.control_hidden) * 0.02
        )
        self.history_fuse = nn.Sequential(
            nn.LayerNorm(2 * self.control_hidden, elementwise_affine=False),
            nn.Linear(
                2 * self.control_hidden, 2 * self.control_hidden, bias=False
            ),
            nn.SiLU(),
            nn.Linear(2 * self.control_hidden, self.control_hidden, bias=False),
        )
        self.grounding_query = nn.Parameter(
            torch.randn(1, 1, self.control_hidden) * 0.02
        )
        self.grounding_cross = nn.MultiheadAttention(
            self.control_hidden,
            heads,
            dropout=0.0,
            bias=False,
            batch_first=True,
        )
        self.history_time = nn.Linear(
            self.control_hidden, self.control_hidden, bias=False
        )
        self.history_block = nn.TransformerEncoderLayer(
            d_model=self.control_hidden,
            nhead=heads,
            dim_feedforward=2 * self.control_hidden,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            bias=False,
        )

        # S3: role-typed one-shot reads. Program attention remains soft and
        # receives a smooth monotone position bias, never a hard state update.
        self.control_role_query = nn.Parameter(
            torch.randn(1, self.windows, self.control_hidden) * 0.02
        )
        self.control_cross = nn.MultiheadAttention(
            self.control_hidden,
            heads,
            dropout=0.0,
            bias=False,
            batch_first=True,
        )
        self.control_query = nn.ModuleList(
            [
                nn.Linear(self.control_hidden, self.control_hidden, bias=False)
                for _ in range(self.windows)
            ]
        )
        self.program_key = nn.Linear(
            self.control_hidden, self.control_hidden, bias=False
        )
        self.progress_head = nn.Sequential(
            nn.LayerNorm(self.control_hidden, elementwise_affine=False),
            nn.Linear(self.control_hidden, 2 * self.control_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(2 * self.control_hidden, 2, bias=True),
        )
        nn.init.zeros_(self.progress_head[-1].weight)
        nn.init.zeros_(self.progress_head[-1].bias)
        self.observable_role = nn.ModuleList(
            [
                nn.Linear(self.control_hidden, self.control_hidden, bias=False)
                for _ in range(self.windows)
            ]
        )
        self.intent_output = nn.ModuleList(
            [self._output_head(self.control_hidden, hidden) for _ in range(3)]
        )
        self.phase_output = self._output_head(self.control_hidden, hidden)
        self.goal_output = self._output_head(self.control_hidden, hidden)
        self.history_output = self._output_head(self.control_hidden, hidden)
        self.window_score = nn.Sequential(
            nn.LayerNorm(3 * self.control_hidden, elementwise_affine=False),
            nn.Linear(3 * self.control_hidden, self.control_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(self.control_hidden, 1, bias=False),
        )
        self.completion_head = nn.Sequential(
            nn.LayerNorm(2 * self.control_hidden, elementwise_affine=False),
            nn.Linear(2 * self.control_hidden, self.control_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(self.control_hidden, 1, bias=True),
        )
        nn.init.normal_(self.completion_head[-1].weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.completion_head[-1].bias, -2.5)

    @staticmethod
    def _output_head(input_width: int, hidden: int) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(input_width, elementwise_affine=False),
            nn.Linear(input_width, hidden, bias=False),
        )

    @staticmethod
    def _resample(value: Tensor, steps: int) -> Tensor:
        if int(value.shape[1]) == int(steps):
            return value
        return F.interpolate(
            value.float().transpose(1, 2),
            size=int(steps),
            mode="linear",
            align_corners=True,
        ).transpose(1, 2).to(dtype=value.dtype)

    @staticmethod
    def _resize_time(value: Tensor, steps: int) -> Tensor:
        return F.interpolate(
            value.float().transpose(1, 2),
            size=int(steps),
            mode="linear",
            align_corners=True,
        ).transpose(1, 2).to(dtype=value.dtype)

    @staticmethod
    def _adjacent_cosine(value: Tensor) -> Tensor:
        return F.cosine_similarity(
            value[:, 1:].float(),
            value[:, :-1].float(),
            dim=-1,
            eps=1e-6,
        ).mean()

    def forward(
        self,
        *,
        goal_tokens: Tensor,
        state_history_tokens: Tensor,
        history_tokens: Tensor,
        grounding_tokens: Tensor,
        collect_diagnostics: bool = True,
    ) -> tuple[StatelessIntentState, dict[str, Tensor]]:
        batch = int(goal_tokens.shape[0])
        expected_widths = (
            ("goal", goal_tokens, self.hidden),
            ("state", state_history_tokens, self.state_input.in_features),
            ("history", history_tokens, self.action_input.in_features),
            ("grounding", grounding_tokens, self.hidden),
        )
        for name, value, width in expected_widths:
            if (
                value.ndim != 3
                or int(value.shape[0]) != batch
                or int(value.shape[1]) < 1
                or int(value.shape[2]) != int(width)
            ):
                raise ValueError(f"intent {name} tokens must be non-empty [B,N,{width}]")

        program_basis = self.ordered_program_basis.to(
            device=goal_tokens.device, dtype=goal_tokens.dtype
        )
        protected_program = self.program_query(program_basis)[None].expand(
            batch, -1, -1
        )
        goal_memory = self.goal_input(goal_tokens)
        language_innovation, _ = self.goal_cross(
            protected_program,
            self.goal_norm(goal_memory),
            self.goal_norm(goal_memory),
            need_weights=False,
        )
        program = protected_program + language_innovation
        program = program + self.program_ffn(program)

        state = self.state_input(state_history_tokens) + self.state_type.to(
            device=state_history_tokens.device, dtype=state_history_tokens.dtype
        )
        history = self.action_input(history_tokens) + self.action_type.to(
            device=history_tokens.device, dtype=history_tokens.dtype
        )
        replay_steps = max(int(state.shape[1]), int(history.shape[1]))
        state = self._resample(state, replay_steps)
        history = self._resample(history, replay_steps)
        observations = self.history_fuse(torch.cat((state, history), dim=-1))

        grounding = self.grounding_input(grounding_tokens) + self.grounding_type.to(
            device=grounding_tokens.device, dtype=grounding_tokens.dtype
        )
        grounding_summary, _ = self.grounding_cross(
            self.grounding_query.to(
                device=grounding.device, dtype=grounding.dtype
            ).expand(batch, -1, -1),
            grounding,
            grounding,
            need_weights=False,
        )
        time_basis = StatelessGoalPhaseMachine._ordered_basis(
            replay_steps + 1, self.control_hidden
        ).to(device=observations.device, dtype=observations.dtype)
        observations = observations + self.history_time(time_basis[:-1])[None]
        observations = torch.cat(
            (
                observations,
                grounding_summary + self.history_time(time_basis[-1:])[None],
            ),
            dim=1,
        )
        length = int(observations.shape[1])
        causal_mask = torch.triu(
            torch.ones(
                length,
                length,
                device=observations.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        observations = self.history_block(observations, src_mask=causal_mask)
        current_observation = observations[:, -1]

        role_query = self.control_role_query.to(
            device=program.device, dtype=program.dtype
        ).expand(batch, -1, -1)
        role_query = role_query + current_observation[:, None]
        program_read, _ = self.control_cross(
            role_query,
            program,
            program,
            need_weights=False,
        )
        progress_raw = self.progress_head(current_observation).float()
        progress = torch.sigmoid(progress_raw[:, :1])
        scale = 0.10 + 0.40 * torch.sigmoid(progress_raw[:, 1:2])
        positions = self.program_position.to(
            device=program.device, dtype=torch.float32
        )[None]
        targets = torch.cat(
            (
                progress,
                (progress + 1.0 / 3.0).clamp(max=1.0),
                progress,
            ),
            dim=-1,
        )
        program_key = self.program_key(program).float()
        weight_rows: list[Tensor] = []
        weighted_program_rows: list[Tensor] = []
        observable_rows: list[Tensor] = []
        for role_index in range(self.windows):
            query = self.control_query[role_index](
                program_read[:, role_index]
            ).float()
            content_logit = torch.einsum(
                "bh,bsh->bs", query, program_key
            ) / math.sqrt(float(self.control_hidden))
            if role_index < 2:
                position_bias = -0.5 * (
                    (positions - targets[:, role_index : role_index + 1])
                    / scale
                ).square()
            else:
                position_bias = torch.nn.functional.logsigmoid(
                    (positions - progress) / scale
                )
            weights = torch.softmax(content_logit + position_bias, dim=-1)
            weighted_program = torch.einsum("bs,bsh->bh", weights, program)
            observable = self.observable_role[role_index](current_observation)
            weight_rows.append(weights)
            weighted_program_rows.append(weighted_program)
            observable_rows.append(observable)

        weights = torch.stack(weight_rows, dim=1)
        weighted_program = torch.stack(weighted_program_rows, dim=1)
        observable_role = torch.stack(observable_rows, dim=1)
        control_role = (
            program_read + weighted_program + observable_role
        ) / math.sqrt(3.0)
        intent_window = torch.stack(
            tuple(
                self.intent_output[index](control_role[:, index])
                for index in range(3)
            ),
            dim=1,
        )
        active_intent, next_intent, remaining_intent = intent_window.unbind(
            dim=1
        )
        phase_window = self.phase_output(
            control_role
            + self.ordered_window_basis.to(
                device=control_role.device, dtype=control_role.dtype
            )[None]
        )
        # Project the complete program with the same trainable map used by the
        # three soft reads.  No trainable display-only projection is allowed.
        goal_program = self.goal_output(program)
        goal_window = self.goal_output(weighted_program)
        history_window = self.history_output(observable_role)
        window_context = (
            intent_window + phase_window + goal_window + history_window
        ) / 2.0
        window_selector = torch.softmax(
            self.window_score(
                torch.cat(
                    (program_read, weighted_program, observable_role), dim=-1
                )
            ).squeeze(-1).float(),
            dim=-1,
        ).to(dtype=window_context.dtype)
        world_context = self._resize_time(
            phase_window, self.world_intervals
        )
        goal_context = self._resize_time(
            goal_window, self.world_intervals
        )
        history_context = self._resize_time(
            history_window, self.world_intervals
        )
        temporal_control = self._resize_time(
            window_context, self.action_horizon
        )
        belief = weights[:, 0]
        entropy = -(
            belief.float() * belief.float().clamp_min(1e-8).log()
        ).sum(dim=-1, keepdim=True) / math.log(float(self.program_states))
        terminal_probability = torch.sigmoid(
            self.completion_head(
                torch.cat(
                    (current_observation, control_role[:, 2]), dim=-1
                )
            ).float()
        ).to(dtype=goal_tokens.dtype)
        state_out = StatelessIntentState(
            active_intent=active_intent,
            next_intent=next_intent,
            remaining_intent=remaining_intent,
            phase_belief=belief.to(dtype=goal_tokens.dtype),
            phase_uncertainty=entropy.to(dtype=goal_tokens.dtype),
            window_selector=window_selector,
            window_context=window_context,
            temporal_control=temporal_control,
            world_context=world_context,
            goal_context=goal_context,
            history_context=history_context,
            goal_program=goal_program,
            progress_coordinate=progress.to(dtype=goal_tokens.dtype),
            terminal_probability=terminal_probability,
        )
        state_out.validate(
            batch=batch,
            program_states=self.program_states,
            intervals=self.world_intervals,
            hidden=self.hidden,
        )
        if not collect_diagnostics:
            return state_out, {}
        metrics = {
            "flow_jepa_stateless_intent_controller_active": belief.new_ones(()),
            "flow_jepa_intent_progress_coordinate": progress.detach().mean(),
            "flow_jepa_intent_progress_scale": scale.detach().mean(),
            "flow_jepa_phase_entropy": entropy.detach().mean(),
            "flow_jepa_phase_max": belief.detach().amax(dim=-1).mean(),
            "flow_jepa_phase_expected_index": (
                belief.detach()
                * torch.arange(
                    self.program_states,
                    device=belief.device,
                    dtype=belief.dtype,
                )[None]
            ).sum(dim=-1).mean(),
            "flow_jepa_phase_terminal_mass": terminal_probability.detach().float().mean(),
            "flow_jepa_intent_window_selector_max": window_selector.detach().float().amax(dim=-1).mean(),
            "flow_jepa_intent_window_selector_entropy": -(
                window_selector.detach().float()
                * window_selector.detach().float().clamp_min(1e-8).log()
            ).sum(dim=-1).mean() / math.log(float(self.windows)),
            "flow_jepa_intent_window_adjacent_cosine": self._adjacent_cosine(
                window_context.detach()
            ),
            "flow_jepa_intent_program_adjacent_cosine": self._adjacent_cosine(
                state_out.goal_program.detach()
            ),
            "flow_jepa_goal_selector_context_norm": goal_context.detach().float().norm(dim=-1).mean(),
            "flow_jepa_history_selector_context_norm": history_context.detach().float().norm(dim=-1).mean(),
            "flow_jepa_phase_context_norm": world_context.detach().float().norm(dim=-1).mean(),
            "flow_jepa_intent_observation_steps": belief.new_tensor(
                float(replay_steps + 1)
            ),
        }
        for index in range(self.program_states):
            metrics[f"flow_jepa_phase_mass_{index}"] = belief.detach()[:, index].mean()
        for index, name in enumerate(("active", "next", "remaining")):
            metrics[f"flow_jepa_intent_{name}_norm"] = (
                getattr(state_out, f"{name}_intent")
                .detach()
                .float()
                .norm(dim=-1)
                .mean()
            )
        return state_out, metrics


class GoalResamplerBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, expansion: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden)
        self.memory_norm = nn.LayerNorm(hidden)
        self.cross = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, int(hidden * expansion)),
            nn.GELU(),
            nn.Linear(int(hidden * expansion), hidden),
        )

    def forward(self, query: Tensor, memory: Tensor, mask: Tensor) -> Tensor:
        normalized_memory = self.memory_norm(memory)
        update, _ = self.cross(
            self.query_norm(query),
            normalized_memory,
            normalized_memory,
            key_padding_mask=~mask,
            need_weights=False,
        )
        query = query + update
        return query + self.ffn(query)


class GoalTokenResampler(nn.Module):
    """Resample frozen language embeddings into a few trainable goal tokens."""

    def __init__(
        self,
        *,
        language_dim: int,
        hidden: int,
        goal_tokens: int,
        heads: int,
        depth: int,
        expansion: float,
    ) -> None:
        super().__init__()
        if min(language_dim, hidden, goal_tokens, heads, depth) <= 0:
            raise ValueError("goal resampler dimensions must be positive")
        self.language_dim = int(language_dim)
        self.hidden = int(hidden)
        self.goal_tokens = int(goal_tokens)
        self.input = nn.Sequential(
            nn.LayerNorm(language_dim),
            nn.Linear(language_dim, hidden),
        )
        self.query = nn.Parameter(torch.randn(1, goal_tokens, hidden) * 0.02)
        self.blocks = nn.ModuleList(
            [
                GoalResamplerBlock(hidden, heads, expansion)
                for _ in range(depth)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden)

    def forward(self, language_tokens: Tensor, language_mask: Tensor) -> Tensor:
        if language_tokens.ndim != 3 or int(language_tokens.shape[-1]) != self.language_dim:
            raise ValueError(
                f"language_tokens must be [B,L,{self.language_dim}], got "
                f"{tuple(language_tokens.shape)}"
            )
        if tuple(language_mask.shape) != tuple(language_tokens.shape[:2]):
            raise ValueError("language_mask must align with language_tokens as [B,L]")
        mask = language_mask.to(device=language_tokens.device, dtype=torch.bool)
        if not bool(mask.any(dim=1).all()):
            raise ValueError("every sample needs at least one valid language token")
        memory = self.input(language_tokens)
        query = self.query.expand(language_tokens.shape[0], -1, -1).to(
            device=memory.device, dtype=memory.dtype
        )
        for block in self.blocks:
            query = block(query, memory, mask)
        return self.output_norm(query)


class StatelessPhaseAdapter(nn.Module):
    """Infer an ordered soft phase belief without recurrent deployment state.

    The returned context is selector-only: callers may add it to world or
    spatial-address queries, but it must not be registered as a global semantic
    value or a direct action writer.  Ordered sinusoidal phase bases keep the
    phase axis meaningful instead of turning it into another unconstrained
    bank of free value tokens.
    """

    def __init__(self, hidden: int, phase_count: int) -> None:
        super().__init__()
        if int(hidden) < 1 or int(phase_count) < 2:
            raise ValueError("stateless phase dimensions are invalid")
        self.hidden = int(hidden)
        self.phase_count = int(phase_count)
        self.condition = nn.Sequential(
            nn.LayerNorm(4 * self.hidden),
            nn.Linear(4 * self.hidden, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.phase_count),
        )
        self.context_proj = nn.Linear(self.hidden, self.hidden, bias=False)
        self.selector_condition_proj = nn.Sequential(
            nn.LayerNorm(2 * self.hidden),
            nn.Linear(2 * self.hidden, self.hidden, bias=False),
        )
        phase = torch.linspace(0.0, 1.0, self.phase_count)
        half = max(self.hidden // 2, 1)
        exponent = torch.arange(half, dtype=torch.float32) / float(max(half - 1, 1))
        frequency = torch.exp(
            -torch.log(torch.tensor(10_000.0, dtype=torch.float32)) * exponent
        )
        angle = phase[:, None] * frequency[None] * (2.0 * torch.pi)
        basis = torch.cat((angle.sin(), angle.cos()), dim=-1)
        if int(basis.shape[-1]) < self.hidden:
            basis = torch.nn.functional.pad(
                basis, (0, self.hidden - int(basis.shape[-1]))
            )
        self.register_buffer("ordered_phase_basis", basis[:, : self.hidden])

    @staticmethod
    def _summary(value: Tensor, *, batch: int, hidden: int, name: str) -> Tensor:
        if (
            value.ndim != 3
            or int(value.shape[0]) != int(batch)
            or int(value.shape[-1]) != int(hidden)
            or int(value.shape[1]) <= 0
        ):
            raise ValueError(f"{name} must be non-empty [B,N,{hidden}]")
        return value.mean(dim=1)

    def forward(
        self,
        *,
        goal_tokens: Tensor,
        history_tokens: Tensor,
        state_tokens: Tensor,
        visual_tokens: Tensor,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        batch = int(state_tokens.shape[0])
        summaries = (
            self._summary(
                goal_tokens, batch=batch, hidden=self.hidden, name="phase goal"
            ),
            self._summary(
                history_tokens,
                batch=batch,
                hidden=self.hidden,
                name="phase history",
            ),
            self._summary(
                state_tokens, batch=batch, hidden=self.hidden, name="phase state"
            ),
            self._summary(
                visual_tokens, batch=batch, hidden=self.hidden, name="phase visual"
            ),
        )
        logits = self.condition(torch.cat(summaries, dim=-1))
        belief = torch.softmax(logits.float(), dim=-1)
        basis = self.ordered_phase_basis.to(
            device=logits.device, dtype=logits.dtype
        )
        phase_context = belief.to(dtype=logits.dtype) @ basis
        phase_context = self.context_proj(phase_context)
        selector_condition = self.selector_condition_proj(
            torch.cat((summaries[0], summaries[1]), dim=-1)
        )
        entropy = -(
            belief.detach() * belief.detach().clamp_min(1e-8).log()
        ).sum(dim=-1)
        entropy = entropy / torch.log(
            belief.new_tensor(float(self.phase_count))
        )
        phase_index = torch.arange(
            self.phase_count, device=belief.device, dtype=belief.dtype
        )
        expectation = (belief * phase_index[None]).sum(dim=-1)
        metrics = {
            "flow_jepa_phase_entropy": entropy.mean(),
            "flow_jepa_phase_max": belief.detach().amax(dim=-1).mean(),
            "flow_jepa_phase_expected_index": expectation.detach().mean(),
            "flow_jepa_phase_expected_index_std": expectation.detach().std(
                unbiased=False
            ),
            "flow_jepa_phase_context_norm": (
                phase_context.detach().float().norm(dim=-1).mean()
            ),
            "flow_jepa_condition_selector_context_norm": (
                selector_condition.detach().float().norm(dim=-1).mean()
            ),
        }
        for index in range(self.phase_count):
            metrics[f"flow_jepa_phase_mass_{index}"] = (
                belief.detach()[:, index].mean()
            )
        return phase_context, selector_condition, metrics


class StatelessHorizonConditionAdapter(nn.Module):
    """Build distinct online selector contexts for each future interval.

    Goal and executed-action history remain separate typed operands.  Four
    ordered horizon queries cross-read the complete compact token banks instead
    of mean-pooling them into one global vector.  State and visual observations
    provide a small online phase prior, but no recurrent state or future teacher
    enters this path.
    """

    def __init__(self, hidden: int, horizon_count: int, heads: int) -> None:
        super().__init__()
        if min(int(hidden), int(horizon_count), int(heads)) < 1:
            raise ValueError("horizon condition dimensions must be positive")
        if int(hidden) % int(heads):
            raise ValueError("horizon condition hidden size must divide heads")
        self.hidden = int(hidden)
        self.horizon_count = int(horizon_count)
        phase = torch.linspace(0.0, 1.0, self.horizon_count)
        half = max(self.hidden // 2, 1)
        exponent = torch.arange(half, dtype=torch.float32) / float(
            max(half - 1, 1)
        )
        frequency = torch.exp(
            -torch.log(torch.tensor(10_000.0, dtype=torch.float32)) * exponent
        )
        angle = phase[:, None] * frequency[None] * (2.0 * torch.pi)
        basis = torch.cat((angle.sin(), angle.cos()), dim=-1)
        if int(basis.shape[-1]) < self.hidden:
            basis = torch.nn.functional.pad(
                basis, (0, self.hidden - int(basis.shape[-1]))
            )
        self.register_buffer("ordered_horizon_basis", basis[:, : self.hidden])

        self.anchor_query = nn.Linear(self.hidden, self.hidden, bias=False)
        self.goal_norm = nn.LayerNorm(self.hidden, elementwise_affine=False)
        self.history_norm = nn.LayerNorm(self.hidden, elementwise_affine=False)
        self.goal_cross = nn.MultiheadAttention(
            self.hidden,
            int(heads),
            dropout=0.0,
            bias=False,
            batch_first=True,
        )
        self.history_cross = nn.MultiheadAttention(
            self.hidden,
            int(heads),
            dropout=0.0,
            bias=False,
            batch_first=True,
        )
        self.state_context = nn.Sequential(
            nn.LayerNorm(self.hidden, elementwise_affine=False),
            nn.Linear(self.hidden, self.hidden, bias=False),
        )
        self.visual_context = nn.Sequential(
            nn.LayerNorm(self.hidden, elementwise_affine=False),
            nn.Linear(self.hidden, self.hidden, bias=False),
        )
        self.phase_query = nn.Sequential(
            nn.LayerNorm(self.hidden, elementwise_affine=False),
            nn.Linear(self.hidden, self.hidden, bias=False),
        )
        self.phase_key = nn.Linear(self.hidden, self.hidden, bias=False)
        self.phase_output = nn.Linear(self.hidden, self.hidden, bias=False)
        self.goal_output = nn.Linear(self.hidden, self.hidden, bias=False)
        self.history_output = nn.Linear(self.hidden, self.hidden, bias=False)

    @staticmethod
    def _check_tokens(
        value: Tensor, *, batch: int, hidden: int, name: str
    ) -> None:
        if (
            value.ndim != 3
            or int(value.shape[0]) != int(batch)
            or int(value.shape[-1]) != int(hidden)
            or int(value.shape[1]) <= 0
        ):
            raise ValueError(f"{name} must be non-empty [B,N,{hidden}]")

    @staticmethod
    def _adjacent_cosine(value: Tensor) -> Tensor:
        if int(value.shape[1]) < 2:
            return value.new_ones((), dtype=torch.float32)
        return torch.nn.functional.cosine_similarity(
            value[:, 1:].float(),
            value[:, :-1].float(),
            dim=-1,
            eps=1e-6,
        ).mean()

    def forward(
        self,
        *,
        goal_tokens: Tensor,
        history_tokens: Tensor,
        state_tokens: Tensor,
        visual_tokens: Tensor,
        collect_diagnostics: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        batch = int(state_tokens.shape[0])
        for name, value in (
            ("horizon goal", goal_tokens),
            ("horizon history", history_tokens),
            ("horizon state", state_tokens),
            ("horizon visual", visual_tokens),
        ):
            self._check_tokens(
                value, batch=batch, hidden=self.hidden, name=name
            )

        basis = self.ordered_horizon_basis.to(
            device=state_tokens.device, dtype=state_tokens.dtype
        )
        anchor = self.anchor_query(basis)[None].expand(batch, -1, -1)
        state_summary = self.state_context(state_tokens.mean(dim=1))
        visual_summary = self.visual_context(visual_tokens.mean(dim=1))
        observation = (state_summary + visual_summary) / (2.0**0.5)
        selector_query = anchor + observation[:, None]

        goal_memory = self.goal_norm(goal_tokens)
        history_memory = self.history_norm(history_tokens)
        goal_read, _ = self.goal_cross(
            selector_query,
            goal_memory,
            goal_memory,
            need_weights=False,
        )
        history_read, _ = self.history_cross(
            selector_query,
            history_memory,
            history_memory,
            need_weights=False,
        )
        phase_query = self.phase_query(
            selector_query + (goal_read + history_read) / (2.0**0.5)
        )
        phase_key = self.phase_key(basis)
        phase_logits = torch.einsum(
            "bar,kr->bak", phase_query.float(), phase_key.float()
        ) / float(self.hidden) ** 0.5
        phase_belief = torch.softmax(phase_logits, dim=-1)
        phase_bank = self.phase_output(
            phase_belief.to(dtype=basis.dtype) @ basis
        )
        goal_bank = self.goal_output(goal_read)
        history_bank = self.history_output(history_read)

        if not collect_diagnostics:
            return phase_bank, goal_bank, history_bank, {}
        belief = phase_belief.detach()
        entropy = -(
            belief * belief.clamp_min(1e-8).log()
        ).sum(dim=-1) / torch.log(
            belief.new_tensor(float(max(self.horizon_count, 2)))
        )
        metrics = {
            "flow_jepa_phase_entropy": entropy.mean(),
            "flow_jepa_phase_max": belief.amax(dim=-1).mean(),
            "flow_jepa_phase_context_norm": (
                phase_bank.detach().float().norm(dim=-1).mean()
            ),
            "flow_jepa_goal_selector_context_norm": (
                goal_bank.detach().float().norm(dim=-1).mean()
            ),
            "flow_jepa_history_selector_context_norm": (
                history_bank.detach().float().norm(dim=-1).mean()
            ),
            "flow_jepa_condition_selector_context_norm": (
                ((goal_bank + history_bank) / (2.0**0.5))
                .detach()
                .float()
                .norm(dim=-1)
                .mean()
            ),
            "flow_jepa_phase_horizon_adjacent_cosine": self._adjacent_cosine(
                phase_bank.detach()
            ),
            "flow_jepa_goal_horizon_adjacent_cosine": self._adjacent_cosine(
                goal_bank.detach()
            ),
            "flow_jepa_history_horizon_adjacent_cosine": self._adjacent_cosine(
                history_bank.detach()
            ),
            "flow_jepa_phase_horizon_variation": (
                phase_bank.detach().float().std(dim=1, unbiased=False).mean()
            ),
            "flow_jepa_goal_horizon_variation": (
                goal_bank.detach().float().std(dim=1, unbiased=False).mean()
            ),
            "flow_jepa_history_horizon_variation": (
                history_bank.detach().float().std(dim=1, unbiased=False).mean()
            ),
        }
        for index in range(self.horizon_count):
            metrics[f"flow_jepa_phase_mass_{index}"] = belief[
                :, :, index
            ].mean()
        return phase_bank, goal_bank, history_bank, metrics


def load_precomputed_t5_condition(
    *,
    condition_path: Path,
    max_tokens: int,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """Load one precomputed T5 condition without instantiating a text model.

    The canonical format is a tensor shaped ``[L,D]``.  ``[1,L,D]`` and dict
    wrappers using common embedding/mask names are also accepted.  A missing
    mask means every stored token is valid.
    """

    if max_tokens <= 0:
        raise ValueError("goal language max_tokens must be positive")
    path = Path(condition_path).expanduser().resolve()
    if path.suffix.lower() not in {".pt", ".pth"}:
        raise ValueError(f"T5 condition must be a .pt/.pth file, got {path}")
    if not path.is_file():
        raise FileNotFoundError(f"T5 condition file does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    mask: Any = None
    if isinstance(payload, dict):
        tokens: Any = None
        for key in (
            "tokens",
            "embeddings",
            "embedding",
            "language_embedding",
            "last_hidden_state",
        ):
            if key in payload:
                tokens = payload[key]
                break
        for key in ("mask", "attention_mask", "language_mask"):
            if key in payload:
                mask = payload[key]
                break
        if tokens is None:
            mask_keys = {"mask", "attention_mask", "language_mask"}
            tensor_values = [
                value
                for key, value in payload.items()
                if key not in mask_keys and torch.is_tensor(value)
            ]
            if len(tensor_values) == 1:
                tokens = tensor_values[0]
            else:
                raise ValueError(
                    "T5 condition dict needs tokens/embeddings/embedding/"
                    "language_embedding/last_hidden_state"
                )
    else:
        tokens = payload
    raw_tokens = torch.as_tensor(tokens)
    original_shape = tuple(int(value) for value in raw_tokens.shape)
    original_dtype = str(raw_tokens.dtype).replace("torch.", "")
    tokens = raw_tokens.detach().to(device="cpu", dtype=torch.float32)
    if tokens.ndim == 2:
        tokens = tokens[None]
    if tokens.ndim != 3 or int(tokens.shape[0]) != 1:
        raise ValueError(
            f"T5 condition tokens must be [L,D] or [1,L,D], got {original_shape}"
        )
    if int(tokens.shape[1]) < 1 or int(tokens.shape[2]) < 1:
        raise ValueError("T5 condition must contain at least one finite token and feature")
    tokens = tokens[:, :max_tokens]
    if not bool(torch.isfinite(tokens).all()):
        raise ValueError("T5 condition contains NaN or infinity")
    if mask is None:
        mask_tensor = torch.ones(tokens.shape[:2], dtype=torch.bool)
    else:
        mask_tensor = torch.as_tensor(mask, dtype=torch.bool)
        if mask_tensor.ndim == 1:
            mask_tensor = mask_tensor[None]
        mask_tensor = mask_tensor[:, : tokens.shape[1]]
    if tuple(mask_tensor.shape) != tuple(tokens.shape[:2]):
        raise ValueError("T5 condition mask must align with tokens as [1,L]")
    if not bool(mask_tensor.any()):
        raise ValueError("T5 condition mask must retain at least one valid token")
    metadata = {
        "source": "precomputed_t5_condition",
        "path": str(path),
        "original_shape": list(original_shape),
        "original_dtype": original_dtype,
        "effective_tokens": int(tokens.shape[1]),
    }
    return tokens.contiguous(), mask_tensor.contiguous(), metadata


__all__ = [
    "GoalTokenResampler",
    "StatelessHorizonConditionAdapter",
    "StatelessPhaseAdapter",
    "load_precomputed_t5_condition",
]
