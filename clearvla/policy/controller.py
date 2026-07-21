"""Recurrent multi-token control plane for hierarchical action refinement."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .gauges import fp32_diagnostic
from .primitives import BiasFreeFFN, sinusoidal_positions


class UnifiedControllerConfig(Protocol):
    hidden_size: int
    action_horizon: int
    num_heads: int
    dropout: float
    hierarchical_mmdit_depth: int
    hierarchical_mmdit_operator_stages: int
    hierarchical_mmdit_operator_depth_logit_init: float
    hierarchical_mmdit_execution_contract: str
    hierarchical_mmdit_control_tokens: int
    hierarchical_mmdit_controller_depth: int
    hierarchical_mmdit_controller_heads: int
    hierarchical_mmdit_controller_ffn_expansion: float
    hierarchical_mmdit_spectral_state: int


@dataclass
class ControllerMemory:
    """Controller memory with explicit global and private value ownership.

    ``address`` is allowed to affect Q/K retrieval only.  ``content`` is the
    value stream consumed by workspace and operation readers, so a controller
    cannot manufacture evidence by writing an address signal into V. The
    global/private fields keep the two responsibilities explicit at the
    workspace boundary instead of flattening them into one value bank.
    """

    content: Tensor
    address: Tensor
    global_content: Tensor
    private_content: Tensor
    global_address: Tensor
    private_address: Tensor


@dataclass(frozen=True)
class ControllerExecutionContract:
    """One central policy with typed, non-overlapping execution actuators.

    ``branch_update_keep_logits`` remains a load-compatible field for older
    manifests, but the active decoder never consumes it.  Residual write
    strength belongs to the host block; the controller only selects an
    operation and its operator capacity.
    """

    operation_value_field: Tensor
    branch_capacity_logits: Tensor
    spectral_shift: Tensor
    # Present in the dataclass for old checkpoint/config compatibility.  The
    # current decoder treats it as non-executable legacy telemetry.
    branch_update_keep_logits: Tensor | None
    operation_axis: str
    residual_amplitude_owner: str


@dataclass
class UnifiedControllerOutput:
    state: Tensor
    global_state: Tensor
    private_state: Tensor
    state_address: Tensor
    memory: ControllerMemory
    execution: ControllerExecutionContract
    operation_value_field: Tensor
    operator_update_logits: Tensor
    operator_depth_logits: Tensor
    # Direct per-frequency aperture displacement.  Frequencies are typed
    # output queries, not anonymous controller slots mixed after readout.
    spectral_shift: Tensor
    spectral_ownership: Tensor
    spectral_competition_loss: Tensor
    metrics: dict[str, Tensor]


@dataclass
class EvidenceExecutionOutput:
    """Typed execution controls for the native-time evidence decoder.

    ``capacity_ratio`` selects how much of the nested operator basis remains
    available. It is not an update-amplitude control. Repetition/skip is
    selected by the controller's candidate value field, not a categorical
    route or dwell classifier. The controller has no value path into the
    evidence bank.
    """

    state: Tensor
    capacity_ratio: Tensor
    capacity_ratios: Tensor
    metrics: dict[str, Tensor]


class NativeExecutionValueReader(nn.Module):
    """Read the value of legal execution candidates from typed state.

    Candidate identity is part of the query chart. The local compatibility
    chart represents dwell ``1..K``; the native dynamic path supplies explicit
    block and dwell identities for its global block-by-dwell chart. A
    terminal identity is a real candidate with a separate block identity, so
    candidate advantages are identifiable relative to doing no more work.
    The reader never produces an action update. Its
    inputs remain attached so a soft execution choice can carry task gradients
    back into the upstream execution context.
    Candidate, block, and horizon identities remain separate instead of
    pooling all candidates into one scalar.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        heads: int,
        block_count: int,
        max_dwell: int,
        horizon: int,
        ffn_expansion: float,
    ) -> None:
        super().__init__()
        hidden_size = int(hidden_size)
        heads = int(heads)
        block_count = int(block_count)
        max_dwell = int(max_dwell)
        if hidden_size % heads:
            raise ValueError("value reader hidden_size must be divisible by heads")
        if block_count < 1 or max_dwell < 1:
            raise ValueError("value reader block count and max dwell must be positive")
        self.hidden_size = hidden_size
        self.block_count = block_count
        self.max_dwell = max_dwell
        self.candidate_count = max_dwell
        self.horizon = int(horizon)
        self.candidate_identity = nn.Parameter(
            torch.randn(self.candidate_count, hidden_size) * 0.02
        )
        self.block_identity = nn.Parameter(torch.randn(block_count, hidden_size) * 0.02)
        self.terminal_identity = nn.Parameter(torch.randn(hidden_size) * 0.02)
        self.register_buffer(
            "horizon_position",
            sinusoidal_positions(range(1, self.horizon + 1), hidden_size)[None],
            persistent=False,
        )
        self.query_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.memory_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.action_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.temporal_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.memory_attention = nn.MultiheadAttention(
            hidden_size, heads, dropout=0.0, batch_first=True
        )
        self.action_attention = nn.MultiheadAttention(
            hidden_size, heads, dropout=0.0, batch_first=True
        )
        self.temporal_attention = nn.MultiheadAttention(
            hidden_size, heads, dropout=0.0, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ffn = BiasFreeFFN(hidden_size, float(ffn_expansion))
        self.context_lift = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.value_head = nn.Linear(hidden_size, 2, bias=False)
        nn.init.zeros_(self.value_head.weight)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        # V93/V94-preview readers had an absolute-value bias. Candidate
        # centering made that bias unidentifiable, so the new advantage reader
        # has no such parameter. Consume only that exact legacy key to keep
        # stage/checkpoint loading compatible without reintroducing the gauge.
        state_dict.pop(f"{prefix}value_head.bias", None)
        # Older checkpoints predate the explicit terminal candidate.  Seed its
        # identity from the initialized module so strict loading remains useful
        # without pretending an operation block is the exit action.
        terminal_key = f"{prefix}terminal_identity"
        if terminal_key not in state_dict:
            state_dict[terminal_key] = self.terminal_identity.detach().clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _ensure_horizon_position(self, horizon: int, reference: Tensor) -> Tensor:
        if int(self.horizon_position.shape[1]) != int(horizon):
            position = sinusoidal_positions(range(1, int(horizon) + 1), self.hidden_size)[None]
        else:
            position = self.horizon_position
        return position.to(device=reference.device, dtype=reference.dtype)

    def parameters_for_value_loss(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.parameters())

    def forward(
        self,
        *,
        state: Tensor,
        global_condition: Tensor,
        time_context: Tensor,
        evidence_tokens: Tensor,
        action_tokens: Tensor,
        block_index: int | Tensor,
        evidence_value_tokens: Tensor | None = None,
        evidence_key_bias: Tensor | None = None,
        candidate_block_index: Tensor | None = None,
        candidate_repeat_index: Tensor | None = None,
    ) -> Tensor:
        if state.ndim != 3 or evidence_tokens.ndim != 3 or action_tokens.ndim != 3:
            raise ValueError("value-reader inputs must be token sequences")
        batch, horizon, hidden = action_tokens.shape
        if hidden != self.hidden_size:
            raise ValueError("value-reader action hidden size is invalid")
        if global_condition.shape != (batch, hidden) or time_context.shape != (batch, hidden):
            raise ValueError("value-reader conditions have the wrong shape")
        if evidence_value_tokens is None:
            evidence_value_tokens = evidence_tokens
        if tuple(evidence_value_tokens.shape) != tuple(evidence_tokens.shape):
            raise ValueError("value-reader evidence selector/value tokens are misaligned")
        evidence_length = int(evidence_tokens.shape[1])
        if evidence_key_bias is None:
            evidence_key_bias = torch.zeros(
                evidence_length, device=action_tokens.device, dtype=torch.float32
            )
        elif tuple(evidence_key_bias.shape) != (evidence_length,):
            raise ValueError("value-reader evidence key bias must be [evidence_tokens]")
        else:
            evidence_key_bias = evidence_key_bias.to(device=action_tokens.device)
        # Keep the complete typed execution context attached. The physical
        # candidate target is detached in the trainer; the forward value path
        # must not add a second, unrelated stop-gradient.
        candidate_identity = self.candidate_identity.to(
            device=action_tokens.device, dtype=action_tokens.dtype
        )
        block_identity = self.block_identity.to(
            device=action_tokens.device, dtype=action_tokens.dtype
        )
        if candidate_block_index is None:
            if isinstance(block_index, Tensor):
                if tuple(block_index.shape) != (batch,):
                    raise ValueError("value-reader block index must be an int or [B]")
                block_index = block_index.to(device=action_tokens.device, dtype=torch.long)
                if bool(((block_index < 0) | (block_index >= self.block_count)).any()):
                    raise ValueError("value-reader block index is outside its repertoire")
                block = block_identity.index_select(0, block_index)
                block = block[:, None]
                candidate = candidate_identity[None].expand(batch, -1, -1)
            else:
                if not 0 <= int(block_index) < self.block_count:
                    raise ValueError("value-reader block index is outside its repertoire")
                block = block_identity[int(block_index)][None, None].expand(batch, -1, -1)
                candidate = candidate_identity[None].expand(batch, -1, -1)
            if candidate_repeat_index is not None:
                raise ValueError("candidate repeat identities require candidate block ids")
        else:
            if not isinstance(block_index, Tensor):
                raise ValueError("candidate block ids require a tensor block index")
            if candidate_block_index.ndim != 2 or int(candidate_block_index.shape[0]) != batch:
                raise ValueError("candidate block ids must be [B,C]")
            candidate_block_index = candidate_block_index.to(
                device=action_tokens.device, dtype=torch.long
            )
            if bool(
                ((candidate_block_index < 0) | (candidate_block_index > self.block_count)).any()
            ):
                raise ValueError("candidate block ids are outside the controller repertoire")
            candidate = candidate_identity
            if candidate_repeat_index is None:
                candidate_repeat_index = torch.arange(
                    int(candidate_block_index.shape[1]),
                    device=action_tokens.device,
                    dtype=torch.long,
                )[None].expand(batch, -1) % self.max_dwell
            if tuple(candidate_repeat_index.shape) != tuple(candidate_block_index.shape):
                raise ValueError("candidate repeat ids must match candidate block ids")
            candidate_repeat_index = candidate_repeat_index.to(
                device=action_tokens.device, dtype=torch.long
            )
            if bool(
                ((candidate_repeat_index < 0) | (candidate_repeat_index >= self.max_dwell)).any()
            ):
                raise ValueError("candidate repeat ids are outside the reader repertoire")
            candidate = candidate_repeat_index.unsqueeze(-1).expand(-1, -1, self.hidden_size)
            candidate = torch.gather(
                candidate_identity[None].expand(batch, -1, -1), 1, candidate
            )
            terminal = candidate_block_index == self.block_count
            safe_block_index = candidate_block_index.clamp_max(self.block_count - 1)
            block = torch.gather(
                block_identity[None].expand(batch, -1, -1),
                1,
                safe_block_index.unsqueeze(-1).expand(-1, -1, self.hidden_size),
            )
            block = torch.where(
                terminal.unsqueeze(-1),
                self.terminal_identity.to(device=block.device, dtype=block.dtype)[None, None],
                block,
            )
        candidate_count = int(candidate.shape[1])
        condition = self.context_lift(
            torch.cat([global_condition, time_context], dim=-1)
        )
        query = candidate + block + condition[:, None]
        # The reader gets the controller's private state, while the explicit
        # global/time tokens carry the shared execution context.  Removing the
        # common mode here prevents a rank-1 controller state from becoming a
        # second broadcast condition for every candidate value.
        private_state = state - state.mean(dim=1, keepdim=True)
        memory = torch.cat(
            [
                private_state,
                evidence_value_tokens,
                global_condition[:, None],
                time_context[:, None],
            ],
            dim=1,
        )
        memory_key_bias = torch.zeros(
            int(memory.shape[1]), device=memory.device, dtype=torch.float32
        )
        memory_key_bias[int(private_state.shape[1]) : int(private_state.shape[1]) + evidence_length] = (
            evidence_key_bias.detach().float()
        )
        memory_attention_mask = memory_key_bias.to(dtype=action_tokens.dtype)[None, None, :].expand(
            batch * self.memory_attention.num_heads,
            candidate_count,
            -1,
        )
        memory_context, _ = self.memory_attention(
            self.query_norm(query),
            self.memory_norm(memory),
            self.memory_norm(memory),
            attn_mask=memory_attention_mask,
            need_weights=False,
        )
        query = query + memory_context
        horizon_position = self._ensure_horizon_position(horizon, action_tokens)
        horizon_query = query[:, :, None, :] + horizon_position[:, None]
        horizon_query = horizon_query.expand(-1, -1, horizon, -1)
        action_memory = action_tokens[:, None].expand(-1, candidate_count, -1, -1)
        action_memory = action_memory.reshape(batch * candidate_count, horizon, hidden)
        action_query = horizon_query.reshape(batch * candidate_count, horizon, hidden)
        action_context, _ = self.action_attention(
            self.action_norm(action_query),
            self.action_norm(action_memory),
            self.action_norm(action_memory),
            need_weights=False,
        )
        value_tokens = action_query + action_context
        temporal_context, _ = self.temporal_attention(
            self.temporal_norm(value_tokens),
            self.temporal_norm(value_tokens),
            self.temporal_norm(value_tokens),
            need_weights=False,
        )
        value_tokens = value_tokens + temporal_context
        value_tokens = value_tokens + self.ffn(self.ffn_norm(value_tokens))
        value_field = self.value_head(value_tokens).reshape(
            batch, candidate_count, horizon, 2
        )
        # Candidate value is an advantage chart, not an absolute scalar field.
        # Removing the common candidate mode in the architecture makes the
        # selector's gauge explicit and prevents an unidentifiable broadcast
        # component from consuming nearly all reader energy.
        if candidate_count > 1:
            value_field = value_field - value_field.mean(dim=1, keepdim=True)
        return value_field


class EvidenceExecutionController(nn.Module):
    """A recurrent, competitive controller for the native-time MMDiT path.

    The controller reads evidence and action state, but only emits typed
    execution decisions.  Slot-specific position codes and source ownership
    attention prevent the multi-token state from becoming repeated copies of
    one pooled summary.  The shared recurrent transition gives all operation
    blocks one control plane while each block keeps its own query.
    """

    def __init__(
        self, config: object, *, block_count: int, max_dwell: int | None = None
    ) -> None:
        super().__init__()
        h = int(config.hidden_size)
        heads = int(getattr(config, "latent_cvae_mmdit_controller_heads", config.num_heads))
        if h % heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        token_count = int(getattr(config, "latent_cvae_mmdit_control_tokens", 8))
        depth = int(getattr(config, "latent_cvae_mmdit_controller_depth", 2))
        max_dwell = int(
            getattr(config, "latent_cvae_mmdit_max_dwell", 2)
            if max_dwell is None
            else max_dwell
        )
        if token_count < 2:
            raise ValueError("native evidence controller needs at least two control tokens")
        if depth < 1 or max_dwell < 1:
            raise ValueError("controller depth and max dwell must be positive")
        self.hidden_size = h
        self.token_count = token_count
        self.block_count = int(block_count)
        self.max_dwell = max_dwell
        self.heads = heads
        self.control_tokens = nn.Parameter(torch.randn(token_count, h) * 0.02)
        # The address/function pair is deliberately separate from the value
        # state.  It gives each slot a stable computation identity without
        # turning that identity into evidence content.
        self.slot_address = nn.Parameter(torch.randn(token_count, h) * 0.03)
        self.slot_function = nn.Parameter(torch.randn(token_count, h) * 0.02)
        self.register_buffer(
            "control_positions",
            sinusoidal_positions(range(1, token_count + 1), h)[None],
            persistent=True,
        )
        centered_positions = self.control_positions - self.control_positions.mean(
            dim=1, keepdim=True
        )
        position_rms = centered_positions.float().square().mean(dim=-1, keepdim=True).sqrt()
        self.register_buffer(
            "control_position_unit",
            centered_positions
            / position_rms.clamp_min(1e-6).to(self.control_positions.dtype),
            persistent=True,
        )
        self.block_queries = nn.Parameter(torch.randn(block_count, h) * 0.02)
        self.source_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.state_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.query_proj = nn.Linear(h, h, bias=False)
        self.key_proj = nn.Linear(h, h, bias=False)
        self.value_proj = nn.Linear(h, h, bias=False)
        self.state_update = nn.ModuleList(
            [nn.GRUCell(h, h) for _ in range(depth)]
        )
        self.state_ffn = nn.ModuleList(
            [BiasFreeFFN(h, float(getattr(config, "latent_cvae_mmdit_controller_ffn_expansion", 2.0))) for _ in range(depth)]
        )
        self.state_ffn_norm = nn.ModuleList(
            [nn.LayerNorm(h, elementwise_affine=False) for _ in range(depth)]
        )
        self.operation_query = nn.Linear(h, h, bias=False)
        self.operation_key = nn.Linear(h, h, bias=False)
        self.operation_value = nn.Linear(h, h, bias=False)
        self.operation_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.capacity_head = nn.Linear(2 * h, 1)
        self.value_reader = NativeExecutionValueReader(
            hidden_size=h,
            heads=heads,
            block_count=self.block_count,
            max_dwell=max_dwell,
            horizon=int(config.action_horizon),
            ffn_expansion=float(
                getattr(config, "latent_cvae_mmdit_controller_ffn_expansion", 2.0)
            ),
        )
        # Full capacity is supplied by the decoder's exact progress boundary.
        # Start the learned raw policy from one explicit configured logit and
        # no random condition-dependent offset; otherwise an untrained capacity
        # head invents sample/block differences before receiving task evidence.
        nn.init.zeros_(self.capacity_head.weight)
        nn.init.constant_(
            self.capacity_head.bias,
            float(getattr(config, "latent_cvae_mmdit_operator_depth_logit_init", 4.0)),
        )

    def predict_execution_value(
        self,
        *,
        state: Tensor,
        global_condition: Tensor,
        time_context: Tensor,
        evidence_tokens: Tensor,
        action_tokens: Tensor,
        block_index: int | Tensor,
        evidence_value_tokens: Tensor | None = None,
        evidence_key_bias: Tensor | None = None,
        candidate_block_index: Tensor | None = None,
        candidate_repeat_index: Tensor | None = None,
    ) -> Tensor:
        return self.value_reader(
            state=state,
            global_condition=global_condition,
            time_context=time_context,
            evidence_tokens=evidence_tokens,
            evidence_value_tokens=evidence_value_tokens,
            evidence_key_bias=evidence_key_bias,
            action_tokens=action_tokens,
            block_index=block_index,
            candidate_block_index=candidate_block_index,
            candidate_repeat_index=candidate_repeat_index,
        )

    def initial_state(self, reference: Tensor) -> Tensor:
        state = self.control_tokens[None].to(
            device=reference.device, dtype=reference.dtype
        ) + self.control_position_unit.to(device=reference.device, dtype=reference.dtype)
        return self._center_slots(state).expand(int(reference.shape[0]), -1, -1)

    @staticmethod
    def _center_slots(state: Tensor) -> Tensor:
        """Keep shared execution context outside the private slot state.

        Global/time context already has explicit source lanes.  Allowing a
        second broadcast component to survive in every recurrent slot makes
        the ownership softmax indifferent to slot identity and turns the
        multi-token controller into one repeated summary.
        """
        return state - state.mean(dim=1, keepdim=True)

    def _source_lanes(
        self,
        *,
        global_condition: Tensor,
        time_context: Tensor,
        evidence_tokens: Tensor,
        action_tokens: Tensor,
        feedback: Tensor,
        evidence_value_tokens: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Build native selector and value lanes with one auditable boundary."""
        if evidence_value_tokens is None:
            evidence_value_tokens = evidence_tokens
        if tuple(evidence_value_tokens.shape) != tuple(evidence_tokens.shape):
            raise ValueError("native evidence selector/value tokens are misaligned")
        selector_sources = torch.cat(
            [
                global_condition[:, None],
                time_context[:, None],
                evidence_tokens,
                action_tokens,
                feedback,
            ],
            dim=1,
        )
        value_sources = torch.cat(
            [
                global_condition[:, None],
                time_context[:, None],
                evidence_value_tokens,
                torch.zeros_like(action_tokens),
                torch.zeros_like(feedback),
            ],
            dim=1,
        )
        return (
            self.key_proj(self.source_norm(selector_sources)),
            self.value_proj(self.source_norm(value_sources)),
        )

    def _split_heads(self, value: Tensor) -> Tensor:
        batch, tokens, hidden = value.shape
        return value.reshape(batch, tokens, self.heads, hidden // self.heads).transpose(1, 2)

    def _attention(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        query = self._split_heads(q)
        key = self._split_heads(k)
        value = self._split_heads(v)
        scores = torch.matmul(query.float(), key.float().transpose(-2, -1))
        scores = scores * (float(query.shape[-1]) ** -0.5)
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights.to(dtype=value.dtype), value)
        attended = attended.transpose(1, 2).reshape(int(q.shape[0]), int(q.shape[1]), -1)
        return attended, weights

    def _competitive_attention(
        self, q: Tensor, k: Tensor, v: Tensor, key_bias: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Dispatch sources to slots and measure ownership separately.

        A plain per-slot softmax lets every slot copy the same source mixture.
        The ownership softmax is over slots, so a source has to be assigned
        across the available control slots before each slot reads it.  This is
        a selector-plane operation; it never changes evidence values.
        """
        query = self._split_heads(q)
        key = self._split_heads(k)
        value = self._split_heads(v)
        logits = torch.matmul(query.float(), key.float().transpose(-2, -1))
        logits = logits * (float(query.shape[-1]) ** -0.5)
        if key_bias is not None:
            if tuple(key_bias.shape) != (int(k.shape[1]),):
                raise ValueError("native controller source key bias has the wrong shape")
            logits = logits + key_bias.to(
                device=logits.device, dtype=logits.dtype
            )[None, None, None, :]
        dispatch = torch.softmax(logits, dim=-1)
        ownership = torch.softmax(logits, dim=-2)
        weights = dispatch * ownership
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        attended = torch.matmul(weights.to(dtype=value.dtype), value)
        attended = attended.transpose(1, 2).reshape(int(q.shape[0]), int(q.shape[1]), -1)
        return attended, weights, ownership

    @staticmethod
    def _pair_cosine(state: Tensor) -> Tensor:
        normalized = F.normalize(state.float(), dim=-1)
        gram = torch.matmul(normalized, normalized.transpose(-1, -2))
        count = int(state.shape[1])
        if count <= 1:
            return gram.new_zeros(())
        mask = ~torch.eye(count, device=state.device, dtype=torch.bool)
        return gram[:, mask].mean()

    def forward(
        self,
        *,
        state: Tensor | None,
        global_condition: Tensor,
        time_context: Tensor,
        action_tokens: Tensor,
        evidence_tokens: Tensor,
        block_index: int | Tensor,
        feedback: Tensor | None = None,
        evidence_value_tokens: Tensor | None = None,
        evidence_key_bias: Tensor | None = None,
    ) -> EvidenceExecutionOutput:
        batch = int(global_condition.shape[0])
        if state is None:
            state = self.initial_state(global_condition)
        if tuple(state.shape) != (batch, self.token_count, self.hidden_size):
            raise ValueError("controller state has the wrong shape")
        if feedback is None:
            feedback = torch.zeros_like(action_tokens)
        if evidence_value_tokens is None:
            evidence_value_tokens = evidence_tokens
        if tuple(evidence_value_tokens.shape) != tuple(evidence_tokens.shape):
            raise ValueError("native evidence selector/value tokens are misaligned")
        evidence_length = int(evidence_tokens.shape[1])
        if evidence_key_bias is None:
            evidence_key_bias = torch.zeros(
                evidence_length, device=state.device, dtype=torch.float32
            )
        elif tuple(evidence_key_bias.shape) != (evidence_length,):
            raise ValueError("native controller evidence key bias must be [evidence_tokens]")
        else:
            evidence_key_bias = evidence_key_bias.to(device=state.device)
        keys, values = self._source_lanes(
            global_condition=global_condition,
            time_context=time_context,
            evidence_tokens=evidence_tokens,
            evidence_value_tokens=evidence_value_tokens,
            action_tokens=action_tokens,
            feedback=feedback,
        )
        source_key_bias = torch.cat(
            [
                torch.zeros(2, device=keys.device, dtype=torch.float32),
                evidence_key_bias.detach().float(),
                torch.zeros(
                    int(action_tokens.shape[1] + feedback.shape[1]),
                    device=keys.device,
                    dtype=torch.float32,
                ),
            ]
        )
        slot_address = self.slot_address[None].to(device=state.device, dtype=state.dtype)
        slot_function = self.slot_function[None].to(device=state.device, dtype=state.dtype)
        ownership_rows: list[Tensor] = []
        for update, norm, ffn in zip(self.state_update, self.state_ffn_norm, self.state_ffn):
            queries = self.query_proj(self.state_norm(state) + slot_address)
            attended, weights, ownership = self._competitive_attention(
                queries, keys, values, source_key_bias
            )
            ownership_rows.append(ownership.detach().float())
            attended = attended + 0.20 * slot_function
            state = update(
                attended.reshape(batch * self.token_count, self.hidden_size),
                state.reshape(batch * self.token_count, self.hidden_size),
            ).reshape(batch, self.token_count, self.hidden_size)
            # Keep a small address-only private component after every shared
            # recurrent transition. Without this, the common global input can
            # erase the slot coordinates and turn all control tokens into one
            # repeated summary before the operation reader sees them.
            state = state + 0.1 * ffn(norm(state))
            state = state + 0.10 * self.control_position_unit.to(
                device=state.device, dtype=state.dtype
            ) + 0.10 * slot_function
            state = self._center_slots(state)

        block_queries = self.block_queries[None].expand(batch, -1, -1)
        control_condition = global_condition
        operation_query = self.operation_query(
            self.operation_norm(block_queries) + control_condition[:, None]
        )
        private_state = state - state.mean(dim=1, keepdim=True)
        operation_keys = self.operation_key(self.state_norm(private_state))
        operation_values = self.operation_value(self.state_norm(private_state))
        operation_context, operation_weights = self._attention(
            operation_query, operation_keys, operation_values
        )
        control_context = torch.cat(
            [
                operation_context,
                control_condition[:, None].expand(-1, self.block_count, -1),
            ],
            dim=-1,
        )
        capacity_ratios = torch.sigmoid(self.capacity_head(control_context)).squeeze(-1)
        if isinstance(block_index, Tensor):
            if tuple(block_index.shape) != (batch,):
                raise ValueError("controller block_index must be an int or [B]")
            block_index = block_index.to(device=state.device, dtype=torch.long)
            if bool(((block_index < 0) | (block_index >= self.block_count)).any()):
                raise ValueError("block_index is outside the native controller repertoire")
            capacity_ratio = capacity_ratios.gather(1, block_index[:, None]).squeeze(1)
        else:
            if not 0 <= int(block_index) < self.block_count:
                raise ValueError("block_index is outside the native controller repertoire")
            capacity_ratio = capacity_ratios[:, int(block_index)]
        attention_entropy = -(
            weights.float().clamp_min(1e-8) * weights.float().clamp_min(1e-8).log()
        ).sum(dim=-1).mean()
        attention_entropy = attention_entropy / max(math.log(float(keys.shape[1])), 1e-6)
        common = state.mean(dim=1, keepdim=True)
        total_energy = state.float().square().mean().clamp_min(1e-8)
        common_energy = common.float().square().mean()
        private_energy = private_state.float().square().mean()
        metrics = {
            "controller_capacity_ratio": capacity_ratio.detach().float().mean(),
            "controller_execution_candidate_count": torch.as_tensor(
                float(self.max_dwell), device=state.device, dtype=torch.float32
            ),
            "controller_source_attention_entropy": attention_entropy.detach(),
            "controller_operation_attention_entropy": (
                -operation_weights.float().clamp_min(1e-8)
                * operation_weights.float().clamp_min(1e-8).log()
            ).sum(dim=-1).mean(),
            "controller_capacity_block_spread": capacity_ratios.detach()
            .float()
            .std(dim=-1, unbiased=False)
            .mean(),
            "controller_slot_pair_cosine": self._pair_cosine(state).detach(),
            "controller_slot_common_mode_ratio": (common_energy / total_energy).detach(),
            "controller_slot_private_energy_ratio": (
                private_energy / total_energy
            ).detach(),
            "controller_state_norm": state.detach().float().norm(dim=-1).mean(),
        }
        if ownership_rows:
            ownership = ownership_rows[-1]
            ownership_load = ownership.mean(dim=(1, 2))
            ownership_profile = ownership.float().mean(dim=1)
            ownership_profile = ownership_profile / ownership_profile.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
            profile_normalized = F.normalize(ownership_profile, dim=-1)
            profile_gram = torch.matmul(
                profile_normalized, profile_normalized.transpose(-1, -2)
            )
            profile_count = int(profile_gram.shape[-1])
            profile_mask = ~torch.eye(
                profile_count, device=profile_gram.device, dtype=torch.bool
            )
            metrics["controller_slot_ownership_diversity"] = (
                ownership_load - ownership_load.mean(dim=-1, keepdim=True)
            ).norm(dim=-1).mean()
            metrics["controller_slot_ownership_profile_diversity"] = (
                1.0 - profile_gram[:, profile_mask]
            ).mean()
            metrics["controller_slot_ownership_max"] = ownership_load.amax(dim=-1).mean()
        return EvidenceExecutionOutput(
            state=state,
            capacity_ratio=capacity_ratio,
            capacity_ratios=capacity_ratios,
            metrics=metrics,
        )


class _RecurrentControllerBlock(nn.Module):
    """Competitive evidence dispatch followed by independent slot updates.

    Sources compete across slots before aggregation.  A shared GRU then
    updates every slot independently; there is deliberately no slot-to-slot
    self-attention here.  Mixing slots at this boundary lets a common summary
    overwrite every private state and makes one-summary control memory a cheap
    fixed point.
    """

    def __init__(self, config: UnifiedControllerConfig) -> None:
        super().__init__()
        h = int(config.hidden_size)
        heads = int(config.hierarchical_mmdit_controller_heads)
        # The zero-output controller boundary must also preserve the host RNG
        # stream. Internal dropout would perturb later workspace/MMDiT masks
        # even while every controller actuator is neutral.
        dropout = 0.0
        self.heads = heads
        self.cross_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.source_key_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.source_value_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.cross_attn = nn.MultiheadAttention(h, heads, dropout=dropout, batch_first=True)
        self.slot_update = nn.GRUCell(h, h)
        # The heavy transition is shared, while a non-evidence function code
        # selects a multiplicative computation chart. Address remains Q/K-only
        # and therefore cannot become workspace value content.
        self.function_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.function_scale = nn.Linear(h, 2 * h, bias=False)
        self.function_strength = 0.25
        self.global_to_private = nn.Linear(h, h, bias=False)
        nn.init.eye_(self.global_to_private.weight)
        self.global_explained_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.global_explained = nn.Linear(h, h, bias=False)
        # The residual evidence path starts neutral. It can learn what the
        # global lane explains without changing the initial source geometry.
        nn.init.zeros_(self.global_explained.weight)
        self.private_content_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.private_content_adapter = nn.Linear(2 * h, h, bias=False)
        # Content specialization is a residual path, not an amplitude gate.
        # The zero initialization preserves the existing controller boundary
        # until downstream readers provide a useful private-content gradient.
        nn.init.zeros_(self.private_content_adapter.weight)
        self.ffn_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.ffn = BiasFreeFFN(h, float(config.hierarchical_mmdit_controller_ffn_expansion))
        self.drop = nn.Dropout(dropout)

    def _competitive_cross_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        source_key_bias: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Assign each source across slots before aggregating slot content."""
        projection = self.cross_attn.in_proj_weight
        if projection is None:
            raise RuntimeError("controller cross-attention requires packed QKV weights")
        hidden = int(query.shape[-1])
        bias = self.cross_attn.in_proj_bias
        query_bias = None if bias is None else bias[:hidden]
        key_bias = None if bias is None else bias[hidden : 2 * hidden]
        value_bias = None if bias is None else bias[2 * hidden :]
        projected_query = F.linear(query, projection[:hidden], query_bias)
        projected_key = F.linear(key, projection[hidden : 2 * hidden], key_bias)
        projected_value = F.linear(value, projection[2 * hidden :], value_bias)

        def split_heads(x: Tensor) -> Tensor:
            batch, tokens, width = x.shape
            return x.reshape(batch, tokens, self.heads, width // self.heads).transpose(1, 2)

        query_heads = split_heads(projected_query)
        key_heads = split_heads(projected_key)
        value_heads = split_heads(projected_value)
        logits = torch.matmul(query_heads.float(), key_heads.float().transpose(-2, -1)) * (
            float(query_heads.shape[-1]) ** -0.5
        )

        # Dispatch and ownership are deliberately distinct, following the
        # continuous dispatch/combine split used by soft expert systems.
        # Dispatch answers what each slot reads; ownership answers which slot
        # a source prefers.  Forcing both axes to equal mass made every slot
        # read the same mixture and created the observed one-summary fixed
        # point.
        ownership = torch.softmax(logits, dim=-2)
        prior_logits = source_key_bias.detach().float()[None, None, None, :]
        dispatch = torch.softmax(logits + prior_logits, dim=-1)
        # Ownership affects the read without imposing an equal-load target.
        # The normalization retains a valid source distribution per slot.
        weights = dispatch * ownership.clamp_min(1e-8)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        attended = torch.matmul(weights.to(dtype=value_heads.dtype), value_heads)
        attended = attended.transpose(1, 2).reshape(
            int(query.shape[0]), int(query.shape[1]), hidden
        )
        output = F.linear(
            attended,
            self.cross_attn.out_proj.weight,
            self.cross_attn.out_proj.bias,
        )
        return output, weights, ownership

    def forward(
        self,
        global_state: Tensor,
        private_state: Tensor,
        global_address: Tensor,
        private_address: Tensor,
        global_function: Tensor,
        private_function: Tensor,
        source_key: Tensor,
        source_value: Tensor,
        source_key_bias: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        key = self.source_key_norm(source_key)
        value = self.source_value_norm(source_value)
        if tuple(source_key_bias.shape) != (int(source_key.shape[1]),):
            raise ValueError("controller source key bias has the wrong shape")
        global_query = self.cross_norm(global_state) + global_address
        global_cross, global_weights, _ = self._competitive_cross_attention(
            global_query,
            key,
            value,
            source_key_bias.to(device=global_query.device),
        )
        global_input_scale, global_update_scale = self.function_scale(
            self.function_norm(global_function)
        ).chunk(2, dim=-1)
        global_cross = global_cross * (
            1.0 + self.function_strength * torch.tanh(global_input_scale)
        )
        global_batch, _, hidden = global_state.shape
        global_state = self.slot_update(
            global_cross.reshape(global_batch, hidden),
            global_state.reshape(global_batch, hidden),
        ).reshape(global_batch, 1, hidden)
        global_update = self.ffn(self.ffn_norm(global_state))
        global_update = global_update * (
            1.0 + self.function_strength * torch.tanh(global_update_scale)
        )
        global_state = global_state + self.drop(global_update)

        # Private slots see the global explanation as context, but keep a
        # centered value lane. Common content therefore belongs to the global
        # memory and cannot be copied into every private slot by the shared
        # recurrent transition.
        private_query = (
            self.cross_norm(private_state)
            + private_address
            + 0.25 * self.global_to_private(global_state)
        )
        explained = self.global_explained(self.global_explained_norm(global_state)).expand(
            -1, int(value.shape[1]), -1
        )
        private_value = value - explained
        private_cross, private_weights, private_ownership = self._competitive_cross_attention(
            private_query,
            key,
            private_value,
            source_key_bias.to(device=private_query.device),
        )
        private_cross = private_cross + self.private_content_adapter(
            torch.cat(
                [
                    self.private_content_norm(private_cross),
                    self.function_norm(private_function),
                ],
                dim=-1,
            )
        )
        private_cross = private_cross - private_cross.mean(dim=1, keepdim=True)

        batch, slots, hidden = private_state.shape
        input_scale, update_scale = self.function_scale(self.function_norm(private_function)).chunk(
            2, dim=-1
        )
        private_cross = private_cross * (1.0 + self.function_strength * torch.tanh(input_scale))
        private_state = self.slot_update(
            private_cross.reshape(batch * slots, hidden),
            private_state.reshape(batch * slots, hidden),
        ).reshape(batch, slots, hidden)
        private_update = self.ffn(self.ffn_norm(private_state))
        private_update = private_update * (1.0 + self.function_strength * torch.tanh(update_scale))
        private_state = private_state + self.drop(private_update)
        private_state = private_state - private_state.mean(dim=1, keepdim=True)
        return (
            global_state,
            private_state,
            global_weights,
            private_weights,
            private_ownership,
            private_value.detach().float().square().mean().sqrt(),
        )


class UnifiedHierarchicalController(nn.Module):
    """One read-only control state for retrieval and candidate operations.

    The recurrent slots have no fixed semantic assignment.  Workspace
    retrieval consumes the complete slot set through its own token interface;
    typed output queries are reserved for operator and compute decisions.
    Changing the number of recurrent slots therefore does not alter any
    downstream semantic assignment. Evidence content is detached at this
    boundary: the controller may decide where to read and which operator to
    use, but cannot rewrite evidence values or take gradient ownership of
    their encoders.
    """

    SOURCE_NAMES = (
        "intent",
        "flow_time",
        "refine_time",
        "action",
        "evidence",
        "stage_role",
        "stage_content",
        "feedback",
    )

    def __init__(
        self,
        config: UnifiedControllerConfig,
        *,
        operator_branch_count: int,
    ) -> None:
        super().__init__()
        h = int(config.hidden_size)
        heads = int(config.hierarchical_mmdit_controller_heads)
        if h % heads:
            raise ValueError("controller hidden_size must be divisible by controller_heads")
        self.hidden_size = h
        self.action_horizon = int(config.action_horizon)
        self.spectral_state = bool(int(getattr(config, "hierarchical_mmdit_spectral_state", 0)))
        self.control_count = int(config.hierarchical_mmdit_control_tokens)
        self.refine_block_count = int(config.hierarchical_mmdit_depth)
        self.execution_contract = str(config.hierarchical_mmdit_execution_contract)
        self.block_owned_execution = self.execution_contract == "typed_block_budget"
        self.operator_count = (
            self.refine_block_count
            if self.block_owned_execution
            else int(config.hierarchical_mmdit_operator_stages)
        )
        self.operator_branch_count = int(operator_branch_count)
        if self.operator_branch_count < 1:
            raise ValueError("controller operator_branch_count must be positive")
        self.control_seed = nn.Parameter(torch.randn(1, self.control_count, h) * 0.02)
        self.source_type = nn.Parameter(torch.randn(1, len(self.SOURCE_NAMES), h) * 0.02)
        self.source_norms = nn.ModuleList(
            [nn.LayerNorm(h, elementwise_affine=False) for _ in self.SOURCE_NAMES]
        )
        self.feedback_lift = nn.Linear(1, h, bias=False)
        self.source_adapters = nn.ModuleList([nn.Linear(h, h) for _ in self.SOURCE_NAMES])
        for adapter in self.source_adapters:
            nn.init.eye_(adapter.weight)
            nn.init.zeros_(adapter.bias)
        self.control_address = nn.Parameter(torch.randn(1, self.control_count, h) * 0.02)
        self.control_function = nn.Parameter(torch.randn(1, self.control_count, h) * 0.02)
        self.register_buffer(
            "control_position",
            sinusoidal_positions(range(self.control_count), h)[None],
            persistent=False,
        )
        self.blocks = nn.ModuleList(
            [
                _RecurrentControllerBlock(config)
                for _ in range(int(config.hierarchical_mmdit_controller_depth))
            ]
        )
        self.final_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.control_address_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.control_address_scale = 0.25
        self.frequency_address = nn.Parameter(torch.randn(1, int(config.action_horizon), h) * 0.02)
        self.register_buffer(
            "frequency_position",
            sinusoidal_positions(range(self.action_horizon), h)[None],
            persistent=False,
        )
        self.frequency_state_proj = nn.Linear(h, h, bias=False)
        self.frequency_shift_head = nn.Linear(h, 2)
        nn.init.eye_(self.frequency_state_proj.weight)
        nn.init.zeros_(self.frequency_shift_head.weight)
        nn.init.zeros_(self.frequency_shift_head.bias)

        self.output_query = nn.Parameter(torch.randn(1, self.operator_count, h) * 0.02)
        self.register_buffer(
            "operator_position",
            sinusoidal_positions(range(self.operator_count), h)[None],
            persistent=False,
        )
        self.output_query_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.output_key_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.output_state_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.output_attn = nn.MultiheadAttention(h, heads, dropout=0.0, batch_first=True)
        # Operation stages compose globally because their dwell/depth choices
        # are coupled. Spectral queries instead use a local frequency operator.
        # Both read the same K/V bank; only their post-read computation differs.
        self.output_coupling_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.output_coupling_attn = nn.MultiheadAttention(h, heads, dropout=0.0, batch_first=True)
        self.output_ffn_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.output_ffn = BiasFreeFFN(h, float(config.hierarchical_mmdit_controller_ffn_expansion))
        self.frequency_function_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.frequency_local = nn.Conv1d(h, h, kernel_size=3, padding=1, groups=h, bias=False)
        with torch.no_grad():
            self.frequency_local.weight.zero_()
            self.frequency_local.weight[:, 0, 0].fill_(0.25)
            self.frequency_local.weight[:, 0, 1].fill_(0.50)
            self.frequency_local.weight[:, 0, 2].fill_(0.25)
        self.frequency_ffn_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.frequency_ffn = BiasFreeFFN(
            h, float(config.hierarchical_mmdit_controller_ffn_expansion)
        )
        self.operator_query_scale = nn.Linear(h, h, bias=False)
        self.frequency_query_scale = nn.Linear(h, h, bias=False)
        self.reader_function_strength = 0.25
        # Keep the legacy doubled head width for state-dict compatibility.
        # Only the depth half is live; the former post-gate update half is
        # retained as a neutral compatibility tensor, while each full MMDiT
        # block keeps sole ownership of its host LayerScale gates.
        operator_output_count = self.operator_branch_count * (
            1 if self.block_owned_execution else 2
        )
        self.operator_head = nn.Linear(h, operator_output_count)
        nn.init.zeros_(self.operator_head.weight)
        nn.init.zeros_(self.operator_head.bias)
        with torch.no_grad():
            self.operator_head.bias.fill_(
                float(config.hierarchical_mmdit_operator_depth_logit_init)
            )

        # The dwell reader estimates a horizon-resolved physical residual value
        # for every semantic operation. External evidence/action sources are
        # detached below, but the internal controller representation remains
        # differentiable so this auxiliary objective can improve the controller
        # state that performs the read. Candidate identity, relation to the
        # current block, and horizon position remain trainable here.
        if self.block_owned_execution:
            stage_owner = torch.arange(self.refine_block_count)
        else:
            stage_owner = torch.div(
                (torch.arange(self.operator_count) + 1) * self.refine_block_count - 1,
                self.operator_count,
                rounding_mode="floor",
            ).clamp_max(self.refine_block_count - 1)
        self.register_buffer("operation_stage_owner", stage_owner, persistent=False)
        self.operation_value_stage = nn.Parameter(torch.randn(1, self.operator_count, h) * 0.02)
        self.operation_value_relation = nn.Parameter(torch.randn(1, 3, h) * 0.02)
        self.operation_value_horizon = nn.Parameter(torch.randn(1, self.action_horizon, h) * 0.02)
        self.register_buffer(
            "operation_value_horizon_position",
            sinusoidal_positions(range(self.action_horizon), h)[None],
            persistent=False,
        )
        self.operation_value_input_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.operation_value_memory_query_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.operation_value_memory_key_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.operation_value_memory_value_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.operation_value_memory_attention = nn.MultiheadAttention(
            h, heads, dropout=0.0, batch_first=True
        )
        self.operation_value_action_query_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.operation_value_action_key_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.operation_value_action_value_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.operation_value_action_attention = nn.MultiheadAttention(
            h, heads, dropout=0.0, batch_first=True
        )
        self.operation_value_aperture_lift = nn.Linear(4, h, bias=False)
        self.operation_value_temporal_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.operation_value_temporal = nn.MultiheadAttention(
            h, heads, dropout=0.0, batch_first=True
        )
        self.operation_value_ffn_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.operation_value_ffn = BiasFreeFFN(
            h, float(config.hierarchical_mmdit_controller_ffn_expansion)
        )
        self.operation_value_head = nn.Linear(h, 2)
        nn.init.zeros_(self.operation_value_head.weight)
        nn.init.zeros_(self.operation_value_head.bias)

    def parameter_groups(self) -> dict[str, tuple[nn.Parameter, ...]]:
        value_reader = (
            self.operation_value_stage,
            self.operation_value_relation,
            self.operation_value_horizon,
            *tuple(self.operation_value_memory_attention.parameters()),
            *tuple(self.operation_value_action_attention.parameters()),
            *tuple(self.operation_value_aperture_lift.parameters()),
            *tuple(self.operation_value_temporal.parameters()),
            *tuple(self.operation_value_ffn.parameters()),
            *tuple(self.operation_value_head.parameters()),
        )
        heads = {
            "operator_controls": tuple(self.operator_head.parameters()),
            "value_reader": value_reader,
        }
        head_ids = {id(parameter) for values in heads.values() for parameter in values}
        heads["backbone"] = tuple(
            parameter for parameter in self.parameters() if id(parameter) not in head_ids
        )
        return heads

    def _typed_source(self, value: Tensor, source_index: int) -> tuple[Tensor, Tensor]:
        if value.ndim != 3 or int(value.shape[-1]) != self.hidden_size:
            raise ValueError(
                f"controller source must be [B,N,{self.hidden_size}], got {tuple(value.shape)}"
            )
        content = self.source_adapters[source_index](
            self.source_norms[source_index](value.detach())
        )
        address = self.source_type[:, source_index : source_index + 1].to(
            device=value.device, dtype=value.dtype
        )
        # Action and feedback are selector inputs only.  They may change the
        # Q/K retrieval geometry, but their values must not enter the memory
        # lane that is later handed to workspace V.  Keeping this distinction
        # here prevents the workspace boundary from being bypassed by a
        # controller state assembled one level above it.
        value_content = (
            # Action/feedback and stage role/content are selector-plane
            # signals. Stage semantics remain available through their typed
            # keys and the workspace stage lane, but cannot become anonymous
            # controller value memory that is later handed to low readers.
            torch.zeros_like(content)
            if source_index in (3, 5, 6, 7)
            else content
        )
        return value_content, content + address

    def _feedback_tokens(self, feedback: Tensor) -> Tensor:
        if feedback.ndim != 2:
            raise ValueError("controller feedback must be [B,F]")
        count = int(feedback.shape[1])
        lifted = self.feedback_lift(feedback.detach().float()[..., None])
        half = self.hidden_size // 2
        position_index = torch.arange(1, count + 1, device=feedback.device, dtype=torch.float32)[
            :, None
        ]
        frequency = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=feedback.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        position = torch.cat(
            [
                torch.sin(position_index * frequency),
                torch.cos(position_index * frequency),
            ],
            dim=-1,
        )
        if int(position.shape[-1]) < self.hidden_size:
            position = F.pad(position, (0, self.hidden_size - int(position.shape[-1])))
        position = position[None, :, : self.hidden_size].to(dtype=lifted.dtype)
        return lifted + position

    @staticmethod
    def _state_metrics(state: Tensor) -> dict[str, Tensor]:
        with fp32_diagnostic(state) as state_fp32:
            normalized = F.normalize(state_fp32, dim=-1)
            gram = torch.matmul(normalized, normalized.transpose(-2, -1))
            count = int(state.shape[1])
            if count > 1:
                off_diagonal = (
                    gram.sum(dim=(-2, -1)) - gram.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
                ) / float(count * (count - 1))
            else:
                off_diagonal = torch.ones(
                    int(state.shape[0]), device=state.device, dtype=torch.float32
                )
            eig = torch.linalg.eigvalsh(gram).clamp_min(0.0)
            direction_participation = eig.sum(dim=-1).square() / eig.square().sum(dim=-1).clamp_min(
                1e-8
            )
            centered = state_fp32 - state_fp32.mean(dim=1, keepdim=True)
            return {
                "controller_state_norm": state_fp32.norm(dim=-1).mean(),
                "controller_state_slot_diversity": centered.norm(dim=-1).mean(),
                "controller_state_pair_cosine": off_diagonal.mean(),
                # Participation of normalized slot directions is a geometric
                # diagnostic, not matrix rank and not evidence of distinct
                # functional semantics by itself.
                "controller_state_direction_participation": (direction_participation.mean()),
                "controller_state_centered_energy_ratio": (
                    centered.square().sum(dim=(-2, -1))
                    / state_fp32.square().sum(dim=(-2, -1)).clamp_min(1e-8)
                ).mean(),
            }

    def _operation_value_field(
        self,
        operator: Tensor,
        *,
        current_block: Tensor | None,
        memory_content: Tensor,
        memory_address: Tensor,
        action_tokens: Tensor,
        committed_spectral_aperture: Tensor | None,
        proposed_spectral_shift: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Predict candidate residual values from detached high-bandwidth state."""

        batch = int(operator.shape[0])
        if tuple(operator.shape) != (batch, self.operator_count, self.hidden_size):
            raise ValueError("operation value reader received an invalid stage tensor")
        if current_block is None:
            current_block = torch.zeros(batch, device=operator.device, dtype=torch.long)
        if tuple(current_block.shape) != (batch,):
            raise ValueError("controller current_block must be [B]")
        memory_shape = (batch, int(memory_content.shape[1]), self.hidden_size)
        if tuple(memory_content.shape) != memory_shape:
            raise ValueError("operation value memory content must be [B,M,H]")
        if tuple(memory_address.shape) != memory_shape:
            raise ValueError("operation value memory address must match memory content")
        action_shape = (batch, self.action_horizon, self.hidden_size)
        if tuple(action_tokens.shape) != action_shape:
            raise ValueError(
                f"operation value action tokens must be {action_shape}, "
                f"got {tuple(action_tokens.shape)}"
            )
        spectral_shape = (batch, self.action_horizon, 2)
        if tuple(proposed_spectral_shift.shape) != spectral_shape:
            raise ValueError("operation value spectral proposal must be [B,horizon,2]")
        if (
            committed_spectral_aperture is not None
            and tuple(committed_spectral_aperture.shape) != spectral_shape
        ):
            raise ValueError("committed spectral aperture must be [B,horizon,2]")
        current_block = current_block.to(device=operator.device, dtype=torch.long)
        if current_block.device.type == "cpu" and (
            bool((current_block < 0).any())
            or bool((current_block >= self.refine_block_count).any())
        ):
            raise ValueError("controller current_block is outside the block repertoire")

        owner = self.operation_stage_owner.to(device=operator.device)[None]
        relation_delta = owner - current_block[:, None]
        relation_index = torch.where(
            relation_delta == 0,
            torch.zeros_like(relation_delta),
            torch.where(
                relation_delta == 1,
                torch.ones_like(relation_delta),
                torch.full_like(relation_delta, 2),
            ),
        )
        relation = (
            self.operation_value_relation.to(device=operator.device, dtype=operator.dtype)
            .expand(batch, -1, -1)
            .gather(
                1, relation_index[..., None].expand(batch, self.operator_count, self.hidden_size)
            )
        )
        # ``operator`` is already the controller-owned post-read state. Keep
        # this path differentiable: detaching it would make value supervision
        # train only the terminal reader head and leave the shared controller
        # representation frozen. The external memory/action lanes below stay
        # detached, so this does not reopen a shortcut into the action path.
        stage = (
            self.operation_value_input_norm(operator)
            + self.operation_value_stage.to(device=operator.device, dtype=operator.dtype)
            + relation
        )
        horizon = self.operation_value_horizon.to(
            device=operator.device, dtype=operator.dtype
        ) + 0.25 * self.operation_value_horizon_position.to(
            device=operator.device, dtype=operator.dtype
        )
        tokens = stage[:, :, None, :] + horizon[:, None, :, :]
        tokens = tokens.reshape(batch * self.operator_count, self.action_horizon, self.hidden_size)

        # Candidate identity is allowed to change the read query, never the
        # controller/workspace value stream.  Both context lanes are detached,
        # making the value objective unable to rewrite the candidates it audits.
        memory_key = (memory_content.detach() + memory_address.detach()).to(dtype=operator.dtype)
        memory_value = memory_content.detach().to(dtype=operator.dtype)
        memory_key = (
            memory_key[:, None]
            .expand(-1, self.operator_count, -1, -1)
            .reshape(
                batch * self.operator_count,
                int(memory_content.shape[1]),
                self.hidden_size,
            )
        )
        memory_value = (
            memory_value[:, None].expand(-1, self.operator_count, -1, -1).reshape_as(memory_key)
        )
        memory_context, _ = self.operation_value_memory_attention(
            self.operation_value_memory_query_norm(tokens),
            self.operation_value_memory_key_norm(memory_key),
            self.operation_value_memory_value_norm(memory_value),
            need_weights=False,
        )
        tokens = tokens + memory_context

        committed = committed_spectral_aperture
        if committed is None:
            committed = torch.zeros(
                spectral_shape,
                device=operator.device,
                dtype=operator.dtype,
            )
        spectral_context = torch.cat(
            [
                committed.detach().to(device=operator.device, dtype=operator.dtype),
                torch.tanh(
                    proposed_spectral_shift.detach().to(
                        device=operator.device, dtype=operator.dtype
                    )
                ),
            ],
            dim=-1,
        )
        action_context = (
            action_tokens.detach()
            + 0.25 * self.frequency_position.to(device=operator.device, dtype=operator.dtype)
            + self.operation_value_aperture_lift(spectral_context)
        )
        action_context = (
            action_context[:, None]
            .expand(-1, self.operator_count, -1, -1)
            .reshape(
                batch * self.operator_count,
                self.action_horizon,
                self.hidden_size,
            )
        )
        action_read, _ = self.operation_value_action_attention(
            self.operation_value_action_query_norm(tokens),
            self.operation_value_action_key_norm(action_context),
            self.operation_value_action_value_norm(action_context),
            need_weights=False,
        )
        tokens = tokens + action_read
        temporal = self.operation_value_temporal_norm(tokens)
        temporal, _ = self.operation_value_temporal(
            temporal, temporal, temporal, need_weights=False
        )
        tokens = tokens + temporal
        tokens = tokens + self.operation_value_ffn(self.operation_value_ffn_norm(tokens))
        field = self.operation_value_head(tokens).reshape(
            batch, self.operator_count, self.action_horizon, 2
        )
        return (
            field,
            memory_context.detach().float().square().mean().sqrt(),
            action_read.detach().float().square().mean().sqrt(),
        )

    def forward(
        self,
        *,
        previous_state: Tensor | None,
        global_intent: Tensor,
        flow_time: Tensor,
        refine_time: Tensor,
        action_tokens: Tensor,
        evidence_tokens: Tensor,
        evidence_ranges: dict[str, tuple[int, int]] | None,
        evidence_role_ranges: dict[str, tuple[tuple[int, int], ...]] | None,
        stage_role: Tensor,
        stage_content: Tensor,
        feedback: Tensor,
        current_block: Tensor | None = None,
        committed_spectral_aperture: Tensor | None = None,
        collect_diagnostics: bool = True,
    ) -> UnifiedControllerOutput:
        batch = int(action_tokens.shape[0])
        singleton_sources = (global_intent, flow_time, refine_time)
        for value in singleton_sources:
            if tuple(value.shape) != (batch, self.hidden_size):
                raise ValueError("controller singleton sources must be [B,H]")
        feedback_tokens = self._feedback_tokens(feedback).to(dtype=action_tokens.dtype)
        values = (
            global_intent[:, None],
            flow_time[:, None],
            refine_time[:, None],
            action_tokens,
            evidence_tokens,
            stage_role,
            stage_content,
            feedback_tokens,
        )
        typed_values: list[Tensor] = []
        typed_keys: list[Tensor] = []
        key_bias_parts: list[Tensor] = []
        ranges: dict[str, tuple[int, int]] = {}
        evidence_metric_ranges: dict[str, tuple[int, int]] = {}
        evidence_role_metric_ranges: dict[str, tuple[tuple[int, int], ...]] = {}
        offset = 0
        for index, (name, value) in enumerate(zip(self.SOURCE_NAMES, values, strict=True)):
            source_value, source_key = self._typed_source(value, index)
            typed_values.append(source_value)
            typed_keys.append(source_key)
            source = source_value
            if name == "evidence" and evidence_role_ranges:
                source_bias = torch.empty(
                    int(source.shape[1]), device=source.device, dtype=torch.float32
                )
                subgroup_count = len(evidence_role_ranges)
                covered = [False] * int(source.shape[1])
                for role_name, role_parts in evidence_role_ranges.items():
                    count = sum(int(stop) - int(start) for start, stop in role_parts)
                    if count <= 0:
                        raise ValueError("controller evidence role must contain tokens")
                    metric_parts: list[tuple[int, int]] = []
                    for start, stop in role_parts:
                        if not 0 <= int(start) < int(stop) <= int(source.shape[1]):
                            raise ValueError(
                                "controller evidence role range is outside evidence tokens"
                            )
                        if any(covered[int(start) : int(stop)]):
                            raise ValueError("controller evidence role ranges overlap")
                        source_bias[int(start) : int(stop)] = -math.log(
                            float(subgroup_count * count)
                        )
                        covered[int(start) : int(stop)] = [True] * (int(stop) - int(start))
                        metric_parts.append((offset + int(start), offset + int(stop)))
                    evidence_role_metric_ranges[role_name] = tuple(metric_parts)
                if not all(covered):
                    raise ValueError(
                        "controller evidence role ranges do not cover every evidence token"
                    )
                key_bias_parts.append(source_bias)
            else:
                key_bias_parts.append(
                    torch.full(
                        (int(source.shape[1]),),
                        -math.log(float(max(int(source.shape[1]), 1))),
                        device=source.device,
                        dtype=torch.float32,
                    )
                )
            ranges[name] = (offset, offset + int(source.shape[1]))
            if name == "evidence" and evidence_ranges:
                for evidence_name, (start, stop) in evidence_ranges.items():
                    if not 0 <= int(start) < int(stop) <= int(source.shape[1]):
                        raise ValueError(
                            "controller evidence source range is outside evidence tokens"
                        )
                    evidence_metric_ranges[evidence_name] = (
                        offset + int(start),
                        offset + int(stop),
                    )
            offset += int(source.shape[1])
        source_values = torch.cat(typed_values, dim=1)
        source_keys = torch.cat(typed_keys, dim=1)
        source_key_bias = torch.cat(key_bias_parts, dim=0)

        seed = self.control_seed.to(device=action_tokens.device, dtype=action_tokens.dtype).expand(
            batch, -1, -1
        )
        control_position = self.control_position.to(
            device=action_tokens.device, dtype=action_tokens.dtype
        )
        raw_state_address = (
            self.control_address.to(device=action_tokens.device, dtype=action_tokens.dtype)
            + 0.25 * control_position
        ).expand(batch, -1, -1)
        state_address = self.control_address_scale * self.control_address_norm(raw_state_address)
        function_code = (
            self.control_function.to(device=action_tokens.device, dtype=action_tokens.dtype)
            + 0.25 * control_position
        ).expand(batch, -1, -1)
        if previous_state is None:
            global_state = seed.mean(dim=1, keepdim=True)
            private_state = seed - global_state
            recurrence_change = torch.zeros((), device=action_tokens.device, dtype=torch.float32)
        else:
            if tuple(previous_state.shape) != tuple(seed.shape):
                raise ValueError(
                    f"controller recurrent state must be {tuple(seed.shape)}, "
                    f"got {tuple(previous_state.shape)}"
                )
            global_state = previous_state.mean(dim=1, keepdim=True)
            private_state = previous_state - global_state
            recurrence_change = (
                1.0
                - F.cosine_similarity(
                    previous_state.detach().float(), seed.detach().float(), dim=-1
                )
            ).mean()

        global_address = state_address.mean(dim=1, keepdim=True)
        global_function = function_code.mean(dim=1, keepdim=True)

        attention_rows: list[Tensor] = []
        ownership_rows: list[Tensor] = []
        residual_rows: list[Tensor] = []
        state_before = global_state + private_state
        for block in self.blocks:
            (
                global_state,
                private_state,
                _global_attention,
                attention,
                ownership,
                residual_rms,
            ) = block(
                global_state,
                private_state,
                global_address,
                state_address,
                global_function,
                function_code,
                source_keys,
                source_values,
                source_key_bias,
            )
            attention_rows.append(attention.detach().float())
            ownership_rows.append(ownership.detach().float())
            residual_rows.append(residual_rms)
        global_state = self.final_norm(global_state)
        private_state = self.final_norm(private_state)
        private_state = private_state - private_state.mean(dim=1, keepdim=True)
        state = global_state + private_state
        if previous_state is not None:
            recurrence_change = (
                1.0
                - F.cosine_similarity(state.detach().float(), state_before.detach().float(), dim=-1)
            ).mean()

        base_queries = self.output_query.to(device=state.device, dtype=state.dtype).expand(
            batch, -1, -1
        )
        operator_query = base_queries[:, : self.operator_count] + 0.25 * self.operator_position.to(
            device=state.device, dtype=state.dtype
        )
        query_groups = [operator_query]
        if self.spectral_state:
            frequency_query = self.frequency_state_proj(
                self.frequency_address.to(device=state.device, dtype=state.dtype)
                + 0.25 * self.frequency_position.to(device=state.device, dtype=state.dtype)
            ).expand(batch, -1, -1)
            query_groups.append(frequency_query)
        queries = torch.cat(query_groups, dim=1)

        # Functional readers receive typed global/private memory.  The
        # compatibility ``state`` remains available to the decoder, but it is
        # no longer the only value bank exposed to the readers.
        memory_content = torch.cat([global_state, private_state], dim=1)
        memory_address = torch.cat([global_address, state_address], dim=1)
        memory_count = int(memory_content.shape[1])
        read_keys = torch.cat([memory_content + memory_address, source_keys], dim=1)
        read_values = torch.cat([memory_content, source_values], dim=1)
        private_count = int(private_state.shape[1])
        memory_prior = torch.cat(
            [
                torch.full(
                    (1,),
                    -math.log(2.0),
                    device=source_key_bias.device,
                    dtype=source_key_bias.dtype,
                ),
                torch.full(
                    (private_count,),
                    -math.log(float(2 * max(private_count, 1))),
                    device=source_key_bias.device,
                    dtype=source_key_bias.dtype,
                ),
            ]
        )
        reader_key_bias = torch.cat(
            [
                memory_prior,
                source_key_bias,
            ]
        )
        attention_bias = (
            reader_key_bias[None]
            .expand(int(queries.shape[1]), -1)
            .to(device=queries.device, dtype=queries.dtype)
        )
        readout, output_attention = self.output_attn(
            self.output_query_norm(queries),
            self.output_key_norm(read_keys),
            self.output_state_norm(read_values),
            need_weights=True,
            average_attn_weights=False,
            attn_mask=attention_bias,
        )

        split_counts = [self.operator_count]
        if self.spectral_state:
            split_counts.append(self.action_horizon)
        split_readout = torch.split(readout, split_counts, dim=1)
        operator = split_readout[0]

        # Stage decisions use global stage-to-stage composition.
        operator = operator * (
            1.0
            + self.reader_function_strength
            * torch.tanh(self.operator_query_scale(self.output_query_norm(operator_query)))
        )
        coupled = self.output_coupling_norm(operator)
        coupled, _ = self.output_coupling_attn(coupled, coupled, coupled, need_weights=False)
        operator = operator + coupled
        operator = operator + self.output_ffn(self.output_ffn_norm(operator))

        operator_output = self.operator_head(operator)
        if self.block_owned_execution:
            operator_depth_logits = operator_output
        else:
            update_stop = self.operator_branch_count
            operator_depth_logits = operator_output[..., update_stop:]
        # Compatibility/audit tensor only. It has no live execution path;
        # sigmoid(20) records the exact neutral boundary instead of exposing
        # a second controller-owned amplitude signal to downstream code.
        operator_update_logits = torch.full_like(operator_depth_logits, 20.0)

        if self.spectral_state:
            frequency = split_readout[1]
            frequency = frequency * (
                1.0
                + self.reader_function_strength
                * torch.tanh(self.frequency_query_scale(self.output_query_norm(frequency_query)))
            )
            local = self.frequency_local(
                self.frequency_function_norm(frequency).transpose(1, 2)
            ).transpose(1, 2)
            frequency = frequency + 0.25 * local
            frequency = frequency + self.frequency_ffn(self.frequency_ffn_norm(frequency))
            spectral_shift = self.frequency_shift_head(frequency)

            frequency_start = self.operator_count
            spectral_ownership = (
                output_attention[:, :, frequency_start:, 1:memory_count]
                .float()
                .mean(dim=1)
                .transpose(1, 2)
            )
            spectral_ownership = spectral_ownership / spectral_ownership.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-8)
            slot_mass = spectral_ownership.mean(dim=-1)
            ownership_entropy = (
                -(spectral_ownership.clamp_min(1e-8) * spectral_ownership.clamp_min(1e-8).log())
                .sum(dim=1)
                .mean()
            )
            slot_balance = (
                slot_mass - 1.0 / float(max(self.control_count, 1))
            ).square().mean() * float(max(self.control_count, 1)) ** 2
            if int(spectral_ownership.shape[-1]) > 1:
                ownership_smoothness = (
                    (spectral_ownership[..., 1:] - spectral_ownership[..., :-1]).square().mean()
                )
            else:
                ownership_smoothness = torch.zeros((), device=state.device, dtype=torch.float32)
            spectral_competition_loss = (
                ownership_entropy / math.log(float(max(self.control_count, 2)))
                + 2.0 * slot_balance
                + 0.05 * ownership_smoothness
            )
        else:
            spectral_ownership = torch.full(
                (
                    batch,
                    self.control_count,
                    self.action_horizon,
                ),
                1.0 / float(max(self.control_count, 1)),
                device=state.device,
                dtype=torch.float32,
            )
            spectral_shift = torch.zeros(
                batch,
                self.action_horizon,
                2,
                device=state.device,
                dtype=state.dtype,
            )
            ownership_entropy = torch.zeros((), device=state.device, dtype=torch.float32)
            slot_balance = torch.zeros((), device=state.device, dtype=torch.float32)
            ownership_smoothness = torch.zeros((), device=state.device, dtype=torch.float32)
            spectral_competition_loss = torch.zeros((), device=state.device, dtype=torch.float32)

        (
            operation_value_field,
            operation_value_memory_context_rms,
            operation_value_action_context_rms,
        ) = self._operation_value_field(
            operator,
            current_block=current_block,
            memory_content=memory_content,
            memory_address=memory_address,
            action_tokens=action_tokens,
            committed_spectral_aperture=committed_spectral_aperture,
            proposed_spectral_shift=spectral_shift,
        )

        def build_output(output_metrics: dict[str, Tensor]) -> UnifiedControllerOutput:
            return UnifiedControllerOutput(
                state=state,
                global_state=global_state,
                private_state=private_state,
                state_address=state_address,
                memory=ControllerMemory(
                    content=memory_content,
                    address=memory_address,
                    global_content=global_state,
                    private_content=private_state,
                    global_address=global_address,
                    private_address=state_address,
                ),
                execution=ControllerExecutionContract(
                    operation_value_field=operation_value_field,
                    branch_capacity_logits=operator_depth_logits,
                    spectral_shift=spectral_shift,
                    # Do not expose a live controller-owned amplitude actuator
                    # to the decoder.  The parameter is retained in the
                    # module for old state-dict compatibility only.
                    branch_update_keep_logits=None,
                    operation_axis=("block" if self.block_owned_execution else "stage"),
                    residual_amplitude_owner="host_block",
                ),
                operation_value_field=operation_value_field,
                operator_update_logits=operator_update_logits,
                operator_depth_logits=operator_depth_logits,
                spectral_shift=spectral_shift,
                spectral_ownership=spectral_ownership,
                spectral_competition_loss=spectral_competition_loss,
                metrics=output_metrics,
            )

        if not collect_diagnostics:
            return build_output({})

        metrics = self._state_metrics(state)
        private_metrics = self._state_metrics(private_state)
        for name, value in private_metrics.items():
            suffix = name.removeprefix("controller_state_")
            metrics[f"controller_private_{suffix}"] = value
        metrics["controller_global_norm"] = global_state.detach().float().norm(dim=-1).mean()
        metrics["controller_private_global_energy_ratio"] = (
            private_state.detach().float().square().sum()
            / state.detach().float().square().sum().clamp_min(1e-8)
        )
        if residual_rows:
            metrics["controller_private_residual_value_rms"] = torch.stack(residual_rows).mean()
        metrics["controller_recurrent_change"] = recurrence_change.detach().float()
        operation_value_fp32 = operation_value_field.detach().float()
        operation_value_centered = operation_value_fp32 - operation_value_fp32.mean(
            dim=1, keepdim=True
        )
        metrics["controller_operation_value_rms"] = operation_value_fp32.square().mean().sqrt()
        metrics["controller_operation_value_stage_spread"] = (
            operation_value_centered.square().mean().sqrt()
        )
        if self.block_owned_execution:
            metrics["controller_operation_value_block_spread"] = (
                operation_value_centered.square().mean().sqrt()
            )
        metrics["controller_operation_value_common_mode_ratio"] = operation_value_fp32.mean(
            dim=1
        ).square().mean().sqrt() / operation_value_fp32.square().mean().sqrt().clamp_min(1e-8)
        metrics["controller_operation_value_memory_context_rms"] = (
            operation_value_memory_context_rms
        )
        metrics["controller_operation_value_action_context_rms"] = (
            operation_value_action_context_rms
        )
        raw_update_keep = torch.sigmoid(operator_update_logits.detach().float())
        raw_depth_keep = torch.sigmoid(operator_depth_logits.detach().float())
        metrics["controller_compat_update_mean"] = raw_update_keep.mean()
        metrics["controller_operator_raw_depth_mean"] = torch.sigmoid(
            operator_depth_logits.detach().float()
        ).mean()
        metrics["controller_compat_update_stage_std"] = (
            raw_update_keep.mean(dim=-1).std(dim=-1, unbiased=False).mean()
        )
        metrics["controller_operator_depth_stage_std"] = (
            raw_depth_keep.mean(dim=-1).std(dim=-1, unbiased=False).mean()
        )
        if self.block_owned_execution:
            metrics["controller_operator_depth_block_std"] = (
                raw_depth_keep.mean(dim=-1).std(dim=-1, unbiased=False).mean()
            )
        update_centered = raw_update_keep - raw_update_keep.mean(dim=(1, 2), keepdim=True)
        depth_centered = raw_depth_keep - raw_depth_keep.mean(dim=(1, 2), keepdim=True)
        metrics["controller_compat_update_depth_correlation"] = (
            (update_centered * depth_centered).sum(dim=(1, 2))
            / (
                update_centered.square().sum(dim=(1, 2)).sqrt()
                * depth_centered.square().sum(dim=(1, 2)).sqrt()
            ).clamp_min(1e-8)
        ).mean()
        metrics["controller_compat_joint_suppression_mass"] = (
            (1.0 - raw_update_keep) * (1.0 - raw_depth_keep)
        ).mean()
        if self.block_owned_execution:
            metrics["controller_host_amplitude_owned"] = torch.ones(
                (), device=state.device, dtype=torch.float32
            )
            metrics["controller_operation_count"] = torch.tensor(
                float(self.operator_count),
                device=state.device,
                dtype=torch.float32,
            )
        output_attention_fp32 = output_attention.detach().float()
        output_entropy = -(
            output_attention_fp32.clamp_min(1e-8) * output_attention_fp32.clamp_min(1e-8).log()
        ).sum(dim=-1)
        output_slot_load = output_attention_fp32.mean(dim=(1, 2))
        output_slot_load = output_slot_load / output_slot_load.sum(dim=-1, keepdim=True).clamp_min(
            1e-8
        )
        output_slot_load_entropy = -(
            output_slot_load.clamp_min(1e-8) * output_slot_load.clamp_min(1e-8).log()
        ).sum(dim=-1)
        metrics["controller_reader_effective_keys"] = torch.exp(output_entropy).mean()
        metrics["controller_reader_key_max"] = output_attention_fp32.amax(dim=-1).mean()
        metrics["controller_reader_load_effective_keys"] = torch.exp(
            output_slot_load_entropy
        ).mean()
        metrics["controller_operator_representation_diversity"] = (
            (operator.detach().float() - operator.detach().float().mean(dim=1, keepdim=True))
            .norm(dim=-1)
            .mean()
        )
        reader_attention = output_attention_fp32.mean(dim=1)
        reader_groups = {
            "operator": (0, self.operator_count),
        }
        if self.spectral_state:
            reader_groups["spectral"] = (
                self.operator_count,
                self.operator_count + self.action_horizon,
            )
        family_attention_profiles: list[Tensor] = []
        for reader_name, (query_start, query_stop) in reader_groups.items():
            family_attention = reader_attention[:, query_start:query_stop]
            family_attention_profiles.append(family_attention.mean(dim=1))
            metrics[f"controller_reader_{reader_name}_memory_attention"] = (
                family_attention[..., :memory_count].sum(dim=-1).mean()
            )
            metrics[f"controller_reader_{reader_name}_global_memory_attention"] = (
                family_attention[..., :1].sum(dim=-1).mean()
            )
            metrics[f"controller_reader_{reader_name}_private_memory_attention"] = (
                family_attention[..., 1:memory_count].sum(dim=-1).mean()
            )
            for source_name, (source_start, source_stop) in ranges.items():
                metrics[f"controller_reader_{reader_name}_source_{source_name}_attention"] = (
                    family_attention[
                        ...,
                        memory_count + source_start : memory_count + source_stop,
                    ]
                    .sum(dim=-1)
                    .mean()
                )
            metrics[f"controller_reader_{reader_name}_mass_error"] = (
                (family_attention.sum(dim=-1) - 1.0).abs().mean()
            )
            metrics[f"controller_reader_{reader_name}_attention_diversity"] = (
                (family_attention - family_attention.mean(dim=1, keepdim=True))
                .square()
                .sum(dim=-1)
                .sqrt()
                .mean()
            )
        family_profiles = torch.stack(family_attention_profiles, dim=1)
        metrics["controller_reader_family_attention_diversity"] = (
            (family_profiles - family_profiles.mean(dim=1, keepdim=True))
            .square()
            .sum(dim=-1)
            .sqrt()
            .mean()
        )
        if self.spectral_state and self.action_horizon > 1:
            spectral_attention = reader_attention[:, self.operator_count :]
            metrics["controller_reader_spectral_attention_local_change"] = (
                (spectral_attention[:, 1:] - spectral_attention[:, :-1])
                .square()
                .sum(dim=-1)
                .sqrt()
                .mean()
            )
        else:
            metrics["controller_reader_spectral_attention_local_change"] = torch.zeros(
                (), device=state.device, dtype=torch.float32
            )
        if self.spectral_state:
            metrics["controller_spectral_representation_local_change"] = (
                (frequency.detach().float()[:, 1:] - frequency.detach().float()[:, :-1])
                .norm(dim=-1)
                .mean()
            )
        else:
            metrics["controller_spectral_representation_local_change"] = torch.zeros(
                (), device=state.device, dtype=torch.float32
            )
        metrics["controller_spectral_ownership_entropy"] = ownership_entropy.detach().float()
        metrics["controller_spectral_slot_balance"] = slot_balance.detach().float()
        metrics["controller_spectral_ownership_smoothness"] = ownership_smoothness.detach().float()
        metrics["controller_spectral_competition_loss"] = spectral_competition_loss.detach().float()
        metrics["controller_spectral_shift_rms"] = (
            spectral_shift.detach().float().square().mean().sqrt()
        )
        metrics["controller_spectral_state"] = torch.tensor(
            float(self.spectral_state), device=state.device, dtype=torch.float32
        )
        metrics["controller_spectral_effective_slots"] = (
            torch.exp(
                -(spectral_ownership.clamp_min(1e-8) * spectral_ownership.clamp_min(1e-8).log())
                .sum(dim=1)
                .mean(dim=-1)
            ).mean()
            if self.spectral_state
            else torch.zeros((), device=state.device, dtype=torch.float32)
        )
        if attention_rows:
            attention = torch.stack(attention_rows).mean(dim=(0, 2, 3))
            for name, (start, stop) in ranges.items():
                metrics[f"controller_source_{name}_attention"] = (
                    attention[:, start:stop].sum(dim=-1).mean()
                )
            for name, (start, stop) in evidence_metric_ranges.items():
                metrics[f"controller_evidence_{name}_attention"] = (
                    attention[:, start:stop].sum(dim=-1).mean()
                )
            for name, role_parts in evidence_role_metric_ranges.items():
                metrics[f"controller_evidence_role_{name}_attention"] = torch.stack(
                    [attention[:, start:stop].sum(dim=-1).mean() for start, stop in role_parts]
                ).sum()
            metrics["controller_source_attention_mass_error"] = (
                (attention.sum(dim=-1) - 1.0).abs().mean()
            )
        if ownership_rows:
            ownership = torch.stack(ownership_rows).mean(dim=0)
            source_entropy = -(ownership.clamp_min(1e-8) * ownership.clamp_min(1e-8).log()).sum(
                dim=-2
            )
            slot_load = ownership.mean(dim=(1, 3))
            slot_load = slot_load / slot_load.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            slot_load_entropy = -(slot_load.clamp_min(1e-8) * slot_load.clamp_min(1e-8).log()).sum(
                dim=-1
            )
            metrics.update(
                {
                    "controller_competition_source_effective_slots": (
                        torch.exp(source_entropy).mean()
                    ),
                    "controller_competition_source_owner_max": (ownership.amax(dim=-2).mean()),
                    "controller_competition_slot_load_effective": (
                        torch.exp(slot_load_entropy).mean()
                    ),
                    "controller_competition_slot_load_max": (slot_load.amax(dim=-1).mean()),
                }
            )
        return build_output(metrics)
