"""Typed, read-only time-domain Evidence MMDiT action bottom.

The historical capability name suggested that the active action path still
contained a CVAE posterior and hierarchical workspace.  Source inspection
shows that formal runs actually select a deterministic latent organizer and a
three-block time-domain Evidence MMDiT.  This module preserves those useful
mechanisms while removing the legacy aliases that made the bottom boundary
ambiguous:

* the protected P3 consequence is read exactly once;
* precision, temporal and state-change lanes remain exact zero when their
  source is zero;
* type, horizon and basis identities are selector-only information;
* noisy action may query evidence but can never write evidence values;
* each host block executes once.  Capacity and continuation are bounded,
  differentiable controls, while execution cost is audit-only.

There is no posterior, target input, hidden workspace, candidate block replay
or version-dependent branch in this implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import ExperimentConfig
from ..interfaces import ObservableHistory
from .compiler import ObjectPolicyPlanDeltaBank
from .routing import smooth_rms_contract
from .types import ControlledTransitionState


def sinusoidal_positions(length: int, width: int, *, device: torch.device) -> Tensor:
    """Deterministic positions with no learned value contribution."""

    position = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    half = max(width // 2, 1)
    frequency = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=device, dtype=torch.float32)
        / float(max(half - 1, 1))
    )[None]
    value = torch.cat((torch.sin(position * frequency), torch.cos(position * frequency)), dim=-1)
    if int(value.shape[-1]) < width:
        value = F.pad(value, (0, width - int(value.shape[-1])))
    return value[:, :width]


def _logit(probability: float) -> float:
    value = min(max(float(probability), 1.0e-4), 1.0 - 1.0e-4)
    return math.log(value / (1.0 - value))


def canonical_state_history(history: ObservableHistory) -> Tensor:
    """Return one causal state sequence whose final row is current state.

    Dataset offsets are ``(-8, -4, 0)``.  The last history row therefore
    already represents the current state.  Replacing that row is robust to a
    caller with small numerical drift and, unlike append, cannot duplicate the
    present observation.
    """

    value = history.state_history
    if value.ndim != 3 or int(value.shape[1]) < 1:
        raise ValueError("state history must contain at least one causal row")
    if int(value.shape[0]) != int(history.state.shape[0]) or int(value.shape[2]) != int(
        history.state.shape[1]
    ):
        raise ValueError("state history and current state do not align")
    if int(value.shape[1]) == 1:
        return history.state[:, None]
    return torch.cat((value[:, :-1], history.state[:, None]), dim=1)


class TimeCondition(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.network = nn.Sequential(
            nn.Linear(hidden, 4 * hidden, bias=False),
            nn.SiLU(),
            nn.Linear(4 * hidden, hidden, bias=False),
        )

    def forward(self, time: Tensor) -> Tensor:
        if time.ndim != 1:
            raise ValueError("flow time must be [B]")
        half = max(self.hidden // 2, 1)
        frequency = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=time.device, dtype=torch.float32)
            / float(max(half - 1, 1))
        )
        angle = time.float()[:, None] * frequency[None]
        embedding = torch.cat((torch.sin(angle), torch.cos(angle)), dim=-1)
        if int(embedding.shape[-1]) < self.hidden:
            embedding = F.pad(embedding, (0, self.hidden - int(embedding.shape[-1])))
        network_dtype = next(self.network.parameters()).dtype
        return self.network(embedding[:, : self.hidden].to(dtype=network_dtype))


class ActionQueryEncoder(nn.Module):
    """Create the sole noisy physical-field query shared by P2/P3/bottom."""

    def __init__(self, *, action_dim: int, hidden: int, horizon: int, basis: int) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.hidden = int(hidden)
        self.horizon = int(horizon)
        self.basis = int(basis)
        self.action = nn.Linear(action_dim, hidden, bias=False)
        self.time = TimeCondition(hidden)
        self.basis_identity = nn.Parameter(torch.randn(1, 1, basis, hidden) * 0.02)
        self.register_buffer(
            "horizon_position",
            sinusoidal_positions(horizon, hidden, device=torch.device("cpu"))[None, :, None],
            persistent=True,
        )

    def forward(self, noisy_action_field: Tensor, time: Tensor) -> Tensor:
        if tuple(noisy_action_field.shape[1:]) != (self.horizon, self.action_dim):
            raise ValueError("noisy physical action field must be [B,T,Aphysical]")
        batch = int(noisy_action_field.shape[0])
        if tuple(time.shape) != (batch,):
            raise ValueError("flow time and noisy action batch do not align")
        action = self.action(noisy_action_field)[:, :, None]
        return (
            action
            + self.time(time).to(dtype=action.dtype)[:, None, None]
            + self.horizon_position.to(device=action.device, dtype=action.dtype)
            + self.basis_identity.to(device=action.device, dtype=action.dtype)
        )


def _floored_unit(value: Tensor, floor: float = 0.25) -> Tensor:
    """RMS normalization with a finite Jacobian and exact-zero semantics."""

    value_f = value.float()
    denominator = torch.sqrt(value_f.square().mean(dim=-1, keepdim=True) + float(floor) ** 2)
    return (value_f / denominator).to(dtype=value.dtype)


class ProtectedConsequenceReader(nn.Module):
    """Read the protected consequence once without a null/optional route."""

    def __init__(self, *, hidden: int, basis: int) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.basis = int(basis)
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.value = nn.Linear(hidden, hidden, bias=False)
        self.basis_key = nn.Parameter(torch.randn(1, 1, basis, hidden) * 0.02)

    def forward(
        self,
        action_query: Tensor,
        protected: Tensor,
        *,
        collect_diagnostics: bool,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if tuple(action_query.shape) != tuple(protected.shape):
            raise ValueError("protected consequence and action query must align")
        if int(action_query.shape[-2]) != self.basis:
            raise ValueError("protected consequence lost its basis axis")
        query = _floored_unit(self.query(action_query.mean(dim=2)))
        key = _floored_unit(
            self.key(protected) + self.basis_key.to(device=protected.device, dtype=protected.dtype)
        )
        score = torch.einsum("bth,btqh->btq", query.float(), key.float())
        score = score / math.sqrt(float(self.hidden))
        probability = torch.softmax(score, dim=-1).to(dtype=protected.dtype)
        update = torch.einsum("btq,btqh->bth", probability, self.value(protected))
        update, _ = smooth_rms_contract(update, 0.50)
        if not collect_diagnostics:
            return update, {}
        probability_f = probability.detach().float()
        entropy = -(probability_f * probability_f.clamp_min(1e-8).log()).sum(dim=-1)
        if self.basis > 1:
            entropy = entropy / math.log(float(self.basis))
        return update, {
            "bottom_protected_update_rms": update.detach().float().square().mean().sqrt(),
            "bottom_protected_basis_entropy": entropy.mean(),
            "bottom_protected_basis_max": probability_f.amax(dim=-1).mean(),
        }


@dataclass(frozen=True)
class TypedEvidenceBank:
    selector: Tensor
    value: Tensor
    lane_ranges: dict[str, tuple[int, int]]

    def validate(self, *, hidden: int) -> None:
        if self.selector.ndim != 3 or tuple(self.selector.shape) != tuple(self.value.shape):
            raise ValueError("bottom selector/value evidence must align as [B,N,H]")
        if int(self.selector.shape[-1]) != int(hidden) or int(self.selector.shape[1]) < 1:
            raise ValueError("bottom evidence bank has an invalid hidden/token axis")


class TypedEvidenceCompiler(nn.Module):
    """Compile optional P3 lanes and observable history into separate K/V."""

    LANE_NAMES = ("precision", "temporal", "state_change")
    VALUE_LANE_NAMES = (*LANE_NAMES, "controlled_transition")

    def __init__(
        self,
        *,
        hidden: int,
        state_dim: int,
        action_dim: int,
        horizon: int,
        basis: int,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.horizon = int(horizon)
        self.basis = int(basis)
        self.optional_key = nn.ModuleDict(
            {name: nn.Linear(hidden, hidden, bias=False) for name in self.LANE_NAMES}
        )
        self.optional_value = nn.ModuleDict(
            {name: nn.Linear(hidden, hidden, bias=False) for name in self.LANE_NAMES}
        )
        self.state_key = nn.Linear(state_dim, hidden, bias=False)
        self.state_value = nn.Linear(state_dim, hidden, bias=False)
        self.action_key = nn.Linear(action_dim, hidden, bias=False)
        self.action_value = nn.Linear(action_dim, hidden, bias=False)
        self.transition_key = nn.Linear(hidden, hidden, bias=False)
        self.transition_value = nn.Linear(hidden, hidden, bias=False)
        # Identities are keys only.  They are never added to values.
        self.type_key = nn.Parameter(torch.randn(6, hidden) * 0.02)
        self.basis_key = nn.Parameter(torch.randn(1, 1, basis, hidden) * 0.02)
        self.register_buffer(
            "horizon_key",
            sinusoidal_positions(horizon, hidden, device=torch.device("cpu"))[None, :, None],
            persistent=True,
        )

    def forward(
        self,
        plan: ObjectPolicyPlanDeltaBank,
        history: ObservableHistory,
        transition: ControlledTransitionState,
    ) -> TypedEvidenceBank:
        plan.validate()
        transition.validate(hidden=self.hidden)
        state_history = canonical_state_history(history)
        batch = int(plan.protected_base.shape[0])
        selector_rows: list[Tensor] = []
        value_rows: list[Tensor] = []
        ranges: dict[str, tuple[int, int]] = {}
        cursor = 0
        horizon_key = self.horizon_key.to(
            device=plan.protected_base.device,
            dtype=plan.protected_base.dtype,
        )
        basis_key = self.basis_key.to(
            device=plan.protected_base.device,
            dtype=plan.protected_base.dtype,
        )
        for index, name in enumerate(self.LANE_NAMES):
            source = getattr(plan, name)
            selector = (
                self.optional_key[name](source)
                + self.type_key[index].to(device=source.device, dtype=source.dtype)
                + horizon_key
                + basis_key
            ).reshape(batch, self.horizon * self.basis, self.hidden)
            value = self.optional_value[name](source).reshape(
                batch, self.horizon * self.basis, self.hidden
            )
            selector_rows.append(selector)
            value_rows.append(value)
            stop = cursor + int(selector.shape[1])
            ranges[name] = (cursor, stop)
            cursor = stop

        state_selector = self.state_key(state_history)
        state_selector = state_selector + self.type_key[3].to(
            device=state_selector.device, dtype=state_selector.dtype
        )
        state_value = self.state_value(state_history)
        selector_rows.append(state_selector)
        value_rows.append(state_value)
        stop = cursor + int(state_selector.shape[1])
        ranges["state_history"] = (cursor, stop)
        cursor = stop

        action_history = history.executed_action_history
        action_selector = self.action_key(action_history)
        action_selector = action_selector + self.type_key[4].to(
            device=action_selector.device, dtype=action_selector.dtype
        )
        action_value = self.action_value(action_history)
        selector_rows.append(action_selector)
        value_rows.append(action_value)
        stop = cursor + int(action_selector.shape[1])
        ranges["executed_action_history"] = (cursor, stop)
        cursor = stop

        transition_selector = self.transition_key(transition.selector)
        transition_selector = transition_selector + self.type_key[5].to(
            device=transition_selector.device,
            dtype=transition_selector.dtype,
        )
        transition_value = self.transition_value(transition.value)
        selector_rows.append(transition_selector)
        value_rows.append(transition_value)
        stop = cursor + int(transition_selector.shape[1])
        ranges["controlled_transition"] = (cursor, stop)

        bank = TypedEvidenceBank(
            selector=torch.cat(selector_rows, dim=1),
            value=torch.cat(value_rows, dim=1),
            lane_ranges=ranges,
        )
        bank.validate(hidden=self.hidden)
        return bank


class BottomConditionOrganizer(nn.Module):
    """Deterministic history bottleneck; no target or posterior path."""

    def __init__(
        self,
        *,
        hidden: int,
        latent: int,
        state_dim: int,
        action_dim: int,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.state = nn.Linear(state_dim, hidden, bias=False)
        self.action = nn.Linear(action_dim, hidden, bias=False)
        self.state_scan = nn.GRUCell(hidden, hidden)
        self.action_scan = nn.GRUCell(hidden, hidden)
        for module in (self.state_scan, self.action_scan):
            nn.init.zeros_(module.bias_ih)
            nn.init.zeros_(module.bias_hh)
        self.initial_state = nn.Parameter(torch.zeros(1, hidden))
        self.initial_action = nn.Parameter(torch.zeros(1, hidden))
        self.condition = nn.Sequential(
            nn.LayerNorm(2 * hidden, elementwise_affine=False),
            nn.Linear(2 * hidden, hidden, bias=False),
            nn.SiLU(),
            nn.Linear(hidden, hidden, bias=False),
        )
        self.latent_mu = nn.Linear(hidden, latent, bias=False)
        self.latent_to_hidden = nn.Linear(latent, hidden, bias=False)
        self.time = TimeCondition(hidden)

    def forward(
        self,
        history: ObservableHistory,
        time: Tensor,
        *,
        collect_diagnostics: bool,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        states = self.state(canonical_state_history(history))
        actions = self.action(history.executed_action_history)
        batch = int(states.shape[0])
        state_hidden = self.initial_state.to(device=states.device, dtype=states.dtype).expand(
            batch, -1
        )
        action_hidden = self.initial_action.to(device=actions.device, dtype=actions.dtype).expand(
            batch, -1
        )
        for index in range(int(states.shape[1])):
            state_hidden = self.state_scan(states[:, index], state_hidden)
        for index in range(int(actions.shape[1])):
            action_hidden = self.action_scan(actions[:, index], action_hidden)
        condition = self.condition(torch.cat((state_hidden, action_hidden), dim=-1))
        latent = self.latent_mu(condition)
        global_condition = self.latent_to_hidden(latent) + self.time(time).to(dtype=condition.dtype)
        if not collect_diagnostics:
            return global_condition, {}
        return global_condition, {
            "bottom_condition_rms": condition.detach().float().square().mean().sqrt(),
            "bottom_latent_rms": latent.detach().float().square().mean().sqrt(),
            "bottom_latent_batch_variance": latent.detach()
            .float()
            .var(dim=0, unbiased=False)
            .mean(),
        }


class BiasFreeFFN(nn.Module):
    def __init__(self, hidden: int, expansion: float) -> None:
        super().__init__()
        inner = int(round(hidden * float(expansion)))
        self.network = nn.Sequential(
            nn.Linear(hidden, inner, bias=False),
            nn.GELU(),
            nn.Linear(inner, hidden, bias=False),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.network(value)


class ReadOnlyEvidenceMMDiTBlock(nn.Module):
    """One causal action block; evidence selectors and values are immutable."""

    def __init__(
        self,
        *,
        hidden: int,
        heads: int,
        expansion: float,
        dropout: float,
        residual_scale_max: float,
        residual_scale_init: float,
        normalization_floor: float,
    ) -> None:
        super().__init__()
        if hidden % heads:
            raise ValueError("bottom hidden size must be divisible by heads")
        self.hidden = int(hidden)
        self.heads = int(heads)
        self.residual_scale_max = float(residual_scale_max)
        self.normalization_floor = float(normalization_floor)
        self.action_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.evidence_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.action_qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.self_out = nn.Linear(hidden, hidden, bias=False)
        self.evidence_query = nn.Linear(hidden, hidden, bias=False)
        self.evidence_key = nn.Linear(hidden, hidden, bias=False)
        self.evidence_value = nn.Linear(hidden, hidden, bias=False)
        self.evidence_out = nn.Linear(hidden, hidden, bias=False)
        self.ffn_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.ffn = BiasFreeFFN(hidden, expansion)
        self.global_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.modulation = nn.Linear(hidden, 6 * hidden)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)
        ratio = min(
            max(float(residual_scale_init) / max(self.residual_scale_max, 1.0e-6), 1.0e-4),
            0.95,
        )
        with torch.no_grad():
            gate_bias = math.atanh(ratio)
            self.modulation.bias[2 * hidden : 3 * hidden].fill_(gate_bias)
            self.modulation.bias[5 * hidden : 6 * hidden].fill_(gate_bias)

    def _split(self, value: Tensor) -> Tensor:
        batch, tokens, hidden = value.shape
        return value.reshape(batch, tokens, self.heads, hidden // self.heads).transpose(1, 2)

    @staticmethod
    def _merge(value: Tensor) -> Tensor:
        batch, heads, tokens, width = value.shape
        return value.transpose(1, 2).reshape(batch, tokens, heads * width)

    @staticmethod
    def _attention(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        score = torch.matmul(query.float(), key.float().transpose(-2, -1))
        score = score / math.sqrt(float(query.shape[-1]))
        if mask is not None:
            score = score.masked_fill(mask[None, None], torch.finfo(score.dtype).min)
        probability = torch.softmax(score, dim=-1).to(dtype=value.dtype)
        return torch.matmul(probability, value), probability

    def forward(
        self,
        action: Tensor,
        evidence: TypedEvidenceBank,
        condition: Tensor,
        *,
        collect_diagnostics: bool,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        evidence.validate(hidden=self.hidden)
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = self.modulation(
            self.global_norm(condition)
        ).chunk(6, dim=-1)
        action_value = self.action_norm(action)
        modulated = action_value * (1.0 + scale_a[:, None]) + shift_a[:, None]
        action_query, action_key, action_value_stream = (
            self._split(part) for part in self.action_qkv(modulated).chunk(3, dim=-1)
        )
        length = int(action.shape[1])
        causal = torch.triu(
            torch.ones(length, length, device=action.device, dtype=torch.bool), diagonal=1
        )
        self_read, _ = self._attention(action_query, action_key, action_value_stream, causal)
        self_direction = _floored_unit(self.self_out(self._merge(self_read)))

        evidence_query = self._split(self.evidence_query(modulated))
        evidence_key = self._split(self.evidence_key(self.evidence_norm(evidence.selector)))
        evidence_value = self._split(
            self.evidence_value(_floored_unit(evidence.value, self.normalization_floor))
        )
        evidence_read, evidence_probability = self._attention(
            evidence_query, evidence_key, evidence_value, None
        )
        evidence_direction = _floored_unit(self.evidence_out(self._merge(evidence_read)))
        direction = (self_direction + evidence_direction) / math.sqrt(2.0)
        attention_gate = self.residual_scale_max * torch.tanh(gate_a)[:, None]
        attention_update = attention_gate * self.dropout(direction)
        intermediate = action + attention_update
        ffn_source = self.ffn_norm(intermediate) * (1.0 + scale_f[:, None]) + shift_f[:, None]
        ffn_direction = _floored_unit(self.ffn(ffn_source))
        ffn_gate_value = self.residual_scale_max * torch.tanh(gate_f)[:, None]
        ffn_update = ffn_gate_value * self.dropout(ffn_direction)
        update = attention_update + ffn_update
        if not collect_diagnostics:
            return update, {}
        probability_f = evidence_probability.detach().float().clamp_min(1e-8)
        entropy = -(probability_f * probability_f.log()).sum(dim=-1)
        if int(probability_f.shape[-1]) > 1:
            entropy = entropy / math.log(float(probability_f.shape[-1]))
        return update, {
            "update_rms": update.detach().float().square().mean().sqrt(),
            "self_update_rms": (attention_gate * self_direction)
            .detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "evidence_update_rms": (attention_gate * evidence_direction)
            .detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "evidence_attention_entropy": entropy.mean(),
            "evidence_attention_max": probability_f.amax(dim=-1).mean(),
            "attention_gate_abs": attention_gate.detach().float().abs().mean(),
            "ffn_gate_abs": ffn_gate_value.detach().float().abs().mean(),
        }


class NestedCapacityOperator(nn.Module):
    """Non-expansive ordered low-rank capacity for one host residual."""

    def __init__(self, *, hidden: int, rank: int, groups: int) -> None:
        super().__init__()
        if min(hidden, rank, groups) <= 0 or rank > hidden or rank % groups:
            raise ValueError("nested capacity dimensions are invalid")
        self.hidden = int(hidden)
        self.rank = int(rank)
        self.groups = int(groups)
        self.channels_per_group = rank // groups
        self.basis_raw = nn.Parameter(torch.empty(hidden, rank))
        self.register_buffer(
            "_eval_basis_cache",
            torch.empty(0, dtype=torch.float32),
            persistent=False,
        )
        self._eval_basis_version = -1
        nn.init.normal_(self.basis_raw, mean=0.0, std=float(hidden) ** -0.5)
        with torch.no_grad():
            self.basis_raw.copy_(self._basis())

    def _orthonormal_basis(self) -> Tensor:
        basis, triangular = torch.linalg.qr(self.basis_raw.float(), mode="reduced")
        diagonal = torch.diagonal(triangular)
        sign = torch.where(diagonal < 0.0, -torch.ones_like(diagonal), torch.ones_like(diagonal))
        return basis * sign[None]

    def _clear_eval_basis_cache(self) -> None:
        self._eval_basis_cache = self.basis_raw.new_empty(0, dtype=torch.float32)
        self._eval_basis_version = -1

    def train(self, mode: bool = True) -> "NestedCapacityOperator":
        # Training needs a fresh differentiable QR after every optimizer
        # update.  Evaluation/deployment weights are immutable, so retaining
        # the same orthonormal basis across five ODE calls is exact and avoids
        # repeating three 512x32 QR decompositions at every step.
        if mode:
            self._clear_eval_basis_cache()
        super().train(mode)
        return self

    def _basis(self) -> Tensor:
        if self.training or torch.is_grad_enabled():
            return self._orthonormal_basis()
        version = int(self.basis_raw._version)
        expected = (self.hidden, self.rank)
        if (
            self._eval_basis_version == version
            and tuple(self._eval_basis_cache.shape) == expected
            and self._eval_basis_cache.device == self.basis_raw.device
        ):
            return self._eval_basis_cache
        basis = self._orthonormal_basis()
        self._eval_basis_cache = basis
        self._eval_basis_version = version
        return basis

    def forward(
        self,
        update: Tensor,
        capacity: Tensor,
        *,
        collect_diagnostics: bool,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if update.ndim != 3 or int(update.shape[-1]) != self.hidden:
            raise ValueError("capacity update must be [B,T,H]")
        if tuple(capacity.shape) != (int(update.shape[0]),):
            raise ValueError("capacity ratio must be [B]")
        basis = self._basis()
        group_depth = capacity.float().clamp(0.0, 1.0) * float(self.groups)
        index = torch.arange(self.groups, device=update.device, dtype=torch.float32)
        group_transparency = (group_depth[:, None] - index[None]).clamp(0.0, 1.0)
        transparency = group_transparency.repeat_interleave(self.channels_per_group, dim=-1)
        coordinates = torch.einsum("bth,hr->btr", update.float(), basis)
        # Capacity owns an ordered low-rank update, not the complement of one.
        # The former subtraction implementation let the H-rank orthogonal
        # complement pass unchanged, so capacity=0 still retained almost the
        # complete host residual while reporting effective_basis_mass=0.
        # Projecting the transparent coordinates is both non-expansive and
        # algebraically zero when no basis group is active.
        contracted = torch.einsum(
            "btr,hr->bth",
            coordinates * transparency[:, None],
            basis,
        )
        contracted = contracted.to(dtype=update.dtype)
        if not collect_diagnostics:
            return contracted, {}
        input_rms = update.detach().float().square().mean(dim=(1, 2)).sqrt()
        output_rms = contracted.detach().float().square().mean(dim=(1, 2)).sqrt()
        return contracted, {
            "capacity_ratio": capacity.detach().float().mean(),
            "effective_basis_mass": transparency.detach().float().sum(dim=-1).mean(),
            "contraction_ratio": (output_rms / input_rms.clamp_min(1e-8)).mean(),
            "nonexpansive_violation": torch.relu(
                output_rms / input_rms.clamp_min(1e-8) - 1.0
            ).amax(),
        }


class ExecutionController(nn.Module):
    """Selector-only control of capacity and soft continuation.

    The controller is queried once per ODE step and never creates alternative
    action graphs.  Its values come from the typed evidence bank; noisy action
    is appended only to the selector lane with literal-zero values.
    """

    def __init__(
        self,
        *,
        hidden: int,
        heads: int,
        tokens: int,
        depth: int,
        blocks: int,
        ffn_expansion: float,
        capacity_logit_init: float,
        exit_probability_init: float,
        normalization_floor: float,
    ) -> None:
        super().__init__()
        if hidden % heads or min(tokens, depth, blocks) <= 0:
            raise ValueError("execution controller dimensions are invalid")
        self.hidden = int(hidden)
        self.heads = int(heads)
        self.tokens = int(tokens)
        self.blocks = int(blocks)
        self.normalization_floor = float(normalization_floor)
        self.control = nn.Parameter(torch.randn(1, tokens, hidden) * 0.02)
        self.address = nn.Parameter(torch.randn(1, tokens, hidden) * 0.02)
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.value = nn.Linear(hidden, hidden, bias=False)
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.gru = nn.ModuleList(nn.GRUCell(hidden, hidden) for _ in range(depth))
        self.ffn = nn.ModuleList(BiasFreeFFN(hidden, ffn_expansion) for _ in range(depth))
        self.block_query = nn.Parameter(torch.randn(1, blocks, hidden) * 0.02)
        self.block_attention = nn.MultiheadAttention(
            hidden, heads, bias=False, dropout=0.0, batch_first=True
        )
        self.capacity = nn.Linear(hidden, 1)
        self.continue_head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.capacity.weight)
        nn.init.constant_(self.capacity.bias, float(capacity_logit_init))
        nn.init.zeros_(self.continue_head.weight)
        nn.init.constant_(self.continue_head.bias, _logit(1.0 - exit_probability_init))
        for recurrent in self.gru:
            if not isinstance(recurrent, nn.GRUCell):
                raise TypeError("execution recurrence must contain GRUCell modules")
            if recurrent.bias_ih is None or recurrent.bias_hh is None:
                raise TypeError("execution recurrence requires biased GRUCell modules")
            nn.init.zeros_(recurrent.bias_ih)
            nn.init.zeros_(recurrent.bias_hh)

    def _split(self, value: Tensor) -> Tensor:
        batch, tokens, hidden = value.shape
        return value.reshape(batch, tokens, self.heads, hidden // self.heads).transpose(1, 2)

    def forward(
        self,
        *,
        evidence: TypedEvidenceBank,
        action: Tensor,
        condition: Tensor,
        collect_diagnostics: bool,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        batch = int(action.shape[0])
        state = (self.control + self.address).to(device=action.device, dtype=action.dtype)
        state = state.expand(batch, -1, -1)
        # Append the raw selector representation.  All selectors are projected
        # exactly once below; pre-projecting action here used the same key
        # matrix twice and gave noisy action a different score scale.
        action_selector = action
        selector = torch.cat((evidence.selector, action_selector, condition[:, None]), dim=1)
        value = torch.cat(
            (
                evidence.value,
                torch.zeros_like(action_selector),
                condition[:, None],
            ),
            dim=1,
        )
        key = self._split(self.key(self.norm(selector)))
        value_heads = self._split(self.value(_floored_unit(value, self.normalization_floor)))
        ownership: Tensor | None = None
        for recurrent, ffn in zip(self.gru, self.ffn, strict=True):
            query = self._split(self.query(self.norm(state + self.address)))
            logits = torch.matmul(query.float(), key.float().transpose(-2, -1))
            logits = logits / math.sqrt(float(query.shape[-1]))
            dispatch = torch.softmax(logits, dim=-1)
            ownership = torch.softmax(logits, dim=-2)
            probability = dispatch * ownership
            probability = probability / probability.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            read = torch.matmul(probability.to(dtype=value_heads.dtype), value_heads)
            read = read.transpose(1, 2).reshape(batch, self.tokens, self.hidden)
            state = recurrent(
                read.reshape(batch * self.tokens, self.hidden),
                state.reshape(batch * self.tokens, self.hidden),
            ).reshape(batch, self.tokens, self.hidden)
            state = state + 0.10 * ffn(self.norm(state))
            state = state - state.mean(dim=1, keepdim=True)
        block_query = self.block_query.to(device=action.device, dtype=action.dtype).expand(
            batch, -1, -1
        )
        block_state, _ = self.block_attention(
            self.norm(block_query), self.norm(state), self.norm(state), need_weights=False
        )
        block_state = block_query + block_state + condition[:, None]
        capacity = torch.sigmoid(self.capacity(block_state).squeeze(-1))
        local_continue = torch.sigmoid(self.continue_head(block_state).squeeze(-1))
        # A later host block cannot execute after an earlier exit.  The head
        # predicts local stay probabilities and the exported continuation is
        # the monotone soft survival probability for each depth.
        continuation = torch.cumprod(local_continue, dim=-1)
        if not collect_diagnostics:
            return capacity, continuation, {}
        metrics = {
            "bottom_capacity_mean": capacity.detach().float().mean(),
            "bottom_capacity_block_std": capacity.detach()
            .float()
            .std(dim=-1, unbiased=False)
            .mean(),
            "bottom_local_continue_mean": local_continue.detach().float().mean(),
            "bottom_continue_mean": continuation.detach().float().mean(),
            "bottom_continue_block_std": continuation.detach()
            .float()
            .std(dim=-1, unbiased=False)
            .mean(),
            "bottom_expected_depth": continuation.detach().float().sum(dim=-1).mean(),
            "bottom_execution_cost_audit": (capacity * continuation)
            .detach()
            .float()
            .sum(dim=-1)
            .mean(),
        }
        if ownership is not None:
            common = state.mean(dim=1, keepdim=True)
            total = state.detach().float().square().mean().clamp_min(1e-8)
            metrics["bottom_controller_common_ratio"] = (
                common.detach().float().square().mean() / total
            )
            metrics["bottom_controller_private_ratio"] = (
                state - common
            ).detach().float().square().mean() / total
            metrics["bottom_controller_ownership_max"] = (
                ownership.detach().float().amax(dim=-2).mean()
            )
        return capacity, continuation, metrics


@dataclass(frozen=True)
class BottomOutput:
    physical_velocity: Tensor
    event_logits: Tensor
    motion_logits: Tensor
    action_query: Tensor
    block_updates: tuple[Tensor, ...]
    evidence_tokens: Tensor

    def validate(self, *, action_dim: int, horizon: int, basis: int, hidden: int) -> None:
        batch = int(self.physical_velocity.shape[0])
        if tuple(self.physical_velocity.shape) != (batch, horizon, action_dim):
            raise ValueError("bottom physical velocity has an invalid shape")
        if tuple(self.event_logits.shape) != (batch, horizon, 3):
            raise ValueError("bottom event logits have an invalid shape")
        if tuple(self.motion_logits.shape) != (batch, horizon):
            raise ValueError("bottom motion logits have an invalid shape")
        if tuple(self.action_query.shape) != (batch, horizon, basis, hidden):
            raise ValueError("bottom action query lost its basis axis")


class EvidenceMMDiTBottom(nn.Module):
    """Three-block typed Evidence MMDiT with bounded adaptive execution."""

    def __init__(self, config: ExperimentConfig, *, physical_action_dim: int) -> None:
        super().__init__()
        config.validate()
        dims = config.dimensions
        bottom = config.bottom
        self.hidden = dims.hidden_size
        self.horizon = dims.action_horizon
        self.basis = dims.action_basis_tokens
        self.native_action_dim = dims.action_dim
        self.physical_action_dim = int(physical_action_dim)
        expected_physical_dim = 2 * (dims.action_dim - 1) + bottom.gripper_field_dim
        if self.physical_action_dim != expected_physical_dim:
            raise ValueError("bottom physical action width does not match the codec")
        self.query_encoder = ActionQueryEncoder(
            action_dim=self.physical_action_dim,
            hidden=dims.hidden_size,
            horizon=dims.action_horizon,
            basis=dims.action_basis_tokens,
        )
        self.protected_reader = ProtectedConsequenceReader(
            hidden=dims.hidden_size,
            basis=dims.action_basis_tokens,
        )
        self.evidence_compiler = TypedEvidenceCompiler(
            hidden=dims.hidden_size,
            state_dim=dims.state_dim,
            action_dim=dims.action_dim,
            horizon=dims.action_horizon,
            basis=dims.action_basis_tokens,
        )
        self.organizer = BottomConditionOrganizer(
            hidden=dims.hidden_size,
            latent=bottom.latent_dim,
            state_dim=dims.state_dim,
            action_dim=dims.action_dim,
        )
        self.blocks = nn.ModuleList(
            ReadOnlyEvidenceMMDiTBlock(
                hidden=dims.hidden_size,
                heads=dims.num_heads,
                expansion=bottom.ffn_expansion,
                dropout=bottom.dropout,
                residual_scale_max=bottom.residual_scale_max,
                residual_scale_init=bottom.residual_scale_init,
                normalization_floor=bottom.normalization_floor,
            )
            for _ in range(bottom.evidence_depth)
        )
        self.capacity = nn.ModuleList(
            NestedCapacityOperator(
                hidden=dims.hidden_size,
                rank=bottom.operator_rank,
                groups=bottom.operator_groups,
            )
            for _ in range(bottom.evidence_depth)
        )
        self.execution = ExecutionController(
            hidden=dims.hidden_size,
            heads=bottom.controller_heads,
            tokens=bottom.controller_tokens,
            depth=bottom.controller_depth,
            blocks=bottom.evidence_depth,
            ffn_expansion=bottom.ffn_expansion,
            capacity_logit_init=bottom.operator_depth_logit_init,
            exit_probability_init=bottom.initial_exit_probability,
            normalization_floor=bottom.normalization_floor,
        )
        self.final_norm = nn.LayerNorm(dims.hidden_size)
        self.velocity_head = nn.Linear(dims.hidden_size, self.physical_action_dim)
        self.event_head = nn.Linear(dims.hidden_size, 3)
        self.motion_head = nn.Linear(dims.hidden_size, 1)
        nn.init.normal_(self.velocity_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.velocity_head.bias)
        nn.init.zeros_(self.event_head.weight)
        nn.init.zeros_(self.event_head.bias)
        nn.init.zeros_(self.motion_head.weight)
        nn.init.zeros_(self.motion_head.bias)

    def action_query(self, noisy_action_field: Tensor, time: Tensor) -> Tensor:
        return self.query_encoder(noisy_action_field, time)

    def forward(
        self,
        *,
        noisy_action_field: Tensor,
        time: Tensor,
        action_query: Tensor,
        plan: ObjectPolicyPlanDeltaBank,
        history: ObservableHistory,
        transition: ControlledTransitionState,
        collect_diagnostics: bool = False,
    ) -> tuple[BottomOutput, dict[str, Tensor]]:
        expected_query = (
            int(noisy_action_field.shape[0]),
            self.horizon,
            self.basis,
            self.hidden,
        )
        if tuple(action_query.shape) != expected_query:
            raise ValueError("bottom and P2/P3 must use the same action query")
        plan.validate()
        evidence = self.evidence_compiler(plan, history, transition)
        protected, protected_metrics = self.protected_reader(
            action_query,
            plan.protected_base,
            collect_diagnostics=collect_diagnostics,
        )
        condition, organizer_metrics = self.organizer(
            history,
            time,
            collect_diagnostics=collect_diagnostics,
        )
        # The noisy query and the one protected consequence are the only base
        # writers.  Optional lanes remain in the read-only evidence V stream.
        action = action_query.mean(dim=2) + protected
        capacity, continuation, execution_metrics = self.execution(
            evidence=evidence,
            action=action,
            condition=condition,
            collect_diagnostics=collect_diagnostics,
        )
        updates: list[Tensor] = []
        block_metrics: dict[str, Tensor] = {}
        for index, (block, operator) in enumerate(zip(self.blocks, self.capacity, strict=True)):
            proposed, local_metrics = block(
                action,
                evidence,
                condition,
                collect_diagnostics=collect_diagnostics,
            )
            contracted, capacity_metrics = operator(
                proposed,
                capacity[:, index],
                collect_diagnostics=collect_diagnostics,
            )
            update = continuation[:, index, None, None].to(dtype=contracted.dtype) * contracted
            action = action + update
            updates.append(update)
            if collect_diagnostics:
                for name, value in local_metrics.items():
                    block_metrics[f"bottom_block_{index + 1}_{name}"] = value
                for name, value in capacity_metrics.items():
                    block_metrics[f"bottom_block_{index + 1}_{name}"] = value
                block_metrics[f"bottom_block_{index + 1}_executed_update_rms"] = (
                    update.detach().float().square().mean().sqrt()
                )
        action = self.final_norm(action)
        output = BottomOutput(
            physical_velocity=self.velocity_head(action),
            event_logits=self.event_head(action),
            motion_logits=self.motion_head(action).squeeze(-1),
            action_query=action_query,
            block_updates=tuple(updates),
            evidence_tokens=evidence.value,
        )
        output.validate(
            action_dim=self.physical_action_dim,
            horizon=self.horizon,
            basis=self.basis,
            hidden=self.hidden,
        )
        if not collect_diagnostics:
            return output, {}
        optional_rms: dict[str, Tensor] = {}
        for name in TypedEvidenceCompiler.VALUE_LANE_NAMES:
            start, stop = evidence.lane_ranges[name]
            optional_rms[f"bottom_{name}_value_rms"] = (
                evidence.value[:, start:stop].detach().float().square().mean().sqrt()
            )
        metrics = {
            **protected_metrics,
            **organizer_metrics,
            **execution_metrics,
            **block_metrics,
            **optional_rms,
            "bottom_evidence_value_rms": evidence.value.detach().float().square().mean().sqrt(),
            "bottom_action_rms": action.detach().float().square().mean().sqrt(),
        }
        return output, metrics


__all__ = [
    "ActionQueryEncoder",
    "BottomOutput",
    "EvidenceMMDiTBottom",
    "ReadOnlyEvidenceMMDiTBlock",
    "TypedEvidenceBank",
    "canonical_state_history",
]
