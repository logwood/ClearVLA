from __future__ import annotations

"""Recurrent multi-token control plane for hierarchical action refinement."""

from dataclasses import dataclass
import math
from typing import Protocol

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .gauges import fp32_diagnostic
from .primitives import BiasFreeFFN


class UnifiedControllerConfig(Protocol):
    hidden_size: int
    num_heads: int
    dropout: float
    hierarchical_mmdit_operator_stages: int
    hierarchical_mmdit_operator_depth_logit_init: float
    hierarchical_mmdit_exit_logit_init: float
    hierarchical_mmdit_control_tokens: int
    hierarchical_mmdit_controller_depth: int
    hierarchical_mmdit_controller_heads: int
    hierarchical_mmdit_controller_ffn_expansion: float


@dataclass
class ControllerMemory:
    """Controller memory with separate addressing and evidence content.

    ``address`` is allowed to affect Q/K retrieval only.  ``content`` is the
    value stream consumed by workspace and operation readers, so a controller
    cannot manufacture evidence by writing an address signal into V.
    """

    content: Tensor
    address: Tensor


@dataclass
class UnifiedControllerOutput:
    state: Tensor
    state_address: Tensor
    memory: ControllerMemory
    operator_logits: Tensor
    operator_update_logits: Tensor
    operator_depth_logits: Tensor
    exit_logit: Tensor
    metrics: dict[str, Tensor]


class _RecurrentControllerBlock(nn.Module):
    """Pre-norm cross/self-attention block shared across refinement steps."""

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
        self.cross_attn = nn.MultiheadAttention(
            h, heads, dropout=dropout, batch_first=True
        )
        self.self_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(
            h, heads, dropout=dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.ffn = BiasFreeFFN(
            h, float(config.hierarchical_mmdit_controller_ffn_expansion)
        )
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
        key_bias = None if bias is None else bias[hidden:2 * hidden]
        value_bias = None if bias is None else bias[2 * hidden:]
        projected_query = F.linear(query, projection[:hidden], query_bias)
        projected_key = F.linear(
            key, projection[hidden:2 * hidden], key_bias
        )
        projected_value = F.linear(value, projection[2 * hidden:], value_bias)

        def split_heads(x: Tensor) -> Tensor:
            batch, tokens, width = x.shape
            return x.reshape(
                batch, tokens, self.heads, width // self.heads
            ).transpose(1, 2)

        query_heads = split_heads(projected_query)
        key_heads = split_heads(projected_key)
        value_heads = split_heads(projected_value)
        logits = torch.matmul(
            query_heads.float(), key_heads.float().transpose(-2, -1)
        ) * (float(query_heads.shape[-1]) ** -0.5)

        # A source first chooses its owning control slots. The source-count
        # prior is applied afterwards because a source-only bias would cancel
        # inside the slot softmax.
        ownership = torch.softmax(logits, dim=-2)
        prior_logits = source_key_bias.detach().float()
        source_prior = torch.exp(prior_logits - prior_logits.max())
        weighted = (ownership + 1e-8) * source_prior[None, None, None]
        weights = weighted / weighted.sum(dim=-1, keepdim=True).clamp_min(1e-8)
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
        state: Tensor,
        state_address: Tensor,
        source_key: Tensor,
        source_value: Tensor,
        source_key_bias: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        query = self.cross_norm(state) + state_address
        key = self.source_key_norm(source_key)
        value = self.source_value_norm(source_value)
        if tuple(source_key_bias.shape) != (int(source_key.shape[1]),):
            raise ValueError("controller source key bias has the wrong shape")
        cross, weights, ownership = self._competitive_cross_attention(
            query,
            key,
            value,
            source_key_bias.to(device=query.device),
        )
        state = state + self.drop(cross)
        self_value = self.self_norm(state)
        self_query = self_value + state_address
        self_update, _ = self.self_attn(
            self_query, self_query, self_value, need_weights=False
        )
        state = state + self.drop(self_update)
        state = state + self.drop(self.ffn(self.ffn_norm(state)))
        return state, weights, ownership


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
        self.control_count = int(config.hierarchical_mmdit_control_tokens)
        self.operator_count = int(config.hierarchical_mmdit_operator_stages)
        self.operator_branch_count = int(operator_branch_count)
        if self.operator_branch_count < 1:
            raise ValueError("controller operator_branch_count must be positive")
        self.control_seed = nn.Parameter(
            torch.randn(1, self.control_count, h) * 0.02
        )
        self.source_type = nn.Parameter(
            torch.randn(1, len(self.SOURCE_NAMES), h) * 0.02
        )
        self.source_norms = nn.ModuleList([
            nn.LayerNorm(h, elementwise_affine=False)
            for _ in self.SOURCE_NAMES
        ])
        self.feedback_lift = nn.Linear(1, h, bias=False)
        self.source_adapters = nn.ModuleList([
            nn.Linear(h, h) for _ in self.SOURCE_NAMES
        ])
        for adapter in self.source_adapters:
            nn.init.eye_(adapter.weight)
            nn.init.zeros_(adapter.bias)
        self.control_address = nn.Parameter(
            torch.randn(1, self.control_count, h) * 0.02
        )
        self.blocks = nn.ModuleList([
            _RecurrentControllerBlock(config)
            for _ in range(int(config.hierarchical_mmdit_controller_depth))
        ])
        self.final_norm = nn.LayerNorm(h, elementwise_affine=False)

        query_counts = (self.operator_count, 1)
        self.query_counts = query_counts
        self.output_query = nn.Parameter(
            torch.randn(1, sum(query_counts), h) * 0.02
        )
        self.output_query_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.output_state_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.output_attn = nn.MultiheadAttention(
            h, heads, dropout=0.0, batch_first=True
        )
        # Keep stage/operator and exit decisions at token bandwidth.  A
        # token-level coupling pass lets every execution query inspect every
        # other query without collapsing them through a shared mean vector.
        self.output_coupling_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.output_coupling_attn = nn.MultiheadAttention(
            h, heads, dropout=0.0, batch_first=True
        )
        self.output_ffn_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.output_ffn = BiasFreeFFN(
            h, float(config.hierarchical_mmdit_controller_ffn_expansion)
        )
        # One route score plus update/depth controls for every branch.  The
        # controls share a stage query, but they are not compressed into one
        # scalar strength and retain distinct execution semantics.
        self.operator_head = nn.Linear(
            h, 1 + 2 * self.operator_branch_count
        )
        self.exit_head = nn.Linear(h, 1)
        for head in (
            self.operator_head,
            self.exit_head,
        ):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        with torch.no_grad():
            self.exit_head.bias.fill_(
                float(config.hierarchical_mmdit_exit_logit_init)
            )
            self.operator_head.bias[1:1 + self.operator_branch_count].fill_(
                float(config.hierarchical_mmdit_operator_depth_logit_init)
            )
            self.operator_head.bias[1 + self.operator_branch_count:].fill_(
                float(config.hierarchical_mmdit_operator_depth_logit_init)
            )

    def parameter_groups(self) -> dict[str, tuple[nn.Parameter, ...]]:
        heads = {
            "operator": tuple(self.operator_head.parameters()),
            "exit": tuple(self.exit_head.parameters()),
        }
        head_ids = {id(parameter) for values in heads.values() for parameter in values}
        heads["backbone"] = tuple(
            parameter for parameter in self.parameters()
            if id(parameter) not in head_ids
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
        address = self.source_type[:, source_index:source_index + 1].to(
            device=value.device, dtype=value.dtype
        )
        return content, content + address

    def _feedback_tokens(self, feedback: Tensor) -> Tensor:
        if feedback.ndim != 2:
            raise ValueError("controller feedback must be [B,F]")
        count = int(feedback.shape[1])
        lifted = self.feedback_lift(feedback.detach().float()[..., None])
        half = self.hidden_size // 2
        position_index = torch.arange(
            1, count + 1, device=feedback.device, dtype=torch.float32
        )[:, None]
        frequency = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=feedback.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        position = torch.cat([
            torch.sin(position_index * frequency),
            torch.cos(position_index * frequency),
        ], dim=-1)
        if int(position.shape[-1]) < self.hidden_size:
            position = F.pad(position, (0, self.hidden_size - int(position.shape[-1])))
        position = position[None, :, :self.hidden_size].to(dtype=lifted.dtype)
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
            effective_rank = eig.sum(dim=-1).square() / eig.square().sum(dim=-1).clamp_min(1e-8)
            centered = state_fp32 - state_fp32.mean(dim=1, keepdim=True)
            return {
                "controller_state_norm": state_fp32.norm(dim=-1).mean(),
                "controller_state_slot_diversity": centered.norm(dim=-1).mean(),
                "controller_state_pair_cosine": off_diagonal.mean(),
                "controller_state_effective_rank": effective_rank.mean(),
            }

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
        evidence_role_metric_ranges: dict[
            str, tuple[tuple[int, int], ...]
        ] = {}
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
                        if any(covered[int(start):int(stop)]):
                            raise ValueError(
                                "controller evidence role ranges overlap"
                            )
                        source_bias[int(start):int(stop)] = -math.log(
                            float(subgroup_count * count)
                        )
                        covered[int(start):int(stop)] = [True] * (
                            int(stop) - int(start)
                        )
                        metric_parts.append((offset + int(start), offset + int(stop)))
                    evidence_role_metric_ranges[role_name] = tuple(metric_parts)
                if not all(covered):
                    raise ValueError(
                        "controller evidence role ranges do not cover every evidence token"
                    )
                key_bias_parts.append(source_bias)
            else:
                key_bias_parts.append(torch.full(
                    (int(source.shape[1]),),
                    -math.log(float(max(int(source.shape[1]), 1))),
                    device=source.device,
                    dtype=torch.float32,
                ))
            ranges[name] = (offset, offset + int(source.shape[1]))
            if name == "evidence" and evidence_ranges:
                for evidence_name, (start, stop) in evidence_ranges.items():
                    if not 0 <= int(start) < int(stop) <= int(source.shape[1]):
                        raise ValueError(
                            "controller evidence source range is outside evidence tokens"
                        )
                    evidence_metric_ranges[evidence_name] = (
                        offset + int(start), offset + int(stop)
                    )
            offset += int(source.shape[1])
        source_values = torch.cat(typed_values, dim=1)
        source_keys = torch.cat(typed_keys, dim=1)
        source_key_bias = torch.cat(key_bias_parts, dim=0)

        seed = self.control_seed.to(
            device=action_tokens.device, dtype=action_tokens.dtype
        ).expand(batch, -1, -1)
        state_address = self.control_address.to(
            device=action_tokens.device, dtype=action_tokens.dtype
        ).expand(batch, -1, -1)
        if previous_state is None:
            state = seed
            recurrence_change = torch.zeros(
                (), device=action_tokens.device, dtype=torch.float32
            )
        else:
            if tuple(previous_state.shape) != tuple(seed.shape):
                raise ValueError(
                    f"controller recurrent state must be {tuple(seed.shape)}, "
                    f"got {tuple(previous_state.shape)}"
                )
            state = previous_state
            recurrence_change = (
                1.0 - F.cosine_similarity(
                    previous_state.detach().float(), seed.detach().float(), dim=-1
                )
            ).mean()

        attention_rows: list[Tensor] = []
        ownership_rows: list[Tensor] = []
        state_before = state
        for block in self.blocks:
            state, attention, ownership = block(
                state,
                state_address,
                source_keys,
                source_values,
                source_key_bias,
            )
            attention_rows.append(attention.detach().float())
            ownership_rows.append(ownership.detach().float())
        state = self.final_norm(state)
        if previous_state is not None:
            recurrence_change = (
                1.0 - F.cosine_similarity(
                    state.detach().float(), state_before.detach().float(), dim=-1
                )
            ).mean()

        queries = self.output_query.to(
            device=state.device, dtype=state.dtype
        ).expand(batch, -1, -1)
        readout, _ = self.output_attn(
            self.output_query_norm(queries),
            self.output_state_norm(state + state_address),
            self.output_state_norm(state),
            need_weights=False,
        )
        readout = queries + readout
        coupled = self.output_coupling_norm(readout)
        coupled, _ = self.output_coupling_attn(
            coupled, coupled, coupled, need_weights=False
        )
        readout = readout + coupled
        readout = readout + self.output_ffn(self.output_ffn_norm(readout))
        operator, exit_query = torch.split(
            readout, self.query_counts, dim=1
        )
        operator_output = self.operator_head(operator)
        operator_logits = operator_output[..., 0]
        update_stop = 1 + self.operator_branch_count
        operator_update_logits = operator_output[..., 1:update_stop]
        operator_depth_logits = operator_output[..., update_stop:]
        exit_logit = self.exit_head(exit_query[:, 0]).squeeze(-1)

        metrics = self._state_metrics(state)
        metrics["controller_recurrent_change"] = recurrence_change.detach().float()
        metrics["controller_operator_logit_rms"] = (
            operator_logits.detach().float().square().mean().sqrt()
        )
        raw_update_keep = torch.sigmoid(
            operator_update_logits.detach().float()
        )
        raw_depth_keep = torch.sigmoid(
            operator_depth_logits.detach().float()
        )
        continue_keep = torch.sigmoid(-exit_logit.detach().float())
        metrics["controller_operator_raw_update_mean"] = raw_update_keep.mean()
        metrics["controller_operator_raw_depth_mean"] = torch.sigmoid(
            operator_depth_logits.detach().float()
        ).mean()
        metrics["controller_operator_update_stage_std"] = (
            raw_update_keep.mean(dim=-1).std(dim=-1, unbiased=False).mean()
        )
        metrics["controller_operator_depth_stage_std"] = (
            raw_depth_keep.mean(dim=-1).std(dim=-1, unbiased=False).mean()
        )
        update_centered = raw_update_keep - raw_update_keep.mean(
            dim=(1, 2), keepdim=True
        )
        depth_centered = raw_depth_keep - raw_depth_keep.mean(
            dim=(1, 2), keepdim=True
        )
        metrics["controller_update_depth_correlation"] = (
            (update_centered * depth_centered).sum(dim=(1, 2))
            / (
                update_centered.square().sum(dim=(1, 2)).sqrt()
                * depth_centered.square().sum(dim=(1, 2)).sqrt()
            ).clamp_min(1e-8)
        ).mean()
        metrics["controller_continue_keep_mean"] = continue_keep.mean()
        metrics["controller_joint_suppression_mass"] = (
            (1.0 - raw_update_keep)
            * (1.0 - raw_depth_keep)
            * (1.0 - continue_keep)[:, None, None]
        ).mean()
        metrics["controller_exit_logit_rms"] = (
            exit_logit.detach().float().square().mean().sqrt()
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
                metrics[f"controller_evidence_role_{name}_attention"] = torch.stack([
                    attention[:, start:stop].sum(dim=-1).mean()
                    for start, stop in role_parts
                ]).sum()
            metrics["controller_source_attention_mass_error"] = (
                attention.sum(dim=-1) - 1.0
            ).abs().mean()
        if ownership_rows:
            ownership = torch.stack(ownership_rows).mean(dim=0)
            source_entropy = -(
                ownership.clamp_min(1e-8) * ownership.clamp_min(1e-8).log()
            ).sum(dim=-2)
            slot_load = ownership.mean(dim=(1, 3))
            slot_load = slot_load / slot_load.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
            slot_load_entropy = -(
                slot_load.clamp_min(1e-8) * slot_load.clamp_min(1e-8).log()
            ).sum(dim=-1)
            metrics.update({
                "controller_competition_source_effective_slots": (
                    torch.exp(source_entropy).mean()
                ),
                "controller_competition_source_owner_max": (
                    ownership.amax(dim=-2).mean()
                ),
                "controller_competition_slot_load_effective": (
                    torch.exp(slot_load_entropy).mean()
                ),
                "controller_competition_slot_load_max": (
                    slot_load.amax(dim=-1).mean()
                ),
            })
        return UnifiedControllerOutput(
            state=state,
            state_address=state_address,
            memory=ControllerMemory(content=state, address=state_address),
            operator_logits=operator_logits,
            operator_update_logits=operator_update_logits,
            operator_depth_logits=operator_depth_logits,
            exit_logit=exit_logit,
            metrics=metrics,
        )
