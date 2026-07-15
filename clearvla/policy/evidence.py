from __future__ import annotations

"""Typed evidence storage, retrieval, and stage-memory control."""

from dataclasses import dataclass
import math
from typing import Protocol

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .primitives import BiasFreeFFN


class PolicyEvidenceConfig(Protocol):
    hidden_size: int
    num_heads: int
    dropout: float
    action_horizon: int
    adaptive_cvae_refine_steps: int
    latent_cvae_horizon_tokens: int
    latent_cvae_stage_slots: int
    latent_cvae_ffn_expansion: float
    latent_cvae_causal_attention: int
    latent_cvae_stage_promote_scale_init: float
    hierarchical_mmdit_unified_controller: int
    hierarchical_mmdit_controller_heads: int
    hierarchical_mmdit_controller_ffn_expansion: float


@dataclass
class PreparedEvidenceMemory:
    tokens: Tensor
    key_bias: Tensor
    ranges: dict[str, tuple[int, int]]
    role_ranges: dict[str, tuple[tuple[int, int], ...]]
    block_kv: tuple[tuple[Tensor, Tensor], ...]
    batch_size: int


@dataclass
class WorkspaceControlOverride:
    """Read-only multi-token state supplied by the unified controller.

    Workspace-owned interface attention converts this state into selector
    controls.  The controller state itself never enters an evidence value
    stream.
    """

    control_tokens: Tensor
    control_addresses: Tensor | None = None


@dataclass
class WorkspaceControllerInterfaceOutput:
    low_query_delta: Tensor
    stage_query_delta: Tensor
    promote_gate: Tensor
    low_control_attention: Tensor
    metrics: dict[str, Tensor]


@dataclass
class StagePromotionOutput:
    next_content: Tensor
    attention_weights: Tensor
    projected: Tensor
    normalized: Tensor
    gated: Tensor
    gate_rows: Tensor
    retain: Tensor
    residual_scale: Tensor


class EvidenceMemoryBank(nn.Module):
    """Owns typed evidence storage and static K/V preparation for workspace reads."""

    SOURCE_NAMES = (
        "layer",
        "scan",
        "lateral",
        "transition",
        "transition_delta",
        "transition_effect",
        "transition_timeline",
        "context",
        "visual",
        "trajectory",
        "rollout",
        "capsule",
        "progress",
        "routed_layer",
    )
    SOURCE_ROLES = {
        "layer": "layer",
        "routed_layer": "layer",
        "trajectory": "geom",
        "rollout": "geom",
        "transition": "transition",
        "transition_delta": "transition",
        "transition_effect": "transition",
        "transition_timeline": "transition",
        "progress": "state",
        "capsule": "state",
        "context": "state",
        "visual": "state",
        "scan": "global",
        "lateral": "global",
    }
    # NOTE (legacy-path gauge caveat): on this legacy bank no source maps to
    # the "event" role -- it is a reserved seat whose real owner exists only
    # in OwnedEvidenceMemoryBank (the hierarchical/v76 line).  Console `wevt`
    # therefore reads 0 on B0/B1-style arms BY CONSTRUCTION; read the
    # transition-family share via the "transition" role instead.  Full
    # taxonomy cleanup is CR9 scope (do_before_v76 §11).
    ROLE_NAMES = ("geom", "transition", "event", "state", "layer", "global")

    def __init__(self, config: PolicyEvidenceConfig) -> None:
        super().__init__()
        h = int(config.hidden_size)
        self.hidden_size = h
        self.type_embed = nn.Parameter(torch.randn(1, len(self.SOURCE_NAMES), h) * 0.02)
        self.source_norm = nn.LayerNorm(h, elementwise_affine=False)

    def _source_role(self, name: str) -> str:
        return self.SOURCE_ROLES.get(name, "global")

    def prepare_sources(
        self,
        sources: dict[str, Tensor],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        allow_empty: bool,
    ) -> tuple[Tensor | None, Tensor, dict[str, tuple[int, int]]]:
        unknown_sources = set(sources).difference(self.SOURCE_NAMES)
        if unknown_sources:
            raise ValueError(f"unknown evidence workspace sources: {sorted(unknown_sources)}")
        parts: list[Tensor] = []
        ranges: dict[str, tuple[int, int]] = {}
        key_bias_parts: list[Tensor] = []
        offset = 0
        for index, name in enumerate(self.SOURCE_NAMES):
            value = sources.get(name)
            if value is None:
                continue
            if value.ndim != 3 or int(value.shape[-1]) != self.hidden_size:
                raise ValueError(f"workspace source {name!r} must be [B,N,H], got {tuple(value.shape)}")
            if int(value.shape[0]) != batch_size:
                raise ValueError(
                    f"workspace source {name!r} batch={int(value.shape[0])} "
                    f"does not match action batch={batch_size}"
                )
            value = value.to(device=device, dtype=dtype)
            if int(value.shape[1]) == 0:
                continue
            typed = self.source_norm(value) + self.type_embed[:, index:index + 1].to(device=device, dtype=dtype)
            parts.append(typed)
            count = int(typed.shape[1])
            ranges[name] = (offset, offset + count)
            key_bias_parts.append(torch.full((count,), -math.log(float(count)), device=device, dtype=torch.float32))
            offset += count
        if not parts:
            if allow_empty:
                return None, torch.zeros(0, device=device, dtype=torch.float32), ranges
            raise RuntimeError("evidence memory bank requires at least one semantic source")
        return torch.cat(parts, dim=1), torch.cat(key_bias_parts, dim=0), ranges

    def prepare_static_memory(
        self,
        sources: dict[str, Tensor],
        *,
        blocks: nn.ModuleList,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> PreparedEvidenceMemory:
        memory, key_bias, ranges = self.prepare_sources(
            sources,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            allow_empty=False,
        )
        assert memory is not None
        role_ranges = {
            role: tuple(
                (start, stop)
                for name, (start, stop) in ranges.items()
                if self._source_role(name) == role
            )
            for role in self.ROLE_NAMES
        }
        role_ranges = {
            role: value for role, value in role_ranges.items() if value
        }
        return PreparedEvidenceMemory(
            tokens=memory,
            key_bias=key_bias,
            ranges=ranges,
            role_ranges=role_ranges,
            block_kv=tuple(block.project_memory(memory) for block in blocks),
            batch_size=batch_size,
        )

    def role_token_counts(self, ranges: dict[str, tuple[int, int]]) -> dict[str, int]:
        counts = {role: 0 for role in self.ROLE_NAMES}
        for name, (start, stop) in ranges.items():
            role = self._source_role(name)
            counts[role] = counts.get(role, 0) + max(int(stop) - int(start), 0)
        return counts

    def role_attention_metrics(
        self,
        weights: Tensor,
        ranges: dict[str, tuple[int, int]],
    ) -> dict[str, Tensor]:
        metrics: dict[str, Tensor] = {}
        device = weights.device
        dtype = weights.dtype
        for role in self.ROLE_NAMES:
            role_ranges = [
                (start, stop)
                for name, (start, stop) in ranges.items()
                if self._source_role(name) == role
            ]
            if role_ranges:
                parts = [weights[..., start:stop].sum(dim=-1).mean() for start, stop in role_ranges]
                metrics[f"workspace_role_{role}_attention"] = torch.stack(parts).sum()
                token_count = sum(max(int(stop) - int(start), 0) for start, stop in role_ranges)
            else:
                metrics[f"workspace_role_{role}_attention"] = torch.zeros((), device=device, dtype=dtype)
                token_count = 0
            metrics[f"workspace_role_{role}_token_count"] = torch.tensor(
                float(token_count),
                device=device,
                dtype=torch.float32,
            )
        return metrics

    def role_key_bias(
        self,
        role_logits: Tensor,
        ranges: dict[str, tuple[int, int]],
    ) -> Tensor:
        batch = int(role_logits.shape[0])
        total_tokens = max((stop for _, stop in ranges.values()), default=0)
        if total_tokens <= 0:
            return role_logits.new_zeros(batch, 0)
        role_to_index = {role: i for i, role in enumerate(self.ROLE_NAMES)}
        bias = role_logits.new_zeros(batch, total_tokens)
        for name, (start, stop) in ranges.items():
            role = self._source_role(name)
            index = role_to_index.get(role)
            if index is None:
                continue
            bias[:, start:stop] = role_logits[:, index:index + 1]
        return bias


class OwnedEvidenceMemoryBank(EvidenceMemoryBank):
    """Strict five-role evidence store for the deterministic intent decoder.

    Global summaries belong to the intent compiler and are intentionally not
    valid values here.  Event and state are first-class sources rather than
    semantics hidden inside a lateral/global summary.
    """

    SOURCE_NAMES = (
        "layer",
        "trajectory",
        "rollout",
        "transition",
        "event",
        "state",
    )
    SOURCE_ROLES = {
        "layer": "layer",
        "trajectory": "geom",
        "rollout": "geom",
        "transition": "transition",
        "event": "event",
        "state": "state",
    }
    ROLE_NAMES = ("geom", "transition", "event", "state", "layer")

    def prepare_sources(
        self,
        sources: dict[str, Tensor],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        allow_empty: bool,
    ) -> tuple[Tensor | None, Tensor, dict[str, tuple[int, int]]]:
        memory, _, ranges = super().prepare_sources(
            sources,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            allow_empty=allow_empty,
        )
        counts = self.role_token_counts(ranges)
        if not allow_empty:
            missing = [role for role in self.ROLE_NAMES if counts.get(role, 0) <= 0]
            if missing:
                raise RuntimeError(
                    "owned evidence memory requires every semantic role; missing "
                    + ", ".join(missing)
                )
        total_tokens = max((stop for _, stop in ranges.values()), default=0)
        key_bias = torch.zeros(total_tokens, device=device, dtype=torch.float32)
        for name, (start, stop) in ranges.items():
            role_count = max(int(counts[self._source_role(name)]), 1)
            # Every active role receives one unit of prior probability mass.
            # Multiple sources inside geom therefore share rather than duplicate
            # its budget.
            key_bias[start:stop] = -math.log(float(role_count))
        return memory, key_bias, ranges


class SemanticEvidenceWorkspaceBlock(nn.Module):
    """AdaLN-conditioned workspace block with one evidence write path."""

    def __init__(
        self,
        config: PolicyEvidenceConfig,
        *,
        ffn_expansion: float | None = None,
        causal_attention: bool | None = None,
    ) -> None:
        super().__init__()
        h = int(config.hidden_size)
        heads = int(config.num_heads)
        if h % heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads for evidence workspace")
        self.config = config
        self.causal_attention = (
            bool(int(getattr(config, "latent_cvae_causal_attention", 1)))
            if causal_attention is None else bool(causal_attention)
        )
        self.heads = heads
        self.head_dim = h // heads
        self.self_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(h, heads, batch_first=True, dropout=float(config.dropout))
        self.cross_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.memory_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.cross_q = nn.Linear(h, h, bias=False)
        self.cross_k = nn.Linear(h, h, bias=False)
        self.cross_v = nn.Linear(h, h, bias=False)
        self.cross_out = nn.Linear(h, h)
        self.ffn_norm = nn.LayerNorm(h, elementwise_affine=False)
        expansion = (
            float(getattr(config, "latent_cvae_ffn_expansion", 2.0))
            if ffn_expansion is None else float(ffn_expansion)
        )
        self.ffn = BiasFreeFFN(h, expansion)
        self.mod = nn.Linear(h, 9 * h)
        self.drop = nn.Dropout(float(config.dropout))
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)
        # Evidence is visible from the first update, while z/time learns how to
        # specialize the read through AdaLN without an initially dead path.
        nn.init.constant_(self.mod.bias[5 * h: 6 * h], math.atanh(0.10))

    @staticmethod
    def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        return x * (1.0 + scale[:, None]) + shift[:, None]

    def _split_heads(self, x: Tensor) -> Tensor:
        b, n, h = x.shape
        return x.reshape(b, n, self.heads, h // self.heads).transpose(1, 2)

    @staticmethod
    def _merge_heads(x: Tensor) -> Tensor:
        b, heads, n, d = x.shape
        return x.transpose(1, 2).reshape(b, n, heads * d)

    def project_memory(self, memory: Tensor) -> tuple[Tensor, Tensor]:
        memory_value = self.memory_norm(memory)
        return (
            self._split_heads(self.cross_k(memory_value)),
            self._split_heads(self.cross_v(memory_value)),
        )

    def forward(
        self,
        workspace: Tensor,
        primary_cond: Tensor,
        *,
        memory_k: Tensor,
        memory_v: Tensor,
        key_bias: Tensor,
        query_context: Tensor,
        read_scale: Tensor | None = None,
        self_attention_mask: Tensor | None = None,
        cross_attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        (
            self_s, self_c, self_g,
            cross_s, cross_c, cross_g,
            ffn_s, ffn_c, ffn_g,
        ) = self.mod(primary_cond).chunk(9, dim=-1)
        causal_mask = self_attention_mask
        if causal_mask is None and self.causal_attention:
            n = int(workspace.shape[1])
            causal_mask = torch.triu(torch.ones(n, n, device=workspace.device, dtype=torch.bool), diagonal=1)
        self_value = self._modulate(self.self_norm(workspace), self_s, self_c)
        self_update, _ = self.self_attn(
            self_value,
            self_value,
            self_value,
            attn_mask=causal_mask,
            need_weights=False,
        )
        workspace = workspace + torch.tanh(self_g)[:, None] * self.drop(self_update)

        # Current action state selects evidence but never enters the workspace
        # value/residual stream. This preserves state-dependent retrieval without
        # creating an action -> workspace -> action echo shortcut.
        cross_query = self.cross_norm(workspace + query_context)
        q = self._split_heads(self.cross_q(self._modulate(cross_query, cross_s, cross_c)))
        scores = torch.matmul(q.float(), memory_k.float().transpose(-2, -1)) * (float(self.head_dim) ** -0.5)
        key_bias = key_bias.to(device=scores.device, dtype=scores.dtype)
        if key_bias.ndim == 1:
            scores = scores + key_bias[None, None, None]
        elif key_bias.ndim == 2:
            scores = scores + key_bias[:, None, None]
        else:
            raise ValueError(f"key_bias must be [M] or [B,M], got {tuple(key_bias.shape)}")
        if cross_attention_mask is not None:
            expected = (int(workspace.shape[1]), int(memory_k.shape[2]))
            if tuple(cross_attention_mask.shape) != expected:
                raise ValueError(
                    f"cross_attention_mask must be {expected}, got {tuple(cross_attention_mask.shape)}"
                )
            scores = scores.masked_fill(
                cross_attention_mask.to(device=scores.device, dtype=torch.bool)[None, None],
                torch.finfo(scores.dtype).min,
            )
        weights = torch.softmax(scores, dim=-1).to(dtype=q.dtype)
        cross_update = torch.matmul(weights, memory_v)
        cross_update = self.cross_out(self._merge_heads(cross_update))
        cross_gain = torch.tanh(cross_g)[:, None]
        if read_scale is not None:
            cross_gain = cross_gain * read_scale.to(device=workspace.device, dtype=workspace.dtype)[:, None, None]
        workspace = workspace + cross_gain * self.drop(cross_update)

        ffn_update = self.ffn(self._modulate(self.ffn_norm(workspace), ffn_s, ffn_c))
        workspace = workspace + torch.tanh(ffn_g)[:, None] * self.drop(ffn_update)
        return workspace, weights


class HierarchicalWorkspaceManager(nn.Module):
    """Condition/stage-driven retrieval manager with no action input.

    Stage content is allowed to shape selector state, role logits, promotion,
    and stage output strength. The low-output scale is intentionally computed
    from condition+step only, so stage cannot modulate the low value stream.
    """

    def __init__(
        self,
        config: PolicyEvidenceConfig,
        role_names: tuple[str, ...],
        *,
        manage_output_strength: bool = True,
        manage_role_strength: bool = True,
    ) -> None:
        super().__init__()
        h = int(config.hidden_size)
        self.hidden_size = h
        self.role_names = tuple(role_names)
        self.manage_output_strength = bool(manage_output_strength)
        self.manage_role_strength = bool(manage_role_strength)
        self.base_fusion = nn.Sequential(
            nn.LayerNorm(2 * h),
            nn.Linear(2 * h, h),
            nn.SiLU(),
            nn.Linear(h, h),
        )
        self.stage_role_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.stage_content_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.stage_query = nn.Linear(h, h, bias=False)
        self.stage_role_key = nn.Linear(h, h, bias=False)
        self.stage_content_key = nn.Linear(h, h, bias=False)
        self.stage_content_value = nn.Linear(h, h, bias=False)
        self.stage_summary_out = nn.Linear(h, h)
        self.select_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.query_shift = nn.Linear(h, h)
        self.role_head = nn.Linear(h, len(self.role_names)) if self.manage_role_strength else None
        self.promote_head = nn.Linear(h, 1)
        self.low_output_head = nn.Linear(h, 1) if self.manage_output_strength else None
        self.stage_output_head = nn.Linear(h, 1) if self.manage_output_strength else None
        controlled_modules = [self.query_shift, self.promote_head]
        if self.role_head is not None:
            controlled_modules.append(self.role_head)
        if self.low_output_head is not None:
            controlled_modules.append(self.low_output_head)
        if self.stage_output_head is not None:
            controlled_modules.append(self.stage_output_head)
        for module in controlled_modules:
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)
        # Begin as a conservative promotion controller. This gate remains
        # effective because it is applied after normalization in the stage cell.
        nn.init.constant_(self.promote_head.bias, math.log(0.10 / 0.90))
        # Stage is a new condition group on top of a pretrained MMDiT. Let it
        # enter with 0.1 prior strength and earn more attention during training.
        if self.stage_output_head is not None:
            stage_fraction = 0.10 / 1.50
            nn.init.constant_(self.stage_output_head.bias, math.log(stage_fraction / (1.0 - stage_fraction)))

    def forward(
        self,
        *,
        primary_cond: Tensor,
        step_embedding: Tensor,
        stage_role: Tensor,
        stage_content: Tensor,
        memory_bank: EvidenceMemoryBank,
        ranges: dict[str, tuple[int, int]],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        base_state = self.base_fusion(torch.cat([primary_cond, step_embedding], dim=-1))
        q = self.stage_query(base_state)[:, None]
        normalized_role = self.stage_role_norm(stage_role)
        normalized_content = self.stage_content_norm(stage_content)
        k = self.stage_role_key(normalized_role) + self.stage_content_key(normalized_content)
        v = self.stage_content_value(normalized_content)
        logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * (float(self.hidden_size) ** -0.5)
        stage_weights = torch.softmax(logits, dim=-1).to(dtype=stage_content.dtype)
        stage_summary = self.stage_summary_out(torch.matmul(stage_weights, v).squeeze(1))
        select_state = self.select_norm(base_state + stage_summary)

        query_shift = self.query_shift(select_state)
        if self.role_head is None:
            role_logits = torch.zeros(
                int(select_state.shape[0]), len(self.role_names),
                device=select_state.device, dtype=select_state.dtype,
            )
            role_key_bias = torch.zeros(
                int(select_state.shape[0]),
                max((stop for _, stop in ranges.values()), default=0),
                device=select_state.device,
                dtype=select_state.dtype,
            )
        else:
            role_logits = self.role_head(select_state)
            role_key_bias = memory_bank.role_key_bias(role_logits, ranges)
        promote_gate = torch.sigmoid(self.promote_head(select_state)).squeeze(-1)
        # This head never sees stage_summary/select_state. That is the explicit
        # stage -> low-value firewall; stage affects only query/role selection.
        if self.low_output_head is None or self.stage_output_head is None:
            # The clean decoder has one owner for final evidence consumption:
            # MMDiT attention.  The manager still selects evidence and controls
            # promotion, but cannot independently silence either output group.
            low_output_strength = torch.ones(
                int(base_state.shape[0]), device=base_state.device, dtype=base_state.dtype
            )
            stage_output_strength = torch.ones_like(low_output_strength)
        else:
            low_output_strength = torch.exp(0.5 * torch.tanh(self.low_output_head(base_state))).squeeze(-1)
            stage_output_strength = 1.5 * torch.sigmoid(self.stage_output_head(select_state)).squeeze(-1)

        role_counts = memory_bank.role_token_counts(ranges)
        active_mask = torch.tensor(
            [role_counts.get(role, 0) > 0 for role in self.role_names],
            device=role_logits.device,
            dtype=torch.bool,
        )
        masked_role_logits = role_logits.float().masked_fill(~active_mask[None], -1e4)
        role_probs = torch.softmax(masked_role_logits, dim=-1)
        role_entropy = -(role_probs.clamp_min(1e-8) * role_probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
        stage_prob = stage_weights.detach().float().clamp_min(1e-8)
        metrics: dict[str, Tensor] = {
            "hierarchical_manager_stage_attention_entropy": -(stage_prob * stage_prob.log()).sum(dim=-1).mean(),
            "hierarchical_manager_stage_attention_max": stage_prob.max(dim=-1).values.mean(),
            "hierarchical_manager_role_entropy": role_entropy.detach(),
            "hierarchical_manager_role_max": role_probs.detach().max(dim=-1).values.mean(),
            "hierarchical_manager_query_shift_norm": query_shift.detach().float().norm(dim=-1).mean(),
            "hierarchical_manager_promote_gate": promote_gate.detach().float().mean(),
            "hierarchical_manager_low_output_strength": low_output_strength.detach().float().mean(),
            "hierarchical_manager_stage_output_strength": stage_output_strength.detach().float().mean(),
            "hierarchical_manager_fixed_output_prior": torch.as_tensor(
                float(not self.manage_output_strength), device=base_state.device, dtype=torch.float32
            ),
            "hierarchical_manager_fixed_role_prior": torch.as_tensor(
                float(not self.manage_role_strength), device=base_state.device, dtype=torch.float32
            ),
        }
        for index, role in enumerate(self.role_names):
            metrics[f"hierarchical_manager_role_{role}_prob"] = role_probs[:, index].detach().float().mean()
        return (
            query_shift,
            role_key_bias,
            promote_gate,
            low_output_strength,
            stage_output_strength,
            metrics,
        )


class WorkspaceControllerInterface(nn.Module):
    """Token-to-token selector bridge from controller state to workspace.

    Workspace role/content tensors are queries only.  Values come exclusively
    from controller state, so this interface can make retrieval and promotion
    state-dependent without creating a stage/action -> evidence value bypass.
    """

    def __init__(self, config: PolicyEvidenceConfig, *, role_count: int) -> None:
        super().__init__()
        h = int(config.hidden_size)
        heads = int(config.hierarchical_mmdit_controller_heads)
        if h % heads:
            raise ValueError(
                "workspace controller interface hidden_size must be divisible by heads"
            )
        # Preserve the host model's dropout stream while the zero-initialized
        # interface is behaviorally neutral.
        dropout = 0.0
        expansion = float(config.hierarchical_mmdit_controller_ffn_expansion)
        self.hidden_size = h
        self.role_count = int(role_count)
        if self.role_count < 1:
            raise ValueError("workspace controller interface needs semantic roles")
        self.low_query_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.low_role_identity = nn.Parameter(
            torch.randn(1, self.role_count, h) * 0.02
        )
        self.low_role_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.low_role_query = nn.Linear(h, h, bias=False)
        self.stage_role_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.stage_content_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.stage_role_query = nn.Linear(h, h, bias=False)
        self.stage_content_query = nn.Linear(h, h, bias=False)
        self.query_type = nn.Parameter(torch.randn(1, 2, h) * 0.02)
        self.query_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.control_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.cross_attn = nn.MultiheadAttention(
            h,
            heads,
            dropout=dropout,
            bias=False,
            batch_first=True,
        )
        self.self_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(
            h,
            heads,
            dropout=dropout,
            bias=False,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.ffn = BiasFreeFFN(h, expansion)
        self.final_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.drop = nn.Dropout(dropout)
        self.low_query_out = nn.Linear(h, h, bias=False)
        self.stage_query_out = nn.Linear(h, h, bias=False)
        self.promote_out = nn.Linear(h, 1, bias=False)
        nn.init.zeros_(self.low_query_out.weight)
        nn.init.zeros_(self.stage_query_out.weight)
        nn.init.zeros_(self.promote_out.weight)
        promote = 0.10
        self.register_buffer(
            "promote_base_logit",
            torch.tensor(math.log(promote / (1.0 - promote))),
            persistent=True,
        )

    @staticmethod
    def _attention_metrics(
        weights: Tensor,
        *,
        prefix: str,
    ) -> dict[str, Tensor]:
        probability = weights.detach().float().mean(dim=1)
        probability = probability / probability.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        entropy = -(
            probability.clamp_min(1e-8) * probability.clamp_min(1e-8).log()
        ).sum(dim=-1)
        control_load = probability.mean(dim=1)
        control_load = control_load / control_load.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        control_load_entropy = -(
            control_load.clamp_min(1e-8) * control_load.clamp_min(1e-8).log()
        ).sum(dim=-1)
        slot_diversity = (
            probability - probability.mean(dim=1, keepdim=True)
        ).square().sum(dim=-1).sqrt()
        return {
            f"{prefix}_attention_entropy": entropy.mean(),
            f"{prefix}_attention_max": probability.max(dim=-1).values.mean(),
            f"{prefix}_effective_control_tokens": torch.exp(entropy).mean(),
            f"{prefix}_slot_diversity": slot_diversity.mean(),
            # ``prefix`` already identifies the control stream.  Do not add a
            # second ``control`` segment; the old key made the runtime read a
            # permanently missing metric and silently print zero.
            f"{prefix}_load_effective_tokens": torch.exp(
                control_load_entropy
            ).mean(),
        }

    def forward(
        self,
        *,
        control_tokens: Tensor,
        control_addresses: Tensor | None = None,
        low_query: Tensor,
        low_role_ids: Tensor,
        stage_role: Tensor,
        stage_content: Tensor,
    ) -> WorkspaceControllerInterfaceOutput:
        if control_tokens.ndim != 3 or int(control_tokens.shape[-1]) != self.hidden_size:
            raise ValueError("workspace controller tokens must be [B,C,H]")
        batch = int(control_tokens.shape[0])
        if control_addresses is None:
            control_addresses = torch.zeros_like(control_tokens)
        if tuple(control_addresses.shape) != tuple(control_tokens.shape):
            raise ValueError("workspace controller addresses must match control tokens")
        low_count = int(low_query.shape[1])
        stage_count = int(stage_role.shape[1])
        if int(control_tokens.shape[1]) < 1 or low_count < 1 or stage_count < 1:
            raise ValueError(
                "workspace controller interface requires control, low, and stage tokens"
            )
        for name, value, expected_count in (
            ("low_query", low_query, low_count),
            ("stage_role", stage_role, stage_count),
            ("stage_content", stage_content, stage_count),
        ):
            if tuple(value.shape) != (batch, expected_count, self.hidden_size):
                raise ValueError(
                    f"workspace interface {name} has invalid shape {tuple(value.shape)}"
                )
        if tuple(low_role_ids.shape) != (low_count,):
            raise ValueError("workspace interface low_role_ids must be [L]")
        low_role_ids = low_role_ids.to(device=low_query.device, dtype=torch.long)
        if low_role_ids.device.type == "cpu" and (
            bool((low_role_ids < 0).any())
            or bool((low_role_ids >= self.role_count).any())
        ):
            raise ValueError("workspace interface low_role_ids are out of range")
        low_role = self.low_role_identity.to(
            device=low_query.device, dtype=low_query.dtype
        ).index_select(1, low_role_ids).expand(batch, -1, -1)
        low = (
            self.low_query_norm(low_query)
            + self.low_role_query(self.low_role_norm(low_role))
            + self.query_type[:, 0:1].to(device=low_query.device, dtype=low_query.dtype)
        )
        stage = (
            self.stage_role_query(self.stage_role_norm(stage_role))
            + self.stage_content_query(self.stage_content_norm(stage_content))
            + self.query_type[:, 1:2].to(device=stage_role.device, dtype=stage_role.dtype)
        )
        query = torch.cat([low, stage], dim=1)
        # Address is retrieval geometry only. It changes K, never V, so the
        # controller can choose which memory slot to read without fabricating
        # evidence content for the workspace.
        control_key = self.control_norm(control_tokens + control_addresses)
        control_value = self.control_norm(control_tokens)
        cross, weights = self.cross_attn(
            self.query_norm(query),
            control_key,
            control_value,
            need_weights=True,
            average_attn_weights=False,
        )
        # Do not add query here.  Every interface output must remain a function
        # of controller values rather than a workspace-only shortcut.
        state = cross
        self_value = self.self_norm(state)
        self_update, _ = self.self_attn(
            self_value, self_value, self_value, need_weights=False
        )
        state = state + self.drop(self_update)
        state = state + self.drop(self.ffn(self.ffn_norm(state)))
        state = self.final_norm(state)
        low_state, stage_state = state.split((low_count, stage_count), dim=1)
        low_delta = self.low_query_out(low_state)
        stage_delta = self.stage_query_out(stage_state)
        promote_gate = torch.sigmoid(
            self.promote_base_logit.to(device=state.device, dtype=state.dtype)
            + self.promote_out(stage_state).squeeze(-1)
        )

        low_weights = weights[:, :, :low_count]
        stage_weights = weights[:, :, low_count:]
        low_probability = low_weights.detach().float().mean(dim=1)
        low_probability = low_probability / low_probability.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        metrics = {
            "workspace_interface_control_response_norm": cross.detach().float().norm(dim=-1).mean(),
            "workspace_interface_state_norm": state.detach().float().norm(dim=-1).mean(),
            "workspace_interface_state_slot_diversity": (
                state.detach().float()
                - state.detach().float().mean(dim=1, keepdim=True)
            ).norm(dim=-1).mean(),
            "workspace_interface_low_query_delta_norm": low_delta.detach().float().norm(dim=-1).mean(),
            "workspace_interface_low_query_delta_ratio": (
                low_delta.detach().float().norm(dim=-1)
                / low_query.detach().float().norm(dim=-1).clamp_min(1e-8)
            ).mean(),
            "workspace_interface_stage_query_delta_norm": stage_delta.detach().float().norm(dim=-1).mean(),
            "workspace_interface_stage_query_delta_ratio": (
                stage_delta.detach().float().norm(dim=-1)
                / (
                    stage_role.detach().float().norm(dim=-1)
                    + stage_content.detach().float().norm(dim=-1)
                ).clamp_min(1e-8)
            ).mean(),
            "workspace_interface_promote_mean": promote_gate.detach().float().mean(),
            "workspace_interface_promote_std": promote_gate.detach().float().std(
                dim=1, unbiased=False
            ).mean(),
            "workspace_interface_control_token_count": torch.tensor(
                float(control_tokens.shape[1]), device=state.device, dtype=torch.float32
            ),
            **self._attention_metrics(
                low_weights, prefix="workspace_interface_low_control"
            ),
            **self._attention_metrics(
                stage_weights, prefix="workspace_interface_stage_control"
            ),
        }
        return WorkspaceControllerInterfaceOutput(
            low_query_delta=low_delta,
            stage_query_delta=stage_delta,
            promote_gate=promote_gate,
            low_control_attention=low_probability,
            metrics=metrics,
        )


class HierarchicalEvidenceWorkspace(nn.Module):
    """Temporary low reads plus persistent, role-separated stage memory."""

    def __init__(
        self,
        config: PolicyEvidenceConfig,
        *,
        owned_evidence: bool = False,
        manage_output_strength: bool = True,
        contract_conditioning: bool = False,
        stratified_roles: bool = False,
        low_count: int | None = None,
        stage_count: int | None = None,
        refine_steps: int | None = None,
        ffn_expansion: float | None = None,
        causal_attention: bool | None = None,
        stage_promote_scale_init: float | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        h = int(config.hidden_size)
        heads = int(config.num_heads)
        if h % heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads for hierarchical workspace")
        self.hidden_size = h
        self.heads = heads
        self.head_dim = h // heads
        self.low_count = int(
            getattr(config, "latent_cvae_horizon_tokens", config.action_horizon)
            if low_count is None else low_count
        )
        self.stage_count = int(
            getattr(config, "latent_cvae_stage_slots", 6)
            if stage_count is None else stage_count
        )
        self.refine_steps = max(int(
            getattr(config, "adaptive_cvae_refine_steps", 1)
            if refine_steps is None else refine_steps
        ), 1)
        self.contract_conditioning = bool(contract_conditioning)
        self.memory_bank = OwnedEvidenceMemoryBank(config) if owned_evidence else EvidenceMemoryBank(config)
        self.stratified_roles = bool(stratified_roles)
        if self.stratified_roles and not owned_evidence:
            raise ValueError("role-stratified workspace requires owned_evidence=True")

        # Value and selector identities are separate parameters. Per-sample
        # stage state is never added to low_value_seed.
        self.low_value_seed = nn.Parameter(torch.randn(1, self.low_count, h) * 0.02)
        self.low_selector_seed = nn.Parameter(torch.randn(1, self.low_count, h) * 0.02)
        self.step_embedding = nn.Parameter(torch.randn(1, self.refine_steps, h) * 0.02)
        self.condition_query = (
            None if self.contract_conditioning else nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
        )
        self.read_contract_norm = nn.LayerNorm(h, elementwise_affine=False) if self.contract_conditioning else None
        self.read_contract_mod = nn.Linear(h, 2 * h) if self.contract_conditioning else None
        if self.read_contract_mod is not None:
            nn.init.normal_(self.read_contract_mod.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.read_contract_mod.bias)
        self.manager = HierarchicalWorkspaceManager(
            config,
            self.memory_bank.ROLE_NAMES,
            manage_output_strength=manage_output_strength,
            manage_role_strength=not self.stratified_roles,
        )
        role_count = len(self.memory_bank.ROLE_NAMES)
        if self.stratified_roles and self.low_count < role_count:
            raise ValueError(
                f"role-stratified workspace needs at least {role_count} low slots, got {self.low_count}"
            )
        if self.stratified_roles and self.low_count % role_count != 0:
            raise ValueError(
                "role-stratified workspace requires an equal number of slots per role: "
                f"low_count={self.low_count}, role_count={role_count}"
            )
        low_role_ids = torch.arange(self.low_count, dtype=torch.long) % max(role_count, 1)
        self.register_buffer(
            "low_slot_role_ids",
            low_role_ids,
            persistent=self.stratified_roles,
        )
        self.low_role_embed = (
            nn.Parameter(torch.randn(1, role_count, h) * 0.02)
            if self.stratified_roles else None
        )

        self.low_stage_query = nn.Linear(h, h, bias=False)
        self.stage_role_selector_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.stage_content_selector_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.low_stage_role_key = nn.Linear(h, h, bias=False)
        self.low_stage_content_key = nn.Linear(h, h, bias=False)
        self.low_stage_role_value = nn.Linear(h, h, bias=False)
        self.low_stage_content_value = nn.Linear(h, h, bias=False)
        self.low_stage_out = nn.Linear(h, h)
        self.low_blocks = nn.ModuleList([
            SemanticEvidenceWorkspaceBlock(
                config,
                ffn_expansion=ffn_expansion,
                causal_attention=causal_attention,
            )
            for _ in range(2)
        ])
        self.low_final_norm = nn.LayerNorm(h, elementwise_affine=False)

        # Role is a persistent learned identity. Content is the only recurrent
        # state and is initialized without adding the role tensor. One shared
        # seed avoids both zero-LayerNorm gain and a second hidden slot-role.
        self.stage_role = nn.Parameter(torch.randn(1, self.stage_count, h) * 0.02)
        stage_seed_count = self.stage_count if self.contract_conditioning else 1
        self.stage_content_seed = nn.Parameter(torch.randn(1, stage_seed_count, h) * 0.02)
        self.stage_init = (
            None if self.contract_conditioning else nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
        )
        if self.stage_init is not None:
            nn.init.zeros_(self.stage_init[-1].weight)
            nn.init.zeros_(self.stage_init[-1].bias)
        self.stage_contract_norm = nn.LayerNorm(h, elementwise_affine=False) if self.contract_conditioning else None
        self.stage_contract_mod = nn.Linear(h, 2 * h) if self.contract_conditioning else None
        if self.stage_contract_mod is not None:
            nn.init.normal_(self.stage_contract_mod.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.stage_contract_mod.bias)
        self.stage_role_query = nn.Linear(h, h, bias=False)
        self.stage_content_query = nn.Linear(h, h, bias=False)
        self.stage_condition_query = nn.Linear(h, h, bias=False)
        self.stage_low_key = nn.Linear(h, h, bias=False)
        self.stage_low_value = nn.Linear(h, h, bias=False)
        self.stage_promote_out = nn.Linear(h, h)
        self.stage_input_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.stage_gru = nn.GRUCell(h, h)
        with torch.no_grad():
            self.stage_gru.bias_ih.zero_()
            self.stage_gru.bias_hh.zero_()
            # PyTorch GRU gate order is reset, update(retain), candidate.
            self.stage_gru.bias_ih[h:2 * h].fill_(0.5)
            self.stage_gru.bias_hh[h:2 * h].fill_(0.5)
        promote_value = (
            float(getattr(config, "latent_cvae_stage_promote_scale_init", 0.05))
            if stage_promote_scale_init is None else float(stage_promote_scale_init)
        )
        promote_init = min(max(promote_value, 1e-4), 1.0 - 1e-4)
        self.stage_promote_scale_logit = nn.Parameter(torch.tensor(math.log(promote_init / (1.0 - promote_init))))
        self.stage_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.stage_content_out = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
        self.stage_role_out = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
        nn.init.eye_(self.stage_content_out[-1].weight)
        nn.init.zeros_(self.stage_content_out[-1].bias)
        nn.init.eye_(self.stage_role_out[-1].weight)
        nn.init.zeros_(self.stage_role_out[-1].bias)
        self.controller_interface: WorkspaceControllerInterface | None = None
        if int(getattr(config, "hierarchical_mmdit_unified_controller", 0)):
            host_rng_state = torch.get_rng_state()
            interface_generator = torch.Generator(device="cpu")
            interface_generator.manual_seed(
                (int(torch.initial_seed()) ^ 0x51A7E1F3) % (2**63 - 1)
            )
            try:
                torch.set_rng_state(interface_generator.get_state())
                self.controller_interface = WorkspaceControllerInterface(
                    config, role_count=role_count
                )
            finally:
                torch.set_rng_state(host_rng_state)

    def _split_heads(self, x: Tensor) -> Tensor:
        b, n, h = x.shape
        return x.reshape(b, n, self.heads, h // self.heads).transpose(1, 2)

    @staticmethod
    def _merge_heads(x: Tensor) -> Tensor:
        b, heads, n, d = x.shape
        return x.transpose(1, 2).reshape(b, n, heads * d)

    def _attention(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        qh = self._split_heads(q)
        kh = self._split_heads(k)
        vh = self._split_heads(v)
        logits = torch.matmul(qh.float(), kh.float().transpose(-2, -1)) * (float(self.head_dim) ** -0.5)
        weights = torch.softmax(logits, dim=-1).to(dtype=q.dtype)
        return self._merge_heads(torch.matmul(weights, vh)), weights

    def _step_state(self, step_index: int, *, batch: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        index = min(max(int(step_index), 0), self.refine_steps - 1)
        return self.step_embedding[:, index].to(device=device, dtype=dtype).expand(batch, -1)

    def prepare_evidence(
        self,
        sources: dict[str, Tensor],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> PreparedEvidenceMemory:
        return self.memory_bank.prepare_static_memory(
            sources,
            blocks=self.low_blocks,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

    def _role_masks(
        self,
        prepared_evidence: PreparedEvidenceMemory,
        *,
        device: torch.device,
    ) -> tuple[Tensor | None, Tensor | None]:
        if not self.stratified_roles:
            return None, None
        role_to_index = {role: index for index, role in enumerate(self.memory_bank.ROLE_NAMES)}
        memory_count = max((stop for _, stop in prepared_evidence.ranges.values()), default=0)
        memory_roles = torch.full((memory_count,), -1, device=device, dtype=torch.long)
        for name, (start, stop) in prepared_evidence.ranges.items():
            role = self.memory_bank._source_role(name)
            memory_roles[start:stop] = int(role_to_index[role])
        if bool((memory_roles < 0).any()):
            raise RuntimeError("owned evidence role mask contains unassigned memory tokens")
        slot_roles = self.low_slot_role_ids.to(device=device)
        cross_mask = slot_roles[:, None] != memory_roles[None, :]
        self_mask = slot_roles[:, None] != slot_roles[None, :]
        if self.low_blocks and self.low_blocks[0].causal_attention:
            self_mask = self_mask | torch.triu(
                torch.ones(self.low_count, self.low_count, device=device, dtype=torch.bool), diagonal=1
            )
        return self_mask, cross_mask

    def init_stage(self, stage_contract: Tensor) -> Tensor:
        batch = int(stage_contract.shape[0])
        content = self.stage_content_seed.to(device=stage_contract.device, dtype=stage_contract.dtype).expand(
            batch, self.stage_count, -1
        )
        if self.contract_conditioning:
            if self.stage_contract_norm is None or self.stage_contract_mod is None:
                raise RuntimeError("factorized stage contract modules are not initialized")
            shift, scale = self.stage_contract_mod(self.stage_contract_norm(stage_contract)).chunk(2, dim=-1)
            content = self.stage_norm(content) * (1.0 + scale[:, None]) + shift[:, None]
        else:
            if self.stage_init is None:
                raise RuntimeError("legacy stage initializer is not initialized")
            content = content + self.stage_init(stage_contract)[:, None]
        return self.stage_norm(content)

    def _low_selector_base(
        self,
        *,
        primary_cond: Tensor,
        read_contract: Tensor | None,
        step_state: Tensor,
    ) -> Tensor:
        batch = int(primary_cond.shape[0])
        selector_seed = self.low_selector_seed.to(device=primary_cond.device, dtype=primary_cond.dtype).expand(batch, -1, -1)
        if self.contract_conditioning:
            if read_contract is None or self.read_contract_norm is None or self.read_contract_mod is None:
                raise ValueError("factorized hierarchical workspace requires a read contract")
            shift, scale = self.read_contract_mod(self.read_contract_norm(read_contract)).chunk(2, dim=-1)
            selector_seed = self.low_final_norm(selector_seed) * (1.0 + scale[:, None]) + shift[:, None]
        else:
            if self.condition_query is None:
                raise RuntimeError("legacy condition query is not initialized")
            selector_seed = selector_seed + self.condition_query(primary_cond)[:, None]
        return selector_seed + step_state[:, None]

    def _low_selector_context(
        self,
        *,
        selector_seed: Tensor,
        manager_shift: Tensor,
        stage_role: Tensor,
        stage_content: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if manager_shift.ndim == 2:
            selector_seed = selector_seed + manager_shift[:, None]
        elif tuple(manager_shift.shape) == tuple(selector_seed.shape):
            selector_seed = selector_seed + manager_shift
        else:
            raise ValueError(
                "hierarchical workspace manager shift must be [B,H] or [B,L,H], "
                f"got {tuple(manager_shift.shape)}"
            )
        q = self.low_stage_query(selector_seed)
        normalized_role = self.stage_role_selector_norm(stage_role)
        normalized_content = self.stage_content_selector_norm(stage_content)
        k = self.low_stage_role_key(normalized_role) + self.low_stage_content_key(normalized_content)
        role_v = self.low_stage_role_value(normalized_role)
        content_v = self.low_stage_content_value(normalized_content)
        role_context, weights = self._attention(q, k, role_v)
        content_context = self._merge_heads(torch.matmul(weights, self._split_heads(content_v)))
        stage_selector = self.low_stage_out(role_context + content_context)
        # stage_selector is returned only as query_context to the evidence
        # cross-attention. It is never added to low_value_seed or low residuals.
        return selector_seed + stage_selector, weights, role_context, content_context

    def _stage_retain_gate(self, x: Tensor, hidden: Tensor) -> Tensor:
        gi = F.linear(x, self.stage_gru.weight_ih, self.stage_gru.bias_ih)
        gh = F.linear(hidden, self.stage_gru.weight_hh, self.stage_gru.bias_hh)
        _, input_update, _ = gi.chunk(3, dim=-1)
        _, hidden_update, _ = gh.chunk(3, dim=-1)
        return torch.sigmoid(input_update + hidden_update)

    def _promote_stage(
        self,
        *,
        low_tokens: Tensor,
        stage_role: Tensor,
        stage_content: Tensor,
        primary_cond: Tensor,
        step_state: Tensor,
        promote_gate: Tensor,
        controller_query_delta: Tensor | None = None,
    ) -> StagePromotionOutput:
        normalized_role = self.stage_role_selector_norm(stage_role)
        normalized_content = self.stage_content_selector_norm(stage_content)
        q = (
            self.stage_role_query(normalized_role)
            + self.stage_content_query(normalized_content)
            + self.stage_condition_query(primary_cond + step_state)[:, None]
        )
        if controller_query_delta is not None:
            if tuple(controller_query_delta.shape) != tuple(q.shape):
                raise ValueError(
                    "workspace stage controller query has the wrong shape"
                )
            q = q + controller_query_delta.to(device=q.device, dtype=q.dtype)
        k = self.stage_low_key(low_tokens)
        v = self.stage_low_value(low_tokens)
        promoted, weights = self._attention(q, k, v)
        promoted = self.stage_promote_out(promoted)
        # Both recurrent and additive promotion paths consume the same
        # non-affine normalized evidence. Otherwise stage_promote_out can grow
        # around the scalar controller through the additive residual path.
        normalized_promoted = self.stage_input_norm(promoted)
        if promote_gate.ndim == 1:
            promote_scale_rows = promote_gate[:, None, None]
        elif tuple(promote_gate.shape) == tuple(stage_content.shape[:2]):
            promote_scale_rows = promote_gate[:, :, None]
        else:
            raise ValueError(
                "hierarchical workspace promotion gate must be [B] or [B,S], "
                f"got {tuple(promote_gate.shape)}"
            )
        promote_scale_rows = promote_scale_rows.to(dtype=promoted.dtype)
        gated_promoted = normalized_promoted * promote_scale_rows
        gru_input = gated_promoted
        flat_input = gru_input.reshape(-1, self.hidden_size)
        flat_hidden = stage_content.reshape(-1, self.hidden_size)
        retain = self._stage_retain_gate(flat_input, flat_hidden)
        recurrent = self.stage_gru(flat_input, flat_hidden).reshape_as(stage_content)
        promote_scale = torch.sigmoid(self.stage_promote_scale_logit).to(device=stage_content.device, dtype=stage_content.dtype)
        next_content = self.stage_norm(recurrent + promote_scale * gated_promoted)
        return StagePromotionOutput(
            next_content=next_content,
            attention_weights=weights,
            projected=promoted,
            normalized=normalized_promoted,
            gated=gated_promoted,
            gate_rows=promote_scale_rows,
            retain=retain,
            residual_scale=promote_scale,
        )

    @staticmethod
    def _slot_diversity(x: Tensor) -> Tensor:
        return (x - x.mean(dim=1, keepdim=True)).detach().float().norm(dim=-1).mean()

    def step(
        self,
        *,
        prepared_evidence: PreparedEvidenceMemory,
        stage_content: Tensor,
        primary_cond: Tensor,
        step_index: int,
        read_contract: Tensor | None = None,
        step_state_override: Tensor | None = None,
        control_override: WorkspaceControlOverride | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        batch = int(primary_cond.shape[0])
        device = primary_cond.device
        dtype = primary_cond.dtype
        if prepared_evidence.batch_size != batch:
            raise ValueError(
                f"cached hierarchical workspace batch={prepared_evidence.batch_size} does not match decode batch={batch}"
            )
        if tuple(stage_content.shape) != (batch, self.stage_count, self.hidden_size):
            raise ValueError(
                f"stage content must be {(batch, self.stage_count, self.hidden_size)}, got {tuple(stage_content.shape)}"
            )
        stage_role = self.stage_role.to(device=device, dtype=dtype).expand(batch, -1, -1)
        if step_state_override is None:
            step_state = self._step_state(step_index, batch=batch, device=device, dtype=dtype)
        else:
            if tuple(step_state_override.shape) != (batch, self.hidden_size):
                raise ValueError(
                    "hierarchical workspace step_state_override must be "
                    f"{(batch, self.hidden_size)}, got {tuple(step_state_override.shape)}"
                )
            step_state = step_state_override.to(device=device, dtype=dtype)
        selector_base = self._low_selector_base(
            primary_cond=primary_cond,
            read_contract=read_contract,
            step_state=step_state,
        )
        stage_query_delta: Tensor | None = None
        if control_override is None:
            (
                manager_shift,
                role_bias,
                promote_gate,
                low_output_strength,
                stage_output_strength,
                manager_metrics,
            ) = self.manager(
                primary_cond=primary_cond,
                step_embedding=step_state,
                stage_role=stage_role,
                stage_content=stage_content,
                memory_bank=self.memory_bank,
                ranges=prepared_evidence.ranges,
            )
        else:
            if self.controller_interface is None:
                raise RuntimeError(
                    "workspace control tokens require a controller interface"
                )
            control_tokens = control_override.control_tokens.to(
                device=device, dtype=dtype
            )
            control_addresses = (
                torch.zeros_like(control_tokens)
                if control_override.control_addresses is None
                else control_override.control_addresses.to(device=device, dtype=dtype)
            )
            if (
                control_tokens.ndim != 3
                or int(control_tokens.shape[0]) != batch
                or int(control_tokens.shape[-1]) != self.hidden_size
                or tuple(control_addresses.shape) != tuple(control_tokens.shape)
            ):
                raise ValueError(
                    "unified workspace control tokens must be [B,C,H]"
                )
            interface = self.controller_interface(
                control_tokens=control_tokens,
                control_addresses=control_addresses,
                low_query=selector_base,
                low_role_ids=self.low_slot_role_ids,
                stage_role=stage_role,
                stage_content=stage_content,
            )
            manager_shift = interface.low_query_delta
            stage_query_delta = interface.stage_query_delta
            promote_gate = interface.promote_gate
            role_bias = torch.zeros(
                batch,
                int(prepared_evidence.tokens.shape[1]),
                device=device,
                dtype=torch.float32,
            )
            low_output_strength = torch.ones(batch, device=device, dtype=dtype)
            stage_output_strength = torch.ones_like(low_output_strength)
            manager_metrics = {
                "hierarchical_manager_query_shift_norm": (
                    manager_shift.detach().float().norm(dim=-1).mean()
                ),
                "hierarchical_manager_promote_gate": promote_gate.detach().float().mean(),
                "hierarchical_manager_low_output_strength": torch.ones(
                    (), device=device, dtype=torch.float32
                ),
                "hierarchical_manager_stage_output_strength": torch.ones(
                    (), device=device, dtype=torch.float32
                ),
                "hierarchical_manager_fixed_output_prior": torch.ones(
                    (), device=device, dtype=torch.float32
                ),
                "hierarchical_manager_fixed_role_prior": torch.zeros(
                    (), device=device, dtype=torch.float32
                ),
                **interface.metrics,
            }
            for role_index, role in enumerate(self.memory_bank.ROLE_NAMES):
                role_mask = self.low_slot_role_ids.to(device=device) == role_index
                if not bool(role_mask.any()):
                    continue
                role_attention = interface.low_control_attention[:, role_mask]
                role_entropy = -(
                    role_attention.clamp_min(1e-8)
                    * role_attention.clamp_min(1e-8).log()
                ).sum(dim=-1)
                manager_metrics[
                    f"workspace_interface_role_{role}_control_entropy"
                ] = role_entropy.mean()
                manager_metrics[
                    f"workspace_interface_role_{role}_effective_control_tokens"
                ] = torch.exp(role_entropy).mean()
                manager_metrics[
                    f"workspace_interface_role_{role}_query_delta_norm"
                ] = interface.low_query_delta[:, role_mask].detach().float().norm(
                    dim=-1
                ).mean()
        query_context, selector_weights, selector_role, selector_content = self._low_selector_context(
            selector_seed=selector_base,
            manager_shift=manager_shift,
            stage_role=stage_role,
            stage_content=stage_content,
        )

        # The low value stream starts from a stage-independent seed. All stage
        # influence is confined to query_context and role_bias above.
        low = self.low_value_seed.to(device=device, dtype=dtype).expand(batch, -1, -1)
        low_seed = low
        key_bias = prepared_evidence.key_bias.to(device=device) + role_bias.to(device=device, dtype=torch.float32)
        self_attention_mask, cross_attention_mask = self._role_masks(
            prepared_evidence, device=device,
        )
        evidence_weight_rows: list[Tensor] = []
        for block_index, block in enumerate(self.low_blocks):
            memory_k, memory_v = prepared_evidence.block_kv[block_index]
            low, weights = block(
                low,
                primary_cond,
                memory_k=memory_k,
                memory_v=memory_v,
                key_bias=key_bias,
                query_context=query_context,
                read_scale=None,
                self_attention_mask=self_attention_mask,
                cross_attention_mask=cross_attention_mask,
            )
            evidence_weight_rows.append(weights.detach().float())
        low_evidence_pre_norm = low
        if self.low_role_embed is not None:
            role_embed = self.low_role_embed[:, self.low_slot_role_ids].to(device=device, dtype=dtype)
            low = low + role_embed
        low = self.low_final_norm(low)
        low_for_action = low

        promotion = self._promote_stage(
            low_tokens=low,
            stage_role=stage_role,
            stage_content=stage_content,
            primary_cond=primary_cond,
            step_state=step_state,
            promote_gate=promote_gate,
            controller_query_delta=stage_query_delta,
        )
        next_stage_content = promotion.next_content
        role_component = self.stage_role_out(stage_role)
        content_component = self.stage_content_out(next_stage_content)
        stage_for_action = role_component + content_component
        low_logit_bias = low_output_strength.clamp_min(1e-4).log()
        stage_logit_bias = stage_output_strength.clamp_min(1e-4).log()

        weights = torch.stack(evidence_weight_rows).mean(dim=0)
        group_weights = torch.stack([
            weights[..., start:stop].sum(dim=-1)
            for start, stop in prepared_evidence.ranges.values()
        ], dim=-1)
        selector_prob = selector_weights.detach().float().clamp_min(1e-8)
        promote_prob = promotion.attention_weights.detach().float().clamp_min(1e-8)
        zero = torch.zeros((), device=device, dtype=torch.float32)
        role_norm = role_component.detach().float().norm(dim=-1).mean()
        content_norm = content_component.detach().float().norm(dim=-1).mean()
        metrics: dict[str, Tensor] = {
            "workspace_token_count": torch.tensor(float(self.low_count), device=device, dtype=torch.float32),
            "workspace_token_norm": low_for_action.detach().float().norm(dim=-1).mean(),
            "workspace_update_norm": (
                low_evidence_pre_norm.detach() - low_seed.detach()
            ).float().norm(dim=-1).mean(),
            "workspace_global_state_norm": zero,
            "workspace_global_slot_delta_norm": zero,
            "workspace_global_slot_diversity": zero,
            "workspace_source_count": torch.tensor(float(len(prepared_evidence.ranges)), device=device, dtype=torch.float32),
            "workspace_cached_token_fraction": torch.ones((), device=device, dtype=torch.float32),
            "workspace_noisy_query_scale": zero,
            "workspace_progress_query_norm": zero,
            "workspace_attention_entropy": -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=-1).mean(),
            "workspace_attention_max": weights.max(dim=-1).values.mean(),
            "workspace_group_attention_entropy": -(
                group_weights.clamp_min(1e-8) * group_weights.clamp_min(1e-8).log()
            ).sum(dim=-1).mean(),
            "workspace_attention_mass_error": (group_weights.sum(dim=-1) - 1.0).abs().mean(),
            "hierarchical_low_token_count": torch.tensor(float(self.low_count), device=device, dtype=torch.float32),
            "hierarchical_low_role_stratified": torch.tensor(
                float(self.stratified_roles), device=device, dtype=torch.float32
            ),
            "hierarchical_low_causal_attention": torch.tensor(
                float(bool(self.low_blocks and self.low_blocks[0].causal_attention)),
                device=device,
                dtype=torch.float32,
            ),
            "hierarchical_low_token_norm": low_for_action.detach().float().norm(dim=-1).mean(),
            "hierarchical_low_selector_stage_entropy": -(selector_prob * selector_prob.log()).sum(dim=-1).mean(),
            "hierarchical_low_selector_stage_max": selector_prob.max(dim=-1).values.mean(),
            "hierarchical_low_selector_stage_effective_slots": torch.exp(
                -(selector_prob * selector_prob.log()).sum(dim=-1).mean()
            ),
            "hierarchical_low_selector_role_norm": selector_role.detach().float().norm(dim=-1).mean(),
            "hierarchical_low_selector_content_norm": selector_content.detach().float().norm(dim=-1).mean(),
            "hierarchical_stage_token_count": torch.tensor(float(self.stage_count), device=device, dtype=torch.float32),
            "hierarchical_stage_role_norm": stage_role.detach().float().norm(dim=-1).mean(),
            "hierarchical_stage_role_diversity": self._slot_diversity(stage_role),
            "hierarchical_stage_content_norm": next_stage_content.detach().float().norm(dim=-1).mean(),
            "hierarchical_stage_content_diversity": self._slot_diversity(next_stage_content),
            "hierarchical_stage_role_content_cosine": F.cosine_similarity(
                stage_role.detach().float(), next_stage_content.detach().float(), dim=-1
            ).mean(),
            "hierarchical_stage_role_output_norm": role_norm,
            "hierarchical_stage_content_output_norm": content_norm,
            "hierarchical_stage_role_output_fraction": role_norm / (role_norm + content_norm).clamp_min(1e-8),
            "hierarchical_stage_update_norm": (
                next_stage_content.detach().float() - stage_content.detach().float()
            ).norm(dim=-1).mean(),
            "hierarchical_stage_retain_mean": promotion.retain.detach().float().mean(),
            "hierarchical_stage_promote_attention_entropy": -(promote_prob * promote_prob.log()).sum(dim=-1).mean(),
            "hierarchical_stage_promote_attention_max": promote_prob.max(dim=-1).values.mean(),
            "hierarchical_stage_promoted_norm": promotion.projected.detach().float().norm(dim=-1).mean(),
            "hierarchical_stage_promoted_projected_rms": (
                promotion.projected.detach().float().square().mean(dim=(1, 2)).sqrt().mean()
            ),
            "hierarchical_stage_promoted_normalized_rms": (
                promotion.normalized.detach().float().square().mean(dim=(1, 2)).sqrt().mean()
            ),
            "hierarchical_stage_promoted_realized_scale": (
                promotion.gated.detach().float().square().mean(dim=(1, 2)).sqrt().mean()
            ),
            "hierarchical_stage_promote_gate_scale_error": (
                promotion.gated.detach().float().square().mean(dim=-1).sqrt()
                - promotion.gate_rows.detach().float().squeeze(-1).abs()
            ).abs().mean(),
            "hierarchical_stage_promote_scale": promotion.residual_scale.detach().float(),
        }
        metrics["workspace_group_effective_sources"] = torch.exp(metrics["workspace_group_attention_entropy"])
        metrics.update(manager_metrics)
        metrics.update(self.memory_bank.role_attention_metrics(weights, prepared_evidence.ranges))
        if self.stratified_roles:
            evidence_delta = (low_evidence_pre_norm.detach() - low_seed.detach()).float()
            for role_index, role in enumerate(self.memory_bank.ROLE_NAMES):
                role_mask = self.low_slot_role_ids.to(device=device) == role_index
                metrics[f"hierarchical_low_role_{role}_update_norm"] = (
                    evidence_delta[:, role_mask].norm(dim=-1).mean()
                )
                metrics[f"hierarchical_low_role_{role}_output_norm"] = (
                    low_for_action.detach().float()[:, role_mask].norm(dim=-1).mean()
                )
        for name, (start, stop) in prepared_evidence.ranges.items():
            metrics[f"workspace_{name}_attention"] = weights[..., start:stop].sum(dim=-1).mean()
        transition_mass = [
            metrics[key]
            for key in (
                "workspace_transition_attention",
                "workspace_transition_delta_attention",
                "workspace_transition_effect_attention",
                "workspace_transition_timeline_attention",
            )
            if key in metrics
        ]
        if transition_mass:
            metrics["workspace_transition_total_attention"] = torch.stack(transition_mass).sum()
        return (
            low_for_action,
            next_stage_content,
            stage_for_action,
            low_logit_bias,
            stage_logit_bias,
            metrics,
        )
