"""Time-domain latent action solver backed by the current evidence bank.

This module is deliberately a separate migration path.  It keeps the useful
part of the early CVAE/MMDiT decoder (ordered layer scan plus distinct action
blocks) without restoring its posterior, adaptive refinement, or old
workspace/controller paths.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .codec import NativeTimePhysicalActionTokenLift
from .controller import EvidenceExecutionController
from .decoder import ActionOnlyPhysicalVelocityHead
from .evidence import OwnedEvidenceMemoryBank
from .gauges import deterministic_module_probe, select_centered_candidate
from .primitives import BiasFreeFFN, TimeEmbedding, sinusoidal_positions
from .refinement import NestedLowRankContractionBank
from .role_delta_attnres import PolicyRoleDeltaBank, RoleDeltaAttnRes


@dataclass
class EvidenceView:
    """Typed, normalized evidence visible to the action solver.

    The solver never receives raw canvas slices or raw layer-contract dicts.
    ``source_tokens`` are selector/key tokens; ``value_tokens`` are the
    separately compiled value stream. ``layer_tokens`` preserves ordered
    depth separately for the organizer's recurrent scan.
    """

    source_tokens: dict[str, Tensor]
    layer_tokens: Tensor
    tokens: Tensor
    value_tokens: Tensor
    intent_tokens: Tensor
    key_bias: Tensor
    ranges: dict[str, tuple[int, int]]
    summaries: dict[str, Tensor]
    masks: dict[str, Tensor]


class EvidenceViewAdapter(nn.Module):
    """Register current trunk outputs into a single typed evidence bank."""

    SOURCE_NAMES = ("layer", "trajectory", "rollout", "transition", "event", "state")
    _LAYER_FIELDS = ("rollout_tokens", "state_tokens", "state_history_tokens")

    def __init__(self, config: Any) -> None:
        super().__init__()
        h = int(config.hidden_size)
        self.hidden_size = h
        self.allow_terminal_layer_subset = bool(
            int(getattr(config, "flow_jepa_strict_role_visual_path", 0))
        )
        self.bank = OwnedEvidenceMemoryBank(config)
        self.layer_field_proj = nn.ModuleDict(
            {name: nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h)) for name in self._LAYER_FIELDS}
        )
        self.source_proj = nn.ModuleDict(
            {
                "trajectory": nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h)),
                "rollout": nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h)),
                "transition": nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h)),
                "state": nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h)),
            }
        )
        self.event_proj = nn.Sequential(nn.LayerNorm(3), nn.Linear(3, h))
        self.layer_depth_embed = nn.Parameter(torch.randn(1, int(config.depth), h) * 0.02)
        self.intent_source_names = (
            "task",
            "state",
            "state_history",
            "executed",
            "proposal",
            "visual",
        )
        self.intent_proj = nn.ModuleDict(
            {
                name: nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
                for name in self.intent_source_names
            }
        )
        if self.allow_terminal_layer_subset:
            # Strict G->W->P ownership removes raw visual intent before this
            # adapter is called. Preserve the key for compatible state dicts,
            # but do not optimize a projection that can never receive input.
            self.intent_proj["visual"].requires_grad_(False)
        self.intent_type_embed = nn.Parameter(
            torch.randn(1, len(self.intent_source_names), h) * 0.02
        )
        self.layer_intent_attention = nn.MultiheadAttention(
            h,
            int(config.num_heads),
            dropout=0.0,
            batch_first=True,
        )
        self.layer_intent_norm = nn.LayerNorm(h, elementwise_affine=False)
        # Value-side layer semantics have their own depth queries.  The mixed
        # layer contract remains a selector chart only; using it as a query
        # here would let noisy/action-conditioned content rewrite the values
        # that later compile into z and the block evidence stream.
        self.clean_layer_queries = nn.Parameter(
            torch.randn(1, int(config.depth), h) * 0.02
        )

    @staticmethod
    def _cat_memory(
        memory: Tensor | list[Tensor] | tuple[Tensor, ...] | None, *, name: str, hidden: int
    ) -> Tensor | None:
        if memory is None:
            return None
        values = [memory] if isinstance(memory, Tensor) else list(memory)
        parts: list[Tensor] = []
        for value in values:
            if not isinstance(value, Tensor) or value.ndim != 3 or int(value.shape[-1]) != hidden:
                raise ValueError(f"{name} memory must contain [B,N,{hidden}] tensors")
            if int(value.shape[1]) > 0:
                parts.append(value)
        return torch.cat(parts, dim=1) if parts else None

    def _layer_rows(self, layer_contracts: list[dict[str, Tensor]], reference: Tensor) -> Tensor:
        depth = int(self.layer_depth_embed.shape[1])
        subset_allowed = self.allow_terminal_layer_subset and 0 < len(layer_contracts) <= depth
        if len(layer_contracts) != depth and not subset_allowed:
            raise RuntimeError(
                f"evidence layer depth mismatch: expected {depth}, "
                f"got {len(layer_contracts)}"
            )
        # A strict role hierarchy exposes only the terminal policy contracts to
        # the final decoder.  They retain their original depth identities.
        depth_offset = depth - len(layer_contracts)
        rows: list[Tensor] = []
        for index, entry in enumerate(layer_contracts):
            fields: list[Tensor] = []
            for name in self._LAYER_FIELDS:
                value = entry.get(name)
                if (
                    not isinstance(value, Tensor)
                    or value.ndim != 3
                    or int(value.shape[-1]) != self.hidden_size
                ):
                    continue
                fields.append(
                    self.layer_field_proj[name](
                        value.to(device=reference.device, dtype=reference.dtype)
                    ).mean(dim=1)
                )
            if not fields:
                raise RuntimeError(f"layer contract {index} has no deploy-safe evidence field")
            row = torch.stack(fields, dim=1).mean(dim=1)
            row = row + self.layer_depth_embed[:, depth_offset + index].to(
                device=row.device, dtype=row.dtype
            )
            rows.append(row)
        return torch.stack(rows, dim=1)

    def forward(
        self,
        *,
        trajectory_tokens: Tensor,
        rollout_tokens: Tensor,
        transition_memory: Tensor | list[Tensor] | tuple[Tensor, ...] | None,
        event_evidence: Tensor,
        state_memory: Tensor | list[Tensor] | tuple[Tensor, ...],
        layer_contracts: list[dict[str, Tensor]],
        intent_memory: dict[str, Tensor] | None = None,
        visual_selector_tokens: Tensor | None = None,
        visual_value_tokens: Tensor | None = None,
        visual_key_bias: Tensor | None = None,
    ) -> EvidenceView:
        if trajectory_tokens.ndim != 3 or int(trajectory_tokens.shape[-1]) != self.hidden_size:
            raise ValueError(f"trajectory_tokens must be [B,N,{self.hidden_size}]")
        reference = trajectory_tokens
        batch = int(reference.shape[0])
        layer_tokens = self._layer_rows(layer_contracts, reference)
        transition = self._cat_memory(transition_memory, name="transition", hidden=self.hidden_size)
        state = self._cat_memory(state_memory, name="state", hidden=self.hidden_size)
        if transition is None or state is None:
            raise RuntimeError(
                "time-domain evidence requires non-empty transition and state sources"
            )
        if rollout_tokens.ndim != 3 or int(rollout_tokens.shape[-1]) != self.hidden_size:
            raise ValueError(f"rollout_tokens must be [B,N,{self.hidden_size}]")
        if event_evidence.ndim != 3 or int(event_evidence.shape[-1]) != 3:
            raise ValueError("event_evidence must be [B,N,3]")
        if any(
            int(value.shape[0]) != batch
            for value in (layer_tokens, rollout_tokens, transition, state, event_evidence)
        ):
            raise ValueError("all evidence sources must share the action batch")

        source_tokens = {
            "layer": layer_tokens,
            "trajectory": self.source_proj["trajectory"](
                trajectory_tokens.to(device=reference.device, dtype=reference.dtype)
            ),
            "rollout": self.source_proj["rollout"](
                rollout_tokens.to(device=reference.device, dtype=reference.dtype)
            ),
            "transition": self.source_proj["transition"](
                transition.to(device=reference.device, dtype=reference.dtype)
            ),
            "event": self.event_proj(
                event_evidence.to(device=reference.device, dtype=reference.dtype)
            ),
            "state": self.source_proj["state"](
                state.to(device=reference.device, dtype=reference.dtype)
            ),
        }
        for name, value in source_tokens.items():
            if value.ndim != 3 or int(value.shape[1]) <= 0:
                raise RuntimeError(f"evidence source {name!r} is empty")
        if intent_memory is None:
            intent_memory = {
                "state": state,
                "proposal": trajectory_tokens,
            }
        intent_parts: list[Tensor] = []
        for index, name in enumerate(self.intent_source_names):
            value = intent_memory.get(name)
            if not isinstance(value, Tensor):
                continue
            if value.ndim != 3 or int(value.shape[0]) != batch or int(value.shape[-1]) != self.hidden_size:
                raise ValueError(f"intent source {name!r} must be [B,N,{self.hidden_size}]")
            if int(value.shape[1]) <= 0:
                continue
            projected = self.intent_proj[name](
                value.to(device=reference.device, dtype=reference.dtype)
            )
            intent_parts.append(
                projected
                + self.intent_type_embed[:, index : index + 1].to(
                    device=reference.device, dtype=reference.dtype
                )
            )
        if not intent_parts:
            raise RuntimeError("native evidence requires at least one clean intent source")
        intent_tokens = torch.cat(intent_parts, dim=1)
        bank_sources = {name: value for name, value in source_tokens.items()}
        prepared_selector = self.bank.prepare_static_memory(
            bank_sources,
            blocks=nn.ModuleList(),
            batch_size=batch,
            device=reference.device,
            dtype=reference.dtype,
        )
        # Layer contracts remain useful as retrieval geometry, but their
        # values are compiled from pre-attention intent memory.  Thus mixed
        # canvas/noisy content may choose which clean intent evidence is read,
        # but cannot become an evidence value by itself.
        layer_count = int(layer_tokens.shape[1])
        layer_query = self.clean_layer_queries[:, -layer_count:].to(
            device=layer_tokens.device, dtype=layer_tokens.dtype
        ).expand(batch, -1, -1)
        layer_value_tokens, _ = self.layer_intent_attention(
            self.layer_intent_norm(layer_query),
            self.layer_intent_norm(intent_tokens),
            self.layer_intent_norm(intent_tokens),
            need_weights=False,
        )
        value_sources = dict(source_tokens)
        value_sources["layer"] = layer_value_tokens
        prepared_values = self.bank.prepare_static_memory(
            value_sources,
            blocks=nn.ModuleList(),
            batch_size=batch,
            device=reference.device,
            dtype=reference.dtype,
        )
        summaries = {name: value.mean(dim=1) for name, value in value_sources.items()}
        masks = {
            name: torch.ones(value.shape[:2], device=value.device, dtype=torch.bool)
            for name, value in source_tokens.items()
        }
        selector_tokens = prepared_selector.tokens
        value_tokens = prepared_values.tokens
        key_bias = prepared_selector.key_bias
        ranges = dict(prepared_selector.ranges)
        if visual_selector_tokens is not None or visual_value_tokens is not None:
            if visual_selector_tokens is None or visual_value_tokens is None:
                raise ValueError("Flow-DINO selector and value lanes must be provided together")
            expected_prefix = (batch,)
            if (
                visual_selector_tokens.ndim != 3
                or visual_value_tokens.ndim != 3
                or tuple(visual_selector_tokens.shape) != tuple(visual_value_tokens.shape)
                or tuple(visual_selector_tokens.shape[:1]) != expected_prefix
                or int(visual_selector_tokens.shape[-1]) != self.hidden_size
            ):
                raise ValueError("Flow-DINO selector/value lanes must align as [B,N,H]")
            visual_selector_tokens = visual_selector_tokens.to(
                device=reference.device, dtype=reference.dtype
            )
            visual_value_tokens = visual_value_tokens.to(
                device=reference.device, dtype=reference.dtype
            )
            start = int(selector_tokens.shape[1])
            stop = start + int(visual_selector_tokens.shape[1])
            selector_tokens = torch.cat((selector_tokens, visual_selector_tokens), dim=1)
            value_tokens = torch.cat((value_tokens, visual_value_tokens), dim=1)
            if visual_key_bias is None:
                visual_key_bias = torch.zeros(
                    stop - start, device=reference.device, dtype=key_bias.dtype
                )
            if tuple(visual_key_bias.shape) != (stop - start,):
                raise ValueError("Flow-DINO key bias must be the global [N] evidence prior")
            key_bias = torch.cat(
                (
                    key_bias,
                    visual_key_bias.to(device=reference.device, dtype=key_bias.dtype),
                ),
                dim=0,
            )
            ranges["flow_dino"] = (start, stop)
        return EvidenceView(
            source_tokens=source_tokens,
            layer_tokens=layer_tokens,
            tokens=selector_tokens,
            value_tokens=value_tokens,
            intent_tokens=intent_tokens,
            key_bias=key_bias,
            ranges=ranges,
            summaries=summaries,
            masks=masks,
        )


class EvidenceConditionOrganizer(nn.Module):
    """Compile clean intent into ``z`` and keep evidence retrieval separate.

    The depth scan consumes the value-safe layer chart, while the top-level
    latent is read from pre-attention intent memory.  Action-conditioned
    transition/event evidence therefore remains available to the action
    reader without becoming a shortcut into the global intent condition.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        h = int(config.hidden_size)
        self.hidden_size = h
        self.z_probe_enabled = bool(int(getattr(config, "latent_cvae_z_probe", 0)))
        self.source_names = EvidenceViewAdapter.SOURCE_NAMES
        self.layer_scan = nn.GRUCell(h, h)
        self.layer_scan_init = nn.Parameter(torch.zeros(1, h))
        self.lateral_proj = nn.ModuleDict(
            {name: nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h)) for name in self.source_names}
        )
        self.lateral_fusion = nn.Sequential(
            nn.LayerNorm(len(self.source_names) * h),
            nn.Linear(len(self.source_names) * h, h),
            nn.SiLU(),
            nn.Linear(h, h),
        )
        self.condition_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.latent_mu = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, int(config.latent_cvae_z_dim)))
        self.latent_to_hidden = nn.Sequential(
            nn.LayerNorm(int(config.latent_cvae_z_dim)), nn.Linear(int(config.latent_cvae_z_dim), h)
        )
        self.condition_to_hidden = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
        self.intent_query = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        self.intent_attention = nn.MultiheadAttention(
            h, int(config.num_heads), dropout=0.0, batch_first=True
        )
        self.intent_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.time = TimeEmbedding(h)
        alpha = float(getattr(config, "latent_cvae_layer_scan_alpha", 0.2))
        alpha = min(max(alpha, 0.0), 0.5)
        self.lateral_scale_logit = nn.Parameter(
            torch.tensor(self._logit(alpha / 0.5 if alpha > 0 else 1e-4))
        )
        # Kept load-compatible for older manifests, but the active native
        # organizer no longer flattens six source summaries into a parallel
        # condition bypass.  Clean intent attention is the sole compiler.
        self.lateral_proj.requires_grad_(False)
        self.lateral_fusion.requires_grad_(False)
        self.lateral_scale_logit.requires_grad_(False)
        # Compatibility readouts remain in the manifest for old checkpoints,
        # but the active native path is z-only at the global boundary.
        self.condition_to_hidden.requires_grad_(False)

    @staticmethod
    def _logit(value: float) -> float:
        value = min(max(float(value), 1e-4), 1.0 - 1e-4)
        return math.log(value / (1.0 - value))

    def forward(self, view: EvidenceView, time: Tensor) -> dict[str, Tensor | dict[str, Tensor]]:
        layer_start, layer_stop = view.ranges["layer"]
        layer = view.value_tokens[:, layer_start:layer_stop]
        state = self.layer_scan_init.to(device=layer.device, dtype=layer.dtype).expand(
            layer.shape[0], -1
        )
        for index in range(int(layer.shape[1])):
            state = self.layer_scan(layer[:, index], state)
        scan = self.condition_norm(state)
        intent_query = self.intent_query.to(
            device=scan.device, dtype=scan.dtype
        ).expand(scan.shape[0], -1, -1)
        intent_context, _ = self.intent_attention(
            self.intent_norm(intent_query),
            self.intent_norm(view.intent_tokens),
            self.intent_norm(view.intent_tokens),
            need_weights=False,
        )
        intent_context = intent_context[:, 0]
        condition = self.condition_norm(scan + intent_context)
        z = self.latent_mu(condition)
        z_token = self.latent_to_hidden(z)
        condition_hidden = self.condition_to_hidden(condition)
        time_hidden = self.time(time.to(dtype=condition.dtype))
        # ``z`` is the only semantic writer for the shared block condition.
        # Full condition features remain available through typed evidence
        # values; keeping condition_hidden in this sum would let the latent
        # bottleneck be bypassed by a parallel global adapter.
        global_condition = z_token + time_hidden
        z_zero_delta = torch.zeros((), device=z.device, dtype=torch.float32)
        z_shuffle_delta = torch.zeros((), device=z.device, dtype=torch.float32)
        if self.z_probe_enabled and not self.training:
            # This is an evaluation-only compiler probe for the active
            # Evidence path. It measures the change in the actual global
            # condition consumed by the native blocks, rather than reviving
            # the legacy CVAE action decoder's private probe.
            with torch.no_grad():
                reference = global_condition.detach().float()
                reference_norm = reference.norm(dim=-1).mean().clamp_min(1e-6)
                zero_token = self.latent_to_hidden(torch.zeros_like(z)).detach().float()
                zero_condition = zero_token + time_hidden.detach().float()
                z_zero_delta = (
                    (zero_condition - reference).norm(dim=-1).mean() / reference_norm
                )
                shuffled_z = z.roll(shifts=1, dims=0)
                shuffled_token = self.latent_to_hidden(shuffled_z).detach().float()
                shuffled_condition = shuffled_token + time_hidden.detach().float()
                z_shuffle_delta = (
                    (shuffled_condition - reference).norm(dim=-1).mean() / reference_norm
                )
        source_metrics = {
            f"evidence_{name}_summary_norm": view.summaries[name]
            .detach()
            .float()
            .norm(dim=-1)
            .mean()
            for name in self.source_names
        }
        metrics = {
            "evidence_condition_scan_norm": scan.detach().float().norm(dim=-1).mean(),
            "evidence_condition_lateral_norm": torch.zeros(
                (), device=scan.device, dtype=torch.float32
            ),
            "evidence_condition_lateral_scale": torch.zeros(
                (), device=scan.device, dtype=torch.float32
            ),
            "evidence_condition_intent_norm": intent_context.detach().float().norm(dim=-1).mean(),
            "evidence_condition_norm": condition.detach().float().norm(dim=-1).mean(),
            "evidence_latent_norm": z.detach().float().norm(dim=-1).mean(),
            "evidence_latent_batch_variance": z.detach().float().var(dim=0, unbiased=False).mean(),
            "evidence_global_condition_norm": global_condition.detach().float().norm(dim=-1).mean(),
            "evidence_z_zero_condition_delta": z_zero_delta,
            "evidence_z_shuffle_condition_delta": z_shuffle_delta,
            **source_metrics,
        }
        return {
            "condition": condition,
            "latent": z,
            "latent_token": z_token,
            "condition_hidden": condition_hidden,
            "time_hidden": time_hidden,
            "global_condition": global_condition,
            "metrics": metrics,
        }


class TimeDomainMMDiTBlock(nn.Module):
    """Time-domain action block for the native flow state.

    ``x_t`` is part of the action-token stream. The block learns how to fuse
    that state with evidence-conditioned tokens; there is no second noisy
    writer and no learned source-ratio controller.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        h = int(config.hidden_size)
        heads = int(config.num_heads)
        if h % heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = h
        self.heads = heads
        self.head_dim = h // heads
        self.residual_scale_max = float(
            getattr(config, "latent_cvae_mmdit_residual_scale_max", 0.25)
        )
        self.action_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.evidence_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.action_qkv = nn.Linear(h, 3 * h)
        self.self_out = nn.Linear(h, h)
        self.evidence_query = nn.Linear(h, h)
        self.evidence_kv = nn.Linear(h, 2 * h)
        self.evidence_out = nn.Linear(h, h)
        self.action_ffn_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.action_ffn = BiasFreeFFN(h, float(getattr(config, "latent_cvae_ffn_expansion", 2.0)))
        self.global_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.action_mod = nn.Linear(h, 6 * h)
        self.drop = nn.Dropout(float(config.dropout))
        nn.init.zeros_(self.action_mod.weight)
        nn.init.zeros_(self.action_mod.bias)
        if int(getattr(config, "flow_jepa_role_hierarchy", 0)):
            # A strictly zero native-MMDiT residual makes the first action-loss
            # backward stop at this gate: grounding/world/policy evidence can
            # only receive gradients after an optimizer step has opened it.
            # The role-hierarchical path instead starts from the same small
            # forward residual used by the hierarchical decoder.  This is an
            # ordinary forward connection, not a gradient surrogate.
            initial_residual = float(
                getattr(config, "hierarchical_mmdit_residual_scale_init", 0.05)
            )
            gate_ratio = min(
                max(initial_residual / max(self.residual_scale_max, 1e-6), 1e-4),
                0.95,
            )
            gate_bias = math.atanh(gate_ratio)
            with torch.no_grad():
                self.action_mod.bias[2 * h : 3 * h].fill_(gate_bias)
                self.action_mod.bias[5 * h : 6 * h].fill_(gate_bias)

    def _split(self, value: Tensor) -> Tensor:
        b, n, h = value.shape
        return value.reshape(b, n, self.heads, h // self.heads).transpose(1, 2)

    def _merge(self, value: Tensor) -> Tensor:
        b, heads, n, d = value.shape
        return value.transpose(1, 2).reshape(b, n, heads * d)

    @staticmethod
    def _unit_rms(value: Tensor) -> Tensor:
        denominator = value.float().square().mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        return value / denominator.to(dtype=value.dtype)

    @staticmethod
    def _attention_stats(weights: Tensor) -> tuple[Tensor, Tensor]:
        probs = weights.detach().float().clamp_min(1e-8)
        entropy = -(probs * probs.log()).sum(dim=-1)
        token_count = int(weights.shape[-1])
        if token_count > 1:
            entropy = entropy / math.log(float(token_count))
        return entropy.mean(), probs.max(dim=-1).values.mean()

    @staticmethod
    def _source_scale(value: float | Tensor, reference: Tensor) -> Tensor:
        if isinstance(value, Tensor):
            scale = value.to(device=reference.device, dtype=reference.dtype)
        else:
            scale = reference.new_tensor(float(value))
        if scale.ndim == 0:
            return scale
        if scale.ndim == 1 and int(scale.shape[0]) == int(reference.shape[0]):
            return scale[:, None, None]
        raise ValueError("source ablation scale must be scalar or [B]")

    @staticmethod
    def _attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Tensor | None,
        key_bias: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * (float(q.shape[-1]) ** -0.5)
        if key_bias is not None:
            scores = (
                scores + key_bias.to(device=scores.device, dtype=scores.dtype)[None, None, None, :]
            )
        if mask is not None:
            scores = scores.masked_fill(mask[None, None], torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1).to(dtype=q.dtype)
        return torch.matmul(weights, v), weights

    def forward(
        self,
        action: Tensor,
        evidence_tokens: Tensor,
        global_condition: Tensor,
        *,
        evidence_value_tokens: Tensor | None = None,
        evidence_key_bias: Tensor | None = None,
        evidence_scale: float | Tensor = 1.0,
        execution_gate: float | Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Apply one host block with an optional capacity gate at its input.

        ``execution_gate`` is deliberately applied to both residual writers
        before the native low-rank contraction.  This makes capacity a
        forward control signal (and keeps its derivative on the action path)
        instead of treating it as a cost-only statistic after the block has
        already written a full update.
        """
        before = action
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = self.action_mod(
            self.global_norm(global_condition)
        ).chunk(6, dim=-1)
        action_value = self.action_norm(action)
        modulated_action = action_value * (1.0 + scale_a[:, None]) + shift_a[:, None]
        aq, ak, av = (
            self._split(part) for part in self.action_qkv(modulated_action).chunk(3, dim=-1)
        )
        action_len = int(action.shape[1])
        self_mask = torch.triu(
            torch.ones(action_len, action_len, device=action.device, dtype=torch.bool),
            diagonal=1,
        )
        self_attended, _ = self._attention(aq, ak, av, self_mask)
        self_direction = self._unit_rms(self.self_out(self._merge(self_attended)))

        if evidence_value_tokens is None:
            evidence_value_tokens = evidence_tokens
        if tuple(evidence_value_tokens.shape) != tuple(evidence_tokens.shape):
            raise ValueError("evidence selector and value tokens must be shape-aligned")
        evidence_selector = self.evidence_norm(evidence_tokens)
        evidence_value = self.evidence_norm(evidence_value_tokens)
        eq = self._split(self.evidence_query(modulated_action))
        # The two halves of the shared projection have stable semantics across
        # every block: selector -> K and value -> V.  Previously both halves
        # were fed from ``evidence_value_tokens``, so the selector lane was only
        # shape-checked and never participated in retrieval.
        # Slice the checkpoint-compatible shared projection instead of running
        # the full 2H projection twice and discarding half of each result.
        h = self.hidden_size
        ek = self._split(
            F.linear(
                evidence_selector,
                self.evidence_kv.weight[:h],
                None if self.evidence_kv.bias is None else self.evidence_kv.bias[:h],
            )
        )
        ev = self._split(
            F.linear(
                evidence_value,
                self.evidence_kv.weight[h:],
                None if self.evidence_kv.bias is None else self.evidence_kv.bias[h:],
            )
        )
        evidence_attended, evidence_weights = self._attention(
            eq,
            ek,
            ev,
            None,
            key_bias=evidence_key_bias,
        )
        evidence_direction = self._unit_rms(self.evidence_out(self._merge(evidence_attended)))

        evidence_factor = self._source_scale(evidence_scale, action)
        evidence_anchor = evidence_direction * evidence_factor
        cross_direction = evidence_anchor
        composition_scale = math.sqrt(0.5)
        composed_direction = composition_scale * (self_direction + cross_direction)
        if execution_gate is None:
            gate = action.new_ones(action.shape[0], 1, 1)
        else:
            gate = self._source_scale(execution_gate, action)
        shared_gate = self.residual_scale_max * torch.tanh(gate_a)[:, None] * gate
        attention_update = shared_gate * self.drop(composed_direction)
        action = action + attention_update

        ffn_direction = self._unit_rms(
            self.action_ffn(
                self.action_ffn_norm(action) * (1.0 + scale_f[:, None]) + shift_f[:, None]
            )
        )
        ffn_gate = self.residual_scale_max * torch.tanh(gate_f)[:, None] * gate
        ffn_update = ffn_gate * self.drop(ffn_direction)
        action = action + ffn_update

        evidence_entropy, evidence_max = self._attention_stats(evidence_weights)
        self_contribution = composition_scale * shared_gate * self_direction
        evidence_contribution = composition_scale * shared_gate * evidence_anchor
        evidence_contribution_norm = evidence_contribution.detach().float().norm(dim=-1).mean()
        return action, {
            "action_update_norm": (action - before).detach().float().norm(dim=-1).mean(),
            "attention_update_norm": attention_update.detach().float().norm(dim=-1).mean(),
            "ffn_update_norm": ffn_update.detach().float().norm(dim=-1).mean(),
            "self_update_norm": self_contribution.detach().float().norm(dim=-1).mean(),
            "evidence_update_norm": evidence_contribution_norm,
            "evidence_attention_entropy": evidence_entropy,
            "evidence_attention_max": evidence_max,
            "residual_gate_mean": shared_gate.detach().float().abs().mean(),
            "ffn_gate_mean": ffn_gate.detach().float().abs().mean(),
            "execution_gate_mean": gate.detach().float().mean(),
            "action_token_norm": action.detach().float().norm(dim=-1).mean(),
        }

    def evidence_reader_parameters(self) -> tuple[nn.Parameter, ...]:
        modules = (self.evidence_query, self.evidence_kv, self.evidence_out)
        return tuple(parameter for module in modules for parameter in module.parameters())

class EvidenceLatentMMDiTActionDecoder(nn.Module):
    """Deterministic latent organizer followed by native-time MMDiT blocks."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        h = int(config.hidden_size)
        horizon = int(config.action_horizon)
        self.hidden_size = h
        self.horizon = horizon
        self.arm_dim = max(int(getattr(config, "arm_dim", 1)), 1)
        self.evidence_adapter = EvidenceViewAdapter(config)
        self.organizer = EvidenceConditionOrganizer(config)
        self.noisy_lift = NativeTimePhysicalActionTokenLift(config)
        self.trajectory_lift = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
        # Retain the legacy module name for checkpoint compatibility.  The
        # proposal is evidence, not a second direct action writer.
        self.trajectory_lift.requires_grad_(False)
        self.policy_delta_bridge_enabled = bool(
            int(getattr(config, "role_attnres_policy_to_mmdit", 0))
        )
        role_value_rms = (
            float(getattr(config, "role_attnres_max_value_rms", 1.0))
            if int(getattr(config, "role_residual_amplitude_contract", 0))
            else None
        )
        role_norm_floor = (
            float(getattr(config, "flow_jepa_routing_norm_floor", 0.25))
            if int(getattr(config, "flow_jepa_variance_safe_routing", 0))
            else None
        )
        self.top_policy_workspace_lift = (
            nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h, bias=False))
            if (
                int(getattr(config, "flow_jepa_role_hierarchy", 0))
                and not self.policy_delta_bridge_enabled
            )
            else None
        )
        self.policy_delta_attnres = (
            RoleDeltaAttnRes(
                h,
                int(getattr(config, "role_attnres_key_dim", 32)),
                max_sources=(
                    (
                        5
                        if int(
                            getattr(
                                config,
                                "flow_jepa_object_intent_dynamics_mainline",
                                0,
                            )
                        )
                        else int(
                            getattr(config, "flow_jepa_policy_blocks", 2)
                        )
                        + 1
                    )
                    * int(getattr(config, "action_basis_tokens", 1))
                ),
                max_value_rms=role_value_rms,
                normalization_floor=role_norm_floor,
            )
            if self.policy_delta_bridge_enabled
            else None
        )
        self.protected_detail_basis_attnres = (
            RoleDeltaAttnRes(
                h,
                int(getattr(config, "role_attnres_key_dim", 32)),
                max_sources=int(getattr(config, "action_basis_tokens", 1)),
                include_null=False,
                max_value_rms=role_value_rms,
                normalization_floor=role_norm_floor,
            )
            if self.policy_delta_bridge_enabled
            else None
        )
        self.top_policy_workspace_fixed_fusion = bool(
            int(getattr(config, "flow_jepa_policy_workspace_fixed_fusion", 0))
        )
        self.top_policy_workspace_horizon_pool = bool(
            int(getattr(config, "flow_jepa_policy_workspace_horizon_pool", 0))
        )
        self.intent_seed_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.context_seed_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.context_seed_norm.requires_grad_(False)
        self.horizon_query = nn.Parameter(torch.randn(1, horizon, h) * 0.02)
        self.register_buffer(
            "horizon_position",
            sinusoidal_positions(range(1, horizon + 1), h)[None],
            persistent=True,
        )
        depth = int(getattr(config, "latent_cvae_mmdit_depth", 3))
        self.blocks = nn.ModuleList([TimeDomainMMDiTBlock(config) for _ in range(depth)])
        self.operator_capacity_enabled = bool(
            int(getattr(config, "latent_cvae_mmdit_operator_capacity", 0))
        )
        self.execution_controller_enabled = bool(
            int(getattr(config, "latent_cvae_mmdit_execution_controller", 0))
        )
        self.dynamic_block_route_enabled = bool(
            int(getattr(config, "latent_cvae_mmdit_dynamic_block_route", 0))
        )
        self.dwell_mode = str(getattr(config, "latent_cvae_mmdit_dwell_mode", "fixed"))
        self.max_dwell = int(getattr(config, "latent_cvae_mmdit_max_dwell", 1))
        self.identity_candidate_enabled = bool(
            int(getattr(config, "latent_cvae_mmdit_identity_candidate", 1))
        )
        self._execution_eval_policy_override: str | None = None
        self._execution_capacity_override: float | None = None
        if self.dwell_mode == "fixed":
            self.max_dwell = 1
        if self.max_dwell < 1:
            raise ValueError("native evidence max dwell must be positive")
        host_rng_state = torch.get_rng_state()
        sidecar_generator = torch.Generator(device="cpu")
        sidecar_generator.manual_seed((int(torch.initial_seed()) ^ 0x92E7A11) % (2**63 - 1))
        try:
            torch.set_rng_state(sidecar_generator.get_state())
            if self.operator_capacity_enabled:
                self.operator_contractions = nn.ModuleList(
                    [
                        NestedLowRankContractionBank(
                            hidden_size=h,
                            condition_size=h,
                            stage_count=1,
                            rank=int(config.latent_cvae_mmdit_operator_rank),
                            group_count=int(config.latent_cvae_mmdit_operator_groups),
                            depth_logit_init=float(
                                config.latent_cvae_mmdit_operator_depth_logit_init
                            ),
                        )
                        for _ in range(depth)
                    ]
                )
            else:
                self.operator_contractions = nn.ModuleList()
            self.execution_controller = (
                EvidenceExecutionController(
                    config,
                    block_count=depth,
                    max_dwell=self.max_dwell,
                )
                if self.execution_controller_enabled
                else None
            )
            if self.execution_controller is not None:
                for contraction in self.operator_contractions:
                    for parameter in contraction.control_parameters():
                        parameter.requires_grad_(False)
        finally:
            torch.set_rng_state(host_rng_state)
        if self.dynamic_block_route_enabled and self.execution_controller is None:
            raise ValueError("dynamic native block routing requires the execution controller")
        self.register_buffer(
            "execution_progress", torch.zeros((), dtype=torch.float32), persistent=True
        )
        self._execution_progress_value = 0.0
        self.action_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.velocity_head = ActionOnlyPhysicalVelocityHead(config)
        self.event_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 3))
        self.motion_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 1))
        self._initialize_outputs(config)

    def _initialize_outputs(self, config: Any) -> None:
        # Match the early deterministic MMDiT decoder: action readouts start
        # weak but non-zero, while event/motion probes start neutral. This
        # keeps the migration from changing the initial flow scale merely
        # because the head class is shared with newer decoders.
        std = float(getattr(config, "latent_cvae_output_init_std", 1e-3))
        for module in self.velocity_head.output_layers():
            if std > 0.0:
                nn.init.normal_(module.weight, mean=0.0, std=std)
            else:
                nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)
        for module in (self.event_head[-1], self.motion_head[-1]):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

    def set_execution_training_step(self, global_step: int) -> float:
        """Open execution controls gradually without changing the host warm-up."""
        warmup = max(
            int(getattr(self.config, "latent_cvae_mmdit_execution_warmup_steps", 200)),
            0,
        )
        transition = max(
            int(getattr(self.config, "latent_cvae_mmdit_execution_transition_steps", 1000)),
            1,
        )
        progress = min(max((int(global_step) - warmup) / float(transition), 0.0), 1.0)
        self.execution_progress.fill_(progress)
        # The tensor buffer is the checkpoint contract.  Read the cache back
        # from it so in-process execution and restored evaluation use the same
        # float32 boundary instead of differing by Python-float rounding.
        self._execution_progress_value = float(self.execution_progress.detach().cpu())
        return self._execution_progress_value

    def load_state_dict(self, state_dict: Any, *args: Any, **kwargs: Any) -> Any:
        """Restore the non-persistent schedule cache beside its buffer.

        ``execution_progress`` is persistent so checkpoints preserve the
        learned execution phase.  The Python scalar is intentionally not a
        parameter or buffer because it is only a branch/cache value; without
        synchronizing it here, a standalone evaluator would load a non-zero
        progress buffer but still take the warm-up-only route and dwell path.
        """
        result = super().load_state_dict(state_dict, *args, **kwargs)
        self._execution_progress_value = float(self.execution_progress.detach().cpu())
        return result

    def _execution_capacity(self, learned: Tensor | None) -> Tensor:
        if learned is None:
            return torch.ones((), device=self.execution_progress.device, dtype=torch.float32)
        if not self.training and self._execution_capacity_override is not None:
            return torch.full_like(
                learned,
                float(self._execution_capacity_override),
            ).clamp(0.0, 1.0)
        progress = self.execution_progress.to(device=learned.device, dtype=learned.dtype)
        return 1.0 - progress * (1.0 - learned.clamp(0.0, 1.0))

    def set_execution_eval_ablation(
        self,
        *,
        policy: str | None = None,
        capacity_gate: float | None = None,
    ) -> None:
        """Set reversible evaluation-only execution overrides.

        The trainer uses this for matched soft/hard/neutral and capacity
        probes.  Training refuses overrides so an ablation cannot silently
        alter the learned contract.
        """

        if policy is not None and policy not in {"soft", "hard", "neutral"}:
            raise ValueError("execution eval policy must be soft, hard, or neutral")
        if capacity_gate is not None and not (0.0 <= float(capacity_gate) <= 1.0):
            raise ValueError("execution capacity override must be in [0, 1]")
        self._execution_eval_policy_override = policy
        self._execution_capacity_override = (
            None if capacity_gate is None else float(capacity_gate)
        )

    def clear_execution_eval_ablation(self) -> None:
        self._execution_eval_policy_override = None
        self._execution_capacity_override = None

    def _execution_eval_policy(self) -> str:
        if self.training:
            return "soft"
        return self._execution_eval_policy_override or str(
            getattr(self.config, "latent_cvae_mmdit_execution_eval_policy", "soft")
        )

    @staticmethod
    def _execution_value_score(value_field: Tensor, *, arm_dim: int = 1) -> Tensor:
        """Collapse the typed arm/gripper value field to candidate scores."""
        if value_field.ndim != 4 or int(value_field.shape[-1]) != 2:
            raise ValueError(
                "execution value field must be [B,candidate,horizon,2]"
            )
        arm_dim = max(int(arm_dim), 1)
        component_weight = value_field.new_tensor([float(arm_dim), 1.0]) / float(arm_dim + 1)
        return (value_field * component_weight).sum(dim=-1).mean(dim=-1)

    @staticmethod
    def _select_execution_candidate(
        value_field: Tensor,
        valid: Tensor,
        *,
        tie_tolerance: float = 1e-5,
        arm_dim: int = 1,
    ) -> Tensor:
        """Select a hard real-operation candidate with candidate-0 tie break.

        A zero-initialized value reader must reproduce the fixed host
        operation, which is one execution. Centering the predicted field here
        removes the unidentifiable common-mode component before selection.
        """
        scores = EvidenceLatentMMDiTActionDecoder._execution_value_score(
            value_field, arm_dim=arm_dim
        )
        if scores.ndim != 2 or valid.shape != scores.shape:
            raise ValueError("execution value scores and mask are misaligned")
        return select_centered_candidate(
            scores,
            valid,
            neutral_index=0,
            tie_tolerance=tie_tolerance,
        )

    @staticmethod
    def _align_tokens(value: Tensor, horizon: int) -> Tensor:
        if int(value.shape[1]) == int(horizon):
            return value
        if int(value.shape[1]) <= 0:
            raise ValueError("cannot align empty trajectory tokens")
        positions = torch.linspace(
            0.0, float(value.shape[1] - 1), int(horizon), device=value.device
        )
        left = positions.floor().long()
        right = positions.ceil().long().clamp_max(int(value.shape[1]) - 1)
        weight = (positions - left.float()).view(1, -1, 1).to(dtype=value.dtype)
        return value[:, left] * (1.0 - weight) + value[:, right] * weight

    def _lift_policy_workspace(self, policy_tokens: Tensor) -> Tensor:
        if self.top_policy_workspace_lift is None:
            raise RuntimeError("policy workspace lift is disabled")
        if self.top_policy_workspace_horizon_pool:
            basis = int(getattr(self.config, "action_basis_tokens", 1))
            expected = self.horizon * basis
            if int(policy_tokens.shape[1]) != expected:
                raise ValueError(
                    "horizon-pooled policy workspace requires "
                    f"T*basis={expected} tokens, got {policy_tokens.shape[1]}"
                )
            # Lift each basis token before pooling so LayerNorm cannot mix the
            # basis average into a new, untyped feature. The lift is bias-free
            # and shared across basis roles.
            return self.top_policy_workspace_lift(policy_tokens).reshape(
                policy_tokens.shape[0],
                self.horizon,
                basis,
                self.hidden_size,
            ).mean(dim=2)
        aligned_policy = self._align_tokens(policy_tokens, self.horizon)
        return self.top_policy_workspace_lift(aligned_policy)

    def _read_policy_delta_bank(
        self,
        action_query: Tensor,
        bank: PolicyRoleDeltaBank,
        *,
        collect_diagnostics: bool = True,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        if self.policy_delta_attnres is None:
            raise RuntimeError("typed policy-delta reader is disabled")
        bank.validate(hidden_size=self.hidden_size, horizon=self.horizon)
        values = bank.values.to(
            device=action_query.device, dtype=action_query.dtype
        )
        batch, sources, horizon, basis, hidden = values.shape
        candidates = values.permute(0, 2, 1, 3, 4).reshape(
            int(batch), int(horizon), int(sources) * int(basis), int(hidden)
        )
        routed, route_metrics = self.policy_delta_attnres(
            action_query,
            candidates,
            collect_diagnostics=collect_diagnostics,
        )
        scale = routed.new_tensor(
            float(
                getattr(
                    self.config,
                    "role_attnres_policy_to_mmdit_scale",
                    0.25,
                )
            )
        )
        routed_update = scale * routed
        protected_route_metrics: dict[str, Tensor] | None = None
        if bank.protected_detail is None:
            protected_update = torch.zeros_like(routed_update)
        else:
            # The W->P detail reader has already applied its fixed scale.
            # Preserve its basis-specific reads until the bottom action query
            # chooses among them.  This route is independent of the G/W/P
            # depth softmax and has no null candidate or learned amplitude
            # gate, so high-frequency detail cannot disappear through an
            # accidental basis mean or source competition.
            if self.protected_detail_basis_attnres is None:
                raise RuntimeError("protected detail has no basis reader")
            protected_values = bank.protected_detail.to(
                device=action_query.device, dtype=action_query.dtype
            )
            protected_update, protected_route_metrics = (
                self.protected_detail_basis_attnres(
                    action_query,
                    protected_values,
                    collect_diagnostics=collect_diagnostics,
                )
            )
        if not collect_diagnostics:
            return routed_update, protected_update, {}
        metrics = {
            "evidence_policy_delta_attnres_entropy": route_metrics["entropy"],
            "evidence_policy_delta_attnres_max": route_metrics["max"],
            "evidence_policy_delta_attnres_null_mass": route_metrics["null_mass"],
            "evidence_policy_delta_attnres_query_rms": route_metrics["query_rms"],
            "evidence_policy_delta_attnres_value_rms": route_metrics["value_rms"],
            "evidence_policy_delta_attnres_raw_value_rms": route_metrics[
                "raw_value_rms"
            ],
            "evidence_policy_delta_attnres_value_compression": route_metrics[
                "value_compression"
            ],
            "evidence_policy_delta_attnres_value_contract_enabled": route_metrics[
                "value_contract_enabled"
            ],
            "evidence_policy_delta_attnres_variance_safe_norm": route_metrics[
                "variance_safe_norm"
            ],
            "evidence_policy_delta_attnres_query_norm_denominator_min": (
                route_metrics["query_norm_denominator_min"]
            ),
            "evidence_policy_delta_attnres_value_norm_denominator_min": (
                route_metrics["value_norm_denominator_min"]
            ),
            "evidence_policy_delta_attnres_update_rms": route_metrics["update_rms"],
            "evidence_policy_delta_attnres_carrier_ratio": route_metrics[
                "carrier_ratio"
            ],
            "evidence_policy_delta_attnres_source_mass_max": route_metrics[
                "source_mass_max"
            ],
            "evidence_policy_delta_attnres_source_effective_count": (
                route_metrics["source_effective_count"]
            ),
            "evidence_policy_delta_attnres_candidate_effective_count": (
                route_metrics["candidate_effective_count"]
            ),
            "evidence_policy_delta_attnres_sample_route_std": route_metrics[
                "sample_route_std"
            ],
            "evidence_policy_delta_attnres_horizon_route_std": route_metrics[
                "query_axis_1_route_std"
            ],
            "evidence_policy_delta_attnres_fixed_scale": scale.detach().float(),
            "evidence_policy_delta_attnres_routed_update_norm": (
                routed_update.detach().float().norm(dim=-1).mean()
            ),
            "evidence_policy_delta_protected_detail_update_norm": (
                protected_update.detach().float().norm(dim=-1).mean()
            ),
        }
        if protected_route_metrics is not None:
            metrics.update(
                {
                    "evidence_protected_detail_basis_entropy": (
                        protected_route_metrics["entropy"]
                    ),
                    "evidence_protected_detail_basis_max": (
                        protected_route_metrics["max"]
                    ),
                    "evidence_protected_detail_basis_query_rms": (
                        protected_route_metrics["query_rms"]
                    ),
                    "evidence_protected_detail_basis_value_rms": (
                        protected_route_metrics["value_rms"]
                    ),
                    "evidence_protected_detail_basis_raw_value_rms": (
                        protected_route_metrics["raw_value_rms"]
                    ),
                    "evidence_protected_detail_basis_value_compression": (
                        protected_route_metrics["value_compression"]
                    ),
                    "evidence_protected_detail_basis_update_rms": (
                        protected_route_metrics["update_rms"]
                    ),
                    "evidence_protected_detail_basis_variance_safe_norm": (
                        protected_route_metrics["variance_safe_norm"]
                    ),
                    "evidence_protected_detail_basis_query_norm_denominator_min": (
                        protected_route_metrics[
                            "query_norm_denominator_min"
                        ]
                    ),
                    "evidence_protected_detail_basis_value_norm_denominator_min": (
                        protected_route_metrics[
                            "value_norm_denominator_min"
                        ]
                    ),
                    "evidence_protected_detail_basis_source_mass_max": (
                        protected_route_metrics["source_mass_max"]
                    ),
                    "evidence_protected_detail_basis_source_effective_count": (
                        protected_route_metrics["source_effective_count"]
                    ),
                    "evidence_protected_detail_basis_candidate_effective_count": (
                        protected_route_metrics["candidate_effective_count"]
                    ),
                    "evidence_protected_detail_basis_sample_route_std": (
                        protected_route_metrics["sample_route_std"]
                    ),
                    "evidence_protected_detail_basis_horizon_route_std": (
                        protected_route_metrics["query_axis_1_route_std"]
                    ),
                }
            )
            protected_source_mass = protected_route_metrics["source_mass"]
            if int(protected_source_mass.numel()) != int(basis):
                raise RuntimeError(
                    "protected detail basis reader lost basis identity"
                )
            for basis_index in range(int(basis)):
                metrics[
                    f"evidence_protected_detail_basis_mass_{basis_index}"
                ] = protected_source_mass[basis_index]
        source_mass = route_metrics["source_mass"]
        expanded_names = tuple(
            f"{name}_basis{basis_index}"
            for name in bank.source_names
            for basis_index in range(int(basis))
        )
        if int(source_mass.numel()) != len(expanded_names):
            raise RuntimeError("typed policy-delta route lost source identity")
        for index, name in enumerate(expanded_names):
            metrics[
                f"evidence_policy_delta_attnres_source_mass_{name}"
            ] = source_mass[index]
        return routed_update, protected_update, metrics

    @staticmethod
    def _time_bin_mean(
        value: Tensor, time: Tensor, low: float, high: float
    ) -> tuple[Tensor, Tensor]:
        if value.ndim != 1 or time.ndim != 1 or int(value.shape[0]) != int(time.shape[0]):
            raise ValueError("time-binned diagnostics require aligned [B] tensors")
        if high >= 1.0:
            mask = (time >= low) & (time <= high)
        else:
            mask = (time >= low) & (time < high)
        count = mask.detach().float().sum()
        if bool(mask.any()):
            mean = value[mask].detach().float().mean()
        else:
            mean = torch.zeros((), device=value.device, dtype=torch.float32)
        return mean, count

    def _apply_native_operation(
        self,
        action: Tensor,
        *,
        block_index: Tensor,
        evidence_tokens: Tensor,
        evidence_value_tokens: Tensor,
        global_condition: Tensor,
        evidence_key_bias: Tensor,
        evidence_scale: float | Tensor,
        capacity_ratios: Tensor | None,
        identity_boundary: bool,
        prepared_factors: tuple[Tensor, ...] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor], list[dict[str, Tensor]]]:
        """Execute one typed host operation for one owner per sample.

        The grouping is the important boundary: each sample uses one real
        block. Capacity owns only the ordered contraction depth; it never
        multiplies the host residual writers. The host block therefore remains
        the sole residual-amplitude owner, matching ``EvidenceExecutionOutput``.
        """
        batch = int(action.shape[0])
        if tuple(block_index.shape) != (batch,):
            raise ValueError("native operation block ids must be [B]")
        if tuple(evidence_value_tokens.shape) != tuple(evidence_tokens.shape):
            raise ValueError("native evidence selector/value tokens are misaligned")
        if capacity_ratios is not None and tuple(capacity_ratios.shape) not in {
            (batch,),
            (batch, len(self.blocks)),
        }:
            raise ValueError("native operation capacity ratios must be [B] or [B,depth]")
        if prepared_factors is not None and len(prepared_factors) != len(self.blocks):
            raise ValueError("native operation factors must match the block repertoire")
        result = torch.empty_like(action)
        metric_rows: list[dict[str, Tensor]] = []
        contraction_rows: list[dict[str, Tensor]] = []
        for owner, block in enumerate(self.blocks):
            rows = torch.nonzero(block_index == owner, as_tuple=False).flatten()
            if int(rows.numel()) == 0:
                continue
            owned_input = action.index_select(0, rows)
            if self.operator_capacity_enabled:
                capacity = (
                    torch.ones(int(rows.numel()), device=action.device, dtype=action.dtype)
                    if capacity_ratios is None
                    else (
                        capacity_ratios.index_select(0, rows)
                        if capacity_ratios.ndim == 1
                        else capacity_ratios.index_select(0, rows)[:, owner]
                    ).to(dtype=action.dtype)
                )
            else:
                capacity = None
            owned_action, block_metrics = block(
                owned_input,
                evidence_tokens.index_select(0, rows),
                global_condition.index_select(0, rows),
                evidence_value_tokens=evidence_value_tokens.index_select(0, rows),
                evidence_key_bias=evidence_key_bias,
                evidence_scale=evidence_scale,
                execution_gate=None,
            )
            raw_update = owned_action - owned_input
            if self.operator_capacity_enabled:
                assert capacity is not None
                contraction = self.operator_contractions[owner]
                contracted_update, contraction_metrics = contraction(
                    raw_update,
                    global_condition.index_select(0, rows),
                    torch.zeros(int(rows.numel()), device=action.device, dtype=torch.long),
                    contraction_progress=self.execution_progress,
                    prepared_factors=(
                        None
                        if identity_boundary
                        else (
                            contraction.prepare_factors()
                            if prepared_factors is None
                            else prepared_factors[owner]
                        )
                    ),
                    depth_ratio_override=capacity,
                    # Keep the ordered capacity continuous in the training
                    # graph. Hardware-sized hard groups are an evaluation
                    # concern, not the native gradient contract.
                    binary_group_selection=False,
                    identity_bypass=identity_boundary,
                )
                owned_action = owned_input + contracted_update
                contraction_rows.append(contraction_metrics)
            result.index_copy_(0, rows, owned_action)
            metric_rows.append(block_metrics)
        if not metric_rows:
            raise RuntimeError("native operation has no valid block owner")
        names = tuple(metric_rows[0])
        metrics = {
            name: torch.stack([row[name] for row in metric_rows]).mean()
            for name in names
        }
        return result, metrics, contraction_rows

    @staticmethod
    def _select_scale_rows(
        value: float | Tensor,
        rows: Tensor,
        *,
        batch: int,
    ) -> float | Tensor:
        if not isinstance(value, Tensor) or value.ndim == 0:
            return value
        if tuple(value.shape) != (batch,):
            raise ValueError("native evidence scale must be scalar or [B]")
        return value.index_select(0, rows)

    @staticmethod
    def _mean_metric_rows(rows: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        if not rows:
            raise RuntimeError("a committed native operation produced no metrics")
        names = tuple(rows[0])
        if any(tuple(row) != names for row in rows[1:]):
            raise RuntimeError("native operation metric schemas are inconsistent")
        return {name: torch.stack([row[name] for row in rows]).mean() for name in names}

    def _apply_selected_native_operations(
        self,
        action: Tensor,
        *,
        block_index: Tensor,
        repeat_count: Tensor,
        evidence_tokens: Tensor,
        evidence_value_tokens: Tensor,
        global_condition: Tensor,
        evidence_key_bias: Tensor,
        evidence_scale: float | Tensor,
        capacity_ratios: Tensor | None,
        identity_boundary: bool,
        prepared_factors: tuple[Tensor, ...] | None,
    ) -> tuple[Tensor, dict[str, Tensor], list[dict[str, Tensor]]]:
        """Execute only the hard operation selected for each sample.

        Repetition is grouped by active rows.  No unselected candidate is
        present in this graph, so the task loss cannot update a block merely
        because it was considered by the value reader.
        """
        batch = int(action.shape[0])
        if tuple(block_index.shape) != (batch,) or tuple(repeat_count.shape) != (batch,):
            raise ValueError("selected native block and repeat count must be [B]")
        if bool(((repeat_count < 1) | (repeat_count > self.max_dwell)).any()):
            raise ValueError("selected native repeat count is outside the dwell repertoire")
        result = action
        metric_rows: list[dict[str, Tensor]] = []
        contraction_rows: list[dict[str, Tensor]] = []
        for repeat_index in range(self.max_dwell):
            rows = torch.nonzero(repeat_count > repeat_index, as_tuple=False).flatten()
            if int(rows.numel()) == 0:
                continue
            row_capacity = (
                None if capacity_ratios is None else capacity_ratios.index_select(0, rows)
            )
            updated, metrics, contractions = self._apply_native_operation(
                result.index_select(0, rows),
                block_index=block_index.index_select(0, rows),
                evidence_tokens=evidence_tokens.index_select(0, rows),
                evidence_value_tokens=evidence_value_tokens.index_select(0, rows),
                global_condition=global_condition.index_select(0, rows),
                evidence_key_bias=evidence_key_bias,
                evidence_scale=self._select_scale_rows(
                    evidence_scale, rows, batch=batch
                ),
                capacity_ratios=row_capacity,
                identity_boundary=identity_boundary,
                prepared_factors=prepared_factors,
            )
            result = result.index_copy(0, rows, updated)
            metric_rows.append(metrics)
            contraction_rows.extend(contractions)
        return result, self._mean_metric_rows(metric_rows), contraction_rows

    def _probe_native_candidates(
        self,
        action: Tensor,
        *,
        prediction_reference: Tensor,
        candidate_blocks: Tensor,
        candidate_repeats: Tensor,
        candidate_mask: Tensor,
        evidence_tokens: Tensor,
        evidence_value_tokens: Tensor,
        global_condition: Tensor,
        evidence_key_bias: Tensor,
        evidence_scale: float | Tensor,
        capacity_ratios: Tensor | None,
        identity_boundary: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Evaluate candidate targets without entering the committed graph.

        Candidate probes are a training-only teacher for the value reader.
        Evaluation performs no probes, so learned dwell and block routing
        translate into real wall-clock execution rather than post-hoc choice
        among already computed actions.
        """
        batch, candidate_count = candidate_blocks.shape
        if tuple(candidate_repeats.shape) != (batch, candidate_count):
            raise ValueError("candidate repeat ids must match candidate blocks")
        if tuple(candidate_mask.shape) != (batch, candidate_count):
            raise ValueError("candidate mask must match candidate blocks")
        expected_prediction_shape = (
            batch,
            self.horizon,
            int(self.config.physical_action_dim),
        )
        if tuple(prediction_reference.shape) != expected_prediction_shape:
            raise ValueError(
                "candidate prediction reference must match the velocity-head output"
            )
        # The action state can remain FP32 under autocast while the velocity
        # head emits BF16.  Reuse the already-computed prefix prediction as the
        # output contract instead of guessing the head dtype from its input.
        predictions = prediction_reference[:, None].expand(
            -1, candidate_count, -1, -1
        ).clone()
        if not self.training:
            return (
                predictions,
                torch.zeros_like(candidate_mask),
                torch.zeros((), device=action.device, dtype=torch.float32),
            )
        probe_mask = candidate_mask.to(device=action.device, dtype=torch.bool)
        operation_mask = probe_mask & (candidate_blocks < len(self.blocks))
        operation_rows = (
            operation_mask.float() * (candidate_repeats.detach().float() + 1.0)
        ).sum(dim=-1)
        modules: list[nn.Module] = [*self.blocks, *self.operator_contractions]
        with deterministic_module_probe(*modules), torch.no_grad():
            probe_factors = (
                None
                if not self.operator_capacity_enabled or identity_boundary
                else tuple(bank.prepare_factors() for bank in self.operator_contractions)
            )
            for candidate_index in range(candidate_count):
                rows = torch.nonzero(
                    operation_mask[:, candidate_index], as_tuple=False
                ).flatten()
                if int(rows.numel()) == 0:
                    continue
                row_capacity = (
                    None
                    if capacity_ratios is None
                    else capacity_ratios.detach().index_select(0, rows)
                )
                candidate_action, _, _ = self._apply_selected_native_operations(
                    action.index_select(0, rows),
                    block_index=candidate_blocks.index_select(0, rows)[:, candidate_index],
                    repeat_count=(
                        candidate_repeats.index_select(0, rows)[:, candidate_index] + 1
                    ),
                    evidence_tokens=evidence_tokens.index_select(0, rows),
                    evidence_value_tokens=evidence_value_tokens.index_select(0, rows),
                    global_condition=global_condition.index_select(0, rows),
                    evidence_key_bias=evidence_key_bias,
                    evidence_scale=self._select_scale_rows(
                        evidence_scale, rows, batch=batch
                    ),
                    capacity_ratios=row_capacity,
                    identity_boundary=identity_boundary,
                    prepared_factors=probe_factors,
                )
                candidate_velocity = self.velocity_head(self.action_norm(candidate_action))
                predictions[:, candidate_index].index_copy_(0, rows, candidate_velocity)
        return predictions.detach(), probe_mask, operation_rows.mean()

    def _run_differentiable_native_candidates(
        self,
        action: Tensor,
        *,
        baseline_velocity: Tensor,
        candidate_blocks: Tensor,
        candidate_repeats: Tensor,
        candidate_mask: Tensor,
        evidence_tokens: Tensor,
        evidence_value_tokens: Tensor,
        global_condition: Tensor,
        evidence_key_bias: Tensor,
        evidence_scale: float | Tensor,
        capacity_ratios: Tensor | None,
        identity_boundary: bool,
        prepared_factors: tuple[Tensor, ...] | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Run legal candidates as an attached training-time action chart.

        The hard deployment path still commits one operation. During learned
        training, however, every legal candidate is evaluated with gradients
        and the caller forms a soft action mixture from the value scores. This
        lets the task loss flow naturally into route/dwell scores instead of
        relying on an auxiliary value loss to cross an integer argmin.
        """
        batch, candidate_count = candidate_blocks.shape
        if tuple(candidate_repeats.shape) != (batch, candidate_count):
            raise ValueError("candidate repeat ids must match candidate blocks")
        if tuple(candidate_mask.shape) != (batch, candidate_count):
            raise ValueError("candidate mask must match candidate blocks")
        mask = candidate_mask.to(device=action.device, dtype=torch.bool)
        operation_mask = mask & (candidate_blocks < len(self.blocks))
        operation_rows = (
            operation_mask.float() * (candidate_repeats.detach().float() + 1.0)
        )
        candidate_actions: list[Tensor] = []
        for candidate_index in range(candidate_count):
            rows = torch.nonzero(
                operation_mask[:, candidate_index], as_tuple=False
            ).flatten()
            candidate_action = action.clone()
            if int(rows.numel()) > 0:
                row_capacity = (
                    None
                    if capacity_ratios is None
                    else capacity_ratios.index_select(0, rows)
                )
                updated, _, _ = self._apply_selected_native_operations(
                    action.index_select(0, rows),
                    block_index=candidate_blocks.index_select(0, rows)[:, candidate_index],
                    repeat_count=(
                        candidate_repeats.index_select(0, rows)[:, candidate_index] + 1
                    ),
                    evidence_tokens=evidence_tokens.index_select(0, rows),
                    evidence_value_tokens=evidence_value_tokens.index_select(0, rows),
                    global_condition=global_condition.index_select(0, rows),
                    evidence_key_bias=evidence_key_bias,
                    evidence_scale=self._select_scale_rows(
                        evidence_scale, rows, batch=batch
                    ),
                    capacity_ratios=row_capacity,
                    identity_boundary=identity_boundary,
                    prepared_factors=prepared_factors,
                )
                candidate_action = candidate_action.index_copy(0, rows, updated)
            candidate_actions.append(candidate_action)
        action_stack = torch.stack(candidate_actions, dim=1)
        candidate_velocity = self.velocity_head(
            self.action_norm(action_stack.reshape(-1, *action_stack.shape[2:]))
        ).reshape(
            batch,
            candidate_count,
            self.horizon,
            int(self.config.physical_action_dim),
        )
        # The terminal/no-op candidate is semantically the already committed
        # prefix. Re-evaluating the same action through a differently shaped
        # BF16 LayerNorm/Linear batch produced ~1e-3 numerical drift in V96/97,
        # so the value target no longer compared an exact identity candidate.
        # Reuse the existing prefix tensor: this changes no operation candidate
        # and preserves its original gradient path exactly.
        terminal = candidate_blocks == len(self.blocks)
        candidate_velocity = torch.where(
            terminal[:, :, None, None],
            baseline_velocity[:, None].to(dtype=candidate_velocity.dtype),
            candidate_velocity,
        )
        return action_stack, candidate_velocity, mask, operation_rows.mean()

    def _soft_execution_probabilities(
        self,
        value_field: Tensor,
        candidate_mask: Tensor,
        candidate_blocks: Tensor,
    ) -> Tensor:
        """Convert attached candidate values into an FP32 policy distribution.

        Action features follow the surrounding autocast dtype, but execution
        probabilities are recurrent state: they are accumulated across
        decisions and meet FP32 controller/value-reader islands.  Owning that
        state in FP32 prevents its dtype from depending on whether the current
        action came from teacher-forced training (FP32) or deploy sampling
        (usually BF16).
        """
        scores = self._execution_value_score(
            value_field.float(), arm_dim=self.arm_dim
        )
        temperature = max(
            float(
                getattr(
                    self.config,
                    "latent_cvae_mmdit_execution_soft_temperature",
                    1.0,
                )
            ),
            1e-3,
        )
        logits = -scores / temperature
        terminal = candidate_blocks == len(self.blocks)
        prior_weight = float(
            getattr(self.config, "latent_cvae_mmdit_terminal_prior_weight", 0.25)
        )
        logits = logits + terminal.to(dtype=logits.dtype) * math.log(prior_weight)
        logits = logits.masked_fill(
            ~candidate_mask.bool(), torch.finfo(logits.dtype).min
        )
        return torch.softmax(logits, dim=-1)

    @staticmethod
    def _mix_candidate_actions(
        candidate_action_stack: Tensor, selection_probabilities: Tensor
    ) -> Tensor:
        """Mix ``[B,C,H,D]`` candidate actions with ``[B,C]`` probabilities."""
        if candidate_action_stack.ndim != 4:
            raise ValueError("candidate action stack must be [B,candidate,horizon,hidden]")
        if selection_probabilities.ndim != 2:
            raise ValueError("candidate selection probabilities must be [B,candidate]")
        if tuple(candidate_action_stack.shape[:2]) != tuple(
            selection_probabilities.shape
        ):
            raise ValueError(
                "candidate actions and selection probabilities must share "
                "batch/candidate axes"
            )
        return (
            candidate_action_stack
            * selection_probabilities[:, :, None, None].to(
                dtype=candidate_action_stack.dtype
            )
        ).sum(dim=1)

    def _global_execution_candidate_chart(
        self, *, batch: int, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        """Return one stable block-by-dwell chart for every execution phase.

        Warm-up, learned training, and deployment all read the same candidate
        identities.  A schedule may change how much probability the learned
        policy owns, but it never changes the meaning or size of the chart.
        """

        blocks = torch.arange(len(self.blocks), device=device, dtype=torch.long)
        repeats = torch.arange(self.max_dwell, device=device, dtype=torch.long)
        candidate_blocks = blocks.repeat_interleave(self.max_dwell)
        candidate_repeats = repeats.repeat(len(self.blocks))
        if self.identity_candidate_enabled:
            candidate_blocks = torch.cat(
                [candidate_blocks, candidate_blocks.new_tensor([len(self.blocks)])]
            )
            candidate_repeats = torch.cat(
                [candidate_repeats, candidate_repeats.new_zeros(1)]
            )
        return (
            candidate_blocks[None].expand(int(batch), -1),
            candidate_repeats[None].expand(int(batch), -1),
        )

    def _mean_field_execution_policy(
        self,
        value_field: Tensor,
        pointer_mass: Tensor,
        terminal_logit_bias: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Propagate a differentiable monotonic execution pointer.

        Pointer ``j`` means that block ``j`` is the next uncommitted host
        operation.  It may execute any still-uncommitted block or terminate;
        after an operation, the pointer advances past that block.  The last
        pointer is terminal and preserves the current action.  No integer
        choice from this policy is fed back into the training graph.
        """

        batch = int(value_field.shape[0])
        depth = len(self.blocks)
        dwell = self.max_dwell
        operation_candidate_count = depth * dwell
        candidate_count = operation_candidate_count + 1
        if tuple(value_field.shape[:2]) != (batch, candidate_count):
            raise ValueError("global execution value chart has the wrong shape")
        if tuple(pointer_mass.shape) != (batch, depth + 1):
            raise ValueError("execution pointer mass must be [B,depth+1]")
        if terminal_logit_bias is not None and tuple(
            terminal_logit_bias.shape
        ) != (batch,):
            raise ValueError("execution terminal logit bias must be [B]")

        # The policy plane owns probabilities, entropy and pointer occupancy in
        # FP32.  Do not inherit their dtype from the action stream: sampling
        # seeds actions from BF16 visual/noise tensors while training seeds them
        # from FP32 target actions.  Letting ``pointer_mass`` follow that split
        # makes ordinary arithmetic silently promote to FP32 and then fail at
        # strict indexed accumulations such as ``index_add``.
        policy_pointer_mass = pointer_mass.float()
        scores = self._execution_value_score(
            value_field.float(), arm_dim=self.arm_dim
        )
        temperature = max(
            float(
                getattr(
                    self.config,
                    "latent_cvae_mmdit_execution_soft_temperature",
                    1.0,
                )
            ),
            1e-3,
        )
        global_probabilities = scores.new_zeros(batch, candidate_count)
        conditional_entropy = scores.new_zeros(batch)
        skip_probability = scores.new_zeros(batch)
        terminal_probability = scores.new_zeros(batch)
        terminal_candidate = operation_candidate_count
        prior_weight = float(
            getattr(self.config, "latent_cvae_mmdit_terminal_prior_weight", 0.25)
        )
        for pointer_index in range(depth):
            local_indices = list(range(pointer_index * dwell, operation_candidate_count))
            local_indices.append(terminal_candidate)
            index = torch.as_tensor(local_indices, device=scores.device, dtype=torch.long)
            local_logits = -scores.index_select(1, index) / temperature
            local_logits[:, -1] = local_logits[:, -1] + math.log(prior_weight)
            if terminal_logit_bias is not None:
                local_logits[:, -1] = (
                    local_logits[:, -1] + terminal_logit_bias.float()
                )
            local_probabilities = torch.softmax(local_logits, dim=-1)
            source_mass = policy_pointer_mass[:, pointer_index]
            contribution = source_mass[:, None] * local_probabilities
            global_probabilities = global_probabilities.index_add(1, index, contribution)
            conditional_entropy = conditional_entropy + source_mass * (
                -(
                    local_probabilities
                    * local_probabilities.clamp_min(1e-8).log()
                ).sum(dim=-1)
            )
            local_blocks = torch.div(index[:-1], dwell, rounding_mode="floor")
            skip_mask = local_blocks > pointer_index
            if bool(skip_mask.any()):
                skip_probability = skip_probability + contribution[:, :-1][:, skip_mask].sum(dim=-1)
            terminal_probability = terminal_probability + contribution[:, -1]

        terminal_mass = policy_pointer_mass[:, depth]
        next_pointer_mass = scores.new_zeros(batch, depth + 1)
        terminal_index = torch.as_tensor([depth], device=scores.device)
        next_pointer_mass = next_pointer_mass.index_add(
            1, terminal_index, (terminal_mass + terminal_probability)[:, None]
        )
        for block_index in range(depth):
            block_mass = global_probabilities[
                :, block_index * dwell : (block_index + 1) * dwell
            ].sum(dim=-1)
            destination = torch.as_tensor(
                [block_index + 1], device=scores.device
            )
            next_pointer_mass = next_pointer_mass.index_add(
                1, destination, block_mass[:, None]
            )
        return (
            global_probabilities,
            next_pointer_mass,
            conditional_entropy,
            skip_probability,
            terminal_probability,
        )

    def _scheduled_hard_policy(
        self,
        value_field: Tensor,
        candidate_mask: Tensor,
        neutral_index: Tensor,
        candidate_blocks: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Hard deployment policy derived from the same scheduled soft chart."""

        batch, candidate_count = candidate_mask.shape
        if tuple(value_field.shape[:2]) != (batch, candidate_count):
            raise ValueError("hard execution value field and mask are misaligned")
        if tuple(neutral_index.shape) != (batch,):
            raise ValueError("hard execution neutral indices must be [B]")
        safe_mask = candidate_mask.bool().clone()
        inactive = ~safe_mask.any(dim=-1)
        if bool(inactive.any()):
            safe_mask[inactive, 0] = True
            neutral_index = torch.where(inactive, torch.zeros_like(neutral_index), neutral_index)
        learned = self._soft_execution_probabilities(
            value_field, safe_mask, candidate_blocks
        )
        neutral = torch.nn.functional.one_hot(
            neutral_index, num_classes=candidate_count
        ).to(dtype=learned.dtype)
        progress = self.execution_progress.to(device=learned.device, dtype=learned.dtype)
        scheduled = (1.0 - progress) * neutral + progress * learned
        selected = scheduled.argmax(dim=-1)
        selected = torch.where(inactive, torch.zeros_like(selected), selected)
        return scheduled, selected

    def _run_dynamic_execution(
        self,
        *,
        action: Tensor,
        global_condition: Tensor,
        time_hidden: Tensor,
        evidence_tokens: Tensor,
        evidence_value_tokens: Tensor,
        evidence_bias: Tensor,
        evidence_scale: float | Tensor,
        execution_terminal_probability: Tensor | None = None,
        execution_terminal_uncertainty: Tensor | None = None,
    ) -> dict[str, Any]:
        """Run one continuous monotonic execution contract.

        Training and default evaluation keep a differentiable distribution
        over the next uncommitted block or terminal identity.  Hard and neutral
        execution are explicit evaluation ablations; no hard decision is fed
        back into the training state.
        """
        if self.execution_controller is None:
            raise RuntimeError("dynamic native execution requires a controller")
        batch = int(action.shape[0])
        device = action.device
        depth = len(self.blocks)
        dwell = self.max_dwell
        candidate_blocks, candidate_repeats = self._global_execution_candidate_chart(
            batch=batch, device=device
        )
        candidate_count = int(candidate_blocks.shape[1])
        operation_candidate_count = depth * dwell
        terminal_candidate = operation_candidate_count
        terminal_logit_bias: Tensor | None = None
        if execution_terminal_probability is not None:
            if tuple(execution_terminal_probability.shape) != (batch, 1):
                raise ValueError("execution terminal probability must be [B,1]")
            if execution_terminal_uncertainty is None or tuple(
                execution_terminal_uncertainty.shape
            ) != (batch, 1):
                raise ValueError("execution terminal uncertainty must be [B,1]")
            probability = execution_terminal_probability.float().clamp(
                1e-4, 1.0 - 1e-4
            )[:, 0]
            uncertainty = execution_terminal_uncertainty.float().clamp_min(0.0)[:, 0]
            terminal_logit_bias = (
                0.10
                * torch.tanh(torch.logit(probability) / 4.0)
                / (1.0 + uncertainty)
            )
        eval_policy = self._execution_eval_policy()
        soft_contract = eval_policy in {"soft", "neutral"}
        controller_state: Tensor | None = None
        feedback: Tensor | None = None
        block_rows: list[dict[str, Tensor]] = []
        capacity_rows: list[Tensor] = []
        capacity_weight_rows: list[Tensor] = []
        dwell_rows: list[Tensor] = []
        execution_cost_rows: list[Tensor] = []
        selection_entropy_rows: list[Tensor] = []
        selection_max_probability_rows: list[Tensor] = []
        learned_selection_entropy_rows: list[Tensor] = []
        controller_rows: list[dict[str, Tensor]] = []
        contraction_rows: list[dict[str, Tensor]] = []
        route_rows: list[Tensor] = []
        hard_route_rows: list[Tensor] = []
        hard_dwell_rows: list[Tensor] = []
        terminal_probability_rows: list[Tensor] = []
        hard_terminal_rows: list[Tensor] = []
        prefix_velocity_rows: list[Tensor] = [self.velocity_head(self.action_norm(action))]
        candidate_prediction_rows: list[Tensor] = []
        candidate_mask_rows: list[Tensor] = []
        value_field_rows: list[Tensor] = []
        value_mask_rows: list[Tensor] = []
        probe_operation_rows: list[Tensor] = []
        block_visit_rows: list[Tensor] = []
        identity_boundary = self._execution_progress_value <= 0.0
        prepared_factors = (
            None
            if not self.operator_capacity_enabled
            else tuple(bank.prepare_factors() for bank in self.operator_contractions)
        )
        if soft_contract:
            # Pointer occupancy is policy state, not an action feature.  Keep it
            # FP32 across every decision even when the sampled action is BF16.
            pointer_mass = torch.zeros(
                batch, depth + 1, device=device, dtype=torch.float32
            )
            pointer_mass[:, 0] = 1.0
        else:
            hard_pointer = torch.zeros(batch, device=device, dtype=torch.long)

        for decision_index in range(depth):
            block_input = action
            # The controller advances on a fixed decision clock.  Candidate
            # block identity already lives on the global value chart, so
            # feeding ``argmax(pointer_mass)`` back here would add a hidden
            # hard transition to the otherwise differentiable training path.
            controller_block = torch.full(
                (batch,), decision_index, device=device, dtype=torch.long
            )
            if not soft_contract:
                active = hard_pointer < depth
                if not bool(active.any()):
                    zero = torch.zeros((), device=device, dtype=torch.float32)
                    idle_metrics = {
                        "action_update_norm": zero,
                        "attention_update_norm": zero,
                        "ffn_update_norm": zero,
                        "self_update_norm": zero,
                        "evidence_update_norm": zero,
                        "evidence_attention_entropy": zero,
                        "evidence_attention_max": zero,
                        "residual_gate_mean": zero,
                        "ffn_gate_mean": zero,
                        "execution_gate_mean": torch.ones(
                            (), device=device, dtype=torch.float32
                        ),
                        "action_token_norm": action.detach().float().norm(dim=-1).mean(),
                    }
                    idle_velocity = self.velocity_head(self.action_norm(action))
                    candidate_prediction_rows.append(
                        idle_velocity[:, None].expand(-1, candidate_count, -1, -1).detach()
                    )
                    empty_mask = torch.zeros(
                        batch, candidate_count, device=device, dtype=torch.bool
                    )
                    candidate_mask_rows.append(empty_mask)
                    value_field_rows.append(
                        action.new_zeros(batch, candidate_count, self.horizon, 2)
                    )
                    value_mask_rows.append(empty_mask)
                    probe_operation_rows.append(zero)
                    block_visit_rows.append(
                        torch.zeros(depth, device=device, dtype=torch.float32)
                    )
                    dwell_rows.append(zero)
                    hard_dwell_rows.append(zero)
                    execution_cost_rows.append(zero)
                    route_rows.append(zero)
                    hard_route_rows.append(zero)
                    terminal_probability_rows.append(torch.ones_like(zero))
                    hard_terminal_rows.append(torch.ones_like(zero))
                    selection_entropy_rows.append(zero)
                    selection_max_probability_rows.append(
                        torch.ones((), device=device, dtype=torch.float32)
                    )
                    learned_selection_entropy_rows.append(zero)
                    block_rows.append(idle_metrics)
                    prefix_velocity_rows.append(idle_velocity)
                    feedback = torch.zeros_like(action)
                    continue
            control = self.execution_controller(
                state=controller_state,
                global_condition=global_condition,
                time_context=time_hidden,
                action_tokens=action,
                evidence_tokens=evidence_tokens,
                evidence_value_tokens=evidence_value_tokens,
                evidence_key_bias=evidence_bias,
                feedback=feedback,
                block_index=controller_block,
            )
            controller_state = control.state
            controller_rows.append(control.metrics)
            capacity_ratios = self._execution_capacity(control.capacity_ratios)
            value_field = self.execution_controller.predict_execution_value(
                state=controller_state,
                global_condition=global_condition,
                time_context=time_hidden,
                evidence_tokens=evidence_tokens,
                evidence_value_tokens=evidence_value_tokens,
                evidence_key_bias=evidence_bias,
                action_tokens=block_input,
                block_index=controller_block,
                candidate_block_index=candidate_blocks,
                candidate_repeat_index=candidate_repeats,
            )

            if soft_contract:
                pointer_before = pointer_mass
                neutral_block = torch.full(
                    (batch,), decision_index, device=device, dtype=torch.long
                )
                neutral_repeat = torch.ones(batch, device=device, dtype=torch.long)
                neutral_action, neutral_metrics, neutral_contractions = (
                    self._apply_selected_native_operations(
                        block_input,
                        block_index=neutral_block,
                        repeat_count=neutral_repeat,
                        evidence_tokens=evidence_tokens,
                        evidence_value_tokens=evidence_value_tokens,
                        global_condition=global_condition,
                        evidence_key_bias=evidence_bias,
                        evidence_scale=evidence_scale,
                        capacity_ratios=capacity_ratios,
                        identity_boundary=identity_boundary,
                        prepared_factors=prepared_factors,
                    )
                )
                contraction_rows.extend(neutral_contractions)
                modules: list[nn.Module] = [*self.blocks, *self.operator_contractions]
                # Candidate evaluation is a differentiable chart, not a source
                # of extra dropout noise or host RNG drift.  The neutral branch
                # therefore remains bitwise identical at progress zero.
                with deterministic_module_probe(*modules):
                    legal_candidate_mask = (
                        (candidate_blocks >= decision_index)
                        & (candidate_blocks <= depth)
                    )
                    candidate_action_stack, candidate_velocity_stack, probe_mask, probe_operation_count = (
                        self._run_differentiable_native_candidates(
                            block_input,
                            baseline_velocity=prefix_velocity_rows[-1],
                            candidate_blocks=candidate_blocks,
                            candidate_repeats=candidate_repeats,
                            candidate_mask=legal_candidate_mask,
                            evidence_tokens=evidence_tokens,
                            evidence_value_tokens=evidence_value_tokens,
                            global_condition=global_condition,
                            evidence_key_bias=evidence_bias,
                            evidence_scale=evidence_scale,
                            capacity_ratios=capacity_ratios,
                            identity_boundary=identity_boundary,
                            prepared_factors=prepared_factors,
                        )
                    )
                (
                    learned_probabilities,
                    learned_next_pointer,
                    learned_entropy,
                    learned_skip,
                    learned_terminal,
                ) = (
                    self._mean_field_execution_policy(
                        value_field,
                        pointer_before,
                        terminal_logit_bias=terminal_logit_bias,
                    )
                )
                terminal_mass = pointer_before[:, depth]
                learned_action = self._mix_candidate_actions(
                    candidate_action_stack, learned_probabilities
                ) + terminal_mass[:, None, None].to(dtype=block_input.dtype) * block_input
                action_progress = self.execution_progress.to(
                    device=action.device, dtype=action.dtype
                )
                policy_progress = self.execution_progress.to(
                    device=action.device, dtype=torch.float32
                )
                if eval_policy == "neutral":
                    action_progress = torch.zeros_like(action_progress)
                    policy_progress = torch.zeros_like(policy_progress)
                action = (
                    (1.0 - action_progress) * neutral_action
                    + action_progress * learned_action
                )

                neutral_pointer = torch.zeros(
                    batch, depth + 1, device=device, dtype=torch.float32
                )
                neutral_pointer[:, decision_index + 1] = 1.0
                pointer_mass = (
                    (1.0 - policy_progress) * neutral_pointer
                    + policy_progress * learned_next_pointer
                )

                neutral_candidate = decision_index * dwell
                neutral_probability = torch.nn.functional.one_hot(
                    torch.full(
                        (batch,), neutral_candidate, device=device, dtype=torch.long
                    ),
                    num_classes=candidate_count,
                ).to(dtype=learned_probabilities.dtype)
                executed_probabilities = (
                    (1.0 - policy_progress) * neutral_probability
                    + policy_progress * learned_probabilities
                )
                complete_probabilities = executed_probabilities.clone()
                complete_probabilities[:, terminal_candidate] = (
                    complete_probabilities[:, terminal_candidate]
                    + policy_progress * terminal_mass
                )
                selection_entropy_rows.append(
                    -(
                        complete_probabilities.float()
                        * complete_probabilities.float().clamp_min(1e-8).log()
                    ).sum(dim=-1).mean()
                )
                selection_max_probability_rows.append(
                    complete_probabilities.detach().float().amax(dim=-1).mean()
                )
                learned_selection_entropy_rows.append(learned_entropy.float().mean())

                safe_candidate_blocks = candidate_blocks.clamp_max(depth - 1)
                candidate_capacity = torch.gather(
                    capacity_ratios,
                    1,
                    safe_candidate_blocks,
                ).float()
                operation_candidate = candidate_blocks < depth
                candidate_capacity = candidate_capacity * operation_candidate.float()
                repeat_count = candidate_repeats.detach().float() + 1.0
                repeat_count = repeat_count * operation_candidate.float()
                operation_probabilities = executed_probabilities[:, :operation_candidate_count]
                operation_mass = operation_probabilities.float().sum(dim=-1)
                expected_capacity_sum = (
                    executed_probabilities.float() * candidate_capacity
                ).sum(dim=-1)
                # Capacity is conditional on an operation actually running.
                # Terminal/no-op mass belongs in dwell and cost, but must not
                # masquerade as removed rank.
                capacity_rows.append(expected_capacity_sum.sum())
                capacity_weight_rows.append(operation_mass.sum())
                expected_dwell = (
                    executed_probabilities.float() * repeat_count
                ).sum(dim=-1)
                dwell_rows.append(expected_dwell.mean())
                execution_cost_rows.append(
                    (
                        executed_probabilities.float()
                        * candidate_capacity
                        * repeat_count
                    ).sum(dim=-1).mean()
                )
                route_rows.append((policy_progress * learned_skip).mean())
                terminal_probability_rows.append(
                    complete_probabilities[:, terminal_candidate].detach().float().mean()
                )

                active_pointer = pointer_before[:, :depth].sum(dim=-1) > 0.0
                audit_pointer = pointer_before.argmax(dim=-1).clamp_max(depth - 1)
                audit_mask = (
                    (
                        (candidate_blocks >= audit_pointer[:, None])
                        & (candidate_blocks < depth)
                    )
                    | (candidate_blocks == depth)
                ) & active_pointer[:, None]
                audit_neutral = audit_pointer * dwell
                _, audit_selected = self._scheduled_hard_policy(
                    value_field, audit_mask, audit_neutral, candidate_blocks
                )
                audit_block = candidate_blocks.gather(1, audit_selected[:, None]).squeeze(1)
                audit_repeat = candidate_repeats.gather(1, audit_selected[:, None]).squeeze(1)
                audit_terminal = audit_block == depth
                hard_terminal_rows.append(
                    ((~active_pointer) | audit_terminal).detach().float().mean()
                )
                hard_dwell_rows.append(
                    torch.where(
                        active_pointer & ~audit_terminal,
                        audit_repeat + 1,
                        torch.zeros_like(audit_repeat),
                    ).detach().float().mean()
                )
                hard_route_rows.append(
                    (
                        active_pointer
                        & ~audit_terminal
                        & (audit_block > audit_pointer)
                    ).detach().float().mean()
                )
                block_visit = []
                for block_index in range(depth):
                    block_visit.append(
                        executed_probabilities[
                            :, block_index * dwell : (block_index + 1) * dwell
                        ].sum(dim=-1).mean()
                    )
                block_visit_rows.append(torch.stack(block_visit))
                committed_metrics = dict(neutral_metrics)
                committed_metrics["action_update_norm"] = (
                    action - block_input
                ).detach().float().norm(dim=-1).mean()
            else:
                active = hard_pointer < depth
                candidate_mask = (
                    (
                        (candidate_blocks >= hard_pointer[:, None])
                        & (candidate_blocks < depth)
                    )
                    | (candidate_blocks == depth)
                ) & active[:, None]
                neutral_index = hard_pointer.clamp_max(depth - 1) * dwell
                hard_probabilities, selected_index = self._scheduled_hard_policy(
                    value_field, candidate_mask, neutral_index, candidate_blocks
                )
                rows = torch.arange(batch, device=device)
                selected_block = candidate_blocks[rows, selected_index]
                selected_repeat = candidate_repeats[rows, selected_index]
                selected_terminal = selected_block == depth
                operation_active = active & ~selected_terminal
                active_rows = torch.nonzero(operation_active, as_tuple=False).flatten()
                action = block_input.clone()
                if int(active_rows.numel()) > 0:
                    updated, committed_metrics, committed_contractions = (
                        self._apply_selected_native_operations(
                            block_input.index_select(0, active_rows),
                            block_index=selected_block.index_select(0, active_rows),
                            repeat_count=selected_repeat.index_select(0, active_rows) + 1,
                            evidence_tokens=evidence_tokens.index_select(0, active_rows),
                            evidence_value_tokens=evidence_value_tokens.index_select(0, active_rows),
                            global_condition=global_condition.index_select(0, active_rows),
                            evidence_key_bias=evidence_bias,
                            evidence_scale=self._select_scale_rows(
                                evidence_scale, active_rows, batch=batch
                            ),
                            capacity_ratios=capacity_ratios.index_select(0, active_rows),
                            identity_boundary=identity_boundary,
                            prepared_factors=prepared_factors,
                        )
                    )
                    action = action.index_copy(0, active_rows, updated)
                    contraction_rows.extend(committed_contractions)
                else:
                    zero = torch.zeros((), device=device, dtype=torch.float32)
                    committed_metrics = {
                        "action_update_norm": zero,
                        "attention_update_norm": zero,
                        "ffn_update_norm": zero,
                        "self_update_norm": zero,
                        "evidence_update_norm": zero,
                        "evidence_attention_entropy": zero,
                        "evidence_attention_max": zero,
                        "residual_gate_mean": zero,
                        "ffn_gate_mean": zero,
                        "execution_gate_mean": torch.ones_like(zero),
                        "action_token_norm": action.detach().float().norm(dim=-1).mean(),
                    }
                candidate_velocity_stack, probe_mask, probe_operation_count = (
                    self._probe_native_candidates(
                        block_input,
                        prediction_reference=prefix_velocity_rows[-1],
                        candidate_blocks=candidate_blocks,
                        candidate_repeats=candidate_repeats,
                        candidate_mask=candidate_mask,
                        evidence_tokens=evidence_tokens,
                        evidence_value_tokens=evidence_value_tokens,
                        global_condition=global_condition,
                        evidence_key_bias=evidence_bias,
                        evidence_scale=evidence_scale,
                        capacity_ratios=capacity_ratios,
                        identity_boundary=identity_boundary,
                    )
                )
                selected_capacity = capacity_ratios[
                    rows, selected_block.clamp_max(depth - 1)
                ]
                selected_capacity = torch.where(
                    operation_active,
                    selected_capacity,
                    torch.zeros_like(selected_capacity),
                )
                capacity_rows.append(selected_capacity.sum())
                capacity_weight_rows.append(operation_active.float().sum())
                dwell_value = torch.where(
                    operation_active,
                    selected_repeat + 1,
                    torch.zeros_like(selected_repeat),
                )
                dwell_rows.append(dwell_value.detach().float().mean())
                hard_dwell_rows.append(dwell_value.detach().float().mean())
                route_value = operation_active & (selected_block > hard_pointer)
                route_rows.append(route_value.detach().float().mean())
                hard_route_rows.append(route_value.detach().float().mean())
                execution_cost_rows.append(
                    (
                        selected_capacity.detach().float()
                        * dwell_value.detach().float()
                    ).mean()
                )
                selection_entropy_rows.append(torch.zeros((), device=device))
                selection_max_probability_rows.append(torch.ones((), device=device))
                learned_selection_entropy_rows.append(
                    -(
                        hard_probabilities.float()
                        * hard_probabilities.float().clamp_min(1e-8).log()
                    ).sum(dim=-1).mean()
                )
                terminal_after = (~active) | selected_terminal
                terminal_probability_rows.append(terminal_after.detach().float().mean())
                hard_terminal_rows.append(terminal_after.detach().float().mean())
                block_visit_rows.append(
                    torch.nn.functional.one_hot(
                        selected_block.clamp_max(depth - 1), num_classes=depth
                    ).detach().float().mul(operation_active[:, None]).mean(dim=0)
                )
                hard_pointer = torch.where(
                    active,
                    torch.where(selected_terminal, torch.full_like(selected_block, depth), selected_block + 1),
                    hard_pointer,
                )

            candidate_prediction_rows.append(candidate_velocity_stack.detach())
            candidate_mask_rows.append(probe_mask)
            value_field_rows.append(value_field)
            value_mask_rows.append(probe_mask)
            probe_operation_rows.append(probe_operation_count)
            block_rows.append(committed_metrics)
            prefix_velocity_rows.append(self.velocity_head(self.action_norm(action)))
            feedback = action - block_input

        return {
            "action": action,
            "controller_state": controller_state,
            "feedback": feedback,
            "block_rows": block_rows,
            "capacity_rows": capacity_rows,
            "capacity_weight_rows": capacity_weight_rows,
            "dwell_rows": dwell_rows,
            "execution_cost_rows": execution_cost_rows,
            "controller_rows": controller_rows,
            "contraction_rows": contraction_rows,
            "route_rows": route_rows,
            "hard_route_rows": hard_route_rows,
            "hard_dwell_rows": hard_dwell_rows,
            "terminal_probability_rows": terminal_probability_rows,
            "hard_terminal_rows": hard_terminal_rows,
            "prefix_velocity_rows": prefix_velocity_rows,
            "dwell_candidate_prediction_rows": candidate_prediction_rows,
            "dwell_candidate_mask_rows": candidate_mask_rows,
            "execution_value_field_rows": value_field_rows,
            "execution_value_mask_rows": value_mask_rows,
            "probe_operation_rows": probe_operation_rows,
            "block_visit_rows": block_visit_rows,
            "selection_entropy_rows": selection_entropy_rows,
            "selection_max_probability_rows": selection_max_probability_rows,
            "learned_selection_entropy_rows": learned_selection_entropy_rows,
        }

    def _finalize_dynamic_output(
        self,
        *,
        result: dict[str, Any],
        organized: dict[str, Tensor | dict[str, Tensor]],
        semantic_seed: Tensor,
        action_state_tokens: Tensor,
        action_state_factor: Tensor,
        evidence_tokens: Tensor,
        evidence_scale: float | Tensor,
        collect_diagnostics: bool = True,
    ) -> dict[str, Tensor]:
        action = self.action_norm(result["action"])
        pred_velocity = self.velocity_head(action)
        event_logits = self.event_head(action)
        motion_logits = self.motion_head(action).squeeze(-1)
        if not collect_diagnostics:
            return {
                "pred_velocity": pred_velocity,
                "event_logits": event_logits,
                "motion_logits": motion_logits,
                "evidence_latent": organized["latent"],
            }
        out: dict[str, Tensor] = {
            "pred_velocity": pred_velocity,
            "event_logits": event_logits,
            "motion_logits": motion_logits,
            "evidence_latent": organized["latent"],
            "evidence_semantic_seed_norm": semantic_seed.detach().float().norm(dim=-1).mean(),
            "evidence_mmd_it_action_state_scale": action_state_factor.detach().float().mean(),
            "evidence_mmd_it_evidence_scale": torch.as_tensor(
                evidence_scale, device=action.device, dtype=torch.float32
            ).mean(),
            "evidence_action_token_norm": action.detach().float().norm(dim=-1).mean(),
            "evidence_action_state_token_norm": action_state_tokens.detach().float().norm(dim=-1).mean(),
            "evidence_mmd_it_action_state_token_norm": action_state_tokens.detach().float().norm(dim=-1).mean(),
            "evidence_condition_token_norm": evidence_tokens.detach().float().norm(dim=-1).mean(),
            "evidence_mmd_it_block_count": torch.tensor(
                float(len(self.blocks)), device=action.device
            ),
        }
        block_rows = result["block_rows"]
        zero = torch.zeros((), device=action.device, dtype=torch.float32)
        for name in (
            "action_update_norm",
            "attention_update_norm",
            "ffn_update_norm",
            "self_update_norm",
            "evidence_update_norm",
            "evidence_attention_entropy",
            "evidence_attention_max",
            "residual_gate_mean",
            "ffn_gate_mean",
            "execution_gate_mean",
        ):
            values = [row[name] for row in block_rows]
            out[f"evidence_mmd_it_{name}"] = torch.stack(values).mean() if values else zero
        out["evidence_mmd_it_update_norm"] = out["evidence_mmd_it_action_update_norm"]
        out["evidence_mmd_it_execution_progress"] = (
            self.execution_progress.detach().float().clone()
        )
        out["evidence_mmd_it_dwell_expected"] = (
            torch.stack(result["dwell_rows"]).mean()
            if result["dwell_rows"]
            else torch.ones((), device=action.device, dtype=torch.float32)
        ).detach().float()
        out["evidence_mmd_it_dwell_compute_fraction"] = (
            out["evidence_mmd_it_dwell_expected"] / float(max(self.max_dwell, 1))
        )
        out["evidence_mmd_it_expected_operations_per_decision"] = out[
            "evidence_mmd_it_dwell_expected"
        ]
        # Historical alias; this value is an expectation per decision, not a
        # discrete count of physically committed kernels under soft execution.
        out["evidence_mmd_it_committed_operation_count"] = out[
            "evidence_mmd_it_dwell_expected"
        ]
        out["evidence_mmd_it_hard_dwell_expected"] = (
            torch.stack(result["hard_dwell_rows"]).mean()
            if result.get("hard_dwell_rows")
            else out["evidence_mmd_it_dwell_expected"].detach()
        ).detach().float()
        chart_operation_count = (
            torch.stack(result["probe_operation_rows"]).mean()
            if result["probe_operation_rows"]
            else torch.zeros((), device=action.device, dtype=torch.float32)
        ).detach().float()
        out["evidence_mmd_it_execution_chart_operation_count"] = chart_operation_count
        out["evidence_mmd_it_candidate_probe_operation_count"] = (
            chart_operation_count
            if self.training
            else torch.zeros_like(chart_operation_count)
        )
        out["evidence_mmd_it_candidate_probe_enabled"] = torch.as_tensor(
            float(self.training), device=action.device, dtype=torch.float32
        )
        out["evidence_mmd_it_execution_eval_policy_code"] = torch.as_tensor(
            {"soft": 0.0, "hard": 1.0, "neutral": 2.0}[self._execution_eval_policy()],
            device=action.device,
            dtype=torch.float32,
        )
        out["evidence_mmd_it_execution_selection_entropy"] = (
            torch.stack(result["selection_entropy_rows"]).mean()
            if result.get("selection_entropy_rows")
            else torch.zeros((), device=action.device, dtype=torch.float32)
        ).detach().float()
        out["evidence_mmd_it_execution_selection_max_probability"] = (
            torch.stack(result["selection_max_probability_rows"]).mean()
            if result.get("selection_max_probability_rows")
            else torch.ones((), device=action.device, dtype=torch.float32)
        ).detach().float()
        out["evidence_mmd_it_learned_selection_entropy"] = (
            torch.stack(result["learned_selection_entropy_rows"]).mean()
            if result.get("learned_selection_entropy_rows")
            else torch.zeros((), device=action.device, dtype=torch.float32)
        ).detach().float()
        out["evidence_mmd_it_capacity_ratio"] = (
            torch.stack(result["capacity_rows"]).sum()
            / torch.stack(result["capacity_weight_rows"]).sum().clamp_min(1e-8)
            if result["capacity_rows"] and result["capacity_weight_rows"]
            else torch.ones((), device=action.device, dtype=torch.float32)
        ).detach().float()
        # Honest names: this is a continuous low-rank gate, not an integer
        # hardware rank or guaranteed compute reduction.  Old aliases stay in
        # the tensor contract for checkpoint/log reader compatibility.
        out["evidence_mmd_it_capacity_gate_mass"] = out[
            "evidence_mmd_it_capacity_ratio"
        ]
        rank = float(getattr(self.config, "latent_cvae_mmdit_operator_rank", 0))
        out["evidence_mmd_it_effective_basis_mass"] = (
            out["evidence_mmd_it_capacity_gate_mass"] * rank
        )
        out["evidence_mmd_it_selected_effective_depth"] = (
            out["evidence_mmd_it_effective_basis_mass"]
        )
        out["evidence_mmd_it_selected_active_group_fraction"] = out[
            "evidence_mmd_it_capacity_ratio"
        ]
        out["evidence_mmd_it_execution_cost"] = (
            torch.stack(result["execution_cost_rows"]).mean()
            if result["execution_cost_rows"]
            else torch.zeros((), device=action.device, dtype=torch.float32)
        ).detach().float()
        out["evidence_mmd_it_terminal_prior_weight"] = torch.as_tensor(
            float(getattr(self.config, "latent_cvae_mmdit_terminal_prior_weight", 0.25)),
            device=action.device,
            dtype=torch.float32,
        )
        out["evidence_mmd_it_terminal_probability"] = (
            torch.stack(result["terminal_probability_rows"]).mean()
            if result.get("terminal_probability_rows")
            else torch.zeros((), device=action.device, dtype=torch.float32)
        ).detach().float()
        out["evidence_mmd_it_hard_terminal_fraction"] = (
            torch.stack(result["hard_terminal_rows"]).mean()
            if result.get("hard_terminal_rows")
            else torch.zeros((), device=action.device, dtype=torch.float32)
        ).detach().float()
        out["evidence_mmd_it_operation_probability"] = (
            1.0 - out["evidence_mmd_it_terminal_probability"]
        ).clamp(0.0, 1.0)
        out["evidence_mmd_it_execution_baseline_pred_velocity"] = torch.stack(
            result["prefix_velocity_rows"][:-1], dim=1
        )
        out["evidence_mmd_it_prefix_pred_velocity"] = torch.stack(
            result["prefix_velocity_rows"], dim=1
        )
        out["evidence_mmd_it_dwell_candidate_pred_velocity"] = torch.stack(
            result["dwell_candidate_prediction_rows"], dim=1
        )
        out["evidence_mmd_it_dwell_candidate_mask"] = torch.stack(
            result["dwell_candidate_mask_rows"], dim=1
        )
        out["evidence_mmd_it_execution_candidate_value_field"] = torch.stack(
            result["execution_value_field_rows"], dim=1
        )
        out["evidence_mmd_it_execution_candidate_value_mask"] = torch.stack(
            result["execution_value_mask_rows"], dim=1
        )
        out["evidence_mmd_it_dynamic_route_next_fraction"] = (
            torch.stack(result["route_rows"]).mean().detach().float()
        )
        out["evidence_mmd_it_hard_route_next_fraction"] = (
            torch.stack(result["hard_route_rows"]).mean()
            if result.get("hard_route_rows")
            else out["evidence_mmd_it_dynamic_route_next_fraction"].detach()
        ).detach().float()
        if result["block_visit_rows"]:
            block_visits = torch.stack(result["block_visit_rows"]).mean(dim=0)
            for block_index, visit in enumerate(block_visits):
                out[f"evidence_mmd_it_committed_block_{block_index}_fraction"] = (
                    visit.detach().float()
                )
        out["evidence_mmd_it_controller_operation_candidate_count"] = torch.as_tensor(
            float(len(self.blocks) * self.max_dwell),
            device=action.device,
            dtype=torch.float32,
        )
        out["evidence_mmd_it_controller_total_candidate_count"] = torch.as_tensor(
            float(len(self.blocks) * self.max_dwell + int(self.identity_candidate_enabled)),
            device=action.device,
            dtype=torch.float32,
        )
        for rows_name, prefix in (("controller_rows", "evidence_mmd_it_"),):
            rows = result[rows_name]
            if rows:
                for name in rows[0]:
                    out[f"{prefix}{name}"] = torch.stack(
                        [row[name] for row in rows]
                    ).mean().detach().float()
        if result["contraction_rows"]:
            for name in (
                "depth_ratio",
                "effective_depth",
                "contraction_ratio",
                "removed_fraction",
                "nonexpansive_violation",
                "boundary_identity_error",
            ):
                out[f"evidence_mmd_it_neutral_{name}"] = torch.stack(
                    [row[name] for row in result["contraction_rows"]]
                ).mean().detach().float()
        # The training action is a scheduled soft execution mixture.  Report
        # its expected ordered depth as the primary capacity metric; neutral or
        # hard-path contraction diagnostics remain separately named above.
        out["evidence_mmd_it_depth_ratio"] = out["evidence_mmd_it_capacity_ratio"]
        out["evidence_mmd_it_effective_depth"] = out[
            "evidence_mmd_it_selected_effective_depth"
        ]
        out["evidence_mmd_it_removed_channel_fraction"] = (
            1.0 - out["evidence_mmd_it_capacity_ratio"]
        ).clamp(0.0, 1.0)
        out["evidence_mmd_it_contraction_ratio"] = out.get(
            "evidence_mmd_it_neutral_contraction_ratio",
            torch.ones((), device=action.device, dtype=torch.float32),
        )
        out["evidence_mmd_it_removed_fraction"] = out.get(
            "evidence_mmd_it_neutral_removed_fraction",
            torch.zeros((), device=action.device, dtype=torch.float32),
        )
        out["evidence_mmd_it_nonexpansive_violation"] = out.get(
            "evidence_mmd_it_neutral_nonexpansive_violation",
            torch.zeros((), device=action.device, dtype=torch.float32),
        )
        out["evidence_mmd_it_boundary_identity_error"] = out.get(
            "evidence_mmd_it_neutral_boundary_identity_error",
            torch.zeros((), device=action.device, dtype=torch.float32),
        )
        metrics = organized["metrics"]
        if not isinstance(metrics, dict):
            raise RuntimeError("organizer metrics must be a dict")
        out.update({key: value for key, value in metrics.items() if isinstance(value, Tensor)})
        for index, row in enumerate(block_rows):
            for name in (
                "action_update_norm",
                "attention_update_norm",
                "ffn_update_norm",
                "self_update_norm",
                "evidence_update_norm",
                "residual_gate_mean",
                "ffn_gate_mean",
            ):
                out[f"evidence_mmd_it_block_{index}_{name}"] = row[name]
            out[f"evidence_mmd_it_block_{index}_update_norm"] = row["action_update_norm"]
        return out

    def forward(
        self,
        *,
        noisy_physical: Tensor,
        time: Tensor,
        trajectory_tokens: Tensor,
        trajectory_workspace_tokens: Tensor | None = None,
        policy_action_tokens: Tensor | None = None,
        policy_role_delta_bank: PolicyRoleDeltaBank | None = None,
        execution_terminal_probability: Tensor | None = None,
        execution_terminal_uncertainty: Tensor | None = None,
        rollout_tokens: Tensor,
        transition_memory: Tensor | list[Tensor] | tuple[Tensor, ...] | None,
        event_evidence: Tensor,
        state_memory: Tensor | list[Tensor] | tuple[Tensor, ...],
        layer_contracts: list[dict[str, Tensor]],
        intent_memory: dict[str, Tensor] | None = None,
        visual_selector_tokens: Tensor | None = None,
        visual_value_tokens: Tensor | None = None,
        visual_key_bias: Tensor | None = None,
        collect_diagnostics: bool = True,
        evidence_scale: float | Tensor = 1.0,
        noisy_scale: float | Tensor = 1.0,
    ) -> dict[str, Tensor]:
        if not self.training:
            # Parent-module ``load_state_dict`` restores child buffers through
            # ``_load_from_state_dict`` and therefore does not invoke the
            # decoder override above.  Refresh the non-persistent branch
            # cache at the eval boundary so standalone checkpoint replay uses
            # the saved execution phase instead of silently reverting to the
            # warm-up path.
            self._execution_progress_value = float(self.execution_progress.detach().cpu())
        if noisy_physical.ndim != 3 or int(noisy_physical.shape[1]) != self.horizon:
            raise ValueError(f"noisy_physical must be [B,{self.horizon},P]")
        if time.ndim != 1 or int(time.shape[0]) != int(noisy_physical.shape[0]):
            raise ValueError("time must be [B] aligned with noisy_physical")
        if self.policy_delta_bridge_enabled:
            if policy_action_tokens is not None:
                raise ValueError(
                    "typed policy-delta bridge cannot also consume the legacy workspace"
                )
            if policy_role_delta_bank is None:
                raise ValueError(
                    "typed policy-to-MMDiT bridge requires a policy role-delta bank"
                )
            policy_role_delta_bank.validate(
                hidden_size=self.hidden_size, horizon=self.horizon
            )
        elif self.top_policy_workspace_lift is None:
            if policy_action_tokens is not None:
                raise ValueError(
                    "policy action workspace was supplied while the role hierarchy is disabled"
                )
            if policy_role_delta_bank is not None:
                raise ValueError(
                    "policy role-delta bank was supplied while its bridge is disabled"
                )
        else:
            if policy_role_delta_bank is not None:
                raise ValueError(
                    "legacy policy workspace and typed policy deltas are mutually exclusive"
                )
            if policy_action_tokens is None:
                raise ValueError("the role hierarchy requires policy action workspace tokens")
            if (
                policy_action_tokens.ndim != 3
                or int(policy_action_tokens.shape[0]) != int(noisy_physical.shape[0])
                or int(policy_action_tokens.shape[-1]) != self.hidden_size
            ):
                raise ValueError(
                    f"policy_action_tokens must be [B,N,{self.hidden_size}]"
                )
        # The workspace projection is the typed trajectory source when it is
        # available. Keep the fallback for callers that have not exposed the
        # workspace view, but never feed both copies into the evidence bank.
        trajectory_source = (
            trajectory_workspace_tokens
            if trajectory_workspace_tokens is not None
            else trajectory_tokens
        )
        view = self.evidence_adapter(
            trajectory_tokens=trajectory_source,
            rollout_tokens=rollout_tokens,
            transition_memory=transition_memory,
            event_evidence=event_evidence,
            state_memory=state_memory,
            layer_contracts=layer_contracts,
            intent_memory=intent_memory,
            visual_selector_tokens=visual_selector_tokens,
            visual_value_tokens=visual_value_tokens,
            visual_key_bias=visual_key_bias,
        )
        organized = self.organizer(view, time)
        global_condition = organized["global_condition"]
        z_token = organized["latent_token"]
        condition_hidden = organized["condition_hidden"]
        time_hidden = organized["time_hidden"]
        if not all(
            isinstance(value, Tensor)
            for value in (global_condition, z_token, condition_hidden, time_hidden)
        ):
            raise RuntimeError("condition organizer returned an invalid semantic seed")
        semantic_seed = self.intent_seed_norm(z_token)
        # ``noisy_physical`` is the flow state x_t, not a second semantic
        # source. Put its lifted representation directly into the action
        # stream so the MMDiT self-attention learns its interaction with the
        # evidence condition.
        action_state_tokens = self.noisy_lift(noisy_physical)
        evidence_tokens = view.tokens
        evidence_value_tokens = view.value_tokens
        evidence_bias = view.key_bias.to(device=evidence_tokens.device)
        action = self.horizon_query.to(
            device=noisy_physical.device, dtype=noisy_physical.dtype
        ).expand(noisy_physical.shape[0], -1, -1)
        action = action + self.horizon_position.to(device=action.device, dtype=action.dtype)
        action = action + semantic_seed[:, None].to(dtype=action.dtype)
        action_state_factor = TimeDomainMMDiTBlock._source_scale(
            noisy_scale, action_state_tokens
        )
        action = action + action_state_tokens * action_state_factor
        policy_delta_metrics: dict[str, Tensor] = {}
        if self.policy_delta_bridge_enabled:
            assert policy_role_delta_bank is not None
            policy_workspace_update, protected_detail_update, policy_delta_metrics = (
                self._read_policy_delta_bank(
                    action,
                    policy_role_delta_bank,
                    collect_diagnostics=collect_diagnostics,
                )
            )
            policy_workspace_scale = action.new_tensor(
                float(
                    getattr(
                        self.config,
                        "role_attnres_policy_to_mmdit_scale",
                        0.25,
                    )
                )
            )
            action = action + policy_workspace_update + protected_detail_update
        elif self.top_policy_workspace_lift is None:
            policy_workspace_update = action.new_zeros(action.shape)
            protected_detail_update = action.new_zeros(action.shape)
            policy_workspace_scale = action.new_zeros(())
        else:
            assert policy_action_tokens is not None
            protected_detail_update = action.new_zeros(action.shape)
            policy_tokens = policy_action_tokens.to(
                device=action.device, dtype=action.dtype
            )
            lifted_policy = self._lift_policy_workspace(policy_tokens)
            if self.top_policy_workspace_fixed_fusion:
                # Both branches have a complete path.  Fixed variance-preserving
                # fusion removes the arbitrary 0.10 bottleneck without adding a
                # learned amplitude gate that could collapse either source.
                policy_workspace_scale = action.new_tensor(math.sqrt(0.5))
                workspace_direction = F.layer_norm(
                    lifted_policy.float(),
                    (int(lifted_policy.shape[-1]),),
                ).to(dtype=action.dtype)
                action_rms = action.detach().float().square().mean(dim=-1, keepdim=True).sqrt()
                workspace_direction = workspace_direction * action_rms.to(dtype=action.dtype)
                policy_workspace_update = policy_workspace_scale * workspace_direction
                action = policy_workspace_scale * action + policy_workspace_update
            else:
                policy_workspace_scale = action.new_tensor(
                    float(getattr(self.config, "flow_jepa_policy_workspace_scale", 0.10))
                )
                policy_workspace_update = policy_workspace_scale * lifted_policy
                action = action + policy_workspace_update
        if (
            self.dynamic_block_route_enabled
            and self.dwell_mode != "learned_shadow"
        ):
            external_terminal_bias_metric = torch.zeros(
                (), device=action.device, dtype=torch.float32
            )
            if execution_terminal_probability is not None:
                if execution_terminal_uncertainty is None:
                    raise ValueError(
                        "external terminal probability requires uncertainty"
                    )
                external_terminal_bias_metric = (
                    0.10
                    * torch.tanh(
                        torch.logit(
                            execution_terminal_probability.float().clamp(
                                1e-4, 1.0 - 1e-4
                            )
                        )
                        / 4.0
                    )
                    / (
                        1.0
                        + execution_terminal_uncertainty.float().clamp_min(0.0)
                    )
                ).mean()
            dynamic_result = self._run_dynamic_execution(
                action=action,
                global_condition=global_condition,
                time_hidden=time_hidden,
                evidence_tokens=evidence_tokens,
                evidence_value_tokens=evidence_value_tokens,
                evidence_bias=evidence_bias,
                evidence_scale=evidence_scale,
                execution_terminal_probability=(
                    execution_terminal_probability
                ),
                execution_terminal_uncertainty=(
                    execution_terminal_uncertainty
                ),
            )
            dynamic_output = self._finalize_dynamic_output(
                result=dynamic_result,
                organized=organized,
                semantic_seed=semantic_seed,
                action_state_tokens=action_state_tokens,
                action_state_factor=action_state_factor,
                evidence_tokens=evidence_tokens,
                evidence_scale=evidence_scale,
                collect_diagnostics=collect_diagnostics,
            )
            if not collect_diagnostics:
                return dynamic_output
            dynamic_output["evidence_top_policy_workspace_scale"] = (
                policy_workspace_scale.detach().float()
            )
            dynamic_output["evidence_execution_terminal_external_bias"] = (
                external_terminal_bias_metric.detach().float()
            )
            dynamic_output["evidence_top_policy_workspace_update_norm"] = (
                policy_workspace_update.detach().float().norm(dim=-1).mean()
            )
            dynamic_output[
                "evidence_top_policy_protected_detail_update_norm"
            ] = protected_detail_update.detach().float().norm(dim=-1).mean()
            dynamic_output["evidence_top_policy_workspace_fixed_fusion"] = action.new_tensor(
                float(self.top_policy_workspace_fixed_fusion)
            )
            dynamic_output[
                "evidence_top_policy_workspace_horizon_pool"
            ] = action.new_tensor(float(self.top_policy_workspace_horizon_pool))
            dynamic_output["evidence_policy_delta_bridge_enabled"] = action.new_tensor(
                float(self.policy_delta_bridge_enabled)
            )
            dynamic_output.update(policy_delta_metrics)
            return dynamic_output
        block_rows: list[dict[str, Tensor]] = []
        controller_state: Tensor | None = None
        feedback: Tensor | None = None
        capacity_rows: list[Tensor] = []
        dwell_rows: list[Tensor] = []
        execution_cost_rows: list[Tensor] = []
        controller_rows: list[dict[str, Tensor]] = []
        contraction_rows: list[dict[str, Tensor]] = []
        prefix_velocity_rows: list[Tensor] = [
            self.velocity_head(self.action_norm(action))
        ]
        dwell_candidate_prediction_rows: list[Tensor] = []
        dwell_candidate_mask_rows: list[Tensor] = []
        execution_value_field_rows: list[Tensor] = []
        execution_value_mask_rows: list[Tensor] = []
        probe_operation_rows: list[Tensor] = []
        selected_effective_depth_rows: list[Tensor] = []
        selected_active_group_rows: list[Tensor] = []
        selection_entropy_rows: list[Tensor] = []
        selection_max_probability_rows: list[Tensor] = []
        identity_boundary = self._execution_progress_value <= 0.0
        prepared_operation_factors = (
            None
            if not self.operator_capacity_enabled or identity_boundary
            else tuple(bank.prepare_factors() for bank in self.operator_contractions)
        )
        for block_index in range(len(self.blocks)):
            block_input = action
            control = None
            if self.execution_controller is not None:
                control = self.execution_controller(
                    state=controller_state,
                    global_condition=global_condition,
                    time_context=time_hidden,
                    action_tokens=action,
                    evidence_tokens=evidence_tokens,
                    evidence_value_tokens=evidence_value_tokens,
                    evidence_key_bias=evidence_bias,
                    feedback=feedback,
                    block_index=block_index,
                )
                controller_state = control.state
                controller_rows.append(control.metrics)
            capacity = self._execution_capacity(
                None if control is None else control.capacity_ratio
            )
            if self.operator_capacity_enabled:
                capacity_rows.append(capacity.detach().float().mean())
            shadow_execution = bool(
                control is not None
                and self.dwell_mode in {"learned", "learned_shadow"}
            )
            learned_execution = bool(
                shadow_execution
                and self.dwell_mode == "learned"
                and self._execution_progress_value > 0.0
            )
            random_execution = bool(
                control is not None and self.dwell_mode == "random"
            )
            probe_execution = shadow_execution or random_execution
            batch = int(action.shape[0])
            candidate_blocks = torch.full(
                (batch, self.max_dwell),
                block_index,
                device=action.device,
                dtype=torch.long,
            )
            candidate_repeats = torch.arange(
                self.max_dwell, device=action.device, dtype=torch.long
            )[None].expand(batch, -1)
            candidate_mask = torch.zeros(
                batch, self.max_dwell, device=action.device, dtype=torch.bool
            )
            candidate_mask[:, : (self.max_dwell if probe_execution else 1)] = True
            if shadow_execution:
                value_field = self.execution_controller.predict_execution_value(
                    state=controller_state,
                    global_condition=global_condition,
                    time_context=time_hidden,
                    evidence_tokens=evidence_tokens,
                    evidence_value_tokens=evidence_value_tokens,
                    evidence_key_bias=evidence_bias,
                    action_tokens=block_input,
                    block_index=block_index,
                )
                selected_index = (
                    self._select_execution_candidate(
                        value_field, candidate_mask, arm_dim=self.arm_dim
                    )
                    if learned_execution
                    else torch.zeros(
                        batch, device=action.device, dtype=torch.long
                    )
                )
            elif random_execution:
                value_field = torch.zeros(
                    batch,
                    self.max_dwell,
                    self.horizon,
                    2,
                    device=action.device,
                    dtype=action.dtype,
                )
                selected_index = torch.randint(
                    self.max_dwell,
                    (batch,),
                    device=action.device,
                    dtype=torch.long,
                )
            else:
                value_field = torch.zeros(
                    batch,
                    self.max_dwell,
                    self.horizon,
                    2,
                    device=action.device,
                    dtype=action.dtype,
                )
                selected_index = torch.ones(
                    batch, device=action.device, dtype=torch.long
                ) - 1
            block_ids = torch.full(
                (batch,), block_index, device=action.device, dtype=torch.long
            )
            with torch.no_grad() if learned_execution else nullcontext():
                hard_action, committed_metrics, committed_contractions = (
                    self._apply_selected_native_operations(
                        block_input,
                        block_index=block_ids,
                        repeat_count=selected_index + 1,
                        evidence_tokens=evidence_tokens,
                        evidence_value_tokens=evidence_value_tokens,
                        global_condition=global_condition,
                        evidence_key_bias=evidence_bias,
                        evidence_scale=evidence_scale,
                        capacity_ratios=(
                            capacity if self.operator_capacity_enabled else None
                        ),
                        identity_boundary=identity_boundary,
                        prepared_factors=prepared_operation_factors,
                    )
                )
            contraction_rows.extend(committed_contractions)
            if learned_execution:
                candidate_action_stack, candidate_velocity_stack, probe_mask, probe_operation_count = (
                    self._run_differentiable_native_candidates(
                        block_input,
                        baseline_velocity=prefix_velocity_rows[-1],
                        candidate_blocks=candidate_blocks,
                        candidate_repeats=candidate_repeats,
                        candidate_mask=candidate_mask,
                        evidence_tokens=evidence_tokens,
                        evidence_value_tokens=evidence_value_tokens,
                        global_condition=global_condition,
                        evidence_key_bias=evidence_bias,
                        evidence_scale=evidence_scale,
                        capacity_ratios=(
                            capacity if self.operator_capacity_enabled else None
                        ),
                        identity_boundary=identity_boundary,
                        prepared_factors=prepared_operation_factors,
                    )
                )
                selection_probabilities = self._soft_execution_probabilities(
                    value_field, candidate_mask, candidate_blocks
                )
                action = self._mix_candidate_actions(
                    candidate_action_stack, selection_probabilities
                )
                selection_entropy_rows.append(
                    -(
                        selection_probabilities.float()
                        * selection_probabilities.float().clamp_min(1e-8).log()
                    ).sum(dim=-1).mean()
                )
                selection_max_probability_rows.append(
                    selection_probabilities.detach().float().amax(dim=-1).mean()
                )
            else:
                action = hard_action
                candidate_velocity_stack, probe_mask, probe_operation_count = (
                    self._probe_native_candidates(
                        block_input,
                        prediction_reference=prefix_velocity_rows[-1],
                        candidate_blocks=candidate_blocks,
                        candidate_repeats=candidate_repeats,
                        candidate_mask=candidate_mask,
                        evidence_tokens=evidence_tokens,
                        evidence_value_tokens=evidence_value_tokens,
                        global_condition=global_condition,
                        evidence_key_bias=evidence_bias,
                        evidence_scale=evidence_scale,
                        capacity_ratios=(
                            capacity if self.operator_capacity_enabled else None
                        ),
                        identity_boundary=identity_boundary,
                    )
                )
                selection_entropy_rows.append(torch.zeros((), device=action.device))
                selection_max_probability_rows.append(torch.ones((), device=action.device))
            if self.operator_capacity_enabled:
                execution_cost_rows.append(
                    (
                        capacity.detach().float()
                        * (selected_index.detach().float() + 1.0)
                    ).mean()
                )
                rank = int(self.config.latent_cvae_mmdit_operator_rank)
                selected_effective_depth_rows.append(
                    (capacity.detach().float() * float(rank))
                    .detach()
                    .float()
                    .mean()
                )
                selected_active_group_rows.append(
                    capacity.detach().float().mean()
                )
            dwell_rows.append((selected_index + 1).detach().float().mean())
            dwell_candidate_prediction_rows.append(candidate_velocity_stack.detach())
            dwell_candidate_mask_rows.append(probe_mask)
            execution_value_field_rows.append(value_field)
            execution_value_mask_rows.append(probe_mask)
            probe_operation_rows.append(probe_operation_count)
            feedback = action - block_input
            block_rows.append(committed_metrics)
            prefix_velocity_rows.append(self.velocity_head(self.action_norm(action)))
        action = self.action_norm(action)
        pred_velocity = self.velocity_head(action)
        event_logits = self.event_head(action)
        motion_logits = self.motion_head(action).squeeze(-1)
        if not collect_diagnostics:
            return {
                "pred_velocity": pred_velocity,
                "event_logits": event_logits,
                "motion_logits": motion_logits,
                "evidence_latent": organized["latent"],
            }
        out: dict[str, Tensor] = {
            "pred_velocity": pred_velocity,
            "event_logits": event_logits,
            "motion_logits": motion_logits,
            "evidence_latent": organized["latent"],
            "evidence_semantic_seed_norm": semantic_seed.detach().float().norm(dim=-1).mean(),
            "evidence_mmd_it_action_state_scale": action_state_factor.detach().float().mean(),
            "evidence_mmd_it_evidence_scale": torch.as_tensor(
                evidence_scale, device=action.device, dtype=torch.float32
            ).mean(),
            "evidence_action_token_norm": action.detach().float().norm(dim=-1).mean(),
            "evidence_action_state_token_norm": action_state_tokens.detach().float().norm(dim=-1).mean(),
            "evidence_mmd_it_action_state_token_norm": action_state_tokens.detach().float().norm(dim=-1).mean(),
            "evidence_top_policy_workspace_scale": policy_workspace_scale.detach().float(),
            "evidence_top_policy_workspace_update_norm": policy_workspace_update.detach().float().norm(
                dim=-1
            ).mean(),
            "evidence_top_policy_protected_detail_update_norm": (
                protected_detail_update.detach().float().norm(dim=-1).mean()
            ),
            "evidence_top_policy_workspace_fixed_fusion": torch.as_tensor(
                float(self.top_policy_workspace_fixed_fusion),
                device=action.device,
                dtype=torch.float32,
            ),
            "evidence_top_policy_workspace_horizon_pool": torch.as_tensor(
                float(self.top_policy_workspace_horizon_pool),
                device=action.device,
                dtype=torch.float32,
            ),
            "evidence_condition_token_norm": evidence_tokens.detach().float().norm(dim=-1).mean(),
            "evidence_mmd_it_block_count": torch.tensor(
                float(len(self.blocks)), device=action.device
            ),
            "evidence_policy_delta_bridge_enabled": torch.as_tensor(
                float(self.policy_delta_bridge_enabled),
                device=action.device,
                dtype=torch.float32,
            ),
        }
        out.update(policy_delta_metrics)

        scalar_metric_names = (
            "action_update_norm",
            "attention_update_norm",
            "ffn_update_norm",
            "self_update_norm",
            "evidence_update_norm",
            "evidence_attention_entropy",
            "evidence_attention_max",
            "residual_gate_mean",
            "ffn_gate_mean",
            "execution_gate_mean",
        )
        zero = torch.zeros((), device=action.device, dtype=torch.float32)
        for name in scalar_metric_names:
            values = [row[name] for row in block_rows]
            out[f"evidence_mmd_it_{name}"] = torch.stack(values).mean() if values else zero
        out["evidence_mmd_it_update_norm"] = out["evidence_mmd_it_action_update_norm"]
        out["evidence_mmd_it_execution_progress"] = (
            self.execution_progress.detach().float().clone()
        )
        out["evidence_mmd_it_dwell_expected"] = (
            torch.stack(dwell_rows).mean()
            if dwell_rows
            else torch.ones((), device=action.device, dtype=torch.float32)
        )
        out["evidence_mmd_it_dwell_compute_fraction"] = (
            out["evidence_mmd_it_dwell_expected"] / float(max(self.max_dwell, 1))
        )
        out["evidence_mmd_it_committed_operation_count"] = out[
            "evidence_mmd_it_dwell_expected"
        ]
        out["evidence_mmd_it_candidate_probe_operation_count"] = (
            torch.stack(probe_operation_rows).mean()
            if probe_operation_rows
            else torch.zeros((), device=action.device, dtype=torch.float32)
        )
        out["evidence_mmd_it_candidate_probe_enabled"] = torch.as_tensor(
            float(self.training), device=action.device, dtype=torch.float32
        )
        out["evidence_mmd_it_execution_selection_entropy"] = (
            torch.stack(selection_entropy_rows).mean()
            if selection_entropy_rows
            else torch.zeros((), device=action.device, dtype=torch.float32)
        )
        out["evidence_mmd_it_execution_selection_max_probability"] = (
            torch.stack(selection_max_probability_rows).mean()
            if selection_max_probability_rows
            else torch.ones((), device=action.device, dtype=torch.float32)
        )
        out["evidence_mmd_it_capacity_ratio"] = (
            torch.stack(capacity_rows).mean()
            if capacity_rows
            else torch.ones((), device=action.device, dtype=torch.float32)
        )
        out["evidence_mmd_it_selected_effective_depth"] = (
            torch.stack(selected_effective_depth_rows).mean()
            if selected_effective_depth_rows
            else torch.as_tensor(
                float(getattr(self.config, "latent_cvae_mmdit_operator_rank", 0)),
                device=action.device,
                dtype=torch.float32,
            )
        )
        out["evidence_mmd_it_selected_active_group_fraction"] = (
            torch.stack(selected_active_group_rows).mean()
            if selected_active_group_rows
            else torch.ones((), device=action.device, dtype=torch.float32)
        )
        out["evidence_mmd_it_execution_cost"] = self.execution_progress.to(
            device=action.device, dtype=torch.float32
        ).clone() * (
            torch.stack(execution_cost_rows).mean()
            if execution_cost_rows
            else torch.zeros((), device=action.device, dtype=torch.float32)
        )
        out["evidence_mmd_it_dynamic_route_next_fraction"] = torch.zeros(
            (), device=action.device, dtype=torch.float32
        )
        out["evidence_mmd_it_execution_baseline_pred_velocity"] = torch.stack(
            prefix_velocity_rows[:-1], dim=1
        )
        out["evidence_mmd_it_prefix_pred_velocity"] = torch.stack(prefix_velocity_rows, dim=1)
        out["evidence_mmd_it_dwell_candidate_pred_velocity"] = torch.stack(
            dwell_candidate_prediction_rows, dim=1
        )
        out["evidence_mmd_it_dwell_candidate_mask"] = torch.stack(
            dwell_candidate_mask_rows, dim=1
        )
        out["evidence_mmd_it_execution_candidate_value_field"] = torch.stack(
            execution_value_field_rows, dim=1
        )
        out["evidence_mmd_it_execution_candidate_value_mask"] = torch.stack(
            execution_value_mask_rows, dim=1
        )
        if controller_rows:
            for name in controller_rows[0]:
                out[f"evidence_mmd_it_{name}"] = torch.stack(
                    [row[name] for row in controller_rows]
                ).mean()
        if contraction_rows:
            for name in (
                "depth_ratio",
                "effective_depth",
                "contraction_ratio",
                "removed_fraction",
                "nonexpansive_violation",
                "boundary_identity_error",
            ):
                out[f"evidence_mmd_it_{name}"] = torch.stack(
                    [row[name] for row in contraction_rows]
                ).mean()

        metrics = organized["metrics"]
        if not isinstance(metrics, dict):
            raise RuntimeError("organizer metrics must be a dict")
        out.update({key: value for key, value in metrics.items() if isinstance(value, Tensor)})
        for index, row in enumerate(block_rows):
            for name in (
                "action_update_norm",
                "attention_update_norm",
                "ffn_update_norm",
                "self_update_norm",
                "evidence_update_norm",
                "residual_gate_mean",
                "ffn_gate_mean",
            ):
                out[f"evidence_mmd_it_block_{index}_{name}"] = row[name]
            out[f"evidence_mmd_it_block_{index}_update_norm"] = row["action_update_norm"]
        return out
