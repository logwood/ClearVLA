"""Typed condition organization and deterministic intent contracts."""

from __future__ import annotations

import math
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .contracts import LAYER_CONTRACT_KEYS
from .contracts import scaled_contract_view as _scaled_contract_view


class PolicyIntentConfig(Protocol):
    hidden_size: int
    depth: int
    hierarchical_mmdit_consequence_scale_max: float
    hierarchical_mmdit_consequence_scale_init: float
    hierarchical_mmdit_source_grad_scale: float
    hierarchical_mmdit_layer_grad_scale: float


class PolicyConditionOrganizer(nn.Module):
    """Turn trunk outputs into typed summaries and owned evidence tokens.

    This module has no action/noise/time input.  Ordered layer information is
    scanned rather than flattened, and global summaries are returned only to
    the intent compiler; they are never inserted into the evidence value bank.
    """

    _LAYER_KEYS = LAYER_CONTRACT_KEYS
    # trajectory_pooled is derived from the noisy-action canvas and therefore
    # belongs to neither stable intent nor world evidence.  Keeping it in a
    # "world" layer summary recreates x_t under a different source name.
    _WORLD_KEYS = frozenset(("rollout_tokens",))
    _DISALLOWED_LAYER_KEYS = frozenset(("trajectory_pooled",))
    _INTENT_SOURCE_NAMES = (
        "task",
        "state",
        "state_history",
        "executed",
        "proposal",
        "visual",
    )

    def __init__(self, config: PolicyIntentConfig) -> None:
        super().__init__()
        self.config = config
        h = int(config.hidden_size)
        self.hidden_size = h
        self.depth = int(config.depth)
        self.layer_key_proj = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h)) for _ in self._LAYER_KEYS]
        )
        self.layer_key_embed = nn.Parameter(torch.randn(1, len(self._LAYER_KEYS), h) * 0.02)
        self.world_key_gate = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 1))
        self.consequence_key_gate = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 1))
        nn.init.zeros_(self.world_key_gate[-1].weight)
        nn.init.zeros_(self.world_key_gate[-1].bias)
        nn.init.zeros_(self.consequence_key_gate[-1].weight)
        nn.init.zeros_(self.consequence_key_gate[-1].bias)
        consequence_max = float(config.hierarchical_mmdit_consequence_scale_max)
        consequence_init = float(config.hierarchical_mmdit_consequence_scale_init)
        ratio = min(max(consequence_init / consequence_max, 1e-4), 1.0 - 1e-4)
        self.consequence_scale_logit = nn.Parameter(torch.tensor(math.log(ratio / (1.0 - ratio))))
        self.layer_proj = nn.ModuleList(
            [
                nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h))
                for _ in range(self.depth)
            ]
        )
        self.layer_embed = nn.Parameter(torch.randn(1, self.depth, h) * 0.02)
        self.layer_scan = nn.GRUCell(h, h)
        self.layer_scan_init = nn.Parameter(torch.zeros(1, h))
        self.layer_stack_norm = nn.LayerNorm(h, elementwise_affine=False)

        def token_projector(input_dim: int = h) -> nn.Module:
            return nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, h))

        self.trajectory_token_proj = token_projector()
        self.rollout_token_proj = token_projector()
        self.transition_token_proj = token_projector()
        self.state_token_proj = token_projector()
        self.event_token_proj = token_projector(3)
        self.intent_token_proj = nn.ModuleDict(
            {name: token_projector() for name in self._INTENT_SOURCE_NAMES}
        )
        self.intent_source_embed = nn.Parameter(
            torch.randn(1, len(self._INTENT_SOURCE_NAMES), h) * 0.02
        )
        self.geom_summary_proj = nn.Sequential(
            nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h)
        )
        self.global_summary_proj = nn.Sequential(
            nn.LayerNorm(len(self._INTENT_SOURCE_NAMES) * h),
            nn.Linear(len(self._INTENT_SOURCE_NAMES) * h, h),
            nn.SiLU(),
            nn.Linear(h, h),
        )
        self.transition_summary_proj = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
        self.state_summary_proj = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
        self.event_summary_proj = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))

    @staticmethod
    def _groups(memory: Tensor | list[Tensor] | tuple[Tensor, ...] | None) -> list[Tensor]:
        if memory is None:
            return []
        return [memory] if isinstance(memory, Tensor) else list(memory)

    def _project_memory(
        self,
        memory: Tensor | list[Tensor] | tuple[Tensor, ...] | None,
        *,
        projector: nn.Module,
        reference: Tensor,
        input_dim: int,
    ) -> Tensor:
        parts: list[Tensor] = []
        grad_scale = float(self.config.hierarchical_mmdit_source_grad_scale)
        for value in self._groups(memory):
            if (
                not isinstance(value, Tensor)
                or value.ndim != 3
                or int(value.shape[-1]) != int(input_dim)
            ):
                raise ValueError(
                    f"owned evidence memory must be [B,N,{input_dim}], got "
                    f"{type(value).__name__}{'' if not isinstance(value, Tensor) else tuple(value.shape)}"
                )
            source = _scaled_contract_view(value, grad_scale)
            parts.append(projector(source.to(device=reference.device, dtype=reference.dtype)))
        if not parts:
            return reference.new_zeros(int(reference.shape[0]), 0, self.hidden_size)
        return torch.cat(parts, dim=1)

    def _project_intent_memory(
        self,
        memory: dict[str, Tensor],
        *,
        reference: Tensor,
    ) -> tuple[Tensor, dict[str, int]]:
        unknown = set(memory).difference(self._INTENT_SOURCE_NAMES)
        missing = set(self._INTENT_SOURCE_NAMES).difference(memory)
        if unknown or missing:
            raise ValueError(
                "intent memory ownership mismatch: "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        grad_scale = float(self.config.hierarchical_mmdit_source_grad_scale)
        summary_parts: list[Tensor] = []
        counts: dict[str, int] = {}
        for index, name in enumerate(self._INTENT_SOURCE_NAMES):
            value = memory[name]
            if value.ndim != 3 or int(value.shape[-1]) != self.hidden_size:
                raise ValueError(
                    f"intent source {name!r} must be [B,N,{self.hidden_size}], got {tuple(value.shape)}"
                )
            if int(value.shape[0]) != int(reference.shape[0]) or int(value.shape[1]) <= 0:
                raise ValueError(
                    f"intent source {name!r} must be nonempty with batch={int(reference.shape[0])}, "
                    f"got {tuple(value.shape)}"
                )
            source = _scaled_contract_view(value, grad_scale)
            projected = self.intent_token_proj[name](
                source.to(device=reference.device, dtype=reference.dtype)
            )
            projected = projected + self.intent_source_embed[:, index : index + 1].to(
                device=reference.device, dtype=reference.dtype
            )
            summary_parts.append(projected.mean(dim=1))
            counts[name] = int(projected.shape[1])
        return torch.cat(summary_parts, dim=-1), counts

    def _layer_summary(
        self, entry: dict[str, Tensor], layer_index: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        world: list[Tensor] = []
        consequence: list[Tensor] = []
        grad_scale = float(self.config.hierarchical_mmdit_layer_grad_scale)
        for key_index, key in enumerate(self._LAYER_KEYS):
            if key in self._DISALLOWED_LAYER_KEYS:
                continue
            value = entry.get(key)
            if (
                not isinstance(value, Tensor)
                or value.ndim != 3
                or int(value.shape[-1]) != self.hidden_size
            ):
                continue
            pooled = _scaled_contract_view(value, grad_scale).mean(dim=1)
            typed = self.layer_key_proj[key_index](pooled)
            typed = typed + self.layer_key_embed[:, key_index].to(
                device=typed.device, dtype=typed.dtype
            )
            (world if key in self._WORLD_KEYS else consequence).append(typed)
        if not world:
            raise RuntimeError(f"layer contract {layer_index} has no world-summary source")

        def select(values: list[Tensor], gate: nn.Module, ref: Tensor) -> Tensor:
            if not values:
                return torch.zeros_like(ref)
            stack = torch.stack(values, dim=1)
            weight = torch.softmax(gate(stack).float(), dim=1).to(dtype=stack.dtype)
            return (stack * weight).sum(dim=1)

        world_summary = select(world, self.world_key_gate, world[0])
        consequence_summary = select(consequence, self.consequence_key_gate, world_summary)
        scale = float(self.config.hierarchical_mmdit_consequence_scale_max) * torch.sigmoid(
            self.consequence_scale_logit
        ).to(device=world_summary.device, dtype=world_summary.dtype)
        combined = world_summary + scale * consequence_summary
        layer = self.layer_proj[layer_index](combined)
        layer = layer + self.layer_embed[:, layer_index].to(device=layer.device, dtype=layer.dtype)
        return self.layer_stack_norm(layer), world_summary, consequence_summary

    def forward(
        self,
        *,
        trajectory_tokens: Tensor,
        trajectory_workspace_tokens: Tensor,
        rollout_tokens: Tensor,
        transition_memory: Tensor | list[Tensor] | tuple[Tensor, ...],
        event_evidence: Tensor,
        state_memory: Tensor | list[Tensor] | tuple[Tensor, ...],
        intent_memory: dict[str, Tensor],
        layer_contracts: list[dict[str, Tensor]],
    ) -> dict[str, Tensor | dict[str, Tensor]]:
        if len(layer_contracts) != self.depth:
            raise RuntimeError(
                f"hierarchical MMDiT requires {self.depth} ordered layer contracts, got {len(layer_contracts)}"
            )
        if trajectory_tokens.ndim != 3 or int(trajectory_tokens.shape[-1]) != self.hidden_size:
            raise ValueError(
                f"trajectory_tokens must be [B,T,H], got {tuple(trajectory_tokens.shape)}"
            )
        reference = trajectory_tokens
        trajectory_evidence = self._project_memory(
            trajectory_workspace_tokens,
            projector=self.trajectory_token_proj,
            reference=reference,
            input_dim=self.hidden_size,
        )
        rollout_evidence = self._project_memory(
            rollout_tokens,
            projector=self.rollout_token_proj,
            reference=reference,
            input_dim=self.hidden_size,
        )
        transition_evidence = self._project_memory(
            transition_memory,
            projector=self.transition_token_proj,
            reference=reference,
            input_dim=self.hidden_size,
        )
        event_tokens = self._project_memory(
            event_evidence,
            projector=self.event_token_proj,
            reference=reference,
            input_dim=3,
        )
        state_tokens = self._project_memory(
            state_memory,
            projector=self.state_token_proj,
            reference=reference,
            input_dim=self.hidden_size,
        )
        intent_summary_input, intent_counts = self._project_intent_memory(
            intent_memory,
            reference=reference,
        )
        required = {
            "trajectory": trajectory_evidence,
            "rollout": rollout_evidence,
            "transition": transition_evidence,
            "event": event_tokens,
            "state": state_tokens,
        }
        empty = [name for name, value in required.items() if int(value.shape[1]) == 0]
        if empty:
            raise RuntimeError("owned evidence sources cannot be empty: " + ", ".join(empty))

        layer_rows: list[Tensor] = []
        world_rows: list[Tensor] = []
        consequence_rows: list[Tensor] = []
        for index, entry in enumerate(layer_contracts):
            layer, world, consequence = self._layer_summary(entry, index)
            layer_rows.append(layer)
            world_rows.append(world)
            consequence_rows.append(consequence)
        layer_stack = torch.stack(layer_rows, dim=1)
        scan = self.layer_scan_init.to(device=reference.device, dtype=reference.dtype).expand(
            int(reference.shape[0]), -1
        )
        for index in range(self.depth):
            scan = self.layer_scan(layer_stack[:, index], scan)
        scan = self.layer_stack_norm(scan)

        trajectory_summary = self.trajectory_token_proj(
            _scaled_contract_view(
                trajectory_tokens, float(self.config.hierarchical_mmdit_source_grad_scale)
            )
        ).mean(dim=1)
        geom_summary = self.geom_summary_proj(trajectory_summary)
        global_summary = self.global_summary_proj(intent_summary_input)
        transition_summary = self.transition_summary_proj(transition_evidence.mean(dim=1))
        event_summary = self.event_summary_proj(event_tokens.mean(dim=1))
        state_summary = self.state_summary_proj(state_tokens.mean(dim=1))
        consequence_scale = float(
            self.config.hierarchical_mmdit_consequence_scale_max
        ) * torch.sigmoid(self.consequence_scale_logit)
        evidence_sources = {
            "layer": layer_stack,
            "trajectory": trajectory_evidence,
            "rollout": rollout_evidence,
            "transition": transition_evidence,
            "event": event_tokens,
            "state": state_tokens,
        }
        metrics = {
            "intent_layer_stack_norm": layer_stack.detach().float().norm(dim=-1).mean(),
            "intent_layer_scan_norm": scan.detach().float().norm(dim=-1).mean(),
            "intent_layer_world_norm": torch.stack(world_rows).detach().float().norm(dim=-1).mean(),
            "intent_layer_consequence_norm": torch.stack(consequence_rows)
            .detach()
            .float()
            .norm(dim=-1)
            .mean(),
            "intent_consequence_scale": consequence_scale.detach().float(),
            "intent_geom_summary_norm": geom_summary.detach().float().norm(dim=-1).mean(),
            "intent_global_summary_norm": global_summary.detach().float().norm(dim=-1).mean(),
            "intent_transition_summary_norm": transition_summary.detach()
            .float()
            .norm(dim=-1)
            .mean(),
            "intent_event_summary_norm": event_summary.detach().float().norm(dim=-1).mean(),
            "intent_state_summary_norm": state_summary.detach().float().norm(dim=-1).mean(),
        }
        for name, value in evidence_sources.items():
            metrics[f"owned_workspace_source_{name}_tokens"] = torch.tensor(
                float(value.shape[1]), device=reference.device, dtype=torch.float32
            )
        for name, count in intent_counts.items():
            metrics[f"intent_source_{name}_tokens"] = torch.tensor(
                float(count), device=reference.device, dtype=torch.float32
            )
        return {
            "layer_scan": scan,
            "geom_summary": geom_summary,
            "global_summary": global_summary,
            "transition_summary": transition_summary,
            "event_summary": event_summary,
            "state_summary": state_summary,
            "evidence_sources": evidence_sources,
            "metrics": metrics,
        }


class IndependentIntentFusion(nn.Module):
    """One contract-specific vector fusion with no slot-template output."""

    def __init__(self, hidden_size: int, source_count: int) -> None:
        super().__init__()
        h = int(hidden_size)
        self.source_count = int(source_count)
        self.net = nn.Sequential(
            nn.LayerNorm(self.source_count * h),
            nn.Linear(self.source_count * h, h),
            nn.SiLU(),
            nn.Linear(h, h),
        )
        self.out_norm = nn.LayerNorm(h, elementwise_affine=False)

    def forward(self, *sources: Tensor) -> Tensor:
        if len(sources) != self.source_count:
            raise ValueError(
                f"intent fusion expected {self.source_count} sources, got {len(sources)}"
            )
        return self.out_norm(self.net(torch.cat(list(sources), dim=-1)))


class IntentContractCompiler(nn.Module):
    """Deterministic replacement for the historical CVAE latent contract.

    Global intent, stage initialization, and evidence-read selection are
    compiled by separate functions.  The API deliberately has no target,
    noisy action, diffusion time, random sample, posterior, or KL term.
    """

    def __init__(self, config: PolicyIntentConfig) -> None:
        super().__init__()
        h = int(config.hidden_size)
        self.global_fusion = IndependentIntentFusion(h, 3)
        self.stage_fusion = IndependentIntentFusion(h, 3)
        self.read_fusion = IndependentIntentFusion(h, 5)

    @staticmethod
    def _cosine(a: Tensor, b: Tensor) -> Tensor:
        return F.cosine_similarity(a.detach().float(), b.detach().float(), dim=-1).mean()

    @staticmethod
    def _batch_diversity(value: Tensor) -> Tensor:
        detached = value.detach().float()
        return (detached - detached.mean(dim=0, keepdim=True)).norm(dim=-1).mean()

    def forward(
        self,
        *,
        layer_scan: Tensor,
        geom_summary: Tensor,
        global_summary: Tensor,
        transition_summary: Tensor,
        event_summary: Tensor,
        state_summary: Tensor,
    ) -> dict[str, Tensor]:
        # Global and stage values are compiled exclusively from pre-DiT,
        # deploy-safe sources.  Post-DiT layer/transition/event summaries can
        # depend on the current flow sample; they may steer retrieval through
        # read_contract, but cannot write noisy-action content into intent or
        # persistent stage initialization.
        global_intent = self.global_fusion(global_summary, geom_summary, state_summary)
        stage_contract = self.stage_fusion(global_summary, geom_summary, state_summary)
        read_contract = self.read_fusion(
            layer_scan,
            geom_summary,
            transition_summary,
            state_summary,
            event_summary,
        )
        return {
            "global_intent": global_intent,
            "stage_contract": stage_contract,
            "read_contract": read_contract,
            "intent_global_norm": global_intent.detach().float().norm(dim=-1).mean(),
            "intent_stage_contract_norm": stage_contract.detach().float().norm(dim=-1).mean(),
            "intent_read_contract_norm": read_contract.detach().float().norm(dim=-1).mean(),
            "intent_global_stage_cosine": self._cosine(global_intent, stage_contract),
            "intent_global_read_cosine": self._cosine(global_intent, read_contract),
            "intent_stage_read_cosine": self._cosine(stage_contract, read_contract),
            "intent_global_batch_diversity": self._batch_diversity(global_intent),
            "intent_stage_batch_diversity": self._batch_diversity(stage_contract),
            "intent_read_batch_diversity": self._batch_diversity(read_contract),
            "intent_global_dynamic_inputs": torch.zeros(
                (), device=global_intent.device, dtype=torch.float32
            ),
            "intent_stage_dynamic_inputs": torch.zeros(
                (), device=global_intent.device, dtype=torch.float32
            ),
            "intent_read_selector_only": torch.ones(
                (), device=global_intent.device, dtype=torch.float32
            ),
            "intent_contract_deterministic": torch.ones(
                (), device=global_intent.device, dtype=torch.float32
            ),
        }
