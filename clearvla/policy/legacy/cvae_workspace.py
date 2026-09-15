from __future__ import annotations

"""Legacy CVAE workspace and condition layout."""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..evidence import EvidenceMemoryBank, PreparedEvidenceMemory, SemanticEvidenceWorkspaceBlock


LegacyPolicyConfig = Any


@dataclass(frozen=True)
class MMDiTConditionLayout:
    """Explicit condition-group slices in the condition-token coordinate."""

    noisy_start: int
    noisy_len: int
    rollout_start: int = 0
    rollout_len: int = 0
    low_start: int = 0
    low_len: int = 0
    stage_start: int = 0
    stage_len: int = 0


class WorkspaceController(nn.Module):
    """Central capacity and role controller for workspace memory retrieval.

    Firewall contract (red-line fix, v74b review): ``value_state`` is computed
    from condition+step ONLY and is the sole input to the workspace value
    FiLM; ``select_state`` may additionally read the action summary but feeds
    only selection-level controls (query modulation, role logits, capacity,
    delay, temperature).  Action content can therefore steer WHERE to read
    but can never write into WHAT the evidence says -- the same discipline
    enforced in v72's progress isolation and the hierarchical manager.
    """

    def __init__(self, config: LegacyPolicyConfig, role_names: tuple[str, ...]) -> None:
        super().__init__()
        h = int(config.hidden_size)
        self.role_names = tuple(role_names)
        self.state_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.action_state = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
        self.step_state = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
        self.workspace_mod = nn.Linear(h, 2 * h)
        self.query_mod = nn.Linear(h, 2 * h)
        self.role_head = nn.Linear(h, len(self.role_names))
        self.capacity_head = nn.Linear(h, 1)
        self.delay_head = nn.Linear(h, 1)
        self.temperature_head = nn.Linear(h, 1)
        for module in (self.workspace_mod, self.query_mod, self.role_head, self.capacity_head, self.delay_head, self.temperature_head):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

    @staticmethod
    def _bounded_modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        # Keep the controller a gentle manager rather than a second action head.
        return x * (1.0 + 0.10 * torch.tanh(scale)[:, None]) + 0.10 * torch.tanh(shift)[:, None]

    def forward(
        self,
        *,
        workspace: Tensor,
        action_query: Tensor,
        primary_cond: Tensor,
        step_context: Tensor,
        memory_bank: EvidenceMemoryBank,
        ranges: dict[str, tuple[int, int]],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        action_summary = action_query.mean(dim=1)
        step_state = self.step_state(step_context.to(device=primary_cond.device, dtype=primary_cond.dtype))
        value_state = self.state_norm(primary_cond + step_state)
        select_state = self.state_norm(primary_cond + step_state + self.action_state(action_summary))
        ws_shift, ws_scale = self.workspace_mod(value_state).chunk(2, dim=-1)
        q_shift, q_scale = self.query_mod(select_state).chunk(2, dim=-1)
        workspace = self._bounded_modulate(workspace, ws_shift, ws_scale)
        action_query = self._bounded_modulate(action_query, q_shift, q_scale)

        capacity_scale = 1.0 + 0.25 * torch.tanh(self.capacity_head(select_state)).squeeze(-1)
        delay_gate = torch.sigmoid(self.delay_head(select_state)).squeeze(-1)
        temperature = 0.5 + 1.5 * torch.sigmoid(self.temperature_head(select_state)).squeeze(-1)
        role_logits = self.role_head(select_state)
        gated_role_logits = role_logits * delay_gate[:, None] / temperature[:, None].clamp_min(1e-4)
        role_key_bias = memory_bank.role_key_bias(gated_role_logits, ranges)

        role_counts = memory_bank.role_token_counts(ranges)
        active_role_mask = torch.tensor(
            [role_counts.get(role, 0) > 0 for role in self.role_names],
            device=gated_role_logits.device,
            dtype=torch.bool,
        )
        masked_role_logits = gated_role_logits.float().masked_fill(~active_role_mask[None], -1e4)
        role_probs = torch.softmax(masked_role_logits, dim=-1)
        role_entropy = -(role_probs.clamp_min(1e-8) * role_probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
        metrics: dict[str, Tensor] = {
            "workspace_controller_capacity": capacity_scale.detach().float().mean(),
            "workspace_controller_delay": delay_gate.detach().float().mean(),
            "workspace_controller_temperature": temperature.detach().float().mean(),
            "workspace_controller_role_entropy": role_entropy.detach().float(),
            "workspace_controller_role_max": role_probs.detach().float().max(dim=-1).values.mean(),
            "workspace_controller_query_delta_norm": (0.10 * torch.tanh(q_shift)).detach().float().norm(dim=-1).mean(),
            "workspace_controller_workspace_delta_norm": (0.10 * torch.tanh(ws_shift)).detach().float().norm(dim=-1).mean(),
        }
        for index, role in enumerate(self.role_names):
            metrics[f"workspace_controller_role_{role}_prob"] = role_probs[:, index].detach().float().mean()
            metrics[f"workspace_controller_role_{role}_logit"] = gated_role_logits[:, index].detach().float().mean()
        return workspace, action_query, role_key_bias, capacity_scale, metrics


class SemanticEvidenceWorkspace(nn.Module):
    """Fuse typed semantic sources into a configurable horizon token field."""

    SOURCE_NAMES = EvidenceMemoryBank.SOURCE_NAMES

    def __init__(self, config: LegacyPolicyConfig) -> None:
        super().__init__()
        self.config = config
        h = int(config.hidden_size)
        self.hidden_size = h
        self.token_count = int(getattr(config, "latent_cvae_horizon_tokens", config.action_horizon))
        self.memory_bank = EvidenceMemoryBank(config)
        self.query = nn.Parameter(torch.randn(1, self.token_count, h) * 0.02)
        self.action_query_proj = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
        self.step_query_proj = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
        self.global_state_proj = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
        global_state = self.global_state_proj[-1]
        if isinstance(global_state, nn.Linear):
            # Time-state is a residual retrieval bias.  Start from the previous
            # workspace behavior and let training open this path deliberately;
            # random z/time injection can dominate the first batches.
            nn.init.zeros_(global_state.weight)
            nn.init.zeros_(global_state.bias)
        self.controller = (
            WorkspaceController(config, EvidenceMemoryBank.ROLE_NAMES)
            if int(getattr(config, "latent_cvae_workspace_controller", 0))
            else None
        )
        self.blocks = nn.ModuleList([SemanticEvidenceWorkspaceBlock(config) for _ in range(2)])
        self.final_norm = nn.LayerNorm(h, elementwise_affine=False)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        # V74A moved the source type embedding into EvidenceMemoryBank.  Keep
        # old v73 checkpoints loadable under both stage1 non-strict and resume
        # strict paths without exposing duplicate parameters.
        old_type_key = prefix + "type_embed"
        new_type_key = prefix + "memory_bank.type_embed"
        if old_type_key in state_dict and new_type_key not in state_dict:
            state_dict[new_type_key] = state_dict.pop(old_type_key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _resize_action(self, action: Tensor) -> Tensor:
        if int(action.shape[1]) == self.token_count:
            return action
        if self.token_count == 1:
            return action.mean(dim=1, keepdim=True)
        if int(action.shape[1]) == 1:
            return action.expand(-1, self.token_count, -1)
        return F.interpolate(
            action.transpose(1, 2).float(),
            size=self.token_count,
            mode="linear",
            align_corners=True,
        ).transpose(1, 2).to(dtype=action.dtype)

    def _slot_aware_global_state(self, global_state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Keep z/time workspace state global in meaning but slot-aware in form."""
        slot_state = global_state[:, None]
        zero = torch.zeros((), device=global_state.device, dtype=torch.float32)
        if (
            not int(getattr(self.config, "latent_cvae_workspace_slot_time_state", 1))
            or float(getattr(self.config, "latent_cvae_workspace_slot_time_scale", 0.10)) <= 0.0
            or self.token_count <= 1
        ):
            return slot_state, zero, zero
        scale = float(getattr(self.config, "latent_cvae_workspace_slot_time_scale", 0.10))
        # Use the learned workspace slot identity as a bounded selector.  This
        # preserves the global z/time signal while avoiding an identical vector
        # being added to every retrieval slot.
        slot_identity = F.layer_norm(
            self.query.to(device=global_state.device, dtype=global_state.dtype),
            (self.hidden_size,),
        )
        slot_delta = scale * global_state[:, None] * torch.tanh(slot_identity)
        slot_state = slot_state + slot_delta
        slot_delta_norm = slot_delta.detach().float().norm(dim=-1).mean()
        slot_diversity = (slot_state - slot_state.mean(dim=1, keepdim=True)).detach().float().norm(dim=-1).mean()
        return slot_state, slot_delta_norm, slot_diversity

    def _prepare_sources(
        self,
        sources: dict[str, Tensor],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        allow_empty: bool,
    ) -> tuple[Tensor | None, Tensor, dict[str, tuple[int, int]]]:
        return self.memory_bank.prepare_sources(
            sources,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            allow_empty=allow_empty,
        )

    def prepare_static_memory(
        self,
        sources: dict[str, Tensor],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> PreparedEvidenceMemory:
        return self.memory_bank.prepare_static_memory(
            sources,
            blocks=self.blocks,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        sources: dict[str, Tensor],
        *,
        action: Tensor,
        primary_cond: Tensor,
        step_context: Tensor,
        static_memory: PreparedEvidenceMemory | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        device = action.device
        dtype = action.dtype
        batch_size = int(action.shape[0])
        if static_memory is not None and static_memory.batch_size != batch_size:
            raise ValueError(
                f"cached workspace batch={static_memory.batch_size} does not match action batch={batch_size}"
            )
        dynamic_memory, dynamic_bias, dynamic_ranges = self._prepare_sources(
            sources,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            allow_empty=static_memory is not None,
        )
        if static_memory is None:
            assert dynamic_memory is not None
            ranges = dynamic_ranges
            key_bias = dynamic_bias
            static_token_count = 0
        else:
            overlap = set(static_memory.ranges).intersection(dynamic_ranges)
            if overlap:
                raise ValueError(f"workspace sources appear in both static and dynamic memory: {sorted(overlap)}")
            static_token_count = int(static_memory.key_bias.numel())
            ranges = dict(static_memory.ranges)
            ranges.update({name: (start + static_token_count, stop + static_token_count) for name, (start, stop) in dynamic_ranges.items()})
            key_bias = torch.cat([static_memory.key_bias.to(device=device), dynamic_bias], dim=0)
        action_query = self.action_query_proj(self._resize_action(action))
        step_query = self.step_query_proj(step_context.to(device=device, dtype=dtype))[:, None]
        if int(getattr(self.config, "latent_cvae_workspace_time_state", 0)):
            global_state = self.global_state_proj(primary_cond.to(device=device, dtype=dtype))
        else:
            global_state = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        global_slot_state, global_slot_delta_norm, global_slot_diversity = self._slot_aware_global_state(global_state)
        workspace = self.query.to(device=device, dtype=dtype).expand(int(action.shape[0]), -1, -1)
        workspace = workspace + step_query + global_slot_state
        workspace_seed = workspace
        read_scale: Tensor | None = None
        controller_metrics: dict[str, Tensor] = {}
        if self.controller is not None:
            workspace, action_query, role_bias, read_scale, controller_metrics = self.controller(
                workspace=workspace,
                action_query=action_query,
                primary_cond=primary_cond,
                step_context=step_context,
                memory_bank=self.memory_bank,
                ranges=ranges,
            )
            key_bias = key_bias.to(device=device) + role_bias.to(device=device, dtype=key_bias.dtype)
        weight_rows: list[Tensor] = []
        for block_index, block in enumerate(self.blocks):
            if dynamic_memory is None:
                dynamic_k = dynamic_v = None
            else:
                dynamic_k, dynamic_v = block.project_memory(dynamic_memory)
            if static_memory is None:
                assert dynamic_k is not None and dynamic_v is not None
                memory_k, memory_v = dynamic_k, dynamic_v
            else:
                static_k, static_v = static_memory.block_kv[block_index]
                memory_k = static_k if dynamic_k is None else torch.cat([static_k, dynamic_k], dim=2)
                memory_v = static_v if dynamic_v is None else torch.cat([static_v, dynamic_v], dim=2)
            workspace, weights = block(
                workspace,
                primary_cond,
                memory_k=memory_k,
                memory_v=memory_v,
                key_bias=key_bias,
                query_context=action_query,
                read_scale=read_scale,
            )
            weight_rows.append(weights.detach().float())
        workspace_pre_norm = workspace
        workspace = self.final_norm(workspace)
        weights = torch.stack(weight_rows).mean(dim=0)
        metrics: dict[str, Tensor] = {
            "workspace_token_count": torch.tensor(float(workspace.shape[1]), device=device, dtype=torch.float32),
            "workspace_token_norm": workspace.detach().float().norm(dim=-1).mean(),
            "workspace_update_norm": (workspace_seed.detach() - workspace_pre_norm.detach()).float().norm(dim=-1).mean(),
            "workspace_global_state_norm": global_state.detach().float().norm(dim=-1).mean(),
            "workspace_global_slot_delta_norm": global_slot_delta_norm,
            "workspace_global_slot_diversity": global_slot_diversity,
            "workspace_source_count": torch.tensor(float(len(ranges)), device=device, dtype=torch.float32),
            "workspace_cached_token_fraction": torch.tensor(
                float(static_token_count) / float(max(int(key_bias.numel()), 1)),
                device=device,
                dtype=torch.float32,
            ),
            "workspace_attention_entropy": -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=-1).mean(),
            "workspace_attention_max": weights.max(dim=-1).values.mean(),
        }
        metrics.update(controller_metrics)
        group_weights = torch.stack([
            weights[..., start:stop].sum(dim=-1)
            for start, stop in ranges.values()
        ], dim=-1)
        metrics["workspace_group_attention_entropy"] = -(
            group_weights.clamp_min(1e-8) * group_weights.clamp_min(1e-8).log()
        ).sum(dim=-1).mean()
        metrics["workspace_group_effective_sources"] = torch.exp(metrics["workspace_group_attention_entropy"])
        metrics["workspace_attention_mass_error"] = (group_weights.sum(dim=-1) - 1.0).abs().mean()
        for name, (start, stop) in ranges.items():
            metrics[f"workspace_{name}_attention"] = weights[..., start:stop].sum(dim=-1).mean()
        metrics.update(self.memory_bank.role_attention_metrics(weights, ranges))
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
        return workspace, metrics
