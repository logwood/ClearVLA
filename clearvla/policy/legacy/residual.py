from __future__ import annotations

"""Legacy residual action decoders retained for checkpoint migration."""

from typing import Any

import torch
from torch import Tensor, nn

from ..codec import TransitionAwarePhysicalVelocityHead
from ..primitives import BiasFreeFFN, TimeEmbedding


LegacyPolicyConfig = Any


class V37StyleResidualActionBlock(nn.Module):
    """V37-style action/high/event token block for residual refinement.

    This deliberately reuses the useful V37 pattern: a compact set of high-level
    slots, horizon action tokens, and event tokens exchange information through
    self-attention, then cross-attend to a high-bandwidth memory bank.  Unlike
    the failed full action-flow replacement, this block is downstream of the
    legacy V40.1 head and only predicts a small zero-initialized residual.
    """

    def __init__(self, config: LegacyPolicyConfig) -> None:
        super().__init__()
        h = int(config.hidden_size)
        self.n1 = nn.LayerNorm(h, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(h, int(config.num_heads), batch_first=True, dropout=float(config.dropout))
        self.n2 = nn.LayerNorm(h, elementwise_affine=False)
        self.mem_norm = nn.LayerNorm(h)
        self.cross = nn.MultiheadAttention(h, int(config.num_heads), batch_first=True, dropout=float(config.dropout))
        self.n3 = nn.LayerNorm(h, elementwise_affine=False)
        self.ffn = BiasFreeFFN(h, config.ffn_expansion)
        self.drop = nn.Dropout(float(config.dropout))
        self.mod = nn.Linear(h, 9 * h)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    @staticmethod
    def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        return x * (1 + scale[:, None]) + shift[:, None]

    def forward(self, x: Tensor, memory: Tensor, time_emb: Tensor) -> Tensor:
        sa_s, sa_c, sa_g, ca_s, ca_c, ca_g, ff_s, ff_c, ff_g = self.mod(time_emb).chunk(9, dim=-1)
        value = self.n1(x)
        qk = self.modulate(value, sa_s, sa_c)
        update, _ = self.self_attn(qk, qk, value, need_weights=False)
        x = x + torch.tanh(sa_g)[:, None] * self.drop(update)
        query = self.modulate(self.n2(x), ca_s, ca_c)
        mem = self.mem_norm(memory)
        update, _ = self.cross(query, mem, mem, need_weights=False)
        x = x + torch.tanh(ca_g)[:, None] * self.drop(update)
        update = self.ffn(self.modulate(self.n3(x), ff_s, ff_c))
        return x + torch.tanh(ff_g)[:, None] * self.drop(update)


class V37StyleResidualActionFlowDenoiser(nn.Module):
    """Zero-start residual action-flow denoiser.

    The legacy V40.1 final velocity remains the base policy.  This module reads
    the same noisy physical action plus V40/V37-style latent memory and emits a
    residual physical velocity.  Its final velocity and event heads are
    zero-initialized, so a stable checkpoint is behavior-preserving at load time
    while gradients can immediately train the residual heads.
    """

    def __init__(self, config: LegacyPolicyConfig) -> None:
        super().__init__()
        self.config = config
        h = int(config.hidden_size)
        self.time = TimeEmbedding(h)
        self.high_slots = int(config.action_flow_residual_high_slots)
        self.horizon_query = nn.Parameter(torch.randn(1, int(config.action_horizon), h) * 0.02)
        self.high_query = nn.Parameter(torch.randn(1, self.high_slots, h) * 0.02)
        self.event_query = nn.Parameter(torch.randn(1, int(config.event_tokens), h) * 0.02)
        self.noisy_action_lift = nn.Sequential(
            nn.LayerNorm(int(config.physical_action_dim)),
            nn.Linear(int(config.physical_action_dim), h),
        )
        self.trajectory_seed = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
        self.memory_summary = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h))
        self.blocks = nn.ModuleList([V37StyleResidualActionBlock(config) for _ in range(int(config.action_flow_residual_depth))])
        self.velocity_head = TransitionAwarePhysicalVelocityHead(config)
        self.event_delta_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 3))
        self.motion_delta_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 1))
        self._zero_initialize_outputs()
        alpha = torch.full((int(config.action_horizon), 1), float(config.action_flow_residual_max_scale), dtype=torch.float32)
        first = max(int(config.first_execution_steps), 1)
        mid = max(int(config.mid_execution_steps), first)
        for i in range(int(config.action_horizon)):
            step = i + 1
            if step <= first:
                alpha[i, 0] = float(config.action_flow_residual_max_scale) * 0.25
            elif step <= mid:
                frac = float(step - first) / float(max(mid - first, 1))
                alpha[i, 0] = float(config.action_flow_residual_max_scale) * (0.25 + 0.75 * frac)
        self.register_buffer("residual_alpha", alpha[None], persistent=True)

    def _make_temporal_action_update_mask(self, config: LegacyPolicyConfig) -> Tensor:
        """Return [decoder_depth, horizon, 1] update gates for V41.1.

        When disabled this is all ones.  When enabled, near horizon tokens only
        update in the first ``near_depth`` blocks, mid tokens update through
        ``mid_depth``, and far tokens update through the full decoder.  High and
        event tokens are intentionally updated in every block; the mask only
        controls horizon action tokens, keeping one clean final action path.
        """
        depth = int(config.latent_action_decoder_depth)
        horizon = int(config.action_horizon)
        mask = torch.ones(depth, horizon, 1, dtype=torch.float32)
        if not int(getattr(config, "latent_action_temporal_depth", 0)):
            return mask
        near_steps = min(max(int(getattr(config, "latent_action_near_steps", 4)), 0), horizon)
        mid_steps = min(max(int(getattr(config, "latent_action_mid_steps", 8)), near_steps), horizon)
        near_depth = min(max(int(getattr(config, "latent_action_near_depth", 2)), 1), depth)
        mid_depth = min(max(int(getattr(config, "latent_action_mid_depth", 4)), near_depth), depth)
        # block index j updates token h only while j < active_depth(h)
        for j in range(depth):
            if near_steps > 0 and j >= near_depth:
                mask[j, :near_steps, :] = 0.0
            if mid_steps > near_steps and j >= mid_depth:
                mask[j, near_steps:mid_steps, :] = 0.0
        return mask

    def _zero_initialize_outputs(self) -> None:
        for module in self.velocity_head.output_layers():
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)
        for seq in (self.event_delta_head, self.motion_delta_head):
            last = seq[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

    def forward(
        self,
        *,
        noisy_physical: Tensor,
        time: Tensor,
        trajectory_pooled: Tensor,
        memory: Tensor,
    ) -> dict[str, Tensor]:
        cfg = self.config
        batch = int(noisy_physical.shape[0])
        dtype = noisy_physical.dtype
        device = noisy_physical.device
        mem_summary = self.memory_summary(memory.mean(dim=1))
        action_tokens = (
            self.horizon_query.to(device=device, dtype=dtype).expand(batch, -1, -1)
            + self.noisy_action_lift(noisy_physical)
            + self.trajectory_seed(trajectory_pooled)
            + mem_summary[:, None]
        )
        high_tokens = self.high_query.to(device=device, dtype=dtype).expand(batch, -1, -1) + mem_summary[:, None]
        event_tokens = self.event_query.to(device=device, dtype=dtype).expand(batch, -1, -1) + mem_summary[:, None]
        tokens = torch.cat([high_tokens, action_tokens, event_tokens], dim=1)
        high_slice = slice(0, self.high_slots)
        action_slice = slice(self.high_slots, self.high_slots + int(cfg.action_horizon))
        event_slice = slice(self.high_slots + int(cfg.action_horizon), self.high_slots + int(cfg.action_horizon) + int(cfg.event_tokens))
        time_emb = self.time(time.to(dtype=dtype)) + mem_summary
        for block in self.blocks:
            tokens = block(tokens, memory, time_emb)
        high = tokens[:, high_slice]
        action = tokens[:, action_slice]
        event = tokens[:, event_slice]
        transition = high.mean(dim=1, keepdim=True).expand(-1, int(cfg.action_horizon), -1)
        # Let event tokens influence the transition latent without making event
        # logits a detached side branch, unlike the original V37 implementation.
        transition = transition + event.mean(dim=1, keepdim=True).expand_as(transition)
        raw_residual = self.velocity_head(action, transition)
        alpha = self.residual_alpha.to(device=device, dtype=dtype)
        residual_velocity = raw_residual * alpha
        return {
            "residual_velocity": residual_velocity,
            "raw_residual_velocity": raw_residual,
            "residual_alpha": alpha,
            "event_delta_logits": self.event_delta_head(action),
            "motion_delta_logits": self.motion_delta_head(action).squeeze(-1),
            "action_tokens": action,
            "high_tokens": high,
            "event_tokens": event,
            "transition_latent": transition,
            "residual_norm": residual_velocity.detach().float().norm(dim=-1).mean(),
            "raw_residual_norm": raw_residual.detach().float().norm(dim=-1).mean(),
            "alpha_mean": alpha.detach().float().mean(),
        }


def _parse_layer_pair_schedule(spec: str, *, decoder_depth: int, num_layers: int) -> list[tuple[int, int]]:
    """Parse a compact layer-pair schedule like ``0:1,1:3,3:5,5:7``.

    Pairs are clamped to available V40 layers.  If fewer pairs than residual
    blocks are provided, the last pair is repeated.  This keeps command-line
    experimentation simple while making the default explicitly hierarchical.
    """

    depth = max(int(decoder_depth), 1)
    layers = max(int(num_layers), 1)
    pairs: list[tuple[int, int]] = []
    for chunk in str(spec or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"invalid layer pair '{chunk}', expected A:B")
        left, right = chunk.split(":", 1)
        try:
            a = int(left)
            b = int(right)
        except ValueError as exc:
            raise ValueError(f"invalid layer pair '{chunk}', expected integer A:B") from exc
        a = min(max(a, 0), layers - 1)
        b = min(max(b, 0), layers - 1)
        pairs.append((a, b))
    if not pairs:
        if depth == 1:
            pairs = [(0, layers - 1)]
        else:
            pairs = []
            for j in range(depth):
                a = round(j * (layers - 1) / max(depth, 1))
                b = round((j + 1) * (layers - 1) / max(depth, 1))
                pairs.append((min(a, layers - 1), min(max(b, a), layers - 1)))
    while len(pairs) < depth:
        pairs.append(pairs[-1])
    return pairs[:depth]


class LayeredV37StyleResidualActionFlowDenoiser(nn.Module):
    """Layer-pair progressive V37 residual action-flow denoiser.

    This is the non-hand-wavy version of hierarchical injection:

    * keep the stable V40.1 legacy velocity as the base policy;
    * collect token-level memories from every V40.1 contract layer;
    * for residual block j, build a memory from bottom anchor L0, pair La/Lb,
      and the token-level difference Lb-La;
    * optionally add a learnable local stage router initialized near the pair,
      so the model can move the hierarchy if the learned V40 layers do not match
      our hand-written schedule;
    * zero-initialize residual/event/motion heads so loading a stable checkpoint
      is behavior-preserving at step 0.
    """

    _LAYER_KEYS = (
        "rollout_tokens",
        "trajectory_pooled",
        "rollout_effect_pred",
        "rollout_delta_pred",
        "policy_effect_tokens",
        "policy_effect_time_tokens",
        "unified_intervention_latent_pred",
        "neutral_latent_pred",
        "milestone_step_delta_pred",
    )

    def __init__(self, config: LegacyPolicyConfig) -> None:
        super().__init__()
        self.config = config
        h = int(config.hidden_size)
        self.hidden_size = h
        self.depth = int(config.action_flow_residual_depth)
        self.high_slots = int(config.action_flow_residual_high_slots)
        self.time = TimeEmbedding(h)
        self.horizon_query = nn.Parameter(torch.randn(1, int(config.action_horizon), h) * 0.02)
        self.high_query = nn.Parameter(torch.randn(1, self.high_slots, h) * 0.02)
        self.event_query = nn.Parameter(torch.randn(1, int(config.event_tokens), h) * 0.02)
        self.noisy_action_lift = nn.Sequential(
            nn.LayerNorm(int(config.physical_action_dim)),
            nn.Linear(int(config.physical_action_dim), h),
        )
        self.trajectory_seed = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
        self.global_summary = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h))
        self.stage_summary = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h))
            for _ in range(self.depth)
        ])
        self.blocks = nn.ModuleList([V37StyleResidualActionBlock(config) for _ in range(self.depth)])
        self.velocity_head = TransitionAwarePhysicalVelocityHead(config)
        self.event_delta_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 3))
        self.motion_delta_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 1))
        self._zero_initialize_outputs()
        pairs = _parse_layer_pair_schedule(
            str(getattr(config, "action_flow_residual_layer_pair_schedule", "0:1,1:3,3:5,5:7")),
            decoder_depth=self.depth,
            num_layers=int(config.depth),
        )
        self.layer_pairs = tuple(pairs)
        if int(getattr(config, "action_flow_residual_stage_router", 1)):
            logits = torch.empty(self.depth, int(config.depth), dtype=torch.float32)
            for j, (a, b) in enumerate(pairs):
                center = 0.5 * float(a + b)
                for k in range(int(config.depth)):
                    logits[j, k] = -1.25 * abs(float(k) - center)
                logits[j, a] += 1.0
                logits[j, b] += 1.0
            self.stage_router_logits = nn.Parameter(logits)
        else:
            self.register_parameter("stage_router_logits", None)
        alpha = torch.full((int(config.action_horizon), 1), float(config.action_flow_residual_max_scale), dtype=torch.float32)
        first = max(int(config.first_execution_steps), 1)
        mid = max(int(config.mid_execution_steps), first)
        for i in range(int(config.action_horizon)):
            step = i + 1
            if step <= first:
                alpha[i, 0] = float(config.action_flow_residual_max_scale) * 0.25
            elif step <= mid:
                frac = float(step - first) / float(max(mid - first, 1))
                alpha[i, 0] = float(config.action_flow_residual_max_scale) * (0.25 + 0.75 * frac)
        self.register_buffer("residual_alpha", alpha[None], persistent=True)

    def _make_temporal_action_update_mask(self, config: LegacyPolicyConfig) -> Tensor:
        """Return [decoder_depth, horizon, 1] update gates for V41.1.

        When disabled this is all ones.  When enabled, near horizon tokens only
        update in the first ``near_depth`` blocks, mid tokens update through
        ``mid_depth``, and far tokens update through the full decoder.  High and
        event tokens are intentionally updated in every block; the mask only
        controls horizon action tokens, keeping one clean final action path.
        """
        depth = int(config.latent_action_decoder_depth)
        horizon = int(config.action_horizon)
        mask = torch.ones(depth, horizon, 1, dtype=torch.float32)
        if not int(getattr(config, "latent_action_temporal_depth", 0)):
            return mask
        near_steps = min(max(int(getattr(config, "latent_action_near_steps", 4)), 0), horizon)
        mid_steps = min(max(int(getattr(config, "latent_action_mid_steps", 8)), near_steps), horizon)
        near_depth = min(max(int(getattr(config, "latent_action_near_depth", 2)), 1), depth)
        mid_depth = min(max(int(getattr(config, "latent_action_mid_depth", 4)), near_depth), depth)
        # block index j updates token h only while j < active_depth(h)
        for j in range(depth):
            if near_steps > 0 and j >= near_depth:
                mask[j, :near_steps, :] = 0.0
            if mid_steps > near_steps and j >= mid_depth:
                mask[j, near_steps:mid_steps, :] = 0.0
        return mask

    def _zero_initialize_outputs(self) -> None:
        for module in self.velocity_head.output_layers():
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)
        for seq in (self.event_delta_head, self.motion_delta_head):
            last = seq[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

    def _layer_entry_memory(self, entry: dict[str, Tensor], *, detach: bool) -> Tensor | None:
        parts: list[Tensor] = []
        for key in self._LAYER_KEYS:
            value = entry.get(key)
            if not isinstance(value, Tensor) or value.ndim != 3 or int(value.shape[-1]) != self.hidden_size:
                continue
            parts.append(value.detach() if detach else value)
        if not parts:
            return None
        # Same ordered keys for every layer => same token length in normal V40.1.
        # If a future variant drops a key in one layer, the caller truncates router
        # mixing to the common token length.
        return torch.cat(parts, dim=1)

    @staticmethod
    def _truncate_all(memories: list[Tensor]) -> list[Tensor]:
        min_len = min(int(m.shape[1]) for m in memories)
        return [m[:, :min_len] for m in memories]

    def _router_memory(self, layer_memories: list[Tensor], stage_index: int) -> tuple[Tensor | None, Tensor, Tensor]:
        ref = layer_memories[0]
        z = torch.zeros((), device=ref.device, dtype=ref.dtype)
        if self.stage_router_logits is None or len(layer_memories) < 1:
            return None, z, z
        usable = self._truncate_all(layer_memories)
        stack = torch.stack(usable, dim=1)  # [B,L,N,H]
        logits = self.stage_router_logits[stage_index, : len(usable)].to(device=ref.device, dtype=torch.float32)
        weights = torch.softmax(logits, dim=0).to(device=ref.device, dtype=stack.dtype)
        mixed = torch.einsum("l,blnh->bnh", weights, stack)
        wf = weights.detach().float().clamp_min(1e-8)
        entropy = -(wf * wf.log()).sum()
        max_weight = wf.max()
        return mixed, entropy.to(device=ref.device, dtype=ref.dtype), max_weight.to(device=ref.device, dtype=ref.dtype)

    def _pair_delta(self, a: Tensor, b: Tensor) -> Tensor:
        n = min(int(a.shape[1]), int(b.shape[1]))
        return b[:, :n] - a[:, :n]

    def _build_stage_memory(
        self,
        *,
        stage_index: int,
        context_memory: Tensor,
        transition_memory: Tensor | None,
        visual_memory: Tensor | None,
        layer_memories: list[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor], dict[str, Tensor]]:
        cfg = self.config
        parts: list[Tensor] = []
        if stage_index == 0 and int(getattr(cfg, "action_flow_residual_context_memory", 1)):
            parts.append(context_memory)
        if stage_index == 1 and int(getattr(cfg, "action_flow_residual_visual_memory", 1)) and visual_memory is not None:
            parts.append(visual_memory)
        if stage_index >= 2 and int(getattr(cfg, "action_flow_residual_transition_memory", 1)) and transition_memory is not None:
            parts.append(transition_memory)
        router_entropy = torch.zeros((), device=context_memory.device, dtype=context_memory.dtype)
        router_max = torch.zeros((), device=context_memory.device, dtype=context_memory.dtype)
        if int(getattr(cfg, "action_flow_residual_layer_memory", 1)) and layer_memories:
            a, b = self.layer_pairs[stage_index]
            a = min(a, len(layer_memories) - 1)
            b = min(b, len(layer_memories) - 1)
            anchor = layer_memories[0]
            mem_a = layer_memories[a]
            mem_b = layer_memories[b]
            if int(getattr(cfg, "action_flow_residual_anchor_memory", 1)):
                parts.append(anchor)
            parts.extend([mem_a, mem_b, self._pair_delta(mem_a, mem_b)])
            mixed, router_entropy, router_max = self._router_memory(layer_memories, stage_index)
            if mixed is not None:
                parts.append(mixed)
        if not parts:
            parts.append(context_memory)
        return torch.cat(parts, dim=1), router_entropy, router_max

    def forward(
        self,
        *,
        noisy_physical: Tensor,
        time: Tensor,
        trajectory_pooled: Tensor,
        context_memory: Tensor,
        transition_memory: Tensor | None,
        visual_memory: Tensor | None,
        layer_contracts: list[dict[str, Tensor]],
    ) -> dict[str, Tensor]:
        cfg = self.config
        batch = int(noisy_physical.shape[0])
        dtype = noisy_physical.dtype
        device = noisy_physical.device
        context_memory = context_memory.to(device=device, dtype=dtype)
        transition_memory = None if transition_memory is None else transition_memory.to(device=device, dtype=dtype)
        visual_memory = None if visual_memory is None else visual_memory.to(device=device, dtype=dtype)
        detach_layers = bool(int(getattr(cfg, "action_flow_residual_layer_detach", 1)))
        layer_memories: list[Tensor] = []
        for entry in layer_contracts:
            memory = self._layer_entry_memory(entry, detach=detach_layers)
            if memory is not None:
                layer_memories.append(memory.to(device=device, dtype=dtype))
        stage_memories: list[Tensor] = []
        entropies: list[Tensor] = []
        max_weights: list[Tensor] = []
        for j in range(self.depth):
            mem_j, ent_j, max_j = self._build_stage_memory(
                stage_index=j,
                context_memory=context_memory,
                transition_memory=transition_memory,
                visual_memory=visual_memory,
                layer_memories=layer_memories,
            )
            stage_memories.append(mem_j)
            entropies.append(ent_j)
            max_weights.append(max_j)
        global_summary = torch.stack([m.mean(dim=1) for m in stage_memories], dim=1).mean(dim=1)
        mem_summary = self.global_summary(global_summary)
        action_tokens = (
            self.horizon_query.to(device=device, dtype=dtype).expand(batch, -1, -1)
            + self.noisy_action_lift(noisy_physical)
            + self.trajectory_seed(trajectory_pooled)
            + mem_summary[:, None]
        )
        high_tokens = self.high_query.to(device=device, dtype=dtype).expand(batch, -1, -1) + mem_summary[:, None]
        event_tokens = self.event_query.to(device=device, dtype=dtype).expand(batch, -1, -1) + mem_summary[:, None]
        tokens = torch.cat([high_tokens, action_tokens, event_tokens], dim=1)
        high_slice = slice(0, self.high_slots)
        action_slice = slice(self.high_slots, self.high_slots + int(cfg.action_horizon))
        event_slice = slice(self.high_slots + int(cfg.action_horizon), self.high_slots + int(cfg.action_horizon) + int(cfg.event_tokens))
        time_base = self.time(time.to(dtype=dtype))
        for j, block in enumerate(self.blocks):
            stage_summary = self.stage_summary[j](stage_memories[j].mean(dim=1))
            tokens = block(tokens, stage_memories[j], time_base + stage_summary)
        high = tokens[:, high_slice]
        action = tokens[:, action_slice]
        event = tokens[:, event_slice]
        transition = high.mean(dim=1, keepdim=True).expand(-1, int(cfg.action_horizon), -1)
        transition = transition + event.mean(dim=1, keepdim=True).expand_as(transition)
        raw_residual = self.velocity_head(action, transition)
        alpha = self.residual_alpha.to(device=device, dtype=dtype)
        residual_velocity = raw_residual * alpha
        router_entropy = torch.stack(entropies).mean() if entropies else torch.zeros((), device=device, dtype=dtype)
        router_max = torch.stack(max_weights).mean() if max_weights else torch.zeros((), device=device, dtype=dtype)
        temporal_action_update_mean = (
            torch.stack([v.to(device=device, dtype=torch.float32) for v in action_update_means]).mean()
            if action_update_means else torch.ones((), device=device, dtype=torch.float32)
        )
        return {
            "residual_velocity": residual_velocity,
            "raw_residual_velocity": raw_residual,
            "residual_alpha": alpha,
            "event_delta_logits": self.event_delta_head(action),
            "motion_delta_logits": self.motion_delta_head(action).squeeze(-1),
            "action_tokens": action,
            "high_tokens": high,
            "event_tokens": event,
            "transition_latent": transition,
            "residual_norm": residual_velocity.detach().float().norm(dim=-1).mean(),
            "raw_residual_norm": raw_residual.detach().float().norm(dim=-1).mean(),
            "alpha_mean": alpha.detach().float().mean(),
            "stage_router_entropy": router_entropy.detach().float(),
            "stage_router_max": router_max.detach().float(),
        }
