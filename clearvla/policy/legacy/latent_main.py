from __future__ import annotations

"""Legacy hierarchical latent-main decoder."""

from typing import Any

import torch
from torch import Tensor, nn

from ..codec import TransitionAwarePhysicalVelocityHead
from ..primitives import BiasFreeFFN, TimeEmbedding
from .residual import LayeredV37StyleResidualActionFlowDenoiser, _parse_layer_pair_schedule


LegacyPolicyConfig = Any


class HierarchicalLatentActionBlock(nn.Module):
    """One block of the V41 latent-main action decoder.

    This is not a side branch.  It is the only final-action path in
    ``final_action_decoder=latent_main_action``.  Each block updates the same
    high/action/event tokens by self-attention, then cross-attends a stage memory
    built from V40 layer memories and controlled transition latents, and finally
    applies stage-conditioned AdaLN/FFN modulation.
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
        self.mod = nn.Linear(2 * h, 9 * h)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    @staticmethod
    def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        return x * (1 + scale[:, None]) + shift[:, None]

    def forward(self, x: Tensor, memory: Tensor, time_emb: Tensor, stage_summary: Tensor) -> Tensor:
        cond = torch.cat([time_emb, stage_summary], dim=-1)
        sa_s, sa_c, sa_g, ca_s, ca_c, ca_g, ff_s, ff_c, ff_g = self.mod(cond).chunk(9, dim=-1)
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


class HierarchicalLatentMainActionDecoder(nn.Module):
    """V41 clean latent-main final action decoder.

    The decoder replaces the old V40 direct/rollout action heads as the final
    policy.  It keeps the V40 trunk and contract latents, but final actions must
    pass through a single hierarchical action decoder.  Every available layer
    memory is injected as an all-layer summary token, while each block also gets
    full token-level memories from its scheduled layer pair and their delta.
    """

    _LAYER_KEYS = LayeredV37StyleResidualActionFlowDenoiser._LAYER_KEYS

    def __init__(self, config: LegacyPolicyConfig) -> None:
        super().__init__()
        self.config = config
        h = int(config.hidden_size)
        self.hidden_size = h
        self.depth = int(config.latent_action_decoder_depth)
        self.high_slots = int(config.latent_action_high_slots)
        self.time = TimeEmbedding(h)
        self.horizon_query = nn.Parameter(torch.randn(1, int(config.action_horizon), h) * 0.02)
        self.high_query = nn.Parameter(torch.randn(1, self.high_slots, h) * 0.02)
        self.event_query = nn.Parameter(torch.randn(1, int(config.event_tokens), h) * 0.02)
        self.noisy_action_lift = nn.Sequential(
            nn.LayerNorm(int(config.physical_action_dim)),
            nn.Linear(int(config.physical_action_dim), h),
        )
        self.trajectory_lift = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
        self.global_summary = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h))
        self.stage_summary = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h))
            for _ in range(self.depth)
        ])
        self.blocks = nn.ModuleList([HierarchicalLatentActionBlock(config) for _ in range(self.depth)])
        self.event_to_action = nn.MultiheadAttention(h, int(config.num_heads), batch_first=True, dropout=float(config.dropout))
        self.event_gate = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.Sigmoid())
        self.event_transition = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h))
        self.velocity_head = TransitionAwarePhysicalVelocityHead(config)
        self.event_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 3))
        self.motion_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 1))
        self._zero_initialize_outputs()
        self.layer_pairs = tuple(_parse_layer_pair_schedule(
            str(getattr(config, "latent_action_layer_schedule", "0:1,1:2,2:3,3:4,4:5,5:6,6:7,7:7")),
            decoder_depth=self.depth,
            num_layers=int(config.depth),
        ))
        self.register_buffer(
            "temporal_action_update_mask",
            self._make_temporal_action_update_mask(config),
            persistent=False,
        )
        if int(getattr(config, "latent_action_stage_router", 0)):
            logits = torch.empty(self.depth, int(config.depth), dtype=torch.float32)
            for j, (a, b) in enumerate(self.layer_pairs):
                center = 0.5 * float(a + b)
                for k in range(int(config.depth)):
                    logits[j, k] = -1.25 * abs(float(k) - center)
                logits[j, a] += 1.0
                logits[j, b] += 1.0
            self.stage_router_logits = nn.Parameter(logits)
        else:
            self.register_parameter("stage_router_logits", None)

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
        for seq in (self.event_head, self.motion_head):
            last = seq[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)
        last = self.event_transition[-1]
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
        return torch.cat(parts, dim=1)

    @staticmethod
    def _pair_delta(a: Tensor, b: Tensor) -> Tensor:
        n = min(int(a.shape[1]), int(b.shape[1]))
        return b[:, :n] - a[:, :n]

    @staticmethod
    def _truncate_all(memories: list[Tensor]) -> list[Tensor]:
        min_len = min(int(m.shape[1]) for m in memories)
        return [m[:, :min_len] for m in memories]

    def _router_memory(self, layer_memories: list[Tensor], stage_index: int) -> tuple[Tensor | None, Tensor, Tensor]:
        ref = layer_memories[0]
        z = torch.zeros((), device=ref.device, dtype=ref.dtype)
        if self.stage_router_logits is None or not layer_memories:
            return None, z, z
        usable = self._truncate_all(layer_memories)
        stack = torch.stack(usable, dim=1)
        logits = self.stage_router_logits[stage_index, : len(usable)].to(device=ref.device, dtype=torch.float32)
        weights = torch.softmax(logits, dim=0).to(device=ref.device, dtype=stack.dtype)
        mixed = torch.einsum("l,blnh->bnh", weights, stack)
        wf = weights.detach().float().clamp_min(1e-8)
        entropy = -(wf * wf.log()).sum()
        max_weight = wf.max()
        return mixed, entropy.to(device=ref.device, dtype=ref.dtype), max_weight.to(device=ref.device, dtype=ref.dtype)

    def _build_stage_memory(
        self,
        *,
        stage_index: int,
        context_memory: Tensor | None,
        transition_memory: Tensor | None,
        visual_memory: Tensor | None,
        layer_memories: list[Tensor],
        all_layer_summary: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        cfg = self.config
        parts: list[Tensor] = []
        ref: Tensor | None = None
        if all_layer_summary is not None:
            parts.append(all_layer_summary)  # one token per layer, every block: every latent is always injected.
            ref = all_layer_summary
        if int(getattr(cfg, "latent_action_layer_memory", 1)) and layer_memories:
            a, b = self.layer_pairs[stage_index]
            a = min(a, len(layer_memories) - 1)
            b = min(b, len(layer_memories) - 1)
            if int(getattr(cfg, "latent_action_anchor_memory", 1)):
                parts.append(layer_memories[0])
            mem_a = layer_memories[a]
            mem_b = layer_memories[b]
            parts.extend([mem_a, mem_b, self._pair_delta(mem_a, mem_b)])
            ref = mem_a
            mixed, ent, mx = self._router_memory(layer_memories, stage_index)
            if mixed is not None:
                parts.append(mixed)
            router_entropy, router_max = ent, mx
        else:
            base = context_memory if context_memory is not None else transition_memory if transition_memory is not None else visual_memory
            if base is None:
                raise RuntimeError("latent_main_action requires at least one memory source")
            parts.append(base)
            ref = base
            router_entropy = torch.zeros((), device=base.device, dtype=base.dtype)
            router_max = torch.zeros((), device=base.device, dtype=base.dtype)
        if int(getattr(cfg, "latent_action_transition_memory", 1)) and transition_memory is not None:
            parts.append(transition_memory)
        if int(getattr(cfg, "latent_action_context_memory", 0)) and context_memory is not None:
            parts.append(context_memory)
        if int(getattr(cfg, "latent_action_visual_memory", 0)) and visual_memory is not None:
            parts.append(visual_memory)
        if ref is None:
            ref = parts[0]
            router_entropy = torch.zeros((), device=ref.device, dtype=ref.dtype)
            router_max = torch.zeros((), device=ref.device, dtype=ref.dtype)
        return torch.cat(parts, dim=1), router_entropy, router_max

    def forward(
        self,
        *,
        noisy_physical: Tensor,
        time: Tensor,
        trajectory_tokens: Tensor,
        context_memory: Tensor | None,
        transition_memory: Tensor | None,
        visual_memory: Tensor | None,
        layer_contracts: list[dict[str, Tensor]],
    ) -> dict[str, Tensor]:
        cfg = self.config
        batch = int(noisy_physical.shape[0])
        dtype = noisy_physical.dtype
        device = noisy_physical.device
        context_memory = None if context_memory is None else context_memory.to(device=device, dtype=dtype)
        transition_memory = None if transition_memory is None else transition_memory.to(device=device, dtype=dtype)
        visual_memory = None if visual_memory is None else visual_memory.to(device=device, dtype=dtype)
        detach_layers = bool(int(getattr(cfg, "latent_action_layer_detach", 0)))
        layer_memories: list[Tensor] = []
        for entry in layer_contracts:
            memory = self._layer_entry_memory(entry, detach=detach_layers)
            if memory is not None:
                layer_memories.append(memory.to(device=device, dtype=dtype))
        if len(layer_memories) < int(cfg.depth) and int(getattr(cfg, "latent_action_layer_memory", 1)):
            # Hard diagnostic rather than silently skipping layers: the whole
            # point of V41 is to make every layer latent participate.
            raise RuntimeError(f"latent_main_action expected memories for {int(cfg.depth)} layers, got {len(layer_memories)}")
        all_layer_summary = None
        if layer_memories:
            all_layer_summary = torch.stack([m.mean(dim=1) for m in layer_memories], dim=1)
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
                all_layer_summary=all_layer_summary,
            )
            stage_memories.append(mem_j)
            entropies.append(ent_j)
            max_weights.append(max_j)
        global_seed = torch.stack([m.mean(dim=1) for m in stage_memories], dim=1).mean(dim=1)
        mem_summary = self.global_summary(global_seed)
        action_tokens = (
            self.horizon_query.to(device=device, dtype=dtype).expand(batch, -1, -1)
            + self.noisy_action_lift(noisy_physical)
            + self.trajectory_lift(trajectory_tokens)
            + mem_summary[:, None]
        )
        high_tokens = self.high_query.to(device=device, dtype=dtype).expand(batch, -1, -1) + mem_summary[:, None]
        event_tokens = self.event_query.to(device=device, dtype=dtype).expand(batch, -1, -1) + mem_summary[:, None]
        tokens = torch.cat([high_tokens, action_tokens, event_tokens], dim=1)
        high_slice = slice(0, self.high_slots)
        action_slice = slice(self.high_slots, self.high_slots + int(cfg.action_horizon))
        event_slice = slice(self.high_slots + int(cfg.action_horizon), self.high_slots + int(cfg.action_horizon) + int(cfg.event_tokens))
        time_emb = self.time(time.to(dtype=dtype))
        temporal_mask = self.temporal_action_update_mask.to(device=device, dtype=dtype)
        action_update_means: list[Tensor] = []
        for j, block in enumerate(self.blocks):
            stage_summary = self.stage_summary[j](stage_memories[j].mean(dim=1))
            tokens_new = block(tokens, stage_memories[j], time_emb, stage_summary)
            if int(getattr(cfg, "latent_action_temporal_depth", 0)):
                # High/event tokens remain deep global reasoning tokens.  Only
                # the action horizon tokens are depth-gated, so near actions are
                # shallow while far actions must pass through deeper rollout and
                # consequence injections.  This is a masked update inside the
                # one main decoder, not a side head.
                m = temporal_mask[j:j + 1]
                old_action = tokens[:, action_slice]
                new_action = tokens_new[:, action_slice]
                mixed_action = old_action + m * (new_action - old_action)
                tokens = torch.cat([
                    tokens_new[:, high_slice],
                    mixed_action,
                    tokens_new[:, event_slice],
                ], dim=1)
                action_update_means.append(m.detach().float().mean())
            else:
                tokens = tokens_new
                action_update_means.append(torch.ones((), device=device, dtype=torch.float32))
        high = tokens[:, high_slice]
        action = tokens[:, action_slice]
        event = tokens[:, event_slice]
        event_context, _ = self.event_to_action(action, event, event, need_weights=False)
        transition = high.mean(dim=1, keepdim=True).expand(-1, int(cfg.action_horizon), -1) + event_context
        if int(getattr(cfg, "latent_action_event_gripper_gate", 1)):
            gate = self.event_gate(event_context)
            transition = transition + gate * self.event_transition(action + event_context)
        else:
            gate = torch.zeros_like(action)
        pred_velocity = self.velocity_head(action, transition)
        event_logits = self.event_head(action + event_context)
        motion_logits = self.motion_head(action).squeeze(-1)
        router_entropy = torch.stack(entropies).mean() if entropies else torch.zeros((), device=device, dtype=dtype)
        router_max = torch.stack(max_weights).mean() if max_weights else torch.zeros((), device=device, dtype=dtype)
        temporal_action_update_mean = (
            torch.stack([v.to(device=device, dtype=torch.float32) for v in action_update_means]).mean()
            if action_update_means else torch.ones((), device=device, dtype=torch.float32)
        )
        return {
            "pred_velocity": pred_velocity,
            "event_logits": event_logits,
            "motion_logits": motion_logits,
            "action_tokens": action,
            "high_tokens": high,
            "event_tokens": event,
            "transition_latent": transition,
            "stage_router_entropy": router_entropy.detach().float(),
            "stage_router_max": router_max.detach().float(),
            "gripper_gate_mean": gate.detach().float().mean(),
            "layer_memory_count": torch.tensor(float(len(layer_memories)), device=device, dtype=dtype),
            "temporal_action_update_mean": temporal_action_update_mean.detach().float(),
            "temporal_near_depth": torch.tensor(float(getattr(cfg, "latent_action_near_depth", 0)), device=device, dtype=dtype),
            "temporal_mid_depth": torch.tensor(float(getattr(cfg, "latent_action_mid_depth", 0)), device=device, dtype=dtype),
        }
