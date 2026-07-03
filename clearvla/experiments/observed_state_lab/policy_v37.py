from __future__ import annotations

"""V37 full-latent world-shaped policy.

V37 removes the explicit frozen-world-token bottleneck used by V35/V36 policy
adapters.  Dense DINO/state/history/action context is kept as a high-bandwidth
memory stream.  A small set of high-level slots is introduced only as an
internal work area: action tokens cross-attend to the full memory and to those
slots, while world/future objectives shape the slots without making them the
only action interface.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .policy import RejectableHistoryProposal, TimeEmbedding
from .policy_v36_2 import (
    ActionExpertBlock,
    HorizonRoleEmbedding,
    PhysicalActionCodec,
    PhysicalActionTokenLift,
    V362PolicyConfig,
)
from .policy_v36_3 import TransitionAwarePhysicalVelocityHead
from .world_model import BiasFreeFFN, sinusoidal_positions


@dataclass(frozen=True)
class V37PolicyConfig(V362PolicyConfig):
    """V37 policy config.

    ``visual_token_dim`` is the dense-DINO token dimension.  The policy never
    receives a compressed world-token product.  It receives the complete current
    dense visual history as key/value memory and learns an implicit hierarchy in
    the shared latent trunk.
    """

    visual_token_dim: int = 768
    visual_history_length: int = 3
    num_cameras: int = 2
    patches_per_camera: int = 576
    high_level_slots: int = 8
    future_aux_offsets: int = 4
    target_future_count: int = 12
    visual_memory_dropout: float = 0.0

    def validate(self) -> None:
        super().validate()
        if min(
            self.visual_token_dim,
            self.visual_history_length,
            self.num_cameras,
            self.patches_per_camera,
            self.high_level_slots,
            self.future_aux_offsets,
            self.target_future_count,
        ) <= 0:
            raise ValueError("V37 visual/high-level dimensions must be positive")
        if self.future_aux_offsets > self.target_future_count:
            raise ValueError("future_aux_offsets cannot exceed target_future_count")
        if not 0 <= self.visual_memory_dropout < 1:
            raise ValueError("visual_memory_dropout must be in [0,1)")

    @property
    def history_length(self) -> int:
        return self.visual_history_length

    @property
    def num_future(self) -> int:
        return self.target_future_count

    @property
    def latent_dim(self) -> int:
        return self.visual_token_dim



class DenseVisualMemory(nn.Module):
    """Project dense current DINO history into policy memory without bottlenecking.

    The output length is history * cameras * patches_per_camera.  This module is
    only a per-token projection plus factorized type embeddings; it does not
    pool visual tokens into a single world summary.
    """

    def __init__(self, config: V37PolicyConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size
        d = config.visual_token_dim
        self.proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, h))
        self.history_type = nn.Parameter(torch.randn(1, config.visual_history_length, 1, 1, h) * 0.02)
        self.camera_type = nn.Parameter(torch.randn(1, 1, config.num_cameras, 1, h) * 0.02)
        self.patch_type = nn.Parameter(torch.randn(1, 1, 1, config.patches_per_camera, h) * 0.02)
        self.out_norm = nn.LayerNorm(h)
        self.drop = nn.Dropout(config.visual_memory_dropout)

    def forward(self, visual: Tensor) -> Tensor:
        cfg = self.config
        if visual.ndim != 5:
            raise ValueError(f"visual must be [B,H,C,P,D], got {tuple(visual.shape)}")
        b, hist, cams, patches, dim = visual.shape
        if hist != cfg.visual_history_length or cams != cfg.num_cameras or patches != cfg.patches_per_camera or dim != cfg.visual_token_dim:
            raise ValueError(
                "V37 dense visual geometry mismatch: "
                f"got {(hist, cams, patches, dim)}, expected "
                f"{(cfg.visual_history_length, cfg.num_cameras, cfg.patches_per_camera, cfg.visual_token_dim)}"
            )
        x = self.proj(visual)
        x = x + self.history_type.to(device=x.device, dtype=x.dtype)
        x = x + self.camera_type.to(device=x.device, dtype=x.dtype)
        x = x + self.patch_type.to(device=x.device, dtype=x.dtype)
        x = self.out_norm(x)
        return self.drop(x.reshape(b, hist * cams * patches, cfg.hidden_size))


class ContextTokenBuilder(nn.Module):
    """Build non-visual proprioceptive/action-history/proposal memory tokens."""

    def __init__(self, config: V37PolicyConfig) -> None:
        super().__init__()
        h = config.hidden_size
        self.config = config
        self.state_proj = nn.Linear(config.state_dim, h)
        self.executed_proj = nn.Linear(config.action_dim, h)
        self.proposal_proj = nn.Identity()
        self.task_token = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        self.state_type = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        self.executed_type = nn.Parameter(torch.randn(1, config.executed_history_length, h) * 0.02)
        self.proposal_type = nn.Parameter(torch.randn(1, config.action_horizon, h) * 0.02)

    def forward(self, state: Tensor, executed_history: Tensor, proposal_tokens: Tensor, proposal_keep: Tensor) -> Tensor:
        batch = state.shape[0]
        task = self.task_token.expand(batch, -1, -1)
        state_token = self.state_proj(state)[:, None] + self.state_type
        executed = self.executed_proj(executed_history) + self.executed_type
        proposal = self.proposal_proj(proposal_tokens) * proposal_keep[:, None, None] + self.proposal_type
        return torch.cat([task, state_token, executed, proposal], dim=1)


class LatentWorldActionBlock(nn.Module):
    """Shared latent block with high-bandwidth visual cross-attention.

    High-level slots and action/event tokens first exchange information through
    self-attention, then cross-attend to the complete dense visual/proprio memory.
    Action tokens therefore never have to pass through a compressed world-token
    bottleneck.
    """

    def __init__(self, config: V37PolicyConfig) -> None:
        super().__init__()
        h = config.hidden_size
        self.n1 = nn.LayerNorm(h, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(h, config.num_heads, batch_first=True, dropout=config.dropout)
        self.n2 = nn.LayerNorm(h, elementwise_affine=False)
        self.mem_norm = nn.LayerNorm(h)
        self.cross = nn.MultiheadAttention(h, config.num_heads, batch_first=True, dropout=config.dropout)
        self.n3 = nn.LayerNorm(h, elementwise_affine=False)
        self.ffn = BiasFreeFFN(h, config.ffn_expansion)
        self.drop = nn.Dropout(config.dropout)
        self.mod = nn.Linear(h, 9 * h)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    @staticmethod
    def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        return x * (1 + scale[:, None]) + shift[:, None]

    def forward(self, x: Tensor, memory: Tensor, time: Tensor) -> Tensor:
        sa_s, sa_c, sa_g, ca_s, ca_c, ca_g, ff_s, ff_c, ff_g = self.mod(time).chunk(9, dim=-1)
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


class FullLatentPlannerV37(nn.Module):
    def __init__(self, config: V37PolicyConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        h = config.hidden_size
        self.visual_memory = DenseVisualMemory(config)
        self.context_memory = ContextTokenBuilder(config)
        self.noisy_physical_lift = PhysicalActionTokenLift(config)
        self.role = HorizonRoleEmbedding(config)
        self.time = TimeEmbedding(h)
        self.horizon_query = nn.Parameter(torch.randn(1, config.action_horizon, h) * 0.02)
        self.high_query = nn.Parameter(torch.randn(1, config.high_level_slots, h) * 0.02)
        self.event_query = nn.Parameter(torch.randn(1, config.event_tokens, h) * 0.02)
        self.high_type = nn.Parameter(torch.randn(1, config.high_level_slots, h) * 0.02)
        self.event_type = nn.Parameter(torch.randn(1, config.event_tokens, h) * 0.02)
        self.register_buffer("horizon_position", sinusoidal_positions(range(1, config.action_horizon + 1), h)[None], persistent=True)
        self.blocks = nn.ModuleList([LatentWorldActionBlock(config) for _ in range(config.depth)])
        self.physical_head = TransitionAwarePhysicalVelocityHead(config)
        self.event_probe = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 3))
        self.motion_probe = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 1))
        self.future_head = nn.Sequential(
            nn.LayerNorm(h),
            nn.Linear(h, h * 2),
            nn.SiLU(),
            nn.Linear(h * 2, config.future_aux_offsets * h),
        )
        self.future_target_proj = nn.Sequential(nn.LayerNorm(config.visual_token_dim), nn.Linear(config.visual_token_dim, h))

    def _memory(self, visual: Tensor, state: Tensor, executed_history: Tensor, proposal_tokens: Tensor, proposal_keep: Tensor) -> Tensor:
        visual_memory = self.visual_memory(visual)
        context = self.context_memory(state, executed_history, proposal_tokens, proposal_keep)
        return torch.cat([context, visual_memory], dim=1)

    def forward(
        self,
        noisy_physical: Tensor,
        time: Tensor,
        visual: Tensor,
        state: Tensor,
        executed_history: Tensor,
        proposal_tokens: Tensor,
        proposal_keep: Tensor | None = None,
    ) -> dict[str, Tensor]:
        cfg = self.config
        if proposal_keep is None:
            proposal_keep = torch.ones(noisy_physical.shape[0], device=noisy_physical.device, dtype=noisy_physical.dtype)
        batch = noisy_physical.shape[0]
        dtype = noisy_physical.dtype
        device = noisy_physical.device
        role = self.role(batch, device=device, dtype=dtype)
        hpos = self.horizon_position.to(device=device, dtype=dtype)
        high = self.high_query.expand(batch, -1, -1) + self.high_type
        horizon = self.horizon_query.expand(batch, -1, -1) + hpos + role + self.noisy_physical_lift(noisy_physical)
        event = self.event_query.expand(batch, -1, -1) + self.event_type
        tokens = torch.cat([high, horizon, event], dim=1)
        high_slice = slice(0, cfg.high_level_slots)
        action_slice = slice(cfg.high_level_slots, cfg.high_level_slots + cfg.action_horizon)
        event_slice = slice(cfg.high_level_slots + cfg.action_horizon, cfg.high_level_slots + cfg.action_horizon + cfg.event_tokens)
        memory = self._memory(visual, state, executed_history, proposal_tokens, proposal_keep)
        time_emb = self.time(time.to(dtype=tokens.dtype))
        for block in self.blocks:
            tokens = block(tokens, memory, time_emb)
        high_tokens = tokens[:, high_slice]
        action_tokens = tokens[:, action_slice]
        event_tokens = tokens[:, event_slice]
        pred_physical_velocity = self.physical_head(action_tokens, high_tokens.mean(dim=1, keepdim=True).expand(-1, cfg.action_horizon, -1))
        future_pred = self.future_head(high_tokens.mean(dim=1)).reshape(batch, cfg.future_aux_offsets, cfg.hidden_size)
        return {
            "planner_tokens": tokens,
            "planner_action_tokens": action_tokens,
            "high_level_tokens": high_tokens,
            "planner_event_tokens": event_tokens,
            "transition_latent": high_tokens.mean(dim=1, keepdim=True).expand(-1, cfg.action_horizon, -1),
            "event_logits": self.event_probe(action_tokens.detach()),
            "motion_logits": self.motion_probe(action_tokens.detach()).squeeze(-1),
            "pred_physical_velocity": pred_physical_velocity,
            "future_latent_pred": future_pred,
        }

    def target_future_latent(self, target_visual: Tensor) -> Tensor:
        """Pool future dense target tokens for leakage-free auxiliary targets.

        Target tokens are never fed to ``forward``.  They are only stop-grad
        supervision for high-level slots.
        """
        cfg = self.config
        if target_visual.ndim != 6:
            raise ValueError(f"target_visual must be [B,F,H,C,P,D], got {tuple(target_visual.shape)}")
        b, fut = target_visual.shape[0], target_visual.shape[1]
        pooled = target_visual.float().mean(dim=(2, 3, 4))
        pooled = self.future_target_proj(pooled.to(device=next(self.parameters()).device, dtype=next(self.parameters()).dtype))
        if fut >= cfg.future_aux_offsets:
            return pooled[:, : cfg.future_aux_offsets]
        pad = pooled[:, -1:].expand(b, cfg.future_aux_offsets - fut, -1)
        return torch.cat([pooled, pad], dim=1)


class V37PolicySystem(nn.Module):
    def __init__(self, policy_config: V37PolicyConfig) -> None:
        super().__init__()
        self.policy_config = policy_config
        self.codec = PhysicalActionCodec(policy_config)
        self.proposal = RejectableHistoryProposal(policy_config)
        self.planner = FullLatentPlannerV37(policy_config)

    def _policy_forward(
        self,
        noisy_physical: Tensor,
        time: Tensor,
        visual: Tensor,
        state: Tensor,
        executed_history: Tensor,
        proposal_tokens: Tensor,
        proposal_keep: Tensor,
    ) -> dict[str, Tensor]:
        return self.planner(noisy_physical, time, visual, state, executed_history, proposal_tokens, proposal_keep)

    def flow_training_forward(
        self,
        visual: Tensor,
        state_history: Tensor,
        executed_history: Tensor,
        state: Tensor,
        target_action: Tensor,
        *,
        target_visual: Tensor | None = None,
        proposal_dropout: float | None = None,
    ) -> dict[str, Tensor]:
        del state_history  # V37 keeps visual/state history in dense visual memory and executed_history tokens.
        proposal = self.proposal(executed_history)
        target_physical = self.codec.encode(target_action, state)
        noise = torch.randn_like(target_physical)
        t = torch.rand(target_physical.shape[0], device=target_physical.device, dtype=target_physical.dtype)
        noisy_physical = (1 - t[:, None, None]) * target_physical + t[:, None, None] * noise
        target_physical_velocity = noise - target_physical
        drop = self.policy_config.proposal_dropout if proposal_dropout is None else float(proposal_dropout)
        keep = (torch.rand(target_physical.shape[0], device=target_physical.device) >= drop).to(target_physical.dtype)
        policy = self._policy_forward(noisy_physical, t, visual, state, executed_history, proposal["tokens"].detach(), keep)
        clean_physical_estimate = noisy_physical - t[:, None, None] * policy["pred_physical_velocity"]
        decoded_action = self.codec.decode(clean_physical_estimate, state)
        out = {
            "pred_physical_velocity": policy["pred_physical_velocity"],
            "target_physical_velocity": target_physical_velocity,
            "target_physical": target_physical,
            "clean_physical_estimate": clean_physical_estimate,
            "proposal_action": proposal["action"],
            "time": t,
            "noisy_physical_action": noisy_physical,
            "pred_action_estimate": decoded_action,
            "event_logits": policy["event_logits"],
            "motion_logits": policy["motion_logits"],
            "transition_latent": policy["transition_latent"],
            "high_level_tokens": policy["high_level_tokens"],
            "future_latent_pred": policy["future_latent_pred"],
        }
        if target_visual is not None:
            out["future_latent_target"] = self.planner.target_future_latent(target_visual).detach().to(dtype=policy["future_latent_pred"].dtype)
        return out

    @torch.no_grad()
    def sample(
        self,
        visual: Tensor,
        state_history: Tensor,
        executed_history: Tensor,
        state: Tensor,
        *,
        steps: int | None = None,
        noise: Tensor | None = None,
        use_proposal: bool = True,
        return_event_logits: bool = False,
    ) -> Tensor | dict[str, Tensor]:
        del state_history
        proposal = self.proposal(executed_history)
        steps = int(steps or self.policy_config.inference_steps)
        if noise is None:
            x = torch.randn(
                visual.shape[0],
                self.policy_config.action_horizon,
                self.policy_config.physical_action_dim,
                device=visual.device,
                dtype=visual.dtype,
            )
        else:
            x = noise.clone()
            if x.shape[-1] == self.policy_config.action_dim:
                x = self.codec.encode(x.to(device=visual.device, dtype=visual.dtype), state.to(device=visual.device, dtype=visual.dtype))
            elif x.shape[-1] != self.policy_config.physical_action_dim:
                raise ValueError("noise must have last dim action_dim or physical_action_dim")
        keep = torch.full((visual.shape[0],), 1.0 if use_proposal else 0.0, device=visual.device, dtype=visual.dtype)
        for index in range(steps, 0, -1):
            t = torch.full((visual.shape[0],), float(index) / float(steps), device=visual.device, dtype=visual.dtype)
            out = self._policy_forward(x, t, visual, state, executed_history, proposal["tokens"], keep)
            x = x - out["pred_physical_velocity"] / float(steps)
        action = self.codec.decode(x, state)
        if return_event_logits:
            zero_t = torch.zeros((visual.shape[0],), device=visual.device, dtype=visual.dtype)
            event = self._policy_forward(x, zero_t, visual, state, executed_history, proposal["tokens"], keep)
            return {"action": action, "physical_action": x, "event_logits": event["event_logits"], "motion_logits": event["motion_logits"]}
        return action

    def parameter_report(self) -> dict[str, int]:
        report = {
            "history_proposal": sum(p.numel() for p in self.proposal.parameters()),
            "physical_action_codec": sum(p.numel() for p in self.codec.parameters()),
            "full_latent_world_shaped_planner": sum(p.numel() for p in self.planner.parameters()),
        }
        report["total"] = sum(p.numel() for p in self.parameters())
        report["trainable"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return report


__all__ = [
    "V37PolicyConfig",
    "DenseVisualMemory",
    "LatentWorldActionBlock",
    "FullLatentPlannerV37",
    "V37PolicySystem",
]
