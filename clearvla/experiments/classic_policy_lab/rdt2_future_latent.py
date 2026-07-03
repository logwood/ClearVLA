from __future__ import annotations

"""Action-semantic future-DINO residual dynamics for ClearVLA.

The world model remains parameter-isolated from the policy, but its action path
is no longer allowed to act as a second current-observation shortcut:

* the action encoder consumes only complete actions, temporal deltas, and
  state-relative deltas;
* action prefix representations at the configured future offsets are injected
  directly into the corresponding future-token blocks;
* a future-change encoder maps ground-truth DINO residuals into the same
  semantic space as the action prefixes;
* an inverse-action head anchors future-change embeddings to explicit action
  summaries;
* predicted future residuals are evaluated by detached semantic/inverse heads,
  closing the action -> future -> action cycle without letting the evaluator
  move together with the prediction.

The public compression, flow, corruption, and detached-forward contracts are
kept compatible with the previous clean future-dynamics release.
"""

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:  # PyTorch >= 2.0
    from torch.func import functional_call as _functional_call
except ImportError:  # pragma: no cover
    from torch.nn.utils.stateless import functional_call as _functional_call

from .rdt2_fm_reference import Attention, CrossAttention, FeedForward, RMSNorm, TimestepEmbedder


@dataclass
class FutureLatentFlowBatch:
    target_normalized: Tensor
    noisy: Tensor
    target_velocity: Tensor
    time: Tensor
    noise: Tensor


def sample_future_latent_flow(
    target_normalized: Tensor,
    *,
    generator: torch.Generator | None = None,
    time: Tensor | None = None,
    noise: Tensor | None = None,
) -> FutureLatentFlowBatch:
    """Sample a straight conditional-flow bridge from N(0,I) to the target."""

    if target_normalized.ndim != 5:
        raise ValueError(
            "target_normalized must be [B,T,C,S,D], "
            f"got {tuple(target_normalized.shape)}"
        )
    batch = target_normalized.shape[0]
    if time is None:
        time = torch.rand(
            (batch,),
            device=target_normalized.device,
            dtype=target_normalized.dtype,
            generator=generator,
        )
    else:
        time = time.to(device=target_normalized.device, dtype=target_normalized.dtype)
    if tuple(time.shape) != (batch,):
        raise ValueError(f"future flow time must be [B], got {tuple(time.shape)}")
    if noise is None:
        noise = torch.randn(
            target_normalized.shape,
            device=target_normalized.device,
            dtype=target_normalized.dtype,
            generator=generator,
        )
    else:
        noise = noise.to(device=target_normalized.device, dtype=target_normalized.dtype)
    if noise.shape != target_normalized.shape:
        raise ValueError("future flow noise must match target_normalized")
    alpha = time.reshape(batch, 1, 1, 1, 1)
    noisy = (1.0 - alpha) * noise + alpha * target_normalized
    return FutureLatentFlowBatch(
        target_normalized=target_normalized,
        noisy=noisy,
        target_velocity=target_normalized - noise,
        time=time,
        noise=noise,
    )


class _PureActionBlock(nn.Module):
    """Standard causal action-only Transformer block.

    There are deliberately no BF16 scalar gates and no current-visual
    cross-attention.  The block cannot relay observation features to the
    future stream under the name of an action representation.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        hidden = int(config["hidden_size"])
        eps = float(config["norm_eps"])
        self.self_norm = RMSNorm(hidden, eps=eps)
        self.self_attn = Attention(config)
        self.ffn_norm = RMSNorm(hidden, eps=eps)
        self.ffn = FeedForward(
            hidden,
            4 * hidden,
            int(config["multiple_of"]),
            config.get("ffn_dim_multiplier"),
        )

    def forward(self, action: Tensor) -> Tensor:
        action = action + self.self_attn(self.self_norm(action), is_causal=True)
        return action + self.ffn(self.ffn_norm(action))


class _FutureModulationBank(nn.Module):
    """Task-specific modulation for self/current/FFN branches.

    Action conditioning is intentionally excluded from this gate bank.  It has
    a direct prefix injection and an ungated residual cross-attention path.
    """

    def __init__(self, hidden_size: int, layers: int, rank: int) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.trunk = nn.Sequential(
            nn.SiLU(),
            nn.Linear(2 * hidden_size, rank),
            nn.SiLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(rank, 9 * hidden_size) for _ in range(layers)])
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            with torch.no_grad():
                for gate_index in (2, 5, 8):
                    start = gate_index * hidden_size
                    head.bias[start : start + hidden_size].fill_(-2.0)

    def prepare(self, condition: Tensor) -> Tensor:
        return self.trunk(condition)

    def for_layer(self, prepared: Tensor, layer_idx: int) -> Tensor:
        return self.heads[layer_idx](prepared)


class _FutureResidualBlock(nn.Module):
    """Future stream with gated self/current branches and ungated action use."""

    def __init__(self, config: dict[str, Any], *, action_cross_scale: float) -> None:
        super().__init__()
        hidden = int(config["hidden_size"])
        eps = float(config["norm_eps"])
        self.hidden_size = hidden
        self.action_cross_scale = float(action_cross_scale)
        self.self_norm = RMSNorm(hidden, eps=eps)
        self.self_attn = Attention(config)
        self.current_norm = RMSNorm(hidden, eps=eps)
        self.current_cond_norm = RMSNorm(hidden, eps=eps)
        self.current_cross = CrossAttention(config)
        self.action_norm = RMSNorm(hidden, eps=eps)
        self.action_cond_norm = RMSNorm(hidden, eps=eps)
        self.action_cross = CrossAttention(config)
        self.ffn_norm = RMSNorm(hidden, eps=eps)
        self.ffn = FeedForward(
            hidden,
            4 * hidden,
            int(config["multiple_of"]),
            config.get("ffn_dim_multiplier"),
        )

    @staticmethod
    def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(
        self,
        future: Tensor,
        modulation: Tensor,
        *,
        current: Tensor,
        action: Tensor,
        action_mask: Tensor,
        future_mask: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if modulation.ndim != 2 or modulation.shape[1] != 9 * self.hidden_size:
            raise ValueError(
                f"future modulation must be [B,{9 * self.hidden_size}], "
                f"got {tuple(modulation.shape)}"
            )
        values = modulation.chunk(9, dim=1)
        shift_self, scale_self, gate_self = values[0:3]
        shift_current, scale_current, gate_current = values[3:6]
        shift_ffn, scale_ffn, gate_ffn = values[6:9]
        gate_self = torch.sigmoid(gate_self)
        gate_current = torch.sigmoid(gate_current)
        gate_ffn = torch.sigmoid(gate_ffn)
        future = future + gate_self.unsqueeze(1) * self.self_attn(
            self._modulate(self.self_norm(future), shift_self, scale_self),
            mask=future_mask,
        )
        future = future + gate_current.unsqueeze(1) * self.current_cross(
            self._modulate(self.current_norm(future), shift_current, scale_current),
            c=self.current_cond_norm(current),
            mask=None,
        )
        # Fixed residual scale: action use cannot be silently gated to zero.
        future = future + self.action_cross_scale * self.action_cross(
            self.action_norm(future),
            c=self.action_cond_norm(action),
            mask=action_mask,
        )
        future = future + gate_ffn.unsqueeze(1) * self.ffn(
            self._modulate(self.ffn_norm(future), shift_ffn, scale_ffn)
        )
        return future, {
            "future_gate_self": gate_self.mean(),
            "future_gate_current": gate_current.mean(),
            "future_gate_ffn": gate_ffn.mean(),
            "future_action_cross_scale": future.new_tensor(self.action_cross_scale),
            # Compatibility metric: action is now structurally on, not sigmoid-gated.
            "future_gate_action": future.new_tensor(1.0),
        }


class _SemanticBlock(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        hidden = int(config["hidden_size"])
        eps = float(config["norm_eps"])
        self.self_norm = RMSNorm(hidden, eps=eps)
        self.self_attn = Attention(config)
        self.ffn_norm = RMSNorm(hidden, eps=eps)
        self.ffn = FeedForward(
            hidden,
            4 * hidden,
            int(config["multiple_of"]),
            config.get("ffn_dim_multiplier"),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.self_attn(self.self_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class FutureChangeEncoder(nn.Module):
    """Encode each future DINO residual into an action-semantic embedding."""

    def __init__(
        self,
        *,
        latent_dim: int,
        semantic_hidden_size: int,
        semantic_dim: int,
        depth: int,
        heads: int,
        kv_heads: int,
        num_future_frames: int,
        num_cameras: int,
        spatial_tokens: int,
        norm_eps: float,
        multiple_of: int,
        ffn_dim_multiplier: float | None,
        use_flash_attn: bool,
    ) -> None:
        super().__init__()
        if semantic_hidden_size % heads != 0 or heads % kv_heads != 0:
            raise ValueError("invalid future semantic attention dimensions")
        self.latent_dim = int(latent_dim)
        self.semantic_hidden_size = int(semantic_hidden_size)
        self.semantic_dim = int(semantic_dim)
        self.num_future_frames = int(num_future_frames)
        self.num_cameras = int(num_cameras)
        self.spatial_tokens = int(spatial_tokens)
        core = {
            "hidden_size": semantic_hidden_size,
            "num_heads": heads,
            "num_kv_heads": kv_heads,
            "norm_eps": norm_eps,
            "multiple_of": multiple_of,
            "ffn_dim_multiplier": ffn_dim_multiplier,
            "use_flash_attn": use_flash_attn,
        }
        self.input_proj = nn.Sequential(
            nn.Linear(latent_dim, semantic_hidden_size),
            nn.SiLU(),
            nn.Linear(semantic_hidden_size, semantic_hidden_size),
        )
        self.blocks = nn.ModuleList([_SemanticBlock(core) for _ in range(depth)])
        self.camera_embedding = nn.Parameter(
            torch.randn(1, 1, num_cameras, 1, semantic_hidden_size) * 0.02
        )
        self.spatial_embedding = nn.Parameter(
            torch.randn(1, 1, 1, spatial_tokens, semantic_hidden_size) * 0.02
        )
        self.time_embedding = nn.Parameter(
            torch.randn(1, num_future_frames, 1, 1, semantic_hidden_size) * 0.02
        )
        self.query_embedding = nn.Parameter(
            torch.randn(1, num_future_frames, 1, semantic_hidden_size) * 0.02
        )
        self.query_norm = nn.LayerNorm(semantic_hidden_size)
        self.patch_norm = nn.LayerNorm(semantic_hidden_size)
        self.query_attn = nn.MultiheadAttention(
            semantic_hidden_size, heads, batch_first=True
        )
        self.output_norm = nn.LayerNorm(semantic_hidden_size)
        self.output_proj = nn.Sequential(
            nn.Linear(semantic_hidden_size, semantic_hidden_size),
            nn.SiLU(),
            nn.Linear(semantic_hidden_size, semantic_dim),
        )

    def forward(self, residual_normalized: Tensor) -> Tensor:
        expected = (
            residual_normalized.shape[0],
            self.num_future_frames,
            self.num_cameras,
            self.spatial_tokens,
            self.latent_dim,
        )
        if tuple(residual_normalized.shape) != expected:
            raise ValueError(
                "future-change input shape mismatch: "
                f"expected {expected}, got {tuple(residual_normalized.shape)}"
            )
        batch = residual_normalized.shape[0]
        semantic_dtype = self.input_proj[0].weight.dtype
        residual_normalized = residual_normalized.to(dtype=semantic_dtype)
        x = self.input_proj(residual_normalized)
        x = x + self.camera_embedding + self.spatial_embedding + self.time_embedding
        x = x.reshape(
            batch * self.num_future_frames,
            self.num_cameras * self.spatial_tokens,
            self.semantic_hidden_size,
        )
        for block in self.blocks:
            x = block(x)
        query = self.query_embedding.expand(batch, -1, -1, -1).reshape(
            batch * self.num_future_frames, 1, self.semantic_hidden_size
        )
        pooled, _ = self.query_attn(
            self.query_norm(query), self.patch_norm(x), self.patch_norm(x), need_weights=False
        )
        embedding = self.output_proj(self.output_norm(pooled[:, 0]))
        embedding = embedding.reshape(batch, self.num_future_frames, self.semantic_dim)
        return F.normalize(embedding.float(), dim=-1).to(dtype=embedding.dtype)


class InverseActionHead(nn.Module):
    def __init__(self, semantic_dim: int, hidden_size: int, summary_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(semantic_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, summary_dim),
        )

    def forward(self, embedding: Tensor) -> Tensor:
        return self.net(embedding.to(dtype=self.net[0].weight.dtype))


class CurrentActionBaselineHead(nn.Module):
    """Diagnostic current-only action-summary predictor.

    Inputs are detached by the caller, so this head measures shortcut strength
    without changing the world-model current encoder.
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        state_dim: int,
        hidden_size: int,
        num_future_frames: int,
        summary_dim: int,
    ) -> None:
        super().__init__()
        self.num_future_frames = int(num_future_frames)
        self.summary_dim = int(summary_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + state_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, num_future_frames * summary_dim),
        )

    def forward(self, current_compressed: Tensor, state: Tensor) -> Tensor:
        semantic_dtype = self.net[0].weight.dtype
        pooled = current_compressed.to(dtype=semantic_dtype).mean(dim=(1, 2))
        pred = self.net(torch.cat([pooled, state.to(dtype=semantic_dtype)], dim=-1))
        return pred.reshape(pred.shape[0], self.num_future_frames, self.summary_dim)


class CleanFutureLatentDynamics(nn.Module):
    """Self-contained action-conditioned DINO residual world model."""

    def __init__(
        self,
        *,
        latent_dim: int,
        action_dim: int,
        state_dim: int,
        action_horizon: int,
        hidden_size: int,
        depth: int,
        modulation_rank: int,
        num_heads: int,
        num_kv_heads: int,
        norm_eps: float,
        multiple_of: int,
        ffn_dim_multiplier: float | None,
        use_flash_attn: bool,
        future_offsets: tuple[int, ...],
        num_future_frames: int,
        num_cameras: int,
        grid_size: int,
        dtype: torch.dtype,
        stat_eps: float = 1e-5,
        motion_weight: float = 1.0,
        motion_weight_cap: float = 4.0,
        semantic_dim: int = 256,
        semantic_hidden_size: int = 256,
        semantic_depth: int = 2,
        semantic_heads: int = 4,
        semantic_kv_heads: int = 2,
        gripper_dim_index: int = -1,
        inverse_transition_threshold: float = 0.10,
        action_cross_scale: float = 0.0,
        semantic_negative_delay: int = 3,
    ) -> None:
        super().__init__()
        positive = (
            latent_dim,
            action_dim,
            state_dim,
            action_horizon,
            hidden_size,
            depth,
            modulation_rank,
            num_heads,
            num_kv_heads,
            num_future_frames,
            num_cameras,
            grid_size,
            semantic_dim,
            semantic_hidden_size,
            semantic_depth,
            semantic_heads,
            semantic_kv_heads,
        )
        if min(positive) <= 0:
            raise ValueError("clean future-dynamics dimensions must be positive")
        if action_dim != state_dim:
            raise ValueError(
                "pure state-relative action encoding currently requires action_dim == state_dim"
            )
        if hidden_size % num_heads != 0 or num_heads % num_kv_heads != 0:
            raise ValueError("invalid clean future-dynamics attention dimensions")
        if stat_eps <= 0 or motion_weight < 0 or motion_weight_cap < 1:
            raise ValueError("invalid future residual normalization/weight settings")
        if inverse_transition_threshold < 0 or semantic_negative_delay < 1:
            raise ValueError("invalid action-semantic transition/negative settings")
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.action_horizon = int(action_horizon)
        self.hidden_size = int(hidden_size)
        self.depth = int(depth)
        self.num_future_frames = int(num_future_frames)
        self.future_offsets = tuple(int(value) for value in future_offsets)
        if len(self.future_offsets) != self.num_future_frames:
            raise ValueError("future_offsets length must match num_future_frames")
        if tuple(sorted(set(self.future_offsets))) != self.future_offsets:
            raise ValueError("future_offsets must be strictly increasing")
        if self.future_offsets[0] <= 0 or self.future_offsets[-1] > action_horizon:
            raise ValueError("future_offsets must lie inside the action horizon")
        self.num_cameras = int(num_cameras)
        self.grid_size = int(grid_size)
        self.spatial_tokens = self.grid_size * self.grid_size
        self.total_future_tokens = self.num_future_frames * self.num_cameras * self.spatial_tokens
        self.total_current_tokens = self.num_cameras * self.spatial_tokens
        self.semantic_dim = int(semantic_dim)
        self.semantic_hidden_size = int(semantic_hidden_size)
        self.inverse_transition_threshold = float(inverse_transition_threshold)
        self.semantic_negative_delay = int(semantic_negative_delay)
        self.gripper_dim_index = int(gripper_dim_index)
        if self.gripper_dim_index < 0:
            self.gripper_dim_index += self.action_dim
        if not (0 <= self.gripper_dim_index < self.action_dim):
            raise ValueError("invalid gripper_dim_index for action semantics")
        self.action_summary_dim = 3 * self.action_dim + 4
        self.action_cross_scale = (
            float(action_cross_scale)
            if action_cross_scale > 0
            else 1.0 / math.sqrt(float(depth))
        )

        action_prefix_mask = torch.zeros(
            self.total_future_tokens, self.action_horizon, dtype=torch.bool
        )
        row = 0
        for offset in self.future_offsets:
            rows = self.num_cameras * self.spatial_tokens
            action_prefix_mask[row : row + rows, : min(offset, self.action_horizon)] = True
            row += rows
        self.register_buffer("future_action_prefix_mask", action_prefix_mask, persistent=True)
        future_temporal_mask = torch.zeros(
            self.total_future_tokens, self.total_future_tokens, dtype=torch.bool
        )
        block = self.num_cameras * self.spatial_tokens
        for query_time in range(self.num_future_frames):
            q0, q1 = query_time * block, (query_time + 1) * block
            future_temporal_mask[q0:q1, : (query_time + 1) * block] = True
        self.register_buffer("future_temporal_mask", future_temporal_mask, persistent=True)
        self.stat_eps = float(stat_eps)
        self.motion_weight = float(motion_weight)
        self.motion_weight_cap = float(motion_weight_cap)

        core = {
            "hidden_size": hidden_size,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "norm_eps": norm_eps,
            "multiple_of": multiple_of,
            "ffn_dim_multiplier": ffn_dim_multiplier,
            "use_flash_attn": use_flash_attn,
        }
        self.current_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size)
        )
        # [absolute action, temporal delta, current-state-relative delta]
        self.action_proj = nn.Sequential(
            nn.Linear(3 * action_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.future_input_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size)
        )
        self.action_blocks = nn.ModuleList([_PureActionBlock(core) for _ in range(depth)])
        self.future_blocks = nn.ModuleList([
            _FutureResidualBlock(core, action_cross_scale=self.action_cross_scale)
            for _ in range(depth)
        ])
        self.action_semantic_proj = nn.Sequential(
            nn.Linear(hidden_size, semantic_hidden_size),
            nn.SiLU(),
            nn.Linear(semantic_hidden_size, semantic_dim),
        )
        self.action_to_future = nn.Linear(hidden_size, hidden_size)
        self.future_change_encoder = FutureChangeEncoder(
            latent_dim=latent_dim,
            semantic_hidden_size=semantic_hidden_size,
            semantic_dim=semantic_dim,
            depth=semantic_depth,
            heads=semantic_heads,
            kv_heads=semantic_kv_heads,
            num_future_frames=num_future_frames,
            num_cameras=num_cameras,
            spatial_tokens=self.spatial_tokens,
            norm_eps=norm_eps,
            multiple_of=max(8, min(multiple_of, semantic_hidden_size)),
            ffn_dim_multiplier=ffn_dim_multiplier,
            use_flash_attn=use_flash_attn,
        )
        self.inverse_action_head = InverseActionHead(
            semantic_dim, semantic_hidden_size, self.action_summary_dim
        )
        self.current_action_baseline_head = CurrentActionBaselineHead(
            latent_dim=latent_dim,
            state_dim=state_dim,
            hidden_size=semantic_hidden_size,
            num_future_frames=num_future_frames,
            summary_dim=self.action_summary_dim,
        )
        self.time_embedder = TimestepEmbedder(hidden_size, dtype=dtype)
        self.task_embedding = nn.Parameter(torch.randn(1, hidden_size) * 0.02)
        self.modulation = _FutureModulationBank(hidden_size, depth, modulation_rank)
        self.output_norm = RMSNorm(hidden_size, eps=norm_eps)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.SiLU(),
            nn.Linear(2 * hidden_size, latent_dim),
        )

        self.current_camera_embedding = nn.Parameter(
            torch.randn(1, num_cameras, 1, hidden_size) * 0.02
        )
        self.current_spatial_embedding = nn.Parameter(
            torch.randn(1, 1, self.spatial_tokens, hidden_size) * 0.02
        )
        self.future_temporal_embedding = nn.Parameter(
            torch.randn(1, num_future_frames, 1, 1, hidden_size) * 0.02
        )
        self.future_camera_embedding = nn.Parameter(
            torch.randn(1, 1, num_cameras, 1, hidden_size) * 0.02
        )
        self.future_spatial_embedding = nn.Parameter(
            torch.randn(1, 1, 1, self.spatial_tokens, hidden_size) * 0.02
        )
        self.action_temporal_embedding = nn.Parameter(
            torch.randn(1, action_horizon, hidden_size) * 0.02
        )
        self.current_type_embedding = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.action_type_embedding = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.future_type_embedding = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)

        self.register_buffer(
            "residual_mean",
            torch.zeros(num_future_frames, num_cameras, latent_dim, dtype=torch.float32),
        )
        self.register_buffer(
            "residual_std",
            torch.ones(num_future_frames, num_cameras, latent_dim, dtype=torch.float32),
        )
        self.register_buffer("residual_stats_ready", torch.tensor(False, dtype=torch.bool))
        self._initialize()
        self.to(dtype=dtype)

    def _apply(self, fn):
        super()._apply(fn)
        self.residual_mean.data = self.residual_mean.data.float()
        self.residual_std.data = self.residual_std.data.float()
        # Low-dimensional semantic objectives and logits need FP32 master
        # parameters; otherwise 1e-4 AdamW steps can quantize away in BF16.
        self.action_semantic_proj.float()
        self.future_change_encoder.float()
        self.inverse_action_head.float()
        self.current_action_baseline_head.float()
        return self

    def _initialize(self) -> None:
        modules = (
            self.current_proj,
            self.action_proj,
            self.future_input_proj,
            self.action_semantic_proj,
            self.action_to_future,
            self.output_proj,
        )
        for module in modules:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        final = self.output_proj[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        nn.init.normal_(self.time_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embedder.mlp[2].weight, std=0.02)

    def set_residual_stats(self, mean: Tensor, std: Tensor) -> None:
        expected = (self.num_future_frames, self.num_cameras, self.latent_dim)
        mean = torch.as_tensor(mean, dtype=torch.float32, device=self.residual_mean.device)
        std = torch.as_tensor(std, dtype=torch.float32, device=self.residual_std.device)
        if tuple(mean.shape) != expected or tuple(std.shape) != expected:
            raise ValueError(
                f"future residual stats must be {expected}, got {tuple(mean.shape)} and {tuple(std.shape)}"
            )
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("future residual statistics must be finite")
        self.residual_mean.copy_(mean)
        self.residual_std.copy_(std.clamp_min(self.stat_eps))
        self.residual_stats_ready.fill_(True)

    def _pool_grid(self, tokens: Tensor) -> Tensor:
        if tokens.ndim < 3 or tokens.shape[-1] != self.latent_dim:
            raise ValueError(f"unexpected DINO token shape={tuple(tokens.shape)}")
        patches = int(tokens.shape[-2])
        side = math.isqrt(patches)
        if side * side != patches:
            raise ValueError(f"DINO patch token count must form a square grid, got {patches}")
        prefix = tokens.shape[:-2]
        flat = tokens.reshape(-1, patches, self.latent_dim).permute(0, 2, 1)
        flat = flat.reshape(-1, self.latent_dim, side, side)
        pooled = F.adaptive_avg_pool2d(flat.float(), (self.grid_size, self.grid_size))
        pooled = pooled.reshape(*prefix, self.latent_dim, self.spatial_tokens).transpose(-1, -2)
        return pooled.to(dtype=tokens.dtype)

    def compress_current(self, current_tokens: Tensor) -> Tensor:
        if current_tokens.ndim == 3:
            batch, total, dim = current_tokens.shape
            if dim != self.latent_dim or total % self.num_cameras:
                raise ValueError("current dense token shape is incompatible with camera count")
            current_tokens = current_tokens.reshape(
                batch, self.num_cameras, total // self.num_cameras, dim
            )
        if current_tokens.ndim != 4 or current_tokens.shape[1] != self.num_cameras:
            raise ValueError(f"current tokens must be [B,C,P,D], got {tuple(current_tokens.shape)}")
        return self._pool_grid(current_tokens)

    def compress_future(self, future_tokens: Tensor) -> Tensor:
        if future_tokens.ndim != 5:
            raise ValueError(f"future tokens must be [B,T,C,P,D], got {tuple(future_tokens.shape)}")
        if (
            future_tokens.shape[1] != self.num_future_frames
            or future_tokens.shape[2] != self.num_cameras
        ):
            raise ValueError("future tokens do not match configured times/cameras")
        return self._pool_grid(future_tokens)

    def residual_target(self, current_compressed: Tensor, future_compressed: Tensor) -> Tensor:
        expected_current = (
            current_compressed.shape[0], self.num_cameras, self.spatial_tokens, self.latent_dim
        )
        if tuple(current_compressed.shape) != expected_current:
            raise ValueError("current compressed latent shape mismatch")
        expected_future = (
            current_compressed.shape[0],
            self.num_future_frames,
            self.num_cameras,
            self.spatial_tokens,
            self.latent_dim,
        )
        if tuple(future_compressed.shape) != expected_future:
            raise ValueError("future compressed latent shape mismatch")
        return future_compressed - current_compressed[:, None]

    def normalize_residual(self, residual: Tensor) -> Tensor:
        if not bool(self.residual_stats_ready.item()):
            raise RuntimeError("future residual statistics are not initialized")
        mean = self.residual_mean.to(device=residual.device, dtype=residual.dtype)[None, :, :, None, :]
        std = self.residual_std.to(device=residual.device, dtype=residual.dtype)[None, :, :, None, :]
        return (residual - mean) / std

    def denormalize_residual(self, normalized: Tensor) -> Tensor:
        mean = self.residual_mean.to(device=normalized.device, dtype=normalized.dtype)[None, :, :, None, :]
        std = self.residual_std.to(device=normalized.device, dtype=normalized.dtype)[None, :, :, None, :]
        return normalized * std + mean

    def motion_weights(self, residual_raw: Tensor) -> Tensor:
        magnitude = residual_raw.float().square().mean(dim=-1, keepdim=True).sqrt()
        baseline = magnitude.mean(dim=3, keepdim=True).clamp_min(1e-6)
        relative = (magnitude / baseline).clamp(max=self.motion_weight_cap)
        weights = 1.0 + self.motion_weight * relative
        weights = weights / weights.mean(dim=3, keepdim=True).clamp_min(1e-6)
        return weights.to(dtype=residual_raw.dtype)

    def _embed_current(self, current_compressed: Tensor) -> Tensor:
        hidden = self.current_proj(current_compressed)
        hidden = hidden + self.current_camera_embedding + self.current_spatial_embedding
        return (
            hidden.reshape(
                current_compressed.shape[0], self.total_current_tokens, self.hidden_size
            )
            + self.current_type_embedding
        )

    def _embed_action_input(
        self,
        action_chunk: Tensor,
        *,
        past_last_action: Tensor,
        state: Tensor,
    ) -> Tensor:
        if tuple(action_chunk.shape[1:]) != (self.action_horizon, self.action_dim):
            raise ValueError("action_chunk must match configured horizon/action_dim")
        if tuple(past_last_action.shape) != (action_chunk.shape[0], self.action_dim):
            raise ValueError("past_last_action must be [B,A]")
        if tuple(state.shape) != (action_chunk.shape[0], self.state_dim):
            raise ValueError("state must be [B,state_dim]")
        boundary = torch.cat([past_last_action[:, None], action_chunk[:, :-1]], dim=1)
        temporal_delta = action_chunk - boundary
        state_relative = action_chunk - state[:, None]
        action = self.action_proj(
            torch.cat([action_chunk, temporal_delta, state_relative], dim=-1)
        )
        return action + self.action_temporal_embedding + self.action_type_embedding

    def encode_action(
        self,
        action_chunk: Tensor,
        *,
        past_last_action: Tensor,
        state: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        action = self._embed_action_input(
            action_chunk, past_last_action=past_last_action, state=state
        )
        for block in self.action_blocks:
            action = block(action)
        indices = torch.as_tensor(
            [offset - 1 for offset in self.future_offsets],
            device=action.device,
            dtype=torch.long,
        )
        prefix_hidden = action.index_select(1, indices)
        semantic_dtype = self.action_semantic_proj[0].weight.dtype
        semantic = self.action_semantic_proj(prefix_hidden.to(dtype=semantic_dtype))
        semantic = F.normalize(semantic.float(), dim=-1).to(dtype=semantic.dtype)
        return action, prefix_hidden, semantic

    def encode_action_semantics(
        self,
        action_chunk: Tensor,
        *,
        past_last_action: Tensor,
        state: Tensor,
    ) -> Tensor:
        return self.encode_action(
            action_chunk, past_last_action=past_last_action, state=state
        )[2]

    def encode_future_change(self, residual_normalized: Tensor) -> Tensor:
        return self.future_change_encoder(residual_normalized)

    @staticmethod
    def _detached_call(module: nn.Module, *args: Tensor) -> Tensor:
        state: dict[str, Tensor] = {
            name: parameter.detach() for name, parameter in module.named_parameters()
        }
        state.update({name: buffer.detach() for name, buffer in module.named_buffers()})
        return _functional_call(module, state, args, strict=False)

    def detached_future_change_embedding(self, residual_normalized: Tensor) -> Tensor:
        return self._detached_call(self.future_change_encoder, residual_normalized)

    def detached_inverse_prediction(self, embedding: Tensor) -> Tensor:
        return self._detached_call(self.inverse_action_head, embedding)

    def inverse_prediction(self, embedding: Tensor) -> Tensor:
        return self.inverse_action_head(embedding)

    def current_only_action_prediction(
        self, current_compressed: Tensor, state: Tensor
    ) -> Tensor:
        return self.current_action_baseline_head(
            current_compressed.detach(), state.detach()
        )

    def build_action_semantic_targets(
        self,
        action_chunk: Tensor,
        *,
        state: Tensor,
        past_last_action: Tensor,
    ) -> dict[str, Tensor]:
        if action_chunk.ndim != 3:
            raise ValueError("action_chunk must be [B,H,A]")
        boundary = torch.cat([past_last_action[:, None], action_chunk[:, :-1]], dim=1)
        delta = action_chunk - boundary
        summaries: list[Tensor] = []
        transition_targets: list[Tensor] = []
        transition_times: list[Tensor] = []
        transition_directions: list[Tensor] = []
        g = self.gripper_dim_index
        for offset in self.future_offsets:
            prefix = action_chunk[:, :offset]
            prefix_delta = delta[:, :offset]
            endpoint_delta = prefix[:, -1] - state
            mean_velocity = prefix_delta.mean(dim=1)
            path_magnitude = prefix_delta.abs().mean(dim=1)
            gripper_delta = prefix_delta[..., g]
            event = gripper_delta.abs() >= self.inverse_transition_threshold
            has_event = event.any(dim=1)
            first_index = event.float().argmax(dim=1)
            denom = float(max(offset - 1, 1))
            event_time = first_index.float() / denom
            event_time = torch.where(has_event, event_time, torch.ones_like(event_time))
            total_direction = (prefix[:, -1, g] - past_last_action[:, g]).sign()
            summary = torch.cat(
                [
                    endpoint_delta,
                    mean_velocity,
                    path_magnitude,
                    prefix[:, -1, g : g + 1],
                    has_event.to(dtype=action_chunk.dtype).unsqueeze(-1),
                    event_time.to(dtype=action_chunk.dtype).unsqueeze(-1),
                    total_direction.unsqueeze(-1),
                ],
                dim=-1,
            )
            summaries.append(summary)
            transition_targets.append(has_event)
            transition_times.append(event_time)
            transition_directions.append(total_direction)
        return {
            "summary": torch.stack(summaries, dim=1),
            "transition": torch.stack(transition_targets, dim=1),
            "transition_time": torch.stack(transition_times, dim=1),
            "transition_direction": torch.stack(transition_directions, dim=1),
        }

    def inverse_loss(
        self,
        prediction: Tensor,
        targets: dict[str, Tensor],
        *,
        prefix: str,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        target = targets["summary"].to(dtype=prediction.dtype)
        if prediction.shape != target.shape:
            raise ValueError("inverse prediction and summary target must share shape")
        a = self.action_dim
        g = self.gripper_dim_index
        arm_indices = [idx for idx in range(a) if idx != g]
        endpoint_pred, endpoint_gt = prediction[..., :a], target[..., :a]
        velocity_pred, velocity_gt = prediction[..., a : 2 * a], target[..., a : 2 * a]
        path_pred, path_gt = prediction[..., 2 * a : 3 * a], target[..., 2 * a : 3 * a]
        arm_pred = torch.cat(
            [endpoint_pred[..., arm_indices], velocity_pred[..., arm_indices], path_pred[..., arm_indices]],
            dim=-1,
        )
        arm_gt = torch.cat(
            [endpoint_gt[..., arm_indices], velocity_gt[..., arm_indices], path_gt[..., arm_indices]],
            dim=-1,
        )
        arm_loss = F.smooth_l1_loss(arm_pred, arm_gt)
        arm_rmse = (arm_pred.float() - arm_gt.float()).square().mean().sqrt()
        base = 3 * a
        gripper_final_pred = prediction[..., base]
        gripper_final_gt = target[..., base]
        gripper_final_loss = F.smooth_l1_loss(gripper_final_pred, gripper_final_gt)
        gripper_final_rmse = (
            gripper_final_pred.float() - gripper_final_gt.float()
        ).square().mean().sqrt()
        transition_logit = prediction[..., base + 1]
        transition_gt = targets["transition"].to(dtype=prediction.dtype)
        transition_loss = F.binary_cross_entropy_with_logits(
            transition_logit, transition_gt
        )
        positive = targets["transition"]
        timing_pred = prediction[..., base + 2].sigmoid()
        timing_gt = targets["transition_time"].to(dtype=prediction.dtype)
        direction_pred = prediction[..., base + 3].tanh()
        direction_gt = targets["transition_direction"].to(dtype=prediction.dtype)
        if positive.any():
            timing_loss = F.smooth_l1_loss(timing_pred[positive], timing_gt[positive])
            direction_loss = F.smooth_l1_loss(
                direction_pred[positive], direction_gt[positive]
            )
            timing_mae = (
                timing_pred[positive].float() - timing_gt[positive].float()
            ).abs().mean()
        else:
            timing_loss = prediction.new_zeros(())
            direction_loss = prediction.new_zeros(())
            timing_mae = prediction.new_zeros(())
        pred_event = transition_logit >= 0
        true_event = targets["transition"]
        tp = (pred_event & true_event).sum().float()
        fp = (pred_event & ~true_event).sum().float()
        fn = (~pred_event & true_event).sum().float()
        f1 = 2.0 * tp / (2.0 * tp + fp + fn).clamp_min(1.0)
        total = (
            arm_loss
            + gripper_final_loss
            + transition_loss
            + timing_loss
            + direction_loss
        )
        return total, {
            f"{prefix}_loss": total,
            f"{prefix}_arm_rmse": arm_rmse,
            f"{prefix}_gripper_final_rmse": gripper_final_rmse,
            f"{prefix}_gripper_f1": f1,
            f"{prefix}_transition_timing_mae": timing_mae,
        }

    def semantic_negative_actions(
        self,
        action_chunk: Tensor,
        *,
        state: Tensor,
        current_compressed: Tensor,
        past_last_action: Tensor,
    ) -> tuple[Tensor, tuple[str, ...]]:
        matched, _ = self.corrupt_actions(
            action_chunk, state=state, current_compressed=current_compressed
        )
        g = self.gripper_dim_index
        delay = min(self.semantic_negative_delay, max(1, self.action_horizon - 1))
        delayed = action_chunk.clone()
        delayed[:, :delay, g] = past_last_action[:, g : g + 1]
        delayed[:, delay:, g] = action_chunk[:, :-delay, g]
        no_transition = action_chunk.clone()
        no_transition[..., g] = past_last_action[:, g : g + 1]
        tail_hold = action_chunk.clone()
        pivot = max(1, self.action_horizon // 2)
        tail_hold[:, pivot:] = action_chunk[:, pivot - 1 : pivot]
        negatives = torch.stack([matched, delayed, no_transition, tail_hold], dim=1)
        return negatives, ("matched", "gripper_delay", "gripper_remove", "tail_hold")

    @staticmethod
    def _weighted_mean(value: Tensor, weight: Tensor | None) -> Tensor:
        if weight is None:
            return value.mean()
        if value.shape != weight.shape:
            raise ValueError("contrastive value and weight must share shape")
        weight = weight.to(device=value.device, dtype=value.dtype)
        return (value * weight).sum() / weight.sum().clamp_min(1e-6)

    @staticmethod
    def contrastive_alignment_terms(
        action_embedding: Tensor,
        future_embedding: Tensor,
        negative_action_embedding: Tensor | None = None,
        *,
        temperature: float,
        structured_negative_weight: float = 0.0,
        negative_valid: Tensor | None = None,
        duplicate_mask: Tensor | None = None,
        sample_weight: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Per-prefix symmetric InfoNCE with optional structured negatives.

        The previous positive-cosine plus hardest-negative hinge admitted the
        collapsed solution ``all embeddings -> one direction``.  Here each
        action prefix must identify its paired future change among all other
        samples in the batch, in both directions.  Structured action
        corruptions are an additional local NCE task, never a hinge objective.
        """
        if action_embedding.shape != future_embedding.shape:
            raise ValueError("positive action/future embeddings must share shape")
        if action_embedding.ndim != 3:
            raise ValueError("action/future embeddings must be [B,T,E]")
        if temperature <= 0:
            raise ValueError("contrastive temperature must be positive")
        batch, prefixes, _ = action_embedding.shape
        action = F.normalize(action_embedding.float(), dim=-1)
        future = F.normalize(future_embedding.float(), dim=-1)
        if sample_weight is not None and tuple(sample_weight.shape) != (batch, prefixes):
            raise ValueError("sample_weight must be [B,T]")
        if duplicate_mask is not None:
            if tuple(duplicate_mask.shape) == (batch, batch):
                duplicate_mask = duplicate_mask.unsqueeze(0).expand(prefixes, -1, -1)
            elif tuple(duplicate_mask.shape) != (prefixes, batch, batch):
                raise ValueError("duplicate_mask must be [B,B] or [T,B,B]")
            duplicate_mask = duplicate_mask.to(device=action.device, dtype=torch.bool).clone()
            eye = torch.eye(batch, device=action.device, dtype=torch.bool).unsqueeze(0)
            duplicate_mask = duplicate_mask & ~eye

        labels = torch.arange(batch, device=action.device)
        a2f_losses: list[Tensor] = []
        f2a_losses: list[Tensor] = []
        a2f_acc: list[Tensor] = []
        f2a_acc: list[Tensor] = []
        batch_hardest: list[Tensor] = []
        for prefix_idx in range(prefixes):
            logits = action[:, prefix_idx] @ future[:, prefix_idx].T
            prefix_duplicate_mask = None
            if duplicate_mask is not None:
                prefix_duplicate_mask = duplicate_mask[prefix_idx]
                logits = logits.masked_fill(prefix_duplicate_mask, -1e4)
            scaled = logits / float(temperature)
            a2f = F.cross_entropy(scaled, labels, reduction="none")
            f2a = F.cross_entropy(scaled.T, labels, reduction="none")
            weight = None if sample_weight is None else sample_weight[:, prefix_idx]
            a2f_losses.append(CleanFutureLatentDynamics._weighted_mean(a2f, weight))
            f2a_losses.append(CleanFutureLatentDynamics._weighted_mean(f2a, weight))
            a2f_acc.append((scaled.argmax(dim=1) == labels).float().mean())
            f2a_acc.append((scaled.T.argmax(dim=1) == labels).float().mean())
            offdiag = torch.eye(batch, device=logits.device, dtype=torch.bool)
            if prefix_duplicate_mask is not None:
                offdiag = offdiag | prefix_duplicate_mask
            candidate = logits.masked_fill(offdiag, -1.0)
            batch_hardest.append(candidate.max(dim=1).values)

        a2f_loss = torch.stack(a2f_losses).mean()
        f2a_loss = torch.stack(f2a_losses).mean()
        symmetric_nce = 0.5 * (a2f_loss + f2a_loss)
        batch_hardest_tensor = torch.stack(batch_hardest, dim=1)

        structured_nce = action.new_zeros(())
        structured_hardest = action.new_full((batch, prefixes), -1.0)
        hardest_negative_index = action.new_zeros(())
        all_negative_cosine = action.new_zeros((batch, 0, prefixes))
        structured_valid_fraction = action.new_zeros(())
        if negative_action_embedding is not None:
            if negative_action_embedding.ndim != 4:
                raise ValueError("negative_action_embedding must be [B,K,T,E]")
            if negative_action_embedding.shape[0] != batch or negative_action_embedding.shape[2:] != action_embedding.shape[1:]:
                raise ValueError("structured negatives must match [B,K,T,E]")
            negative = F.normalize(negative_action_embedding.float(), dim=-1)
            all_negative_cosine = torch.einsum("bkte,bte->bkt", negative, future)
            if negative_valid is None:
                negative_valid = torch.ones(
                    batch, negative.shape[1], prefixes,
                    device=action.device, dtype=torch.bool
                )
            elif tuple(negative_valid.shape) == (batch, negative.shape[1]):
                negative_valid = negative_valid.unsqueeze(-1).expand(-1, -1, prefixes)
            elif tuple(negative_valid.shape) != (batch, negative.shape[1], prefixes):
                raise ValueError("negative_valid must be [B,K] or [B,K,T]")
            negative_valid = negative_valid.to(device=action.device, dtype=torch.bool)
            structured_valid_fraction = negative_valid.float().mean()
            local_losses: list[Tensor] = []
            local_hardest_indices: list[Tensor] = []
            for prefix_idx in range(prefixes):
                positive_logit = (action[:, prefix_idx] * future[:, prefix_idx]).sum(dim=-1, keepdim=True)
                prefix_negative_valid = negative_valid[:, :, prefix_idx]
                negative_logits = all_negative_cosine[:, :, prefix_idx]
                negative_logits = negative_logits.masked_fill(~prefix_negative_valid, -1e4)
                logits = torch.cat([positive_logit, negative_logits], dim=1) / float(temperature)
                target = torch.zeros(batch, device=action.device, dtype=torch.long)
                row_loss = F.cross_entropy(logits, target, reduction="none")
                weight = None if sample_weight is None else sample_weight[:, prefix_idx]
                local_losses.append(CleanFutureLatentDynamics._weighted_mean(row_loss, weight))
                valid_cosine = all_negative_cosine[:, :, prefix_idx].masked_fill(~prefix_negative_valid, -1.0)
                values, indices = valid_cosine.max(dim=1)
                structured_hardest[:, prefix_idx] = values
                local_hardest_indices.append(indices.float().mean())
            structured_nce = torch.stack(local_losses).mean()
            hardest_negative_index = torch.stack(local_hardest_indices).mean()

        total = symmetric_nce + float(structured_negative_weight) * structured_nce
        positive = (action * future).sum(dim=-1)
        hardest = torch.maximum(batch_hardest_tensor, structured_hardest)
        sample_margin = (positive - hardest).mean(dim=1)
        return {
            "loss": total,
            "symmetric_nce_loss": symmetric_nce,
            "action_to_future_nce_loss": a2f_loss,
            "future_to_action_nce_loss": f2a_loss,
            "structured_nce_loss": structured_nce,
            "positive_cosine": positive.mean(),
            "negative_cosine": hardest.mean(),
            "batch_negative_cosine": batch_hardest_tensor.mean(),
            "structured_negative_cosine": structured_hardest.mean(),
            "margin": (positive - hardest).mean(),
            "sample_margin": sample_margin,
            "action_to_future_top1": torch.stack(a2f_acc).mean(),
            "future_to_action_top1": torch.stack(f2a_acc).mean(),
            "structured_valid_fraction": structured_valid_fraction,
            "hardest_negative_index": hardest_negative_index,
            "all_negative_cosine": all_negative_cosine,
        }

    @staticmethod
    def embedding_regularization_terms(
        action_embedding: Tensor,
        future_embedding: Tensor,
        *,
        std_target: float,
    ) -> dict[str, Tensor]:
        """VICReg-style anti-collapse regularization on the shared space."""
        if action_embedding.shape != future_embedding.shape or action_embedding.ndim != 3:
            raise ValueError("embedding regularization expects matching [B,T,E] tensors")
        if std_target < 0:
            raise ValueError("embedding std target must be non-negative")

        def one_branch(x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
            flat = x.float().reshape(-1, x.shape[-1])
            centered = flat - flat.mean(dim=0, keepdim=True)
            if flat.shape[0] > 1:
                std = torch.sqrt(centered.var(dim=0, unbiased=True) + 1e-4)
                covariance = centered.T @ centered / float(flat.shape[0] - 1)
            else:
                std = torch.sqrt(centered.square().mean(dim=0) + 1e-4)
                covariance = centered.T @ centered
            variance_loss = F.relu(float(std_target) - std).mean()
            covariance = covariance - torch.diag_embed(torch.diagonal(covariance))
            covariance_loss = covariance.square().sum() / float(max(1, x.shape[-1]))
            return variance_loss, covariance_loss, std.mean()

        action_var, action_cov, action_std = one_branch(action_embedding)
        future_var, future_cov, future_std = one_branch(future_embedding)
        return {
            "variance_loss": 0.5 * (action_var + future_var),
            "covariance_loss": 0.5 * (action_cov + future_cov),
            "action_std": action_std,
            "future_std": future_std,
        }

    def _embed_future(self, noisy: Tensor, action_prefix_hidden: Tensor) -> Tensor:
        expected = (
            noisy.shape[0],
            self.num_future_frames,
            self.num_cameras,
            self.spatial_tokens,
            self.latent_dim,
        )
        if tuple(noisy.shape) != expected:
            raise ValueError(f"unexpected noisy future residual shape={tuple(noisy.shape)}")
        hidden = self.future_input_proj(noisy)
        hidden = (
            hidden
            + self.future_temporal_embedding
            + self.future_camera_embedding
            + self.future_spatial_embedding
        )
        # Direct, ungated action-prefix conditioning for t+offset blocks.
        action_bias = self.action_to_future(action_prefix_hidden)
        hidden = hidden + action_bias[:, :, None, None, :]
        return (
            hidden.reshape(noisy.shape[0], self.total_future_tokens, self.hidden_size)
            + self.future_type_embedding
        )

    def _forward_core(
        self,
        *,
        current_compressed: Tensor,
        action_chunk: Tensor,
        state: Tensor,
        past_last_action: Tensor,
        future_noisy: Tensor,
        future_time: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]:
        current = self._embed_current(current_compressed)
        action, action_prefix_hidden, action_embedding = self.encode_action(
            action_chunk, past_last_action=past_last_action, state=state
        )
        future = self._embed_future(future_noisy, action_prefix_hidden)
        time = self.time_embedder(future_time)
        if time.shape[0] == 1:
            time = time.expand(action.shape[0], -1)
        prepared = self.modulation.prepare(
            torch.cat([time, self.task_embedding.expand_as(time)], dim=-1)
        )
        action_mask = self.future_action_prefix_mask.to(device=future.device).unsqueeze(0).expand(
            future.shape[0], -1, -1
        )
        future_mask = self.future_temporal_mask.to(device=future.device).unsqueeze(0).expand(
            future.shape[0], -1, -1
        )
        future_diagnostics: list[dict[str, Tensor]] = []
        for layer_idx, future_block in enumerate(self.future_blocks):
            future, row = future_block(
                future,
                self.modulation.for_layer(prepared, layer_idx),
                current=current,
                action=action,
                action_mask=action_mask,
                future_mask=future_mask,
            )
            future_diagnostics.append(row)
        velocity = self.output_proj(self.output_norm(future)).reshape(
            future.shape[0],
            self.num_future_frames,
            self.num_cameras,
            self.spatial_tokens,
            self.latent_dim,
        )
        diagnostics: dict[str, Tensor] = {
            # Compatibility plus explicit structural-state metrics.
            "future_action_encoder_self_gate": velocity.new_tensor(1.0),
            "future_action_encoder_current_gate": velocity.new_tensor(0.0),
            "future_action_encoder_ffn_gate": velocity.new_tensor(1.0),
            "future_action_embedding_rms": action_embedding.square().mean().sqrt(),
            "future_action_prefix_hidden_rms": action_prefix_hidden.square().mean().sqrt(),
            "future_action_direct_injection_rms": self.action_to_future(
                action_prefix_hidden
            ).square().mean().sqrt(),
        }
        for key in future_diagnostics[0]:
            diagnostics[key] = torch.stack([row[key] for row in future_diagnostics]).mean()
        return velocity, diagnostics, {
            "action_tokens": action,
            "action_prefix_hidden": action_prefix_hidden,
            "action_embedding": action_embedding,
        }

    def forward(self, **kwargs: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        velocity, diagnostics, _ = self._forward_core(**kwargs)
        return velocity, diagnostics

    def forward_with_aux(
        self, **kwargs: Tensor
    ) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]:
        return self._forward_core(**kwargs)

    def detached_parameter_forward(self, **kwargs: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        state: dict[str, Tensor] = {
            name: parameter.detach() for name, parameter in self.named_parameters()
        }
        state.update({name: buffer.detach() for name, buffer in self.named_buffers()})
        return _functional_call(self, state, (), kwargs, strict=False)

    @staticmethod
    def corrupt_actions(
        action_chunk: Tensor,
        *,
        state: Tensor,
        current_compressed: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if action_chunk.ndim != 3:
            raise ValueError("action_chunk must be [B,H,A]")
        if state.ndim != 2 or state.shape[0] != action_chunk.shape[0]:
            raise ValueError("state must be [B,D]")
        if current_compressed.ndim != 4 or current_compressed.shape[0] != action_chunk.shape[0]:
            raise ValueError("current_compressed must be [B,C,S,D]")
        batch = action_chunk.shape[0]
        if batch <= 1:
            return torch.flip(action_chunk, dims=(1,)), action_chunk.new_tensor(2.0)
        state_descriptor = F.normalize(state.detach().float(), dim=-1)
        visual_descriptor = F.normalize(
            current_compressed.detach().float().mean(dim=(1, 2)), dim=-1
        )
        descriptor = torch.cat([state_descriptor, visual_descriptor], dim=-1)
        descriptor_distance = torch.cdist(descriptor, descriptor)
        action_distance = (
            action_chunk.detach().float()[:, None]
            - action_chunk.detach().float()[None, :]
        ).square().mean(dim=(2, 3))
        invalid = torch.eye(batch, dtype=torch.bool, device=action_chunk.device)
        invalid |= action_distance <= 1e-8
        descriptor_distance = descriptor_distance.masked_fill(invalid, float("inf"))
        nearest = descriptor_distance.argmin(dim=1)
        no_valid = torch.isinf(descriptor_distance).all(dim=1)
        if no_valid.any():
            fallback = (torch.arange(batch, device=action_chunk.device) + 1) % batch
            nearest = torch.where(no_valid, fallback, nearest)
        corrupted = action_chunk.index_select(0, nearest)
        return corrupted, action_chunk.new_tensor(3.0)

    def weighted_mse(
        self,
        predicted: Tensor,
        target: Tensor,
        weights: Tensor,
        *,
        reduction: str = "mean",
    ) -> Tensor:
        if predicted.shape != target.shape:
            raise ValueError("predicted and target future velocities must share shape")
        error = (predicted - target).square().mean(dim=-1, keepdim=True) * weights
        if reduction == "none":
            return error.mean(dim=(1, 2, 3, 4))
        if reduction != "mean":
            raise ValueError(f"unsupported reduction={reduction!r}")
        return error.mean()

    def flow_metrics(
        self,
        predicted_velocity: Tensor,
        flow: FutureLatentFlowBatch,
        *,
        current_compressed: Tensor,
        future_compressed: Tensor,
        residual_raw: Tensor,
        motion_weights: Tensor,
        prefix: str = "future_latent",
    ) -> dict[str, Tensor]:
        if predicted_velocity.shape != flow.target_velocity.shape:
            raise ValueError("future velocity shape mismatch")
        error_channel = (predicted_velocity - flow.target_velocity).square()
        error_patch = error_channel.mean(dim=-1)
        weighted_flow = self.weighted_mse(
            predicted_velocity, flow.target_velocity, motion_weights
        )
        unweighted_flow = error_channel.mean()
        remaining = 1.0 - flow.time.reshape(-1, 1, 1, 1, 1)
        residual_endpoint_normalized = flow.noisy + remaining * predicted_velocity
        residual_endpoint = self.denormalize_residual(residual_endpoint_normalized)
        future_endpoint = current_compressed[:, None] + residual_endpoint
        absolute_error = future_endpoint - future_compressed
        endpoint_rmse = absolute_error.float().square().mean().sqrt()
        residual_error = residual_endpoint_normalized - flow.target_normalized
        residual_r2 = 1.0 - residual_error.float().square().mean() / flow.target_normalized.float().var(
            unbiased=False
        ).clamp_min(1e-8)
        target_flat = flow.target_velocity.float().reshape(flow.target_velocity.shape[0], -1)
        pred_flat = predicted_velocity.float().reshape(predicted_velocity.shape[0], -1)
        cosine = F.cosine_similarity(pred_flat, target_flat, dim=1).mean()
        magnitude = residual_raw.float().square().mean(dim=-1).sqrt()
        threshold = magnitude.median(dim=3, keepdim=True).values
        dynamic = magnitude >= threshold
        static = ~dynamic
        dynamic_mse = error_patch.float()[dynamic].mean() if dynamic.any() else error_patch.new_zeros(())
        static_mse = error_patch.float()[static].mean() if static.any() else error_patch.new_zeros(())
        metrics: dict[str, Tensor] = {
            f"{prefix}_flow_mse": weighted_flow,
            f"{prefix}_flow_mse_unweighted": unweighted_flow,
            f"{prefix}_absolute_endpoint_rmse": endpoint_rmse,
            f"{prefix}_residual_endpoint_r2": residual_r2,
            f"{prefix}_velocity_cosine": cosine,
            f"{prefix}_time_mean": flow.time.mean(),
            f"{prefix}_residual_target_rms": flow.target_normalized.square().mean().sqrt(),
            f"{prefix}_velocity_target_rms": flow.target_velocity.square().mean().sqrt(),
            f"{prefix}_velocity_pred_rms": predicted_velocity.square().mean().sqrt(),
            f"{prefix}_dynamic_patch_mse": dynamic_mse,
            f"{prefix}_static_patch_mse": static_mse,
            f"{prefix}_motion_weight_mean": motion_weights.mean(),
            f"{prefix}_motion_weight_max": motion_weights.max(),
        }
        per_time_camera = error_channel.mean(dim=(0, 3, 4))
        for time_idx in range(self.num_future_frames):
            metrics[f"{prefix}_t{time_idx}_mse"] = per_time_camera[time_idx].mean()
            for camera_idx in range(self.num_cameras):
                metrics[f"{prefix}_t{time_idx}_c{camera_idx}_mse"] = per_time_camera[
                    time_idx, camera_idx
                ]
        return metrics


__all__ = [
    "CleanFutureLatentDynamics",
    "CurrentActionBaselineHead",
    "FutureChangeEncoder",
    "FutureLatentFlowBatch",
    "InverseActionHead",
    "sample_future_latent_flow",
]
