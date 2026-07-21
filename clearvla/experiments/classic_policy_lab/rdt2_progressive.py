from __future__ import annotations

"""Progressive history-anchored RDT2-FM experiment.

This module is intentionally separate from :mod:`rdt2_fm_reference`.  The v18
reference remains an unchanged baseline.  The progressive model tests a more
control-oriented decomposition:

* an explicit history-only trajectory prior;
* a residual flow bridge around that prior rather than a Gaussian full-action
  generation problem;
* first-action and near-prefix exits supervised inside the shared action
  decoder;
* stage-shared low-rank modulation instead of one 2H -> 9H hypernetwork per
  block;
* a cheap pooled-visual correction path for the earliest executable action.

The full dense-token cross-attention path remains available for near-prefix and
full-chunk refinement.  External RDT2-VQ KV caches remain supported for the
full path, although the cheap pooled-visual first-action correction is available
only when dense condition tokens are present.
"""

import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from clearvla.experiments.residual_flow_lab.flow import ResidualBridgeConfig, sample_residual_bridge
from .rdt2_fm_reference import (
    Attention,
    CrossAttention,
    FeedForward,
    FinalLayer,
    RMSNorm,
    TimestepEmbedder,
    get_multimodal_pos_embed,
)
from collections import OrderedDict


def _weighted_mse(pred: Tensor, target: Tensor, weights: Tensor) -> Tensor:
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must share shape, got {tuple(pred.shape)} vs {tuple(target.shape)}"
        )
    if weights.ndim != 1 or weights.shape[0] != pred.shape[1]:
        raise ValueError(f"weights must be [H={pred.shape[1]}], got {tuple(weights.shape)}")
    return ((pred - target).square() * weights.reshape(1, -1, 1)).mean()


def _prefix_weights(
    horizon: int,
    *,
    first: float,
    first4: float,
    first8: float,
    tail: float,
    device: torch.device | None = None,
) -> Tensor:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if min(first, first4, first8, tail) <= 0:
        raise ValueError("prefix weights must be positive")
    weights = torch.full((horizon,), float(tail), device=device)
    weights[: min(8, horizon)] = float(first8)
    weights[: min(4, horizon)] = float(first4)
    weights[0] = float(first)
    return weights / weights.mean()


@dataclass(frozen=True)
class ProgressiveRDT2FMConfig:
    action_dim: int = 7
    state_dim: int = 7
    prediction_horizon: int = 24
    hidden_size: int = 512
    depth: int = 8
    num_heads: int = 8
    num_kv_heads: int = 4
    num_register_tokens: int = 4
    norm_eps: float = 1e-5
    multiple_of: int = 256
    ffn_dim_multiplier: float | None = None
    use_flash_attn: bool = True
    num_inference_timesteps: int = 5

    # Dense-token plugin contract.  The external-KV path leaves these unset.
    lang_adaptor: str | None = None
    lang_token_dim: int | None = None

    # Explicit local-motion path.
    history_hidden_size: int = 128
    history_layers: int = 1
    prior_residual_scale: float = 1.0
    history_noise_std: float = 0.01

    # Progressive exits.  Values count completed transformer blocks.
    fast_exit_layer: int = 2
    prefix_exit_layer: int = 4
    prefix_length: int = 4
    visual_start_layer: int = 2

    # Stage-shared low-rank modulation.
    modulation_rank: int = 128

    # Prefix-priority flow objective.
    first_position_weight: float = 8.0
    first4_position_weight: float = 4.0
    first8_position_weight: float = 2.0
    tail_position_weight: float = 1.0
    prior_loss_weight: float = 0.50
    fast_exit_loss_weight: float = 1.00
    prefix_exit_loss_weight: float = 0.50
    full_flow_loss_weight: float = 1.00

    # Residual bridge around the learned history source.
    bridge_clean_probability: float = 0.50
    bridge_mild_probability: float = 0.35
    bridge_strong_probability: float = 0.15
    bridge_mild_noise_std: float = 0.05
    bridge_strong_noise_std: float = 0.15
    bridge_mild_velocity_bias_std: float = 0.02
    bridge_strong_velocity_bias_std: float = 0.06

    def validate(self) -> None:
        positive = (
            self.action_dim,
            self.state_dim,
            self.prediction_horizon,
            self.hidden_size,
            self.depth,
            self.num_heads,
            self.num_kv_heads,
            self.history_hidden_size,
            self.history_layers,
            self.fast_exit_layer,
            self.prefix_exit_layer,
            self.prefix_length,
            self.modulation_rank,
        )
        if min(positive) <= 0:
            raise ValueError("progressive RDT2-FM dimensions must be positive")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if not (self.fast_exit_layer <= self.prefix_exit_layer <= self.depth):
            raise ValueError("require fast_exit_layer <= prefix_exit_layer <= depth")
        if not (0 <= self.visual_start_layer <= self.fast_exit_layer):
            raise ValueError("visual_start_layer must be in [0, fast_exit_layer]")
        if self.prefix_length > self.prediction_horizon:
            raise ValueError("prefix_length cannot exceed prediction_horizon")
        if self.lang_adaptor is not None and self.lang_token_dim is None:
            raise ValueError("lang_token_dim is required when lang_adaptor is enabled")
        if self.prior_residual_scale < 0 or self.history_noise_std < 0:
            raise ValueError("prior scales must be non-negative")
        probabilities = (
            self.bridge_clean_probability,
            self.bridge_mild_probability,
            self.bridge_strong_probability,
        )
        if any(value < 0 for value in probabilities) or abs(sum(probabilities) - 1.0) > 1e-6:
            raise ValueError("bridge probabilities must be non-negative and sum to 1")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    def bridge_config(self) -> ResidualBridgeConfig:
        return ResidualBridgeConfig(
            clean_probability=self.bridge_clean_probability,
            mild_probability=self.bridge_mild_probability,
            strong_probability=self.bridge_strong_probability,
            mild_noise_std=self.bridge_mild_noise_std,
            strong_noise_std=self.bridge_strong_noise_std,
            mild_velocity_bias_std=self.bridge_mild_velocity_bias_std,
            strong_velocity_bias_std=self.bridge_strong_velocity_bias_std,
        )


class HistoryTrajectoryPrior(nn.Module):
    """Small history-only trajectory source anchored at the physical hold prior."""

    def __init__(self, config: ProgressiveRDT2FMConfig) -> None:
        super().__init__()
        self.config = config
        h = int(config.history_hidden_size)
        self.action_in = nn.Linear(config.action_dim, h)
        self.state_in = nn.Linear(config.state_dim, h)
        self.encoder = nn.GRU(h, h, num_layers=config.history_layers, batch_first=True)
        self.future_queries = nn.Parameter(torch.randn(config.prediction_horizon, h) * 0.02)
        heads = 4 if h % 4 == 0 else 1
        self.cross = nn.MultiheadAttention(h, heads, batch_first=True)
        self.norm = nn.LayerNorm(h)
        self.residual_head = nn.Linear(h, config.action_dim)
        self.context_proj = nn.Linear(h, config.hidden_size)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(
        self, past_actions: Tensor, state_tokens: Tensor, physical_prior: Tensor
    ) -> tuple[Tensor, Tensor]:
        cfg = self.config
        if past_actions.ndim != 3 or past_actions.shape[-1] != cfg.action_dim:
            raise ValueError(
                f"past_actions must be [B,H,{cfg.action_dim}], got {tuple(past_actions.shape)}"
            )
        if state_tokens.ndim != 2 or state_tokens.shape[-1] != cfg.state_dim:
            raise ValueError(
                f"state_tokens must be [B,{cfg.state_dim}], got {tuple(state_tokens.shape)}"
            )
        if tuple(physical_prior.shape[1:]) != (cfg.prediction_horizon, cfg.action_dim):
            raise ValueError(
                f"physical_prior must be [B,{cfg.prediction_horizon},{cfg.action_dim}]"
            )
        history = past_actions
        if self.training and cfg.history_noise_std > 0:
            history = history + torch.randn_like(history) * cfg.history_noise_std
        state = self.state_in(state_tokens)
        encoded, hidden = self.encoder(self.action_in(history))
        query = self.future_queries.unsqueeze(0).expand(history.shape[0], -1, -1) + state.unsqueeze(
            1
        )
        attended, _ = self.cross(query, encoded, encoded, need_weights=False)
        query = self.norm(query + attended)
        delta = torch.tanh(self.residual_head(query)) * cfg.prior_residual_scale
        prior = physical_prior + delta
        context = self.context_proj(hidden[-1] + state)
        return prior, context


class StageModulationBank(nn.Module):
    """Stage-shared low-rank modulation with tiny block-specific affine adapters."""

    def __init__(self, config: ProgressiveRDT2FMConfig) -> None:
        super().__init__()
        self.config = config
        h, rank = config.hidden_size, config.modulation_rank
        self.trunk = nn.Sequential(nn.SiLU(), nn.Linear(3 * h, rank), nn.SiLU())
        self.stage_heads = nn.ModuleList([nn.Linear(rank, 9 * h) for _ in range(3)])
        self.block_scale = nn.Parameter(torch.ones(config.depth, 9 * h))
        self.block_bias = nn.Parameter(torch.zeros(config.depth, 9 * h))
        for head in self.stage_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _stage(self, layer_idx: int) -> int:
        cfg = self.config
        if layer_idx < cfg.fast_exit_layer:
            return 0
        if layer_idx < cfg.prefix_exit_layer:
            return 1
        return 2

    def prepare(self, condition: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if condition.ndim != 2 or condition.shape[1] != 3 * self.config.hidden_size:
            raise ValueError(
                f"condition must be [B,{3 * self.config.hidden_size}], got {tuple(condition.shape)}"
            )
        latent = self.trunk(condition)
        return tuple(head(latent) for head in self.stage_heads)  # type: ignore[return-value]

    def for_layer(self, prepared: tuple[Tensor, Tensor, Tensor], layer_idx: int) -> Tensor:
        base = prepared[self._stage(layer_idx)]
        return base * self.block_scale[layer_idx] + self.block_bias[layer_idx]


class ProgressiveRDTBlock(nn.Module):
    def __init__(self, layer_idx: int, config: ProgressiveRDT2FMConfig) -> None:
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.hidden_size = int(config.hidden_size)
        core = {
            "hidden_size": config.hidden_size,
            "num_heads": config.num_heads,
            "num_kv_heads": config.num_kv_heads,
            "norm_eps": config.norm_eps,
            "multiple_of": config.multiple_of,
            "ffn_dim_multiplier": config.ffn_dim_multiplier,
            "use_flash_attn": config.use_flash_attn,
        }
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = Attention(core)
        self.cross_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.cond_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.cross_attn = CrossAttention(core)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn = FeedForward(
            config.hidden_size,
            4 * config.hidden_size,
            config.multiple_of,
            config.ffn_dim_multiplier,
        )

    @staticmethod
    def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(
        self,
        x: Tensor,
        modulation: Tensor,
        *,
        c: Tensor | None = None,
        ck: Tensor | None = None,
        cv: Tensor | None = None,
        mask: Tensor | None = None,
        use_cross_attention: bool = True,
    ) -> Tensor:
        if modulation.ndim != 2 or modulation.shape[1] != 9 * self.hidden_size:
            raise ValueError(
                f"modulation must be [B,{9 * self.hidden_size}], got {tuple(modulation.shape)}"
            )
        (
            shift_attn,
            scale_attn,
            gate_attn,
            shift_cross,
            scale_cross,
            gate_cross,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = modulation.chunk(9, dim=1)
        h = x + gate_attn.unsqueeze(1) * self.attn(
            self._modulate(self.attn_norm(x), shift_attn, scale_attn)
        )
        if use_cross_attention:
            if c is not None:
                cross = self.cross_attn(
                    self._modulate(self.cross_norm(h), shift_cross, scale_cross),
                    c=self.cond_norm(c),
                    mask=mask,
                )
            else:
                cross = self.cross_attn(
                    self._modulate(self.cross_norm(h), shift_cross, scale_cross),
                    ck=ck,
                    cv=cv,
                    mask=mask,
                )
            h = h + gate_cross.unsqueeze(1) * cross
        return h + gate_mlp.unsqueeze(1) * self.ffn(
            self._modulate(self.ffn_norm(h), shift_mlp, scale_mlp)
        )


class ExitHead(nn.Module):
    """Lightweight native exit head for executable prefixes."""

    def __init__(self, hidden_size: int, output_size: int, norm_eps: float) -> None:
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=norm_eps)
        self.fc1 = nn.Linear(hidden_size, 2 * hidden_size)
        self.fc2 = nn.Linear(2 * hidden_size, output_size)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.silu(self.fc1(self.norm(x))))


class PooledVisualFirstCorrector(nn.Module):
    """Cheap dense-token visual correction for the earliest first-action exit."""

    def __init__(self, hidden_size: int, action_dim: int) -> None:
        super().__init__()
        self.delta = nn.Sequential(
            nn.Linear(3 * hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, action_dim)
        )
        self.gate = nn.Sequential(
            nn.Linear(3 * hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, action_dim)
        )
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(
        self, *, state: Tensor, history: Tensor, dense_condition: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        if dense_condition is None:
            zeros = state.new_zeros((state.shape[0], self.delta[-1].out_features))
            return zeros, zeros
        visual = dense_condition.mean(dim=1)
        joint = torch.cat([state, history, visual], dim=-1)
        gate = torch.sigmoid(self.gate(joint))
        correction = gate * self.delta(joint)
        return correction, gate


@dataclass
class ProgressiveVelocityOutput:
    fast_first: Tensor
    prefix: Tensor
    full: Tensor | None
    visual_gate_mean: Tensor
    visual_correction_rms: Tensor


class ProgressiveRDTCore(nn.Module):
    def __init__(self, config: ProgressiveRDT2FMConfig, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.depth = config.depth
        self.t_embedder = TimestepEmbedder(config.hidden_size, dtype=dtype)
        self.blocks = nn.ModuleList(
            [ProgressiveRDTBlock(idx, config) for idx in range(config.depth)]
        )
        self.modulation = StageModulationBank(config)
        self.final_layer = FinalLayer(
            config.action_dim,
            {
                "hidden_size": config.hidden_size,
                "norm_eps": config.norm_eps,
            },
        )
        self.first_exit_head = ExitHead(config.hidden_size, config.action_dim, config.norm_eps)
        self.prefix_exit_head = ExitHead(config.hidden_size, config.action_dim, config.norm_eps)
        self.visual_first_corrector = PooledVisualFirstCorrector(
            config.hidden_size, config.action_dim
        )
        self.num_register_tokens = config.num_register_tokens
        self.register_tokens = nn.Parameter(
            torch.randn(1, config.num_register_tokens, config.hidden_size)
        )
        self.x_pos_emb = nn.Parameter(
            torch.zeros(
                1, config.prediction_horizon + config.num_register_tokens, config.hidden_size
            )
        )
        self.state_pos_emb = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self._initialize(dtype)

    def _initialize(self, dtype: torch.dtype) -> None:
        def basic(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(basic)
        cfg = self.config
        x_pos = get_multimodal_pos_embed(
            cfg.hidden_size,
            OrderedDict(
                [("action", cfg.prediction_horizon), ("register", cfg.num_register_tokens)]
            ),
        )
        state_pos = get_multimodal_pos_embed(cfg.hidden_size, OrderedDict([("state", 1)]))
        self.x_pos_emb.data.copy_(torch.from_numpy(x_pos).float().unsqueeze(0))
        self.state_pos_emb.data.copy_(torch.from_numpy(state_pos).float().unsqueeze(0))
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for head in self.modulation.stage_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.ffn.fc2.weight)
        nn.init.zeros_(self.final_layer.ffn.fc2.bias)
        nn.init.zeros_(self.first_exit_head.fc2.weight)
        nn.init.zeros_(self.first_exit_head.fc2.bias)
        nn.init.zeros_(self.prefix_exit_head.fc2.weight)
        nn.init.zeros_(self.prefix_exit_head.fc2.bias)
        nn.init.zeros_(self.visual_first_corrector.delta[-1].weight)
        nn.init.zeros_(self.visual_first_corrector.delta[-1].bias)
        nn.init.zeros_(self.visual_first_corrector.gate[-1].weight)
        nn.init.zeros_(self.visual_first_corrector.gate[-1].bias)
        self.to(dtype=dtype)

    def _condition_for_layer(
        self,
        layer_idx: int,
        *,
        dense_condition: Tensor | None,
        kv_cache: list[tuple[Tensor, Tensor]] | None,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        if kv_cache is not None:
            key, value = kv_cache[layer_idx % len(kv_cache)]
            return None, key.transpose(1, 2), value.transpose(1, 2)
        return dense_condition, None, None

    def forward(
        self,
        *,
        x: Tensor,
        timesteps: Tensor,
        state_condition: Tensor,
        history_condition: Tensor,
        dense_condition: Tensor | None,
        kv_cache: list[tuple[Tensor, Tensor]] | None,
        attention_mask: Tensor | None,
        stop_after: str = "full",
    ) -> ProgressiveVelocityOutput:
        cfg = self.config
        if stop_after not in {"fast", "prefix", "full"}:
            raise ValueError(f"unknown stop_after={stop_after!r}")
        if dense_condition is None and kv_cache is None:
            raise ValueError("progressive RDT2-FM requires dense condition tokens or KV cache")
        time = self.t_embedder(timesteps)
        if time.shape[0] == 1:
            time = time.expand(x.shape[0], -1)
        state = state_condition.squeeze(1) + self.state_pos_emb.squeeze(1)
        if state.shape != history_condition.shape:
            raise ValueError("state and history conditions must both be [B,H]")
        joint = torch.cat([time, state, history_condition], dim=-1)
        x = torch.cat([x, self.register_tokens.expand(x.shape[0], -1, -1)], dim=1) + self.x_pos_emb
        fast = prefix = full = None
        visual_correction = visual_gate = None
        prepared_modulation = self.modulation.prepare(joint)
        for layer_idx, block in enumerate(self.blocks):
            use_cross = layer_idx >= cfg.visual_start_layer
            c = ck = cv = None
            if use_cross:
                c, ck, cv = self._condition_for_layer(
                    layer_idx, dense_condition=dense_condition, kv_cache=kv_cache
                )
            x = block(
                x,
                self.modulation.for_layer(prepared_modulation, layer_idx),
                c=c,
                ck=ck,
                cv=cv,
                mask=attention_mask,
                use_cross_attention=use_cross,
            )
            completed = layer_idx + 1
            if completed == cfg.fast_exit_layer:
                visual_correction, visual_gate = self.visual_first_corrector(
                    state=state,
                    history=history_condition,
                    dense_condition=dense_condition,
                )
                fast = self.first_exit_head(x[:, :1]) + visual_correction.unsqueeze(1)
                if stop_after == "fast":
                    break
            if completed == cfg.prefix_exit_layer:
                prefix = self.prefix_exit_head(x[:, : cfg.prefix_length])
                if visual_correction is None:
                    visual_correction, visual_gate = self.visual_first_corrector(
                        state=state, history=history_condition, dense_condition=dense_condition
                    )
                prefix = prefix.clone()
                prefix[:, :1] = prefix[:, :1] + visual_correction.unsqueeze(1)
                if stop_after == "prefix":
                    break
        if fast is None:
            raise AssertionError("fast exit was not reached")
        if prefix is None and stop_after != "fast":
            raise AssertionError("prefix exit was not reached")
        if stop_after == "full":
            final_modulation = torch.cat([time, state], dim=-1)
            full = self.final_layer(x, final_modulation)[:, : -cfg.num_register_tokens]
        if visual_gate is None or visual_correction is None:
            visual_correction, visual_gate = self.visual_first_corrector(
                state=state, history=history_condition, dense_condition=dense_condition
            )
        return ProgressiveVelocityOutput(
            fast_first=fast,
            prefix=fast if prefix is None else prefix,
            full=full,
            visual_gate_mean=visual_gate.mean(),
            visual_correction_rms=visual_correction.square().mean().sqrt(),
        )


class ProgressiveRDT2FM(nn.Module):
    """History-anchored progressive residual-flow action expert."""

    def __init__(
        self,
        config: ProgressiveRDT2FMConfig = ProgressiveRDT2FMConfig(),
        *,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.model = ProgressiveRDTCore(config, dtype=dtype)
        self.lang_adaptor = self._build_adapter(
            config.lang_adaptor, config.lang_token_dim, config.hidden_size
        )
        self.act_adaptor = self._build_adapter("mlp3x_silu", config.action_dim, config.hidden_size)
        self.state_adaptor = self._build_adapter("mlp3x_silu", config.state_dim, config.hidden_size)
        self.history_prior = HistoryTrajectoryPrior(config)
        self.bridge = config.bridge_config()
        self.register_buffer(
            "position_weights",
            _prefix_weights(
                config.prediction_horizon,
                first=config.first_position_weight,
                first4=config.first4_position_weight,
                first8=config.first8_position_weight,
                tail=config.tail_position_weight,
            ),
            persistent=False,
        )
        self.pred_horizon = config.prediction_horizon
        self.action_dim = config.action_dim
        self.num_inference_timesteps = config.num_inference_timesteps
        self.to(dtype=dtype)

    @staticmethod
    def _build_adapter(
        kind: str | None, in_features: int | None, out_features: int
    ) -> nn.Module | None:
        if kind is None:
            return None
        if in_features is None:
            raise ValueError(f"in_features required for adapter {kind}")
        if kind == "linear":
            return nn.Linear(in_features, out_features)
        match = re.match(r"^mlp(\d+)x_silu$", kind)
        if not match:
            raise ValueError(f"unknown adapter type: {kind}")
        depth = int(match.group(1))
        modules: list[nn.Module] = [nn.Linear(in_features, out_features)]
        for _ in range(1, depth):
            modules.extend([nn.SiLU(), nn.Linear(out_features, out_features)])
        return nn.Sequential(*modules)

    def _adapt_dense(self, dense_tokens: Tensor | None) -> Tensor | None:
        if dense_tokens is None:
            return None
        return self.lang_adaptor(dense_tokens) if self.lang_adaptor is not None else dense_tokens

    def predict_prior(
        self, *, state_tokens: Tensor, past_actions: Tensor, physical_prior: Tensor
    ) -> tuple[Tensor, Tensor]:
        if state_tokens.ndim != 2:
            raise ValueError("state_tokens must be [B,D]")
        return self.history_prior(past_actions, state_tokens, physical_prior)

    def _velocity(
        self,
        *,
        state_tokens: Tensor,
        past_actions: Tensor,
        physical_prior: Tensor,
        residual_state: Tensor,
        timesteps: Tensor,
        dense_tokens: Tensor | None,
        kv_cache: list[tuple[Tensor, Tensor]] | None,
        attention_mask: Tensor | None,
        stop_after: str,
        learned_prior: Tensor | None = None,
        history_context: Tensor | None = None,
    ) -> tuple[ProgressiveVelocityOutput, Tensor]:
        if learned_prior is None or history_context is None:
            learned_prior, history_context = self.predict_prior(
                state_tokens=state_tokens, past_actions=past_actions, physical_prior=physical_prior
            )
        state = self.state_adaptor(state_tokens.unsqueeze(1))
        action = self.act_adaptor(residual_state)
        output = self.model(
            x=action,
            timesteps=timesteps,
            state_condition=state,
            history_condition=history_context,
            dense_condition=self._adapt_dense(dense_tokens),
            kv_cache=kv_cache,
            attention_mask=attention_mask,
            stop_after=stop_after,
        )
        return output, learned_prior

    def compute_loss(
        self,
        *,
        state_tokens: Tensor,
        past_actions: Tensor,
        physical_prior: Tensor,
        action_gt: Tensor,
        dense_tokens: Tensor | None = None,
        kv_cache: list[tuple[Tensor, Tensor]] | None = None,
        attention_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        learned_prior, history_context = self.predict_prior(
            state_tokens=state_tokens, past_actions=past_actions, physical_prior=physical_prior
        )
        bridge = sample_residual_bridge(learned_prior.detach(), action_gt, self.bridge)
        output, _ = self._velocity(
            state_tokens=state_tokens,
            past_actions=past_actions,
            physical_prior=physical_prior,
            residual_state=bridge.residual_state,
            timesteps=bridge.time,
            dense_tokens=dense_tokens,
            kv_cache=kv_cache,
            attention_mask=attention_mask,
            stop_after="full",
            learned_prior=learned_prior,
            history_context=history_context,
        )
        if output.full is None:
            raise AssertionError("full output missing")
        weights = self.position_weights.to(device=action_gt.device, dtype=action_gt.dtype)
        prior_loss = _weighted_mse(learned_prior, action_gt, weights)
        full_flow = _weighted_mse(output.full, bridge.target_velocity, weights)
        first_flow = F.mse_loss(output.fast_first, bridge.target_velocity[:, :1])
        prefix_flow = _weighted_mse(
            output.prefix,
            bridge.target_velocity[:, : self.config.prefix_length],
            weights[: self.config.prefix_length],
        )
        total = (
            self.config.prior_loss_weight * prior_loss
            + self.config.fast_exit_loss_weight * first_flow
            + self.config.prefix_exit_loss_weight * prefix_flow
            + self.config.full_flow_loss_weight * full_flow
        )
        return {
            "loss": total,
            "prior_mse": prior_loss.detach(),
            "full_flow_mse": full_flow.detach(),
            "fast_first_flow_mse": first_flow.detach(),
            "prefix_flow_mse": prefix_flow.detach(),
            "visual_gate_mean": output.visual_gate_mean.detach(),
            "visual_correction_rms": output.visual_correction_rms.detach(),
            "target_residual_rms": bridge.target_residual.detach().square().mean().sqrt(),
            "source_residual_rms": bridge.source_residual.detach().square().mean().sqrt(),
        }

    @torch.no_grad()
    def _integrate(
        self,
        *,
        state_tokens: Tensor,
        past_actions: Tensor,
        physical_prior: Tensor,
        dense_tokens: Tensor | None,
        kv_cache: list[tuple[Tensor, Tensor]] | None,
        attention_mask: Tensor | None,
        steps: int,
        mode: str,
    ) -> tuple[Tensor, Tensor]:
        if steps <= 0:
            raise ValueError("steps must be positive")
        learned_prior, history_context = self.predict_prior(
            state_tokens=state_tokens, past_actions=past_actions, physical_prior=physical_prior
        )
        residual = torch.zeros_like(learned_prior)
        dt = 1.0 / steps
        time = torch.zeros(
            (state_tokens.shape[0],), device=state_tokens.device, dtype=state_tokens.dtype
        )
        if mode == "fast":
            length = 1
        elif mode == "prefix":
            length = self.config.prefix_length
        elif mode == "full":
            length = self.config.prediction_horizon
        else:
            raise ValueError(f"unknown integration mode={mode!r}")
        for _ in range(steps):
            output, _ = self._velocity(
                state_tokens=state_tokens,
                past_actions=past_actions,
                physical_prior=physical_prior,
                residual_state=residual,
                timesteps=time,
                dense_tokens=dense_tokens,
                kv_cache=kv_cache,
                attention_mask=attention_mask,
                stop_after=mode,
                learned_prior=learned_prior,
                history_context=history_context,
            )
            velocity = (
                output.fast_first
                if mode == "fast"
                else output.prefix
                if mode == "prefix"
                else output.full
            )
            if velocity is None:
                raise AssertionError("requested progressive velocity missing")
            residual[:, :length] = residual[:, :length] + velocity * dt
            time = time + dt
        return learned_prior + residual, learned_prior

    @torch.no_grad()
    def predict_first_action(self, **kwargs: Any) -> Tensor:
        chunk, _ = self._integrate(mode="fast", **kwargs)
        return chunk[:, 0]

    @torch.no_grad()
    def predict_prefix_action(self, **kwargs: Any) -> Tensor:
        chunk, _ = self._integrate(mode="prefix", **kwargs)
        return chunk[:, : self.config.prefix_length]

    @torch.no_grad()
    def predict_action(self, **kwargs: Any) -> Tensor:
        chunk, _ = self._integrate(mode="full", **kwargs)
        return chunk

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def config_dict(self) -> dict[str, Any]:
        return asdict(self.config)

    @staticmethod
    def _resolve_state_dict(source: str | Path | dict[str, Tensor]) -> dict[str, Tensor]:
        if isinstance(source, (str, Path)):
            payload = torch.load(source, map_location="cpu", weights_only=False)
        else:
            payload = source
        if (
            isinstance(payload, dict)
            and "module" in payload
            and isinstance(payload["module"], dict)
        ):
            payload = payload["module"]
        if isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
            payload = payload["model"]
        if not isinstance(payload, dict):
            raise TypeError("checkpoint must resolve to a state_dict")
        return {str(key).removeprefix("module."): value for key, value in payload.items()}

    def load_compatible_reference_state_dict(
        self, source: str | Path | dict[str, Tensor]
    ) -> dict[str, Any]:
        """Reuse only tensors whose names and shapes remain meaningful.

        Attention, FFN, timestep, positional, adaptor and final-head tensors can
        transfer.  Per-block AdaLN tensors deliberately do not transfer because
        the progressive model replaces them with stage-shared low-rank banks.
        """
        source_state = self._resolve_state_dict(source)
        target_state = self.state_dict()
        matched: dict[str, Tensor] = {}
        skipped_shape: dict[str, dict[str, list[int]]] = {}
        unexpected: list[str] = []
        for key, value in source_state.items():
            if key not in target_state:
                unexpected.append(key)
                continue
            if tuple(value.shape) != tuple(target_state[key].shape):
                skipped_shape[key] = {
                    "source": list(value.shape),
                    "target": list(target_state[key].shape),
                }
                continue
            matched[key] = value
        self.load_state_dict(matched, strict=False)
        return {
            "matched_tensors": len(matched),
            "source_tensors": len(source_state),
            "target_tensors": len(target_state),
            "missing_target_keys": sorted(set(target_state) - set(matched)),
            "unexpected_source_keys": sorted(unexpected),
            "shape_mismatches": skipped_shape,
        }


__all__ = ["ProgressiveRDT2FM", "ProgressiveRDT2FMConfig", "HistoryTrajectoryPrior"]
