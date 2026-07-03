from __future__ import annotations

"""Control-relevant condition interface for RDT2-FM.

This experiment keeps the released-style RDT2-FM motor core intact: the same
14 RDT blocks, the same per-block AdaLN modulation, the same action/state
adaptors, the same flow-matching target, and the same sampling loop.  Only the
condition path changes.

The condition path is split by refresh rate:

* ``SceneTaskCompiler`` runs once per observation.  It preserves camera-slot
  identity, resamples each view independently, and fuses task-conditioned scene
  memory tokens.
* ``ControlReadout`` runs either once (``static`` ablation) or once per flow
  step (``dynamic`` main variant).  In dynamic mode, noisy action tokens,
  current state, and the flow timestep decide which scene evidence is exposed
  to the unchanged motor core.

The static and dynamic variants share one implementation so their difference
is explicit and testable.  This file deliberately does not add history input,
first-action supervision, shallow-tail generation, MoE routing, or a learned
trajectory prior.
"""

from dataclasses import asdict, dataclass
import math
import re
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import LogisticNormal

from .rdt2_fm_reference import Attention, FeedForward, FinalLayer, RDTBlock, RMSNorm, TimestepEmbedder, get_multimodal_pos_embed


def _adapter(kind: str, in_features: int, out_features: int) -> nn.Module:
    if kind == "linear":
        return nn.Linear(in_features, out_features)
    match = re.fullmatch(r"mlp(\d+)x_silu", kind)
    if match is None:
        raise ValueError(f"unsupported adaptor kind: {kind!r}")
    depth = int(match.group(1))
    if depth <= 0:
        raise ValueError("MLP adaptor depth must be positive")
    layers: list[nn.Module] = [nn.Linear(in_features, out_features)]
    for _ in range(1, depth):
        layers.extend([nn.SiLU(), nn.Linear(out_features, out_features)])
    return nn.Sequential(*layers)


def _entropy(probabilities: Tensor, *, dim: int = -1) -> Tensor:
    values = probabilities.float().clamp_min(1e-12)
    return -(values * values.log()).sum(dim=dim)


def _masked_mean(values: Tensor, mask: Tensor, *, dim: int) -> Tensor:
    weights = mask.to(dtype=values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    denom = weights.sum(dim=dim).clamp_min(1.0)
    return (values * weights).sum(dim=dim) / denom


@dataclass(frozen=True)
class ControlInterfaceRDT2FMConfig:
    action_dim: int = 7
    state_dim: int = 7
    prediction_horizon: int = 24

    # Keep the motor core aligned with the reference path.
    hidden_size: int = 1024
    depth: int = 14
    num_heads: int = 8
    num_kv_heads: int = 4
    num_register_tokens: int = 4
    norm_eps: float = 1e-5
    multiple_of: int = 256
    ffn_dim_multiplier: float | None = None
    use_flash_attn: bool = True
    num_inference_timesteps: int = 5

    # Dense frozen visual tokens, currently DINOv2 patch tokens.
    dense_token_dim: int = 768
    visual_adaptor: str = "linear"
    camera_count: int = 2
    time_slots: int = 1

    # The interface is deliberately narrower than the 1024-wide motor core.
    # Its job is evidence organization, not another full action transformer.
    interface_hidden_size: int = 512
    interface_num_heads: int = 8
    interface_num_kv_heads: int = 4
    interface_multiple_of: int = 128

    # Stage A: per-slot resampling and scene-task fusion.
    slot_tokens: int = 4
    slot_resampler_depth: int = 1
    scene_tokens: int = 16
    scene_fusion_depth: int = 2
    default_task_tokens: int = 2

    # Stage B: static or action-aware control readout.
    interface_mode: str = "dynamic"  # static | dynamic
    action_summary_tokens: int = 4
    action_summary_depth: int = 1
    control_tokens: int = 8
    control_readout_depth: int = 2

    def validate(self) -> None:
        positive = (
            self.action_dim,
            self.state_dim,
            self.prediction_horizon,
            self.hidden_size,
            self.depth,
            self.num_heads,
            self.num_kv_heads,
            self.num_register_tokens,
            self.dense_token_dim,
            self.camera_count,
            self.time_slots,
            self.interface_hidden_size,
            self.interface_num_heads,
            self.interface_num_kv_heads,
            self.interface_multiple_of,
            self.slot_tokens,
            self.slot_resampler_depth,
            self.scene_tokens,
            self.scene_fusion_depth,
            self.default_task_tokens,
            self.action_summary_tokens,
            self.action_summary_depth,
            self.control_tokens,
            self.control_readout_depth,
            self.num_inference_timesteps,
        )
        if min(positive) <= 0:
            raise ValueError("all control-interface dimensions must be positive")
        if self.interface_mode not in {"static", "dynamic"}:
            raise ValueError("interface_mode must be 'static' or 'dynamic'")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if self.interface_hidden_size % self.interface_num_heads != 0:
            raise ValueError("interface_hidden_size must be divisible by interface_num_heads")
        if self.interface_num_heads % self.interface_num_kv_heads != 0:
            raise ValueError("interface_num_heads must be divisible by interface_num_kv_heads")
        if self.time_slots != 1:
            raise ValueError("the controlled RDT2-FM dataset path currently exposes one synchronized image timestep")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


class InspectableCrossAttention(nn.Module):
    """Cross attention with optional averaged attention maps for diagnostics."""

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

    def forward(
        self,
        query: Tensor,
        memory: Tensor,
        *,
        memory_mask: Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        key_padding_mask = None if memory_mask is None else memory_mask.logical_not()
        out, weights = self.attn(
            query,
            memory,
            memory,
            key_padding_mask=key_padding_mask,
            need_weights=return_attention,
            average_attn_weights=True,
        )
        return out, weights if return_attention else None


class QueryReadoutBlock(nn.Module):
    """Pre-norm latent-query block used only in the condition interface."""

    def __init__(self, config: ControlInterfaceRDT2FMConfig, *, depth: int) -> None:
        super().__init__()
        width = config.interface_hidden_size
        core = {
            "hidden_size": width,
            "num_heads": config.interface_num_heads,
            "num_kv_heads": config.interface_num_kv_heads,
            "norm_eps": config.norm_eps,
            "multiple_of": config.interface_multiple_of,
            "ffn_dim_multiplier": config.ffn_dim_multiplier,
            "use_flash_attn": config.use_flash_attn,
        }
        self.self_norm = RMSNorm(width, eps=config.norm_eps)
        self.self_attn = Attention(core)
        self.cross_norm = RMSNorm(width, eps=config.norm_eps)
        self.memory_norm = RMSNorm(width, eps=config.norm_eps)
        self.cross_attn = InspectableCrossAttention(width, config.interface_num_heads)
        self.ffn_norm = RMSNorm(width, eps=config.norm_eps)
        self.ffn = FeedForward(width, 4 * width, config.interface_multiple_of, config.ffn_dim_multiplier)
        self.scale = 1.0 / math.sqrt(max(depth, 1))

    def forward(
        self,
        query: Tensor,
        memory: Tensor,
        *,
        memory_mask: Tensor | None,
        return_attention: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        query = query + self.scale * self.self_attn(self.self_norm(query))
        cross, weights = self.cross_attn(
            self.cross_norm(query),
            self.memory_norm(memory),
            memory_mask=memory_mask,
            return_attention=return_attention,
        )
        query = query + self.scale * cross
        query = query + self.scale * self.ffn(self.ffn_norm(query))
        return query, weights


@dataclass
class SceneTaskMemory:
    tokens: Tensor
    mask: Tensor
    # Final scene-query attention over [task tokens, camera-slot tokens].
    source_attention: Tensor | None = None
    source_group_ids: Tensor | None = None
    source_group_names: tuple[str, ...] = ()
    diagnostics: dict[str, Any] | None = None


@dataclass
class ControlMemory:
    tokens: Tensor
    diagnostics: dict[str, Any] | None = None


class SceneTaskCompiler(nn.Module):
    """Compile dense per-camera patches into reusable scene-task memory."""

    def __init__(self, config: ControlInterfaceRDT2FMConfig) -> None:
        super().__init__()
        self.config = config
        h = config.interface_hidden_size
        self.visual_in = _adapter(config.visual_adaptor, config.dense_token_dim, h)
        self.camera_embed = nn.Parameter(torch.randn(config.camera_count, h) * 0.02)
        self.time_embed = nn.Parameter(torch.randn(config.time_slots, h) * 0.02)
        self.missing_embed = nn.Parameter(torch.randn(config.camera_count, config.slot_tokens, h) * 0.02)
        self.slot_queries = nn.Parameter(torch.randn(config.camera_count, config.slot_tokens, h) * 0.02)
        self.slot_blocks = nn.ModuleList([
            QueryReadoutBlock(config, depth=config.slot_resampler_depth)
            for _ in range(config.slot_resampler_depth)
        ])
        self.default_task = nn.Parameter(torch.randn(1, config.default_task_tokens, h) * 0.02)
        self.scene_queries = nn.Parameter(torch.randn(1, config.scene_tokens, h) * 0.02)
        self.scene_blocks = nn.ModuleList([
            QueryReadoutBlock(config, depth=config.scene_fusion_depth)
            for _ in range(config.scene_fusion_depth)
        ])
        self.out_norm = RMSNorm(h, eps=config.norm_eps)

    def _reshape_dense(self, dense_tokens: Tensor, attention_mask: Tensor | None) -> tuple[Tensor, Tensor]:
        cfg = self.config
        if dense_tokens.ndim != 3 or dense_tokens.shape[-1] != cfg.dense_token_dim:
            raise ValueError(f"dense_tokens must be [B,L,{cfg.dense_token_dim}], got {tuple(dense_tokens.shape)}")
        batch, length, _ = dense_tokens.shape
        slots = cfg.camera_count * cfg.time_slots
        if length % slots != 0:
            raise ValueError(f"dense token length={length} is not divisible by camera_count*time_slots={slots}")
        patches = length // slots
        dense = dense_tokens.reshape(batch, slots, patches, cfg.dense_token_dim)
        if attention_mask is None:
            mask = torch.ones((batch, slots, patches), dtype=torch.bool, device=dense_tokens.device)
        else:
            if attention_mask.shape != dense_tokens.shape[:2]:
                raise ValueError("attention_mask must match dense token batch and length")
            mask = attention_mask.to(dtype=torch.bool).reshape(batch, slots, patches)
        return dense, mask

    def forward(
        self,
        *,
        dense_tokens: Tensor,
        attention_mask: Tensor | None,
        task_tokens: Tensor | None = None,
        camera_valid_mask: Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> SceneTaskMemory:
        cfg = self.config
        dense, patch_mask = self._reshape_dense(dense_tokens, attention_mask)
        batch, slots, patches, _ = dense.shape
        if slots != cfg.camera_count:
            # time_slots is intentionally fixed to one in the controlled branch.
            raise AssertionError("unexpected slot count")
        if camera_valid_mask is None:
            camera_valid = torch.ones((batch, cfg.camera_count), dtype=torch.bool, device=dense.device)
        else:
            if camera_valid_mask.shape != (batch, cfg.camera_count):
                raise ValueError(f"camera_valid_mask must be [B,{cfg.camera_count}], got {tuple(camera_valid_mask.shape)}")
            camera_valid = camera_valid_mask.to(dtype=torch.bool)
        patch_mask = patch_mask & camera_valid.unsqueeze(-1)

        visual = self.visual_in(dense)
        visual = visual + self.camera_embed.view(1, cfg.camera_count, 1, -1)
        visual = visual + self.time_embed[0].view(1, 1, 1, -1)
        flat_visual = visual.reshape(batch * cfg.camera_count, patches, cfg.interface_hidden_size)
        flat_mask = patch_mask.reshape(batch * cfg.camera_count, patches)
        # Avoid all-masked softmax rows. Missing slots are replaced explicitly below.
        safe_mask = flat_mask.clone()
        missing_rows = safe_mask.sum(dim=1) == 0
        if missing_rows.any():
            safe_mask[missing_rows, 0] = True
            flat_visual = flat_visual.clone()
            flat_visual[missing_rows, 0] = 0

        query = self.slot_queries.unsqueeze(0).expand(batch, -1, -1, -1)
        query = query + self.camera_embed.view(1, cfg.camera_count, 1, -1)
        query = query + self.time_embed[0].view(1, 1, 1, -1)
        query = query.reshape(batch * cfg.camera_count, cfg.slot_tokens, cfg.interface_hidden_size)
        slot_attention = None
        for block in self.slot_blocks:
            query, slot_attention = block(query, flat_visual, memory_mask=safe_mask, return_attention=return_diagnostics)
        slot_tokens = query.reshape(batch, cfg.camera_count, cfg.slot_tokens, cfg.interface_hidden_size)
        slot_tokens = torch.where(
            camera_valid.view(batch, cfg.camera_count, 1, 1),
            slot_tokens,
            self.missing_embed.unsqueeze(0).expand(batch, -1, -1, -1),
        )

        if task_tokens is None:
            task = self.default_task.expand(batch, -1, -1)
        else:
            if task_tokens.ndim != 3 or task_tokens.shape[0] != batch or task_tokens.shape[-1] != cfg.interface_hidden_size:
                raise ValueError(f"task_tokens must be [B,T,{cfg.interface_hidden_size}], got {tuple(task_tokens.shape)}")
            task = task_tokens
        flattened_slots = slot_tokens.reshape(batch, cfg.camera_count * cfg.slot_tokens, cfg.interface_hidden_size)
        fusion_memory = torch.cat([task, flattened_slots], dim=1)
        # Keep every camera slot visible to scene fusion.  Missing cameras are
        # represented by learned missing tokens rather than silently removed, so
        # downstream queries can distinguish "no observation" from a real blank view.
        fusion_mask = torch.ones(
            (batch, task.shape[1] + cfg.camera_count * cfg.slot_tokens),
            dtype=torch.bool,
            device=task.device,
        )
        scene = self.scene_queries.expand(batch, -1, -1)
        scene_attention = None
        for block in self.scene_blocks:
            scene, scene_attention = block(scene, fusion_memory, memory_mask=fusion_mask, return_attention=return_diagnostics)
        scene = self.out_norm(scene)

        task_groups = torch.zeros((task.shape[1],), dtype=torch.long, device=scene.device)
        camera_groups = torch.arange(1, cfg.camera_count + 1, device=scene.device).repeat_interleave(cfg.slot_tokens)
        group_ids = torch.cat([task_groups, camera_groups], dim=0)
        group_names = ("task", *tuple(f"camera_{index}" for index in range(cfg.camera_count)))
        diagnostics = None
        if return_diagnostics:
            if scene_attention is None:
                raise AssertionError("scene attention was not returned")
            source_mass = []
            for group in range(len(group_names)):
                source_mass.append(scene_attention[..., group_ids == group].sum(dim=-1).mean(dim=1))
            diagnostics = {
                "scene_attention_entropy": _entropy(scene_attention).mean(dim=1),
                "scene_source_mass": torch.stack(source_mass, dim=1),
                "camera_valid_fraction": camera_valid.float(),
            }
            if slot_attention is not None:
                diagnostics["slot_attention_entropy"] = _entropy(slot_attention).mean(dim=1).reshape(batch, cfg.camera_count)
        return SceneTaskMemory(
            tokens=scene,
            mask=torch.ones(scene.shape[:2], dtype=torch.bool, device=scene.device),
            source_attention=scene_attention,
            source_group_ids=group_ids,
            source_group_names=group_names,
            diagnostics=diagnostics,
        )


class ActionSummaryEncoder(nn.Module):
    """Summarize the current noisy trajectory hypothesis for dynamic readout."""

    def __init__(self, config: ControlInterfaceRDT2FMConfig, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.config = config
        width = config.interface_hidden_size
        self.action_in = _adapter("mlp2x_silu", config.action_dim, width)
        self.action_pos = nn.Parameter(torch.randn(1, config.prediction_horizon, width) * 0.02)
        self.summary_queries = nn.Parameter(torch.randn(1, config.action_summary_tokens, width) * 0.02)
        self.blocks = nn.ModuleList([
            QueryReadoutBlock(config, depth=config.action_summary_depth)
            for _ in range(config.action_summary_depth)
        ])
        self.out_norm = RMSNorm(width, eps=config.norm_eps)
        self.to(dtype=dtype)

    def forward(self, noisy_action: Tensor, *, return_diagnostics: bool = False) -> tuple[Tensor, dict[str, Tensor] | None]:
        cfg = self.config
        if noisy_action.ndim != 3 or tuple(noisy_action.shape[1:]) != (cfg.prediction_horizon, cfg.action_dim):
            raise ValueError(f"noisy_action must be [B,{cfg.prediction_horizon},{cfg.action_dim}], got {tuple(noisy_action.shape)}")
        memory = self.action_in(noisy_action) + self.action_pos
        query = self.summary_queries.expand(noisy_action.shape[0], -1, -1)
        attention = None
        for block in self.blocks:
            query, attention = block(query, memory, memory_mask=None, return_attention=return_diagnostics)
        query = self.out_norm(query)
        diagnostics = None
        if return_diagnostics and attention is not None:
            diagnostics = {"action_summary_entropy": _entropy(attention).mean(dim=1)}
        return query, diagnostics


class ControlReadout(nn.Module):
    """Expose scene evidence to the motor core, statically or per flow step."""

    def __init__(self, config: ControlInterfaceRDT2FMConfig, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.config = config
        h = config.interface_hidden_size
        self.control_queries = nn.Parameter(torch.randn(1, config.control_tokens, h) * 0.02)
        self.state_in = _adapter("mlp2x_silu", config.state_dim, h)
        self.time = TimestepEmbedder(h, dtype=dtype)
        self.summary = ActionSummaryEncoder(config, dtype=dtype)
        self.dynamic_bias = nn.Linear(3 * h, h)
        self.blocks = nn.ModuleList([
            QueryReadoutBlock(config, depth=config.control_readout_depth)
            for _ in range(config.control_readout_depth)
        ])
        self.out_norm = RMSNorm(h, eps=config.norm_eps)
        self.to_motor = nn.Linear(h, config.hidden_size)
        self.to(dtype=dtype)

    def forward(
        self,
        *,
        scene: SceneTaskMemory,
        state_tokens: Tensor,
        noisy_action: Tensor | None,
        timesteps: Tensor | None,
        return_diagnostics: bool = False,
    ) -> ControlMemory:
        cfg = self.config
        if state_tokens.ndim != 2 or state_tokens.shape[-1] != cfg.state_dim:
            raise ValueError(f"state_tokens must be [B,{cfg.state_dim}], got {tuple(state_tokens.shape)}")
        batch = state_tokens.shape[0]
        query = self.control_queries.expand(batch, -1, -1)
        summary_diag = None
        if cfg.interface_mode == "dynamic":
            if noisy_action is None or timesteps is None:
                raise ValueError("dynamic control readout requires noisy_action and timesteps")
            summary, summary_diag = self.summary(noisy_action, return_diagnostics=return_diagnostics)
            time = self.time(timesteps)
            if time.shape[0] == 1:
                time = time.expand(batch, -1)
            state = self.state_in(state_tokens)
            bias = self.dynamic_bias(torch.cat([summary.mean(dim=1), state, time], dim=-1))
            query = query + bias.unsqueeze(1)
        attention = None
        for block in self.blocks:
            query, attention = block(query, scene.tokens, memory_mask=scene.mask, return_attention=return_diagnostics)
        query = self.out_norm(query)
        diagnostics = None
        if return_diagnostics:
            if attention is None:
                raise AssertionError("control attention was not returned")
            diagnostics = {
                "control_attention_entropy": _entropy(attention).mean(dim=1),
            }
            if summary_diag is not None:
                diagnostics.update(summary_diag)
            if scene.source_attention is not None and scene.source_group_ids is not None:
                effective = torch.bmm(attention, scene.source_attention)
                masses = []
                for group in range(len(scene.source_group_names)):
                    masses.append(effective[..., scene.source_group_ids == group].sum(dim=-1).mean(dim=1))
                diagnostics["effective_source_mass"] = torch.stack(masses, dim=1)
        return ControlMemory(tokens=self.to_motor(query), diagnostics=diagnostics)


class ControlConditionRDT(nn.Module):
    """Reference-shaped RDT motor core consuming compact control memory."""

    def __init__(self, config: ControlInterfaceRDT2FMConfig, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.config = config
        rdt_cfg: dict[str, Any] = {
            "hidden_size": config.hidden_size,
            "depth": config.depth,
            "num_heads": config.num_heads,
            "num_kv_heads": config.num_kv_heads,
            "num_register_tokens": config.num_register_tokens,
            "norm_eps": config.norm_eps,
            "multiple_of": config.multiple_of,
            "ffn_dim_multiplier": config.ffn_dim_multiplier,
            "use_flash_attn": config.use_flash_attn,
        }
        self.hidden_size = config.hidden_size
        self.num_register_tokens = config.num_register_tokens
        self.t_embedder = TimestepEmbedder(config.hidden_size, dtype=dtype)
        self.blocks = nn.ModuleList([RDTBlock(index, rdt_cfg) for index in range(config.depth)])
        self.final_layer = FinalLayer(config.action_dim, rdt_cfg)
        self.register_tokens = nn.Parameter(torch.randn(1, config.num_register_tokens, config.hidden_size))
        x_pos = get_multimodal_pos_embed(config.hidden_size, {"action": config.prediction_horizon, "register": config.num_register_tokens})
        state_pos = get_multimodal_pos_embed(config.hidden_size, {"state": 1})
        self.x_pos_emb = nn.Parameter(torch.from_numpy(x_pos).float().unsqueeze(0))
        self.state_pos_emb = nn.Parameter(torch.from_numpy(state_pos).float().unsqueeze(0))
        self._initialize(dtype)

    def _initialize(self, dtype: torch.dtype) -> None:
        def basic(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.apply(basic)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.ffn.fc2.weight)
        nn.init.zeros_(self.final_layer.ffn.fc2.bias)
        self.to(dtype=dtype)

    def forward(self, *, action_tokens: Tensor, timesteps: Tensor, state_tokens: Tensor, control_tokens: Tensor) -> Tensor:
        if state_tokens.ndim != 3 or tuple(state_tokens.shape[1:]) != (1, self.config.hidden_size):
            raise ValueError(f"adapted state_tokens must be [B,1,{self.config.hidden_size}], got {tuple(state_tokens.shape)}")
        time = self.t_embedder(timesteps)
        if time.shape[0] == 1:
            time = time.expand(action_tokens.shape[0], -1)
        state = state_tokens + self.state_pos_emb
        modulation = torch.cat([time.unsqueeze(1), state], dim=1).reshape(action_tokens.shape[0], 2 * self.hidden_size)
        registers = self.register_tokens.expand(action_tokens.shape[0], -1, -1)
        x = torch.cat([action_tokens, registers], dim=1) + self.x_pos_emb
        for block in self.blocks:
            x = block(x, modulation, c=control_tokens)
        return self.final_layer(x, modulation)[:, :-self.num_register_tokens]


class RDT2ControlInterface(nn.Module):
    """Reference motor core with a two-stage control-relevant condition interface."""

    def __init__(self, config: ControlInterfaceRDT2FMConfig = ControlInterfaceRDT2FMConfig(), *, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.scene = SceneTaskCompiler(config)
        self.control = ControlReadout(config, dtype=dtype)
        self.motor = ControlConditionRDT(config, dtype=dtype)
        self.action_adaptor = _adapter("mlp3x_silu", config.action_dim, config.hidden_size)
        self.state_adaptor = _adapter("mlp3x_silu", config.state_dim, config.hidden_size)
        self.num_inference_timesteps = config.num_inference_timesteps
        self.pred_horizon = config.prediction_horizon
        self.action_dim = config.action_dim
        self.to(dtype=dtype)

    def sample_timesteps(self, batch_size: int, device: torch.device) -> Tensor:
        distribution = LogisticNormal(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))
        return distribution.sample((batch_size,))[:, 0]

    def prepare_scene(
        self,
        *,
        dense_tokens: Tensor,
        attention_mask: Tensor | None,
        task_tokens: Tensor | None = None,
        camera_valid_mask: Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> SceneTaskMemory:
        return self.scene(
            dense_tokens=dense_tokens,
            attention_mask=attention_mask,
            task_tokens=task_tokens,
            camera_valid_mask=camera_valid_mask,
            return_diagnostics=return_diagnostics,
        )

    def predict_velocity(
        self,
        *,
        state_tokens: Tensor,
        noisy_action: Tensor,
        timesteps: Tensor,
        scene: SceneTaskMemory,
        prepared_control: ControlMemory | None = None,
        return_diagnostics: bool = False,
    ) -> tuple[Tensor, dict[str, Any] | None]:
        if state_tokens.ndim != 2:
            raise ValueError(f"state_tokens must be [B,D], got {tuple(state_tokens.shape)}")
        control = prepared_control
        if control is None:
            control = self.control(
                scene=scene,
                state_tokens=state_tokens,
                noisy_action=noisy_action if self.config.interface_mode == "dynamic" else None,
                timesteps=timesteps if self.config.interface_mode == "dynamic" else None,
                return_diagnostics=return_diagnostics,
            )
        action = self.action_adaptor(noisy_action)
        state = self.state_adaptor(state_tokens).unsqueeze(1)
        velocity = self.motor(action_tokens=action, timesteps=timesteps, state_tokens=state, control_tokens=control.tokens)
        diagnostics = None
        if return_diagnostics:
            diagnostics = {}
            if scene.diagnostics is not None:
                diagnostics.update(scene.diagnostics)
            if control.diagnostics is not None:
                diagnostics.update(control.diagnostics)
            diagnostics["control_rms"] = control.tokens.square().mean(dim=(1, 2)).sqrt()
            diagnostics["scene_rms"] = scene.tokens.square().mean(dim=(1, 2)).sqrt()
        return velocity, diagnostics

    def compute_loss(
        self,
        *,
        state_tokens: Tensor,
        action_gt: Tensor,
        dense_tokens: Tensor,
        attention_mask: Tensor | None = None,
        task_tokens: Tensor | None = None,
        camera_valid_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        batch = action_gt.shape[0]
        noise = torch.randn_like(action_gt)
        timesteps = self.sample_timesteps(batch, action_gt.device).to(dtype=action_gt.dtype)
        noisy = action_gt * timesteps.view(-1, 1, 1) + noise * (1 - timesteps.view(-1, 1, 1))
        scene = self.prepare_scene(
            dense_tokens=dense_tokens,
            attention_mask=attention_mask,
            task_tokens=task_tokens,
            camera_valid_mask=camera_valid_mask,
        )
        control = self.control(
            scene=scene,
            state_tokens=state_tokens,
            noisy_action=noisy if self.config.interface_mode == "dynamic" else None,
            timesteps=timesteps if self.config.interface_mode == "dynamic" else None,
            return_diagnostics=False,
        )
        action = self.action_adaptor(noisy)
        state = self.state_adaptor(state_tokens).unsqueeze(1)
        velocity = self.motor(action_tokens=action, timesteps=timesteps, state_tokens=state, control_tokens=control.tokens)
        loss = F.mse_loss(velocity, action_gt - noise)
        return {
            "loss": loss,
            "flow_mse": loss.detach(),
            "scene_rms": scene.tokens.square().mean().sqrt().detach(),
            "control_rms": control.tokens.square().mean().sqrt().detach(),
        }

    @torch.no_grad()
    def predict_action(
        self,
        *,
        state_tokens: Tensor,
        dense_tokens: Tensor,
        attention_mask: Tensor | None = None,
        task_tokens: Tensor | None = None,
        camera_valid_mask: Tensor | None = None,
        noisy_action: Tensor | None = None,
        generator: torch.Generator | None = None,
        inference_steps: int | None = None,
        return_diagnostics: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Any]]:
        batch = state_tokens.shape[0]
        if noisy_action is None:
            noisy_action = torch.randn((batch, self.pred_horizon, self.action_dim), device=state_tokens.device, dtype=state_tokens.dtype, generator=generator)
        steps = int(inference_steps or self.num_inference_timesteps)
        if steps <= 0:
            raise ValueError("inference_steps must be positive")
        scene = self.prepare_scene(
            dense_tokens=dense_tokens,
            attention_mask=attention_mask,
            task_tokens=task_tokens,
            camera_valid_mask=camera_valid_mask,
            return_diagnostics=return_diagnostics,
        )
        dt = 1.0 / steps
        time = torch.tensor([0.0], device=state_tokens.device, dtype=state_tokens.dtype)
        # Static mode is a true one-observation readout ablation: compile scene and
        # control memory once, then reuse it throughout flow integration.
        static_control = None
        if self.config.interface_mode == "static":
            static_control = self.control(
                scene=scene,
                state_tokens=state_tokens,
                noisy_action=None,
                timesteps=None,
                return_diagnostics=return_diagnostics,
            )
        step_diagnostics: list[dict[str, Any]] = []
        for _ in range(steps):
            velocity, diagnostics = self.predict_velocity(
                state_tokens=state_tokens,
                noisy_action=noisy_action,
                timesteps=time,
                scene=scene,
                prepared_control=static_control,
                return_diagnostics=return_diagnostics,
            )
            noisy_action = noisy_action + velocity * dt
            if diagnostics is not None:
                step_diagnostics.append(diagnostics)
            time = time + dt
        if not return_diagnostics:
            return noisy_action
        return noisy_action, {
            "source_group_names": scene.source_group_names,
            "flow_steps": step_diagnostics,
        }

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def parameter_groups(self) -> dict[str, int]:
        return {
            "scene_compiler": sum(p.numel() for p in self.scene.parameters()),
            "control_readout": sum(p.numel() for p in self.control.parameters()),
            "motor_core": sum(p.numel() for p in self.motor.parameters()),
            "action_adaptor": sum(p.numel() for p in self.action_adaptor.parameters()),
            "state_adaptor": sum(p.numel() for p in self.state_adaptor.parameters()),
        }

    def config_dict(self) -> dict[str, Any]:
        return asdict(self.config)


__all__ = [
    "ActionSummaryEncoder",
    "ControlInterfaceRDT2FMConfig",
    "ControlMemory",
    "ControlReadout",
    "RDT2ControlInterface",
    "SceneTaskCompiler",
    "SceneTaskMemory",
]
