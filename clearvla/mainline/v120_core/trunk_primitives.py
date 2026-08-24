"""Shared world/action trunk components used by V38 and the current policy."""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .codec import PhysicalActionTokenLift
from .primitives import BiasFreeFFN, sinusoidal_positions
from .role_delta_attnres import (
    smooth_absolute_contract,
    smooth_rms_contract,
    variance_floored_centered_norm,
)


class TrunkPrimitiveConfig(Protocol):
    action_basis_tokens: int
    action_dim: int
    action_horizon: int
    action_history_enabled: int
    action_history_token_count: int
    arm_dim: int
    base_effect_hidden: int
    canvas_dropout: float
    canvas_registers: int
    controlled_base_mode: str
    controlled_delta_dropout: float
    controlled_delta_rank: int
    dropout: float
    executed_history_length: int
    ffn_expansion: float
    first_execution_steps: int
    future_anchors: int
    future_grid_size: int
    flow_jepa_complete_numerical_contract: int
    flow_jepa_routing_norm_floor: float
    gripper_field_dim: int
    gripper_field_mode: str
    hidden_size: int
    goal_conditioning_enabled: int
    goal_token_count: int
    latent_action_tokens: int
    mid_execution_steps: int
    neutral_action_tokens: int
    num_cameras: int
    num_heads: int
    patches_per_camera: int
    physical_action_dim: int
    role_dropout: float
    rollout_tail_full_step: int
    rollout_tail_start_step: int
    state_dim: int
    visual_history_length: int
    visual_memory_dropout: float
    visual_token_dim: int
    role_residual_amplitude_contract: int
    role_residual_contract_after_gate: int
    role_residual_max_update_rms: float


class HorizonRoleEmbedding(nn.Module):
    """Explicit execution/planning role embedding for horizon tokens."""

    def __init__(self, config: TrunkPrimitiveConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.execution = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        self.mid = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        self.tail = nn.Parameter(torch.randn(1, 1, h) * 0.02)

    def forward(self, batch: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        h = self.execution.shape[-1]
        role = torch.empty(1, self.config.action_horizon, h, device=device, dtype=dtype)
        role[:, : self.config.first_execution_steps] = self.execution.to(device=device, dtype=dtype)
        role[:, self.config.first_execution_steps : self.config.mid_execution_steps] = self.mid.to(
            device=device, dtype=dtype
        )
        role[:, self.config.mid_execution_steps :] = self.tail.to(device=device, dtype=dtype)
        return role.expand(batch, -1, -1)


class DenseVisualMemory(nn.Module):
    """Per-token DINO projection with factorized identity embeddings."""

    def __init__(self, config: TrunkPrimitiveConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size
        d = config.visual_token_dim
        self.proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, h))
        self.history_type = nn.Parameter(
            torch.randn(1, config.visual_history_length, 1, 1, h) * 0.02
        )
        self.camera_type = nn.Parameter(torch.randn(1, 1, config.num_cameras, 1, h) * 0.02)
        self.patch_type = nn.Parameter(torch.randn(1, 1, 1, config.patches_per_camera, h) * 0.02)
        self.out_norm = nn.LayerNorm(h)
        self.drop = nn.Dropout(config.visual_memory_dropout)

    def forward(self, visual: Tensor) -> Tensor:
        cfg = self.config
        if visual.ndim != 5:
            raise ValueError(f"visual must be [B,H,C,P,D], got {tuple(visual.shape)}")
        b, hist, cams, patches, dim = visual.shape
        expected = (
            cfg.visual_history_length,
            cfg.num_cameras,
            cfg.patches_per_camera,
            cfg.visual_token_dim,
        )
        if (hist, cams, patches, dim) != expected:
            raise ValueError(
                f"V38 visual geometry mismatch: got {(hist, cams, patches, dim)}, expected {expected}"
            )
        x = self.proj(visual)
        x = x + self.history_type.to(device=x.device, dtype=x.dtype)
        x = x + self.camera_type.to(device=x.device, dtype=x.dtype)
        x = x + self.patch_type.to(device=x.device, dtype=x.dtype)
        x = self.out_norm(x)
        return self.drop(x.reshape(b, hist * cams * patches, cfg.hidden_size))


class RolloutTargetCodec(nn.Module):
    """Build rollout init/target tokens from DINO grids.

    The target projection is frozen.  This prevents a learned target projector
    from collapsing the future residual objective into an easy private code.
    """

    def __init__(self, config: TrunkPrimitiveConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size
        d = config.visual_token_dim
        self.init_proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, h))
        self.target_proj = nn.Linear(d, h, bias=False)
        nn.init.orthogonal_(self.target_proj.weight)
        self.target_proj.weight.requires_grad_(False)

    def spatial_pool_tokens(self, tokens: Tensor) -> Tensor:
        """Pool [B,F,C,P,D] or [B,C,P,D] to [B,F,C*G*G,D]."""
        cfg = self.config
        original_ndim = tokens.ndim
        if original_ndim == 4:
            tokens = tokens[:, None]
        if tokens.ndim != 5:
            raise ValueError(f"tokens must be [B,F,C,P,D] or [B,C,P,D], got {tuple(tokens.shape)}")
        b, f, c, p, d = tokens.shape
        side = int(round(float(p) ** 0.5))
        if side * side == p:
            x = tokens.reshape(b * f * c, side, side, d).permute(0, 3, 1, 2).float()
            x = F.adaptive_avg_pool2d(x, (cfg.future_grid_size, cfg.future_grid_size))
            x = x.permute(0, 2, 3, 1).reshape(
                b, f, c * cfg.future_grid_size * cfg.future_grid_size, d
            )
        else:
            g2 = cfg.future_grid_size * cfg.future_grid_size
            idx = torch.linspace(0, p, steps=g2 + 1, device=tokens.device).long()
            pooled = []
            for i in range(g2):
                lo, hi = int(idx[i]), max(int(idx[i + 1]), int(idx[i]) + 1)
                pooled.append(tokens[..., lo:hi, :].float().mean(dim=-2))
            x = torch.stack(pooled, dim=3).reshape(b, f, c * g2, d)
        if original_ndim == 4:
            return x[:, 0]
        return x

    def current_grid(self, visual: Tensor) -> Tensor:
        if visual.ndim != 5:
            raise ValueError(f"visual must be [B,H,C,P,D], got {tuple(visual.shape)}")
        return self.spatial_pool_tokens(visual[:, -1])  # [B,C*G*G,D]

    def rollout_init(self, visual: Tensor) -> Tensor:
        cfg = self.config
        current = self.current_grid(visual)
        current_h = self.init_proj(
            current.to(device=next(self.parameters()).device, dtype=next(self.parameters()).dtype)
        )
        return (
            current_h[:, None]
            .expand(-1, cfg.future_anchors, -1, -1)
            .reshape(visual.shape[0], cfg.future_token_count, cfg.hidden_size)
        )

    @torch.no_grad()
    def target_effect(self, visual: Tensor, target_visual: Tensor) -> Tensor:
        cfg = self.config
        if target_visual.ndim != 6:
            raise ValueError(
                f"target_visual must be [B,F,H,C,P,D], got {tuple(target_visual.shape)}"
            )
        future = target_visual[:, : cfg.future_anchors, -1]
        current = visual[:, -1][:, None].expand(-1, cfg.future_anchors, -1, -1, -1)
        residual = future.float() - current.float()
        pooled = self.spatial_pool_tokens(residual)  # [B,K,C*G*G,D]
        target = self.target_proj(
            pooled.to(device=next(self.parameters()).device, dtype=next(self.parameters()).dtype)
        )
        return target.reshape(visual.shape[0], cfg.future_token_count, cfg.hidden_size).detach()


class UnifiedCanvasSeed(nn.Module):
    """Build the initial canvas with rollout tokens initialized from current vision."""

    ROLE_TASK = 0
    ROLE_STATE = 1
    ROLE_STATE_HISTORY = 2
    ROLE_EXECUTED = 3
    ROLE_PROPOSAL = 4
    ROLE_NOISY_ACTION = 5
    ROLE_ROLLOUT = 6
    ROLE_REGISTER = 7

    def __init__(self, config: TrunkPrimitiveConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.state_proj = nn.Linear(config.state_dim, h)
        self.state_history_proj = nn.Linear(config.state_dim, h)
        self.executed_proj = nn.Linear(config.action_dim, h)
        self.proposal_proj = nn.Identity()
        self.noisy_physical_lift = PhysicalActionTokenLift(config)
        self.horizon_role = HorizonRoleEmbedding(config)
        self.action_basis_embed = nn.Parameter(
            torch.randn(1, 1, config.action_basis_tokens, h) * 0.02
        )
        self.role_embed = nn.Parameter(torch.randn(8, h) * 0.02)
        self.role_drop = nn.Dropout(config.role_dropout)
        self.task_token = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        sibling_conditioning = bool(
            int(getattr(config, "action_history_enabled", 0))
            or int(getattr(config, "goal_conditioning_enabled", 0))
        )
        self.goal_private_condition = (
            nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
            if int(getattr(config, "goal_conditioning_enabled", 0))
            else None
        )
        self.action_private_condition = (
            nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
            if int(getattr(config, "action_history_enabled", 0))
            else None
        )
        self.shared_condition_mixer = (
            nn.Sequential(
                nn.LayerNorm(h),
                nn.Linear(h, 2 * h),
                nn.SiLU(),
                nn.Linear(2 * h, h),
            )
            if sibling_conditioning
            else None
        )
        exact_sibling_nulls = (
            (
                not int(getattr(config, "goal_conditioning_enabled", 0))
                or int(getattr(config, "goal_condition_exact_null", 0))
            )
            and (
                not int(getattr(config, "action_history_enabled", 0))
                or int(getattr(config, "action_history_condition_exact_null", 0))
            )
        )
        if self.shared_condition_mixer is not None and exact_sibling_nulls:
            # Every active sibling condition is represented as
            # f(condition)-f(0). The final affine bias cancels identically in
            # that exact-null difference, so it is an unidentifiable degree
            # of freedom rather than a learnable condition parameter.
            self.shared_condition_mixer[-1].bias.requires_grad_(False)
        self.rollout_anchor_type = nn.Parameter(torch.randn(1, config.future_anchors, 1, h) * 0.02)
        self.rollout_grid_type = nn.Parameter(
            torch.randn(
                1, 1, config.num_cameras * config.future_grid_size * config.future_grid_size, h
            )
            * 0.02
        )
        self.stage_type = (
            nn.Parameter(torch.randn(1, 1, h) * 0.02)
            if int(getattr(config, "flow_jepa_enabled", 0))
            and not int(getattr(config, "flow_jepa_late_bottleneck", 0))
            else None
        )
        self.registers = nn.Parameter(torch.randn(1, config.canvas_registers, h) * 0.02)
        self.proposal_type = nn.Parameter(torch.randn(1, config.action_horizon, h) * 0.02)
        self.executed_type = nn.Parameter(torch.randn(1, config.executed_history_length, h) * 0.02)
        if int(getattr(config, "action_history_enabled", 0)):
            self.executed_proj.requires_grad_(False)
            self.executed_type.requires_grad_(False)
        self.state_history_type = nn.Parameter(
            torch.randn(1, config.visual_history_length, h) * 0.02
        )
        self.drop = nn.Dropout(config.canvas_dropout)
        self.register_buffer(
            "horizon_position",
            sinusoidal_positions(range(1, config.action_horizon + 1), h)[None],
            persistent=True,
        )

    def clean_action_basis_tokens(
        self,
        batch: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Return action-basis identity without the noisy physical trajectory.

        P1 factual addressing may use horizon and basis identity, but it must
        not let a flow-matching sample or a hold/shuffle counterfactual redefine
        the observed scene.  Role dropout is deliberately excluded as well so
        one factual read can be reused within a counterfactual bundle.
        """

        batch = int(batch)
        if batch < 1:
            raise ValueError("clean action basis requires a positive batch")
        horizon = (
            self.horizon_position.to(device=device, dtype=dtype)
            + self.horizon_role(batch, device=device, dtype=dtype)
            + self.role_embed[self.ROLE_NOISY_ACTION].to(
                device=device, dtype=dtype
            ).reshape(1, 1, -1)
        )
        return horizon[:, :, None] + self.action_basis_embed.to(
            device=device, dtype=dtype
        )

    def forward(
        self,
        *,
        noisy_physical: Tensor,
        state: Tensor,
        state_history: Tensor,
        executed_history: Tensor,
        executed_memory: Tensor | None = None,
        proposal_tokens: Tensor,
        proposal_keep: Tensor,
        rollout_init: Tensor,
        stage_init: Tensor | None = None,
        goal_tokens: Tensor | None = None,
        goal_condition_keep: Tensor | None = None,
        action_history_condition_keep: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, slice]]:
        cfg = self.config
        b = noisy_physical.shape[0]
        device = noisy_physical.device
        dtype = noisy_physical.dtype
        role = self.role_drop(self.role_embed.to(device=device, dtype=dtype))
        if int(getattr(cfg, "goal_conditioning_enabled", 0)):
            if goal_tokens is None or self.goal_private_condition is None:
                raise ValueError("goal_tokens are required when goal conditioning is enabled")
            expected_goal = int(getattr(cfg, "goal_token_count", 1))
            if tuple(goal_tokens.shape) != (b, expected_goal, cfg.hidden_size):
                raise ValueError(
                    f"goal_tokens must be [B,{expected_goal},{cfg.hidden_size}], got "
                    f"{tuple(goal_tokens.shape)}"
                )
            if self.shared_condition_mixer is None:
                raise RuntimeError("shared condition mixer is missing")
            goal_content = self.shared_condition_mixer(
                self.goal_private_condition(goal_tokens.to(device=device, dtype=dtype))
            )
            if int(getattr(cfg, "goal_condition_exact_null", 0)):
                if (
                    goal_condition_keep is None
                    or tuple(goal_condition_keep.shape) != (b,)
                ):
                    raise ValueError(
                        "exact goal null semantics require goal_condition_keep [B]"
                    )
                null_goal = self.shared_condition_mixer(
                    self.goal_private_condition(torch.zeros_like(goal_tokens))
                )
                goal_content = (goal_content - null_goal) * goal_condition_keep.to(
                    device=device, dtype=dtype
                )[:, None, None]
            task = (
                goal_content
                + self.task_token.to(device=device, dtype=dtype)
                + role[self.ROLE_TASK]
            )
        else:
            task = (
                self.task_token.expand(b, -1, -1).to(device=device, dtype=dtype)
                + role[self.ROLE_TASK]
            )
        state_tok = self.state_proj(state)[:, None] + role[self.ROLE_STATE]
        hist = (
            self.state_history_proj(state_history)
            + self.state_history_type.to(device=device, dtype=dtype)
            + role[self.ROLE_STATE_HISTORY]
        )
        if int(getattr(cfg, "action_history_enabled", 0)):
            if executed_memory is None or self.action_private_condition is None:
                raise ValueError(
                    "executed_memory is required when action history compression is enabled"
                )
            expected_history = int(getattr(cfg, "action_history_token_count"))
            if tuple(executed_memory.shape) != (b, expected_history, cfg.hidden_size):
                raise ValueError(
                    f"executed_memory must be [B,{expected_history},{cfg.hidden_size}], got "
                    f"{tuple(executed_memory.shape)}"
                )
            if self.shared_condition_mixer is None:
                raise RuntimeError("shared condition mixer is missing")
            action_content = self.shared_condition_mixer(
                self.action_private_condition(
                    executed_memory.to(device=device, dtype=dtype)
                )
            )
            if int(getattr(cfg, "action_history_condition_exact_null", 0)):
                if (
                    action_history_condition_keep is None
                    or tuple(action_history_condition_keep.shape) != (b,)
                ):
                    raise ValueError(
                        "exact history null semantics require "
                        "action_history_condition_keep [B]"
                    )
                null_action = self.shared_condition_mixer(
                    self.action_private_condition(torch.zeros_like(executed_memory))
                )
                action_content = (
                    action_content - null_action
                ) * action_history_condition_keep.to(
                    device=device, dtype=dtype
                )[:, None, None]
            executed = action_content + role[self.ROLE_EXECUTED]
        else:
            executed = (
                self.executed_proj(executed_history)
                + self.executed_type.to(device=device, dtype=dtype)
                + role[self.ROLE_EXECUTED]
            )
        proposal = (
            self.proposal_proj(proposal_tokens) * proposal_keep[:, None, None]
            + self.proposal_type.to(device=device, dtype=dtype)
            + role[self.ROLE_PROPOSAL]
        )
        noisy_base = (
            self.noisy_physical_lift(noisy_physical)
            + self.horizon_position.to(device=device, dtype=dtype)
            + self.horizon_role(b, device=device, dtype=dtype)
            + role[self.ROLE_NOISY_ACTION]
        )
        noisy = (
            noisy_base[:, :, None, :] + self.action_basis_embed.to(device=device, dtype=dtype)
        ).reshape(b, cfg.action_horizon * cfg.action_basis_tokens, cfg.hidden_size)
        if rollout_init.shape != (b, cfg.future_token_count, cfg.hidden_size):
            raise ValueError(
                f"rollout_init must be [B,{cfg.future_token_count},{cfg.hidden_size}], got {tuple(rollout_init.shape)}"
            )
        rollout = rollout_init.to(device=device, dtype=dtype).reshape(
            b,
            cfg.future_anchors,
            cfg.num_cameras * cfg.future_grid_size * cfg.future_grid_size,
            cfg.hidden_size,
        )
        rollout = (
            rollout
            + self.rollout_anchor_type.to(device=device, dtype=dtype)
            + self.rollout_grid_type.to(device=device, dtype=dtype)
        )
        rollout = (
            rollout.reshape(b, cfg.future_token_count, cfg.hidden_size) + role[self.ROLE_ROLLOUT]
        )
        if stage_init is None:
            stage = rollout.new_empty(b, 0, cfg.hidden_size)
        else:
            if self.stage_type is None:
                raise ValueError("stage_init was supplied while hierarchical Flow-DINO is disabled")
            if tuple(stage_init.shape) != (b, 1, cfg.hidden_size):
                raise ValueError(
                    f"stage_init must be [B,1,{cfg.hidden_size}], got {tuple(stage_init.shape)}"
                )
            stage = (
                stage_init.to(device=device, dtype=dtype)
                + self.stage_type.to(device=device, dtype=dtype)
                + role[self.ROLE_ROLLOUT]
            )
        registers = (
            self.registers.expand(b, -1, -1).to(device=device, dtype=dtype)
            + role[self.ROLE_REGISTER]
        )
        # Goal and executed-action memory are adjacent sibling conditions.
        # State remains in the same directed context region, but no longer
        # physically separates the two modalities on the serialized canvas.
        named_parts = [
            ("state", state_tok),
            ("state_history", hist),
            ("task", task),
            ("executed", executed),
            ("proposal", proposal),
            ("trajectory", noisy),
            ("stage", stage),
            ("rollout", rollout),
            ("registers", registers),
        ]
        starts: dict[str, int] = {}
        offset = 0
        for name, part in named_parts:
            starts[name] = offset
            offset += part.shape[1]
        slices = {
            "task": slice(starts["task"], starts["task"] + task.shape[1]),
            "state": slice(starts["state"], starts["state"] + 1),
            "state_history": slice(
                starts["state_history"],
                starts["state_history"] + cfg.visual_history_length,
            ),
            "executed": slice(
                starts["executed"], starts["executed"] + executed.shape[1]
            ),
            "proposal": slice(
                starts["proposal"], starts["proposal"] + cfg.action_horizon
            ),
            "trajectory": slice(
                starts["trajectory"],
                starts["trajectory"] + cfg.action_horizon * cfg.action_basis_tokens,
            ),
            "stage": slice(starts["stage"], starts["stage"] + stage.shape[1]),
            "rollout": slice(
                starts["rollout"], starts["rollout"] + cfg.future_token_count
            ),
            "registers": slice(
                starts["registers"], starts["registers"] + cfg.canvas_registers
            ),
        }
        return self.drop(torch.cat([part for _, part in named_parts], dim=1)), slices


class TemporalDynamicsBoundDiTBlock(nn.Module):
    """Canvas block with explicit action-to-rollout transition sublayer."""

    def __init__(self, config: TrunkPrimitiveConfig, *, role: str = "shared") -> None:
        super().__init__()
        if role not in {"shared", "grounding", "world", "policy"}:
            raise ValueError(f"unsupported DiT block role: {role}")
        self.role = role
        self.residual_amplitude_contract = bool(
            int(getattr(config, "role_residual_amplitude_contract", 0))
        )
        self.residual_max_update_rms = float(
            getattr(config, "role_residual_max_update_rms", 0.50)
        )
        self.residual_contract_after_gate = bool(
            int(getattr(config, "role_residual_contract_after_gate", 0))
        )
        self.complete_numerical_contract = bool(
            int(getattr(config, "flow_jepa_complete_numerical_contract", 0))
            and role in {"grounding", "world", "policy"}
        )
        self.normalization_floor = float(
            getattr(config, "flow_jepa_routing_norm_floor", 0.25)
        )
        # LayerNorm's learned affine is useful for memory K/V, but an
        # unconstrained scale would reopen the Jacobian bound established by
        # the variance floor. This smooth eighth-order bound is effectively
        # identity near the initialized scale of one and asymptotes to four.
        self.normalization_affine_max = 4.0
        # The compact mainline executes the first V120 policy block at every
        # dynamic call.  Schema35 proved that bounding only the written
        # residual does not bound the hidden AdaLN Jacobian when a downstream
        # consumer drives the block into saturation.  Reuse the established
        # smooth absolute bound (eighth-order and effectively identity around
        # zero) for this policy-only modulation boundary.  Grounding keeps its
        # exact inherited modulation and is not changed by this repair.
        self.modulation_amplitude_contract = bool(
            self.complete_numerical_contract and role == "policy"
        )
        self.modulation_absolute_max = self.normalization_affine_max
        self.directed_canvas_attention = bool(
            int(getattr(config, "flow_jepa_enabled", 0))
            and int(getattr(config, "flow_jepa_directed_canvas_attention", 0))
        )
        self.action_free_world_factual = bool(
            role == "world"
            and int(
                getattr(
                    config,
                    "flow_jepa_action_free_world_factual",
                    0,
                )
            )
        )
        self.grounded_fact_only = bool(
            role == "grounding"
            and (
                int(
                    getattr(
                        config,
                        "flow_jepa_grounded_intent_effect_mainline",
                        0,
                    )
                )
                or int(
                    getattr(
                        config,
                        "flow_jepa_object_intent_dynamics_mainline",
                        0,
                    )
                )
            )
        )
        self.policy_explicit_handoff_only = bool(
            role == "policy"
            and int(getattr(config, "flow_jepa_policy_plan_compiler", 0))
        )
        self.grounded_policy_explicit_only = bool(
            role == "policy"
            and (
                int(
                    getattr(
                        config,
                        "flow_jepa_grounded_intent_effect_mainline",
                        0,
                    )
                )
                or int(
                    getattr(
                        config,
                        "flow_jepa_object_intent_dynamics_mainline",
                        0,
                    )
                )
            )
        )
        self.visual_cross_enabled = not (
            role == "policy"
            and bool(int(getattr(config, "flow_jepa_strict_role_visual_path", 0)))
        )
        self.world_anchor_write_only = bool(
            role == "world"
            and int(getattr(config, "flow_jepa_world_anchor_write_only", 0))
        )
        self.future_anchors = int(config.future_anchors)
        self.num_cameras = int(config.num_cameras)
        self.future_grid_size = int(config.future_grid_size)
        h = config.hidden_size
        self.n1 = nn.LayerNorm(h, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(
            h, config.num_heads, batch_first=True, dropout=config.dropout
        )
        self.n2 = nn.LayerNorm(h, elementwise_affine=False)
        self.mem_norm = nn.LayerNorm(h)
        self.cross = nn.MultiheadAttention(
            h, config.num_heads, batch_first=True, dropout=config.dropout
        )
        if (
            not self.visual_cross_enabled
            and bool(int(getattr(config, "flow_jepa_late_policy_detail", 0)))
        ):
            # V102 policy blocks read observation detail only through the
            # explicit late reader. Preserve legacy modules in the state dict,
            # but do not advertise the disabled visual bypass as trainable.
            self.mem_norm.requires_grad_(False)
            self.cross.requires_grad_(False)
        self.n_dyn_q = nn.LayerNorm(h, elementwise_affine=False)
        self.n_dyn_kv = nn.LayerNorm(h)
        self.rollout_cross = nn.MultiheadAttention(
            h, config.num_heads, batch_first=True, dropout=config.dropout
        )
        if role == "policy":
            # Policy blocks own trajectory writes only. Their stage/rollout
            # transition branch is therefore unreachable by construction
            # (see the role guards in forward). Keep the modules for old
            # checkpoint layouts, but never present dead parameters to the
            # V96+ optimizer.
            self.n_dyn_kv.requires_grad_(False)
            self.rollout_cross.requires_grad_(False)
        # A single stage key would otherwise compete with O(100) context/action
        # keys in one softmax.  This dedicated residual bridge gives the
        # coarse stage prediction an explicit, measurable influence on every
        # spatial window token while retaining direct visual evidence for fine
        # patch detail.
        if int(getattr(config, "flow_jepa_enabled", 0)) and not int(
            getattr(config, "flow_jepa_late_bottleneck", 0)
        ):
            self.stage_to_window = nn.Sequential(
                nn.LayerNorm(h, elementwise_affine=False), nn.Linear(h, 2 * h, bias=False)
            )
            self.stage_window_norm = nn.LayerNorm(h, elementwise_affine=False)
            self.stage_to_window_gate_logit = nn.Parameter(torch.tensor(-2.0))
            nn.init.normal_(self.stage_to_window[-1].weight, mean=0.0, std=3e-3)
        else:
            self.stage_to_window = None
            self.stage_window_norm = None
            self.register_parameter("stage_to_window_gate_logit", None)
        self.n3 = nn.LayerNorm(h, elementwise_affine=False)
        self.ffn = BiasFreeFFN(h, config.ffn_expansion)
        self.drop = nn.Dropout(config.dropout)
        self.mod = nn.Linear(h, 12 * h)
        if role == "shared":
            self.register_parameter("role_embedding", None)
        else:
            self.role_embedding = nn.Parameter(torch.randn(1, h) * 0.02)
        nn.init.normal_(self.mod.weight, mean=0.0, std=3e-3)
        nn.init.zeros_(self.mod.bias)
        with torch.no_grad():
            for idx in (2, 5, 8, 11):
                self.mod.bias[idx * h : (idx + 1) * h].fill_(-2.0)

    def _contract_residual(self, update: Tensor) -> tuple[Tensor, Tensor]:
        if not self.residual_amplitude_contract:
            return update, update.new_ones(
                (*update.shape[:-1], 1), dtype=torch.float32
            )
        return smooth_rms_contract(update, self.residual_max_update_rms)

    def _contract_modulation(
        self, shift: Tensor, scale: Tensor
    ) -> tuple[Tensor, Tensor]:
        if self.modulation_amplitude_contract:
            shift = smooth_absolute_contract(
                shift, self.modulation_absolute_max
            )
            scale = smooth_absolute_contract(
                scale, self.modulation_absolute_max
            )
        return shift, scale

    def modulate(self, x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        shift, scale = self._contract_modulation(shift, scale)
        return x * (1 + scale[:, None]) + shift[:, None]

    @staticmethod
    def _directed_attention_mask(
        length: int,
        slices: dict[str, slice],
        *,
        device: torch.device,
        role: str = "shared",
        action_free_world_factual: bool = False,
        policy_explicit_handoff_only: bool = False,
        grounded_fact_only: bool = False,
        grounded_policy_explicit_only: bool = False,
    ) -> Tensor:
        """Enforce clean-context -> action -> future ownership.

        Clean context/register queries cannot read noisy-action, stage, or
        future-query regions.  Action queries cannot read stage/future.  The
        stage query can read current/action context but not window predictions;
        window queries can read the stage.  This establishes the serial
        context -> action -> stage -> window direction without a recurrent
        hidden state or a second training pass.
        """

        mask = torch.zeros(length, length, device=device, dtype=torch.bool)
        action = slices["trajectory"]
        stage = slices.get("stage", slice(slices["rollout"].start, slices["rollout"].start))
        future = slices["rollout"]
        forbidden_start = int(action.start)
        forbidden_stop = int(future.stop)
        for name in ("task", "state", "state_history", "executed", "proposal", "registers"):
            query = slices[name]
            mask[query, forbidden_start:forbidden_stop] = True
        mask[action, int(stage.start) : int(future.stop)] = True
        if role == "grounding":
            # G1-G3 own observation alignment/canonicalization.  Their
            # spatial rollout must not become a second action-denoising path:
            # the current x_t trajectory and history-derived proposal first
            # enter future/consequence organization at the W boundary.
            mask[future, action] = True
            mask[future, slices["proposal"]] = True
            # The stage token is immediately written back into the window by
            # ``stage_to_window``.  Masking only the final window query would
            # therefore leave trajectory/proposal -> stage -> window as a
            # one-block bypass around the directed ownership contract.
            mask[stage, action] = True
            mask[stage, slices["proposal"]] = True
            if grounded_fact_only:
                # The grounded G group owns current facts, not intent or
                # temporal organization.  T5/task, historical state and
                # executed actions enter only through S after the literal G3
                # boundary.  Mask both G outputs because stage is written
                # directly into rollout later in this same block.
                for name in ("task", "state_history", "executed"):
                    mask[future, slices[name]] = True
                    mask[stage, slices[name]] = True
            # Grounding modulation/dynamics reuse the clean context slices on
            # the next block. Keep those slices from becoming an indirect
            # proposal carrier (proposal -> context -> rollout).
            for name in (
                "task",
                "state",
                "state_history",
                "executed",
                "registers",
            ):
                mask[slices[name], slices["proposal"]] = True
        if role == "world" and action_free_world_factual:
            # W owns an observation/intent-conditioned factual chart.  The
            # flow-matching sample x_t enters only at P2/policy; otherwise P1
            # could recover it indirectly through the W rollout.
            mask[future, action] = True
            # W also writes stage into the factual window in the same block.
            # Keep that intermediate carrier action-free as well.
            mask[stage, action] = True
        if role == "policy":
            # World/future tokens are online predictions, not teacher targets.
            # Ancestral P blocks may read them directly.  V115 instead owns an
            # explicit W->P bridge/FutureEffect handoff, so retaining this
            # attention edge would restore the unsupervised world-residual
            # bypass that the new interface removes.
            if not policy_explicit_handoff_only:
                mask[action, future] = False
            if grounded_policy_explicit_only:
                # Grounded P1 receives Goal/History/G only through the
                # StatelessIntentState-conditioned precision reader that has
                # already written the trajectory query.  Letting the following
                # generic P1 refinement attend task/history/proposal directly
                # would recreate a parallel condition path around S.
                for name in (
                    "task",
                    "state",
                    "state_history",
                    "executed",
                    "proposal",
                    "registers",
                ):
                    mask[action, slices[name]] = True
        if int(stage.stop) > int(stage.start):
            mask[stage, future] = True
        return mask

    def _role_write_mask(
        self, canvas: Tensor, slices: dict[str, slice]
    ) -> Tensor:
        if self.role == "shared":
            return torch.ones(
                1, int(canvas.shape[1]), 1, device=canvas.device, dtype=canvas.dtype
            )
        allowed = {
            "grounding": (
                "task",
                "state",
                "state_history",
                "executed",
                "proposal",
                "registers",
                "stage",
                "rollout",
            ),
            "world": ("stage", "rollout"),
            "policy": ("trajectory",),
        }[self.role]
        mask = torch.zeros(
            1, int(canvas.shape[1]), 1, device=canvas.device, dtype=canvas.dtype
        )
        for name in allowed:
            region = slices.get(name)
            if region is not None and int(region.stop) > int(region.start):
                mask[:, region] = 1.0
        return mask

    def _structure_world_rollout_update(self, update: Tensor) -> Tensor:
        """Restrict a world write to one vector per anchor and camera.

        Grounding owns xy-specific observations.  World blocks may aggregate
        those cells when deciding a temporal/camera update, but cannot write a
        different residual into each xy slot.  Broadcasting is applied after
        dropout, so stochastic training cannot accidentally reintroduce a
        cell-specific world residual.
        """

        if not self.world_anchor_write_only:
            return update
        batch, tokens, hidden = update.shape
        expected = (
            self.future_anchors
            * self.num_cameras
            * self.future_grid_size
            * self.future_grid_size
        )
        if int(tokens) != expected:
            raise ValueError(
                "world rollout update does not match "
                f"anchors*cameras*grid^2={expected}: got {tokens}"
            )
        grouped = update.reshape(
            batch,
            self.future_anchors,
            self.num_cameras,
            self.future_grid_size,
            self.future_grid_size,
            hidden,
        )
        pooled = grouped.mean(dim=(3, 4), keepdim=True)
        return pooled.expand_as(grouped).reshape_as(update)

    def _structure_world_canvas_update(
        self, update: Tensor, slices: dict[str, slice]
    ) -> Tensor:
        if not self.world_anchor_write_only:
            return update
        rollout_slice = slices["rollout"]
        rollout_update = self._structure_world_rollout_update(
            update[:, rollout_slice]
        )
        return torch.cat(
            (
                update[:, : int(rollout_slice.start)],
                rollout_update,
                update[:, int(rollout_slice.stop) :],
            ),
            dim=1,
        )

    def forward(
        self,
        canvas: Tensor,
        visual_memory: Tensor,
        mod_embed: Tensor,
        slices: dict[str, slice],
        *,
        visual_value_memory: Tensor | None = None,
        rollout_query_context: Tensor | None = None,
        collect_diagnostics: bool = True,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        role_mod = (
            0.0
            if self.role_embedding is None
            else self.role_embedding.to(device=mod_embed.device, dtype=mod_embed.dtype)
        )
        sa_s, sa_c, sa_g, ca_s, ca_c, ca_g, dy_s, dy_c, dy_g, ff_s, ff_c, ff_g = self.mod(
            mod_embed + role_mod
        ).chunk(12, dim=-1)
        modulation_shift_raw_max_abs = torch.stack(
            (sa_s, ca_s, dy_s, ff_s), dim=0
        ).detach().float().abs().amax()
        modulation_scale_raw_max_abs = torch.stack(
            (sa_c, ca_c, dy_c, ff_c), dim=0
        ).detach().float().abs().amax()
        contracted_modulations = tuple(
            self._contract_modulation(shift, scale)
            for shift, scale in (
                (sa_s, sa_c),
                (ca_s, ca_c),
                (dy_s, dy_c),
                (ff_s, ff_c),
            )
        )
        modulation_shift_max_abs = torch.stack(
            tuple(row[0] for row in contracted_modulations), dim=0
        ).detach().float().abs().amax()
        modulation_scale_max_abs = torch.stack(
            tuple(row[1] for row in contracted_modulations), dim=0
        ).detach().float().abs().amax()
        write_mask = self._role_write_mask(canvas, slices)
        residual_contract_metrics: dict[str, Tensor] = {}
        residual_raw_rows: list[Tensor] = []
        residual_proposed_rows: list[Tensor] = []
        residual_bounded_rows: list[Tensor] = []
        residual_written_rows: list[Tensor] = []
        residual_compression_rows: list[Tensor] = []
        normalization_denominator_rows: list[Tensor] = []
        normalization_gain_rows: list[Tensor] = []

        def normalize(module: nn.LayerNorm, value: Tensor) -> Tensor:
            if not self.complete_numerical_contract:
                return module(value)
            normalized, denominator = variance_floored_centered_norm(
                value, self.normalization_floor
            )
            gain = normalized.new_tensor(
                1.0 / self.normalization_floor, dtype=torch.float32
            )
            if module.elementwise_affine:
                if module.weight is None or module.bias is None:
                    raise RuntimeError("affine LayerNorm is missing weight or bias")
                bounded_weight = smooth_absolute_contract(
                    module.weight, self.normalization_affine_max
                ).float()
                normalized = (
                    normalized.float()
                    * bounded_weight.to(
                        device=normalized.device, dtype=normalized.dtype
                    )
                    + module.bias.to(
                        device=normalized.device, dtype=normalized.dtype
                    )
                )
                gain = (
                    bounded_weight.abs().amax() / self.normalization_floor
                ).detach()
            if collect_diagnostics:
                normalization_denominator_rows.append(
                    denominator.detach().float().amin()
                )
                normalization_gain_rows.append(gain.detach().float())
            return normalized.to(dtype=value.dtype)

        def stabilize(
            name: str,
            residual: Tensor,
            metric_mask: Tensor | None = None,
            *,
            gate: Tensor | None = None,
            write_mask: Tensor | None = None,
        ) -> Tensor:
            proposal = residual
            if self.residual_contract_after_gate:
                if gate is not None:
                    proposal = proposal * gate[:, None]
                if write_mask is not None:
                    proposal = proposal * write_mask
            bounded, scale = self._contract_residual(proposal)
            written = bounded
            if not self.residual_contract_after_gate:
                if gate is not None:
                    written = written * gate[:, None]
                if write_mask is not None:
                    written = written * write_mask
            if not collect_diagnostics:
                return written
            raw_token_rms = (
                residual.detach().float().square().mean(dim=-1).sqrt()
            )
            proposed_token_rms = (
                proposal.detach().float().square().mean(dim=-1).sqrt()
            )
            bounded_token_rms = (
                bounded.detach().float().square().mean(dim=-1).sqrt()
            )
            written_token_rms = (
                written.detach().float().square().mean(dim=-1).sqrt()
            )
            compression_token = 1.0 - scale.detach().float()[..., 0]
            if metric_mask is None:
                raw_rms = raw_token_rms.mean()
                proposed_rms = proposed_token_rms.mean()
                bounded_rms = bounded_token_rms.mean()
                written_rms = written_token_rms.mean()
                compression = compression_token.mean()
            else:
                weight = metric_mask.detach().float()[..., 0].expand_as(
                    raw_token_rms
                )
                denominator = weight.sum().clamp_min(1.0)
                raw_rms = (raw_token_rms * weight).sum() / denominator
                proposed_rms = (
                    proposed_token_rms * weight
                ).sum() / denominator
                bounded_rms = (
                    bounded_token_rms * weight
                ).sum() / denominator
                written_rms = (
                    written_token_rms * weight
                ).sum() / denominator
                compression = (
                    compression_token * weight
                ).sum() / denominator
            residual_contract_metrics[f"residual_{name}_raw_rms"] = raw_rms
            residual_contract_metrics[
                f"residual_{name}_proposed_rms"
            ] = proposed_rms
            residual_contract_metrics[
                f"residual_{name}_bounded_rms"
            ] = bounded_rms
            residual_contract_metrics[
                f"residual_{name}_compression"
            ] = compression
            residual_contract_metrics[
                f"residual_{name}_written_rms"
            ] = written_rms
            residual_raw_rows.append(raw_rms)
            residual_proposed_rows.append(proposed_rms)
            residual_bounded_rows.append(bounded_rms)
            residual_written_rows.append(written_rms)
            residual_compression_rows.append(compression)
            return written

        value = normalize(self.n1, canvas)
        qk = self.modulate(value, sa_s, sa_c)
        self_qk_rms = qk.detach().float().square().mean().sqrt()
        query_context_rms = canvas.new_zeros((), dtype=torch.float32)
        if rollout_query_context is not None:
            rollout_slice = slices["rollout"]
            full_expected = (
                int(canvas.shape[0]),
                int(rollout_slice.stop) - int(rollout_slice.start),
                int(canvas.shape[-1]),
            )
            compact_expected = (
                int(canvas.shape[0]),
                self.future_anchors,
                int(canvas.shape[-1]),
            )
            context_shape = tuple(rollout_query_context.shape)
            if self.role != "world" or context_shape not in {
                full_expected,
                compact_expected,
            }:
                raise ValueError(
                    "rollout query context is world-only and must be either "
                    "[B,anchor,H] or the complete rollout chart"
                )
            selector_context = rollout_query_context.to(
                device=qk.device, dtype=qk.dtype
            )
            if context_shape == compact_expected:
                rollout_qk = qk[:, rollout_slice].reshape(
                    int(canvas.shape[0]),
                    self.future_anchors,
                    self.num_cameras,
                    self.future_grid_size,
                    self.future_grid_size,
                    int(canvas.shape[-1]),
                )
                rollout_qk = (
                    rollout_qk
                    + selector_context[:, :, None, None, None]
                ).reshape(
                    int(canvas.shape[0]),
                    int(rollout_slice.stop) - int(rollout_slice.start),
                    int(canvas.shape[-1]),
                )
            else:
                rollout_qk = qk[:, rollout_slice] + selector_context
            qk = torch.cat(
                (
                    qk[:, : int(rollout_slice.start)],
                    rollout_qk,
                    qk[:, int(rollout_slice.stop) :],
                ),
                dim=1,
            )
            if collect_diagnostics:
                query_context_rms = (
                    selector_context.detach().float().square().mean().sqrt()
                )
        attention_mask = (
            self._directed_attention_mask(
                int(canvas.shape[1]),
                slices,
                device=canvas.device,
                role=self.role,
                action_free_world_factual=self.action_free_world_factual,
                policy_explicit_handoff_only=(
                    self.policy_explicit_handoff_only
                ),
                grounded_fact_only=self.grounded_fact_only,
                grounded_policy_explicit_only=(
                    self.grounded_policy_explicit_only
                ),
            )
            if self.directed_canvas_attention
            else None
        )
        update, _ = self.self_attn(
            qk, qk, value, attn_mask=attention_mask, need_weights=False
        )
        g_sa = torch.sigmoid(sa_g)
        update = self._structure_world_canvas_update(self.drop(update), slices)
        if self.residual_contract_after_gate:
            update = stabilize(
                "self",
                update,
                write_mask,
                gate=g_sa,
                write_mask=write_mask,
            )
            canvas = canvas + update
        else:
            update = stabilize("self", update, write_mask)
            canvas = canvas + write_mask * g_sa[:, None] * update

        g_ca = torch.sigmoid(ca_g)
        if self.visual_cross_enabled:
            query = self.modulate(normalize(self.n2, canvas), ca_s, ca_c)
            memory_key = normalize(self.mem_norm, visual_memory)
            memory_value = normalize(
                self.mem_norm,
                visual_memory if visual_value_memory is None else visual_value_memory
            )
            if tuple(memory_key.shape) != tuple(memory_value.shape):
                raise ValueError("visual selector and value memories must be shape-aligned")
            update, _ = self.cross(query, memory_key, memory_value, need_weights=False)
            update = self._structure_world_canvas_update(self.drop(update), slices)
            if self.residual_contract_after_gate:
                update = stabilize(
                    "visual",
                    update,
                    write_mask,
                    gate=g_ca,
                    write_mask=write_mask,
                )
                canvas = canvas + update
            else:
                update = stabilize("visual", update, write_mask)
                canvas = canvas + write_mask * g_ca[:, None] * update
            effective_visual_gate = (
                g_ca.mean()
                if collect_diagnostics
                else canvas.new_zeros((), dtype=torch.float32)
            )
        else:
            # In the strict role path the policy group can read the world
            # rollout through directed canvas self-attention, but it cannot
            # re-read raw/DINO memory and bypass the grounding/world owners.
            effective_visual_gate = g_ca.mean() * 0.0

        context_names = (
            ["state"]
            if self.grounded_fact_only
            else [
                "task",
                "state",
                "state_history",
                "executed",
            ]
        )
        if self.role != "grounding":
            context_names.append("proposal")
        if self.role != "grounding" and not self.action_free_world_factual:
            context_names.append("trajectory")
        context_parts = [
            canvas[:, slices[name]] for name in context_names
        ]
        context_kv = normalize(
            self.n_dyn_kv, torch.cat(context_parts, dim=1)
        )
        stage_slice = slices.get("stage")
        has_stage = stage_slice is not None and int(stage_slice.stop) > int(stage_slice.start)
        g_dyn = torch.sigmoid(dy_g)
        if has_stage and self.role != "policy":
            stage = canvas[:, stage_slice]
            stage_q = self.modulate(
                normalize(self.n_dyn_q, stage), dy_s, dy_c
            )
            stage_update, _ = self.rollout_cross(
                stage_q, context_kv, context_kv, need_weights=False
            )
            stage_update = self.drop(stage_update)
            if self.residual_contract_after_gate:
                stage_update = stabilize(
                    "stage", stage_update, gate=g_dyn
                )
                stage = stage + stage_update
            else:
                stage_update = stabilize("stage", stage_update)
                stage = stage + g_dyn[:, None] * stage_update
        rollout = canvas[:, slices["rollout"]]
        if has_stage and self.role != "policy":
            if (
                self.stage_to_window is None
                or self.stage_window_norm is None
                or self.stage_to_window_gate_logit is None
            ):
                raise RuntimeError("stage canvas tokens require the hierarchical stage bridge")
            stage_gamma, stage_beta = self.stage_to_window(stage).chunk(2, dim=-1)
            stage_window_gate = torch.sigmoid(
                self.stage_to_window_gate_logit.to(device=rollout.device, dtype=rollout.dtype)
            )
            stage_window_update = stage_window_gate * (
                torch.tanh(stage_gamma)
                * normalize(self.stage_window_norm, rollout)
                + stage_beta
            )
            stage_window_update = self._structure_world_rollout_update(
                self.drop(stage_window_update)
            )
            stage_window_update = stabilize(
                "stage_to_window", stage_window_update
            )
            rollout = rollout + stage_window_update
        else:
            stage_window_gate = rollout.new_zeros(())
            stage_window_update = rollout.new_zeros(rollout.shape)
        if self.role != "policy":
            rollout_parts = context_parts + ([stage] if has_stage else [])
            kv = normalize(
                self.n_dyn_kv, torch.cat(rollout_parts, dim=1)
            )
            q = self.modulate(
                normalize(self.n_dyn_q, rollout), dy_s, dy_c
            )
            if rollout_query_context is not None:
                selector_context = rollout_query_context.to(
                    device=q.device, dtype=q.dtype
                )
                if tuple(selector_context.shape) == (
                    int(q.shape[0]),
                    self.future_anchors,
                    int(q.shape[-1]),
                ):
                    q = (
                        q.reshape(
                            int(q.shape[0]),
                            self.future_anchors,
                            self.num_cameras,
                            self.future_grid_size,
                            self.future_grid_size,
                            int(q.shape[-1]),
                        )
                        + selector_context[:, :, None, None, None]
                    ).reshape_as(q)
                else:
                    q = q + selector_context
            update, _ = self.rollout_cross(q, kv, kv, need_weights=False)
            update = self._structure_world_rollout_update(self.drop(update))
            if self.residual_contract_after_gate:
                update = stabilize("rollout", update, gate=g_dyn)
                rollout = rollout + update
            else:
                update = stabilize("rollout", update)
                rollout = rollout + g_dyn[:, None] * update
        # Reassemble functionally.  In-place assignment into canvas views
        # invalidates autograd because the updated stage is also consumed by
        # the rollout branch in this same block.
        if has_stage and self.role != "policy":
            canvas = torch.cat(
                (
                    canvas[:, : int(stage_slice.start)],
                    stage,
                    canvas[:, int(stage_slice.stop) : int(slices["rollout"].start)],
                    rollout,
                    canvas[:, int(slices["rollout"].stop) :],
                ),
                dim=1,
            )
        else:
            canvas = torch.cat(
                (
                    canvas[:, : int(slices["rollout"].start)],
                    rollout,
                    canvas[:, int(slices["rollout"].stop) :],
                ),
                dim=1,
            )

        ffn_input = self.modulate(normalize(self.n3, canvas), ff_s, ff_c)
        ffn_input_rms = ffn_input.detach().float().square().mean().sqrt()
        update = self.ffn(ffn_input)
        g_ff = torch.sigmoid(ff_g)
        update = self._structure_world_canvas_update(self.drop(update), slices)
        if self.residual_contract_after_gate:
            update = stabilize(
                "ffn",
                update,
                write_mask,
                gate=g_ff,
                write_mask=write_mask,
            )
            canvas = canvas + update
        else:
            update = stabilize("ffn", update, write_mask)
            canvas = canvas + write_mask * g_ff[:, None] * update
        if not collect_diagnostics:
            return canvas, {}
        summary_zero = canvas.new_zeros((), dtype=torch.float32)
        residual_contract_metrics.update(
            {
                "residual_contract_enabled": canvas.new_tensor(
                    float(self.residual_amplitude_contract),
                    dtype=torch.float32,
                ),
                "residual_contract_max_rms": canvas.new_tensor(
                    self.residual_max_update_rms,
                    dtype=torch.float32,
                ),
                "residual_contract_after_gate": canvas.new_tensor(
                    float(self.residual_contract_after_gate),
                    dtype=torch.float32,
                ),
                "residual_raw_rms": (
                    torch.stack(residual_raw_rows).mean()
                    if residual_raw_rows
                    else summary_zero
                ),
                "residual_proposed_rms": (
                    torch.stack(residual_proposed_rows).mean()
                    if residual_proposed_rows
                    else summary_zero
                ),
                "residual_bounded_rms": (
                    torch.stack(residual_bounded_rows).mean()
                    if residual_bounded_rows
                    else summary_zero
                ),
                "residual_written_rms": (
                    torch.stack(residual_written_rows).mean()
                    if residual_written_rows
                    else summary_zero
                ),
                "residual_compression": (
                    torch.stack(residual_compression_rows).mean()
                    if residual_compression_rows
                    else summary_zero
                ),
                "normalization_contract_enabled": canvas.new_tensor(
                    float(self.complete_numerical_contract),
                    dtype=torch.float32,
                ),
                "modulation_contract_enabled": canvas.new_tensor(
                    float(self.modulation_amplitude_contract),
                    dtype=torch.float32,
                ),
                "modulation_shift_max_abs": modulation_shift_max_abs,
                "modulation_scale_max_abs": modulation_scale_max_abs,
                "modulation_shift_raw_max_abs": modulation_shift_raw_max_abs,
                "modulation_scale_raw_max_abs": modulation_scale_raw_max_abs,
                "self_qk_rms": self_qk_rms,
                "ffn_input_rms": ffn_input_rms,
                "normalization_denominator_min": (
                    torch.stack(normalization_denominator_rows).amin()
                    if normalization_denominator_rows
                    else canvas.new_ones((), dtype=torch.float32)
                ),
                "normalization_gain_max": (
                    torch.stack(normalization_gain_rows).amax()
                    if normalization_gain_rows
                    else canvas.new_ones((), dtype=torch.float32)
                ),
            }
        )
        return canvas, {
            "gate_self": g_sa.mean(),
            "gate_visual": effective_visual_gate,
            "visual_cross_enabled": canvas.new_tensor(float(self.visual_cross_enabled)),
            "gate_stage": g_dyn.mean() if has_stage else g_dyn.mean() * 0.0,
            "gate_stage_to_window": stage_window_gate,
            "stage_to_window_update_norm": stage_window_update.detach().float().norm(dim=-1).mean(),
            "gate_rollout": g_dyn.mean(),
            "gate_ffn": g_ff.mean(),
            "role_grounding": canvas.new_tensor(float(self.role == "grounding")),
            "role_world": canvas.new_tensor(float(self.role == "world")),
            "role_policy": canvas.new_tensor(float(self.role == "policy")),
            "rollout_query_context_rms": query_context_rms,
            "world_anchor_write_only": canvas.new_tensor(
                float(self.world_anchor_write_only)
            ),
            **residual_contract_metrics,
        }


class CanvasPhysicalVelocityHead(nn.Module):
    def __init__(self, config: TrunkPrimitiveConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.net = nn.Sequential(
            nn.LayerNorm(config.action_basis_tokens * h),
            nn.Linear(config.action_basis_tokens * h, 2 * h),
            nn.SiLU(),
            nn.Linear(2 * h, config.physical_action_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def pooled(self, trajectory_tokens: Tensor) -> Tensor:
        cfg = self.config
        b = trajectory_tokens.shape[0]
        return trajectory_tokens.reshape(
            b, cfg.action_horizon, cfg.action_basis_tokens, cfg.hidden_size
        ).mean(dim=2)

    def forward(self, trajectory_tokens: Tensor) -> Tensor:
        cfg = self.config
        b = trajectory_tokens.shape[0]
        grouped = trajectory_tokens.reshape(
            b, cfg.action_horizon, cfg.action_basis_tokens, cfg.hidden_size
        )
        return self.net(
            grouped.reshape(b, cfg.action_horizon, cfg.action_basis_tokens * cfg.hidden_size)
        )


class RolloutActionResidualHead(nn.Module):
    """Tail action residual that must read rollout latent tokens."""

    def __init__(self, config: TrunkPrimitiveConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.query_norm = nn.LayerNorm(h)
        self.rollout_norm = nn.LayerNorm(h)
        self.cross = nn.MultiheadAttention(
            h, config.num_heads, batch_first=True, dropout=config.dropout
        )
        self.net = nn.Sequential(
            nn.LayerNorm(h),
            nn.Linear(h, 2 * h),
            nn.SiLU(),
            nn.Linear(2 * h, config.physical_action_dim),
        )
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)
        alpha = torch.zeros(config.action_horizon, dtype=torch.float32)
        start = max(int(config.rollout_tail_start_step), 1)
        full = max(int(config.rollout_tail_full_step), start)
        for i in range(config.action_horizon):
            step = i + 1
            if step < start:
                value = 0.0
            elif step >= full:
                value = 1.0
            else:
                value = float(step - start + 1) / float(max(full - start + 1, 1))
            alpha[i] = value
        self.register_buffer("alpha", alpha[None, :, None], persistent=True)

    def forward(self, trajectory_pooled: Tensor, rollout_tokens: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(trajectory_pooled)
        kv = self.rollout_norm(rollout_tokens)
        update, _ = self.cross(q, kv, kv, need_weights=False)
        residual = self.net(update)
        alpha = self.alpha.to(device=residual.device, dtype=residual.dtype)
        return residual * alpha, alpha


class ControlledResidualLatentDynamics(nn.Module):
    """Predict future DINO residual as weak visual base + action-centered delta.

    ``base_effect`` is visual-only and intentionally low-capacity.  ``basis`` is
    a set of local transition directions derived from the current visual rollout
    base.  ``coeff(action)`` is produced by action/state/proposal tokens, but the
    actual delta uses ``coeff(action) - coeff(neutral_context)``.  This centering
    removes action-independent coefficient bias from the controlled path and
    forces the delta to represent an intervention relative to a learned no-op
    context instead of another average-future predictor.
    """

    def __init__(self, config: TrunkPrimitiveConfig) -> None:
        super().__init__()
        self.config = config
        self.base_mode = str(getattr(config, "controlled_base_mode", "learned"))
        if self.base_mode not in {"learned", "fixed_zero"}:
            raise ValueError(f"unsupported controlled_base_mode={self.base_mode!r}")
        h = int(config.hidden_size)
        r = int(config.controlled_delta_rank)
        base_hidden = int(config.base_effect_hidden)
        self.base_head = nn.Sequential(
            nn.LayerNorm(h),
            nn.Linear(h, base_hidden),
            nn.SiLU(),
            nn.Linear(base_hidden, h),
        )
        self.basis_head = nn.Sequential(
            nn.LayerNorm(h),
            nn.Linear(h, 2 * h),
            nn.SiLU(),
            nn.Linear(2 * h, r * h),
        )
        self.action_queries = nn.Parameter(
            torch.randn(1, int(config.latent_action_tokens), h) * 0.02
        )
        self.neutral_queries = nn.Parameter(
            torch.randn(1, int(config.neutral_action_tokens), h) * 0.02
        )
        self.neutral_bias = nn.Parameter(torch.zeros(1, 1, h))
        self.action_kv_norm = nn.LayerNorm(h)
        self.action_cross = nn.MultiheadAttention(
            h, config.num_heads, batch_first=True, dropout=config.dropout
        )
        self.rollout_query_norm = nn.LayerNorm(h)
        self.action_latent_norm = nn.LayerNorm(h)
        self.coeff_cross = nn.MultiheadAttention(
            h, config.num_heads, batch_first=True, dropout=config.dropout
        )
        # Direct trajectory summary keeps the coefficient path from being a
        # purely second-order product of tiny random attention features.  It is
        # still a network path, not a rule: coefficients are generated from the
        # actual action/trajectory tokens and are shared across real/hold/shuffle
        # counterfactual forwards.
        self.direct_action_norm = nn.LayerNorm(h)
        self.direct_action_mlp = nn.Sequential(
            nn.LayerNorm(h),
            nn.Linear(h, h),
            nn.SiLU(),
            nn.Linear(h, h),
        )
        self.coeff_head = nn.Sequential(
            nn.LayerNorm(2 * h),
            nn.Linear(2 * h, h),
            nn.SiLU(),
            nn.Linear(h, r),
        )
        self.delta_drop = nn.Dropout(config.controlled_delta_dropout)
        # Do not LayerNorm the intervention delta: amplitude is part of the
        # causal signal.  LayerNorm can amplify tiny centered coefficient noise
        # into a full-magnitude average residual.  Start with a moderate gain
        # and let training adjust it.
        self.delta_gain = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        nn.init.normal_(self.base_head[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.base_head[-1].bias)
        # The controlled path must be alive before long training.  Using 1e-3
        # for both basis and coefficients makes their product almost zero and
        # lets the weak base dominate the preflight task.  Keep the base tiny,
        # but initialize basis/coefficients at normal transformer residual scale.
        nn.init.normal_(self.basis_head[-1].weight, mean=0.0, std=3e-2)
        nn.init.zeros_(self.basis_head[-1].bias)
        nn.init.normal_(self.coeff_head[-1].weight, mean=0.0, std=5e-2)
        nn.init.zeros_(self.coeff_head[-1].bias)
        nn.init.normal_(self.direct_action_mlp[-1].weight, mean=0.0, std=5e-2)
        nn.init.zeros_(self.direct_action_mlp[-1].bias)
        if self.base_mode == "fixed_zero":
            # A learned base has no identifiable target: base + delta can stay
            # correct while both terms grow in opposite directions.  The
            # no-change origin is the only target-free baseline with a unique
            # residual decomposition, so new V39 runs freeze this legacy head.
            self.base_head.requires_grad_(False)

    def _coeff(
        self,
        rollout_base: Tensor,
        context_kv: Tensor,
        *,
        action_tokens: Tensor | None,
        neutral: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return coefficients, latent action tokens, and direct summary.

        ``neutral=True`` deliberately excludes trajectory/action tokens and uses
        learned neutral queries.  The output is still state/context dependent,
        so it can absorb no-op phase/context bias, but it cannot carry the
        candidate action chunk.  The centered coefficient subtracts this term.
        """
        b, n, h = rollout_base.shape
        if neutral:
            queries = self.neutral_queries.expand(b, -1, -1).to(
                device=rollout_base.device, dtype=rollout_base.dtype
            )
            kv = self.action_kv_norm(context_kv)
            direct = self.neutral_bias.to(
                device=rollout_base.device, dtype=rollout_base.dtype
            ).expand(b, n, -1)
        else:
            queries = self.action_queries.expand(b, -1, -1).to(
                device=rollout_base.device, dtype=rollout_base.dtype
            )
            if action_tokens is None:
                kv_source = context_kv
                action_source = context_kv
            else:
                kv_source = torch.cat([context_kv, action_tokens], dim=1)
                action_source = action_tokens
            kv = self.action_kv_norm(kv_source)
            direct_action = self.direct_action_mlp(
                self.direct_action_norm(action_source).mean(dim=1)
            )
            direct = direct_action[:, None, :].expand(-1, n, -1)

        latent_action, _ = self.action_cross(queries, kv, kv, need_weights=False)
        rq = self.rollout_query_norm(rollout_base)
        la = self.action_latent_norm(latent_action)
        action_context, _ = self.coeff_cross(rq, la, la, need_weights=False)
        coeff = torch.tanh(self.coeff_head(torch.cat([rq, action_context + direct], dim=-1)))
        return coeff, latent_action, direct

    def forward(
        self,
        rollout_base: Tensor,
        context_kv: Tensor,
        action_tokens: Tensor | None = None,
        *,
        transition_tokens: Tensor | None = None,
    ) -> dict[str, Tensor]:
        cfg = self.config
        b, n, h = rollout_base.shape
        if n != cfg.future_token_count or h != cfg.hidden_size:
            raise ValueError(
                f"rollout_base must be [B,{cfg.future_token_count},{cfg.hidden_size}], got {tuple(rollout_base.shape)}"
            )
        transition = rollout_base if transition_tokens is None else transition_tokens
        if transition.shape != rollout_base.shape:
            raise ValueError(
                f"transition_tokens must match rollout_base {tuple(rollout_base.shape)}, got {tuple(transition.shape)}"
            )
        if self.base_mode == "fixed_zero":
            base_effect = torch.zeros_like(rollout_base)
        else:
            base_effect = self.base_head(rollout_base)
        # The baseline and transition representation are deliberately separate.
        # V39 can keep a fixed, identifiable origin while still using the full
        # deep rollout canvas to construct action-conditioned directions.
        basis = self.basis_head(transition).reshape(b, n, cfg.controlled_delta_rank, h)
        coeff_action, latent_action, _ = self._coeff(
            transition, context_kv, action_tokens=action_tokens, neutral=False
        )
        coeff_neutral, latent_neutral, _ = self._coeff(
            transition, context_kv, action_tokens=None, neutral=True
        )
        coeff_delta = coeff_action - coeff_neutral
        controlled_delta = (
            torch.einsum("bnr,bnrh->bnh", coeff_delta, basis)
            / float(cfg.controlled_delta_rank) ** 0.5
        )
        controlled_delta = self.delta_drop(
            controlled_delta
            * self.delta_gain.to(device=controlled_delta.device, dtype=controlled_delta.dtype)
        )
        pred_effect = base_effect + controlled_delta
        base_norm = base_effect.detach().float().norm(dim=-1).mean()
        delta_norm = controlled_delta.detach().float().norm(dim=-1).mean()
        effect_norm = pred_effect.detach().float().norm(dim=-1).mean().clamp_min(1e-6)
        expansion_ratio = (
            effect_norm.new_ones(())
            if self.base_mode == "fixed_zero"
            else (base_norm + delta_norm) / effect_norm
        )
        return {
            "rollout_base_effect_pred": base_effect,
            "rollout_delta_pred": controlled_delta,
            "rollout_effect_pred": pred_effect,
            "rollout_transition_basis": basis,
            "rollout_action_coeff": coeff_action,
            "rollout_neutral_coeff": coeff_neutral,
            "rollout_centered_coeff": coeff_delta,
            "latent_action_tokens": latent_action,
            "latent_neutral_tokens": latent_neutral,
            "rollout_coeff_abs_mean": coeff_action.detach().float().abs().mean(),
            "rollout_neutral_coeff_abs_mean": coeff_neutral.detach().float().abs().mean(),
            "rollout_centered_coeff_abs_mean": coeff_delta.detach().float().abs().mean(),
            "rollout_basis_norm": basis.detach().float().norm(dim=-1).mean(),
            "rollout_delta_norm": delta_norm,
            "rollout_base_norm": base_norm,
            "rollout_decomposition_expansion_ratio": expansion_ratio,
            "rollout_base_is_fixed_zero": base_norm.new_tensor(
                float(self.base_mode == "fixed_zero")
            ),
            "rollout_delta_gain": self.delta_gain.detach().float().abs(),
        }
