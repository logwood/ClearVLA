from __future__ import annotations

"""Current serial-owned hierarchical MMDiT action decoder."""

import math
from typing import Protocol

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .codec import PhysicalActionCodec, PhysicalActionTokenLift
from .evidence import HierarchicalEvidenceWorkspace
from .gauges import time_stratified_attention
from .intent import IndependentIntentFusion, IntentContractCompiler, PolicyConditionOrganizer
from .primitives import BiasFreeFFN, TimeEmbedding, sinusoidal_positions
from .refinement import NestedLowRankContractionBank


class PolicyDecoderConfig(Protocol):
    hidden_size: int
    action_horizon: int
    num_heads: int
    dropout: float
    arm_dim: int
    gripper_field_dim: int
    gripper_field_mode: str
    physical_action_dim: int
    hierarchical_mmdit_depth: int
    hierarchical_mmdit_refine_steps: int
    hierarchical_mmdit_low_slots: int
    hierarchical_mmdit_stage_slots: int
    hierarchical_mmdit_ffn_expansion: float
    hierarchical_mmdit_noisy_causal: int
    hierarchical_mmdit_noisy_gate_min: float
    hierarchical_mmdit_noisy_gate_power: float
    hierarchical_mmdit_stage_promote_scale_init: float
    hierarchical_mmdit_output_init_std: float
    hierarchical_mmdit_operator_stages: int
    hierarchical_mmdit_operator_rank: int
    hierarchical_mmdit_operator_groups: int
    hierarchical_mmdit_operator_depth_logit_init: float
    hierarchical_mmdit_operator_contraction_warmup_steps: int
    hierarchical_mmdit_operator_contraction_transition_steps: int
    hierarchical_mmdit_schedule_mode: str
    hierarchical_mmdit_random_prefix_probability: float
    hierarchical_mmdit_exhaustion_mode: str
    hierarchical_mmdit_action_response_thresholds: tuple[float, float, float]
    hierarchical_mmdit_stage_pressure_thresholds: tuple[float, float, float]
    hierarchical_mmdit_action_response_floor: float
    hierarchical_mmdit_exhaustion_confirm_steps: int
    hierarchical_mmdit_residual_scale_init: float
    hierarchical_mmdit_residual_scale_max: float
    hierarchical_mmdit_output_contract: int


class ConditionNeutralActionInitializer(nn.Module):
    """Causal horizon geometry that cannot inspect condition or x_t."""

    def __init__(self, config: PolicyDecoderConfig) -> None:
        super().__init__()
        h = int(config.hidden_size)
        horizon = int(config.action_horizon)
        self.hidden_size = h
        self.horizon = horizon
        self.seed = nn.Parameter(torch.randn(1, horizon, h) * 0.02)
        self.register_buffer(
            "horizon_position",
            sinusoidal_positions(range(1, horizon + 1), h)[None],
            persistent=True,
        )
        self.norm1 = nn.LayerNorm(h, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            h, int(config.num_heads), batch_first=True, dropout=float(config.dropout)
        )
        self.norm2 = nn.LayerNorm(h, elementwise_affine=False)
        self.ffn = BiasFreeFFN(h, float(config.hierarchical_mmdit_ffn_expansion))
        self.drop = nn.Dropout(float(config.dropout))
        # Start as pure horizon geometry.  The conditioner remains absent and
        # the two optional internal transforms earn their influence through
        # training instead of injecting random cold-start motion.
        self.attn_gate = nn.Parameter(torch.zeros(()))
        self.ffn_gate = nn.Parameter(torch.zeros(()))
        self.out_norm = nn.LayerNorm(h, elementwise_affine=False)

    def forward(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        x = self.seed.to(device=device, dtype=dtype).expand(batch_size, -1, -1)
        x = x + self.horizon_position.to(device=device, dtype=dtype)
        causal = torch.triu(
            torch.ones(self.horizon, self.horizon, device=device, dtype=torch.bool), diagonal=1
        )
        value = self.norm1(x)
        update, _ = self.attn(value, value, value, attn_mask=causal, need_weights=False)
        x = x + torch.tanh(self.attn_gate).to(dtype=dtype) * self.drop(update)
        x = x + torch.tanh(self.ffn_gate).to(dtype=dtype) * self.drop(self.ffn(self.norm2(x)))
        x = self.out_norm(x)
        return x, {
            "hierarchical_mmdit_initializer_norm": x.detach().float().norm(dim=-1).mean(),
            "hierarchical_mmdit_initializer_slot_diversity": (
                x.detach().float() - x.detach().float().mean(dim=1, keepdim=True)
            ).norm(dim=-1).mean(),
        }


class ActionOnlyPhysicalVelocityHead(nn.Module):
    """Typed physical velocity readout with no auxiliary correction latent."""

    def __init__(self, config: PolicyDecoderConfig) -> None:
        super().__init__()
        h = int(config.hidden_size)
        ad = int(config.arm_dim)
        self.norm = nn.LayerNorm(h)
        self.parseval_gripper = str(config.gripper_field_mode) == "parseval_temporal"
        if self.parseval_gripper:
            self.arm_field = nn.Linear(h, 2 * ad)
            self.grip_field = nn.Linear(h, int(config.gripper_field_dim))
        else:
            self.arm_abs = nn.Linear(h, ad)
            self.arm_delta = nn.Linear(h, ad)
            self.grip_value = nn.Linear(h, 1)
            self.grip_delta = nn.Linear(h, 1)
            self.grip_extra = nn.Linear(h, max(int(config.gripper_field_dim) - 2, 0))

    def output_layers(self) -> tuple[nn.Linear, ...]:
        if self.parseval_gripper:
            return self.arm_field, self.grip_field
        return self.arm_abs, self.arm_delta, self.grip_value, self.grip_delta, self.grip_extra

    def forward(self, tokens: Tensor) -> Tensor:
        x = self.norm(tokens)
        if self.parseval_gripper:
            return torch.cat([self.arm_field(x), self.grip_field(x)], dim=-1)
        parts = [self.arm_abs(x), self.arm_delta(x), self.grip_value(x), self.grip_delta(x)]
        if int(self.grip_extra.out_features) > 0:
            parts.append(self.grip_extra(x))
        return torch.cat(parts, dim=-1)


class OwnedHierarchicalActionBlock(nn.Module):
    """A V77 refinement block with a post-gate contraction sidecar.

    The host block owns content, direction, AdaLN, and its original LayerScale
    gates.  Semantic stages can only remove components from the complete gated
    branch update; they never receive a second residual-amplitude control.
    """

    _BRANCH_NAMES = ("self", "noisy", "stage", "low", "ffn")

    def __init__(self, config: PolicyDecoderConfig, *, operator_stage_count: int) -> None:
        super().__init__()
        h = int(config.hidden_size)
        heads = int(config.num_heads)
        if h % heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.config = config
        self.hidden_size = h
        self.heads = heads
        self.head_dim = h // heads
        self.operator_stage_count = int(operator_stage_count)
        if self.operator_stage_count < 1:
            raise ValueError("each refinement block must own at least one operator stage")
        self.state_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.condition_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.global_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.self_qkv = nn.Linear(h, 3 * h)
        self.self_out = nn.Linear(h, h)
        self.cross_q = nn.Linear(h, h)
        self.noisy_kv = nn.Linear(h, 2 * h)
        self.stage_kv = nn.Linear(h, 2 * h)
        self.low_kv = nn.Linear(h, 2 * h)
        self.noisy_out = nn.Linear(h, h)
        self.stage_out = nn.Linear(h, h)
        self.low_out = nn.Linear(h, h)
        self.ffn = BiasFreeFFN(h, float(config.hierarchical_mmdit_ffn_expansion))
        self.out_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.scale_max = float(config.hierarchical_mmdit_residual_scale_max)
        # Preserve the V77 shared AdaLN/LayerScale owner.  These five rows are
        # part of the host function, not a sidecar control surface.
        self.mod = nn.Linear(h, 2 * h + len(self._BRANCH_NAMES))
        self.drop = nn.Dropout(float(config.dropout))
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)
        # Preserve the established V77 profile when residual_scale_init=0.05.
        base_step = float(config.hierarchical_mmdit_residual_scale_init)
        initial_steps = torch.tensor((
            0.4 * base_step,
            1.6 * base_step,
            0.8 * base_step,
            1.2 * base_step,
            0.4 * base_step,
        ), dtype=torch.float32)
        with torch.no_grad():
            for offset, initial in enumerate(initial_steps.tolist()):
                ratio = min(max(initial / self.scale_max, -0.999), 0.999)
                self.mod.bias[2 * h + offset] = math.atanh(ratio)

    @staticmethod
    def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        return x * (1.0 + scale[:, None]) + shift[:, None]

    @staticmethod
    def _sample_rms(x: Tensor) -> Tensor:
        return x.float().square().mean(dim=(1, 2)).sqrt()

    @staticmethod
    def _base_project(projection: nn.Linear, feature: Tensor) -> tuple[Tensor, Tensor]:
        weight = projection.weight
        target_norm = math.sqrt(float(projection.out_features))
        raw_parameter_rms = weight.float().square().sum().sqrt() / target_norm
        return projection(feature), raw_parameter_rms.detach()

    @classmethod
    def _normalize_residual(cls, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Preserve the V77 residual direction at the contraction boundary."""
        raw = x.float()
        denominator = raw.square().mean(dim=(1, 2), keepdim=True).add(1e-6).sqrt()
        direction = (raw / denominator).to(dtype=x.dtype)
        return (
            direction,
            cls._sample_rms(x).detach(),
            cls._sample_rms(direction).detach(),
        )

    def _split_heads(self, x: Tensor) -> Tensor:
        b, n, h = x.shape
        return x.reshape(b, n, self.heads, h // self.heads).transpose(1, 2)

    @staticmethod
    def _merge_heads(x: Tensor) -> Tensor:
        b, heads, n, d = x.shape
        return x.transpose(1, 2).reshape(b, n, heads * d)

    @staticmethod
    def _attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        score = torch.matmul(q.float(), k.float().transpose(-2, -1)) * (float(q.shape[-1]) ** -0.5)
        if mask is not None:
            score = score.masked_fill(
                mask.to(device=score.device, dtype=torch.bool)[None, None],
                torch.finfo(score.dtype).min,
            )
        weight = torch.softmax(score, dim=-1).to(dtype=q.dtype)
        return torch.matmul(weight, v), weight

    @staticmethod
    def _row_norm(x: Tensor) -> Tensor:
        return x.detach().float().norm(dim=-1).mean(dim=1)

    def _compose_update(
        self,
        *,
        branch: str,
        projection_input: Tensor,
        base_projection: nn.Linear,
        contraction: NestedLowRankContractionBank,
        contraction_progress: Tensor,
        operator_cond: Tensor,
        stage_index: Tensor,
        base_gate: Tensor,
        contraction_identity_bypass: bool | None = None,
        stage_candidates: Tensor | None = None,
        stage_probabilities: Tensor | None = None,
        prepared_factors: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        base_output, base_parameter_rms = self._base_project(
            base_projection, projection_input
        )
        projection_rms_rows = self._sample_rms(projection_input.detach())
        # Reconstruct the complete V77 branch first.  The sidecar receives the
        # already gated update, so identity mode preserves both its value and
        # Jacobian and contraction cannot create a new amplitude owner.
        base_activated = self.drop(base_output)
        base_direction, base_activated_rms_rows, base_normalized_rms_rows = (
            self._normalize_residual(base_activated)
        )
        host_update = (
            base_gate[:, None, None].to(dtype=base_direction.dtype)
            * base_direction
        )
        update, contraction_metrics = contraction(
            host_update,
            operator_cond,
            stage_index,
            stage_candidates=stage_candidates,
            stage_probabilities=stage_probabilities,
            contraction_progress=contraction_progress,
            prepared_factors=prepared_factors,
            identity_bypass=contraction_identity_bypass,
        )
        with torch.no_grad():
            base_rms_rows = self._sample_rms(base_output)
            host_update_rms_rows = self._sample_rms(host_update)
            update_rms_rows = self._sample_rms(update)
            direction_change_rows = self._sample_rms(
                update - host_update
            )
            direction_cosine_rows = F.cosine_similarity(
                update.float(),
                host_update.float(),
                dim=-1,
            ).mean(dim=1)
            expected_host_rms_rows = (
                base_gate.detach().float().abs() * base_normalized_rms_rows
            )
            gate_scale_error_rows = (
                host_update_rms_rows - expected_host_rms_rows
            ).abs()
        metrics = dict(contraction_metrics)
        metrics.update({
            "base_rms": base_rms_rows.detach().mean(),
            "base_parameter_rms": base_parameter_rms,
            "projection_input_rms": projection_rms_rows.detach().mean(),
            "base_data_gain": (
                base_rms_rows / projection_rms_rows.clamp_min(1e-6)
            ).detach().mean(),
            "base_activated_rms": base_activated_rms_rows.detach().mean(),
            "base_normalized_rms": base_normalized_rms_rows.detach().mean(),
            "host_update_rms": host_update_rms_rows.detach().mean(),
            "contracted_rms": update_rms_rows.detach().mean(),
            "direction_change": direction_change_rows.detach().mean(),
            "direction_cosine": direction_cosine_rows.detach().mean(),
            "base_gate": base_gate.detach().mean(),
            "base_gate_abs_mean": base_gate.detach().float().abs().mean(),
            "gate_scale_error": gate_scale_error_rows.detach().mean(),
            "realized_scale": update_rms_rows.detach().mean(),
            "operator_gain": (
                update_rms_rows / projection_rms_rows.clamp_min(1e-6)
            ).detach().mean(),
            "update_rms": update_rms_rows.detach().mean(),
            "host_update_rms_rows": host_update_rms_rows.detach(),
            "base_gate_rows": base_gate.detach(),
            "operator_gain_rows": (
                update_rms_rows / projection_rms_rows.clamp_min(1e-6)
            ).detach(),
            "update_rms_rows": update_rms_rows.detach(),
            "direction_change_rows": direction_change_rows.detach(),
        })
        return update, metrics

    @classmethod
    def _branch_geometry(
        cls,
        updates: tuple[Tensor, ...],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Measure alignment against the unequal-norm orthogonal baseline."""
        if not updates:
            raise ValueError("branch geometry requires at least one update")
        detached_updates = tuple(update.detach().float() for update in updates)
        branch_token_norms = torch.stack(
            [update.norm(dim=-1) for update in detached_updates], dim=2
        )
        branch_rows = branch_token_norms.mean(dim=1)
        branch_sum = detached_updates[0]
        for update in detached_updates[1:]:
            branch_sum = branch_sum + update
        denominator = branch_rows.sum(dim=-1)
        valid = denominator > 1e-8
        net_rows = branch_sum.norm(dim=-1).mean(dim=1)
        # Preserve token-local amplitude structure: under orthogonality the
        # expected net norm is sqrt(sum_i ||u_i,t||^2) at each horizon token,
        # not sqrt(sum_i mean_t(||u_i,t||)^2).
        orthogonal_rows = branch_token_norms.square().sum(dim=2).sqrt().mean(dim=1)
        cancellation_rows = torch.where(
            valid,
            1.0 - net_rows / denominator.clamp_min(1e-8),
            torch.zeros_like(denominator),
        ).clamp(0.0, 1.0)
        orthogonal_baseline_rows = torch.where(
            valid,
            1.0 - orthogonal_rows / denominator.clamp_min(1e-8),
            torch.zeros_like(denominator),
        ).clamp(0.0, 1.0)

        token_norm_sum = branch_token_norms.sum(dim=2)
        pair_denominator = (
            token_norm_sum.square() - branch_token_norms.square().sum(dim=2)
        )
        pair_numerator = (
            branch_sum.square().sum(dim=-1)
            - torch.stack(
                [update.square().sum(dim=-1) for update in detached_updates], dim=2
            ).sum(dim=2)
        )
        weighted_pair_cosine = torch.where(
            pair_denominator > 1e-8,
            pair_numerator / pair_denominator.clamp_min(1e-8),
            torch.zeros_like(pair_denominator),
        ).clamp(-1.0, 1.0).mean()
        return (
            branch_rows,
            branch_sum,
            cancellation_rows.mean(),
            orthogonal_baseline_rows.mean(),
            (cancellation_rows - orthogonal_baseline_rows).mean(),
            weighted_pair_cosine,
        )

    @staticmethod
    def _attention_stats(weight: Tensor) -> tuple[Tensor, Tensor]:
        prob = weight.detach().float().clamp_min(1e-8)
        entropy = -(prob * prob.log()).sum(dim=-1).mean()
        maximum = prob.max(dim=-1).values.mean()
        return entropy, maximum

    def _cross_update(
        self,
        action: Tensor,
        memory: Tensor,
        *,
        branch: str,
        kv_proj: nn.Linear,
        out_proj: nn.Linear,
        contraction: NestedLowRankContractionBank,
        contraction_progress: Tensor,
        shift: Tensor,
        scale: Tensor,
        operator_cond: Tensor,
        stage_index: Tensor,
        base_gate: Tensor,
        contraction_identity_bypass: bool | None = None,
        stage_candidates: Tensor | None = None,
        stage_probabilities: Tensor | None = None,
        mask: Tensor | None = None,
        prepared_factors: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        query_value = self._modulate(self.state_norm(action), shift, scale)
        q = self._split_heads(self.cross_q(query_value))
        key, value = kv_proj(self.condition_norm(memory)).chunk(2, dim=-1)
        k = self._split_heads(key)
        v = self._split_heads(value)
        attended, weight = self._attention(q, k, v, mask)
        projection_input = self._merge_heads(attended)
        update, operator_metrics = self._compose_update(
            branch=branch,
            projection_input=projection_input,
            base_projection=out_proj,
            contraction=contraction,
            contraction_progress=contraction_progress,
            operator_cond=operator_cond,
            stage_index=stage_index,
            stage_candidates=stage_candidates,
            stage_probabilities=stage_probabilities,
            base_gate=base_gate,
            contraction_identity_bypass=contraction_identity_bypass,
            prepared_factors=prepared_factors,
        )
        return action + update, update, weight, operator_metrics

    def forward(
        self,
        action: Tensor,
        *,
        noisy_tokens: Tensor,
        stage_tokens: Tensor,
        low_tokens: Tensor,
        shared_cond: Tensor,
        operator_cond: Tensor,
        contractions: nn.ModuleDict,
        contraction_progress: Tensor,
        stage_index: Tensor,
        contraction_identity_bypass: bool | None = None,
        stage_candidates: Tensor | None = None,
        stage_probabilities: Tensor | None = None,
        contraction_factors: dict[str, Tensor] | None = None,
        low_role_ids: Tensor | None = None,
        low_role_names: tuple[str, ...] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        before = action
        normalized_shared = self.global_norm(shared_cond)
        mod = self.mod(normalized_shared)
        shift = 0.5 * torch.tanh(mod[:, :self.hidden_size])
        scale = 0.5 * torch.tanh(
            mod[:, self.hidden_size:2 * self.hidden_size]
        )
        gates = self.scale_max * torch.tanh(
            mod[:, 2 * self.hidden_size:]
        )
        if contraction_progress.ndim != 0:
            raise ValueError("contraction_progress must be scalar")

        value = self._modulate(self.state_norm(action), shift, scale)
        sq, sk, sv = (self._split_heads(part) for part in self.self_qkv(value).chunk(3, dim=-1))
        self_mask = torch.triu(
            torch.ones(int(action.shape[1]), int(action.shape[1]), device=action.device, dtype=torch.bool),
            diagonal=1,
        )
        self_attended, self_weight = self._attention(sq, sk, sv, self_mask)
        self_projection_input = self._merge_heads(self_attended)
        self_update, self_operator = self._compose_update(
            branch="self",
            projection_input=self_projection_input,
            base_projection=self.self_out,
            contraction=contractions["self"],
            contraction_progress=contraction_progress,
            operator_cond=operator_cond,
            stage_index=stage_index,
            stage_candidates=stage_candidates,
            stage_probabilities=stage_probabilities,
            base_gate=gates[:, 0],
            contraction_identity_bypass=contraction_identity_bypass,
            prepared_factors=None if contraction_factors is None else contraction_factors["self"],
        )
        action = action + self_update

        noisy_mask = None
        if bool(int(self.config.hierarchical_mmdit_noisy_causal)):
            action_pos = torch.arange(int(action.shape[1]), device=action.device)[:, None]
            noisy_pos = torch.arange(int(noisy_tokens.shape[1]), device=action.device)[None]
            noisy_mask = noisy_pos > action_pos
        (
            action,
            noisy_update,
            noisy_weight,
            noisy_operator,
        ) = self._cross_update(
            action,
            noisy_tokens,
            branch="noisy",
            kv_proj=self.noisy_kv,
            out_proj=self.noisy_out,
            contraction=contractions["noisy"],
            contraction_progress=contraction_progress,
            shift=shift,
            scale=scale,
            operator_cond=operator_cond,
            stage_index=stage_index,
            stage_candidates=stage_candidates,
            stage_probabilities=stage_probabilities,
            base_gate=gates[:, 1],
            contraction_identity_bypass=contraction_identity_bypass,
            mask=noisy_mask,
            prepared_factors=None if contraction_factors is None else contraction_factors["noisy"],
        )
        (
            action,
            stage_update,
            stage_weight,
            stage_operator,
        ) = self._cross_update(
            action,
            stage_tokens,
            branch="stage",
            kv_proj=self.stage_kv,
            out_proj=self.stage_out,
            contraction=contractions["stage"],
            contraction_progress=contraction_progress,
            shift=shift,
            scale=scale,
            operator_cond=operator_cond,
            stage_index=stage_index,
            stage_candidates=stage_candidates,
            stage_probabilities=stage_probabilities,
            base_gate=gates[:, 2],
            contraction_identity_bypass=contraction_identity_bypass,
            prepared_factors=None if contraction_factors is None else contraction_factors["stage"],
        )
        (
            action,
            low_update,
            low_weight,
            low_operator,
        ) = self._cross_update(
            action,
            low_tokens,
            branch="low",
            kv_proj=self.low_kv,
            out_proj=self.low_out,
            contraction=contractions["low"],
            contraction_progress=contraction_progress,
            shift=shift,
            scale=scale,
            operator_cond=operator_cond,
            stage_index=stage_index,
            stage_candidates=stage_candidates,
            stage_probabilities=stage_probabilities,
            base_gate=gates[:, 3],
            contraction_identity_bypass=contraction_identity_bypass,
            prepared_factors=None if contraction_factors is None else contraction_factors["low"],
        )
        ffn_value = self._modulate(self.state_norm(action), shift, scale)
        ffn_hidden = self.ffn.net[1](self.ffn.net[0](ffn_value))
        ffn_update, ffn_operator = self._compose_update(
            branch="ffn",
            projection_input=ffn_hidden,
            base_projection=self.ffn.net[2],
            contraction=contractions["ffn"],
            contraction_progress=contraction_progress,
            operator_cond=operator_cond,
            stage_index=stage_index,
            stage_candidates=stage_candidates,
            stage_probabilities=stage_probabilities,
            base_gate=gates[:, 4],
            contraction_identity_bypass=contraction_identity_bypass,
            prepared_factors=None if contraction_factors is None else contraction_factors["ffn"],
        )
        pre_norm_action = action + ffn_update
        pre_norm_rms = self._sample_rms(pre_norm_action).detach()
        action = self.out_norm(pre_norm_action)

        updates = (self_update, noisy_update, stage_update, low_update, ffn_update)
        (
            branch_rows,
            branch_sum,
            serial_cancellation,
            orthogonal_baseline,
            cancellation_excess,
            weighted_pair_cosine,
        ) = self._branch_geometry(updates)
        fractions = branch_rows / branch_rows.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        total_rows = self._row_norm(action - before)
        before_rows = self._row_norm(before).clamp_min(1e-6)
        self_entropy, self_max = self._attention_stats(self_weight)
        noisy_entropy, noisy_max = self._attention_stats(noisy_weight)
        stage_entropy, stage_max = self._attention_stats(stage_weight)
        low_entropy, low_max = self._attention_stats(low_weight)
        metrics: dict[str, Tensor] = {
            "action_update_norm": total_rows.mean(),
            "action_update_ratio": (total_rows / before_rows).mean(),
            "action_pre_norm_rms": pre_norm_rms.mean(),
            "action_post_norm_rms": self._sample_rms(action).detach().mean(),
            "action_sidecar_progress": contraction_progress.detach(),
            "action_serial_cancellation_fraction": serial_cancellation,
            "action_serial_cancellation_orthogonal_baseline": orthogonal_baseline,
            "action_serial_cancellation_excess": cancellation_excess,
            "action_branch_weighted_cosine": weighted_pair_cosine,
            "action_state_cosine": F.cosine_similarity(
                action.detach().float(), before.detach().float(), dim=-1
            ).mean(),
            "action_noisy_stage_cosine": F.cosine_similarity(
                noisy_update.detach().float(), stage_update.detach().float(), dim=-1
            ).mean(),
            "action_stage_low_cosine": F.cosine_similarity(
                stage_update.detach().float(), low_update.detach().float(), dim=-1
            ).mean(),
            "action_noisy_low_cosine": F.cosine_similarity(
                noisy_update.detach().float(), low_update.detach().float(), dim=-1
            ).mean(),
            "action_self_update_norm": branch_rows[:, 0].mean(),
            "action_noisy_update_norm": branch_rows[:, 1].mean(),
            "action_stage_update_norm": branch_rows[:, 2].mean(),
            "action_low_update_norm": branch_rows[:, 3].mean(),
            "action_ffn_update_norm": branch_rows[:, 4].mean(),
            "action_noisy_update_fraction": fractions[:, 1].mean(),
            "action_stage_update_fraction": fractions[:, 2].mean(),
            "action_low_update_fraction": fractions[:, 3].mean(),
            "action_noisy_update_fraction_rows": fractions[:, 1],
            "action_stage_update_fraction_rows": fractions[:, 2],
            "action_low_update_fraction_rows": fractions[:, 3],
            "action_workspace_update_fraction_rows": fractions[:, 2] + fractions[:, 3],
            "action_self_attention_entropy": self_entropy,
            "action_self_attention_max": self_max,
            "action_noisy_attention_entropy": noisy_entropy,
            "action_noisy_attention_max": noisy_max,
            "action_stage_attention_entropy": stage_entropy,
            "action_stage_attention_max": stage_max,
            "action_low_attention_entropy": low_entropy,
            "action_low_attention_max": low_max,
        }
        operator_rows = {
            "self": self_operator,
            "noisy": noisy_operator,
            "stage": stage_operator,
            "low": low_operator,
            "ffn": ffn_operator,
        }
        for name, operator_metrics in operator_rows.items():
            for metric_name in (
                "depth_ratio", "depth_ratio_min", "depth_ratio_max",
                "raw_depth_ratio", "effective_depth", "available_depth",
                "transparency_mean", "transparency_min", "transparency_max",
                "contraction_progress", "depth_usage_cost",
                "contraction_ratio", "subspace_energy_fraction",
                "removed_fraction", "removed_rms",
                "boundary_identity_error", "nonexpansive_violation",
                "nested_order_violation", "basis_norm_error",
                "basis_orthogonality_error", "basis_raw_norm", "operator_gain",
                "update_rms", "base_rms", "base_parameter_rms",
                "projection_input_rms", "base_data_gain", "base_activated_rms",
                "base_normalized_rms", "host_update_rms", "contracted_rms",
                "direction_change", "direction_cosine", "base_gate",
                "base_gate_abs_mean", "gate_scale_error", "realized_scale",
            ):
                metrics[f"action_{name}_{metric_name}"] = operator_metrics[metric_name]
            for metric_name in (
                "depth_ratio_rows", "effective_depth_rows",
                "contraction_ratio_rows", "subspace_energy_fraction_rows",
                "removed_fraction_rows",
                "nonexpansive_violation_rows",
                "operator_gain_rows", "update_rms_rows", "host_update_rms_rows",
                "base_gate_rows",
                "direction_change_rows",
            ):
                metrics[f"action_{name}_{metric_name}"] = operator_metrics[metric_name]
        if low_role_ids is not None:
            role_ids = low_role_ids.to(device=low_weight.device, dtype=torch.long).reshape(-1)
            if int(role_ids.numel()) != int(low_weight.shape[-1]):
                raise ValueError(
                    "low_role_ids must match low token count: "
                    f"{int(role_ids.numel())} vs {int(low_weight.shape[-1])}"
                )
            role_names = tuple(low_role_names or ())
            role_count = len(role_names)
            if role_count <= 0:
                raise ValueError("low_role_names cannot be empty when low_role_ids are provided")
            role_rows = torch.stack([
                low_weight.detach().float()[..., role_ids == role_index].sum(dim=-1)
                for role_index in range(role_count)
            ], dim=-1)
            role_prob = role_rows.clamp_min(1e-8)
            role_entropy = -(role_prob * role_prob.log()).sum(dim=-1).mean()
            metrics["action_low_role_entropy"] = role_entropy
            metrics["action_low_role_effective_count"] = torch.exp(role_entropy)
            metrics["action_low_role_max"] = role_rows.max(dim=-1).values.mean()
            for role_index, role in enumerate(role_names):
                metrics[f"action_low_role_{role}_attention"] = role_rows[..., role_index].mean()
        return action, metrics


class HierarchicalMMDiTActionDecoder(nn.Module):
    """Owned evidence plus condition-modulated stage-adaptive refinement."""

    def __init__(self, config: PolicyDecoderConfig) -> None:
        super().__init__()
        self.config = config
        h = int(config.hidden_size)
        self.hidden_size = h
        self.refine_block_count = int(config.hierarchical_mmdit_depth)
        self.operator_stage_count = int(config.hierarchical_mmdit_operator_stages)
        self.refine_steps = int(config.hierarchical_mmdit_refine_steps)
        self.organizer = PolicyConditionOrganizer(config)
        self.intent_compiler = IntentContractCompiler(config)
        self.action_initializer = ConditionNeutralActionInitializer(config)
        self.time = TimeEmbedding(h)
        self.time_lift = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
        parseval_gripper = str(config.gripper_field_mode) == "parseval_temporal"
        self.noisy_action_lift = (
            PhysicalActionTokenLift(config)
            if parseval_gripper
            else nn.Sequential(
                nn.LayerNorm(int(config.physical_action_dim)),
                nn.Linear(int(config.physical_action_dim), h),
            )
        )
        self.workspace = HierarchicalEvidenceWorkspace(
            config,
            owned_evidence=True,
            manage_output_strength=False,
            contract_conditioning=True,
            stratified_roles=True,
            low_count=int(config.hierarchical_mmdit_low_slots),
            stage_count=int(config.hierarchical_mmdit_stage_slots),
            refine_steps=self.refine_steps,
            ffn_expansion=float(config.hierarchical_mmdit_ffn_expansion),
            # Low slots are semantic shelves, not horizon positions.  Only
            # cross-role communication is blocked; an arbitrary causal order
            # inside one role would throw away usable evidence capacity.
            causal_attention=False,
            stage_promote_scale_init=float(config.hierarchical_mmdit_stage_promote_scale_init),
        )
        # This decoder owns progress through refine_block_identity + budget_proj and
        # always supplies step_state_override. Keep the fallback parameter in
        # the state dict for checkpoint compatibility, but do not optimize an
        # unreachable second definition of decoder time.
        self.workspace.step_embedding.requires_grad_(False)
        self.blocks = nn.ModuleList([
            OwnedHierarchicalActionBlock(
                config,
                operator_stage_count=(
                    ((block_index + 1) * self.operator_stage_count)
                    // self.refine_block_count
                    - (block_index * self.operator_stage_count)
                    // self.refine_block_count
                ),
            )
            for block_index in range(self.refine_block_count)
        ])
        self.refine_block_identity = nn.Parameter(
            torch.randn(1, self.refine_block_count, h) * 0.02
        )
        # New selector/sidecar capacity uses an isolated CPU RNG stream.  The
        # V77 host modules below therefore keep their original initialization
        # sequence when the sidecar is added or removed.
        host_rng_state = torch.get_rng_state()
        sidecar_generator = torch.Generator(device="cpu")
        sidecar_generator.manual_seed(
            (int(torch.initial_seed()) ^ 0x5A17C0DE) % (2**63 - 1)
        )
        try:
            torch.set_rng_state(sidecar_generator.get_state())
            self.operator_stage_identity = nn.Parameter(
                torch.randn(1, self.operator_stage_count, h) * 0.02
            )
            self.stage_selector_control = nn.Sequential(
                nn.LayerNorm(4),
                nn.Linear(4, h, bias=False),
                nn.SiLU(),
                nn.Linear(h, h, bias=False),
            )
            self.stage_selector_query = nn.Sequential(
                nn.LayerNorm(5 * h),
                nn.Linear(5 * h, h, bias=False),
                nn.SiLU(),
                nn.Linear(h, h, bias=False),
            )
            self.exit_controller = nn.Sequential(
                nn.LayerNorm(h),
                nn.Linear(h, h, bias=False),
                nn.SiLU(),
                nn.Linear(h, 1),
            )
            nn.init.zeros_(self.exit_controller[-1].weight)
            nn.init.zeros_(self.exit_controller[-1].bias)
            self.operator_condition = nn.Sequential(
                nn.LayerNorm(3 * h),
                nn.Linear(3 * h, h),
                nn.SiLU(),
                nn.Linear(h, h),
                nn.LayerNorm(h, elementwise_affine=False),
            )
            sidecar_rng_state = torch.get_rng_state()
        finally:
            torch.set_rng_state(host_rng_state)
        self.budget_proj = nn.Sequential(nn.Linear(2, h), nn.SiLU(), nn.Linear(h, h))
        self.step_state_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.workspace_condition = nn.Sequential(
            nn.LayerNorm(2 * h),
            nn.Linear(2 * h, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.LayerNorm(h, elementwise_affine=False),
        )
        self.shared_condition = nn.Sequential(
            nn.LayerNorm(3 * h),
            nn.Linear(3 * h, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.LayerNorm(h, elementwise_affine=False),
        )
        self.condition_type = nn.Parameter(torch.randn(1, 3, h) * 0.02)
        self.action_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.velocity_head = ActionOnlyPhysicalVelocityHead(config)
        self.response_codec = PhysicalActionCodec(config)
        self.event_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 3))
        self.motion_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 1))
        # CR7 fallback (do_before_v76 item 23): a dedicated, restricted output
        # contract for the event/motion subheads only.  It shares NO layer with
        # the global intent contract, never touches the velocity head, and its
        # injection projection is zero-initialized so flag=1 starts exactly at
        # the action-only behavior.  Prepared in advance so a gripper-event
        # regression at E3 costs a flag flip, not a coding window.
        if int(getattr(config, "hierarchical_mmdit_output_contract", 0)):
            self.output_contract_fusion = IndependentIntentFusion(h, 3)
            self.output_contract_proj = nn.Linear(h, h)
            nn.init.zeros_(self.output_contract_proj.weight)
            nn.init.zeros_(self.output_contract_proj.bias)
        else:
            self.output_contract_fusion = None
            self.output_contract_proj = None
        self._initialize_outputs()
        # Contraction is a sidecar around an already-complete refinement
        # function.  Build it after the base decoder and restore CPU RNG state
        # so adding/removing contraction capacity cannot silently reinitialize
        # unrelated policy modules constructed later.
        host_rng_state = torch.get_rng_state()
        try:
            torch.set_rng_state(sidecar_rng_state)
            self.operator_contractions = nn.ModuleList([
                nn.ModuleDict({
                    name: NestedLowRankContractionBank(
                        hidden_size=h,
                        condition_size=h,
                        stage_count=block.operator_stage_count,
                        rank=int(config.hierarchical_mmdit_operator_rank),
                        group_count=int(config.hierarchical_mmdit_operator_groups),
                        depth_logit_init=float(
                            config.hierarchical_mmdit_operator_depth_logit_init
                        ),
                    )
                    for name in OwnedHierarchicalActionBlock._BRANCH_NAMES
                })
                for block in self.blocks
            ])
        finally:
            torch.set_rng_state(host_rng_state)
        self.register_buffer(
            "contraction_progress",
            torch.zeros((), dtype=torch.float32),
            persistent=True,
        )
        self._contraction_progress_value = 0.0

    def _initialize_outputs(self) -> None:
        std = float(self.config.hierarchical_mmdit_output_init_std)
        for module in self.velocity_head.output_layers():
            if std > 0.0:
                nn.init.normal_(module.weight, mean=0.0, std=std)
            else:
                nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)
        for head in (self.event_head, self.motion_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        # Keep the Python fast-path flag synchronized with the persistent
        # schedule buffer for evaluation/resume loads that do not immediately
        # call set_operator_contraction_training_step().
        self._contraction_progress_value = float(
            self.contraction_progress.detach().cpu()
        )

    def _step_state(
        self,
        block_index: Tensor,
        *,
        progress_fraction: float,
        remaining_fraction: float,
        dtype: torch.dtype,
    ) -> Tensor:
        if block_index.ndim != 1:
            raise ValueError("block_index must be one-dimensional")
        block_index = block_index.to(dtype=torch.long)
        if block_index.device.type == "cpu":
            if bool((block_index < 0).any()) or bool(
                (block_index >= self.refine_block_count).any()
            ):
                raise ValueError("block_index is outside the refinement block repertoire")
        batch_size = int(block_index.shape[0])
        device = block_index.device
        budget = torch.tensor(
            [
                min(max(float(progress_fraction), 0.0), 1.0),
                min(max(float(remaining_fraction), 0.0), 1.0),
            ],
            device=device,
            dtype=dtype,
        )[None].expand(batch_size, -1)
        identities = self.refine_block_identity[0].to(
            device=device, dtype=dtype
        ).index_select(0, block_index)
        return self.step_state_norm(identities + self.budget_proj(budget))

    @staticmethod
    def _gate_noisy_tokens(noisy: Tensor, time: Tensor) -> tuple[Tensor, Tensor]:
        """V82 keeps noisy evidence full-strength; time remains an explicit token."""
        del time
        return noisy, torch.ones((), device=noisy.device, dtype=torch.float32)

    def factor_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(
            parameter
            for bank in self.operator_contractions
            for contraction in bank.values()
            for parameter in contraction.factor_parameters()
        )

    def contraction_control_parameters(self) -> tuple[nn.Parameter, ...]:
        return (
            *(
                parameter
                for bank in self.operator_contractions
                for contraction in bank.values()
                for parameter in contraction.control_parameters()
            ),
            self.operator_stage_identity,
            *tuple(self.stage_selector_control.parameters()),
            *tuple(self.stage_selector_query.parameters()),
            *tuple(self.exit_controller.parameters()),
        )

    def stage_selector_parameters(self) -> tuple[nn.Parameter, ...]:
        return (
            self.operator_stage_identity,
            *tuple(self.stage_selector_control.parameters()),
            *tuple(self.stage_selector_query.parameters()),
        )

    def exit_controller_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.exit_controller.parameters())

    def scale_invariant_base_parameters(self) -> tuple[nn.Parameter, ...]:
        return ()

    def set_operator_contraction_training_step(self, global_step: int) -> float:
        warmup = int(self.config.hierarchical_mmdit_operator_contraction_warmup_steps)
        transition = int(self.config.hierarchical_mmdit_operator_contraction_transition_steps)
        progress = min(max((int(global_step) - warmup) / float(transition), 0.0), 1.0)
        self.contraction_progress.fill_(progress)
        self._contraction_progress_value = progress
        return progress

    def prepare_contraction_factors(self) -> tuple[dict[str, Tensor], ...]:
        return tuple({
            name: contraction.prepare_factors()
            for name, contraction in bank.items()
        } for bank in self.operator_contractions)

    def _block_for_step(self, step_index: int) -> int:
        return min(
            (max(int(step_index), 0) * self.refine_block_count)
            // max(self.refine_steps, 1),
            self.refine_block_count - 1,
        )

    def _stage_bounds(self, block_index: int) -> tuple[int, int]:
        block_index = min(max(int(block_index), 0), self.refine_block_count - 1)
        start = (block_index * self.operator_stage_count) // self.refine_block_count
        stop = ((block_index + 1) * self.operator_stage_count) // self.refine_block_count
        return start, max(stop, start + 1)

    def _fixed_stage_candidates(self, block_index: int, *, device: torch.device) -> Tensor:
        """Return the local semantic-stage shelf owned by one refine block."""
        start, stop = self._stage_bounds(block_index)
        return torch.arange(start, min(stop, self.operator_stage_count), device=device)

    def _operator_stage_to_block(self, stage_index: Tensor) -> Tensor:
        """Map contraction ownership to block identity without leaking stage content."""
        # Invert the floor-partition used by _stage_bounds. Multiplying the
        # stage id directly fails when stage_count is not divisible by depth.
        return torch.div(
            (stage_index.to(dtype=torch.long) + 1) * self.refine_block_count - 1,
            self.operator_stage_count,
            rounding_mode="floor",
        ).clamp_max(self.refine_block_count - 1)

    def _operator_stage_to_local(self, stage_index: Tensor, block_index: Tensor) -> Tensor:
        """Translate global semantic-stage ids into each block's local bank."""
        starts = torch.div(
            block_index.to(dtype=torch.long) * self.operator_stage_count,
            self.refine_block_count,
            rounding_mode="floor",
        )
        return stage_index.to(dtype=torch.long) - starts

    def _stage_selector_action_summary(self, action: Tensor) -> Tensor:
        """Read live action state without giving routing gradients ownership of it."""
        if action.ndim != 3 or int(action.shape[-1]) != self.hidden_size:
            raise ValueError(
                f"stage selector action must be [B,N,{self.hidden_size}], "
                f"got {tuple(action.shape)}"
            )
        normalized = F.layer_norm(
            action.detach().float(), (self.hidden_size,)
        )
        return normalized.mean(dim=1)

    def _controller_query(
        self,
        *,
        global_intent: Tensor,
        time_state: Tensor,
        step_state: Tensor,
        action: Tensor,
        control_state: Tensor,
    ) -> Tensor:
        """Encode read-only live state shared by stage selection and exit control."""
        batch = int(action.shape[0])
        expected_hidden = (batch, self.hidden_size)
        for name, value in (
            ("global_intent", global_intent),
            ("time_state", time_state),
            ("step_state", step_state),
        ):
            if tuple(value.shape) != expected_hidden:
                raise ValueError(
                    f"controller {name} must be {expected_hidden}, got {tuple(value.shape)}"
                )
        if tuple(control_state.shape) != (batch, 4):
            raise ValueError(
                f"controller control_state must be {(batch, 4)}, "
                f"got {tuple(control_state.shape)}"
            )
        action_state = self._stage_selector_action_summary(action)
        control_context = self.stage_selector_control(
            control_state.detach().float()
        )
        return F.normalize(
            self.stage_selector_query(torch.cat([
                global_intent.detach().float(),
                time_state.detach().float(),
                step_state.detach().float(),
                action_state,
                control_context.float(),
            ], dim=-1)).float(),
            dim=-1,
        )

    def _exit_logit(self, controller_query: Tensor) -> Tensor:
        """Apply the exit head without transferring gradient ownership."""
        if (
            controller_query.ndim != 2
            or int(controller_query.shape[-1]) != self.hidden_size
        ):
            raise ValueError(
                "exit controller query must be [B,H], got "
                f"{tuple(controller_query.shape)}"
            )
        return self.exit_controller(controller_query.detach()).squeeze(-1)

    def _select_owned_stage(
        self,
        *,
        block_index: int,
        stage_content: Tensor,
        global_intent: Tensor,
        time_state: Tensor,
        step_state: Tensor,
        action: Tensor,
        control_state: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        candidates_1d = self._fixed_stage_candidates(block_index, device=stage_content.device)
        batch = int(stage_content.shape[0])
        candidates = candidates_1d[None].expand(batch, -1)
        query = self._controller_query(
            global_intent=global_intent,
            time_state=time_state,
            step_state=step_state,
            action=action,
            control_state=control_state,
        )
        identity = self.operator_stage_identity[:, candidates_1d].to(
            device=stage_content.device, dtype=stage_content.dtype
        )
        # Role identity and live stage content meet only in selector geometry.
        # Normalize them separately so neither can buy routing authority by
        # increasing its value norm.
        keys = F.normalize(
            F.normalize(stage_content[:, candidates_1d].detach().float(), dim=-1)
            + F.normalize(identity.float(), dim=-1),
            dim=-1,
        )
        logits = torch.einsum("bh,bkh->bk", query, keys) * math.sqrt(float(self.hidden_size))
        learned_probabilities = torch.softmax(logits, dim=-1)
        candidate_count = int(candidates.shape[1])
        exploration_value = (
            max(0.0, 1.0 - float(self._contraction_progress_value))
            if self.training and candidate_count > 1
            else 0.0
        )
        exploration = torch.full(
            (batch,), exploration_value, device=query.device, dtype=torch.float32
        )
        if exploration_value > 0.0:
            uniform = torch.full_like(
                learned_probabilities, 1.0 / float(candidate_count)
            )
            probabilities = (
                learned_probabilities * (1.0 - exploration_value)
                + uniform * exploration_value
            )
            if self._contraction_progress_value > 0.0:
                selected_local = torch.multinomial(
                    probabilities.float(), num_samples=1
                ).squeeze(1)
            else:
                # The contraction is an exact identity during warmup. Do not
                # consume RNG here and perturb the host block's dropout stream.
                selected_local = learned_probabilities.argmax(dim=-1)
        else:
            probabilities = learned_probabilities
            selected_local = probabilities.argmax(dim=-1)
        stage_index = candidates.gather(1, selected_local[:, None]).squeeze(1)
        learned_detached = learned_probabilities.detach().float().clamp_min(1e-8)
        entropy = -(learned_detached * learned_detached.log()).sum(dim=-1)
        return (
            stage_index,
            candidates,
            probabilities,
            entropy,
            learned_detached.max(dim=-1).values,
            query.detach(),
            exploration,
        )

    def _select_adaptive_stages(
        self,
        *,
        block_index: Tensor,
        stage_content: Tensor,
        global_intent: Tensor,
        time_state: Tensor,
        step_state: Tensor,
        action: Tensor,
        control_state: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Select one local semantic stage for each per-sample block owner."""
        batch = int(stage_content.shape[0])
        selected = torch.empty(batch, device=stage_content.device, dtype=torch.long)
        entropy = torch.zeros(batch, device=stage_content.device, dtype=torch.float32)
        maximum = torch.ones(batch, device=stage_content.device, dtype=torch.float32)
        queries = torch.zeros(
            batch, self.hidden_size, device=stage_content.device, dtype=torch.float32
        )
        exploration = torch.zeros(batch, device=stage_content.device, dtype=torch.float32)
        for owner in range(self.refine_block_count):
            indices = torch.nonzero(block_index == owner, as_tuple=False).flatten()
            if int(indices.numel()) == 0:
                continue
            (
                owner_stage,
                _,
                _,
                owner_entropy,
                owner_maximum,
                owner_query,
                owner_exploration,
            ) = self._select_owned_stage(
                block_index=owner,
                stage_content=stage_content.index_select(0, indices),
                global_intent=global_intent.index_select(0, indices),
                time_state=time_state.index_select(0, indices),
                step_state=step_state.index_select(0, indices),
                action=action.index_select(0, indices),
                control_state=control_state.index_select(0, indices),
            )
            selected.index_copy_(0, indices, owner_stage)
            entropy.index_copy_(0, indices, owner_entropy)
            maximum.index_copy_(0, indices, owner_maximum)
            queries.index_copy_(0, indices, owner_query)
            exploration.index_copy_(0, indices, owner_exploration)
        return (
            selected,
            selected[:, None],
            torch.ones(batch, 1, device=stage_content.device, dtype=torch.float32),
            entropy,
            maximum,
            queries,
            exploration,
        )

    def _fixed_schedule(self, *, device: torch.device) -> tuple[Tensor, Tensor, int]:
        blocks = torch.tensor(
            [self._block_for_step(step) for step in range(self.refine_steps)],
            device=device,
            dtype=torch.long,
        )
        return blocks, torch.ones(self.refine_steps, device=device, dtype=torch.bool), self.refine_steps

    def _random_dwell_schedule(self, *, device: torch.device) -> tuple[Tensor, Tensor, int]:
        prefix_probability = float(self.config.hierarchical_mmdit_random_prefix_probability)
        use_prefix = self.refine_block_count > 1 and float(torch.rand(())) < prefix_probability
        if use_prefix:
            active_block_count = int(torch.randint(1, self.refine_block_count, ()).item())
        else:
            active_block_count = self.refine_block_count
        active_length = int(torch.randint(active_block_count, self.refine_steps + 1, ()).item())
        dwell = torch.ones(active_block_count, dtype=torch.long)
        for _ in range(active_length - active_block_count):
            dwell[int(torch.randint(0, active_block_count, ()).item())] += 1
        schedule = torch.repeat_interleave(torch.arange(active_block_count), dwell)
        active = torch.zeros(self.refine_steps, dtype=torch.bool)
        active[:active_length] = True
        if active_length < self.refine_steps:
            schedule = torch.cat([
                schedule,
                torch.full((self.refine_steps - active_length,), active_block_count - 1, dtype=torch.long),
            ])
        return schedule.to(device=device), active.to(device=device), active_length

    def _semantic_physical_components(self, value: Tensor) -> dict[str, Tensor]:
        """Return per-sample physical norms without conflating field semantics."""
        ad = int(self.config.arm_dim)
        gf = int(self.config.gripper_field_dim)
        value = value.float()
        arm_field = value[..., : 2 * ad]
        if self.response_codec.uses_arm_manifold:
            arm_native, _, arm_null = self.response_codec.project_arm_tangent(arm_field)
            arm_native_energy = arm_native.float().square().sum(dim=-1)
            arm_null_energy = 0.5 * (
                arm_null[..., :ad].float().square()
                + arm_null[..., ad : 2 * ad].float().square()
            ).sum(dim=-1)
        else:
            arm_native_energy = 0.5 * (
                arm_field[..., :ad].square() + arm_field[..., ad : 2 * ad].square()
            ).sum(dim=-1)
            arm_null_energy = torch.zeros_like(arm_native_energy)
        gripper_field = value[..., 2 * ad : 2 * ad + gf]
        if self.response_codec.uses_parseval_gripper_field:
            native = self.response_codec.decode_gripper_field(gripper_field).float()[..., 0]
            null = gripper_field - self.response_codec.project_gripper_field(gripper_field)
            gripper_native_energy = native.square()
            gripper_null_energy = null.float().square().sum(dim=-1)
        else:
            gripper_native_energy = gripper_field.square().mean(dim=-1)
            gripper_null_energy = torch.zeros_like(gripper_native_energy)
        combined = (
            arm_native_energy
            + arm_null_energy
            + gripper_native_energy
            + gripper_null_energy
        ) / float(ad + 1)
        return {
            "combined": combined.mean(dim=-1).clamp_min(0.0).sqrt(),
            "arm": (arm_native_energy / float(ad)).mean(dim=-1).clamp_min(0.0).sqrt(),
            "gripper": gripper_native_energy.mean(dim=-1).clamp_min(0.0).sqrt(),
            "arm_null": (arm_null_energy / float(ad)).mean(dim=-1).clamp_min(0.0).sqrt(),
            "gripper_null": gripper_null_energy.mean(dim=-1).clamp_min(0.0).sqrt(),
        }

    def _semantic_physical_rms(self, value: Tensor) -> Tensor:
        """Physical field norm with six arm dimensions plus one gripper unit."""
        return self._semantic_physical_components(value)["combined"]

    def _physical_response(
        self,
        before: Tensor,
        after: Tensor,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        components = self._semantic_physical_components(after.float() - before.float())
        absolute = components["combined"]
        reference = 0.5 * (
            self._semantic_physical_rms(before.float())
            + self._semantic_physical_rms(after.float())
        )
        floor = float(self.config.hierarchical_mmdit_action_response_floor)
        return absolute, absolute / reference.clamp_min(floor), components

    @staticmethod
    def _stage_pressure(before: Tensor, after: Tensor) -> tuple[Tensor, Tensor]:
        delta = (after.detach().float() - before.detach().float()).square().mean(dim=(1, 2)).sqrt()
        reference = before.detach().float().square().mean(dim=(1, 2)).sqrt()
        return delta, delta / reference.clamp_min(1e-4)

    def _threshold_rows(self, time: Tensor, values: tuple[float, float, float]) -> Tensor:
        thresholds = torch.tensor(values, device=time.device, dtype=torch.float32)
        bins = torch.clamp((time.detach().float().clamp(0.0, 1.0) * 3.0).long(), max=2)
        return thresholds.index_select(0, bins)

    @staticmethod
    def _mean_metrics(rows: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        if not rows:
            return {}
        keys = set.intersection(*(set(row) for row in rows))
        return {
            key: torch.stack([row[key] for row in rows], dim=0).mean(dim=0)
            for key in keys
            if all(torch.is_tensor(row[key]) for row in rows)
        }

    def _run_owned_blocks(
        self,
        action: Tensor,
        *,
        block_index: Tensor,
        noisy_tokens: Tensor,
        stage_tokens: Tensor,
        low_tokens: Tensor,
        shared_cond: Tensor,
        operator_cond: Tensor,
        stage_index: Tensor,
        stage_candidates: Tensor,
        stage_probabilities: Tensor,
        contraction_factors: tuple[dict[str, Tensor], ...] | None,
        uniform_owner: int | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Dispatch samples to distinct full-rank blocks without soft mixing."""
        batch = int(action.shape[0])
        if tuple(block_index.shape) != (batch,):
            raise ValueError("block_index must contain one owner per sample")
        if uniform_owner is not None:
            owner = int(uniform_owner)
            if not 0 <= owner < self.refine_block_count:
                raise ValueError("uniform refinement owner is outside the block repertoire")
            start, stop = self._stage_bounds(owner)
            local_stage = stage_index - start
            local_candidates = stage_candidates - start
            if local_candidates.device.type == "cpu" and (
                bool((local_candidates < 0).any())
                or bool((local_candidates >= stop - start).any())
            ):
                raise ValueError("operator-stage candidates crossed a block ownership boundary")
            return self.blocks[owner](
                action,
                noisy_tokens=noisy_tokens,
                stage_tokens=stage_tokens,
                low_tokens=low_tokens,
                shared_cond=shared_cond,
                operator_cond=operator_cond,
                contractions=self.operator_contractions[owner],
                contraction_progress=self.contraction_progress,
                stage_index=local_stage,
                contraction_identity_bypass=(
                    self._contraction_progress_value == 0.0
                ),
                stage_candidates=local_candidates,
                stage_probabilities=stage_probabilities,
                contraction_factors=(
                    None if contraction_factors is None
                    else contraction_factors[owner]
                ),
                low_role_ids=self.workspace.low_slot_role_ids,
                low_role_names=self.workspace.memory_bank.ROLE_NAMES,
            )
        group_indices: list[Tensor] = []
        group_actions: list[Tensor] = []
        group_metrics: list[dict[str, Tensor]] = []
        group_sizes: list[int] = []
        for owner, block in enumerate(self.blocks):
            indices = torch.nonzero(block_index == owner, as_tuple=False).flatten()
            size = int(indices.numel())
            if size == 0:
                continue
            start, stop = self._stage_bounds(owner)
            local_stage = stage_index.index_select(0, indices) - start
            local_candidates = stage_candidates.index_select(0, indices) - start
            if local_candidates.device.type == "cpu" and (
                bool((local_candidates < 0).any())
                or bool((local_candidates >= stop - start).any())
            ):
                raise ValueError("operator-stage candidates crossed a block ownership boundary")
            owned_action, owned_metrics = block(
                action.index_select(0, indices),
                noisy_tokens=noisy_tokens.index_select(0, indices),
                stage_tokens=stage_tokens.index_select(0, indices),
                low_tokens=low_tokens.index_select(0, indices),
                shared_cond=shared_cond.index_select(0, indices),
                operator_cond=operator_cond.index_select(0, indices),
                contractions=self.operator_contractions[owner],
                contraction_progress=self.contraction_progress,
                stage_index=local_stage,
                contraction_identity_bypass=(
                    self._contraction_progress_value == 0.0
                ),
                stage_candidates=local_candidates,
                stage_probabilities=stage_probabilities.index_select(0, indices),
                contraction_factors=(
                    None if contraction_factors is None
                    else contraction_factors[owner]
                ),
                low_role_ids=self.workspace.low_slot_role_ids,
                low_role_names=self.workspace.memory_bank.ROLE_NAMES,
            )
            group_indices.append(indices)
            group_actions.append(owned_action)
            group_metrics.append(owned_metrics)
            group_sizes.append(size)
        if not group_actions:
            raise RuntimeError("no refinement block owned the current batch")

        grouped_order = torch.cat(group_indices, dim=0)
        restore_order = grouped_order.argsort()
        merged_action = torch.cat(group_actions, dim=0).index_select(0, restore_order)
        common_keys = set.intersection(*(set(metrics) for metrics in group_metrics))
        merged_metrics: dict[str, Tensor] = {}
        for key in common_keys:
            values = [metrics[key] for metrics in group_metrics]
            if not all(torch.is_tensor(value) for value in values):
                continue
            if all(
                value.ndim > 0 and int(value.shape[0]) == size
                for value, size in zip(values, group_sizes, strict=True)
            ):
                merged_metrics[key] = torch.cat(values, dim=0).index_select(
                    0, restore_order
                )
            elif all(tuple(value.shape) == tuple(values[0].shape) for value in values):
                merged_metrics[key] = sum(
                    value * (float(size) / float(batch))
                    for value, size in zip(values, group_sizes, strict=True)
                )
        return merged_action, merged_metrics

    def _detached_velocity_prediction(self, action: Tensor) -> Tensor:
        """Read physical action diagnostics without poisoning AMP's weight cache."""
        # This trainable head is called again for the gradient-bearing output at
        # the end of forward().  Under an outer autocast context, making its
        # first call inside no_grad can cache detached parameter casts and erase
        # the later head gradients.  Detaching the input and result keeps this
        # probe outside the objective while preserving a grad-capable AMP cache.
        return self.velocity_head(self.action_norm(action.detach())).detach()

    def _run_refinement(
        self,
        *,
        action: Tensor,
        stage_content: Tensor,
        prepared_evidence: PreparedEvidenceMemory,
        contracts: dict[str, Tensor],
        noisy: Tensor,
        time: Tensor,
        time_state: Tensor,
        workspace_condition: Tensor,
        routing_mode: str,
    ) -> dict[str, object]:
        batch = int(action.shape[0])
        device = action.device
        dtype = action.dtype
        if routing_mode not in {"fixed", "random_dwell", "adaptive", "learned"}:
            raise ValueError(f"unsupported refinement routing mode: {routing_mode!r}")
        if routing_mode == "random_dwell":
            schedule, schedule_active, scheduled_steps = self._random_dwell_schedule(device=device)
        elif routing_mode in {"fixed", "learned"}:
            schedule, schedule_active, scheduled_steps = self._fixed_schedule(device=device)
        else:
            schedule = torch.zeros(self.refine_steps, device=device, dtype=torch.long)
            schedule_active = torch.ones(self.refine_steps, device=device, dtype=torch.bool)
            scheduled_steps = self.refine_steps

        current_block = torch.zeros(batch, device=device, dtype=torch.long)
        current_stage = torch.zeros(batch, device=device, dtype=torch.long)
        active_rows = torch.ones(batch, device=device, dtype=torch.bool)
        exhaustion_count = torch.zeros(batch, device=device, dtype=torch.long)
        unresolved_rows = torch.zeros(batch, device=device, dtype=torch.bool)
        block_visit_count = torch.zeros(
            batch, self.refine_block_count, device=device, dtype=torch.float32
        )
        previous_response_rel = torch.zeros(batch, device=device, dtype=torch.float32)
        previous_pressure_rel = torch.zeros(batch, device=device, dtype=torch.float32)
        has_previous_response = torch.zeros(batch, device=device, dtype=torch.float32)
        action_threshold = self._threshold_rows(
            time,
            tuple(float(value) for value in self.config.hierarchical_mmdit_action_response_thresholds),
        )
        stage_threshold = self._threshold_rows(
            time,
            tuple(float(value) for value in self.config.hierarchical_mmdit_stage_pressure_thresholds),
        )

        workspace_rows: list[dict[str, Tensor]] = []
        mmdit_rows: list[dict[str, Tensor]] = []
        condition_norm_rows: list[Tensor] = []
        step_state_rows: list[Tensor] = []
        response_abs_rows: list[Tensor] = []
        response_rel_rows: list[Tensor] = []
        response_arm_rows: list[Tensor] = []
        response_gripper_rows: list[Tensor] = []
        response_arm_null_rows: list[Tensor] = []
        response_gripper_null_rows: list[Tensor] = []
        pressure_abs_rows: list[Tensor] = []
        pressure_rel_rows: list[Tensor] = []
        stage_id_rows: list[Tensor] = []
        block_id_rows: list[Tensor] = []
        active_history: list[Tensor] = []
        selector_entropy_rows: list[Tensor] = []
        selector_max_rows: list[Tensor] = []
        selector_query_rows: list[Tensor] = []
        selector_exploration_rows: list[Tensor] = []
        exit_logit_rows: list[Tensor] = []
        exit_probability_rows: list[Tensor] = []
        exit_candidate_rows: list[Tensor] = []
        prediction = self._detached_velocity_prediction(action)
        prediction_rows: list[Tensor] = [prediction]
        contraction_factors = (
            None
            if self._contraction_progress_value == 0.0
            else self.prepare_contraction_factors()
        )

        for step_index in range(self.refine_steps):
            if routing_mode not in {"adaptive", "learned"} and step_index >= scheduled_steps:
                break
            if routing_mode == "adaptive":
                uniform_owner = None
                block_index = current_block
                step_active = active_rows
            else:
                uniform_owner = int(schedule[step_index].item())
                block_index = schedule[step_index].expand(batch)
                step_active = schedule_active[step_index].expand(batch)
                if routing_mode == "learned":
                    step_active = step_active & active_rows
            if routing_mode in {"adaptive", "learned"} and not bool(step_active.any()):
                break

            remaining = float(scheduled_steps - step_index - 1) / float(max(self.refine_steps, 1))
            progress = (
                1.0
                if self.refine_steps <= 1
                else float(step_index) / float(self.refine_steps - 1)
            )
            step_state = self._step_state(
                block_index,
                progress_fraction=progress,
                remaining_fraction=remaining,
                dtype=dtype,
            )
            local_dwell = block_visit_count.gather(
                1, block_index[:, None]
            ).squeeze(1) / float(max(self.refine_steps, 1))
            selector_control_state = torch.stack([
                torch.log1p(previous_response_rel.clamp_min(0.0)),
                torch.log1p(previous_pressure_rel.clamp_min(0.0)),
                local_dwell,
                has_previous_response,
            ], dim=-1)
            if routing_mode == "adaptive":
                (
                    stage_index,
                    stage_candidates,
                    stage_probabilities,
                    selector_entropy,
                    selector_max,
                    selector_query,
                    selector_exploration,
                ) = self._select_adaptive_stages(
                    block_index=block_index,
                    stage_content=stage_content,
                    global_intent=contracts["global_intent"],
                    time_state=time_state,
                    step_state=step_state,
                    action=action,
                    control_state=selector_control_state,
                )
            else:
                (
                    stage_index,
                    stage_candidates,
                    stage_probabilities,
                    selector_entropy,
                    selector_max,
                    selector_query,
                    selector_exploration,
                ) = self._select_owned_stage(
                    block_index=uniform_owner,
                    stage_content=stage_content,
                    global_intent=contracts["global_intent"],
                    time_state=time_state,
                    step_state=step_state,
                    action=action,
                    control_state=selector_control_state,
                )
            operator_condition = self.operator_condition(torch.cat([
                contracts["global_intent"],
                time_state,
                step_state,
            ], dim=-1))
            shared_condition = self.shared_condition(torch.cat([
                contracts["global_intent"],
                time_state,
                step_state,
            ], dim=-1))
            previous_stage = stage_content
            (
                low,
                candidate_stage,
                stage_for_action,
                _low_logit_bias,
                _stage_logit_bias,
                workspace_metrics,
            ) = self.workspace.step(
                prepared_evidence=prepared_evidence,
                stage_content=stage_content,
                primary_cond=workspace_condition,
                step_index=step_index,
                read_contract=contracts["read_contract"],
                step_state_override=step_state,
            )
            stage_content = torch.where(
                step_active[:, None, None], candidate_stage, previous_stage
            )
            low = low + self.condition_type[:, 0:1].to(device=device, dtype=dtype)
            stage_for_action = stage_for_action + self.condition_type[:, 1:2].to(device=device, dtype=dtype)
            noisy_typed = noisy + self.condition_type[:, 2:3].to(device=device, dtype=dtype)
            candidate_action, mmdit_metrics = self._run_owned_blocks(
                action,
                block_index=block_index,
                noisy_tokens=noisy_typed,
                stage_tokens=stage_for_action,
                low_tokens=low,
                shared_cond=shared_condition,
                operator_cond=operator_condition,
                stage_index=stage_index,
                stage_candidates=stage_candidates,
                stage_probabilities=stage_probabilities,
                contraction_factors=contraction_factors,
                uniform_owner=uniform_owner,
            )
            action = torch.where(step_active[:, None, None], candidate_action, action)
            next_prediction = self._detached_velocity_prediction(action)
            with torch.no_grad():
                response_abs, response_rel, response_components = self._physical_response(
                    prediction, next_prediction
                )
                pressure_abs, pressure_rel = self._stage_pressure(previous_stage, stage_content)
                response_abs = torch.where(step_active, response_abs, torch.zeros_like(response_abs))
                response_rel = torch.where(step_active, response_rel, torch.zeros_like(response_rel))
                response_components = {
                    key: torch.where(step_active, value, torch.zeros_like(value))
                    for key, value in response_components.items()
                }
                pressure_abs = torch.where(step_active, pressure_abs, torch.zeros_like(pressure_abs))
                pressure_rel = torch.where(step_active, pressure_rel, torch.zeros_like(pressure_rel))
                previous_response_rel = torch.where(
                    step_active, response_rel, previous_response_rel
                )
                previous_pressure_rel = torch.where(
                    step_active, pressure_rel, previous_pressure_rel
                )
                has_previous_response = torch.where(
                    step_active,
                    torch.ones_like(has_previous_response),
                    has_previous_response,
                )
                block_visit_count = block_visit_count.scatter_add(
                    1,
                    block_index[:, None],
                    step_active.float()[:, None],
                )

            post_local_dwell = block_visit_count.gather(
                1, block_index[:, None]
            ).squeeze(1) / float(max(self.refine_steps, 1))
            exit_control_state = torch.stack([
                torch.log1p(response_rel.detach().float().clamp_min(0.0)),
                torch.log1p(pressure_rel.detach().float().clamp_min(0.0)),
                post_local_dwell.detach().float(),
                torch.ones_like(post_local_dwell, dtype=torch.float32),
            ], dim=-1)
            exit_query = self._controller_query(
                global_intent=contracts["global_intent"],
                time_state=time_state,
                step_state=step_state,
                action=action,
                control_state=exit_control_state,
            )
            # Oracle route supervision owns only the exit head.  The shared
            # query encoder also serves stage selection, so allowing this loss
            # through it would silently change stage routing while claiming to
            # train only the stop/continue decision.
            exit_logit = self._exit_logit(exit_query)
            exit_probability = torch.sigmoid(exit_logit.detach().float())
            if routing_mode == "adaptive":
                exit_candidate = step_active
            else:
                next_step = step_index + 1
                block_boundary = (
                    next_step >= scheduled_steps
                    or not bool(schedule_active[next_step])
                    or int(schedule[next_step].item()) != int(schedule[step_index].item())
                )
                exit_candidate = step_active & block_boundary

            workspace_rows.append(workspace_metrics)
            mmdit_rows.append(mmdit_metrics)
            condition_norm_rows.append(torch.stack([
                low.detach().float().norm(dim=-1).mean(),
                stage_for_action.detach().float().norm(dim=-1).mean(),
                noisy_typed.detach().float().norm(dim=-1).mean(),
            ]).mean())
            step_state_rows.append(step_state.detach().float().norm(dim=-1).mean())
            response_abs_rows.append(response_abs)
            response_rel_rows.append(response_rel)
            response_arm_rows.append(response_components["arm"])
            response_gripper_rows.append(response_components["gripper"])
            response_arm_null_rows.append(response_components["arm_null"])
            response_gripper_null_rows.append(response_components["gripper_null"])
            pressure_abs_rows.append(pressure_abs)
            pressure_rel_rows.append(pressure_rel)
            current_stage = stage_index
            stage_id_rows.append(stage_index.detach())
            block_id_rows.append(block_index.detach())
            active_history.append(step_active.detach())
            selector_entropy_rows.append(selector_entropy.detach())
            selector_max_rows.append(selector_max.detach())
            selector_query_rows.append(selector_query.detach())
            selector_exploration_rows.append(selector_exploration.detach())
            exit_logit_rows.append(exit_logit)
            exit_probability_rows.append(exit_probability)
            exit_candidate_rows.append(exit_candidate.detach())
            prediction_rows.append(next_prediction)
            prediction = next_prediction

            if routing_mode == "adaptive":
                action_live = response_rel > action_threshold
                pressure_live = pressure_rel > stage_threshold
                can_advance = current_block < self.refine_block_count - 1
                advance = step_active & ~action_live & pressure_live & can_advance
                exhausted = step_active & ~action_live & ~advance
                current_block = torch.where(advance, current_block + 1, current_block)
                exhaustion_count = torch.where(
                    action_live | advance,
                    torch.zeros_like(exhaustion_count),
                    exhaustion_count + exhausted.to(dtype=exhaustion_count.dtype),
                )
                stop = exhaustion_count >= int(self.config.hierarchical_mmdit_exhaustion_confirm_steps)
                unresolved_rows = unresolved_rows | (
                    stop & pressure_live & (current_block == self.refine_block_count - 1)
                )
                active_rows = active_rows & ~stop
            elif routing_mode == "learned":
                final_candidate = step_index + 1 >= scheduled_steps
                learned_stop = exit_candidate & (
                    (exit_probability > 0.5) | final_candidate
                )
                active_rows = active_rows & ~learned_stop

        budget_exhausted_rows = (
            active_rows.detach().clone()
            if routing_mode == "adaptive"
            else torch.zeros_like(active_rows)
        )
        unresolved_rows = unresolved_rows | budget_exhausted_rows

        while len(response_abs_rows) < self.refine_steps:
            zero = torch.zeros(batch, device=device, dtype=torch.float32)
            response_abs_rows.append(zero)
            response_rel_rows.append(zero)
            response_arm_rows.append(zero)
            response_gripper_rows.append(zero)
            response_arm_null_rows.append(zero)
            response_gripper_null_rows.append(zero)
            pressure_abs_rows.append(zero)
            pressure_rel_rows.append(zero)
            stage_id_rows.append(current_stage.detach().clone())
            block_id_rows.append(current_block.detach().clone())
            active_history.append(torch.zeros(batch, device=device, dtype=torch.bool))
            selector_entropy_rows.append(torch.zeros(batch, device=device, dtype=torch.float32))
            selector_max_rows.append(torch.zeros(batch, device=device, dtype=torch.float32))
            selector_query_rows.append(torch.zeros(
                batch, self.hidden_size, device=device, dtype=torch.float32
            ))
            selector_exploration_rows.append(zero)
            exit_logit_rows.append(torch.zeros(
                batch, device=device, dtype=torch.float32
            ))
            exit_probability_rows.append(zero)
            exit_candidate_rows.append(torch.zeros(
                batch, device=device, dtype=torch.bool
            ))
            prediction_rows.append(prediction)

        active_stack = torch.stack(active_history, dim=1)
        return {
            "action": action,
            "stage_content": stage_content,
            "workspace_rows": workspace_rows,
            "mmdit_rows": mmdit_rows,
            "condition_norm_rows": condition_norm_rows,
            "step_state_rows": step_state_rows,
            "prediction_rows": torch.stack(prediction_rows, dim=1),
            "response_abs_rows": torch.stack(response_abs_rows, dim=1),
            "response_rel_rows": torch.stack(response_rel_rows, dim=1),
            "response_arm_rows": torch.stack(response_arm_rows, dim=1),
            "response_gripper_rows": torch.stack(response_gripper_rows, dim=1),
            "response_arm_null_rows": torch.stack(response_arm_null_rows, dim=1),
            "response_gripper_null_rows": torch.stack(response_gripper_null_rows, dim=1),
            "pressure_abs_rows": torch.stack(pressure_abs_rows, dim=1),
            "pressure_rel_rows": torch.stack(pressure_rel_rows, dim=1),
            "stage_id_rows": torch.stack(stage_id_rows, dim=1),
            "block_id_rows": torch.stack(block_id_rows, dim=1),
            "active_rows": active_stack,
            "selector_entropy_rows": torch.stack(selector_entropy_rows, dim=1),
            "selector_max_rows": torch.stack(selector_max_rows, dim=1),
            "selector_query_rows": torch.stack(selector_query_rows, dim=1),
            "selector_exploration_rows": torch.stack(selector_exploration_rows, dim=1),
            "exit_logit_rows": torch.stack(exit_logit_rows, dim=1),
            "exit_probability_rows": torch.stack(exit_probability_rows, dim=1),
            "exit_candidate_rows": torch.stack(exit_candidate_rows, dim=1),
            "executed_steps": active_stack.float().sum(dim=1),
            "unresolved_rows": unresolved_rows,
            "budget_exhausted_rows": budget_exhausted_rows,
        }

    def forward(
        self,
        *,
        noisy_physical: Tensor,
        time: Tensor,
        trajectory_tokens: Tensor,
        trajectory_workspace_tokens: Tensor,
        rollout_tokens: Tensor,
        transition_memory: Tensor | list[Tensor] | tuple[Tensor, ...],
        event_evidence: Tensor,
        state_memory: Tensor | list[Tensor] | tuple[Tensor, ...],
        intent_memory: dict[str, Tensor],
        layer_contracts: list[dict[str, Tensor]],
    ) -> dict[str, Tensor]:
        device = noisy_physical.device
        dtype = noisy_physical.dtype
        batch = int(noisy_physical.shape[0])
        organized = self.organizer(
            trajectory_tokens=trajectory_tokens,
            trajectory_workspace_tokens=trajectory_workspace_tokens,
            rollout_tokens=rollout_tokens,
            transition_memory=transition_memory,
            event_evidence=event_evidence,
            state_memory=state_memory,
            intent_memory=intent_memory,
            layer_contracts=layer_contracts,
        )
        contracts = self.intent_compiler(
            layer_scan=organized["layer_scan"],
            geom_summary=organized["geom_summary"],
            global_summary=organized["global_summary"],
            transition_summary=organized["transition_summary"],
            event_summary=organized["event_summary"],
            state_summary=organized["state_summary"],
        )
        evidence_sources = organized["evidence_sources"]
        if not isinstance(evidence_sources, dict):
            raise TypeError("condition organizer returned invalid evidence sources")
        prepared = self.workspace.prepare_evidence(
            evidence_sources,
            batch_size=batch,
            device=device,
            dtype=dtype,
        )
        stage_content = self.workspace.init_stage(contracts["stage_contract"])
        action, initializer_metrics = self.action_initializer(
            batch_size=batch,
            device=device,
            dtype=dtype,
        )
        noisy, noisy_gate_mean = self._gate_noisy_tokens(
            self.noisy_action_lift(noisy_physical), time
        )
        time_state = self.time_lift(self.time(time.to(dtype=dtype)))
        workspace_condition = self.workspace_condition(torch.cat([
            contracts["global_intent"],
            time_state,
        ], dim=-1))
        initial_action = action
        initial_stage = stage_content
        exhaustion_mode = str(self.config.hierarchical_mmdit_exhaustion_mode)
        if not self.training and exhaustion_mode == "adaptive":
            routing_mode = "adaptive"
        elif not self.training and exhaustion_mode == "learned":
            routing_mode = "learned"
        elif self.training and str(self.config.hierarchical_mmdit_schedule_mode) == "random_dwell":
            routing_mode = "random_dwell"
        else:
            routing_mode = "fixed"
        refinement = self._run_refinement(
            action=action,
            stage_content=stage_content,
            prepared_evidence=prepared,
            contracts=contracts,
            noisy=noisy,
            time=time,
            time_state=time_state,
            workspace_condition=workspace_condition,
            routing_mode=routing_mode,
        )
        action = refinement["action"]
        if not torch.is_tensor(action):
            raise TypeError("refinement returned an invalid action tensor")
        workspace_rows = refinement["workspace_rows"]
        mmdit_rows = refinement["mmdit_rows"]
        condition_norm_rows = refinement["condition_norm_rows"]
        step_state_rows = refinement["step_state_rows"]
        if not all(isinstance(rows, list) for rows in (
            workspace_rows, mmdit_rows, condition_norm_rows, step_state_rows,
        )):
            raise TypeError("refinement returned invalid metric rows")

        shadow: dict[str, object] | None = None
        if not self.training and exhaustion_mode in {"shadow", "learned_shadow"}:
            with torch.no_grad():
                shadow = self._run_refinement(
                    action=initial_action.detach(),
                    stage_content=initial_stage.detach(),
                    prepared_evidence=prepared,
                    contracts=contracts,
                    noisy=noisy,
                    time=time,
                    time_state=time_state,
                    workspace_condition=workspace_condition,
                    routing_mode=(
                        "learned" if exhaustion_mode == "learned_shadow" else "adaptive"
                    ),
                )

        action = self.action_norm(action)
        pred_velocity = self.velocity_head(action)
        output_contract_norm = torch.zeros((), device=device, dtype=torch.float32)
        if self.output_contract_fusion is not None and self.output_contract_proj is not None:
            # Restricted contract: event/motion subheads only; velocity reads
            # raw action tokens above and is untouched by construction.
            g_out = self.output_contract_fusion(
                organized["layer_scan"],
                organized["transition_summary"],
                organized["event_summary"],
            )
            subhead_tokens = action + self.output_contract_proj(g_out)[:, None]
            output_contract_norm = g_out.detach().float().norm(dim=-1).mean()
        else:
            subhead_tokens = action
        event_logits = self.event_head(subhead_tokens)
        motion_logits = self.motion_head(subhead_tokens).squeeze(-1)
        response_abs_rows = refinement["response_abs_rows"]
        response_rel_rows = refinement["response_rel_rows"]
        response_arm_rows = refinement["response_arm_rows"]
        response_gripper_rows = refinement["response_gripper_rows"]
        response_arm_null_rows = refinement["response_arm_null_rows"]
        response_gripper_null_rows = refinement["response_gripper_null_rows"]
        pressure_abs_rows = refinement["pressure_abs_rows"]
        pressure_rel_rows = refinement["pressure_rel_rows"]
        stage_id_rows = refinement["stage_id_rows"]
        block_id_rows = refinement["block_id_rows"]
        active_rows = refinement["active_rows"]
        executed_steps = refinement["executed_steps"]
        unresolved_rows = refinement["unresolved_rows"]
        budget_exhausted_rows = refinement["budget_exhausted_rows"]
        probe_predictions = refinement["prediction_rows"]
        selector_entropy_rows = refinement["selector_entropy_rows"]
        selector_max_rows = refinement["selector_max_rows"]
        selector_query_rows = refinement["selector_query_rows"]
        selector_exploration_rows = refinement["selector_exploration_rows"]
        exit_logit_rows = refinement["exit_logit_rows"]
        exit_probability_rows = refinement["exit_probability_rows"]
        exit_candidate_rows = refinement["exit_candidate_rows"]
        probe_tensors = (
            response_abs_rows, response_rel_rows, response_arm_rows, response_gripper_rows,
            response_arm_null_rows, response_gripper_null_rows, pressure_abs_rows, pressure_rel_rows,
            stage_id_rows, block_id_rows, active_rows, executed_steps, unresolved_rows,
            budget_exhausted_rows, probe_predictions,
            selector_entropy_rows, selector_max_rows, selector_query_rows,
            selector_exploration_rows, exit_logit_rows, exit_probability_rows,
            exit_candidate_rows,
        )
        if not all(torch.is_tensor(value) for value in probe_tensors):
            raise TypeError("refinement returned invalid probe tensors")
        active_float = active_rows.float()
        active_denominator = active_float.sum().clamp_min(1.0)
        exit_candidate_float = exit_candidate_rows.float() * active_float
        exit_candidate_denominator = exit_candidate_float.sum().clamp_min(1.0)
        final_step_index = active_float.sum(dim=1).long().clamp_min(1) - 1
        final_stage_rows = stage_id_rows.gather(1, final_step_index[:, None]).squeeze(1)
        final_block_rows = block_id_rows.gather(1, final_step_index[:, None]).squeeze(1)
        if self.refine_steps > 1:
            transition_active = active_float[:, 1:] * active_float[:, :-1]
            stage_advances = (
                (stage_id_rows[:, 1:] > stage_id_rows[:, :-1]).float() * transition_active
            )
            stage_advance_rate = stage_advances.sum() / transition_active.sum().clamp_min(1.0)
            block_advances = (
                (block_id_rows[:, 1:] > block_id_rows[:, :-1]).float()
                * transition_active
            )
            block_advance_rate = (
                block_advances.sum() / transition_active.sum().clamp_min(1.0)
            )
            selector_query_change_rows = (
                1.0 - F.cosine_similarity(
                    selector_query_rows[:, 1:].float(),
                    selector_query_rows[:, :-1].float(),
                    dim=-1,
                )
            ).clamp_min(0.0)
            selector_query_change = (
                selector_query_change_rows * transition_active
            ).sum() / transition_active.sum().clamp_min(1.0)
            same_block_active = transition_active * (
                block_id_rows[:, 1:] == block_id_rows[:, :-1]
            ).float()
            selector_same_block_query_change = (
                selector_query_change_rows * same_block_active
            ).sum() / same_block_active.sum().clamp_min(1.0)
        else:
            stage_advance_rate = torch.zeros((), device=device, dtype=torch.float32)
            block_advance_rate = torch.zeros((), device=device, dtype=torch.float32)
            selector_query_change = torch.zeros((), device=device, dtype=torch.float32)
            selector_same_block_query_change = torch.zeros(
                (), device=device, dtype=torch.float32
            )

        def active_mean(value: Tensor) -> Tensor:
            return (value.detach().float() * active_float).sum() / active_denominator

        def active_quantile(value: Tensor, quantile: float) -> Tensor:
            selected = value.detach().float()[active_rows]
            if int(selected.numel()) == 0:
                return torch.zeros((), device=device, dtype=torch.float32)
            return torch.quantile(selected, float(quantile))

        time_bins = torch.clamp(
            (time.detach().float().clamp(0.0, 1.0) * 3.0).long(), max=2
        )

        def active_time_quantile(
            value: Tensor,
            time_bin: int,
            quantile: float,
        ) -> Tensor:
            selected_mask = active_rows & (time_bins[:, None] == int(time_bin))
            selected = value.detach().float()[selected_mask]
            if int(selected.numel()) == 0:
                return torch.zeros((), device=device, dtype=torch.float32)
            return torch.quantile(selected, float(quantile))

        result: dict[str, Tensor] = {
            "hierarchical_mmdit_output_contract_norm": output_contract_norm,
            "pred_velocity": pred_velocity,
            "event_logits": event_logits,
            "motion_logits": motion_logits,
            "action_tokens": action,
            "transition_latent": action,
            "hierarchical_mmdit_action_token_norm": action.detach().float().norm(dim=-1).mean(),
            "hierarchical_mmdit_condition_token_norm": torch.stack(condition_norm_rows).mean(),
            "hierarchical_mmdit_noisy_token_norm": noisy.detach().float().norm(dim=-1).mean(),
            "hierarchical_mmdit_noisy_gate_mean": noisy_gate_mean,
            "hierarchical_mmdit_step_state_norm": torch.stack(step_state_rows).mean(),
            "hierarchical_mmdit_refine_steps": torch.tensor(float(self.refine_steps), device=device),
            "hierarchical_mmdit_distinct_blocks": torch.tensor(
                float(self.refine_block_count), device=device
            ),
            "hierarchical_mmdit_full_rank_block_count": torch.tensor(
                float(self.refine_block_count), device=device
            ),
            "hierarchical_mmdit_shared_core_count": torch.zeros((), device=device),
            "hierarchical_mmdit_operator_stage_count": torch.tensor(
                float(self.operator_stage_count), device=device
            ),
            "hierarchical_mmdit_refine_block_count": torch.tensor(
                float(self.refine_block_count), device=device
            ),
            "hierarchical_mmdit_mandatory_low_rank_writer": torch.zeros((), device=device),
            "hierarchical_mmdit_shared_full_rank_path": torch.zeros((), device=device),
            "hierarchical_mmdit_distinct_full_rank_path": torch.ones((), device=device),
            "hierarchical_mmdit_step_conditioned_full_rank": torch.ones((), device=device),
            "hierarchical_mmdit_shared_base_scale_identifiable": torch.zeros((), device=device),
            "hierarchical_mmdit_shared_base_bias_free": torch.zeros((), device=device),
            "hierarchical_mmdit_stage_nested_contraction": torch.ones((), device=device),
            "hierarchical_mmdit_contraction_sidecar": torch.ones((), device=device),
            "hierarchical_mmdit_post_gate_sidecar": torch.ones((), device=device),
            "hierarchical_mmdit_shared_amplitude_owner": torch.ones((), device=device),
            "hierarchical_mmdit_duplicate_amplitude_owner": torch.zeros((), device=device),
            "hierarchical_mmdit_operator_geometry_identifiable": torch.ones((), device=device),
            "hierarchical_mmdit_operator_boundary_identity": torch.ones((), device=device),
            "hierarchical_mmdit_operator_nested_path": torch.ones((), device=device),
            "hierarchical_mmdit_operator_continuous_depth": torch.ones((), device=device),
            "hierarchical_mmdit_operator_nonexpansive": torch.ones((), device=device),
            "hierarchical_mmdit_operator_post_contraction_renorm": torch.zeros((), device=device),
            "hierarchical_mmdit_operator_stage_local_selection": torch.ones((), device=device),
            "hierarchical_mmdit_block_state_normalized": torch.ones((), device=device),
            "hierarchical_mmdit_factor_cache_per_forward": torch.ones((), device=device),
            "hierarchical_mmdit_operator_rank": torch.tensor(
                float(self.config.hierarchical_mmdit_operator_rank), device=device
            ),
            "hierarchical_mmdit_operator_groups": torch.tensor(
                float(self.config.hierarchical_mmdit_operator_groups), device=device
            ),
            "hierarchical_mmdit_operator_contraction_progress": (
                self.contraction_progress.detach().float()
            ),
            "hierarchical_mmdit_stage_selector_entropy": active_mean(
                selector_entropy_rows
            ),
            "hierarchical_mmdit_stage_selector_max": active_mean(
                selector_max_rows
            ),
            "hierarchical_mmdit_stage_selector_exploration": active_mean(
                selector_exploration_rows
            ),
            "hierarchical_mmdit_stage_selector_query_change": (
                selector_query_change.detach().float()
            ),
            "hierarchical_mmdit_stage_selector_same_block_query_change": (
                selector_same_block_query_change.detach().float()
            ),
            "hierarchical_mmdit_stage_selector_dynamic_state": torch.ones(
                (), device=device
            ),
            "hierarchical_mmdit_stage_selector_read_only_inputs": torch.ones(
                (), device=device
            ),
            "hierarchical_mmdit_exit_logits": exit_logit_rows,
            "hierarchical_mmdit_exit_probability": (
                exit_probability_rows.detach().float() * exit_candidate_float
            ).sum() / exit_candidate_denominator,
            "hierarchical_mmdit_exit_candidate_rate": exit_candidate_float.mean(),
            "hierarchical_mmdit_post_block_exit_controller": torch.ones(
                (), device=device
            ),
            "hierarchical_mmdit_exit_controller_read_only_inputs": torch.ones(
                (), device=device
            ),
            "hierarchical_mmdit_executed_steps": executed_steps.detach().float().mean(),
            "hierarchical_mmdit_early_exit_rate": (
                executed_steps.detach().float() < float(self.refine_steps)
            ).float().mean(),
            "hierarchical_mmdit_stage_advance_rate": stage_advance_rate,
            "hierarchical_mmdit_block_advance_rate": block_advance_rate,
            "hierarchical_mmdit_action_response_abs": active_mean(response_abs_rows),
            "hierarchical_mmdit_action_response_rel": active_mean(response_rel_rows),
            "hierarchical_mmdit_action_response_p25": active_quantile(response_rel_rows, 0.25),
            "hierarchical_mmdit_action_response_p50": active_quantile(response_rel_rows, 0.50),
            "hierarchical_mmdit_action_response_p75": active_quantile(response_rel_rows, 0.75),
            "hierarchical_mmdit_action_response_arm": active_mean(response_arm_rows),
            "hierarchical_mmdit_action_response_gripper": active_mean(response_gripper_rows),
            "hierarchical_mmdit_action_response_arm_null": active_mean(response_arm_null_rows),
            "hierarchical_mmdit_action_response_gripper_null": active_mean(
                response_gripper_null_rows
            ),
            "hierarchical_mmdit_stage_pressure_abs": active_mean(pressure_abs_rows),
            "hierarchical_mmdit_stage_pressure_rel": active_mean(pressure_rel_rows),
            "hierarchical_mmdit_stage_pressure_p25": active_quantile(pressure_rel_rows, 0.25),
            "hierarchical_mmdit_stage_pressure_p50": active_quantile(pressure_rel_rows, 0.50),
            "hierarchical_mmdit_stage_pressure_p75": active_quantile(pressure_rel_rows, 0.75),
            "hierarchical_mmdit_unresolved_rate": unresolved_rows.detach().float().mean(),
            "hierarchical_mmdit_budget_exhausted_rate": (
                budget_exhausted_rows.detach().float().mean()
            ),
            "hierarchical_mmdit_final_stage": final_stage_rows.detach().float().mean(),
            "hierarchical_mmdit_final_block": final_block_rows.detach().float().mean(),
            "hierarchical_mmdit_random_dwell_active": torch.tensor(
                float(routing_mode == "random_dwell"), device=device
            ),
            "hierarchical_mmdit_adaptive_execution_active": torch.tensor(
                float(routing_mode == "adaptive"), device=device
            ),
            "hierarchical_mmdit_learned_execution_active": torch.tensor(
                float(routing_mode == "learned"), device=device
            ),
            "hierarchical_mmdit_single_consumption_owner": torch.ones((), device=device),
            "hierarchical_mmdit_serial_composition": torch.ones((), device=device),
            "hierarchical_mmdit_competitive_market": torch.zeros((), device=device),
            "owned_workspace_state_pre_dit": torch.ones((), device=device),
            "owned_workspace_trajectory_is_proposal": torch.ones((), device=device),
            "intent_noisy_input_present": torch.zeros((), device=device),
            "owned_workspace_fixed_role_prior": torch.ones((), device=device),
            "owned_workspace_role_count": torch.tensor(
                float(len(self.workspace.memory_bank.ROLE_NAMES)), device=device
            ),
            # Raw detached probe fields deliberately avoid the diagnostic
            # prefix; runtime reduces them into scalar logs before aggregation.
            "refinement_probe_pred_velocity": probe_predictions.detach(),
            "refinement_probe_action_response_abs": response_abs_rows.detach(),
            "refinement_probe_action_response_rel": response_rel_rows.detach(),
            "refinement_probe_stage_pressure_abs": pressure_abs_rows.detach(),
            "refinement_probe_stage_pressure_rel": pressure_rel_rows.detach(),
            "refinement_probe_stage_ids": stage_id_rows.detach(),
            "refinement_probe_block_ids": block_id_rows.detach(),
            "refinement_probe_active": active_rows.detach(),
            "refinement_probe_exit_candidates": exit_candidate_rows.detach(),
            **initializer_metrics,
        }
        for time_bin in range(3):
            for label, quantile in (("p25", 0.25), ("p50", 0.50), ("p75", 0.75)):
                result[
                    f"hierarchical_mmdit_action_response_t{time_bin}_{label}"
                ] = active_time_quantile(response_rel_rows, time_bin, quantile)
                result[
                    f"hierarchical_mmdit_stage_pressure_t{time_bin}_{label}"
                ] = active_time_quantile(pressure_rel_rows, time_bin, quantile)
        for stage_index in range(self.operator_stage_count):
            result[f"hierarchical_mmdit_stage_{stage_index}_usage"] = (
                ((stage_id_rows == stage_index).float() * active_float).sum() / active_denominator
            )
        for block_index in range(self.refine_block_count):
            result[f"hierarchical_mmdit_block_{block_index}_usage"] = (
                ((block_id_rows == block_index).float() * active_float).sum()
                / active_denominator
            )
        for step_index in range(self.refine_steps):
            step_active = active_float[:, step_index]
            step_denominator = step_active.sum().clamp_min(1.0)
            result[f"hierarchical_mmdit_step_{step_index}_active_rate"] = step_active.mean()
            result[f"hierarchical_mmdit_step_{step_index}_action_response"] = (
                response_rel_rows[:, step_index].detach().float() * step_active
            ).sum() / step_denominator
            result[f"hierarchical_mmdit_step_{step_index}_stage_pressure"] = (
                pressure_rel_rows[:, step_index].detach().float() * step_active
            ).sum() / step_denominator
        if shadow is not None:
            shadow_steps = shadow["executed_steps"]
            shadow_unresolved = shadow["unresolved_rows"]
            shadow_budget_exhausted = shadow["budget_exhausted_rows"]
            shadow_stage_ids = shadow["stage_id_rows"]
            shadow_block_ids = shadow["block_id_rows"]
            shadow_active = shadow["active_rows"]
            if all(torch.is_tensor(value) for value in (
                shadow_steps, shadow_unresolved, shadow_budget_exhausted,
                shadow_stage_ids, shadow_block_ids, shadow_active,
            )):
                result["hierarchical_mmdit_shadow_executed_steps"] = shadow_steps.float().mean()
                result["hierarchical_mmdit_shadow_unresolved_rate"] = shadow_unresolved.float().mean()
                result["hierarchical_mmdit_shadow_budget_exhausted_rate"] = (
                    shadow_budget_exhausted.float().mean()
                )
                result["hierarchical_mmdit_shadow_early_exit_rate"] = (
                    shadow_steps.float() < float(self.refine_steps)
                ).float().mean()
                shadow_count = shadow_active.float().sum(dim=1).long().clamp_min(1) - 1
                final_shadow_stage = shadow_stage_ids.gather(1, shadow_count[:, None]).squeeze(1)
                result["hierarchical_mmdit_shadow_final_stage"] = final_shadow_stage.float().mean()
                final_shadow_block = shadow_block_ids.gather(
                    1, shadow_count[:, None]
                ).squeeze(1)
                result["hierarchical_mmdit_shadow_final_block"] = (
                    final_shadow_block.float().mean()
                )
                shadow_denominator = shadow_active.float().sum().clamp_min(1.0)
                for stage_index in range(self.operator_stage_count):
                    result[f"hierarchical_mmdit_shadow_stage_{stage_index}_usage"] = (
                        (shadow_stage_ids == stage_index).float() * shadow_active.float()
                    ).sum() / shadow_denominator
                for block_index in range(self.refine_block_count):
                    result[f"hierarchical_mmdit_shadow_block_{block_index}_usage"] = (
                        (shadow_block_ids == block_index).float() * shadow_active.float()
                    ).sum() / shadow_denominator
                shadow_predictions = shadow.get("prediction_rows")
                if torch.is_tensor(shadow_predictions):
                    result["refinement_shadow_probe_pred_velocity"] = (
                        shadow_predictions.detach()
                    )
                    result["refinement_shadow_probe_active"] = shadow_active.detach()
        organizer_metrics = organized["metrics"]
        if isinstance(organizer_metrics, dict):
            result.update({key: value for key, value in organizer_metrics.items() if torch.is_tensor(value)})
        result.update({
            key: value
            for key, value in contracts.items()
            if key.startswith("intent_") and torch.is_tensor(value)
        })
        workspace_mean = self._mean_metrics(workspace_rows)
        for key, value in workspace_mean.items():
            result[f"owned_{key}"] = value
        mmdit_mean = self._mean_metrics(mmdit_rows)
        for key, value in mmdit_mean.items():
            if not key.endswith("_rows"):
                result[f"hierarchical_mmdit_{key}"] = value
        depth_costs = [
            row[f"action_{branch}_depth_usage_cost"]
            for row in mmdit_rows
            for branch in OwnedHierarchicalActionBlock._BRANCH_NAMES
            if f"action_{branch}_depth_usage_cost" in row
        ]
        if depth_costs:
            result["hierarchical_mmdit_depth_usage_regularizer"] = torch.stack(
                depth_costs
            ).mean()
        if mmdit_rows:
            measured_steps = len(mmdit_rows)
            measured_stage_ids = stage_id_rows[:, :measured_steps]
            measured_block_ids = block_id_rows[:, :measured_steps]
            measured_active = active_float[:, :measured_steps]
            stage_masks = tuple(
                measured_active * (measured_stage_ids == operator_stage).float()
                for operator_stage in range(self.operator_stage_count)
            )
            stage_denominators = tuple(mask.sum().clamp_min(1.0) for mask in stage_masks)
            block_masks = tuple(
                measured_active * (measured_block_ids == block_index).float()
                for block_index in range(self.refine_block_count)
            )
            block_denominators = tuple(mask.sum().clamp_min(1.0) for mask in block_masks)
            for branch in OwnedHierarchicalActionBlock._BRANCH_NAMES:
                metric_values: dict[str, Tensor] = {}
                for metric_name in (
                    "depth_ratio_rows",
                    "effective_depth_rows",
                    "contraction_ratio_rows",
                    "subspace_energy_fraction_rows",
                    "removed_fraction_rows",
                    "nonexpansive_violation_rows",
                    "operator_gain_rows",
                    "update_rms_rows",
                    "host_update_rms_rows",
                    "base_gate_rows",
                    "direction_change_rows",
                ):
                    key = f"action_{branch}_{metric_name}"
                    if all(key in row for row in mmdit_rows):
                        metric_values[metric_name] = torch.stack(
                            [row[key] for row in mmdit_rows], dim=1
                        ).detach().float()
                for operator_stage, (stage_mask, stage_denominator) in enumerate(
                    zip(stage_masks, stage_denominators, strict=True)
                ):
                    for metric_name, values in metric_values.items():
                        output_name = metric_name.removesuffix("_rows")
                        result[
                            f"hierarchical_mmdit_stage_{operator_stage}_{branch}_{output_name}"
                        ] = (values * stage_mask).sum() / stage_denominator
                for block_index, (block_mask, block_denominator) in enumerate(
                    zip(block_masks, block_denominators, strict=True)
                ):
                    for metric_name, values in metric_values.items():
                        output_name = metric_name.removesuffix("_rows")
                        result[
                            f"hierarchical_mmdit_block_{block_index}_{branch}_{output_name}"
                        ] = (values * block_mask).sum() / block_denominator
        if all(key in mmdit_mean for key in (
            "action_noisy_update_fraction_rows",
            "action_workspace_update_fraction_rows",
            "action_low_update_fraction_rows",
            "action_stage_update_fraction_rows",
        )):
            stratified = time_stratified_attention(
                time,
                mmdit_mean["action_noisy_update_fraction_rows"],
                mmdit_mean["action_workspace_update_fraction_rows"],
                mmdit_mean["action_low_update_fraction_rows"],
                mmdit_mean["action_stage_update_fraction_rows"],
            )
            for key, value in stratified.items():
                renamed = key.replace("_attn_", "_update_fraction_")
                result[f"hierarchical_{renamed}"] = value
        return result
