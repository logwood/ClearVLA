from __future__ import annotations

"""Current serial-owned hierarchical MMDiT action decoder."""

import math
from typing import Protocol

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .codec import PhysicalActionTokenLift
from .evidence import HierarchicalEvidenceWorkspace
from .gauges import time_stratified_attention
from .intent import IndependentIntentFusion, IntentContractCompiler, PolicyConditionOrganizer
from .primitives import BiasFreeFFN, TimeEmbedding, sinusoidal_positions


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
    """Serial action refinement with one explicit function per condition role.

    Unlike the historical MMDiT market, noisy/stage/low evidence never compete
    for one softmax budget.  Each branch transforms the state produced by the
    preceding branch, so depth is function composition rather than a wider set
    of interchangeable residual writers.
    """

    _BRANCH_NAMES = ("self", "noisy", "stage", "low", "ffn")

    def __init__(self, config: PolicyDecoderConfig) -> None:
        super().__init__()
        h = int(config.hidden_size)
        heads = int(config.num_heads)
        if h % heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.config = config
        self.hidden_size = h
        self.heads = heads
        self.head_dim = h // heads
        self.scale_max = float(config.hierarchical_mmdit_residual_scale_max)
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
        # Shared AdaLN geometry plus five scalar LayerScale controls.  The
        # controls are bounded; they regulate numerical step size, not whether
        # a semantic source exists in the graph.
        self.mod = nn.Linear(h, 2 * h + len(self._BRANCH_NAMES))
        self.drop = nn.Dropout(float(config.dropout))
        self.out_norm = nn.LayerNorm(h, elementwise_affine=False)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)
        initial = {
            "self": 0.02,
            "noisy": 0.08,
            "stage": 0.04,
            "low": 0.06,
            "ffn": 0.02,
        }
        with torch.no_grad():
            for index, name in enumerate(self._BRANCH_NAMES):
                ratio = min(max(initial[name] / self.scale_max, -0.999), 0.999)
                self.mod.bias[2 * h + index] = math.atanh(ratio)

    @staticmethod
    def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        return x * (1.0 + scale[:, None]) + shift[:, None]

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

    @staticmethod
    def _sample_rms(x: Tensor) -> Tensor:
        """RMS over the complete token field controlled by one sample gate."""
        return x.float().square().mean(dim=(1, 2)).clamp_min(0.0).sqrt()

    @classmethod
    def _normalize_residual(cls, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Remove branch-wide scale while preserving relative horizon amplitudes."""
        raw = x.float()
        denominator = raw.square().mean(dim=(1, 2), keepdim=True).add(1e-6).sqrt()
        normalized = (raw / denominator).to(dtype=x.dtype)
        return normalized, cls._sample_rms(x).detach(), cls._sample_rms(normalized).detach()

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
        kv_proj: nn.Linear,
        out_proj: nn.Linear,
        shift: Tensor,
        scale: Tensor,
        gate: Tensor,
        mask: Tensor | None = None,
        value_gate: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        query_value = self._modulate(self.state_norm(action), shift, scale)
        q = self._split_heads(self.cross_q(query_value))
        key, value = kv_proj(self.condition_norm(memory)).chunk(2, dim=-1)
        k = self._split_heads(key)
        v = self._split_heads(value)
        attended, weight = self._attention(q, k, v, mask)
        projected = self.drop(out_proj(self._merge_heads(attended)))
        direction, projected_rms, normalized_rms = self._normalize_residual(projected)
        amplitude = gate
        if value_gate is not None:
            if tuple(value_gate.shape) != (int(action.shape[0]),):
                raise ValueError(
                    "cross-update value_gate must be one scalar per sample, got "
                    f"{tuple(value_gate.shape)}"
                )
            amplitude = amplitude * value_gate.to(device=amplitude.device, dtype=amplitude.dtype)
        # All scale controls act after non-affine normalization. Projection
        # weights can choose direction/content, but cannot counterfeit amplitude.
        update = amplitude[:, None, None] * direction
        realized_scale = self._sample_rms(update).detach() / normalized_rms.clamp_min(1e-8)
        return action + update, update, weight, projected_rms, normalized_rms, realized_scale

    def forward(
        self,
        action: Tensor,
        *,
        noisy_tokens: Tensor,
        stage_tokens: Tensor,
        low_tokens: Tensor,
        global_cond: Tensor,
        noisy_value_gate: Tensor | None = None,
        low_role_ids: Tensor | None = None,
        low_role_names: tuple[str, ...] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        before = action
        mod = self.mod(self.global_norm(global_cond))
        # Bound AdaLN itself as well as the residual gates.  Otherwise a small
        # residual gate can coexist with an arbitrarily large normalized
        # query/value transform and recreate the old scale gauge internally.
        shift = 0.5 * torch.tanh(mod[:, :self.hidden_size])
        scale = 0.5 * torch.tanh(mod[:, self.hidden_size:2 * self.hidden_size])
        gates = self.scale_max * torch.tanh(mod[:, 2 * self.hidden_size:])

        value = self._modulate(self.state_norm(action), shift, scale)
        sq, sk, sv = (self._split_heads(part) for part in self.self_qkv(value).chunk(3, dim=-1))
        self_mask = torch.triu(
            torch.ones(int(action.shape[1]), int(action.shape[1]), device=action.device, dtype=torch.bool),
            diagonal=1,
        )
        self_attended, self_weight = self._attention(sq, sk, sv, self_mask)
        self_projected = self.drop(self.self_out(self._merge_heads(self_attended)))
        self_direction, self_projected_rms, self_normalized_rms = self._normalize_residual(self_projected)
        self_update = gates[:, 0, None, None] * self_direction
        self_realized_scale = self._sample_rms(self_update).detach() / self_normalized_rms.clamp_min(1e-8)
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
            noisy_projected_rms,
            noisy_normalized_rms,
            noisy_realized_scale,
        ) = self._cross_update(
            action,
            noisy_tokens,
            kv_proj=self.noisy_kv,
            out_proj=self.noisy_out,
            shift=shift,
            scale=scale,
            gate=gates[:, 1],
            mask=noisy_mask,
            value_gate=noisy_value_gate,
        )
        (
            action,
            stage_update,
            stage_weight,
            stage_projected_rms,
            stage_normalized_rms,
            stage_realized_scale,
        ) = self._cross_update(
            action,
            stage_tokens,
            kv_proj=self.stage_kv,
            out_proj=self.stage_out,
            shift=shift,
            scale=scale,
            gate=gates[:, 2],
        )
        (
            action,
            low_update,
            low_weight,
            low_projected_rms,
            low_normalized_rms,
            low_realized_scale,
        ) = self._cross_update(
            action,
            low_tokens,
            kv_proj=self.low_kv,
            out_proj=self.low_out,
            shift=shift,
            scale=scale,
            gate=gates[:, 3],
        )
        ffn_value = self._modulate(self.state_norm(action), shift, scale)
        ffn_projected = self.drop(self.ffn(ffn_value))
        ffn_direction, ffn_projected_rms, ffn_normalized_rms = self._normalize_residual(ffn_projected)
        ffn_update = gates[:, 4, None, None] * ffn_direction
        ffn_realized_scale = self._sample_rms(ffn_update).detach() / ffn_normalized_rms.clamp_min(1e-8)
        pre_norm_action = action + ffn_update
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
        pre_norm_rows = branch_sum.norm(dim=-1).mean(dim=1)
        output_norm_rows = self._row_norm(action - pre_norm_action)
        before_rows = self._row_norm(before).clamp_min(1e-6)
        self_entropy, self_max = self._attention_stats(self_weight)
        noisy_entropy, noisy_max = self._attention_stats(noisy_weight)
        stage_entropy, stage_max = self._attention_stats(stage_weight)
        low_entropy, low_max = self._attention_stats(low_weight)
        metrics: dict[str, Tensor] = {
            "action_update_norm": total_rows.mean(),
            "action_update_ratio": (total_rows / before_rows).mean(),
            "action_pre_norm_update_norm": pre_norm_rows.mean(),
            "action_output_norm_update_norm": output_norm_rows.mean(),
            "action_output_norm_update_ratio": (
                output_norm_rows / pre_norm_rows.clamp_min(1e-6)
            ).mean(),
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
        projected_rms_rows = (
            self_projected_rms,
            noisy_projected_rms,
            stage_projected_rms,
            low_projected_rms,
            ffn_projected_rms,
        )
        normalized_rms_rows = (
            self_normalized_rms,
            noisy_normalized_rms,
            stage_normalized_rms,
            low_normalized_rms,
            ffn_normalized_rms,
        )
        realized_scale_rows = (
            self_realized_scale,
            noisy_realized_scale,
            stage_realized_scale,
            low_realized_scale,
            ffn_realized_scale,
        )
        expected_scale_rows = [gates[:, index].detach().float().abs() for index in range(len(self._BRANCH_NAMES))]
        if noisy_value_gate is not None:
            expected_scale_rows[1] = expected_scale_rows[1] * noisy_value_gate.detach().float().abs()
        for index, name in enumerate(self._BRANCH_NAMES):
            metrics[f"action_{name}_gate"] = gates[:, index].detach().float().mean()
            metrics[f"action_{name}_gate_abs_mean"] = gates[:, index].detach().float().abs().mean()
            metrics[f"action_{name}_projected_rms"] = projected_rms_rows[index].mean()
            metrics[f"action_{name}_normalized_rms"] = normalized_rms_rows[index].mean()
            metrics[f"action_{name}_realized_scale"] = realized_scale_rows[index].mean()
            metrics[f"action_{name}_gate_scale_error"] = (
                realized_scale_rows[index] - expected_scale_rows[index]
            ).abs().mean()
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
    """Owned-evidence deterministic action decoder used before V76 depth work."""

    def __init__(self, config: PolicyDecoderConfig) -> None:
        super().__init__()
        self.config = config
        h = int(config.hidden_size)
        self.hidden_size = h
        self.block_count = int(config.hierarchical_mmdit_depth)
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
        # This decoder owns progress through block_identity + budget_proj and
        # always supplies step_state_override. Keep the fallback parameter in
        # the state dict for checkpoint compatibility, but do not optimize an
        # unreachable second definition of decoder time.
        self.workspace.step_embedding.requires_grad_(False)
        self.blocks = nn.ModuleList([
            OwnedHierarchicalActionBlock(config)
            for _ in range(self.block_count)
        ])
        self.block_identity = nn.Parameter(torch.randn(1, self.block_count, h) * 0.02)
        self.budget_proj = nn.Sequential(nn.Linear(2, h), nn.SiLU(), nn.Linear(h, h))
        self.step_state_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.workspace_condition = nn.Sequential(
            nn.LayerNorm(2 * h),
            nn.Linear(2 * h, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.LayerNorm(h, elementwise_affine=False),
        )
        self.global_condition = nn.Sequential(
            nn.LayerNorm(3 * h),
            nn.Linear(3 * h, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.LayerNorm(h, elementwise_affine=False),
        )
        self.condition_type = nn.Parameter(torch.randn(1, 3, h) * 0.02)
        self.action_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.velocity_head = ActionOnlyPhysicalVelocityHead(config)
        self.event_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 3))
        self.motion_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 1))
        # CR7 fallback (do_before_v76 §23): a dedicated, restricted output
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
        # State-dict compatibility with the short-lived competitive-market
        # decoder.  The serial decoder has a mandatory noisy branch, so a
        # learnable group-logit subsidy is neither consumed nor optimized.
        self.register_buffer("noisy_market_bias", torch.zeros(()), persistent=True)
        self._initialize_outputs()

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

    def _step_state(
        self,
        step_index: int,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, int]:
        block_index = min(max(int(step_index), 0), self.block_count - 1)
        if self.refine_steps <= 1:
            progress = 1.0
        else:
            progress = float(step_index) / float(self.refine_steps - 1)
        remaining = float(self.refine_steps - step_index - 1) / float(max(self.refine_steps, 1))
        budget = torch.tensor([progress, remaining], device=device, dtype=dtype)[None].expand(batch_size, -1)
        identity = self.block_identity[:, block_index].to(device=device, dtype=dtype).expand(batch_size, -1)
        return self.step_state_norm(identity + self.budget_proj(budget)), block_index

    def _gate_noisy_tokens(self, noisy: Tensor, time: Tensor) -> tuple[Tensor, Tensor | None, Tensor]:
        """Return (tokens, per-sample value gate or None, gate mean gauge).

        Tokens are returned UNGATED in both modes: pre-block multiplicative
        scaling is arithmetically cancelled by the block's scale-invariant
        cond_norm (LayerNorm), which is why the historical outer gate never
        did anything.  Mode 0 codifies that no-gate regime honestly (gauge
        reads 1.0); mode 1 hands the g(t) schedule to the block, which applies
        it after non-affine branch normalization where it cannot be cancelled.
        """
        if int(getattr(self.config, "hierarchical_mmdit_noisy_gate_mode", 0)) == 0:
            return noisy, None, torch.ones((), device=noisy.device, dtype=torch.float32)
        minimum = float(self.config.hierarchical_mmdit_noisy_gate_min)
        power = float(self.config.hierarchical_mmdit_noisy_gate_power)
        gate = minimum + (1.0 - minimum) * time.float().clamp(0.0, 1.0).pow(power)
        return noisy, gate, gate.detach().mean()

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
        noisy, noisy_value_gate, noisy_gate_mean = self._gate_noisy_tokens(self.noisy_action_lift(noisy_physical), time)
        time_state = self.time_lift(self.time(time.to(dtype=dtype)))
        workspace_condition = self.workspace_condition(torch.cat([
            contracts["global_intent"],
            time_state,
        ], dim=-1))
        workspace_rows: list[dict[str, Tensor]] = []
        mmdit_rows: list[dict[str, Tensor]] = []
        condition_norm_rows: list[Tensor] = []
        step_state_rows: list[Tensor] = []
        for step_index in range(self.refine_steps):
            step_state, block_index = self._step_state(
                step_index,
                batch_size=batch,
                device=device,
                dtype=dtype,
            )
            global_condition = self.global_condition(torch.cat([
                contracts["global_intent"],
                time_state,
                step_state,
            ], dim=-1))
            (
                low,
                stage_content,
                stage_for_action,
                _low_logit_bias,
                _stage_logit_bias,
                workspace_metrics,
            ) = self.workspace.step(
                prepared_evidence=prepared,
                stage_content=stage_content,
                primary_cond=workspace_condition,
                step_index=step_index,
                read_contract=contracts["read_contract"],
                step_state_override=step_state,
            )
            low = low + self.condition_type[:, 0:1].to(device=device, dtype=dtype)
            stage_for_action = stage_for_action + self.condition_type[:, 1:2].to(device=device, dtype=dtype)
            noisy_typed = noisy + self.condition_type[:, 2:3].to(device=device, dtype=dtype)
            action, mmdit_metrics = self.blocks[block_index](
                action,
                noisy_tokens=noisy_typed,
                stage_tokens=stage_for_action,
                low_tokens=low,
                global_cond=global_condition,
                noisy_value_gate=noisy_value_gate,
                low_role_ids=self.workspace.low_slot_role_ids,
                low_role_names=self.workspace.memory_bank.ROLE_NAMES,
            )
            workspace_rows.append(workspace_metrics)
            mmdit_rows.append(mmdit_metrics)
            condition_norm_rows.append(torch.stack([
                low.detach().float().norm(dim=-1).mean(),
                stage_for_action.detach().float().norm(dim=-1).mean(),
                noisy_typed.detach().float().norm(dim=-1).mean(),
            ]).mean())
            step_state_rows.append(step_state.detach().float().norm(dim=-1).mean())

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
            "hierarchical_mmdit_noisy_market_bias": self.noisy_market_bias.detach().float(),
            "hierarchical_mmdit_step_state_norm": torch.stack(step_state_rows).mean(),
            "hierarchical_mmdit_refine_steps": torch.tensor(float(self.refine_steps), device=device),
            "hierarchical_mmdit_distinct_blocks": torch.tensor(float(self.block_count), device=device),
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
            **initializer_metrics,
        }
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
