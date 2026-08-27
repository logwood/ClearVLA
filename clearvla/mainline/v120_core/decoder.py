"""Current serial-owned hierarchical MMDiT action decoder."""

from __future__ import annotations

import math
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .codec import (
    DCTFlowCodec,
    FrequencyPhysicalActionTokenLift,
    NativeTimePhysicalActionTokenLift,
    PhysicalActionCodec,
    SoftSpectralAperture,
    TemporalDCT,
)
from .controller import UnifiedControllerOutput, UnifiedHierarchicalController
from .evidence import (
    HierarchicalEvidenceWorkspace,
    PreparedEvidenceMemory,
    WorkspaceControlOverride,
)
from .gauges import (
    deterministic_module_probe,
    fp32_diagnostic,
    time_stratified_attention,
)
from .intent import IndependentIntentFusion, IntentContractCompiler, PolicyConditionOrganizer
from .primitives import BiasFreeFFN, TimeEmbedding, sinusoidal_positions
from .refinement import NestedLowRankContractionBank


class PolicyDecoderConfig(Protocol):
    hidden_size: int
    action_horizon: int
    num_heads: int
    dropout: float
    arm_dim: int
    arm_flow_mode: str
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
    hierarchical_mmdit_unified_controller: int
    hierarchical_mmdit_control_tokens: int
    hierarchical_mmdit_controller_depth: int
    hierarchical_mmdit_controller_heads: int
    hierarchical_mmdit_controller_ffn_expansion: float
    hierarchical_mmdit_spectral_state: int
    hierarchical_mmdit_spectral_arm_start_fraction: float
    hierarchical_mmdit_spectral_gripper_start_fraction: float
    hierarchical_mmdit_spectral_temperature: float
    hierarchical_mmdit_spectral_schedule_power: float
    hierarchical_mmdit_spectral_controller_shift_limit: float
    hierarchical_mmdit_spectral_competition_loss_weight: float
    hierarchical_mmdit_spectral_competition_warmup_steps: int
    hierarchical_mmdit_operation_candidate_probes: int
    hierarchical_mmdit_operation_value_warmup_steps: int
    hierarchical_mmdit_dwell_mode: str
    hierarchical_mmdit_execution_contract: str
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

    def __init__(
        self,
        config: PolicyDecoderConfig,
        *,
        frequency_positions: bool = False,
    ) -> None:
        super().__init__()
        h = int(config.hidden_size)
        horizon = int(config.action_horizon)
        self.hidden_size = h
        self.horizon = horizon
        self.frequency_positions = bool(frequency_positions)
        self.seed = nn.Parameter(torch.randn(1, horizon, h) * 0.02)
        self.register_buffer(
            "horizon_position",
            sinusoidal_positions(
                range(
                    0 if frequency_positions else 1,
                    horizon + (0 if frequency_positions else 1),
                ),
                h,
            )[None],
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
        causal = None
        if not self.frequency_positions:
            causal = torch.triu(
                torch.ones(
                    self.horizon,
                    self.horizon,
                    device=device,
                    dtype=torch.bool,
                ),
                diagonal=1,
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
            )
            .norm(dim=-1)
            .mean(),
        }


class ActionOnlyPhysicalVelocityHead(nn.Module):
    """Native-time velocity readout followed by deterministic field analysis."""

    def __init__(self, config: PolicyDecoderConfig) -> None:
        super().__init__()
        h = int(config.hidden_size)
        ad = int(config.arm_dim)
        self.codec = PhysicalActionCodec(config)
        self.norm = nn.LayerNorm(h)
        host_rng_state = torch.get_rng_state()
        try:
            self.gripper_gate = nn.Linear(h, h, bias=False)
            nn.init.zeros_(self.gripper_gate.weight)
        finally:
            # The new exact-zero owner must not perturb any inherited output
            # initialization or the loader RNG derived after model creation.
            torch.set_rng_state(host_rng_state)
        self.arm_manifold = str(config.arm_flow_mode) == "manifold_native"
        self.parseval_gripper = str(config.gripper_field_mode) == "parseval_temporal"
        if self.arm_manifold:
            self.arm_native = nn.Linear(h, ad)
            self.arm_abs = None
            self.arm_delta = None
        else:
            self.arm_abs = nn.Linear(h, ad)
            self.arm_delta = nn.Linear(h, ad)
            self.arm_native = None
        if self.parseval_gripper:
            self.grip_native = nn.Linear(h, 1)
            self.grip_value = None
            self.grip_delta = None
            self.grip_extra = None
        else:
            self.grip_native = None
            self.grip_value = nn.Linear(h, 1)
            self.grip_delta = nn.Linear(h, 1)
            self.grip_extra = nn.Linear(h, max(int(config.gripper_field_dim) - 2, 0))

    def output_layers(self) -> tuple[nn.Linear, ...]:
        arm_layers = (
            (self.arm_native,) if self.arm_native is not None else (self.arm_abs, self.arm_delta)
        )
        gripper_layers = (
            (self.grip_native,)
            if self.grip_native is not None
            else (self.grip_value, self.grip_delta, self.grip_extra)
        )
        if any(layer is None for layer in (*arm_layers, *gripper_layers)):
            raise RuntimeError("physical velocity head has an incomplete chart")
        return tuple(layer for layer in (*arm_layers, *gripper_layers) if layer is not None)

    def forward_with_gripper_state(
        self,
        tokens: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return the physical field and its continuous gripper-private owner."""

        base_read = self.norm(tokens)
        gate = torch.tanh(self.gripper_gate(base_read))
        gripper_state = tokens + tokens * gate.to(dtype=tokens.dtype)
        gripper_read = self.norm(gripper_state)
        if self.arm_native is not None:
            arm_field = self.codec.encode_arm_tangent(self.arm_native(base_read))
        else:
            if self.arm_abs is None or self.arm_delta is None:
                raise RuntimeError("legacy arm velocity heads are not initialized")
            arm_field = torch.cat(
                [self.arm_abs(base_read), self.arm_delta(base_read)],
                dim=-1,
            )
        if self.grip_native is not None:
            grip_field = self.codec.encode_gripper_tangent(
                self.grip_native(gripper_read)
            )
        else:
            if self.grip_value is None or self.grip_delta is None or self.grip_extra is None:
                raise RuntimeError("legacy gripper velocity heads are not initialized")
            grip_parts = [
                self.grip_value(gripper_read),
                self.grip_delta(gripper_read),
            ]
            if int(self.grip_extra.out_features) > 0:
                grip_parts.append(self.grip_extra(base_read))
            grip_field = torch.cat(grip_parts, dim=-1)
        return torch.cat([arm_field, grip_field], dim=-1), gripper_state, gate

    def forward(self, tokens: Tensor) -> Tensor:
        field, _, _ = self.forward_with_gripper_state(tokens)
        return field


class OwnedHierarchicalActionBlock(nn.Module):
    """A V77 refinement block with a post-gate contraction sidecar.

    The host block owns content, direction, AdaLN, and its original LayerScale
    gates. Semantic stages can only contract the complete gated branch update.
    A unified controller may retain a relative fraction afterwards, but cannot
    replace or reparameterize the host amplitude.
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
        # These rows remain the sole owner of host residual amplitude.  A
        # controller keep is only a relative mask on the completed operation.
        base_step = float(config.hierarchical_mmdit_residual_scale_init)
        initial_steps = torch.tensor(
            (
                0.4 * base_step,
                1.6 * base_step,
                0.8 * base_step,
                1.2 * base_step,
                0.4 * base_step,
            ),
            dtype=torch.float32,
        )
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
    def _base_project(
        projection: nn.Linear,
        feature: Tensor,
        *,
        collect_diagnostics: bool = True,
    ) -> tuple[Tensor, Tensor]:
        output = projection(feature)
        if not collect_diagnostics:
            return output, feature.new_zeros((), dtype=torch.float32)
        weight = projection.weight
        target_norm = math.sqrt(float(projection.out_features))
        raw_parameter_rms = weight.float().square().sum().sqrt() / target_norm
        return output, raw_parameter_rms.detach()

    @classmethod
    def _normalize_residual(
        cls,
        x: Tensor,
        *,
        collect_diagnostics: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Preserve the V77 residual direction at the contraction boundary."""
        raw = x.float()
        denominator = raw.square().mean(dim=(1, 2), keepdim=True).add(1e-6).sqrt()
        direction = (raw / denominator).to(dtype=x.dtype)
        if not collect_diagnostics:
            denominator_rows = denominator[:, 0, 0].detach()
            return direction, denominator_rows, denominator_rows
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
        update_keep: Tensor | None = None,
        contraction_identity_bypass: bool | None = None,
        stage_candidates: Tensor | None = None,
        stage_probabilities: Tensor | None = None,
        raw_depth_ratio_override: Tensor | None = None,
        binary_group_selection: bool = False,
        prepared_factors: Tensor | None = None,
        collect_diagnostics: bool = True,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        base_output, base_parameter_rms = self._base_project(
            base_projection,
            projection_input,
            collect_diagnostics=collect_diagnostics,
        )
        projection_rms_rows = self._sample_rms(projection_input.detach())
        if tuple(base_gate.shape) != (int(projection_input.shape[0]),):
            raise ValueError("base_gate must contain one host gate per sample")
        if update_keep is None:
            update_keep_rows = torch.ones_like(base_gate, dtype=torch.float32)
        else:
            if tuple(update_keep.shape) != tuple(base_gate.shape):
                raise ValueError("update_keep must match base_gate")
            update_keep_rows = update_keep.float().clamp(0.0, 1.0)

        # Reconstruct the complete host branch first. Contraction consumes the
        # already gated host update. A direct caller may still pass the old
        # explicit ``update_keep`` for checkpoint-compatible low-level tests,
        # but the unified controller never supplies it; its capacity mask can
        # remove operator directions without owning their numerical scale.
        base_activated = self.drop(base_output)
        base_direction, base_activated_rms_rows, base_normalized_rms_rows = (
            self._normalize_residual(
                base_activated,
                collect_diagnostics=collect_diagnostics,
            )
        )
        host_update = base_gate[:, None, None].to(dtype=base_direction.dtype) * base_direction
        contracted_update, contraction_metrics = contraction(
            host_update,
            operator_cond,
            stage_index,
            stage_candidates=stage_candidates,
            stage_probabilities=stage_probabilities,
            contraction_progress=contraction_progress,
            prepared_factors=prepared_factors,
            identity_bypass=contraction_identity_bypass,
            raw_depth_ratio_override=raw_depth_ratio_override,
            binary_group_selection=binary_group_selection,
            collect_diagnostics=collect_diagnostics,
        )
        update = contracted_update * update_keep_rows[:, None, None].to(
            dtype=contracted_update.dtype
        )
        if not collect_diagnostics:
            return update, {}
        with torch.no_grad():
            base_rms_rows = self._sample_rms(base_output)
            host_update_rms_rows = self._sample_rms(host_update)
            contracted_rms_rows = self._sample_rms(contracted_update)
            update_rms_rows = self._sample_rms(update)
            direction_change_rows = self._sample_rms(contracted_update - host_update)
            direction_cosine_rows = F.cosine_similarity(
                contracted_update.float(),
                host_update.float(),
                dim=-1,
            ).mean(dim=1)
            expected_host_rms_rows = base_gate.detach().float().abs() * base_normalized_rms_rows
            gate_scale_error_rows = (host_update_rms_rows - expected_host_rms_rows).abs()
            keep_scale_error_rows = (
                update_rms_rows - contracted_rms_rows * update_keep_rows.detach().abs()
            ).abs()
            effective_gate_rows = base_gate.detach().float() * update_keep_rows.detach()
        metrics = dict(contraction_metrics)
        metrics.update(
            {
                "base_rms": base_rms_rows.detach().mean(),
                "base_parameter_rms": base_parameter_rms,
                "projection_input_rms": projection_rms_rows.detach().mean(),
                "base_data_gain": (base_rms_rows / projection_rms_rows.clamp_min(1e-6))
                .detach()
                .mean(),
                "base_activated_rms": base_activated_rms_rows.detach().mean(),
                "base_normalized_rms": base_normalized_rms_rows.detach().mean(),
                "host_update_rms": host_update_rms_rows.detach().mean(),
                "contracted_rms": contracted_rms_rows.detach().mean(),
                "direction_change": direction_change_rows.detach().mean(),
                "direction_cosine": direction_cosine_rows.detach().mean(),
                "base_gate": base_gate.detach().mean(),
                "base_gate_abs_mean": base_gate.detach().float().abs().mean(),
                "effective_gate": effective_gate_rows.mean(),
                "effective_gate_abs_mean": effective_gate_rows.abs().mean(),
                "gate_scale_error": gate_scale_error_rows.detach().mean(),
                "keep_scale_error": keep_scale_error_rows.detach().mean(),
                "update_keep": update_keep_rows.detach().mean(),
                "realized_scale": update_rms_rows.detach().mean(),
                "operator_gain": (update_rms_rows / projection_rms_rows.clamp_min(1e-6))
                .detach()
                .mean(),
                "update_rms": update_rms_rows.detach().mean(),
                "host_update_rms_rows": host_update_rms_rows.detach(),
                "contracted_rms_rows": contracted_rms_rows.detach(),
                "base_gate_rows": base_gate.detach(),
                "effective_gate_rows": effective_gate_rows,
                "update_keep_rows": update_keep_rows.detach(),
                "operator_gain_rows": (
                    update_rms_rows / projection_rms_rows.clamp_min(1e-6)
                ).detach(),
                "update_rms_rows": update_rms_rows.detach(),
                "direction_change_rows": direction_change_rows.detach(),
            }
        )
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
        pair_denominator = token_norm_sum.square() - branch_token_norms.square().sum(dim=2)
        pair_numerator = branch_sum.square().sum(dim=-1) - torch.stack(
            [update.square().sum(dim=-1) for update in detached_updates], dim=2
        ).sum(dim=2)
        weighted_pair_cosine = (
            torch.where(
                pair_denominator > 1e-8,
                pair_numerator / pair_denominator.clamp_min(1e-8),
                torch.zeros_like(pair_denominator),
            )
            .clamp(-1.0, 1.0)
            .mean()
        )
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
        update_keep: Tensor | None = None,
        contraction_identity_bypass: bool | None = None,
        stage_candidates: Tensor | None = None,
        stage_probabilities: Tensor | None = None,
        raw_depth_ratio_override: Tensor | None = None,
        binary_group_selection: bool = False,
        mask: Tensor | None = None,
        prepared_factors: Tensor | None = None,
        collect_diagnostics: bool = True,
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
            update_keep=update_keep,
            stage_candidates=stage_candidates,
            stage_probabilities=stage_probabilities,
            raw_depth_ratio_override=raw_depth_ratio_override,
            binary_group_selection=binary_group_selection,
            base_gate=base_gate,
            contraction_identity_bypass=contraction_identity_bypass,
            prepared_factors=prepared_factors,
            collect_diagnostics=collect_diagnostics,
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
        raw_depth_ratio_overrides: dict[str, Tensor] | None = None,
        branch_update_keeps: Tensor | None = None,
        binary_group_selection: bool = False,
        contraction_factors: dict[str, Tensor] | None = None,
        spectral_token_mask: Tensor | None = None,
        low_role_ids: Tensor | None = None,
        low_role_names: tuple[str, ...] | None = None,
        collect_diagnostics: bool = True,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        before = action
        normalized_shared = self.global_norm(shared_cond)
        mod = self.mod(normalized_shared)
        shift = 0.5 * torch.tanh(mod[:, : self.hidden_size])
        scale = 0.5 * torch.tanh(mod[:, self.hidden_size : 2 * self.hidden_size])
        legacy_gates = self.scale_max * torch.tanh(mod[:, 2 * self.hidden_size :])
        if branch_update_keeps is None:
            update_keeps = torch.ones_like(legacy_gates, dtype=torch.float32)
        else:
            expected = (int(action.shape[0]), len(self._BRANCH_NAMES))
            if tuple(branch_update_keeps.shape) != expected:
                raise ValueError(
                    f"branch_update_keeps must have shape {expected}, "
                    f"got {tuple(branch_update_keeps.shape)}"
                )
            update_keeps = branch_update_keeps.float().clamp(0.0, 1.0)
        if contraction_progress.ndim != 0:
            raise ValueError("contraction_progress must be scalar")

        value = self._modulate(self.state_norm(action), shift, scale)
        sq, sk, sv = (self._split_heads(part) for part in self.self_qkv(value).chunk(3, dim=-1))
        self_mask = torch.triu(
            torch.ones(
                int(action.shape[1]), int(action.shape[1]), device=action.device, dtype=torch.bool
            ),
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
            raw_depth_ratio_override=(
                None if raw_depth_ratio_overrides is None else raw_depth_ratio_overrides["self"]
            ),
            binary_group_selection=binary_group_selection,
            base_gate=legacy_gates[:, 0],
            update_keep=update_keeps[:, 0],
            contraction_identity_bypass=contraction_identity_bypass,
            prepared_factors=None if contraction_factors is None else contraction_factors["self"],
            collect_diagnostics=collect_diagnostics,
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
            raw_depth_ratio_override=(
                None if raw_depth_ratio_overrides is None else raw_depth_ratio_overrides["noisy"]
            ),
            binary_group_selection=binary_group_selection,
            base_gate=legacy_gates[:, 1],
            update_keep=update_keeps[:, 1],
            contraction_identity_bypass=contraction_identity_bypass,
            mask=noisy_mask,
            prepared_factors=None if contraction_factors is None else contraction_factors["noisy"],
            collect_diagnostics=collect_diagnostics,
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
            raw_depth_ratio_override=(
                None if raw_depth_ratio_overrides is None else raw_depth_ratio_overrides["stage"]
            ),
            binary_group_selection=binary_group_selection,
            base_gate=legacy_gates[:, 2],
            update_keep=update_keeps[:, 2],
            contraction_identity_bypass=contraction_identity_bypass,
            prepared_factors=None if contraction_factors is None else contraction_factors["stage"],
            collect_diagnostics=collect_diagnostics,
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
            raw_depth_ratio_override=(
                None if raw_depth_ratio_overrides is None else raw_depth_ratio_overrides["low"]
            ),
            binary_group_selection=binary_group_selection,
            base_gate=legacy_gates[:, 3],
            update_keep=update_keeps[:, 3],
            contraction_identity_bypass=contraction_identity_bypass,
            prepared_factors=None if contraction_factors is None else contraction_factors["low"],
            collect_diagnostics=collect_diagnostics,
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
            raw_depth_ratio_override=(
                None if raw_depth_ratio_overrides is None else raw_depth_ratio_overrides["ffn"]
            ),
            binary_group_selection=binary_group_selection,
            base_gate=legacy_gates[:, 4],
            update_keep=update_keeps[:, 4],
            contraction_identity_bypass=contraction_identity_bypass,
            prepared_factors=None if contraction_factors is None else contraction_factors["ffn"],
            collect_diagnostics=collect_diagnostics,
        )
        pre_norm_action = action + ffn_update
        normalized_action = self.out_norm(pre_norm_action)
        if spectral_token_mask is not None:
            expected_mask = (int(action.shape[0]), int(action.shape[1]))
            if tuple(spectral_token_mask.shape) != expected_mask:
                raise ValueError(
                    "spectral_token_mask must be [B,frequency], got "
                    f"{tuple(spectral_token_mask.shape)} vs {expected_mask}"
                )
            # This is a frequency ownership aperture, not an unconstrained
            # residual amplitude gate: it only controls which coefficient
            # directions a refinement level is allowed to change.
            action = before + spectral_token_mask[..., None].to(dtype=normalized_action.dtype) * (
                normalized_action - before
            )
        else:
            action = normalized_action
        if not collect_diagnostics:
            return action, {}

        pre_norm_rms = self._sample_rms(pre_norm_action).detach()
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
                "depth_ratio",
                "depth_ratio_min",
                "depth_ratio_max",
                "raw_depth_ratio",
                "effective_depth",
                "available_depth",
                "transparency_mean",
                "transparency_min",
                "transparency_max",
                "contraction_progress",
                "depth_usage_cost",
                "contraction_ratio",
                "subspace_energy_fraction",
                "removed_fraction",
                "removed_rms",
                "boundary_identity_error",
                "nonexpansive_violation",
                "nested_order_violation",
                "basis_norm_error",
                "basis_orthogonality_error",
                "basis_raw_norm",
                "operator_gain",
                "update_rms",
                "base_rms",
                "base_parameter_rms",
                "projection_input_rms",
                "base_data_gain",
                "base_activated_rms",
                "base_normalized_rms",
                "host_update_rms",
                "contracted_rms",
                "direction_change",
                "direction_cosine",
                "base_gate",
                "base_gate_abs_mean",
                "effective_gate",
                "effective_gate_abs_mean",
                "gate_scale_error",
                "keep_scale_error",
                "realized_scale",
                "update_keep",
            ):
                metrics[f"action_{name}_{metric_name}"] = operator_metrics[metric_name]
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
                "contracted_rms_rows",
                "base_gate_rows",
                "effective_gate_rows",
                "direction_change_rows",
                "update_keep_rows",
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
            role_rows = torch.stack(
                [
                    low_weight.detach().float()[..., role_ids == role_index].sum(dim=-1)
                    for role_index in range(role_count)
                ],
                dim=-1,
            )
            role_prob = role_rows.clamp_min(1e-8)
            role_entropy = -(role_prob * role_prob.log()).sum(dim=-1).mean()
            metrics["action_low_role_entropy"] = role_entropy
            metrics["action_low_role_effective_count"] = torch.exp(role_entropy)
            metrics["action_low_role_max"] = role_rows.max(dim=-1).values.mean()
            for role_index, role in enumerate(role_names):
                metrics[f"action_low_role_{role}_attention"] = role_rows[..., role_index].mean()
        return action, metrics


class SpectralPhysicalVelocityHead(nn.Module):
    """Emit coefficient-space velocities with the physical tangent contract."""

    def __init__(self, config: PolicyDecoderConfig) -> None:
        super().__init__()
        h = int(config.hidden_size)
        self.flow_codec = DCTFlowCodec(config)
        arm_channels = (
            int(config.arm_dim) if self.flow_codec.uses_arm_manifold else 2 * int(config.arm_dim)
        )
        grip_channels = (
            1 if self.flow_codec.uses_parseval_gripper_field else int(config.gripper_field_dim)
        )
        self.norm = nn.LayerNorm(h)
        self.arm = nn.Linear(h, arm_channels)
        self.gripper = nn.Linear(h, grip_channels)
        self.arm_field_channels = 2 * int(config.arm_dim)

    def output_layers(self) -> tuple[nn.Linear, ...]:
        return self.arm, self.gripper

    def forward(
        self,
        tokens: Tensor,
        *,
        coefficient_mask: Tensor | None = None,
    ) -> Tensor:
        normalized = self.norm(tokens)
        arm = self.arm(normalized)
        gripper = self.gripper(normalized)
        if coefficient_mask is not None:
            expected = (
                int(tokens.shape[0]),
                int(tokens.shape[1]),
                self.flow_codec.physical_dim,
            )
            if tuple(coefficient_mask.shape) != expected:
                raise ValueError(
                    "spectral coefficient mask must match the expanded flow "
                    f"shape {expected}, got {tuple(coefficient_mask.shape)}"
                )
            # The aperture acts in the independent arm/gripper coefficient
            # coordinates. Multiplying an already-expanded redundant field by
            # a frequency mask would generally leave its tangent subspace.
            arm = arm * coefficient_mask[..., :1].to(dtype=arm.dtype)
            gripper = gripper * coefficient_mask[
                ..., self.arm_field_channels : self.arm_field_channels + 1
            ].to(dtype=gripper.dtype)
        return self.flow_codec.expand_tangent_velocity(
            arm,
            gripper,
        )


class HierarchicalMMDiTActionDecoder(nn.Module):
    """Owned evidence plus condition-modulated stage-adaptive refinement."""

    def __init__(self, config: PolicyDecoderConfig) -> None:
        super().__init__()
        self.config = config
        h = int(config.hidden_size)
        self.hidden_size = h
        self.action_horizon = int(config.action_horizon)
        self.physical_action_dim = int(config.physical_action_dim)
        self.spectral_state = bool(int(getattr(config, "hierarchical_mmdit_spectral_state", 0)))
        self.spectral_codec = TemporalDCT(self.action_horizon) if self.spectral_state else None
        self.refine_block_count = int(config.hierarchical_mmdit_depth)
        self.operator_stage_count = int(config.hierarchical_mmdit_operator_stages)
        self.execution_contract = str(config.hierarchical_mmdit_execution_contract)
        self.block_owned_execution = self.execution_contract == "typed_block_budget"
        # Semantic stage slots remain a memory/retrieval repertoire.  V89's
        # execution repertoire contains only real, parameter-distinct MMDiT
        # blocks; V88 keeps the historical stage-as-operation interpretation.
        self.control_operation_count = (
            self.refine_block_count if self.block_owned_execution else self.operator_stage_count
        )
        self.refine_steps = int(config.hierarchical_mmdit_refine_steps)
        self.organizer = PolicyConditionOrganizer(config)
        self.intent_compiler = IntentContractCompiler(config)
        self.action_initializer = ConditionNeutralActionInitializer(
            config,
            frequency_positions=self.spectral_state,
        )
        self.time = TimeEmbedding(h)
        self.time_lift = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
        self.native_time_action_chart = not self.spectral_state and (
            str(config.arm_flow_mode) == "manifold_native"
            or str(config.gripper_field_mode) == "parseval_temporal"
        )
        if self.spectral_state:
            self.noisy_action_lift = FrequencyPhysicalActionTokenLift(config)
        elif self.native_time_action_chart:
            self.noisy_action_lift = NativeTimePhysicalActionTokenLift(config)
        else:
            self.noisy_action_lift = nn.Sequential(
                nn.LayerNorm(int(config.physical_action_dim)),
                nn.Linear(int(config.physical_action_dim), h),
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
        self.blocks = nn.ModuleList(
            [
                OwnedHierarchicalActionBlock(
                    config,
                    operator_stage_count=(
                        1
                        if self.block_owned_execution
                        else (
                            ((block_index + 1) * self.operator_stage_count)
                            // self.refine_block_count
                            - (block_index * self.operator_stage_count) // self.refine_block_count
                        )
                    ),
                )
                for block_index in range(self.refine_block_count)
            ]
        )
        self.refine_block_identity = nn.Parameter(torch.randn(1, self.refine_block_count, h) * 0.02)
        # New selector/sidecar capacity uses an isolated CPU RNG stream.  The
        # V77 host modules below therefore keep their original initialization
        # sequence when the sidecar is added or removed.
        host_rng_state = torch.get_rng_state()
        sidecar_generator = torch.Generator(device="cpu")
        sidecar_generator.manual_seed((int(torch.initial_seed()) ^ 0x5A17C0DE) % (2**63 - 1))
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
        self.velocity_head = (
            SpectralPhysicalVelocityHead(config)
            if self.spectral_state
            else ActionOnlyPhysicalVelocityHead(config)
        )
        self.spectral_aperture = (
            SoftSpectralAperture(
                self.action_horizon,
                arm_channels=2 * int(config.arm_dim),
                gripper_channels=(int(config.physical_action_dim) - 2 * int(config.arm_dim)),
                arm_start_fraction=float(config.hierarchical_mmdit_spectral_arm_start_fraction),
                gripper_start_fraction=float(
                    config.hierarchical_mmdit_spectral_gripper_start_fraction
                ),
                temperature=float(config.hierarchical_mmdit_spectral_temperature),
                schedule_power=float(config.hierarchical_mmdit_spectral_schedule_power),
                controller_shift_limit=float(
                    config.hierarchical_mmdit_spectral_controller_shift_limit
                ),
            )
            if self.spectral_state
            else None
        )
        self.response_codec = PhysicalActionCodec(config)
        self.event_head = nn.Sequential(
            nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 3)
        )
        self.motion_head = nn.Sequential(
            nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 1)
        )
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
            self.operator_contractions = nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
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
                        }
                    )
                    for block in self.blocks
                ]
            )
        finally:
            torch.set_rng_state(host_rng_state)
        self.unified_controller: UnifiedHierarchicalController | None = None
        if int(config.hierarchical_mmdit_unified_controller):
            host_rng_state = torch.get_rng_state()
            controller_generator = torch.Generator(device="cpu")
            controller_generator.manual_seed((int(torch.initial_seed()) ^ 0x43C07A01) % (2**63 - 1))
            try:
                torch.set_rng_state(controller_generator.get_state())
                self.unified_controller = UnifiedHierarchicalController(
                    config,
                    operator_branch_count=len(OwnedHierarchicalActionBlock._BRANCH_NAMES),
                )
            finally:
                torch.set_rng_state(host_rng_state)
            self.operator_stage_identity.requires_grad_(False)
            for legacy_module in (
                self.stage_selector_control,
                self.stage_selector_query,
                self.exit_controller,
                self.workspace.manager,
                self.operator_condition,
            ):
                legacy_module.requires_grad_(False)
            for bank in self.operator_contractions:
                for contraction in bank.values():
                    contraction.depth_weight.requires_grad_(False)
                    contraction.depth_bias.requires_grad_(False)
        self.register_buffer(
            "contraction_progress",
            torch.zeros((), dtype=torch.float32),
            persistent=True,
        )
        self._contraction_progress_value = 0.0
        self._operation_value_dwell_active = False

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
        self._contraction_progress_value = float(self.contraction_progress.detach().cpu())

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
        identities = (
            self.refine_block_identity[0]
            .to(device=device, dtype=dtype)
            .index_select(0, block_index)
        )
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
        if self.unified_controller is not None:
            return ()
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
        if self.unified_controller is not None:
            return ()
        return (
            self.operator_stage_identity,
            *tuple(self.stage_selector_control.parameters()),
            *tuple(self.stage_selector_query.parameters()),
        )

    def exit_controller_parameters(self) -> tuple[nn.Parameter, ...]:
        if self.unified_controller is not None:
            return ()
        return tuple(self.exit_controller.parameters())

    def unified_controller_parameters(self) -> tuple[nn.Parameter, ...]:
        if self.unified_controller is None:
            return ()
        interface = self.workspace.controller_interface
        if interface is None:
            raise RuntimeError("unified controller is missing its workspace interface")
        return (
            *tuple(self.unified_controller.parameters()),
            *tuple(interface.parameters()),
        )

    def unified_controller_parameter_groups(
        self,
    ) -> dict[str, tuple[nn.Parameter, ...]]:
        if self.unified_controller is None:
            return {}
        groups = self.unified_controller.parameter_groups()
        interface = self.workspace.controller_interface
        if interface is None:
            raise RuntimeError("unified controller is missing its workspace interface")
        return {
            **groups,
            "workspace_interface": tuple(interface.parameters()),
        }

    def scale_invariant_base_parameters(self) -> tuple[nn.Parameter, ...]:
        return ()

    def set_operator_contraction_training_step(self, global_step: int) -> float:
        warmup = int(self.config.hierarchical_mmdit_operator_contraction_warmup_steps)
        transition = int(self.config.hierarchical_mmdit_operator_contraction_transition_steps)
        progress = min(max((int(global_step) - warmup) / float(transition), 0.0), 1.0)
        self.contraction_progress.fill_(progress)
        self._contraction_progress_value = progress
        self._operation_value_dwell_active = int(global_step) >= int(
            self.config.hierarchical_mmdit_operation_value_warmup_steps
        )
        return progress

    def prepare_contraction_factors(self) -> tuple[dict[str, Tensor], ...]:
        return tuple(
            {name: contraction.prepare_factors() for name, contraction in bank.items()}
            for bank in self.operator_contractions
        )

    def _block_for_step(self, step_index: int) -> int:
        return min(
            (max(int(step_index), 0) * self.refine_block_count) // max(self.refine_steps, 1),
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

    def _control_operation_candidates(
        self,
        block_index: int,
        *,
        device: torch.device,
    ) -> Tensor:
        """Return execution candidates without conflating memory stages."""
        if self.block_owned_execution:
            return torch.tensor([int(block_index)], device=device, dtype=torch.long)
        return self._fixed_stage_candidates(block_index, device=device)

    def _control_operation_to_block(self, operation_index: Tensor) -> Tensor:
        if self.block_owned_execution:
            return operation_index.to(dtype=torch.long).clamp(0, self.refine_block_count - 1)
        return self._operator_stage_to_block(operation_index)

    def _canonical_stage_for_blocks(self, block_index: Tensor) -> Tensor:
        """Keep stage diagnostics in memory coordinates, never function coordinates."""
        starts = torch.div(
            block_index.to(dtype=torch.long) * self.operator_stage_count,
            self.refine_block_count,
            rounding_mode="floor",
        )
        return starts.clamp_max(self.operator_stage_count - 1)

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
                f"stage selector action must be [B,N,{self.hidden_size}], got {tuple(action.shape)}"
            )
        normalized = F.layer_norm(action.detach().float(), (self.hidden_size,))
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
                f"controller control_state must be {(batch, 4)}, got {tuple(control_state.shape)}"
            )
        action_state = self._stage_selector_action_summary(action)
        control_context = self.stage_selector_control(control_state.detach().float())
        return F.normalize(
            self.stage_selector_query(
                torch.cat(
                    [
                        global_intent.detach().float(),
                        time_state.detach().float(),
                        step_state.detach().float(),
                        action_state,
                        control_context.float(),
                    ],
                    dim=-1,
                )
            ).float(),
            dim=-1,
        )

    def _exit_logit(self, controller_query: Tensor) -> Tensor:
        """Apply the exit head without transferring gradient ownership."""
        if controller_query.ndim != 2 or int(controller_query.shape[-1]) != self.hidden_size:
            raise ValueError(
                f"exit controller query must be [B,H], got {tuple(controller_query.shape)}"
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
            uniform = torch.full_like(learned_probabilities, 1.0 / float(candidate_count))
            probabilities = (
                learned_probabilities * (1.0 - exploration_value) + uniform * exploration_value
            )
            if self._contraction_progress_value > 0.0:
                selected_local = torch.multinomial(probabilities.float(), num_samples=1).squeeze(1)
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

    def _select_fixed_unified_stage(
        self,
        *,
        block_index: Tensor,
        controller_state: Tensor,
        uniform_owner: int | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Reproduce the neutral V87 stage assignment without learned routing."""

        batch = int(block_index.shape[0])
        device = block_index.device
        selected = torch.empty(batch, device=device, dtype=torch.long)
        if uniform_owner is not None:
            stage = self._fixed_stage_candidates(uniform_owner, device=device)[0]
            selected.fill_(int(stage.item()))
        else:
            for owner in range(self.refine_block_count):
                rows = torch.nonzero(block_index == owner, as_tuple=False).flatten()
                if int(rows.numel()) == 0:
                    continue
                stage = self._fixed_stage_candidates(owner, device=device)[0]
                selected.index_fill_(0, rows, int(stage.item()))
        one = torch.ones(batch, device=device, dtype=torch.float32)
        zero = torch.zeros_like(one)
        diagnostic_query = F.normalize(controller_state.detach().float().mean(dim=1), dim=-1)
        return (
            selected,
            selected[:, None],
            one[:, None],
            zero,
            one,
            diagnostic_query,
            zero,
        )

    def _select_unified_operation(
        self,
        *,
        current_block: Tensor,
        operation_value_field: Tensor,
        controller_state: Tensor,
        step_index: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Choose a monotonic current/next executable function from residual value."""

        batch = int(operation_value_field.shape[0])
        expected = (batch, self.control_operation_count, self.action_horizon, 2)
        if tuple(operation_value_field.shape) != expected:
            raise ValueError(
                "unified operation value field has the wrong shape, expected "
                f"{expected}, got {tuple(operation_value_field.shape)}"
            )
        legal = torch.zeros(
            batch,
            self.control_operation_count,
            device=operation_value_field.device,
            dtype=torch.bool,
        )
        nominal_operation = torch.zeros(
            batch, device=operation_value_field.device, dtype=torch.long
        )
        fixed_owner = self._block_for_step(step_index)
        for owner in range(self.refine_block_count):
            rows = torch.nonzero(current_block == owner, as_tuple=False).flatten()
            if int(rows.numel()) == 0:
                continue
            current = self._control_operation_candidates(owner, device=operation_value_field.device)
            legal[rows[:, None], current[None]] = True
            nominal_owner = owner
            if owner + 1 < self.refine_block_count:
                following = self._control_operation_candidates(
                    owner + 1, device=operation_value_field.device
                )
                legal[rows[:, None], following[None]] = True
                if fixed_owner >= owner + 1:
                    nominal_owner = owner + 1
            nominal = self._control_operation_candidates(
                nominal_owner, device=operation_value_field.device
            )[0]
            nominal_operation.index_fill_(0, rows, int(nominal.item()))

        component_weight = torch.tensor(
            [float(self.config.arm_dim), 1.0],
            device=operation_value_field.device,
            dtype=torch.float32,
        ) / float(int(self.config.arm_dim) + 1)
        candidate_value = (
            (operation_value_field.float() * component_weight[None, None, None])
            .sum(dim=-1)
            .mean(dim=-1)
        )
        legal_value = candidate_value.masked_fill(~legal, torch.finfo(candidate_value.dtype).max)
        minimum, greedy = legal_value.min(dim=-1)
        nominal_value = legal_value.gather(1, nominal_operation[:, None]).squeeze(1)
        selected_operation = torch.where(nominal_value <= minimum, nominal_operation, greedy)
        selected_block = self._control_operation_to_block(selected_operation)
        operation_candidates = selected_operation[:, None]
        operation_probabilities = torch.ones(
            batch, 1, device=operation_value_field.device, dtype=torch.float32
        )
        legal_count = legal.float().sum(dim=-1).clamp_min(1.0)
        legal_mean = candidate_value.masked_fill(~legal, 0.0).sum(dim=-1) / legal_count
        legal_spread = torch.sqrt(
            ((candidate_value - legal_mean[:, None]).square() * legal.float()).sum(dim=-1)
            / legal_count
        )
        diagnostic_query = F.normalize(controller_state.detach().float().mean(dim=1), dim=-1)
        fixed_agreement = selected_operation == nominal_operation
        return (
            selected_block,
            selected_operation,
            operation_candidates,
            operation_probabilities,
            legal_spread,
            candidate_value.gather(1, selected_operation[:, None]).squeeze(1),
            diagnostic_query,
            fixed_agreement,
        )

    def _fixed_schedule(self, *, device: torch.device) -> tuple[Tensor, Tensor, int]:
        blocks = torch.tensor(
            [self._block_for_step(step) for step in range(self.refine_steps)],
            device=device,
            dtype=torch.long,
        )
        return (
            blocks,
            torch.ones(self.refine_steps, device=device, dtype=torch.bool),
            self.refine_steps,
        )

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
            schedule = torch.cat(
                [
                    schedule,
                    torch.full(
                        (self.refine_steps - active_length,),
                        active_block_count - 1,
                        dtype=torch.long,
                    ),
                ]
            )
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
                arm_null[..., :ad].float().square() + arm_null[..., ad : 2 * ad].float().square()
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
            arm_native_energy + arm_null_energy + gripper_native_energy + gripper_null_energy
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
            self._semantic_physical_rms(before.float()) + self._semantic_physical_rms(after.float())
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

    def _controller_operator_controls(
        self,
        output: UnifiedControllerOutput,
        *,
        operation_index: Tensor,
        operation_candidates: Tensor,
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        """Resolve the typed compute contract for one executable operation.

        The controller selects an operation and its nested capacity only.
        Host LayerScale remains the sole residual write-strength owner.  The
        old ``branch_update_keep_logits`` field is intentionally ignored so
        the compatibility contract cannot resurrect an amplitude shortcut.
        """
        batch, candidate_count = operation_candidates.shape
        expected_axis = "block" if self.block_owned_execution else "stage"
        if output.execution.operation_axis != expected_axis:
            raise RuntimeError(
                "controller/decoder execution axes disagree: expected "
                f"{expected_axis!r}, got {output.execution.operation_axis!r}"
            )
        expected = (
            batch,
            self.control_operation_count,
            len(OwnedHierarchicalActionBlock._BRANCH_NAMES),
        )
        if tuple(output.operator_update_logits.shape) != expected:
            raise ValueError("unified operator update logits have the wrong shape")
        if tuple(output.execution.branch_capacity_logits.shape) != expected:
            raise ValueError("unified operator depth logits have the wrong shape")
        gather_index = operation_candidates[:, :, None].expand(batch, candidate_count, expected[-1])
        raw_depth = torch.sigmoid(
            output.execution.branch_capacity_logits.gather(1, gather_index).float()
        )
        progress = float(self._contraction_progress_value)
        candidate_depth_keep = 1.0 - progress * (1.0 - raw_depth)
        selected_mask = (operation_candidates == operation_index[:, None]).float()
        selected_depth_keep = (candidate_depth_keep * selected_mask[:, :, None]).sum(dim=1)
        return raw_depth, None, selected_depth_keep

    def _local_contraction_coordinates(
        self,
        *,
        owner: int,
        stage_index: Tensor,
        stage_candidates: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Map memory-stage ids to the block-owned contraction chart."""
        if self.block_owned_execution:
            # There is one nested capacity path per real block/branch.  Stage
            # content remains available through cross-attention but cannot
            # silently select a second function basis inside the same block.
            return torch.zeros_like(stage_index), torch.zeros_like(stage_candidates)
        start, stop = self._stage_bounds(owner)
        local_stage = stage_index - start
        local_candidates = stage_candidates - start
        if local_candidates.device.type == "cpu" and (
            bool((local_candidates < 0).any()) or bool((local_candidates >= stop - start).any())
        ):
            raise ValueError("operator-stage candidates crossed a block ownership boundary")
        return local_stage, local_candidates

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
        stage_raw_depth_ratios: Tensor | None,
        stage_update_keeps: Tensor | None,
        contraction_factors: tuple[dict[str, Tensor], ...] | None,
        binary_group_selection: bool = False,
        spectral_token_mask: Tensor | None = None,
        uniform_owner: int | None = None,
        collect_diagnostics: bool = True,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Dispatch samples to distinct full-rank blocks without soft mixing."""
        batch = int(action.shape[0])
        if tuple(block_index.shape) != (batch,):
            raise ValueError("block_index must contain one owner per sample")
        if self.block_owned_execution and stage_update_keeps is not None:
            raise RuntimeError("typed_block_budget forbids a controller-owned update keep")
        if spectral_token_mask is not None and tuple(spectral_token_mask.shape) != (
            batch,
            int(action.shape[1]),
        ):
            raise ValueError("spectral_token_mask must be [B,frequency] for owned blocks")
        if uniform_owner is not None:
            owner = int(uniform_owner)
            if not 0 <= owner < self.refine_block_count:
                raise ValueError("uniform refinement owner is outside the block repertoire")
            local_stage, local_candidates = self._local_contraction_coordinates(
                owner=owner,
                stage_index=stage_index,
                stage_candidates=stage_candidates,
            )
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
                contraction_identity_bypass=(self._contraction_progress_value == 0.0),
                stage_candidates=local_candidates,
                stage_probabilities=stage_probabilities,
                raw_depth_ratio_overrides=(
                    None
                    if stage_raw_depth_ratios is None
                    else {
                        name: stage_raw_depth_ratios[..., branch_index]
                        for branch_index, name in enumerate(
                            OwnedHierarchicalActionBlock._BRANCH_NAMES
                        )
                    }
                ),
                branch_update_keeps=stage_update_keeps,
                binary_group_selection=binary_group_selection,
                contraction_factors=(
                    None if contraction_factors is None else contraction_factors[owner]
                ),
                spectral_token_mask=spectral_token_mask,
                low_role_ids=self.workspace.low_slot_role_ids,
                low_role_names=self.workspace.memory_bank.ROLE_NAMES,
                collect_diagnostics=collect_diagnostics,
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
            local_stage, local_candidates = self._local_contraction_coordinates(
                owner=owner,
                stage_index=stage_index.index_select(0, indices),
                stage_candidates=stage_candidates.index_select(0, indices),
            )
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
                contraction_identity_bypass=(self._contraction_progress_value == 0.0),
                stage_candidates=local_candidates,
                stage_probabilities=stage_probabilities.index_select(0, indices),
                raw_depth_ratio_overrides=(
                    None
                    if stage_raw_depth_ratios is None
                    else {
                        name: stage_raw_depth_ratios.index_select(0, indices)[..., branch_index]
                        for branch_index, name in enumerate(
                            OwnedHierarchicalActionBlock._BRANCH_NAMES
                        )
                    }
                ),
                branch_update_keeps=(
                    None
                    if stage_update_keeps is None
                    else stage_update_keeps.index_select(0, indices)
                ),
                binary_group_selection=binary_group_selection,
                contraction_factors=(
                    None if contraction_factors is None else contraction_factors[owner]
                ),
                spectral_token_mask=(
                    None
                    if spectral_token_mask is None
                    else spectral_token_mask.index_select(0, indices)
                ),
                low_role_ids=self.workspace.low_slot_role_ids,
                low_role_names=self.workspace.memory_bank.ROLE_NAMES,
                collect_diagnostics=collect_diagnostics,
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
        if not collect_diagnostics:
            return merged_action, {}
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
                merged_metrics[key] = torch.cat(values, dim=0).index_select(0, restore_order)
            elif all(tuple(value.shape) == tuple(values[0].shape) for value in values):
                merged_metrics[key] = sum(
                    value * (float(size) / float(batch))
                    for value, size in zip(values, group_sizes, strict=True)
                )
        return merged_action, merged_metrics

    def _spectral_aperture_for(
        self,
        progress: Tensor | float,
        *,
        controller_shift: Tensor | None = None,
    ) -> dict[str, Tensor] | None:
        if not self.spectral_state or self.spectral_aperture is None:
            return None
        return self.spectral_aperture(
            progress,
            controller_shift=controller_shift,
        )

    def _velocity_prediction(
        self,
        action: Tensor,
        *,
        spectral_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Read the physical flow while keeping coefficient diagnostics local."""
        if not self.spectral_state:
            output = self.velocity_head(self.action_norm(action))
            return output, None
        if self.spectral_codec is None:
            raise RuntimeError("spectral decoder is missing its DCT codec")
        if not isinstance(self.velocity_head, SpectralPhysicalVelocityHead):
            raise RuntimeError("spectral decoder is missing its coefficient head")
        output = self.velocity_head(
            self.action_norm(action),
            coefficient_mask=spectral_mask,
        )
        return self.spectral_codec.decode(output), output

    def _detached_velocity_prediction(
        self,
        action: Tensor,
        *,
        spectral_mask: Tensor | None = None,
    ) -> Tensor:
        """Read physical action diagnostics without poisoning AMP's weight cache."""
        # This trainable head is called again for the gradient-bearing output at
        # the end of forward().  Under an outer autocast context, making its
        # first call inside no_grad can cache detached parameter casts and erase
        # the later head gradients.  Detaching the input and result keeps this
        # probe outside the objective while preserving a grad-capable AMP cache.
        prediction, _ = self._velocity_prediction(
            action.detach(),
            spectral_mask=spectral_mask,
        )
        return prediction.detach()

    @torch.no_grad()
    def _probe_operation_candidates(
        self,
        *,
        action: Tensor,
        current_block: Tensor,
        active_rows: Tensor,
        noisy_tokens: Tensor,
        stage_tokens: Tensor,
        low_tokens: Tensor,
        shared_cond: Tensor,
        operator_cond: Tensor,
        unified_output: UnifiedControllerOutput,
        contraction_factors: tuple[dict[str, Tensor], ...] | None,
        spectral_token_mask: Tensor | None = None,
        baseline_spectral_coefficient_mask: Tensor | None = None,
        candidate_spectral_coefficient_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Evaluate legal current/next-function candidates without gradient flow."""
        batch = int(action.shape[0])
        candidate_count = self.control_operation_count + 1
        # Candidate values are supervised in the physical flow metric.  Keep
        # that audit boundary in FP32 even when an owned MMDiT block executes
        # under BF16 autocast; indexed assignment does not promote its source.
        candidate_predictions = torch.zeros(
            batch,
            candidate_count,
            self.action_horizon,
            self.physical_action_dim,
            device=action.device,
            dtype=torch.float32,
        )
        candidate_predictions[:, 0] = self._detached_velocity_prediction(
            action,
            spectral_mask=baseline_spectral_coefficient_mask,
        ).float()
        candidate_mask = torch.zeros(batch, candidate_count, device=action.device, dtype=torch.bool)
        candidate_mask[:, 0] = active_rows
        rows_parts: list[Tensor] = []
        operation_parts: list[Tensor] = []
        for owner in range(self.refine_block_count):
            rows = torch.nonzero((current_block == owner) & active_rows, as_tuple=False).flatten()
            if int(rows.numel()) == 0:
                continue
            owners = [owner]
            if owner + 1 < self.refine_block_count:
                owners.append(owner + 1)
            for candidate_owner in owners:
                operations = self._control_operation_candidates(
                    candidate_owner, device=action.device
                )
                for operation in operations.tolist():
                    rows_parts.append(rows)
                    operation_parts.append(torch.full_like(rows, int(operation)))
                    candidate_mask[rows, int(operation) + 1] = True
        if not rows_parts:
            return candidate_predictions, candidate_mask
        probe_rows = torch.cat(rows_parts, dim=0)
        probe_operations = torch.cat(operation_parts, dim=0)
        probe_blocks = self._control_operation_to_block(probe_operations)
        probe_stages = (
            self._canonical_stage_for_blocks(probe_blocks)
            if self.block_owned_execution
            else probe_operations
        )
        probe_action, _ = self._run_owned_blocks(
            action.index_select(0, probe_rows),
            block_index=probe_blocks,
            noisy_tokens=noisy_tokens.index_select(0, probe_rows),
            stage_tokens=stage_tokens.index_select(0, probe_rows),
            low_tokens=low_tokens.index_select(0, probe_rows),
            shared_cond=shared_cond.index_select(0, probe_rows),
            operator_cond=operator_cond.index_select(0, probe_rows),
            stage_index=probe_stages,
            stage_candidates=probe_stages[:, None],
            stage_probabilities=torch.ones(
                int(probe_rows.numel()),
                1,
                device=action.device,
                dtype=torch.float32,
            ),
            stage_raw_depth_ratios=torch.sigmoid(
                unified_output.execution.branch_capacity_logits[
                    probe_rows, probe_operations
                ].float()
            ),
            stage_update_keeps=None,
            binary_group_selection=self.unified_controller is not None,
            contraction_factors=contraction_factors,
            spectral_token_mask=(
                None
                if spectral_token_mask is None
                else spectral_token_mask.index_select(0, probe_rows)
            ),
            uniform_owner=None,
            collect_diagnostics=False,
        )
        probe_prediction = self._detached_velocity_prediction(
            probe_action,
            spectral_mask=(
                None
                if candidate_spectral_coefficient_mask is None
                else candidate_spectral_coefficient_mask.index_select(0, probe_rows)
            ),
        ).float()
        candidate_predictions[probe_rows, probe_operations + 1] = probe_prediction
        return candidate_predictions, candidate_mask

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
        collect_diagnostics: bool = True,
    ) -> dict[str, object]:
        batch = int(action.shape[0])
        device = action.device
        dtype = action.dtype
        if routing_mode not in {"fixed", "random_dwell", "adaptive", "learned"}:
            raise ValueError(f"unsupported refinement routing mode: {routing_mode!r}")
        dynamic_learned = routing_mode == "learned" and self.unified_controller is not None
        if routing_mode == "random_dwell":
            schedule, schedule_active, scheduled_steps = self._random_dwell_schedule(device=device)
        elif routing_mode == "fixed" or (routing_mode == "learned" and not dynamic_learned):
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
        previous_response_abs = torch.zeros(batch, device=device, dtype=torch.float32)
        previous_response_arm = torch.zeros(batch, device=device, dtype=torch.float32)
        previous_response_gripper = torch.zeros(batch, device=device, dtype=torch.float32)
        previous_response_arm_null = torch.zeros(batch, device=device, dtype=torch.float32)
        previous_response_gripper_null = torch.zeros(batch, device=device, dtype=torch.float32)
        previous_pressure_abs = torch.zeros(batch, device=device, dtype=torch.float32)
        previous_pressure_rel = torch.zeros(batch, device=device, dtype=torch.float32)
        has_previous_response = torch.zeros(batch, device=device, dtype=torch.float32)
        controller_state: Tensor | None = None
        branch_count = len(OwnedHierarchicalActionBlock._BRANCH_NAMES)
        previous_update_keeps = torch.ones(batch, branch_count, device=device, dtype=torch.float32)
        previous_depth_keeps = torch.ones_like(previous_update_keeps)
        previous_continue_keep = torch.ones(batch, device=device, dtype=torch.float32)
        if routing_mode == "adaptive":
            action_threshold = self._threshold_rows(
                time,
                tuple(
                    float(value)
                    for value in self.config.hierarchical_mmdit_action_response_thresholds
                ),
            )
            stage_threshold = self._threshold_rows(
                time,
                tuple(
                    float(value)
                    for value in self.config.hierarchical_mmdit_stage_pressure_thresholds
                ),
            )
        else:
            action_threshold = torch.zeros(batch, device=device, dtype=torch.float32)
            stage_threshold = torch.zeros_like(action_threshold)

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
        controller_rows: list[dict[str, Tensor]] = []
        operation_value_rows: list[Tensor] = []
        operation_update_rows: list[Tensor] = []
        operation_candidate_prediction_rows: list[Tensor] = []
        operation_candidate_mask_rows: list[Tensor] = []
        operation_decision_rows: list[Tensor] = []
        operation_fixed_agreement_rows: list[Tensor] = []
        operation_predicted_spread_rows: list[Tensor] = []
        operation_selected_value_rows: list[Tensor] = []
        spectral_aperture_rows: list[dict[str, Tensor]] = []
        spectral_competition_rows: list[Tensor] = []
        initial_aperture = self._spectral_aperture_for(
            torch.zeros(batch, device=device, dtype=torch.float32)
        )
        final_spectral_aperture = initial_aperture
        prediction = self._detached_velocity_prediction(
            action,
            spectral_mask=(
                None if initial_aperture is None else initial_aperture["coefficient_mask"]
            ),
        )
        prediction_rows: list[Tensor] = [prediction]
        contraction_factors = (
            None if self._contraction_progress_value == 0.0 else self.prepare_contraction_factors()
        )

        for step_index in range(self.refine_steps):
            if routing_mode not in {"adaptive", "learned"} and step_index >= scheduled_steps:
                break
            if routing_mode == "adaptive" or dynamic_learned:
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
            decision_block = block_index.detach().clone()

            remaining = float(scheduled_steps - step_index - 1) / float(max(self.refine_steps, 1))
            progress = (
                1.0 if self.refine_steps <= 1 else float(step_index) / float(self.refine_steps - 1)
            )
            step_state = self._step_state(
                block_index,
                progress_fraction=progress,
                remaining_fraction=remaining,
                dtype=dtype,
            )
            local_dwell = block_visit_count.gather(1, block_index[:, None]).squeeze(1) / float(
                max(self.refine_steps, 1)
            )
            selector_control_state = torch.stack(
                [
                    torch.log1p(previous_response_rel.clamp_min(0.0)),
                    torch.log1p(previous_pressure_rel.clamp_min(0.0)),
                    local_dwell,
                    has_previous_response,
                ],
                dim=-1,
            )
            workspace_control: WorkspaceControlOverride | None = None
            unified_output = None
            unified_operation_decision: Tensor | None = None
            stage_raw_depth_ratios: Tensor | None = None
            stage_update_keeps: Tensor | None = None
            selected_depth_keeps: Tensor | None = None
            control_operation_index: Tensor | None = None
            control_operation_candidates: Tensor | None = None
            if self.unified_controller is not None:
                response_feedback = torch.stack(
                    [
                        previous_response_abs,
                        previous_response_rel,
                        previous_response_arm,
                        previous_response_gripper,
                        previous_response_arm_null,
                        previous_response_gripper_null,
                        previous_pressure_abs,
                        previous_pressure_rel,
                        local_dwell,
                        has_previous_response,
                    ],
                    dim=-1,
                )
                controller_feedback = torch.cat(
                    [
                        response_feedback,
                        previous_update_keeps,
                        previous_depth_keeps,
                        previous_continue_keep[:, None],
                    ],
                    dim=-1,
                )
                stage_role = self.workspace.stage_role.to(device=device, dtype=dtype).expand(
                    batch, -1, -1
                )
                committed_spectral_aperture = None
                if final_spectral_aperture is not None:
                    committed_spectral_aperture = torch.stack(
                        [
                            final_spectral_aperture["arm_mask"],
                            final_spectral_aperture["gripper_mask"],
                        ],
                        dim=-1,
                    )
                unified_output = self.unified_controller(
                    previous_state=controller_state,
                    global_intent=contracts["global_intent"],
                    flow_time=time_state,
                    refine_time=step_state,
                    action_tokens=action,
                    evidence_tokens=prepared_evidence.tokens,
                    evidence_ranges=prepared_evidence.ranges,
                    evidence_role_ranges=prepared_evidence.role_ranges,
                    stage_role=stage_role,
                    stage_content=stage_content,
                    feedback=controller_feedback,
                    current_block=block_index,
                    committed_spectral_aperture=committed_spectral_aperture,
                    collect_diagnostics=collect_diagnostics,
                )
                controller_state = unified_output.state
                (
                    predicted_block,
                    predicted_operation,
                    predicted_operation_candidates,
                    predicted_operation_probabilities,
                    operation_predicted_spread,
                    operation_selected_value,
                    operation_selector_query,
                    operation_fixed_agreement,
                ) = self._select_unified_operation(
                    current_block=block_index,
                    operation_value_field=(unified_output.execution.operation_value_field),
                    controller_state=unified_output.state,
                    step_index=step_index,
                )
                unified_operation_decision = torch.where(
                    predicted_block > block_index,
                    torch.ones_like(predicted_block),
                    torch.zeros_like(predicted_block),
                )
                if dynamic_learned:
                    block_index = predicted_block
                    control_operation_index = predicted_operation
                    control_operation_candidates = predicted_operation_candidates
                    if self.block_owned_execution:
                        stage_index = self._canonical_stage_for_blocks(block_index)
                        stage_candidates = stage_index[:, None]
                    else:
                        stage_index = predicted_operation
                        stage_candidates = predicted_operation_candidates
                    stage_probabilities = predicted_operation_probabilities
                    selector_query = operation_selector_query
                    selector_exploration = torch.zeros(batch, device=device, dtype=torch.float32)
                    selector_entropy = torch.zeros_like(selector_exploration)
                    selector_max = torch.ones_like(selector_exploration)
                    step_active = active_rows
                else:
                    (
                        stage_index,
                        stage_candidates,
                        stage_probabilities,
                        selector_entropy,
                        selector_max,
                        selector_query,
                        selector_exploration,
                    ) = self._select_fixed_unified_stage(
                        block_index=block_index,
                        controller_state=unified_output.state,
                        uniform_owner=uniform_owner,
                    )
                    if self.block_owned_execution:
                        control_operation_index = block_index
                        control_operation_candidates = block_index[:, None]
                    else:
                        control_operation_index = stage_index
                        control_operation_candidates = stage_candidates
                step_state = self._step_state(
                    block_index,
                    progress_fraction=progress,
                    remaining_fraction=remaining,
                    dtype=dtype,
                )
                if control_operation_index is None or control_operation_candidates is None:
                    raise RuntimeError("unified execution contract did not resolve an operation")
                (
                stage_raw_depth_ratios,
                stage_update_keeps,
                selected_depth_keeps,
                ) = self._controller_operator_controls(
                    unified_output,
                    operation_index=control_operation_index,
                operation_candidates=control_operation_candidates,
            )
                workspace_control = WorkspaceControlOverride(
                    control_tokens=unified_output.memory.content,
                    control_addresses=unified_output.memory.address,
                    global_token=unified_output.memory.global_content,
                    private_tokens=unified_output.memory.private_content,
                    global_address=unified_output.memory.global_address,
                    private_addresses=unified_output.memory.private_address,
                )
                controller_rows.append(unified_output.metrics)
                spectral_competition_rows.append(unified_output.spectral_competition_loss)
                operation_value_rows.append(unified_output.execution.operation_value_field)
                operation_update_rows.append(unified_output.operator_update_logits)
            elif routing_mode == "adaptive":
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
            committed_spectral_coefficient_mask = (
                None
                if final_spectral_aperture is None
                else final_spectral_aperture["coefficient_mask"]
            )
            spectral_shift = None
            if self.spectral_state and unified_output is not None:
                spectral_shift = unified_output.execution.spectral_shift.to(dtype=dtype)
            aperture = self._spectral_aperture_for(
                torch.full(
                    (batch,),
                    float(progress),
                    device=device,
                    dtype=torch.float32,
                ),
                controller_shift=spectral_shift,
            )
            spectral_token_mask = None if aperture is None else aperture["token_mask"]
            spectral_coefficient_mask = None if aperture is None else aperture["coefficient_mask"]
            shared_condition = self.shared_condition(
                torch.cat(
                    [
                        contracts["global_intent"],
                        time_state,
                        step_state,
                    ],
                    dim=-1,
                )
            )
            if self.unified_controller is None:
                operator_condition = self.operator_condition(
                    torch.cat(
                        [
                            contracts["global_intent"],
                            time_state,
                            step_state,
                        ],
                        dim=-1,
                    )
                )
            else:
                # Unified raw-depth overrides make the legacy per-bank depth
                # conditioner semantically unreachable.  Keep only the shape
                # contract without paying for or training a second controller.
                operator_condition = shared_condition.detach()
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
                control_override=workspace_control,
                typed_stage_memory=self.unified_controller is not None,
                collect_diagnostics=collect_diagnostics,
            )
            stage_content = torch.where(step_active[:, None, None], candidate_stage, previous_stage)
            low = low + self.condition_type[:, 0:1].to(device=device, dtype=dtype)
            stage_for_action = stage_for_action + self.condition_type[:, 1:2].to(
                device=device, dtype=dtype
            )
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
                stage_raw_depth_ratios=stage_raw_depth_ratios,
                stage_update_keeps=stage_update_keeps,
                binary_group_selection=self.unified_controller is not None,
                contraction_factors=contraction_factors,
                spectral_token_mask=spectral_token_mask,
                uniform_owner=uniform_owner,
                collect_diagnostics=collect_diagnostics,
            )
            if (
                unified_output is not None
                and int(self.config.hierarchical_mmdit_operation_candidate_probes)
                and (self.training or collect_diagnostics)
            ):
                with deterministic_module_probe(self.blocks, self.operator_contractions):
                    candidate_predictions, candidate_mask = self._probe_operation_candidates(
                        action=action,
                        current_block=decision_block,
                        active_rows=active_rows,
                        noisy_tokens=noisy_typed,
                        stage_tokens=stage_for_action,
                        low_tokens=low,
                        shared_cond=shared_condition,
                        operator_cond=operator_condition,
                        unified_output=unified_output,
                        contraction_factors=contraction_factors,
                        spectral_token_mask=spectral_token_mask,
                        baseline_spectral_coefficient_mask=(committed_spectral_coefficient_mask),
                        candidate_spectral_coefficient_mask=(spectral_coefficient_mask),
                    )
                operation_candidate_prediction_rows.append(candidate_predictions)
                operation_candidate_mask_rows.append(candidate_mask)
            action = torch.where(step_active[:, None, None], candidate_action, action)
            if aperture is not None:
                if final_spectral_aperture is None:
                    raise RuntimeError("spectral aperture state disappeared during refinement")
                final_spectral_aperture = {
                    name: torch.where(
                        step_active.reshape(batch, *([1] * (value.ndim - 1))),
                        value,
                        final_spectral_aperture[name],
                    )
                    for name, value in aperture.items()
                }
            effective_spectral_mask = (
                None
                if final_spectral_aperture is None
                else final_spectral_aperture["coefficient_mask"]
            )
            if selected_depth_keeps is not None:
                with torch.no_grad():
                    if stage_update_keeps is not None:
                        previous_update_keeps = torch.where(
                            step_active[:, None],
                            stage_update_keeps.detach().float(),
                            previous_update_keeps,
                        )
                    previous_depth_keeps = torch.where(
                        step_active[:, None],
                        selected_depth_keeps.detach().float(),
                        previous_depth_keeps,
                    )
                    previous_continue_keep = step_active.detach().float()
            next_prediction = self._detached_velocity_prediction(
                action,
                spectral_mask=effective_spectral_mask,
            )
            if final_spectral_aperture is not None:
                spectral_aperture_rows.append(
                    {
                        key: value.detach().float()
                        for key, value in final_spectral_aperture.items()
                        if torch.is_tensor(value)
                    }
                )
            with torch.no_grad():
                response_abs, response_rel, response_components = self._physical_response(
                    prediction, next_prediction
                )
                pressure_abs, pressure_rel = self._stage_pressure(previous_stage, stage_content)
                response_abs = torch.where(
                    step_active, response_abs, torch.zeros_like(response_abs)
                )
                response_rel = torch.where(
                    step_active, response_rel, torch.zeros_like(response_rel)
                )
                response_components = {
                    key: torch.where(step_active, value, torch.zeros_like(value))
                    for key, value in response_components.items()
                }
                pressure_abs = torch.where(
                    step_active, pressure_abs, torch.zeros_like(pressure_abs)
                )
                pressure_rel = torch.where(
                    step_active, pressure_rel, torch.zeros_like(pressure_rel)
                )
                previous_response_rel = torch.where(
                    step_active, response_rel, previous_response_rel
                )
                previous_response_abs = torch.where(
                    step_active, response_abs, previous_response_abs
                )
                previous_response_arm = torch.where(
                    step_active, response_components["arm"], previous_response_arm
                )
                previous_response_gripper = torch.where(
                    step_active,
                    response_components["gripper"],
                    previous_response_gripper,
                )
                previous_response_arm_null = torch.where(
                    step_active,
                    response_components["arm_null"],
                    previous_response_arm_null,
                )
                previous_response_gripper_null = torch.where(
                    step_active,
                    response_components["gripper_null"],
                    previous_response_gripper_null,
                )
                previous_pressure_abs = torch.where(
                    step_active, pressure_abs, previous_pressure_abs
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

            post_local_dwell = block_visit_count.gather(1, block_index[:, None]).squeeze(1) / float(
                max(self.refine_steps, 1)
            )
            operation_decision = torch.full((batch,), -1, device=device, dtype=torch.long)
            if unified_operation_decision is not None:
                operation_decision = unified_operation_decision
            exit_control_state = torch.stack(
                [
                    torch.log1p(response_rel.detach().float().clamp_min(0.0)),
                    torch.log1p(pressure_rel.detach().float().clamp_min(0.0)),
                    post_local_dwell.detach().float(),
                    torch.ones_like(post_local_dwell, dtype=torch.float32),
                ],
                dim=-1,
            )
            if unified_output is None:
                exit_query = self._controller_query(
                    global_intent=contracts["global_intent"],
                    time_state=time_state,
                    step_state=step_state,
                    action=action,
                    control_state=exit_control_state,
                )
                # Oracle route supervision owns only the legacy exit head.
                exit_logit = self._exit_logit(exit_query)
                exit_probability = torch.sigmoid(exit_logit.detach().float())
            else:
                exit_logit = torch.zeros(batch, device=device, dtype=torch.float32)
                exit_probability = torch.zeros_like(exit_logit)
            if routing_mode == "adaptive":
                exit_candidate = step_active
            elif dynamic_learned:
                exit_candidate = torch.zeros(batch, device=device, dtype=torch.bool)
            else:
                next_step = step_index + 1
                block_boundary = (
                    next_step >= scheduled_steps
                    or not bool(schedule_active[next_step])
                    or int(schedule[next_step].item()) != int(schedule[step_index].item())
                )
                exit_candidate = step_active & block_boundary

            current_stage = stage_index
            prediction = next_prediction
            if collect_diagnostics:
                workspace_rows.append(workspace_metrics)
                mmdit_rows.append(mmdit_metrics)
                condition_norm_rows.append(
                    torch.stack(
                        [
                            low.detach().float().norm(dim=-1).mean(),
                            stage_for_action.detach().float().norm(dim=-1).mean(),
                            noisy_typed.detach().float().norm(dim=-1).mean(),
                        ]
                    ).mean()
                )
                step_state_rows.append(step_state.detach().float().norm(dim=-1).mean())
                response_abs_rows.append(response_abs)
                response_rel_rows.append(response_rel)
                response_arm_rows.append(response_components["arm"])
                response_gripper_rows.append(response_components["gripper"])
                response_arm_null_rows.append(response_components["arm_null"])
                response_gripper_null_rows.append(response_components["gripper_null"])
                pressure_abs_rows.append(pressure_abs)
                pressure_rel_rows.append(pressure_rel)
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
                if unified_output is not None:
                    operation_fixed_agreement_rows.append(operation_fixed_agreement.detach())
                    operation_predicted_spread_rows.append(
                        operation_predicted_spread.detach().float()
                    )
                    operation_selected_value_rows.append(operation_selected_value.detach().float())
                prediction_rows.append(next_prediction)

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
                stop = exhaustion_count >= int(
                    self.config.hierarchical_mmdit_exhaustion_confirm_steps
                )
                unresolved_rows = unresolved_rows | (
                    stop & pressure_live & (current_block == self.refine_block_count - 1)
                )
                active_rows = active_rows & ~stop
            elif routing_mode == "learned":
                if unified_output is None:
                    final_candidate = step_index + 1 >= scheduled_steps
                    learned_decision = exit_probability > 0.5
                    learned_stop = exit_candidate & (learned_decision | final_candidate)
                    active_rows = active_rows & ~learned_stop
                else:
                    current_block = torch.where(step_active, block_index, current_block)
            if collect_diagnostics:
                operation_decision_rows.append(operation_decision.detach())

        if not collect_diagnostics:
            return {
                "action": action,
                "stage_content": stage_content,
                "final_spectral_aperture": final_spectral_aperture,
            }

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
            selector_query_rows.append(
                torch.zeros(batch, self.hidden_size, device=device, dtype=torch.float32)
            )
            selector_exploration_rows.append(zero)
            exit_logit_rows.append(torch.zeros(batch, device=device, dtype=torch.float32))
            exit_probability_rows.append(zero)
            exit_candidate_rows.append(torch.zeros(batch, device=device, dtype=torch.bool))
            prediction_rows.append(prediction)
            operation_decision_rows.append(
                torch.full((batch,), -1, device=device, dtype=torch.long)
            )
        while (
            self.unified_controller is not None
            and (self.training or collect_diagnostics)
            and len(operation_value_rows) < self.refine_steps
        ):
            operation_value_rows.append(
                torch.zeros(
                    batch,
                    self.control_operation_count,
                    self.action_horizon,
                    2,
                    device=device,
                    dtype=dtype,
                )
            )
            operation_update_rows.append(
                torch.zeros(
                    batch,
                    self.control_operation_count,
                    branch_count,
                    device=device,
                    dtype=torch.float32,
                )
            )
            if int(self.config.hierarchical_mmdit_operation_candidate_probes):
                operation_candidate_prediction_rows.append(
                    torch.zeros(
                        batch,
                        self.control_operation_count + 1,
                        self.action_horizon,
                        self.physical_action_dim,
                        device=device,
                        dtype=torch.float32,
                    )
                )
                operation_candidate_mask_rows.append(
                    torch.zeros(
                        batch,
                        self.control_operation_count + 1,
                        device=device,
                        dtype=torch.bool,
                    )
                )
            operation_fixed_agreement_rows.append(
                torch.ones(batch, device=device, dtype=torch.bool)
            )
            operation_predicted_spread_rows.append(
                torch.zeros(batch, device=device, dtype=torch.float32)
            )
            operation_selected_value_rows.append(
                torch.zeros(batch, device=device, dtype=torch.float32)
            )

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
            "controller_rows": controller_rows,
            "operation_value_rows": (
                torch.stack(operation_value_rows, dim=1) if operation_value_rows else None
            ),
            "operation_update_rows": (
                torch.stack(operation_update_rows, dim=1) if operation_update_rows else None
            ),
            "operation_candidate_prediction_rows": (
                torch.stack(operation_candidate_prediction_rows, dim=1)
                if operation_candidate_prediction_rows
                else None
            ),
            "operation_candidate_mask_rows": (
                torch.stack(operation_candidate_mask_rows, dim=1)
                if operation_candidate_mask_rows
                else None
            ),
            "operation_decision_rows": torch.stack(operation_decision_rows, dim=1),
            "operation_fixed_agreement_rows": (
                torch.stack(operation_fixed_agreement_rows, dim=1)
                if operation_fixed_agreement_rows
                else None
            ),
            "operation_predicted_spread_rows": (
                torch.stack(operation_predicted_spread_rows, dim=1)
                if operation_predicted_spread_rows
                else None
            ),
            "operation_selected_value_rows": (
                torch.stack(operation_selected_value_rows, dim=1)
                if operation_selected_value_rows
                else None
            ),
            "spectral_competition_loss": (
                torch.stack(spectral_competition_rows).mean()
                if spectral_competition_rows
                else torch.zeros((), device=device, dtype=torch.float32)
            ),
            "spectral_aperture_rows": spectral_aperture_rows,
            "final_spectral_aperture": final_spectral_aperture,
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
        collect_diagnostics: bool = True,
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
        noisy_source = noisy_physical
        if self.spectral_state:
            if self.spectral_codec is None:
                raise RuntimeError("spectral decoder is missing its DCT codec")
            noisy_source = self.spectral_codec.encode(noisy_physical)
        noisy_native = self.noisy_action_lift(noisy_source)
        if self.native_time_action_chart:
            noisy_native = noisy_native + self.action_initializer.horizon_position.to(
                device=device, dtype=dtype
            )
        noisy, noisy_gate_mean = self._gate_noisy_tokens(noisy_native, time)
        time_state = self.time_lift(self.time(time.to(dtype=dtype)))
        workspace_condition = self.workspace_condition(
            torch.cat(
                [
                    contracts["global_intent"],
                    time_state,
                ],
                dim=-1,
            )
        )
        initial_action = action
        initial_stage = stage_content
        exhaustion_mode = str(self.config.hierarchical_mmdit_exhaustion_mode)
        dwell_mode = str(self.config.hierarchical_mmdit_dwell_mode)
        if self.unified_controller is not None:
            if dwell_mode == "learned" and (
                not self.training or self._operation_value_dwell_active
            ):
                routing_mode = "learned"
            else:
                # Both fixed and shadow execute the exact V87 path. Shadow
                # still emits value fields and detached candidate probes.
                routing_mode = "fixed"
        else:
            if not self.training and exhaustion_mode == "adaptive":
                routing_mode = "adaptive"
            elif not self.training and exhaustion_mode == "learned":
                routing_mode = "learned"
            elif (
                self.training
                and str(self.config.hierarchical_mmdit_schedule_mode) == "random_dwell"
            ):
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
            collect_diagnostics=collect_diagnostics,
        )
        action = refinement["action"]
        if not torch.is_tensor(action):
            raise TypeError("refinement returned an invalid action tensor")
        final_aperture = refinement["final_spectral_aperture"]
        if final_aperture is not None and not isinstance(final_aperture, dict):
            raise TypeError("refinement returned an invalid final spectral aperture")
        if collect_diagnostics:
            workspace_rows = refinement["workspace_rows"]
            mmdit_rows = refinement["mmdit_rows"]
            condition_norm_rows = refinement["condition_norm_rows"]
            step_state_rows = refinement["step_state_rows"]
            if not all(
                isinstance(rows, list)
                for rows in (
                    workspace_rows,
                    mmdit_rows,
                    condition_norm_rows,
                    step_state_rows,
                )
            ):
                raise TypeError("refinement returned invalid metric rows")

        shadow: dict[str, object] | None = None
        if (
            collect_diagnostics
            and self.unified_controller is None
            and not self.training
            and exhaustion_mode in {"shadow", "learned_shadow"}
        ):
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
                    routing_mode=("learned" if exhaustion_mode == "learned_shadow" else "adaptive"),
                )

        pred_velocity, pred_velocity_coefficients = self._velocity_prediction(
            action,
            spectral_mask=(None if final_aperture is None else final_aperture["coefficient_mask"]),
        )
        action = self.action_norm(action)
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
        if self.spectral_state:
            if self.spectral_codec is None:
                raise RuntimeError("spectral decoder is missing its DCT codec")
            # Event/motion heads retain their existing [B,time,...] contract.
            # The action state is frequency-indexed, so restore time semantics
            # before these auxiliary heads read token positions. This is an
            # orthonormal change of token chart, not a second action branch.
            subhead_tokens = self.spectral_codec.decode(subhead_tokens)
        event_logits = self.event_head(subhead_tokens)
        motion_logits = self.motion_head(subhead_tokens).squeeze(-1)
        if not collect_diagnostics:
            minimal = {
                "pred_velocity": pred_velocity,
                "event_logits": event_logits,
                "motion_logits": motion_logits,
            }
            if pred_velocity_coefficients is not None:
                minimal["pred_velocity_coefficients"] = pred_velocity_coefficients
            return minimal
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
        controller_rows = refinement["controller_rows"]
        operation_value_rows = refinement["operation_value_rows"]
        operation_update_rows = refinement["operation_update_rows"]
        operation_candidate_prediction_rows = refinement["operation_candidate_prediction_rows"]
        operation_candidate_mask_rows = refinement["operation_candidate_mask_rows"]
        operation_decision_rows = refinement["operation_decision_rows"]
        operation_fixed_agreement_rows = refinement["operation_fixed_agreement_rows"]
        operation_predicted_spread_rows = refinement["operation_predicted_spread_rows"]
        operation_selected_value_rows = refinement["operation_selected_value_rows"]
        spectral_competition_loss = refinement["spectral_competition_loss"]
        spectral_aperture_rows = refinement["spectral_aperture_rows"]
        probe_tensors = (
            response_abs_rows,
            response_rel_rows,
            response_arm_rows,
            response_gripper_rows,
            response_arm_null_rows,
            response_gripper_null_rows,
            pressure_abs_rows,
            pressure_rel_rows,
            stage_id_rows,
            block_id_rows,
            active_rows,
            executed_steps,
            unresolved_rows,
            budget_exhausted_rows,
            probe_predictions,
            selector_entropy_rows,
            selector_max_rows,
            selector_query_rows,
            selector_exploration_rows,
            exit_logit_rows,
            exit_probability_rows,
            exit_candidate_rows,
            operation_decision_rows,
        )
        if not all(torch.is_tensor(value) for value in probe_tensors):
            raise TypeError("refinement returned invalid probe tensors")
        active_float = active_rows.float()
        active_denominator = active_float.sum().clamp_min(1.0)
        operation_decision_valid = (operation_decision_rows >= 0) & active_rows.bool()
        operation_decision_denominator = operation_decision_valid.float().sum().clamp_min(1.0)
        if self.unified_controller is not None and not all(
            torch.is_tensor(value)
            for value in (
                operation_value_rows,
                operation_fixed_agreement_rows,
                operation_predicted_spread_rows,
                operation_selected_value_rows,
            )
        ):
            raise TypeError("refinement returned invalid operation value tensors")
        function_candidate_cosine = torch.zeros((), device=device, dtype=torch.float32)
        function_candidate_diversity = torch.zeros_like(function_candidate_cosine)
        function_candidate_update_rms = torch.zeros_like(function_candidate_cosine)
        function_candidate_update_spread = torch.zeros_like(function_candidate_cosine)
        function_candidate_pair_count = torch.zeros_like(function_candidate_cosine)
        function_candidate_valid_count = torch.zeros_like(function_candidate_cosine)
        if (
            self.block_owned_execution
            and torch.is_tensor(operation_candidate_prediction_rows)
            and torch.is_tensor(operation_candidate_mask_rows)
        ):
            with fp32_diagnostic(operation_candidate_prediction_rows) as candidate_prediction_fp32:
                candidate_update = (
                    candidate_prediction_fp32[..., 1:, :, :]
                    - candidate_prediction_fp32[..., :1, :, :]
                )
                valid_candidate = (
                    operation_candidate_mask_rows[..., 1:].bool() & active_rows.bool()[..., None]
                )
                update_flat = candidate_update.flatten(start_dim=-2)
                update_unit = F.normalize(update_flat, dim=-1)
                nonzero_candidate = update_flat.square().sum(dim=-1) > 1e-12
                pair_cosine = torch.einsum("bsch,bsdh->bscd", update_unit, update_unit)
                pair_candidate = valid_candidate & nonzero_candidate
                pair_valid = pair_candidate[..., :, None] & pair_candidate[..., None, :]
                upper = torch.triu(
                    torch.ones(
                        int(valid_candidate.shape[-1]),
                        int(valid_candidate.shape[-1]),
                        device=device,
                        dtype=torch.bool,
                    ),
                    diagonal=1,
                )
                pair_valid = pair_valid & upper
                pair_weight = pair_valid.float()
                function_candidate_pair_count = pair_weight.sum()
                function_candidate_cosine = (
                    pair_cosine * pair_weight
                ).sum() / function_candidate_pair_count.clamp_min(1.0)
                function_candidate_diversity = torch.where(
                    function_candidate_pair_count > 0.0,
                    1.0 - function_candidate_cosine,
                    torch.zeros_like(function_candidate_cosine),
                )
                update_rms = candidate_update.square().mean(dim=(-2, -1)).sqrt()
                valid_float = valid_candidate.float()
                function_candidate_valid_count = valid_float.sum(dim=-1).mean()
                function_candidate_update_rms = (
                    update_rms * valid_float
                ).sum() / valid_float.sum().clamp_min(1.0)
                row_count = valid_float.sum(dim=-1).clamp_min(1.0)
                row_mean = (update_rms * valid_float).sum(dim=-1) / row_count
                row_spread = torch.sqrt(
                    ((update_rms - row_mean[..., None]).square() * valid_float).sum(dim=-1)
                    / row_count
                )
                active_pair_rows = (valid_float.sum(dim=-1) > 1.0).float()
                function_candidate_update_spread = (
                    row_spread * active_pair_rows
                ).sum() / active_pair_rows.sum().clamp_min(1.0)
        exit_candidate_float = exit_candidate_rows.float() * active_float
        exit_candidate_denominator = exit_candidate_float.sum().clamp_min(1.0)
        final_step_index = active_float.sum(dim=1).long().clamp_min(1) - 1
        final_stage_rows = stage_id_rows.gather(1, final_step_index[:, None]).squeeze(1)
        final_block_rows = block_id_rows.gather(1, final_step_index[:, None]).squeeze(1)
        if self.refine_steps > 1:
            transition_active = active_float[:, 1:] * active_float[:, :-1]
            stage_advances = (
                stage_id_rows[:, 1:] > stage_id_rows[:, :-1]
            ).float() * transition_active
            stage_advance_rate = stage_advances.sum() / transition_active.sum().clamp_min(1.0)
            block_advances = (
                block_id_rows[:, 1:] > block_id_rows[:, :-1]
            ).float() * transition_active
            block_advance_rate = block_advances.sum() / transition_active.sum().clamp_min(1.0)
            selector_query_change_rows = (
                1.0
                - F.cosine_similarity(
                    selector_query_rows[:, 1:].float(),
                    selector_query_rows[:, :-1].float(),
                    dim=-1,
                )
            ).clamp_min(0.0)
            selector_query_change = (
                selector_query_change_rows * transition_active
            ).sum() / transition_active.sum().clamp_min(1.0)
            same_block_active = (
                transition_active * (block_id_rows[:, 1:] == block_id_rows[:, :-1]).float()
            )
            selector_same_block_query_change = (
                selector_query_change_rows * same_block_active
            ).sum() / same_block_active.sum().clamp_min(1.0)
        else:
            stage_advance_rate = torch.zeros((), device=device, dtype=torch.float32)
            block_advance_rate = torch.zeros((), device=device, dtype=torch.float32)
            selector_query_change = torch.zeros((), device=device, dtype=torch.float32)
            selector_same_block_query_change = torch.zeros((), device=device, dtype=torch.float32)

        def active_mean(value: Tensor) -> Tensor:
            return (value.detach().float() * active_float).sum() / active_denominator

        def active_quantile(value: Tensor, quantile: float) -> Tensor:
            selected = value.detach().float()[active_rows]
            if int(selected.numel()) == 0:
                return torch.zeros((), device=device, dtype=torch.float32)
            return torch.quantile(selected, float(quantile))

        time_bins = torch.clamp((time.detach().float().clamp(0.0, 1.0) * 3.0).long(), max=2)

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

        zero_chart_error = torch.zeros((), device=device, dtype=torch.float32)
        arm_tangent_null_ratio = zero_chart_error
        gripper_tangent_null_ratio = zero_chart_error
        noisy_gripper_chart_null_ratio = zero_chart_error
        with fp32_diagnostic(pred_velocity) as pred_velocity_fp32:
            if self.response_codec.uses_arm_manifold:
                arm_field = pred_velocity_fp32[..., : 2 * int(self.config.arm_dim)]
                _, _, arm_null = self.response_codec.project_arm_tangent(arm_field)
                arm_tangent_null_ratio = (
                    arm_null.square().sum() / arm_field.square().sum().clamp_min(1e-8)
                )
            if self.response_codec.uses_parseval_gripper_field:
                arm_span = 2 * int(self.config.arm_dim)
                grip_field = pred_velocity_fp32[..., arm_span:]
                grip_null = grip_field - self.response_codec.project_gripper_field(grip_field)
                gripper_tangent_null_ratio = (
                    grip_null.square().sum() / grip_field.square().sum().clamp_min(1e-8)
                )
        if self.response_codec.uses_parseval_gripper_field:
            with fp32_diagnostic(noisy_physical) as noisy_physical_fp32:
                arm_span = 2 * int(self.config.arm_dim)
                noisy_grip_field = noisy_physical_fp32[..., arm_span:]
                noisy_grip_null = noisy_grip_field - self.response_codec.project_gripper_field(
                    noisy_grip_field
                )
                noisy_gripper_chart_null_ratio = (
                    noisy_grip_null.square().sum() / noisy_grip_field.square().sum().clamp_min(1e-8)
                )

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
            "hierarchical_mmdit_native_time_chart_active": torch.tensor(
                float(self.native_time_action_chart), device=device
            ),
            "hierarchical_mmdit_spectral_state": torch.tensor(
                float(self.spectral_state), device=device
            ),
            "hierarchical_mmdit_spectral_competition_loss": spectral_competition_loss,
            "hierarchical_mmdit_spectral_final_progress": (
                torch.zeros((), device=device, dtype=torch.float32)
                if final_aperture is None
                else final_aperture["progress"].detach().float().mean()
            ),
            "hierarchical_mmdit_native_time_chart_complete": torch.tensor(
                float(
                    self.response_codec.uses_arm_manifold
                    and self.response_codec.uses_parseval_gripper_field
                ),
                device=device,
            ),
            "hierarchical_mmdit_native_time_position_alignment": torch.tensor(
                float(self.native_time_action_chart), device=device
            ),
            "hierarchical_mmdit_velocity_arm_tangent_null_ratio": arm_tangent_null_ratio,
            "hierarchical_mmdit_velocity_gripper_tangent_null_ratio": gripper_tangent_null_ratio,
            "hierarchical_mmdit_noisy_gripper_chart_null_ratio": noisy_gripper_chart_null_ratio,
            "hierarchical_mmdit_step_state_norm": torch.stack(step_state_rows).mean(),
            "hierarchical_mmdit_refine_steps": torch.tensor(
                float(self.refine_steps), device=device
            ),
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
            "hierarchical_mmdit_unified_controller": torch.tensor(
                float(self.unified_controller is not None), device=device
            ),
            "hierarchical_mmdit_control_token_count": torch.tensor(
                float(
                    0 if self.unified_controller is None else self.unified_controller.control_count
                ),
                device=device,
            ),
            "hierarchical_mmdit_unified_operator_depth_owner": torch.tensor(
                float(self.unified_controller is not None), device=device
            ),
            "hierarchical_mmdit_legacy_operator_depth_owner": torch.tensor(
                float(self.unified_controller is None), device=device
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
            "hierarchical_mmdit_stage_nested_contraction": torch.tensor(
                float(not self.block_owned_execution), device=device
            ),
            "hierarchical_mmdit_contraction_sidecar": torch.ones((), device=device),
            "hierarchical_mmdit_post_gate_sidecar": torch.ones((), device=device),
            "hierarchical_mmdit_shared_amplitude_owner": torch.ones((), device=device),
            "hierarchical_mmdit_duplicate_amplitude_owner": torch.zeros((), device=device),
            "hierarchical_mmdit_unified_update_amplitude_owner": torch.zeros((), device=device),
            "hierarchical_mmdit_host_update_amplitude_owner": torch.ones((), device=device),
            "hierarchical_mmdit_unified_relative_update_keep_owner": torch.zeros(
                (), device=device
            ),
            "hierarchical_mmdit_structured_control_width": torch.tensor(
                float(
                    self.control_operation_count * len(OwnedHierarchicalActionBlock._BRANCH_NAMES)
                ),
                device=device,
            ),
            "hierarchical_mmdit_operator_geometry_identifiable": torch.ones((), device=device),
            "hierarchical_mmdit_operator_boundary_identity": torch.ones((), device=device),
            "hierarchical_mmdit_operator_nested_path": torch.ones((), device=device),
            "hierarchical_mmdit_operator_continuous_depth": torch.ones((), device=device),
            "hierarchical_mmdit_operator_nonexpansive": torch.ones((), device=device),
            "hierarchical_mmdit_operator_post_contraction_renorm": torch.zeros((), device=device),
            "hierarchical_mmdit_operator_stage_local_selection": torch.tensor(
                float(not self.block_owned_execution), device=device
            ),
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
            "hierarchical_mmdit_stage_selector_entropy": active_mean(selector_entropy_rows),
            "hierarchical_mmdit_stage_selector_max": active_mean(selector_max_rows),
            "hierarchical_mmdit_stage_selector_exploration": active_mean(selector_exploration_rows),
            "hierarchical_mmdit_stage_selector_query_change": (
                selector_query_change.detach().float()
            ),
            "hierarchical_mmdit_stage_selector_same_block_query_change": (
                selector_same_block_query_change.detach().float()
            ),
            "hierarchical_mmdit_stage_selector_dynamic_state": torch.ones((), device=device),
            "hierarchical_mmdit_stage_selector_read_only_inputs": torch.ones((), device=device),
            "hierarchical_mmdit_executed_steps": executed_steps.detach().float().mean(),
            "hierarchical_mmdit_stage_advance_rate": (
                torch.zeros((), device=device, dtype=torch.float32)
                if self.block_owned_execution
                else stage_advance_rate
            ),
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
            "hierarchical_mmdit_final_stage": (
                torch.full((), -1.0, device=device)
                if self.block_owned_execution
                else final_stage_rows.detach().float().mean()
            ),
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
            "hierarchical_mmdit_value_dwell_shadow_active": torch.tensor(
                float(
                    self.unified_controller is not None
                    and str(self.config.hierarchical_mmdit_dwell_mode) == "shadow"
                ),
                device=device,
            ),
            "hierarchical_mmdit_value_dwell_warmup_active": torch.tensor(
                float(
                    self.unified_controller is not None
                    and self.training
                    and str(self.config.hierarchical_mmdit_dwell_mode) == "learned"
                    and not self._operation_value_dwell_active
                ),
                device=device,
            ),
            "hierarchical_mmdit_operation_decision_shadow_active": torch.tensor(
                float(self.unified_controller is not None and routing_mode != "learned"),
                device=device,
            ),
            "hierarchical_mmdit_operation_value_predicted_spread": (
                active_mean(operation_predicted_spread_rows)
                if torch.is_tensor(operation_predicted_spread_rows)
                else torch.zeros((), device=device, dtype=torch.float32)
            ),
            "hierarchical_mmdit_operation_selected_value": (
                active_mean(operation_selected_value_rows)
                if torch.is_tensor(operation_selected_value_rows)
                else torch.zeros((), device=device, dtype=torch.float32)
            ),
            "hierarchical_mmdit_operation_fixed_path_agreement": (
                (operation_fixed_agreement_rows.float() * active_float).sum() / active_denominator
                if torch.is_tensor(operation_fixed_agreement_rows)
                else torch.ones((), device=device, dtype=torch.float32)
            ),
            "hierarchical_mmdit_operation_monotonic_violation": (
                (
                    (block_id_rows[:, 1:] < block_id_rows[:, :-1]).float()
                    * active_float[:, 1:]
                    * active_float[:, :-1]
                ).sum()
                / (active_float[:, 1:] * active_float[:, :-1]).sum().clamp_min(1.0)
                if self.refine_steps > 1
                else torch.zeros((), device=device, dtype=torch.float32)
            ),
            "hierarchical_mmdit_operation_value_reader_firewall": torch.tensor(
                float(self.unified_controller is not None),
                device=device,
                dtype=torch.float32,
            ),
            **{
                f"hierarchical_mmdit_operation_{name}_rate": (
                    ((operation_decision_rows == index) & operation_decision_valid).float().sum()
                    / operation_decision_denominator
                )
                for index, name in enumerate(("stay", "advance"))
            },
            "hierarchical_mmdit_single_consumption_owner": torch.ones((), device=device),
            "hierarchical_mmdit_serial_composition": torch.ones((), device=device),
            "hierarchical_mmdit_competitive_market": torch.zeros((), device=device),
            "owned_workspace_state_pre_dit": torch.ones((), device=device),
            "owned_workspace_trajectory_is_proposal": torch.ones((), device=device),
            "intent_noisy_input_present": torch.zeros((), device=device),
            "owned_workspace_fixed_role_prior": torch.tensor(
                float(self.unified_controller is None), device=device
            ),
            "owned_workspace_unified_role_selector": torch.tensor(
                float(self.unified_controller is not None), device=device
            ),
            "owned_workspace_controller_token_interface": torch.tensor(
                float(self.workspace.controller_interface is not None), device=device
            ),
            "owned_workspace_controller_value_firewall": torch.tensor(
                float(self.workspace.controller_interface is not None), device=device
            ),
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
            **initializer_metrics,
        }
        if self.block_owned_execution:
            result.update(
                {
                    "hierarchical_mmdit_control_operation_count": torch.tensor(
                        float(self.control_operation_count), device=device
                    ),
                    "hierarchical_mmdit_block_owned_execution": torch.ones((), device=device),
                    "hierarchical_mmdit_memory_stage_execution_decoupled": torch.ones(
                        (), device=device
                    ),
                    "hierarchical_mmdit_block_nested_contraction": torch.ones((), device=device),
                    "hierarchical_mmdit_function_candidate_cosine": (function_candidate_cosine),
                    "hierarchical_mmdit_function_candidate_diversity": (
                        function_candidate_diversity
                    ),
                    "hierarchical_mmdit_function_candidate_update_rms": (
                        function_candidate_update_rms
                    ),
                    "hierarchical_mmdit_function_candidate_update_spread": (
                        function_candidate_update_spread
                    ),
                    "hierarchical_mmdit_function_candidate_pair_count": (
                        function_candidate_pair_count
                    ),
                    "hierarchical_mmdit_function_candidate_valid_count": (
                        function_candidate_valid_count
                    ),
                }
            )
        if self.unified_controller is None:
            result.update(
                {
                    "hierarchical_mmdit_exit_logits": exit_logit_rows,
                    "hierarchical_mmdit_exit_probability": (
                        exit_probability_rows.detach().float() * exit_candidate_float
                    ).sum()
                    / exit_candidate_denominator,
                    "hierarchical_mmdit_exit_candidate_rate": exit_candidate_float.mean(),
                    "hierarchical_mmdit_post_block_exit_controller": torch.ones((), device=device),
                    "hierarchical_mmdit_exit_controller_read_only_inputs": torch.ones(
                        (), device=device
                    ),
                    "hierarchical_mmdit_early_exit_rate": (
                        executed_steps.detach().float() < float(self.refine_steps)
                    )
                    .float()
                    .mean(),
                    "refinement_probe_exit_candidates": exit_candidate_rows.detach(),
                }
            )
        if pred_velocity_coefficients is not None:
            result["pred_velocity_coefficients"] = pred_velocity_coefficients
        if spectral_aperture_rows:
            for name in (
                "arm_mask",
                "gripper_mask",
                "token_mask",
                "arm_cutoff",
                "gripper_cutoff",
                "controller_shift_rms",
                "controller_global_shift_rms",
                "frequency_warp_rms",
                "frequency_spacing_min",
                "frequency_spacing_max",
                "progress",
            ):
                values = [row[name].mean() for row in spectral_aperture_rows if name in row]
                if values:
                    result[f"hierarchical_mmdit_spectral_{name}"] = torch.stack(values).mean()
        if final_aperture is not None:
            for name in (
                "arm_mask",
                "gripper_mask",
                "token_mask",
                "arm_cutoff",
                "gripper_cutoff",
                "controller_shift_rms",
                "controller_global_shift_rms",
                "frequency_warp_rms",
                "frequency_spacing_min",
                "frequency_spacing_max",
            ):
                result[f"hierarchical_mmdit_spectral_final_{name}"] = (
                    final_aperture[name].detach().float().mean()
                )
        if (
            torch.is_tensor(operation_value_rows)
            and torch.is_tensor(operation_update_rows)
            and torch.is_tensor(operation_candidate_prediction_rows)
            and torch.is_tensor(operation_candidate_mask_rows)
        ):
            result["hierarchical_mmdit_operation_value_field"] = operation_value_rows
            result["hierarchical_mmdit_operation_update_logits"] = operation_update_rows
            result["hierarchical_mmdit_operation_candidate_predictions"] = (
                operation_candidate_prediction_rows.detach()
            )
            result["hierarchical_mmdit_operation_candidate_mask"] = (
                operation_candidate_mask_rows.detach()
            )
        for time_bin in range(3):
            for label, quantile in (("p25", 0.25), ("p50", 0.50), ("p75", 0.75)):
                result[f"hierarchical_mmdit_action_response_t{time_bin}_{label}"] = (
                    active_time_quantile(response_rel_rows, time_bin, quantile)
                )
                result[f"hierarchical_mmdit_stage_pressure_t{time_bin}_{label}"] = (
                    active_time_quantile(pressure_rel_rows, time_bin, quantile)
                )
        if not self.block_owned_execution:
            for stage_index in range(self.operator_stage_count):
                result[f"hierarchical_mmdit_stage_{stage_index}_usage"] = (
                    (stage_id_rows == stage_index).float() * active_float
                ).sum() / active_denominator
        for block_index in range(self.refine_block_count):
            result[f"hierarchical_mmdit_block_{block_index}_usage"] = (
                (block_id_rows == block_index).float() * active_float
            ).sum() / active_denominator
            result[f"hierarchical_mmdit_block_{block_index}_dwell_steps"] = (
                ((block_id_rows == block_index).float() * active_float).sum(dim=1).mean()
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
            step_operation_valid = operation_decision_valid[:, step_index]
            step_operation_denominator = step_operation_valid.float().sum().clamp_min(1.0)
            for decision_index, decision_name in enumerate(("stay", "advance")):
                result[f"hierarchical_mmdit_step_{step_index}_operation_{decision_name}_rate"] = (
                    (operation_decision_rows[:, step_index] == decision_index)
                    & step_operation_valid
                ).float().sum() / step_operation_denominator
        if shadow is not None:
            shadow_steps = shadow["executed_steps"]
            shadow_unresolved = shadow["unresolved_rows"]
            shadow_budget_exhausted = shadow["budget_exhausted_rows"]
            shadow_stage_ids = shadow["stage_id_rows"]
            shadow_block_ids = shadow["block_id_rows"]
            shadow_active = shadow["active_rows"]
            shadow_operation_decisions = shadow["operation_decision_rows"]
            if all(
                torch.is_tensor(value)
                for value in (
                    shadow_steps,
                    shadow_unresolved,
                    shadow_budget_exhausted,
                    shadow_stage_ids,
                    shadow_block_ids,
                    shadow_active,
                    shadow_operation_decisions,
                )
            ):
                result["hierarchical_mmdit_shadow_executed_steps"] = shadow_steps.float().mean()
                result["hierarchical_mmdit_shadow_unresolved_rate"] = (
                    shadow_unresolved.float().mean()
                )
                result["hierarchical_mmdit_shadow_budget_exhausted_rate"] = (
                    shadow_budget_exhausted.float().mean()
                )
                result["hierarchical_mmdit_shadow_early_exit_rate"] = (
                    (shadow_steps.float() < float(self.refine_steps)).float().mean()
                )
                shadow_count = shadow_active.float().sum(dim=1).long().clamp_min(1) - 1
                final_shadow_stage = shadow_stage_ids.gather(1, shadow_count[:, None]).squeeze(1)
                result["hierarchical_mmdit_shadow_final_stage"] = final_shadow_stage.float().mean()
                final_shadow_block = shadow_block_ids.gather(1, shadow_count[:, None]).squeeze(1)
                result["hierarchical_mmdit_shadow_final_block"] = final_shadow_block.float().mean()
                shadow_decision_valid = (shadow_operation_decisions >= 0) & shadow_active.bool()
                shadow_decision_denominator = shadow_decision_valid.float().sum().clamp_min(1.0)
                for index, name in enumerate(("stay", "advance", "exit")):
                    result[f"hierarchical_mmdit_shadow_operation_{name}_rate"] = (
                        (shadow_operation_decisions == index) & shadow_decision_valid
                    ).float().sum() / shadow_decision_denominator
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
                    result["refinement_shadow_probe_pred_velocity"] = shadow_predictions.detach()
                    result["refinement_shadow_probe_active"] = shadow_active.detach()
        organizer_metrics = organized["metrics"]
        if isinstance(organizer_metrics, dict):
            result.update(
                {key: value for key, value in organizer_metrics.items() if torch.is_tensor(value)}
            )
        result.update(
            {
                key: value
                for key, value in contracts.items()
                if key.startswith("intent_") and torch.is_tensor(value)
            }
        )
        workspace_mean = self._mean_metrics(workspace_rows)
        for key, value in workspace_mean.items():
            result[f"owned_{key}"] = value
        mmdit_mean = self._mean_metrics(mmdit_rows)
        for key, value in mmdit_mean.items():
            if not key.endswith("_rows"):
                result[f"hierarchical_mmdit_{key}"] = value
        if isinstance(controller_rows, list) and controller_rows:
            controller_mean = self._mean_metrics(controller_rows)
            for key, value in controller_mean.items():
                result[f"hierarchical_mmdit_{key}"] = value
        depth_costs = [
            row[f"action_{branch}_depth_usage_cost"]
            for row in mmdit_rows
            for branch in OwnedHierarchicalActionBlock._BRANCH_NAMES
            if f"action_{branch}_depth_usage_cost" in row
        ]
        if depth_costs:
            result["hierarchical_mmdit_depth_usage_regularizer"] = torch.stack(depth_costs).mean()
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
                    "contracted_rms_rows",
                    "base_gate_rows",
                    "effective_gate_rows",
                    "update_keep_rows",
                    "direction_change_rows",
                ):
                    key = f"action_{branch}_{metric_name}"
                    if all(key in row for row in mmdit_rows):
                        metric_values[metric_name] = (
                            torch.stack([row[key] for row in mmdit_rows], dim=1).detach().float()
                        )
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
                        result[f"hierarchical_mmdit_block_{block_index}_{branch}_{output_name}"] = (
                            values * block_mask
                        ).sum() / block_denominator
        if all(
            key in mmdit_mean
            for key in (
                "action_noisy_update_fraction_rows",
                "action_workspace_update_fraction_rows",
                "action_low_update_fraction_rows",
                "action_stage_update_fraction_rows",
            )
        ):
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
