from __future__ import annotations

"""Legacy CVAE action decoders retained for migration and ablations."""

import math
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..codec import PhysicalActionTokenLift, TransitionAwarePhysicalVelocityHead
from ..contracts import scaled_contract_view as _scaled_contract_view
from ..evidence import HierarchicalEvidenceWorkspace, PreparedEvidenceMemory
from ..primitives import BiasFreeFFN, TimeEmbedding
from .cvae_workspace import MMDiTConditionLayout, SemanticEvidenceWorkspace
from .residual import LayeredV37StyleResidualActionFlowDenoiser


LegacyPolicyConfig = Any


class LatentCVAEActionBlock(nn.Module):
    """Small FiLM-conditioned token block for the V42 CVAE action head.

    This intentionally avoids the V41/V41.1 heavy memory cross-attention stack.
    All V40 latents are first fused into one condition vector; every decoder
    block receives that condition through AdaLN/FiLM, keeping the final action
    path compact and stable.
    """

    def __init__(self, config: LegacyPolicyConfig) -> None:
        super().__init__()
        self.config = config
        h = int(config.hidden_size)
        heads = int(config.num_heads)
        self.n1 = nn.LayerNorm(h, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(h, heads, batch_first=True, dropout=float(config.dropout))
        self.n2 = nn.LayerNorm(h, elementwise_affine=False)
        self.ffn = BiasFreeFFN(h, float(getattr(config, "latent_cvae_ffn_expansion", 2.0)))
        self.drop = nn.Dropout(float(config.dropout))
        self.cond_mod_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.mod = nn.Linear(h, 6 * h)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    @staticmethod
    def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        return x * (1.0 + scale[:, None]) + shift[:, None]

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        sa_s, sa_c, sa_g, ff_s, ff_c, ff_g = self.mod(self.cond_mod_norm(cond)).chunk(6, dim=-1)
        value = self.n1(x)
        qk = self._modulate(value, sa_s, sa_c)
        attn_mask = None
        if int(getattr(self.config, "latent_cvae_causal_attention", 1)):
            n = int(x.shape[1])
            attn_mask = torch.triu(torch.ones(n, n, device=x.device, dtype=torch.bool), diagonal=1)
        update, _ = self.self_attn(qk, qk, value, attn_mask=attn_mask, need_weights=False)
        x = x + torch.tanh(sa_g)[:, None] * self.drop(update)
        update = self.ffn(self._modulate(self.n2(x), ff_s, ff_c))
        return x + torch.tanh(ff_g)[:, None] * self.drop(update)


class LatentCVAEMMDiTBlock(nn.Module):
    """Compact MMDiT-style mixer for CVAE action tokens.

    Action tokens and condition tokens use separate QKV/O and MLP parameters,
    then action queries attend over the concatenated action+condition keys.
    The condition stream is read-only by default to avoid action information
    being written into condition tokens and returning as a shortcut.
    """

    def __init__(
        self,
        config: LegacyPolicyConfig,
        *,
        ffn_expansion: float | None = None,
        causal_attention: bool | None = None,
        noisy_causal: bool | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        h = int(config.hidden_size)
        heads = int(config.num_heads)
        if h % heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads for LatentCVAEMMDiTBlock")
        self.hidden_size = h
        self.heads = heads
        self.head_dim = h // heads
        self.causal_attention = (
            bool(int(getattr(config, "latent_cvae_causal_attention", 1)))
            if causal_attention is None else bool(causal_attention)
        )
        self.noisy_causal = (
            bool(int(getattr(config, "latent_cvae_mmdit_noisy_causal", 1)))
            if noisy_causal is None else bool(noisy_causal)
        )
        self.action_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.cond_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.action_qkv = nn.Linear(h, 3 * h)
        self.cond_qkv = nn.Linear(h, 3 * h)
        self.action_out = nn.Linear(h, h)
        self.cond_out = nn.Linear(h, h)
        self.action_ffn_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.cond_ffn_norm = nn.LayerNorm(h, elementwise_affine=False)
        expansion = (
            float(getattr(config, "latent_cvae_ffn_expansion", 2.0))
            if ffn_expansion is None else float(ffn_expansion)
        )
        self.action_ffn = BiasFreeFFN(h, expansion)
        self.cond_ffn = BiasFreeFFN(h, expansion)
        self.global_cond_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.action_mod = nn.Linear(h, 6 * h)
        self.cond_mod = nn.Linear(h, 6 * h)
        self.drop = nn.Dropout(float(config.dropout))
        nn.init.zeros_(self.action_mod.weight)
        nn.init.zeros_(self.action_mod.bias)
        nn.init.zeros_(self.cond_mod.weight)
        nn.init.zeros_(self.cond_mod.bias)

    @staticmethod
    def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        return x * (1.0 + scale[:, None]) + shift[:, None]

    def _split_heads(self, x: Tensor) -> Tensor:
        b, n, h = x.shape
        return x.reshape(b, n, self.heads, h // self.heads).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        b, heads, n, d = x.shape
        return x.transpose(1, 2).reshape(b, n, heads * d)

    @staticmethod
    def _attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Tensor | None = None,
        key_bias: Tensor | None = None,
        batch_key_bias: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * (float(q.shape[-1]) ** -0.5)
        if key_bias is not None:
            scores = scores + key_bias.to(device=scores.device, dtype=scores.dtype)[None, None, None]
        if batch_key_bias is not None:
            # V70: per-sample additive logit bias (e.g. the t-gate on noisy
            # keys), shape [B, K] broadcast over heads and queries.
            scores = scores + batch_key_bias.to(device=scores.device, dtype=scores.dtype)[:, None, None, :]
        if mask is not None:
            scores = scores.masked_fill(mask[None, None], torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1).to(dtype=q.dtype)
        return torch.matmul(weights, v), weights

    def _action_mask(self, action_len: int, cond_len: int, noisy_start: int, noisy_len: int, device: torch.device) -> Tensor | None:
        if not self.causal_attention:
            return None
        total = action_len + cond_len
        mask = torch.zeros(action_len, total, device=device, dtype=torch.bool)
        future_action = torch.triu(torch.ones(action_len, action_len, device=device, dtype=torch.bool), diagonal=1)
        mask[:, :action_len] = future_action
        if self.noisy_causal and noisy_len > 0:
            horizon = torch.arange(action_len, device=device)[:, None]
            noisy_pos = torch.arange(noisy_len, device=device)[None]
            future_noisy = noisy_pos > horizon
            start = action_len + int(noisy_start)
            stop = min(start + noisy_len, total)
            if start < stop:
                mask[:, start:stop] = future_noisy[:, : stop - start]
        return mask

    @staticmethod
    def _action_key_bias(
        *,
        action_len: int,
        cond_len: int,
        rollout_start: int,
        rollout_len: int,
        device: torch.device,
    ) -> Tensor | None:
        if rollout_len <= 0:
            return None
        total = int(action_len) + int(cond_len)
        start = int(action_len) + int(rollout_start)
        stop = min(start + int(rollout_len), total)
        if start >= stop:
            return None
        # Preserve every spatial rollout token without granting the group extra
        # prior mass merely because it has more tokens. Under equal logits the
        # complete rollout grid starts with roughly one horizon group's budget.
        reference = max(int(action_len), 1)
        group_ratio = max(float(stop - start) / float(reference), 1e-6)
        bias = torch.zeros(total, device=device, dtype=torch.float32)
        bias[start:stop] = -math.log(group_ratio)
        return bias

    @staticmethod
    def _hierarchical_key_bias(
        *,
        action_len: int,
        cond_len: int,
        low_start: int,
        low_len: int,
        stage_start: int,
        stage_len: int,
        noisy_start: int,
        noisy_len: int,
        device: torch.device,
    ) -> Tensor:
        """Give each condition group one action-horizon unit of prior mass."""

        total = int(action_len) + int(cond_len)
        reference = max(int(action_len), 1)
        bias = torch.zeros(total, device=device, dtype=torch.float32)
        for start, length in (
            (low_start, low_len),
            (stage_start, stage_len),
            (noisy_start, noisy_len),
        ):
            absolute_start = int(action_len) + int(start)
            absolute_stop = min(absolute_start + int(length), total)
            if int(length) <= 0 or absolute_start >= absolute_stop:
                continue
            group_ratio = max(float(absolute_stop - absolute_start) / float(reference), 1e-6)
            bias[absolute_start:absolute_stop] = -math.log(group_ratio)
        return bias

    def forward(
        self,
        action: Tensor,
        cond_tokens: Tensor,
        global_cond: Tensor,
        *,
        noisy_start: int,
        noisy_len: int,
        rollout_start: int,
        rollout_len: int,
        low_start: int,
        low_len: int,
        stage_start: int,
        stage_len: int,
        update_condition: bool,
        noisy_logit_bias: Tensor | None = None,
        low_logit_bias: Tensor | None = None,
        stage_logit_bias: Tensor | None = None,
        noisy_value_gate: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        action_before = action
        cond_before = cond_tokens
        stable_global = self.global_cond_norm(global_cond)
        a_sa_s, a_sa_c, a_sa_g, a_ff_s, a_ff_c, a_ff_g = self.action_mod(stable_global).chunk(6, dim=-1)
        c_sa_s, c_sa_c, c_sa_g, c_ff_s, c_ff_c, c_ff_g = self.cond_mod(stable_global).chunk(6, dim=-1)

        a_value = self.action_norm(action)
        c_value = self.cond_norm(cond_tokens)
        a_qkv = self.action_qkv(self._modulate(a_value, a_sa_s, a_sa_c)).chunk(3, dim=-1)
        c_qkv = self.cond_qkv(self._modulate(c_value, c_sa_s, c_sa_c)).chunk(3, dim=-1)
        aq, ak, av = (self._split_heads(part) for part in a_qkv)
        cq, ck, cv = (self._split_heads(part) for part in c_qkv)
        k_all = torch.cat([ak, ck], dim=2)
        v_all = torch.cat([av, cv], dim=2)
        if noisy_value_gate is not None and int(noisy_len) > 0:
            # Real value-domain t-gate (noisy_gate_mode=1).  Applied AFTER
            # cond_norm/QKV so the scale-invariant LayerNorm cannot cancel it:
            # bids (keys) keep full strength at every t, only the information
            # amplitude delivered on attention is attenuated at low t.
            gate_start = int(action.shape[1]) + int(noisy_start)
            gate_stop = min(gate_start + int(noisy_len), int(v_all.shape[2]))
            if gate_start < gate_stop:
                scale = noisy_value_gate.to(device=v_all.device, dtype=v_all.dtype)
                v_all = torch.cat([
                    v_all[:, :, :gate_start],
                    v_all[:, :, gate_start:gate_stop] * scale[:, None, None, None],
                    v_all[:, :, gate_stop:],
                ], dim=2)
        mask = self._action_mask(int(action.shape[1]), int(cond_tokens.shape[1]), int(noisy_start), int(noisy_len), action.device)
        hierarchical_groups = int(low_len) > 0 or int(stage_len) > 0
        if hierarchical_groups:
            key_bias = self._hierarchical_key_bias(
                action_len=int(action.shape[1]),
                cond_len=int(cond_tokens.shape[1]),
                low_start=int(low_start),
                low_len=int(low_len),
                stage_start=int(stage_start),
                stage_len=int(stage_len),
                noisy_start=int(noisy_start),
                noisy_len=int(noisy_len),
                device=action.device,
            )
        else:
            key_bias = self._action_key_bias(
                action_len=int(action.shape[1]),
                cond_len=int(cond_tokens.shape[1]),
                rollout_start=int(rollout_start),
                rollout_len=int(rollout_len),
                device=action.device,
            )
        batch_key_bias = None
        if any(value is not None for value in (noisy_logit_bias, low_logit_bias, stage_logit_bias)):
            total = int(action.shape[1]) + int(cond_tokens.shape[1])
            batch_key_bias = torch.zeros(
                int(action.shape[0]), total, device=action.device, dtype=torch.float32
            )
            for local_start, length, value in (
                (noisy_start, noisy_len, noisy_logit_bias),
                (low_start, low_len, low_logit_bias),
                (stage_start, stage_len, stage_logit_bias),
            ):
                if value is None or int(length) <= 0:
                    continue
                start = int(action.shape[1]) + int(local_start)
                stop = min(start + int(length), total)
                if start < stop:
                    batch_key_bias[:, start:stop] = value.float().reshape(-1, 1)
        action_attn, weights = self._attention(aq, k_all, v_all, mask, key_bias, batch_key_bias)
        action = action + torch.tanh(a_sa_g)[:, None] * self.drop(self.action_out(self._merge_heads(action_attn)))
        action = action + torch.tanh(a_ff_g)[:, None] * self.drop(
            self.action_ffn(self._modulate(self.action_ffn_norm(action), a_ff_s, a_ff_c))
        )

        cond_update_norm = torch.zeros((), device=action.device, dtype=torch.float32)
        if update_condition:
            cond_attn, _ = self._attention(cq, ck, cv, None)
            cond_tokens = cond_tokens + torch.tanh(c_sa_g)[:, None] * self.drop(self.cond_out(self._merge_heads(cond_attn)))
            cond_tokens = cond_tokens + torch.tanh(c_ff_g)[:, None] * self.drop(
                self.cond_ffn(self._modulate(self.cond_ffn_norm(cond_tokens), c_ff_s, c_ff_c))
            )
            cond_update_norm = (cond_tokens - cond_before).detach().float().norm(dim=-1).mean()

        action_len = int(action_before.shape[1])
        cond_start = action_len
        batch = int(weights.shape[0])
        cond_len = int(cond_tokens.shape[1])
        detached_weights = weights.detach().float()
        cond_mass_rows = detached_weights[..., cond_start:].sum(dim=-1).mean(dim=(1, 2))
        cond_mass = cond_mass_rows.mean()

        prior_logits = torch.zeros(batch, action_len, cond_len, device=action.device, dtype=torch.float32)
        if key_bias is not None:
            prior_logits = prior_logits + key_bias[cond_start:].float()[None, None]
        if batch_key_bias is not None:
            prior_logits = prior_logits + batch_key_bias[:, None, cond_start:].float()
        cond_prior = prior_logits.exp()
        if mask is not None:
            cond_prior = cond_prior.masked_fill(mask[None, :, cond_start:], 0.0)
        cond_prior_total = cond_prior.sum(dim=-1).clamp_min(1e-6)

        def group_stats(local_start: int, length: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
            start = max(int(local_start), 0)
            stop = min(start + max(int(length), 0), cond_len)
            if start >= stop:
                zeros = torch.zeros(batch, device=action.device, dtype=torch.float32)
                scalar = torch.zeros((), device=action.device, dtype=torch.float32)
                return zeros, scalar, zeros, scalar
            absolute_start = cond_start + start
            absolute_stop = cond_start + stop
            mass_rows = detached_weights[..., absolute_start:absolute_stop].sum(dim=-1).mean(dim=(1, 2))
            expected_rows = (
                cond_prior[..., start:stop].sum(dim=-1) / cond_prior_total
            ).mean(dim=1)
            enrichment = (
                (mass_rows / cond_mass_rows.clamp_min(1e-6)) / expected_rows.clamp_min(1e-6)
            ).mean()
            return mass_rows, mass_rows.mean(), expected_rows, enrichment

        noisy_mass_rows, noisy_mass, _, _ = group_stats(noisy_start, noisy_len)
        rollout_mass_rows, rollout_mass, _, rollout_enrichment = group_stats(
            rollout_start, rollout_len
        )
        low_mass_rows, low_mass, low_expected_rows, low_enrichment = group_stats(low_start, low_len)
        stage_mass_rows, stage_mass, stage_expected_rows, stage_enrichment = group_stats(stage_start, stage_len)
        if hierarchical_groups:
            workspace_mass_rows = low_mass_rows + stage_mass_rows
            workspace_expected_rows = low_expected_rows + stage_expected_rows
            workspace_enrichment = (
                (workspace_mass_rows / cond_mass_rows.clamp_min(1e-6))
                / workspace_expected_rows.clamp_min(1e-6)
            ).mean()
        else:
            workspace_mass_rows = rollout_mass_rows
            workspace_enrichment = rollout_enrichment
        metrics = {
            "action_update_norm": (action - action_before).detach().float().norm(dim=-1).mean(),
            "cond_update_norm": cond_update_norm,
            "action_cond_attn": cond_mass,
            "action_noisy_attn": noisy_mass,
            "action_low_attn": low_mass,
            "action_stage_attn": stage_mass,
            "action_low_enrichment": low_enrichment,
            "action_stage_enrichment": stage_enrichment,
            "action_workspace_attn": workspace_mass_rows.mean(),
            "action_workspace_enrichment": workspace_enrichment,
            "action_rollout_attn": rollout_mass,
            "action_rollout_enrichment": rollout_enrichment,
            "action_noisy_attn_rows": noisy_mass_rows,
            "action_low_attn_rows": low_mass_rows,
            "action_stage_attn_rows": stage_mass_rows,
            "action_workspace_attn_rows": workspace_mass_rows,
            "action_rollout_attn_rows": rollout_mass_rows,
        }
        return action, cond_tokens, metrics


class AdaptiveRecurrentCVAERefinementBlock(nn.Module):
    """Shared causal refinement block for adaptive recurrent CVAE actions.

    The block is intentionally small and parameter-shared across refinement
    steps.  Prefix and routed layer context are token-local biases; a causal
    self-attention update then lets later horizon positions read earlier
    predicted action/state summaries without letting this become a full V41
    memory cross-attention decoder.
    """

    def __init__(self, config: LegacyPolicyConfig) -> None:
        super().__init__()
        h = int(config.hidden_size)
        heads = int(config.num_heads)
        self.n1 = nn.LayerNorm(h, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(h, heads, batch_first=True, dropout=float(config.dropout))
        self.n2 = nn.LayerNorm(h, elementwise_affine=False)
        self.ffn = BiasFreeFFN(h, float(getattr(config, "latent_cvae_ffn_expansion", 2.0)))
        self.drop = nn.Dropout(float(config.dropout))
        self.mod = nn.Linear(h, 6 * h)
        self.continue_gate = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 1))
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)
        nn.init.zeros_(self.continue_gate[-1].weight)
        nn.init.zeros_(self.continue_gate[-1].bias)

    @staticmethod
    def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        return x * (1.0 + scale[:, None]) + shift[:, None]

    def forward(self, x: Tensor, cond: Tensor, routed: Tensor, prefix: Tensor) -> tuple[Tensor, Tensor]:
        token_cond = routed + prefix
        keep = torch.sigmoid(self.continue_gate(x + token_cond))
        sa_s, sa_c, sa_g, ff_s, ff_c, ff_g = self.mod(cond).chunk(6, dim=-1)
        value = self.n1(x + token_cond)
        qk = self._modulate(value, sa_s, sa_c)
        n = int(x.shape[1])
        causal_mask = torch.triu(torch.ones(n, n, device=x.device, dtype=torch.bool), diagonal=1)
        update, _ = self.self_attn(qk, qk, value, attn_mask=causal_mask, need_weights=False)
        x = x + keep * torch.tanh(sa_g)[:, None] * self.drop(update)
        update = self.ffn(self._modulate(self.n2(x + token_cond), ff_s, ff_c))
        x = x + keep * torch.tanh(ff_g)[:, None] * self.drop(update)
        return x, keep.detach().float().mean()


class AdaptiveCVAEMicroRefineBlock(nn.Module):
    """Controller-style refine block whose only action output is an update.

    The internal causal block may build a stronger control state, but that
    state is never written to the action tokens directly.  Action changes must
    pass through the bounded micro-step control law.
    """

    def __init__(self, config: LegacyPolicyConfig) -> None:
        super().__init__()
        self.config = config
        h = int(config.hidden_size)
        self.controller = AdaptiveRecurrentCVAERefinementBlock(config)
        self.gain_head = nn.Sequential(nn.LayerNorm(6 * h), nn.Linear(6 * h, h), nn.SiLU(), nn.Linear(h, 3))
        self.reference = nn.Sequential(nn.LayerNorm(5 * h), nn.Linear(5 * h, h), nn.SiLU(), nn.Linear(h, h))
        self.feedforward = nn.Sequential(nn.LayerNorm(5 * h), nn.Linear(5 * h, h), nn.SiLU(), nn.Linear(h, h))
        self.function_bank = AdaptiveCVAEFunctionBank(config)
        self._init_residual(self.reference, std=float(getattr(config, "latent_cvae_output_init_std", 1e-3)))
        self._init_residual(self.feedforward, std=float(getattr(config, "latent_cvae_output_init_std", 1e-3)))
        gain_head = self.gain_head[-1]
        if isinstance(gain_head, nn.Linear):
            nn.init.zeros_(gain_head.weight)
            step_bias = self._bounded_sigmoid_bias(
                value=float(getattr(config, "adaptive_cvae_micro_step_init", 0.12)),
                lo=float(getattr(config, "adaptive_cvae_micro_min_step", 0.03)),
                hi=float(getattr(config, "adaptive_cvae_micro_max_step", 0.35)),
            )
            kp_bias = self._bounded_sigmoid_bias(
                value=float(getattr(config, "adaptive_cvae_micro_kp_init", 0.18)),
                lo=0.0,
                hi=float(getattr(config, "adaptive_cvae_micro_kp_max", 0.60)),
            )
            kd_bias = self._bounded_sigmoid_bias(
                value=float(getattr(config, "adaptive_cvae_micro_kd_init", 0.08)),
                lo=0.0,
                hi=float(getattr(config, "adaptive_cvae_micro_kd_max", 0.45)),
            )
            with torch.no_grad():
                gain_head.bias.copy_(torch.tensor([step_bias, kp_bias, kd_bias], dtype=gain_head.bias.dtype))

    @staticmethod
    def _init_residual(module: nn.Module, *, std: float) -> None:
        last = module[-1] if isinstance(module, nn.Sequential) else None
        if isinstance(last, nn.Linear):
            if std > 0:
                nn.init.normal_(last.weight, mean=0.0, std=std)
            else:
                nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    @staticmethod
    def _bounded_sigmoid_bias(*, value: float, lo: float, hi: float) -> float:
        if hi <= lo:
            return 0.0
        frac = min(max((float(value) - lo) / (hi - lo), 1e-4), 1.0 - 1e-4)
        return math.log(frac / (1.0 - frac))

    @staticmethod
    def _bounded_sigmoid(raw: Tensor, *, lo: float, hi: float) -> Tensor:
        if hi <= lo:
            return torch.full_like(raw, float(lo))
        return float(lo) + (float(hi) - float(lo)) * torch.sigmoid(raw.float())

    def _gains(
        self,
        *,
        action: Tensor,
        control_state: Tensor,
        cond_tokens: Tensor,
        progress_context: Tensor,
        context_dir: Tensor,
        step_bias: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        cfg = self.config
        raw = self.gain_head(torch.cat([action, control_state, cond_tokens, progress_context, context_dir, step_bias], dim=-1)).float()
        raw_step, raw_kp, raw_kd = raw.split(1, dim=-1)
        ds = self._bounded_sigmoid(
            raw_step,
            lo=float(getattr(cfg, "adaptive_cvae_micro_min_step", 0.03)),
            hi=float(getattr(cfg, "adaptive_cvae_micro_max_step", 0.35)),
        ).to(device=action.device, dtype=action.dtype)
        kp = self._bounded_sigmoid(
            raw_kp,
            lo=0.0,
            hi=float(getattr(cfg, "adaptive_cvae_micro_kp_max", 0.60)),
        ).to(device=action.device, dtype=action.dtype)
        kd = self._bounded_sigmoid(
            raw_kd,
            lo=0.0,
            hi=float(getattr(cfg, "adaptive_cvae_micro_kd_max", 0.45)),
        ).to(device=action.device, dtype=action.dtype)
        return ds, kp, kd

    def _field(
        self,
        *,
        action: Tensor,
        prev_update: Tensor,
        cond_time: Tensor,
        progress_context: Tensor,
        context_dir: Tensor,
        step_bias: Tensor,
        semantic_bias: Tensor,
        progress_weights: Tensor | None,
        role_basis: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        cond_tokens = cond_time[:, None].expand(-1, int(action.shape[1]), -1)
        prefix = progress_context + step_bias + semantic_bias
        control_state, keep = self.controller(action, cond_time, context_dir, prefix)
        ds, kp, kd = self._gains(
            action=action,
            control_state=control_state,
            cond_tokens=cond_tokens,
            progress_context=progress_context,
            context_dir=context_dir,
            step_bias=step_bias,
        )
        reference_delta = self.reference(torch.cat([control_state, cond_tokens, progress_context, context_dir, step_bias], dim=-1))
        feedback = kp * torch.tanh(reference_delta)
        damping = kd * prev_update
        feedforward = self.feedforward(torch.cat([action, control_state, progress_context, context_dir, step_bias], dim=-1))
        function = self.function_bank(control_state + progress_context + context_dir, progress_weights, role_basis)
        control = feedforward + function + feedback - damping
        return control, ds, kp, kd, {
            "reference": reference_delta,
            "feedforward": feedforward,
            "feedback": feedback,
            "damping": damping,
            "function": function,
            "control": control,
            "controller": control_state - action,
            "keep": keep.to(device=action.device),
        }

    def forward(
        self,
        *,
        action: Tensor,
        prev_update: Tensor,
        cond_time: Tensor,
        progress_context: Tensor,
        context_dir: Tensor,
        step_bias: Tensor,
        semantic_bias: Tensor,
        progress_weights: Tensor | None,
        role_basis: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        cfg = self.config
        control1, ds, kp, kd, terms1 = self._field(
            action=action,
            prev_update=prev_update,
            cond_time=cond_time,
            progress_context=progress_context,
            context_dir=context_dir,
            step_bias=step_bias,
            semantic_bias=semantic_bias,
            progress_weights=progress_weights,
            role_basis=role_basis,
        )
        if int(getattr(cfg, "adaptive_cvae_micro_heun", 1)):
            pred = action + ds * control1
            control2, _, _, _, terms2 = self._field(
                action=pred,
                prev_update=ds * control1,
                cond_time=cond_time,
                progress_context=progress_context,
                context_dir=context_dir,
                step_bias=step_bias,
                semantic_bias=semantic_bias,
                progress_weights=progress_weights,
                role_basis=role_basis,
            )
            control = 0.5 * (control1 + control2)
            terms = {
                key: 0.5 * (terms1[key] + terms2[key])
                for key in ("feedforward", "feedback", "damping", "function", "control", "controller")
            }
            terms["reference"] = terms1["reference"]
            terms["keep"] = 0.5 * (terms1["keep"] + terms2["keep"])
            terms["heun_error"] = (control2 - control1).detach().float().norm(dim=-1).mean()
        else:
            control = control1
            terms = dict(terms1)
            terms["heun_error"] = torch.zeros((), device=action.device, dtype=torch.float32)
        update = float(getattr(cfg, "adaptive_cvae_micro_update_scale", 1.0)) * ds * control
        return update, ds, kp, kd, terms


def _progress_role_basis(steps: int, dim: int) -> Tensor:
    if steps < 1 or dim < 2:
        raise ValueError("progress role basis requires steps >= 1 and dim >= 2")
    pos = torch.linspace(-1.0, 1.0, steps, dtype=torch.float32)
    cols = [pos, pos.square(), torch.sin(math.pi * pos), torch.cos(math.pi * pos)]
    freq = 2.0
    while len(cols) < dim:
        cols.append(torch.sin(freq * math.pi * pos))
        if len(cols) < dim:
            cols.append(torch.cos(freq * math.pi * pos))
        freq += 1.0
    basis = torch.stack(cols[:dim], dim=-1)
    return F.normalize(basis, dim=-1)


class AdaptiveCVAEFunctionBank(nn.Module):
    """Low-rank function experts selected by latent progress routing."""

    def __init__(self, config: LegacyPolicyConfig) -> None:
        super().__init__()
        h = int(config.hidden_size)
        experts = int(getattr(config, "adaptive_cvae_progress_steps", 6))
        rank = int(getattr(config, "adaptive_cvae_function_rank", 64))
        role_dim = int(getattr(config, "adaptive_cvae_progress_role_dim", 16))
        self.experts = experts
        self.rank = rank
        self.in_norm = nn.LayerNorm(h)
        self.role_down = nn.Linear(role_dim, h * rank, bias=False)
        self.role_up = nn.Linear(role_dim, rank * h, bias=False)
        self.down = nn.Parameter(torch.empty(experts, h, rank))
        self.up = nn.Parameter(torch.empty(experts, rank, h))
        nn.init.normal_(self.down, mean=0.0, std=0.02)
        nn.init.zeros_(self.up)
        nn.init.normal_(self.role_down.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.role_up.weight)

    def forward(self, x: Tensor, weights: Tensor | None, role_basis: Tensor | None = None) -> Tensor:
        if weights is None or int(weights.shape[-1]) != self.experts:
            return torch.zeros_like(x)
        value = self.in_norm(x)
        down = self.down.to(device=x.device, dtype=x.dtype)
        up = self.up.to(device=x.device, dtype=x.dtype)
        if role_basis is not None:
            role = role_basis.to(device=x.device, dtype=x.dtype)
            down = down + self.role_down(role).reshape(self.experts, int(x.shape[-1]), self.rank)
            up = up + self.role_up(role).reshape(self.experts, self.rank, int(x.shape[-1]))
        hidden = torch.einsum("bth,ehr->bter", value, down)
        update = torch.einsum("bter,erh->bteh", F.silu(hidden), up)
        return torch.einsum("bte,bteh->bth", weights.to(device=x.device, dtype=x.dtype), update)

class LatentCVAEActionDecoder(nn.Module):
    """V42 compact latent-conditioned CVAE action head.

    The final policy is still a single path: V40 latent/consequence trunk ->
    CVAE condition -> action tokens -> 14-D physical velocity.  The old V40
    direct/rollout heads are not used as a base and no residual side branch is
    added.  Training uses q(z | condition, target physical action); inference
    uses p(z | condition), deterministically by default.
    """

    _LAYER_KEYS = LayeredV37StyleResidualActionFlowDenoiser._LAYER_KEYS
    _CONSEQUENCE_LAYER_KEYS = _LAYER_KEYS[2:]

    def __init__(self, config: LegacyPolicyConfig) -> None:
        super().__init__()
        self.config = config
        h = int(config.hidden_size)
        self.hidden_size = h
        self.z_dim = int(getattr(config, "latent_cvae_z_dim", 64))
        self.depth = int(getattr(config, "latent_cvae_decoder_depth", 3))
        self.time = TimeEmbedding(h)
        self.horizon_query = nn.Parameter(torch.randn(1, int(config.action_horizon), h) * 0.02)
        parseval_gripper = str(getattr(config, "gripper_field_mode", "legacy_handcrafted")) == "parseval_temporal"
        self.noisy_action_lift = (
            PhysicalActionTokenLift(config)
            if parseval_gripper
            else nn.Sequential(nn.LayerNorm(int(config.physical_action_dim)), nn.Linear(int(config.physical_action_dim), h))
        )
        self.trajectory_lift = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
        self.time_lift = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
        # One projection per V40 layer.  This makes every layer latent explicitly
        # enter the condition vector rather than being silently averaged away.
        self.layer_proj = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h))
            for _ in range(int(config.depth))
        ])
        self.layer_key_proj = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
            for _ in self._LAYER_KEYS
        ])
        for proj in self.layer_key_proj:
            nn.init.eye_(proj[-1].weight)
            nn.init.zeros_(proj[-1].bias)
        self.layer_key_embed = nn.Parameter(torch.randn(1, len(self._LAYER_KEYS), h) * 0.02)
        self.layer_key_gate = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 1))
        nn.init.zeros_(self.layer_key_gate[-1].weight)
        nn.init.zeros_(self.layer_key_gate[-1].bias)
        consequence_scale_max = float(getattr(config, "latent_cvae_consequence_scale_max", 0.50))
        consequence_scale_init = float(getattr(config, "latent_cvae_consequence_scale_init", 0.10))
        consequence_scale_ratio = min(max(consequence_scale_init / consequence_scale_max, 1e-4), 1.0 - 1e-4)
        consequence_scale_logit = math.log(consequence_scale_ratio / (1.0 - consequence_scale_ratio))
        self.layer_consequence_scale_logits = nn.Parameter(
            torch.full((len(self._CONSEQUENCE_LAYER_KEYS),), consequence_scale_logit)
        )
        self._consequence_scale_index = {
            key: index for index, key in enumerate(self._CONSEQUENCE_LAYER_KEYS)
        }
        self.layer_embed = nn.Parameter(torch.randn(1, int(config.depth), h) * 0.02)
        self.transition_proj = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h))
        self.context_proj = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h))
        self.visual_proj = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h))
        self.traj_summary_proj = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h))
        self.condition_contract_norm = nn.LayerNorm(h, elementwise_affine=False)
        cond_in = int(config.depth) * h + 4 * h
        self.condition_fusion = nn.Sequential(nn.LayerNorm(cond_in), nn.Linear(cond_in, h), nn.SiLU(), nn.Linear(h, h))
        if int(getattr(config, "latent_cvae_layer_scan", 0)):
            self.layer_scan = nn.GRUCell(h, h)
            self.layer_scan_init = nn.Parameter(torch.zeros(1, h))
            self.layer_scan_fusion = nn.Sequential(nn.LayerNorm(5 * h), nn.Linear(5 * h, h), nn.SiLU(), nn.Linear(h, h))
        else:
            self.layer_scan = None
            self.layer_scan_init = None
            self.layer_scan_fusion = None
        self.prior = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 2 * self.z_dim))
        if parseval_gripper:
            self.posterior_action = nn.Sequential(
                PhysicalActionTokenLift(config),
                nn.LayerNorm(h),
                nn.Linear(h, h),
                nn.SiLU(),
                nn.Linear(h, h),
            )
        else:
            self.posterior_action = nn.Sequential(
                nn.LayerNorm(int(config.physical_action_dim)),
                nn.Linear(int(config.physical_action_dim), h),
                nn.SiLU(),
                nn.Linear(h, h),
            )
        self.posterior = nn.Sequential(nn.LayerNorm(2 * h), nn.Linear(2 * h, h), nn.SiLU(), nn.Linear(h, 2 * self.z_dim))
        self.z_to_token = nn.Sequential(nn.LayerNorm(self.z_dim), nn.Linear(self.z_dim, h))
        self.blocks = nn.ModuleList([LatentCVAEActionBlock(config) for _ in range(self.depth)])
        if int(getattr(config, "latent_cvae_mmdit_decoder", 0)):
            mmdit_depth = int(getattr(config, "latent_cvae_mmdit_depth", self.depth))
            self.mmdit_blocks = nn.ModuleList([LatentCVAEMMDiTBlock(config) for _ in range(mmdit_depth)])
            self.mmdit_traj_cond_proj = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
            self.mmdit_rollout_cond_proj = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
            self.mmdit_rollout_type = nn.Parameter(torch.randn(1, 1, h) * 0.02)
            nn.init.eye_(self.mmdit_rollout_cond_proj[-1].weight)
            nn.init.zeros_(self.mmdit_rollout_cond_proj[-1].bias)
            self.mmdit_cond_global_proj = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
            self.mmdit_z_global_proj = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
            self.mmdit_progress_proj = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
            self.mmdit_step_cond_proj = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
            self.mmdit_type_embed = nn.Parameter(torch.randn(1, 6, h) * 0.02)
            self.mmdit_action_norm = nn.LayerNorm(h)
            self.mmdit_primary_condition_norm = nn.LayerNorm(h, elementwise_affine=False)
            self.mmdit_noisy_norm = (
                nn.LayerNorm(h)
                if int(getattr(config, "latent_cvae_mmdit_noisy_logit_gate", 0))
                else None
            )
            self.evidence_workspace = SemanticEvidenceWorkspace(config)
            self.hierarchical_workspace = (
                HierarchicalEvidenceWorkspace(config)
                if int(getattr(config, "latent_cvae_hierarchical_workspace", 0))
                else None
            )
            if self.hierarchical_workspace is not None:
                # Keep legacy parameters loadable for checkpoint compatibility,
                # but exclude unused legacy workspace/action paths from
                # gradients. Otherwise the old z-conditioned action blocks can
                # solve the sample before low/stage MMDiT refinement is used.
                self.evidence_workspace.requires_grad_(False)
                self.blocks.requires_grad_(False)
        else:
            self.mmdit_blocks = nn.ModuleList()
            self.mmdit_traj_cond_proj = None
            self.mmdit_rollout_cond_proj = None
            self.mmdit_rollout_type = None
            self.mmdit_cond_global_proj = None
            self.mmdit_z_global_proj = None
            self.mmdit_progress_proj = None
            self.mmdit_step_cond_proj = None
            self.mmdit_type_embed = None
            self.mmdit_action_norm = None
            self.mmdit_primary_condition_norm = None
            self.mmdit_noisy_norm = None
            self.evidence_workspace = None
            self.hierarchical_workspace = None
        self.event_gate = nn.Sequential(nn.LayerNorm(2 * h), nn.Linear(2 * h, h), nn.Sigmoid())
        self.event_transition = nn.Sequential(nn.LayerNorm(2 * h), nn.Linear(2 * h, h), nn.SiLU(), nn.Linear(h, h))
        self.velocity_head = TransitionAwarePhysicalVelocityHead(config)
        self.event_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 3))
        self.motion_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 1))
        self._initialize_outputs()

    def _initialize_outputs(self) -> None:
        std = float(getattr(self.config, "latent_cvae_output_init_std", 1e-3))
        for module in self.velocity_head.output_layers():
            if std > 0:
                nn.init.normal_(module.weight, mean=0.0, std=std)
            else:
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

    def _normalize_condition_source(self, value: Tensor) -> Tensor:
        if int(getattr(self.config, "latent_cvae_condition_source_norm", 1)):
            return F.layer_norm(value, (self.hidden_size,))
        return value

    def _consequence_scales(self, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        scale_max = float(getattr(self.config, "latent_cvae_consequence_scale_max", 0.50))
        return scale_max * torch.sigmoid(self.layer_consequence_scale_logits).to(device=device, dtype=dtype)

    def _layer_entry_summary(
        self,
        entry: dict[str, Tensor],
        *,
        detach: bool,
    ) -> tuple[Tensor | None, dict[str, Tensor]]:
        groups: list[Tensor] = []
        consequence_group: list[bool] = []
        active_scales: list[Tensor] = []
        grad_scale = float(getattr(self.config, "latent_cvae_layer_grad_scale", 0.0))
        bounded_fusion = bool(int(getattr(self.config, "latent_cvae_bounded_consequence_fusion", 1)))
        for key_index, key in enumerate(self._LAYER_KEYS):
            value = entry.get(key)
            if not isinstance(value, Tensor) or value.ndim != 3 or int(value.shape[-1]) != self.hidden_size:
                continue
            source = _scaled_contract_view(value, grad_scale) if detach else value
            pooled = source.mean(dim=1)
            typed = self._normalize_condition_source(self.layer_key_proj[key_index](pooled))
            scale_index = self._consequence_scale_index.get(key)
            is_consequence = scale_index is not None
            if scale_index is not None and bounded_fusion:
                scale = self._consequence_scales(device=typed.device, dtype=typed.dtype)[scale_index]
                typed = typed * scale
                active_scales.append(scale)
            elif scale_index is not None:
                active_scales.append(typed.new_ones(()))
            typed = typed + self.layer_key_embed[:, key_index].to(device=typed.device, dtype=typed.dtype)
            groups.append(typed)
            consequence_group.append(is_consequence)
        if not groups:
            zero = self.layer_consequence_scale_logits.detach().new_zeros(())
            return None, {
                "consequence_scale_mean": zero,
                "consequence_gate_preference": zero,
                "consequence_mix_ratio": zero,
            }
        stack = torch.stack(groups, dim=1)
        logits = self.layer_key_gate(stack).float()
        consequence_mask = torch.tensor(consequence_group, device=stack.device, dtype=torch.bool)
        world_mask = ~consequence_mask
        global_weights = torch.softmax(logits, dim=1).to(dtype=stack.dtype)
        if any(consequence_group) and not all(consequence_group):
            consequence_count = sum(consequence_group)
            world_count = len(consequence_group) - consequence_count
            consequence_score = torch.logsumexp(logits[:, consequence_mask], dim=1) - math.log(consequence_count)
            world_score = torch.logsumexp(logits[:, world_mask], dim=1) - math.log(world_count)
            gate_preference = torch.sigmoid(consequence_score - world_score).mean()
        elif any(consequence_group):
            gate_preference = logits.new_ones(())
        else:
            gate_preference = logits.new_zeros(())
        if bounded_fusion:
            # Select semantics within each family, then mix the families
            # explicitly. A single global softmax could otherwise undo the
            # consequence scale by assigning all mass to one conditioned key.
            if any(consequence_group):
                consequence_weights = torch.softmax(logits[:, consequence_mask], dim=1).to(dtype=stack.dtype)
                consequence_summary = (stack[:, consequence_mask] * consequence_weights).sum(dim=1)
            else:
                consequence_summary = stack.new_zeros(stack.shape[0], stack.shape[-1])
            if not all(consequence_group):
                world_weights = torch.softmax(logits[:, world_mask], dim=1).to(dtype=stack.dtype)
                world_summary = (stack[:, world_mask] * world_weights).sum(dim=1)
            else:
                world_summary = stack.new_zeros(stack.shape[0], stack.shape[-1])
        else:
            consequence_summary = (
                (stack[:, consequence_mask] * global_weights[:, consequence_mask]).sum(dim=1)
                if any(consequence_group) else stack.new_zeros(stack.shape[0], stack.shape[-1])
            )
            world_summary = (
                (stack[:, world_mask] * global_weights[:, world_mask]).sum(dim=1)
                if not all(consequence_group) else stack.new_zeros(stack.shape[0], stack.shape[-1])
            )
        consequence_norm = consequence_summary.detach().float().norm(dim=-1).mean()
        world_norm = world_summary.detach().float().norm(dim=-1).mean()
        mix_ratio = consequence_norm / (world_norm + consequence_norm).clamp_min(1e-6)
        scale_mean = torch.stack(active_scales).mean() if active_scales else stack.new_zeros(())
        return world_summary + consequence_summary, {
            "consequence_scale_mean": scale_mean.detach().float(),
            "consequence_gate_preference": gate_preference.detach().float(),
            "consequence_mix_ratio": mix_ratio,
        }

    @staticmethod
    def _memory_summary(memory: Tensor | list[Tensor] | tuple[Tensor, ...] | None, ref: Tensor, proj: nn.Module) -> Tensor:
        if memory is None:
            return torch.zeros_like(ref)
        groups = [memory] if isinstance(memory, Tensor) else list(memory)
        pooled: list[Tensor] = []
        for value in groups:
            if not isinstance(value, Tensor) or value.ndim != 3 or int(value.shape[-1]) != int(ref.shape[-1]):
                raise ValueError(f"CVAE memory groups must be [B,N,H], got {type(value).__name__}")
            pooled.append(value.to(device=ref.device, dtype=ref.dtype).mean(dim=1))
        if not pooled:
            return torch.zeros_like(ref)
        return proj(torch.stack(pooled, dim=1)).mean(dim=1)

    @staticmethod
    def _memory_tokens(
        memory: Tensor | list[Tensor] | tuple[Tensor, ...] | None,
        ref: Tensor,
        proj: nn.Module,
    ) -> Tensor:
        if memory is None:
            return ref.new_zeros(int(ref.shape[0]), 0, int(ref.shape[-1]))
        groups = [memory] if isinstance(memory, Tensor) else list(memory)
        pooled: list[Tensor] = []
        for value in groups:
            if not isinstance(value, Tensor) or value.ndim != 3 or int(value.shape[-1]) != int(ref.shape[-1]):
                raise ValueError(f"CVAE memory groups must be [B,N,H], got {type(value).__name__}")
            pooled.append(value.to(device=ref.device, dtype=ref.dtype).mean(dim=1))
        if not pooled:
            return ref.new_zeros(int(ref.shape[0]), 0, int(ref.shape[-1]))
        return proj(torch.stack(pooled, dim=1))

    @staticmethod
    def _maybe_detach_memory(
        memory: Tensor | list[Tensor] | tuple[Tensor, ...] | None,
        *,
        detach: bool,
    ) -> Tensor | list[Tensor] | tuple[Tensor, ...] | None:
        if memory is None or not detach:
            return memory
        if isinstance(memory, Tensor):
            return memory.detach()
        return [value.detach() if isinstance(value, Tensor) else value for value in memory]

    def _split_gaussian(self, params: Tensor) -> tuple[Tensor, Tensor]:
        mu, logvar = params.chunk(2, dim=-1)
        mu_bound = float(getattr(self.config, "latent_cvae_mu_bound", 0.0))
        if mu_bound > 0:
            mu = torch.tanh(mu / mu_bound) * mu_bound
        min_std = float(getattr(self.config, "latent_cvae_min_std", 0.0))
        min_logvar = -8.0
        if min_std > 0:
            min_logvar = max(min_logvar, 2.0 * math.log(max(min_std, 1e-6)))
        return mu, logvar.clamp(min=min_logvar, max=4.0)

    @staticmethod
    def _kl_diag_gaussians(q_mu: Tensor, q_logvar: Tensor, p_mu: Tensor, p_logvar: Tensor) -> Tensor:
        q_var = q_logvar.exp()
        p_var = p_logvar.exp().clamp_min(1e-6)
        kl = 0.5 * (p_logvar - q_logvar + (q_var + (q_mu - p_mu).square()) / p_var - 1.0)
        return kl.sum(dim=-1).mean()

    def _emit_action(self, action: Tensor, cond: Tensor) -> dict[str, Tensor]:
        cfg = self.config
        gate_input = torch.cat([action, cond[:, None].expand(-1, int(cfg.action_horizon), -1)], dim=-1)
        if int(getattr(cfg, "latent_cvae_event_gripper_gate", 1)):
            gate = self.event_gate(gate_input)
            transition = action + gate * self.event_transition(gate_input)
        else:
            gate = torch.zeros_like(action)
            transition = action
        pred_velocity = self.velocity_head(action, transition)
        event_logits = self.event_head(action + transition)
        motion_logits = self.motion_head(action).squeeze(-1)
        return {
            "pred_velocity": pred_velocity,
            "event_logits": event_logits,
            "motion_logits": motion_logits,
            "action_tokens": action,
            "transition_latent": transition,
            "gripper_gate_mean": gate.detach().float().mean(),
        }

    def _noisy_time_gate(self, time: Tensor) -> Tensor | None:
        if not int(getattr(self.config, "latent_cvae_noisy_gate", 0)):
            return None
        min_gate = float(getattr(self.config, "latent_cvae_noisy_gate_min", 0.05))
        power = float(getattr(self.config, "latent_cvae_noisy_gate_power", 1.5))
        t = time.float().clamp(0.0, 1.0)
        return (min_gate + (1.0 - min_gate) * t.pow(power))[:, None, None]

    def _gated_noisy_branch(self, noisy_physical: Tensor, time: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        branch = self.noisy_action_lift(noisy_physical)
        gate = self._noisy_time_gate(time)
        if gate is None:
            gate_mean = torch.ones((), device=branch.device, dtype=torch.float32)
        else:
            branch = branch * gate.to(device=branch.device, dtype=branch.dtype)
            gate_mean = gate.detach().float().mean()
        return branch, gate_mean, branch.detach().float().norm(dim=-1).mean()

    def _mmdit_progress_tokens(self, *, batch: int, cond_time: Tensor, z: Tensor) -> Tensor | None:
        progress_fn = getattr(self, "_latent_progress", None)
        if not callable(progress_fn):
            return None
        if not int(getattr(self.config, "adaptive_cvae_progress_memory", 0)):
            return None
        return progress_fn(batch=batch, cond_time=cond_time, z=z)

    def _mmdit_primary_condition(self, *, z: Tensor, time_emb: Tensor) -> Tensor:
        if self.mmdit_primary_condition_norm is None:
            raise RuntimeError("MMDiT primary-condition modules are not initialized")
        dtype = time_emb.dtype
        primary = self.z_to_token(z.to(device=time_emb.device, dtype=dtype)) + self.time_lift(time_emb)
        return self.mmdit_primary_condition_norm(primary)

    def _mmdit_primary_z_effect(self, *, z: Tensor, time_emb: Tensor, primary_cond: Tensor) -> Tensor:
        with torch.no_grad():
            zero_primary = self._mmdit_primary_condition(z=torch.zeros_like(z), time_emb=time_emb)
            return (primary_cond.detach().float() - zero_primary.detach().float()).norm(dim=-1).mean()

    def _workspace_query_action(self, action: Tensor, noisy: Tensor) -> tuple[Tensor, Tensor]:
        if not int(getattr(self.config, "latent_cvae_workspace_noisy_query", 0)):
            return action, torch.zeros((), device=action.device, dtype=torch.float32)
        if action.shape != noisy.shape:
            raise ValueError(f"workspace action/noisy query mismatch: {tuple(action.shape)} vs {tuple(noisy.shape)}")
        action_norm = action.detach().float().norm(dim=-1, keepdim=True).clamp_min(1e-4)
        noisy_norm = noisy.detach().float().norm(dim=-1, keepdim=True).clamp_min(1e-4)
        scale = (action_norm / noisy_norm).clamp(max=8.0)
        # Query-only conditioning: detach x_t and match its token norm to the
        # current action query. Evidence values remain condition-only, so this
        # cannot become a second noisy-action residual stream.
        noisy_query = noisy.detach() * scale.to(device=action.device, dtype=action.dtype)
        return action + noisy_query, scale.mean()

    @staticmethod
    def _time_stratified_attention(
        time: Tensor,
        noisy_rows: Tensor,
        workspace_rows: Tensor,
        low_rows: Tensor | None = None,
        stage_rows: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """V72 S3 gauge: x_t vs workspace attention share, stratified by flow time.

        Emits per-bucket SUM and COUNT rather than a per-batch ratio so the
        epoch-level averaging pipeline stays statistically exact:
        mean_over_batches(sum) / mean_over_batches(count) equals the true
        stratified mean, whereas averaging per-batch ratios would weight
        empty/sparse buckets incorrectly. Buckets: t in [0,1/3), [1/3,2/3),
        [2/3,1]. t=0 is data, t=1 is noise; the shortcut-vs-legitimate-need
        question lives at LOW t, where deploy-time x_t is nearly the model's
        own output and train-time x_t is nearly the oracle.
        """
        t = time.detach().float().reshape(-1)
        noisy_rows = noisy_rows.detach().float().reshape(-1)
        workspace_rows = workspace_rows.detach().float().reshape(-1)
        low_rows = torch.zeros_like(workspace_rows) if low_rows is None else low_rows.detach().float().reshape(-1)
        stage_rows = torch.zeros_like(workspace_rows) if stage_rows is None else stage_rows.detach().float().reshape(-1)
        out: dict[str, Tensor] = {}
        edges = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0 + 1e-6)
        for i in range(3):
            mask = ((t >= edges[i]) & (t < edges[i + 1])).float()
            out[f"mmdit_noisy_attn_t{i}_sum"] = (noisy_rows * mask).sum()
            out[f"mmdit_workspace_attn_t{i}_sum"] = (workspace_rows * mask).sum()
            out[f"mmdit_low_attn_t{i}_sum"] = (low_rows * mask).sum()
            out[f"mmdit_stage_attn_t{i}_sum"] = (stage_rows * mask).sum()
            out[f"mmdit_attn_t{i}_count"] = mask.sum()
        return out

    def _mmdit_condition_tokens(
        self,
        *,
        noisy_tokens: Tensor,
        trajectory_tokens: Tensor,
        rollout_tokens: Tensor | None,
        cond_time: Tensor,
        z_token: Tensor,
        layer_stack: Tensor | None,
        progress_tokens: Tensor | None,
        workspace_tokens: Tensor | None = None,
        low_workspace_tokens: Tensor | None = None,
        stage_workspace_tokens: Tensor | None = None,
    ) -> tuple[Tensor, MMDiTConditionLayout, Tensor]:
        if (
            self.mmdit_traj_cond_proj is None
            or self.mmdit_cond_global_proj is None
            or self.mmdit_z_global_proj is None
            or self.mmdit_type_embed is None
        ):
            raise RuntimeError("MMDiT condition modules are not initialized")
        dtype = noisy_tokens.dtype
        device = noisy_tokens.device
        type_embed = self.mmdit_type_embed.to(device=device, dtype=dtype)
        hierarchical = low_workspace_tokens is not None or stage_workspace_tokens is not None
        if hierarchical:
            if low_workspace_tokens is None or stage_workspace_tokens is None:
                raise ValueError("hierarchical MMDiT conditions require both low and stage token groups")
            low_group = low_workspace_tokens.to(device=device, dtype=dtype) + type_embed[:, 0:1]
            stage_group = stage_workspace_tokens.to(device=device, dtype=dtype) + type_embed[:, 4:5]
            noisy_group = noisy_tokens + type_embed[:, 2:3]
            low_start = 0
            stage_start = int(low_group.shape[1])
            noisy_start = stage_start + int(stage_group.shape[1])
            cond_tokens = torch.cat([low_group, stage_group, noisy_group], dim=1)
            layout = MMDiTConditionLayout(
                noisy_start=noisy_start,
                noisy_len=int(noisy_group.shape[1]),
                low_start=low_start,
                low_len=int(low_group.shape[1]),
                stage_start=stage_start,
                stage_len=int(stage_group.shape[1]),
            )
            return cond_tokens, layout, cond_tokens.detach().float().norm(dim=-1).mean()
        if workspace_tokens is not None:
            workspace_group = workspace_tokens.to(device=device, dtype=dtype) + type_embed[:, 0:1]
            noisy_start = int(workspace_group.shape[1])
            cond_tokens = torch.cat([workspace_group, noisy_tokens + type_embed[:, 2:3]], dim=1)
            cond_norm = cond_tokens.detach().float().norm(dim=-1).mean()
            # The generic balanced group range in LatentCVAEMMDiTBlock is used
            # for the workspace here. Source-level rollout mass is measured by
            # SemanticEvidenceWorkspace itself.
            layout = MMDiTConditionLayout(
                noisy_start=noisy_start,
                noisy_len=int(noisy_tokens.shape[1]),
                rollout_start=0,
                rollout_len=noisy_start,
            )
            return cond_tokens, layout, cond_norm
        groups: list[Tensor] = []
        if layer_stack is not None:
            groups.append(layer_stack.to(device=device, dtype=dtype) + type_embed[:, 0:1])
        traj_tokens = self.mmdit_traj_cond_proj(trajectory_tokens.to(device=device, dtype=dtype)) + type_embed[:, 1:2]
        groups.append(traj_tokens)
        rollout_start = sum(int(group.shape[1]) for group in groups)
        rollout_len = 0
        if rollout_tokens is not None:
            if self.mmdit_rollout_cond_proj is None or self.mmdit_rollout_type is None:
                raise RuntimeError("MMDiT rollout condition modules are not initialized")
            rollout_group = self.mmdit_rollout_cond_proj(
                rollout_tokens.to(device=device, dtype=dtype)
            ) + self.mmdit_rollout_type.to(device=device, dtype=dtype)
            groups.append(rollout_group)
            rollout_len = int(rollout_group.shape[1])
        noisy_start = sum(int(group.shape[1]) for group in groups)
        groups.append(noisy_tokens + type_embed[:, 2:3])
        global_tokens = torch.stack([
            self.mmdit_cond_global_proj(cond_time.to(device=device, dtype=dtype)),
            self.mmdit_z_global_proj(z_token.to(device=device, dtype=dtype)),
        ], dim=1) + type_embed[:, 3:4]
        groups.append(global_tokens)
        if progress_tokens is not None and self.mmdit_progress_proj is not None:
            groups.append(self.mmdit_progress_proj(progress_tokens.to(device=device, dtype=dtype)) + type_embed[:, 4:5])
        cond_tokens = torch.cat(groups, dim=1)
        cond_norm = cond_tokens.detach().float().norm(dim=-1).mean()
        layout = MMDiTConditionLayout(
            noisy_start=noisy_start,
            noisy_len=int(noisy_tokens.shape[1]),
            rollout_start=rollout_start,
            rollout_len=rollout_len,
        )
        return cond_tokens, layout, cond_norm

    def _decode_with_z_mmdit(
        self,
        *,
        noisy_physical: Tensor,
        time: Tensor,
        trajectory_tokens: Tensor,
        rollout_tokens: Tensor | None,
        cond: Tensor,
        z: Tensor,
        layer_stack: Tensor | None = None,
        evidence_sources: dict[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        if self.mmdit_action_norm is None:
            raise RuntimeError("MMDiT action modules are not initialized")
        batch = int(noisy_physical.shape[0])
        dtype = noisy_physical.dtype
        device = noisy_physical.device
        time_emb = self.time(time.to(dtype=dtype))
        primary_cond = self._mmdit_primary_condition(z=z, time_emb=time_emb)
        primary_z_effect = self._mmdit_primary_z_effect(z=z, time_emb=time_emb, primary_cond=primary_cond)
        if self.mmdit_noisy_norm is not None:
            # V70: volume-normalized x_t evidence + logit-domain t-gate.  The
            # lift output is LayerNormed to market-standard volume; the gate
            # becomes an additive log g(t) bias on the noisy attention logits.
            noisy_tokens = self.mmdit_noisy_norm(self.noisy_action_lift(noisy_physical))
            gate = self._noisy_time_gate(time)
            if gate is None:
                noisy_logit_bias = None
                noisy_gate_mean = torch.ones((), device=device, dtype=torch.float32)
            else:
                noisy_logit_bias = gate.reshape(int(gate.shape[0])).float().clamp_min(1e-6).log()
                noisy_gate_mean = gate.detach().float().mean()
            noisy_token_norm = noisy_tokens.detach().float().norm(dim=-1).mean()
        else:
            noisy_tokens, noisy_gate_mean, noisy_token_norm = self._gated_noisy_branch(noisy_physical, time)
            noisy_logit_bias = None
        z_token = self.z_to_token(z.to(device=device, dtype=dtype))
        action = self.horizon_query.to(device=device, dtype=dtype).expand(batch, -1, -1)
        progress_tokens = self._mmdit_progress_tokens(batch=batch, cond_time=primary_cond, z=z)
        if self.evidence_workspace is None:
            raise RuntimeError("MMDiT evidence workspace is not initialized")
        workspace_sources = dict(evidence_sources or {})
        if rollout_tokens is not None:
            workspace_sources["rollout"] = rollout_tokens
        progress_query_context = torch.zeros(batch, self.hidden_size, device=device, dtype=dtype)
        progress_as_value = bool(int(getattr(self.config, "latent_cvae_workspace_progress_value", 1)))
        if progress_tokens is not None and progress_as_value:
            workspace_sources["progress"] = progress_tokens
        elif progress_tokens is not None:
            progress_query_context = progress_tokens.to(device=device, dtype=dtype).mean(dim=1)
        workspace_query, workspace_query_scale = self._workspace_query_action(action, noisy_tokens)
        workspace_tokens, workspace_metrics = self.evidence_workspace(
            workspace_sources,
            action=workspace_query,
            primary_cond=primary_cond,
            step_context=progress_query_context,
        )
        workspace_metrics["workspace_noisy_query_scale"] = workspace_query_scale
        workspace_metrics["workspace_progress_query_norm"] = progress_query_context.detach().float().norm(dim=-1).mean()
        cond_tokens, layout, cond_token_norm = self._mmdit_condition_tokens(
            noisy_tokens=noisy_tokens,
            trajectory_tokens=trajectory_tokens,
            rollout_tokens=rollout_tokens,
            cond_time=primary_cond,
            z_token=z_token,
            layer_stack=layer_stack,
            progress_tokens=progress_tokens,
            workspace_tokens=workspace_tokens,
        )
        action_updates: list[Tensor] = []
        cond_updates: list[Tensor] = []
        cond_attn_rows: list[Tensor] = []
        noisy_attn_rows: list[Tensor] = []
        rollout_attn_rows: list[Tensor] = []
        rollout_enrichment_rows: list[Tensor] = []
        noisy_attn_sample_rows: list[Tensor] = []
        workspace_attn_sample_rows: list[Tensor] = []
        update_condition = bool(int(getattr(self.config, "latent_cvae_mmdit_cond_update", 0)))
        for block in self.mmdit_blocks:
            action, cond_tokens, metrics = block(
                action,
                cond_tokens,
                primary_cond,
                noisy_start=layout.noisy_start,
                noisy_len=layout.noisy_len,
                rollout_start=layout.rollout_start,
                rollout_len=layout.rollout_len,
                low_start=layout.low_start,
                low_len=layout.low_len,
                stage_start=layout.stage_start,
                stage_len=layout.stage_len,
                update_condition=update_condition,
                noisy_logit_bias=noisy_logit_bias,
            )
            action_updates.append(metrics["action_update_norm"].to(device=device))
            cond_updates.append(metrics["cond_update_norm"].to(device=device))
            cond_attn_rows.append(metrics["action_cond_attn"].to(device=device))
            noisy_attn_rows.append(metrics["action_noisy_attn"].to(device=device))
            rollout_attn_rows.append(metrics["action_workspace_attn"].to(device=device))
            rollout_enrichment_rows.append(metrics["action_workspace_enrichment"].to(device=device))
            noisy_attn_sample_rows.append(metrics["action_noisy_attn_rows"].to(device=device))
            workspace_attn_sample_rows.append(metrics["action_workspace_attn_rows"].to(device=device))
        action = self.mmdit_action_norm(action)
        out = self._emit_action(action, primary_cond)
        z0 = torch.zeros((), device=device, dtype=torch.float32)
        action_update = torch.stack(action_updates).mean() if action_updates else z0
        cond_update = torch.stack(cond_updates).mean() if cond_updates else z0
        cond_attn = torch.stack(cond_attn_rows).mean() if cond_attn_rows else z0
        noisy_attn = torch.stack(noisy_attn_rows).mean() if noisy_attn_rows else z0
        rollout_attn = torch.stack(rollout_attn_rows).mean() if rollout_attn_rows else z0
        rollout_enrichment = torch.stack(rollout_enrichment_rows).mean() if rollout_enrichment_rows else z0
        action_norm = action.detach().float().norm(dim=-1).mean()
        noisy_ratio = noisy_token_norm / action_norm.clamp_min(1e-6)
        workspace_rollout = workspace_metrics.get("workspace_rollout_attention", z0)
        workspace_enrichment = workspace_rollout * workspace_metrics["workspace_source_count"].clamp_min(1.0)
        out.update({
            "adaptive_noisy_gate_mean": noisy_gate_mean.to(device=device),
            "adaptive_noisy_branch_norm": noisy_token_norm.to(device=device),
            "adaptive_noisy_branch_ratio": noisy_ratio.to(device=device),
            "mmdit_action_update_norm": action_update,
            "mmdit_cond_update_norm": cond_update,
            "mmdit_action_cond_attention": cond_attn,
            "mmdit_action_noisy_attention": noisy_attn,
            "mmdit_action_workspace_attention": rollout_attn,
            "mmdit_action_workspace_enrichment": rollout_enrichment,
            "mmdit_action_rollout_attention": workspace_rollout,
            "mmdit_action_rollout_enrichment": workspace_enrichment,
            "mmdit_action_token_norm": action_norm,
            "mmdit_condition_token_norm": cond_token_norm.to(device=device),
            "mmdit_noisy_token_norm": noisy_token_norm.to(device=device),
            "primary_condition_norm": primary_cond.detach().float().norm(dim=-1).mean(),
            "primary_z_effect_norm": primary_z_effect,
            **self._time_stratified_attention(
                time,
                torch.stack(noisy_attn_sample_rows).mean(dim=0) if noisy_attn_sample_rows else torch.zeros(batch, device=device, dtype=torch.float32),
                torch.stack(workspace_attn_sample_rows).mean(dim=0) if workspace_attn_sample_rows else torch.zeros(batch, device=device, dtype=torch.float32),
            ),
            **workspace_metrics,
        })
        return out

    def _decode_with_z(
        self,
        *,
        noisy_physical: Tensor,
        time: Tensor,
        trajectory_tokens: Tensor,
        rollout_tokens: Tensor | None,
        cond: Tensor,
        z: Tensor,
        layer_stack: Tensor | None = None,
        evidence_sources: dict[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        if int(getattr(self.config, "latent_cvae_mmdit_decoder", 0)):
            return self._decode_with_z_mmdit(
                noisy_physical=noisy_physical,
                time=time,
                trajectory_tokens=trajectory_tokens,
                rollout_tokens=rollout_tokens,
                cond=cond,
                z=z,
                layer_stack=layer_stack,
                evidence_sources=evidence_sources,
            )
        del layer_stack, rollout_tokens, evidence_sources
        batch = int(noisy_physical.shape[0])
        dtype = noisy_physical.dtype
        device = noisy_physical.device
        time_emb = self.time(time.to(dtype=dtype))
        cond_time = cond + self.time_lift(time_emb)
        noisy_branch, noisy_gate_mean, noisy_branch_norm = self._gated_noisy_branch(noisy_physical, time)
        action = (
            self.horizon_query.to(device=device, dtype=dtype).expand(batch, -1, -1)
            + noisy_branch
            + self.trajectory_lift(trajectory_tokens)
            + self.z_to_token(z.to(dtype=dtype))[:, None]
            + cond_time[:, None]
        )
        noisy_branch_ratio = noisy_branch_norm / action.detach().float().norm(dim=-1).mean().clamp_min(1e-6)
        if not hierarchical_refine:
            for block in self.blocks:
                action = block(action, cond_time)
        out = self._emit_action(action, cond)
        out.update({
            "adaptive_noisy_gate_mean": noisy_gate_mean.to(device=device),
            "adaptive_noisy_branch_norm": noisy_branch_norm.to(device=device),
            "adaptive_noisy_branch_ratio": noisy_branch_ratio.to(device=device),
        })
        return out

    def _condition(
        self,
        *,
        trajectory_tokens: Tensor,
        trajectory_workspace_tokens: Tensor | None,
        context_memory: Tensor | list[Tensor] | tuple[Tensor, ...] | None,
        transition_memory: Tensor | list[Tensor] | tuple[Tensor, ...] | None,
        visual_memory: Tensor | list[Tensor] | tuple[Tensor, ...] | None,
        layer_contracts: list[dict[str, Tensor]],
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        cfg = self.config
        dtype = trajectory_tokens.dtype
        device = trajectory_tokens.device
        batch = int(trajectory_tokens.shape[0])
        detach_layers = bool(int(getattr(cfg, "latent_cvae_layer_detach", 1)))
        use_layer_memory = bool(int(getattr(cfg, "latent_cvae_layer_memory", 1)))
        summaries: list[Tensor] = []
        summary_stats: list[dict[str, Tensor]] = []
        if use_layer_memory:
            for entry in layer_contracts:
                summary, entry_stats = self._layer_entry_summary(entry, detach=detach_layers)
                if summary is not None:
                    summaries.append(summary.to(device=device, dtype=dtype))
                    summary_stats.append(entry_stats)
        if use_layer_memory and len(summaries) < int(cfg.depth):
            raise RuntimeError(f"{str(getattr(cfg, 'final_action_decoder', 'latent_cvae_action'))} expected summaries for {int(cfg.depth)} layers, got {len(summaries)}")
        if use_layer_memory and summaries:
            projected = []
            for i in range(int(cfg.depth)):
                src = summaries[min(i, len(summaries) - 1)]
                projected.append(self._normalize_condition_source(self.layer_proj[i](src)))
            layer_stack = torch.stack(projected, dim=1) + self.layer_embed.to(device=device, dtype=dtype)
        else:
            layer_stack = torch.zeros(batch, int(cfg.depth), self.hidden_size, device=device, dtype=dtype)
        layer_flat = layer_stack.reshape(batch, int(cfg.depth) * self.hidden_size)
        traj = self._normalize_condition_source(self.traj_summary_proj(trajectory_tokens.mean(dim=1)))
        transition_source = self._maybe_detach_memory(
            transition_memory,
            detach=bool(int(getattr(cfg, "latent_cvae_transition_detach", 1))),
        )
        transition_tokens = self._memory_tokens(transition_source, traj, self.transition_proj) if int(getattr(cfg, "latent_cvae_transition_memory", 1)) else traj.new_zeros(batch, 0, self.hidden_size)
        context_tokens = self._memory_tokens(context_memory, traj, self.context_proj) if int(getattr(cfg, "latent_cvae_context_memory", 0)) else traj.new_zeros(batch, 0, self.hidden_size)
        visual_tokens = self._memory_tokens(visual_memory, traj, self.visual_proj) if int(getattr(cfg, "latent_cvae_visual_memory", 0)) else traj.new_zeros(batch, 0, self.hidden_size)
        trans = transition_tokens.mean(dim=1) if int(transition_tokens.shape[1]) > 0 else torch.zeros_like(traj)
        ctx = context_tokens.mean(dim=1) if int(context_tokens.shape[1]) > 0 else torch.zeros_like(traj)
        vis = visual_tokens.mean(dim=1) if int(visual_tokens.shape[1]) > 0 else torch.zeros_like(traj)
        transition_raw_norm = trans.detach().float().norm(dim=-1).mean()
        trans = self._normalize_condition_source(trans)
        ctx = self._normalize_condition_source(ctx)
        vis = self._normalize_condition_source(vis)
        lateral_cond = self.condition_fusion(torch.cat([layer_flat, trans, ctx, vis, traj], dim=-1))
        zero_stat = torch.zeros((), device=device, dtype=torch.float32)
        scale_stats = [item["consequence_scale_mean"].to(device=device) for item in summary_stats]
        preference_stats = [item["consequence_gate_preference"].to(device=device) for item in summary_stats]
        mix_stats = [item["consequence_mix_ratio"].to(device=device) for item in summary_stats]
        cond_stats = {
            "cvae_condition_scan_norm": zero_stat,
            "cvae_condition_lateral_norm": lateral_cond.detach().float().norm(dim=-1).mean(),
            "cvae_layer_summary_norm": (
                torch.stack([value.detach().float().norm(dim=-1).mean() for value in summaries]).mean()
                if summaries else zero_stat
            ),
            "cvae_transition_source_raw_norm": transition_raw_norm,
            "cvae_transition_condition_norm": trans.detach().float().norm(dim=-1).mean(),
            "cvae_consequence_scale_mean": torch.stack(scale_stats).mean() if scale_stats else zero_stat,
            "cvae_consequence_gate_preference": torch.stack(preference_stats).mean() if preference_stats else zero_stat,
            "cvae_consequence_mix_ratio": torch.stack(mix_stats).mean() if mix_stats else zero_stat,
        }
        if (
            int(getattr(cfg, "latent_cvae_layer_scan", 0))
            and use_layer_memory
            and self.layer_scan is not None
            and self.layer_scan_init is not None
            and self.layer_scan_fusion is not None
        ):
            state = self.layer_scan_init.to(device=device, dtype=layer_stack.dtype).expand(batch, -1)
            for i in range(int(layer_stack.shape[1])):
                state = self.layer_scan(layer_stack[:, i], state)
            scan_cond = self.layer_scan_fusion(torch.cat([state.to(dtype=dtype), trans, ctx, vis, traj], dim=-1))
            alpha = float(getattr(cfg, "latent_cvae_layer_scan_alpha", 0.2))
            raw_cond = scan_cond + alpha * lateral_cond
            cond = self.condition_contract_norm(scan_cond) + alpha * self.condition_contract_norm(lateral_cond)
            cond_stats["cvae_condition_scan_norm"] = scan_cond.detach().float().norm(dim=-1).mean()
        else:
            scan_cond = None
            raw_cond = lateral_cond
            cond = self.condition_contract_norm(lateral_cond)
        cond_stats["cvae_condition_raw_norm"] = raw_cond.detach().float().norm(dim=-1).mean()
        cond = self.condition_contract_norm(cond)
        evidence_sources: dict[str, Tensor] = {}
        if int(getattr(cfg, "latent_cvae_workspace_global_sources", 1)):
            evidence_sources["lateral"] = lateral_cond[:, None]
        if int(getattr(cfg, "latent_cvae_workspace_trajectory_source", 1)):
            evidence_sources["trajectory"] = (
                trajectory_tokens
                if trajectory_workspace_tokens is None
                else trajectory_workspace_tokens.to(device=device, dtype=dtype)
            )
        if (
            use_layer_memory
            and summaries
            and int(getattr(cfg, "latent_cvae_workspace_layer_source", 1))
        ):
            evidence_sources["layer"] = layer_stack
        fixed_zero_base = str(getattr(cfg, "controlled_base_mode", "learned")) == "fixed_zero"
        if fixed_zero_base and int(transition_tokens.shape[1]) >= 2:
            # The identifiable rollout has no separate effect token because
            # effect == delta. Keep the remaining two sources semantically
            # explicit instead of treating their mean as an anonymous memory.
            evidence_sources["transition_delta"] = transition_tokens[:, 0:1]
            evidence_sources["transition_timeline"] = transition_tokens[:, 1:2]
            if int(transition_tokens.shape[1]) > 2:
                evidence_sources["transition"] = transition_tokens[:, 2:]
        elif int(transition_tokens.shape[1]) >= 3:
            evidence_sources["transition_delta"] = transition_tokens[:, 0:1]
            evidence_sources["transition_effect"] = transition_tokens[:, 1:2]
            evidence_sources["transition_timeline"] = transition_tokens[:, 2:3]
            if int(transition_tokens.shape[1]) > 3:
                evidence_sources["transition"] = transition_tokens[:, 3:]
        elif int(transition_tokens.shape[1]) > 0:
            evidence_sources["transition"] = transition_tokens
        if scan_cond is not None and int(getattr(cfg, "latent_cvae_workspace_global_sources", 1)):
            evidence_sources["scan"] = scan_cond[:, None]
        if int(getattr(cfg, "latent_cvae_context_memory", 0)):
            evidence_sources["context"] = context_tokens
        if int(getattr(cfg, "latent_cvae_visual_memory", 0)):
            evidence_sources["visual"] = visual_tokens
        layer_count = torch.tensor(float(len(summaries)), device=device, dtype=dtype)
        return cond, layer_count, layer_stack, evidence_sources, cond_stats

    def forward(
        self,
        *,
        noisy_physical: Tensor,
        time: Tensor,
        trajectory_tokens: Tensor,
        rollout_tokens: Tensor | None,
        context_memory: Tensor | list[Tensor] | tuple[Tensor, ...] | None,
        transition_memory: Tensor | list[Tensor] | tuple[Tensor, ...] | None,
        visual_memory: Tensor | list[Tensor] | tuple[Tensor, ...] | None,
        layer_contracts: list[dict[str, Tensor]],
        trajectory_workspace_tokens: Tensor | None = None,
        target_physical: Tensor | None = None,
    ) -> dict[str, Tensor]:
        cfg = self.config
        dtype = noisy_physical.dtype
        device = noisy_physical.device
        cond, layer_count, layer_stack, evidence_sources, cond_stats = self._condition(
            trajectory_tokens=trajectory_tokens,
            trajectory_workspace_tokens=trajectory_workspace_tokens,
            context_memory=context_memory,
            transition_memory=transition_memory,
            visual_memory=visual_memory,
            layer_contracts=layer_contracts,
        )
        rollout_condition = self._maybe_detach_memory(
            rollout_tokens,
            detach=bool(int(getattr(cfg, "latent_cvae_transition_detach", 1))),
        )
        if rollout_condition is not None and not isinstance(rollout_condition, Tensor):
            raise TypeError("rollout_tokens must be a Tensor or None")
        p_mu, p_logvar = self._split_gaussian(self.prior(cond))

        # V42.1: the deploy/inference prior path is always the main output and
        # therefore receives the normal policy losses through pred_velocity.
        # The posterior path is computed only as an auxiliary reconstruction
        # target so it cannot learn to hide target-action information in z.
        if int(getattr(cfg, "latent_cvae_inference_sample", 0)) and not self.training:
            prior_z = p_mu + torch.randn_like(p_mu) * torch.exp(0.5 * p_logvar)
        else:
            prior_z = p_mu
        prior_out = self._decode_with_z(
            noisy_physical=noisy_physical,
            time=time,
            trajectory_tokens=trajectory_tokens,
            rollout_tokens=rollout_condition,
            cond=cond,
            z=prior_z,
            layer_stack=layer_stack,
            evidence_sources=evidence_sources,
        )

        # CR0 probe (do_before_v76 §14.2): flag-gated eval-time z interventions.
        # Two counterfactual prior decodes measure how much the deployed
        # velocity actually depends on z: zero (channel removed) and batch
        # shuffle (wrong sample's z).  Uses a dedicated CPU generator so the
        # global RNG stream -- and therefore paired-seed comparability -- is
        # untouched.  Costs two extra decodes per eval batch; keep the flag
        # off for training arms and enable it only in short diagnostic runs.
        z_zero_delta = torch.zeros((), device=device, dtype=torch.float32)
        z_shuffle_delta = torch.zeros((), device=device, dtype=torch.float32)
        if int(getattr(cfg, "latent_cvae_z_probe", 0)) and not self.training:
            with torch.no_grad():
                probe_kwargs = dict(
                    noisy_physical=noisy_physical,
                    time=time,
                    trajectory_tokens=trajectory_tokens,
                    rollout_tokens=rollout_condition,
                    cond=cond,
                    layer_stack=layer_stack,
                    evidence_sources=evidence_sources,
                )
                zero_out = self._decode_with_z(z=torch.zeros_like(prior_z), **probe_kwargs)
                probe_gen = torch.Generator(device="cpu")
                probe_gen.manual_seed(20260710 + int(prior_z.shape[0]))
                perm = torch.randperm(int(prior_z.shape[0]), generator=probe_gen).to(prior_z.device)
                shuffle_out = self._decode_with_z(z=prior_z[perm], **probe_kwargs)
                reference = prior_out["pred_velocity"].detach().float()
                reference_norm = reference.norm(dim=-1).mean().clamp_min(1e-6)
                z_zero_delta = (
                    (zero_out["pred_velocity"].detach().float() - reference).norm(dim=-1).mean()
                    / reference_norm
                )
                z_shuffle_delta = (
                    (shuffle_out["pred_velocity"].detach().float() - reference).norm(dim=-1).mean()
                    / reference_norm
                )

        # CR1/B1 (do_before_v76 §15): with latent_cvae_variational=0 the
        # posterior/KL/sampling scaffold is bypassed while the deploy function
        # is kept BIT-IDENTICAL (prior_z = mu_p(cond), tanh mu_bound and std
        # clamps untouched -- they are part of the deployed mapping and their
        # removal belongs to B2/B3, not here).  This arm answers exactly one
        # question: does variational TRAINING itself buy reproducible value?
        posterior_used = target_physical is not None and bool(
            int(getattr(cfg, "latent_cvae_variational", 1))
        )
        kl = torch.zeros((), device=device, dtype=dtype)
        post_std = torch.zeros((), device=device, dtype=torch.float32)
        mu_gap = torch.zeros((), device=device, dtype=torch.float32)
        post_z_norm = torch.zeros((), device=device, dtype=torch.float32)
        post_out: dict[str, Tensor] | None = None
        if posterior_used:
            target_physical = target_physical.to(device=device, dtype=dtype)
            target_feat = self.posterior_action(target_physical).mean(dim=1)
            q_mu, q_logvar = self._split_gaussian(self.posterior(torch.cat([cond, target_feat], dim=-1)))
            eps = torch.randn_like(q_mu)
            post_z = q_mu + eps * torch.exp(0.5 * q_logvar)
            post_out = self._decode_with_z(
                noisy_physical=noisy_physical,
                time=time,
                trajectory_tokens=trajectory_tokens,
                rollout_tokens=rollout_condition,
                cond=cond,
                z=post_z,
                layer_stack=layer_stack,
                evidence_sources=evidence_sources,
            )
            kl = self._kl_diag_gaussians(q_mu.float(), q_logvar.float(), p_mu.float(), p_logvar.float()).to(dtype=dtype)
            post_std = torch.exp(0.5 * q_logvar).detach().float().mean()
            mu_gap = (q_mu.detach().float() - p_mu.detach().float()).norm(dim=-1).mean()
            post_z_norm = post_z.detach().float().norm(dim=-1).mean()

        prior_std = torch.exp(0.5 * p_logvar).detach().float().mean()
        result = {
            "pred_velocity": prior_out["pred_velocity"],
            "event_logits": prior_out["event_logits"],
            "motion_logits": prior_out["motion_logits"],
            "action_tokens": prior_out["action_tokens"],
            "transition_latent": prior_out["transition_latent"],
            "cvae_kl": kl,
            "cvae_z_zero_delta": z_zero_delta,
            "cvae_z_shuffle_delta": z_shuffle_delta,
            "cvae_prior_std": prior_std,
            "cvae_post_std": post_std,
            "cvae_z_norm": prior_z.detach().float().norm(dim=-1).mean(),
            "cvae_prior_z_norm": prior_z.detach().float().norm(dim=-1).mean(),
            "cvae_post_z_norm": post_z_norm,
            "cvae_mu_gap": mu_gap,
            "cvae_condition_norm": cond.detach().float().norm(dim=-1).mean(),
            "cvae_condition_raw_norm": cond_stats["cvae_condition_raw_norm"],
            "cvae_condition_scan_norm": cond_stats["cvae_condition_scan_norm"],
            "cvae_condition_lateral_norm": cond_stats["cvae_condition_lateral_norm"],
            "cvae_layer_summary_norm": cond_stats["cvae_layer_summary_norm"],
            "cvae_transition_source_raw_norm": cond_stats["cvae_transition_source_raw_norm"],
            "cvae_transition_condition_norm": cond_stats["cvae_transition_condition_norm"],
            "cvae_rollout_token_norm": (
                torch.zeros((), device=device, dtype=torch.float32)
                if rollout_condition is None
                else rollout_condition.detach().float().norm(dim=-1).mean()
            ),
            "cvae_rollout_token_count": torch.tensor(
                0.0 if rollout_condition is None else float(rollout_condition.shape[1]),
                device=device,
                dtype=torch.float32,
            ),
            "cvae_consequence_scale_mean": cond_stats["cvae_consequence_scale_mean"],
            "cvae_consequence_gate_preference": cond_stats["cvae_consequence_gate_preference"],
            "cvae_consequence_mix_ratio": cond_stats["cvae_consequence_mix_ratio"],
            "cvae_posterior_used": torch.tensor(float(posterior_used), device=device, dtype=dtype),
            "gripper_gate_mean": prior_out["gripper_gate_mean"],
            "layer_memory_count": layer_count,
            "cvae_prior_pred_norm": prior_out["pred_velocity"].detach().float().norm(dim=-1).mean(),
        }
        for key in (
            "adaptive_refine_update_mean",
            "adaptive_noisy_gate_mean",
            "adaptive_noisy_branch_norm",
            "adaptive_noisy_branch_ratio",
            "adaptive_route_entropy",
            "adaptive_route_max",
            "adaptive_route_effective_slots",
            "adaptive_progress_entropy",
            "adaptive_progress_max",
            "adaptive_progress_effective_slots",
            "adaptive_progress_norm",
            "adaptive_continue_mean",
            "adaptive_prefix_norm",
            "adaptive_progress_seed_entropy",
            "adaptive_progress_seed_max",
            "adaptive_progress_seed_effective_slots",
            "adaptive_progress_seed_norm",
            "adaptive_route_temperature_mean",
            "adaptive_route_time_query_norm",
            "adaptive_semantic_bias_norm",
            "adaptive_output_adapter_norm",
            "adaptive_function_delta_norm",
            "adaptive_base_highfreq_norm",
            "adaptive_refine_step_bias_norm",
            "adaptive_capsule_layer_entropy",
            "adaptive_capsule_layer_max",
            "adaptive_capsule_layer_effective_slots",
            "adaptive_condition_strength_mean",
            "adaptive_condition_strength_std",
            "adaptive_condition_strength_max",
            "adaptive_condition_strength_min",
            "adaptive_condition_residual_norm",
            "adaptive_context_direction_norm",
            "adaptive_micro_step_mean",
            "adaptive_micro_step_std",
            "adaptive_micro_progress_mean",
            "adaptive_micro_kp_mean",
            "adaptive_micro_kd_mean",
            "adaptive_micro_feedforward_norm",
            "adaptive_micro_feedback_norm",
            "adaptive_micro_damping_norm",
            "adaptive_micro_function_norm",
            "adaptive_micro_control_norm",
            "adaptive_micro_update_norm",
            "adaptive_micro_heun_error",
            "adaptive_micro_refine_block_norm",
            "adaptive_micro_controller_norm",
            "adaptive_micro_pred_velocity",
            "adaptive_micro_event_logits",
            "adaptive_micro_supervision_logits",
            "adaptive_regularizer",
            "adaptive_route_entropy_regularizer",
            "mmdit_action_update_norm",
            "mmdit_cond_update_norm",
            "mmdit_action_cond_attention",
            "mmdit_action_noisy_attention",
            "mmdit_action_workspace_attention",
            "mmdit_action_workspace_enrichment",
            "mmdit_action_low_attention",
            "mmdit_action_stage_attention",
            "mmdit_action_low_enrichment",
            "mmdit_action_stage_enrichment",
            "mmdit_action_rollout_attention",
            "mmdit_action_rollout_enrichment",
            "mmdit_action_token_norm",
            "mmdit_condition_token_norm",
            "mmdit_noisy_token_norm",
            "mmdit_noisy_attn_t0_sum",
            "mmdit_noisy_attn_t1_sum",
            "mmdit_noisy_attn_t2_sum",
            "mmdit_workspace_attn_t0_sum",
            "mmdit_workspace_attn_t1_sum",
            "mmdit_workspace_attn_t2_sum",
            "mmdit_low_attn_t0_sum",
            "mmdit_low_attn_t1_sum",
            "mmdit_low_attn_t2_sum",
            "mmdit_stage_attn_t0_sum",
            "mmdit_stage_attn_t1_sum",
            "mmdit_stage_attn_t2_sum",
            "mmdit_attn_t0_count",
            "mmdit_attn_t1_count",
            "mmdit_attn_t2_count",
            "primary_condition_norm",
            "primary_z_effect_norm",
            "workspace_progress_update_norm",
            "workspace_progress_action_dependence",
            "legacy_stem_effect_ratio",
            "workspace_token_count",
            "workspace_token_norm",
            "workspace_update_norm",
            "workspace_global_state_norm",
            "workspace_global_slot_delta_norm",
            "workspace_global_slot_diversity",
            "workspace_source_count",
            "workspace_cached_token_fraction",
            "workspace_attention_entropy",
            "workspace_attention_max",
            "workspace_group_attention_entropy",
            "workspace_group_effective_sources",
            "workspace_attention_mass_error",
            "workspace_action_update_ratio",
            "workspace_noisy_query_scale",
            "workspace_progress_query_norm",
            "workspace_role_geom_attention",
            "workspace_role_transition_attention",
            "workspace_role_event_attention",
            "workspace_role_state_attention",
            "workspace_role_layer_attention",
            "workspace_role_global_attention",
            "workspace_role_geom_token_count",
            "workspace_role_transition_token_count",
            "workspace_role_event_token_count",
            "workspace_role_state_token_count",
            "workspace_role_layer_token_count",
            "workspace_role_global_token_count",
            "workspace_controller_capacity",
            "workspace_controller_delay",
            "workspace_controller_temperature",
            "workspace_controller_role_entropy",
            "workspace_controller_role_max",
            "workspace_controller_query_delta_norm",
            "workspace_controller_workspace_delta_norm",
            "workspace_controller_role_geom_prob",
            "workspace_controller_role_transition_prob",
            "workspace_controller_role_event_prob",
            "workspace_controller_role_state_prob",
            "workspace_controller_role_layer_prob",
            "workspace_controller_role_global_prob",
            "workspace_controller_role_geom_logit",
            "workspace_controller_role_transition_logit",
            "workspace_controller_role_event_logit",
            "workspace_controller_role_state_logit",
            "workspace_controller_role_layer_logit",
            "workspace_controller_role_global_logit",
            "hierarchical_low_token_count",
            "hierarchical_low_token_norm",
            "hierarchical_low_selector_stage_entropy",
            "hierarchical_low_selector_stage_max",
            "hierarchical_low_selector_stage_effective_slots",
            "hierarchical_low_selector_role_norm",
            "hierarchical_low_selector_content_norm",
            "hierarchical_stage_token_count",
            "hierarchical_stage_role_norm",
            "hierarchical_stage_role_diversity",
            "hierarchical_stage_content_norm",
            "hierarchical_stage_content_diversity",
            "hierarchical_stage_role_content_cosine",
            "hierarchical_stage_role_output_norm",
            "hierarchical_stage_content_output_norm",
            "hierarchical_stage_role_output_fraction",
            "hierarchical_stage_update_norm",
            "hierarchical_stage_retain_mean",
            "hierarchical_stage_promote_attention_entropy",
            "hierarchical_stage_promote_attention_max",
            "hierarchical_stage_promoted_norm",
            "hierarchical_stage_promoted_projected_rms",
            "hierarchical_stage_promoted_normalized_rms",
            "hierarchical_stage_promoted_realized_scale",
            "hierarchical_stage_promote_gate_scale_error",
            "hierarchical_stage_promote_scale",
            "hierarchical_manager_stage_attention_entropy",
            "hierarchical_manager_stage_attention_max",
            "hierarchical_manager_role_entropy",
            "hierarchical_manager_role_max",
            "hierarchical_manager_query_shift_norm",
            "hierarchical_manager_promote_gate",
            "hierarchical_manager_low_output_strength",
            "hierarchical_manager_stage_output_strength",
            "hierarchical_manager_role_geom_prob",
            "hierarchical_manager_role_transition_prob",
            "hierarchical_manager_role_event_prob",
            "hierarchical_manager_role_state_prob",
            "hierarchical_manager_role_layer_prob",
            "hierarchical_manager_role_global_prob",
            "workspace_layer_attention",
            "workspace_scan_attention",
            "workspace_lateral_attention",
            "workspace_transition_attention",
            "workspace_transition_delta_attention",
            "workspace_transition_effect_attention",
            "workspace_transition_timeline_attention",
            "workspace_transition_total_attention",
            "workspace_context_attention",
            "workspace_visual_attention",
            "workspace_trajectory_attention",
            "workspace_rollout_attention",
            "workspace_capsule_attention",
            "workspace_progress_attention",
            "workspace_routed_layer_attention",
        ):
            if key in prior_out:
                result[f"cvae_{key}"] = prior_out[key]
        if post_out is not None:
            result.update({
                "post_pred_velocity": post_out["pred_velocity"],
                "post_event_logits": post_out["event_logits"],
                "post_motion_logits": post_out["motion_logits"],
                "post_action_tokens": post_out["action_tokens"],
                "post_transition_latent": post_out["transition_latent"],
                "cvae_post_pred_norm": post_out["pred_velocity"].detach().float().norm(dim=-1).mean(),
                "cvae_post_gripper_gate_mean": post_out["gripper_gate_mean"],
            })
        return result

class AdaptiveRecurrentCVAEActionDecoder(LatentCVAEActionDecoder):
    """CVAE action head with z-primary refinement and typed evidence workspace.

    In the MMDiT path, z and flow time own AdaLN modulation. Layer, transition,
    rollout, trajectory, capsule, and progress evidence first compete inside a
    configurable workspace; each refine step then performs one action update
    from that workspace plus the noisy action field. The legacy recurrent path
    remains available only when the MMDiT decoder is disabled.
    """

    def __init__(self, config: LegacyPolicyConfig) -> None:
        super().__init__(config)
        h = int(config.hidden_size)
        ph = int(config.physical_action_dim)
        action_horizon = int(config.action_horizon)
        self.refine_steps = int(getattr(config, "adaptive_cvae_refine_steps", 3))
        self.progress_steps = int(getattr(config, "adaptive_cvae_progress_steps", 6))
        self.progress_role_dim = int(getattr(config, "adaptive_cvae_progress_role_dim", 16))
        self.context_capsule_count = int(getattr(config, "adaptive_cvae_context_capsule_count", self.progress_steps))
        self.register_buffer("progress_role_basis", _progress_role_basis(self.progress_steps, self.progress_role_dim), persistent=False)
        self.register_buffer("layer_role_basis", _progress_role_basis(int(config.depth), self.progress_role_dim), persistent=False)
        self.register_buffer("refine_step_role_basis", _progress_role_basis(max(self.refine_steps, 1), self.progress_role_dim), persistent=False)
        self.register_buffer("context_capsule_role_basis", _progress_role_basis(self.context_capsule_count, self.progress_role_dim), persistent=False)
        self.register_buffer("progress_slot_position", torch.linspace(0.0, 1.0, self.progress_steps, dtype=torch.float32), persistent=False)
        self.progress_query = nn.Parameter(torch.randn(1, self.progress_steps, h) * 0.02)
        self.context_capsule_query = nn.Parameter(torch.randn(1, self.context_capsule_count, h) * 0.02)
        self.progress_route_query_bias = nn.Parameter(torch.zeros(1, action_horizon, h))
        self.layer_route_query_bias = nn.Parameter(torch.zeros(1, action_horizon, h))
        self.context_route_query_bias = nn.Parameter(torch.zeros(1, action_horizon, h))
        self.progress_role_lift = nn.Sequential(nn.LayerNorm(self.progress_role_dim), nn.Linear(self.progress_role_dim, h))
        self.context_capsule_role_lift = nn.Sequential(nn.LayerNorm(self.progress_role_dim), nn.Linear(self.progress_role_dim, h))
        self.progress_z_lift = nn.Sequential(nn.LayerNorm(self.z_dim), nn.Linear(self.z_dim, h))
        self.progress_block = LatentCVAEActionBlock(config)
        self.progress_contract_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.workspace_progress_update = nn.Sequential(
            nn.LayerNorm(4 * h),
            nn.Linear(4 * h, h),
            nn.SiLU(),
            nn.Linear(h, h),
        )
        self.context_capsule_block = LatentCVAEActionBlock(config)
        self.progress_action_query = nn.Linear(h, h, bias=False)
        self.progress_key = nn.Linear(h, h, bias=False)
        self.progress_value = nn.Linear(h, h, bias=False)
        self.progress_role_key = nn.Linear(self.progress_role_dim, h, bias=False)
        self.progress_role_value = nn.Linear(self.progress_role_dim, h, bias=False)
        self.action_role_query = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, self.progress_role_dim))
        self.progress_role_logit_scale = nn.Parameter(torch.tensor(1.0))
        self.layer_role_key = nn.Linear(self.progress_role_dim, h, bias=False)
        self.layer_role_logit_scale = nn.Parameter(torch.tensor(0.5))
        self.route_temperature = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 1))
        self.refine_step_role_lift = nn.Sequential(nn.LayerNorm(self.progress_role_dim), nn.Linear(self.progress_role_dim, h))
        self.progress_seed_adapter = nn.Sequential(nn.LayerNorm(2 * h), nn.Linear(2 * h, h), nn.SiLU(), nn.Linear(h, h))
        self.seed_function_bank = AdaptiveCVAEFunctionBank(config)
        self.prefix_lift = nn.Sequential(nn.LayerNorm(2 * ph), nn.Linear(2 * ph, h), nn.SiLU(), nn.Linear(h, h))
        self.route_query = nn.Linear(h, h, bias=False)
        self.route_key = nn.Linear(h, h, bias=False)
        self.route_value = nn.Linear(h, h, bias=False)
        self.route_time_query = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
        self.context_layer_query = nn.Linear(h, h, bias=False)
        self.context_layer_key = nn.Linear(h, h, bias=False)
        self.context_layer_value = nn.Linear(h, h, bias=False)
        self.context_layer_role_key = nn.Linear(self.progress_role_dim, h, bias=False)
        self.context_layer_role_logit_scale = nn.Parameter(torch.tensor(0.5))
        self.context_route_query = nn.Linear(h, h, bias=False)
        self.context_route_key = nn.Linear(h, h, bias=False)
        self.context_route_value = nn.Linear(h, h, bias=False)
        self.context_route_role_key = nn.Linear(self.progress_role_dim, h, bias=False)
        self.context_route_role_value = nn.Linear(self.progress_role_dim, h, bias=False)
        self.context_role_logit_scale = nn.Parameter(torch.tensor(1.0))
        self.context_direction_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.context_residual_adapter = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h))
        self.condition_strength_head = nn.Sequential(nn.LayerNorm(5 * h), nn.Linear(5 * h, h), nn.SiLU(), nn.Linear(h, 1))
        self.micro_progress_init = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 1))
        self.micro_gain_head = nn.Sequential(nn.LayerNorm(5 * h), nn.Linear(5 * h, h), nn.SiLU(), nn.Linear(h, 3))
        self.micro_reference = nn.Sequential(nn.LayerNorm(4 * h), nn.Linear(4 * h, h), nn.SiLU(), nn.Linear(h, h))
        self.micro_feedforward = nn.Sequential(nn.LayerNorm(4 * h), nn.Linear(4 * h, h), nn.SiLU(), nn.Linear(h, h))
        self.micro_context_modulation = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h))
        self.micro_error_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.micro_function_bank = AdaptiveCVAEFunctionBank(config)
        self.micro_refine_block = AdaptiveCVAEMicroRefineBlock(config)
        self.micro_supervision_router = nn.Sequential(nn.LayerNorm(5 * h), nn.Linear(5 * h, h), nn.SiLU(), nn.Linear(h, 1))
        self.token_semantic_adapter = nn.Sequential(nn.LayerNorm(4 * h), nn.Linear(4 * h, h), nn.SiLU(), nn.Linear(h, h))
        self.refine_function_bank = AdaptiveCVAEFunctionBank(config)
        self.output_semantic_adapter = nn.Sequential(nn.LayerNorm(3 * h), nn.Linear(3 * h, h), nn.SiLU(), nn.Linear(h, h))
        self.output_function_bank = AdaptiveCVAEFunctionBank(config)
        self.refine_block = AdaptiveRecurrentCVAERefinementBlock(config)
        self._init_residual(self.progress_seed_adapter, std=float(getattr(config, "latent_cvae_output_init_std", 1e-3)))
        self._init_residual(self.workspace_progress_update, std=float(getattr(config, "latent_cvae_output_init_std", 1e-3)))
        self._init_residual(self.context_residual_adapter, std=float(getattr(config, "latent_cvae_output_init_std", 1e-3)))
        self._init_residual(self.micro_reference, std=float(getattr(config, "latent_cvae_output_init_std", 1e-3)))
        self._init_residual(self.micro_feedforward, std=float(getattr(config, "latent_cvae_output_init_std", 1e-3)))
        self._init_residual(self.micro_context_modulation, std=float(getattr(config, "latent_cvae_output_init_std", 1e-3)))
        self._init_residual(self.micro_supervision_router, std=0.0)
        self._init_residual(self.token_semantic_adapter, std=0.0)
        self._init_residual(self.output_semantic_adapter, std=0.0)
        role_query = self.action_role_query[-1]
        if isinstance(role_query, nn.Linear):
            nn.init.normal_(role_query.weight, mean=0.0, std=0.02)
            nn.init.zeros_(role_query.bias)
        step_lift = self.refine_step_role_lift[-1]
        if isinstance(step_lift, nn.Linear):
            nn.init.normal_(step_lift.weight, mean=0.0, std=0.02)
            nn.init.zeros_(step_lift.bias)
        temp_head = self.route_temperature[-1]
        if isinstance(temp_head, nn.Linear):
            nn.init.zeros_(temp_head.weight)
            nn.init.zeros_(temp_head.bias)
        route_time = self.route_time_query[-1]
        if isinstance(route_time, nn.Linear):
            nn.init.zeros_(route_time.weight)
            nn.init.zeros_(route_time.bias)
        strength_head = self.condition_strength_head[-1]
        if isinstance(strength_head, nn.Linear):
            lo = float(getattr(config, "adaptive_cvae_condition_strength_min", 0.03))
            hi = float(getattr(config, "adaptive_cvae_condition_strength_max", 1.50))
            init = float(getattr(config, "adaptive_cvae_condition_strength_init", 0.35))
            if hi > lo:
                frac = min(max((init - lo) / (hi - lo), 1e-4), 1.0 - 1e-4)
                bias = math.log(frac / (1.0 - frac))
            else:
                bias = 0.0
            nn.init.zeros_(strength_head.weight)
            nn.init.constant_(strength_head.bias, bias)
        progress_init = self.micro_progress_init[-1]
        if isinstance(progress_init, nn.Linear):
            nn.init.zeros_(progress_init.weight)
            nn.init.constant_(progress_init.bias, -2.0)
        gain_head = self.micro_gain_head[-1]
        if isinstance(gain_head, nn.Linear):
            nn.init.zeros_(gain_head.weight)
            step_bias = self._bounded_sigmoid_bias(
                value=float(getattr(config, "adaptive_cvae_micro_step_init", 0.12)),
                lo=float(getattr(config, "adaptive_cvae_micro_min_step", 0.03)),
                hi=float(getattr(config, "adaptive_cvae_micro_max_step", 0.35)),
            )
            kp_bias = self._bounded_sigmoid_bias(
                value=float(getattr(config, "adaptive_cvae_micro_kp_init", 0.18)),
                lo=0.0,
                hi=float(getattr(config, "adaptive_cvae_micro_kp_max", 0.60)),
            )
            kd_bias = self._bounded_sigmoid_bias(
                value=float(getattr(config, "adaptive_cvae_micro_kd_init", 0.08)),
                lo=0.0,
                hi=float(getattr(config, "adaptive_cvae_micro_kd_max", 0.45)),
            )
            with torch.no_grad():
                gain_head.bias.copy_(torch.tensor([step_bias, kp_bias, kd_bias], dtype=gain_head.bias.dtype))

    @staticmethod
    def _init_residual(module: nn.Module, *, std: float) -> None:
        last = module[-1] if isinstance(module, nn.Sequential) else None
        if isinstance(last, nn.Linear):
            if std > 0:
                nn.init.normal_(last.weight, mean=0.0, std=std)
            else:
                nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    @staticmethod
    def _bounded_sigmoid_bias(*, value: float, lo: float, hi: float) -> float:
        if hi <= lo:
            return 0.0
        frac = min(max((float(value) - lo) / (hi - lo), 1e-4), 1.0 - 1e-4)
        return math.log(frac / (1.0 - frac))

    @staticmethod
    def _horizon_bias(param: Tensor, action: Tensor) -> Tensor:
        horizon = int(action.shape[1])
        bias = param.to(device=action.device, dtype=action.dtype)
        if horizon <= int(bias.shape[1]):
            return bias[:, :horizon]
        repeat = math.ceil(horizon / int(bias.shape[1]))
        return bias.repeat(1, repeat, 1)[:, :horizon]

    def _route_time_bias(self, route_cond: Tensor | None, action: Tensor) -> Tensor | None:
        if route_cond is None or not int(getattr(self.config, "adaptive_cvae_route_time_query", 0)):
            return None
        return self.route_time_query(route_cond.to(device=action.device, dtype=action.dtype))[:, None]

    def _coarse_temporal_base(self, action: Tensor) -> Tensor:
        stride = max(int(getattr(self.config, "adaptive_cvae_coarse_stride", 1)), 1)
        strength = min(max(float(getattr(self.config, "adaptive_cvae_coarse_strength", 1.0)), 0.0), 1.0)
        if stride <= 1 or strength >= 1.0:
            return action
        horizon = int(action.shape[1])
        coarse_chunks: list[Tensor] = []
        for start in range(0, horizon, stride):
            end = min(start + stride, horizon)
            pooled = action[:, start:end].mean(dim=1, keepdim=True)
            coarse_chunks.append(pooled.expand(-1, end - start, -1))
        coarse = torch.cat(coarse_chunks, dim=1)
        return coarse + strength * (action - coarse)

    def _context_dropout(self, value: Tensor) -> Tensor:
        p = float(getattr(self.config, "adaptive_cvae_context_dropout", 0.0))
        if p <= 0:
            return value
        return F.dropout(value, p=p, training=self.training)

    @staticmethod
    def _sparsemax(logits: Tensor, dim: int = -1) -> Tensor:
        z = logits.float()
        z = z - z.max(dim=dim, keepdim=True).values
        z_sorted, _ = torch.sort(z, dim=dim, descending=True)
        range_shape = [1] * z.ndim
        range_shape[dim] = int(z.shape[dim])
        support_index = torch.arange(1, int(z.shape[dim]) + 1, device=z.device, dtype=z.dtype).view(range_shape)
        z_cumsum = z_sorted.cumsum(dim)
        support = 1.0 + support_index * z_sorted > z_cumsum
        support_size = support.sum(dim=dim, keepdim=True).clamp_min(1)
        tau_index = support_size.to(dtype=torch.long) - 1
        tau = (z_cumsum.gather(dim, tau_index) - 1.0) / support_size.to(dtype=z.dtype)
        return torch.clamp(z - tau, min=0.0)

    def _adaptive_route_temperature(self, action: Tensor) -> Tensor:
        base = float(getattr(self.config, "adaptive_cvae_route_temperature", 1.0))
        if not int(getattr(self.config, "adaptive_cvae_route_adaptive_temperature", 1)):
            return torch.full((*action.shape[:2], 1), base, device=action.device, dtype=torch.float32)
        lo = float(getattr(self.config, "adaptive_cvae_route_min_temperature", 0.35))
        hi = float(getattr(self.config, "adaptive_cvae_route_max_temperature", 1.25))
        raw = self.route_temperature(action).float()
        temp = lo + (hi - lo) * torch.sigmoid(raw)
        return temp * base

    def _route_weights(self, logits: Tensor, action: Tensor) -> Tensor:
        logits = logits / self._adaptive_route_temperature(action).clamp_min(1e-6)
        slots = int(logits.shape[-1])
        topk = int(getattr(self.config, "adaptive_cvae_route_topk", 0))
        if topk > 0 and topk < slots:
            values, indices = logits.topk(topk, dim=-1)
            masked = torch.full_like(logits, -1e9)
            logits = masked.scatter(-1, indices, values)
        if int(getattr(self.config, "adaptive_cvae_route_sparsemax", 1)):
            return self._sparsemax(logits, dim=-1)
        return torch.softmax(logits.float(), dim=-1)

    def _role_route_logits(self, action: Tensor, role_basis: Tensor, *, scale: Tensor) -> Tensor:
        if not int(getattr(self.config, "adaptive_cvae_role_query", 1)):
            return torch.zeros(*action.shape[:2], int(role_basis.shape[0]), device=action.device, dtype=torch.float32)
        query = self.action_role_query(action).float()
        role = role_basis.to(device=action.device, dtype=action.dtype).float()
        logits = torch.einsum("btd,sd->bts", F.normalize(query, dim=-1), F.normalize(role, dim=-1))
        return logits * scale.float().clamp(0.0, 4.0)

    def _refine_step_bias(self, step: int, action: Tensor) -> Tensor:
        if not int(getattr(self.config, "adaptive_cvae_step_roles", 1)):
            return torch.zeros_like(action)
        index = min(max(int(step), 0), int(self.refine_step_role_basis.shape[0]) - 1)
        role = self.refine_step_role_basis[index].to(device=action.device, dtype=action.dtype)
        bias = self.refine_step_role_lift(role)[None, None]
        return bias.expand(int(action.shape[0]), int(action.shape[1]), -1)

    def _route_entropy_floor(self, entropy: Tensor, slots: int) -> Tensor:
        ratio = float(getattr(self.config, "adaptive_cvae_route_entropy_floor_ratio", 0.0))
        if ratio <= 0 or slots <= 1:
            return torch.zeros((), device=entropy.device, dtype=entropy.dtype)
        floor = math.log(float(slots)) * ratio
        return F.relu(torch.as_tensor(floor, device=entropy.device, dtype=entropy.dtype) - entropy)

    @staticmethod
    def _prefix_features(clean_physical: Tensor) -> Tensor:
        batch, horizon, _ = clean_physical.shape
        prefix_sum = torch.cat([
            torch.zeros(batch, 1, clean_physical.shape[-1], device=clean_physical.device, dtype=clean_physical.dtype),
            torch.cumsum(clean_physical, dim=1)[:, :-1],
        ], dim=1)
        count = torch.arange(horizon, device=clean_physical.device, dtype=clean_physical.dtype).clamp_min(1.0)
        prefix_mean = prefix_sum / count[None, :, None]
        prefix_last = torch.cat([
            torch.zeros(batch, 1, clean_physical.shape[-1], device=clean_physical.device, dtype=clean_physical.dtype),
            clean_physical[:, :-1],
        ], dim=1)
        return torch.cat([prefix_last, prefix_mean], dim=-1)

    def _route_layers(
        self,
        action: Tensor,
        layer_stack: Tensor | None,
        route_cond: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        cfg = self.config
        if (
            layer_stack is None
            or not int(getattr(cfg, "adaptive_cvae_layer_routing", 1))
            or not int(getattr(cfg, "latent_cvae_layer_memory", 1))
        ):
            z = torch.zeros((), device=action.device, dtype=torch.float32)
            return torch.zeros_like(action), z, z
        layer_stack = layer_stack.to(device=action.device, dtype=action.dtype)
        q = self.route_query(action)
        if int(getattr(cfg, "adaptive_cvae_route_query_bias", 1)):
            q = q + self._horizon_bias(self.layer_route_query_bias, action)
        time_bias = self._route_time_bias(route_cond, action)
        if time_bias is not None:
            q = q + time_bias
        layer_role = self.layer_role_basis.to(device=action.device, dtype=action.dtype)
        k = self.route_key(layer_stack) + self.layer_role_key(layer_role)[None]
        v = self.route_value(layer_stack)
        if int(getattr(cfg, "adaptive_cvae_route_cosine", 1)):
            q_route = F.normalize(q.float(), dim=-1)
            k_route = F.normalize(k.float(), dim=-1)
            logits = torch.einsum("bth,blh->btl", q_route, k_route)
        else:
            logits = torch.einsum("bth,blh->btl", q, k).float() * (float(self.hidden_size) ** -0.5)
        logits = logits + self._role_route_logits(action, layer_role, scale=self.layer_role_logit_scale)
        weights = self._route_weights(logits, action).to(dtype=action.dtype)
        routed = torch.einsum("btl,blh->bth", weights, v)
        wf = weights.float().clamp_min(1e-8)
        entropy = -(wf * wf.log()).sum(dim=-1).mean()
        max_weight = wf.detach().max(dim=-1).values.mean()
        return routed, entropy, max_weight

    def _context_capsules(
        self,
        *,
        cond_time: Tensor,
        layer_stack: Tensor | None,
        progress: Tensor | None,
    ) -> tuple[Tensor | None, Tensor, Tensor]:
        cfg = self.config
        z = torch.zeros((), device=cond_time.device, dtype=torch.float32)
        if (
            layer_stack is None
            or not int(getattr(cfg, "adaptive_cvae_context_capsules", 1))
            or not int(getattr(cfg, "adaptive_cvae_layer_routing", 1))
            or not int(getattr(cfg, "latent_cvae_layer_memory", 1))
        ):
            return None, z, z
        batch = int(cond_time.shape[0])
        dtype = cond_time.dtype
        device = cond_time.device
        layer_stack = layer_stack.to(device=device, dtype=dtype)
        capsule_role = self.context_capsule_role_basis.to(device=device, dtype=dtype)
        query = (
            self.context_capsule_query.to(device=device, dtype=dtype).expand(batch, -1, -1)
            + self.context_capsule_role_lift(capsule_role)[None]
            + cond_time[:, None]
        )
        if progress is not None:
            progress = progress.to(device=device, dtype=dtype)
            if int(progress.shape[1]) == int(query.shape[1]):
                query = query + progress
            else:
                progress_role = self.progress_role_basis.to(device=device, dtype=dtype)
                role_logits = torch.einsum(
                    "cd,pd->cp",
                    F.normalize(capsule_role.float(), dim=-1),
                    F.normalize(progress_role.float(), dim=-1),
                )
                role_weights = torch.softmax(role_logits, dim=-1).to(dtype=dtype)
                query = query + torch.einsum("cp,bph->bch", role_weights, progress)
        layer_role = self.layer_role_basis.to(device=device, dtype=dtype)
        q = self.context_layer_query(query)
        k = self.context_layer_key(layer_stack) + self.context_layer_role_key(layer_role)[None]
        v = self.context_layer_value(layer_stack)
        if int(getattr(cfg, "adaptive_cvae_route_cosine", 1)):
            logits = torch.einsum("bch,blh->bcl", F.normalize(q.float(), dim=-1), F.normalize(k.float(), dim=-1))
        else:
            logits = torch.einsum("bch,blh->bcl", q, k).float() * (float(self.hidden_size) ** -0.5)
        role_logits = torch.einsum(
            "cd,ld->cl",
            F.normalize(capsule_role.float(), dim=-1),
            F.normalize(layer_role.float(), dim=-1),
        )
        logits = logits + role_logits[None] * self.context_layer_role_logit_scale.float().clamp(0.0, 4.0)
        weights = self._route_weights(logits, query).to(dtype=dtype)
        capsules = torch.einsum("bcl,blh->bch", weights, v) + self.context_capsule_role_lift(capsule_role)[None]
        capsules = self.context_capsule_block(capsules, cond_time)
        wf = weights.float().clamp_min(1e-8)
        entropy = -(wf * wf.log()).sum(dim=-1).mean()
        max_weight = wf.detach().max(dim=-1).values.mean()
        return capsules, entropy, max_weight

    def _route_context_capsules(
        self,
        action: Tensor,
        capsules: Tensor | None,
        route_cond: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        if capsules is None:
            z = torch.zeros((), device=action.device, dtype=torch.float32)
            return torch.zeros_like(action), z, z, None
        cfg = self.config
        capsules = capsules.to(device=action.device, dtype=action.dtype)
        role = self.context_capsule_role_basis.to(device=action.device, dtype=action.dtype)
        q = self.context_route_query(action)
        if int(getattr(cfg, "adaptive_cvae_route_query_bias", 1)):
            q = q + self._horizon_bias(self.context_route_query_bias, action)
        time_bias = self._route_time_bias(route_cond, action)
        if time_bias is not None:
            q = q + time_bias
        k = self.context_route_key(capsules) + self.context_route_role_key(role)[None]
        v = self.context_route_value(capsules) + self.context_route_role_value(role)[None]
        if int(getattr(cfg, "adaptive_cvae_route_cosine", 1)):
            logits = torch.einsum("bth,bch->btc", F.normalize(q.float(), dim=-1), F.normalize(k.float(), dim=-1))
        else:
            logits = torch.einsum("bth,bch->btc", q, k).float() * (float(self.hidden_size) ** -0.5)
        logits = logits + self._role_route_logits(action, role, scale=self.context_role_logit_scale)
        weights = self._route_weights(logits, action).to(dtype=action.dtype)
        routed = torch.einsum("btc,bch->bth", weights, v)
        wf = weights.float().clamp_min(1e-8)
        entropy = -(wf * wf.log()).sum(dim=-1).mean()
        max_weight = wf.detach().max(dim=-1).values.mean()
        return routed, entropy, max_weight, weights

    def _semantic_context_residual(
        self,
        *,
        action: Tensor,
        cond_time: Tensor,
        context: Tensor,
        progress_context: Tensor,
        step_bias: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        context_dir = self.context_direction_norm(context)
        if not int(getattr(self.config, "adaptive_cvae_direct_condition_residual", 0)):
            strength = torch.zeros(*action.shape[:2], 1, device=action.device, dtype=action.dtype)
            return torch.zeros_like(action), strength, context_dir
        residual = self.context_residual_adapter(context_dir)
        if not int(getattr(self.config, "adaptive_cvae_condition_strength", 1)):
            strength = torch.ones(*action.shape[:2], 1, device=action.device, dtype=action.dtype)
            return residual, strength, context_dir
        cond_tokens = cond_time[:, None].expand(-1, int(action.shape[1]), -1)
        strength_input = torch.cat([action, cond_tokens, context_dir, progress_context, step_bias], dim=-1)
        raw = self.condition_strength_head(strength_input).float()
        lo = float(getattr(self.config, "adaptive_cvae_condition_strength_min", 0.03))
        hi = float(getattr(self.config, "adaptive_cvae_condition_strength_max", 1.50))
        if hi <= lo:
            strength = torch.full_like(raw, lo)
        else:
            strength = lo + (hi - lo) * torch.sigmoid(raw)
        strength = strength.to(device=action.device, dtype=action.dtype)
        return residual * strength, strength, context_dir

    def _latent_progress(self, *, batch: int, cond_time: Tensor, z: Tensor) -> Tensor | None:
        cfg = self.config
        if not int(getattr(cfg, "adaptive_cvae_progress_memory", 1)):
            return None
        dtype = cond_time.dtype
        device = cond_time.device
        role = self.progress_role_basis.to(device=device, dtype=dtype)
        progress = (
            self.progress_query.to(device=device, dtype=dtype).expand(batch, -1, -1)
            + self.progress_role_lift(role)[None]
            + cond_time[:, None]
        )
        if not int(getattr(cfg, "latent_cvae_mmdit_decoder", 0)):
            progress = progress + self.progress_z_lift(z.to(device=device, dtype=dtype))[:, None]
        return self.progress_contract_norm(self.progress_block(progress, cond_time))

    def _route_progress_full(
        self,
        action: Tensor,
        progress: Tensor | None,
        route_cond: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        if progress is None:
            z = torch.zeros((), device=action.device, dtype=torch.float32)
            return torch.zeros_like(action), z, z, None
        cfg = self.config
        progress = progress.to(device=action.device, dtype=action.dtype)
        q = self.progress_action_query(action)
        if int(getattr(cfg, "adaptive_cvae_route_query_bias", 1)):
            q = q + self._horizon_bias(self.progress_route_query_bias, action)
        time_bias = self._route_time_bias(route_cond, action)
        if time_bias is not None:
            q = q + time_bias
        role = self.progress_role_basis.to(device=action.device, dtype=action.dtype)
        k = self.progress_key(progress) + self.progress_role_key(role)[None]
        v = self.progress_value(progress) + self.progress_role_value(role)[None]
        if int(getattr(cfg, "adaptive_cvae_route_cosine", 1)):
            logits = torch.einsum("bth,bsh->bts", F.normalize(q.float(), dim=-1), F.normalize(k.float(), dim=-1))
        else:
            logits = torch.einsum("bth,bsh->bts", q, k).float() * (float(self.hidden_size) ** -0.5)
        logits = logits + self._role_route_logits(action, role, scale=self.progress_role_logit_scale)
        weights = self._route_weights(logits, action).to(dtype=action.dtype)
        routed = torch.einsum("bts,bsh->bth", weights, v)
        wf = weights.float().clamp_min(1e-8)
        entropy = -(wf * wf.log()).sum(dim=-1).mean()
        max_weight = wf.detach().max(dim=-1).values.mean()
        return routed, entropy, max_weight, weights

    def _micro_initial_progress(self, action: Tensor) -> Tensor:
        return torch.sigmoid(self.micro_progress_init(action).float()).squeeze(-1)

    def _route_progress_monotonic(
        self,
        action: Tensor,
        progress: Tensor | None,
        progress_center: Tensor | None,
        route_cond: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        if (
            progress is None
            or progress_center is None
            or not int(getattr(self.config, "adaptive_cvae_micro_monotonic_progress", 1))
        ):
            return self._route_progress_full(action, progress, route_cond=route_cond)
        cfg = self.config
        progress = progress.to(device=action.device, dtype=action.dtype)
        q = self.progress_action_query(action)
        if int(getattr(cfg, "adaptive_cvae_route_query_bias", 1)):
            q = q + self._horizon_bias(self.progress_route_query_bias, action)
        time_bias = self._route_time_bias(route_cond, action)
        if time_bias is not None:
            q = q + time_bias
        role = self.progress_role_basis.to(device=action.device, dtype=action.dtype)
        k = self.progress_key(progress) + self.progress_role_key(role)[None]
        v = self.progress_value(progress) + self.progress_role_value(role)[None]
        if int(getattr(cfg, "adaptive_cvae_route_cosine", 1)):
            logits = torch.einsum("bth,bsh->bts", F.normalize(q.float(), dim=-1), F.normalize(k.float(), dim=-1))
        else:
            logits = torch.einsum("bth,bsh->bts", q, k).float() * (float(self.hidden_size) ** -0.5)
        logits = logits + self._role_route_logits(action, role, scale=self.progress_role_logit_scale)
        position = self.progress_slot_position.to(device=action.device, dtype=torch.float32)
        distance = (progress_center.float().unsqueeze(-1) - position[None, None]).square()
        logits = logits - float(getattr(cfg, "adaptive_cvae_micro_progress_distance_scale", 4.0)) * distance
        weights = self._route_weights(logits, action).to(dtype=action.dtype)
        routed = torch.einsum("bts,bsh->bth", weights, v)
        wf = weights.float().clamp_min(1e-8)
        entropy = -(wf * wf.log()).sum(dim=-1).mean()
        max_weight = wf.detach().max(dim=-1).values.mean()
        return routed, entropy, max_weight, weights

    @staticmethod
    def _bounded_sigmoid(raw: Tensor, *, lo: float, hi: float) -> Tensor:
        if hi <= lo:
            return torch.full_like(raw, float(lo))
        return float(lo) + (float(hi) - float(lo)) * torch.sigmoid(raw.float())

    def _micro_gains(
        self,
        *,
        action: Tensor,
        cond_time: Tensor,
        progress_context: Tensor,
        context_dir: Tensor,
        step_bias: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        cond_tokens = cond_time[:, None].expand(-1, int(action.shape[1]), -1)
        raw = self.micro_gain_head(torch.cat([action, cond_tokens, progress_context, context_dir, step_bias], dim=-1)).float()
        raw_step, raw_kp, raw_kd = raw.split(1, dim=-1)
        ds = self._bounded_sigmoid(
            raw_step,
            lo=float(getattr(self.config, "adaptive_cvae_micro_min_step", 0.03)),
            hi=float(getattr(self.config, "adaptive_cvae_micro_max_step", 0.35)),
        ).to(device=action.device, dtype=action.dtype)
        kp = self._bounded_sigmoid(
            raw_kp,
            lo=0.0,
            hi=float(getattr(self.config, "adaptive_cvae_micro_kp_max", 0.60)),
        ).to(device=action.device, dtype=action.dtype)
        kd = self._bounded_sigmoid(
            raw_kd,
            lo=0.0,
            hi=float(getattr(self.config, "adaptive_cvae_micro_kd_max", 0.45)),
        ).to(device=action.device, dtype=action.dtype)
        return ds, kp, kd

    def _micro_control_field(
        self,
        *,
        action: Tensor,
        prev_velocity: Tensor,
        cond_time: Tensor,
        progress_context: Tensor,
        context_dir: Tensor,
        step_bias: Tensor,
        progress_weights: Tensor | None,
        kp: Tensor,
        kd: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        cond_tokens = cond_time[:, None].expand(-1, int(action.shape[1]), -1)
        reference_delta = self.micro_reference(torch.cat([cond_tokens, progress_context, context_dir, step_bias], dim=-1))
        error = torch.tanh(reference_delta)
        feedback = kp * error
        damping = kd * prev_velocity
        feedforward = self.micro_feedforward(torch.cat([action, progress_context, context_dir, step_bias], dim=-1))
        function = self._function_delta(self.micro_function_bank, action + progress_context + context_dir, progress_weights)
        control = feedforward + function + feedback - damping
        return control, {
            "reference": reference_delta,
            "feedforward": feedforward,
            "feedback": feedback,
            "damping": damping,
            "function": function,
            "control": control,
        }

    def _micro_integrate(
        self,
        *,
        action: Tensor,
        prev_velocity: Tensor,
        cond_time: Tensor,
        progress_context: Tensor,
        context_dir: Tensor,
        step_bias: Tensor,
        progress_weights: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        ds, kp, kd = self._micro_gains(
            action=action,
            cond_time=cond_time,
            progress_context=progress_context,
            context_dir=context_dir,
            step_bias=step_bias,
        )
        control1, terms1 = self._micro_control_field(
            action=action,
            prev_velocity=prev_velocity,
            cond_time=cond_time,
            progress_context=progress_context,
            context_dir=context_dir,
            step_bias=step_bias,
            progress_weights=progress_weights,
            kp=kp,
            kd=kd,
        )
        if int(getattr(self.config, "adaptive_cvae_micro_heun", 1)):
            pred = action + ds * control1
            control2, terms2 = self._micro_control_field(
                action=pred,
                prev_velocity=ds * control1,
                cond_time=cond_time,
                progress_context=progress_context,
                context_dir=context_dir,
                step_bias=step_bias,
                progress_weights=progress_weights,
                kp=kp,
                kd=kd,
            )
            control = 0.5 * (control1 + control2)
            heun_error = (control2 - control1).detach().float().norm(dim=-1).mean()
            terms = {
                key: 0.5 * (terms1[key] + terms2[key])
                for key in ("feedforward", "feedback", "damping", "function", "control")
            }
            terms["reference"] = terms1["reference"]
            terms["heun_error"] = heun_error
        else:
            control = control1
            terms = dict(terms1)
            terms["heun_error"] = torch.zeros((), device=action.device, dtype=torch.float32)
        update = float(getattr(self.config, "adaptive_cvae_micro_update_scale", 1.0)) * ds * control
        return update, ds, kp, kd, terms

    def _route_progress(
        self,
        action: Tensor,
        progress: Tensor | None,
        route_cond: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        routed, entropy, max_weight, _ = self._route_progress_full(action, progress, route_cond=route_cond)
        return routed, entropy, max_weight

    def _progress_seed_delta(self, action: Tensor, progress_context: Tensor) -> Tensor:
        if not int(getattr(self.config, "adaptive_cvae_progress_z_injection", 1)):
            return torch.zeros_like(action)
        return self.progress_seed_adapter(torch.cat([action, progress_context], dim=-1))

    def _workspace_update_progress(
        self,
        progress: Tensor | None,
        *,
        action: Tensor,
        workspace: Tensor,
        step_context: Tensor,
    ) -> tuple[Tensor | None, Tensor, Tensor]:
        """Per-step progress update.

        Returns (progress, update_norm, action_dependence).

        V72 shelf discipline: progress is a workspace evidence source, so raw
        action content flowing into its values creates an
        action -> progress -> workspace -> action echo one refine step later,
        invisible to the attention-share gauges. With
        latent_cvae_progress_action_isolation=1 the action summary input is
        zeroed (parameter shapes unchanged; checkpoints stay compatible).

        action_dependence is an unconditional detached probe (probe-not-touch):
        the update MLP is re-evaluated with the action input zeroed and the
        relative delta is reported. The MLP is deterministic (LN/Linear/SiLU,
        no dropout), so the extra forward consumes no RNG and paired-seed
        comparability across runs is preserved. Under isolation=1 it reads 0
        by construction, confirming the cut.
        """
        zero_scalar = torch.zeros((), device=action.device, dtype=torch.float32)
        if progress is None:
            return None, zero_scalar, zero_scalar
        slots = int(progress.shape[1])
        action_summary = action.mean(dim=1, keepdim=True).expand(-1, slots, -1)
        workspace_summary = workspace.mean(dim=1, keepdim=True).expand(-1, slots, -1)
        step_summary = step_context[:, None].expand(-1, slots, -1)
        isolate = bool(int(getattr(self.config, "latent_cvae_progress_action_isolation", 0)))
        action_input = torch.zeros_like(action_summary) if isolate else action_summary
        delta = self.workspace_progress_update(torch.cat([
            progress,
            action_input,
            workspace_summary,
            step_summary,
        ], dim=-1))
        with torch.no_grad():
            if isolate:
                action_dependence = zero_scalar
            else:
                delta_no_action = self.workspace_progress_update(torch.cat([
                    progress.detach(),
                    torch.zeros_like(action_summary),
                    workspace_summary.detach(),
                    step_summary.detach(),
                ], dim=-1))
                reference = delta.detach().float()
                action_dependence = (
                    (reference - delta_no_action.float()).norm(dim=-1).mean()
                    / reference.norm(dim=-1).mean().clamp_min(1e-8)
                )
        progress = self.progress_contract_norm(progress + delta)
        return progress, delta.detach().float().norm(dim=-1).mean(), action_dependence

    def _function_delta(self, bank: AdaptiveCVAEFunctionBank, x: Tensor, weights: Tensor | None) -> Tensor:
        if not int(getattr(self.config, "adaptive_cvae_function_adapters", 1)):
            return torch.zeros_like(x)
        return bank(x, weights, self.progress_role_basis)

    def _token_semantic_bias(
        self,
        *,
        action: Tensor,
        cond_time: Tensor,
        routed: Tensor,
        progress_context: Tensor,
    ) -> Tensor:
        if not int(getattr(self.config, "adaptive_cvae_token_semantic_adapter", 1)):
            return torch.zeros_like(action)
        cond_tokens = cond_time[:, None].expand(-1, int(action.shape[1]), -1)
        return self.token_semantic_adapter(torch.cat([action, cond_tokens, routed, progress_context], dim=-1))

    def _output_semantic_delta(self, *, action: Tensor, cond_time: Tensor, progress: Tensor | None) -> tuple[Tensor, Tensor]:
        if not int(getattr(self.config, "adaptive_cvae_output_adapter", 0)):
            return torch.zeros_like(action), torch.zeros_like(action)
        progress_context, _, _, progress_weights = self._route_progress_full(action, progress, route_cond=cond_time)
        progress_context = self._context_dropout(progress_context)
        cond_tokens = cond_time[:, None].expand(-1, int(action.shape[1]), -1)
        semantic_delta = self.output_semantic_adapter(torch.cat([action, cond_tokens, progress_context], dim=-1))
        function_delta = self._function_delta(self.output_function_bank, action + progress_context, progress_weights)
        return semantic_delta + function_delta, function_delta

    def _decode_with_z(
        self,
        *,
        noisy_physical: Tensor,
        time: Tensor,
        trajectory_tokens: Tensor,
        rollout_tokens: Tensor | None,
        cond: Tensor,
        z: Tensor,
        layer_stack: Tensor | None = None,
        evidence_sources: dict[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        cfg = self.config
        batch = int(noisy_physical.shape[0])
        dtype = noisy_physical.dtype
        device = noisy_physical.device
        time_emb = self.time(time.to(dtype=dtype))
        z0 = torch.zeros((), device=device, dtype=torch.float32)
        mmdit_refine = bool(int(getattr(cfg, "latent_cvae_mmdit_decoder", 0)) and len(self.mmdit_blocks) > 0)
        hierarchical_refine = bool(mmdit_refine and int(getattr(cfg, "latent_cvae_hierarchical_workspace", 0)))
        primary_cond = self._mmdit_primary_condition(z=z, time_emb=time_emb) if mmdit_refine else cond + self.time_lift(time_emb)
        primary_z_effect = self._mmdit_primary_z_effect(z=z, time_emb=time_emb, primary_cond=primary_cond) if mmdit_refine else z0
        cond_time = primary_cond
        if mmdit_refine and self.mmdit_noisy_norm is not None:
            # V70: volume-normalized x_t evidence + logit-domain t-gate (same
            # pattern as _decode_with_z_mmdit; this refine path is the live
            # decoder for the adaptive subclass, so the fix must land here).
            noisy_branch = self.mmdit_noisy_norm(self.noisy_action_lift(noisy_physical))
            gate = self._noisy_time_gate(time)
            if gate is None:
                noisy_logit_bias = None
                noisy_gate_mean = torch.ones((), device=device, dtype=torch.float32)
            else:
                noisy_logit_bias = gate.reshape(int(gate.shape[0])).float().clamp_min(1e-6).log()
                noisy_gate_mean = gate.detach().float().mean()
            noisy_branch_norm = noisy_branch.detach().float().norm(dim=-1).mean()
        else:
            noisy_branch, noisy_gate_mean, noisy_branch_norm = self._gated_noisy_branch(noisy_physical, time)
            noisy_logit_bias = None
        base_raw = self.horizon_query.to(device=device, dtype=dtype).expand(batch, -1, -1)
        if not mmdit_refine:
            base_raw = base_raw + noisy_branch + self.trajectory_lift(trajectory_tokens) + cond_time[:, None]
        noisy_branch_ratio = noisy_branch_norm / base_raw.detach().float().norm(dim=-1).mean().clamp_min(1e-6)
        base_action = self._coarse_temporal_base(base_raw)
        base_highfreq = (base_raw - base_action).detach().float().norm(dim=-1).mean()
        # Hierarchical stage memory replaces the external progress/capsule
        # system on the MMDiT mainline. Legacy modules remain intact for A/Bs.
        progress = None if hierarchical_refine else self._latent_progress(batch=batch, cond_time=cond_time, z=z)
        seed_entropy = z0
        seed_max = z0
        seed_temperature = z0
        route_floor_terms: list[Tensor] = []
        regularizer_terms: list[Tensor] = []
        function_rows: list[Tensor] = []
        if mmdit_refine:
            seed_delta = torch.zeros_like(base_action)
            action = base_action
        elif progress is not None and int(getattr(cfg, "adaptive_cvae_progress_z_injection", 1)):
            seed_temperature = self._adaptive_route_temperature(base_action).detach().float().mean()
            seed_context, seed_entropy, seed_max, seed_weights = self._route_progress_full(
                base_action,
                progress,
                route_cond=primary_cond,
            )
            route_floor_terms.append(self._route_entropy_floor(seed_entropy, int(progress.shape[1])))
            seed_context = self._context_dropout(seed_context)
            seed_function = self._function_delta(self.seed_function_bank, base_action + seed_context, seed_weights)
            seed_delta = self._progress_seed_delta(base_action, seed_context) + seed_function
            seed_delta = seed_delta * float(getattr(cfg, "adaptive_cvae_seed_scale", 1.0))
            regularizer_terms.append(seed_delta.float().square().mean())
            function_rows.append(seed_function.detach().float().norm(dim=-1).mean())
            action = base_action + seed_delta
        else:
            seed_delta = torch.zeros_like(base_action)
            action = base_action + self.z_to_token(z.to(device=device, dtype=dtype))[:, None]
        # CR0 probe (do_before_v76 §7/§14.1): the frozen legacy CVAE stem is
        # still an active conditional operator; measure how much it moves the
        # action state so B0 attribution has numbers instead of assumptions.
        stem_before = action
        for block in self.blocks:
            action = block(action, cond_time)
        legacy_stem_effect_ratio = (
            (action - stem_before).detach().float().norm(dim=-1).mean()
            / stem_before.detach().float().norm(dim=-1).mean().clamp_min(1e-6)
        )
        if hierarchical_refine:
            context_capsules = None
            capsule_layer_entropy = z0
            capsule_layer_max = z0
        else:
            context_capsules, capsule_layer_entropy, capsule_layer_max = self._context_capsules(
                cond_time=cond_time,
                layer_stack=layer_stack,
                progress=progress,
            )
        if context_capsules is not None and layer_stack is not None:
            route_floor_terms.append(self._route_entropy_floor(capsule_layer_entropy, int(layer_stack.shape[1])))
        mmdit_cond_token_norm = z0
        workspace_static_memory: PreparedEvidenceMemory | None = None
        hierarchical_evidence: PreparedEvidenceMemory | None = None
        hierarchical_stage_content: Tensor | None = None
        if mmdit_refine:
            if self.mmdit_step_cond_proj is None or self.mmdit_type_embed is None or self.mmdit_action_norm is None:
                raise RuntimeError("MMDiT refine modules are not initialized")
            static_sources = dict(evidence_sources or {})
            if rollout_tokens is not None:
                static_sources["rollout"] = rollout_tokens
            if hierarchical_refine:
                if self.hierarchical_workspace is None:
                    raise RuntimeError("hierarchical evidence workspace is not initialized")
                # Full layer_stack and all deploy-safe evidence remain raw,
                # static values. No action-routed layer/progress value is added.
                hierarchical_evidence = self.hierarchical_workspace.prepare_evidence(
                    static_sources,
                    batch_size=batch,
                    device=device,
                    dtype=dtype,
                )
                hierarchical_stage_content = self.hierarchical_workspace.init_stage(primary_cond)
            else:
                if self.evidence_workspace is None:
                    raise RuntimeError("MMDiT evidence workspace is not initialized")
                # Full layer memory is consumed by the legacy step-dependent
                # router; other invariant sources reuse block-specific K/V.
                static_sources.pop("layer", None)
                if context_capsules is not None:
                    static_sources["capsule"] = context_capsules
                workspace_static_memory = self.evidence_workspace.prepare_static_memory(
                    static_sources,
                    batch_size=batch,
                    device=device,
                    dtype=dtype,
                )
        micro_enabled = bool((not mmdit_refine) and int(getattr(cfg, "adaptive_cvae_micro_control", 1)) and progress is not None)
        progress_center = self._micro_initial_progress(action) if micro_enabled else None
        prev_velocity = torch.zeros_like(action)

        update_rows: list[Tensor] = []
        mmdit_action_update_rows: list[Tensor] = []
        mmdit_cond_update_rows: list[Tensor] = []
        mmdit_cond_attn_rows: list[Tensor] = []
        mmdit_noisy_attn_rows: list[Tensor] = []
        mmdit_workspace_attn_rows: list[Tensor] = []
        mmdit_workspace_enrichment_rows: list[Tensor] = []
        mmdit_low_attn_rows: list[Tensor] = []
        mmdit_stage_attn_rows: list[Tensor] = []
        mmdit_low_enrichment_rows: list[Tensor] = []
        mmdit_stage_enrichment_rows: list[Tensor] = []
        mmdit_noisy_attn_sample_rows: list[Tensor] = []
        mmdit_workspace_attn_sample_rows: list[Tensor] = []
        mmdit_low_attn_sample_rows: list[Tensor] = []
        mmdit_stage_attn_sample_rows: list[Tensor] = []
        workspace_progress_update_rows: list[Tensor] = []
        workspace_progress_dependence_rows: list[Tensor] = []
        workspace_metric_rows: dict[str, list[Tensor]] = {}
        entropy_rows: list[Tensor] = []
        max_rows: list[Tensor] = []
        progress_entropy_rows: list[Tensor] = []
        progress_max_rows: list[Tensor] = []
        continue_rows: list[Tensor] = []
        prefix_rows: list[Tensor] = []
        semantic_rows: list[Tensor] = []
        step_bias_rows: list[Tensor] = []
        temperature_rows: list[Tensor] = []
        condition_strength_mean_rows: list[Tensor] = []
        condition_strength_std_rows: list[Tensor] = []
        condition_strength_max_rows: list[Tensor] = []
        condition_strength_min_rows: list[Tensor] = []
        condition_residual_rows: list[Tensor] = []
        context_direction_rows: list[Tensor] = []
        micro_step_rows: list[Tensor] = []
        micro_step_std_rows: list[Tensor] = []
        micro_progress_rows: list[Tensor] = []
        micro_kp_rows: list[Tensor] = []
        micro_kd_rows: list[Tensor] = []
        micro_feedforward_rows: list[Tensor] = []
        micro_feedback_rows: list[Tensor] = []
        micro_damping_rows: list[Tensor] = []
        micro_function_rows: list[Tensor] = []
        micro_control_rows: list[Tensor] = []
        micro_update_rows: list[Tensor] = []
        micro_heun_rows: list[Tensor] = []
        micro_block_rows: list[Tensor] = []
        micro_controller_rows: list[Tensor] = []
        micro_pred_rows: list[Tensor] = []
        micro_event_rows: list[Tensor] = []
        micro_supervision_logit_rows: list[Tensor] = []
        t = time.to(device=device, dtype=dtype)[:, None, None]
        for step in range(max(self.refine_steps, 0)):
            if hierarchical_refine:
                if (
                    self.hierarchical_workspace is None
                    or hierarchical_evidence is None
                    or hierarchical_stage_content is None
                ):
                    raise RuntimeError("hierarchical workspace state was not prepared")
                (
                    low_workspace_tokens,
                    hierarchical_stage_content,
                    stage_workspace_tokens,
                    low_logit_bias,
                    stage_logit_bias,
                    workspace_metrics,
                ) = self.hierarchical_workspace.step(
                    prepared_evidence=hierarchical_evidence,
                    stage_content=hierarchical_stage_content,
                    primary_cond=primary_cond,
                    step_index=step,
                )
                z_token = self.z_to_token(z.to(device=device, dtype=dtype))
                step_cond_tokens, layout, mmdit_cond_token_norm = self._mmdit_condition_tokens(
                    noisy_tokens=noisy_branch,
                    trajectory_tokens=trajectory_tokens,
                    rollout_tokens=rollout_tokens,
                    cond_time=primary_cond,
                    z_token=z_token,
                    layer_stack=layer_stack,
                    progress_tokens=None,
                    low_workspace_tokens=low_workspace_tokens,
                    stage_workspace_tokens=stage_workspace_tokens,
                )
                before = action
                mmdit_block = self.mmdit_blocks[min(step, len(self.mmdit_blocks) - 1)]
                action, _, mmdit_metrics = mmdit_block(
                    action,
                    step_cond_tokens,
                    primary_cond,
                    noisy_start=layout.noisy_start,
                    noisy_len=layout.noisy_len,
                    rollout_start=layout.rollout_start,
                    rollout_len=layout.rollout_len,
                    low_start=layout.low_start,
                    low_len=layout.low_len,
                    stage_start=layout.stage_start,
                    stage_len=layout.stage_len,
                    update_condition=False,
                    noisy_logit_bias=noisy_logit_bias,
                    low_logit_bias=low_logit_bias,
                    stage_logit_bias=stage_logit_bias,
                )
                update = action - before
                update_energy = update.float().square().mean()
                action_energy = before.detach().float().square().mean().clamp_min(1e-6)
                update_ratio_sq = update_energy / action_energy
                update_ratio = update_ratio_sq.detach().clamp_min(0.0).sqrt()
                regularizer_terms.append(F.relu(update_ratio_sq - 0.25).square())
                workspace_metrics["workspace_action_update_ratio"] = update_ratio
                for key, value in workspace_metrics.items():
                    workspace_metric_rows.setdefault(key, []).append(value.to(device=device))
                prev_velocity = update
                keep = torch.ones((), device=device, dtype=torch.float32)
                mmdit_action_update_rows.append(mmdit_metrics["action_update_norm"].to(device=device))
                mmdit_cond_update_rows.append(mmdit_metrics["cond_update_norm"].to(device=device))
                mmdit_cond_attn_rows.append(mmdit_metrics["action_cond_attn"].to(device=device))
                mmdit_noisy_attn_rows.append(mmdit_metrics["action_noisy_attn"].to(device=device))
                mmdit_workspace_attn_rows.append(mmdit_metrics["action_workspace_attn"].to(device=device))
                mmdit_workspace_enrichment_rows.append(mmdit_metrics["action_workspace_enrichment"].to(device=device))
                mmdit_low_attn_rows.append(mmdit_metrics["action_low_attn"].to(device=device))
                mmdit_stage_attn_rows.append(mmdit_metrics["action_stage_attn"].to(device=device))
                mmdit_low_enrichment_rows.append(mmdit_metrics["action_low_enrichment"].to(device=device))
                mmdit_stage_enrichment_rows.append(mmdit_metrics["action_stage_enrichment"].to(device=device))
                mmdit_noisy_attn_sample_rows.append(mmdit_metrics["action_noisy_attn_rows"].to(device=device))
                mmdit_workspace_attn_sample_rows.append(mmdit_metrics["action_workspace_attn_rows"].to(device=device))
                mmdit_low_attn_sample_rows.append(mmdit_metrics["action_low_attn_rows"].to(device=device))
                mmdit_stage_attn_sample_rows.append(mmdit_metrics["action_stage_attn_rows"].to(device=device))
                update_rows.append(update.detach().float().norm(dim=-1).mean())
                continue_rows.append(keep)
                continue

            step_bias = self._refine_step_bias(step, action)
            route_action = action + step_bias
            temperature_rows.append(self._adaptive_route_temperature(route_action).detach().float().mean())
            if mmdit_refine:
                progress_context, progress_entropy, progress_max, _ = self._route_progress_full(
                    route_action,
                    progress,
                    route_cond=primary_cond,
                )
                if progress is not None:
                    route_floor_terms.append(self._route_entropy_floor(progress_entropy, int(progress.shape[1])))
                progress_context = self._context_dropout(progress_context)
                routed_layer, layer_entropy, layer_max = self._route_layers(
                    route_action,
                    layer_stack,
                    route_cond=primary_cond,
                )
                if layer_stack is not None and int(getattr(cfg, "adaptive_cvae_layer_routing", 1)):
                    route_floor_terms.append(self._route_entropy_floor(layer_entropy, int(layer_stack.shape[1])))
                workspace_sources: dict[str, Tensor] = {}
                progress_query_context = torch.zeros(batch, self.hidden_size, device=device, dtype=dtype)
                progress_as_value = bool(int(getattr(cfg, "latent_cvae_workspace_progress_value", 1)))
                if progress is not None and progress_as_value:
                    workspace_sources["progress"] = progress_context
                elif progress is not None:
                    progress_query_context = progress_context.mean(dim=1)
                if int(getattr(cfg, "latent_cvae_layer_memory", 1)):
                    workspace_sources["routed_layer"] = routed_layer
                step_context = step_bias.mean(dim=1) + progress_query_context
                assert self.evidence_workspace is not None
                workspace_query, workspace_query_scale = self._workspace_query_action(action, noisy_branch)
                workspace_tokens, workspace_metrics = self.evidence_workspace(
                    workspace_sources,
                    action=workspace_query,
                    primary_cond=primary_cond,
                    step_context=step_context,
                    static_memory=workspace_static_memory,
                )
                workspace_metrics["workspace_noisy_query_scale"] = workspace_query_scale
                workspace_metrics["workspace_progress_query_norm"] = progress_query_context.detach().float().norm(dim=-1).mean()
                z_token = self.z_to_token(z.to(device=device, dtype=dtype))
                step_cond_tokens, layout, mmdit_cond_token_norm = self._mmdit_condition_tokens(
                    noisy_tokens=noisy_branch,
                    trajectory_tokens=trajectory_tokens,
                    rollout_tokens=rollout_tokens,
                    cond_time=primary_cond,
                    z_token=z_token,
                    layer_stack=layer_stack,
                    progress_tokens=progress,
                    workspace_tokens=workspace_tokens,
                )
                before = action
                mmdit_block = self.mmdit_blocks[min(step, len(self.mmdit_blocks) - 1)]
                action, _, mmdit_metrics = mmdit_block(
                    action,
                    step_cond_tokens,
                    primary_cond,
                    noisy_start=layout.noisy_start,
                    noisy_len=layout.noisy_len,
                    rollout_start=layout.rollout_start,
                    rollout_len=layout.rollout_len,
                    low_start=layout.low_start,
                    low_len=layout.low_len,
                    stage_start=layout.stage_start,
                    stage_len=layout.stage_len,
                    update_condition=False,
                    noisy_logit_bias=noisy_logit_bias,
                )
                update = action - before
                update_energy = update.float().square().mean()
                action_energy = before.detach().float().square().mean().clamp_min(1e-6)
                update_ratio_sq = update_energy / action_energy
                # The trainable regularizer stays in squared-energy coordinates.
                # sqrt at update=0 has an infinite derivative and poisoned the
                # zero-gated MMDiT initialization on its first backward pass.
                update_ratio = update_ratio_sq.detach().clamp_min(0.0).sqrt()
                # Architectural normalization is primary; this is only a soft
                # trust-region fuse that activates on genuinely runaway updates.
                regularizer_terms.append(F.relu(update_ratio_sq - 0.25).square())
                progress, progress_update_norm, progress_action_dependence = self._workspace_update_progress(
                    progress,
                    action=action,
                    workspace=workspace_tokens,
                    step_context=step_context,
                )
                workspace_progress_update_rows.append(progress_update_norm)
                workspace_progress_dependence_rows.append(progress_action_dependence)
                workspace_metrics["workspace_action_update_ratio"] = update_ratio.detach()
                for key, value in workspace_metrics.items():
                    workspace_metric_rows.setdefault(key, []).append(value.to(device=device))
                prev_velocity = update
                keep = torch.ones((), device=device, dtype=torch.float32)
                mmdit_action_update_rows.append(mmdit_metrics["action_update_norm"].to(device=device))
                mmdit_cond_update_rows.append(mmdit_metrics["cond_update_norm"].to(device=device))
                mmdit_cond_attn_rows.append(mmdit_metrics["action_cond_attn"].to(device=device))
                mmdit_noisy_attn_rows.append(mmdit_metrics["action_noisy_attn"].to(device=device))
                mmdit_workspace_attn_rows.append(mmdit_metrics["action_workspace_attn"].to(device=device))
                mmdit_workspace_enrichment_rows.append(mmdit_metrics["action_workspace_enrichment"].to(device=device))
                mmdit_low_attn_rows.append(mmdit_metrics["action_low_attn"].to(device=device))
                mmdit_stage_attn_rows.append(mmdit_metrics["action_stage_attn"].to(device=device))
                mmdit_low_enrichment_rows.append(mmdit_metrics["action_low_enrichment"].to(device=device))
                mmdit_stage_enrichment_rows.append(mmdit_metrics["action_stage_enrichment"].to(device=device))
                mmdit_noisy_attn_sample_rows.append(mmdit_metrics["action_noisy_attn_rows"].to(device=device))
                mmdit_workspace_attn_sample_rows.append(mmdit_metrics["action_workspace_attn_rows"].to(device=device))
                mmdit_low_attn_sample_rows.append(mmdit_metrics["action_low_attn_rows"].to(device=device))
                mmdit_stage_attn_sample_rows.append(mmdit_metrics["action_stage_attn_rows"].to(device=device))
                update_rows.append(update.detach().float().norm(dim=-1).mean())
                entropy_rows.append(layer_entropy.to(device=device))
                max_rows.append(layer_max.to(device=device))
                progress_entropy_rows.append(progress_entropy.to(device=device))
                progress_max_rows.append(progress_max.to(device=device))
                continue_rows.append(keep)
                step_bias_rows.append(step_bias.detach().float().norm(dim=-1).mean())
                continue
            if micro_enabled:
                progress_context, progress_entropy, progress_max, progress_weights = self._route_progress_monotonic(
                    route_action,
                    progress,
                    progress_center,
                    route_cond=primary_cond,
                )
            else:
                progress_context, progress_entropy, progress_max, progress_weights = self._route_progress_full(
                    route_action,
                    progress,
                    route_cond=primary_cond,
                )
            if progress is not None:
                route_floor_terms.append(self._route_entropy_floor(progress_entropy, int(progress.shape[1])))
            progress_context = self._context_dropout(progress_context)
            prefix = progress_context + step_bias
            if int(getattr(cfg, "adaptive_cvae_prefix_memory", 1)):
                current = self._emit_action(action, cond)
                clean = noisy_physical - t * current["pred_velocity"]
                if int(getattr(cfg, "adaptive_cvae_prefix_detach", 1)):
                    clean = clean.detach()
                prefix = prefix + self.prefix_lift(self._prefix_features(clean))
            if context_capsules is not None:
                context, entropy, max_weight, _ = self._route_context_capsules(
                    route_action,
                    context_capsules,
                    route_cond=primary_cond,
                )
                route_floor_terms.append(self._route_entropy_floor(entropy, int(context_capsules.shape[1])))
                direct_routed, strength, context_dir = self._semantic_context_residual(
                    action=route_action,
                    cond_time=cond_time,
                    context=context,
                    progress_context=progress_context,
                    step_bias=step_bias,
                )
                if micro_enabled:
                    routed = self.micro_context_modulation(context_dir) + direct_routed
                else:
                    routed = direct_routed
                strength_f = strength.detach().float()
                condition_strength_mean_rows.append(strength_f.mean())
                condition_strength_std_rows.append(strength_f.std(unbiased=False))
                condition_strength_max_rows.append(strength_f.max())
                condition_strength_min_rows.append(strength_f.min())
                condition_residual_rows.append(direct_routed.detach().float().norm(dim=-1).mean())
                context_direction_rows.append(context_dir.detach().float().norm(dim=-1).mean())
                regularizer_terms.append(routed.float().square().mean())
            else:
                routed, entropy, max_weight = self._route_layers(route_action, layer_stack, route_cond=primary_cond)
                if (
                    layer_stack is not None
                    and int(getattr(cfg, "adaptive_cvae_layer_routing", 1))
                    and int(getattr(cfg, "latent_cvae_layer_memory", 1))
                ):
                    route_floor_terms.append(self._route_entropy_floor(entropy, int(layer_stack.shape[1])))
                context_dir = self.context_direction_norm(routed)
                if micro_enabled:
                    routed = self.micro_context_modulation(context_dir)
            routed = self._context_dropout(routed)
            semantic_bias = self._token_semantic_bias(
                action=route_action,
                cond_time=cond_time,
                routed=routed,
                progress_context=progress_context,
            )
            regularizer_terms.append(semantic_bias.float().square().mean())
            if micro_enabled:
                function_bias = torch.zeros_like(action)
            else:
                function_bias = self._function_delta(self.refine_function_bank, route_action + routed + progress_context, progress_weights)
            regularizer_terms.append(function_bias.float().square().mean())
            function_rows.append(function_bias.detach().float().norm(dim=-1).mean())
            prefix = prefix + semantic_bias + function_bias
            before = action
            if micro_enabled:
                if int(getattr(cfg, "adaptive_cvae_micro_refine_block", 1)):
                    micro_update, ds, kp, kd, micro_terms = self.micro_refine_block(
                        action=action,
                        prev_update=prev_velocity,
                        cond_time=cond_time,
                        progress_context=progress_context,
                        context_dir=context_dir,
                        step_bias=step_bias,
                        semantic_bias=semantic_bias,
                        progress_weights=progress_weights,
                        role_basis=self.progress_role_basis,
                    )
                    action = action + micro_update
                    prev_velocity = micro_update
                    keep = micro_terms.get("keep", z0).to(device=device)
                    block_delta = torch.zeros_like(action)
                    micro_controller_rows.append(micro_terms["controller"].detach().float().norm(dim=-1).mean())
                else:
                    micro_update, ds, kp, kd, micro_terms = self._micro_integrate(
                        action=action,
                        prev_velocity=prev_velocity,
                        cond_time=cond_time,
                        progress_context=progress_context,
                        context_dir=context_dir,
                        step_bias=step_bias,
                        progress_weights=progress_weights,
                    )
                    action = action + micro_update
                    block_before = action
                    block_action, keep = self.refine_block(action, cond_time, routed, prefix)
                    block_delta = (block_action - block_before) * float(getattr(cfg, "adaptive_cvae_micro_refine_block_scale", 0.30))
                    action = block_before + block_delta
                    prev_velocity = action - before
                if progress_center is not None:
                    progress_center = (progress_center + ds.squeeze(-1).float()).clamp(0.0, 1.0)
                    micro_progress_rows.append(progress_center.detach().float().mean())
                micro_step_rows.append(ds.detach().float().mean())
                micro_step_std_rows.append(ds.detach().float().std(unbiased=False))
                micro_kp_rows.append(kp.detach().float().mean())
                micro_kd_rows.append(kd.detach().float().mean())
                micro_feedforward_rows.append(micro_terms["feedforward"].detach().float().norm(dim=-1).mean())
                micro_feedback_rows.append(micro_terms["feedback"].detach().float().norm(dim=-1).mean())
                micro_damping_rows.append(micro_terms["damping"].detach().float().norm(dim=-1).mean())
                micro_function_rows.append(micro_terms["function"].detach().float().norm(dim=-1).mean())
                micro_control_rows.append(micro_terms["control"].detach().float().norm(dim=-1).mean())
                micro_update_rows.append(micro_update.detach().float().norm(dim=-1).mean())
                micro_heun_rows.append(micro_terms["heun_error"].to(device=device))
                micro_block_rows.append(block_delta.detach().float().norm(dim=-1).mean())
                function_rows.append(micro_terms["function"].detach().float().norm(dim=-1).mean())
                regularizer_terms.append(micro_update.float().square().mean())
                regularizer_terms.append(micro_terms["control"].float().square().mean())
                if int(getattr(cfg, "adaptive_cvae_micro_supervision", 1)):
                    micro_out = self._emit_action(action, cond)
                    micro_pred_rows.append(micro_out["pred_velocity"])
                    micro_event_rows.append(micro_out["event_logits"])
                    supervision_features = torch.cat([action, progress_context, context_dir, step_bias, semantic_bias], dim=-1)
                    micro_supervision_logit_rows.append(self.micro_supervision_router(supervision_features).squeeze(-1))
            else:
                action, keep = self.refine_block(action, cond_time, routed, prefix)
                prev_velocity = action - before
            update_rows.append((action - before).detach().float().norm(dim=-1).mean())
            entropy_rows.append(entropy.to(device=device))
            max_rows.append(max_weight.to(device=device))
            progress_entropy_rows.append(progress_entropy.to(device=device))
            progress_max_rows.append(progress_max.to(device=device))
            continue_rows.append(keep.to(device=device))
            prefix_rows.append(prefix.detach().float().norm(dim=-1).mean())
            semantic_rows.append(semantic_bias.detach().float().norm(dim=-1).mean())
            step_bias_rows.append(step_bias.detach().float().norm(dim=-1).mean())

        if mmdit_refine and self.mmdit_action_norm is not None:
            action = self.mmdit_action_norm(action)
        if mmdit_refine:
            output_delta = torch.zeros_like(action)
            output_function = torch.zeros_like(action)
        else:
            output_delta, output_function = self._output_semantic_delta(action=action, cond_time=cond_time, progress=progress)
            output_scale = float(getattr(cfg, "adaptive_cvae_output_scale", 1.0))
            output_delta = output_delta * output_scale
            output_function = output_function * output_scale
        regularizer_terms.append(output_delta.float().square().mean())
        function_rows.append(output_function.detach().float().norm(dim=-1).mean())
        emit_condition = primary_cond if mmdit_refine else cond
        out = self._emit_action(action + output_delta, emit_condition)
        progress_norm = progress.detach().float().norm(dim=-1).mean() if progress is not None else z0
        workspace_summary = {
            key: torch.stack(values).mean()
            for key, values in workspace_metric_rows.items()
            if values
        }
        workspace_rollout = workspace_summary.get("workspace_rollout_attention", z0)
        workspace_source_count = workspace_summary.get("workspace_source_count", z0)
        workspace_rollout_enrichment = workspace_rollout * workspace_source_count.clamp_min(1.0)
        route_time_bias = self._route_time_bias(primary_cond, action)
        route_time_norm = route_time_bias.detach().float().norm(dim=-1).mean() if route_time_bias is not None else z0
        out.update({
            "adaptive_noisy_gate_mean": noisy_gate_mean.to(device=device),
            "adaptive_noisy_branch_norm": noisy_branch_norm.to(device=device),
            "adaptive_noisy_branch_ratio": noisy_branch_ratio.to(device=device),
            "adaptive_refine_update_mean": torch.stack(update_rows).mean() if update_rows else z0,
            "adaptive_route_entropy": torch.stack(entropy_rows).mean() if entropy_rows else z0,
            "adaptive_route_max": torch.stack(max_rows).mean() if max_rows else z0,
            "adaptive_route_effective_slots": torch.exp(torch.stack(entropy_rows).mean()) if entropy_rows else z0,
            "adaptive_progress_entropy": torch.stack(progress_entropy_rows).mean() if progress_entropy_rows else z0,
            "adaptive_progress_max": torch.stack(progress_max_rows).mean() if progress_max_rows else z0,
            "adaptive_progress_effective_slots": torch.exp(torch.stack(progress_entropy_rows).mean()) if progress_entropy_rows else z0,
            "adaptive_progress_norm": progress_norm,
            "adaptive_continue_mean": torch.stack(continue_rows).mean() if continue_rows else z0,
            "adaptive_prefix_norm": torch.stack(prefix_rows).mean() if prefix_rows else z0,
            "adaptive_progress_seed_entropy": seed_entropy.to(device=device),
            "adaptive_progress_seed_max": seed_max.to(device=device),
            "adaptive_progress_seed_effective_slots": torch.exp(seed_entropy.to(device=device)) if progress is not None else z0,
            "adaptive_progress_seed_norm": seed_delta.detach().float().norm(dim=-1).mean(),
            "adaptive_route_temperature_mean": torch.stack([seed_temperature.to(device=device), *temperature_rows]).mean() if temperature_rows else seed_temperature.to(device=device),
            "adaptive_route_time_query_norm": route_time_norm,
            "adaptive_semantic_bias_norm": torch.stack(semantic_rows).mean() if semantic_rows else z0,
            "adaptive_output_adapter_norm": output_delta.detach().float().norm(dim=-1).mean(),
            "adaptive_function_delta_norm": torch.stack(function_rows).mean() if function_rows else z0,
            "adaptive_base_highfreq_norm": base_highfreq,
            "legacy_stem_effect_ratio": legacy_stem_effect_ratio,
            "adaptive_refine_step_bias_norm": torch.stack(step_bias_rows).mean() if step_bias_rows else z0,
            "adaptive_capsule_layer_entropy": capsule_layer_entropy.to(device=device),
            "adaptive_capsule_layer_max": capsule_layer_max.to(device=device),
            "adaptive_capsule_layer_effective_slots": torch.exp(capsule_layer_entropy.to(device=device)) if context_capsules is not None else z0,
            "adaptive_condition_strength_mean": torch.stack(condition_strength_mean_rows).mean() if condition_strength_mean_rows else z0,
            "adaptive_condition_strength_std": torch.stack(condition_strength_std_rows).mean() if condition_strength_std_rows else z0,
            "adaptive_condition_strength_max": torch.stack(condition_strength_max_rows).mean() if condition_strength_max_rows else z0,
            "adaptive_condition_strength_min": torch.stack(condition_strength_min_rows).mean() if condition_strength_min_rows else z0,
            "adaptive_condition_residual_norm": torch.stack(condition_residual_rows).mean() if condition_residual_rows else z0,
            "adaptive_context_direction_norm": torch.stack(context_direction_rows).mean() if context_direction_rows else z0,
            "mmdit_action_update_norm": torch.stack(mmdit_action_update_rows).mean() if mmdit_action_update_rows else z0,
            "mmdit_cond_update_norm": torch.stack(mmdit_cond_update_rows).mean() if mmdit_cond_update_rows else z0,
            "mmdit_action_cond_attention": torch.stack(mmdit_cond_attn_rows).mean() if mmdit_cond_attn_rows else z0,
            "mmdit_action_noisy_attention": torch.stack(mmdit_noisy_attn_rows).mean() if mmdit_noisy_attn_rows else z0,
            "mmdit_action_workspace_attention": torch.stack(mmdit_workspace_attn_rows).mean() if mmdit_workspace_attn_rows else z0,
            "mmdit_action_workspace_enrichment": torch.stack(mmdit_workspace_enrichment_rows).mean() if mmdit_workspace_enrichment_rows else z0,
            "mmdit_action_low_attention": torch.stack(mmdit_low_attn_rows).mean() if mmdit_low_attn_rows else z0,
            "mmdit_action_stage_attention": torch.stack(mmdit_stage_attn_rows).mean() if mmdit_stage_attn_rows else z0,
            "mmdit_action_low_enrichment": torch.stack(mmdit_low_enrichment_rows).mean() if mmdit_low_enrichment_rows else z0,
            "mmdit_action_stage_enrichment": torch.stack(mmdit_stage_enrichment_rows).mean() if mmdit_stage_enrichment_rows else z0,
            "mmdit_action_rollout_attention": workspace_rollout,
            "mmdit_action_rollout_enrichment": workspace_rollout_enrichment,
            "mmdit_action_token_norm": action.detach().float().norm(dim=-1).mean() if mmdit_refine else z0,
            "mmdit_condition_token_norm": mmdit_cond_token_norm.to(device=device) if mmdit_refine else z0,
            "mmdit_noisy_token_norm": noisy_branch_norm.to(device=device) if mmdit_refine else z0,
            "primary_condition_norm": primary_cond.detach().float().norm(dim=-1).mean() if mmdit_refine else z0,
            "primary_z_effect_norm": primary_z_effect,
            "workspace_progress_update_norm": torch.stack(workspace_progress_update_rows).mean() if workspace_progress_update_rows else z0,
            "workspace_progress_action_dependence": torch.stack(workspace_progress_dependence_rows).mean() if workspace_progress_dependence_rows else z0,
            **self._time_stratified_attention(
                time,
                torch.stack(mmdit_noisy_attn_sample_rows).mean(dim=0) if mmdit_noisy_attn_sample_rows else torch.zeros(batch, device=device, dtype=torch.float32),
                torch.stack(mmdit_workspace_attn_sample_rows).mean(dim=0) if mmdit_workspace_attn_sample_rows else torch.zeros(batch, device=device, dtype=torch.float32),
                torch.stack(mmdit_low_attn_sample_rows).mean(dim=0) if mmdit_low_attn_sample_rows else None,
                torch.stack(mmdit_stage_attn_sample_rows).mean(dim=0) if mmdit_stage_attn_sample_rows else None,
            ),
            "adaptive_micro_step_mean": torch.stack(micro_step_rows).mean() if micro_step_rows else z0,
            "adaptive_micro_step_std": torch.stack(micro_step_std_rows).mean() if micro_step_std_rows else z0,
            "adaptive_micro_progress_mean": torch.stack(micro_progress_rows).mean() if micro_progress_rows else z0,
            "adaptive_micro_kp_mean": torch.stack(micro_kp_rows).mean() if micro_kp_rows else z0,
            "adaptive_micro_kd_mean": torch.stack(micro_kd_rows).mean() if micro_kd_rows else z0,
            "adaptive_micro_feedforward_norm": torch.stack(micro_feedforward_rows).mean() if micro_feedforward_rows else z0,
            "adaptive_micro_feedback_norm": torch.stack(micro_feedback_rows).mean() if micro_feedback_rows else z0,
            "adaptive_micro_damping_norm": torch.stack(micro_damping_rows).mean() if micro_damping_rows else z0,
            "adaptive_micro_function_norm": torch.stack(micro_function_rows).mean() if micro_function_rows else z0,
            "adaptive_micro_control_norm": torch.stack(micro_control_rows).mean() if micro_control_rows else z0,
            "adaptive_micro_update_norm": torch.stack(micro_update_rows).mean() if micro_update_rows else z0,
            "adaptive_micro_heun_error": torch.stack(micro_heun_rows).mean() if micro_heun_rows else z0,
            "adaptive_micro_refine_block_norm": torch.stack(micro_block_rows).mean() if micro_block_rows else z0,
            "adaptive_micro_controller_norm": torch.stack(micro_controller_rows).mean() if micro_controller_rows else z0,
            "adaptive_regularizer": torch.stack(regularizer_terms).mean() if regularizer_terms else z0,
            "adaptive_route_entropy_regularizer": torch.stack(route_floor_terms).mean() if route_floor_terms else z0,
            **workspace_summary,
        })
        if micro_pred_rows:
            out["adaptive_micro_pred_velocity"] = torch.stack(micro_pred_rows, dim=1)
            out["adaptive_micro_event_logits"] = torch.stack(micro_event_rows, dim=1)
            out["adaptive_micro_supervision_logits"] = torch.stack(micro_supervision_logit_rows, dim=1)
        return out
