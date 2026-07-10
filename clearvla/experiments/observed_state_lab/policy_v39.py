from __future__ import annotations

"""V39 staged mid-cut latent-contract temporal policy.

V39 keeps the V38.6.2 action-centered controlled-residual policy path, but it
adds an explicit *mid-cut contract* inside the DiT block stack.  The first
training stage can stop at the cut and train only intentionally weak heads.  The
second stage resumes from that checkpoint, runs the remaining DiT blocks, and
trains the formal policy head while preserving the mid-cut contract with a small
auxiliary loss.

The important contract is architectural: simple readout heads are attached to a
DiT midpoint, before the final decoder has enough capacity to hide shortcuts.
These heads are not meant to be the deployable policy; they are probes that make
motion/contact/future information readable at Z_mid.
"""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .policy import RejectableHistoryProposal
from .policy_v36_2 import ParsevalGripperTemporalFrame, PhysicalActionCodec, PhysicalActionTokenLift
from .policy_v36_3 import TransitionAwarePhysicalVelocityHead
from .world_model import BiasFreeFFN
from .policy_v38 import (
    CanvasPhysicalVelocityHead,
    ControlledResidualLatentDynamics,
    DenseVisualMemory,
    RolloutActionResidualHead,
    RolloutTargetCodec,
    TemporalDynamicsBoundDiTBlock,
    UnifiedCanvasSeed,
    V38PolicyConfig,
)
from .policy import TimeEmbedding


@dataclass(frozen=True)
class V39PolicyConfig(V38PolicyConfig):
    """Configuration for the staged mid-cut latent contract policy."""

    # Number of DiT blocks executed before exposing Z_mid.  With the default
    # depth=6 this means blocks[0:3] form the world/latent builder and
    # blocks[3:6] act as the formal policy/refinement stack.
    midcut_layer: int = 3
    # Lightweight contract heads.  Kept small on purpose: these heads should
    # read structure from Z_mid, not manufacture it after the fact.
    midcut_future_gain_init: float = 0.10
    # Old checkpoints omitted this field and therefore retain the learned
    # decomposition. New training CLI runs explicitly select fixed_zero, which
    # makes rollout_effect == controlled_delta and removes base/delta gauge
    # freedom without discarding the deep rollout representation.
    controlled_base_mode: str = "learned"

    # V39.1: optional multi-layer contract adapters.  When enabled, every DiT
    # block exposes a tiny side adapter and weak readout heads.  The adapters
    # are contract probes with a scaled gradient into the trunk, so we do not
    # have to guess a single hard cut layer.
    layer_contract_adapters: int = 0
    layer_contract_adapter_dim: int = 128
    layer_contract_grad_scale: float = 1.0
    layer_contract_residual_scale: float = 0.50

    # V39.2: every layer first exposes a supervised latent head, then a single
    # shared flow-matching action probe reads only that latent.  This keeps the
    # action probe from bypassing the latent contract while avoiding a full
    # action decoder per layer.
    layer_shared_fm_probe: int = 0
    layer_fm_probe_hidden: int = 256

    # V39.3: recurrent milestone consequence.  Instead of directly reading a
    # single future latent from each layer, a small shared latent transition cell
    # rolls the layer-local latent forward K milestone steps under candidate
    # action segments.  The resulting z^1..z^K are compared against sparse
    # future-anchor targets.
    layer_recurrent_consequence: int = 0
    layer_consequence_steps: int = 6
    layer_consequence_hidden: int = 256
    layer_consequence_delta_scale: float = 1.0
    layer_consequence_initial_gain: float = 0.10

    # V40.1: unified intervention-latent contract.  The old split
    # ``latent branch + causal head`` is collapsed into a single intervention
    # encoder that jointly reads layer-local latent tokens, robot/history tokens,
    # and candidate action-segment tokens.  It emits an action-conditioned latent
    # residual plus policy effect tokens.  Optional feedback depth is allowed but
    # defaults to zero in the V40.1 CLI; future information enters through
    # targets/losses and counterfactual views, not by re-running a heavy inner
    # transformer.
    layer_causal_feedback_depth: int = 0
    layer_causal_memory_tokens: int = 4
    layer_low_causal_weight: float = 1.0
    layer_high_causal_weight: float = 1.0
    layer_low_latent_weight: float = 1.0
    layer_high_latent_weight: float = 1.0
    layer_causal_event_from_effect: int = 1
    # Disabled by default: the current in-batch implementation swaps latent
    # canvas tokens that may already contain action information, so it is not a
    # strict same-action/different-state intervention.
    layer_state_counterfactual: int = 0
    # Optional consequence-cell input cleanup.  When enabled, a no-grad preview
    # denoising pass supplies a deployable clean-action estimate to the
    # consequence cell instead of the label-derived noisy bridge state.
    action_consequence_self_condition: int = 0
    # Loss-free shortcut diagnostic: rerun each consequence cell with rollout
    # tokens zeroed and log how much its output shifts.
    layer_zero_base_diagnostic: int = 1

    # Optional final decoder used only in policy-stage experiments.  The safe
    # variants keep the warm-started V40.1 legacy velocity head as the base
    # policy and let a V37-style latent/action/event-token denoiser predict a
    # small zero-initialized residual.  ``layered_residual_action_flow`` is the
    # stronger version: each residual decoder block receives a different
    # layer-pair memory from the V40.1 contract hierarchy, so the network uses
    # hierarchical latent injection instead of pretending we have a supervised
    # temporal diffusion trajectory.
    final_action_decoder: str = "legacy"
    action_flow_residual_depth: int = 2
    action_flow_residual_high_slots: int = 4
    action_flow_residual_max_scale: float = 0.20
    action_flow_residual_visual_memory: int = 1
    action_flow_residual_context_memory: int = 1
    action_flow_residual_transition_memory: int = 1
    action_flow_residual_layer_memory: int = 1
    action_flow_residual_layer_pair_schedule: str = "0:1,1:3,3:5,5:7"
    action_flow_residual_layer_detach: int = 1
    action_flow_residual_stage_router: int = 1
    action_flow_residual_anchor_memory: int = 1

    # V41: clean latent-main action decoder.  This mode keeps the V40.1 data,
    # codec, losses, metrics, and latent/consequence trunk, but it does not use
    # the old direct/rollout action head as a legacy base and it does not add an
    # external residual path.  The final physical velocity is emitted by one
    # hierarchical decoder whose blocks repeatedly inject every V40 layer memory
    # plus controlled-delta/event context.
    latent_action_decoder_depth: int = 8
    latent_action_high_slots: int = 4
    latent_action_layer_schedule: str = "0:1,1:2,2:3,3:4,4:5,5:6,6:7,7:7"
    latent_action_visual_memory: int = 0
    latent_action_context_memory: int = 0
    latent_action_transition_memory: int = 1
    latent_action_layer_memory: int = 1
    latent_action_anchor_memory: int = 1
    latent_action_stage_router: int = 0
    latent_action_layer_detach: int = 0
    latent_action_event_gripper_gate: int = 1
    # V41.1: horizon-dependent compute depth inside the single main decoder.
    # Near actions remain shallow; mid/far tokens keep receiving deeper
    # consequence/rollout injections.  This preserves one final action path and
    # does not reintroduce legacy/residual side heads.
    latent_action_temporal_depth: int = 0
    latent_action_near_steps: int = 4
    latent_action_mid_steps: int = 8
    latent_action_near_depth: int = 2
    latent_action_mid_depth: int = 4

    # V42: compact latent-conditioned CVAE action head.  This is the preferred
    # clean replacement for the oversized V41 Transformer decoder in low-data
    # settings.  It keeps one final action path and no legacy/residual bypass:
    # every V40 layer latent is pooled into the condition, a small posterior
    # q(z | latent, target action) is used only during training, and inference
    # uses the conditional prior p(z | latent).  The decoder is deliberately
    # shallow and FiLM-conditioned instead of cross-attending huge memories.
    latent_cvae_z_dim: int = 64
    latent_cvae_decoder_depth: int = 3
    latent_cvae_ffn_expansion: float = 2.0
    latent_cvae_layer_memory: int = 1
    latent_cvae_transition_memory: int = 1
    latent_cvae_transition_detach: int = 1
    latent_cvae_context_memory: int = 0
    latent_cvae_visual_memory: int = 0
    latent_cvae_layer_detach: int = 1
    latent_cvae_layer_grad_scale: float = 0.0
    # Keep rollout/consequence features visible to the final decoder without
    # letting their unconstrained scale or gradients turn them into an
    # auxiliary action-prediction path.
    latent_cvae_condition_source_norm: int = 1
    latent_cvae_bounded_consequence_fusion: int = 1
    latent_cvae_consequence_scale_init: float = 0.10
    latent_cvae_consequence_scale_max: float = 0.50
    latent_cvae_event_gripper_gate: int = 1
    latent_cvae_inference_sample: int = 0
    latent_cvae_output_init_std: float = 1e-3
    latent_cvae_mu_bound: float = 1.5
    latent_cvae_min_std: float = 0.5
    latent_cvae_causal_attention: int = 1
    # V53-A1: suppress the direct x_t/noisy-action branch near low flow time,
    # where x_t is close to the answer and can become a cheap denoising shortcut.
    latent_cvae_noisy_gate: int = 0
    latent_cvae_noisy_gate_min: float = 0.05
    latent_cvae_noisy_gate_power: float = 1.5
    # V53-B1: recurrent scan over ordered per-layer summaries.  This keeps
    # layer-depth information as a sequence instead of flattening all layer
    # summaries into one hypercolumn condition.
    latent_cvae_layer_scan: int = 0
    latent_cvae_layer_scan_alpha: float = 0.2
    # V60: MMDiT-lite bottom decoder.  The CVAE prior/posterior contract stays
    # intact, but the final action tokens read layer/trajectory/noisy/progress
    # tokens through a compact joint-attention mixer.  Noisy action is kept out
    # of the action residual stream by default, so the velocity head cannot use
    # a pure x_t -> output linear shortcut.
    latent_cvae_mmdit_decoder: int = 0
    latent_cvae_mmdit_depth: int = 3
    latent_cvae_mmdit_cond_update: int = 0
    latent_cvae_mmdit_noisy_causal: int = 1
    # V70: close the volume degree of freedom on the x_t evidence.  Noisy
    # condition tokens pass through LayerNorm (same volume scale as every
    # other market participant) and the t-gate moves from value scaling to an
    # additive log g(t) bias on the noisy attention logits -- same gating
    # semantics, but no longer defeatable by lift amplification, and the
    # attention-share gauges become honest influence readings.
    latent_cvae_mmdit_noisy_logit_gate: int = 0
    # V72: shelf discipline -- the evidence workspace is for world evidence;
    # content the action wrote must not return as evidence. When enabled, the
    # per-step progress update no longer receives the raw action summary
    # (zeros are fed in its place so parameter shapes and checkpoints stay
    # compatible across both arms of the A/B). Progress then evolves from
    # workspace evidence + step context only. The action->progress->workspace
    # echo was measured growing monotonically through v69 (update norm 5.96 at
    # epoch 1 -> 10-13.8 at epoch 8) while val was saturated.
    latent_cvae_progress_action_isolation: int = 0
    # V65: z is the primary denoising condition. All other semantic sources
    # first negotiate through a horizon-aligned evidence workspace; this count
    # controls its information bandwidth without tying it to action_horizon.
    latent_cvae_horizon_tokens: int = 24
    # V66: let workspace queries inspect the deploy-safe noisy flow state. This
    # changes evidence selection only; noisy actions never enter workspace V.
    latent_cvae_workspace_noisy_query: int = 0
    # Diagnostic ablation for the full-resolution trajectory canvas in workspace
    # values. The MMDiT action stream still receives noisy/action tokens; setting
    # this to zero tests whether trajectory is useful evidence or an x_t echo.
    latent_cvae_workspace_trajectory_source: int = 1
    # V73/V74: structured workspace discipline.  ``global_sources`` controls
    # whether scan/lateral summaries are also exposed as workspace values after
    # already shaping the global CVAE condition.  ``layer_source`` controls the
    # static full-layer menu; the adaptive MMDiT path can still read layers via
    # per-step routed_layer/capsules.  ``progress_value`` controls whether
    # progress is a workspace value or only contributes to the step query/state.
    # ``time_state`` injects the existing primary_cond (z + time_lift(time))
    # into workspace slots; it deliberately reuses the live MMDiT time
    # definition instead of creating a second time embedding.  ``slot_time``
    # keeps that signal slot-aware so it does not become a uniform 24-token
    # bias that washes out local/event retrieval.
    latent_cvae_workspace_global_sources: int = 1
    latent_cvae_workspace_layer_source: int = 1
    latent_cvae_workspace_progress_value: int = 1
    latent_cvae_workspace_time_state: int = 0
    latent_cvae_workspace_slot_time_state: int = 1
    latent_cvae_workspace_slot_time_scale: float = 0.10
    # V74B: central controller for workspace retrieval strengths, role bias,
    # query modulation, and delayed capacity without changing the 24-token
    # MMDiT interface.
    latent_cvae_workspace_controller: int = 0
    # V75: hierarchical evidence workspace. Low slots are temporary evidence
    # reads, stage slots are persistent across refine steps, and action tokens
    # consume both only through MMDiT. Stage state may alter low retrieval
    # queries/selectors, but never enters the low value/residual stream.
    latent_cvae_hierarchical_workspace: int = 0
    latent_cvae_stage_slots: int = 6
    latent_cvae_stage_promote_scale_init: float = 0.05
    # V43: adaptive recurrent CVAE action decoder.  This mode keeps the V42
    # prior/posterior CVAE contract but lets the final action tokens run a
    # small shared recurrent refinement loop.  Each token can read a causal
    # summary of earlier predicted physical actions and softly route to the
    # V40 layer summaries, so later horizon positions can depend on earlier
    # predicted state without returning to the oversized V41 cross-attention
    # decoder.
    adaptive_cvae_refine_steps: int = 3
    adaptive_cvae_progress_memory: int = 1
    adaptive_cvae_progress_steps: int = 6
    adaptive_cvae_prefix_memory: int = 0
    adaptive_cvae_layer_routing: int = 1
    adaptive_cvae_route_cosine: int = 1
    adaptive_cvae_route_temperature: float = 1.0
    adaptive_cvae_prefix_detach: int = 1
    adaptive_cvae_progress_z_injection: int = 1
    adaptive_cvae_route_query_bias: int = 1
    adaptive_cvae_route_time_query: int = 0
    adaptive_cvae_token_semantic_adapter: int = 1
    adaptive_cvae_output_adapter: int = 0
    adaptive_cvae_context_dropout: float = 0.05
    adaptive_cvae_route_entropy_floor_ratio: float = 0.35
    adaptive_cvae_function_adapters: int = 1
    adaptive_cvae_function_rank: int = 64
    adaptive_cvae_progress_role_dim: int = 16
    adaptive_cvae_route_topk: int = 0
    adaptive_cvae_route_sparsemax: int = 1
    adaptive_cvae_route_adaptive_temperature: int = 1
    adaptive_cvae_route_min_temperature: float = 0.35
    adaptive_cvae_route_max_temperature: float = 1.25
    adaptive_cvae_role_query: int = 1
    adaptive_cvae_step_roles: int = 1
    adaptive_cvae_coarse_stride: int = 4
    adaptive_cvae_coarse_strength: float = 0.35
    adaptive_cvae_seed_scale: float = 0.35
    adaptive_cvae_output_scale: float = 0.05
    adaptive_cvae_context_capsules: int = 1
    adaptive_cvae_context_capsule_count: int = 6
    adaptive_cvae_direct_condition_residual: int = 0
    adaptive_cvae_condition_strength: int = 0
    adaptive_cvae_condition_strength_min: float = 0.03
    adaptive_cvae_condition_strength_max: float = 1.50
    adaptive_cvae_condition_strength_init: float = 0.35
    adaptive_cvae_micro_control: int = 1
    adaptive_cvae_micro_refine_block: int = 1
    adaptive_cvae_micro_supervision: int = 1
    adaptive_cvae_micro_heun: int = 1
    adaptive_cvae_micro_monotonic_progress: int = 1
    adaptive_cvae_micro_min_step: float = 0.03
    adaptive_cvae_micro_max_step: float = 0.35
    adaptive_cvae_micro_step_init: float = 0.12
    adaptive_cvae_micro_kp_max: float = 0.60
    adaptive_cvae_micro_kp_init: float = 0.18
    adaptive_cvae_micro_kd_max: float = 0.45
    adaptive_cvae_micro_kd_init: float = 0.08
    adaptive_cvae_micro_update_scale: float = 1.0
    adaptive_cvae_micro_refine_block_scale: float = 0.30
    adaptive_cvae_micro_progress_distance_scale: float = 4.0

    def validate(self) -> None:
        super().validate()
        if str(self.controlled_base_mode) not in {"learned", "fixed_zero"}:
            raise ValueError("controlled_base_mode must be learned or fixed_zero")
        if int(self.midcut_layer) < 1 or int(self.midcut_layer) > int(self.depth):
            raise ValueError("midcut_layer must be in [1, depth]")
        if float(self.midcut_future_gain_init) <= 0:
            raise ValueError("midcut_future_gain_init must be positive")
        if int(self.layer_contract_adapters) not in (0, 1):
            raise ValueError("layer_contract_adapters must be 0 or 1")
        if int(self.layer_contract_adapter_dim) < 8:
            raise ValueError("layer_contract_adapter_dim must be >= 8")
        if not (0.0 <= float(self.layer_contract_grad_scale) <= 1.0):
            raise ValueError("layer_contract_grad_scale must be in [0, 1]")
        if float(self.layer_contract_residual_scale) < 0:
            raise ValueError("layer_contract_residual_scale must be non-negative")
        if int(self.layer_shared_fm_probe) not in (0, 1):
            raise ValueError("layer_shared_fm_probe must be 0 or 1")
        if int(self.layer_fm_probe_hidden) < 16:
            raise ValueError("layer_fm_probe_hidden must be >= 16")
        if int(self.layer_recurrent_consequence) not in (0, 1):
            raise ValueError("layer_recurrent_consequence must be 0 or 1")
        if int(self.layer_consequence_steps) < 1:
            raise ValueError("layer_consequence_steps must be >= 1")
        if int(self.layer_consequence_hidden) < 16:
            raise ValueError("layer_consequence_hidden must be >= 16")
        if float(self.layer_consequence_delta_scale) <= 0:
            raise ValueError("layer_consequence_delta_scale must be positive")
        if float(self.layer_consequence_initial_gain) <= 0:
            raise ValueError("layer_consequence_initial_gain must be positive")
        if int(self.layer_causal_feedback_depth) < 0:
            raise ValueError("layer_causal_feedback_depth must be >= 0")
        if int(self.layer_causal_memory_tokens) < 1:
            raise ValueError("layer_causal_memory_tokens must be >= 1")
        for name in (
            "layer_low_causal_weight", "layer_high_causal_weight",
            "layer_low_latent_weight", "layer_high_latent_weight",
        ):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if int(self.layer_causal_event_from_effect) not in (0, 1):
            raise ValueError("layer_causal_event_from_effect must be 0 or 1")
        if int(self.layer_state_counterfactual) not in (0, 1):
            raise ValueError("layer_state_counterfactual must be 0 or 1")
        if int(self.action_consequence_self_condition) not in (0, 1):
            raise ValueError("action_consequence_self_condition must be 0 or 1")
        if int(self.layer_zero_base_diagnostic) not in (0, 1):
            raise ValueError("layer_zero_base_diagnostic must be 0 or 1")
        if str(self.final_action_decoder) not in {"legacy", "residual_action_flow", "layered_residual_action_flow", "latent_main_action", "latent_cvae_action", "adaptive_recurrent_cvae_action"}:
            raise ValueError("final_action_decoder must be legacy, residual_action_flow, layered_residual_action_flow, latent_main_action, latent_cvae_action, or adaptive_recurrent_cvae_action")
        if int(self.action_flow_residual_depth) < 1:
            raise ValueError("action_flow_residual_depth must be >= 1")
        if int(self.action_flow_residual_high_slots) < 1:
            raise ValueError("action_flow_residual_high_slots must be >= 1")
        if float(self.action_flow_residual_max_scale) < 0:
            raise ValueError("action_flow_residual_max_scale must be non-negative")
        for name in (
            "action_flow_residual_visual_memory",
            "action_flow_residual_context_memory",
            "action_flow_residual_transition_memory",
            "action_flow_residual_layer_memory",
            "action_flow_residual_layer_detach",
            "action_flow_residual_stage_router",
            "action_flow_residual_anchor_memory",
        ):
            if int(getattr(self, name)) not in (0, 1):
                raise ValueError(f"{name} must be 0 or 1")
        if int(self.latent_action_decoder_depth) < 1:
            raise ValueError("latent_action_decoder_depth must be >= 1")
        if int(self.latent_action_high_slots) < 1:
            raise ValueError("latent_action_high_slots must be >= 1")
        for name in (
            "latent_action_visual_memory",
            "latent_action_context_memory",
            "latent_action_transition_memory",
            "latent_action_layer_memory",
            "latent_action_anchor_memory",
            "latent_action_stage_router",
            "latent_action_layer_detach",
            "latent_action_event_gripper_gate",
            "latent_action_temporal_depth",
        ):
            if int(getattr(self, name)) not in (0, 1):
                raise ValueError(f"{name} must be 0 or 1")
        if int(self.latent_action_near_steps) < 0 or int(self.latent_action_mid_steps) < 0:
            raise ValueError("latent_action_near_steps/mid_steps must be non-negative")
        if int(self.latent_action_near_steps) > int(self.latent_action_mid_steps):
            raise ValueError("latent_action_near_steps must be <= latent_action_mid_steps")
        if int(self.latent_action_mid_steps) > int(self.action_horizon):
            raise ValueError("latent_action_mid_steps cannot exceed action_horizon")
        if int(self.latent_action_mid_depth) < int(self.latent_action_near_depth):
            raise ValueError("latent_action_mid_depth must be >= latent_action_near_depth")
        if int(self.latent_action_near_depth) > int(self.latent_action_decoder_depth) or int(self.latent_action_mid_depth) > int(self.latent_action_decoder_depth):
            raise ValueError("latent_action_near_depth/mid_depth cannot exceed latent_action_decoder_depth")
        if int(self.latent_cvae_z_dim) < 1:
            raise ValueError("latent_cvae_z_dim must be >= 1")
        if int(self.latent_cvae_decoder_depth) < 1:
            raise ValueError("latent_cvae_decoder_depth must be >= 1")
        if float(self.latent_cvae_ffn_expansion) < 1.0:
            raise ValueError("latent_cvae_ffn_expansion must be >= 1")
        for name in (
            "latent_cvae_layer_memory", "latent_cvae_transition_memory",
            "latent_cvae_transition_detach", "latent_cvae_context_memory", "latent_cvae_visual_memory",
            "latent_cvae_layer_detach", "latent_cvae_condition_source_norm",
            "latent_cvae_bounded_consequence_fusion",
            "latent_cvae_event_gripper_gate",
            "latent_cvae_inference_sample", "latent_cvae_noisy_gate", "latent_cvae_layer_scan",
            "latent_cvae_mmdit_decoder", "latent_cvae_mmdit_cond_update", "latent_cvae_mmdit_noisy_causal",
            "latent_cvae_mmdit_noisy_logit_gate",
            "latent_cvae_progress_action_isolation",
            "latent_cvae_workspace_noisy_query", "latent_cvae_workspace_trajectory_source",
            "latent_cvae_workspace_global_sources", "latent_cvae_workspace_layer_source",
            "latent_cvae_workspace_progress_value", "latent_cvae_workspace_time_state",
            "latent_cvae_workspace_slot_time_state", "latent_cvae_workspace_controller",
            "latent_cvae_hierarchical_workspace",
        ):
            if int(getattr(self, name)) not in (0, 1):
                raise ValueError(f"{name} must be 0 or 1")
        if float(self.latent_cvae_workspace_slot_time_scale) < 0.0:
            raise ValueError("latent_cvae_workspace_slot_time_scale must be >= 0")
        if int(self.latent_cvae_mmdit_depth) < 1:
            raise ValueError("latent_cvae_mmdit_depth must be >= 1")
        if int(self.latent_cvae_horizon_tokens) < 1:
            raise ValueError("latent_cvae_horizon_tokens must be >= 1")
        if int(self.latent_cvae_stage_slots) < 1:
            raise ValueError("latent_cvae_stage_slots must be >= 1")
        if not (0.0 <= float(self.latent_cvae_stage_promote_scale_init) <= 1.0):
            raise ValueError("latent_cvae_stage_promote_scale_init must be in [0, 1]")
        if int(self.latent_cvae_hierarchical_workspace):
            if str(self.final_action_decoder) != "adaptive_recurrent_cvae_action":
                raise ValueError("hierarchical workspace requires adaptive_recurrent_cvae_action")
            if not int(self.latent_cvae_mmdit_decoder):
                raise ValueError("hierarchical workspace requires latent_cvae_mmdit_decoder=1")
            incompatible = (
                "latent_cvae_workspace_noisy_query",
                "latent_cvae_workspace_time_state",
                "latent_cvae_workspace_controller",
                "latent_cvae_workspace_progress_value",
                "adaptive_cvae_route_time_query",
                "adaptive_cvae_progress_memory",
                "adaptive_cvae_layer_routing",
                "adaptive_cvae_context_capsules",
            )
            enabled = [name for name in incompatible if int(getattr(self, name))]
            if enabled:
                raise ValueError(
                    "hierarchical workspace owns selection/state routing; disable incompatible paths: "
                    + ", ".join(enabled)
                )
        if str(self.final_action_decoder) in ("latent_cvae_action", "adaptive_recurrent_cvae_action"):
            if int(self.latent_cvae_layer_memory) and not int(self.layer_contract_adapters):
                raise ValueError(
                    f"{self.final_action_decoder} with latent_cvae_layer_memory=1 "
                    "requires layer_contract_adapters=1"
                )
        if int(self.adaptive_cvae_refine_steps) < 0:
            raise ValueError("adaptive_cvae_refine_steps must be >= 0")
        if (
            int(self.latent_cvae_mmdit_decoder)
            and str(self.final_action_decoder) == "adaptive_recurrent_cvae_action"
            and int(self.adaptive_cvae_refine_steps) < 1
        ):
            raise ValueError("adaptive MMDiT decoder requires adaptive_cvae_refine_steps >= 1")
        if int(self.adaptive_cvae_progress_steps) < 1:
            raise ValueError("adaptive_cvae_progress_steps must be >= 1")
        for name in (
            "adaptive_cvae_progress_memory",
            "adaptive_cvae_prefix_memory",
            "adaptive_cvae_layer_routing",
            "adaptive_cvae_prefix_detach",
            "adaptive_cvae_progress_z_injection",
            "adaptive_cvae_route_query_bias",
            "adaptive_cvae_route_time_query",
            "adaptive_cvae_token_semantic_adapter",
            "adaptive_cvae_output_adapter",
            "adaptive_cvae_function_adapters",
        ):
            if int(getattr(self, name)) not in (0, 1):
                raise ValueError(f"{name} must be 0 or 1")
        if int(self.adaptive_cvae_function_rank) < 1:
            raise ValueError("adaptive_cvae_function_rank must be >= 1")
        if int(self.adaptive_cvae_progress_role_dim) < 2:
            raise ValueError("adaptive_cvae_progress_role_dim must be >= 2")
        if str(self.final_action_decoder) == "adaptive_recurrent_cvae_action":
            if int(self.latent_cvae_layer_memory) and not int(self.layer_contract_adapters):
                raise ValueError("adaptive_recurrent_cvae_action with latent_cvae_layer_memory=1 requires layer_contract_adapters=1")
        if float(self.latent_cvae_output_init_std) < 0:
            raise ValueError("latent_cvae_output_init_std must be non-negative")
        if float(self.latent_cvae_mu_bound) < 0:
            raise ValueError("latent_cvae_mu_bound must be non-negative")
        if float(self.latent_cvae_min_std) < 0:
            raise ValueError("latent_cvae_min_std must be non-negative")
        if int(self.latent_cvae_causal_attention) not in (0, 1):
            raise ValueError("latent_cvae_causal_attention must be 0 or 1")
        if not (0.0 <= float(self.latent_cvae_noisy_gate_min) <= 1.0):
            raise ValueError("latent_cvae_noisy_gate_min must be in [0, 1]")
        if float(self.latent_cvae_noisy_gate_power) <= 0:
            raise ValueError("latent_cvae_noisy_gate_power must be positive")
        if float(self.latent_cvae_layer_scan_alpha) < 0:
            raise ValueError("latent_cvae_layer_scan_alpha must be non-negative")
        if not (0.0 <= float(self.latent_cvae_layer_grad_scale) <= 1.0):
            raise ValueError("latent_cvae_layer_grad_scale must be in [0, 1]")
        if float(self.latent_cvae_consequence_scale_max) <= 0:
            raise ValueError("latent_cvae_consequence_scale_max must be positive")
        if not (
            0.0
            <= float(self.latent_cvae_consequence_scale_init)
            <= float(self.latent_cvae_consequence_scale_max)
        ):
            raise ValueError("latent_cvae_consequence_scale_init must be in [0, max]")
        if int(self.adaptive_cvae_route_cosine) not in (0, 1):
            raise ValueError("adaptive_cvae_route_cosine must be 0 or 1")
        if float(self.adaptive_cvae_route_temperature) <= 0:
            raise ValueError("adaptive_cvae_route_temperature must be positive")
        if not (0.0 <= float(self.adaptive_cvae_context_dropout) < 1.0):
            raise ValueError("adaptive_cvae_context_dropout must be in [0, 1)")
        if not (0.0 <= float(self.adaptive_cvae_route_entropy_floor_ratio) <= 1.0):
            raise ValueError("adaptive_cvae_route_entropy_floor_ratio must be in [0, 1]")
        if int(self.adaptive_cvae_route_topk) < 0:
            raise ValueError("adaptive_cvae_route_topk must be >= 0")
        for name in (
            "adaptive_cvae_route_sparsemax",
            "adaptive_cvae_route_adaptive_temperature",
            "adaptive_cvae_role_query",
            "adaptive_cvae_step_roles",
            "adaptive_cvae_context_capsules",
            "adaptive_cvae_direct_condition_residual",
            "adaptive_cvae_condition_strength",
            "adaptive_cvae_micro_control",
            "adaptive_cvae_micro_refine_block",
            "adaptive_cvae_micro_supervision",
            "adaptive_cvae_micro_heun",
            "adaptive_cvae_micro_monotonic_progress",
        ):
            if int(getattr(self, name)) not in (0, 1):
                raise ValueError(f"{name} must be 0 or 1")
        if float(self.adaptive_cvae_route_min_temperature) <= 0:
            raise ValueError("adaptive_cvae_route_min_temperature must be positive")
        if float(self.adaptive_cvae_route_max_temperature) < float(self.adaptive_cvae_route_min_temperature):
            raise ValueError("adaptive_cvae_route_max_temperature must be >= min temperature")
        if int(self.adaptive_cvae_coarse_stride) < 1:
            raise ValueError("adaptive_cvae_coarse_stride must be >= 1")
        if not (0.0 <= float(self.adaptive_cvae_coarse_strength) <= 1.0):
            raise ValueError("adaptive_cvae_coarse_strength must be in [0, 1]")
        for name in ("adaptive_cvae_seed_scale", "adaptive_cvae_output_scale"):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if int(self.adaptive_cvae_context_capsule_count) < 1:
            raise ValueError("adaptive_cvae_context_capsule_count must be >= 1")
        if float(self.adaptive_cvae_condition_strength_min) < 0:
            raise ValueError("adaptive_cvae_condition_strength_min must be non-negative")
        if float(self.adaptive_cvae_condition_strength_max) < float(self.adaptive_cvae_condition_strength_min):
            raise ValueError("adaptive_cvae_condition_strength_max must be >= min")
        if not (
            float(self.adaptive_cvae_condition_strength_min)
            <= float(self.adaptive_cvae_condition_strength_init)
            <= float(self.adaptive_cvae_condition_strength_max)
        ):
            raise ValueError("adaptive_cvae_condition_strength_init must be within [min, max]")
        if float(self.adaptive_cvae_micro_min_step) < 0:
            raise ValueError("adaptive_cvae_micro_min_step must be non-negative")
        if float(self.adaptive_cvae_micro_max_step) < float(self.adaptive_cvae_micro_min_step):
            raise ValueError("adaptive_cvae_micro_max_step must be >= min step")
        if not (
            float(self.adaptive_cvae_micro_min_step)
            <= float(self.adaptive_cvae_micro_step_init)
            <= float(self.adaptive_cvae_micro_max_step)
        ):
            raise ValueError("adaptive_cvae_micro_step_init must be within [min, max]")
        if float(self.adaptive_cvae_micro_kp_max) < 0 or float(self.adaptive_cvae_micro_kd_max) < 0:
            raise ValueError("adaptive_cvae_micro gain maxima must be non-negative")
        if not (0.0 <= float(self.adaptive_cvae_micro_kp_init) <= float(self.adaptive_cvae_micro_kp_max)):
            raise ValueError("adaptive_cvae_micro_kp_init must be within [0, kp_max]")
        if not (0.0 <= float(self.adaptive_cvae_micro_kd_init) <= float(self.adaptive_cvae_micro_kd_max)):
            raise ValueError("adaptive_cvae_micro_kd_init must be within [0, kd_max]")
        if float(self.adaptive_cvae_micro_update_scale) < 0:
            raise ValueError("adaptive_cvae_micro_update_scale must be non-negative")
        if float(self.adaptive_cvae_micro_refine_block_scale) < 0:
            raise ValueError("adaptive_cvae_micro_refine_block_scale must be non-negative")
        if float(self.adaptive_cvae_micro_progress_distance_scale) < 0:
            raise ValueError("adaptive_cvae_micro_progress_distance_scale must be non-negative")
        if int(self.latent_action_near_depth) < 1 or int(self.latent_action_mid_depth) < 1:
            raise ValueError("latent_action_near_depth/mid_depth must be >= 1")
        if int(self.latent_action_near_depth) > int(self.latent_action_mid_depth):
            raise ValueError("latent_action_near_depth must be <= latent_action_mid_depth")
        if int(self.latent_action_mid_depth) > int(self.latent_action_decoder_depth):
            raise ValueError("latent_action_mid_depth cannot exceed latent_action_decoder_depth")
        if int(self.layer_recurrent_consequence) and int(self.layer_consequence_steps) != int(self.future_anchors):
            raise ValueError(
                "layer_consequence_steps must equal future_anchors while the milestone losses use one target per anchor"
            )
        if int(self.layer_recurrent_consequence) and int(self.layer_consequence_steps) > int(self.action_horizon):
            raise ValueError("layer_consequence_steps cannot exceed action_horizon")


def _align_milestone_tokens_to_horizon(tokens: Tensor, horizon: int) -> Tensor:
    """Expand one pooled token per action segment onto the action timeline."""

    if tokens.ndim != 3:
        raise ValueError(f"milestone tokens must be [B,K,H], got {tuple(tokens.shape)}")
    horizon = int(horizon)
    steps = int(tokens.shape[1])
    if horizon < 1 or steps < 1 or steps > horizon:
        raise ValueError(f"expected 1 <= milestone steps <= horizon, got steps={steps} horizon={horizon}")
    rows: list[Tensor] = []
    for step in range(steps):
        lo = int(round(step * horizon / float(steps)))
        hi = int(round((step + 1) * horizon / float(steps)))
        hi = max(hi, lo + 1)
        hi = min(hi, horizon)
        rows.append(tokens[:, step:step + 1].expand(-1, hi - lo, -1))
    aligned = torch.cat(rows, dim=1)
    if aligned.shape[1] != horizon:
        raise RuntimeError(f"milestone alignment produced {aligned.shape[1]} tokens for horizon={horizon}")
    return aligned


def _rollout_tokens_to_action_horizon(tokens: Tensor, config: V39PolicyConfig) -> Tensor:
    """Pool rollout spatial tokens per anchor, then align anchors to action time."""

    if tokens.ndim != 3:
        raise ValueError(f"rollout tokens must be [B,F*G,H], got {tuple(tokens.shape)}")
    grid = int(config.num_cameras) * int(config.future_grid_size) * int(config.future_grid_size)
    expected = int(config.future_anchors) * grid
    if int(tokens.shape[1]) != expected:
        raise ValueError(f"rollout token count must be future_anchors*grid={expected}, got {tokens.shape[1]}")
    milestones = tokens.reshape(
        tokens.shape[0], int(config.future_anchors), grid, tokens.shape[-1]
    ).mean(dim=2)
    return _align_milestone_tokens_to_horizon(milestones, int(config.action_horizon))


class MidcutContractHeads(nn.Module):
    """Intentionally weak readouts from the DiT midpoint.

    The heads are deliberately no stronger than LayerNorm + Linear.  If these
    heads cannot read motion/event/future information, the information is not
    sufficiently explicit at the mid-cut latent.
    """

    def __init__(self, config: V39PolicyConfig) -> None:
        super().__init__()
        self.config = config
        h = int(config.hidden_size)
        self.action_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, config.physical_action_dim))
        self.event_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 3))
        self.motion_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 1))
        self.rollout_effect_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
        self.rollout_delta_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
        self.transition_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
        self.future_gain = nn.Parameter(torch.tensor(float(config.midcut_future_gain_init), dtype=torch.float32))
        # Start action/event readouts small but not exactly zero.  A fully
        # zero final Linear makes the first backward step update only the head
        # itself and gives essentially no gradient to the upstream latent.
        # Small random init keeps the head weak while allowing the contract
        # loss to shape the DiT canvas from the beginning.
        for module in (self.action_head[-1], self.event_head[-1], self.motion_head[-1]):
            nn.init.normal_(module.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(module.bias)
        for module in (self.rollout_effect_head[-1], self.rollout_delta_head[-1], self.transition_head[-1]):
            nn.init.normal_(module.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(module.bias)

    def trajectory_pooled(self, trajectory_tokens: Tensor) -> Tensor:
        cfg = self.config
        b = trajectory_tokens.shape[0]
        grouped = trajectory_tokens.reshape(b, cfg.action_horizon, cfg.action_basis_tokens, cfg.hidden_size)
        return grouped.mean(dim=2)

    def forward(self, canvas: Tensor, slices: dict[str, slice]) -> dict[str, Tensor]:
        cfg = self.config
        trajectory = canvas[:, slices["trajectory"]]
        rollout = canvas[:, slices["rollout"]]
        registers = canvas[:, slices["registers"]]
        trajectory_pooled = self.trajectory_pooled(trajectory)
        gain = self.future_gain.to(device=canvas.device, dtype=canvas.dtype)
        effect = self.rollout_effect_head(rollout) * gain
        delta = self.rollout_delta_head(rollout) * gain
        event_context = _rollout_tokens_to_action_horizon(delta, cfg)
        transition_base = delta.mean(dim=1, keepdim=True)
        transition = self.transition_head(transition_base).expand(-1, cfg.action_horizon, -1)
        return {
            "midcut_canvas_tokens": canvas,
            "midcut_trajectory_tokens": trajectory,
            "midcut_rollout_tokens": rollout,
            "midcut_register_tokens": registers,
            "midcut_state_tokens": canvas[:, slices["state"]],
            "midcut_state_history_tokens": canvas[:, slices["state_history"]],
            "midcut_executed_tokens": canvas[:, slices["executed"]],
            "midcut_proposal_tokens": canvas[:, slices["proposal"]],
            "midcut_trajectory_pooled": trajectory_pooled,
            "midcut_pred_physical_velocity": self.action_head(trajectory_pooled),
            "midcut_direct_physical_velocity": self.action_head(trajectory_pooled),
            "midcut_rollout_residual_velocity": torch.zeros(
                trajectory_pooled.shape[0], cfg.action_horizon, cfg.physical_action_dim,
                device=trajectory_pooled.device, dtype=trajectory_pooled.dtype,
            ),
            "midcut_rollout_alpha": torch.zeros(1, cfg.action_horizon, 1, device=trajectory_pooled.device, dtype=trajectory_pooled.dtype),
            "midcut_rollout_effect_pred": effect,
            "midcut_rollout_delta_pred": delta,
            "midcut_rollout_base_effect_pred": torch.zeros_like(effect),
            "midcut_event_logits": self.event_head(event_context),
            "midcut_motion_logits": self.motion_head(trajectory_pooled).squeeze(-1),
            "midcut_transition_latent": transition,
            "midcut_rollout_delta_norm": delta.detach().float().norm(dim=-1).mean(),
            "midcut_rollout_effect_norm": effect.detach().float().norm(dim=-1).mean(),
            "midcut_future_gain": gain.detach().float().abs(),
        }


class LayerContractAdapterHeads(nn.Module):
    """Tiny per-layer adapter contract for V39.1.

    It first applies a small bottleneck residual adapter, then reuses the same
    deliberately weak readout family as the mid-cut contract.  The adapter keeps
    the probe local and cheap; the heads stay too weak to manufacture motion or
    contact structure after the trunk.
    """

    def __init__(self, config: V39PolicyConfig, *, layer_index: int) -> None:
        super().__init__()
        self.config = config
        self.layer_index = int(layer_index)
        h = int(config.hidden_size)
        b = int(config.layer_contract_adapter_dim)
        self.adapter = nn.Sequential(
            nn.LayerNorm(h),
            nn.Linear(h, b),
            nn.GELU(),
            nn.Linear(b, h),
        )
        nn.init.normal_(self.adapter[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.adapter[-1].bias)
        self.readout = MidcutContractHeads(config)

    def forward(self, canvas: Tensor, slices: dict[str, slice]) -> dict[str, Tensor]:
        scale = torch.as_tensor(
            float(self.config.layer_contract_residual_scale),
            device=canvas.device,
            dtype=canvas.dtype,
        )
        adapted = canvas + scale * self.adapter(canvas)
        mid = self.readout(adapted, slices)
        out: dict[str, Tensor] = {
            key[len("midcut_"):]: value for key, value in mid.items() if key.startswith("midcut_")
        }
        out["layer_index"] = torch.as_tensor(self.layer_index, device=canvas.device, dtype=torch.long)
        return out


class SharedLayerFlowActionProbe(nn.Module):
    """Shared lightweight flow-matching action probe for V39.2.

    Each per-layer adapter first predicts a world/future latent.  This probe then
    reads only the layer-local latent summaries plus the current noisy physical
    action and flow time.  The parameters are shared across layers so lower loss
    identifies a better latent layer rather than a stronger per-layer action
    decoder.
    """

    def __init__(self, config: V39PolicyConfig) -> None:
        super().__init__()
        self.config = config
        h = int(config.hidden_size)
        ph = int(config.physical_action_dim)
        mid = int(config.layer_fm_probe_hidden)
        self.noisy_proj = nn.Linear(ph, h)
        self.latent_proj = nn.Sequential(nn.LayerNorm(2 * h), nn.Linear(2 * h, h), nn.SiLU(), nn.Linear(h, h))
        self.time = TimeEmbedding(h)
        self.net = nn.Sequential(
            nn.LayerNorm(h),
            nn.Linear(h, mid),
            nn.SiLU(),
            nn.Linear(mid, ph),
        )
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        *,
        trajectory_pooled: Tensor,
        rollout_effect_pred: Tensor,
        rollout_delta_pred: Tensor,
        noisy_physical: Tensor,
        time: Tensor,
    ) -> Tensor:
        if noisy_physical.shape[:2] != trajectory_pooled.shape[:2]:
            raise ValueError(
                f"noisy_physical and trajectory_pooled horizon mismatch: "
                f"{tuple(noisy_physical.shape)} vs {tuple(trajectory_pooled.shape)}"
            )
        latent_summary = torch.cat(
            [rollout_effect_pred.mean(dim=1), rollout_delta_pred.mean(dim=1)],
            dim=-1,
        )
        latent_bias = self.latent_proj(latent_summary).to(dtype=trajectory_pooled.dtype)[:, None, :]
        t = self.time(time.to(dtype=trajectory_pooled.dtype)).to(dtype=trajectory_pooled.dtype)[:, None, :]
        x = self.noisy_proj(noisy_physical.to(dtype=trajectory_pooled.dtype)) + trajectory_pooled + latent_bias + t
        return self.net(x)




class LayerRoleScheduler(nn.Module):
    """Deterministic layer-role schedule for V40 latent/causal contracts.

    Lower layers are expected to expose action-sensitive local transition deltas;
    upper layers are expected to expose stable world/future latents.  The schedule
    returns scalar gains used both for prediction mixing and for diagnostics.
    """

    def __init__(self, config: V39PolicyConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, layer_index: int | Tensor, *, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        count = max(int(self.config.depth) - 1, 1)
        if torch.is_tensor(layer_index):
            idx = layer_index.to(device=device, dtype=dtype)
        else:
            idx = torch.as_tensor(float(layer_index), device=device, dtype=dtype)
        progress = (idx / float(count)).clamp(0.0, 1.0)
        c_low = float(self.config.layer_low_causal_weight)
        c_high = float(self.config.layer_high_causal_weight)
        l_low = float(self.config.layer_low_latent_weight)
        l_high = float(self.config.layer_high_latent_weight)
        causal = c_low + (c_high - c_low) * progress
        latent = l_low + (l_high - l_low) * progress
        return causal, latent



class UnifiedInterventionBlock(nn.Module):
    """One light state-action interaction block for V40.1.

    The block is deliberately not a second DiT.  It performs one cross-attention
    step from grid-local intervention state into compact context tokens, followed
    by a small FFN.  Setting ``layer_causal_feedback_depth=0`` bypasses these
    blocks and leaves the FiLM-gated delta path as the main transition operator.
    """

    def __init__(self, hidden: int, heads: int, mid: int) -> None:
        super().__init__()
        self.qn = nn.LayerNorm(hidden)
        self.kn = nn.LayerNorm(hidden)
        self.cross = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.fn = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(nn.Linear(hidden, mid), nn.SiLU(), nn.Linear(mid, hidden))
        nn.init.normal_(self.ffn[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.ffn[-1].bias)

    def forward(self, state: Tensor, context: Tensor) -> Tensor:
        update, _ = self.cross(self.qn(state), self.kn(context), self.kn(context), need_weights=False)
        state = state + update
        state = state + self.ffn(self.fn(state)).to(dtype=state.dtype)
        return state


class RecurrentMilestoneConsequenceCell(nn.Module):
    """V40.1 unified intervention-latent encoder.

    Public name is preserved for checkpoint/CLI compatibility, but the object is
    no longer a separate action-only consequence head.  It is a single
    intervention-latent head that jointly encodes:

    * layer-local rollout/world tokens;
    * current state token and state-history tokens;
    * executed-action history tokens;
    * optional trajectory/proposal canvas tokens;
    * candidate future action segments.

    It emits an action-conditioned residual latent.  The residual is supervised
    by future-latent targets, while action and state counterfactual views test
    whether the same unified head really depends on both the intervention and
    the originating state/frame context.
    """

    def __init__(self, config: V39PolicyConfig) -> None:
        super().__init__()
        self.config = config
        h = int(config.hidden_size)
        ph = int(config.physical_action_dim)
        mid = int(config.layer_consequence_hidden)
        self.gripper_frame = (
            ParsevalGripperTemporalFrame(config.action_horizon, config.gripper_field_dim)
            if str(getattr(config, "gripper_field_mode", "legacy_handcrafted")) == "parseval_temporal"
            else None
        )
        semantic_ph = 2 * int(config.arm_dim) + 1 if self.gripper_frame is not None else ph
        self.action_summary_dim = semantic_ph * 5 + 4
        self.action_encoder = nn.Sequential(
            nn.LayerNorm(self.action_summary_dim),
            nn.Linear(self.action_summary_dim, mid),
            nn.SiLU(),
            nn.Linear(mid, h),
        )
        self.step_embed = nn.Embedding(int(config.layer_consequence_steps), h)
        self.layer_embed = nn.Embedding(int(config.depth), h)
        self.memory_tokens = nn.Parameter(torch.randn(1, int(config.layer_causal_memory_tokens), h) * 0.02)
        self.context_proj = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
        self.action_film = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, mid), nn.SiLU(), nn.Linear(mid, 2 * h))
        self.context_gate = nn.Sequential(nn.LayerNorm(2 * h), nn.Linear(2 * h, mid), nn.SiLU(), nn.Linear(mid, 1))
        self.delta_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, mid), nn.SiLU(), nn.Linear(mid, h))
        self.neutral_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, mid), nn.SiLU(), nn.Linear(mid, h))
        self.policy_effect_proj = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, mid), nn.SiLU(), nn.Linear(mid, h))
        self.interaction_blocks = nn.ModuleList([
            UnifiedInterventionBlock(h, int(config.num_heads), mid)
            for _ in range(int(config.layer_causal_feedback_depth))
        ])
        self.effect_norm = nn.LayerNorm(h)
        self.effect_gain = nn.Parameter(torch.tensor(float(config.layer_consequence_initial_gain), dtype=torch.float32))
        self.delta_scale = nn.Parameter(torch.tensor(float(config.layer_consequence_delta_scale), dtype=torch.float32))
        for module in (
            self.action_encoder[-1], self.context_proj[-1], self.action_film[-1],
            self.context_gate[-1], self.delta_head[-1], self.neutral_head[-1], self.policy_effect_proj[-1],
        ):
            nn.init.normal_(module.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(module.bias)

    def _segment_action(self, action_physical: Tensor) -> Tensor:
        cfg = self.config
        k = int(cfg.layer_consequence_steps)
        if self.gripper_frame is not None:
            ad = int(cfg.arm_dim)
            gripper_field = action_physical[..., 2 * ad :]
            action_physical = torch.cat(
                [action_physical[..., : 2 * ad], self.gripper_frame.synthesis(gripper_field)],
                dim=-1,
            )
        b, horizon, ph = action_physical.shape
        if horizon <= 0:
            raise ValueError("action_physical horizon must be positive")
        rows: list[Tensor] = []
        for step in range(k):
            lo = int(round(step * horizon / float(k)))
            hi = int(round((step + 1) * horizon / float(k)))
            hi = max(hi, lo + 1)
            hi = min(hi, horizon)
            seg = action_physical[:, lo:hi]
            mean = seg.mean(dim=1)
            first = seg[:, 0]
            last = seg[:, -1]
            delta = last - first
            std = seg.float().std(dim=1, unbiased=False).to(dtype=action_physical.dtype)
            ad = int(getattr(cfg, "arm_dim", max((ph - 2) // 2, 0)))
            if ad > 0 and 2 * ad + 2 == ph:
                # action_physical is [arm_abs, arm_delta, gripper_value, gripper_delta].
                grip_value = 2 * ad
                grip_mean = seg[..., grip_value].mean(dim=1, keepdim=True)
                grip_delta = last[:, grip_value:grip_value + 1] - first[:, grip_value:grip_value + 1]
                arm = seg[..., : 2 * ad]
            else:
                g = int(cfg.gripper_dim_index)
                if g < 0:
                    g += ph
                g = min(max(g, 0), ph - 1)
                grip_mean = seg[..., g].mean(dim=1, keepdim=True)
                grip_delta = last[:, g:g + 1] - first[:, g:g + 1]
                arm = torch.cat([seg[..., :g], seg[..., g + 1:]], dim=-1) if ph > 1 else seg[..., :0]
            arm_norm = arm.float().norm(dim=-1).mean(dim=1, keepdim=True).to(dtype=action_physical.dtype) if arm.numel() else torch.zeros(b, 1, device=action_physical.device, dtype=action_physical.dtype)
            action_norm = seg.float().norm(dim=-1).mean(dim=1, keepdim=True).to(dtype=action_physical.dtype)
            rows.append(torch.cat([mean, first, last, delta, std, grip_mean, grip_delta, arm_norm, action_norm], dim=-1))
        return torch.stack(rows, dim=1)

    def _compact_tokens(self, x: Tensor | None, *, max_tokens: int = 8) -> Tensor | None:
        if x is None:
            return None
        if x.ndim != 3:
            raise ValueError(f"context tokens must be [B,N,H], got {tuple(x.shape)}")
        if x.shape[1] <= max_tokens:
            return x
        # Uniform deterministic subsampling keeps the head lightweight while
        # still excluding more than a single frame/state token in counterfactuals.
        idx = torch.linspace(0, x.shape[1] - 1, steps=max_tokens, device=x.device).round().long()
        return x.index_select(1, idx)

    def _context_bank(
        self,
        *,
        base_tokens: Tensor,
        state_tokens: Tensor | None,
        state_history_tokens: Tensor | None,
        executed_tokens: Tensor | None,
        trajectory_tokens: Tensor | None,
        proposal_tokens: Tensor | None,
        action_token: Tensor,
        layer_token: Tensor,
    ) -> tuple[Tensor, Tensor]:
        b = base_tokens.shape[0]
        mem = self.memory_tokens.to(device=base_tokens.device, dtype=base_tokens.dtype).expand(b, -1, -1)
        parts = [
            base_tokens,
            self._compact_tokens(state_tokens, max_tokens=2),
            self._compact_tokens(state_history_tokens, max_tokens=4),
            self._compact_tokens(executed_tokens, max_tokens=4),
            self._compact_tokens(proposal_tokens, max_tokens=4),
            self._compact_tokens(trajectory_tokens, max_tokens=8),
            action_token[:, None, :],
            layer_token[:, None, :],
            mem,
        ]
        kept = [p for p in parts if p is not None]
        bank = self.context_proj(torch.cat(kept, dim=1)).to(dtype=base_tokens.dtype)
        # Pool each semantic group before averaging groups.  This prevents the
        # spatial rollout grid from numerically overwhelming the much shorter
        # state/history groups and keeps explicit context active even when the
        # optional cross-attention feedback depth is zero.
        grouped = torch.stack([part.mean(dim=1) for part in kept], dim=1)
        summary = self.context_proj(grouped).mean(dim=1).to(dtype=base_tokens.dtype)
        return bank, summary

    @staticmethod
    def _align_milestone_tokens_to_horizon(tokens: Tensor, horizon: int) -> Tensor:
        return _align_milestone_tokens_to_horizon(tokens, horizon)

    def forward(
        self,
        *,
        rollout_tokens: Tensor,
        action_physical: Tensor,
        state_tokens: Tensor | None = None,
        state_history_tokens: Tensor | None = None,
        executed_tokens: Tensor | None = None,
        trajectory_tokens: Tensor | None = None,
        proposal_tokens: Tensor | None = None,
        layer_index: int | Tensor | None = None,
    ) -> dict[str, Tensor]:
        cfg = self.config
        b = int(rollout_tokens.shape[0])
        k = int(cfg.layer_consequence_steps)
        grid = int(cfg.num_cameras) * int(cfg.future_grid_size) * int(cfg.future_grid_size)
        h = int(cfg.hidden_size)
        if rollout_tokens.shape[1] != int(cfg.future_token_count):
            raise ValueError(
                f"rollout_tokens must have future_token_count={cfg.future_token_count}, got {rollout_tokens.shape[1]}"
            )
        grouped = rollout_tokens.reshape(b, int(cfg.future_anchors), grid, h)
        action_segments = self._segment_action(action_physical.to(device=rollout_tokens.device, dtype=rollout_tokens.dtype))
        action_embed = self.action_encoder(action_segments).to(dtype=rollout_tokens.dtype)
        step_ids = torch.arange(k, device=rollout_tokens.device)
        step_embed = self.step_embed(step_ids).to(dtype=rollout_tokens.dtype)
        if layer_index is None:
            layer_id = torch.zeros((), device=rollout_tokens.device, dtype=torch.long)
        elif torch.is_tensor(layer_index):
            layer_id = layer_index.to(device=rollout_tokens.device, dtype=torch.long).clamp(0, int(cfg.depth) - 1)
        else:
            layer_id = torch.as_tensor(int(layer_index), device=rollout_tokens.device, dtype=torch.long).clamp(0, int(cfg.depth) - 1)
        layer_token = self.layer_embed(layer_id)[None].expand(b, -1).to(dtype=rollout_tokens.dtype)
        scale = self.delta_scale.to(device=rollout_tokens.device, dtype=rollout_tokens.dtype).abs()
        gain = self.effect_gain.to(device=rollout_tokens.device, dtype=rollout_tokens.dtype).abs()
        effect_state = torch.zeros(b, grid, h, device=rollout_tokens.device, dtype=rollout_tokens.dtype)
        preds: list[Tensor] = []
        deltas: list[Tensor] = []
        gates: list[Tensor] = []
        policy_tokens: list[Tensor] = []
        neutral_tokens: list[Tensor] = []
        intervene_tokens: list[Tensor] = []
        for step in range(k):
            # Validation requires one intervention step per future anchor, so
            # predictions and targets share the same temporal indexing.
            anchor = step
            base = grouped[:, anchor]
            a = action_embed[:, step] + step_embed[step][None] + layer_token
            context, context_summary = self._context_bank(
                base_tokens=base,
                state_tokens=state_tokens,
                state_history_tokens=state_history_tokens,
                executed_tokens=executed_tokens,
                trajectory_tokens=trajectory_tokens,
                proposal_tokens=proposal_tokens,
                action_token=a,
                layer_token=layer_token,
            )
            neutral = base + self.neutral_head(base).to(dtype=rollout_tokens.dtype)
            intervention = neutral + effect_state
            for block in self.interaction_blocks:
                intervention = block(intervention, context)
            joint_condition = a + context_summary
            gamma_beta = self.action_film(joint_condition).to(dtype=rollout_tokens.dtype)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            modulated = intervention * (1.0 + gamma[:, None, :]) + beta[:, None, :]
            gate_in = torch.cat([modulated, joint_condition[:, None, :].expand(-1, grid, -1)], dim=-1)
            gate = torch.sigmoid(self.context_gate(gate_in).to(dtype=rollout_tokens.dtype))
            raw_delta = torch.tanh(self.delta_head(modulated).to(dtype=rollout_tokens.dtype))
            # V40.1 keeps the local/cumulative contract closed, but restores the
            # normalized increment used by the earlier K4/A6 branch.  The
            # unnormalized gated delta is often too small for action-shuffle
            # contrast to see; LayerNorm provides a per-token direction
            # amplifier.  Crucially, the *same* increment is logged/supervised as
            # milestone_step_delta_pred and accumulated into rollout_effect_pred,
            # so delta matching and cumulative rollout remain mathematically
            # consistent.
            local_delta = scale * gate * raw_delta
            step_delta = gain * self.effect_norm(local_delta).to(dtype=rollout_tokens.dtype)
            effect_state = effect_state + step_delta
            z_intervene = neutral + effect_state
            preds.append(effect_state)
            deltas.append(step_delta)
            gates.append(gate)
            policy_tokens.append(self.policy_effect_proj(z_intervene).to(dtype=rollout_tokens.dtype))
            neutral_tokens.append(neutral)
            intervene_tokens.append(z_intervene)
        pred = torch.stack(preds, dim=1)
        delta_stack = torch.stack(deltas, dim=1)
        gate_stack = torch.stack(gates, dim=1)
        policy_stack = torch.stack(policy_tokens, dim=1)
        neutral_stack = torch.stack(neutral_tokens, dim=1)
        intervene_stack = torch.stack(intervene_tokens, dim=1)
        flat_pred = pred.reshape(b, k * grid, h)
        flat_delta = delta_stack.reshape(b, k * grid, h)
        flat_policy = policy_stack.reshape(b, k * grid, h)
        time_policy = self._align_milestone_tokens_to_horizon(
            policy_stack.mean(dim=2), int(cfg.action_horizon)
        )
        return {
            "milestone_rollout_effect_pred": flat_pred,
            "milestone_rollout_delta_pred": flat_pred,
            "milestone_step_delta_pred": flat_delta,
            "milestone_policy_effect_tokens": flat_policy,
            "milestone_policy_time_tokens": time_policy,
            "milestone_neutral_latent_pred": neutral_stack.reshape(b, k * grid, h),
            "milestone_intervention_latent_pred": intervene_stack.reshape(b, k * grid, h),
            "milestone_gate_mean": gate_stack.detach().float().mean(),
            "milestone_step_delta_norm": delta_stack.detach().float().norm(dim=-1).mean(),
            "milestone_effect_norm": pred.detach().float().norm(dim=-1).mean(),
            "milestone_effect_std": pred.detach().float().std(unbiased=False),
            "milestone_effect_gain": gain.detach().float().abs(),
        }

def _scaled_contract_view(x: Tensor, scale: float) -> Tensor:
    """Return x with gradients scaled by ``scale`` but values unchanged."""

    if scale >= 1.0:
        return x
    if scale <= 0.0:
        return x.detach()
    return x.detach() + float(scale) * (x - x.detach())


def _zeros_like_scalar(reference: Tensor) -> Tensor:
    return torch.zeros((), device=reference.device, dtype=reference.dtype)



class V37StyleResidualActionBlock(nn.Module):
    """V37-style action/high/event token block for residual refinement.

    This deliberately reuses the useful V37 pattern: a compact set of high-level
    slots, horizon action tokens, and event tokens exchange information through
    self-attention, then cross-attend to a high-bandwidth memory bank.  Unlike
    the failed full action-flow replacement, this block is downstream of the
    legacy V40.1 head and only predicts a small zero-initialized residual.
    """

    def __init__(self, config: V39PolicyConfig) -> None:
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
        self.mod = nn.Linear(h, 9 * h)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    @staticmethod
    def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        return x * (1 + scale[:, None]) + shift[:, None]

    def forward(self, x: Tensor, memory: Tensor, time_emb: Tensor) -> Tensor:
        sa_s, sa_c, sa_g, ca_s, ca_c, ca_g, ff_s, ff_c, ff_g = self.mod(time_emb).chunk(9, dim=-1)
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


class V37StyleResidualActionFlowDenoiser(nn.Module):
    """Zero-start residual action-flow denoiser.

    The legacy V40.1 final velocity remains the base policy.  This module reads
    the same noisy physical action plus V40/V37-style latent memory and emits a
    residual physical velocity.  Its final velocity and event heads are
    zero-initialized, so a stable checkpoint is behavior-preserving at load time
    while gradients can immediately train the residual heads.
    """

    def __init__(self, config: V39PolicyConfig) -> None:
        super().__init__()
        self.config = config
        h = int(config.hidden_size)
        self.time = TimeEmbedding(h)
        self.high_slots = int(config.action_flow_residual_high_slots)
        self.horizon_query = nn.Parameter(torch.randn(1, int(config.action_horizon), h) * 0.02)
        self.high_query = nn.Parameter(torch.randn(1, self.high_slots, h) * 0.02)
        self.event_query = nn.Parameter(torch.randn(1, int(config.event_tokens), h) * 0.02)
        self.noisy_action_lift = nn.Sequential(
            nn.LayerNorm(int(config.physical_action_dim)),
            nn.Linear(int(config.physical_action_dim), h),
        )
        self.trajectory_seed = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
        self.memory_summary = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h))
        self.blocks = nn.ModuleList([V37StyleResidualActionBlock(config) for _ in range(int(config.action_flow_residual_depth))])
        self.velocity_head = TransitionAwarePhysicalVelocityHead(config)
        self.event_delta_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 3))
        self.motion_delta_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 1))
        self._zero_initialize_outputs()
        alpha = torch.full((int(config.action_horizon), 1), float(config.action_flow_residual_max_scale), dtype=torch.float32)
        first = max(int(config.first_execution_steps), 1)
        mid = max(int(config.mid_execution_steps), first)
        for i in range(int(config.action_horizon)):
            step = i + 1
            if step <= first:
                alpha[i, 0] = float(config.action_flow_residual_max_scale) * 0.25
            elif step <= mid:
                frac = float(step - first) / float(max(mid - first, 1))
                alpha[i, 0] = float(config.action_flow_residual_max_scale) * (0.25 + 0.75 * frac)
        self.register_buffer("residual_alpha", alpha[None], persistent=True)

    def _make_temporal_action_update_mask(self, config: V39PolicyConfig) -> Tensor:
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
        for seq in (self.event_delta_head, self.motion_delta_head):
            last = seq[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

    def forward(
        self,
        *,
        noisy_physical: Tensor,
        time: Tensor,
        trajectory_pooled: Tensor,
        memory: Tensor,
    ) -> dict[str, Tensor]:
        cfg = self.config
        batch = int(noisy_physical.shape[0])
        dtype = noisy_physical.dtype
        device = noisy_physical.device
        mem_summary = self.memory_summary(memory.mean(dim=1))
        action_tokens = (
            self.horizon_query.to(device=device, dtype=dtype).expand(batch, -1, -1)
            + self.noisy_action_lift(noisy_physical)
            + self.trajectory_seed(trajectory_pooled)
            + mem_summary[:, None]
        )
        high_tokens = self.high_query.to(device=device, dtype=dtype).expand(batch, -1, -1) + mem_summary[:, None]
        event_tokens = self.event_query.to(device=device, dtype=dtype).expand(batch, -1, -1) + mem_summary[:, None]
        tokens = torch.cat([high_tokens, action_tokens, event_tokens], dim=1)
        high_slice = slice(0, self.high_slots)
        action_slice = slice(self.high_slots, self.high_slots + int(cfg.action_horizon))
        event_slice = slice(self.high_slots + int(cfg.action_horizon), self.high_slots + int(cfg.action_horizon) + int(cfg.event_tokens))
        time_emb = self.time(time.to(dtype=dtype)) + mem_summary
        for block in self.blocks:
            tokens = block(tokens, memory, time_emb)
        high = tokens[:, high_slice]
        action = tokens[:, action_slice]
        event = tokens[:, event_slice]
        transition = high.mean(dim=1, keepdim=True).expand(-1, int(cfg.action_horizon), -1)
        # Let event tokens influence the transition latent without making event
        # logits a detached side branch, unlike the original V37 implementation.
        transition = transition + event.mean(dim=1, keepdim=True).expand_as(transition)
        raw_residual = self.velocity_head(action, transition)
        alpha = self.residual_alpha.to(device=device, dtype=dtype)
        residual_velocity = raw_residual * alpha
        return {
            "residual_velocity": residual_velocity,
            "raw_residual_velocity": raw_residual,
            "residual_alpha": alpha,
            "event_delta_logits": self.event_delta_head(action),
            "motion_delta_logits": self.motion_delta_head(action).squeeze(-1),
            "action_tokens": action,
            "high_tokens": high,
            "event_tokens": event,
            "transition_latent": transition,
            "residual_norm": residual_velocity.detach().float().norm(dim=-1).mean(),
            "raw_residual_norm": raw_residual.detach().float().norm(dim=-1).mean(),
            "alpha_mean": alpha.detach().float().mean(),
        }


def _parse_layer_pair_schedule(spec: str, *, decoder_depth: int, num_layers: int) -> list[tuple[int, int]]:
    """Parse a compact layer-pair schedule like ``0:1,1:3,3:5,5:7``.

    Pairs are clamped to available V40 layers.  If fewer pairs than residual
    blocks are provided, the last pair is repeated.  This keeps command-line
    experimentation simple while making the default explicitly hierarchical.
    """

    depth = max(int(decoder_depth), 1)
    layers = max(int(num_layers), 1)
    pairs: list[tuple[int, int]] = []
    for chunk in str(spec or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"invalid layer pair '{chunk}', expected A:B")
        left, right = chunk.split(":", 1)
        try:
            a = int(left)
            b = int(right)
        except ValueError as exc:
            raise ValueError(f"invalid layer pair '{chunk}', expected integer A:B") from exc
        a = min(max(a, 0), layers - 1)
        b = min(max(b, 0), layers - 1)
        pairs.append((a, b))
    if not pairs:
        if depth == 1:
            pairs = [(0, layers - 1)]
        else:
            pairs = []
            for j in range(depth):
                a = round(j * (layers - 1) / max(depth, 1))
                b = round((j + 1) * (layers - 1) / max(depth, 1))
                pairs.append((min(a, layers - 1), min(max(b, a), layers - 1)))
    while len(pairs) < depth:
        pairs.append(pairs[-1])
    return pairs[:depth]


class LayeredV37StyleResidualActionFlowDenoiser(nn.Module):
    """Layer-pair progressive V37 residual action-flow denoiser.

    This is the non-hand-wavy version of hierarchical injection:

    * keep the stable V40.1 legacy velocity as the base policy;
    * collect token-level memories from every V40.1 contract layer;
    * for residual block j, build a memory from bottom anchor L0, pair La/Lb,
      and the token-level difference Lb-La;
    * optionally add a learnable local stage router initialized near the pair,
      so the model can move the hierarchy if the learned V40 layers do not match
      our hand-written schedule;
    * zero-initialize residual/event/motion heads so loading a stable checkpoint
      is behavior-preserving at step 0.
    """

    _LAYER_KEYS = (
        "rollout_tokens",
        "trajectory_pooled",
        "rollout_effect_pred",
        "rollout_delta_pred",
        "policy_effect_tokens",
        "policy_effect_time_tokens",
        "unified_intervention_latent_pred",
        "neutral_latent_pred",
        "milestone_step_delta_pred",
    )

    def __init__(self, config: V39PolicyConfig) -> None:
        super().__init__()
        self.config = config
        h = int(config.hidden_size)
        self.hidden_size = h
        self.depth = int(config.action_flow_residual_depth)
        self.high_slots = int(config.action_flow_residual_high_slots)
        self.time = TimeEmbedding(h)
        self.horizon_query = nn.Parameter(torch.randn(1, int(config.action_horizon), h) * 0.02)
        self.high_query = nn.Parameter(torch.randn(1, self.high_slots, h) * 0.02)
        self.event_query = nn.Parameter(torch.randn(1, int(config.event_tokens), h) * 0.02)
        self.noisy_action_lift = nn.Sequential(
            nn.LayerNorm(int(config.physical_action_dim)),
            nn.Linear(int(config.physical_action_dim), h),
        )
        self.trajectory_seed = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
        self.global_summary = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h))
        self.stage_summary = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h))
            for _ in range(self.depth)
        ])
        self.blocks = nn.ModuleList([V37StyleResidualActionBlock(config) for _ in range(self.depth)])
        self.velocity_head = TransitionAwarePhysicalVelocityHead(config)
        self.event_delta_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 3))
        self.motion_delta_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 1))
        self._zero_initialize_outputs()
        pairs = _parse_layer_pair_schedule(
            str(getattr(config, "action_flow_residual_layer_pair_schedule", "0:1,1:3,3:5,5:7")),
            decoder_depth=self.depth,
            num_layers=int(config.depth),
        )
        self.layer_pairs = tuple(pairs)
        if int(getattr(config, "action_flow_residual_stage_router", 1)):
            logits = torch.empty(self.depth, int(config.depth), dtype=torch.float32)
            for j, (a, b) in enumerate(pairs):
                center = 0.5 * float(a + b)
                for k in range(int(config.depth)):
                    logits[j, k] = -1.25 * abs(float(k) - center)
                logits[j, a] += 1.0
                logits[j, b] += 1.0
            self.stage_router_logits = nn.Parameter(logits)
        else:
            self.register_parameter("stage_router_logits", None)
        alpha = torch.full((int(config.action_horizon), 1), float(config.action_flow_residual_max_scale), dtype=torch.float32)
        first = max(int(config.first_execution_steps), 1)
        mid = max(int(config.mid_execution_steps), first)
        for i in range(int(config.action_horizon)):
            step = i + 1
            if step <= first:
                alpha[i, 0] = float(config.action_flow_residual_max_scale) * 0.25
            elif step <= mid:
                frac = float(step - first) / float(max(mid - first, 1))
                alpha[i, 0] = float(config.action_flow_residual_max_scale) * (0.25 + 0.75 * frac)
        self.register_buffer("residual_alpha", alpha[None], persistent=True)

    def _make_temporal_action_update_mask(self, config: V39PolicyConfig) -> Tensor:
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
        for seq in (self.event_delta_head, self.motion_delta_head):
            last = seq[-1]
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
        # Same ordered keys for every layer => same token length in normal V40.1.
        # If a future variant drops a key in one layer, the caller truncates router
        # mixing to the common token length.
        return torch.cat(parts, dim=1)

    @staticmethod
    def _truncate_all(memories: list[Tensor]) -> list[Tensor]:
        min_len = min(int(m.shape[1]) for m in memories)
        return [m[:, :min_len] for m in memories]

    def _router_memory(self, layer_memories: list[Tensor], stage_index: int) -> tuple[Tensor | None, Tensor, Tensor]:
        ref = layer_memories[0]
        z = torch.zeros((), device=ref.device, dtype=ref.dtype)
        if self.stage_router_logits is None or len(layer_memories) < 1:
            return None, z, z
        usable = self._truncate_all(layer_memories)
        stack = torch.stack(usable, dim=1)  # [B,L,N,H]
        logits = self.stage_router_logits[stage_index, : len(usable)].to(device=ref.device, dtype=torch.float32)
        weights = torch.softmax(logits, dim=0).to(device=ref.device, dtype=stack.dtype)
        mixed = torch.einsum("l,blnh->bnh", weights, stack)
        wf = weights.detach().float().clamp_min(1e-8)
        entropy = -(wf * wf.log()).sum()
        max_weight = wf.max()
        return mixed, entropy.to(device=ref.device, dtype=ref.dtype), max_weight.to(device=ref.device, dtype=ref.dtype)

    def _pair_delta(self, a: Tensor, b: Tensor) -> Tensor:
        n = min(int(a.shape[1]), int(b.shape[1]))
        return b[:, :n] - a[:, :n]

    def _build_stage_memory(
        self,
        *,
        stage_index: int,
        context_memory: Tensor,
        transition_memory: Tensor | None,
        visual_memory: Tensor | None,
        layer_memories: list[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor], dict[str, Tensor]]:
        cfg = self.config
        parts: list[Tensor] = []
        if stage_index == 0 and int(getattr(cfg, "action_flow_residual_context_memory", 1)):
            parts.append(context_memory)
        if stage_index == 1 and int(getattr(cfg, "action_flow_residual_visual_memory", 1)) and visual_memory is not None:
            parts.append(visual_memory)
        if stage_index >= 2 and int(getattr(cfg, "action_flow_residual_transition_memory", 1)) and transition_memory is not None:
            parts.append(transition_memory)
        router_entropy = torch.zeros((), device=context_memory.device, dtype=context_memory.dtype)
        router_max = torch.zeros((), device=context_memory.device, dtype=context_memory.dtype)
        if int(getattr(cfg, "action_flow_residual_layer_memory", 1)) and layer_memories:
            a, b = self.layer_pairs[stage_index]
            a = min(a, len(layer_memories) - 1)
            b = min(b, len(layer_memories) - 1)
            anchor = layer_memories[0]
            mem_a = layer_memories[a]
            mem_b = layer_memories[b]
            if int(getattr(cfg, "action_flow_residual_anchor_memory", 1)):
                parts.append(anchor)
            parts.extend([mem_a, mem_b, self._pair_delta(mem_a, mem_b)])
            mixed, router_entropy, router_max = self._router_memory(layer_memories, stage_index)
            if mixed is not None:
                parts.append(mixed)
        if not parts:
            parts.append(context_memory)
        return torch.cat(parts, dim=1), router_entropy, router_max

    def forward(
        self,
        *,
        noisy_physical: Tensor,
        time: Tensor,
        trajectory_pooled: Tensor,
        context_memory: Tensor,
        transition_memory: Tensor | None,
        visual_memory: Tensor | None,
        layer_contracts: list[dict[str, Tensor]],
    ) -> dict[str, Tensor]:
        cfg = self.config
        batch = int(noisy_physical.shape[0])
        dtype = noisy_physical.dtype
        device = noisy_physical.device
        context_memory = context_memory.to(device=device, dtype=dtype)
        transition_memory = None if transition_memory is None else transition_memory.to(device=device, dtype=dtype)
        visual_memory = None if visual_memory is None else visual_memory.to(device=device, dtype=dtype)
        detach_layers = bool(int(getattr(cfg, "action_flow_residual_layer_detach", 1)))
        layer_memories: list[Tensor] = []
        for entry in layer_contracts:
            memory = self._layer_entry_memory(entry, detach=detach_layers)
            if memory is not None:
                layer_memories.append(memory.to(device=device, dtype=dtype))
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
            )
            stage_memories.append(mem_j)
            entropies.append(ent_j)
            max_weights.append(max_j)
        global_summary = torch.stack([m.mean(dim=1) for m in stage_memories], dim=1).mean(dim=1)
        mem_summary = self.global_summary(global_summary)
        action_tokens = (
            self.horizon_query.to(device=device, dtype=dtype).expand(batch, -1, -1)
            + self.noisy_action_lift(noisy_physical)
            + self.trajectory_seed(trajectory_pooled)
            + mem_summary[:, None]
        )
        high_tokens = self.high_query.to(device=device, dtype=dtype).expand(batch, -1, -1) + mem_summary[:, None]
        event_tokens = self.event_query.to(device=device, dtype=dtype).expand(batch, -1, -1) + mem_summary[:, None]
        tokens = torch.cat([high_tokens, action_tokens, event_tokens], dim=1)
        high_slice = slice(0, self.high_slots)
        action_slice = slice(self.high_slots, self.high_slots + int(cfg.action_horizon))
        event_slice = slice(self.high_slots + int(cfg.action_horizon), self.high_slots + int(cfg.action_horizon) + int(cfg.event_tokens))
        time_base = self.time(time.to(dtype=dtype))
        for j, block in enumerate(self.blocks):
            stage_summary = self.stage_summary[j](stage_memories[j].mean(dim=1))
            tokens = block(tokens, stage_memories[j], time_base + stage_summary)
        high = tokens[:, high_slice]
        action = tokens[:, action_slice]
        event = tokens[:, event_slice]
        transition = high.mean(dim=1, keepdim=True).expand(-1, int(cfg.action_horizon), -1)
        transition = transition + event.mean(dim=1, keepdim=True).expand_as(transition)
        raw_residual = self.velocity_head(action, transition)
        alpha = self.residual_alpha.to(device=device, dtype=dtype)
        residual_velocity = raw_residual * alpha
        router_entropy = torch.stack(entropies).mean() if entropies else torch.zeros((), device=device, dtype=dtype)
        router_max = torch.stack(max_weights).mean() if max_weights else torch.zeros((), device=device, dtype=dtype)
        temporal_action_update_mean = (
            torch.stack([v.to(device=device, dtype=torch.float32) for v in action_update_means]).mean()
            if action_update_means else torch.ones((), device=device, dtype=torch.float32)
        )
        return {
            "residual_velocity": residual_velocity,
            "raw_residual_velocity": raw_residual,
            "residual_alpha": alpha,
            "event_delta_logits": self.event_delta_head(action),
            "motion_delta_logits": self.motion_delta_head(action).squeeze(-1),
            "action_tokens": action,
            "high_tokens": high,
            "event_tokens": event,
            "transition_latent": transition,
            "residual_norm": residual_velocity.detach().float().norm(dim=-1).mean(),
            "raw_residual_norm": raw_residual.detach().float().norm(dim=-1).mean(),
            "alpha_mean": alpha.detach().float().mean(),
            "stage_router_entropy": router_entropy.detach().float(),
            "stage_router_max": router_max.detach().float(),
        }


class HierarchicalLatentActionBlock(nn.Module):
    """One block of the V41 latent-main action decoder.

    This is not a side branch.  It is the only final-action path in
    ``final_action_decoder=latent_main_action``.  Each block updates the same
    high/action/event tokens by self-attention, then cross-attends a stage memory
    built from V40 layer memories and controlled transition latents, and finally
    applies stage-conditioned AdaLN/FFN modulation.
    """

    def __init__(self, config: V39PolicyConfig) -> None:
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

    def __init__(self, config: V39PolicyConfig) -> None:
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

    def _make_temporal_action_update_mask(self, config: V39PolicyConfig) -> Tensor:
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


class LatentCVAEActionBlock(nn.Module):
    """Small FiLM-conditioned token block for the V42 CVAE action head.

    This intentionally avoids the V41/V41.1 heavy memory cross-attention stack.
    All V40 latents are first fused into one condition vector; every decoder
    block receives that condition through AdaLN/FiLM, keeping the final action
    path compact and stable.
    """

    def __init__(self, config: V39PolicyConfig) -> None:
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


@dataclass
class PreparedEvidenceMemory:
    key_bias: Tensor
    ranges: dict[str, tuple[int, int]]
    block_kv: tuple[tuple[Tensor, Tensor], ...]
    batch_size: int


@dataclass(frozen=True)
class MMDiTConditionLayout:
    """Explicit condition-group slices in the condition-token coordinate."""

    noisy_start: int
    noisy_len: int
    rollout_start: int = 0
    rollout_len: int = 0
    low_start: int = 0
    low_len: int = 0
    stage_start: int = 0
    stage_len: int = 0


class EvidenceMemoryBank(nn.Module):
    """Owns typed evidence storage and static K/V preparation for workspace reads."""

    SOURCE_NAMES = (
        "layer",
        "scan",
        "lateral",
        "transition",
        "transition_delta",
        "transition_effect",
        "transition_timeline",
        "context",
        "visual",
        "trajectory",
        "rollout",
        "capsule",
        "progress",
        "routed_layer",
    )
    SOURCE_ROLES = {
        "layer": "layer",
        "routed_layer": "layer",
        "trajectory": "geom",
        "rollout": "geom",
        "transition": "transition",
        "transition_delta": "transition",
        "transition_effect": "transition",
        "transition_timeline": "transition",
        "progress": "state",
        "capsule": "state",
        "context": "state",
        "visual": "state",
        "scan": "global",
        "lateral": "global",
    }
    ROLE_NAMES = ("geom", "transition", "event", "state", "layer", "global")

    def __init__(self, config: V39PolicyConfig) -> None:
        super().__init__()
        h = int(config.hidden_size)
        self.hidden_size = h
        self.type_embed = nn.Parameter(torch.randn(1, len(self.SOURCE_NAMES), h) * 0.02)
        self.source_norm = nn.LayerNorm(h, elementwise_affine=False)

    def _source_role(self, name: str) -> str:
        return self.SOURCE_ROLES.get(name, "global")

    def prepare_sources(
        self,
        sources: dict[str, Tensor],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        allow_empty: bool,
    ) -> tuple[Tensor | None, Tensor, dict[str, tuple[int, int]]]:
        unknown_sources = set(sources).difference(self.SOURCE_NAMES)
        if unknown_sources:
            raise ValueError(f"unknown evidence workspace sources: {sorted(unknown_sources)}")
        parts: list[Tensor] = []
        ranges: dict[str, tuple[int, int]] = {}
        key_bias_parts: list[Tensor] = []
        offset = 0
        for index, name in enumerate(self.SOURCE_NAMES):
            value = sources.get(name)
            if value is None:
                continue
            if value.ndim != 3 or int(value.shape[-1]) != self.hidden_size:
                raise ValueError(f"workspace source {name!r} must be [B,N,H], got {tuple(value.shape)}")
            if int(value.shape[0]) != batch_size:
                raise ValueError(
                    f"workspace source {name!r} batch={int(value.shape[0])} "
                    f"does not match action batch={batch_size}"
                )
            value = value.to(device=device, dtype=dtype)
            if int(value.shape[1]) == 0:
                continue
            typed = self.source_norm(value) + self.type_embed[:, index:index + 1].to(device=device, dtype=dtype)
            parts.append(typed)
            count = int(typed.shape[1])
            ranges[name] = (offset, offset + count)
            key_bias_parts.append(torch.full((count,), -math.log(float(count)), device=device, dtype=torch.float32))
            offset += count
        if not parts:
            if allow_empty:
                return None, torch.zeros(0, device=device, dtype=torch.float32), ranges
            raise RuntimeError("evidence memory bank requires at least one semantic source")
        return torch.cat(parts, dim=1), torch.cat(key_bias_parts, dim=0), ranges

    def prepare_static_memory(
        self,
        sources: dict[str, Tensor],
        *,
        blocks: nn.ModuleList,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> PreparedEvidenceMemory:
        memory, key_bias, ranges = self.prepare_sources(
            sources,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            allow_empty=False,
        )
        assert memory is not None
        return PreparedEvidenceMemory(
            key_bias=key_bias,
            ranges=ranges,
            block_kv=tuple(block.project_memory(memory) for block in blocks),
            batch_size=batch_size,
        )

    def role_token_counts(self, ranges: dict[str, tuple[int, int]]) -> dict[str, int]:
        counts = {role: 0 for role in self.ROLE_NAMES}
        for name, (start, stop) in ranges.items():
            role = self._source_role(name)
            counts[role] = counts.get(role, 0) + max(int(stop) - int(start), 0)
        return counts

    def role_attention_metrics(
        self,
        weights: Tensor,
        ranges: dict[str, tuple[int, int]],
    ) -> dict[str, Tensor]:
        metrics: dict[str, Tensor] = {}
        device = weights.device
        dtype = weights.dtype
        for role in self.ROLE_NAMES:
            role_ranges = [
                (start, stop)
                for name, (start, stop) in ranges.items()
                if self._source_role(name) == role
            ]
            if role_ranges:
                parts = [weights[..., start:stop].sum(dim=-1).mean() for start, stop in role_ranges]
                metrics[f"workspace_role_{role}_attention"] = torch.stack(parts).sum()
                token_count = sum(max(int(stop) - int(start), 0) for start, stop in role_ranges)
            else:
                metrics[f"workspace_role_{role}_attention"] = torch.zeros((), device=device, dtype=dtype)
                token_count = 0
            metrics[f"workspace_role_{role}_token_count"] = torch.tensor(
                float(token_count),
                device=device,
                dtype=torch.float32,
            )
        return metrics

    def role_key_bias(
        self,
        role_logits: Tensor,
        ranges: dict[str, tuple[int, int]],
    ) -> Tensor:
        batch = int(role_logits.shape[0])
        total_tokens = max((stop for _, stop in ranges.values()), default=0)
        if total_tokens <= 0:
            return role_logits.new_zeros(batch, 0)
        role_to_index = {role: i for i, role in enumerate(self.ROLE_NAMES)}
        bias = role_logits.new_zeros(batch, total_tokens)
        for name, (start, stop) in ranges.items():
            role = self._source_role(name)
            index = role_to_index.get(role)
            if index is None:
                continue
            bias[:, start:stop] = role_logits[:, index:index + 1]
        return bias


class WorkspaceController(nn.Module):
    """Central capacity and role controller for workspace memory retrieval."""

    def __init__(self, config: V39PolicyConfig, role_names: tuple[str, ...]) -> None:
        super().__init__()
        h = int(config.hidden_size)
        self.role_names = tuple(role_names)
        self.state_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.action_state = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
        self.step_state = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
        self.workspace_mod = nn.Linear(h, 2 * h)
        self.query_mod = nn.Linear(h, 2 * h)
        self.role_head = nn.Linear(h, len(self.role_names))
        self.capacity_head = nn.Linear(h, 1)
        self.delay_head = nn.Linear(h, 1)
        self.temperature_head = nn.Linear(h, 1)
        for module in (self.workspace_mod, self.query_mod, self.role_head, self.capacity_head, self.delay_head, self.temperature_head):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

    @staticmethod
    def _bounded_modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        # Keep the controller a gentle manager rather than a second action head.
        return x * (1.0 + 0.10 * torch.tanh(scale)[:, None]) + 0.10 * torch.tanh(shift)[:, None]

    def forward(
        self,
        *,
        workspace: Tensor,
        action_query: Tensor,
        primary_cond: Tensor,
        step_context: Tensor,
        memory_bank: EvidenceMemoryBank,
        ranges: dict[str, tuple[int, int]],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        action_summary = action_query.mean(dim=1)
        step_state = self.step_state(step_context.to(device=primary_cond.device, dtype=primary_cond.dtype))
        value_state = self.state_norm(primary_cond + step_state)
        select_state = self.state_norm(primary_cond + step_state + self.action_state(action_summary))
        ws_shift, ws_scale = self.workspace_mod(value_state).chunk(2, dim=-1)
        q_shift, q_scale = self.query_mod(select_state).chunk(2, dim=-1)
        workspace = self._bounded_modulate(workspace, ws_shift, ws_scale)
        action_query = self._bounded_modulate(action_query, q_shift, q_scale)

        capacity_scale = 1.0 + 0.25 * torch.tanh(self.capacity_head(select_state)).squeeze(-1)
        delay_gate = torch.sigmoid(self.delay_head(select_state)).squeeze(-1)
        temperature = 0.5 + 1.5 * torch.sigmoid(self.temperature_head(select_state)).squeeze(-1)
        role_logits = self.role_head(select_state)
        gated_role_logits = role_logits * delay_gate[:, None] / temperature[:, None].clamp_min(1e-4)
        role_key_bias = memory_bank.role_key_bias(gated_role_logits, ranges)

        role_counts = memory_bank.role_token_counts(ranges)
        active_role_mask = torch.tensor(
            [role_counts.get(role, 0) > 0 for role in self.role_names],
            device=gated_role_logits.device,
            dtype=torch.bool,
        )
        masked_role_logits = gated_role_logits.float().masked_fill(~active_role_mask[None], -1e4)
        role_probs = torch.softmax(masked_role_logits, dim=-1)
        role_entropy = -(role_probs.clamp_min(1e-8) * role_probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
        metrics: dict[str, Tensor] = {
            "workspace_controller_capacity": capacity_scale.detach().float().mean(),
            "workspace_controller_delay": delay_gate.detach().float().mean(),
            "workspace_controller_temperature": temperature.detach().float().mean(),
            "workspace_controller_role_entropy": role_entropy.detach().float(),
            "workspace_controller_role_max": role_probs.detach().float().max(dim=-1).values.mean(),
            "workspace_controller_query_delta_norm": (0.10 * torch.tanh(q_shift)).detach().float().norm(dim=-1).mean(),
            "workspace_controller_workspace_delta_norm": (0.10 * torch.tanh(ws_shift)).detach().float().norm(dim=-1).mean(),
        }
        for index, role in enumerate(self.role_names):
            metrics[f"workspace_controller_role_{role}_prob"] = role_probs[:, index].detach().float().mean()
            metrics[f"workspace_controller_role_{role}_logit"] = gated_role_logits[:, index].detach().float().mean()
        return workspace, action_query, role_key_bias, capacity_scale, metrics


class SemanticEvidenceWorkspaceBlock(nn.Module):
    """AdaLN-conditioned workspace block with one evidence write path."""

    def __init__(self, config: V39PolicyConfig) -> None:
        super().__init__()
        h = int(config.hidden_size)
        heads = int(config.num_heads)
        if h % heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads for evidence workspace")
        self.config = config
        self.heads = heads
        self.head_dim = h // heads
        self.self_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(h, heads, batch_first=True, dropout=float(config.dropout))
        self.cross_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.memory_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.cross_q = nn.Linear(h, h, bias=False)
        self.cross_k = nn.Linear(h, h, bias=False)
        self.cross_v = nn.Linear(h, h, bias=False)
        self.cross_out = nn.Linear(h, h)
        self.ffn_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.ffn = BiasFreeFFN(h, float(getattr(config, "latent_cvae_ffn_expansion", 2.0)))
        self.mod = nn.Linear(h, 9 * h)
        self.drop = nn.Dropout(float(config.dropout))
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)
        # Evidence is visible from the first update, while z/time learns how to
        # specialize the read through AdaLN without an initially dead path.
        nn.init.constant_(self.mod.bias[5 * h: 6 * h], math.atanh(0.10))

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

    def project_memory(self, memory: Tensor) -> tuple[Tensor, Tensor]:
        memory_value = self.memory_norm(memory)
        return (
            self._split_heads(self.cross_k(memory_value)),
            self._split_heads(self.cross_v(memory_value)),
        )

    def forward(
        self,
        workspace: Tensor,
        primary_cond: Tensor,
        *,
        memory_k: Tensor,
        memory_v: Tensor,
        key_bias: Tensor,
        query_context: Tensor,
        read_scale: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        (
            self_s, self_c, self_g,
            cross_s, cross_c, cross_g,
            ffn_s, ffn_c, ffn_g,
        ) = self.mod(primary_cond).chunk(9, dim=-1)
        causal_mask = None
        if int(getattr(self.config, "latent_cvae_causal_attention", 1)):
            n = int(workspace.shape[1])
            causal_mask = torch.triu(torch.ones(n, n, device=workspace.device, dtype=torch.bool), diagonal=1)
        self_value = self._modulate(self.self_norm(workspace), self_s, self_c)
        self_update, _ = self.self_attn(
            self_value,
            self_value,
            self_value,
            attn_mask=causal_mask,
            need_weights=False,
        )
        workspace = workspace + torch.tanh(self_g)[:, None] * self.drop(self_update)

        # Current action state selects evidence but never enters the workspace
        # value/residual stream. This preserves state-dependent retrieval without
        # creating an action -> workspace -> action echo shortcut.
        cross_query = self.cross_norm(workspace + query_context)
        q = self._split_heads(self.cross_q(self._modulate(cross_query, cross_s, cross_c)))
        scores = torch.matmul(q.float(), memory_k.float().transpose(-2, -1)) * (float(self.head_dim) ** -0.5)
        key_bias = key_bias.to(device=scores.device, dtype=scores.dtype)
        if key_bias.ndim == 1:
            scores = scores + key_bias[None, None, None]
        elif key_bias.ndim == 2:
            scores = scores + key_bias[:, None, None]
        else:
            raise ValueError(f"key_bias must be [M] or [B,M], got {tuple(key_bias.shape)}")
        weights = torch.softmax(scores, dim=-1).to(dtype=q.dtype)
        cross_update = torch.matmul(weights, memory_v)
        cross_update = self.cross_out(self._merge_heads(cross_update))
        cross_gain = torch.tanh(cross_g)[:, None]
        if read_scale is not None:
            cross_gain = cross_gain * read_scale.to(device=workspace.device, dtype=workspace.dtype)[:, None, None]
        workspace = workspace + cross_gain * self.drop(cross_update)

        ffn_update = self.ffn(self._modulate(self.ffn_norm(workspace), ffn_s, ffn_c))
        workspace = workspace + torch.tanh(ffn_g)[:, None] * self.drop(ffn_update)
        return workspace, weights


class SemanticEvidenceWorkspace(nn.Module):
    """Fuse typed semantic sources into a configurable horizon token field."""

    SOURCE_NAMES = EvidenceMemoryBank.SOURCE_NAMES

    def __init__(self, config: V39PolicyConfig) -> None:
        super().__init__()
        self.config = config
        h = int(config.hidden_size)
        self.hidden_size = h
        self.token_count = int(getattr(config, "latent_cvae_horizon_tokens", config.action_horizon))
        self.memory_bank = EvidenceMemoryBank(config)
        self.query = nn.Parameter(torch.randn(1, self.token_count, h) * 0.02)
        self.action_query_proj = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
        self.step_query_proj = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
        self.global_state_proj = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
        global_state = self.global_state_proj[-1]
        if isinstance(global_state, nn.Linear):
            # Time-state is a residual retrieval bias.  Start from the previous
            # workspace behavior and let training open this path deliberately;
            # random z/time injection can dominate the first batches.
            nn.init.zeros_(global_state.weight)
            nn.init.zeros_(global_state.bias)
        self.controller = (
            WorkspaceController(config, EvidenceMemoryBank.ROLE_NAMES)
            if int(getattr(config, "latent_cvae_workspace_controller", 0))
            else None
        )
        self.blocks = nn.ModuleList([SemanticEvidenceWorkspaceBlock(config) for _ in range(2)])
        self.final_norm = nn.LayerNorm(h, elementwise_affine=False)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        # V74A moved the source type embedding into EvidenceMemoryBank.  Keep
        # old v73 checkpoints loadable under both stage1 non-strict and resume
        # strict paths without exposing duplicate parameters.
        old_type_key = prefix + "type_embed"
        new_type_key = prefix + "memory_bank.type_embed"
        if old_type_key in state_dict and new_type_key not in state_dict:
            state_dict[new_type_key] = state_dict.pop(old_type_key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _resize_action(self, action: Tensor) -> Tensor:
        if int(action.shape[1]) == self.token_count:
            return action
        if self.token_count == 1:
            return action.mean(dim=1, keepdim=True)
        if int(action.shape[1]) == 1:
            return action.expand(-1, self.token_count, -1)
        return F.interpolate(
            action.transpose(1, 2).float(),
            size=self.token_count,
            mode="linear",
            align_corners=True,
        ).transpose(1, 2).to(dtype=action.dtype)

    def _slot_aware_global_state(self, global_state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Keep z/time workspace state global in meaning but slot-aware in form."""
        slot_state = global_state[:, None]
        zero = torch.zeros((), device=global_state.device, dtype=torch.float32)
        if (
            not int(getattr(self.config, "latent_cvae_workspace_slot_time_state", 1))
            or float(getattr(self.config, "latent_cvae_workspace_slot_time_scale", 0.10)) <= 0.0
            or self.token_count <= 1
        ):
            return slot_state, zero, zero
        scale = float(getattr(self.config, "latent_cvae_workspace_slot_time_scale", 0.10))
        # Use the learned workspace slot identity as a bounded selector.  This
        # preserves the global z/time signal while avoiding an identical vector
        # being added to every retrieval slot.
        slot_identity = F.layer_norm(
            self.query.to(device=global_state.device, dtype=global_state.dtype),
            (self.hidden_size,),
        )
        slot_delta = scale * global_state[:, None] * torch.tanh(slot_identity)
        slot_state = slot_state + slot_delta
        slot_delta_norm = slot_delta.detach().float().norm(dim=-1).mean()
        slot_diversity = (slot_state - slot_state.mean(dim=1, keepdim=True)).detach().float().norm(dim=-1).mean()
        return slot_state, slot_delta_norm, slot_diversity

    def _prepare_sources(
        self,
        sources: dict[str, Tensor],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        allow_empty: bool,
    ) -> tuple[Tensor | None, Tensor, dict[str, tuple[int, int]]]:
        return self.memory_bank.prepare_sources(
            sources,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            allow_empty=allow_empty,
        )

    def prepare_static_memory(
        self,
        sources: dict[str, Tensor],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> PreparedEvidenceMemory:
        return self.memory_bank.prepare_static_memory(
            sources,
            blocks=self.blocks,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        sources: dict[str, Tensor],
        *,
        action: Tensor,
        primary_cond: Tensor,
        step_context: Tensor,
        static_memory: PreparedEvidenceMemory | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        device = action.device
        dtype = action.dtype
        batch_size = int(action.shape[0])
        if static_memory is not None and static_memory.batch_size != batch_size:
            raise ValueError(
                f"cached workspace batch={static_memory.batch_size} does not match action batch={batch_size}"
            )
        dynamic_memory, dynamic_bias, dynamic_ranges = self._prepare_sources(
            sources,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            allow_empty=static_memory is not None,
        )
        if static_memory is None:
            assert dynamic_memory is not None
            ranges = dynamic_ranges
            key_bias = dynamic_bias
            static_token_count = 0
        else:
            overlap = set(static_memory.ranges).intersection(dynamic_ranges)
            if overlap:
                raise ValueError(f"workspace sources appear in both static and dynamic memory: {sorted(overlap)}")
            static_token_count = int(static_memory.key_bias.numel())
            ranges = dict(static_memory.ranges)
            ranges.update({name: (start + static_token_count, stop + static_token_count) for name, (start, stop) in dynamic_ranges.items()})
            key_bias = torch.cat([static_memory.key_bias.to(device=device), dynamic_bias], dim=0)
        action_query = self.action_query_proj(self._resize_action(action))
        step_query = self.step_query_proj(step_context.to(device=device, dtype=dtype))[:, None]
        if int(getattr(self.config, "latent_cvae_workspace_time_state", 0)):
            global_state = self.global_state_proj(primary_cond.to(device=device, dtype=dtype))
        else:
            global_state = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        global_slot_state, global_slot_delta_norm, global_slot_diversity = self._slot_aware_global_state(global_state)
        workspace = self.query.to(device=device, dtype=dtype).expand(int(action.shape[0]), -1, -1)
        workspace = workspace + step_query + global_slot_state
        workspace_seed = workspace
        read_scale: Tensor | None = None
        controller_metrics: dict[str, Tensor] = {}
        if self.controller is not None:
            workspace, action_query, role_bias, read_scale, controller_metrics = self.controller(
                workspace=workspace,
                action_query=action_query,
                primary_cond=primary_cond,
                step_context=step_context,
                memory_bank=self.memory_bank,
                ranges=ranges,
            )
            key_bias = key_bias.to(device=device) + role_bias.to(device=device, dtype=key_bias.dtype)
        weight_rows: list[Tensor] = []
        for block_index, block in enumerate(self.blocks):
            if dynamic_memory is None:
                dynamic_k = dynamic_v = None
            else:
                dynamic_k, dynamic_v = block.project_memory(dynamic_memory)
            if static_memory is None:
                assert dynamic_k is not None and dynamic_v is not None
                memory_k, memory_v = dynamic_k, dynamic_v
            else:
                static_k, static_v = static_memory.block_kv[block_index]
                memory_k = static_k if dynamic_k is None else torch.cat([static_k, dynamic_k], dim=2)
                memory_v = static_v if dynamic_v is None else torch.cat([static_v, dynamic_v], dim=2)
            workspace, weights = block(
                workspace,
                primary_cond,
                memory_k=memory_k,
                memory_v=memory_v,
                key_bias=key_bias,
                query_context=action_query,
                read_scale=read_scale,
            )
            weight_rows.append(weights.detach().float())
        workspace_pre_norm = workspace
        workspace = self.final_norm(workspace)
        weights = torch.stack(weight_rows).mean(dim=0)
        metrics: dict[str, Tensor] = {
            "workspace_token_count": torch.tensor(float(workspace.shape[1]), device=device, dtype=torch.float32),
            "workspace_token_norm": workspace.detach().float().norm(dim=-1).mean(),
            "workspace_update_norm": (workspace_seed.detach() - workspace_pre_norm.detach()).float().norm(dim=-1).mean(),
            "workspace_global_state_norm": global_state.detach().float().norm(dim=-1).mean(),
            "workspace_global_slot_delta_norm": global_slot_delta_norm,
            "workspace_global_slot_diversity": global_slot_diversity,
            "workspace_source_count": torch.tensor(float(len(ranges)), device=device, dtype=torch.float32),
            "workspace_cached_token_fraction": torch.tensor(
                float(static_token_count) / float(max(int(key_bias.numel()), 1)),
                device=device,
                dtype=torch.float32,
            ),
            "workspace_attention_entropy": -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=-1).mean(),
            "workspace_attention_max": weights.max(dim=-1).values.mean(),
        }
        metrics.update(controller_metrics)
        group_weights = torch.stack([
            weights[..., start:stop].sum(dim=-1)
            for start, stop in ranges.values()
        ], dim=-1)
        metrics["workspace_group_attention_entropy"] = -(
            group_weights.clamp_min(1e-8) * group_weights.clamp_min(1e-8).log()
        ).sum(dim=-1).mean()
        metrics["workspace_group_effective_sources"] = torch.exp(metrics["workspace_group_attention_entropy"])
        metrics["workspace_attention_mass_error"] = (group_weights.sum(dim=-1) - 1.0).abs().mean()
        for name, (start, stop) in ranges.items():
            metrics[f"workspace_{name}_attention"] = weights[..., start:stop].sum(dim=-1).mean()
        metrics.update(self.memory_bank.role_attention_metrics(weights, ranges))
        transition_mass = [
            metrics[key]
            for key in (
                "workspace_transition_attention",
                "workspace_transition_delta_attention",
                "workspace_transition_effect_attention",
                "workspace_transition_timeline_attention",
            )
            if key in metrics
        ]
        if transition_mass:
            metrics["workspace_transition_total_attention"] = torch.stack(transition_mass).sum()
        return workspace, metrics


class HierarchicalWorkspaceManager(nn.Module):
    """Condition/stage-driven retrieval manager with no action input.

    Stage content is allowed to shape selector state, role logits, promotion,
    and stage output strength. The low-output scale is intentionally computed
    from condition+step only, so stage cannot modulate the low value stream.
    """

    def __init__(self, config: V39PolicyConfig, role_names: tuple[str, ...]) -> None:
        super().__init__()
        h = int(config.hidden_size)
        self.hidden_size = h
        self.role_names = tuple(role_names)
        self.base_fusion = nn.Sequential(
            nn.LayerNorm(2 * h),
            nn.Linear(2 * h, h),
            nn.SiLU(),
            nn.Linear(h, h),
        )
        self.stage_role_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.stage_content_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.stage_query = nn.Linear(h, h, bias=False)
        self.stage_role_key = nn.Linear(h, h, bias=False)
        self.stage_content_key = nn.Linear(h, h, bias=False)
        self.stage_content_value = nn.Linear(h, h, bias=False)
        self.stage_summary_out = nn.Linear(h, h)
        self.select_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.query_shift = nn.Linear(h, h)
        self.role_head = nn.Linear(h, len(self.role_names))
        self.promote_head = nn.Linear(h, 1)
        self.low_output_head = nn.Linear(h, 1)
        self.stage_output_head = nn.Linear(h, 1)
        for module in (
            self.query_shift,
            self.role_head,
            self.promote_head,
            self.low_output_head,
            self.stage_output_head,
        ):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)
        # Begin as a conservative promotion controller. This gate remains
        # effective because it is applied after normalization in the stage cell.
        nn.init.constant_(self.promote_head.bias, math.log(0.10 / 0.90))
        # Stage is a new condition group on top of a pretrained MMDiT. Let it
        # enter with 0.1 prior strength and earn more attention during training.
        stage_fraction = 0.10 / 1.50
        nn.init.constant_(self.stage_output_head.bias, math.log(stage_fraction / (1.0 - stage_fraction)))

    def forward(
        self,
        *,
        primary_cond: Tensor,
        step_embedding: Tensor,
        stage_role: Tensor,
        stage_content: Tensor,
        memory_bank: EvidenceMemoryBank,
        ranges: dict[str, tuple[int, int]],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        base_state = self.base_fusion(torch.cat([primary_cond, step_embedding], dim=-1))
        q = self.stage_query(base_state)[:, None]
        normalized_role = self.stage_role_norm(stage_role)
        normalized_content = self.stage_content_norm(stage_content)
        k = self.stage_role_key(normalized_role) + self.stage_content_key(normalized_content)
        v = self.stage_content_value(normalized_content)
        logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * (float(self.hidden_size) ** -0.5)
        stage_weights = torch.softmax(logits, dim=-1).to(dtype=stage_content.dtype)
        stage_summary = self.stage_summary_out(torch.matmul(stage_weights, v).squeeze(1))
        select_state = self.select_norm(base_state + stage_summary)

        query_shift = self.query_shift(select_state)
        role_logits = self.role_head(select_state)
        role_key_bias = memory_bank.role_key_bias(role_logits, ranges)
        promote_gate = torch.sigmoid(self.promote_head(select_state)).squeeze(-1)
        # This head never sees stage_summary/select_state. That is the explicit
        # stage -> low-value firewall; stage affects only query/role selection.
        low_output_strength = torch.exp(0.5 * torch.tanh(self.low_output_head(base_state))).squeeze(-1)
        stage_output_strength = 1.5 * torch.sigmoid(self.stage_output_head(select_state)).squeeze(-1)

        role_counts = memory_bank.role_token_counts(ranges)
        active_mask = torch.tensor(
            [role_counts.get(role, 0) > 0 for role in self.role_names],
            device=role_logits.device,
            dtype=torch.bool,
        )
        masked_role_logits = role_logits.float().masked_fill(~active_mask[None], -1e4)
        role_probs = torch.softmax(masked_role_logits, dim=-1)
        role_entropy = -(role_probs.clamp_min(1e-8) * role_probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
        stage_prob = stage_weights.detach().float().clamp_min(1e-8)
        metrics: dict[str, Tensor] = {
            "hierarchical_manager_stage_attention_entropy": -(stage_prob * stage_prob.log()).sum(dim=-1).mean(),
            "hierarchical_manager_stage_attention_max": stage_prob.max(dim=-1).values.mean(),
            "hierarchical_manager_role_entropy": role_entropy.detach(),
            "hierarchical_manager_role_max": role_probs.detach().max(dim=-1).values.mean(),
            "hierarchical_manager_query_shift_norm": query_shift.detach().float().norm(dim=-1).mean(),
            "hierarchical_manager_promote_gate": promote_gate.detach().float().mean(),
            "hierarchical_manager_low_output_strength": low_output_strength.detach().float().mean(),
            "hierarchical_manager_stage_output_strength": stage_output_strength.detach().float().mean(),
        }
        for index, role in enumerate(self.role_names):
            metrics[f"hierarchical_manager_role_{role}_prob"] = role_probs[:, index].detach().float().mean()
        return (
            query_shift,
            role_key_bias,
            promote_gate,
            low_output_strength,
            stage_output_strength,
            metrics,
        )


class HierarchicalEvidenceWorkspace(nn.Module):
    """Temporary low reads plus persistent, role-separated stage memory."""

    def __init__(self, config: V39PolicyConfig) -> None:
        super().__init__()
        self.config = config
        h = int(config.hidden_size)
        heads = int(config.num_heads)
        if h % heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads for hierarchical workspace")
        self.hidden_size = h
        self.heads = heads
        self.head_dim = h // heads
        self.low_count = int(getattr(config, "latent_cvae_horizon_tokens", config.action_horizon))
        self.stage_count = int(getattr(config, "latent_cvae_stage_slots", 6))
        self.refine_steps = max(int(getattr(config, "adaptive_cvae_refine_steps", 1)), 1)
        self.memory_bank = EvidenceMemoryBank(config)

        # Value and selector identities are separate parameters. Per-sample
        # stage state is never added to low_value_seed.
        self.low_value_seed = nn.Parameter(torch.randn(1, self.low_count, h) * 0.02)
        self.low_selector_seed = nn.Parameter(torch.randn(1, self.low_count, h) * 0.02)
        self.step_embedding = nn.Parameter(torch.randn(1, self.refine_steps, h) * 0.02)
        self.condition_query = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
        self.manager = HierarchicalWorkspaceManager(config, EvidenceMemoryBank.ROLE_NAMES)

        self.low_stage_query = nn.Linear(h, h, bias=False)
        self.stage_role_selector_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.stage_content_selector_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.low_stage_role_key = nn.Linear(h, h, bias=False)
        self.low_stage_content_key = nn.Linear(h, h, bias=False)
        self.low_stage_role_value = nn.Linear(h, h, bias=False)
        self.low_stage_content_value = nn.Linear(h, h, bias=False)
        self.low_stage_out = nn.Linear(h, h)
        self.low_blocks = nn.ModuleList([SemanticEvidenceWorkspaceBlock(config) for _ in range(2)])
        self.low_final_norm = nn.LayerNorm(h, elementwise_affine=False)

        # Role is a persistent learned identity. Content is the only recurrent
        # state and is initialized without adding the role tensor. One shared
        # seed avoids both zero-LayerNorm gain and a second hidden slot-role.
        self.stage_role = nn.Parameter(torch.randn(1, self.stage_count, h) * 0.02)
        self.stage_content_seed = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        self.stage_init = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
        nn.init.zeros_(self.stage_init[-1].weight)
        nn.init.zeros_(self.stage_init[-1].bias)
        self.stage_role_query = nn.Linear(h, h, bias=False)
        self.stage_content_query = nn.Linear(h, h, bias=False)
        self.stage_condition_query = nn.Linear(h, h, bias=False)
        self.stage_low_key = nn.Linear(h, h, bias=False)
        self.stage_low_value = nn.Linear(h, h, bias=False)
        self.stage_promote_out = nn.Linear(h, h)
        self.stage_input_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.stage_gru = nn.GRUCell(h, h)
        with torch.no_grad():
            self.stage_gru.bias_ih.zero_()
            self.stage_gru.bias_hh.zero_()
            # PyTorch GRU gate order is reset, update(retain), candidate.
            self.stage_gru.bias_ih[h:2 * h].fill_(0.5)
            self.stage_gru.bias_hh[h:2 * h].fill_(0.5)
        promote_init = min(max(float(getattr(config, "latent_cvae_stage_promote_scale_init", 0.05)), 1e-4), 1.0 - 1e-4)
        self.stage_promote_scale_logit = nn.Parameter(torch.tensor(math.log(promote_init / (1.0 - promote_init))))
        self.stage_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.stage_content_out = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
        self.stage_role_out = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
        nn.init.eye_(self.stage_content_out[-1].weight)
        nn.init.zeros_(self.stage_content_out[-1].bias)
        nn.init.eye_(self.stage_role_out[-1].weight)
        nn.init.zeros_(self.stage_role_out[-1].bias)

    def _split_heads(self, x: Tensor) -> Tensor:
        b, n, h = x.shape
        return x.reshape(b, n, self.heads, h // self.heads).transpose(1, 2)

    @staticmethod
    def _merge_heads(x: Tensor) -> Tensor:
        b, heads, n, d = x.shape
        return x.transpose(1, 2).reshape(b, n, heads * d)

    def _attention(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        qh = self._split_heads(q)
        kh = self._split_heads(k)
        vh = self._split_heads(v)
        logits = torch.matmul(qh.float(), kh.float().transpose(-2, -1)) * (float(self.head_dim) ** -0.5)
        weights = torch.softmax(logits, dim=-1).to(dtype=q.dtype)
        return self._merge_heads(torch.matmul(weights, vh)), weights

    def _step_state(self, step_index: int, *, batch: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        index = min(max(int(step_index), 0), self.refine_steps - 1)
        return self.step_embedding[:, index].to(device=device, dtype=dtype).expand(batch, -1)

    def prepare_evidence(
        self,
        sources: dict[str, Tensor],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> PreparedEvidenceMemory:
        return self.memory_bank.prepare_static_memory(
            sources,
            blocks=self.low_blocks,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

    def init_stage(self, primary_cond: Tensor) -> Tensor:
        batch = int(primary_cond.shape[0])
        content = self.stage_content_seed.to(device=primary_cond.device, dtype=primary_cond.dtype).expand(
            batch, self.stage_count, -1
        )
        content = content + self.stage_init(primary_cond)[:, None]
        return self.stage_norm(content)

    def _low_selector_context(
        self,
        *,
        primary_cond: Tensor,
        step_state: Tensor,
        manager_shift: Tensor,
        stage_role: Tensor,
        stage_content: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch = int(primary_cond.shape[0])
        selector_seed = self.low_selector_seed.to(device=primary_cond.device, dtype=primary_cond.dtype).expand(batch, -1, -1)
        selector_seed = selector_seed + self.condition_query(primary_cond)[:, None] + step_state[:, None] + manager_shift[:, None]
        q = self.low_stage_query(selector_seed)
        normalized_role = self.stage_role_selector_norm(stage_role)
        normalized_content = self.stage_content_selector_norm(stage_content)
        k = self.low_stage_role_key(normalized_role) + self.low_stage_content_key(normalized_content)
        role_v = self.low_stage_role_value(normalized_role)
        content_v = self.low_stage_content_value(normalized_content)
        role_context, weights = self._attention(q, k, role_v)
        content_context = self._merge_heads(torch.matmul(weights, self._split_heads(content_v)))
        stage_selector = self.low_stage_out(role_context + content_context)
        # stage_selector is returned only as query_context to the evidence
        # cross-attention. It is never added to low_value_seed or low residuals.
        return selector_seed + stage_selector, weights, role_context, content_context

    def _stage_retain_gate(self, x: Tensor, hidden: Tensor) -> Tensor:
        gi = F.linear(x, self.stage_gru.weight_ih, self.stage_gru.bias_ih)
        gh = F.linear(hidden, self.stage_gru.weight_hh, self.stage_gru.bias_hh)
        _, input_update, _ = gi.chunk(3, dim=-1)
        _, hidden_update, _ = gh.chunk(3, dim=-1)
        return torch.sigmoid(input_update + hidden_update)

    def _promote_stage(
        self,
        *,
        low_tokens: Tensor,
        stage_role: Tensor,
        stage_content: Tensor,
        primary_cond: Tensor,
        step_state: Tensor,
        promote_gate: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        normalized_role = self.stage_role_selector_norm(stage_role)
        normalized_content = self.stage_content_selector_norm(stage_content)
        q = (
            self.stage_role_query(normalized_role)
            + self.stage_content_query(normalized_content)
            + self.stage_condition_query(primary_cond + step_state)[:, None]
        )
        k = self.stage_low_key(low_tokens)
        v = self.stage_low_value(low_tokens)
        promoted, weights = self._attention(q, k, v)
        promoted = self.stage_promote_out(promoted)
        gated_promoted = promoted * promote_gate[:, None, None].to(dtype=promoted.dtype)
        # Normalize first. Reversing this order would cancel the scalar gate
        # and silently turn promotion strength into a fake control surface.
        gru_input = self.stage_input_norm(promoted) * promote_gate[:, None, None].to(dtype=promoted.dtype)
        flat_input = gru_input.reshape(-1, self.hidden_size)
        flat_hidden = stage_content.reshape(-1, self.hidden_size)
        retain = self._stage_retain_gate(flat_input, flat_hidden)
        recurrent = self.stage_gru(flat_input, flat_hidden).reshape_as(stage_content)
        promote_scale = torch.sigmoid(self.stage_promote_scale_logit).to(device=stage_content.device, dtype=stage_content.dtype)
        next_content = self.stage_norm(recurrent + promote_scale * gated_promoted)
        return next_content, weights, promoted, retain, promote_scale

    @staticmethod
    def _slot_diversity(x: Tensor) -> Tensor:
        return (x - x.mean(dim=1, keepdim=True)).detach().float().norm(dim=-1).mean()

    def step(
        self,
        *,
        prepared_evidence: PreparedEvidenceMemory,
        stage_content: Tensor,
        primary_cond: Tensor,
        step_index: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        batch = int(primary_cond.shape[0])
        device = primary_cond.device
        dtype = primary_cond.dtype
        if prepared_evidence.batch_size != batch:
            raise ValueError(
                f"cached hierarchical workspace batch={prepared_evidence.batch_size} does not match decode batch={batch}"
            )
        if tuple(stage_content.shape) != (batch, self.stage_count, self.hidden_size):
            raise ValueError(
                f"stage content must be {(batch, self.stage_count, self.hidden_size)}, got {tuple(stage_content.shape)}"
            )
        stage_role = self.stage_role.to(device=device, dtype=dtype).expand(batch, -1, -1)
        step_state = self._step_state(step_index, batch=batch, device=device, dtype=dtype)
        (
            manager_shift,
            role_bias,
            promote_gate,
            low_output_strength,
            stage_output_strength,
            manager_metrics,
        ) = self.manager(
            primary_cond=primary_cond,
            step_embedding=step_state,
            stage_role=stage_role,
            stage_content=stage_content,
            memory_bank=self.memory_bank,
            ranges=prepared_evidence.ranges,
        )
        query_context, selector_weights, selector_role, selector_content = self._low_selector_context(
            primary_cond=primary_cond,
            step_state=step_state,
            manager_shift=manager_shift,
            stage_role=stage_role,
            stage_content=stage_content,
        )

        # The low value stream starts from a stage-independent seed. All stage
        # influence is confined to query_context and role_bias above.
        low = self.low_value_seed.to(device=device, dtype=dtype).expand(batch, -1, -1)
        low_seed = low
        key_bias = prepared_evidence.key_bias.to(device=device) + role_bias.to(device=device, dtype=torch.float32)
        evidence_weight_rows: list[Tensor] = []
        for block_index, block in enumerate(self.low_blocks):
            memory_k, memory_v = prepared_evidence.block_kv[block_index]
            low, weights = block(
                low,
                primary_cond,
                memory_k=memory_k,
                memory_v=memory_v,
                key_bias=key_bias,
                query_context=query_context,
                read_scale=None,
            )
            evidence_weight_rows.append(weights.detach().float())
        low_pre_norm = low
        low = self.low_final_norm(low)
        low_for_action = low

        next_stage_content, promote_weights, promoted, retain, promote_scale = self._promote_stage(
            low_tokens=low,
            stage_role=stage_role,
            stage_content=stage_content,
            primary_cond=primary_cond,
            step_state=step_state,
            promote_gate=promote_gate,
        )
        role_component = self.stage_role_out(stage_role)
        content_component = self.stage_content_out(next_stage_content)
        stage_for_action = role_component + content_component
        low_logit_bias = low_output_strength.clamp_min(1e-4).log()
        stage_logit_bias = stage_output_strength.clamp_min(1e-4).log()

        weights = torch.stack(evidence_weight_rows).mean(dim=0)
        group_weights = torch.stack([
            weights[..., start:stop].sum(dim=-1)
            for start, stop in prepared_evidence.ranges.values()
        ], dim=-1)
        selector_prob = selector_weights.detach().float().clamp_min(1e-8)
        promote_prob = promote_weights.detach().float().clamp_min(1e-8)
        zero = torch.zeros((), device=device, dtype=torch.float32)
        role_norm = role_component.detach().float().norm(dim=-1).mean()
        content_norm = content_component.detach().float().norm(dim=-1).mean()
        metrics: dict[str, Tensor] = {
            "workspace_token_count": torch.tensor(float(self.low_count), device=device, dtype=torch.float32),
            "workspace_token_norm": low_for_action.detach().float().norm(dim=-1).mean(),
            "workspace_update_norm": (low_pre_norm.detach() - low_seed.detach()).float().norm(dim=-1).mean(),
            "workspace_global_state_norm": zero,
            "workspace_global_slot_delta_norm": zero,
            "workspace_global_slot_diversity": zero,
            "workspace_source_count": torch.tensor(float(len(prepared_evidence.ranges)), device=device, dtype=torch.float32),
            "workspace_cached_token_fraction": torch.ones((), device=device, dtype=torch.float32),
            "workspace_noisy_query_scale": zero,
            "workspace_progress_query_norm": zero,
            "workspace_attention_entropy": -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=-1).mean(),
            "workspace_attention_max": weights.max(dim=-1).values.mean(),
            "workspace_group_attention_entropy": -(
                group_weights.clamp_min(1e-8) * group_weights.clamp_min(1e-8).log()
            ).sum(dim=-1).mean(),
            "workspace_attention_mass_error": (group_weights.sum(dim=-1) - 1.0).abs().mean(),
            "hierarchical_low_token_count": torch.tensor(float(self.low_count), device=device, dtype=torch.float32),
            "hierarchical_low_token_norm": low_for_action.detach().float().norm(dim=-1).mean(),
            "hierarchical_low_selector_stage_entropy": -(selector_prob * selector_prob.log()).sum(dim=-1).mean(),
            "hierarchical_low_selector_stage_max": selector_prob.max(dim=-1).values.mean(),
            "hierarchical_low_selector_stage_effective_slots": torch.exp(
                -(selector_prob * selector_prob.log()).sum(dim=-1).mean()
            ),
            "hierarchical_low_selector_role_norm": selector_role.detach().float().norm(dim=-1).mean(),
            "hierarchical_low_selector_content_norm": selector_content.detach().float().norm(dim=-1).mean(),
            "hierarchical_stage_token_count": torch.tensor(float(self.stage_count), device=device, dtype=torch.float32),
            "hierarchical_stage_role_norm": stage_role.detach().float().norm(dim=-1).mean(),
            "hierarchical_stage_role_diversity": self._slot_diversity(stage_role),
            "hierarchical_stage_content_norm": next_stage_content.detach().float().norm(dim=-1).mean(),
            "hierarchical_stage_content_diversity": self._slot_diversity(next_stage_content),
            "hierarchical_stage_role_content_cosine": F.cosine_similarity(
                stage_role.detach().float(), next_stage_content.detach().float(), dim=-1
            ).mean(),
            "hierarchical_stage_role_output_norm": role_norm,
            "hierarchical_stage_content_output_norm": content_norm,
            "hierarchical_stage_role_output_fraction": role_norm / (role_norm + content_norm).clamp_min(1e-8),
            "hierarchical_stage_update_norm": (
                next_stage_content.detach().float() - stage_content.detach().float()
            ).norm(dim=-1).mean(),
            "hierarchical_stage_retain_mean": retain.detach().float().mean(),
            "hierarchical_stage_promote_attention_entropy": -(promote_prob * promote_prob.log()).sum(dim=-1).mean(),
            "hierarchical_stage_promote_attention_max": promote_prob.max(dim=-1).values.mean(),
            "hierarchical_stage_promoted_norm": promoted.detach().float().norm(dim=-1).mean(),
            "hierarchical_stage_promote_scale": promote_scale.detach().float(),
        }
        metrics["workspace_group_effective_sources"] = torch.exp(metrics["workspace_group_attention_entropy"])
        metrics.update(manager_metrics)
        metrics.update(self.memory_bank.role_attention_metrics(weights, prepared_evidence.ranges))
        for name, (start, stop) in prepared_evidence.ranges.items():
            metrics[f"workspace_{name}_attention"] = weights[..., start:stop].sum(dim=-1).mean()
        transition_mass = [
            metrics[key]
            for key in (
                "workspace_transition_attention",
                "workspace_transition_delta_attention",
                "workspace_transition_effect_attention",
                "workspace_transition_timeline_attention",
            )
            if key in metrics
        ]
        if transition_mass:
            metrics["workspace_transition_total_attention"] = torch.stack(transition_mass).sum()
        return (
            low_for_action,
            next_stage_content,
            stage_for_action,
            low_logit_bias,
            stage_logit_bias,
            metrics,
        )


class LatentCVAEMMDiTBlock(nn.Module):
    """Compact MMDiT-style mixer for CVAE action tokens.

    Action tokens and condition tokens use separate QKV/O and MLP parameters,
    then action queries attend over the concatenated action+condition keys.
    The condition stream is read-only by default to avoid action information
    being written into condition tokens and returning as a shortcut.
    """

    def __init__(self, config: V39PolicyConfig) -> None:
        super().__init__()
        self.config = config
        h = int(config.hidden_size)
        heads = int(config.num_heads)
        if h % heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads for LatentCVAEMMDiTBlock")
        self.hidden_size = h
        self.heads = heads
        self.head_dim = h // heads
        self.action_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.cond_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.action_qkv = nn.Linear(h, 3 * h)
        self.cond_qkv = nn.Linear(h, 3 * h)
        self.action_out = nn.Linear(h, h)
        self.cond_out = nn.Linear(h, h)
        self.action_ffn_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.cond_ffn_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.action_ffn = BiasFreeFFN(h, float(getattr(config, "latent_cvae_ffn_expansion", 2.0)))
        self.cond_ffn = BiasFreeFFN(h, float(getattr(config, "latent_cvae_ffn_expansion", 2.0)))
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
        if not int(getattr(self.config, "latent_cvae_causal_attention", 1)):
            return None
        total = action_len + cond_len
        mask = torch.zeros(action_len, total, device=device, dtype=torch.bool)
        future_action = torch.triu(torch.ones(action_len, action_len, device=device, dtype=torch.bool), diagonal=1)
        mask[:, :action_len] = future_action
        if int(getattr(self.config, "latent_cvae_mmdit_noisy_causal", 1)) and noisy_len > 0:
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

    def __init__(self, config: V39PolicyConfig) -> None:
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

    def __init__(self, config: V39PolicyConfig) -> None:
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

    def __init__(self, config: V39PolicyConfig) -> None:
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

    def __init__(self, config: V39PolicyConfig) -> None:
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

        posterior_used = target_physical is not None
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

    def __init__(self, config: V39PolicyConfig) -> None:
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
        for block in self.blocks:
            action = block(action, cond_time)
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


class TemporalMidcutWorldActionDiT(nn.Module):
    """V38 DiT split into a mid-cut contract trunk and a policy tail."""

    def __init__(self, config: V39PolicyConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        h = int(config.hidden_size)
        self.visual_memory = DenseVisualMemory(config)
        self.rollout_codec = RolloutTargetCodec(config)
        self.seed = UnifiedCanvasSeed(config)
        self.time = TimeEmbedding(h)
        self.content_mod = nn.Sequential(nn.LayerNorm(2 * h), nn.Linear(2 * h, h), nn.SiLU(), nn.Linear(h, h))
        nn.init.normal_(self.content_mod[-1].weight, mean=0.0, std=2e-2)
        nn.init.zeros_(self.content_mod[-1].bias)
        self.content_mod_scale = nn.Parameter(torch.tensor(0.10))
        self.blocks = nn.ModuleList([TemporalDynamicsBoundDiTBlock(config) for _ in range(config.depth)])
        self.midcut_norm = nn.LayerNorm(h)
        self.midcut_heads = MidcutContractHeads(config)
        if int(config.layer_contract_adapters):
            self.layer_contract_heads = nn.ModuleList([
                LayerContractAdapterHeads(config, layer_index=i) for i in range(int(config.depth))
            ])
        else:
            self.layer_contract_heads = nn.ModuleList()
        self.layer_fm_probe = SharedLayerFlowActionProbe(config) if int(config.layer_shared_fm_probe) else None
        self.layer_role_scheduler = LayerRoleScheduler(config)
        self.layer_consequence_cell = RecurrentMilestoneConsequenceCell(config) if int(config.layer_recurrent_consequence) else None
        self.final_norm = nn.LayerNorm(h)
        self.direct_physical_head = CanvasPhysicalVelocityHead(config)
        self.rollout_residual_head = RolloutActionResidualHead(config)
        self.controlled_dynamics = ControlledResidualLatentDynamics(config)
        self.event_probe = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 3))
        self.motion_probe = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 1))
        final_decoder = str(getattr(config, "final_action_decoder", "legacy"))
        if final_decoder == "residual_action_flow":
            self.residual_action_flow_denoiser = V37StyleResidualActionFlowDenoiser(config)
            self.latent_main_action_decoder = None
            self.latent_cvae_action_decoder = None
        elif final_decoder == "layered_residual_action_flow":
            self.residual_action_flow_denoiser = LayeredV37StyleResidualActionFlowDenoiser(config)
            self.latent_main_action_decoder = None
            self.latent_cvae_action_decoder = None
        elif final_decoder == "latent_main_action":
            self.residual_action_flow_denoiser = None
            self.latent_main_action_decoder = HierarchicalLatentMainActionDecoder(config)
            self.latent_cvae_action_decoder = None
        elif final_decoder == "latent_cvae_action":
            self.residual_action_flow_denoiser = None
            self.latent_main_action_decoder = None
            self.latent_cvae_action_decoder = LatentCVAEActionDecoder(config)
        elif final_decoder == "adaptive_recurrent_cvae_action":
            self.residual_action_flow_denoiser = None
            self.latent_main_action_decoder = None
            self.latent_cvae_action_decoder = AdaptiveRecurrentCVAEActionDecoder(config)
        else:
            self.residual_action_flow_denoiser = None
            self.latent_main_action_decoder = None
            self.latent_cvae_action_decoder = None
        if self.latent_cvae_action_decoder is not None or self.latent_main_action_decoder is not None:
            # These readers belong to the legacy action tower. Keep the modules
            # for checkpoint compatibility and the parameter-free pooled()
            # helper, but do not allocate gradients/optimizer state for outputs
            # that the complete latent decoder never consumes.
            self.direct_physical_head.requires_grad_(False)
            self.rollout_residual_head.requires_grad_(False)
            self.motion_probe.requires_grad_(False)

    def _mod_embed(self, canvas: Tensor, visual_memory: Tensor, time_emb: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        summary = torch.cat([canvas.mean(dim=1), visual_memory.mean(dim=1)], dim=-1)
        content_delta = self.content_mod(summary) * self.content_mod_scale.to(device=canvas.device, dtype=canvas.dtype)
        return time_emb + content_delta, content_delta, time_emb

    def _promote_midcut(self, mid: dict[str, Tensor], *, gates: dict[str, Tensor], content_norm: Tensor, time_norm: Tensor) -> dict[str, Tensor]:
        pred = mid["midcut_pred_physical_velocity"]
        effect = mid["midcut_rollout_effect_pred"]
        delta = mid["midcut_rollout_delta_pred"]
        z = _zeros_like_scalar(pred)
        out = {
            **mid,
            "canvas_tokens": mid["midcut_canvas_tokens"],
            "trajectory_tokens": mid["midcut_trajectory_tokens"],
            "rollout_tokens": mid["midcut_rollout_tokens"],
            "register_tokens": mid["midcut_register_tokens"],
            "direct_physical_velocity": mid["midcut_direct_physical_velocity"],
            "rollout_residual_velocity": mid["midcut_rollout_residual_velocity"],
            "rollout_alpha": mid["midcut_rollout_alpha"],
            "pred_physical_velocity": pred,
            "rollout_effect_pred": effect,
            "rollout_base_effect_pred": mid["midcut_rollout_base_effect_pred"],
            "rollout_delta_pred": delta,
            "future_latent_pred": effect,
            "action_effect_pred": effect,
            "event_logits": mid["midcut_event_logits"],
            "motion_logits": mid["midcut_motion_logits"],
            "transition_latent": mid["midcut_transition_latent"],
            "rollout_coeff_abs_mean": z,
            "rollout_neutral_coeff_abs_mean": z,
            "rollout_centered_coeff_abs_mean": z,
            "rollout_basis_norm": z,
            "rollout_delta_norm": mid["midcut_rollout_delta_norm"],
            "rollout_base_norm": z,
            "rollout_delta_gain": mid["midcut_future_gain"],
            "gate_self": gates.get("gate_self", z),
            "gate_visual": gates.get("gate_visual", z),
            "gate_rollout": gates.get("gate_rollout", z),
            "gate_ffn": gates.get("gate_ffn", z),
            "mod_content_norm": content_norm,
            "mod_time_norm": time_norm,
            "mod_content_to_time": content_norm / time_norm.clamp_min(1e-6),
            "midcut_stop": torch.ones((), device=pred.device, dtype=pred.dtype),
        }
        return out

    def forward(
        self,
        noisy_physical: Tensor,
        time: Tensor,
        visual: Tensor,
        state_history: Tensor,
        state: Tensor,
        executed_history: Tensor,
        proposal_tokens: Tensor,
        proposal_keep: Tensor | None = None,
        *,
        stop_at_midcut: bool = False,
        consequence_physical: Tensor | None = None,
        cvae_target_physical: Tensor | None = None,
        enable_layer_contracts: bool = True,
        enable_final_action_decoder: bool = True,
    ) -> dict[str, Tensor]:
        cfg = self.config
        if proposal_keep is None:
            proposal_keep = torch.ones(noisy_physical.shape[0], device=noisy_physical.device, dtype=noisy_physical.dtype)
        if consequence_physical is None:
            consequence_physical = noisy_physical
        else:
            consequence_physical = consequence_physical.to(device=noisy_physical.device, dtype=noisy_physical.dtype)
        visual_memory = self.visual_memory(visual)
        rollout_init = self.rollout_codec.rollout_init(visual)
        canvas, slices = self.seed(
            noisy_physical=noisy_physical,
            state=state,
            state_history=state_history,
            executed_history=executed_history,
            proposal_tokens=proposal_tokens,
            proposal_keep=proposal_keep,
            rollout_init=rollout_init,
        )
        rollout_seed = canvas[:, slices["rollout"]].detach()
        time_emb = self.time(time.to(dtype=canvas.dtype))
        gate_rows: list[dict[str, Tensor]] = []
        content_norm_rows: list[Tensor] = []
        time_norm_rows: list[Tensor] = []
        midcut: dict[str, Tensor] | None = None
        layer_contracts: list[dict[str, Tensor]] = []
        # The latent-main decoder is the final action path, so inference/eval
        # must still materialize layer contracts even when callers disable
        # auxiliary contract evaluation for speed.  We do not add extra losses;
        # we only expose the latents needed by the action decoder.
        final_decoder = str(getattr(cfg, "final_action_decoder", "legacy"))
        force_layer_contracts = (
            final_decoder == "latent_main_action"
            or (final_decoder in {"latent_cvae_action", "adaptive_recurrent_cvae_action"} and bool(int(getattr(cfg, "latent_cvae_layer_memory", 1))))
        )
        effective_layer_contracts = bool(enable_layer_contracts) or force_layer_contracts
        cut = int(cfg.midcut_layer)
        contract_grad_scale = float(getattr(cfg, "layer_contract_grad_scale", 1.0))
        for index, block in enumerate(self.blocks, start=1):
            mod_emb, content_delta, time_row = self._mod_embed(canvas, visual_memory, time_emb)
            content_norm_rows.append(content_delta.float().norm(dim=-1).mean())
            time_norm_rows.append(time_row.float().norm(dim=-1).mean())
            canvas, gates = block(canvas, visual_memory, mod_emb, slices)
            gate_rows.append(gates)
            if effective_layer_contracts and len(self.layer_contract_heads) > 0:
                contract_canvas = _scaled_contract_view(canvas, contract_grad_scale)
                layer_entry = self.layer_contract_heads[index - 1](contract_canvas, slices)
                if self.layer_consequence_cell is not None:
                    # V40: split the layer contract into an explicit world-latent
                    # object and an action-causal object.  Lower layers lean on
                    # the causal branch; upper layers lean on the latent branch.
                    # We keep the old direct outputs for forensics only.
                    latent_effect = layer_entry["rollout_effect_pred"]
                    latent_delta = layer_entry["rollout_delta_pred"]
                    cons = self.layer_consequence_cell(
                        rollout_tokens=layer_entry["rollout_tokens"],
                        action_physical=consequence_physical,
                        state_tokens=layer_entry.get("state_tokens"),
                        state_history_tokens=layer_entry.get("state_history_tokens"),
                        executed_tokens=layer_entry.get("executed_tokens"),
                        trajectory_tokens=layer_entry.get("trajectory_tokens"),
                        proposal_tokens=layer_entry.get("proposal_tokens"),
                        layer_index=index - 1,
                    )
                    causal_gain, latent_gain = self.layer_role_scheduler(
                        index - 1, device=latent_effect.device, dtype=latent_effect.dtype,
                    )
                    causal_effect = cons["milestone_rollout_effect_pred"]
                    causal_delta = cons["milestone_rollout_delta_pred"]
                    if latent_effect.shape[1] != causal_effect.shape[1]:
                        latent_effect_for_mix = latent_effect[:, : causal_effect.shape[1]]
                        latent_delta_for_mix = latent_delta[:, : causal_delta.shape[1]]
                    else:
                        latent_effect_for_mix = latent_effect
                        latent_delta_for_mix = latent_delta
                    layer_entry["latent_rollout_effect_pred"] = latent_effect
                    layer_entry["latent_rollout_delta_pred"] = latent_delta
                    layer_entry["causal_rollout_effect_pred"] = causal_effect
                    layer_entry["causal_rollout_delta_pred"] = causal_delta
                    layer_entry["direct_rollout_effect_pred"] = latent_effect
                    layer_entry["direct_rollout_delta_pred"] = latent_delta
                    # V40.1: one unified intervention-latent head is the
                    # supervised object.  The weak direct latent readout remains
                    # only for forensics; it is no longer mixed into the main
                    # rollout prediction where it can blur causal semantics.
                    layer_entry["rollout_effect_pred"] = causal_effect
                    layer_entry["rollout_delta_pred"] = causal_delta
                    layer_entry["policy_effect_tokens"] = cons["milestone_policy_effect_tokens"]
                    layer_entry["policy_effect_time_tokens"] = cons["milestone_policy_time_tokens"]
                    layer_entry["milestone_step_delta_pred"] = cons["milestone_step_delta_pred"]
                    layer_entry["unified_intervention_latent_pred"] = cons["milestone_intervention_latent_pred"]
                    layer_entry["neutral_latent_pred"] = cons["milestone_neutral_latent_pred"]
                    layer_entry["layer_causal_gain"] = causal_gain.detach().float()
                    layer_entry["layer_latent_gain"] = latent_gain.detach().float()
                    if bool(enable_layer_contracts) and int(getattr(cfg, "layer_zero_base_diagnostic", 0)):
                        # Loss-free shortcut probe.  If zeroing the rollout
                        # tokens barely moves the consequence output, the cell
                        # is probably relying on action features instead of the
                        # state/rollout context.
                        with torch.no_grad():
                            cons_zero = self.layer_consequence_cell(
                                rollout_tokens=torch.zeros_like(layer_entry["rollout_tokens"]),
                                action_physical=consequence_physical,
                                state_tokens=layer_entry.get("state_tokens"),
                                state_history_tokens=layer_entry.get("state_history_tokens"),
                                executed_tokens=layer_entry.get("executed_tokens"),
                                trajectory_tokens=layer_entry.get("trajectory_tokens"),
                                proposal_tokens=layer_entry.get("proposal_tokens"),
                                layer_index=index - 1,
                            )
                            base_eff = cons["milestone_rollout_effect_pred"].detach().float()
                            zero_eff = cons_zero["milestone_rollout_effect_pred"].float()
                            zero_shift = (
                                (base_eff - zero_eff).norm(dim=-1).mean()
                                / base_eff.norm(dim=-1).mean().clamp_min(1e-6)
                            )
                        layer_entry["consequence_zero_base_shift"] = zero_shift
                    if bool(enable_layer_contracts) and int(getattr(cfg, "layer_state_counterfactual", 0)) and int(layer_entry["rollout_tokens"].shape[0]) > 1:
                        flat_state = layer_entry["rollout_tokens"].detach().float().flatten(1)
                        dist_state = torch.cdist(flat_state, flat_state, p=2)
                        eye_state = torch.eye(dist_state.shape[0], device=dist_state.device, dtype=torch.bool)
                        dist_state = dist_state.masked_fill(eye_state, -1.0)
                        state_perm = dist_state.argmax(dim=1)
                        cons_state = self.layer_consequence_cell(
                            rollout_tokens=layer_entry["rollout_tokens"][state_perm],
                            action_physical=consequence_physical,
                            state_tokens=None if layer_entry.get("state_tokens") is None else layer_entry["state_tokens"][state_perm],
                            state_history_tokens=None if layer_entry.get("state_history_tokens") is None else layer_entry["state_history_tokens"][state_perm],
                            executed_tokens=None if layer_entry.get("executed_tokens") is None else layer_entry["executed_tokens"][state_perm],
                            trajectory_tokens=None if layer_entry.get("trajectory_tokens") is None else layer_entry["trajectory_tokens"][state_perm],
                            proposal_tokens=None if layer_entry.get("proposal_tokens") is None else layer_entry["proposal_tokens"][state_perm],
                            layer_index=index - 1,
                        )
                        layer_entry["rollout_effect_pred_shuffle_state"] = cons_state["milestone_rollout_effect_pred"]
                        layer_entry["rollout_delta_pred_shuffle_state"] = cons_state["milestone_rollout_delta_pred"]
                        layer_entry["milestone_step_delta_pred_shuffle_state"] = cons_state["milestone_step_delta_pred"]
                        layer_entry["policy_effect_tokens_shuffle_state"] = cons_state["milestone_policy_effect_tokens"]
                    if int(getattr(cfg, "layer_causal_event_from_effect", 1)):
                        event_src = cons["milestone_policy_time_tokens"]
                        layer_entry["event_logits"] = self.event_probe(event_src)
                    for key in ("milestone_gate_mean", "milestone_step_delta_norm", "milestone_effect_norm", "milestone_effect_std", "milestone_effect_gain"):
                        layer_entry[key] = cons[key]
                if self.layer_fm_probe is not None:
                    probe_velocity = self.layer_fm_probe(
                        trajectory_pooled=layer_entry["trajectory_pooled"],
                        rollout_effect_pred=layer_entry["rollout_effect_pred"],
                        rollout_delta_pred=layer_entry["rollout_delta_pred"],
                        noisy_physical=noisy_physical,
                        time=time,
                    )
                    # In V39.2/V39.3 the action-flow probe is downstream of
                    # the layer latent.  It replaces the per-layer direct
                    # action head for contract losses, while remaining shared
                    # across all layers.
                    layer_entry["pred_physical_velocity"] = probe_velocity
                    layer_entry["direct_physical_velocity"] = probe_velocity
                    layer_entry["layer_fm_probe_velocity"] = probe_velocity
                layer_contracts.append(layer_entry)
            if index == cut:
                mid_canvas = self.midcut_norm(canvas)
                midcut = self.midcut_heads(mid_canvas, slices)
                if stop_at_midcut:
                    content_norm = torch.stack(content_norm_rows).mean() if content_norm_rows else _zeros_like_scalar(canvas)
                    time_norm = torch.stack(time_norm_rows).mean() if time_norm_rows else _zeros_like_scalar(canvas)
                    gate_mean = {
                        key: torch.stack([row[key] for row in gate_rows]).mean()
                        for key in ("gate_self", "gate_visual", "gate_rollout", "gate_ffn")
                    }
                    promoted = self._promote_midcut(midcut, gates=gate_mean, content_norm=content_norm, time_norm=time_norm)
                    if layer_contracts:
                        promoted["layer_contracts"] = layer_contracts
                    return promoted
        if midcut is None:
            # Defensive fallback; validate() should prevent this.
            midcut = self.midcut_heads(self.midcut_norm(canvas), slices)
        canvas = self.final_norm(canvas)
        trajectory = canvas[:, slices["trajectory"]]
        rollout = canvas[:, slices["rollout"]]
        registers = canvas[:, slices["registers"]]
        trajectory_pooled = self.direct_physical_head.pooled(trajectory)
        context_kv = torch.cat([
            canvas[:, slices["state"]],
            canvas[:, slices["state_history"]],
            canvas[:, slices["executed"]],
            canvas[:, slices["proposal"]],
        ], dim=1)
        if str(getattr(cfg, "controlled_base_mode", "learned")) == "fixed_zero":
            dynamics = self.controlled_dynamics(
                rollout_init.to(device=rollout.device, dtype=rollout.dtype),
                context_kv,
                action_tokens=trajectory,
                transition_tokens=rollout,
            )
        else:
            # Preserve the exact learned-base path for historical checkpoints.
            dynamics = self.controlled_dynamics(
                rollout,
                context_kv,
                action_tokens=trajectory,
            )
        controlled_delta = dynamics["rollout_delta_pred"]
        rollout_effect_pred = dynamics["rollout_effect_pred"]
        event_context = _rollout_tokens_to_action_horizon(controlled_delta, cfg)
        decoder_mode = str(getattr(cfg, "final_action_decoder", "legacy"))
        direct_velocity: Tensor | None = None
        rollout_residual_velocity: Tensor | None = None
        rollout_alpha: Tensor | None = None
        legacy_velocity: Tensor | None = None
        pred_physical_velocity: Tensor
        legacy_event_logits: Tensor
        legacy_motion_logits: Tensor
        residual_action_flow: dict[str, Tensor] | None = None
        latent_main_action: dict[str, Tensor] | None = None
        latent_cvae_action: dict[str, Tensor] | None = None
        if not enable_final_action_decoder:
            # Counterfactual rollout branches consume only dynamics and layer
            # contracts. Running the final CVAE/MMDiT tower here duplicated a
            # full prior decode whose action output was immediately discarded.
            pred_physical_velocity = torch.zeros_like(noisy_physical)
            legacy_event_logits = event_context.new_zeros(
                int(event_context.shape[0]), int(event_context.shape[1]), 3
            )
            legacy_motion_logits = event_context.new_zeros(
                int(event_context.shape[0]), int(event_context.shape[1])
            )
        elif self.latent_cvae_action_decoder is not None:
            context_memory = [
                canvas[:, slices["state"]],
                canvas[:, slices["state_history"]],
                canvas[:, slices["executed"]],
                canvas[:, slices["proposal"]],
            ] if int(getattr(cfg, "latent_cvae_context_memory", 0)) else None
            # Rollout has its own full-resolution workspace source. Transition
            # memory therefore carries only explicit consequence semantics and
            # does not duplicate the same rollout grid through a pooled path.
            if int(getattr(cfg, "latent_cvae_transition_memory", 1)):
                if str(getattr(cfg, "controlled_base_mode", "learned")) == "fixed_zero":
                    # effect == delta under a fixed-zero base. Feeding both would
                    # duplicate one condition under two semantic names.
                    transition_memory = [controlled_delta, event_context]
                else:
                    transition_memory = [controlled_delta, rollout_effect_pred, event_context]
            else:
                transition_memory = None
            latent_cvae_action = self.latent_cvae_action_decoder(
                noisy_physical=noisy_physical,
                time=time,
                trajectory_tokens=trajectory_pooled,
                trajectory_workspace_tokens=trajectory,
                rollout_tokens=rollout,
                context_memory=context_memory,
                transition_memory=transition_memory,
                visual_memory=visual_memory if int(getattr(cfg, "latent_cvae_visual_memory", 0)) else None,
                layer_contracts=layer_contracts,
                target_physical=cvae_target_physical,
            )
            pred_physical_velocity = latent_cvae_action["pred_velocity"]
            legacy_event_logits = latent_cvae_action["event_logits"]
            legacy_motion_logits = latent_cvae_action["motion_logits"]
        elif self.latent_main_action_decoder is not None:
            context_memory = context_kv if int(getattr(cfg, "latent_action_context_memory", 0)) else None
            transition_parts = [rollout, controlled_delta, event_context]
            if str(getattr(cfg, "controlled_base_mode", "learned")) != "fixed_zero":
                transition_parts.insert(2, rollout_effect_pred)
            transition_memory = torch.cat(transition_parts, dim=1) if int(getattr(cfg, "latent_action_transition_memory", 1)) else None
            latent_main_action = self.latent_main_action_decoder(
                noisy_physical=noisy_physical,
                time=time,
                trajectory_tokens=trajectory_pooled,
                context_memory=context_memory,
                transition_memory=transition_memory,
                visual_memory=visual_memory if int(getattr(cfg, "latent_action_visual_memory", 0)) else None,
                layer_contracts=layer_contracts,
            )
            pred_physical_velocity = latent_main_action["pred_velocity"]
            legacy_event_logits = latent_main_action["event_logits"]
            legacy_motion_logits = latent_main_action["motion_logits"]
        else:
            # Legacy action readers are needed only by legacy/residual decoder
            # modes. CVAE/MMDiT is a complete final path, so computing a second
            # rollout-to-action tower there wastes memory and creates misleading
            # anchor diagnostics for a path that deployment never uses.
            direct_velocity = self.direct_physical_head(trajectory)
            rollout_residual_velocity, rollout_alpha = self.rollout_residual_head(trajectory_pooled, controlled_delta)
            legacy_velocity = direct_velocity + rollout_residual_velocity
            pred_physical_velocity = legacy_velocity
            legacy_event_logits = self.event_probe(event_context)
            legacy_motion_logits = self.motion_probe(trajectory_pooled.detach()).squeeze(-1)
        if self.latent_cvae_action_decoder is None and self.latent_main_action_decoder is None and self.residual_action_flow_denoiser is not None:
            assert legacy_velocity is not None
            if decoder_mode == "layered_residual_action_flow":
                context_memory = torch.cat([context_kv, registers], dim=1) if int(getattr(cfg, "action_flow_residual_context_memory", 1)) else context_kv
                transition_parts = [rollout, controlled_delta, event_context]
                if str(getattr(cfg, "controlled_base_mode", "learned")) != "fixed_zero":
                    transition_parts.insert(2, rollout_effect_pred)
                transition_memory = torch.cat(transition_parts, dim=1) if int(getattr(cfg, "action_flow_residual_transition_memory", 1)) else None
                residual_action_flow = self.residual_action_flow_denoiser(
                    noisy_physical=noisy_physical,
                    time=time,
                    trajectory_pooled=trajectory_pooled,
                    context_memory=context_memory,
                    transition_memory=transition_memory,
                    visual_memory=visual_memory if int(getattr(cfg, "action_flow_residual_visual_memory", 1)) else None,
                    layer_contracts=layer_contracts,
                )
            else:
                memory_parts: list[Tensor] = []
                if int(getattr(cfg, "action_flow_residual_context_memory", 1)):
                    memory_parts.append(context_kv)
                    memory_parts.append(registers)
                if int(getattr(cfg, "action_flow_residual_transition_memory", 1)):
                    memory_parts.extend([rollout, controlled_delta, rollout_effect_pred, event_context])
                if int(getattr(cfg, "action_flow_residual_visual_memory", 1)):
                    memory_parts.append(visual_memory)
                if int(getattr(cfg, "action_flow_residual_layer_memory", 1)) and layer_contracts:
                    last_layer = layer_contracts[-1]
                    for key in ("policy_effect_time_tokens", "policy_effect_tokens", "rollout_effect_pred", "rollout_delta_pred"):
                        value = last_layer.get(key)
                        if isinstance(value, Tensor) and value.ndim == 3 and value.shape[-1] == cfg.hidden_size:
                            memory_parts.append(value)
                residual_memory = torch.cat(memory_parts, dim=1) if memory_parts else context_kv
                residual_action_flow = self.residual_action_flow_denoiser(
                    noisy_physical=noisy_physical,
                    time=time,
                    trajectory_pooled=trajectory_pooled,
                    memory=residual_memory,
                )
            pred_physical_velocity = legacy_velocity + residual_action_flow["residual_velocity"]
            legacy_event_logits = legacy_event_logits + residual_action_flow["event_delta_logits"]
            legacy_motion_logits = legacy_motion_logits + residual_action_flow["motion_delta_logits"]
        gate_mean = {
            key: torch.stack([row[key] for row in gate_rows]).mean() if gate_rows else _zeros_like_scalar(canvas)
            for key in ("gate_self", "gate_visual", "gate_rollout", "gate_ffn")
        }
        content_norm = torch.stack(content_norm_rows).mean() if content_norm_rows else _zeros_like_scalar(canvas)
        time_norm = torch.stack(time_norm_rows).mean() if time_norm_rows else _zeros_like_scalar(canvas)
        with torch.no_grad():
            rollout_seed_final = self.final_norm(rollout_seed.to(device=rollout.device, dtype=rollout.dtype))
            rollout_deep_update_norm = (rollout.detach() - rollout_seed_final).float().norm(dim=-1).mean()
        out = {
            **midcut,
            "layer_contracts": layer_contracts,
            "canvas_tokens": canvas,
            "trajectory_tokens": trajectory,
            "rollout_tokens": rollout,
            "register_tokens": registers,
            "rollout_deep_update_norm": rollout_deep_update_norm,
            "rollout_deep_token_norm": rollout.detach().float().norm(dim=-1).mean(),
            "pred_physical_velocity": pred_physical_velocity,
            "action_flow_residual_velocity": (
                torch.zeros_like(pred_physical_velocity)
                if residual_action_flow is None else residual_action_flow["residual_velocity"]
            ),
            "action_flow_residual_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if residual_action_flow is None else residual_action_flow["residual_norm"]
            ),
            "action_flow_raw_residual_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if residual_action_flow is None else residual_action_flow["raw_residual_norm"]
            ),
            "action_flow_residual_alpha_mean": (
                _zeros_like_scalar(pred_physical_velocity)
                if residual_action_flow is None else residual_action_flow["alpha_mean"]
            ),
            "action_flow_stage_router_entropy": (
                _zeros_like_scalar(pred_physical_velocity)
                if residual_action_flow is None else residual_action_flow.get("stage_router_entropy", _zeros_like_scalar(pred_physical_velocity))
            ),
            "action_flow_stage_router_max": (
                _zeros_like_scalar(pred_physical_velocity)
                if residual_action_flow is None else residual_action_flow.get("stage_router_max", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_action_stage_router_entropy": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_main_action is None else latent_main_action.get("stage_router_entropy", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_action_stage_router_max": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_main_action is None else latent_main_action.get("stage_router_max", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_action_gripper_gate_mean": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_main_action is None else latent_main_action.get("gripper_gate_mean", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_action_layer_memory_count": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_main_action is None else latent_main_action.get("layer_memory_count", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_action_temporal_update_mean": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_main_action is None else latent_main_action.get("temporal_action_update_mean", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_action_temporal_near_depth": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_main_action is None else latent_main_action.get("temporal_near_depth", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_action_temporal_mid_depth": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_main_action is None else latent_main_action.get("temporal_mid_depth", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_kl": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_kl", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_prior_std": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_prior_std", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_post_std": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_post_std", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_z_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_z_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_condition_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_condition_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_condition_scan_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_condition_scan_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_condition_lateral_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_condition_lateral_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_layer_summary_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_layer_summary_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_transition_condition_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_transition_condition_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_transition_source_raw_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_transition_source_raw_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_rollout_token_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_rollout_token_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_rollout_token_count": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_rollout_token_count", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_consequence_scale_mean": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_consequence_scale_mean", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_consequence_gate_preference": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_consequence_gate_preference", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_consequence_mix_ratio": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_consequence_mix_ratio", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_posterior_used": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_posterior_used", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_gripper_gate_mean": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("gripper_gate_mean", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_layer_memory_count": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("layer_memory_count", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_prior_z_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_prior_z_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_post_z_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_post_z_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_mu_gap": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_mu_gap", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_prior_pred_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_prior_pred_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_post_pred_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_post_pred_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_post_gripper_gate_mean": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_post_gripper_gate_mean", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_refine_update_mean": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_refine_update_mean", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_noisy_gate_mean": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_noisy_gate_mean", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_noisy_branch_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_noisy_branch_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_noisy_branch_ratio": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_noisy_branch_ratio", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_route_entropy": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_route_entropy", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_route_max": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_route_max", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_route_effective_slots": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_route_effective_slots", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_progress_entropy": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_progress_entropy", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_progress_max": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_progress_max", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_progress_effective_slots": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_progress_effective_slots", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_progress_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_progress_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_continue_mean": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_continue_mean", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_prefix_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_prefix_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_progress_seed_entropy": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_progress_seed_entropy", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_progress_seed_max": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_progress_seed_max", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_progress_seed_effective_slots": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_progress_seed_effective_slots", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_progress_seed_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_progress_seed_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_route_temperature_mean": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_route_temperature_mean", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_semantic_bias_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_semantic_bias_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_output_adapter_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_output_adapter_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_function_delta_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_function_delta_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_base_highfreq_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_base_highfreq_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_refine_step_bias_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_refine_step_bias_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_capsule_layer_entropy": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_capsule_layer_entropy", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_capsule_layer_max": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_capsule_layer_max", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_capsule_layer_effective_slots": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_capsule_layer_effective_slots", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_condition_strength_mean": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_condition_strength_mean", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_condition_strength_std": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_condition_strength_std", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_condition_strength_max": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_condition_strength_max", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_condition_strength_min": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_condition_strength_min", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_condition_residual_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_condition_residual_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_context_direction_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_context_direction_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_micro_step_mean": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_micro_step_mean", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_micro_step_std": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_micro_step_std", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_micro_progress_mean": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_micro_progress_mean", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_micro_kp_mean": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_micro_kp_mean", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_micro_kd_mean": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_micro_kd_mean", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_micro_feedforward_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_micro_feedforward_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_micro_feedback_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_micro_feedback_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_micro_damping_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_micro_damping_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_micro_function_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_micro_function_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_micro_control_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_micro_control_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_micro_update_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_micro_update_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_micro_heun_error": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_micro_heun_error", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_micro_refine_block_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_micro_refine_block_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_regularizer": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_regularizer", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_adaptive_route_entropy_regularizer": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_adaptive_route_entropy_regularizer", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_mmdit_action_update_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_mmdit_action_update_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_mmdit_cond_update_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_mmdit_cond_update_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_mmdit_action_cond_attention": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_mmdit_action_cond_attention", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_mmdit_action_noisy_attention": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_mmdit_action_noisy_attention", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_mmdit_action_rollout_attention": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_mmdit_action_rollout_attention", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_mmdit_action_rollout_enrichment": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_mmdit_action_rollout_enrichment", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_mmdit_action_token_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_mmdit_action_token_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_mmdit_condition_token_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_mmdit_condition_token_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "latent_cvae_mmdit_noisy_token_norm": (
                _zeros_like_scalar(pred_physical_velocity)
                if latent_cvae_action is None else latent_cvae_action.get("cvae_mmdit_noisy_token_norm", _zeros_like_scalar(pred_physical_velocity))
            ),
            "rollout_effect_pred": rollout_effect_pred,
            "rollout_base_effect_pred": dynamics["rollout_base_effect_pred"],
            "rollout_delta_pred": controlled_delta,
            "rollout_coeff_abs_mean": dynamics["rollout_coeff_abs_mean"],
            "rollout_neutral_coeff_abs_mean": dynamics["rollout_neutral_coeff_abs_mean"],
            "rollout_centered_coeff_abs_mean": dynamics["rollout_centered_coeff_abs_mean"],
            "rollout_basis_norm": dynamics["rollout_basis_norm"],
            "rollout_delta_norm": dynamics["rollout_delta_norm"],
            "rollout_base_norm": dynamics["rollout_base_norm"],
            "rollout_decomposition_expansion_ratio": dynamics["rollout_decomposition_expansion_ratio"],
            "rollout_base_is_fixed_zero": dynamics["rollout_base_is_fixed_zero"],
            "rollout_delta_gain": dynamics["rollout_delta_gain"],
            "future_latent_pred": rollout_effect_pred,
            "action_effect_pred": rollout_effect_pred,
            "event_logits": legacy_event_logits,
            "motion_logits": legacy_motion_logits,
            "transition_latent": event_context,
            "gate_self": gate_mean["gate_self"],
            "gate_visual": gate_mean["gate_visual"],
            "gate_rollout": gate_mean["gate_rollout"],
            "gate_ffn": gate_mean["gate_ffn"],
            "mod_content_norm": content_norm,
            "mod_time_norm": time_norm,
            "mod_content_to_time": content_norm / time_norm.clamp_min(1e-6),
            "midcut_stop": torch.zeros((), device=canvas.device, dtype=canvas.dtype),
        }
        if legacy_velocity is not None:
            assert direct_velocity is not None
            assert rollout_residual_velocity is not None
            assert rollout_alpha is not None
            out.update({
                "direct_physical_velocity": direct_velocity,
                "rollout_residual_velocity": rollout_residual_velocity,
                "legacy_physical_velocity": legacy_velocity,
                "rollout_alpha": rollout_alpha,
            })
        if latent_cvae_action is not None:
            for key, value in latent_cvae_action.items():
                if key.startswith("cvae_") and isinstance(value, Tensor):
                    out.setdefault(f"latent_{key}", value)
        if latent_cvae_action is not None and "post_pred_velocity" in latent_cvae_action:
            out.update({
                "post_pred_velocity": latent_cvae_action["post_pred_velocity"],
                "post_event_logits": latent_cvae_action.get("post_event_logits", legacy_event_logits),
                "post_motion_logits": latent_cvae_action.get("post_motion_logits", legacy_motion_logits),
            })
        if latent_cvae_action is not None:
            for key in (
                "cvae_adaptive_micro_controller_norm",
                "cvae_adaptive_micro_pred_velocity",
                "cvae_adaptive_micro_event_logits",
                "cvae_adaptive_micro_supervision_logits",
            ):
                if key in latent_cvae_action:
                    out[f"latent_{key}"] = latent_cvae_action[key]
        return out

    @torch.no_grad()
    def target_rollout_effect(self, visual: Tensor, target_visual: Tensor) -> Tensor:
        return self.rollout_codec.target_effect(visual, target_visual)


class V39PolicySystem(nn.Module):
    def __init__(self, policy_config: V39PolicyConfig) -> None:
        super().__init__()
        self.policy_config = policy_config
        self.codec = PhysicalActionCodec(policy_config)
        self.proposal = RejectableHistoryProposal(policy_config)
        self.planner = TemporalMidcutWorldActionDiT(policy_config)

    def _policy_forward(
        self,
        noisy_physical: Tensor,
        time: Tensor,
        visual: Tensor,
        state_history: Tensor,
        state: Tensor,
        executed_history: Tensor,
        proposal_tokens: Tensor,
        proposal_keep: Tensor,
        *,
        stop_at_midcut: bool = False,
        consequence_physical: Tensor | None = None,
        cvae_target_physical: Tensor | None = None,
        enable_layer_contracts: bool = True,
        enable_final_action_decoder: bool = True,
    ) -> dict[str, Tensor]:
        return self.planner(
            noisy_physical,
            time,
            visual,
            state_history,
            state,
            executed_history,
            proposal_tokens,
            proposal_keep,
            stop_at_midcut=stop_at_midcut,
            consequence_physical=consequence_physical,
            cvae_target_physical=cvae_target_physical,
            enable_layer_contracts=enable_layer_contracts,
            enable_final_action_decoder=enable_final_action_decoder,
        )

    @torch.no_grad()
    def build_rollout_target_pack(self, visual: Tensor, target_visual: Tensor) -> dict[str, Tensor]:
        target = self.planner.target_rollout_effect(visual, target_visual).detach()
        return {"rollout_effect_target": target, "future_latent_target": target, "action_effect_target": target}

    def flow_training_forward(
        self,
        visual: Tensor,
        state_history: Tensor,
        executed_history: Tensor,
        state: Tensor,
        target_action: Tensor,
        *,
        action_state: Tensor | None = None,
        target_visual: Tensor | None = None,
        rollout_target_pack: dict[str, Tensor] | None = None,
        future_training_pack: dict[str, Tensor] | None = None,
        proposal_dropout: float | None = None,
        make_counterfactuals: bool = True,
        stop_at_midcut: bool = False,
    ) -> dict[str, Tensor]:
        del future_training_pack
        proposal = self.proposal(executed_history)
        if self.codec.uses_arm_manifold and action_state is None:
            raise ValueError(
                "manifold_native training requires action_state in action-normalizer coordinates"
            )
        codec_state = state if action_state is None else action_state
        target_physical = self.codec.encode(target_action, codec_state)
        noise = self.codec.sample_noise(
            target_physical.shape[0],
            device=target_physical.device,
            dtype=target_physical.dtype,
            action_state=codec_state,
        )
        t = torch.rand(target_physical.shape[0], device=target_physical.device, dtype=target_physical.dtype)
        noisy_physical = (1 - t[:, None, None]) * target_physical + t[:, None, None] * noise
        target_physical_velocity = noise - target_physical
        drop = self.policy_config.proposal_dropout if proposal_dropout is None else float(proposal_dropout)
        keep = (torch.rand(target_physical.shape[0], device=target_physical.device) >= drop).to(target_physical.dtype)

        consequence_input = noisy_physical
        preview_velocity: Tensor | None = None
        use_self_condition = (
            int(getattr(self.policy_config, "action_consequence_self_condition", 0))
            and int(getattr(self.policy_config, "layer_recurrent_consequence", 0))
            and int(getattr(self.policy_config, "layer_contract_adapters", 0))
        )
        if use_self_condition:
            with torch.no_grad():
                preview = self._policy_forward(
                    noisy_physical.detach(),
                    t.detach(),
                    visual,
                    state_history,
                    state,
                    executed_history,
                    proposal["tokens"].detach(),
                    keep,
                    stop_at_midcut=stop_at_midcut,
                    consequence_physical=noisy_physical.detach(),
                    cvae_target_physical=None,
                    enable_layer_contracts=False,
                )
                preview_velocity = preview["pred_physical_velocity"].detach()
                consequence_input = (noisy_physical - t[:, None, None] * preview_velocity).detach()
                del preview

        action_policy = self._policy_forward(
            noisy_physical,
            t,
            visual,
            state_history,
            state,
            executed_history,
            proposal["tokens"].detach(),
            keep,
            stop_at_midcut=stop_at_midcut,
            consequence_physical=consequence_input,
            cvae_target_physical=target_physical,
        )
        # V70 (H3 fix): every training-time clean estimate is projected onto the
        # physical manifold BEFORE any decode/loss use, matching deployment
        # where each integration step is projected.  This closes the null
        # arbitrage channel (arm decode blend was null-sensitive) and aligns
        # the train/deploy decode geometry.  Null components remain fully
        # visible to the velocity-space flow/null losses.
        clean_physical_estimate = self.codec.project_physical(
            noisy_physical - t[:, None, None] * action_policy["pred_physical_velocity"],
            codec_state,
        )
        decoded_action = self.codec.decode(clean_physical_estimate, codec_state)
        if "post_pred_velocity" in action_policy:
            post_clean = self.codec.project_physical(
                noisy_physical - t[:, None, None] * action_policy["post_pred_velocity"],
                codec_state,
            )
            action_policy["post_clean_physical_estimate"] = post_clean
            action_policy["post_pred_action_estimate"] = self.codec.decode(post_clean, codec_state)
        out = {
            **action_policy,
            "pred_physical_velocity": action_policy["pred_physical_velocity"],
            "target_physical_velocity": target_physical_velocity,
            "target_physical": target_physical,
            "clean_physical_estimate": clean_physical_estimate,
            "proposal_action": proposal["action"],
            "time": t,
            "noisy_physical_action": noisy_physical,
            "source_physical_noise": noise,
            "pred_action_estimate": decoded_action,
            "future_conditioned_action_loss": torch.zeros((), device=target_physical.device, dtype=target_physical.dtype),
        }
        if preview_velocity is not None:
            out["consequence_self_condition"] = torch.ones((), device=target_physical.device, dtype=target_physical.dtype)
            out["consequence_self_condition_target_mse"] = (
                consequence_input.float() - target_physical.detach().float()
            ).square().mean()
            out["consequence_self_condition_noisy_mse"] = (
                consequence_input.float() - noisy_physical.detach().float()
            ).square().mean()
            out["consequence_preview_flow"] = (
                preview_velocity.float() - target_physical_velocity.detach().float()
            ).square().mean()
        if "midcut_pred_physical_velocity" in action_policy:
            mid_clean = self.codec.project_physical(
                noisy_physical - t[:, None, None] * action_policy["midcut_pred_physical_velocity"],
                codec_state,
            )
            out["midcut_clean_physical_estimate"] = mid_clean
            out["midcut_pred_action_estimate"] = self.codec.decode(mid_clean, codec_state)
        if "layer_contracts" in action_policy:
            for entry in action_policy["layer_contracts"]:
                clean = self.codec.project_physical(
                    noisy_physical - t[:, None, None] * entry["pred_physical_velocity"],
                    codec_state,
                )
                entry["clean_physical_estimate"] = clean
                entry["pred_action_estimate"] = self.codec.decode(clean, codec_state)

        pack = rollout_target_pack
        if pack is None and target_visual is not None:
            pack = self.build_rollout_target_pack(visual, target_visual)
        def _merge_layer_contract_counterfactuals(variant: dict[str, Tensor], suffix: str) -> None:
            base_layers = out.get("layer_contracts")
            variant_layers = variant.get("layer_contracts")
            if not isinstance(base_layers, list) or not isinstance(variant_layers, list):
                return
            for base_entry, var_entry in zip(base_layers, variant_layers):
                for key in (
                    "rollout_effect_pred", "rollout_delta_pred", "rollout_base_effect_pred",
                    "milestone_step_delta_pred", "causal_rollout_effect_pred", "causal_rollout_delta_pred",
                    "policy_effect_tokens", "unified_intervention_latent_pred", "neutral_latent_pred",
                    "rollout_effect_pred_shuffle_state", "rollout_delta_pred_shuffle_state",
                    "milestone_step_delta_pred_shuffle_state",
                ):
                    if key in var_entry:
                        base_entry[f"{key}_{suffix}"] = var_entry[key]

        if pack is not None:
            target = pack["rollout_effect_target"].to(device=target_physical.device, dtype=action_policy["rollout_effect_pred"].dtype)
            out["rollout_effect_target"] = target
            out["future_latent_target"] = target
            out["future_latent_velocity_target"] = target
            out["action_effect_target"] = target
            if make_counterfactuals:
                hold_action = codec_state[:, None].expand_as(target_action)
                hold_physical = self.codec.encode(hold_action, codec_state)
                hold_noisy = (1 - t[:, None, None]) * hold_physical + t[:, None, None] * noise
                hold_policy = self._policy_forward(
                    hold_noisy.detach(),
                    t.detach(),
                    visual,
                    state_history,
                    state,
                    executed_history,
                    proposal["tokens"].detach(),
                    keep,
                    stop_at_midcut=stop_at_midcut,
                    consequence_physical=hold_physical,
                    enable_final_action_decoder=False,
                )
                out["rollout_effect_pred_hold_action"] = hold_policy["rollout_effect_pred"]
                out["rollout_delta_pred_hold_action"] = hold_policy["rollout_delta_pred"]
                out["rollout_base_effect_pred_hold_action"] = hold_policy["rollout_base_effect_pred"]
                if "midcut_rollout_effect_pred" in hold_policy:
                    out["midcut_rollout_effect_pred_hold_action"] = hold_policy["midcut_rollout_effect_pred"]
                    out["midcut_rollout_delta_pred_hold_action"] = hold_policy["midcut_rollout_delta_pred"]
                    out["midcut_rollout_base_effect_pred_hold_action"] = hold_policy["midcut_rollout_base_effect_pred"]
                _merge_layer_contract_counterfactuals(hold_policy, "hold_action")
                if target_physical.shape[0] > 1:
                    # V40: within-batch hard action negative.  V39 used a
                    # reverse-batch permutation, which can become an adjacent
                    # near-duplicate on ordered validation batches.  The hard
                    # negative must be encoded in the current sample's action
                    # state; directly permuting target_physical would mix
                    # state-relative coordinate frames and inflate shuffle
                    # diagnostics.
                    batch = int(target_action.shape[0])
                    cand_action = (
                        target_action.detach()[None]
                        .expand(batch, -1, -1, -1)
                        .reshape(batch * batch, int(target_action.shape[1]), int(target_action.shape[2]))
                    )
                    cand_state = (
                        codec_state.detach()[:, None]
                        .expand(-1, batch, -1)
                        .reshape(batch * batch, int(codec_state.shape[-1]))
                    )
                    cand_physical = self.codec.encode(cand_action, cand_state).reshape(
                        batch,
                        batch,
                        int(target_physical.shape[1]),
                        int(target_physical.shape[2]),
                    )
                    dist = (
                        cand_physical.detach().float()
                        - target_physical.detach().float()[:, None]
                    ).flatten(2).norm(dim=-1)
                    eye = torch.eye(batch, device=dist.device, dtype=torch.bool)
                    dist = dist.masked_fill(eye, -1.0)
                    perm = dist.argmax(dim=1)
                    shuffle_physical = cand_physical[torch.arange(batch, device=target_physical.device), perm]
                else:
                    shuffle_physical = target_physical
                shuffle_noisy = (1 - t[:, None, None]) * shuffle_physical + t[:, None, None] * noise
                shuffle_policy = self._policy_forward(
                    shuffle_noisy.detach(),
                    t.detach(),
                    visual,
                    state_history,
                    state,
                    executed_history,
                    proposal["tokens"].detach(),
                    keep,
                    stop_at_midcut=stop_at_midcut,
                    consequence_physical=shuffle_physical,
                    enable_final_action_decoder=False,
                )
                out["rollout_effect_pred_shuffle_action"] = shuffle_policy["rollout_effect_pred"]
                out["rollout_delta_pred_shuffle_action"] = shuffle_policy["rollout_delta_pred"]
                out["rollout_base_effect_pred_shuffle_action"] = shuffle_policy["rollout_base_effect_pred"]
                if "midcut_rollout_effect_pred" in shuffle_policy:
                    out["midcut_rollout_effect_pred_shuffle_action"] = shuffle_policy["midcut_rollout_effect_pred"]
                    out["midcut_rollout_delta_pred_shuffle_action"] = shuffle_policy["midcut_rollout_delta_pred"]
                    out["midcut_rollout_base_effect_pred_shuffle_action"] = shuffle_policy["midcut_rollout_base_effect_pred"]
                _merge_layer_contract_counterfactuals(shuffle_policy, "shuffle_action")
        return out

    @torch.no_grad()
    def sample(
        self,
        visual: Tensor,
        state_history: Tensor,
        executed_history: Tensor,
        state: Tensor,
        *,
        action_state: Tensor | None = None,
        steps: int | None = None,
        noise: Tensor | None = None,
        use_proposal: bool = True,
        return_event_logits: bool = False,
        stop_at_midcut: bool = False,
    ) -> Tensor | dict[str, Tensor]:
        """Sample an action chunk without teacher-forcing target actions.

        ``stop_at_midcut`` is used for contract-stage validation.  It evaluates
        the deployable mid-cut simple heads by iterative denoising from noise;
        unlike ``flow_training_forward`` it never receives ``target_action`` and
        therefore cannot leak validation labels into action metrics.
        """
        proposal = self.proposal(executed_history)
        steps = int(steps or self.policy_config.inference_steps)
        if steps <= 0:
            raise ValueError("steps must be positive")
        if self.codec.uses_arm_manifold and action_state is None:
            raise ValueError("manifold_native sampling requires action_state in action-normalizer coordinates")
        codec_state = (state if action_state is None else action_state).to(
            device=visual.device, dtype=visual.dtype,
        )
        if noise is None:
            x = self.codec.sample_noise(
                visual.shape[0],
                device=visual.device,
                dtype=visual.dtype,
                action_state=codec_state,
            )
        else:
            x = noise.clone()
            if x.shape[-1] == self.policy_config.action_dim:
                x = self.codec.encode(x.to(device=visual.device, dtype=visual.dtype), codec_state)
            elif x.shape[-1] != self.policy_config.physical_action_dim:
                raise ValueError("noise must have last dim action_dim or physical_action_dim")
            else:
                x = x.to(device=visual.device, dtype=visual.dtype)
            x = self.codec.project_physical(x, codec_state)
        keep = torch.full((visual.shape[0],), 1.0 if use_proposal else 0.0, device=visual.device, dtype=visual.dtype)
        use_self_condition = (
            int(getattr(self.policy_config, "action_consequence_self_condition", 0))
            and int(getattr(self.policy_config, "layer_recurrent_consequence", 0))
            and int(getattr(self.policy_config, "layer_contract_adapters", 0))
        )
        sample_diagnostic_sums: dict[str, Tensor] = {}
        sample_diagnostic_count = 0
        for index in range(steps, 0, -1):
            t = torch.full((visual.shape[0],), float(index) / float(steps), device=visual.device, dtype=visual.dtype)
            consequence_input = x
            if use_self_condition:
                preview = self._policy_forward(
                    x,
                    t,
                    visual,
                    state_history,
                    state,
                    executed_history,
                    proposal["tokens"],
                    keep,
                    stop_at_midcut=stop_at_midcut,
                    consequence_physical=x,
                    enable_layer_contracts=False,
                )
                consequence_input = (x - t[:, None, None] * preview["pred_physical_velocity"]).detach()
                del preview
            out = self._policy_forward(
                x, t, visual, state_history, state, executed_history, proposal["tokens"], keep,
                stop_at_midcut=stop_at_midcut,
                consequence_physical=consequence_input,
                enable_layer_contracts=False,
            )
            for key, value in out.items():
                keep_diagnostic = (
                    key.startswith("latent_cvae_workspace_")
                    or key.startswith("latent_cvae_mmdit_")
                    or key.startswith("latent_cvae_hierarchical_")
                    or key in (
                        "latent_cvae_primary_condition_norm",
                        "latent_cvae_primary_z_effect_norm",
                    )
                )
                if keep_diagnostic and torch.is_tensor(value) and value.numel() == 1:
                    scalar = value.detach().float()
                    sample_diagnostic_sums[key] = sample_diagnostic_sums.get(key, torch.zeros_like(scalar)) + scalar
            sample_diagnostic_count += 1
            raw_next = x - out["pred_physical_velocity"] / float(steps)
            projected_next = self.codec.project_physical(raw_next, codec_state)
            # Pre-projection null drift: raw per-step magnitude plus the
            # step-size-normalized rate (x steps == / h) so runs with different
            # inference step counts stay comparable.
            null_drift = (raw_next - projected_next).detach().float()
            arm_span = 2 * int(self.codec.arm_dim)
            arm_null_norm = null_drift[..., :arm_span].norm(dim=-1).mean()
            grip_null_norm = null_drift[..., arm_span:].norm(dim=-1).mean()
            for null_key, null_value in (
                ("arm_null_preproject", arm_null_norm),
                ("arm_null_preproject_rate", arm_null_norm * float(steps)),
                ("grip_null_preproject", grip_null_norm),
                ("grip_null_preproject_rate", grip_null_norm * float(steps)),
            ):
                sample_diagnostic_sums[null_key] = sample_diagnostic_sums.get(null_key, torch.zeros_like(null_value)) + null_value
            x = projected_next
        action = self.codec.decode(x, codec_state)
        if return_event_logits:
            zero_t = torch.zeros((visual.shape[0],), device=visual.device, dtype=visual.dtype)
            event = self._policy_forward(
                x, zero_t, visual, state_history, state, executed_history, proposal["tokens"], keep,
                stop_at_midcut=stop_at_midcut,
                enable_layer_contracts=False,
            )
            result = {
                "action": action,
                "physical_action": x,
                "event_logits": event["event_logits"],
                "motion_logits": event["motion_logits"],
            }
            for key, value in sample_diagnostic_sums.items():
                result[f"sample_{key}"] = value / float(max(sample_diagnostic_count, 1))
            return result
        return action

    def parameter_report(self) -> dict[str, int]:
        report = {
            "history_proposal": sum(p.numel() for p in self.proposal.parameters()),
            "physical_action_codec": sum(p.numel() for p in self.codec.parameters()),
            "midcut_contract_heads": sum(p.numel() for p in self.planner.midcut_heads.parameters()),
            "layer_contract_adapters": sum(p.numel() for p in self.planner.layer_contract_heads.parameters()),
            "layer_shared_fm_probe": (0 if self.planner.layer_fm_probe is None else sum(p.numel() for p in self.planner.layer_fm_probe.parameters())),
            "layer_recurrent_consequence": (0 if self.planner.layer_consequence_cell is None else sum(p.numel() for p in self.planner.layer_consequence_cell.parameters())),
            "layer_causal_effect_branch": (0 if self.planner.layer_consequence_cell is None else sum(p.numel() for p in self.planner.layer_consequence_cell.parameters())),
            "layer_role_scheduler": sum(p.numel() for p in self.planner.layer_role_scheduler.parameters()),
            "residual_action_flow_denoiser": (
                0 if self.planner.residual_action_flow_denoiser is None
                else sum(p.numel() for p in self.planner.residual_action_flow_denoiser.parameters())
            ),
            "latent_main_action_decoder": (
                0 if getattr(self.planner, "latent_main_action_decoder", None) is None
                else sum(p.numel() for p in self.planner.latent_main_action_decoder.parameters())
            ),
            "latent_cvae_action_decoder": (
                0 if getattr(self.planner, "latent_cvae_action_decoder", None) is None
                else sum(p.numel() for p in self.planner.latent_cvae_action_decoder.parameters())
            ),
            "staged_midcut_dit": sum(p.numel() for p in self.planner.parameters()),
        }
        report["total"] = sum(p.numel() for p in self.parameters())
        report["trainable"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return report


__all__ = [
    "V39PolicyConfig",
    "MidcutContractHeads",
    "LayerContractAdapterHeads",
    "SharedLayerFlowActionProbe",
    "V37StyleResidualActionBlock",
    "V37StyleResidualActionFlowDenoiser",
    "LayerRoleScheduler",
    "UnifiedInterventionBlock",
    "RecurrentMilestoneConsequenceCell",
    "_align_milestone_tokens_to_horizon",
    "_rollout_tokens_to_action_horizon",
    "TemporalMidcutWorldActionDiT",
    "V39PolicySystem",
]
