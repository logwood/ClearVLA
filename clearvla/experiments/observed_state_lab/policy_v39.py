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
from .policy_v36_2 import PhysicalActionCodec
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
    # V48.1: use deploy-available self-conditioning for action consequence.
    # The old target-action consequence was a leakage path; raw noisy action is
    # safe but semantically weak.  This option runs a no-grad prior pass and
    # feeds the detached clean action estimate into the consequence cell.
    action_consequence_self_condition: int = 0

    # Current policy-stage decoder.  Older residual-action-flow and V41
    # latent-main heads were removed from the active policy surface; keeping a
    # single final action path avoids silent bypasses and stale optimizer groups.
    final_action_decoder: str = "adaptive_recurrent_cvae_action"

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
    latent_cvae_layer_detach: int = 0
    latent_cvae_layer_grad_scale: float = 0.0
    latent_cvae_event_gripper_gate: int = 1
    latent_cvae_inference_sample: int = 0
    latent_cvae_output_init_std: float = 1e-3
    latent_cvae_mu_bound: float = 1.5
    latent_cvae_min_std: float = 0.5
    latent_cvae_causal_attention: int = 1
    # V50.1: denoise a continuous trajectory representation instead of
    # directly iterating independent horizon action tokens.  The decoder keeps
    # the public pflow/action interface unchanged, but the recurrent main state
    # is a small set of smooth trajectory control tokens expanded over the
    # action horizon through a fixed RBF basis.
    latent_cvae_trajectory_denoise: int = 1
    latent_cvae_trajectory_control_points: int = 8
    latent_cvae_trajectory_context: int = 1
    latent_cvae_trajectory_mid_supervision: int = 1
    latent_cvae_trajectory_update_scale: float = 1.0
    latent_cvae_trajectory_context_scale: float = 0.50
    # V55: output-only arm coefficient head.  This keeps the recurrent denoise
    # state in full action-token space and only writes the arm velocity through
    # an orthonormal DCT basis at the final head.  It is not a smoothing loss
    # and it does not project intermediate refine states.
    latent_cvae_arm_coeff_output: int = 0
    latent_cvae_arm_coeff_points: int = 8
    latent_cvae_arm_coeff_basis: str = "dct"
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
    adaptive_cvae_token_semantic_adapter: int = 1
    adaptive_cvae_output_adapter: int = 0
    adaptive_cvae_context_dropout: float = 0.05
    adaptive_cvae_route_entropy_floor_ratio: float = 0.35
    adaptive_cvae_function_adapters: int = 0
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
    # V49: bounded temporal-block x native-action denoising matrix.  This is
    # deliberately not an action generator: it only changes the Gaussian bridge
    # geometry used by flow training/sampling.
    block_action_denoise_matrix: int = 0
    block_action_denoise_blocks: str = "0:4,4:12,12:24"
    block_action_denoise_rank: int = 2
    block_action_denoise_interaction_scale: float = 0.15
    block_action_noise_scale_min: float = 0.75
    block_action_noise_scale_max: float = 1.25
    block_action_noise_scale_init: float = 1.00
    block_action_velocity_loss_min: float = 0.75
    block_action_velocity_loss_max: float = 1.25
    block_action_velocity_loss_init: float = 1.00
    block_action_x0_mix_min: float = 0.00
    block_action_x0_mix_max: float = 0.20
    block_action_x0_mix_init: float = 0.00

    # V53-A1: t-gate on the direct noisy-action injection into the CVAE
    # action decoder.  gate = min + (1 - min) * t^p, so at low flow-time
    # (x_t close to the clean target) the shortcut branch is suppressed and
    # the decoder must lean on the trunk-derived condition instead.
    latent_cvae_noisy_gate: int = 0
    latent_cvae_noisy_gate_min: float = 0.05
    latent_cvae_noisy_gate_power: float = 1.5
    # V53-A3: no-loss diagnostic; rerun the consequence cell with the layer's
    # rollout tokens zeroed and log the relative output shift.  A small shift
    # means the cell is parroting the injected action features.
    layer_zero_base_diagnostic: int = 0
    # V53-B1: depth-scan condition.  A gated recurrent scan over the ordered
    # per-layer summaries replaces the flat concat as the primary condition;
    # a small alpha-weighted lateral concat path is kept for warm-up.
    latent_cvae_layer_scan: int = 0
    latent_cvae_layer_scan_alpha: float = 0.2
    # V53-B2: soft monotonic alignment between refine steps and layer depth
    # (and capsules and layer depth): logits -= scale * (pos_kv - center)^2.
    adaptive_cvae_monotonic_layer_route: int = 0
    adaptive_cvae_layer_route_distance_scale: float = 3.0
    # V53-C1: action tokens cross-attend to the final-canvas trajectory +
    # rollout tokens (full resolution) instead of relying only on the pooled
    # trajectory lift.  Raises trunk bandwidth into the tail.
    latent_cvae_canvas_cross_attention: int = 0
    # V53-C2: chain the per-refine-step lateral writers (trajectory_context ->
    # semantic_bias -> function_delta -> refine input) so each writer
    # reads the accumulated output of its predecessors instead of all writers
    # reading the same input in parallel.
    adaptive_cvae_serial_writers: int = 0
    # V53.1: unify the trajectory control-point coefficient space.  When on,
    # the model-internal projection and both runtime coefficient supervisions
    # share one ridge pseudo-inverse analysis operator, so expand(project(x))
    # is the (regularized) orthogonal projection onto the basis span.
    latent_cvae_trajectory_pinv: int = 0
    latent_cvae_trajectory_ridge: float = 1e-2
    # V53.1: restrict trajectory mid supervision to the homogeneous refine
    # segment (+ final state).  Seed/block/canvas intermediate states are no
    # longer pushed to emit the final answer directly.
    latent_cvae_trajectory_mid_refine_only: int = 0
    # V53.2: hinge penalty on the gated x_t branch's share of the base token
    # norm.  Stops noisy_action_lift from amplifying itself to cancel the
    # t-gate (observed xratio creep).  0 disables.
    latent_cvae_noisy_ratio_max: float = 0.0
    # V53.3: hard per-token norm cap on trajectory_context (the only ungated
    # lateral context channel; observed running away to ~2).  Renormalization
    # cannot be amplified around, unlike scales or loss penalties.  0 disables.
    latent_cvae_trajectory_context_norm_max: float = 0.0
    # V53.5 (#1 fix): exempt the horizon positional embedding from the coarse
    # base and the control-point projection.  The smooth manifold constrains
    # trajectory *content* only; the time coordinate system stays intact.
    latent_cvae_trajectory_pos_exempt: int = 0

    def validate(self) -> None:
        super().validate()
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
        if int(self.latent_cvae_noisy_gate) not in (0, 1):
            raise ValueError("latent_cvae_noisy_gate must be 0 or 1")
        if not (0.0 <= float(self.latent_cvae_noisy_gate_min) <= 1.0):
            raise ValueError("latent_cvae_noisy_gate_min must be in [0, 1]")
        if float(self.latent_cvae_noisy_gate_power) <= 0:
            raise ValueError("latent_cvae_noisy_gate_power must be positive")
        if int(self.layer_zero_base_diagnostic) not in (0, 1):
            raise ValueError("layer_zero_base_diagnostic must be 0 or 1")
        if int(self.latent_cvae_layer_scan) not in (0, 1):
            raise ValueError("latent_cvae_layer_scan must be 0 or 1")
        if float(self.latent_cvae_layer_scan_alpha) < 0:
            raise ValueError("latent_cvae_layer_scan_alpha must be non-negative")
        if int(self.adaptive_cvae_monotonic_layer_route) not in (0, 1):
            raise ValueError("adaptive_cvae_monotonic_layer_route must be 0 or 1")
        if float(self.adaptive_cvae_layer_route_distance_scale) < 0:
            raise ValueError("adaptive_cvae_layer_route_distance_scale must be non-negative")
        if int(self.latent_cvae_canvas_cross_attention) not in (0, 1):
            raise ValueError("latent_cvae_canvas_cross_attention must be 0 or 1")
        if int(self.adaptive_cvae_serial_writers) not in (0, 1):
            raise ValueError("adaptive_cvae_serial_writers must be 0 or 1")
        if int(self.latent_cvae_trajectory_pinv) not in (0, 1):
            raise ValueError("latent_cvae_trajectory_pinv must be 0 or 1")
        if float(self.latent_cvae_trajectory_ridge) < 0:
            raise ValueError("latent_cvae_trajectory_ridge must be non-negative")
        if int(self.latent_cvae_trajectory_mid_refine_only) not in (0, 1):
            raise ValueError("latent_cvae_trajectory_mid_refine_only must be 0 or 1")
        if float(self.latent_cvae_noisy_ratio_max) < 0:
            raise ValueError("latent_cvae_noisy_ratio_max must be non-negative")
        if float(self.latent_cvae_trajectory_context_norm_max) < 0:
            raise ValueError("latent_cvae_trajectory_context_norm_max must be non-negative")
        if int(self.latent_cvae_trajectory_pos_exempt) not in (0, 1):
            raise ValueError("latent_cvae_trajectory_pos_exempt must be 0 or 1")
        if int(self.latent_cvae_arm_coeff_output) not in (0, 1):
            raise ValueError("latent_cvae_arm_coeff_output must be 0 or 1")
        if int(self.latent_cvae_arm_coeff_points) < 1:
            raise ValueError("latent_cvae_arm_coeff_points must be >= 1")
        if str(self.latent_cvae_arm_coeff_basis).lower() != "dct":
            raise ValueError("latent_cvae_arm_coeff_basis currently supports only dct")
        if str(self.final_action_decoder) != "adaptive_recurrent_cvae_action":
            raise ValueError("final_action_decoder must be adaptive_recurrent_cvae_action")
        if int(self.latent_cvae_z_dim) < 1:
            raise ValueError("latent_cvae_z_dim must be >= 1")
        if int(self.latent_cvae_decoder_depth) < 1:
            raise ValueError("latent_cvae_decoder_depth must be >= 1")
        if float(self.latent_cvae_ffn_expansion) < 1.0:
            raise ValueError("latent_cvae_ffn_expansion must be >= 1")
        for name in (
            "latent_cvae_layer_memory", "latent_cvae_transition_memory",
            "latent_cvae_transition_detach", "latent_cvae_context_memory", "latent_cvae_visual_memory",
            "latent_cvae_layer_detach", "latent_cvae_event_gripper_gate",
            "latent_cvae_inference_sample",
            "latent_cvae_trajectory_denoise",
            "latent_cvae_trajectory_context",
            "latent_cvae_trajectory_mid_supervision",
        ):
            if int(getattr(self, name)) not in (0, 1):
                raise ValueError(f"{name} must be 0 or 1")
        if int(self.adaptive_cvae_refine_steps) < 0:
            raise ValueError("adaptive_cvae_refine_steps must be >= 0")
        if int(self.adaptive_cvae_progress_steps) < 1:
            raise ValueError("adaptive_cvae_progress_steps must be >= 1")
        for name in (
            "adaptive_cvae_progress_memory",
            "adaptive_cvae_prefix_memory",
            "adaptive_cvae_layer_routing",
            "adaptive_cvae_prefix_detach",
            "adaptive_cvae_progress_z_injection",
            "adaptive_cvae_route_query_bias",
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
        if int(self.latent_cvae_trajectory_control_points) < 2:
            raise ValueError("latent_cvae_trajectory_control_points must be >= 2")
        if int(self.latent_cvae_trajectory_mid_supervision) not in (0, 1):
            raise ValueError("latent_cvae_trajectory_mid_supervision must be 0 or 1")
        if float(self.latent_cvae_trajectory_update_scale) < 0:
            raise ValueError("latent_cvae_trajectory_update_scale must be non-negative")
        if float(self.latent_cvae_trajectory_context_scale) < 0:
            raise ValueError("latent_cvae_trajectory_context_scale must be non-negative")
        if not (0.0 <= float(self.latent_cvae_layer_grad_scale) <= 1.0):
            raise ValueError("latent_cvae_layer_grad_scale must be in [0, 1]")
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
        if int(self.block_action_denoise_matrix) not in (0, 1):
            raise ValueError("block_action_denoise_matrix must be 0 or 1")
        if int(self.block_action_denoise_rank) < 0:
            raise ValueError("block_action_denoise_rank must be >= 0")
        if float(self.block_action_denoise_interaction_scale) < 0:
            raise ValueError("block_action_denoise_interaction_scale must be non-negative")
        for lo_name, hi_name, init_name in (
            ("block_action_noise_scale_min", "block_action_noise_scale_max", "block_action_noise_scale_init"),
            ("block_action_velocity_loss_min", "block_action_velocity_loss_max", "block_action_velocity_loss_init"),
            ("block_action_x0_mix_min", "block_action_x0_mix_max", "block_action_x0_mix_init"),
        ):
            lo = float(getattr(self, lo_name))
            hi = float(getattr(self, hi_name))
            init = float(getattr(self, init_name))
            if lo < 0 or hi < lo:
                raise ValueError(f"{lo_name}/{hi_name} must define a non-negative bounded interval")
            if not (lo <= init <= hi):
                raise ValueError(f"{init_name} must be inside [{lo_name}, {hi_name}]")
        if int(self.block_action_denoise_matrix):
            _parse_block_action_ranges(str(self.block_action_denoise_blocks), int(self.action_horizon))
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


def _parse_block_action_ranges(spec: str, horizon: int) -> tuple[tuple[int, int], ...]:
    """Parse contiguous denoising blocks covering the whole action horizon."""

    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError("action_horizon must be positive")
    rows: list[tuple[int, int]] = []
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"block_action_denoise_blocks entry must be start:end, got {item!r}")
        left, right = item.split(":", 1)
        start, end = int(left), int(right)
        if start < 0 or end <= start or end > horizon:
            raise ValueError(f"invalid block range {item!r} for horizon={horizon}")
        rows.append((start, end))
    if not rows:
        raise ValueError("block_action_denoise_blocks cannot be empty")
    rows = sorted(rows)
    if rows[0][0] != 0 or rows[-1][1] != horizon:
        raise ValueError(f"block ranges must cover [0,{horizon}) exactly, got {rows}")
    for prev, cur in zip(rows, rows[1:]):
        if prev[1] != cur[0]:
            raise ValueError(f"block ranges must be contiguous without gaps/overlaps, got {rows}")
    return tuple(rows)


def _bounded_logit(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    frac = min(max((float(value) - lo) / (hi - lo), 1e-4), 1.0 - 1e-4)
    return math.log(frac / (1.0 - frac))


class _FactorizedBlockActionScalar(nn.Module):
    """Bounded low-rank block x native-action scalar."""

    def __init__(
        self,
        *,
        blocks: int,
        actions: int,
        rank: int,
        lo: float,
        hi: float,
        init: float,
        interaction_scale: float,
    ) -> None:
        super().__init__()
        self.blocks = int(blocks)
        self.actions = int(actions)
        self.rank = max(int(rank), 0)
        self.lo = float(lo)
        self.hi = float(hi)
        self.interaction_scale = float(interaction_scale)
        self.base = nn.Parameter(torch.tensor(_bounded_logit(float(init), self.lo, self.hi), dtype=torch.float32))
        self.block_bias = nn.Parameter(torch.zeros(self.blocks, 1, dtype=torch.float32))
        self.action_bias = nn.Parameter(torch.zeros(1, self.actions, dtype=torch.float32))
        if self.rank > 0:
            self.block_u = nn.Parameter(torch.randn(self.blocks, self.rank, dtype=torch.float32) * 1e-3)
            self.action_v = nn.Parameter(torch.randn(self.actions, self.rank, dtype=torch.float32) * 1e-3)
        else:
            self.register_parameter("block_u", None)
            self.register_parameter("action_v", None)

    def raw_matrix(self) -> Tensor:
        raw = self.base + self.block_bias + self.action_bias
        if self.rank > 0 and self.block_u is not None and self.action_v is not None:
            raw = raw + self.interaction_scale * torch.einsum("br,ar->ba", self.block_u, self.action_v)
        return raw

    def forward(self) -> Tensor:
        raw = self.raw_matrix()
        if self.hi <= self.lo:
            return torch.full_like(raw, self.lo)
        return self.lo + (self.hi - self.lo) * torch.sigmoid(raw)

    def interaction_norm(self) -> Tensor:
        if self.rank <= 0 or self.block_u is None or self.action_v is None:
            return torch.zeros((), device=self.base.device, dtype=self.base.dtype)
        return torch.einsum("br,ar->ba", self.block_u, self.action_v).square().mean()


class BlockActionDenoiseMatrix(nn.Module):
    """Safe block x native-action denoising-parameter controller."""

    def __init__(self, config: V39PolicyConfig) -> None:
        super().__init__()
        self.config = config
        self.ranges = _parse_block_action_ranges(str(config.block_action_denoise_blocks), int(config.action_horizon))
        blocks = len(self.ranges)
        actions = int(config.action_dim)
        rank = int(config.block_action_denoise_rank)
        scale = float(config.block_action_denoise_interaction_scale)
        self.noise_scale = _FactorizedBlockActionScalar(
            blocks=blocks,
            actions=actions,
            rank=rank,
            lo=float(config.block_action_noise_scale_min),
            hi=float(config.block_action_noise_scale_max),
            init=float(config.block_action_noise_scale_init),
            interaction_scale=scale,
        )
        self.velocity_loss = _FactorizedBlockActionScalar(
            blocks=blocks,
            actions=actions,
            rank=rank,
            lo=float(config.block_action_velocity_loss_min),
            hi=float(config.block_action_velocity_loss_max),
            init=float(config.block_action_velocity_loss_init),
            interaction_scale=scale,
        )
        self.x0_mix = _FactorizedBlockActionScalar(
            blocks=blocks,
            actions=actions,
            rank=rank,
            lo=float(config.block_action_x0_mix_min),
            hi=float(config.block_action_x0_mix_max),
            init=float(config.block_action_x0_mix_init),
            interaction_scale=scale,
        )
        # Only the Gaussian endpoint scale is learned from data.  The loss/x0
        # scalars are fixed diagnostic/curriculum knobs so they cannot learn to
        # down-weight hard cells and make the objective look better.
        for module in (self.velocity_loss, self.x0_mix):
            for param in module.parameters():
                param.requires_grad_(False)
        block_ids = torch.empty(int(config.action_horizon), dtype=torch.long)
        for idx, (start, end) in enumerate(self.ranges):
            block_ids[start:end] = idx
        block_lengths = torch.as_tensor([end - start for start, end in self.ranges], dtype=torch.float32)
        self.register_buffer("block_ids", block_ids, persistent=False)
        self.register_buffer("block_lengths", block_lengths, persistent=False)
        self.register_buffer("physical_to_action", self._physical_to_action_index(config), persistent=False)

    def _expand_native(self, native_matrix: Tensor, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        horizon = native_matrix[self.block_ids.to(device=native_matrix.device)]
        return horizon.to(device=device, dtype=dtype)[None]

    def _native_horizon_square_mean(self, native_matrix: Tensor) -> Tensor:
        weights = self.block_lengths.to(device=native_matrix.device, dtype=native_matrix.dtype)[:, None]
        denom = weights.sum().clamp_min(1.0) * max(int(native_matrix.shape[-1]), 1)
        return (native_matrix.square() * weights).sum() / denom

    def _normalize_noise_budget(self, native_matrix: Tensor) -> tuple[Tensor, Tensor]:
        raw_rms = self._native_horizon_square_mean(native_matrix).clamp_min(1e-12).sqrt()
        return native_matrix / raw_rms.clamp_min(1e-6), raw_rms

    @staticmethod
    def _physical_to_action_index(config: V39PolicyConfig) -> Tensor:
        action_dim = int(config.action_dim)
        physical_dim = int(config.physical_action_dim)
        arm_dim = int(getattr(config, "arm_dim", max(action_dim - 1, 0)))
        grip_index = int(getattr(config, "gripper_dim_index", -1))
        if grip_index < 0:
            grip_index = action_dim + grip_index
        grip_index = min(max(grip_index, 0), max(action_dim - 1, 0))
        mapping = torch.arange(physical_dim, dtype=torch.long).clamp(max=max(action_dim - 1, 0))
        if arm_dim > 0 and physical_dim == 2 * arm_dim + 2 and action_dim >= arm_dim + 1:
            mapping = torch.empty(physical_dim, dtype=torch.long)
            mapping[:arm_dim] = torch.arange(arm_dim, dtype=torch.long)
            mapping[arm_dim:2 * arm_dim] = torch.arange(arm_dim, dtype=torch.long)
            mapping[2 * arm_dim] = grip_index
            mapping[2 * arm_dim + 1] = grip_index
        return mapping

    def _expand(self, native_matrix: Tensor, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        physical = native_matrix[:, self.physical_to_action.to(device=native_matrix.device)]
        horizon = physical[self.block_ids.to(device=native_matrix.device)]
        return horizon.to(device=device, dtype=dtype)[None]

    def matrices(self, ref: Tensor) -> dict[str, Tensor]:
        device, dtype = ref.device, ref.dtype
        noise_raw = self.noise_scale()
        noise_native, noise_raw_rms = self._normalize_noise_budget(noise_raw)
        noise_rms = self._native_horizon_square_mean(noise_native).clamp_min(1e-12).sqrt()
        loss_native = self.velocity_loss()
        x0_native = self.x0_mix()
        smooth = torch.zeros((), device=noise_native.device, dtype=noise_native.dtype)
        if noise_native.shape[0] > 1:
            smooth = (
                (noise_native[1:] - noise_native[:-1]).square().mean()
                + (loss_native[1:] - loss_native[:-1]).square().mean()
                + (x0_native[1:] - x0_native[:-1]).square().mean()
            )
            noise_boundary_jump = (noise_native[1:] - noise_native[:-1]).abs().mean()
        else:
            noise_boundary_jump = torch.zeros((), device=noise_native.device, dtype=noise_native.dtype)
        deviation = (
            self._native_horizon_square_mean(noise_native - 1.0)
            + (loss_native - 1.0).square().mean()
            + x0_native.square().mean()
        )
        interaction = (
            self.noise_scale.interaction_norm()
            + self.velocity_loss.interaction_norm()
            + self.x0_mix.interaction_norm()
        )
        regularizer = smooth + 0.10 * deviation + 0.10 * interaction
        grip_idx = int(getattr(self.config, "gripper_dim_index", -1))
        if grip_idx < 0:
            grip_idx = int(self.config.action_dim) + grip_idx
        grip_idx = min(max(grip_idx, 0), int(self.config.action_dim) - 1)
        arm_mask = torch.ones(int(self.config.action_dim), device=noise_native.device, dtype=torch.bool)
        arm_mask[grip_idx] = False
        return {
            "native_noise_scale": self._expand_native(noise_native, device=device, dtype=dtype),
            "noise_scale": self._expand(noise_native, device=device, dtype=dtype),
            # Loss/x0 weights are diagnostics or fixed reweighting knobs; the
            # main loss must not teach them to down-weight hard cells.
            "velocity_loss_weight": self._expand(loss_native.detach(), device=device, dtype=dtype),
            "x0_mix": self._expand(x0_native.detach(), device=device, dtype=dtype),
            "regularizer": regularizer.to(device=device, dtype=dtype),
            "smoothness": smooth.detach().to(device=device, dtype=torch.float32),
            "deviation": deviation.detach().to(device=device, dtype=torch.float32),
            "interaction_norm": interaction.detach().to(device=device, dtype=torch.float32),
            "noise_arm_mean": noise_native[:, arm_mask].detach().mean().to(device=device, dtype=torch.float32),
            "noise_gripper_mean": noise_native[:, grip_idx].detach().mean().to(device=device, dtype=torch.float32),
            "noise_near_mean": noise_native.detach()[0].mean().to(device=device, dtype=torch.float32),
            "noise_tail_mean": noise_native.detach()[-1].mean().to(device=device, dtype=torch.float32),
            "noise_min": noise_native.detach().min().to(device=device, dtype=torch.float32),
            "noise_max": noise_native.detach().max().to(device=device, dtype=torch.float32),
            "noise_std": noise_native.detach().std(unbiased=False).to(device=device, dtype=torch.float32),
            "noise_raw_rms": noise_raw_rms.detach().to(device=device, dtype=torch.float32),
            "noise_rms": noise_rms.detach().to(device=device, dtype=torch.float32),
            "noise_boundary_jump": noise_boundary_jump.detach().to(device=device, dtype=torch.float32),
            "loss_arm_mean": loss_native.detach()[:, arm_mask].mean().to(device=device, dtype=torch.float32),
            "loss_gripper_mean": loss_native.detach()[:, grip_idx].mean().to(device=device, dtype=torch.float32),
            "x0_near_mean": x0_native.detach()[0].mean().to(device=device, dtype=torch.float32),
            "x0_tail_mean": x0_native.detach()[-1].mean().to(device=device, dtype=torch.float32),
        }


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
        self.action_summary_dim = ph * 5 + 4
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
        self.mod = nn.Linear(h, 6 * h)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    @staticmethod
    def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        return x * (1.0 + scale[:, None]) + shift[:, None]

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        sa_s, sa_c, sa_g, ff_s, ff_c, ff_g = self.mod(cond).chunk(6, dim=-1)
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
        # V53.4: return the full [B,T,1] keep map so the caller can test the
        # "gate as crude feedback controller" hypothesis (per-sample keep vs
        # sample difficulty / flow time), instead of only the scalar mean.
        return x, keep


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


def _continuous_trajectory_basis(horizon: int, controls: int) -> Tensor:
    if horizon < 1 or controls < 2:
        raise ValueError("continuous trajectory basis requires horizon >= 1 and controls >= 2")
    if controls > horizon:
        raise ValueError("continuous trajectory controls cannot exceed horizon")
    time = torch.linspace(0.0, 1.0, horizon, dtype=torch.float32)
    centers = torch.linspace(0.0, 1.0, controls, dtype=torch.float32)
    width = 1.35 / max(float(controls - 1), 1.0)
    basis = torch.exp(-0.5 * ((time[:, None] - centers[None]) / width).square())
    return basis / basis.sum(dim=-1, keepdim=True).clamp_min(1e-6)


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
        self.z_dim = int(getattr(config, "latent_cvae_z_dim", 64))
        self.depth = int(getattr(config, "latent_cvae_decoder_depth", 3))
        self.time = TimeEmbedding(h)
        self.horizon_query = nn.Parameter(torch.randn(1, int(config.action_horizon), h) * 0.02)
        self.noisy_action_lift = nn.Sequential(nn.LayerNorm(int(config.physical_action_dim)), nn.Linear(int(config.physical_action_dim), h))
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
        self.layer_embed = nn.Parameter(torch.randn(1, int(config.depth), h) * 0.02)
        self.transition_proj = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h))
        self.context_proj = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h))
        self.visual_proj = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h))
        self.traj_summary_proj = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h))
        cond_in = int(config.depth) * h + 4 * h
        self.condition_fusion = nn.Sequential(nn.LayerNorm(cond_in), nn.Linear(cond_in, h), nn.SiLU(), nn.Linear(h, h))
        # V53-B1: gated recurrent depth scan over ordered layer summaries.
        if int(getattr(config, "latent_cvae_layer_scan", 0)):
            self.layer_scan = nn.GRUCell(h, h)
            self.layer_scan_init = nn.Parameter(torch.zeros(1, h))
            self.layer_scan_fusion = nn.Sequential(nn.LayerNorm(5 * h), nn.Linear(5 * h, h), nn.SiLU(), nn.Linear(h, h))
        else:
            self.layer_scan = None
            self.layer_scan_init = None
            self.layer_scan_fusion = None
        # V53-C1: cross-attention from action tokens into full final-canvas
        # trajectory + rollout tokens (trunk-bandwidth path).
        if int(getattr(config, "latent_cvae_canvas_cross_attention", 0)):
            self.canvas_memory_norm = nn.LayerNorm(h)
            self.canvas_query_norm = nn.LayerNorm(h, elementwise_affine=False)
            self.canvas_cross = nn.MultiheadAttention(h, int(config.num_heads), batch_first=True, dropout=float(config.dropout))
            self.canvas_gate = nn.Parameter(torch.tensor(-2.0))
        else:
            self.canvas_memory_norm = None
            self.canvas_query_norm = None
            self.canvas_cross = None
            self.canvas_gate = None
        self.prior = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 2 * self.z_dim))
        self.posterior_action = nn.Sequential(
            nn.LayerNorm(int(config.physical_action_dim)),
            nn.Linear(int(config.physical_action_dim), h),
            nn.SiLU(),
            nn.Linear(h, h),
        )
        self.posterior = nn.Sequential(nn.LayerNorm(2 * h), nn.Linear(2 * h, h), nn.SiLU(), nn.Linear(h, 2 * self.z_dim))
        self.z_to_token = nn.Sequential(nn.LayerNorm(self.z_dim), nn.Linear(self.z_dim, h))
        self.blocks = nn.ModuleList([LatentCVAEActionBlock(config) for _ in range(self.depth)])
        self.event_gate = nn.Sequential(nn.LayerNorm(2 * h), nn.Linear(2 * h, h), nn.Sigmoid())
        self.event_transition = nn.Sequential(nn.LayerNorm(2 * h), nn.Linear(2 * h, h), nn.SiLU(), nn.Linear(h, h))
        self.velocity_head = TransitionAwarePhysicalVelocityHead(config)
        self.event_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 3))
        self.motion_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 1))
        self._initialize_outputs()

    def _initialize_outputs(self) -> None:
        std = float(getattr(self.config, "latent_cvae_output_init_std", 1e-3))
        for module in (
            self.velocity_head.arm_abs,
            self.velocity_head.arm_delta,
            self.velocity_head.grip_value,
            self.velocity_head.grip_delta,
        ):
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

    def _layer_entry_summary(self, entry: dict[str, Tensor], *, detach: bool) -> Tensor | None:
        groups: list[Tensor] = []
        grad_scale = float(getattr(self.config, "latent_cvae_layer_grad_scale", 0.0))
        for key_index, key in enumerate(self._LAYER_KEYS):
            value = entry.get(key)
            if not isinstance(value, Tensor) or value.ndim != 3 or int(value.shape[-1]) != self.hidden_size:
                continue
            source = _scaled_contract_view(value, grad_scale) if detach else value
            pooled = source.mean(dim=1)
            typed = self.layer_key_proj[key_index](pooled)
            typed = typed + self.layer_key_embed[:, key_index].to(device=typed.device, dtype=typed.dtype)
            groups.append(typed)
        if not groups:
            return None
        stack = torch.stack(groups, dim=1)
        weights = torch.softmax(self.layer_key_gate(stack).float(), dim=1).to(dtype=stack.dtype)
        return (stack * weights).sum(dim=1)

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
        """V53-A1: t-dependent gate for the direct x_t branch.

        gate = min + (1 - min) * t^p.  At t -> 0 (x_t ~ clean target) the
        shortcut branch is suppressed to ``min``; at t -> 1 (pure noise) it
        passes unchanged.  Returns None when disabled.
        """
        cfg = self.config
        if not int(getattr(cfg, "latent_cvae_noisy_gate", 0)):
            return None
        min_gate = float(getattr(cfg, "latent_cvae_noisy_gate_min", 0.05))
        power = float(getattr(cfg, "latent_cvae_noisy_gate_power", 1.5))
        t = time.float().clamp(0.0, 1.0)
        return (min_gate + (1.0 - min_gate) * t.pow(power))[:, None, None]

    def _gated_noisy_branch(self, noisy_physical: Tensor, time: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        branch = self.noisy_action_lift(noisy_physical)
        gate = self._noisy_time_gate(time)
        if gate is None:
            gate_mean = torch.ones((), device=branch.device, dtype=torch.float32)
        else:
            branch = branch * gate.to(dtype=branch.dtype)
            gate_mean = gate.detach().float().mean()
        return branch, gate_mean, branch.detach().float().norm(dim=-1).mean()

    def _canvas_cross_update(self, action: Tensor, canvas_memory: Tensor | None) -> Tensor | None:
        """V53-C1: gated cross-attention update from full canvas tokens."""
        if canvas_memory is None or not int(getattr(self.config, "latent_cvae_canvas_cross_attention", 0)):
            return None
        mem = self.canvas_memory_norm(canvas_memory.to(device=action.device, dtype=action.dtype))
        query = self.canvas_query_norm(action)
        update, _ = self.canvas_cross(query, mem, mem, need_weights=False)
        return torch.sigmoid(self.canvas_gate).to(dtype=action.dtype) * update

    def _decode_with_z(
        self,
        *,
        noisy_physical: Tensor,
        time: Tensor,
        trajectory_tokens: Tensor,
        cond: Tensor,
        z: Tensor,
        layer_stack: Tensor | None = None,
        canvas_memory: Tensor | None = None,
    ) -> dict[str, Tensor]:
        del layer_stack
        batch = int(noisy_physical.shape[0])
        dtype = noisy_physical.dtype
        device = noisy_physical.device
        time_emb = self.time(time.to(dtype=dtype))
        cond_time = cond + self.time_lift(time_emb)
        noisy_branch, _, _ = self._gated_noisy_branch(noisy_physical, time)
        action = (
            self.horizon_query.to(device=device, dtype=dtype).expand(batch, -1, -1)
            + noisy_branch
            + self.trajectory_lift(trajectory_tokens)
            + self.z_to_token(z.to(dtype=dtype))[:, None]
            + cond_time[:, None]
        )
        for block in self.blocks:
            action = block(action, cond_time)
        canvas_update = self._canvas_cross_update(action, canvas_memory)
        if canvas_update is not None:
            action = action + canvas_update
        return self._emit_action(action, cond)

    def _condition(
        self,
        *,
        trajectory_tokens: Tensor,
        context_memory: Tensor | list[Tensor] | tuple[Tensor, ...] | None,
        transition_memory: Tensor | list[Tensor] | tuple[Tensor, ...] | None,
        visual_memory: Tensor | list[Tensor] | tuple[Tensor, ...] | None,
        layer_contracts: list[dict[str, Tensor]],
    ) -> tuple[Tensor, Tensor, Tensor]:
        cfg = self.config
        dtype = trajectory_tokens.dtype
        device = trajectory_tokens.device
        batch = int(trajectory_tokens.shape[0])
        detach_layers = bool(int(getattr(cfg, "latent_cvae_layer_detach", 0)))
        use_layer_memory = bool(int(getattr(cfg, "latent_cvae_layer_memory", 1)))
        summaries: list[Tensor] = []
        if use_layer_memory:
            for entry in layer_contracts:
                summary = self._layer_entry_summary(entry, detach=detach_layers)
                if summary is not None:
                    summaries.append(summary.to(device=device, dtype=dtype))
        if use_layer_memory and len(summaries) < int(cfg.depth):
            raise RuntimeError(f"{str(getattr(cfg, 'final_action_decoder', 'adaptive_recurrent_cvae_action'))} expected summaries for {int(cfg.depth)} layers, got {len(summaries)}")
        if use_layer_memory and summaries:
            projected = []
            for i in range(int(cfg.depth)):
                src = summaries[min(i, len(summaries) - 1)]
                projected.append(self.layer_proj[i](src))
            layer_stack = torch.stack(projected, dim=1) + self.layer_embed.to(device=device, dtype=dtype)
        else:
            layer_stack = torch.zeros(batch, int(cfg.depth), self.hidden_size, device=device, dtype=dtype)
        layer_flat = layer_stack.reshape(batch, int(cfg.depth) * self.hidden_size)
        traj = self.traj_summary_proj(trajectory_tokens.mean(dim=1))
        transition_source = self._maybe_detach_memory(
            transition_memory,
            detach=bool(int(getattr(cfg, "latent_cvae_transition_detach", 1))),
        )
        trans = self._memory_summary(transition_source, traj, self.transition_proj) if int(getattr(cfg, "latent_cvae_transition_memory", 1)) else torch.zeros_like(traj)
        ctx = self._memory_summary(context_memory, traj, self.context_proj) if int(getattr(cfg, "latent_cvae_context_memory", 0)) else torch.zeros_like(traj)
        vis = self._memory_summary(visual_memory, traj, self.visual_proj) if int(getattr(cfg, "latent_cvae_visual_memory", 0)) else torch.zeros_like(traj)
        lateral_cond = self.condition_fusion(torch.cat([layer_flat, trans, ctx, vis, traj], dim=-1))
        zero_stat = torch.zeros((), device=device, dtype=torch.float32)
        cond_stats = {
            "cvae_condition_scan_norm": zero_stat,
            "cvae_condition_lateral_norm": lateral_cond.detach().float().norm(dim=-1).mean(),
        }
        if int(getattr(cfg, "latent_cvae_layer_scan", 0)) and use_layer_memory:
            # V53-B1: depth scan.  Information from layer k reaches the
            # condition only through the recurrent state that already absorbed
            # layers < k; the flat concat stays as a small alpha side-path.
            state = self.layer_scan_init.to(device=device, dtype=layer_stack.dtype).expand(batch, -1)
            for i in range(int(layer_stack.shape[1])):
                state = self.layer_scan(layer_stack[:, i], state)
            scan_cond = self.layer_scan_fusion(torch.cat([state.to(dtype=dtype), trans, ctx, vis, traj], dim=-1))
            alpha = float(getattr(cfg, "latent_cvae_layer_scan_alpha", 0.2))
            cond = scan_cond + alpha * lateral_cond
            cond_stats["cvae_condition_scan_norm"] = scan_cond.detach().float().norm(dim=-1).mean()
        else:
            cond = lateral_cond
        layer_count = torch.tensor(float(len(summaries)), device=device, dtype=dtype)
        return cond, layer_count, layer_stack, cond_stats

    def forward(
        self,
        *,
        noisy_physical: Tensor,
        time: Tensor,
        trajectory_tokens: Tensor,
        context_memory: Tensor | list[Tensor] | tuple[Tensor, ...] | None,
        transition_memory: Tensor | list[Tensor] | tuple[Tensor, ...] | None,
        visual_memory: Tensor | list[Tensor] | tuple[Tensor, ...] | None,
        layer_contracts: list[dict[str, Tensor]],
        target_physical: Tensor | None = None,
        canvas_memory: Tensor | None = None,
    ) -> dict[str, Tensor]:
        cfg = self.config
        dtype = noisy_physical.dtype
        device = noisy_physical.device
        cond, layer_count, layer_stack, cond_stats = self._condition(
            trajectory_tokens=trajectory_tokens,
            context_memory=context_memory,
            transition_memory=transition_memory,
            visual_memory=visual_memory,
            layer_contracts=layer_contracts,
        )
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
            cond=cond,
            z=prior_z,
            layer_stack=layer_stack,
            canvas_memory=canvas_memory,
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
                cond=cond,
                z=post_z,
                layer_stack=layer_stack,
                canvas_memory=canvas_memory,
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
            "cvae_posterior_used": torch.tensor(float(posterior_used), device=device, dtype=dtype),
            "gripper_gate_mean": prior_out["gripper_gate_mean"],
            "layer_memory_count": layer_count,
            "cvae_prior_pred_norm": prior_out["pred_velocity"].detach().float().norm(dim=-1).mean(),
            **cond_stats,
        }
        for key in (
            "adaptive_continue_std",
            "adaptive_continue_first",
            "adaptive_continue_last",
            "adaptive_continue_per_sample",
            "adaptive_continue_time_corr",
            "adaptive_noisy_gate_mean",
            "adaptive_noisy_branch_norm",
            "adaptive_noisy_branch_ratio",
            "adaptive_canvas_cross_norm",
            "adaptive_canvas_gate",
            "adaptive_refine_update_mean",
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
            "adaptive_semantic_bias_norm",
            "adaptive_output_adapter_norm",
            "adaptive_function_delta_norm",
            "adaptive_base_highfreq_norm",
            "adaptive_trajectory_control_norm",
            "adaptive_trajectory_token_norm",
            "adaptive_trajectory_update_norm",
            "adaptive_trajectory_context_norm",
            "adaptive_trajectory_projection_error",
            "adaptive_trajectory_control_smoothness",
            "adaptive_trajectory_update_smoothness",
            "adaptive_trajectory_update_energy",
            "adaptive_trajectory_projection_regularizer",
            "adaptive_trajectory_pred_velocity",
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
            "adaptive_regularizer",
            "adaptive_route_entropy_regularizer",
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
    """V43 adaptive recurrent CVAE action head.

    This keeps V42's compact CVAE prior/posterior contract, then refines action
    tokens with a shared causal block.  The refinement path is deliberately
    small: action tokens read a compact bank of latent progress slots plus a
    soft route over per-layer summaries.  Progress slots are internal ordered
    latent states, not physical horizon steps.
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
        requested_controls = int(getattr(config, "latent_cvae_trajectory_control_points", min(12, action_horizon)))
        self.trajectory_control_points = max(2, min(action_horizon, requested_controls))
        self.register_buffer(
            "trajectory_basis",
            _continuous_trajectory_basis(action_horizon, self.trajectory_control_points),
            persistent=False,
        )
        # V53.1: one shared analysis operator [C, T].  Ridge pseudo-inverse when
        # latent_cvae_trajectory_pinv=1 (expand∘project becomes the regularized
        # orthogonal projection onto span(B)); otherwise the legacy
        # column-normalized transpose is kept for reproducibility.
        basis_f = self.trajectory_basis.float()
        if int(getattr(config, "latent_cvae_trajectory_pinv", 0)):
            ridge = max(float(getattr(config, "latent_cvae_trajectory_ridge", 1e-2)), 0.0)
            gram = basis_f.transpose(0, 1) @ basis_f
            eye = torch.eye(int(gram.shape[0]), dtype=torch.float32)
            analysis = torch.linalg.solve(gram + ridge * eye, basis_f.transpose(0, 1))
        else:
            analysis = (basis_f / basis_f.sum(dim=0, keepdim=True).clamp_min(1e-6)).transpose(0, 1)
        self.register_buffer("trajectory_analysis", analysis, persistent=False)
        self.progress_query = nn.Parameter(torch.randn(1, self.progress_steps, h) * 0.02)
        self.context_capsule_query = nn.Parameter(torch.randn(1, self.context_capsule_count, h) * 0.02)
        self.progress_route_query_bias = nn.Parameter(torch.zeros(1, action_horizon, h))
        self.layer_route_query_bias = nn.Parameter(torch.zeros(1, action_horizon, h))
        self.context_route_query_bias = nn.Parameter(torch.zeros(1, action_horizon, h))
        self.trajectory_control_adapter = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h))
        self.trajectory_update_adapter = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h))
        self.trajectory_context_adapter = nn.Sequential(nn.LayerNorm(5 * h), nn.Linear(5 * h, h), nn.SiLU(), nn.Linear(h, h))
        self.trajectory_context_norm = nn.LayerNorm(h, elementwise_affine=False)
        self.progress_role_lift = nn.Sequential(nn.LayerNorm(self.progress_role_dim), nn.Linear(self.progress_role_dim, h))
        self.context_capsule_role_lift = nn.Sequential(nn.LayerNorm(self.progress_role_dim), nn.Linear(self.progress_role_dim, h))
        self.progress_z_lift = nn.Sequential(nn.LayerNorm(self.z_dim), nn.Linear(self.z_dim, h))
        self.progress_block = LatentCVAEActionBlock(config)
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
        self.prefix_lift = (
            nn.Sequential(nn.LayerNorm(2 * ph), nn.Linear(2 * ph, h), nn.SiLU(), nn.Linear(h, h))
            if int(getattr(config, "adaptive_cvae_prefix_memory", 0))
            else None
        )
        self.route_query = nn.Linear(h, h, bias=False)
        self.route_key = nn.Linear(h, h, bias=False)
        self.route_value = nn.Linear(h, h, bias=False)
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
        self.token_semantic_adapter = nn.Sequential(nn.LayerNorm(4 * h), nn.Linear(4 * h, h), nn.SiLU(), nn.Linear(h, h))
        self.refine_function_bank = AdaptiveCVAEFunctionBank(config)
        if int(getattr(config, "adaptive_cvae_output_adapter", 0)):
            self.output_semantic_adapter = nn.Sequential(nn.LayerNorm(3 * h), nn.Linear(3 * h, h), nn.SiLU(), nn.Linear(h, h))
            self.output_function_bank = AdaptiveCVAEFunctionBank(config)
        else:
            self.output_semantic_adapter = None
            self.output_function_bank = None
        self.refine_block = AdaptiveRecurrentCVAERefinementBlock(config)
        self._init_residual(self.progress_seed_adapter, std=float(getattr(config, "latent_cvae_output_init_std", 1e-3)))
        self._init_residual(self.context_residual_adapter, std=float(getattr(config, "latent_cvae_output_init_std", 1e-3)))
        self._init_residual(self.token_semantic_adapter, std=0.0)
        if self.output_semantic_adapter is not None:
            self._init_residual(self.output_semantic_adapter, std=0.0)
        self._init_residual(self.trajectory_control_adapter, std=0.0)
        self._init_residual(self.trajectory_update_adapter, std=0.0)
        self._init_residual(self.trajectory_context_adapter, std=0.0)
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
    def _horizon_bias(param: Tensor, action: Tensor) -> Tensor:
        horizon = int(action.shape[1])
        bias = param.to(device=action.device, dtype=action.dtype)
        if horizon <= int(bias.shape[1]):
            return bias[:, :horizon]
        repeat = math.ceil(horizon / int(bias.shape[1]))
        return bias.repeat(1, repeat, 1)[:, :horizon]

    def _trajectory_enabled(self) -> bool:
        return bool(int(getattr(self.config, "latent_cvae_trajectory_denoise", 1)))

    def _trajectory_basis(self, ref: Tensor) -> Tensor:
        return self.trajectory_basis.to(device=ref.device, dtype=ref.dtype)

    def _trajectory_project(self, tokens: Tensor) -> Tensor:
        analysis = self.trajectory_analysis.to(device=tokens.device, dtype=tokens.dtype)
        return torch.einsum("ct,bth->bch", analysis, tokens)

    def _trajectory_expand(self, controls: Tensor) -> Tensor:
        basis = self._trajectory_basis(controls)
        return torch.einsum("tc,bch->bth", basis, controls)

    def _trajectory_control_from_tokens(self, tokens: Tensor) -> Tensor:
        controls = self._trajectory_project(tokens)
        return controls + self.trajectory_control_adapter(controls)

    def _trajectory_update_from_tokens(self, update: Tensor) -> Tensor:
        controls = self._trajectory_project(update)
        return controls + self.trajectory_update_adapter(controls)

    @staticmethod
    def _trajectory_control_smoothness(controls: Tensor) -> Tensor:
        if int(controls.shape[1]) < 2:
            return controls.float().square().mean() * 0.0
        first = controls[:, 1:] - controls[:, :-1]
        loss = 0.25 * first.float().square().mean()
        if int(controls.shape[1]) >= 3:
            second = controls[:, 2:] - 2.0 * controls[:, 1:-1] + controls[:, :-2]
            loss = loss + second.float().square().mean()
        return loss

    def _apply_trajectory_update(
        self,
        *,
        action: Tensor,
        controls: Tensor | None,
        token_update: Tensor,
        pos: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor, Tensor]:
        if controls is None or not self._trajectory_enabled():
            return action + token_update, controls, token_update, torch.zeros_like(token_update)
        control_update = (
            float(getattr(self.config, "latent_cvae_trajectory_update_scale", 1.0))
            * self._trajectory_update_from_tokens(token_update)
        )
        next_controls = controls + control_update
        next_action = self._trajectory_expand(next_controls)
        if pos is not None:
            # V53.5 (#1 fix): the positional basis is carried outside the
            # smooth control-point manifold; the invariant is
            # action == expand(controls) + pos throughout the loop.
            next_action = next_action + pos
        return next_action, next_controls, next_action - action, control_update

    def _record_trajectory_state(
        self,
        rows: list[Tensor],
        *,
        action: Tensor,
        cond: Tensor,
        stage: str = "refine",
    ) -> None:
        if not (
            self.training
            and self._trajectory_enabled()
            and int(getattr(self.config, "latent_cvae_trajectory_mid_supervision", 1))
        ):
            return
        # V53.1: only the homogeneous refine segment (+ final state) is a
        # meaningful coarse-to-fine progression.  Seed/block/canvas states are
        # excluded from mid supervision so they are not pushed to emit the
        # final answer directly through the shared velocity head.
        if (
            int(getattr(self.config, "latent_cvae_trajectory_mid_refine_only", 0))
            and stage not in ("refine", "final")
        ):
            return
        rows.append(self._emit_action(action, cond)["pred_velocity"])

    def _trajectory_step_context(
        self,
        *,
        action: Tensor,
        trajectory_tokens: Tensor,
        progress_context: Tensor,
        context_dir: Tensor,
        step_bias: Tensor,
    ) -> Tensor:
        if not (
            self._trajectory_enabled()
            and int(getattr(self.config, "latent_cvae_trajectory_context", 1))
        ):
            return torch.zeros_like(action)
        source = trajectory_tokens.to(device=action.device, dtype=action.dtype)
        if int(source.shape[1]) != int(action.shape[1]):
            if int(source.shape[1]) == 1:
                source = source.expand(-1, int(action.shape[1]), -1)
            else:
                source = F.interpolate(
                    source.transpose(1, 2),
                    size=int(action.shape[1]),
                    mode="linear",
                    align_corners=True,
                ).transpose(1, 2)
        raw = torch.cat([action, source, progress_context, context_dir, step_bias], dim=-1)
        scale = float(getattr(self.config, "latent_cvae_trajectory_context_scale", 1.0))
        context = scale * self.trajectory_context_adapter(raw)
        cap = float(getattr(self.config, "latent_cvae_trajectory_context_norm_max", 0.0))
        if cap > 0:
            norm = context.float().norm(dim=-1, keepdim=True)
            context = context * (cap / norm.clamp_min(cap)).to(dtype=context.dtype)
        return context

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

    def _monotonic_positions(self, count: int, device: torch.device) -> Tensor:
        if count <= 1:
            return torch.zeros(count, device=device, dtype=torch.float32)
        return torch.linspace(0.0, 1.0, count, device=device, dtype=torch.float32)

    def _step_center(self, step: int | None) -> float | None:
        if step is None or not int(getattr(self.config, "adaptive_cvae_monotonic_layer_route", 0)):
            return None
        total = max(int(self.refine_steps), 1)
        return (float(step) + 0.5) / float(total)

    def _monotonic_route_bias(self, logits: Tensor, *, step: int | None, slot_count: int) -> Tensor:
        """V53-B2: soft depth alignment: logits -= scale * (slot_pos - center)^2."""
        center = self._step_center(step)
        if center is None:
            return logits
        scale = float(getattr(self.config, "adaptive_cvae_layer_route_distance_scale", 3.0))
        if scale <= 0:
            return logits
        pos = self._monotonic_positions(slot_count, logits.device)
        return logits - scale * (pos.view(*([1] * (logits.ndim - 1)), slot_count) - center).square()

    def _route_layers(self, action: Tensor, layer_stack: Tensor | None, *, step: int | None = None) -> tuple[Tensor, Tensor, Tensor]:
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
        logits = self._monotonic_route_bias(logits, step=step, slot_count=int(layer_stack.shape[1]))
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
        if int(getattr(cfg, "adaptive_cvae_monotonic_layer_route", 0)):
            # V53-B2: static capsule<->layer depth alignment so capsule index c
            # prefers layers at matching relative depth.
            align_scale = float(getattr(cfg, "adaptive_cvae_layer_route_distance_scale", 3.0))
            if align_scale > 0:
                cap_pos = self._monotonic_positions(int(logits.shape[1]), logits.device)
                lay_pos = self._monotonic_positions(int(logits.shape[2]), logits.device)
                logits = logits - align_scale * (cap_pos[:, None] - lay_pos[None, :]).square()[None]
        weights = self._route_weights(logits, query).to(dtype=dtype)
        capsules = torch.einsum("bcl,blh->bch", weights, v) + self.context_capsule_role_lift(capsule_role)[None]
        capsules = self.context_capsule_block(capsules, cond_time)
        wf = weights.float().clamp_min(1e-8)
        entropy = -(wf * wf.log()).sum(dim=-1).mean()
        max_weight = wf.detach().max(dim=-1).values.mean()
        return capsules, entropy, max_weight

    def _route_context_capsules(self, action: Tensor, capsules: Tensor | None, *, step: int | None = None) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        if capsules is None:
            z = torch.zeros((), device=action.device, dtype=torch.float32)
            return torch.zeros_like(action), z, z, None
        cfg = self.config
        capsules = capsules.to(device=action.device, dtype=action.dtype)
        role = self.context_capsule_role_basis.to(device=action.device, dtype=action.dtype)
        q = self.context_route_query(action)
        if int(getattr(cfg, "adaptive_cvae_route_query_bias", 1)):
            q = q + self._horizon_bias(self.context_route_query_bias, action)
        k = self.context_route_key(capsules) + self.context_route_role_key(role)[None]
        v = self.context_route_value(capsules) + self.context_route_role_value(role)[None]
        if int(getattr(cfg, "adaptive_cvae_route_cosine", 1)):
            logits = torch.einsum("bth,bch->btc", F.normalize(q.float(), dim=-1), F.normalize(k.float(), dim=-1))
        else:
            logits = torch.einsum("bth,bch->btc", q, k).float() * (float(self.hidden_size) ** -0.5)
        logits = logits + self._role_route_logits(action, role, scale=self.context_role_logit_scale)
        logits = self._monotonic_route_bias(logits, step=step, slot_count=int(capsules.shape[1]))
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
            + self.progress_z_lift(z.to(device=device, dtype=dtype))[:, None]
        )
        return self.progress_block(progress, cond_time)

    def _route_progress_full(self, action: Tensor, progress: Tensor | None) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        if progress is None:
            z = torch.zeros((), device=action.device, dtype=torch.float32)
            return torch.zeros_like(action), z, z, None
        cfg = self.config
        progress = progress.to(device=action.device, dtype=action.dtype)
        q = self.progress_action_query(action)
        if int(getattr(cfg, "adaptive_cvae_route_query_bias", 1)):
            q = q + self._horizon_bias(self.progress_route_query_bias, action)
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

    def _route_progress(self, action: Tensor, progress: Tensor | None) -> tuple[Tensor, Tensor, Tensor]:
        routed, entropy, max_weight, _ = self._route_progress_full(action, progress)
        return routed, entropy, max_weight

    def _progress_seed_delta(self, action: Tensor, progress_context: Tensor) -> Tensor:
        if not int(getattr(self.config, "adaptive_cvae_progress_z_injection", 1)):
            return torch.zeros_like(action)
        return self.progress_seed_adapter(torch.cat([action, progress_context], dim=-1))

    def _function_delta(self, bank: AdaptiveCVAEFunctionBank, x: Tensor, weights: Tensor | None) -> Tensor:
        if not int(getattr(self.config, "adaptive_cvae_function_adapters", 0)):
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
        progress_context, _, _, progress_weights = self._route_progress_full(action, progress)
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
        cond: Tensor,
        z: Tensor,
        layer_stack: Tensor | None = None,
        canvas_memory: Tensor | None = None,
    ) -> dict[str, Tensor]:
        cfg = self.config
        batch = int(noisy_physical.shape[0])
        dtype = noisy_physical.dtype
        device = noisy_physical.device
        time_emb = self.time(time.to(dtype=dtype))
        cond_time = cond + self.time_lift(time_emb)
        z0 = torch.zeros((), device=device, dtype=torch.float32)
        noisy_branch, noisy_gate_mean, noisy_branch_norm = self._gated_noisy_branch(noisy_physical, time)
        pos = self.horizon_query.to(device=device, dtype=dtype).expand(batch, -1, -1)
        base_raw = (
            pos
            + noisy_branch
            + self.trajectory_lift(trajectory_tokens)
            + cond_time[:, None]
        )
        noisy_branch_ratio = noisy_branch_norm / base_raw.detach().float().norm(dim=-1).mean().clamp_min(1e-6)
        # V53.2: hinge on the live (grad-carrying) branch share so the lift
        # cannot amplify itself to cancel the t-gate.
        noisy_ratio_max = float(getattr(cfg, "latent_cvae_noisy_ratio_max", 0.0))
        if noisy_ratio_max > 0:
            live_ratio = noisy_branch.float().norm(dim=-1).mean() / base_raw.detach().float().norm(dim=-1).mean().clamp_min(1e-6)
            noisy_ratio_regularizer = F.relu(live_ratio - noisy_ratio_max).square()
        else:
            noisy_ratio_regularizer = torch.zeros((), device=device, dtype=torch.float32)
        # V53.5 (#1 fix): exempt the horizon positional basis from every
        # temporal smoothing operator (coarse base + control-point projection).
        # v51 smoothed the full token, collapsing adjacent-position
        # distinguishability to ~2% and destroying event timing downstream.
        pos_exempt = bool(int(getattr(cfg, "latent_cvae_trajectory_pos_exempt", 0)))
        pos_anchor = pos if pos_exempt else torch.zeros_like(pos)
        base_action = self._coarse_temporal_base(base_raw - pos_anchor) + pos_anchor
        base_highfreq = (base_raw - base_action).detach().float().norm(dim=-1).mean()
        progress = self._latent_progress(batch=batch, cond_time=cond_time, z=z)
        seed_entropy = z0
        seed_max = z0
        seed_temperature = z0
        route_floor_terms: list[Tensor] = []
        regularizer_terms: list[Tensor] = []
        function_rows: list[Tensor] = []
        trajectory_enabled = self._trajectory_enabled()
        trajectory_control_rows: list[Tensor] = []
        trajectory_token_rows: list[Tensor] = []
        trajectory_update_rows: list[Tensor] = []
        trajectory_context_rows: list[Tensor] = []
        trajectory_projection_rows: list[Tensor] = []
        trajectory_control_smooth_rows: list[Tensor] = []
        trajectory_update_smooth_rows: list[Tensor] = []
        trajectory_update_energy_rows: list[Tensor] = []
        trajectory_projection_regularizer_terms: list[Tensor] = []
        trajectory_pred_rows: list[Tensor] = []

        def track_trajectory_state(
            *,
            action_state: Tensor,
            controls: Tensor | None,
            token_update: Tensor | None = None,
            actual_update: Tensor | None = None,
            control_update: Tensor | None = None,
            stage: str = "refine",
        ) -> None:
            if not trajectory_enabled or controls is None:
                return
            trajectory_control_rows.append(controls.detach().float().norm(dim=-1).mean())
            trajectory_token_rows.append(action_state.detach().float().norm(dim=-1).mean())
            trajectory_control_smooth_rows.append(self._trajectory_control_smoothness(controls))
            if control_update is not None:
                trajectory_update_rows.append(control_update.detach().float().norm(dim=-1).mean())
                trajectory_update_smooth_rows.append(self._trajectory_control_smoothness(control_update))
                trajectory_update_energy_rows.append(control_update.float().square().mean())
            if token_update is not None and actual_update is not None:
                projection = actual_update - token_update
                trajectory_projection_rows.append(projection.detach().float().norm(dim=-1).mean())
                trajectory_projection_regularizer_terms.append(projection.float().square().mean())
            self._record_trajectory_state(trajectory_pred_rows, action=action_state, cond=cond, stage=stage)
        if progress is not None and int(getattr(cfg, "adaptive_cvae_progress_z_injection", 1)):
            seed_temperature = self._adaptive_route_temperature(base_action).detach().float().mean()
            seed_context, seed_entropy, seed_max, seed_weights = self._route_progress_full(base_action, progress)
            route_floor_terms.append(self._route_entropy_floor(seed_entropy, int(progress.shape[1])))
            seed_context = self._context_dropout(seed_context)
            seed_function = self._function_delta(self.seed_function_bank, base_action + seed_context, seed_weights)
            seed_delta = self._progress_seed_delta(base_action, seed_context) + seed_function
            seed_delta = seed_delta * float(getattr(cfg, "adaptive_cvae_seed_scale", 1.0))
            regularizer_terms.append(seed_delta.float().square().mean())
            function_rows.append(seed_function.detach().float().norm(dim=-1).mean())
            token_action = base_action + seed_delta
        else:
            seed_delta = torch.zeros_like(base_action)
            token_action = base_action + self.z_to_token(z.to(device=device, dtype=dtype))[:, None]
        if trajectory_enabled:
            trajectory_control = self._trajectory_control_from_tokens(token_action - pos_anchor)
            action = self._trajectory_expand(trajectory_control) + pos_anchor
            trajectory_projection_rows.append((action - token_action).detach().float().norm(dim=-1).mean())
            trajectory_projection_regularizer_terms.append((action - token_action).float().square().mean())
            track_trajectory_state(action_state=action, controls=trajectory_control, stage="seed")
        else:
            trajectory_control = None
            action = token_action
        for block in self.blocks:
            before = action
            block_action = block(action, cond_time)
            if trajectory_enabled:
                token_update = block_action - before
                action, trajectory_control, actual_update, trajectory_update = self._apply_trajectory_update(
                    action=before,
                    controls=trajectory_control,
                    token_update=token_update,
                    pos=pos_anchor,
                )
                track_trajectory_state(
                    action_state=action,
                    controls=trajectory_control,
                    token_update=token_update,
                    actual_update=actual_update,
                    control_update=trajectory_update,
                    stage="block",
                )
            else:
                action = block_action
        canvas_cross_norm = z0
        canvas_update = self._canvas_cross_update(action, canvas_memory)
        if canvas_update is not None:
            # V53-C1: full-resolution trunk read.  Routed through the
            # trajectory control-point projection like every other update so
            # the action stays on the smooth manifold.
            action, trajectory_control, canvas_actual, canvas_traj_update = self._apply_trajectory_update(
                action=action,
                controls=trajectory_control,
                token_update=canvas_update,
                pos=pos_anchor,
            )
            track_trajectory_state(
                action_state=action,
                controls=trajectory_control,
                token_update=canvas_update,
                actual_update=canvas_actual,
                control_update=canvas_traj_update,
                stage="canvas",
            )
            canvas_cross_norm = canvas_update.detach().float().norm(dim=-1).mean()
            regularizer_terms.append(canvas_update.float().square().mean())
        context_capsules, capsule_layer_entropy, capsule_layer_max = self._context_capsules(
            cond_time=cond_time,
            layer_stack=layer_stack,
            progress=progress,
        )
        if context_capsules is not None and layer_stack is not None:
            route_floor_terms.append(self._route_entropy_floor(capsule_layer_entropy, int(layer_stack.shape[1])))

        update_rows: list[Tensor] = []
        entropy_rows: list[Tensor] = []
        max_rows: list[Tensor] = []
        progress_entropy_rows: list[Tensor] = []
        progress_max_rows: list[Tensor] = []
        continue_rows: list[Tensor] = []
        keep_rows: list[Tensor] = []
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
        t = time.to(device=device, dtype=dtype)[:, None, None]
        for step in range(max(self.refine_steps, 0)):
            step_bias = self._refine_step_bias(step, action)
            route_action = action + step_bias
            temperature_rows.append(self._adaptive_route_temperature(route_action).detach().float().mean())
            progress_context, progress_entropy, progress_max, progress_weights = self._route_progress_full(route_action, progress)
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
                context, entropy, max_weight, _ = self._route_context_capsules(route_action, context_capsules, step=step)
                route_floor_terms.append(self._route_entropy_floor(entropy, int(context_capsules.shape[1])))
                context_dir = self.context_direction_norm(context)
                routed, strength, context_dir = self._semantic_context_residual(
                    action=route_action,
                    cond_time=cond_time,
                    context=context,
                    progress_context=progress_context,
                    step_bias=step_bias,
                )
                strength_f = strength.detach().float()
                condition_strength_mean_rows.append(strength_f.mean())
                condition_strength_std_rows.append(strength_f.std(unbiased=False))
                condition_strength_max_rows.append(strength_f.max())
                condition_strength_min_rows.append(strength_f.min())
                condition_residual_rows.append(routed.detach().float().norm(dim=-1).mean())
                context_direction_rows.append(context_dir.detach().float().norm(dim=-1).mean())
                regularizer_terms.append(routed.float().square().mean())
            else:
                routed, entropy, max_weight = self._route_layers(route_action, layer_stack, step=step)
                if (
                    layer_stack is not None
                    and int(getattr(cfg, "adaptive_cvae_layer_routing", 1))
                    and int(getattr(cfg, "latent_cvae_layer_memory", 1))
                ):
                    route_floor_terms.append(self._route_entropy_floor(entropy, int(layer_stack.shape[1])))
                context_dir = self.context_direction_norm(routed)
            routed = self._context_dropout(routed)
            trajectory_context = self._trajectory_step_context(
                action=route_action,
                trajectory_tokens=trajectory_tokens,
                progress_context=progress_context,
                context_dir=context_dir,
                step_bias=step_bias,
            )
            if trajectory_enabled:
                context_dir = self.trajectory_context_norm(context_dir + trajectory_context)
                routed = routed + trajectory_context
                trajectory_context_rows.append(trajectory_context.detach().float().norm(dim=-1).mean())
                regularizer_terms.append(trajectory_context.float().square().mean())
            cond_step = cond_time + trajectory_context.mean(dim=1) if trajectory_enabled else cond_time
            # V53-C2: serial writers.  Each lateral writer reads the running
            # accumulation of its predecessors (trajectory_context ->
            # semantic_bias -> function_delta -> refine input) instead
            # of all writers reading the same input in parallel.  Outputs are
            # still applied exactly as before; only conditioning is chained.
            serial_writers = bool(int(getattr(cfg, "adaptive_cvae_serial_writers", 0)))
            writer_chain = trajectory_context if trajectory_enabled else torch.zeros_like(route_action)
            semantic_bias = self._token_semantic_bias(
                action=route_action + writer_chain if serial_writers else route_action,
                cond_time=cond_step,
                routed=routed,
                progress_context=progress_context,
            )
            if serial_writers:
                writer_chain = writer_chain + semantic_bias
            regularizer_terms.append(semantic_bias.float().square().mean())
            function_source = (
                route_action + writer_chain + routed + progress_context
                if serial_writers
                else route_action + routed + progress_context
            )
            function_bias = self._function_delta(self.refine_function_bank, function_source, progress_weights)
            if serial_writers:
                writer_chain = writer_chain + function_bias
            regularizer_terms.append(function_bias.float().square().mean())
            function_rows.append(function_bias.detach().float().norm(dim=-1).mean())
            prefix = prefix + semantic_bias + function_bias
            before = action
            block_action, keep_map = self.refine_block(action, cond_step, routed, prefix)
            keep = keep_map.detach().float().mean()
            keep_rows.append(keep_map.detach().float().squeeze(-1))
            token_update = block_action - before
            action, trajectory_control, actual_update, trajectory_update = self._apply_trajectory_update(
                action=before,
                controls=trajectory_control,
                token_update=token_update,
                pos=pos_anchor,
            )
            track_trajectory_state(
                action_state=action,
                controls=trajectory_control,
                token_update=token_update,
                actual_update=actual_update,
                control_update=trajectory_update,
            )
            update_rows.append((action - before).detach().float().norm(dim=-1).mean())
            entropy_rows.append(entropy.to(device=device))
            max_rows.append(max_weight.to(device=device))
            progress_entropy_rows.append(progress_entropy.to(device=device))
            progress_max_rows.append(progress_max.to(device=device))
            continue_rows.append(keep.to(device=device))
            prefix_rows.append(prefix.detach().float().norm(dim=-1).mean())
            semantic_rows.append(semantic_bias.detach().float().norm(dim=-1).mean())
            step_bias_rows.append(step_bias.detach().float().norm(dim=-1).mean())

        output_delta, output_function = self._output_semantic_delta(action=action, cond_time=cond_time, progress=progress)
        output_scale = float(getattr(cfg, "adaptive_cvae_output_scale", 1.0))
        output_delta = output_delta * output_scale
        output_function = output_function * output_scale
        regularizer_terms.append(output_delta.float().square().mean())
        function_rows.append(output_function.detach().float().norm(dim=-1).mean())
        if trajectory_enabled:
            final_action, trajectory_control, final_update, final_trajectory_update = self._apply_trajectory_update(
                action=action,
                controls=trajectory_control,
                token_update=output_delta,
                pos=pos_anchor,
            )
            track_trajectory_state(
                action_state=final_action,
                controls=trajectory_control,
                token_update=output_delta,
                actual_update=final_update,
                control_update=final_trajectory_update,
                stage="final",
            )
        else:
            final_action = action + output_delta
        if keep_rows:
            keep_per_sample = torch.stack(keep_rows, dim=1).mean(dim=(1, 2))
            out_extra_keep = keep_per_sample
            t_flat = time.detach().float().reshape(-1)
            if int(keep_per_sample.shape[0]) > 1 and t_flat.std() > 1e-6 and keep_per_sample.std() > 1e-8:
                kc = keep_per_sample - keep_per_sample.mean()
                tc = t_flat - t_flat.mean()
                time_corr = (kc * tc).mean() / (kc.std() * tc.std()).clamp_min(1e-8)
            else:
                time_corr = torch.zeros((), device=device, dtype=torch.float32)
        else:
            out_extra_keep = torch.zeros(batch, device=device, dtype=torch.float32)
            time_corr = torch.zeros((), device=device, dtype=torch.float32)
        out = self._emit_action(final_action, cond)
        progress_norm = progress.detach().float().norm(dim=-1).mean() if progress is not None else z0
        out.update({
            "adaptive_noisy_gate_mean": noisy_gate_mean.to(device=device),
            "adaptive_noisy_branch_norm": noisy_branch_norm.to(device=device),
            "adaptive_noisy_branch_ratio": noisy_branch_ratio.to(device=device),
            "adaptive_noisy_ratio_regularizer": noisy_ratio_regularizer.to(device=device),
            "adaptive_canvas_cross_norm": canvas_cross_norm.to(device=device) if torch.is_tensor(canvas_cross_norm) else z0,
            "adaptive_canvas_gate": torch.sigmoid(self.canvas_gate.detach()).float() if self.canvas_gate is not None else z0,
            "adaptive_refine_update_mean": torch.stack(update_rows).mean() if update_rows else z0,
            "adaptive_route_entropy": torch.stack(entropy_rows).mean() if entropy_rows else z0,
            "adaptive_route_max": torch.stack(max_rows).mean() if max_rows else z0,
            "adaptive_route_effective_slots": torch.exp(torch.stack(entropy_rows).mean()) if entropy_rows else z0,
            "adaptive_progress_entropy": torch.stack(progress_entropy_rows).mean() if progress_entropy_rows else z0,
            "adaptive_progress_max": torch.stack(progress_max_rows).mean() if progress_max_rows else z0,
            "adaptive_progress_effective_slots": torch.exp(torch.stack(progress_entropy_rows).mean()) if progress_entropy_rows else z0,
            "adaptive_progress_norm": progress_norm,
            "adaptive_continue_mean": torch.stack(continue_rows).mean() if continue_rows else z0,
            "adaptive_continue_std": torch.stack(keep_rows, dim=1).std() if keep_rows else z0,
            "adaptive_continue_first": keep_rows[0].mean() if keep_rows else z0,
            "adaptive_continue_last": keep_rows[-1].mean() if keep_rows else z0,
            "adaptive_continue_per_sample": out_extra_keep,
            "adaptive_continue_time_corr": time_corr,
            "adaptive_prefix_norm": torch.stack(prefix_rows).mean() if prefix_rows else z0,
            "adaptive_progress_seed_entropy": seed_entropy.to(device=device),
            "adaptive_progress_seed_max": seed_max.to(device=device),
            "adaptive_progress_seed_effective_slots": torch.exp(seed_entropy.to(device=device)) if progress is not None else z0,
            "adaptive_progress_seed_norm": seed_delta.detach().float().norm(dim=-1).mean(),
            "adaptive_route_temperature_mean": torch.stack([seed_temperature.to(device=device), *temperature_rows]).mean() if temperature_rows else seed_temperature.to(device=device),
            "adaptive_semantic_bias_norm": torch.stack(semantic_rows).mean() if semantic_rows else z0,
            "adaptive_output_adapter_norm": output_delta.detach().float().norm(dim=-1).mean(),
            "adaptive_function_delta_norm": torch.stack(function_rows).mean() if function_rows else z0,
            "adaptive_base_highfreq_norm": base_highfreq,
            "adaptive_trajectory_control_norm": torch.stack(trajectory_control_rows).mean() if trajectory_control_rows else z0,
            "adaptive_trajectory_token_norm": torch.stack(trajectory_token_rows).mean() if trajectory_token_rows else z0,
            "adaptive_trajectory_update_norm": torch.stack(trajectory_update_rows).mean() if trajectory_update_rows else z0,
            "adaptive_trajectory_context_norm": torch.stack(trajectory_context_rows).mean() if trajectory_context_rows else z0,
            "adaptive_trajectory_projection_error": torch.stack(trajectory_projection_rows).mean() if trajectory_projection_rows else z0,
            "adaptive_trajectory_control_smoothness": torch.stack(trajectory_control_smooth_rows).mean() if trajectory_control_smooth_rows else z0,
            "adaptive_trajectory_update_smoothness": torch.stack(trajectory_update_smooth_rows).mean() if trajectory_update_smooth_rows else z0,
            "adaptive_trajectory_update_energy": torch.stack(trajectory_update_energy_rows).mean() if trajectory_update_energy_rows else z0,
            "adaptive_trajectory_projection_regularizer": torch.stack(trajectory_projection_regularizer_terms).mean() if trajectory_projection_regularizer_terms else z0,
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
            "adaptive_regularizer": torch.stack(regularizer_terms).mean() if regularizer_terms else z0,
            "adaptive_route_entropy_regularizer": torch.stack(route_floor_terms).mean() if route_floor_terms else z0,
        })
        if trajectory_pred_rows:
            out["adaptive_trajectory_pred_velocity"] = torch.stack(trajectory_pred_rows, dim=1)
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
        self.latent_cvae_action_decoder = AdaptiveRecurrentCVAEActionDecoder(config)
        self.block_action_denoise = (
            BlockActionDenoiseMatrix(config)
            if int(getattr(config, "block_action_denoise_matrix", 0))
            else None
        )

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
        time_emb = self.time(time.to(dtype=canvas.dtype))
        gate_rows: list[dict[str, Tensor]] = []
        content_norm_rows: list[Tensor] = []
        time_norm_rows: list[Tensor] = []
        midcut: dict[str, Tensor] | None = None
        layer_contracts: list[dict[str, Tensor]] = []
        # The adaptive CVAE final path consumes layer contracts.  Inference/eval
        # must still materialize them even when auxiliary contract losses are off.
        force_layer_contracts = bool(int(getattr(cfg, "latent_cvae_layer_memory", 1)))
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
                    if int(getattr(cfg, "layer_zero_base_diagnostic", 0)):
                        # V53-A3: rerun the consequence cell with this layer's
                        # rollout tokens zeroed.  Loss-free diagnostic: a small
                        # relative shift means the cell output is a function of
                        # the injected action features alone (action parroting).
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
                    if int(getattr(cfg, "layer_state_counterfactual", 0)) and int(layer_entry["rollout_tokens"].shape[0]) > 1:
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
        dynamics = self.controlled_dynamics(
            rollout_init.to(device=canvas.device, dtype=canvas.dtype),
            context_kv,
            action_tokens=trajectory,
        )
        controlled_delta = dynamics["rollout_delta_pred"]
        rollout_effect_pred = dynamics["rollout_effect_pred"]
        event_context = _rollout_tokens_to_action_horizon(controlled_delta, cfg)
        direct_velocity = self.direct_physical_head(trajectory)
        rollout_residual_velocity, rollout_alpha = self.rollout_residual_head(trajectory_pooled, controlled_delta)
        legacy_velocity = direct_velocity + rollout_residual_velocity
        context_memory = [
            canvas[:, slices["state"]],
            canvas[:, slices["state_history"]],
            canvas[:, slices["executed"]],
            canvas[:, slices["proposal"]],
        ] if int(getattr(cfg, "latent_cvae_context_memory", 0)) else None
        transition_memory = [rollout, controlled_delta, rollout_effect_pred, event_context] if int(getattr(cfg, "latent_cvae_transition_memory", 1)) else None
        canvas_memory = (
            torch.cat([trajectory, rollout], dim=1)
            if int(getattr(cfg, "latent_cvae_canvas_cross_attention", 0))
            else None
        )
        latent_cvae_action = self.latent_cvae_action_decoder(
            noisy_physical=noisy_physical,
            time=time,
            trajectory_tokens=trajectory_pooled,
            context_memory=context_memory,
            transition_memory=transition_memory,
            visual_memory=visual_memory if int(getattr(cfg, "latent_cvae_visual_memory", 0)) else None,
            layer_contracts=layer_contracts,
            target_physical=cvae_target_physical,
            canvas_memory=canvas_memory,
        )
        pred_physical_velocity = latent_cvae_action["pred_velocity"]
        legacy_event_logits = latent_cvae_action["event_logits"]
        legacy_motion_logits = latent_cvae_action["motion_logits"]
        gate_mean = {
            key: torch.stack([row[key] for row in gate_rows]).mean() if gate_rows else _zeros_like_scalar(canvas)
            for key in ("gate_self", "gate_visual", "gate_rollout", "gate_ffn")
        }
        content_norm = torch.stack(content_norm_rows).mean() if content_norm_rows else _zeros_like_scalar(canvas)
        time_norm = torch.stack(time_norm_rows).mean() if time_norm_rows else _zeros_like_scalar(canvas)
        out = {
            **midcut,
            "layer_contracts": layer_contracts,
            "canvas_tokens": canvas,
            "trajectory_tokens": trajectory,
            "rollout_tokens": rollout,
            "register_tokens": registers,
            "direct_physical_velocity": direct_velocity,
            "rollout_residual_velocity": rollout_residual_velocity,
            "legacy_physical_velocity": legacy_velocity,
            "rollout_alpha": rollout_alpha,
            "pred_physical_velocity": pred_physical_velocity,
            "rollout_effect_pred": rollout_effect_pred,
            "rollout_base_effect_pred": dynamics["rollout_base_effect_pred"],
            "rollout_delta_pred": controlled_delta,
            "rollout_coeff_abs_mean": dynamics["rollout_coeff_abs_mean"],
            "rollout_neutral_coeff_abs_mean": dynamics["rollout_neutral_coeff_abs_mean"],
            "rollout_centered_coeff_abs_mean": dynamics["rollout_centered_coeff_abs_mean"],
            "rollout_basis_norm": dynamics["rollout_basis_norm"],
            "rollout_delta_norm": dynamics["rollout_delta_norm"],
            "rollout_base_norm": dynamics["rollout_base_norm"],
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
        # V53.2: generic diagnostics passthrough.  Every decoder key with the
        # cvae_ prefix is exported as latent_cvae_*; this replaces ~200 lines of
        # hand-enumerated entries and cannot drift out of sync with the decoder.
        for key, value in latent_cvae_action.items():
            if key.startswith("cvae_"):
                out[f"latent_{key}"] = value
        out["latent_cvae_gripper_gate_mean"] = latent_cvae_action["gripper_gate_mean"]
        out["latent_cvae_layer_memory_count"] = latent_cvae_action["layer_memory_count"]
        if latent_cvae_action is not None and "post_pred_velocity" in latent_cvae_action:
            out.update({
                "post_pred_velocity": latent_cvae_action["post_pred_velocity"],
                "post_event_logits": latent_cvae_action.get("post_event_logits", legacy_event_logits),
                "post_motion_logits": latent_cvae_action.get("post_motion_logits", legacy_motion_logits),
            })
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

    def _block_action_denoise_params(self, ref: Tensor) -> dict[str, Tensor] | None:
        matrix = getattr(self.planner, "block_action_denoise", None)
        if matrix is None or not int(getattr(self.policy_config, "block_action_denoise_matrix", 0)):
            return None
        return matrix.matrices(ref)

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
        codec_state = state if action_state is None else action_state
        target_physical = self.codec.encode(target_action, codec_state)
        # Deploy-safe proposal base for proposal-residual coefficient losses.
        # It is detached by construction: residual supervision must not pull the
        # history proposal into a target-conditioned shortcut.
        proposal_physical = self.codec.encode(proposal["action"].detach(), codec_state).detach()
        t = torch.rand(target_physical.shape[0], device=target_physical.device, dtype=target_physical.dtype)
        block_action_params = self._block_action_denoise_params(target_physical)
        if block_action_params is not None:
            native_noise_scale = block_action_params["native_noise_scale"].to(
                device=target_action.device,
                dtype=target_action.dtype,
            )
            native_terminal = torch.randn_like(target_action) * native_noise_scale
            noisy_terminal = self.codec.encode(native_terminal, codec_state)
        else:
            noise = torch.randn_like(target_physical)
            noisy_terminal = noise
        noisy_physical = (1 - t[:, None, None]) * target_physical + t[:, None, None] * noisy_terminal
        target_physical_velocity = noisy_terminal - target_physical
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
                )
                preview_velocity = preview["pred_physical_velocity"].detach()
                consequence_input = (noisy_physical - t[:, None, None] * preview_velocity).detach()
                if block_action_params is not None:
                    consequence_input = self._project_physical_action_manifold(consequence_input, codec_state).detach()
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
        clean_physical_estimate = noisy_physical - t[:, None, None] * action_policy["pred_physical_velocity"]
        decoded_action = self.codec.decode(clean_physical_estimate, codec_state)
        if "post_pred_velocity" in action_policy:
            post_clean = noisy_physical - t[:, None, None] * action_policy["post_pred_velocity"]
            action_policy["post_clean_physical_estimate"] = post_clean
            action_policy["post_pred_action_estimate"] = self.codec.decode(post_clean, codec_state)
        out = {
            **action_policy,
            "pred_physical_velocity": action_policy["pred_physical_velocity"],
            "target_physical_velocity": target_physical_velocity,
            "target_physical": target_physical,
            "clean_physical_estimate": clean_physical_estimate,
            "proposal_action": proposal["action"],
            "proposal_physical": proposal_physical,
            "proposal_keep": keep.detach(),
            "time": t,
            "noisy_physical_action": noisy_physical,
            "pred_action_estimate": decoded_action,
            "future_conditioned_action_loss": torch.zeros((), device=target_physical.device, dtype=target_physical.dtype),
        }
        if block_action_params is not None:
            out.update({
                "block_action_native_noise_scale": block_action_params["native_noise_scale"],
                "block_action_noise_scale": block_action_params["noise_scale"],
                "block_action_velocity_loss_weight": block_action_params["velocity_loss_weight"],
                "block_action_x0_mix": block_action_params["x0_mix"],
                "block_action_denoise_regularizer": block_action_params["regularizer"],
                "block_action_denoise_smoothness": block_action_params["smoothness"],
                "block_action_denoise_deviation": block_action_params["deviation"],
                "block_action_denoise_interaction_norm": block_action_params["interaction_norm"],
                "block_action_noise_arm_mean": block_action_params["noise_arm_mean"],
                "block_action_noise_gripper_mean": block_action_params["noise_gripper_mean"],
                "block_action_noise_near_mean": block_action_params["noise_near_mean"],
                "block_action_noise_tail_mean": block_action_params["noise_tail_mean"],
                "block_action_noise_min": block_action_params["noise_min"],
                "block_action_noise_max": block_action_params["noise_max"],
                "block_action_noise_std": block_action_params["noise_std"],
                "block_action_noise_raw_rms": block_action_params["noise_raw_rms"],
                "block_action_noise_rms": block_action_params["noise_rms"],
                "block_action_noise_boundary_jump": block_action_params["noise_boundary_jump"],
                "block_action_loss_arm_mean": block_action_params["loss_arm_mean"],
                "block_action_loss_gripper_mean": block_action_params["loss_gripper_mean"],
                "block_action_x0_near_mean": block_action_params["x0_near_mean"],
                "block_action_x0_tail_mean": block_action_params["x0_tail_mean"],
            })
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
            mid_clean = noisy_physical - t[:, None, None] * action_policy["midcut_pred_physical_velocity"]
            out["midcut_clean_physical_estimate"] = mid_clean
            out["midcut_pred_action_estimate"] = self.codec.decode(mid_clean, codec_state)
        if "layer_contracts" in action_policy:
            for entry in action_policy["layer_contracts"]:
                clean = noisy_physical - t[:, None, None] * entry["pred_physical_velocity"]
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
                hold_noisy = (1 - t[:, None, None]) * hold_physical + t[:, None, None] * noisy_terminal
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
                    # near-duplicate on ordered validation batches.  Here each
                    # sample borrows the batch action with the largest chunkwise
                    # physical-action distance, preserving the same observation
                    # and target future while making the intervention materially
                    # different whenever the batch contains alternatives.
                    flat = target_physical.detach().float().flatten(1)
                    dist = torch.cdist(flat, flat, p=2)
                    eye = torch.eye(dist.shape[0], device=dist.device, dtype=torch.bool)
                    dist = dist.masked_fill(eye, -1.0)
                    perm = dist.argmax(dim=1)
                    shuffle_action = target_action[perm]
                else:
                    shuffle_action = target_action
                shuffle_physical = self.codec.encode(shuffle_action, codec_state)
                shuffle_noisy = (1 - t[:, None, None]) * shuffle_physical + t[:, None, None] * noisy_terminal
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

    def _project_physical_action_manifold(self, physical: Tensor, action_state: Tensor) -> Tensor:
        decoded = self.codec.decode(physical, action_state.to(device=physical.device, dtype=physical.dtype))
        return self.codec.encode(decoded, action_state.to(device=physical.device, dtype=physical.dtype))

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
        block_action_params = self._block_action_denoise_params(
            torch.empty(
                visual.shape[0],
                self.policy_config.action_horizon,
                self.policy_config.physical_action_dim,
                device=visual.device,
                dtype=visual.dtype,
            )
        )
        # V53.5 (#3 fix): the codec boundary/decode anchor must live in the
        # ACTION-normalizer coordinates used during training.  ``state`` is
        # state-normalizer encoded and remains only a legacy fallback; deploy
        # and eval callers must pass ``action_state``.
        codec_state = (state if action_state is None else action_state).to(device=visual.device, dtype=visual.dtype)
        if noise is None:
            if block_action_params is not None:
                native_noise = torch.randn(
                    visual.shape[0],
                    self.policy_config.action_horizon,
                    self.policy_config.action_dim,
                    device=visual.device,
                    dtype=visual.dtype,
                )
                native_scale = block_action_params["native_noise_scale"].to(device=visual.device, dtype=visual.dtype)
                x = self.codec.encode(native_noise * native_scale, codec_state)
            else:
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
                native_noise = x.to(device=visual.device, dtype=visual.dtype)
                if block_action_params is not None:
                    native_noise = native_noise * block_action_params["native_noise_scale"].to(
                        device=visual.device,
                        dtype=visual.dtype,
                    )
                x = self.codec.encode(native_noise, codec_state)
            elif x.shape[-1] != self.policy_config.physical_action_dim:
                raise ValueError("noise must have last dim action_dim or physical_action_dim")
            else:
                x = x.to(device=visual.device, dtype=visual.dtype)
                if block_action_params is not None:
                    x = x * block_action_params["noise_scale"].to(device=x.device, dtype=x.dtype)
                    x = self._project_physical_action_manifold(x, codec_state)
        keep = torch.full((visual.shape[0],), 1.0 if use_proposal else 0.0, device=visual.device, dtype=visual.dtype)
        use_self_condition = (
            int(getattr(self.policy_config, "action_consequence_self_condition", 0))
            and int(getattr(self.policy_config, "layer_recurrent_consequence", 0))
            and int(getattr(self.policy_config, "layer_contract_adapters", 0))
        )
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
                if block_action_params is not None:
                    consequence_input = self._project_physical_action_manifold(consequence_input, codec_state).detach()
                del preview
            out = self._policy_forward(
                x, t, visual, state_history, state, executed_history, proposal["tokens"], keep,
                stop_at_midcut=stop_at_midcut,
                consequence_physical=consequence_input,
                enable_layer_contracts=False,
            )
            x = x - out["pred_physical_velocity"] / float(steps)
            if block_action_params is not None:
                x = self._project_physical_action_manifold(x, codec_state)
        action = self.codec.decode(x, codec_state)
        if return_event_logits:
            zero_t = torch.zeros((visual.shape[0],), device=visual.device, dtype=visual.dtype)
            event = self._policy_forward(
                x, zero_t, visual, state_history, state, executed_history, proposal["tokens"], keep,
                stop_at_midcut=stop_at_midcut,
                enable_layer_contracts=False,
            )
            return {"action": action, "physical_action": x, "event_logits": event["event_logits"], "motion_logits": event["motion_logits"]}
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
            "latent_cvae_action_decoder": (
                0 if getattr(self.planner, "latent_cvae_action_decoder", None) is None
                else sum(p.numel() for p in self.planner.latent_cvae_action_decoder.parameters())
            ),
            "block_action_denoise_matrix": (
                0 if getattr(self.planner, "block_action_denoise", None) is None
                else sum(p.numel() for p in self.planner.block_action_denoise.parameters())
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
    "LayerRoleScheduler",
    "UnifiedInterventionBlock",
    "RecurrentMilestoneConsequenceCell",
    "BlockActionDenoiseMatrix",
    "_align_milestone_tokens_to_horizon",
    "_rollout_tokens_to_action_horizon",
    "TemporalMidcutWorldActionDiT",
    "V39PolicySystem",
]
# V53: vertical-depth revision (t-gated x_t branch, boosting layer contracts,
# depth-scan condition, monotonic layer routing, canvas cross-attention,
# serialized refine writers).
