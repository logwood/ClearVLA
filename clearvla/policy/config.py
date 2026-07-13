from __future__ import annotations

"""Configuration lineage for the current staged policy."""

from dataclasses import dataclass


@dataclass(frozen=True)
class V362PolicyConfig:
    action_dim: int = 7
    state_dim: int = 7
    action_horizon: int = 24
    executed_history_length: int = 3
    hidden_size: int = 512
    num_heads: int = 8
    depth: int = 6
    action_decoder_depth: int = 4
    proposal_depth: int = 2
    ffn_expansion: float = 4.0
    proposal_dropout: float = 0.25
    dropout: float = 0.05
    event_tokens: int = 3
    gripper_dim_index: int = -1
    inference_steps: int = 5
    first_execution_steps: int = 4
    mid_execution_steps: int = 8
    physical_decode_delta_blend: float = 0.25
    gripper_field_dim: int = 12
    gripper_field_mode: str = "legacy_handcrafted"
    # Historical runs sampled arm_abs/arm_delta independently. New runs can
    # instead sample one native arm trajectory and map it into the redundant
    # [absolute, delta] coordinates used by the policy.
    arm_flow_mode: str = "legacy_independent"
    arm_noise_temporal_rho: float = 0.0

    def validate(self) -> None:
        if min(
            self.action_dim,
            self.state_dim,
            self.action_horizon,
            self.executed_history_length,
            self.hidden_size,
            self.num_heads,
            self.depth,
            self.action_decoder_depth,
            self.proposal_depth,
            self.event_tokens,
            self.inference_steps,
            self.first_execution_steps,
            self.mid_execution_steps,
        ) <= 0:
            raise ValueError("V36.2 policy dimensions must be positive")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.action_dim != self.state_dim:
            raise ValueError("action/state dimensions must match")
        if not 0 <= self.proposal_dropout < 1:
            raise ValueError("proposal_dropout must be in [0,1)")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        if not 0 <= self.physical_decode_delta_blend <= 1:
            raise ValueError("physical_decode_delta_blend must be in [0,1]")
        if int(self.gripper_field_dim) < 2:
            raise ValueError("gripper_field_dim must be >= 2")
        if str(self.gripper_field_mode) not in {"legacy_handcrafted", "parseval_temporal"}:
            raise ValueError("gripper_field_mode must be legacy_handcrafted or parseval_temporal")
        if str(self.arm_flow_mode) not in {"legacy_independent", "manifold_native"}:
            raise ValueError("arm_flow_mode must be legacy_independent or manifold_native")
        if not 0.0 <= float(self.arm_noise_temporal_rho) < 1.0:
            raise ValueError("arm_noise_temporal_rho must be in [0,1)")
        if self.first_execution_steps > self.action_horizon:
            raise ValueError("first_execution_steps cannot exceed action_horizon")
        if self.mid_execution_steps > self.action_horizon:
            raise ValueError("mid_execution_steps cannot exceed action_horizon")

    @property
    def gripper_index(self) -> int:
        return self.gripper_dim_index if self.gripper_dim_index >= 0 else self.action_dim + self.gripper_dim_index

    @property
    def arm_dim(self) -> int:
        return self.action_dim - 1

    @property
    def physical_action_dim(self) -> int:
        # arm_abs + arm_delta + expanded gripper field. Legacy mode reserves
        # value/delta channels; Parseval mode reconstructs the native gripper
        # trajectory jointly from every field channel.
        return 2 * self.arm_dim + int(self.gripper_field_dim)


@dataclass(frozen=True)
class V38PolicyConfig(V362PolicyConfig):
    """Configuration for the latent-dynamics-bound canvas."""

    visual_token_dim: int = 768
    visual_history_length: int = 3
    num_cameras: int = 2
    patches_per_camera: int = 576
    canvas_registers: int = 12
    future_anchors: int = 4
    target_future_count: int = 12
    visual_memory_dropout: float = 0.0
    canvas_dropout: float = 0.0
    role_dropout: float = 0.10
    action_basis_tokens: int = 4
    future_grid_size: int = 4
    # Kept for checkpoint/context compatibility.  V38.5 does not use a
    # future-noisy input branch by default.
    future_flow_loss_weight: float = 0.0
    # Tail action residual binding schedule.  Early actions may be read directly
    # from action tokens; mid/tail actions increasingly must read rollout tokens.
    rollout_tail_start_step: int = 8
    rollout_tail_full_step: int = 13
    # V38.6.2 action-centered controlled residual dynamics.  ``base_effect``
    # has deliberately small capacity; ``controlled_delta`` is produced by
    # action coefficients centered against a neutral/no-op coefficient.
    controlled_delta_rank: int = 8
    base_effect_hidden: int = 128
    latent_action_tokens: int = 8
    controlled_delta_dropout: float = 0.0
    neutral_action_tokens: int = 4

    def validate(self) -> None:
        super().validate()
        if min(
            self.visual_token_dim,
            self.visual_history_length,
            self.num_cameras,
            self.patches_per_camera,
            self.canvas_registers,
            self.future_anchors,
            self.target_future_count,
            self.action_basis_tokens,
            self.future_grid_size,
        ) <= 0:
            raise ValueError("V38 dimensions must be positive")
        if self.future_anchors > self.target_future_count:
            raise ValueError("future_anchors cannot exceed target_future_count")
        if not 0 <= self.visual_memory_dropout < 1:
            raise ValueError("visual_memory_dropout must be in [0,1)")
        if not 0 <= self.canvas_dropout < 1:
            raise ValueError("canvas_dropout must be in [0,1)")
        if not 0 <= self.role_dropout < 1:
            raise ValueError("role_dropout must be in [0,1)")
        if self.rollout_tail_start_step < 1 or self.rollout_tail_full_step < self.rollout_tail_start_step:
            raise ValueError("invalid rollout tail binding schedule")
        if min(self.controlled_delta_rank, self.base_effect_hidden, self.latent_action_tokens, self.neutral_action_tokens) <= 0:
            raise ValueError("controlled residual dynamics dimensions must be positive")
        if not 0 <= self.controlled_delta_dropout < 1:
            raise ValueError("controlled_delta_dropout must be in [0,1)")

    @property
    def future_token_count(self) -> int:
        return int(self.future_anchors) * int(self.num_cameras) * int(self.future_grid_size) * int(self.future_grid_size)

    @property
    def history_length(self) -> int:
        return self.visual_history_length

    @property
    def num_future(self) -> int:
        return self.target_future_count

    @property
    def latent_dim(self) -> int:
        return self.visual_token_dim


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
    # CR1/B1: 1 = legacy variational training (posterior/KL/aux decode);
    # 0 = deterministic bypass with identical deploy mapping (prior mean only).
    latent_cvae_variational: int = 1
    # CR0 (§14.2): eval-time z zero/shuffle intervention probes on the legacy
    # decoder.  Costs two extra decodes per eval batch; diagnostic runs only.
    latent_cvae_z_probe: int = 0
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

    # Pre-V76 phase 1: deterministic intent contracts plus one-owner MMDiT
    # evidence consumption.  This is an independent final decoder rather than
    # another behavior flag inside the historical CVAE tower.  It has no
    # posterior, target-action input, latent sampling, or legacy output bypass.
    hierarchical_mmdit_depth: int = 3
    hierarchical_mmdit_refine_steps: int = 3
    hierarchical_mmdit_low_slots: int = 25
    hierarchical_mmdit_stage_slots: int = 6
    hierarchical_mmdit_ffn_expansion: float = 2.0
    hierarchical_mmdit_layer_grad_scale: float = 0.0
    hierarchical_mmdit_source_grad_scale: float = 0.0
    hierarchical_mmdit_consequence_scale_init: float = 0.10
    hierarchical_mmdit_consequence_scale_max: float = 0.50
    hierarchical_mmdit_noisy_causal: int = 1
    hierarchical_mmdit_noisy_gate_min: float = 0.05
    hierarchical_mmdit_noisy_gate_power: float = 1.5
    hierarchical_mmdit_stage_promote_scale_init: float = 0.05
    hierarchical_mmdit_output_init_std: float = 1e-3
    hierarchical_mmdit_residual_scale_max: float = 0.20
    # V77: every residual writer is normalized before its scalar amplitude
    # control, so projection weights cannot bypass the ownership contract.
    hierarchical_mmdit_architecture_version: str = "serial_owned_rms_v3"
    # CR7 fallback (§23): dedicated restricted contract for event/motion
    # subheads (never velocity).  0 = action-only output (mainline first try).
    hierarchical_mmdit_output_contract: int = 0
    # Compatibility switch retained for v76a metadata/checkpoints.  The clean
    # serial decoder no longer has a condition-group market, so this flag is a
    # no-op there; legacy MMDiT paths keep their historical behavior.
    hierarchical_mmdit_noisy_market_bias: int = 0
    # 0 = no noisy value gate; 1 = a post-normalization residual-amplitude schedule.
    hierarchical_mmdit_noisy_gate_mode: int = 0
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
        if str(self.final_action_decoder) not in {"legacy", "residual_action_flow", "layered_residual_action_flow", "latent_main_action", "latent_cvae_action", "adaptive_recurrent_cvae_action", "hierarchical_mmdit_action"}:
            raise ValueError("final_action_decoder must be legacy, residual_action_flow, layered_residual_action_flow, latent_main_action, latent_cvae_action, adaptive_recurrent_cvae_action, or hierarchical_mmdit_action")
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
            "latent_cvae_inference_sample", "latent_cvae_variational", "latent_cvae_z_probe",
            "latent_cvae_noisy_gate", "latent_cvae_layer_scan",
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
        if int(self.hierarchical_mmdit_depth) < 1:
            raise ValueError("hierarchical_mmdit_depth must be >= 1")
        if int(self.hierarchical_mmdit_refine_steps) < 1:
            raise ValueError("hierarchical_mmdit_refine_steps must be >= 1")
        if int(self.hierarchical_mmdit_low_slots) < 1:
            raise ValueError("hierarchical_mmdit_low_slots must be >= 1")
        if int(self.hierarchical_mmdit_stage_slots) < 1:
            raise ValueError("hierarchical_mmdit_stage_slots must be >= 1")
        if float(self.hierarchical_mmdit_ffn_expansion) < 1.0:
            raise ValueError("hierarchical_mmdit_ffn_expansion must be >= 1")
        for name in ("hierarchical_mmdit_layer_grad_scale", "hierarchical_mmdit_source_grad_scale"):
            if not (0.0 <= float(getattr(self, name)) <= 1.0):
                raise ValueError(f"{name} must be in [0, 1]")
        if float(self.hierarchical_mmdit_consequence_scale_max) <= 0.0:
            raise ValueError("hierarchical_mmdit_consequence_scale_max must be positive")
        if not (
            0.0 <= float(self.hierarchical_mmdit_consequence_scale_init)
            <= float(self.hierarchical_mmdit_consequence_scale_max)
        ):
            raise ValueError("hierarchical_mmdit_consequence_scale_init must be in [0, max]")
        if int(self.hierarchical_mmdit_noisy_causal) not in (0, 1):
            raise ValueError("hierarchical_mmdit_noisy_causal must be 0 or 1")
        if int(self.hierarchical_mmdit_output_contract) not in (0, 1):
            raise ValueError("hierarchical_mmdit_output_contract must be 0 or 1")
        if int(self.hierarchical_mmdit_noisy_market_bias) not in (0, 1):
            raise ValueError("hierarchical_mmdit_noisy_market_bias must be 0 or 1")
        if int(self.hierarchical_mmdit_noisy_gate_mode) not in (0, 1):
            raise ValueError("hierarchical_mmdit_noisy_gate_mode must be 0 or 1")
        if not (0.0 <= float(self.hierarchical_mmdit_noisy_gate_min) <= 1.0):
            raise ValueError("hierarchical_mmdit_noisy_gate_min must be in [0, 1]")
        if float(self.hierarchical_mmdit_noisy_gate_power) <= 0.0:
            raise ValueError("hierarchical_mmdit_noisy_gate_power must be positive")
        if not (0.0 <= float(self.hierarchical_mmdit_stage_promote_scale_init) <= 1.0):
            raise ValueError("hierarchical_mmdit_stage_promote_scale_init must be in [0, 1]")
        if float(self.hierarchical_mmdit_output_init_std) < 0.0:
            raise ValueError("hierarchical_mmdit_output_init_std must be non-negative")
        if not (0.0 < float(self.hierarchical_mmdit_residual_scale_max) <= 1.0):
            raise ValueError("hierarchical_mmdit_residual_scale_max must be in (0, 1]")
        if str(self.hierarchical_mmdit_architecture_version) != "serial_owned_rms_v3":
            raise ValueError(
                "unsupported hierarchical_mmdit_architecture_version: "
                f"{self.hierarchical_mmdit_architecture_version!r}"
            )
        if str(self.final_action_decoder) == "hierarchical_mmdit_action" and not int(self.layer_contract_adapters):
            raise ValueError("hierarchical_mmdit_action requires layer_contract_adapters=1")
        if str(self.final_action_decoder) == "hierarchical_mmdit_action":
            if int(self.hierarchical_mmdit_low_slots) < 5 or int(self.hierarchical_mmdit_low_slots) % 5 != 0:
                raise ValueError(
                    "hierarchical_mmdit_action requires low slots to be a positive multiple of five"
                )
            if int(self.hierarchical_mmdit_refine_steps) < int(self.hierarchical_mmdit_depth):
                raise ValueError(
                    "hierarchical_mmdit_refine_steps must cover every distinct action block"
                )
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
