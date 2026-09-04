"""Configuration lineage for the current staged policy."""

from __future__ import annotations

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
    # Deployment command ABI.  The active mainline normally uses the
    # continuous physical gripper field; CALVIN explicitly selects a
    # two-class command-state readout and never consumes that field as a
    # command source.
    gripper_output_mode: str = "continuous"
    # Historical runs sampled arm_abs/arm_delta independently. New runs can
    # instead sample one native arm trajectory and map it into the redundant
    # [absolute, delta] coordinates used by the policy.
    arm_flow_mode: str = "legacy_independent"
    arm_noise_temporal_rho: float = 0.0
    # Native arm source endpoint. ``ar1`` preserves historical behavior;
    # ``boundary_multiscale`` is state-anchored and mixes full-rank position,
    # velocity, and acceleration operators before the existing field/DCT chart.
    arm_source_mode: str = "ar1"
    arm_source_scale: float = 1.0
    arm_source_innovation_weight: float = 0.50
    arm_source_velocity_weight: float = 0.35
    arm_source_acceleration_weight: float = 0.15

    def validate(self) -> None:
        if (
            min(
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
            )
            <= 0
        ):
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
        if str(self.gripper_output_mode) not in {"continuous", "calvin_binary_command"}:
            raise ValueError(
                "gripper_output_mode must be continuous or calvin_binary_command"
            )
        if str(self.arm_flow_mode) not in {
            "legacy_independent",
            "manifold_native",
            "relative_command_direct",
        }:
            raise ValueError(
                "arm_flow_mode must be legacy_independent, manifold_native, or "
                "relative_command_direct"
            )
        if not 0.0 <= float(self.arm_noise_temporal_rho) < 1.0:
            raise ValueError("arm_noise_temporal_rho must be in [0,1)")
        if str(self.arm_source_mode) not in {"ar1", "boundary_multiscale"}:
            raise ValueError("arm_source_mode must be ar1 or boundary_multiscale")
        if float(self.arm_source_scale) <= 0.0:
            raise ValueError("arm_source_scale must be positive")
        source_weights = (
            float(self.arm_source_innovation_weight),
            float(self.arm_source_velocity_weight),
            float(self.arm_source_acceleration_weight),
        )
        if any(value < 0.0 for value in source_weights) or sum(source_weights) <= 0.0:
            raise ValueError("arm source weights must be non-negative with positive sum")
        if (
            str(self.arm_source_mode) == "boundary_multiscale"
            and float(self.arm_source_innovation_weight) <= 0.0
        ):
            raise ValueError("boundary_multiscale requires positive arm_source_innovation_weight")
        if str(self.arm_source_mode) != "ar1" and str(self.arm_flow_mode) != "manifold_native":
            raise ValueError("non-AR arm source modes require arm_flow_mode=manifold_native")
        if self.first_execution_steps > self.action_horizon:
            raise ValueError("first_execution_steps cannot exceed action_horizon")
        if self.mid_execution_steps > self.action_horizon:
            raise ValueError("mid_execution_steps cannot exceed action_horizon")

    @property
    def gripper_index(self) -> int:
        return (
            self.gripper_dim_index
            if self.gripper_dim_index >= 0
            else self.action_dim + self.gripper_dim_index
        )

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
        if (
            min(
                self.visual_token_dim,
                self.visual_history_length,
                self.num_cameras,
                self.patches_per_camera,
                self.canvas_registers,
                self.future_anchors,
                self.target_future_count,
                self.action_basis_tokens,
                self.future_grid_size,
            )
            <= 0
        ):
            raise ValueError("V38 dimensions must be positive")
        if self.future_anchors > self.target_future_count:
            raise ValueError("future_anchors cannot exceed target_future_count")
        if not 0 <= self.visual_memory_dropout < 1:
            raise ValueError("visual_memory_dropout must be in [0,1)")
        if not 0 <= self.canvas_dropout < 1:
            raise ValueError("canvas_dropout must be in [0,1)")
        if not 0 <= self.role_dropout < 1:
            raise ValueError("role_dropout must be in [0,1)")
        if (
            self.rollout_tail_start_step < 1
            or self.rollout_tail_full_step < self.rollout_tail_start_step
        ):
            raise ValueError("invalid rollout tail binding schedule")
        if (
            min(
                self.controlled_delta_rank,
                self.base_effect_hidden,
                self.latent_action_tokens,
                self.neutral_action_tokens,
            )
            <= 0
        ):
            raise ValueError("controlled residual dynamics dimensions must be positive")
        if not 0 <= self.controlled_delta_dropout < 1:
            raise ValueError("controlled_delta_dropout must be in [0,1)")

    @property
    def future_token_count(self) -> int:
        return (
            int(self.future_anchors)
            * int(self.num_cameras)
            * int(self.future_grid_size)
            * int(self.future_grid_size)
        )

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

    # Flow-DINO JEPA top representation.  The optical-flow core follows the
    # SEA-RAFT design: direct initial flow, correlation-pyramid lookup,
    # iterative refinement, and uncertainty.  V95/V96 retain the cached-DINO
    # implementation.  The opt-in raw-grounding contract adds a full observed
    # RGB pyramid and a 3+3+2 role hierarchy while keeping cached DINO as the
    # semantic lane and future-only teacher.
    flow_jepa_enabled: int = 0
    flow_jepa_grid_size: int = 8
    flow_jepa_feature_dim: int = 96
    flow_jepa_flow_iters: int = 3
    flow_jepa_corr_levels: int = 3
    flow_jepa_corr_radius: int = 2
    flow_jepa_mask_ratio: float = 0.375
    flow_jepa_mask_block_size: int = 2
    flow_jepa_motion_mask_fraction: float = 0.60
    # Optional teacher-side target allocation for the historical absolute
    # prediction path.  This changes only which predicted future patches
    # receive JEPA loss: future teacher features never enter forward
    # conditioning.  Predictive-change mode deliberately disables this option
    # because its online context mask must be identical to its target mask.
    flow_jepa_teacher_balanced_target_mask: int = 0
    flow_jepa_teacher_mask_past_fraction: float = 0.25
    flow_jepa_teacher_mask_change_fraction: float = 0.50
    # Strict future-prediction contract for the raw V96+ path.  One
    # observation-only spatial mask is reused across real horizons, hides the
    # latest RGB/DINO chart before trainable cross-cell mixing during training,
    # and selects the supervised future coordinates.  Deployment keeps the
    # observation complete.  The prediction head emits a change in the frozen
    # teacher chart rather than an absolute future token.  Cached final-layer
    # DINO remains semantic evidence for the world path, so this does not
    # falsely claim that its internal attention was masked before the frozen
    # backbone.
    flow_jepa_predictive_change_contract: int = 0
    flow_jepa_uncertainty_floor: float = 0.03
    flow_jepa_directed_canvas_attention: int = 1
    flow_jepa_late_bottleneck: int = 0
    flow_jepa_dense_depth: int = 2
    flow_jepa_fine_radius: int = 2
    flow_jepa_reader_radius: int = 1
    flow_jepa_reader_heads: int = 2
    flow_jepa_raw_image_enabled: int = 0
    flow_jepa_role_hierarchy: int = 0
    flow_jepa_raw_base_channels: int = 32
    flow_jepa_raw_mid_radius: int = 2
    flow_jepa_raw_high_radius: int = 1
    flow_jepa_raw_reader_radius: int = 3
    flow_jepa_raw_reader_heads: int = 4
    flow_jepa_raw_activation_checkpoint: int = 1
    # V99 anti-collapse contract. Legacy V98 remains constructible with zero
    # for controlled reproduction; the current experiment wrapper enables it.
    flow_jepa_zero_flow_guard: int = 0
    # V100 removes the two remaining visual shortcuts.  Grounding/world blocks
    # own raw evidence; policy blocks read the resulting world canvas, and the
    # final decoder consumes only policy/world products rather than raw visual
    # tokens a second time.  The complementary reader always adds a pooled
    # low-frequency base and a flow-addressed high-frequency residual, so
    # neither lane can win a router by suppressing the other.
    flow_jepa_strict_role_visual_path: int = 0
    flow_jepa_complementary_raw_detail: int = 0
    # The forward-flow reader emits current raw detail in the preceding
    # source-frame coordinate chart.  V101 fuses it with that matching source
    # DINO chart, leaving latest DINO as a separate current-coordinate chart.
    flow_jepa_source_aligned_raw_fusion: int = 0
    flow_jepa_grounding_blocks: int = 3
    flow_jepa_world_blocks: int = 3
    flow_jepa_policy_blocks: int = 2
    flow_jepa_policy_workspace_scale: float = 0.10
    flow_jepa_policy_workspace_fixed_fusion: int = 0
    # V102 separates temporal world organization from spatial observation
    # detail. World blocks may write one residual per anchor/camera but cannot
    # manufacture an xy-specific residual; the exact post-reader high-frequency
    # raw residual is instead read once at the world -> policy boundary.
    flow_jepa_world_anchor_write_only: int = 0
    flow_jepa_late_policy_detail: int = 0
    flow_jepa_late_policy_detail_scale: float = 0.25
    # Post-V102 soft multi-resolution address lattice.  The observation-only
    # compiler keeps several DINO hypotheses and continuous raw candidates;
    # the world/policy query performs the final spatial/camera read.  Zero
    # preserves the exact V102 compressed-detail path.
    flow_jepa_soft_address_lattice: int = 0
    flow_jepa_address_slots: int = 4
    flow_jepa_address_route_dim: int = 32
    flow_jepa_address_query_chunk: int = 4
    # V107 makes ``flow_jepa_raw_reader_heads`` factual on the active soft
    # lattice path.  Every glimpse owns an independent query/posterior and
    # reads a narrow value before the per-glimpse results are concatenated.
    # Zero preserves the single-expectation V106 reader exactly.
    flow_jepa_policy_multi_glimpse_address: int = 0
    # Positive values keep flow as a genuine soft geometric expert. Content
    # may still override it through the posterior, but the model cannot remove
    # the flow contribution by shrinking one learned scalar to zero.
    flow_jepa_address_flow_prior_floor: float = 0.0
    # Post-V103 geometry contract.  Every learned displacement is represented
    # as an in-image source-relative coordinate by a smooth asymmetric chart.
    # This prevents validity masks from becoming an optimization escape hatch
    # while retaining continuous gradients and exact identity at zero flow.
    flow_jepa_bounded_flow_coordinates: int = 0
    # Long-horizon anchors are accumulated in chronological order from an
    # observation-only perceptual history state.  Zero retains the historical
    # parallel horizon queries for checkpoint-compatible reproduction.
    flow_jepa_sequential_horizon_memory: int = 0
    # V105 gives each real future horizon its own soft read over the existing
    # observation-only multi-resolution bank.  The 8x8 W chart supplies
    # queries; continuous raw candidates remain values, and no future teacher
    # enters forward conditioning.  In V105-V107 the result is a small fixed
    # residual into the JEPA prediction head only.
    flow_jepa_horizon_soft_address: int = 0
    flow_jepa_horizon_address_update_scale: float = 0.10
    # Keep the target 8x8 query-cell identity through continuous fine-offset
    # selection.  The implementation chunks target cells to bound memory.
    # The same flag gives the bias-free value projection a variance-preserving
    # initialization instead of the V105/V106 near-silent 1e-3 initialization.
    flow_jepa_horizon_cell_fine_address: int = 0
    # V108 moves the same owned soft-address read to the G3 -> W1 boundary and
    # writes its bounded residual into the existing rollout carrier.  The
    # final JEPA head then consumes the final rollout without rereading the
    # bank.  Zero retains the exact V107 late auxiliary topology.
    flow_jepa_online_horizon_address: int = 0
    # V109 replaces the V108 single G3->W1 value read with a progressive clean
    # selector state.  G1 updates complete-chart hypotheses, G2 rectifies
    # geometry/fine support, and G3 compiles a canonical selector basis.  The
    # first high-resolution value read remains at the existing W->P boundary.
    # Zero preserves the exact V108 topology and checkpoint graph.
    flow_jepa_progressive_grounding_address: int = 0
    # V110 preserves literal current RGB, learned raw detail, DINO semantics,
    # raw-pair appearance and flow geometry as distinct evidence types.  W
    # predicts soft future transport from exact current anchors; P1 retains a
    # local micro-grid and P2 performs the first typed local fusion.  Zero
    # restores the V109 progressive graph exactly.
    flow_jepa_coordinate_typed_raw_detail: int = 0
    flow_jepa_raw_micro_grid: int = 3
    # V111 turns the V110 typed evidence labels into functional ownership.
    # G keeps public scene state separate from semantic/appearance/geometry
    # sidecars, W represents chronological interval innovations, and P uses
    # factorized source/fine addressing followed by typed local operations.
    # Zero preserves the serialized and numerical V110 graph exactly.
    flow_jepa_structured_ownership_bottleneck: int = 0
    # Post-V111 ownership repair.  Keep the G3 public chart explicitly
    # owner-neutral, carry semantic/appearance/geometry/interval selector
    # states through W1-W3, and consume the appearance sidecar in P1's actual
    # joint source/fine posterior before the single precision value read.
    # Zero preserves the serialized and numerical V111 graph exactly.
    flow_jepa_pre_value_owner_routing: int = 0
    flow_jepa_pre_value_owner_update_scale: float = 0.10
    # Functional mainline repair.  Route typed W owner innovations before one
    # hidden-width reconstruction, require the W appearance verifier in P1,
    # keep phase/goal/history distinct per horizon, and fuse typed P2 local
    # operations through a protected policy carrier.  Zero is exact V112.
    flow_jepa_functional_mainline_routing: int = 0
    # Post-V113 utility/precision repair. P1 forms four action-invariant,
    # basis-aware factual glimpses per horizon exactly once; P2 then lets all
    # action basis tokens use that shared factual bank. RGB/detail base and
    # zero-mean 3x3 precision lanes remain separately protected. Zero restores
    # the serialized and numerical V113 graph.
    flow_jepa_utility_precision_mainline: int = 0
    flow_jepa_action_free_world_factual: int = 0
    # V115 repairs the V114 shared read without duplicating it. Four explicitly
    # typed factual queries form one basis-free P1 bank; every P2 action basis
    # then performs its own soft cross-read over those four already-selected
    # facts. Zero preserves the exact V114 query/concatenation path.
    flow_jepa_shared_factual_glimpse_bank: int = 0
    # V115 loss-only Teacher-G and the single online effect state. Future
    # supports are associated to current G3 slots with same-camera soft
    # semantic/geometric transport; fixed cell identity is no longer a
    # supervision assumption.
    flow_jepa_g_aligned_future_effect: int = 0
    flow_jepa_teacher_g_ema_decay: float = 0.995
    # V115 replaces the shallow mean-pooled phase adapter with an ordered
    # four-state goal program replayed from observable state/action history
    # and the completed G3 facts.  Goal, phase and history remain distinct
    # selector operands and no future teacher enters this online machine.
    flow_jepa_stateless_goal_phase_machine: int = 0
    # Explicit topology contract.  The string is serialized so a checkpoint
    # cannot silently reinterpret the third W block as a third P block.
    flow_jepa_top_role_schedule: str = "3-3-2"
    # The third policy stage is a typed plan compiler, not another generic
    # visual/world block.  It writes four provenance-preserving delta lanes
    # for the existing bottom Evidence-MMDiT/CVAE/workspace stack.
    flow_jepa_policy_plan_compiler: int = 0
    # V116 keeps the V115 3-2-3 graph but makes the future-effect boundary
    # fully supervised, gives P2 a structured zero-preserving effect read,
    # separates terminal evidence from action deltas, and uses a four-state
    # non-terminal phase belief.  Zero preserves V115 exactly.
    flow_jepa_supervised_effect_mainline: int = 0
    # V117 replaces the recurrently multiplied phase belief with a three-block
    # stateless intent controller. The four-state attention is audit-only;
    # typed continuous goal/history/progress controls drive W and P.
    flow_jepa_stateless_intent_controller: int = 0
    # The online rollout retains four compact anchors, while the sole W->P
    # effect object owns three near/mid/late window slots.
    flow_jepa_window_effect_bank: int = 0
    flow_jepa_future_slots: int = 4
    # Move the structured effect read into the real P2 block. P3 then compiles
    # already-formed P1/P2 innovations and cannot reopen the effect field.
    flow_jepa_effect_read_in_p2: int = 0
    # Post-V117 architectural capability.  This single flag replaces the
    # scalar-positioned intent heads, repeated-current W targets, fixed P2
    # temporal mass and optional P3 effect lane with the coherent differential
    # intent/effect 3-2-3 graph.  Zero is the exact V117 parent.
    flow_jepa_differential_intent_effect_mainline: int = 0
    # Grounded intent/effect capability.  This is a sibling of the V118
    # differential graph, not another historical layer on top of it.  It
    # retains four real teacher intervals and object slots from G through W
    # and P2, while S is an observable intent organizer without phase/progress
    # forward variables.
    flow_jepa_grounded_intent_effect_mainline: int = 0
    # Capability-named successor to the V119 top.  It reuses the healthy
    # observation-only pre-G/P1 foundation, but replaces the local-hypothesis
    # "objects", pseudo-phase S, weakened W and three-lane P3 with a distinct
    # global-object/plan-recognition/future-dynamics graph.  The historical
    # grounded capability remains bit-for-bit selectable when this is zero.
    flow_jepa_object_intent_dynamics_mainline: int = 0
    # The formal V116 action-flow objective samples more late/noisy times.
    # Uniform remains available for matched structural smoke and ancestry.
    flow_matching_time_distribution: str = "uniform"
    # Resource policy for the sole high-resolution P1 read. The query chunk is
    # chosen so physical_batch * query_chunk stays near this budget; the old
    # fixed chunk remains the minimum for flags-off ancestry.
    flow_jepa_address_query_batch_budget: int = 32
    # Bound the temporary microgrid contraction without launching one complete
    # contraction per 3x3 cell.
    flow_jepa_microgrid_tile: int = 3
    # Keep selector logits, geometry, masking, normalization and diagnostics in
    # FP32 while allowing selected RGB/detail value contractions in model dtype.
    flow_jepa_p1_mixed_precision: int = 0
    # Small physical batches have enough activation headroom and should not pay
    # backward recomputation. Large production batches retain checkpointing.
    flow_jepa_checkpoint_min_batch: int = 4
    # V106 replaces normalize-after-cancellation routing with a
    # zero-preserving variance-floor contract.  The complete numerical
    # contract extends that same bounded-Jacobian rule through learned
    # correlation features, the role-block normalization stack, and the
    # continuous cycle-visibility evidence consumed by address/motion keys.
    # Values themselves remain upper-bounded and are never expanded merely to
    # manufacture a confident route.
    flow_jepa_variance_safe_routing: int = 0
    flow_jepa_complete_numerical_contract: int = 0
    flow_jepa_routing_norm_floor: float = 0.25
    # Learned correlation features have nominal RMS near one.  A 0.10 floor
    # changes that regime by less than one percent while bounding a cancelled
    # feature's inverse-norm derivative.  The floor is expressed in RMS units
    # and converted to the matching width-aware L2 denominator.
    flow_jepa_correlation_rms_floor: float = 0.10
    # Cycle visibility transitions across this fraction of its local
    # source-relative consistency threshold.  Hard visibility remains an
    # audit metric only and never enters online address values.
    flow_jepa_visibility_transition_fraction: float = 0.10
    flow_jepa_horizon_value_max_rms: float = 0.50
    # V106 interval-stage semantics.  ``window_offsets`` remain the four
    # compact query labels, while boundaries/supports describe the real
    # teacher frames.  The online organizer never receives these targets.
    flow_jepa_interval_stage_delta: int = 0
    flow_jepa_interval_boundaries: tuple[int, ...] = ()
    flow_jepa_interval_support_offsets: tuple[int, ...] = ()
    flow_jepa_interval_stage_update_scale: float = 0.10
    # In addition to the coarse spatial W write, expose the bounded signed
    # interval increment as a provenance-preserving W->P typed candidate.
    flow_jepa_interval_stage_typed_value: int = 0
    # Typed 3-3-2 delta bridges.  These route real per-block residual values
    # at role boundaries while keeping the cumulative carrier, protected raw
    # detail, clean semantic seed, and noisy action state outside softmax.
    role_attnres_enabled: int = 0
    role_attnres_key_dim: int = 32
    role_attnres_ground_to_world: int = 0
    role_attnres_world_to_policy: int = 0
    role_attnres_policy_to_mmdit: int = 0
    role_attnres_ground_to_world_scale: float = 0.10
    role_attnres_world_to_policy_scale: float = 0.10
    role_attnres_policy_to_mmdit_scale: float = 0.25
    # Post-V103 residual stability contract.  Role blocks and typed AttnRes
    # values use smooth RMS compression in the normalized hidden chart.  The
    # carrier remains outside every route softmax and no gradient is detached.
    role_residual_amplitude_contract: int = 0
    role_residual_max_update_rms: float = 0.50
    role_attnres_max_value_rms: float = 1.00
    # Contract the actual gated proposal at the legal write boundary.  Zero
    # retains V106's gate-after-contract arithmetic for matched ablation.
    role_residual_contract_after_gate: int = 0
    # Trajectory workspace tokens are laid out as [time, basis].  Pooling basis
    # tokens inside each time step preserves event timing; generic interpolation
    # across the flattened T*basis axis is retained only for old checkpoints.
    flow_jepa_policy_workspace_horizon_pool: int = 0
    flow_jepa_history_offsets: tuple[int, ...] = (-8, -4, 0)
    # Window anchors remain inside the explicitly modelled action chunk.  The
    # separate stage horizon is allowed to extend beyond it because its target
    # is a single global representation delta rather than a deterministic
    # patch-wise transition.  Empty/zero values retain a shape-safe fallback
    # for small direct-construction tests and historical callers; production
    # entry points always write the real dataset offsets into the config.
    flow_jepa_window_offsets: tuple[int, ...] = ()
    flow_jepa_stage_offset: int = 0
    flow_jepa_stage_tokens: int = 1

    # Clean conditioning memory.  The action path preserves recent executed
    # actions verbatim and resamples the older prefix into a few summary tokens.
    # Relative-time encodings are derived from the real dataset offsets, so
    # changing history support does not create a new learned position table.
    action_history_enabled: int = 0
    executed_action_offsets: tuple[int, ...] = ()
    action_history_recent_tokens: int = 4
    action_history_summary_tokens: int = 3
    action_history_condition_dropout: float = 0.0
    action_history_condition_exact_null: int = 0
    # Historical policies trained the future proposal only through its
    # auxiliary action-regression head. Zero keeps this causal, deployment-time
    # condition attached to the final action loss through ordinary autograd.
    action_history_proposal_detach: int = 1

    # Frozen language embeddings are resampled by a small trainable Perceiver
    # into Goal Tokens.  Goal and action memory later share a condition mixer,
    # but retain private stems and role embeddings so they are not identified.
    goal_conditioning_enabled: int = 0
    goal_token_count: int = 4
    goal_language_dim: int = 768
    goal_language_max_tokens: int = 32
    goal_resampler_depth: int = 2
    goal_condition_dropout: float = 0.0
    goal_condition_exact_null: int = 0
    # Stateless long-horizon phase belief.  It has no recurrent deployment
    # state and is used only to perturb world/address selector queries.
    stateless_phase_enabled: int = 0
    stateless_phase_count: int = 4
    stateless_phase_query_scale: float = 0.10

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
    # Keep the native evidence execution path end-to-end differentiable by
    # default. Legacy ablations may still opt into a causal stop-gradient.
    latent_cvae_transition_detach: int = 0
    latent_cvae_context_memory: int = 0
    latent_cvae_visual_memory: int = 0
    # Detach is an explicit ablation; the main evidence path is end-to-end.
    latent_cvae_layer_detach: int = 0
    latent_cvae_layer_grad_scale: float = 1.0
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
    # CR0 (item 14.2): eval-time z zero/shuffle intervention probes on the legacy
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
    # tokens through the native-time MMDiT path. The flow state x_t is part of
    # the action-token stream; it is not exposed as a separately controlled
    # noisy writer.
    latent_cvae_mmdit_decoder: int = 0
    latent_cvae_mmdit_depth: int = 3
    latent_cvae_mmdit_cond_update: int = 0
    latent_cvae_mmdit_noisy_causal: int = 1
    # Legacy compatibility fields for older decoders. The native-time MMDiT
    # path does not use a separate noisy reader or source gate.
    latent_cvae_mmdit_noisy_logit_gate: int = 0
    # The host MMDiT residual gate remains the only update-scale owner.
    latent_cvae_mmdit_residual_scale_max: float = 0.25
    # Deprecated V91 source-router bound. Kept in the config surface so old
    # launch scripts remain readable; the evidence-owned correction path does
    # not use a two-source route.
    latent_cvae_mmdit_source_route_delta_max: float = 1.0
    # Retained for checkpoint and CLI compatibility with the previous V91
    # decoder. The native-time path no longer constructs a noisy correction
    # budget or a source controller from these values.
    latent_cvae_mmdit_noisy_correction_min: float = 0.05
    latent_cvae_mmdit_noisy_correction_max: float = 0.75
    latent_cvae_mmdit_noisy_correction_power: float = 1.5
    latent_cvae_mmdit_noisy_correction_logit_delta: float = 1.0
    latent_cvae_mmdit_controller_modulation_scale: float = 0.25
    # Evaluation-only source ablation controls. Defaults preserve the normal
    # path and are never learned by the model.
    latent_cvae_mmdit_evidence_scale: float = 1.0
    latent_cvae_mmdit_noisy_scale: float = 1.0
    # V92: native-time execution plane. The controller gates the host
    # residual writers before the ordered contraction bank; the bank closes
    # the same ordered directions continuously and the controller selects
    # capacity/repetition.
    latent_cvae_mmdit_operator_capacity: int = 0
    latent_cvae_mmdit_operator_rank: int = 32
    # Rank-level ordered capacity is the differentiable training contract.
    # Hardware-sized grouping can still be selected explicitly for deployment,
    # but the default must not hide small changes such as rank 32 -> 29.
    latent_cvae_mmdit_operator_groups: int = 32
    latent_cvae_mmdit_operator_depth_logit_init: float = 4.0
    latent_cvae_mmdit_execution_controller: int = 0
    # Let the native execution plane choose any still-uncommitted host block or
    # terminal identity. Training/default evaluation use the attached soft
    # action chart; hard execution remains an explicit ablation.
    latent_cvae_mmdit_dynamic_block_route: int = 0
    latent_cvae_mmdit_control_tokens: int = 8
    latent_cvae_mmdit_controller_depth: int = 2
    latent_cvae_mmdit_controller_heads: int = 8
    latent_cvae_mmdit_controller_ffn_expansion: float = 2.0
    latent_cvae_mmdit_max_dwell: int = 2
    latent_cvae_mmdit_dwell_mode: str = "fixed"
    latent_cvae_mmdit_execution_soft_temperature: float = 1.0
    # Stopping is a real identity candidate.  Its smaller prior expresses that
    # a premature exit is more consequential than applying one normal block;
    # task gradients may still overcome that prior when stopping is correct.
    latent_cvae_mmdit_identity_candidate: int = 1
    latent_cvae_mmdit_terminal_prior_weight: float = 0.25
    latent_cvae_mmdit_execution_eval_policy: str = "soft"
    latent_cvae_mmdit_execution_warmup_steps: int = 200
    latent_cvae_mmdit_execution_transition_steps: int = 1000
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
    # ``depth`` counts the explicit refinement ownership blocks;
    # ``refine_steps`` is the maximum recurrent compute budget.  Operator
    # stages are a separate semantic repertoire: the default three blocks each
    # select locally between two of six stage-owned contraction paths.
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
    # V82: every refinement block owns a full-rank MMDiT function. External
    # stage banks continuously contract its normalized branch updates along
    # ordered orthonormal paths. Maximum depth is the original operation.
    hierarchical_mmdit_operator_stages: int = 6
    hierarchical_mmdit_operator_rank: int = 32
    hierarchical_mmdit_operator_groups: int = 32
    hierarchical_mmdit_operator_depth_logit_init: float = 2.0
    hierarchical_mmdit_exit_logit_init: float = -4.0
    hierarchical_mmdit_operator_contraction_warmup_steps: int = 200
    hierarchical_mmdit_operator_contraction_transition_steps: int = 1500
    # V85: one recurrent, multi-token control plane for evidence retrieval,
    # operator selection, stage promotion, and candidate operation policy. The
    # recurrent slots are exchangeable latent state, not eight fixed roles.
    hierarchical_mmdit_unified_controller: int = 0
    hierarchical_mmdit_control_tokens: int = 8
    hierarchical_mmdit_controller_depth: int = 2
    hierarchical_mmdit_controller_heads: int = 8
    hierarchical_mmdit_controller_ffn_expansion: float = 2.0
    # V87: run the hierarchical MMDiT state in the complete orthonormal DCT
    # coefficient chart. Refinement changes a soft bandwidth, never the state
    # dimensionality or the flow bridge.
    hierarchical_mmdit_spectral_state: int = 0
    hierarchical_mmdit_spectral_arm_start_fraction: float = 0.16
    hierarchical_mmdit_spectral_gripper_start_fraction: float = 0.33
    hierarchical_mmdit_spectral_temperature: float = 1.5
    hierarchical_mmdit_spectral_schedule_power: float = 1.0
    hierarchical_mmdit_spectral_controller_shift_limit: float = 2.0
    hierarchical_mmdit_spectral_competition_loss_weight: float = 0.0
    hierarchical_mmdit_spectral_competition_warmup_steps: int = 200
    hierarchical_mmdit_operation_candidate_probes: int = 0
    # V88: value-supervised monotonic dwell. Fixed and shadow retain the exact
    # V87 execution path; learned changes execution only after the warm-up.
    hierarchical_mmdit_operation_value_warmup_steps: int = 200
    hierarchical_mmdit_dwell_mode: str = "fixed"
    # V89: separate the central control policy from its typed actuators.  The
    # Both contracts now route operations and nested capacity only. The old
    # name remains load-compatible, but controller update-keep logits are not
    # consumed; residual amplitude stays with each block's host LayerScale.
    hierarchical_mmdit_execution_contract: str = "legacy_stage_keep"
    # Training schedule and deployment routing are orthogonal controls. Random
    # dwell is a legacy schedule option; learned operation selection is driven
    # by the unified controller's legal candidate policy.
    hierarchical_mmdit_schedule_mode: str = "fixed"
    hierarchical_mmdit_random_prefix_probability: float = 0.0
    hierarchical_mmdit_exhaustion_mode: str = "off"
    hierarchical_mmdit_action_response_thresholds: tuple[float, float, float] = (0.0, 0.0, 0.0)
    hierarchical_mmdit_stage_pressure_thresholds: tuple[float, float, float] = (0.0, 0.0, 0.0)
    hierarchical_mmdit_action_response_floor: float = 0.05
    hierarchical_mmdit_exhaustion_confirm_steps: int = 2
    # V77 host-gate profile. Contraction and controller keeps act on the
    # completed gated operation; neither may replace the learned host gate.
    hierarchical_mmdit_residual_scale_init: float = 0.05
    hierarchical_mmdit_residual_scale_max: float = 0.20
    hierarchical_mmdit_architecture_version: str = "post_gate_contraction_sidecar_v12_value_dwell"
    # CR7 fallback (item 23): dedicated restricted contract for event/motion
    # subheads (never velocity).  0 = action-only output (mainline first try).
    hierarchical_mmdit_output_contract: int = 0
    # Compatibility switch retained for v76a metadata/checkpoints.  The clean
    # serial decoder no longer has a condition-group market, so this flag is a
    # no-op there; legacy MMDiT paths keep their historical behavior.
    hierarchical_mmdit_noisy_market_bias: int = 0
    # The post-gate sidecar requires 0: noisy evidence is not amplitude-gated
    # by a second schedule outside the V77 host block.
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

    @property
    def flow_jepa_effective_window_offsets(self) -> tuple[int, ...]:
        explicit = tuple(int(value) for value in self.flow_jepa_window_offsets)
        if explicit:
            return explicit
        anchors = int(self.future_anchors)
        horizon = int(self.action_horizon)
        return tuple(
            max(1, int(round((index + 1) * horizon / float(anchors))))
            for index in range(anchors)
        )

    @property
    def flow_jepa_effective_stage_offset(self) -> int:
        explicit = int(self.flow_jepa_stage_offset)
        if explicit > 0:
            return explicit
        return int(self.flow_jepa_effective_window_offsets[-1]) + 1

    @property
    def flow_jepa_effective_interval_boundaries(self) -> tuple[int, ...]:
        explicit = tuple(int(value) for value in self.flow_jepa_interval_boundaries)
        if explicit:
            return explicit
        # V106 production defaults.  They are returned only as a shape-safe
        # fallback; the formal validator requires explicit serialization.
        return (4, 8, 16, 32, 48)

    @property
    def flow_jepa_effective_interval_support_offsets(self) -> tuple[int, ...]:
        explicit = tuple(
            int(value) for value in self.flow_jepa_interval_support_offsets
        )
        if explicit:
            return explicit
        return tuple(range(4, 49, 4))

    @property
    def flow_jepa_interval_windows(self) -> tuple[tuple[int, int], ...]:
        boundaries = self.flow_jepa_effective_interval_boundaries
        return tuple(
            (int(boundaries[index]), int(boundaries[index + 1]))
            for index in range(len(boundaries) - 1)
        )

    @property
    def flow_jepa_target_offsets(self) -> tuple[int, ...]:
        """Future teacher order for the active representation contract.

        V95 uses local windows followed by one global stage target.  The
        late-bottleneck contract has no separate stage object: every horizon,
        including the far horizon, remains a spatial future-evidence chart.
        """

        if int(self.flow_jepa_interval_stage_delta):
            return self.flow_jepa_effective_interval_support_offsets
        windows = self.flow_jepa_effective_window_offsets
        if int(self.flow_jepa_late_bottleneck):
            return windows
        return (*windows, int(self.flow_jepa_effective_stage_offset))

    @property
    def flow_jepa_action_offsets(self) -> tuple[int, ...]:
        """Prefix of world horizons that partitions the deploy action chunk."""

        offsets = tuple(
            value
            for value in self.flow_jepa_effective_window_offsets
            if int(value) <= int(self.action_horizon)
        )
        if not offsets or int(offsets[-1]) != int(self.action_horizon):
            raise ValueError(
                "Flow-DINO horizons must contain action_horizon as the final action anchor"
            )
        return offsets

    @property
    def effective_executed_action_offsets(self) -> tuple[int, ...]:
        explicit = tuple(int(value) for value in self.executed_action_offsets)
        if explicit:
            return explicit
        return tuple(range(-int(self.executed_history_length), 0))

    @property
    def action_history_token_count(self) -> int:
        if not int(self.action_history_enabled):
            return int(self.executed_history_length)
        recent = min(
            int(self.action_history_recent_tokens),
            int(self.executed_history_length),
        )
        summaries = (
            int(self.action_history_summary_tokens)
            if int(self.executed_history_length) > recent
            else 0
        )
        return recent + summaries

    def validate(self) -> None:
        super().validate()
        if str(self.controlled_base_mode) not in {"learned", "fixed_zero"}:
            raise ValueError("controlled_base_mode must be learned or fixed_zero")
        if int(self.flow_jepa_enabled) not in (0, 1):
            raise ValueError("flow_jepa_enabled must be 0 or 1")
        if int(self.action_history_enabled) not in (0, 1):
            raise ValueError("action_history_enabled must be 0 or 1")
        if int(self.goal_conditioning_enabled) not in (0, 1):
            raise ValueError("goal_conditioning_enabled must be 0 or 1")
        for name in (
            "action_history_condition_exact_null",
            "action_history_proposal_detach",
            "goal_condition_exact_null",
            "stateless_phase_enabled",
        ):
            if int(getattr(self, name)) not in (0, 1):
                raise ValueError(f"{name} must be 0 or 1")
        if int(self.action_history_condition_exact_null) and not int(
            self.action_history_enabled
        ):
            raise ValueError(
                "exact action-history null semantics require action_history_enabled"
            )
        if int(self.goal_condition_exact_null) and not int(
            self.goal_conditioning_enabled
        ):
            raise ValueError("exact goal null semantics require goal conditioning")
        if int(self.stateless_phase_count) < 2:
            raise ValueError("stateless_phase_count must be at least two")
        if not 0.0 < float(self.stateless_phase_query_scale) <= 1.0:
            raise ValueError("stateless_phase_query_scale must be in (0,1]")
        if int(self.stateless_phase_enabled) and not (
            int(self.goal_conditioning_enabled)
            or int(self.action_history_enabled)
        ):
            raise ValueError(
                "stateless phase belief requires goal or action-history conditioning"
            )
        if int(self.stateless_phase_enabled) and not (
            int(self.flow_jepa_strict_role_visual_path)
            and int(self.flow_jepa_late_policy_detail)
        ):
            raise ValueError(
                "stateless phase belief requires the strict late-detail selector path"
            )
        executed_offsets = self.effective_executed_action_offsets
        if len(executed_offsets) != int(self.executed_history_length):
            raise ValueError(
                "executed_action_offsets must match executed_history_length"
            )
        if tuple(sorted(set(executed_offsets))) != executed_offsets or max(executed_offsets) >= 0:
            raise ValueError(
                "executed_action_offsets must be strictly increasing past offsets"
            )
        if int(self.action_history_enabled):
            if not 1 <= int(self.action_history_recent_tokens) <= int(
                self.executed_history_length
            ):
                raise ValueError(
                    "action_history_recent_tokens must be in [1,executed_history_length]"
                )
            if int(self.action_history_summary_tokens) < 1:
                raise ValueError("action_history_summary_tokens must be positive")
        if int(self.goal_conditioning_enabled):
            if min(
                int(self.goal_token_count),
                int(self.goal_language_dim),
                int(self.goal_language_max_tokens),
                int(self.goal_resampler_depth),
            ) < 1:
                raise ValueError("goal conditioning dimensions must be positive")
        if not 0.0 <= float(self.action_history_condition_dropout) < 1.0:
            raise ValueError("action_history_condition_dropout must be in [0,1)")
        if not 0.0 <= float(self.goal_condition_dropout) < 1.0:
            raise ValueError("goal_condition_dropout must be in [0,1)")
        if int(self.flow_jepa_directed_canvas_attention) not in (0, 1):
            raise ValueError("flow_jepa_directed_canvas_attention must be 0 or 1")
        if int(self.flow_jepa_late_bottleneck) not in (0, 1):
            raise ValueError("flow_jepa_late_bottleneck must be 0 or 1")
        if int(self.flow_jepa_raw_image_enabled) not in (0, 1):
            raise ValueError("flow_jepa_raw_image_enabled must be 0 or 1")
        if int(self.flow_jepa_raw_image_enabled) and not int(self.flow_jepa_enabled):
            raise ValueError("raw-image grounding requires Flow-DINO JEPA")
        if int(self.flow_jepa_raw_image_enabled) and not int(
            self.flow_jepa_late_bottleneck
        ):
            raise ValueError(
                "raw-image grounding requires the late-bottleneck evidence layout"
            )
        if int(self.flow_jepa_raw_activation_checkpoint) not in (0, 1):
            raise ValueError("flow_jepa_raw_activation_checkpoint must be 0 or 1")
        if int(self.flow_jepa_zero_flow_guard) not in (0, 1):
            raise ValueError("flow_jepa_zero_flow_guard must be 0 or 1")
        if int(self.flow_jepa_zero_flow_guard) and not int(
            self.flow_jepa_raw_image_enabled
        ):
            raise ValueError("flow_jepa_zero_flow_guard requires raw-image Flow-JEPA")
        if int(self.flow_jepa_strict_role_visual_path) not in (0, 1):
            raise ValueError("flow_jepa_strict_role_visual_path must be 0 or 1")
        if int(self.flow_jepa_complementary_raw_detail) not in (0, 1):
            raise ValueError("flow_jepa_complementary_raw_detail must be 0 or 1")
        if int(self.flow_jepa_source_aligned_raw_fusion) not in (0, 1):
            raise ValueError("flow_jepa_source_aligned_raw_fusion must be 0 or 1")
        if int(self.flow_jepa_source_aligned_raw_fusion) and not int(
            self.flow_jepa_complementary_raw_detail
        ):
            raise ValueError(
                "source-aligned raw fusion requires complementary raw detail"
            )
        if int(self.flow_jepa_strict_role_visual_path) and not int(
            self.flow_jepa_role_hierarchy
        ):
            raise ValueError(
                "flow_jepa_strict_role_visual_path requires the role hierarchy"
            )
        if int(self.flow_jepa_complementary_raw_detail) and not (
            int(self.flow_jepa_raw_image_enabled)
            and int(self.flow_jepa_zero_flow_guard)
        ):
            raise ValueError(
                "complementary raw detail requires raw-image Flow-JEPA and the zero-flow guard"
            )
        if int(self.flow_jepa_role_hierarchy) not in (0, 1):
            raise ValueError("flow_jepa_role_hierarchy must be 0 or 1")
        if bool(int(self.flow_jepa_raw_image_enabled)) != bool(
            int(self.flow_jepa_role_hierarchy)
        ):
            raise ValueError(
                "raw-image grounding and the 3-group DiT hierarchy must be enabled together"
            )
        if min(
            int(self.flow_jepa_grid_size),
            int(self.flow_jepa_feature_dim),
            int(self.flow_jepa_flow_iters),
            int(self.flow_jepa_corr_levels),
            int(self.flow_jepa_corr_radius),
            int(self.flow_jepa_mask_block_size),
            int(self.flow_jepa_dense_depth),
            int(self.flow_jepa_fine_radius),
            int(self.flow_jepa_reader_radius),
            int(self.flow_jepa_reader_heads),
            int(self.flow_jepa_raw_base_channels),
            int(self.flow_jepa_raw_mid_radius),
            int(self.flow_jepa_raw_high_radius),
            int(self.flow_jepa_raw_reader_radius),
            int(self.flow_jepa_raw_reader_heads),
        ) < 1:
            raise ValueError("Flow-DINO JEPA dimensions and iteration counts must be positive")
        if int(self.flow_jepa_feature_dim) % 8:
            raise ValueError("flow_jepa_feature_dim must be divisible by 8")
        if int(self.flow_jepa_raw_image_enabled) and (
            int(self.hidden_size) % int(self.flow_jepa_raw_reader_heads)
        ):
            raise ValueError("hidden_size must be divisible by flow_jepa_raw_reader_heads")
        if not 0.0 <= float(self.flow_jepa_mask_ratio) < 1.0:
            raise ValueError("flow_jepa_mask_ratio must be in [0,1)")
        if not 0.0 <= float(self.flow_jepa_motion_mask_fraction) <= 1.0:
            raise ValueError("flow_jepa_motion_mask_fraction must be in [0,1]")
        if int(self.flow_jepa_teacher_balanced_target_mask) not in (0, 1):
            raise ValueError("flow_jepa_teacher_balanced_target_mask must be 0 or 1")
        if int(self.flow_jepa_predictive_change_contract) not in (0, 1):
            raise ValueError("flow_jepa_predictive_change_contract must be 0 or 1")
        if int(self.flow_jepa_predictive_change_contract) and not (
            int(self.flow_jepa_raw_image_enabled)
            and int(self.flow_jepa_late_bottleneck)
            and int(self.flow_jepa_zero_flow_guard)
            and not int(self.flow_jepa_teacher_balanced_target_mask)
            and float(self.flow_jepa_address_flow_prior_floor) > 0.0
        ):
            raise ValueError(
                "predictive-change contract requires raw late-bottleneck Flow-JEPA, "
                "the zero-flow guard, one shared online context/target mask "
                "(teacher-balanced target selection disabled), and a positive "
                "soft-address flow-prior floor"
            )
        teacher_past = float(self.flow_jepa_teacher_mask_past_fraction)
        teacher_change = float(self.flow_jepa_teacher_mask_change_fraction)
        if teacher_past < 0.0 or teacher_change < 0.0 or teacher_past + teacher_change > 1.0:
            raise ValueError(
                "teacher target-mask fractions must be non-negative and sum to at most 1"
            )
        if float(self.flow_jepa_uncertainty_floor) <= 0.0:
            raise ValueError("flow_jepa_uncertainty_floor must be positive")
        if not 0.0 < float(self.flow_jepa_policy_workspace_scale) <= 1.0:
            raise ValueError("flow_jepa_policy_workspace_scale must be in (0,1]")
        if int(self.flow_jepa_policy_workspace_fixed_fusion) not in (0, 1):
            raise ValueError("flow_jepa_policy_workspace_fixed_fusion must be 0 or 1")
        if int(self.flow_jepa_policy_workspace_fixed_fusion) and not int(
            self.flow_jepa_strict_role_visual_path
        ):
            raise ValueError("fixed policy-workspace fusion requires the strict role visual path")
        if int(self.flow_jepa_world_anchor_write_only) not in (0, 1):
            raise ValueError("flow_jepa_world_anchor_write_only must be 0 or 1")
        if int(self.flow_jepa_late_policy_detail) not in (0, 1):
            raise ValueError("flow_jepa_late_policy_detail must be 0 or 1")
        if int(self.flow_jepa_soft_address_lattice) not in (0, 1):
            raise ValueError("flow_jepa_soft_address_lattice must be 0 or 1")
        for name in (
            "flow_jepa_bounded_flow_coordinates",
            "flow_jepa_sequential_horizon_memory",
            "flow_jepa_horizon_soft_address",
            "flow_jepa_variance_safe_routing",
            "flow_jepa_complete_numerical_contract",
            "flow_jepa_interval_stage_delta",
            "flow_jepa_policy_multi_glimpse_address",
            "flow_jepa_horizon_cell_fine_address",
            "flow_jepa_online_horizon_address",
            "flow_jepa_progressive_grounding_address",
            "flow_jepa_coordinate_typed_raw_detail",
            "flow_jepa_structured_ownership_bottleneck",
            "flow_jepa_pre_value_owner_routing",
            "flow_jepa_functional_mainline_routing",
            "flow_jepa_utility_precision_mainline",
            "flow_jepa_action_free_world_factual",
            "flow_jepa_shared_factual_glimpse_bank",
            "flow_jepa_g_aligned_future_effect",
            "flow_jepa_stateless_goal_phase_machine",
            "flow_jepa_policy_plan_compiler",
            "flow_jepa_supervised_effect_mainline",
            "flow_jepa_object_intent_dynamics_mainline",
            "flow_jepa_p1_mixed_precision",
            "flow_jepa_interval_stage_typed_value",
            "role_residual_amplitude_contract",
            "role_residual_contract_after_gate",
        ):
            if int(getattr(self, name)) not in (0, 1):
                raise ValueError(f"{name} must be 0 or 1")
        if int(self.flow_jepa_bounded_flow_coordinates) and not int(
            self.flow_jepa_raw_image_enabled
        ):
            raise ValueError(
                "bounded flow coordinates require raw-image Flow-JEPA"
            )
        if int(self.flow_jepa_sequential_horizon_memory) and not (
            int(self.flow_jepa_predictive_change_contract)
            and int(self.flow_jepa_raw_image_enabled)
        ):
            raise ValueError(
                "sequential horizon memory requires predictive raw Flow-JEPA"
            )
        if int(self.flow_jepa_horizon_soft_address) and not (
            int(self.flow_jepa_sequential_horizon_memory)
            and int(self.flow_jepa_soft_address_lattice)
            and int(self.flow_jepa_predictive_change_contract)
        ):
            raise ValueError(
                "horizon soft address requires sequential predictive JEPA and "
                "the observation-only soft address lattice"
            )
        if int(self.flow_jepa_policy_multi_glimpse_address) and not (
            int(self.flow_jepa_soft_address_lattice)
            and int(self.flow_jepa_late_policy_detail)
        ):
            raise ValueError(
                "multi-glimpse policy addressing requires the late soft-address reader"
            )
        if int(self.flow_jepa_horizon_cell_fine_address) and not int(
            self.flow_jepa_horizon_soft_address
        ):
            raise ValueError(
                "cell-specific horizon fine addressing requires horizon soft address"
            )
        if int(self.flow_jepa_online_horizon_address) and not (
            int(self.flow_jepa_horizon_soft_address)
            and int(self.flow_jepa_horizon_cell_fine_address)
            and int(self.flow_jepa_role_hierarchy)
            and int(self.flow_jepa_strict_role_visual_path)
            and int(self.flow_jepa_late_policy_detail)
            and int(self.flow_jepa_late_bottleneck)
        ):
            raise ValueError(
                "online horizon address requires the cell-specific horizon reader, "
                "strict role hierarchy, late observation bank, and late bottleneck"
            )
        if int(self.flow_jepa_progressive_grounding_address) and not (
            int(self.flow_jepa_online_horizon_address)
            and int(self.flow_jepa_soft_address_lattice)
            and int(self.flow_jepa_horizon_cell_fine_address)
            and int(self.flow_jepa_role_hierarchy)
            and int(self.flow_jepa_strict_role_visual_path)
            and int(self.flow_jepa_late_policy_detail)
            and int(self.flow_jepa_grounding_blocks) == 3
        ):
            raise ValueError(
                "progressive grounding address requires the complete V108 "
                "soft-lattice path and exactly three grounding blocks"
            )
        if int(self.flow_jepa_coordinate_typed_raw_detail) and not (
            int(self.flow_jepa_progressive_grounding_address)
            and int(self.flow_jepa_complete_numerical_contract)
            and int(self.flow_jepa_policy_multi_glimpse_address)
        ):
            raise ValueError(
                "coordinate-typed raw detail requires the complete V109 "
                "progressive graph, the finite-gain numerical contract, and "
                "multi-glimpse policy addressing"
            )
        micro_grid = int(self.flow_jepa_raw_micro_grid)
        if int(self.flow_jepa_coordinate_typed_raw_detail) and (
            micro_grid < 3 or micro_grid % 2 == 0
        ):
            raise ValueError("flow_jepa_raw_micro_grid must be an odd integer >= 3")
        if int(self.flow_jepa_structured_ownership_bottleneck) and not (
            int(self.flow_jepa_coordinate_typed_raw_detail)
            and int(self.flow_jepa_interval_stage_delta)
            and int(self.flow_jepa_interval_stage_typed_value)
            and int(self.flow_jepa_sequential_horizon_memory)
        ):
            raise ValueError(
                "structured ownership requires the complete V110 typed path, "
                "chronological horizon memory, and typed interval-delta supervision"
            )
        v115_role_schedule = (
            str(self.flow_jepa_top_role_schedule).strip() == "3-2-3"
        )
        if str(self.flow_jepa_top_role_schedule).strip() not in {
            "3-3-2",
            "3-2-3",
        }:
            raise ValueError(
                "flow_jepa_top_role_schedule must be 3-3-2 or 3-2-3"
            )
        if int(self.flow_jepa_pre_value_owner_routing) and not (
            int(self.flow_jepa_structured_ownership_bottleneck)
            and (
                int(self.flow_jepa_world_blocks) == 3
                or (
                    v115_role_schedule
                    and int(self.flow_jepa_world_blocks) == 2
                )
            )
        ):
            raise ValueError(
                "pre-value owner routing requires the complete V111 ownership "
                "graph and either the ancestral three-W or V115 two-W schedule"
            )
        if int(self.flow_jepa_functional_mainline_routing) and not (
            int(self.flow_jepa_pre_value_owner_routing)
            and int(self.stateless_phase_enabled)
            and int(self.goal_conditioning_enabled)
            and int(self.action_history_enabled)
            and (
                int(self.flow_jepa_world_blocks) == 3
                or (
                    v115_role_schedule
                    and int(self.flow_jepa_world_blocks) == 2
                )
            )
            and int(self.future_anchors) == int(self.stateless_phase_count)
        ):
            raise ValueError(
                "functional mainline routing requires the complete V112 graph, "
                "typed goal/history phase conditioning, a legal W schedule, and one "
                "stateless phase query per future anchor"
            )
        if int(self.flow_jepa_utility_precision_mainline) and not (
            int(self.flow_jepa_functional_mainline_routing)
            and int(self.flow_jepa_action_free_world_factual)
            and int(self.action_basis_tokens) > 1
            and int(self.flow_jepa_raw_reader_heads) > 1
        ):
            raise ValueError(
                "utility precision mainline requires the complete V113 graph, "
                "an action-free factual W path, multiple action basis tokens, "
                "and multiple factual glimpses"
            )
        if int(self.flow_jepa_action_free_world_factual) and not int(
            self.flow_jepa_functional_mainline_routing
        ):
            raise ValueError(
                "action-free factual W requires the functional V113 mainline"
            )
        if int(self.flow_jepa_shared_factual_glimpse_bank) and not (
            int(self.flow_jepa_utility_precision_mainline)
            and int(self.flow_jepa_raw_reader_heads) == 4
        ):
            raise ValueError(
                "shared factual glimpse bank requires the V114 utility/precision "
                "mainline and exactly four factual glimpses"
            )
        if int(self.flow_jepa_g_aligned_future_effect) and not (
            int(self.flow_jepa_shared_factual_glimpse_bank)
            and int(self.flow_jepa_progressive_grounding_address)
            and int(self.flow_jepa_coordinate_typed_raw_detail)
            and int(self.flow_jepa_interval_stage_delta)
        ):
            raise ValueError(
                "G-aligned future effect requires the V115 factual glimpse "
                "bank, completed G3 typed address state, and interval teacher"
            )
        if not 0.0 <= float(self.flow_jepa_teacher_g_ema_decay) < 1.0:
            raise ValueError(
                "flow_jepa_teacher_g_ema_decay must be in [0,1)"
            )
        if int(self.flow_jepa_stateless_goal_phase_machine) and not (
            int(self.flow_jepa_g_aligned_future_effect)
            and int(self.flow_jepa_functional_mainline_routing)
            and int(self.goal_conditioning_enabled)
            and int(self.stateless_phase_count) == 4
            and int(self.future_anchors) == 4
        ):
            raise ValueError(
                "stateless goal-phase machine requires the V115 G-aligned "
                "functional path, precomputed language conditioning, four "
                "program states, and four intervals"
            )
        if v115_role_schedule:
            if not (
                int(self.flow_jepa_grounding_blocks) == 3
                and int(self.flow_jepa_world_blocks) == 2
                and int(self.flow_jepa_policy_blocks) == 3
                and int(self.depth) == 8
                and int(self.flow_jepa_stateless_goal_phase_machine)
                and int(self.flow_jepa_policy_plan_compiler)
            ):
                raise ValueError(
                    "the 3-2-3 schedule requires depth=8, G/W/P=3/2/3, "
                    "the stateless goal-phase machine, and policy plan compiler"
                )
        if int(self.flow_jepa_policy_plan_compiler) and not (
            v115_role_schedule
            and int(self.flow_jepa_g_aligned_future_effect)
            and int(self.flow_jepa_shared_factual_glimpse_bank)
        ):
            raise ValueError(
                "policy plan compiler requires the complete V115 3-2-3 "
                "future-effect and shared-factual path"
            )
        if int(self.flow_jepa_supervised_effect_mainline) and not (
            int(self.flow_jepa_policy_plan_compiler)
            and int(self.flow_jepa_g_aligned_future_effect)
            and int(self.flow_jepa_stateless_goal_phase_machine)
            and v115_role_schedule
        ):
            raise ValueError(
                "supervised effect mainline requires the complete V115 "
                "G3/W2/P3 graph"
            )
        if int(self.flow_jepa_stateless_intent_controller) and not (
            int(self.flow_jepa_supervised_effect_mainline)
            and int(self.flow_jepa_stateless_goal_phase_machine)
            and int(self.stateless_phase_count) == 4
        ):
            raise ValueError(
                "stateless intent controller requires the complete V116 "
                "four-program supervised-effect graph"
            )
        if int(self.flow_jepa_window_effect_bank) and not (
            int(self.flow_jepa_stateless_intent_controller)
            and int(self.flow_jepa_supervised_effect_mainline)
            and int(self.flow_jepa_future_slots) == 3
            and int(self.future_anchors) == 4
        ):
            raise ValueError(
                "window effect bank requires V117 intent control, three "
                "near/mid/late effect slots, and four online W anchors"
            )
        if int(self.flow_jepa_effect_read_in_p2) and not (
            int(self.flow_jepa_window_effect_bank)
            and int(self.flow_jepa_policy_plan_compiler)
        ):
            raise ValueError(
                "P2 effect read requires the V117 window-effect bank and P3 compiler"
            )
        if int(self.flow_jepa_differential_intent_effect_mainline) and not (
            int(self.flow_jepa_stateless_intent_controller)
            and int(self.flow_jepa_window_effect_bank)
            and int(self.flow_jepa_effect_read_in_p2)
            and int(self.flow_jepa_future_slots) == 3
            and v115_role_schedule
        ):
            raise ValueError(
                "differential intent/effect mainline requires the complete "
                "V117 3-2-3 parent with three effect slots"
            )
        if int(self.flow_jepa_grounded_intent_effect_mainline):
            grounded_required = (
                int(self.flow_jepa_progressive_grounding_address)
                and int(self.flow_jepa_pre_value_owner_routing)
                and int(self.flow_jepa_functional_mainline_routing)
                and int(self.flow_jepa_shared_factual_glimpse_bank)
                and int(self.flow_jepa_g_aligned_future_effect)
                and int(self.flow_jepa_stateless_goal_phase_machine)
                and int(self.flow_jepa_policy_plan_compiler)
                and int(self.flow_jepa_supervised_effect_mainline)
                and int(self.flow_jepa_action_free_world_factual)
                and int(self.flow_jepa_future_slots) == 4
                and int(self.future_anchors) == 4
                and v115_role_schedule
            )
            if not grounded_required:
                raise ValueError(
                    "grounded intent/effect mainline requires the observable "
                    "3-2-3 factual parent and four real effect intervals"
                )
            if int(self.flow_jepa_differential_intent_effect_mainline):
                raise ValueError(
                    "grounded and differential intent/effect mainlines are "
                    "mutually exclusive capabilities"
                )
            if int(self.flow_jepa_window_effect_bank) or int(
                self.flow_jepa_stateless_intent_controller
            ):
                raise ValueError(
                    "grounded intent/effect does not use the V117 three-slot "
                    "window bank or pseudo-program intent controller"
                )
        if int(self.flow_jepa_object_intent_dynamics_mainline):
            object_required = (
                int(self.flow_jepa_progressive_grounding_address)
                and int(self.flow_jepa_coordinate_typed_raw_detail)
                and int(self.flow_jepa_structured_ownership_bottleneck)
                and int(self.flow_jepa_pre_value_owner_routing)
                and int(self.flow_jepa_functional_mainline_routing)
                and int(self.flow_jepa_shared_factual_glimpse_bank)
                and int(self.flow_jepa_g_aligned_future_effect)
                and int(self.flow_jepa_policy_plan_compiler)
                and int(self.flow_jepa_action_free_world_factual)
                and int(self.flow_jepa_future_slots) == 4
                and int(self.future_anchors) == 4
                and v115_role_schedule
            )
            if not object_required:
                raise ValueError(
                    "object-intent dynamics requires the typed observation/P1 "
                    "foundation, four intervals, and the 3-2-3 schedule"
                )
            if int(self.flow_jepa_grounded_intent_effect_mainline) or int(
                self.flow_jepa_differential_intent_effect_mainline
            ):
                raise ValueError(
                    "object-intent dynamics is a distinct top capability and "
                    "cannot be combined with V118/V119 top graphs"
                )
        if int(self.flow_jepa_future_slots) < 1:
            raise ValueError("flow_jepa_future_slots must be positive")
        if str(self.flow_matching_time_distribution) not in {
            "uniform",
            "beta_1_5_1",
        }:
            raise ValueError(
                "flow_matching_time_distribution must be uniform or beta_1_5_1"
            )
        if int(self.flow_jepa_p1_mixed_precision) and not int(
            self.flow_jepa_utility_precision_mainline
        ):
            raise ValueError(
                "P1 mixed precision requires the utility precision mainline"
            )
        if min(
            int(self.flow_jepa_address_query_batch_budget),
            int(self.flow_jepa_microgrid_tile),
            int(self.flow_jepa_checkpoint_min_batch),
        ) < 1:
            raise ValueError(
                "P1 resource budgets and checkpoint batch threshold must be positive"
            )
        if int(self.flow_jepa_address_query_batch_budget) < int(
            self.flow_jepa_address_query_chunk
        ):
            raise ValueError(
                "P1 query batch budget cannot be smaller than the ancestry chunk"
            )
        if int(self.flow_jepa_microgrid_tile) > int(
            self.flow_jepa_raw_micro_grid
        ) ** 2:
            raise ValueError(
                "P1 microgrid tile cannot exceed the selected microgrid cells"
            )
        if not 0.0 < float(self.flow_jepa_pre_value_owner_update_scale) <= 0.25:
            raise ValueError(
                "flow_jepa_pre_value_owner_update_scale must be in (0,0.25]"
            )
        if not 0.0 < float(self.flow_jepa_horizon_address_update_scale) <= 1.0:
            raise ValueError(
                "flow_jepa_horizon_address_update_scale must be in (0,1]"
            )
        if not 0.0 < float(self.flow_jepa_routing_norm_floor) <= 1.0:
            raise ValueError(
                "flow_jepa_routing_norm_floor must be in (0,1]"
            )
        if not 0.0 < float(self.flow_jepa_correlation_rms_floor) <= 1.0:
            raise ValueError(
                "flow_jepa_correlation_rms_floor must be in (0,1]"
            )
        if not 0.0 < float(self.flow_jepa_visibility_transition_fraction) <= 1.0:
            raise ValueError(
                "flow_jepa_visibility_transition_fraction must be in (0,1]"
            )
        if float(self.flow_jepa_horizon_value_max_rms) <= 0.0:
            raise ValueError("flow_jepa_horizon_value_max_rms must be positive")
        if int(self.flow_jepa_variance_safe_routing) and not (
            int(self.flow_jepa_horizon_soft_address)
            and int(self.role_attnres_enabled)
        ):
            raise ValueError(
                "variance-safe routing requires the horizon address and typed "
                "role AttnRes paths"
            )
        if int(self.flow_jepa_complete_numerical_contract) and not (
            int(self.flow_jepa_variance_safe_routing)
            and int(self.flow_jepa_bounded_flow_coordinates)
            and int(self.role_residual_amplitude_contract)
        ):
            raise ValueError(
                "the complete numerical contract requires variance-safe "
                "routing, bounded flow coordinates, and bounded role residuals"
            )
        if not 0.0 < float(self.flow_jepa_interval_stage_update_scale) <= 1.0:
            raise ValueError(
                "flow_jepa_interval_stage_update_scale must be in (0,1]"
            )
        if int(self.flow_jepa_interval_stage_delta):
            if not (
                int(self.flow_jepa_sequential_horizon_memory)
                and int(self.flow_jepa_horizon_soft_address)
                and int(self.flow_jepa_predictive_change_contract)
                and int(self.flow_jepa_late_bottleneck)
            ):
                raise ValueError(
                    "interval-stage delta requires the late-bottleneck "
                    "sequential predictive horizon-address path"
                )
            boundaries = self.flow_jepa_effective_interval_boundaries
            supports = self.flow_jepa_effective_interval_support_offsets
            if len(boundaries) != int(self.future_anchors) + 1:
                raise ValueError(
                    "interval boundaries must contain future_anchors + 1 entries"
                )
            if (
                tuple(sorted(set(boundaries))) != boundaries
                or tuple(sorted(set(supports))) != supports
                or boundaries[0] <= 0
            ):
                raise ValueError(
                    "interval boundaries/support offsets must be strictly "
                    "increasing and positive"
                )
            support_set = set(supports)
            for start, end in self.flow_jepa_interval_windows:
                interval_support = tuple(
                    value for value in supports if start <= value <= end
                )
                if (
                    start not in support_set
                    or end not in support_set
                    or len(interval_support) < 2
                ):
                    raise ValueError(
                        "every interval stage requires both boundaries and at "
                        "least two real support frames"
                    )
        if int(self.flow_jepa_interval_stage_typed_value) and not (
            int(self.flow_jepa_interval_stage_delta)
            and int(self.role_attnres_world_to_policy)
        ):
            raise ValueError(
                "typed interval-stage value requires the interval organizer and W->P AttnRes"
            )
        if int(self.role_residual_amplitude_contract) and not (
            int(self.role_attnres_enabled)
            and int(self.flow_jepa_role_hierarchy)
        ):
            raise ValueError(
                "role residual amplitude contract requires typed role AttnRes"
            )
        if int(self.role_residual_contract_after_gate) and not int(
            self.role_residual_amplitude_contract
        ):
            raise ValueError(
                "post-gate residual contract requires the role residual amplitude contract"
            )
        if float(self.role_residual_max_update_rms) <= 0.0:
            raise ValueError("role_residual_max_update_rms must be positive")
        if float(self.role_attnres_max_value_rms) <= 0.0:
            raise ValueError("role_attnres_max_value_rms must be positive")
        if min(
            int(self.flow_jepa_address_slots),
            int(self.flow_jepa_address_route_dim),
            int(self.flow_jepa_address_query_chunk),
        ) < 1:
            raise ValueError("soft address lattice dimensions must be positive")
        if not 0.0 <= float(self.flow_jepa_address_flow_prior_floor) < 4.0:
            raise ValueError("address flow-prior floor must be in [0,4)")
        if float(self.flow_jepa_address_flow_prior_floor) > 0.0 and not int(
            self.flow_jepa_soft_address_lattice
        ):
            raise ValueError("address flow-prior floor requires the soft address lattice")
        for name in (
            "role_attnres_enabled",
            "role_attnres_ground_to_world",
            "role_attnres_world_to_policy",
            "role_attnres_policy_to_mmdit",
        ):
            if int(getattr(self, name)) not in (0, 1):
                raise ValueError(f"{name} must be 0 or 1")
        if int(self.role_attnres_key_dim) < 1:
            raise ValueError("role_attnres_key_dim must be positive")
        for name in (
            "role_attnres_ground_to_world_scale",
            "role_attnres_world_to_policy_scale",
            "role_attnres_policy_to_mmdit_scale",
        ):
            if not 0.0 < float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must be in (0,1]")
        enabled_role_bridges = (
            int(self.role_attnres_ground_to_world)
            or int(self.role_attnres_world_to_policy)
            or int(self.role_attnres_policy_to_mmdit)
        )
        if enabled_role_bridges and not int(self.role_attnres_enabled):
            raise ValueError("individual role AttnRes bridges require role_attnres_enabled=1")
        if int(self.role_attnres_enabled) and not (
            int(self.flow_jepa_role_hierarchy)
            and int(self.flow_jepa_strict_role_visual_path)
        ):
            raise ValueError("role AttnRes requires the strict 3-3-2 role hierarchy")
        if int(self.role_attnres_policy_to_mmdit) and int(
            self.flow_jepa_policy_workspace_fixed_fusion
        ):
            raise ValueError(
                "typed policy-to-MMDiT bridge replaces fixed policy-workspace fusion"
            )
        if int(self.flow_jepa_policy_workspace_horizon_pool) not in (0, 1):
            raise ValueError("flow_jepa_policy_workspace_horizon_pool must be 0 or 1")
        if not 0.0 < float(self.flow_jepa_late_policy_detail_scale) <= 1.0:
            raise ValueError("flow_jepa_late_policy_detail_scale must be in (0,1]")
        if int(self.flow_jepa_world_anchor_write_only) and not (
            int(self.flow_jepa_strict_role_visual_path)
            and int(self.flow_jepa_role_hierarchy)
        ):
            raise ValueError(
                "anchor-only world writes require the strict role hierarchy"
            )
        if int(self.flow_jepa_policy_workspace_horizon_pool) and not int(
            self.flow_jepa_role_hierarchy
        ):
            raise ValueError(
                "policy workspace horizon pooling requires the role hierarchy"
            )
        if int(self.flow_jepa_late_policy_detail) and not (
            int(self.flow_jepa_complementary_raw_detail)
            and int(self.flow_jepa_policy_workspace_horizon_pool)
        ):
            raise ValueError(
                "late policy detail requires complementary raw detail and "
                "horizon-pooled policy workspace"
            )
        if int(self.flow_jepa_soft_address_lattice) and not (
            int(self.flow_jepa_late_policy_detail)
            and int(self.flow_jepa_strict_role_visual_path)
            and int(self.flow_jepa_raw_image_enabled)
        ):
            raise ValueError(
                "soft address lattice requires raw late-policy detail and the "
                "strict visual ownership path"
            )
        if int(self.flow_jepa_enabled):
            if int(self.visual_history_length) < 2:
                raise ValueError("Flow-DINO JEPA requires at least two visual history frames")
            history_offsets = tuple(int(value) for value in self.flow_jepa_history_offsets)
            if len(history_offsets) != int(self.visual_history_length):
                raise ValueError(
                    "flow_jepa_history_offsets must match visual_history_length"
                )
            if tuple(sorted(set(history_offsets))) != history_offsets:
                raise ValueError("flow_jepa_history_offsets must be strictly increasing")
            if int(self.future_grid_size) != int(self.flow_jepa_grid_size):
                raise ValueError(
                    "Flow-DINO JEPA requires future_grid_size == flow_jepa_grid_size "
                    "so queries, masks, and teacher targets share one spatial chart"
                )
            side = int(round(float(self.patches_per_camera) ** 0.5))
            if side * side != int(self.patches_per_camera):
                raise ValueError("Flow-DINO JEPA requires a square DINO patch grid")
            window_offsets = self.flow_jepa_effective_window_offsets
            if len(window_offsets) != int(self.future_anchors):
                raise ValueError(
                    "flow_jepa_window_offsets must contain exactly future_anchors entries"
                )
            if tuple(sorted(set(window_offsets))) != window_offsets or window_offsets[0] <= 0:
                raise ValueError(
                    "flow_jepa_window_offsets must be strictly increasing positive offsets"
                )
            if int(self.flow_jepa_late_bottleneck):
                _ = self.flow_jepa_action_offsets
                if int(self.flow_jepa_stage_offset) != 0:
                    raise ValueError(
                        "late-bottleneck Flow-DINO folds the far horizon into window offsets; "
                        "flow_jepa_stage_offset must be 0"
                    )
                if int(self.flow_jepa_stage_tokens) != 0:
                    raise ValueError(
                        "late-bottleneck Flow-DINO has no separate stage token"
                    )
                native_side = int(round(float(self.patches_per_camera) ** 0.5))
                if native_side < int(self.flow_jepa_grid_size):
                    raise ValueError(
                        "late-bottleneck reader grid cannot exceed the native DINO patch grid"
                    )
                if int(self.flow_jepa_raw_image_enabled):
                    groups = (
                        int(self.flow_jepa_grounding_blocks),
                        int(self.flow_jepa_world_blocks),
                        int(self.flow_jepa_policy_blocks),
                    )
                    if min(groups) < 1 or sum(groups) != int(self.depth):
                        raise ValueError(
                            "raw-grounding DiT groups must be positive and sum to depth"
                        )
            else:
                if int(self.flow_jepa_raw_image_enabled):
                    raise ValueError(
                        "raw-image grounding requires the late-bottleneck future-chart contract"
                    )
                if window_offsets[-1] != int(self.action_horizon):
                    raise ValueError(
                        "the final flow_jepa_window_offset must equal action_horizon"
                    )
                if int(self.flow_jepa_effective_stage_offset) <= int(window_offsets[-1]):
                    raise ValueError(
                        "flow_jepa_stage_offset must be later than every window offset"
                    )
                if int(self.flow_jepa_stage_tokens) != 1:
                    raise ValueError(
                        "the hierarchical V95 Flow-DINO contract requires one global stage token"
                    )
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
            "layer_low_causal_weight",
            "layer_high_causal_weight",
            "layer_low_latent_weight",
            "layer_high_latent_weight",
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
        if str(self.final_action_decoder) not in {
            "legacy",
            "residual_action_flow",
            "layered_residual_action_flow",
            "latent_main_action",
            "latent_cvae_action",
            "adaptive_recurrent_cvae_action",
            "hierarchical_mmdit_action",
            "evidence_latent_mmdit_action",
        }:
            raise ValueError(
                "final_action_decoder must be legacy, residual_action_flow, layered_residual_action_flow, latent_main_action, latent_cvae_action, adaptive_recurrent_cvae_action, hierarchical_mmdit_action, or evidence_latent_mmdit_action"
            )
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
        if int(self.latent_action_near_depth) > int(self.latent_action_decoder_depth) or int(
            self.latent_action_mid_depth
        ) > int(self.latent_action_decoder_depth):
            raise ValueError(
                "latent_action_near_depth/mid_depth cannot exceed latent_action_decoder_depth"
            )
        if int(self.latent_cvae_z_dim) < 1:
            raise ValueError("latent_cvae_z_dim must be >= 1")
        if int(self.latent_cvae_decoder_depth) < 1:
            raise ValueError("latent_cvae_decoder_depth must be >= 1")
        if float(self.latent_cvae_ffn_expansion) < 1.0:
            raise ValueError("latent_cvae_ffn_expansion must be >= 1")
        for name in (
            "latent_cvae_layer_memory",
            "latent_cvae_transition_memory",
            "latent_cvae_transition_detach",
            "latent_cvae_context_memory",
            "latent_cvae_visual_memory",
            "latent_cvae_layer_detach",
            "latent_cvae_condition_source_norm",
            "latent_cvae_bounded_consequence_fusion",
            "latent_cvae_event_gripper_gate",
            "latent_cvae_inference_sample",
            "latent_cvae_variational",
            "latent_cvae_z_probe",
            "latent_cvae_noisy_gate",
            "latent_cvae_layer_scan",
            "latent_cvae_mmdit_decoder",
            "latent_cvae_mmdit_cond_update",
            "latent_cvae_mmdit_noisy_causal",
            "latent_cvae_mmdit_noisy_logit_gate",
            "latent_cvae_progress_action_isolation",
            "latent_cvae_workspace_noisy_query",
            "latent_cvae_workspace_trajectory_source",
            "latent_cvae_workspace_global_sources",
            "latent_cvae_workspace_layer_source",
            "latent_cvae_workspace_progress_value",
            "latent_cvae_workspace_time_state",
            "latent_cvae_workspace_slot_time_state",
            "latent_cvae_workspace_controller",
            "latent_cvae_hierarchical_workspace",
        ):
            if int(getattr(self, name)) not in (0, 1):
                raise ValueError(f"{name} must be 0 or 1")
        if float(self.latent_cvae_workspace_slot_time_scale) < 0.0:
            raise ValueError("latent_cvae_workspace_slot_time_scale must be >= 0")
        if int(self.latent_cvae_mmdit_depth) < 1:
            raise ValueError("latent_cvae_mmdit_depth must be >= 1")
        if not (0.0 < float(self.latent_cvae_mmdit_residual_scale_max) <= 1.0):
            raise ValueError("latent_cvae_mmdit_residual_scale_max must be in (0, 1]")
        if float(self.latent_cvae_mmdit_source_route_delta_max) < 0.0:
            raise ValueError("latent_cvae_mmdit_source_route_delta_max must be >= 0")
        correction_min = float(self.latent_cvae_mmdit_noisy_correction_min)
        correction_max = float(self.latent_cvae_mmdit_noisy_correction_max)
        if not (0.0 <= correction_min < correction_max <= 1.0):
            raise ValueError(
                "latent_cvae_mmdit_noisy_correction_min/max must satisfy "
                "0 <= min < max <= 1"
            )
        if float(self.latent_cvae_mmdit_noisy_correction_power) <= 0.0:
            raise ValueError("latent_cvae_mmdit_noisy_correction_power must be positive")
        if float(self.latent_cvae_mmdit_noisy_correction_logit_delta) < 0.0:
            raise ValueError(
                "latent_cvae_mmdit_noisy_correction_logit_delta must be >= 0"
            )
        if float(self.latent_cvae_mmdit_controller_modulation_scale) < 0.0:
            raise ValueError(
                "latent_cvae_mmdit_controller_modulation_scale must be >= 0"
            )
        if float(self.latent_cvae_mmdit_evidence_scale) < 0.0:
            raise ValueError("latent_cvae_mmdit_evidence_scale must be >= 0")
        if float(self.latent_cvae_mmdit_noisy_scale) < 0.0:
            raise ValueError("latent_cvae_mmdit_noisy_scale must be >= 0")
        if int(self.latent_cvae_mmdit_operator_capacity) not in (0, 1):
            raise ValueError("latent_cvae_mmdit_operator_capacity must be 0 or 1")
        if int(self.latent_cvae_mmdit_execution_controller) not in (0, 1):
            raise ValueError("latent_cvae_mmdit_execution_controller must be 0 or 1")
        if int(self.latent_cvae_mmdit_dynamic_block_route) not in (0, 1):
            raise ValueError("latent_cvae_mmdit_dynamic_block_route must be 0 or 1")
        if int(self.latent_cvae_mmdit_dynamic_block_route) and not int(
            self.latent_cvae_mmdit_execution_controller
        ):
            raise ValueError("dynamic native block routing requires the execution controller")
        if int(self.latent_cvae_mmdit_operator_capacity):
            if int(self.latent_cvae_mmdit_operator_rank) < 1:
                raise ValueError("latent_cvae_mmdit_operator_rank must be >= 1")
            if int(self.latent_cvae_mmdit_operator_rank) > int(self.hidden_size):
                raise ValueError("latent_cvae_mmdit_operator_rank cannot exceed hidden_size")
            if int(self.latent_cvae_mmdit_operator_groups) < 1 or int(
                self.latent_cvae_mmdit_operator_rank
            ) % int(self.latent_cvae_mmdit_operator_groups):
                raise ValueError("operator rank must be divisible by positive operator groups")
            if float(self.latent_cvae_mmdit_operator_depth_logit_init) <= 0.0:
                raise ValueError("operator depth-logit initialization must be positive")
        if int(self.latent_cvae_mmdit_execution_controller):
            if int(self.latent_cvae_mmdit_control_tokens) < 2:
                raise ValueError("native controller needs at least two control tokens")
            if int(self.latent_cvae_mmdit_controller_depth) < 1:
                raise ValueError("native controller depth must be >= 1")
            if int(self.latent_cvae_mmdit_controller_heads) < 1 or int(
                self.hidden_size
            ) % int(self.latent_cvae_mmdit_controller_heads):
                raise ValueError("native controller heads must divide hidden_size")
            if float(self.latent_cvae_mmdit_controller_ffn_expansion) < 1.0:
                raise ValueError("native controller FFN expansion must be >= 1")
        if int(self.latent_cvae_mmdit_max_dwell) < 1:
            raise ValueError("native controller max dwell must be >= 1")
        if str(self.latent_cvae_mmdit_dwell_mode) not in {
            "fixed",
            "random",
            "learned_shadow",
            "learned",
        }:
            raise ValueError("unsupported native controller dwell mode")
        if str(self.latent_cvae_mmdit_dwell_mode) != "fixed" and not int(
            self.latent_cvae_mmdit_execution_controller
        ):
            raise ValueError("adaptive native dwell requires the execution controller")
        if str(self.latent_cvae_mmdit_dwell_mode) != "fixed" and int(
            self.latent_cvae_mmdit_max_dwell
        ) < 2:
            raise ValueError("adaptive native dwell requires max_dwell >= 2")
        if float(self.latent_cvae_mmdit_execution_soft_temperature) <= 0.0:
            raise ValueError("native execution soft temperature must be positive")
        if int(self.latent_cvae_mmdit_identity_candidate) not in (0, 1):
            raise ValueError("latent_cvae_mmdit_identity_candidate must be 0 or 1")
        if int(self.latent_cvae_mmdit_dynamic_block_route) and not int(
            self.latent_cvae_mmdit_identity_candidate
        ):
            raise ValueError("dynamic native execution requires an identity candidate")
        if not (0.0 < float(self.latent_cvae_mmdit_terminal_prior_weight) <= 1.0):
            raise ValueError("native terminal prior weight must be in (0, 1]")
        if str(self.latent_cvae_mmdit_execution_eval_policy) not in {
            "soft",
            "hard",
            "neutral",
        }:
            raise ValueError("native execution eval policy must be soft, hard, or neutral")
        if int(self.latent_cvae_mmdit_execution_warmup_steps) < 0:
            raise ValueError("native execution warmup must be >= 0")
        if int(self.latent_cvae_mmdit_execution_transition_steps) < 1:
            raise ValueError("native execution transition must be >= 1")
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
            0.0
            <= float(self.hierarchical_mmdit_consequence_scale_init)
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
        if not (
            0.0
            < float(self.hierarchical_mmdit_residual_scale_init)
            <= float(self.hierarchical_mmdit_residual_scale_max)
        ):
            raise ValueError("hierarchical_mmdit_residual_scale_init must be in (0, max]")
        if int(self.hierarchical_mmdit_operator_stages) < 1:
            raise ValueError("hierarchical_mmdit_operator_stages must be positive")
        if int(self.hierarchical_mmdit_operator_stages) < int(self.hierarchical_mmdit_depth):
            raise ValueError("operator stages must cover every refinement ownership block")
        if int(self.hierarchical_mmdit_operator_rank) < 1:
            raise ValueError("hierarchical_mmdit_operator_rank must be positive")
        if int(self.hierarchical_mmdit_operator_rank) > int(self.hidden_size):
            raise ValueError("hierarchical_mmdit_operator_rank cannot exceed hidden_size")
        if int(self.hierarchical_mmdit_operator_groups) < 1:
            raise ValueError("hierarchical_mmdit_operator_groups must be positive")
        if int(self.hierarchical_mmdit_operator_rank) % int(
            self.hierarchical_mmdit_operator_groups
        ):
            raise ValueError("hierarchical_mmdit_operator_rank must be divisible by groups")
        if float(self.hierarchical_mmdit_operator_depth_logit_init) <= 0.0:
            raise ValueError("operator depth logit init must be positive")
        if int(self.hierarchical_mmdit_operator_contraction_warmup_steps) < 0:
            raise ValueError("operator contraction warmup steps must be non-negative")
        if int(self.hierarchical_mmdit_operator_contraction_transition_steps) < 1:
            raise ValueError("operator contraction transition steps must be positive")
        if int(self.hierarchical_mmdit_unified_controller) not in (0, 1):
            raise ValueError("hierarchical_mmdit_unified_controller must be 0 or 1")
        if int(self.hierarchical_mmdit_control_tokens) < 1:
            raise ValueError("hierarchical_mmdit_control_tokens must be positive")
        if int(self.hierarchical_mmdit_controller_depth) < 1:
            raise ValueError("hierarchical_mmdit_controller_depth must be positive")
        if int(self.hierarchical_mmdit_controller_heads) < 1:
            raise ValueError("hierarchical_mmdit_controller_heads must be positive")
        if int(self.hierarchical_mmdit_unified_controller) and int(self.hidden_size) % int(
            self.hierarchical_mmdit_controller_heads
        ):
            raise ValueError("hidden_size must be divisible by controller_heads")
        if float(self.hierarchical_mmdit_controller_ffn_expansion) < 1.0:
            raise ValueError("controller_ffn_expansion must be >= 1")
        if int(self.hierarchical_mmdit_spectral_state) not in (0, 1):
            raise ValueError("hierarchical_mmdit_spectral_state must be 0 or 1")
        if int(self.hierarchical_mmdit_spectral_state):
            if str(self.arm_flow_mode) != "manifold_native":
                raise ValueError("complete spectral flow requires arm_flow_mode=manifold_native")
            if str(self.gripper_field_mode) != "parseval_temporal":
                raise ValueError(
                    "complete spectral flow requires gripper_field_mode=parseval_temporal"
                )
        for name in (
            "hierarchical_mmdit_spectral_arm_start_fraction",
            "hierarchical_mmdit_spectral_gripper_start_fraction",
        ):
            value = float(getattr(self, name))
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        if float(self.hierarchical_mmdit_spectral_temperature) <= 0.0:
            raise ValueError("hierarchical_mmdit_spectral_temperature must be positive")
        if float(self.hierarchical_mmdit_spectral_schedule_power) <= 0.0:
            raise ValueError("hierarchical_mmdit_spectral_schedule_power must be positive")
        if float(self.hierarchical_mmdit_spectral_controller_shift_limit) < 0.0:
            raise ValueError(
                "hierarchical_mmdit_spectral_controller_shift_limit must be non-negative"
            )
        if float(self.hierarchical_mmdit_spectral_competition_loss_weight) < 0.0:
            raise ValueError(
                "hierarchical_mmdit_spectral_competition_loss_weight must be non-negative"
            )
        if int(self.hierarchical_mmdit_spectral_competition_warmup_steps) < 0:
            raise ValueError(
                "hierarchical_mmdit_spectral_competition_warmup_steps must be non-negative"
            )
        if (
            int(self.hierarchical_mmdit_unified_controller)
            and str(self.final_action_decoder) != "hierarchical_mmdit_action"
        ):
            raise ValueError(
                "hierarchical_mmdit_unified_controller requires "
                "final_action_decoder=hierarchical_mmdit_action"
            )
        if str(self.hierarchical_mmdit_schedule_mode) not in {"fixed", "random_dwell"}:
            raise ValueError("hierarchical_mmdit_schedule_mode must be fixed or random_dwell")
        if int(self.hierarchical_mmdit_operation_candidate_probes) not in {0, 1}:
            raise ValueError("hierarchical_mmdit_operation_candidate_probes must be 0 or 1")
        if int(self.hierarchical_mmdit_operation_value_warmup_steps) < 0:
            raise ValueError("hierarchical_mmdit_operation_value_warmup_steps must be non-negative")
        if str(self.hierarchical_mmdit_dwell_mode) not in {
            "fixed",
            "shadow",
            "learned",
        }:
            raise ValueError("hierarchical_mmdit_dwell_mode must be fixed, shadow, or learned")
        if str(self.hierarchical_mmdit_execution_contract) not in {
            "legacy_stage_keep",
            "typed_block_budget",
        }:
            raise ValueError(
                "hierarchical_mmdit_execution_contract must be "
                "legacy_stage_keep or typed_block_budget"
            )
        if str(self.hierarchical_mmdit_execution_contract) == "typed_block_budget" and not int(
            self.hierarchical_mmdit_unified_controller
        ):
            raise ValueError("typed_block_budget requires hierarchical_mmdit_unified_controller=1")
        if str(self.hierarchical_mmdit_dwell_mode) != "fixed":
            if not int(self.hierarchical_mmdit_unified_controller):
                raise ValueError("value dwell requires hierarchical_mmdit_unified_controller=1")
            if not int(self.hierarchical_mmdit_operation_candidate_probes):
                raise ValueError(
                    "value dwell requires hierarchical_mmdit_operation_candidate_probes=1"
                )
        if not 0.0 <= float(self.hierarchical_mmdit_random_prefix_probability) <= 1.0:
            raise ValueError("hierarchical_mmdit_random_prefix_probability must be in [0,1]")
        if str(self.hierarchical_mmdit_exhaustion_mode) not in {
            "off",
            "shadow",
            "adaptive",
            "learned_shadow",
            "learned",
        }:
            raise ValueError(
                "hierarchical_mmdit_exhaustion_mode must be off, shadow, adaptive, "
                "learned_shadow, or learned"
            )
        for name in (
            "hierarchical_mmdit_action_response_thresholds",
            "hierarchical_mmdit_stage_pressure_thresholds",
        ):
            values = tuple(float(value) for value in getattr(self, name))
            if len(values) != 3 or any(value < 0.0 for value in values):
                raise ValueError(f"{name} must contain three non-negative values")
        if str(self.hierarchical_mmdit_exhaustion_mode) in {"shadow", "adaptive"}:
            action_thresholds = tuple(
                float(value) for value in self.hierarchical_mmdit_action_response_thresholds
            )
            stage_thresholds = tuple(
                float(value) for value in self.hierarchical_mmdit_stage_pressure_thresholds
            )
            if any(value <= 0.0 for value in action_thresholds):
                raise ValueError(
                    "shadow/adaptive exhaustion requires three calibrated positive action-response thresholds"
                )
            if any(value <= 0.0 for value in stage_thresholds):
                raise ValueError(
                    "shadow/adaptive exhaustion requires three calibrated positive stage-pressure thresholds"
                )
        if int(self.hierarchical_mmdit_unified_controller) and str(
            self.hierarchical_mmdit_exhaustion_mode
        ) in {"shadow", "adaptive"}:
            raise ValueError(
                "the unified controller does not use threshold-based shadow/adaptive exhaustion; "
                "use off, learned_shadow, or learned"
            )
        if float(self.hierarchical_mmdit_action_response_floor) <= 0.0:
            raise ValueError("hierarchical_mmdit_action_response_floor must be positive")
        if int(self.hierarchical_mmdit_exhaustion_confirm_steps) < 1:
            raise ValueError("hierarchical_mmdit_exhaustion_confirm_steps must be positive")
        if str(self.hierarchical_mmdit_architecture_version) != (
            "post_gate_contraction_sidecar_v12_value_dwell"
        ):
            raise ValueError(
                "unsupported hierarchical_mmdit_architecture_version: "
                f"{self.hierarchical_mmdit_architecture_version!r}"
            )
        if str(self.final_action_decoder) == "hierarchical_mmdit_action" and not int(
            self.layer_contract_adapters
        ):
            raise ValueError("hierarchical_mmdit_action requires layer_contract_adapters=1")
        if str(self.final_action_decoder) == "evidence_latent_mmdit_action":
            if not int(self.layer_contract_adapters):
                raise ValueError("evidence_latent_mmdit_action requires layer_contract_adapters=1")
            if int(self.hierarchical_mmdit_spectral_state):
                raise ValueError(
                    "evidence_latent_mmdit_action is a native-time migration path; "
                    "disable hierarchical_mmdit_spectral_state"
                )
            if int(self.latent_cvae_mmdit_cond_update):
                raise ValueError("evidence_latent_mmdit_action requires read-only condition tokens")
        if str(self.final_action_decoder) == "hierarchical_mmdit_action":
            if int(self.hierarchical_mmdit_refine_steps) < int(self.hierarchical_mmdit_depth):
                raise ValueError(
                    "hierarchical_mmdit_refine_steps must cover every distinct refinement block"
                )
            if (
                int(self.hierarchical_mmdit_low_slots) < 5
                or int(self.hierarchical_mmdit_low_slots) % 5 != 0
            ):
                raise ValueError(
                    "hierarchical_mmdit_action requires low slots to be a positive multiple of five"
                )
            if int(self.hierarchical_mmdit_stage_slots) < int(
                self.hierarchical_mmdit_operator_stages
            ):
                raise ValueError("hierarchical_mmdit_stage_slots must cover every operator stage")
            if str(self.hierarchical_mmdit_schedule_mode) == "random_dwell" and int(
                self.hierarchical_mmdit_refine_steps
            ) < int(self.hierarchical_mmdit_depth):
                raise ValueError(
                    "random_dwell requires enough refine steps to cover every refinement block"
                )
            if int(self.hierarchical_mmdit_noisy_gate_mode) != 0:
                raise ValueError(
                    "post_gate_contraction_sidecar_v12 does not add an external noisy amplitude gate"
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
        if str(self.final_action_decoder) in (
            "latent_cvae_action",
            "adaptive_recurrent_cvae_action",
        ):
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
                raise ValueError(
                    "adaptive_recurrent_cvae_action with latent_cvae_layer_memory=1 requires layer_contract_adapters=1"
                )
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
        if float(self.adaptive_cvae_route_max_temperature) < float(
            self.adaptive_cvae_route_min_temperature
        ):
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
        if float(self.adaptive_cvae_condition_strength_max) < float(
            self.adaptive_cvae_condition_strength_min
        ):
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
        if not (
            0.0 <= float(self.adaptive_cvae_micro_kp_init) <= float(self.adaptive_cvae_micro_kp_max)
        ):
            raise ValueError("adaptive_cvae_micro_kp_init must be within [0, kp_max]")
        if not (
            0.0 <= float(self.adaptive_cvae_micro_kd_init) <= float(self.adaptive_cvae_micro_kd_max)
        ):
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
        if int(self.layer_recurrent_consequence):
            expected_consequence_steps = (
                len(self.flow_jepa_action_offsets)
                if int(self.flow_jepa_enabled)
                else int(self.future_anchors)
            )
            if int(self.layer_consequence_steps) != expected_consequence_steps:
                raise ValueError(
                    "layer_consequence_steps must equal the number of action-aligned "
                    f"future anchors ({expected_consequence_steps})"
                )
        if int(self.layer_recurrent_consequence) and int(self.layer_consequence_steps) > int(
            self.action_horizon
        ):
            raise ValueError("layer_consequence_steps cannot exceed action_horizon")
