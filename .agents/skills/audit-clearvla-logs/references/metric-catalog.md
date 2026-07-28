# ClearVLA Log Metric Catalog

## Contents

- Evidence hierarchy
- Compact display naming
- Objective and optimization
- Prediction and validation
- Structure and control
- Gradients and interventions
- Coverage and comparison

## Evidence hierarchy

1. `loss_ledger_residual` verifies whether registered weighted contributions reconstruct the scalar sent to backward.
2. `loss_group_*` gives objective budget by action, rollout, execution, latent, and layer groups.
3. `loss_contrib_*` gives weighted component contributions.
4. Raw losses describe task error in their native scale; they do not describe optimization dominance without a weight.
5. Audit gauges such as execution cost, null-space geometry, or sampling probes may be detached and absent from backward.

For pre-ledger logs, the audit utility reconstructs known terms from the serialized trainer config and labels the result `estimated-known-terms`.

## Compact display naming

The V94 console uses medium-length display labels while the JSONL retains the
canonical source keys. The audit utility accepts both the old abbreviated
labels and the current display labels.

- Keep familiar statistical abbreviations such as `rmse`, `f1`, `lr`, and
  `grad`; avoid opaque implementation abbreviations such as `pfn`, `stdr`, and
  `dnratio`.
- Use one to three semantic words: `native_flow`, `rollout_std_ratio`,
  `effective_basis_mass`, and `value_top1_acc`.
- Distinguish raw losses, weighted contributions, and diagnostics:
  `event_loss`, `top_contrib`, and `grip_event_ratio` are different kinds of
  quantities.
- Distinguish training flow errors from sampled validation errors:
  `flow_first8` versus `first8_rmse`.
- `capacity_gate_mass` and `effective_basis_mass` are continuous transparency
  measures, not a physically pruned integer rank or measured compute saving.
  The run header's decoder depth is a separate configured architecture axis.
- `cost_proxy` remains a detached execution-budget proxy, not measured compute
  cost and not an objective contribution.

## Objective and optimization

| Family | Canonical metrics | Read together |
|---|---|---|
| Main action | `physical_flow`, `physical_flow_native`, `arm_fm_per_dim`, `gripper_fm_field` | native/uniform anchors, arm/gripper balance |
| Decode closure | `decoded_action`, `physical_delta_consistency`, `transition_l1` | sampled validation RMSE and event timing |
| Event/motion | `event`, `motion`, event/motion P/R/F1 | predicted and target positive counts |
| Rollout | `rollout_dynamics`, `rollout_contrast`, `rollout_variance`, `rollout_norm`, `rollout_milestone_delta_match` | std ratio, delta norm ratio, tail behavior |
| Layer ownership | `layer_contract`, `loss_contrib_layer_contract`, per-layer contracts | adapter/consequence gradients and schedule |
| Execution value | value loss, target/predicted spread, correlation, pairwise/decision accuracy, common-mode ratio | candidate coverage and selection entropy |

Known compatibility aliases require care: historical `future_latent` and `action_effect` names do not necessarily denote independent objectives. The utility detects numerically duplicate rollout series and checks whether both effective weights are active.

## Prediction and validation

- `full_rmse` is not enough. Always include `first_rmse`, `first8_rmse`, `tail_rmse`, and `tail_first_ratio`.
- Separate `arm_full_rmse` from `gripper_full_rmse`; a lower aggregate can hide semantic-channel regression.
- Read gripper precision/recall/F1 with `gripper_event_ratio` and predicted/target event counts.
- Read event-head metrics separately from decoded gripper events. Agreement is a closure property, not guaranteed by head accuracy.
- Report `event_head_minus_decoded_gripper_f1`; the decoded gripper trajectory
  remains the deployment behavior, while the event head is auxiliary evidence.
- `proposal_utility_mse_gain > 0` indicates improvement over the no-proposal ablation. Always report ablation coverage.
- Training pflow and sampled validation RMSE use different procedures; a large training decrease with flat validation is a real mismatch signal, not a numerical contradiction.

## Structure and control

- `execution_progress` defines the phase. Capacity/depth remaining full during warmup is expected.
- Read `capacity_gate_mass` with `effective_basis_mass`, removed fraction, and
  configured basis count. Do not call this hardware rank reduction.
- Read route and dwell as soft/hard pairs and include their gaps.
- Read terminal prior, soft terminal probability, hard terminal fraction, and
  terminal target/predicted margins together. A prior below one is a policy
  bias, not an execution-cost loss.
- Soft/hard terminal fractions are terminal occupancy after each decision,
  averaged over the execution clock; they are not raw exit-event counts.
- Execution cost is audit-only on the native Evidence path unless source inspection proves otherwise.
- High value common-mode with near-chance decision accuracy indicates weak candidate discrimination even when pairwise accuracy is above chance.
- Selection entropy needs the legal candidate count for an absolute interpretation; max probability supplies a more portable companion gauge.

## V99 visual-flow identifiability

- `flow_jepa_raw_identity_warp_error` is the exact zero-flow baseline on fixed
  RGB/census evidence. It is not a loss contribution by itself.
- Read `flow_jepa_raw_warp_gain_over_zero` with
  `flow_jepa_raw_moving_warp_gain`, `flow_jepa_raw_static_warp_gain`, and
  `flow_jepa_raw_observable_motion_fraction`. A useful flow should have positive
  moving-region gain while static flow remains close to identity; flow magnitude
  alone has no target range.
- Prefer `flow_jepa_raw_moving_correlation_entropy` and
  `flow_jepa_raw_moving_correlation_margin` over their whole-image counterparts
  when diagnosing matching. Their region weights come from fixed observations,
  so static background cannot hide a moving-region correspondence collapse.
- `flow_jepa_identity_advantage_loss` is an active conditional constraint only
  when its explicit trainer weight is positive. It does not impose minimum
  motion on static pixels.
- V99 `flow_jepa_raw_address_flow_mass` is a two-lane probability between a
  flow-centred detail read and a pooled content fallback. It is no longer
  comparable in absolute value with V97/V98 mass over two equal-size local
  candidate banks.
- `flow_jepa_raw_address_center_separation` measures displacement in reader-cell
  units, not usefulness. Pair it with address logit advantage, lane value
  difference, and warp gain over zero.
- `flow_jepa_raw_address_zero_flow_value_delta` and
  `flow_jepa_raw_address_shuffled_flow_value_delta` are validation-only causal
  gauges. The first detects identity-address dependence; the second checks
  whether the reader is specific to the predicted address rather than merely
  sensitive to any nonzero displacement. Neither is a backward objective.

## V100 strict complementary visual path

- `flow_jepa_raw_additive_detail_path=1` changes the address-mass semantics.
  `flow_jepa_raw_address_flow_mass` is displayed as `raw_detail_share` and is
  the projected high-frequency energy share; the companion is `raw_base_share`.
  They are not V99 lane probabilities and must not be compared across versions.
- `flow_jepa_raw_detail_fused_with_latest_dino=1` means latest DINO content and
  the complementary raw read are combined by a fixed variance-preserving sum.
  There is no learned fusion gate to collapse. `refined_visual_tokens` should
  equal the pre-refinement evidence count because this path fuses rather than
  appends a parallel raw bank.
- `flow_jepa_static_identity_loss` is active only with a positive explicit
  weight. It penalizes static-region warp error above the exact identity error;
  it never requests nonzero motion. Read it with `static_gain`.
- `flow_jepa_future_change` is the active continuous change-weighted objective;
  `flow_jepa_future_change_direction` remains the direction-only diagnostic.
  Always use its `loss_contrib_*` entry to judge optimization budget.
- A V100 action-gradient claim requires nonzero gradients through grounding,
  world, and policy block groups. `raw_dino_fused=1` alone proves construction,
  not deployed action utility; keep using matched zero/shuffle action probes.

## V101 information-balanced long horizon

- `[v101-balance]` is the compact view of sampling and temporal allocation.
  `flow_without_info_balance` is the same physical-flow residual without the
  optional per-window information multiplier; V101 keeps that multiplier at
  one by default because the sampler already supplies bounded strata.
- `action_h1_4`, `action_h5_12`, and `action_h13_24` are unweighted physical
  flow errors for the real 4/12/24 action-anchor bands. Read them with
  `horizon_weight_first` and `horizon_weight_tail`; the configured weights are
  normalized to mean one and are not separate loss terms.
- V101 validation adds `action_band_1_4_rmse`, `action_band_5_12_rmse`, and
  `action_band_13_24_rmse` from deploy-style sampled actions. These are the
  decision metrics for a long-horizon plateau; the training-band flow errors
  use teacher-forced flow matching and are not interchangeable with them.
- `flow_jepa_horizon_balance_active=1` means JEPA losses normalize each real
  future offset independently and average offsets. It prevents a high-energy
  far target from setting the denominator for near horizons; it does not force
  equal predictions or equal teacher change magnitudes.
- `teacher_past_quota`, `teacher_change_quota`, and `teacher_uniform_quota`
  describe disjoint exact per-horizon loss-mask allocations. Future teacher
  change selects loss positions only after the online forward; it is never a
  model input. `selected_change_ratio` compares normalized change inside the
  selected mask with the whole chart and is diagnostic, not an objective.
- `history_keep`, `goal_keep`, and `proposal_keep` are realized training-batch
  condition fractions. History dropout covers direct history, compressed
  history memory, and proposal tokens together so the same condition cannot
  survive under an alias; there is no inverse-probability amplitude scaling.
- `top_policy_fixed_fusion=1` replaces the historical fixed `0.10` policy
  workspace addition with a fixed variance-preserving two-branch fusion. It is
  not a learned gate. Require nonzero `top_policy_lift` gradient and action-level
  flow interventions before claiming useful top-to-bottom dependence.
- `raw_source_dino_fused=1` records the V101 coordinate contract: forward-flow
  raw detail is indexed by the preceding/source grid and is fused with that
  source DINO chart. `raw_dino_fused` (latest-chart fusion) should be zero for
  V101. Mixing a source-indexed raw token with the same-index latest DINO token
  is a coordinate mismatch, not a harmless alternative fusion convention.
- The V101 address intervention uses deterministic spatial misalignment for
  the shuffle control and episode-cluster bootstrap intervals. This avoids
  treating stride-1 neighbouring windows as independent or as strong shuffled
  donors.
- V101 action-path probe schema v3 separates
  `world_residual_anchor_shuffle` from `world_residual_spatial_shuffle`.
  Both keep the grounding/position chart and camera identity fixed and move
  only the residual written by world blocks. Use the combined shuffle only to
  measure interaction; it cannot identify whether temporal or xy ownership is
  responsible.
- V101 schema v3 `raw_detail_zero` and `raw_detail_spatial_shuffle` intervene
  after the complementary reader's output projection. The detail residual is
  defined exactly as `full_reader_output - base_only_reader_output`; DINO and
  the complete base-only output remain fixed. These modes test whether raw
  high-frequency detail reaches deployed action. They are distinct from
  `raw_address_*`, which changes coordinates while leaving detail accessible.
- Read `flow_jepa_raw_post_reader_detail_*_residual_norm` with the matching
  intervention delta before interpreting a null action effect. A zero
  representation delta means the intervention failed to alter the intended
  boundary; a nonzero representation delta with null action delta indicates
  downstream filtering or compensation.
- Schema v3 batch selection is episode-stratified when dataset window
  descriptors are available. Report selected episode count and event-episode
  coverage with every bootstrap interval; increasing adjacent windows without
  increasing trajectory clusters does not strengthen generalization evidence.

## V102 anchor world and late raw detail

- `flow_jepa_world_spatial_residual_norm` measures only the residual written by
  world blocks relative to the grounding boundary. Under the anchor/camera-only
  contract it should be at numerical zero, while
  `flow_jepa_world_anchor_camera_residual_norm` should remain nonzero. This pair
  distinguishes a working restriction from a world path that simply stopped
  writing.
- `flow_jepa_raw_detail_deferred_to_policy=1` and
  `flow_jepa_raw_detail_action_independent_compile=1` mean the exact
  post-reader high-frequency residual is not fused into DINO/world memory and
  its early bank cannot read noisy action or world tokens.
- `flow_jepa_late_detail_attention_entropy` and
  `flow_jepa_late_detail_attention_max` are per-camera spatial-read
  diagnostics. Read them with `flow_jepa_late_detail_update_norm` and
  `flow_jepa_late_detail_trajectory_ratio`; concentrated attention without a
  nonzero update does not establish an active value path.
- `flow_jepa_late_detail_fixed_scale` is a fixed structural coefficient, not a
  learned gate. `flow_jepa_late_detail_token_count` must equal
  `camera_count * reader_grid^2`; a different value means camera/xy ownership
  was lost before the late read. `grad_late_raw_detail_reader`, raw-address,
  raw-flow, and raw pyramid gradients establish whether the action/event loss
  reaches the route.
- `evidence_top_policy_workspace_horizon_pool=1` means every basis token is
  lifted separately and then pooled within its own action time step. It replaces
  interpolation over the flattened `T*basis` axis, which mixed basis and event
  time.
- Sampling emits `sample_flow_jepa_late_detail_*` and
  `sample_flow_jepa_world_*` means across ODE steps. Use them to verify that the
  cached observation-only detail bank remains active beyond the first step.

## V103 typed predictive path

- A real V103 run is identified by active predictive-change, soft-address, and
  typed policy-delta contracts. Do not interpret a historical `[v102-*]`
  prefix as source identity; current logging derives `v103` from the active
  contract fields and the serialized run context remains authoritative.
- `flow_jepa_future_query_adjacent_cosine` near one means the four horizon
  queries became common-mode. Under V103 this can follow directly from the
  parallel latest-motion seed; it is not evidence of a chronological +48
  memory.
- `flow_jepa_world_spatial_residual_norm` is legal and expected to be nonzero
  in V103 because `flow_jepa_world_anchor_write_only=0`. Read it with
  `flow_jepa_world_anchor_camera_residual_norm`, not against V102's
  near-zero-xy invariant.
- `flow_jepa_late_detail_trajectory_ratio` is scale relative to the receiving
  trajectory. A nonzero detail update with a ratio collapsing toward zero can
  mean the carrier is expanding around it; it is not proof that the raw reader
  stopped producing values.
- Typed-route effective counts and semantic-axis route standard deviations are
  factual diversity diagnostics. They establish neither task utility nor a
  target distribution; no entropy/balance threshold should be optimized from
  them.

## V104 bounded geometry, residual amplitude, and sequential horizons

- V104 must report all three active contracts:
  `flow_jepa_bounded_flow_coordinates=1`,
  `flow_jepa_sequential_horizon_memory=1`, and
  `role_residual_contract_enabled=1`. A partial combination is an ablation, not
  the formal V104 model.
- `flow_jepa_raw_mid_boundary_compression` and
  `flow_jepa_raw_high_boundary_compression` are relative differences between
  unconstrained proposals and their smooth in-image coordinates. They are not
  penalties, gates, clipped-pixel fractions, or desired nonzero values.
  Sustained high compression means the learned refiner repeatedly requests
  motion beyond available source-relative geometry.
- `flow_jepa_motion_evidence_flow_magnitude` is the displacement magnitude
  after resolution normalization. Compare native/raw grid-unit flow only with
  warp/validity/address geometry; compare the normalized value with learned
  motion-carrier stability. `flow_jepa_motion_evidence_normalized=1` records
  this unit contract.
- `role_residual_raw_rms` is the pre-contract mean of actual G/W/P sublayer
  writes; `role_residual_bounded_rms` is what can reach the role addition;
  `role_residual_compression` is their smooth saturation activity. Read the
  same raw/written/compression triplet for W->P and bottom P->MMDiT/protected
  detail values. Compression is factual, not an objective.
- A written RMS below its configured maximum proves only amplitude safety.
  Persistent raw RMS growth with compression approaching one still indicates
  optimizer pressure against the interface and should be reported even if the
  carrier stays finite.
- `flow_jepa_perceptual_history_entropy` and
  `flow_jepa_perceptual_history_latest_mass` describe the observed pair-history
  read. Low entropy is not automatically collapse if one pair is genuinely
  informative; use history permutation/null interventions to judge causality.
- `flow_jepa_horizon_transition_update_rms` and
  `flow_jepa_horizon_transition_state_delta` establish that the shared
  chronological transition writes. Their per-horizon variants localize a
  stalled or exploding step.
- `flow_jepa_future_query_adjacent_cosine` remains a collapse diagnostic in
  V104. The sequential mechanism makes values below one structurally possible;
  it does not impose a cosine target. Pair it with transition state deltas and
  per-horizon JEPA/action validation before claiming long-horizon use.
- Healthy V104 evidence is relational rather than a magic range: raw flow stays
  finite, valid sampling does not trend to zero, moving warp gain does not turn
  increasingly negative, written role/value RMS respects its contract, late
  detail ratio is not being numerically drowned, and transition state deltas
  remain finite and nonzero. Only trained logs and matched interventions decide
  usefulness.

## V105 horizon-specific soft address and reliable future delta

- A formal V105 row requires
  `flow_jepa_horizon_soft_address=1`,
  `flow_jepa_future_reliable_normalization=1`,
  `flow_jepa_horizon_address_supervision_active=1`, and the complete V104 fields.
  A prefix without those serialized fields is not sufficient evidence.
- `flow_jepa_future_raw_delta_loss` is the unnormalized static/change anchor.
  `flow_jepa_future_reliable_normalized_loss` is the normalized magnitude term
  after continuous teacher-change reliability attenuation.
  `flow_jepa_future_change_reliability` reports the mean attenuation. A low
  value is evidence that normalized targets are weak, not permission to drop
  the raw anchor.
- `flow_jepa_future_target_delta_scale` is the raw teacher-change RMS,
  `flow_jepa_future_current_reference_scale` is the current-chart RMS, and
  `flow_jepa_future_normalization_scale` is their smooth V105 denominator
  `sqrt(delta_rms^2 + (0.05 * current_rms)^2)`. If the normalization scale
  simply tracks a tiny delta while the current reference is much larger, the
  reliable contract is not active or the log is not a complete V105 row.
- `flow_jepa_horizon_address` is the weighted teacher-only spatial KL.
  Read it with `flow_jepa_horizon_address_teacher_reliability`; a small KL with
  near-zero reliability is not proof of correct addressing.
- `flow_jepa_horizon_address_teacher_entropy` and
  `flow_jepa_horizon_address_predicted_entropy` compare target and predicted
  spatial distributions. There is no desired entropy target. Use the
  per-horizon KL/reliability fields and causal probes before calling a broad
  distribution collapsed.
- `flow_jepa_horizon_address_route_entropy`/`route_max` describe the soft
  source-cell/slot posterior, while `fine_entropy`/`fine_max` describe
  continuous sub-cell candidate selection. Neither is a loss or quota.
- `flow_jepa_horizon_address_update_rms` is the applied fixed-scale residual
  entering only the future predictor; `update_ratio` is relative to its W
  carrier. A nonzero update proves execution, not utility.
- `flow_jepa_horizon_address_variation` is adjacent-horizon total-variation
  distance between predicted relevance distributions. Zero indicates
  common-mode addresses; a large value is not automatically better.
- `flow_jepa_horizon_address_cross_cell_distance` measures how far the soft
  source-cell expectation moves from its target query cell in normalized
  8x8 coordinates. Nonzero values establish non-identity access but not
  correct pixel correspondence.
- `grad_flow_dino_horizon_address` must be interpreted with future/address
  loss contributions and compiler/W gradients. The horizon reader is
  deliberately absent from ordinary deployment action sampling when
  diagnostics are disabled.

## V106 interval-stage increment and variance-safe routing

- A successful formal startup must print
  `[v106-preflight] ... interval_teacher=pass supports=12
  numerical_contract=pass ...`. Deployment sampling alone is insufficient
  because it does not construct frozen future targets or identify the
  finite-gain graph.
- A formal V106 row requires
  `flow_jepa_interval_stage_enabled=1`,
  `flow_jepa_variance_safe_routing=1`,
  `flow_jepa_complete_numerical_contract=1`, a positive
  `loss_contrib_flow_jepa_interval_stage`, and the complete V105 fields.
- Horizon labels now mean intervals:
  H4=[4,8], H12=[8,16], H24=[16,32], H48=[32,48].
  Do not compare their target semantics directly with V104/V105 point-anchor
  rows.
- `flow_jepa_interval_stage` is the complete weighted source metric before its
  trainer weight. `raw`, `normalized`, `direction`, and `endpoint` are
  components, not separately added objectives.
- `flow_jepa_interval_stage_target_scale` and `reliability` report whether the
  signed teacher progression is identifiable relative to the current frozen
  chart. A low reliability attenuates normalized/direction terms but does not
  remove the raw increment anchor.
- `flow_jepa_interval_stage_direction_floor_min` is the frozen-scale floor
  used by signed-direction supervision. It must stay positive: a zero or
  missing floor would make the near-zero-initialized progression path
  vulnerable to a cosine-normalization backward singularity.
- `flow_jepa_future_direction_floor_min` is the corresponding V106 floor on
  the inherited predictive-change direction term. Both direction floors must
  be read alongside global/Flow-DINO preclip gradients; fixing only the new
  interval loss would leave the parent JEPA loss ill-conditioned.
- `flow_jepa_interval_stage_written_delta_rms` is the actual bounded W->P write;
  `carrier_ratio` prices it against the incoming W chart. The supervised
  progression is the same bounded tensor before the fixed write scale, so a
  separate prediction head cannot claim success while the deployed write is
  zero.
- Compact `interval_h4/h12/h24/h48=l:/r:/w:` fields report per-interval total
  source loss, teacher reliability, and actual fixed-scale W->P write RMS.
  They are the first check for a far-stage average hiding a dead short-range
  spatial increment (or the reverse).
- Interval encodings are query/key selectors only; values come from the
  observable W chart through bias-free projections. An exact-zero W chart must
  therefore produce an exact-zero interval write rather than a learned
  horizon-only template.
- `flow_jepa_horizon_address_value_precontract_rms`,
  `value_channel_std`, and `value_contraction` distinguish available address
  evidence from upper-bound compression. V106 never unit-normalizes a small
  cancelled address value.
- The query/key/value denominator minima at horizon address, G->W, W->P,
  P->MMDiT, and protected detail must stay finite and at least the configured
  variance floor in a formal run. Read them together with
  `grad_flow_dino_interval_stage`, `grad_flow_dino_horizon_address`, and all
  three AttnRes gradient groups.
- `corr_feature_rms_min` is the smallest learned semantic/raw correlation
  feature RMS before normalization. `corr_norm_denom_min` must remain at
  least `0.10`, and `corr_norm_gain_max` must be at most 10 under the formal
  launcher. A falling feature RMS is no longer itself a gradient explosion.
- `raw_occ` is the continuous online occlusion consumed by address and motion
  evidence; `hard_occ_audit` is the detached historical threshold for
  comparison only. `visibility_width_min` must be positive and
  `visibility_gain_bound` must be at most 5 with the formal 0.10 transition
  fraction and 0.5 threshold floor.
- `role_norm_denom_min` must be at least `0.25`.
  `role_norm_gain_max` includes the smoothly bounded learnable affine scale
  and must not exceed 16. Read it with G/W/P preclip gradients: bounded
  residual RMS alone does not prove bounded internal attention/FFN Jacobians.
- Use `interval_stage_zero` and `interval_stage_episode_shuffle` from the V106
  model-path probe to decide whether the bounded stage write reaches deployed
  action. A nonzero representation write alone is not causal action evidence.

## V108/V109 deployed and progressive address ownership

- V108's `online_address_write` is the single G3->W1 value write retained as a
  diagnostic baseline. In a V109 row it must be absent: V109 disables that
  read even though the serialized V108 parent flag remains enabled.
- `progressive_address=1` proves the V109 graph executed. Read
  `g1_coarse_entropy/max`, `g2_fine_entropy/center_shift`,
  `g3_coarse_prior/summary`, `world_address_entropy/horizon_variation`, and
  `policy_address_prior` as one ordered chain; no single nonzero field proves
  deployed utility.
- `flow_jepa_progressive_g2_dynamic_center_distance` distinguishes genuine
  rematerialization around the corrected centre from merely adding a new
  logit prior to the old fine candidates. A zero value can be legitimate for
  a sample, but a run-wide exact zero alongside nonzero G1 centre movement is
  a structural warning.
- `flow_jepa_progressive_g2_dynamic_candidate_valid` measures geometric
  coverage after rematerialization. It is not a loss and must not be driven to
  one by hard clipping or a validity quota.
- G3 coarse/fine prior contraction minima are numerical bounds on selector
  logits, not route-mass targets. Entropy remains audit-only.
- `flow_jepa_progressive_world_source_prior_max` measures the W-owned
  horizon-specific posterior over G3 source-cell/slot states, while
  `flow_jepa_progressive_policy_world_prior_rms` proves that the paired bounded
  source prior reached P's only high-resolution value read. A healthy
  teacher-facing world relevance metric with a missing/zero policy-world prior
  means the posterior was logged but not connected to action.
- `grad_flow_dino_progressive_g1/g2/g3/world_query` must be interpreted with
  `grad_late_raw_detail_reader` and the raw/soft compiler gradients. Persistent
  zero G1/G2/G3 gradients under action training indicate a broken forward
  edge; different magnitudes are expected because the groups have different
  parameter counts.
- Use the V109 `address_g1/g2/g3_zero` and episode-shuffle modes. A changed
  internal posterior with unchanged action is attenuation or compensation,
  not a causal pass. The first aggregated high-resolution value remains the
  W->P late reader; G2 only materializes unaggregated candidates.

## V110 coordinate-typed current values and future transport

- `typed_raw=1` identifies the V110 graph. `literal_rgb` is the RMS of the
  fixed affine native-resolution current-RGB chart, not a learned feature norm
  or an area-pooled proxy. Read it with
  `p1_micro_value`, `p1_spatial_variation`, and `p2_detail_output`. These metric
  prefixes name the two W->P ingress stages, not the two subsequent policy DiT
  blocks. A nonzero chart with a zero ingress value path is a disconnected
  precision owner.
- Semantic, appearance and geometry logits are deliberately separate in G2,
  W and P. Different magnitudes are expected. A permanently exact-zero type
  with nonzero input RMS and nonzero sibling logits indicates a dead typed
  query/scorer, not successful specialization.
- `g3_sem_summary`, `g3_app_summary`, and `g3_geo_summary` are the three
  separate G->W token families. They may specialize to different scales, but
  an absent family or one fused-only `g3_summary` path violates the V110 source
  contract.
- `future_transport_offset` measures motion relative to the exact current
  anchor. `future_transport_variation` measures horizon differentiation.
  `future_transport_spatial_logit` proves that transport changed the W JEPA
  relevance/source posterior, while `p1_future_transport_logit` proves that
  it also reached the fine candidate read. Offset alone is not main-path
  evidence.
- `future_visibility` starts at 0.5 by construction. It is a smooth likelihood
  term, not a hard gate or target. Do not reward movement away from 0.5 without
  paired action/teacher evidence.
- Read `grad_flow_dino_progressive_future_transport` with
  `grad_flow_dino_progressive_world_query`,
  `grad_late_raw_detail_typed_p1_selector`,
  `grad_late_raw_detail_literal_rgb_value`,
  `grad_late_raw_detail_learned_detail_value`, and
  `grad_late_raw_detail_typed_p2_condition`. These are ordinary loss
  gradients, not injected gradient objectives. They distinguish address
  selection, the two value owners, and conditional local organization; the
  aggregate `grad_late_raw_detail_typed_p2` remains a size-dependent health
  summary rather than evidence that every lane is connected.
- `raw_value_zero` and the public probe mode `literal_current_rgb_zero` are
  intentionally separate (`literal_rgb_zero` is the latter's internal encoder
  intervention name).
  The former removes learned detail only; the latter removes literal current
  RGB only. Compare both and their spatial shuffles before attributing action
  utility to high-resolution observation content.
- `future_transport_neutral` and `future_transport_spatial_shuffle` isolate
  P's consumption of the learned transport while retaining the already
  compiled W source prior. They are not a full intervention on the W teacher
  relevance; interpret their stated boundary exactly.
- `flow_jepa_typed_p1_activation_checkpoint=1` records that the typed P
  micro-grid contraction is configured for activation recomputation during
  training. It is a memory-execution contract, not a learned gate, objective,
  pruning decision or evidence of task utility. The value should not change
  posterior outputs or gradients relative to the materialized reference.

## V111 functional evidence ownership

- `structured_ownership=1` identifies the opt-in V111 graph. The public G3
  token count must be `camera_count * grid^2`; `g3_owner_tokens` counts the
  three lower-width canonical typed key banks that W and P actually consume,
  not an extra full-width diagnostic copy.
- `g2_owner_sem_app_l1`, `g2_owner_app_geo_l1`, and
  `g3_owner_sem_app_l1` measure separation between soft owner posteriors. Zero
  is not automatically failure at initialization, but a persistent exact zero
  together with identical owner gradients/intervention deltas means the typed
  responsibilities have collapsed back to a shared selector.
- `g3_sem_owner_rms`, `g3_app_owner_rms`, and `g3_geo_owner_rms` are RMS values
  of the active canonical sidecars. They are not generic W memory. The generic
  path receives only the bounded public camera-spatial chart.
- `world_public_ratio` compares the horizon-shared W query component with its
  zero-mean horizon innovation. Read it with `world_innovation` and
  `world_owner_sem_geo_l1`; a large public ratio is descriptive, while a zero
  innovation or zero horizon/source variation is a temporal-path warning.
- `world_owner_slot_contract_min` and `world_owner_source_contract_min` are
  smooth finite-gain scale factors on the new private G3->W and W->P logit
  priors. Falling values indicate compression, not a learned route gate; they
  must be read with owner gradient and posterior-separation fields.
- `p1_owner_route_l1` and `p1_owner_fine_l1` audit owner-specific source and
  fine posteriors before the exact RGB/detail value read. They do not prove
  action utility; use the V111 semantic/appearance/geometry zero and shuffle
  modes and require a matched deployed-action delta.
- The five `grad_late_raw_detail_typed_p2_*_owner` fields must be interpreted
  as ordinary gradients through independent policy, semantic, appearance,
  geometry and horizon readouts. The owner readouts sum only after local value
  reading; none is an injected gradient or auxiliary loss.
- `[v111-owner-grad]` also expands the semantic/appearance/geometry G2 query,
  G3 slot and W query gradients. This prevents a healthy aggregate G2/G3/W
  norm from hiding one disconnected owner.

## Gradients and interventions

- Compare gradients by module and phase; do not compare a scalar parameter norm and a large-module norm as if they had identical dimensionality.
- Zero capacity gradients during execution warmup are expected. Persistent zeros after progress opens are suspicious.
- Layer adapter and consequence gradients establish whether ownership supervision reaches the intended interfaces.
- `z_zero` and `z_shuffle` are intervention deltas, not loss terms. Zero values are interpretable only when the active-path probe ran and coverage is nonzero.

## Coverage and comparison

- Full action validation may cover every batch while sampling gauges and proposal/execution ablations cover only a subset.
- Use `eval_sampling_diagnostic_coverage`, `eval_proposal_ablation_coverage`, and
  `eval_execution_ablation_coverage` in every conclusion about probes or ablations.
- Compare primary soft execution against matched-noise hard, neutral,
  full-capacity, and three-basis-reduction RMSE before crediting the controller.
- Cross-run comparisons require matching data split and action-normalizer fingerprint for native-coordinate anchors.
- Treat partial Stage1 loads (`missing`/`skipped` keys) as structural cold-start evidence.
