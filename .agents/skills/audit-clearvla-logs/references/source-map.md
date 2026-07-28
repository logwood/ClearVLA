# ClearVLA Loss and Logging Source Map

## Contents

- Objective construction
- Backward and aggregation
- Validation
- Logging
- Static audit watchlist

## Objective construction

- `clearvla/experiments/observed_state_lab/policy_runtime_v36_3.py`
  - `flow_losses`: base physical flow, proposal, event, motion, decoded-action, delta, and gripper closure objectives.
  - `gripper_transition_metrics`, `event_head_metrics`, `motion_head_metrics`: validation semantics.
- `clearvla/experiments/observed_state_lab/policy_runtime_v39.py`
  - `rollout_dynamics_loss`, `rollout_delta_loss`, `milestone_delta_match_loss`: rollout targets and compatibility overlap.
  - `flow_losses`: execution value supervision, rollout/latent/controller objectives, and audit-only execution cost.
  - `midcut_contract_losses`, `layer_contract_losses`: stage/layer auxiliary objectives.
  - `_attach_v94_loss_ledger`: exact Evidence objective contribution ledger.
  - V101 future JEPA reductions normalize each horizon independently; batch and
    epoch logs expose the resulting temporal-allocation diagnostics.
- `clearvla/data/samplers.py` and
  `clearvla/experiments/observed_state_lab/dataset.py`
  - V101 `InformationBalancedBatchSampler` mixes a no-replacement uniform lane
    with bounded action-motion and gripper-event strata. Sampling signals use
    action/state arrays only and never inspect model predictions or images.
- `clearvla/policy/system.py`
  - V101 builds the teacher-change target mask after the online forward, applies
    complete-branch action-history/goal conditioning dropout during training,
    and supports deterministic teacher-forced representation validation.
- `clearvla/policy/flow_dino_evidence.py`
  - `_RawPyramidFlow`: V99 fixed RGB/census warp target, observable-motion
    balancing, zero-flow baseline, and conditional identity-advantage loss.
  - `_DenseRawFlowRefiner`: reliability changes search breadth; under the V99
    guard it does not multiply the coarse displacement toward identity.
  - `_RawDeformableAddressReader`: V99 flow-local detail lane versus pooled
    content fallback and address-separation diagnostics. Validation also runs
    zero-flow and shuffled-flow reader interventions without adding a loss.
  - V100 complementary mode: a complete low-frequency base is added to a
    flow-addressed high-frequency residual, then fused into the matching latest
    DINO chart with no learned amplitude gate.
  - V103 predictive mode shares one observation-derived mask across the early
    online RGB/DINO context and future-change loss, retains multiple soft
    address hypotheses, and passes source/target raw pair keys without placing
    source appearance in the target-only high-frequency value.
  - V104 `_smooth_bound_flow_to_image` maps every coarse/recurrent/mid/high
    proposal to a differentiable source-relative in-image chart.
    `_normalize_flow_evidence` changes only the learned motion K/V units;
    native geometry remains available to warp, address, and diagnostics.
  - V104 `_compose_future_queries` reads all observed pairwise perceptual
    motion and applies one shared attached transition in chronological
    +4/+12/+24/+48 order. Future teacher values and labelled actions are not
    inputs to this function.
- `clearvla/policy/role_delta_attnres.py`
  - `smooth_rms_contract` is the shared normalized-chart amplitude interface.
    `RoleDeltaAttnRes` routes contracted full-width values while using
    normalized selector keys; raw/written RMS and compression are diagnostics.
- `clearvla/policy/trunk_primitives.py`, `clearvla/policy/trunk.py`, and
  `clearvla/policy/time_domain_mmdit.py`
  - V100 strict mode disables policy-block raw visual cross-attention, removes
    the final decoder's visual bank/intent duplicate, and exposes only terminal
    policy contracts to the decoder while retaining their original depth ids.
  - V101 also removes the hidden raw-visual mean from policy AdaLN modulation;
    the final Evidence decoder receives policy workspace through fixed
    variance-preserving fusion instead of the historical `0.10` addition.
  - Forward-flow raw reads remain source-grid indexed and V101 fuses them with
    the matching preceding/source DINO chart; latest DINO stays an independent
    current-coordinate chart.
  - The V101 action-path probe applies world residual interventions between the
    world and policy block groups. Schema v3 has independent future-anchor and
    xy shuffles while preserving the grounding/position seed.
  - Schema v3 post-reader detail interventions define the high-frequency
    interface residual after `selector_out`/`value_out`, keep the base-only
    reader output and DINO fixed, and either remove or spatially misalign only
    that residual. They are evaluation-only and never enter backward.
  - V102 compiles that exact residual from RGB/source/motion only, caches the
    compressed observation bank for counterfactuals and all ODE steps, and
    defers action/world-conditioned selection to the world-to-policy boundary.
  - V102 structures every world residual writer after dropout as
    `[anchor,camera,mean_xy]`, reads each camera's raw chart independently at
    the late boundary, and combines cameras with a fixed coefficient.
  - V102 policy workspace alignment lifts each `[time,basis]` token before
    pooling basis within time; it never interpolates over flattened
    `time*basis`.
  - V103 G->W, W->P, and P->MMDiT bridges route typed block deltas rather than
    cumulative hidden states. The carrier stays outside the selector softmax
    and protected detail remains a separate bottom lane.
  - V104 contracts every actual G/W/P residual sublayer before its legal role
    addition and contracts typed values at every Delta AttnRes ownership
    boundary. Limits are fixed in the normalized chart and cannot be enlarged
    by the receiving carrier.
  - V105 reads the existing observation-only soft lattice with one
    horizon/camera/W query chart, keeps fine candidates continuous, and adds a
    fixed small residual only to the future JEPA predictor. Frozen future
    change supervises the relevance logits in the loss and never enters the
    forward reader.

## Backward and aggregation

- In `train_v39_policy`, inspect the construction of `total_loss`, the auxiliary scale, assignment back to `losses["loss"]`, ledger attachment, `.backward()`, gradient diagnostics, clipping, and optimizer step in that order.
- `_sync_loss_row` supplies batch console values; `_accumulate_metric_tensors` and `_finalize_metric_tensors` supply epoch means.
- The epoch JSONL is the archival record. Compact console lines are a decision-oriented projection.

## Validation

- `evaluate_v39_policy` uses deploy-style sampling for action metrics.
- Sampling interventions and proposal ablations have independent budgets and coverage.
- Layer-contract teacher-forced evaluation is separately labeled and must not be compared directly with sampled action RMSE.

## Logging

- `_evidence_serial_log_line`: compact V94 batch loss/execution/gradient lines.
- `_evidence_epoch_log_line`: compact epoch, validation, and intervention lines.
- `_evidence_log_version`: derives V103/V104/V105/V106 console identity from active
  structural contract fields; serialized run context and source fingerprint
  remain the authoritative identity.
- `_filter_inactive_evidence_epoch_metrics`: removes inactive zero placeholders from Evidence epoch records.
- The historical fallback `[v39-layer]` formatter is broad and placeholder-heavy; use it as raw evidence, not as a recommended schema.
- `clearvla/tools/audit_policy_logs.py`: normalized parser, objective budget, trends, rule findings, and run comparison.
- `clearvla/cli/eval_v39_policy.py`,
  `scripts/run_v104_model_path_probe.sh`,
  `scripts/run_v105_model_path_probe.sh`,
  `scripts/run_v106_model_path_probe.sh`, and
  `clearvla/tools/summarize_v101_action_path_probe.py`: a formal V104/V105/V106
  checkpoint probe requires the matching complete validator, emits a distinct
  schema/identity field, and is accepted by the shared causal-probe summarizer.
- `scripts/current_v106_interval_stage_flow_jepa.sh` is the formal V106
  launcher. `_validate_complete_v106_model_contract` requires the complete
  V105 parent, exact interval supports/boundaries, variance-safe routing, the
  complete numerical contract with minimum role/correlation/visibility floors,
  and a positive interval-stage objective before optimizer construction.
- `_preflight_evidence_dynamic_sampling` also builds and validates one real
  V106 interval target pack before printing success. It covers the future
  teacher path that deploy-only sampling cannot execute.
- `scripts/current_v108_online_horizon_address_flow_jepa.sh` retains the V108
  single G3->W1 address-write baseline. The wrapper now uses inheritable
  version/contract defaults so a formal child cannot be mislabeled V108.
- `scripts/current_v109_progressive_grounding_address_flow_jepa.sh` is the
  formal V109 launcher. `_validate_complete_v109_model_contract` requires the
  complete V108 ancestry plus the progressive-address flag.
- `scripts/run_v109_model_path_probe.sh` selects the V109/v8 probe contract
  and the progressive checkpoint path.
- `flow_dino_evidence.py::_ProgressiveGroundingAddressOrganizer` owns the
  typed G1/G2/G3 selector transitions. The soft compiler's
  `progressive_fine_candidates` rematerializes unaggregated candidates around
  G2 centres. At W->P the organizer forms teacher-target and source-state
  marginals from one W/G3 compatibility tensor; the source-state marginal is
  consumed by `trunk.py::LateRawDetailPolicyReader`, which performs the first
  final value aggregation.

## Static audit watchlist

Re-check these against current source rather than assuming they remain true:

1. `rollout_delta_loss` and `milestone_delta_match_loss` can become the same formula when `milestone_step_delta_pred` is present. Enabling both weights duplicates supervision.
2. Compatibility metrics named `future_latent` and `action_effect` are aliases; source determines which tensor their weights actually optimize.
3. The exact ledger is currently attached to the Evidence policy path, while other decoders rely on estimated reconstruction.
4. The layer ledger exposes the total weighted layer contribution but not every internal layer sub-objective as a separately named exact contribution.
5. Event-head accuracy is highly vulnerable to hold-class imbalance; decoded event counts are the deployment-facing closure check.
6. Compact console output is intentionally incomplete; use `v39_policy_epochs.jsonl` for exhaustive epoch fields.
7. Raw metric equality, constant norms, or zero lateral paths may be architectural invariants. Confirm source semantics before calling them collapse.
8. V97/V98 and V99 raw address mass use different candidate semantics and are
   not directly comparable. Use zero-flow warp gain and moving/static splits for
   the controlled identifiability comparison.
9. V100 `raw_detail_share`/`raw_base_share` are projected additive-energy
   shares, not a router probability. Verify `raw_dino_fused`, all three role
   gradient groups, and matched action interventions before crediting flow.
10. V101 balanced sampling changes the training distribution but not validation.
    Compare full sequential validation and report the sampler summary; do not
    infer population improvement from training loss alone.
11. V101 future teacher change owns only loss-mask selection. Any occurrence in
    online conditioning, query construction, or policy inputs is future leakage
    and should be treated as a structural regression.
12. Under V102, a nonzero world xy residual, missing late-detail sampling
    metrics after a multi-step decode, or absent late-reader/raw-reader
    gradients is a structural regression rather than ordinary slow learning.
13. Under V103, large world xy residuals are legal, but simultaneous unbounded
    raw flow, falling valid fraction, negative moving warp gain, exploding
    world residuals, and a collapsing detail/trajectory ratio form one
    coordinate-plus-amplitude failure chain. Do not mislabel it as merely weak
    address entropy.
14. A formal V104 run requires all three new flags. Verify config -> module
    construction -> forward metric -> serialized run context -> launcher
    validator; a version prefix alone is insufficient.
15. V104 boundary compression and residual/value compression are factual
    saturation diagnostics. Never add them to total loss without a separate
    architectural decision and controlled experiment.
16. V104 sequential state exists only inside one forward call. Any persistent
    episode cache, target-future input, action-label input, hidden detach, or
    independent parallel latest-motion seed is a structural regression.
17. V105 future teachers may appear only in
    `_flow_jepa_horizon_address_terms` and future-loss diagnostics after policy
    forward. Any teacher tensor in `_HorizonSoftAddressJEPA`, the observation
    bank, W query construction, trajectory, or bottom decoder is leakage.
18. In V105, dividing reliability-weighted normalized loss by reliability
    mass cancels the intended attenuation when all teacher changes are weak.
    The denominator must remain the valid-position count.
19. V105 reliability must not use only the future delta's own RMS as its
    reference: uniform rescaling would leave the ratio unchanged. Verify the
    current-teacher-relative smooth normalization scale and compare
    `future_target_delta_scale`, `future_current_reference_scale`, and
    `future_normalization_scale` together.
20. The V105 horizon reader's value path must map an all-zero fine-value bank
    to an exact zero update. A learned normalization bias or value-projection
    bias would create an address-independent future residual shortcut.
21. A bounded V106 residual write does not bound the normalization that
    produced its attention/FFN input. Verify the active learned-correlation
    denominator, continuous-vs-hard occlusion split, and G/W/P internal
    normalization denominator/gain metrics. A delayed simultaneous rise in
    cycle error, occlusion, and G/W/P preclip without a schedule boundary is a
    numerical interface transition, not evidence that the residual cap worked.
22. In V109, reusing only `SoftAddressLatticeBank.fine_keys/fine_values` after
    G1 moves a complete-chart mode is a false address connection: priors can
    change while the actual high-resolution candidates remain at the old
    centre. Verify the dense-chart fields, G2 dynamic-candidate metrics, and
    that the P reader consumes the progressive state's dynamic candidates.
23. V109 must not call `_HorizonSoftAddressJEPA` on the deployed path. The
    parent `flow_jepa_online_horizon_address=1` remains serialized only for
    ancestry; the progressive flag suppresses the V108 G3->W1 value read and
    the V107 late auxiliary value read.
24. Under V109, a nonzero
    `flow_jepa_progressive_world_posterior_entropy` is not main-path evidence
    by itself. Verify the paired source prior, its horizon variation,
    `flow_jepa_progressive_policy_world_prior_rms`, and an action intervention.
    The old `_HorizonSoftAddressJEPA`, compiler-only target-value projection,
    and compiler-only coarse-geometry projection must not remain trainable dead
    branches in the progressive graph.
25. V110 stores literal current RGB, learned detail, DINO semantics, raw-pair
    appearance and geometry as separate fields through G3. The derived
    combined compatibility key is ancestry-only; W and P must score the typed
    keys with distinct projections, and the G->W bridge must expose separate
    semantic/appearance/geometry summary-token families. A source review that
    finds only renamed fields feeding one shared scorer or one fused summary
    token has not implemented the V110 contract.
    `dense_current_rgb` must retain the input image side; resizing it to the
    learned raw-feature side before normalized-coordinate sampling is a
    precision bottleneck, not a harmless cache optimization.
26. V110 future transport is not complete if it is only concatenated into a
    late MLP condition. Its bounded spatial compatibility must alter the W
    teacher relevance and paired W->P source bias, then alter P1 fine routing.
    Future centres are distributions attached to current observed anchors;
    they never index a future raw image.
27. The V110 W->P ingress-A value path (logged with the historical `p1_*`
    prefix) must preserve a 3x3 micro axis for literal RGB and learned detail.
    Coordinates and typed keys may affect attention queries or keys but cannot
    manufacture a value: jointly zero RGB/detail must produce an exact zero
    ingress-B update before the actual policy P1/P2 DiT blocks.
28. In typed V110 mode, the replaced V109 G2 rectifier/G3 scalar scorer/G3
    summary remain serialized for ancestry but must be frozen. Gradient
    diagnostics must include the typed G2/G3 queries, typed W queries, future
    transport, and the P2 local refiners rather than reporting the frozen
    parent modules as a false zero-gradient failure.
29. The typed G2 rectifier has exactly four consumed outputs: x/y correction,
    support log-scale, and geometric-prior strength. Three outputs are an
    indexing failure; more than four reintroduce an optimizer-owned dead
    channel. Audit the per-output gradient test rather than inferring use from
    the module-level norm.
30. V110 learned projections (and their adjacent key normalization) at G2, W,
    and P must run in the active autocast domain. Explicit
    `autocast(enabled=False)` islands begin only after projection and receive
    `.float()` activations for similarities, geometry, reductions and
    probabilities. A Float32 `Linear` called inside such an island with a BF16
    activation is a runtime contract failure even if an FP32 unit test passes.
31. V110 must not materialize a full
    `[B,Q,glimpse,C,G,G,slot,candidate,micro]` posterior and then a
    `[B,Q,glimpse,C,G,G,slot,micro,value]` tensor before coarse aggregation.
    Stream or fuse the micro contraction and activation-checkpoint the typed P
    query chunk. This is an associative evaluation-order repair: route/fine
    probabilities, 49 candidates, nine micro cells, typed value ownership and
    natural gradients must remain unchanged. Lowering the grid, candidate
    radius, slot count or micro-grid is not an equivalent memory fix.
32. V111 makes G2 ownership functional: semantic produces its own hypothesis
    posterior, appearance produces its own verification posterior, geometry
    owns coordinate expectation, and only appearance plus geometry form the
    final fine localization posterior. Reusing that joint posterior as the
    appearance sidecar is a false separation.
33. V111 G3 generic visual memory contains one bounded public
    camera-spatial chart. The active private sidecars are the three canonical
    route-width key banks and their typed slot posteriors. Do not retain an
    unconsumed full-hidden owner-summary tensor merely for logging; it wastes
    memory and makes the sidecar count semantically false.
34. V111 W transport is chronological interval composition. Appearance cannot
    vote on final coarse transport/source relevance, but its private G3 slot
    posterior must still reach its matching W source sidecar and P appearance
    verifier. Otherwise the appearance G3 scorer is an audit-only dead branch.
35. V111 P uses a joint semantic/geometry source posterior and an
    appearance/geometry fine posterior for the precision value read, while
    retaining separate typed owner posteriors for the five local P2 readouts.
    RGB/detail are the only values; coordinates and modality identities may
    affect keys but jointly zero RGB/detail must still yield exact zero.
