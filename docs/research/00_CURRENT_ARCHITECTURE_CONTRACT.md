# Current ClearVLA Architecture Contract

This file is the compact source of truth for the active mainline. It describes
only the graph that current source executes. Historical reasons belong in the
audit ledger; prospective ideas belong under `docs/research/auxiliary/`.

## Agent quick contract

```text
capability:             object_intent_dynamics_323
manifest schema:        36
topology:               G1 G2 G3 / W1 W2 / P1 P2 P3
training:               single-stage end-to-end
behavioral reference:   V120 long, commit 0b92d359a2889a0a1b1eba256007c00ccbc54f3c
checkpoint policy:      fresh run; Schema35 exact resume rejected
deployment integration: five Euler updates at 0,.2,.4,.6,.8
endpoint heads:         one full dynamic forward at 1.0, action not updated
```

Schema36 closes the dynamic-P1/P3 double exit and P2 owner mismatch exposed by
the failed Schema35 run. It does not add blocks, route quotas,
hard gates, entropy targets, artificial gradients, new external losses, or
bottom capacity. Pre-G, the V120 static high-resolution P1 reader, controlled
transition, Evidence MMDiT, CVAE/workspace, execution controller and action
heads remain the active main path.

The three rules to check before changing the graph are:

1. A tensor name is not ownership. Preserve its camera, K, interval, type and
   static/dynamic axes until the named consumer has used them.
2. Teacher data may alter targets and losses only. It cannot alter any online
   state or deployment action.
3. Protected factual bases stay outside optional routing. Optional effect and
   precision innovations must have an exact algebraic zero.

## Active graph in execution order

```text
current RGB/DINO/raw-pair/history + learned flow
  -> Pre-G observation bank
  -> G1: coarse hypotheses
  -> G2: rematerialized N=49 candidates
  -> G3: bounded conditional-K correction
  -> global K+null grounder -> GroundedFactSet

T5 goal tokens + state/executed-action history + G typed facts
  -> Stateless Intent Organizer S
  -> public interval carrier + typed common/residual + temporal control

G facts + S + one clean CoarseAction intent
  -> W1: protected common and intervals 4-8 / 8-16
  -> W2: intervals 16-32 / 32-48, conditioned on W1 near
  -> supervised FutureObjectDynamics only

G3 chart + S + clean action bases
  -> static P1, once per observation
  -> FactualPrecisionDock(protected_detail)

noisy-action query + static P1 + dynamic P1 query residual
  -> P2 effect query over supervised W fields
  -> consequence = static fact + W effect + interaction
  -> P3 precision/effect/temporal/state-change lanes
  -> controlled transition + retained bottom
  -> physical action field / event and motion heads
```

Training adds one detached plane:

```text
future DINO supports + current GroundedFactSet
  -> FP32/no-grad Teacher-G partial association
  -> FutureObjectDynamics target
  -> future loss only
```

## G and global K ownership

- DINO/content participates exactly once in the K+null base competition.
- Semantic and appearance are independently bounded conditional-K
  corrections. They can change which K slot owns a candidate, but cannot
  change content's object-vs-null mass.
- Geometry never votes on object identity. It reweights spatial support only
  after the physical K support exists.
- G3 preserves the G2 object/null decision and applies only a bounded,
  common-mode-free conditional-K residual. A zero G3 residual is an exact
  parent identity.
- Typed semantic/appearance/geometry reads use parameter-free reweighting
  inside the physical support. They cannot revive a zero-mass candidate.
- `DenseFactChart.g3_public_scene_audit` is audit-only. It is not a second
  public value path.

The final observable support fields are named by their actual semantics:

```text
chart_availability         [B,K,1]
camera_chart_availability  [B,K,C,1]
camera_evidence_mass       [B,K,C,1]
```

Availability is not existence probability, reliability, or a learned gate.

## Teacher target semantics

Teacher association is FP32 and no-grad. For every future support it forms a
fixed-dustbin partial assignment over real camera/cell candidates.

- Dustbin is match uncertainty, not physical disappearance.
- Semantic successor uses the exact identity fallback:

  ```text
  successor = matched_content + dustbin_probability * current_reference
  semantic_delta = successor - current_reference
  ```

- Visibility and persistence targets are exact zero until an independently
  observable occlusion/exit label exists.
- Future selector support is current chart availability, not dustbin,
  reliability, existence, or predicted visibility.
- Transport and covariance remain `[B,I,K,C,*]`. Dustbin is allocated over the
  currently observable camera measure as the zero-displacement hypothesis, so
  an ambiguous tiny real match cannot be renormalized into certain motion.
- Covariance is the full per-camera first/second moment including that identity
  component.
- Teacher uncertainty/reliability remain detached diagnostics only. They do
  not mask loss or enter P2 values.
- Intervals stay `4-8 / 8-16 / 16-32 / 32-48`; supports inside each interval
  use fixed uniform averaging.

Teacher builds once per training batch and zero times in deployment.

## Stateless Intent Organizer S

S reads only T5 goal tokens, observable state/executed-action history and G
typed facts. It cannot read frame progress, phase labels, noisy action or
Teacher fields.

It retains four interval identities and separate Goal/History/G innovations.
The small observable-state objective predicts adjacent interval increments:

```text
m0 = mean(state[4:8])
m1 = mean(state[8:16])
m2 = mean(state[16:32])
m3 = mean(state[32:48])

target = [m0-current_state, m1-m0, m2-m1, m3-m2]
```

The existing SmoothL1 coefficient is unchanged. Cumulative reconstruction is
audit-only. CoarseAction retains its own window-action supervision.

## W1/W2 causal field

W exports one `FutureObjectDynamics`; no generic W hidden is visible outside W.

- W1 owns common plus near intervals `4-8 / 8-16`.
- W2 may read W1 near and owns only far intervals `16-32 / 32-48`.
- W2 cannot process, interact with, or rewrite W1 common/near.
- Typed×base interaction occurs once after the corresponding generic owner:
  W1 common, W1 near, W2 far.
- The interaction is bias-free and starts as `1e-3 * identity`; a zero typed
  owner remains exact zero, while the protected object/goal base is not hidden
  behind two consecutive zero Jacobians.
- Four-interval residual closure is charged only to the two far rows:

  ```text
  near_final = near_w1
  correction = (sum(near_w1) + sum(far_raw)) / 2
  far_final = far_raw - correction
  ```

  The final floating-point closure is also charged to the last far row.
  Therefore `d(near)/d(far)=0`, while near may causally affect far.
- Semantic, transport, visibility and persistence are decoded only from their
  matching typed owners. Online uncertainty/reliability fields do not exist.
- W camera geometry is produced by a shared camera-equivariant head over typed
  geometry plus current camera coordinate, flow prior, support width and chart
  availability. It never predicts one object displacement and expands C.
- Predicted covariance is PSD with diagonal in `[(2/7)^2, 1]` and bounded
  correlation.

## P1, P2 and P3 ownership

Static P1 is the retained V120 reader:

- 24 factual queries;
- semantic, appearance, geometry and coverage glimpses;
- full N=49 posterior;
- one 3x3 RGB/detail/coordinate microgrid read;
- `[B,24,4,H]` factual output;
- constructed once per observation.

The dynamic boundary is explicit:

```text
CompletedP1PolicyState
  factual_base               = static protected_detail
  policy_query_residual      = dynamic action/time query refinement
  effect_query               = action + factual_base + policy residual
```

`factual_base` is independent of noisy action and time. The dynamic residual
conditions only P2's effect query. It cannot enter P3 precision, the protected
fact, transition factual source or bottom protected base. The inherited P1
policy block retains its ordinary residual computation, but its AdaLN
shift/scale crosses a smooth absolute bound of 4 before attention/FFN use;
raw and contracted modulation, Q/K, FFN input and every residual stage are
logged separately.

P2:

- reads supervised W semantic and geometry fields, matching S typed intent,
  the effect query and observable chart support;
- uses one normalized physical camera measure for both transport value
  reduction and camera-coordinate scoring;
- uses bounded covariance-aware camera mixture scores; coordinate changes K
  matching only and never votes directly on interval time;
- keeps semantic and geometry as complementary typed values with an exact zero
  null per type;
- owns one protected public-S interval prior, after which semantic and geometry
  each add only their matching typed-S and W evidence and select their own
  interval/null; there is no outer type competition;
- adds the two one-sided-bounded values directly, so either owner is preserved
  when the other is exact zero; the caller contracts the combined effect once;
- visibility/persistence remain neutral W supervision and diagnostics. Without
  an independent label they have no P2 action value or route vote;
- cannot reopen RGB/DINO or read generic W hidden.

Consequence is zero-preserving:

```text
effect = bias_free_project(P2_read)
interaction = bias_free_project(tanh(static_fact_projection) * effect)
protected_consequence = static_fact + effect + interaction
```

P3 has four active lanes:

- precision reads static high-resolution detail only, modulated by the current
  action query;
- effect reads only `W_effect + interaction`;
- temporal requires S temporal control, `W_effect + interaction`, and action;
- state-change reads independent observable state-change evidence.

With neutral W, effect and temporal are exact zero while precision and
state-change remain legal. The bottom source order and modules are unchanged.

## Typed interfaces

```text
GroundedFactSet / ObjectFactSet
  public_content                         [B,D]
  content / semantic / appearance / geometry [B,K,*]
  physical and typed candidate assignments  [B,K,C,8,8,M]
  camera_coordinates / transport_prior       [B,K,C,2]
  chart_availability                         [B,K,1]
  camera_chart_availability                  [B,K,C,1]

FutureObjectDynamics
  current_reference                    [B,K,D]
  successor_content / semantic_delta   [B,4,K,D]
  transport_mean                       [B,4,K,C,2]
  transport_covariance                 [B,4,K,C,3]
  visibility / persistence             [B,4,K,1]
  chart_availability                   [B,K,1]
  future_selector_validity             [B,4,K,1]  # diagnostic
  camera_coordinates                   [B,K,C,2]
  camera_chart_availability / weights  [B,K,C,1]

ObjectTopTrainingTargets
  teacher_dynamics                     detached FutureObjectDynamics
  current_loss_support                 [B,K,C,1]

CompletedP1PolicyState
  factual_base / policy_query_residual / effect_query [B,24,4,H]
```

K and camera permutations must be equivariant through G->Teacher->W->P2.

## Loss and optimizer ownership

- Action flow matching remains the primary objective.
- Grounding retains one dense reconstruction MSE with external weight `0.25`.
- S increment loss and CoarseAction loss retain their existing coefficients.
- Future semantic, transport, covariance, visibility and persistence keep their
  existing external budget. Removing online uncertainty does not redistribute
  its internal `0.10` coefficient.
- Semantic/status losses use object-reduced current support. Transport and
  covariance use full `[B,I,K,C,1]` current support.
- Dustbin, association reliability and selector validity never mask future
  losses.
- V120 optimizer decay ownership, decoder-local clip, global clip and first
  non-finite parameter sentinel remain unchanged.
- Every non-neutral trainable top parameter must receive ordinary autograd once
  initial zero-output boundaries have taken their first optimizer update.
  Visibility/persistence and their W appearance projection are the explicit
  exception at the exact zero target; perturbing either head must reconnect all
  three parameters through the neutral status loss.

## Runtime, identity and observability

- Observation, G, S, W, static P1 and transition source build once per
  observation.
- Dynamic P1, P2, P3, transition, layer contracts and bottom run at five Euler
  update nodes plus the clean endpoint-head forward.
- The endpoint forward cannot change the integrated action.
- Startup prints schema, manifest/source/git fingerprints, action/state
  normalizer fingerprints, total/trainable parameters and a compact G/S/W/P/
  bottom parameter summary. Full inventory is serialized in run context.
- Active static-P1 logs expose query rows, query chunk, N=49 candidate count,
  3x3 microgrid side/token count/value RMS and spatial variation.
- S logs increment prediction/target and cumulative audit error.
- W/Teacher logs retain per-interval semantic/transport, dustbin/reliability
  diagnostics and target RMS; no inactive uncertainty-loss metric is emitted.

Active manifest ABI:

```text
schema:       36
observation:  restored_v120_three_frame_flow_dino_progressive_g123_bank
top:          single_content_k_identity_incremental_stateless_intent_causal_w_near_far_camera_specific_effect_matched_semantic_geometry_p2_static_fact_single_precision_p3
bottom:       restored_v120_shared_seed_typed_bounded_dynamic_p1_query_only_four_active_plan_lanes_exact_g3_anchor_transition_evidence_mmdit_dense512_execution
training:     v120_mirrored_physical_flow_observed_current_grounding_partial_ot_neutral_status_camera_specific_future_loss_support_event_boost_v120_decay_three_owner_clip
runtime:      cached_observation_progressive_gsw_exact_p1_v120_nodes_clean_endpoint_teacher_isolated_active_ablations_only
```

Parameter counts are measured at launch and stored per module; they are not
hard-coded here. Any future count change must name the added/removed owner.

## Verification and run

Required local regression:

```powershell
$env:PYTHONPATH='.'
uv run pytest -q tests/test_mainline_action_field.py tests/test_mainline_structural_contracts.py tests/test_mainline_policy.py tests/test_mainline_manifest.py tests/test_mainline_checkpoint.py
```

Fresh smoke and long run:

```bash
CUDA_VISIBLE_DEVICES=0 \
OUT_DIR=runs/schema36_p1_p2_closure_smoke \
nohup bash scripts/smoke_mainline.sh > schema36_p1_p2_closure_smoke.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 \
OUT_DIR=runs/schema36_p1_p2_closure_b8 \
nohup bash scripts/train_mainline.sh > schema36_p1_p2_closure_b8.log 2>&1 &
```

Storage defaults:

```text
raw HDF5:    /data/liang.zhang/dataset/grab_pen_single/grab_pen_single
decoded cache:/data/senwang/data/cache_336
DINO cache:  /data/senwang/data/dinov2_cache_336
T5 weights:  /data/senwang/checkpoint/grasp_pen_embed.pt
batch/workers: 8 / 4
```

## Authoritative source map

```text
identity/config:       clearvla/mainline/manifest.py, config.py
observation/Pre-G:     clearvla/mainline/model/observation.py, restored_observation.py
G/global K:            clearvla/mainline/model/grounding.py
Teacher:               clearvla/mainline/model/teacher.py
S/CoarseAction:        clearvla/mainline/model/intent.py
W:                     clearvla/mainline/model/dynamics.py
P2/consequence/P3:     clearvla/mainline/model/compiler.py
static/dynamic P1:     clearvla/mainline/model/policy.py, restored_bottom.py, v120_p1.py
top orchestration:     clearvla/mainline/model/top.py
policy/bottom:         clearvla/mainline/model/policy.py, restored_bottom.py
loss/optimizer:        clearvla/mainline/training/losses.py, optimizer.py, engine.py
runtime/logging:       clearvla/mainline/runtime/, clearvla/mainline/train.py
audit:                 clearvla/tools/audit_policy_logs.py
open problems:         docs/research/CURRENT_MAINLINE_ISSUES.md
```
