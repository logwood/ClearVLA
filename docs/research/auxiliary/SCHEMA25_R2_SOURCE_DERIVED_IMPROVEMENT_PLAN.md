# Schema25 R2 source-derived improvement plan

Status: `R2-D01/A01 VALIDATED; WG01/P202/GRIP02 LOCALLY CLOSED; ONE-BATCH CPU BF16 SMOKE PASSED; GPU BEHAVIOR UNRUN`

Attribution checkpoint (2026-08-27): the behavior-preserving diagnostic
surface records P2 `type x action-band x physical-interval` posterior mass,
expected interval, support and band-pair total variation on the existing
bounded deployment diagnostics. Full validation records gripper RMSE by action
band, target-event-relative persistence, both deployed codec branches and an
exhaustive `action-band x event-context` error partition. R2-A01 additionally
implements four evaluation-only, matched-noise P2 value counterfactuals:
semantic/geometry crossed with near intervals `0/1` or far intervals `2/3`.
They alter selected common and residual values only after posterior formation;
the primary path, model state, loss, optimizer, checkpoint ABI and RNG stream
are unchanged. The retained mainline suite passes `164/164`. The completed
replay covers all `179` validation batches / `1,432` samples and the configured
`16` matched counterfactual batches. A dedicated
read-only validation loader accepts only the intentional active-source identity
difference while requiring identical manifest, semantic config, dataset,
language artifact and complete model-state ABI.
`scripts/validate_mainline_checkpoint.sh` loads no optimizer, schedule or
checkpoint RNG state and writes no checkpoint. The 24-row event histograms,
P2 posterior intervention, projected event-source intervention and P3
post-route/intervention diagnostics remain unimplemented; A01 no longer needs
them to choose the next source boundary.

Implementation checkpoint (2026-08-28): all three authorized behavioral units
are present in the local worktree. WG01 changes only transport
common/innovation supervision to the existing target-scale-covariant row loss
and preserves the R1 raw-coordinate value as a detached audit. P202 gives
spatial K/K*C selection and physical-I termination separate bias-free query
owners initialized as exact copies. GRIP02 adds one exact-zero bounded
gripper-private state shared only by deployed value/delta and the supervised
final event head. No P3, codec, sampler, checkpoint-selector, config, objective
coefficient or execution-policy edit was made.

The measured production inventory is `169,199,006` total / `152,828,275`
trainable parameters, 1,389 parameter tensors, 1,067 trainable tensors, 23
optimizer groups and 1,395 state keys. Relative to R1 the delta is exactly
`+786,432` trainable parameters and three state keys; the post-construction RNG
digest remains
`d3bcc995a57b40e359a6370a4dc3eea1638fa4a210f3082e41f6791a75513c21`.
The retained suite passes `168/168`; focused Ruff, compileall and diff checks
pass. A fresh one-batch CPU BF16 forward/backward at the complete production
dimensions (`H=512`, 256 patches/camera, 169,199,006 parameters) completed in
109.58 seconds locally with finite loss `3.18999` and finite pre-clip gradient
`4.22802`; all seven new scalar fields were present and finite. The retained
five-step deployment guard also passes. First-forward spatial/terminal P2 VJPs,
query delta and gripper gate/state delta were exact zero at the inherited
zero-value/exact-copy/zero-gate initialization, while the gripper-private state
VJP was nonzero (`4.94517e-6`). These are implementation guards, not
CUDA-memory, throughput or formal R2 behavior evidence.

This document records the source derivation and execution contract for the next
version after the completed Schema25 R1 replay and its first formal run. The
three source-backed units have passed their producer-to-consumer review and
local implementation boundaries; the compact adopted facts are also reflected
in `00_CURRENT_ARCHITECTURE_CONTRACT.md`. Because a separate remote smoke is
temporarily unavailable, the user explicitly authorized the local guard as its
substitute. Formal acceptance still requires the complete CUDA run and its
runtime/memory evidence.

The version is deliberately not defined as "more validation fields."  The
validation additions are a low-cost attribution layer around later structural
changes. The next training candidate contains three independently reversible
units--geometry supervision, P2 query ownership and continuous gripper-state
closure--because each now has a distinct source defect. It does not include a
P3, codec, sampler, checkpoint-selector or generic loss-weight bundle.

## 1. Executive decision

The next version is assembled as three independently auditable source units in
one formal training candidate, following the user's decision to complete the
three units together:

1. **R2-WG01:** make camera-transport supervision covariant to the target's
   normalized-coordinate scale without changing W forward values, Teacher,
   coefficients or runtime;
2. **R2-P202:** separate the action projection used for spatial K/K*C selection
   from the projection used by the physical-I terminal, with an exact-copy
   initialization and no time prior;
3. **R2-GRIP02:** introduce one zero-initialized, bounded, gripper-private
   continuous state shared by deployed value/delta heads and the supervised
   event head, while leaving arm, auxiliary gripper coordinates, codec and
   event logits-as-probe semantics intact;
4. update only the manifest component identities, source/state inventories,
   tests and a small JSONL-only health surface required by those units;
5. spend one formal training run after all three commits independently pass
   forward, reverse, identity and checkpoint-boundary review.

The three units are not one causal claim. They touch different producers,
losses and consumers and must remain separately revertible. Combining them in
one run is an experiment-budget decision. Per-unit health is read from its own
pre-registered metrics; the final decision remains task behavior, especially
far/event-context gripper error with arm and near-action guards.

## 2. Evidence ledger

The categories below must remain separate.  A source fact is not automatically
a causal diagnosis, and an observational correlation is not authorization for
a repair.

### 2.1 Source-confirmed facts

#### Data and target boundary

- `ObservedStateDatasetConfig.action_offset` is zero.  Every target contains
  24 action rows beginning at the current center.
- Row 0 uses current `qpos`/`action_state` as its boundary; later rows use the
  preceding action.  The sampler, physical codec, event loss and validation
  decoded-event metric use the same boundary convention.
- The retained dataset probe for the active grab-pen data reports exactly zero
  gripper difference between `qpos[t]` and `action[t]`.  Its first-event
  histogram has no row-0 entry.  There is therefore no current evidence for a
  row-0 pseudo-event, and R2 must not change this boundary pre-emptively.
- In that probe, event steps are about `3.027%`, hold steps about `96.973%`, and
  an event run averages `1.0076` rows.  The task is consequently dominated by
  carrying the state created by a short transition through later hold rows.
- Events can originate inside the same coarse open/closed bucket.  Continuous
  gripper value remains primary; a hard binary state penalty or gate is not
  justified.

Relevant sources:

- `clearvla/mainline/data/dataset.py`
- `clearvla/cli/probe_gripper_dataset.py`
- `docs/research/dino_control_probe_gripper_dataset.json`

#### Gripper physical field and output heads

- `PhysicalActionFieldCodec` supervises six gripper coordinates:
  absolute value, adjacent delta, value relative to current state, previous
  value, absolute delta and positive delta.
- Deployment decoding reads only the first two coordinates.  It combines the
  absolute branch with the current-state-plus-cumulative-delta branch using
  fixed weights `0.75 / 0.25`.
- `ActionOnlyPhysicalVelocityHead` already owns separate value, delta and four
  auxiliary gripper projections.  All six receive physical flow supervision;
  only value and delta determine the decoded action.
- The decoded action loss, smooth-delta loss and physical-delta consistency
  loss all reach the deployed continuous gripper.  The final event cross
  entropy is a separate auxiliary term.

Relevant sources:

- `clearvla/mainline/model/action_codec.py`
- `clearvla/mainline/v120_core/decoder.py`
- `clearvla/mainline/training/losses.py`

#### Two different objects currently carry the name "event"

- The action-consumed `event_evidence` is a frozen weak three-coordinate
  readout of layer-contract rollout delta.  Its residual adapter and downstream
  projection are trainable through action losses, but the readout is not
  supervised by the gripper event target.
- That layer-contract canvas reads state, state history, executed history and
  the transition selector.  It does not read S/W/P, the current noisy action,
  flow time or the final gripper event logits.
- The supervised final event head reads final action hidden states.  Its loss
  can shape the shared decoder representation during training, but its logits
  are a sibling output, not an input to the physical velocity head.
- Deployment performs an endpoint head call at `t=1`; that call reports final
  event/motion logits and deliberately does not modify the integrated physical
  field.

Thus there are two non-equivalent event meanings with different producers,
gradients and consumers.  This is a confirmed semantic ownership split.  It
does **not** yet prove that either path causes the gripper regression.

Relevant sources:

- `clearvla/mainline/v120_core/layer_contracts.py`
- `clearvla/mainline/model/restored_bottom.py`
- `clearvla/mainline/v120_core/time_domain_mmdit.py`
- `clearvla/mainline/v120_core/decoder.py`
- `clearvla/mainline/runtime/sampling.py`

#### P2 action-time to World-interval terminal

- R1f first performs semantic K and geometry K*C spatial selection separately
  for every World interval, preserving `[B,T,Q,I,Z,H]`.
- It then computes one no-null softmax over the four physical World intervals
  for each action row, basis and type.  The query contains explicit action-row
  identity; the selected W key contains interval identity and is conditioned by
  already selected S metadata.
- The terminal is soft all-to-all.  There is no hard chronological mask and no
  fixed row-to-interval assignment.
- Current `object_p2_interval_i_mass` averages the posterior over action rows,
  bases and types.  It cannot reveal which World interval a near or far action
  band uses.
- The P2 semantic and geometry values add before the single caller-owned
  contract.  Their independent survival is a locked R1f ownership property,
  not evidence that their time allocation is correct.

Relevant source:

- `clearvla/mainline/model/compiler.py`
- `docs/research/auxiliary/R1F_P201_SPATIAL_PHYSICAL_TERMINAL_WORKSHEET.md`

#### P3 and bottom ingress

- R1g keeps temporal and state-change as the only optional P3 lanes.  Each is
  independently contracted to RMS `0.35`, independently routed over its four
  bases plus null, and then the two routed values are added.
- The optional-lane sum and protected P1 precision read are jointly multiplied
  by the inherited bottom scale `0.25`.  Protected consequence/detail remains
  separate.
- Present metrics expose pre-route lane RMS and aggregate routed RMS, but not
  each post-route lane, their cosine/cancellation, or their gripper-band action
  effect.
- R1g combined ownership deletion, removal of the old state-change `0.05`
  multiplier, removal of two `/sqrt(2)` divisors, and lane-local routing in one
  source unit.  A later amplitude repair must not silently rebundle those
  choices.

Relevant sources:

- `clearvla/mainline/model/compiler.py`
- `clearvla/mainline/model/restored_bottom.py`
- `clearvla/mainline/v120_core/time_domain_mmdit.py`
- `docs/research/auxiliary/R1G_P301_B01_UNIQUE_LANE_BOTTOM_INGRESS_WORKSHEET.md`

#### Validation and checkpoint selection

- Validation currently reports aggregate gripper RMSE and aggregate
  near/mid/far RMSE, but not their intersection.
- `SamplingResult.physical_field` already retains the final 18-D field, so
  absolute-versus-cumulative gripper branch diagnostics require no additional
  model forward.
- `best.pt` is selected only by seven-dimensional average normalized action
  RMSE.  Gripper contributes one of seven native dimensions and far gripper
  state has no explicit selection authority.  `latest.pt` is also saved.

Relevant sources:

- `clearvla/mainline/runtime/evaluation.py`
- `clearvla/mainline/runtime/sampling.py`
- `clearvla/mainline/train.py`

### 2.2 Run observations

These are directional behavior observations, not bit-exact attribution to one
R1 source slice.

- Schema25 base at epoch 8: physical action `0.078874`, arm `0.062689`,
  gripper `0.141310`, decoded event F1 `0.418327`.
- R1 at epoch 8: physical action `0.080177`, arm `0.057911`, gripper
  `0.157721`, decoded event F1 `0.393925`.
- R1 therefore improves arm while worsening gripper.  From R1 epoch 6 to 8,
  decoded event F1 rises while gripper RMSE rebounds, so final event F1 is not
  a sufficient proxy for persistent gripper state.
- Across R1 epochs 1-7, W2 geometry-state RMS remains `0.36-0.46` and its
  interval variation remains `0.24-0.29`; the geometry carrier is not zero or
  interval-collapsed.  Predicted transport remains only `0.022-0.031` against
  Teacher `0.087-0.096`.  At epoch 7, geometry common/interval input VJPs are
  about `40-45x` below semantic despite an active dynamics gradient group.
- A01 shows semantic allocation does vary by action band: expected interval is
  `1.144` for rows 1-4 and `1.658` for rows 13-24, with band-pair total
  variation `0.211`.  Geometry allocation is almost flat (`0.0176`).
- Removing semantic intervals `2/3` raises matched far-gripper RMSE from
  `0.140827` to `0.271796`; removing geometry intervals `2/3` changes it only
  to `0.140556`.  Far semantic evidence is useful, while learned geometry has
  almost no action authority at its current magnitude.
- Full validation gripper RMSE is `0.059981 / 0.125307 / 0.195153` over the
  three bands.  Absolute-only is `0.061225 / 0.125126 / 0.192019` and
  cumulative-delta-only is `0.062570 / 0.137393 / 0.224843`; both fail before
  blending and the delta branch is worse at long horizon.
- Event and post-event rows dominate gripper squared error.  Error is largest
  around the event and declines with event distance, but each event-context
  category becomes harder in later absolute action bands.
- R1 P3 temporal/state-change RMS changes smoothly and does not show a sharp
  late-epoch explosion.  P3 is not part of this R2 behavioral candidate.

### 2.3 Current inferences

1. The upstream World stream has not collapsed, and the specific claim that
   far semantic P2 values are unused is false.
2. Geometry is weak after a healthy hidden carrier because raw small-coordinate
   regression makes its direct supervision and boundary VJP scale far below
   semantic.  This is a loss-geometry problem, not evidence for a hard output
   gain or deletion.
3. P2 already learns useful time allocation, but one `source_query` parameter
   set simultaneously owns spatial K/K*C selection and physical-I termination.
   That ownership overlap is independent of whether interval 3 mass is high.
4. The codec blend is not the primary gripper failure.  Both deployed branches
   are poor, and the supervised event head has no gripper-private continuous
   state shared with those branches.
5. Local event classification and persistent continuous state remain distinct.
   The repair may let the existing event loss shape a shared gripper state, but
   event logits remain a probe and never gate the physical field.

## 3. Complete active dataflow maps

### 3.1 Time-evidence path

```text
current observations
  -> G object chart
  -> S typed relevance and public interval context
  -> W common + four interval innovations
  -> P2 type-local spatial K / K*C selection, retaining I
  -> P2 action-query terminal removes I independently for semantic/geometry
  -> complementary raw sum
  -> one P2 RMS contract
  -> protected consequence
  -> P3 temporal/state-change optional lanes + protected P1 carriers
  -> bottom evidence/action decoder
  -> six gripper physical coordinates
  -> 0.75 absolute + 0.25 cumulative-delta decode
  -> 24 native action rows
```

Backward paths:

- action, decoded-action, event, motion and execution losses can reach P2 and
  W through the bottom shared representation;
- future Teacher losses reach W directly but do not supervise which action row
  selects which W interval;
- no explicit loss names a correct action-band/World-interval relation.

`top.effect_reader.*` is one checkpointed module and one `p2_effect_reader`
optimizer role.  Its semantic and geometry spatial choices use independent
keys/values but the same two type-local action projections are currently reused
for the later I terminal.  W is cached once per observation; the action-query
dependent P2 read is repeated by the five sampler updates and the endpoint
model call.  P1, dynamic precision, P3 and transition evidence are legal
parallel bottom inputs, so action success does not prove P2 ownership closure.

### 3.2 Event and persistent gripper path

```text
raw gripper action sequence + current qpos
  -> per-row signed delta threshold
  -> three-class final event target
  -> final event head loss
  -> shared bottom hidden state
       \-> event logits (reported only)
       \-> gripper value/delta/aux physical velocity heads
             -> five ODE updates
             -> absolute branch + cumulative-delta branch
             -> fixed continuous decode
             -> decoded event metric and gripper RMSE

transition.selector
  -> trainable layer adapter
  -> frozen weak three-coordinate readout
  -> trainable event projection
  -> action-consumed evidence bank
```

The lower branch is not supervised by the final event target.  The upper final
event logits are not a deployed control input.  Any R2 event claim must state
which branch it means.

During a training forward, physical flow, decoded action, smooth-delta and
physical-delta losses reach the continuous velocity head; event focal loss
reaches the sibling final event head, and both also reach their shared action
trunk.  In sampling, the velocity head is additionally called for execution
candidate/prefix values inside one bottom invocation; only the final physical
field from each of five ODE updates is integrated.  The endpoint call reports
event/motion without another physical update.  All action/event/motion heads
are checkpointed under `bottom.decoder.*` and share optimizer role
`bottom_heads`; arm heads, four auxiliary gripper heads, the motion head and the
layer-contract rollout event evidence are legal alternate consumers/bypasses.

### 3.3 Camera-geometry producer and reverse path

```text
two adjacent learned image flows
  -> normalized per-cell flow_xy [B,C,Y,X,2]
  -> geometry-conditioned object read
  -> camera transport prior [B,K,C,2]
  -> S geometry common/residual metadata
  -> W1/W2 typed geometry hidden state [B,I,K,H]
  -> camera-specific zero-preserving conditioning [B,I,K,C,H]
  -> shared transport head
  -> 0.50*tanh(common) + 0.50*tanh(interval innovation)
  -> transport [B,I,K,C,2]
       \-> direct future transport loss
       \-> P2 geometry key/value and coordinate score
       \-> camera-specific Teacher association prior (detached target path)
```

Teacher forms same-camera displacement moments in normalized coordinates and
contributes no deployment value. The direct loss decomposes common and
interval innovation but currently applies raw-coordinate SmoothL1 only;
semantic uses raw plus target-scale-normalized and directional terms. Current
camera support is detached for loss masking. Action losses can also reach W
through P2, while `future_dynamics` reaches W directly. All W parameters are
owned by optimizer role `dynamics`; transport has no separate checkpoint
field, and W is computed once per observation cache while P2 consumes it at
each of five ODE updates plus the endpoint head call.

Invalid camera support gives exact-zero transport and no P2 geometry value.
At legal support the transport head is near the origin, so tanh saturation is
not the observed bottleneck. Alternate action paths--semantic P2, factual P1,
dynamic precision, P3 state change and transition--remain legal and explain
why weak geometry need not make the whole policy weak.

## 4. Structural issue register

| ID | Status | Issue | R2 disposition |
|---|---|---|---|
| `R2-WG01` | implemented; local guards pass; behavior unrun | W geometry hidden state remains healthy, but raw small-coordinate transport loss yields a roughly `40-45x` semantic/geometry boundary-VJP gap and prediction stays near one third of Teacher | existing exact-zero target-scale-covariant row loss now owns transport; forward, Teacher and coefficients remain unchanged |
| `R2-P202` | implemented; local guards pass; behavior unrun | one `source_query` set owned both spatial K/K*C selection and physical-I termination | independent terminal query is an exact parameter copy at construction; no time prior or interval target |
| `R2-GRIP02` | implemented; local guards pass; behavior unrun | both deployed branches fail, while final event supervision and continuous value/delta shared only the global action trunk and no gripper-private state | one zero-initialized bounded multiplicative state now feeds value, delta and final event head; auxiliary heads, codec and logits-as-probe remain intact |
| `R2-EVT-01` | semantic split confirmed; not selected | action-consumed frozen rollout readout and supervised final event head share a name but not meaning | leave the rollout source unchanged; do not infer that it is the continuous gripper owner |
| `R2-P3-01` | deferred | independently contracted lanes add without post-route attribution | no P3 edit in this version |
| `R2-SEL-01` | deferred | one-of-seven gripper dimension and far hold cannot select `best.pt` | keep `best.pt` and `latest.pt`; do not add a fourth behavioral/source unit now |
| `R2-DATA-01` | closed for current data | possible row-0 qpos/action mismatch | existing probe shows exact gripper alignment and no h0 event; no source change |
| `R2-CODEC-01` | closed as first cause | both absolute and cumulative-delta branches fail before fusion | retain all six field coordinates and the `0.75/0.25` deployment decode |

## 5. R2 slice order

### R2-D01: bounded validation surface

This slice is behavior-preserving and checkpoint-compatible.  It adds no
forward, no loss, no parameter and no training selection change.

#### Full-validation metrics

Add physical and normalized variants where both charts are meaningful:

```text
validation_gripper_band_1_4_rmse_{physical,normalized}
validation_gripper_band_5_12_rmse_{physical,normalized}
validation_gripper_band_13_24_rmse_{physical,normalized}

validation_gripper_post_event_1_2_rmse_physical
validation_gripper_post_event_3_6_rmse_physical
validation_gripper_post_event_7_plus_rmse_physical
validation_gripper_post_event_rows_{1_2,3_6,7_plus}

validation_gripper_absolute_branch_band_{1_4,5_12,13_24}_rmse_normalized
validation_gripper_delta_branch_band_{1_4,5_12,13_24}_rmse_normalized
validation_gripper_branch_disagreement_band_{1_4,5_12,13_24}_rms_normalized
```

Post-event rows are defined from a target event until the next target event or
the end of the 24-row chunk.  They are grouped by distance from the latest
target event, not by absolute row alone.  Rows before any event are excluded.
This measures persistence without pretending the gripper is binary.

Use `SamplingResult.physical_field` to decode:

- absolute prediction from gripper coordinate 0;
- cumulative-delta prediction from current state plus cumulative coordinate 1;
- their existing blended deployment prediction;
- branch disagreement before blending.

The implemented D01 surface keeps the compact three-band, post-event and branch
tables plus the exhaustive `action-band x event-context` partition.  The
originally proposed 24-row event histograms were not required for the A01
decision and remain unimplemented; they are not carried into R2-B01.

#### Bounded structural metrics

On the already configured diagnostic validation batches, retain the P2
posterior long enough to report:

```text
validation_deploy_object_p2_{semantic,geometry}_band_{1_4,5_12,13_24}_interval_{0..3}_mass
validation_deploy_object_p2_{semantic,geometry}_band_{1_4,5_12,13_24}_expected_interval
validation_deploy_object_p2_{semantic,geometry}_band_pair_total_variation
```

The matrix is descriptive.  No monotonicity target, entropy quota or fixed
"far rows must choose interval 3" rule is introduced.

The following P3 post-route surface was also proposed but not implemented:

```text
validation_deploy_p3_temporal_routed_rms
validation_deploy_p3_state_change_routed_rms
validation_deploy_p3_lane_cosine
validation_deploy_p3_lane_sum_over_component_rms
validation_deploy_p3_{temporal,state_change}_band_{1_4,5_12,13_24}_rms
```

It remains deferred with P3 itself.  If reopened later, it must be computed
where routed tensors already exist rather than reconstructed from an aggregate
output.

### R2-A01: existing-checkpoint matched interventions

This was a validation pass, not a training experiment.  It reused the same R1
checkpoint, validation rows and initial physical noise for every branch.  The
executed first item covered 16/179 diagnostic batches: selected values from
intervals `0/1` and `2/3` were zeroed after spatial selection and before the
interval terminal, separately for semantic and geometry, while keys and
posteriors were preserved.

The result was decision-complete: far semantic values are strongly useful and
current geometry values are nearly action-inert.  Therefore the originally
conditional P2-posterior, projected rollout-event-source and P3-lane
interventions were not implemented.  They remain separate future probes, not
implicit parts of R2-B01.

The executed intervention reports paired primary-versus-counterfactual:

- action delta RMSE for arm and gripper in all three action bands;
- gripper MSE gain/loss in all three bands;
- post-event distance-bin gain/loss;
- decoded event precision, recall, ratio and timing on the same subset.

No intervention result is interpreted from action-delta magnitude alone.
Large action change with worse error is harmful; near-zero change is evidence
of non-use, not evidence that the upstream representation is unnecessary.

### R2-B01: three independent behavioral units

The local source candidate now contains the following three independently
revertible semantic units in order. Their forward/reverse audits and focused
tests are closed; commit boundaries and any push remain pending. They share one
later training run but no parameter, loss term or semantic claim.

#### R2-WG01: target-scale-covariant camera transport supervision

Source diagnosis:

- W2 geometry hidden state is active and interval-varying;
- the transport head operates near zero rather than in tanh saturation;
- transport uses raw small-coordinate SmoothL1 while semantic uses raw,
  target-scale-normalized and directional error;
- the learned output remains near one third of Teacher and geometry boundary
  VJP is roughly `40-45x` below semantic;
- A01 shows that the resulting small geometry value is almost action-inert.

Authorized edit:

- use the existing exact-zero, target-scale-covariant
  `row_loss(..., scale_floored=True)` for transport common and interval
  innovation;
- keep `0.55/0.15/0.05`, outer `future_dynamics=0.10`, Teacher targets,
  camera support, W forward algebra, covariance and P2 unchanged;
- retain an audit-only raw-coordinate transport term so the change of formal
  loss units is explicit in JSONL rather than hidden by the old metric name.

This is mathematically an effective rescaling, but it is derived from each
Teacher target scale with the existing smooth floor; it is not a manually
chosen geometry gain or a prediction-amplitude target. Prediction equal to
target still has exact zero loss, including a legal zero target. The edit adds
no parameter, buffer, runtime call or checkpoint tensor.

Acceptance is relational: transport prediction/Teacher ratio and geometry
common/interval VJPs should recover from R1 without semantic, covariance or
task behavior collapsing. No fixed ratio is placed in the objective.

#### R2-P202: stage-private physical-terminal query

Source diagnosis:

- R1f correctly separates spatial selection from physical-I termination in
  tensor axes, but both stages call the same two `source_query` projections;
- spatial K/K*C addressing and temporal I removal therefore update one
  parameter owner despite having different candidate axes and meanings;
- A01 proves that the current terminal has useful semantic behavior, so its
  scores, supports, no-null rule and values must not be replaced by a schedule.

Authorized edit:

- retain `source_query` exclusively for spatial semantic K and geometry K*C;
- add two `terminal_query` bias-free linear projections as a `deepcopy` of the
  corresponding spatial projections, consuming no new initialization RNG;
- use `terminal_query` only in `temporal_terminal` and keep selected W key,
  S's zero-preserving conditioner, bounded temperatures, support, posterior
  and value contraction byte-for-byte otherwise unchanged.

At construction, spatial and terminal projections are exactly equal and the
complete P2 forward is exactly the R1 function. Training may separate them
without a hard row-to-interval mask, monotonic prior, entropy quota, learned
null, type competition or interval gain. The action query still receives both
reverse paths, while each projection receives only its named stage gradient.

Add JSONL-only output-VJP metrics for the spatial and terminal query tensors
and a detached projection-delta RMS. Tests must prove initialization identity,
stage-local parameter VJPs, retained K/K*C/I axes, semantic/geometry
independence, all-invalid zero and unchanged A01 intervention locality.

#### R2-GRIP02: continuous gripper-private state closure

Source diagnosis:

- value, delta and four auxiliary field heads are independent projections of
  the same general action token;
- the supervised final event head is a sibling projection and its logits never
  enter the physical field;
- both deployed value and cumulative-delta branches fail before fusion, while
  event-related rows dominate error;
- the action-consumed frozen layer-contract event readout is a different
  rollout evidence object and is not the supervised gripper event owner.

Authorized edit inside the active `ActionOnlyPhysicalVelocityHead`:

```text
base_state = action_token
base_read = existing_velocity_norm(base_state)
gate = tanh(bias_free_gate(base_read))
gripper_state = base_state + base_state * gate
gripper_read = existing_velocity_norm(gripper_state)
```

The new gate weight is exact zero at construction and its constructor must
leave the existing initialization RNG stream unchanged. The multiplicative
form bounds each relative coordinate factor to `[0,2]` and cannot create a
state from an exact-zero token.

- arm absolute/delta and the four non-deployed auxiliary gripper heads continue
  to read `base_read`;
- deployed gripper value and adjacent-delta heads read `gripper_read`;
- the supervised final event head reads the exact same raw `gripper_state`
  through its own existing normalization;
- the active Schema25 profile is the six-channel `legacy_handcrafted` branch;
  the shared head's inactive `parseval_temporal` branch must remain valid by
  routing its sole native gripper read through `gripper_read`, with the same
  zero-gate identity;
- the motion head, layer-contract rollout event source, six-channel targets,
  formal losses, event threshold, five-step sampler and `0.75/0.25` decode stay
  unchanged.

The shared state, not the event logits, is the continuous control owner. At
zero gate the complete physical field and event logits equal R1 exactly. The
event loss can subsequently shape a state that is physically consumed by
value/delta without imposing an event gate or binary gripper state. Candidate
execution-value calls must pass through the same velocity-head state; the two
final event-head call sites must consume the state returned by that same head
rather than recompute an alternate branch.

Add JSONL-only gate RMS, state-delta RMS and state VJP. Tests must prove zero
initial identity; nonzero gate locality to value/delta/event only; arm,
auxiliary and motion invariance; decoded continuous reachability; candidate
execution-path coverage; and finite reverse gradients from flow, decoded,
delta-consistency and event losses.

#### Combined identity and rollback

Measured parameter delta is three bias-free `512x512` matrices: two P2 terminal
queries and one gripper gate, `+786,432` trainable parameters and `+3` state
keys. Optimizer group count remains `23`; the P2 tensors belong to
`p2_effect_reader` and the gate to `bottom_heads`.

The top, bottom and training component ABI strings and active-source digest
must change. Exact R1 resume and bottom-only migration are rejected because
the bottom ABI changes; R2 starts fresh. Capability schema remains the adopted
Schema25 replay lineage. Each commit remains independently revertible even
though the formal run uses their ordered combination.

### R2-K01: checkpoint selection stays out of the source candidate

Keep the existing `best.pt` criterion and `latest.pt`; do not add
`best_gripper_tail.pt` in the same source version.  The late-gripper blind spot
is real, but a new selector would be a fourth decision with its own artifact
semantics.  During R2 review, evaluate both existing artifacts against the same
full validation table and task rollout when available.  Decoded event F1, far
gripper RMSE and the seven-dimensional average are reported separately; no one
metric and no invented weighted composite declares model quality by itself.

## 6. Behavioral implementation boundary

The authorized production boundary is deliberately narrow and split by unit:

- `clearvla/mainline/training/losses.py`: R2-WG01 changes only the transport
  common/innovation row-loss geometry and adds a raw-coordinate audit term;
- `clearvla/mainline/model/compiler.py`: R2-P202 adds the copied terminal query,
  routes only physical-I termination through it and emits bounded query-stage
  diagnostics;
- `clearvla/mainline/v120_core/decoder.py`: R2-GRIP02 owns construction of the
  exact-zero gate and one canonical field-plus-gripper-state readout;
- `clearvla/mainline/v120_core/time_domain_mmdit.py`: every candidate velocity
  call uses that canonical readout, while both final event-head call sites use
  the returned gripper state; motion continues to read the original action
  state;
- `clearvla/mainline/manifest.py`: update only the top, bottom and training
  component ABI identifiers required by the three changed semantics;
- `clearvla/mainline/runtime/logging.py` registers only the
  `gripper_private_` scalar prefix needed to archive the named GRIP02 health
  fields; the other new keys already use active prefixes and no new console
  panel is added;
- focused tests cover exact initialization identity, axes/support, stage-local
  gradients, alternate call sites, optimizer ownership and checkpoint/state
  inventory before the retained mainline suite is run.

Explicitly outside the boundary are observation, grounding, S, the W forward
producer, Teacher association and targets, P1, P3, the layer-contract rollout
event readout, the physical codec, data sampler, event target/threshold,
sampler fractions, execution policy, loss coefficients, checkpoint selector
and configuration surface.  Any need to edit one of those reopens the plan
instead of being absorbed as a convenience fix.

### 6.1 Test-first closure

The implemented guards and independently checked boundaries are:

| Unit | Test/check location | Closed assertions |
|---|---|---|
| R2-WG01 | `tests/test_mainline_structural_contracts.py` | supported matching and unsupported rows remain exact zero; the covariant term owns backward; the R1 raw-coordinate term is detached and explicit |
| R2-P202 | `tests/test_mainline_structural_contracts.py` | copied spatial/terminal weights are exact; spatial-only loss reaches only `source_query`; terminal-only loss reaches only `terminal_query`; inherited axis/support tests remain green |
| R2-GRIP02 | `tests/test_mainline_policy.py` | zero-gate full-field identity; forced-gate locality to deployed gripper coordinates; event-to-gate reachability; arm/auxiliary/motion isolation; both active final call sites use the shared state helper |
| ABI | `tests/test_mainline_manifest.py` | only top, bottom and training component identities change; observation/runtime and capability schema remain unchanged |
| Combined | production inventory plus retained optimizer/checkpoint tests | exact `+786,432`/three-key delta, 23 optimizer groups, unchanged construction RNG and rejection of incompatible old ABI |

The bounded training keys are implemented as:

```text
loss_future_transport_raw_coordinate
gradient_tensor_p2_spatial_query_rms
gradient_tensor_p2_terminal_query_rms
object_p2_terminal_query_delta_rms
gripper_private_gate_rms
gripper_private_state_delta_rms
gradient_tensor_gripper_private_state_rms
```

They are scalar JSONL fields.  They add no target, loss term, stop criterion,
console matrix or checkpoint buffer.  The raw-coordinate transport scalar
reuses the raw row error already computed inside the target-scale-covariant
objective, so it does not launch a duplicate loss calculation.  The remaining
six state/VJP scalars run only under the existing diagnostic flag: every 20
formal training batches, with state/projection scalars also retained on the
configured 16 validation batches; VJP hooks attach only during training.
They request no extra forward, sampling replay or high-dimensional tensor dump.

## 7. Verification and acceptance

### 7.1 Diagnostic slice

Release-blocking checks:

- primary sampled actions are bitwise identical before and after D01 when
  interventions are disabled;
- no parameter, optimizer group, persistent buffer or checkpoint field changes
  in D01;
- full-validation counts exactly reconstruct aggregate gripper square error;
- gripper-band counts sum to all 24 rows;
- post-event bins stop at the next target event and never count pre-event rows;
- absolute and cumulative branch metrics reconstruct the existing blended
  decode at the configured weights;
- P2 band/type/interval masses sum to one on supported rows and zero on
  unsupported rows;
- disabled intervention paths are exact identity;
- matched interventions reuse the primary initial physical noise;
- focused tests, retained mainline suite, touched-file Ruff, compileall and
  `git diff --check` pass; changed production ranges introduce no new Pyright
  error. The repository-wide static wrapper remains red on the pre-existing
  broad Ruff/Pyright baseline and is not repaired by unrelated R2 edits.

### 7.2 Behavioral R2 run

Before the one formal run, record the R1 checkpoint value for every metric with
an R1 analogue on identical rows/noise, and record all zero-initialized R2
identity metrics before the first update.  Success is relational:

- physical gripper RMSE in rows 13-24 and the far action-band event-context
  cells must improve over R1; full gripper RMSE must also improve rather than
  trading all hold rows for a narrow event win;
- arm RMSE and rows 1-4 gripper RMSE are explicit guards; the improvement cannot
  be only a lower global average or a larger decoded-event F1;
- predicted event count/target ratio, precision/recall and timing must rule out
  a trivial over-triggering win;
- R2-WG01 must increase useful transport learning relative to Teacher and its
  common/interval boundary VJPs without collapsing semantic, covariance,
  camera support or task behavior; the raw-coordinate audit remains finite;
- R2-P202 must show gradients in both spatial and terminal query owners, a
  finite learned projection delta after their exact-copy start, and retained
  useful far semantic value attribution; no required interval mass is imposed;
- R2-GRIP02 must show a nonzero but finite bounded gate, a physically consumed
  state with gradients from continuous and event losses, and improvement in
  both deployed branch tables before interpreting the fixed blend;
- all three all-invalid/exact-zero identities, tensor axes and alternate
  consumer paths must remain closed in validation and checkpoint inventory.

Do not stop a healthy run from one early noisy event-F1 movement.  Hard stops
remain non-finite values, broken identities, missing gradients after the
pre-registered initial window, an intended trainable path remaining exact
identity, or repeated paired anti-effect at two consecutive pre-registered
validation gates.  A single metric miss identifies a unit for rollback; it
does not retroactively merge the three causal claims.  Numerical thresholds
are fixed in the run protocol from the R1 distribution before launch, not
invented while watching R2.

## 8. Explicit non-goals and anti-hardening rules

R2 does not authorize:

- a hard event gate on gripper output;
- a binary open/closed target replacing continuous gripper state;
- a required event frequency, entropy target or attention quota;
- a hard chronological P2 mask or a fixed "far row -> interval 3" schedule;
- increasing interval-3, state-change, gripper-delta, event, geometry-output or
  P2 gains because a current magnitude looks small;
- changing the six-channel codec, fixed decode, sampler fractions, event
  threshold or objective coefficients;
- interpreting R2-WG01's target-scale-covariant error as permission to add a
  manual geometry gain or an amplitude target;
- allowing the copied P2 terminal query to acquire a time prior, row schedule,
  null bypass or shared-gradient shortcut;
- letting event logits gate the field, or letting the new gripper state leak
  into arm, auxiliary gripper or motion heads;
- restoring every historical gripper donor as a bundle;
- adding a new checkpoint selector in the same source version;
- using aggregate normalized RMSE or event F1 as the sole success criterion;
- retaining high-cardinality console rows that already exist in JSONL.

## 9. Closed implementation assumptions and remaining behavioral uncertainty

The implementation-local assumptions are closed:

1. WG01 reuses the existing smooth target-scale floor, keeps supported
   exact-match and unsupported rows at exact zero, and exposes the R1 raw loss
   only as a detached audit.
2. `terminal_query` is an exact construction-time tensor copy, leaves the
   inherited RNG stream unchanged and receives only physical-I terminal
   gradients; `source_query` receives only spatial K/K*C gradients.
3. The zero gripper gate preserves the inherited field exactly. The canonical
   velocity helper is used by every final and candidate call; a nonzero gate can
   affect only deployed gripper value/delta and the final event state.
4. The final event head receives the same raw gripper-private state whose
   normalized view reaches value/delta. Arm, four auxiliary gripper coordinates
   and motion retain the R1 source.
5. The measured delta is exactly `+786,432`, three state keys and no optimizer
   group or post-construction RNG change; component ABI and manifest digest are
   updated accordingly.

One unresolved limitation remains behavioral: the combined formal run cannot
independently estimate each unit's task-level counterfactual. Separate semantic
units, per-unit health fields and reversible source identities preserve
diagnosis, but a later ablation is required only if two units move in opposing
directions. This is an explicit experiment-budget limitation, not a claim of
causal separability from the combined run.

## 10. Immediate execution sequence

1. **Complete:** implement and validate the behavior-preserving D01/A01 surface
   on the exact R1 checkpoint, including matched P2 values, codec branches and
   `action-band x event-context` attribution.
2. **Complete:** map geometry, P2 and gripper independently from producer through
   every transformation, consumer, loss, optimizer, checkpoint owner,
   repetition site, reverse path and legal bypass.
3. **Complete:** translate the implementation assumptions into focused identity,
   ownership, gradient, axis and inventory guards.
4. **Complete locally:** implement R2-WG01 and repeat its forward/reverse audit.
5. **Complete locally:** implement R2-P202 and repeat its forward/reverse audit.
6. **Complete locally:** implement R2-GRIP02 and inspect every active
   velocity/event consumer in dynamic and fallback paths.
7. **Complete:** finish the final touched-file static checks, retained suite,
   exact diff review and documentation closure.
8. **Authorized:** create the reversible R2 semantic commit identity and push
   it to `codex/schema25-r1-replay`.
9. **Complete local substitute:** run one fresh production-dimension CPU BF16
   training batch plus the retained five-step deployment guard. A separate
   remote smoke is temporarily unavailable by explicit user decision;
   therefore the formal run must treat its preflight and first reporting
   window as the CUDA runtime/memory gate. Do not insert an old-checkpoint
   experiment before it.
