# Schema25 R2 source-derived improvement plan

Status: `R2-D01 AND MATCHED P2-VALUE R2-A01 IMPLEMENTED; EXISTING-CHECKPOINT A01 VALIDATION PENDING; NO BEHAVIORAL R2 EDIT OR TRAINING AUTHORIZED`

Implementation checkpoint (2026-08-27): the behavior-preserving diagnostic
surface records P2 `type x action-band x physical-interval` posterior mass,
expected interval, support and band-pair total variation on the existing
bounded deployment diagnostics. Full validation records gripper RMSE by action
band, target-event-relative persistence, both deployed codec branches and an
exhaustive `action-band x event-context` error partition. R2-A01 additionally
implements four evaluation-only, matched-noise P2 value counterfactuals:
semantic/geometry crossed with near intervals `0/1` or far intervals `2/3`.
They alter selected common and residual values only after posterior formation;
the primary path, model state, loss, optimizer, checkpoint ABI and RNG stream
are unchanged. The retained mainline suite passes `164/164`. A dedicated
read-only validation loader accepts only the intentional active-source identity
difference while requiring identical manifest, semantic config, dataset,
language artifact and complete model-state ABI.
`scripts/validate_mainline_checkpoint.sh` loads no optimizer, schedule or
checkpoint RNG state and writes no checkpoint. The 24-row event histograms,
P2 posterior intervention, projected event-source intervention and P3
post-route/intervention diagnostics remain unimplemented and unauthorized by
this checkpoint.

This document plans the next version after the completed Schema25 R1 source
replay and its first formal run.  It is prospective auxiliary research memory,
not an override of `00_CURRENT_ARCHITECTURE_CONTRACT.md`.  A behavioral R2
implementation may start only after the diagnostic gate below selects one
source-backed semantic unit and that unit receives its own producer-to-consumer
worksheet.

The version is deliberately not defined as "more validation fields."  The
validation additions are a low-cost attribution layer around one later
structural change.  Conversely, the plan does not authorize simultaneous P2,
P3, event, codec, sampler and loss-weight edits merely because each has an open
question.

## 1. Executive decision

The next run should be built in this order:

1. add a bounded validation surface and matched-noise interventions that can
   be run on the existing R1 checkpoint without training;
2. decide whether the first behavioral R2 unit belongs to the P2
   action-time/World-interval terminal, the gripper transition-to-hold path, or
   one P3 optional lane;
3. implement exactly one of those behavioral units;
4. retain the ordinary V120 checkpoint and add one gripper-tail candidate
   checkpoint, without changing the training objective or inventing one scalar
   "health score";
5. spend one formal training run only after the selected unit has a closed
   forward path, reverse path, zero semantics and paired R1 validation baseline.

Current priority is **P2 time allocation first**, not a new gripper module.
R1 has more W interval variation than the Schema25 base, but its present P2
marginal assigns substantially less mass to interval 3.  Existing diagnostics
average over all 24 action rows, four bases and two P2 types, so they cannot say
whether far action rows actually consume far World evidence.  That missing
fact must be recovered before altering event supervision or the action codec.

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
- Schema25 base reports W interval variation `0.085714` and P2 interval-3
  marginal mass `0.438278` at epoch 8.
- R1 reports W/public prediction interval variation about `0.161` at epoch 8,
  while validation-deployment P2 interval-3 marginal mass is only
  `0.198-0.245` over epochs 1-7.  P2 entropy remains high (`0.947-0.983`).
- R1 P3 temporal/state-change RMS changes smoothly and does not show a sharp
  late-epoch explosion.  P3 remains an attribution target, not the first
  presumed cause.

### 2.3 Current inferences

1. The upstream World stream has not simply collapsed; R1 contains measurable
   interval structure.
2. The first missing causal fact is whether R1f sends that structure to the
   correct action horizon rows.
3. Local event classification and persistent post-event state are different
   tasks.  The present validation surface conflates them.
4. The event semantic split is real, but modifying it before P2 and gripper
   branch attribution would be speculative.
5. The legacy six-channel codec is open to measurement but is not presently an
   authorized repair.  Historical recovery evidence does not support blaming
   it first.

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

## 4. Structural issue register

| ID | Status | Issue | R2 disposition |
|---|---|---|---|
| `R2-P2-01` | highest-priority causal candidate | P2's action-band/World-interval allocation is hidden by a global mean; R1 retains W variation but shifts marginal mass away from interval 3 | add band/type matrix and matched interval interventions first; eligible as the one R2 behavioral unit |
| `R2-GRIP-01` | confirmed observability defect | aggregate gripper and aggregate horizon metrics cannot localize post-event hold failure | add gripper x horizon, post-event distance and branch metrics over full validation |
| `R2-EVT-01` | confirmed semantic ownership split; causal role unknown | action-consumed frozen rollout readout and supervised final event head share a name but not meaning | retype names in the eventual source edit; run projected-source intervention before delete/share decisions |
| `R2-P3-01` | bounded structural risk | independently contracted lanes add without post-route per-lane attribution | add post-route lane/cancellation metrics; intervene only if P2/event gates are inconclusive |
| `R2-SEL-01` | confirmed artifact-selection mismatch | one-of-seven gripper dimension and far hold cannot select `best.pt` | preserve `best.pt`; add one parallel gripper-tail candidate checkpoint |
| `R2-DATA-01` | closed for current data | possible row-0 qpos/action mismatch | existing probe shows exact gripper alignment and no h0 event; no source change |
| `R2-CODEC-01` | open measurement only | fixed `0.75/0.25` decode may hide branch-specific failure | report both branches and disagreement; no codec change in the first R2 run unless the diagnostic gate selects it in a later plan |

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

Also record target and predicted open/close/any event counts by each of the 24
row positions.  The 24-row histograms stay JSONL-only.  A compact console line
may contain only the three physical gripper-band RMSE values, three post-event
RMSE values and three branch-disagreement values.

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

For P3, add post-route values before their sum:

```text
validation_deploy_p3_temporal_routed_rms
validation_deploy_p3_state_change_routed_rms
validation_deploy_p3_lane_cosine
validation_deploy_p3_lane_sum_over_component_rms
validation_deploy_p3_{temporal,state_change}_band_{1_4,5_12,13_24}_rms
```

These metrics must be computed where the routed tensors already exist.  Do not
reconstruct a lane from an aggregate output.

### R2-A01: existing-checkpoint matched interventions

This is a validation pass, not a training experiment.  Reuse the same R1
checkpoint, validation rows and initial physical noise for every branch.
Coverage remains bounded; start with the current 16/179 diagnostic batches and
increase only if paired confidence is inadequate.

Run in priority order:

1. **P2 interval value interventions**: zero selected values from intervals
   `0/1` and `2/3` after spatial selection but before the interval terminal,
   separately for semantic and geometry.  Preserve keys/posteriors in this
   first pass so the result measures value contribution rather than a new
   selector.
2. **P2 posterior intervention** only if value attribution is ambiguous:
   replace the learned terminal posterior by its supported uniform
   counterfactual while keeping selected values fixed.  Label it as a
   counterfactual, never as a proposed runtime rule.
3. **Projected rollout-event source**: set the already projected event source
   token to algebraic zero, and separately batch-shuffle it.  Intervene after
   `event_proj`; raw zero before affine projection is not a neutral value.
4. **P3 lanes**, only if the first three results are inconclusive: zero
   temporal and state-change after each lane's basis read and before their sum.

Every intervention reports paired primary-versus-counterfactual:

- action delta RMSE for arm and gripper in all three action bands;
- gripper MSE gain/loss in all three bands;
- post-event distance-bin gain/loss;
- decoded event precision, recall, ratio and timing on the same subset.

No intervention result is interpreted from action-delta magnitude alone.
Large action change with worse error is harmful; near-zero change is evidence
of non-use, not evidence that the upstream representation is unnecessary.

### R2-G01: select exactly one behavioral structural unit

R2-G01 is not chosen by preference.  It is selected by the paired results.

#### Branch A: P2 time-allocation repair

Select this branch when all of the following hold:

1. W retains nontrivial interval variation;
2. far action bands do not use a meaningfully different interval distribution
   from near bands, or later-interval values have materially less far-gripper
   effect than earlier-interval values;
3. the same paired intervention localizes the deficit to P2 rather than to a
   downstream gripper branch;
4. arm improvement cannot explain away the gripper-specific far deficit.

The later implementation worksheet must preserve R1f spatial ownership:

- K and K*C are still removed separately inside every physical interval;
- I remains explicit until the action-query terminal;
- semantic and geometry remain complementary, independently surviving values;
- no hard row-to-interval mask, forced monotonic schedule, learned null, type
  competition, entropy target or far-interval gain is allowed;
- any new time relation must be a soft, reversible residual around the existing
  terminal and must have an identity/zero initialization path;
- the repair changes only the action-time/World-interval relation, not W
  prediction, S relevance, P3 amplitude or the action codec.

The exact relation is deliberately not frozen in this plan.  It will be chosen
from the measured failure (missing horizon differentiation versus wrong value
ownership), then audited as its own semantic unit.

#### Branch B: continuous gripper transition-to-hold closure

Select this branch only if P2 demonstrates real, correctly localized far
evidence but post-event state still fails.

The branch diagnosis uses the two existing deployed coordinates:

- absolute good / cumulative delta bad: do not increase delta authority;
- cumulative delta good / absolute bad: investigate the fixed decode and
  absolute-state carrier;
- both good but blend bad: the fusion itself is the candidate;
- both bad after events: the shared action representation, not codec blending,
  is the candidate;
- event logits improve while neither deployed branch improves: event ownership
  is not closed.

Any repair must keep continuous value primary and may not use a hard event
gate, hard open/closed state, mandatory event quota or direct F1 optimization.
The final event head may remain a probe, or it may share a gripper-specific
latent with deployed value/delta heads, but it cannot be called a control owner
unless the deployed continuous field actually consumes the shared state.

Do not import the historical Parseval gripper field or change the global
`0.75/0.25` blend in this first branch merely because those donors exist.  A
codec change requires its own projection, target, decode, loss, optimizer,
checkpoint and deployment worksheet.

#### Branch C: one P3 optional-lane repair

Select this branch only if a matched post-route lane intervention improves
far/post-event gripper error on the same rows without merely suppressing all
action, and P2/event diagnostics do not already explain the failure.

The permitted unit is one lane's value or one aggregate-sum boundary.  It may
not combine:

- lane deletion;
- a new RMS target;
- restored `/sqrt(2)` divisors;
- a restored `0.05` multiplier;
- a bottom write-scale change.

Those are separate hypotheses and remain separate versions.

#### No-branch outcome

If none of the three branches meets its gate, R2 training does not start.  The
correct result is an unresolved attribution report, not a speculative bundle.

### R2-K01: artifact selection without objective hardening

Keep the existing `best.pt` criterion and `latest.pt`.  Add one parallel
`best_gripper_tail.pt`, selected by
`validation_gripper_band_13_24_rmse_physical`.  It is a candidate artifact, not
the sole declaration of model quality.

At run completion compare at least:

- ordinary `best.pt`;
- `best_gripper_tail.pt`;
- `latest.pt`.

Decoded event F1 must not select a checkpoint by itself.  If task rollout
evaluation is available later, it adjudicates among these candidates; no
weighted scalar composite is invented in R2.

## 6. Implementation boundary for the diagnostic phase

Expected behavior-preserving files:

- `clearvla/mainline/runtime/evaluation.py`
- `clearvla/mainline/runtime/sampling.py` only if the accumulator needs an
  explicit field/codec view; no sampler math changes
- `clearvla/mainline/model/compiler.py` for posterior diagnostics and a
  test-only/eval-only intervention seam
- `clearvla/mainline/model/restored_bottom.py`
- `clearvla/mainline/v120_core/time_domain_mmdit.py` for post-route/event-source
  eval interventions
- `clearvla/mainline/train.py` for bounded aggregation and the additional
  candidate checkpoint
- `clearvla/mainline/runtime/logging.py` for a bounded console panel
- focused tests for accounting, matched noise, algebraic zero and no-forward
  guarantees

No other production module enters R2-D01/A01 without reopening this plan.
In particular, the data sampler, event threshold, loss weights, physical codec,
W producer and Teacher target are outside the diagnostic edit boundary.

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
- P3 post-route lane sum reconstructs the current aggregate routed value;
- disabled intervention paths are exact identity;
- matched interventions reuse the primary initial physical noise;
- focused tests, retained mainline suite, compileall, Ruff, Pyright and
  `git diff --check` pass.

### 7.2 Behavioral R2 run

Before the one formal run, record an R1 checkpoint baseline for every new
metric on the identical validation rows and noise.  Success is relational:

- the selected causal statistic must move in the intended direction;
- physical gripper RMSE in rows 13-24 and post-event far bins must improve over
  the paired R1 checkpoint;
- the improvement cannot be only a higher event F1 or a lower global average;
- arm RMSE and near-band gripper must remain explicit guards;
- predicted event count/target ratio and timing must rule out a trivial
  over-triggering win;
- the selected structural path must have nonzero paired action effect in the
  region it claims to repair.

Do not stop a healthy run from one early noisy event-F1 movement.  Hard stops
remain non-finite values, broken invariants, missing gradients on the selected
unit, exact identity of a path that was supposed to be active, or a repeated
paired anti-effect after the pre-registered evaluation gate.  Numerical stop
thresholds belong in the selected unit's implementation protocol after the R1
baseline distribution is measured; they are not guessed here.

## 8. Explicit non-goals and anti-hardening rules

R2 does not authorize:

- a hard event gate on gripper output;
- a binary open/closed target replacing continuous gripper state;
- a required event frequency, entropy target or attention quota;
- a hard chronological P2 mask or a fixed "far row -> interval 3" schedule;
- increasing interval-3, state-change, gripper-delta or event loss gain because
  its current magnitude looks small;
- changing the six-channel codec before branch diagnostics;
- changing sampler fractions, event threshold and objective weights in the
  same run as a structural unit;
- restoring every historical gripper donor as a bundle;
- using aggregate normalized RMSE or event F1 as the sole success criterion;
- retaining high-cardinality console rows that already exist in JSONL.

## 9. Unresolved assumptions to close before behavioral implementation

1. Does the P2 interval posterior differ by action band and type at deployment?
2. Which W interval values causally change far gripper versus far arm output?
3. Is the late gripper error carried by the absolute branch, cumulative-delta
   branch, or their fixed fusion?
4. Does post-event error grow with distance from the latest event?
5. Does the projected layer-contract event source have gripper-specific action
   effect, generic action effect, or no effect?
6. Do P3 temporal and state-change reinforce, cancel, or act on different
   horizons after routing?
7. Which epoch would the gripper-tail checkpoint select relative to ordinary
   `best.pt` and `latest.pt`?

Until these are answered, the only authorized work is R2-D01/A01 planning,
implementation and validation on the existing checkpoint.  The answer to the
gate, not the size of the source diff, determines the one behavioral R2 unit.

## 10. Immediate execution sequence

1. **Complete:** write the R2-D01 boundary and first observation slice from the
   exact active source.
2. **Complete:** implement P2 band/type/I and gripper band/post-event
   validation with behavior-invariance tests, plus a source-delta-bounded
   read-only existing-checkpoint validation entry point.
3. **Complete:** run the first full validation pass on the existing R1
   checkpoint and establish that the marginal observation is not yet
   decision-complete.
4. **Complete in source; validation pending:** close the full A01 boundary,
   implement only the matched P2 interval-value seam plus codec-branch and
   `action-band x event-context` attribution, and preserve the same primary
   physical noise/cache for every counterfactual.
5. **Next:** rerun one read-only full validation pass on the existing R1
   checkpoint; no training, then produce a short decision table selecting P2,
   continuous gripper/event, one
   P3 lane, or no branch.
6. Write and review the selected behavioral unit's complete forward/backward
   worksheet.
7. Implement that one unit plus `best_gripper_tail.pt`.
8. Run the retained test matrix, then request authorization for the single R2
   formal training run.
