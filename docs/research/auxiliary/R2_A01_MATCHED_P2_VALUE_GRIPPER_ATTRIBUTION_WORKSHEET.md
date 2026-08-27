# R2-A01 matched P2-value and gripper-attribution worksheet

Status: `IMPLEMENTED, STATICALLY CLOSED, AND VALIDATED ON THE EXISTING R1 CHECKPOINT`;
validation-only; no behavioral R2 unit, training run, loss change or checkpoint
write is authorized by this worksheet itself.

This worksheet is subordinate to
`../00_CURRENT_ARCHITECTURE_CONTRACT.md` and
`SCHEMA25_R2_SOURCE_DERIVED_IMPROVEMENT_PLAN.md`. It closes only the next
read-only attribution slice on the existing R1 checkpoint. The result may
select a later behavioral worksheet; it cannot itself make a route mandatory.

## 1. Why another bounded validation pass is necessary

The first R2-D01 replay established two useful but incomplete observations:

- P2 semantic allocation changes by action band, while geometry allocation is
  nearly row-invariant;
- gripper error is largest immediately after a target event and then declines,
  while aggregate error still rises strongly with action horizon.

Those marginals do not identify causality. A route can carry posterior mass
without contributing a useful value, and the post-event marginal pools all 24
action rows. The next pass therefore combines one matched P2 value
counterfactual with diagnostics computed from the already returned physical
field. No new training or dataset sample is required.

## 2. Exact source and run evidence

Active source before editing:

```text
commit                                           6ef21c3d1f5eb80249454a22dc00f995e75e52ac
compiler.py blob                                 5538f1d51347ab3c1f4e16eeaddcdba3270132be
evaluation.py blob                               0ccbb894bba5de1b608d3743e0bc9016753ee75b
train.py blob                                    cb78ca568edb11234bd58c7fd98620879e60f06f
checkpoints.py blob                              3e0c4f3c9bcb93c84e531d4f48c784cf8fadf3b0
sampling.py blob                                 e7e7ce49e5daec2c757774db3482973a631364cb
policy.py blob                                   1a06ca465d74815bfba10b2b55743917599bff40
top.py blob                                      e49c205b839a752916d6225c88dfc2d8fc3025f0
losses.py blob                                   af03984ba0bbcf68e09ec4af8117f13660f12580
action_codec.py blob                             4635e8728988858d3937a6dc06eaadca10dce4d2
```

The epoch-8 validation replay used all 179 batches / 1,432 samples and the
same R1 model state. The bounded P2 surface covered 16 evenly distributed
batches. Its accounting identities were exact and 73 pre-existing console
metrics were string-identical to the pre-diagnostic replay.

Relevant observations:

```text
P2 semantic band-pair total variation            0.211087
P2 geometry band-pair total variation            0.017577
P2 semantic effect RMS                           0.112451
P2 geometry effect RMS                           0.014242
gripper RMSE bands 1-4 / 5-12 / 13-24            0.059981 / 0.125307 / 0.195153
post-event RMSE distances 1-2 / 3-6 / 7+          0.376999 / 0.306177 / 0.180002
decoded event recall / event ratio               0.296242 / 0.504053
```

The training JSONL also shows a producer-side scale asymmetry:

```text
epoch-7 W2 semantic common / interval input VJP  1.121e-5 / 2.795e-6
epoch-7 W2 geometry common / interval input VJP  2.477e-7 / 6.479e-8
epoch-7 predicted / teacher transport RMS        0.03097 / 0.08718
epoch-8 future transport loss                    0.001659
effective outer semantic contribution            about 0.02052
effective outer transport contribution           about 0.0000249
```

The raw-coordinate SmoothL1 and inherited internal weight are source facts,
but the approximately 40-45x boundary-VJP difference does not prove that
raising transport supervision would improve action. That action relevance is
the causal fact this slice measures.

## 3. Complete active boundary before editing

### 3.1 W geometry producer and direct supervision

`ObjectFutureTeacher` forms camera-specific normalized-coordinate
displacement moments `[B,I,K,C,2]` in FP32. The null hypothesis contributes
zero displacement inside each current observable camera denominator.

`ObjectFutureDynamicsCompiler` constructs geometry common and interval
innovation from the S geometry owner, conditions them zero-preservingly on the
current camera transport prior, and decodes each with the shared bounded
`transport_head`. The exported mean is:

```text
0.50 * tanh(transport_head(common_geometry))
  + 0.50 * tanh(transport_head(interval_geometry))
```

masked by current camera availability. `future_dynamics_terms` supervises
transport common and innovation with unscaled raw-coordinate SmoothL1 under
the inherited internal weight `0.15` and outer future weight `0.10`.
Teacher targets and current support are detached. W parameters remain owned by
optimizer role `dynamics` and serialize in the ordinary model state.

### 3.2 P2 spatial and temporal transformations

For every physical interval, P2 semantic removes only K and geometry removes
only K*C. Geometry consumes current coordinate, transport and FP32 covariance
before C disappears. The selected result retains:

```text
key / common / residual / selected S              [B,T,Q,I,Z=2,H]
observable interval support                       [B,I,Z]
```

The terminal conditions the selected W key with selected S metadata, then
normalizes a no-null posterior over I independently for semantic and geometry.
It selects common and residual values with that same posterior, adds the two
types, and applies one caller-owned P2 RMS contract. The effect enters:

- protected consequence directly;
- P3 temporal private evidence through effect plus interaction;
- controlled transition through protected consequence;
- the bottom protected-detail basis read.

Factual P1, dynamic P1 precision, transition, P3 state-change and the bottom's
other evidence remain legal alternate action paths. Therefore a weak P2
counterfactual effect means non-use by this checkpoint, not that the upstream
representation is universally unnecessary.

Action, decoded-action, event, motion and execution losses backpropagate from
the bottom through consequence/P2 into W. Direct future losses reach W without
passing P2. The semantic-plus-geometry sum gives the two P2 value tensors an
identical immediate VJP; equal logged VJPs at that exact addition boundary do
not imply equal upstream gradients or learned values.

### 3.3 Gripper producer, loss, decode and deployment

The physical action field has 18 coordinates. Six belong to gripper:

```text
absolute, adjacent delta, relative-to-current, previous, |delta|, relu(delta)
```

Only absolute and adjacent delta are deployed. The final normalized gripper is
`0.75 * absolute + 0.25 * (current + cumsum(delta))`. The other four heads
receive flow supervision but are not decoded.

Formal gripper flow averages all six coordinates. Event rows are about four
percent of validation rows; event/hold-balanced flow is diagnostic-only. The
separate event focal loss trains a sibling endpoint head on shared bottom
features, but that head neither gates nor edits the physical field. Five ODE
updates run at `t=[0,.2,.4,.6,.8]`; the endpoint `t=1` call supplies event and
motion logits without changing the integrated field.

`SamplingResult.physical_field` already returns the final 18-D field.
Absolute, cumulative-delta and their disagreement can therefore be measured
without another forward and without changing codec math.

### 3.4 Validation, noise and checkpoint boundary

The full primary validation uses a deterministic per-batch generator. Existing
proposal and execution ablations reuse `prediction.initial_physical_noise`.
R2-A01 must do the same and must reuse the same encoded observation cache.

The read-only validation loader requires the exact model state ABI and permits
only explicitly listed source differences. The three production files needed
by this slice (`compiler.py`, `evaluation.py`, `train.py`) are already inside
that bounded validation-source set. This slice adds no parameter, buffer,
optimizer role, checkpoint field, RNG draw on the primary path or checkpoint
write.

## 4. Authorized P2 value counterfactual

Four evaluation-only modes are allowed:

```text
semantic_near_zero      zero semantic values at intervals 0 and 1
semantic_far_zero       zero semantic values at intervals 2 and 3
geometry_near_zero      zero geometry values at intervals 0 and 1
geometry_far_zero       zero geometry values at intervals 2 and 3
```

The intervention is applied after spatial selection and before the temporal
weighted sum. It multiplies both selected common and selected residual values
for the named type/interval group by exact zero. It must not change:

- selected W keys;
- selected S context;
- support;
- interval scores or posterior;
- the other type or interval group;
- P2's caller-owned contract;
- normal training or deployment when disabled.

This is a measurement counterfactual, not a proposed mask. It intentionally
measures the complete value contribution carried by the selected interval
group, including its share of the repeated common view. It does not claim that
common is physically owned by that interval.

The seam is plain evaluation state on `ObjectFutureEffectReader`, never a
persistent buffer. Training rejects a non-neutral mode. Validation sets one
mode only around a complete five-update-plus-endpoint sample and clears it in
`finally`, so endpoint event metrics correspond to the same counterfactual.

## 5. Authorized gripper diagnostics

The primary full-validation pass adds no model forward and records:

1. absolute-branch, cumulative-delta-branch and branch-disagreement RMSE by
   action band, in normalized and physical gripper units;
2. a complete per-band partition of gripper error into target-event context:
   `before_any_event`, `event`, `post_1_2`, `post_3_6`, `post_7_plus`, with
   row counts.

The five event-context categories are mutually exclusive and exhaustive
inside each band. Their squared-error sums and counts must reconstruct the
existing band gripper RMSE. A later target event resets the post-event clock.

On the bounded matched subset, each P2 mode reports:

- counterfactual gripper RMSE and MSE gain versus its paired primary in all
  three action bands;
- arm and gripper action-delta RMSE in all three bands;
- counterfactual post-event RMSE and paired MSE gain at distances 1-2, 3-6 and
  7+;
- decoded event precision, recall, F1, event ratio and timing MAE.

Positive MSE gain means the counterfactual reduced error. Action-delta
magnitude alone is never called beneficial.

## 6. Forward and reverse acceptance

Forward checks:

- disabled seam gives bit-identical P2 output and state keys;
- each mode changes only the named value cells and preserves posterior metrics;
- primary validation action is independent of whether A01 accounting is
  enabled;
- every counterfactual uses the exact primary initial noise and static cache;
- branch blend reconstructs the existing decoded gripper;
- event-context cells reconstruct every band's square error and row count;
- all new scalars are finite and JSONL-only except a bounded coverage/key-gain
  console summary if retained.

Reverse checks:

- no intervention enters training, loss construction or backward;
- no key/posterior/support gradient path is redefined;
- no parameter or persistent buffer is added;
- optimizer groups and checkpoint model keys are unchanged;
- clearing the seam restores the exact primary function even after an
  exception.

## 7. Unresolved assumptions after source review

The validation result, not this worksheet, must close these assumptions:

1. Is geometry's small value action-irrelevant, or does removing it worsen far
   gripper despite its weak direct supervision?
2. Do later semantic interval values help far gripper more than near values,
   even though the mid action band currently prefers near intervals?
3. Is the far gripper rise caused by event-row composition, early persistence,
   long persistence, or no-prior-event rows?
4. Does the absolute branch fail first, the cumulative-delta branch fail
   first, or do both fail before their fixed blend?
5. Does any P2 removal improve gripper only by damaging arm or by suppressing
   decoded events?

No loss rescaling, time prior, hard interval schedule, gripper gate, codec
change, sampler change, checkpoint selector or P3 edit is authorized until
these measurements are returned and reviewed.

## 8. Authorized edit and verification boundary

Production edits are limited to:

- `clearvla/mainline/model/compiler.py`: evaluation-only value seam;
- `clearvla/mainline/runtime/evaluation.py`: branch and joint-slice accounting;
- `clearvla/mainline/train.py`: bounded matched replay and metric reduction.
- `clearvla/mainline/runtime/logging.py`: bounded branch/coverage/key-gain
  console projection; the lossless JSONL remains authoritative.

Focused tests may change in the existing test files. This worksheet and the
R2 plan may be updated with implementation closure. The compact current
architecture contract receives only a source-state note; no behavioral
architecture invariant changes.

Required verification:

- focused P2 locality/identity tests;
- gripper branch blend and joint-partition reconstruction tests;
- matched-noise validation orchestration test or source-backed invariant;
- complete retained mainline suite;
- compileall, Ruff, Pyright and `git diff --check`;
- state-key and parameter-count identity with the pre-slice graph.

## 9. Implementation and static closure

The implemented source follows the authorized forward boundary exactly:

1. `ObjectFutureEffectReader` owns four plain evaluation modes. The neutral
   mode returns the original selected common/residual tensors directly. A
   non-neutral mode zeros only the named `[I,Z]` value cells after the
   interval posterior is complete and before its two value contractions.
2. `_validate` reuses the primary encoded cache and
   `prediction.initial_physical_noise` for every complete five-update plus
   endpoint counterfactual. The mode is cleared in `finally`; the
   counterfactual endpoint event logits and action therefore belong to the
   same intervention.
3. Full primary validation decodes the absolute and cumulative-delta gripper
   branches from the already returned physical field. It also partitions each
   action band into `before_any_event`, `event`, `post_1_2`, `post_3_6` and
   `post_7_plus` without an additional model forward.
4. The paired accumulator reports gripper MSE gain, arm/gripper action delta,
   post-event gain and decoded-event health for each of the four modes. Only a
   bounded decision-facing subset reaches the console; all scalars remain in
   JSONL.

The consumer-to-producer reverse review also closes: the new metrics terminate
in detached validation reductions; no new branch reaches a training loss or
backward path. The intervention does not alter keys, selected S context,
support, interval scores, posterior, the other value type, P2's caller-owned
contract or any alternate P1/P3/transition/bottom consumer. The primary sampler
does not draw an additional random tensor.

Static graph identity remains the R1h/LC-01 inventory:

```text
total parameters                    168,412,574
trainable parameters                152,041,843
parameter tensors                   1,386
trainable/optimizer tensors         1,064
optimizer groups                    23
state-key names                     1,392
ordered state-key sentinel          be7b4b58a8e2ec25c1e3b5c455f303a0954d20a984201173b5de12d2b1f14a20
post-construction CPU RNG sentinel  d3bcc995a57b40e359a6370a4dc3eea1638fa4a210f3082e41f6791a75513c21
```

`_eval_value_intervention` is absent from `state_dict()` and defaults to
`none`. No parameter, persistent buffer, optimizer owner or checkpoint field
was added.

Verification on the final source:

- focused runtime/structural tests: `94 passed`;
- complete retained mainline matrix: `164 passed`;
- `compileall`, Ruff check and `git diff --check`: pass;
- Pyright: `0 errors` (environment/baseline warnings remain non-blocking);
- tests prove neutral bit identity, localized value removal, posterior-metric
  invariance, exact noise reuse/finally cleanup, branch-decode identity and
  exhaustive joint-slice reconstruction.

Final pre-commit Git blobs:

| File | Git blob |
|---|---|
| `clearvla/mainline/model/compiler.py` | `53d38fa057de13d82c50782b7b23400aaad1270b` |
| `clearvla/mainline/runtime/evaluation.py` | `e091d7bb7366a68dc241971c73ad19cbfb20438c` |
| `clearvla/mainline/runtime/logging.py` | `c0573edf99cc7b23854dc3e338a0d7ea0a7b5a47` |
| `clearvla/mainline/train.py` | `14e2bfdb96c412a3732f352774364f7a42d69825` |
| `tests/test_mainline_runtime.py` | `f86fd9546231346ccd544c56a760c6eb606b3bb5` |
| `tests/test_mainline_structural_contracts.py` | `34f03697f481691cae7a0f37608fa7489a54db47` |

## 10. Existing-checkpoint validation closure

The read-only replay completed on 2026-08-27 from commit
`3219431a49f82b3984fdc4f31c77e76fdccd2b23`, using checkpoint
`runs/schema25_r1_b8_20260826_220907/checkpoints/latest.pt`.  The primary pass
covered all `179` validation batches / `1,432` samples.  Each matched P2
counterfactual covered the configured `16` batches and reused the primary
encoded cache and initial physical noise.

Decision-facing results:

```text
full gripper RMSE near / mid / far                 0.059981 / 0.125307 / 0.195153
absolute branch near / mid / far                   0.061225 / 0.125126 / 0.192019
cumulative-delta branch near / mid / far           0.062570 / 0.137393 / 0.224843
decoded event recall / event ratio                  0.296242 / 0.504053
P2 semantic / geometry band-pair TV                 0.211087 / 0.017577
P2 semantic / geometry effect RMS                   0.112451 / 0.014242

matched-subset primary far-gripper RMSE             0.140827
semantic-far-zero far-gripper RMSE                  0.271796
semantic-far-zero far-gripper action delta          0.165580
geometry-far-zero far-gripper RMSE                  0.140556
geometry-far-zero far-gripper action delta          0.002619
```

The five assumptions from Section 7 resolve as follows:

1. Geometry is nearly action-inert at the learned R1 magnitude on this matched
   subset.  Removing its far values changes predicted far gripper by only
   `0.002619` action-delta RMSE, while task RMSE moves from `0.140827` to
   `0.140556`.  This does not authorize deleting geometry; the producer is
   under-supervised and its W2 value is only about one third of Teacher
   transport magnitude.
2. Later semantic values are indispensable to far action.  Removing intervals
   `2/3` raises matched far-gripper RMSE from `0.140827` to `0.271796`, reduces
   decoded event F1 from about `0.459` to `0.233`, and also changes far arm.
   Therefore the proposed hard conclusion that P2 simply fails to use far
   semantic evidence is rejected.
3. The far gripper rise is not a monotonically accumulating post-event drift.
   Post-event RMSE falls from `0.376999` at distance `1-2` to `0.180002` at
   distance `7+`; however the same event/post-event categories become much
   harder in later absolute action bands.  Event-related rows account for the
   large majority of gripper squared error in every band.
4. Both deployed branches fail before fusion.  The cumulative-delta branch is
   worse in the middle and far bands, while the absolute branch is already
   poor.  The deployed blend is only modestly worse than the absolute branch,
   so changing `0.75/0.25` is not the primary repair.
5. No P2 value removal produces a useful gripper win that is hidden by arm or
   event behavior.  Semantic removal is strongly harmful; geometry removal is
   nearly inert at current scale.

This closes A01.  It rejects a hard far-interval schedule and a first codec
blend edit.  It leaves three distinct source problems for the parent R2 plan:
weak W geometry supervision/output use, overlapping spatial-versus-temporal
P2 query ownership, and missing gripper-private closure between supervised
event semantics and deployed continuous value/delta heads.
